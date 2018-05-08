#!/usr/bin/env python

import os
import time

import numpy as np
import keras
from keras import backend as K

import Dataset

def build_input():
    x = keras.layers.Input(shape=(192, 256, 3))
    return x

def build_conv(x, nf, size, stride=1):
    x = keras.layers.Conv2D(nf,
                            (size, size),
                            strides=(stride, stride),
                            kernel_initializer=keras.initializers.RandomNormal(0, 0.02),
                            padding='same')(x)
    return x

def build_resnet(x):
    y = build_conv(x, 256, 3, 1)
    y = build_conv(y, 256, 3, 1)
    return keras.layers.Add()([y, x])

def build_deconv(x, nf, size, stride=2):
    x = keras.layers.Conv2DTranspose(nf, size, strides=(stride, stride), padding='same')(x)
    return x

def build_generator():
    inputs = x = build_input()

    x = build_conv(x, 64, 7)
    x = build_conv(x, 128, 3, 2)
    x = build_conv(x, 256, 3, 2)

    for i in range(6):
        x = build_resnet(x)

    x = build_deconv(x, 128, 3)
    x = build_deconv(x, 64, 3)
    outputs = x = build_conv(x, 3, 7)

    return keras.models.Model(inputs=inputs, outputs=outputs)

def build_discriminator():
    inputs = x = build_input()

    x = build_conv(x, 64, 4, 2)
    x = build_conv(x, 128, 4, 2)
    x = build_conv(x, 256, 4, 2)
    outputs = x = build_conv(x, 1, 4, 2)

    return keras.models.Model(inputs=inputs, outputs=outputs)

def gan_loss(output, target):
    diff = output - target
    dims = list(range(1, K.ndim(diff)))
    return K.expand_dims((K.mean(diff ** 2, dims)), 0)

def cycle_loss(cycle, real):
    diff = K.abs(cycle - real)
    dims = list(range(1, K.ndim(diff)))
    return K.expand_dims((K.mean(diff ** 2, dims)), 0)

def gen_loss(inputs):
    disc_fake_B, cycle_A, orig_A, disc_fake_A, cycle_B, orig_B = inputs
    gen_A_loss = gan_loss(disc_fake_B, K.ones_like(disc_fake_B))
    gen_B_loss = gan_loss(disc_fake_A, K.ones_like(disc_fake_A))
    cycle_A_loss = cycle_loss(cycle_A, orig_A)
    cycle_B_loss = cycle_loss(cycle_B, orig_B)
    loss = gen_A_loss + gen_B_loss + 10 * (cycle_A_loss + cycle_B_loss)
    return loss

def disc_loss(inputs):
    disc_true, disc_false = inputs
    true_loss = gan_loss(disc_true, K.ones_like(disc_true))
    false_loss = gan_loss(disc_false, K.zeros_like(disc_false))
    loss = 0.5 * (true_loss + false_loss)
    return loss


class Trainer:
    def __init__(self, cycle_gan, batch_size=4):
        self.mb = Dataset.minibatch(batch_size)
        self.cycle_gan = cycle_gan
        self.batch_size = batch_size
        self.fake_A_pool = Dataset.ImagePool()
        self.fake_B_pool = Dataset.ImagePool()
        self.target = np.zeros((batch_size, 1))
        self.epoch = 0
        self.preprocessed = False

    def train_one_batch(self):
        if not self.preprocessed:
            print("Preprocessing training data...")
            start = time.time()
        epoch, A, B = next(self.mb)
        if not self.preprocessed:
            print("Preprocessing done. Took", round(time.time() - start, 1), "seconds.")
            self.preprocessed = True

        tmp_fake_A = K.function([self.cycle_gan.net_B2A_gen.inputs[0],
                                 K.learning_phase()],
                                [self.cycle_gan.net_B2A_gen.outputs[0]])([A,1])[0]
        tmp_fake_B = K.function([self.cycle_gan.net_A2B_gen.inputs[0],
                                 K.learning_phase()],
                                [self.cycle_gan.net_A2B_gen.outputs[0]])([B,1])[0]
        fake_b = self.fake_B_pool.replace(tmp_fake_B)
        fake_a = self.fake_A_pool.replace(tmp_fake_A)
        self.cycle_gan.train_gen.train_on_batch([A, B], self.target)
        self.cycle_gan.train_disc_A.train_on_batch([A, fake_a], self.target)
        self.cycle_gan.train_disc_B.train_on_batch([B, fake_b], self.target)

        return epoch

    def train_one_epoch(self):
        print("Training epoch", self.epoch)
        start = time.time()
        epoch = self.epoch
        while epoch == self.epoch:
            epoch = self.train_one_batch()
        self.epoch = epoch
        print("Done. Took", round(time.time() - start, 1), "seconds.")


class CycleGAN:
    def setTrainable(self, gen, discA, discB):
        for layer in self.net_A2B_gen.layers:
            layer.trainable = gen
        for layer in self.net_B2A_gen.layers:
            layer.trainable = gen
        for layer in self.net_A_disc.layers:
            layer.trainable = discA
        for layer in self.net_B_disc.layers:
            layer.trainable = discB

    def __init__(self):
        # Generator and discriminator networks
        self.net_A2B_gen = build_generator()
        self.net_B2A_gen = build_generator()
        self.net_A_disc = build_discriminator()
        self.net_B_disc = build_discriminator()

        orig_A = self.net_A2B_gen.inputs[0]
        orig_B = self.net_B2A_gen.inputs[0]
        fake_B = self.net_A2B_gen.outputs[0]
        fake_A = self.net_B2A_gen.outputs[0]

        # Train function for generators
        disc_fake_B = self.net_B_disc(fake_B)
        cycle_A = self.net_B2A_gen(fake_B)
        disc_fake_A = self.net_A_disc(fake_A)
        cycle_B = self.net_A2B_gen(fake_A)

        self.setTrainable(True, False, False)

        inputs = [orig_A, orig_B]
        outputs = [disc_fake_B, cycle_A, orig_A, disc_fake_A, cycle_B, orig_B]
        outputs = keras.layers.Lambda(gen_loss)(outputs)
        self.train_gen = keras.models.Model(inputs, outputs)
        adam_opt = keras.optimizers.Adam(lr=2e-4, beta_1=0.5, beta_2=0.999,
                                         epsilon=None, decay=0.0)
        self.train_gen.compile(adam_opt, 'mae')

        # Train function for discriminator A
        disc_A = self.net_A_disc(orig_A)
        false_A = build_input()
        disc_false_A = self.net_A_disc(false_A)
        self.setTrainable(False, True, False)
        inputs = [orig_A, false_A]
        outputs = keras.layers.Lambda(disc_loss)([disc_A, disc_false_A])
        self.train_disc_A = keras.models.Model(inputs, outputs)
        self.train_disc_A.compile(adam_opt, 'mae')

        # Train function for discriminator b
        disc_B = self.net_B_disc(orig_B)
        false_B = build_input()
        disc_false_B = self.net_B_disc(false_B)
        self.setTrainable(False, False, True)
        inputs = [orig_B, false_B]
        outputs = keras.layers.Lambda(disc_loss)([disc_B, disc_false_B])
        self.train_disc_B = keras.models.Model(inputs, outputs)
        self.train_disc_B.compile(adam_opt, 'mae')

    def to_spectrum(self, img):
        conv = False
        if len(img.shape) == 3:
            conv = True
            img = np.array([img])
        scr_img = self.net_A2B_gen.predict(img)
        if conv:
            scr_img = scr_img[0]
        return scr_img

    def to_rgb(self, img):
        conv = False
        if len(img.shape) == 3:
            conv = True
            img = np.array([img])
        rgb_img = self.net_B2A_gen.predict(img)
        if conv:
            rgb_img = rgb_img[0]
        return rgb_img

    def save(self, version):
        os.makedirs('models/', exist_ok=True)
        self.net_A2B_gen.save ('models/model_A2B_gen-{}.h5'.format(version))
        self.net_B2A_gen.save ('models/model_B2A_gen-{}.h5'.format(version))
        self.net_A_disc.save  ('models/model_A_disc-{}.h5'.format(version))
        self.net_B_disc.save  ('models/model_B_disc-{}.h5'.format(version))
        self.train_gen.save   ('models/model_train_gen-{}.h5'.format(version))
        self.train_disc_A.save('models/model_train_disc_A-{}.h5'.format(version))
        self.train_disc_B.save('models/model_train_disc_B-{}.h5'.format(version))

    def load(self, version):
        self.net_A2B_gen      = keras.models.load_model('models/model_A2B_gen-{}.h5'.format(version))
        self.net_B2A_gen      = keras.models.load_model('models/model_B2A_gen-{}.h5'.format(version))
        self.net_A_disc       = keras.models.load_model('models/model_A_disc-{}.h5'.format(version))
        self.net_B_disc       = keras.models.load_model('models/model_B_disc-{}.h5'.format(version))
        self.net_train_gen    = keras.models.load_model('models/model_train_gen-{}.h5'.format(version))
        self.net_train_disc_A = keras.models.load_model('models/model_train_disc_A-{}.h5'.format(version))
        self.net_train_disc_B = keras.models.load_model('models/model_train_disc_B-{}.h5'.format(version))
