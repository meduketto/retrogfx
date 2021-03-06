#!/usr/bin/env python

import os
import glob
import random

import cv2
import numpy as np

import Spectrum

def read_rgb(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (256, 192), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32')
    img = (img - 127.5) / 127.5
    return img

def read_scr(path):
    data = open(path, 'rb').read()
    img = Spectrum.scr_to_image(data)
    img = (img - 127.5) / 127.5
    return img

def to_image(nn):
    return (nn * 127.5 + 127.5).astype('uint8')

def batch(name_glob, read_func, batch_size):
    data = np.array([read_func(name) for name in glob.glob(name_glob)])
    assert batch_size < data.shape[0]
    np.random.shuffle(data)
    epoch = 0
    pos = 0
    while True:
        if pos + batch_size >= data.shape[0]:
            pos = 0
            epoch += 1
            np.random.shuffle(data)
        yield epoch, data[pos:pos + batch_size]
        pos += batch_size

def minibatch(batch_size, rgb_glob='image_rgb/*', scr_glob='image_scr/*.scr'):
    rgb_data = batch(rgb_glob, read_rgb, batch_size)
    scr_data = batch(scr_glob, read_scr, batch_size)
    while True:
        epoch1, A = next(rgb_data)
        epoch2, B = next(scr_data)
        yield max(epoch1, epoch2), A, B


class ImagePool:
    def __init__(self, size=200):
        self.size = size
        self.n = 0
        self.images = []

    def replace(self, images):
        new_images = []
        for image in images:
            if self.n < self.size:
                self.n += 1
                self.images.append(image)
                new_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    i = random.randint(0, self.size - 1)
                    tmp = self.images[i]
                    self.images[i] = image
                    new_images.append(tmp)
                else:
                    new_images.append(image)
        return np.stack(new_images, axis=0)
