#!/usr/bin/env python

import os
import glob

import cv2
import numpy as np

import Spectrum

def read_rgb(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (256, 192), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32')
    img /= 255.
    return img

def read_scr(path):
    data = open(path, 'rb').read()
    return data


class Dataset:
    def __init__(self, path='pairs/'):
        self.path = path
        indexes = self.get_indexes()
        n = len(indexes)
        n_test = min(5, n % 50)
        self.train = self.read_data(indexes[:-n_test])
        self.test = self.read_data(indexes[-n_test:])

    def read_data(self, indexes):
        print(indexes)
        rgb = np.array([self.read_rgb(i) for i in indexes])
        scr = [self.read_scr(i) for i in indexes]
        bitmap = np.array([Spectrum.scr_to_bitmap(data) for data in scr])
        attr = np.array([Spectrum.scr_to_attr(data) for data in scr])
        return rgb, bitmap, attr

    def read_rgb(self, i):
        return read_rgb(os.path.join(self.path, str(i)))

    def read_scr(self, i):
        return read_scr(os.path.join(self.path, '{}.scr'.format(i)))

    def get_indexes(self):
        files = glob.glob(os.path.join(self.path, '*.scr'))
        indexes = [int(x.rsplit('/', 1)[1].split('.')[0]) for x in files]
        return sorted(indexes)


if __name__ == "__main__":
    d = Dataset()
    print(d.test)
