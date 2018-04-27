#!/usr/bin/env python

import numpy as np

screenWidth = 256

screenHeight = 192

colors = ( (  0,   0,   0), (  0,   0,   0),
           (  0,   0, 215), (  0,   0, 255),
           (215,   0,   0), (255,   0,   0),
           (215,   0, 215), (255,   0, 255),
           (  0, 215,   0), (  0, 255,   0),
           (  0, 215, 215), (  0, 255, 255),
           (215, 215,   0), (255, 255,   0),
           (215, 215, 215), (255, 255, 255) )

np_colors = np.divide(np.array(colors), 255.)

def getColors(data, x, y):
    attrpos = 6144 + (x // 8) + (y // 8) * 32
    attr = data[attrpos]
    bright = 1 if (attr & 0x40) else 0
    ink = ((attr & 0x07) * 2) + bright
    paper = (((attr & 0x38) >> 3) * 2) + bright
    return np_colors[ink], np_colors[paper]

def nativeToNumpy(data):
    image = np.zeros((screenHeight, screenWidth, 3))
    for x in range(0, screenWidth):
        for y in range(0, screenHeight):
            ink, paper = getColors(data, x, y)
            third = y // 64
            ty = y % 64
            block = ty % 8
            by = ty // 8
            pos = third * 2048 + block * 256 + by * 32 + x // 8
            pixel = data[pos] & (1 << (7 - (x % 8)))
            image[y,x] = ink if pixel else paper
    return image


#data = open(filename, "rb").read()
