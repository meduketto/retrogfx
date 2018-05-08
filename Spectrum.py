#!/usr/bin/env python

import numpy as np

screen_width = 256

screen_height = 192

attr_width = 32

attr_height = 24

colors = ( (  0,   0,   0), (  0,   0,   0),
           (  0,   0, 215), (  0,   0, 255),
           (215,   0,   0), (255,   0,   0),
           (215,   0, 215), (255,   0, 255),
           (  0, 215,   0), (  0, 255,   0),
           (  0, 215, 215), (  0, 255, 255),
           (215, 215,   0), (255, 255,   0),
           (215, 215, 215), (255, 255, 255) )

np_colors = np.array(colors)

def get_attr(data, x, y):
    attrpos = 6144 + (x // 8) + (y // 8) * 32
    attr = data[attrpos]
    bright = 1 if (attr & 0x40) else 0
    ink = attr & 0x07
    paper = (attr & 0x38) >> 3
    return bright, ink, paper

def get_colors(data, x, y):
    bright, ink, paper = get_attr(data, x, y)
    return np_colors[ink*2 + bright], np_colors[paper*2 + bright]

def scr_to_image(data):
    image = np.zeros((screen_height, screen_width, 3))
    for x in range(0, screen_width):
        for y in range(0, screen_height):
            ink, paper = get_colors(data, x, y)
            third = y // 64
            ty = y % 64
            block = ty % 8
            by = ty // 8
            pos = third * 2048 + block * 256 + by * 32 + x // 8
            pixel = data[pos] & (1 << (7 - (x % 8)))
            image[y,x] = ink if pixel else paper
    return image
