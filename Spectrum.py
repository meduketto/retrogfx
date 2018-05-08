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

def scr_to_bitmap(data):
    bitmap = np.zeros((screen_height, screen_width, 1))
    for x in range(0, screen_width):
        for y in range(0, screen_height):
            third = y // 64
            ty = y % 64
            block = ty % 8
            by = ty // 8
            pos = third * 2048 + block * 256 + by * 32 + x // 8
            pixel = data[pos] & (1 << (7 - (x % 8)))
            if pixel:
                bitmap[y, x] = 1.0
    return bitmap

def scr_to_attr(data):
    attr = np.zeros((attr_height, attr_width, 17))
    for x in range(0, attr_width):
        for y in range(0, attr_height):
            bright, ink, paper = get_attr(data, x * 8, y * 8)
            attr[y, x, 0] = bright * 1.0
            attr[y, x, 1 + ink] = 1.0
            attr[y, x, 1 + 8 + paper] = 1.0
    return attr

def nn_to_scr(bitmap, attr):
    data = bytearray(6912)
    for x in range(0, screen_width):
        for y in range(0, screen_height):
            if bitmap[y, x] < 0.5:
                continue
            third = y // 64
            ty = y % 64
            block = ty % 8
            by = ty // 8
            pos = third * 2048 + block * 256 + by * 32 + x // 8
            bit = 7 - (x % 8)
            data[pos] |= 1 << bit
    for x in range(0, attr_width):
        for y in range(0, attr_height):
            b = 0
            if attr[y, x, 0] >= 0.5:
                b |= 0x40
            ink = np.argmax(attr[y,x, 1:9])
            paper = np.argmax(attr[y, x, 9:])
            b |= ink
            b |= paper << 3
            data[6144 + x + y * 32] = b
    return data
