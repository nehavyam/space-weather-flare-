# basic image stuff on a 2d list/array

import numpy as np


def blur(img):
    # 3x3 average filter
    h, w = img.shape
    out = np.zeros((h, w), dtype=np.float32)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            s = 0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    s += img[i + di, j + dj]
            out[i, j] = s / 9
    return out


def process(img):
    """resize-ish normalize + blur. returns 1x32x32 for the cnn"""
    img = np.array(img, dtype=np.float32)
    m = np.max(np.abs(img))
    if m > 0:
        img = img / m
    img = blur(img)
    return img.reshape(1, img.shape[0], img.shape[1])
