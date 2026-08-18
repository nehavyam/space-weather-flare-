# fake sun pictures for training (real nasa files were too big)

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from image_processing import process

SIZE = 32


def make_one_image(rng, complex_region=False):
    img = np.zeros((SIZE, SIZE), dtype=np.float32)
    # two round spots + and -
    cy, cx = 16, 16
    for i in range(SIZE):
        for j in range(SIZE):
            img[i, j] += 0.9 * np.exp(-((i - cy) ** 2 + (j - (cx - 6)) ** 2) / 18)
            img[i, j] -= 0.9 * np.exp(-((i - cy) ** 2 + (j - (cx + 6)) ** 2) / 18)
            img[i, j] += rng.normal(0, 0.03)

    if complex_region:
        # extra messy spots (much stronger so the cnn can tell them apart)
        for _k in range(5):
            y = int(rng.integers(6, 26))
            x = int(rng.integers(6, 26))
            sign = 1 if rng.random() > 0.5 else -1
            for i in range(SIZE):
                for j in range(SIZE):
                    img[i, j] += sign * 0.9 * np.exp(-((i - y) ** 2 + (j - x) ** 2) / 6)

    return np.clip(img, -1, 1)


def make_dataset(n=400, seed=42, out_folder=None):
    rng = np.random.default_rng(seed)
    images = np.zeros((n, 1, SIZE, SIZE), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    for i in range(n):
        messy = rng.random() < 0.3
        pic = make_one_image(rng, messy)
        y = 1 if messy else 0
        images[i] = process(pic)
        labels[i] = y
        if i % 100 == 0:
            print("made", i)
    if out_folder:
        os.makedirs(out_folder, exist_ok=True)
        np.savez_compressed(os.path.join(out_folder, "dataset.npz"), images=images, labels=labels)
    return images, labels


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    make_dataset(out_folder=os.path.join(here, "..", "data"))
    print("saved data/dataset.npz")
