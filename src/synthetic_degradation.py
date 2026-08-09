import random

import numpy as np
from scipy.ndimage import gaussian_filter


def apply_gaussian_blur(img, sigma):
    if sigma <= 0:
        return img
    return gaussian_filter(img, sigma=sigma).astype(np.float32)


def apply_speckle_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return (img * (1 + noise)).astype(np.float32)


def apply_gaussian_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return (img + noise).astype(np.float32)


def apply_block_downsample(img, factor=2):
    h, w = img.shape[-2:]
    h_crop = h - (h % factor)
    w_crop = w - (w % factor)
    img = img[:h_crop, :w_crop]
    return (
        img.reshape(h_crop // factor, factor, w_crop // factor, factor)
        .mean(axis=(1, 3))
        .astype(np.float32)
    )


def generate_synthetic_pair(
    gt_img,
    downsample_factor=2,
    blur_sigma_range=(0.5, 2.5),
    speckle_sigma_range=(0.01, 0.08),
    gaussian_sigma_range=(0.01, 0.05),
):

    ops = ["blur", "speckle", "gaussian"]
    random.shuffle(ops)

    img = gt_img.copy()
    for op in ops:
        if op == "blur":
            sigma = random.uniform(*blur_sigma_range)
            img = apply_gaussian_blur(img, sigma)
        elif op == "speckle":
            sigma = random.uniform(*speckle_sigma_range)
            img = apply_speckle_noise(img, sigma)
        elif op == "gaussian":
            sigma = random.uniform(*gaussian_sigma_range)
            img = apply_gaussian_noise(img, sigma)

    img = apply_block_downsample(img, factor=downsample_factor)

    return img.astype(np.float32), gt_img.astype(np.float32)