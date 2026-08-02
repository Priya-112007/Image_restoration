import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

def list_paired_files(gt_dir, deg_dir):
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.npy")))
    deg_files = sorted(glob.glob(os.path.join(deg_dir, "*.npy")))

    if len(gt_files) != len(deg_files):
        raise ValueError(
            f"Mismatched pair counts: {len(gt_files)} GT files vs "
            f"{len(deg_files)} degraded files. Check your data folders."
        )

    for g, d in zip(gt_files, deg_files):
        if os.path.basename(g) != os.path.basename(d):
            raise ValueError(
                f"Filename mismatch: '{os.path.basename(g)}' vs "
                f"'{os.path.basename(d)}'. Files must have identical names "
                f"in both folders."
            )

    return list(zip(gt_files, deg_files))


def split_pairs(pairs, val_fraction=0.1, seed=42):
    pairs = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    split_idx = int(len(pairs) * (1 - val_fraction))
    return pairs[:split_idx], pairs[split_idx:]

def random_crop_pair(deg, gt, deg_patch=64):
    scale = gt.shape[-1] // deg.shape[-1]
    h, w = deg.shape[-2:]

    patch = min(deg_patch, h, w)

    top = random.randint(0, h - patch)
    left = random.randint(0, w - patch)

    deg_crop = deg[..., top: top + patch, left: left + patch]
    gt_crop = gt[
        ...,
        top * scale: (top + patch) * scale,
        left * scale: (left + patch) * scale,
    ]
    return deg_crop, gt_crop


def add_extra_speckle(deg, sigma_range=(0.01, 0.05), p=0.3):
    if random.random() < p:
        sigma = random.uniform(*sigma_range)
        noise = np.random.normal(0, sigma, deg.shape).astype(np.float32)
        deg = deg * (1 + noise)
    return deg

class RestorationDataset(Dataset):
    def __init__(self, pairs, patch_size=64, train=True):
        self.pairs = pairs
        self.patch_size = patch_size
        self.train = train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        gt_path, deg_path = self.pairs[idx]
        gt = np.load(gt_path).astype(np.float32)
        deg = np.load(deg_path).astype(np.float32)

        if gt.ndim == 3:
            gt = gt[..., 0]
        if deg.ndim == 3:
            deg = deg[..., 0]

        if self.train:
            deg, gt = random_crop_pair(deg, gt, self.patch_size)

            if random.random() < 0.5:
                deg, gt = np.fliplr(deg).copy(), np.fliplr(gt).copy()
            if random.random() < 0.5:
                deg, gt = np.flipud(deg).copy(), np.flipud(gt).copy()
            if random.random() < 0.5:
                k = random.choice([1, 2, 3])
                deg, gt = np.rot90(deg, k).copy(), np.rot90(gt, k).copy()

            deg = add_extra_speckle(deg)

        deg_t = torch.from_numpy(np.ascontiguousarray(deg)).unsqueeze(0)
        gt_t = torch.from_numpy(np.ascontiguousarray(gt)).unsqueeze(0)
        return deg_t, gt_t