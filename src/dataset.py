import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from synthetic_degradation import generate_synthetic_pair


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
    def __init__(
        self,
        pairs,
        patch_size=64,
        train=True,
        use_synthetic_degradation=False,
        synthetic_prob=0.3,
        use_cutmix=False,
        cutmix_prob=0.3,
        use_gamma_jitter=False,
    ):
        self.pairs = pairs
        self.patch_size = patch_size
        self.train = train
        self.use_synthetic_degradation = use_synthetic_degradation
        self.synthetic_prob = synthetic_prob
        self.use_cutmix = use_cutmix
        self.cutmix_prob = cutmix_prob
        self.use_gamma_jitter = use_gamma_jitter

    def __len__(self):
        return len(self.pairs)

    def _load_pair(self, idx):
        gt_path, deg_path = self.pairs[idx]
        gt = np.load(gt_path).astype(np.float32)
        deg = np.load(deg_path).astype(np.float32)
        if gt.ndim == 3:
            gt = gt[..., 0]
        if deg.ndim == 3:
            deg = deg[..., 0]
        # Safeguard: clip input dynamic range to [0.0, 1.0]
        deg = np.clip(deg, 0.0, 1.0)
        gt = np.clip(gt, 0.0, 1.0)
        return deg, gt

    def _apply_cutmix(self, deg, gt):
        other_idx = random.randrange(len(self.pairs))
        other_deg, other_gt = self._load_pair(other_idx)
        other_deg_crop, other_gt_crop = random_crop_pair(other_deg, other_gt, self.patch_size)

        scale = gt.shape[-1] // deg.shape[-1]
        ph, pw = deg.shape[-2:]
        frac = random.uniform(0.2, 0.4)
        box_h = max(1, int(ph * frac))
        box_w = max(1, int(pw * frac))
        top = random.randint(0, ph - box_h)
        left = random.randint(0, pw - box_w)

        deg = deg.copy()
        gt = gt.copy()
        deg[top:top + box_h, left:left + box_w] = other_deg_crop[top:top + box_h, left:left + box_w]
        gt[top * scale:(top + box_h) * scale, left * scale:(left + box_w) * scale] = (
            other_gt_crop[top * scale:(top + box_h) * scale, left * scale:(left + box_w) * scale]
        )
        return deg, gt

    def __getitem__(self, idx):
        deg, gt = self._load_pair(idx)

        if self.train:
            use_synth = self.use_synthetic_degradation and random.random() < self.synthetic_prob
            if use_synth:
                scale = gt.shape[-1] // deg.shape[-1]
                gt_patch = self.patch_size * scale
                gt_patch = min(gt_patch, gt.shape[-2], gt.shape[-1])
                gt_patch -= gt_patch % scale  # keep evenly divisible for block downsample
                top = random.randint(0, gt.shape[-2] - gt_patch)
                left = random.randint(0, gt.shape[-1] - gt_patch)
                gt_crop = gt[top:top + gt_patch, left:left + gt_patch]
                deg, gt = generate_synthetic_pair(gt_crop, downsample_factor=scale)
            else:
                deg, gt = random_crop_pair(deg, gt, self.patch_size)

            if random.random() < 0.5:
                deg, gt = np.fliplr(deg).copy(), np.fliplr(gt).copy()
            if random.random() < 0.5:
                deg, gt = np.flipud(deg).copy(), np.flipud(gt).copy()
            if random.random() < 0.5:
                k = random.choice([1, 2, 3])
                deg, gt = np.rot90(deg, k).copy(), np.rot90(gt, k).copy()

            if self.use_cutmix and random.random() < self.cutmix_prob:
                deg, gt = self._apply_cutmix(deg, gt)

            if self.use_gamma_jitter and random.random() < 0.5:
                gamma = random.uniform(0.85, 1.15)
                gt = np.power(np.clip(gt, 0, 1), gamma).astype(np.float32)
                deg = np.power(np.clip(deg, 0, 1), gamma).astype(np.float32)

        deg = np.clip(deg, 0.0, 1.0)
        gt = np.clip(gt, 0.0, 1.0)
        deg_t = torch.from_numpy(np.ascontiguousarray(deg)).unsqueeze(0)
        gt_t = torch.from_numpy(np.ascontiguousarray(gt)).unsqueeze(0)
        return deg_t, gt_t