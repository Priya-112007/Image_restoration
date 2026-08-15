import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import RestorationDataset, list_paired_files, split_pairs
from losses import ssim as ssim_fn, psnr as psnr_fn
from model import build_model
from synthetic_degradation import (
    apply_gaussian_blur, apply_speckle_noise,
    apply_gaussian_noise, apply_block_downsample,
)


def run_degradation_diagnostics(gt_dir="data/train/GT", deg_dir="data/train/NoisyLR",
                                 checkpoint_path="weights.pt", stage="stage3_nafnet_unet"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Degradation Diagnostics on device: {device}")

    pairs = list_paired_files(gt_dir, deg_dir)
    _, val_pairs = split_pairs(pairs, val_fraction=0.1, seed=42)

    model = build_model(stage).to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["ema"] if "ema" in ckpt else ckpt["model"]
        model.load_state_dict(state_dict, strict=False)
    model.eval()

    degradation_types = [
        "1. Speckle Noise Only",
        "2. Gaussian Noise Only",
        "3. Blur Only",
        "4. 2x Downsampling Only",
        "5. Full Real Degradation",
    ]

    results = {d: {"ssim": [], "psnr": []} for d in degradation_types}

    with torch.no_grad():
        for gt_path, deg_path in val_pairs:
            gt_np = np.load(gt_path).astype(np.float32)
            if gt_np.ndim == 2:
                gt_np = np.expand_dims(gt_np, axis=0)

            # Generate isolated degradations
            deg_dict = {}

            # 1. Speckle Noise Only (+ downsample)
            speckle_deg = apply_block_downsample(apply_speckle_noise(gt_np[0], sigma=0.05), factor=2)
            deg_dict["1. Speckle Noise Only"] = np.expand_dims(speckle_deg, axis=0)

            # 2. Gaussian Noise Only (+ downsample)
            gauss_deg = apply_block_downsample(apply_gaussian_noise(gt_np[0], sigma=0.03), factor=2)
            deg_dict["2. Gaussian Noise Only"] = np.expand_dims(gauss_deg, axis=0)

            # 3. Blur Only (+ downsample)
            blur_deg = apply_block_downsample(apply_gaussian_blur(gt_np[0], sigma=1.5), factor=2)
            deg_dict["3. Blur Only"] = np.expand_dims(blur_deg, axis=0)

            # 4. 2x Downsampling Only
            ds_deg = apply_block_downsample(gt_np[0], factor=2)
            deg_dict["4. 2x Downsampling Only"] = np.expand_dims(ds_deg, axis=0)

            # 5. Full Real Dataset Degradation
            real_deg_np = np.load(deg_path).astype(np.float32)
            if real_deg_np.ndim == 2:
                real_deg_np = np.expand_dims(real_deg_np, axis=0)
            deg_dict["5. Full Real Degradation"] = real_deg_np

            # Evaluate each degradation type
            gt_tensor = torch.from_numpy(gt_np).unsqueeze(0).to(device)

            for d_name, deg_arr in deg_dict.items():
                deg_tensor = torch.from_numpy(deg_arr).unsqueeze(0).to(device)
                pred_tensor = model(deg_tensor).clamp(0, 1)

                s = ssim_fn(pred_tensor, gt_tensor).item()
                p = psnr_fn(pred_tensor, gt_tensor).item()

                results[d_name]["ssim"].append(s)
                results[d_name]["psnr"].append(p)

    print("\n" + "=" * 65)
    print("       DEGRADATION BOTTLENECK DIAGNOSTIC RESULTS")
    print("=" * 65)
    print(f"{'Degradation Type':<28} | {'Validation SSIM':<16} | {'Validation PSNR':<15}")
    print("-" * 65)

    for d_name in degradation_types:
        avg_s = np.mean(results[d_name]["ssim"])
        avg_p = np.mean(results[d_name]["psnr"])
        print(f"{d_name:<28} | {avg_s:<16.4f} | {avg_p:<15.2f} dB")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", default="data/train/GT")
    parser.add_argument("--deg_dir", default="data/train/NoisyLR")
    parser.add_argument("--checkpoint", default="weights.pt")
    parser.add_argument("--stage", default="stage3_nafnet_unet")
    args = parser.parse_args()

    run_degradation_diagnostics(args.gt_dir, args.deg_dir, args.checkpoint, args.stage)
