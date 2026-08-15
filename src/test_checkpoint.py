import argparse
import csv
import os
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from dataset import RestorationDataset, list_paired_files, split_pairs
from losses import ssim as ssim_fn, psnr as psnr_fn, lpips_loss
from model import build_model


def load_model(stage, checkpoint_path, device):
    model = build_model(stage).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    # prefer EMA weights if present — that's what evaluate.py uses too
    state_dict = ckpt["ema"] if "ema" in ckpt else ckpt["model"]
    model.load_state_dict(state_dict)
    model.eval()
    return model, ckpt


def save_comparison(deg, pred, gt, save_path, title=""):
    deg_np = deg.squeeze().cpu().numpy()
    pred_np = pred.squeeze().cpu().numpy()
    gt_np = gt.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, name in zip(axes, [deg_np, pred_np, gt_np],
                              ["NoisyLR", "Prediction", "Ground Truth"]):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(name)
        ax.axis("off")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--deg_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage", required=True,
                         choices=["stage0_baseline", "stage1_film", "stage2_hybrid", "stage3_nafnet_unet"])
    parser.add_argument("--output_dir", default="./eval_results")
    parser.add_argument("--save_visuals", type=int, default=5,
                         help="Number of side-by-side comparison PNGs to save (0 to disable).")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42,
                         help="MUST match the seed used in train.py's split_pairs (default "
                              "42, unchanged there), or this evaluates a DIFFERENT random "
                              "subset than the model was actually validated on during "
                              "training, making the comparison meaningless.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    pairs = list_paired_files(args.gt_dir, args.deg_dir)
    _, val_pairs = split_pairs(pairs, val_fraction=args.val_fraction, seed=args.seed)
    print(f"Validation set: {len(val_pairs)} pairs (same split used during training)")

    val_ds = RestorationDataset(val_pairs, train=False)

    model, ckpt = load_model(args.stage, args.checkpoint, device)
    ckpt_ssim = ckpt.get("val_ssim", ckpt.get("best_val_ssim", "?"))
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(training-time recorded SSIM: {ckpt_ssim})")

    results = []

    with torch.no_grad():
        for idx in range(len(val_ds)):
            deg, gt = val_ds[idx]
            gt_path, deg_path = val_pairs[idx]
            filename = os.path.basename(gt_path)

            deg_b = deg.unsqueeze(0).to(device)
            gt_b = gt.unsqueeze(0).to(device)

            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            pred = model(deg_b).clamp(0, 1)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.time() - t0) * 1000

            s = ssim_fn(pred, gt_b).item()
            p = psnr_fn(pred, gt_b).item()
            l = lpips_loss(pred, gt_b).item()

            results.append({"file": filename, "ssim": s, "psnr": p,
                             "lpips": l, "inference_ms": elapsed_ms})

            if args.save_visuals and idx < args.save_visuals:
                base_name = os.path.splitext(filename)[0]
                save_path = os.path.join(args.output_dir, f"compare_{base_name}.png")
                save_comparison(
                    deg_b, pred, gt_b, save_path,
                    title=f"{filename}  SSIM={s:.3f}  PSNR={p:.1f}  LPIPS={l:.3f}",
                )

            print(f"[{idx + 1}/{len(val_ds)}] {filename}  "
                  f"SSIM={s:.4f}  PSNR={p:.2f}  LPIPS={l:.4f}")

    avg_ssim = float(np.mean([r["ssim"] for r in results]))
    avg_psnr = float(np.mean([r["psnr"] for r in results]))
    avg_lpips = float(np.mean([r["lpips"] for r in results]))
    avg_ms = float(np.mean([r["inference_ms"] for r in results]))

    print("\n" + "=" * 55)
    print(f"Stage:               {args.stage}")
    print(f"Checkpoint:          {args.checkpoint}")
    print(f"Validation pairs:    {len(results)}")
    print(f"Avg SSIM:            {avg_ssim:.4f}")
    print(f"Avg PSNR:            {avg_psnr:.2f} dB")
    print(f"Avg LPIPS:           {avg_lpips:.4f}")
    print(f"Avg inference time:  {avg_ms:.2f} ms/image  (this run's device: {device})")
    print("=" * 55)

    worst = sorted(results, key=lambda r: r["ssim"])[:3]
    print("\nWorst 3 by SSIM (candidates for an honest failure-case slide):")
    for r in worst:
        print(f"  {r['file']}: SSIM={r['ssim']:.4f}  PSNR={r['psnr']:.2f}  LPIPS={r['lpips']:.4f}")

    csv_path = os.path.join(args.output_dir, f"{args.stage}_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "ssim", "psnr", "lpips", "inference_ms"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nPer-image results saved to: {csv_path}")

    summary_path = os.path.join(args.output_dir, f"{args.stage}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Stage: {args.stage}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Validation pairs: {len(results)}\n")
        f.write(f"Avg SSIM: {avg_ssim:.4f}\n")
        f.write(f"Avg PSNR: {avg_psnr:.2f}\n")
        f.write(f"Avg LPIPS: {avg_lpips:.4f}\n")
        f.write(f"Avg inference: {avg_ms:.2f} ms/image\n")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()