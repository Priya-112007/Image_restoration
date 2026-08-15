import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import RestorationDataset, list_paired_files, split_pairs
from losses import ssim as ssim_fn, psnr as psnr_fn, lpips_loss
from model import build_model


def load_model(stage, checkpoint_path, device, use_film=False):
    model = build_model(stage, use_film=use_film).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["ema"] if "ema" in ckpt else ckpt["model"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def generate_advanced_visuals(gt_dir, deg_dir, checkpoint_path, stage="stage3_nafnet_unet",
                               output_dir="./results", num_samples=5, use_film=False):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = list_paired_files(gt_dir, deg_dir)
    _, val_pairs = split_pairs(pairs, val_fraction=0.1, seed=42)
    val_ds = RestorationDataset(val_pairs, train=False)

    model = load_model(stage, checkpoint_path, device, use_film=use_film)

    indices = list(range(min(num_samples, len(val_ds))))

    with torch.no_grad():
        for idx in indices:
            deg, gt = val_ds[idx]
            filename = os.path.basename(val_pairs[idx][0])
            base_name = os.path.splitext(filename)[0]

            deg_b = deg.unsqueeze(0).to(device)
            gt_b = gt.unsqueeze(0).to(device)

            pred_b = model(deg_b).clamp(0, 1)

            s = ssim_fn(pred_b, gt_b).item()
            p = psnr_fn(pred_b, gt_b).item()
            l = lpips_loss(pred_b, gt_b).item()

            deg_np = deg_b.squeeze().cpu().numpy()
            gt_np = gt_b.squeeze().cpu().numpy()
            pred_np = pred_b.squeeze().cpu().numpy()

            diff_map = np.abs(pred_np - gt_np)

            fig, axes = plt.subplots(1, 5, figsize=(25, 5))

            # 1. Noisy LR
            axes[0].imshow(deg_np, cmap="gray", vmin=0, vmax=1)
            axes[0].set_title("NoisyLR Input")
            axes[0].axis("off")

            # 2. Ground Truth
            axes[1].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            # 3. Prediction
            axes[2].imshow(pred_np, cmap="gray", vmin=0, vmax=1)
            axes[2].set_title(f"Prediction (SSIM={s:.4f})")
            axes[2].axis("off")

            # 4. Difference Map (Grayscale)
            axes[3].imshow(diff_map, cmap="gray", vmin=0, vmax=0.3)
            axes[3].set_title("Difference Map (|Pred - GT|)")
            axes[3].axis("off")

            # 5. Error Heatmap (Inferno)
            im = axes[4].imshow(diff_map, cmap="inferno", vmin=0, vmax=0.3)
            axes[4].set_title(f"Error Heatmap (PSNR={p:.2f}dB)")
            axes[4].axis("off")
            fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

            fig.suptitle(f"Sample: {filename}  (SSIM={s:.4f}  PSNR={p:.2f}dB  LPIPS={l:.4f})", fontsize=14)
            plt.tight_layout()

            save_path = os.path.join(output_dir, f"advanced_eval_{base_name}.png")
            plt.savefig(save_path, dpi=120)
            plt.close(fig)
            print(f"Saved advanced visual figure: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", default="data/train/GT")
    parser.add_argument("--deg_dir", default="data/train/NoisyLR")
    parser.add_argument("--checkpoint", default="checkpoints/stage3_nafnet_unet_best_ssim.pt")
    parser.add_argument("--stage", default="stage3_nafnet_unet")
    parser.add_argument("--use_film", action="store_true")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()

    generate_advanced_visuals(
        args.gt_dir, args.deg_dir, args.checkpoint,
        stage=args.stage, output_dir=args.output_dir,
        num_samples=args.num_samples, use_film=args.use_film,
    )


if __name__ == "__main__":
    main()
