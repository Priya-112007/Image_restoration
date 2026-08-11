import os

import matplotlib.pyplot as plt
import torch

from dataset import RestorationDataset, list_paired_files, split_pairs
from model import build_model


def load_val_set(gt_dir, deg_dir, val_fraction=0.1, seed=42):
    pairs = list_paired_files(gt_dir, deg_dir)
    _, val_pairs = split_pairs(pairs, val_fraction=val_fraction, seed=seed)
    return RestorationDataset(val_pairs, train=False), val_pairs


def load_model(stage, checkpoint_path, device):
    model = build_model(stage).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["ema"] if "ema" in ckpt else ckpt["model"]
    model.load_state_dict(state_dict)
    model.eval()
    return model


def show_predictions(gt_dir, deg_dir, checkpoints, indices=None, num_samples=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_ds, val_pairs = load_val_set(gt_dir, deg_dir)

    models = {
        label: load_model(stage, path, device)
        for label, (stage, path) in checkpoints.items()
    }

    if indices is None:
        indices = list(range(num_samples))

    n_cols = 2 + len(models)

    with torch.no_grad():
        for idx in indices:
            deg, gt = val_ds[idx]
            filename = os.path.basename(val_pairs[idx][0])
            deg_b = deg.unsqueeze(0).to(device)
            gt_b = gt.unsqueeze(0).to(device)

            preds = {}
            for label, model in models.items():
                preds[label] = model(deg_b).clamp(0, 1).squeeze().cpu().numpy()

            fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
            axes[0].imshow(deg_b.squeeze().cpu().numpy(), cmap="gray")
            axes[0].set_title("NoisyLR")
            axes[1].imshow(gt_b.squeeze().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            axes[1].set_title("Ground Truth")
            for i, (label, pred_img) in enumerate(preds.items()):
                axes[2 + i].imshow(pred_img, cmap="gray", vmin=0, vmax=1)
                axes[2 + i].set_title(label)
            for ax in axes:
                ax.axis("off")
            fig.suptitle(f"{filename}  (validation index {idx})")
            plt.tight_layout()
            plt.show()