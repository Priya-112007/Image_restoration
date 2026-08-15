import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_curves(csv_path, output_dir="./results"):
    os.makedirs(output_dir, exist_ok=True)
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    
    epochs = data["epoch"]
    train_loss = data["train_loss"]
    val_loss = data["val_loss"]
    val_ssim = data["val_ssim"]
    val_psnr = data["val_psnr"]
    val_lpips = data["val_lpips"]
    lr = data["lr"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Training Loss
    axes[0, 0].plot(epochs, train_loss, 'b-', label="Train Loss")
    axes[0, 0].set_title("Training Loss vs Epoch")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True)

    # 2. Validation Loss
    axes[0, 1].plot(epochs, val_loss, 'r-', label="Val Loss")
    axes[0, 1].set_title("Validation Loss vs Epoch")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True)

    # 3. Learning Rate
    axes[0, 2].plot(epochs, lr, 'g-', label="Learning Rate")
    axes[0, 2].set_title("Learning Rate vs Epoch")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("LR")
    axes[0, 2].set_yscale("log")
    axes[0, 2].grid(True)

    # 4. SSIM
    axes[1, 0].plot(epochs, val_ssim, 'm-', label="Val SSIM")
    axes[1, 0].set_title("SSIM vs Epoch")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("SSIM")
    axes[1, 0].grid(True)

    # 5. PSNR
    axes[1, 1].plot(epochs, val_psnr, 'c-', label="Val PSNR")
    axes[1, 1].set_title("PSNR (dB) vs Epoch")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("PSNR (dB)")
    axes[1, 1].grid(True)

    # 6. LPIPS
    axes[1, 2].plot(epochs, val_lpips, 'k-', label="Val LPIPS")
    axes[1, 2].set_title("LPIPS vs Epoch")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("LPIPS")
    axes[1, 2].grid(True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "learning_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved learning curve plots to {plot_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to history CSV file")
    parser.add_argument("--output_dir", default="./results")
    args = parser.parse_args()
    plot_curves(args.csv_path, args.output_dir)

if __name__ == "__main__":
    main()
