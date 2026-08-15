import argparse
import copy
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from dataset import RestorationDataset, list_paired_files, split_pairs
from model import build_model, PatchDiscriminator
import csv

from losses import (
    restoration_loss, ssim as ssim_fn, lpips_loss,
    discriminator_loss, generator_adversarial_loss,
)

def make_warmup_cosine_scheduler(optimizer, epochs, warmup_epochs=5):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, (epochs - warmup_epochs))
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_psnr(pred, gt):
    mse = torch.mean((pred - gt) ** 2)
    if mse.item() == 0:
        return 100.0
    return (10 * torch.log10(1.0 / mse)).item()

def validate(model, val_loader, device):
    model.eval()
    total_loss, total_ssim, total_psnr, total_lpips, n = 0.0, 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for deg, gt in val_loader:
            deg, gt = deg.to(device), gt.to(device)
            pred = model(deg).clamp(0, 1)
            loss, _ = restoration_loss(pred, gt)
            total_loss += loss.item()
            total_ssim += ssim_fn(pred, gt).item()
            total_psnr += compute_psnr(pred, gt)
            total_lpips += lpips_loss(pred, gt).item()
            n += 1
    model.train()
    return total_loss / max(n, 1), total_ssim / max(n, 1), total_psnr / max(n, 1), total_lpips / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                         choices=["stage0_baseline", "stage1_film", "stage2_hybrid", "stage3_nafnet_unet"])
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--deg_dir", required=True)
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20,
                         help="Early stopping patience: stop if no val SSIM improvement for N epochs.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--use_film", action="store_true", help="Enable FiLM conditioning in NAFNetUNet.")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--use_synthetic_degradation", action="store_true")
    parser.add_argument("--synthetic_prob", type=float, default=0.3)
    parser.add_argument("--use_cutmix", action="store_true")
    parser.add_argument("--cutmix_prob", type=float, default=0.3)
    parser.add_argument("--use_gamma_jitter", action="store_true")
    parser.add_argument("--w_ssim", type=float, default=0.3)
    parser.add_argument("--w_edge", type=float, default=0.1)
    parser.add_argument("--w_freq", type=float, default=0.1)
    parser.add_argument("--use_gan", action="store_true")
    parser.add_argument("--w_adv", type=float, default=0.01)
    parser.add_argument("--disc_lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    pairs = list_paired_files(args.gt_dir, args.deg_dir)
    train_pairs, val_pairs = split_pairs(pairs, val_fraction=0.1, seed=42)
    print(f"Train pairs: {len(train_pairs)}  |  Val pairs: {len(val_pairs)}")

    train_ds = RestorationDataset(
        train_pairs, patch_size=args.patch_size, train=True,
        use_synthetic_degradation=args.use_synthetic_degradation,
        synthetic_prob=args.synthetic_prob,
        use_cutmix=args.use_cutmix,
        cutmix_prob=args.cutmix_prob,
        use_gamma_jitter=args.use_gamma_jitter,
    )
    val_ds = RestorationDataset(val_pairs, patch_size=args.patch_size, train=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=(device == "cuda"), persistent_workers=True, prefetch_factor=2,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    model = build_model(args.stage, use_film=args.use_film).to(device)
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_warmup_cosine_scheduler(optimizer, args.epochs, args.warmup_epochs)

    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    discriminator, disc_optimizer = None, None
    if args.use_gan:
        discriminator = PatchDiscriminator(c=32).to(device)
        disc_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=args.disc_lr)

    last_ckpt_path = os.path.join(args.ckpt_dir, f"{args.stage}_last.pt")
    best_ssim_path = os.path.join(args.ckpt_dir, f"{args.stage}_best_ssim.pt")
    best_psnr_path = os.path.join(args.ckpt_dir, f"{args.stage}_best_psnr.pt")
    best_lpips_path = os.path.join(args.ckpt_dir, f"{args.stage}_best_lpips.pt")
    final_submission_path = os.path.join(args.ckpt_dir, f"final_submission.pt")
    history_csv_path = os.path.join(args.ckpt_dir, f"{args.stage}_history.csv")

    start_epoch = 0
    best_val_ssim = -1.0
    best_val_psnr = -1.0
    best_val_lpips = 999.0
    no_improve_epochs = 0

    if not args.no_resume and os.path.exists(last_ckpt_path):
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_ssim = ckpt.get("best_val_ssim", -1.0)
        best_val_psnr = ckpt.get("best_val_psnr", -1.0)
        best_val_lpips = ckpt.get("best_val_lpips", 999.0)
        print(f"Resumed '{args.stage}' from epoch {start_epoch} (best SSIM: {best_val_ssim:.4f})")

    history_records = []

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0

        for deg, gt in train_loader:
            deg, gt = deg.to(device), gt.to(device)

            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                 dtype=torch.float16 if device == "cuda" else torch.float32,
                                 enabled=(device == "cuda")):
                pred = model(deg)

            if args.use_gan:
                disc_optimizer.zero_grad()
                with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                     dtype=torch.float16 if device == "cuda" else torch.float32,
                                     enabled=(device == "cuda")):
                    d_loss = discriminator_loss(discriminator, gt, pred)
                scaler.scale(d_loss).backward()
                scaler.unscale_(disc_optimizer)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=args.grad_clip)
                scaler.step(disc_optimizer)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu",
                                 dtype=torch.float16 if device == "cuda" else torch.float32,
                                 enabled=(device == "cuda")):
                loss, loss_parts = restoration_loss(
                    pred, gt, w_ssim=args.w_ssim, w_edge=args.w_edge, w_freq=args.w_freq,
                )
                if args.use_gan:
                    adv_loss = generator_adversarial_loss(discriminator, pred)
                    loss = loss + args.w_adv * adv_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.mul_(0.999).add_(p, alpha=0.001)

            running_loss += loss.item()

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        avg_train_loss = running_loss / max(len(train_loader), 1)

        val_loss, val_ssim, val_psnr, val_lpips = validate(ema_model, val_loader, device)
        epoch_time = time.time() - epoch_start
        print(f"[{args.stage}] Epoch {epoch:03d}/{args.epochs}: train_loss={avg_train_loss:.4f} "
              f"val_loss={val_loss:.4f} SSIM={val_ssim:.4f} PSNR={val_psnr:.2f}dB LPIPS={val_lpips:.4f} "
              f"lr={current_lr:.2e} ({epoch_time:.1f}s)")

        history_records.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_ssim": val_ssim,
            "val_psnr": val_psnr,
            "val_lpips": val_lpips,
            "lr": current_lr,
            "epoch_time": epoch_time,
        })

        # Save history CSV
        with open(history_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_ssim", "val_psnr", "val_lpips", "lr", "epoch_time"])
            writer.writeheader()
            writer.writerows(history_records)

        improved = False

        # Specialized Checkpoint 1: Best SSIM
        if val_ssim > best_val_ssim:
            best_val_ssim = val_ssim
            improved = True
            torch.save({
                "model": model.state_dict(), "ema": ema_model.state_dict(),
                "epoch": epoch, "val_ssim": val_ssim, "val_psnr": val_psnr, "val_lpips": val_lpips,
            }, best_ssim_path)
            # Also update final submission checkpoint
            torch.save({
                "model": model.state_dict(), "ema": ema_model.state_dict(),
                "epoch": epoch, "val_ssim": val_ssim, "val_psnr": val_psnr, "val_lpips": val_lpips,
            }, final_submission_path)
            print(f"  -> new best SSIM checkpoint saved ({val_ssim:.4f})")

        # Specialized Checkpoint 2: Best PSNR
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save({
                "model": model.state_dict(), "ema": ema_model.state_dict(),
                "epoch": epoch, "val_ssim": val_ssim, "val_psnr": val_psnr, "val_lpips": val_lpips,
            }, best_psnr_path)
            print(f"  -> new best PSNR checkpoint saved ({val_psnr:.2f}dB)")

        # Specialized Checkpoint 3: Best LPIPS
        if val_lpips < best_val_lpips:
            best_val_lpips = val_lpips
            torch.save({
                "model": model.state_dict(), "ema": ema_model.state_dict(),
                "epoch": epoch, "val_ssim": val_ssim, "val_psnr": val_psnr, "val_lpips": val_lpips,
            }, best_lpips_path)

        # Early Stopping Logic
        if improved:
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= args.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs "
                      f"({args.patience} epochs without SSIM improvement).")
                break

        # Save Last Checkpoint
        torch.save({
            "model": model.state_dict(), "ema": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "epoch": epoch, "best_val_ssim": best_val_ssim, "best_val_psnr": best_val_psnr,
        }, last_ckpt_path)

    print(f"Training complete. Peak SSIM: {best_val_ssim:.4f} | Peak PSNR: {best_val_psnr:.2f}dB | Best LPIPS: {best_val_lpips:.4f}")
    print(f"Checkpoints saved to {args.ckpt_dir}: best_ssim.pt, best_psnr.pt, best_lpips.pt, final_submission.pt, last.pt")
if __name__ == "__main__":
    main()