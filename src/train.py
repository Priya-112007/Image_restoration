import argparse
import copy
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from dataset import RestorationDataset, list_paired_files, split_pairs
from losses import restoration_loss, ssim as ssim_fn
from model import build_model

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


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
    total_ssim, total_psnr, n = 0.0, 0.0, 0
    with torch.no_grad():
        for deg, gt in val_loader:
            deg, gt = deg.to(device), gt.to(device)
            pred = model(deg).clamp(0, 1)
            total_ssim += ssim_fn(pred, gt).item()
            total_psnr += compute_psnr(pred, gt)
            n += 1
    model.train()
    return total_ssim / max(n, 1), total_psnr / max(n, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                         choices=["stage0_baseline", "stage1_film", "stage2_hybrid"])
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--deg_dir", required=True)
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8,
                         help="Default 8 is a safe size for a free-tier T4 GPU.")
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--warmup_epochs", type=int, default=5,
                         help="Epochs to linearly ramp up the learning rate before cosine decay.")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                         help="Max gradient norm. Prevents the kind of mid-training "
                              "instability seen with attention-based blocks (Stage 2).")
    parser.add_argument("--no_resume", action="store_true",
                         help="Start fresh instead of resuming from a checkpoint.")
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("WARNING: no GPU detected. In Colab: Runtime -> Change runtime "
              "type -> select a GPU. Training on CPU will be extremely slow.")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    pairs = list_paired_files(args.gt_dir, args.deg_dir)
    train_pairs, val_pairs = split_pairs(pairs, val_fraction=0.1, seed=42)
    print(f"Train pairs: {len(train_pairs)}  |  Val pairs: {len(val_pairs)}")

    train_ds = RestorationDataset(train_pairs, patch_size=args.patch_size, train=True)
    val_ds = RestorationDataset(val_pairs, patch_size=args.patch_size, train=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    model = build_model(args.stage).to(device)
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = make_warmup_cosine_scheduler(optimizer, args.epochs, args.warmup_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    last_ckpt_path = os.path.join(args.ckpt_dir, f"{args.stage}_last.pt")
    best_ckpt_path = os.path.join(args.ckpt_dir, f"{args.stage}_best.pt")

    start_epoch = 0
    best_val_ssim = -1.0

    if not args.no_resume and os.path.exists(last_ckpt_path):
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_ssim = ckpt.get("best_val_ssim", -1.0)
        print(f"Resumed '{args.stage}' from epoch {start_epoch} "
              f"(best val SSIM so far: {best_val_ssim:.4f})")

    use_wandb = args.use_wandb and WANDB_AVAILABLE
    if args.use_wandb and not WANDB_AVAILABLE:
        print("wandb not installed — continuing without experiment tracking. "
              "Run: pip install wandb")
    if use_wandb:
        wandb.init(project="kla-restoration", name=args.stage, resume="allow")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0

        for deg, gt in train_loader:
            deg, gt = deg.to(device), gt.to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                 enabled=(device == "cuda")):
                pred = model(deg)
                loss, loss_parts = restoration_loss(pred, gt)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.mul_(0.999).add_(p, alpha=0.001)

            running_loss += loss.item()

        current_lr = optimizer.param_groups[0]["lr"]  # LR actually used this epoch
        scheduler.step()
        avg_loss = running_loss / max(len(train_loader), 1)
        print(f"[{args.stage}] Epoch {epoch}: loss={avg_loss:.4f} "
              f"lr={current_lr:.2e} ({time.time() - epoch_start:.1f}s)")

        if epoch % args.val_every == 0 or epoch == args.epochs - 1:
            val_ssim, val_psnr = validate(ema_model, val_loader, device)
            print(f"  -> val SSIM={val_ssim:.4f}  val PSNR={val_psnr:.2f}")

            if use_wandb:
                wandb.log({"epoch": epoch, "train_loss": avg_loss,
                           "val_ssim": val_ssim, "val_psnr": val_psnr})

            if val_ssim > best_val_ssim:
                best_val_ssim = val_ssim
                torch.save({
                    "model": model.state_dict(),
                    "ema": ema_model.state_dict(),
                    "epoch": epoch,
                    "val_ssim": val_ssim,
                    "val_psnr": val_psnr,
                }, best_ckpt_path)
                print(f"  -> new best checkpoint saved (SSIM={val_ssim:.4f})")
        torch.save({
            "model": model.state_dict(),
            "ema": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_ssim": best_val_ssim,
        }, last_ckpt_path)

    print(f"Training complete. Best val SSIM: {best_val_ssim:.4f}")
    print(f"Best checkpoint: {best_ckpt_path}")

if __name__ == "__main__":
    main()