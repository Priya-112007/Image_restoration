import argparse
import os
import time
import numpy as np
import torch

from dataset import RestorationDataset, list_paired_files, split_pairs
from losses import ssim as ssim_fn, psnr as psnr_fn, lpips_loss
from model import build_model, count_parameters


def estimate_gflops(model, input_size=(1, 1, 256, 256), device="cpu"):
    try:
        from thop import profile
        x = torch.randn(*input_size).to(device)
        macs, params = profile(model, inputs=(x,), verbose=False)
        gflops = (macs * 2) / 1e9
        return gflops
    except Exception:
        # Fallback estimation for conv2d layers
        total_macs = 0
        x = torch.randn(*input_size).to(device)
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                h_out = x.shape[-2] // m.stride[0]
                w_out = x.shape[-1] // m.stride[1]
                macs = m.in_channels * m.out_channels * m.kernel_size[0] * m.kernel_size[1] * h_out * w_out / m.groups
                total_macs += macs
        return (total_macs * 2) / 1e9


def benchmark_model(gt_dir, deg_dir, checkpoint_path, stage="stage3_nafnet_unet", use_film=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Benchmarking on device: {device}")

    pairs = list_paired_files(gt_dir, deg_dir)
    _, val_pairs = split_pairs(pairs, val_fraction=0.1, seed=42)
    val_ds = RestorationDataset(val_pairs, train=False)

    model = build_model(stage, use_film=use_film).to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["ema"] if "ema" in ckpt else ckpt["model"]
        model.load_state_dict(state_dict, strict=False)
    model.eval()

    params, size_mb = count_parameters(model)
    gflops = estimate_gflops(model, input_size=(1, 1, 256, 256), device=device)

    # Warmup
    dummy = torch.randn(1, 1, 256, 256).to(device)
    for _ in range(5):
        _ = model(dummy)

    results = []
    latencies = []

    with torch.no_grad():
        for idx in range(len(val_ds)):
            deg, gt = val_ds[idx]
            deg_b = deg.unsqueeze(0).to(device)
            gt_b = gt.unsqueeze(0).to(device)

            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            pred = model(deg_b).clamp(0, 1)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.time() - t0) * 1000.0

            s = ssim_fn(pred, gt_b).item()
            p = psnr_fn(pred, gt_b).item()
            l = lpips_loss(pred, gt_b).item()

            latencies.append(elapsed_ms)
            results.append({"ssim": s, "psnr": p, "lpips": l})

    avg_ssim = float(np.mean([r["ssim"] for r in results]))
    avg_psnr = float(np.mean([r["psnr"] for r in results]))
    avg_lpips = float(np.mean([r["lpips"] for r in results]))
    avg_latency = float(np.mean(latencies))
    fps = 1000.0 / max(avg_latency, 1e-3)

    metrics = {
        "stage": stage,
        "use_film": use_film,
        "checkpoint": checkpoint_path,
        "psnr": avg_psnr,
        "ssim": avg_ssim,
        "lpips": avg_lpips,
        "latency_ms": avg_latency,
        "fps": fps,
        "params": params,
        "size_mb": size_mb,
        "gflops": gflops,
    }

    print("\n" + "=" * 60)
    print(f"Stage:               {stage} (FiLM={use_film})")
    print(f"Checkpoint:          {checkpoint_path}")
    print(f"Avg PSNR:            {avg_psnr:.2f} dB")
    print(f"Avg SSIM:            {avg_ssim:.4f}")
    print(f"Avg LPIPS:           {avg_lpips:.4f}")
    print(f"Inference Latency:   {avg_latency:.2f} ms/image ({fps:.1f} FPS)")
    print(f"Parameters:          {params / 1e6:.2f} M ({size_mb:.2f} MB)")
    print(f"GFLOPs (256x256):    {gflops:.2f}")
    print("=" * 60)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", default="data/train/GT")
    parser.add_argument("--deg_dir", default="data/train/NoisyLR")
    parser.add_argument("--checkpoint", default="checkpoints/stage3_nafnet_unet_best_ssim.pt")
    parser.add_argument("--stage", default="stage3_nafnet_unet")
    parser.add_argument("--use_film", action="store_true")
    args = parser.parse_args()

    benchmark_model(args.gt_dir, args.deg_dir, args.checkpoint, stage=args.stage, use_film=args.use_film)


if __name__ == "__main__":
    main()
