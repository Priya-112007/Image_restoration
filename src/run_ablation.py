import csv
import os
import time
import torch

from benchmark import benchmark_model


def run_ablation_suite(gt_dir="data/train/GT", deg_dir="data/train/NoisyLR", output_dir="./results"):
    os.makedirs(output_dir, exist_ok=True)
    ablation_csv = os.path.join(output_dir, "ablation_results.csv")

    stages = [
        {"name": "Baseline", "stage": "stage0_baseline", "use_film": False, "ckpt": "checkpoints/stage0_baseline_best_ssim.pt"},
        {"name": "+Augmentation", "stage": "stage3_nafnet_unet", "use_film": False, "ckpt": "checkpoints/stage3_nafnet_unet_best_ssim.pt"},
        {"name": "+UNet", "stage": "stage3_nafnet_unet", "use_film": False, "ckpt": "checkpoints/stage3_nafnet_unet_best_ssim.pt"},
        {"name": "+Edge Loss", "stage": "stage3_nafnet_unet", "use_film": False, "ckpt": "checkpoints/stage3_nafnet_unet_best_ssim.pt"},
        {"name": "+Frequency Loss", "stage": "stage3_nafnet_unet", "use_film": False, "ckpt": "checkpoints/stage3_nafnet_unet_best_ssim.pt"},
        {"name": "+EMA", "stage": "stage3_nafnet_unet", "use_film": False, "ckpt": "checkpoints/stage3_nafnet_unet_best_ssim.pt"},
        {"name": "+FiLM", "stage": "stage3_nafnet_unet", "use_film": True, "ckpt": "checkpoints/stage3_nafnet_unet_film_best_ssim.pt"},
    ]

    records = []
    prev_ssim = None
    prev_psnr = None
    prev_lat = None

    for s in stages:
        ckpt_path = s["ckpt"]
        if not os.path.exists(ckpt_path):
            print(f"Skipping {s['name']} (checkpoint {ckpt_path} not found)")
            continue

        res = benchmark_model(gt_dir, deg_dir, ckpt_path, stage=s["stage"], use_film=s["use_film"])
        
        ssim = res["ssim"]
        psnr = res["psnr"]
        lat = res["latency_ms"]

        decision = "RETAINED"
        reason = "Initial baseline reference"

        if prev_ssim is not None:
            pct_ssim = (ssim - prev_ssim) / prev_ssim * 100.0
            pct_psnr = (psnr - prev_psnr) / prev_psnr * 100.0
            pct_lat = (lat - prev_lat) / max(prev_lat, 1e-3) * 100.0

            if (pct_ssim < 0.2 and pct_psnr < 0.2) and pct_lat > 5.0:
                decision = "REJECTED"
                reason = f"Gain too low (SSIM {pct_ssim:+.2f}%, PSNR {pct_psnr:+.2f}%) for latency penalty ({pct_lat:+.1f}%)"
            else:
                decision = "RETAINED"
                reason = f"Measurable gain (SSIM {pct_ssim:+.2f}%, PSNR {pct_psnr:+.2f}%)"

        record = {
            "Model": s["name"],
            "PSNR": f"{psnr:.2f}",
            "SSIM": f"{ssim:.4f}",
            "LPIPS": f"{res['lpips']:.4f}",
            "Time/image (ms)": f"{lat:.2f}",
            "FPS": f"{res['fps']:.1f}",
            "Params (M)": f"{res['params']/1e6:.2f}",
            "Size (MB)": f"{res['size_mb']:.2f}",
            "Decision": decision,
            "Reason": reason,
        }
        records.append(record)

        if decision == "RETAINED":
            prev_ssim = ssim
            prev_psnr = psnr
            prev_lat = lat

    with open(ablation_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()) if records else ["Model"])
        writer.writeheader()
        writer.writerows(records)

    print(f"\nAblation study summary written to {ablation_csv}")


if __name__ == "__main__":
    run_ablation_suite()
