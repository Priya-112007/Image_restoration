import argparse
import glob
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

# Dynamically resolve src path for offline execution
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from model import build_model
except ImportError:
    from src.model import build_model  # type: ignore

def group_files_by_shape(files):
    """Groups file paths by array shape using fast mmap header inspection."""
    groups = defaultdict(list)
    for f in files:
        arr = np.load(f, mmap_mode="r")
        shape = arr.shape[-2:]  # (H, W), ignore leading channel dimension if present
        groups[shape].append(f)
    return groups

def load_batch(paths, device, dtype):
    arrays = []
    for p in paths:
        arr = np.load(p).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]
        # Safeguard: handle nan/inf and clip input range to [0.0, 1.0]
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        arr = np.clip(arr, 0.0, 1.0)
        arrays.append(arr)
    batch = np.stack(arrays, axis=0)
    tensor = torch.from_numpy(batch).unsqueeze(1).to(device=device, dtype=dtype)
    return tensor

def main():
    parser = argparse.ArgumentParser(description="Official KLA Entry Script run.py - Team Twatosphere")
    parser.add_argument("input_dir", help="Path to input directory containing degraded .npy files")
    parser.add_argument("output_dir", help="Path to output directory for restored .npy files")
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to trained checkpoint file (defaults to models/weights.pt or weights.pt)",
    )
    parser.add_argument(
        "--stage",
        default="stage3_nafnet_unet",
        choices=["stage0_baseline", "stage1_film", "stage2_hybrid", "stage3_nafnet_unet"],
        help="Model architecture stage configuration",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
    args = parser.parse_args()

    t_start = time.time()

    # Hardware detection (CUDA / NVIDIA GPU offline execution)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[Team Twatosphere] Running on device: {device} (dtype: {dtype})")

    # Locate weights file
    weights_path = args.weights
    if weights_path is None:
        possible_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "weights.pt"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.pt"),
        ]
        for p in possible_paths:
            if os.path.exists(p):
                weights_path = p
                break
        if weights_path is None:
            weights_path = possible_paths[0]

    # Initialize model
    model = build_model(args.stage).to(device)
    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location=device)
        state_dict = ckpt["ema"] if "ema" in ckpt else (ckpt["model"] if "model" in ckpt else ckpt)
        model.load_state_dict(state_dict, strict=False)
        print(f"[Team Twatosphere] Successfully loaded model weights: {weights_path}")
    else:
        print(f"[Team Twatosphere] WARNING: Checkpoint {weights_path} not found! Running uninitialized model.")

    model.eval()
    model.to(dtype)

    # NVIDIA GPU Optimization (torch.compile for PyTorch 2.0+)
    if device == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[Team Twatosphere] PyTorch 2.0+ torch.compile enabled for GPU acceleration.")
        except Exception:
            pass

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all .npy files
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.npy")))
    if len(files) == 0:
        # Fallback check for nested search if given directory has sub-folders
        files = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.npy"), recursive=True))

    if len(files) == 0:
        raise FileNotFoundError(f"No .npy files found in input directory: {args.input_dir}")

    print(f"[Team Twatosphere] Found {len(files)} degraded input .npy file(s)")

    shape_groups = group_files_by_shape(files)
    t_ready = time.time()
    processed = 0

    with torch.no_grad():
        for _, group_files in shape_groups.items():
            for i in range(0, len(group_files), args.batch_size):
                batch_files = group_files[i:i + args.batch_size]
                batch = load_batch(batch_files, device, dtype)

                # Forward pass
                out = model(batch).float().cpu().numpy()

                for j, f in enumerate(batch_files):
                    restored = out[j, 0]  # Shape (H, W) grayscale 2D array
                    
                    # Sanitize output: handle NaN/Inf and strictly bound to [0.0, 1.0]
                    restored = np.nan_to_num(restored, nan=0.0, posinf=1.0, neginf=0.0)
                    restored = np.clip(restored, 0.0, 1.0).astype(np.float32)

                    # Write restored array with identical filename
                    out_path = os.path.join(args.output_dir, os.path.basename(f))
                    np.save(out_path, restored)

                processed += len(batch_files)

    t_end = time.time()
    print("============================================================")
    print(f"[Team Twatosphere] Restoration Complete!")
    print(f"Startup + Model Load : {t_ready - t_start:.2f}s")
    print(f"Inference + File Write: {t_end - t_ready:.2f}s")
    print(f"Total Execution Time : {t_end - t_start:.2f}s")
    print(f"Processed Images     : {processed}")
    print(f"Average Latency      : {(t_end - t_start) / max(processed, 1) * 1000:.1f} ms/image")
    print(f"Restored Output Path : {args.output_dir}")
    print("============================================================")

if __name__ == "__main__":
    main()
