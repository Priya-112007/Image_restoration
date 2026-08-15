import argparse
import glob
import os
import time
from collections import defaultdict

import numpy as np
import torch

from model import build_model

def group_files_by_shape(files):
    """Groups file paths by their array shape, WITHOUT loading full files
    into memory (uses mmap for a fast peek at just the header)."""
    groups = defaultdict(list)
    for f in files:
        arr = np.load(f, mmap_mode="r")
        shape = arr.shape[-2:]  # (H, W), ignore any leading channel dim
        groups[shape].append(f)
    return groups

def load_batch(paths, device, dtype):
    arrays = []
    for p in paths:
        arr = np.load(p).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]
        # Safeguard: clip input dynamic range to [0.0, 1.0]
        arr = np.clip(arr, 0.0, 1.0)
        arrays.append(arr)
    batch = np.stack(arrays, axis=0)
    tensor = torch.from_numpy(batch).unsqueeze(1).to(device=device, dtype=dtype)
    return tensor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--weights",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights.pt"),
        help="Path to the trained checkpoint. Defaults to weights.pt in the repo root.",
    )
    parser.add_argument("--stage", default="stage3_nafnet_unet",
                         choices=["stage0_baseline", "stage1_film", "stage2_hybrid", "stage3_nafnet_unet"],
                         help="Must match the stage the weights file was trained as.")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    t_start = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Using device: {device}")

    model = build_model(args.stage).to(device)
    if os.path.exists(args.weights):
        ckpt = torch.load(args.weights, map_location=device)
        state_dict = ckpt["ema"] if "ema" in ckpt else ckpt["model"]
        model.load_state_dict(state_dict)
    model.eval()
    model.to(dtype)

    if device == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception:
            pass

    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.npy")))
    if len(files) == 0:
        raise FileNotFoundError(f"No .npy files found in {args.input_dir}")
    print(f"Found {len(files)} input files")

    shape_groups = group_files_by_shape(files)
    print(f"Grouped into {len(shape_groups)} distinct input shape(s): "
          f"{list(shape_groups.keys())}")

    t_ready = time.time()
    processed = 0

    with torch.no_grad():
        for shape, group_files in shape_groups.items():
            for i in range(0, len(group_files), args.batch_size):
                batch_files = group_files[i:i + args.batch_size]
                batch = load_batch(batch_files, device, dtype)

                out = model(batch).float().clamp(0, 1).cpu().numpy()

                for j, f in enumerate(batch_files):
                    restored = out[j, 0]
                    out_path = os.path.join(args.output_dir, os.path.basename(f))
                    np.save(out_path, restored)

                processed += len(batch_files)

    t_end = time.time()
    print(f"Startup + model load : {t_ready - t_start:.2f}s")
    print(f"Inference + write    : {t_end - t_ready:.2f}s")
    print(f"Total                : {t_end - t_start:.2f}s "
          f"({processed} images, {(t_end - t_start) / max(processed, 1) * 1000:.1f} ms/image)")

if __name__ == "__main__":
    main()