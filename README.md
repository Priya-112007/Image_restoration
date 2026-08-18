# AI-Based Blind Image Restoration for Semiconductor Metrology

[![Team: Twatosphere](https://img.shields.io/badge/Team-Twatosphere-8A2BE2?style=for-the-badge&logo=shield)](https://github.com/Priya-112007/Image_restoration)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-76B900?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Team Details
- **Team Name**: **Twatosphere**
- **Problem Statement**: **AI-Based Restoration of Degraded Images** (SEMICON Hackathon 2026)

---

## 📌 Project Overview & Solution Architecture

This repository contains the state-of-the-art submission for Team **Twatosphere**. Our model, **Stage 3 NAFNet-UNet v2 with Large Kernel Attention (LKA) & Multi-Scale Deep Supervision**, is an ultra-lightweight ($0.77\text{M}$ parameters), high-speed deep learning framework specifically engineered to restore severely degraded semiconductor wafer inspection images.

### 🔬 Compound Degradations Addressed
1. **Multiplicative Laser Speckle Noise**: Granular interference patterns from coherent laser illumination.
2. **Additive Electronic Gaussian Read Noise**: High-frequency electronic sensor noise.
3. **Spatial Gaussian Optical Blur ($\sigma=1.5\text{--}2.5$)**: Primary structural bottleneck causing edge spreading.
4. **$2\times$ Spatial Downsampling**: Loss of sub-nanometer boundary details and spatial aliasing.

---

## 🏆 Verified Empirical Achievements

| Metric | Baseline Score | Stage 2 Score | **Stage 3 Peak (Ours)** | Total Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Validation SSIM** | 0.5717 | 0.6558 | **$\mathbf{0.7719}$** | **$+0.2002\text{ SSIM}$** ($+35.0\%$ Relative Boost!) |
| **Validation PSNR** | 24.72 dB | 25.62 dB | **$\mathbf{28.10\text{ dB}}$** | **$+3.38\text{ dB}$** Signal Fidelity Gain |
| **Validation LPIPS** | 0.3405 | 0.2200 | **$\mathbf{0.1737}$** | Crisp Perceptual Sharpening |
| **Sample `001599.npy` SSIM** | 0.5200 | 0.7206 | **$\mathbf{0.8207}$** | **Broke 0.82 SSIM Milestone!** |

---

## 📁 Required Submission Repository Structure

```
Twatosphere/
├── run.py                                     # Official Entry Script (python run.py <input-dir> <output-dir>)
├── requirements.txt                           # Complete environment dependencies with versions
├── README.md                                  # Setup, replication, and technical architecture guide
├── weights.pt                                 # Model checkpoint weights (2.94 MB)
├── models/
│   └── weights.pt                             # Duplicate checkpoint in models/ folder
├── evaluate.py                                # Standalone evaluation alias script
├── WALKTHROUGH.md                             # Comprehensive technical deep-dive report
├── PROJECT_PROPOSED_SOLUTION_AND_METHODOLOGY.md # 6-section system analysis
├── src/
│   ├── model.py                               # NAFNet-UNet v2 & LKA block architecture
│   ├── train.py                               # Training reproduction script
│   ├── losses.py                              # Charbonnier + MS-SSIM + Edge + Freq loss functions
│   ├── dataset.py                             # Dataset loader & synthetic noise augmentations
│   └── benchmark.py                           # Benchmark evaluation & latency profiler
├── results/
│   ├── advanced_eval_001599.png               # 5-panel output comparison figure (Epoch 53)
│   ├── learning_curves.png                    # Diagnostic 6-plot learning curve chart
│   ├── system_pipeline_architecture.png       # Macro system pipeline diagram
│   └── detailed_micro_module_pipeline.png     # Micro-module architectural diagram
└── data/
    ├── Test_NoisyLR/NoisyLR/                  # Test set input degraded .npy arrays (400 files)
    └── test_output/                           # Restored output .npy arrays (400 files)
```

---

## ⚡ Instructions for KLA Evaluators: Execution Guide

### 1. Environment Setup
Clone the repository and install required packages:

```bash
git clone https://github.com/Priya-112007/Image_restoration.git
cd Image_restoration
pip install -r requirements.txt
```

### 2. Standalone Inference Execution (`run.py`)
Run the official entry script `run.py` directly from the repository root:

```bash
python run.py <input-degraded-dir> <output-restored-dir>
```

#### Example Usage Command:
```bash
python run.py data/Test_NoisyLR/NoisyLR data/test_output
```

The script automatically:
- ✅ Reads all `.npy` files from `<input-degraded-dir>`.
- ✅ Creates `<output-restored-dir>` if it does not already exist.
- ✅ Loads trained model weights from `models/weights.pt` (or `weights.pt`).
- ✅ Generates one restored `.npy` file per input file with **identical filenames**.
- ✅ Ensures restored outputs are 2D grayscale arrays of shape **$(256, 256)$**.
- ✅ Sanitizes outputs (removes any `NaN` / `Inf` values and strictly bounds range to **$[0.0, 1.0]$**).
- ✅ Runs **100% offline on NVIDIA GPUs** without requiring internet access, API keys, or user interaction.

---

## 🏛️ Deep Technical Architecture Breakdown

```
[Input: 1x128x128] ---> [Torch Clamp [0, 1]] ---> [Conv2d(1, 32, 3)] ---> [Enc Level 1: 2x LKA Block (32c)]
                                                                                  |
                                                                        [Down1: Conv2d(32, 64, 2, s=2)]
                                                                                  |
                                                                       ---> [Enc Level 2: 2x LKA Block (64c)]
                                                                                  |
                                                                        [Down2: Conv2d(64, 128, 2, s=2)]
                                                                                  |
                                                                       ---> [Bottleneck: 4x LKA Block (128c)]
                                                                                  |
                                                                        [Up2: PixelShuffle(2) -> 64c]
                                                                                  |
                                                                       ---> [Dec Level 2: Fuse + 2x LKA Block (64c)]
                                                                             |                          |
                                                                 [Aux Head 2: Conv+PS]       [Up1: PixelShuffle(2) -> 32c]
                                                                             |                          |
                                                                 [Out Half: 1x128x128]      ---> [Dec Level 1: Fuse + 2x LKA Block (32c)]
                                                                                                        |
                                                                                             [SR Head: Conv+PixelShuffle(2)]
                                                                                                        |
                                                                                             [Skip Base: Bicubic 2x]
                                                                                                        |
                                                                                             [Clamp(Out + Skip, 0, 1)]
                                                                                                        |
                                                                                             [Output: 1x256x256]
```

1. **23-Pixel Large Kernel Attention (LKA)**: Factorizes depthwise convolutions into $5\times 5 \text{ DW} \to 7\times 7 \text{ Dilated DW (dilation=3)} \to 1\times 1 \text{ Conv}$, expanding spatial receptive field to **23 pixels** for optical blur inversion.
2. **Non-Linear Activation Free (NAF) `SimpleGate`**: Replaces GELU/ReLU with channel gating ($x_1 \times x_2$), computing non-linear features at zero extra activation cost.
3. **Multi-Scale Deep Supervision**: Auxiliary projection head `aux_head2` enforces coarse structural loss during training, detached during evaluation ($\mathbf{0.0\text{ ms}}$ latency penalty).
4. **PixelShuffle Super-Resolution Head**: Upsamples feature channels directly to $256\times 256$ spatial resolution, combined with long bicubic skip additions.

---

## 🏋️ Training Reproduction

To reproduce model training from scratch:

```bash
python src/train.py \
  --stage stage3_nafnet_unet \
  --gt_dir data/train/GT \
  --deg_dir data/train/NoisyLR \
  --ckpt_dir checkpoints \
  --epochs 150 \
  --patch_size 256 \
  --batch_size 8 \
  --use_synthetic_degradation \
  --w_ssim 0.4 \
  --w_edge 0.1 \
  --w_freq 0.1
```

---

## 🚀 NVIDIA H100 & Production Hardware Compatibility

- **100% Native PyTorch Operators**: Built using standard PyTorch modules (`Conv2d`, `GroupNorm`, `SimpleGate`, `PixelShuffle`) with zero custom C++ CUDA compilation.
- **NVIDIA H100 GPU Optimization**: Supports PyTorch 2.0 `torch.compile(mode="reduce-overhead")` and FP16 AMP, delivering sub-4.5ms latency ($>200\text{ FPS}$) on NVIDIA H100 GPUs.
- **Memory Safety**: Uses $<1.8\text{ GB}$ VRAM, ensuring zero Out-Of-Memory (OOM) failures under benchmarking.

---

## 📜 License
This project is licensed under the MIT License.
