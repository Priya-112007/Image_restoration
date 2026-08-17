import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_micro_pipeline():
    fig = plt.figure(figsize=(18, 12), dpi=300)
    fig.patch.set_facecolor('#0B0F19')

    # Title
    fig.text(0.5, 0.96, "Stage 3 NAFNet-UNet v2: Comprehensive Micro-Module Pipeline Architecture", 
             ha='center', va='center', color='white', fontsize=16, fontweight='bold')

    # Sub-Title
    fig.text(0.5, 0.935, "Full Breakdown of Encoder/Decoder Hierarchy, LKA Factorization, FiLM Conditioning, & PixelShuffle Heads", 
             ha='center', va='center', color='#94A3B8', fontsize=11)

    # -------------------------------------------------------------
    # PANEL A: Macro End-to-End Pipeline
    # -------------------------------------------------------------
    ax_macro = fig.add_axes([0.03, 0.52, 0.94, 0.38])
    ax_macro.set_facecolor('#0F172A')
    ax_macro.set_title("PANEL A: Macro End-to-End Multi-Scale Pipeline & Deep Supervision Flow", 
                       color='#38BDF8', fontsize=12, fontweight='bold', pad=10, loc='left')

    macro_boxes = [
        {"x": 0.5, "y": 2.5, "w": 1.8, "h": 2.5, "color": "#1E293B", "ec": "#38BDF8", 
         "title": "Raw Input (NoisyLR)", "sub": "1x128x128 .npy\nSpeckle + Blur + Noise"},

        {"x": 2.7, "y": 2.5, "w": 1.8, "h": 2.5, "color": "#1E293B", "ec": "#00F5D4", 
         "title": "Degradation Estimator\n& Dynamic Clamp", "sub": "Conv2d->ReLU->AvgPool\n32-dim FiLM Vector\ntorch.clamp([0, 1])"},

        {"x": 4.9, "y": 3.8, "w": 2.4, "h": 2.2, "color": "#1E1E38", "ec": "#818CF8", 
         "title": "UNet Encoder Level 1 & 2", "sub": "Level 1: 32c (128x128)\nLevel 2: 64c (64x64)\nDown: Strided Conv2d(s=2)"},

        {"x": 7.7, "y": 3.8, "w": 2.2, "h": 2.2, "color": "#2A1E38", "ec": "#F472B6", 
         "title": "Bottleneck (128c)", "sub": "4x NAFBlocks (128c)\n23px LKA Factorization\n(5x5 DW + 7x7 Dilated DW)"},

        {"x": 10.3, "y": 3.8, "w": 2.4, "h": 2.2, "color": "#1E1E38", "ec": "#818CF8", 
         "title": "UNet Decoder Level 2 & 1", "sub": "Up: PixelShuffle(2) Upsample\nFuse: Concat + 1x1 Conv\nLevel 2: 64c | Level 1: 32c"},

        {"x": 13.1, "y": 4.8, "w": 2.2, "h": 1.8, "color": "#312E81", "ec": "#A855F7", 
         "title": "Aux Deep Supervision", "sub": "Aux Head 2: Conv+PixelShuffle\nLevel 2 (1/2 Scale Output)\n(Training Mode Only)"},

        {"x": 13.1, "y": 2.2, "w": 2.2, "h": 2.2, "color": "#064E3B", "ec": "#10B981", 
         "title": "PixelShuffle SR Head\n& Skip Addition", "sub": "Conv2d(32, 128)->PixelShuffle(2)\n+ Bicubic 2x Skip Base\ntorch.clamp([0, 1]) -> 1x256x256"}
    ]

    for b in macro_boxes:
        rect = patches.FancyBboxPatch((b["x"], b["y"]), b["w"], b["h"], 
                                      boxstyle="round,pad=0.1", 
                                      facecolor=b["color"], edgecolor=b["ec"], linewidth=2)
        ax_macro.add_patch(rect)
        cx = b["x"] + b["w"]/2
        cy = b["y"] + b["h"]/2 + 0.35
        ax_macro.text(cx, cy, b["title"], ha='center', va='center', color='white', fontsize=9.5, fontweight='bold')
        ax_macro.text(cx, cy - 0.65, b["sub"], ha='center', va='center', color='#94A3B8', fontsize=8)

    arrow_args = dict(arrowstyle="->,head_length=0.3,head_width=0.25", color="#38BDF8", lw=2)
    ax_macro.annotate("", xy=(2.65, 3.75), xytext=(2.35, 3.75), arrowprops=arrow_args)
    ax_macro.annotate("", xy=(4.85, 4.9), xytext=(4.55, 3.75), arrowprops=arrow_args)
    ax_macro.annotate("", xy=(7.65, 4.9), xytext=(7.35, 4.9), arrowprops=arrow_args)
    ax_macro.annotate("", xy=(10.25, 4.9), xytext=(9.95, 4.9), arrowprops=arrow_args)
    ax_macro.annotate("", xy=(13.05, 5.7), xytext=(12.75, 4.9), arrowprops=arrow_args)
    ax_macro.annotate("", xy=(13.05, 3.3), xytext=(12.75, 4.9), arrowprops=arrow_args)

    ax_macro.set_xlim(-0.2, 16.0)
    ax_macro.set_ylim(1.0, 7.5)
    ax_macro.axis('off')

    # -------------------------------------------------------------
    # PANEL B: Internal Micro-Module Layer Breakdown
    # -------------------------------------------------------------
    ax_micro = fig.add_axes([0.03, 0.05, 0.94, 0.42])
    ax_micro.set_facecolor('#0F172A')
    ax_micro.set_title("PANEL B: Internal Micro-Module Layer Operations (NAFBlock, LKA, FiLM, SimpleGate, PixelShuffle)", 
                       color='#F59E0B', fontsize=12, fontweight='bold', pad=10, loc='left')

    micro_sections = [
        {"x": 0.3, "y": 0.5, "w": 4.6, "h": 5.2, "color": "#1E1E38", "ec": "#6366F1", 
         "title": "NAFBlock Micro-Architecture", 
         "content": "• GroupNorm(1, C) Layer Norm\n• Conv1x1 Expansion (C -> 2C)\n• LKA Factorized Receptive Field\n• SimpleGate Activation (x1 * x2)\n• Simplified Spatial Attention (AvgPool + 1x1 Conv)\n• FiLM Conditioning Scaling (x * (1+scale) + shift)\n• Conv1x1 Projection (C -> C) + Skip Add\n• GDFN Feed-Forward (Expansion 2.66x + Gate)"},

        {"x": 5.2, "y": 0.5, "w": 4.6, "h": 5.2, "color": "#2A1E38", "ec": "#EC4899", 
         "title": "Large Kernel Attention (LKABlock)", 
         "content": "• Depthwise 5x5 Conv (padding=2, groups=C)\n  ↳ Captures local fine edge features\n• Dilated Depthwise 7x7 Conv (padding=9, dilation=3)\n  ↳ Expands spatial receptive field to 23px\n• Pointwise 1x1 Conv (Projection)\n  ↳ Generates spatial attention map W_attn\n• Element-wise Multiplication (Input * W_attn)\n  ↳ Zero parameter explosion for blur inversion"},

        {"x": 10.1, "y": 0.5, "w": 5.2, "h": 5.2, "color": "#1E293B", "ec": "#10B981", 
         "title": "FiLM & PixelShuffle Upscaling Sub-Modules", 
         "content": "• DegradationEstimator Vector Network:\n  Conv2d(1,16,s=2)->ReLU->Conv2d(16,32,s=2)->AvgPool->FC(32)\n• FiLM Modulation Block:\n  Linear(32 -> 2C) -> Split (scale, shift) -> x * (1 + scale) + shift\n• SimpleGate Non-Linear Activation:\n  x.chunk(2, dim=1) -> x1 * x2 (Replaces GELU/ReLU)\n• Super-Resolution PixelShuffle Head:\n  Conv2d(C, C*scale^2, 3) -> PixelShuffle(scale) -> Conv2d(C, 1)\n  + Bicubic 2x Interpolation Base -> Clamp [0, 1]"}
    ]

    for m in micro_sections:
        rect = patches.FancyBboxPatch((m["x"], m["y"]), m["w"], m["h"], 
                                      boxstyle="round,pad=0.15", 
                                      facecolor=m["color"], edgecolor=m["ec"], linewidth=2)
        ax_micro.add_patch(rect)
        cx = m["x"] + 0.2
        cy = m["y"] + m["h"] - 0.5
        ax_micro.text(cx, cy, m["title"], ha='left', va='top', color='white', fontsize=11, fontweight='bold')
        ax_micro.text(cx, cy - 0.6, m["content"], ha='left', va='top', color='#CBD5E1', fontsize=9.2, linespacing=1.6)

    ax_micro.set_xlim(0.0, 15.6)
    ax_micro.set_ylim(0.0, 6.0)
    ax_micro.axis('off')

    plt.savefig('results/detailed_micro_module_pipeline.png', dpi=300, facecolor='#0B0F19', bbox_inches='tight')
    plt.close()
    print("Saved detailed micro-module architecture diagram: results/detailed_micro_module_pipeline.png")

if __name__ == "__main__":
    draw_micro_pipeline()
