import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_pipeline():
    fig, ax = plt.subplots(figsize=(16, 7), dpi=300)
    fig.patch.set_facecolor('#0B0F19')
    ax.set_facecolor('#0B0F19')

    # Title
    ax.text(8.0, 6.3, "Stage 3 NAFNet-UNet v2: End-to-End System Architecture Pipeline", 
            ha='center', va='center', color='white', fontsize=16, fontweight='bold')

    boxes = [
        {"x": 0.5, "y": 3.0, "w": 1.6, "h": 2.2, "color": "#1E293B", "ec": "#38BDF8", 
         "title": "Raw Input\n(NoisyLR)", "sub": "(1 x 128 x 128)\nSpeckle + Blur + Noise"},

        {"x": 2.5, "y": 3.0, "w": 1.6, "h": 2.2, "color": "#1E293B", "ec": "#00F5D4", 
         "title": "Dynamic Range\nSafeguard", "sub": "torch.clamp(x, 0, 1)\nDynamic Scaling"},

        {"x": 4.5, "y": 3.8, "w": 2.2, "h": 1.8, "color": "#1E1E38", "ec": "#818CF8", 
         "title": "4-Level UNet Encoder", "sub": "Level 1: 32c | Level 2: 64c\nLKA Blocks (5x5 + 7x7 DW)"},

        {"x": 7.1, "y": 3.8, "w": 1.8, "h": 1.8, "color": "#2A1E38", "ec": "#F472B6", 
         "title": "Bottleneck", "sub": "4x LKA Blocks (128c)\n23px Receptive Field"},

        {"x": 9.3, "y": 3.8, "w": 2.2, "h": 1.8, "color": "#1E1E38", "ec": "#818CF8", 
         "title": "Multi-Scale Decoder", "sub": "PixelShuffle Upsampling\nAux Deep Supervision Head"},

        {"x": 11.9, "y": 3.0, "w": 1.6, "h": 2.2, "color": "#1E293B", "ec": "#34D399", 
         "title": "Residual Head\n& Skip Base", "sub": "Bicubic 2x Skip Addition\nFinal Clamp [0, 1]"},

        {"x": 13.9, "y": 3.0, "w": 1.6, "h": 2.2, "color": "#064E3B", "ec": "#10B981", 
         "title": "Restored Output\n(Ground Truth)", "sub": "(1 x 256 x 256)\nSSIM: 0.7492 | PSNR: 27.37dB"}
    ]

    # Draw boxes
    for b in boxes:
        rect = patches.FancyBboxPatch((b["x"], b["y"]), b["w"], b["h"], 
                                      boxstyle="round,pad=0.15", 
                                      facecolor=b["color"], edgecolor=b["ec"], linewidth=2)
        ax.add_patch(rect)
        cx = b["x"] + b["w"]/2
        cy = b["y"] + b["h"]/2 + 0.3
        ax.text(cx, cy, b["title"], ha='center', va='center', color='white', fontsize=11, fontweight='bold')
        ax.text(cx, cy - 0.6, b["sub"], ha='center', va='center', color='#94A3B8', fontsize=8.5)

    # Arrows
    arrow_args = dict(arrowstyle="->,head_length=0.4,head_width=0.3", color="#38BDF8", lw=2.5)
    
    ax.annotate("", xy=(2.45, 4.1), xytext=(2.15, 4.1), arrowprops=arrow_args)
    ax.annotate("", xy=(4.45, 4.7), xytext=(4.15, 4.1), arrowprops=arrow_args)
    ax.annotate("", xy=(7.05, 4.7), xytext=(6.75, 4.7), arrowprops=arrow_args)
    ax.annotate("", xy=(9.25, 4.7), xytext=(8.95, 4.7), arrowprops=arrow_args)
    ax.annotate("", xy=(11.85, 4.1), xytext=(11.55, 4.7), arrowprops=arrow_args)
    ax.annotate("", xy=(13.85, 4.1), xytext=(13.55, 4.1), arrowprops=arrow_args)

    # Bicubic Long Skip Curve
    ax.annotate("", xy=(12.0, 3.2), xytext=(1.3, 3.2), 
                arrowprops=dict(arrowstyle="->,head_length=0.4,head_width=0.3", 
                                color="#F59E0B", lw=2.0, connectionstyle="arc3,rad=-0.35"))
    ax.text(6.6, 1.8, "Long Skip Connection (Bicubic 2x Interpolation)", 
            ha='center', va='center', color='#F59E0B', fontsize=10, fontweight='bold')

    ax.set_xlim(-0.2, 16.0)
    ax.set_ylim(1.0, 7.0)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('results/system_pipeline_architecture.png', dpi=300, facecolor='#0B0F19', bbox_inches='tight')
    plt.close()
    print("Saved pipeline architecture diagram: results/system_pipeline_architecture.png")

if __name__ == "__main__":
    draw_pipeline()
