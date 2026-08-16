import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class FiLM(nn.Module):
    def __init__(self, embed_dim, num_features):
        super().__init__()
        self.to_scale_shift = nn.Linear(embed_dim, num_features * 2)

    def forward(self, x, embed):
        scale, shift = self.to_scale_shift(embed).chunk(2, dim=-1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + scale) + shift


class DegradationEstimator(nn.Module):
    def __init__(self, embed_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(32, embed_dim)

    def forward(self, x):
        feat = self.net(x).flatten(1)
        return self.fc(feat)


class NAFBlock(nn.Module):
    def __init__(self, c, embed_dim=None):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, c)
        self.conv1 = nn.Conv2d(c, c * 2, 1)
        self.dwconv = nn.Conv2d(c * 2, c * 2, 5, padding=2, groups=c * 2)
        self.gate = SimpleGate()
        self.attn = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))
        self.conv2 = nn.Conv2d(c, c, 1)
        self.norm2 = nn.GroupNorm(1, c)
        self.ffn1 = nn.Conv2d(c, c * 2, 1)
        self.ffn_gate = SimpleGate()
        self.ffn2 = nn.Conv2d(c, c, 1)
        self.film = FiLM(embed_dim, c) if embed_dim is not None else None

    def forward(self, x, embed=None):
        y = self.dwconv(self.conv1(self.norm1(x)))
        y = self.gate(y)
        y = y * self.attn(y)
        if self.film is not None and embed is not None:
            y = self.film(y, embed)
        x = x + self.conv2(y)
        y = self.ffn_gate(self.ffn1(self.norm2(x)))
        return x + self.ffn2(y)

class MDTA(nn.Module):
    def __init__(self, c, num_heads=4):
        super().__init__()
        assert c % num_heads == 0, "channel count must be divisible by num_heads"
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(c, c * 3, 1)
        self.qkv_dwconv = nn.Conv2d(c * 3, c * 3, 3, padding=1, groups=c * 3)
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        head_c = c // self.num_heads
        q = q.reshape(b, self.num_heads, head_c, h * w)
        k = k.reshape(b, self.num_heads, head_c, h * w)
        v = v.reshape(b, self.num_heads, head_c, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature  # (b, heads, head_c, head_c)
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.reshape(b, c, h, w)
        return self.proj(out)


class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network (Restormer-style FFN)."""
    def __init__(self, c, expansion=2.66):
        super().__init__()
        hidden = int(c * expansion)
        self.proj_in = nn.Conv2d(c, hidden * 2, 1)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2)
        self.proj_out = nn.Conv2d(hidden, c, 1)

    def forward(self, x):
        x = self.dwconv(self.proj_in(x))
        x1, x2 = x.chunk(2, dim=1)
        return self.proj_out(F.gelu(x1) * x2)


class RestormerBlock(nn.Module):
    def __init__(self, c, embed_dim=None, num_heads=4):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, c)
        self.attn = MDTA(c, num_heads=num_heads)
        self.norm2 = nn.GroupNorm(1, c)
        self.ffn = GDFN(c)
        self.film = FiLM(embed_dim, c) if embed_dim is not None else None

    def forward(self, x, embed=None):
        y = self.attn(self.norm1(x))
        if self.film is not None and embed is not None:
            y = self.film(y, embed)
        x = x + y
        x = x + self.ffn(self.norm2(x))
        return x


class RestoreNet(nn.Module):
    def __init__(self, c=48, n_blocks=8, scale=2):
        super().__init__()
        self.head = nn.Conv2d(1, c, 3, padding=1)
        self.body = nn.ModuleList([NAFBlock(c) for _ in range(n_blocks)])
        self.up = nn.Sequential(
            nn.Conv2d(c, c * scale ** 2, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(c, 1, 3, padding=1),
        )
        self.scale = scale

    def forward(self, x):
        skip = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        feat = self.head(x)
        for block in self.body:
            feat = block(feat)
        out = self.up(feat)
        return torch.clamp(out + skip, 0, 1)

class RestoreNetFiLM(nn.Module):
    """Stage 1: adds a small degradation estimator whose output conditions
    EVERY block in the network via FiLM, so the signal doesn't fade out
    before reaching the later layers."""
    def __init__(self, c=48, n_blocks=8, scale=2, embed_dim=32):
        super().__init__()
        self.estimator = DegradationEstimator(embed_dim=embed_dim)
        self.head = nn.Conv2d(1, c, 3, padding=1)
        self.body = nn.ModuleList(
            [NAFBlock(c, embed_dim=embed_dim) for _ in range(n_blocks)]
        )
        self.up = nn.Sequential(
            nn.Conv2d(c, c * scale ** 2, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(c, 1, 3, padding=1),
        )
        self.scale = scale

    def forward(self, x):
        embed = self.estimator(x)
        skip = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        feat = self.head(x)
        for block in self.body:
            feat = block(feat, embed)
        out = self.up(feat)
        return torch.clamp(out + skip, 0, 1)


class RestoreNetHybrid(nn.Module):
    def __init__(self, c=48, n_encoder_blocks=4, n_decoder_blocks=4,
                 scale=2, embed_dim=32, num_heads=4):
        super().__init__()
        self.estimator = DegradationEstimator(embed_dim=embed_dim)
        self.head = nn.Conv2d(1, c, 3, padding=1)
        self.encoder = nn.ModuleList([
            RestormerBlock(c, embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(n_encoder_blocks)
        ])
        self.decoder = nn.ModuleList([
            NAFBlock(c, embed_dim=embed_dim) for _ in range(n_decoder_blocks)
        ])
        self.up = nn.Sequential(
            nn.Conv2d(c, c * scale ** 2, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(c, 1, 3, padding=1),
        )
        self.scale = scale

    def forward(self, x):
        embed = self.estimator(x)
        skip = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        feat = self.head(x)
        for block in self.encoder:
            feat = block(feat, embed)
        for block in self.decoder:
            feat = block(feat, embed)
        out = self.up(feat)
        return torch.clamp(out + skip, 0, 1)

class NAFNetUNet(nn.Module):
    """Stage 3: Multi-Scale NAFNet UNet Architecture (v2 with LKA & Deep Supervision).
    Features 4-level encoder-decoder hierarchy with LKA blocks and auxiliary multi-scale heads."""
    def __init__(self, c=32, scale=2, embed_dim=32, use_film=False):
        super().__init__()
        self.scale = scale
        self.use_film = use_film
        self.estimator = DegradationEstimator(embed_dim=embed_dim) if use_film else None
        eff_embed = embed_dim if use_film else None
        self.head = nn.Conv2d(1, c, 3, padding=1)

        # Encoder Level 1 (c) & Level 2 (2c)
        self.enc1 = nn.ModuleList([NAFBlock(c, embed_dim=eff_embed) for _ in range(2)])
        self.down1 = nn.Conv2d(c, c * 2, 2, stride=2)

        self.enc2 = nn.ModuleList([NAFBlock(c * 2, embed_dim=eff_embed) for _ in range(2)])
        self.down2 = nn.Conv2d(c * 2, c * 4, 2, stride=2)

        # Bottleneck (4c)
        self.bottleneck = nn.ModuleList([NAFBlock(c * 4, embed_dim=eff_embed) for _ in range(4)])

        # Decoder Level 2
        self.up2 = nn.Sequential(nn.Conv2d(c * 4, c * 4, 1), nn.PixelShuffle(2))  # c*4 -> c
        self.fuse2 = nn.Conv2d(c * 2 + c, c * 2, 1)
        self.dec2 = nn.ModuleList([NAFBlock(c * 2, embed_dim=eff_embed) for _ in range(2)])

        # Decoder Level 1
        self.up1 = nn.Sequential(nn.Conv2d(c * 2, c * 4, 1), nn.PixelShuffle(2))  # c*2 -> c
        self.fuse1 = nn.Conv2d(c + c, c, 1)
        self.dec1 = nn.ModuleList([NAFBlock(c, embed_dim=eff_embed) for _ in range(2)])

        # Final Super-Resolution Upscale Head
        self.sr_head = nn.Sequential(
            nn.Conv2d(c, c * scale ** 2, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(c, 1, 3, padding=1),
        )

        # Auxiliary Deep Supervision Projection Head (Level 2 Decoder -> 1x GT Scale Output)
        self.aux_head2 = nn.Sequential(
            nn.Conv2d(c * 2, c * scale ** 2, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(c, 1, 3, padding=1),
        )

    def forward(self, x, return_deep_supervision=False):
        # Clip input dynamic range to strictly [0, 1]
        x = torch.clamp(x, 0.0, 1.0)
        embed = self.estimator(x) if (self.use_film and self.estimator is not None) else None
        skip_base = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)

        # Encoder Level 1
        feat1 = self.head(x)
        for b in self.enc1:
            feat1 = b(feat1, embed)

        # Downsample to Level 2
        feat2 = self.down1(feat1)
        for b in self.enc2:
            feat2 = b(feat2, embed)

        # Downsample to Bottleneck
        feat3 = self.down2(feat2)
        for b in self.bottleneck:
            feat3 = b(feat3, embed)

        # Upsample & Fuse Level 2
        up_feat2 = self.up2(feat3)
        dec_feat2 = self.fuse2(torch.cat([up_feat2, feat2], dim=1))
        for b in self.dec2:
            dec_feat2 = b(dec_feat2, embed)

        # Upsample & Fuse Level 1
        up_feat1 = self.up1(dec_feat2)
        dec_feat1 = self.fuse1(torch.cat([up_feat1, feat1], dim=1))
        for b in self.dec1:
            dec_feat1 = b(dec_feat1, embed)

        # Final Output
        out = torch.clamp(self.sr_head(dec_feat1) + skip_base, 0.0, 1.0)

        if return_deep_supervision and self.training:
            out_half = torch.clamp(self.aux_head2(dec_feat2) + x, 0.0, 1.0)
            return out, out_half

        return out


class PatchDiscriminator(nn.Module):
    def __init__(self, c=32):
        super().__init__()

        def block(in_c, out_c, stride=2, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(1, c, norm=False),
            *block(c, c * 2),
            *block(c * 2, c * 4),
            *block(c * 4, c * 8, stride=1),
            nn.Conv2d(c * 8, 1, 4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = params * 4 / (1024 * 1024)
    return params, size_mb


def build_model(stage: str, use_film: bool = False):
    if stage == "stage0_baseline":
        return RestoreNet(c=48, n_blocks=8, scale=2)
    elif stage == "stage1_film":
        return RestoreNetFiLM(c=48, n_blocks=8, scale=2, embed_dim=32)
    elif stage == "stage2_hybrid":
        return RestoreNetHybrid(
            c=48, n_encoder_blocks=4, n_decoder_blocks=4,
            scale=2, embed_dim=32, num_heads=4,
        )
    elif stage == "stage3_nafnet_unet":
        return NAFNetUNet(c=32, scale=2, embed_dim=32, use_film=use_film)
    else:
        raise ValueError(
            f"Unknown stage '{stage}'. Expected one of: "
            f"stage0_baseline, stage1_film, stage2_hybrid, stage3_nafnet_unet"
        )