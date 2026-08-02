import torch
import torch.nn.functional as F

def charbonnier_loss(pred, gt, eps=1e-6):
    return torch.mean(torch.sqrt((pred - gt) ** 2 + eps ** 2))

def _gaussian_window(window_size=11, sigma=1.5, device="cpu", dtype=torch.float32):
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.outer(g).unsqueeze(0).unsqueeze(0)
    return window

def ssim(pred, gt, window_size=11, data_range=1.0):
    """Structural Similarity Index, computed directly on GPU tensors so it
    can be used both as a loss term and as a validation metric, with no
    numpy round-trip needed."""
    window = _gaussian_window(window_size, device=pred.device, dtype=pred.dtype)
    pad = window_size // 2

    mu_p = F.conv2d(pred, window, padding=pad)
    mu_g = F.conv2d(gt, window, padding=pad)

    mu_p_sq, mu_g_sq, mu_pg = mu_p ** 2, mu_g ** 2, mu_p * mu_g

    sigma_p_sq = F.conv2d(pred * pred, window, padding=pad) - mu_p_sq
    sigma_g_sq = F.conv2d(gt * gt, window, padding=pad) - mu_g_sq
    sigma_pg = F.conv2d(pred * gt, window, padding=pad) - mu_pg

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu_pg + c1) * (2 * sigma_pg + c2)) / (
        (mu_p_sq + mu_g_sq + c1) * (sigma_p_sq + sigma_g_sq + c2)
    )
    return ssim_map.mean()

def psnr(pred, gt, data_range=1.0):
    mse = torch.mean((pred - gt) ** 2)
    if mse.item() == 0:
        return torch.tensor(100.0)
    return 10 * torch.log10((data_range ** 2) / mse)

_SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
_SOBEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

def edge_loss(pred, gt):
    kx = _SOBEL_X.view(1, 1, 3, 3).to(device=pred.device, dtype=pred.dtype)
    ky = _SOBEL_Y.view(1, 1, 3, 3).to(device=pred.device, dtype=pred.dtype)

    pred_gx = F.conv2d(pred, kx, padding=1)
    pred_gy = F.conv2d(pred, ky, padding=1)
    gt_gx = F.conv2d(gt, kx, padding=1)
    gt_gy = F.conv2d(gt, ky, padding=1)

    return torch.mean(torch.abs(pred_gx - gt_gx) + torch.abs(pred_gy - gt_gy))

def restoration_loss(pred, gt, w_ssim=0.2, w_edge=0.1):
    l_charb = charbonnier_loss(pred, gt)
    l_ssim = 1 - ssim(pred, gt)
    l_edge = edge_loss(pred, gt)
    total = l_charb + w_ssim * l_ssim + w_edge * l_edge
    parts = {
        "charbonnier": l_charb.item(),
        "ssim_loss": l_ssim.item(),
        "edge_loss": l_edge.item(),
    }
    return total, parts