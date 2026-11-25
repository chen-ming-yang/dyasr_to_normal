"""Frame-aligned normal→dysarthric mel converter.

Predicts a residual on top of the source log-mel, so a zero residual is the
original (already audible) normal mel. The network only has to learn S06's
spectral distortion.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class MelVC(nn.Module):
    def __init__(self, n_mels: int = 80, channels: int = 192, dropout: float = 0.1):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv1d(n_mels, channels, 7, padding=3),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )
        dilations = (1, 2, 4, 8, 1, 2, 4, 8, 1, 2)
        self.blocks = nn.Sequential(
            *[ResBlock(channels, d, dropout) for d in dilations]
        )
        self.out_conv = nn.Conv1d(channels, n_mels, 7, padding=3)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """src [B, 80, T] log-mel → converted log-mel, same shape."""
        h = self.blocks(self.in_conv(src))
        return src + self.out_conv(h)


def conversion_loss(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """pred/tgt [B, 80, T]. Speech frames (higher mel energy) weigh more.

    mask is [B, 1, T] with 1 on valid frames.
    """
    if mask is None:
        mask = pred.new_ones(pred.size(0), 1, pred.size(-1))
    with torch.no_grad():
        energy = tgt.exp().mean(dim=1, keepdim=True)
        scale = (energy * mask).sum(dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        weight = (energy / scale.clamp_min(1e-6)).clamp(0.3, 3.0) * mask

    # Upper bins carry consonants. Weight them more so L1 cannot ignore them.
    ramp = torch.linspace(1.0, 2.0, pred.size(1), device=pred.device, dtype=pred.dtype).view(1, -1, 1)
    l1 = ((pred - tgt).abs() * weight * ramp).sum() / (weight * ramp).sum().clamp_min(1.0)

    w_d = weight[..., 1:] * weight[..., :-1]
    d_pred = pred[..., 1:] - pred[..., :-1]
    d_tgt = tgt[..., 1:] - tgt[..., :-1]
    delta = ((d_pred - d_tgt).abs() * w_d).sum() / w_d.sum().clamp_min(1.0) / pred.size(1)

    f_pred = pred[:, 1:, :] - pred[:, :-1, :]
    f_tgt = tgt[:, 1:, :] - tgt[:, :-1, :]
    w_f = weight[:, :, :] 
    # weight is [B,1,T]; freq delta is [B,79,T]
    freq = ((f_pred - f_tgt).abs() * w_f).sum() / w_f.sum().clamp_min(1.0) / f_pred.size(1)

    multi = pred.new_zeros(())
    x, y, w = pred, tgt, weight
    scales = 0
    while x.size(-1) >= 4 and scales < 3:
        multi = multi + ((x - y).abs() * w).sum() / w.sum().clamp_min(1.0) / x.size(1)
        x = F.avg_pool1d(x, kernel_size=2, stride=2)
        y = F.avg_pool1d(y, kernel_size=2, stride=2)
        w = F.avg_pool1d(w, kernel_size=2, stride=2)
        scales += 1
    multi = multi / max(scales, 1)

    # Local contrast: L1 alone collapses to a flat conditional mean.
    def _local_std(x: torch.Tensor) -> torch.Tensor:
        mean = F.avg_pool1d(x, kernel_size=9, stride=1, padding=4)
        var = F.avg_pool1d((x - mean).pow(2), kernel_size=9, stride=1, padding=4)
        return var.clamp_min(1e-8).sqrt()

    contrast = ((_local_std(pred) - _local_std(tgt)).abs() * weight).sum() / weight.sum().clamp_min(1.0) / pred.size(1)

    pe = torch.log(pred.exp().mean(dim=1).clamp_min(1e-8))
    te = torch.log(tgt.exp().mean(dim=1).clamp_min(1e-8))
    energy = ((pe - te).abs() * mask.squeeze(1)).sum() / mask.sum().clamp_min(1.0)

    total = l1 + 0.5 * delta + 0.5 * freq + 0.25 * multi + 0.5 * contrast + 0.3 * energy
    return {
        "loss": total,
        "l1": l1.detach(),
        "delta": delta.detach(),
        "multi": multi.detach(),
    }
