"""Multi-resolution STFT loss (HiFi-GAN style) for natural-sounding conversion."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def stft_mag(x: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
    return torch.stft(
        x.squeeze(1),
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window=torch.hann_window(win, device=x.device),
        center=True,
        pad_mode="reflect",
        return_complex=True,
    ).abs()


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, resolutions=((2048, 512, 2048), (1024, 256, 1024), (512, 128, 512))):
        super().__init__()
        self.resolutions = resolutions

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """pred/target [B, 1, T] waveforms."""
        sc_total = pred.new_zeros(())
        mag_total = pred.new_zeros(())
        for n_fft, hop, win in self.resolutions:
            p = stft_mag(pred, n_fft, hop, win)
            t = stft_mag(target, n_fft, hop, win)
            sc = (p - t).norm(dim=(1, 2)) / (t.norm(dim=(1, 2)) + 1e-8)
            sc_total = sc_total + sc.mean()
            mag = (torch.log(p + 1e-8) - torch.log(t + 1e-8)).abs().mean()
            mag_total = mag_total + mag
        n = len(self.resolutions)
        loss = sc_total / n + mag_total / n
        return loss, {"sc": sc_total.detach() / n, "mag": mag_total.detach() / n}
