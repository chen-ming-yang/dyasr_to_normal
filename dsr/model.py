"""Frame-synchronous NAR mel decoder: Conv1d stack + Transformer."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MelDecoder(nn.Module):
    def __init__(
        self,
        hubert_dim: int = 768,
        spk_dim: int = 192,
        d_model: int = 512,
        n_mels: int = 80,
        n_conv: int = 3,
        n_layers: int = 4,
        n_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        n_chars: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_chars = n_chars
        self.in_proj = nn.Linear(hubert_dim + spk_dim, d_model)
        convs = []
        for _ in range(n_conv):
            convs += [
                nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        self.conv = nn.Sequential(*convs)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, n_mels)
        self.char_head = nn.Linear(d_model, n_chars) if n_chars > 0 else None
        self._reset()

    def _reset(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        if self.char_head is not None:
            nn.init.xavier_uniform_(self.char_head.weight)
            nn.init.zeros_(self.char_head.bias)

    def forward(
        self,
        hubert: torch.Tensor,  # [B, T, H]
        spk: torch.Tensor,  # [B, S] or [B, 1, S]
        padding_mask: torch.Tensor | None = None,  # [B, T] True=pad
        time_mask_prob: float = 0.0,
        time_mask_width: int = 10,
        return_hidden: bool = False,
    ):
        b, t, _ = hubert.shape
        if spk.dim() == 2:
            spk = spk.unsqueeze(1).expand(-1, t, -1)
        x = torch.cat([hubert, spk], dim=-1)
        x = self.in_proj(x)

        if self.training and time_mask_prob > 0:
            x = _time_mask(x, padding_mask, time_mask_prob, time_mask_width)

        # conv over time
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        hidden = self.out_norm(x)
        mel = self.out_proj(hidden)
        if return_hidden:
            return mel, hidden
        return mel


def _time_mask(
    x: torch.Tensor,
    padding_mask: torch.Tensor | None,
    prob: float,
    width: int,
) -> torch.Tensor:
    """Light SpecAugment-style time masking on content frames."""
    b, t, d = x.shape
    if t < 2 or width < 1:
        return x
    out = x.clone()
    for i in range(b):
        if torch.rand(1).item() > prob:
            continue
        valid = t
        if padding_mask is not None:
            valid = int((~padding_mask[i]).sum().item())
        if valid <= 1:
            continue
        w = int(torch.randint(1, min(width, valid) + 1, (1,)).item())
        start = int(torch.randint(0, max(valid - w, 1), (1,)).item())
        out[i, start : start + w] = 0
    return out


def interpolate_feats(feats: torch.Tensor, target_len: int) -> torch.Tensor:
    """Linear interpolate [B, T, D] -> [B, target_len, D]."""
    if feats.size(1) == target_len:
        return feats
    x = feats.transpose(1, 2)  # [B, D, T]
    x = F.interpolate(x, size=target_len, mode="linear", align_corners=True)
    return x.transpose(1, 2)
