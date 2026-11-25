"""Small adapter on frozen Qwen3-TTS 12 Hz codes.

Character onsets slice the normal codes and the dysarthric codes into pairs.
Inside a character the normal codes are lined up with the spoken part; frames
after that are the pause, which the adapter has to predict as its own codes.
The waveform decoder stays frozen.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_Q = 16
N_CODE = 2048
SR = 24000
HOP = 1920  # 12.5 Hz


def sec_to_code(t: float) -> int:
    return int(round(t * SR / HOP))


class CodecAdapter(nn.Module):
    def __init__(self, d: int = 256, layers: int = 4, n_q: int = N_Q, n_code: int = N_CODE):
        super().__init__()
        self.n_q = n_q
        self.emb = nn.ModuleList([nn.Embedding(n_code, d) for _ in range(n_q)])
        self.pause = nn.Parameter(torch.zeros(d))
        self.in_proj = nn.Linear(d + 2, d)
        block = nn.TransformerEncoderLayer(
            d_model=d, nhead=4, dim_feedforward=d * 4, dropout=0.1,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.enc = nn.TransformerEncoder(block, num_layers=layers)
        self.heads = nn.ModuleList([nn.Linear(d, n_code) for _ in range(n_q)])

    def forward(
        self,
        src: torch.Tensor,
        src_index: torch.Tensor,
        progress: torch.Tensor,
        is_pause: torch.Tensor,
        pad: torch.Tensor,
    ) -> list[torch.Tensor]:
        """src [B, Ts, Q], src_index/progress/is_pause/pad [B, Tt]."""
        e = 0
        for q in range(self.n_q):
            e = e + self.emb[q](src[:, :, q].clamp(0, N_CODE - 1))
        idx = src_index.clamp(min=0).unsqueeze(-1).expand(-1, -1, e.size(-1))
        gathered = torch.gather(e, 1, idx)
        gathered = torch.where(is_pause.unsqueeze(-1), self.pause.view(1, 1, -1), gathered)
        x = torch.cat([gathered, progress.unsqueeze(-1), is_pause.unsqueeze(-1).float()], dim=-1)
        h = self.enc(self.in_proj(x), src_key_padding_mask=~pad)
        return [head(h) for head in self.heads]


def adapter_loss(logits: list[torch.Tensor], tgt: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    """tgt [B, Tt, Q], pad [B, Tt] True on valid frames."""
    loss = tgt.new_zeros(())
    weight = pad.float()
    denom = weight.sum().clamp_min(1.0)
    for q, logit in enumerate(logits):
        ce = F.cross_entropy(logit.transpose(1, 2), tgt[:, :, q].clamp(0, N_CODE - 1), reduction="none")
        # The first codebook carries the phonetic target. Later books carry detail.
        scale = 2.0 if q == 0 else 0.5
        loss = loss + scale * (ce * weight).sum() / denom
    return loss / (2.0 + 0.5 * (len(logits) - 1))
