#!/usr/bin/env python3
"""Direction 1: per-character semantic-vector head (char pooled).

Each character span pools the dysarthric layer-17 features into ONE vector,
trained to match the pooled normal-audio code vector of that character.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

CACHE = Path("/cmy/cmy/dysa/runs/dsr/code_cache_v4")
OUT = Path("/cmy/cmy/dysa/runs/dsr/train_char_v5")


class CharSet(Dataset):
    """One item = one character. feat [L,1024], target_vec [8]."""

    def __init__(self, split: str, min_frames: int = 3):
        self.items = []
        for line in open(CACHE / "index.jsonl"):
            rec = json.loads(line)
            if rec["split"] != split:
                continue
            data = np.load(rec["path"])
            feat = data["feat"].astype(np.float32)
            target = data["target"].astype(np.float32)
            for a, b in data["spans"]:
                if b - a < min_frames:
                    continue
                tv = target[a:b].mean(axis=0)
                nrm = np.linalg.norm(tv)
                if nrm > 1e-6:
                    tv = tv / nrm
                self.items.append((feat[a:b], tv))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        feat, tv = self.items[idx]
        return torch.from_numpy(feat), torch.from_numpy(tv)


def collate(batch):
    feats = [b[0] for b in batch]
    targets = torch.stack([b[1] for b in batch], dim=0)
    max_len = max(f.shape[0] for f in feats)
    B = len(feats)
    padded = torch.zeros(B, max_len, feats[0].shape[1])
    mask = torch.ones(B, max_len, dtype=torch.bool)
    for i, f in enumerate(feats):
        padded[i, : f.shape[0]] = f
        mask[i, f.shape[0] :] = False
    return padded, mask, targets


class CharPoolHead(nn.Module):
    def __init__(self, d_in: int = 1024, d_h: int = 512, d_out: int = 8):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(d_in, 1))
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_h),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_h, d_h),
            nn.GELU(),
            nn.Linear(d_h, d_out),
        )

    def forward(self, x, mask):
        # attention pooling over valid frames
        w = self.attn(x).masked_fill(~mask.unsqueeze(-1), -1e9)
        w = torch.softmax(w, dim=1)
        pooled = (x * w).sum(dim=1)
        return F.normalize(self.mlp(pooled), dim=-1)


def run_epoch(loader, head, opt, device, train: bool):
    head.train(train)
    dot_sum = 0.0
    n_sum = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for feat, mask, target in loader:
            feat = feat.to(device)
            mask = mask.to(device)
            target = F.normalize(target.to(device), dim=-1)
            pred = head(feat, mask)
            cos = (pred * target).sum(dim=-1)
            loss = (1.0 - cos).mean()
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            dot_sum += float(cos.sum().item())
            n_sum += len(cos)
    return dot_sum / max(n_sum, 1.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    train_ds = CharSet("train")
    dev_ds = CharSet("dev")
    print(f"train chars {len(train_ds)} dev chars {len(dev_ds)}")
    head = CharPoolHead().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=64, shuffle=shuffle, collate_fn=collate)

    train_loader = loader(train_ds, True)
    dev_loader = loader(dev_ds, False)
    history = []
    best = -1.0
    bad = 0
    for ep in range(1, 60):
        t0 = time.time()
        tr = run_epoch(train_loader, head, opt, device, True)
        va = run_epoch(dev_loader, head, None, device, False)
        row = {"epoch": ep, "train_cos": tr, "dev_cos": va, "sec": round(time.time() - t0, 1)}
        history.append(row)
        print(f"epoch {ep:02d}  train_cos={tr:.3f}  dev_cos={va:.3f}  ({row['sec']}s)", flush=True)
        ckpt = {"head": head.state_dict(), "history": history, "best_dev_cos": best}
        torch.save(ckpt, OUT / "last.pt")
        if va > best + 1e-4:
            best = va
            bad = 0
            ckpt["best_dev_cos"] = best
            torch.save(ckpt, OUT / "best.pt")
            print(f"  ^ best dev cos {best:.3f}", flush=True)
        else:
            bad += 1
            if bad >= 8:
                print("early stop")
                break
    with open(OUT / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"done best_dev_cos={best:.3f}")


if __name__ == "__main__":
    main()
