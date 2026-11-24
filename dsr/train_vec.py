#!/usr/bin/env python3
"""Train a small head to match normal-audio code vectors. Char clips, then sentences."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, DataLoader, Dataset

CACHE = Path("/cmy/cmy/dysa/runs/dsr/code_cache_v4")
OUT = Path("/cmy/cmy/dysa/runs/dsr/train_codes_v4")


class VecSet(Dataset):
    def __init__(self, split: str, kind: str):
        self.items = []
        for line in open(CACHE / "index.jsonl"):
            rec = json.loads(line)
            if rec["split"] != split:
                continue
            data = np.load(rec["path"])
            feat = data["feat"].astype(np.float32)
            target = data["target"].astype(np.float32)
            speech = data["speech"].astype(np.float32)
            if kind == "sent":
                self.items.append((feat, target, speech))
            else:
                for a, b in data["spans"]:
                    if b - a < 2:
                        continue
                    self.items.append((feat[a:b], target[a:b], speech[a:b]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        feat, target, speech = self.items[idx]
        return (
            torch.from_numpy(feat),
            torch.from_numpy(target),
            torch.from_numpy(speech),
        )


def collate(batch):
    feat = pad_sequence([b[0] for b in batch], batch_first=True)
    target = pad_sequence([b[1] for b in batch], batch_first=True)
    speech = pad_sequence([b[2] for b in batch], batch_first=True)
    return feat, target, speech


class VecHead(nn.Module):
    def __init__(self, d_in: int = 1024, d_h: int = 512, d_out: int = 8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_in, d_h, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(d_h, d_h, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(d_h, d_h, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.proj = nn.Linear(d_h, d_out)

    def forward(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return F.normalize(self.proj(h), dim=-1)


def run_epoch(loader, head, opt, device, train: bool):
    head.train(train)
    dot_sum = 0.0
    n_sum = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for feat, target, speech in loader:
            feat = feat.to(device)
            target = F.normalize(target.to(device), dim=-1)
            speech = speech.to(device)
            pred = head(feat)
            cos = (pred * target).sum(dim=-1)
            denom = speech.sum().clamp_min(1.0)
            loss = ((1.0 - cos) * speech).sum() / denom
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            dot_sum += float((cos * speech).sum().item())
            n_sum += float(speech.sum().item())
    return dot_sum / max(n_sum, 1.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    train_char = VecSet("train", "char")
    train_sent = VecSet("train", "sent")
    dev_sent = VecSet("dev", "sent")
    print(f"train chars {len(train_char)} sents {len(train_sent)} dev sents {len(dev_sent)}")
    head = VecHead().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=32, shuffle=shuffle, collate_fn=collate)

    dev_loader = loader(dev_sent, False)
    history = []
    best = -1.0
    bad = 0
    phases = [
        ("char", train_char, 5),
        ("joint", ConcatDataset([train_char, train_sent]), 20),
    ]
    for phase, ds, epochs in phases:
        train_loader = loader(ds, True)
        for ep in range(1, epochs + 1):
            t0 = time.time()
            tr = run_epoch(train_loader, head, opt, device, True)
            va = run_epoch(dev_loader, head, None, device, False)
            row = {"phase": phase, "epoch": ep, "train_cos": tr, "dev_cos": va, "sec": round(time.time() - t0, 1)}
            history.append(row)
            print(f"{phase} {ep:02d}  train_cos={tr:.3f}  dev_cos={va:.3f}  ({row['sec']}s)", flush=True)
            ckpt = {"head": head.state_dict(), "history": history, "best_dev_cos": best}
            torch.save(ckpt, OUT / "last.pt")
            if phase == "joint" and va > best + 1e-4:
                best = va
                bad = 0
                ckpt["best_dev_cos"] = best
                torch.save(ckpt, OUT / "best.pt")
                print(f"  ^ best dev cos {best:.3f}", flush=True)
            elif phase == "joint":
                bad += 1
                if bad >= 5:
                    print("early stop")
                    break
    with open(OUT / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"done best_dev_cos={best:.3f}")


if __name__ == "__main__":
    main()
