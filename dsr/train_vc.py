#!/usr/bin/env python3
"""Train frame-aligned normal→dysarthric mel VC on vc_cache pairs."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from vc_model import MelVC, conversion_loss

ROOT = Path("/cmy/cmy/dysa")
CACHE = ROOT / "runs/dsr/vc_cache"


class PairDataset(Dataset):
    def __init__(self, cache: Path, split: str, crop: int, train: bool):
        self.files = sorted((cache / split).glob("*.npz"))
        self.crop = crop
        self.train = train
        if not self.files:
            raise FileNotFoundError(cache / split)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        z = np.load(self.files[idx])
        src = torch.from_numpy(z["src"])  # [T, 80]
        tgt = torch.from_numpy(z["tgt"])
        n = src.size(0)
        valid = n
        if self.train and n > self.crop:
            start = random.randint(0, n - self.crop)
            src = src[start : start + self.crop]
            tgt = tgt[start : start + self.crop]
            valid = self.crop
        elif self.train and n < self.crop:
            pad = self.crop - n
            src = torch.nn.functional.pad(src, (0, 0, 0, pad))
            tgt = torch.nn.functional.pad(tgt, (0, 0, 0, pad))
            valid = n
        return src.transpose(0, 1).contiguous(), tgt.transpose(0, 1).contiguous(), valid


def collate(batch):
    srcs, tgts, lengths = zip(*batch)
    max_t = max(s.size(-1) for s in srcs)
    b = len(srcs)
    src = srcs[0].new_zeros(b, 80, max_t)
    tgt = tgts[0].new_zeros(b, 80, max_t)
    mask = torch.zeros(b, 1, max_t)
    for i, (s, t, n) in enumerate(zip(srcs, tgts, lengths)):
        src[i, :, : s.size(-1)] = s
        tgt[i, :, : t.size(-1)] = t
        mask[i, :, : int(n)] = 1.0
    return src, tgt, mask


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    n = 0
    ident = 0.0
    moved = 0.0
    for src, tgt, mask in loader:
        src = src.to(device)
        tgt = tgt.to(device)
        mask = mask.to(device)
        pred = model(src)
        denom = mask.sum() * src.size(1)
        l1 = ((pred - tgt).abs() * mask).sum() / denom.clamp_min(1.0)
        base = ((src - tgt).abs() * mask).sum() / denom.clamp_min(1.0)
        shift = ((pred - src).abs() * mask).sum() / denom.clamp_min(1.0)
        bs = src.size(0)
        total += l1.item() * bs
        ident += base.item() * bs
        moved += shift.item() * bs
        n += bs
    return {"l1": total / n, "identity_l1": ident / n, "residual_l1": moved / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/dsr/train_vc")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--crop", type=int, default=320)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--channels", type=int, default=192)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    train_ds = PairDataset(args.cache, "train", args.crop, train=True)
    dev_ds = PairDataset(args.cache, "dev", args.crop, train=False)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=4,
        collate_fn=collate, pin_memory=True, drop_last=True,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=8, shuffle=False, num_workers=2,
        collate_fn=collate, pin_memory=True,
    )

    model = MelVC(channels=args.channels, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    log_path = args.out / "train_log.jsonl"
    best = 1e9
    bad = 0
    best_path = args.out / "best.pt"
    print(f"device={device} params={n_params/1e6:.2f}M train={len(train_ds)} dev={len(dev_ds)}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        steps = 0
        for src, tgt, mask in train_loader:
            src = src.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            pred = model(src)
            losses = conversion_loss(pred, tgt, mask)
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += losses["loss"].item()
            steps += 1
        sched.step()
        dev = evaluate(model, dev_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": running / max(steps, 1),
            "dev_l1": dev["l1"],
            "identity_l1": dev["identity_l1"],
            "residual_l1": dev["residual_l1"],
            "lr": sched.get_last_lr()[0],
            "sec": round(time.time() - t0, 2),
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"epoch {epoch:03d} train {row['train_loss']:.4f} "
            f"dev_l1 {dev['l1']:.4f} identity {dev['identity_l1']:.4f} "
            f"residual {dev['residual_l1']:.4f} {row['sec']:.1f}s",
            flush=True,
        )
        torch.save(
            {"model": model.state_dict(), "args": vars(args), "epoch": epoch, "dev": dev},
            args.out / "last.pt",
        )
        if dev["l1"] < best - 1e-4:
            best = dev["l1"]
            bad = 0
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "epoch": epoch, "dev": dev},
                best_path,
            )
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {epoch}, best dev_l1 {best:.4f}", flush=True)
                break
    print(f"best dev_l1 {best:.4f} -> {best_path}", flush=True)


if __name__ == "__main__":
    main()
