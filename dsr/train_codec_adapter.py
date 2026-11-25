#!/usr/bin/env python3
"""Train the codec adapter on character-aligned normal/dysarthric code pairs."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from codec_adapter import N_Q, CodecAdapter, adapter_loss

ROOT = Path("/cmy/cmy/dysa")
CACHE = ROOT / "runs/dsr/codec_pairs"
OUT = ROOT / "runs/dsr/train_codec_adapter"
MAX_T = 160


def char_example(src: np.ndarray, tgt: np.ndarray, span: np.ndarray):
    sa, sb, ta, tb, speech_n = [int(x) for x in span]
    src_seg = src[sa:sb]
    tgt_seg = tgt[ta:tb]
    if len(src_seg) < 1 or len(tgt_seg) < 1:
        return None
    if len(tgt_seg) > MAX_T:
        tgt_seg = tgt_seg[:MAX_T]
        speech_n = min(speech_n, MAX_T)
    speech_n = int(np.clip(speech_n, 1, len(tgt_seg)))
    index = np.zeros(len(tgt_seg), dtype=np.int64)
    progress = np.zeros(len(tgt_seg), dtype=np.float32)
    is_pause = np.zeros(len(tgt_seg), dtype=np.bool_)
    if speech_n > 0:
        index[:speech_n] = np.linspace(0, len(src_seg) - 1, speech_n).round().astype(np.int64)
        progress[:speech_n] = np.linspace(0, 1, speech_n, dtype=np.float32)
    if speech_n < len(tgt_seg):
        index[speech_n:] = len(src_seg) - 1
        progress[speech_n:] = 1.0
        is_pause[speech_n:] = True
    return src_seg.astype(np.int64), tgt_seg.astype(np.int64), index, progress, is_pause


def load_split(split: str) -> list[dict]:
    items = []
    for path in sorted((CACHE / split).glob("*.npz")):
        z = np.load(path)
        src, tgt, spans = z["src"], z["tgt"], z["spans"]
        for span in spans:
            ex = char_example(src, tgt, span)
            if ex is not None:
                items.append(ex)
    return items


def collate(batch, device):
    ts = max(b[0].shape[0] for b in batch)
    tt = max(b[1].shape[0] for b in batch)
    bsz = len(batch)
    src = torch.zeros(bsz, ts, N_Q, dtype=torch.long, device=device)
    tgt = torch.zeros(bsz, tt, N_Q, dtype=torch.long, device=device)
    index = torch.zeros(bsz, tt, dtype=torch.long, device=device)
    progress = torch.zeros(bsz, tt, device=device)
    is_pause = torch.zeros(bsz, tt, dtype=torch.bool, device=device)
    pad = torch.zeros(bsz, tt, dtype=torch.bool, device=device)
    for i, (s, t, idx, prog, pause) in enumerate(batch):
        src[i, : len(s)] = torch.from_numpy(s)
        tgt[i, : len(t)] = torch.from_numpy(t)
        index[i, : len(t)] = torch.from_numpy(idx)
        progress[i, : len(t)] = torch.from_numpy(prog)
        is_pause[i, : len(t)] = torch.from_numpy(pause)
        pad[i, : len(t)] = True
    return src, tgt, index, progress, is_pause, pad


def run_epoch(model, items, opt, device, train: bool):
    model.train(train)
    order = list(range(len(items)))
    if train:
        random.shuffle(order)
    total = 0.0
    steps = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for start in range(0, len(order), 32):
            batch = [items[i] for i in order[start:start + 32]]
            src, tgt, index, progress, is_pause, pad = collate(batch, device)
            logits = model(src, index, progress, is_pause, pad)
            loss = adapter_loss(logits, tgt, pad)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += float(loss)
            steps += 1
    return total / max(steps, 1)


def main():
    random.seed(7)
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    train_items = load_split("train")
    dev_items = load_split("dev")
    print(f"chars train={len(train_items)} dev={len(dev_items)}", flush=True)
    model = CodecAdapter().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    best = 1e9
    bad = 0
    log = OUT / "train_log.jsonl"
    for epoch in range(1, 41):
        t0 = time.time()
        tr = run_epoch(model, train_items, opt, device, True)
        dv = run_epoch(model, dev_items, opt, device, False)
        row = {"epoch": epoch, "train": tr, "dev": dv, "sec": round(time.time() - t0, 1)}
        with open(log, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"epoch {epoch:02d} train {tr:.3f} dev {dv:.3f} {row['sec']}s", flush=True)
        if dv < best - 1e-3:
            best = dv
            bad = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "dev": dv}, OUT / "best.pt")
        else:
            bad += 1
            if bad >= 6:
                print("early stop", flush=True)
                break
    print("best dev", round(best, 3), OUT / "best.pt", flush=True)


if __name__ == "__main__":
    main()
