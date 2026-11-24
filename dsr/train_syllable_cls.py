#!/usr/bin/env python3
"""Train a dysarthric char -> syllable(tone) classifier.

Feature = layer-17 pooled per char. Label = pinyin with tone (e.g. 'wo3').
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from pypinyin import lazy_pinyin, Style
from torch.utils.data import DataLoader, Dataset

from semantic_frontend import SemanticFrontend, resample_to_16k

ROOT = Path("/cmy/cmy/dysa")
ALIGN = ROOT / "data/char_align/s06_part1"
OUT = ROOT / "runs/dsr/train_syllable_cls"


def read_mono(path: Path):
    y, sr = sf.read(str(path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    return resample_to_16k(y, sr), sr


def frame_span(start, end, dur, n):
    a = int(round(start / dur * n))
    b = int(round(end / dur * n))
    return max(0, min(a, n)), max(0, min(b, n))


def trim_trailing_silence(wav_seg: np.ndarray, sr: int, frame_ms: int = 25, rel_thresh: float = 0.08) -> int:
    """Return the end sample index where trailing silence begins.

    Uses frame RMS; walks from the end backwards, stopping once energy
    exceeds rel_thresh * peak energy. Keeps at least 1 frame.
    """
    if len(wav_seg) == 0:
        return 0
    frame = int(sr * frame_ms / 1000)
    if frame <= 0:
        frame = int(sr * 0.025)
    nf = max(1, len(wav_seg) // frame)
    rms = np.array([
        np.sqrt((wav_seg[j * frame:(j + 1) * frame] ** 2).mean())
        for j in range(nf)
    ])
    peak = rms.max() + 1e-8
    # find last frame with energy above threshold
    voiced = np.where(rms > rel_thresh * peak)[0]
    if len(voiced) == 0:
        # nearly all silence; keep the first frame only
        return frame
    last = voiced[-1]
    # keep through the voiced frame (plus one trailing frame margin)
    end = (last + 2) * frame
    return min(end, len(wav_seg))


class SyllableSet(Dataset):
    def __init__(self, split: str, fe, min_frames: int = 3):
        self.items = []
        pinyin_to_id = {}
        for line in open(ALIGN / f"{split}.jsonl"):
            r = json.loads(line)
            dys, sr = read_mono(Path(r["audio"]))
            dys_dur = len(dys) / 16000.0
            feat = fe.features(dys)
            Td = feat.shape[0]
            onsets = r["onsets"]
            ends = onsets[1:] + [dys_dur]
            py = lazy_pinyin(r["chars"], style=Style.TONE3)
            for i in range(len(r["chars"])):
                a, b = frame_span(onsets[i], ends[i], dys_dur, Td)
                if b - a < min_frames:
                    continue
                # trim trailing silence on the waveform side, then remap to feature frames
                a_s = int(onsets[i] * sr)
                b_s = int(ends[i] * sr)
                trim_s = trim_trailing_silence(dys[a_s:b_s], sr)
                new_end = min(ends[i], onsets[i] + trim_s / sr)
                a2, b2 = frame_span(onsets[i], new_end, dys_dur, Td)
                if b2 - a2 < 2:
                    b2 = min(a2 + 2, b)
                p = py[i]
                if p not in pinyin_to_id:
                    pinyin_to_id[p] = len(pinyin_to_id)
                self.items.append((feat[a2:b2].float().cpu().numpy(), pinyin_to_id[p]))
        self.pinyin_to_id = pinyin_to_id
        self.id_to_pinyin = {v: k for k, v in pinyin_to_id.items()}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        feat, label = self.items[idx]
        return torch.from_numpy(feat), label


def collate(batch):
    feats = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    max_len = max(f.shape[0] for f in feats)
    B = len(feats)
    padded = torch.zeros(B, max_len, feats[0].shape[1])
    mask = torch.ones(B, max_len, dtype=torch.bool)
    for i, f in enumerate(feats):
        padded[i, : f.shape[0]] = f
        mask[i, f.shape[0] :] = False
    return padded, mask, labels


class SyllableClassifier(nn.Module):
    def __init__(self, d_in: int = 1024, d_h: int = 512, n_class: int = 622):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(d_in, 1))
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_h),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_h, d_h),
            nn.GELU(),
            nn.Linear(d_h, n_class),
        )

    def forward(self, x, mask):
        w = self.attn(x).masked_fill(~mask.unsqueeze(-1), -1e9)
        w = torch.softmax(w, dim=1)
        pooled = (x * w).sum(dim=1)
        return self.mlp(pooled)


def run_epoch(loader, model, opt, device, train: bool):
    model.train(train)
    total_correct = 0
    total_n = 0
    total_loss = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for feat, mask, labels in loader:
            feat = feat.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            logits = model(feat, mask)
            loss = F.cross_entropy(logits, labels)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            pred = logits.argmax(-1)
            total_correct += int((pred == labels).sum().item())
            total_n += len(labels)
            total_loss += loss.item() * len(labels)
    return total_correct / max(total_n, 1), total_loss / max(total_n, 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    fe = SemanticFrontend("cuda")
    train_ds = SyllableSet("train", fe)
    dev_ds = SyllableSet("dev", fe)
    n_class = len(train_ds.pinyin_to_id)
    print(f"train chars {len(train_ds)} dev chars {len(dev_ds)}  classes {n_class}")
    # save vocab
    with open(OUT / "pinyin_vocab.json", "w") as f:
        json.dump(train_ds.id_to_pinyin, f, ensure_ascii=False)

    model = SyllableClassifier(d_in=1024, d_h=512, n_class=n_class).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=64, shuffle=shuffle, collate_fn=collate)

    train_loader = loader(train_ds, True)
    dev_loader = loader(dev_ds, False)
    history = []
    best = -1.0
    bad = 0
    for ep in range(1, 60):
        t0 = time.time()
        tr_acc, tr_loss = run_epoch(train_loader, model, opt, device, True)
        va_acc, va_loss = run_epoch(dev_loader, model, None, device, False)
        row = {"epoch": ep, "train_acc": tr_acc, "dev_acc": va_acc, "train_loss": tr_loss, "dev_loss": va_loss, "sec": round(time.time() - t0, 1)}
        history.append(row)
        print(f"epoch {ep:02d}  train_acc={tr_acc:.3f}  dev_acc={va_acc:.3f}  ({row['sec']}s)", flush=True)
        ckpt = {"model": model.state_dict(), "history": history, "best_dev_acc": best, "n_class": n_class, "id_to_pinyin": train_ds.id_to_pinyin}
        torch.save(ckpt, OUT / "last.pt")
        if va_acc > best + 1e-4:
            best = va_acc
            bad = 0
            ckpt["best_dev_acc"] = best
            torch.save(ckpt, OUT / "best.pt")
            print(f"  ^ best dev acc {best:.3f}", flush=True)
        else:
            bad += 1
            if bad >= 8:
                print("early stop")
                break
    with open(OUT / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"done best_dev_acc={best:.3f}")


if __name__ == "__main__":
    main()
