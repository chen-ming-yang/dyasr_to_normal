#!/usr/bin/env python3
"""Gate: measure top-K syllable accuracy of the v8 classifier on dev (and train)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pypinyin import lazy_pinyin, Style

from semantic_frontend import SemanticFrontend, resample_to_16k
from train_syllable_cls import SyllableClassifier

ROOT = Path("/cmy/cmy/dysa")
ALIGN = ROOT / "data/char_align/s06_part1"
CLS = ROOT / "runs/dsr/train_syllable_cls/best.pt"
OUT = ROOT / "runs/dsr/topk"


def read_mono(path: Path):
    y, sr = sf.read(str(path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    return resample_to_16k(y, sr), sr


def frame_span(start, end, dur, n):
    a = int(round(start / dur * n))
    b = int(round(end / dur * n))
    return max(0, min(a, n)), max(0, min(b, n))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    fe = SemanticFrontend("cuda")

    ckpt = torch.load(str(CLS), map_location="cpu", weights_only=False)
    id_to_pinyin = {int(k): v for k, v in ckpt["id_to_pinyin"].items()}
    cls = SyllableClassifier(d_in=1024, d_h=512, n_class=ckpt["n_class"]).to(device)
    cls.load_state_dict(ckpt["model"])
    cls.eval()
    print(f"classes {ckpt['n_class']} best_dev_acc {ckpt.get('best_dev_acc')}")

    report = {}
    for split in ["train", "dev"]:
        topk = {k: 0 for k in [1, 3, 5, 10, 20, 50]}
        n = 0
        for line in open(ALIGN / f"{split}.jsonl"):
            r = json.loads(line)
            dys, _ = read_mono(Path(r["audio"]))
            dys_dur = len(dys) / 16000.0
            feat = fe.features(dys)
            Td = feat.shape[0]
            onsets = r["onsets"]
            ends = onsets[1:] + [dys_dur]
            py = lazy_pinyin(r["chars"], style=Style.TONE3)
            for i in range(len(r["chars"])):
                a, b = frame_span(onsets[i], ends[i], dys_dur, Td)
                if b - a < 2:
                    continue
                seg = feat[a:b].unsqueeze(0)
                mask = torch.ones(1, seg.shape[1], dtype=torch.bool, device=device)
                with torch.no_grad():
                    logits = cls(seg.to(device), mask)
                probs = torch.softmax(logits, dim=-1)[0]
                order = torch.argsort(probs, descending=True).cpu().numpy()
                truth_id = next((int(k) for k, v in id_to_pinyin.items() if v == py[i]), None)
                if truth_id is None:
                    continue
                rank = int(np.where(order == truth_id)[0][0]) + 1
                for k in topk:
                    if rank <= k:
                        topk[k] += 1
                n += 1
        report[split] = {f"top{k}": topk[k] / max(n, 1) for k in topk}
        report[split]["n"] = n
        line_str = "  ".join(f"top{k}={report[split][f'top{k}']:.3f}" for k in topk)
        print(f"{split}: n={n}  {line_str}")

    with open(OUT / "topk.json", "w") as f:
        json.dump(report, f, indent=2)
    print("saved ->", OUT / "topk.json")


if __name__ == "__main__":
    main()
