#!/usr/bin/env python3
"""Offline beam-search accuracy sweep (no s2mel). Find best lm_weight."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pypinyin import lazy_pinyin, Style

from semantic_frontend import SemanticFrontend, resample_to_16k
from train_syllable_cls import SyllableClassifier
from infer_lm import NgramLM, beam_search

ROOT = Path("/cmy/cmy/dysa")
ALIGN = ROOT / "data/char_align/s06_part1"
CLS = ROOT / "runs/dsr/train_syllable_cls/best.pt"
NGRAM = ROOT / "runs/dsr/syllable_ngram/ngram.json"


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
    device = torch.device("cuda")
    fe = SemanticFrontend("cuda")
    ckpt = torch.load(str(CLS), map_location="cpu", weights_only=False)
    id_to_pinyin = {int(k): v for k, v in ckpt["id_to_pinyin"].items()}
    cls = SyllableClassifier(d_in=1024, d_h=512, n_class=ckpt["n_class"]).to(device)
    cls.load_state_dict(ckpt["model"])
    cls.eval()
    lm = NgramLM(NGRAM)

    # collect dev samples with candidates
    samples = []
    for line in open(ALIGN / "dev.jsonl"):
        r = json.loads(line)
        dys, _ = read_mono(Path(r["audio"]))
        dys_dur = len(dys) / 16000.0
        feat = fe.features(dys)
        Td = feat.shape[0]
        onsets = r["onsets"]
        ends = onsets[1:] + [dys_dur]
        truth_py = lazy_pinyin(r["chars"], style=Style.TONE3)
        cands = []
        for si in range(len(onsets)):
            a, b = frame_span(onsets[si], ends[si], dys_dur, Td)
            if b - a < 2:
                continue
            seg = feat[a:b].unsqueeze(0)
            mask = torch.ones(1, seg.shape[1], dtype=torch.bool, device=device)
            with torch.no_grad():
                logits = cls(seg.to(device), mask)
            probs = torch.softmax(logits, dim=-1)[0]
            top = torch.topk(probs, k=10)
            cands.append([(id_to_pinyin[int(pid)], float(np.log(p))) for pid, p in zip(top.indices.cpu().numpy(), top.values.cpu().numpy())])
        if cands and len(cands) == len(truth_py):
            samples.append((truth_py, cands))

    print(f"dev samples {len(samples)}")

    for lw in [0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3]:
        hit = 0
        tot = 0
        for truth, cands in samples:
            seq = beam_search(cands, lm, beam_width=8, top_k=10, lm_weight=lw)
            for a, b in zip(truth, seq):
                if a == b:
                    hit += 1
                tot += 1
        print(f"lm_weight={lw:.2f}  syllable_acc={hit/max(tot,1):.3f}")


if __name__ == "__main__":
    main()
