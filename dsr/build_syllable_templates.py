#!/usr/bin/env python3
"""Build syllable templates: dysarthric pooled feature (key) -> normal code sequence (value)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from semantic_frontend import SemanticFrontend, resample_to_16k

ROOT = Path("/cmy/cmy/dysa")
ALIGN = ROOT / "data/char_align/s06_part1"
OUT = ROOT / "runs/dsr/syllable_templates"


def read_mono(path: Path):
    wav, sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    return resample_to_16k(wav, sr), sr


def frame_span(start: float, end: float, dur: float, n: int):
    if dur <= 0 or n <= 0:
        return 0, 0
    a = int(round(start / dur * n))
    b = int(round(end / dur * n))
    return max(0, min(a, n)), max(0, min(b, n))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fe = SemanticFrontend("cuda")
    keys = []  # pooled dysarthric layer-17 features [N, 1024]
    metas = []  # char, uid, code seq

    for line in open(ALIGN / "train.jsonl"):
        r = json.loads(line)
        uid = r["id"]
        dys, _ = read_mono(Path(r["audio"]))
        normal, _ = read_mono(Path(r["normal_audio"]))
        dys_dur = len(dys) / 16000.0
        nor_dur = len(normal) / 16000.0

        feat = fe.features(dys)  # [T, 1024]
        codes = fe.codes(normal)  # [Tn]
        Td = feat.shape[0]

        # char boundaries on dys side = onset_i .. onset_{i+1}
        onsets = r["onsets"]
        ends = onsets[1:] + [dys_dur]

        # normal char boundaries need meta from normal_stretched_targets
        meta_path = ROOT / "normal_stretched_targets" / f"{uid.zfill(4)}" / "meta.json"
        # fallback: search
        if not meta_path.exists():
            import glob
            cand = list((ROOT / "normal_stretched_targets").glob(f"*_{uid}/meta.json"))
            meta_path = cand[0] if cand else None
        if meta_path is None:
            continue
        meta = json.load(open(meta_path))

        for i, ch in enumerate(r["chars"]):
            a, b = frame_span(onsets[i], ends[i], dys_dur, Td)
            if b - a < 2:
                continue
            key = feat[a:b].mean(axis=0)  # [1024]
            key = key / (torch.norm(key) + 1e-8)
            nm = meta["chars"][i]
            n0, n1 = frame_span(float(nm["normal_start"]), float(nm["normal_end"]), nor_dur, len(codes))
            seq = codes[n0:n1]
            if len(seq) == 0:
                continue
            keys.append(key.float().cpu().numpy())
            metas.append({"char": ch, "uid": uid, "codes": seq.astype(np.int16)})

    keys = np.stack(keys)
    np.save(OUT / "keys.npy", keys)
    with open(OUT / "meta.jsonl", "w") as f:
        for m in metas:
            f.write(json.dumps({"char": m["char"], "uid": m["uid"], "codes": m["codes"].tolist()}) + "\n")
    print(f"templates: {len(metas)}  keys {keys.shape}")
    # report unique chars
    chars = {}
    for m in metas:
        chars.setdefault(m["char"], 0)
        chars[m["char"]] += 1
    print("unique chars:", len(chars), "median samples/char:", int(np.median(list(chars.values()))))


if __name__ == "__main__":
    main()
