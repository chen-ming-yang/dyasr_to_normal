#!/usr/bin/env python3
"""Build syllable library: pinyin(tone) -> {key vec, audio segments, code seqs}."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pypinyin import lazy_pinyin, Style

from semantic_frontend import SemanticFrontend, resample_to_16k

ROOT = Path("/cmy/cmy/dysa")
ALIGN = ROOT / "data/char_align/s06_part1"
OUT = ROOT / "runs/dsr/syllable_lib"


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

    # 1. collect samples per pinyin
    samples = []  # {pinyin, char, uid, key[1024], normal_wav, a0, a1 (sec), codes}
    for line in open(ALIGN / "train.jsonl"):
        r = json.loads(line)
        uid = r["id"]
        dys, _ = read_mono(Path(r["audio"]))
        normal, _ = read_mono(Path(r["normal_audio"]))
        dys_dur = len(dys) / 16000.0
        nor_dur = len(normal) / 16000.0
        feat = fe.features(dys)
        codes = fe.codes(normal)
        Td = feat.shape[0]
        onsets = r["onsets"]
        ends = onsets[1:] + [dys_dur]
        py = lazy_pinyin(r["chars"], style=Style.TONE3)

        meta_path = ROOT / "normal_stretched_targets" / f"{uid.zfill(4)}" / "meta.json"
        if not meta_path.exists():
            cand = list((ROOT / "normal_stretched_targets").glob(f"*_{uid}/meta.json"))
            meta_path = cand[0] if cand else None
        if meta_path is None:
            continue
        meta = json.load(open(meta_path))

        for i, ch in enumerate(r["chars"]):
            a, b = frame_span(onsets[i], ends[i], dys_dur, Td)
            if b - a < 2:
                continue
            key = feat[a:b].mean(0)
            key = key / (torch.norm(key) + 1e-8)
            nm = meta["chars"][i]
            n0, n1 = frame_span(float(nm["normal_start"]), float(nm["normal_end"]), nor_dur, len(codes))
            seq = codes[n0:n1]
            if len(seq) == 0:
                continue
            samples.append({
                "pinyin": py[i],
                "char": ch,
                "uid": uid,
                "key": key.float().cpu().numpy(),
                "normal_wav": str(Path(r["normal_audio"])),
                "a0": float(nm["normal_start"]),
                "a1": float(nm["normal_end"]),
                "codes": seq.astype(np.int16),
            })

    # 2. group by pinyin, mean key per syllable
    from collections import defaultdict
    by_py = defaultdict(list)
    for s in samples:
        by_py[s["pinyin"]].append(s)

    keys = []
    syll_meta = []
    for py, lst in sorted(by_py.items()):
        mean_key = np.stack([s["key"] for s in lst]).mean(0)
        mean_key = mean_key / (np.linalg.norm(mean_key) + 1e-8)
        keys.append(mean_key.astype(np.float32))
        syll_meta.append({
            "pinyin": py,
            "n": len(lst),
            "samples": lst,
        })

    keys = np.stack(keys)
    np.save(OUT / "keys.npy", keys)
    with open(OUT / "meta.jsonl", "w") as f:
        for sm in syll_meta:
            f.write(json.dumps({
                "pinyin": sm["pinyin"],
                "n": sm["n"],
                "samples": [
                    {"char": s["char"], "uid": s["uid"], "normal_wav": s["normal_wav"],
                     "a0": s["a0"], "a1": s["a1"], "codes": s["codes"].tolist()}
                    for s in sm["samples"]
                ],
            }) + "\n")

    print(f"samples {len(samples)}  syllables {len(syll_meta)}")
    ns = [sm["n"] for sm in syll_meta]
    print(f"median samples/syllable {int(np.median(ns))}  min {min(ns)}  max {max(ns)}")
    # show a few
    for sm in syll_meta[:5]:
        print(f"  {sm['pinyin']}  n={sm['n']}  chars={sorted(set(s['char'] for s in sm['samples']))[:6]}")


if __name__ == "__main__":
    main()
