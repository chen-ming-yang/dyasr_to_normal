#!/usr/bin/env python3
"""Extract IndexTTS semantic codes and lay them on the dysarthric timeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from semantic_frontend import SILENCE, SemanticFrontend, resample_to_16k

ROOT = Path("/cmy/cmy/dysa")
MANIFEST = ROOT / "normal_stretched_targets"
OUT = ROOT / "runs/dsr/code_cache"


def read_mono(path: Path):
    wav, sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav, sr


def fit_codes(codes: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    if len(codes) == 0:
        return np.full(n, SILENCE, dtype=np.int64)
    if len(codes) >= n:
        return codes[:n].astype(np.int64)
    reps = int(np.ceil(n / len(codes)))
    return np.tile(codes, reps)[:n].astype(np.int64)


def frame_span(start: float, end: float, dur: float, n: int) -> tuple[int, int]:
    if dur <= 0 or n <= 0:
        return 0, 0
    a = int(round(start / dur * n))
    b = int(round(end / dur * n))
    a = max(0, min(a, n))
    b = max(0, min(b, n))
    return a, b


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fe = SemanticFrontend("cuda")
    index = []
    for split in ("train", "dev", "test"):
        rows = [json.loads(l) for l in open(MANIFEST / f"{split}.jsonl")]
        for i, row in enumerate(rows, 1):
            uid = row["id"]
            meta = json.load(open(Path(row["mel_path"]).with_name("meta.json")))
            dys, dsr = read_mono(Path(row["dys_wav"]))
            if dsr != 16000:
                dys = resample_to_16k(dys, dsr)
            normal, nsr = read_mono(Path(row["dys_wav"]).with_name("normal.wav"))
            normal = resample_to_16k(normal, nsr)

            dys_feat = fe.features(dys).numpy().astype(np.float16)
            normal_codes = fe.codes(normal)
            td = dys_feat.shape[0]
            tn = len(normal_codes)
            dys_dur = len(dys) / 16000
            nor_dur = len(normal) / 16000
            labels = np.full(td, SILENCE, dtype=np.int16)
            spans = []
            for ch in meta["chars"]:
                speech_end = float(ch["dys_start"]) + float(ch["dys_dur"]) - float(ch["silence_pad_sec"])
                a, b = frame_span(float(ch["dys_start"]), speech_end, dys_dur, td)
                n0, n1 = frame_span(float(ch["normal_start"]), float(ch["normal_end"]), nor_dur, tn)
                seg = normal_codes[n0:n1]
                if b - a < 1:
                    continue
                labels[a:b] = fit_codes(seg, b - a).astype(np.int16)
                spans.append((a, b))
            path = OUT / split / f"{uid}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                path,
                feat=dys_feat,
                labels=labels,
                spans=np.asarray(spans, dtype=np.int32).reshape(-1, 2),
            )
            index.append({"id": uid, "split": split, "path": str(path), "n_frames": int(td), "n_chars": len(spans)})
            if i % 50 == 0 or i == len(rows):
                print(f"{split} {i}/{len(rows)} {uid} frames={td} chars={len(spans)}", flush=True)
    with open(OUT / "index.jsonl", "w") as f:
        for rec in index:
            f.write(json.dumps(rec) + "\n")
    print("done", len(index))


if __name__ == "__main__":
    main()
