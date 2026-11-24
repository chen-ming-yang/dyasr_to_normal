#!/usr/bin/env python3
"""Lay normal-audio code vectors onto dysarthric frames with per-character DTW."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from semantic_frontend import SemanticFrontend, resample_to_16k

ROOT = Path("/cmy/cmy/dysa")
OLD = ROOT / "runs/dsr/code_cache"
MANIFEST = ROOT / "normal_stretched_targets"
OUT = ROOT / "runs/dsr/code_cache_v4"


def read_mono(path: Path):
    wav, sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return resample_to_16k(wav, sr), sr


def frame_span(start: float, end: float, dur: float, n: int) -> tuple[int, int]:
    if dur <= 0 or n <= 0:
        return 0, 0
    a = int(round(start / dur * n))
    b = int(round(end / dur * n))
    a = max(0, min(a, n))
    b = max(0, min(b, n))
    return a, b


def dtw_assign(dys: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """For each dys frame, average the DTW-matched normal vectors. Both are [T, D]."""
    dys_n = dys / (np.linalg.norm(dys, axis=1, keepdims=True) + 1e-8)
    nor_n = normal / (np.linalg.norm(normal, axis=1, keepdims=True) + 1e-8)
    cost = 1.0 - dys_n @ nor_n.T
    n, m = cost.shape
    inf = 1e9
    dp = np.full((n + 1, m + 1), inf, dtype=np.float64)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i, j] = cost[i - 1, j - 1] + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    i, j = n, m
    buckets: list[list[int]] = [[] for _ in range(n)]
    while i > 0 and j > 0:
        buckets[i - 1].append(j - 1)
        opts = [(dp[i - 1, j - 1], i - 1, j - 1), (dp[i - 1, j], i - 1, j), (dp[i, j - 1], i, j - 1)]
        _, i, j = min(opts, key=lambda t: t[0])
    assigned = np.zeros_like(dys, dtype=np.float32)
    for t, js in enumerate(buckets):
        if not js:
            continue
        assigned[t] = normal[js].mean(axis=0)
    return assigned


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fe = SemanticFrontend("cuda")
    index = []
    for split in ("train", "dev", "test"):
        rows = [json.loads(line) for line in open(MANIFEST / f"{split}.jsonl")]
        for n_done, row in enumerate(rows, 1):
            uid = row["id"]
            old = np.load(OLD / split / f"{uid}.npz")
            feat = old["feat"]
            meta = json.load(open(Path(row["mel_path"]).with_name("meta.json")))
            normal, nsr = read_mono(Path(row["dys_wav"]).with_name("normal.wav"))
            z_n, _ = fe.code_vectors(normal)
            z_n = z_n.numpy().astype(np.float32)
            dys, _ = read_mono(Path(row["dys_wav"]))
            dys_dur = len(dys) / 16000.0
            nor_dur = len(normal) / 16000.0
            td = feat.shape[0]
            target = np.zeros((td, 8), dtype=np.float32)
            speech = np.zeros(td, dtype=np.uint8)
            spans = []
            for ch in meta["chars"]:
                speech_end = float(ch["dys_start"]) + float(ch["dys_dur"]) - float(ch["silence_pad_sec"])
                a, b = frame_span(float(ch["dys_start"]), speech_end, dys_dur, td)
                n0, n1 = frame_span(float(ch["normal_start"]), float(ch["normal_end"]), nor_dur, len(z_n))
                if b - a < 1 or n1 - n0 < 1:
                    continue
                # DTW uses the cached dys features against a time-matched slice of normal vectors.
                # Normal vectors are already the code space; align by cosine inside the character.
                dys_z = fe.project(torch.from_numpy(feat[a:b].astype(np.float32))).cpu().numpy()
                assigned = dtw_assign(dys_z, z_n[n0:n1])
                target[a:b] = assigned
                speech[a:b] = 1
                spans.append((a, b))
            path = OUT / split / f"{uid}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                path,
                feat=feat,
                target=target.astype(np.float16),
                speech=speech,
                spans=np.asarray(spans, dtype=np.int32).reshape(-1, 2),
            )
            index.append({"id": uid, "split": split, "path": str(path), "n_frames": int(td), "n_chars": len(spans)})
            if n_done % 50 == 0 or n_done == len(rows):
                print(f"{split} {n_done}/{len(rows)} {uid}", flush=True)
    with open(OUT / "index.jsonl", "w") as f:
        for rec in index:
            f.write(json.dumps(rec) + "\n")
    print("done", len(index))


if __name__ == "__main__":
    main()
