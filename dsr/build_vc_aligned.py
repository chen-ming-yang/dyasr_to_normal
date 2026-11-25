#!/usr/bin/env python3
"""Build normal→dysarthric mel pairs on the normal timeline.

v1 placed stretched normal audio onto the dysarthric clock. Inside a character
the two signals were not the same phone at the same frame (voiced correlation
~0.30), so framewise L1 learned a blur.

v2 keeps each normal sentence at its own timing. Within every character, the
dysarthric phone is DTW-aligned onto that normal slice. Frames outside a
character keep the normal mel, so pauses stay pauses. Mel uses BigVGAN's own
filterbank (fmax = sr/2) so the vocoder is not asked to read the wrong bins.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

ROOT = Path("/cmy/cmy/dysa")
sys.path.insert(0, str(ROOT / "index-tts-full"))

from indextts.s2mel.modules.bigvgan.meldataset import mel_spectrogram

ALIGN = ROOT / "data/char_align/s06_part1"
OUT = ROOT / "runs/dsr/vc_cache_v2"
SR = 22050
HOP = 256
N_FFT = 1024
N_MELS = 80


def load_align() -> dict[str, dict]:
    rows = {}
    for split in ("train", "dev", "test"):
        for line in open(ALIGN / f"{split}.jsonl"):
            row = json.loads(line)
            row["split"] = split
            rows[row["id"]] = row
    return rows


def read_22k(path: Path) -> np.ndarray:
    y, sr = sf.read(str(path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    if sr != SR:
        y = torchaudio.functional.resample(torch.from_numpy(y).unsqueeze(0), sr, SR).squeeze(0).numpy()
    return np.ascontiguousarray(y, dtype=np.float32)


def wav_to_mel(y: np.ndarray, device: torch.device) -> np.ndarray:
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)))
    wav = torch.from_numpy(y).unsqueeze(0).to(device)
    mel = mel_spectrogram(wav, N_FFT, N_MELS, SR, HOP, N_FFT, 0, None, center=False)
    return mel.squeeze(0).T.contiguous().cpu().numpy().astype(np.float32)  # [T, 80]


def sec_to_frame(t: float, n: int) -> int:
    return int(np.clip(round(t * SR / HOP), 0, n))


def trim_phone(wav: np.ndarray, frame_ms: int = 20, rel: float = 0.08, pad_ms: int = 30) -> np.ndarray:
    """Drop leading and trailing silence inside one dysarthric character."""
    if len(wav) < int(SR * 0.04):
        return wav
    frame = max(1, int(SR * frame_ms / 1000))
    nf = len(wav) // frame
    if nf < 1:
        return wav
    rms = np.array([np.sqrt((wav[j * frame:(j + 1) * frame] ** 2).mean() + 1e-12) for j in range(nf)])
    voiced = np.where(rms > rel * (rms.max() + 1e-8))[0]
    if len(voiced) == 0:
        return wav
    pad = int(SR * pad_ms / 1000)
    a = max(0, int(voiced[0]) * frame - pad)
    b = min(len(wav), (int(voiced[-1]) + 1) * frame + pad)
    return wav[a:b]


def dtw_warp(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Warp tgt mel onto src's time axis. One tgt frame per src frame (no averaging)."""
    ts, tt = len(src), len(tgt)
    if ts < 1 or tt < 1:
        return np.zeros((max(ts, 0), src.shape[-1] if src.ndim == 2 else N_MELS), dtype=np.float32)
    if ts < 2 or tt < 2:
        idx = np.linspace(0, tt - 1, ts).round().astype(np.int64)
        return tgt[idx]
    s = src - src.mean(axis=1, keepdims=True)
    t = tgt - tgt.mean(axis=1, keepdims=True)
    # [Ts, Tt]
    cost = np.mean((s[:, None, :] - t[None, :, :]) ** 2, axis=-1)
    acc = np.empty((ts, tt), dtype=np.float64)
    acc[0, 0] = cost[0, 0]
    for i in range(1, ts):
        acc[i, 0] = acc[i - 1, 0] + cost[i, 0]
    for j in range(1, tt):
        acc[0, j] = acc[0, j - 1] + cost[0, j]
    for i in range(1, ts):
        ci = cost[i]
        prev = acc[i - 1]
        row = acc[i]
        row[0] = acc[i, 0]
        for j in range(1, tt):
            row[j] = ci[j] + min(prev[j], row[j - 1], prev[j - 1])
    i, j = ts - 1, tt - 1
    buckets: list[list[int]] = [[] for _ in range(ts)]
    while True:
        buckets[i].append(j)
        if i == 0 and j == 0:
            break
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            opts = ((acc[i - 1, j - 1], i - 1, j - 1), (acc[i - 1, j], i - 1, j), (acc[i, j - 1], i, j - 1))
            _, i, j = min(opts, key=lambda x: x[0])
    out = np.empty_like(src)
    for i, js in enumerate(buckets):
        out[i] = tgt[js[len(js) // 2]] if js else tgt[min(i, tt - 1)]
    return out


def align_utterance(normal: np.ndarray, dys: np.ndarray, normal_chars: list[dict], onsets: list[float], device) -> tuple[np.ndarray, np.ndarray, float]:
    src = wav_to_mel(normal, device)
    dys_dur = len(dys) / SR
    ends = [onsets[i + 1] if i + 1 < len(onsets) else dys_dur for i in range(len(onsets))]
    tgt = src.copy()
    speech = np.zeros(len(src), dtype=np.bool_)
    matched = []
    for ch, t0, t1 in zip(normal_chars, onsets, ends):
        a = sec_to_frame(float(ch["start"]), len(src))
        b = sec_to_frame(float(ch["end"]), len(src))
        if b - a < 2:
            continue
        i0 = int(np.clip(round(t0 * SR), 0, len(dys)))
        i1 = int(np.clip(round(t1 * SR), 0, len(dys)))
        piece = trim_phone(dys[i0:i1])
        if len(piece) < int(0.03 * SR):
            continue
        dys_mel = wav_to_mel(piece, device)
        if len(dys_mel) < 2:
            continue
        warped = dtw_warp(src[a:b], dys_mel)
        tgt[a:b] = warped
        speech[a:b] = True
        if speech.sum() > 0:
            s = src[a:b]
            matched.append(np.corrcoef(s.ravel(), warped.ravel())[0, 1])
    corr = float(np.mean(matched)) if matched else 0.0
    return src, tgt, corr


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    align = load_align()
    OUT.mkdir(parents=True, exist_ok=True)
    corrs = []
    for split in ("train", "dev", "test"):
        out_split = OUT / split
        out_split.mkdir(parents=True, exist_ok=True)
        rows = [r for r in align.values() if r["split"] == split]
        rows.sort(key=lambda r: r["id"])
        for i, row in enumerate(rows, 1):
            uid = row["id"]
            audio = Path(row["audio"])
            # listen folder sits next to the copied dys wav when present; otherwise the align path.
            listen = None
            for cand in Path("/cmy/cmy/dysa/listen_samples_all").glob(f"*_{uid}"):
                listen = cand
                break
            if listen is None:
                raise FileNotFoundError(uid)
            normal = read_22k(listen / "normal.wav")
            dys = read_22k(listen / "dys.wav")
            normal_chars = json.loads((listen / "normal_chars.json").read_text())["chars"]
            if len(normal_chars) != len(row["onsets"]):
                raise RuntimeError(f"{uid} char/onset mismatch {len(normal_chars)} vs {len(row['onsets'])}")
            src, tgt, corr = align_utterance(normal, dys, normal_chars, row["onsets"], device)
            np.savez(out_split / f"{uid}.npz", src=src, tgt=tgt)
            corrs.append(corr)
            if i % 50 == 0 or i == len(rows):
                print(f"{split} {i}/{len(rows)} {uid} frames={len(src)} char_corr={corr:.3f}", flush=True)
    print(f"mean within-char corr {np.mean(corrs):.3f} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
