#!/usr/bin/env python3
"""Route A: rule-based directed distortion of normal audio into pseudo-dysarthric audio.

Pipeline per utterance:
  normal audio (24kHz)
    -> per-character: trim trailing silence
    -> directed distortion on speech parts (initial confusion, duration, devoicing, amplitude)
    -> re-insert dysarthric pauses at char boundaries (with crossfade)
    -> output 16kHz pseudo-dysarthric audio
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from pypinyin import lazy_pinyin, Style

ROOT = Path("/cmy/cmy/dysa")
ALIGN = ROOT / "data/char_align/s06_part1"
OUT = ROOT / "runs/dsr/synth_distort"
OUT_SR = 16000
NORMAL_SR = 24000


def trim_trailing_silence(wav_seg, sr, frame_ms=25, rel_thresh=0.08):
    if len(wav_seg) == 0:
        return 0
    frame = max(1, int(sr * frame_ms / 1000))
    nf = max(1, len(wav_seg) // frame)
    rms = np.array([np.sqrt((wav_seg[j * frame:(j + 1) * frame] ** 2).mean()) for j in range(nf)])
    peak = rms.max() + 1e-8
    voiced = np.where(rms > rel_thresh * peak)[0]
    if len(voiced) == 0:
        return frame
    return min((voiced[-1] + 2) * frame, len(wav_seg))


def initial_of(pinyin):
    """Return the initial consonant of a pinyin with tone, or '' if none."""
    import re
    m = re.match(r'^([bpmfdtnlgkhjqxrzcsyw]|zh|ch|sh)', pinyin)
    return m.group(0) if m else ''


def devoice_initial(seg, sr):
    """Simulate zh/ch/sh -> z/c/s (retroflex->alveolar) by low-passing the initial."""
    # apply a low-pass to remove retroflex high-frequency energy
    from scipy.signal import butter, sosfilt
    cutoff = 2500  # Hz
    sos = butter(4, cutoff, btype='low', fs=sr, output='sos')
    return sosfilt(sos, seg).astype(np.float32)


def distort_initial(seg, sr, pinyin, seed):
    rng = np.random.default_rng(seed)
    ini = initial_of(pinyin)
    if ini in ('zh', 'ch', 'sh') and rng.random() < 0.6:
        # retroflex -> alveolar: low-pass + slight energy cut
        seg = devoice_initial(seg, sr)
        seg = seg * 0.85
    elif ini in ('z', 'c', 's') and rng.random() < 0.3:
        # occasional reverse confusion
        seg = seg * 0.9
    return seg.astype(np.float32)


def distort_duration(seg, sr, seed):
    """Randomly lengthen vowels (time-stretch with pitch preserved)."""
    import librosa
    rng = np.random.default_rng(seed)
    if rng.random() < 0.5:
        rate = float(rng.uniform(1.0, 1.8))  # slow down = longer
        out = librosa.effects.time_stretch(seg.astype(np.float32), rate=rate)
        return out.astype(np.float32)
    return seg


def distort_amplitude(seg, seed):
    rng = np.random.default_rng(seed)
    # random envelope modulation (uneven loudness)
    if rng.random() < 0.5:
        t = np.linspace(0, 1, len(seg))
        env = 1.0 + 0.3 * np.sin(2 * np.pi * rng.uniform(3, 8) * t + rng.uniform(0, 6.28))
        seg = seg * env
    return seg.astype(np.float32)


def crossfade(a, b, fade_len):
    if fade_len <= 0 or len(a) == 0 or len(b) == 0:
        return np.concatenate([a, b]).astype(np.float32)
    fade_len = min(fade_len, len(a), len(b))
    fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
    fade_out = np.linspace(1, 0, fade_len, dtype=np.float32)
    a[-fade_len:] *= fade_out
    b[:fade_len] *= fade_in
    return np.concatenate([a, b]).astype(np.float32)


def process_one(row, meta, seed):
    rng = np.random.default_rng(seed)
    normal, sr = sf.read(row["normal_audio"], dtype="float32")
    if normal.ndim > 1:
        normal = normal.mean(1)
    if sr != NORMAL_SR:
        normal = torchaudio.functional.resample(
            torch.from_numpy(normal).unsqueeze(0), sr, NORMAL_SR
        ).squeeze(0).numpy()

    py = lazy_pinyin(row["chars"], style=Style.TONE3)
    pieces = []
    for i, ch in enumerate(row["chars"]):
        m = meta["chars"][i]
        a = int(m["normal_start"] * NORMAL_SR)
        b = int(m["normal_end"] * NORMAL_SR)
        a = max(0, min(a, len(normal)))
        b = max(a + 1, min(b, len(normal)))
        seg = normal[a:b]
        # trim trailing silence
        keep = trim_trailing_silence(seg, NORMAL_SR)
        seg = seg[:keep]
        if len(seg) < NORMAL_SR * 0.02:
            pieces.append(seg)
            continue
        # directed distortion on the char (initial, duration, amplitude)
        # initial = first ~25% of the char (rough)
        ini_len = int(len(seg) * 0.25)
        ini_seg = seg[:ini_len]
        rest = seg[ini_len:]
        ini_seg = distort_initial(ini_seg, NORMAL_SR, py[i], seed + i * 1000)
        seg = np.concatenate([ini_seg, rest])
        seg = distort_duration(seg, NORMAL_SR, seed + i * 2000)
        seg = distort_amplitude(seg, seed + i * 3000)
        pieces.append(seg.astype(np.float32))

    # re-insert pauses at boundaries with crossfade
    out = pieces[0]
    for j in range(1, len(pieces)):
        # random pause length (dysarthric pauses are irregular)
        pause_s = float(rng.uniform(0.05, 0.35))
        pause = np.zeros(int(pause_s * NORMAL_SR), dtype=np.float32)
        out = crossfade(out, pause, int(0.015 * NORMAL_SR))
        out = crossfade(out, pieces[j], int(0.015 * NORMAL_SR))

    # downsample to 16k
    out16 = torchaudio.functional.resample(
        torch.from_numpy(out).unsqueeze(0), NORMAL_SR, OUT_SR
    ).squeeze(0).numpy()
    return out16.astype(np.float32)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # process first 3 train utterances as a smoke test
    rows = [json.loads(l) for l in open(ALIGN / "train.jsonl")][:3]
    for i, r in enumerate(rows, 1):
        meta_path = list((ROOT / "normal_stretched_targets").glob(f"*_{r['id']}/meta.json"))[0]
        meta = json.load(open(meta_path))
        wav = process_one(r, meta, seed=i)
        folder = OUT / f"{i:02d}_{r['id']}"
        folder.mkdir(parents=True, exist_ok=True)
        sf.write(folder / "synth_dys.wav", wav, OUT_SR)
        sf.write(folder / "normal.wav", sf.read(r["normal_audio"], dtype="float32")[0], NORMAL_SR)
        print(f"[{i}/3] {r['id']} {r['text']}  synth={len(wav)/OUT_SR:.2f}s")
    print("done ->", OUT)


if __name__ == "__main__":
    main()
