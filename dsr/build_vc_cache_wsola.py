#!/usr/bin/env python3
"""Build normal->dysarthric mel pairs where the source is WSOLA-stretched.

Unlike v3 (DTW in mel space), the source is the normal waveform stretched with
the same WSOLA routine used at inference, so training and inference see the
same input distribution. The stretch uses the *true* dysarthric character
durations, so the source and the real dysarthric mel stay frame-aligned and the
model only has to learn the spectral distortion.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

ROOT = Path("/cmy/cmy/dysa")
sys.path.insert(0, str(ROOT / "index-tts-full"))
sys.path.insert(0, str(ROOT / "dsr"))

from indextts.s2mel.modules.bigvgan.meldataset import mel_spectrogram
from duration import (
    SR,
    char_targets,
    listen_dir,
    load_rows,
    read_wav,
    wsola_from_index,
)

N_FFT = 1024
HOP = 256
N_MELS = 80
OUT = ROOT / "runs/dsr/vc_cache_wsola"


def wav_to_mel(y: np.ndarray, device: torch.device) -> np.ndarray:
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)))
    wav = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)).unsqueeze(0).to(device)
    mel = mel_spectrogram(wav, N_FFT, N_MELS, SR, HOP, N_FFT, 0, None, center=False)
    return mel.squeeze(0).T.contiguous().cpu().numpy().astype(np.float32)  # [T, 80]


def resample(y: np.ndarray, sr: int, out_sr: int) -> np.ndarray:
    if sr == out_sr:
        return np.ascontiguousarray(y, dtype=np.float32)
    return torchaudio.functional.resample(
        torch.from_numpy(y.astype(np.float32)).unsqueeze(0), sr, out_sr,
    ).squeeze(0).numpy()


def stretch_preserving_pauses(
    normal: np.ndarray,
    normal_sr: int,
    chars: list[dict],
    targets: list[tuple[float, float]],
) -> np.ndarray:
    """WSOLA each character, then keep the true pause as silence (no crossfade)."""
    audio = resample(normal, normal_sr, SR)
    scale = SR / float(normal_sr)
    pieces: list[np.ndarray] = []
    for ch, (speech, pause) in zip(chars, targets):
        a = int(np.clip(round(float(ch["start"]) * normal_sr * scale), 0, len(audio)))
        b = int(np.clip(round(float(ch["end"]) * normal_sr * scale), a, len(audio)))
        seg = np.ascontiguousarray(audio[a:b], dtype=np.float32)
        speech_n = max(1, int(round(speech * SR)))
        if b <= a or len(seg) < 16:
            pieces.append(np.zeros(speech_n, dtype=np.float32))
        else:
            pieces.append(wsola_from_index(seg, np.linspace(0, len(seg) - 1, speech_n), SR))
        pause_n = int(round(max(0.0, pause) * SR))
        if pause_n:
            pieces.append(np.zeros(pause_n, dtype=np.float32))
    if not pieces:
        return np.zeros(SR // 10, dtype=np.float32)
    return np.concatenate(pieces)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        dest = OUT / split
        dest.mkdir(parents=True, exist_ok=True)
        rows = load_rows(split)
        for i, row in enumerate(rows, 1):
            uid = row["id"]
            folder = listen_dir(uid)
            chars = json.loads((folder / "normal_chars.json").read_text())["chars"]
            if len(chars) != len(row["onsets"]):
                print("skip", uid, "char mismatch", flush=True)
                continue
            normal, nsr = read_wav(folder / "normal.wav")
            dys, dsr = read_wav(folder / "dys.wav")
            n_durs = [max(0.0, float(c["end"]) - float(c["start"])) for c in chars]
            targets = char_targets(dys, dsr, row["onsets"], n_durs)
            src_wav = stretch_preserving_pauses(normal, nsr, chars, targets)
            src_mel = wav_to_mel(src_wav, device)
            tgt_mel = wav_to_mel(resample(dys, dsr, SR), device)
            n = min(len(src_mel), len(tgt_mel))
            src_mel = src_mel[:n]
            tgt_mel = tgt_mel[:n]
            np.savez(dest / f"{uid}.npz", src=src_mel, tgt=tgt_mel)
            if i % 50 == 0 or i == len(rows):
                print(f"{split} {i}/{len(rows)} {uid} frames={n} src_sec={len(src_wav)/SR:.2f} dys_sec={len(dys)/dsr:.2f}", flush=True)
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
