#!/usr/bin/env python3
"""Cache paired (normal_stretched_mel -> dys_mel) for normal-to-dysarthric VC."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

ROOT = Path("/cmy/cmy/dysa")
MANIFEST = ROOT / "normal_stretched_targets"
OUT = ROOT / "runs/dsr/vc_cache"

MEL_SR = 22050
N_FFT = 1024
HOP = 256
WIN = 1024
N_MELS = 80
FMIN = 0.0
FMAX = 8000.0


_mel_basis = None
_hann_window = None


def get_mel_basis(device):
    global _mel_basis, _hann_window
    if _mel_basis is None:
        import librosa
        mel = librosa.filters.mel(sr=MEL_SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
        _mel_basis = torch.from_numpy(mel).float().to(device)
        _hann_window = torch.hann_window(WIN).to(device)
    return _mel_basis, _hann_window


def wav_to_logmel(y, device):
    if y.ndim > 1:
        y = y.mean(axis=0)
    wav = torch.from_numpy(y.astype(np.float32)).to(device)
    pad = (N_FFT - HOP) // 2
    wav = torch.nn.functional.pad(wav.unsqueeze(0), (pad, pad), mode="reflect").squeeze(0)
    mel_basis, window = get_mel_basis(device)
    spec = torch.stft(wav, n_fft=N_FFT, hop_length=HOP, win_length=WIN, window=window,
                      center=False, pad_mode="reflect", normalized=False, onesided=True, return_complex=True)
    mag = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)
    mel = torch.matmul(mel_basis, mag)
    logmel = torch.log(torch.clamp(mel, min=1e-5))
    return logmel.T.contiguous().cpu().numpy().astype(np.float32)  # [T, 80]


def load_16k_mono(path):
    y, sr = sf.read(str(path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    if sr != MEL_SR:
        y = torchaudio.functional.resample(torch.from_numpy(y).unsqueeze(0), sr, MEL_SR).squeeze(0).numpy()
    return y.astype(np.float32)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for split in ["train", "dev", "test"]:
        rows = [json.loads(l) for l in open(MANIFEST / f"{split}.jsonl")]
        out_split = OUT / split
        out_split.mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(rows, 1):
            uid = r["id"]
            # input: normal_stretched mel (already computed)
            src_mel = np.load(r["mel_path"]).astype(np.float32)
            # target: dysarthric mel (compute from dys wav)
            dys = load_16k_mono(r["dys_wav"])
            tgt_mel = wav_to_logmel(dys, device)
            # align lengths
            n = min(len(src_mel), len(tgt_mel))
            src_mel = src_mel[:n]
            tgt_mel = tgt_mel[:n]
            np.savez(out_split / f"{uid}.npz", src=src_mel, tgt=tgt_mel)
            if i % 50 == 0 or i == len(rows):
                print(f"{split} {i}/{len(rows)} {uid} mel={n}", flush=True)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
