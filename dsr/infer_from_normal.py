#!/usr/bin/env python3
"""Normal audio -> S06 dysarthric audio, without the patient's recording.

Duration comes from the character model. The stretched waveform's mel goes
through the frozen MelVC residual converter and BigVGAN.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path("/cmy/cmy/dysa")
sys.path.insert(0, str(ROOT / "index-tts-full"))
sys.path.insert(0, str(ROOT / "dsr"))

from indextts.s2mel.modules.bigvgan import bigvgan
from indextts.s2mel.modules.bigvgan.meldataset import mel_spectrogram
from duration import SR, CharDuration, Vocab, listen_dir, read_wav, render, syllable_of
from vc_model import MelVC

N_FFT = 1024
HOP = 256
N_MELS = 80
RESIDUAL_SCALE = 0.6
GATE_MODE = "energy_freq"
DUR_CKPT = ROOT / "runs/dsr/train_duration/best.pt"
VC_CKPT = ROOT / "runs/dsr/train_vc_wsola_stft/best.pt"
BIGVGAN_DIR = ROOT / "bigvgan_v2_22khz_80band_256x"
MANIFEST = ROOT / "normal_stretched_targets"
OUT = ROOT / "runs/dsr/vc_listen_dur"


def pick(split: str, n: int) -> list[dict]:
    rows = [json.loads(line) for line in open(MANIFEST / f"{split}.jsonl")]
    if len(rows) <= n:
        return rows
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n)]


def wav_to_mel(y: np.ndarray, device: torch.device) -> torch.Tensor:
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)))
    wav = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)).unsqueeze(0).to(device)
    mel = mel_spectrogram(wav, N_FFT, N_MELS, SR, HOP, N_FFT, 0, None, center=False)
    return mel  # [1, 80, T]


@torch.no_grad()
def vocode(bv, mel: torch.Tensor) -> np.ndarray:
    wav = bv(mel.float()).squeeze().float().cpu().numpy()
    peak = float(np.max(np.abs(wav))) + 1e-8
    if peak > 1:
        wav = wav / peak
    return wav.astype(np.float32)


def peak_norm(wav: np.ndarray, peak: float = 0.95) -> np.ndarray:
    scale = peak / (float(np.max(np.abs(wav))) + 1e-8)
    return (wav * scale).astype(np.float32)


def residual_gate(
    mel: torch.Tensor,
    mode: str = "energy_freq",
    energy_lo: float = 0.15,
    freq_lo: float = 0.35,
    freq_hi: float = 1.0,
) -> torch.Tensor:
    """Per-frame, per-bin gate for where the residual may act.

    mode:
      "none"        -> all ones (full residual everywhere)
      "energy"      -> speech frames only, silence stays clean
      "energy_freq" -> speech frames, weighted up toward high bins (consonants)

    Returns [1, 80, T] in [0, 1].
    """
    if mode == "none":
        return torch.ones_like(mel)
    energy = mel.exp().mean(dim=1, keepdim=True)  # [1, 1, T]
    emax = energy.amax(dim=-1, keepdim=True).clamp_min(1e-6)
    e = (energy / emax).clamp(0.0, 1.0)
    e_gate = ((e - energy_lo) / (1.0 - energy_lo)).clamp(0.0, 1.0)
    if mode == "energy":
        return e_gate.expand_as(mel)
    ramp = torch.linspace(freq_lo, freq_hi, mel.size(1), device=mel.device).view(1, -1, 1)
    return e_gate * ramp


def convert_chars(normal, normal_sr, chars, dur_model, vocab, vc, bv, device) -> tuple[np.ndarray, np.ndarray, float]:
    """Normal waveform plus character bounds -> stretched waveform and vocoded dysarthric audio."""
    n = len(chars)
    durs = [max(0.05, float(c["end"]) - float(c["start"])) for c in chars]
    feats = []
    for k, ch in enumerate(chars):
        ini, fin, tone = syllable_of(ch["char"])
        feats.append(vocab.encode(
            ini, fin, tone, durs[k], k / max(n - 1, 1), k == 0, k == n - 1, durs[k - 1] if k else 0.0,
        ))
    feat = torch.from_numpy(np.stack(feats)).to(device)
    with torch.no_grad():
        ratio, pause = dur_model(feat)
    ratio_np = ratio.cpu().numpy()
    pause_np = pause.cpu().numpy()
    stretched = render(normal, normal_sr, chars, ratio_np, pause_np, SR)
    mel = wav_to_mel(stretched, device)
    with torch.no_grad():
        pred = vc(mel)
        gate = residual_gate(mel, mode=GATE_MODE)
        pred = mel + RESIDUAL_SCALE * gate * (pred - mel)
    pred_wav = vocode(bv, pred)
    pred_dur = float((ratio_np * np.array(durs) + pause_np).sum())
    return stretched, pred_wav, pred_dur


def match_loudness(wav: np.ndarray, ref: np.ndarray) -> np.ndarray:
    scale = float(np.sqrt((ref ** 2).mean()) + 1e-8) / float(np.sqrt((wav ** 2).mean()) + 1e-8)
    loud = wav * scale
    peak = float(np.max(np.abs(loud)))
    if peak > 0.99:
        loud = loud * (0.99 / peak)
    return loud.astype(np.float32)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dur_ck = torch.load(DUR_CKPT, map_location="cpu", weights_only=False)
    vocab = Vocab.from_dict(dur_ck["vocab"])
    dur_model = CharDuration(vocab.size).to(device).eval()
    dur_model.load_state_dict(dur_ck["model"])

    vc_ck = torch.load(VC_CKPT, map_location="cpu", weights_only=False)
    channels = vc_ck.get("args", {}).get("channels", 192)
    vc = MelVC(channels=channels, dropout=vc_ck.get("args", {}).get("dropout", 0.1)).to(device).eval()
    vc.load_state_dict(vc_ck["model"])

    bv = bigvgan.BigVGAN.from_pretrained(str(BIGVGAN_DIR), use_cuda_kernel=False)
    bv.remove_weight_norm()
    bv = bv.eval().to(device)

    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for split, n in (("train", 3), ("dev", 3)):
        for i, row in enumerate(pick(split, n), 1):
            uid = row["id"]
            folder_src = listen_dir(uid)
            chars = json.loads((folder_src / "normal_chars.json").read_text())["chars"]
            normal, normal_sr = read_wav(folder_src / "normal.wav")
            dys, dys_sr = read_wav(folder_src / "dys.wav")
            stretched, pred_wav, pred_dur = convert_chars(
                normal, normal_sr, chars, dur_model, vocab, vc, bv, device,
            )
            loud = match_loudness(pred_wav, dys)

            folder = OUT / f"{split}_{i:02d}_{uid}"
            folder.mkdir(parents=True, exist_ok=True)
            sf.write(folder / "normal.wav", normal, normal_sr)
            sf.write(folder / "dys.wav", dys, dys_sr)
            sf.write(folder / "stretch.wav", stretched, SR)
            sf.write(folder / "pred_vc_loud.wav", loud, SR)
            item = {
                "split": split,
                "id": uid,
                "text": row["text"],
                "pred_dur": round(pred_dur, 3),
                "dys_dur": round(len(dys) / dys_sr, 3),
                "normal_dur": round(len(normal) / normal_sr, 3),
                "dir": str(folder),
            }
            summary.append(item)
            print(
                f"{split} {uid} {row['text']} pred {pred_dur:.2f}s dys {item['dys_dur']:.2f}s normal {item['normal_dur']:.2f}s",
                flush=True,
            )
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
