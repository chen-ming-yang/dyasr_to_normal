"""Dataset for S06 dysarthric -> stretched-normal mel training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

MEL_SR = 22050
HOP = 256
MIN_SPAN_FRAMES = 2


def speech_frame_spans(chars: list[dict], mel_len: int) -> list[tuple[int, int, str]]:
    """Speech-only spans on the mel axis. Silence pads are excluded."""
    spans = []
    for c in chars:
        start = float(c["dys_start"])
        end = start + float(c["dys_dur"]) - float(c.get("silence_pad_sec", 0.0))
        i0 = int(round(start * MEL_SR / HOP))
        i1 = int(round(end * MEL_SR / HOP))
        i0 = max(0, min(i0, mel_len))
        i1 = max(0, min(i1, mel_len))
        if i1 - i0 < MIN_SPAN_FRAMES:
            continue
        spans.append((i0, i1, c["char"]))
    return spans


class DsrMelDataset(Dataset):
    def __init__(self, manifest: Path, max_seconds: float = 20.0):
        self.rows = []
        with open(manifest) as f:
            for line in f:
                self.rows.append(json.loads(line))
        self.max_seconds = max_seconds

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        wav, sr = sf.read(r["dys_wav"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            raise ValueError(f"{r['id']}: expected 16 kHz dys wav, got {sr}")
        max_len = int(self.max_seconds * sr)
        if len(wav) > max_len:
            wav = wav[:max_len]
        mel = np.load(r["mel_path"]).astype(np.float32)  # [T, 80]
        # trim mel to match truncated wav roughly
        hop = 256
        mel_sr = 22050
        max_mel = int(round(len(wav) / 16000 * mel_sr / hop))
        if mel.shape[0] > max_mel > 0:
            mel = mel[:max_mel]
        meta_path = Path(r["mel_path"]).with_name("meta.json")
        chars = json.load(open(meta_path))["chars"]
        spans = speech_frame_spans(chars, mel.shape[0])
        return {
            "id": r["id"],
            "wav": torch.from_numpy(wav),
            "mel": torch.from_numpy(mel),
            "wav_len": len(wav),
            "mel_len": mel.shape[0],
            "spans": spans,
        }


def collate_batch(batch: list[dict]):
    ids = [b["id"] for b in batch]
    wavs = [b["wav"] for b in batch]
    mels = [b["mel"] for b in batch]
    wav_lens = torch.tensor([b["wav_len"] for b in batch], dtype=torch.long)
    mel_lens = torch.tensor([b["mel_len"] for b in batch], dtype=torch.long)

    wav_pad = pad_sequence(wavs, batch_first=True)  # [B, T]
    mel_pad = pad_sequence(mels, batch_first=True)  # [B, T, 80]
    mel_padding = (
        torch.arange(mel_pad.size(1), device=mel_pad.device)[None, :]
        >= mel_lens[:, None]
    )
    return {
        "ids": ids,
        "wav": wav_pad,
        "wav_lens": wav_lens,
        "mel": mel_pad,
        "mel_lens": mel_lens,
        "mel_padding": mel_padding,
        "spans": [b["spans"] for b in batch],
    }
