#!/usr/bin/env python3
"""Encode paired normal/dysarthric audio into Qwen3-TTS codes, sliced by character."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/cmy/cmy/dysa")
sys.path.insert(0, str(ROOT / "dsr"))

from codec_adapter import HOP, SR, sec_to_code
from duration import content_end, listen_dir, load_rows, read_wav, trim_phone

TTS_DIR = ROOT / "Qwen3-TTS-12Hz-0.6B-Base"
OUT = ROOT / "runs/dsr/codec_pairs"


def codes_of(tokenizer, wav: np.ndarray, sr: int) -> np.ndarray:
    enc = tokenizer.encode(wav, sr=sr)
    codes = enc.audio_codes[0].detach().cpu().to(torch.int16).numpy()
    return np.ascontiguousarray(codes)


def main():
    from qwen_tts import Qwen3TTSModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen3TTSModel.from_pretrained(str(TTS_DIR), device_map="cuda:0" if device == "cuda" else "cpu", dtype=torch.bfloat16)
    tokenizer = model.model.speech_tokenizer
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev"):
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
            src = codes_of(tokenizer, normal, nsr)
            tgt = codes_of(tokenizer, dys, dsr)
            dys_end = content_end(dys, dsr)
            ends = [row["onsets"][j + 1] if j + 1 < len(row["onsets"]) else dys_end for j in range(len(row["onsets"]))]
            rec = []
            for ch, t0, t1 in zip(chars, row["onsets"], ends):
                sa, sb = sec_to_code(float(ch["start"])), sec_to_code(float(ch["end"]))
                ta, tb = sec_to_code(float(t0)), sec_to_code(float(t1))
                sa, sb = int(np.clip(sa, 0, len(src))), int(np.clip(sb, 0, len(src)))
                ta, tb = int(np.clip(ta, 0, len(tgt))), int(np.clip(tb, 0, len(tgt)))
                i0 = int(np.clip(round(t0 * dsr), 0, len(dys)))
                i1 = int(np.clip(round(max(t1, t0) * dsr), 0, len(dys)))
                voiced = trim_phone(dys[i0:i1], dsr)
                speech_n = min(tb - ta, max(1, int(round(len(voiced) / HOP))))
                if sb - sa < 1 or tb - ta < 1:
                    continue
                rec.append((sa, sb, ta, tb, speech_n))
            if not rec:
                continue
            arr = np.array(rec, dtype=np.int32)
            np.savez(
                dest / f"{uid}.npz",
                src=src.astype(np.int16),
                tgt=tgt.astype(np.int16),
                spans=arr,
            )
            if i % 20 == 0 or i == len(rows):
                print(f"{split} {i}/{len(rows)} {uid} src={len(src)} tgt={len(tgt)} chars={len(rec)}", flush=True)
    print("done", OUT, flush=True)


if __name__ == "__main__":
    main()
