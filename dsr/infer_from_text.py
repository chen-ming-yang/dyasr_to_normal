#!/usr/bin/env python3
"""Text -> S06 dysarthric audio.

Qwen3-TTS clones a normal reading in S06's voice, Qwen3-ForcedAligner cuts
the characters, then the duration model, MelVC and BigVGAN do the conversion.
No dysarthric recording is read.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path("/cmy/cmy/dysa")
sys.path.insert(0, str(ROOT / "index-tts-full"))
sys.path.insert(0, str(ROOT / "dsr"))
sys.path.insert(0, str(ROOT / "scripts"))

from indextts.s2mel.modules.bigvgan import bigvgan
from duration import SR, CharDuration, Vocab
from infer_from_normal import (
    BIGVGAN_DIR,
    DUR_CKPT,
    VC_CKPT,
    convert_chars,
    peak_norm,
)
from vc_model import MelVC

TTS_DIR = ROOT / "Qwen3-TTS-12Hz-0.6B-Base"
ALIGNER_DIR = Path("/cmy/migration-gpu1-20261113/Liu/model/qwen3asr/Qwen3-ForcedAligner-0.6B")
REF_WAV = Path("/cmy/cmy/dysa/同事新标注_06_part1_合并/Audio/06/S006T004E000N00019.wav")


def _listen_samples():
    spec = importlib.util.spec_from_file_location(
        "build_listen_samples", ROOT / "scripts" / "build_listen_samples.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def han_chars(text: str) -> str:
    return "".join(ch for ch in text if "\u4e00" <= ch <= "\u9fff")


def align_text(aligner, wav_path: Path, text: str) -> list[dict]:
    mod = _listen_samples()
    parsed = mod.align_batch(aligner, [str(wav_path)], [han_chars(text)])[0]
    return mod.reconcile_chars(han_chars(text), parsed)


def load_converters(device):
    dur_ck = torch.load(DUR_CKPT, map_location="cpu", weights_only=False)
    vocab = Vocab.from_dict(dur_ck["vocab"])
    dur_model = CharDuration(vocab.size).to(device).eval()
    dur_model.load_state_dict(dur_ck["model"])
    vc_ck = torch.load(VC_CKPT, map_location="cpu", weights_only=False)
    vc = MelVC(
        channels=vc_ck.get("args", {}).get("channels", 192),
        dropout=vc_ck.get("args", {}).get("dropout", 0.1),
    ).to(device).eval()
    vc.load_state_dict(vc_ck["model"])
    bv = bigvgan.BigVGAN.from_pretrained(str(BIGVGAN_DIR), use_cuda_kernel=False)
    bv.remove_weight_norm()
    bv = bv.eval().to(device)
    return dur_model, vocab, vc, bv


def synthesize_normal(tts, text: str, ref_wav: Path) -> tuple[np.ndarray, int]:
    wavs, sr = tts.generate_voice_clone(
        text=text,
        language="Chinese",
        ref_audio=str(ref_wav),
        x_vector_only_mode=True,
    )
    wav = np.asarray(wavs[0], dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    return wav, int(sr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", action="append", required=True, help="repeat for several sentences")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/dsr/vc_listen_text")
    ap.add_argument("--ref-wav", type=Path, default=REF_WAV)
    ap.add_argument("--tts", type=Path, default=TTS_DIR)
    ap.add_argument("--aligner", type=Path, default=ALIGNER_DIR)
    args = ap.parse_args()
    if not args.ref_wav.exists():
        raise FileNotFoundError(args.ref_wav)
    if not args.tts.exists():
        raise FileNotFoundError(args.tts)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from qwen_tts import Qwen3TTSModel

    print("loading TTS", args.tts, flush=True)
    tts = Qwen3TTSModel.from_pretrained(str(args.tts), device_map="cuda:0", dtype=torch.bfloat16)
    normals = []
    args.out.mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(args.text, 1):
        wav, sr = synthesize_normal(tts, text, args.ref_wav)
        folder = args.out / f"{i:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        sf.write(folder / "normal.wav", wav, sr)
        normals.append((text, folder, wav, sr))
        print(f"tts {i} {text} {len(wav)/sr:.2f}s", flush=True)
    del tts
    torch.cuda.empty_cache()

    from qwen_asr import Qwen3ForcedAligner

    print("loading aligner", args.aligner, flush=True)
    aligner = Qwen3ForcedAligner.from_pretrained(str(args.aligner), dtype=torch.bfloat16, device_map="cuda:0")
    aligned = []
    for text, folder, wav, sr in normals:
        chars = align_text(aligner, folder / "normal.wav", text)
        (folder / "normal_chars.json").write_text(json.dumps({
            "text": text,
            "chars": chars,
        }, ensure_ascii=False, indent=2))
        aligned.append((text, folder, wav, sr, chars))
        print(f"aligned {text} chars={len(chars)}", flush=True)
    del aligner
    torch.cuda.empty_cache()

    dur_model, vocab, vc, bv = load_converters(device)
    summary = []
    for text, folder, wav, sr, chars in aligned:
        stretched, pred, pred_dur = convert_chars(wav, sr, chars, dur_model, vocab, vc, bv, device)
        loud = peak_norm(pred)
        sf.write(folder / "stretch.wav", stretched, SR)
        sf.write(folder / "pred_vc_loud.wav", loud, SR)
        summary.append({
            "text": text,
            "normal_dur": round(len(wav) / sr, 3),
            "pred_dur": round(pred_dur, 3),
            "dir": str(folder),
        })
        print(f"dys {text} pred {pred_dur:.2f}s -> {folder}", flush=True)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("done ->", args.out, flush=True)


if __name__ == "__main__":
    main()
