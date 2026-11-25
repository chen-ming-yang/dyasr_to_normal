#!/usr/bin/env python3
"""Normal audio -> dysarthric waveform through the codec adapter.

Character bounds come from the normal-audio aligner. How many codes each
character gets comes from the duration model, which was trained on the
dysarthric onsets. The Qwen decoder turns the predicted codes into a waveform.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path("/cmy/cmy/dysa")
sys.path.insert(0, str(ROOT / "dsr"))

from codec_adapter import HOP, SR, CodecAdapter, sec_to_code
from duration import CharDuration, Vocab, read_wav, syllable_of
from infer_from_normal import DUR_CKPT, peak_norm

TTS_DIR = ROOT / "Qwen3-TTS-12Hz-0.6B-Base"
CKPT = ROOT / "runs/dsr/train_codec_adapter/best.pt"


@torch.no_grad()
def predict_char(model, src_seg: np.ndarray, speech_n: int, pause_n: int, device) -> np.ndarray:
    total = max(1, speech_n + pause_n)
    speech_n = min(speech_n, total)
    index = np.zeros(total, dtype=np.int64)
    progress = np.ones(total, dtype=np.float32)
    is_pause = np.ones(total, dtype=np.bool_)
    if speech_n > 0 and len(src_seg) > 0:
        index[:speech_n] = np.linspace(0, len(src_seg) - 1, speech_n).round().astype(np.int64)
        progress[:speech_n] = np.linspace(0, 1, speech_n, dtype=np.float32)
        is_pause[:speech_n] = False
    src = torch.from_numpy(src_seg.astype(np.int64)).unsqueeze(0).to(device)
    if src.numel() == 0:
        src = torch.zeros(1, 1, 16, dtype=torch.long, device=device)
    logits = model(
        src,
        torch.from_numpy(index).unsqueeze(0).to(device),
        torch.from_numpy(progress).unsqueeze(0).to(device),
        torch.from_numpy(is_pause).unsqueeze(0).to(device),
        torch.ones(1, total, dtype=torch.bool, device=device),
    )
    pred = torch.stack([logit.argmax(-1) for logit in logits], dim=-1)
    return pred.squeeze(0).cpu().numpy().astype(np.int64)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal", type=Path, required=True)
    ap.add_argument("--chars", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from qwen_tts import Qwen3TTSModel

    tts = Qwen3TTSModel.from_pretrained(str(TTS_DIR), device_map="cuda:0", dtype=torch.bfloat16)
    tokenizer = tts.model.speech_tokenizer
    dur_ck = torch.load(DUR_CKPT, map_location="cpu", weights_only=False)
    vocab = Vocab.from_dict(dur_ck["vocab"])
    dur_model = CharDuration(vocab.size).to(device).eval()
    dur_model.load_state_dict(dur_ck["model"])
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    adapter = CodecAdapter().to(device).eval()
    adapter.load_state_dict(ckpt["model"])

    wav, sr = read_wav(args.normal)
    chars = json.loads(args.chars.read_text())["chars"]
    enc = tokenizer.encode(wav, sr=sr)
    src = enc.audio_codes[0].detach().cpu().numpy()
    durs = [max(0.05, float(c["end"]) - float(c["start"])) for c in chars]
    feats = []
    for k, ch in enumerate(chars):
        ini, fin, tone = syllable_of(ch["char"])
        feats.append(vocab.encode(
            ini, fin, tone, durs[k], k / max(len(chars) - 1, 1), k == 0, k == len(chars) - 1,
            durs[k - 1] if k else 0.0,
        ))
    with torch.no_grad():
        ratio, pause = dur_model(torch.from_numpy(np.stack(feats)).to(device))
    ratio, pause = ratio.cpu().numpy(), pause.cpu().numpy()
    pieces = []
    for ch, r, p, n_dur in zip(chars, ratio, pause, durs):
        a, b = sec_to_code(float(ch["start"])), sec_to_code(float(ch["end"]))
        a, b = int(np.clip(a, 0, len(src))), int(np.clip(b, 0, len(src)))
        seg = src[a:b]
        if len(seg) < 1:
            seg = src[:1]
        speech_n = max(1, int(round(float(np.clip(r, 0.3, 2.0)) * max(n_dur, 0.05) * SR / HOP)))
        pause_n = int(round(max(0.0, float(p)) * SR / HOP))
        if pause_n < int(0.10 * SR / HOP):
            pause_n = 0
        pieces.append(predict_char(adapter, seg, speech_n, pause_n, device))
    codes = torch.from_numpy(np.concatenate(pieces)).unsqueeze(0)
    wavs, out_sr = tokenizer.decode({"audio_codes": codes})
    out = peak_norm(np.asarray(wavs[0], dtype=np.float32))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, out, out_sr)
    print(f"{args.out} {len(out)/out_sr:.2f}s frames={codes.shape[1]}", flush=True)


if __name__ == "__main__":
    main()
