#!/usr/bin/env python3
"""Run a few test utterances and export listen folders."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from transformers import HubertModel, Wav2Vec2FeatureExtractor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import MelDecoder, interpolate_feats

MEL_SR = 22050
HOP = 256
N_FFT = 1024
WIN = 1024
N_MELS = 80
FMIN = 0.0
FMAX = 8000.0


def load_wav(path: Path, target_sr: int | None = None):
    y, sr = sf.read(str(path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if target_sr is not None and sr != target_sr:
        import librosa

        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return y, sr


@torch.no_grad()
def predict_mel(hubert, extractor, decoder, spk_emb, wav_16k, device):
    inputs = extractor(wav_16k, sampling_rate=16000, return_tensors="pt")
    h = hubert(inputs.input_values.to(device)).last_hidden_state  # [1, T_h, D]
    # mel length from wav duration at BigVGAN frame rate
    mel_len = max(1, int(round(len(wav_16k) / 16000 * MEL_SR / HOP)))
    h = interpolate_feats(h, mel_len)
    spk = spk_emb.unsqueeze(0)
    mel = decoder(h, spk, padding_mask=None)[0]  # [T, 80]
    return mel.cpu()


def mel_to_wav_griffin(mel: torch.Tensor) -> np.ndarray:
    """Rough vocoder for listening when BigVGAN is unavailable."""
    import librosa

    # mel is log-mel; invert approximately
    mel_np = mel.numpy().T  # [80, T]
    mag = np.exp(mel_np)
    # approximate linear spectrogram via mel filter pseudo-inverse
    mel_basis = librosa.filters.mel(sr=MEL_SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
    inv = np.linalg.pinv(mel_basis)
    spec = np.maximum(inv @ mag, 1e-10)
    wav = librosa.griffinlim(spec, n_iter=64, hop_length=HOP, win_length=WIN, n_fft=N_FFT)
    return wav.astype(np.float32)


def try_load_bigvgan(path: Path, device):
    if not path.exists():
        return None
    try:
        # official bigvgan package if installed
        from bigvgan import BigVGAN

        model = BigVGAN.from_pretrained(str(path)).to(device)
        model.remove_weight_norm()
        model.eval()
        return model
    except Exception as e1:
        try:
            # fallback: some checkpoints expose Generator via torch.hub style
            sys.path.insert(0, str(path))
            import bigvgan as bv

            model = bv.BigVGAN.from_pretrained(str(path)).to(device)
            model.remove_weight_norm()
            model.eval()
            return model
        except Exception as e2:
            print(f"BigVGAN load failed: {e1} | {e2}")
            return None


@torch.no_grad()
def vocode(mel: torch.Tensor, bigvgan, device) -> np.ndarray:
    if bigvgan is None:
        return mel_to_wav_griffin(mel)
    # BigVGAN expects [B, n_mels, T]
    x = mel.T.unsqueeze(0).to(device)
    wav = bigvgan(x).squeeze().float().cpu().numpy()
    return wav.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("/cmy/cmy/dysa/runs/dsr/train/best.pt"))
    ap.add_argument("--hubert", type=Path, default=Path("/cmy/cmy/dysa/chinese-hubert-base"))
    ap.add_argument("--test-manifest", type=Path, default=Path("/cmy/cmy/dysa/normal_stretched_targets/test.jsonl"))
    ap.add_argument("--bigvgan", type=Path, default=Path("/cmy/cmy/dysa/bigvgan_v2_22khz_80band_256x"))
    ap.add_argument("--out-dir", type=Path, default=Path("/cmy/cmy/dysa/runs/dsr/listen_pred"))
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    conf = ckpt.get("args", {})
    spk_emb = ckpt["spk_emb"].float().to(device)

    extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(args.hubert))
    hubert = HubertModel.from_pretrained(str(args.hubert)).to(device).eval()
    for p in hubert.parameters():
        p.requires_grad = False

    state = ckpt["decoder"]
    n_chars = 0
    if "char_head.weight" in state:
        n_chars = int(state["char_head.weight"].shape[0])
    decoder = MelDecoder(
        hubert_dim=hubert.config.hidden_size,
        spk_dim=int(spk_emb.numel()),
        d_model=conf.get("d_model", 512),
        n_layers=conf.get("n_layers", 4),
        n_conv=conf.get("n_conv", 3),
        n_heads=conf.get("n_heads", 8),
        ffn_dim=conf.get("ffn_dim", 2048),
        dropout=0.0,
        n_chars=n_chars,
    ).to(device)
    decoder.load_state_dict(state)
    decoder.eval()

    bigvgan = try_load_bigvgan(args.bigvgan, device)
    vocoder_name = "bigvgan" if bigvgan is not None else "griffin_lim"
    print(f"vocoder: {vocoder_name}")

    rows = [json.loads(l) for l in open(args.test_manifest)][: args.n]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    index = []

    for i, r in enumerate(rows, 1):
        uid = r["id"]
        folder = args.out_dir / f"{i:02d}_{uid}"
        folder.mkdir(parents=True, exist_ok=True)

        dys, _ = load_wav(Path(r["dys_wav"]), 16000)
        mel = predict_mel(hubert, extractor, decoder, spk_emb, dys, device)
        pred_wav = vocode(mel, bigvgan, device)

        # also vocode target mel for fair comparison under same vocoder
        tgt_mel = torch.from_numpy(np.load(r["mel_path"]).astype(np.float32))
        tgt_voc = vocode(tgt_mel, bigvgan, device)

        sf.write(folder / "dys.wav", dys, 16000)
        sf.write(folder / "pred.wav", pred_wav, MEL_SR)
        sf.write(folder / "target_mel_vocoded.wav", tgt_voc, MEL_SR)
        # original synthetic normal (24k) if available via listen folder
        listen = Path(r["dys_wav"]).parent
        normal = listen / "normal.wav"
        if normal.exists():
            nwav, nsr = load_wav(normal)
            sf.write(folder / "normal.wav", nwav, nsr)
        stretched = Path(r["normal_stretched_wav"])
        if stretched.exists():
            sw, ssr = load_wav(stretched)
            sf.write(folder / "normal_stretched.wav", sw, ssr)

        np.save(folder / "pred_mel.npy", mel.numpy())
        meta = {
            "id": uid,
            "text": r["text"],
            "vocoder": vocoder_name,
            "ckpt_epoch": ckpt.get("epoch"),
            "best_dev_l1": ckpt.get("best_dev_l1"),
            "files": {
                "dys": "dys.wav",
                "pred": "pred.wav",
                "target_mel_vocoded": "target_mel_vocoded.wav",
                "normal": "normal.wav" if normal.exists() else None,
                "normal_stretched": "normal_stretched.wav" if stretched.exists() else None,
            },
        }
        with open(folder / "meta.json", "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        index.append(meta)
        print(f"[{i}/{len(rows)}] {uid}  {r['text']}  pred={len(pred_wav)/MEL_SR:.2f}s")

    with open(args.out_dir / "index.json", "w") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"done -> {args.out_dir}")


if __name__ == "__main__":
    main()
