#!/usr/bin/env python3
"""Vocode a few normal→dysarthric VC outputs for listening.

Each folder has:
  normal_stretched.wav  source (time-aligned normal)
  dys.wav               real S06
  pred_vc.wav           BigVGAN(converted mel)
  src_voc.wav           BigVGAN(source mel)   vocoder reference
  tgt_voc.wav           BigVGAN(real dys mel) vocoder reference
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path("/cmy/cmy/dysa")
sys.path.insert(0, str(ROOT / "index-tts-full"))
sys.path.insert(0, str(ROOT / "dsr"))

from indextts.s2mel.modules.bigvgan import bigvgan
from vc_model import MelVC

CACHE = ROOT / "runs/dsr/vc_cache"
MANIFEST = ROOT / "normal_stretched_targets"
CKPT = ROOT / "runs/dsr/train_vc/best.pt"
OUT = ROOT / "runs/dsr/vc_listen"
BIGVGAN_DIR = ROOT / "bigvgan_v2_22khz_80band_256x"
SR = 22050


def pick(manifest: Path, split: str, n: int) -> list[dict]:
    rows = [json.loads(l) for l in open(manifest / f"{split}.jsonl")]
    # spread across the split instead of taking only the first n
    if len(rows) <= n:
        return rows
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n)]


@torch.no_grad()
def vocode(bv, mel_tc: np.ndarray, device) -> np.ndarray:
    mel = torch.from_numpy(mel_tc).T.unsqueeze(0).to(device)  # [1, 80, T]
    wav = bv(mel.float().detach().clone()).squeeze().float().cpu().numpy()
    peak = np.max(np.abs(wav)) + 1e-8
    if peak > 1:
        wav = wav / peak
    return wav.astype(np.float32)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=CKPT)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--highlight-id", default="")
    ap.add_argument("--highlight-dir", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    channels = ckpt.get("args", {}).get("channels", 192)
    dropout = ckpt.get("args", {}).get("dropout", 0.1)
    model = MelVC(channels=channels, dropout=dropout).to(device).eval()
    model.load_state_dict(ckpt["model"])

    bv = bigvgan.BigVGAN.from_pretrained(str(BIGVGAN_DIR), use_cuda_kernel=False)
    bv.remove_weight_norm()
    bv = bv.eval().to(device)

    args.out.mkdir(parents=True, exist_ok=True)
    summary = []
    for split, n in (("train", 3), ("dev", 3)):
        for i, row in enumerate(pick(args.manifest, split, n), 1):
            uid = row["id"]
            z = np.load(args.cache / split / f"{uid}.npz")
            src = torch.from_numpy(z["src"]).T.unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(src).squeeze(0).T.cpu().numpy()
            folder = args.out / f"{split}_{i:02d}_{uid}"
            folder.mkdir(parents=True, exist_ok=True)
            normal_wav = Path(row["dys_wav"]).with_name("normal.wav")
            if normal_wav.exists():
                shutil.copy(normal_wav, folder / "normal.wav")
            if Path(row["normal_stretched_wav"]).exists():
                shutil.copy(row["normal_stretched_wav"], folder / "normal_stretched.wav")
            shutil.copy(row["dys_wav"], folder / "dys.wav")
            sf.write(folder / "src_voc.wav", vocode(bv, z["src"], device), SR)
            sf.write(folder / "tgt_voc.wav", vocode(bv, z["tgt"], device), SR)
            pred_wav = vocode(bv, pred, device)
            sf.write(folder / "pred_vc.wav", pred_wav, SR)
            dys_wav, _ = sf.read(row["dys_wav"], dtype="float32")
            if dys_wav.ndim > 1:
                dys_wav = dys_wav.mean(1)
            scale = float(np.sqrt((dys_wav ** 2).mean()) + 1e-8) / float(np.sqrt((pred_wav ** 2).mean()) + 1e-8)
            loud = pred_wav * scale
            peak = float(np.max(np.abs(loud)))
            if peak > 0.99:
                loud = loud * (0.99 / peak)
            sf.write(folder / "pred_vc_loud.wav", loud.astype(np.float32), SR)
            l1 = float(np.abs(pred - z["tgt"]).mean())
            base = float(np.abs(z["src"] - z["tgt"]).mean())
            changed = np.abs(z["src"] - z["tgt"]).mean(1) > 0.05
            std_ratio = float(pred[changed].std() / (z["tgt"][changed].std() + 1e-8)) if changed.any() else 1.0
            summary.append({"split": split, "id": uid, "text": row["text"], "l1": l1, "identity_l1": base, "std_ratio": std_ratio, "dir": str(folder)})
            print(f"{split} {uid} {row['text']} l1 {l1:.3f} identity {base:.3f} std_ratio {std_ratio:.2f}", flush=True)
    if args.highlight_id and args.highlight_dir is not None:
        found = None
        for split in ("train", "dev", "test"):
            path = args.cache / split / f"{args.highlight_id}.npz"
            if path.exists():
                found = (split, path)
                break
        if found is None:
            raise FileNotFoundError(args.highlight_id)
        split, path = found
        row = next(r for r in (json.loads(l) for l in open(args.manifest / f"{split}.jsonl")) if r["id"] == args.highlight_id)
        z = np.load(path)
        src = torch.from_numpy(z["src"]).T.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(src).squeeze(0).T.cpu().numpy()
        folder = args.highlight_dir
        folder.mkdir(parents=True, exist_ok=True)
        normal_wav = Path(row["dys_wav"]).with_name("normal.wav")
        if normal_wav.exists():
            shutil.copy(normal_wav, folder / "normal.wav")
        pred_wav = vocode(bv, pred, device)
        ref, ref_sr = sf.read(normal_wav if normal_wav.exists() else row["dys_wav"], dtype="float32")
        if ref.ndim > 1:
            ref = ref.mean(1)
        scale = float(np.sqrt((ref ** 2).mean()) + 1e-8) / float(np.sqrt((pred_wav ** 2).mean()) + 1e-8)
        loud = pred_wav * scale
        peak = float(np.max(np.abs(loud)))
        if peak > 0.99:
            loud = loud * (0.99 / peak)
        sf.write(folder / "pred_vc_loud.wav", loud.astype(np.float32), SR)
        tgt_wav = vocode(bv, z["tgt"], device)
        scale_t = float(np.sqrt((ref ** 2).mean()) + 1e-8) / float(np.sqrt((tgt_wav ** 2).mean()) + 1e-8)
        tgt_loud = tgt_wav * scale_t
        peak = float(np.max(np.abs(tgt_loud)))
        if peak > 0.99:
            tgt_loud = tgt_loud * (0.99 / peak)
        sf.write(folder / "tgt_vc_loud.wav", tgt_loud.astype(np.float32), SR)
        changed = np.abs(z["src"] - z["tgt"]).mean(1) > 0.05
        std_ratio = float(pred[changed].std() / (z["tgt"][changed].std() + 1e-8))
        print(
            f"highlight {args.highlight_id} l1 {np.abs(pred-z['tgt']).mean():.3f} "
            f"identity {np.abs(z['src']-z['tgt']).mean():.3f} std_ratio {std_ratio:.2f} -> {folder}",
            flush=True,
        )
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("done ->", args.out)


if __name__ == "__main__":
    main()
