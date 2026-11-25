#!/usr/bin/env python3
"""Train normal→dysarthric mel VC with a multi-resolution STFT loss through BigVGAN.

Plain mel-L1 over-smooths the spectrogram, so the GAN vocoder falls back to a
periodic excitation and sounds metallic ("电音"). Decoding the predicted mel
with the frozen BigVGAN and matching the real dysarthric waveform in STFT space
optimises exactly the thing the ear hears, restoring aperiodicity and killing
the buzz.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

ROOT = Path("/cmy/cmy/dysa")
sys.path.insert(0, str(ROOT / "index-tts-full"))
sys.path.insert(0, str(ROOT / "dsr"))

from indextts.s2mel.modules.bigvgan import bigvgan
from stft_loss import MultiResolutionSTFTLoss
from vc_model import MelVC, conversion_loss

CACHE = ROOT / "runs/dsr/vc_cache_v3"
SR = 22050
HOP = 256
FRAME_CROP = 256  # -> 256*256 = 65536 samples, BigVGAN's own segment size


class VocoderDataset(Dataset):
    def __init__(self, split: str):
        self.files = sorted((CACHE / split).glob("*.npz"))
        self.ids = [f.stem for f in self.files]
        if not self.files:
            raise FileNotFoundError(CACHE / split)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        uid = self.ids[idx]
        z = np.load(self.files[idx])
        src = torch.from_numpy(z["src"])  # [T, 80]
        tgt = torch.from_numpy(z["tgt"])
        listen = next(Path("/cmy/cmy/dysa/listen_samples_all").glob(f"*_{uid}"))
        wav, sr = sf.read(str(listen / "dys.wav"), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        if sr != SR:
            wav = torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0), sr, SR).squeeze(0).numpy()
        wav = wav.astype(np.float32)
        return src, tgt, wav


def collate(batch):
    srcs, tgts, wavs = zip(*batch)
    n = min(s.size(0) for s in srcs)
    # crop to a common time crop (min length in the batch, capped)
    crop = min(n, FRAME_CROP)
    starts = [random.randint(0, s.size(0) - crop) if s.size(0) > crop else 0 for s in srcs]
    src = torch.stack([s[st : st + crop].T for s, st in zip(srcs, starts)])
    tgt = torch.stack([t[st : st + crop].T for t, st in zip(tgts, starts)])
    # aligned wav segment: frame st*HOP to (st+crop)*HOP
    wav_segs = []
    for w, st in zip(wavs, starts):
        a = st * HOP
        b = a + crop * HOP
        seg = w[a:b]
        if len(seg) < crop * HOP:
            seg = np.pad(seg, (0, crop * HOP - len(seg)))
        wav_segs.append(torch.from_numpy(seg))
    wav = torch.stack(wav_segs).unsqueeze(1)  # [B, 1, N]
    mask = torch.ones(src.size(0), 1, crop, dtype=torch.bool)
    return src, tgt, wav, mask


@torch.no_grad()
def evaluate(model, vocoder, loader, stft_loss, device):
    model.eval()
    n = 0
    mel_l1 = 0.0
    stft = 0.0
    for src, tgt, wav, mask in loader:
        src = src.to(device)
        tgt = tgt.to(device)
        wav = wav.to(device)
        pred = model(src)
        mel_l1 += ((pred - tgt).abs() * mask.to(device)).sum().item() / (mask.sum().item() * 80)
        pred_wav = vocoder(pred.float())
        loss, _ = stft_loss(pred_wav, wav)
        stft += loss.item()
        n += 1
    return {"mel_l1": float(mel_l1 / n), "stft": float(stft / n)}


def main():
    global CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "runs/dsr/train_vc_stft")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--channels", type=int, default=192)
    ap.add_argument("--stft-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--cache", type=Path, default=CACHE)
    args = ap.parse_args()

    CACHE = args.cache

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    train_ds = VocoderDataset("train")
    dev_ds = VocoderDataset("dev")
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, collate_fn=collate, pin_memory=True, drop_last=True)
    dev_loader = DataLoader(dev_ds, batch_size=8, shuffle=False, num_workers=2, collate_fn=collate, pin_memory=True)

    model = MelVC(channels=args.channels).to(device)
    if args.resume is not None:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        print("resumed from", args.resume)
    vocoder = bigvgan.BigVGAN.from_pretrained(str(ROOT / "bigvgan_v2_22khz_80band_256x"), use_cuda_kernel=False)
    vocoder.remove_weight_norm()
    vocoder = vocoder.to(device).eval()
    for p in vocoder.parameters():
        p.requires_grad = False

    stft_loss = MultiResolutionSTFTLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    log_path = args.out / "train_log.jsonl"
    best = 1e9
    bad = 0
    best_path = args.out / "best.pt"
    print(f"device={device} train={len(train_ds)} dev={len(dev_ds)}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        steps = 0
        opt.zero_grad(set_to_none=True)
        for bi, (src, tgt, wav, mask) in enumerate(train_loader):
            src = src.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            wav = wav.to(device, non_blocking=True)
            pred = model(src)
            mel_loss = conversion_loss(pred, tgt, mask.to(device))["loss"]
            pred_wav = vocoder(pred.float())
            stft, _ = stft_loss(pred_wav, wav)
            loss = (mel_loss + args.stft_weight * stft) / args.accum
            loss.backward()
            running += loss.item() * args.accum
            steps += 1
            if (bi + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
        sched.step()
        dev = evaluate(model, vocoder, dev_loader, stft_loss, device)
        score = dev["stft"] + 0.1 * dev["mel_l1"]
        row = {"epoch": epoch, "train_loss": running / max(steps, 1), "dev_mel_l1": dev["mel_l1"], "dev_stft": dev["stft"], "score": score, "sec": round(time.time() - t0, 2)}
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"epoch {epoch:03d} train {row['train_loss']:.4f} dev_mel_l1 {dev['mel_l1']:.4f} dev_stft {dev['stft']:.4f} {row['sec']:.1f}s", flush=True)
        torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "dev": dev}, args.out / "last.pt")
        if score < best - 1e-4:
            best = score
            bad = 0
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "dev": dev}, best_path)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {epoch}, best score {best:.4f}", flush=True)
                break
    print(f"best score {best:.4f} -> {best_path}", flush=True)


if __name__ == "__main__":
    main()
