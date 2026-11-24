#!/usr/bin/env python3
"""Train S06 frame-synchronous mel decoder (frozen HuBERT + frozen speaker emb)."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import HubertModel, Wav2Vec2FeatureExtractor

from dataset import DsrMelDataset, collate_batch
from model import MelDecoder, interpolate_feats


def masked_l1(pred, target, padding_mask):
    """pred/target [B,T,C]; padding_mask True=pad."""
    mask = (~padding_mask).unsqueeze(-1).float()
    diff = (pred - target).abs() * mask
    denom = mask.sum() * pred.size(-1)
    return diff.sum() / denom.clamp_min(1.0)


def build_char_vocab(dataset) -> list[str]:
    chars = set()
    for row in dataset:
        for _, _, ch in row["spans"]:
            chars.add(ch)
    return sorted(chars)


def char_ce_and_acc(decoder, hidden, spans, mel_lens, char_to_id):
    """Pool decoder hidden over speech spans and classify characters.

    Returns (ce, n_correct, n_chars). CE is 0 and detached-safe if no spans.
    """
    pooled = []
    labels = []
    for b, sample_spans in enumerate(spans):
        limit = int(mel_lens[b].item())
        for i0, i1, ch in sample_spans:
            idx = char_to_id.get(ch)
            if idx is None:
                continue
            i0 = max(0, min(int(i0), limit))
            i1 = max(0, min(int(i1), limit))
            if i1 - i0 < 2:
                continue
            pooled.append(hidden[b, i0:i1].mean(dim=0))
            labels.append(idx)
    if not pooled or decoder.char_head is None:
        zero = hidden.sum() * 0.0
        return zero, 0, 0
    logits = decoder.char_head(torch.stack(pooled, dim=0))
    target = torch.tensor(labels, device=hidden.device, dtype=torch.long)
    ce = F.cross_entropy(logits, target)
    correct = int((logits.argmax(dim=-1) == target).sum().item())
    return ce, correct, len(labels)


@torch.no_grad()
def encode_hubert(hubert, extractor, wav, wav_lens, device):
    """wav [B,T] float32 on device -> list of [T_h, D] then pad & return lens."""
    # Feature extractor expects cpu numpy / list; do per-sample for variable length
    feats = []
    lengths = []
    for i in range(wav.size(0)):
        n = int(wav_lens[i].item())
        arr = wav[i, :n].detach().cpu().numpy()
        inputs = extractor(arr, sampling_rate=16000, return_tensors="pt")
        h = hubert(inputs.input_values.to(device)).last_hidden_state[0]  # [T_h, D]
        feats.append(h)
        lengths.append(h.size(0))
    max_t = max(lengths)
    d = feats[0].size(-1)
    out = wav.new_zeros(len(feats), max_t, d)
    for i, f in enumerate(feats):
        out[i, : f.size(0)] = f
    return out, torch.tensor(lengths, device=device, dtype=torch.long)


def run_epoch(
    loader,
    hubert,
    extractor,
    decoder,
    spk_emb,
    optimizer,
    device,
    train: bool,
    args,
    char_to_id: dict[str, int],
):
    decoder.train(train)
    total_l1 = 0.0
    total_frames = 0
    total_ce = 0.0
    total_correct = 0
    total_chars = 0
    use_char = args.char_loss_weight > 0 and decoder.char_head is not None
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            wav = batch["wav"].to(device)
            wav_lens = batch["wav_lens"].to(device)
            mel = batch["mel"].to(device)
            mel_pad = batch["mel_padding"].to(device)
            mel_lens = batch["mel_lens"].to(device)

            with torch.no_grad():
                h, _ = encode_hubert(hubert, extractor, wav, wav_lens, device)
                # interpolate each sample to its mel length, then re-pad
                synced = []
                for i in range(h.size(0)):
                    t_mel = int(mel_lens[i].item())
                    synced.append(interpolate_feats(h[i : i + 1], t_mel)[0])
                from torch.nn.utils.rnn import pad_sequence

                h_sync = pad_sequence(synced, batch_first=True)
                # ensure matches mel time
                if h_sync.size(1) < mel.size(1):
                    h_sync = F.pad(h_sync, (0, 0, 0, mel.size(1) - h_sync.size(1)))
                elif h_sync.size(1) > mel.size(1):
                    h_sync = h_sync[:, : mel.size(1)]

            spk = spk_emb.unsqueeze(0).expand(wav.size(0), -1)
            pred, hidden = decoder(
                h_sync,
                spk,
                padding_mask=mel_pad,
                time_mask_prob=args.time_mask_prob if train else 0.0,
                time_mask_width=args.time_mask_width,
                return_hidden=True,
            )
            l1 = masked_l1(pred, mel, mel_pad)
            if use_char:
                ce, n_correct, n_chars = char_ce_and_acc(
                    decoder, hidden, batch["spans"], mel_lens, char_to_id
                )
                loss = l1 + args.char_loss_weight * ce
            else:
                ce = l1.new_zeros(())
                n_correct, n_chars = 0, 0
                loss = l1

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.grad_clip)
                optimizer.step()

            n_frames = int((~mel_pad).sum().item())
            total_l1 += l1.item() * n_frames
            total_frames += n_frames
            if n_chars:
                total_ce += ce.item() * n_chars
                total_correct += n_correct
                total_chars += n_chars
    return {
        "l1": total_l1 / max(total_frames, 1),
        "ce": total_ce / max(total_chars, 1),
        "char_acc": total_correct / max(total_chars, 1),
        "n_chars": total_chars,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("/cmy/cmy/dysa/normal_stretched_targets"))
    ap.add_argument("--hubert", type=Path, default=Path("/cmy/cmy/dysa/chinese-hubert-base"))
    ap.add_argument("--spk-emb", type=Path, default=Path("/cmy/cmy/dysa/runs/dsr/s06_spk_emb.pt"))
    ap.add_argument("--out-dir", type=Path, default=Path("/cmy/cmy/dysa/runs/dsr/train_char_aux"))
    ap.add_argument("--char-loss-weight", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-conv", type=int, default=3)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--ffn-dim", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--time-mask-prob", type=float, default=0.3)
    ap.add_argument("--time-mask-width", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.spk_emb.exists():
        raise SystemExit(f"missing speaker emb: {args.spk_emb} (run extract_spk_emb.py first)")

    spk_pack = torch.load(args.spk_emb, map_location="cpu", weights_only=False)
    spk_emb = spk_pack["embedding"].float().to(device)
    spk_dim = int(spk_emb.numel())
    print(f"speaker emb dim={spk_dim} source={spk_pack.get('source')}")

    print(f"loading HuBERT from {args.hubert}")
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(args.hubert))
    hubert = HubertModel.from_pretrained(str(args.hubert)).to(device)
    hubert.eval()
    for p in hubert.parameters():
        p.requires_grad = False
    hubert_dim = hubert.config.hidden_size

    train_ds = DsrMelDataset(args.data_root / "train.jsonl")
    dev_ds = DsrMelDataset(args.data_root / "dev.jsonl")
    char_vocab = build_char_vocab(train_ds) if args.char_loss_weight > 0 else []
    char_to_id = {c: i for i, c in enumerate(char_vocab)}
    print(f"char vocab: {len(char_vocab)}")

    decoder = MelDecoder(
        hubert_dim=hubert_dim,
        spk_dim=spk_dim,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_conv=args.n_conv,
        n_heads=args.n_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        n_chars=len(char_vocab),
    ).to(device)
    n_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"decoder trainable params: {n_params/1e6:.2f}M")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    conf = vars(args).copy()
    conf["data_root"] = str(args.data_root)
    conf["hubert"] = str(args.hubert)
    conf["spk_emb"] = str(args.spk_emb)
    conf["out_dir"] = str(args.out_dir)
    conf["spk_source"] = spk_pack.get("source")
    conf["hubert_dim"] = hubert_dim
    conf["spk_dim"] = spk_dim
    conf["n_params"] = n_params
    conf["n_chars"] = len(char_vocab)
    conf["char_vocab"] = char_vocab
    with open(args.out_dir / "config.json", "w") as f:
        json.dump(conf, f, indent=2)

    best = math.inf
    bad = 0
    history = []
    log_path = args.out_dir / "train_log.jsonl"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(
            train_loader, hubert, extractor, decoder, spk_emb, optimizer, device, True, args, char_to_id
        )
        va = run_epoch(
            dev_loader, hubert, extractor, decoder, spk_emb, None, device, False, args, char_to_id
        )
        dt = time.time() - t0
        row = {
            "epoch": epoch,
            "train_l1": tr["l1"],
            "dev_l1": va["l1"],
            "train_ce": tr["ce"],
            "dev_ce": va["ce"],
            "train_char_acc": tr["char_acc"],
            "dev_char_acc": va["char_acc"],
            "sec": round(dt, 1),
        }
        history.append(row)
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"epoch {epoch:03d}  train_l1={tr['l1']:.4f}  dev_l1={va['l1']:.4f}  "
            f"train_char_acc={tr['char_acc']:.3f}  dev_char_acc={va['char_acc']:.3f}  ({dt:.1f}s)",
            flush=True,
        )

        ckpt = {
            "epoch": epoch,
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_dev_l1": best,
            "args": conf,
            "spk_emb": spk_emb.detach().cpu(),
            "char_vocab": char_vocab,
        }
        torch.save(ckpt, args.out_dir / "last.pt")
        if va["l1"] < best - 1e-4:
            best = va["l1"]
            bad = 0
            ckpt["best_dev_l1"] = best
            torch.save(ckpt, args.out_dir / "best.pt")
            print(f"  ^ new best dev_l1={best:.4f}", flush=True)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {epoch}, best_dev_l1={best:.4f}")
                break

    with open(args.out_dir / "history.json", "w") as f:
        json.dump({"best_dev_l1": best, "history": history}, f, indent=2)
    print(f"done. best_dev_l1={best:.4f}  ckpt={args.out_dir}/best.pt")


if __name__ == "__main__":
    main()
