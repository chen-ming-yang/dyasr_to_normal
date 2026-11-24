#!/usr/bin/env python3
"""Train the v3 linear semantic-code head. Char clips first, then full sentences."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from semantic_frontend import N_CLASS, SILENCE

CACHE = Path("/cmy/cmy/dysa/runs/dsr/code_cache")
OUT = Path("/cmy/cmy/dysa/runs/dsr/train_codes_v3")


class CodeSet(Dataset):
    def __init__(self, split: str, kind: str):
        self.items = []
        for line in open(CACHE / "index.jsonl"):
            rec = json.loads(line)
            if rec["split"] != split:
                continue
            data = np.load(rec["path"])
            feat = data["feat"].astype(np.float32)
            labels = data["labels"].astype(np.int64)
            if kind == "sent":
                self.items.append((feat, labels))
            else:
                for a, b in data["spans"]:
                    if b - a < 1:
                        continue
                    self.items.append((feat[a:b], labels[a:b]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        feat, labels = self.items[idx]
        return torch.from_numpy(feat), torch.from_numpy(labels)


def collate(batch):
    feats = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    feat = pad_sequence(feats, batch_first=True)
    lab = pad_sequence(labels, batch_first=True, padding_value=-100)
    return feat, lab


def run_epoch(loader, head, opt, device, train: bool):
    head.train(train)
    total_loss = 0.0
    total_n = 0
    speech_correct = 0
    speech_n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for feat, lab in loader:
            feat = feat.to(device)
            lab = lab.to(device)
            logits = head(feat)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, N_CLASS), lab.reshape(-1), ignore_index=-100
            )
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            pred = logits.argmax(-1)
            valid = lab != -100
            speech = valid & (lab != SILENCE)
            n = int(valid.sum().item())
            total_loss += loss.item() * n
            total_n += n
            speech_correct += int((pred.eq(lab) & speech).sum().item())
            speech_n += int(speech.sum().item())
    return {
        "ce": total_loss / max(total_n, 1),
        "speech_acc": speech_correct / max(speech_n, 1),
        "speech_n": speech_n,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    train_char = CodeSet("train", "char")
    train_sent = CodeSet("train", "sent")
    dev_sent = CodeSet("dev", "sent")
    print(f"train chars {len(train_char)} train sents {len(train_sent)} dev sents {len(dev_sent)}")

    head = nn.Linear(1024, N_CLASS).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=32, shuffle=shuffle, collate_fn=collate, num_workers=0)

    dev_loader = loader(dev_sent, False)
    history = []
    best = -1.0
    bad = 0

    phases = [
        ("char", train_char, 5),
        ("joint", torch.utils.data.ConcatDataset([train_char, train_sent]), 20),
    ]
    for phase, ds, epochs in phases:
        train_loader = loader(ds, True)
        for ep in range(1, epochs + 1):
            t0 = time.time()
            tr = run_epoch(train_loader, head, opt, device, True)
            va = run_epoch(dev_loader, head, None, device, False)
            row = {
                "phase": phase,
                "epoch": ep,
                "train_ce": tr["ce"],
                "train_speech_acc": tr["speech_acc"],
                "dev_ce": va["ce"],
                "dev_speech_acc": va["speech_acc"],
                "sec": round(time.time() - t0, 1),
            }
            history.append(row)
            print(
                f"{phase} {ep:02d}  train_acc={tr['speech_acc']:.3f}  "
                f"dev_acc={va['speech_acc']:.3f}  dev_ce={va['ce']:.3f}  ({row['sec']}s)",
                flush=True,
            )
            ckpt = {"head": head.state_dict(), "history": history, "best_dev_speech_acc": best}
            torch.save(ckpt, OUT / "last.pt")
            if phase == "joint" and va["speech_acc"] > best + 1e-4:
                best = va["speech_acc"]
                bad = 0
                ckpt["best_dev_speech_acc"] = best
                torch.save(ckpt, OUT / "best.pt")
                print(f"  ^ best dev speech acc {best:.3f}", flush=True)
            elif phase == "joint":
                bad += 1
                if bad >= 5:
                    print("early stop")
                    break
        if phase == "char":
            torch.save({"head": head.state_dict(), "history": history}, OUT / "after_char.pt")

    with open(OUT / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"done best_dev_speech_acc={best:.3f} -> {OUT}/best.pt")


if __name__ == "__main__":
    main()
