#!/usr/bin/env python3
"""Train the per-character duration model on S06 paired alignments."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from duration import MAX_STRETCH, CharDuration, Vocab, collect, featurize

ROOT = Path("/cmy/cmy/dysa")
OUT = ROOT / "runs/dsr/train_duration"


def mae(pred_speech: np.ndarray, pred_pause: np.ndarray, rows: list[dict]) -> dict[str, float]:
    speech = np.array([r["speech"] for r in rows])
    pause = np.array([r["pause"] for r in rows])
    by_id: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_id.setdefault(r["id"], []).append(i)
    utt = []
    for idxs in by_id.values():
        idx = np.array(idxs)
        utt.append(abs((pred_speech[idx] + pred_pause[idx]).sum() - (speech[idx] + pause[idx]).sum()))
    return {
        "speech": float(np.abs(pred_speech - speech).mean()),
        "pause": float(np.abs(pred_pause - pause).mean()),
        "utt": float(np.mean(utt)),
    }


def predict(model, x, dur, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        ratio, pause = model(torch.from_numpy(x).to(device))
    ratio = ratio.cpu().numpy()
    pause = pause.cpu().numpy()
    return ratio * dur, pause


def main():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)

    train_rows = collect("train")
    dev_rows = collect("dev")
    initials = sorted({r["ini"] for r in train_rows})
    finals = sorted({r["fin"] for r in train_rows})
    for token in ("∅",):
        if token not in initials:
            initials = [token] + initials
        if token not in finals:
            finals = [token] + finals
    vocab = Vocab(initials, finals)
    x_tr, dur_tr, y_tr = featurize(train_rows, vocab)
    x_dv, dur_dv, _ = featurize(dev_rows, vocab)

    med_ratio = float(np.median(y_tr[:, 0]))
    med_pause = float(np.median(y_tr[:, 1]))
    base = mae(np.full(len(dev_rows), med_ratio) * dur_dv, np.full(len(dev_rows), med_pause), dev_rows)
    print(
        f"train_chars={len(train_rows)} dev_chars={len(dev_rows)} "
        f"median_ratio={med_ratio:.2f} median_pause={med_pause:.2f}s "
        f"baseline speech {base['speech']:.3f} pause {base['pause']:.3f} utt {base['utt']:.3f}",
        flush=True,
    )

    model = CharDuration(vocab.size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xt = torch.from_numpy(x_tr).to(device)
    dt = torch.from_numpy(dur_tr).to(device)
    yt = torch.from_numpy(y_tr).to(device)

    best = 1e9
    bad = 0
    best_state = None
    log_path = OUT / "train_log.jsonl"
    for epoch in range(1, 201):
        model.train()
        t0 = time.time()
        order = torch.randperm(xt.size(0), device=device)
        running = 0.0
        steps = 0
        for start in range(0, xt.size(0), 256):
            idx = order[start:start + 256]
            ratio, pause = model(xt[idx])
            speech = ratio * dt[idx]
            loss = (speech - yt[idx, 0] * dt[idx]).abs().mean() + (pause - yt[idx, 1]).abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += loss.item()
            steps += 1
        pred_s, pred_p = predict(model, x_dv, dur_dv, device)
        dev = mae(pred_s, pred_p, dev_rows)
        score = dev["speech"] + dev["pause"]
        row = {"epoch": epoch, "train_loss": running / max(steps, 1), **dev, "sec": round(time.time() - t0, 2)}
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"epoch {epoch:03d} train {row['train_loss']:.4f} "
            f"speech {dev['speech']:.3f} pause {dev['pause']:.3f} utt {dev['utt']:.3f}",
            flush=True,
        )
        if score < best - 1e-4:
            best = score
            bad = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_dev = dev
            best_epoch = epoch
        else:
            bad += 1
            if bad >= 20:
                print(f"early stop at epoch {epoch}", flush=True)
                break

    model.load_state_dict(best_state)
    torch.save(
        {
            "model": best_state,
            "vocab": vocab.to_dict(),
            "max_stretch": MAX_STRETCH,
            "epoch": best_epoch,
            "dev": best_dev,
            "baseline": base,
        },
        OUT / "best.pt",
    )
    print(
        f"best epoch {best_epoch} speech {best_dev['speech']:.3f} pause {best_dev['pause']:.3f} "
        f"utt {best_dev['utt']:.3f} -> {OUT / 'best.pt'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
