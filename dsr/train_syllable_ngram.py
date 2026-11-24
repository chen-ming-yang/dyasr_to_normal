#!/usr/bin/env python3
"""Train a syllable-level n-gram language model.

Text is converted to pinyin (TONE3) first, so the model works on sound sequences
only (no Chinese characters at inference time).
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import Counter, defaultdict

from pypinyin import lazy_pinyin, Style

ROOT = Path("/cmy/cmy/dysa")
ALIGN = ROOT / "data/char_align/s06_part1"
OUT = ROOT / "runs/dsr/syllable_ngram"


def to_pinyin_seq(text: str):
    return tuple(lazy_pinyin(text, style=Style.TONE3))


def build_ngram(sequences, order: int = 4):
    """Return dict: context tuple -> Counter of next syllable."""
    model = defaultdict(Counter)
    for seq in sequences:
        for n in range(1, order + 1):
            for i in range(len(seq) - n + 1):
                ctx = seq[i:i + n - 1] if n > 1 else ()
                nxt = seq[i + n - 1]
                model[ctx][nxt] += 1
    return model


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # collect all sentences (train + dev + test) as pinyin sequences
    seqs = []
    for split in ["train", "dev", "test"]:
        for line in open(ALIGN / f"{split}.jsonl"):
            r = json.loads(line)
            seqs.append(to_pinyin_seq(r["text"]))

    print(f"sentences {len(seqs)}")

    # add <s> and </s> boundaries
    padded = []
    for s in seqs:
        padded.append(("<s>",) + s + ("</s>",))
    model = build_ngram(padded, order=4)

    # convert to JSON-serializable; empty context -> "<BOS>"
    out = {}
    for ctx, cnt in model.items():
        key = "\x01".join(ctx) if ctx else "<BOS>"
        out[key] = dict(cnt)
    with open(OUT / "ngram.json", "w") as f:
        json.dump(out, f, ensure_ascii=False)

    # stats
    vocab = set()
    for s in seqs:
        vocab.update(s)
    print(f"vocab size {len(vocab)}")
    print(f"ngram contexts {len(out)}")
    print("saved ->", OUT / "ngram.json")


if __name__ == "__main__":
    main()
