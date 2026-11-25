"""Predict S06's per-character speech length and following pause from normal audio.

The spectral converter already assumes a dysarthric timeline. This module
builds that timeline from the normal recording alone: each character's normal
duration, syllable and position predict how long the phone is held and how
much silence follows it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from pypinyin import Style, lazy_pinyin

ROOT = Path("/cmy/cmy/dysa")
ALIGN = ROOT / "data/char_align/s06_part1"
LISTEN = ROOT / "listen_samples_all"

MAX_STRETCH = 2.0
SR = 22050


_LISTEN_INDEX: dict[str, Path] | None = None


def listen_dir(uid: str) -> Path:
    global _LISTEN_INDEX
    if _LISTEN_INDEX is None:
        _LISTEN_INDEX = {p.name.split("_", 1)[-1]: p for p in LISTEN.iterdir() if p.is_dir()}
    return _LISTEN_INDEX[uid]


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    return np.ascontiguousarray(y, dtype=np.float32), int(sr)


def trim_phone(wav: np.ndarray, sr: int, frame_ms: int = 20, rel: float = 0.08, pad_ms: int = 30) -> np.ndarray:
    if len(wav) < int(sr * 0.04):
        return wav
    frame = max(1, int(sr * frame_ms / 1000))
    nf = len(wav) // frame
    if nf < 1:
        return wav
    rms = np.array([np.sqrt((wav[j * frame:(j + 1) * frame] ** 2).mean() + 1e-12) for j in range(nf)])
    voiced = np.where(rms > rel * (rms.max() + 1e-8))[0]
    if len(voiced) == 0:
        return wav
    pad = int(sr * pad_ms / 1000)
    a = max(0, int(voiced[0]) * frame - pad)
    b = min(len(wav), (int(voiced[-1]) + 1) * frame + pad)
    return wav[a:b]


def content_end(wav: np.ndarray, sr: int, frame_ms: int = 20, rel: float = 0.08, pad_ms: int = 80) -> float:
    """Drop only the silence after the last phone, keeping internal pauses."""
    frame = max(1, int(sr * frame_ms / 1000))
    nf = len(wav) // frame
    if nf < 1:
        return len(wav) / sr
    rms = np.array([np.sqrt((wav[j * frame:(j + 1) * frame] ** 2).mean() + 1e-12) for j in range(nf)])
    voiced = np.where(rms > rel * (rms.max() + 1e-8))[0]
    if len(voiced) == 0:
        return len(wav) / sr
    end = min(len(wav), (int(voiced[-1]) + 1) * frame + int(sr * pad_ms / 1000))
    return end / sr


_TONE = re.compile(r"(\d)$")


def syllable_of(char: str) -> tuple[str, str, str]:
    ini = lazy_pinyin(char, style=Style.INITIALS)[0] or "∅"
    fin = lazy_pinyin(char, style=Style.FINALS)[0] or "∅"
    py = lazy_pinyin(char, style=Style.TONE3)[0]
    m = _TONE.search(py)
    tone = m.group(1) if m else "0"
    return ini, fin, tone


class Vocab:
    def __init__(self, initials: list[str], finals: list[str]):
        self.initials = initials
        self.finals = finals
        self.init_index = {s: i for i, s in enumerate(initials)}
        self.final_index = {s: i for i, s in enumerate(finals)}

    @property
    def size(self) -> int:
        # log dur, pos, is_first, is_last, log prev dur, has_prev, initial, final, tone
        return 6 + len(self.initials) + len(self.finals) + 5

    def encode(self, ini: str, fin: str, tone: str, normal_dur: float, pos: float, is_first: bool, is_last: bool, prev_dur: float) -> np.ndarray:
        x = np.zeros(self.size, dtype=np.float32)
        x[0] = np.log(max(normal_dur, 0.05))
        x[1] = pos
        x[2] = 1.0 if is_first else 0.0
        x[3] = 1.0 if is_last else 0.0
        x[4] = np.log(max(prev_dur, 0.05)) if prev_dur > 0 else 0.0
        x[5] = 1.0 if prev_dur > 0 else 0.0
        base = 6
        x[base + self.init_index.get(ini, self.init_index["∅"])] = 1.0
        base += len(self.initials)
        x[base + self.final_index.get(fin, self.final_index["∅"])] = 1.0
        base += len(self.finals)
        tone_i = int(tone) if tone.isdigit() and 1 <= int(tone) <= 5 else 0
        if tone_i:
            x[base + tone_i - 1] = 1.0
        return x

    def to_dict(self) -> dict:
        return {"initials": self.initials, "finals": self.finals}

    @classmethod
    def from_dict(cls, d: dict) -> "Vocab":
        return cls(d["initials"], d["finals"])


def load_rows(split: str) -> list[dict]:
    return [json.loads(line) for line in open(ALIGN / f"{split}.jsonl")]


def char_targets(dys: np.ndarray, sr: int, onsets: list[float], normal_durs: list[float], max_stretch: float = MAX_STRETCH) -> list[tuple[float, float]]:
    end = content_end(dys, sr)
    ends = [onsets[i + 1] if i + 1 < len(onsets) else end for i in range(len(onsets))]
    out = []
    for t0, t1, n_dur in zip(onsets, ends, normal_durs):
        i0 = int(np.clip(round(t0 * sr), 0, len(dys)))
        i1 = int(np.clip(round(max(t1, t0) * sr), 0, len(dys)))
        piece = dys[i0:i1]
        interval = len(piece) / sr
        n_dur = max(float(n_dur), 0.05)
        if len(piece) < int(0.02 * sr):
            out.append((min(n_dur, max_stretch * n_dur), 0.0))
            continue
        voiced = len(trim_phone(piece, sr)) / sr
        speech = float(min(voiced, max_stretch * n_dur))
        pause = float(max(0.0, interval - speech))
        out.append((speech, pause))
    return out


def collect(split: str) -> list[dict]:
    rows = []
    for row in load_rows(split):
        folder = listen_dir(row["id"])
        chars = json.loads((folder / "normal_chars.json").read_text())["chars"]
        if len(chars) != len(row["onsets"]):
            continue
        dys, sr = read_wav(folder / "dys.wav")
        n_durs = [max(0.0, float(c["end"]) - float(c["start"])) for c in chars]
        targets = char_targets(dys, sr, row["onsets"], n_durs)
        n = len(chars)
        for i, (ch, (speech, pause)) in enumerate(zip(chars, targets)):
            n_dur = n_durs[i]
            if n_dur < 0.03:
                continue
            ini, fin, tone = syllable_of(ch["char"])
            rows.append({
                "id": row["id"],
                "split": split,
                "char": ch["char"],
                "ini": ini,
                "fin": fin,
                "tone": tone,
                "normal_dur": n_dur,
                "pos": i / max(n - 1, 1),
                "is_first": i == 0,
                "is_last": i == n - 1,
                "prev_dur": n_durs[i - 1] if i else 0.0,
                "speech": speech,
                "pause": pause,
            })
    return rows


def featurize(rows: list[dict], vocab: Vocab) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.stack([
        vocab.encode(r["ini"], r["fin"], r["tone"], r["normal_dur"], r["pos"], r["is_first"], r["is_last"], r["prev_dur"])
        for r in rows
    ])
    dur = np.array([r["normal_dur"] for r in rows], dtype=np.float32)
    target = np.stack([
        [min(r["speech"] / r["normal_dur"], MAX_STRETCH), r["pause"]] for r in rows
    ]).astype(np.float32)
    return x, dur, target


class CharDuration(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(x)
        ratio = MAX_STRETCH * torch.sigmoid(raw[:, 0])
        pause = torch.nn.functional.softplus(raw[:, 1])
        return ratio, pause


def _match_tail(y: np.ndarray, center: int, tail: np.ndarray, search: int, win: int) -> int:
    """Pick the frame near `center` whose start best continues `tail` (one pitch period)."""
    hop = len(tail)
    lo = max(0, center - search)
    hi = min(len(y) - win, center + search)
    if hi <= lo:
        return int(np.clip(center, 0, max(0, len(y) - win)))
    t = tail - float(tail.mean())
    t_norm = float(np.linalg.norm(t)) + 1e-8

    def best_among(cands: np.ndarray) -> int:
        heads = np.stack([y[int(c):int(c) + hop] for c in cands])
        h = heads - heads.mean(axis=1, keepdims=True)
        denom = np.linalg.norm(h, axis=1) * t_norm + 1e-8
        corr = (h @ t) / denom
        return int(cands[int(np.argmax(corr))])

    step = 2
    coarse = best_among(np.arange(lo, hi + 1, step))
    fine_lo = max(lo, coarse - step)
    fine_hi = min(hi, coarse + step)
    return best_among(np.arange(fine_lo, fine_hi + 1))


def wsola_from_index(y: np.ndarray, src_index: np.ndarray, sr: int) -> np.ndarray:
    """Overlap-add `y` along a source-sample map. Negative indices are silence.

    The analysis frame is chosen to line up with the previous frame's tail, so
    voiced joins keep the pitch period instead of a phase-vocoder buzz.
    """
    win = int(round(sr * 0.025))
    if win % 2:
        win += 1
    hop = win // 2
    search = int(round(sr * 0.006))
    window = np.hanning(win).astype(np.float32)
    out_len = int(len(src_index))
    if out_len == 0:
        return np.zeros(0, dtype=np.float32)
    padded = np.pad(y.astype(np.float32, copy=False), (win, win))
    acc = np.zeros(out_len + win, dtype=np.float32)
    norm = np.zeros(out_len + win, dtype=np.float32)
    template: np.ndarray | None = None
    for out_pos in range(0, out_len, hop):
        center = float(src_index[out_pos])
        if center < 0:
            template = None
            continue
        center_i = int(center) + win
        if template is None:
            pos = int(np.clip(center_i, 0, len(padded) - win))
        else:
            pos = _match_tail(padded, center_i, template, search, win)
        frame = padded[pos:pos + win]
        acc[out_pos:out_pos + win] += frame * window
        norm[out_pos:out_pos + win] += window
        template = frame[-hop:].copy()
    out = acc.copy()
    steady = norm > 0.5
    out[steady] = acc[steady] / norm[steady]
    return out[:out_len].astype(np.float32)


def _crossfade(left: np.ndarray, right: np.ndarray, n: int) -> np.ndarray:
    n = min(n, len(left) // 2, len(right) // 2)
    if n < 2:
        return np.concatenate([left, right])
    ramp = np.linspace(0.0, np.pi / 2, n, dtype=np.float32)
    mid = left[-n:] * np.cos(ramp) + right[:n] * np.sin(ramp)
    return np.concatenate([left[:-n], mid, right[n:]])


def _fade_out(y: np.ndarray, n: int) -> np.ndarray:
    n = min(n, len(y))
    if n < 2:
        return y
    out = y.copy()
    out[-n:] *= np.cos(np.linspace(0.0, np.pi / 2, n, dtype=np.float32))
    return out


def _fade_in(y: np.ndarray, n: int) -> np.ndarray:
    n = min(n, len(y))
    if n < 2:
        return y
    out = y.copy()
    out[:n] *= np.sin(np.linspace(0.0, np.pi / 2, n, dtype=np.float32))
    return out


def render(normal: np.ndarray, normal_sr: int, chars: list[dict], ratios: np.ndarray, pauses: np.ndarray, out_sr: int = SR) -> np.ndarray:
    """Stretch each character with WSOLA, then join them without hard gaps.

    A pause under 100 ms is replaced by a 20 ms crossfade. A longer pause stays,
    but the phone fades through the first and last 40 ms of that gap. Each
    character is stretched from its own slice, so the window cannot copy the
    next character into the tail.
    """
    import librosa

    if len(chars) == 0:
        return np.zeros(out_sr // 10, dtype=np.float32)
    if normal_sr != out_sr:
        audio = librosa.resample(normal.astype(np.float32), orig_sr=normal_sr, target_sr=out_sr)
        scale = out_sr / float(normal_sr)
    else:
        audio = normal.astype(np.float32, copy=False)
        scale = 1.0
    xfade = int(round(0.020 * out_sr))
    fade = int(round(0.040 * out_sr))
    phones: list[np.ndarray] = []
    gaps: list[float] = []
    for ch, ratio, pause in zip(chars, ratios, pauses):
        a = int(np.clip(round(float(ch["start"]) * normal_sr * scale), 0, len(audio)))
        b = int(np.clip(round(float(ch["end"]) * normal_sr * scale), a, len(audio)))
        n_dur = max(float(ch["end"]) - float(ch["start"]), 0.05)
        speech_n = max(xfade * 2, int(round(float(np.clip(ratio, 0.3, MAX_STRETCH)) * n_dur * out_sr)))
        if b <= a:
            phones.append(np.zeros(speech_n, dtype=np.float32))
        else:
            seg = np.ascontiguousarray(audio[a:b], dtype=np.float32)
            phones.append(wsola_from_index(seg, np.linspace(0, len(seg) - 1, speech_n), out_sr))
        gaps.append(float(pause))
    out = phones[0]
    for phone, pause in zip(phones[1:], gaps[:-1]):
        if pause < 0.10:
            out = _crossfade(out, phone, xfade)
            continue
        silence = int(round(pause * out_sr))
        hole = max(0, silence - 2 * fade)
        out = np.concatenate([_fade_out(out, fade), np.zeros(hole, dtype=np.float32), _fade_in(phone, fade)])
    return out.astype(np.float32)
