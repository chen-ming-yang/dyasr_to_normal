#!/usr/bin/env python3
"""End-to-end: acoustic top-K candidates -> syllable n-gram beam search -> s2mel."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from pypinyin import lazy_pinyin, Style

ROOT = Path("/cmy/cmy/dysa")
sys.path.insert(0, str(ROOT / "index-tts-full"))
sys.path.insert(0, str(ROOT / "dsr"))
sys.path.insert(0, str(ROOT / "dsr" / "stubs"))

from semantic_frontend import SemanticFrontend, resample_to_16k
from train_syllable_cls import SyllableClassifier
from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from indextts.s2mel.modules.audio import mel_spectrogram
from indextts.s2mel.modules.bigvgan import bigvgan

import soundfile as sf
import torchaudio

LIB = ROOT / "runs/dsr/syllable_lib"
CLS = ROOT / "runs/dsr/train_syllable_cls/best.pt"
NGRAM = ROOT / "runs/dsr/syllable_ngram/ngram.json"
OUT = ROOT / "runs/dsr/compare_lm"
MEL_SR = 22050

UNIFORM_LOG = -np.log(670.0)


def read_mono(path):
    y, sr = sf.read(str(path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    return y, sr


def frame_span(start, end, dur, n):
    a = int(round(start / dur * n))
    b = int(round(end / dur * n))
    return max(0, min(a, n)), max(0, min(b, n))


class NgramLM:
    def __init__(self, path: Path):
        self.raw = json.load(open(path))
        self.ng = {}
        for k, v in self.raw.items():
            ctx = () if k == "<BOS>" else tuple(k.split("\x01"))
            self.ng[ctx] = v

    def logprob(self, word, context, max_order=4):
        """Backoff n-gram log prob. context is a tuple of preceding syllables."""
        for n in range(max_order, 0, -1):
            ctx = context[-(n - 1):] if n > 1 else ()
            if ctx in self.ng and word in self.ng[ctx]:
                total = sum(self.ng[ctx].values())
                return np.log(self.ng[ctx][word] / total)
        return UNIFORM_LOG


@torch.no_grad()
def build_reference(fe, s2mel, campplus, cfg, ref16, ref22, device):
    inputs = fe.processor(ref16, sampling_rate=16000, return_tensors="pt")
    spk = fe.model(input_features=inputs.input_features.to(device), attention_mask=inputs.attention_mask.to(device), output_hidden_states=True).hidden_states[17]
    spk = (spk - fe.mean) / fe.std
    _, S_ref = fe.codec.quantize(spk)
    ref_mel = mel_spectrogram(torch.from_numpy(ref22).to(device).float().unsqueeze(0), n_fft=1024, num_mels=80, sampling_rate=MEL_SR, hop_size=256, win_size=1024, fmin=0, fmax=None, center=False)
    ref_len = torch.LongTensor([ref_mel.size(2)]).to(device)
    fb = torchaudio.compliance.kaldi.fbank(torch.from_numpy(ref16).to(device).float().unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
    fb = fb - fb.mean(dim=0, keepdim=True)
    style = campplus(fb.unsqueeze(0))
    prompt_cond = s2mel.models["length_regulator"](S_ref, ylens=ref_len, n_quantizers=3, f0=None)[0]
    return style, prompt_cond, ref_mel


def beam_search(cands, lm, beam_width=8, top_k=10, lm_weight=0.3, max_order=4):
    """cands: list of (list of (syllable, log_prob)) per position.
    Returns best syllable sequence.
    """
    beams = [((), 0.0)]  # (sequence, score)
    for pos, cand in enumerate(cands):
        new_beams = []
        for seq, score in beams:
            ctx = seq[-(max_order - 1):]
            for syll, ac_logp in cand:
                lm_logp = lm.logprob(syll, ctx, max_order)
                new_score = score + ac_logp + lm_weight * lm_logp
                new_beams.append((seq + (syll,), new_score))
        # prune
        new_beams.sort(key=lambda x: -x[1])
        beams = new_beams[:beam_width]
    beams.sort(key=lambda x: -x[1])
    return list(beams[0][0])


def main():
    device = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    fe = SemanticFrontend("cuda")
    cfg = OmegaConf.load(str(ROOT / "indextts2/config.yaml"))

    ckpt = torch.load(str(CLS), map_location="cpu", weights_only=False)
    id_to_pinyin = {int(k): v for k, v in ckpt["id_to_pinyin"].items()}
    cls = SyllableClassifier(d_in=1024, d_h=512, n_class=ckpt["n_class"]).to(device)
    cls.load_state_dict(ckpt["model"])
    cls.eval()

    syll_meta = [json.loads(l) for l in open(LIB / "meta.jsonl")]
    py_to_sample = {sm["pinyin"]: sm["samples"][0] for sm in syll_meta}

    lm = NgramLM(NGRAM)

    s2mel = MyModel(cfg.s2mel, use_gpt_latent=True)
    s2mel, *_ = load_checkpoint2(s2mel, None, str(ROOT / "indextts2/s2mel.pth"), load_only_params=True)
    s2mel = s2mel.to(device).eval()
    for p in s2mel.parameters():
        p.requires_grad = False
    s2mel.models["cfm"].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
    campplus = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus.load_state_dict(torch.load(str(ROOT / "campplus/campplus_cn_common.bin"), map_location="cpu"))
    campplus = campplus.to(device).eval()
    bv = bigvgan.BigVGAN.from_pretrained(str(ROOT / "bigvgan_v2_22khz_80band_256x"), use_cuda_kernel=False)
    bv.remove_weight_norm()
    bv = bv.eval().to(device)

    ref, sr = sf.read(str(ROOT / "listen_samples_all/0001_S006T004E000N00000/normal.wav"), dtype="float32")
    if ref.ndim > 1:
        ref = ref.mean(1)
    ref16 = resample_to_16k(ref, sr)
    ref22 = torchaudio.functional.resample(torch.from_numpy(ref.astype("float32")).unsqueeze(0), sr, MEL_SR).squeeze(0).numpy()
    style, prompt_cond, ref_mel = build_reference(fe, s2mel, campplus, cfg, ref16, ref22, device)

    for split in ["train", "dev"]:
        rows = [json.loads(l) for l in open(ROOT / f"data/char_align/s06_part1/{split}.jsonl")][:3]
        for i, r in enumerate(rows, 1):
            uid = r["id"]
            dys, dsr = read_mono(Path(r["audio"]))
            dys16 = resample_to_16k(dys, dsr)
            feat = fe.features(dys16)
            Td = feat.shape[0]
            dys_dur = len(dys16) / 16000.0
            onsets = r["onsets"]
            ends = onsets[1:] + [dys_dur]
            truth_py = lazy_pinyin(r["chars"], style=Style.TONE3)

            # per-position top-K candidates
            cands = []
            for si in range(len(onsets)):
                a, b = frame_span(onsets[si], ends[si], dys_dur, Td)
                if b - a < 2:
                    continue
                seg = feat[a:b].unsqueeze(0)
                mask = torch.ones(1, seg.shape[1], dtype=torch.bool, device=device)
                with torch.no_grad():
                    logits = cls(seg.to(device), mask)
                probs = torch.softmax(logits, dim=-1)[0]
                top = torch.topk(probs, k=10)
                cands.append([(id_to_pinyin[int(pid)], float(np.log(p))) for pid, p in zip(top.indices.cpu().numpy(), top.values.cpu().numpy())])

            if not cands:
                continue
            # beam search decode
            seq = beam_search(cands, lm, beam_width=8, top_k=10, lm_weight=0.3)

            # synthesize from decoded sequence (natural speed, no stretch)
            code_parts = []
            for p in seq:
                samp = py_to_sample.get(p)
                if samp is None:
                    continue
                code_parts.append(np.asarray(samp["codes"], dtype=np.int64))
            codes = np.concatenate(code_parts) if code_parts else np.zeros(1, dtype=np.int64)

            codes_t = torch.from_numpy(codes).to(device).unsqueeze(0)
            S_infer = fe.codec.quantizer.vq2emb(codes_t.unsqueeze(1)).transpose(1, 2)
            target_lengths = (torch.LongTensor([len(codes)]) * 1.72).long().to(device)
            cond = s2mel.models["length_regulator"](S_infer, ylens=target_lengths, n_quantizers=3, f0=None)[0]
            cat = torch.cat([prompt_cond, cond], dim=1)
            vc = s2mel.models["cfm"].inference(cat, torch.LongTensor([cat.size(1)]).to(device), ref_mel, style, None, 25, inference_cfg_rate=0.7)
            vc = vc[:, :, ref_mel.size(-1):]
            with torch.no_grad():
                wav_b = bv(vc.float().detach().clone()).squeeze().float().cpu().numpy()

            folder = OUT / f"{split}_{i:02d}_{uid}"
            folder.mkdir(parents=True, exist_ok=True)
            sf.write(folder / "dys.wav", dys16, 16000)
            sf.write(folder / "lm_corrected.wav", wav_b, MEL_SR)
            hit = sum(1 for a, b in zip(truth_py, seq) if a == b)
            print(f"[{split} {i}] {uid} {r['text']}", flush=True)
            print(f"   真实={truth_py}", flush=True)
            print(f"   纠错后={list(seq)}  命中={hit}/{len(truth_py)}", flush=True)

    print("done ->", OUT)


if __name__ == "__main__":
    main()
