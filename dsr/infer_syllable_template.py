#!/usr/bin/env python3
"""Direction 2 inference: onset -> retrieve syllable template -> DTW -> s2mel -> wav."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path("/cmy/cmy/dysa")
sys.path.insert(0, str(ROOT / "index-tts-full"))
sys.path.insert(0, str(ROOT / "dsr"))
sys.path.insert(0, str(ROOT / "dsr" / "stubs"))

from semantic_frontend import SemanticFrontend, resample_to_16k
from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from indextts.s2mel.modules.audio import mel_spectrogram
from indextts.s2mel.modules.bigvgan import bigvgan

import soundfile as sf
import torchaudio

CFG_PATH = ROOT / "indextts2/config.yaml"
S2MEL_PATH = ROOT / "indextts2/s2mel.pth"
CAMPLUS_PATH = ROOT / "campplus/campplus_cn_common.bin"
BIGVGAN_DIR = ROOT / "bigvgan_v2_22khz_80band_256x"
TEMPLATES = ROOT / "runs/dsr/syllable_templates"
OUT = ROOT / "runs/dsr/s2mel_out_v6"

MEL_SR = 22050


def resample_codes(seq, n):
    seq = np.asarray(seq)
    if len(seq) == 0:
        return np.zeros(n, dtype=np.int64)
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    if len(seq) == n:
        return seq.astype(np.int64)
    idx = np.linspace(0, len(seq) - 1, n).round().astype(np.int64)
    return seq[idx]


def load_models(device):
    cfg = OmegaConf.load(str(CFG_PATH))
    fe = SemanticFrontend(str(device))
    s2mel = MyModel(cfg.s2mel, use_gpt_latent=True)
    s2mel, *_ = load_checkpoint2(s2mel, None, str(S2MEL_PATH), load_only_params=True)
    s2mel = s2mel.to(device).eval()
    for p in s2mel.parameters():
        p.requires_grad = False
    s2mel.models["cfm"].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
    campplus = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus.load_state_dict(torch.load(str(CAMPLUS_PATH), map_location="cpu"))
    campplus = campplus.to(device).eval()
    bv = bigvgan.BigVGAN.from_pretrained(str(BIGVGAN_DIR), use_cuda_kernel=False)
    bv.remove_weight_norm()
    bv = bv.eval().to(device)
    keys = np.load(TEMPLATES / "keys.npy")
    metas = [json.loads(l) for l in open(TEMPLATES / "meta.jsonl")]
    return fe, s2mel, campplus, bv, cfg, keys, metas


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


def main():
    device = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    fe, s2mel, campplus, bv, cfg, keys, metas = load_models(device)

    ref, sr = sf.read(str(ROOT / "listen_samples_all/0001_S006T004E000N00000/normal.wav"), dtype="float32")
    if ref.ndim > 1:
        ref = ref.mean(1)
    ref16 = resample_to_16k(ref, sr)
    ref22 = torchaudio.functional.resample(torch.from_numpy(ref.astype("float32")).unsqueeze(0), sr, MEL_SR).squeeze(0).numpy()
    style, prompt_cond, ref_mel = build_reference(fe, s2mel, campplus, cfg, ref16, ref22, device)

    keys_t = torch.from_numpy(keys).to(device)
    rows = [json.loads(l) for l in open(ROOT / "data/char_align/s06_part1/test.jsonl")][:6]

    for i, r in enumerate(rows, 1):
        uid = r["id"]
        dys, dsr = sf.read(str(r["audio"]), dtype="float32")
        if dys.ndim > 1:
            dys = dys.mean(1)
        dys16 = resample_to_16k(dys, dsr)
        feat = fe.features(dys16)  # [T,1024]
        Td = feat.shape[0]
        dys_dur = len(dys16) / 16000.0
        onsets = r["onsets"]
        ends = onsets[1:] + [dys_dur]

        # retrieve codes per syllable and lay on dys timeline
        codes = np.full(Td, 0, dtype=np.int64)
        for si in range(len(onsets)):
            a = int(round(onsets[si] / dys_dur * Td))
            b = int(round(ends[si] / dys_dur * Td))
            a = max(0, min(a, Td)); b = max(0, min(b, Td))
            if b - a < 2:
                continue
            q = feat[a:b].mean(0); q = q / (torch.norm(q) + 1e-8)
            q = q.float().to(device)
            sims = keys_t @ q
            best = int(torch.argmax(sims).item())
            seq = np.asarray(metas[best]["codes"], dtype=np.int64)
            codes[a:b] = resample_codes(seq, b - a)

        codes_t = torch.from_numpy(codes).to(device).unsqueeze(0)
        S_infer = fe.codec.quantizer.vq2emb(codes_t.unsqueeze(1)).transpose(1, 2)
        target_lengths = (torch.LongTensor([len(codes)]) * 1.72).long().to(device)
        cond = s2mel.models["length_regulator"](S_infer, ylens=target_lengths, n_quantizers=3, f0=None)[0]
        cat = torch.cat([prompt_cond, cond], dim=1)
        vc = s2mel.models["cfm"].inference(cat, torch.LongTensor([cat.size(1)]).to(device), ref_mel, style, None, 25, inference_cfg_rate=0.7)
        vc = vc[:, :, ref_mel.size(-1):]
        with torch.no_grad():
            wav = bv(vc.float().detach().clone()).squeeze().float().cpu().numpy()

        folder = OUT / f"{i:02d}_{uid}"
        folder.mkdir(parents=True, exist_ok=True)
        sf.write(folder / "dys.wav", dys16, 16000)
        sf.write(folder / "pred.wav", wav, MEL_SR)
        print(f"[{i}/{len(rows)}] {uid} {r['text']} wav={len(wav)/MEL_SR:.2f}s", flush=True)

    print("done ->", OUT)


if __name__ == "__main__":
    main()
