#!/usr/bin/env python3
"""End-to-end inference: dysarthric wav -> v5 char vectors -> codes -> s2mel -> BigVGAN."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path("/cmy/cmy/dysa")
sys.path.insert(0, str(ROOT / "index-tts-full"))
sys.path.insert(0, str(ROOT / "dsr"))
sys.path.insert(0, str(ROOT / "dsr" / "stubs"))

from semantic_frontend import SemanticFrontend, resample_to_16k
from train_char import CharPoolHead

from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from indextts.s2mel.modules.audio import mel_spectrogram
from indextts.s2mel.modules.bigvgan import bigvgan

import soundfile as sf

CFG_PATH = ROOT / "indextts2/config.yaml"
S2MEL_PATH = ROOT / "indextts2/s2mel.pth"
CAMPLUS_PATH = ROOT / "campplus/campplus_cn_common.bin"
BIGVGAN_DIR = ROOT / "bigvgan_v2_22khz_80band_256x"
HEAD_PATH = ROOT / "runs/dsr/train_char_v5/best.pt"
MANIFEST = ROOT / "normal_stretched_targets/test.jsonl"
OUT = ROOT / "runs/dsr/s2mel_out"

MEL_SR = 22050
SEMANTIC_SR = 50.0


def load_models(device):
    cfg = OmegaConf.load(str(CFG_PATH))
    fe = SemanticFrontend(str(device))

    s2mel = MyModel(cfg.s2mel, use_gpt_latent=True)
    s2mel, *_ = load_checkpoint2(s2mel, None, str(S2MEL_PATH), load_only_params=True)
    s2mel = s2mel.to(device)
    s2mel.eval()
    for p in s2mel.parameters():
        p.requires_grad = False
    s2mel.models["cfm"].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    campplus = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus.load_state_dict(torch.load(str(CAMPLUS_PATH), map_location="cpu"))
    campplus = campplus.to(device).eval()

    bv = bigvgan.BigVGAN.from_pretrained(str(BIGVGAN_DIR), use_cuda_kernel=False)
    bv.remove_weight_norm()
    bv = bv.eval().to(device)

    head = CharPoolHead().to(device)
    ckpt = torch.load(str(HEAD_PATH), map_location="cpu", weights_only=False)
    head.load_state_dict(ckpt["head"])
    head.eval()
    return fe, s2mel, campplus, bv, head, cfg


def mel_fn(wav_22k, cfg):
    return mel_spectrogram(
        wav_22k,
        n_fft=cfg.s2mel.preprocess_params.spect_params.n_fft,
        num_mels=cfg.s2mel.preprocess_params.spect_params.n_mels,
        sampling_rate=MEL_SR,
        hop_size=cfg.s2mel.preprocess_params.spect_params.hop_length,
        win_size=cfg.s2mel.preprocess_params.spect_params.win_length,
        fmin=cfg.s2mel.preprocess_params.spect_params.get("fmin", 0),
        fmax=None,
        center=False,
    )


@torch.no_grad()
def build_reference(fe, s2mel, campplus, cfg, ref_wav_16k, ref_wav_22k, device):
    import torchaudio

    inputs = fe.processor(ref_wav_16k, sampling_rate=16000, return_tensors="pt")
    feat = fe.model(
        input_features=inputs.input_features.to(device),
        attention_mask=inputs.attention_mask.to(device),
        output_hidden_states=True,
    ).hidden_states[17]
    feat = (feat - fe.mean) / fe.std
    spk_cond_emb = feat
    _, S_ref = fe.codec.quantize(spk_cond_emb)
    ref_mel = mel_fn(torch.from_numpy(ref_wav_22k).to(device).float().unsqueeze(0), cfg)
    ref_len = torch.LongTensor([ref_mel.size(2)]).to(device)
    fb = torchaudio.compliance.kaldi.fbank(
        torch.from_numpy(ref_wav_16k).to(device).float().unsqueeze(0),
        num_mel_bins=80, dither=0, sample_frequency=16000,
    )
    fb = fb - fb.mean(dim=0, keepdim=True)
    style = campplus(fb.unsqueeze(0))
    prompt_condition = s2mel.models["length_regulator"](S_ref, ylens=ref_len, n_quantizers=3, f0=None)[0]
    return style, prompt_condition, ref_mel


@torch.no_grad()
def predict_codes(fe, head, dys_16k, spans, device):
    """spans: list of (a,b) frame indices on the layer-17 grid; return per-frame codes."""
    inputs = fe.processor(dys_16k, sampling_rate=16000, return_tensors="pt")
    h = fe.model(
        input_features=inputs.input_features.to(device),
        attention_mask=inputs.attention_mask.to(device),
        output_hidden_states=True,
    ).hidden_states[17][0]
    h = (h - fe.mean) / fe.std
    T = h.shape[0]
    codes = np.full(T, -1, dtype=np.int64)
    codebook = fe.codebook().to(device)  # [8192, 8]
    for a, b in spans:
        if b - a < 2:
            continue
        seg = h[a:b].unsqueeze(0)
        mask = torch.ones(1, seg.shape[1], dtype=torch.bool, device=device)
        vec = head(seg, mask)  # [1, 8]
        d = vec @ codebook.t()
        cid = int(d.argmax(-1).item())
        codes[a:b] = cid
    # fill any leftover (silence) frames with the nearest speech frame's code
    for t in range(T):
        if codes[t] != -1:
            continue
        j = t
        while j >= 0 and codes[j] == -1:
            j -= 1
        k = t
        while k < T and codes[k] == -1:
            k += 1
        cand = None
        if j >= 0:
            cand = codes[j]
        if k < T:
            cand = codes[k] if cand is None else (cand if t - j <= k - t else codes[k])
        codes[t] = cand if cand is not None else 0
    return codes.astype(np.int64)


def main():
    import torchaudio

    device = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    fe, s2mel, campplus, bv, head, cfg = load_models(device)

    # reference: first normal.wav as S06 timbre prompt
    ref_path = ROOT / "listen_samples_all/0001_S006T004E000N00000/normal.wav"
    ref, sr = sf.read(str(ref_path), dtype="float32")
    if ref.ndim > 1:
        ref = ref.mean(1)
    ref_16k = resample_to_16k(ref, sr)
    ref_22k = torchaudio.functional.resample(
        torch.from_numpy(ref.astype(np.float32)).unsqueeze(0), sr, MEL_SR
    ).squeeze(0).numpy()
    style, prompt_condition, ref_mel = build_reference(fe, s2mel, campplus, cfg, ref_16k, ref_22k, device)

    rows = [json.loads(l) for l in open(MANIFEST)][:6]
    index = []
    for i, r in enumerate(rows, 1):
        uid = r["id"]
        # spans from v4 cache (aligned)
        cache = np.load(ROOT / f"runs/dsr/code_cache_v4/test/{uid}.npz")
        spans = cache["spans"]
        dys, dsr = sf.read(str(r["dys_wav"]), dtype="float32")
        if dys.ndim > 1:
            dys = dys.mean(1)
        dys_16k = resample_to_16k(dys, dsr)
        codes = predict_codes(fe, head, dys_16k, spans, device)
        codes_t = torch.from_numpy(codes).to(device).unsqueeze(0)

        S_infer = fe.codec.quantizer.vq2emb(codes_t.unsqueeze(1))  # [B,1,T,1024] -> vq2emb
        S_infer = S_infer.transpose(1, 2)  # [B, T, 1024]
        target_lengths = (torch.LongTensor([len(codes)]) * 1.72).long().to(device)
        cond = s2mel.models["length_regulator"](S_infer, ylens=target_lengths, n_quantizers=3, f0=None)[0]
        cat_condition = torch.cat([prompt_condition, cond], dim=1)
        vc_target = s2mel.models["cfm"].inference(
            cat_condition,
            torch.LongTensor([cat_condition.size(1)]).to(device),
            ref_mel, style, None, 25, inference_cfg_rate=0.7,
        )
        vc_target = vc_target[:, :, ref_mel.size(-1):]
        with torch.no_grad():
            wav = bv(vc_target.float().detach().clone()).squeeze().float().cpu().numpy()

        folder = OUT / f"{i:02d}_{uid}"
        folder.mkdir(parents=True, exist_ok=True)
        sf.write(folder / "dys.wav", dys_16k, 16000)
        sf.write(folder / "pred.wav", wav, MEL_SR)
        index.append({"id": uid, "text": r["text"], "codes": codes.tolist()})
        print(f"[{i}/{len(rows)}] {uid} {r['text']} mel={tuple(vc_target.shape)} wav={len(wav)/MEL_SR:.2f}s", flush=True)

    with open(OUT / "index.json", "w") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
