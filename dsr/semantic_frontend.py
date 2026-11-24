"""IndexTTS semantic frontend: Wav2Vec2-BERT layer 17 + RepCodec."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/cmy/cmy/dysa")
SRC = ROOT / "index-tts-full"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from indextts.utils.maskgct.models.codec.kmeans.repcodec_model import RepCodec
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel

W2V_DIR = ROOT / "w2v-bert-2.0"
CODEC_PATH = ROOT / "maskgct/semantic_codec/model.safetensors"
STATS_PATH = ROOT / "indextts2/wav2vec2bert_stats.pt"
CODEBOOK = 8192
SILENCE = 8192
N_CLASS = 8193
LAYER = 17


class Cfg:
    codebook_size = 8192
    hidden_size = 1024
    codebook_dim = 8
    vocos_dim = 384
    vocos_intermediate_dim = 2048
    vocos_num_layers = 12
    num_quantizers = 1
    downsample_scale = 1


class SemanticFrontend:
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self.processor = SeamlessM4TFeatureExtractor.from_pretrained(str(W2V_DIR))
        self.model = Wav2Vec2BertModel.from_pretrained(str(W2V_DIR)).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        stats = torch.load(STATS_PATH, map_location="cpu", weights_only=False)
        self.mean = stats["mean"].to(self.device)
        self.std = torch.sqrt(stats["var"]).to(self.device)
        self.codec = RepCodec(cfg=Cfg()).to(self.device).eval()
        from safetensors.torch import load_file

        state = load_file(str(CODEC_PATH))
        self.codec.load_state_dict(state, strict=True)
        for p in self.codec.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def features(self, wav_16k: np.ndarray) -> torch.Tensor:
        """Return normalized layer-17 features [T, 1024] on CPU."""
        wav = np.asarray(wav_16k, dtype=np.float32)
        inputs = self.processor(wav, sampling_rate=16000, return_tensors="pt")
        out = self.model(
            input_features=inputs.input_features.to(self.device),
            attention_mask=inputs.attention_mask.to(self.device),
            output_hidden_states=True,
        )
        feat = out.hidden_states[LAYER]
        feat = (feat - self.mean) / self.std
        return feat[0].float().cpu()

    @torch.no_grad()
    def project(self, feat: torch.Tensor) -> torch.Tensor:
        """Normalized layer-17 features [T, 1024] -> L2-normalized code vectors [T, 8]."""
        if feat.dim() == 2:
            feat = feat.unsqueeze(0)
        feat = feat.to(self.device)
        hidden = self.codec.encoder(feat.transpose(1, 2))
        if hidden.dim() == 3 and hidden.shape[-1] == 1024:
            hidden = hidden.transpose(1, 2)
        quant = self.codec.quantizer.quantizers[0]
        z_e = quant.in_project(hidden)
        z_e = torch.nn.functional.normalize(z_e.transpose(1, 2), dim=-1)
        return z_e[0].float()

    @torch.no_grad()
    def code_vectors(self, wav_16k: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        """L2-normalized 8-d vectors before nearest-code lookup, plus code ids.

        Returns (z [T, 8] CPU, codes [T]).
        """
        feat = self.features(wav_16k).to(self.device).unsqueeze(0)
        encoded = self.codec.encoder(feat.transpose(1, 2))
        if encoded.shape[-1] != 1024:
            encoded = encoded.transpose(1, 2)
        quant = self.codec.quantizer.quantizers[0]
        z_e = quant.in_project(encoded.transpose(1, 2))
        z_e = torch.nn.functional.normalize(z_e.transpose(1, 2), dim=-1)[0]
        indices, _ = self.codec.quantize(feat)
        codes = indices.detach().cpu().numpy() if torch.is_tensor(indices) else np.asarray(indices)
        codes = np.squeeze(codes)
        if codes.ndim > 1:
            codes = codes[0]
        return z_e.float().cpu(), codes.astype(np.int64)

    def codebook(self) -> torch.Tensor:
        weight = self.codec.quantizer.quantizers[0].codebook.weight.detach().float()
        return torch.nn.functional.normalize(weight, dim=-1).cpu()

    @torch.no_grad()
    def codes(self, wav_16k: np.ndarray) -> np.ndarray:
        feat = self.features(wav_16k).to(self.device)
        indices, _ = self.codec.quantize(feat.unsqueeze(0))
        idx = indices
        if isinstance(idx, torch.Tensor):
            idx = idx.detach().cpu().numpy()
        idx = np.asarray(idx)
        idx = np.squeeze(idx)
        if idx.ndim > 1:
            idx = idx[0]
        return idx.astype(np.int64)


def resample_to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return wav.astype(np.float32)
    import torchaudio

    wav_t = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
    return torchaudio.functional.resample(wav_t, sr, 16000).squeeze(0).numpy()
