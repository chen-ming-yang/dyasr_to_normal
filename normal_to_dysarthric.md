# 正常音 → 构音（S06）

输入正常音频，或只输入一句中文，输出 S06 的构音。推理不读真人构音的时间轴，真人录音只在训练字长和频谱时使用。含糊和拖音是要保留的构音，去掉的是金属声、字间硬切口和回声。

这是 `dyasr_to_normal`（构音 → 正常）的反方向，用于生成 S06 风格的构音样本。

## 1. 整条链路

```text
文本
  → Qwen3-TTS-12Hz-0.6B-Base，x_vector_only，参考音是 S06 真人录音
  → 24 kHz 正常音
  → Qwen3-ForcedAligner-0.6B 切汉字
正常音 + 字边界（已有 normal.wav 时从这里进入）
  → 字长模型：每个字的拉长倍数和字后停顿
  → 逐字 WSOLA 拉伸，短停顿交叉叠化，长停顿淡入淡出
  → BigVGAN 的 80 维 log-mel（22050 Hz，n_fft 1024，hop 256，fmax 11025）
  → MelVC（STFT 微调）：输出 = 源 mel + 0.6 × 门 × 残差
  → 冻结 BigVGAN
  → 峰值归一到 0.95
```

文本入口是 `dsr/infer_from_text.py`，已有正常音和字边界时走 `dsr/infer_from_normal.py` 的 `convert_chars`。

```bash
python dsr/infer_from_text.py --text "你要合成的句子"
```

权重和模型（路径写的是实验机）：

| 步骤 | 位置 |
|---|---|
| TTS | `Qwen3-TTS-12Hz-0.6B-Base/` |
| 对齐器 | `Qwen3-ForcedAligner-0.6B` |
| 音色参考 | S06 一条真人录音 |
| 字长 | `runs/dsr/train_duration/best.pt` |
| 频谱 | `runs/dsr/train_vc_wsola_stft/best.pt` |
| 声码器 | `bigvgan_v2_22khz_80band_256x` |

## 2. 字长

脚本 `dsr/duration.py`、`dsr/train_duration.py`。1012 条训练、128 条验证。每个字一条样本，训练 5582 字，验证 745 字。

目标从真人构音的字区间里切出来。区间内按能量裁出发音段，发音时长封顶为正常字长的 2 倍，多出来的算进停顿。句尾文件末尾的静音先裁掉，不记在最后一个字上。

输入是正常字长的对数、句中位置、是否首字或尾字、上一字字长，以及声母、韵母、声调的 one-hot。网络是两层 MLP（128、64），输出拉长倍数（0 到 2，sigmoid）和停顿秒数（softplus）。

验证集：发音时长误差 0.098 秒，停顿误差 0.109 秒，整句时长误差 0.40 秒。固定中位数基线分别是 0.173 秒、0.138 秒、0.68 秒。最好是第 62 轮。

## 3. 拉伸和拼接

实现在 `duration.render`。不用 librosa 相位声码器，那个会发出金属声。每个字先从正常音里切出来，只在这一段上做 WSOLA：窗长 25 ms，半叠，对齐搜索 ±6 ms。窗口不准读到下一个字，否则字尾会带上下一个字的起音，交叉淡化时同一小段再响一次，听感是短回声。

拼接：

- 预测停顿短于 100 ms：不留空，20 ms 等功率交叉淡化。
- 停顿达到 100 ms：保留这段静音，前后各 40 ms 淡出、淡入。

## 4. 频谱

频谱模型是 `MelVC`（`dsr/vc_model.py`，10 个空洞残差块，输出 = 源 mel + 残差），为 WSOLA 时间轴重训过。

配对用 `dsr/build_vc_cache_wsola.py` 重做：源是正常音按真人构音字时长做 WSOLA 拉伸后的 mel，目标是真人构音 mel，所以训练和推理的输入分布一致。先训 mel 版 `train_vc_wsola/best.pt`（dev L1 1.228，恒等基线 2.412），再用冻结 BigVGAN 的多分辨率 STFT 对着真人构音波形微调成 `train_vc_wsola_stft/best.pt`（验证 STFT 2.509 → 2.394）。

推理时输出是 `源 mel + 0.6 × 门 × 残差`。两个旋钮（`infer_from_normal.py` 的 `RESIDUAL_SCALE` 和 `GATE_MODE`）从试听定出来：

- 残差只加 0.6 倍。加满（1.0）时构音变形更重，但电音也回来。
- 残差过一个 `energy_freq` 门，只在「语音帧、偏高频的辅音频段」加变形，静音段和低频保持正常音。去掉这个门、对整段加全量残差，电音会明显加重。

电音的来源是残差在静音段和低频段乱加，BigVGAN 对被抹平的 mel 会用周期性激励去补。门加收小的残差把这块压住了。

mel 必须用 BigVGAN 自己的 `mel_spectrogram`（`fmax=None`，即 11025 Hz）。换成 fmax 8000 的滤波器，声码器会读错频带。

## 5. 大模型尝试（未收敛）

用 Qwen3-TTS 直接生成构音码的方向试过一版，脚本在 `dsr/codec_adapter.py`、`dsr/build_codec_pairs.py`、`dsr/train_codec_adapter.py`、`dsr/infer_codec_adapter.py`。

做法：正常音和构音都用 Qwen3-TTS 的冻结编解码器变成 12.5 Hz 的 16 层码，字级标注把它们按字切成对，训练一个小 Transformer 从正常码预测构音码。开发集交叉熵最好 7.03（随机 7.62），对齐后的正常码直接抄给构音只有 1.1% 第一层码对得上。5500 字不够一个小网络从零对上两套离散码，没有导出预测音频。

结论：要让大模型直接出构音，适配层得加在已经会生成合法语音码的 talker 上，用构音码做交叉熵，而不是让新网络从零猜 2048 类码。

## 6. 还没做

1. STFT 微调只把验证 STFT 降了约 4.5%。若 `pred_stft_s0.6_energy_freq` 仍嗡，瓶颈不在这两个旋钮，而在 BigVGAN 对被抹平的 mel 补周期性激励，要换成波形域生成（扩散或 flow matching）。
2. 字长模型给多数字都预测了一小段停顿，更长的停顿仍是字间空隙。
3. 音色参考不是当初克隆 1270 条正常音用的那一句，换参考音会改变 TTS 音色。
