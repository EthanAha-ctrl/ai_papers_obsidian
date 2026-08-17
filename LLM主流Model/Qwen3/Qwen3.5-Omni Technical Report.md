---
source_pdf: Qwen3.5-Omni Technical Report.pdf
paper_sha256: ddd68086e90fa908d65731effdc345a1ac2d9a0c8ef8a15fb73463de95bcf364
processed_at: '2026-08-11T20:36:24-07:00'
target_folder: LLM主流Model/Qwen3
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Qwen3.5-Omni：用人话拆解

## 一句话总结

Qwen 团队把 text、image、audio、video 全塞进一个 hundreds-of-billions 参数的模型里，让它能听、能看、能想、能说、还能动手干活，而且 streaming latency 压到 235ms，比 GPT-4o 公开 demo 还快。

参考：https://qwen.ai/blog?id=qwen3.5

---

## 为什么要做 omni？动机的 intuition

你想想人类怎么跟世界交互——你一边看对面的人的表情，一边听他说话的语气，一边嘴里回答，手还在敲键盘。这是 omni 的、实时的、agentic 的。

之前的 LLM 只能读写 text，相当于一个又聋又哑又瞎的人只会打字。GPT-4o 和 Gemini 1.5 开了 native omni 的头（https://openai.com/index/hello-gpt-4o/ ， https://arxiv.org/abs/2507.06261 ），Qwen3.5-Omni 是 Qwen 团队在这条路上的第三代迭代。

关键 ambition：native omni agent model。not only perceive and reason, but also act。这意味着 model 自己会调 WebSearch、会调 FunctionCall、会从 audio-visual instructions 直接写 code（他们叫 Audio-Visual Vibe Coding）。这是冲着 AGI 路径去的，不是做一个 chatbot。

---

## Thinker-Talker 架构：大脑和嘴巴分开

### 直觉

你跟人聊天的时候，大脑先想清楚要说什么内容，然后嘴巴嗓子负责把内容用声音表达出来——语调、情绪、停顿、重音都由嘴巴嗓子决定。Thinker-Talker 就是把这个过程 explicit 拆开。

- **Thinker** = 大脑。Hybrid MoE Transformer，负责理解所有模态输入、推理、生成 text。
- **Talker** = 嘴巴嗓子。也是 Hybrid MoE Transformer，负责把 Thinker 想好的内容用 streaming speech 输出。

### 关键设计：Talker 直接吃 Thinker 的 hidden state

这是最重要的 architecture choice。Talker not only receives text tokens from Thinker, but directly receives high-level representations。

为什么？因为 text token 丢了太多东西。同样一句 "okay"，text 上就是 5 个字符，但 voice 上可以是敷衍的 "okay"、惊讶的 "okay"、生气的 "okay"。这些 paraphrase 信息——prosody、emotion、loudness、pause pattern——如果只从 text token 重建，就全丢了。

直接从 Thinker 的 hidden representation 到 voice，保留了这些 richness。这接近 GPT-4o 的 "single model handles audio in and out" 的思路，but more interpretable because the split is explicit。

### Talker 的下游 pipeline

```
Thinker hidden state
    ↓
Talker (Hybrid MoE Transformer) → autoregressively predict multi-codebook sequence
    ↓
MTP module (Dense Transformer) → output residual codebooks for current frame
    ↓
Code2Wav (causal ConvNet) → incrementally synthesize waveform
    ↓
streaming audio output
```

MTP 是 multi-token prediction，一次预测一帧的多个 RVQ codebook。Code2Wav 是 lightweight ConvNet，causal 的所以能 streaming。整个下游 chain 计算量很小，batch-friendly，适合 serving。

---

## AuT：音频怎么变成 token

### 问题

Transformer 吃的是 discrete token sequence，但音频是连续波形。要先把音频压成 token。

### AuT 的 pipeline

1. Waveform resample 到 16 kHz
2. 算 mel-spectrogram：25ms window、10ms hop、128 mel channels
3. 4 个 Conv2D blocks 下采样 16 倍
4. 进 self-attention layers
5. 输出 6.25Hz 的 token，即每 160ms 音频 = 1 个 token

### Mel 频谱的公式直觉

$$\text{Mel}_{t,f} = \log\left(\sum_k |X_t[k]|^2 \cdot H_{f,k}\right)$$

变量解释：
- $t$：frame index，每 10ms 一个 frame
- $f$：mel filter index，从 1 到 128
- $X_t[k]$：第 $t$ frame 的 STFT 第 $k$ 个 frequency bin 的复值，取模平方得到 power
- $H_{f,k}$：mel filterbank matrix 的元素，第 $f$ 个 mel filter 在第 $k$ 个 frequency bin 的权重
- 外层 $\log$：人耳对响度的感知是对数的

直觉：mel scale 模拟人耳对 low frequency 更敏感的特性。25ms window 是 speech 的 stationary 假设上限，10ms hop 是 frame rate 的 sweet spot。

### 6.25Hz 的实际意义

4 个 Conv2D stride=2 下采样 $2^4 = 16$ 倍，100Hz mel frame 变成 6.25Hz token。

推算一下：
- 1 秒音频 = 6.25 tokens
- 1 分钟 = 375 tokens
- 1 小时 = 22,500 tokens
- 10 小时 = 225,000 tokens

Qwen3.5-Omni 的 context 是 256k，正好塞下 10 小时音频。这就是为什么他们声称支持 10 hours of audio understanding。

### 40M hours 的 scale

AuT 是从零训练的，吃了 40 million hours 的 audio-text pair 数据，由 Qwen3-ASR 生成。这个 scale 在公开 paper 里是 top tier 的。作为对比，LibriSpeech 才 960 hours，GigaSpeech 大约 10,000 hours。40M hours 是 GigaSpeech 的 4000 倍。

中文 : 英文 : 多语言 = 3.5 : 3.5 : 3 的配比，比 Qwen3-Omni 大幅扩展了多语言覆盖。

### Dynamic attention window size

训练时动态改变 attention window 大小。why？因为 streaming inference 时 prefill cache 是 chunk-wise 的，attention window 不能太大；但 offline understanding 时可以用 full attention 提升性能。训练时随机 window size 让模型同时适应两种 regime，避免 train-test distribution shift。

---

## ARIA：最 elegant 的创新

### 问题背景

Text 和 speech 的 tokenization rate 天然不匹配。

举个例子：英文 "hello" 在 text tokenizer 里可能是 1 个 token，在 speech codec 里可能是 5 个 token。中文 "你好" 在 text 里是 2 个 token，在 speech codec 里可能是 6 个 token。

传统做法（Qwen3-Omni 用的）：dual-track，用 MFA (Montreal Forced Alignment) 算出固定比例，然后两条 track 分开生成。

问题：
1. MFA 算出来的 alignment 是 offline 的，streaming 时不一定准
2. 固定 rate 对不同语言不 work——中文 rate 跟英文 rate 不一样
3. 实际表现：skipped words、incorrect pronunciations、数字渲染混乱

### ARIA 的妙招

ARIA = Adaptive Rate Interleave Alignment。把 dual-track 改成 single-stream interleaved。

核心约束：对生成的任何 prefix $p$，cumulative speech-to-text token ratio 不超过 item-level global ratio。

形式化：
$$\forall \text{ prefix } p, \quad \frac{N_{\text{speech}}(p)}{N_{\text{text}}(p)} \leq R_{\text{global}}$$

变量解释：
- $N_{\text{speech}}(p)$：prefix $p$ 中累计生成的 speech token 数
- $N_{\text{text}}(p)$：prefix $p$ 中累计生成的 text token 数
- $R_{\text{global}}$：整个 training sample 的 speech token 总数除以 text token 总数，是一个常数

### 直觉

ARIA 把 "对齐" 从 external MFA rule 转成 model internal self-consistency。

模型在训练中自己学会：生成 text token 时要考虑已经生成了多少 speech token，不能 speech 跑太快把 text 甩开。rate 是 adaptive 的，不同语言、不同语境模型自己 adapt。

这就像教小孩说话：你 not 告诉他每个字对应几个音节，而是告诉他 "你说的音节总数不能超过字数的 $R$ 倍"，让他自己学会 rate control。

### 工程收益

1. 减少 dual-track 同步开销——一条 stream 就够了
2. 更高效的 token scheduling——decoding 时不需要跨 track 协调
3. 匹配 streaming interaction 的 incremental regime——天生适合边说边生成

这是 single idea 改变整个 serving stack 的例子。

---

## Hybrid MoE + GDN：怎么 handle 256k context

### 长 audio/video 序列的 KV-cache 灾难

10 小时音频 = 225,000 tokens。传统 Transformer 的 KV-cache 会爆炸。即使 MoE 让 active parameters 变小，KV-cache 还是按 total sequence length 线性增长。

### GDN (Gated Delta Net) 的 intuition

GDN 是 Hybrid MoE 架构里的一个 module，专门用于加速长序列建模。

直觉类比：你听 10 小时 podcast，不会记住每一秒的声音，但你会持续更新一个 "compressed memory"——key points、speaker identity、topic shifts。GDN 就是给 neural net 装了这样一个 compressed memory。

具体上，GDN 通过 gated delta mechanism 决定哪些历史信息以 "delta" 形式更新到 memory state，not 保留所有历史 KV pair。这大幅减小 KV-cache I/O overhead，对长 audio-video 的 streaming inference 是 critical 的。

公式上（推测，paper 没给 explicit 形式）：
$$M_t = M_{t-1} + g_t \odot \Delta_t$$
$$g_t = \sigma(W_g \cdot h_t + U_g \cdot x_t)$$
$$\Delta_t = W_\Delta \cdot h_t + U_\Delta \cdot x_t$$

变量解释：
- $M_t$：第 $t$ 步的 memory state
- $g_t$：gate vector，元素在 $(0,1)$ 之间，决定更新多少
- $\Delta_t$：delta 信号，要加到 memory 的更新量
- $\sigma$：sigmoid
- $h_t$：第 $t$ 步的 hidden state
- $x_t$：第 $t$ 步的 input
- $W_g, U_g, W_\Delta, U_\Delta$：可学习参数
- $\odot$：element-wise product

这是 DeltaNet 一脉的思路（参考 https://arxiv.org/abs/2502.10343 ），memory 是 state-based 的而不是 KV-pair-based 的，所以 memory footprint 是 $O(1)$ per layer per token，not $O(L)$。

### Streaming latency 拆解

Table 2 给了详细数字。以 Qwen3.5-Omni-Flash, 1 concurrency, audio input 为例：

| Component | Latency |
|-----------|---------|
| Thinker TTFT (time-to-first-token) | 80ms |
| Talker TTFC (time-to-first-chunk) | 56ms |
| Thinker TPOP (time-per-output-token) | 5.6ms |
| Talker TPOP + Codec Decode | 14.2ms + 3~5ms |
| **Overall Latency** | **235ms** |

注意 Overall Latency ≠ 各行简单加和，因为 ARIA 是 unified interleaved stream，反映的是 end-to-end critical path 到第一个可播放 audio packet 的时间。

### Generation RTF 的直觉

RTF = Real-Time Factor = 生成时间 / 音频时长。

- 1 conc: Flash 0.178, Plus 0.187
- 8 conc: Flash 0.257, Plus 0.334

RTF < 1 意味着生成比播放快，有余量做 smooth streaming。Plus 在 8 concurrency 下 RTF 0.334 非常健康，1 秒音频只需 0.334 秒生成。

对比：GPT-4o 公开 demo latency 大约 320ms。Qwen3.5-Omni-Flash 的 235ms 实际比 GPT-4o demo 还快。

---

## Pretraining 三阶段：怎么 scale 出 omni 能力

### Stage 1: Encoder Alignment (S1)

LLM frozen，初始化自 Qwen3.5 base。Vision encoder 来自 Qwen3.5-VL。Audio encoder 是 AuT。两个 encoder 分别训练，先训 adapter 再训 encoder。

直觉：先让 encoder 学会把 audio/image 转成 LLM 能理解的 representation space。如果一开始就 joint train，encoder 输出跟 LLM 表征空间错位，梯度信号会乱。

### Stage 2: General Stage (S2)

All parameters unfrozen。约 4T tokens。

Token 分布：
- Text: 0.92T
- Audio: 1.99T
- Image: 0.95T
- Video: 0.14T
- Video-Audio: 0.29T

Sequence length: 32,768

直觉：Audio token 数量是 text 的两倍多。这是因为 audio token rate 高（6.25Hz），1 小时音频 22,500 tokens，要训练 audio understanding 必须 scale audio data。这个配比 reflect 了不同模态的 information density 差异。

### Stage 3: Long Context (S3)

Sequence length 拉到 262,144 (256k)。提升 long audio 和 long video 在训练数据中的比例。

直觉：长序列训练计算贵，且如果一开始就训长序列，模型学不到 short context 的精细能力。stage-wise 训练是 long-context training 的 standard recipe——Gemini 1.5（https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf ）也是类似 path。

---

## Post-training：怎么让 omni model 不变笨

这是 paper 里最 insightful 的部分之一。核心问题：native omni training 会让 text 能力退化吗？Qwen3.5-Omni 用一套 three-stage post-training pipeline 证明了不会。

### Thinker 三阶段

#### Stage 1: Specialist Distillation

先训一组 domain-specialized teacher models，全部从 Qwen3.5 base fine-tune：
- Text teacher：agentic、coding、reasoning
- Vision teacher
- Audio teacher

每个 teacher 在自己领域做到 specialist SOTA。然后用这些 teachers 生成 domain-specific data，distill 进 unified model。

直觉：single model 早期同时学 text+audio+video 会互相干扰，gradient conflict 严重。Mixture of teachers 让每个 teacher 先专业化，再通过 distillation 让 unified model 学到所有 teacher 的能力。这避免了 early-stage joint training 的 gradient conflict 问题。

#### Stage 2: On-Policy Distillation (OPD)——modality gap bridging

核心 insight：模型 text-conditioned 能力强，audio-conditioned 能力弱。

给定 audio-text pair $(a, x)$：
1. 用 text input $x$ 生成 response $y_{\text{text}}$（高质量）
2. 用 audio input $a$ 配 $y_{\text{text}}$ 作为 distillation target 训练

训练目标：
$$\mathcal{L}_{\text{OPD}} = -\sum_t \log p_\theta(y_{\text{text}}^{(t)} \mid a, y_{\text{text}}^{(<t)}, \text{context})$$

变量解释：
- $y_{\text{text}}^{(t)}$：text-conditioned 生成的 response 的第 $t$ 个 token
- $a$：audio input
- $y_{\text{text}}^{(<t)}$：已生成的前 $t-1$ 个 token
- $\text{context}$：对话历史
- $\theta$：模型参数

直觉：这是 modality gap bridging 的核心 trick。模型 text pathway 是 "高水位"，audio pathway 是 "低水位"。OPD 把 text pathway 的 knowledge 蒸馏到 audio pathway。模型 not 从头学 audio reasoning，而是抄 text pathway 的作业。

这也解释了 Table 4 里 Qwen3.5-Omni-Plus 在 IFEval 上 89.7、IFBench 上 52.6，甚至略超 Qwen3.5-Plus-Instruct 的 89.7 和 51.1。OPD + Interaction-Aligned RL 反过来提升了 instruction following。

#### Stage 3: Interaction-Aligned RL

针对 multi-turn 对话中出现的具体问题：
- 多轮后 unintended language code-switching
- Persona inconsistency
- Instruction-following 在 long context 中退化

构造 multi-turn interaction trajectories，设计围绕 user experience 的 reward signals，做 RL。

直觉：这阶段 not 提升单轮任务能力，而是优化长时间交互的稳定性。这是 production-ready 必须的一步——标准 SFT 模型往往在 multi-turn 后退化，user 体验差。

### Talker 四阶段

1. **General Stage**：20M+ hours multilingual speech data，加入 instruction-following speech generation 任务。超越 monotonic mapping from multimodal representations to speech。

2. **Long-Context Stage**：data quality stratification + continual pre-training (CPT)。用 Qwen3-Omni-Captioner（https://arxiv.org/abs/2510.12720 ）减少 noisy data 引入的 hallucination。Context 拉到 64k。

3. **RL Stage**：DPO（https://arxiv.org/abs/2305.18290 ）+ GSPO（https://arxiv.org/abs/2507.18071 ）。

DPO 目标函数：
$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

变量解释：
- $x$：prompt
- $y_w$：preferred response (winner)
- $y_l$：dispreferred response (loser)
- $\pi_\theta$：current policy model
- $\pi_{\text{ref}}$：frozen reference model
- $\beta$：KL penalty 系数，控制偏离 reference 的程度
- $\sigma$：sigmoid function

直觉：DPO 让 model 学会 preferred response 的 log-prob 比 dispreferred 高。关键是不需要单独训练 reward model，直接用 preference pair 训 policy。

4. **Speaker Fine-tuning**：lightweight fine-tuning for target speaker characteristics。

---

## Evaluation 关键 insight

### Text→Text：omni training 不损害 text

Table 4 显示 Qwen3.5-Omni-Plus 在 MMLU-Pro、C-Eval、IFEval 等基准上几乎打平 Qwen3.5-Plus-Instruct。IFBench 甚至略超。

直觉：这反驳了 "omni training 必然损害 text" 的 common belief。关键在 OPD + Specialist Distillation 让 text 能力被 protected 甚至 enhanced。

### Audio→Text：全面超越 Gemini-3.1 Pro

几个关键数字：

| Benchmark | Gemini-3.1 Pro | Qwen3.5-Omni-Plus | 差距 |
|-----------|---------------|-------------------|------|
| MMAU | 81.1 | 82.2 | +1.1 |
| MMSU | 81.3 | 82.8 | +1.5 |
| RUL-MuchoMusic | 59.6 | 72.4 | +12.8 |
| SongFormBench-CN (acc) | 78.1 | 87.1 | +9.0 |
| VoiceBench | 88.9 | 93.1 | +4.2 |
| Fleurs (top60) WER | 7.32 | 6.55 | -0.77 |
| Wenetspeech WER | 23.67 | 4.30 | -19.37 |
| Opencpop WER | 6.83 | 1.49 | -5.34 |

直觉：音乐理解（RUL-MuchoMusic +12.8）和中文 ASR（Wenetspeech -19.37）的巨大优势反映 40M hours audio training data 中包含大量中文和音乐数据。Opencpop 是 singing voice transcription，从 6.83 降到 1.49 说明 AuT 学到了 singing voice 的 robust representation。

### Vision→Text：video 全面超越

| Benchmark | Qwen3.5-Plus-Instruct | Qwen3.5-Omni-Plus | 差距 |
|-----------|----------------------|-------------------|------|
| VideoMME (w/o sub.) | 81.0 | 81.9 | +0.9 |
| MLVU (M-Avg) | 85.1 | 86.8 | +1.7 |
| LVBench | 68.6 | 71.2 | +2.6 |

直觉：静态 vision 能力打平，video 全面超越。Video 任务的提升来自 joint audio-visual training——audio-visual stream 是 real-world phenomena 的 natural representation，model 见过音视频配对后能更好理解 video。

### Audio-Visual→Text：跟 Gemini 互有胜负

| Benchmark | Gemini-3.1 Pro | Qwen3.5-Omni-Plus |
|-----------|---------------|-------------------|
| DailyOmni | 82.7 | 84.6 |
| VideoMME w/ audio | 89.0 | 83.7 |
| Qualcomm IVD | 66.2 | 68.5 |
| OmniGAIA | 68.9 | 57.2 |

直觉：Qwen 在 real-world interactive scenarios（Qualcomm IVD）和 captioning（Omni-Cloze）领先，但 OmniGAIA 大幅落后。OmniGAIA 是 omni-modal agentic tool use benchmark，说明 omni-modal setting 下 agentic 能力还远未成熟。

### Cross-Lingual Voice Cloning：SOTA

Table 11 里 12 个 language pair 中 10 个 SOTA。最惊艳的：zh-to-korean 从 CosyVoice 3 的 14.4 降到 4.03，72% 相对降幅。

直觉：native omni training 让 cross-lingual phoneme mapping 自然涌现。模型同时见过大量中文和韩文 speech data，能学到 cross-lingual phoneme mapping。这是 dedicated TTS model 难以做到的——CosyVoice 3 主要训中文 TTS，cross-lingual 能力受限于训练数据。

---

## Emergent capability: Audio-Visual Vibe Coding

这是 paper 里最 forward-looking 的 observation。模型能直接从 audio-visual instructions 生成 executable code。

直觉：这 not 单独训出来的能力，是 native omni training + agentic training 的 emergent 结果。模型在 representation space 中同时见过 code data 和 audio-visual data，产生了 cross-modal composition——"听一段描述，看一段演示，直接写出对应 code"。

这接近 OpenAI Operator 或 Anthropic Computer Use，但在 omni-modal 层面。是 omni agent 的 future direction。

---

## 跟 GPT-4o / Gemini 的对比直觉

### Architecture 路径

GPT-4o 和 Gemini 1.5 的 architecture 都未公开。Qwen3.5-Omni 显式采用 Thinker-Talker + Hybrid MoE，更可解释。

Thinker-Talker 的 advantage 是 explicit separation of understanding and generation，Talker 直接吃 Thinker hidden state 保留了 paraphrase richness。Disadvantage 是双 model 的 serving complexity。

### Latency

GPT-4o 公开 demo latency ~320ms。Qwen3.5-Omni-Flash audio input 235ms，actually faster。这反映 Hybrid MoE + chunked prefilling + GDN 的工程优化到位。

### Capability breadth

Qwen3.5-Omni 在 audio understanding 和 ASR 上超越 Gemini-3.1 Pro。在 video-audio 融合理解上落后 Gemini-3.1 Pro（VideoMME w/ audio 83.7 vs 89.0）。在 agentic tool use 上大幅落后（OmniGAIA 57.2 vs 68.9）。

直觉：Gemini 1.5 是 native multimodal trained from scratch（https://arxiv.org/abs/2507.06261 ），audio-visual fusion 的 foundation 更扎实。Qwen3.5-Omni 是 modular design（Thinker-Talker），fusion 需要更多 explicit design。

---

## 局限与未来推测

### Paper 里露出的 gap

1. **VideoMME w/ audio 落后 Gemini-3.1 Pro**：audio-visual fusion 理解仍需提升
2. **OmniGAIA 大幅落后**：omni-modal agentic tool use 还不成熟
3. **WildSpeech-Bench 略低**：wild speech robustness 不如 Gemini
4. **SEED test-zh 不如 CosyVoice 3**：dedicated TTS model 在单语言上仍有优势

### 推测的下一步

1. **AuT scaling**：40M hours 已经 top tier，但 Gemini 可能更多。下一步 100M+ hours。

2. **TM-RoPE 的彻底替换**：当前用 explicit timestamp text string 替代 TM-RoPE 的部分功能，但 position encoding 仍是 hybrid 的。未来可能完全用 timecode text + learned position embedding。

3. **MTP 扩展到 video generation**：当前 MTP 只用于 speech codec。未来可能扩展到 video frame prediction，路径类似 Sora 的 diffusion + transformer。

4. **更激进的 RL from environment**：当前 Interaction-Aligned RL 只针对 multi-turn text。未来可能扩展到 RL from tool use feedback 闭环，让 model 真正 learn from action consequence。

5. **Mega-context**：256k 已是 SOTA，但 10 小时音频就塞满了。未来 mega-context (1M+) 或 alternative compression（如 retrieval-augmented streaming memory）是 necessary path。

---

## 关键 Takeaways

1. **Thinker-Talker + hidden state 直传** 是 omni model 保留 paraphrase richness 的关键 architecture choice
2. **ARIA** 把 text-speech alignment 从 external rule 变成 internal self-consistency，是 elegant single idea 改变 serving stack 的例子
3. **Hybrid MoE + GDN** 让 256k context 在 streaming 下 tractable，GDN 的 state-based memory 是 long audio 的 enabler
4. **OPD** 把 text pathway 的 "高水位" 蒸馏到 audio pathway 的 "低水位"，是 modality gap bridging 的核心 trick
5. **Specialist Distillation + OPD + Interaction-Aligned RL** 三阶段 post-training 让 omni model 不损失 text capability
6. **Audio-Visual Vibe Coding** 是 omni agent 的 emergent future direction

参考 paper：
- Qwen2.5-Omni: https://arxiv.org/abs/2503.20215
- Qwen3-Omni: https://arxiv.org/abs/2509.17765
- GPT-4o: https://openai.com/index/hello-gpt-4o/
- Gemini 2.5: https://arxiv.org/abs/2507.06261
- DPO: https://arxiv.org/abs/2305.18290
- GSPO: https://arxiv.org/abs/2507.18071
- CosyVoice 3: https://arxiv.org/abs/2505.17589
- MiniMax-Speech: https://arxiv.org/abs/2505.07916
- Seed-TTS: https://arxiv.org/abs/2406.02430
- MMAU: https://arxiv.org/abs/2410.19168
- MMSU: https://arxiv.org/abs/2506.04779
- MMAR: https://arxiv.org/abs/2505.13032
- Video-MME: https://arxiv.org/abs/2405.21075
- MMMU: https://arxiv.org/abs/2311.16502
- DeltaNet: https://arxiv.org/abs/2502.10343
- Omni-Captioner: https://arxiv.org/abs/2510.12720

Qwen3.5-Omni 的核心贡献不在 single algorithm breakthrough，而在 systematic engineering at scale。从 AuT、Hybrid MoE、ARIA、three-stage pretraining 到 OPD post-training，每个环节都有 deliberate design。这种 holistic approach 是 omni-modal model 从 research prototype 走向 production 的必要 path，也是 native omni agent 的 working blueprint。

---

# Qwen3.5-Omni Technical Report 深度解读

## 1. 模型定位与核心突破

Qwen3.5-Omni 是 Qwen-Omni 系列的第三代迭代，参数规模扩展到 hundreds of billions 级别，支持 256k context length。其核心定位是 **native omni agent model**，即同时具备 perception、reasoning、action 三层能力，跨越 text、image、audio、audio-visual 四种输入模态。

参考链接：
- 官方 Online Demo: https://huggingface.co/spaces/Qwen/Qwen3.5-Omni-Online-Demo
- ModelScope Studio: https://modelscope.cn/studios/Qwen/Qwen3.5-Omni-Online-Demo

它对标的 baseline 是 Gemini-3.1 Pro，在 215 个 audio 与 audio-visual subtasks 上达到 SOTA。值得注意的是，它维持了 text-only 和 vision-only Qwen3.5 同等性能，说明 omni-modal training 不会显著削弱单模态能力。

---

## 2. Thinker-Talker 双轨架构

Qwen3.5-Omni 沿用 Qwen2.5-Omni (https://arxiv.org/abs/2503.20215) 提出的 Thinker-Talker 双轨架构，但对其做了五项关键升级：

### 2.1 架构总览（对应 Figure 2）

```
[Audio Input]  →  AuT Encoder (6.25Hz tokens)
                                   ↓
[Image/Video]  →  SigLIP2 Encoder   →  [Chunked Prefilling]
                                   ↓           ↓
[Text]        →  Qwen3.5 Tokenizer  →  [Thinker: Hybrid MoE Transformer]
                                                   ↓
                                              text tokens
                                                   ↓
                                       [Talker: Hybrid MoE Transformer]
                                                   ↓
                                       [MTP: Dense Transformer]
                                                   ↓
                                       [Code2Wav: causal ConvNet]
                                                   ↓
                                            streaming waveform
```

关键设计要点：
- **Thinker 与 Talker 都采用 Hybrid MoE Transformer**，对应 Qwen3.5 backbone
- **Thinker 负责文本生成与多模态推理**，输出 high-level representations 给 Talker
- **Talker 直接接收 Thinker 的高层表征**，不通过 text token 中转，所以 voice 可承载 text 之外的 paraphrase 信息（如 prosody、emotion）
- Talker autoregressively 预测 multi-codebook sequence，MTP module 输出当前 frame 的 residual codebooks，Code2Wav renderer 增量合成波形

### 2.2 AuT (Audio Transformer) 音频编码器

AuT 是从零训练的 audio encoder，消耗了 40 million hours 的 audio-text pair 数据（由 Qwen3-ASR 生成）。架构如图 3 所示。

**输入处理流程**：
1. Waveform resample 到 16 kHz
2. 转换为 128-channel mel-spectrogram（25ms window, 10ms hop）
3. 通过 4 个 Conv2D blocks 下采样 16 倍
4. 进入 self-attention layers
5. 输出 token rate 为 **6.25Hz**，即每个 output frame 对应约 160ms 原始信号

**核心公式（Mel 频谱生成）**：
$$\text{Mel}_{t,f} = \log\left(\sum_{k} |X_t[k]|^2 \cdot H_{f,k}\right)$$

其中：
- $t$ 是 frame index（10ms 间隔）
- $f$ 是 mel filter index（1 到 128）
- $X_t[k]$ 是第 $t$ frame 的 STFT
- $H_{f,k}$ 是 mel filterbank matrix 的元素

**下采样 Conv2D blocks**：4 个 Conv2D，stride=2，理论下采样率为 $2^4 = 16$ 倍，将 100Hz 的 mel frame 转换为 6.25Hz 的 token。

**Multilingual 数据配比**：中文 : 英文 : 多语言 = 3.5 : 3.5 : 3，对比 Qwen3-Omni 显著提升了多语言覆盖。

**Dynamic attention window size training mechanism**：训练时动态改变 attention window 大小，平衡 streaming prefill caching 与 offline audio understanding 的性能。这是 streaming-first 设计的关键，避免 offline 训练与 online inference 的 train-test distribution shift。

### 2.3 Multi-Modal Perception 与 Timestamp 设计

#### 2.3.1 Tokenizer 设计

Qwen3.5 tokenizer 采用 byte-level BPE，词表大小为 **250k**（从 150k 扩展），在大多数语言上编码解码效率提升 10-60%。

#### 2.3.2 TM-RoPE 的关键改进

Qwen3-Omni 使用 TM-RoPE (Temporal-Modality RoPE) 处理时序感知，但存在两个问题：

**问题 1**：直接将 temporal position ID 绑定到绝对时间，长视频输入会产生过于稀疏的 temporal position IDs，削弱 long-range temporal modeling 能力。

**问题 2**：这种设计需要大规模、不同 fps 均匀分布的训练样本，增加数据构造成本。

**改进方案**：在每个 video 或 audio-video temporal patch 前插入 explicit timestamp，以 formatted text string 表示（如 `"t=12.34s"`），让模型自然地学习 timecode representation。

形式化上，对于视频序列 $\{v_1, v_2, \ldots, v_T\}$，将其变为：
$$\{[\text{ts}_1], v_1, [\text{ts}_2], v_2, \ldots, [\text{ts}_T], v_T\}$$

其中 $[\text{ts}_i]$ 是文本化的 timestamp token。

**音频序列的随机 timestamp 插入**：在音频序列中随机间隔插入 timestamps，进一步提升跨模态对齐。这是 zero-shot 长时序泛化的关键。

#### 2.3.3 Position ID 设计

- 音频：每 160ms 一个 temporal ID
- 视频：monotonically increasing temporal IDs，动态调整保持 160ms/ID 的统一时间分辨率
- 多模态之间 position numbering 是 contiguous，下一个模态从上一个模态最大 position ID + 1 开始

这种设计避免了 position conflict，且与 absolute time 锚定，支持 streaming input of arbitrary duration。

---

## 3. ARIA：核心对齐创新

### 3.1 问题背景

传统 dual-track Talker 设计（如 Qwen3-Omni）的问题：
1. Text token 与 speech token 的 tokenization rate 不匹配
2. 导致 streaming synthesis 不稳定、不自然
3. 表现为：skipped words、incorrect pronunciations、ambiguous rendering of numbers

### 3.2 ARIA 的核心思想

ARIA (Adaptive Rate Interleave Alignment) 将 dual-channel 生成范式统一为 single-channel interleaved formulation。

**关键约束（adaptive rate constraint）**：
对于生成的任何 prefix，cumulative speech-to-text token ratio 不超过 item-level global ratio：

$$\forall \text{ prefix } p, \quad \frac{N_{\text{speech}}(p)}{N_{\text{text}}(p)} \leq R_{\text{global}}$$

其中：
- $N_{\text{speech}}(p)$ 是 prefix $p$ 中累计的 speech token 数
- $N_{\text{text}}(p)$ 是 prefix $p$ 中累计的 text token 数
- $R_{\text{global}}$ 是该训练样本中 speech token 与 text token 的全局比例

### 3.3 与传统方法的对比

| 方法 | 对齐方式 | 问题 |
|------|----------|------|
| Qwen3-Omni dual-track | MFA-derived alignment | 固定比例，无法适应不同语言 |
| Fixed-rate interleaving | 预设固定 rate（如 1 text : 4 speech） | 低效率语言（如中文）tokenization 慢，导致 skipped words |
| ARIA | adaptive rate constraint | 动态决定何时生成 text/speech token，自然支持任意 text-token prefix 后接 speech-token continuation |

### 3.4 ARIA 的优势

ARIA 的 single-stream formulation 带来三个工程优势：
1. **减少 dual-track 同步开销**
2. **更高效的 token scheduling**
3. **匹配 streaming interaction 的 incremental regime**

这是一个非常优雅的设计——把对齐问题从外部 MFA 强制约束，转化为模型内部的 self-consistency 约束，让模型在训练中自己学会 rate-adaptive generation。

---

## 4. Streaming 与 Concurrency 设计

### 4.1 延迟拆解（Table 1）

| Module | Architecture | Streaming |
|--------|--------------|-----------|
| Audio Encoder | AuT | √ |
| Vision Encoder | SigLIP2 | × |
| Thinker | Hybrid MoE Transformer | √ |
| Talker | Hybrid MoE Transformer | √ |
| MTP | Dense Transformer | √ |
| Code2wav | ConvNet | √ |

**First-Packet Latency**:
- Audio Input: Plus 435ms, Flash 235ms
- Video Input: Plus 651ms, Flash 426ms

### 4.2 Chunked Prefilling 机制

Vision 与 audio encoders 沿时间维度输出 chunks，显著降低 Time-To-First-Token (TTFT)。这意味着：
- Audio 编码可以 chunk-wise 进行（边听边编码）
- 视觉编码虽非 streaming，但 prefilling 仍可分块

### 4.3 Gated Delta Net (GDN) 模块

Hybrid MoE 架构包含 GDN module，对长音频-视频序列建模特别有效。

**为什么 GDN 关键**：
- 10 小时音频 = 225,000 tokens
- 传统 Transformer 的 KV-cache 会极其庞大
- GDN 通过 delta-based memory 减小 KV-cache I/O overhead

**直觉理解**：GDN 类似于一种 "compressed memory"，通过门控机制决定哪些历史信息需要以 "delta" 形式更新到 memory 中，而不是完整保留所有历史 KV。这对长音频的 streaming inference 是 critical 的，否则 KV-cache I/O 会成为瓶颈。

### 4.4 延迟拆解表（Table 2）详解

以 Qwen3.5-Omni-Flash, 1 concurrency, audio input 为例：

| Component | Latency |
|-----------|---------|
| Thinker TTFT | 80ms |
| Talker TTFC | 56ms |
| Thinker TPOP | 5.6ms |
| Talker TPOP + Codec Decode | 14.2ms + 3~5ms |
| **Overall Latency** | **235ms** |

注意 Overall Latency ≠ 各行简单相加，因为 ARIA 是 unified interleaved stream，反映 end-to-end critical path。

**Generation RTF (Real-Time Factor)**：
$$\text{RTF} = \frac{\text{generation time}}{\text{audio duration}}$$

- 1 conc: Flash 0.178, Plus 0.187
- 8 conc: Flash 0.257, Plus 0.334

RTF < 1 意味着生成速度快于播放，有 margin 做 smooth streaming。Plus 在 8 concurrency 下 RTF 0.334 仍然非常健康，意味着 1 秒音频仅需 0.334 秒生成。

---

## 5. Pretraining 三阶段策略

### 5.1 Stage 1: Encoder Alignment (S1)

- **LLM frozen**，初始化自 Qwen3.5
- Vision encoder 来自 Qwen3.5
- Audio encoder 是 AuT
- 两个 encoder 分开训练，先训 adapter 再训 encoder

**目的**：建立 encoder 与 LLM 之间的语义对齐 foundation，避免大规模 multimodal training 时 encoder 输出与 LLM 表征空间错位。

### 5.2 Stage 2: General Stage (S2)

- **All parameters unfrozen**
- ~4T tokens
- Token 分布（Table 显示）：
  - Text: 0.92T
  - Audio: 1.99T
  - Image: 0.95T
  - Video: 0.14T
  - Video-Audio: 0.29T
- Sequence length: 32,768

**直觉**：Audio token 数量远超其他模态，反映 audio token rate 高（6.25Hz），1 小时音频 = 22,500 tokens，要训练 audio understanding 必须 scale audio data。

### 5.3 Stage 3: Long Context (S3)

- Sequence length: **262,144** (256k)
- 提升 long audio 与 long video 比例
- 显著改善 long sequence 理解能力

**为什么需要 S3 单独训练**：长序列训练成本高，且如果一开始就训练长序列，模型难以学到 short-context 的精细能力。stage-wise 训练是 long-context 训练的标准做法（参考 Gemini 1.5, https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf）。

---

## 6. Post-training 双轨策略

### 6.1 Thinker 三阶段 Post-training

#### Stage 1: Specialist Distillation

先训练一组 domain-specialized teacher models（基于 Qwen3.5 base），分别 SFT + RL：
- Text teacher: agentic、coding、foundational reasoning
- Vision teacher
- Audio teacher

然后用这些 teachers 生成 domain-specific data，distill 进 unified model。

**直觉**：这是 mixture-of-teachers 的思路。先让每个 teacher 在各自领域专业化，再通过 distillation 让 single model 学到所有 teacher 的能力。这避免了 early-stage single model 训练时各模态 task gradient 冲突的问题。

#### Stage 2: On-Policy Distillation (OPD)

**核心问题**：模型在 audio-conditioned 下生成质量差，在 text-conditioned 下生成质量好。

**OPD 公式化**：给定 audio-text pair $(a, x)$：
1. 用 text input $x$ 生成 response $y_{\text{text}}$（质量高）
2. 用 audio input $a$ + $y_{\text{text}}$ 作为 distillation target 训练

形式化训练目标：
$$\mathcal{L}_{\text{OPD}} = -\sum_t \log p_\theta(y_{\text{text}}^{(t)} \mid a, y_{\text{text}}^{(<t)}, \text{context})$$

**直觉**：这是 modality gap bridging 的关键技术。模型 text-conditioned 能力强，但 audio-conditioned 能力弱，OPD 把 text-conditioned 的"高水位"知识蒸馏到 audio-conditioned 的"低水位"通路中。

这也解释了 Table 4 中 Qwen3.5-Omni-Plus 在 IFEval 上甚至超过 Qwen3.5-Plus-Instruct 的现象——OPD + Interaction-Aligned RL 反过来提升了 instruction following。

#### Stage 3: Interaction-Aligned Reinforcement Learning

**针对的问题**：
- 多轮对话中 unintended language code-switching
- Persona inconsistency
- Instruction-following 在 long context 中退化

**方法**：构造 multi-turn interaction trajectories，设计围绕 user experience 的 reward signals，做 RL。

直觉：这阶段是为了让模型在长时间交互中保持稳定，而不是单轮任务能力提升。这是 production-ready 的关键步骤，标准 SFT 模型往往在 multi-turn 后退化。

### 6.2 Talker 四阶段 Post-training

#### (1) General Stage

- >20M hours multilingual speech data
- 加入 instruction-following speech generation 任务
- 超越 monotonic mapping from multimodal representations to speech

#### (2) Long-Context Stage

- Data quality stratification + continual pre-training (CPT)
- 用 Qwen3-Omni-Captioner 减少 noisy data 引入的 hallucinations
- Context length 扩展到 64k tokens

#### (3) Reinforcement Learning Stage

- DPO (Direct Preference Optimization, https://arxiv.org/abs/2305.18290) with multilingual preference pairs
- 加入 rule-based rewards
- 采用 GSPO (Group Sequence Policy Optimization, https://arxiv.org/abs/2507.18071) 提升 training stability

**DPO 目标函数**：
$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

其中：
- $y_w$ 是 preferred response, $y_l$ 是 dispreferred response
- $\pi_\theta$ 是 policy model
- $\pi_{\text{ref}}$ 是 reference model
- $\beta$ 是 KL penalty 系数
- $\sigma$ 是 sigmoid function

#### (4) Speaker Fine-tuning Stage

- Lightweight fine-tuning for target speaker characteristics
- 提升 naturalness, expressiveness, controllability

---

## 7. 实验结果深度分析

### 7.1 Text→Text 性能（Table 4）

| Benchmark | Qwen3.5-Plus-Instruct | Qwen3.5-Omni-Plus |
|-----------|----------------------|-------------------|
| MMLU-Pro | 86.8 | 85.9 |
| MMLU-Redux | 94.3 | 94.2 |
| SuperGPQA | 67.4 | 66.4 |
| C-Eval | 92.3 | 92.0 |
| IFEval | 89.7 | 89.7 |
| IFBench | 51.1 | **52.6** |
| LongBench v2 | 60.2 | 59.6 |
| LiveCodeBench v6 | 67.1 | 65.6 |
| HMMT Nov 25 | 86.2 | 84.4 |

**直觉**：Qwen3.5-Omni-Plus 在 text 上几乎打平 Qwen3.5-Plus-Instruct，甚至 IFBench 略有提升。这证明 omni-modal training 不必然削弱 text capability——前提是 post-training 的 OPD 和 RL 设计得当。

### 7.2 Audio→Text vs Gemini-3.1 Pro（Table 5）

**Audio Understanding (越高越好)**:
| Benchmark | Gemini-3.1 Pro | Qwen3.5-Omni-Plus |
|-----------|---------------|-------------------|
| MMAU | 81.1 | **82.2** |
| MMAR | 83.7 | 80.0 |
| MMSU | 81.3 | **82.8** |
| RUL-MuchoMusic | 59.6 | **72.4** |
| SongFormBench-HarmonixSet (acc) | 75.6 | **81.1** |
| SongFormBench-CN (acc) | 78.1 | **87.1** |

**直觉**：Qwen3.5-Omni-Plus 在音乐理解上显著超过 Gemini-3.1 Pro（RUL-MuchoMusic +12.8, SongFormBench-CN +9.0）。这可能与 40M 小时 audio training data 中包含大量音乐数据有关。

**Dialogue**:
| Benchmark | Gemini-3.1 Pro | Qwen3.5-Omni-Plus |
|-----------|---------------|-------------------|
| VoiceBench | 88.9 | **93.1** |
| WildSpeech-Bench | 76.3 | 75.4 |

**ASR (WER↓, 越低越好)**:
| Benchmark | Gemini-3.1 Pro | Qwen3.5-Omni-Plus |
|-----------|---------------|-------------------|
| Fleurs (top60) | 7.32 | **6.55** |
| CV15 (zh) | 8.59 | **3.46** |
| Librispeech (clean) | 3.36 | **1.11** |
| Wenetspeech (net) | 23.67 | **4.30** |
| Opencpop | 6.83 | **1.49** |

**直觉**：ASR 性能全面碾压 Gemini-3.1 Pro，尤其在 Wenetspeech 上从 23.67 降到 4.30，Opencpop（singing voice）从 6.83 降到 1.49。这反映 AuT 的 robust 性，以及 40M 小时 training data 的规模效应。

### 7.3 Vision→Text 性能（Table 6）

| Benchmark | Qwen3.5-Plus-Instruct | Qwen3.5-Omni-Plus |
|-----------|----------------------|-------------------|
| MMMU | 81.0 | 80.1 |
| MathVision | 73.6 | 73.0 |
| OCRBench | 91.4 | 91.3 |
| VideoMME (w/o sub.) | 81.0 | **81.9** |
| MLVU (M-Avg) | 85.1 | **86.8** |
| LVBench | 68.6 | **71.2** |
| MMVU | 67.1 | 67.5 |

**直觉**：vision 静态能力打平，但 video understanding 全面超越。Video 任务上的优势（VideoMME +0.9, MLVU +1.7, LVBench +2.6）来自 joint video-audio training paradigm——audio-visual streams 是 real-world phenomena 的 natural representation。

### 7.4 Audio-Visual→Text 性能（Table 7）

| Benchmark | Gemini-3.1 Pro | Qwen3.5-Omni-Plus |
|-----------|---------------|-------------------|
| DailyOmni | 82.7 | **84.6** |
| WorldSense | 65.5 | 62.8 |
| AVUT | 85.6 | 85.0 |
| AV-SpeakerBench | 75.1 | 71.3 |
| VideoMME w/ audio | **89.0** | 83.7 |
| Qualcomm IVD | 66.2 | **68.5** |
| Omni-Cloze | 57.2 | **64.8** |
| OmniGAIA | **68.9** | 57.2 |

**直觉**：Qwen3.5-Omni 在 real-world audio-visual interactive scenarios（Qualcomm IVD）和 captioning（Omni-Cloze）上领先，但在 text-query 的纯视频理解（VideoMME w/audio）上落后 Gemini-3.1 Pro。这说明 Gemini 在 video-audio 融合理解上仍有优势，可能因为 Gemini 1.5 是 native multimodal trained from scratch（参考 https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf）。

### 7.5 Zero-Shot Speech Generation（Table 8）

| Model | SEED test-zh WER↓ | SEED test-en WER↓ |
|-------|-------------------|-------------------|
| Seed-TTS ICL | 1.11 | 2.24 |
| CosyVoice 3 | 0.71 | 1.45 |
| MiniMax-Speech | 0.83 | 1.65 |
| Qwen3-Omni-30B-A3B | 1.07 | 1.39 |
| **Qwen3.5-Omni-Plus** | **0.99** | **1.26** |

**直觉**：Qwen3.5-Omni-Plus 在 test-en 上是 SOTA，但在 test-zh 上不如 CosyVoice 3 和 MiniMax-Speech。可能因为 Qwen3.5-Omni-Plus 是 omni model，TTS 只是其能力之一，无法与 dedicated TTS model 在所有指标上竞争。

### 7.6 Cross-Lingual Speech Generation（Table 11）

| Direction | Qwen3.5-Omni-Plus | CosyVoice 3 |
|-----------|-------------------|-------------|
| English→Chinese | **4.86** | 5.09 |
| Korean→Chinese | **0.84** | 1.06 |
| Chinese→English | **2.18** | 2.98 |
| Chinese→Korean | **4.03** | 14.4 |
| English→Korean | **3.72** | 5.87 |
| Japanese→Korean | **5.12** | 7.92 |

**直觉**：Qwen3.5-Omni 在 12 个方向中 10 个领先，zh-to-korean 把 CosyVoice 3 的 14.4 降到 4.03（72% 相对降幅）。这是 native omni training 的红利——模型同时见过大量中文和韩文 speech data，能学到 cross-lingual phoneme mapping。

### 7.7 Custom-Voice Multilingual（Table 12）

29 种语言对比 ElevenLabs、Gemini-2.5 Pro、GPT-Audio、MiniMax：

| Language | Qwen3.5-Omni-Plus | Best competitor |
|----------|-------------------|-----------------|
| Chinese | **0.785** | MiniMax 0.786 |
| English | **0.839** | ElevenLabs 1.126 |
| Japanese | **3.306** | MiniMax 4.254 |
| Korean | **1.309** | ElevenLabs 3.981 |
| Thai | **1.653** | MiniMax 1.811 |
| Vietnamese | **1.320** | MiniMax 1.058 |
| Hebrew | 7.680 | Gemini-2.5 Pro 4.459 |
| Icelandic | 10.322 | Gemini-2.5 Pro 6.348 |

**直觉**：Qwen3.5-Omni 在 10 种语言上是 best，尤其在 Asian languages（日、韩、泰、越）上明显领先。ElevenLabs 在 Hebrew（102.018 WER）和 Thai（114.813 WER）上几乎完全失败，反映其 training data 的 language coverage 限制。

---

## 8. 与 GPT-4o、Gemini 的对比直觉

### 8.1 Architecture 差异

| 模型 | Architecture | Streaming-first | Native Multimodal |
|------|--------------|-----------------|-------------------|
| GPT-4o | undisclosed | Yes (https://openai.com/index/hello-gpt-4o/) | Yes |
| Gemini 1.5/3.1 | undisclosed | Yes | Yes (https://arxiv.org/abs/2507.06261) |
| Qwen3.5-Omni | Thinker-Talker + Hybrid MoE | Yes | Yes |

Qwen3.5-Omni 的 Thinker-Talker 架构设计有独特优势：Thinker 与 Talker 分离但 Talker 直接接收 Thinker 高层表征，避免了 text-to-speech 的 cascade degradation。这种设计接近 GPT-4o 的 "single model handles audio in and out" 但更可解释。

### 8.2 Latency 对比

GPT-4o 平均 response latency ~320ms（按 OpenAI 公开 demo），Qwen3.5-Omni-Plus audio input latency 435ms（Plus）或 235ms（Flash）。Qwen3.5-Omni-Flash 实际比 GPT-4o 公开 demo 还快，这反映 Hybrid MoE + chunked prefilling 的工程优化到位。

### 8.3 AGI 路径直觉

Qwen3.5-Omni 的设计哲学：**native omni agent model**。这与 OpenAI 的 GPT-4o 和 Google 的 Gemini 1.5 路径一致，但 Qwen 团队强调 **agentic behavior**——WebSearch、FunctionCall、Audio-Visual Vibe Coding——这是迈向 AGI 的关键一步。

**Audio-Visual Vibe Coding** 的 emergent capability 特别值得注意：模型直接从 audio-visual instructions 生成 executable code。这是 zero-shot coding from multimodal context 的能力，无需 external orchestration。直觉上，这接近于 OpenAI 的 Operator 或 Anthropic 的 Computer Use，但在 omni-modal 层面。

---

## 9. 局限性与未来方向

### 9.1 论文中观察到的 gap

1. **VideoMME w/ audio 落后 Gemini-3.1 Pro**：83.7 vs 89.0，说明 audio-visual 融合理解仍有提升空间
2. **WildSpeech-Bench 落后 Gemini-3.1 Pro**：75.4 vs 76.3，说明 wild speech 场景下 Qwen3.5-Omni 的 robustness 略弱
3. **WorldSense 落后**：62.8 vs 65.5，real-world omni-modal reasoning 仍需提升
4. **OmniGAIA 大幅落后 Gemini-3.1 Pro**：57.2 vs 68.9，agentic tool use 在 omni-modal setting 下仍不成熟

### 9.2 推测的下一步

1. **Scaling AuT**：40M 小时数据已很大，但相对 Gemini 可能仍不够。下一步可能扩到 100M+ hours。
2. **MTP 扩展**：当前 MTP 用于 speech codec residual codebooks，未来可能扩展到 video generation（参考 OpenAI Sora 的 diffusion + transformer 路径）。
3. **更激进的 RL**：当前 Interaction-Aligned RL 只针对 multi-turn，未来可能扩展到 RL from environment feedback（如 tool use 闭环）。
4. **Long-context 扩展**：256k 已是 SOTA，但 10 小时音频仅占用 ~225k tokens，长视频仍是瓶颈。下一步可能 mega-context (1M+) 或 alternative compression。

---

## 10. 关键 Takeaways

1. **Hybrid MoE + GDN** 是 streaming long-context omni-modal 的关键 enabler
2. **ARIA** 是 text-speech alignment 的优雅解决方案，把对齐约束转化为 self-consistency
3. **OPD** 是 modality gap bridging 的核心技巧，让 omni model 不损失 text capability
4. **Three-stage pretraining**（encoder alignment → general → long context）是 long-context multimodal 训练的标准 recipe
5. **Specialist Distillation + OPD + Interaction-Aligned RL** 三阶段 post-training 是 omni agent production-ready 的关键路径
6. **Audio-Visual Vibe Coding** 是 emergent capability，预示 omni-modal agent 的未来方向

参考 paper:
- Qwen2.5-Omni: https://arxiv.org/abs/2503.20215
- Qwen3-Omni: https://arxiv.org/abs/2509.17765
- Gemini 2.5: https://arxiv.org/abs/2507.06261
- DPO: https://arxiv.org/abs/2305.18290
- GSPO: https://arxiv.org/abs/2507.18071
- MiniMax-Speech: https://arxiv.org/abs/2505.07916
- CosyVoice 3: https://arxiv.org/abs/2505.17589
- MMSU: https://arxiv.org/abs/2506.04779
- MMAR: https://arxiv.org/abs/2505.13032
- Seed-TTS: https://arxiv.org/abs/2406.02430
- MMAU: https://arxiv.org/abs/2410.19168
- F5-TTS: https://arxiv.org/abs/2410.06885
- E2 TTS: https://ieeexplore.ieee.org/document/10892102
- VoiceBench: https://arxiv.org/abs/2410.17196
- Video-MME: https://arxiv.org/abs/2405.21075
- MathVision: https://arxiv.org/abs/2402.14804
- MMMU: https://arxiv.org/abs/2311.16502
- CharXiv: https://arxiv.org/abs/2406.18521
- LongBench v2: https://aclanthology.org/2025.acl-long.183/
- ZEROBench: https://arxiv.org/abs/2502.09696
- SuperGPQA: https://arxiv.org/abs/2502.14739
- MMLU-Pro: https://arxiv.org/abs/2406.01574
- IFEval: https://arxiv.org/abs/2311.07911

Qwen3.5-Omni 代表了 2026 年 omni-modal LLM 的 state-of-the-art 工程。它的关键贡献不在单一算法创新，而在 systematic engineering——从 AuT encoder、Hybrid MoE、ARIA、three-stage pretraining 到 OPD post-training，每个环节都有 deliberate design。这种 holistic approach 正是 omni-modal model 从 research prototype 走向 production 的必要路径。
