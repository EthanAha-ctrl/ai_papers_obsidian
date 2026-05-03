**ElevenLabs** 是一家总部位于 **London** 和 **Warsaw** 的 **AI Audio** 公司，由 **Piotr Dąbkowski**（前 **Google** 机器学习工程师）和 **Mati Staniszewski**（前 **Palantir** 部署策略师）于 **2022年** 联合创立。两位创始人均为 **Poland** 裔，他们创业的初始动机源自对 **Hollywood** 电影 **Polish** 配音质量低下的不满——这个看似简单的痛点驱动了一家目前估值 **$11 Billion** 的公司。ElevenLabs 已经从 "单纯的 TTS 工具" 演进为一个 **full-stack AI Audio platform**

- **最新融资**: 2026年2月，完成 **$500M Series D**，由 **Sequoia Capital** 领投，**a16z** 跟投，估值 **$11B**（较上一轮三倍增长）
- **ARR**: 2025年达到 **$330M**，2026年目标翻倍至 ~**$660M**
- **团队**: 约 **500-580人**
- **客户**: 41% 的 **Fortune 500** 企业在使用

| 产品线                      | 功能描述                  | 核心技术                                    |
| ------------------------ | --------------------- | --------------------------------------- |
| **Text-to-Speech (TTS)** | 将文本转化为极度拟人的语音         | Eleven v3 Model                         |
| **Voice Cloning**        | 克隆任何人的声音              | Speaker Embedding + Few-shot Adaptation |
| **ElevenLabs Agents**    | 实时对话式 AI Voice Agent  | End-to-end Voice Pipeline               |
| **AI Dubbing**           | 自动将视频/音频翻译配音到 70+ 种语言 | Multilingual TTS + Voice Preservation   |
| **Scribe (STT)**         | Speech-to-Text 转录     | ASR Model (Scribe v2)                   |
| **Sound Effects**        | AI 生成音效               | Audio Generation Model                  |
| **AI Music**             | AI 生成音乐               | Music Generation Model                  |
| **Voice Library**        | 10,000+ 预制 Voice 供选择  | Community-contributed Voices            |
### 3.1 TTS Model Architecture: 从 Autoregressive Transformer 到 Latent Diffusion

ElevenLabs 的模型演进代表了整个 TTS 行业的架构转变：

#### 早期模型（v1/v2）：Autoregressive Transformer

传统 TTS Pipeline 通常分为三个阶段：

```
Text → [Text Encoder] → [Acoustic Model] → [Vocoder] → Audio Waveform
```

**Autoregressive (AR) Transformer** 的工作方式类似 GPT——逐个预测下一个 **audio token**：

$$P(x_1, x_2, ..., x_T) = \prod_{t=1}^{T} P(x_t | x_1, ..., x_{t-1}, \text{text})$$

其中：
- $x_t$ = 第 $t$ 个 **audio token**（通常是通过 **codec model** 如 **EnCodec** 离散化的音频表示）
- $T$ = 总的 token 序列长度
- 条件 $\text{text}$ = 输入文本的 embedding

**问题**：
- **累积误差 (error accumulation)**：每一步的小误差会滚雪球
- **延迟高 (high latency)**：必须按顺序生成，无法并行
- **幻觉 (hallucination)**：可能跳过或重复词汇

#### Eleven v3：Transformer-based Diffusion Network

根据公开信息，v3 模型约 **300M parameters**，采用 **Transformer-based Diffusion** 架构，平均延迟 **~120ms**。

**Diffusion Model** 的核心思想是：

**前向过程 (Forward Process)**：逐步向 clean audio latent 添加 Gaussian noise

$$q(z_t | z_{t-1}) = \mathcal{N}(z_t; \sqrt{1 - \beta_t} \cdot z_{t-1}, \beta_t \cdot I)$$

其中：
- $z_t$ = 第 $t$ 步的 **noisy latent representation**
- $z_0$ = 原始 clean audio 的 latent encoding
- $\beta_t$ = 第 $t$ 步的 **noise schedule**，控制噪声添加速率
- $I$ = Identity matrix
- $\mathcal{N}$ = Gaussian distribution

**反向过程 (Reverse Process)**：模型学习去噪，从纯噪声恢复 audio

$$p_\theta(z_{t-1} | z_t, \text{text}, \text{speaker}) = \mathcal{N}(z_{t-1}; \mu_\theta(z_t, t, c), \sigma_t^2 I)$$

其中：
- $\theta$ = 模型可学习参数
- $\mu_\theta$ = 模型预测的去噪后均值
- $c$ = **conditioning signal**（包含 text embedding、speaker embedding、emotion tags 等）
- $\sigma_t$ = 第 $t$ 步的方差

**训练目标** 通常简化为预测添加的噪声：

$$\mathcal{L} = \mathbb{E}_{z_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(z_t, t, c) \|^2 \right]$$

其中：
- $\epsilon$ = 实际添加的 Gaussian noise
- $\epsilon_\theta$ = 模型预测的 noise
- $\| \cdot \|^2$ = L2 loss（均方误差）

**为什么 Diffusion 优于 AR 用于 Audio？**

| 特性 | Autoregressive | Diffusion |
|------|---------------|-----------|
| 生成方式 | 逐 token 顺序生成 | 全序列并行去噪 |
| 延迟 | 高（与序列长度成正比） | 低（固定步数去噪） |
| 质量一致性 | 误差累积 | 全局一致性更好 |
| 情感控制 | 困难 | 通过 conditioning 更自然 |

> 参考: [WildRun AI Guide](https://wildrunai.com/blog/elevenlabs-complete-guide-to-ai-voice-synthesis-technology) | [Morvoice Latent Diffusion](https://www.morvoice.com/zh/blog/latent-diffusion-vs-transformers-audio-generation) | [ARDiT Paper](https://arxiv.org/abs/2406.05551)

---

### 3.2 Audio Tags 系统：Director-level 情感控制

**Eleven v3** 最突出的创新是 **Audio Tags**——一种在文本中嵌入的 **inline markup**，让用户像导演一样控制语音的情感和风格：

```
[whispers] I have a secret to tell you. [excited] But it's amazing news!
[laughs] I can't believe it happened! [sad] But then everything changed.
```

支持的 Tag 类型：

| 类别 | 示例 Tags |
|------|----------|
| **Emotion** | `[curious]` `[crying]` `[mischievously]` `[excited]` `[sad]` |
| **Delivery** | `[whispers]` `[shouts]` `[sings]` `[monotone]` |
| **Sound Effects** | `[laughs]` `[sighs]` `[gasps]` `[clears throat]` |
| **Pacing** | `[slowly]` `[quickly]` `[pauses]` |

**技术实现原理**：

Audio Tags 本质上是 **conditional control signals**。在 Diffusion 模型的去噪过程中，这些 tags 被编码为额外的 **conditioning vectors** $c_{\text{tag}}$，与 text embedding $c_{\text{text}}$ 和 speaker embedding $c_{\text{spk}}$ 一起注入模型：

$$\mu_\theta(z_t, t, c) = \mu_\theta(z_t, t, c_{\text{text}} \oplus c_{\text{spk}} \oplus c_{\text{tag}})$$

其中 $\oplus$ 表示 conditioning 的融合操作（可能是 concatenation、cross-attention 或 **FiLM conditioning**——Feature-wise Linear Modulation）。

**FiLM Conditioning** 的公式为：

$$\text{FiLM}(h, c) = \gamma(c) \odot h + \beta(c)$$

其中：
- $h$ = 中间 hidden representation
- $\gamma(c)$ = 由 conditioning signal $c$ 生成的 scale factor
- $\beta(c)$ = 由 conditioning signal $c$ 生成的 shift factor
- $\odot$ = element-wise 乘法

这使得模型可以在 **inference time** 动态调整语音的情感特征，而无需为每种情感单独训练模型。

> 参考: [ElevenLabs v3 Audio Tags](https://elevenlabs.io/blog/v3-audiotags) | [ElevenLabs v3 页面](https://elevenlabs.io/v3)

---

### 3.3 Voice Cloning：Speaker Embedding 技术

ElevenLabs 提供两种 Voice Cloning 模式：

#### Instant Voice Cloning (IVC)
- 仅需 **1分钟** 的音频样本
- 基于 **few-shot adaptation**：模型在 inference 时将音频样本作为 **conditioning signal**
- 不改变模型权重

工作流程：

```
Audio Sample → [Speaker Encoder] → Speaker Embedding (128-512维向量)
                                        ↓
Text Input → [Text Encoder] ──→ [TTS Model + Speaker Conditioning] → Cloned Voice Audio
```

**Speaker Encoder** 是一个独立的神经网络（通常基于 **d-vector** 或 **x-vector** 架构），将任意长度的音频压缩为一个固定维度的 **speaker embedding** $e_s \in \mathbb{R}^d$：

$$e_s = f_{\text{enc}}(\text{mel}(a))$$

其中：
- $a$ = 原始音频 waveform
- $\text{mel}(a)$ = Mel-spectrogram 表示
- $f_{\text{enc}}$ = Speaker Encoder 网络
- $d$ = embedding 维度（通常 128-512）

这个 embedding 捕获了说话者的 **timbre（音色）**、**pitch range（音域）**、**speaking rhythm（节奏）** 等声学特征。

#### Professional Voice Cloning (PVC)
- 需要 **30分钟+** 的高质量音频
- 通过 **fine-tuning** 模型的部分参数来更精确地捕捉声音特征
- 生成质量更高、更难以与真人区分

> 参考: [ElevenLabs Voice Cloning Docs](https://elevenlabs.io/docs/eleven-api/concepts/voice-cloning) | [ElevenLabs Voice Cloning](https://elevenlabs.io/voice-cloning)

---

### 3.4 Scribe：Speech-to-Text (ASR) 模型

**Scribe v2** 是 ElevenLabs 的 ASR (Automatic Speech Recognition) 模型，号称是世界上最准确的转录模型：

- 支持 **99+ 种语言**
- 内置 **Speaker Diarization**（说话人分离）
- 提供 **Scribe v2 Realtime** 用于实时转录
- 在多项 benchmark 上超越 **OpenAI Whisper**

ASR 模型的一般架构：

```
Audio Waveform → [Feature Extractor (Mel/MFCC)] → [Encoder (Transformer/Conformer)] → [Decoder] → Text
```

核心使用的可能是 **Conformer** 架构（Convolution + Transformer 的混合），其 attention 机制为：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

其中：
- $Q$ = Query matrix
- $K$ = Key matrix  
- $V$ = Value matrix
- $d_k$ = Key 的维度，$\sqrt{d_k}$ 用于缩放防止 dot-product 过大

> 参考: [ElevenLabs Scribe](https://elevenlabs.io/blog/meet-scribe) | [ElevenLabs STT](https://elevenlabs.io/speech-to-text)

---

### 3.5 Conversational AI Agents

**ElevenLabs Agents** 是一个 **end-to-end Voice Agent** 平台，将整个语音对话 pipeline 整合在一起：

```
User Speech → [Scribe STT] → Text → [LLM (GPT-4/Claude/etc.)] → Response Text → [TTS v3] → Voice Response
                                              ↑
                                    [Knowledge Base / Tools / API]
```

关键技术指标：
- **TTFB (Time To First Byte)**: ~120ms（TTS 部分）
- 支持 **interruption handling**（用户打断时的处理）
- 支持 **function calling**（Agent 可以调用外部 API）
- 部署方式：**WebSocket**、**Phone (Twilio)**、**Widget embed**

这本质上是一个 **real-time audio loop**：

$$\text{Latency}_{\text{total}} = t_{\text{STT}} + t_{\text{LLM}} + t_{\text{TTS}} + t_{\text{network}}$$

ElevenLabs 通过控制 STT 和 TTS 两端来最小化总延迟。

> 参考: [ElevenLabs Agents](https://elevenlabs.io/agents) | [Deepgram Voice Agent Guide](https://deepgram.com/learn/elevenlabs-real-time-voice-agent)

---

## 四、商业模型与定价

ElevenLabs 采用 **SaaS subscription + usage-based** 混合模式：

| Plan | 价格 | 特点 |
|------|------|------|
| **Free** | $0/month | 10,000 characters/month |
| **Starter** | $5/month | 30,000 characters |
| **Creator** | $22/month | 100,000 characters |
| **Pro** | $99/month | 500,000 characters + PVC |
| **Scale** | $330/month | 2,000,000 characters |
| **Enterprise** | 定制 | 无限量 + SLA + 专属支持 |

**收入构成**: API usage（开发者集成）+ 直接平台使用 + Enterprise 合同。41% 的 **Fortune 500** 公司使用他们的服务，显示出强大的 **B2B** 渗透力。

> 参考: [Sacra Analysis](https://sacra.com/c/elevenlabs/) | [GetLatka Revenue](https://getlatka.com/companies/elevenlabs.io)

---

## 五、第一性原理思考：为什么 ElevenLabs 能赢？

从第一性原理来看，**语音是信息传递最自然的界面**。人类进化了数十万年的语言能力，而文字只有几千年历史。因此：

1. **Voice 是 AI 的终极 interface layer**：随着 LLM 变得越来越聪明，**Voice** 成为人机交互的最后一公里。ElevenLabs 占据了这个 chokepoint。

2. **Quality 的 compounding effect**：TTS 质量一旦突破 "uncanny valley"（恐怖谷），adoption 会呈指数增长。ElevenLabs 的 v3 已经到达或超越了这个 threshold。

3. **Full-stack audio 的 network effect**：
   - STT (Scribe) + LLM integration + TTS (v3) = 完整的 **Voice Agent** stack
   - 每增加一个产品，用户切换成本就增加
   - Voice Library 的 community 贡献形成了数据飞轮

4. **Data moat**：每天处理的数十亿 characters 的 TTS 请求产生了海量的训练反馈数据（用户选择、重试、评分），这是竞争对手难以复制的。

5. **从 "工具" 到 "平台" 的跃迁**：ElevenLabs 不再只是一个 TTS API——它是一个 **AI audio operating system**，涵盖了从创作（TTS、Music、Sound Effects）到部署（Agents）到理解（Scribe）的全链路。

---

## 六、竞争格局

| 竞争对手 | 核心产品 | 差异化 |
|---------|---------|--------|
| **OpenAI** | Whisper (STT) + TTS API | LLM 集成优势 |
| **Google Cloud TTS** | WaveNet / Neural2 | 云生态绑定 |
| **Amazon Polly** | Neural TTS | AWS 生态 |
| **Microsoft Azure Speech** | Neural TTS + STT | Enterprise 渗透 |
| **PlayHT** | TTS API | 价格竞争 |
| **Cartesia** | Sonic TTS | 超低延迟 |
| **Deepgram** | STT + TTS | Real-time 专注 |

ElevenLabs 的核心壁垒在于 **voice quality** 仍然是行业最佳，同时 **full-stack** 覆盖使其成为一站式选择。

> 参考: [ElevenLabs Review 2026](https://9to6ai.com/elevenlabs-review-2026-best-ai-voice-tool/) | [Forbes Profile](https://www.forbes.com/profile/mati-staniszewski/)

---

**总结**：ElevenLabs 本质上是在做一件事——**让 AI 拥有人类级别的声音**。从第一性原理看，这等同于构建 AI 时代的 "声带"。当所有 AI Agent 都需要说话时，ElevenLabs 就是那个提供声带的公司。这解释了为什么它从一个简单的 "配音质量不好" 的 insight，在不到4年内成长为一家 $11B 的公司。