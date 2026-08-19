---
source_pdf: Gemini Embedding 2 A Native Multimodal.pdf
paper_sha256: f8844b01afdea7eaeaa0abd3f89f01c3e311cc505f1d83e4cd58e193c8cf76e7
processed_at: '2026-08-19T09:02:04-07:00'
target_folder: LLM-engine
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Gemini Embedding 2 人话版

Karpathy 你好，我用大白话再过一遍这篇 paper，不讲那些虚的，直接说它干了啥、为什么这么干、效果怎么样。

参考：
- Paper 前代: https://arxiv.org/abs/2503.07891
- Gecko (Google embedding 路线前身): https://arxiv.org/abs/2403.20327
- Matryoshka Representation Learning: https://arxiv.org/abs/2205.14547
- Model Soups: https://arxiv.org/abs/2203.05482
- MoCa (bidirectional 改造): https://arxiv.org/abs/2506.23115

---

## 一句话总结

Google 把 Gemini 2.5 这个 multimodal LLM 直接拿来当 backbone，fine-tune 成一个 embedding 模型，能同时吃 text、image、video、audio，输出一个 unified vector。效果上把 CLIP、SigLIP 2、Voyage 这些 specialist 模型按在地上摩擦。

---

## 它在解决什么问题

之前的 multimodal embedding 模型（CLIP、ALIGN、SigLIP 2、CoCa）都是 dual-tower 架构：image 走 image encoder，text 走 text encoder，两个 encoder 各算各的，最后才用 cosine similarity 拉到一起。

这有两个硬伤：

**硬伤 1：modality 之间深层交互根本没发生。** 两个 encoder 在各自的 transformer 里独立计算，到最后 cosine 那一刻才见面。这就像让两个人分别读完一本书再交流心得，跟两个人一起边读边讨论，理解的深度完全不在一个量级。

**硬伤 2：mixed input 处理不了。** 你想 embed 一个 query："这张图里的红色汽车在视频第几秒出现的"，同时输入 image + video + text，dual-tower 根本接不了这种 interleaved input。它只能分别 embed 每个 modality 然后 concatenate，这就丢失了 token 级的精细对齐。

Gemini Embedding 2 的解法很粗暴：直接拿 Gemini 2.5 当 backbone。Gemini 2.5 本来就是 native multimodal，transformer 内部每层都有 cross-modal attention，image token、text token、video token、audio token 全部混在一起 attend。这就是所谓的 "native multimodal"，modality 交互发生在每一层，不是只在最后。

---

## 架构长什么样

整个 pipeline 三步：

```
raw input (text/image/video/audio/混合)
  → Gemini 的 native tokenizer 转成 token sequence
  → bidirectional transformer M (从 Gemini 2.5 初始化)
  → mean pooling 把所有 token embedding 平均一下
  → random linear projection f 投到目标维度
  → 最终 embedding E (3072 维)
```

形式化：

$$\mathbf{T}_{\text{embed}} = \mathcal{M}(\mathbf{T}) \in \mathbb{R}^{L \times d_M}$$

$$\mathbf{P}_{\text{embed}} = \mathcal{P}(\mathbf{T}_{\text{embed}}) \in \mathbb{R}^{d_M}$$

$$\mathbf{E} = f(\mathbf{P}_{\text{embed}}) \in \mathbb{R}^{d}$$

变量说人话：
- $\mathbf{T}$：输入的 token sequence，长度 $L$（比如一张图变成 256 个 visual token，text 10 个 token，总长 266）
- $\mathcal{M}$：transformer backbone，hidden dimension 是 $d_M$
- $\mathcal{P}$：pooler，就是 mean pooling，沿 sequence 维度取平均
- $f$：一个 random initialized 的 linear layer，把 $d_M$ 投到 $d=3072$
- $\mathbf{E}$：最终输出的 embedding

### 为什么用 bidirectional 不用 causal

Gemini 2.5 原本是 causal attention（autoregressive generation 用的）。但 embedding 任务需要 dense representation，每个 token 都要积累全局 context。Causal attention 的问题在于最后一个 token 能看到所有前面的 token，但第一个 token 什么都看不到，信息分布极度不均。Mean pooling 把所有 token 平均，causal 的话前面的 token 信息密度太低，等于拖后腿。

Bidirectional 让每个 token 都能 attend 到所有其他 token，所有 token 信息密度均匀，mean pooling 才能真正 work。

相关工作 MoCa [21] 的做法是在 causal backbone 上加 joint reconstruction objective 强制 bidirectional reasoning，属于 patch causal 的缺陷。Gemini Embedding 2 直接用 bidirectional，绕过了这个 trick。

### 为什么 mean pooling

[23] 的实证研究说简单的 mean pooling 在 decoder-based LM 适配 encoder task 时已经够用。CLS token 容易受 sequence length 影响，attention pooling 又增加额外参数和复杂度。Mean pooling 对长度变化 robust，video、长文档这种长度波动大的输入尤其友好。

### Linear projection $f$ 为什么 random initialized

$f$ 是 dimension adapter，把 Gemini 的 hidden size 投到 3072 维 output。Random init 是因为这部分需要 from scratch 学习如何把 transformer representation 投到 contrastive learning 期望的 cosine-similarity-friendly space。同时 MRL [25] 要求不同 prefix 维度都能独立可用，$f$ 必须是 dense matrix，前 768、1536、3072 维分别 optimized。

### Modality-specific tokenization

Paper 没给细节，但可以推断：
- **Text**：Gemini 的 SentencePiece tokenizer
- **Image**：Gemini native vision encoder（ViT 变体 + patch embedding），输出 visual tokens
- **Video**：1 FPS 采样，up to 32 frames（实验部分提到），每帧当 image 处理 + temporal position encoding
- **Audio**：Gemini native audio tokenizer（Enformer 或升级版）

所有 modality 的 token 进同一个 transformer，这就是 native multimodal 的本质。

---

## 损失函数

### 基本形式

NCE loss with in-batch negatives [24]。每个 training example 有 query $q_i$、positive target $p_i^+$，可选 hard negative $p_i^-$。text-only task 还有 task string $t$（如 "question answering"）。

Embedding 计算：

$$\mathbf{q}_i = f(\text{mean\_pool}(\mathcal{M}(t \oplus q_i)))$$

$$\mathbf{p}_i^{\pm} = f(\text{mean\_pool}(\mathcal{M}(p_i^{\pm})))$$

变量说人话：
- $t$：task string（instruction prefix，如 "retrieval"、"classification"）
- $\oplus$：字符串拼接
- $q_i$：query 内容
- $p_i^+$：positive target
- $p_i^-$：hard negative target

### 主 Loss（公式 2）

$$\mathcal{L} = \frac{1}{B} \sum_{i=1}^{B} \left[ -\log \frac{e^{\text{sim}(\mathbf{q}_i, \mathbf{p}_i^+) / \tau}}{e^{\text{sim}(\mathbf{q}_i, \mathbf{p}_i^+) / \tau} + e^{\text{sim}(\mathbf{q}_i, \mathbf{p}_i^-) / \tau} + \sum_{j=1}^{B} \text{mask}(i, j) e^{\text{sim}(\mathbf{q}_i, \mathbf{p}_j^+) / \tau}} \right]$$

说人话：分子是 query 和 positive 的相似度，分母是 positive + hard negative + batch 内所有其他 positive（作为 in-batch negatives）。Temperature $\tau$ 控制分布的 sharpness。

$\text{sim}(\mathbf{x}, \mathbf{y}) = \mathbf{x}^\top \mathbf{y} / \|\mathbf{x}\| \|\mathbf{y}\|$ 是 cosine similarity。

### Mask 项（公式 3）

$$\text{mask}(i, j) = \begin{cases} 0 & \text{if } q_i = q_j \text{ or } p_i^+ = p_j^+ \\ 1 & \text{otherwise} \end{cases}$$

这个 mask 解决一个具体问题：classification task 的 label 数量少，同一个 batch 里很可能多个 example 的 positive target 是同一个 label（比如 "sports" 标签）。如果不 mask，这些相同 label 会被当作 hard negatives，loss 信号严重错误。Classification task 的 in-batch negative contrastive 训练里这个 mask 几乎必须。

### Task string dropout

训练时随机 drop 掉 task string $t$，让模型对 "有 instruction" 和 "无 instruction" 两种输入都 robust。这是 input-level regularization，让 embedding space 同时支持 instruction-tuned query 和 raw query。Inference 时用户用不用 task prefix 都行。

### MRL 多分辨率

$$\mathcal{L}_{\text{total}} = \sum_{k \in \{768, 1536, 3072\}} \mathcal{L}_k(\mathbf{E}_{[:k]})$$

让 embedding 前 768 维、前 1536 维、前 3072 维分别都能独立用于 retrieval。像俄罗斯套娃，外层是内层的精化。

**实际价值**：RAG 场景用 768 维做 ANN search（快、省存储），用 3072 维做 re-ranking（精度高）。一个模型、一份 inference 多次复用。

---

## 训练 Recipe

三阶段，这是 paper 最有价值的部分。

### Stage 1: Pre-Fine-Tuning (PFT)

**目的**：把 autoregressive generation 模型转成 encoder。范式转换：从"预测下一个 token"到"产出有判别力的 dense vector"。

**做法**：
- 大量可能 noisy 的 query-target pairs
- multi-task setup
- 仅 image、text、code（**不包含 audio/video**）
- 单 task 构造 batch
- 大 batch size（提供更稳定 gradients，缓解 noisy input 冲击）

**为什么 PFT 不加 audio/video**？我推测两个原因：
1. PFT 是范式转换，需要数据相对干净。Text/image/code 是结构化 paired data，audio/video 的 paired data 稀少且 noisy
2. Audio/video 的 native 处理需要模型先具备 bidirectional fusion 能力，PFT 阶段模型还没"学会"encoder 范式，加 audio/video 容易 training collapse

### Stage 2: Fine-Tuning (FT)

**目的**：在掌握 encoder 范式后，加入所有 modality 和 task 类型，构建 unified embedding space。

**做法**：
- text、code、document、image、audio、video 全 modality tasks
- 大部分 task 包含 query、target、hard negative triplets
- 针对 task 调优 batch size（不同 task sample 复杂度不同）
- 单 task 构造 batch
- single-modality batch + cross-modality batch 交替

**关键设计**：single-modality batch + cross-modality batch 交替训练。Single-modality batch 让 model 在每个 modality 内部建立 robust representation（intra-modality alignment）。Cross-modality batch 让 model 在 modality 间对齐（inter-modality alignment）。两者结合避免 modality collapse（model 只学 text-text 忽略 cross-modal）或 modality bias（参考 MM-Embed [22]）。

**Sampling rate 是 empirical 调出来的**：Paper 明确说"the sampling rates of each of the different tasks are defined empirically"，"balancing overall performance across all modalities was sensitive to hyper-parameters like sampling rates and batch sizes"。工业级训练的常见现象：理论分析给不出最优配比，只能 grid search。Google 靠 scale 和算力做了大量 ablation。

### Stage 3: Model Soup

**目的**：systematize 不同 checkpoint 组合，提升 generalization。

**三种 soup 策略**：
1. **同 training run 不同 checkpoints 平均**（SWA 风格 [26]）：SGD 后期在 flat minima 附近游走，平均权重等价于找更宽的 optima，泛化好
2. **不同 training runs 的 checkpoints 平均**（Model Soups [27]）：不同 run 用不同 seed 或 task 比例，soup 后平均掉 task-specific 偏好
3. **Weighted average**：不同权重组合

Table 6 的 ablation 很说明问题：

| Config | MSR-VTT | YouCook2 | Vatex |
|--------|---------|----------|-------|
| Baseline | 68.2 | 55.9 | 69.2 |
| + MSR-VTT data | 75.0 (+6.8) | 56.1 (+0.2) | 71.7 (+2.5) |
| + MSR-VTT & Vatex | 76.1 (+7.9) | 55.3 (-0.6) | 79.5 (+10.3) |
| Soup 2:1 | 71.7 (+3.5) | 56.1 | 74.5 (+5.3) |
| Soup 1:1 | 73.7 (+5.5) | 56.8 (+0.9) | 76.8 (+7.6) |

加 in-domain data 提升目标 task 但伤害 out-of-domain task，因为 fine-tune 让 model "偏向" 目标 task distribution。Model soup 通过权重平均，让原 base model 的 robustness 和 fine-tuned model 的 task-specific 能力共存。本质是 loss landscape 在 base 和 fine-tuned 之间存在一条相对 flat 的路径，线性插值的权重在两个 basin 之间保持稳定。

---

## 实验结果核心数据

### Multimodal Retrieval (Table 1)

| Benchmark | Gemini Embedding 2 | Amazon Nova MME | Voyage-3.5-MM | Legacy Google |
|-----------|---------------------|-----------------|---------------|---------------|
| GUIEC (Img→Img) | **79.4** | 68.6 | 69.4 | 69.5 |
| MSCOCO (T→I) | **62.9** | 57.2 | 58.1 | 53.1 |
| Flickr30k (T→I) | 89.1 | 81.6 | **89.9** | 81.4 |
| DOCCI (T→I) | **93.4** | 84.0 | 83.8 | 74.0 |
| TextCaps (T→I) | **89.6** | 76.0 | 79.4 | 74.0 |
| MSCOCO (I→T) | **78.8** | 68.3 | 74.5 | 68.2 |
| Flickr30k (I→T) | **97.4** | 87.5 | 94.5 | 94.0 |
| Vatex (T→V) | **68.8** | 60.3 | 55.2 | 54.9 |
| MSR-VTT (T→V) | **68.0** | 67.0 | 63.0 | 57.9 |
| YouCook2 (T→V) | **52.5** | 34.7 | 31.4 | 34.9 |
| EncyclopedicVQA | **71.5** | - | 58.6 | - |
| ViDoRe V2 | 64.9 | 60.6 | **65.5** | 28.9 |
| **Overall** | **77.2** | 68.2 | 70.0 | 64.1 |
| Modality | **V/A/I/T** | V/A/I/T | V/I/T | I/T |

几个关键 takeaways：

1. **DOCCI 和 TextCaps（long-caption）优势显著**（93.4 vs 84.0 vs 83.8）。Long caption 需要理解图像的 spatial reasoning、scene text、long-tail concepts，late-fusion 模型做不到这个 level。这是 deep fusion 的直接 evidence。

2. **EncyclopedicVQA 71.5 vs 58.6（+12.9）**。这个 task 是 image+text 联合 query 检索 answer，最能体现 native multimodal 价值。Interleaved input "image + 问题" 直接 embed 成一个 vector 检索答案，late-fusion 必须先分别 encode 再 concat，性能差是必然的。

3. **YouCook2 52.5 vs 34.7**。YouCook2 是 cooking instruction video，需要 fine-grained temporal understanding（"在视频第几秒切洋葱"）。Gemini Embedding 2 在 video 上的 native 处理让它在 temporal reasoning 上优势明显。

4. **ViDoRe V2 64.9 vs Voyage 65.5**。Document retrieval 上 Gemini 略输 Voyage。推测原因是 document 视觉结构（layout、table、figure）需要更专门的 visual-document encoder，Gemini Embedding 2 的 generalist 设计在这点上没特别优化。

### MMTEB (Table 2)

| Metric | Gemini Embedding 2 | Amazon Nova MME | Gemini Embedding (text) | Voyage-3.5/Code-3 |
|--------|---------------------|-----------------|--------------------------|-------------------|
| MTEB(Multilingual) Mean | **69.9** | 63.8 | 68.4 | 58.5 |
| MTEB Code | **84.0** | - | 76.0 | - |
| CoIR | **82.3** | - | 73.9 | 78.5 |

Multimodal 模型在 pure text 上超越 text-only 模型（69.9 vs 68.4），这违反直觉。一般担心 multimodal 训练稀释 text 能力，但这里 multimodal 训练反而**增强了** text representation。推测原因是 multimodal data 提供了更多 "grounded" 信号（text 配 image、video），让 text embedding 有 visual anchor，representation 更 robust。

MTEB Code 84.0 把 Voyage-Code-3（code-specific 模型）甩开，这背后是 Gemini 本身的 code pre-training + synthetic data augmentation 的功劳。

### MSEB Audio Retrieval (Table 3)

| Setup | Average mrr@10 | In-Lang | Cross-Lang |
|-------|----------------|---------|------------|
| w/ ASR | 70.40 | 73.58 | 67.55 |
| w/ Native Audio | **73.99** | **75.58** | **72.56** |

**这是 paper 最强的 evidence 之一**。

**Cascade baseline (ASR → Embedding) 的根本问题**：error propagation。ASR 一旦错一个字（"recognize speech" vs "wreck a nice beach"），下游 retrieval 就拿到完全错误的 query。Native audio 处理保留 acoustic signal 的 ambiguity（prosody、intonation、emphasis），让 embedding 在 latent space 中保留所有可能 interpretation，retrieval 阶段用 corpus 的 prior 帮助 disambiguate。

**Cross-lang 增益最大（+5.01）的原因更深**：cross-lingual retrieval 要求模型理解 query 和 document 是不同语言的同义内容。ASR pipeline 把 audio 强制压成 source language text，丢失了 "这是 X 语言的 Y 概念" 这种 meta 信息。Native audio embedding 直接进入 modality-agnostic 的 latent space，cross-lingual alignment 在 latent space 自然完成，不受 intermediate text 语言的 phonetic 边界限制。

这背后对应的是 phonetic universality 假设：人类对不同语言的相同概念在脑中有共享的 semantic representation，无关输入 modality。Gemini Embedding 2 的 native audio 重现了这个现象。

### Specialized Domains (Table 4)

| Model | MicroVQA | ArtCap | AstroLLaVA | Recipe1M-Ingr | Recipe1M-Inst |
|-------|----------|--------|------------|---------------|----------------|
| CLIP Large P14 | 44.4 | 49.4 | 28.8 | 76.5 | 74.6 |
| ALIGN Base | 48.1 | 49.2 | 18.4 | 70.3 | 70.8 |
| SigLIP 2 Giant | 33.3 | 8.4 | 13.2 | 81.2 | 80.4 |
| TIPS Giant P14 | 20.0 | **65.2** | 10.1 | 66.0 | 65.6 |
| Voyage-3.5-MM | 53.3 | 48.7 | 30.3 | - | - |
| **Gemini Embedding 2** | **79.3** | 67.7 | **64.4** | **90.2** | **92.1** |

**这是 paper 最重要的 zero-shot generalization evidence**。

Baseline 模型在 specialized domain 上波动巨大。TIPS 在 ArtCap 上 65.2 但 MicroVQA 只有 20.0；SigLIP 2 Giant 在 Recipe1M 上 81.2 但 AstroLLaVA 只有 13.2。这是典型的 **specialist curse**：在某个 domain 训练充分的模型会 overfit 到该 domain 的 visual distribution。

Gemini Embedding 2 在所有 4 个 domain 上都是 SOTA 或接近 SOTA。Generalization 来源推测是 Gemini 2.5 pre-training 时见过的数据 distribution 极其广泛（从天文图像到显微镜图像到艺术品），fine-tune embedding 时这些 distribution 的 representation 已经在 transformer parameters 里"睡着"，只需要 contrastive loss "唤醒"。

### Synthetic Data Ablation (Table 5)

| Model | Average | CodeFeedbackMT | CodeFeedbackST | SyntheticText2SQL |
|-------|---------|----------------|----------------|---------------------|
| Gemini Embedding | 70.5 | 56.3 | 85.3 | 70.0 |
| Gemini Emb 2 w/o Synthetic | 73.0 | 57.9 | 85.5 | 75.7 |
| Gemini Emb 2 w/ Synthetic | **86.3 (+15.8)** | **92.3** | 88.6 | 78.1 |

**CodeFeedbackMT 从 57.9 跳到 92.3（+34.4 个点）**，单纯靠 synthetic data augmentation。

Code retrieval 的核心难度是 query 和 document 的 surface form 差异极大。用户用自然语言描述问题，document 是代码。Gemini 生成 synthetic (query, code) pairs 让训练数据从 "自然存在的 sparse pairs" 变成 "dense + high quality pairs"。这是 LLM-as-data-generator 范式在 embedding 训练上的体现。

这个 finding 联系到 Gecko [15] 和前代 Gemini Embedding [18]：Google 的 embedding 路线一直把 synthetic data 当 first-class citizen，从 Gecko 的两-step distillation pipeline 到现在 Gemini 生成 synthetic data。本质是用 LLM 的 world knowledge "填充" 数据稀疏的 task space。

---

## 相关工作的位置

| 路线 | 代表 | 核心方法 | 与 Gemini Emb 2 的差异 |
|------|------|----------|------------------------|
| Dual-tower CLIP-style | CLIP [1], ALIGN [2], SigLIP 2 [3] | Paired (image, text) contrastive | Late fusion，无 interleaved input 支持 |
| Caption-augmented | CoCa [4] | Contrastive + captioning | 多任务但仍 dual encoder |
| LLM-as-embedder | NV-Embed [16], Gecko [15] | Decoder-only LLM + contrastive | 主要 text-only，multimodal 受限 |
| Modality-aware pre-training | MoCa [21] | Joint reconstruction | 改造 causal backbone，未做 audio/video |
| Modality-aware hard negative | MM-Embed [22] | Hard negative mining | 解决 modality bias 但仍偏 text |
| Native MLLM embedder | SAIL-Embedding [19], Amazon Nova MME [20] | MLLM backbone + contrastive | 同路线，scale 和 modality 覆盖有差异 |

Gemini Embedding 2 是第一个同时在 (1) native audio, (2) native video, (3) interleaved multimodal input, (4) MRL multi-resolution, (5) Model Soup, (6) synthetic data augmentation 6 个维度上都做到 SOTA 的模型。Paper §2 末尾说"they predominantly address these axes in isolation"，这是准确的 self-positioning。

---

## Paper 的盲点和 weakness

1. **Backbone scale 不公平**。Gemini 2.5 参数量在 hundreds of billions 量级，SOTA 性能很大部分来自 backbone scale 而非 embedding methodology。其他 team 用同等 scale MLLM 当 backbone 能否复现？Open question。

2. **Tokenization 细节不透明**。没给 video 的 32 frame sampling strategy、audio frame rate、image patch size。对 reproducibility 关键。

3. **ViDoRe V2 不是 SOTA**。Document retrieval 64.9 vs Voyage 65.5。对企业级 RAG 用户，document retrieval 是核心场景。

4. **Long video 处理**。限制 video 在 1 FPS up to 32 frames，即 32 秒。对 longer video（电影、lecture）不够。Paper 没讨论如何处理。

5. **Audio 实验范围窄**。MSEB 只评估 retrieval split，没 ASR、speaker identification、audio event classification 的全面评估。

6. **Model Soup 权重选择**。只展示 2:1 和 1:1 两个比例，没 systematize 权重 search 方法。

---

## 给你的核心 takeaways

1. **Embedding model 进入 "MLLM-as-backbone" 时代**。Text embedding 经历 BERT → GPT → LLM-as-embedder，multimodal embedding 经历 CLIP → CoCa → MLLM-as-embedder。Gemini Embedding 2 是这个浪潮的旗舰。

2. **Native multimodal = modality 在 transformer 每层交互**。不是只在最后 cosine 时才交互。Interleaved input 只有 native multimodal 才能处理。

3. **PFT + FT + Model Soup 是新一代 embedding 训练 standard recipe**。PFT 解决范式转换（autoregressive → encoder），FT 解决 task-specific 能力，Model Soup 解决 generalization。

4. **Synthetic data 是 embedding 性能的 hidden multiplier**。LLM 生成 synthetic data 把稀疏 task space 填充为稠密，用 LLM 的 world knowledge 蒸馏到 embedding space。

5. **Audio native 处理对 cross-lingual retrieval 有质变提升**。未来 multimodal RAG 的方向：不要 ASR 中间层，直接 audio-to-document retrieval。

6. **MRL 让 embedding 兼顾 speed 和 quality**。768 维 retrieval，3072 维 re-ranking，一个模型多场景复用。

7. **Generalist 的胜利**。Specialized domain（显微镜、天文、烹饪）的 zero-shot 性能显示，训练充分的 generalist embedding 可以 outperform domain-specific 模型。Embedding model 市场可能 consolidation。

8. **Backbone scale 决定上限**。Embedding methodology 的 innovation 边际效应递减，backbone scale 是主导因素。

---

## 实操建议

1. **直接调用 API 而非自训**。复现需要 Gemini-scale backbone + 大量 multimodal paired data + multi-stage training infra，成本极高。

2. **MRL 优先用 768 维**。除非对 accuracy 要求极高，768 维够用，存储和 retrieval 速度快 4 倍。

3. **Audio input 直接走 native，不要 ASR**。MSEB 已经证明 native audio 在 cross-lingual 上有质变优势。RAG 场景有 audio query 不要做 ASR 中间层。

4. **Video 限制在 32 秒内**。超过 32 秒需要 chunk 后多 embedding 或 hierarchical aggregation。

5. **Domain-specific 先试 zero-shot**。Table 4 显示 Gemini Embedding 2 在显微镜、天文、烹饪等专业领域 zero-shot 都 SOTA。先跑 zero-shot baseline 再决定要不要 fine-tune。

6. **Model Soup 1:1 是 safe default**。Table 6 显示 1:1 在 most task 上 balance 最好。

---

## 开放问题

1. **End-to-end RAG training**：embedding + retrieval + LLM generation 联合训练，让 embedding 为 downstream generation 优化而非 retrieval metric 优化。

2. **Agentic retrieval**：agent 多步 retrieval 中，每步 query 是上一步 LLM 输出 + context 的组合。Embedding 如何在 iterative query distribution 上保持 stable？

3. **Ranking signal integration**：Paper §6 提到 "including other signals from search systems like ranking can be hugely beneficial"。Click-through data、dwell time、user feedback 作为 weak supervision，让 embedding 对齐用户真实偏好。

4. **Long video / long document**：32 frame / 8K token 限制需要突破。Hierarchical embedding（chunk-level + document-level）、sparse attention、retrieval-augmented embedding。

5. **Streaming audio / video**：实时会议检索、live video search 需要 incremental embedding update 机制。

6. **Evaluation framework for interleaved multimodal retrieval**：Paper §6 末尾说"we invite the broader academic community to contribute novel evaluation frameworks"。当前 benchmark 都是 single-modality input，没有 benchmark 评估 "image+text → video+audio" 这种 complex interleaved retrieval。

希望这个版本更直白，build 起你对 Gemini Embedding 2 的完整直觉。如果想 drill down 某个具体方面（比如 audio tokenizer 实现、Model Soup weight search、synthetic data generation prompt 设计），可以继续展开。

---

# Gemini Embedding 2: Native Multimodal Embedding Model 深度技术解读

Karpathy 你好，这篇 paper 来自 Google 的 Gemini Embedding Team，发表于 2025 年（arXiv 应在 2503.07891 之后的版本），是此前 text-only Gemini Embedding 的升级版，核心目标是构建一个真正 native multimodal 的统一 embedding 模型。我会从架构、损失函数、训练 recipe、实验数据多个层次做深度拆解，帮你 build 起完整的 intuition。

参考链接：
- Paper arXiv (前代): https://arxiv.org/abs/2503.07891
- MMTEB benchmark: https://arxiv.org/abs/2502.13595
- Gecko (前身工作): https://arxiv.org/abs/2403.20327
- Matryoshka Representation Learning: https://arxiv.org/abs/2205.14547
- Model Soups: https://arxiv.org/abs/2203.05482
- SWA (Stochastic Weight Averaging): https://arxiv.org/abs/1803.05407
- CoIR: https://arxiv.org/abs/2407.02883
- ViDoRe V2: https://arxiv.org/abs/2505.17166
- MSEB: https://arxiv.org/abs/2602.07143
- SigLIP 2: https://arxiv.org/abs/2502.14753

---

## 1. 核心动机与定位

现有 multimodal embedding 模型如 CLIP、ALIGN、SigLIP 2、CoCa 都采用 late-fusion 双塔架构，每个模态独立 encoder，通过 paired cross-modal data 拉到统一空间。这种范式有两个根本缺陷：

**缺陷 A：modality 间交互浅层化**。双塔只在最后 cosine 之前进行交互，模态间深层语义交互在 transformer 内部根本没机会发生。

**缺陷 B：mixed-modality input 无法处理**。比如 "这段视频中第 5 秒画面里的红色汽车"，文字和视频需要 token 级 interleaving 才能精准定位，双塔架构根本不支持这种 query 形态。

Gemini Embedding 2 的解法是直接拿 Gemini 2.5 这个 native MLLM 当 backbone，让它原本就具备的 deep cross-modal fusion 能力成为 embedding 的天然基础。这等价于把 Gemini 当作 embedding 模型的 "pre-trained backbone"，把后续的 fine-tune 视作传统意义上的 "pre-training + fine-tuning" 范式。

---

## 2. 架构深度解析

### 2.1 整体 pipeline

整个 pipeline 可以形式化为三个 stage：

```
Input (raw modality) 
  → Tokenization (Gemini's native tokenizer) 
  → Transformer M (bidirectional attention, initialized from Gemini)
  → Pooler P (mean pooling)
  → Linear projection f (randomly initialized)
  → Embedding E ∈ R^d
```

形式化表达：

$$\mathbf{T}_{\text{embed}} = \mathcal{M}(\mathbf{T}) \in \mathbb{R}^{L \times d_M}$$

$$\mathbf{P}_{\text{embed}} = \mathcal{P}(\mathbf{T}_{\text{embed}}) \in \mathbb{R}^{d_M}$$

$$\mathbf{E} = f(\mathbf{P}_{\text{embed}}) \in \mathbb{R}^{d}$$

变量含义：
- $\mathbf{T}$：input token sequence，长度 $L$
- $\mathcal{M}$：transformer backbone（bidirectional attention，initialized from Gemini 2.5）
- $d_M$：transformer hidden dimension
- $\mathcal{P}$：pooler，本文采用 mean pooling（沿 sequence axis 平均）
- $f$：randomly initialized linear projection
- $d$：output embedding dimension = 3,072

### 2.2 几个关键的架构选择解读

**为什么用 bidirectional attention 而非 causal attention？**

Embedding 任务需要 context-aware 的 dense representation。Causal attention 的每个 token 只能看到前面的 token，最后 token 的 hidden state 信息密度严重不均，把"全部信息"压到一个 token 上是低效的。Bidirectional 让所有 token 都能 attend 到所有 token，每个 token 都能积累全局上下文，再做 mean pooling 时信息分布均匀。这个 motivation 来自 [23] 的实证发现。

相关工作 MoCa [21] 提出了在 causal backbone 上加 joint reconstruction objective 来"伪 bidirectional"，MM-Embed [22] 则通过 modality-aware hard negative mining 解决 modality bias。Gemini Embedding 2 直接用 bidirectional，绕过了这些 trick 的需求。

**为什么用 mean pooling 而非 attention pooling 或 CLS token？**

[23] 的研究表明，简单的 mean pooling 在 decoder-based LM 适配 encoder task 时已经足够有效。Mean pooling 的另一个优势是它对 token sequence 长度变化 robust，适合 video、长文档这种长度变化大的输入。CLS token 容易受 sequence length 影响，attention pooling 又增加额外可训练参数和复杂度。

**Random linear projection $f$ 的作用？**

这里 $f$ 是 **randomly initialized**，作用是 dimension adapter。$d_M$ 是 Gemini 的 hidden size，$d=3072$ 是 embedding 输出维度，两者不匹配。注意 $f$ 是 from-scratch 训练的，因为它要学习如何把 transformer 的 representation space 投影到对比学习期望的 cosine-similarity-friendly space。同时 MRL [25] 需要 $f$ 输出的不同 prefix 维度都能独立可用，所以 $f$ 必须是一个 dense 矩阵，前 768、1536、3072 维分别 optimized。

### 2.3 Modality-specific tokenization

这是这篇 paper 最有意思的细节之一。Paper 提到"different modalities require different steps to convert the raw format into a sequence of tokens"，但没有详细给出每种 modality 的具体 tokenizer 实现。从 inference 逻辑可以推断：

- **Text**：标准 Gemini tokenizer（SentencePiece-based）
- **Image**：Gemini 的 native vision encoder（可能是 ViT 变体 + patch embedding），输出 visual tokens
- **Video**：1 FPS 采样，up to 32 frames（实验部分提到），每帧当 image 处理后，加上 temporal position encoding
- **Audio**：Gemini 的 native audio tokenizer（应该是 Enformer 或其升级版，参考 AudioLM / USM 路线），输出 audio tokens

所有 modality 的 token 都进入同一个 transformer $\mathcal{M}$ 的 embedding space，这是 native multimodal 的本质。对比 CLIP 的 dual-encoder，CLIP 是 image encoder 和 text encoder 各算各的，最后才 contrastive，而 Gemini Embedding 2 在 transformer 内部就已经把 image token、text token、video token、audio token 放在一起 attention，模态交互发生在每一层。

---

## 3. 训练目标函数深度拆解

### 3.1 主损失函数

Paper 使用 NCE loss with in-batch negatives [24]，每个训练 example 包含 query $q_i$、positive target $p_i^+$，可选 hard negative $p_i^-$。text-only task 还有 task string $t$（如 "question answering"、"fact checking"）。

Embedding 计算公式：

$$\mathbf{q}_i = f(\text{mean\_pool}(\mathcal{M}(t \oplus q_i)))$$
$$\mathbf{p}_i^{\pm} = f(\text{mean\_pool}(\mathcal{M}(p_i^{\pm})))$$

变量含义：
- $t$：task string（instruction prefix）
- $\oplus$：concatenation
- $q_i$：query 原始内容
- $p_i^+$：positive target
- $p_i^-$：hard negative target
- mean_pool：沿 sequence 维度做平均
- $f$：linear projection 到 $\mathbb{R}^d$

**关键细节：task string dropout**。训练时随机 drop 掉 task string $t$，让模型对 "有 instruction" 和 "无 instruction" 两种输入都 robust。这等价于一种 input-level regularization，让 embedding space 同时支持 instruction-tuned query 和 raw query。这在 inference 时尤其有用：用户可能用 task prefix 也可能不用，模型都能产出高质量 embedding。

### 3.2 主 Loss 公式 (公式 2)

$$\mathcal{L} = \frac{1}{B} \sum_{i=1}^{B} \left[ -\log \frac{e^{\text{sim}(\mathbf{q}_i, \mathbf{p}_i^+) / \tau}}{e^{\text{sim}(\mathbf{q}_i, \mathbf{p}_i^+) / \tau} + e^{\text{sim}(\mathbf{q}_i, \mathbf{p}_i^-) / \tau} + \sum_{j=1}^{B} \text{mask}(i, j) e^{\text{sim}(\mathbf{q}_i, \mathbf{p}_j^+) / \tau}} \right]$$

变量含义：
- $B$：batch size
- $\tau$：temperature parameter（学习率敏感，越大分布越平滑）
- $\text{sim}(\mathbf{x}, \mathbf{y}) = \mathbf{x}^\top \mathbf{y} / \|\mathbf{x}\| \|\mathbf{y}\|$：cosine similarity
- $\text{mask}(i, j)$：避免 in-batch 中相同 query 或相同 positive 被错误当作 negative

mask 定义（公式 3）：

$$\text{mask}(i, j) = \begin{cases} 0 & \text{if } q_i = q_j \text{ or } p_i^+ = p_j^+ \\ 1 & \text{otherwise} \end{cases}$$

**这个 mask 的必要性**：在 classification task 中，label 数量小，同一个 batch 里很可能多个 example 的 positive target 是同一个 label embedding（比如分类任务里 "sports" 标签的 embedding）。如果不 mask，这些相同 label 会被当作 hard negatives，loss 会被严重错误信号污染。这个 trick 在分类任务的 in-batch negative contrastive 训练里几乎是必须的。

**Hard negative 项处理**：如果没提供 hard negative，分母第二项 $e^{\text{sim}(\mathbf{q}_i, \mathbf{p}_i^-) / \tau}$ 省略。这是 conditional structure，让 loss 同时支持 triplet loss 模式（有 hard neg）和纯 in-batch negative 模式（无 hard neg）。

### 3.3 Matryoshka Representation Learning (MRL)

Paper 用 MRL [25] 让模型同时支持多个 embedding 维度。Gemini Embedding 2 输出 $d=3072$ 维，但同时 optimized 768 和 1536 维的 sub-dimensions。

**Intuition**：MRL 的核心思想是让 embedding 的前 k 维就能形成一个有效的 representation，前 k+1 维是前 k 维的 refinement。这就像俄罗斯套娃（Matryoshka doll），外层套娃是内层套娃的精化。

**实现细节**：训练时把 loss 拆成多个 loss，分别在 embedding 的不同 prefix 上计算：

$$\mathcal{L}_{\text{total}} = \sum_{k \in \{768, 1536, 3072\}} \mathcal{L}_k(\mathbf{E}_{[:k]})$$

其中 $\mathbf{E}_{[:k]}$ 表示 embedding 的前 $k$ 维。

**实际价值**：在 RAG、ANN search、document retrieval 场景，可以用 768 维的 embedding 做 IVF/PQ indexing，加速 retrieval；用 3072 维做 re-ranking。同一个模型、同一份 embedding 多次复用，这是工业落地时降低存储和计算成本的关键。

---

## 4. 训练 Recipe 深度拆解

这个 paper 最 valuable 的部分是训练 recipe，它揭示了如何让一个 native multimodal backbone "变成" embedding model。整体分三个阶段：

### 4.1 Pre-Fine-Tuning (PFT)

**目的**：把 auto-regressive generation 模型适配为 encoder。这是一个范式转换：从"预测下一个 token"到"产出有判别力的 dense vector"。

**数据策略**：
- 使用大量可能 noisy 的 query-target pairs
- multi-task setup
- 仅 image、text、code tasks（这一阶段**不包含 audio/video**）
- 单 task 构造 batch（同一 batch 内所有 example 来自同一个 task）

**Batch size 选择**：大 batch size 提供更稳定的 gradients，缓解 noisy input 的冲击。这符合 contrastive learning 的一般规律：负样本越多，contrastive signal 越强，但 noisy positive 也会让 training 不稳定，大 batch 平均化这种噪声。

**为什么 PFT 阶段不加 audio/video**？我推测有两个原因：
1. PFT 是范式转换，需要数据相对干净。Text/image/code 是相对结构化的 paired data，audio/video 的 paired data 更稀少且 noisy
2. Audio/video 的 native 处理需要模型先具备 bidirectional fusion 的能力，PFT 阶段模型还没"学会"encoder 范式，加上 audio/video 容易 training collapse

### 4.2 Fine-Tuning (FT)

**目的**：在已经掌握 encoder 范式的基础上，加入所有 modality 和所有 task 类型，构建真正的 unified embedding space。

**数据策略**：
- text、code、document、image、audio、video 全 modality tasks
- 大部分 task 包含 query、target、hard negative triplets
- 针对不同 task 调优 batch size（这个细节很重要，不同 task 的 sample 复杂度不同）
- 单 task 构造 batch
- 包含 single-modality batches 和 cross-modality batches

**关键设计**：single-modality batch + cross-modality batch 交替训练。这相当于一种 curriculum learning：
- Single-modality batch：让 model 在每个 modality 内部建立 robust 的 representation（intra-modality alignment）
- Cross-modality batch：让 model 在 modality 间对齐 representation（inter-modality alignment）

两者结合才能避免 "modality collapse"（model 只学 text-text 而忽略 cross-modal）或 "modality bias"（参考 MM-Embed [22] 的讨论）。

**Sampling rate 是 empirical 调出来的**：paper 明确说"the sampling rates of each of the different tasks are defined empirically"且"balancing overall performance across all modalities was sensitive to hyper-parameters like sampling rates and batch sizes"。这是工业级训练的常见现象：理论分析给不出最优配比，只能 grid search。Google 这里靠 scale 和算力做了大量 ablation。

### 4.3 Model Soup

**目的**：systematize 不同 checkpoint 的组合，提升 generalization。

**三种 soup 策略**：

1. **同 training run 的不同 checkpoints 平均**（SWA 风格 [26]）：原理是 SGD 后期会在 flat minima 附近游走，平均权重等价于找一个更宽的 optima，泛化更好
2. **不同 training runs 的 checkpoints 平均**（Model Soups 风格 [27]）：不同 run 用不同 seed 或不同 task 比例，soup 后平均掉 task-specific 的偏好
3. **Weighted average**：用不同权重组合，paper 在 ablation Table 6 里展示了 2:1 和 1:1 的 base : fine-tuned 比例

Table 6 的结果很有启发：
- Baseline (Gemini Embedding 2): MSR-VTT=68.2, YouCook2=55.9, Vatex=69.2
- + MSR-VTT data (FT_mix-m): MSR-VTT=75.0 (+6.8), Vatex=71.7 (+2.5), YouCook2=56.1 (+0.2)
- + MSR-VTT & Vatex data (FT_mix-mv): MSR-VTT=76.1 (+7.9), Vatex=79.5 (+10.3), YouCook2=55.3 (-0.6)
- Soup 2:1: MSR-VTT=71.7 (+3.5), YouCook2=56.1, Vatex=74.5 (+5.3)
- Soup 1:1: MSR-VTT=73.7 (+5.5), YouCook2=56.8 (+0.9), Vatex=76.8 (+7.6)

**Intuition**：加 in-domain data 提升目标 task 但伤害 out-of-domain task，因为 fine-tune 让 model "偏向" 目标 task 的 distribution。Model soup 通过权重平均，让原 base model 的 robustness 和 fine-tuned model 的 task-specific 能力共存。这个现象的本质是 loss landscape 在 base 和 fine-tuned 之间存在一条相对 flat 的路径，线性插值的权重在两个 basin 之间保持稳定。

---

## 5. 实验结果全表分析

### 5.1 Multimodal Retrieval (Table 1)

| Benchmark | Gemini Embedding 2 | Amazon Nova MME | Voyage-3.5-MM | Legacy Google |
|-----------|---------------------|-----------------|---------------|---------------|
| GUIEC (Img→Img) | **79.4** | 68.6 | 69.4 | 69.5 |
| MSCOCO (T→I, R@1) | **62.9** | 57.2 | 58.1 | 53.1 |
| Flickr30k (T→I) | 89.1 | 81.6 | **89.9** | 81.4 |
| DOCCI (T→I) | **93.4** | 84.0 | 83.8 | 74.0 |
| TextCaps (T→I) | **89.6** | 76.0 | 79.4 | 74.0 |
| MSCOCO (I→T) | **78.8** | 68.3 | 74.5 | 68.2 |
| Flickr30k (I→T) | **97.4** | 87.5 | 94.5 | 94.0 |
| Vatex (T→V) | **68.8** | 60.3 | 55.2 | 54.9 |
| MSR-VTT (T→V) | **68.0** | 67.0 | 63.0 | 57.9 |
| YouCook2 (T→V) | **52.5** | 34.7 | 31.4 | 34.9 |
| EncyclopedicVQA (I+T→T, R@20) | **71.5** | - | 58.6 | - |
| ViDoRe V2 | 64.9 | 60.6 | **65.5** | 28.9 |
| **Overall** | **77.2** | 68.2 | 70.0 | 64.1 |
| Modality | **V/A/I/T** | V/A/I/T | V/I/T | I/T |

**关键观察**：

1. **DOCCI 和 TextCaps 这种 long-caption benchmark 优势显著**（93.4 vs 84.0 vs 83.8）。这印证了 deep fusion 的价值——long caption 需要模型理解图像的 spatial reasoning、scene text、long-tail concepts，late-fusion 模型做不到这个 level。

2. **EncyclopedicVQA 71.5 vs 58.6（+12.9）**：这个 task 是 image+text 联合 query 检索 answer，最能体现 native multimodal 的价值。Interleaved input "image + 问题" 直接 embed 成一个 vector 检索答案，late-fusion 模型必须 concat 两个 vector 再处理，性能差是必然的。

3. **YouCook2 52.5 vs 34.7**：YouCook2 是 cooking instruction video，需要 fine-grained temporal understanding（"在视频第几秒切洋葱"）。Gemini Embedding 2 在 video 上的 native 处理（1 FPS, up to 32 frames + temporal position encoding）让它在 temporal reasoning 上优势明显。

4. **ViDoRe V2 64.9 vs Voyage 65.5**：document retrieval 上 Gemini 略输 Voyage。我推测原因是 document 视觉结构（layout、table、figure）的 OCR-like 处理需要更专门的 visual-document encoder（参考 paper §2 提到的 "tiled mixtures of vision encoders"），Gemini Embedding 2 的 generalist 设计在这点上没特别优化。

### 5.2 MMTEB (Table 2)

| Metric | Gemini Embedding 2 | Amazon Nova MME | Gemini Embedding (text) | Voyage-3.5/Voyage-Code-3 |
|--------|---------------------|-----------------|--------------------------|---------------------------|
| MTEB(Multilingual) Mean(Task) | **69.9** | 63.8 | 68.4 | 58.5 |
| MTEB(Multilingual) Mean(Type) | **61.2** | - | 59.6 | 51.9 |
| Bitext Mining | 85.4 | - | 79.3 | 60.5 |
| Classification | 73.1 | 54.6 | - | 58.5 |
| Retrieval | 70.0 | - | 67.7 | 64.0 |
| STS | 79.4 | - | 79.4 | 70.0 |
| MTEB Code | **84.0** | - | 76.0 | - |
| CoIR | **82.3** | - | 73.9 | 78.5 |
| Modality | V/A/I/T | V/A/I/T | T | T |

**Key insight**：
- Multimodal 模型在 pure text 上超越 text-only 模型（69.9 vs 68.4），这违反直觉。一般会担心 multimodal 训练稀释 text 能力，但这里 multimodal 训练反而**增强了** text representation。我推测原因是 multimodal data 提供了更多 "grounded" 信号（text 配 image、video），让 text embedding 不再漂浮在纯符号空间，而是有 visual anchor，representation 更鲁棒
- MTEB Code 84.0 vs Voyage-Code-3 的差距大（Voyage-Code-3 是 code-specific 模型）：Gemini 本身的 code pre-training 能力 + synthetic data augmentation（详见 §5.2 ablation）造就了这个优势

### 5.3 MSEB (Audio Retrieval, Table 3)

| Model Setup | Average mrr@10 | Passage In-Lang | Passage Cross-Lang |
|-------------|----------------|-----------------|---------------------|
| Gemini Embedding 2 w/ ASR | 70.40 | 73.58 | 67.55 |
| Gemini Embedding 2 w/ Native Audio | **73.99** | **75.58** | **72.56** |

**这个 ablation 是 paper 最强的 evidence 之一**。

**Cascade baseline (ASR → Embedding) 的根本问题**：error propagation。ASR 一旦错一个字（"recognize speech" vs "wreck a nice beach"），下游 retrieval 就拿到完全错误的 query。Native audio 处理保留了 acoustic signal 的 ambiguity（prosody、intonation、emphasis），让 embedding 在 latent space 中保留所有可能 interpretation，retrieval 阶段可以用 corpus 的 prior 帮助 disambiguate。

**Cross-lang 增益最大（+5.01）的原因更深**：cross-lingual retrieval要求模型理解 query 和 document 是不同语言的同义内容。ASR pipeline 把 audio 强制压成 source language text，丢失了 "这是 X 语言的 Y 概念" 这种 meta 信息。Native audio embedding 直接进入 modality-agnostic 的 latent space，cross-lingual alignment 在 latent space 自然完成，不受 intermediate text 语言的 phonetic 边界限制。

这背后对应的是 phonetic universality 假设：人类对不同语言的相同概念在脑中有共享的 semantic representation，无关输入 modality。Gemini Embedding 2 的 native audio 部分重现了这个 phenomenon。

### 5.4 Specialized Domains (Table 4)

| Model | MicroVQA | ArtCap | AstroLLaVA | Recipe1M-Ingr | Recipe1M-Inst |
|-------|----------|--------|------------|---------------|----------------|
| CLIP Base P32 | 34.1 | 34.1 | 21.2 | 64.6 | 61.1 |
| CLIP Large P14 | 44.4 | 49.4 | 28.8 | 76.5 | 74.6 |
| ALIGN Base | 48.1 | 49.2 | 18.4 | 70.3 | 70.8 |
| SigLIP 2 Giant | 33.3 | 8.4 | 13.2 | 81.2 | 80.4 |
| TIPS Giant P14 | 20.0 | **65.2** | 10.1 | 66.0 | 65.6 |
| Voyage-3.5-MM | 53.3 | 48.7 | 30.3 | - | - |
| **Gemini Embedding 2** | **79.3** | 67.7 | **64.4** | **90.2** | **92.1** |

**这是 paper 最重要的 zero-shot generalization evidence**。

**关键观察**：baseline 模型在 specialized domain 上波动巨大。TIPS 在 ArtCap 上 65.2 但在 MicroVQA 上只有 20.0；SigLIP 2 Giant 在 Recipe1M 上 81.2 但在 AstroLLaVA 上只有 13.2。这是典型的 **specialist curse**：在某个 domain 训练充分的模型会 overfit 到该 domain 的 visual distribution。

**Gemini Embedding 2 在所有 4 个 domain 上都是 SOTA 或接近 SOTA**：
- MicroVQA（显微镜生物图像）：79.3，超过 CLIP Large 几乎 35 个点
- AstroLLaVA（天文图像）：64.4，超过 CLIP Large 35 个点
- Recipe1M（烹饪食材/步骤）：90.2 / 92.1，超过 SigLIP 2 Giant 约 10 个点
- ArtCap 仅次 TIPS Giant（67.7 vs 65.2，Gemini 更高）

**这个 generalization 的来源是什么？** 我推测是 Gemini 2.5 pre-training 时见过的数据 distribution 极其广泛（从天文图像到显微镜图像到艺术品），fine-tune embedding 时这些 distribution 的 representation 已经在 transformer parameters 里"睡着"，只需要 contrastive loss "唤醒"。这就是为什么 paper 强调"initalizing Gemini Embedding 2 from Gemini can be understood as the 'pre-training' stage of the embedding model"。

### 5.5 Synthetic Data Ablation (Table 5)

| Model | Average | CodeFeedbackMT | CodeFeedbackST | SyntheticText2SQL |
|-------|---------|----------------|----------------|---------------------|
| Gemini Embedding | 70.5 | 56.3 | 85.3 | 70.0 |
| Gemini Embedding 2 w/o Synthetic | 73.0 | 57.9 | 85.5 | 75.7 |
| Gemini Embedding 2 w/ Synthetic | **86.3 (+15.8)** | **92.3** | 88.6 | 78.1 |

**这是 paper 中最 dramatic 的 ablation 结果**。CodeFeedbackMT 从 57.9 跳到 92.3（+34.4 个点！），单纯靠 synthetic data augmentation。

**Intuition**：Code retrieval 的核心难度是 query 和 document 的 surface form 差异极大。用户用自然语言描述问题，document 是代码。Gemini 生成 synthetic (query, code) pairs 的能力让训练数据从"自然存在的 sparse pairs"变成"dense + high quality pairs"。这就是 LLM-as-data-generator 范式在 embedding 训练上的体现。

**这个 finding 联系到 Gecko [15] 和前代 Gemini Embedding [18]**：Google 的 embedding 路线一直把 synthetic data 当 first-class citizen，从 Gecko 的两-step distillation pipeline 到现在的 Gemini 生成 synthetic data。本质上是用 LLM 的 world knowledge "填充" 数据稀疏的 task space。

---

## 6. 与相关工作的 differentiation matrix

| 路线 | 代表工作 | 核心方法 | 与 Gemini Embedding 2 的差异 |
|------|---------|----------|------------------------------|
| Dual-tower CLIP-style | CLIP [1], ALIGN [2], SigLIP 2 [3] | Paired (image, text) contrastive | Late fusion，无 interleaved input 支持 |
| Caption-augmented | CoCa [4] | Contrastive + captioning | 多任务但仍是 dual encoder |
| LLM-as-embedder | NV-Embed [16], Gecko [15] | Decoder-only LLM + contrastive | 主要 text-only，multimodal 受限 |
| Modality-aware continual pre-training | MoCa [21] | Joint reconstruction objective | 改造 causal backbone，未做 audio/video |
| Modality-aware hard negative | MM-Embed [22] | Hard negative mining | 解决 modality bias 但仍偏 text |
| Native multimodal LLM embedder | SAIL-Embedding [19], Amazon Nova MME [20] | MLLM backbone + contrastive | 与 Gemini Embedding 2 同路线，但 scale 和 modality 覆盖有差异 |

**Gemini Embedding 2 的独特位置**：它是第一个同时在 (1) native audio, (2) native video, (3) interleaved multimodal input, (4) MRL multi-resolution, (5) Model Soup, (6) synthetic data augmentation 6 个维度上都做到 SOTA 的模型。Paper §2 末尾说"they predominantly address these axes in isolation"，这是准确的 self-positioning。

---

## 7. Paper 的隐藏 assumptions 和 potential weaknesses

我帮你 build intuition 也包括识别 paper 的盲点：

**Assumption 1: Gemini 2.5 是公平的 starting point**。这个 paper 的 baseline 是 "Gemini 2.5 当 backbone"，但 Gemini 2.5 本身是巨大的 multimodal LLM，参数量在 hundreds of billions 量级。这意味着 Gemini Embedding 2 的 SOTA 性能很大一部分来自 backbone 的 scale，而非 embedding methodology 本身。如果其他 team 用同等 scale 的 MLLM 当 backbone，能不能复现这个性能？这是个 open question。

**Assumption 2: Tokenization 细节不透明**。Paper 没有详细给出 video 的 32 frame sampling strategy、audio 的 frame rate、image 的 patch size。这些工程细节对 reproducibility 关键。

**Weakness 1: ViDoRe V2 不是 SOTA**。Document retrieval 64.9 vs Voyage 65.5。Paper 解释这是 trade-off（"given the breadth of tasks it is simultaneously optimized for"），但对企业级 RAG 用户来说，document retrieval 是核心场景。

**Weakness 2: Long video 处理**。Paper 限制 video 在 1 FPS up to 32 frames，即 32 秒视频。对于 longer video（电影、lecture），这个上限不够。Paper 没讨论如何处理 long video。

**Weakness 3: Audio 部分的实验范围窄**。MSEB 只评估 retrieval split，没有 ASR、speaker identification、audio event classification 等 audio-specific task 的全面评估。

**Weakness 4: Model Soup 的权重选择**。Paper 在 ablation 里只展示了 2:1 和 1:1 两个比例，没有 systematize 权重 search 的方法。实际部署时怎么选最优权重？

---

## 8. Intuition 总结（给你的核心 takeaways）

构建这个模型 intuition 的核心 mental model：

1. **Embedding model 现在已经进入 "MLLM-as-backbone" 时代**。像 text embedding 经历 BERT → GPT → LLM-as-embedder 演进一样，multimodal embedding 经历 CLIP → CoCa → MLLM-as-embedder 演进。Gemini Embedding 2 是这个浪潮的旗舰。

2. **Native multimodal ≠ dual tower**。Native 意味着 modality 在 transformer 每一层都交互，不是只在最后 cosine 时才交互。这是为什么 interleaved input（image + text + video 混合 query）只有 native multimodal 才能处理。

3. **Pre-fine-tune + Fine-tune + Model Soup 是新一代 embedding 训练的 standard recipe**。PFT 解决范式转换（autoregressive → encoder），FT 解决 task-specific 能力，Model Soup 解决 generalization。

4. **Synthetic data 是 embedding 性能的 hidden multiplier**。LLM 生成 synthetic data 把稀疏 task space 填充为稠密 task space，这本质是用 LLM 的 world knowledge 蒸馏到 embedding space。

5. **Audio native 处理对 cross-lingual retrieval 有质变提升**。这暗示了未来 multimodal RAG 的方向：不要 ASR 中间层，直接 audio-to-document retrieval。

6. **MRL 让 embedding 兼顾 speed 和 quality**。768 维做 retrieval，3072 维做 re-ranking，一个模型多场景复用。

7. **Generalist 的胜利**：specialized domain（显微镜、天文、烹饪）的 zero-shot 性能显示，一个训练充分的 generalist embedding model 可以 outperform domain-specific 模型。这预示着 embedding model 市场可能 consolidation：少数几个 generalist 模型主导，domain-specific 模型市场萎缩。

8. **Backbone scale 决定上限**。Gemini Embedding 2 的性能很大部分来自 Gemini 2.5 的 scale。Embedding methodology 的 innovation 边际效应递减，backbone scale 是主导因素。

---

## 9. 延伸思考与潜在研究方向

Paper §6 提到的 future work 和我额外想到的：

1. **End-to-end RAG training**：当前 embedding 模型训练时是 "embedding + retrieval" 分离的，未来可以让 embedding 直接和 retrieval + LLM generation 联合训练，让 embedding 为 downstream generation 优化而非 retrieval metric 优化。

2. **Agentic retrieval**：agent 多步 retrieval 中，每一步的 query 都是上一步 LLM 输出 + 原始 context 的组合。Embedding 模型如何在这种 iterative query distribution 上保持 stable？

3. **Ranking signal integration**：Paper §6 提到 "including other signals from search systems like ranking can be hugely beneficial"。这可能指 click-through data、dwell time、user feedback 等行为信号作为 weak supervision，让 embedding 对齐用户真实偏好。

4. **Long video / long document 处理**：32 frame / 8K token 限制需要突破。可能的路线：hierarchical embedding（chunk-level + document-level）、sparse attention、retrieval-augmented embedding。

5. **Streaming audio / video**：当前 model 是 batch input，对于 streaming scenario（实时会议检索、live video search）需要 incremental embedding update 机制。

6. **Federated / privacy-preserving embedding**：企业场景下，文档不能离开本地，embedding 模型如何 fine-tune？

7. **Embedding + LLM joint reasoning**：embedding retrieve 出 top-k passages 后，LLM reasoning 时如何利用 embedding 的 similarity score 作为 calibrated probability？这是 RAG 透明化的关键。

8. **Evaluation framework for interleaved multimodal retrieval**：Paper §6 末尾提到"we invite the broader academic community to contribute novel evaluation frameworks to help benchmark these emerging capabilities"。这是真 gap：当前 benchmark 都是 single-modality input，没有 benchmark 评估 "image+text → video+audio" 这种 complex interleaved retrieval。

---

## 10. 给你的实操建议

如果你（Karpathy）要用 Gemini Embedding 2 或构建类似系统：

1. **直接调用 API 而非自训**：复现这个工作需要 Gemini-scale backbone + 大量 multimodal paired data + multi-stage training infra，成本极高。直接用 Google API 更现实。

2. **MRL 优先用 768 维**：除非对 accuracy 要求极高，否则 768 维够用，存储和 retrieval 速度快 4 倍。

3. **Audio input 直接走 native，不要 ASR**：paper 已经用 MSEB 证明 native audio 在 cross-lingual 上有质变优势。如果你的 RAG 场景有 audio query，不要做 ASR 中间层。

4. **Video 限制在 32 秒内**：超过 32 秒的视频需要 chunk 后多 embedding 或 hierarchical aggregation。

5. **如果做 domain-specific，先试 zero-shot**：Table 4 显示 Gemini Embedding 2 在显微镜、天文、烹饪等专业领域 zero-shot 都很 SOTA。先跑 zero-shot baseline 再决定要不要 fine-tune。

6. **Model Soup 1:1 是 safe default**：Table 6 显示 1:1 在 most task 上 balance 最好。

希望这个深度拆解帮你 build 起对 Gemini Embedding 2 的完整 intuition。如果你想 drill down 某个具体方面（比如 audio tokenizer 的实现、Model Soup 的 weight search、synthetic data generation 的 prompt 设计），我可以进一步展开。
