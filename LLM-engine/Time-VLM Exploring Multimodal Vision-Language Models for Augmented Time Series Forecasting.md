---
source_pdf: Time-VLM Exploring Multimodal Vision-Language Models for Augmented Time
  Series Forecasting.pdf
paper_sha256: 9d3b637c91b513c4e70918b474e1831324ad5ad59fd0b84e07e23ae9f6543240
processed_at: '2026-08-12T16:17:25-07:00'
target_folder: LLM-engine
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Time-VLM 用人话讲

Andrej，我换个更直觉的方式给你讲这篇 paper，尽量用类比和 "aha moment" 来 build intuition。

---

## 1. 这篇 paper 到底在干嘛？

一句话：**把 time series 变成 "图片+文字"，喂给一个已经学会看图说话的 AI（VLM），让它帮你预测未来。**

### 为什么这么干？

想象你是一个 stock analyst。你有三种信息源：
- **原始数字** (price sequence): 100, 102, 105, 103... — 精确但枯燥
- **K-line chart** (vision): 一眼能看出 trend、support/resistance、pattern
- **News text** (text): "Fed hinted at rate cut" — 给你 context

你不会只看其中一个，你会三个一起看。**Time-VLM 就是让 AI 也这么干**。

---

## 2. 核心洞察

### 2.1 现有方法的尴尬

**Time-LLM** (用 LLM 预测 time series) 的问题：
- 把 continuous numbers 硬转成 text tokens，好比把高清视频压缩成 GIF — 信息 loss
- LLM 的 word embedding 不擅长表达 "3.14159" 和 "3.14160" 之间的微小差异

**VisionTS** (只用 vision) 的问题：
- 把 time series 画成 image，CNN/ViT 能提取 pattern
- 但没有 semantic understanding — model 知道 "这个图有个 bump"，但不知道 "这是 Black Friday 的 sales spike"

### 2.2 Aha Moment

VLM (ViLT, CLIP) 已经在 COCO 上学会 **"图片-文字对齐"**：
- 给它一张狗的图，它能理解 "a dog playing in the park"
- 这个 aligned space 里，vision 和 text 是 "绑在一起" 的

作者的 insight: **如果我把 time series 变成 image + text，扔进这个 aligned space，VLM 会自动 "理解" 这个 time series 在干嘛**。

更妙的是，作者用 UMAP 可视化 (Figure 4) 发现：
- COCO 的 image-only embedding 和 text-only embedding 各自形成独立 cluster，离 time series 远
- 但 COCO 的 **multimodal pair embedding** 和 time series data **高度重合**

这说明 VLM 的 multimodal space 是一个 "universal semantic space"，time series 天然能投影进去。Text 在中间起 bridge 作用。

---

## 3. 三个 "翻译官" (Learners)

Time-VLM 雇了三个翻译官，把 time series 翻译成三种 "语言" 喂给 VLM：

### 3.1 RAL (Retrieval-Augmented Learner) — 纯时间维度翻译官

**干啥**: 把 raw numbers 切成 patches，记到一个小本本 (memory bank) 上，每次预测时翻翻历史 "有没有见过类似 pattern"。

**怎么干**:
1. 512 个 time step → 切成 patch (length 16, stride 8) → 约 63 个 patches
2. 每个 patch encode 成 128-dim vector
3. Memory bank 存历史 patches (circular buffer，满了就覆盖最老的)
4. **Local memory**: 当前 patch 去 memory bank 里找 top-k 最像的 → "上次出现这种 pattern 是什么时候？"
5. **Global memory**: 用 self-attention 看所有 patches 的 overall 结构 → "这段数据的 big picture 是啥？"
6. 两层 memory add 起来 → temporal features

**直觉**: 像你做 stock 分析时，既看 "这只股票历史上有没有类似的走势" (local)，也看 "整个大盘现在的 state" (global)。

**为什么有效**: Ablation 显示 RAL 贡献 35.6% 性能，local (17.2%) > global (4.3%)。说明 "pattern matching" 比 "context summary" 更重要。

### 3.2 VAL (Vision-Augmented Learner) — 视觉翻译官

**干啥**: 把 time series 变成 64×64 的 "彩色图片"，让 VLM 的 vision encoder 能看。

**怎么变**:
1. **加料**: 先把 time series 加上 FFT 频域信息 + periodicity encoding (sin/cos)
   - 原始: [x1, x2, ..., x512]
   - 加料后: [x, FFT, sin(2πt/P), cos(2πt/P)] — 变成 3 "channel"
2. **卷积**: 1D conv (抓 local) → 2D conv (抓 spatial pattern) → output C 个 channel
3. **缩放**: Bilinear interpolation resize 到 64×64
4. **Normalize**: min-max 到 [0, 255] — 匹配 VLM 的 input distribution

**直觉**: 把 time series "画" 成一张 heatmap，x 轴是 time，y 轴是 value，颜色深浅表示数值。VLM 的 vision encoder 在 COCO 上学过看 natural image，现在看这种 "synthetic image" 也能 extract feature。

**为什么有效**: 贡献 9% 性能。Figure 6 展示了生成的 image，能看到 trend、seasonality、frequency pattern 都被 encode 进去了。

**为什么 64×64 够用**: 512 个 time step 压成 64×64 = 4096 pixels，看起来 lossy，但 VLM 不需要 pixel-level 精度，它需要 "high-level pattern"，64×64 够了。

### 3.3 TAL (Text-Augmented Learner) — 文字翻译官

**干啥**: 给 time series 写一段 "description"，比如:

```
This time series has:
- Mean: 25.3, Std: 4.2
- Trend: slightly increasing
- Period: 24 (daily cycle)
- Dataset: Electricity consumption
- Input length: 512, Predict horizon: 96
```

**为什么贡献只有 2.1%**: ViLT 只输出 11 个 text token (out of 156 total)，text 信号太稀疏。这其实是 VLM 的 limitation — ViLT 的 language capability 弱。

**Future potential**: 如果用 LLaVA 这种 LLM-based VLM，text 信号可能更强。但作者试了 BLIP-2 (3.7B params) 反而更差 (Table 8)，说明不是越大越好，而是 cross-modal alignment 的 quality 更重要。

---

## 4. 融合 — 怎么把三个翻译官的意见合起来？

### 4.1 Cross-Modal Attention (核心创新)

**问题**: Temporal features (from RAL) 和 multimodal features (from VLM) 在不同的 "space" 里，直接 concat 不 work。

**解决方案**: 用 temporal 当 query，multimodal 当 key/value 做 cross-attention:

$$\text{head}_i = \text{softmax}\left(\frac{Q W_i^Q (K W_i^K)^{\top}}{\sqrt{d_k}}\right) V W_i^V$$

- $Q$ = temporal features (我在 time t 的 pattern 是啥？)
- $K, V$ = multimodal features (VLM 的 visual+text embedding)
- Model 学到: "对于当前 temporal pattern，multimodal space 里有什么 relevant context？"

**直觉**: Temporal 是 "提问者"，multimodal 是 "百科全书"。Temporal 问: "我现在这个 pattern 像啥？"，Multimodal 答: "像一张上升 trend 的图，context 是 daily cycle"。

### 4.2 Gated Fusion

$$\mathbf{G} = \sigma(\mathbf{W}_g [\mathbf{F}_{tem}; \mathbf{F}_{mm}] + \mathbf{b}_g)$$
$$\mathbf{F}_{fused} = \mathbf{G} \odot \mathbf{F}_{attn} + (1 - \mathbf{G}) \odot \mathbf{F}_{mm}$$

- $G$: gate，每个 dimension 一个 weight ∈ [0, 1]
- $G$ 接近 1 → trust temporal more
- $G$ 接近 0 → trust multimodal more

**直觉**: Model 动态决定 "信谁"。Few-shot 时 data 少，multimodal prior 可信度高；Full-shot 时 data 多，temporal features 可信度高。Figure 3 显示 gate weights 分布 balanced，说明两者都有用。

---

## 5. 实验讲人话

### 5.1 Few-shot (5% / 10% data)

**故事**: 你只有 5% 的 training data，传统 model (PatchTST, FEDformer) 全崩，因为它们需要大量 data 学 pattern。Time-VLM 有 VLM 的 pre-trained knowledge 当 prior，所以 "用少量 data 就能 fine-tune 出好结果"。

**数字**: 
- ETTh1 5% data: Time-VLM 0.442 vs Time-LLM 0.627 (降 29.5%)
- 这 29.5% 就是 VLM prior 的价值

### 5.2 Zero-shot Cross-domain

**故事**: 在 ETTh1 上 train，直接 predict ETTh2 (不同 county 的 transformer temperature)。Time-VLM 能 transfer，因为 VLM 的 multimodal space 是 domain-agnostic 的。

**数字**:
- ETTh1→ETTh2: 0.338 vs Time-LLM 0.353 (降 4.2%)
- ETTm1→ETTh2: 0.354 vs 0.381 (降 7.1%)

### 5.3 Full-shot 长期预测 — 这里 Time-VLM 没赢

**故事**: Data 充足时，Time-VLM 的 VLM prior 反而成了 "包袱" — frozen VLM 的 capacity 有限，限制了 model 的上限。Specialized unimodal model (PatchTST) 在 Weather, ECL, Traffic 上略好。

**数字** (Table 5 Weather):
- Time-VLM: 0.172 MSE
- Time-LLM: 0.225 MSE (这里 Time-LLM 好于 Time-VLM!)

**Intuition**: Pre-trained prior 是双刃剑。Data 少时是救星，data 多时是 constraint。

### 5.4 Ablation (Table 6) — 谁最重要

| 移除哪个 | 性能下降 |
|---------|---------|
| RAL (temporal) | 35.6% ↓↓ |
| RAL Local | 17.2% ↓ |
| RAL Global | 4.3% ↓ |
| VAL (vision) | 9.0% ↓ |
| TAL (text) | 2.1% ↓ |

**直觉**: Temporal 最重要 (数据本身的信息)，Vision 次之 (补充 pattern)，Text 最弱 (VLM 的 language 太弱)。

### 5.5 VLM 选择 (Table 8) — 不是越大越好

| VLM | Params | MSE |
|-----|--------|-----|
| ViLT | 129M | 0.336 |
| CLIP | 168M | 0.339 |
| BLIP-2 | 3.7B | 0.342 |
| Custom (ViT+BERT) | 213M | 0.348 |

**惊人发现**: 3.7B 的 BLIP-2 不如 129M 的 ViLT！

**解释**: 关键是 **cross-modal alignment 的 quality**，不是 model size。ViLT 虽然小，但它在 COCO 上学的 image-text alignment 很 pure。BLIP-2 虽然大，但它的 LLM part 可能 "稀释" 了 alignment。

**Custom baseline 最差**: 证明 separately trained ViT + BERT 没有 aligned space，fusion 困难。这验证了 paper 的 core hypothesis — **pre-aligned multimodal space 是 key inductive bias**。

---

## 6. 限制和 Future Direction

### 6.1 作者承认的 limitation

1. **TAL 几乎没用**: 2.1% 贡献，VLM 的 text understanding 太弱
2. **Full-shot 不如 specialized model**: VLM prior 在 data 充足时是 constraint
3. **Irregular pattern 处理差**: Traffic 这种 non-stationary data，image transformation 可能丢失 sudden change

### 6.2 我的思考 — 更深层的 limitation

1. **Image 64×64 是 bottleneck**: 512 time step 压成 4096 pixel，对 long sequence 有损。能否用 higher resolution 或 adaptive resolution？
2. **Memory bank 太简单**: Circular buffer + cosine similarity。能否用 learned retrieval (像 RETRO, REALM)？或 hierarchical memory (不同时间尺度)？
3. **VLM frozen 是双刃剑**: 保证 efficiency 但限制 adaptation。能否 partial unfreeze + LoRA?
4. **没有 TS-specific VLM**: VLM 在 COCO 上 pre-train，与 time series domain gap 大。能否 build TS-VLM (time series + text pair pre-training)?
5. **Multivariate 处理粗糙**: D=862 (Traffic) 时 average along D 维度，损失变量间关系。能否用 cross-variable attention?

### 6.3 联想 — 更大的 picture

这篇 paper 让我想到几个 trend:

**Trend 1: Everything as Multimodal**
- Time series + vision + text
- Audio + text (Whisper)
- Video + audio + text (Video-LLaVA)
- 未来: Sensor data + vision + text + domain knowledge

**Trend 2: Foundation Model for Time Series**
- Chronos (Amazon): LLM tokenization for TS
- MOIRAI (Salesforce): TS foundation model
- Lag-Llama: Llama for TS
- Time-VLM: VLM for TS
- 未来: Multimodal TS foundation model?

**Trend 3: Retrieval-Augmented TS**
- Time-VLM 的 RAL 是 retrieval-based
- 类似 RETRO, RETRIEVER for LLM
- 未来: Large-scale TS memory + retrieval?

**Trend 4: Frozen Foundation + Lightweight Adapter**
- Time-VLM: frozen VLM + adapters (143M)
- LoRA, prefix tuning 的思路
- 效率高，但要权衡 capacity

---

## 7. 对比 Time-LLM (Table 7) — Efficiency

| | Time-VLM | Time-LLM |
|---|---------|---------|
| Params | 143M | 3405M (1/24) |
| Memory (ETTh1) | 2.6 GB | 37.7 GB |
| Memory (Traffic) | 24.9 GB | OOM |
| Speed | 0.2-0.5 s/iter | 0.3-0.6 s/iter |

**Practical impact**: Time-VLM 能在 single 48GB GPU 跑所有 dataset。Time-LLM 在 Traffic 上直接 OOM。这对 deployment 重要。

---

## 8. 核心公式再讲一遍 (人话版)

### Patch embedding
```
[1,2,3,4,5,6,7,8] → [[1,2,3,4],[3,4,5,6],[5,6,7,8]] (patch_len=4, stride=2)
```

### Memory retrieval
```
Current patch: [上升, 高波动]
Memory bank: [[上升, 低波动], [下降, 高波动], [上升, 高波动], ...]
→ top-k: [[上升, 高波动]] (最像的)
→ MLP → local memory feature
```

### Cross-modal attention
```
Q (temporal): "我现在是上升 pattern"
K (multimodal): VLM 知道的所有 pattern
V (multimodal): VLM 的 pattern embedding
→ Attention: "你的上升 pattern 对应 VLM space 里的这个 region"
```

### Gated fusion
```
G = sigmoid(learned_weights × [temporal, multimodal])
F_fused = G × temporal_attended + (1-G) × multimodal
→ "我现在 60% 信 temporal，40% 信 multimodal"
```

---

## 9. Final Intuition Summary

**Time-VLM 的哲学**: Time series forecasting 不只是 number crunching，它有 visual pattern (trend, seasonality) 和 semantic context (domain knowledge)。VLM 的 pre-aligned space 提供了一个天然的 "universal semantic space"，让 time series 能利用 vision+text 的 pre-trained knowledge。

**Key insight**:
1. **Multimodal > Unimodal**: Vision + Text + Temporal 互补，比单模态强
2. **Pre-aligned > Post-aligned**: VLM 的 pre-trained alignment 是 key inductive bias，比 separately trained ViT+BERT 强
3. **Frozen + Adapter > Full fine-tune**: 保留 VLM prior，用 lightweight adapter 适配 TS
4. **Few-shot/Zero-shot sweet spot**: Data 少时 prior 价值最大
5. **Not bigger is better**: ViLT (129M) > BLIP-2 (3.7B)，alignment quality > model size

**Open questions for you**:
- 如果 build 一个 TS-specific VLM (pre-train on time series + text pairs)，能提升多少？
- 能否用 diffusion model 生成更 informative time series image?
- Memory bank 能否扩展到 large-scale retrieval (像 RAG for LLM)?
- Cross-modal attention 能否换成更 sophisticated 的 fusion (如 mixture of experts)?

---

## References

- [Time-VLM GitHub](https://github.com/CityMind-Lab/ICML25-TimeVLM)
- [Time-LLM (ICLR 2024)](https://arxiv.org/abs/2310.01728)
- [VisionTS (NeurIPS 2024)](https://arxiv.org/abs/2408.17253)
- [TimesNet (ICLR 2023)](https://arxiv.org/abs/2210.02186)
- [PatchTST (ICLR 2023)](https://arxiv.org/abs/2211.14730)
- [ViLT (ICML 2021)](https://arxiv.org/abs/2102.03334)
- [CLIP (ICML 2021)](https://arxiv.org/abs/2103.00020)
- [BLIP-2 (ICML 2023)](https://arxiv.org/abs/2301.12597)
- [LLaVA (NeurIPS 2023)](https://arxiv.org/abs/2304.08485)
- [Chronos (Amazon)](https://arxiv.org/abs/2403.07815)
- [MOIRAI (Salesforce)](https://arxiv.org/abs/2402.02592)
- [GPT4TS (NeurIPS 2023)](https://arxiv.org/abs/2302.11939)
- [LLMTime (NeurIPS 2023)](https://arxiv.org/abs/2310.07841)
- [Foundation Models for TS Survey](https://arxiv.org/abs/2407.03552)

---

Andrej，这篇 paper 的 essence 是: **用 VLM 的 pre-aligned multimodal space 作为 time series 的 semantic anchor，通过 cross-modal attention + gated fusion 实现 multimodal forecasting**。它不是追求 SOTA on full-shot benchmark，而是探索 "VLM 能否 help TS" 这个方向。Few-shot/zero-shot 的 strong performance 证明了 prior 的价值，但 full-shot 的 limitation 也说明 pre-trained prior 有天花板。真正的 breakthrough 可能需要 TS-specific multimodal foundation model。

---

# Time-VLM 深度解析

Andrej，这篇 ICML 2025 的 paper 是 CityMind-Lab 的工作，核心 idea 是把 pre-trained Vision-Language Models (VLMs) 引入 time series forecasting，通过 unifying temporal、visual、textual 三种 modality 来增强预测能力。让我从动机、architecture、formula、实验四个层面为你 build intuition。

---

## 1. 核心动机 (Motivation) — 为什么 VLM 适合 time series？

### 1.1 现有方法的瓶颈

**Text-augmented methods** (Time-LLM, GPT4TS, UniTime) 利用 LLM 的 reasoning capability，但有两个 fundamental issues：
- **Modality gap**: continuous time series vs discrete text 之间的 representation alignment 会导致 information loss
- **Fine-grained temporal dynamics** 难以被 word embeddings 捕获

**Vision-augmented methods** (TimesNet, VisionTS, TimeMixer++) 利用 time series 到 image 的 natural alignment，但缺乏 semantic interpretability，无法 incorporate domain-specific knowledge。

### 1.2 Key Insight

VLMs (ViLT, CLIP, BLIP-2) 已经 pre-trained 在 aligned vision-language space 上，这个 aligned space 提供了一个天然的 bridge：
- **Text** 提供 semantic context (domain knowledge, statistical description)
- **Vision** 捕获 spatial-temporal patterns (trend, seasonality, frequency components)
- **Time series** 编码 sequential dynamics

三者在 VLM 的 shared embedding space 中可以 cross-modal interaction，而不是简单 concatenation。

### 1.3 一个有趣的 interpretability 实验 (Figure 4)

作者用 UMAP 可视化了四类 embeddings：
- COCO-Image (纯视觉) → 独立 cluster
- COCO-Text (纯文本) → 独立 cluster，与 time series 距离远
- COCO-Pair (VLM 多模态输出) → 与 time series data **maximal overlap**，且位于 time series clusters 的 center
- Time series 自身生成的 image-text pair 通过 VLM 编码后，与 COCO-Pair 高度重合

这个观察揭示了一个深刻的 insight：**VLM 的 multimodal space 是 vision 与 temporal modalities 之间的 mediator**，textual semantics 在其中起到了 bridge 的作用。这就是为什么 Time-VLM 在 few-shot/zero-shot 上表现突出——它利用了 VLM 在 COCO 上学到的 cross-modal alignment 作为 inductive bias。

---

## 2. Architecture 深度解析

整个 framework 由三个 learner 组成，最终通过 frozen VLM + gating fusion 产出 prediction。

### 2.1 Retrieval-Augmented Learner (RAL) — Temporal branch

RAL 的设计灵感来自 retrieval-augmented generation，核心是建立一个 memory bank 来捕获 hierarchical temporal dependencies。

#### 2.1.1 Patch Embedding

输入 $x_{enc} \in \mathbb{R}^{B \times L \times D}$，其中：
- $B$: batch size
- $L$: sequence length (默认 512)
- $D$: number of variables (ETT 是 7，Traffic 是 862)

通过 patch length $pl=16$, stride $st=8$ 切分得到 $N_p$ 个 patches：
$$N_p = \frac{L - pl}{st} + 1$$

每个 patch linearly projected 到 $d_{model}$-dim (默认 128)，加上 positional embedding，得到 $E_p \in \mathbb{R}^{B \times N_p \times d_{model}}$。

#### 2.1.2 Hierarchical Memory 机制

这是 RAL 最有创意的部分，分为 local 和 global 两层：

**Local Memory (检索历史相似 pattern)**:
$$\text{sim}(P, \mathcal{M}) = P \cdot \mathcal{M}^{\top}$$
其中 $\mathcal{M} \in \mathbb{R}^{M \times d_{model}}$ 是 memory bank，$M$ 是最大 capacity。对每个 batch sample $i$，retrieve top-k similar patches：
$$M_{local}^{(i)} = \text{MLP}(\text{topk}(E_p^{(i)}))$$
这个操作相当于在历史样本中找 "similar past"，类似 Attention 但是 explicit memory retrieval。Local memory 贡献了 17.2% 的性能（ablation study）。

**Global Memory (long-range 依赖)**:
$$\text{Attn}(P) = \text{MultiHead}(Q, K, V)$$
$$M_{global} = \frac{1}{N_p} \sum_{i=1}^{N_p} \text{Attn}(P)_i$$
这里 $Q, K, V$ 都是 $P$ 的 linear projections，标准 self-attention，然后 temporal averaging 得到 global representation。Global memory 贡献 4.3% 性能。

**Fusion**:
$$M_{fused} = M_{local} + M_{global}$$
element-wise addition，简单但 effective。

**Intuition**: Local memory 像是 "pattern matching" — 在历史中找 similar 片段；Global memory 像是 "context summarization" — 用 self-attention 捕获 overall structure。两者互补。

#### 2.1.3 Memory Bank Update

使用 circular buffer 策略，每次 forward pass 把当前 patch embeddings (temporal-averaged) 加入 memory bank，保留最近的 patterns。这保证 memory bank 持续更新但不爆炸。

---

### 2.2 Vision-Augmented Learner (VAL) — Vision branch

VAL 的任务是 adaptive 地把 time series transform 成 informative image，让 frozen VLM vision encoder 能 extract useful features。

#### 2.2.1 Frequency Encoding (FFT)

$$\text{FFT}(x_{enc}) = \sum_{t=0}^{L-1} x_{enc}(t) \cdot e^{-2\pi i k t / L}$$
- $k$: frequency index
- $L$: sequence length
- $t$: time step
- $i$: 虚数单位

FFT 提取 frequency components，concat 到原始 time series 上，shape 变为 $\mathbb{R}^{B \times L \times D \times 2}$。这让 image 同时携带 time-domain 和 frequency-domain 信息。

#### 2.2.2 Periodicity Encoding

$$\text{encoding}(t) = \left[\sin\left(\frac{2\pi t}{P}\right), \cos\left(\frac{2\pi t}{P}\right)\right]$$
- $t$: time step
- $P$: periodicity hyperparameter (ETTh1=24, ETTm1=96, Weather=144)

这是 Transformer 经典的 positional encoding 的变体，但用 dataset-specific periodicity。Concat 后 shape 变为 $\mathbb{R}^{B \times L \times D \times 3}$。

**Intuition**: 不同 dataset 有不同 dominant periodicity (daily, weekly)，用 dataset-specific $P$ 让 model 显式知道"这个数据有这个周期"，而不是从 data 中 implicitly 学习。

#### 2.2.3 Multi-scale Convolution

三层 convolution 提取 hierarchical features：
1. **1D Conv**: 捕获 local temporal dependencies → $\mathbb{R}^{B \times D \times H_{hidden} \times L}$
2. **Average along D** (变量维度): $\mathbb{R}^{B \times H_{hidden} \times L}$
3. **2D Conv #1**: halve channel dim
4. **2D Conv #2**: map to $C$ output channels

#### 2.2.4 Image Interpolation & Normalization

**Bilinear interpolation** resize 到 (H, W) = (64, 64)：
$$\mathbf{I}(x, y) = \sum_{i=1}^{2} \sum_{j=1}^{2} \mathbf{I}(x_i, y_j) \cdot w_{ij}$$
- $(x, y)$: target pixel coordinate
- $(x_i, y_j)$: 4 nearest neighbors
- $w_{ij}$: weights based on relative distance

**Min-max normalization** to [0, 255]:
$$\mathbf{I}_{norm} = 255 \cdot \frac{\mathbf{I}_{raw} - \text{Min}(\mathbf{I}_{raw})}{\text{Max}(\mathbf{I}_{raw}) - \text{Min}(\mathbf{I}_{raw}) + \epsilon}$$
- $\epsilon = 10^{-5}$ 防止除零

最终得到 $\mathbf{I}_{norm} \in \mathbb{R}^{B \times C \times H \times W}$，feed 给 frozen VLM vision encoder。

**Intuition**: 这一步很关键，因为 VLM 是在 natural images 上 pre-train 的，pixel distribution 必须匹配 VLM 的 expected input distribution。如果不 normalize，VLM 可能 output garbage。

---

### 2.3 Text-Augmented Learner (TAL) — Text branch

TAL 生成 contextual textual prompts，包含：
- **Statistical properties**: min, max, median, trend direction
- **Contextual information**: periodic description, input window length, forecasting horizon
- **Domain-specific knowledge** (如果可用): 例如 electricity consumption patterns

这些 prompts 通过 frozen VLM text encoder 编码。

**Limitation**: 在 ablation study 中，TAL 只贡献 2.1% 性能。作者解释为 VLM 的 text token 太稀疏 (ViLT 只有 11 个 text tokens out of 156 total)。这其实暗示了一个方向：如果用更强 language capability 的 VLM (如 LLaVA)，TAL 的作用可能更大。

---

### 2.4 Multimodal Fusion — 关键设计

#### 2.4.1 Embedding Extraction

Frozen VLM (ViLT-B/32) 处理 (image, text) pair，产出 multimodal embedding $\mathbf{F}_{mm} \in \mathbb{R}^{B \times L_f \times d_h}$：
- $L_f = 156$ (VLM token length)
- $d_h = 768$ (VLM hidden dim)

同时 RAL 产出 temporal embedding $\mathbf{F}_{tem} \in \mathbb{R}^{B \times N_p \times d_{model}}$。

#### 2.4.2 Cross-Modal Multi-Head Attention (CM-MHA)

为了解决 distribution shift between temporal 和 multimodal features，先 project 到 shared $d_{model}$-dim space，然后用 **temporal as query, multimodal as key/value** 的 cross-attention：

$$\text{CM-MHA}(Q, K, V) = \text{Cat}(\text{head}_1, \dots, \text{head}_h) W^O$$
$$\text{head}_i = \text{softmax}\left(\frac{Q W_i^Q (K W_i^K)^{\top}}{\sqrt{d_k}}\right) V W_i^V$$

其中：
- $Q = \mathbf{F}_{tem} W^Q$ (temporal 作为 query)
- $K = \mathbf{F}_{mm} W^K$ (multimodal 作为 key)
- $V = \mathbf{F}_{mm} W^V$ (multimodal 作为 value)
- $W_i^Q, W_i^K, W_i^V, W^O$: learnable projection matrices
- $d_k = d_{model}/h$: head dimension
- $h$: attention heads

**Intuition**: Temporal features "询问" multimodal features：对于当前 temporal pattern，multimodal space 里有什么 relevant context？这种 asymmetric design 让 temporal 引导 multimodal 信息提取，而不是平等融合。

Residual + LayerNorm:
$$\mathbf{F}_{attn} = \text{LayerNorm}(\mathbf{F}_{tem} + \text{CM-MHA}(Q, K, V))$$

#### 2.4.3 Gated Fusion

动态加权两个 modality:
$$\mathbf{G} = \sigma(\mathbf{W}_g [\mathbf{F}_{tem}; \mathbf{F}_{mm}] + \mathbf{b}_g)$$
$$\mathbf{F}_{fused} = \mathbf{G} \odot \mathbf{F}_{attn} + (1 - \mathbf{G}) \odot \mathbf{F}_{mm}$$

- $\mathbf{W}_g, \mathbf{b}_g$: learnable
- $\sigma$: sigmoid
- $\odot$: element-wise product

**Intuition**: Gate 让 model 动态决定何时 trust temporal features，何时 trust multimodal features。在 few-shot 场景，multimodal (VLM prior) 占比应该更高；在 full-shot 场景，temporal (data-driven) 占比应该更高。

Figure 3 显示 gate weights 的分布是 balanced 的，说明 fusion 有效。

---

### 2.5 Loss Function

End-to-end MSE training:
$$\mathcal{L} = \frac{1}{H} \sum_{h=1}^{H} \|\hat{\mathbf{Y}}_h - \mathbf{Y}_h\|^2$$
- $H$: prediction horizon (96, 192, 336, 720)
- $\hat{\mathbf{Y}}_h, \mathbf{Y}_h$: predicted vs ground truth at step $h$

**关键设计**: VLM **frozen**，只训练 RAL, VAL, TAL 和 fusion module。这让 model parameter 只有 143M，是 Time-LLM (3405M) 的 1/20。

---

## 3. 实验结果深度分析

### 3.1 Few-shot Performance (Table 1, 2)

**5% training data**:
- ETTh1: MSE 0.442 vs Time-LLM 0.627 (29.5% 降低)
- Weather: MSE 0.240 vs Time-LLM 0.260 (7.7% 降低)
- ECL: MSE 0.218 vs Time-LLM 0.179 (注意 Time-VLM 略差)

**10% training data**: 性能 gap 缩小，但 Time-VLM 仍领先多数 dataset。

**Intuition**: Few-shot 场景下，VLM 的 pre-trained multimodal knowledge 提供了强 prior，弥补了 data scarcity。这正是 Time-VLM 的核心优势——**leverage pre-trained knowledge 而不是从 scratch 学习**。

### 3.2 Zero-shot Cross-domain (Table 3)

ETTh1 → ETTh2: MSE 0.338 vs Time-LLM 0.353 (4.2% 降低)
ETTm1 → ETTh2: MSE 0.354 vs Time-LLM 0.381 (7.1% 降低)

**Intuition**: Zero-shot 性能强说明 VLM 的 multimodal space 提供了 domain-agnostic 的 temporal representation，能 transfer 到 unseen domain。但 ETTm2 → ETTh1 (0.432 vs 0.414) Time-VLM 略差，说明 transfer 不完美。

### 3.3 Long-term Forecasting (Table 5)

Full-data 场景下 Time-VLM 与 Time-LLM 接近，某些 dataset (Weather, ECL, Traffic) 略差。这说明：
- **Few-shot/zero-shot 是 Time-VLM 的 sweet spot**
- **Full-shot 时，specialized unimodal models 可能更强**

作者承认这个 limitation，未来方向是 domain-specific adaptation。

### 3.4 Ablation Study (Table 6) — Weather dataset

| Variant | Avg MSE | % Degradation |
|---------|---------|---------------|
| Full | 0.233 | - |
| w/o RAL | 0.316 | 35.6% ↑ |
| w/o RAL_L | 0.273 | 17.2% ↑ |
| w/o RAL_G | 0.243 | 4.3% ↑ |
| w/o VAL | 0.254 | 9.0% ↑ |
| w/o TAL | 0.238 | 2.1% ↑ |

**关键 takeaway**:
- RAL 最重要 (35.6%)，Local memory > Global memory
- VAL 贡献 9%
- TAL 贡献最小 (2.1%)，因为 text token 稀疏

### 3.5 VLM Backbone Comparison (Table 8)

| VLM | Params (M) | Memory (MiB) | Speed (s/iter) | MSE |
|-----|-----------|--------------|----------------|-----|
| ViLT | 128.9 | 1346 | 0.36 | 0.336 |
| CLIP | 168.4 | 1174 | 0.12 | 0.339 |
| BLIP-2 | 3763.1 | 25200 | 0.98 | 0.342 |
| Custom (ViT+BERT) | 213.2 | 1474 | 0.17 | 0.348 |

**令人意外的发现**:
- BLIP-2 (3.7B params) **不如** ViLT (128.9M)
- Custom 组合 (separately trained ViT + BERT) 最差，证明 pre-aligned multimodal space 的 inductive bias 是关键

**Intuition**: 不是 model 越大越好，而是 cross-modal alignment 的 pre-training 是关键。Custom 组合没有 aligned space，所以 fusion 困难。

### 3.6 Computational Efficiency (Table 7)

Time-VLM 只需 1.9-24.9 GB memory，而 Time-LLM 需要 37+ GB (在 Weather/ECL/Traffic 上 OOM)。

**Practical significance**: Time-VLM 可以在 single GPU (48GB) 上跑所有 dataset，Time-LLM 不行。

---

## 4. Hyperparameter Sensitivity (Figure 5)

- **Sequence length**: 96-1024，512 最优。过长引入 noise。
- **Normalization constant**: 0.4 最优，平衡 feature scaling 和 training stability。
- **Model dimension**: 短期 dataset 128-256 够用，长期/高维需要 512。
- **Gate network dimension**: 256 对中等 range 最优，长期或高维需要 336-512。

---

## 5. My Critical Thoughts

### 5.1 优点
1. **First** unify temporal + visual + textual 三模态的工作，开辟新方向
2. Frozen VLM 设计让参数量小，效率高
3. Cross-modal attention + gated fusion 的设计 elegant
4. UMAP 可视化 (Figure 4) 提供了 strong evidence 支持 motivation

### 5.2 潜在问题
1. **TAL 贡献太小** (2.1%)，text modality 几乎没用。如果用更强的 LLM-based VLM (LLaVA, BLIP-2 with LLM)，可能改善，但作者用 BLIP-2 反而更差。这暗示 VLM 的 text understanding 本身不够。
2. **Image transformation 是 lossy** 的。64×64 image 是否能 preserve 全部 temporal information？特别是 long sequence (512) 压缩到 64×64 = 4096 pixels，信息密度有限。
3. **Full-shot 不如 specialized models** on Weather, ECL, Traffic，说明 VLM prior 在 data sufficient 时反而可能 limit capacity。
4. **Memory bank 设计简单**，circular buffer + cosine similarity。更复杂的 retrieval 机制 (如 learned retrieval, hierarchical memory) 可能更好。
5. **没有探讨 VLM 的 pre-training data 影响**。如果 VLM 在 time series 相关图像 (financial charts, weather maps) 上 pre-train，性能可能大幅提升。

### 5.3 联想与扩展

这篇 paper 让我想到几个 related direction：

1. **Time series as language**: Time-LLM, Chronos (Amazon), MOIRAI (Salesforce) 都在探索这个方向。Time-VLM 是 multimodal 版本。
2. **Vision-time series alignment**: VisionTS (NeurIPS 2024) 用 MAE on ImageNet 直接 zero-shot forecasting，证明了 vision pre-training 的 transferability。
3. **Foundation models for time series**: Lag-Llama, Chronos, MOIRAI, Time-VLM 都在尝试 build TS foundation model。
4. **Multimodal financial forecasting**: 结合 chart images + news text + price series 是 obvious application。
5. **Medical time series**: ECG + clinical notes + patient history 是另一个 multimodal sweet spot。
6. **Pre-training VLM on time series**: 目前 VLM 在 natural images 上 pre-train，如果 build 一个 TS-VLM (time series + text pre-trained)，可能大幅提升。这是作者 future work 之一。

### 5.4 与 Time-LLM 的本质区别

Time-LLM 是 **reprogram LLM** 把 TS 当文本；Time-VLM 是 **leverage VLM 的 aligned space** 把 TS 投影到 multimodal space。前者是 modality conversion，后者是 modality embedding。Time-VLM 保留更多原始信息，但需要三个 learner 协同。

---

## 6. References & Further Reading

- [Time-VLM GitHub](https://github.com/CityMind-Lab/ICML25-TimeVLM)
- [Time-LLM (ICLR 2024)](https://arxiv.org/abs/2310.01728) - Text-augmented baseline
- [VisionTS (NeurIPS 2024)](https://arxiv.org/abs/2408.17253) - Vision-only baseline
- [TimesNet (ICLR 2023)](https://arxiv.org/abs/2210.02186) - 2D temporal modeling
- [PatchTST (ICLR 2023)](https://arxiv.org/abs/2211.14730) - Patch-based TS Transformer
- [ViLT (ICML 2021)](https://arxiv.org/abs/2102.03334) - Vision-language transformer
- [CLIP (ICML 2021)](https://arxiv.org/abs/2103.00020) - Contrastive image-text pretraining
- [BLIP-2 (ICML 2023)](https://arxiv.org/abs/2301.12597) - Bootstrapping language-image pretraining
- [LLaVA (NeurIPS 2023)](https://arxiv.org/abs/2304.08485) - Visual instruction tuning
- [Chronos (Amazon)](https://arxiv.org/abs/2403.07815) - LLM for TS forecasting
- [MOIRAI (Salesforce)](https://arxiv.org/abs/2402.02592) - TS foundation model
- [TimeMixer++](https://arxiv.org/abs/2410.16032) - Multi-scale time-image transformation
- [GPT4TS (NeurIPS 2023)](https://arxiv.org/abs/2302.11939) - One fits all
- [LLMTime (NeurIPS 2023)](https://arxiv.org/abs/2310.07841) - LLMs as zero-shot TS forecasters
- [Foundation Models for TS Survey (KDD 2024)](https://arxiv.org/abs/2407.03552)

---

## 7. 总结 Intuition

Time-VLM 的核心 contribution 是 **证明了 VLM 的 aligned multimodal space 是 time series forecasting 的 strong prior**，特别是在 data-scarce 场景。三个 learner (RAL, VAL, TAL) 各司其职：RAL 处理 raw temporal，VAL 转换到 vision，TAL 提供 text context，三者通过 cross-modal attention + gated fusion 整合。

**关键 takeaway**:
1. VLM 的 pre-aligned space 比 separate vision + language encoders 强 (Table 8)
2. Cross-modal attention (temporal query, multimodal KV) 是 effective fusion strategy
3. Frozen VLM + lightweight adapters 是 efficient paradigm (143M vs 3405M)
4. Few-shot/zero-shot 是 multimodal prior 的 sweet spot
5. TAL 当前 weak，未来 stronger text understanding 的 VLM 可能 unlock 更大 potential

这个工作 opens up many questions：能否 build TS-specific VLM？能否 extend to multivariate TS 的高维场景？能否与 LLM agent 结合做 multimodal TS reasoning？这些都是值得探索的方向。
