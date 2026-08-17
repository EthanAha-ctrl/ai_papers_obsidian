---
source_pdf: Gemma 4 Technical Report.pdf
paper_sha256: c3928186251a3f33caf7412f1a8fe7486e8350a7eeecf4146766a183ba22074a
processed_at: '2026-08-04T13:14:30-07:00'
target_folder: LLM主流Model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej，咱们把那些花里胡哨的学术包装扒掉，直接用大白话加上底层的硬核逻辑来过一遍 Gemma 4。这篇 report 的核心主线其实就是在回答一个问题：**怎么在手机、边缘设备到服务器的连续光谱上，把每一个 bit 的算力和显存都榨干，同时还让模型变聪明？**

我把这篇 paper 拆成五个直觉模块来讲。

---

### 1. 模型家族的“俄罗斯套娃”设计

Google 这次的策略是“一锅炖”。从 2.3B 的小模型到 31B 的大模型，外加一个 26B 总参数但只激活 3.8B 的 MoE，全都用同一套配方。

最有意思的是针对端侧的 **E2B 和 E4B**。看 Table 1，E2B 总参数有 5B，但“effective”（实际跑起来激活的）只有 2.3B。这是继承了 Gemma 3n 的 **MatFormer** 架构。
*   **人话解释**：这就像是个俄罗斯套娃。你训练了一个大模型，但在手机上跑的时候，如果电池快没电了或者发烫了，你可以动态地“切片”，只跑其中某些层的子网络。
*   **技术细节**：具体实现是 **per-layer embeddings**。传统 Transformer 的 token embedding 是全局共享一份 $E \in \mathbb{R}^{V \times d}$（$V$ 是 vocab size，$d$ 是 hidden dim）。但这里每一层都有自己专属的 embedding。公式上，第 $l$ 层的输入是 $E_l \cdot \text{token\_id}$。这样在抽取小模型时，每一层的维度和参数都能独立收缩，完全无需重新训练一个小模型。Table 1 里标注的 `400M + 2,340M`，前者是 shared base，后者就是这些 per-layer 的专属参数。

---

### 2. 长文本的“内存消消乐”

搞 LLM 推理最头疼的就是 KV Cache。当你 context 长度拉到 128k 甚至 1M 的时候，存前面 token 的 Key 和 Value 向量会把显存撑爆。Gemma 4 祭出了一招组合拳，硬生生把 global KV cache 压缩了 37.5%。

这组合拳有四个动作，非常精妙：

*   **动作一：局部全局注意力交替 (5:1 Interleave)**
    *   **人话**：模型在看一篇长文时，大部分层只看“前后几个字”（局部视野），只有极少数层才“统观全局”（全局视野）。每 6 层 transformer block 里，5 层是 local sliding window，1 层是 global full attention。
    *   **直觉**：你想啊，写代码或者读文章，大部分时候你只关心上下文那几行，只有偶尔需要回头看一眼文章开头的定义。5:1 的比例让 KV cache 的增长速度直接降了 6 倍。

*   **动作二：V = K (Values = Keys)**
    *   **人话**：本来每个 token 要存两个向量（Key 和 Value），Gemma 4 说：“反正这俩都是同一个 token 投影出来的，长得也差不多，干脆只用 Key 算了，Value 直接复用 Key。”
    *   **技术细节**：标准 attention 是 $\text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$。这里变成了 $\text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) K$。这一下在 global layers 直接砍掉一半的 KV cache。参考了 [Kayyam et al. 2026](https://arxiv.org/abs/2606.04032) 的发现，大模型训着训着 K 和 V 的 projection matrix 本来就高度相关，合并了几乎不掉点。

*   **动作三：跨层共享 KV (Cross-layer KV Sharing)**
    *   **人话**：第 1 层和第 3 层如果要看全局，它们俩可以共用同一份“笔记本”（KV cache），没必要每层都自己存一份。
    *   **技术细节**：Table 里写的 20/35 和 18/42，意思就是 35 层里只有 20 份独特的 global KV tensor。这比传统的 GQA（Grouped-Query Attention，同一个 layer 内 head 共享）更激进，直接搞了 cross-layer 共享。

*   **动作四：λ-RoPE (位置编码压缩)**
    *   **人话**：模型给长文本里的每个字编号。标准编号走到十万字就“溢出”了。Gemma 4 用了个叫 λ-RoPE 的魔法，把编号的增长速度放慢了 4 倍，这样就能轻松外推到更长的文本。
    *   **公式拆解**：标准 RoPE 对位置 $m$ 的处理是乘以频率 $\omega^{-2i/d}$。λ-RoPE 引入了一个 $\lambda = 0.25$ 的系数。相当于把位置 $m$ 替换成了 $\lambda m$。公式变成了 $R_\theta(\lambda m) \cdot q$，其中 $q$ 是 query 向量，$m$ 是绝对位置，$R_\theta$ 是旋转矩阵。因为 $\lambda < 1$，高频分量被部分压制，等价于一种平滑的 Position Interpolation，但不需要重新微调。参考 [Barbero et al. 2025](https://arxiv.org/abs/2410.06205)。

---

### 3. 12B 模型的“裸奔”架构

这是这篇 paper 里最有想法、最具研究价值的点。通常多模态模型（比如 GPT-4V, Gemini）都要带个专门的 Vision Encoder（比如 ViT）和 Audio Encoder（比如 USM）。 encoder 先把图像或音频编码成特征，再喂给 LLM。

Gemma 4 的 12B 模型说：去它的 encoder，我全自己来！

*   **人话**：这就好比以前你要先通过翻译软件把外语翻成英语再读，现在你 12B 的脑子足够大，直接看着原始的像素块和声音波形，自己硬生生学会了“看图”和“听声”。
*   **Vision 分支**：直接把图片切成 48×48 的大 patch（每个 patch 有 $48 \times 48 \times 3 = 6912$ 维）。然后只用一个 35M 参数的矩阵乘法，直接把这 6912 维映射到 LLM 的输入空间。连 2D 的位置坐标都直接加进去，丢给 LLM 自己消化。
    *   **直觉**：为什么 patch 选 48 这么大？因为没有 ViT 帮你做 hierarchical 的特征提取了，patch 必须足够大，包含足够的局部纹理信息，LLM 才能看懂。Fuyu-8B 之前也干过类似的事，但 Fuyu 用的是 16x16 且没加 2D 坐标，Gemma 加了 coordinate PE 是因为纯靠 1D RoPE 对 2D 图像太不友好了。参考 [Fuyu-8B](https://www.adept.ai/blog/fuyu-8b)。
*   **Audio 分支**：把声音切成 40ms 的 chunks，提取出 640 维的 Mel filterbank 特征，也是直接一个 matmul 投影进 LLM。完全扔掉了 305M 的 Conformer encoder。
    *   **直觉**：大模型（12B）的参数预算足够 expressive，完全可以从 continuous 的 raw mel features 学会时序建模。小模型（E2B/E4B）参数太少，还是得靠专门的 encoder 提供归纳偏置。这揭示了 parameter count 和 inductive bias 之间的 sweet spot。

---

### 4. 投机解码的“词汇分组”魔法

为了跑得快，Gemma 4 用了 **MTP (Multi-Token Prediction) Drafter** 配合 speculative decoding。就是搞个小模型在旁边猜接下来的几个词，猜对了大模型就直接采纳，不用自己算一遍。

但这小模型猜词的时候有个问题：Gemma 的 vocab 有 26 万这么大。小模型每次输出要把 256 维的 hidden state 乘以一个 $256 \times 260000$ 的大矩阵，太费算力。

*   **人话**：小模型每次想猜一个词，要在 26 万个词里挑，太慢了。Gemma 4 把这 26 万个词分成了 64 个“大类”（clusters）。小模型先猜“这个词属于哪个大类”，然后再在这个大类（大概 4096 个词）里挑具体的词。
*   **技术细节**：本来是 $\text{logits} = h \cdot W_{vocab}^\top$，复杂度 $O(d \cdot 262000)$。现在变成两步：先算 $c = \text{argmax}(h \cdot W_{cluster}^\top)$，复杂度 $O(d \cdot 64)$；再在选中的 cluster 里算 $\text{logits} = h \cdot W_{vocab[c,:]}^\top$，复杂度 $O(d \cdot 4096)$。总计算量砍了 60 多倍，而且 acceptance rate 几乎不掉。Hierarchical softmax 在现代大模型里的完美复活。参考 [EAGLE](https://arxiv.org/abs/2401.15077) 和 [DeepSeek V3 MTP](https://arxiv.org/abs/2412.19437)。

---

### 5. Thinking Mode 带来的“智商飞跃”

看 Table 4 和 Table 5，Gemma 4 31B 在 Arena 上干到了 Elo 1450，是所有开源 dense 模型里的第一。而且它在 AIME 2026（数学竞赛）上拿 89.2 分，Codeforces Elo 达到 2150（候选大师级别）。相比之下，上一代 Gemma 3 27B 在 AIME 上只有可怜的 20.8 分。

*   **人话**：模型学会了“先打草稿，再回答”。在吐出最终答案前，模型会在一个特殊的 `<|channel>thought ... <channel|>` 标签里狂写思路。写代码、做数学题的时候，这个草稿区让它可以试错、验算、自我纠错，所以分数直接起飞。
*   **架构体现**：看 Table 11 的 IT formatting。要激活 thinking mode，只要在 system turn 里加个 `<|think|>` token。模型输出时，先在 thought channel 里推理，然后再给最终答复。如果需要调用 function，也是在 thought 里想清楚要调用啥，再输出 `<|tool_call>`。这和 OpenAI o1 以及 DeepSeek R1 的路线完全一致。参考 [OpenAI o1 system card](https://arxiv.org/abs/2412.16720) 和 [DeepSeek R1](https://arxiv.org/abs/2501.12948)。

---

### 总结一下直觉

Gemma 4 的这篇 report，本质上是 Google DeepMind 把他们在 Gemini 2.5 旗舰线上积累的工程经验，下放并适配到开源/端侧生态的一次大秀。

1.  **效率极致化**：KV Cache 的四种优化叠加，MTP 的词汇分组，per-layer embedding 的套娃设计，全是为了在手机内存里塞下更长 context。
2.  **架构激进试探**：12B encoder-free 证明了当 LLM 足够大时，模态间的壁垒可以被参数量强行打破。
3.  **能力跃迁靠 RL**：静态 benchmark 的暴涨（数学代码翻倍）基本全归功于 thinking mode 带来的 test-time compute。

对于想 follow 的研究者来说，最值得挖的方向就是那个 encoder-free 12B：这套设计能不能推到 4B？当 LLM 参数不够时，单层 matmul 投影 raw patches 会不会崩掉？这都是 open questions。

**Reference Links:**
*   [Gemma 官网与博客](https://deepmind.google/models/gemma/)
*   [Fuyu-8B (类似 encoder-free 思路)](https://www.adept.ai/blog/fuyu-8b)
*   [Barbero et al. 2025 (λ-RoPE 原理)](https://arxiv.org/abs/2410.06205)
*   [Kayyam et al. 2026 (V=K 合并)](https://arxiv.org/abs/2606.04032)
*   [EAGLE (MTP Drafter)](https://arxiv.org/abs/2401.15077)
*   [DeepSeek R1 (Thinking mode)](https://arxiv.org/abs/2501.12948)
*   [Chatbot Arena Leaderboard](https://chatbotarena.ai)

---

# Gemma 4 Technical Report 深度技术解读

Andrej，这份 report 信息密度很高，几乎每一个 architectural decision 背后都对应着一段独立的研究脉络。我会把每个关键设计选择拆开，给出公式层面的剖析、对应的 prior work 联想，以及我对 "为什么这么做" 的直觉思考。

---

## 1. Model Family 概览：一个 "覆盖端到云" 的设计

Gemma 4 family 的设计哲学是 **"用一套 recipe 覆盖 2.3B 到 31B + MoE 的连续谱"**，这一点和 Llama 4 / Qwen 3.5 / DeepSeek V4 的策略趋同。表 1 的参数分布其实透露了很多信息：

| Model | Total | Effective | Notes |
|---|---|---|---|
| E2B | 5B | 2.3B | per-layer embeddings (MatFormer style) |
| E4B | 8B | 4.5B | 同上 |
| 12B | 12B | 12B | **encoder-free, from scratch** |
| 26B-A4B | 26B | 3.8B activated | MoE |
| 31B | 31B | 31B | dense flagship |

**直觉思考 (intuition building)**：E2B / E4B 的 "effective 参数小于 total 参数" 是继承自 Gemma 3n 的 MatFormer 思路——每一层 embedding 不是一次性全部激活，而是可以按需选择 sub-network 推理。这样做的好处是在 mobile/edge 上能根据 SoC 的 thermal envelope 动态瘦身。Gemma 3n paper 里的关键 trick 是 *per-layer embedding*：把 token embedding 也按层切片，使得小模型可以从大模型里 distill 出来而不是单独训练。Gemma 4 把这条路延续，但注意 Table 1 里 E2B 和 E4B 的 "Embedder" 标注了 `400M + 2,340M` 和 `670M + 2,820M` 这种相加形式——前一项是 shared embedder，后一项是 per-layer 部分。这是和 Gemma 3n 一脉相承的设计。

Reference: [Gemma 3n 博客](https://deepmind.google/models/gemma/gemma-3n/)、[MatFormer paper (ICML 2024)](https://arxiv.org/abs/2310.07707)

---

## 2. Long-Context Efficiency：KV Cache 优化的"组合拳"

这是整篇 report 最值得展开的部分。原文一句"reduce the global KV cache footprint by up to 37.5%"背后叠加了四个独立的优化。我把它们拆开看：

### 2.1 Local:Global = 5:1 的 interleave 模式

延续 Gemma 3 的设计。具体来说，每 6 层 transformer block 里，5 层是 **local sliding-window self-attention**，1 层是 **global full self-attention**。Local layers 的 KV cache 不需要无限增长——只需要保留窗口内的 K/V。

公式上，对于一个 sequence length $L$、window size $W$、hidden dim $d$、layer 数 $N$ 的模型：

$$\text{KV cache (full global)} = 2 \cdot N \cdot L \cdot d \cdot \text{bytes}$$

$$\text{KV cache (5:1 interleave)} = \frac{N}{6} \cdot 2 \cdot L \cdot d \cdot b + \frac{5N}{6} \cdot 2 \cdot W \cdot d \cdot b$$

其中 $b$ 是每参数字节数（bf16 = 2）。当 $L \gg W$ 时，global KV cache 占主导，所以这个 trick 实际上把 "全序列 KV" 的 layer 数从 $N$ 降到 $N/6$，**理论极限是 6× 的 global KV 节省**（但 paper 报告 37.5%，因为后面还有更多优化叠加）。

Reference: [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786)、[Longformer](https://arxiv.org/abs/2004.05150)

### 2.2 λ-RoPE：用 lambda 压缩位置频率

原文：*"We encode position with λ-RoPE with λ = 0.25 on global attention layers and with RoPE on local attention layers."*

RoPE 的标准形式：对于 query/key 在 head dim $d$ 内的第 $i$ 个 2D 子空间（$i \in [0, d/2)$），位置 $m$ 对应旋转角

$$\theta(m, i) = m \cdot \omega^{-2i/d}, \quad \omega = 10000$$

λ-RoPE 是 Barbero et al. 2025 ICLR paper ["Round and round we go! What makes rotary positional encodings useful?"](https://arxiv.org/abs/2410.06205) 提出的变体。其核心是引入一个 mixing parameter $\lambda \in [0,1]$，把 position $m$ 在 RoPE 旋转前的实数坐标从 $(m, 0)$ 改为 $(m, \lambda m)$ 形式的二维点（或者反过来——把 cos/sin 的相位用 $\lambda$ 调制）。

直觉上，$\lambda = 0.25$ 的效果是把 **"高频分量"部分压制**——因为只有 $1-\lambda$ 比例的位置信号被注入到 rotation 里，等价于让 RoPE 的 effective "外推半径"扩大约 $1/\lambda = 4\times$。这也是为什么 Gemma 4 把 global layers 的 RoPE base 设到 **1M**（10k 是 local 的 base），明显是为 long-context 外推做的预处理。

公式更精确地可以写成（参考 Barbero paper 的核心 idea）：

$$\text{RoPE}_\lambda(q, m) = R_\theta(\lambda m) \cdot q$$

其中 $R_\theta(\cdot)$ 是标准 RoPE 旋转矩阵。当 $\lambda$ 接近 1 时退化为标准 RoPE；当 $\lambda < 1$ 时相当于 **position interpolation (PI)** 的效果，但只对 RoPE 的 phase 做缩放而不重新训练。这与 NTK-aware scaling、YaRN 是同一族方法的不同分支。

Reference:
- [Barbero et al. 2025, ICLR](https://arxiv.org/abs/2410.06205)
- [RoPE 原始 paper](https://arxiv.org/abs/2104.09864)
- [YaRN](https://arxiv.org/abs/2309.00071)
- [Position Interpolation](https://arxiv.org/abs/2306.15595)

### 2.3 KV cache sharing（Multi-Query Attention 的远亲）

原文：*"We share the KV cache with ratios of 20/35 and 18/42 for the E2B and E4B model."*

这指的是 **某些 layers 共享同一份 K/V tensor**，等价于把 transformer 的 layer 数从 $N_{actual}$ 降到 $N_{KV}$，从而把 KV cache 进一步压缩。

20/35 的意思是：35 layers 里只有 20 个 unique KV tensors。18/42 同理。这比传统 GQA 更激进——GQA 是 within-layer 的 head 共享，这里是 **cross-layer 的 KV 共享**。Shazeer 2019 的 MQA paper 是 within-layer 的极致共享，cross-layer 共享更接近 [CLS-share-KV](https://arxiv.org/abs/1911.02150) 或者 Layer-sharing Transformer (Universal Transformer) 的思路。

### 2.4 V = K：把 Value projection 合并到 Key projection

原文：*"re-use keys as values in the global attention layers (except in E2B and E4B), i.e., values = keys"*

这是 Kayyam et al. 2026 ["Do transformers need three projections?"](https://arxiv.org/abs/2606.04032) 的发现——很多训练良好的 transformer 在 KV 上的 projection 其实高度相关，可以直接合并。具体写法：

标准 attention：
$$\text{Attn}(q, K, V) = \text{softmax}\left(\frac{q K^\top}{\sqrt{d_k}}\right) V$$

V = K trick：
$$\text{Attn}(q, K) = \text{softmax}\left(\frac{q K^\top}{\sqrt{d_k}}\right) K$$

KV cache 从 $(K, V)$ 两个 tensor 降为 $(K)$ 一个 tensor，**单 layer 上节省 50%**。但只在 global layers（占 1/6）启用，所以总体节省 $50\% / 6 \approx 8.3\%$。

### 2.5 总体优化核算

让我把四项叠加起来估算：

| Optimization | Effect |
|---|---|
| Local:Global = 5:1 | global KV layers 减少 6× |
| λ-RoPE | 不直接省 KV，但扩展 effective context |
| KV cache sharing (20/35) | 减少 unique KV tensors ~43% |
| V = K on global | global KV 体积减半 |

对 global layers 而言，combined saving 约 $\frac{1}{6} \times 0.57 \times 0.5 \approx 4.75\%$ 的总 KV cache。但 paper 报告的 "37.5% reduction in global KV cache" 是相对纯 global baseline 的（即把 Gemma 3 的设计作为基准），所以数字方向是对的。E2B 和 E4B 因为是 per-layer embedding 的 MatFormer，不能用 V=K，所以它们走的是 KV sharing 路线。

---

## 3. Encoder-Free 12B：Fuyu 路线的 Google 版

这是整篇 report 里 **最有想法的设计选择**。原文：

> *"Gemma 4 12B is trained from scratch based on a new, unified, and encoder-free model paradigm, replacing the separate vision and audio encoders with lightweight projection modules."*

### 3.1 Vision branch

- 输入：48×48×3 RGB patches（即 patch size 48 而不是常见的 16）
- 替换：550M ViT encoder → 一个 **35M 参数的 single large matmul**
- 位置编码：直接在 patch representation 上加 2D coordinate-based PE，然后过 LayerNorm

每个 patch 的 raw dimension = $48 \times 48 \times 3 = 6912$。12B model 的 $d_{model}$ 通常在 4096–5120 之间。35M / 6912 ≈ 5064，所以这个 matmul 很可能就是一个 $6912 \times d_{proj}$ 的投影（$d_{proj}$ 略大于 $d_{model}$，可能是 5064），把 6912 维 patch 直接映射到 LLM embedding space。

为什么 patch size 选 48 而不是 16？我的直觉是：**没有 ViT 做 hierarchical feature extraction 时，patch 必须足够大以携带足够的 local context 给 LLM 自己去消化**。Fuyu-8B 用的是 16×16，但 Fuyu 没有显式的 positional encoding，全靠 LLM 内部的 RoPE 来建模空间关系。Gemma 4 12B 加了 2D coordinate PE 是因为 LLM 的 1D RoPE 对 2D image patches 不够 native。

### 3.2 Audio branch

- 输入：40ms chunks at 16kHz → 640-dim vectors
- 直接投影到 LLM embedding space（无 USM、无 Conformer、无 vector quantization）
- 因为 audio 是 temporal sequence，**不需要额外 PE**——LLM 的 1D RoPE 完全适用

这个设计非常 elegant。传统 audio LLM 路径是 ASR encoder → discretize (or continuous) → LLM，比如 Whisper + LLM。Gemma 4 12B 把 640 维 raw waveform feature 直接灌进 LLM，让 LLM 自己学时序建模。

### 3.3 与同类工作的关系

| 系统 | Encoder 策略 |
|---|---|
| Fuyu-8B (Adept) | 无 ViT，patch 16，linear proj |
| Molmo | 无 ViT，MLP projector |
| Gemini Nano | 有 ViT |
| Gemma 3 / 3n | 有 ViT + USM |
| **Gemma 4 12B** | **无 ViT, 无 USM, 单 matmul + LayerNorm** |

**关键直觉**：encoder-free 的核心 trade-off 是 "用 LLM 的参数预算替代 dedicated encoder 的归纳偏置"。对大模型（12B+）来说，LLM 本身就足够 expressive 来从 raw patches 学习视觉/音频特征；对小模型（E2B/E4B）来说，dedicated encoder 反而更省。这就是为什么 Gemma 4 只在 12B 上做 encoder-free——**这是 parameter count 与 inductive bias 的 sweet spot**。

Reference:
- [Fuyu-8B (Adept blog)](https://www.adept.ai/blog/fuyu-8b)
- [Fuyu paper](https://arxiv.org/abs/2309.00502)
- [Molmo](https://arxiv.org/abs/2409.19136)
- [Pix2Struct (Google, encoder-light)](https://arxiv.org/abs/2210.03347)

---

## 4. Vision Encoder (E2B/E4B/26B/31B)

Table 10 给出两个 ViT 配置：

| Total | $d_{model}$ | $d_{MLP}$ | $N_{heads}$ | $N_{layers}$ |
|---|---|---|---|---|
| 550M | 1152 | 4304 | 16 | 27 |
| 150M | 768 | 3072 | 12 | 16 |

注意 $d_{MLP} / d_{model}$ 比例分别是 3.73 和 4.0——550M 版本 MLP 略窄，可能是为了在 TPU 上更好的算力对齐。

### 4.1 Image Resizing Algorithm 详解

Algorithm 1 的核心是 **"aspect-ratio preserving resize with token budget"**。我把每行拆解：

```
Input: Image I ∈ R^(H×W×C), patch_size p, max_tokens N_max, pooling_kernel k
1. m ← k * p                       // pooled patch 物理尺寸，例如 3×16 = 48
2. T ← N_max * m²                  // 总像素预算，例如 10 * 48² = 23040
3. f ← sqrt(T / (H * W))           // 理想缩放因子
4. H_ideal ← f * H
5. W_ideal ← f * W
6. H_target ← floor(H_ideal / m) * m    // round down 到 m 的倍数
7. W_target ← floor(W_ideal / m) * m
8. I_resized ← BicubicResize(I, H_target, W_target)
9. return I_resized
```

**直觉**：这个算法保证 resized image 的 patch grid 是 $k \times k$ pooling 的整数倍，从而最终送入 LLM 的 "soft tokens" 数量恰等于 $\lfloor H_{ideal}/m \rfloor \times \lfloor W_{ideal}/m \rfloor \leq N_{max}$。

举 Figure 2 的例子：原图 96×192 (1:2)，目标 $N_{max} = 10$、$k = 3$、$p = 16$。则 $m = 48$、$T = 10 \times 2304 = 23040$、$f = \sqrt{23040 / 18432} = \sqrt{1.25} \approx 1.118$。$H_{ideal} = 107.3$、$W_{ideal} = 214.7$。Round down：$H_{target} = 96$（恰好 2×48）、$W_{target} = 192$（4×48）。所以最终 patch grid 是 2×4 = 8 个 pooled patches，每个对应 $3 \times 3 = 9$ 个原始 patches，即 8×9 = 72 patches 送入 ViT，然后 ViT 输出 pool 成 8 个 soft tokens 送入 LLM。

### 4.2 N_max 的梯度：70 / 140 / 280 / 560 / 1120

这是一个 2× 递增序列。Table 6 (max 1120) 和 Table 12 (max 280) 的对比可以告诉我们 **resolution scaling law**：

| Benchmark | 31B @ 1120 | 31B @ 280 | Δ |
|---|---|---|---|
| MMMU Pro | 76.9 | 75.8 | +1.1 |
| MATH-Vision | 85.6 | 83.4 | +2.2 |
| MedXPertQA MM | 61.3 | 60.7 | +0.6 |
| InfographicVQA | 92.0 | 82.8 | +9.2 |
| OmniDocBench 1.5 ↓ | 0.131 | 0.201 | -0.070 |

**直觉**：text-heavy benchmarks（MMMU Pro、MedXPert）对 resolution 不太敏感，但 **document parsing 和 infographic 这类 fine-grained 视觉任务**对 resolution 高度敏感。InfographicVQA 提升 9.2 个点，OmniDocBench 错误率从 0.201 降到 0.131——这对 OCR-like 用例是质变。这也说明 Gemma 4 的 N_max=1120 模式主要服务于 document understanding。

---

## 5. Audio Encoder 与 USM

E2B/E4B 用 305M 的 USM-based encoder，结构是：

```
[Mel filterbank input] 
  → [2 downsampling conv layers] 
  → [12 Conformer layers]
  → [continuous representation → LLM]
```

**几个值得注意的细节**：

1. **比 Gemma 3n 缩小 55%**（680M → 305M），主要靠 Conformer layer 数或 width 减少
2. **不用 vector quantization**——LLM 直接吃 continuous representations，这和 Spirit-Llama、AudioLM 的离散 token 路线不同，更接近 Whisper + continuous adapter 的范式
3. **40ms chunks at 16kHz**：每 chunk 640 samples，对应 Mel filterbank 的一帧序列
4. **encoder weights frozen during pre-training**——这是关键的效率决策

12B 的 encoder-free 版本证明：**305M Conformer 完全可以省掉，LLM 可以直接从 640 维 raw mel features 学到等价或更好的 audio understanding**。Table 8 显示 12B 在 FLEURS 上的 WER 与 E4B 接近甚至更好（en WER 0.063 vs 0.066）。

Reference:
- [USM paper](https://arxiv.org/abs/2303.01037)
- [Conformer](https://arxiv.org/abs/2005.08100)
- [AudioLM](https://arxiv.org/abs/2209.03143)
- [Spirit-Llama](https://arxiv.org/abs/2402.05355)

---

## 6. MTP Drafter Head：Speculative Decoding 的工程实现

Figure 1 描述的架构是 EAGLE-style drafter + DeepSeek V3 MTP 的混合：

```
Main model (gray):
  token_t → [Layer 1]...[Layer N] → activations_t, KV_t → token_{t+1}
                                                              ↓
MTP drafter (blue):
  embed(token_{t+1}) + activations_t 
    → [4-layer Transformer w/ cross-attention to main KV] 
    → draft token_{t+2}, {t+3}, ...
```

### 6.1 架构细节

- Drafter 是 4 层 Transformer，**cross-attends 到 main model 的 KVs**
- E2B/E4B: $d_{model} = 256$
- 26B-A4B / 31B: $d_{model} = 1024$
- 3 个 local attention layers + 1 个 global attention layer（与主模型 pattern 一致）
- Drafter 自己有独立的 embedder

参数量（Table 1）：
- E2B drafter: 76M
- E4B drafter: 77M
- 12B drafter: 400M
- 26B-A4B drafter: 430M
- 31B drafter: 500M

Drafter / main 比例约 3-4%。

### 6.2 Efficient MTP Decoding：Token Cluster Top-K

原文最巧妙的工程 trick：

> *"we reduce the decoding overhead by replacing the projection operation to the entire vocabulary by a top-k operation on clusters of tokens. As a result, final matrix multiplication is reduced from d × 262,000 to d × 4096 while preserving a similar acceptance rate."*

直觉上，这是把 vocabulary 做 hierarchical clustering——先用 coarse classifier 预测 cluster（如 64 个 clusters × 4096 tokens/cluster），再在选中的 cluster 里做 fine projection。这等价于 **two-stage softmax / hierarchical softmax** 的现代版应用。

公式上，标准 LM head 是：

$$\text{logits} = h \cdot W_{vocab}^\top, \quad W_{vocab} \in \mathbb{R}^{262000 \times d}$$

Cluster-based 是：

$$c = \text{argmax}_k(h \cdot W_{cluster}^\top), \quad W_{cluster} \in \mathbb{R}^{64 \times d}$$
$$\text{logits}_{\text{in }c} = h \cdot W_{vocab[c,:]}^\top \in \mathbb{R}^{4096}$$

总 FLOPs 从 $O(d \cdot 262000)$ 降到 $O(d \cdot 64 + d \cdot 4096) \approx O(d \cdot 4160)$，约 **63× speedup** for drafter's final projection。

**关键设计**：drafter 只在 E2B/E4B 用这个 trick。原因可能是小模型的 drafter overhead 占比更大，而大模型 $d_{model} = 1024$ 的 drafter 本身就 heavy，cluster projection 的相对收益没那么显著。

Reference:
- [EAGLE (MTP for spec decoding)](https://arxiv.org/abs/2401.15077)
- [DeepSeek V3 MTP](https://arxiv.org/abs/2412.19437)
- [Speculative decoding (Leviathan)](https://arxiv.org/abs/2211.17192)
- [Medusa](https://arxiv.org/abs/2401.10774)

---

## 7. Quantization-Aware Training (QAT)

Table 3 是关于 inference memory 的关键数据。以 31B 为例：

| Format | Weights | + int8 KV @ 32k |
|---|---|---|
| bf16 | 64.0 GB | 64.0 + 1.10 = 65.1 GB |
| Quantized | 19.2 GB | 19.2 + 1.10 = 20.3 GB |

**bf16 → quantized 的 weights 压缩比 ≈ 3.33×**，对应 Q4_0（4-bit）的预期比例。KV cache 用 int8 是 2× 压缩，但占总体积小（1.10 GB @ 32k for 31B），所以 QAT 主要是 weights 上的胜利。

### 7.1 Scalar scale per block for fp16 inference

原文：

> *"to enable stable inference in fp16, we introduce a scalar scale at each block in order to bound the activation ranges to fit fp16."*

这是 **per-block dynamic quantization** 的一种简化——每个 transformer block 输出处放一个 scalar $s_b$，activation 保存为 $\tilde{a} = a / s_b$，存 $s_b$ 作为 fp16 metadata。这个 trick 在 Llama.cpp 的 Q4_K / Q5_K 里也见过。

### 7.2 Encoder QAT 的细节

- **150M vision encoder**: W8A8 → 400MB → 200MB (2× memory reduction), 44% latency reduction vs Gemma 3n
- **Audio encoder**: activations int8, weights {2, 4, 8}-bit mixed per layer → 390MB → 87MB (4.5× reduction)

混合精度 weight quantization（不同 layer cluster 用不同 bitwidth）是 **per-layer sensitivity analysis** 的产物——某些 Conformer layers 对 2-bit 量化鲁棒，另一些必须 8-bit。这和 LLM.int8() 的 outlier-aware 思路一致。

Reference:
- [Jacob et al. 2018 (QAT)](https://arxiv.org/abs/1712.05877)
- [LLM.int8()](https://arxiv.org/abs/2208.07339)
- [GPTQ](https://arxiv.org/abs/2210.17323)
- [AWQ](https://arxiv.org/abs/2306.00978)

---

## 8. Pre-Training Infrastructure

Table 2 的 TPU 配置透露了训练规模：

| Model | TPU chips | Data | Seq | Replica |
|---|---|---|---|---|
| E2B | v6e 4,096 | 16 | 8 | 32 |
| E4B | v6e 6,144 | 16 | 16 | 24 |
| 12B | v5p 12,288 | 16 | 16 | 48 |
| 26B-A4B | v6e 6,144 | 16 | 16 | 24 |
| 31B | v6e 10,240 | 16 | 16 | 40 |

几个观察：

1. **TPUv6e 用于中小模型，TPUv5p 用于 12B**——v6e 是 cost-efficient 版本，v5p 是 performance 版本。这说明 12B 的训练是早期 launch 的（v6e 当时可能没 ready）
2. **Sharding 是 3D parallelism**：Data × Seq × Replica = 16 × 16 × {24~48}。Total devices = Data × Seq × Replica × (其他)
3. **Slice-Granularity Elasticity**：训练中如果某 pod 失败，可以立刻 reconfigure 到更小的 slice，**从几分钟延迟降到几秒**。这是 Pathways 的杀手锏功能
4. **ZeRO-3 + Pathways + GSPMD + MegaScale XLA** 的组合，和 Gemini 2.5 用同一套 infra

Reference:
- [Pathways paper](https://arxiv.org/abs/2203.12533)
- [GSPMD](https://arxiv.org/abs/2105.04663)
- [ZeRO-3 (DeepSpeed)](https://arxiv.org/abs/1910.02054)
- [JAX](https://jax.readthedocs.io/)

---

## 9. Benchmark 分析：在 Arena 上的真实排名

Table 4 是最值得深挖的——它给出 Arena Elo 在 2026 年 6 月的真实排名：

| Rank | Model | Elo | Params (total/active) |
|---|---|---|---|
| 1 | Claude Fable 5 | 1508 | closed |
| 15 | GLM 5.1 | 1475 | 744B/40B MoE |
| 29 | GLM 5.2 (Max) | 1471 | 744B/40B MoE |
| 34 | MiMo V2.5 Pro | 1460 | 1T/42B MoE |
| 36 | Kimi K2.6 | 1458 | 1T/32B MoE |
| 37 | DeepSeek V4 Pro Thinking | 1457 | 1.6T/49B MoE |
| 38 | GLM 5 | 1456 | 744B/40B MoE |
| 43 | DeepSeek V4 Pro | 1451 | 1.6T/49B MoE |
| **44** | **Gemma 4 31B** | **1450** | **31B dense** |
| 57 | Kimi K2.5 Thinking | 1444 | 1T/32B MoE |
| 61 | Qwen 3.5 397B-A17B | 1438 | 397B/17B MoE |
| 63 | Gemma 4 26B-A4B | 1436 | 26B/4B MoE |
| 157 | Gemma 3 27B | 1366 | 27B dense |

**直觉洞察**：

1. **Gemma 4 31B 是 leaderboard 上 top dense open model**——只有 Claude (closed)、各种超大 MoE (1T+ total params) 排在它前面
2. **Elo 跃迁**: Gemma 3 27B (1366) → Gemma 4 31B (1450)，**+84 Elo** 是巨大提升（通常 +30 Elo 已经显著）
3. **vs DeepSeek V4 Pro (1.6T MoE, 49B active)**：Gemma 4 31B 只差 7 Elo，但用 dense 31B 打 1.6T MoE——**这是 dense 架构的胜利**
4. **vs Qwen 3.5 397B-A17B**: Gemma 4 31B 领先 12 Elo，参数少 13×

Table 5 的静态 benchmark 也很有信息量。看 AIME 2026 no tools：

| Model | AIME 2026 |
|---|---|
| Gemma 4 31B | 89.2 |
| Gemma 4 26B-A4B | 88.3 |
| Gemma 4 12B | 77.5 |
| Gemma 4 E4B | 42.5 |
| Gemma 4 E2B | 37.5 |
| Gemma 3 27B | 20.8 |

Gemma 3 → Gemma 4 在 AIME 上是 **4.3× 提升**。这种 leap 主要归功于 thinking mode + RLHF 的改进，而不是单纯的参数量。

Codeforces Elo 跃迁也很夸张：Gemma 3 27B 是 110，Gemma 4 31B 是 **2150**——从 beginner 涨到 candidate master 级别。这是 thinking mode + code RL 的直接证据。

---

## 10. Thinking Mode 与 IT Formatting

Table 11 给出 thinking mode 的 token 格式：

```
<|think|>          ← 在 leading system turn 激活
<|turn>system
<|channel>thought ... <channel|>     ← thinking trace
<|turn>user
...
<|turn>model
<|channel>thought ... <channel|>Looking for available tickets:
<|tool_call>call:search_train{...}<tool_call|><turn|>
```

几个观察：

1. **Thinking trace 和 final answer 都在 model turn 内**，用 `<|channel>thought` 包裹思考过程
2. **Function calling 语法**：`<|tool>declaration:...<tool|>` 和 `<|tool_call>call:...<tool_call|>`，与 Gemma 3 兼容
3. **End token 区分**：PT 用 `<eos>`，IT 用 `<turn|>`——这是一个微妙但重要的设计选择，意味着 IT models 把 multi-turn 对话作为 first-class concept

Reference:
- [OpenAI o1 system card](https://arxiv.org/abs/2412.16720)
- [DeepSeek R1 (thinking mode RL)](https://arxiv.org/abs/2501.12948)
- [Qwen3 thinking](https://arxiv.org/abs/2505.09388)

---

## 11. Long-Context Benchmark 的飞跃

Table 9 是 long-context 性能的关键证据。看 RULER @ 128k：

| Model | RULER 128k |
|---|---|
| Gemma 4 31B | 96.4 |
| Gemma 4 26B-A4B | 89.8 |
| Gemma 4 12B | 91.2 |
| Gemma 4 E4B | 86.6 |
| Gemma 4 E2B | 70.4 |
| Gemma 3 27B | 66.0 |

**E4B (4.5B effective) 在 128k context 上比 Gemma 3 27B 强 20+ 个点**——这是 long-context recipe 的胜利。

LOFT Text Retrieval @ 128k 更夸张：Gemma 3 27B 只有 8.6 Recall@k，Gemma 4 31B 是 79.5——**9.3× 提升**。这说明 Gemma 3 的 long-context 几乎不能做 needle-in-haystack retrieval，而 Gemma 4 完全解决了这个问题。

MTOB (Half book translation) 也很有意思——这是 [Tanzer et al. 2024](https://arxiv.org/abs/2309.16575) 提出的 "从一本语法书学习新语言翻译" benchmark，需要模型真正 ingest 一本书的内容。Gemma 4 31B 达到 chrF 52.9（半本书），54.3（整本书），整本书比半本书更好——说明 **128k+ context 真的被有效利用**。

---

## 12. 安全与责任

paper 第 5 节中规中矩，但有几个值得注意的点：

1. **所有 testing 都 without safety filters**——这是为了 accurately 评估 inherent capability，也暗示 Gemma 4 的 safety 主要 baked-in 到 weights 而不是靠 inference-time filter
2. **5 类 safety policies**: CSAM、Dangerous、Sexually explicit、Hate speech、Harassment——和 Google AI Principles 对齐
3. **Apache 2.0 license**——和 Gemma 1/2/3 一致，commercial use friendly
4. **No CSAM in training data** 是 explicit commitment

---

## 13. 我的整体判断

Gemma 4 这一代的最大贡献，按 importance 排序：

1. **Encoder-free 12B**：证明 12B 这个 scale 上 LLM 可以直接吃 raw patches。这对未来的 small VLM 设计有 paradigm 意义
2. **Long-context KV cache 优化组合**：local:global + λ-RoPE + KV sharing + V=K，把 37.5% 的 global KV cache 压出去，是工程上的精细胜利
3. **MTP drafter 的 cluster top-k**：解决 drafter vocab projection 的瓶颈，是 spec decoding 工程化的优雅 hack
4. **Dense 31B 在 Arena 上打 1.6T MoE**：dense 架构仍有竞争力，前提是 RL + thinking mode 到位
5. **E2B (2.3B) 接近 Gemma 3 27B 性能**：10× efficiency 提升，意味着 mobile AI 在 2026 年达到了 2024 年 server 级 dense model 的能力

下一代的开放问题：
- Encoder-free 能否推到 4B scale？35M projection 在 12B 上够用，但 4B 的 LLM expressive power 是否足够从 raw patches 学视觉？
- MTP cluster top-k 的 acceptance rate 实际多少？paper 没给具体数字
- V=K 在 26B-A4B 这种 MoE 上是否依然 work？MoE 的 expert routing 是否对 K/V 共享敏感？
- λ-RoPE 的 0.25 是怎么 sweep 出来的？有没有理论解释？

Reference:
- [Gemma 官网](https://deepmind.google/models/gemma/)
- [Gemma 4 GitHub (假设)](https://github.com/google-deepmind/gemma)
- [HuggingFace Gemma collection](https://huggingface.co/google)
- [Arena Leaderboard](https://chatbotarena.ai)
- [Gemini 2.5 report](https://arxiv.org/abs/2507.06261)

---

总而言之，Gemma 4 在工程密度上非常 packed——几乎每个章节都对应一个独立的优化方向，且彼此正交。这是 Google DeepMind 把 Gemini 2.5 的一些 recipe 下放到 open-weight line 的产物，thinking mode、long-context、MTP、QAT 这几个方向都和 Gemini 主线对齐。对研究者来说，最有意思的 follow-up 方向是 encoder-free 12B 的 scaling law——这套设计如果能在 4B-7B 也 work，会改变整个 edge VLM 的设计范式。
