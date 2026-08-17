---
source_pdf: Massive Spikes in LLMs are Bias Vectors Mechanistic Uncovering and.pdf
paper_sha256: 6abecb900f7b14efd0897baa57ab92e030f5a464cd266413161a992149b8d57c
processed_at: '2026-08-05T16:37:52-07:00'
target_folder: Pruning-Quantization
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版

好，用大白话重新讲一遍，Andrej。

---

## 这篇 paper 在讲什么

你跑一个 LLM，每个 token 在每一层都有一个 4096 维的向量（LLaMA-2-7B 的 hidden dimension）。大部分 token，比如 "cat", "the", "quickly"，这些向量会随输入内容变化 — 它们在编码信息。

但有少数特殊 token，像 "\n"（换行）、空格、"<BOS>"（beginning of sequence），它们的向量里会有几个 channel 突然飙到几千倍大。这就是所谓的 **massive spike**。这个现象 2024 年初被 Sun et al. 发现（https://arxiv.org/abs/2402.17762），量化的人特别头疼，因为这几个巨值会把整个 dynamic range 拉爆，导致 4-bit 量化直接崩掉。

这篇 paper 的核心 claim 就一句话：

> **这些 spike 不是数字噪声，是模型的结构零件。** 经过 RMSNorm 标准化之后，这些 token 的 4096 维向量基本指向同一个固定方向，不管你喂什么输入。

---

## 为什么这件事很怪

想象你有 64 条不同的输入序列，每条都有个 "\n"。你把这 64 个 "\n" token 在某一层的向量全部取出来，经过 RMSNorm（就是除以 L2 norm），然后看它们在 4096 维空间里的方向。

正常 token（比如 64 个不同的 "the"）：方差大概 $\sim 10^0$，方向到处乱跑，因为 "the" 在不同 context 里编码不同信息。

Sink token（64 个 "\n"）：方差 $\sim 10^{-4}$，方向几乎完全一样。

**差了 4 个数量级。** 这等于在告诉你：这些 token 根本不编码输入信息，它们是模型的"固定零件"。

这篇 paper 给这些固定零件起了个名字：**Bias Vector $\mathbf{b}$**。

---

## 这些零件干嘛用

要理解这个，得先说一个 transformer 的老毛病。

Softmax 有个数学性质：输出永远正，且和为 1。这意味着 attention head **没法输出零向量**。但有时候一个 head 真的什么都不想做 — 比如某个 head 专门用来找 "主谓关系"，但当前句子没有主谓结构，这个 head 理想情况下应该输出 $\mathbf{0}$，让 residual stream 保持不变。

做不到。Softmax 不允许零。所以模型需要一个"垃圾桶"来吸收多余的 attention probability，这个垃圾桶就是 **attention sink**（https://arxiv.org/abs/2309.17453）。同时，这个垃圾桶 token 的 value vector 必须 ≈ 0，否则即使 attention 都倒给它了，它还是会污染 residual stream。这就是 **value-state drain**。

Spike 就是产生这两个机制的**上游触发器**。模型在 "\n" 这类 token 上制造巨大 activation，然后用这些 activation 来"装配" attention sink 和 value-state drain。

---

## 怎么"装配"的 — 这才是 paper 的精彩部分

这篇 paper 用几何分析揭示了三个权重 $W_K, W_Q, W_V$ 如何协同工作，把 bias vector $\mathbf{b}$ 变成一个完整的 no-op 机器。

### 第一步：$W_K$ 把 sink token 隔离出来

RMSNorm 之后，sink token 和 semantic token 在 input space 是近似正交的（cosine similarity ≈ 0，因为 sink 固定在一个方向，semantic 到处散布）。

经过 $W_K$ 投影到 key space 之后，它们的 cosine similarity 被推到 **−0.55**。在 4096 维空间里，这是近乎对立的方向 — $W_K$ 把 sink key 立成了一座"灯塔"，和所有其它 key 拉开最大距离。

### 第二步：$W_Q$ 把所有 query 拉向 sink key

更妙的是 $W_Q$。它把**所有** semantic query 投影到严格对齐 $\mathbf{k}_{\text{sink}}$ 的方向，cosine similarity ≈ +0.50，而且极稳定（标准差 0.02）。

这意味着：不管输入是什么内容，所有 query 都被 $W_Q$ 主动拉向 sink key。这就是 attention sink 的几何实现 — 不是靠 sink 自己 magnitude 大吓人，是靠 $W_Q$ 主动把所有 query 推过去。

### 第三步：$W_V$ 把 sink value 投到 null-space

光有 attention sink 不够 — 如果 sink token 的 value 不为零，它还是会污染 residual。所以 $W_V$ 得把 $\mathbf{b}$ 投影到接近零。

Paper 用 SVD 分析 $W_V$：

$$W_V = U \Sigma V^\top$$

- $U, V$: orthogonal matrices
- $\Sigma$: diagonal, singular values $\sigma_1 \geq \sigma_2 \geq \dots \geq 0$
- $V$ 的列向量是 input space 的 orthonormal basis

然后看 bias vector $\mathbf{b}$ 在这些 singular vectors 上的 energy 分布：

$$\text{CSE}_{\mathbf{b}}(k) = \sum_{i=1}^{k} (\mathbf{b} \cdot \mathbf{v}_i)^2$$

- $\mathbf{v}_i$: $W_V$ 的第 $i$ 个 right singular vector
- $(\mathbf{b} \cdot \mathbf{v}_i)^2$: $\mathbf{b}$ 在第 $i$ 个 spectral component 上的 energy

结果：semantic tokens 的 energy 在整个 spectrum 上 broad spread（用到了所有 amplification modes），但 **sink token 的 energy 完全集中在 spectral tail**（$\sigma_i \approx 0$ 的部分），也就是 null-space。

换句话说，$W_V$ 学到了一个 **spectral gatekeeper** 行为 — 对 semantic 开大门，对 bias vector 走后门。这样即使 attention 给 sink 分配了最大 weight，$W_V \mathbf{b} \approx \mathbf{0}$，对 residual stream 没影响。

---

## RoPE 的麻烦和模型的"解决方案"

这里有个潜在问题。LLM 用 RoPE 给 token 加位置编码，原理是把 4096 维空间切成 128 个 2D 子空间，每个子空间按 token position 旋转一个角度。LLaMA-2 的实现是 channel $i$ 和 channel $i + 64$ 配对（$d = 128$ per head），旋转角度 $m\theta_i$，其中 $\theta_i = 10000^{-2i/d}$ 指数衰减。

问题：如果 bias signal 在某个 channel 上，RoPE 旋转会随 position 变化，把 query-key 的 alignment 弄乱，sink 机制就崩了。

Paper 发现模型用了两个 trick 来对付这个：

### Trick 1: 躲到低频 channel

$\theta_i$ 随 $i$ 指数衰减，高 index → 低频率 → 旋转慢。模型把 dominant sink features 放在 $i \approx 55$ out of 64 的位置，接近最高 index。这样即使 sequence 很长，相对 phase shift $(m-n)\theta_i$ 也很小。

### Trick 2: Coherent pairing

LLaMA-2 的 RoPE 把 channel $i$ 和 $i + 64$ 配对。模型在 ~80% 的情况下，**同时 activate 这两个配对 channel**。

为什么？如果只用单个 channel $i$，RoPE 旋转会把 energy 从 $i$ 转到 $i + 64$，内积会 sinusoidally 振荡。但如果两个 channel 都 activate，模型 attend 的是 2D 向量的 magnitude，而 RoPE 旋转 **preserve L2 norm**。

Paper Appendix G 给了数学证明。核心公式是 post-RoPE contribution：

$$S_{\text{post}}(\Delta m) = \|\mathbf{q}_{i^*}\| \|\mathbf{k}_{i^*}\| \cos(\alpha_{i^*} - \Delta m \cdot \theta_{i^*})$$

- $i^*$: 选定的 bias subspace index
- $\alpha_{i^*}$: pre-RoPE 时 query 和 key 的夹角
- $\Delta m = m - n$: relative position
- $\theta_{i^*}$: 该 channel 的 RoPE frequency

要让 $S_{\text{post}}$ 在长 context 下保持大，需要 $\theta_{i^*} \approx 0$，所以必须用高 index channel。然后 coherent pairing 消掉 $\sin\phi$ 的 cross-terms，把旋转 variance 完全 cancel。

**这是模型在 training 中自动学到的 spectral engineering，没人显式设计。** 非常 subtle。

---

## 拿这个 insight 做量化：INSERTQUANT

既然 sink token 是 deterministic 的固定向量，那量化时根本不需要 runtime 计算。INSERTQUANT 的策略：

### Step 1: Clamp（砍掉）

检测到 sink tokens 后，在 quantization 之前把它们 clamp 到 zero。哪些 projection 的输入可以砍？

- $W_{\text{MLP}}, W_O, W_Q$ 的输入：可以砍。因为 sink 的 MLP/output states 不和 semantic tokens 交互，它的 query 不影响别人的 attention score。
- $W_K, W_V$ 的输入：不能直接砍，因为 semantic tokens 要 attend 到 sink 的 key/value（sink 机制在这里形成）。但好消息是 $\mathbf{b} \to \mathbf{k}_{\text{sink}}$ 和 $\mathbf{b} \to \mathbf{v}_{\text{sink}}$ 都是 deterministic 的，可以用 pre-computed template 代替。

### Step 2: Insert（补回来）

在每层 attention 的 BMM（batched matrix multiplication）输入处，直接 inject pre-computed 的 $\mathbf{k}_{\text{sink}}$ 和 $\mathbf{v}_{\text{sink}}$ template vectors。

### Step 3: Residual restoration

最后几层，模型本来会用 MLP 的 negative spike 来 cancel positive residual spike（disappearance phase）。Clamp 会破坏这个 cancellation，所以在 disappearance phase 之后用 pre-computed template 替换 sink token 在 residual stream 的值。

### 检测和 template 匹配

Sink token detection 用 infinity norm thresholding:

$$\tau = \alpha \cdot \text{median}(\{\|\mathbf{x}_t\|_\infty\}_{t=1}^T)$$

- $\|\mathbf{x}_t\|_\infty$: token $t$ 的 infinity norm（最大绝对值 channel）
- $\alpha$: scaling factor
- 超过 $\tau$ 的 token 就是 sink

Template 有 $K$ 个（LLaMA-2 是 2 个，Mistral 是 3 个）。匹配策略：如果检测到的 sink 数量等于 $K$，按顺序匹配；不等就用 MSE fallback：

$$k^* = \arg\min_k \|\mathbf{x}_s - \mathbf{t}_k\|_2^2$$

- $\mathbf{x}_s$: detected sink state
- $\mathbf{t}_k$: 第 $k$ 个 template

---

## 实验结果

### LLM (Table 1)

W4A4sKV4（4-bit 权重、4-bit activation、4-bit 静态 KV cache）:

| 模型 | 方法 | CSR Accuracy | WikiText PPL |
|------|------|-------------|--------------|
| LLaMA-2-7B | RTN（直接量化） | 35.56 | 5859.52 |
| LLaMA-2-7B | QuaRot（旋转） | 34.45 | 424.73 |
| LLaMA-2-7B | PrefixQuant（前缀） | 60.17 | 6.21 |
| LLaMA-2-7B | **INSERTQUANT** | **60.26** | 6.31 |

RTN 完全崩了（PPL 5859），旋转方法也没救（PPL 424），PrefixQuant 和 INSERTQUANT 基本持平，PPL 都在 6 左右。

### ViT (Table 2) — 这是跨模态的关键

| Bits | 方法 | DINOv2 ViT-B IN Acc |
|------|------|---------------------|
| W8A8sKV8 | RTN | **6.00**（完全崩） |
| W8A8sKV8 | INSERTQUANT | **74.10** |

ViT-B 在 W8A8 下 RTN 直接崩到 6%，INSERTQUANT 保留 74%。这个 gap 说明 spike 问题在 vision model 里一样严重，而且 INSERTQUANT 的 embedding-space 操作能跨模态 work，而 PrefixQuant 靠 token identity（"\n" 之类）的 heuristic 就不行。

### Overhead

Latency overhead 4.69%（A100，Triton kernel），memory overhead 0.007%（0.948 MB templates for 13.5 GB model）。基本可以忽略。

---

## 为什么这件事有意思

对你 build intuition 来说，我觉得有几个点值得咀嚼：

### 1. Transformer 不是均匀计算

同一个 layer 里，semantic tokens 和 sink tokens 走完全不同的路径。Semantic tokens 用 $W_V$ 的 amplification modes（大 singular values），sink tokens 走 null-space（小 singular values）。就像同一个 cortical area 里既有 computational neurons 也有 neuromodulatory neurons，功能完全不同。

### 2. Pre-norm 是 "crystallizer"

RMSNorm 把 magnitude 和 direction 解耦。Spike 的 extreme magnitude 被标准化掉，剩下的 direction 在 sink tokens 上是固定的。Pre-norm 不是简单的 normalization，它在把"信号"和"结构"分离。

### 3. RoPE 频域的 "safe harbor"

模型在 RoPE 的 frequency spectrum 里找到了一个稳定区间 — 最高 index、最低频率。然后围绕它构建 elaborate mechanism（coherent pairing）来 stabilize。这是 emergent 的 spectral engineering，没人显式教模型这么做。

### 4. 量化的 mechanistic 路径

传统量化把 outlier 当数字问题处理（clip, smooth, rotate）。这篇 paper 开了 mechanistic quantization 的路 — 先理解 outlier 的 functional role，再设计 surgical intervention。理解 → 手术 → 高保真低比特。

### 5. 开放问题

这篇 paper 没回答：**为什么 training 会 converge 到这个几何解？** bias vector 是在 training 早期就形成，还是 gradually emerge？这是 developmental interpretability 的开放问题。如果是 transformer + softmax + residual 的必然，那 attention sink 就是架构的"盲肠"（可以移除，如 Gated Attention, https://arxiv.org/abs/2508.10925）。如果是当前 training recipe 的偶然，那未来可能通过 regularization 从源头避免。

---

## 一句话总结

**Transformer 自己发明了"垃圾桶"来绕开 softmax 的数学约束，而且为了防止位置编码把垃圾桶弄坏，它还主动把垃圾桶放在了旋转最慢的频段。量化的人发现这个垃圾桶是固定零件，于是直接拆掉再粘回去，activation 就干净了。**

References:
- Massive activations (Sun et al. 2024): https://arxiv.org/abs/2402.17762
- Attention sink (Xiao et al. 2023): https://arxiv.org/abs/2309.17453
- Systematic outliers (An et al. 2025): https://arxiv.org/abs/2502.06415
- PrefixQuant (Chen et al. 2024): https://arxiv.org/abs/2410.13835
- QuaRot (Ashkboos et al. 2024): https://arxiv.org/abs/2404.02856
- RoPE (Su et al. 2024): https://arxiv.org/abs/2104.09864
- Vision transformers need registers (Darcet et al. 2023): https://arxiv.org/abs/2309.16588
- Attention is off by one (Evan Miller): https://www.evanmiller.org/attention-is-off-by-one.html
- Gated Attention (Qiu et al. 2026): https://arxiv.org/abs/2508.10925
- Active-dormant heads (Guo et al. 2024): https://arxiv.org/abs/2410.13835
- Spectral filters (Cancedda 2024): https://arxiv.org/abs/2402.09221

---

# Massive Spikes in LLMs are Bias Vectors — Mechanistic Uncovering and Spike-Free Quantization

非常高兴能和你讨论这篇paper, Andrej。这个工作恰好处于 mechanistic interpretability 和 efficient inference 的交叉点, 我觉得对你 build intuition 关于 transformer 内部几何结构会非常有帮助。我会从 high-level hypothesis 一直讲到 RoPE 的频域细节, 然后串到 INSERTQUANT 的工程实现。

---

## 1. 一句话 intuition

这篇 paper 的核心 claim: **那些吓人的 massive spikes (在 "\n", "<BOS>", 空格等 low-semantic tokens 上出现的几千倍 magnitude 的 activation) 不是 numerical noise, 也不是单纯的 scalar bias, 而是 rigid structural bias vectors 的 scalar 投影**。这些 vectors 经过 RMSNorm 后 converge 到几乎不变的方向, 模型主动用 $W_K, W_Q, W_V$ 三个 projection weights 把它们"装配"成 attention sink + value-state drain 机制, 实现 no-op (即 attention head "什么都不做")。

这个 reframing 从 scalar → vector 是关键, 因为它把一个看起来像 outlier 的现象, 变成了一个可以被 geometrically analyzed、可以被 pre-computed、可以被 surgically removed 的 structural component。

References:
- 原始 massive activations paper (Sun et al. 2024): https://arxiv.org/abs/2402.17762
- Attention sink 原始 paper (Xiao et al. 2023): https://arxiv.org/abs/2309.17453
- Systematic outliers (An et al. 2025): https://arxiv.org/abs/2502.06415
- "Attention is off by one" (Evan Miller): https://www.evanmiller.org/attention-is-off-by-one.html

---

## 2. 背景: 为什么需要 spike? (Build intuition from softmax constraint)

在讲 vector bias 之前, 先回顾下为什么 transformer 需要 spike 这种东西存在。

### 2.1 Softmax 的"sum-to-one"诅咒

Transformer 的 attention head 在理想情况下, 如果一个 head 当前不需要 retrieve 任何信息, 它应该输出 zero vector, 让 residual stream 保持不变 (即 no-op)。

但 softmax 有一个数学性质:
$$\text{softmax}(q \cdot K^T) \in \mathbb{R}^T, \quad \sum_t \text{softmax}_t = 1, \quad \text{softmax}_t > 0 \quad \forall t$$

输出严格正且和为 1, **没法产生零向量**。所以模型需要一个"垃圾桶"来吸收多余的 probability mass — 这就是 attention sink。同时, 对应的 value vector 必须接近零, 这样即使 attention weight 很大, 实际 contribute 到 residual stream 的也是 zero — 这就是 value-state drain。

### 2.2 Spike 的角色: 上游 catalyst

Spike 是产生 attention sink 的**上游催化剂**。它在特定 tokens 上产生极端 activation magnitude, 这些 tokens 之后被 $W_K$ 投影成特殊 key, 被 $W_Q$ 全局吸引, 被 $W_V$ 投影到 null-space。

### 2.3 Spike 的 lifecycle

An et al. (2025) 的工作刻画了 spike 的 rigid lifecycle:
- **Emergence phase**: 在 early layers (Layer 1 左右) 的 MLP down-projection 输入处出现
- **Persistence phase**: 在中间 layers (Layer 2 ~ Layer 30 for LLaMA-2-7B) 沿 residual stream 传播
- **Disappearance phase**: 在最后几层 (Layer 31) 被 MLP 的 negative spike 主动 cancel 掉

这意味着 spike 不是一个 transient noise, 它有明确的 birth 和 death。

---

## 3. Bias Vector Hypothesis (核心理论贡献)

### 3.1 从 scalar bias 到 vector bias

Sun et al. (2024) 的 Bias Hypothesis 把 spike 解释为 residual stream 里的 high-level scalar bias。这篇 paper 认为这还不够 — spike 的 magnitude 是 scalar, 但它对应的 normalized vector 是 rigid 的。

### 3.2 形式化定义

给定 pre-norm activation $\mathbf{x} \in \mathbb{R}^d$, 定义 standardized vector (post-RMSNorm, pre-scaling):

$$\hat{\mathbf{x}} = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\|\mathbf{x}\|_2^2 + \epsilon}}$$

变量解释:
- $\mathbf{x}$: pre-norm activation of a token, dimension $d$ (LLaMA-2-7B 是 4096)
- $d$: hidden dimension
- $\|\mathbf{x}\|_2^2$: L2 norm squared
- $\epsilon$: small constant 防止除零
- 这里**故意不乘 learnable scaling $\gamma$**, 因为 $\gamma$ 是 layer-specific modulation, 会混淆"bias"和"modulation"

Hypothesis: 对于任何 sink token, $\hat{\mathbf{x}}_{\text{sink}} \to \mathbf{b}$, 其中 $\mathbf{b}$ 是 invariant across samples 的 rigid bias vector。

### 3.3 离散 bias 集合

进一步, 不是所有 sink token 都 converge 到同一个 $\mathbf{b}$。模型维护一个 discrete set $\{\mathbf{b}_1, \dots, \mathbf{b}_K\}$, 每个 sink token 对应一个 prototype $\mathbf{b}_k$。$K$ 是 model-specific:
- LLaMA-2-7B: $K = 2$ (两个 templates 几乎相同, dominant negative spike 在 channel ~1400, positive spike 在 channel ~2600)
- Mistral-7B-v0.3: $K = 3$ (共享 global structure, dominant negative spike 在 channel ~2100, 但 secondary features 有可见差异)

### 3.4 Empirical evidence: 方差分析

Figure 1 的核心数据:
- **Sink tokens** 的 channel-wise variance: $\sim 10^{-4}$ (经过 RMSNorm 后)
- **Semantic tokens** 的 channel-wise variance: $\sim 10^{0}$

这个 4 个数量级的差异是 hypothesis 的 smoking gun。Semantic tokens 必须变化以编码信息, sink tokens 则被"冻结"在一个固定方向。

### 3.5 Functional verification: 替换测试

最直接的验证: 把 dynamic sink token 替换成 pre-computed static mean, 看 downstream performance 是否保持。Table 1 的 FP16 列显示, INSERTQUANT 在 full precision 下和 baseline 几乎完全 parity (LLaMA-2-7B CSR 62.44 vs 62.44, WikiText-2 PPL 5.48 vs 5.47)。这证明 sink token **不承载 semantic meaning**, 纯粹是 structural bias。

---

## 4. Mechanistic Analysis: Bias Vector 如何驱动 No-op

这部分是 paper 最漂亮的工作, 它把 attention sink 和 value-state drain 这两个现象, 用 $W_K, W_Q, W_V$ 和 $\mathbf{b}$ 的几何交互定量解释清楚。

### 4.1 Attention Sink: $W_K$ 隔离 + $W_Q$ 全局吸引

#### 4.1.1 $W_K$ 作为 contrastive amplifier

Pre-projection 时, standardized sink token $\hat{\mathbf{x}}_{\text{sink}}$ 和 semantic token $\hat{\mathbf{x}}_{\text{sem}}$ 的 cosine similarity ≈ 0 (正交, 因为 sink 在一个固定方向, semantic 散布在其它方向)。

经过 $W_K$ 投影后:
$$\mathbf{k}_{\text{sink}} = W_K \hat{\mathbf{x}}_{\text{sink}}, \quad \mathbf{k}_{\text{sem}} = W_K \hat{\mathbf{x}}_{\text{sem}}$$

它们的 cosine similarity 被推到 **−0.55** (Figure 3a)。在 $d = 4096$ 的高维空间, 这是 near-antipodal separation — 极端对立。

**Intuition**: $W_K$ 学到了一个 projection, 它把 $\mathbf{b}$ 放大到一个特殊方向, 同时把 semantic tokens 推到相反方向。这就像在 key space 里给 sink token 立了一个"灯塔", 让它和其它所有 token 都拉开距离。

#### 4.1.2 $W_Q$ 作为 global attractor

更妙的是 $W_Q$ 的行为: 它把 semantic queries $\mathbf{q}_{\text{sem}} = W_Q \hat{\mathbf{x}}_{\text{sem}}$ 投影到**严格对齐** $\mathbf{k}_{\text{sink}}$ 的方向:
$$\cos(\mathbf{q}_{\text{sem}}, \mathbf{k}_{\text{sink}}) \approx +0.50$$

这个 alignment 在整个 persistence phase 极其稳定 ($\sigma \approx 0.02$), 远超 baseline query-key interaction (~−0.20)。

**Intuition**: $W_Q$ 学到了一个"无条件朝向 sink"的 projection。无论输入 semantic 是什么, query 都被推向 sink key 的方向。这就是 attention sink 的几何实现 — 不是靠 sink 自己 magnitude 大, 而是靠 $W_Q$ 主动把所有 query 拉过去。

这呼应了 Gu et al. (2024) 的 "static key" 观察 (https://arxiv.org/abs/2410.10781), 但这篇 paper 给出了 quantitative 的几何解释。

### 4.2 Value-state Drain: $W_V$ 的 spectral null-space 投影

光有 attention sink 还不够 — 如果 sink token 的 value vector 不为零, 它还是会污染 residual stream。所以 $W_V$ 必须把 $\mathbf{b}$ 投影到接近零。

#### 4.2.1 Spectral analysis setup

对 $W_V \in \mathbb{R}^{d \times d}$ 做 SVD:
$$W_V = U \Sigma V^\top$$
- $U, V$: orthogonal matrices, columns 是 left/right singular vectors
- $\Sigma$: diagonal matrix of singular values $\sigma_1 \geq \sigma_2 \geq \dots \geq 0$
- $V$ 的 columns $\mathbf{v}_i$ 形成 input space 的 orthonormal basis, 按 singular value magnitude 排序

#### 4.2.2 Cumulative Spectral Energy (CSE)

对 normalized input token $\mathbf{z}$, 投影到 right singular vectors:
$$\mathbf{c} = \mathbf{z} V, \quad c_i = \mathbf{z} \cdot \mathbf{v}_i$$

Cumulative Spectral Energy 定义为:
$$\text{CSE}_{\mathbf{z}}(k) = \sum_{i=1}^{k} (\mathbf{z} \cdot \mathbf{v}_i)^2$$

变量解释:
- $\mathbf{z}$: normalized input token (可以是 sink 或 semantic)
- $\mathbf{v}_i$: $W_V$ 的第 $i$ 个 right singular vector
- $k$: 累积到第 $k$ 个 singular value
- $(\mathbf{z} \cdot \mathbf{v}_i)^2$: token 在第 $i$ 个 spectral component 上的 energy

#### 4.2.3 关键观察 (Figure 3b)

- **Semantic tokens** (蓝线): energy 在整个 spectrum 上 broad distribution (近似线性增长), 说明它们利用了 $W_V$ 的所有 amplification modes
- **Sink tokens** (红线): energy **完全集中在 spectral tail** ($\sigma_i \approx 0$ 的部分), 也就是 null-space

这意味着 $\mathbf{v}_{\text{sink}} = W_V \mathbf{b} \approx \mathbf{0}$, 即使 attention 给 sink 分配了最大 weight, 实际 contribute 到 residual 的也接近零。

**Intuition**: $W_V$ 学到了一个 spectral gatekeeper 行为 — 对 semantic tokens 开大门 (amplification modes), 对 bias vector 走后门 (null-space)。这种 orthogonality 是 structural 的, 在所有 persistence phase layers 都一致 (Figure 8)。

#### 4.2.4 Layer 31 的 phase transition

到了 Layer 31 (disappearance phase), $W_V$ 不再具备这个 spectral filtering 性质 — 因为 spike 已经被 MLP cancel, 不需要 drain 了。这是 lifecycle 的几何证据。

References:
- Cancedda (2024) spectral filters: https://arxiv.org/abs/2402.09221
- Guo et al. (2024a) active-dormant heads: https://arxiv.org/abs/2410.13835
- Guo et al. (2024b) value matters: https://aclanthology.org/2024.emnlp-main.1231/

---

## 5. RoPE 下的稳定性: Zones of Rotational Stability

这部分是 paper 最 unexpected 也最深的发现 — 模型主动"保护" bias vector 免受 RoPE 旋转扰动。

### 5.1 问题 setup

RoPE (https://arxiv.org/abs/2104.09864) 把 $d$ 维 representation 分成 $d/2$ 个 2D subspaces, 对 position $m$ 的 token, 第 $i$ 个 subspace 旋转 $m\theta_i$。

LLaMA-2-7B 用的是 "rotate_half" 实现 (和原始 RoPE 不同):
- Channel $i$ 和 channel $i + d/2$ 配对 (不是相邻 $2i, 2i+1$)
- 对 pair $\mathbf{v} = [x_i, x_{i+d/2}]^\top$ 在 position $m$ 应用:

$$\begin{pmatrix} x_i' \\ x_{i+d/2}' \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_i \\ x_{i+d/2} \end{pmatrix}$$

变量解释:
- $m$: token position (0, 1, 2, ...)
- $\theta_i = b^{-2i/d}$: channel-specific frequency, $b$ 通常 = 10000
- $i$: channel index, $0 \leq i < d/2$
- $x_i, x_{i+d/2}$: paired channels

**Key property**: $\theta_i$ 随 $i$ 增大**指数衰减**, 所以高 index → 低 frequency → 慢旋转。

### 5.2 RoPE 对 attention sink 的威胁

Semantic query at position $m$ 和 sink key at position $n$ 的 attention score, 在第 $i$ 个 subspace 上被 relative phase shift $(m-n)\theta_i$ 调制。如果 bias signal 在高频 channel, 随 $m$ 变化, attention score 会 oscillate, sink 机制就崩了。

### 5.3 两个 stabilization 策略 (Figure 3c)

#### 5.3.1 Channel contribution score

为了定量分析哪些 channel 真正驱动 attention sink, 定义:
$$\mathbf{c} = \mathbf{q}_{\text{sem}} \odot \mathbf{k}_{\text{sink}} \in \mathbb{R}^d$$

变量解释:
- $\mathbf{q}_{\text{sem}}$: 随机采样的 semantic query
- $\mathbf{k}_{\text{sink}}$: 随机采样的 sink key
- $\odot$: element-wise (Hadamard) product
- $\mathbf{c}_j$: 第 $j$ 个 channel 对 attention score的贡献

#### 5.3.2 Positive Contribution Ratio

只看 positive 贡献 (因为 sink 需要 large positive score):
$$R = \frac{\sum_{j \in \text{top-2}} \mathbf{c}_j}{\sum_{k: \mathbf{c}_k > 0} \mathbf{c}_k}$$

变量解释:
- numerator: top-2 channel 的 contribution 之和
- denominator: 所有 positive channel 的 contribution 之和
- $R$: top-2 channel 占总 positive contribution的比例

**Empirical result**: top-2 channels 贡献了 ~75% 的 positive attention score (Figure 3c 蓝线)。这种极端 sparsity 是 stabilization 的前提。

#### 5.3.3 策略 1: Low-frequency bias

观察 (Figure 3c 绿线): dominant sink features 集中在 $i \approx 55$ out of 64 (即接近 $d/2 = 64$ 的最高 index)。由于 $\theta_i = b^{-2i/d}$ 指数衰减, 这意味着 model 用的是**最低 frequency band**, 旋转最慢, 相对 phase shift 最小。

#### 5.3.4 策略 2: Coherent pairing

观察 (Figure 3c 红线): ~80% 的 cases, top-2 channels 形成精确的 RoPE pair $(i, i + d/2)$。

**为什么需要 pairing?** 考虑单个 channel $i$ 的 query-key 内积:
- 如果只 activate $i$, RoPE 旋转会把 energy 从 $i$ 转到 $i + d/2$, 内积会 sinusoidally oscillate
- 如果同时 activate $i$ 和 $i + d/2$, 模型 attend 的是 2D vector 的 magnitude, 而 RoPE 旋转**preserve L2 norm**

Table 4 给出 LLaMA-2-7B Layer 15 的具体数据: Head 0 用 channels 118 和 54 ($118 - 54 = 64 = d/2$), Head 23 用 117 和 53, 等等。Attn score ratio 在 paired heads 上高达 80~90%+, unpaired heads (如 Head 1 用 52 和 63) 只有 58.4%。

### 5.4 Mathematical justification (Appendix G)

#### 5.4.1 Post-RoPE bias contribution

$$S_{\text{post,bias}}(\Delta m) = \sum_{i \in \mathcal{T}_{\text{bias}}} \|\mathbf{q}_i\|_2 \|\mathbf{k}_i\|_2 \cos(\alpha_i - \Delta m \cdot \theta_i)$$

变量解释:
- $\mathcal{T}_{\text{bias}}$: bias subspace 的 index 集合
- $\mathbf{q}_i, \mathbf{k}_i$: 第 $i$ 个 2D subspace 的 query/key sub-vectors
- $\alpha_i$: pre-RoPE 时 $\mathbf{q}_i$ 和 $\mathbf{k}_i$ 的夹角
- $\Delta m = m - n$: query 和 key 的 relative position
- $\theta_i$: channel $i$ 的 RoPE frequency

#### 5.4.2 为什么 $|\mathcal{T}_{\text{bias}}|$ 必须小

如果把 energy 分到两个不同 frequency 的 subspaces ($\theta_a \neq \theta_b$):
$$S = E_a \cos(\alpha_a - \Delta m \cdot \theta_a) + E_b \cos(\alpha_b - \Delta m \cdot \theta_b)$$

两个 wave 以不同速度旋转, 随 $\Delta m$ 增大必然 drift out of phase, 产生 destructive interference。所以 model 被迫把 bias 集中到单一 subspace $i^*$。

#### 5.4.3 为什么必须是 low-frequency

单一 subspace 下:
$$S = E_{\text{total}} \cos(\alpha_{i^*} - \Delta m \cdot \theta_{i^*})$$

要让 $S$ 在长 context 下保持大, 需要 $\Delta m \cdot \theta_{i^*} \approx 0$, 即 $\theta_{i^*} \approx 0$。RoPE 频率指数衰减, 所以必须用高 index channel。

#### 5.4.4 为什么必须 coherent pairing

展开 post-RoPE contribution:
$$S = \underbrace{q_{i^*}(k_{i^*}\cos\phi - k_{i^*+d/2}\sin\phi)}_{\text{Axis } i^*} + \underbrace{q_{i^*+d/2}(k_{i^*}\sin\phi + k_{i^*+d/2}\cos\phi)}_{\text{Axis } i^*+d/2}$$

其中 $\phi = \Delta m \cdot \theta_{i^*}$。

如果只用 single channel ($q_{i^*+d/2} \approx 0$), 会引入 $-q_{i^*} k_{i^*+d/2} \sin\phi$ 的 first-order 旋转 penalty, energy 随 $\Delta m$ 衰减。

如果 coherent activate 两个 channels, 并且 geometrically align ($q_{i^*} k_{i^*+d/2} \approx q_{i^*+d/2} k_{i^*}$), 则 $\sin\phi$ cross-terms **完美 cancel**, 旋转 variance 被消除。

**Intuition**: 这是 RoPE 下的 rotational invariance trick — 不是真的 invariant, 而是通过 coherent pairing + low-frequency, 把 rotation 的破坏性影响降到可忽略。这暗示模型在 training 时学到了非常 subtle 的 spectral engineering。

References:
- RoPE 原始 paper: https://arxiv.org/abs/2104.09864
- Ruscio et al. (2025) geometric approach: https://arxiv.org/abs/2508.02546

---

## 6. INSERTQUANT: 从 insight 到 PTQ framework

### 6.1 Core insight

既然 sink tokens converge 到 rigid $\mathbf{b}$, 它们**不需要 runtime computation**。可以:
1. **Clamp**: 把 spike-carrying tokens 在 quantization 前 set to zero
2. **Insert**: 在后续层 inject pre-computed template vectors $\mathbf{k}_{\text{sink}}, \mathbf{v}_{\text{sink}}$

这把 spike 从 quantization loop 里完全切除, activation space 变成 strictly spike-free, per-tensor static quantization 就能用了。

### 6.2 哪些 projections 可以 clamp?

这个分析很精细:
- **$W_{\text{MLP}}, W_O, W_Q$ 的输入**: functionally isolated, 可以 clamp。因为 sink 的 MLP/Output states 不和 semantic tokens 交互, 它的 query vector $\mathbf{q}_{\text{sink}}$ 不影响其它 tokens 的 attention scores。
- **$W_K, W_V$ 的输入**: functionally critical, 不能简单 clamp。因为 semantic tokens 必须 attend 到 sink 的 key/value (attention sink 和 value-state drain 在这里形成)。

但好消息是: 由于 $\mathbf{b}$ structurally resilient to RoPE, $\mathbf{b} \to \mathbf{k}_{\text{sink}}$ 是 deterministic 的, $\mathbf{b} \to \mathbf{v}_{\text{sink}}$ 是 strictly deterministic (因为 $W_V$ 把它投到 null-space, 旋转不旋转都接近零)。所以可以直接用 pre-computed templates 替代。

### 6.3 Calibration phase

#### 6.3.1 Sink token detection (progressive)

**Threshold**:
$$\tau = \alpha \cdot \text{median}(\mathcal{M})$$

变量解释:
- $\mathcal{M} = \{\|\mathbf{x}_t\|_\infty\}_{t=1}^T$: 序列中所有 token 的 infinity norm 集合
- $\alpha$: scaling hyperparameter
- $\tau$: detection threshold
- 任何 $\|\mathbf{x}_i\|_\infty > \tau$ 的 token 被标记为 sink

**Progressive strategy**:
- **Stage 1 (Layer 0)**: 用 $\alpha_0$ 做初始 scan, 捕捉 early-onset spikes (一部分 sink tokens 已经显现)
- **Stage 2 (Layer 1)**: 用 $\alpha_1$ 做 definitive scan (标准 emergence point), 捕捉剩余的

两阶段 union 形成 final sink set $\mathcal{S}$, 在所有后续 layers 共享。

**默认**: 第一个 token (position 0) 总是被 include, 因为 autoregressive LM 中 BOS 永远是 sink。

#### 6.3.2 Template extraction & matching

Dictionary: $\mathcal{T} = \{\mathbf{t}_1, \dots, \mathbf{t}_K\}$, $K$ 是 model-specific (LLaMA-2-7B $K=2$, Mistral-7B $K=3$)。

对每个 persistence phase layer $l$, 从 calibration samples 提取 mean Key 和 Value vectors 作为 templates。

**Assignment strategy**:
- 如果 $N_{\text{sink}} = K$: sequential alignment (第 1 个 sink → $\mathbf{t}_1$, 第 2 个 → $\mathbf{t}_2$, ...) — 利用 causal attention + position-dependent rotation 的自然顺序
- 如果 $N_{\text{sink}} \neq K$: MSE-based fallback
$$k^* = \arg\min_k \|\mathbf{x}_s - \mathbf{t}_k^{(\text{detection})}\|_2^2$$
  变量解释:
  - $\mathbf{x}_s$: detected sink token 的 state
  - $\mathbf{t}_k^{(\text{detection})}$: 第 $k$ 个 template 在 detection phase 的版本
  - $k^*$: 最优匹配 template index

### 6.4 Runtime clamp-and-insert

对每个 persistence phase layer:
1. **Clamp**: 检测到 sink tokens 后, 在 down-projection 之前把它们 clamp 到 zero
2. **Insert**: 在 BMM (batched matrix multiplication) 输入处, 直接 inject pre-computed $\mathbf{k}_{\text{sink}}, \mathbf{v}_{\text{sink}}$

### 6.5 Residual restoration

Clamp 会破坏模型 intrinsic 的 cancellation 机制 (最后几层 MLP 的 negative spike 用来 cancel positive residual spike)。所以 disappearance phase 之后, 用 pre-computed template 替换 sink token 在 residual stream 的值。

### 6.6 Spike-aware calibration

Standard MSE-based clipping search 有两个 adaptation:

**1. Semantic-only MSE objective**:
$$\alpha^* = \arg\min_\alpha \text{MSE}(X_{\text{semantic}}, Q_\alpha(X_{\text{semantic}}))$$

变量解释:
- $X_{\text{semantic}}$: semantic tokens 的 activation
- $Q_\alpha$: clipping factor $\alpha$ 下的 quantization function
- 不 include sink tokens, 因为它们会被 insert 修正

**2. Structure preservation constraint**:
$$R_{\text{correct}} = \frac{|\mathcal{S}_{\text{quant}} \cap \mathcal{S}_{\text{fp}}|}{|\mathcal{S}_{\text{fp}}|}$$

变量解释:
- $\mathcal{S}_{\text{quant}}, \mathcal{S}_{\text{fp}}$: quantized 和 full-precision 下检测到的 sink set
- $R_{\text{correct}}$: detection 一致性

如果 $R_{\text{correct}} < 0.8$, 拒绝该 clipping factor, 防止 aggressive clipping 漏检 sink tokens。

---

## 7. Experiments

### 7.1 LLM quantization results (Table 1)

Setup: 4-bit weight, 4-bit activation, 4-bit static KV cache (W4A4sKV4), per-channel weight, per-tensor static activation, group-wise dynamic KV (group 128)。

| Model | Method | CSR FP16 | CSR W4A4sKV4 | WikiText PPL FP16 | WikiText PPL W4A4sKV4 |
|-------|--------|----------|--------------|-------------------|----------------------|
| LLaMA-3.2-3B | RTN | 60.99 | 34.57 | 7.81 | 7126.79 |
| | QuaRot-RTN | 61.04 | 38.42 | 7.81 | 59.68 |
| | PrefixQuant | 61.01 | 57.06 | 7.78 | 10.14 |
| | **InsertQuant** | 60.91 | **57.19** | 7.82 | 10.96 |
| LLaMA-2-7B | RTN | 62.44 | 35.56 | 5.47 | 5859.52 |
| | QuaRot-RTN | 62.44 | 34.45 | 5.47 | 424.73 |
| | PrefixQuant | 62.50 | 60.17 | 5.47 | 6.21 |
| | **InsertQuant** | 62.44 | **60.26** | 5.48 | 6.31 |
| LLaMA-2-70B | RTN | 68.95 | 36.22 | 3.32 | 1352.96 |
| | QuaRot-RTN | 68.91 | 53.49 | 3.32 | 9.64 |
| | PrefixQuant | 68.39 | 65.56 | 3.32 | 4.38 |
| | **InsertQuant** | 68.84 | 65.22 | 3.33 | 4.83 |
| Mistral-7B-v0.3 | RTN | 66.89 | 34.92 | 5.32 | 135012.38 |
| | QuaRot-RTN | 66.92 | 40.18 | 5.32 | 55.91 |
| | PrefixQuant | 66.33 | 64.26 | 5.34 | 5.89 |
| | **InsertQuant** | 65.84 | 62.75 | 5.35 | 6.12 |

**Key observations**:
- FP16 下 INSERTQUANT 和 baseline parity, 证明 surgical intervention 不破坏 functionality
- W4A4sKV4 下, RTN 完全崩溃 (PPL 几千到几万), QuaRot 帮助有限, PrefixQuant 和 INSERTQUANT 接近 SOTA
- INSERTQUANT 在 LLaMA-2-7B 上甚至 slightly 超过 PrefixQuant (CSR 60.26 vs 60.17)

### 7.2 ViT generalization (Table 2) — 这是 cross-modal 的关键证明

| Bits (W-A-KV) | Method | CLIP ViT-L IN Acc. | Flickr (T2I/I2T) | DINOv2 ViT-L IN | DINOv2 ViT-B IN |
|---------------|--------|---------------------|------------------|-----------------|-----------------|
| 16-16-16 | Baseline | 73.00 | 84.70/65.80 | 86.20 | 84.70 |
| | InsertQuant | 71.50 | — | 87.70 | 84.50 |
| 8-8s-8 | RTN | 65.00 | 83.20/61.08 | 84.50 | **6.00** |
| | InsertQuant | 70.00 | 84.90/63.04 | 86.50 | **74.10** |
| 4-8s-8 | RTN | 63.20 | 82.80/60.26 | 77.10 | 2.20 |
| | InsertQuant | 66.20 | 83.70/63.28 | 85.70 | 54.30 |
| 4-6s-6 | RTN | 46.20 | 68.80/52.08 | 0.30 | 1.00 |
| | InsertQuant | 57.50 | 81.40/60.42 | 33.00 | 1.60 |

**ViT-specific adaptations**:
1. Spike detection 移到 deeper layers (e.g., Layer 12 for ViT, vs Layer 1 for LLM)
2. 省略 rotation transformation (QuaRot), 因为 pre-LayerNorm 结构破坏 rotational invariance

**Why this matters**: PrefixQuant 依赖 discrete token identity (e.g., "\n", "<BOS>"), 本质上无法 generalize 到 ViT (image patches 没有离散 token)。INSERTQUANT 在 embedding space 操作, 所以可以跨 modality。DINOv2 ViT-B 在 W8A8 下 RTN 完全 collapse 到 6%, INSERTQUANT 保留 74.10% — 这个 gap 触目惊心。

References:
- Darcet et al. (2023) vision transformers need registers: https://arxiv.org/abs/2309.16588
- PrefixQuant (Chen et al. 2024): https://arxiv.org/abs/2410.13835
- QuaRot (Ashkboos et al. 2024): https://arxiv.org/abs/2404.02856

### 7.3 System overhead

- **Latency**: 4.69% overhead on NVIDIA A100 (5-step warmup), 用 Triton 实现 spike detection 和 template insertion
- **Memory**: 0.948 MB templates for 13.5 GB LLaMA-2-7B, 增加 0.007%

这个 overhead 相对 W4A4 带来的 universal speedup 完全 negligible。

---

## 8. Limitations & Open Questions

### 8.1 论文承认的局限

1. **Architectural scope**: 只对 naturally exhibit massive spikes 的模型有效。新架构如 GPT-OSS (Agarwal et al. 2025, https://arxiv.org/abs/2508.10925) 和 Gated Attention (Qiu et al. 2026) 通过 architectural modification natively suppress spikes, INSERTQUANT 对它们没必要。

2. **Theoretical proof 缺失**: paper 解释了 spike 如何执行 no-op, 但没解释**为什么 training process 会 converge 到这个 geometric solution**。这是 developmental interpretability 的开放问题。

### 8.2 我看到的一些延伸问题

**8.2.1 Training dynamics**: bias vector 是在 training 早期就形成, 还是 gradually emerge? 如果能 track $\mathbf{b}$ 在 training checkpoint 中的 evolution, 可能揭示 model 学到 no-op 的 critical period。

**8.2.2 Connection to circuit analysis**: 这个工作和 Anthropic 的 circuits work (https://transformer-circuits.pub/) 有天然联系。$W_K, W_Q, W_V$ 的 coordinated behavior 实际上构成了一个 "no-op circuit", 可以用 activation patching 进一步验证。

**8.2.3 Multi-modal sink tokens**: ViT 的 register tokens (Darcet et al. 2023) 和 LLM 的 sink tokens 是 convergent evolution 吗? 还是 shared inductive bias? 如果是后者, 可能暗示这是 transformer architecture + softmax + residual stream 的**必然** emergence, 而不是 training 的偶然。

**8.2.4 Mixture of biases ($K$)**: 为什么 LLaMA-2 是 $K=2$, Mistral 是 $K=3$? 这和 tokenizer design、training data composition、还是 architecture 有关? 不同 $\mathbf{b}_k$ 之间是否有 functional hierarchy (e.g., $\mathbf{b}_1$ 用于 paragraph boundary, $\mathbf{b}_2$ 用于 sentence boundary)?

**8.2.5 Long context 的 RoPE behavior**: paper 证明 low-frequency + coherent pairing 在 persistence phase 稳定。但在 long context (e.g., 128k tokens) 下, 即使最低 frequency 也会累积显著 phase shift。这是否是 long context degradation 的 hidden factor?

**8.2.6 Connection to SOLAR 10.7B / LLaMA-3.1 的 RoPE scaling**: LLaMA-3.1 用了 NTK-aware RoPE scaling (https://arxiv.org/abs/2309.15217). 这种 scaling 会如何影响 low-frequency band 的 bias vector 稳定性? 是有助于还是有害?

**8.2.7 Speculative decoding 的 implication**: 如果 sink tokens 是 deterministic, spec decoder 可以直接 skip 它们的 forward pass, 进一步加速。

**8.2.8 Mixture-of-Experts (MoE)**: 在 Mixtral / DeepSeek-MoE 这类模型中, sink tokens 会被 route 到哪些 experts? 如果 routing 是 deterministic 的, INSERTQUANT 可以更激进地优化。

References:
- Barbero et al. (2025) why LLMs attend to first token: https://arxiv.org/abs/2504.02732
- Yu et al. (2024) attention calibration: https://arxiv.org/abs/2406.15765

---

## 9. 对你 (Karpathy) 的 intuition 的一些思考

读完这篇 paper, 我觉得有几个点对你 build intuition 特别有价值:

### 9.1 Transformer 不是"均匀"的计算图

我们习惯把 transformer 想成 uniform layer stack, 但这个工作揭示**不同 token 在同一 layer 扮演完全不同角色**:
- Semantic tokens: 承载信息, 经历 $W_V$ 的 amplification modes
- Sink tokens: structural bias, 被 $W_V$ 投到 null-space

这有点像 biological neural circuits 中, 同一个 cortical area 既有"computational" neurons 也有"neuromodulatory" neurons, 它们在同一个 anatomical space 但功能完全不同。

### 9.2 Pre-norm 是 bias vector 的"crystallizer"

RMSNorm 的标准化作用是: 把 magnitude 信息剥离, 保留方向。这样 spike 的 extreme magnitude 就被 normalized 掉, 剩下的是 $\mathbf{b}$ 的方向。这个方向在 sink tokens 上是**不变的**。

Pre-norm 不是简单的 normalization, 它把 magnitude 和 direction 解耦, 让模型可以独立用 magnitude 做 attention sink signal (在 key space), 用 direction 做 bias vector (在 residual stream)。

### 9.3 RoPE 频域的"spectral niche"

模型在 RoPE 的 frequency spectrum 里找到了一个"safe harbor" — 最高 index, 最低 frequency。这是一个 emergent 的 spectral engineering, 没人显式设计, 但 training 自动 converge 到这里。

这和 biological systems 中"evolutionary niche"的概念很像 — 系统在 constraint landscape 里找到了 local optimum, 然后围绕它构建 elaborate mechanism (coherent pairing) 来 stabilize。

### 9.4 Quantization 的 mechanistic 路径

传统 quantization 把 activation outlier 当 numerical 问题处理 (clip, smooth, rotate)。这个 paper 开启了 mechanistic quantization 路径 — **先理解 outlier 的 functional role, 再设计 quantization 策略**。

这暗示未来 quantization 研究会更深地和 mechanistic interpretability 交织。理解 model 内部 circuit → 设计 surgical intervention → 实现 high-fidelity low-bit。

---

## 10. 总结: 这个工作在更广 map 上的位置

```
Mechanistic Interpretability                  Efficient Inference
        |                                            |
        v                                            v
  Attention Sink (Xiao 2023)              SmoothQuant (Xiao 2023a)
  Massive Activations (Sun 2024)          QuaRot (Ashkboos 2024)
  Systematic Outliers (An 2025)           PrefixQuant (Chen 2024)
        |                                            |
        +------------ Bias Vector Hypothesis --------+
                         (This paper)
                              |
                              v
                       INSERTQUANT (PTQ)
                       Cross-modal (LLM + ViT)
                       Mechanistic + Surgical
```

这篇 paper 的 contribution 不仅是 INSERTQUANT 这个 framework, 更重要的是 **reframing**: 把 outlier 当 structural vector, 用几何 + 频域分析揭示 model 内部的 implicit design, 然后用这个理解驱动 quantization。

对你来说, 我觉得最值得思考的方向是: **这种 implicit structural learning 是 transformer + softmax + residual 的必然, 还是当前 training recipe 的偶然?** 如果是必然, 那 attention sink 就是 transformer 的 "盲肠" — 可以移除 (如 Gated Attention, Qiu 2026), 但需要 architectural 替代。如果是偶然, 那 future training recipe 可能通过 regularization 避免 spike 形成, 从源头解决 quantization 问题。

Hope this helps build your intuition, Andrej! 进一步想 deep dive 哪个部分 (e.g., RoPE 数学细节、SVD spectral analysis 的 implementation、或者 ViT 的 register token vs LLM sink token 的 comparative analysis), 随时告诉我。
