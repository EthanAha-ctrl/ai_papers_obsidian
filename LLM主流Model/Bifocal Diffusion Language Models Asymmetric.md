---
source_pdf: Bifocal Diffusion Language Models Asymmetric.pdf
paper_sha256: 89c6ec358993d41675adec6902cb4a7c4e5b61efe329d4da2a932647bd2f8659
processed_at: '2026-07-18T18:38:27-07:00'
target_folder: LLM主流Model
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Bifocal Diffusion Language Models (R2LM) 深度解读

Andrej, 这篇 paper 直击当前 dLLM (discrete diffusion language model) 研究的核心痛点, 用一个极其 elegant 的 asymmetric 设计绕过了 attention 与 KV cache 的死结。我把整个故事拆开讲, 顺便把每一处工程细节和 intuition 都铺开。

---

## 1. 核心问题: dLLM 的 architectural dilemma

dLLM 通过 iteratively denoise masked tokens 实现 parallel generation, 一个 N-token 序列只需要 $T \ll N$ 步就能完成, 理论上比 AR (autoregressive) 快 3-8×。但是这里有个根本矛盾:

- ❶ **Bidirectional attention** (LLaDA, Dream): 每个 position 都能 attend 到 full context, 质量好, 但是每个 token 的 KV representation 依赖于未来尚未 resolved 的 tokens, 因此 **prefix KV cache 完全失效**。每个 denoising step 都要重算 $O(B(P+G)^2)$ 的 attention。
- ❷ **Causal attention** (WeDLM, CARD): 标准 left-to-right dependency, KV cache 可以用, 但是 **右侧 context 全部丢失**, 质量显著下降。

现有 work 都是不同程度的妥协 (table 1):
- **Block diffusion** (BD3-LM): 在 fixed block 内 bidirectional, 跨 block causal, 右侧 context 限制在一个 block 内 (32-128 tokens)
- **Hybrid AR-Diffusion** (ReFusion): 双模式架构, 复杂
- **Causal variants**: 接受 right-context 质量差距

核心 insight: **能不能通过非 attention 的通路来提供 right-context, 让 prefix KV cache 保持完整?** 这就是 Bifocal paradigm 的发问。

---

## 2. 信息论 motivation: 这个 gap 是什么?

这部分是整篇 paper 最漂亮的理论支撑, 我详细解释。

### 2.1 MDLM objective (equation 2)

$$\mathcal{L}(\theta) = \mathbb{E}_{t, \mathbf{x}, \mathbf{z}_t}\left[w(t) \sum_{i \in \mathcal{M}_t} -\log p_\theta(x_i \mid \mathbf{z}_t)\right], \quad w(t) = \frac{1}{1-t}$$

变量解释:
- $\theta$: model parameters
- $t \in [0, 1]$: continuous time, forward process 的时间变量
- $\mathbf{x} = (x_1, \dots, x_L)$: clean token sequence, 长度 $L$
- $\mathbf{z}_t$: corrupted sequence at time $t$, 下标 $t$ 表示时间步
- $w(t) = 1/(1-t)$: ELBO weight, 当 $t \to 1$ (几乎全 mask) 时 weight 爆炸, 这正是 absorbing state 的特性
- $\mathcal{M}_t$: 在 time $t$ 被 mask 的 position 集合
- $i$: position index, 下标
- $p_\theta(x_i \mid \mathbf{z}_t)$: 给定 corrupted sequence, 预测 position $i$ 处 clean token 的概率

Forward corruption process (equation 1):
$$q(z_t^i = m \mid x_i) = \alpha_t \cdot \mathbf{1}[m = x_i] + (1 - \alpha_t) \cdot \mathbf{1}[m = [MASK]]$$

- $\alpha_t \in [0, 1]$: survival probability, 采用 linear schedule $\alpha_t = 1 - t$
- $z_t^i$: position $i$ 在 time $t$ 的 corrupted token, 上标 $i$ 是 position
- $[MASK]$: 特殊 absorbing mask token
- $m$: candidate token value

### 2.2 Right-Context Gap (equation 3) — 这个公式是整个 paper 的灵魂

$$\Delta_i^t := H(x_i \mid \mathbf{x}_{\mathcal{O}_t}^{\leq i}) - H(x_i \mid \mathbf{x}_{\mathcal{O}_t}) := I\left(x_i; \mathbf{x}_{\mathcal{O}_t}^{>i} \mid \mathbf{x}_{\mathcal{O}_t}^{\leq i}\right) \geq 0$$

变量解释:
- $\Delta_i^t$: position $i$ 在 time $t$ 处, bidirectional 和 causal 之间的 Bayes-optimal information gap
- $H(\cdot \mid \cdot)$: conditional entropy
- $\mathbf{x}_{\mathcal{O}_t}^{\leq i} = \{x_j : j \in \mathcal{O}_t, j \leq i\}$: 在 time $t$ 已经 observed 的, 且位置 $\leq i$ 的 tokens (left context, 包含 $i$ 自己)
- $\mathbf{x}_{\mathcal{O}_t}^{>i} = \{x_j : j \in \mathcal{O}_t, j > i\}$: 已 observed 的, 位置 $>i$ 的 tokens (right context)
- $\mathcal{O}_t$: time $t$ 的 observed position 集合 (没被 mask 的)
- $I(\cdot ; \cdot \mid \cdot)$: conditional mutual information

**直觉**: causal dLLM 只看 left, bidirectional dLLM 看左右两边, 这两个 Bayes-optimal entropy 之差, 恰好等于 target token $x_i$ 与右侧 observed context 之间的 conditional mutual information (给定左侧 context)。这是 chain rule of mutual information 的直接推论。

关键 insight: 在 mask rate $\gamma$ 下, position $i$ 右侧平均有 $(1-\gamma)(L-i)$ 个 observed tokens, 因此 **越靠左的 position $\Delta_i^t$ 越大**, 越靠右越小, 最右端 masked position 的 $\Delta_i^t \to 0$。这个分布性质后面会用来解释为什么 R2LM 在 long-target tasks 上提升最明显。

### 2.3 Additive log-posterior decomposition (Proposition 1, equation 4)

$$\log p(x_i \mid \mathbf{x}_{\mathcal{O}_t}) = \underbrace{\log p(x_i \mid \mathbf{x}_{\mathcal{O}_t}^{\leq i})}_{\text{left-context term}} + \underbrace{\log \frac{p(x_i \mid \mathbf{x}_{\mathcal{O}_t})}{p(x_i \mid \mathbf{x}_{\mathcal{O}_t}^{\leq i})}}_{\Delta_i^{\text{R2L}}: \text{right-context correction}}$$

变量:
- 第一项: causal attention 已经能精确计算的 left-context term
- 第二项 $\Delta_i^{\text{R2L}}$: right-context correction, 是一个 log-ratio

并且 (equation 5):
$$\mathbb{E}_{(x_i, \mathbf{x}_{\mathcal{O}_t})}\left[\Delta_i^{\text{R2L}}\right] = \mathbb{E}_{\mathbf{x}_{\mathcal{O}_t}}\left[D_{\text{KL}}\left(p(x_i \mid \mathbf{x}_{\mathcal{O}_t}) \|\ p(x_i \mid \mathbf{x}_{\mathcal{O}_t}^{\leq i})\right)\right] = I\left(x_i; \mathbf{x}_{\mathcal{O}_t}^{>i} \mid \mathbf{x}_{\mathcal{O}_t}^{\leq i}\right)$$

也就是说, $\Delta_i^{\text{R2L}}$ 在期望意义下就是 information gap。

**这个分解直接给出了 architecture blueprint**:
- ❶ 保留 causal pathway → 精确提供 left-context term + KV cache 可重用
- ❷ 加一个 residual signal, 由 $j > i$ 的 tokens 驱动, 来近似 $\Delta_i^{\text{R2L}}$
- ❸ zero-init residual → 增强后的 model 在初始化时 bit-identical 于 causal baseline

注意 paper 诚实地说: LayerNorm 和 softmax 破坏严格可加性, 所以 equation 4 是 **inductive bias** 而非严格等式。这个 honesty 我很欣赏。

---

## 3. R2LM Architecture 详解 (figure 2)

### 3.1 整体设计

Causal Transformer backbone 完全 unmodified, 提供:
- 精确的 left-context term
- 标准 prefix KV cache 兼容性

Reverse-direction Mamba SSM 作为 **sidecar**, 通过 forward hooks 挂在每 $k$-th decoder layer (实验中 $k=4$, 28-layer Qwen3-1.7B 上是 $H=7$ hooks), 提供 right-context correction $\Delta_i^{\text{R2L}}$。

### 3.2 R2L stream 的四步 (这是核心)

设 $\mathbf{h} \in \mathbb{R}^{B \times L \times d}$ 是 layer $\ell$ 的 hidden states, 其中 $B$ 是 batch size, $L$ 是 sequence length, $d$ 是 model dimension (Qwen3-1.7B 上 $d = 2048$)。

**Step ❶ Sequence flip**:
$$\mathbf{h}_{\text{flip}} = \text{flip}(\mathbf{h}, \text{dim}=1)$$

将 sequence 维度 (dim=1) 反转。**这一步至关重要**: Mamba 是 left-to-right scan, 如果不 flip, 它就只是把 left context 摘要一遍——而 causal attention 已经做这件事了, 完全 redundant。Flip 之后, right-side tokens 先进入 scan, 形成 "left-to-right scan over the reversed sequence" = "right-to-left scan over the original sequence"。

**Step ❷ Selective scan**:
$$\mathbf{h}_{\text{mamba}} = \text{Mamba}(\mathbf{h}_{\text{flip}})$$

用 Mamba-1 selective-scan block (Gu & Dao 2024)。Mamba 的隐藏状态 $\mathbf{h}_t = \bar{A}\mathbf{h}_{t-1} + \bar{B}x_t$ 是一个 **position-aware summary of preceding tokens**, linear time, 状态大小 $d_{\text{state}} = 16$。这正是 $\Delta_i^{\text{R2L}}$ 期待的"compressed right context"。

为什么选 Mamba 而非其他选项?Paper 给了理由: reverse self-attention 也会破坏 KV cache; reverse RNN 容量太小; linear attention 不够 expressive。Mamba 的 selective scan 是 input-dependent ($\Delta_t, A_t, B_t, C_t$ 都依赖 $x_t$), 容量与 position-aware 兼顾。

**Step ❸ Unflip and normalize**:
$$\mathbf{h}_{\text{R2L}} = \text{LN}(\text{flip}(\mathbf{h}_{\text{mamba}}, \text{dim}=1))$$

再 flip 回来, 恢复原始 sequence order。这样 $\mathbf{h}_{\text{R2L}, i}$ (位置 $i$) 摘要的是 $j \geq i$ 的 tokens, 正好与 backbone 的 position $i$ 对齐。LayerNorm 是局部的, 作用是 **bound residual magnitude**, 防止破坏 backbone 的数值范围。

**Step ❹ Gated residual**:
$$\mathbf{h}^+ = \mathbf{h} + \mathbf{h}_{\text{R2L}} \cdot \tanh(s)$$

- $s \in \mathbb{R}$: 每个 hooked layer 一个 learnable scalar gate, **初始化为 0**
- $\tanh(s)$: 饱和函数, 把 contribution bound 在 $|\tanh(s)| \leq 1$

为什么用 scalar 而非 vector gate?Paper 说: 给一个 single trackable per-layer quantity, 便于观察训练动态。$\tanh$ saturation 思路直接借鉴 **LayerScale** (Touvron et al. 2021) 和 **ControlNet** (Zhang et al. 2023)。

**初始化时 $s = 0$ → $\tanh(0) = 0$ → residual = 0 → augmented model = causal model
