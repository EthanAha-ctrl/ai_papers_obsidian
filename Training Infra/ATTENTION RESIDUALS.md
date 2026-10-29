---
source_pdf: ATTENTION RESIDUALS.pdf
paper_sha256: 444673994328f7be8aee9d96fb240596b6f254f06ebaa53a2673413a244198c9
processed_at: '2026-07-18T10:48:38-07:00'
target_folder: LLM-Training/Training Infra
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Attention Residuals (AttnRes) — Kimi Technical Report 深度讲解

这篇 paper 来自 Kimi Team (Moonshot AI), repo: https://github.com/MoonshotAI/Attention-Residuals 。核心 idea 非常 elegant, 把 residual connection 重新理解为 **depth 维度的 recurrence**, 然后用 sequence 维度上 Transformer 解决 RNN bottleneck 的同样方法 — attention — 来解决 depth 维度上 standard residual 的 bottleneck。

---

## 1. Motivation: Standard Residual 是 Depth-wise Linear Attention

### 1.1 Residual 的两个角色

Residual connection (He et al., 2015, https://arxiv.org/abs/1512.03385) 通常被理解为 gradient highway:

$$h_l = h_{l-1} + f_{l-1}(h_{l-1})$$

- $h_l \in \mathbb{R}^d$: layer $l$ 的 input hidden state, $d$ 是 hidden dim
- $f_{l-1}$: layer $l-1$ 的 transformation (attention 或 MLP)
- $l \in \{1, \ldots, L\}$: layer index, $L$ 是总层数

但 paper 强调 residual 还有第二个被忽视的角色: **depth-wise information aggregation**。把 recurrence 展开:

$$h_l = h_1 + \sum_{i=1}^{l-1} f_i(h_i)$$

这里 $h_1$ 是 token embedding。每一层接收到的其实是 **所有前序层输出的等权求和 (uniform weighting with unit coefficients)**。这就是 paper 的核心观察: residual 定义了 depth 维度上信息如何聚合, 但这个聚合的权重是 fixed 的 1, 没有 selectivity。

### 1.2 PreNorm Dilution 问题

PreNorm (Xiong et al., 2020, https://arxiv.org/abs/2002.04745) 是现代 LLM 的标配, 但它有个 well-known 问题: hidden state magnitude 随 depth 增长为 $O(L)$。因为 $h_l = h_{l-1} + f_{l-1}(h_{l-1})$ 是加法累积, 而 PreNorm 把 $f$ 的 input normalize 到固定 scale, 所以深层 layer 的相对贡献被 dilute。深层 layer 为了保持 influence, 必须学越来越大的 output, 这会 destabilize training。这就是 Fig.5(b) 里 baseline output magnitude 单调增长的原因。

---

## 2. Time-Depth Duality: 从 RNN 到 Attention 的类比

### 2.1 RNN over Time 的瓶颈

RNN 把所有历史信息压缩进单个 state $h_t$:

$$h_t = f(h_{t-1}, x_t)$$

每个时间步只能 access 压缩后的 $h_{t-1}$, 无法 selectively retrieve 早期信息。Transformer (Vaswani et al., 2017, https://arxiv.org/abs/1706.03762) 用 attention 替换 recurrence, 让每个 position 直接 attend 到所有之前的 positions, 用 data-dependent weights 做 selective retrieval。

### 2.2 Residual over Depth 的对称问题

Standard residual 完全对称: 每个 layer 只能 access $h_{l-1}$ (compressed state), 无法 selectively retrieve 早期 layer 的 individual outputs。具体 limitations:

1. **No selective access**: attention layer 和 MLP layer 接收相同的 aggregated state, 但它们可能 benefit from 不同的 weighting
2. **Irreversible loss**: aggregation 丢失的信息无法在深层恢复
3. **Output growth**: 深层 layer 学更大 output 来 gain influence

### 2.3 Duality 的形式化

Paper 在 §6.1 建立了一个非常漂亮的 duality:

| Sequence 维度 (time) | Depth 维度 (layer) |
|---|---|
| RNN recurrence $h_t = f(h_{t-1}, x_t)$ | Residual $h_l = h_{l-1} + f_{l-1}(h_{l-1})$ |
| Highway network (gated recurrence) | Data-dependent gates (RetNet, GLA) |
| Linear attention $S_t = S_{t-1} + k_t v_t^\top$ | Standard residual (additive state) |
| Delta rule / DeltaNet | DDL (Deep Delta Learning) |
| GLA (Gated Linear Attention) | MRLA |
| TTT (Test-Time Training) | (gradient step view of residual) |
| **Transformer softmax attention** | **AttnRes (本文)** |

关键 insight: TTT (Sun et al., 2024, https://arxiv.org/abs/2407.04620) 把 RNN step 看作 gradient descent on self-supervised loss:

$$W_t = W_{t-1} - \eta \nabla \ell(W_{t-1}; x_t)$$

当 $f$ linear 时退化为 linear attention $S_t = S_{t-1} + k_t v_t^\top$。Standard residual 在 depth 维度有完全相同的 additive form。所以 AttnRes 就是把 depth-wise recurrence 升级为 depth-wise softmax attention, 正如 Transformer 把 sequence-wise recurrence 升级为 sequence-wise softmax attention。

---

## 3. Attention Residuals 公式详解

### 3.1 Full AttnRes (Eq.1, Eq.2, Eq.3, Eq.4)

**核心更新公式 (Eq.1):**

$$h_l = \alpha_{0l} \cdot h_1 + \sum_{i=1}^{l-1} \alpha_{il} \cdot f_i(h_i)$$

- $h_l$: layer $l$ 的 input
- $h_1$: token embedding (作为 source 0, 记作 $v_0$)
- $f_i(h_i)$: layer $i$ 的 output, 记作 $v_i$ (for $i \geq 1$)
- $\alpha_{il}$: layer $l$ 给 source $i$ 的 attention weight, 满足 $\sum_{i=0}^{l-1} \alpha_{il} = 1$

**Attention weight 计算 (Eq.2):**

$$\alpha_{il} = \frac{\phi(q_l, k_i)}{\sum_{j=0}^{l-1} \phi(q_l, k_j)}$$

- $\phi(q, k) = \exp(q^\top \text{RMSNorm}(k))$: kernel function, 用 RMSNorm (Zhang & Sennrich, 2019, https://arxiv.org/abs/1910.07467) 防止大 magnitude layer 主导 softmax
- $q_l = w_l \in \mathbb{R}^d$: **layer-specific learnable pseudo-query** — 这是关键设计, 是个 per-layer 的可学习参数, 与 input 和 hidden state 都 decouple
- $k_i = v_i$: key 和 value 都是 layer output 本身 (self-attention over depth)

**Query/Key/Value 定义 (Eq.3):**

$$q_l = w_l, \quad k_i = v_i = \begin{cases} h_1 & i = 0 \\ f_i(h_i) & 1 \leq i \leq l-1 \end{cases}$$

**最终 input (Eq.4):**

$$h_l = \sum_{i=0}^{l-1} \alpha_{il} \cdot v_i$$

### 3.2 关键 Design Choice: Pseudo-query $w_l$ 是 learned parameter

这是整个方法最巧妙的地方。$w_l$ **不是**从 hidden state 投影出来的, 而是 layer-specific 的 free parameter。这带来两个巨大好处:

1. **Parallelizable**: 因为 $w_l$ 不依赖 forward computation, 一个 block 内所有 layer 的 attention 可以并行计算 (batched matmul), 不用等 sequential outputs
2. **Inference-friendly**: decoding 时不需要 sequential memory access, 可以 amortize

Ablation (Table 4) 显示 input-dependent query (从 hidden state 投影) 能到 1.731 (比 learned query 的 1.737 更好), 但需要 $d \times d$ projection per layer 且 decoding 时 sequential, 所以默认用 learned query。

### 3.3 初始化: 必须 zero-init

**所有 $w_l$ 必须初始化为 0**。这样初始 $\phi(0, k) = \exp(0) = 1$ 对所有 source uniform, attention weights 退化为等权平均 $\alpha_{il} = 1/l$, 避免 training volatility。这是个 empirically validated 的关键 trick。

### 3.4 为什么用 softmax 而不是 sigmoid

Ablation (Table 4): sigmoid 给 1.741, softmax 给 1.737。原因是 softmax 的 **competitive normalization** 强制 sources 之间竞争 probability mass, 产生 sharper selection。Sigmoid 允许多个 source 同时高权重, 缺少 "选择" 的压力。

---

## 4. Block AttnRes: 工程化的折中

### 4.1 为什么需要 Block

Full AttnRes 要求每层 attend 所有前序 layer outputs:
- Memory: $O(Ld)$ per token (存所有 layer outputs)
- Compute: $O(L^2 d)$ per token

在 vanilla training 里, 这些 activations 本来就为 backprop 保留, 所以无额外开销。但在大规模训练中:
- **Activation recomputation**: 本来会 free 然后 recompute 的 activations 现在必须 keep alive
- **Pipeline parallelism**: 每个 layer output 都要跨 stage 传输, $O(Ld)$ communication

### 4.2 Block 结构 (Eq.5, Eq.6)

把 $L$ 层分成 $N$ 个 blocks, 每个 block $S = L/N$ 层。

**Intra-block accumulation (Eq.5):**

$$b_n = \sum_{j \in B_n} f_j(h_j)$$

- $B_n$: block $n$ 的 layer index 集合
- $b_n$: block $n$ 内所有 layer outputs 的求和 (standard residual aggregation within block)
- $b_n^i$: block $n$ 内前 $i$ 层的 partial sum, $b_n = b_n^S$

**Inter-block attention (Eq.6):**

Value matrix 对于 block $n$ 的第 $i$ 层:

$$V = \begin{cases} [b_0, b_1, \ldots, b_{n-1}]^\top & \text{if } i = 1 \text{ (block 首层)} \\ [b_0, b_1, \ldots, b_{n-1}, b_n^{i-1}]^\top & \text{if } i \geq 2 \text{ (后续层)} \end{cases}$$

- $b_0 = h_1$: token embedding 始终作为 source
- Block 首层 attend 之前所有完整 block 的 representations
- 后续层额外 attend 当前 block 的 evolving partial sum $b_n^{i-1}$

### 4.3 Block 数 N 的选择

- $N = L$: 退化为 Full AttnRes
- $N = 1$: 退化为 standard residual (只 embedding 隔离为 $b_0$)
- Empirically $N \approx 8$ recover most benefit (Fig.6)

Fig.6 的 block size sweep 很 informative: $S = 2, 4, 8$ 都在 1.746 附近, $S = 16, 32$ 退化向 baseline。说明只要 block 数足够 (~8), finer-grained blocking 边际收益递减。

### 4.4 PyTorch Pseudocode 解读 (Fig.2)

```python
def block_attn_res(blocks, partial_block, proj, norm):
    V = torch.stack(blocks + [partial_block])  # [N+1, B, T, D]
    K = norm(V)  # RMSNorm on keys
    logits = torch.einsum('d, nbtd -> nbt', proj.weight.squeeze(), K)
    # proj.weight 就是 w_l, squeeze 后是 [d]
    # einsum 计算 w_l^\top RMSNorm(v_i) for each source
    h = torch.einsum('nbt, nbtd -> btd', logits.softmax(0), V)
    # softmax over source dimension (dim 0), weighted sum
    return h
```

注意 `forward` 里每个 transformer block 调用两次 `block_attn_res`: 一次在 attention 前, 一次在 MLP 前, 各自有独立的 $w_l$ 和 RMSNorm。Block boundary 每 `block_size // 2` 个 transformer layer 触发一次 (因为每个 transformer layer = 1 attention + 1 MLP = 2 layers)。

---

## 5. Structured Matrix 视角 (§6.2) — Build Intuition 的关键

这是 paper 最深刻的 section。定义 **depth mixing matrix** $M \in \mathbb{R}^{L \times L}$, $M_{il}$ 是 layer $l$ 给 source $i$ 的权重:

$$h_l = \sum_{i=0}^{l-1} M_{il} v_i$$

不同 residual variant 的区别在于 $M$ 的结构 (semiseparable rank) 和 weight 是否 input-dependent。

### 5.1 Standard Residual: 全 1 下三角, rank-1 semiseparable

$$M = \begin{bmatrix} 1 & & & \\ 1 & 1 & & \\ \vdots & \vdots & \ddots & \\ 1 & 1 & \cdots & 1 \end{bmatrix}$$

展开 $h_l = h_{l-1} + f_{l-1}(h_{l-1})$ 得 $h_l = \sum_{i=0}^{l-1} v_i$, 所以 $M_{il} = 1$ for all $i < l$。这是 **all-ones lower triangular matrix**。

### 5.2 Highway: 1-semiseparable, input-dependent

$$h_l = (1-g_l) h_{l-1} + g_l f_{l-1}(h_{l-1})$$

定义 carry product $\gamma_{il}^\times = \prod_{j=i+1}^{l} (1-g_j)$:

- $M_{0l} = \gamma_{1l}^\times$ (embedding 的权重)
- $M_{il} = g_{i+1} \gamma_{i+1:l}^\times$ for $i \geq 1$

因为 cumulative product 通过 scalar gates 分解, $M$ 是 **1-semiseparable** (Dao & Gu, 2024, https://arxiv.org/abs/2405.21060), 和 standard residual 同 rank 但 input-dependent。权重和为 1, 所以 Highway 是 **stick-breaking attention** (Tan et al., 2025, https://openreview.net/forum?id=...) 的 depth-wise instance, 无显式 softmax。

### 5.3 (m)HC: m-semiseparable, state expansion

Hyper-Connections (Zhu et al., 2025, https://arxiv.org/abs/2409.19606) 和 mHC (Xie et al., 2026, https://arxiv.org/abs/2512.24880) 维持 $m$ 个 parallel streams $H_l \in \mathbb{R}^{d \times m}$:

$$H_l = H_{l-1} A_l + f_{l-1}(H_{l-1} \alpha_{l-1}) \beta_{l-1}^\top$$

- $A_l \in \mathbb{R}^{m \times m}$: learned transition matrix
- $\alpha_{l-1} \in \mathbb{R}^m$: mix streams 成 $f$ 的 single input
- $\beta_{l-1} \in \mathbb{R}^m$: distribute output 回 streams

展开得 effective weight (Eq.10):

$$M_{il} = \beta_i^\top A_{i+1:l}^\times \alpha_l$$

其中 $A_{i:j}^\times = \prod_{k=i+1}^{j} A_k$。$m \times m$ transitions 使 $M$ 是 **m-semiseparable**。这对应 sequence 维度的 **state expansion** (HGRN2, Qin et al., 2024, https://arxiv.org/abs/2404.07904), 把 recurrent state 从 $d$ 扩展到 $d \times m$。

### 5.4 (m)HC = Depth-wise Linear Attention

这是最关键的 insight。把 $M_{il} = \beta_i^\top A_{i+1:l}^\times \alpha_l$ 重新解读:

- $\alpha_l$: layer $l$ 发出的 **query**
- $\beta_i$: 总结 layer $i$ 贡献的 **key**
- $A_{i+1:l}^\times$: depth-relative positional operator, govern query-key interaction across intervening layers

当 $\phi(q, k) = \varphi(q)^\top \varphi(k)$ (feature map 分解), depth-wise attention collapse 成 recurrence — 这正是 MRLA-GLA 和 DDL-DeltaNet 对应的 structure。

所以: **(m)HC = depth-wise linear attention with matrix-valued states**, **AttnRes = depth-wise softmax attention**。这完整了 sequence-depth duality: linear attention (Katharopoulos et al., 2020, https://arxiv.org/abs/2006.16236) → softmax attention 的升级, 在 depth 维度重演。

### 5.5 Full vs Block AttnRes 的 rank

- Full AttnRes: $M_{il} = \alpha_{il}$, dense, rank-$L$
- Block AttnRes: completed block 内所有 sources 共享 block-level key/value $b_n$, 所以 $M_{il} = \alpha_{nl}$ for $i \in B_n$; current block 内每层多一个 distinct source (partial sum)。Effective rank 在 $N$ 和 $N+S$ 之间, interpolates standard residual ($N=1$) 和 Full ($N=L$)。

---

## 6. Infrastructure: 让 Block AttnRes 在 scale 上 practical

### 6.1 Cross-stage Caching (Eq.7, Eq.8)

Pipeline parallelism 用 interleaved schedule (Narayanan et al., 2021, https://arxiv.org/abs/2104.04473), $P$ physical stages, $V$ virtual stages per physical stage, $C = PV$ total chunks。

**Naive communication:**

$$Comm_{naive} = \sum_{j=1}^{C-1} j N_p d = \frac{C(C-1)}{2} N_p d$$

- $N_p$: 每个 physical stage 平均产生的 block representations 数
- $d$: hidden dim
- $j N_p$: 第 $j$ 个 chunk 累积的 blocks 数

**Cross-stage caching:** 每个 physical stage 处理多个 virtual stages, 之前 virtual stage 收到的 blocks 缓存在 local memory, 不用重传。

$$Comm_{cached} = \underbrace{\frac{P(P-1)}{2} N_p d}_{\text{first virtual stage}} + \underbrace{(V-1) P^2 N_p d}_{\text{subsequent virtual stages}}$$

- First virtual stage ($v=1$): 无 cache, 正常累积
- Subsequent ($v \geq 2$): 每次只传 $\sim P N_p$ incremental blocks

Peak per-transition cost 从 $O(C) = O(PV)$ 降到 $O(P)$, 提升 $V$ 倍。Fig.3 的例子 ($P=4, V=2$): 第二个 virtual stage caching 消除 6 次 redundant block 传输。

### 6.2 Two-phase Computation (Algorithm 1)

这是 inference 的核心优化。Pseudo-query $w_l$ decouple from forward computation, 所以一个 block 内所有 $S$ 层的 queries 可以 batch。

**Phase 1: Parallel inter-block attention**

```
Q ← [w_l]_{l ∈ B_n}           # [S, d], 所有 S 层的 pseudo-queries
K, V ← [b_0; ...; b_{n-1}]   # 之前所有 block representations
{o_l^(1), m_l^(1), ℓ_l^(1)} ← ATTENTION_WITH_STATS(Q, K, V)
```

- $o_l^{(1)}$: unnormalized output (sum of $\phi \cdot v$)
- $m_l^{(1)}$: max of logits (for online softmax)
- $\ell_l^{(1)}$: log-sum-exp of logits (normalization denominator)

把 S 次 read 摊销为 1 次, 大幅减少 memory access。

**Phase 2: Sequential intra-block attention + online softmax merge**

```
for l in B_n:
    if i == 0:
        h_l = o_l^(1) / ℓ_l^(1)     # 只有 inter-block
    else:
        o_l^(2), m_l^(2), ℓ_l^(2) ← ATTENTION(w_l, b_n^i, b_n^i)  # intra-block
        m_l = max(m_l^(1), m_l^(2))
        h_l ∝ e^{m_l^(1) - m_l} o_l^(1) + e^{m_l^(2) - m_l} o_l^(2)
        denominator = e^{m_l^(1) - m_l} ℓ_l^(1) + e^{m_l^(2) - m_l} ℓ_l^(2)
    b_n^i = b_n^{i-1} + f_l(h_l)    # 更新 partial sum
```

Online softmax (Milakov & Gimelshein, 2018, https://arxiv.org/abs/1805.02867) merge 公式:

$$h_l = \frac{e^{m_l^{(1)} - \bar{m}_l} o_l^{(1)} + e^{m_l^{(2)} - \bar{m}_l} o_l^{(2)}}{e^{m_l^{(1)} - \bar{m}_l} \ell_l^{(1)} + e^{m_l^{(2)} - \bar{m}_l} \ell_l^{(2)}}$$

- $\bar{m}_l = \max(m_l^{(1)}, m_l^{(2)})$: 全局 max, 保证数值稳定
- 分子: 两 phase 的 unnormalized outputs 加权求和
- 分母: 两 phase 的 normalization denominators 加权求和

这保证 exact equivalence with 一次性算完整 attention。Phase 2 是 elementwise, 可以 kernel fusion with surrounding ops (RMSNorm 等)。

### 6.3 Memory Access Cost 对比 (Table 1)

| Method | Read | Write | Total I/O (symbolic) | Typical (L=128, N=8, S=16, m=4) |
|---|---|---|---|---|
| Standard Residual | 2d | d | 3d | 3d |
| mHC (m=4 streams) | — | — | $(8m+2)d + 2m^2 + 4m$ | 34d |
| Full AttnRes (two-phase) | $(S+N-2)d$ | 2d | $(S+N)d$ | 24d |
| **Block AttnRes (two-phase)** | $(\frac{N}{S}+3)d$ | 2d | $(\frac{N}{S}+5)d$ | **5.5d** |

Block AttnRes 的 5.5d vs mHC 的 34d, 几乎 6× 更少 I/O, 同时 loss 更好。这是 paper 的工程卖点。

### 6.4 Appendix B: Full AttnRes 的 two-phase I/O 推导

对 Full AttnRes, partition 纯粹是 inference scheduling device, 不改架构。

**Phase 1 inter-block reads (Eq.11, 12):**

$$Read_{inter}^{(n)} = 2(n-1)Sd \quad \text{(block n, factor 2 for K and V)}$$

$$Read_{inter} = \sum_{n=1}^{N} 2(n-1)Sd = 2Sd \cdot \frac{N(N-1)}{2} = dL(N-1)$$

**Phase 1 writes (Eq.13):** $Write_{inter} = Ld$ (每层一个 d-dim output)

**Phase 2 intra-block reads (Eq.14):**

$$Read_{intra}^{(n)} = \sum_{t=1}^{S} 2(t-1)d = S(S-1)d$$

**Total per layer (Eq.16, 17):**

$$Read_{per\ layer} = (N-1)d + (S-1)d = (S+N-2)d$$

$$Write_{per\ layer} = 2d$$

$$\boxed{Total\ I/O\ per\ layer = (S+N)d}$$

从 $O(L)$ 降到 $O(S+N) = O(L/N + N)$, 当 $N \approx \sqrt{L}$ 时最优, 但实践中 $N \approx 8$ 已足够。

---

## 7. 实验: Scaling Laws + 48B 模型

### 7.1 Scaling Laws (Table 2, Fig.4)

5 个 model sizes (194M – 528M activated params), 每个 size 训 3 个 variant。Power law 拟合 $\mathcal{L} = A \times C^{-\alpha}$:

| Variant | Fitted curve |
|---|---|
| Baseline | $\mathcal{L} = 1.891 \times C^{-0.057}$ |
| Block AttnRes | $\mathcal{L} = 1.870 \times C^{-0.058}$ |
| Full AttnRes | $\mathcal{L} = 1.865 \times C^{-0.057}$ |

- 三者 slope 相似, 但 AttnRes 一致更低 loss
- 5.6 PFLOP/s-days 处: Block AttnRes 1.692 vs Baseline 1.714, 等效 **1.25× compute advantage**
- Full vs Block 差距随 scale 缩小, 最大 size 仅 0.001

Table 2 还列了 mHC-lite (Yang & Gao, 2026, https://arxiv.org/abs/2601.05732): Full AttnRes 优于 mHC, Block AttnRes 在更低 I/O (5.5d vs 34d) 下 match mHC。

### 7.2 48B Kimi Linear 主实验 (Table 3)

基于 Kimi Linear (Zhang et al., 2025, https://arxiv.org/abs/2510.26692): 48B total / 3B activated, 27 transformer blocks (54 layers), 8/256 routed experts + 1 shared expert, hybrid KDA/MLA attention。Block AttnRes with 6 layers/block → 9 blocks + embedding = 10 sources。

训练 1.4T tokens (1T WSD pre-training + 400B mid-training annealing), 然后扩展到 32K context。因为 MLA 用 NoPE (Yang et al., 2025, https://arxiv.org/abs/2501.18795), context extension 不需要 YaRN 或 attention rescaling。

| Benchmark | Baseline | AttnRes | Δ |
|---|---|---|---|
| MMLU | 73.5 | 74.6 | +1.1 |
| GPQA-Diamond | 36.9 | 44.4 | **+7.5** |
| BBH | 76.3 | 78.0 | +1.7 |
| Math | 53.5 | 57.1 | **+3.6** |
| HumanEval | 59.1 | 62.2 | **+3.1** |
| C-Eval | 79.6 | 82.5 | +2.9 |

GPQA-Diamond +7.5 和 Math +3.6 这种 multi-step reasoning 任务提升最大, 符合 hypothesis: improved depth-wise information flow benefits compositional tasks, 后层可以 selectively retrieve 早期 representations 来 build upon。

### 7.3 Training Dynamics (Fig.5)

三个 panel 揭示 AttnRes 如何改变训练:

**(a) Validation loss:** AttnRes 全程更低, decay phase 差距拉大。

**(b) Output magnitude per block:** 
- Baseline: 单调增长, 典型 PreNorm dilution, 深层 layer 必须学更大 output
- AttnRes: **bounded periodic pattern** — selective aggregation 在 block boundary reset accumulation, magnitude 不跨 block 累积

**(c) Gradient magnitude per block:**
- Baseline: 早期层梯度 disproportionately large, 因为 residual weights 全 1 无法 regulate gradient flow
- AttnRes: **更均匀的 gradient distribution** — learnable softmax weights 引入 sources 间竞争, 自动 balance gradient

### 7.4 Architecture Sweep (Fig.7)

固定 compute ($6.5 \times 10^{19}$ FLOPs) 和 active params ($2.3 \times 10^8$), 枚举 25 个 config 在 $d_{model}/L_b \times H/L_b$ grid 上。

- 两者都在 $H/L_b \approx 0.3$ 最优
- **Baseline 最优 $d_{model}/L_b \approx 60$**, **AttnRes 最优 $\approx 45$**
- 更低 $d_{model}/L_b$ = 更深更窄, 说明 AttnRes 能更有效利用 depth
- AttnRes 在全部 25 个 config 都优于 baseline (0.019–0.063)

但作者 caveat: deeper models 推理 latency 更高 (sequential computation, Pope et al., 2022, https://arxiv.org/abs/2211.05102), 这个 depth preference 不直接等于 deployment recommendation, 而是 diagnostic。

### 7.5 Learned Attention Patterns (Fig.8)

16-head model 的 depth-wise attention weights $\alpha_{i \to l}$ heatmap:

1. **Diagonal dominance**: locality 是主信息路径, 每层最强 attend 前一层
2. **Embedding persistence**: $h_1$ (source 0) 全程保持 non-trivial weight, 尤其是 pre-attention layers — 这是 **depth-wise attention sink**, mirror sequence-wise attention sink (Xiao et al., 2023, https://arxiv.org/abs/2309.17453)
3. **Layer specialization**: pre-MLP inputs 有 sharper diagonal (local), pre-attention inputs 有 broader receptive field — 因为 attention routing 信息跨层, MLP 局部操作
4. **Learned skip connections**: off-diagonal concentrations, e.g. layer 4 attend 早期 sources, layers 15-16 reach back
5. **Block 保留 structure**: diagonal dominance, embedding persistence, specialization 都 transfer 到 block variant, 说明 block-wise compression 像 implicit regularization

---

## 8. Ablation 逻辑链 (Table 4) — Build Intuition

16-layer model, 逐个验证 design choice:

| Variant | Loss | Insight |
|---|---|---|
| Baseline (PreNorm) | 1.766 | — |
| DenseFormer | 1.767 | input-independent scalar coefficients 无提升, **input-dependent weighting 必要** |
| mHC | 1.747 | input-dependent via m streams + mixing matrices, 有效但 linear attention |
| **AttnRes Full** | **1.737** | softmax attention > linear attention, explicit content-dependent selection |
| w/ input-dependent query | 1.731 | 更好但 $d \times d$ projection + sequential decoding, 不值得 |
| w/ input-independent mixing | 1.749 | 移除 query/key, 用 learnable scalars, 显著退化 — **content-dependence 关键** |
| w/ sigmoid | 1.741 | softmax 的 competitive normalization 更好 |
| w/o RMSNorm | 1.743 | RMSNorm 防止大 magnitude layer 主导 |
| SWA (W=1+8) | 1.764 | sliding window 只看最近 8 层, 远不如 Block — **distant layers 重要** |
| Block (S=4) | 1.746 | block 结构几乎不损性能 |
| w/ multihead (H=16) | 1.752 | per-head depth aggregation 反而差 — **depth mixture 应 channel-uniform** |
| Block w/o RMSNorm | 1.750 | Block 更需要 RMSNorm, 因为 block representations 累积更多层, magnitude 差异更大 |

关键 takeaways:
1. **Input-dependence 是必须的** (DenseFormer 失败, input-independent mixing 退化)
2. **Softmax > sigmoid > linear** (竞争性归一化重要)
3. **Distant layers 重要** (SWA 远不如 Block)
4. **Depth mixture channel-uniform** (multihead 有害 — layer output 要么整体 relevant 要么不 relevant)
5. **RMSNorm on keys 关键** (防止 magnitude bias)

---

## 9. Related Work 的全景 (Table 5)

Paper 把所有 residual variants 分成三类:

### 9.1 Single-state recurrence (layer l 只见 $h_{l-1}$)
- Residual, ReZero (Bachlechner et al., 2020, https://arxiv.org/abs/2003.04887), LayerScale (Touvron et al., 2021, https://arxiv.org/abs/2103.17239), Highway (Srivastava et al., 2015, https://arxiv.org/abs/1505.00387), DeepNorm (Wang et al., 2022, https://arxiv.org/abs/2203.00555), KEEL (Chen & Wei, 2026, https://arxiv.org/abs/2601.19895)

### 9.2 Multi-state recurrence (m streams)
- SiameseNorm (Li et al., 2026, https://arxiv.org/abs/2602.08064): 2 streams, 一个 PreNorm 一个 PostNorm
- HC/mHC: m streams with learned mixing
- DDL (Zhang et al., 2026, https://arxiv.org/abs/2601.00417): matrix state + delta rule

### 9.3 Cross-layer access (layer l 可见 individual earlier outputs)
- DenseNet (Huang et al., 2018, https://arxiv.org/abs/1608.06993): concatenate all
- ELMo (Peters et al., 2018, https://aclanthology.org/N18-1202/): softmax scalar weights
- DenseFormer (Pagliardini et al., 2024, https://arxiv.org/abs/2402.02622): learned scalar coefficients
- MRLA (Fang et al., 2023, https://arxiv.org/abs/2302.03985): element-wise sigmoid, 但 separable query-key product 更接近 linear attention
- MUDDFormer (Xiao et al., 2025, ICML): position-dependent weights via small MLP, 4 decoupled streams
- Value Residual Learning (Zhou et al., 2025, https://aclanthology.org/2025.acl-long.1375/): 只 access 单个早期层
- LAuReL (Menghani et al., 2025, https://arxiv.org/abs/2411.07501): low-rank projections over previous k activations

AttnRes 的独特组合: **softmax-normalized + input-dependent + selective access to all preceding layers + 单个 d-dim pseudo-query per layer + block structure** ($O(L^2) \to O(LN)$)。

---

## 10. 总结: 为什么 AttnRes work

回到 build intuition 的核心:

1. **Standard residual 是 depth-wise linear attention with fixed unit weights**: 信息聚合无 selectivity, magnitude 无 control, gradient flow 无 regulation

2. **AttnRes 升级到 depth-wise softmax attention**: 每层可以 content-dependently 选择哪些早期 representations 更 relevant, softmax 的 competitive normalization 自动 balance magnitude 和 gradient

3. **Pseudo-query $w_l$ 的 decoupling 是工程关键**: 允许 parallel/batched computation, two-phase inference schedule, 把 $O(L)$ I/O 降到 $O(S+N)$

4. **Block 结构是 practical sweet spot**: $N \approx 8$ blocks recover most of Full AttnRes gain, memory/communication 从 $O(Ld)$ 降到 $O(Nd)$, cross-stage caching 进一步消除 pipeline redundancy

5. **PreNorm dilution 被 naturally 解决**: 不再是 unbounded additive accumulation, 而是 selective aggregation, magnitude 在 block boundary reset, gradient 更均匀

6. **Depth-wise attention sinks 出现**: embedding $h_1$ 持续被 attend, mirror sequence-wise attention sinks, 暗示 depth 维度也有类似的 "default retrieval" 机制

这个 work 的 beauty 在于: 它没有发明全新的 mechanism, 而是识别出 residual connection 和 RNN 的 duality, 然后把 sequence 维度上已经被验证的 attention 解决方案 完整地移植到 depth 维度。Structured matrix 视角 (§6.2) 把这个 duality 形式化: standard residual / Highway / mHC 都是 depth-wise linear attention 的 instance (不同 semiseparable rank), AttnRes 是 depth-wise softmax attention, 完成了 linear → softmax 的升级 — 正是 Transformer 当年对 RNN 做的事。

参考链接:
- Paper repo: https://github.com/MoonshotAI/Attention-Residuals
- Kimi Linear: https://arxiv.org/abs/2510.26692
- Hyper-Connections: https://arxiv.org/abs/2409.19606
- mHC: https://arxiv.org/abs/2512.24880
- DenseFormer: https://arxiv.org/abs/2402.02622
- TTT: https://arxiv.org/abs/2407.04620
- Online softmax: https://arxiv.org/abs/1805.02867
- Residual (He et al.): https://arxiv.org/abs/1512.03385
- Attention is All You Need: https://arxiv.org/abs/1706.03762
- PreNorm: https://arxiv.org/abs/2002.04745
- Linear attention: https://arxiv.org/abs/2006.16236
- Structured matrices (Mamba-2): https://arxiv.org/abs/2405.21060
- Attention sinks: https://arxiv.org/abs/2309.17453
