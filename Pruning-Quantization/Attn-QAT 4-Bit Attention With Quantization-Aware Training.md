---
source_pdf: Attn-QAT 4-Bit Attention With Quantization-Aware Training.pdf
paper_sha256: 32fdbd6fa5f60c94221c75e8ff39b2154ea5305508eca0adf56161de0dd5aaaa
processed_at: '2026-07-18T11:26:00-07:00'
target_folder: Pruning-Quantization
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Attn-QAT 深度解读：让 4-bit Attention 真正可用

Andrej, 这篇 paper 是 PyTorch / LMsys / NVIDIA 圈子近期关于 **4-bit attention QAT** 的第一个系统性工作。它的核心 contribution 不在于提出了什么花哨的 outlier suppression 算法，而在于揭露了一个被长期忽视的事实：**FlashAttention 的 backward pass 中隐藏着对 forward precision 的隐式假设**，naive 的 QAT (FP4 forward + BF16 backward) 会 break 这些假设从而导致 exploding gradients。我会深入展开每个公式和 algorithm 的细节，并试图 build 你的 intuition。

---

## 1. Motivation：为什么 attention 是 4-bit quantization 的最后一块硬骨头

### 1.1 NVFP4 format 的硬约束

NVIDIA Blackwell (SM100/SM120) 引入了 native **NVFP4 tensor core**，arithmetic intensity 相对 FP8 翻倍，memory traffic 也降低。NVFP4 是 OCP Microscaling (MX) format 的变体：

- OCP MXFP4: block size = 32, scale factor 是 E8M0 (8-bit exponent only)
- NVIDIA NVFP4: block size = **16**, scale factor 是 **E4M3** (更细粒度)

每个 FP4 element 是 **e2m1** 格式：2 bit exponent + 1 bit mantissa + 1 sign bit，所以只有 **15 个 distinct representable values**（±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6, ±inf/nan 占位）。Dynamic range 极窄。

**Quantization 公式（Eq. 1, 2）解析**：

$$
\phi(\mathbf{X}): \quad s_{ij} = \frac{\max(|\mathbf{X}_{ij}|)}{6}, \quad \hat{\mathbf{X}}_{ij} = \left\lceil \frac{\mathbf{X}_{ij}}{s_{ij}} \right\rfloor
$$

变量含义：
- $\mathbf{X} \in \mathbb{R}^{N \times d}$: 待量化的 tensor（如 Q, K, V 或 P）
- $ij$ 下标: block 索引，每个 block 16 个元素，$X_{ij} \in \mathbb{R}^{1 \times 16}$
- $s_{ij}$: 该 block 的 scale factor (E4M3 格式)
- 数字 **6**: FP4 e2m1 的最大 finite value，确保 $\max(|X_{ij}|)$ 刚好映射到 FP4 的 max
- $\lceil \cdot \rfloor$: round-to-nearest 到 FP4-representable grid
- $\hat{\mathbf{X}}_{ij}$: 量化后的 FP4 tensor

Dequant 就是反向：$\mathbf{X}'_{ij} = s_{ij} \cdot \hat{\mathbf{X}}_{ij}$。

**Eq. 3 的 FP4MM** 是 Blackwell 的 hardware primitive：

$$
\mathbf{C} = \mathrm{FP4MM}(\mathbf{A}, \hat{s}_A, \mathbf{B}, \hat{s}_B)
$$

数学上等价于 $\mathbf{C} = \mathrm{BF16MM}(\phi^{-1}(\phi(\mathbf{A})), \phi^{-1}(\phi(\mathbf{B})))$，但 hardware 上一次 cycle 完成 dequant + matmul + scale 应用。

### 1.2 Attention 为何对 4-bit 敏感

这里要 build 的 intuition 是：**attention 的数值分布与 linear layer 本质不同**。

| Property | Linear Layer (W) | Attention (QK^T, PV) |
|----------|------------------|----------------------|
| Distribution | Gaussian-ish, well-behaved | Heavy-tailed, with structured outliers |
| Range utilization | Full | Sparse (P ∈ [0,1], 大部分接近 0) |
| Calibration data | Static (weight) | Dynamic (per-token activation) |
| Outlier mechanism | Persistent channel outliers | Position-dependent, attention-pattern driven |

SageAttention3 (Zhang et al., 2025, https://arxiv.org/abs/2505.11594) 是 training-free SOTA，它引入了两个 heuristic：

**(a) Q/K Smoothing (Eq. 4)**：
$$
\gamma(\mathbf{Q}_i) = \mathbf{Q}_i - \bar{\mathbf{q}}_i, \quad \gamma(\mathbf{K}_j) = \mathbf{K}_j - \bar{\mathbf{k}}
$$
其中 $\bar{\mathbf{q}}_i = \mathrm{mean}(\mathbf{Q}_i)$, $\bar{\mathbf{k}} = \mathrm{mean}(\mathbf{K})$。这把 $\mathbf{QK}^\top$ 分解为：

$$
\mathbf{S}_{ij} = \gamma(\mathbf{Q}_i)\gamma(\mathbf{K}_j)^\top + \underbrace{\Delta \mathbf{S}_{ij}}_{\bar{\mathbf{q}}_i \gamma(\mathbf{K}_j)^\top} + \underbrace{\mathbf{b}}_{\bar{\mathbf{q}}_i \bar{\mathbf{k}}^\top + \gamma(\mathbf{Q}_i) \bar{\mathbf{k}}^\top}
$$

直觉：减去 mean 让 $\gamma(Q), \gamma(K)$ 的 range 缩小，从而 NVFP4 quantization 误差相对变小。$\Delta S$ 和 $b$ 用高精度补。

**(b) Two-level quantization of P**：P ∈ [0,1] 时 NVFP4 几乎全部映射到 {0, 0.5, 1}，浪费精度。SageAttention3 把每行 rescale 到 $[0, 448 \times 6] = [0, 2688]$（448 是 E4M3 scale max，6 是 FP4e2m1 max），再用 standard FP4 quantize。

但 paper Figure 1 显示：在 Wan 2.1 14B 上，SageAttention3 仍然有显著 quality drop（temporal flickering, dynamic degree 下降严重）。原因——这些 heuristic 都是 **post-hoc** 的，无法对 attention 内部的 dynamic pattern 做完全 compensation。这就 motivation 了 QAT。

---

## 2. 核心 Insight：FlashAttention Backward 的隐式精度假设

这是 paper 最有价值的部分。我先把 FA 的 backward 数学拆开。

### 2.1 标准 FA backward 回顾

给定 $\mathbf{O}_i = \sum_j \mathbf{P}_{ij} \mathbf{V}_j$，其中 $\mathbf{P}_i = \mathrm{softmax}(\mathbf{S}_i)$，$\mathbf{S}_i = \mathbf{Q}_i \mathbf{K}^\top / \sqrt{d}$。

Softmax 的 Jacobian（Eq. 8）：
$$
\mathbf{dS}_i = (\mathrm{diag}(\mathbf{P}_i) - \mathbf{P}_i \mathbf{P}_i^\top) \mathbf{dP}_i = \mathbf{P}_i \odot \mathbf{dP}_i - (\mathbf{P}_i^\top \mathbf{dP}_i) \mathbf{P}_i
$$

这里 $\mathbf{P}_i^\top \mathbf{dP}_i$ 是一个**标量**，但是它需要 full row of $\mathbf{P}_i$，导致 FA 的 backward memory 复杂度从 linear 退化成 quadratic。

FA 的 trick 是用 identity（Eq. 9）：

$$
\mathbf{P}_i^\top \mathbf{dP}_i = \sum_j \mathbf{P}_{ij} \mathbf{dO}_i^\top \mathbf{V}_j = \mathbf{dO}_i^\top \underbrace{\sum_j \mathbf{P}_{ij} \mathbf{V}_j}_{= \mathbf{O}_i} = \mathbf{dO}_i^\top \mathbf{O}_i
$$

**这个 identity 的关键**：等式右边 $\mathbf{O}_i$ 是 forward pass 已经计算并保存的 attention output。所以 FA backward 只需要存 $\mathbf{O}_i$（线性 memory）而非 full $\mathbf{P}$ 矩阵（二次 memory）。

### 2.2 QAT 下 identity 被 break 的原因

现在 forward 是 FP4：
- Forward 中实际计算的是 $\mathbf{O}_i = \sum_j \mathbf{P}_{ij}^F \mathbf{V}_j^F$，其中 $\mathbf{P}^F = \phi^{-1}(\phi(\mathbf{P}))$ 是 fake-quantized。
- 但 identity (Eq. 9) 的中间步骤 $\sum_j \mathbf{P}_{ij} \mathbf{V}_j$ 要求 $\mathbf{P}_{ij}$ 是**高精度的 P**，而不是 $\mathbf{P}_{ij}^F$。

这就是 paper 说的 "implicit precision assumption"：FA 假设 forward 和 backward 共享同一精度，所以 $\mathbf{O}_i$ 同时满足 forward 输出和 backward identity 的两个角色。QAT 中两者精度分裂，identity 不再成立。

**Intuition**：你可以把 $\mathbf{O}_i$ 想成两个 "logical roles"：
1. **作为 forward 的输出**：被 next layer 消费，应该 reflect FP4 误差
2. **作为 backward 的 cache**：用于重建 $\mathbf{P}^\top \mathbf{dP}$，应该匹配 backward 中 $\mathbf{dP} = \mathbf{dO}_i (\mathbf{V}_j^F)^\top$ 的精度（这里 $\mathbf{V}^F$ 是 fake-quantized 的）

如果用 role 1 的 $\mathbf{O}_i$（低精度）做 role 2 的事情，会引入 $\mathbf{P}^F$ vs $\mathbf{P}$ 的 mismatch，gradient 出现系统性 bias，最终 exploding gradients。

### 2.3 解决方案：双输出 $\mathbf{O}$ 和 $\mathbf{O}'$

paper 的方案非常 elegant：forward 中**同时计算两个 output**：

- $\mathbf{O}_i = \sum_j \mathbf{P}_{ij}^F \mathbf{V}_j^F$ — 用于 next layer（低精度，QAT 一致性）
- $\mathbf{O}'_i = \sum_j \mathbf{P}_{ij} \mathbf{V}_j^F$ — **仅用于 backward 中的 $\mathbf{dO}_i^\top \mathbf{O}'_i$ 标量项**（高精度 P，低精度 V）

注意 $\mathbf{O}'$ 用 $\mathbf{V}^F$ 而不是 $\mathbf{V}$，这是因为 backward 的 $\mathbf{dP}$ 公式（Eq. 7）：
$$
\mathbf{dP} = \mathbf{dO}_i (\mathbf{V}_j^F)^\top
$$
本来就基于 $\mathbf{V}^F$。所以 $\mathbf{O}'$ 的精度要匹配 $\mathbf{V}^F$，但 $\mathbf{P}$ 部分要高精度。

### 2.4 第二个 issue：backward 中 P 的 recompute precision

FA 的 forward 不存 P，backward 从 log-sum-exp $\mathbf{L}_i = \mathbf{m}_i + \log(\mathbf{l}_i)$ recompute：
$$
\mathbf{P}_i = \exp(\mathbf{S}_i - \mathbf{L}_i)
$$

**问题**：在 QAT forward 中，$\mathbf{P}$ 经过了 fake quantization $\mathbf{P}^F = \phi^{-1}(\phi(\mathbf{P}))$，然后才用于 $\mathbf{O} = \mathbf{P}^F \mathbf{V}^F$。如果 backward 直接 recompute $\mathbf{P}$ 然后用高精度 $\mathbf{P}$ 算 $\mathbf{dP}, \mathbf{dV}$，那么：
- Forward 的 effective computational graph 是 $\mathbf{O} = \phi^{-1}(\phi(\mathbf{P})) \cdot \mathbf{V}^F$
- Backward recompute 出来的却是高精度 $\mathbf{P}$
- **STE 假设被破坏**：STE 要求 backward 的 gradient 对应 forward 的 fake-quantized activation

paper 在 Algorithm 3 line 11 显式 fake-quantize recompute 出来的 $\mathbf{P}$：
$$
\mathbf{P}^F \leftarrow \phi^{-1}(\phi(\mathbf{P}))
$$
然后 $\mathbf{dV}_j \mathrel{+}= (\mathbf{P}^F)^\top \mathbf{dO}_i$。

**Ablation 结果（Table 2 Exp 8）**：去掉这个 fake quantization，最终 VBench 分数差不多（0.8254 vs 0.8252），但是 Figure 3a 显示 **gradient norm 明显更 noisy**。这个 ablation 很有意思——说明这个 fix 主要影响的是 optimization stability 而非 final quality，类似于 "训练能收敛但崎岖"。

---

## 3. Algorithm 详解

### 3.1 Algorithm 1: Inference Forward

```
输入: Q ∈ R^{N_q × d}, K, V ∈ R^{N_k × d}, tile sizes B_q, B_k
1. 把 Q, K, V 量化为 NVFP4: (Q̂, ŝ_Q), (K̂, ŝ_K), (V̂, ŝ_V)
2. Tile 化: {Q_i}, {K_j, V_j}
3. for each Q tile i:
   for each K, V tile j:
     S = FP4MM(Q̂_i, ŝ_Q, K̂_j, ŝ_K) / √d     # FP4 score
     m_new = max(m_i, rowmax(S))
     α = exp(m_i - m_new)                      # rescale factor
     P̃ = exp(S - m_new)                        # unnormalized P
     l_i = α ⊙ l_i + rowsum(P̃)                # normalizer
     (P̃̂, ŝ_P̃) = φ(P̃)                          # 量化 P
     O_i = diag(α) O_i + FP4MM(P̃̂, ŝ_P̃, V̂_j, ŝ_V)   # FP4 aggregation
   O_i = diag(l_i)^{-1} O_i
   L_i = m + log(l_i)
```

注意两个细节：
- **P 在量化之前是 FP32**（line 10），因为 softmax numerics 需要 FP32。量化发生在 P̃ 之后，aggregation 之前。
- L (log-sum-exp) 用 FP32 存，用于 backward recompute。

### 3.2 Algorithm 2: Training Forward

```
1. Q^F = φ^{-1}(φ(Q)), K^F = ..., V^F = ...    # fake quantization
2. for each tile:
   for each j:
     S = Q_i^F (K_j^F)^T / √d                    # BF16 matmul (emulating FP4)
     ... standard online softmax ...
     P̃^F = φ^{-1}(φ(P̃))                          # fake quant P
     O_i = diag(α) O_i + P̃^F V_j^F               # low-precision output
     O'_i = diag(α) O'_i + P̃ V_j^F               # HIGH-precision P, fake-quant V
3. Return O, L, O'
```

关键差异在 **line 13**：$\mathbf{O}'_i$ 用未量化的 $\tilde{\mathbf{P}}$（FP32）和 fake-quantized $\mathbf{V}^F$（BF16）。这就是 paper 揭示的核心 trick。

**Memory cost**：多了一个 $\mathbf{O}' \in \mathbb{R}^{N_q \times d}$ 的 buffer。对于 head_dim=128, seq_len=8192, 这个是 4MB per head per layer。在 Appendix B.1 中作者说需要 full gradient checkpointing 来避免 OOM。

### 3.3 Algorithm 3: Backward

```
1. 输入: Q^F, K^F, V^F, dO, L, O' (high-precision)
2. D = rowsum(dO ⊙ O')                          # 关键: 用 O' 不是 O
3. for j = 1 to T_k:
   for i = 1 to T_q:
     S = Q_i^F (K_j^F)^T / √d                    # recompute score
     P = exp(S - L_i)                            # recompute P (FP32)
     P^F = φ^{-1}(φ(P))                          # fake quant P
     dV_j += (P^F)^T dO_i                        # gradient on V
     dP = dO_i (V_j^F)^T                         # gradient on P (uses V^F)
     dS = P ⊙ (dP - D_i) / √d                    # softmax backward
     dQ_i += dS K_j^F
     dK_j += dS^T Q_i^F
```

仔细看 line 3 和 line 14：
- Line 3: $\mathbf{D} = \mathrm{rowsum}(\mathbf{dO} \odot \mathbf{O}')$ — 这等价于 $\mathbf{P}_i^\top \mathbf{dP}_i$ 的 row-wise scalar，用高精度 $\mathbf{O}'$
- Line 14: $\mathbf{dS} = \mathbf{P} \odot (\mathbf{dP} - \mathbf{D}_i) / \sqrt{d}$ — softmax Jacobian，注意这里用的是**高精度 P**（不是 $\mathbf{P}^F$），因为 softmax 本身应该在高精度下进行（standard FA 也是这样）

**一个 subtle 点**：line 11 fake-quant $\mathbf{P}$ 得到 $\mathbf{P}^F$ 用于 $\mathbf{dV}$，但 line 14 的 $\mathbf{dS}$ 用高精度 $\mathbf{P}$。这是因为：
- $\mathbf{dV}$ 是关于 $\mathbf{V}$ 的梯度，对应 forward 的 $\mathbf{O} = \mathbf{P}^F \mathbf{V}^F$，所以梯度需要用 $\mathbf{P}^F$ 保持一致
- $\mathbf{dS}$ 是 softmax 内部的梯度，softmax 在 FP32 下计算（forward 也是 FP32），所以这里保持 FP32

---

## 4. 实验数据深入分析

### 4.1 Diffusion 实验（Table 1, 2）

**Wan 2.1 14B（Table 1）**：

| Exp | Method | Overall Quality | Dynamic Degree | Temporal Flickering |
|-----|--------|-----------------|----------------|---------------------|
| 1 | BF16 | 0.8335 | 0.5193 | 0.9759 |
| 2 | FP4 (naive) | 0.7968 | 0.2983 | 0.9783 |
| 3 | SageAttention3 | 0.8203 | 0.4751 | 0.9758 |
| 4 | Attn-QAT | 0.8279 | 0.3646 | 0.9828 |

观察：
- **Dynamic Degree** 这个指标对 motion magnitude 敏感，FP4 掉得最厉害（0.5193 → 0.2983）。Attn-QAT 只恢复到 0.3646，没完全 match BF16。这暗示 FP4 在捕捉 video 中大动态运动时仍有 residual error。
- **Temporal Flickering** 上 Attn-QAT (0.9828) 反而**超过** BF16 (0.9759)。可能原因：FP4 的"平滑效应"（quantization 阻尼高频 noise）让 temporal 维度更稳定，类似一种 implicit regularization。
- SageAttention3 vs Attn-QAT 在 14B 上差距不大（0.8203 vs 0.8279），但在 1.3B（Table 2）上 SageAttention3 反而**更差**（0.7834 vs 0.8252），这暗示小模型对 outlier mitigation 的依赖更强，而 QAT 在小模型上效果反而更明显。

**Wan 2.1 1.3B Ablation（Table 2）**：

| Exp | Variant | Overall Quality |
|-----|---------|-----------------|
| 4 | Attn-QAT | 0.8252 |
| 5 | + SmoothK | 0.8232 |
| 6 | + Two-level quant P | 0.8257 |
| 7 | - High prec. O in BWD | **0.7185** ← 崩溃 |
| 8 | - Fake quant P in BWD | 0.8254 |

Exp 7 的崩溃验证了 paper 的核心理论：缺少高精度 $\mathbf{O}'$ 时，softmax backward 的 $\mathbf{P}^\top \mathbf{dP}$ 项算错，导致 systematic gradient bias。这个 drop 不是渐进的，是 catastrophic 的，说明这个 identity 是"必要条件"而非"优化项"。

Exp 5, 6 的 ablation 证明：**outlier mitigation heuristic 在 QAT 框架下完全 redundant**。这从侧面回答了一个 deep question：QAT 是否让模型学会自己 "suppress outliers"？很可能是的——经过 QAT，模型 weight 主动 shift 到一个 representation，使得 attention activation 的分布对 NVFP4 友好（少 outlier、range 利用率高）。这种"representation learning for quantization-friendly distribution"是 QAT 比 PTQ 强大的根本原因。

### 4.2 LLM 实验（Table 3, 4）

**Qwen3-14B Continued Training（Table 4）**：

| Precision | MMLU | WinoGrande | ARC-c | WikiText↓ |
|-----------|------|------------|-------|-----------|
| BF16 | 0.8044 | 0.7403 | 0.5922 | 0.5700 |
| FP4 | 0.7965 | 0.7214 | 0.5734 | 0.5763 |
| Attn-QAT | 0.7984 | **0.7585** | **0.6084** | 0.5778 |

有趣的是 Attn-QAT 在 WinoGrande 和 ARC-c 上**超过** BF16 baseline。这种"quantization as regularizer"现象在 8-bit QAT 文献里也有报道（比如 Jacob et al. 2018, https://arxiv.org/abs/1712.05877），可能因为 FP4 误差起到了 noise injection 作用，类似于 dropout。

**Llama 3.1-70B**：Attn-QAT 部分恢复但没完全 match BF16，作者归因于 training budget 不足（4 B200 × 6 小时）。这暗示 **70B 级别模型需要更长的 QAT schedule**，可能是 future work 的方向。

**SFT 实验（Table 3）**：直接用 Attn-QAT 做 SFT（不先做 continued pre-training QAT），在 Qwen3-14B 上几乎完全 match BF16 SFT 性能。这个结果 practical value 很高：意味着生产中可以 skip 掉 dedicated QAT 阶段，直接在 SFT 数据上 QAT。

### 4.3 Kernel Benchmark（Figure 5）

RTX 5090 上 Attn-QAT 比 SageAttention3 快 **1.1x-1.5x**。原因：
- 去掉 Q/K smoothing 的 preprocessing overhead
- 去掉 P 的 two-level quantization
- Single-level NVFP4 quantization 路径更短

这个结果实际上颠覆了一个常见 assumption：**"加 outlier mitigation 损失 speed 换 accuracy"**。在 QAT 框架下，accuracy 由训练 recover，inference 时反而可以走最简单的 NVFP4 路径，**同时拿到 accuracy 和 speed**。这是 QAT vs PTQ 的另一个被低估的优势。

---

## 5. 与 FlashAttention-4 / Blackwell 的关系

paper Section 5 提到他们正在基于 **FlashAttention-4 (FA4) CuTe-DSL kernel** (https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd_sm100.py, Dao et al. 2025) 开发 SM100 native 版本。FA4 支持：
- Block-sparse attention
- Paged attention（用于 LLM serving）
- Native NVFP4 GEMM（取代 fake quant emulation）

这意味着 Attn-QAT 的最终形态可能是：训练时用 native FP4 GEMM（而非 fake quant），实现真正的 train-inference 一致性 + 训练加速。当前 paper 用 fake quant 训练（BF16 GEMM + FP4 emulation），train-test mismatch 通过 Figure 4 验证可忽略，但理论上 native FP4 training 会更好。

---

## 6. 联想与 Open Problems

### 6.1 与 Native FP4 Training 的关系

paper Section 4 区分了 QAT 和 native low-bit training：
- **QAT**: forward 模拟低精度（fake quant），backward 用高精度梯度
- **Native low-bit training**: forward 和 backward 都用低精度 GEMM，从 scratch 训练

最近的 native FP4 工作（Abecassis et al. 2025, https://arxiv.org/abs/2509.25149; Chmiel et al. 2025, https://arxiv.org/abs/2505.19115; BitNet a4.8, https://arxiv.org/abs/2411.04965）已经证明 linear layer 可以 native FP4 训练。**但 attention 的 native FP4 backward 还是 open problem**。Attn-QAT 的两个发现（precision matching, high-precision $\mathbf{O}'$）为这个方向提供了重要 insight：即使要做 native FP4 backward，softmax 内部（$\mathbf{P}^\top \mathbf{dP}$ 项）可能必须保留 FP32，否则 identity 不成立。

### 6.2 KV Cache 4-bit 的 implication

Attn-QAT 训练的模型，其 KV cache 可以直接用 NVFP4 存储（paper Section 5 提到 future work）。这比当前的 KV cache quantization 方法（如 KIVI https://arxiv.org/abs/2402.02750, KVQuant https://arxiv.org/abs/2401.18079）有本质优势：那些方法都是 PTQ，对 heavy-tailed KV 敏感；Attn-QAT 训练的模型 KV 分布已经 quantization-friendly。

### 6.3 与 GPT-Q / AWQ / SmoothQuant 的对比

| Method | Layer | Training? | Outlier Mitigation | 4-bit Quality |
|--------|-------|-----------|--------------------|---------------|
| GPT-Q | Linear | PTQ | Hessian-based | Good for weight |
| AWQ (https://arxiv.org/abs/2306.00978) | Linear | PTQ | Activation-aware scaling | Good for weight |
| SmoothQuant (https://arxiv.org/abs/2211.10438) | Linear | PTQ | Migration to weight | Good for 8-bit |
| SageAttention3 (https://arxiv.org/abs/2505.11594) | Attention | PTQ | Q/K smoothing + 2-level P | Marginal at 4-bit |
| **Attn-QAT** | Attention | QAT | **None (learned)** | **BF16-level** |

这张表揭示一个 pattern：**PTQ + outlier mitigation 在 8-bit 够用，在 4-bit 触顶**。4-bit 时代的 attention quantization 几乎必须走 QAT 路线。这跟 linear layer 早期的发展史（Jacob et al. 2018 QAT → 后期 GPT-Q/AWQ PTQ 成熟）形成对比——attention 因为 fused kernel 复杂性，QAT 起步晚。

### 6.4 Triton Kernel 实现细节

paper 提到训练 kernel 用 Triton，inference 用 CUDA。Triton kernel 基于 Triton reference attention (Tillet et al. 2019, https://arxiv.org/abs/2403.02923) 修改。NVFP4 emulation 在 Blackwell 上用 PTX inline：
- `cvt.rn.satfinite.e2m1x2.f32`: 两个 FP32 → 两个 FP4 e2m1
- `cvt.rn.f16x2.e2m1x2`: 两个 FP16 → 两个 FP4 e2m1

非 Blackwell GPU 上用 explicit bitwise ops 模拟。这个设计让 kernel 可以在 H100/A100 上做 development/debugging，只在 Blackwell 上 production 部署。

### 6.5 与 vLLM 集成

paper 修改了 vLLM (https://arxiv.org/abs/2309.06180) 的 Triton paged-attention 实现来支持 NVFP4 fake quant。这暗示 Attn-QAT 的 LLM serving 集成路径已经打通，一旦 FA4 SM100 kernel ready，可以直接进入主流 serving stack。

### 6.6 关于 $\mathbf{O}'$ Memory Overhead 的思考

paper 没有详细讨论 $\mathbf{O}'$ 的 memory cost，但这是 deployment 中的关键问题。对于 long context（如 128K），$\mathbf{O}' \in \mathbb{R}^{N_q \times d}$ 可能是 GB 级。可能的优化方向：
1. **Recompute $\mathbf{O}'$ on-the-fly in backward**：但需要重新做一次 PV matmul
2. **Block-wise $\mathbf{O}'$ storage**：只在 backward 的当前 tile 上计算 $\mathbf{O}'$
3. **Mixed precision $\mathbf{O}'$**：用 FP16 而非 FP32 存（但可能损失 identity 精度）

Appendix B.1 提到用 full gradient checkpointing 来 avoid OOM，暗示 $\mathbf{O}'$ 确实是 memory pressure 的来源。

---

## 7. 个人 Critical Thoughts

1. **Train-test mismatch 验证不够严格**：Figure 4 只做了 visual comparison，没有 numerical MSE 报告。理论上 fake quant (BF16 GEMM + FP4 emulation) 和 real quant (NVFP4 GEMM) 在 rounding mode 上可能有差异，尤其当 BF16 matmul 的 intermediate accumulator 精度不同于 NVFP4 tensor core 的 accumulator 时。

2. **Llama 3.1-70B 没完全 recover**：作者归因于 training budget，但另一个可能性是 70B 模型对 attention precision 的 sensitivity 本身更高（更多 head, 更长 pre-training）。需要更系统的 scaling law 实验。

3. **Long context 没测**：所有实验 seq_len ≤ 8192。long context 下 P 的 sparsity 更高，NVFP4 quantization 可能更 challenging。FA4 的 block-sparse 版本会有不同 trade-off。

4. **没对比 GQA / MQA**：现代 LLM 用 GQA (Grouped-Query Attention), K/V head 数 < Q head 数。这会改变 KV 量化的 sensitivity profile。paper 没单独 ablation。

5. **Diffusion 的 dynamic degree 没完全 recover**：这暗示 FP4 对高动态 motion 仍有 information loss。可能的 fix 是 mixed-precision attention（部分 head 用 FP8），但会损失 speed。

---

## References

- SageAttention3: https://arxiv.org/abs/2505.11594
- SageAttention2: https://arxiv.org/abs/2411.10958
- SageAttention (original): https://arxiv.org/abs/2410.02367
- FlashAttention: https://arxiv.org/abs/2205.14135
- FlashAttention-4 SM100 kernel: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd_sm100.py
- SmoothQuant: https://arxiv.org/abs/2211.10438
- AWQ: https://arxiv.org/abs/2306.00978
- GPT-Q: https://arxiv.org/abs/2210.17323
- Jacob et al. QAT: https://arxiv.org/abs/1712.05877
- LLM-QAT: https://arxiv.org/abs/2405.06061
- NVFP4 Pretraining (Abecassis et al.): https://arxiv.org/abs/2509.25149
- FP4 All The Way (Chmiel et al.): https://arxiv.org/abs/2505.19115
- BitNet a4.8: https://arxiv.org/abs/2411.04965
- DeepSeek-V3 (FP8 training): https://arxiv.org/abs/2412.19437
- FP8-LM (Peng et al.): https://arxiv.org/abs/2310.18313
- Wan 2.1: https://arxiv.org/abs/2503.20314
- Qwen3: https://arxiv.org/abs/2505.09388
- Llama 3: https://arxiv.org/abs/2407.21783
- VBench: https://arxiv.org/abs/2311.13513
- vLLM / PagedAttention: https://arxiv.org/abs/2309.06180
- Triton: https://arxiv.org/abs/2403.02923 (or original Tillet et al. 2019)
- OCP MX Format Spec: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-f
- STE (Bengio et al.): https://arxiv.org/abs/1308.3432
- KIVI KV cache: https://arxiv.org/abs/2402.02750
- KVQuant: https://arxiv.org/abs/2401.18079
- lm-eval-harness: https://github.com/EleutherAI/lm-evaluation-harness
- EvalScope: https://github.com/modelscope/evalscope
- Olmo 3 / Dolci-instruct: https://arxiv.org/abs/2512.13961
- NVIDIA Blackwell architecture: https://www.nvidia.com/en-us/data-center/blackwell-architecture/

---

**TL;DR Intuition**：FlashAttention 把 attention output $\mathbf{O}$ 用了两次——一次作为 forward 输出，一次作为 backward 中重建 $\mathbf{P}^\top \mathbf{dP}$ 的 cache。这两个 role 在 full precision 下是一致的，QAT 把它们 split 成低精度（forward）和高精度（backward identity）两个版本，所以必须额外维护 $\mathbf{O}'$。同时 backward 中 recompute 的 $\mathbf{P}$ 也要 fake-quantize 以保持 STE 一致性。这两个 fix 看起来琐碎，但它们是从 "QAT 在 linear layer work" 推广到 "QAT 在 fused attention kernel work" 的关键 missing piece。QAT 比 PTQ + outlier mitigation 更好的根本原因是：**让 representation 自己适应 quantizer，而非用 heuristic 拉伸 quantizer 来适应 representation**。
