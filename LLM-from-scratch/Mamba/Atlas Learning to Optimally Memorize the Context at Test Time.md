---
source_pdf: Atlas Learning to Optimally Memorize the Context at Test Time.pdf
paper_sha256: b5301e6f08db30ea783e36b7ae2558768cedfc8a51079a20e9f9a5f4ebb938c6
processed_at: '2026-07-18T10:17:15-07:00'
target_folder: LLM-from-scratch/Mamba
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Atlas: Learning to Optimally Memorize the Context at Test Time 深度解析

Andrej, 这篇 paper 让我从 first principles 重新思考了 "memory" 在 sequence models 中的角色。我会从你喜欢的 "build intuition" 视角来拆解，把每个公式变量、每个 design choice 都讲透。

---

## 1. 核心直觉：把 RNN 当作 "test-time meta-learner"

这篇 paper 站在一个非常重要的视角上：**所有 sequence models 本质上都是 associative memory modules，在 test time 通过 inner-loop optimization 把 context 压缩到 memory parameters 里**。这个视角来自 Behrouz 等人的 Miras framework (arXiv:2504.13173, https://arxiv.org/abs/2504.13173)。

让我 build 这个 intuition：

### 1.1 Memory perspective 的数学定义

Definition 1 (来自 paper)：
$$M^* = \arg\min_M \quad \mathcal{L}(M(\mathcal{K}); \mathcal{V})$$

变量解释：
- $M$: memory module，可以是一个 matrix $M \in \mathbb{R}^{d_v \times d_k}$，也可以是一个 deep MLP
- $\mathcal{K} \subseteq \mathbb{R}^{d_k}$: keys 集合，$d_k$ 是 key dimension
- $\mathcal{V} \subseteq \mathbb{R}^{d_v}$: values 集合，$d_v$ 是 value dimension
- $\mathcal{L}$: attentional bias，即 internal objective，决定 memory 的 "类型" 和 "优先级"

**关键 insight**: 这意味着 sequence model = meta-learner with two loops:
- **Inner loop**: 用 gradient descent (或其它 optimizer) 优化 memory parameters $\theta_M = \{W_1, W_2, \dots, W_{\mathcal{L}_M}\}$
- **Outer loop**: 用标准 SGD/AdamW 优化 model 的其他参数 (projections, MLPs, etc.)

### 1.2 用这个视角 unify 所有 architecture

让我把这个 unification 用具体例子展开：

**Linear Attention** (Katharopoulos et al. 2020, https://arxiv.org/abs/2006.16236):
$$\ell_t := \langle M_{t-1} \mathbf{k}_t, \mathbf{v}_t \rangle$$
$$M_t = M_{t-1} - \eta_t \nabla \langle M_{t-1} \mathbf{k}_t, \mathbf{v}_t \rangle = M_{t-1} + \eta_t \mathbf{v}_t \mathbf{k}_t^\top$$

变量解释：
- $M_{t-1} \in \mathbb{R}^{d_v \times d_k}$: 时间 $t-1$ 的 memory matrix
- $\mathbf{k}_t \in \mathbb{R}^{d_k}$, $\mathbf{v}_t \in \mathbb{R}^{d_v}$: 当前 token 的 key 和 value
- $\eta_t$: learning rate (可以是 data-dependent)
- $\nabla$ 是对 $M$ 的梯度

**DeltaNet** (Schlag et al. 2021, https://arxiv.org/abs/2102.11174):
$$\ell_t := \|M_{t-1} \mathbf{k}_t - \mathbf{v}_t\|_2^2$$
$$M_t = (I - \eta_t \mathbf{k}_t \mathbf{k}_t^\top) M_{t-1} + \eta_t \mathbf{v}_t \mathbf{k}_t^\top$$

这里用 $\ell_2$ regression loss，gradient descent 会自动给出 Delta rule —— 这就是 fast weight programmer 的精髓。

**Transformer attention** 的 Nadaraya-Watson 视角 (Equation 17)：
$$\mathcal{M}^* = \arg\min_M \sum_{i=1}^L s(\mathbf{k}_i, \mathbf{q}) \|\mathbf{v}_i - \mathcal{M}\|_2^2 = \sum_{i=1}^L \frac{s(\mathbf{k}_i, \mathbf{q})}{\sum_j s(\mathbf{k}_j, \mathbf{q})} \mathbf{v}_i$$

这里 $s(\cdot, \cdot)$ 是 similarity function (softmax kernel)。**Transformer 是 non-parametric solution，每次都从头优化 memory $\mathcal{M}$**，而 RNN 是 parametric，维护一个 persistent state。

---

## 2. 三个核心问题：为什么 modern RNNs 还是不够好？

paper 识别出三个 disjoint 问题，我觉得这是 paper 最 valuable 的贡献之一：

### Problem 1: Limited Memory Capacity

**Proposition 1**: matrix-valued memory with $d_v \times d_k$ parameters 和 $\ell_2$ attentional bias，最多能 perfectly memorize $O(d_k)$ 个 linearly independent $(\mathbf{k}, \mathbf{v})$ pairs。

**直觉**: 这是 linear algebra 的 rank constraint。如果 $K = [\mathbf{k}_1, \dots, \mathbf{k}_m] \in \mathbb{R}^{d_k \times m}$，要解 $MK = V$，需要 $\text{rank}(V) \leq \text{rank}(K) \leq d_k$。所以 capacity 被 key dimension 限制。

**Theorem 1 (Deep memory 的 effect)**: 用 $\mathcal{L}_M \geq 2$ 层的 MLP 作为 memory，capacity 至少 $O(d_k d_v)$，至多 $O(d_k d_v \sum_{i=1}^{\mathcal{L}_M} \min\{d_h^{(j)}\}_{j \geq i} d_h^{(j+1)})$。

变量解释：
- $\mathcal{L}_M$: memory MLP 的层数
- $d_h^{(j)}$: 第 $j$ 层的 hidden dimension
- $d_h^{(0)} := d_k$, $d_h^{(\mathcal{L}_M)} := d_v$

**为什么 deep memory 有用？** 因为 ReLU network 是 piecewise affine，每个 linear region 里 $M(\mathbf{x}) = A\mathbf{x} + B$，而 $A = W^{(\mathcal{L}_M)} D^{(\mathcal{L}_M-1)} W^{(\mathcal{L}_M-1)} \cdots D^{(1)} W^{(1)}$，其中 $D^{(\ell)}$ 是 diagonal $\{0,1\}$ mask。不同 activation pattern 给出不同 affine map，所以 effective rank 大大提升。

### Problem 2: Online Nature of Update

现有 RNNs 在每个 time step $t$ 只 optimize 当前 token：
$$\min_M \ell(M; \mathbf{k}_t, \mathbf{v}_t) + \text{Ret}_t(M, M_{t-1})$$

**问题**: 这意味着 memory 贪心地 memorize individual tokens，忽略 context。一个 event 可能由多个 tokens 组成，online update 无法 capture 这个 structure。

paper 提出的 **Omega rule** (Equation 9)：
$$\min_M \sum_{i=t-c+1}^t \gamma_i^{(t)} \|M(\mathbf{k}_i) - \mathbf{v}_i\|_2^2$$

变量解释：
- $c \in \mathbb{N}_{\geq 1}$: local context window size
- $\gamma_i^{(t)} \in [0,1]$: decay/gate term，控制第 $i$ 个 token 在当前 optimization 中的 weight
- 当 $c=1$: 退化成 online Delta rule
- 当 $c=\infty$: 全局 optimization（像 attention 那样）

**关键 insight**: $\gamma_i^{(t)}$ 是 input-dependent，相当于 in-context pruning —— 模型可以学会在 context 里 "skip" 不重要的 tokens。

### Problem 3: Less Expressive Memory Management

gradient descent 只用 first-order information，会 converge 到 spurious local minima。paper 提出用 **Muon optimizer** (Jordan et al. 2024, https://kellerjordan.github.io/posts/muon/) 替换 vanilla GD。

---

## 3. Atlas 的核心架构

### 3.1 OmegaNet: 第一个 building block

OmegaNet = Omega rule + polynomial kernels + deep memory + GD with momentum

Update rule (Equation 10):
$$\mathcal{M}_t = \alpha_t \mathcal{M}_{t-1} - \underbrace{\nabla \sum_{i=t-c+1}^t \gamma_i^{(t)} \|M(\phi(\mathbf{k}_i)) - \mathbf{v}_i\|_2^2}_{\text{Surprise of the context}}$$

Linear memory 特例 (Equation 11):
$$\mathcal{M}_t = (\text{diag}(\alpha_t) - \sum_{i=t-c+1}^t \gamma_i^{(t)} \phi(\mathbf{k}_i)\phi(\mathbf{k}_i)^\top) \mathcal{M}_{t-1} - \sum_{i=t-c+1}^t \gamma_i^{(t)} \mathbf{v}_i \phi(\mathbf{k}_i)^\top$$

变量解释：
- $\alpha_t$: weight decay / retention gate (input-dependent)
- $\phi(\cdot)$: polynomial kernel mapping，把 $\mathbf{k} \in \mathbb{R}^{d_k}$ 映射到更高维空间
- $\phi(\mathbf{k}_i)\phi(\mathbf{k}_i)^\top \in \mathbb{R}^{D \times D}$ where $D = \binom{d_k + p}{p}$ 是 polynomial feature 的维度

**为什么 polynomial kernel 提升容量？** Proposition 2 证明了：用 degree-$p$ polynomial mapping，capacity 从 $O(d_k)$ 跳到 $O(d_k^p)$。

直觉：polynomial kernel 等价于 implicit high-dimensional feature space，类似 kernel methods。具体地：
$$\phi_p(x) = [x^\beta]_{|\beta| \leq p}$$

例如 $p=2$, $d_k=2$ 时：$\phi_2([x_1, x_2]) = [1, x_1, x_2, x_1^2, x_1 x_2, x_2^2]$，从 2D 变成 6D。

### 3.2 Atlas: 加入 Muon optimizer

**Atlas** 的 update rule (Equation 32-33):
$$\mathcal{M}_t = \alpha_t \mathcal{M}_{t-1} - \eta_t \text{NewtonSchulz-}k(S_t)$$
$$S_t = \theta_t S_{t-1} + \nabla \sum_{i=t-c+1}^t \gamma_i^{(t)} \|M(\phi^*(\mathbf{k}_i)) - \mathbf{v}_i\|_2^2$$

变量解释：
- $S_t$: momentum term，accumulates gradients over context window
- $\theta_t$: momentum coefficient
- $\text{NewtonShulz-}k(\cdot)$: Newton-Schulz iteration 算 $k$ 步，approximate 最近的 semi-orthogonal matrix
- $k$: Newton-Schulz steps，可以看作 **internal test-time compute parameter** —— 多算几步 = 多花 compute 来更好地 memorize

**Muon 的本质**: Muon 不是普通 momentum，它对 momentum matrix 做 orthogonalization。Newton-Schulz iteration 计算 $S_t (S_t^\top S_t)^{-1/2}$ 的近似，等价于取 $S_t$ 的 polar decomposition 的 orthogonal part。这相当于：
1. 抓住 gradient 的 "方向" (orthogonal component)
2. 丢弃 "magnitude distortion" (singular values)
3. 给出更好的 conditioning，类似 second-order method 的效果

参考 Muon 原文：https://kellerjordan.github.io/posts/muon/

### 3.3 Atlas 的 memory 架构

paper 用 2-layer MLP with residual connection (Equation 42):
$$M(\cdot) = (\cdot) + W_1 \sigma(W_2 (\cdot))$$

扩展版 Atlas++ 用 gated MLP (Equation 43):
$$M(\cdot) = (\cdot) + W_1 (\sigma(W_2 (\cdot)) \otimes W_3 (\cdot))$$

变量解释：
- $W_1, W_2, W_3$: 可学习的 linear layers
- $\sigma$: GELU activation (Hendrycks & Gimpel 2016, https://arxiv.org/abs/1606.08415)
- $\otimes$: element-wise multiplication (gating)
- 残差连接保证 $M$ 可以 approximate identity，便于稳定训练

### 3.4 Hybrid 架构 (MAG / MAL)

paper 还把 Atlas 和 SWA 结合 (Figure 3)：
- **MAG** (Memory as Gate): Atlas memory 输出 gate SWA 的输出
- **MAL** (Memory as Layer): Atlas memory 和 SWA 串联，类似 Samba (https://arxiv.org/abs/2406.07522)

---

## 4. DeepTransformers: 严格推广 Transformer

### 4.1 关键推导：softmax kernel = infinite-dim feature map

paper 在 Section 4.2 做了一个 elegant 的推导。softmax kernel 不能 separable，所以 Transformer 不能写成 recurrence。但是 exponential kernel 可以写成 infinite-dim feature map 的 inner product (Equation 22-23):
$$\phi^*(x) = \begin{pmatrix} 1 \\ x \\ x^{\otimes 2}/\sqrt{2!} \\ x^{\otimes 3}/\sqrt{3!} \\ \vdots \end{pmatrix}, \quad \phi_P(x) = x^{\otimes p}$$

$$\exp(\mathbf{q}_t^\top \mathbf{k}_t) = \phi^*(\mathbf{q}_t)^\top \phi^*(\mathbf{k}_t)$$

变量解释：
- $x^{\otimes p} = x \otimes x^{\otimes (p-1)}$: Kronecker product "self-tensoring"
- $\phi^*(\cdot)$: infinite-dim feature map，对应 exponential kernel
- $\phi_P(\cdot)$: finite-dim polynomial feature map

**这个 observation 让 Transformer 变成 RNN！** 看 Equation 26:
$$M_t = M_{t-1} + \mathbf{v}_t \phi^*(\mathbf{k}_t)^\top = \sum_{i=1}^t \mathbf{v}_i \phi^*(\mathbf{k}_i)^\top$$
$$\Rightarrow \mathbf{y}_t = M_t \phi^*(\mathbf{q}_t) = \sum_{i=1}^t \mathbf{v}_i \exp(\mathbf{q}_t^\top \mathbf{k}_i)$$

这正是 **unnormalized Transformer attention**！

### 4.2 DeepTransformer = Transformer + deep memory

把 linear memory $M$ 换成 deep MLP $M(\cdot)$，并且用 Hebbian rule (Equation 25):
$$\mathcal{M}_t = \mathcal{M}_{t-1} - \nabla \langle \mathcal{M}_{t-1}(\phi^*(\mathbf{k}_t)), \mathbf{v}_t \rangle$$

Linear memory 特例退化成 Transformer。所以 **DeepTransformer 严格 generalize Transformer**。

### 4.3 Dot (Deep Omega Transformer) = Transformer + Omega rule

把 Hebbian 换成 Omega rule (Equation 27):
$$\mathcal{M}_t = \mathcal{M}_{t-1} - \nabla \sum_{i=t-c+1}^t \gamma_i^{(t)} \|M(\phi^*(\mathbf{k}_i)) - \mathbf{v}_i\|_2^2$$

Online case $c=1$ (Equation 31):
$$\mathbf{y}_t = (I - \eta_t \phi^*(\mathbf{k}_t) \exp(\mathbf{q}_t^\top \mathbf{k}_t)) \mathcal{M}_{t-1} - \eta_t \mathbf{v}_t \exp(\mathbf{q}_t^\top \mathbf{k}_t)$$

**直觉**: Dot 不仅 append new $(\mathbf{k}, \mathbf{v})$ (像 Transformer)，还会用 new value **replace** 之前预测的 value —— 这是 Delta rule 的精神。

---

## 5. Parallel Training：让 Omega rule 可训练

这是 paper 的工程核心。Naive implementation 需要 materialize $c$ 个 gradients，memory overhead 太大。

### 5.1 Chunk-wise parallel computation

paper 把 sequence 切成 chunks of size $b$。Define $t' = t - \bmod(t, b)$，即 chunk 起始位置。Intra-chunk parallel，inter-chunk recurrent (Equation 16):
$$M_t = \alpha_t \cdots \alpha_{t'} M_{t'} - \sum_{n=t'}^t \frac{\alpha_t \cdots \alpha_{t'}}{\alpha_n \cdots \alpha_{t'}} \underbrace{\sum_{i=n-c+1}^n \nabla \ell(M_{t'}; \mathbf{k}_i, \mathbf{v}_i)}_{G_t}$$

变量解释：
- $M_{t'}$: 上一 chunk 的最后 state (frozen during current chunk)
- $G_t$: 在 $M_{t'}$ 处计算的 sliding window gradient sum
- $\alpha_t \cdots \alpha_{t'}$: decay chain，weight 过去 contributions

**关键 trick**: 用 sliding window mask $M_s$ 在 broadcasting (einsum) 时 mask 掉超出 window 的位置。当 $c=1$, $M_s$ 是 identity；$c>1$ 时，对角线前 $c-1$ 个位置也置 1。

### 5.2 Atlas 的 parallel training (Equation 39-41)

Atlas 的 momentum recurrence 可以 fully parallel:
$$S_t = \underbrace{\theta_t \cdots \theta_1}_{\beta_t} S_0 - \sum_{i=1}^t \frac{\theta_t \cdots \theta_1}{\theta_i \cdots \theta_1} \eta_i u_i = \beta_t S_0 - \Theta \odot E \odot G$$

变量解释：
- $\beta_t$: cumulative momentum decay
- $\Theta, E$: diagonal matrices with $\theta$ 和 $\eta$ values
- $G$: gradient matrix
- $\odot$: broadcasting multiplication

然后:
$$S_t' \gets \text{Newton-Schulz}_5(S_t)$$
$$\mathcal{M}_t = \mathcal{M}_{t-1} + S_t'$$

**为什么能 parallel？** 因为 momentum recurrence 独立于 memory state $M$，可以一次性算出所有 $S_t$，再分别做 Newton-Schulz。

参考 TTT 的 parallel training: https://arxiv.org/abs/2407.04620

---

## 6. 实验数据深度分析

### 6.1 Language Modeling (Table 2)

760M params / 30B tokens：
| Model | Wiki ppl ↓ | LMB ppl ↓ | Avg acc ↑ |
|-------|-----------|-----------|-----------|
| Transformer++ | 25.21 | 27.64 | 48.69 |
| Gated DeltaNet-H2* (hybrid) | 19.88 | 20.83 | 51.49 |
| Titans (LMM) | 20.04 | 21.96 | 51.56 |
| **Atlas** | **18.92** | 21.01 | **52.77** |
| Atlas++ | 19.04 | 20.03 | 53.09 |
| Atlas (MAG) | 18.62 | 21.18 | 53.08 |

1.3B params / 100B tokens：
| Model | Wiki ppl ↓ | Avg acc ↑ |
|-------|-----------|-----------|
| Transformer++ | 18.53 | 52.25 |
| Titans (LMM) | 15.60 | 56.82 |
| **Atlas++** | **14.40** | **58.03** |

**Key observations**:
1. Atlas (non-hybrid) 已经 beat Samba 和 Gated DeltaNet-H2 (hybrid)
2. MAG hybrid 进一步提升 (18.62 vs 18.92 Wiki ppl)
3. Atlas++ (gated MLP memory) 比 Atlas 略好，说明 memory 架构 matters

### 6.2 Long Context: BABILong (Figure 4)

这是 paper 的 highlight result：**Atlas 在 10M context length 上保持 +80% accuracy**，而 Titans 在 10M 上 collapse。

参考 BABILong: https://arxiv.org/abs/2406.04271

paper 归因于三个因素：
1. Muon optimizer → 更好的 memory management
2. Polynomial kernels → 更大 capacity
3. Context memorization (vs token memorization) → 更鲁棒

### 6.3 Needle in Haystack (Table 3)

S-NIAH-N (single needle, number) at 16K context:
| Model | 2K | 4K | 8K | 16K |
|-------|-----|-----|-----|------|
| TTT | 60.2 | 36.6 | 10.2 | 4.4 |
| DeltaNet | 47.2 | 15.4 | 12.8 | 5.4 |
| Titans | 100.0 | 99.8 | 93.4 | 80.2 |
| **Atlas** | 100.0 | 100.0 | 93.0 | 84.0 |
| Atlas (MAG) | 100 | 99.2 | 97.4 | 97.0 |

Atlas 在 16K 上比 Titans 高 4 个点，hybrid MAG 高 17 个点。

### 6.4 Ablation Study (Table 6)

| Component | Wiki ppl | CS Reasoning |
|-----------|----------|--------------|
| Atlas (full) | 19.97 | 52.77 |
| + Gated MLP Memory | 19.53 | 53.09 |
| + Attn (MAG) | 19.90 | 53.08 |
| Linear Memory | 21.03 | 49.74 |
| w/o Muon | 19.65 | 52.56 |
| c=1 (no context) | 21.98 | 49.26 |
| w/o Polynomial | 22.14 | 50.57 |

**Key takeaways**:
1. **c=1** (退化成 online) 损失最大：ppl 从 19.97 → 21.98 (+2.01)，证明 context memorization 是核心
2. **w/o Polynomial** 损失次大：ppl → 22.14 (+2.17)，证明 polynomial kernel 关键
3. **Linear Memory** 损失也大：ppl → 21.03 (+1.06)，证明 deep memory 重要
4. **w/o Muon** 损失较小：ppl → 19.65，但这里有意思 —— w/o Muon 的 ppl 反而更低？说明 Muon 主要帮助 generalization (CS Reasoning 52.56 vs 52.77)，而不是 training loss

### 6.5 MAD Synthetic Benchmark (Table 4)

| Model | Compression | Noisy ICR | Fuzzy ICR | Sel. Copy | Memorization | Avg |
|-------|-------------|-----------|-----------|-----------|--------------|-----|
| Transformers | 49.4 | 100 | 48.2 | 95.9 | 83.8 | 75.46 |
| Titans | 49.6 | 100 | 49.7 | 99.4 | 83.5 | 76.44 |
| **Atlas** | **51.6** | 100 | **54.9** | 99.6 | **91.4** | **79.50** |

Atlas 在 memorization 上达到 91.4%，比 Transformers 高 7.6 个点 —— 这是 memory capacity 的直接体现。

参考 MAD: https://arxiv.org/abs/2403.17844

### 6.6 In-Context Recall (Table 5)

| Model | SWDE | NQ | DROP | FDA | SQUAD | TQA | Avg |
|-------|------|-----|------|-----|-------|-----|-----|
| Transformers | 84.9 | 23.0 | 28.4 | 72.5 | 48.1 | 64.4 | 53.55 |
| Titans | 65.1 | 20.7 | 27.2 | 37.3 | 42.6 | 61.0 | 42.31 |
| **Atlas** | 66.8 | 21.9 | 27.4 | 40.7 | 44.1 | 61.3 | 43.70 |

Atlas 还是不如 Transformer (43.70 vs 53.55)，但已经 close the gap 比 Titans 好。这符合 Wen et al. 2024 (https://arxiv.org/abs/2402.18510) 的观察：RNNs 在 in-context retrieval 上有 fundamental bottleneck，但 Atlas 通过 context memorization + polynomial features 部分缓解。

---

## 7. Learnability Experiments (Section 6.4, Figure 6)

paper 做了一个有意思的 small-scale experiment：测试 MLP memory 在 online fashion 下学习不同 function classes 的能力。

5 个 settings:
1. **Low Rank Mappings**: $o_j = W^\top i_j$, $W = XY$ low-rank
2. **MLP Mappings**: $o_j = M(i_j)$, $M$ 是 random MLP
3. **Attention+MLP**: 需要部分 memorize past inputs
4. **Attention Outputs as Inputs**: input 有 correlations
5. **SWA+MLP**: sliding window attention + MLP

**Observations**:
- Setting 1 (low rank) 最易学
- Setting 4 (correlated inputs) 比 Setting 2 更快学，说明 optimizer 能利用 correlations
- Settings 3 & 5 (需要 memorize past) 最难
- **Surprising**: Setting 3 (global attention) 比 Setting 5 (SWA) 学得更好，counterintuitive。paper 假设是因为 SWA 需要 forget old inputs，但 optimizer 学不会 forget

这给 Atlas 的设计提供了 motivation：**Muon + context memorization 帮助 forget/manage memory**。

---

## 8. 与相关工作的联系

### 8.1 Titans (Behrouz et al. 2024, https://arxiv.org/abs/2501.00663)

Titans 是 Atlas 的直接 predecessor。Titans 用 momentum GD + persistent memory tokens，但仍然是 online (c=1)。Atlas 在 BABILong 10M 上碾压 Titans，证明 context memorization 是 key。

### 8.2 DeltaNet family (Schlag, Yang et al.)

DeltaNet 用 Delta rule，Gated DeltaNet 加 forget gate，RWKV-7 (https://arxiv.org/abs/2503.14456) 用 dynamic decay。这些都是 online (c=1) 的 special case。Atlas 的 Omega rule 严格 generalize 它们。

### 8.3 PolySketchFormer (Kacham et al. 2024, https://arxiv.org/abs/2407.10260)

PolySketchFormer 用 polynomial kernel sketching 加速 Transformer。Atlas 借鉴 polynomial kernel idea 但用在 recurrent memory 上。

### 8.4 TTT (Sun et al. 2024, https://arxiv.org/abs/2407.04620)

TTT 也是 test-time training with deep memory，但用 online GD + chunk-wise parallel。Atlas 用 Muon + sliding window，更 expressive。

### 8.5 Mamba / SSMs (Gu & Dao 2023, https://arxiv.org/abs/2312.00752)

Mamba 是 data-dependent SSM，可以看作 linear RNN with input-dependent decay。但 SSMs 的 memory 是 vector-valued，capacity 受限。Atlas 用 deep memory + polynomial features 大幅提升 capacity。

### 8.6 Hopfield Networks / Modern Hopfield (Ramsauer et al. 2021, https://arxiv.org/abs/2008.02217)

Modern Hopfield 用 exponential kernel 实现 exponential capacity。Atlas 的 DeepTransformer 推导正好连接到这个：$\phi^*(\cdot)$ 就是 Modern Hopfield 的 exponential feature map。

### 8.7 Muon Optimizer (Jordan et al. 2024)

Muon 是 Keller Jordan 在 modded-NanoGPG 项目里提出的，用 Newton-Schulz iteration 做 momentum orthogonalization。Atlas 是第一个把 Muon 用作 inner-loop optimizer 的工作。

链接：https://kellerjordan.github.io/posts/muon/

---

## 9. Intuition Building: 为什么 Atlas work？

让我总结一下 build intuition 的几个 key insights：

### Insight 1: Memory capacity 是 function of feature dim, not param count

 Proposition 2 告诉我们：用 degree-$p$ polynomial，capacity 从 $O(d_k)$ 跳到 $O(d_k^p)$。这是 "免费午餐" —— 不增加 input projection 的参数，只增加 memory matrix 的 effective dimension。

**类比**: 这就像 kernel SVM —— 不显式映射到高维空间，但用 kernel trick 获得 high-dim 的表达力。

### Insight 2: Online update 是 "greedy"，context update 是 "joint"

Online update: 每个 token 独立 optimize memory。如果两个 tokens 的 keys 相似，后来的会 "overwrite" 前面的。

Context update (Omega rule): 同时考虑 $c$ 个 tokens，optimizer 找到一个 balance。$\gamma_i^{(t)}$ 允许模型学会 weight 不同 tokens。

**类比**: SGD vs mini-batch GD。Mini-batch 给出更稳定的 gradient direction。

### Insight 3: Muon = orthogonalized momentum = implicit second-order

Newton-Schulz iteration 算 $S(S^\top S)^{-1/2}$，这相当于取 $S$ 的 polar decomposition 的 orthogonal part。直觉上：
- 普通 momentum: $S_t = \theta S_{t-1} - \eta \nabla \ell$，会 accumulate noise
- Muon: 把 $S_t$ 投影到最近的 orthogonal matrix，去掉 magnitude distortion

**类比**: Muon 像 natural gradient 的 cheap approximation —— 它 normalize gradient 的 "shape"，让 optimization 更 stable。

### Insight 4: Test-time compute 通过 Newton-Schulz steps

Paper 提到一个很 elegant 的 observation：**$k$ (Newton-Schulz steps) 是 internal test-time compute parameter**。多算几步 = 多花 compute 来 better memorize。这和 test-time scaling 的趋势一致。

参考 test-time compute scaling: https://arxiv.org/abs/2408.03314

### Insight 5: Transformer = RNN with infinite-dim memory

$\phi^*(\cdot)$ 是 infinite-dim feature map，所以 Transformer 的 "memory" 是 unbounded。Atlas 用 polynomial $\phi_p(\cdot)$ 近似这个 infinite-dim，用 finite memory 容量换取 RNN 的 efficiency。

**Trade-off**: 
- $p \to \infty$: Atlas → DeepTransformer (full capacity, 但 slow)
- $p = 1$: Atlas 退化成 linear attention (fast, 但 capacity 低)
- 实践中 $p=2$ 或 $p=3$ 是 sweet spot

---

## 10. 我的一些思考和疑问

### Q1: 为什么 w/o Muon 的 perplexity 反而更低？

Table 6 显示 w/o Muon 的 Wiki ppl = 19.65 < Atlas 的 19.97。但 CS Reasoning 上 Atlas 更好 (52.77 vs 52.56)。这暗示 Muon 像 regularizer —— 牺牲一点 training fit 换取 generalization。这可能因为 Newton-Schulz 的 orthogonalization 抑制了 overfitting。

### Q2: Context window $c$ 的 scaling law？

Figure 5 显示 $c$ 增大性能提升，但 paper 没给 $c$ vs FLOPs 的 trade-off。直觉上 $c$ 增大让 sliding window gradient sum 更 expensive。值得研究 $c$ 的 scaling law。

### Q3: 能否用更高级的 optimizer？

Muon 是 momentum + orthogonalization。能否用 Shampoo (https://arxiv.org/abs/2002.09018) 或 AdaHessian (https://arxiv.org/abs/2006.00719) 作为 inner-loop optimizer？这会是自然的 next step。

### Q4: 和 in-context learning 的关系？

Atlas 的 context memorization 让我想到 in-context learning 的 "induction heads" 机制 (https://arxiv.org/abs/2209.11895)。Omega rule 的 $\gamma_i^{(t)}$ 能否学会做 "induction" —— 即 match 当前 token 和之前相似的 tokens？

### Q5: Memory capacity 和 grokking 的关系？

Deep memory 的 capacity 上界是 $O(d_k d_v \sum_i \min\{d_h^{(j)}\} d_h^{(j+1)})$，这让我想到 grokking 现象 —— 模型在某个 threshold 突然 generalize。是否 memory capacity 超过某个 threshold 会 trigger grokking？

---

## 11. 总结：Atlas 的 contribution map

```
Problem 1: Limited Memory Capacity
  → Solution: Polynomial kernels φ_p(·)
  → Proposition 2: capacity O(d_k) → O(d_k^p)

Problem 2: Online Nature
  → Solution: Omega rule (sliding window)
  → Equation 9: min_M Σ γ_i^(t) ||M(k_i) - v_i||^2
  → Context memorization vs token memorization

Problem 3: Weak Memory Management
  → Solution: Muon optimizer
  → Equation 32: M_t = α_t M_{t-1} - η_t NewtonShulz-k(S_t)
  → Approximate second-order info

Bonus: DeepTransformers
  → Transformer = RNN with φ*(·) infinite-dim memory
  → DeepTransformer = Transformer with deep memory
  → Dot = Transformer + Omega rule
  → Strict generalization
```

**核心 intuition**: 把 sequence model 看作 test-time meta-learner，把 capacity (feature map)、optimization scope (context window)、optimizer expressiveness (Muon) 三个维度都加强，就能接近 Transformer 的性能同时保持 RNN 的 efficiency。

---

## Reference Links

- Paper: https://arxiv.org/abs/2502.04247 (Atlas, this paper - 推测 arXiv ID)
- Titans: https://arxiv.org/abs/2501.00663
- Miras framework: https://arxiv.org/abs/2504.13173
- Muon optimizer: https://kellerjordan.github.io/posts/muon/
- TTT: https://arxiv.org/abs/2407.04620
- DeltaNet: https://arxiv.org/abs/2102.11174
- Linear Attention: https://arxiv.org/abs/2006.16236
- RWKV-7: https://arxiv.org/abs/2503.14456
- Mamba: https://arxiv.org/abs/2312.00752
- PolySketchFormer: https://arxiv.org/abs/2407.10260
- Modern Hopfield: https://arxiv.org/abs/2008.02217
- BABILong: https://arxiv.org/abs/2406.04271
- RULER: https://arxiv.org/abs/2404.06654
- MAD: https://arxiv.org/abs/2403.17844
- GELU: https://arxiv.org/abs/1606.08415
- Samba: https://arxiv.org/abs/2406.07522
- In-context retrieval bottleneck: https://arxiv.org/abs/2402.18510
- Induction heads: https://arxiv.org/abs/2209.11895
- Test-time compute: https://arxiv.org/abs/2408.03314
- Shampoo optimizer: https://arxiv.org/abs/2002.09018
- FineWeb: https://arxiv.org/abs/2406.17557

---

希望这个 deep dive 帮你 build 起对 Atlas 的 intuition, Andrej。这篇 paper 真正的贡献在于把 "memory as optimization" 的视角推到极致 —— capacity (polynomial features)、scope (sliding window)、optimizer (Muon) 三个 lever 都拉满。我觉得最 exciting 的方向是 **Newton-Schulz steps $k$ 作为 test-time compute knob**，这可能是 inference-time scaling 的新 axis。
