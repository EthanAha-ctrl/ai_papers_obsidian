---
source_pdf: THE CURIOUS CASE OF IN-TRAINING COMPRESSION.pdf
paper_sha256: 4b4054b7bafca2fabf82d65a2f1770de0a9f53a0e1a7a48c013dd8ab3d74625d
processed_at: '2026-08-12T13:59:23-07:00'
target_folder: Pruning-Quantization
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 COMPRESSM

## 一句话版本

**先让模型"发育"出大尺寸的脑子, 然后在训练过程中把"没用的脑区"砍掉, 后面 90% 的训练在小脑子上跑——结果又快又好。**

## 为什么需要这东西

现在的 SSM (Mamba, LRU 这些) 内部都有一个 hidden state, 你可以理解成模型的"工作记忆"。维度越大, 记得越复杂, 但算起来越贵。

你想得到一个小模型, 传统做法是:

1. 先花大代价训一个大模型
2. 训完以后再 pruning / distillation 压成小的

痛点很明显: **你得先付大模型的完整训练代价**。能不能在训练过程中就把没用的部分砍掉, 让后续训练直接享受小模型的便宜?

COMPRESSM 说: 能。

## 核心直觉

关键 observation 是: **大模型的"好"不只在最终权重, 还在于训练初期它 explore 出来的那片 high-dimensional landscape**。

打个比方: 你让一个 3 岁小孩直接学微积分, 他学不会。但你先让他正常长大、大脑发育完整, 然后在他 20 岁的时候告诉他"你只需要用数学那部分脑区, 别的可以休眠"——他马上就能用。

直接训小模型, 就像让 3 岁小孩学微积分。从 high-dim 起步再 prune, 就像让成年人选择性地"用而不弃"——那些重要的脑区已经长好了, 剩下的休眠也不影响。

**大维度提供的不是最终容量, 是训练初期那段 rich 的 exploration**。COMPRESSM 做的就是: 保留这段 exploration 的成果, 砍掉不再需要的 overhead。

## 怎么知道哪些维度"没用"

这里借了 control theory 几十年的老工具, 叫 **Hankel Singular Values (HSV)**。

每个 state dimension 有一个 HSV, 这个值告诉你: 这个维度**既容易被 input 影响, 又容易影响 output** 的程度。

- HSV 大 = input 进得来, output 出得去, 这维度很 active, 重要
- HSV 小 = input 进不来或 output 出不去, 这维度基本在摸鱼, 可以丢

所以算法很简单: 算每个维度的 HSV, 把小的砍掉, 留大的。

## 为什么训练中敢砍

你可能会担心: 训练刚开始我觉得这维度没用砍了, 万一训练后期它变得有用了呢?

Paper 用 Weyl's theorem 论证 + 实验观察证明这种担心基本多余:

**理论上**: 每次梯度更新引起的 HSV 变化是有上界的, 不会突然跳变。HSV 是连续演化的。

**实验上**: 他们画了训练过程中 HSV 的轨迹, 发现:
1. 训练初期有个 brief reshaping phase, 大的 HSV 和小的 HSV 分开
2. 分开之后, 排序基本稳定, 小的不会突然变大
3. 小 HSV 的 cumulative energy contribution 很快收敛, 不增长

换句话说: **早期看起来没用的维度, 后期几乎一直没用**。早期砍掉是安全的。

## 算法长什么样

非常 intuitive:

```
训练前 10% (warmup 阶段):
  每隔一段, 算一次 HSV
  把能量贡献最小的那些维度砍掉
  保留 (1-τ) 的总能量 (τ 是你设的容忍度)

训练后 90%:
  在已经变小了的模型上继续训
  享受速度
```

就这么多。核心 reduction 操作是 control theory 里的 **balanced truncation**——先把 state 变换到一个坐标系, 在那个坐标系里 Gramians 对角化, 然后直接切掉对应小 HSV 的坐标。这套理论 1980 年代就成熟了, 有 error bound 保证: 砍掉的 HSV 之和越小, reduced system 行为越接近原 system。

## 效果有多好

几个数字让你感受一下:

**CIFAR10 + LRU**:
- 直接训小模型 (state dim 57): accuracy 78.2%
- COMPRESSM 从 384 训练中砍到 57: accuracy **84.4%**
- 同样大小, 高出 6 个百分点

**sMNIST + LRU**:
- 直接训 dim 13: 92.6%
- COMPRESSM 从 256 砍到 13: **95.9%**
- 高出 3.3 个百分点

**训练速度**:
- CIFAR10 上从 384 砍到 92: 1.5× speedup, accuracy 几乎不掉 (85.7% vs baseline 86.5%)
- Mamba 上从 128 砍到 14: **4× speedup**

对比直接训小模型, 既快又准——这就是"start large, shrink"的威力。

## 几个有意思的 ablation

**砍 top HSV 会怎样?** 灾难。即使后面 90% 训练用来 recover, 也 recover 不回来。说明 top HSV 对应的维度真的承载了 dominant dynamics, 砍了就废了。

**砍 random HSV 会怎样?** 介于 baseline 和 top 之间。说明 HSV ranking 确实在做有意义的事情, 不只是随便砍就行。

**多砍几次还是少砍几次?** 差别不大。4 次是个好折中。

**早砍还是晚砍?** 几乎没差别。所以肯定早砍, 因为早砍能让后面更多训练享受小模型的便宜。

## 对 Mamba 的小插曲

Mamba 是个特殊情况: 它的每个 channel 是独立的 (SISO), 增大 state dim 对全局 expressivity 帮助不大。所以 Mamba 上 state dim 和性能弱相关——dim=8 和 dim=128 性能差不多。

这种情况下 COMPRESSM 主要给的是 **speedup** (4×), 不是性能提升。Paper 很诚实地承认了这个 caveat。

但作者指出: 像 Griffin、Mamba2、DeltaNet 这种 MIMO 架构, 一个 state vector 控制整个 feature vector, state dim 就是真正的 global capacity, COMPRESSM 应该更有效。这部分留作 future work。

## 跟其他压缩方法比

**vs Hankel Nuclear Norm Regularization**: 这个方法用 regularizer 强制 HSV 衰减, 训练时每步都要算 HSV, 训练慢 16 倍, 而且 regularizer 本身限制了 capacity, 训出来的大模型就不如 baseline。属于"既贵又差"。

**vs Knowledge Distillation**: 当 student 和 teacher 差不多大时 KD 不错, 但 student 远小于 teacher 时 KD 性能掉得厉害。而且 KD 要先训完 teacher, 然后训 student 时还要 forward teacher 拿 logits, 总时间比 COMPRESSM 长得多。

COMPRESSM 的独特价值: **不需要训完大模型**, compression 发生在训练过程中, 既省了后期 compression 的麻烦, 又让大部分训练享受小模型的便宜。

## 我觉得最深的 insight

Paper 里有个 ablation 我觉得特别值得品味: 早砍和晚砍性能几乎一样。

这说明什么? 说明**大模型 initialization 的价值, 在训练初期那一点点时间里就已经"定型"了**。warmup 阶段那 10% 的训练, 足以让 HSV spectrum 分离出 important 和 unimportant 维度。一旦分离完成, 后面再训也改变不了排序。

这跟 lottery ticket hypothesis 有 conceptual 呼应, 但 mechanism 不同。Lottery ticket 说大模型里有 lucky subnetwork; COMPRESSM 说大模型的 HSV spectrum 天然就告诉你哪些维度是核心, 砍掉小的对核心无害。

更深一层: 这暗示 **SGD 在 high-dim 空间中的 inductive bias 是 sparse 的**——大维度提供的"额外空间"大部分是 SGD 探索时的脚手架, 建好房子之后脚手架就可以拆了。COMPRESSM 就是那个拆脚手架的工具, 用 control theory 给出的"哪些是脚手架"的判断。

## 局限性

Paper 自己也提到了:

1. **SISO 架构上效果有限**: Mamba 这种每个 channel 独立的, state dim 和性能弱相关, COMPRESSM 主要价值是 speedup
2. **Mamba 的 LTI surrogate 是 hack**: 用 input average 做 stationary proxy 是 crude approximation, 可能 loss 掉 input-dependent 的重要 dynamics
3. **JIT overhead**: JAX 上每次 reduction 触发 ~5s recompilation, 在短训练上可能成为瓶颈
4. **理论 gap**: Weyl 只给了 per-step continuity, 但 "ordering stability" 和 "bottom-r 不增长" 是 empirical observation, 不是 theorem。某些 pathological training 可能违反

## 我会怎么看这篇 paper

如果你让我用一句话评价: **这是一个把 control theory 几十年的老工具 (balanced truncation, HSV) 精巧地 inject 到 deep learning training pipeline 里的 work, execution 很干净, insight 很有启发性**。

它不是 game-changer, 但它提出了一个新 paradigm: **compression 不一定是 post-hoc 的, 可以是 in-training 的**, 而且 control theory 给的 importance measure 比神经网络圈常用的 magnitude pruning 之类更有原则、更有保证。

代码在 https://github.com/camail-official/compressm, 值得跑一下 reproduction。特别建议看他们的 HSV tracking 可视化——看 HSV 在训练中怎么演化、排序怎么稳定, 对建立直觉很有帮助。

如果未来有人把这个扩展到 Mamba2 / GLA / DeltaNet 这种 MIMO 架构, 并且做 rank-aware CUDA kernel 把 JIT overhead 干掉, 这条路可能会成为 SSM 训练的 standard pipeline。

---

# COMPRESSM: In-Training Compression of State Space Models 深度解析

## 1. Core Intuition: 为什么这个方法值得仔细看

传统的 model compression pipeline 是: 先把 large model 训练到 convergence, 然后再做 pruning / distillation / quantization。这种 paradigm 的核心痛点在于 **upfront training cost**: 你必须先付出大模型的完整训练代价, 才能得到小模型。

COMPRESSM 提出了一种截然不同的策略: 让 SSM 在训练初期保持大 state dimension, 在 warmup 阶段就把"无用的"维度 prune 掉, 让后续 90% 的训练在更小的 dimension 上进行。这样既保留了 large model 的 expressivity initialization benefit, 又获得了 small model 的 training efficiency。

直觉上, 这类似于"先发育出完整的器官, 再萎缩掉不需要的部分"——biological development 中常见的策略, 比"直接长成精简版本"更容易达到 functional optimum。

**Reference links:**
- Project code: https://github.com/camail-official/compressm
- LRU paper (Orvieto et al., 2023): https://arxiv.org/abs/2303.06349
- Mamba paper: https://arxiv.org/abs/2312.00752
- S4 paper: https://arxiv.org/abs/2111.00396

---

## 2. Control Theory 背景详解

### 2.1 Discrete LTI System

整个 framework 的起点是 discrete Linear Time-Invariant (LTI) system:

$$
h(k+1) = Ah(k) + Bx(k), \quad h(0) = h_0
$$
$$
y(k) = Ch(k) + Dx(k)
$$

**变量解释:**
- $h(k) \in \mathbb{R}^n$: hidden state at timestep $k$, $n$ 就是 state dimension, 也是这个 paper 要 compress 的对象
- $x(k) \in \mathbb{R}^p$: input vector at time $k$, $p$ 是 input dimension
- $y(k) \in \mathbb{R}^q$: output vector, $q$ 是 output dimension
- $A \in \mathbb{R}^{n \times n}$: state transition matrix (在现代 SSM 如 LRU 中通常是 diagonal)
- $B \in \mathbb{R}^{n \times p}$: input-to-state matrix
- $C \in \mathbb{R}^{q \times n}$: state-to-output matrix
- $D \in \mathbb{R}^{q \times p}$: feedthrough (通常 skip connection, 不学习)

**三个关键 assumption:**
- Assumption 2.1 (Stability): $A$ 的所有 eigenvalues $|\lambda_i| < 1$, 保证 system 不会 blow up
- Assumption 2.2 (Controllability): $(A, B)$ 可控, 意味着 input 能 drive state 到任意位置
- Assumption 2.3 (Observability): $(A, C)$ 可观, 意味着 output 能反推 initial state

这三个 assumption 确保 system 是 "well-posed and non-degenerate", 是后续 Gramian 分析的基础。

### 2.2 Gramians: 能量的度量

**Controllability Gramian P** 满足 discrete Lyapunov equation:

$$
APA^\top - P + BB^\top = 0
$$

closed-form solution:
$$
P = \sum_{i=0}^{\infty} A^i B B^\top (A^\top)^i
$$

**直觉**: $P$ 衡量 input 能向每个 state direction 注入多少 energy。$P$ 的大 entries 对应那些"容易被 input 影响"的 state directions。

**Observability Gramian Q** 满足:

$$
A^\top Q A - Q + C^\top C = 0
$$

$$
Q = \sum_{i=0}^{\infty} (A^\top)^i C^\top C A^i
$$

**直觉**: $Q$ 衡量每个 state direction 对 output 贡献多少 energy。$Q$ 的大 entries 对应"容易被 output 观测"的 state directions。

### 2.3 对角 SSM 的特殊形式 (Equation 14)

对于 LRU / S5 这类使用 diagonal $A = \text{diag}(\lambda_1, ..., \lambda_n)$ 的 SSM, Lyapunov equation 有 entry-wise closed form:

$$
P_{ij} = \frac{(BB^\top)_{ij}}{1 - \lambda_i \lambda_j}, \quad Q_{ij} = \frac{(C^\top C)_{ij}}{1 - \lambda_i \lambda_j}
$$

**变量解释**:
- $P_{ij}$: Gramian 的第 $(i,j)$ 个 entry
- $\lambda_i, \lambda_j$: $A$ 的第 $i, j$ 个 eigenvalue (因为 $A$ 是 diagonal, 所以也是 diagonal entry)
- 分母 $1 - \lambda_i \lambda_j$ 来自 geometric series $\sum (\lambda_i \lambda_j)^k = 1/(1-\lambda_i \lambda_j)$, 要求 $|\lambda_i \lambda_j| < 1$

这个 closed form 极其重要, 因为它让 Gramian computation 从 $O(n^3)$ 降到 $O(n^2 p)$, 使 in-training compression 计算上可行。

### 2.4 Balanced Realization

Theorem 2.6 (Antoulas 2005) 指出, 任何 stable minimal LTI system 都存在一个 **balanced realization**, 使得:

$$
P = Q = W = \text{diag}(\sigma_1, ..., \sigma_n), \quad \sigma_1 \geq ... \geq \sigma_n > 0
$$

这些 $\sigma_i$ 就是 **Hankel Singular Values (HSVs)**, 它们衡量 joint controllability and observability of each state direction。

HSVs 也可以从 non-balanced realization 直接计算:
$$
\sigma = \text{sort}_\downarrow \left(\sqrt{\text{spec}(PQ)}\right) \quad \text{(Equation 4)}
$$

**Intuition**: 
- 大 HSV = state 既容易被 input 激发, 又容易被 output 观测 (重要!)
- 小 HSV = state 要么难被 input 激发, 要么难被 output 观测 (可以丢掉)

### 2.5 Balanced Truncation Error Bound

把 balanced system 分块:
$$
W = \text{diag}(\Sigma_1, \Sigma_2)
$$

其中 $\Sigma_1$ 是 top $r$ 个 HSV, $\Sigma_2$ 是剩下的。Truncation 后的 reduced system $\hat{\mathcal{G}}$ 满足:

$$
\|\mathcal{G} - \hat{\mathcal{G}}\|_\infty \leq 2 \sum_{i=r+1}^{n} \sigma_i \quad \text{(Equation 6)}
$$

**变量解释:**
- $\|\cdot\|_\infty$: $H_\infty$ norm, 即从 input 到 output 的 worst-case gain
- $\sigma_i$: 第 $i$ 大的 HSV
- $r$: retained dimension

**这个 bound 是整个 COMPRESSM 的理论基石**: 它告诉我们, 只要被 truncate 的 HSV 之和足够小, reduced system 的 input-output behavior 就接近 original。

Reference: Antoulas, "Approximation of Large-Scale Dynamical Systems", SIAM, 2005. https://epubs.siam.org/doi/abs/10.1137/1.9780898718713.ch7

---

## 3. 为什么 In-Training Compression 可行: Weyl's Theorem

这是 paper 最 technical 也最关键的理论 contribution。

### 3.1 训练中的 perturbation model

Gradient step 把 $(A, B, C)$ 变成 $(A', B', C')$:
$$
A' = A + \delta A, \quad B' = B + \delta B, \quad C' = C + \delta C \quad \text{(Equation 12)}
$$

由于 Gramians 是 $(A, B, C)$ 的 continuous function, 有:
$$
P' = P + \delta P, \quad Q' = Q + \delta Q \quad \text{(Equation 13)}
$$

### 3.2 构造 Hermitian matrix H

$PQ$ 一般 non-symmetric, 但它 similar to symmetric positive definite matrix $P^{1/2} Q P^{1/2}$。定义:

$$
H = (P^{1/2} Q P^{1/2})^{1/2}
$$

$H$ 的 eigenvalues 正好是 HSVs。由于 $H$ 是 continuous function of $(A,B,C)$, 有:
$$
H' = H + \delta H
$$

### 3.3 Weyl's Theorem (1912)

**Theorem 2.7**: 对 Hermitian matrices $W, W'$ with $\delta W = W' - W$:

$$
|\sigma_i(W') - \sigma_i(W)| \leq \max_{i=1,...,n} |\sigma_i(\delta W)| = \max(|\sigma_1(\delta W)|, |\sigma_n(\delta W)|) \quad \text{(Equation 7)}
$$

**变量解释**:
- $\sigma_i(W)$: $W$ 的第 $i$ 大 eigenvalue (按降序排列)
- $\sigma_1(\delta W)$: $\delta W$ 的最大 eigenvalue
- $\sigma_n(\delta W)$: $\delta W$ 的最小 eigen值 (可能为负)

**直觉**: 每个单个 gradient step 引起的 eigenvalue 变化, 不会超过 perturbation matrix 本身的 spectral radius。

### 3.4 Lemma 3.1: HSV Continuity

应用 Weyl's theorem 到 $H$ 和 $H' = H + \delta H$:

> 每个 HSV 在单个 gradient step 之间的变化, 至多为 $\delta H$ 的最大 absolute eigenvalue。

**为什么这至关重要**: 这意味着 HSV 不会"突然跳变"。我们可以 robustly track 每个 state dimension 对应的 HSV 轨迹 (用 linear sum assignment 在 bound overlap 时做 alignment)。

### 3.5 Empirical 观察支撑

Figure 2 展示了在 sMNIST 上训练 single LRU block (state dim 8) 的前 25k steps:
- **左图**: 原始 HSV 集合
- **中左图**: $\delta H$ 的 max absolute eigenvalue (perturbation bound)
- **中右图**: 把 bound 叠加在每个 HSV 上作为 error margin, 用 linear sum assignment tracking
- **右图**: bottom $r$ 个 HSV 的 cumulative energy contribution

关键观察:
1. $\delta H$ 的 spectral radius 远小于 HSV 之间的 gaps
2. 因此 HSV ordering 在训练中基本保持不变 (rare order crossings)
3. Bottom-$r$ HSV 的 cumulative contribution 迅速稳定, 不会突然增加

Reference: Weyl, 1912. https://api.semanticscholar.org/CorpusID:120278241

---

## 4. COMPRESSM 算法详解

### 4.1 主算法 (Section 3.1)

```
Input: SSM block with current state dim n, energy threshold τ
Output: Reduced SSM block with state dim r

1. Extract (A, B, C) from model weights
2. Solve Lyapunov equations to get P, Q
   (用 Equation 14 如果 A 是 diagonal)
3. Compute HSVs: σ = sort↓(√spec(PQ))
4. Find smallest r such that:
   ∑_{i=1}^{r} σ_i ≥ (1-τ) ∑_{i=1}^{n} σ_i   (Equation 8)
5. If r < 0.95n (reduction 足够大), compute balancing transform T
   否则跳过
6. Transform to balanced: (A_b, B_b, C_b) = (T⁻¹AT, T⁻¹B, CT)  (Equation 9)
7. Truncate: (A_r, B_r, C_r) = (A_b[:r,:r], B_b[:r,:], C_b[:,:r])  (Equation 10)
8. Replace model weights with (A_r, B_r, C_r)
   (可能需要 re-diagonalize 取决于 architecture)
```

**关键设计选择:**
- **Threshold (1-τ)**: 保留 $(1-\tau)$ 比例的总 HSV energy。$\tau$ 越大, 压得越狠。
- **0.95 cutoff (Step 5)**: 避免 trivial reduction (例如只减 1 维不值得 computational overhead)
- **Schedule**: 4 次 equidistant reductions 在 learning rate warmup 期间 (前 10% steps)

### 4.2 Pragmatic Variant (Section 3.2)

如果不想调 $\tau$, 可以用 validation-guided 版本:

1. Save 当前 checkpoint
2. Apply truncation (固定 truncation 10% states)
3. Train 几步, 评估 validation
4. 如果 validation 改善: 继续下一步 reduction
5. 如果 validation 退化: revert 到 saved checkpoint, 停止 reduction

**Trade-off**: 不需要调 $\tau$, 但 final dimension variance 很大 (Table 5: sMNIST 上 $121.3 \pm 89.8$ vs tolerance-based 的 $191.4 \pm 4.7$)。

### 4.3 Algorithmic Complexity (Section D.2)

| 步骤 | 复杂度 | 备注 |
|------|--------|------|
| Gramian computation | $O(n^2 p + n^2 q)$ | 对 diagonal A |
| HSV computation (SVD of PQ) | $O(n^3)$ | cubic |
| Rank selection | $O(n)$ | cumulative sum |
| Balanced transformation | $O(n^3)$ | cubic |
| **Total** | $O(n^3 + n^2 q + n^2 p)$ | independent of sequence length |

**重要 observation**: Reduction pipeline cost 与 sequence length **无关**, 因为它只作用于 state space dimension $n$。

实测 (Figure 12): $n=128$ 时 reduction pipeline < 0.1s; $n=512$ 时 ~3s。相比 training time saved, 这是 negligible overhead。

---

## 5. 实验结果深度分析

### 5.1 主结果 (Table 1)

让我详细解读 Table 1 中几个关键 dataset:

**CIFAR10 (6-layer LRU, initial n=384):**
| Tolerance τ | Final State Dim | COMPRESSM Acc | Baseline Acc |
|-------------|-----------------|---------------|--------------|
| 0.15 | 57.4 | 84.4% | 78.2% |
| 0.05 | 160.8 | 85.8% | 84.2% |
| 0.02 | 327.2 | 86.1% | 86.0% |
| 0 (no reduction) | 384 | - | 86.5% |

**惊人观察**: COMPRESSM 在 dim=57 时 (84.4%) 甚至超过 baseline 在 dim=57 时 (78.2%) 约 **6 个百分点**。这正是 "start large, shrink" vs "train directly at small dim" 的核心优势。

**sMNIST (1-layer LRU, initial n=256):**
| τ | Final Dim | COMPRESSM | Baseline |
|---|-----------|-----------|---------|
| 0.04 | 12.7 | 95.9% | 92.6% |
| 0.005 | 76.3 | 96.9% | 96.4% |
| 0.001 | 191.4 | 97.2% | 97.3% |
| 0 | 256 | - | 97.3% |

dim=13 时 COMPRESSM (95.9%) vs Baseline (92.6%), 差距 **3.3 个百分点**。

**AAN (initial n=256)**: COMPRESSM 和 baseline 几乎一致, 因为这个 task 上 state dim 与 performance 弱相关。这验证了 paper 的 caveat: "如果 state dim 与 performance 无关, COMPRESSM 不能 magically 产生更好的小模型。"

### 5.2 Training Speedup (Figure 3b, CIFAR10)

| Method | State Dim | Accuracy | Speedup |
|--------|-----------|----------|---------|
| Full baseline | 384 | 86.5% | 1.0× |
| Direct small | 92 | 81.8% | 1.6× |
| COMPRESSM | 92 | 85.7% | 1.5× |

COMPRESSM 达到与 baseline 几乎相同的 accuracy (85.7% vs 86.5%), 同时有 1.5× speedup; 而 direct small model 虽然 1.6× speedup 但 accuracy 掉到 81.8%。

### 5.3 Speedup 公式 (Section D.1.1)

Formal speedup model:

$$
T_{\text{base}} = E \cdot (s \cdot t_{\text{train}}(n_0) + t_{\text{eval}}(n_0))
$$

$$
T_{\text{red}} = \sum_{k=0}^{R} E_k \cdot (s \cdot t_{\text{train}}(n_k) + t_{\text{eval}}(n_k)) + \sum_{i=1}^{R} (t_{\text{analysis}}(n_{i-1}) + t_{\text{jit}})
$$

$$
S = \frac{T_{\text{base}}}{T_{\text{red}}}
$$

**变量解释:**
- $s$: gradient steps per epoch
- $E$: total epochs
- $R$: number of reductions
- $n_k$: state dimension during $k$-th reduction phase
- $E_k$: epochs run with dimension $n_k$, $\sum E_k = E$
- $t_{\text{train}}(n)$: per-step training time at dim $n$
- $t_{\text{eval}}(n)$: per-epoch evaluation time at dim $n$
- $t_{\text{analysis}}(n)$: reduction pipeline cost (HSV + truncation + diagonalization)
- $t_{\text{jit}}$: fixed JIT recompilation overhead (~5s in JAX)

**Key insight**: speedup 主要来自 $t_{\text{train}}(n)$ 在 $n$ 减小时显著下降 (Figure 11a 显示从 $n=256$ 到 $n=8$ 训练 step 时间下降 ~3×)。Reduction overhead 是 fixed cost, amortized over remaining training。

具体实例 (paper 中的 example):
- $s=5000, E=40, R=4$
- Reduction schedule: $n_k = [256, 216, 176, 136, 96]$
- $E_0=E_1=E_2=E_3=1, E_4=36$
- 实测 $T_{\text{base}} = 680.1s$, $T_{\text{red}} = 321.3s$
- **Speedup $S \approx 2.1\times$**

---

## 6. Ablation Studies 深度解读

### 6.1 Balanced Truncation Sanity Check (Section C.1)

三种 HSV selection scheme 对比 (256 → 32, 4 次 reduction):
1. **Bottom HSV truncation** (正确): 性能与 dim=256 baseline 几乎一致
2. **Random HSV truncation**: 性能介于 baseline dim=32 和 dim=256 之间
3. **Top HSV truncation** (adversarial): 灾难性下降, 即使后续 90% 训练也无法恢复

**关键 conclusion**: 这证明了 COMPRESSM 不是简单的 "pruning works", 而是 control-theoretic principled approach 必不可少。Top HSV 一旦被 remove, 网络无法 recover, 因为这些 dimension 包含 dominant dynamics。

### 6.2 Number of Reductions (Section C.2)

从 $R=1$ 到 $R=16$ reductions (256 → 32):
- $R=16$ (最 incremental): 唯一在 final reduction step 没有掉点的
- $R=8$: 有 small drop 但能 recover
- 总体: marginal 性能差异, 但 $R$ 越大 overhead 越多

**Practical takeaway**: $R=4$ 是合理折中 (paper 的 default)。

### 6.3 Reduction Window (Section C.3)

Reduction 在 5 个不同 window 进行: 0-20k, 25-45k, 50-70k, 100-120k, 150-170k steps:
- **No significant performance difference** between early and late windows
- Early reduction 有更高 noise 但 final accuracy 一致

**这是 COMPRESSM 最强的结论之一**: 既然 late reduction 没有性能优势, 早 reduction 能带来 90% training 的 speedup, 那就应该 early reduce。这直接验证了 in-training compression 的核心 motivation。

---

## 7. 与其他 Compression 方法对比 (Section 4.3, F)

### 7.1 vs Hankel Nuclear Norm (HNN) Regularization (Table 6, sMNIST)

| Method | Final Dim | Accuracy | Speed × |
|--------|-----------|----------|---------|
| COMPRESSM | 28 | 96.9% | 2.6× |
| HNN Regularization | 28 | 95.8% | 0.06× |

**HNN 的三个致命问题:**
1. **Computational cost**: 每步都要算 HSV, training 慢 ~16×
2. **Performance drop**: regularizer 强制 HSV 快速衰减, 限制 model capacity, 即使不 reduce 也达不到 baseline
3. **Compression efficiency**: 训练完大模型再 reduce, 失去 in-training speedup

### 7.2 vs Knowledge Distillation (Table 7, CIFAR10)

| Method | Final Dim | Accuracy | Speed × |
|--------|-----------|----------|---------|
| COMPRESSM | 57 | 84.4% | 1.58× |
| KD | 57 | 79.4% | 0.55× |
| COMPRESSM | 93 | 85.7% | 1.52× |
| KD | 93 | 83.5% | 0.52× |

**KD 的问题:**
- 当 student dim 接近 teacher dim (384) 时, KD 与 COMPRESSM 接近
- 当 student dim 大幅小于 teacher dim 时, KD 性能显著 drop
- KD 需要: (1) 先训练 teacher 到 completion; (2) training student 时 forward teacher 拿 logits; 总时间更长

KD 的 loss:
$$
\mathcal{L} = (1-\alpha)\mathcal{H}(y, \sigma(z_s)) + \alpha T^2 D_{\text{KL}}(\sigma(z_t/T) \| \sigma(z_s/T))
$$

变量:
- $\alpha$: balancing weight (这里 0.5)
- $T$: temperature (这里 2)
- $z_s, z_t$: student/teacher logits
- $\mathcal{H}$: cross-entropy
- $D_{\text{KL}}$: KL divergence

---

## 8. 对 Mamba 的扩展 (Section E)

### 8.1 Selective SSM 的挑战

Mamba 是 Linear Parameter Varying (LPV) system:
$$
h(k+1) = A(x(k))h(k) + B(x(k))x(k)
$$
$$
y(k) = C(x(k))h(k) + D(x(k))x(k) \quad \text{(Equations 18-19)}
$$

**三个挑战:**
1. $A(x_k), B(x_k), C(x_k)$ 随 input 变化 → Gramians input-dependent
2. $B(x_k), C(x_k)$ 跨 channel 共享 → per-channel balancing 会破坏 shared projection
3. CUDA kernel 写死了 full state dim → 即使 reduce rank 也不省 runtime

### 8.2 Practical Mamba Reduction Workflow

**Step 1: Mean LTI Surrogate**
$$
\bar{A} = \frac{1}{|\mathcal{X}|}\sum_{x \in \mathcal{X}} A(x), \quad \bar{B}, \bar{C} \text{ 同理}
$$

通过 averaging over input space 得到 stationary proxy。

**Step 2: Per-channel Balancing**
对每个 channel $i$, 计算 $P^i, Q^i$ → HSVs $\sigma^i$ → retained dim $r^i$ → balancing transform $T^i$

**Step 3: Store Transforms**
保留 $T^i$ 和 $(T^i)^{-1}$, 在 runtime 应用到 fresh projections:
$$
\bar{B}^i(x_k) = (T^i)^{-1} B(x_k), \quad \bar{C}^i(x_k) = C(x_k) T^i
$$

**Step 4: Rank-aware CUDA kernel**
向量 $\mathbf{r} = (r^1, ..., r^{d_{\text{inner}}})$ 传给 selective-scan kernel, 每个channel loop 在 $r^i$ 终止。

### 8.3 Mamba 实验结果 (Section E.3.3)

**CIFAR10 + Mamba**:
- State dim 128 → 8 几乎无 performance drop (Figure 17)
- 原因: Mamba 是 SISO 架构, state dim 与 global capacity 关系弱

**IMDB + Mamba** (Figure 18):
- COMPRESSM (orange, $\tau=0.001$): 平均 dim 12 from 128, 稳定 competitive performance
- Random dropping (red): 也能 competitive, 但 variance 大
- 这表明 Mamba 的 HSV spectrum 非常 tail-heavy, 大部分能量集中在前几个 dimension

**Speedup (Figure 19)**:
- Dim 128 → 14 with COMPRESSM: training time 约等于始终 dim=16
- 相比 dim=128 baseline: **~4× speedup**

### 8.4 SISO vs MIMO 的根本区别 (Section E.2)

这是理解 COMPRESSM 适用范围的关键:

**MIMO** (如 Griffin, DeltaNet):
- 一个 $n$-dim state 控制整个 $d$-dim feature vector
- $n$ 是 global capacity measure
- 增大 $n$ 直接扩展 shared dynamical subspace
- → COMPRESSM 在这里会 very effective

**SISO** (如 S4, S5, Mamba, Liquid-S4):
- 每个 channel 独立处理, 用独立 1D state-space
- State 只 mediate 单个 input-output mapping
- 增大 per-channel $n$ 的 expressive benefit 被 dilute
- → COMPRESSM 在这里效果有限 (但 speedup 仍有效)

**Future work 方向**: 扩展到 Gated Linear Attention, Mamba2 (MIMO variant), Gated DeltaNet。

References:
- Mamba2 (Dao & Gu, 2024): https://arxiv.org/abs/2405.21060
- Gated DeltaNet: https://arxiv.org/abs/2412.06464
- Griffin (De et al., 2024): https://arxiv.org/abs/2402.19427

---

## 9. 实验细节与 Hyperparameters (Table 3)

| Task | Depth | h (hidden) | n (state) | Steps | Batch | α_LR | WD | Dropout |
|------|-------|------------|-----------|-------|-------|------|-----|--------|
| sMNIST | 1 | 8 | 256 | 200k | 50 | - | - | 0.1 |
| CIFAR10 | 6 | 512 | 384 | 180k | 50 | 0.25 | 0.05 | 0.1 |
| ListOps | 6 | 128 | 256 | 80k | 32 | 0.5 | 0.05 | 0.0 |
| IMDB | 1 | 128 | 256 | 50k | 32 | 0.1 | 0.05 | 0.1 |
| AAN | 6 | 128 | 192 | 100k | 64 | 0.5 | 0.05 | 0.1 |
| Pathfinder | 6 | 192 | 256 | 500k | 64 | 0.25 | 0.05 | 0.0 |

**关键 implementation 细节:**
- Learning rate: warmup 从 $10^{-7}$ → $10^{-3}$ (前 10% steps), 然后 cosine decay 回 $10^{-7}$
- Reduction 在 warmup 期间进行 (4 次 equidistant)
- IMDB 特殊处理: 等 1k steps 后再 reduce, 只做 2 次 reduction 到 3k steps (避免 overfit)
- Reduction 只在 reduced dim < 95% 当前 dim 时执行

---

## 10. 关键 Insight 总结

### 10.1 三层 insight

**Level 1 (操作层)**: 在 training warmup 期间做 balanced truncation, 保留 top-r HSV 对应的 state dimensions。

**Level 2 (理论层)**: Weyl's theorem 保证 HSV continuity, 使我们能 robustly track importance across gradient steps。Empirically, HSV ordering 在 initial transient 后基本 stable, bottom-$r$ cumulative energy 不增长。

**Level 3 (insight)**: **Large model 的 initialization 阶段 提供 valuable inductive bias**, 即使后续 truncate 掉大量 dimensions, 保留的 dimensions 已经"知道"如何 compose 出 effective dynamics。直接 train small model 缺少这种 compositional initialization, 所以效果差。

### 10.2 为什么 "start large, shrink" 优于 "train small"

这与 **lottery ticket hypothesis** 有 conceptual 联系, 但 mechanism 不同:
- Lottery ticket: 大模型中存在 lucky subnetwork, 直接 train small 找不到
- COMPRESSM: 大模型的 HSV spectrum 自然 prune 掉 dead dimensions, 保留的 dimensions 已经经过 joint optimization

更深层的 connection 可能与 **gradient flow in high-dim spaces** 有关: 大 dimension 提供更多 escape routes from saddle points, 优化更容易达到 good basin; truncate 后, 优化已经在 good basin 中, 小 dim 足以 maintain。

### 10.3 Limitations 与 Open Questions

1. **SISO 架构限制**: 在 SISO SSMs (Mamba/S4/S5) 上, state dim 与 global capacity 弱相关, COMPRESSM 主要价值是 speedup 而非性能提升。需要扩展到 MIMO (Mamba2, GLA, DeltaNet)。

2. **HSV spectrum tail-heaviness**: 在 Mamba 上观察到极 tail-heavy 的 HSV 分布, 使 random dropping 也能 competitive。这暗示 control-theoretic importance ranking 在某些 architecture 上可能 overkill。

3. **Selective system 的 LTI surrogate**: averaging over input space 是 crude approximation, 可能 loss important input-dependent dynamics。需要更 principled 的 LTV reduction。

4. **JIT recompilation overhead**: JAX 上每次 reduction 触发 ~5s JIT overhead, 这在小模型 / 短训练上可能 dominate。需要 persistent rank-aware kernel。

5. **理论与实践 gap**: Weyl's theorem 给 per-step continuity, 但 "ordering stability" 和 "bottom-$r$ contribution non-growth" 是 empirical, 没有理论保证。某些 pathological training dynamics 可能 violate 这两条。

---

## 11. 相关工作与延伸阅读

### Control Theory 视角
- **Balanced Truncation**: Antoulas (2005), "Approximation of Large-Scale Dynamical Systems" — MOR 的经典教科书
- **Hankel Singular Values**: 最早由 Glover (1984) 引入 model reduction
- **Moore (1981)**: Principal component analysis of linear systems, 最早提出 balanced realization

### SSM 文献
- **S4** (Gu et al., 2021): https://arxiv.org/abs/2111.00396 — Structured State Space sequence model, diagonal + low-rank structure
- **S5** (Smith et al., 2022): https://arxiv.org/abs/2208.04933 — Simplified SSM, MIMO architecture
- **LRU** (Orvieto et al., 2023): https://arxiv.org/abs/2303.06349 — Linear Recurrent Unit, complex diagonal state matrix
- **Liquid-S4** (Hasani et al., 2022): https://arxiv.org/abs/2209.12951
- **Mamba** (Gu & Dao, 2024): https://arxiv.org/abs/2312.00752 — Selective SSM, SISO LPV
- **Mamba2** (Dao & Gu, 2024): https://arxiv.org/abs/2405.21060 — MIMO variant, structured state space duality
- **Griffin** (De et al., 2024): https://arxiv.org/abs/2402.19427 — MIMO gated linear recurrences + local attention
- **Gated DeltaNet** (Yang et al., 2024): https://arxiv.org/abs/2412.06464 — Delta rule update

### Compression 相关
- **Knowledge Distillation** (Hinton et al., 2015): https://arxiv.org/abs/1503.02531
- **HNN Regularization** (Forgione et al., 2024): https://arxiv.org/abs/2403.14833
- **HNN for MIMO SSMs** (Schwerdtner et al., 2025): https://arxiv.org/abs/2510.22951
- **H₂-optimal reduction for S4** (Sakamoto & Sato, 2025)
- **Post-hoc MOR for S4** (Ezoe & Sato, 2024)

### Linear Attention 视角
- **Linear Attention** (Katharopoulos et al., 2020): https://proceedings.mlr.press/v119/katharopoulos20a.html
- **Gated Linear Attention** (Yang et al., 2023): https://arxiv.org/abs/2312.06635
- **Parallel Delta Rule** (Yang et al., 2025): https://arxiv.org/abs/2406.06484

### Benchmark
- **Long Range Arena** (Tay et al., 2020): https://arxiv.org/abs/2011.04006

---

## 12. 最后的 Intuition Building

把 COMPRESSM 放在更广阔的 framework 中理解:

**Control theory 告诉我们**: 一个 LTI system 的 "essential complexity" 由其 HSV spectrum 决定。Top HSV 对应 dominant modes, bottom HSV 对应 negligible modes。

**Deep learning 告诉我们**: SGD 在 high-dim parameter space 中倾向于 find sparse effective solutions (inductive bias toward simple functions)。

**COMPRESSM 的 contribution 是 bridge 这两个**: 在 SGD 还在 explore phase (warmup) 时, HSV spectrum 已经揭示了 essential modes; truncate 掉 negligible modes 后, SGD 在 smaller, "pre-aligned" 的 subspace 中 exploit, 既快又不会丢 essential dynamics。

**与 Neural Tangent Kernel / Lottery Ticket / Pruning at initialization 的关系**: 这些方法都试图 identify "important" parameters/subspaces, 但 COMPRESSM 的独特之处在于:
1. Importance measure (HSV) 来自 control theory, 有 input-output behavior 的保证
2. 在 training 中而非 training 后做 reduction
3. Reduction 是 structured (state-space level), 不是 individual parameter level

**Potential future extensions I'd love to see**:
1. **Adaptive reduction schedule**: 不是固定 4 次, 而是 data-driven 触发 (当 HSV spectrum stable 后 reduce)
2. **LTV-aware reduction**: 不用 mean surrogate, 而是用 ensemble of Gramians 或 worst-case Gramians
3. **Apply to Transformer / Linear Attention**: linear attention 的 recurrent form 也是 state space, 可能适用
4. **Hardware co-design**: persistent rank-aware kernels, 避免 JIT overhead
5. **Theory**: 证明 HSV ordering stability under SGD (可能需要 NTK / mean-field 分析)

---

希望这个深度解读能帮助你 build 出对 COMPRESSM 的完整 intuition。这个 work 把 control theory 几十年的成熟工具 (balanced truncation, Hankel singular values, Weyl perturbation theory) 精巧地 inject 到 deep learning training pipeline 中, 是 cross-disciplinary 的 nicely executed example。Code 在 https://github.com/camail-official/compressm, 值得 run 一遍 reproduction。
