---
source_pdf: DEBORA EFFICIENT BILEVEL OPTIMIZATION-BASED LOW-RANK ADAPTATION.pdf
paper_sha256: 68e73da213c9204254247d1d5de9564246c2433d5738189b2be9dc1057a4faea
processed_at: '2026-08-18T04:34:34-07:00'
target_folder: Pruning-Quantization
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# dEBORA: 用人话讲讲这篇 paper

Andrej，咱们抛开那些 math derivation，直接从 **engineer 的视角** 来聊聊这篇 paper 到底在干嘛，以及为什么我觉得它挺 elegant 的。

## 1. 一句话总结

dEBORA 就是：**不想手动调 LoRA 的 rank 了吗？那我把它扔进一个 bilevel optimization 框架，让算法自己学出来每一层该用多少 rank，而且计算成本跟普通 LoRA 差不多。**

## 2. 痛点是什么

LoRA 的核心 insight 是：pre-trained model 的 weight update $\Delta W$ 是 low-rank 的。所以我们可以用 $W_0 + BA$ 来 fine-tune，其中 $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$，$r$ 就是 rank。

听起来很美，但有个坑：**$r$ 到底设多少？**

- $r$ 太大：参数多，跟 full fine-tuning 没区别，还容易 overfit
- $r$ 太小：模型 expressiveness 不够，performance 掉
- 更糟的是：**不同 layer 需要的 rank 是不一样的**

工业界现在的做法基本是 **拍脑袋**：$r=8$ 或 $r=16$，所有 layer 一样。这显然不是 optimal 的。

AdaLoRA 试图解决这个问题，思路是：先给一个大的 $r$，然后训练过程中根据某种 "importance score" 逐渐 prune 掉不重要的 singular values。但这个 importance score 是 **heuristic** 的，而且整个 process 有点像 hack。

dEBORA 说：**别 hack 了，这是个 optimization problem，咱们 solve 它。**

## 3. dEBORA 的核心 idea

### 3.1 Bilevel formulation

把 LoRA 的参数分成两组：
- **Singular values** $s = (s_1, \ldots, s_r)$：这些控制 rank budget。如果某些 $s_i \to 0$，对应的 rank 就被 "砍掉" 了。
- **Basis matrices** $B$：就是 LoRA 里的 $U, V$ 矩阵。

然后定义一个 **双层 optimization**：

**Outer loop (upper level)**: 决定 $s$
$$\min_s f_1(B^*(s), s) \quad \text{s.t.} \quad \|s\|_1 \leq \tau, \; s \geq 0$$

**Inner loop (lower level)**: 在给定 $s$ 下，优化 $B$
$$B^*(s) = \arg\min_B f_2(B, s)$$

其中 $f_1, f_2$ 是在 **不同 data split** 上算的 loss，这样可以避免 overfitting（类似 DARTS 的思路）。

**Intuition**: 你可以把 $\tau$ 理解成总 budget，upper level 在分配 budget 给不同 singular values，lower level 拿着分配好的 budget 去训练 basis matrices。Upper level 看哪些 singular values 没用，就把 budget 收回来，分给有用的。

### 3.2 为什么之前没人这么做

Bilevel optimization 听起来美，但有个致命问题：**计算 hypergradient 需要 Hessian**。

具体来说，你要算 $\frac{d}{ds} f_1(B^*(s), s)$，根据 chain rule：

$$\frac{d}{ds} f_1 = \partial_B f_1 \cdot \partial_s B^*(s) + \partial_s f_1$$

那个 $\partial_s B^*(s)$ 是怎么来的？因为 $B^*(s)$ 是 $s$ 的函数。要算它，你需要 solve：

$$\partial_B^2 f_2 \cdot \partial_s B^*(s) = -\partial_s \partial_B f_2$$

注意左边那个 $\partial_B^2 f_2$ —— 这是 **Hessian**！对于 LLM 来说，参数 dimension 是 billions，算 Hessian 是 **impossible** 的。

BiLoRA (2024) 确实用了 bilevel，但它是用 implicit differentiation 硬算的，计算 cost 很高。

### 3.3 dEBORA 的 magic trick

**Key insight**: dEBORA 用了一个特殊的 parameterization，叫 **CP decomposition**：

$$\Psi = \sum_{i=1}^{r} s_i \cdot u_i^{(1)} \otimes u_i^{(2)} \otimes \cdots \otimes u_i^{(d)}$$

这里 $s_i$ 是 scalar，$u_i^{(j)}$ 是第 $j$ 个 mode 的 basis vector。

这个 parameterization 有个 **magical property**：$\Psi$ 对每个 basis matrix $U^{(j)}$ 是 **multilinear** 的。这意味着 Hessian $\partial_{U^{(j)}}^2 \Psi = 0$ —— 对角块是零！

所以那个 implicit gradient system 简化为：

$$\sum_{j \neq i} \partial_{U^{(i)}, U^{(j)}}^2 \Psi \cdot \partial_s U^{(j)} = -\partial_s \partial_{U^{(i)}} \Psi$$

对于 matrix case ($d=2$)，这个系统可以 **closed-form solve**：

$$\partial_{s_\eta} U_{j\beta}^{(i)} = -\delta_{\beta\eta} U_{j\beta}^{(i)} s_\beta^{-1}$$

变量解释：
- $s_\eta$ 是第 $\eta$ 个 singular value
- $U_{j\beta}^{(i)}$ 是第 $i$ 个 basis matrix 的第 $j$ 行第 $\beta$ 列
- $\delta_{\beta\eta}$ 是 Kronecker delta，只有 $\beta = \eta$ 时为 1，否则 0

**这个公式说的是：basis matrix 对 singular value 的导数，正比于自身除以 singular value**。非常 elegant！

把这个代回 hypergradient 公式，得到 **Hessian-free** 的近似：

$$G(s) = \partial_s f_1^* - \text{diag}\left[\sum_{i} U^{*(i),\top} \partial_{U^{(i)}} f_1^*\right] \odot s^{-1}$$

变量解释：
- $\partial_s f_1^*$ 是 $f_1$ 对 $s$ 的直接偏导（容易算）
- $U^{*(i),\top}$ 是最优 basis matrix 的转置
- $\partial_{U^{(i)}} f_1^*$ 是 $f_1$ 对 basis 的偏导（一次 backprop 就有）
- $\odot$ 是逐元素乘
- $s^{-1}$ 是逐元素倒数

**你只需要一次 backprop，就能算出 hypergradient！** 不需要 Hessian，不需要 conjugate gradient，不需要 implicit differentiation。

**误差界**: $\|G(s) - \text{true hypergradient}\| \lesssim K\beta$

变量解释：
- $K$ 是 loss landscape 的 Hessian norm bound (越小说明 landscape 越平滑)
- $\beta$ 是某个 operator 的 invertibility bound

这个 error bound 告诉我们：**只要 loss landscape 不太崎岖 ($K$ 小)，且参数化结构 well-conditioned ($\beta$ 小)，近似就是好的**。

## 4. Frank-Wolfe: 为什么用它

现在有了 hypergradient $\tilde{G}(s)$，怎么更新 $s$？

约束是 $\|s\|_1 \leq \tau, s \geq \varepsilon$，这是一个 **simplex**。

通常的做法是 projected gradient descent：走一步，然后 project 回 simplex。但 projection 在 simplex 上虽然不难，也不算便宜。

**Frank-Wolfe** 是个更聪明的选择。它的核心 idea 是：**不 project，而是解一个 linear subproblem**。

每步：
1. **Frank-Wolfe direction**: 找一个 simplex vertex $z_n$ 使得 $\tilde{G}(s_n)^\top z_n$ 最小
   - 在 simplex 上，这就是找 $\tilde{G}(s_n)$ 最小的 entry
   - Cost: $O(r)$，非常便宜
2. **Away-step direction**: 找当前 active set 里，使得 $\tilde{G}(s_n)^\top y_n$ 最大的 vertex
   - 这是 "踢走" 不重要 component 的方向
3. 选下降更快的方向
4. Line search 找步长
5. **Truncation**: 当某些 $s_i < \varepsilon$，直接砍掉，rank 自动降

**为什么 Frank-Wolfe 适合这里**：
1. **Projection-free**: 在 simplex 上，linear minimization oracle 几乎免费
2. **Sparse solutions**: Frank-Wolfe 天然产生 sparse iterates，对应 rank reduction
3. **Identifiability**: Away-step variant 能在有限步内识别 optimal face，保证找到正确的 rank structure

## 5. 收敛性: 它真的会收敛吗

Paper 给了两个 main theorems：

### Theorem 6.2: Sublinear convergence

假设 stochastic hypergradient $\tilde{G}$ 的 variance 有界 (Assumption 6.1)：

$$\mathbb{E}[e(s, \bar{s})^2] \leq \chi^2$$

其中 $e(s, \bar{s}) = (\nabla f(\bar{s}) - \tilde{G}(\bar{s}))^\top (s - \bar{s})$ 是 gradient error。

如果 $\chi$ 足够小 (跟 batch size 有关)，那么：

$$\mathbb{E}[g_T^*] \leq \sqrt{\frac{2\Delta^2 M (f(s_0) - f^*)}{T \rho (1-\eta)^2}}$$

变量解释：
- $g_T^*$ 是 Frank-Wolfe gap 的最小值 (收敛到 0 意味着达到 stationary point)
- $\Delta$ 是 feasible set 的 diameter
- $M$ 是 gradient 的 Lipschitz 常数
- $T$ 是 iteration 数
- $\rho, \eta$ 是算法参数

**Rate**: $O(1/\sqrt{T})$，这是 non-convex stochastic optimization 的 standard rate。

### Theorem 6.5: Identifiability

这个更酷：**一旦 algorithm 接近 stationary point，它会 "stick" 在 optimal face 上**。

意思是：如果 optimal solution 是 sparse 的 (只有几个 $s_i$ 非零)，algorithm 会在有限步内 **确定性地识别** 这个 sparse structure，之后只在那个 low-dimensional face 上优化。

这跟 AdaLoRA 的 heuristic pruning 有本质区别：**AdaLoRA 可能 prune 错，dEBORA 有理论保证**。

条件是 strict complementarity:
$$\lambda_v^{\text{MIN}}(s^*) - 2\chi > 0$$

其中 $\lambda_v^{\text{MIN}}(s^*)$ 是 stationary point 处 active set 外 vertices 的最小 "远离度"。

## 6. 实验: 真的好用吗

### GLUE benchmark (DeBERTaV3-base)

| Method | # Params | Avg Performance |
|--------|----------|-----------------|
| Full FT | 184M | baseline |
| AdaLoRA | 1.27M | comparable |
| **dEBORA** | **0.4M** | **comparable** |

**dEBORA 用 3 倍少的参数，达到类似 performance**。

### ResNet50 on CIFAR-10

| Method | Accuracy | # Params |
|--------|----------|----------|
| AdaLoRA | 93.28% | 316K |
| **dEBORA** | **93.74%** | **64K** |

**5 倍参数 reduction，accuracy 还更高**。

### Stable Diffusion

dEBORA 用 0.4M params，loss 0.231；AdaLoRA 用 4.7M params，loss 0.245。

## 7. 计算成本

Per-iteration cost: $\mathcal{O}(T_{\max} C_r + L r n^2)$

变量解释：
- $T_{\max}$: lower-level optimization 的 iteration 数 (user-defined)
- $C_r$: rank-$r$ 的一次 lower-level step 成本 (跟普通 LoRA step 一样)
- $L$: layer 数
- $n$: layer dimension

对比：
- LoRA: $\mathcal{O}(C_r)$
- AdaLoRA: $\mathcal{O}(C_r + L r n^2)$

dEBORA 单步比 AdaLoRA 贵 $T_{\max}$ 倍，但实际 GPU time 可比，因为：
1. Memory consumption 更低 (Figure 2)
2. GPU power usage 更低 (Figure 3)
3. Time-to-accuracy 可比 (Figure 1)

参考 [arXiv paper](https://arxiv.org/abs/2407.05671)。

## 8. 我的 take

### 8.1 我喜欢的点

1. **Structure exploitation**: CP decomposition 的 multilinearity 被用来 avoid Hessian，这是 mathematical insight 驱动 algorithmic efficiency 的好例子。

2. **Frank-Wolfe 的选择**: 在 simplex 上 projection-free，而且天然 sparse，跟 rank selection 问题完美 match。

3. **Theory + Practice**: 不只是 algorithm，还有 convergence analysis 和 identifiability guarantee，实验也 solid。

### 8.2 我的 concern

1. **$T_{\max}$ 的选择**: lower-level optimization 要跑多少步才够？这是个 hyperparameter，可能需要 tuning。

2. **Closed-form approximation 的 accuracy**: Theorem 4.1 的 error bound 是 $\lesssim K\beta$，但实际中 $K$ 和 $\beta$ 多大？对于 LLM 这种 deep architecture，loss landscape 可能很崎岖，$K$ 可能不小。

3. **Non-convexity**: 整个 bilevel problem 是 non-convex 的，convergence 只到 stationary point，不保证 global optimum。虽然 experiment 表明 practical performance 好，但理论 guarantee 弱于 convex case。

4. **Comparison fairness**: AdaLoRA 的 importance score 是用 SVD sensitivity，这个 metric 本身可能 suboptimal。dEBORA 用 bilevel framework 直接 optimize rank allocation，理论上更 principled，但实验中 AdaLoRA 也 tune 得不错，gap 不大。

### 8.3 联想

1. **跟 Neural Architecture Search 的 connection**: Bilevel optimization 在 NAS (DARTS) 里很流行，dEBORA 把这个思路用到 PEFT 里，可以看成 "architecture search for rank"。

2. **跟 Lottery Ticket Hypothesis 的 connection**: dEBORA 找 sparse structure，跟 lottery ticket 的 idea 有呼应。区别是 dEBORA 是动态 find sparse structure，lottery ticket 是 post-hoc prune。

3. **跟 Mixture of Experts 的 connection**: 不同 rank allocation 给不同 layer，可以看成 "expert allocation" 的一种形式。

4. **Extension 到 Riemannian**: Paper 提到可以 extend 到 Riemannian setting (Stiefel/oblique manifold)，实验里有 oblique 和 Stiefel 版本。这跟 [Zangrando et al., 2024](https://arxiv.org/abs/2407.05671) 的 Riemannian rank-adaptive training 有呼应。

## 9. 总结

dEBORA 是一个 **theory-guided** 的 rank adaptation 方法。它的核心 contribution 是：

1. **Bilevel formulation**: 把 rank selection 变成 optimization problem
2. **Hessian-free hypergradient**: 利用 CP structure 的 multilinearity
3. **Frank-Wolfe with away-step**: Projection-free, sparse solutions, identifiability
4. **理论保证**: Sublinear convergence + face identification

实验表明它能在 **显著更少参数** 下达到 comparable performance，computation cost 可控。

对于 practitioner 来说，如果你觉得 LoRA 的 rank tuning 是个 pain point，dEBORA 提供了一个 principled 的替代方案。当然，如果你 fine-tune 规模不大，普通 LoRA + grid search rank 可能就够了；但对于大规模 fine-tuning (e.g., LLM adaptation)，dEBORA 的 automated rank allocation 可能 worth the extra implementation complexity。

参考资源：
- [dEBORA Paper (arXiv)](https://arxiv.org/abs/2407.05671)
- [PEFT Library (HuggingFace)](https://github.com/huggingface/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [AdaLoRA Paper](https://arxiv.org/abs/2303.10512)
- [Frank-Wolfe Survey](https://arxiv.org/abs/2211.14103)
- [Bilevel Optimization Survey](https://arxiv.org/abs/2106.08966)
- [BiLoRA Paper](https://arxiv.org/abs/2403.13037)
- [DARTS Paper](https://arxiv.org/abs/1806.09055)

---

# dEBORA: 基于 Bilevel Optimization 的高效低秩适应方法深度解析

## 1. 核心动机与问题定义

让我先从 **直觉角度** 来剖析这篇 paper 的核心贡献。传统 LoRA 方法存在一个根本性矛盾：我们需要在 **模型性能** 和 **参数效率** 之间权衡，而这通过 **静态 rank 配置** 是无法达到帕累托最优的。dEBORA 的核心 insight 是：将 rank selection 问题 formulate 为一个 **bilevel optimization** 问题，其中 rank 参数 $s$ 作为上层变量，basis matrices $B$ 作为下层变量。

这让我想到一个有趣的类比：整个架构类似 **Federal System**，中央政府 (upper level) 分配预算给各州，各州政府 (lower level) 用这些预算优化本地发展，而中央政府根据各州表现来动态调整预算分配。

### 1.1 参数化结构

对于 matrix case ($d=2$)，低秩更新可以表示为：
$$\Psi = U S V^\top = \sum_{i=1}^{r} s_i u_i \otimes v_i$$

其中：
- $U, V \in \mathbb{R}^{n \times r}$ 是 low-rank basis matrices
- $S = \text{diag}(s_1, \ldots, s_r)$ 是 diagonal singular value matrix
- $r$ 是 rank，直接控制可训练参数数量
- $u_i, v_i$ 分别是 $U, V$ 的第 $i$ 列

对于 tensor case ($d > 2$)，使用 CP (Canonical Polyadic) decomposition：
$$\Psi(B, s) = \sum_{i=1}^{r} s_i u_i^{(1)} \otimes u_i^{(2)} \otimes \cdots \otimes u_i^{(d)}$$

其中 $B = (U^{(1)}, \ldots, U^{(d)})$ 是 basis matrices 的集合，$s = (s_1, \ldots, s_r) \geq 0$ 是 scaling factors。

这种参数化的优势在于：
1. **参数数量扩展性更好**：对于卷积层，LoRA 需 $O(r(F + Ck^2))$ 参数，而 dEBORA 的 CP-like factorization 只需 $O(r(2k + C + F))$
2. **Rank 与 basis 解耦**：可以独立优化 $s$ 和 $B$
3. **天然适合 rank adaptation**：$s$ 的稀疏性直接对应 rank reduction

## 2. Bilevel Optimization 框架

### 2.1 问题形式化

给定数据集划分为 $D_1$ 和 $D_2$ 两部分，定义两个 loss function：
$$f_i(B, s) = \frac{1}{|D_i|} \sum_{(x,y) \in D_i} \ell(B, s; x, y), \quad i = 1, 2$$

Bilevel optimization 问题形式化为：

$$\min_{s \in \mathbb{R}^r: s \geq 0, \|s\|_1 \leq \tau} f_1(B^*(s), s)$$
$$\text{s.t. } B^*(s) \in \arg\min_{B \in \mathcal{V}} f_2(B, s)$$

其中 $\tau > 0$ 是控制 singular values 稀疏性的正则化参数。

**关键 intuition**：
- Upper level: 决定每层的 rank budget allocation ($s$)
- Lower level: 在给定 $s$ 下优化 basis matrices ($B$)
- $\|s\|_1 \leq \tau$ 限制总 rank budget，促进稀疏性

### 2.2 Hypergradient 计算的挑战

传统方法需要计算 hypergradient：
$$\frac{d}{ds} f_1(B^*(s), s) = \partial_B f_1(B^*(s), s) \partial_s B^*(s) + \partial_s f_1(B^*(s), s)$$

难点在于 $\partial_s B^*(s)$，需要求解 **implicit gradient system**：
$$\partial_B^2 f_2(B^*(s), s) \partial_s B^*(s) = -\partial_s \partial_B f_2(B^*(s), s)$$

这要求计算 Hessian matrix $\partial_B^2 f_2$，对于大型神经网络是 **computationally prohibitive** 的。

## 3. Theorem 4.1: Hessian-Free Hypergradient Approximation

这是 paper 的 **核心理论贡献**。通过利用参数化的特殊结构，推导出一个 closed-form 近似，完全避免 Hessian 计算。

### 3.1 Matrix Case ($d=2$)

**假设**：
1. **Locally approximately constant gradient**: 存在 $K \geq 0$ 使得
   $$\|\nabla^2 L_2(\Psi(B^*(s), s))\| \leq K, \quad \forall s: \|s\|_1 \leq \tau$$
   这意味着 loss landscape 在 $\Psi$ 空间局部平滑。

2. **Bilinear operator uniformly invertible and bounded**:
   $$\|(\partial_\Psi L_2^* \partial_B^2 \Psi^*)^{-1}\| \leq \beta$$
   这保证了 implicit gradient system 的适定性。

**结论**: Closed-form hypergradient approximation:
$$G(s) := \partial_s f_1^* - \text{diag}\left[U^{*(1),\top} \partial_{U^{(1)}} f_1^* + U^{*(2),\top} \partial_{U^{(2)}} f_1^*\right] \odot s^{-1}$$

其中：
- $\odot$ 是 Hadamard product (逐元素乘)
- $s^{-1}$ 是逐元素倒数
- $U^{*(i),\top}$ 是第 $i$ 个 basis matrix 的转置
- $\partial_{U^{(i)}} f_1^*$ 是 $f_1$ 对第 $i$ 个 basis 的偏导数

**误差界**:
$$\|G(s) - \frac{d}{ds} f_1(B^*(s), s)\| \lesssim K\beta$$

### 3.2 Tensor Case ($d > 2$)

使用 **tridiagonal approximation** of $\partial_B^2 \Psi^*$:
$$G(s) := \partial_s f_1^* - \text{diag}\left[\sum_{i=1}^{d} U^{*(i),\top} \partial_{U^{(i)}} f_1^*\right] \odot s^{-1}$$

**误差界**:
$$\|G(s) - \frac{d}{ds} f_1(B^*(s), s)\| \lesssim K\beta + \sum_{i=1}^{d} \sum_{|j-i| \geq 2} \|\partial_{U^{(i)}, U^{(j)}}^2 \Psi^*\|$$

### 3.3 直觉解释

让我来 **build intuition** 为什么这个近似 work：

**关键 insight**: 由于 $\Psi(B, s)$ 对 basis matrices $B$ 是 **multilinear** 的，Hessian $\partial_B^2 f_2$ 中 **diagonal terms 为零**，即 $\partial_{U^{(i)}}^2 \Psi = 0$。这意味着主要的信息来自于 cross terms $\partial_{U^{(i)}, U^{(j)}}^2 \Psi$。

对于 $d=2$，系统可以 **精确求解** in closed form：
$$\partial_{s_\eta} U_{j\beta}^{(2)} = -\delta_{\beta\eta} U_{j\beta}^{(2)} s_\beta^{-1}$$
$$\partial_{s_\eta} U_{j\beta}^{(1)} = -\delta_{\beta\eta} U_{j\beta}^{(1)} s_\beta^{-1}$$

其中 $\delta_{\beta\eta}$ 是 Kronecker delta。这给出了一个非常 elegant 的结果：basis matrix 对 singular value 的导数 **正比于自身除以 singular value**。

对于 $d > 2$，需要 tridiagonal approximation 来处理 cross terms，但主要结构保持不变。

## 4. 约束修改与 Stochastic Frank-Wolfe

### 4.1 扰动 $L^1$ Simplex

为了避免 $s_i \to 0$ 时的数值不稳定性，修改约束为：
$$S := \{s \in \mathbb{R}^r | s \geq \varepsilon, \|s\|_1 \leq \tilde{\tau}\}$$

其中 $\tilde{\tau} = \tau + r\varepsilon$，$\varepsilon > 0$ 是小常数。这仍然是 convex set，适合 projection-free 方法。

### 4.2 Stochastic Away-Step Frank-Wolfe 算法

**核心思想**: Frank-Wolfe 算法通过 linear minimization oracle 在 convex set 上优化，**避免 projection**。Away-step variant 处理 non-vertex optimal solutions。

#### 算法步骤

**Step 1: Frank-Wolfe Direction**
$$h_n = z_n - s_n, \quad z_n = \arg\min_{s \in S} \tilde{G}(s_n)^\top s$$

由于 $S$ 是 sparse simplex，$z_n$ 通过查看 $\tilde{G}(s_n)$ 的 entries 高效计算。

**Step 2: Convergence Criterion**
$$\text{Stop if } -\tilde{G}(s_n)^\top h_n \leq \tilde{p}$$

**Step 3: Away-Step Direction**
$$b_n = s_n - y_n, \quad y_n = \arg\max_{s \in C_n} \tilde{G}(s_n)^\top s$$

其中 $C_n = \{s \in S : \text{supp}_\varepsilon(s) = \text{supp}_\varepsilon(s_n)\}$ 是当前 active set。

**Step 4: Steepest Direction**
$$d_n = \begin{cases} h_n & \text{if } -\tilde{G}(s_n)^\top h_n \geq -\tilde{G}(s_n)^\top b_n \\ b_n & \text{otherwise} \end{cases}$$

**Step 5: Line Search**
$$s_{n+1} = s_n + \alpha_n d_n$$

**Step 6: Truncation** (当 $n \geq n_0$)
移除 $s_{n+1}$ 中小于 $\varepsilon$ 的 entries，相应减少 $U, V$ 的列数，即 **动态降低 rank**。

### 4.3 直觉解释

Frank-Wolfe 算法可以理解为 **greedy expert** 策略：
- 每步选择当前看来最有价值的方向（Frank-Wolfe direction: 最小化 linear approximation）
- 同时考虑远离当前最差方向（Away-step: 最大化 linear approximation）
- 选择下降更快的方向

**关键优势**:
1. **Projection-free**: 在 simplex 上，linear minimization oracle 只需 $O(r)$ 计算
2. **Sparse solutions**: 自然产生稀疏解，对应 rank reduction
3. **Identifiability**: 能在有限步内识别 optimal face

## 5. 理论保证

### 5.1 Theorem 6.2: 收敛性

**假设 6.1**: Stochastic approximation $\tilde{G}$ 满足
$$\mathbb{E}[e(s, \bar{s})^2] \leq \chi^2, \quad \forall s \in S$$

其中 $e(s, \bar{s}) = (\nabla f(\bar{s}) - \tilde{G}(\bar{s}))^\top (s - \bar{s})$。

**结论**: 若 $\chi \leq \frac{\eta}{2+2\eta}\tilde{p}$ ($0 \leq \eta < 1/3$)，step size 满足 (8)-(10)，则
$$\mathbb{E}[g_T^*] \leq \sqrt{\frac{2\Delta^2 M (f(s_0) - f^*)}{T \rho (1-\eta)^2}}$$

其中 $g_T^* = \min_{0 \leq n \leq T-1} g_n^{FW}$ 是 Frank-Wolfe gap 的最小值，$\Delta$ 是 $S$ 的直径，$M$ 是 gradient 的 Lipschitz 常数。

**收敛速率**: $O(1/\sqrt{T})$，sublinear rate，对于非凸 objectives with Lipschitz gradient。

### 5.2 Theorem 6.5: Identifiability

**关键定义**:
- **Exposed face**: $\mathcal{F}_e(\nabla f(s)) = \arg\min_{z \in S} \nabla f(s)^\top z$
- **Minimal face**: $\mathcal{F}(s)$ 包含 $s$ 的最小 face
- $\lambda_v(s) = \nabla f(s)^\top (v - s)$
- $\lambda_v^{\text{MIN}}(s^*) = \min_{v \in V^+(S)} \lambda_v(s^*)$

**假设**:
1. $\lambda_v^{\text{MIN}}(s^*) - 2\chi > 0$ (strict complementarity)
2. $\mathcal{F}_e(\nabla f(s^*)) = \mathcal{F}(s^*)$ (strict complementarity)

**结论**: 存在 $\Gamma(s^*) > 0$ 使得若 $s_n \in B_{\Gamma(s^*)} \cap \mathcal{F}(s^*)$，则 $s_{n+1} \in \mathcal{F}(s^*)$。

**重要性**: 一旦算法接近 stationary point，会 **识别** 包含它的 face，从而 **降维** 问题，可以使用更精细的优化方法。

## 6. 实验结果分析

### 6.1 GLUE Benchmark (Table 1)

在 DeBERTaV3-base 上的结果：

| Method | # Params (M) | MNLI | SST-2 | CoLA | QQP | QNLI | RTE | MRPC | STS-B |
|--------|-------------|------|-------|------|-----|------|-----|------|-------|
| Full FT | 184 | 89.90 | 95.63 | 69.19 | 92.40/89.80 | 94.03 | 83.75 | 89.46 | 91.60 |
| AdaLoRA | 1.27 | 90.44 | 95.64 | 68.76 | 90.59/90.65 | 94.11 | 86.00 | 89.44 | 91.41 |
| dEBORA τ=16 | 0.4 | 90.01 | 95.29 | 68.72 | 91.88/89.20 | 93.42 | 83.75 | 90.16 | 90.84 |
| dEBORA (Stiefel) τ=16 | 0.8 | 89.79 | 95.65 | 68.39 | 89.74/86.58 | 93.83 | 84.12 | 91.18 | 91.54 |

**关键观察**:
1. dEBORA 用 **0.4M 参数** (比 AdaLoRA 少 3 倍) 达到 comparable performance
2. 在 MRPC 上 dEBORA 超过 AdaLoRA (90.16 vs 89.44)
3. Stiefel 约束版本更稳定但参数稍多

### 6.2 Tensor Layers (Table 2)

**ResNet50 on CIFAR-10**:
| Method | Val. Acc (%) | # Params |
|--------|-------------|----------|
| LoRA (r=16) | 89.66 | 1.17M |
| AdaLoRA | 93.28 | 316K |
| dEBORA τ=8 | 93.74 | 64K |

dEBORA 用 **64K 参数** (比 AdaLoRA 少 5 倍) 达到 **更高准确率** (93.74% vs 93.28%)！

**Stable Diffusion**:
| Method | Loss | # Params |
|--------|------|----------|
| AdaLoRA | 0.245 | 4.7M |
| dEBORA (Oblique) τ=16 | 0.231 | 0.4M |

dEBORA 用 **0.4M 参数** (比 AdaLoRA 少 12 倍) 达到 **更低 loss** (0.231 vs 0.245)。

## 7. 计算复杂度分析 (Appendix G)

**Per-iteration cost**:
$$\mathcal{O}(T_{\max} C_r + L r n^2)$$

其中：
- $T_{\max}$: lower-level optimization 的最大迭代数
- $C_r$: rank-$r$ 下的一次 lower-level step 成本
- $L$: 层数
- $r$: rank
- $n$: 层维度

**对比**:
- LoRA: $\mathcal{O}(C_r)$
- AdaLoRA: $\mathcal{O}(C_r + L r n^2)$ (包括 sensitivity metric 计算)

dEBORA 单次迭代更贵 (多 $T_{\max}$ 因子)，但：
1. GPU memory consumption 更低
2. Average GPU power usage 更低
3. Time-to-accuracy 可比

## 8. 与相关工作的关系

### 8.1 Bilevel Optimization in Deep Learning

dEBORA 与传统 bilevel optimization 的关键区别：
- **避免 implicit differentiation**: 利用参数化结构得到 closed-form
- **Stochastic Frank-Wolfe**: projection-free，适合 large-scale
- **Identifiability guarantees**: 有限步识别 optimal structure

相关参考：
- [Franceschi et al., 2018](https://proceedings.mlr.press/v70/franceschi17a.html): BPTT for hyperparameter optimization
- [Lorraine et al., 2020](https://proceedings.mlr.press/v108/lorraine20a.html): Implicit differentiation for millions of hyperparameters
- [Grazzi et al., 2020](https://proceedings.mlr.press/v119/grazzi20a.html): Approximate implicit differentiation

### 8.2 Low-Rank Adaptation Methods

- [LoRA (Hu et al., 2022)](https://openreview.net/forum?id=nZeVKeeFYf9): 原始 LoRA 方法
- [AdaLoRA (Zhang et al., 2023)](https://arxiv.org/abs/2303.10512): Adaptive budget allocation
- [DyLoRA (Valipour et al., 2023)](https://aclanthology.org/2023.eacl-main.284/): Dynamic rank during training
- [BiLoRA (Qiang et al., 2024)](https://arxiv.org/abs/2403.13037): Bi-level optimization with implicit differentiation

### 8.3 Frank-Wolfe Algorithms

- [Guélat & Marcotte, 1986](https://link.springer.com/article/10.1007/BF01582143): Original away-step FW
- [Bomze et al., 2020](https://epubs.siam.org/doi/10.1137/19M1259213): Active set complexity
- [Braun et al., 2022](https://arxiv.org/abs/2211.14103): Conditional gradient methods survey

## 9. Intuition Building: 为什么 dEBORA Work?

让我从几个角度来 **build intuition**:

### 9.1 参数化结构的妙用

CP decomposition 的 **multilinear** 性质是关键：
- $\Psi$ 对每个 $U^{(i)}$ 是线性的
- Hessian 的 diagonal blocks 为零
- Cross terms 虽然存在，但在 $d=2$ 时可以 **精确处理**

这让我想到 **tensor network** 的思想：高阶结构通过低秩分解，计算复杂度从 exponential 降到 polynomial。

### 9.2 Bilevel 的解耦效应

将 $s$ 和 $B$ 分到不同 level 有几个好处：
1. **避免 overfitting**: $f_1$ 和 $f_2$ 在不同数据上，类似 DARTS 的思路
2. **Gradient flow 更清晰**: $s$ 的更新基于 $B^*(s)$ 的响应
3. **Rank adaptation 有理论依据**: 不仅仅是 heuristic pruning

### 9.3 Frank-Wolfe 的稀疏性

Frank-Wolfe 在 polytope 上优化天然产生 **sparse solutions**：
- 每步只激活一个 vertex
- Away-step 移除不重要的 active vertex
- 最终 $s$ 在 simplex vertices 上稀疏分布

这直接对应 **rank selection**：不重要的 singular values 被推向零，对应 rank reduction。

### 9.4 Identifiability 的几何意义

Theorem 6.5 说一旦接近 stationary point，算法会 **stick to the optimal face**。几何上：
- Simplex 的每个 face 对应一个 active set
- Strict complementarity 保证 unique optimal face
- Away-step 确保 active set 不膨胀

这意味着 dEBORA 能 **确定性地** 找到每层的 optimal rank，而 AdaLoRA 的 pruning 是 heuristic 的。

## 10. 实践建议与未来方向

### 10.1 实践考虑

1. **$\tau$ 的选择**: 控制 total rank budget，需要根据模型大小调整
2. **$\varepsilon$ 的选择**: 防止数值不稳定，通常 $10^{-4}$ 量级
3. **$T_{\max}$ 的权衡**: 更大更准确但更慢
4. **Manifold 约束**: Stiefel 更稳定但需要 retraction，Oblique 是折中

### 10.2 局限性

1. **Closed-form 近似误差**: $K\beta$ 项可能在大模型上不 negligible
2. **Lower-level 优化成本**: $T_{\max}$ 因子可能成为 bottleneck
3. **非凸 landscape**: 收敛到 stationary point 不保证 global optimum

### 10.3 未来方向

1. **Riemannian extension**: 论文提到可以推广到 Riemannian setting
2. **Online setting**: 适应 streaming data
3. **Multi-task**: 跨任务的 rank allocation
4. **Combination with quantization**: 类似 QLoRA

## 11. 个人 Reflection

作为深度学习研究者，我觉得 dEBORA 的 **理论 elegance** 和 **实践 efficiency** 的结合非常 impressive。几个让我思考的点：

1. **Closed-form approximation 的 power**: 通过利用问题结构避免 expensive computation，这是 **mathematics-informed ML** 的好例子

2. **Frank-Wolfe 的 revival**: 在大规模 ML 中，projection-free 方法重新获得关注，因为 projections 可能很贵

3. **Bilevel thinking**: 将 hyperparameter tuning formulate 为 optimization 而非 grid search，是 **automated ML** 的正确方向

4. **Identifiability 的重要性**: 不仅仅是收敛，而是 **identifying optimal structure**，这对 interpretability 也很重要

参考资源：
- [Original Paper (arXiv)](https://arxiv.org/abs/2407.05671)
- [PEFT Library](https://github.com/huggingface/peft)
- [Frank-Wolfe Algorithms](https://arxiv.org/abs/2211.14103)
- [Bilevel Optimization Survey](https://arxiv.org/abs/2106.08966)

这篇 work 让我看到了 **optimization theory** 和 **deep learning practice** 的深度融合，通过 mathematical structure 来 reduce computational complexity，这是解决大规模 ML 问题的关键 path。
