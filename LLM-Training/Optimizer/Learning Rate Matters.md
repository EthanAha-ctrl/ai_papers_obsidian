---
source_pdf: Learning Rate Matters.pdf
paper_sha256: 4b2e04a6b54a5864807cfd1a6fc84033fa92e9b313ae8b34df6545c87bce5558
processed_at: '2026-08-05T13:39:13-07:00'
target_folder: LLM-Training/Optimizer
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版

兄弟，这篇paper说白了就一句话：**大家都在卷 LoRA 的新花样，结果发现把 learning rate 调对了，vanilla LoRA 跟你那堆花里胡哨的 variant 差不了一两个点。**

---

## 故事是这样的

这两年 LoRA 的 variants 爆炸式增长。PiSSA 说 "我用 SVD 的 top singular values 来初始化，比 LoRA 强 10%！" DoRA 说 "我把 magnitude 和 direction 拆开，强 37%！" 一堆人都在发 paper 声称自己吊打 vanilla LoRA。

但这篇作者发现一个尴尬的事实：翻了 52 篇 LoRA 相关 paper，**只有 1 篇老老实实 tune 了 learning rate、batch size、rank 三个东西**。不到 30% 的人 tune 了 learning rate。

这啥意思呢？就是你拿你的新方法，精心调了 hyperparameter，然后拿 vanilla LoRA 用默认参数跑一遍，说 "看，我比他强 10%"。这不公平啊。

---

## 做了啥实验

作者拿了 5 个方法 — LoRA、PiSSA、MiLoRA、Init[AB]、DoRA，在 Qwen3-0.6B、Gemma-3-1B、Llama-2-7B 上跑 math 和 code 任务。

Learning rate 从 $10^{-6}$ 到 $10^{-3}$ 扫了 12-16 个点，batch size 试了 16/64/128，rank 从 4 到 256。

结果呢？**所有方法的 peak performance 差距只有 0.43% 到 1.75%**。

比如 Qwen 上跑 math，$r=128$：

| 方法 | 最好成绩 | 最优 learning rate |
|------|---------|-------------------|
| LoRA | 49.60% | $2 \times 10^{-4}$ |
| PiSSA | 49.51% | $6.3 \times 10^{-5}$ |
| MiLoRA | 49.40% | $1.1 \times 10^{-4}$ |
| DoRA | 49.45% | $2 \times 10^{-4}$ |
| Init[AB] | 49.29% | $1.1 \times 10^{-4}$ |

你看，差距才 0.43%，基本就是 noise level。

**但关键在于：每个方法的最优 learning rate 不一样**。PiSSA 需要比 LoRA 小 3 倍左右的 learning rate。你要是都用同一个 learning rate 跑，那 PiSSA 当然显得不行，或者 LoRA 当然显得不行。

---

## 为啥每个方法的最优 learning rate 不一样

这是这篇 paper 最有价值的部分。作者用 Hessian 的最大 eigenvalue $\lambda_{\text{max}}$ 来解释。

直觉是这样的：learning rate 的上限取决于 loss surface 在你当前位置有多 "sharp"。越 sharp，你步子迈大了就容易飞出去，得用小 learning rate。越 flat，你可以迈大步。

$$\eta^* \propto \frac{1}{\lambda_{\text{max}}}$$

那不同 LoRA variant 的 $\lambda_{\text{max}}$ 为啥不一样呢？

- **PiSSA**：用 top-$r$ singular values 初始化，这些 $\sigma_i$ 都很大，参数 perturb 一下 loss 就剧烈变化，$\lambda_{\text{max}}$ 大，得用小 learning rate
- **MiLoRA**：用 bottom-$r$ singular values，都是些很小的值，perturb 一下 loss 几乎不动，$\lambda_{\text{max}}$ 小，可以用大 learning rate
- **Vanilla LoRA**：$BA=0$，介于两者之间

作者实测了 Hessian，PiSSA 的 $\lambda_{\text{max}}$ 比 LoRA 大约一个数量级，完美解释了为啥它需要更小的 learning rate。

---

## 还有个有意思的发现

PiSSA 在大 learning rate 下特别 robust。比如 $\eta = 1.1 \times 10^{-3}$ 的时候，其他方法都崩了（accuracy 掉到 0），PiSSA 还能保持 27% 左右。

为啥？因为 PiSSA 的 $\lambda_{\text{max}}$ 大，所以它的 "divergence threshold" $2/\lambda_{\text{max}}$ 对应的 learning rate 也大。别的方法已经飞出去了，PiSSA 还在 catapult regime 里头。

---

## Rank 的 effect

不同 rank 下，方法的相对表现会变：

- **PiSSA**：低 rank 时比 LoRA 差（差 1.67%），高 rank 时比 LoRA 好一点（好 0.2-0.3%）
- **MiLoRA**：反过来，低 rank 好一点，高 rank 差一点
- **DoRA**：低 rank 时确实比 LoRA 好 1% 左右，但比原 paper 报告的小很多
- **Init[AB]**：中等 rank 最好

所以没有哪个方法 universal 地强，都是看 rank regime 的。

---

## Batch size 的事

作者还发现，learning rate 比 batch size 重要得多。固定 learning rate 只 tune batch size，效果很差。固定 batch size tune learning rate，效果好得多。

而且最优 learning rate 跟 batch size 成正比，$\eta \propto B$，这就是经典的 linear scaling rule。

---

## 这篇 paper 的意义

1. **对 community 的警醒**：你发 paper 说你的方法比 LoRA 强，先确保你把 LoRA 的 learning rate tune 了再说。不然你的 "improvement" 可能就是个 bug。

2. **理论解释**：不同方法需要不同 learning rate 这件事，不是 empirical observation 就完了，背后有 Hessian geometry 的支撑。这给了你一个 principled 的方式来猜 learning rate range。

3. **Practical guidance**：如果你要在生产环境用 LoRA，vanilla LoRA + 仔细 tune learning rate 就够了。不用追那些 fancy variants。省时间省算力。

4. **对 LoRA research 方向的暗示**：在 weight space 做 low-rank adaptation 这条路可能快到头了。要突破可能得换赛道，比如 representation fine-tuning (ReFT) 或者 nonlinear adaptation。

---

## 一句话总结

**Learning rate 是个 magic number，调对了啥方法都差不多。调不对，你的 "innovation" 可能只是你碰巧选了个对你有利的 learning rate。**

---

# Learning Rate Matters: 深入解析

Andrej, 这篇 paper 触及了 deep learning 研究中一个你长期关注的核心问题 — empirical rigor 与 methodological claims 之间的 gap. 我会尽可能详细地拆解, build intuition, 并把相关 learning theory 联系起来.

---

## 1. 核心论点与动机

这篇 paper 的论点很直接: LoRA 的诸多 variants (PiSSA, MiLoRA, DoRA, Init[AB]) 在 prior work 中报告的 performance gains, 很大程度上是 suboptimal hyperparameter tuning 造成的 illusion. 一旦对 learning rate 进行系统 search, vanilla LoRA 与这些 sophisticated variants 之间的 gap 收敛到 1-2% 以内.

作者做了一个非常 striking 的统计 (Figure 2): 在 52 篇 LoRA PEFT 论文中, 只有 1 篇同时 tune 了 learning rate, batch size, 和 rank 三个维度. 少于 30% tune 了 learning rate. 这与 Melis et al. (2017) 揭示 LSTM 改进实为 hyperparameter tuning 的故事高度相似 — https://arxiv.org/abs/1707.05589

这个现象在 deep learning 历史上反复出现: Chatfield et al. (2014) 对 image classification, Shchur et al. (2018) 对 GNN, Lucic et al. (2018) 对 GAN, Ferrari Dacrema et al. (2019) 对 recommender systems, Schmidt et al. (2021) 对 optimizers. 每次 community 意识到 baselines 被不公平对待时, 都会发现 "improvements" 大幅缩水.

---

## 2. LoRA PEFT Methods 技术拆解

### 2.1 Vanilla LoRA

给定 pretrained layer 的 weight $W_{\text{pre}} \in \mathbb{R}^{m \times n}$, LoRA 注入两个 trainable matrices:

$$h = W_{\text{pre}} x + \gamma_r B A x$$

变量含义:
- $W_{\text{pre}}$: pretrained weight matrix, shape $m \times n$
- $A \in \mathbb{R}^{r \times n}$: down-projecting matrix, 把 $n$ 维 input 压缩到 $r$ 维
- $B \in \mathbb{R}^{m \times r}$: up-projecting matrix, 把 $r$ 维扩展回 $m$ 维
- $r$: rank, $r \ll \min(m, n)$, 这是 "low-rank" 的来源
- $\gamma_r = \alpha / r$: scaling factor, $\alpha$ 是 hyperparameter
- $x \in \mathbb{R}^n$: layer input
- $h \in \mathbb{R}^m$: layer output

Initialization: $B_0 = 0$, $A_0 \sim \mathcal{N}(0, \sigma^2)$ (Kaiming). 这个 asymmetry 的设计意图是让 $B_0 A_0 = 0$, 从而训练 start exactly from pretrained checkpoint.

Intuition: LoRA 假设 task-specific weight update $\Delta W$ 具有 low intrinsic dimensionality (Li et al. 2018, https://arxiv.org/abs/1804.08838). 所以用 $BA$ 来 approximate $\Delta W$, 其中 $BA$ 的 rank 至多为 $r$.

### 2.2 PiSSA (Principal Singular Values and Singular Vectors Adaptation)

PiSSA 的核心 idea: 与其让 $BA$ 从 0 开始慢慢学, 不如直接用 $W_{\text{pre}}$ 的 top-$r$ principal components 来 initialize $BA$.

对 $W_{\text{pre}}$ 做 SVD: $W_{\text{pre}} = \sum_i \sigma_i u_i v_i^T$, 其中 $\sigma_i$ 按 descending order 排列.

$$B_0 A_0 = \sum_{i=1}^{r} \sigma_i u_i v_i^T$$

$$B_0 = \sum_{i=1}^{r} \sqrt{\sigma_i} u_i e_i^T, \quad A_0 = \sum_{i=1}^{r} \sqrt{\sigma_i} e_i v_i^T$$

变量含义:
- $\sigma_i$: 第 $i$ 大的 singular value
- $u_i \in \mathbb{R}^m$: left singular vector
- $v_i \in \mathbb{R}^n$: right singular vector
- $e_i \in \mathbb{R}^r$: 第 $i$ 个 standard basis vector

Residual matrix:
$$W_{\text{res}} = W_{\text{pre}} - B_0 A_0 = \sum_{i=r+1}^{\min(m,n)} \sigma_i u_i v_i^T$$

Forward pass 变成: $h = W_{\text{res}} x + \gamma_r B A x$

Intuition: PiSSA 把 $W_{\text{pre}}$ 的 "最重要部分" (top-$r$ singular components, 对应最大 $\sigma_i$) 放到 trainable $BA$ 里, 把 "次要部分" freeze 在 $W_{\text{res}}$ 里. 作者声称这样 convergence 更快, loss/gradient norm 曲线接近 full fine-tuning.

关键 insight: 因为 top singular values $\sigma_1, \ldots, \sigma_r$ 通常很大, $B_0$ 和 $A_0$ 的 entries 也相应大, 这会导致 loss landscape 在 initialization 处的 curvature 很大 — Hessian 的 $\lambda_{\text{max}}$ 很大. 这正是 PiSSA 需要更低 learning rate 的根本原因.

### 2.3 MiLoRA (Minor singular components)

MiLoRA 与 PiSSA 对称: 用 bottom-$r$ minor components 来 initialize $BA$.

$$B_0 A_0 = \sum_{i=\min(m,n)-r+1}^{\min(m,n)} \sigma_i u_i v_i^T$$

$$W_{\text{res}} = \sum_{i=1}^{\min(m,n)-r} \sigma_i u_i v_i^T$$

Intuition: MiLoRA 的 philosophy 完全不同. 它认为 principal components 编码的是 pretrained model 的 general knowledge, 不应该被 fine-tuning 破坏. Minor components 对应 "noise" 或 task-irrelevant directions, 修改它们既能 adapt to new task, 又能 maximally retain pretrained knowledge (减少 catastrophic forgetting).

因为 minor singular values 很小, $B_0, A_0$ 的 entries 也小, Hessian 的 $\lambda_{\text{max}}$ 应该比 PiSSA 小, 所以 MiLoRA 能用更大的 learning rate.

### 2.4 Init[AB]

Hayou et al. (2024a, https://arxiv.org/abs/2406.08447) 理论分析了 LoRA initialization, 发现 Init[A] (default, 只随机化 $A$) 比 Init[B] (只随机化 $B$) 好. Li et al. (2025, https://arxiv.org/abs/2505.23194) 进一步提出 Init[AB]: 同时随机化 $B$ 和 $A$:

$$B_0 \sim \mathcal{N}(0, \sigma^2), \quad A_0 \sim \mathcal{N}(0, \sigma^2)$$

因为 $B_0 A_0 \neq 0$, 需要引入 $W_{\text{res}} = W_{\text{pre}} - B_0 A_0$.

Intuition: 双侧随机化在 stability, training efficiency, hyperparameter robustness 之间取得平衡. 理论上, 这避免了 single-side initialization 导致的 gradient magnitude imbalance.

### 2.5 DoRA (Weight-Decomposed Low-Rank Adaptation)

DoRA 把 weight update 拆成 magnitude 和 direction 两个独立部分:

$$h = \gamma_r \left( \frac{m}{\|W_{\text{pre}} + BA\|_c} \odot (W_{\text{pre}} + BA) \right) x$$

变量含义:
- $m \in \mathbb{R}^{1 \times n}$: trainable magnitude vector, initialized $m_0 = \|W_{\text{pre}}\|_c$
- $\|\cdot\|_c$: columnwise norm (对每列取 norm)
- $\odot$: element-wise multiplication with broadcasting across columns
- $B, A$: 与 vanilla LoRA 相同的 initialization

Intuition: DoRA 的 motivation 来自观察 — vanilla LoRA 只能 uniformly scale magnitude 和 direction, 而 full fine-tuning 可以 independently 改变两者. DoRA 通过额外引入 $m$ vector (参数量增加很少) 来解耦这两个维度.

DoRA 声称在 low-rank regime (小 $r$) 下显著优于 LoRA. 但这篇 paper 显示, 经过 learning rate tuning 后, DoRA 的 advantage 大幅缩水.

---

## 3. 为什么 Learning Rate Matters: 理论基础

### 3.1 SGD Update Rule 与 Local Geometry

SGD update:
$$\theta_{t+1} = \theta_t - \eta g(\theta_t)$$

- $\theta_t$: 参数 at step $t$
- $\eta$: learning rate
- $g(\theta_t) = \nabla \mathcal{L}(\theta_t)$: gradient
- $\mathcal{L}$: loss function

Local geometry 由 gradient $g(\theta_t)$ 和 Hessian $H(\theta_t) = \nabla^2 \mathcal{L}(\theta_t)$ 刻画.

### 3.2 Optimal Learning Rate 与 Hessian Eigenvalue

LeCun et al. (1992) 的经典结果 (https://papers.nips.cc/paper/1992/hash/)

$$\eta^* \propto \frac{1}{\lambda_{\text{max}}(H(\theta))}$$

- $\eta^*$: optimal learning rate
- $\lambda_{\text{max}}$: Hessian 的最大 eigenvalue, 即 loss landscape 在该点的 maximal curvature (sharpness)

更精确地, 在 quadratic approximation 下:
- $1/\lambda_{\text{max}} \leq \eta^* < 2/\lambda_{\text{max}}$: efficient learning
- $\eta > 2/\lambda_{\text{max}}$: divergence

Lewkowycz et al. (2020) 发现 "catapult" regime (https://arxiv.org/abs/2003.02218):
$$2/\lambda_{\text{max}} \leq \eta^* \leq 12/\lambda_{\text{max}}$$

在这个 regime, 现代 architectures 反而达到 optimal performance. 这与 Cohen et al. (2021) 的 "edge of stability" 现象 (https://arxiv.org/abs/2103.00065) 一致 — training 过程中 $\lambda_{\text{max}}$ 会自适应调整, 使 $\eta$ 恰好处于 stability edge.

Intuition building: 
- $\lambda_{\text{max}}$ 大 → landscape 很 "sharp" (narrow valley) → 大 learning rate 会 overshoot → 需要小 $\eta$
- $\lambda_{\text{max}}$ 小 → landscape 很 "flat" (wide basin) → 小 learning rate 收敛太慢 → 可以用大 $\eta$

### 3.3 为什么不同 LoRA Variants 有不同 $\lambda_{\text{max}}$

这是这篇 paper 最深刻的 insight. 不同 initialization 策略确立了不同的 starting point $\theta_0$, 导致:
- 不同的 $g(\theta_0)$
- 不同的 $H(\theta_0)$
- 不同的 training trajectory

对于 PiSSA: $B_0, A_0$ 包含 top-$r$ singular values $\sigma_1, \ldots, \sigma_r$ (很大). 当这些参数 perturb 时, loss 变化剧烈, 所以 Hessian 的 entries 大, $\lambda_{\text{max}}$ 大.

对于 MiLoRA: $B_0, A_0$ 包含 bottom-$r$ singular values (很小). Perturbing 这些参数对 loss 影响小, $\lambda_{\text{max}}$ 小.

Figure 6 的实验验证了这一点: PiSSA 的 $\lambda_{\text{max}}$ 比 LoRA 大约一个数量级, 而 MiLoRA 和 Init[AB] 的 $\lambda_{\text{max}}$ 与 LoRA 接近 (在 Qwen 上约 2×).

---

## 4. 实验设计与关键结果

### 4.1 Setup

Models: Qwen3-0.6B, Gemma-3-1B, Llama-2-7B
Tasks: mathematical reasoning (MetaMathQA → GSM8K, MATH), code generation (CodeFeedback → HumanEval, MBPP)

Hyperparameter search:
- Learning rate: $10^{-6}$ 到 $10^{-3}$, log scale, 每个数量级 4 个点 ({1.1247, 2.0000, 3.5566, 6.3246} × 10^*), 最多 16 个 grid points
- Batch size: {16, 64, 128}
- Rank: {4, 8, 16, 32, 64, 128, 256}

Scaling factor $\gamma_r = \alpha / r = 1$ (设 $\alpha = r$), 这 factored out scaling factor 的 tuning 需求.

### 4.2 核心结果 1: Performance Parity

Figure 1 (Qwen3-0.6B, math, $r=128$): 所有方法的 peak accuracy 在 0.43% 范围内. LoRA (最优) 49.60%, DoRA 49.45%, Init[AB] 49.29%, MiLoRA 49.40%, PiSSA 49.51%.

但 crucially, optimal learning rates 不同:
- LoRA: ~$2 \times 10^{-4}$
- PiSSA: ~$6.32 \times 10^{-5}$ (约 3× 更小)
- MiLoRA: ~$1.12 \times 10^{-4}$

Table 1 (Gemma-3-1B, math, $r=128$): 所有方法 peak 在 20.0–21.0% 范围, 差距 0.52%.

Figure 4 (Llama-2-7B): math 差距 0.43%, code 差距 1.75%.

### 4.3 核心结果 2: Rank-Dependent Behaviors

Figure 5 (Gemma, math):
- PiSSA: low rank ($r \leq 32$) 时 underperforms LoRA by up to 1.67%, high rank ($r \geq 128$) 时 slightly outperforms (0.22–0.33%)
- MiLoRA: 相反 trend, low rank ($r=8$) outperforms LoRA by 0.8%, high rank ($r=256$) underperforms by 0.63%
- Init[AB]: medium rank ($r=128$) outperforms, 其他 rank 与 LoRA 相当
- DoRA: low rank ($r=8$) outperforms by 1.1%, 但 gain 比 original paper 报告的小

Intuition: 这些 rank-dependent behaviors 暗示, 不同方法在不同 rank regime 下有不同的 inductive bias. PiSSA 用 top-$r$ components, 当 $r$ 小时, 这些 components 可能 too "rigid" (主要编码 general knowledge); 当 $r$ 大时, 有足够 capacity 来 adapt. MiLoRA 相反.

### 4.4 核心结果 3: Batch Size Effects

Table 1 的 joint optimization 显示:
- Learning rate tuning 比 batch size tuning 重要得多
- DoRA 固定 LR=$2 \times 10^{-5}$ 只 tune batch size: max accuracy 11.16%
- DoRA 固定 batch size 任意值, tune LR: accuracy 20.5–21.0%
- Optimal LR 与 batch size 成正比, 符合 Goyal et al. (2017) 的 scaling rule (https://arxiv.org/abs/1706.02677): $\eta \propto B$

这解释了 Schulman & Lab (2025) 报告的 LoRA 在 $B=128$ 时 degradation — 很可能是 suboptimal LR.

### 4.5 PiSSA 的 "Robustness at Large LR"

一个 intriguing observation (Figure 4): 在 $\eta = 1.1 \times 10^{-3}$, PiSSA 仍保持 27.83% (math) 和 26.90% (code), 而其他方法 collapse 到 near-zero.

Intuition: PiSSA 的 Hessian $\lambda_{\text{max}}$ 大, 所以即使在 large LR 下, 它仍处于 catapult regime ($2/\lambda_{\text{max}} \leq \eta \leq 12/\lambda_{\text{max}}$), 而其他方法已经超过 $12/\lambda_{\text{max}}$ 进入 divergence.

---

## 5. Hessian 分析细节

### 5.1 Block-wise Hessian Estimation

作者没有把所有层的 LoRA 参数 concatenate 成一个巨大 Hessian, 而是 layer-wise, matrix-type-wise 估计:

$$\lambda_{\text{max}}^l = \lambda_{\text{max}}(H^l)$$

- $H^l$: 第 $l$ 个 matrix type 的 Hessian, 对应参数 $\theta^l = \{B_0^l, A_0^l\}$
- $l$ indexes through matrix types (Q, K, V, O, Gate, Up, Down) 和 Transformer layers

### 5.2 Lanczos Algorithm

用 Lanczos iteration (而不是 Power Iteration, 因为后者 converge 到最大 magnitude eigenvalue, 而作者要 algebraically largest). Algorithm 1:

- Iterations $m = 100$
- Tolerance $\epsilon = 5 \times 10^{-3}$
- Hessian-Vector Product (HVP) via double backward (Algorithm 2)
- Re-orthogonalization (Paige 1970) 避免 numerical instability
- Float32 precision

HVP 的关键: 
$$Hv = \nabla_\theta (g^T v) = \nabla_\theta (\nabla_\theta \mathcal{L}^T v)$$

通过 `torch.autograd.functional.vhp` 实现, 不需要 explicitly form $H$.

Sample size: $N = 500$ training samples, batch size $B = 5$. Figure 10 验证 500 samples 足够 stable.

### 5.3 Figure 6 的 Interpretation

Ratio $\lambda_{\text{max}, t}^{Q, i} / \lambda_{\text{max}, \text{LoRA}}^{Q, i}$:
- $t \in \{\text{PiSSA, MiLoRA, Init[AB]}\}$
- $i = 1, \ldots, L$: layer index

结果:
- PiSSA: ratio 中位数约 10× (consistently across layers)
- MiLoRA, Init[AB]: ratio 接近 1, 在 Qwen 上约 2×

这与 observed optimal LR ratios 吻合:
- PiSSA optimal LR / LoRA optimal LR ≈ 1/3 (Figure 1)
- MiLoRA / LoRA ≈ 1/3.2 (Qwen)
- Init[AB] / LoRA ≈ 1/1.8 (Qwen)

虽然不是完美 1:1 对应 (因为 Hessian 只在 initialization 点估计, training 过程中会变化), 但 trend 完全一致.

---

## 6. 对 PEFT Research 的启示

### 6.1 Methodological Concerns

Figure 2 的统计令人震惊: 52 篇论文中, 只有 1 篇同时 tune LR, BS, rank. 这意味着大多数 reported gains 可能是 artifact of unfair comparison.

这与 Lipton & Steinhardt (2019) 批评的 "troubling trends in ML scholarship" (https://queue.acm.org/detail.cfm?id=3358457) 完全吻合.

### 6.2 Scaling Factor 的 Redundancy

Appendix C 讨论: Zhang et al. (2025h) 证明 tuning LR 理论上 equivalent to tuning scaling factor $\gamma_r$. 这 validates 作者固定 $\gamma_r = 1$ 的决定.

这是 Karpathy 你会欣赏的 insight — 很多看似独立的 hyperparameters 实际上是 coupled 的, 调一个等价于调另一个. 理解这种 equivalence 比 brute-force tuning 更重要.

### 6.3 LoRA 的 Saturation Hypothesis

作者在 Section H 暗示: weight-based low-rank adaptation 策略可能 approaching saturation. 这与 Biderman et al. (2024) "LoRA learns less and forgets less" (https://arxiv.org/abs/2405.09673) 的发现一致 — LoRA 与 full FT 之间仍有 gap, 但 closing this gap 可能需要 orthogonal paradigms:

- Hidden representation fine-tuning (ReFT, Wu et al. 2024, https://arxiv.org/abs/2404.03592)
- Non-linear function adaptation (Yin et al. 2025, https://arxiv.org/abs/2509.13240)

---

## 7. Karpathy-Style 的 Intuition Building

### 7.1 Loss Landscape 的 Mental Model

想象 loss landscape 是一个高维 surface. Pretrained model 位于一个 wide basin (pretraining 找到的 flat minimum). Fine-tuning 要在这个 basin 附近找到 task-specific 的 better point.

不同 LoRA variants 的 initialization 相当于选择不同的 "entry points" 到这个 landscape:
- Vanilla LoRA: 从 pretrained point 出发, $BA=0$, landscape flat (small $\lambda_{\text{max}}$)
- PiSSA: 跳到 principal subspace, landscape sharp (large $\lambda_{\text{max}}$), 但离 optimal solution 可能更近
- MiLoRA: 跳到 minor subspace, landscape flat, 但可能需要走更远

### 7.2 为什么 Sharpness 与 Generalization 相关

Dinh et al. (2017, https://arxiv.org/abs/1703.04933) 指出 sharp minima 可以 generalize, 挑战了 "flat minima generalize better" 的传统 wisdom. 但 Lyu et al. (2022, https://arxiv.org/abs/2106.05759) 证明 normalization layers 通过 sharpness reduction 改善 generalization.

对于 LoRA, sharpness (via $\lambda_{\text{max}}$) 主要影响 optimization dynamics, 不一定直接决定 generalization. 但 sharpness 决定了 optimal LR, 而 LR 影响 final solution 的 generalization.

### 7.3 Edge of Stability 与 Adaptive Sharpness

Cohen et al. (2021) 发现 training 过程中 $\lambda_{\text{max}}$ 会 self-adjust: 当 $\eta \cdot \lambda_{\text{max}} > 2$ 时, training 不 diverge, 而是 $\lambda_{\text{max}}$ 减小, 直到 $\eta \cdot \lambda_{\text{max}} \approx 2$. 这解释了为什么 large LR training 仍能 work — landscape 会 "flatten" to accommodate.

对于 LoRA variants, 这意味着即使 initialization 处 $\lambda_{\text{max}}$ 不同, training 过程中会 converge 到类似的 stability edge. 这可能是为什么 final performance 相近的深层原因.

### 7.4 与 Neural Tangent Kernel (NTK) 的联系

LoRA 的 low-rank 结构改变了 NTK 的 spectrum. 对于 PiSSA, 因为初始化在 principal directions, NTK 的 top eigenvalues 对应这些 directions, 导致 sharp gradient dynamics. 对于 MiLoRA, NTK spectrum 更 flat.

这与 Yang et al. (2023, https://arxiv.org/abs/2308.13111) 的 Bayesian LoRA 分析, 以及 Xu et al. (2025, https://arxiv.org/abs/2503.06982) 的 gradient flow perspective 相关.

---

## 8. 局限性与 Future Directions

### 8.1 实验 Scope

只测了 decoder-only LLMs up to 7B, 只测了 math 和 code. 是否 extend 到 70B+ 或 instruction following, dialogue, reasoning tasks 仍需验证.

### 8.2 Fixed Secondary Hyperparameters

LR scheduler, warmup steps, adapter placement 都 fixed. 这些可能与方法有 interaction effects.

### 8.3 Hessian 只在 Initialization 估计

更完整的分析应该 track $\lambda_{\text{max}}$ throughout training. 这与 edge of stability 理论直接相关.

### 8.4 DoRA 的 Hessian 没分析

DoRA 与 LoRA initialization 相同, 所以初始 Hessian 相同. 但 DoRA 的 magnitude vector $m$ 会导致 training 过程中 Hessian evolution 不同. 作者 defer 到 future work.

### 8.5 与 LoRA-GA, LoRA-One 等的关系

Section H 提到 LoRA-One (Zhang et al. 2025g, https://arxiv.org/abs/2502.01235) 报告了 ~2% improvement on Llama, 即使在 comprehensive LR sweep 下. 这 suggests 某些 variants 可能确实有 genuine methodological advantage, 但需要单独验证.

---

## 9. 相关参考链接

核心论文:
- LoRA: https://arxiv.org/abs/2106.09685
- PiSSA: https://arxiv.org/abs/2404.02948
- DoRA: https://arxiv.org/abs/2402.09353
- MiLoRA: https://arxiv.org/abs/2406.09044
- Init[AB]: https://arxiv.org/abs/2505.23194

Learning theory:
- LeCun et al. 1992 (optimal LR): https://papers.nips.cc/paper/1992
- Lewkowycz et al. 2020 (catapult): https://arxiv.org/abs/2003.02218
- Cohen et al. 2021 (edge of stability): https://arxiv.org/abs/2103.00065
- Dinh et al. 2017 (sharp minima): https://arxiv.org/abs/1703.04933

Empirical re-evaluation precedents:
- Melis et al. 2017 (LSTM): https://arxiv.org/abs/1707.05589
- Schmidt et al. 2021 (optimizers): https://arxiv.org/abs/2007.01547
- Biderman et al. 2024 (LoRA learns less): https://arxiv.org/abs/2405.09673

LoRA hyperparameter theory:
- Hayou et al. 2024a (initialization): https://arxiv.org/abs/2406.08447
- Hayou et al. 2024b (LoRA+): https://arxiv.org/abs/2402.12354
- Kalajdzievski 2023 (scaling factor): https://arxiv.org/abs/2312.03732
- Liu et al. 2025 (loss landscape): https://arxiv.org/abs/2410.11225
- Zhang et al. 2025h (unified framework): https://arxiv.org/abs/2507.06558

Alternative PEFT paradigms:
- ReFT: https://arxiv.org/abs/2404.03592
- GaLore: https://arxiv.org/abs/2403.03507
- VeRA: https://arxiv.org/abs/2310.11454

---

## 10. 总结

这篇 paper 的 contribution 有三层:

1. **Empirical**: 系统证明 vanilla LoRA + proper LR tuning ≈ sophisticated variants, 打破了许多 reported gains 的 illusion.

2. **Theoretical**: 通过 Hessian $\lambda_{\text{max}}$ 分析, 解释了为什么不同 variants 需要不同 LR — initialization 策略决定了 starting point 的 sharpness.

3. **Methodological**: 呼吁 PEFT community 采用更 rigorous 的 hyperparameter search protocol, 特别是 across LR, BS, rank 三个维度.

对于 build intuition, 最关键的 takeaway 是: **不要把 methodological sophistication 与 empirical improvement混淆. 很多时候, 一个 well-tuned simple baseline 能匹敌 sophisticated method. 理解 loss landscape geometry (via Hessian) 比 invent 新的 initialization 策略更能揭示 training dynamics 的本质.**

这与 Karpathy 你一直倡导的 "understand fundamentals, build intuition from first principles" 的理念完全一致. LoRA variants 的 proliferation 反映了 community 倾向于 incremental architectural tweaks, 而忽视了对 optimization dynamics 的深入理解. 这篇 paper 是一个重要的 corrective.
