---
source_pdf: Neural Thickets.pdf
paper_sha256: 7a7afe84b012a7b82d5fbf48bcf337d697a5634bf44ff55b40c0b120c09c6a50
processed_at: '2026-08-05T22:24:02-07:00'
target_folder: LLM-Training/Optimizer
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej，要 "用人话" 把这篇 paper 的 intuition 讲透，我们完全可以抛开那些复杂的数学符号，直接在脑海里构建一个几何画面。你非常熟悉 loss landscape 的可视化，我们就用这个来讲。

### 1. 直觉构建：从 "大海捞针" 到 "灌木丛"

想象你在训练一个 neural network。Parameter space 是一个几十亿维的极其巨大的空间。

**没 Pretraining 的世界 (Needle in a Haystack)**
如果你用 Xavier initialization 从 scratch 训练一个 billion-parameter 的 model，你的起点在这个巨大空间里是一个随机点。好的 solutions (能让 model 正确输出的 weights) 在哪里？它们就像是散落在无穷大沙漠里的几根针。Random guessing 几乎不可能命中。你必须用 gradient descent，因为 gradient 就像一个非常聪明的本地向导，能一步步带你走出沙漠找到那根针。

**Pretraining 之后的 Thicket Regime**
现在，你拿了一个经过海量数据 pretrain 过的 32B LLM。Pretraining 做了一件神奇的事：它把整个 loss landscape 的几何形状彻底重塑了。Pretraining 把 weights 拉到了一个极其特殊的区域。在这个区域里，好的 solutions 不再是稀缺的针，而是像热带雨林里的灌木丛一样密集。

如果你在这个 pretrained weights $\theta$ 周围加一点微小的随机高斯噪声 $\epsilon$（相当于把 weights 随机踹一脚），你会发现：**有很大概率，这个被踹了一脚的 model，在某个特定 task（比如数学题）上的表现反而变好了！**

为什么？因为在高维空间里，随机向量几乎是正交的。你随机踹一脚，实际上是在高维空间里探索了一个与当前 weights 完全不同的 functional subspace。由于 pretraining 赋予了 model 极强的 representation 能力，这些随机的 subspace 里，有相当大一部分恰好编码了某种 "specialist" 能力。这就像灌木丛里每一根树枝都代表一种特定的解题套路。

### 2. 机制拆解：为什么 Random Guessing 会管用？

Paper 里定义了一个核心概念叫 **Solution Density $\delta(m)$**。用人话说，就是 "踹一脚模型，性能提升 $m$ 的概率有多大"。

公式长这样：
$$ \delta(m) = \mathbb{P}_{\epsilon \sim \mathcal{N}(0, \sigma^2 \mathbf{I})} \left[ s(\pmb{\theta} + \epsilon) \geq s(\pmb{\theta}) + m \right] $$
*   $\theta$: Pretrained weights。
*   $\epsilon$: 随机高斯噪声。
*   $\sigma$: 噪声的幅度（paper 里通常很小，比如 0.005）。
*   $s$: Accuracy 指标。

Paper 实验发现，$\delta(m)$ 随着 model size 单调递增！在 0.5B 模型上，踹一脚大概率变差。但在 32B 模型上，踹一脚命中 +5% accuracy 的概率高得惊人。Scaling 把沙漠变成了灌木丛。

### 3. Specialists vs Generalists (专才与通才)

你可能会问：既然踹一脚能变好，那我多踹几脚，找一个在所有 task（数学、代码、化学）上都变好的 direction 不就行了？

Paper 给出了极其反直觉的结论：**不存在这种通才 direction**。所有的随机 perturbation 都是 "专才"。

为了证明这一点，作者定义了 **Spectral Discordance $\mathcal{D}$**。它计算的是不同 task 性能排名之间的相关性。
$$ \mathcal{D} = 1 - \frac{1}{M(M-1)} \sum_{j \neq k} \mathbf{C}_{jk} $$
*   $M$: 任务数量。
*   $\mathbf{C}_{jk}$: Task $j$ 和 Task $k$ 性能的 Pearson correlation。

如果 $\mathcal{D} \to 1$，说明任务间毫无关联，是纯专才。实验发现，随着 model scale 增大，$\mathcal{D}$ 越来越大。这意味着在 pretrained weights 周围，你随机踹出来的都是 "数学强但代码烂" 或者 "代码强但化学烂" 的 specialists。

### 4. RandOpt 算法：集思广益

既然周围全是各种各样的专才，我们怎么利用它们？作者提出了 **RandOpt**。逻辑极其简单：

**Training (O(1) Steps, 完全并行):**
1. 取 pretrained model $\theta$。
2. 随机生成 $N=5000$ 个带噪声的 weights：$\theta_i' = \theta + \sigma \cdot \epsilon_i$。
3. 在一个小 validation set 上测试这 5000 个 models。
4. 选出表现最好的 $K=50$ 个 models，记下它们的 seeds。

注意，这里没有任何 backpropagation，没有任何 sequential update。5000 个 models 可以在 5000 个 GPUs 上完全并行跑完。所以它的 wall-clock 时间是 $O(1)$！相比之下，PPO 和 GRPO 需要 $T$ 步 sequential 更新，wall-clock 时间是 $O(T)$。

**Inference (Majority Voting):**
遇到一道新题，让这选出的 50 个 specialist models 分别给出答案，然后做 majority vote（多数表决）。
$$ \hat{y} = \mathrm{mode} \left( \left\{ \underset{y}{\arg \max} f_{\theta_i}(y|x) \mathrm{~} | \mathrm{~} i \in \mathcal{I}_{\mathrm{top}} \right\} \right) $$
*   $\mathcal{I}_{\mathrm{top}}$: 选出的 50 个 models 的 index 集合。
*   $\mathrm{mode}$: 取众数。

你本质上是构建了一个 improvised ensemble。这 50 个 model 各有专长，集思广益的结果极其强大。

### 5. 实验数据对比 (Performance Table)

我们来看 Qwen2.5-3B-Instruct 在 GSM8K (数学题) 上的表现：

| Model | Method | GSM8K Accuracy | Training Wall-Clock Cost |
| :--- | :--- | :--- | :--- |
| Qwen2.5-3B | Base | 79.8% | 0 |
| Qwen2.5-3B | PPO | 83.1% | O(T) (需大量 sequential steps) |
| Qwen2.5-3B | GRPO | 83.2% | O(T) |
| Qwen2.5-3B | **RandOpt** | **87.1%** | **O(1)** (只需一次并行 forward) |

在相等的 training FLOPs 预算下，RandOpt 完全没有用 gradient descent，却打败了 PPO 和 GRPO。这是 Loss Landscape 几何特性改变带来的直接红利。

### 6. 蒸馏 解决 Inference Cost

RandOpt 唯一的痛点是 inference 时要跑 50 次 forward pass。作者用极简的 distillation 解决了这个问题。

用这 50 个 models 生成一堆带推理过程的答案，然后作为 label 去 SFT 原始的 base model：
$$ \mathcal{L}_{\mathrm{Distill}}(\theta) = - \sum_{t=T_x+1}^{T} \log p_{\theta}(s_t \mid x, s_{<t}) $$
*   $s$: 完整的 token 序列，包含 input $x$, reasoning $r$, answer $y$。
*   $T_x$: Input 的长度。
*   求和从 $T_x+1$ 开始，意味着只对 reasoning 和 answer 计算 loss，mask 掉 input prompt。

惊人的是，这个 distillation 只需要跑 10 个 SGD iteration，计算成本只占训练阶段的 2%。 distilled model 轻松达到了接近 50-model ensemble 的效果。

### 7. 更深层的联想 (Hallucinated Connections)

这篇 paper 的结论可以解释近期 LLM 社区里很多 "玄学" 现象：

*   **Spurious Rewards 为什么管用？** Shao et al. (2025) 发现用完全错误的 random reward 训 LLM 也能提升 reasoning。Neural Thickets 给出完美解释：因为大模型周围全是对的 direction，只要你的 gradient update 没有把 model 踹出这片 thicket，随便往哪个方向走，都有大概率撞上一个 specialist。[Spurious Rewards Paper](https://arxiv.org/abs/2506.10947)
*   **LoRA 与 Intrinsic Dimension:** Aghajanyan et al. (2020) 证明 fine-tuning 发生在极低维的 subspace 中。这意味着 task-relevant directions 在高维空间中非常 "粗"。高斯采样的随机向量有极高概率与这些粗方向重合，这就是为什么 random guessing 命中率极高的几何原因。[Intrinsic Dimension Paper](https://arxiv.org/abs/2012.13255)
*   **Model Editing & Weight Space:** Dravid et al. (2024) 发现 diffusion models 里简单的 weight manipulation 就能编辑生成结果。这说明 pretraining 在 weight space 里隐式地构建了大量的 semantic directions，你随便碰一下就可能触发某个 semantic feature。[Weight Space Editing Paper](https://arxiv.org/abs/2407.06406)
*   **Test-Time Compute:** RandOpt 的 training 阶段本质上是在 weight space 做了 Best-of-N。这和 test-time 在 output space 做 Best-of-N (majority vote) 有异曲同工之妙，只是搜索空间变了。[Test-Time Scaling Laws](https://arxiv.org/abs/2408.03314)

### 8. Format Thicket vs Reasoning Thicket

Paper 里还有个非常 sharp 的观察。RandOpt 在 GSM8K 上的提升，有多少是真的学会了推理，有多少只是碰巧把答案格式调对了（比如刚好在答案前加了个 `####`）？

作者分解发现，有 19% 的提升来自 format fix，12.3% 来自真正的 reasoning fix。这说明 thicket 是分类型的。有的灌木丛是 "格式专才"，有的是 "推理专才"。这进一步证实了 pretrained model 是一个极其复杂的 distribution，表面的 accuracy 只能反映它的冰山一角。

总结一句话：**Pretraining 把你带到了一个充满各路高手的广场上，你完全不需要辛苦地去培养一个全才，你只需要随便拉住 50 个路人，对他们的问题进行举手表决，就能得到比专业训练更好的答案。** 这彻底改变了我们对 post-training 的直觉。

---

Andrej, 这篇 paper 《Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights》 的 核心思想 极其 fascinating。作者 Yulu Gan 和 Phillip Isola 发现，当 model scale 足够大 且 经过 充分 pretraining 后，weight space 的 local geometry 会发生 根本性 改变。传统观点认为 random guessing 在高维空间中是 hopeless 的，但是 这篇 paper 证明了 在 large pre-trained models 周围，存在着 密集的、diverse 的 task-specific experts。因此，我们 甚至 不需要 gradient descent，仅仅 依靠 random sampling 和 ensembling，就 能 达到 甚至 超越 GRPO 和 PPO 的 performance。

这里 是 对这篇 paper 的 极致 详细 的 技术讲解，旨在 为你 构建 深刻 的 intuition。

### 1. Intuition Building: The Geometry of Loss Landscape (直觉构建：Loss Landscape 的几何结构)

在 传统 deep learning 中，从 scratch 训练 一个 billion-parameter 的 model 属于 "needle in a haystack" regime。因为 高维空间 中，random initialization 落在 有效 subspace 的 概率 几乎 为 零，所以 我们 必须依赖 gradient descent 这种 structured search algorithm 来 寻找 minima。

但是 这篇 paper 指出，pretraining 彻底 改变了 loss landscape 的 topology。由于 pretraining objective 聚合了 无数 个 downstream tasks，pretrained weights $\theta$ 实际上 位于 一个 极其 平坦 的 basin 之中。在这个 basin 内部，如果 我们 沿着 单一 task 的 loss 方向 看，$\theta$ 甚至 可能 不是 最低点，而是 位于 一个 "accuracy valley" 之中。周围 遍布着 能够 提升 特定 task 性能 的 "hills" (即 task-specific experts)。

为什么 random sampling 能够 找到 这些 experts？这 归功于 高维空间 的 几何 特性。在 极高维 空间 中，从 $\mathcal{N}(0, \sigma^2 \mathbf{I})$ 采样 的 random vectors $\epsilon$ 彼此之间 几乎 是 orthogonal 的。如果 我们 在 $\theta$ 附近 添加 这些 orthogonal 的 perturbations，实际上 是 在 探索 截然不同 的 functional subspaces。因为 pretraining 赋予了 model 极强 的 泛化 能力，这些 subspace 中 有 相当大 比例 仅仅 破坏了 模型 对 其他 tasks 的 能力，却 增强了对 某一特定 task 的 能力。这就 构成了 "Neural Thicket" (神经灌木丛)。

### 2. Mathematical Formulations & Variable Explanations (数学公式与变量解析)

#### 2.1 Solution Density (解的密度)
为了 量化 "thicket" 的 密集 程度，作者 定义了 Solution Density $\delta(m)$。

$$ \delta(m) = \mathbb{P}_{\epsilon \sim \mathcal{N}(0, \sigma^2 \mathbf{I})} \left[ s(\pmb{\theta} + \epsilon) \geq s(\pmb{\theta}) + m \right] $$

*   **$\delta(m)$**: Solution Density。表示 在 随机 扰动 下，模型 性能 提升 至少 $m$ 的 概率。
*   **$m$**: Performance improvement margin (性能提升阈值)。
*   **$\mathbb{P}$**: Probability (概率)。
*   **$\epsilon$**: Random Gaussian noise vector (高斯随机噪声向量)。
*   **$\mathcal{N}(0, \sigma^2 \mathbf{I})$**: Multivariate Gaussian distribution with mean 0 and covariance $\sigma^2 \mathbf{I}$ (多元高斯分布)。$\sigma$ 控制 扰动 的 局部 范围 (paper 中 取 0.005)，$\mathbf{I}$ 是 identity matrix。
*   **$s(\pmb{\theta})$**: Performance metric (如 accuracy) of the model with parameters $\pmb{\theta}$。
*   **$\pmb{\theta} \in \mathbb{R}^d$**: Pretrained model parameters (预训练模型参数)。

这个公式 揭示了 一个 scaling law：随着 model size 增加，$\delta(m)$ 单调 递增。在 32B 模型 上，随机 猜测 命中 +5% accuracy improvement 的 概率 远高于 0.5B 模型。

#### 2.2 Spectral Discordance (谱分歧度)
为了 验证 采样 出的 solutions 是 specialists (专才) 还是 generalists (通才)，作者 定义了 Spectral Discordance $\mathcal{D}$。

$$ \mathcal{D} = 1 - \frac{1}{M(M-1)} \sum_{j \neq k} \mathbf{C}_{jk} $$

*   **$\mathcal{D}$**: Spectral Discordance。衡量 任务 排名 之间的 不一致性。
*   **$M$**: Number of tasks (任务数量)。
*   **$\mathbf{C}_{jk}$**: Pearson correlation matrix $\mathbf{C}$ 中 的 元素，表示 task $j$ 和 task $k$ 性能 percentile rank 之间的 相关性。
*   **Theoretical Bound**: $\mathcal{D} \in [0, \frac{M}{M-1}]$。当 $\mathcal{D} \to 1$ 时，意味着 tasks 之间 呈现 orthogonal rankings (完全 专才化)；当 $\mathcal{D} \to 0$ 时，意味着 parallel rankings (完全 通才化)。

实验 发现 $\mathcal{D}$ 随着 model scale 单调 递增，证明 了 大模型 周围 的 perturbations 主要是 specialists。

#### 2.3 The RandOpt Algorithm (RandOpt 算法公式)

RandOpt 的 核心在于 完全 并行 的 random guessing 和 随后 的 ensembling。

**Training Phase (Random Guessing & Checking):**
$$ \pmb{\theta}' = \pmb{\theta} + \pmb{\sigma} \cdot \pmb{\epsilon}(s) $$
*   **$\pmb{\theta}'$**: Perturbed model parameters。
*   **$\pmb{\sigma}$**: Scaling factor sampled from a set $\Sigma$。
*   **$\pmb{\epsilon}(s)$**: Noise vector generated by random seed $s$。

选择 Top-K models:
$$ \mathcal{T}_{\mathrm{top}} = \mathop{\mathrm{arg}}_{i \in [N]} \mathrm{K} (v_i) $$
*   **$\mathcal{T}_{\mathrm{top}}$**: Indices of the top-K performing models。
*   **$N$**: Population size (总采样数)。
*   **$v_i$**: Score of model $i$ on validation set $\mathcal{D}_{\mathrm{train}}$。

**Inference Phase (Ensembling):**
$$ \hat{y} = \mathrm{mode} \left( \left\{ \underset{y}{\arg \max} f_{\theta_i}(y|x) \mathrm{~} | \mathrm{~} i \in \mathcal{I}_{\mathrm{top}} \right\} \right) $$
*   **$\hat{y}$**: Final predicted answer。
*   **$\mathrm{mode}$**: Majority voting function (众数函数)。
*   **$f_{\theta_i}(y|x)$**: Model $i$ predicting $y$ given input $x$。
*   **$\mathcal{I}_{\mathrm{top}}$**: The set of top-K model indices。

#### 2.4 Distillation Objective (蒸馏损失函数)
为了 降低 K 次前向传播 的 inference cost，作者 将 top-K ensemble 蒸馏 回 单个 model。

$$ \mathcal{L}_{\mathrm{Distill}}(\theta) = - \sum_{t=T_x+1}^{T} \log p_{\theta}(s_t \mid x, s_{<t}) $$
*   **$\mathcal{L}_{\mathrm{Distill}}(\theta)$**: Negative log-likelihood loss。
*   **$\theta$**: Model parameters being fine-tuned。
*   **$s = (s_1, s_2, \ldots, s_T)$**: Full token sequence, concatenation of input $x$, reasoning trace $r$, and final answer $y$ (即 $[x; r; y]$)。
*   **$T$**: Total length of the sequence $s$。
*   **$T_x$**: Length of the input question $x$。注意 求和 是 从 $T_x+1$ 开始，意味着 我们 只对 reasoning 和 answer 计算 loss，mask 掉了 input prompt。
*   **$s_t$**: Token at position $t$。
*   **$s_{<t}$**: All tokens preceding position $t$。
*   **$p_{\theta}$**: Predictive probability of the model。

### 3. Experimental Data Table Analysis (实验数据表解析)

下面 是 基于 paper 中 Table 4 提取 的 针对 Qwen2.5-3B-Inst 在 GSM8K 任务上 的 性能 对比：

| Model | Method | GSM8K Accuracy (%) | Wall-Clock Steps | Training FLOPs Normalized |
| :--- | :--- | :--- | :--- | :--- |
| Qwen2.5-3B-Inst | Base | 79.8 ± 0.4 | 0 | 0 |
| Qwen2.5-3B-Inst | TT-MV† (K=50) | 82.5 ± 0.2 | 0 | 0 (Test-time only) |
| Qwen2.5-3B-Inst | PPO | 83.1 ± 0.2 | O(T) | 1x |
| Qwen2.5-3B-Inst | GRPO | 83.2 ± 0.2 | O(T) | 1x |
| Qwen2.5-3B-Inst | ES | 85.8 ± 5.1 | O(T) | 1x |
| Qwen2.5-3B-Inst | RandOpt (N=3000, K=50) | **87.1 ± 0.8** | **O(1)** | 1x |
| Qwen2.5-3B-Inst | ES + TT-MV | **87.9 ± 0.9** | O(T) | 1x |

从 表格 可以 看出，RandOpt 在 完全 没有 sequential backpropagation (O(1) steps) 的 情况下，利用 相同 的 training FLOPs，实现 了 最高 的 accuracy。这 极具 说服力 地 证明了 在 thicket regime 下，structured search algorithm 并非 必需。

### 4. The 1D Signal Toy Model (1D 信号 Toy Model 解析)

为了 剖析 thicket 产生 的 原因，作者 设计了 一个 极简 的 1D autoregressive prediction 实验。

*   **Setup**: 训练 一个 small MLP $f_{\theta}$ 预测 1D signal 的 下一个 值。Pretraining data 包含 sinusoidal, linear, harmonic, sigmoidal 等 多种 信号。
*   **Regime 1: No Pretraining (Xavier Init)**。属于 "needle in haystack"。Random perturbations 几乎 无法 产生 任何 符合 函数 形状 的 预测，除非 把 $\sigma$ 放得 极大，但这 会 破坏 模型。
*   **Regime 2: Pretraining on Mixed Signals**。属于 "thicket regime"。Base model 给出 一个 average 预测。Random perturbations 会 使 模型 偏向 于 某一种 signal type (例如 更像 sine wave)。Top-K 采样 能够 找到 那些恰好 匹配 test signal 形状 的 perturbations。
*   **Regime 3: Pretraining on Single Signal Type**。属于 "plateau regime"。Base model 已经 是 这个 signal type 的 完美 预测器，random guessing 无法 提供 任何 提升。

这个 toy model 清晰 地 展示了 多样性 在 pretraining 中 的 作用。Pretraining on a distribution of many different tasks is critical to thickets forming.

### 5. Types of Thickets: Reasoning vs Format (Thicket 的类型：Reasoning vs Format)

Paper 中 最 有 洞察力 的 部分 之一 是 对 "Thicket Types" 的 解构。作者 将 GSM8K 上 的 提升 分解为：

1.  **Reasoning Thicket**: Base model 无法 解出 题目，perturbation 修复了 推理 逻辑，使其 能够 给出 正确 答案。
2.  **Format Thicket**: Base model 其实 算对 了，但是 没有 在 `####` 符号 后面 给出 答案，导致 strict checker 判错。Perturbation 恰好 修复了 输出 格式。

实验 数据 表明，RandOpt 带来 的 提升 中，有 19.0% 来自 format fix，12.3% 来自 真实 reasoning fix。这 说明 thicket 并非 单一 维度 的，它 可以 是 reasoning thicket，也可以 是 formatting thicket，甚至 可以 是 personality thicket 或 color thicket (见 Appendix J 的 diffusion model 例子)。这 进一步 强化 了 "pretrained model 是 一个 distribution" 的 观点：不同的 perturbations 激活了 不同 的 surface behaviors。

### 6. Broader Connections & Hallucinated Links (更广泛的联想与相关文献)

这篇 paper 的 结论 与 近期 多项 研究 产生 共鸣：

*   **Lottery Ticket Hypothesis**: Frankle & Carbin (2019) 提出 好 的 initialization 就像 中 彩票。Neural Thickets 则 指出，pretraining 之后，周围 全是 中奖彩票。你可以 参考 [Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635)。
*   **Intrinsic Dimensionality & LoRA**: Aghajanyan et al. (2020) 发现 fine-tuning 实际 发生在 极低维 的 subspace 中。这 解释了 为什么 random Gaussian perturbation (尽管 在 全维空间 中) 能够 高效 命中 reward-improving directions——因为 low-dimensional task-relevant directions 在高维 空间 中 占据了 极大 的 角度 展幅。你可以 参考 [Intrinsic Dimensionality](https://arxiv.org/abs/2012.13255) 和 [LoRA](https://arxiv.org/abs/2106.09685)。
*   **Dropout as Bayesian Approximation**: Gal & Ghahramani (2016) 证明 在 weights 上 加 Gaussian noise 等价于 variational inference。RandOpt 无意间 证明 了 pretrained weights 隐式 定义了 一个 极好 的 posterior distribution。参考 [Dropout as Bayesian](https://arxiv.org/abs/1506.02142)。
*   **Spurious Rewards**: Shao et al. (2025) 发现 即便 用 random/spurious rewards 训练 LLM 也能 提升 reasoning 性能。Neural Thickets 给出 了 完美 解释：因为 在 thicket regime 下，大部分 random directions 本身 就是 提升 性能 的，错误 的 reward gradient 只要不把你推离 thicket，就 依然 能 偶然 命中 好 的 方向。参考 [Spurious Rewards](https://arxiv.org/abs/2506.10947)。
*   **MAML & Meta-Learning**: Finn et al. (2017) 的 MAML 试图 寻找 一个 init，使得 任何 task 都 只需 一步 梯度 就能 适配。这篇 paper 暗示，大规模 pretraining 隐式 地 实现了 MAML 的 目标。参考 [MAML](https://arxiv.org/abs/1703.03400)。

总而言之，Neural Thickets 彻底 改变了 我们 对 post-training 的 intuition。Pretraining 并非 只是 寻找 一个 起点，它 实际上 是 在 极高维 空间 中 构建了 一个 充满 专才 的 森林。Gradient descent 只是 在 这片 森林 里 砍出 一条 路，而 RandOpt 证明 了，我们 其实 可以 直接 闭上眼睛 随机 摘取 K 个 果实，然后 通过 majority vote 组合成 一个 强大 的 集体 智慧。这 对于 理解 LLM 的 representation geometry 以及 设计 未来 超大规模 并行 训练 架构 具有 深远 的 指导 意义。
