---
source_pdf: AskingtheRightQuestions Improving ReasoningwithGeneratedSteppingStones.pdf
paper_sha256: a12944b0f1bbe46e616558c44ed7ceb90d87ce84458f92d4beae961c71b92d05
processed_at: '2026-07-18T09:23:06-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Asking the Right Questions: Improving Reasoning with Generated Stepping Stones 深度讲解

Hey Andrej, 这是一篇非常 meta-level 的 reasoning paper，作者来自 FAIR at Meta、Stanford 和 Oxford，一作是 Hengyuan Hu（曾经在 Meta 工作，现在在 Stanford）和 Tingchen Fu。这个工作的核心 insight 其实非常人本主义——**人类解决难题时，不会一上来就硬刚，而是先构造 stepping stones**：simplifications、special cases、alternative framings，build intuition 之后再回来啃原题。ARQ (Asking the Right Questions) 就是把这个 cognitive strategy 显式地注入 LLM reasoning pipeline。

下面我会从 motivation、formulation、inference-time 方法、post-training 流程、实验结果和 broader implications 几个层面详细拆解。

---

## 1. Motivation: 为什么这个问题值得研究

当前 reasoning LLM 的 post-training recipe 基本上是两段式：
1. **SFT on curated CoT data**（e.g. [DeepSeek-R1](https://arxiv.org/abs/2501.12948), [s1](https://arxiv.org/abs/2501.19393), [LIMO](https://openreview.net/forum?id=T2TZ0RY4Zk)）
2. **RL on verifiable rewards**（math exact match, code unit tests）

Inference-time 又叠了 majority voting、self-refine、tree search 等 scaffold（参考 [Welleck et al. 2024](https://arxiv.org/abs/2406.16838) 的 survey）。

但是这些方法都在做一个事：**教模型如何更高效地解 *给定* 的问题**。没有人系统地研究：模型能不能 *自己提出一个相关但更简单* 的问题，先解这个 stepping stone，然后用它的解来 inform 原题的求解？

这其实是 Newell & Simon 1972 年 Human Problem Solving 里的经典思想——人类在 long-horizon reasoning 中天然会这么做（参考 [Ho et al. 2022, Nature](https://www.nature.com/articles/s41586-022-04743-9)）。Deep learning researcher 自己也是这么做的：先在小 dataset/toy setting 上 prototype，验证 idea 的 correctness，再 scale up。这篇 paper 的核心 question：**LLM 能不能学会 ask the right questions?**

---

## 2. Formal Framework: ARQ 的数学结构

### 2.1 Notation

- $x$: 用 natural language 表述的 problem
- $\pi$: LLM-based stochastic solver，每次采样 $y \sim \pi(x)$
- $R(y)$: external reward function / verifier
- $\phi$: LLM-based stochastic stepping stone generator，$z \sim \phi(x)$
- $z$: 生成的 stepping stone 问题
- $y_z$: stepping stone 的解，$y_z \sim \pi(z)$

### 2.2 Inference-time pipeline

标准 pipeline 是 $y \sim \pi(x)$。ARQ 改成三段：

$$z \sim \phi(x) \quad \text{(1. 生成 stepping stone)}$$
$$y_z \sim \pi(z) \quad \text{(2. 解 stepping stone)}$$
$$y \sim \pi(x; z, y_z) \quad \text{(3. 把 (z, y_z) 当 in-context example 解原题)}$$

这里有个细节值得 build intuition：**$(z, y_z)$ 是作为 in-context example prepend 到原 prompt 里的**，本质上是 one-shot in-context learning，但 demonstration 不是训练集里随便挑的，而是 model 自己 *针对当前问题* 构造出来的。这有点像 retrieval-augmented reasoning，但 retrieval 的 source 不是 corpus，而是 generator 自己的"想象空间"。

### 2.3 Stone quality 的 score function (核心公式)

这是整篇 paper 的 theoretical anchor：

$$S(z, x) = \mathbb{E}_{y_z \sim \pi(z),\, y \sim \pi(z, y_z, x)} R(x, y) \tag{1}$$

逐项解释：
- $S(z, x)$: stepping stone $z$ 在 target problem $x$ 上的 score
- $\mathbb{E}$: 期望算子
- $y_z \sim \pi(z)$: 从 solver $\pi$ 采样 stepping stone $z$ 的解，下标 $z$ 表示这是 stepping stone 的解
- $y \sim \pi(z, y_z, x)$: 给定 stepping stone $z$、其解 $y_z$ 和原问题 $x$，从 solver 采样原问题的解
- $R(x, y)$: reward function，在 math 任务上就是 exact match

这个期望是 over MC rollouts 的：实验里用 $k=20$ rollouts 来估计 $S(z, x)$。

**Intuition**: 这个 score function 本质上是把"这块石头好不好"定义成"用了这块石头之后 solver 在原题上的期望成功率"。这是个非常巧妙的 *indirect* reward——它不评价 stone 本身的 quality（stone 本身没有 ground truth），而是评价 stone 对 downstream task 的 *causal contribution*。这跟 [process reward models](https://arxiv.org/abs/2305.20050) 的思路有相似之处，但 PRM 评价的是 reasoning step，ARQ 评价的是 *auxiliary problem*。

### 2.4 Multi-stone 扩展

两种策略，理解清楚很重要：

**Sequential** (build a curriculum):
$$z_i \sim \phi(x; z_1, \ldots, z_{i-1})$$
$$y_{z_i} \sim \pi(z_i)$$
$$y \sim \pi(x; z_1, y_{z_1}, \ldots, z_i, y_{z_i})$$

每生成一个新 stone，都把已有的 stones + 原题一起塞进去，让 generator "填补 gap"。这构建的是一个 *平铺式* 的 curriculum。

**Recursive** (decompose the problem):
$$z_i \sim \phi(z_{i-1}), \quad z_0 = x$$
$$y_{z_{i-1}} \sim \pi(z_{i-1}; z_i, y_{z_i})$$

新 stone 总是基于上一个 stone 生成，求解时是 *reverse order*——先解最简单（最深）的 stone，然后用它的解 inform 上一层 stone，最终回到 $x$。这构建的是 *嵌套式* decomposition。

**Important detail**: 实验里 sequential 大幅提升，recursive 反而 degrade。原因可能是 recursive 路径上错误会 *propagate*——任何一个中间 stone 的解错了，下游全错。而 sequential 是 *additive* context，错一个 stone 不会污染整个 context。

### 2.5 一个 implementation subtlety

paper 提到：把 generator 和 solver 的 thinking tokens 都 *strip 掉*，只保留 final output 再传给下一个 module，性能会更好。这是个
