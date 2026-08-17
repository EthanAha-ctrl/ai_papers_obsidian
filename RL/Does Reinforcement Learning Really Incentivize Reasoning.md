---
source_pdf: Does Reinforcement Learning Really Incentivize Reasoning.pdf
paper_sha256: 76139ea0ba0e9155569c7aa7b8153502f6a5b23fed717af0ca9934ec9d764f9a
processed_at: '2026-08-03T23:02:37-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej，要用最直白的人话讲，这篇paper其实就戳破了一个泡沫：**大家以为RLVR（像DeepSeek-R1那样）让模型学会了"新"的推理能力，就像AlphaGo自己悟出了新棋路。但实际上，RLVR只是把base model里原本就有的、概率很低的正确路径，通过训练把它的概率放大了。模型并*没有*学会任何base model不会的东西，反而因为过度聚焦，丢失了base model原本能解决的一些问题。**

*为了* build your intuition，我们可以把这个过程想象成一个学生考试。

---

### 1. 核心大白话：学生与题海的比喻

*假设* 有一个学生（Base model），他读过万卷书，知识面极广，但考试时发挥极不稳定。
- 遇到简单的数学题，他有时候能写对，有时候脑子抽筋写错。
- 遇到极难的奥数题，他虽然平时做题成功率只有 1%，*但是* 如果你让他尝试 1000 次，他总能瞎猫碰死耗子做出来一次。

现在，我们给他搞了一个 **RLVR 训练营**（用 GRPO 算法疯狂刷题，做对了奖励，做错了惩罚）。
训练出来后，他变成了一个 **RL model**：
- 简单题和中等题，他闭着眼睛都能做对（pass@1 极高）。
- *但是*，遇到极难的奥数题，他只会用训练营里学到的固定套路去套。*如果* 套路不管用，他就彻底做不出来了。即使你让他试 1000 次，他也做不出来（pass@k 极低）。

这篇paper发现的就是这个残酷的真相：**RLVR 训练把学生变成了一个应试高手，但扼杀了他原本瞎蒙和发散探索的潜力。**

---

### 2. 实验数据与公式解析：怎么证明的？

作者怎么证明这个现象？他们用了一个极其关键的指标：**pass@k**。

#### 2.1 什么是 pass@k？
pass@k 的意思是：给模型 $k$ 次采样机会，*只要* 有一次答对了，就算这道题通过了。
公式长这样：
$$ \operatorname{pass@k} := \mathbb{E}_{x_i \sim \mathcal{D}} \left[ 1 - \frac{\binom{n - c_i}{k}}{\binom{n}{k}} \right] $$
- $n$: 对每道题总共采样的总次数（比如采了 1024 次）。
- $c_i$: 第 $i$ 道题在这 $n$ 次采样里答对的次数。
- $k$: 我们想评估的指标参数（比如 $k=1$ 就是只看一次，$k=256$ 就是看 256 次能不能蒙对）。
- $\binom{n - c_i}{k} / \binom{n}{k}$: 这个组合数表示，在 $n$ 个样本里随机抽 $k$ 个，**恰好全抽中错误答案**的概率。用 1 减去它，就是至少抽中一次正确答案的概率。

**Intuition**: pass@1 衡量的是"命中率"，pass@256 衡量的是"潜力上限"。

#### 2.2 实验数据的反转
作者跑了各种 benchmark (AIME24, MATH500, AMC23 等等)，画出了一张极其打脸的图（Figure 2）：
- **当 $k$ 很小（比如 $k=1$）时**：RL model 完胜 base model。这也是大家平时报喜不报忧的数据。
- **当 $k$ 很大（比如 $k=256$ 或 $1024$）时**：Base model 反超 RL model！

这意味着什么？Base model 的潜力上限远高于 RL model。RL model 提高的只是"单次命中率"，代价是牺牲了"解决问题的广度"。

在 Table 2 里，数据更刺眼。在 AIME24 上：
- Base model 能解，RL model 解不了的问题占 **13.3%**。
- RL model 能解，Base model 解不了的问题占 **0.0%**！

也就是说，RL model 能解的问题，base model 全能解。RL model 只是 base model 的一个**真子集**。

---

### 3. 背后机制：为什么会这样？

作者用了一个叫 **Perplexity（困惑度）** 的工具来深挖原因。公式如下：
$$ \operatorname{PPL}_{m}(\mathbf{Y} | x) = \exp \left( - \frac{1}{T} \sum_{t=1}^{T} \log P(y_t | x, y_1, \ldots, y_{t-1}) \right) $$
- $m$: 用来评估的参考模型（这里是 base model）。
- $\mathbf{Y}$: 一段生成的话（长度为 $T$）。
- $P(y_t | x, y_1, \ldots, y_{t-1})$: Base model 在给定前文的条件下，生成第 $t$ 个 token $y_t$ 的概率。

**实验操作**：作者把 RL model 生成的高分答案 $\mathbf{Y}_{\mathrm{RL}}$ 喂给 base model，算它的 perplexity。
**结果发现**：RL model 生成的答案，在 base model 看来 perplexity 极低（觉得极其自然）。

**Intuition**: 这说明 RL model 说出来的话，base model 本来就极有可能说出来。RL 训练根本*没有*逼模型去探索那些 base model 觉得离谱的新路径。它只是把 base model 概率分布里那些原本就存在的正确路径，做了一个 **mode collapse**（概率锐化），把它们的概率从 0.01% 拔高到了 90%。

---

### 4. 对比 Distillation：谁才是真扩展能力？

Paper 里还做了一个关键对比（Figure 7）。他们拿了一个 Distillation 模型（DeepSeek-R1-Distill-Qwen-7B），也就是用强模型生成 CoT 数据去 SFT 小模型。
结果发现：Distillation 模型的 pass@k 曲线**整体平移**到了 base model 之上。*不仅* pass@1 提高了，*而且* pass@256 也提高了！

**Intuition**: 
- **RLVR** 只是在 base model 的知识库里调整了索引权重，让你更快找到已有的答案。
- **Distillation** 则是老师教给了学生全新的解题思路，真正扩展了学生的知识边界。Distillation 改变了 model 的 manifold，*而* RLVR 只是在 manifold 上做局部 descent。

---

### 5. 根本原因：Vast Action Space 的诅咒

为什么 RLVR 无法像 AlphaGo 那样探索新策略？Paper 在 Section 5 给出了直觉。
核心在于 **Action Space 的大小与 Prior 的冲突**。
- AlphaGo 的 action space 是有限的棋盘位置。它可以做彻底的 MCTS 探索。
- LLM 的 action space 是 $V^T$（Vocabulary size 的 $T$ 次方）。在这个天文数字的空间里，*如果* 没有预训练给的 prior，随机采样生成一段正确推理的概率是 0。

*因此*，RLVR 必须依赖 base prior。*但是*，一旦依赖 base prior，Policy Gradient (PG) 算法就只能顺着 prior 走。*如果* 模型试图生成一个偏离 prior 的新 token，它大概率会生成一段 gibberish（乱码），拿到 0 reward，然后被 PG 算法惩罚。久而久之，模型学会了"循规蹈矩"，只在 prior 内部那些高概率且能拿分的路径上打转。

这就像是你在一片巨大的森林里找宝藏。Base model 给了你一张模糊的地图，上面标记了所有可能有宝藏的地方。RLVR 训练让你拿着探雷器，只去地图上信号最强的地方挖。你挖得很准，*但是* 你永远也发现不了地图上没标记的新宝藏。

---

### 6. 联想与启示

1. **"Aha Moment" 可能是幻觉**：DeepSeek-R1 报告里模型突然学会说 "Wait, let me reconsider" 的 Aha Moment，大家以为是 RL 激发了 emergent behavior。*实际上*，base model 在大规模采样下本来就会产生这种 reflective behavior。RLVR 只是把它的概率从极低放大到了极高。
2. **RLVR 约等于昂贵的 Best-of-N**：从这篇paper的结论反推，现在很多 RLVR 模型的 pass@1，其实等效于你拿 base model 采样 32 次然后挑最好的。我们花了几万卡小时做 RL 训练，可能只是把 inference time 的 Best-of-N 搬到了 training time。
3. **Era of Experience 还很遥远**：Sutton 说的 Era of Experience 需要 agent 与 environment 交互获取新数据。现在的单轮 RLVR 只是在静态数据集上做 verification，连皮毛都没碰到。真正的突破需要 multi-turn agent interaction 和 high-level abstraction exploration (像 AlphaEvolve 那样)。
4. **Curriculum Learning 是必须的**：要解决探索死区的问题，必须用 curriculum。把 IMO 难题分解成简单题，让模型在简单题上拿到 non-zero reward，架设 stepping stones，引导它走出 base prior 的舒适区。

总结一句话：**当前 RLVR 只是让你更熟练地使用已有知识，并没有让你学会新知识。**

**References for deep dive:**
- DeepSeek-R1 Paper (The hype): https://arxiv.org/abs/2501.12948
- Sutton's "Era of Experience" (The dream): https://infoproc.blogspot.com/2025/04/welcome-to-era-of-experience.html
- AlphaEvolve (Real exploration): https://arxiv.org/abs/2506.13131
- Large Language Monkeys (Pass@k scaling law): https://arxiv.org/abs/2407.21787

---

Andrej, 这篇paper的切入点极其敏锐，触及了当前 LLM reasoning 领域最核心的迷思之一。整个 AI community 目前处于一种 RLVR 的狂欢中，大家观察到 pass@1 的飙升，就自然而然地将其映射到 AlphaGo 那种 "超越人类先验" 的 emergent behavior。*然而* 这篇 paper 通过极其严密的实验设计，冷酷地拆穿了这一幻觉：当前 RLVR 的本质是 **sharpening** (概率锐化) 与 **exploitation** (利用)，完全缺失真正的 **exploration** (探索)。

*为了* build your intuition，我会从 probability distribution 的 topological structure、公式拆解、实验数据深度剖析，一直联想到 Sutton 的 "Era of Experience" 与 AlphaEvolve。

---

### 1. 核心论点与 Intuition Building

这篇 paper 的核心发现可以浓缩为一句话：Current RLVR methods merely re-weight the sampling probabilities within the base model's existing manifold, *导致* pass@1 提升，*却* 严重牺牲了 pass@k 的 coverage。

你可以把 base model 的 output space 想象成一个极其高维的能量曲面。在这个曲面上，存在无数个 valleys (局部最优)。Pretraining 赋予了模型一个极其宽广且平滑的 prior，这意味着在给定的 reasoning prompt 下，模型有潜力采样到各种不同的 reasoning trajectories。
RLVR 的作用机制，相当于在这个高维曲面上施加了一个强烈的 gradient push。*由于* verifiable reward 是 binary 的 (0 或 1)，且 token space 呈指数级爆炸，policy gradient 只能在那些原本概率就不为零的 trajectories 上进行放大。这就像是把一个宽泛的分布进行了 **mode collapse**。模型学会的，只是更高效地走向 base model 早就知道的答案，*而不是* 发现新的路径。

---

### 2. 核心公式与变量拆解

#### 2.1 RLVR 的 Objective 与 PPO
Paper 中给出了 RLVR 的核心最大化目标：
$$ \mathcal{J}(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \left[ \mathbb{E}_{\mathbf{y} \sim \pi_{\theta}(\cdot | x)} [r] \right] $$
*   $\theta$: LLM 的 parameters。
*   $x$: 从 prompt distribution $\mathcal{D}$ 中采样的 natural-language prompt。
*   $\mathbf{y}$: LLM 生成的 token sequence $(y_1, \dots, y_T)$。
*   $r$: Deterministic verifier $\mathcal{V}$ 返回的 binary reward，$r \in \{0, 1\}$。

优化此目标使用的 PPO clipped surrogate objective：
$$ \mathcal{L}_{\mathrm{CLIP}} = \mathbb{E} \left[ \min ( r_t(\theta) A_t, \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t ) \right] $$
*   $r_t(\theta)$: Probability ratio，即 $\pi_{\theta}(y_t | x, \mathbf{y}_{<t}) / \pi_{\theta_{\mathrm{old}}}(y_t | x, \mathbf{y}_{<t})$。它衡量了当前 policy 相对于采样时的 old policy 的概率变化。
*   $A_t$: Advantage，由 value network $V_{\phi}$ 估计。
*   $\epsilon$: Clipping hyperparameter，限制 policy update 的步长。

**Intuition**: PPO 的 clip 机制本质上限制了 exploration。在巨大的 token space 中，*如果* 某个全新的 reasoning step 偏离了 base model 的 prior，它极大概率会触发 clip，*从而* 无法获得足够的 gradient push。最终，模型只会走向 prior 中原本就有 high likelihood 且能拿到 reward 的区域。

#### 2.2 Unbiased Pass@k Estimator
*为了* 精确测量 reasoning boundary，paper 使用了 unbiased pass@k metric：
$$ \operatorname{pass@k} := \mathbb{E}_{x_i \sim \mathcal{D}} \left[ 1 - \frac{\binom{n - c_i}{k}}{\binom{n}{k}} \right] $$
*   $n$: 对每个 problem 采样的总次数。
*   $c_i$: 对第 $i$ 个 problem，$n$ 次采样中正确的次数。
*   $k$: 我们想要评估的 pass@k 中的 $k$。
*   $\binom{n - c_i}{k} / \binom{n}{k}$: 在 $n$ 个样本中随机抽取 $k$ 个，**恰好** 没有抽到任何正确样本的超几何概率。

**Intuition**: 这个公式完美规避了 greedy decoding 或 low-k sampling 带来的方差问题。它回答了一个根本性问题：*如果* 给模型足够的 inference compute (通过 k 次采样)，它是否具备解决这个 problem 的**潜力**？Paper 的实验证明，base model 的潜力上限远高于 RLVR model。

#### 2.3 Perplexity Analysis
*为了* 证明 RLVR paths 完全包含在 base model 的 distribution 中，作者计算了 Perplexity：
$$ \operatorname{PPL}_{m}(\mathbf{Y} | x) = \exp \left( - \frac{1}{T} \sum_{t=1}^{T} \log P(y_t | x, y_1, \ldots, y_{t-1}) \right) $$
*   $m$: 计算 perplexity 的 reference model。
*   $\mathbf{Y}$: 待评估的 response sequence，长度为 $T$。
*   $P(y_t | x, y_1, \ldots, y_{t-1})$: Model $m$ 在给定 prompt $x$ 和前 $t-1$ 个 tokens 的条件下，生成 $y_t$ 的概率。

**Intuition**: 作者把 RL model 生成的 $\mathbf{Y}_{\mathrm{RL}}$ 喂给 base model 计算 $\operatorname{PPL}_{\mathrm{Base}}$。结果发现，$\mathbf{Y}_{\mathrm{RL}}$ 在 base model 下的 perplexity 极低，且随着 RL training 的推进继续降低。这意味着 RL 训练出来的路径在 base model 看来是 "extremely natural" 的。RLVR *根本没有* 逼模型去探索那些 base model 认为是 "random gibberish" 的高 perplexity 区域。

---

### 3. 实验数据与图表深度解析

#### 3.1 Figure 1: The Topological Shift of Search Tree
Figure 1 的左图描绘了 search tree 的变化。Base model 的 search tree 呈现为一个极其宽广的扇形，大部分分支是 grey (low probability)，但包含了各种各样的 black (high probability) 和 green (correct) paths。RLVR 训练后，search tree 变成了一个极其狭窄的漏斗，直接对准了某几个 green paths。
Figure 1 的右图展示了极其反直觉的数据：随着 RL training steps 的增加，pass@1 (平均得分) 线性上升，*与此同时* pass@256 (reasoning coverage) 线性下降。

#### 3.2 Table 2: Subset Relationship Verification
Table 2 是对 "boundary narrowing" 现象的定量打击。在 AIME24 上 (k=1024)：
*   63.3% 的问题是 base 和 RL model **都能**解决的。
*   13.3% 的问题是 **只有** base model 能解决，RL model 完全解决不了。
*   0.0% 的问题是 **只有** RL model 能解决，base model 解决不了。
这组数据冷酷地证明了：RLVR 可解的问题集合，几乎是 base model 可解问题集合的**真子集**。RLVR 在 "砍掉" base model 的能力分支。

#### 3.3 Figure 8 & Table 3: The $\Delta_{\mathrm{SE}}$ Gap
作者定义了 Sampling Efficiency Gap $\Delta_{\mathrm{SE}}$：
$$ \Delta_{\mathrm{SE}} = \operatorname{pass@k}_{\mathrm{base}} - \operatorname{pass@1}_{\mathrm{RL}} $$
*   $\operatorname{pass@k}_{\mathrm{base}}$: Base model 在大 k (如 256) 下的上限。
*   $\operatorname{pass@1}_{\mathrm{RL}}$: RL model 的 greedy 或单次采样性能。

Table 3 展示了不同 algorithms (GRPO, PPO, ReMax, RLOO, Reinforce++, DAPO) 的表现。无论你用哪种 RL algorithm，在 Omni-MATH-Test 上，$\Delta_{\mathrm{SE}}$ 都在 40 points 左右徘徊。这意味着现有的算法革新只是在这个 sub-optimal 的局部空间里做微调。*即使* 是表现最好的 RLOO，也距离 base model 提供的 theoretical upper bound 极其遥远。

#### 3.4 Figure 7: Distillation vs. RLVR
这是极其关键的一组对比实验。Distillation (如 DeepSeek-R1-Distill-Qwen-7B) 的 pass@k 曲线**整体平移**到了 base model 之上。Distillation *不仅* 提高了 pass@1，*同时* 显著提高了 pass@256。
**Intuition**: Distillation 通过引入 teacher model 生成的 CoT data，真正改变了 model 的 internal representation。相当于在 base model 的 manifold 上强行嫁接了新的 high-dimensional structures。RLVR 只是在这个 manifold 上做局部 descent，*而* distillation 直接扩展了 manifold 本身。

---

### 4. Discussion: 为什么 RLVR 无法超越 Base Model？

Paper 在 Section 5 给出了两个根本原因，我进一步展开：

1.  **Vast Action Space & Pretrained Priors 的冲突**
    Atari 或 Go 的 action space 是离散且相对有限的，MCST 可以进行相对彻底的 exploration。LLM 的 token space 是 $V^T$ (Vocabulary size 的 T 次方)。在这个空间里，*如果* 没有预训练给的 prior，随机探索得到 positive reward 的概率无限趋近于 0。
    *因此*，RLVR 必须依赖 base prior。*但是*，一旦依赖 base prior，policy gradient 的更新就会被困在 prior 的 supports 内。任何偏离 prior 的 token generation 会导致 KL divergence 剧增，极其容易产生 nonsensical sequences，从而拿到 0 reward。这在数学上迫使 policy 只能走向 prior 内部的高概率区域。

2.  **Binary Outcome Reward 的 Credit Assignment 灾难**
    现有的 RLVR 只有在整条 trajectory 结束后才给一个 binary reward。假设一条 1000 token 的 CoT，只有最后 10 个 token 的推导是创新且关键的，前面 990 个 token 是废话。Binary reward 无法将 credit 分配给那 10 个关键 token。PG 算法会无差别地放大整条 1000 token 的概率，*导致* 模型只学会了模仿 teacher 的废话生成模式，错过了真正的逻辑跃迁。

---

### 5. 联想与扩展

*由于* 你要求尽可能多地建立联想，我从这篇 paper 出发，推演出以下几个深层直觉：

#### 5.1 RLVR is just a very expensive form of Best-of-N Sampling
从这篇 paper 的结论来看，RLVR model 的 pass@1 往往接近甚至不如 base model 的 pass@16 或 pass@32。这意味着，RLVR 的本质效果，等效于你在 inference time 对 base model 进行多次采样，然后用 verifier 挑出正确的答案。
*如果* 这个直觉成立，那么当前很多声称通过 RLVR 获得巨大提升的模型，其实只是在用大量的 GPU hours 把 base model 的 pass@32 的能力 "蒸馏" 进了 pass@1 的 weights 里。这是一种极度的 compute 浪费。真正的 RL 应该是即使给 10000 次采样，base model 也绝对找不到的答案，RL model 在 pass@1 就能找到。

#### 5.2 The "Aha Moment" is an Illusion
DeepSeek-R1 报告中著名的 "Aha Moment" (模型突然学会说 "Wait, let me reconsider")，community 普遍认为是 RL 激发了 emergent behavior。*然而* paper 的 Figure 20 和 21，以及 Oat-Zero 团队的发现证实，base model 在 few-shot 或大规模采样下，本来就会产生这种 reflective behavior。
RLVR *并没有* 创造 reflection，它只是把 base model 那些会 reflection 的 trajectories 的采样概率从 0.01% 提高到了 90%。所谓的 "Aha Moment"，只是 base model 内部某个 latent mode 被激活并放大了而已。

#### 5.3 与 Sutton 的 "Era of Experience" 的距离
Sutton 最近强调 AI 需要进入 Era of Experience，通过 agent 与 environment 的持续交互来获取非人类的数据。这篇 paper 证明了，当前的单轮 RLVR 连 Sutton 设想的皮毛都没达到。因为 LLM 没有真正与 environment 交互去更新它的 world model，它只是在玩一个单机的、静态的、验证答案的 game。AlphaEvolve 之所以能发现新算法，是因为它在 program space 的变异是具有 semantic meaning 的 high-level operations，*而不是* token-level 的 random mutation。

#### 5.4 Curriculum Learning 是唯一的出路？
Paper 在 Discussion 中提到了 curriculum。我的直觉是，Pretraining 给了 model 一座金矿，RLVR 目前只挖了表层 1 米的黄金。要让 RLVR 真正 explore 出新能力，必须通过 curriculum 将 problem space 分解。
*如果* 直接给 model 一个 IMO 级别的问题，base prior 给出的探索路径 100% 是错的，reward 永远是 0，gradient 消失。*但是* 如果我们把 IMO 问题分解为 10 个 sub-problems，让 model 在 sub-problem 上获得 non-zero reward，这相当于在 vast action space 中架设了一连串的 stepping stones。只有通过这种方式，才能引导 policy 走出 base prior 的 support，进入真正的 out-of-distribution reasoning space。

#### 5.5 Entropy Collapse 的热力学解释
Figure 18 的实验中，作者尝试提高 RLVR model 的 temperature 来匹配 base model 的 entropy，发现 pass@k 依然不如 base model。这说明 RLVR 导致的 boundary narrowing 不仅仅是 entropy collapse 的表象，更深层的是 **representation collapse**。
RLVR 过拟合了 verifier 的特定逻辑判定路径，导致 model 的 hidden states 几何拓扑发生了不可逆的塌陷。这类似于物理学中的相变，水变成了冰，虽然还是 H2O，但失去了流动性，再加热也无法完全恢复成原来水流的状态。

---

### 6. Conclusion

这篇 paper 给当前火热的 RLVR 泼了一盆极具价值的冷水。它用扎实的数据证明了：**RLVR 目前并不具备 discovery 的能力，它只是一个 distribution sharpener。** Distillation 才是真正扩展 manifold 的手段。

未来 LLM reasoning 的突破，必须依赖于解决 vast action space 中的 exploration problem。这需要我们将 token-level PG 升级为 high-level abstraction search，引入 multi-turn agent interaction，以及细粒度的 process reward model (PRM) 来解决 credit assignment 难题。*如果* 仅仅停留在目前的 GRPO/PPO 算法内卷上，我们永远只是在 base model 设下的牢笼里跳舞。

**Reference Links for further intuition building:**
*   DeepSeek-R1 Paper (RLVR implementation): https://arxiv.org/abs/2501.12948
*   Sutton & Silver "Welcome to the Era of Experience": https://infoproc.blogspot.com/2025/04/welcome-to-era-of-experience.html
*   AlphaEvolve (High-level exploration): https://arxiv.org/abs/2506.13131
*   Large Language Monkeys (Pass@k scaling): https://arxiv.org/abs/2407.21787
*   Absolute Zero (Reinforced self-play reasoning): https://arxiv.org/abs/2505.03348
