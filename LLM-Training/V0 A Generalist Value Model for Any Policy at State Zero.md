---
source_pdf: V0 A Generalist Value Model for Any Policy at State Zero.pdf
paper_sha256: 6fd44f0745ab7809d3729555ada9e5b6b417e14a8afef1809dd051a2a3691367
processed_at: '2026-08-13T00:06:25-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# V₀ 用人话说一遍

## 一句话版本

你有一个 LLM 在做 RL 训练，你想知道"这道题这个模型能不能做对"。传统做法是训练一个 value model 跟 policy 绑一起跑，policy 变了它就得重训。V₀ 说：别把 policy 的能力塞进 value model 的权重里，把它当成"上下文"喂进去——给 value model 看这个 policy 历史上做过哪些题、对错如何，它自己就能推断。

## 问题是什么

LLM post-training 的 RL（PPO/GRPO/RLVR）里，你需要一个 baseline 来算 advantage。PPO 的做法是训练一个 value model $V(x)$ 预测"policy 在 prompt $x$ 上能拿多少 reward"。这个 value model 跟 policy 一样大，policy 每变一步它就得跟着训一步，不然就 stale 了。这叫 **coupling dilemma**——两个模型绑死，又贵又容易不稳定。

GRPO（DeepSeek 那条线）的解法是干脆不要 value model，对同一个 prompt 采样 $G$ 个 rollout，用 group mean 当 baseline。问题是：如果这道题太简单（$G$ 个全对）或太难（$G$ 个全错），advantage 全是 0，gradient 死掉。你得采样很多次才能避免这个 collapse，sampling cost 又上去了。

所以这是一个 efficiency vs. stability 的 trade-off：PPO 端训练贵，GRPO 端采样贵。

## V₀ 的核心 insight

一个 policy 的"能力"不一定要编码在 value model 的参数里。你可以把它显式地当作 input 的一部分。

具体来说：给定 policy $\pi$，你收集它历史上做过的 $N$ 道题和对错记录 $\mathcal{C}_\pi = \{(x_1, r_1), (x_2, r_2), ..., (x_N, r_N)\}$。然后 value model 的输入是 `(新题目 $x$, 历史记录 $\mathcal{C}_\pi$)`，输出是"这道新题 policy $\pi$ 能做对的概率"。

这跟你看一个学生做了 100 道题的成绩单，就能大致判断他在新题上能不能做对一样。你不需要把学生的"能力"装进你脑子里的某个参数，你只需要看他的历史记录。

**这就是 in-context learning 应用到 value estimation 上。** Policy 从"隐变量"（藏在 value model 权重里）变成了"显式 context"（一段历史记录）。

好处是：policy 变了，你只要更新 context（换一批历史记录），value model 参数完全不动。零 gradient 适应任意 policy。

## 架构怎么实现的

难点在于：LLM 的 embedding 是 1024 维 entangled 向量，而你想要做的"根据历史推断概率"这件事，本质上是一个 tabular Bayesian inference 问题（给定一堆 (feature, label) 对，预测新 feature 的 label 概率）。

LLM embedding 跟 tabular data 之间有个 gap：tabular data 每一列有固定含义（age、income），LLM embedding 每一维没有固定含义。直接喂给 tabular 推断模型会乱掉。

V₀ 的架构三段式：

```
题目文本 → [Semantic Backbone: Qwen3-Embedding-0.6B, frozen]
         → 1024 维 embedding h
         → [Residual Query Adapter: 168 个 learnable queries + cross-attention]
         → 168×6 维 structured features z
         → [TabPFN: frozen, in-context Bayesian classifier]
         → 概率 P(r=1 | x, C)
```

中间那个 Residual Query Adapter 是关键创新。它就像一个棱镜——白光（entangled embedding）进去，被折射成不同光谱（168 个独立的 capability 维度）。每个 query 学会去"探测"embedding 里的某一种能力维度（比如算术复杂度、几何推理、代码能力），通过 cross-attention 把这些维度提取出来，变成 tabular feature 的列。

168 这个数字也有讲究：$168 \times 6 = 1008 \approx 1024$，维度差不多对齐，6 能被 3 整除（TabPFN 内部按 3 个一组处理 feature）。

TabPFN 本身是 Hollmann et al. 在 Nature 2025 发的 tabular foundation model，pre-train 时在 millions 个 synthetic tabular dataset 上做 in-context 分类，单次 forward pass 就能做 Bayesian posterior inference。V₀ 直接拿来当 probabilistic reasoning head，freeze 住不训。

整个 pipeline 里只有 Residual Query Adapter 是 trainable 的，参数量很小。Backbone 和 TabPFN 都 frozen。

## 理论上最漂亮的部分：shortcut learning 的分析

训练时有个坑：如果你直接用 cross-entropy loss 让模型预测 $P(r=1 | x, \mathcal{C})$，模型会偷懒。

为什么？因为不同 policy 的能力不同。一个 70B 模型的 context 里 70% 都做对，一个 1.5B 模型的 context 里 30% 做对。模型只要看 context 的"平均成功率" $\mu(\mathcal{C})$，就能猜个大概——不看具体 query $x$ 也能把 loss 降下来。

用信息论的话说：

$$I(Y; X, \mathcal{C}) = \underbrace{I(Y; \mathcal{C})}_{\text{偷懒就能拿到的}} + \underbrace{I(Y; X | \mathcal{C})}_{\text{真正要学的}}$$

$I(Y; \mathcal{C})$ 是"光看 context 就能拿到的信息"，$I(Y; X | \mathcal{C})$ 是"给定 context 后，看 query 还能多拿到的信息"。Gradient descent 偏好简单函数，会优先 fit 前者，忽略后者。

**解决办法**：加 pairwise ranking loss。从同一个 context $\mathcal{C}$ 里抽两道题 $x_i, x_j$，一道做对一道做错，让模型比较 logit 差：

$$\mathcal{L}_{\text{rank}} = -\log \sigma(s(x_i, \mathcal{C}) - s(x_j, \mathcal{C}))$$

妙处在于：同一个 context 下，任何 context-dependent 的 bias $b(\mathcal{C})$ 在做差时被减掉了：

$$s(x_i, \mathcal{C}) + b(\mathcal{C}) - s(x_j, \mathcal{C}) - b(\mathcal{C}) = s(x_i, \mathcal{C}) - s(x_j, \mathcal{C})$$

所以模型没法再靠"猜 policy 强弱"拿分，必须真正去比较 $x_i$ 和 $x_j$ 的相对难度。这迫使它学 $I(Y; X | \mathcal{C})$。

这个思路跟 DPO 几乎是同构的——DPO 用 log-ratio 差把 reward model 解耦到 policy 上，V₀ 用 logit 差把 value model 解耦到 context 上。都是 shift-invariance 的妙用。

但纯 ranking loss 只给相对序，下游 routing 任务需要 calibrated 概率。所以最终 loss 是：

$$\mathcal{L} = 0.25 \cdot \mathcal{L}_{\text{rank}} + 0.75 \cdot \mathcal{L}_{\text{CE}}$$

ranking 负责区分能力（discrimination），CE 负责校准概率（calibration），两者互补。

## 怎么用的

### 场景一：训练时的 budget allocation

GRPO 给每个 prompt 固定 sample $G$ 次，很浪费——简单题全对、难题全错，advantage 都塌掉。

V₀ 在 rollout 之前先预测"这个 prompt 当前 policy 的成功率 $p$"，然后根据 $p$ 动态分配 budget $B$。

作者推导出一个 utility 函数（衡量"分配 $B$ 次 rollout 能产生多少 gradient signal"）：

$$\text{Utility}(B, p) = B(1-p)[1 - (1-p)^{B-1}]$$

直觉：$(1-p)$ 是失败概率（要有错样本可学），$(1-p)^{B-1}$ 是剩下全失败的概率（要避免全错），中间 $B$ 是总样本数。这个公式在"有对有错"的 sweet spot 附近最大化。

用 greedy 算法分配 budget，在 OlympiadBench 上比 GRPO baseline 高 2%，AIME 2024 上从 41% 提到 50%。

### 场景二：推理时的 model routing

你有 11 个 model（0.6B 到 32B），来一个 query，你要选最便宜且能做对的 model。

每个 model 当成一个 policy，构造它的 capability context。V₀ 预测"每个 model 在这个 query 上的成功率 × cost 权衡"，选最优的。

通过调 cost-tradeoff 系数 $\beta$，画出 Pareto frontier——在准确率 vs. 推理成本上比 EmbedLLM、Model-SAT 都好。

最大优势：新 model 加入 fleet 或 pricing 变了，只需要更新 context，V₀ 参数不动，zero-shot 适应。

## 实验结果说了什么

主实验（Table 1）：三个架构（1.5B、4B、7B）上，V₀ 的 Intra-AUC（同一 policy 内区分题目的能力）全面碾压。1.5B 上 0.913 vs. Vanilla Value Model 的 0.840。甚至比"每步从头重训 value model"的 oracle 还好——说明 in-context 推断比 parameter fitting 更适合 non-stationary tracking。

Generalization（Table 2）：把 query ID 严格 disjoint 分到 train/test。Vanilla VM collapse 到 0.56（基本随机），V₀ 还有 0.71。证明 V₀ 学到的是"如何从历史推断能力"的 meta-knowledge，不是记住了具体 prompt。

Context length（Table 9）：$N \leq 128$ 时 AUC 卡在 0.54 附近——历史记录太少，不足以刻画 policy 能力。$N=256$ 后才真正 work。这跟统计学习理论一致：高维能力需要足够样本密度。

## 我觉得 deep 的地方和局限

**Deep 的地方**：把 value estimation 从"针对一个 policy 拟合一个 value function"重新定义成"学习一个 meta-function：如何从历史推断能力"。这是范式转变，跟 GPT-3 在 NLP 上展示 in-context learning 是同一类思路，只是搬到了 RL post-training 上。本质上是在学一个"能力推断器"，而非"能力本身"。

**架构上的 deep**：把语义理解和统计推断分工——Semantic Backbone 负责理解题目讲什么，TabPFN 负责根据统计模式做 Bayesian 推断。中间 Residual Query Adapter 当"翻译器"。LLM 自己做概率推断 calibration 很差，V₀ 用专门训练的 TabPFN 来做这部分，是个很实用的分工。

**局限一**：只在 state zero（initial prompt）上，不能做 token-level process supervision。当前 PRM (Process Reward Model) 还是需要 coupled training。如果能扩展到 token-level，会是很大突破。

**局限二**：只在 verifiable reward（0/1）上验证。continuous reward、preference reward 场景没测。

**局限三**：context $\mathcal{C}_\pi$ 是 random sample 的 256 对。实际上应该有 active selection——选 informative history，跟 Bayesian active learning 接轨。Paper 没做这个。

**局限四**：只在 single-turn reasoning 上测。multi-turn agentic、long-horizon RL 场景下，context 表征要重新设计。

**联想**：这个范式跟 Anthropic 的 Constitutional AI、RLAIF 有点像——都是用另一个 model 来提供 signal。但 V₀ 的不同在于它不是"提供 reward"，而是"提供对 policy 能力的认知"，更 meta 一层。如果 reward 本身来自 AI feedback（continuous score），V₀ 的 framework 需要扩展，但 concept 上是通的。

另外一个联想：V₀ 在 inference routing 上的应用，本质是把 inference-time compute scaling 从"best-of-N sampling"扩展到"best-of-fleet routing"——根据 query 难度动态选 model，而非在一个 model 上多 sample。这跟 o1 / R1 的 test-time compute 思路互补，可以叠加。

---

**相关链接**：
- V₀ 项目主页：https://now-join-us.github.io/V0
- TabPFN (Nature 2025)：https://www.nature.com/articles/s41586-024-08306-4
- TabPFN (ICLR 2023)：https://arxiv.org/abs/2207.01848
- Müller "Transformers can do Bayesian inference" (ICLR 2022)：https://arxiv.org/abs/2112.10510
- DeepSeekMath GRPO：https://arxiv.org/abs/2402.03300
- DPO：https://arxiv.org/abs/2305.18290
- Perceiver IO：https://arxiv.org/abs/2103.03206
- BLIP-2 Q-Former：https://arxiv.org/abs/2301.12597
- EmbedLLM (ICLR 2025)：https://openreview.net/forum?id=EmbedLLM
- Knapsack RL：https://arxiv.org/abs/2509.25849
- DOTS：https://arxiv.org/abs/2506.05316
- CURES：https://arxiv.org/abs/2510.01037
- PRM (Lightman 2023)：https://arxiv.org/abs/2305.20050
- DAPO：https://arxiv.org/abs/2503.14476
- DeepSeek-R1：https://arxiv.org/abs/2501.12948
- Inference scaling (Snell 2024)：https://arxiv.org/abs/2408.03314
- RLAIF (Bai 2022)：https://arxiv.org/abs/2212.08073

---

# V₀：一个 Generalist Value Model 的深度解析

## 1. 这篇论文的核心 motivation

Karpathy 你应该深有体会，LLM post-training 阶段，RLHF/PPO 最让人头疼的就是 value model。PPO 的 Actor-Critic 设计有 coupling dilemma：policy $\pi_\theta$ 不断 evolution，value model $V^\pi$ 必须同步训练来 track non-stationary target，cost 巨大。GRPO (DeepSeek 那条线) 把 value model 整个砍掉，用 group rollout 的 mean reward 做 baseline $\hat{V}(x) \approx \frac{1}{G}\sum R_i$，把 training cost 转嫁到 sampling cost：复杂 reasoning 任务上 reward 极易 collapse 成 all-zeros 或 all-ones，advantage 全为 0，gradient signal 死掉。

V₀ 提出一个非常 elegant 的 reframe：**policy capability 不应该藏在 value model 的 weights 里，应该作为 explicit context 输入**。也就是从 $V^\pi(s_0)$ 转到 $V(\mathcal{C}_\pi, s_0)$，其中 $\mathcal{C}_\pi = \{(x_i, r_i)\}_{i=1}^N$ 是 historical query-performance pairs。这样 $V_0$ 就是一个 generalist function，可以 zero-gradient adapt 到任意 policy——只要给它看这个 policy 的历史表现。

这本质上把 value estimation 从 **parameter fitting** 变成了 **in-context learning**，跟 TabPFN、GPT-3 in-context learning、Vision-Language Models 的 in-context demonstration 是同一类思路。

**项目主页**：https://now-join-us.github.io/V0
**arXiv**（按论文题目搜索）：https://arxiv.org/abs/2506.02834 (注: 此为推测链接, 实际需查证)

---

## 2. State Zero 的设定与 PPD 视角

V₀ 专注在 **state zero** $s_0$，也就是 initial prompt（不是 trajectory 中的 intermediate state）。所以这个 value model 输入是 `(query x, context $\mathcal{C}_\pi$)`，输出 $P(r=1 \mid x, \mathcal{C}_\pi)$。它不能告诉你 trajectory 中间 token 的 value，只能预测整条 rollout 完成后的 outcome reward。

形式化上，作者把这个问题写成 Posterior Predictive Distribution (PPD)：

$$P(r \mid x, \mathcal{C}_\pi) = \int P(r \mid x, \mathcal{M}) P(\mathcal{M} \mid \mathcal{C}_\pi) d\mathcal{M} \tag{1}$$

变量含义：
- $r \in \{0, 1\}$：binary outcome (verifiable reward)
- $x$：target query
- $\mathcal{M}$：latent capability model（隐变量，描述 policy 的真实能力分布）
- $\mathcal{C}_\pi = \{(x_i, r_i)\}_{i=1}^N$：观测到的历史 query-performance pairs
- $P(\mathcal{M} \mid \mathcal{C}_\pi)$：给定历史后对 capability 的 posterior
- $P(r \mid x, \mathcal{M})$：给定 capability 和 query 后的 likelihood

这个公式本质上是 Bayesian model averaging，跟 Müller et al. 2022 ICLR "Transformers can do Bayesian inference" 的工作 ([paper](https://arxiv.org/abs/2112.10510)) 一脉相承——他们证明 Transformer 可以 implicit 实现 Gaussian process 的 PPD。TabPFN 就是基于这个理论训练出来的 tabular foundation model ([Nature 2025](https://www.nature.com/articles/s41586-024-08306-4))。

---

## 3. V₀ 的架构：Semantic-Perception to Structured-Reasoning

架构图（Figure 2）分三块，这是这篇 paper 最有意思的工程部分。

### 3.1 Semantic Backbone

用 frozen **Qwen3-Embedding-0.6B**（$d_{\text{embed}}=1024$），对 context queries $\{x_i\}$ 和 target $x_t$ 都做 pooling，得到 $\mathbf{h} = \text{Pool}(f_{\text{enc}}(x)) \in \mathbb{R}^{d_{\text{embed}}}$。

这一步把离散 instruction 映射到连续 semantic manifold，捕捉 domain、difficulty 等 latent 信息。

### 3.2 Residual Query Adapter（核心创新）

问题在于：LLM embedding 是高度 entangled 的 1024 维向量，但 TabPFN 期望 **structured tabular features**——每列有固定含义（age, income...）。直接喂进去 TabPFN 会乱掉。

作者把这个 adapter 比作"Semantic Prism"，把自然光（entangled embedding）折射成不同光谱（独立 capability channels）。

设计上有两组 queries：

1. **Static Queries** $\mathbf{Q}_{\text{static}} \in \mathbb{R}^{K \times d_{\text{embed}}}$：168 个 learnable 参数，捕捉通用 capability dimensions（比如 arithmetic complexity、几何推理能力、code generation ability）。$K=168$。

2. **Dynamic offset** $\Delta \mathbf{Q} = G(\mathbf{h})$：由一个 generator $G$ 基于 input embedding $\mathbf{h}$ 生成 instance-specific offset。

最终 queries:
$$\mathbf{Q} = \mathbf{Q}_{\text{static}} + G(\mathbf{h})$$

然后通过 Multi-Head Attention，用 $\mathbf{Q}$ 去 probe embedding：

$$\mathbf{z} = \text{MHA}(\mathbf{q}=\mathbf{Q}, \text{k/v}=\mathbf{h}) \in \mathbb{R}^{K \times d_{\text{embed}}} \tag{4}$$

变量含义：
- $\mathbf{q}$：query 矩阵，shape $K \times d_{\text{embed}}$（其实是 $K \times d_{\text{proj}}$，论文里 $d_{\text{proj}}=6$）
- $\mathbf{k}, \mathbf{v}$：来自 backbone 的 $\mathbf{h}$，做 key/value
- $\mathbf{z}$：output features，$K$ 个 channels，每个 $d_{\text{proj}}$ 维

注意 $d_{\text{proj}}=6$ 的选择很巧妙：$168 \times 6 = 1008 \approx 1024$，并且 6 能被 3 整除（TabPFN 内部按 3 个一组 encode features）。

**这个设计联想**：跟 Perceiver (Andrew Jaegle 2021, [arxiv](https://arxiv.org/abs/2103.03206)) 用 learnable latents 做 cross-attention 把 high-dim input 压到 fixed-size；跟 DETR 的 object queries；跟 BLIP-2 的 Q-Former 几乎是同一思想家族——用一组可学习 queries 当 bottleneck，把 entangled representation 重构成 structured latent。

### 3.3 TabPFN Inference Head

TabPFN 是 Hollmann et al. (Nature 2025, [paper](https://www.nature.com/articles/s41586-024-08306-4)) 提出的 tabular foundation model，基于 pre-train 时在 millions of synthetic tabular datasets 上做 in-context Bayesian classification。它接收 transformed pairs $\{(\mathbf{z}_i, r_i)\}_{i=1}^N$ 作为 reference set，对 target $\mathbf{z}_t$ 给出 PPD：

$$\hat{r}_t \sim P(r \mid \mathbf{z}_t, \{(\mathbf{z}_i, r_i)\}_{i=1}^N) \tag{5}$$

整个 head **不更新参数**（论文实验 Table 6 证明 tune TabPFN 反而过拟合），单 forward pass 完成 Bayesian inference。这是把 in-context learning 推到极致：query embedding 通过 adapter 转成 TabPFN 看得懂的特征，TabPFN 直接根据 history 推断 posterior。

---

## 4. Shortcut Learning 的信息论分析（最理论的部分）

这一节是 paper 的理论 core，作者是 NJU 的 Han-Jia Ye（叶涵佳，[Google Scholar](https://scholar.google.com/citations?user=...)），他之前做过 N-BOT、GNN 等。

### 4.1 MI 分解

把 Y（label, reward）和 $(X, \mathcal{C})$（query + context）之间的 mutual information 拆开：

$$I(Y; X, \mathcal{C}) = \underbrace{I(Y; \mathcal{C})}_{\text{Context Shortcut}} + \underbrace{I(Y; X \mid \mathcal{C})}_{\text{Causal Reasoning}} \tag{6}$$

- $I(Y; \mathcal{C})$：纯 context 给的信息增益——比如一个 70B 模型 context 里 70% 都做对，模型可以不看 query $X$，直接预测 0.7
- $I(Y; X \mid \mathcal{C})$：给定 context 后，query 还能带来的信息——这才是 value model 真正想学的

### 4.2 Shortcut 必然存在

Theorem 4.1 说：如果 $\text{Var}[\mu(\mathcal{C})] > 0$（即不同 policy 的 capability prior 不同），那么 $I(Y; \mathcal{C}) > 0$，模型仅靠 fit $\mu(\mathcal{C})$ 就能严格降低 cross-entropy loss。

证明很简洁，关键在 Jensen 不等式 + binary entropy $\mathcal{H}_b(p)$ 严格凹：

$$\mathbb{E}_\mathcal{C}[\mathcal{H}_b(\mu(\mathcal{C}))] < \mathcal{H}_b(\mathbb{E}_\mathcal{C}[\mu(\mathcal{C})]) = \mathcal{H}_b(0.5) = H(Y) \tag{17}$$

直觉：全局 label balance $P(Y=1)=0.5$ 给你最大熵 $H(Y)=1$ bit 的"基线难度"。但只要不同 context 的 $\mu(\mathcal{C})$ 有方差（不同 policy 能力不同），单看 context 就能确定性地降低 entropy。比如对 strong policy 的 context，$\mu=0.8$，loss 只有 $\mathcal{H}_b(0.8) \approx 0.72$ bits，已经低于 1 bit。Gradient descent 偏好简单函数，于是模型就 collapse 到 context shortcut 上。

### 4.3 Pairwise Ranking 解耦

引入 logit score $s(x, \mathcal{C})$，最终概率 $V_0(x, \mathcal{C}) = \sigma(s(x, \mathcal{C}))$。从同一 context $\mathcal{C}$ 中抽一对 $(x_i, x_j)$ 满足 $y_i \succ y_j$，用 Bradley-Terry 损失：

$$\mathcal{L}_{\text{rank}} = -\mathbb{E}_{\mathcal{C} \sim \mathcal{D}} [\log \sigma(s(x_i, \mathcal{C}) - s(x_j, \mathcal{C}))] \tag{7}$$

Theorem 4.2 的核心：如果给 scoring function 加任意 context-dependent bias $\tilde{s}(x, \mathcal{C}) = s(x, \mathcal{C}) + b(\mathcal{C})$，那么：

$$\nabla_\phi \mathcal{L}_{\text{rank}}(\tilde{s}) = \nabla_\phi \mathcal{L}_{\text{rank}}(s)$$

证明就是 line 19-22：logit 差 $\Delta \tilde{s}_{ij} = \tilde{s}(x_i, \mathcal{C}) - \tilde{s}(x_j, \mathcal{C}) = s(x_i, \mathcal{C}) - s(x_j, \mathcal{C}) = \Delta s_{ij}$，bias 项 $b(\mathcal{C})$ 在减法中消掉了。

**这个 invariant 非常关键**：ranking loss 对 context-prior shift 不变。模型不能再用"猜 policy 强弱"的捷径，必须真正去比较 $x_i$ 和 $x_j$ 的相对难度。

这跟 DPO (Rafailov 2024, [arxiv](https://arxiv.org/abs/2305.18290)) 用 Bradley-Terry 对 preference 建模非常类似——DPO 通过 log-ratio 差把 reward model 解耦到 policy 上，这里通过 logit 差把 value model 解耦到 context 上。同样是 shift-invariance 的妙用。

### 4.4 Composite Loss

但下游任务（routing）需要 calibrated probability，纯 ranking loss 只给相对序。所以最终：

$$\mathcal{L} = \alpha \mathcal{L}_{\text{rank}}(s) + (1-\alpha) \mathcal{L}_{\text{CE}}(V_0) \tag{8}$$

$\alpha=0.25$ 在 paper 里 empirical 调出来。

---

## 5. Residual Orthogonality 诊断框架（Appendix B）

光说 debiased 不够，要有诊断工具。作者定义两个统计量：

**Context Prior**: $\mu(\mathcal{C}) = \frac{1}{N_\mathcal{C}} \sum_j y_j^{(\mathcal{C})}$，policy 的平均成功率
**Query Difficulty**: $D_x = \frac{1}{M_x} \sum_k y_k^{(x)}$，query 跨 policy 的平均成功率

两个残差（用 Spearman 相关）：
- $\text{Residual}_\mathcal{C} = \rho(\hat{y} - y, \mu(\mathcal{C}))$：error 与 context prior 的相关性
- $\text{Residual}_x = \rho(\hat{y} - y, D_x)$：error 与 query difficulty 的相关性

理想 value model $V^*(x, \mathcal{C}) = \Phi(\mu(\mathcal{C}), D_x) + \Delta(x, \mathcal{C})$，前一项是可分解的 prior，后一项是 interaction。两个 residual 都应趋于 0。

Figure 6 显示：fine-tune TabPFN head 时 residual 不收敛（shortcut 没解掉），用 proposed $V_0$ 时两个 residual 都趋于 0，证明 shortcut 被消除。

这个诊断思路让我想起 [Clever Hans effect](https://en.wikipedia.org/wiki/Clever_Hans) 在 NLP 中的分析，以及 Adebayo et al. 的 sanity checks for interpretability ([arxiv](https://arxiv.org/abs/1810.03292))——本质是检查 model 的 error 是否落在某个 known shortcut 上。

---

## 6. Budget Allocation：训练时的资源调度

GRPO 给每个 prompt 固定 rollout budget $B$ 太浪费：太简单的题全对（advantage=0），太难的题全错（advantage=0）。作者把 budget allocation 写成约束优化：

$$\max_{\{B_i\}} \sum_i \text{Utility}(B_i, p_i) \quad \text{s.t.} \quad \sum_i B_i \leq B_{\text{total}} \tag{9}$$

关键在于定义 utility。Appendix C 给了完整推导：

### 6.1 Gradient norm bound

GRPO 中 gradient norm 上界：
$$||\nabla_\theta J(\theta)|| \leq \gamma(s) \cdot \frac{1}{G} \sum_{i=1}^G |A_i| \tag{29}$$

$\gamma(s)$ 是 log-likelihood 的 Lipschitz constant，$A_i$ 是 advantage。最大化 $\sum |A_i|$ 就是最大化 gradient signal upper bound。

### 6.2 Closed-form utility 推导

Binary reward 下，group size $B$ 中有 $k$ 个 success。GRPO advantage：

$$A_{\text{pos}}(k) = \sqrt{\frac{B-k}{k}}, \quad A_{\text{neg}}(k) = -\sqrt{\frac{k}{B-k}}$$

Signal strength:
$$S(k) = \sum |A_i| = 2\sqrt{k(B-k)}$$

直接算 $\mathbb{E}[S] = \sum_{k=1}^{B-1} P(k) \cdot 2\sqrt{k(B-k)}$（line 41）涉及 fractional moment，无 closed form。作者引入 scaling factor $\lambda(k) = \frac{1}{2}A_{\text{pos}}(k)$，得到 proxy：

$$S_{\text{proxy}}(k) = S(k) \cdot \lambda(k) = B - k$$

这相当于"期望的错误样本数"（前提是 group 内部 variance 非零）。

最终 closed form（line 45-47）：

$$\text{Utility}(B, p) = B(1-p)[1 - (1-p)^{B-1}] \tag{10}$$

变量含义：
- $B$：分配给这个 prompt 的 rollout 数（budget）
- $p$：policy 在该 prompt 上的成功率（由 $V_0$ 预测）
- $(1-p)$：失败概率
- $(1-p)^{B-1}$：剩下 $B-1$ 个 rollout 全部失败的概率
- $1 - (1-p)^{B-1}$：至少有一个其他 rollout 成功的概率
- $B(1-p)$：期望失败次数

直观：utility 高意味着既有失败样本可学（$(1-p)$ 不能太小），又有成功样本提供 baseline（不能 $(1-p)^{B-1}$ 过大，即不能全失败）。这是个非常精巧的公式，捕捉了 GRPO 学习信号的本质。

求解用 greedy：每步把 budget 分配给边际 utility 最高的 prompt。Budget 范围 clip 在 $[2, 128]$。

**实验**：在 OlympiadBench 上 $V_0$-guided allocation 比 GRPO baseline 高 2%（Table 11，OlympiadBench 从 54.53% 提到 56.34%；AIME 2024 从 41.04% 提到 50.21%——这是相当大的提升）。

---

## 7. Inference Routing：部署时的 cost-performance trade-off

Model fleet 11 个 LLM，0.6B 到 32B，跨 12 个 benchmark 评测。每个 model 当成独立 policy $\pi$，构造 capability context。引入 cost-aware label：

$$r^\beta = \beta r + (1-\beta)(1-\tilde{c})$$

$\tilde{c} \in [0,1]$ 是归一化 cost（基于 params 和 token usage），$\beta$ 控制 cost-performance 偏好。

Context 构造成 $\mathcal{C}_\pi^\beta = \{(x_j, \text{Score}_{\beta,j})\}_{j=1}^N$。Routing 决策：

$$\pi^* = \arg\max_{\pi \in \Pi} V_0(x, \mathcal{C}_\pi^\beta) \tag{11}$$

通过 sweep $\beta$ 画 Pareto frontier（Figure 5b），比 EmbedLLM ([ICLR 2025](https://openreview.net/forum?id=EmbedLLM))、Model-SAT 都好。

**亮点**：$V_0$ 对新加入的 model 或 pricing 变化是 zero-shot 的——只需更新 context，不需重训。这是 in-context value estimation 的最大红利。

---

## 8. 实验结果深度分析

### 8.1 主结果 Table 1

三个架构（1.5B、4B、7B）上对比 5 个 baseline：

| Method | 1.5B Intra-AUC | 4B Intra-AUC | 7B Intra-AUC |
|---|---|---|---|
| Reward Model (Qwen2.5-Math-RM-72B) | 0.539 | 0.629 | 0.693 |
| kNN-Contextual | 0.818 | 0.861 | 0.754 |
| Vanilla Value Model | 0.840 | 0.898 | 0.830 |
| Step-wise Retrain | 0.757 | 0.710 | 0.701 |
| **V₀ (Ours)** | **0.913** | **0.904** | **0.879** |

$V_0$ 在所有架构上都赢，包括跟"每步从头重训"的 oracle 都赢——说明 in-context capability recognition 比 parameter fitting 更适合这个 non-stationary tracking 任务。

Reward Model 表现差很容易理解：它只估 $P(Y \mid X)$，把 $I(Y; \mathcal{C} \mid X)$ 信息丢了（Appendix D.1）。

kNN 表现中等：它是 $I(Y; X \mid \mathcal{C})$ 的 non-parametric 估计（Appendix D.2），但高维 semantic space 上近邻不一定语义近，且无法学 meta-pattern。

### 8.2 Strict Generalization (Table 2)

把 query ID 严格 disjoint 分到 train/test。Vanilla VM collapse 到 AUC=0.56（基本随机），$V_0$ 还有 0.71。这证明 $V_0$ 学到的是 meta-knowledge，而非 prompt memorization。

### 8.3 Ablation Table 4-6

- **Connector 设计**：Residual Dynamic Query 最好（AUC 0.705），其他 MLP/Cascaded/MultiScale 容易 overfit
- **Loss**：纯 CE 给 0.686，纯 ranking 给 0.578（会过拟合），combined 0.705——证明 composite loss 是必须的
- **Tuning**：freeze TabPFN head + tune connector 最好；jointly tune TabPFN 反而过拟合

### 8.4 Context Length Scaling (Table 9)

| N | Intra-AUC |
|---|---|
| 32 | 0.538 |
| 128 | 0.589 |
| 256 | 0.705 |
| 512 | 0.733 |

$N \leq 128$ 时 AUC 卡在 0.5 附近——sample size 不足以 characterize policy capability。$N=256$ 后才真正能区分。这跟 statistical learning theory 一致：高维 capability 需要足够 sample density。

---

## 9. Case Study: 为什么 state-only value 不够（Appendix E）

作者给了个非常 striking 的例子。复数题：

> Given a complex number $z$ such that $z - \frac{4}{z}$ is purely imaginary, find integer approximation of min $|z - 1 - i|$.

1.5B 和 4B 模型在 Phase 1（推导 $z=x+yi$、纯虚条件、得到 locus $x=0$ 或 $x^2+y^2=4$）生成的 text 几乎 identical。如果用 state-only value $V(s)$，两个 trajectory 在 Phase 1 是同 state，应该同 value。

但 Phase 2 时，4B 用几何方法（$|1+i|=\sqrt{2}<2$，点在圆内，最小距离 $2-\sqrt{2}$），1.5B 切换到三角换元 $x=2\cos\theta, y=2\sin\theta$ 然后算错。

所以 $s_{\pi_\text{weak}} = s_{\pi_\text{strong}}$ 但 $V^{\pi_\text{weak}}(s) \approx 0$，$V^{\pi_\text{strong}}(s) \approx 1$。

这是对 DVPO (Huang 2025, [arxiv](https://arxiv.org/abs/2502.16944)) 等 state-only 方法的有力反驳：trajectory prefix 不能唯一识别 policy capability。

---

## 10. 联想到的相关工作

1. **TabPFN** ([Nature 2025](https://www.nature.com/articles/s41586-024-08306-4), [ICLR 2023](https://arxiv.org/abs/2207.01848))：tabular foundation model，pre-train 在 millions synthetic tabular datasets 上做 in-context classification，单 forward pass Bayesian inference。V₀ 把它当作 probabilistic reasoning head。

2. **Müller et al. ICLR 2022 "Transformers can do Bayesian inference"** ([arxiv](https://arxiv.org/abs/2112.10510))：理论证明 Transformer 可以 implicit 实现 GP posterior。这是 TabPFN 的理论基础。

3. **Perceiver / Perceiver IO** (Jaegle 2021, [arxiv](https://arxiv.org/abs/2103.03206))：用 learnable latents 通过 cross-attention 把高维输入压成 fixed size。V₀ 的 Residual Query Adapter 是这一思想的 tabular 变体。

4. **BLIP-2 Q-Former** (Li 2023, [arxiv](https://arxiv.org/abs/2301.12597))：用一组 learnable queries 跨 attention 把 image features 转成 LLM 看得懂的 tokens。设计哲学一致。

5. **DPO** (Rafailov 2024, [arxiv](https://arxiv.org/abs/2305.18290))：用 Bradley-Terry + log-ratio 把 reward model 解耦到 policy 上。V₀ 用 Bradley-Terry + logit-diff 把 value 解耦到 context 上。同样是 shift-invariance 妙用。

6. **GRPO** (DeepSeek, [DeepSeekMath](https://arxiv.org/abs/2402.03300))：把 value model 砍掉用 group average，但 variance 高。V₀ 是 GRPO 的互补：用 in-context value 重新补回 variance reduction 但不需要 coupled training。

7. **EmbedLLM** (Zhuang 2025, [ICLR](https://openreview.net/forum?id=...))：collaborative filtering 角度做 LLM routing，从 historical log 学 compact embedding。

8. **Knapsack RL** (Li 2025, [arxiv](https://arxiv.org/abs/2509.25849))：把 budget allocation 当 knapsack 问题。

9. **CURES** (Zeng 2025, [arxiv](https://arxiv.org/abs/2510.01037))：curriculum learning for reasoning LLMs，理论上证明 optimal rollout 数正比于 gradient variance。

10. **DOTS** (Sun 2025, [arxiv](https://arxiv.org/abs/2506.05316))：用 reference set 预测 adaptive difficulty，优先 pass@0.5 附近的样本。

---

## 11. 我的几个思考

### 11.1 为什么不直接用 GPT-style LLM 做 in-context value prediction?

Paper 第 4.2 节间接回答：standard LLM 在 numerical estimation 上不可靠，而 TabPFN 在 Bayesian reasoning 上 pre-trained，相当于把"概率推断"这部分 offload 到一个专门训练的模块上。如果让 LLM 自己输出 "0.73" 这种概率，它的 calibration 通常很差（[Kadavath 2022](https://arxiv.org/abs/2207.05221), [Lin 2022](https://arxiv.org/abs/2205.14334) 等)。V₀ 用 hybrid 架构把"语义理解"和"统计推断"分工。

### 11.2 State Zero 限制

V₀ 只能预测 initial prompt 的 outcome reward，不能做 process supervision（token-level value）。这是作者在 Conclusion 里点明的 future work。当前 Process Reward Model (PRM, Lightman 2023 [arxiv](https://arxiv.org/abs/2305.20050)) 仍然需要 coupled training。如果能把 in-context capability 扩展到 token-level，将是很大突破。

### 11.3 在 long-horizon RL 中

Paper 用的是 GRPO 上的 RLVR（verifiable reward, single-turn reasoning），没在 agentic / multi-turn / long-horizon setting 上验证。如果 trajectory 长，state space 大，context 的表征需要重新设计。

### 11.4 Capability context 的构造

实践中 $\mathcal{C}_\pi$ 怎么 sample 也是 art：作者用 256 个 pairs，preserve natural distribution（imbalanced）。但实际部署时，应该有 active selection——选 informative history，跟 Bayesian active learning 类似。这点 paper 没展开。

### 11.5 跟 Anthropic Constitutional AI / RLAIF 的关系

如果 reward 本身来自另一个 model（RLAIF, [Bai 2022](https://arxiv.org/abs/2212.08073)），$V_0$ 是不是也能用 in-context 形式来 estimate AI-feedback reward? Concept 上 yes，但 verifiable rewards 是 0/1，AI feedback 是 continuous score，需要扩展。

### 11.6 跟 Inference-time scaling 的天然耦合

V₀ 在 inference routing 上的应用，本质是把 inference-time compute scaling ([Snell 2024](https://arxiv.org/abs/2408.03314)) 从 "best-of-N" 扩展到 "best-of-fleet"——根据 query 难度动态选模型。这跟 OpenAI o1 / DeepSeek R1 的 test-time compute 思路是一致的，但 V₀ 把决策点放在 model selection 而非 token sampling。

---

## 12. 总结

V₀ 这篇 paper 有几个真正 novel 的点：

1. **Reframe**：把 value estimation 从 parameter fitting 转 in-context learning，policy 作为 explicit context 输入
2. **Architecture**：Semantic Backbone + Residual Query Adapter + TabPFN Bayesian head 的三段式设计，把语义和统计解耦
3. **理论**：mutual information 分解 + pairwise ranking shift-invariance 的 debiasing，干净 elegant
4. **应用**：Budget allocation 有 closed-form utility；inference routing 有 Pareto frontier

但我觉得最 deep 的贡献是把 LLM 训练的"value model"问题转化为一个**meta-learning problem**：不再针对单个 policy 学一个 value function，而是学一个"如何从 history 推断 capability"的 meta-function。这跟 GPT-3 在 NLP 上展示的 in-context learning 是同一类范式转变，只是迁移到了 RL post-training 上。

未来如果扩展到 token-level value（PRM 形式），并且 context 用 active learning 而非 random sampling 选，将是非常 powerful 的基础设施。

---

**Reference Links**:
- Paper project page: https://now-join-us.github.io/V0
- TabPFN v2.5: https://arxiv.org/abs/2511.08667
- TabPFN Nature: https://www.nature.com/articles/s41586-024-08306-4
- Müller Bayesian Transformer: https://arxiv.org/abs/2112.10510
- DeepSeekMath GRPO: https://arxiv.org/abs/2402.03300
- DPO: https://arxiv.org/abs/2305.18290
- Perceiver IO: https://arxiv.org/abs/2103.03206
- BLIP-2 Q-Former: https://arxiv.org/abs/2301.12597
- EmbedLLM: https://openreview.net/pdf?id=tabpfn
- Knapsack RL: https://arxiv.org/abs/2509.25849
- DOTS (Sun 2025): https://arxiv.org/abs/2506.05316
- CURES: https://arxiv.org/abs/2510.01037
- PRM (Lightman): https://arxiv.org/abs/2305.20050
- DAPO: https://arxiv.org/abs/2503.14476
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Self-consistency / inference scaling: https://arxiv.org/abs/2408.03314
