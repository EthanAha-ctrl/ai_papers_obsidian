---
source_pdf: F-GRPO Don’t Let Your Policy Learn the Obvious and Forget the Rare.pdf
paper_sha256: feaa4a6d2638e9516bff4dd2a2010e2d5622e3558bafe2f4782f57ab8b215057
processed_at: '2026-08-04T06:25:17-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej，这篇 paper 我读透了，用最直白的人话给你拆解一下。这文章抓的核心痛点特别精准，而且解法极其轻巧。

一句话概括：**大家都在用的 GRPO（N=8），正好卡在一个“学得快但忘得快”的尴尬区，把模型本来会的罕见正确解给洗掉了。这篇文章借用了 Focal Loss 的思路，给简单 prompt 降权，强行保住了模型的 solution diversity。**

下面我分模块把直觉和数学细节给你讲清楚。

### 1. 为什么会发生 RLVR Sharpening？

现在做 RLVR（比如跑 GRPO 训 Qwen），大家普遍观察到一个现象：**pass@1 涨了，但 pass@256 跌了。**
直觉上，这代表模型其实没学到什么新知识，它只是在 base model 原本就会的解里面，挑了几个最常见的“死记硬背”下来，把那些原本概率很低但偶尔能蒙对的“罕见好解”的 probability mass 给清零了。这就是所谓的 distribution sharpening。

这篇文章把锅精准砸到了 **group size N** 上。

### 2. 核心机制：Group Size N 的非单调陷阱

为什么 N 会导致 sharpening？作者推导了一个极漂亮的闭式公式（Lemma 3.1），把 rollout 采样分成了三个区域：
1. **Rare-correct** ($\tau$)：罕见但正确的解。
2. **Common-correct** ($\mu_{pos} - \tau$)：常见且正确的解。
3. **Incorrect** ($1 - \mu_{pos}$)：错解。

GRPO 的 advantage 是 $\hat{A}_i = (R_i - \bar{R}) / \sigma_R$。这里有个致命点：**如果 batch 里 N 个 rollout 全对或全错，方差 $\sigma_R = 0$，梯度直接变成 0。**

作者算了一个“Tail-miss probability” $\Pr(\mathcal{B}_\tau)$，也就是**“batch 里既有对又有错（产生梯度），但恰好没采到那个罕见好解”**的概率：

$$ \Pr(\mathcal{B}_\tau) = (1-\tau)^N - (\mu_{pos}-\tau)^N - (1-\mu_{pos})^N $$

**变量啥意思：**
* $N$: group size
* $\tau$: rare-correct 的总概率 mass
* $\mu_{pos}$: 所有正确解的总概率 mass
* $(1-\tau)^N$: N 次都没采到 rare-correct 的概率
* 减掉的后两项: 减去 batch 全对（无梯度）和全错（无梯度）的情况。

这公式画出来是个**非单调的钟形曲线**，直接把文献里打架的结论全 reconcile 了：

* **N=2（极小）**：大部分时候全对或全错，$\sigma_R = 0$。模型基本没动，自然保留了 base model 的 diversity。Wu et al. 说的“N=2 就够了”就是这原因，但代价是 pass@1 根本学不动。
* **N=8（中间）**：开始有 mixed reward 了，梯度来了。但是 $(1-\tau)^N$ 还没掉下去！也就是说，**模型开始学了，但学的都是常见好解，没采到罕见好解**。这是 sharpening 最严重的灾区。大家为了省算力都爱用 N=8，正好踩坑。
* **N=32+（极大）**：$(1-\tau)^N \to 0$，罕见好解几乎必被采到。Hu et al. 倡导的“scale N”就是这个道理，但 4 倍算力太贵了。

### 3. 概率质量流失的物理机制

光漏采还不致命，致命的是漏采之后 mass 怎么流的。Proposition 3.2 用 categorical framework 拆解了 one-step update：

$$ \Delta z_i = \frac{\eta}{N} p_i (R_i - S_R) $$

**变量啥意思：**
* $\Delta z_i$: logit 更新量
* $\eta$: learning rate
* $p_i$: 当前 action 的概率
* $R_i$: reward
* $S_R$: batch baseline $S_R = R_c P_{pos} + R_w P_{neg}$

因为 softmax 归一化，**未被采样的 action（$R_i = 0$）会被 baseline $S_R$ 硬拉下来**。当 batch 里正确居多（$S_R > 0$）时，所有未被采样的好解，logit 全部被无脑往下推！

这就是核心反直觉的地方：**RLVR 确实在增加总的 correct mass，但它在内部做的是抢劫——把没被采到的罕见好解的 mass，抢过来补给被采到的常见好解。** 

### 4. F-GRPO 解法：Focal Weight

既然知道是“成功率高”的 prompt 在搞归一化抢劫，作者的解法极度简单。借鉴你肯定特别熟的 Focal Loss（$(1-p_t)^\gamma$），给每个 prompt 算一个权重 $g(x)$：

$$ g(x) := (1 - \hat{\mu}_{pos}(x))^\gamma $$

**变量啥意思：**
* $\hat{\mu}_{pos}(x)$: 实测成功率 $X/N$，比如 8 个 rollout 对了 7 个，就是 $7/8 = 0.875$。
* $\gamma$: 控制压制力度的超参（一般取 0.5 或 1.0）。

把这权重乘到 advantage 上：
$$ \hat{A}_i^{F\text{-}GRPO} := g(x) \cdot \hat{A}_i^{GRPO} $$

**直觉上**：如果一个 prompt 模型已经做得很顺了（比如对了 7/8），那它的 baseline $S_R$ 必然很大，抢劫未采样好解的力度极强。Focal weight 直接把这个 prompt 的梯度打个七折（$g = (1 - 0.875)^{0.5} \approx 0.35$），把梯度预算让给那些“8 个里只对 1 个”的 hard prompt。这跟做 Object Detection 时 Focal Loss 压低 easy negative 的逻辑完全同构。实现上就是一行代码乘个标量，没有任何额外网络。

### 5. 实验数据看门道

看 Qwen2.5-7B 上的数据（Table 2）最说明问题：

| Method | Avg. Math pass@1 / pass@256 | Avg. OOD pass@1 / pass@256 | $\Delta\text{NLL}_{rare}$ |
| :--- | :--- | :--- | :--- |
| GRPO N=2 | 36.2 / **75.0** | 18.0 / **67.3** | 0.19 |
| GRPO N=8 | 37.3 / 64.1 | 17.1 / 55.9 | **0.68** |
| GRPO N=32 | **39.2** / 70.1 | 17.7 / 61.7 | 0.52 |
| **F-GRPO N=8** | 38.6 / 70.3 | **19.2** / 63.3 | 0.46 |

解读这张表的逻辑：
1. **N=2 vs N=8**：N=8 的 pass@1 涨了，但 pass@256 瀑跌 10 个点。N=8 就是 concentration zone 的 signature。
2. **N=8 vs N=32**：老老实实加钱把 N 调到 32，pass@256 救回了 70.1，但算力翻了 4 倍。
3. **F-GRPO (N=8)**：用 N=8 的算力，pass@256 干到了 70.3，直接对打 N=32。而且 OOD pass@1 还更高（19.2）。
4. **$\Delta\text{NLL}_{rare}$ 是个很有意思的指标**：拿 base model 概率极低但确实是正确的 trajectories，看训完之后 NLL 变高多少。N=8 最惨（偏离 0.68），F-GRPO 救回了一半（0.46），说明它确实物理上保住了这些罕见解的 mass。

### 6. 我的几点深层联想

1. **Focal Loss 在 LLM RL 里的重生**：当年 Focal Loss 解决的是 dense object detection 里正负样本极度不均衡的问题。现在 LLM RLVR 遇到的是 prompt difficulty 不均衡。容易的 prompt 梯度信号太强，疯狂 sharpening；难的 prompt 梯度信号太弱。F-GRPO 本质上是在做 implicit curriculum learning，通过 reweighting 强行把梯度 distribution 拉平。
2. **CISPO 和 DAPO 也能叠**：DAPO 改的是 clipping bound（鼓励低概率 action 往上探），CISPO 改的是 importance sampling 的 clip 位置。这些都在 token-level 动刀子。F-GRPO 在 prompt-level 动刀子，完全 orthogonal，所以实验里 F-DAPO, F-CISPO 全都涨。
3. **$\hat{\mu}_{pos}$ 的 Noise**：N=8 的时候，成功率只取离散值 $\{0, 1/8, ..., 1\}$，方差极大。Paper 没讨论如果刚好这 8 次运气好/不好导致 $\hat{\mu}$ 估偏了会怎样。不过 $\gamma=0.5$ 这种软指数函数本身对 noise 有一定鲁棒性。
4. **未来可能的方向**：这里的 $g(x)$ 用的是单次 batch 的成功率。如果用 running average 的 EMA 成功率来代替，可能能进一步降噪，把那些“本来很简单但这次恰好全做错”的 prompt 滤掉。

### 总结
这 paper 的价值在于：**它指出 LLM RLVR 的 sharpening 不是 RL 算法的原罪，而是 finite sampling 在特定 N 下的数学必然。** 给 advantage 乘个 Focal weight，用几乎为零的代码成本，拿回了 4 倍算力才能换来的 diversity。

**References:**
* [F-GRPO 原文 (arxiv)](https://arxiv.org/abs/2510.00000) *(注：paper 未提供 arXiv ID，此为占位，实际请查阅作者后续公开版本)*
* [Focal Loss for Dense Object Detection (Lin et al., ICCV 2017)](https://arxiv.org/abs/1708.02002)
* [DeepSeekMath / GRPO (Shao et al., 2024)](https://arxiv.org/abs/2402.03300)
* [DAPO (Yu et al., 2025)](https://arxiv.org/abs/2503.14476)
* [BroRL / Hu et al., 2025](https://arxiv.org/abs/2510.01180)
* [Does RL Really Incentivize Reasoning Capacity (Yue et al., 2025)](https://arxiv.org/abs/2504.13837)
* [verl RL framework](https://arxiv.org/abs/2409.19256)

---

# F-GRPO 深度解析：Group Size 的非单调性与 Focal Weighting

Andrej，这篇 paper 我读了几遍，挺有意思的。核心贡献其实就两件事：(1) 用一个 closed-form tail-miss probability 把 GRPO 里 group size N 的角色讲清楚了——它**不是单调的**，而是有一个 "concentration zone" 在中间 N 值；(2) 借 Focal loss 的 functional form 给 advantage 加一个 scalar weight，极其轻量。我尽量把直觉和数学细节都讲清楚。

---

## 1. 问题的本质：为什么 RLVR 会 sharpen distribution

先说 background intuition。RLVR 训练后大家观察到一个普遍现象：pass@1 涨，pass@k (k 大) 跌。这意味着 policy 不是"学到了新能力"，而是**把 probability mass 集中到 base model 已经能采到的几个 common solution 上**，把 rare-correct solutions 的 mass 压扁了。

这个 paper 的切入点很 specific：group-relative methods (GRPO/DAPO/CISPO) 的 advantage 是从 N 个 rollout 里估的，group size N 直接决定了"哪些 trajectory 会进入 gradient"。所以 sharpening 不是 RL 本身的 evil，是 **finite group sampling 的一个 failure mode**。

---

## 2. Lemma 3.1：Tail-miss probability 的 closed form

这是全文最漂亮的 result。

### Setup

对一个 prompt $x$，policy 的 rollout 空间 $\Omega_x$ 分三块：

| Region | Probability mass | 含义 |
|---|---|---|
| $\mathcal{C}_{rare}(x)$ | $\tau(x)$ | rare-correct solutions (难采到的正确解) |
| $\mathcal{C}(x) \setminus \mathcal{C}_{rare}(x)$ | $\mu_{pos}(x) - \tau(x)$ | common-correct solutions |
| $\Omega_x \setminus \mathcal{C}(x)$ | $1 - \mu_{pos}(x)$ | incorrect |

其中：
- $\mu_{pos}(x) := \Pr_{o \sim \pi_\theta}[o \in \mathcal{C}(x)]$ 是 success probability
- $\tau(x) := \Pr[o \in \mathcal{C}_{rare}(x)]$ 是 rare-correct mass
- 比值 $\rho(x) := \tau(x)/\mu_{pos}(x)$ 衡量 "rare fraction"

### 关键事件

采样 N 个 i.i.d. rollouts，令 $X$ = 正确 rollout 数。GRPO 的 advantage 在 $\sigma_R = 0$ (全对或全错) 时为零——也就是 **没有 learning signal**。所以学习只发生在 active event:

$$\mathcal{A}_N := \{0 < X < N\}$$

我们关心的是 **active 但没采到 rare-correct** 这个事件 $\mathcal{B}_\tau := \mathcal{A}_N \cap \{\sum_i Y_i = 0\}$，其中 $Y_i = \mathbb{I}[o_i \in \mathcal{C}_{rare}]$。

### Lemma 3.1 的推导 (Appendix A)

把 rollout 空间三分，用 inclusion-exclusion：

- "没采到 rare-correct" 的概率 = $(1-\tau)^N$（所有 N 个 rollout 都不在 rare region）
- 在此条件下，group inactive 有两种 disjoint 情况：
  - 全部落在 common-correct：$(\mu_{pos}-\tau)^N$
  - 全部 incorrect：$(1-\mu_{pos})^N$

所以：

$$\boxed{\Pr(\mathcal{B}_\tau) = (1-\tau)^N - (\mu_{pos}-\tau)^N - (1-\mu_{pos})^N}$$

**变量含义**：
- 上标 $N$：group size
- $\tau$、$\mu_{pos}$：如上定义
- $(1-\tau)^N$：coverage factor，随 N 增大单调递减
- 后两项：减掉 inactive 情况

### 三个 regime 的直觉

这个公式揭示了 N 的 **非单调性**，三个 regime：

1. **小 N (e.g., N=2)**：$\Pr(\mathcal{A}_N)$ 接近 0，大多数 group 都是 homogeneous (全对或全错)，zero gradient。Policy 改得很慢 → **diversity preserved through inactivity**。这对应 Wu et al. 2025b 的 "N=2 就够了"的发现——但他们观察的是 pass@1，没看到 sharpening 是因为根本没学。

2. **中间 N (e.g., N=8~32)**：$\Pr(\mathcal{A}_N)$ 涨上来了 (active)，但 $(1-\tau)^N$ 还没掉下去 (coverage 不好)。这是 **sharpening 最严重的 regime**，$\Pr(\mathcal{B}_\tau)$ 达到 peak。实际工程里因为 compute 限制，N=8 几乎是 default，正好踩坑。

3. **大 N**：$(1-\tau)^N \to 0$，rare-correct 几乎一定被采到，unsampled mass 也小。对应 Hu et al. 2025 的 "scale N for coverage"。

这就 reconcile 了文献里看似矛盾的结论：**small N 和 large N 都 preserve diversity，但通过完全不同的 mechanism** (inactivity vs coverage)。中间 N 是 worst。

看 Figure 2：固定 $\mu_{pos}$、变 $\rho$，peak 位置和高度都变。$\rho$ 小 (rare fraction 小) → peak 往右移、往上抬。直觉是 rare region 越小，需要更大 N 才能 cover 到。

---

## 3. Proposition 3.2：unsampled-correct mass 为什么会缩

Lemma 3.1 告诉我们"什么时候 rare-correct 会被 miss"，但没讲"miss 之后 mass 怎么流"。这一节用 Hu et al. 的 categorical framework 把机制讲透。

### Categorical setup

softmax policy $p = \text{softmax}(z)$ over finite action space $\mathcal{A}$，分成 correct $\mathcal{P}$ 和 incorrect $\mathcal{N}$。采样后：

- $A \subseteq \mathcal{P}$: sampled correct actions
- $B \subseteq \mathcal{N}$: sampled incorrect actions  
- $U$: unsampled actions
- $P_{pos} = \sum_{i \in A} p_i$, $P_{neg} = \sum_{i \in B} p_i$
- $A_2 = \sum_{i \in A} p_i^2$, $B_2 = \sum_{i \in B} p_i^2$ (concentration measures / second moments)
- $U_{pos,2} = \sum_{i \in U \cap \mathcal{P}} p_i^2$, $U_{neg,2}$ 类似
- **batch baseline**: $S_R := R_c P_{pos} + R_w P_{neg}$

注意：unsampled actions $i \in U$ 的 reward 设为 0（它们不在 batch 里），但**它们仍然受 update 影响**，因为 softmax 是 coupled 的。

### One-step logit update (Eq. 7)

对 TRPO-style linear surrogate 做一阶展开（用 $\partial p_i / \partial z_j = p_i(\delta_{ij} - p_j)$）：

$$\Delta z_i = \frac{\eta}{N} p_i (R_i - S_R)$$

**变量**：
- $\eta$：learning rate
- $N$：group size (在分母里 → 大 N 稀释 update)
- $p_i$：action $i$ 的当前概率
- $R_i$：action $i$ 的 reward（sampled 时按 Eq. 1，unsampled 时为 0）
- $S_R$：batch baseline

**对 unsampled action $i \in U$**：因为 $R_i = 0$：

$$\Delta z_i = -\frac{\eta}{N} S_R p_i$$

这是关键！**unsampled actions 的 logit update 只取决于 baseline $S_R$ 的符号**。当 $S_R > 0$ (reward-positive batch，即 batch 里正确多于错误)，所有 unsampled logits 被往下推——包括 unsampled-correct actions。

### Proposition 3.2 的 $\Delta Q_{u,pos}$

定义 unsampled-correct mass:

$$Q_{u,pos} := \sum_{i \in U \cap \mathcal{P}} p_i = Q_{pos} - P_{pos}$$

用 subset-mass identity (Appendix C, Eq. 21)：

$$\Delta Q_S = \sum_{i \in S} p_i \Delta z_i - Q_S \sum_{j \in \mathcal{A}} p_j \Delta z_j$$

第一项是 direct effect（logit 直接变化），第二项是 **normalization coupling**（其他地方 mass 涨了，softmax 把 $S$ 里的 mass 抽走）。

代入 $S = U \cap \mathcal{P}$，整理得：

$$\boxed{\Delta Q_{u,pos} = \frac{\eta}{N} \left[ \underbrace{-S_R U_{pos,2}}_{\text{direct drift}} - Q_{u,pos} \underbrace{\left( (R_c - S_R) A_2 + (R_w - S_R) B_2 - S_R U_2 \right)}_{\text{normalization coupling}} \right]}$$

### 两项的物理意义

**Direct drift** $-S_R U_{pos,2}$：
- 当 $S_R > 0$：negative，把 unsampled-correct mass 往下推
- magnitude $\propto U_{pos,2}$：unsampled-correct 的 concentration（second moment 越大说明 mass 越集中）

**Normalization coupling** 拆三块：
- $(R_c - S_R) A_2 \geq 0$：sampled-correct actions 涨概率，通过 normalization 从 unsampled-correct 抽走 mass。这是主要"drain"
- $(R_w - S_R) B_2 \leq 0$：sampled-incorrect actions 掉概率，donate mass 给其他 actions（包括 unsampled-correct）。这个方向是**反向**的，救一下
- $-S_R U_2$：unsampled actions 整体掉概率

### 核心反直觉 insight

即使 $\Delta Q_{pos} > 0$ (Hu et al. 证明的，total correct mass 增加)，**$\Delta Q_{u,pos}$ 可以同时 $< 0$**。也就是说：

> RLVR 可以**增加** total correct mass，同时**减少** unsampled-correct mass——把 correct mass 集中到 sampled-correct subset 上。

这就是 sharpening 的机制。Hu et al. 说大 N 能让 $\Delta Q_{pos} \geq 0$，但没讲 unsampled-correct 内部的 redistribution。这 paper 补上了。

### 与 Lemma 3.1 的衔接

Lemma 3.1 说"什么时候 miss"，Prop 3.2 说"miss 之后 mass 怎么流"。中间 N regime 既频繁 miss rare-correct (Prop 3.1)，miss 之后又因为 $S_R > 0$ 把它们的 mass 推下去 (Prop 3.2)。两个机制叠加 → sharpening 在中间 N 最严重。

---

## 4. F-GRPO：从理论到方法

### 4.1 为什么用 $\hat{\mu}_{pos}$ 作 proxy

理论分析指出 $S_R > 0$ 是 concentration 的 driver。但 $S_R$ 依赖 $P_{pos}$、$P_{neg}$（distinct sampled mass），不可观测。需要一个 observable per-prompt signal。

**empirical success rate**:

$$\hat{\mu}_{pos}(x) := \frac{\bar{R}(x) - R_w}{R_c - R_w} = \frac{X}{N} \in [0, 1]$$

$X$ = batch 里正确 rollout 数。这是 $\mu_{pos}(x)$ 的无偏估计。

**关键 Lemma B.1 + Corollary B.2**：$\mathbb{E}[S_R | X = k]$ 是 $k$ 的 non-decreasing function。

证明思路（Appendix B）：condition on $X = k$，则 $k$ 个 correct rollouts i.i.d. 来自 restricted distribution $q_{pos}(o) = \pi(o)/\mu_{pos}$ over $\mathcal{C}$。对固定 $o \in \mathcal{C}$，"o 被 sample 到" 的概率是 $1 - (1-q_{pos}(o))^k$，是 $k$ 的 non-decreasing function。线性求和 → $\mathbb{E}[P_{pos} | X = k]$ non-decreasing in $k$。同理 $\mathbb{E}[P_{neg} | X = k]$ non-increasing。当 $R_w \leq 0$，$R_w \cdot \mathbb{E}[P_{neg}|X=k]$ 也是 non-decreasing → $S_R$ non-decreasing。

**直觉**：batch 里正确数越多，sampled-correct mass 越大，baseline $S_R$ 越正，concentration pressure 越强。所以 high $\hat{\mu}_{pos}$ 标记的就是 "这个 prompt 会让 unsampled-correct mass 被抽走" 的 regime。

### 4.2 Focal weight

借 Focal loss (Lin et al. 2017) 的形式——Focal loss 用 $(1-p_t)^\gamma$ down-weight well-classified examples，这里用 $(1-\hat{\mu}_{pos})^\gamma$ down-weight high-success prompts：

$$g(x) := (1 - \hat{\mu}_{pos}(x))^\gamma, \quad \gamma \geq 0$$

- $\gamma = 0$：$g \equiv 1$，恢复 standard GRPO
- $\gamma > 0$：$\hat{\mu}_{pos} \to 1$ 时 $g \to 0$，hard suppress easy prompts
- $\hat{\mu}_{pos} = 0$ (全错)：$g = 1$，full weight（但其实这种情况 $\hat{A}^{GRPO}$ 也很小，因为 $\sigma_R$ 小）

### 4.3 F-GRPO advantage

$$\hat{A}_i^{F\text{-}GRPO} := g(x) \cdot \hat{A}_i^{GRPO} = (1 - \hat{\mu}_{pos}(x))^\gamma \cdot \frac{R_i - \bar{R}}{\sigma_R + \epsilon}$$

**实现上**：每个 prompt 算一个 scalar $g(x) \in [0,1]$，乘到该 prompt 所有 rollout 的所有 token 的 advantage 上。**没有额外网络，唯一新 hyperparameter 是 $\gamma$**。

### 4.4 Figure 3 的直觉

横轴 $\mu_{pos}$，纵轴 scaled advantage magnitude。binary reward 下，GRPO advantage 的 magnitude 是有 analytic form 的。correct rollout 的 advantage $= (1-\hat\mu)/\sqrt{\hat\mu(1-\hat\mu)/N}$（粗略），incorrect rollout 的相反。Focal weight 把高 $\hat\mu$ 区的 magnitude 压下去，**把 gradient 贡献从 easy prompts 转移到 hard prompts**。这跟 curriculum learning 的思路相通，但 mechanism 不同：curriculum 调 prompt 顺序，Focal 调 prompt-level gradient scale。

### 4.5 为什么对 DAPO/CISPO 也 work

DAPO 改 clipping bound ($\epsilon_{high} > \epsilon_{low}$)，CISPO 改 importance weight clipping 位置（clip $r$ 而非 $r \cdot A$）。这些改的是 **token-level clipping mechanism**，而 concentration 现象源于 **group-relative advantage estimation 的 sampling dynamics**——是更上游的问题。所以 Focal weight 是 orthogonal 的，可以叠加。Table 1 验证了这点：F-DAPO、F-CISPO 都有提升。

---

## 5. Experiments 关键数据

### Table 2: Group size regime 的 empirical 验证 (Qwen2.5-7B)

| Method | Avg. Math pass@1/pass@256 | Avg. OOD pass@1/pass@256 | $\Delta\text{NLL}_{rare}$ |
|---|---|---|---|
| GRPO N=2 | 36.2 / **75.0** | 18.0 / **67.3** | 0.19 |
| GRPO N=8 | 37.3 / 64.1 | 17.1 / 55.9 | **0.68** |
| GRPO N=32 | **39.2** / 70.1 | 17.7 / 61.7 | 0.52 |
| **F-GRPO N=8** | 38.6 / 70.3 | **19.2** / 63.3 | 0.46 |

读这张表的方式：
- N=2 vs N=8：pass@1 涨 1.1，pass@256 跌 10.9 (math)。这就是 sharpening regime 的 signature
- N=8 vs N=32：增加 N 把 pass@256 救回来 (64.1 → 70.1)，但代价是 4× rollout compute
- **F-GRPO N=8 vs GRPO N=32**：pass@256 几乎打平 (70.3 vs 70.1 math, 63.3 vs 61.7 OOD)，用 1/4 compute。OOD pass@1 还更好 (19.2 vs 17.7)

$\Delta\text{NLL}_{rare}$ 是 Appendix F.2 的 proxy：取 base model 正确但 low-probability 的 trajectories (top 1% by base NLL)，算 trained model 对它们的 NLL。值越大 → 偏离 base distribution 越多 → rare-correct mass 被压越多。**N=8 最差 (0.68)，F-GRPO 救回一半 (0.46)**，符合理论。

### Table 1: 三模型 × 三方法 (N=8)

Qwen2.5-7B 上 Focal 的提升：
- GRPO → F-GRPO: math pass@256 +6.2 (64.1→70.3), OOD pass@256 +7.4 (55.9→63.3), OOD pass@1 +2.1
- DAPO → F-DAPO: math pass@256 +3.2, OOD pass@256 +5.2
- CISPO → F-CISPO: math pass@256 +3.6, OOD pass@256 +6.9

**7/9 个 model-method 组合里 OOD pass@1 也提升**（平均 +1.1）。这是 paper 强调的：preserving diversity 不牺牲 single-attempt accuracy，甚至因为更宽的 solution manifold 而 improve OOD generalization。

### Table 4: vs entropy bonus / KL penalty

- F-GRPO: math pass@1 **38.6** (best), OOD pass@256 **63.3** (best)
- GRPO + entropy bonus: math pass@1 37.8, OOD pass@256 59.9
- GRPO + KL penalty: math pass@256 72.0 (best, 但需 reference model in memory), OOD pass@256 60.0

F-GRPO 简单且强。KL penalty 在 math pass@256 上略胜，但 memory overhead 大。

### Categorical simulation (Figure 4)

128k actions, 10k correct。跟踪 $\mathcal{M}_{ret}(t)$ (retained positive mass, Appendix J Eq. 26):

$$\mathcal{M}_{ret}(t) = 1 - \frac{\sum_{a \in \mathcal{A}^+} \max(0, \pi_0(a) - \pi_t(a))}{\sum_{a \in \mathcal{A}^+} \pi_0(a)}$$

值接近 1 = diversity 保留，接近 0 = concentration。

三个 regime 在 simulation 里清楚呈现：
- (I) 小 N：$Q_{pos}$ 慢涨，$\mathcal{M}_{ret} \approx 1$
- (II) 中间 N (shaded zone)：$Q_{pos}$ 快涨，$\mathcal{M}_{ret}$ 崩
- (III) 大 N：两个都好

特别有意思的是 **$N = 131,072$ ($= 2^{17}$)** 整个 training 保持 $\mathcal{M}_{ret} \approx 1$。这对应 Lemma 3.1 的 prediction：在该 $\tau \approx 6.3 \times 10^{-5}$ 下，$\Pr(\mathcal{B}_\tau) < 10^{-3}$ 当 $N \geq 2^{17}$。理论预测和 simulation 吻合。$\gamma = 1$ (dashed) 在 concentration zone 救 $\mathcal{M}_{ret}$ 明显。

---

## 6. 直觉总结 & 与 related work 的关系

### 核心直觉

GRPO 的 advantage $\hat{A}_i = (R_i - \bar{R})/\sigma_R$ 是 group-relative 的。当 group 全对或全错，$\sigma_R = 0$，zero gradient。所以**学习只发生在 mixed-reward group**。但 mixed-reward group 又恰恰是那种"采到了 common-correct、没采到 rare-correct"的配置——因为 rare-correct 概率小，被采到的概率低。一旦没采到，rare-correct 的 mass 就被 $S_R > 0$ 的 direct drift 和 normalization coupling 双重往下推。

这是个 **catch-22**：要 learning signal 必须 mixed-reward，要 mixed-reward 在中间 N 就大概率 miss rare-correct。Focal weight 的解法是**保留 mixed-reward group 的 learning signal，但 scale down 那些高 success rate 的 prompt**（它们的 rare-correct 最容易被 miss 且 mass 被抽走最狠），把 gradient budget 让给 low-success 的 hard prompts。

### 与几个关键 related work 的差异

- **Yue et al. 2025 / Ni et al. 2025**: 说 RLVR 不引入新知识，只是 sharpen。这 paper 给出 **finite sampling 的具体 mechanism**——sharpening 不是 RL 本身的 evil，是 group size 选错了
- **Wu et al. 2025b "GRPO is secretly DPO"**: 说 N=2 就够。这 paper 不反对，但指出 N=2 preserve diversity 是通过 **inactivity**（大多数 group zero gradient），不是真学得好。如果想 pass@1 涨，必须增加 N，而增加 N 就进入 concentration zone
- **Hu et al. 2025 "BroRL"**: advocate 大 N for coverage。这 paper reconcile：大 N 确实 work，但 compute 太贵，Focal 是 cheap alternative
- **He et al. 2025 "Rewarding the unlikely"**: 改 trajectory-level reward，up-weight rare correct。F-GRPO 改的是 **prompt-level gradient scale**，不改 reward。mechanism 不同——他们针对 rank bias，这 paper 针对 $S_R > 0$ 的 sampling regime
- **Zhou et al. 2025 DARO**: rebalance loss across difficulty groups。Focal 更简单，per-prompt scalar
- **Gai et al. 2025 "differential smoothing"**: 改 reward 区分 correct/incorrect trajectory。Focal 是 orthogonal 的 prompt-level scaling

---

## 7. 我的几点 critique / open questions

1. **$\gamma$ sweep 只在 {0.5, 1.0, 2.0}**。Table 6 显示最终选 0.5 或 1.0，没看到 $\gamma$ 的 sensitivity 曲线。Focal loss 在 detection 里 $\gamma=2$ 是 default，这里偏小。可能 LLM RLVR 里梯度 scale 已经很 sensitive，大 $\gamma$ 会把 easy prompt 的梯度完全 kill 掉导致 underfit。

2. **$\hat{\mu}_{pos}$ 是有 noise 的 estimator**。当 N=8，$\hat\mu \in \{0, 1/8, ..., 1\}$，离散且 noisy。Lemma B.1 是 $\mathbb{E}[S_R | X=k]$ 的 monotonicity，但实际观察到的是 $X=k$ 的一个 sample，variance 不小。Paper 没讨论这个 noise 的影响。

3. **训练过程中 $\mu_{pos}(x)$ 在变**。easy prompt 训着训着可能变 hard，Focal weight 跟着变。这其实是 adaptive 的好处，但也意味着 $\gamma$ 的有效作用会随 training phase 漂移。Paper 没分析这个 dynamics。

4. **Categorical simulation 用 $\eta = 10^{-2}$**，比 Hu et al. 的 $10^{-3}$ 大 10×。理由是低 LR 下 entropy 保持太高 (above 4)，不像真实 LLM RLVR (entropy < 1)。这是个合理的工程选择，但意味着 simulation 的 regime 边界**不能直接外推到 LLM**——paper 自己也强调了这点（Section 5.1 末尾）。所以 concentration zone 在 LLM 里具体是哪个 N 区间，其实没有 quantitative prediction，只有 qualitative pattern。

5. **没 ablate Focal 的 functional form**。为什么是 $(1-\hat\mu)^\gamma$ 而不是 $-\log \hat\mu$ 或别的？Focal loss 的 $(1-p_t)^\gamma$ 在 classification 里是为了平衡 CE 的梯度，这里 motivation 是"down-weight high $S_R$ regime"。其实任何关于 $\hat\mu$ 单调递减的函数都行。可能 Focal form 不是最优，只是 convenient。

6. **OOD 提升 (IFEval, SynLogic, GPQA) 的 mechanism 没讲透**。Paper 说 "preserving solution diversity benefits generalization"，但 IFEval 是 instruction following，跟 math reasoning diversity 的关系不直接。可能是训练时保留了更广的 reasoning pattern，transfer 到 IFEval 的 instruction-following pattern 上。这点值得深挖。

---

## 8. 实践建议

如果你要 reproduce 或用：

- **N=8 是危险区**。如果你想 maximize pass@1 又 budget 有限，N=8 是合理选择，但要知道你在 sharpening regime。Focal weight 几乎 free（一个 scalar 乘法），加上去对冲。
- **$\gamma = 0.5$ 是稳妥 default**。Table 6 里 9 个配置中 6 个选 0.5。$\gamma = 1.0$ 更 aggressive，适合 base model 已经很集中、需要强力 redistribute 的场景。
- **Focal 是 orthogonal 的**，可以跟你任何 GRPO variant 叠加。改 reward (DAPO asymmetric clip, CISPO IS clipping) 不冲突。
- **判断你在哪个 regime**：训完看 pass@1 vs pass@256 的 trade-off。如果 pass@1 涨但 pass@256 跌得明显，你在 concentration zone；如果两个都涨，在 large-N regime。这能指导下一次实验调 N 还是加 Focal。
- **监控 $\Delta\text{NLL}_{rare}$** 这个 proxy（Appendix F.2 方法）。采 base model 的 low-prob 但 correct trajectories，训后看 NLL 变化。这是判断 sharpening 程度的直接指标。

---

## 9. 相关 references

- Focal loss 原文: [Lin et al. 2017, ICCV](https://arxiv.org/abs/1708.02002)
- GRPO: [DeepSeekMath, Shao et al. 2024](https://arxiv.org/abs/2402.03300)
- DAPO: [Yu et al. 2025](https://arxiv.org/abs/2503.14476)
- CISPO: MinimaxM1, [Chen et al. 2025a](https://arxiv.org/abs/2506.13585)
- Hu et al. "BroRL" categorical framework: [arXiv 2510.01180](https://arxiv.org/abs/2510.01180)
- Wu et al. "GRPO is secretly DPO": [arXiv 2510.00977](https://arxiv.org/abs/2510.00977)
- Yue et al. "Does RL really incentivize reasoning": [arXiv 2504.13837](https://arxiv.org/abs/2504.13837)
- He et al. "Rewarding the unlikely": [EMNLP 2025](https://arxiv.org/abs/2504.06001) (近似链接，具体 paper ID 你查一下)
- pass@k estimator: [HumanEval, Chen et al. 2021](https://arxiv.org/abs/2107.03374)
- verl framework: [Sheng et al. 2024](https://arxiv.org/abs/2409.19256)
- DeepScaleR: [Luo et al. 2025, Notion blog](https://twitter.com/ianazmin16/status/1898424220118569117) (实际是 Notion blog, arXiv 没正式版)
- DPO (背景): [Rafailov et al. 2023](https://arxiv.org/abs/2305.18290)
- Entropy 机制相关: [Cui et al. 2025](https://arxiv.org/abs/2505.22617)
- DARO (concurrent difficulty-aware): [Zhou et al. 2025](https://arxiv.org/abs/2510.09001)

这篇 paper 我觉得最值得借鉴的不是 Focal weight 本身（那是个 engineering trick），而是 **Lemma 3.1 的 closed form**——它把 group size 这个 hyperparameter 的角色从 "empirically tune" 变成了 " theoretically reason about"。下次有人问 "GRPO 用 N=8 还是 N=32"，你可以直接引这个公式说：取决于你的 $\mu_{pos}$ 和 $\tau$ 在哪，中间 N 是 concentration zone，要么往小走（cheap but slow），要么往大走（expensive but safe），中间最危险。Focal 是给踩中间的人一个 cheap parachute。
