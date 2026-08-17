---
source_pdf: Vector Policy Optimization Training for Diversity Improves Test-Time Search.pdf
paper_sha256: fb1e3d4f01a6526a6e29d0c37daab64cd2004836cab0ff20bde2919bbf3f08bc
processed_at: '2026-08-13T00:11:04-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# VPO 用人话版

## 一句话版本

GRPO 把模型训成只会说一种话的复读机，VPO 让模型学会在同一轮对话里输出"风格各异但各有道理"的几个答案，好让下游的 search 程序从一堆多样的候选里挑出最好的。

## 问题出在哪

你想像你训练一个 model 做 LeetCode 题。标准做法是 GRPO——给它一道题，让它生成多个 solution，根据 pass / fail 打分，强化好的、抑制差的。听起来没问题，但你训练久了会发现一件尴尬的事：

**model 开始所有 sample 都长得几乎一样**。

它学会了那一个"在平均意义上最优"的解法，然后每次都吐这个。你 sampling 10 次，10 次基本是同一段代码，可能只是变量名改了一下。diversity 掉到地板上（Maze 实验里 GRPO 的 diversity 是 0.003，几乎就是 0）。

这事本身不算 bug——这是 objective 的必然结果。你告诉它"最大化 pass@1"，它当然会 collapse 到那个最优解。问题在于：

**真实部署里，model 不是单独用的，它外面套着一层 search**。

比如 AlphaEvolve 这种 evolutionary search，会生成一堆 candidate → 选好的 → 改写 → 再选 → ... 这个 pipeline 要 work，前提是 model 能给它"不一样的"候选。你给它 30 个几乎一样的 candidate，它再怎么 search 也跳不出那个 basin。

所以标准 RL post-training 在 **训练阶段**就把 **inference 阶段需要的多样性**给杀死了。

## VPO 的核心 idea

既然 downstream 一定会 search，那 training 就别想"我要找出那个最优解"，改成"我要给 search 准备一桌菜，每道菜口味不同但都好吃"。

**怎么让 model 真的产生不一样的解？靠 reward 的维度结构。**

很多 task 的 reward 天然是 vector 不是 scalar：
- 代码生成：每个 test case 是一个维度（pass / fail）
- Multi-hop QA：每个 hop 的 citation 是一个维度
- Tool use：format / tool name / arg key / arg value 各是一个维度
- Maze：reach exit / collect gold / collect diamond / avoid lava 四个维度

标准 GRPO 把这些维度加权求和变成 scalar，然后只优化那一个 scalar。VPO 说：别 collapse，让 model 学会**在不同的 reward weighting 下各自专精**。

具体怎么做？两个 trick 同时上：

### Trick 1: 一次 rollout 出多个答案

标准做法：一次 forward pass 出一个 answer。要多样性？independent sample N 次。

VPO 做法：让 model 在**同一次** autoregressive decode 里顺序吐 $m=3$ 个答案，用 delimiter 分隔。这样生成第 2 个答案时，model 能 attend 到第 1 个答案，知道"这片已经覆盖了，我换条路"。多样性变成 in-context mechanism，而不是 sampling noise 的副产品。

**但光这个不够**。为什么？因为 reward 还是固定的 scalar，gradient 会把每个位置都往同一个 optimum 推，最后 3 个答案还是 collapse 到同一个。Multi-RLVR baseline 就证明了这点——它在 Maze / MuSiQue / ToolRL 上的 reward-space diversity 训练没几步就崩了。

### Trick 2: 每次训练随机换 weighting

VPO 在每个 rollout 上从 Dirichlet 分布采样一个 random weight $w = (w_1, ..., w_d)$，然后用这个 $w$ 算 scalar reward。一次训练里采样 K 个不同的 $w$，对 set $S$ 的 reward 是：

$$R(S) = \mathbb{E}_{w \sim \text{Dir}(1)}\left[\max_{y \in S} w^\top r(x, y)\right]$$

人话翻译：**"我从一堆 random 的 reward weighting 里随便挑一个，看你这套答案里哪个在这个 weighting 下最好；然后我对所有 weighting 求平均"**。

这玩意儿是个 **coverage objective**。如果你的 3 个答案全一样，那不管 $w$ 怎么取，$\max_{y \in S} w^\top r(x, y)$ 都是同一个值——你只 cover 了 simplex 上一个点。但如果 3 个答案各自在不同 $w$ 下最优，你 cover 了 simplex 上三个点，期望 max 就高。

**核心 intuition**：VPO 把 policy optimization 变成了 **reward simplex 上的 coverage problem**。model 不再被推向一个点，而是被推向 Pareto front。

两个 trick 缺一不可：
- Multi-answer 提供 diversity 的 **capacity**（你"能"不一样）
- Stochastic scalarization 提供 diversity 的 **incentive**（你"想"不一样）

## 为什么这个能 work - 训练 dynamics 的深层原因

paper 里给了两个 explanation，第二个比第一个更 subtle 也更有意思。

**Explanation 1 (static)**: 优化 $R(S)$ 让 policy 输出覆盖 Pareto front 的 diverse sets，search 自然能 extract value。这个 explanation 直白，但只是说"objective 长这样所以结果长这样"。

**Explanation 2 (dynamic, 更有意思)**: 想像一个 candidate 在标准 $w^*$ 下 score 一般，但在某个其它 $w$ 下非常好。GRPO 训练会怎么处理它？它的 advantage 是负的（因为它低于 group mean），gradient 会 push it away，它就消失了。这个 strategy 再也不会被 visit。

VPO 训练呢？在采到那个让它高分的 $w$ 的 rollout 里，它的 advantage 是正的，被 reinforce。在采到不利的 $w$ 的 rollout 里，advantage 是负的，被抑制。**净效果是这个 strategy 被保持下来**——它在某些 weighting 下活着，没被一刀切干掉。

这意味着 VPO 让 model 维持一个**更宽的 reasoning strategy 池子**，包括那些在 $w^*$ 下看起来 suboptimal 但其实包含有用 partial pattern 的 strategy。这些 pattern 在 test-time search 重组时可能产生比 $w^*$-trained policy 永远访问不到的高分 solution。

这个 explanation 难直接 measure，但 LiveCodeBench + OpenEvolve 的 case study 就是 indirect evidence：VPO 在 200 个 search iteration 后还能 crack 新问题，GRPO 早早 plateau——说明 VPO 的初始 candidate pool 里就有 GRPO pool 里没有的"种子"。

## 实验上最 striking 的几个数

### Maze 上的反直觉结果

Maze 的 reward 是 4 维，GRPO 优化的是 uniform mean $\bar{r} = (r_1 + r_2 + r_3 + r_4)/4$。

VPO 训练时用的是 Dirichlet 采样的 random $w$，**从没直接优化 uniform mean**。

但 eval 时用 uniform mean 比 best@k：

| Method | best@3 | best@30 |
|---|---|---|
| GRPO | 0.432 | 0.432 |
| **VPO** | **0.512** | **0.593** |

**你没用 uniform mean 训练，但你在 uniform mean 上 beat 了直接训练 uniform mean 的方法。** 这就是 reward diversity 的力量——通过 cover 整个 simplex，你顺带把 uniform mean 那个点附近的 best 也 cover 了，而且 cover 得比专门盯一个点的方法还好。

### LiveCodeBench + OpenEvolve

这是真正的 scaling test。训 Qwen2.5-Coder-7B 一个 epoch，唯一区别是 advantage estimator。

- Pass@1（无 search）：GRPO 赢。符合直觉，因为没 search 时 scalar 直接优化 single shot 就是最好
- Best@k（简单 search）：VPO 在每个 k 都赢，gap 随 k 增大
- **OpenEvolve（complex evolutionary search, 200 iterations）**：32 个最难问题（两方法在 best@30 都是 0），VPO 持续发现新 solution，crack 了 GRPO 完全 touch 不到的问题

这是最强的证据。diversity 的价值在 **search 越复杂、问题越难**时越明显。

### Goal-Conditioned baseline 为什么失败

一个 natural alternative 是：把 $w$ 作为 input 喂给 model，训它 condition on $w$ 最大化 $w^\top r$。这是 classic multi-objective RL 的做法。

结果：mode collapse + 忽略 conditioning。即使你 explicitly 告诉它"现在请优先优化第 3 个维度"，model 也搞不太定——把 text-encoded preference 翻译成 behavior 这件事本身对 LLM 就难。

VPO 用 in-context exploration 绕过了这个困难：不用让 model 理解 $w$ 是什么意思，直接让它在同一次 rollout 里自动 spread。

## 什么时候 VPO 不管用

paper 在 Appendix F 给了个 clean diagnostic：测量 on-policy reward vector 的 Pearson 相关性 $\bar{\rho}$。

| Domain | $\bar{\rho}_{VPO}$ | VPO vs GRPO |
|---|---|---|
| Maze | 0.39 | +0.161 |
| MuSiQue | 0.12 | +0.023 |
| EUREQA | 0.05 | +0.022 |
| ToolRL | 0.86 | +0.028 |
| UltraFeedback | **0.95** | **-0.004** |

UltraFeedback 用 ArmoRM-5 的 5 个维度，on-policy 下这 5 维高度 collinear，simplex 实际退化成一条线，每个 prompt 基本只有一个 Pareto-optimal candidate。这种情况下 VPO **就该输**——因为 reward 实际是 scalar，没有 diversity 可以 cover。

实践指导意义：**先量 on-policy collinearity，再决定要不要用 VPO**。

## Limitations

- **牺牲 pass@1**：VPO 明确 trade single-shot performance for search-augmented performance。如果你的应用就是一次 inference 出一个 answer，别用 VPO，用 GRPO
- **需要 vector reward**：reward 必须能分解成多个维度。好在多数 realistic task 都有 natural decomposition（per-test-case、per-hop、per-criterion）
- **Compute 比较难严格对齐**：VPO 一个 rollout 出 $m$ 个答案，比 single-answer 的 rollout 长，虽然有 prefix amortize 但不完全 equal。paper 用 3× rollout 的 baseline 部分缓解

## Big picture

paper 最后一句很强："As test-time search becomes more standardized, optimizing for diversity may need to become the **default** post-training objective."

这句话的 implication 是：如果 inference compute 越来越便宜，test-time search 越来越 standard，那 post-training 的 objective 就该重新设计——从"训练 model 找出那个最优解"变成"训练 model 给 search 准备 diverse 且 competent 的候选池"。VPO 是这个方向上的一个 clean instantiation。

---

**最压缩的 intuition**：test-time search 需要 diversity，但 standard RL 训练会杀 diversity。VPO 通过 (1) 一次 rollout 出多个 answer 给 model 提供"能 diverse 的能力"，(2) 随机采样 reward weighting 给 model 提供"想 diverse 的动机"，把 policy optimization 变成 reward simplex 上的 coverage problem。Search 越复杂，问题越难，VPO 越有用。Reward 维度越 collinear，VPO 越没用。

---

# VPO: Vector Policy Optimization 深度解读

## 一、Big Picture - 这篇paper在回答什么问题

这篇paper瞄准一个非常具体的modern AI system痛点：当LLM被嵌进search-augmented pipeline（AlphaEvolve、Best-of-N、Tree-of-Thoughts）里时，RL post-training（特别是GRPO这类policy gradient方法）会**主动摧毁**search procedure需要的diversity。

核心观察链条是这样的：

1. GRPO优化标量 reward → 概率质量集中到单一mode → pass@k在大k时saturate（额外的samples变成near-duplicates，参见[GX-Chen et al. 2025](https://arxiv.org/abs/2510.20817)关于KL-reg造成mode collapse的结构性论证，[Kirk et al. 2024](https://openreview.net/forum?id=PXD3FAVHJT)关于RLHF侵蚀pass@k的实证）
2. 但下游系统需要diverse candidate pool才能extract value from test-time compute
3. 标准方法试图让单个训练算法同时handle exploration和exploitation，这个设计是错的

VPO的核心proposition是把这两个职责**完全切开**：training阶段只生产"diverse且competent"的candidate sets，test-time search负责exploitation（selection）。这本质上是把policy optimization重新frame成reward simplex上的**coverage problem**。

## 二、Reward Diversity - 概念上的关键转移

paper提出一个非常precise的diversity定义叫**reward diversity**，这是理解整个方法的关键。diversity不是token-level的variation，也不是semantic diversity，也不是noisy sampling带来的diversity。reward diversity指：candidate pool里的每个solution各自在不同的reward component weighting下是最优的。

形式化设定：
- prompt $x$，response $y \sim \pi_\theta(\cdot | x)$
- reward自然分解为 $d$ 维向量 $r(x, y) = [r_1(x,y), \ldots, r_d(x,y)] \in \mathbb{R}^d$
  - 代码生成：per-test-case correctness（[Chen et al. 2021, HumanEval](https://arxiv.org/abs/2107.03374)）
  - RLHF：per-criterion preference scores
  - Multi-hop reasoning：per-hop correctness（[MuSiQue, Trivedi et al. 2022](https://aclanthology.org/2022.tacl-1.32/)）
  - Agentic tasks：per-tool-call结构/内容分（[ToolRL, Qian et al. 2025](https://arxiv.org/abs/2504.13958)）

任何weight $w \in \Delta^{d-1}$（simplex上）诱导一个标量目标 $w^\top r(x,y)$。标准post-training固定 $w^\*$ 并最大化 $\mathbb{E}_{y\sim\pi_\theta}[w^{*\top} r(x,y)]$。

**关键洞察**：即使deployment scalar $w^\*$ 已知，保留在其它 $w$ 下optimal的candidate仍然有用。原因是search在**集合**上操作，而不是单response。一个reward-diverse pool给search更多机会发现under $w^\*$ 本身的高分solution。很多在 $w^\*$ 下local suboptimal的candidate包含部分推理pattern、分解、strategy，最终组合出更高分的outcome。

这与classical multi-objective RL（[Roijers et al. 2013](https://www.jair.org/index.php/jair/article/view/10836)）和lexicase selection（[Spector 2012](https://hampshire.edu/), [La Cava et al. 2019](https://direct.mit.edu/evco/article-abstract/27/3/377/1041/A-Probabilistic-and-Multi-Objective-Analysis-of)）有关联，但目标不同：不追求condition-on-preference的policy，deployment objective已知且固定就是 $w^\*$。只是search-augmented regime下，最优优化 $w^\*$ 的方式可能是train一个**保持reward diversity**的policy。

## 三、VPO方法 - 两个互补组件

### 3.1 Multi-Answer Chains as In-Context Exploration

跟随[Puri et al. 2026 "Reaching Beyond the Mode"](https://arxiv.org/abs/2603.24844)，模型在单次autoregressive rollout内生成 $m$ 个candidates $S = \{y_1, \dots, y_m\}$。candidates顺序输出，用delimiter token分隔，所以生成 $y_i$ 时prefix已包含 $y_1, \ldots, y_{i-1}$。

这从根本上改变了exploration的性质：

- **标准independent sampling**：diversity仅来自stochastic decoding施加在固定conditional distribution上，只能产生policy已集中的mode周围的小variation
- **Multi-answer rollout**：每个新candidate可以attend到前面已经emit的candidates，模型有capacity识别solution space哪些区域被覆盖，主动把后续candidate推向不同区域
- Diversity成为explicit、in-context mechanism，而不是sampling noise的副产品

但是！这个机制只提供diversity的**capacity**，不提供**incentive**。paper在Section 5实验验证：Multi-RLVR（multi-answer + 固定scalar reward）的reward diversity仍然在训练早期就collapse。原因显然——固定 $w^\*$ 下，gradient把chain每个位置都推向同一个scalar optimum。

### 3.2 Set-Level Optimization via Stochastic Scalarization

核心目标函数（**Equation 1**）：

$$R(S) = \mathbb{E}_{w \sim \text{Dir}(\alpha)}\left[\max_{y \in S} w^\top r(x, y)\right]$$

变量含义逐一解释：

- $S = \{y_1, \ldots, y_m\}$：一个prompt $x$ 对应的candidate set
- $w \in \Delta^{d-1}$：$d$ 维 simplex 上的 weight vector（满足 $w_i \geq 0$ 且 $\sum_i w_i = 1$）
- $\text{Dir}(\alpha)$：Dirichlet distribution，concentration parameter $\alpha \in \mathbb{R}_{>0}^d$；paper用 $\alpha = \mathbf{1}$（即各分量均为1），对应 simplex 上的**uniform distribution**
- $r(x,y) \in \mathbb{R}^d$：vector reward，第 $i$ 个分量 $r_i$ 评估response第 $i$ 个aspect
- $\max_{y \in S} w^\top r(x,y)$：在给定weighting $w$ 下，set $S$ 中scalarized reward最高的candidate的分数
- 整体：对所有可能的weighting $w$ 求期望，得到set $S$ 的"coverage score"

这个objective直接奖励reward space的coverage：set中不同element在不同 $w$ 采样下是最优的。collapse到identical responses的set只在simplex很窄的区域表现好；span多个trade-off的set在多种scalarization下都好。本质上在直接优化**期望best-of-N over sampled $w$**。

Monte-Carlo估计（**Equation 2**）：

$$\hat{R}(S^{(g)}) = \frac{1}{K} \sum_{k=1}^{K} \max_{s \in S^{(g)}} w^{(k)\top} r(x, s)$$

变量含义：

- $S^{(g)} = \{y_1^{(g)}, \ldots, y_m^{(g)}\}$：第 $g$ 个rollout产生的set（一个group共 $G$ 个rollouts）
- $w^{(1)}, \ldots, w^{(K)} \overset{\text{iid}}{\sim} \text{Dir}(\mathbf{1})$：$K$ 个iid采样的scalarization weights
- 关键设计：**这 $K$ 个weights在 $G$ 个rollouts间共享**，所以 $G$ 个sets在相同的 $w$ draws下评估，使得GRPO的advantage比较well-defined

### 3.3 与GRPO的结合

VPO只修改reward计算，可以与任何policy gradient方法结合。paper用GRPO（[Shao et al. 2024, DeepSeekMath](https://arxiv.org/abs/2402.03300)）作为backbone：

1. 每个prompt $x$ 采样 $G$ 个rollouts，每个产生 $m$ 个completions
2. 共享 $K$ 个scalarization weights $w^{(1)}, \ldots, w^{(K)} \sim \text{Dir}(\mathbf{1})$
3. 计算per-rollout reward $\hat{R}(S^{(g)})$
4. GRPO advantage: $\hat{A}_i = (\text{score}_i - \mu_g) / (\sigma_g + \epsilon)$，$\epsilon = 10^{-6}$，用population std
5. Advantage uniform apply到rollout $g$ 中每个token（通过response mask broadcast）

注意：**无critic network、无GAE**，这是GRPO的标准做法。

### 3.4 两个组件为什么necessary - 消融直觉

| 组件 | 提供 | 缺什么 |
|---|---|---|
| Multi-answer alone | capacity for diversity | incentive to be diverse |
| Random scalarization alone | incentive for diversity | 稳定性（每个rollout只一个answer，scalarization不停切换造成gradient instability）|
| VPO（两者结合）| 稳定的set-level objective | - |

直觉上：multi-answer让模型"能"产生不同candidates（in-context mechanism）；stochastic scalarization让模型"想"产生不同candidates（gradient signal按不同 $w$ 分配给chain中不同位置）；二者缺一不可。

## 四、实验设计精妙之处

### 4.1 四个domain - 跨越不同reward structure

| Domain | Model | $d$ | Reward structure | GRPO scalar |
|---|---|---|---|---|
| **Maze** | Qwen3-4B | 4 | 1 binary completion + 3 continuous (gold/diamond/lava) | uniform mean |
| **MuSiQue** | Qwen3-1.7B | 5 | 4 binary hop indicators + 1 continuous F1 | $(\sum \text{hop}_i + 3\cdot\text{answer\_f1})/7$ |
| **EUREQA** | Qwen3-8B | 5 | 5 binary entity-EM | uniform mean |
| **ToolRL** | Qwen3-1.7B | 4 | 1 binary format + 3 continuous F1 | uniform mean |

Maze特别巧妙：它是**人为设计**的trade-off。9×9 grid，用Prim's algorithm生成spanning tree，再注入 $n_{\text{cycles}} \sim \text{Unif}\{18, \ldots, 28\}$ 个额外opening。关键设计是budget约束：$budget = \max(\text{via\_gold}, \text{via\_diam}) + 7$，强制 $\text{via\_both} > budget$，所以**没有任何单一路径能同时收集gold和diamond corner并到达E**。这就engineer出了reward competition——这是控制实验的精妙之处。

### 4.2 Baseline设计 - 精准定位每个组件的贡献

paper的6个baseline不是随便选的，每个都精准test一个假设：

1. **GRPO**：test普通scalar RL是否已经够好
2. **Multi-RLVR** ([Puri et al. 2026](https://arxiv.org/abs/2603.24844))：test multi-answer generation alone是否sufficient
3. **Random-Weighting GRPO**：test randomizing scalarization alone（无set-level）是否够
4. **Max-at-k** ([Bagirov et al. 2025](https://arxiv.org/abs/2510.23393))：test直接优化inference-aware best@k是否够
5. **MaxRL** ([Tajwar et al. 2026](https://arxiv.org/abs/2602.02710))：test更强scalar search-aware objective能否恢复VPO的gain
6. **Goal-Conditioned GRPO**：test classic multi-objective RL替代方案——条件在 $w$ 上——是否更好

### 4.3 评估指标

核心指标**best@k**：

$$\text{best@}k(x) = \max_{s \in S_k(x)} w^{*\top} r(x, s)$$

- $w^\*$：per-domain GRPO training scalar（注意，**评估用training scalar**，这是个fair的setup，因为部署目标已知）
- $S_k(x)$：从训练好的policy对prompt $x$ 采样的 $k$ 个completion的pool
- Multi-answer方法：draw $\lceil k/m \rceil$ 个独立multi-answer chains，concatenate，取前 $k$ 个
- Single-answer方法：draw $k$ 个iid completion

辅助指标**reward-space diversity**：pool中所有candidate pair的reward vector的 $L_1$ 距离的均值。collapse到单mode的pool即使token表面有variation，div也→0。

## 五、核心实验结果深度分析

### 5.1 主结果（Tables 1-4）

看 MuSiQue (Table 1) 的 best@k 随 $k$ 增长的曲线：

| Method | best@3 | best@5 | best@10 | best@30 | diversity |
|---|---|---|---|---|---|
| GRPO | 0.711 | 0.716 | 0.721 | 0.728 | 0.054 |
| Multi-RLVR | 0.599 | 0.616 | 0.627 | 0.633 | 0.814 |
| **VPO** | **0.742** | **0.780** | **0.809** | **0.832** | 0.587 |

GRPO在 $k=3$ 到 $k=30$ 间几乎不涨（0.711 → 0.728），diversity只有0.054——典型的mode collapse。VPO从0.742一路涨到0.832，diversity保持0.587（vs Multi-RLVR的0.814但score低）。这说明Multi-RLVR有diversity但quality差，VPO达到了diversity+quality的sweet spot。

Maze (Table 2) 更明显：

| Method | best@3 | best@30 | diversity |
|---|---|---|---|
| GRPO | 0.432 | 0.432 | **0.003** |
| VPO | 0.512 | 0.593 | 1.006 |

GRPO的diversity低到0.003——基本所有samples是near-duplicates。VPO的diversity是1.006，而且在Maze上VPO用的是**uniform mean**评估，正是GRPO直接优化的目标。即使如此VPO还胜出，这是counter-intuitive的关键证据。

### 5.2 排除compute解释（Table 5）

一个合理的concern：VPO每个rollout产生 $m=3$ 个candidates，是否只是因为有3×的evaluator signal？paper给了GRPO/GDPO（[Liu et al. 2026](https://arxiv.org/abs/2601.05242)）3×的rollouts（n=24 vs VPO的n=8）：

| Method | $\mathbb{E}_w[\text{best@}k]$ |
|---|---|
| GRPO (n=24, 3× compute) | 0.763 |
| GDPO (n=24, 3× compute) | 0.765 |
| VPO (n=8) | **0.779** |

注意这给baseline 3×的**LM compute too**，因为single-answer方法的chain是independent的。所以比较非常conservative against VPO，但VPO仍胜。这排除compute解释，也排除"per-component normalization更好"的假设（GDPO track GRPO，indicates normalization不是binding constraint）。

### 5.3 Goal-Conditioned baseline为何失败（Table 6）

| Method | best@3 | $\mathbb{E}_w[\text{best@}k]$ |
|---|---|---|
| G.C. $w = w^*$ | 0.205 | 0.201 |
| G.C. $w \sim \text{Dir}(\mathbf{1})$ | 0.205 | 0.201 |
| VPO | 0.512 | 0.512 |

Goal-conditioned policy**mode collapse**了（best@3 = best@6），并开始**忽略conditioning**。即使explicitly给 $w$ 作为input，model仍难以把text-encoded preference转成behavior。这是interesting finding，说明in-context exploration比explicit conditioning更有效。

### 5.4 LiveCodeBench case study - 真正的scaling test

[LiveCodeBench (Jain et al. 2025)](https://openreview.net/forum?id=chfJJYC3iL)有严格temporal held-out cut（每个problem有contest date，held-out slice Aug 2024–Feb 2025严格晚于training data，构造上排除contamination）。Train Qwen2.5-Coder-7B-Instruct on DeepCoder corpus for one epoch，唯一区别是advantage estimator。

**Figure 4 关键观察**：

- **(A) Pass@1**：GRPO更好。这是符合直觉的——无search时scalar baseline wins
- **(B) Best@k**（用 $m=3$ candidate chain）：VPO在每个 $k$ 都高于GRPO，gap随 $k$ 增长
- **(C, D) OpenEvolve** ([Sharma 2025](https://github.com/algorithmicsuperintelligence/openevolve))：32个**最难**的held-out problem（best@30时两方法都0分），200 iterations，$m=3$ per iteration（≈600 candidates/problem）

OpenEvolve结果是**最striking**的evidence：VPO持续发现新solution，crack了GRPO完全无法touch的问题，GRPO早早就plateau了。这是VPO thesis的核心实证——**diversity在non-trivial search procedure下价值最大**，且问题越难、search越capable，VPO的benefit越明显。

### 5.5 Reward Collinearity - 失败模式诊断（Appendix F）

这是paper最漂亮的分析之一。paper提供一个**predictive diagnostic**：测量on-policy reward vector的Pearson correlation $\bar{\rho}$（off-diagonal平均值），预测VPO是否help。

| Domain | $\bar{\rho}_{\text{VPO}}$ | $\bar{\rho}_{\text{GRPO}}$ | best@16 VPO | best@16 GRPO | Δ |
|---|---|---|---|---|---|
| Maze | 0.39 | 0.37 | 0.593 | 0.432 | **+0.161** |
| MuSiQue | 0.12 | 0.11 | 0.864 | 0.841 | +0.023 |
| EUREQA | 0.05 | 0.03 | 0.204 | 0.182 | +0.022 |
| ToolRL | 0.86 | 0.62 | 0.953 | 0.924 | +0.028 |
| UltraFeedback | **0.95** | 0.82 | 0.767 | 0.772 | **-0.004** |

UltraFeedback ([Cui et al. 2023](https://arxiv.org/abs/2310.01377)) + [ArmoRM-5](https://arxiv.org/abs/2406.12845)的5个nominally distinct维度实际near-collinear，simplex collapse到near-line，每个prompt基本只有一个Pareto-optimal candidate，VPO**败给**GRPO（虽然headroom最大）。

这是个很clean的结论：**VPO只在reward components genuinely compete时work**。当reward effectively scalar时，VPO不该（也不会）beat scalar GRPO。这给实践者一个直接的diagnostic工具。

## 六、Training Setup细节（App. B）

- Backbone: GRPO on [veRL](https://arxiv.org/abs/2409.19256) (Sheng et al. 2024)
- Advantage: z-score $\hat{A}_i = (\text{score}_i - \mu_g)/(\sigma_g + \epsilon)$, $\epsilon=10^{-6}$
- PPO clip $\epsilon=0.2$（symmetric, dual-clip $c=3.0$）
- ppo_epochs=1, token-mean loss aggregation, **无entropy bonus**
- KL: low-variance $k_3$ estimator against frozen reference (= SFT init), $\beta_{\text{KL}}=10^{-3}$
- AdamW, lr=$10^{-6}$, $(\beta_1,\beta_2)=(0.9, 0.999)$, weight decay 0.01, gradient clip 1.0, no warmup
- FSDP1, bf16 mixed precision (param dtype fp32)
- Training rollouts: vLLM, temperature 1.0, top-p 1.0, top-k -1
- Per-domain batch: Qwen≤4B (Maze/MuSiQue/ToolRL) train 128, mini 64, micro 8, n=8; Qwen 7B/8B (EUREQA) train 64, mini 32, micro 2, n=8

注意 **没有entropy bonus**，这与传统鼓励exploration的做法相反——VPO通过objective本身（stochastic scalarization）提供incentive，不需要外加regularizer fight against mode collapse（如paper Section 6最后所说："coverage of the reward simplex is the equilibrium rather than something a regularizer fights for"）。

## 七、Intuition总结 - 为什么VPO work

paper给两个complementary解释（Section 7）：

**解释1（static）**：优化 $R(S)$ 的policy会产生覆盖Pareto front的reward-diverse sets，test-time search从中extract value。

**解释2（dynamic，更subtle）**：在 $w^\*$ 下score差但在某些其它 $w$ 下score好的candidate，在那些 $w$ 被采样的rollouts上仍得到positive gradient；固定 $w^\*$ 的训练会push这些candidate away。所以VPO能保持broader set of reasoning strategies存活足够长时间被refine，包括 $w^\*$-trained policy**永远不会visit**的strategy。这可能是为什么VPO在 $w^\*$ 评估下仍beat $w^\*$-trained policy的根因——但paper承认这点hard to measure。

第二个解释连接到[Setlur et al. 2025 e^3](https://arxiv.org/abs/2506.09026)和[Qu et al. 2026 POPE](https://arxiv.org/abs/2601.18779)关于privileged on-policy exploration的工作，但VPO通过reward structure而不是额外training mechanism实现exploration。

## 八、Limitations

paper诚实列出三个limitation：

1. **Compute equalization难**：每个method output长度不同，VPO每rollout产生 $m$ 个completion（虽然reasoning prefix部分被 $m$ 个solution amortize）。Section 5的3×rollout ablation部分address this。
2. **需要vector reward**：scalar-only reward时VPO退化成standard RL。但其实多数realistic task都有natural vector decomposition（per-test-case、per-criterion、per-hop），这个约束相对mild。
3. **牺牲pass@1换pass@k**：VPO明确trade off single-shot performance for search-augmented performance。**VPO只适用于test-time search是system一部分的regime**。

## 九、对test-time scaling的implication

paper最后一句话很strong："As test-time search becomes more standardized, optimizing for diversity may need to become the default post-training objective."

如果这个prediction成立，VPO可能标志着RL post-training范式的shift：从"optimize for one best response"到"optimize for coverage of competent solutions"。这呼应[Brown et al. 2024 "Large Language Monkeys"](https://arxiv.org/abs/2407.21787)关于inference-time scaling的实证，以及[Snell et al. 2024关于test-time scaling laws](https://arxiv.org/abs/2408.03314)的趋势。当inference compute变得便宜且standardized，training objective需要重新设计以feed这个pipeline。

## 十、可能的extension和open question

几个paper没完全回答的方向：

1. **$m$ 和 $K$ 的选择**：paper全程用 $m=3$，没系统研究 $m$ 如何trade off capacity vs context length / 训练稳定性
2. **Dirichlet $\alpha$ 的选择**：固定 $\alpha=1$（uniform），但 $\alpha$ 越小越concentrated在vertex（更激进的trade-off specialization），$\alpha$ 越大越接近uniform weighting（更接近GRPO）。这是个未探索的design knob
3. **与process reward model的结合**：VPO用outcome reward，但multi-answer chain里不同position可以有per-step reward，可能进一步enhance in-context exploration
4. **长horizon agentic tasks**：paper的task都是单turn或短chain，长horizon下multi-answer chain的attention结构可能带来挑战
5. **DPO变体**：paper用GRPO，但VPO的reward计算block可以直接drop-in到DPO/DAPO等preference optimization方法

## 参考链接

- Paper PDF (基于提供的markdown): 本conversation的attachment
- [DeepSeekMath/GRPO - Shao et al. 2024](https://arxiv.org/abs/2402.03300)
- [Reaching Beyond the Mode - Puri et al. 2026 (Multi-RLVR来源)](https://arxiv.org/abs/2603.24844)
- [AlphaEvolve - Novikov et al. 2025](https://arxiv.org/abs/2506.13131)
- [OpenEvolve (Sharma 2025)](https://github.com/algorithmicsuperintelligence/openevolve)
- [LiveCodeBench - Jain et al. 2025](https://openreview.net/forum?id=chfJJYC3iL)
- [MuSiQue - Trivedi et al. 2022](https://aclanthology.org/2022.tacl-1.32/)
- [ToolRL - Qian et al. 2025](https://arxiv.org/abs/2504.13958)
- [KL-reg causes mode collapse - GX-Chen et al. 2025](https://arxiv.org/abs/2510.20817)
- [RLHF erodes pass@k - Kirk et al. 2024](https://openreview.net/forum?id=PXD3FAVHJT)
- [Multi-objective RL survey - Roijers et al. 2013](https://www.jair.org/index.php/jair/article/view/10836)
- [Lexicase selection - La Cava et al. 2019](https://direct.mit.edu/evco/article-abstract/27/3/377/1041/)
- [GDPO - Liu et al. 2026](https://arxiv.org/abs/2601.05242)
- [MaxRL - Tajwar et al. 2026](https://arxiv.org/abs/2602.02710)
- [Max@k - Bagirov et al. 2025](https://arxiv.org/abs/2510.23393)
- [veRL - Sheng et al. 2024](https://arxiv.org/abs/2409.19256)
- [e^3 exploration - Setlur et al. 2025](https://arxiv.org/abs/2506.09026)
- [Large Language Monkeys - Brown et al. 2024](https://arxiv.org/abs/2407.21787)

---

**核心intuition take-away**：VPO把RL post-training重新frame成reward simplex上的coverage problem。通过(a)单rollout内生成 $m$ 个candidates提供in-context diversity的capacity，(b)用Dirichlet采样的随机scalarization在set-level上提供diversity的incentive，VPO让policy保持一个reward-diverse candidate pool，让test-time search能持续extract value as $k$ grows。这是把"explore during training, exploit during search"的职责彻底分开的clean instantiation，并且通过collinearity诊断predictable地work或不work。
