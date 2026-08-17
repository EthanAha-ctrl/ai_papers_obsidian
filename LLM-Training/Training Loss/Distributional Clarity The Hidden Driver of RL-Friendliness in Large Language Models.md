---
source_pdf: Distributional Clarity The Hidden Driver of RL-Friendliness in Large Language
  Models.pdf
paper_sha256: 5faf590fa9d00dab4976d07800aa8a44a29183b3cafb81ccc586c7a2deda1fb8
processed_at: '2026-08-03T22:35:27-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：这篇paper到底在说啥

## 一句话总结

**Qwen天生就在内部probability分布上把"对答案"和"错答案"分得很开，Llama分不开，所以Qwen能被RL训练好，Llama训不动。如果你硬逼Llama在那些"分不开"的样本上多下功夫，它也能涨。**

就这么简单。

---

## 问题从哪儿来

大家做RLVR（用RL训练LLM推理）都发现一个尴尬事：

- Qwen2.5-7B 用 DAPO 训练，AIME24 能从 5.5 涨到 12.2
- Llama-3.1-8B 用一模一样的 DAPO 训练，AIME24 只从 3.7 涨到 6.1
- OctoThinker（Llama改的）也只从 1.5 涨到 4.9

同样的算法，同样的数据，结果差距悬殊。社区给了个名字叫"RL-Friendliness"——有的model就是"RL-friendly"，有的就是"RL-resistant"。

之前的解释主要两条路线：

1. **Data-centric**：Llama mid-training没见过足够高质量数学语料，pretraining留下的"底子"不够，需要补data。代表工作OctoThinker https://arxiv.org/abs/2506.20512
2. **Behavior-centric**：模型需要先具备self-verification、backtracking这些"cognitive behaviors"才能self-improve。代表工作Gandhi et al. https://arxiv.org/abs/2503.01307

这篇paper说：这些都对，但漏了一个orthogonal视角——**模型内部的probability分布长什么样**。

---

## 三阶段分析：从现象到本质

### Stage 1：现象——能解的题差不多，pass rate差很多

作者拿Qwen2.5-7B和OctoThinker-8B（都是DAPO训练后的版本）在AIME24上每个题都sample 256次，算每个题的pass rate $\rho(q)$（就是256次里答对的比例）。

发现一个诡异的事：

- 两个模型"能解的题"重合度其实很高（71.4% / 90.9%）
- 但在重合的题里，Qwen几乎每个题的pass rate都比OctoThinker高
- 把数据画在scatter plot上（Figure 2），绝大多数点在对角线下方

**人话**：不是Llama不会做这些题，而是Llama做这些题时"时灵时不灵"，Qwen则是"基本都灵"。

这就引出一个关键insight：**RL-Friendliness不是"能不能解"的问题，是"解得稳不稳"的问题**。

### Stage 2：机制——probability分布长得不一样

那"稳不稳"在model内部对应什么？作者定义了一个length-normalized sequence probability：

$$P(o|q) = \left(\prod_{i=1}^{L} P(y_i | q, y_{<i})\right)^{\frac{1}{L}}$$

人话解释：
- $o = (y_1, y_2, \dots, y_L)$ 是response，长度 $L$
- $P(y_i | q, y_{<i})$ 是model给第 $i$ 个token的概率
- 整个response的joint probability是所有token概率的连乘
- 取 $\frac{1}{L}$ 次方（geometric mean）是为了消除长度影响，不然长response概率天然就低

对每个query，sample一批responses，按对错分两组，画kernel density estimate（Figure 3）：

- **Qwen**：对responses的概率分布是一个mode，错responses的概率分布是另一个mode，两个mode分得开开的
- **Llama/OctoThinker**：对和错的概率分布严重overlap，甚至错的response概率更高

**人话**：Qwen内部"知道自己什么时候是对的"，给对答案分配的概率系统性高于错答案。Llama内部混乱，给对错答案分配的概率差不多，甚至错答案概率更高。

这跟RL有啥关系？看Stage 2的理论分析（Appendix B）。

#### 为啥分布分开RL才能work

GRPO/DAPO的policy gradient本质上是：

$$\nabla_\theta J \propto \underbrace{\mathbb{E}_{o_j \sim \pi^+}[\nabla_\theta \pi_\theta(o_j|q)]}_{\text{推高对的}} - \underbrace{\mathbb{E}_{o_k \sim \pi^-}[\nabla_\theta \pi_\theta(o_k|q)]}_{\text{压低错的}}$$

对softmax输出层，关键公式是：

$$\nabla_\theta \pi_\theta(y|x) = \pi(y|q)(1 - \pi(y|q)) \cdot \nabla_\theta z$$

那个 $\pi(1-\pi)$ 是Bernoulli variance——在 $\pi=0.5$ 时最大（最不稳定），在 $\pi \to 0$ 或 $\pi \to 1$ 时趋零（最稳定）。

所以gradient variance近似正比于：

$$\text{Var}(\nabla_\theta J) \propto \text{Var}_{\pi^+}[\pi(1-\pi)] + \text{Var}_{\pi^-}[\pi(1-\pi)]$$

人话：
- 如果对responses概率都聚集在某个高值附近（compactness），那 $\pi(1-\pi)$ 在group内接近常数，variance小，gradient方向稳定
- 如果错responses概率也都聚集在某个低值附近，同理
- 如果两个group还分得开（separation），那gradient方向是朝着"推高对、压低错"的明确方向
- 如果两个group混在一起，$\pi(1-\pi)$ 在group内随机变化，gradient方向乱跳，update相当于在原地抖动

**核心insight**：RL训练的噪声主要不来自reward稀疏或数据，来自**model自身probability分布的"散度"**。

#### Silhouette Coefficient：一个数度量"分开程度"

作者从cluster analysis借来Silhouette Coefficient $S$，量化这个"分开程度"。

对每个response $P_i$，它到同类其他点的平均距离 $a_i$，到异类点的平均距离 $b_i$，定义：

$$s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$$

- $b_i - a_i$：希望异类远（$b_i$大）、同类近（$a_i$小），所以越大越好
- 除以 $\max(a_i, b_i)$ 是normalize到 $[-1, 1]$
- $s_i \to 1$：理想分离
- $s_i \to 0$：临界overlap
- $s_i \to -1$：分到错误cluster

query-level的 $S$ 就是所有 $s_i$ 的平均。

**人话**：$S$ 是一个数，告诉你model在这个query上把对错response分得有多清楚。$S>0$ 说明分得开，$S<0$ 说明分不开甚至倒置。

#### 关键实验数据（Figure 4）

按 $S$ 分两组看pass rate：

| Model | $S>0$ queries pass rate | $S<0$ queries pass rate | Gap |
|---|---|---|---|
| Qwen | 52.3% | 6.5% | 8× |
| OctoThinker | 40.8% | 3.1% | 13× |

差距巨大。这说明 $S$ 几乎是个"训练前的天气预报"——训练前算一下 $S$，就能预测model在哪些query上能被RL改善。

### Stage 3：解释——S高低对应什么语义现象

作者用LLM-as-judge（Qwen3-32B）分析MATH-500上的错误，分三档severity：

| Severity | 错误类型 | 例子 |
|---|---|---|
| High (Fundamental) | Core logic或knowledge错 | 用错定理、计划方法本身错、repetition loop |
| Mid (Execution) | 计算错误 | 公式对但算错、step hallucination |
| Low (Presentation) | 格式问题 | 没加 \boxed{}、generation提前截断 |

发现（Figure 5）：
- Qwen的high severity错误占比明显低
- Llama的fundamental errors中 **62.2%** 都对应 $S<0$
- Low severity errors更倾向于 $S>0$

**人话**：$S$ 低的错误是"模型压根不知道正确方向"的逻辑错误，$S$ 高的错误是"模型知道答案只是格式没对齐"的小毛病。

然后看solution stability（Figure 6）：
- Qwen在每个题上稳定收敛到少数几个valid method
- Llama/OctoThinker的"correct responses"看起来method很多样，但其实是reasoning instability——每次靠不同spurious path蒙对
- 高 $S$ queries → method ratio低（稳定）
- 低 $S$ queries → method ratio高（飘忽）

**完整因果链**：低 $S$ → model内部对"什么是正确approach"模糊 → 每次sample走不同spurious path → RL的target policy在动 → RL无法converge

---

## 干预实验：Silhouette-Aware Reweighting

既然低 $S$ 是瓶颈，那就专门在这些query上多花训练预算。

### 标准DAPO advantage

$$\hat{A}_{i,t} = \frac{R_i - \mu(\{R_j\}_{j=1}^G)}{\sigma(\{R_j\}_{j=1}^G)}$$

- $R_i$ 是response $o_i$ 的reward（0或1）
- $\mu, \sigma$ 是group rewards的均值和标准差
- 所有问题用同样的weight（1）

### 加reweighting

$$\tilde{A}_{i,t} = \hat{A}_{i,t} \cdot w(q)$$

$$w(q) = \exp(-\beta \cdot S')$$

其中 $S'$ 是rectified silhouette：

$$S' = \begin{cases} S & \text{if } P_{pos} \geq P_{neg} \\ -|S| & \text{if } P_{pos} < P_{neg} \end{cases}$$

**为啥要rectify**？因为原始 $S$ 在inverted distribution（错response概率反而比对response高）下也可能为正——只要inverted分布也separated。直接优化 $S$ 会把分布往错误方向推得更开。Rectify之后，inverted case强制 $S' = -|S|$，告诉weight说"这个query不仅分不开，还方向反了，要特别关注"。

**$w(q)$ 的行为**（Figure 7，$\beta=0.2$）：
- $S' < 0$（低clarity）：$w > 1$，放大gradient signal
- $S' > 0$（高clarity）：$w < 1$，压制已easy的样本
- 用 $\exp$ 保证 $w > 0$（不会翻转advantage符号）
- $\beta$ 控制sensitivity，$\beta \to 0$ 退化为standard DAPO

**人话**：把更多训练预算花在model最confused的queries上，逼它澄清decision boundary，而不是在已经分得清楚的easy queries上overfit。

---

## 实验结果：三个family都涨

### 主结果（Table 1）

我只列AIME24和average：

| Model | AIME24 | 6-bench Avg |
|---|---|---|
| Qwen base | 5.5 | 22.0 |
| Qwen + DAPO | 12.2 | 42.1 |
| Qwen + DAPO-Silhouette | **18.1** | **43.0** |
| OctoThinker base | 1.5 | 14.3 |
| OctoThinker + DAPO | 4.9 | 27.8 |
| OctoThinker + DAPO-Silhouette | **8.2** | **30.1** |
| Llama base | 3.7 | 18.8 |
| Llama + DAPO | 6.1 | 21.3 |
| Llama + DAPO-Silhouette | **7.9** | **22.3** |

**关键观察**：
- OctoThinker在AIME24从4.9翻倍到8.2，AIME25也翻倍
- Llama从6.1涨到7.9（30%相对提升），对一个"几乎RL不动"的family算显著
- 即使Qwen已经强，仍能从12.2→18.1，说明distributional clarity是universal bottleneck
- 在easy benchmarks（AMC23、MATH500）上gains小，因为本来就high-S主导

### S和pass rate的correlation

训练过程中S和pass rate的Pearson correlation $r = 0.815$。说明 $S$ 不是static指标，是跟训练动态耦合的state variable。

不同family的trajectory：
- Llama/OctoThinker：起点 $S$ 极低，standard DAPO几乎不推 $S$，Silhouette strategy大幅推高 $S$，pass rate同步上升
- Qwen：起点 $S$ 较高，仍能进一步推高

### Ablation关键结论（Table 2）

**1. Fisher Ratio也能work**（29.8 vs 30.1）

换一个metric衡量distributional clarity：

$$F = \frac{(\mu_{pos} - \mu_{neg})^2}{\sigma_{pos}^2 + \sigma_{neg}^2}$$

- 分子：类间距离平方（separation）
- 分母：类内方差和（compactness的倒数）
- 直观：1D Linear Discriminant Analysis的objective

效果跟Silhouette差不多。**说明benefit不来自特定metric，来自"distributional clarity"这个property本身**。

**2. Inter-class Separation Only**（28.8 vs 30.1）

只最大化margin不compactness，效果不如完整Silhouette。**证明intra-class compactness同样关键**。

**3. Pass-Rate Reweighting**（28.7 vs 30.1）

按pass rate给低分query加权（传统hard sample mining），效果远不如Silhouette。**说明benefit不是简单的hard sample mining**——pass rate只看outcome，S看distributional structure，后者信息丰富得多。

**4. Random Reweighting**（27.5 < baseline 27.8）

随机加noise反而略掉。**说明reweighting本身不能产生gain，需要informative signal**。

**5. $\beta$ 敏感性**

$\beta=0.1$（温和）：avg 28.8
$\beta=0.2$（适中）：avg 30.1，AIME24最好8.2
$\beta=0.5$（激进）：avg 30.8，但AIME24只6.6

**说明challenging benchmarks需要更温和的reweighting**（避免overfit hard queries），easy benchmarks可受益于更激进。

---

## Build你的intuition

### Intuition 1：RL是"决策边界锐化"，不是"行为习得"

model base本身在某些query上已经有内部清晰度，RL只是把这种清晰度投影到probability空间并amplify。低清晰度的query，RL无法amplify一个不存在的东西。

### Intuition 2：Pass rate = model给correct response的概率mass的empirical估计

Qwen和Llama能解的题差不多，但Qwen每次sample都稳定hit，Llama每次走不同spurious path偶尔蒙对。这跟RL无关，是pretraining/midtraining留下的"内部清晰度"差异。

### Intuition 3：S是"训练前的天气预报"

训练前sample 16次算 $S$，就能预测model在哪些query上会被RL改善。工程上很有价值——可以提前filter低 $S$ queries，或选高 $S$ model省算力。

### Intuition 4：Error severity ↔ Distributional clarity的对应

- High severity errors（logic错）→ 低 $S$：model压根不知道正确方向
- Low severity errors（格式错）→ 高 $S$：model知道答案，只是输出格式没对齐

用 $S$ 就能自动诊断"无药可救"vs"小毛病"。

### Intuition 5：Solution stability是RL能否converge的前提

RL要learn一个stable policy $\pi^*$。如果base model每次sample走不同path，RL的target policy在动——不同rollout对应不同target，根本无法converge。

高 $S$ → model内部已收敛到少数valid strategy → RL只需strengthen这些strategy → stable training
低 $S$ → model每次乱走 → RL的target飘忽 → training stuck

---

## 我觉得这篇paper真正牛的地方

**Concept-level**：提出"distributional clarity"这个model property概念，把三个看似无关的现象（pass rate disparity、error severity、solution stability）统一在一个framework下。

**Method-level**：把cluster analysis的Silhouette Coefficient移植到RLVR分析。Silhouette是1970s老东西，用在LLM probability distribution上确实新颖。

**Practical-level**：Silhouette-Aware Reweighting极简——advantage乘一个weight就行，不改architecture，不加data，三个family都涨。

**Diagnostic-level**：三阶段分析框架（phenomenon → mechanism → interpretation）可以作为template分析其他RL failure modes。未来某个model在code RL上不work，可以套用：先看pass rate disparity，再看probability分布的 $S$，再看error severity和solution stability。

---

## 可能的延伸方向（作者没做但值得探究）

1. **Token-level Silhouette**：现在用sequence-level geometric mean，但不同token对clarity贡献不同，token-level可能更细粒度
2. **Multi-turn / Agentic setting**：每个turn都有correct/incorrect，Silhouette要扩展到trajectory level
3. **Long reasoning (>10k tokens)**：paper limitation承认没测。Long CoT下sequence probability estimation noise大，$S$ 可能不稳定
4. **Curriculum Learning based on $S$**：先训高 $S$ queries建立稳定gradient signal，再引入低 $S$
5. **$S$ 作为model selection metric**：在多个base model间选RL-Friendliness最高的，不用实际跑RL（省算力）
6. **跟mechanistic interpretability结合**：高 $S$ vs 低 $S$ 的model，attention pattern和residual stream上有什么差异？揭示distributional clarity的mechanistic origin
7. **Mid-training data selection based on $S$**：mid-training选data时优先选能提升 $S$ 的corpora，而非只看knowledge coverage

---

## 跟其他工作的关联

- **DAPO** https://arxiv.org/abs/2503.14476：本paper的backbone algorithm
- **GRPO / DeepSeekMath** https://arxiv.org/abs/2402.03300：advantage normalization的origin
- **DeepSeek-R1** https://arxiv.org/abs/2501.12948：RLVR的里程碑工作
- **OctoThinker** https://arxiv.org/abs/2506.20512：data-centric路线的代表
- **Gandhi cognitive behaviors** https://arxiv.org/abs/2503.01307：behavior-centric路线的代表
- **SimplerLZoo** https://arxiv.org/abs/2503.18892：系统研究Llama的RL瓶颈
- **Spurious Rewards** https://arxiv.org/abs/2506.10947：反思reward signal本身的问题
- **Reasoning with Exploration** https://arxiv.org/abs/2506.14758：entropy-based exploration，跟Silhouette互补
- **Seed-GRPO** https://arxiv.org/abs/2505.12346：semantic entropy估计uncertainty
- **ICPO** https://arxiv.org/abs/2511.21005：confidence-aware advantage reshape
- **Negative Reinforcement** https://arxiv.org/abs/2506.01347：negative-only RLVR也能work
- **Murphy PML book** https://probml.github.io/pml-book/book1.html：Silhouette Coefficient的reference
- **Bishop PRML** https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/：Fisher Ratio的reference
- **verl framework** https://arxiv.org/abs/2409.19056：训练用的RL框架

---

## 局限性

paper自己承认的：

1. **依赖group sampling**（$G=16$）：Silhouette估计需要足够多samples，budget极小时失效
2. **只测standard CoT**：没测ultra-long reasoning（>10k tokens），那种情况下sequence probability estimation noise大
3. **Math reasoning only**：code、agentic等其他domain是否适用还没验证
4. **Binary reward only**：continuous reward下"correct/incorrect"分类要重新定义
5. **Rectified $S'$ 的设计有点heuristic**：当 $P_{pos} < P_{neg}$ 时强制 $S' = -|S|$，可能有更principled的处理方式

---

## 最终感想

这篇paper的核心信息其实很朴素：**model在RL之前内部状态长什么样，决定了RL能不能work**。

之前社区把"RL-Friendliness"归结为data问题或behavior问题，这篇paper说，还有一个更底层的structural property——model的probability分布是否已经把对错response分开了。如果分开了，RL是amplification；如果没分开，RL是noise amplification。

这个insight可以推而广之：任何"用signal去强化某个behavior"的训练，都需要model内部对"target behavior vs non-target behavior"有清晰区分。区分度越高，训练越高效。区分度低的时候，再多的训练都只是在噪声上叠加噪声。

Silhouette-Aware Reweighting本身是个很小的trick，但背后的思想——**先诊断model在哪里confused，再针对性训练**——是个值得推广的范式。比传统的"统一训练所有data"或"hard sample mining"都更principled。

用最直白的话说：**与其让model反复刷它已经会的题，不如让它专门练它最迷糊的题**。听起来像常识，但这次有了理论解释和实验验证。

---

# Distributional Clarity: RL-Friendliness的隐藏驱动力

## 1. 核心问题与动机

这篇paper想解释一个LLM社区普遍观察到的诡异现象:同样的RLVR pipeline (GRPO/DAPO),同样的训练data,Qwen family能从RL拿到巨大收益,Llama family却几乎纹丝不动。这就是所谓的"RL-Friendliness gap"。

之前的解释主要走data-centric路线:
- **Cognitive Behaviors** (Gandhi et al., 2025): 认为模型需要先具备self-verification、backtracking等"认知行为",fine-tune这些pattern能部分补齐gap。https://arxiv.org/abs/2503.01307
- **OctoThinker** (Wang et al., 2025c): 认为Llama midtraining阶段没见过足够高质量数学corpora,需要补mid-training。https://arxiv.org/abs/2506.20512
- **SimplerLZoo** (Zeng et al., 2025): 系统研究Llama在zero RL下的瓶颈。https://arxiv.org/abs/2503.18892
- **Spurious Rewards** (Shao et al., 2025a): 反思RLVR training signal本身可能含spurious成分。https://arxiv.org/abs/2506.10947

这篇paper开辟orthogonal视角:不问"模型学到了什么",而问"模型的内在probability landscape长什么样"。核心thesis:**distributional clarity** —— 即模型给correct responses分配的概率 vs incorrect responses分配的概率,在1D probability space上是否形成well-separated、intra-class compact的两个cluster —— 是RL能否work的前提条件。

---

## 2. Three-Stage Analysis Framework

### Stage 1: Phenomenon — Pass Rate Disparity

设query $q$,采样 $K$ 个独立responses,pass rate定义为:

$$\rho(q) = \frac{1}{K}\sum_{k=1}^{K}\mathbb{1}[\text{verify}(o_k, q) = 1]$$

其中 $\mathbb{1}[\cdot]$ 是indicator function,$o_k$ 是第 $k$ 个sampled response。

实验在AIME 2024上 $K=256$,对比Qwen2.5-7B (DAPO) 和 OctoThinker-8B (DAPO)。关键观察:

- 两个模型能解的problem set重合度很高 (intersection约71.4% / 90.9% of各自solvable set)
- 但在intersection中,大多数数据点落在 $y=x$ 对角线下方,意味着Qwen对相同问题拿到的pass rate系统性更高
- **RL-Friendliness的本质在于"how reliably a model solves it",而不仅仅是"what it can solve"**

这个观察的intuition:RL的credit assignment依赖group-relative reward signal。如果模型本身在某个query上correct response出现频率极低,group内几乎全是negative samples,advantage normalization后positive signal被淹没。

### Stage 2: Mechanism — Compactness & Separation

定义length-normalized sequence probability:

$$P(o|q) = \left(\prod_{i=1}^{L} P(y_i | q, y_{<i})\right)^{\frac{1}{L}} \tag{1}$$

变量含义:
- $o = (y_1, y_2, \dots, y_L)$: 长度为 $L$ 的response
- $y_i$: 第 $i$ 个token
- $y_{<i}$: 前缀tokens $y_1, \dots, y_{i-1}$
- $P(y_i | q, y_{<i})$: autoregressive model在prefix条件下生成 $y_i$ 的概率
- 外层 $\frac{1}{L}$: 几何平均,做length normalization

为何用geometric mean而非arithmetic mean?因为token probabilities连乘本身就是joint probability,geometric mean相当于取log后做arithmetic mean,等价于average log-likelihood per token,这才是perplexity的逆,长度无关。

**关键观察 (Figure 3)**: Qwen的correct responses概率分布和incorrect responses概率分布呈现两个well-separated mode;而Llama/OctoThinker的两个分布严重overlap,甚至incorrect responses的概率分布拖尾超过correct responses。

#### Silhouette Coefficient的数学定义

对单个query $q$,有 $K$ 个responses,每个response有序列概率 $P_i$,被分成两个cluster:$C_{\text{same}}$ (与 $P_i$ 同类的cluster) 和 $C_{\text{opposite}}$ (异类cluster)。

**Intra-cluster distance** $a_i$:

$$a_i = \frac{1}{|C_{\text{same}}| - 1}\sum_{P_j \in C_{\text{same}}, j \neq i} |P_i - P_j| \tag{2}$$

- $|C_{\text{same}}|$: 同簇样本数,减1排除自身
- $|P_i - P_j|$: 1D L1距离
- 直观含义:$P_i$ 到其同簇其他点的平均距离,$a_i$ 越小 → 同类越compact

**Inter-cluster distance** $b_i$:

$$b_i = \frac{1}{|C_{\text{opposite}}|}\sum_{P_k \in C_{\text{opposite}}} |P_i - P_k| \tag{3}$$

- $|C_{\text{opposite}}|$: 异簇样本数
- 直观含义:$P_i$ 到所有异簇点的平均距离,$b_i$ 越大 → 异类越separated

**Individual silhouette** $s_i$:

$$s_i = \frac{b_i - a_i}{\max(a_i, b_i)} \tag{4}$$

- 分子 $b_i - a_i$: 希望 $b_i$ 大 (异类远) 且 $a_i$ 小 (同类近),所以越大越好
- 分母 $\max(a_i, b_i)$: normalization,把 $s_i$ 压缩到 $[-1, 1]$
- $s_i \to 1$: 理想cluster结构
- $s_i \to 0$: 临界overlap
- $s_i \to -1$: $P_i$ 被分到错误cluster,即 $a_i \gg b_i$

**Query-level coefficient**:

$$S = \frac{1}{|\mathcal{O}|}\sum_{P_i \in \mathcal{O}} s_i$$

- $|\mathcal{O}|$: 所有response的总数 ($= K$)
- $S \in [-1, 1]$

为什么用Silhouette Coefficient而非更直接的Fisher Discriminant Ratio?Silhouette是per-sample的,能给出每个response的"局部clearance",而Fisher只给group-level的ratio。但作者ablation实验证明两者效果相当,说明**核心是distributional clarity本身,而非具体metric formulation**。

#### 高低S的pass rate差距 (Figure 4)

- Qwen: $S>0$ 的queries pass rate = 52.3%, $S<0$ 的queries pass rate = 6.5% (8× gap)
- OctoThinker: $S>0$ = 40.8%, $S<0$ = 3.1% (13× gap)

这个gap的存在说明S本身就是一个predictor of trainability,几乎是一个"训练前的天气预报"。

### Stage 3: Interpretation — Error Severity & Solution Stability

#### Error Taxonomy (Table 3)

作者用Qwen3-32B做LLM-as-a-judge,把错误分为三个severity:

| Severity | Code | Description |
|---|---|---|
| **High (Fundamental)** | E1.1 | Misunderstanding question |
| | E1.2 | Constraint violation |
| | E2.1 | Knowledge error (wrong formula/theorem) |
| | E2.2 | Planning/method error |
| | E5.1 | Repetition loop |
| | E5.2 | Irrelevant/incoherent |
| **Mid (Execution)** | E3.1 | Calculation error |
| | E3.2 | Step hallucination |
| **Low (Presentation)** | E4.1 | Format error (missing \boxed{}) |
| | E4.2 | Premature stop |

**关键发现 (Figure 5)**:
- Qwen的错误中high severity占比例明显较低
- Llama的fundamental errors中 **62.2%** 都对应 $S<0$ (低clarity)
- Low severity errors更倾向于 $S>0$ (模型方向对了,只是格式问题)

Intuition: 如果模型对问题的core logic本身就模糊,它在probability space上自然分不开correct和incorrect response;反之,只是格式错误说明模型内部已经"知道"答案,probability自然分配得清楚。

#### Solution Stability (Algorithm 1)

用incremental clustering算法:对每个query收集所有correct responses,逐个用LLM judge判断是否属于已有method cluster (基于相同theorem/formula/logical strategy),否则新建cluster。

定义 **distinct solution ratio** = 不同method cluster数 / 总correct responses数。

**Figure 6的发现**:
- Qwen: ratio低且随difficulty稳定 → 收敛到少数几个valid strategy
- OctoThinker/Llama: ratio高 → 但其实是reasoning instability
- $S \geq 0$ 的queries → ratio显著低
- $S < 0$ 的queries → ratio显著高

**因果链总结**: 低distributional clarity → 模型在内部representation上对"什么是正确approach"模糊 → 每次sample时随机走不同spurious path → distinct ratio高 → RL无法稳定识别要reinforce的behavior → RL训练失败

---

## 3. Theoretical Grounding: Gradient Variance Analysis (Appendix B)

这是整篇paper最关键的insight来源。GRPO/DAPO的policy gradient可以写成:

$$\nabla_\theta J \propto \mathbb{E}_{o_j \sim \pi^+}[\nabla_\theta \pi_\theta(o_j|q)] - \mathbb{E}_{o_k \sim \pi^-}[\nabla_\theta \pi_\theta(o_k|q)] \tag{9}$$

变量含义:
- $\pi^+(\cdot|q)$: 给定query $q$ 时correct responses的分布
- $\pi^-(\cdot|q)$: incorrect responses的分布
- $\pi_\theta(o|q)$: 参数为 $\theta$ 的policy对response $o$ 给的概率
- $o_j$ 和 $o_k$: 分别从两个分布sampled

定义两个group的expected gradient:
$$g_+ = \mathbb{E}_{o_j \sim \pi^+}[\nabla_\theta \pi_\theta(o_j|q)], \quad g_- = \mathbb{E}_{o_k \sim \pi^-}[\nabla_\theta \pi_\theta(o_k|q)]$$

假设correct和incorrect responses独立sampled,总gradient variance:

$$\text{Var}(\nabla_\theta J) = \text{Var}(g_+) + \text{Var}(g_-) \tag{10}$$

**Transformer softmax层的关键性质**: 对于logit $z$ 和probability $\pi = \text{softmax}(z)$:

$$\frac{\partial \pi}{\partial z} = \pi(1-\pi)$$

由chain rule:

$$\nabla_\theta \pi_\theta(y|x) = \pi(y|q)(1 - \pi(y|q)) \cdot \nabla_\theta z \tag{11}$$

- $\nabla_\theta z$: logit对参数的gradient,主要来自backbone的hidden representation
- $\pi(1-\pi)$: 这一项是Bernoulli variance,在 $\pi=0.5$ 时最大,在 $\pi \to 0$ 或 $\pi \to 1$ 时趋零

由于transformer的hidden representations经过LayerNorm等normalization,variance大致稳定。所以gradient的stochasticity主要由 $\pi(1-\pi)$ 决定:

$$\text{Var}(\nabla_\theta J) \propto \text{Var}_{\pi^+}[\pi(1-\pi)] + \text{Var}_{\pi^-}[\pi(1-\pi)] \tag{12}$$

这里 $\text{Var}_{\pi^+}[\pi(1-\pi)]$ 是 $\text{Var}_{o_j \sim \pi^+}[\pi(o_j|q)(1-\pi(o_j|q))]$ 的简写。

**核心结论**: 
- 当correct responses的概率 $P(o_j|q)$ 紧密分布在某个值 $\pi_+$ 周围 (intra-class compactness),$\text{Var}_{\pi^+}[\pi(1-\pi)]$ 很小
- 当incorrect responses的概率紧密分布在 $\pi_-$ 周围,同理
- 两个cluster还well-separated → gradient方向稳定 + 真正朝着"增correct减incorrect"目标

如果分布overlap,有些correct responses概率接近0 (gradient ≈ 0,信号弱),有些incorrect responses概率接近0.5 (gradient大但方向错乱),variance爆炸,RL signal被噪声淹没。

这个分析给了一个非常漂亮的intuition:**RL训练的噪声主要不来自reward sparsity或data,而来自模型自身probability分布的"散度"**。

---

## 4. Silhouette-Aware Reweighting

### Standard DAPO/GRPO Advantage

对group of $G$ outputs $\{o_i\}_{i=1}^G$ 和对应rewards $\{R_i\}$:

$$\hat{A}_{i,t} = \frac{R_i - \mu(\{R_j\}_{j=1}^G)}{\sigma(\{R_j\}_{j=1}^G)} \tag{5}$$

- $R_i$: response $o_i$ 的verifiable reward (0 or 1)
- $\mu(\{R_j\})$: group rewards的均值
- $\sigma(\{R_j\})$: group rewards的standard deviation
- $\hat{A}_{i,t}$: response $o_i$ 在token位置 $t$ 的advantage (group内所有token共享同一advantage,这就是GRPO的核心简化)

**问题**: 如果group内全错 ($R_i=0, \forall i$),$\sigma=0$ 出现除零;或者极不平衡 (1正15负),positive样本的advantage被巨大负值主导,梯度信号方向不稳。这些情况正好对应低S queries。

### Silhouette-Aware Reweighting

引入query-specific weight $w(q)$:

$$\tilde{A}_{i,t} = \hat{A}_{i,t} \cdot w(q) \tag{6}$$

**Rectified Silhouette** $S'$ 处理"inverted distribution" (即incorrect responses概率反而高于correct responses):

$$S' = \begin{cases} S & \text{if } P_{pos} \geq P_{neg} \\ -|S| & \text{if } P_{pos} < P_{neg} \end{cases} \tag{7}$$

- $P_{pos}$: correct responses的平均序列概率
- $P_{neg}$: incorrect responses的平均序列概率
- $S$: 原始Silhouette Coefficient
- $S'$: rectified版本,当分布倒置时强制为负

为什么不直接用 $S$?因为原始 $S$ 在inverted case下仍可能为正 (如果inverted分布也separated),优化 $S$ 会把分布往错误方向推得更开。

**Reweighting factor**:

$$w(q) = \exp(-\beta \cdot S') \tag{8}$$

- $\beta > 0$: sensitivity hyperparameter (paper默认 $\beta = 0.2$)
- $S' < 0$ (低clarity): $w(q) > 1$ → 放大gradient signal
- $S' > 0$ (高clarity): $w(q) < 1$ → 压制已easy的样本

为何用 $\exp$ 而非线性? $\exp$ 保证 $w(q) > 0$ (不会翻转advantage符号),且单调递减。$\beta$ 控制sensitivity: $\beta \to 0$ 时 $w \to 1$ (退化为standard DAPO),$\beta \to \infty$ 时 $w$ 变成sharp的hard sample mining。

**Intuition**: 把更多"训练预算"花在模型最confused的queries上,迫使它澄清decision boundary,而不是在已经分得清楚的easy queries上overfit。

---

## 5. 实验结果分析

### Main Results (Table 1)

| Model | AIME24 | AIME25 | MATH500 | AMC23 | Minerva | Olympiad | Avg |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B (base) | 5.5 | 2.6 | 50.7 | 30.5 | 19.4 | 23.1 | 22.0 |
| + DAPO | 12.2 | 11.8 | 79.7 | 70.2 | 36.4 | 42.4 | 42.1 |
| + DAPO-Silhouette | **18.1** | 12.0 | 80.4 | 70.2 | 34.3 | 43.2 | 43.0 |
| OctoThinker-8B (base) | 1.5 | 0.7 | 39.0 | 18.0 | 13.3 | 13.2 | 14.3 |
| + DAPO | 4.9 | 2.1 | 59.4 | 46.9 | 27.6 | 25.8 | 27.8 |
| + DAPO-Silhouette | **8.2** | **5.0** | 62.6 | 47.2 | 29.6 | 27.8 | **30.1** |
| Llama-3.1-8B-Instruct | 3.7 | 0.4 | 46.8 | 24.7 | 21.7 | 15.4 | 18.8 |
| + DAPO | 6.1 | 0.5 | 51.4 | 25.2 | 25.1 | 19.4 | 21.3 |
| + DAPO-Silhouette | **7.9** | 0.4 | 53.1 | 27.7 | 25.4 | 19.4 | **22.3** |

**关键观察**:
- Llama在AIME24从6.1→7.9,绝对值小但相对涨幅30%,对一个"几乎RL不动"的family算显著
- OctoThinker在AIME24翻倍 4.9→8.2,AIME25翻倍 2.1→5.0
- Qwen自身已强,但仍能从12.2→18.1 (48%相对提升),说明distributional clarity是universal bottleneck
- 在easy benchmarks (AMC23, MATH500) 上gains较小,因为本来就是high-S queries主导

### S vs Pass Rate的correlation (Figure 8)

训练过程中S和pass rate的Pearson correlation $r = 0.815$,非常强。这说明S不是一个static指标,而是一个**与训练动态耦合的state variable**。

不同模型family的trajectory:
- Llama/OctoThinker: 起点S极低,standard DAPO几乎不推S,而Silhouette strategy把S大幅推高,pass rate同步上升
- Qwen: 起点S较高,但仍能进一步推高

### Robustness & Ablation (Table 2)

| Method | AIME24 | AIME25 | MATH500 | Avg |
|---|---|---|---|---|
| DAPO (baseline) | 4.9 | 2.1 | 59.4 | 27.8 |
| DAPO-Silhouette ($\beta=0.2$) | 8.2 | 5.0 | 62.6 | 30.1 |
| **Metric Generalization** | | | | |
| DAPO-Fisher Ratio | 7.5 | 4.5 | 63.6 | 29.8 |
| **Mechanism Validation** | | | | |
| Inter-class Separation Only | 5.8 | 2.2 | 62.4 | 28.8 |
| Pass-Rate Reweighting | 5.0 | 2.6 | 61.1 | 28.7 |
| Random Reweighting | 5.0 | 1.8 | 60.5 | 27.5 |
| **Hyperparameter Sensitivity** | | | | |
| $\beta = 0.1$ | 6.9 | 3.4 | 62.8 | 28.8 |
| $\beta = 0.5$ | 6.6 | 6.3 | 65.5 | 30.8 |

**关键ablation结论**:

1. **Fisher Ratio也能work** (29.8 vs 30.1): 说明benefit不来自Silhouette Coefficient这个特定metric,而来自"distributional clarity"这个underlying property本身。Fisher Ratio定义为:
   $$F = \frac{(\mu_{pos} - \mu_{neg})^2}{\sigma_{pos}^2 + \sigma_{neg}^2}$$
   - $\mu_{pos}, \sigma_{pos}^2$: correct responses概率的均值和方差
   - $\mu_{neg}, \sigma_{neg}^2$: incorrect responses概率的均值和方差
   - 分子: 类间距离平方 (separation)
   - 分母: 类内方差和 (compactness倒数)
   - 直观: 等价于1D Linear Discriminant Analysis的objective

2. **Inter-class Separation Only** (28.8) 弱于完整Silhouette (30.1): 证明intra-class compactness同样关键,只推separation会陷入"虽远但不紧"的状态,gradient variance仍高

3. **Pass-Rate Reweighting** (28.7) 远不如Silhouette: 证明benefit不是简单的hard sample mining。Pass-rate本质是outcome statistics,而Silhouette是distributional structure,后者信息更丰富

4. **Random Reweighting** (27.5) 略低于baseline: 证明reweighting本身不能产生gain,需要informative signal

5. **$\beta$ 敏感性**: $\beta=0.1$ (温和) 平均28.8,$\beta=0.5$ (激进) 平均30.8,$\beta=0.2$ (适中) AIME24最好8.2。说明challenging benchmarks需要更温和的reweighting (避免overfit hard queries),而easy benchmarks可受益于更激进reweighting

### Fisher Ratio的具体实现 (Appendix C.3)

由于Fisher Ratio无界,paper加clamping:

$$w(q) = \text{clip}(\exp(-\beta \cdot F'), 0.95, 1.05)$$

- $F'$: rectified Fisher Ratio (类似 $S'$ 处理inverted case)
- $\beta = 0.01$ (Fisher数值大,需小$\beta$)
- clip范围 $[0.95, 1.05]$: 防止极端reweighting破坏training stability

这个clip细节暴露了Silhouette的一个隐藏优势: bounded $[-1, 1]$ 自然给出数值稳定的reweighting range,而Fisher需要手动clip。

---

## 6. 训练配置细节 (Table 4)

| Hyperparameter | Value |
|---|---|
| Max prompt length | 2048 |
| Max response length | 8192 |
| Advantage estimator | GRPO |
| Clip ratio (low) | 0.2 |
| Clip ratio (high) | 0.28 |
| Responses per prompt | 16 ($G$) |
| Sampling temperature | 1.0 |
| Sampling top-p | 1.0 |
| KL in reward | False |
| KL loss | False |
| Optimizer | AdamW |
| Learning rate | $1 \times 10^{-6}$ |
| Warmup steps | 10 |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |
| Batch size | 512 |
| Mini-batch size | 32 |

**值得注意的细节**:
- **KL全关**: DAPO的core innovation是去除KL penalty (vs PPO),让模型自由drift。这对Silhouette-Aware Reweighting特别友好,因为我们的reweighting已经在做"局部regularization"
- **Asymmetric clip ratio** (0.2/0.28): DAPO的clip-higher trick,防止negative advantage过度dominate,保留exploration
- **Group size 16**: 这是计算Silhouette的最小sample数,paper limitation提到"statistical precision benefits from sufficient sample size"
- **Temperature 1.0 + top-p 1.0**: 几乎是greedy-free的最大entropy sampling,保证group diversity

### Evaluation sampling

- AIME 2024/2025: $K = 256$ (高精度pass rate估计)
- MATH-500: $K = 32$
- AMC23/Minerva/OlympiadBench: $K = 16$
- Decoding: temperature=0.6, top-p=0.95, max tokens=8192

---

## 7. 完整Intuition总结

让我把整个故事线串起来,构建你的intuition:

### Intuition 1: RL训练的本质是gradient方向的稳定性

GRPO/DAPO每次update的gradient,由两部分组成:$g_+$ (推高correct responses概率) 和 $-g_-$ (压低incorrect responses概率)。Gradient的方差主要来自 $\pi(1-\pi)$ 项。

当模型内部已经"清楚"哪些response是对的 ($\pi_+ \approx$ 集中) 哪些是错的 ($\pi_-$ 集中),每个group内的 $\pi(1-\pi)$ 都接近常数,gradient variance小,update方向稳定一致,RL高效推进。

当模型内部"模糊",correct和incorrect的概率混在一起,$\pi(1-\pi)$ 在group内随机变化,gradient方向噪声大,update相当于在原地抖动。

### Intuition 2: Pass rate disparity背后是probability mass分配

为什么Qwen和Llama能解的题目overlap很大但pass rate差很多?因为**pass rate就是模型给correct response分配的概率mass的empirical估计**。Qwen在内部representation上已经"知道"这些query的correct approach,所以每次sample都能稳定hit;Llama每次sample都随机走不同spurious path,偶尔蒙对,大部分时间走偏。

这跟RL无关,这是**pretraining/midtraining留下的"内部清晰度"差异**。RL只能在已有清晰度上做amplification,无法从模糊中创造清晰。

### Intuition 3: Silhouette Coefficient是"训练前的天气预报"

S本质上测量模型对单个query的"内部confident assignment"程度。高S意味着模型在probability space上已经把correct和incorrect分得很开,RL只需要把这两个cluster的距离再拉开一点;低S意味着两个cluster粘在一起,RL的signal (gradient) 完全无法区分要push哪个方向。

S在训练前就能算出来 (sample $K=16$ 次就行),所以它是一个**prognostic metric**,可以预测某个model在某个query上是否会被RL改善。这在工程上很有价值:可以提前filter掉低S queries或low-S models。

### Intuition 4: 重加权低S样本 = 主动澄清decision boundary

Standard DAPO对每个query一视同仁。但实际上,高S query RL已经能轻松handle (gradient稳定),低S query才是真正的训练瓶颈。把更多gradient budget投到低S query上,等价于"花更多iteration去澄清模型最模糊的decision boundary"。

这跟传统的hard sample mining (pass-rate reweighting) 不同:pass-rate只看outcome (对/错),而S看的是**模型内部representation的清晰度**。一个model可能pass rate高但S低 (蒙对的多),也可能pass rate低但S高 (只是separation不够sharp)。

### Intuition 5: Error severity与distributional clarity的对应关系

为什么high severity errors (logic error, knowledge error) 对应低S?因为这些errors的本质是模型内部"不知道正确方向",反映在probability space上就是correct和incorrect混在一起。

为什么low severity errors (format error) 对应高S?因为模型内部已经"知道答案",只是输出格式没对齐,probability上correct response cluster非常紧致,跟incorrect responses分离。

这给了一个非常有用的诊断工具:用S就能自动识别哪些错误是"无药可救" (需要重新pretrain) 而哪些只是"小毛病" (RL能fix)。

### Intuition 6: Solution stability是RL训练目标可实现性的proxy

RL训练目标是learn一个stable policy $\pi^*$。如果base model本身在不同sample下走不同spurious path,RL需要learn的"target policy"是模糊的 — 不同rollout对应不同target。这种情况下RL根本无法converge,因为它的target在动。

高S的queries → model内部已经收敛到少数几个valid strategy → RL只需要strengthen这些strategy → stable training
低S的queries → model每次走不同路 → RL的target飘忽 → training stuck

---

## 8. 进一步思考与可能的延伸

### 跟其他工作的潜在连接

1. **Confidence-aware RL**: 工作如ICPO (Wang et al., 2025a, https://arxiv.org/abs/2511.21005) 也用model confidence reshape advantage。区别:ICPO用single-sample confidence,Silhouette用group-level distributional structure。

2. **Entropy-based exploration**: Reasoning with Exploration (Cheng et al., 2025, https://arxiv.org/abs/2506.14758) 用token-level entropy进advantage。这跟Silhouette互补:entropy调控exploration,Silhouette调控decision boundary的sharpness。

3. **Semantic entropy**: Seed-GRPO (Chen et al., 2025b, https://arxiv.org/abs/2505.12346) 用semantic entropy估计uncertainty。Semantic entropy衡量"答案的语义diversity",Silhouette衡量"概率的separation",两者相关性高但侧重不同。

4. **Spurious Rewards** (Shao et al., 2025a): paper揭示RLVR在noisy reward下仍能work。Silhouette的视角补充:即使reward clean,如果model自身distributional clarity低,RL依然会失败 — 所以瓶颈不一定在reward signal,也可能在model的internal structure。

5. **Mid-training matters** (OctoThinker, https://arxiv.org/abs/2506.20512; Midtraining bridges distributions, Liu et al., 2025b, https://arxiv.org/abs/2510.14865): mid-training用高质量数学corpora可能本质上就是在build distributional clarity。这是一个值得探究的方向:**mid-training data选择是否应该基于"提升distributional clarity"而非"覆盖更多knowledge"?**

6. **Negative Reinforcement** (Zhu et al., 2025, https://arxiv.org/abs/2506.01347): negative-only RLVR也能improve reasoning。这跟Silhouette框架一致:negative reinforcement本质就是把 $\pi_-$ cluster推离correct cluster,改善inter-class separation。

### 可能的扩展方向

1. **Token-level Silhouette**: paper用sequence-level geometric mean。但不同token对distributional clarity贡献不同,可能在token level定义更细的silhouette会更有信息量。

2. **Multi-turn / Agentic setting**: paper局限于single-turn math reasoning。在multi-turn agent setting,每个turn都有correct/incorrect responses,Silhouette需要扩展到trajectory level。

3. **Long reasoning (>10k tokens)**: paper limitation承认没研究ultra-long reasoning。Long CoT下sequence probability的estimation noise大,Silhouette可能不稳定。

4. **Curriculum Learning based on S**: 用S做curriculum,先训练高S queries建立稳定gradient signal,再逐步引入低S queries。这可能比uniform reweighting更稳定。

5. **Active data selection**: 用S做online data filtering,只保留training过程中S有提升的queries。结合DeepSeek-R1的group sampling,几乎不增加overhead。

6. **S as model selection metric**: 在multiple base models之间选择,可以用average S作为RL-Friendliness的predictor,而不需要实际跑RL (省算力)。

7. **Connection to mechanistic interpretability**: 高S vs 低S的model,在attention pattern和residual stream上有什么差异?这可能揭示distributional clarity的mechanistic origin。

### 局限性

- **需要group sampling** ($G=16$): Silhouette估计依赖足够多的samples,sample budget极小时失效。虽然GRPO/DAPO本来就要group sample,但Silhouette的precision仍然受sample数限制
- **只测standard CoT**: 没测long reasoning (>10k tokens)。Long CoT的sequence probability受estimation noise影响大
- **Math reasoning only**: 其他domain (code, agentic) 是否同样适用还需验证
- **Single reward signal**: 只考虑verifiable reward (0/1)。对continuous reward,需要重新定义Silhouette的"correct/incorrect"分类
- **Rectified S的design choice**: 当 $P_{pos} < P_{neg}$ 时强制 $S' = -|S|$,这个设计有点heuristic,可能有更principled的处理方式 (比如直接flip gradient sign)

---

## 9. 论文的真正贡献

让我用最简洁的方式总结这篇paper的价值:

**Conceptual contribution**: 提出一个新的model property概念"distributional clarity",作为RL-Friendliness的hidden driver。这个概念把三个看似无关的现象 — pass rate disparity、error severity、solution stability — 统一在一个framework下。

**Methodological contribution**: 把cluster analysis的Silhouette Coefficient移植到RLVR分析,提供了一个可计算、可解释、可干预的metric。Silhouette本身是1970s的classical metric,但用在LLM的probability distribution上确实新颖。

**Practical contribution**: Silhouette-Aware Reweighting虽然简单,但universal effective across三个model family。无需修改model architecture,无需额外data,只需在advantage计算时乘一个weight。

**Diagnostic contribution**: 三阶段分析框架 (phenomenon → mechanism → interpretation) 可以作为template分析其他RL failure modes。比如未来如果某个model family在code RL上不work,可以套用这个framework:先看pass rate disparity,再看probability distribution的Silhouette,再看error severity和solution stability。

---

## 10. 参考链接

- Paper本篇: Distributional Clarity (本paper)
- DAPO: https://arxiv.org/abs/2503.14476
- GRPO / DeepSeekMath: https://arxiv.org/abs/2402.03300
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- OctoThinker: https://arxiv.org/abs/2506.20512
- Cognitive Behaviors (Gandhi): https://arxiv.org/abs/2503.01307
- SimplerLZoo: https://arxiv.org/abs/2503.18892
- Spurious Rewards: https://arxiv.org/abs/2506.10947
- Midtraining bridges distributions: https://arxiv.org/abs/2510.14865
- Reasoning with Exploration (entropy): https://arxiv.org/abs/2506.14758
- Seed-GRPO (semantic entropy): https://arxiv.org/abs/2505.12346
- ICPO: https://arxiv.org/abs/2511.21005
- Negative Reinforcement: https://arxiv.org/abs/2506.01347
- Qwen2.5: https://qwenlm.github.io/blog/qwen2.5/
- Llama 3: https://arxiv.org/abs/2407.21783
- Murphy, Probabilistic ML (Silhouette Coefficient reference): https://probml.github.io/pml-book/book1.html
- Bishop, Pattern Recognition and ML (Fisher Ratio reference): https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/
- verl framework: https://arxiv.org/abs/2409.19056
- AIME 2024: https://artofproblemsolving.com/wiki/index.php/2024_AIME
- MATH dataset: https://arxiv.org/abs/2103.03874
- OlympiadBench: https://arxiv.org/abs/2406.12709
- Minerva: https://arxiv.org/abs/2206.14858

---

整体评价:这篇paper属于"用对metric看对现象"的典范。Silhouette Coefficient本身不是新东西,GRPO也不是新东西,但把两者结合揭示出一个之前被data-centric narrative遮盖的structural property,而且给出可操作的intervention。实验在三个model family上一致improvement,ablation充分,理论分析虽简但抓住了gradient variance这个关键点。Limitation部分诚实承认group sampling依赖和long CoT未覆盖,留出了后续工作空间。

对构建你的intuition最有帮助的一点:**把RL training看作一个"决策边界锐化"过程,而非"行为习得"过程**。模型base已经在某些query上有内部清晰度,RL只是把这种清晰度投影到probability空间并amplify。低清晰度的query,RL无法amplify一个不存在的东西 — 这时需要的是intervention (reweighting) 或重新pretraining,而不是更多RL steps。
