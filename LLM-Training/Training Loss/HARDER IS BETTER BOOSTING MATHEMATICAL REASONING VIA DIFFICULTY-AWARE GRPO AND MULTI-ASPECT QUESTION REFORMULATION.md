---
source_pdf: HARDER IS BETTER BOOSTING MATHEMATICAL REASONING VIA DIFFICULTY-AWARE
  GRPO AND MULTI-ASPECT QUESTION REFORMULATION.pdf
paper_sha256: 27a9fd4b5935e199c7c4d81758e5115134faceae263b8a29602dddd9d6cc5fa4
processed_at: '2026-08-04T23:28:29-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，我换个更口语的方式聊聊这篇paper在说什么。

---

## 一句话概括

训练reasoning模型的时候，大家都用GRPO，但GRPO有个隐藏的bug——它天然地更关注中等难度的问题，对那些**有点难但还能解出来**的问题反而给的学习信号很弱。这篇paper就是来修这个bug的，同时还搞了个数据增强方法，专门把题目改得更难来配合训练。

---

## GRPO到底哪出了问题

先回顾GRPO在干嘛。给一道题，让模型采样8个回答，对每个回答打分（对就1分错就0分），然后把分数归一化成advantage，做policy gradient。

归一化用的是standard deviation。这看起来很标准，但问题藏在数学里。

假设一道题模型答对概率是 $p$。那么8个回答里大概有 $8p$ 个对、$8(1-p)$ 个错。你算每个回答的advantage绝对值之和，结果是 $2G\sqrt{p(1-p)}$，其中 $G=8$。

这个函数长啥样？是个倒U形。$p=0.5$（一半对一半错）时最大，$p$ 接近0或1时趋于零。

翻译成人话：

- 模型已经几乎全做对的题 → gradient信号很弱 → 学不到啥
- 模型几乎全做错的题 → gradient信号也很弱 → 也学不到啥
- 模型一半对一半错的题 → gradient信号最强 → 学得最猛

乍一看好像挺合理？中等难度的题学习信号最强，没毛病啊。

**但问题在于**，你真正想让模型突破的，是那些它偶尔能做对、但大部分时候做错的题。比如 $p=0.2$ 的题——模型暴露了自己的weakness，而且至少有正确答案可以学。这些题恰恰是GRPO给gradient最小的。

这就是paper说的"implicit imbalance"。

---

## DGPO怎么修的

两步走，思路非常清晰。

### 第一步：把分母从std换成MAD

原来的advantage是：
$$\hat{A} = \frac{r_i - \text{mean}}{\text{std}}$$

改成：
$$\hat{A} = \frac{r_i - \text{mean}}{\text{MAD}}$$

MAD就是Mean Absolute Deviation，平均绝对偏差。这个改动看起来很小，但效果很妙。

你算新的total update magnitude，结果是 $G$，一个常数。跟 $p$ 完全无关了。

也就是说，不管一道题的accuracy是0.1还是0.9，只要它不是全对全错（即至少有一个对一个错），它对policy gradient的总贡献都是固定的 $G$。

这就把GRPO那个倒U形的bias给抹平了。每道valid题获得同等的"说话权"。

### 第二步：在平等基础上偏向难题

光抹平还不够。paper认为难题应该有更大的说话权，因为那是model最需要突破的地方。

具体做法是对batch里每道valid题算一个difficulty score $D_s = -\text{mean}(\{r_i\})$。全做对 $D=-1$（最容易），全做错 $D=0$（最难）。然后用一个temperature $T=2$ 的softmax做reweighting。

温度2挺温和的，batch里最难的题和最易的题权重比大概是1.65倍，不会太极端。

---

## 为什么要分两步

这是paper我觉得最聪明的设计哲学。

如果你直接在GRPO基础上做reweight（比如GRPO-AD就是这么干的），你实际上在同时对抗两个纠缠在一起的effect：原始的imbalance和你新加的reweight。你调一个hyperparameter，两个effect一起变，很难控制。

DGPO先balance（让所有题贡献相等），再reweight（在相等基础上偏向难题），两个step各自的目标清晰，互不干扰。消融实验也证实了——DGAE单独贡献+0.94%，DQW单独贡献+1.14%，两者叠加有+2.18%，基本是加和的。

---

## MQR：把题目改难

算法这边搞定了，数据这边呢？

现有的数据增强方法要么是生成全新的题目（answer质量难保证），要么是简单paraphrase（难度没变）。paper提出MQR，从三个维度把题目改难，但强制保持原始答案不变。

### 三个维度

**Background**：加一段看似相关但实际无用的故事背景。比如原本"一个蛋糕6欧元，汇率1.25"的题，给你加一段Montmartre区的历史、patisserie的故事。考验模型从narrative noise里extract数学本质的能力。

**Term**：发明一个抽象术语来重新定义问题里的核心概念。比如把"需要付的差额"叫做"euro-gap $\epsilon$"。考验模型理解abstract definition并mapping回具体含义的能力。

**Sub-Problem**：把题目里一个直接给定的数字，变成一个独立的小问题。比如汇率1.25不直接给你了，而是让你先解 $x+y=9, x^2+y^2=41$，算出 $r = \max/\min$，这个 $r$ 就是汇率。考验multi-step + cross-domain推理。这个维度最powerful，单独贡献+1.63%。

### 为什么保持答案不变是关键

因为这样就不需要重新生成solution。competition-level的数学题，让LLM重新解题很容易出错，answer质量没保证。但只改题面不改答案，reformulator只需要rephrase能力，不需要solve能力。所以连7B级别的open-source model都能当reformulator用。

而且就算reformulation搞砸了（答案变了），在RLVR的严格binary reward下，模型的回答几乎不可能巧合地匹配到原始答案，所以这些坏数据的reward全为0，自动成为invalid query被过滤掉。这是个很优雅的self-correction机制。

---

## 合在一起的效果

Qwen2.5-Math-7B上：

- GRPO baseline：37.61%
- 单用DGPO：39.79%（+2.18%）
- 单用MQR：41.04%（+3.43%）
- MathForge（两者合体）：42.17%（+4.56%）

数据那边提升空间比算法那边还大，这是有意思的observation。而且两者合体超过任何单独的，说明是synergistic loop：MQR造更难的题，DGPO聚焦学这些难题，相互促进。

跨model也work——1.5B、3B、7B、DeepSeek-Math-7B、Qwen2.5-VL-3B（多模态几何推理）全部都有提升。DGPO还能叠加到DAPO、GPG、GSPO上，作为general enhancement module。DAPO+DGPO甚至达到39.91%，比standalone DGPO还高。

---

## Training dynamics里有个有趣的细节

DGPO训练出来的模型output length比GRPO短。也就是说，模型不仅答得更准，而且reasoning path更精炼，少废话。

MQR那边则展现了"train harder, test better"——训练时accuracy更低（题更难），但测试时accuracy更高。典型的anti-overfitting信号，说明model学到的是更robust的reasoning capability，而不是training set的surface pattern。

---

## 总结一下核心insight

1. GRPO用std做normalization，在binary reward下天然产生倒U形的difficulty bias，压制了难题的学习信号
2. 把std换成MAD，这个bias直接消失，所有valid题获得平等的gradient贡献
3. 在平等基础上再用softmax reweight偏向难题，两步解耦，可控性强
4. 数据侧用LLM从三个正交维度把题改难但保持答案，廉价且有效
5. 两者形成synergistic loop，而且都model-agnostic、domain-agnostic

paper的title "Harder is Better"确实点到了本质——reasoning模型的训练，你给它喂更难的东西、并且让算法聚焦于更难的东西，它就学得更好。这个insight本身不复杂，但paper用数学证明了GRPO在哪一步背离了这个原则，然后用很clean的两步修正fix了它。

---

# MathForge：通过Difficulty-Aware GRPO和Multi-Aspect Question Reformulation增强数学推理

Andrej，这篇paper触及了一个非常深刻的RL training dynamics问题。我try从底层intuition开始拆解。

## 1. 核心问题的intuition

GRPO作为RLVR的de-facto standard，在DeepSeek-R1之后被广泛采用。其核心机制是：对同一个question $q$ 采样 $G$ 个response $\{o_i\}_{i=1}^G$，用group-relative的accuracy reward $\{r_i\}$ 来estimate advantage，从而avoid critic model。这种design的优雅之处在于critic-free，但paper揭示了一个**隐式的update magnitude imbalance**——简单的GRPO实际在训练中"偏好"中等难度的问题，而suppress了harder but still solvable的问题的gradient contribution。

这一点对reasoning训练非常关键：**harder yet solvable** problems才是最有价值的training signal，因为它们：
1. 暴露了model的underdeveloped capability
2. 至少有1个correct response可以作为targeted learning signal
3. Mastering harder problems往往能backward transfer到easier ones

## 2. The Implicit Imbalance of GRPO - 数学推导

### 2.1 GRPO的advantage function

GRPO的group-relative advantage estimation (GRAE):
$$\hat{A}_{\text{GR},i} = \frac{r_i - \text{mean}(\{r_i\}_{i=1}^G)}{\text{std}(\{r_i\}_{i=1}^G)}$$

其中：
- $r_i \in \{0, 1\}$：第 $i$ 个response的binary accuracy reward
- $\text{mean}(\cdot)$ 和 $\text{std}(\cdot)$：G个reward的均值和标准差
- 分子：当前reward相对group mean的偏离
- 分母：用std normalize，让advantage scale-invariant

### 2.2 单个question的total update magnitude

paper的Theorem 1给出了一个closed-form result。推导过程值得仔细看：

对一个query $q$，假设accuracy rate $p = \frac{1}{G}\sum_{i=1}^G r_i$（即G个response中有 $Gp$ 个对，$G(1-p)$ 个错）。那么：

$$\sum_{i=1}^G |\hat{A}_{\text{GR},i}| = \frac{\sum_{i=1}^G |r_i - p|}{\text{std}(\{r_i\})}$$

由于 $r_i$ 是binary的，$|r_i - p|$ 的sum可以拆分：
- 对 $Gp$ 个 $r_i = 1$ 的项：$|1 - p| = (1-p)$
- 对 $G(1-p)$ 个 $r_i = 0$ 的项：$|0 - p| = p$

而binary variable的 $\text{std} = \sqrt{p(1-p)}$，所以：

$$\sum_{i=1}^G |\hat{A}_{\text{GR},i}| = \frac{Gp(1-p) + G(1-p)p}{\sqrt{p(1-p)}} = \frac{2Gp(1-p)}{\sqrt{p(1-p)}} = 2G\sqrt{p(1-p)}$$

这是一个关键结果——总update magnitude与 $p$ 的关系是一个倒U形曲线，在 $p=0.5$ 时取最大值 $G$，在 $p \to 0$ 或 $p \to 1$ 时趋近于0。

### 2.3 为什么这是问题？

直觉上，这个曲线告诉我们：
- $p \approx 0.5$（中等难度）：update magnitude最大，~$G$
- $p \approx 1$（太容易）：update magnitude小，model已经会了
- $p \approx 0$（太难）：update magnitude也小，几乎没有正样本可学

但是**最理想的training material是 $p$ 稍小（比如0.2-0.4）的问题**——这些是model挣扎但仍能偶尔答对的boundary questions。GRPO的反U形曲线恰恰suppress了这部分最有价值的signal。

## 3. DGPO: Difficulty-Aware Group Policy Optimization

DGPO用两步走的方式解决：
- **Step 1: DGAE** - balance掉imbalance，让所有valid query的update magnitude归一
- **Step 2: DQW** - 在归一基础上进一步upweight harder questions

### 3.1 DGAE: Difficulty-Balanced Group Advantage Estimation

核心改动：把分母从 $\text{std}$ 换成 $\text{MAD}$（Mean Absolute Deviation）

$$\hat{A}_{\text{DG},i} = \frac{r_i - \text{mean}(\{r_i\}_{i=1}^G)}{\text{MAD}(\{r_i\}_{i=1}^G)}$$

其中：
$$\text{MAD}(\{r_i\}_{i=1}^G) = \frac{1}{G}\sum_{i=1}^G |r_i - \text{mean}(\{r_i\}_{i=1}^G)|$$

**Theorem 2**给出新的total update magnitude：

$$\sum_{i=1}^G |\hat{A}_{\text{DG},i}| = \frac{\sum_{i=1}^G |r_i - \text{mean}|}{\frac{1}{G}\sum_{i=1}^G |r_i - \text{mean}|} = G$$

这是一个**常数 $G$**，与 $p$ 无关！这意味着不论一个query是难还是易，只要它有非零的variance（即有正样本也有负样本），它对policy gradient的总贡献都是 $G$。

**intuition**: MAD和std在binary reward情况下关系是 $\text{std} = \frac{1}{\sqrt{p(1-p)}} \cdot \frac{\text{MAD}}{2}$，所以用MAD做normalization等价于把 $2\sqrt{p(1-p)}$ 这个dependency直接消掉了。在continuous reward情况下，Theorem 2也成立（不需要binary约束），这比Theorem 1更general。

### 3.2 DQW: Difficulty-Aware Question-Level Weighting

DGAE解决了"水平衡"，但是对所有valid query同等看待。DQW在此基础上进一步把harder questions的权重提上来。

对batch中第 $s$ 个valid query $q_s$，定义difficulty score：

$$D_s = -\text{mean}(\{r_{si}\}_{i=1}^G)$$

即负的平均accuracy reward。$D_s \in (-1, 0)$，越大表示越难（更接近0其实更易，但加负号后更负代表更难；wait，重新check：mean reward ∈ [0,1]，所以 $D_s = -\text{mean} \in [-1, 0]$，$D_s = -1$ 表示全对（最容易），$D_s = 0$ 表示全错（最难））。

然后用temperature-controlled softmax做weighting：

$$\lambda_s = B_v \cdot \frac{\exp(D_s / T)}{\sum_{s=1}^{B_v} \exp(D_s / T)}$$

其中：
- $B_v$：batch中valid query数量（即 $\{r_{si}\}$ 不全0也不全1的query）
- $T$：temperature，控制distribution sharpness
- 乘 $B_v$ 和归一化分母保证权重均值是1

**为什么需要valid query的概念？** 如果一个query的所有G个response要么全对要么全错，那 $\text{std} = \text{MAD} = 0$，advantage会undefined，gradient是0。这种query被排除掉，使用token-level loss averaging over valid queries，对应paper里的"valid token-level loss averaging"。

### 3.3 完整的DGPO objective

$$\mathcal{J}_{\text{DGPO}}(\theta) = \mathbb{E}\left[\frac{1}{\sum_{s=1}^{B_v}\sum_{i=1}^G |o_{si}|}\sum_{s=1}^{B_v} \lambda_s \sum_{i=1}^G \sum_{t=1}^{|o_{si}|} \min\left[I_{sit}(\theta)\hat{A}_{\text{DG},si}, \text{clip}(I_{sit}(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_{\text{DG},si}\right]\right]$$

其中 $I_{sit}(\theta) = \frac{\pi_\theta(o_{si,t} | q_s, o_{si,<t})}{\pi_{\theta_{\text{old}}}(o_{si,t} | q_s, o_{si,<t})}$ 是token-level importance sampling ratio。

这里值得注意的design choice：
- 去掉了KL divergence penalty（跟随GPG和DAPO的发现）
- token-level averaging over valid queries（避免gradient magnitude的不稳定）
- 仍保留PPO的clipping机制

### 3.4 Temperature $T$ 的意义

$T$ 控制weighting distribution的"尖锐度"：
- $T \to 0$：分布极尖锐，几乎把所有weight都给最难的query，risk of overfitting到极端hard case
- $T \to \infty$：分布退化为uniform $\lambda_s = 1$，DQW失效

由于 $D_s \in (-1, 0)$，$T = 2.0$时，最大/最小权重比：
$$\frac{\exp(0/T)}{\exp(-1/T)} = e^{1/T} = e^{1/2} \approx 1.65$$

这是个比较温和的reweighting。Ablation显示 $T = 2.0$ 最好（39.79%），$T = 1.0$ 下降到39.03%，$T = 10.0$ 下降到39.27%。

### 3.5 "Balance-then-Reweight"的两步design

paper特别强调这种两步design的interpretability：
- DGAE首先normalize所有query的update magnitude到 $G$（balance）
- DQW在归一基础上reweight（reweight）

这种separation of concerns的好处是：如果只做reweight（如GRPO-AD），那么imbalance和reweight两个effect会纠缠在一起，tuning困难。先balance再reweight，每个component有清晰的target，hyperparameter sensitivity降低。

## 4. MQR: Multi-Aspect Question Reformulation

### 4.1 三个reformulation aspects

MQR用LLM（默认OpenAI o3）从三个独立维度reformulate原始question：

**Aspect 1: Background（添加故事背景）**
> "Add a story background that is not related to the core mathematical content of the given question, but seems to be related to the question."

这个aspect考验model从noise中识别critical mathematical information的能力。背景看似相关但实际无数学意义，model必须学会区分essential math structure vs decorative narrative。

**Aspect 2: Term（发明抽象术语）**
> "Invent a new, abstract mathematical term to define a concept that is central to the given question, and restate the entire question using this term."

例如原paper示例中，把"需要付的欧元差额"重命名为"euro-gap $\epsilon$"，要求model能理解新引入的abstract定义并mapping回原始数学概念。这考验**abstract concept comprehension**。

**Aspect 3: Sub-Problem（嵌入子问题）**
> "Convert a key numerical condition of the given question which have a definite value into an independent sub-problem. The sub-problem may belong to any branch of mathematics."

paper示例中，原本直接给出的汇率1 euro = 1.25 USD被替换为：先解一个数论/代数问题 $x + y = 9, x^2 + y^2 = 41$，求 $r = \max(x,y)/\min(x,y)$，这个 $r$ 就是汇率。

这个aspect最powerful，因为它：
1. 增加了reasoning chain的长度
2. 引入cross-domain knowledge的要求（一个算术题可能前置需要数论）
3. 改变了original question的representation

### 4.2 关键约束：保持gold answer不变

MQR的核心约束是所有reformulation必须preserve原始gold answer。这样做的好处：
- 不需要重新生成solution（避免answer quality问题，特别是competition-level题）
- 保留原始的数学logical structure
- 把reformulator的能力需求降到最低——只需要reformulate，不需要solve

### 4.3 MQR的quality assessment

paper用OpenAI o3做automated verification，prompt要求判断rewritten question是否能yield same final answer as original：
- Background: 99% equivalence
- Term: 97% equivalence  
- Sub-Problem: 97% equivalence

且如果reformulation失败导致问题unsolvable或answer不同，由于RLVR的binary reward匹配要求严格，policy model的response几乎不可能巧合地match到原始answer，所以这些corrupted question的reward会全为0，成为invalid query，不会贡献有害gradient。这是MQR和RLVR结合的一个优雅self-correction机制。

### 4.4 Cost

22500个reformulated questions，平均每个question消耗：
- Input: 255.05 tokens
- Output reasoning: 820.27 tokens  
- Output reformulated question: 138.33 tokens

总成本约$184（OpenAI o3），相当便宜。

## 5. Experiments分析

### 5.1 主实验结果（Qwen2.5-Math-7B on MATH）

| Method | AIME24 | AIME25 | AMC23 | MATH500 | Minerva | Olympiad | Avg. | ΔGRPO |
|--------|--------|--------|-------|---------|---------|----------|------|-------|
| Base | 12.19 | 4.79 | 35.23 | 48.60 | 15.07 | 16.33 | 22.04 | - |
| GRPO | 20.94 | 8.44 | 58.98 | 72.20 | 27.76 | 37.33 | 37.61 | - |
| Dr.GRPO | 21.04 | 8.23 | 58.59 | 72.05 | 28.58 | 35.89 | 37.40 | -0.21 |
| GPG | 21.98 | 9.06 | 59.61 | 72.05 | 27.21 | 37.67 | 37.93 | +0.32 |
| DAPO | 21.25 | 8.75 | 58.20 | 72.70 | 29.50 | 37.22 | 37.94 | +0.33 |
| GSPO | 19.38 | 8.33 | 60.16 | 73.00 | 28.12 | 37.26 | 37.71 | +0.10 |
| GRPO-AD | 21.56 | 9.48 | 59.06 | 73.25 | 29.14 | 37.07 | 38.26 | +0.65 |
| **DGPO** | **23.85** | **10.21** | **61.02** | **74.25** | **31.07** | **38.33** | **39.79** | **+2.18** |
| **MQR** | **25.00** | **11.77** | 59.38 | **77.85** | 31.43 | **40.81** | **41.04** | **+3.43** |
| **MathForge** | 24.58 | **12.60** | 59.84 | **79.95** | **33.36** | **42.67** | **42.17** | **+4.56** |

观察：
1. DGPO单用就比所有strong baselines（DAPO、GPG、GSPO、GRPO-AD）都好，验证了balance-then-reweight的有效性
2. MQR比DGPO还强（+3.43 vs +2.18），说明data side的改进空间比algorithm side还大
3. MathForge（两者结合）达到+4.56%，体现synergistic loop——MQR扩展data frontier，DGPO有效学习augmented data

特别值得注意的是AIME25和MATH500上的提升最显著（MathForge在MATH500上79.95% vs GRPO 72.20%，绝对提升7.75%）。MATH500是分布内evaluation（与MATH training set同分布），说明MQR显著扩大了distribution内的coverage；而AIME25是极难的新benchmark，说明generalization也得到提升。

### 5.2 Ablation of DGPO components

| Method | Avg. | ΔGRPO |
|--------|------|-------|
| GRPO | 37.61 | - |
| DGPO (w/o DGAE & DQW) | 37.71 | +0.10 |
| DGPO (w/o DQW) | 38.65 | +1.04 |
| DGPO (full) | **39.79** | +2.18 |

- valid token-level loss averaging（即GPG的baseline改动）：+0.10%
- +DGAE（balance）：+0.94%
- +DQW（reweight）：+1.14%

DGAE和DQW贡献相当，说明balance和reweight都是必要的。

### 5.3 MQR的ablation

| Data | Avg. | ΔOri. |
|------|------|-------|
| Original | 39.90 | - |
| MetaMath-Rephrasing | 40.73 | +0.83 |
| +Background | 40.95 | +1.05 |
| +Term | 41.24 | +1.34 |
| +Sub-Problem | 41.53 | +1.63 |
| MQR (all three) | **42.17** | **+2.27** |

观察：
1. 简单的MetaMath rephrasing（用GPT-3.5 paraphrase）只有+0.83%，说明rephrasing alone不够
2. 三个aspect单独加都有效，但Sub-Problem最强（+1.63%），因为它是真正增加了推理complexity
3. 三个aspect合在一起有synergistic effect（+2.27 > 任何单个aspect）

### 5.4 Reformulator generality

| Reformulator | Avg. | ΔOri. |
|--------------|------|-------|
| Original | 39.90 | - |
| Qwen2.5-7B-Instruct | 41.09 | +1.19 |
| Qwen3-30B-A3B-Thinking | 41.85 | +1.95 |
| OpenAI o3 | **42.17** | **+2.27** |

即使是7B级别的open-source reformulator也有+1.19%的提升。这印证了MQR对reformulator要求低——只需要reformulate而不需要solve。

### 5.5 Training dynamics的insight

paper的Figure 1显示DGPO训练中model的output length比GRPO短，说明DGPO不仅提升accuracy还鼓励更concise的reasoning path。这很有趣——harder questions被upweight后，model学到更direct的解决路径，trim掉冗余step。

Figure 2的MQR training dynamics展示了"train harder, test better"现象：
- MQR-augmented data上training accuracy更低（题目更难）
- 但在unseen MATH500上evaluation accuracy更高

这种generalization gap的反转说明MQR避免overfitting到training set的surface pattern，强迫model学到更robust的reasoning capability。

### 5.6 Cross-model generalization

| Model | GRPO | DGPO | MQR | MathForge |
|-------|------|------|-----|-----------|
| Qwen2.5-Math-1.5B | 29.39 | 30.71 (+1.32) | 32.44 (+3.05) | **33.84** (+4.45) |
| Qwen2.5-3B | 25.47 | 27.19 (+1.72) | 27.72 (+2.25) | **29.01** (+3.54) |
| DeepSeek-Math-7B | 14.91 | 16.53 (+1.62) | 16.78 (+1.87) | **17.77** (+2.86) |

所有size和family的model都受益，证明MathForge是model-agnostic的principle。注意小model（1.5B）提升反而最显著（+4.45%），可能是因为小model对training data quality更敏感。

### 5.7 Multimodal domain

在GeoQA-8k上训练Qwen2.5-VL-3B-Instruct：

| Method | GeoQA | ΔGRPO |
|--------|-------|-------|
| Base | 39.79 | - |
| GRPO | 57.43 | - |
| Dr.GRPO | 57.96 | +0.53 |
| GPG | 59.02 | +1.59 |
| DAPO | 59.02 | +1.59 |
| GSPO | 57.16 | -0.27 |
| GRPO-AD | 58.09 | +0.66 |
| **DGPO** | **59.95** | **+2.52** |

DGPO在多模态几何推理上也最好，证明principle的domain-agnosticism——只要能定义quantifiable difficulty measure（如accuracy rate），DGPO就适用。

### 5.8 与其他policy optimization方法的compatibility

| Method | Base | +DGPO |
|--------|------|-------|
| GPG | 37.93 | 38.92 (+0.99) |
| DAPO | 37.94 | **39.91** (+1.97) |
| GSPO | 37.71 | 39.32 (+1.61) |

DGPO可以作为enhancement module叠加到其他方法上，特别是**DAPO+DGPO达到了39.91%，比standalone DGPO还高（39.79%）**。这暗示DGPO解决的是fundamental的imbalance issue，与其他方法的mechanical improvements（如DAPO的length penalty, dynamic sampling）正交。

## 6. 整体Intuition总结

把所有insights连起来：

1. **GRPO的隐式imbalance源于std normalization在binary reward上的特殊性**——$2\sqrt{p(1-p)}$ 这个因子让中等难度问题被过度强调，harder problems被suppress。

2. **MAD normalization的elegance**——MAD天然消除了 $p$ 的dependency，让所有valid query的update magnitude归一到 $G$。这等价于对GRPO的advantage做了一个 $1/(2\sqrt{p(1-p)})$ 的implicit reweighting。

3. **两步走的design philosophy**——balance和reweight解耦，让每个component的effect可解释、可控制。GRPO-AD直接reweight时imbalance和reweight两个effect纠缠，所以效果差（+0.65%）。

4. **MQR的synergistic aspects**——Background考验noise filtering, Term考验abstract understanding, Sub-Problem考验multi-step + cross-domain reasoning。三种cognitive challenge正交，combine产生synergy。

5. **"Train harder, test better"**——这是core insight。MQR扩大了data的frontier，DGPO在augmented data上集中学习，整个MathForge形成self-reinforcing loop：data更难 → DGPO更聚焦难case → model capability boundary被push outward → 在original distribution上更稳健。

6. **RLVR + answer-preserving augmentation的self-correction**——MQR的failure case（answer变了）会自然成为invalid query被filter掉，因为RLVR的严格reward matching几乎不可能巧合match。这是个优雅的robustness mechanism。

## 7. References

- GRPO原始paper (DeepSeek-Math): https://arxiv.org/abs/2402.03300
- DeepSeek-R1 (RLVR的代表): https://arxiv.org/abs/2501.12948
- Open-R1 (paper使用的codebase): https://github.com/huggingface/open-r1
- DAPO: https://arxiv.org/abs/2503.14476
- GPG: https://arxiv.org/abs/2504.02546
- GSPO: https://arxiv.org/abs/2507.18071
- GRPO-LEAD (GRPO-AD的来源): https://arxiv.org/abs/2504.09696
- Dr.GRPO: https://arxiv.org/abs/2503.20783
- MetaMath: https://arxiv.org/abs/2310.05506
- Qwen2.5-Math: https://arxiv.org/abs/2409.12122
- MATH dataset: https://arxiv.org/abs/2103.03874
- PPO (基础): https://arxiv.org/abs/1707.06347
- TRL library: https://github.com/huggingface/trl
- OlympiadBench: https://arxiv.org/abs/2402.14008
- NuminaMath-CoT: https://huggingface.co/datasets/AI-MO/NuminaMath-CoT
- GeoQA: https://arxiv.org/abs/2105.14517

## 8. 一些可能延伸的direction

如果你想继续探索这条线：

1. **DGAE的continuous reward generalization**——Theorem 2已经证明对continuous reward也成立，那么可以尝试把DGAE用到process reward model的场景，或者format reward + accuracy reward的混合场景。

2. **DQW的dynamic difficulty**——目前 $D_s$ 是当前batch计算的instantaneous difficulty，可以考虑用exponential moving average跟踪historical difficulty，平滑训练初期的高方差。

3. **MQR的自动aspect discovery**——三个aspect是手工设计的，可以用LLM自动从mathematical reasoning的cognitive dimensions中induce更多aspect，比如"增加constraint的维度"、"引入counterfactual reasoning"等。

4. **Difficulty curriculum**——DGPO的reweighting是batch内relative的，可以扩展到cross-batch的curriculum learning，让training trajectory从easier questions逐步过渡到harder ones。

5. **MQR + Self-Play**——结合Liang et al. (2025)的self-play思路，让policy model自己generate harder questions（基于自己的failure mode），形成更tight的self-improvement loop。

希望这个分析对build你的intuition有帮助。这篇paper的核心message其实非常simple and elegant：在RLVR training中，harder problems被systematically underweighted，fix这个imbalance并且explicitly emphasize harder ones，能带来显著的reasoning capability提升。"Harder is Better"不仅是slogan，还有mathematical foundation。
