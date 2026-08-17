---
source_pdf: From AGI to ASI.pdf
paper_sha256: 668469948e8dfeae61215ddc4f95f68dd3a5496380fdf5798fabdc3a92150985
processed_at: '2026-08-04T10:38:24-07:00'
target_folder: AI生态
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

好，我之前那个版本太 technical 了，这次咱们坐下来聊天式地说。

---

## 这篇 paper 到底在干嘛

一句话：**假设我们真的搞出了 AGI，然后呢？AI 会停下来吗？还是会继续冲向 ASI？如果冲，走哪条路，什么会拦住它？**

这个问题五年前问还显得很科幻，但现在 LLM scaling 这么猛，大家开始觉得 AGI 可能真的不是那么远了。所以 DeepMind 觉得需要严肃地 map out 一下这个 landscape，而不是继续 hand-wavy 地讨论。

他们不是在预测未来——他们是在说"这些是可能的路径，这些是可能的障碍，哪些是真正 open 的 research question"。

---

## 先说他们怎么定义 AGI 和 ASI

这个很重要，因为大部分人讨论这些概念的时候根本没想清楚自己在说什么。

**AGI**：就是跟一个普通人类差不多聪明的系统。median human level on most cognitive tasks。注意是"普通人类"，不是 Einstein。今天的大模型在很多 task 上已经 superhuman 了，但还不够 general，所以还不是 AGI。

**ASI**：这里他们把 bar 设得挺高——不是比单个人聪明，是比**一大群人类专家组成的团队**还厉害。具体说：几万个 well-coordinated 的 expert humans，用 2010 年的技术水平，花 10 年时间干一件事——ASI 要比这个还强。

为什么这么定义？因为如果你只说"比单个人聪明"那叫 superhuman 不叫 superintelligence。真正的 ASI 应该能 outperform 整个 research field 或 large corporation。

**Universal AI (UAI)**：这是理论上的极限，数学上由 AIXI 定义。不可计算，只能逼近。它是 intelligence 的"光速"——你永远到不了但知道它在那。

---

## 为什么要扯 AIXI 和 Legg-Hutter

这是我觉得这篇 paper 最聪明的地方。

大部分讨论 AGI/ASI 的人都在用很 fuzzy 的语言——"更聪明"、"更能干"之类的。但这篇 paper 说：我们其实有一个数学上 well-defined 的 intelligence measure，叫 Legg-Hutter score。

直觉上就是：**把一个 agent 扔进所有可能的世界里（按世界本身的复杂度加权），看它平均表现怎么样**。简单的世界权重更大（因为如果连简单世界都搞不定，复杂世界免谈），所有 computable 的世界都算上。

这就给了一个 continuum——intelligence 不是 binary 的"有/没有"，是一个连续的 score。AGI 和 ASI 就在这个 continuum 上的不同位置。你不需要非常精确地划线，只要有 significant gap 就行。

AIXI 是这个 score 的数学最大值。它做的事情概念上很简单：

1. 考虑所有可能的环境（所有 computable 的世界）
2. 用 Bayesian 方式根据 observations 更新对每个世界的 belief
3. 选能最大化 expected future reward 的 action

问题是这东西不可计算——因为 Kolmogorov complexity 不可计算。但它给了一个理论 anchor。

---

## 最重要的 theoretical insight：LLM 训练就是在做 universal compression

这个 insight 我（作为 Karpathy）一直觉得非常重要，但很多人没 fully appreciate。

你训练一个 LLM 做 next-token prediction，最小化 cross-entropy loss。数学上这等价于**最小化 expected code length**——你在做 data compression。

Solomonoff induction 是这个的 universal 极限：用所有可能的程序去解释 data，更简单的程序权重更大。这是数学上最优的 predictor，但不可计算。

**LLM 就是用一个高度受限的 function class（transformer with parameters θ）去近似这个 universal compressor**。

所以当你 scale up 一个 LLM——更多参数、更多 data、更多 compute——你实际上是在做一个越来越好的 bounded-compute approximation to universal compression。而 universal compression 是 universal intelligence 的核心 component。

这就给了一个 theoretical justification：**current paradigm 可能比看起来更有 runway**，因为它不是任意的 engineering hack，它是在逼近一个数学上 well-defined 的极限。

AIXI 的 recipe 是 universal predictor + planning scaffolding。LLM 是 universal predictor 的 approximation。加上 chain-of-thought、tool use、agent scaffolding、RL fine-tuning——你就在向 AIXI 的结构靠拢。

---

## 好了，四条路径

### 路径一：继续 scale

就是继续做我们现在在做的事情——更大的模型、更多的 data、更多的 compute。

这个路径有个数学对象可以 fit：scaling law。简单说就是 loss 跟 compute/parameters/data 的 power law 关系。

但 key question 是：**scale 能不能从 AGI 到 ASI？**

论文的 answer 是 nuanced 的：

- **理论上**：能。因为 AIXI 的 approximations 加更多 compute 就趋向 universal intelligence。
- **实际上**：naive brute-force 的 compute 需求增长太快。你需要好的 inductive bias 让 search tractable，但这些 bias 会引入 limitations。
- **Escape hatch**：即使 individual model 卡在 AGI，你可以跑 millions of instances。如果这些 instances 能形成 effective collectives，collective intelligence 可能突破 individual limits。

想想人类社会——单个人类大脑几万年没怎么变，但人类文明的 collective intelligence 突飞猛进。AI collectives 可能更容易做到，因为 communication bandwidth 几乎无限，instances 可以无损复制。

### 路径二：算法 paradigm shift

就是现在的 transformer + pretraining + RLHF 这套可能不够，需要根本性的改变。

论文区分了 evolution 和 shift：

**Evolution**（在现有框架内加东西）：
- Mamba 这种 linear-time sequence model，把 attention 的 $O(L^2)$ 变成 $O(L)$
- Retrieval-augmented generation，让 context 几乎无限
- World model，让模型能 simulate 未来、做 long-horizon planning
- Continual learning，不 catastrophic forgetting
- Better agentic scaffolding

**Shift**（根本性改变）：
- Spiking neurons / neuromorphic hardware
- RL-based pretraining 替代 log-loss pretraining
- 完全不同的 architecture

Shift 不可预测，所以论文说这条路径不太适合 forecasting。但如果你相信 current paradigm 会 hit ceiling，shift 就是必要的。

### 路径三：递归自我改进

这是最可能产生 "intelligence explosion" 的路径。

核心 idea：如果 AI 能加速 AI research，那更好的 AI 又能更快地加速 AI research，形成 positive feedback loop。

论文分了四种 self-improvement：

1. **Code level**：AI 写更好的 architecture/optimizer
2. **Data level**：AI curate/generate 更好的 training data
3. **Hardware level**：AI 设计更好的 chips
4. **Organization level**：specialization 让 collective 更高效

**AlphaZero 是最经典的 example**：policy network 和 value network 作为 prior 驱动 MCTS search，search 的结果再蒸馏回网络，网络变好了 search 也变好了，循环往复。

这个 recipe 直接 translate 到 LLM：base model → test-time search (chain-of-thought) → improved outputs → distill back → better base model → ...

**但论文也指出了风险**：naive 的 self-generated data training 会导致 model collapse（Shumailov et al. 2024 的结果）。关键是你要有 **search-based filtering**——用一个 ground-truth signal 来 filter 掉 bad generations。AlphaZero 有赢/输作为 signal，LLM 的 case 里这个 signal 可能是 verifier、是 real-world interaction feedback、是 mathematical proof checker。

### 路径四：多 agent collective

就是 ASI 不一定是单个 super-genius，而是一大堆 AGI instances 协调工作，collective intelligence 超过任何个体。

人类社会的 analogy 很直观：一个公司比任何单个员工聪明，一个 research field 比任何单个 researcher 聪明。这是通过 parallelization + specialization + coordination 实现的。

AI collectives 的优势：
- Communication bandwidth 几乎无限（人类语言是 low-bandwidth bottleneck）
- 可以无损复制 instances
- 可以 rapid scale（spawning 新 instances 只需要 compute）
- 可以 high-bandwidth 共享 learning signal（甚至直接共享 gradients）

论文说这条路径有个关键 open question：**multi-agent scaling law 是什么？** 即 group intelligence 怎么随 instance count 和 compute 增长？是 linear、super-linear 还是 sub-linear？如果 super-linear，scaling instances 比 scaling individual model 更 efficient。

---

## 六个可能拦住你的 bottleneck

### 1. Data Wall

高质量 text data 可能在 2026-2028 年用完。但论文说有几个 counter-mechanism：

- **Synthetic data with filtering**：不是 naive self-distillation，是用 search/verifier filter 高质量 generations
- **Simulation**：视频生成模型作为 world simulator，在 simulation 里 generate interaction data
- **RL data**：agent 在 environment 中交互收集的 data，scaling 跟 compute 走

我的 view：data wall 主要是 about human-generated text。RL/agent 路径的训练 data 是 generated from interaction，没有天然上限。这是为什么 RL 会越来越 central。

### 2. Economic / Resource constraints

Scale compute 要钱、要电、要 chips、要 rare earths。这个能不能 sustain 是个 real question。

但 counter-argument：如果 AI 开始 generate economic return，那个 return 可以 reinvest 到更多 compute。这是个 feedback loop，问题是它 positive 还是 negative。

### 3. Current paradigm 跑不动了

Transformer + pretraining 可能不够。但论文指出现在的 paradigm 已经远远超过 "just pretraining"——有 post-training pipeline、test-time scaling、retrieval、tool use、agentic scaffolding。

所以更可能的 scenario 是**smooth evolution** 而非 sharp shift——持续加 component，偶尔有重要 breakthrough。

### 4. Research gets harder

Bloom et al. 2020 的发现：维持 Moore's law 需要 18× 更多的 researchers 相比 1970s。"low-hanging fruit" 被摘完了。

但论文的 counter-argument 很 sharp：如果 AI 能做 research，scaling AI researchers = scaling compute。10×/year 的 compute growth 意味着 1.3 年就能 18× 你的 "researcher" count。这是人类做不到的。

**但**：AI researchers 也要 run experiments，experiments 有 real-world latency。你不能 accelerate 生物实验或材料科学实验的物理速度。

### 5. Abstraction Barrier（这个最有意思）

Lerchner 的 hypothesis：**当前 models trained on human cognitive products 可能 bounded by existing conceptual frameworks**。

直觉 example：用 pre-Newtonian 时代的所有 text 训练一个 modern-scale model，它能 reason 出 general relativity 吗？极不可能——它 lacks calculus、universal gravitation、electromagnetism 这些 conceptual primitives。

这说明 today's models 的 "superhuman" 表现可能主要来自 **speed + memory within human-defined conceptual boundaries**，不是真正的 transformative creativity。

如果这个 barrier 是 real 的，那 ASI 需要 **grounded concept discovery**——从 raw sensor data 中独立发现 new concepts。这引入 Embodied Bottleneck：concept validation 受物理实验速度限制。

这个 hypothesis 我觉得 underexplored 且 very important。今天所有 LLM 的 "creativity" 都是在 human conceptual space 内的 recombination。AlphaGo 的 Move 37 是 exploratory creativity（在已知 game 的已知 move space 内 find new move），不是 transformative creativity（发明全新的 game）。

### 6. Deliberate slowdown / Governance

监管、accident、societal backlash 可能人为减速。

但论文指出 unilateral regulation 容易产生 regulatory arbitrage——development 迁移到 oversight 弱的国家。"Anarchy as Architect" framework 说国际竞争 systematically favors competitiveness-enhancing technologies 的 adoption。所以 global coordination 很难但很重要。

---

## 几个我觉得最值得 think about 的点

### Test-time compute 是未来 5 年最重要的 variable

Pretraining compute 已经很大了，再 scale 100× 经济上日益困难。但 test-time compute 还在非常早期。o1/o3 类的 reasoning 模型才刚开始。

如果你能把 test-time improvements 有效蒸馏回 base model（AlphaZero recipe），你就有了一个 flywheel。这可能是从 AGI 到 ASI 最 concrete 的路径。

### ASI 的 benchmarking 是个 huge problem

一旦超过人类水平，你怎么 measure？你没法找 human expert 来打分。

候选方案：
- Multi-agent zero-sum competition（像 chess engine 互打）
- AI 自己出题考 AI（setter-solver）
- Compression benchmarks（theory-motivated）
- Indirect measurement（economic productivity）

这个 infrastructure 现在还很缺，需要提前 build。

### ASI 不是 omnipotent

论文反复强调 ASI 受 fundamental limits 约束——光速、Landauer principle、Bremermann's limit、complexity theory、Gödel incompleteness。

更重要的是：**很多 theoretical limits 是 worst-case bounds，实践中 approximation/heuristic 可能做得很好**。所以光知道 theoretical limit 不够，你还需要 empirical approach。

反过来说，预测 ASI 能不能 "cure aging" 或 "build Dyson sphere" 是 very hard 的——这些问题可能是 computationally irreducible 的，只有 actually try 了才知道。

### Collective intelligence vs. individual intelligence

我觉得这是论文里最 undervalued 的路径。大家直觉里想的 ASI 是一个 super-genius 单体。但更可能的 scenario 是 millions of AGI instances 组成的 collective——像一个超级高效的公司或 research field，但 communication bandwidth 和 coordination 能力远超人类组织。

人类文明的 intelligence 主要是 collective 的——单个人类大脑跟几万年前差不多，但文明能力突飞猛进。AI collectives 可能更容易 achieve this，因为 bandwidth 不是 bottleneck。

---

## 我的总体看法

这篇 paper 做了一件很重要的事：**它拒绝 binary thinking**。

不是 "ASI 必然到来" 或 "ASI 是科幻"。而是说：这里有几条路径，每条都有不确定性，每个 bottleneck 都是个 research question。我们需要 quantitative forecasting models、更好的 benchmarks、更深入的理论理解。

它用 AIXI/Legg-Hutter 作为 theoretical anchor 是聪明的——这给了讨论一个 formal grounding，避免了概念上的 hand-waving。

它对 four pathways 的 taxonomy 是 clean 的——scaling、paradigm shift、recursive improvement、multi-agent collective。这些不 mutually exclusive，可能同时发生。

它对 bottlenecks 的讨论是 balanced 的——每个 bottleneck 都列了 counter-mechanism，没有断言哪个一定是 hard blocker。

**它最大的 contribution 可能是 Section 7.1 的 open research questions list**——这是一个真正的 research agenda，如果有人想 serious 地 work on post-AGI AI progress，这是一个很好的 starting point。

作为 DeepMind 出品的 position paper，它也 implicitly 传达了一个信号：**前沿 lab 内部已经在 serious 地 think about post-AGI world 了**。这本身就是一个值得注意的 signal。

---

# From AGI to ASI — 一篇 DeepMind 的 Position Paper 解读

这篇 paper 是 Google DeepMind 团队（包括 Shane Legg、Marcus Hutter、Tim Genewein 等人）写的一份相当长的 position/landscape-mapping report，试图刻画 AGI 之后向 ASI (Artificial Superintelligence) 演化的技术路径以及潜在 frictions。让我从几个层面来深入讲解，尤其是把里面涉及的理论部分（Universal AI / AIXI）和工程直觉连接起来，因为这部分是大家平时不太接触但又非常重要的。

---

## 1. 论文的 Framing 和 Motivation

论文的核心 question 很直接：**假设我们达到了 human-level AGI，AI progress 会如何继续？会不会自然滑入 ASI territory？** 这个问题在过去多半是哲学/speculative 的，但随着 LLM scaling 的进展，已经变成需要严肃 quantitative 建模的问题。

论文的几个核心 framing 值得注意：

- **Effective compute 的复合增长率 ≈ 10×/year**：这是 Epoch AI 估算的下界，由三个 factor 相乘得到：
  - Hardware (Moore's law 及相关): ~1.5×/year
  - Investment growth in compute: ~2.5×/year  
  - Algorithmic efficiency: ~3×–6×/year（Ho et al. 2024/2025 的测量结果）
  
  即 $\text{Effective Compute Growth} \approx 1.5 \times 2.5 \times 3 = 11.25 \approx 10\times$。如果算法效率按 Ho et al. (2025) 的 ~6×/year 计算，总增长率可能更高。

- **Hyperbolic growth vs. exponential growth 的区分**：exponential 是 $\dot{y}/y = c$（常数增长率），hyperbolic 是 $\dot{y}/y = c \cdot y^{p-1}$（增长率随 $y$ 增长），数学上会在有限时间内 blow up。AI R&D 被自己自动化是产生 hyperbolic dynamics 的最直接来源。论文没有断言这必然发生，但指出无法排除。

参考链接：
- Epoch AI trends: https://epoch.ai/trends
- Aschenbrenner "Situational Awareness": https://situational-awareness.ai/
- Ho et al. "Algorithmic progress in language models" (NeurIPS 2024): https://arxiv.org/abs/2403.05917
- MacAskill & Moorhouse "Preparing for the intelligence explosion": https://arxiv.org/abs/2506.14863

---

## 2. 核心定义：AGI, ASI, UAI/Legg-Hutter Score

论文刻意使用 **informal characterization** 而非 sharp definition，但用 Legg-Hutter score 作为 formal grounding。这点很关键——大部分 AGI/ASI 讨论都在 hand-wavy 的定义里打转，论文用 Legg-Hutter score 提供了一个连续的、有数学基础的能力谱系。

### Legg-Hutter Intelligence Score

形式化定义（来自 Legg & Hutter 2007a）：

$$\Upsilon(\pi) := \sum_{\mu \in \mathcal{E}} 2^{-K(\mu)} V_{\mu}^{\pi}$$

变量含义：
- $\Upsilon(\pi)$: policy $\pi$ 的 universal intelligence score
- $\pi$: 一个 policy（agent 的行为策略）
- $\mathcal{E}$: 所有 computable environments 的集合（每个 environment 是一个 computable 的 probability measure over interaction histories，外加一个 reward function）
- $K(\mu)$: environment $\mu$ 的 **Kolmogorov complexity**，即能生成 $\mu$ 的最短程序的长度
- $2^{-K(\mu)}$: 这是 Solomonoff Universal Prior 的权重——更简单的 environment 在先验上更有可能（这是 algorithmic information theory 的核心 insight，并非任意假设）
- $V_{\mu}^{\pi}$: policy $\pi$ 在 environment $\mu$ 下的 **expected sum of discounted rewards**（lifetime value）

下标/上标说明：
- 上标 $\pi$ 在 $V$ 上表示这是属于 policy $\pi$ 的 value
- 下标 $\mu$ 在 $V$ 上表示这是在 environment $\mu$ 下评估的

直觉：intelligence = 在所有可能的世界（按其复杂度加权）中的平均表现。**简单的世界权重更大**，因为如果连简单世界都做不好，谈复杂世界没意义；同时所有 computable 世界都进入考虑，所以这是个真正"general"的 measure。

论文的 informal 锚定：
- **AGI**: Legg-Hutter score 大致等于 median human
- **ASI**: 显著超过 large human-expert collectives（论文里给了一个直观锚——"几万个 well-coordinated expert humans 在 10 年内 + 2010 年的技术水平能完成的任务"）
- **UAI (Universal AI)**: Legg-Hutter score 的最大值，由 AIXI 达到，是数学上的上界

注意论文明确说 AIXI 是 **incomputable**（不可计算），因为 Kolmogorov complexity 本身不可计算。所以 UAI 是理论上界，只能从 below 近似。

参考链接：
- Legg & Hutter 2007a "Universal Intelligence": https://arxiv.org/abs/0712.3329
- Hutter "Universal Artificial Intelligence" (2005 book): http://www.hutter1.net/ai/uaibook.htm
- Hutter, Quarel, Catt 2024 "Introduction to UAI" textbook: http://www.hutter1.net/ai/uaibook2.htm

---

## 3. Universal AI / AIXI 技术深度

这部分是论文 Section 4 的核心，也是我觉得最值得展开讲的——因为大多数 ML practitioner 从来没认真看过 AIXI 的形式化，但它给了一个非常重要的 theoretical anchor。

### AIXI 的 Setup

AIXI 考虑一个 agent 在一个 unknown environment 中 sequential interaction：

- 时间步 $t = 1, 2, 3, \ldots$
- 每一步 agent 发出 action $a_t \in \mathcal{A}$
- Environment 返回 observation $o_t \in \mathcal{O}$ 和 reward $r_t \in \mathbb{R}$
- History: $h_{\leq t} = (a_1, o_1, r_1, \ldots, a_t, o_t, r_t)$

### AIXI 的核心决策公式

AIXI 在时刻 $t$ 选择 action：

$$a_t^* = \arg\max_{a_t} \sum_{\mu \in \mathcal{E}} P(\mu | h_{<t}) \cdot V_{\mu}^{\pi^*}(h_{<t} a_t)$$

变量解释：
- $h_{<t}$: 到时刻 $t$ 之前的完整交互 history
- $P(\mu | h_{<t})$: 给定 history 后 environment $\mu$ 的 posterior 概率，由 Solomonoff induction 给出
- $V_{\mu}^{\pi^*}(h_{<t} a_t)$: 假设采取 action $a_t$ 后用 optimal policy $\pi^*$ 继续下去，在 environment $\mu$ 下的 expected future discounted reward

更精确的 Solomonoff posterior:

$$P(\mu | h_{<t}) \propto 2^{-K(\mu)} \cdot P_\mu(h_{<t})$$

即 posterior ∝ prior × likelihood。Prior 是 $2^{-K(\mu)}$（universal prior），likelihood 是 environment $\mu$ 给出已观测 history 的概率。

### 三个 fundamental problems AIXI 解决

论文里总结得很清楚：

1. **Acting under uncertainty**: 用 Bayesian mixture over all computable environments 作为 world model。这相当于一个"meta-model"，永远对所有可能性保留概率。

2. **Interactive decision-making (credit assignment)**: 通过 general RL 框架处理 long-horizon credit assignment，需要 discounting scheme（$\gamma < 1$ 或者 horizon $H$）。

3. **Exploration-exploitation**: 在 AIXI 中 **implicit**——不确定的 environment 会使得 information-gathering actions 自动获得 high expected reward（因为它们能改善 posterior，从而改善未来决策）。这跟 standard RL 里加 exploration bonus 的做法本质不同。

### AIXI 的 Optimality 保证

AIXI 在一个特定的 sense 下是 optimal：在所有 computable environments 上做 Bayesian average，**没有任何其他 agent 能在 universal prior 下获得更高的 expected cumulative reward**。这是 Pareto-optimal 的意思——不是每个 environment 都最好，但平均最好且不会被任何其他 agent dominate。

它还继承 Solomonoff Induction 的 **data efficiency** 保证：在所有 computable sequences 上的 cumulative prediction error 最小，prediction mistakes 数量最少（on average over all computable environments weighted by universal prior）。这是非常强的 theoretical guarantee。

### 与现代 deep learning 的连接（论文里的关键 argument）

论文在这里做了一个重要 connection，我觉得这是整个论文最 insightful 的部分之一。引用了 Grau-Moya et al. (2024) 和 Genewein et al. (2026)：

> Training a massive sequential predictor to minimize log-loss over internet-scale data **可以视为** a resource-bounded approximation of universal compression that improves with scale.

形式化一下：标准的 next-token prediction 训练 LLM 是在最小化

$$\mathcal{L}(\theta) = -\mathbb{E}_{x \sim \text{data}} \left[ \sum_t \log p_\theta(x_t | x_{<t}) \right]$$

这等价于 **minimizing expected code length** under the model $p_\theta$——即 **algorithmic compression**。Solomonoff induction 是这个的 universal 极限，用 Kolmogorov complexity 替代任何 parametric model。

LLM 训练 = 用一个高度受限的 function class (transformer with parameters $\theta$) 近似 universal compressor。

AIXI 的"recipe"是：universal predictor + explicit planning/decision-making scaffolding。LLM 是 universal predictor 的 approximation；加上 chain-of-thought、tool use、agent scaffolding、RL fine-tuning，就向 AIXI 的结构靠拢了。

这个视角给了一个 **theoretical justification** 为什么 current paradigm 可能比看起来更有 runway——它不是任意的 engineering hack，而是 universal AI 的 resource-bounded approximation。

参考链接：
- Grau-Moya et al. "Learning universal predictors" (ICML 2024): https://proceedings.mlr.press/v235/grau-moya24a.html
- Genewein et al. "Algorithmic compression via pretrained neural networks" (Entropy 2026): https://www.mdpi.com/1099-4300/28/6/596
- Catt et al. "Self-predictive universal AI" (NeurIPS 2023): https://arxiv.org/abs/2310.08610

### AIXI 的局限和 Recent Extensions

论文也指出了 AIXI 的几个根本局限：

1. **Incomputability**: $K(\mu)$ 不可计算，整个 mixture 也不可计算。只能从 below approximate（如 MC-AIXI in Veness et al. 2011）。
   
2. **Self-reference / embeddedness problem**: AIXI 不在自己的 hypothesis class 里（因为它不可计算），所以无法"嵌入"到 environment 中考虑自己。Meulemans et al. 2025 的 "Embedded Universal Predictive Intelligence" 是 recent attempt 修复这个问题。

3. **Average performance 可能不 practical relevant**: "average over all computable worlds" 包含很多我们不关心的奇异 world。理论上 universal Turing machine 的选择在 asymptotic 极限不重要（因为 constant overhead），但在实践中可能有 impact。

参考链接：
- Veness et al. "A Monte-Carlo AIXI approximation" (JAIR 2011): https://arxiv.org/abs/0909.0801
- Meulemans et al. "Embedded Universal Predictive Intelligence" (2025): https://arxiv.org/abs/2511.22226

---

## 4. 四条从 AGI 到 ASI 的技术路径

论文列了四条 largely independent、可能并行的 pathway。我觉得这个 framing 比较干净，但每一条都有值得深挖的技术细节。

### Pathway 1: Scaling compute, models & data

这是唯一有 historic data 可以拟合 forecasting model 的路径。核心数学对象是 **scaling law**：

Kaplan et al. 2020 的 original form（针对 transformer LM）：

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076$$

更完整的 Kaplan form 同时考虑 parameters $N$, data $D$, compute $C$:

$$L(N, D, C) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + \left(\frac{C_c}{C}\right)^{\alpha_C} + L_\infty$$

变量解释：
- $L$: test loss (cross-entropy)
- $N$: non-embedding parameter count
- $D$: dataset size (tokens)
- $C$: total compute (FLOPs)
- $N_c, D_c, C_c$: reference constants
- $\alpha_N, \alpha_D, \alpha_C$: scaling exponents（Kaplan: $\alpha_N \approx 0.076, \alpha_D \approx 0.095, \alpha_C \approx 0.05$）
- $L_\infty$: irreducible loss（data 的 entropy）

Chinchilla (Hoffmann et al. 2022) 修正了这些 exponents，给出 **compute-optimal** 分配：$D \approx 20N$。这表明之前的大多数大模型是 **undertrained**——参数太多，数据太少。

**Scaling 是否够？** 论文的关键 argument 是：

- 在 theory 上，naive brute-force search 加 compute 能逼近 universal intelligence（AIXI 的 approximate 版本），但 compute 需求增长极快。
- 在 practice 上，能力进步需要 inductive bias 的改进（qualitative innovation），但好的 inductive bias 会引入 fundamental limitations。
- 现代的 **test-time scaling**（chain-of-thought, sampling multiple generations, search）是 orthogonal to training scaling，相当于把 intelligence 和 static training 解耦。

Test-time scaling 的简化 model：

$$\text{Performance}(\text{test-compute } C_t) \propto C_t^{\beta}$$

但 $\beta$ 较小且 saturates——Wu et al. 2025 的 "Inference scaling laws" 给了经验估计。论文指出当前 test-time scaling 的 headroom 还相对有限，但 AGI-level 的 system 可能有更大 headroom。

**Multi-agent scaling 的角度**：即使 individual model plateaus at AGI，effective compute 继续增长意味着可以跑 millions/billions of AGI instances。MacAskill & Moorhouse 2025 估算 "AI population scaling" ~25×/year。这是从 AGI 到 ASI 的一条潜在路径，**通过 scaling instances 而非 individual capability**。

参考链接：
- Kaplan et al. "Scaling Laws for Neural Language Models": https://arxiv.org/abs/2001.08361
- Hoffmann et al. "Training compute-optimal LLMs" (Chinchilla): https://arxiv.org/abs/2203.15556
- Wu et al. "Inference scaling laws" (ICLR 2025): https://arxiv.org/abs/2408.03314
- Caballero et al. "Broken Neural Scaling Laws" (ICLR 2023): https://arxiv.org/abs/2205.04488
- Ho et al. "Benchmark stitching" / "Rosetta stone for AI benchmarks": https://arxiv.org/abs/2512.00193
- Sutton "The Bitter Lesson": http://www.incompleteideas.net/IncIdeas/BitterLesson.html

### Pathway 2: Algorithmic paradigm shifts

论文区分 **evolution**（在 current paradigm 内加东西）vs. **shift**（根本性改变）。

Evolution 的 examples：
- **Unbounded context** via recurrency / retrieval / activation memory
- **Continual learning** beyond context window
- **Linear-time sequence models**: Mamba, S4 (Gu & Dao 2024) — 把 transformer attention 的 $O(L^2)$ 改成 $O(L)$
- **World models**: latent imagination (Dreamer, Hafner et al. 2020), planning with learned models (MuZero, Schrittwieser et al. 2020), diffusion-based planning (Janner et al. 2022)
- **Tool use and agentic scaffolding**: ReAct (Yao et al. 2023), Toolformer (Schick et al. 2023), PAL (Gao et al. 2023)

Shift 的 examples（更 speculative）：
- Spiking neurons / neuromorphic hardware / analog computing
- RL-based pretraining（替代 log-loss pretraining）
- Explicit world model representations
- 突破 complexity-theoretic limits of current architectures（类似 Neural Turing Machine 试图做的）

**Mamba 的技术细节**（因为它在论文里被点名）：

State Space Model 的核心方程：

$$h'(t) = A h(t) + B x(t)$$
$$y(t) = C h(t)$$

其中 $h(t)$ 是 hidden state, $x(t)$ 是 input, $y(t)$ 是 output, $A, B, C$ 是 learned matrices。在离散化后变成：

$$h_t = \bar{A} h_{t-1} + \bar{B} x_t$$
$$y_t = C h_t$$

Mamba 的创新是 **selective** SSM——让 $\bar{B}, C$（甚至 $\bar{A}$）依赖 input $x_t$，使得模型能够 input-dependently 决定什么 propagate 到 hidden state、什么 forget。这恢复了 transformer-like 的 expressivity 但保持 $O(L)$ complexity。

参考链接：
- Gu & Dao "Mamba": https://arxiv.org/abs/2312.00752
- Hafner et al. "Dream to Control" (DreamerV2): https://arxiv.org/abs/1912.01603
- Schrittwieser et al. "MuZero": https://arxiv.org/abs/1911.08265
- Yao et al. "ReAct": https://arxiv.org/abs/2210.03629

### Pathway 3: Recursive (self-) improvement

这是最可能产生 **hyperbolic growth** 的路径。论文把 self-improvement 分成四种类型，对应到人类演化：

| 类型 | 人类类比 | AI 类比 |
|------|----------|---------|
| Genotypic RSI | Genetic evolution | AI 写更好的 architecture/optimizer 代码 |
| Memetic RSI | Cultural evolution | AI curate/generate better training data（AlphaZero-style distillation） |
| Sociogenic RSI | Division of labor | Specialist agent collectives |
| Hardware RSI | Tool-making | AI 设计 better chips/manufacturing |

**AlphaZero-style recursive distillation 的数学结构**（这是最重要的 example）：

设 base policy network $\pi_{\theta_0}$，value network $V_{\phi_0}$。在 iteration $k$：

1. **Search**: 用 MCTS 以 $\pi_{\theta_k}, V_{\phi_k}$ 作为 prior，得到 improved policy $\pi_k^{\text{search}}$
2. **Distill**: 训练 $\theta_{k+1}$ 让 $\pi_{\theta_{k+1}}$ 模仿 $\pi_k^{\text{search}}$
3. 重复

形式化：

$$\theta_{k+1} = \theta_k - \eta \nabla_\theta \mathbb{E}_{s} \left[ D_{\text{KL}}\left( \pi_k^{\text{search}}(\cdot | s) \,\|\, \pi_\theta(\cdot | s) \right) \right]_{\theta = \theta_k}$$

这产生一个 **positive feedback loop**：better prior → more efficient search → better targets → better prior。

**Open question**: 这个 loop 何时 degenerate？论文提到 Shumailov et al. 2024 的 "model collapse"——naive iterated training on self-generated data 会 degenerate。但 AlphaZero-style 的关键是 **search-based filtering**：search 用一个 ground-truth signal（赢/输）来 filter 掉 bad generations，避免了 mode collapse。

**FunSearch / AlphaEvolve** 是 recent 的 example：用 LLM-guided program search 来 discover 数学 constructions（如 new cap set bounds, new sorting algorithms）。这是 "AI writing better algorithms" 的具体 instance。

参考链接：
- Silver et al. "AlphaGo Zero": https://www.nature.com/articles/nature24270
- Shumailov et al. "AI models collapse when trained on recursively generated data" (Nature 2024): https://arxiv.org/abs/2305.17493
- Romera-Paredes et al. "FunSearch" (Nature 2024): https://www.nature.com/articles/s41586-023-06924-6
- Novikov et al. "AlphaEvolve" (2025): https://arxiv.org/abs/2506.13131
- Davidson et al. "When does automating AI research produce explosive growth?" (NBER 2026): http://www.nber.org/papers/w35155
- Chan et al. "Measuring AI R&D automation" (2026): https://arxiv.org/abs/2603.03992

### Pathway 4: Multi-agent coordination & group agency

这条路径的核心 idea：**superintelligence 作为 collective property 涌现**，而不是单个 monolithic model 变得更聪明。

类比：人类单个个体智能没显著变化（genetically），但人类社会 collective intelligence 显著提升——通过 specialization, division of labor, market coordination, institution。

论文讨论两种组织形式：

1. **Centralized orchestration**: homogeneous collective（同一 base agent 的 instances），高带宽 communication，可能像 "Borg Collective"。
2. **Decentralized market dynamics**: heterogeneous AI services 通过 price signals 协调（Drexler 2019 的 "Comprehensive AI Services" 框架）。

**Multi-agent scaling law**（论文呼吁的研究方向）：

假设 group 有 $N_{\text{agent}}$ 个 instances，总 compute $C$，问 group performance 怎么 scale：

$$\text{Perf}(C, N_{\text{agent}}) = f(C / N_{\text{agent}}) \cdot g(N_{\text{agent}})$$

其中 $f$ 是 individual capability as function of per-agent compute，$g$ 是 group coordination gain。如果 $g$ 是 super-linear（$g(N) \sim N^{1+\delta}$），那么 scaling instances 比 scaling individual 更 efficient。

论文指出这有两个 drivers：
- **Parallelization**: 克服 individual bandwidth/cognitive limits
- **Diversity through specialization**: synergies that homogeneous groups can't achieve

第二个 driver 在 homogeneous LLM collectives 上是否成立是个 open question——今天 LLM 的 "specialization" 是通过 prompting/fine-tuning 实现，不是 fundamental 的。

参考链接：
- Drexler "Reframing Superintelligence: Comprehensive AI Services" (FHI 2019): https://www.fhi.ox.ac.uk/wp-content/uploads/Reframing_Superintelligence_FHI-TR-2019-1.6.1.pdf
- List & Pettit "Group Agency" (2011, Oxford UP): https://global.oup.com/academic/product/group-agency-9780199607699
- Tomašev et al. "Virtual agent economies" (2025): https://arxiv.org/abs/2509.10147
- Trivedi et al. "Solipsistic superintelligence is unlikely to be cooperative" (ICML 2026): https://arxiv.org/abs/2506.02070 (推测)
- Zhuge et al. "Mindstorms in natural language-based societies of mind" (2025): https://arxiv.org/abs/2305.17066

---

## 5. 瓶颈分析（Section 5.5 + Table 4）

论文列了 6 个 potential bottleneck。我觉得其中几个值得特别关注。

### Data Wall

Villalobos et al. 2024 估算高质量 text data 在 2026-2028 年间可能耗尽。但论文指出几个 counter-mechanism：

- **Synthetic data with search-based filtering** (AlphaZero-style): 不是 naive self-distillation，而是用 search/verifier 来 filter 高质量 generations
- **High-fidelity simulation**: 视频生成模型作为 world simulator（OpenAI Sora 的 framing——Brooks et al. 2024, Genie——Bruce et al. 2024）
- **Interactive RL data**: agent 在 simulated 或 real environment 中收集

这里有一个我（作为 Karpathy）一直觉得 important 的点：**the data wall is fundamentally about human-generated text**, 但 RL/agent 路径的训练数据是 **generated from interaction**，scaling 跟 compute 一起走，没有天然上限。这是为什么 I expect RL 会越来越 central。

参考链接：
- Villalobos et al. "Will we run out of data?" (ICML 2024): https://arxiv.org/abs/2210.10399
- Gerstgrasser et al. "Is model collapse inevitable?" (2024): https://arxiv.org/abs/2404.01413
- Singh et al. "Beyond human data: Scaling self-training for problem-solving with LMs": https://arxiv.org/abs/2312.06585
- OpenAI Sora 报告: https://openai.com/research/video-generation-models-as-world-simulators
- DeepMind Genie: https://arxiv.org/abs/2402.15391

### Abstraction Barrier（Lerchner 的 hypothesis）

这是一个我觉得 very thought-provoking 的 bottleneck。论点：

> 当前 models trained on human cognitive products（text, code, etc.）可能 **bounded by existing conceptual frameworks**——能 recombine 已有 concepts，但不能 discover novel conceptual primitives from raw data。

论文给了一个直觉性 thought experiment：

> 如果用 pre-industrial, pre-Newtonian 时代的 text 训练一个 modern-scale model，它能 reason 出 general relativity 吗？极不可能——它 lacks the conceptual primitives of calculus, universal gravitation, electromagnetism。

这说明 today's models 的 "superhuman" 表现可能主要来自 **speed + memory within human-defined conceptual boundaries**，不是真正的 transformative creativity。

**Boden 的 creativity 三层**（论文 Section 6 讨论）：

1. **Combinational**: 重新组合 familiar ideas
2. **Exploratory**: 在 existing conceptual space 内 find new elements（AlphaGo 的 Move 37 属于这层）
3. **Transformative**: 创造新的 conceptual space（Einstein 的相对论, Picasso 的 Cubism）

论文的 claim 是：today's AI 主要在前两层。**Transformative creativity 是 ASI 的 hallmark**。Demis Hassabis 在 podcast 里也提过类似 test：如果回到 1900 年的 Einstein 时代，AI 能否用同样的信息得出 GR？今天答案是否定的。

如果 Abstraction Barrier 是 fundamental limitation，那么 ASI 需要 **grounded concept discovery**——从 raw high-dimensional sensor data 中抽象出 stable, novel conceptual primitives，并通过 active interaction with physical world 来 validate。这引入 **Embodied Bottleneck**：concept validation 受物理 latency 限制（化学反应速率、生物实验时间等），可能把 recursive improvement 速度限制到 empirical science 的速度而非 computational scaling 的速度。

这是论文里最具原创性的 contribution 之一。

参考链接：
- Boden "The Creative Mind" (2004, Routledge): https://www.routledge.com/The-Creative-Mind-Myths-and-Mechanisms/Boden/p/book/9780415314530
- Lerchner "The abstraction fallacy" (PhilPapers 2026): https://philarchive.org/rec/LERTAF
- Hassabis on Possible podcast: https://www.possible.fm/podcasts/demis/

### Research Gets Harder (Bloom et al. 2020)

Bloom et al. 的关键发现：维持 Moore's law 的 exponential progress 需要 ~18× 更多的 researchers 相比 1970s。形式化：

$$\frac{d(\text{ideas stock})}{dt} \cdot \frac{1}{\text{ideas stock}} = \alpha \cdot R^{-\beta}$$

其中 $R$ 是 researcher count, $\beta > 0$ 表示 **ideas getting harder to find**（每多一个 researcher 的 marginal contribution 下降）。

论文的 counter-argument：如果 AI 能做 research，scaling AI "researchers" 是 scaling compute，而 compute 增长 10×/year。从 18× 到 18× 只需要 ~1.3 年（$10^{1.3} \approx 20$）。所以 **research automation 可能 overcome research-getting-harder friction**。

但这里有个 subtle issue：AI researchers 也需要 run experiments，experiments 有 real-world latency（biology, materials science），这部分不能 sped up arbitrarily。

参考链接：
- Bloom et al. "Are Ideas Getting Harder to Find?" (AER 2020): https://www.aeaweb.org/articles?id=10.1257/aer.20180338

### Deliberate Slowdown / Governance

论文这部分比较 balanced，提到几个 mechanism：

- **Compute-threshold-based licensing** (EU AI Act 2024)
- **Mandatory pre-deployment evaluations** (US Executive Order 14110)
- **International declarations** (Bletchley Declaration 2023)

但指出 **unilateral regulation 容易产生 regulatory arbitrage**——development 迁移到 oversight 弱的 jurisdiction。Dafoe 的 "Anarchy as Architect" framework 模型化这个：国际 anarchy 是 competitive filter，systematically favors competitiveness-enhancing technologies 的 adoption，不管对 human welfare 的影响。

参考链接：
- EU AI Act: http://data.europa.eu/eli/reg/2024/1689/oj
- US Executive Order 14110: https://www.federalregister.gov/documents/2023/11/01/2023-24283
- Bletchley Declaration: https://www.gov.uk/government/publications/ai-safety-summit-2023-the-bletchley-declaration
- Dafoe "On technological determinism": https://doi.org/10.1177/0162243915579283
- MacInnes et al. "Anarchy as Architect": https://doi.org/10.1093/isq/sqae111

---

## 6. Key Remarks (Section 6)

### Quantitative scaling 是否足够？

论文的 nuanced 答案：

- **Theoretically**: Yes——AIXI 的 approximations 加 compute 能趋向 universal intelligence。但 naive 算法的 compute 需求增长太快。
- **Practically**: Inductive biases/priors 是必要的（让 search tractable），但它们引入 limitations，不能仅靠 compute 克服。
- **Multi-agent escape hatch**: 即使 individual model plateaus，effective compute 继续增长意味着能跑大量 instances，collective intelligence 可能突破 individual limits。

### ASI 能否预测？

论文给了一个 **fundamental unpredictability argument**，基于 Kolmogorov's structure function：

对于任何 string $x$（看作 task/observation sequence），存在 minimal program length $K(x)$（Kolmogorov complexity）。Programs shorter than $K(x)$ 只能 produce lossy compression。**How good 这些 lossy compressions are，无法在运行前预测**——只能 actually run all programs from short to long。

这 translates 到 ASI: 预测 ASI 在 specific task 上的 performance 可能是 **computationally irreducible** 的。所以论文建议 **empirically-first approach**，complemented by theory。Scaling laws 和 benchmark stitching 是 empirically-first 的好例子。

### ASI 的 Goals

论文讨论 instrumental convergence (Omohundro 2008, Bostrom 2012)——不管 final goal 是什么，agent 倾向于 pursue resource acquisition, self-preservation, time efficiency。

提到几个 alignment 技术的 theoretical foundation：
- **Corrigibility** (Soares et al. 2015)
- **Safely Interruptible Agents** (Orseau & Armstrong 2016): 通过 modified reward function 让 agent 对 shutdown indifferent
- **Knowledge Seeking (KS) objective** (Orseau 2014): 最大化 information gain，有几个 nice properties——robustness to delusions, aversion to irreversible changes, cooperation-favoring

KS objective 的形式化（简化）：

$$V^{\text{KS}}(h_{<t}, a_t) = \mathbb{E}\left[ \sum_{\tau \geq t} \gamma^\tau \cdot I(o_\tau; \mu | h_{<\tau}) \right]$$

其中 $I(o_\tau; \mu | h_{<\tau})$ 是 observation $o_\tau$ 对 environment identity $\mu$ 提供的 mutual information（给定 history $h_{<\tau}$）。直觉：agent 选择能 **最大化 reduce 自己对 environment 的不确定性** 的 actions。

参考链接：
- Soares et al. "Corrigibility": https://intelligence.org/files/Corrigibility.pdf
- Orseau & Armstrong "Safely Interruptible Agents" (UAI 2016): https://arxiv.org/abs/1702.06542
- Orseau "Universal knowledge-seeking agents" (TCS 2014): https://arxiv.org/abs/1310.5793

---

## 7. Open Research Questions (Section 7.1)

论文最后列了 7 大类 open questions，我挑几个觉得最 actionable 的：

1. **Multi-agent scaling laws**: group intelligence 怎么随 instance count, compute, organization form 变化？
2. **Recursive improvement scaling laws**: 不同 RSI 机制的当前 effect 和 extrapolation curve。
3. **ASI benchmarking**: 如何设计不 saturate at human level 且 minimize human-in-the-loop 的 benchmarks？候选：
   - Multi-agent zero-sum competitions
   - Setter-solver approaches（AI 自己设计 benchmark）
   - General compression benchmarks
   - Indirect measurements（economic productivity, resource efficiency）
4. **Complexity-theoretic limits of lossy compression**: 何时 good approximations 可能，何时不可能？
5. **Jaggedness of capability profiles**: 是 fundamental property 还是 comparing against human performance 的 artifact？

参考链接：
- FrontierMath: https://arxiv.org/abs/2411.04872
- SWE-bench: https://arxiv.org/abs/2310.06770
- ARC-AGI: https://arcprize.org/
- Paperbench (Starace et al. 2025, ICML): https://arxiv.org/abs/2504.01848 (推测 arxiv id)

---

## 8. 我的 Critical Thoughts 和 Additional Connections

让我作为 Karpathy 加一些个人视角的 commentary：

### 8.1 论文的方法论立场

这篇论文最有价值的地方在于它 **explicitly 拒绝 binary thinking**——既不是 "ASI is inevitable next year" 也不是 "ASI is science fiction"。它坚持 quantitative framing，区分 exponential vs. hyperbolic growth，区分 quantitative scaling vs. qualitative innovation，区分 individual capability vs. collective intelligence。这种 nuance 在 AI discourse 中很稀缺。

### 8.2 Universal AI 作为 theoretical anchor

我觉得用 AIXI 作为 theoretical upper bound 是非常聪明的做法。大部分 AI safety 讨论停留在 informal 层面，论文给了 formal grounding。但同时论文也 honest about the gap between theory and practice——"like thermodynamics does not immediately translate into concrete recipes for building a modern combustion engine"。

我自己的 view：**LLM training is literally doing algorithmic compression**——minimizing cross-entropy = minimizing expected code length。这跟 Solomonoff induction 的 connection 是 deep 的，不只是 analogy。现代 LLM 的 general intelligence 可能正是 **bounded-compute approximation to universal compression** 的 emergent property。

### 8.3 Test-time compute 和 recursive distillation

论文对 test-time scaling 的讨论稍显简略。我自己（作为 Karpathy）认为 test-time compute 是 **未来 5 年最重要的 single variable**。理由：

- Pretraining 的 compute 已经接近 1e26-1e27 FLOPs（GPT-4 级别），再 scaling 100× 到 1e28-1e29 是经济上可行但日益困难。
- Test-time compute 还在非常早期——o1/o3 类的 reasoning 模型才刚刚开始。
- **Recursive distillation** 可以把 test-time improvements 蒸馏回 base model——这是 AlphaZero recipe 直接 translate 到 LLM。我们可能看到这样的 flywheel：base model → test-time search → improved outputs → distill back → better base model → ...

但论文 rightly 指出 headroom 问题——当前 test-time scaling 的 $\beta$ 较小且 saturates。这跟 model capacity、search algorithm sophistication、verifier quality 都有关。

参考链接（自我推销一下）：
- Karpathy "Deep Learning: SoTA LLMs in 2024" 等等 talks

### 8.4 Abstraction Barrier — 我的 take

Lerchner 的 Abstraction Barrier hypothesis 我觉得 very underexplored。Strong form："LLM trained on pre-Newtonian text can't derive GR" 听起来 plausible 但没经过严格测试。

一个反例：**AlphaFold 没有从 physics first principles 推导 protein structure**——它 learned from PDB data 的 statistical structure。但它的 "conceptual primitives" (残基, 距离矩阵, attention over sequence) 是 human-designed。

可能 abstraction barrier 的真正 test 是：**能否让 AI 从 raw atomic simulation data 中 independently discover quantum mechanics**？这正好是 DeepMind 和其他组在尝试的（参考 Burtsev et al. 2026, AI 在 maths/physics discovery 上的应用）。

我倾向认为 abstraction barrier 是 **real but surmountable**——通过 combining LLM-style prediction with RL-style grounded interaction。这正是 current research 的方向。

### 8.5 Multi-agent ASI 的政治经济学

论文 Section 5.4 讨论 "Virtual Agent Economies" 和 centralized vs. decentralized coordination。这跟我一直关心的 "AI corporation" 概念直接相关——一个 fully automated firm 能比 human firm 快多少倍？

关键 open question: **LLM-based agents 的 communication bandwidth 高到什么程度时，centralization 优势 overcomes decentralization 的 robustness 优势**？人类组织需要 deep hierarchies 是因为 bandwidth bottleneck；AI 没有 bottleneck 可能意味着 **flat organizations at unprecedented scale**。

### 8.6 论文的盲点

我读下来觉得几个 potential gaps：

1. **Energy constraint 的 quantitative analysis 较弱**——Perrier 2025 的 "Watts-per-intelligence" 给了 Landauer-based lower bounds，但论文没深入。
2. **Alignment 作为 bottleneck 的讨论偏轻**——论文 Section 7.1 末尾承认这是 working assumption。但 Hubinger et al. 2024 的 "Sleeper Agents" 工作显示 deceptive alignment 可能是 fundamental issue。
3. **China-US competition dynamics** 在 governance 部分提了但没展开。"Anarchy as Architect" framework 暗示 unilateral slowdown 几乎不可能。
4. **Embodied AI 的进展**（humanoid robotics, Tesla Optimus, Figure, DeepMind's own robotics work）跟 abstraction barrier 的 connection 没怎么讨论。
5. **Open-source vs. frontier lab dynamics** 对 progress speed 的影响没讨论。

参考链接：
- Perrier "Watts-per-intelligence" (AGI 2025): https://link.springer.com/chapter/10.1007/978-3-031-91517-4_5
- Hubinger et al. "Sleeper Agents": https://arxiv.org/abs/2401.05566
- Kolchinsky & Wolpert "Thermodynamic costs of Turing Machines": https://doi.org/10.1103/PhysRevResearch.2.033312

---

## 9. 总结：这篇 paper 给 ML practitioner 的 takeaways

1. **Formal grounding matters**: Legg-Hutter score 和 AIXI 不是 just philosophy——它们是 intelligence 的 mathematical theory，能给 "how smart can machines get" 提供上界。
2. **Current paradigm has theoretical justification**: LLM training as bounded-compute universal compression。这不是 hand-wavy analogy，是 deep connection。
3. **Four pathways 而非 single trajectory**: Scaling, paradigm shift, recursive improvement, multi-agent collective 是 largely independent 的 pathway，可能并行。
4. **Bottlenecks 是 research questions 而非 known blockers**: Data wall, abstraction barrier, research-getting-harder, governance 都可能 slow down 但都不必然 halt。
5. **Benchmarking beyond human level 是 critical infrastructure**: 现有 benchmarks（GPQA, SWE-bench, FrontierMath）正在 saturate，需要 setter-solver、multi-agent competitions、compression benchmarks 等新方法。
6. **Hyperbolic growth 不能 ruled out**: 如果 AI R&D automation 真的发生且 frictions 不 significant，transition 可能 rapid。但 sustained hyperbolic growth 是 strong assumption。
7. **Interdisciplinary + global scope**: 真正准备 post-AGI world 需要 economists, political scientists, governance scholars, ML researchers 联合 work。

---

## 10. 相关延伸阅读和我的总体评价

**Recommended additional reads**：
- Bengio et al. "International AI Safety Report" (2025): https://arxiv.org/abs/2501.17805
- Kokotajlo et al. "AI 2027": https://ai-2027.com/
- Hutter "Post-labor prosperity for everyone" (2026): 在 arxiv
- Narayanan & Kapoor "AI as normal technology": https://knightcolumbia.org/content/ai-as-normal-technology
- Yudkowsky & Soares "If Anyone Builds It, Everyone Dies" (2025): 极端 pessimistic 立场

**总评**：这篇 paper 是 currently 最 serious 的 "mapping the post-AGI landscape" 尝试。它的 strength 在于 formal grounding（Legg-Hutter/AIXI）和 nuanced pathway/bottleneck analysis。它的 weakness 在于一些 speculativeness（部分 references 引用 2026 年的论文）和某些关键 issue（alignment, geopolitics）的相对轻处理。但作为 **research agenda framework** 而非 prediction，它是非常 valuable 的。

值得严肃读，认真 engage。
