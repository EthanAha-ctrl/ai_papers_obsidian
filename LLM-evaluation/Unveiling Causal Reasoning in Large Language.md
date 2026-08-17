---
source_pdf: Unveiling Causal Reasoning in Large Language.pdf
paper_sha256: b5c06a2ebb23b7ecca45c364ad05ded65eab57e2c43ebbba405a8e6392166d11
processed_at: '2026-08-12T20:26:53-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇paper

Andrej, 我换个更接地气的方式重新讲, 核心就几个point。

---

## 一句话版本

**现在的LLM看起来会causal reasoning, 其实只是在"背答案", 遇到没见过的新情况就露馅了。**

---

## 这个问题为啥重要

你想想, 你问GPT "为什么今天下雨路上会湿", 它答得头头是道。你以为它懂causality了, 但你问它一个2024年1月才发生的新事件, 或者你编一个counterfactual场景 "如果火车改成social hub会怎样", 它就开始胡说八道。

这就引出一个问题: **它到底是真懂因果, 还是只是把训练数据里见过的因果句子背下来了?**

---

## Level-1 vs Level-2 — 用Kahneman的话讲

paper借用 *Thinking Fast and Slow* 的框架:

- **Level-1**: 快, 直觉, 背过的知识直接retrieve。比如你问"抽烟导致什么", 它直接从参数里捞"肺癌"。
- **Level-2**: 慢, 真推理, 遇到没见过的case能从first principles推出来。比如你问一个虚构的新事件, 它能像人一样分析"哦这个动作会导致X, 因为..."

当前LLM只有Level-1, Level-2基本没有。理想状态是两个level能像人一样切换 — 简单的用System 1, 难的切到System 2。但现在没有这个切换机制。

---

## 为什么autoregression本质上不causal — 这是paper最核心的argument

这个point你应该最有体感。回想你的nanoGPT, 训练就是在做:

$$\max \sum_t \log P(w_{t+1} \mid w_1, ..., w_t)$$

也就是给定前面的tokens, 预测下一个token。这个objective本质上是在学 **observational joint distribution** 的marginal, 即 "在training data里, 看到A之后通常跟着B"。

但causal reasoning要的是另一回事。Pearl的do-calculus告诉我们, 真正的causal effect是:

$$P(Y \mid do(X=x))$$

即"如果我**主动干预**把X设成x, Y会怎样"。这和"观察到X=x时Y的分布"是两码事 — 因为有confounder C可能同时影响X和Y。

**关键intuition**: autoregressive LLM永远只学"观察到什么之后通常出现什么", 它没有任何机制去model"如果我intervene会怎样"。它的inductive bias就是observational的。

paper还引用了Hume的point: **sequential ≠ causal**。"A在B之前发生"不蕴含"A导致B"。但autoregression就是靠sequential prediction工作的, 所以它会把很多sequential pattern误当causal pattern学进去。

举个paper Figure 2的例子: 句子 "rain → school closure → cannot go to school → learn programming at home"。表面看起来是一条因果链, 但实际上"school closure"和"cannot go to school"是同一个event的两种说法, 不是两步因果。autoregressive model看这种句子, 就会把sequential structure当causal structure学进去, 导致在新context上出错。

---

## 怎么证明LLM只是Level-1 — CausalProbe 2024

paper的empirical strategy很直接: **造一个LLM没见过的benchmark, 看它掉多少分**。

如果LLM真会reasoning, 换新corpus应该不掉分; 如果是背答案, 换新corpus就崩。

具体做法:
- 从BBC和The Guardian抓2024年1-4月的文章 (这些LLM都没见过, 因为training cutoff更早)
- 用GPT 3.5从文章里自动生成causal Q&A pairs
- 分成三个子集: Easy (简单选), Hard (含伪造的假因果对), Multiple-choice (正确答案数1-4个)

Table 1的cutoff对比很clear:
- LLaMA 2 cutoff Sep 2022
- LLaMA 3 cutoff Mar 2023  
- GPT 3.5 cutoff Sep 2021
- Claude 3 cutoff Aug 2023
- **CausalProbe corpus: Jan 2024**, 全部晚于这些

结果从Figure 1(d)和Table 2看, 非常striking:

| Benchmark | Claude 3 (Vanilla) |
|-----------|-------------------|
| COPA (2011, 见过) | 0.991 |
| e-CARE (2020前, 见过) | 0.861 |
| CausalNet (ChatGPT生成, 半新) | 0.933 |
| C-E (全新) | 0.758 |
| C-H (全新+counterfactual干扰) | 0.692 |

**单调下降, 从99%掉到69%**。Claude 3 Opus这种SOTA model在CausalProbe-H上连70%都到不了。这就是Level-1 hypothesis的最强empirical evidence — 换新数据就崩, 说明它主要靠memorization。

---

## 用Min-K% Prob验证"真的没见过"

有人可能质疑: "你怎么知道LLM真没见过这些corpus?"

paper用了一个叫 **Min-K% Prob** 的membership inference attack方法。intuition很简单:

- 如果一段text对LLM是见过的, 那它对每个token的预测概率都不会太低 (因为记得)
- 如果没见过, 那总会有一些token它觉得"surprise", assign很低的概率

具体公式: 对token sequence x = (x_1, ..., x_n), 取log-probability最低的K%个token, 算它们的平均log-prob:

$$\text{Min-K\%Prob}(x) = \frac{1}{N} \sum_{x_i \in \text{MIN-K\%}(x)} \log p(x_i \mid x_1, ..., x_{i-1})$$

其中N是这K%个token的个数。**值越小 = 越fresh = 越没见过**。

Table 3的结果很clear (LLaMA 2, Min-20%):
- COPA: 10.57 (见过)
- e-CARE: 12.98 (见过)
- CausalNet: 9.45 (半新)
- **C-E: 7.27 (最新)**
- **C-H: 7.86 (最新)**

CausalProbe的Min-K% Prob值显著低于其他benchmark, empirically确认了它确实fresher。

---

## SCM Formalization — paper的理论core

这部分我觉得是paper最valuable的, 虽然它没完全展开, 但给了个good framework。

paper把textual causal reasoning cast成一个SCM:

```
    C (confounder: 物理世界law)
   / \
  ↓   ↓
  X → Y      (X cause Y, semantic level)
   \ /
    ↓
    T (observed text, conditioned collider)
    ↑
    ϵ (exogenous: 语言表达方式variation)
```

变量解释:
- **X**: cause的semantic content (比如"smoking")
- **Y**: effect的semantic content (比如"lung cancer")  
- **C**: driving X→Y的underlying physical law (confounder)
- **ϵ**: 表达层noise (语言类型, 主动/被动语态, context等)
- **T**: 观测到的natural language text
- **h**: encoding function, h(X, Y, ϵ) = T

**为什么T是conditioned collider**: T是X, Y, ϵ的common child。在causal reasoning任务里, T是被观测的(给定text), 即conditioned on T = T_0。经典causal inference结论: **conditioning on a collider opens path between its parents**, 所以X和Y之间通过T产生association。这就是为什么text能提供关于causal relation的信息 — 但只是association, 不是causal effect。

**Target**: 给定cause X = X_0和text T = T_0, 我们想找最优effect Y*:

$$\arg\max_{Y \sim P_Y} P(Y \mid X = X_0, T = T_0, C) \tag{1}$$

但C是unobserved confounder, 不能直接condition。用全概率公式marginalize C:

$$\arg\max_{Y \sim P_Y} \mathbb{E}_{C \sim P_C} P(Y \mid X = X_0, T = T_0, C) = \arg\max_{Y \sim P_Y} P(Y \mid X = X_0, T = T_0) \tag{2}$$

**Eq.(2)的intuition**: 正确的Y*应该是在所有可能的"物理law如何驱动X→Y"的解释上做expectation后的最可能Y。本质是Bayesian model averaging over confounder hypotheses。

**LLM的问题**: 它从training data学到的 P(Y|X, T) 是一个point estimate, implicit assume了一种特定的C (训练分布里的那个)。遇到新context (新C分布), point estimate就fail。G²-Reasoner用RAG retrieve external knowledge, 就是在approximate这个 E_{C~P_C} 的integral。

---

## G²-Reasoner — 朝Level-2的尝试

paper的method其实不复杂, 两个module:

1. **RAG**: 用Contriever + Faiss从一个小knowledge base (~16MB) retrieve相关general knowledge, append到prompt里
2. **Goal-oriented prompt**: 在prompt里显式告诉LLM"你的目标是找出正确的causal relation", 而不是让它aimlessly生成

架构图 (Figure 3):

```
[Question] 
   ↓
[RAG retrieve from 16MB knowledge base]
   ↓
[Goal-oriented prompt template]
   ↓
[LLM]
   ↓
[Answer]
```

对应到Eq.(2):
- RAG retrieval ≈ E_{C~P_C}的Monte Carlo approximation (retrieve top-k个最相关的C hypothesis)
- Goal prompt ≈ 诱导LLM做intervention-style reasoning, 而不是pure observational prediction

---

## 结果怎么样

Table 2的关键数字:

**C-H (最难, fresh+counterfactual)上G²-Reasoner vs Vanilla的提升:**
- LLaMA 2: 0.565 → 0.582 (+1.7%)
- LLaMA 3: 0.652 → 0.658 (+0.6%)
- GPT 3.5: 0.671 → 0.693 (+2.2%)
- Claude 3: 0.692 → 0.696 (+0.4%)

提升有, 但不大。paper自己承认这是个"first step", 没真正达到Level-2。

几个interesting的sub-finding:

**CoT经常比Vanilla还差** — 比如CausalNet上LLaMA 2 CoT 0.666 < Vanilla 0.673。这和你之前讲过的long-context failure一致: chain越长, LLM越容易偏离initial goal [\[40\]](https://arxiv.org/abs/2406.12920)。

**RAG单独通常不如Vanilla** — 说明naive retrieve knowledge反而干扰LLM内部reasoning。G²-Reasoner的优势来自RAG + goal prompt的组合, goal prompt是关键。

**CausalProbe-M上有个有意思的发现**: 用partial match (可以漏选正确选项, 但不能选错) 时, GPT 3.5达75%, Claude 3达85%。更重要的是, **LLMs很少犯false positive错误** — 它一旦说某个是causal relation, 通常是对的; 它的问题在false negative, 漏掉真正的causal relation。说明它像一个保守的detector, 识别能力大于发现能力。

---

## 我觉得paper的亮点和弱点

**亮点**:
1. **问题定义清晰**: Level-1 vs Level-2的taxonomy借Kahneman, 通俗又准确
2. **Empirical strategy直接**: 用time-shifted corpus避开memorization, 干净
3. **SCM formalization有启发性**: 把textual causal reasoning cast成confounder marginalization问题, 给了RAG-based method的theoretical justification
4. **Min-K% Prob验证freshness**: 不是空口说"我们corpus新", 而是用quantitative method验证

**弱点**:
1. **Level-2定义不operational**: 没给可验证的判据, 啥叫"达到Level-2"? 容易circular
2. **G²-Reasoner提升marginal**: 最难benchmark上只提升1-2%, 离Level-2还远
3. **Knowledge base太小**: 16MB真的小, paper自己说用Wikipedia API能大幅提升, 但没做scaling实验
4. **Counterfactual定义松**: CausalProbe-H的"fake pairs"不是Pearl严格意义的counterfactual, 更像plausibility check
5. **Eq.(2)只是framework**: 没quantitatively estimate P_C, 也没验证marginalization是否正确做了

---

## 对你的intuition building价值

1. **LLM是observational learner**: autoregressive loss只能学 P(Y|X), 学不到 P(Y|do(X))。这是architectural constraint, scale解决不了。

2. **Performance gap = memorization signal**: COPA 99% → CausalProbe-H 69%的gap就是Level-1 evidence。这个gap可以作为一个diagnostic tool — 任何新model都可以用"见过vs没见过"的performance gap来测它reasoning的真实性。

3. **SCM视角有用**: 把"text reasoning"重新cast成"conditioned collider + confounder marginalization"给了个formal language。虽然paper没完全exploit这个framework, 但这个方向值得深挖。

4. **RAG是patch, 不是solution**: 真正的Level-2可能需要architectural change — 显式intervention operator, world model, 或causal representation learning。G²-Reasoner只是用prompt engineering + RAG硬凑, 离本质solution还远。

5. **CoT不万能**: paper数据显示CoT有时更差, 这和你讲过的"LLM知道但不告诉"现象一致。reasoning不是简单"想得长一点", 需要结构化的搜索过程。

---

## 可能的后续方向

如果你想沿着这个方向思考:

a) **Interventional decoding**: 在decoding时显式引入do-operator, 比如用causal model辅助next-token prediction, 让LLM知道"这是intervention不是observation"

b) **RAG scaling law for causal reasoning**: knowledge base size vs causal reasoning performance的关系, 这paper完全没做

c) **World model + LLM**: 结合world model (能simulate intervention的)和LLM的语言能力, 类似 [\[20\]](https://arxiv.org/abs/2305.10601) Reasoning via Planning的思路但加causal structure

d) **Counterfactual training**: 在训练时就引入counterfactual data augmentation, 让LLM在训练时就区分observational vs interventional

e) **Causal representation learning**: 学disentangled representation, 让LLM的内部representation有causal structure, 这是 [\[65\]](https://arxiv.org/abs/2410.15319) 的方向

---

## Web Links

- Paper repo: [https://github.com/Haoang97/CausalProbe-2024](https://github.com/Haoang97/CausalProbe-2024)
- BBC corpus: [https://www.bbc.com/](https://www.bbc.com/)
- The Guardian corpus: [https://www.theguardian.com/](https://www.theguardian.com/)
- General Knowledge dataset: [HuggingFace](https://huggingface.co/datasets/MuskumPillerum/General-Knowledge)
- Min-K% Prob paper: [arXiv:2310.16789](https://arxiv.org/abs/2310.16789)
- Pearl Causality: [book](https://www.cambridge.org/core/books/causality/455346EBF0F5A6F0E8014D1DC2D5A4F5)
- Kahneman TFS: [Wikipedia](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow)
- Causal Parrots (similar conclusion): [TMLR 2023](https://arxiv.org/abs/2305.07371)
- Embers of Autoregression: [arXiv:2309.13638](https://arxiv.org/abs/2309.13638)
- Long context failure (CoT worse): [arXiv:2406.12920](https://arxiv.org/abs/2406.12920)
- Reasoning via Planning with World Model: [arXiv:2305.10601](https://arxiv.org/abs/2305.10601)
- Causality for LLMs survey: [arXiv:2410.15319](https://arxiv.org/abs/2410.15319)

---

**最后一句话总结**: 这paper本质是在说, LLM的causal reasoning是"mirage" — 看起来有, 实际是memorization。autoregression的inductive bias决定了它只能学observational pattern, 学不到interventional reasoning。要往Level-2走, 可能需要architectural change, 光靠prompt engineering和RAG只是patch。这个结论你应该会认同, 因为它和你一直讲的"LLM是loss minimizer不是reasoner"完全aligned。

---

# Unveiling Causal Reasoning in LLMs: Reality or Mirage? — 深度技术解读

Andrej, 这篇paper直击当前LLM研究的一个软肋。核心thesis可以一句话概括：**当前LLM展现出来的"causal reasoning"主要是参数中embedded的causal knowledge的retrieval (level-1), 而非真正像人类那样的novel causal reasoning (level-2)**。下面我会从formalization、mechanism、benchmark、method四个层面展开, build你的intuition。

---

## 1. 二级causal reasoning taxonomy — Kahneman的"快与慢"映射

paper借用Kahneman *Thinking, Fast and Slow* [\[27\]](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow) 的dual-process theory:

| 层级 | 定义 | 类比 | LLM实现 |
|------|------|------|---------|
| **Level-1** | 从参数和context中retrieve已embedded的causal knowledge, fast, 简单cause-effect | System 1, 直觉 | next-token prediction + memorized patterns |
| **Level-2** | 用sophisticated reasoning机制 + parametric knowledge + context deduce *new/unseen* causal knowledge, slow | System 2, 慢思考 | 当前LLM尚未真正实现 |

这里的intuition-building关键点: **Level-1不是"差"的, 它对常见case高效**, 问题在于当context对LLM unfamiliar或counterfactual时, Level-1失败, 而Level-2缺失。理想状态是adaptive switching (类似人类快慢思考协作), 但当前LLM只有System 1.

paper §3的Remark 1限定了研究scope:
- 仅single cause-effect pair (excludes mediators, multiple causes)
- 仅qualitative reasoning (excludes treatment effect estimation这类quantitative tasks)
- 两类query: "what is the reason..." / "what is the result..."

---

## 2. Methodological Argument: 为什么Autoregression本质上not Causal

这是paper最精彩的部分, 我详细展开。

### 2.1 Hume的sequential vs logical causality

paper引用David Hume *A Treatise of Human Nature* [\[21\]](https://www.gutenberg.org/files/4705/4705-h/4705-h.htm): **sequential causality ≠ logical causality**。即: "A先发生, B后发生"不蕴含"A导致B"。

Figure 2给的toy example: 句子"rain → school closure → cannot go to school → learn the new programming language at home"。表面sequential chain看似合理, 但实际causal链是错的 — 真正的链应该是"rain → cannot go to school" + "cannot go to school → learn at home", 而"school closure"和"cannot go to school"是同一event的两个描述, 不是两步cause。

### 2.2 Autoregressive loss的形式化缺陷

decoder-only transformer训练时优化autoregressive loss [\[35\]](https://arxiv.org/abs/1801.10198):
- 给定context **c** = (c_1, ..., c_k)
- 已生成tokens (w_1, ..., w_t)
- next token分布 P(w_{t+1} | **c**, w_1, ..., w_t) 通过softmax获得

paper指出两个核心failure modes:

**Failure mode 1**: 如果context **c** 不是sequentially causal **且** 对LLM unfamiliar → LLM会misunderstand其中的causal knowledge。

**Failure mode 2**: 即使 P(w*_{t+1} | **c**, w_1, ..., w_t) 很大 (next-token probability高), 但由(..., w_{t-1}, w_t, w*_{t+1})组成的text与causality laws或context **c** 不一致, LLM仍会输出错误causal answer。

这个insight其实和你一直讲的"LLM是loss minimizer, 不是reasoner"完全一致。LLM学的是 P(next token | history) 的joint distribution的marginal, 而causal reasoning需要的是interventional distribution P(Y | do(X=x)) (Pearl [\[46\]](https://en.wikipedia.org/wiki/Causality_(book))). 

观察vsintervention的gap, 在autoregressive framework里没有被显式model — 这正是SCM [\[46\]](https://ftp.cs.ucla.edu/pub/stat_ser/r350.pdf) 和do-calculus要解决的核心问题。LLM只能从observational data中学习correlational pattern。

### 2.3 与你的nanoGPT直觉的联系

如果你回想nanoGPT的实现, `logits = model(idx)` 然后 `probs = F.softmax(logits, dim=-1)`, 再sample — 这个pipeline严格就是paper描述的autoregressive过程。没有任何mechanism告诉model"如果do(X)而不是observe(X), distribution应该如何shift"。LLM的inductive bias [\[42\]](https://arxiv.org/abs/2309.13638)就是observational。

---

## 3. Structural Causal Model形式化 — 这部分最值得深挖

paper §5用SCM formalize textual causal reasoning, 这才是真正的intuition-building gold mine。

### 3.1 Causal graph (Figure 4)

```
    C (confounder: 物理世界/虚拟世界laws)
   / \
  ↓   ↓
  X → Y      (X cause Y)
   \ /
    ↓
    T (conditioned collider: natural language text)
    ↑
    ϵ (exogenous: 语言类型, context, 表达方式)
```

变量定义:
- **X**: cause semantic variable (原因的semantic content, 比如"smoking")
- **Y**: effect semantic variable (结果的semantic content, 比如"lung cancer")
- **C**: confounding variable — driving X→Y的underlying laws of physical world or imagined/virtual worlds
- **ϵ**: random exogenous variable — 表达层variation (语言类型、context、active/passive voice等)
- **T**: natural language expression (观测到的text)
- **h**: mapping function, h(X, Y, ϵ) = T (把semantic causal concepts编码成readable text)

### 3.2 为什么T是conditioned collider

T是X, Y, ϵ的common child (collider)。在textual causal reasoning任务中, T是被观测的(确定性的), 即T = T_0 (conditioned on observed text)。

根据经典causal inference [\[46\]](https://ftp.cs.ucla.edu/pub/stat_ser/r350.pdf): **conditioning on a collider opens the path between its parents**, 创造X和Y之间的association。这正是为什么natural language text能提供关于causal relationship的信息 — 但只是association, 不是真正的causal effect。

### 3.3 Target公式

对cause-to-effect task, 给定观测X = X_0 (问题中描述的cause)和T = T_0 (观测的text), 我们想找:

$$\arg\max_{Y \sim P_Y} \mathbb{P}[Y \mid X = X_0, T = T_0, C] \tag{1}$$

但C是unobserved confounder, 我们不能直接condition在它上。应用全概率公式marginalize C:

$$\arg\max_{Y \sim P_Y} \mathbb{E}_{C \sim P_C} \mathbb{P}[Y \mid X = X_0, T = T_0, C] = \arg\max_{Y \sim P_Y} \mathbb{P}[Y \mid X = X_0, T = T_0] \tag{2}$$

其中 **P_C 是general knowledge base**。

这个公式很关键: 它告诉我们, 要做genuine causal reasoning, 必须要marginalize over所有可能的C (即所有可能驱动X→Y的physical/virtual laws)。理想情况下需要complete general knowledge base, 现实中不可能, 但Eq.(2)给出了G²-Reasoner的conceptual foundation — 用RAG检索external knowledge来近似E_{C~P_C}。

### 3.4 公式的intuition

Eq.(2)的物理意义: **正确的causal answer Y*应该是在所有可能的"C如何驱动X→Y"的解释上做expectation后的最可能Y**。这本质上是Bayesian model averaging over confounder hypotheses。

LLM的问题: 它从training data学到的 P(Y | X, T) 是一个point estimate (一种特定的C implicit in training distribution), 而不是marginalize over C的integral。所以遇到新context (新C分布), point estimate就fail。

---

## 4. CausalProbe-2024 Benchmark — Empirical Probe

paper §4.2的核心实验设计: 通过确保benchmark corpus **晚于LLM training data cutoff**, 强制LLM不能靠memorization回答。

### 4.1 Data cutoff时间表 (Table 1)

| Model/Benchmark | Cutoff |
|-----------------|--------|
| LLaMA 2 7B chat | Sep 2022 |
| LLaMA 3 8B instruct | Mar 2023 |
| GPT 3.5 turbo | Sep 2021 |
| Claude 3 opus | Aug 2023 |
| **CausalProbe 2024** | **Jan 2024** |

corpus来源: BBC [link](https://www.bbc.com/) 和 The Guardian [link](https://www.theguardian.com/), 2024年1月1日 - 4月29日, 涵盖technology, environment, business, health, world news, culture, climate等categories (Table 10: BBC 967篇, Guardian 2702篇)。

### 4.2 三个子benchmark的设计哲学

| 子集 | 设计 | 测试什么 |
|------|------|----------|
| **CausalProbe-E** (Easy) | 单选, 模仿CausalQA [\[6\]](https://aclanthology.org/2022.coling-1.289/) 格式 | 新corpus上的genuine causal reasoning |
| **CausalProbe-H** (Hard) | 含made-up fake cause-effect pairs | counterfactual disturbance下的辨别能力 |
| **CausalProbe-M** (Multi-choice) | 正确答案数1-4个 | 防random guessing, 区分true/false positive |

CausalProbe-H最关键: 它在真实article的summary基础上, 用GPT 3.5生成正确cause-effect pairs **+ 人为捏造的incorrect pairs**, 组成多选question。这测试了LLM能否在counterfactual干扰下仍选出真正的causal relation — 这是Level-2 reasoning的hallmark。

CausalProbe-M的partial match评估: missing一些正确选项可接受, 但选错任何错误选项算fail。Figure 12显示正确答案数近似Gaussian分布, 主要集中在2-3个。

### 4.3 Min-K% Prob — 训练数据membership inference attack

paper §E用Min-K% Prob [\[53\]](https://arxiv.org/abs/2310.16789) 量化corpus的freshness。这个方法很巧妙:

假设: 如果sequence x对LLM是unseen的, 那么其中少数token会被assign低log-probability; 如果是seen的, 几乎不会有这种"surprise"token。

形式化: 给定token sequence x = (x_1, x_2, ..., x_n), 第i个token的条件log概率:
$$\log p(x_i \mid x_1, x_2, \dots, x_{i-1})$$

定义MIN-K%(x)为x中log-probability最低的K% tokens的集合。该集合的平均log-prob:
$$\text{Min-K\%Prob}(x) = \frac{1}{N} \sum_{x_i \in \text{MIN-K\%}(x)} \log p(x_i \mid x_1, x_2, \dots, x_{i-1})$$

其中N = |MIN-K%(x)|, **值越小说明越fresh (越unseen)**。

Table 3结果(以LLaMA 2为例):
| Benchmark | Min-10% | Min-20% | Min-30% |
|-----------|---------|---------|---------|
| COPA | 13.27 | 10.57 | 8.97 |
| e-CARE | 14.48 | 12.98 | 10.89 |
| CausalNet | 11.3 | 9.45 | 8.00 |
| **C-E** | **7.27** | **5.90** | **5.69** |
| **C-H** | **7.86** | **6.65** | **6.49** |

CausalProbe的Min-K% Prob值显著低于COPA和e-CARE, 证实了它的freshness。这是paper的solid empirical evidence。

---

## 5. G²-Reasoner — 朝Level-2迈步

### 5.1 架构 (Figure 3)

```
[Causal Question]
       ↓
       ↓
[RAG Retrieval: Contriever + Faiss] ← [General Knowledge DB (~16MB)]
       ↓
[Retrieved Knowledge]
       ↓
[Goal-oriented Prompt Template] ← [Causal Reasoning Goal]
       ↓
[LLM (LLaMA / GPT / Claude)]
       ↓
[Causal Answer]
```

两个module:
1. **General knowledge module**: RAG系统 [\[32\]](https://arxiv.org/abs/2005.11401) — 用Meta Contriever [\[24\]](https://arxiv.org/abs/2112.09118) 作retriever, Faiss [\[13\]](https://arxiv.org/abs/2401.08281) 作vector DB, 知识源是HuggingFace的[General-Knowledge dataset](https://huggingface.co/datasets/MuskumPillerum/General-Knowledge) (~16MB)
2. **Goal-driven prompt**: 类似human reasoning时"以目标为导向" — 数学证明时心中始终有target proposition, 几何证明时始终refer to三条axioms

### 5.2 与Eq.(2)的对应

RAG retrieval ≈ E_{C~P_C}的approximation — 通过retrieve相关external knowledge, 模拟marginalize over confounder C。但RAG只retrieve了top-k最相关knowledge, 是Monte Carlo approximation, 不是integral。

Goal-oriented prompt ≈ 用language诱导LLM做"intervention-style" reasoning, 即不只是observe P(Y|X, T), 而是explicitly问"在X=X_0的情况下, 目标Y是什么"。

### 5.3 主要实验结果 (Table 2)

完整结果表(EM metric):

| Benchmark | Method | LLaMA 2 | LLaMA 3 | GPT 3.5 | Claude 3 |
|-----------|--------|---------|---------|---------|----------|
| COPA | Vanilla | 0.752 | 0.937 | 0.948 | 0.991 |
| | CoT | 0.812 | 0.944 | 0.951 | 0.991 |
| | RAG | 0.757 | 0.912 | 0.936 | 0.990 |
| | G²-Reasoner | **0.813** | **0.948** | **0.953** | 0.990 |
| e-CARE | Vanilla | 0.684 | 0.778 | 0.814 | 0.861 |
| | G²-Reasoner | **0.701** | **0.779** | **0.821** | 0.849 |
| CausalNet | Vanilla | 0.673 | 0.857 | 0.897 | 0.933 |
| | G²-Reasoner | 0.681 | 0.855 | 0.898 | 0.929 |
| **C-E** | Vanilla | 0.616 | 0.715 | 0.732 | 0.758 |
| | G²-Reasoner | **0.642** | **0.718** | **0.746** | 0.758 |
| **C-H** | Vanilla | 0.565 | 0.652 | 0.671 | 0.692 |
| | G²-Reasoner | **0.582** | **0.658** | **0.693** | 0.696 |

几个striking observations:

1. **Performance单调下降随freshness**: COPA (≈99% Claude) → e-CARE (86%) → CausalNet (93%) → C-E (76%) → C-H (69%). 这本身就是Level-1 hypothesis的最强empirical evidence。

2. **CoT经常比Vanilla还差**: 在COPA上LLaMA 2 CoT 0.812 > Vanilla 0.752, 但CausalNet CoT 0.666 < Vanilla 0.673, C-H CoT 0.573略高于Vanilla但GPT 3.5 CoT 0.662 < Vanilla 0.670。这与 [\[40\]](https://arxiv.org/abs/2406.12920) 揭示的long-context failure一致 — chain越长, 越偏离initial goal。

3. **RAG单独通常不如Vanilla**: 例如e-CARE LLaMA 3 RAG 0.760 < Vanilla 0.778。这说明naive retrieve knowledge反而干扰LLM的内部reasoning — G²-Reasoner的优势来自RAG + goal prompt的组合。

4. **G²-Reasoner在fresh benchmark上提升最显著**: C-H上LLaMA 2从0.565→0.582 (+1.7%), GPT 3.5从0.671→0.693 (+2.2%)。在stale benchmark (COPA) 上提升几乎为0, 因为已经ceiling了。

### 5.4 CausalProbe-M结果与partial match insight

CausalProbe-M要求exact match时所有模型崩盘, 但用**partial match** (可以漏掉正确选项但不能选错) 时, GPT 3.5达75%, Claude 3达85%。最interesting的发现: **LLMs很少犯false positive错误** — 即一旦它claim某个选项是正确的causal relation, 它通常是对的; 它的问题在false negative (漏掉真正的causal relation)。

这暗示: LLM对causal relation的*识别*能力高于*完整发现*能力, 像一个保守的detector而非完整的reasoner。

---

## 6. Quality Control — Crowdsourcing验证

paper §H的quality control对结果可信度至关重要:

- 17名volunteers (硕士+, 英语流利)
- 选拔标准: 难度评分≤7, accuracy≥80%
- 13人通过 (Table 4)
- 260个CausalProbe-H样本分给3人/题
- 89.2% qualified (CausalProbe-HQ)
- 在CausalProbe-HQ上重新测试, 模型仍差 (Table 5: LLaMA 2 0.547, Claude 0.733)

这排除了"模型失败是因为benchmark本身有错"的alternative hypothesis, 增强结论的可信度。

---

## 7. Limitations & 你的视角可能关注的点

paper §A承认:
1. G²-Reasoner只是"一步", 没真正达到Level-2
2. 无法完全排除LLM见过"类似"事件 (虽然corpus是新的, 但conceptual content可能overlap)

**你可能会质疑的几个点:**

a) **Level-2定义模糊**: paper没给出可操作的Level-2判据。是不是所有"在新context上做对"就算Level-2? 那retrieval augmented model做的算吗? 这里有个hidden circularity。

b) **General knowledge base只有16MB**: 这是巨大bottleneck。paper自己承认用Wikipedia API能大幅提升, 但resource限制没做。这个scaling behavior值得专门研究 — RAG-based causal reasoning的performance vs knowledge base size的scaling law。

c) **Counterfactual的真正定义**: paper的CausalProbe-H引入"fake cause-effect pairs"作为counterfactual, 但Pearl [\[46\]](https://ftp.cs.ucla.edu/pub/stat_ser/r350.pdf) 意义上的counterfactual是 P(Y | do(X=x'), X=x_obs, Y=y_obs), 需要structural equation。CausalProbe-H的"counterfactual"更接近"plausibility check", 没到Pearl的层级。

d) **SCM formalization的可证伪性**: Eq.(2)是conceptual framework, 但paper没给出如何quantitatively estimate P_C或验证marginalization是否正确做了。这部分更多是inspiration, 不是quantitative model。

---

## 8. 与broader research landscape的连接

- **Causal parrots** [\[72\]](https://arxiv.org/abs/2305.07371): 同样结论, "LLMs talk causality but are not causal"
- **CLadder** [\[25\]](https://arxiv.org/abs/2310.04343): SCM-based benchmark, 测Pearl causal ladder
- **CRAB** [\[50\]](https://arxiv.org/abs/2311.04284): 真实事件causal reasoning benchmark
- **Counterfactual reasoning limits** [\[33\]](https://arxiv.org/abs/2305.16572), [\[66\]](https://arxiv.org/abs/2307.02477): LLM在hypothetical scenarios下表现差
- **Embers of autoregression** [\[42\]](https://arxiv.org/abs/2309.13638): 训练目标决定inductive bias, 与paper §4.1呼应
- **Long context failure** [\[40\]](https://arxiv.org/abs/2406.12920): transformers知道但不"说" — 解释了为什么CoT有时反而更差

更远连接:
- **Self-RAG** [\[3\]](https://arxiv.org/abs/2310.11511): G²-Reasoner的实现参考
- **Min-K% Prob** [\[53\]](https://arxiv.org/abs/2310.16789): membership inference attack, 量化freshness
- **Yule 1926** [\[71\]](https://www.jstor.org/stable/2341102): nonsense correlation in time-series, autoregression的统计起源

---

## 9. 给你的intuition building总结

1. **LLM = observational learner, 不是interventional reasoner**: autoregressive loss只能学 P(Y|X), 学不到 P(Y|do(X)). Level-2 reasoning需要interventional reasoning, 而autoregression的inductive bias阻止了这点。

2. **Knowledge memorization ≠ reasoning**: Performance gap between COPA (memorized)和CausalProbe (fresh)就是这个point的empirical manifestation。这个gap是Level-1 vs Level-2的proxy。

3. **SCM视角给了一个formal language**: 把"textual causal reasoning"重新cast为"conditioned collider + confounder marginalization"问题, 这让RAG-based method有了theoretical justification — retrieve knowledge ≈ marginalize over confounder C.

4. **G²-Reasoner是patch, 不是solution**: 真正的Level-2可能需要architectural change — 比如显式intervention operator, world model [\[20\]](https://arxiv.org/abs/2305.10601), 或causal representation learning [\[65\]](https://arxiv.org/abs/2410.15319)。

5. **Open question**: LLM能否通过scale跨越Level-1→Level-2? 还是必须改变architecture? paper的argument (autoregression不causal)暗示architecture change是必须的, 但community还在debate [\[42\]](https://arxiv.org/abs/2309.13638), [\[73\]](https://arxiv.org/abs/2304.05524).

---

## References & Links

- Paper GitHub: [https://github.com/Haoang97/CausalProbe-2024](https://github.com/Haoang97/CausalProbe-2024)
- CausalProbe corpora: BBC [link](https://www.bbc.com/), The Guardian [link](https://www.theguardian.com/)
- General Knowledge dataset: [HuggingFace](https://huggingface.co/datasets/MuskumPillerum/General-Knowledge)
- Contriever: [arXiv:2112.09118](https://arxiv.org/abs/2112.09118)
- Faiss: [arXiv:2401.08281](https://arxiv.org/abs/2401.08281)
- Min-K% Prob: [arXiv:2310.16789](https://arxiv.org/abs/2310.16789)
- Pearl's Causality: [book link](https://www.cambridge.org/core/books/causality/455346EBF0F5A6F0E8014D1DC2D5A4F5)
- Kahneman TFS: [Wikipedia](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow)
- Hume Treatise: [Project Gutenberg](https://www.gutenberg.org/files/4705/4705-h/4705-h.htm)
- Causal Parrots: [TMLR 2023](https://arxiv.org/abs/2305.07371)
- CLadder: [NeurIPS 2023](https://arxiv.org/abs/2310.04343)
- CRAB: [EMNLP 2023](https://aclanthology.org/2023.emnlp-main.471/)
- Embers of Autoregression: [arXiv:2309.13638](https://arxiv.org/abs/2309.13638)
- Self-RAG: [ICLR 2024](https://arxiv.org/abs/2310.11511)
- Reasoning with World Model: [EMNLP 2023](https://arxiv.org/abs/2305.10601)
- Counterfactual limits: [arXiv:2305.16572](https://arxiv.org/abs/2305.16572), [arXiv:2307.02477](https://arxiv.org/abs/2307.02477)
- Causality for LLMs survey: [arXiv:2403.09606](https://arxiv.org/abs/2403.09606), [arXiv:2410.15319](https://arxiv.org/abs/2410.15319)
- Yule 1926 nonsense correlations: [JSTOR](https://www.jstor.org/stable/2341102)
- LLaMA 2: [arXiv:2307.09288](https://arxiv.org/abs/2307.09288)
- Claude 3 family: [Anthropic](https://www.anthropic.com/news/claude-3-family)
- GPT-4: [arXiv:2303.08774](https://arxiv.org/abs/2303.08774)
- CausalQA: [COLING 2022](https://aclanthology.org/2022.coling-1.289/)
- e-CARE: [ACL 2022](https://aclanthology.org/2022.findings-acl.176/)
- COPA: [AAAI 2011](https://ojs.aaai.org/index.php/AAAI/article/view/7928)

如果你想进一步drill down, 我建议重点关注两个方向:(a) 如何把Pearl的do-calculus显式融入LLM decoding (interventional decoding); (b) RAG-based causal reasoning的knowledge base scaling law。这两个方向都是open, 且和你的micrograd / nanoGPT pedagogical风格能很好结合。
