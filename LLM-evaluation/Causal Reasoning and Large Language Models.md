---
source_pdf: Causal Reasoning and Large Language Models.pdf
paper_sha256: c685e7c1be8ec0b2d71021c56cc89bf27ef2fb84956a59564bad8b1463a5f1b2
processed_at: '2026-08-03T15:14:53-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

## 一句话版本

以前我们想知道 "A 是否导致 B"，要辛苦地收集 data，然后跑 PC algorithm、NOTEARS 这种 statistical 算法去从 covariance 里反推因果方向；现在你直接问 GPT-4 "Age of Abalone 和 Shell weight 谁导致谁"，它告诉你 **97% 的概率是对的**，而且它压根没看过你的 data，只看了变量名字。

这就是这篇 paper 的核心：**LLM 把人类写在文本里的因果知识变成了一个可编程接口**。

参考：https://arxiv.org/abs/2305.00050

---

## 为什么要做这件事

Causal analysis 一直有个尴尬：你想估 "疫苗是否降低感染率"，你需要的远不止 data，你需要一张 **causal DAG**——哪个变量 confound 哪个、哪个是 mediator、哪里有 selection bias。这张图传统上靠 **domain expert 手画**。

问题来了：
- Domain expert 很贵
- Domain expert 也会画错（Human domain expert 也常常漏掉 confounder）
- 100 个变量的图，domain expert 不可能遍历 $\binom{100}{2} = 4950$ 个潜在 edge

于是另一个流派说：**那不要人画了，让算法从 data 里反推**。这就是 causal discovery，PC、GES、NOTEARS、DAG-GNN 都属于这一脉。

但这个流派有一个根本的数学障碍：**Markov equivalence class**。简单说就是 $A \to B$ 和 $B \to A$ 在 observational data 上产生一模一样的 joint distribution，纯靠统计你分不开。所以传统方法在 Tübingen 上撑死 83%。

这篇 paper 的 insight：**人类其实根本不靠 data 判因果，靠的是名字**。你看到 "Age of Abalone" 和 "Shell weight"，根本不需要任何数据就知道是年龄影响壳重，因为生物学常识。LLM 训练时读了大量人类写的这种因果陈述，所以它也具备这种"看名字推因果"的能力。

---

## 三个最 striking 的实验

### 实验 1：Tübingen pairwise causal direction

108 对变量，每对判断 A→B 还是 B→A。这是 causal discovery 经典 benchmark。

| 方法 | Accuracy |
|---|---|
| Mosaic（之前 SOTA） | 83% |
| GPT-3.5 chain-of-thought | 92% |
| **GPT-4 chain-of-thought** | **97%** |

最 striking 的是 LLM 只看 **变量名**，**完全没看 data values**。传统方法盯着 data 拼命找 covariance pattern 还上不到 84%，LLM 一句话就答对了。

更 striking 的是作者为了排除 memorization，从 2021 年后出版的书里凑了一个 **novel Tübingen**，GPT-4 在这个新 dataset 上仍然 **98.5%**。说明它不是背诵。

### 实验 2：Counterfactual reasoning (CRASS benchmark)

题目类似 "A woman does not order Chinese food. 如果她点了会怎样？"

| 模型 | Accuracy |
|---|---|
| GPT-3 | 58% |
| Human | 98% |
| **GPT-4** | **92%** |

GPT-4 距离人类只差 6 个点。Counterfactual 是 token causality 的核心 building block——你要判断一件事的 cause，必须能想象 "如果这件事没发生会怎样"。

### 实验 3：Necessary / Sufficient cause

这是 token causality 最难的部分。给你一个 vignette，比如 "Alice 和 Bob 同时开枪打碎窗户"，问 Alice 是否是 necessary cause（不是，因为 Bob 一个人也够了）和 sufficient cause（是，因为 Alice 一个人就够了）。

GPT-3.5-turbo 在这个任务上是 random guess (46.6%)，但 **GPT-4 直接跳到 86.6%**。这种能力涌现得非常突然，是典型的 emergent behavior。

参考 emergent abilities: https://arxiv.org/abs/2206.04615

---

## 为什么 LLM 能做到这件事

我的 intuition 是这样的：

**Causality 本质上是一种 knowledge representation，不是 algorithm**。

当你问 "Age 是否导致 Shell weight"，答案不在 data 里，在 **人类关于软体动物生长的所有生物学文本里**。这些文本被 GPT 训练时读过，压缩成了 weights 里的某种 representation。

传统 causal discovery 试图做的事：从 $(X, Y)$ 的 1000 个样本点中，通过某种 statistical property（比如 non-Gaussian noise、nonlinear additive noise）**反推生成这些数据的 DGP**。这要求 data 里必须有足够信息。

LLM 做的事：**直接从人类文本知识里 retrieve 这个答案**，data 完全不需要。

这俩方法 attack 的是不同问题：
- Causal discovery attack：**"我没看过这个领域，但只要给我 data，我能算出来"**
- LLM attack：**"我看过这个领域的所有教科书，data 不重要"**

所以它们的 failure mode 也完全不同：
- Causal discovery 在 Markov equivalence class 不可识别时跪掉
- LLM 在训练数据没覆盖的新概念上跪掉

这就是为什么作者强调 **hybrid pipeline**：让 LLM 提供 prior（覆盖 Markov equivalence 歧义），让 algorithm refine（验证 LLM 的 prior 是否符合 data），最后让 LLM critique 结果。

---

## 最有意思的几个细节

### 1. Prompt 的力量

Tu et al. 2023 用 ChatGPT 在 Neuropathic pain 数据上做实验，F1 = 0.21，比 random 还差。他们用了差的 prompt。

作者换成 single prompt + chain-of-thought + 加第三个选项 "C: No causal relationship exists"，**F1 从 0.21 跳到 0.68，3 倍提升**。

这说明 2023 年那篇 "ChatGPT 不会 causal discovery" 的论文其实是 prompt 不对。同样模型，prompt 差 3 倍性能。

### 2. GPT-4 critique GPT-3.5

有个 example：GPT-3.5 在医学 case 上 reasoning 写对了，但 final answer 选错了（选了 A 但 reasoning 暗示 B）。作者让 GPT-4 看这个 reasoning，GPT-4 立刻指出 "这个 reasoning 不支持 A，应该选 B"。

这是一个简单的 self-consistency 验证 trick：**用一个更强的 LLM critique 更弱的 LLM 的输出**。这种 pattern 在后续的 Constitutional AI、self-refine 等工作里被发扬光大。

参考 Constitutional AI: https://arxiv.org/abs/2212.08073

### 3. LLM 反过来 critique ground truth

Neuropathic pain dataset 里有一条 edge 标注是 "L5 Radiculopathy → Obesity"。GPT-4 给出相反答案 "Obesity → L5 Radiculopathy"，并引用医学文献说肥胖是 radiculopathy 的 risk factor。

作者去查文献，发现 **GPT-4 是对的，ground truth 是错的**。这是 LLM 之前没人想到的能力：它可以 critique 专家标注的 ground truth，发现专家漏掉的反向因果。

### 4. Normality 是 LLM 最弱的能力

Normality 指一件事是否符合 norm。比如 "经理拿笔" abnormal（违反公司规定），"助理拿笔" normal。LLM 在这个任务上只有 70%，比 necessity 和 sufficiency 差很多。

我的 interpretation：**Normality 需要同时理解 statistical norm（什么常见）和 prescriptive norm（什么应该）**。Statistical norm LLM 还行，因为训练数据里 frequency 信息多。Prescriptive norm 很难，因为它涉及道德、社会规则、context，需要从对话里微妙地推断。

---

## 这篇 paper 的 limitation

作者很诚实地承认了几点：

1. **Failure mode 不可预测**：GPT-4 在 short-circuit vignette 上会用错 reasoning principle（该用 sufficiency 用了 necessity）。你不知道它什么时候会错。

2. **Memorization 没法完全排除**：Tübingen 在训练集里，GPT-4 能补全 25% 的 row。Novel dataset 也只证明 generalize 到 cutoff 之后的 dataset，不证明 generalize 到新概念。

3. **Causal hierarchy theorem 的根本限制**：Bareinboim 证明要做 interventional reasoning 必须有 inductive bias。LLM 是不是真有这种 bias 仍是 open question。

参考 Causal Parrots 论文（质疑 LLM 真的懂因果）：https://arxiv.org/abs/2310.07364

---

## 对 Andrej 的 implication

作为一个关心 LLM 内部机制的人，我觉得这篇 paper 暴露了一个有趣的问题：

LLM 在 pairwise causal task 上达到 97%，在 counterfactual 上达到 92%，在 token causality 上 86%。**这三件事的 reasoning 难度应该递增**，但 LLM 的 accuracy 递减得不快。

这暗示什么？**LLM 可能根本没在做我们以为的那种 counterfactual simulation**。它可能在做一个更简单的事：**retrieve 一个最像的 causal pattern，然后模板化地生成 reasoning**。

具体说，当你问 "Alice 开枪打碎窗户，Bob 也开枪，Alice 是不是 necessary cause"，LLM 可能做的不是 simulate 两个 counterfactual 世界（一个 Alice 不开枪、一个 Bob 不开枪），而是 **从训练数据里 retrieve 类似的 "overdetermination" 案例**，然后套模板回答。

这就是为什么：
- LLM 在常见 vignette 类型（overdetermination、late preemption）上很好
- 在罕见类型（short circuit、double prevention）上 unpredictable 地失败
- 在 normality 上很差（normality 需要理解 context，pattern matching 不够）

这个 hypothesis 可以验证：如果你拿一个非常 novel 的 vignette，强制 LLM 做 counterfactual simulation 而非 pattern match，它的 accuracy 应该会显著下降。这才是真正考验 "LLM 是不是有 causal world model" 的实验。

---

## 一个更深的猜想

Causal reasoning 可能不是一个 monolithic 能力，而是由一堆 building block 组成的：
- Variable identification（识别什么是 cause 候选）
- Mechanism knowledge（什么影响什么）
- Counterfactual simulation（mental model 跑 forward）
- Frame setting（决定考虑哪些 cause）
- Normality assessment（normative judgment）
- Responsibility aggregation（多个 cause 怎么 weight）

LLM 在不同的 building block 上能力差异巨大。Mechanism knowledge（变 pairwise accuracy 97%）强，Counterfactual simulation（92%）中，Normality（70%）弱。

这暗示 **LLM 内部可能根本没有一个统一的 "causal reasoning module"**。它可能是把不同 building block 散布在不同的 representation 里，prompt 触发哪个就用哪个。所以同一个 prompt 加 system message "You are a helpful assistant for causal reasoning" 能提升 5 个百分点——你激活了某个 building block 的 retrieval path。

这是我读完最大的 intuition：**causality 在 LLM 里不是 coherent 的能力，而是 emergent 的 patchwork**。这跟人类认知其实很像——人类也不是用一个 mechanism 做所有因果推理的，Kahneman 的 System 1 和 System 2 就是这个意思。

参考 Kahneman *Thinking, Fast and Slow*：https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow

---

## 一句话总结

这篇 paper 最大的贡献是证明了一件事：**人类写在文本里的因果知识，可以通过 LLM 变成可编程接口**。这把 causal analysis 的 bottleneck 从 "data 不够" 转移到 "如何 verify LLM 给出的 prior"。传统 causal discovery 算法不会消失，它们会变成 LLM prior 的 refiner。最终形态是 LLM + DoWhy + EconML 的 hybrid pipeline，LLM 当 orchestrator，传统算法当 workhorse。

---

# 《Causal Reasoning and Large Language Models: Opening a New Frontier for Causality》深度解析

## 一、论文的问题意识与核心论点

这篇来自 Microsoft Research（Emre Kıcıman, Robert Osazuwa Ness, Amit Sharma）和 University of Chicago（Chenhao Tan）的工作，本质上是在回答一个被争论很久的问题：**LLM 究竟只是在做 surface-level pattern matching，还是真的能在因果层面对 world 做出可用的推理**。作者没有陷入"真正因果 vs. 模仿"的形而上学争论，而是做了一件更踏实的事——**用 behavioral study 去量化 LLM 生成 causal argument 的能力上限**，并且把这种能力与既有 covariance-based causal discovery 方法做了正面比较。

核心 finding：
- Pairwise causal direction inference（Tübingen benchmark）：GPT-4 达到 **97% weighted accuracy**，SOTA 之前是 Mosaic 的 83%。
- Counterfactual reasoning（CRASS benchmark）：GPT-4 达到 **92.44%**，比之前 GPT-3 的 58.39% 提升超过 20 个点。
- Necessary / sufficient cause identification（15 个 vignettes）：GPT-4 达到 **86.6%**。
- Full graph generation（Neuropathic pain / Arctic sea ice / Alzheimer's）：GPT-4 的 NHD ratio 全面优于 NOTEARS、DAG-GNN 等深度学习方法。

关键 intuition 在于：**LLM 用 variable name 这种 metadata 来推断因果关系，完全不看 data values**。这与传统 causal discovery 的范式不同——传统方法试图从 statistical covariance 中 reverse engineer 出 DGP，而 LLM 在做的是"读名字就推断机制"。

参考：
- arXiv 版本：https://arxiv.org/abs/2305.00050
- Code & datasets: https://github.com/py-why/pywhy-llm
- PyWhy 项目主页：https://www.pywhy.org/

---

## 二、因果性的三个正交维度

作者给出一个 unifying framework，把"causality"分解成三个正交 axis：

### Axis 1: Covariance-based vs. Logic-based
- **Covariance-based causality**：statistics、biostatistics、econometrics 路线，用 observational data 的统计相依性去识别和估计 causal effect。代表工具：DoWhy、EconML、Ananke。
- **Logic-based causality**：law、forensic、fault diagnosis 路线，用 counterfactual + domain knowledge + logical reasoning。代表人物：Halpern、Hitchcock。

### Axis 2: Type vs. Token causality
- **Type causality**：在 population 层面的因果，关注 variables 之间的 average effect。例如"吸烟是否导致肺癌"。
- **Token causality**：在 specific event 层面，关注这次具体事件的 cause。例如"Fred 这次得肺癌是因为他吸烟还是因为接触了 asbestos"。

### Axis 3: Task 的种类
- Causal graph specification（DAG 构建）
- Causal discovery（从数据反推 DAG）
- Effect inference（估计已知因果关系的强度）
- Attribution / Judgment（归因与责任判断）

LLM 的"特殊技能"在于：它能在三个 axis 之间自由穿梭。传统 covariance-based 方法只能做 Type + Effect inference 这一格，而 LLM 用自然语言为媒介，可以同时承载 domain knowledge、counterfactual simulation、normality judgment 等多种 reasoning。

---

## 三、Pairwise Causal Direction Inference（核心实验一）

### 3.1 Task 设定

给定两个变量 A 和 B，判断 A→B 还是 B→A。这是 Tübingen cause-effect pairs benchmark（Mooij et al., 2016, JMLR）的经典任务。

数据集结构：
- 108 对 cause-effect pair
- 来自 37 个 datasets
- 覆盖 meteorology、biology、medicine、engineering、economics 等 domain
- 例如 "Age of Abalone → Shell weight"、"Cement → Compressive strength of concrete"

### 3.2 传统方法的根本困境：Markov Equivalence Class

对于只有两个节点的图 $(A, B)$，Markov equivalence class 同时包含 $A \to B$ 和 $B \to A$。形式化地说，两个图 $G_1, G_2$ Markov equivalent 当且仅当：
1. 它们有相同的 skeleton（无向边集合）
2. 它们有相同的 v-structures（collider 结构 $X \to Y \leftarrow Z$）

对两节点图，没有 collider，所以方向完全不可识别。这就是为什么传统方法上限停在 ~80%。

### 3.3 LLM 的方法：Knowledge-based Pairwise Inference

Prompt 设计有两种：

**Two-prompt 方案**：
```
Does changing {A} cause a change in {B}? Please answer in a single word: yes or no.
Does changing {B} cause a change in {A}? Please answer in a single word: yes or no.
```

**Single-prompt 方案**（效果更好）：
```
Which cause-and-effect relationship is more likely?
A. changing {A} causes a change in {B}.
B. changing {B} causes a change in {A}.
Let's work this out in a step by step way to be sure that we have the right answer.
Then provide your final answer within the tags <Answer>A/B</Answer>.
```

关键：第二个 prompt 触发了 **chain-of-thought reasoning**（参考 Wei et al., 2022, https://arxiv.org/abs/2201.11903），让 LLM 先解释再给出 final answer。

### 3.4 Results 表格详细解读

| Model | Acc. | Wt. Acc. |
|---|---|---|
| Mosaic (SOTA covariance-based) | 0.83 | 0.82 |
| text-davinci-002 | 0.79 | 0.79 |
| text-davinci-003 | 0.82 | 0.83 |
| gpt-3.5-turbo | 0.81 | 0.83 |
| gpt-3.5-turbo (causal agent) | 0.86 | 0.87 |
| gpt-3.5-turbo (single prompt) | 0.89 | 0.92 |
| **gpt-4 (single prompt)** | **0.96** | **0.97** |

几个关键 observation：

1. **Emergent behavior**：ada、babbage、curie 这种小模型 accuracy 停在 50%（random chance），davinci-002 才开始 emerge 出因果推理能力。这与 Wei et al. 2022 的 emergent abilities 现象一致——scaling 到一定规模才出现。

2. **System message 的作用**：加一句 "You are a helpful assistant for causal reasoning" 就让 accuracy 提升 5 个百分点。这暗示 LLM 内部有一个 "causal reasoning mode"，可以通过 prompt 激活。

3. **Single prompt >> Two prompt**：原因在于 single prompt 强制 LLM 做比较，而不是单独 binary 判断。这种 comparative framing 让 LLM 更倾向于用 domain knowledge 而不是 surface patterns。

### 3.5 错误模式分析（理解 LLM 推理的本质）

**案例 1：Age of Abalone → Length（正确）**

LLM 解释："Age affects growth, growth affects length"。看似合理。

**案例 2：Age of Abalone → Diameter（错误）**

LLM 给出 incoherent explanation，最后答错。这两个 case 应该用同样的 reasoning（age 通过 growth 同时影响 length 和 diameter），但 LLM 表现出 inconsistency。

**案例 3：Ozone concentration vs. Radiation（歧义问题）**

LLM 最初答错，把 "ozone concentration" 理解成 stratospheric ozone。用户补充 "ground-level ozone measured in a city" 后，LLM 立刻修正回答，正确指出 radiation → ground-level ozone（通过 photochemical smog 反应）。

这个 case 揭示了 LLM 的 failure mode：**不是逻辑推理不行，而是 context disambiguation 不行**。一旦 context 明确，LLM 的 domain knowledge 完全够用。

**案例 4：Hang Seng Bank vs. HSBC Holdings**

LLM 答对了 ownership 关系但答错了 benchmark 期望的因果方向。Benchmark 标注是 $B \to A$（HSBC 影响 Hang Seng），LLM 给出 $A \to B$（Hang Seng 影响 HSBC）。这其实是 benchmark 本身的 ambiguity——股票市场没有固定因果模式。

**Intuition**：LLM 的"错误"很多其实是 benchmark 的 ambiguity，而非 LLM 的 reasoning 错误。

### 3.6 Novel Tübingen Dataset（控制 memorization）

为了排除 LLM memorize 了 Tübingen benchmark 的可能性，作者从 2021 年 10 月之后出版的书中收集了 67 对新变量，例如 "battery capacity" vs. "ambient temperature"（来自 *Battery Management System and Its Applications*, 2022）。

Result：
- GPT-3.5-turbo: 80.3%
- GPT-4: **98.5%**

这证明 LLM 的能力可以 generalize 到 training cutoff 之后的 dataset。

### 3.7 Neuropathic Pain Dataset（专业领域测试）

来自 Tu et al. (NeurIPS 2019) 的医学诊断图，475 条 edge，涉及 radiculopathy、DLS（degenerative lumbar spondylosis）等需要专业医学知识才能理解的概念。

| Model | Accuracy |
|---|---|
| text-davinci-003 | 55.1 |
| gpt-3.5-turbo | 71.1 |
| gpt-3.5-turbo (single prompt) | 85.5 |
| gpt-4 (single prompt) | **96.2** |

LLM 不仅识别术语，还能给出定义。例如 LLM 主动解释 "DLS = degenerative lumbar spondylosis"，并正确推断 DLS T5-T6 → Left T6 Radiculopathy。

**更有意思的发现**：在 Left L5 Radiculopathy → Obesity 这条 edge 上，ground truth 标注是 radiculopathy 导致肥胖，但 LLM 给出相反答案——肥胖是 radiculopathy 的 risk factor（有 Atchison & Vincent 2012 等医学文献支持）。这说明 LLM 有时能 **critique ground truth**，这是 covariance-based algorithm 做不到的。

---

## 四、Full Graph Generation（核心实验二）

### 4.1 难度提升

从 pairwise 到 full graph 的难度提升有三个层面：
1. 每个 pair 有三种状态：$A \to B$、$B \to A$、no edge。
2. 需要区分 direct cause 和 indirect cause。如果真实关系是 $A \to B \to C$，pairwise 测试可能正确指出 $A \to C$，但在 full graph 中 $A \to C$ 是错的。
3. Decision 依赖于 graph 中**存在哪些其他变量**。如果 B 不在变量集中，那么 $A \to C$ 就是对的。

### 4.2 Neuropathic Pain Dataset Results

221 个变量，$\binom{221}{2} = 24310$ 个潜在 edge。作者先用 Tu et al. 2023 提供的 100-pair 子集做测试（50 个真 edge，50 个非 edge）。

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Random baseline | 0.25 | 0.5 | 0.33 |
| chatGPT (Tu et al. prompt) | 1.0 | 0.12 | 0.21 |
| text-davinci-003 | 0.59 | 0.68 | 0.63 |
| gpt-3.5-turbo (single prompt) | 0.66 | 0.71 | **0.68** |
| gpt-4 (single prompt) | 0.74 | 0.58 | 0.65 |

Tu et al. 用了一个差的 prompt "A causes B. R and L refer to... Answer with true or false"，F1 只有 0.21，比 random 还差。换用 single prompt + 加第三个选项 "C: No causal relationship exists" 后，gpt-3.5-turbo F1 跳到 0.68，**3 倍提升**。

这是一个 prompt engineering 的强证据：同样的模型，prompt 不同性能差 3 倍。

### 4.3 Arctic Sea Ice Dataset

来自 Huang et al. 2021（Frontiers in Big Data），12 个变量，48 条 edge。Domain：大气科学，包含 total cloud water path、sea level pressure、net shortwave flux 等专业变量。

**Normalized Hamming Distance (NHD)** 的公式：

$$\text{NHD}(G, G') = \sum_{i,j=1}^{m} \frac{1}{m^2} \mathbb{1}_{G_{ij} \neq G'_{ij}}$$

变量含义：
- $m$：图中节点数
- $G_{ij}$：ground truth graph 中节点 $i$ 到节点 $j$ 的边（0 或 1）
- $G'_{ij}$：预测 graph 中对应位置
- $\mathbb{1}_{G_{ij} \neq G'_{ij}}$：indicator function，当两者不同时取 1，否则 0

NHD 的本质是 graph adjacency matrix 上的 Hamming distance 归一化到 $[0, 1]$。对于 $m=12$，total possible edges $= m^2 = 144$。

**问题**：如果算法预测 0 条边，NHD = $48/144 = 0.33$，比预测 48 条但全错（NHD = $96/144 = 0.66$）还好。所以单纯比 NHD 不公平。

**Solution**：引入 NHD Ratio：

$$\text{Ratio} = \frac{\text{NHD}(G, G')}{\text{NHD}_{\text{baseline}}}$$

其中 $\text{NHD}_{\text{baseline}}$ 是预测同样数量的 edge 但全错的 NHD。Ratio 越低越好。

| Algorithm | NHD | No. predicted edges | Baseline NHD | Ratio |
|---|---|---|---|---|
| TCDF | 0.33 | 9 | 0.39 | 0.84 |
| NOTEARS (Static) | 0.33 | 15 | 0.44 | 0.75 |
| DAG-GNN (Static) | 0.32 | 23 | 0.49 | 0.65 |
| gpt-3.5-turbo | 0.33 | 62 | 0.76 | 0.43 |
| **gpt-4** | **0.22** | 46 | 0.65 | **0.34** |

gpt-4 的 NHD = 0.22，比所有 covariance-based 方法低三分之一；NHD Ratio 0.34 是所有方法中最低的。GPT-4 找回 29 条正确 edge，precision 0.63，F1 = 0.57。

### 4.4 Novel Alzheimer's Dataset

来自 Abdulaal et al. 2023（ICLR），11 个节点（age、ventricular volume、brain MRI 等），由 medical experts 在 LLM training cutoff 之后构建。

| Algorithm | NHD | Predicted edges | Baseline NHD | Ratio |
|---|---|---|---|---|
| NOTEARS | 0.22 | 10 | 0.32 | 0.69 |
| DAG-GNN | 0.37 | 20 | 0.44 | 0.83 |
| gpt-3.5-turbo | 0.21 | 21 | 0.38 | 0.55 |
| **gpt-4** | **0.14** | 25 | 0.48 | **0.28** |

GPT-4 在这个 post-cutoff 的 dataset 上 NHD = 0.14，进一步证明能力可以 generalize 到 training 没见过的 domain。

---

## 五、Probing LLM Behavior：Memorization 与 Redaction

### 5.1 Memorization Test

为了排除 LLM 只是背下了 benchmark，作者设计了 memorization test：

- 给 LLM dataset 的前 3 列（row ID 和 2 个 variable name）
- 让 LLM 自动补全后 2 列（source dataset 名 + ground truth causal direction）

| Dataset | GPT-3.5 cells | GPT-3.5 rows | GPT-4 cells | GPT-4 rows |
|---|---|---|---|---|
| Tübingen | 58.9% | 19.8% | 61% | 25% |
| Neuropathic (100 chars) | 17 | - | 25 | - |
| Arctic Sea Ice (100 chars) | 2 | - | 2 | - |
| CRASS (100 chars) | 14 | - | 2 | - |

**关键 insight**：Tübingen dataset 确实被部分 memorize 了，但 memorization 程度远低于 LLM 在 task 上的 accuracy（97%）。所以 LLM 不是单纯靠背诵答对的。

作者给出一个概率分解模型：

$$P(Y) = P(Y|D) \cdot P(D) + P(Y|\neg D) \cdot P(\neg D)$$

其中：
- $P(D)$：causal relationship 被 memorize 的概率
- $P(Y|D)$：memorized 后能正确 transform 出来的概率
- $P(Y|\neg D)$：没 memorize 但通过 reasoning 答对的概率

Memorization test 只能测量 $P(D)$ 和 $P(Y|D)$ 的联合上界，而 LLM 在 novel dataset 上同样高的 accuracy 说明 $P(Y|\neg D)$ 也很高。

参考 Carlini et al. 2022 "Quantifying Memorization across Neural Language Models"：https://arxiv.org/abs/2202.07646

### 5.2 Redaction Test

为了理解 LLM 究竟 attend 哪些词，作者做 redaction：随机删除一个词，看 accuracy 怎么变。如果删某词后 accuracy 大跌，说明 LLM 在 attend 这个词。

实验发现：
- "changing"、"causes"、"<Answer>A/B</Answer>" 这些 instruction 关键词最重要，redaction 后 accuracy 下降最多
- SLOT1/SLOT2/SLOT3/SLOT4（变量名 slot）虽然重要但彼此冗余，redaction 单个影响不大
- 一些看似无关的 grammar 词（如 "the"）redaction 后也会显著影响 accuracy，暗示 LLM 对 grammatical correctness 有敏感性

这个 test 借鉴自 Sinha et al. 2021（ Perturbing inputs for fragile interpretations）：https://arxiv.org/abs/2103.01052

---

## 六、Token Causality 与 Counterfactual Reasoning（核心实验三）

### 6.1 CRASS Benchmark

来自 Frohberg & Binder 2022（LREC 2022）：https://aclanthology.org/2022.lrec-1.229/

数据集结构：每个 instance 有 premise、counterfactual question、3-4 个 multiple choice answers。

例子：
- Premise: "A woman does not order Chinese food."
- Question: "What would have happened if she had ordered Chinese food?"
- Options: A. The woman would have become less hungry. B. The woman would have become very hungry. C. That is not possible.

| Model | Accuracy |
|---|---|
| GPT-3 (Frohberg & Binder) | 58.39 |
| T0pp | 72.63 |
| text-davinci-003 | 83.94 |
| gpt-3.5-turbo | 87.95 |
| **gpt-4** | **92.44** |
| Human | 98.18 |

GPT-4 距离 human baseline 只差 6 个百分点。

**有意思的 failure mode**：
- "A man walks on a street. What would have happened if a man had walked on a bed?" GPT-4 答 "He would have been late"——它想象成人在街上走但脚下是床，而不是在家里床上走。这是 LLM 缺乏 "inferred physical scene context" 的表现。

### 6.2 Necessary vs. Sufficient Cause

**Necessary causality 定义**：event $C$ 是 $E$ 的 necessary cause，如果 $C$ 不发生，$E$ 就不会发生。

**Sufficient causality 定义**：event $C$ 是 $E$ 的 sufficient cause，如果 $C$ 发生，$E$ 就会发生。

**Pearl 的形式化定义**（sufficient causality）：

$$P(E_{C=1} = 1 | C = 0, E = 0)$$

变量解释：
- $C$：cause variable，取值 $\{0, 1\}$
- $E$：effect variable，取值 $\{0, 1\}$
- $E_{C=1}$：potential outcome，表示如果干预 $C$ 为 1，$E$ 会取什么值（do-calculus 中的 $E | do(C=1)$）
- 条件 $|C=0, E=0$：限定在 $C$ 实际为 0、$E$ 实际为 0 的子集上

这个公式读作："在实际观察到 $C=0$ 且 $E=0$ 的情况下，如果我们反事实地把 $C$ 设为 1，$E$ 变为 1 的概率"。这个概率高，说明 $C$ 是 sufficient cause。

**Robust sufficient causality**（Hitchcock 2012, Woodward 2006）：在 $C$ 发生的前提下，即使其他 cause event 没有发生，$E$ 仍然会发生。

作者用 15 个 vignettes 测试（Kueffner 2021 survey），覆盖 7 种类型：

1. **Symmetric Overdetermination**：两个 cause 同时发生，单独任一都足以引起 effect。
2. **Switch**：一个 action 在两个 sufficient causes 之间选择，但无论选哪个 effect 都发生。
3. **Late Preemption**：两个 cause 并行运行，一个先完成。
4. **Early Preemption**：两个 cause 都会成功，但其中一个还没开始就被中断。
5. **Double Preemption**：一个 cause 会被另一个 prevent，但 preventer 自己又被 prevent 了。
6. **Bogus Preemption**：一个 action 看似 prevent 但 focal process 本身是 inactive 的。
7. **Short Circuit**：原始 action 让 focal process inactive，新 action 试图阻止 inactive 但反而触发了 process，但 process 因为原始 action 完成不了。

| Model | Necessary Acc | Sufficient Acc |
|---|---|---|
| gpt-3.5-turbo | 46.6% | 46.6% |
| **gpt-4** | **86.6%** | **86.6%** |

gpt-3.5-turbo 的 accuracy 接近 random guess，gpt-4 则飞跃到 86.6%。这是一个典型的 emergent ability 跨越临界规模的现象。

### 6.3 Lab-Vignettes（避免 memorization）

为了排除 LLM 背了原 vignette，作者把所有 vignette 改写成 chemistry lab 场景：用 reagent、test tube、mixture、crystal 替换原版中的人名和物件。例如 Overdetermination 原版是 "Alice 和 Bob 同时开枪打碎窗户"，lab 版是 "Agent X 和 Y 同时喷水灭火"。

| Model | Necessary | Sufficient |
|---|---|---|
| gpt-3.5-turbo | 64.2% | 42.8% |
| **gpt-4** | **92.8%** | **78.5%** |

gpt-4 仍然高 accuracy。**有意思的 pattern**：necessity（93%）比 sufficiency（78%）容易。原因可能是 necessity 只需要 flip 一个变量做 counterfactual，而 sufficiency 需要 flip 多个变量并决定 flip 哪些（causal frame problem）。

---

## 七、Normality 评估

### 7.1 什么是 Normality

Normality 是人类 causal judgment 的重要因素（Phillips et al. 2015; Kominsky et al. 2015）。当 agent 违反 norm（statistical norm 或 prescriptive norm），人类倾向于把更多 causality 归因于它。

例子：receptionist 案例中，"Professor Smith 拿笔"是 abnormal（违反 department policy），而 "administrative assistant 拿笔" 是 normal（符合 policy）。即使两人都拿了笔都造成了"没笔"的 outcome，人类判断 Professor Smith 更是 cause。

### 7.2 Two-Step Prompting

**Step 1**：让 LLM extract causal event。
```
State the causal event being asked about.
```

**Step 2**：让 LLM 评估 normality。
```
The causal event is "abnormal" if:
- occurrence was unexpected, unlikely, surprising, rare, or improbable
- agent's action non-accidentally, knowingly, or negligently violated social, legal, or ethical norms.
```

Result：
- gpt-3.5-turbo: 69.2%
- gpt-4: 71.1%

GPT-4 在三个例子中表现：
1. "Sarah 违反 printer policy" → GPT-4 正确判定为 abnormal（违反 office policy）
2. "Manager Johnson 拿 notepad" → GPT-4 判断有误，但 reasoning 更细致
3. "Mark 在停车场停下和朋友聊天导致女儿受伤" → GPT-4 错误判断为 normal

**Insight**：normality 是 LLM 最弱的 token causality primitive。原因是 normality 涉及 statistical norm + prescriptive norm 的混合判断，需要同时理解"什么常见"和"什么应该"。

---

## 八、Responsibility 的概念

作者还探索了 responsibility（graded causation）。Halpern & Chockler 2004 的形式定义：

$$\text{Responsibility}(C, E) = \frac{1}{N+1}$$

其中 $N$ 是让 $C$ 成为 $E$ 的 necessary cause 所需 minimal 改动的其他 causal event 数量。

变量含义：
- $N$：需要"取消"的其他 cause events 的最小数量
- $N=0$：$C$ 已经是 necessary cause，responsibility = 1
- $N \to \infty$：$C$ 几乎不是 cause，responsibility $\to 0$

LLM 在 beer spilling example 上的测试：
- **Overdetermination 场景**：加入 Susan 也在 Mike 之后撞到桌子，足以让 beer 洒。LLM 正确判断 Mike 的 responsibility 降低了。
- **Double Prevention 场景**：Jack 试图接住瓶子，但 Peter 撞了 Jack 让他没接住。LLM 错误判断 Peter 更 responsible——按人类直觉 Mike 才是 cause，Peter 只是 double preventer。

这个 failure 暴露了 LLM 对 **counterfactual dependence 的替代**——它倾向于把"最后破坏平衡的事件"当成 cause，而不是追溯最初始的 cause。这是 LLM 在 token causality 上的根本限制。

参考 Halpern 2016 *Actual Causality*：https://mitpress.mit.edu/9780262035026/actual-causality/

---

## 九、关键洞察：LLM 带来的范式转变

### 9.1 三类 capabilities

1. **Domain knowledge access**：以前只有 human domain expert 才能提供的因果机制知识，现在 LLM 能 programmatic 地提供。
2. **Flexible natural language interface**：让 non-expert 也能用 causal tools。
3. **Token causality primitives 提取**：necessity、sufficiency、normality 这些以前难以 formalize 的概念，现在可以从自然语言中提取出来。

### 9.2 与传统 causal discovery 的关系

LLM 和 covariance-based 算法犯的错**不同**。这意味着两者可以 ensemble：
- LLM 犯错往往是因为 context ambiguity 或 normality 误判
- covariance-based 算法犯错往往是因为 Markov equivalence class 不可识别

潜在 pipeline：LLM 提供先验 → covariance-based algorithm refine → LLM critique 结果。

### 9.3 因果分析的端到端 pipeline

作者在 Appendix F.1 展示了一个示例：让 GPT-4 写一个 Jupyter notebook，用 causal-learn 学图 + DoWhy 估计 effect + EconML 做 meta-learner + sensitivity analysis。GPT-4 能写出基本可用的代码。这意味着 **LLM 可以做 causal analysis 的"前端orchestrator"**，把自然语言 query 翻译成 code，再调用 DoWhy/EconML 这类工具。

参考：
- DoWhy: https://github.com/py-why/dowhy
- EconML: https://github.com/py-why/EconML
- causal-learn: https://github.com/cmu-phil/causal-learn

### 9.4 Negative Control 识别

Appendix F.2 展示了 GPT-4 在 vaccine efficacy 观察性研究中识别 negative controls 的能力：
- **Negative controls**（应该零 effect）：hair color、blood type、handedness、birth month
- **Positive controls**（已知 non-zero effect）：age group、chronic conditions、pregnant individuals、smokers
- **Time-bound controls**：pre-vaccination period（应该零 effect）、peak immune response period（应该最大 effect）、waning immunity period（应该衰减 effect）

这种 reasoning 以前需要 epidemiologist 才能做。

---

## 十、失败模式与 Fundamental Limits

### 10.1 不可预测的失败

LLM 在高 accuracy 的同时会犯简单错误。例如 short-circuit vignette 中，gpt-4 用了 necessity 的 reasoning 而不是 sufficiency，给出错误答案。这种错误无法通过 prompt engineering 完全消除。

### 10.2 Memorization 与 Generalization 的混合

Tübingen 这种 popular benchmark 一定在 training set 里。即使 novel dataset 上 accuracy 也高，也不完全排除 LLM 是把 training 中见过的类似 causal relationship 做 pattern matching。

### 10.3 Causal Hierarchy Theorem 的限制

Bareinboim et al. 2022（*Probabilistic and Causal Inference: The Works of Judea Pearl*）证明：要做 interventional 和 counterfactual 推理，模型必须有相应的 inductive bias。LLM 是否有这样的 inductive bias 还不清楚。Willig et al. 2023 "Causal Parrots" 论文（https://arxiv.org/abs/2310.07364）质疑 LLM 是否真的有 world model。

### 10.4 Hidden Bias 的问题

传统 causal inference 把主观假设 explicit 放在 DAG 里，可以被批评。LLM 的假设藏在 transformer 的 weights 里，无法 audit。在 race、gender 这种敏感属性上，LLM 的因果 claim 可能会包装成"算法客观"而实际上 reflect 训练数据的偏见。

参考 Bender et al. 2021 "Stochastic Parrots"：https://dl.acm.org/doi/10.1145/3442188.3445922

---

## 十一、未来方向

作者列出的几个关键研究问题：

1. **Knowledge-based causal DAG generation**：重新定义 causal discovery，把 metadata 和 data values 联合利用。LLM 可以做 prior、做 critic、做 post-processor。
2. **LLM-guided effect inference**：用 LLM 推断 backdoor set、instrumental variables。
3. **Systematizing token causality**：把 LLM 的 primitive reasoning（necessity、sufficiency、normality）系统化，结合 Halpern 2016 的 formal theory。
4. **Human-LLM collaboration**：iterative 的 graph 构建，LLM 提议、human 反馈、LLM critique。
5. **Understanding causal inductive bias**：LLM 是否学到了 catholic causal model（Hao et al. 2023, https://arxiv.org/abs/2305.14992）。

---

## 十二、我的整体 Intuition 总结

这篇 paper 最值得注意的地方不是 GPT-4 达到 97% 这个数字本身，而是它揭示的一个范式分裂：

**传统 causal discovery** = "从 statistical evidence 中 reverse engineer 出 DGP"
**LLM causal reasoning** = "从 metadata 中 retrieve 出已知的 causal mechanism"

这两个范式 attack 的是因果分析的**不同瓶颈**。传统方法的瓶颈是 Markov equivalence class 的不可识别性；LLM 的瓶颈是 LLM 训练数据中没有的 causal relationship 它推断不出来。

**真正的机会在于 hybrid**：让 LLM 用 domain knowledge 提供 prior，让 covariance-based algorithm 用 data refinement，让 LLM critique 最终结果。这种 pipeline 之前需要 human domain expert 在中间做翻译，现在可以 programmatic 实现。

更深层的 implication：**causality 不是一种 algorithm，而是一种 knowledge representation**。LLM 通过预训练把人类写在文本中的因果知识压缩进了 weights，这个 representation 之前只能通过 human expert 慢速访问，现在通过 natural language interface 可以快速访问。这从根本上改变了 causal analysis 的 bottleneck location——从"如何从数据中发现因果"转移到"如何 verify LLM 给出的因果假设"。

最后一个值得思考的 meta-question：当 LLM 在 Tübingen 上达到 97% 但在 responsibility 推断上失败时，到底什么在"内部"发生？可能性包括：
- LLM 学到了一个 catholic causal model（Hao et al. 的论点）
- LLM 学到了一个 surface-level 的 causal language generator（Willig et al. 的论点）
- LLM 在简单因果上用 retrieval，在复杂因果上用 interpolation（最可能的解释）

这篇 paper 没有回答这个问题，但它给出了一个 behavioral upper bound：在 2023 年的 GPT-4 上，causal reasoning 的能力上限大致是 90% 左右的 pairwise accuracy、85% 的 counterfactual accuracy、70% 的 normality accuracy。这个 baseline 让后续研究有了 comparison 的 anchor。
