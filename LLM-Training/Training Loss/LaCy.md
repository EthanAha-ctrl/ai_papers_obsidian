---
source_pdf: LaCy.pdf
paper_sha256: f7754c9c1726038176420359b5d5355cb487f2ad969d2ff410c5018534a20c6b
processed_at: '2026-08-05T11:38:12-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Hey Andrej, 让我们抛开 paper 里那些 academic 的包装，用最直白的 engineer 视角来过一遍这篇 paper 的 core idea。

想象我们在训练一个只有 334M 或 1.3B 的小模型 (SLM)。这个模型就像是容量极小的内存，强行往里塞 Wikipedia 的事实知识，它不仅记不住，还会导致参数被污染，推理时产生严重的 hallucination。

业界目前的解法是给它开个外挂：引入一个特殊的 `<CALL>` token。模型遇到不知道的词，就吐出 `<CALL>`，此时系统挂起，去请求一个更大的模型（比如 Llama 3.2 1B）或者 RAG 系统，拿到正确的词后填回去，小模型继续往下生成。这叫 model cascade。

这里最核心的工程难题是：**在 pretraining 阶段，我们怎么教小模型“什么时候该认怂去 `<CALL>`”？**

### 1. 纯靠 Loss 筛选 token 为什么会翻车

以前的做法（比如 Rho-1）很直觉：看 cross-entropy loss。如果某个 token 的 loss 一直降不下去，说明小模型学不会，那就把 ground truth 替换成 `<CALL>`，让它学怎么求助。

听起来没毛病，但这里有巨大的语义陷阱。假设 ground truth 是：“Turing was an English mathematician”。
*   **情况 A**：小模型预测成了 “scientist”。此时 loss 很高，因为 “scientist” 不等于 “mathematician”。但这句话在事实上没有任何错误！如果这时候触发 `<CALL>`，纯属浪费 cascade 的预算。
*   **情况 B**：小模型预测成了 “linguist”。此时 loss 同样很高，但这句完全错了，属于严重幻觉，这时候必须触发 `<CALL>`。

纯 loss 信号是一个无情的概率匹配机器，它根本分不清“语义上合理的同义词替换”和“灾难性的事实错误”。在 loss 的眼里，情况 A 和情况 B 的惩罚是一样的。如果仅凭高 loss 就去 `<CALL>`，模型会在大量无需求助的语法节点上疯狂呼叫，把 budget 全部耗光。

### 2. LaCy 的解法：引入轻量级 Symbolic Prior

既然纯靠 neural network 的 loss 信号有盲区，LaCy 的思路就是拉一个确定性的 symbolic engine 来帮忙——也就是 spaCy grammar parser。

作者用 spaCy 对 pretraining 数据进行预处理，把 token 分成两类：
*   **Factual tokens**：人名、地名、日期、机构名等首次出现的实体。这类 token 只有一个正确答案，预测错了就是事实错误。
*   **Non-factual tokens**：语法结构词、介词、或者是实体的重复出现（上下文已经有的词，小模型靠 in-context learning 就能抄过来）。

LaCy 的核心逻辑就是两步走：
1. 先用 spaCy 把 factual tokens 挑出来。
2. 在这些 factual tokens 里，再看小模型的 loss，挑出 loss 最高的那批（实验中是 top 60%），把它们替换成 `<CALL>` token 去训练。

用公式表达就是：
$$ C_{\mathrm{LaCy}}(x_i) = C_{\mathrm{spaCy}}(x_i) \cdot \mathbb{I}\Big[i \text{ is in the top } n\% \text{ of } \mathcal{L}(\mathcal{B}; \theta)\Big] $$

这里 $C_{\mathrm{spaCy}}(x_i)$ 是 spaCy 判断这个词是否是事实的 mask，$\mathcal{L}(\mathcal{B}; \theta)$ 是当前 batch 的 loss。两者相乘，意味着只对“且为事实，且模型真的猜不准”的 token 触发 `<CALL>`。

这样一来，像前面说的 “scientist” 替换 “mathematician” 的情况，因为 “scientist” 不是 spaCy 认定的事实实体，它就不会被纳入 `<CALL>` 候选，模型会正常学习这种语言泛化能力。而如果是预测爱因斯坦的出生年份，spaCy 判定它是 DATE 实体，且模型 loss 极高，此时就会稳稳触发 `<CALL>`。

### 3. 训练目标的重构

在具体的 loss 计算上，LaCy 做了非常精细的处理。
$$ \mathcal{L}_{\mathrm{LaCy}}(\mathrm{x}; \theta) = -\frac{1}{N} \sum_{i=1}^{N} \Big[ C_{\mathrm{LaCy}}(x_{i+1}) \log p(\mathrm{<CALL>} \mid x_{1:i}; \theta) + (1 - C_{\mathrm{LaCy}}(x_{i+1})) \log p_{\backslash \mathrm{<CALL>}}(x_{i+1} \mid x_{1:i}; \theta) \Big] $$

公式里的 $p_{\backslash \mathrm{<CALL>}}$ 非常关键。当模型在预测非 `<CALL>` 的正常 token 时，我们必须在 softmax 层把 `<CALL>` token 的 logit 强制设为 $-\infty$，然后重新归一化。这保证了模型在学习正常词汇分布时，概率质量不会被 `<CALL>` 这个特殊 token 稀释掉。

### 4. 违反直觉的实验现象：Loss 与 FactScore 脱钩

这篇 paper 最让 deep learning practitioner 细思极恐的发现是：在 token-selection 这种场景下，**Validation Loss 彻底失去了作为评估指标的意义**。

如果你看实验图表，LaCy 的 validation loss 非常高，比 baseline 差得多。按照传统的 scaling law 直觉，loss 越低模型越好，那 LaCy 简直就是垃圾。但到了下游的 FactScore 测试（生成传记并逐句验证事实准确性），LaCy 却比所有 baseline 都好。

为什么会这样？因为当你把目标 token 替换成 `<CALL>` 时，你其实改变了模型要拟合的数据分布。LaCy 刻意把那些最难的事实 token 拿出来让模型输出 `<CALL>`，模型在这些位置当然会有极高的 loss，因为它被强行训练去输出一个完全不同的 token。Loss 升高在这里是**设计使然**，代表模型成功学会了“我不知道这件事”。这也是为什么作者强烈呼吁，做 cascade 和 token deferral 的研究，千万别只盯着 loss 曲线看，一定要跑下游任务。

### 5. 延伸的直觉：SLM 正在演化为 Orchestrator

顺着这篇 paper 往深处想，SLM 的角色正在发生根本性转变。

以前我们认为 SLM 是一个被压缩的知识库。在这篇 paper 的架构下，SLM 正在变成一个 **Orchestrator（调度器）**。它的 parameter capacity 不再用来存储事实，而是用来学习：
1.  **Language fluency**：怎么把话说通顺。
2.  **Epistemic calibration**：判断“这个知识我内部有没有”，如果没有，准确触发路由机制。

这其实是一种将 LLM 的 **System 1 (直觉生成)** 和 **System 2 (知识检索)** 物理剥离的尝试。spaCy 在这里充当了一个非常粗糙但极其便宜的 System 2 守门员。相比于用 GPT-4o 去做 LLM Judge 来标注哪些 token 是事实（成本极高，需要 233 小时/A100/1B tokens），spaCy 只需要在 CPU 上跑，成本极低（152 小时/CPU/1B tokens），这让这种 Routing 机制真正具备了 scaling 到海量 pretraining 数据的可行性。

### References & Links

*   **LaCy 核心机制**: 结合 spaCy 和 loss，实现 token-level routing ([Paper](https://arxiv.org/abs/2505.15962))
*   **Loss 筛选的局限性**: Rho-1 完全依赖 loss 差值，容易在语义等价处浪费 budget ([Rho-1](https://openreview.net/forum?id=0NMzBwgaAJ))
*   **事实评估标准**: FactScore 将生成文本拆解为 atomic facts 进行验证 ([FactScore](https://aclanthology.org/2023.emnlp-main.741/))
*   **知识存储极限**: Allen-Zhu 的研究证明了 parameter size 对知识记忆的硬性上限 ([Physics of LLMs](https://arxiv.org/abs/2309.14316))
*   **LLM Judge 成本对比**: LaCy 用 CPU-based spaCy 替代 GPU-based LLM judge，大幅降低 routing 标注成本 ([spaCy](https://spacy.io))

---

这篇 paper 探讨了一个非常触及 Large Language Models (LLMs) 本质的问题：Small Language Models (SLMs) 在 pretraining 阶段，到底应该把哪些 knowledge 压缩进自己的 parameters 里，又该把哪些 knowledge 委托给外部的 tool 或者更大的 model。

对于 SLMs 而言，其 parameter capacity 极其有限（例如文中使用的 334M 和 1.3B GPT-2 architecture），强行记忆所有的 factual knowledge 会导致严重的 hallucination。当前的解决路径是引入一个特殊的 `<CALL>` token，当 SLM 发现自己无法预测某个 token 时，就输出 `<CALL>`，进而触发一个更大、更昂贵的 cascade model（如 Llama 3.2 1B 或 Qwen 3 32B + RAG）来接管生成。

核心的直觉构建在于：**Loss 并不是一个衡量 token 是否值得 SLM 学习的好指标。**

### 1. 为什么 Loss 具有欺骗性

传统的 learnability theory 以及像 Rho-1 这样的方法，倾向于使用 cross-entropy loss（或者 reference model 与 training model 的 loss 差值）来筛选 token。Loss 高，意味着 model 预测错了，所以不应该学。然而，这篇 paper 指出了这种纯 loss-driven 逻辑中的致命盲区。

作者引入了 **Acceptability** 的概念。Accuracy（预测 token 与 ground truth 严格匹配）与 Acceptability（预测 token 在语义和事实上与 ground truth 兼容）之间存在巨大的鸿沟。

*   **高 Loss 且 不可接受**：例如 "Alan Turing was an English mathematician"，model 预测成了 "linguist"。此时 loss 极高，且属于严重的 factual error，SLM 绝对应该 `<CALL>`。
*   **高 Loss 但 可接受**：例如 "Entre Campos station is part of the metro system"，ground truth 是 "Yellow"（黄线），model 预测成了 "metro"。此时 loss 依然很高，因为 "metro" 的概率与 "Yellow" 不匹配，但这句话在事实上完全成立。如果此时触发 `<CALL>`，不仅浪费了 cascade 的预算，还会让 SLM 错失学习通用语言模式的机会。

纯 loss 信号完全无法区分上述两种情况。为了 build your intuition，我们可以想象 loss 曲线在一个高维空间里寻找局部最优，它在 "Yellow" 和 "metro" 之间划下了一道鸿沟，但从人类 factual correctness 的视角来看，这两者本该是同一个盆地。

### 2. LaCy 的核心方法与公式解析

LaCy 的核心哲学是：结合 spaCy grammar parser 提供的 semantic/factual prior，与 loss 提供的 learnability signal。

首先，定义 spaCy 的 factuality mask。对于 batch $\mathcal{B}$ 中的第 $i$ 个 token $x_i$：
$$ C_{\mathrm{spaCy}}(x_i) : \mathcal{V} \to \{0, 1\} $$
这里 $\mathcal{V}$ 是 vocabulary，输出 $1$ 代表这是一个事实性 token（如人名、地名、日期的首次出现），输出 $0$ 代表是语法结构或非关键内容。

接着，定义 LaCy 的 call mask：
$$ C_{\mathrm{LaCy}}(x_i) = C_{\mathrm{spaCy}}(x_i) \cdot \mathbb{I}\Big[i \text{ is in the top } n\% \text{ of } \mathcal{L}(\mathcal{B}; \theta)\Big] $$
这里：
*   $C_{\mathrm{LaCy}}(x_i)$ 是最终的指示函数，决定该 token 是否被替换为 `<CALL>`。
*   $\mathbb{I}$ 是 indicator function。
*   $\mathcal{L}(\mathcal{B}; \theta)$ 是在当前 batch 内计算的 per-token loss。
*   $n\%$ 表示在事实性 token 中，只挑选 loss 最高的那部分。在实验中，为了与 baseline 保持相同的 call budget（15% 的 tokens），$n$ 被设定为 60%。这意味着 25% 的 spaCy fact tokens 中，取 loss 最高的 60%，刚好占总 tokens 的 15%。

修改后的 pretraining objective 为：
$$ \mathcal{L}_{\mathrm{LaCy}}(\mathrm{x}; \theta) = -\frac{1}{N} \sum_{i=1}^{N} \Big[ C_{\mathrm{LaCy}}(x_{i+1}) \log p(\mathrm{<CALL>} \mid x_{1:i}; \theta) + (1 - C_{\mathrm{LaCy}}(x_{i+1})) \log p_{\backslash \mathrm{<CALL>}}(x_{i+1} \mid x_{1:i}; \theta) \Big] $$
变量解析：
*   $N$ 是 sequence length。
*   $p_{\backslash \mathrm{<CALL>}}$ 表示在计算常规 token 的概率时，将 `<CALL>` token 的 logit 设为 $-\infty$，然后重新归一化。这保证了 model 在预测正常词时，不会因为 `<CALL>` 的存在而稀释概率质量。

### 3. 数据处理的精妙之处：First-mention Heuristic

作者使用 spaCy 的 `en_core_web_sm` model 进行 Named Entity Recognition (NER) 和 noun chunking。这里有一个极具直觉的工程设计：**只标注实体的首次出现为 fact token。**

为什么？因为 autoregressive language model 的特性。如果一段文本是 "Marie Curie discovered radium. Marie Curie was a physicist."。在预测第二个 "Marie Curie" 时，context 中已经包含了这两个词，SLM 完全可以通过 copy mechanism 或者 in-context learning 来预测，无需消耗宝贵的 parametric memory。将这种 token 排除在 fact 之外，极大提高了 data labeling 的信噪比。

### 4. 实验数据与架构图解析

实验主要在 GPT-2 334M 和 1.3B 上进行，数据集是 OLMo2 的 dwiki (Wikipedia subset, ~3B tokens)。Cascade partner 是 Llama 3.2 1B 或 Qwen 3 32B + RAG。

**Figure 2 & Table 解析:**
在 Biography Generation 任务中，使用 FactScore (Min et al., 2023) 评估。FactScore 会将生成的一长串文本拆解为 atomic facts，然后逐一验证其事实准确性。
*   **LaCy** 实现了最高的 FactScore（相对于 baseline 提升了 6.88%）。
*   在关掉 `<CALL>` 功能（将 call logit 设为 $-\infty$）并在 PopQA 和 BigBench QA 上测试时，LaCy 表现出最低的 **Fact Leakage**。这证明了 LaCy 确实在 pretraining 阶段阻止了事实性知识被强行压缩进 SLM 的 weights 中。

**Figure 4 解析:**
图中展示了关于 Errol Flynn 的传记生成。Baseline 产生了极其荒谬的幻觉（将 Errol Flynn 说成是美国电影的名字）。Rho-1 因为盲目依据 loss 来 call，导致它在很多无意义的 token 处触发了 call，且自身 retained 的事实存在严重错误（如出生日期和获奖情况）。LaCy 则精准地在 nationality, profession, dates 这些事实性节点触发了 `<CALL>`，从 Llama 3.2 1B 处获取了正确的 token，保持了生成文本的流畅性和准确性。

### 5. 反直觉的 Loss 与 FactScore 的脱钩

在 **Figure 7** 中，作者展示了一个令很多 deep learning practitioner 头疼的现象：Validation Loss 与下游 task 性能（FactScore）完全不相关。

*   Call Loss（在 `<CALL>` 位置的 loss）：LaCy 的 call loss 极高（5.72），远超 baseline。因为 LaCy 刻意挑选了那些 model 本来就猜不到的 hard fact tokens 去学习输出 `<CALL>`。
*   Non-call Loss（在正常生成 token 位置的 loss）：LaCy 略高于纯 loss-driven 方法。

如果仅仅看 validation loss 曲线，你会觉得 LaCy 是一个糟糕的 model。但是，它在下游 FactScore 上却取得了最好的成绩。直觉上，当我们改变了 target distribution（把难事实替换成了 `<CALL>`），validation loss 所评估的已经不是同一个数据分布了。将不同 token-selection 策略训练出的 model 直接比较 loss，犹如比较苹果和橘子。

### 6. 更广泛的联想与延伸

这篇 paper 触及了当前 LLM scaling 的一些核心痛点。Allen-Zhu 和 Li 在 "Physics of Language Models" 中提到，knowledge storage 在 parameter 达到一定阈值后会变得 lossy。Morris 等人 (2025) 也探讨了 memorization 的极限。

LaCy 的 approach 本质上是在做一个 **Hard Routing** 机制。与 Mixture of Experts (MoE) 不同，MoE 是在 model 内部将 hidden states 路由给不同的 FFN experts，而 LaCy 是在 token generation 级别，将 computational graph 路由给了完全不同的 model 甚至外部 RAG system。

这种 paradigm 下的 SLM 逐渐演变成了一个 **Orchestrator** 或 **Router**。它不需要自己知道答案，它只需要知道“什么时候自己不知道”。这与 Cohen 等人 (2024) 提出的 `[IDK]` token 在哲学上是一致的，但 LaCy 更进一步，它指出了单纯依靠 model 自身的 calibration（confidence score）是不够的，因为 LLMs 往往对错误的 fact 也表现出极高的 confidence。引入外部的、确定性的 symbolic engine（spaCy grammar parser）来提供 prior，是弥补 neural network calibration 缺陷的有效手段。

另一个有趣的联想是 **Memorization Sinks** (Ghosal et al., 2025)。如果我们在 pretraining 时把所有 facts 都 mask 掉只学习 `<CALL>`，model 的 capacity 全部用来学习 syntax 和 reasoning。这可能会导致 SLM 在 NLU (Natural Language Understanding) tasks 上表现更好。虽然 Table 1 显示在 ARC Easy, HellaSwag, PIQA, SIQA 上，LaCy 与 baseline 差异不大，但这说明事实性知识的 offloading 不会损害 reasoning capacity。如果我们把 model scale 进一步拉大，这种将 knowledge 与 reasoning 彻底解耦的架构，是否能催生一种全新的、极小但极具推理能力的 reasoning core？

### References & Links

*   LaCy paper context: 这是由 Apple 和 University of Cambridge 团队（Szilvia Ujváry, Michael Kirchhof 等）在 2026 年 2 月发布的预印本。
*   Rho-1 (Lin et al., 2024): https://openreview.net/forum?id=0NMzBwgaAJ (Loss-based token selection)
*   FactScore (Min et al., 2023): https://aclanthology.org/2023.emnlp-main.741/ (Fine-grained atomic evaluation of factual precision)
*   Physics of Language Models (Allen-Zhu & Li, 2024): https://arxiv.org/abs/2309.14316 (Knowledge storage limits)
*   Memorization Sinks (Ghosal et al., 2025): https://openreview.net/forum?id=sRJrMPu5Uu (Isolating memorization during training)
*   spaCy NLP toolkit (Honnibal et al., 2020): https://spacy.io (Used for grammar/fact parsing)
*   Toolformer (Schick et al., 2023): https://openreview.net/forum?id=Yacmpz84TH (Teaching models to use tools)
*   LMLM (Zhao et al., 2025): https://arxiv.org/abs/2505.15962 (Pre-training large memory language models)
