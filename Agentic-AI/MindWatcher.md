---
source_pdf: MindWatcher.pdf
paper_sha256: 32cec4ad2004016944b0572758e4cd6f4514692ae5c423c29dd4dfa2f8fc3dc4
processed_at: '2026-08-05T18:27:48-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# MindWatcher 大白话版

## 这篇paper到底在干啥

一句话：**一个32B的模型，靠"会调工具"+"会看图"，在多模态agent benchmark上干翻了GPT-5 mini和Gemini 2.5 Flash。**

但真正的价值不在这，而是它顺带挖出了几个让人不舒服的结论：RL训agent是有"基因天花板"的，benchmark分数很多时候在偷测模型的内部world knowledge，搜索引擎换个能差40分。

---

## 一、痛点是怎么来的

现在LLM不管吹得多猛，本质上被三件事卡死：

1. **Long-tail knowledge** —— 你问它"2025年11月18日和Nike续约的那个活塞队后卫叫啥"，它不知道
2. **Real-time information** —— 训练之后发生的事，它只能编
3. **Fine-grained domain knowledge** —— 看图认一辆冷门车、一棵稀奇植物，参数里根本没存

给模型挂工具是常规解法，但传统做法有坑：

- **Workflow agents**（像DSP/ConstrainedFlow那种）：流程写死，遇到开放域就崩，cross-modal更崩。参考 [Khattab et al., DSP, arXiv:2212.14024](https://arxiv.org/abs/2212.14024)
- **Multi-agent系统**（像HuggingGPT/AutoGen）：planner agent + tool agent分开部署，模型冗余、chained interaction慢。参考 [Shen et al., HuggingGPT, NeurIPS 2023](https://arxiv.org/abs/2303.17580), [Wu et al., AutoGen](https://arxiv.org/abs/2308.08155)
- **TIR agent**（像ReAct）：一个模型同时干planning和acting，end-to-end。但现有TIR agent几乎都是文本场景，视觉能力约等于零，就挂个image search。参考 [Yao et al., ReAct, ICLR 2022](https://arxiv.org/abs/2210.03629)

而且训TIR agent这件事本身是个灾难。SFT上去之后，模型学到的是"**模仿格式**"而不是"**学策略**"——简单题乱调工具，复杂题死循环。作者实测SFT会让base model吃很重的"alignment tax"，通用能力掉。

所以MindWatcher的结论很直接：**扔掉SFT，上纯RL**。

---

## 二、MindWatcher怎么把TIR agent做对的

### 2.1 Interleaved Thinking —— 不再是"先想后做"

传统ReAct是：`Thought → Action → Observation → Thought → ...` 这种串行。

MindWatcher把thought和tool call揉进同一个decoding序列里，通过 `` 和 `<tool_call>...</tool_call>` 这两个tag穿插。模型可以在任何一个step选择继续想还是切到调工具，反过来也行。

形式化成MDP。初始状态 $s_0$ 是用户prompt，trajectory是：

$$Y = \{a_0, obs_0, a_1, obs_1, \ldots, obs_{n-1}, a_n\}$$

- $a_j$：第 $j$ 个action（可能包含thinking和tool call两部分）
- $obs_j$：环境对 $a_j$ 的反馈
- $a_n$：终止action，含最终answer

统一action space：$\mathcal{A} = \mathcal{A}_{thought} \cup \mathcal{A}_{tool}$

Intuition：传统做法thought和tool call是分开的token stream，MindWatcher把它们当同一种token处理，policy $\pi_\theta(a_t | s_t)$ 就一个autoregressive loop，模型自己学会什么时候该停下来调工具。这种设计的好处是模型不会被"必须先想完再调"的格式绑架。

### 2.2 Multimodal CoT —— reasoning链里能操作图片

这是这篇paper最cinematic的部分。MindWatcher的reasoning chain可以长这样：

```

<tool_call>{"name": "zoom", "bbox": [186,145,637,356]}</tool_call>
<tool_response>[返回放大后的图]</tool_response>

<tool_call>{"name": "web_search", "query": "..."}</tool_call>
...
```

模型可以**在reasoning过程中裁剪、放大局部图、做visual search**。这就是"thinking with images"。参考 [Zheng et al., DeepEyes, arXiv:2505.14362](https://arxiv.org/abs/2505.14362) 也做了类似的事。

---

## 三、训练算法：Step-wise Normalized GRPO

这是这篇paper最该看的算法部分。

### 3.1 为什么标准GRPO在agentic场景下会崩

标准GRPO对一个group内G条trajectory，每条算sequence-level reward $r_i$，advantage做group-internal normalization：

$$\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r} \tag{2}$$

- $\mu_r$：G条trajectory reward的均值
- $\sigma_r$：G条trajectory reward的标准差

标准多轮agent的目标函数长这样（公式3）：

$$J(\theta) = \frac{1}{G} \sum_{i=1}^{G} \frac{1}{\sum_{j=0}^{n} |a_j|} \sum_{j=0}^{n} \sum_{t=T_j}^{T_j+|a_j|} \min\left[\frac{\pi_\theta(t|s_t)}{\pi_{\theta_{old}}(t|s_t)} \cdot A_{i,t}, \text{clip}\left(\frac{\pi_\theta(t|s_t)}{\pi_{\theta_{old}}(t|s_t)}, 1-\epsilon, 1+\epsilon\right) \cdot A_{i,t}\right]$$

变量：
- $G$：group size（一个prompt采样G条trajectory）
- $n$：trajectory中action segment数量
- $|a_j|$：第 $j$ 个action segment的token长度
- $T_j$：第 $j$ 个action segment的起始token位置
- $\pi_\theta$：当前policy
- $\pi_{\theta_{old}}$：旧policy（用于importance sampling ratio）
- $\epsilon$：PPO的clip范围

**问题在哪**：interleaved thinking下，一个trajectory里有多个"think+tool-call"cycle，每个cycle长度差异巨大。有的cycle就是一句"我需要搜一下"——10个token；有的cycle是长篇visual reasoning——500个token。直接按token求和，**长cycle会主导梯度**，短cycle的信号被淹没。

### 3.2 Step-wise Normalization的双层归一化

作者改了目标函数（公式4）：

$$J(\theta) = \frac{1}{G} \sum_{i=1}^{G} \frac{1}{n_i} \sum_{j=1}^{n_i} \frac{1}{|a_j|} \sum_{t \in a_j} \min\left[\frac{\pi_\theta(t|s_t)}{\pi_{\theta_{old}}(t|s_t)} \cdot \hat{A}_i, \text{clip}\left(\frac{\pi_\theta(t|s_t)}{\pi_{\theta_{old}}(t|s_t)}, 1-\epsilon, 1+\epsilon\right) \cdot \hat{A}_i\right]$$

变量：
- $n_i$：第 $i$ 条trajectory里action segment的数量
- $|a_j|$：第 $j$ 个action segment的token长度
- $\hat{A}_i$：trajectory-level advantage（用公式2算的）

两层normalization的意思：

1. **Action-Step Normalization** $\left(\frac{1}{n_i}\right)$：每条trajectory权重相等，不管它有几个cycle。一个trajectory有3个cycle和有10个cycle，权重一样。
2. **Token-Length Normalization** $\left(\frac{1}{|a_j|}\right)$：每个action segment内部按token平均。短cycle和长cycle的梯度贡献被拉平。

Intuition：相当于给每个"想+调"的cycle等权投票权，不让长reasoning段独占梯度。这是agentic RL里很关键的一个trick，对应ARPO [Dong et al., arXiv:2507.19849](https://arxiv.org/abs/2507.19849) 用entropy balancing防训练collapse、LLDS [Deng et al., arXiv:2512.04220](https://arxiv.org/abs/2512.04220) 用likelihood preservation防lazy likelihood displacement，思路类似——都在解决agentic RL训练不稳定。

### 3.3 Hybrid Reward三层结构

最终reward（公式8）：

$$R_{total} = R_{acc} + \lambda_{fmt} \cdot R_{fmt} + \lambda_{halluc} \cdot R_{halluc}$$

作者用 $\lambda_{fmt} = 0.1$，$\lambda_{halluc} = 0.05$。

**① Outcome Accuracy Reward $R_{acc}$（公式5）**

$$R_{acc} = \begin{cases} 1.0 & \text{if Judge returns "1"} \\ 0.0 & \text{if Judge returns "0"} \end{cases}$$

为什么不用正则：开放式multimodal QA答案太发散，"45,610"和"45610"、"NYC"和"New York City"得算对。作者用LLM-as-Judge，judge prompt会强制做unit conversion、precision tolerance、cross-language alignment。

**② Format Reward $R_{fmt}$（公式6）**

$$R_{fmt} = \begin{cases} 0.5 & \text{strictly follows schema} \\ -0.5 - 0.01 \times \text{len(residue)} & \text{format error or residue detected} \end{cases}$$

关键：tag外面有任何非空白字符都要罚，按残留长度乘0.01。为什么这么严？作者实测这种"chitchat残留"（比如 `<tool_call>...</tool_call> 我现在去搜一下`）训练后期会导致output collapse。这是过拟合格式漂移的常见failure mode。

**③ Hallucination Tool-call Penalty $R_{halluc}$（公式7）**

$$R_{halluc} = \min(0, (N_{resp} - N_{call}) \times 0.2)$$

变量：
- $N_{call}$：模型生成的 `<tool_call>` 数量
- $N_{resp}$：环境实际返回的 `<tool_response>` 数量

这个penalty只罚不奖（$\min(0, \cdot)$）。Intuition：模型有时候不等environment反馈，连续生成两个 `<tool_call>`，相当于"幻觉"了第一个的执行结果。这个penalty强制turn-taking协议。

---

## 四、工具集与本地检索库

### 4.1 五个核心工具

| Tool | Input | Output | 用途 |
|---|---|---|---|
| Region Cropping/Zooming | image + bbox | new image | 局部放大、聚焦关键区域 |
| Object Grounding & Visual Search | image + bbox + category | type name + confidence | 在MWRD里检索相似实体 |
| External Text Retrieval | text query | top-10 results | 网页搜索，返回title+abstract |
| Webpage Content Extraction | url + window + goal | structured result | 用Jina抓网页，可全文/窗口/AI摘要三种模式 |
| Local Code Interpreter | python code | stdout/stderr/result | sandbox执行，不能联网 |

### 4.2 MWRD本地检索库

外部visual search API贵且不准（互联网图噪太多）。作者自建：

- **8大类**：Person, Car, Plant, Animal, Logo, Landmark, Fruit & Vegetable, Dish
- **50k entities**，每个entity配3-10张高质量图，共**300k+ images**
- **precision > 99%**：domain expert人工filtering
- 定期维护更新

Intuition：训练时如果用Google Vision API之类的外部检索，调用一次几秒、几分钱，跑大规模RL训练成本爆炸。本地库把visual search变成一个低延迟、零成本的internal tool，RL才能跑起来。

---

## 五、数据构造两条pipeline

### 5.1 Private Images Pipeline

三阶段：

**Phase 1：Source Knowledge Annotation + Generation**
- 用object localization + fine-grained retrieval，从源图里抽bbox和retrieval label，建立image-text mapping
- 基于visual label做web search构建dynamic knowledge graph
- 用knowledge graph生成初始QA pair

**Phase 2：Timeliness + Uniqueness Verification**
两个坑要填：
- **Temporal Stability**：search engine环境动态变化，数据生成到训练之间有时间差，answer会drift
- **Answer Uniqueness**：开放式问题答案不唯一，reward model难判
- human-in-the-loop两阶段审核

**Phase 3：Difficulty Grading based on Tool Invocation**

这个设计很巧妙。作者发现**主观难度和agent实际难度脱节**——人类觉得难的记忆题，搜索引擎一搜就出来了，对agent反而简单。

所以Tool-Invocation Screening Engine放弃主观判断，用两个量化指标定义难度：
- **tool invocation rounds数**（要调几次工具）
- **multi-tool combination复杂度**（怎么组合工具）

这样构造的curriculum learning数据真正贴合agent学习曲线。

### 5.2 Sports News Pipeline

为什么选sports：objectively verifiable（比分有唯一真值）、resistance to ambiguity（统计事实不受主观污染）、multimodal richness（球员jersey、比分牌、动作镜头）。

三阶段：

**Phase 1：Domain-Specific Ingestion + Filtering**
- focused crawler抓权威sports portals
- 启发式filter丢掉空body和没图的样本

**Phase 2：LLM-Based Semantic Auditing**

"Data Auditor" agent做feasibility check：
- 保留：completed event + clear timeline + 实体动作视觉对应
- 拒绝：rumors/predictions/gossip/vague summaries
- 约40%被过滤

**Phase 3：Constraint-Aware QA Generation**

三条强制约束：
- **Temporal Anchoring**：强制把"yesterday"解析成绝对时间戳（如"2025年11月18日"），避免"data rot"
- **Visual-Textual Dependency**：问题里不直接说球员名，而是说"右边的8号球衣球员"，强迫agent先visual识别再search
- **De-referencing Context**：禁止"根据文章"这种meta-reference，agent只看到standalone问题+图，逼它用search tool拿文章里原本的信息

### 5.3 训练数据规模

- Online RL（真实internet环境）：1,639（private images）+ 2,949（sports news）+ 5,000（open-source: WebSailor [Li et al.](https://arxiv.org/abs/2507.02592), Tool-Star [Dong et al.](https://arxiv.org/abs/2505.16410), SimpleDeepSearcher [Sun et al.](https://arxiv.org/abs/2505.16834)）
- Offline RL：约20,000 samples

---

## 六、MWE-Bench构造

6类共1416 instances：Car 373, Animal 351, Plant 397, Person 63, Landmark 90, Sports 142。

关键：**与训练集严格不overlap**。private images用internal database里training set之外的entity；sports用和训练数据**完全不同时间点**的事件合并corpus，再用强LLM抽atomic facts构造复杂查询。

---

## 七、实验结果解读

### 7.1 主结果（Table 1）

**Direct Inference下**：
- Qwen3-VL 32B Thinking只有22.60，GPT-4o 27.75，Gemini 2.5 Pro 42.09 SOTA
- 关键insight：**knowledge cutoff新旧和性能不线性相关**。Qwen3-VL最新但分数低，因为它参数里没存够长尾知识。

**ReAct/Agent模式下**：
- Qwen3-VL 32B从22.60 → 66.95（3倍）
- GPT-5 mini在Sports从13.38 → 80.28（6倍）
- MindWatcher-32B：**75.35 SOTA**，Car/Animal/Plant/Person四项最高

**蒸馏小模型**：
- MindWatcher-2B：64.76
- MindWatcher-3B：64.48
- MindWatcher-4B：69.63
- 都接近或超过Qwen3-VL 32B Thinking的66.95

Intuition：小模型配上好的tool-use能力，可以大幅弥补参数知识不足。这是agent范式对小模型的杠杆效应。

### 7.2 跨benchmark结果（Table 2）

- MMSearch：MindWatcher-32B 58.82 SOTA
- SimpleVQA：MindWatcher-32B超过Qwen3-VL 32B base
- WebWalkerQA：纯文本benchmark，MindWatcher保持竞争力，证明multimodal agentic RL没损害文本reasoning

### 7.3 蒸馏效果（Table 3）

MindWatcher-3B最戏剧：base是Qwen2.5-VL-3B-Instruct（24.93），蒸馏后64.48，涨了近40分。

---

## 八、三个关键实验发现

### 8.1 Tool Capacity主导性能（Table 4）

同一个agent用三个搜索引擎跑sports数据：

| Engine | Basketball-EN | Basketball-CN | Football-EN | Football-CN | Avg |
|---|---|---|---|---|---|
| Sogou | 2.53 | 15.19 | 3.57 | 12.5 | 8.51 |
| Bing | 13.92 | 20.25 | 8.93 | 23.21 | 16.66 |
| Quark | 20.25 | 39.24 | 28.57 | **55.36** | 34.81 |

中文足球Quark比Sogou高42.86%。**工具capacity的variance经常盖过算法优化和模型scale的variance**。

Implication：benchmark必须把tool-induced variance算进去，不然你在测的不是模型reasoning能力，是搜索引擎的indexing质量。

### 8.2 Genetic Inheritance in Agentic RL（Figure 4）—— 最重磅发现

作者比较MindWatcher-32B（RL训的）和它的base Qwen2.5-VL-32B，以及GPT-5 mini，按tool-call round数分桶看accuracy。

**发现一：Decision Trigger Boundary差异**

GPT-5 mini在Round 0（不调工具）的样本里，accuracy只有51.2%。它**盲目自信**，该调工具时不调，开局就丢分。Round 1之后accuracy很稳，说明它的execution能力没问题，是被"什么时候该调工具"的决策卡住的。

MindWatcher的Round 0比例更低，触发工具的阈值更合理。

**发现二：Performance Shadowing / Genetic Inheritance**

MindWatcher和Qwen2.5-VL-32B在不同tool-call round上的accuracy曲线**下降斜率几乎一致**。sample分布也几乎没偏移。

含义：**RL能精修tool-invocation和reasoning proficiency，但突破不了base model的长程推理和多模态处理天花板**。foundation model对RL-derived agent施加了fundamental performance constraint。

作者把这叫"Genetic Constraint"。RL是策略优化器，但本质上和base model能力强耦合。

### 8.3 World Knowledge污染benchmark（Case Study 1）

那道Manuela Sáenz的题：
- MindWatcher（base是Qwen2.5-VL）：内部知识不认识Manuela Sáenz，没有起点，没法写精确query，搜不出来
- Qwen3-VL 32B Thinking：内部知识认识Manuela Sáenz，能据此写精确query，搜出来答对

两个模型TIR能力可能差不多，但benchmark分数被base model的long-tail world knowledge污染了。当tools不足以弥补知识gap时，benchmark本质上在测foundation model的内部知识。

---

## 九、Infrastructure（Appendix A.2）

基于Verl [Sheng et al.](https://arxiv.org/abs/2409.19256) 的step-wise synchronous sampling框架：

1. **vLLM并行inference生成action**
2. **synchronized barrier收集environment feedback**
3. **瓶颈不是rollout本身**，是tool-call latency
4. **异步tool调用层**：asyncio + semaphore concurrency control，heterogeneous tools并行dispatch，受API QPS约束
5. **Tokenization Offloading**：environment observation的tokenization从master node卸到分布式CPU workers
6. **LLM-as-Judge异步调用**：trajectory一完成就触发judge

这是"step级同步、tool级异步"的hybrid架构，最大化硬件利用率。

---

## 十、Appendix A.3的延伸：SFT也有Genetic Inheritance

蒸馏出的MindWatcher-2B/3B/4B和它们base模型对比：
- SFT后的tool-calling frequency分布**不稳定**，不像RL那样和base保持一致
- 但accuracy曲线**仍然和base同步下滑**，说明SFT也突破不了base的认知天花板
- 区别：SFT的曲线斜率不如RL那么"parallel"，说明SFT引入更多noise，optimization不如RL系统化

---

## 十一、几条值得带走的intuition

1. **SFT训agent是格式模仿，不是策略学习**。简单题乱调工具、复杂题死循环是SFT的典型failure mode。MindWatcher直接放弃SFT上纯RL。

2. **Interleaved thinking下，action segment长度方差极大**，标准GRPO的token-level sum会让长segment主导梯度。Step-wise Normalization用dual normalization（trajectory内action-step + action-segment内token）拉平梯度贡献。

3. **Agentic RL的reward必须三层**：outcome（事实对错）+ format（schema严格性，残留字符要罚）+ hallucination（不env反馈就连调工具要罚）。format和hallucination的penalty是为了防训练后期output collapse和turn-taking漂移。

4. **Tool capacity的variance经常盖过模型scale的variance**。换个搜索引擎分数差40分，这意味着agent benchmark必须报告tool-induced variance，不然测的是搜索引擎不是模型。

5. **Genetic Inheritance是agentic RL的天花板**。RL能精修策略，但无法突破base model的长程推理和多模态处理能力上限。base model的accuracy decay斜率会被RL-derived agent"继承"。要突破得换更强的base，光靠RL不够。

6. **Benchmark分数被world knowledge污染**。当tools不足以弥补知识gap时，benchmark本质在测foundation model的long-tail knowledge。比较两个agent的TIR能力，得控制base model的内部知识变量。

7. **小模型+好工具能力可以大幅弥补参数知识不足**。MindWatcher-3B从24.93涨到64.48，蒸馏比从头训base更高效。

8. **Multimodal CoT的关键不是挂个image search，而是让模型在reasoning链里直接操作图片**（裁剪、放大、grounding），把visual perception externalize到reasoning过程中。

---

## 十二、可能的延伸方向

- Genetic Inheritance这个现象如果成立，agentic RL的scaling law研究就得重新想——scale base model和scale RL effort的marginal return怎么分配？
- MWRD本地库这种"agent专属检索库"会不会成为新的infrastructure层？类似RAG但更agent-native。
- Step-wise Normalization能不能推广到所有multi-turn RL场景？比如code agent、browser agent里tool call之间也长度方差巨大。
- Hallucination Tool-call Penalty的0.2系数和 $\lambda_{halluc}=0.05$ 怎么选的？论文没给ablation，值得自己复现时sweep一下。

相关工作的arXiv链接：
- [ReAct](https://arxiv.org/abs/2210.03629)
- [DeepEyes](https://arxiv.org/abs/2505.14362)
- [WebWatcher](https://arxiv.org/abs/2508.05748)
- [ARPO](https://arxiv.org/abs/2507.19849)
- [LLDS](https://arxiv.org/abs/2512.04220)
- [WebSailor](https://arxiv.org/abs/2507.02592)
- [Verl/HybridFlow](https://arxiv.org/abs/2409.19256)
- [OpenAI o3/o4-mini](https://openai.com/index/introducing-o3-and-o4-mini/)

---

# MindWatcher 论文深度解析

## 一、论文整体定位与核心贡献

MindWatcher 来自 Li Auto Inc 的 MindGPT-ov Team，是一篇关于 **Tool-Integrated Reasoning (TIR) agent** 的工作。论文的核心 contribution 可以拆解为五个层面：

1. **Interleaved Thinking + Multimodal CoT 范式** —— 让模型在 reasoning 任意中间阶段灵活切换 thinking 和 tool calling
2. **Step-wise Normalized GRPO 算法** —— 针对 agentic RL 中 action segment 长度差异巨大的问题做 dual normalization
3. **Hybrid Reward 设计** —— Outcome + Format + Hallucination Penalty 三层 reward
4. **MWRD 本地多模态检索库** —— 50k entities, 300k+ images, 8 大类，precision > 99%
5. **MWE-Bench** —— 6 类共 1416 instances 的 multimodal agent benchmark

并发现了一个非常重要的现象：**Genetic Inheritance in Agentic RL** —— RL 无法突破 foundation model 的认知天花板。

项目链接：
- GitHub: https://github.com/TIMMY-CHAN/MindWatcher
- HuggingFace: https://huggingface.co/datasets/Lost-Cloud/MWE-Bench

---

## 二、Working Paradigm：MDP 形式化与 Interleaved Thinking

### 2.1 MDP 建模

MindWatcher 将 TIR 过程建模为 Markov Decision Process (MDP)。给定初始 prompt $s_0$，agent 与 environment 交互生成 interleaved 序列：

$$Y = \{a_0, obs_0, a_1, obs_1, \ldots, obs_{n-1}, a_n\} \tag{1}$$

变量含义：
- $a_j$：第 $j$ 个 action，通常包含 thinking 和 tool call
- $obs_j$：environment 对 $a_j$ 的 observation 反馈
- $n$：trajectory 中 action 的总数
- $a_n$：终止 action，包含最终 answer

关键设计在于 **unified action space**：

$$\mathcal{A} = \mathcal{A}_{thought} \cup \mathcal{A}_{tool}$$

这打破了传统 ReAct 中 thought 和 action 分离的设计，让 ` `
