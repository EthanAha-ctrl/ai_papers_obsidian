---
source_pdf: RethinkingMemoryMechanismsofFoundationAgents intheSecondHalf.pdf
paper_sha256: e54ac435051c5a4ca6dd003023b4830d3e77768712f9b42a008f97ddb59c5340
processed_at: '2026-08-11T23:45:56-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说这篇paper

## 一句话version

AI的"上半场"大家都忙着刷benchmark分数，现在分数刷得差不多了，发现真用起来还是不行 — 因为真实场景下agent要跟人长期互动、跨session记住事情、处理爆炸的context，**memory这个东西就变成了核心瓶颈**。这篇survey就是把2023到2025年所有关于agent memory的工作拉过来，按三个维度（存什么形式、怎么工作、给谁用）整理了一遍。

---

## 为什么memory突然火了？

先说intuition — 你训练一个大模型，它本质上是个stateless function。你给它输入，它给你输出，输入一变它就忘了之前发生什么。

这个在"上半场"没问题，因为benchmark都是这样的：给你一道数学题，你答完就完事。MMLU、HumanEval、SWE-bench都是single-shot的。

但real world不是这样：

- ChatGPT你要跟它聊几个月，它得记住你是谁、你喜欢什么
- 一个coding agent要帮你debug一个project，得记住之前试过什么、哪些方案失败了
- 一个web agent要帮你订机票，得记住你的偏好、常旅客号、上次订的日期

所有这些场景，**context window都不够用**。你不可能把几百轮对话全塞进prompt里。所以就需要某种memory机制 — 往外存，等需要的时候再拿回来。

这篇paper就是说：好，现在这个领域paper几百篇了，咱们得梳理一下到底大家在干什么。

---

## 三个维度看memory

作者提了个taxonomy，三个正交维度：

### 维度1：存什么形式

**External memory** — 存在model外面，用的时候再retrieve进来

最常见的就是RAG那一套 — 把文本切块、embedding、存vector database，query的时候做ANN search拿top-k回来塞进prompt。

除了vector，还有几种变体：
- 直接存text文件（比如MemGPT就是搞了个"main context + external context"的OS-style管理）
- 存成graph（knowledge graph，nodes是entity，edges是relation）
- 存成tree（RAPTOR那种递归summarize建hierarchy）
- 分层store（不同类型信息存不同module — persona放一块、episodes放一块、skills放一块）

**Internal memory** — 直接存在model内部

三种：
- **Weights**: 通过pre-training/post-training把knowledge编进参数里。好处是recall极快，坏处是update cost极高，还有catastrophic forgetting风险。Model editing（ROME、MEMIT那些）就是想局部改几个fact但别把别的搞坏。
- **Latent state**: 复用hidden states。Transformer-XL那种，把上一段segment的activation传给下一段。但这是临时的，session结束就没了。
- **KV cache**: 这个大家都熟，decoding的时候存历史token的K和V避免重算。但线性增长，长context显存爆。所以有一堆压缩工作（H2O、SnapKV、PyramidKV），核心idea就是发现attention score其实集中在少数"heavy hitter" tokens上，大部分token可以丢掉。

实战中大家往往hybrid用 — parametric memory存稳定的general knowledge，external memory存dynamic的experience，KV cache做fast short-term。

### 维度2：memory怎么工作

借用人脑的5种memory类型作为conceptual scaffold：

**Sensory memory** — 极短时感知buffer

text agent里基本不显式建模。但在multimodal/embodied agent里很重要 — 比如video agent要缓存最近几秒的frame embedding，不然每帧都重新encode太慢。

**Working memory** — 当前的在线工作区

在LLM agent里就是context window + KV cache本身。你要在固定budget下维持task-relevant state。这个的challenge是：context会爆，但naively截断又怕丢重要信息。所以有一堆"什么时候压缩、压缩成什么样"的工作。

**Episodic memory** — 具体经历

"上次你订机票的时候，我们试了三个网站，第二个网站报错了" — 这种带时间、带情境的具体event记录。

Stanford那个Generative Agents（smallville小镇）就用了这个 — 每个event存成带timestamp的entry，retrieval的时候综合recency、importance、relevance打分。

**Semantic memory** — 抽象knowledge

"用户喜欢靠窗座位" 这种从多次episodic经历里蒸馏出来的稳定fact。

从episodic到semantic是个abstraction过程 — 把具体经历aggregate成通用知识。HippoRAG用knowledge graph做这个，A-Mem用LLM自己判断怎么update。

**Procedural memory** — 可复用的skill

"search → read → extract → cite"这种workflow，一旦学会就能复用。

Voyager（Minecraft那个）是经典例子 — 把成功行为编成executable JavaScript code存进skill library，下次遇到类似场景直接调。

这5种之间的进化路径很有意思：早期都是explicit text templates，慢慢往implicit parametric neural policy演化。就是说，以前是LLM读一段skill描述然后照着做，现在是把skill直接bake进weights让它"本能地"会做。

### 维度3：给谁用

这个维度是这篇paper的创新点，之前的survey都没明确区分。

**User-centric memory** — 为用户个性化服务

记住这个具体用户是谁、喜欢什么、之前聊过什么。核心目标：让同一个agent对不同用户表现出不同behavior，随用户preference演化而adapt。

应用场景：long-term personalization、persistent user simulation、privacy-preserving memory。

**Agent-centric memory** — 为agent自我进化服务

Agent自己在执行task过程中积累的经验，跟具体用户无关。比如一个coding agentdebug了100个bug后，它应该记住"这种pattern的bug通常是因为XX原因"，下次遇到类似bug直接能解决。

应用场景：long-horizon tasks、domain-specific long-tail、cross-task knowledge transfer、strategy/skill learning。

这俩optimization goal完全不同 — user-centric优化individual user satisfaction，agent-centric优化cross-user generalization。一个medical assistant既要记住你这个病人的病史（user-centric），又要积累看1000个病人的诊疗经验（agent-centric）。

---

## Memory怎么操作

### 单agent的5个核心operation

1. **Storage & Index**: 写入时打标签（embedding、timestamp、task_id）
2. **Loading & Retrieval**: 需要时根据query拿相关的回来。Trade-off是拿太多引入noise，拿太少miss关键信息。
3. **Update & Refresh**: 发现新信息跟旧的矛盾了，要能改。比如用户说"我换工作了"，之前存的"用户在Google工作"要update。
4. **Compression & Summarization**: 把细碎episodic logs压缩成compact的semantic summary，减少storage + 提高检索效率。
5. **Forgetting & Retention**: 决定什么该忘什么该留。Ebbinghaus遗忘曲线那个公式 $R(t) = e^{-t/S}$ 就是这里用的，教育场景会simulate知识decay。

### 多agent系统的memory

这里复杂多了，因为多个agent要共享/隔离memory。

**4种架构**：
- **Private-only**: 每个agent各存各的。隐私好但浪费。
- **Shared-workspace**: 大家共享一个pool。省但容易污染。
- **Hybrid**: 分private和shared两层，用policy决定新信息往哪写。这是最实用的。
- **Orchestrated**: 有个central controller统一调度。ChatDev那种。

**3种routing方式**：
- Orchestrator-based: 中心决策
- Agent-initiated: 每个agent自己决定读什么
- Memory-driven: 直接retrieval决定

**Conflict resolution**：
两个agent写矛盾信息怎么办？要么write-time control（只允许一个memory manager agent改），要么iterative feedback loop慢慢收敛。

---

## 怎么学memory操作策略

这是paper我觉得最有意思的部分 — 三种paradigm：

### Prompt-based（不训练）

把memory policy写进prompt。比如"每5轮对话summarize一次history"、"importance score低于3的memory删掉"。

好处：不训练、可解释、好调。坏处：没credit assignment，policy不会自己变好。

### Fine-tuning（SFT）

把memory policy internalize到weights里。比如训练模型让它学会"什么时候该触发retrieval"、"query怎么rewrite"。

比prompt-based稳定，但还是static的 — 训完就固定了。

### Reinforcement Learning（RL）

这个是frontier — 把memory operation本身当action，用task reward倒推优化。

比如Memory-R1定义了4个atomic action: ADD、UPDATE、DELETE、NOOP。Agent观察当前memory state + 用户query，决定执行哪个action。用RL训练，最终task performance作为reward。

MEM1更激进 — 学一个policy让agent在long-horizon任务里memory usage保持near-constant，该compress就compress，该discard就discard。

这个方向的intuition是：**memory management本身是个decision-making问题**，不是heuristic能搞定的。你要balance short-term need vs long-term utility，这个trade-off得learn。

---

## Scaling问题

Real world的context explosion有三个axis：

1. **Interaction horizon**: 对话越久、tool-use轮次越多，累积的context越大。ReAct-style agent每step都append reasoning trace + tool output，线性甚至指数增长。

2. **Environment complexity**: 真实环境有各种structured artifact（API response、file、database），naive flatten成token既慢又丢结构。

3. **Environment quantity**: 跨多个environment协作。比如OSWorld里GUI agent要操作browser、file viewer、news app，每个environment有自己的state，memory要能作为environment之间的interface。

---

## Evaluation

这里paper有个很全的table。两类benchmark：

**User-centric**: 15个，包括LongMemEval（500 questions，500K conversation），LoCoMo、MemoryBank等。评估指标主要是retrieval accuracy、preference following、memory integrity。

**Agent-centric**: 27个，包括WebArena、OSWorld、SWE-bench等。评估success rate、pass rate。

一个关键观察：**大部分benchmark还是静态的、短horizon的**。真正能评估"跨月级别的preference drift"、"contradictory information over time"的benchmark还没有。这是个大gap。

---

## Applications

12个domain：education、scientific research、gaming、robotics、healthcare、dialogue、workflow automation、software engineering、streaming/recommendation、information search、finance、legal。

每个domain的memory用法都不一样。比如：
- Education: memory作为学生的cognitive digital twin，跟踪知识掌握度，模拟Ebbinghaus遗忘曲线
- Software engineering: memory存global code context + 失败trajectory，方便multi-file debug
- Legal: 每条memory都要attach provenance，因为hallucinated memory在法律场景是致命的

---

## 6个Future Direction

1. **Continual learning + self-evolving**: 现有continual learning只防forgetting，agent memory要track evolving state + procedural behavior。

2. **Multi-human-agent memory organization**: 当前multi-agent是episodic的，task结束就reset。需要collaborative social memory — 记住合作者的preferences、feedback pattern。

3. **Memory infrastructure & efficiency**: 当前text-centric memory token开销太大。方向是organized text → compressed latent → internalized parametric，memory size往constant收敛。

4. **Life-long personalization + trustworthy**: 长期personalization的挑战是staleness、concept drift、credit assignment。另外memory module容易遭extraction attack，privacy是deployment blocker。

5. **Multimodal + embodied + world model**: 把memory升级成explicit predictive world model — memory不是passive log，是controllable internal state，跟action一起被优化。DreamerV3那个方向。

6. **Real-world benchmarking**: 现有benchmark假设stationary user intent、reset-centric task、short horizon。需要closed-loop、longitudinal、execution-grounded的benchmark。

---

## 我的几个intuition

**1. Memory本质上是个context engineering问题**

上半场我们的战场是"怎么造更好的model"，下半场是"怎么管理agent自己的context across time"。这就像从"造更快的CPU"到"设计更好的操作系统"的转变 — model是CPU，memory是RAM + disk + filesystem，agent framework是OS。

**2. 三种substrate对应三个时间尺度**

- KV cache: millisecond级，单次intra-session
- Latent state: second到minute级，单session内
- External memory: hour到year级，cross-session
- Parametric memory: permanent，cross-lifetime

一个完整agent system应该hybrid用all of them。

**3. RL for memory是个big deal**

Prompt-based是cheap heuristic，SFT是static internalization，RL是真正能learn long-term credit assignment的。Memory operation本身是sequential decision making，heuristic rule搞不定所有edge case。MEM1和Memory-R1是早期实验，效果已经很显著 — agent能在超长horizon下保持near-constant memory usage，这个很反直觉。

**4. User-centric vs agent-centric的区分是个conceptual breakthrough**

之前所有survey都把memory当一个monolithic概念。这篇明确区分"为用户个性化"和"为agent自我进化"，optimization goal完全不同。一个medical assistant既要记住你这个病人的病史（user-centric），又要积累看1000个病人的诊疗经验（agent-centric）。这俩memory的update policy、retention policy、privacy requirement都不一样。

**5. Evaluation是真正的bottleneck**

没有好benchmark，再fancy的memory design也无法证明自己的utility。LongMemEval虽然有500 questions，但还是静态的。真正能评估"agent在3个月期间如何adapt to用户preference drift"的benchmark还不存在。这个gap谁来填谁就是next big thing。

**6. Memory + World Model的synergy是long-term bet**

Section 9.5的idea是 — memory不是被动存东西，而是active的predictive model，能simulate"如果我存这个信息，未来会有什么consequence"。这个把memory operation和planning统一了。DreamerV3、RWKV、Mamba那一脉的工作都在往这个方向收敛。如果做成了，agent的memory就不是"存什么"的问题，而是"world model里的state representation是什么"的问题。

---

## 一句话总结

这篇survey的贡献不是发明新东西，是给一个飞速发展但缺乏common vocabulary的field建立了taxonomy。接下来几年大家讨论agent memory会自然用substrate/cognitive/subject这套语言，就像现在讨论transformer用encoder/decoder/attention一样。这是field成熟的标志。

---

# 这篇Paper讲什么 — Rethinking Memory Mechanisms of Foundation Agents

## 总体Intuition

这篇survey的核心论点是 **AI进入了"下半场"** ：benchmark score不再是瓶颈，real-world utility才是。上半场的success recipe是train bigger model + push benchmark score (MMLU > 90%, MATH接近饱和)；下半场的challenges是long-horizon、dynamic、user-dependent environment下的context explosion。**Memory是连接理想benchmark performance和real-world utility的关键桥梁**，因为static one-shot capability已经不够，agent必须accumulate、retain、selectively reuse information across extended interactions。

作者提出了三维度的unified taxonomy：
- **Memory Substrate**: 信息存什么形式
- **Memory Cognitive Mechanism**: memory如何function（5种atomic type）
- **Memory Subject**: memory为谁服务

下面我从底层细节逐步build intuition。

---

## 1. Memory Substrates — 存什么形式

### 1.1 External Memory

External memory的核心特征是 **计算和存储分离** — LLM weights负责推理，外部database负责存储。这种设计支持scalable、easy-to-update、cross-session retention，避免expensive retraining。Trade-off：retrieval latency + retrieval noise。

**四种实现**：

#### Vector Index (RAG-style)
用embedding把memory items映射到高维向量空间，做approximate nearest-neighbor (ANN) search。检索公式：

$$\text{score}(q, m_i) = \cos(\mathbf{e}(q), \mathbf{e}(m_i)) = \frac{\mathbf{e}(q) \cdot \mathbf{e}(m_i)}{\|\mathbf{e}(q)\| \cdot \|\mathbf{e}(m_i)\|}$$

其中 $\mathbf{e}(\cdot): \mathcal{X} \to \mathbb{R}^d$ 是embedding function（如text-embedding-3-large, $d=3072$），$q$ 是query，$m_i$ 是第 $i$ 个memory item。

主流索引结构：
- **HNSW (Hierarchical Navigable Small World)**: 多层graph，每层的连接密度递减。Query复杂度 $O(\log N)$，比暴力 $O(N)$ 快很多。参考 [HNSW paper](https://arxiv.org/abs/1603.09320)
- **IVF (Inverted File)**: k-means聚类，query只搜最近的 $n_{probe}$ 个cluster
- **PQ (Product Quantization)**: 把 $d$ 维向量切成 $m$ 个子向量，每个子向量独立quantize，压缩存储

代表项目：[Chroma](https://www.trychroma.com/), [Pinecone](https://www.pinecone.io/), [Faiss](https://github.com/facebookresearch/faiss), [Weaviate](https://weaviate.io/)

#### Text-record Memory
持久化的human-readable text artifacts。典型设计：**core summary + episodic logs + semantic list**。MemGPT ([paper](https://arxiv.org/abs/2310.08560), [repo](https://github.com/cpacker/MemGPT))是代表，把memory抽象成OS-style的main context + external context，用LLM自己管理page in/out。

Pros：transparency, fast integration；Cons：summarization lossy，需要careful pruning。

#### Structural Store
三个子类：

**Relational tables**: SQL-backed, 用joins和indexes做高效cross-table retrieval。适合structured facts。

**Graph-based memory**: knowledge graph，nodes=entities, edges=relations。新信息到来时通过embedding similarity + keyword search + traversal检测edge changes, resolve conflicts。代表：[HippoRAG](https://arxiv.org/abs/2405.14831) — 模仿海马体的pattern separation + completion:

$$\text{retrieve}(q) = \text{PageRank}(\text{PPR}(q, G), \alpha=0.15)$$

其中 $G$ 是knowledge graph, PPR是Personalized PageRank，从query对应的seed node扩散。

**Tree-based memory**: hierarchical, 每个node存aggregated summary + embedding。代表：[RAPTOR](https://arxiv.org/abs/2401.18059) — recursive abstractive summarization构建tree，retrieval时可以选择不同abstraction level。

#### Hierarchical Store
多个专用memory module，由metamemory manager协调。代表：[Mem-α](https://arxiv.org/abs/2509.25911), [MIRIX](https://arxiv.org/abs/2507.07957)。典型配置：
- **core memory**: persistent persona + user facts (几KB)
- **episodic memory**: timestamped events (几十MB)
- **semantic memory**: abstract concepts (MB级)
- **procedural memory**: workflow JSON
- **resource memory**: documents + media
- **knowledge store**: credentials, with access control

### 1.2 Internal Memory

Internal memory直接住在model内部，speed最快，但update cost高。

#### Weights (Parametric Memory)
通过pre-training、post-training、model editing写入参数。三种策略：

**Continual Learning**: 用regularization避免catastrophic forgetting。[EWC (Elastic Weight Consolidation)](https://arxiv.org/abs/1612.00796):

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{new}}(\theta) + \lambda \sum_i F_i (\theta_i - \theta_i^*)^2$$

其中：
- $\mathcal{L}_{\text{new}}$ 是新任务loss
- $F_i$ 是Fisher information matrix第 $i$ 个对角元素：$F_i = \mathbb{E}\left[\left(\frac{\partial \log p}{\partial \theta_i}\right)^2\right]$
- $\theta_i^*$ 是旧任务最优参数
- $\lambda$ 是retention vs plasticity的trade-off

**Model Editing**: 局部更新参数。[ROME](https://arxiv.org/abs/2202.05262) 用rank-one update改transformer的mid-layer MLP：
$$\Lambda \leftarrow \Lambda + \frac{(v^* - W k^*) k^{*T}}{k^{*T} C^{-1} k^*}$$
其中 $\Lambda$ 是down-projection weight, $v^*$ 是desired output, $k^*$ 是key vector, $C$ 是key的covariance。[MEMKIT](https://arxiv.org/abs/2310.08560), [MEMIT](https://arxiv.org/abs/2210.07229) 扩展到thousands of edits。

**Distillation**: 把prompt/contextual behavior压进weights。如 [MemoryLLM](https://arxiv.org/abs/2402.04324)。

#### Latent-State Memory
复用hidden states across segments。代表 [Transformer-XL](https://arxiv.org/abs/1901.02860):

$$\mathbf{h}_{\tau+1}^{(l)} = f\left(\mathbf{h}_{\tau+1}^{(l-1)}, \text{sg}(\mathbf{h}_{\tau}^{(l)})\right)$$

其中 $\tau$ 是segment index, $l$ 是layer, $\text{sg}(\cdot)$ 是stop-gradient。这个state memory不是durable的，session结束就reset。

更现代的设计：[Titans](https://arxiv.org/abs/2501.00663) 把memory写成test-time neural network的weight update。

#### KV Cache
Transformer decoding的核心优化。Self-attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中 $Q \in \mathbb{R}^{n \times d_k}$, $K \in \mathbb{R}^{n \times d_k}$, $V \in \mathbb{R}^{n \times d_v}$, $n$ 是sequence length, $d_k$ 是key维度。

KV cache避免每生成一个新token就重算所有历史tokens的 $K, V$。但是cache线性增长，长context下显存爆掉。

**压缩方法**：
- **[H2O (Heavy-Hitter Oracle)](https://arxiv.org/abs/2306.14048)**: 观察到attention score集中在少数"heavy hitter" tokens，保留这些 + 最近的tokens。压缩比 ~10x。
- **[SnapKV](https://arxiv.org/abs/2404.14469)**: 在pre-fill阶段提前识别important key vectors
- **[PyramidKV](https://arxiv.org/abs/2406.02069)**: 不同layer保留不同数量的KV，浅层多、深层少

### 1.3 Substrate Trade-offs

| Substrate | Access Speed | Scalability | Update Cost | Persistence | Cross-session |
|-----------|-------------|-------------|-------------|-------------|---------------|
| Vector Index | 慢 (ANN search) | 高 | 低 (just insert) | 永久 | Yes |
| Weights | 极快 | 低 | 极高 (retraining) | 永久 | Yes |
| Latent-State | 极快 | 低 | 低 | 临时 | No |
| KV Cache | 极快 | 中 (linear) | 低 | 临时 | No |
| Graph | 中 | 中 | 中 | 永久 | Yes |

实战往往用 **hybrid**：parametric memory存general knowledge，latent memory做fast short-term reasoning，external memory做scalable experience storage。

---

## 2. Memory Cognitive Mechanisms — 怎么function

参考[Baddeley (2020)](https://www.cambridge.org/core/books/working-memory/)的工作memory模型和[Tulving (1972)](https://psycnet.apa.org/record/1973-08037-000)的episodic vs semantic distinction，paper定义5种atomic cognitive type。

### 2.1 Sensory Memory (短暂感知buffer)

原始感知信号的短暂保留。在text agent中通常不显式建模，但在multimodal/embodied agent中很关键。代表实现：
- 视觉：保留最近2-5秒的frame embedding
- 听频：audio buffer平滑perception
- 代表：[SAM 2](https://arxiv.org/abs/2408.00714), [ReSurgSAM2](https://arxiv.org/abs/2503.10214)

### 2.2 Working Memory (在线工作区)

短期存储 + active manipulation。在LLM agent里就是 **context window本身** + KV cache。Capacity是hard constraint。

两条research line：
- **Pre-write shaping**: 压缩、folding、abstraction在写入前就做。如 [ACON](https://arxiv.org/abs/2510.00615)
- **Post-write management**: 在固定budget下动态更新、evict。如 [MEM1](https://arxiv.org/abs/2506.15841) 用RL学习哪些信息保留，达到near-constant memory usage。

### 2.3 Episodic Memory (具体经历)

时间context下的具体events。结构化存储：trajectory + timestamp + environment context。

$$\text{episode}_i = (\text{state}_{i,0}, \text{action}_{i,0}, \text{reward}_{i,0}, ..., \text{state}_{i,T}, \text{action}_{i,T}, \text{reward}_{i,T}, \text{context}_i)$$

代表：
- [Reflexion](https://arxiv.org/abs/2303.11366): 存失败trial的verbal feedback
- [Generative Agents (Stanford smallville)](https://arxiv.org/abs/2304.03442): memory stream with timestamp + recency + importance + relevance scoring:

$$\text{score}(m_i) = \alpha \cdot \text{recency}(m_i) + \beta \cdot \text{importance}(m_i) + \gamma \cdot \text{relevance}(m_i, q)$$

其中recency用exponential decay, importance是LLM打分1-10, relevance是cosine similarity。

### 2.4 Semantic Memory (抽象knowledge)

抽象的facts、concepts、world knowledge。从episodic memory通过 **semanticization** 蒸馏出来：

$$\text{semantic\_memory} = \text{aggregate}(\text{episodic\_memory}_1, ..., \text{episodic\_memory}_n)$$

代表：
- [HippoRAG](https://arxiv.org/abs/2405.14831): KG-based
- [A-Mem](https://arxiv.org/abs/2502.12110): agentic memory with LLM-driven update
- [WISE](https://arxiv.org/abs2405.14720): lifelong model editing

### 2.5 Procedural Memory (skill library)

可复用action patterns、workflows、tool usage。从episodic memory蒸馏为executable skill：

$$\text{skill}_i = \text{compress}(\{\text{trajectory}_j | \text{success}_j = \text{True}\})$$

代表：
- **[Voyager (Minecraft)](https://arxiv.org/abs/2305.16291)**: skill library存executable JavaScript code，新skill写好后自动测试，通过就入库
- **[Agent Workflow Memory](https://arxiv.org/abs/2409.07429)**: 从successful trajectories提取workflow templates
- **[LEGOMem](https://arxiv.org/abs/2510.04851)**: modular procedural memory for multi-agent

进化路径：**explicit non-parametric templates → implicit parametric neural policies**。早期是文本skill description，近期用RL把skill internalize到weights。

### Cognitive Mechanism × Subject 交叉

Paper的Figure 5显示：
- **Working/Sensory/Procedural** 主要agent-centric (支持task execution)
- **Semantic/Episodic** 在两侧都有 (user prefs vs agent experience)

---

## 3. Memory Subjects — 为谁服务

### 3.1 User-Centric Memory

服务于具体用户的personalization。10种ability：
- **FE** (Fact Extraction): 从对话中抽取reusable facts
- **MR** (Multi-Session Reasoning): 跨session整合evidence
- **TR** (Temporal Reasoning): 时序推理
- **UR** (Update & Refresh): 当新信息contradict旧信息时更新
- **CS** (Compression): 压缩long history
- **FR** (Forgetting): 选择性遗忘obsolete信息
- **UP** (User Facts & Preferences): 用户persona + 偏好演化
- **AS** (Assistant Facts): 跟踪assistant自己之前的承诺
- **IC** (Implicit Inference): 隐式多跳推理
- **AB** (Abstain & Boundary): 在缺信息时说"I don't know"

代表benchmark: [LongMemEval](https://arxiv.org/abs/2410.10813), [LoCoMo](https://arxiv.org/abs/2402.10753), [PrefEval](https://arxiv.org/abs/2502.14665)

### 3.2 Agent-Centric Memory

Agent自己的experience accumulation，跨user泛化。4个motivation：
- **Long-Horizon Tasks**: 几百step的coding, web navigation
- **Domain-Specific Long-Tail**: 罕见bug、领域specific troubleshooting
- **Cross-Task Knowledge Transfer**: 跨task的策略迁移
- **Strategy & Skill Learning**: environment-grounded procedural memory

代表：[Voyager](https://arxiv.org/abs/2305.16291), [Synapse](https://arxiv.org/abs/2402.03610), [Agent KB](https://arxiv.org/abs/2507.06229)

---

## 4. Memory Operation Mechanism

### 4.1 Single-Agent: 5种核心operation

**Storage & Index**: write-time indexing with embeddings + metadata (timestamp, task_id, entities)

**Loading & Retrieval**: 
$$\text{top-}k = \text{argsort}_{i} \text{score}(q, m_i)[:k]$$

Trade-off：retrieve太多 → noise，太少 → missing critical info

**Update & Refresh**: 检测inconsistency → rewrite/merge。如 [MemGPT](https://arxiv.org/abs/2310.08560)的recursive summarization

**Compression & Summarization**: episodic → semantic distillation

**Forgetting & Retention**: 两种策略：
- **Heuristic**: recency decay + importance threshold
- **Learned**: RL policy learn what to forget

Ebbinghaus forgetting curve:
$$R(t) = e^{-t/S}$$
其中 $R(t)$ 是retention at time $t$, $S$ 是memory strength。教育应用[Agent4Edu](https://arxiv.org/abs/2501.15526)用这个curve模拟知识decay。

### 4.2 Multi-Agent: 4种architecture + 3种routing + 2种conflict resolution

#### 4种Memory Architecture

**Private-only**: 每个agent独立memory。代表[RecAgent](https://arxiv.org/abs/2402.11441), [TradingGPT](https://arxiv.org/abs/2309.03736)。优点：强isolation，隐私好；缺点：重复存储，浪费。

**Shared-workspace**: 共享pool。代表[MetaGPT](https://arxiv.org/abs/2308.00352)。优点：信息复用；缺点：需要filter避免noise。

**Hybrid**: private + shared + policy决定写哪。代表 [Collaborative Memory](https://arxiv.org/abs/2505.18279):
- Write policy: $p_{\text{write}}(\text{shared} | \text{info}) = \sigma(W \cdot \text{info})$
- Read policy: 用access graph构建permission-limited view

**Orchestrated**: 集中controller协调。代表[ChatDev](https://arxiv.org/abs/2307.07924), [MIRIX](https://arxiv.org/abs/2507.07957)。Meta Memory Manager作为router。

#### 3种Routing

**Orchestrator-based**: 集中决策，全局state。如[LEGOMem](https://arxiv.org/abs/2510.04851) orchestrator管理task + memory assignment。

**Agent-initiated**: 每个agent本地决定。如[SRMT](https://arxiv.org/abs/2501.13200)用cross-attention over shared memory。

**Memory-driven**: retrieval from store决定。如[G-Memory](https://arxiv.org/abs/2506.07398)构建hierarchical graph + neighborhood expansion。

#### Conflict Resolution

**Write Control**: [Memory-R1](https://arxiv.org/abs/2508.19828)定义4种atomic action: ADD, UPDATE, DELETE, NOOP。Memory manager是唯一能mutate的agent。

**Feedback Loop**: [EvoMem](https://arxiv.org/abs/2511.01912)用verifier + constraint memory + feedback memory迭代优化。

---

## 5. Learning Policies — 怎么学memory操作

### 5.1 Prompt-based (无训练)

**Static**: fixed human-designed rules。如 MemGPT的hierarchy, Reflexion的verbal reflection。

**Dynamic**: 测试时adapt。如EvoMem, ACON, [ReasoningBank](https://arxiv.org/abs/2509.25140)。

Pros: 无训练cost, high interpretability；Cons: 无credit assignment, 长期policy优化弱。

### 5.2 Fine-tuning (SFT)

把memory policy内化到weights。三种target：
- **Content internalization**: 蒸馏context到params
- **Access behavior**: 学习retrieval接口
- **Stabilization**: 边界控制避免drift

代表：[Memory3](https://arxiv.org/abs/2407.01178), [WISE](https://arxiv.org/abs/2405.14720)

### 5.3 Reinforcement Learning

最powerful的paradigm — memory operation本身成为policy action。

**Step-Level**: Memory-R1定义action space = {ADD, UPDATE, DELETE, NOOP}，用task reward优化。

**Trajectory-Level**: [MEM1](https://arxiv.org/abs/2506.15841) — RL学什么时候write/compress，目标是near-constant memory usage over long horizon。

**Cross-Episode**: [MCTR](https://arxiv.org/abs/2511.23262), [Retroformer](https://arxiv.org/abs/2304.04470) — 跨episode distill strategy。

**RL formulation (MDP)**:
- State $s_t$: current context + memory content
- Action $a_t$: memory operation (read/write/compress/forget)
- Reward $r_t$: task performance + memory efficiency
- Policy $\pi(a|s)$: memory controller

$$\pi^* = \arg\max_\pi \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^T \gamma^t r_t\right]$$

其中 $\gamma \in [0,1)$ 是discount factor，trade-off short-term vs long-term reward。

---

## 6. Scaling — Memory/Context/Environment

### 6.1 Context-Limited Simple Environment (上半场的benchmark)

经典benchmark如MMLU, HotpotQA, SQuAD都是static、bounded、episodic的。Agent既不需要cross-query state也不需要处理temporal drift。这些benchmark上overfit的agent在real world常常fail。

### 6.2 Context-Exploded Real-World (下半场的战场)

**三个scaling axis**:

1. **Interaction Horizon**: 长对话、长horizon tool use。ReAct-style agent每step都累积planning trace + tool output，context size **linear甚至exponential增长**。

2. **Environment Complexity**: heterogeneous modalities, schema-aware artifacts。Naive flatten成token sequence既inefficient又structurally lossy。

3. **Environment Quantity**: multi-environment coordination。如[OSWorld](https://arxiv.org/abs/2404.07972)的GUI agent要跨browser/file viewer/news app操作。Memory作为environment之间的interface。

---

## 7. Evaluation

### 7.1 Metrics (3类)

**Accuracy-based**: Accuracy, F1, Recall@K, MAP, NDCG@K, SR, Pass@K, Memory Integrity (MI), False Memory Rate (FMR)

NDCG@K:
$$\text{NDCG@K} = \frac{1}{\text{IDCG@K}} \sum_{i=1}^{K} \frac{rel_i}{\log_2(i+1)}$$

其中 $rel_i$ 是position $i$ 的relevance grade, IDCG@K是ideal ranking下的DCG，归一化消除不同query的难度差异。

**Similarity-based**: BLEU, ROUGE, Distinct-n, BERTScore, FactScore, Perplexity

**LLM-as-Judge**: Response Correctness, Faithfulness, Preference Following。如 [LongMemEval](https://arxiv.org/abs/2410.10813)用GPT-4 judge。

### 7.2 Benchmarks

**User-centric (15个)**: MSC, DuLeMon, MemoryBank, PerLTQA, LoCoMo, DialSim, LOCCO, MemoryAgentBench, LongMemEval, HaluMem, PersonaMem, PrefEval, MemBench, MemoryBench, ConvoMem

**Agent-centric (27个)**: HotpotQA, BrowseComp, Mind2Web, WebArena, WebShop, GAIA, OSWorld, AppWorld, τ-Bench, SWE-Bench, PaperBench, ALFRED, ALFWorld, MineDojo, EgoSchema, Video-MME, LongVideoBench, MT-Mind2Web, Evo-Memory, LifelongAgentBench, OdysseyBench等

每个benchmark的详细ability annotation见paper Table 3和Table 4。

---

## 8. Applications (12个领域)

| Domain | Memory核心功能 | 代表工作 |
|--------|----------------|----------|
| Education | 学习者cognitive digital twin | [LOOM](https://arxiv.org/abs/2511.21037), [Agent4Edu](https://arxiv.org/abs/2501.15526), [WebCoach](https://arxiv.org/abs/2511.12997) |
| Scientific Research | 跨stage reasoning provenance | [IterResearch](https://arxiv.org/abs/2511.07327), [GAM](https://arxiv.org/abs/2511.18423), [MirrorMind](https://arxiv.org/abs/2511.16997), [AISAC](https://arxiv.org/abs/2511.14043) |
| Gaming | bottom-up skill + social dynamics | [Voyager](https://arxiv.org/abs/2305.16291), [GITM](https://arxiv.org/abs/2305.17144), [Generative Agents](https://arxiv.org/abs/2304.03442) |
| Robotics | spatial graph + trajectory summary | [Memo](https://arxiv.org/abs/2510.19732), [MG-Nav](https://arxiv.org/abs/2511.22609), [JARVIS-1](https://arxiv.org/abs/2406.04384) |
| Healthcare | longitudinal emotion + adherence | [TheraMind](https://arxiv.org/abs/2510.25758), [DAM](https://arxiv.org/abs/2510.27418), [Mem-PAL](https://arxiv.org/abs/2511.13410) |
| Dialogue | persistent persona relationship | [MemGPT](https://arxiv.org/abs/2310.08560), [O-Mem](https://arxiv.org/abs/2511.13748), [MemoChat](https://arxiv.org/abs/2308.08239) |
| Workflow Automation | procedural template induction | [AWM](https://arxiv.org/abs/2409.07429), [ToolMem](https://arxiv.org/abs/2510.06664) |
| Software Engineering | global code context + failure recall | [MetaGPT](https://arxiv.org/abs/2308.00352), [ChatDev](https://arxiv.org/abs/2307.07924) |
| Streaming & Recommendation | 长range temporal pattern | [WorldMM](https://arxiv.org/abs/2512.02425), [GCAgent](https://arxiv.org/abs/2511.12027) |
| Information Search | knowledge synthesis workspace | [AgentFold](https://arxiv.org/abs/2510.24699), [MemSearcher](https://arxiv.org/abs/2511.02805) |
| Finance | strategic consistency + risk | [FinMem](https://arxiv.org/abs/2402.03755), [FinCon](https://arxiv.org/abs/2402.03755), [QuantAgent](https://arxiv.org/abs/2402.03755) |
| Legal | provenance + multi-doc reasoning | [MALR](https://arxiv.org/abs/2407.07913), [StaffPro](https://arxiv.org/abs/2507.21636) |

---

## 9. Future Directions (6大open challenge)

### 9.1 Continual Learning & Self-Evolving
当前continual learning只防forgetgetting，agent memory需要 **track evolving interaction state + procedural behavior**。需新的benchmarks评估sustained adaptation + behavioral stability under non-stationary objectives。

### 9.2 Multi-Human-Agent Memory Organization
当前multi-agent是episodic的，task结束interaction重置。需要 **collaborative/social memory** — 记住合作者preferences + feedback patterns，calibrate trust。

### 9.3 Memory Infrastructure & Efficiency
当前text-centric memory token开销大。三个level的solution：
- **Organized text-based**: schema-aware structured storage
- **Compressed latent memory**: episodic/semantic encoded成compact vectors
- **Internalized parametric**: experience吸收进params，constant-sized memory

代表：[MEM1](https://arxiv.org/abs/2506.15841), [Mem-α](https://arxiv.org/abs/2509.25911), [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym)

### 9.4 Life-Long Personalization & Trustworthy
Long-horizon personalization的核心challenge：
- **Memory staleness**: 旧preference是否还relevant
- **Concept drift**: 用户taste怎么演化
- **Credit assignment**: 哪个interaction是关键转折
- **Privacy leakage**: memory module易被extraction攻击
- **Memory poisoning**: 对抗性注入

### 9.5 Multimodal, Embodied, World-Model
Vision/audio/tactile/proprioception融合。核心方向：把memory升级为 **explicit predictive world model**：

$$s_{t+1} = f(s_t, a_t, m_t)$$

其中 $s_t$ 是latent state, $a_t$ 是action, $m_t$ 是memory update。Memory operations成为internal actions，与external actions联合优化。如 [DreamerV3](https://arxiv.org/abs/2301.04104)。

### 9.6 Real-World Benchmarking
Current benchmarks的3个gap：
- **Stationary user intent**: 假设preference不变
- **Reset-centric task design**: 跨episode不能积累
- **Short evaluation horizon**: 几周vs几个月

未来需要：closed-loop + longitudinal + execution-grounded + memory-sensitive invariants + provenance metadata + resource-utility trade-off量化。

---

## 10. 我的intuition takeaway

1. **Memory = 下半场的context engineering**: 上半场battle是model size + training data，下半场battle是 **how agent manages its own context across time**。这跟我之前说的"LLM OS"的intuition一致 — agent本质是个operating system，memory是它的RAM + disk。

2. **External memory是当下的pragmatic solution, internalized memory是long-term bet**: 因为internal memory的update cost太高，目前external memory + retrieval是主流。但[parametric memory](https://arxiv.org/abs/2407.01178)那条线很有潜力 — 把memory直接bake进weights，常数size。

3. **5种cognitive type不是独立的，是compositional**: 真实agent系统都hybrid。比如Voyager有episodic + procedural，MemGPT有working + episodic + semantic。这个taxonomy是 **conceptual scaffold**，不是implementation spec。

4. **Memory subject的区分是关键创新点**: 之前的survey都把memory当monolithic概念。这篇paper明确区分 **user-centric (为user个性化)** vs **agent-centric (为agent自我进化)**，optimization goal完全不同。

5. **RL for memory是the next frontier**: Prompt-based是"cheap heuristic"，SFT是"internalized heuristic"，RL是真正 **learned policy**。MEM1和Memory-R1是早期实验，效果已经显著。这个方向未来会爆发。

6. **Evaluation是真正的bottleneck**: LongMemEval虽然有500 questions，但仍是静态的。Real-world long-horizon memory benchmark — 跨weeks/months、preference drift、counterfactual feedback — 这个gap还没人填。

7. **隐私和安全是deployment blocker**: [HaluMem](https://arxiv.org/abs/2511.03506)和[Wang et al. 2025a](https://arxiv.org/abs/2503.03704)已经证明memory module能被black-box extraction attack。 healthcare/legal领域的deployment必须有verifiable、auditable、user-controllable memory。

8. **World model + Memory的synergy**: Section 9.5最ambitious的方向 — memory不是passive log，而是 **controllable internal state evolving over time**。memory operation成为planning的一部分。这个把[RWKV](https://arxiv.org/abs/2305.13048), [Mamba](https://arxiv.org/abs/2312.00752)的state space和world model的latent dynamics统一了。

---

## Key References

- [MemGPT: LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) | [repo](https://github.com/cpacker/MemGPT)
- [Generative Agents: Interactive Simulacra](https://arxiv.org/abs/2304.03442)
- [Voyager: Open-Ended Embodied Agent](https://arxiv.org/abs/2305.16291)
- [Reflexion: Verbal RL](https://arxiv.org/abs/2303.11366) | [repo](https://github.com/noahshinn/reflexion)
- [HippoRAG: Neurobiologically Inspired Memory](https://arxiv.org/abs/2405.14831)
- [RAPTOR: Recursive Abstractive Tree](https://arxiv.org/abs/2401.18059)
- [LongMemEval: Long-term Chat Memory Benchmark](https://arxiv.org/abs/2410.10813)
- [LoCoMo: Long Conversation Memory](https://arxiv.org/abs/2402.10753)
- [MEM1: RL for Memory](https://arxiv.org/abs/2506.15841)
- [Memory-R1: RL Memory Management](https://arxiv.org/abs/2508.19828)
- [Mem-α: Learning Memory Construction](https://arxiv.org/abs/2509.25911)
- [MIRIX: Multi-Agent Memory System](https://arxiv.org/abs/2507.07957)
- [A-Mem: Agentic Memory](https://arxiv.org/abs/2502.12110)
- [Titanans: Test-Time Memorization](https://arxiv.org/abs/2501.00663)
- [Awesome Agent Memory (paper repo)](https://github.com/AgentMemoryWorld/Awesome-Agent-Memory)

---

最后一句intuition：这篇survey在我看来是 **memory for LLM agents的"Attention is All You Need"时刻的综述版本** — 它没发明新东西，但把分散的hundreds of papers归到一个clean taxonomy下，让整个field第一次有了common vocabulary。接下来的几年，大家会用substrate/cognitive/subject这套语言讨论memory design，就像现在用encoder/decoder/attention讨论transformer architecture一样。
