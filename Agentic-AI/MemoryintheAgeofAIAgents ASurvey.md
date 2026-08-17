---
source_pdf: MemoryintheAgeofAIAgents ASurvey.pdf
paper_sha256: 10de3c050903bfa1113c9a954380e2786f42d25a9ea24478bf6cc69fef2e2b42
processed_at: '2026-08-05T17:33:02-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说这篇 Memory Survey

## 一句话说清楚这篇 paper 在干嘛

现在大家都在搞 agent，但 agent 有个要命的问题：**LLM 是条金鱼**。你跟它聊了三页，它 context 一满，前面的全忘。你让它做一个跨几天的任务，它第二天完全不记得第一天干了啥。

所以大家想了个办法——给 agent 加个"记忆本"。但问题是，每个人加的"记忆本"长得完全不一样：有人用 vector database，有人用 knowledge graph，有人直接 fine-tune 进 weights，有人搞 KV cache 复用。术语也乱，"episodic memory"、"semantic memory"、"parametric memory" 这些词被混着用。

这篇 paper 就是来收拾这个烂摊子的。它说：别再乱叫了，我们把 memory 这件事拆成三个正交的维度——**形态**、**功能**、**动态**——然后你往这个框架里一放，就知道自己在干啥。

GitHub paper list: https://github.com/Shichun-Liu/Agent-Memory-Paper-List

---

## 为什么要费劲搞 memory？LLM 不是有 context window 吗？

你看，LLM 的 context window 现在都能到 1M tokens 了，Claude 甚至号称能塞一整本书。那为啥还要 memory？

**因为 context window 是 RAM，不是硬盘**。

RAM 有几个问题：
1. **断电就没了**——你关掉这一轮对话，context 清空，下轮从零开始
2. **贵且慢**——你塞 1M tokens 进去，每生成一个 token 都要 attend 这 1M tokens，算力爆炸
3. **信号被稀释**——你把 100 轮对话全塞进去，模型实际 attend 到的关键信息被淹没在噪声里，反而表现变差
4. **不能跨 task**——任务 A 学到的经验，任务 B 开新 session 完全用不到

Memory 就是来解决这四件事的。它是 **persistent、structured、retrievable** 的外部状态，让 agent 能跨 turn、跨 task、跨 session 保持连续性。

这篇 paper 给了一个数学表述。Agent 的 policy 写成：

$$a_t = \pi_i(o_t^i, m_t^i, \mathcal{Q})$$

翻译成人话：agent 在时刻 $t$ 做决策 $a_t$，依据三样东西：
- $o_t^i$：它当下看到的环境观察
- $m_t^i$：从 memory 里捞出来的相关信息
- $\mathcal{Q}$：当前 task 的描述

如果你把 $m_t^i$ 设成 null，就是普通的 ReAct agent。如果你让 $m_t^i$ 随历史演化，就得到有记忆的 agent。**memory 就是 policy 输入里的一个维度，不是 bolt-on 的外挂**。

---

## 三个维度看 Memory

### 维度一：Form —— memory 长什么样

这是 Section 3。问的是：memory 物理上以什么形态存在？答案是三种：

#### 1. Token-level Memory：写在纸上的

这是最直觉的——把 memory 存成显式的 discrete units，人能读、能改、能审计。

你可以理解成 **agent 的笔记本**。它记对话、记 user profile、记成功经验、记失败教训。

按笔记本的组织方式又分三种：

- **Flat (1D)**：一坨平铺的笔记，没有结构。像一堆便利贴散在桌上。
  - Reflexion 就是把"反思"写在一张便利贴上，下次任务开始前看一眼
  - Mem0 就是让 LLM 自己总结对话，存成 entry
  - https://github.com/noahshinn/reflexion
  - https://github.com/mem0ai/mem0

- **Planar (2D)**：单层结构，有 graph 或 tree 关系。像一张思维导图。
  - A-MEM 把 memory 做成卡片，相关卡片放一个盒子里，盒子之间有 link
  - HippoRAG 用知识图谱做 memory，做 multi-hop retrieval
  - https://github.com/agiresearch/A-mem
  - https://github.com/OSU-NLP-Group/HippoRAG

- **Hierarchical (3D)**：多层 + 跨层 link。像一栋楼，每层都是个平面，但楼层之间有楼梯。
  - GraphRAG 用 community detection 把 entity-level subgraph 聚成 community-level summary
  - Zep 做三层 temporal KG：episodic / semantic / community
  - HiAgent 用 subgoal 切分，active subgoal 留详细 trace，completed subgoal 折成 summary
  - https://github.com/getzep/zep

**为啥要分层？** 你想啊，你做一道复杂菜，操作时是看具体步骤（底层），但偶尔要抬头看下整体菜谱框架（中层），偶尔要回想一下这类菜的通用套路（顶层）。不同粒度对应不同 reasoning 需求。

Token-level memory 的好处是透明——你能 debug 它、audit 它、transfer 它。坏处是密度低，retrieval 噪声大，redundancy 多。

#### 2. Parametric Memory：刻进脑子里的

这是把信息直接训练进 model weights。像你学会了骑自行车，骑的时候不用想，已经 instinct 了。

分两种：

- **Internal**：直接改 base model 参数。比如 CharacterGLM 把角色人设 fine-tune 进 weights。问题是更新成本高、容易忘旧的。

- **External**：加 adapter 或 LoRA 模块，不改 backbone。像 WISE 把"原始知识"和"编辑后知识"存两个 LoRA，推理时按需 route。这就像给 agent 换不同的"技能包"。
  - https://github.com/namespace-ua/WISE

Parametric memory 的好处是 zero-latency access——它已经是 model 的一部分，不用检索。坏处是难更新、难解释、难审计。

#### 3. Latent Memory：潜意识里的

这是介于上面两者之间。memory 不是显式 tokens，也不是固定 weights，而是 model 内部的 continuous representations——KV cache、hidden states、latent embeddings。

分三种来源：

- **Generate**：让一个模块生成新的 latent representation。比如 Gist 训一个 LLM 把长 prompt 压成几个 "gist tokens"。MemoRAG 把 global memory 压成 hidden states。这就像你读完一本书，脑子里形成一个"书的整体印象"，下次问起细节时从这印象出发去 reconstruct。
  - https://arxiv.org/abs/2310.08560

- **Reuse**：直接复用之前 forward pass 的 KV cache。Memorizing Transformers 把过去 tokens 的 KV 存下来，下次用 KNN 检索。这就像你算过一遍的中间结果存 cache，下次直接读。
  - https://openreview.net/forum?id=TrjbxzRcnf-

- **Transform**：把现有 KV cache 做 selection / compression。SnapKV 用 head-wise voting 找重要 prefix KV。PyramidKV 在 layer 之间重新分配 KV budget。这就像你给一摞书做精读笔记，扔掉不重要的，留下核心。
  - https://arxiv.org/abs/2406.02069

Latent memory 的好处是密度高、跨模态统一、推理快。坏处是黑盒——你不知道里面存了啥，没法直接 inspect 或 edit。

**三种形态的 trade-off 可以画个三角**：
- Token-level：透明、可编辑、密度低
- Parametric：内化、零延迟、不可解释
- Latent：中间态、黑盒、密度高

未来 hybrid 系统大概三种都用：底层 parametric 当 instinct，中层 latent 当 working state，上层 token-level 当 explicit knowledge。这跟人脑的 neocortex / hippocampus / PFC 分工挺像。

---

### 维度二：Function —— agent 为什么需要 memory

这是 Section 4。问的是：memory 干嘛用？答案分三种，对应认知科学的三类记忆。

#### 1. Factual Memory：知道什么

对应人脑的 declarative memory，你可以有意识 recall 它。再分两类：
- **Episodic**：具体经历。昨天用户说了啥、上周我做了啥任务。
- **Semantic**：抽象事实。用户的职业是医生、Python 的 list 是 mutable。

Agent 里这是 continuum：先 log episodic traces，再通过 summarization、reflection、entity extraction 转成 semantic fact base。

按 entity 分两类：
- **User factual**：关于 user 的事实。Identity、preference、routine。
- **Environment factual**：关于外部世界。文档状态、资源、其他 agent 的能力。

功能上保证三件事：**consistency**（不自相矛盾）、**coherence**（上下文连贯）、**adaptability**（个性化）。

代表工作：
- MemoryBank：按 timestamp 组织 dialogue history
- Mem0：LLM-driven summarization 把对话压成 memory entries
- HippoRAG：用 KG 做 multi-hop reasoning 的 factual memory

#### 2. Experiential Memory：怎么变强

对应人脑的 procedural memory，是你"会做"但说不清怎么做的事。Agent 版本是它把过去的 task execution 经验抽象成可复用的东西。

按抽象级别分四档：

- **Case-based**：保留 raw trajectory。最不抽象，但 fidelity 最高。下次遇到类似问题直接 replay。
  - Voyager 把 Minecraft 探索的 raw trajectory 存下来
  - ExpeL 通过 trial-and-error 收集成功轨迹作 exemplar
  - https://github.com/MineDojo/Voyager

- **Strategy-based**：提炼出 reasoning pattern / workflow / insight。
  - Reflexion 把失败反思提炼成 textual insight
  - Buffer of Thoughts 维护一个 thought template 的 meta-buffer
  - AWM 从成功 trajectory 抽 workflow，下次任务开始前先 retrieve workflow 作 scaffold
  - https://arxiv.org/abs/2303.11366

- **Skill-based**：把经验 compile 成 executable code / API / function。
  - Voyager 的 ever-growing skill library
  - Darwin Gödel Machine 可以安全地重写自己的代码
  - SkillWeaver 在 web 任务上自动 discover 和 hone skills
  - MCP（Model Context Protocol）做标准化的 tool 接口

- **Hybrid**：多种 representation 组合。比如 ExpeL 既存 trajectory 又存 textual insight。G-Memory 让 repeated 成功 cases 渐渐 compile 成 skills。

**为啥要抽象？** 你想，case-based 像你做饭时录视频存下来，下次再看一遍。Strategy 像你把视频总结成"先热油再下蒜"。Skill 像你直接写个菜谱程序。抽象程度越高，generalization 越强但 fidelity 越低。

#### 3. Working Memory：现在在想啥

对应 Baddeley 的 working memory 模型——**capacity-limited、actively controlled**。LLM 的 context window 默认是个 passive read-only buffer，没人主动管理它。Working memory 的目标就是把这个 buffer 变成 controllable、updatable 的 workspace。

分两类场景：

- **Single-turn**：单次处理超长输入。
  - LLMLingua 用 perplexity prune 不重要 tokens
  - Gist 把长 prompt 压成几个 latent tokens
  - VideoAgent 把视频流压成 temporal event descriptions
  - https://arxiv.org/abs/2310.06390

- **Multi-turn**：跨多 turn 维护 state。
  - MEM1 维护一个 shared internal state，把新 observation merge 进 prior memory
  - HiAgent 用 subgoal 作 memory unit，完成 sub-trajectory 折成 summary
  - Context-Folding 训一个 learnable folding policy，agent 自己决定何时 fold
  - https://arxiv.org/abs/2506.15841
  - https://arxiv.org/abs/2510.11967

**Hierarchical folding 是个特别 elegant 的 idea**。你做一个大任务时，正在做的子任务保持精细 trace，完成的子任务折成高层 summary。这就像你写代码时，正在调的函数记得每个变量值，调完的函数只记得"它干了啥"。

---

### 维度三：Dynamics —— memory 怎么运转

这是 Section 5。问的是：memory 怎么形成、怎么演化、怎么检索？这是三个 operator。

#### 1. Formation：怎么把信息变成 memory

把 raw interaction 压成 compact knowledge。五种方式：

- **Semantic Summarization**：宏观压缩。
  - Incremental：chunk by chunk merge（MemGPT、Mem0）
  - Partitioned：先 partition 再分别 summarize（MemoryBank 按 day/session）

- **Knowledge Distillation**：提炼具体 cognitive asset。
  - Factual：从对话提炼 thought、user intent
  - Experiential：从 trajectory 提炼 insight、workflow

- **Structured Construction**：组织成图/树结构。
  - Entity-level：从 text 抽 entity 和 relation 做 KG
  - Chunk-level：以 chunk 为节点建树或图

- **Latent Representation**：encode 成 vector 或 KV。
  - 多模态对齐尤其受益

- **Parametric Internalization**：直接训练进 weights。
  - Knowledge editing：ROME、MEMIT、MEND
  - Capability learning：SFT on reasoning traces

#### 2. Evolution：memory 库怎么维护

这是 memory system 的 "garbage collection + compaction"。

- **Consolidation**：把碎片整合成结构。
  - Local：合并相似 entries（RMM 的 top-K + LLM merge）
  - Cluster-level：跨 cluster fusion（PREMem 的 generalization / refinement）
  - Global：distill system-level insights（Matrix 的 task-agnostic principles）

- **Updating**：处理冲突。
  - External：MemGPT 的 replace/delete → Zep 的 temporal annotation 软删除 → MOOM 的 dual-phase（online soft + offline reflective）
  - Internal：model editing（ROME 的 gradient tracing + rank-one update）
  - Hybrid：ChemAgent 同步 external + internal

- **Forgetting**：删过时的。
  - Time-based：MemGPT evict oldest，MAICC 的 gradual weight decay
  - Frequency-based：XMem 的 LFU，MemOS 的 LRU
  - Importance-driven：composite score（time + frequency），LLM-based assessment

**Intuition**：Evolution 像数据库的 view materialization + conflict resolution + cache eviction。Consolidation 是 pre-compute frequent queries；Updating 是 optimistic concurrency control；Forgetting 是 LRU/TTL。这些机制让 memory 不会无限膨胀。

#### 3. Retrieval：怎么从 memory 里查

四步 pipeline：

- **Timing & Intent**：何时 retrieve，retrieve 哪个 source
  - MemGPT 让 LLM 自己 invoke retrieval function
  - ComoRAG 先 fast response 再决定是否 deep retrieve
  - MemGen 用 latent memory trigger 检测 critical moments

- **Query Construction**：怎么把 user query 翻译成 memory index language
  - Query Decomposition：拆成子查询（Visconde）
  - Query Rewriting：HyDE 生成 hypothetical document，用它 embedding 来检索

- **Retrieval Strategies**：怎么搜
  - Lexical：BM25
  - Semantic：Sentence-BERT、CLIP
  - Graph：AriGraph 的 K-hop expansion，HippoRAG 的 personalized PageRank
  - Generative：直接 generate document ID（不太用）
  - Hybrid：lexical + semantic + graph 组合

- **Post-Retrieval Processing**：怎么 refine 结果
  - Re-ranking：Semantic Anchoring 的 entity/discourse alignment
  - Aggregation：ComoRAG 的 Integration Agent，G-Memory 按 agent role 定制

**整个 retrieval pipeline 就是模拟人脑的 associative memory activation**。你想一个东西，相关记忆会被 parallel 激活。Agent 要模拟这个，需要多 stage pipeline。

---

## 几个我特别想强调的 intuition

### 1. Memory hierarchy 完全可以类比电脑存储

- Register / L1 cache = active context tokens
- L2/L3 = retrieved fragments in current context
- Main memory = latent memory (KV cache)
- SSD = token-level flat (vector DB)
- HDD = token-level hierarchical (KG)
- Tape = cold storage of episodic traces
- ROM = parametric memory (base weights)

每层 latency / capacity / cost / persistence 都不同。Agent 需要在层间 dynamic route，就像 CPU cache hierarchy 那样。

### 2. RL + Memory 是最有意思的方向

Paper Section 7.3 给出一个三阶段 evolution：

- **RL-free**：heuristic + prompt-driven。MemOS、Mem0、Dynamic Cheatsheet、ExpeL 都算。LLM 参与 memory management 但没专门训练。
- **RL-assisted**：RL 治理部分 pipeline。Mem-α 把 memory construction 训成 RL policy。Context-Folding 把 folding 训成 learnable policy。
- **Fully RL-driven**（未来）：让 agent 自己发明 memory organization。不照搬人脑 analogy，让 RL 在 task reward 下 explore 出 novel data structure。

**这个 trajectory 跟 LLM 自身发展对称**——从 hand-crafted features 到 RL-learned policies。如果 RL 能让 LLM 学会 reasoning 和 tool use，它也应该能让 LLM 学会 memory management。

实现方式大概是把 memory operations（add/update/delete/forget/retrieve/generate）作 action space，task performance 作 reward，这本质是个 meta-RL 问题。

### 3. Memory generation > Memory retrieval

Paper 7.1 提出 paradigm shift：从 "查 memory" 到 "生成 memory"。

Classical RAG 是查静态仓库。但理想 memory 应该是 context-adaptive 的——同一个 memory item，在不同 task里要 surface 出不同 abstraction level。Retrieve 是查 materialized view，generate 是 compute on demand。

代表工作：
- MemGen 直接从 reasoning state 生成 latent memory token，bypass explicit retrieval
- ComoRAG retrieve 然后 generate refined summary

未来大概是 hybrid：retrieve anchor facts 作 grounding，再 generate task-specific summary around them。

### 4. Offline consolidation 是被低估的方向

人脑有 sleep 阶段做 offline consolidation——hippocampus 的 episodic traces 在 sleep 时 replay，慢慢 consolidate 进 neocortex 成 semantic memory。

Agent 现在都是 online——边交互边更新。但 agent 可能也需要 "sleep" 阶段：decouple from environment，做 memory reorganization、generative replay、active forgetting、index optimization。这能解决 stability-plasticity dilemma——通过 periodic compaction 把 vast episodic streams 压成 efficient parametric intuition。

参考 Complementary Learning Systems theory (Kumaran 2016, McClelland 1995)，未来 agent 架构可能引入 dedicated consolidation intervals。

---

## 主要开源框架和 benchmark 一览

### Frameworks
- MemGPT: https://github.com/cpacker/MemGPT
- Mem0: https://github.com/mem0ai/mem0
- Letta (MemGPT production): https://github.com/letta-ai/letta
- Zep: https://github.com/getzep/zep
- A-MEM: https://github.com/agiresearch/A-mem
- MemOS: https://github.com/MemTensorAGI/MemOS
- LangMem: https://github.com/langchain-ai/langmem
- Cognee: https://github.com/topoteretes/cognee
- Memary: https://github.com/kingjulio8238/Memary

### Benchmarks
- LongMemEval: https://github.com/xiaowu0162/LongMemEval
- LoCoMo: https://github.com/snap-research/locomo
- MemBench: https://github.com/tan-zhu-99/MemBench
- StreamBench: https://github.com/zjyyzh/StreamBench
- HaluMem: https://github.com/Auster0wsl/HaluMem
- SWE-bench: https://github.com/SWE-agent/SWE-bench
- GAIA: https://github.com/S-Agarwal/gaia-benchmark
- WebArena: https://github.com/web-arena-x/webarena
- Mind2Web: https://github.com/OSU-NLP-Group/Mind2Web

---

## 最后的话

这篇 paper 真正的贡献不是某个具体 method 的 analysis，而是提供了一个 **conceptual framework**。Forms × Functions × Dynamics 这三个维度是正交的，你拿任何一个 memory system 套进去都能定位。

对研究者来说，最有意思的 frontier 大概是 **RL meets memory**。Memory management 正在从 hand-crafted 走向 learned，这跟 LLM 自身从 hand-crafted features 走向 RL-learned reasoning 的轨迹完全对称。Fully RL-driven memory 的 payoff 是真正 self-evolving agent——agent 能自己发明适合当前 task 的 memory organization，而不是硬套人脑 analogy。

这事儿 technical challenge 不小（long-horizon credit assignment, sparse reward, combinatorial action space），但 payoff 巨大。如果能 work，agent 就真的能"持续学习"了，不用每隔几个月就重训一个新 checkpoint。

希望这个"人话版"讲清楚了。如果你想深挖哪个具体方向，我可以再展开。

---

# Memory in the Age of AI Agents: A Survey — 深度讲解

## 0. Paper 的大图景

这篇 paper 要回答一个看似简单实则深刻的问题：当我们说一个 LLM agent 有 "memory" 时，我们到底在说什么？为什么需要 memory？memory 以什么形式存在？memory 如何随时间演化？

作者的核心 thesis 是：**传统的 long-term / short-term memory 二分法已经无法 capture 当代 agent memory 系统的复杂度**。他们提出一个三维 taxonomy —— **Forms × Functions × Dynamics** —— 来重新组织这个领域。

为什么这件事重要？因为 LLM 本身是 **stateless conditional generator**：parameters 训练完就冻住了，context window 一关就全忘。但 agent 必须能在时间中持续存在、adapt、和 environment 交互。Memory 就是把 static LLM 变成 adaptive agent 的那个 **bridge**。从这个角度，memory 不是 bolt-on 的辅助模块，而是 agent 的 first-class primitive。

Paper 的 GitHub paper list 在这里：https://github.com/Shichun-Liu/Agent-Memory-Paper-List

---

## 1. Formalization：把 agent 和 memory 数学化

### 1.1 Agent system

设 agent 集合 $\mathcal{I} = \{1, \ldots, N\}$，$N=1$ 是 single-agent（如 ReAct），$N>1$ 是 multi-agent（如 debate, planner-executor）。

Environment 有 state space $S$，transition 是 controlled stochastic process：

$$s_{t+1} \sim \Psi(s_{t+1} \mid s_t, a_t)$$

变量含义：
- $s_t \in S$：environment 在时间 $t$ 的状态
- $a_t$：在时间 $t$ 执行的 action
- $\Psi$：stochastic transition kernel，给定当前 state 和 action，给出下一个 state 的分布

每个 agent $i$ 接收 observation：

$$o_t^i = O_i(s_t, h_t^i, \mathcal{Q})$$

变量含义：
- $o_t^i$：agent $i$ 在时间 $t$ 的 observation
- $h_t^i$：agent $i$ 可见的交互历史（之前的 messages、tool outputs、reasoning traces 等）
- $\mathcal{Q}$：task specification，例如 user instruction 或 goal description（在同一 task 内视为 fixed）
- $O_i$：observation function，把 environment state + history + task 映射成 agent 看到的局部观察

Action space 是异构的：natural language generation、tool invocation、planning actions、environment control、communication。它们都通过 autoregressive LLM backbone 生成，所以 policy 写成：

$$a_t = \pi_i(o_t^i, m_t^i, \mathcal{Q})$$

变量含义：
- $\pi_i$：agent $i$ 的 policy（一个 LLM-based stochastic policy）
- $m_t^i$：memory-derived signal，这是 memory system 在这一步注入的信息（可能为 null）
- 其他同上

**Intuition**: 这个公式把 "memory" 显式写进了 policy 的输入。如果你把 $m_t^i$ 设为 null，就退化成 standard ReAct-style agent；如果你让 $m_t^i$ 随历史演化，就得到了有 memory 的 agent。Memory 不是 policy 之外的东西，它是 policy 输入的一个维度。

### 1.2 Memory system

Memory state 是一个 evolving object：

$$\mathcal{M}_t \in \mathbb{M}$$

其中 $\mathbb{M}$ 是 admissible memory configurations 的空间。关键点：**paper 不对 $\mathcal{M}_t$ 的内部结构做任何假设**。它可以是 text buffer、key-value store、vector database、graph、或者 hybrid。这是一个非常 general 的抽象。

Memory lifecycle 由三个 operator 刻画：

**Formation**（formation operator $F$）：

$$\mathcal{M}_{t+1}^{\text{form}} = F(\mathcal{M}_t, \phi_t)$$

变量含义：
- $\phi_t$：时间 $t$ 产生的 informational artifacts（tool outputs、reasoning traces、partial plans、self-evaluations、environment feedback）
- $F$：formation operator，selectively 把 artifacts 转换成 memory candidates
- $\mathcal{M}_{t+1}^{\text{form}}$：formation 后的中间 memory state

**Evolution**（evolution operator $E$）：

$$\mathcal{M}_{t+1} = E(\mathcal{M}_{t+1}^{\text{form}})$$

变量含义：
- $E$：evolution operator，把 formed candidates 整合进 existing memory base
- $\mathcal{M}_{t+1}$：最终的 new memory state

$E$ 可以做 consolidation（合并冗余）、conflict resolution、discarding low-utility info、restructuring。

**Retrieval**（retrieval operator $R$）：

$$m_t^i = R(\mathcal{M}_t, o_t^i, \mathcal{Q})$$

变量含义：
- $R$：retrieval operator，构造 task-aware query 并返回 relevant memory content
- $m_t^i$：retrieved memory signal，formatted 成 LLM policy 可以直接消费的格式（text snippets 或 structured summary）

**Intuition**: 这三个 operator 不是每一步都 invoke。不同的 system 在不同的 temporal frequency 上调用它们。比如有些 system 只在 $t=0$ 时 retrieve 一次：

$$m_t^i = \begin{cases} R(\mathcal{M}_0, o_0^i, \mathcal{Q}), & t=0 \\ \perp, & t>0 \end{cases}$$

其中 $\perp$ 是 null retrieval。这是 MemGPT 风格的 "load once then operate" 模式。

而 formation 可以从最 trivial 的 $\mathcal{M}_{t+1}^{\text{form}} = \mathcal{M}_t \cup \{o_t^i\}$（raw logging），到 sophisticated 的 reusable pattern extraction。

**关键 insight**: short-term 和 long-term memory 不是 architectural module 的区别，而是 **temporal invocation pattern** 的区别。同一套 $F, E, R$ operator，invoke 频率不同，就产生不同 temporal scale 的 memory 效应。这比硬性划分 STM/LTM 模块要 flexible 得多。

### 1.3 与 LLM memory / RAG / context engineering 的区别

这是 paper 的一个 important clarification。

**vs LLM memory**: "LLM memory" 这个术语在 2023-2024 很模糊。很多被称为 LLM memory 的工作（如 MemoryBank, MemGPT）实际上是 agent memory，因为它们解决的是 agent 跨 turn 跨 task 的 persistent state 问题。真正属于 LLM-internal memory 的是 KV cache 管理、long-context architecture（RWKV, Mamba）、attention sparsity 这些 model-internal 优化，它们 expand model 的 representational capacity，不 furnish agent with evolving external memory base。

**vs RAG**: Classical RAG 检索 external static knowledge source，task-specific，不 maintain evolving internal memory。Agent memory 是 agent 在 interaction 中持续 incorporate 自己产生的新信息。但 boundary 在 blur：HippoRAG 同时被 RAG 和 memory 社区认领。Practical 区分在 task domain —— RAG 多在 HotpotQA, 2WikiMQA, MuSiQue 这种 knowledge-intensive QA 上评，agent memory 多在 LoCoMo, LongMemEval, GAIA, SWE-bench, StreamBench 这种 sustained multi-turn interaction 上评。

**vs Context engineering**: Context engineering 把 context window 当成 constrained computational resource，optimize information payload。它在 working memory 这个层面和 agent memory 重合（rolling summary, token pruning 都是两边共享的技术）。但 context engineering 关注 inference-time 的 interface correctness 和 resource allocation，agent memory 关注 persistent cognitive state 和跨 task 的 identity 连续性。一个是 "怎么把信息塞进 window"，一个是 "agent 知道什么、经历过什么、这些怎么 evolve"。

参考：
- MemGPT: https://github.com/cpacker/MemGPT  
- Mem0: https://github.com/mem0ai/mem0
- HippoRAG: https://github.com/OSU-NLP-Group/HippoRAG
- Zep: https://github.com/getzep/zep

---

## 2. Forms：memory 以什么形式存在

这是 paper Section 3。作者识别出三种 dominant realization：

### 2.1 Token-level Memory

Memory 以 explicit、discrete、externally accessible 的 unit 存储。Token 这里是 broad 概念：text tokens, visual tokens, audio frames 都算。关键性质：**transparent, editable, interpretable**。

按 inter-unit topology 的复杂度分三类：

**Flat Memory (1D)**：no explicit topology。Memory 是 sequence 或 bag of units。

代表工作：
- Reflexion (Shinn et al. 2023): trajectory 作 short-term, self-reflection 作 long-term。https://arxiv.org/abs/2303.11366
- Mem0 (Chhikara et al. 2025): LLM-driven summarization 把对话压缩成 memory entries
- MemoryBank (Zhong et al. 2024): 按 timestamp 组织 dialogue history 和 event summaries
- MIRIX (Wang & Chen 2025): 六种 optimized flat memory types
- Voyager (Wang et al. 2024): executable skill code library，本质是 flat 但 content 是 code

**Planar Memory (2D)**：single-layer structured organization (graph, tree, table)，no cross-layer relations。

代表工作：
- A-MEM (Xu et al. 2025): card-based connected memory，按 relevance 把相关 memory 放同一 box。https://github.com/agiresearch/A-mem
- COMET (Kim et al. 2024): context-aware persona graph
- KGT (Sun et al. 2024): user preference 作 KG nodes/edges
- Ret-LLM (Modarressi et al. 2023): triplet-based relation-centric table
- HippoRAG (Gutierrez et al. 2024): KG + personalized PageRank

**Hierarchical Memory (3D)**：multi-layer with inter-layer links，形成 volumetric structured space。

代表工作：
- GraphRAG (Edge et al. 2025): multi-level community graph via community detection
- Zep (Rasmussen et al. 2025): 三层 temporal KG（episodic / semantic / community subgraph）
- HippoRAG 2 (Gutiérrez et al. 2025): 加深 passage integration 和 online LLM filtering
- G-Memory (Zhang et al. 2025): 三层 graph（interaction / query / insight）for multi-agent
- HiAgent (Hu et al. 2025): subgoal-centered hierarchical working memory
- AriGraph (Anokhin et al. 2024): semantic KG world model + episodic links

**Intuition**: Topology 越复杂，retrieval 表达力越强（multi-hop reasoning, vertical abstraction traversal），但 construction 和 maintenance cost 越高。Flat 适合 broad recall 和 fast update；planar 适合 relational reasoning；hierarchical 适合 multi-granularity abstraction。这有点像 database design：blob storage vs. relational DB vs. hierarchical document store 的 trade-off。

### 2.2 Parametric Memory

Memory 存在 model parameters 里。Information 通过 statistical pattern of parameter space 编码，forward pass 时 implicit access。

**Internal Parametric Memory**: 改 base model 的 weights。

按 training phase 分：
- Pre-train: TNL (Qin 2024), StreamingLLM (Xiao 2024), LMLM (Zhao 2025), Function Token (Zhang 2025)
- Mid-train: Agent-Founder (Su 2025), Early Experience (Zhang 2025) —— mid-training 阶段 integrate agent experience
- Post-train: Character-LM (Shao 2023), CharacterGLM (Zhou 2024), SELF-PARAM (Wang 2025), model editing 工作 ROME (Meng 2022), MEMIT (Meng 2023), MEND (Mitchell 2022), AlphaEdit (Fang 2025)

**External Parametric Memory**: 加 auxiliary parameters（adapters, LoRA），不改 backbone。

- MLP-Memory (Wei 2025): MLP 整合 RAG 知识
- K-Adapter (Wang 2021): task-specific adapter
- WISE (Wang 2024): dual-parameter memory + routing
- ELDER (Li 2025): multiple LoRA + routing function
- Memory Decoder (Cao 2025): plug-and-play，不修改 base model 但实现 parameter-internalized 推理速度

**Intuition**: Internal parametric memory 是 "把记忆变成 instinct"——zero-latency access，但 update cost 极高且 prone to catastrophic forgetting。External parametric memory 是 "modular memory module"——可以 add/remove/replace，但 influence 是 indirect 的，要经过 attention pathway 才能影响 output。这跟人脑的 complementary learning systems (CLS) 理论很像：hippocampus fast learning, neocortex slow consolidation。

### 2.3 Latent Memory

Memory 在 model 内部 representation 里（KV cache, activations, hidden states, latent embeddings），不是 explicit tokens 也不是 dedicated parameter sets。

按 latent state 的 origin 分三类：

**Generate**: 由独立 model 或 module 生成 new latent representations。
- Gist (Mu 2023): train LLM 把长 prompt 压成一组 gist tokens
- ICAE (Ge 2024): in-context autoencoder
- AutoCompressor (Chevalier 2023): 长 doc 压成 summary vectors
- MemoRAG (Qian 2025): LLM 产生 compact hidden-state memories
- MemoryLLM (Wang 2024): model 内部 persistent memory tokens
- M+ (Wang 2025): cross-layer long-term memory tokens
- LM2 (Kang 2025): 每层加 matrix-shaped latent memory slots
- Titans (Behrouz 2025): 把 long-range info 压进 online-updated MLP weight
- MemGen (Zhang 2025): 动态生成 latent memory fragment 并在 decoding 时 weave 进 reasoning stream

**Reuse**: 直接复用 prior computation 的 internal activations，主要是 KV cache reuse。
- Memorizing Transformers (Wu 2022): 存 past KV pairs，KNN retrieval
- FOT (Tworkowski 2023): memory-attention layers 做 KNN-based retrieval
- LONGMEM (Wang 2023): residual SideNet，historical KV embeddings 作 persistent memory

**Transform**: 把 existing latent state 做变换（selection, aggregation, compression）。
- Scissorhands (Liu 2023): 按 attention score prune tokens
- SnapKV (Li 2024): head-wise voting 聚合 high-importance prefix KV
- PyramidKV (Cai 2024): layer-wise KV budget reallocation
- H2O (Zhang 2023): 保留 recent tokens + H2 heavy hitter tokens
- Memory³ (Yang 2024): 只存 critical KV pairs
- RazorAttention (Tang 2025): compute effective attention span，保留 local window + compensation tokens

**Intuition**: Latent memory 是介于 explicit token 和 fixed parameter 之间的中间态。它比 token-level density 高（不 decode 成 text），比 parametric flexible（可以在 inference 时 dynamic update）。它 trade interpretability 换 expressive capacity。对 multimodal alignment 尤其有优势——视觉、音频、text 都可以 encode 成 latent vector，统一在同一空间里操作。

从认知科学角度，latent memory 有点像 "implicit memory"——你不能直接 introspect 它，但它影响行为。Token-level 像 "explicit/declarative memory"，可以 consciously access。Parametric 像 "procedural memory"，已经 deeply internalized。

---

## 3. Functions：agent 为什么需要 memory

Paper Section 4。这是 paper 最有 cognitive science grounding 的部分。

### 3.1 Factual Memory（"agent 知道什么"）

对应 neuroscience 的 **declarative memory**（Riedel & Blokland 2015），可 consciously access，分 episodic 和 semantic 两个 subsystem（Tulving 1972, Squire 2004）。

- **Episodic**: 存储 what/where/when of personally experienced events
- **Semantic**: 存储 general factual knowledge independent of acquisition context

在 agent 里这是 continuum：先 log 具体 interaction history 作 episodic traces，再通过 summarization、reflection、entity extraction、fact induction 转换成 semantic fact base。

Functional 上有三个性质：
- **Consistency**: 跨时间稳定行为，避免自相矛盾
- **Coherence**: 健壮的 context awareness，对话连贯
- **Adaptability**: 基于 stored user profile 个性化

按 entity 分两类：

**User Factual Memory**: 关于 user 的事实（identity, preferences, routines, commitments）。
- Dialogue Coherence: MemGPT, TiM (Liu 2023), MemoryBank, Mem0, RMM (Tan 2025), COMEDY (Chen 2025)
- Goal Consistency: RecurrentGPT (Zhou 2023), Memolet (Yen 2024), MemGuide (Du 2025), A-MEM, M3-Agent (Long 2025)

**Environment Factual Memory**: 关于 external world 的事实（documents, codebases, tools, other agents）。
- Knowledge Persistence: HippoRAG, MemTree, LMLM, M+, WISE, MemoryLLM, CAM
- Shared Access (multi-agent): MetaGPT (Hong 2024), GameGPT (Chen 2023), Generative Agents (Park 2023), S³ (Gao 2023), G-Memory, OASIS (Yang 2025)

**Intuition**: Factual memory 解决 "statelessness" 问题。一个 stateless LLM 每次都从零开始，会问重复问题、忘记之前承诺、自相矛盾。Factual memory 给 agent 一个 persistent identity 和 persistent world model。

### 3.2 Experiential Memory（"agent 如何变好"）

对应 neuroscience 的 **non-declarative / procedural memory**（Squire 2004, Seger & Spiering 2011），biological 系统用 distributed neural circuits 做 implicit skill acquisition。Agent 通常用 explicit data structure（vector DB, symbolic logs），这给 agent 一个 biological counterpart 没有的能力：**introspect、edit、reason over own procedural knowledge**。

这是 "era of experience"（Sutton 2025）的 foundation。通过 maintain experience repository，agent 走出 parametric update 的 prohibitive cost，实现 non-parametric continual learning。

按 abstraction level 分四类：

**Case-based Memory**: 保留 minimally processed 历史 records，高 fidelity，支持 replay 和 imitation。
- Trajectories: Memento (Zhou 2025, soft Q-learning 选 high-utility trajectories), JARVIS-1 (Wang 2025, Minecraft survival), Auto-scaling Continuous Memory (Wu 2025, GUI 压成 continuous embeddings), Early Experience (Zhang 2025, mid-training 注入 reward-free traces)
- Solutions: ExpeL (Zhao 2024, trial-and-error + textual insights), Synapse (Zheng 2024, state-action episodes 作 exemplars), MapCoder (Islam 2024, example code 作 playbook), FinCon (Yu 2024, financial PnL trajectories)

**Strategy-based Memory**: 提炼 transferable reasoning patterns, workflows, insights。分 atomic Insights / sequential Workflows / schematic Patterns 三种 granularity。
- Insights: Reflexion (Shinn 2023, self-reflection feedback), H²R (Ye 2025, decoupled plan/exec insights), R2D2 (Huang 2025, remember + reflect + dynamic decision), ReasoningBank (Ouyang 2025, success + failure 抽 reasoning units), Memory-R1 (Yan 2025, RL-trained LLMExtract module), Mem-α (Wang 2025, learnable insight extraction policy)
- Workflows: AWM (Wang 2024, induce reusable workflows on Mind2Web/WebArena), AgentKB (Tang 2025, workflows 作 transferable procedural knowledge)
- Patterns: Buffer of Thoughts (Yang 2024, meta-buffer of thought templates), RecMind (Wang 2024, self-inspiring planning), PRINCIPLES (Kim 2025, self-play 合成 strategy memory)

**Skill-based Memory**: 把 procedural capacity 编码成 executable actions。从 internal code 到 externalized standardized interfaces 的 continuum。Skill 必须 callable、outcome verifiable、composable。
- Code Snippets: Voyager (Wang 2024, ever-growing skill library), Darwin Gödel Machine (Zhang 2025, safely rewrite own code)
- Functions and Scripts: CREATOR (Qian 2023), RepairAgent (Bouzenia 2024), LEGOMem (Han 2025), Memp (Fang 2025), SkillWeaver (Zheng 2025)
- APIs: Gorilla (Patil 2024), ToolLLM (Qin 2024), COLT (Qu 2024), ToolRerank (Zheng 2024), ToolGen (Wang 2025), ToolRet (Shi 2025), DRAFT (Qu 2025)
- MCPs: Alita (Qiu 2025), LearnAct (Liu 2025), MemTool (Lumer 2025) —— Model Context Protocol 统一 agent 发现和使用 tools 的接口

**Hybrid Memory**: 多种 representation 组合。
- ExpeL: trajectories + textual insights
- AgentKB: hierarchical，workflow 指导 planning + 具体 solution 路径
- R2D2: replay buffer + reflective mechanism
- ChemAgent (Tang 2025): execution cases + decomposable skill modules
- LARP (Yan 2023): semantic + episodic + procedural，open-world game cognitive architecture
- G-Memory, Memp: 重复成功 cases 渐渐 compile 成 efficient skills

**Intuition**: 这部分 mirror 了人类 skill acquisition 的阶段：raw episodes → distilled heuristics → compiled procedures。区别在于 agent 可以 explicit represent 和 edit 这些 levels，而人类大多 implicit。这给了 agent 一个 "meta-cognitive" 能力：agent 可以 reason about 自己的 skills，决定哪个 level 的 abstraction 适合当前 task。Voyager 的 skill library 自动增长就是一个例子——agent 把成功探索 compile 成 reusable code，下次遇到类似情况直接 call，不用重新探索。

### 3.3 Working Memory（"agent 现在想什么"）

对应 Baddeley 的 multicomponent model 和 Cowan 的 embedded-processes account——**capacity-limited, dynamically controlled**，supporting active manipulation under attentional focus and interference control。

LLM 的 context window 默认是 passive read-only buffer，没有 explicit mechanisms 主动 select、sustain、transform。Huang et al. 2025 的 behavioral evidence 表明 current models 不 exhibit human-like working memory 特性。

Paper 定义 working memory 为 **single episode 内的 active context management mechanisms**。目标：把 context window 从 passive buffer 变成 controllable, updatable, interference-resistant workspace。

**Single-turn Working Memory**: 处理 massive immediate inputs（长 doc, high-dim multimodal streams）。
- Input Condensation:
  - Hard: LLMLingua (Jiang 2023, perplexity-based token discard), LongLLMLingua (Jiang 2024), CompAct (Yoon 2024, iterative max-info-gain)
  - Soft: Gist (Mu 2023), ICAE (Ge 2024), AutoCompressors (Chevalier 2023) —— encode 进 dense latent vectors
  - Hybrid: HyCo2 (Liao 2025)
- Observation Abstraction: Synapse (rewrite HTML DOM 成 task-relevant summaries), VideoAgent (Wang 2024, temporal event descriptions), MA-LMM (He 2024, dual-bank visual features), Context-as-Memory (Yu 2025, frame field-of-view overlap filtering)

**Multi-turn Working Memory**: 处理 temporal state maintenance。
- State Consolidation: MEM1 (Zhou 2025, shared internal state merges new obs + prior memory), MemGen (latent memory tokens in reasoning stream), MemAgent (Yu 2025, GRPO-optimized summarization), MemSearcher (Yuan 2025, end-to-end RL), ReSum (Wu 2025, reasoning states via RL), ACON (Kang 2025, jointly compress obs + history), IterResearch (Chen 2025, MDP-inspired workspace reconstruction)
- Hierarchical Folding: HiAgent (subgoal 作 memory unit, 完成 sub-trajectory 折叠成 summary), Context-Folding (Sun 2025, learnable folding policy), AgentFold (Ye 2025), DeepAgent (Li 2025, structured episodic + working memories)
- Cognitive Planning: SayPlan (Rana 2023, 3D scene graphs 作 queryable memory), Agent-S (Agashe 2025), KARMA (Wang 2025, hybrid long/short memory for household), PRIME (Tran 2025, retrieval in planning loop)

**Intuition**: Working memory 的 challenge 是 "如何在 fixed attention budget 下 maintain task-relevant info density"。Single-turn 处理 spatial bottleneck（input 太大），multi-turn 处理 temporal bottleneck（history 太长）。Hierarchical folding 是个特别 elegant 的 idea——按 subgoal 切分 trajectory，active subgoal 保留 fine-grained traces，completed subgoal 折叠成 summary。这相当于一个 dynamic 的 "zoom level"：你在做某事时 zoom in，做完 zoom out 概括。这和人类 task execution 的 chunking 很像。

---

## 4. Dynamics：memory 如何 operate 和 evolve

Paper Section 5。这是最有 "system" flavor 的部分。

### 4.1 Memory Formation

把 raw context encode 成 compact knowledge。必要性来自 scaling limitations：full-context prompting 遇到 computational overhead、memory footprint 爆炸、OOD length 性能下降。

五种 formation operation：

**Semantic Summarization**: 长 raw data → compact summary，保留 global high-level semantics。
- Incremental: MemGPT (chunk-by-chunk merge), Mem0 (LLM-driven), Mem1 (PPO-optimized), MemAgent (GRPO-optimized)
- Partitioned: MemoryBank (day/session segmentation), ReadAgent (semantic clustering), LightMem (topic-clustered), DeepSeek-OCR (optical 2D mapping), FDVS (multi-source signal integration for video)

**Knowledge Distillation**: 提取 specific cognitive assets。
- Factual: TiM (dialogue → thoughts), RMM (topic-based memory), MemGuide (user intent), M3-Agent (egocentric visual → text-addressable facts), Video-RAG (audio/subtitle/object → textual notes)
- Experiential: AgentRR (success → plans), AWM (workflows from success), Memp (gold trajectories → abstract procedures), Matrix/SAGE/R2D2 (failure-driven reflection), ExpeL (contrastive success vs failure), H²R (two-tier reflection: plan + subgoal level), Memory-R1 (RL-trained extraction), Mem-α (learnable extraction policy)

**Structured Construction**: 把 amorphous data组织成 explicit topological representation。
- Entity-Level: KGT, Mem0g, D-SMART (OWL-compliant KG fragment via neuro-symbolic pipeline), GraphRAG (community detection + iterative summarization), AriGraph (semantic + episodic dual-layer), Zep (3-layer temporal KG: episodic / semantic / community), HippoRAG
- Chunk-Level: HAT (hierarchical aggregate tree), RAPTOR (UMAP + GMM clustering), MemTree (bottom-up insertion + summary updates), H-MEM (top-down 4-level JSON hierarchy), A-MEM (discrete notes + semantic links), PREMem (cross-session reasoning pattern clustering), CAM (ego graph + iterative summarization + overlapping cluster disentanglement), G-Memory (3-tier: interaction / query / insight)

**Latent Representation**: encode 进 machine-native format (embeddings, KV states)。
- Textual: MemoryLLM (self-updatable latent embeddings), M+ (cross-layer long-term memory tokens), MemGen (latent memory trigger + weaver)
- Multimodal: CoMEM (Q-Former 压 vision-language), Encode-Store-Retrieve (Ego-LLaVA video → language → vector), Mem2Ego (landmark semantics 作 latent memory), KARMA (hybrid long/short multimodal embeddings)

**Parametric Internalization**: 把 external memory consolidate 进 model weights。
- Knowledge Internalization: MEND (auxiliary network for fast edits), ROME (causal tracing + rank-one update), MEMIT (batch editing via multi-layer residual), CoLoR (LoRA-based)
- Capability Internalization: ToolFormer (SFT on API calls), reasoning trace learning via SFT/DPO/GRPO

**Intuition**: Formation 是 "信息密度提升" 的过程。Raw interaction log 的 information density 很低（大量 redundant tokens）。每种 formation operation 在不同维度上 compress：summarization 压 global semantics，distillation 压 specific cognitive assets，structured construction 加 topology 信息，latent representation 跳过 human-readable format，parametric internalization 把 retrievable info 变成 intrinsic competence。这五种可以组合，比如先 structured construction 建图，再对图做 summarization。

### 4.2 Memory Evolution

把 newly formed memories integrate 进 existing repository。Naive append 忽略 semantic dependencies 和 contradictions，也忽略 temporal validity。

**Consolidation**: short-term traces → structured long-term knowledge。识别 semantic relationships，整合成 higher-level abstractions。
- Local Consolidation: RMM (top-K similar + LLM merge decision), VLN (capacity saturated → pooling)
- Cluster-level Fusion: PREMem (align new clusters + fusion modes like generalization/refinement), EverMemOS (similarity to MemScene centroids), TiM (hashing bucket + LLM merge), CAM (cluster merge into representative summary)
- Global Integration: MOOM (role profile integration), Matrix (iterative global optimization), AgentFold / Context Folding (internalize compression ability for multi-step)

**Updating**: 修冲突、补新信息，maintain factual consistency。
- External Memory Update: MemGPT/D-SMART/Mem0g (LLM detect conflict + replace/delete) → Zep (temporal annotation, soft deletion) → MOOM/LightMem (dual-phase: soft online + offline reflective consolidation) → Mem-α (policy-learning for update decisions)
- Model Editing: ROME (gradient tracing + targeted update), Memory Editor Networks (meta-editor predict parameter adjustments), MEMORYLLM (periodic memory token replacement), M+ (dual-layer with obsolete discard)
- Hybrid: ChemAgent (external + internal synchronization)

**Forgetting**: 删过时、redundant、low-value info，free capacity，maintain focus。
- Time-based: MemGPT (evict oldest on overflow), stochastic token replacement with K/N ratio (simulating exponential forgetting), MAICC (soft forgetting via gradual weight decay)
- Frequency-based: XMem (LFU), KARMA (counting Bloom filters), MemOS (LRU)
- Importance-driven: composite score (temporal decay + access frequency), VLN (similarity clustering pooling), Livia (emotional salience + contextual relevance), TiM/MemTool (LLM-based importance assessment)

**Intuition**: Evolution 是 memory system 的 "garbage collection + compaction" 阶段。Consolidation 像 DB 的 view materialization（把 frequent queries pre-compute 成 summary），Updating 像 conflict resolution（乐观锁 + eventual consistency），Forgetting 像 cache eviction policy（LRU/LFU/TTL）。这些机制让 memory 不会无限膨胀导致 retrieval 噪声增加。重要 trade-off：heuristic forgetting 可能 eliminate long-tail knowledge（很少 access 但 essential for correct decision-making），所以很多 system 在 storage cost 不是 critical bottleneck 时 avoid 直接删除。

### 4.3 Memory Retrieval

从 memory bank 中 retrieve relevant knowledge fragments 支持当前 reasoning。四个 stage：

**Retrieval Timing and Intent**: 何时 trigger、检索哪个 memory source。
- Automated Timing: MemGPT/MemTool (LLM 主动 invoke retrieval function), ComoRAG/PRIME (fast response + 评估 + deep retrieval on failure), MemGen (latent memory triggers from rollout states)
- Automated Intent: AgentRR (switch procedural template vs experiential abstraction based on feedback), MemOS (MemScheduler dynamic select parametric/activation/plaintext memory), H-MEM (index-based routing, coarse-to-fine domain → episode)

**Query Construction**: 把 raw query 转成 effective retrieval signal。
- Query Decomposition: Visconde, ChemAgent (decompose into sub-problems), PRIME/MA-RAG (Planner Agent for global plan), AgentKB (teacher observe student failures → targeted sub-queries)
- Query Rewriting: HyDE (generate hypothetical document, embed for retrieval), MemoRAG (compressed global memory + query → draft answer as rewritten query), MemGuide (LLM 生成 command-like intent phrase), Rewrite-Retrieve-Read (RL-trained rewriter), ToC (Tree of Clarifications)

**Retrieval Strategies**:
- Lexical: TF-IDF, BM25
- Semantic: Sentence-BERT, CLIP
- Graph: AriGraph, EMG-RAG, Mem0g, SGMem (identify relevant nodes + K-hop neighbor expansion), HippoRAG (personalized PageRank), CAM (LLM-steered subgraph exploration), D-SMART (LLM planner + beam search), Zep/MemoTime (temporal constraints)
- Generative: 直接 generate document identifiers (Tay 2022), tight integration 但 scalability 有限
- Hybrid: AgentKB/MIRIX (lexical + semantic), Semantic Anchoring (parallel semantic + symbolic inverted), Generative Agents (recency + importance + relevance scoring), MAICC (similarity + global/predicted returns), MemoriesDB (temporal + semantic + relational unified)

**Post-Retrieval Processing**: refine retrieved fragments。
- Re-ranking and Filtering: Semantic Anchoring (vector sim + entity/discourse alignment), RCR-Router (role relevance + task-stage priority + recency), Learn-to-Memorize (RL for score aggregation), Memory-R1/Westhäußer (LLM-based evaluator), Memento (Q-learning for contribution probability), MemGuide (fine-tuned LLaMA-8B re-ranker)
- Aggregation and Compression: ComoRAG (Integration Agent combines semantically aligned historical signals), MA-RAG (Extractor Agent fine-grained content selection), G-Memory (consolidate insights + customize per agent role)

**Intuition**: Retrieval 是 memory system 的 "query optimizer"。Classical RAG 只关注 retrieval accuracy，agent memory retrieval 还要关注 timing（何时 retrieve）、intent（retrieve 哪个 source）、query transformation（怎么把 user query 翻译成 memory index language）、post-processing（怎么把 retrieved fragments 变成 coherent context）。这四步组合起来就是一个 "associative memory activation" 过程——人类 brain 是 massively parallel associative retrieval，agent 要 simulate 这个就需要 multi-stage pipeline。

---

## 5. Resources

### 5.1 Benchmarks

Memory-specific benchmarks：
- MemBench (Tan 2025): 53,000 samples, interactive scenarios
- LoCoMo (Maharana 2024): 300 samples, conversational memory
- LongMemEval (Wu 2025): 5 categories / 500 samples, interactive memory
- PersonaMem (Jiang 2025): 15 tasks / 180 samples, dynamic user profiling
- PerLTQA (Du 2024): 8,593 samples, social personalized interactions
- MPR (Zhang 2025): 108,000 samples, user personalization
- PrefEval (Zhao 2025): 3,000 samples, personal preferences
- LOCCO (Jia 2025): 3,080 samples, chronological conversations
- MemoryBank (Zhong 2024): 194 samples, user memory updating
- StreamBench (Wu 2024): 9,702 samples, continuous online learning
- LifelongAgentBench (Zheng 2025): 1,396 samples, lifelong learning
- LongBench / LongBench v2 / RULER / BABILong / MM-Needle / HaluMem: long-context and hallucination evaluation

Agent benchmarks that implicitly stress memory:
- ALFWorld, ScienceWorld, BabyAI: embodied
- WebShop, WebArena, MMInA: web interaction
- SWE-Bench Verified: code repair
- GAIA, xBench-DS: deep research
- ToolBench: API tool use
- AgentGym, AgentBoard: multi-task

### 5.2 Open-source Frameworks

主流 framework 对比（paper Table 9）：
- MemGPT: hierarchical (S/LTM), LoCoMo eval
- Mem0: graph + vector, LoCoMo eval
- Memobase: structured profiles
- MIRIX: structured memory
- MemoryOS: hierarchical (S/M/LTM)
- MemOS: tree memory + memcube, LoCoMo/PreFEval/LongMemEval/PersonaMem eval
- Zep: temporal knowledge graph, LongMemEval eval
- LangMem, SuperMemory, Cognee, Memary, Pinecone, Chroma, Weaviate, Second Me, MemU, MemEngine, Memori, ReMe, AgentMemory, MineContext, Acontext, PowerMem, HindSight

---

## 6. Frontiers：未来方向

Paper Section 7。这是 paper 最 forward-looking 的部分。

### 6.1 Memory Retrieval → Memory Generation

Classical paradigm 把 memory 当 static repository，retrieval 是 query → relevant entries。Emerging paradigm 强调 agent 主动 **generate** memory representations on demand。

两条路线：
- Retrieve-then-Generate: ComoRAG, G-Memory, CoMEM —— retrieved items 作 raw material for reconstruction
- Direct Generation: MemGen, VisMem —— 直接从 context / interaction history / latent states 生成 memory，bypass explicit lookup

Future 三性质：context adaptive（granularity/abstraction 随 task 变）、heterogeneous signal integration（fusion 跨 modality）、learned and self-optimizing（通过 RL 优化 generation policy）。

**Intuition**: 这有点像 DB 的 "materialized view" vs "computed on the fly" 的对比。Retrieve 是查 materialized view，generate 是 compute on demand。Generate 的好处是 task-adaptive，坏处是 no grounding in stored facts。Hybrid approach 可能是未来——retrieve 一部分 anchor facts，再 generate task-specific summary around them。

### 6.2 Automated Memory Management

从 hand-crafted 到 automatically constructed memory systems。早期 system 用 fixed thresholds、predefined rules。Recent work（CAM, Memory-R1）让 agent 自主 manage memory evolution 和 retrieval，但仍受 task-specific 限制。

Future 方向：
- Tool-based strategy: 把 memory operations (add/update/delete/retrieval) 作 explicit tool calls，agent 在 reasoning loop 内 decide when to call
- Self-organizing memory structures: hierarchical + adaptive architectures，dynamic link/index/reconstruct

### 6.3 Reinforcement Learning Meets Agent Memory

这是最有 Karpathy flavor 的部分。Paper 给出一个三阶段 evolution（Figure 11）：

**RL-free Memory Systems**: heuristic + prompt-driven pipelines。MemOS, Mem0, MemoBase, Dynamic Cheatsheet, ExpeL, EvolveR, G-Memory。LLM 参与 memory management 但 no dedicated training for memory control。

**RL-assisted Memory Systems**: RL governs 部分 memory pipeline。
- RMM: policy gradient for memory chunk ranking
- Mem-α: RL-trained agent for memory construction
- Memory-R1: similar philosophy, LLMExtract + trained fusion
- Context Folding, Memory-as-Action, MemSearcher, IterResearch: RL-trained for ultra-long multi-turn working memory management

**Future: Fully RL-driven Memory Systems**:
- Minimal reliance on human-engineered priors（不照搬 cortical/hippocampal analogy，让 agent 通过 RL invent novel memory organizations）
- Complete control over all stages（formation + evolution + retrieval 都 learned end-to-end）

**Intuition**: 这跟 LLM 自身的发展轨迹很像——从 hand-crafted features 到 RL-learned policies。Memory management 正在经历同样的 paradigm shift。Fully RL-driven memory 的迷人之处在于：agent 可能发明人类想不到的 memory organization。比如人脑受 biological constraints（neuron spiking, synaptic plasticity rules）限制，但 digital agent 可以用 very different 的 data structure（dynamic hash table, learned index, hierarchical VAE）。RL 让 agent 在 task reward signal 下 explore 这些 design choices。

一个可能的实现：把 memory operation 作 action space，agent 在每个 reasoning step 选 add/update/delete/forget/retrieve/generate 等 memory action。Reward 来自 long-horizon task performance。这就是把 memory management 变成 meta-RL 问题。

### 6.4 Multimodal Memory

两个方向：
1. Multimodal agents 存/retrieve/utilize 跨 sensory inputs 的 memories (M3-Agent, MemoryVLA, Embodied VideoAgent, KARMA, Mem2Ego)
2. Memory 作 unified models 的 enabling component，增强 multimodal generation consistency (Context-as-Memory, WorldMM)

Future challenge: 真正 omnimodal memory，flexible accommodate diverse modalities 同时 preserve semantic alignment 和 temporal coherence。

### 6.5 Shared Memory in Multi-Agent Systems

从 isolated local memories + explicit message passing → centralized shared memory (global vector store, blackboard) → future: agent-aware shared memory (role/expertise/trust-conditioned access), learning-driven shared memory management, latent memory for heterogeneous signal abstraction。

代表工作：MetaGPT, GameGPT, Generative Agents, S³, Memory Sharing, G-Memory, OASIS, Collaborative Memory (Rezazadeh 2025, dynamic access control)。

### 6.6 Memory for World Model

World model 的核心是 high-fidelity physical world simulation。Memory 是 long-term consistency 的 cornerstone。

三个 architectural path：
- State-Space Models (SSMs): Mamba-style，compress infinite history into fixed-size recursive state
- Explicit Memory Banks: UniWM (hierarchical + feature similarity gating), WorldMem/Context-as-Memory (flat bank + geometric retrieval for 3D consistency)
- Sparse Memory and Retrieval: Genie Envisioner, Ctrl-World (sparse historical frame injection + pose-conditioned retrieval)

Future：Dual-System Architecture（System 1 fast instinctive + System 2 slow deliberative）, Active Memory Policies（cognitive workspace actively curate/summarize/discard）。

### 6.7 Trustworthy Memory

From trustworthy RAG → trustworthy memory。Memory module 可能 leak private data (Wang 2025)。Three pillars：

1. **Privacy preservation**: granular permissioned memory, user-governed retention, encrypted/on-device storage, federated access, differential privacy, memory redaction, adaptive forgetting
2. **Explainability**: traceable access paths, self-rationalizing retrievals, counterfactual reasoning（"如果没这个 memory 会怎样？"）, memory attention visualization, causal influence graphs
3. **Hallucination robustness**: conflict detection, multi-doc reasoning, uncertainty-aware generation, abstention under low-confidence, multi-agent cross-checking, mechanistic interpretability

参考 RAMDocs + MADAM-RAG (Wang 2025), RAGChecker (Ru 2024)。

### 6.8 Human-Cognitive Connections

Current agent memory structural alignment with human cognition：
- Context window + external DB ↔ Atkinson-Shiffrin multi-store model (working vs LTM)
- Interaction logs + world knowledge + code skills ↔ Tulving's episodic / semantic / procedural

但 fundamental divergence 在 retrieval dynamics：人脑是 constructive reconstruction (Schacter & Addis 2007)，agent 多用 verbatim retrieval。

Future: 
- Offline consolidation（analogous to sleep）, Complementary Learning Systems theory (Kumaran 2016, McClelland 1995)
- Generative replay, active forgetting, offline index optimization
- Generative memory（synthesizes latent tokens on demand, mirror brain's reconstructive nature）
- Sleep-like consolidation cycles 把 vast episodic streams compact 成 efficient parametric intuition

**Intuition**: 这个 direction 很有意思。当前 agent memory 都是 "online" 的——边交互边更新。但人脑有 sleep 阶段做 offline consolidation，hippocampus 的 episodic traces 在 sleep 期间 replay，慢慢 consolidate 进 neocortex 作 semantic memory。Agent 可能也需要 offline "睡眠" 阶段：decouple from environment，做 memory reorganization, generative replay, active forgetting, index optimization。这能解决 stability-plasticity dilemma——通过 periodic compaction 把 vast episodic streams 压成 efficient parametric intuition。

---

## 7. 我的一些 intuition 和联想

### 7.1 Memory 作为 first-class primitive

Paper 反复强调 memory 不是 bolt-on 模块，是 first-class primitive。这让我想到 programming language 的 first-class citizens——一个东西是 first-class 意味着它可以 be passed as argument, returned from function, stored in data structure。Memory 在 agent 里 first-class 意味着：可以被 agent 自己 reason about、edit、compose、route。这跟 OS kernel 的 memory management 抽象很像——process 不直接 access physical memory，而是通过 virtual memory abstraction（page, segment, mmap）。Agent 也需要类似 abstraction layer：不直接操作 raw text buffer，而是通过 memory operations（add/update/forget/retrieve/generate）。

### 7.2 Forms 之间的 trade-off

三种 forms 的 trade-off 可以用一个 2D plot 来理解：
- x 轴：interpretability（token-level > latent > parametric）
- y 轴：information density per unit cost（parametric > latent > token-level）

Token-level memory 适合需要 audit、debug、compliance 的场景（legal, medical, enterprise knowledge base）。Latent memory 适合 multimodal, on-device, privacy-sensitive 场景。Parametric memory 适合 deeply internalized skills, persona, domain expertise。

未来可能是 hybrid：底层 parametric memory 作 "instinct"，中层 latent memory 作 "working state"，上层 token-level memory 作 "explicit knowledge"。这对应人脑的 neocortex / hippocampus / prefrontal cortex 分工。

### 7.3 RL + Memory 的 meta-learning 视角

把 memory operations 作 action space，task performance 作 reward，这是 meta-RL 的一种形式。Agent 在 outer loop 学 memory management policy，在 inner loop 用 managed memory 做 task。这跟 MAML, RL² 等meta-RL 方法精神相似——学一个能快速 adapt 的 policy。

但 memory management 比 traditional meta-RL 复杂：state space 是 memory configuration（可能 high-dim），action space 是 memory operations（discrete + continuous），reward 是 long-horizon task success。这需要 hierarchical RL, option framework, 或者 latent action 之类的技术。

### 7.4 与 attention mechanism 的关系

Transformer 的 self-attention 本质是一个 content-addressable memory——每个 token 都可以 attend 到之前所有 tokens。但 attention 的 "memory" 是 within-forward-pass 的，不 persist across forward passes。

KV cache 是这个 memory 的 materialized form——存下来避免 recompute。KV cache reuse 类工作把这个 memory 跨 forward pass persist 化。这就是 latent memory 的 reuse 范式。

更广义看，memory system 是 "把 attention 的 content-addressability 推广到跨 task" 的 mechanism。Token-level memory 用 explicit retrieval simulate content-addressability，latent memory 用 KNN over KV cache simulate，parametric memory 用 gradient-based learning internalize。三种 forms 是同一个 fundamental mechanism 的不同 implementation。

### 7.5 与 neuroscience 的对应

- Token-level memory ↔ explicit/declarative memory (hippocampus-dependent, can consciously access)
- Parametric memory ↔ procedural memory (basal ganglia-dependent, implicit)
- Latent memory ↔ 工作记忆的 neural population activity（sustained firing, persistent activity in PFC）
- Working memory ↔ PFC 的 capacity-limited active maintenance
- Memory consolidation ↔ sleep replay, hippocampus → neocortex transfer
- Forgetting ↔ synaptic decay + active inhibition (Anderson & Hulbert 2021)
- Memory generation ↔ constructive memory (Schacter & Addis 2007) —— 这个对应还很不成熟，是未来方向

### 7.6 Memory hierarchy 与 computer architecture 的类比

完全可以借用 computer memory hierarchy 来理解 agent memory：
- Register / L1 cache ↔ active context tokens (working memory)
- L2/L3 cache ↔ retrieved memory fragments in current context
- Main memory ↔ latent memory (KV cache, hidden states)
- SSD ↔ token-level flat memory (vector DB)
- HDD ↔ token-level hierarchical memory (KG, hierarchical tree)
- Tape / archival storage ↔ cold storage of episodic traces
- ROM / firmware ↔ parametric memory (base model weights)

每层 trade off latency, capacity, cost, persistence。Agent 需要在层之间 dynamic route 信息，就像 CPU 的 cache hierarchy 一样。

### 7.7 一些可能的 research direction

读完这篇 paper，我觉得有几个方向可能被低估：

1. **Differentiable memory indexing**: 现在大多数 retrieval 用 discrete index (vector similarity top-K, graph K-hop)。如果 indexing 本身是 differentiable 的，整个 retrieval pipeline 就可以 end-to-end train。这跟 learnable data structures (learned index, differentiable hash table) 有关。

2. **Memory compression with theoretical guarantees**: 现在的 summarization 多是 heuristic。信息论视角下，理想 memory compression 应该 preserve task-relevant information, drop task-irrelevant information。可以用 information bottleneck framework formalize 这个。

3. **Causal memory**: 现在的 memory 多是 correlational（什么 happened, in what context）。Causal memory（什么 caused what, what would have happened if...）可能对 long-horizon planning 更重要。Counterfactual memory 是 explainability 的一个 emerging direction。

4. **Memory as world model**: 最近 world model 工作（DreamerV3, JEPA, Genie）显示 latent dynamics model 对 planning 很重要。Memory system 和 world model 可能 converge——memory store past states, world model predict future states, 两者都用 latent representation。Memory-augmented world model 是 natural synthesis。

5. **Memory programming language**: 现在每个 framework 都自己 define memory operations。如果有一个统一的 memory programming language（类似 SQL 之于 relational DB），可以 decouple memory implementation from memory use。MCP 可能是这个方向的 early signal。

---

## 8. 总结

这篇 paper 是一个很 comprehensive 的 survey，把碎片化的 agent memory 研究整合到一个 Forms × Functions × Dynamics 三维 taxonomy 下。它最强的贡献不是单个 method 的 analysis，而是提供了一个 conceptual framework 让后续工作可以定位自己。

对 Karpathy 这样的研究者，最有意思的部分可能是 Section 7 的 frontiers，特别是 RL meets memory 那节。Memory management 从 hand-crafted → RL-assisted → fully RL-driven 的 trajectory 跟 LLM 自身的发展轨迹对称。如果 RL 能让 LLM 学会 reasoning 和 tool use，它也应该能让 LLM 学会 memory management。这件事的 technical challenge 很大（long-horizon credit assignment, sparse reward, combinatorial action space），但 payoff 也很大——真正 self-evolving agent 的 holy grail。

Paper GitHub: https://github.com/Shichun-Liu/Agent-Memory-Paper-List

主要 framework 参考：
- MemGPT: https://github.com/cpacker/MemGPT
- Mem0: https://github.com/mem0ai/mem0
- Letta (MemGPT 的 production version): https://github.com/letta-ai/letta
- Zep: https://github.com/getzep/zep
- HippoRAG: https://github.com/OSU-NLP-Group/HippoRAG
- Reflexion: https://github.com/noahshinn/reflexion
- Voyager: https://github.com/MineDojo/Voyager
- Generative Agents: https://github.com/joonspk-research/generative_agents
- A-MEM: https://github.com/agiresearch/A-mem
- MemOS: https://github.com/MemTensorAGI/MemOS
- LangMem: https://github.com/langchain-ai/langmem
- Cognee: https://github.com/topoteretes/cognee
- Memary: https://github.com/kingjulio8238/Memary

主要 benchmark 参考：
- LongMemEval: https://github.com/xiaowu0162/LongMemEval
- LoCoMo: https://github.com/snap-research/locomo
- MemBench: https://github.com/tan-zhu-99/MemBench
- HaluMem: https://github.com/Auster0wsl/HaluMem
- StreamBench: https://github.com/zjyyzh/StreamBench
- SWE-bench: https://github.com/SWE-agent/SWE-bench
- GAIA: https://github.com/S-Agarwal/gaia-benchmark
- WebArena: https://github.com/web-arena-x/webarena
- Mind2Web: https://github.com/OSU-NLP-Group/Mind2Web
