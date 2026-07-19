---
source_pdf: Are We Ready For An Agent-Native Memory System.pdf
paper_sha256: 6cad9bfcded2f1801c09049daa716a33e570cc407ddfbe40f32d3aa850361d85
processed_at: '2026-07-18T09:12:03-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Paper 深度解读: Are We Ready For An Agent-Native Memory System?

Andrej, 这篇 paper 来自 Shanghai Jiao Tong University (Xuanhe Zhou, Fan Wu) 和 Tsinghua University (Guoliang Li) 团队，发表于 2026 年左右，是一篇非常少见的、把 agent memory 当成 **database/data management system** 来系统拆解和 benchmarking 的工作。它的核心论点是：现有 evaluation 把 memory 当作一个 monolithic black box，只看 end-to-end F1/BLEU，而忽略了 system-level 的 cost, robustness, lifecycle governance，导致我们对 "agent-native memory" 这个概念其实准备不足。

下面我从 intuition 出发逐层拆解。

---

## 1. 核心动机: 为什么需要重新审视 agent memory?

### 1.1 Memory 已从 RAG 演化成一个数据管理子系统

传统 RAG 是一个 **stateless, read-only retrieval primitive**:
$$
\text{RAG}(q) = \text{LLM}\left( q \,\Vert\, \text{top-}k\text{-retrieve}(q, \mathcal{D}_{\text{static}}) \right)
$$
其中 $\mathcal{D}_{\text{static}}$ 是静态 corpus。RAG 不持久化 agent 状态，也不处理 conflict / versioning。

而 agent memory 是一个 **persistent, updatable state manager**，要处理四件事:
1. **Storage**: 怎么存 (text, vector, graph, KV-cache, parametric weights)
2. **Extraction**: 怎么从 raw interaction stream 提取 memory primitive
3. **Retrieval/Routing**: 怎么按 query 动态取出相关子集
4. **Maintenance**: 怎么处理 conflict、过期、容量、consolidation

作者形式化定义:
$$
M_{sys} = \langle \mathcal{R}, S, Q, \mathcal{U} \rangle
$$
- $\mathcal{R}$: memory representation + physical storage
- $S$: extraction mechanism (input stream → memory primitive)
- $Q$: retrieval/routing function (query context → relevant subset)
- $\mathcal{U}$: maintenance policies (conflict resolution, capacity management, semantic consolidation)

### 1.2 Agent memory 与 traditional DB workload 的根本差异

这是这篇 paper 最有 insight 的一段。作者指出 agent memory ≠ OLTP/OLAP:

| 维度 | Traditional DB | Agent Memory |
|---|---|---|
| Query 形态 | predicate-based, 精确逻辑表达式 | natural language, partial context, latent intent → 需要 approximate matching, query rewriting, LLM-guided retrieval |
| Update 语义 | overwrite tuple under schema | 持续、可能 conflict 的 observations，需要 multi-versioning, invalidation, precedence |
| Workload 异构性 | 单一 workload type | 一个 workload 同时混合 long-context synthesis, episodic recall, structured fact lookup, temporal reasoning, streaming update |
| Consistency model | ACID / BASE | 没有 transaction boundary，需要 hybrid execution strategies |

这个 framing 很关键，因为它直接解释了为什么单纯跑 F1 是不够的——同样的 F1，可能在 retrieval 阶段就失败了 (evidence 没捞回来)，也可能在 maintenance 阶段失败了 (evidence 捞回来了但版本错了)。

参考链接:
- Anthropic context engineering blog: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- CIDR 2026 "Agent-First Data Systems": https://www.cidrdb.org/

---

## 2. 四模块 Taxonomy 深度解析

作者把 12 个系统按四模块做了完整分类 (Table 1)。这是 paper 的核心架构图 (Figure 1)。

### 2.1 Memory Representation and Storage ($\mathcal{R}$)

分为两个子轴: **logical representation** 和 **physical storage**。

#### Logical Representation (Figure 2):

**❶ Token-Level Sequence** — 一维 token 序列，无显式结构
- Explicit Discrete Text Token: Mem0 (discrete facts), MemoChat (JSON memos), MemAgent (1024-token bounded text), MEM1 (`<IS>` tags)
- Implicit Continuous Vector Token: dense embeddings attached to facts, KV-cache tensors (MemoRAG)

**❷ Graph & Tree-Based Topology** — 显式拓扑
- Temporal Knowledge Graphs: Zep (episode/entity/community subgraphs), Mem0$^\*$ (directed labeled graph with entities as vertices, triplets like `LIVES_IN` as edges)
- Hierarchical Tree: MemTree (dynamic directed tree, node = (content, embedding, topological pointers, depth $d$))

**❸ Heterogeneous Composite** — 多部分 container
- MemOS MemCube: 三种 payload (plain-text, activation, parametric memory) + structured metadata
- A-MEM: atomic notes with JSON attributes
- Letta: context tiers

#### Physical Storage (Figure 3):

**❶ Transient In-Context Register** — 完全在 KV cache / context window 里，零 disk I/O。MemoChat 把 JSON memos 直接塞 context, MemAgent 把 summary tokens 作为 KV cache tensors。

**❷ Specialized Single-Engine**:
- Dense Vector DB: Mem0, MemTree
- Graph DB: Zep, Mem0$^\*$ (用 Neo4j + Cypher)
- Relational SQL: Letta (PostgreSQL + pgvector), LightMem (append-only factual streams)
- File/Object Store

**❸ Heterogeneous Multi-Engine**:
- SimpleMem: LanceDB with IVF-PQ, 同时维护 dense embeddings + BM25 + SQL predicates
- MemoryOS: dense cosine + discrete Jaccard 混合
- MemOS: Vector + Graph DB via adapter

### 2.2 Memory Extraction ($S$) — Figure 4

**❶ Raw Sequence Concatenation** — 零开销，直接拼接。MEM1, MemAgent。
**❷ Schema-Free Semantic Extraction** — LLM 蒸馏 standalone facts。Mem0 ("User is vegetarian and dairy-free")。
**❸ Schema-Constrained Structured Extraction** — LLM 强制填 schema。Zep/Mem0$^\*$ 提取 typed directed edges (`LIVES_IN`, `WORKS_AT`), Zep 还有 reflection-based verification 抑制 hallucinated triplets。MemoChat 把对话塞进严格 JSON schema。

### 2.3 Memory Retrieval and Routing ($Q$) — Figure 5

**❶ Native Attention-Based Retrieval** — 直接用 transformer self-attention 做 implicit retrieval。MEM1 用 2D attention mask 保 causal consistency, MemAgent 把 blocks 直接拼进 prompt template。
**❷ Semantic-Based Dense Retrieval** — 标准 KNN:
$$
\text{retrieve}(q) = \text{top-}k\left( \text{sim}(E(q), E(m_i)) \right)_{i=1}^{N}
$$
Mem0, LightMem (cosine), MemTree (collapsed-tree broadcast 全局 cosine 分布)。

**❸ Topological Subgraph Traversal** — 走 graph edges。Mem0$^\*$ entity-centric 递归遍历 local subgraph。A-MEM 先 dense KNN 找 anchor, 再 localized graph traversal。

**❹ Autonomous Agentic Routing** — LLM 自己当 query planner
- Function Call Invocation: Letta 让 LLM emit `archival_storage.search()` 
- Generative Query Expansion: SimpleMem Intent-Aware Retrieval Planning, LLM dissects query, 计算 adaptive search depth, synthesize query variants

**❺ Multi-Stage Hybrid Execution**:
- Sequential Hybrid: MemoryOS 先 Boolean filter 再 semantic ranking
- Parallel Ensemble: Zep 同时跑 cosine + BM25 + BFS, 再用 RRF (Reciprocal Rank Fusion), MMR (Maximal Marginal Relevance), cross-encoder reranking

### 2.4 Memory Maintenance ($\mathcal{U}$) — Figure 6

**❶ Timestamp-Based Multi-Versioning** — 不物理删除，用 timestamp + validity flag 逻辑 invalidation。Zep, Mem0$^\*$, LightMem (append-only timestamped streams), SimpleMem (ISO-8601 严格时序优先), MemOS (Update API + differential writes 生成 multi-version chain)。

**❷ Capacity-Driven Physical Eviction**:
- Constraint-Based Hard Eviction: FIFO, 固定 token limit。MemAgent (fixed segment boundary overwrite), MEM1 (system-enforced FIFO truncation), Letta (OS-inspired queue manager, token 超 threshold 强制 flush)
- Score-Based Priority Eviction: MemoryOS Heat score = $f(\text{retrieval frequency}) \cdot e^{-\lambda \Delta t}$, evict 最低 heat 的 segments

**❸ LLM-Driven Semantic Consolidation**:
- Inline Semantic Compaction: 写入时 merge redundant assertions。SimpleMem (on-the-fly synthesis), MemTree (递归触发 parent node summarization)
- Tool-Driven CRUD: Mem0 通过 LLM tool-calling 发 `UPDATE`/`DELETE` 命令

**❹ Continuous Parametric Optimization** — 异步 background fine-tuning。MemoRAG 用 RLGF (Reinforcement Learning with Generation Feedback) 离线优化。

---

## 3. End-to-End Assessment: 5 个 RQ

### 3.1 RQ1: Overall Effectiveness

**Setup**: 12 systems + 2 baselines (Long Context, Embedding RAG), 3 benchmarks:
- LoCoMo (long-conversation QA, EM + Answer F1)
- LongMemEval (multi-session, Substring EM, ROUGE-L F1/Recall, GPT-5.4 LLM Judge)
- DB-Bench (procedural execution, EM + Task Success Rate)

**关键发现 O1**: 没有单一架构 dominate。Leading systems 随 workload 漂移:
- LongMemEval: 结构化系统领先。Zep 48.0 LLM Judge Acc, Cognee 35.3 ROUGE-L F1
- LoCoMo exactness: MemOS 11.5 EM
- DB-Bench: Long Context 48.20 EM, MemoChat 55.40 Task Success Rate

**Finding 1 (Workload-Aligned Memory)**: 三种 bottleneck 对应三种最优形态:
1. Dispersed cross-session reasoning → relation/time-aware (Zep, Cognee)
2. Long coherent dialogue → coarse-to-fine filtering (MemOS, MemoryOS)
3. Stateful execution → trace-preserving (Long Context, MemoChat)

**O2**: EM 适合 short canonical grounded outputs，但对 paraphrastic synthesis 或 executable success 不够。DB-Bench 上 Long Context EM 最高，但 MemoChat Task Success Rate 更高。

### 3.2 RQ2: Retrieval Fidelity

**Setup**: 8 systems, LoCoMo gold evidence, Recall@K + Recall@10 over 6 evidence-distance bins (1-5 到 26-31 sessions)。

**O1 (Structured Evidence Expansion)**: SimpleMem Recall@1 最高 (39.0)，但 A-MEM (69.5/85.9) 和 MemTree (59.7/80.5) 在 Recall@5/@10 上反超，且随 evidence distance 稳定。Embedding RAG 在 short-gap 后急剧 drop。

这给了一个非常重要的直觉: **memory retrieval 不是 top-1 ranking 问题，是 evidence-completion 问题**。需要把分散在多 turn 的 evidence 都捞回来。

**Finding 2 (Evidence-Centric Memory Organization)**: 
1. Early localization 和 evidence assembly 是分离的 design targets
2. Explicit structure (links/hierarchy) 在 evidence scattered 时最有价值
3. Flat similarity 只在 short-range 有效

### 3.3 RQ3: Memory Evolution Robustness

**Setup**: Update Robustness Comparison (LongMemEval Knowledge Update + Temporal Reasoning, LoCoMo Temporal) + Backbone Robustness (4 LLM backbones on LoCoMo)。

**O4 (Temporal State Externalization)**: 
- Zep 在 Knowledge Update 最强 (44.4 Substring EM, 36.8 ROUGE-L F1) — graph 结构容易 handle fact revision
- Cognee 在 Temporal Reasoning 最强 (18.7 / 35.8) — relation-organized retrieval 适合 dispersed evidence
- MemOS 在 LoCoMo EM 最高 (8.9), Cognee Answer F1 最高 (28.1) — hybrid filter 适合 latest-state grounding

**O5 (Backbone Robustness)**: Figure 9 显示换 backbone (4 个 LLM) 主要改变 answer quality 的绝对值，不改变 memory pipeline 的 ordering。MemOS 在 4 个 backbone 上稳定 (32.2, 41.2, 38.6, 41.2)，唯一 reversal 是局部: A-MEM 在 GPT-5.4-mini 和 GPT-5.4 上超过 MemTree。

**Finding 3**: Reliable post-update behavior 是 pipeline-level design problem, 不是 model-capacity problem。Stronger backbone 只在 grounding 成功后 refine answer expression，不能修复 stale/conflicting memory。

### 3.4 RQ4: Long Horizon Stability

**Setup**: LongBench (Short/Medium/Long context), LongMemEval (session count bins), LoCoMo (evidence distance bins)。

**O6**: LongBench 上 SimpleMem 几乎不退化 (35.2 → 34.9)，Long Context 急剧 drop (42.6 → 19.0)。LoCoMo 上 Embedding RAG 从 37.1 drop 到 7.4，graph/consolidated systems (Cognee, MemOS, MemoryOS) 保持高位。

**Finding 4 (Horizon-Structured Memory)**: 长 horizon 的核心挑战从 "存更多 history" 转向 "选对 abstraction":
1. Multi-view filtering (SimpleMem) — 长输入 distractor 多
2. Relation-aware indexing (Cognee, Zep) — evidence 跨多 turn/session
3. Coarse-to-fine summarization (MemOS, MemoryOS) — 先定位 session 再解 local detail

### 3.5 RQ5: Operational Cost — 这是数据库视角最有价值的部分

**Setup**: Avg. Operation Latency/Query (= construction time + query time), Normalized Utility (6 个 metric min-max 归一化均值)。

**O7 (Localized Maintenance)** — Figure 11 的 Pareto frontier:
- LightMem: 48.3 utility @ 3.67s — **最优 cost-efficiency**
- MemTree: 63.5 utility @ 15.9s
- MemoChat: 28.0 @ 15.4s (低效)
- Mem0: 21.4 @ 35.9s (低效)
- MemoryOS: 82.0 @ 28.6s
- Cognee: >84 utility @ 116.5s
- Zep: >84 utility @ 155.1s

LongBench 上 latency 分化更大: LightMem 17.3s, MemTree 116.7s, Mem0 374.2s, MemoChat 460.2s, MemoryOS 490.0s, A-MEM 552.1s。

**Finding 5 (Operational Scaling Rule)**: 效率由 **maintenance scope** 决定，不是 structure 本身:
1. Localized update + search 最优 (LightMem, MemTree)
2. Richer organization 只在 upkeep 不触发 broad recomputation 时有效
3. Long-context workload 下 whole-memory coordination 是 dominant cost driver

---

## 4. Fine-Grained Component Comparison: 4 个 Module 的 Ablation

### 4.1 M1: Representation (Table 3)

LightMem 三变体:
- User-Only Raw: 24.2 EM / 38.9 Ans F1 / 26.0 Substr EM / 31.4 ROUGE-L (全胜)
- User-Only Summary: 8.5 / 15.6 / 11.7 / 17.4 (大幅退化)
- User-Only Compressed: 23.6 / 38.6 / 10.7 / 19.1 (LoCoMo 接近, LongMemEval 崩)

MemTree Flat-biased vs Deeper Tree: 仅微小提升 (18.2→18.7 EM)。

**Finding 6 (Representation Granularity)**: **保留 usable evidence 比让 memory 更紧凑或更结构化更重要**。这是反直觉的——大家直觉认为 summary/structure 会更好，但实测发现 abstraction 主要丢失信息。Hierarchy 改善 access，不能恢复已删除内容。

### 4.2 M2: Extraction (Table 4)

- MemoChat Heuristic Topic vs LLM Topic: Heuristic 在 LongMemEval 反超 (10.7 vs 7.3 Substr EM)，LoCoMo 持平
- MemOS Fast Memorize vs Fine Memorize: Fast 在 LoCoMo 大幅胜出 (25.5 vs 2.5 EM), Fine 在 LongMemEval 略胜
- LightMem Hybrid Raw (user+assistant) vs User-Only Raw: Hybrid 在 LoCoMo 略胜 (25.5 vs 24.2)

**Finding 7 (Late Filtering Principle)**: **Extraction 应该在 write time preserve context，不要激进 filter**:
1. Coarser segmentation 保留 thread-spanning cues
2. Limited rewriting 保留 compositional reasoning 需要的 details
3. 同时存 user 和 assistant turns 保留 clarification cues

### 4.3 M3: Retrieval/Routing (Table 5)

- A-MEM Hybrid-Balanced vs Hybrid Sparse-Leaning: Balanced 胜 (24.6 vs 23.0 Ans F1)
- SimpleMem No Planning vs Planning Only vs Planning+Reflect: Planning Only 最优 (20.7 / 90.6 / 21.7 / 27.9)，加 reflect 反而退化

**Finding 8**: 不要无脑加 complexity:
1. Moderate hybrid fusion > sparse-leaning fusion (处理语义相似但词形多样)
2. Lightweight planning > direct retrieval (multi-constraint query)
3. **加 reflect 在 route 已确定后反而 hurt**——多余 deliberation 削弱 routing decision

### 4.4 M4: Maintenance (Figure 12)

- MemoryOS Conservative-Merge (raise topic-similarity threshold): 23.2→23.5 Ans F1, 22.4→22.8 Substr EM (提升)
- MemoryOS Delayed-Flush: 23.2→20.6/19.5 (退化)
- MemoChat Topic1 (force single-topic): 16.6→16.2 / 18.4→16.8 (退化)

**Finding 9 (Maintenance Design Principle)**: **Conservative integration 是最佳 default**:
1. Conservative merging 保留 cross-turn linkages 给 long-horizon reasoning
2. Delayed flushing 让 recent evidence 在 query time 仍然 fragmented
3. Overly coarse summarization 抹掉 sparse but useful cues

---

## 5. 关键 Insight 综合与 Intuition Building

把所有 findings 串起来，几个 deep insights:

### 5.1 "Late Filtering" 原则

这是最反直觉但最一致 finding。整个 paper 在 M1, M2, M4 都指向同一个结论: **写的时候不要丢信息，读的时候再 filter**。

直觉解释: LLM 在 write time 不知道未来 query 长什么样。一旦在 extraction/summarization 阶段丢了某个 date 或 name, 后面 multi-hop reasoning 需要这个 detail 时就永远捞不回来。Maintenance 阶段的 "Conservative-Merge" 之所以胜过 "Delayed-Flush" 和 aggressive summarization, 是因为它保留 cross-turn linkages。

数学上可以理解成一个 information-theoretic argument:
$$
I(\text{retrieved}; \text{query}) \leq I(\text{stored}; \text{query}) \leq I(\text{raw}; \text{query})
$$
任何 extraction 一步 $S$ 都会引入信息损失 $H(\text{raw} | S(\text{raw}))$，这个损失对未知未来 query 是不可逆的。

### 5.2 Evidence Completion 而非 Top-1 Ranking

RQ2 的发现: SimpleMem Recall@1 最高但 Recall@10 不行; A-MEM/MemTree 反过来。说明 memory retrieval 本质是 **evidence assembly** 问题，相关 support 可能 old, scattered, 跨多 turn。

这暗示未来的 memory system 应该有显式的 "evidence completeness estimator"，而不是单纯做 similarity ranking。

### 5.3 Maintenance Scope 决定 Cost, 不是 Structure 本身

RQ5 的 Pareto frontier 显示: LightMem/MemTree (localized maintenance) 在 cost-efficiency 上 dominate。Cognee/Zep (graph-wide consolidation) 高 utility 但 cost 暴涨 2 个数量级。

直觉: graph memory 的写入代价是 $O(\text{affected subgraph})$，而 multi-version append-only log 是 $O(1)$。所以 graph memory 适合 "需要 revision + temporal reasoning" 的场景 (LongMemEval Knowledge Update 上 Zep 确实最强)，但生产部署要考虑 write amplification。

### 5.4 Backbone 只在 Grounding 成功后起作用

RQ3 O5: 换 LLM backbone 主要改变 answer realization，不改变 memory pipeline ordering。这意味着 **memory system 的好坏应该在 retrieval/grounding 阶段 evaluate, 而不是 end-to-end F1**。End-to-end F1 把 retrieval quality 和 generation quality 混在一起，掩盖了真正的 bottleneck。

### 5.5 Temporal State Externalization 是 Pipeline-Level Problem

RQ3 Finding 3: revision 应该 baked into representation (entity-bound), query-time selectivity 应该匹配 workload bottleneck, LLM scaling 只在 grounding 后有用。这意味着如果 memory system 在 representation 层没把 "later fact" 绑定到 "same entity/event"，而是 append 成 undifferentiated text，再强的 LLM 也救不了。

---

## 6. Paper 的局限与未来方向

Paper 自己承认的一些 gap:
1. **Eval coverage**: 没覆盖一些新的 memory systems 如 Letta 最新版本、MemoRAG 的 parametric optimization path
2. **Cost model**: latency 测量没有 isolate CPU/GPU/IO breakdown
3. **Long-horizon 评估**: LongBench/LongMemEval/LoCoMo 的 horizon 仍然有限，真正 lifelong learning (LifelongAgentBench) 的 thousand-session 级别没测
4. **Multi-modal memory**: 文本 only，没有 vision/audio streams

未来方向 (paper + 我的联想):
- **Native memory OS**: MemOS, MemoryOS 已经在朝这个方向走，参考 https://github.com/MemTensor/MemOS
- **Probabilistic memory**: 用 Bayesian view 处理 conflict，而非 timestamp-only
- **Parametric + symbolic hybrid**: 把 parametric memory (LoRA on facts) 和 symbolic memory (graph) 显式 bind
- **Memory-native benchmarks**: 需要类似 SWE-bench 的真实 agent workload benchmark，而非 chatbot-centric LoCoMo

---

## 7. 对你 (Andrej) 的针对性思考

你之前在 "Software 2.0/3.0" 框架里讨论过 LLM 作为新计算 substrate。这篇 paper 实际上在问: agent memory 是这个 substrate 上的 "main memory + filesystem"。它揭示了:

1. **Memory hierarchy 需要重新设计**: 不是 L1/L2/L3 cache，而是 (a) in-context KV cache (working memory), (b) external vector/graph store (long-term), (c) parametric weights (consolidated knowledge)。Letta 用 OS metaphor (core memory vs archival), MemOS 用 MemCube 统一这三层。这非常像你讲的 "memory hierarchy" 但加了 LLM-specific 语义层。

2. **Write path 比 read path 难**: 这篇 paper 反复强调 "Late Filtering Principle" 和 "Conservative Consolidation"——因为 LLM write-time 不知道 future query distribution。这本质是 "memory compression 的不可逆性"，跟传统 DB 的 overwrite 不一样。

3. **Benchmark gap**: F1/BLEU 把 memory system 当 black box, 掩盖了 retrieval vs generation bottleneck。真正应该测的是 evidence-level Recall@K + temporal consistency + cost-per-query 的 Pareto frontier。这篇 paper 的四模块 framework 是朝这个方向迈的第一步。

---

## Reference Links

- Paper code (作者开源): https://github.com/OpenDataBox/MemoryData
- MemOS: https://arxiv.org/abs/2507.03724
- Mem0: https://arxiv.org/abs/2504.19413
- Zep: https://arxiv.org/abs/2501.13956
- MemGPT/Letta: https://arxiv.org/abs/2310.08560
- A-MEM: https://arxiv.org/abs/2502.12110
- LightMem: https://arxiv.org/abs/2510.18866
- MemoRAG / MemAgent: https://arxiv.org/abs/2507.02259
- MEM1: https://arxiv.org/abs/2506.15841
- LoCoMo benchmark: https://aclanthology.org/2024.acl-long.201/
- LongMemEval: https://arxiv.org/abs/2410.10813
- LongBench: https://arxiv.org/abs/2308.14508
- LifelongAgentBench: https://arxiv.org/abs/2505.11942
- MemoryAgentBench (ICLR 2026): https://openreview.net/forum?id=MemoryAgentBench
- Anthropic context engineering: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- CIDR 2026 Agent-First Data Systems: https://www.cidrdb.org/cidr2026/
- VLDB LLM for Data Management tutorial: https://www.vldb.org/

总结一句: 这篇 paper 给了 agent memory 第一个真正的 **database-style 系统 benchmark**，核心 takeaway 是 **不要在 write time 丢信息**、**retrieval 是 evidence assembly 问题**、**maintenance scope 决定 cost**、**LLM backbone 只在 grounding 成功后才有用**。整个领域离真正的 "agent-native memory system" 还差一个 native cost model + 真实 lifelong workload benchmark + multi-modal extension。
