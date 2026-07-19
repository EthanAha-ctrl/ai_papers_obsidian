---
source_pdf: Agentic Very Long Video Understanding.pdf
paper_sha256: 1ff65a471f1120dac5ed8361cde4ae8687f9f2238e1375e82a851d4cb9bc0da0
processed_at: '2026-07-18T05:14:41-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Agentic Very Long Video Understanding — 深度技术解析

Andrej，这篇 paper 我觉得非常对你胃口，因为它把 "long-context" 这个老问题用 "agentic + structured memory" 的方式重新定义了一遍，而且赌注很大：**50 小时（一周连续 egocentric video）** 的 horizons，这已经超出任何 MLLM context window 的物理边界了。我下面尽量把每个关键设计决策、公式、实验数据都拆开讲，目标是让你 build 出对 "why this works" 的 intuition。

参考链接：
- Paper PDF (arXiv): https://arxiv.org/abs/2507.18990
- EgoLife dataset & benchmark: https://github.com/EgoLife2025/EgoLife
- LangGraph (agent orchestration 框架): https://github.com/langchain-ai/langgraph
- SigLIP 2 (visual encoder): https://arxiv.org/abs/2502.14786
- GraphRAG (Microsoft, 思想源头): https://arxiv.org/abs/2404.16130
- Video-MME benchmark: https://arxiv.org/abs/2405.21075
- LightRAG (类似的 graph RAG baseline): https://arxiv.org/abs/2410.05779

---

## 1. 问题重构：从 "long video" 到 "very long video"

### 1.1 长度的范式跃迁

文献里 "long" 一直在漂移：MSR-VTT (~60s) → Ego4D (分钟级) → Video-MME Long (30-60 min) → **EgoLife (50 hours / 7 days)**。后者才是真正的 "very long"。

驱动场景非常具体：Ray-Ban Meta glasses、Project Aria、Echo Frames 这类 always-on wearable，需要 recall user 一周内 "我周四早上有没有和 Tasha 说过狗的事"。这种 query 有几个 MLLM 难以处理的特征：

1. **Entity-centric**：query 是关于特定 person/object/location 的
2. **Multi-hop temporal**：需要先定位事件，再推断关系，跨多天
3. **Compositional**：例如 "all times I talked to person X this week" —— 需要 filter + aggregate
4. **Habit tracking**：例如 "how often did I drink water this week?" —— 需要重复行为统计

naive RAG 在这里会挂，因为 embedding-based retrieval 无法保持 entity identity 的连贯性（同一个人在不同片段被描述的方式不同，embedding 会漂移）。

### 1.2 形式化任务

$$H : (\mathcal{V}, \mathcal{AT}, Q) \to A$$

- $\mathcal{V} = \{v_t\}_{t=1}^{T}$：1 FPS 采样的 video frames
- $\mathcal{AT} = \{u_i, t_{start_i}, t_{end_i}\}_{i=1}^{N}$：transcribed speech $u_i$ 带时间戳
- $Q$：natural language query
- $A$：textual answer

**Intuition**：这个 mapping 太大，单次 forward pass 不可能。必然要 decompose 成 sub-tasks + retrieval。

---

## 2. 核心创新：Entity Scene Graph with Temporal Edges

这是这篇 paper 的灵魂。我把它和 GraphRAG (Microsoft) 对比一下你就懂了。

### 2.1 Graph 定义

$$G = (V, E)$$

- **Nodes** $V$：entities，每个 node 有 type $\tau(v) \in \{\text{person}, \text{object}, \text{location}\}$
- **Edges** $E$：relationships，每条 edge 是一个 5-tuple：

$$e = (v_s, v_t, r, t_{\text{start}}, t_{\text{end}})$$

变量含义：
- $v_s$：source node
- $v_t$：target node  
- $r \in \mathcal{R}$：relation type
- $t_{\text{start}}, t_{\text{end}}$：关系成立的时间区间

$$\mathcal{R} = \{\text{talks-to}, \text{interacts-with}, \text{mentions}, \text{uses}\}$$

存储格式（SQLite3 一行）：

$$(v_s, \tau(v_s), v_t, \tau(v_t), r, t_{\text{start}}, t_{\text{end}}, d^*)$$

其中 $d^*$ 是抽取这条 edge 时的 supporting text snippet，方便后续 reasoning。

**Intuition**：GraphRAG 是静态 graph（没有时间维度），VideoMindPalace 是 room-level 结构 graph（无法泛化到 open scenes），本文核心创新是 **每条 edge 都带时间区间**。这让 graph 变成 "time-aware" 的，可以增量构建 —— 新 video 到了就 append 新的 edges，不需要重建。

### 2.2 Graph 构建 Pipeline

从 text documents $\mathcal{D}$（audio transcripts + visual captions + predicted locations）抽取：

$$(V_d, E_d) = \mathcal{F}(d)$$

$\mathcal{F}$ 是 LLM-based extractor（基于 LangChain 的 `LLMGraphTransformer`，用 GPT-4.1），对每个 document $d$ 联合抽取 entities 和 relationships。

聚合：

$$(V, E) = \left(\bigcup_{d \in \mathcal{D}} V_d, \bigcup_{d \in \mathcal{D}} E_d\right)$$

**关键技术细节**：`LLMGraphTransformer` 不支持 metadata annotation，所以作者额外加了一个 **temporal annotator** step（用 LLM），根据 transcript timestamps 和 caption 时间区间给每条 edge 打 $t_{\text{start}}, t_{\text{end}}$。Prompt 规则很巧妙：
- First, try to use only timestamps already present in transcript utterances
- If no supporting utterances exist, use the entire interval from the caption as $t_{\text{start}}$ and $t_{\text{end}}$

### 2.3 Graph 统计

EgoLife 7 天数据，共抽取 **13,968 条 relationships**。分布很有意思：

- Source node：person (13,930 / 13,968 = 99.7%)
- Target node：location (1,314), object (6,449), person (6,167)

**Intuition**：graph 高度 "person-centric"（因为 egocentric video 的本质），target 分布在 object 和 person 之间，说明既捕捉了 "人-人交互" 也捕捉了 "人-物使用"。

---

## 3. EGAgent 架构 — 完整拆解

### 3.1 六大组件

```
┌──────────────────────────────────────────────────────────────┐
│                    EGAgent Pipeline                         │
│                                                              │
│  ┌──────────────┐                                            │
│  │ Planning     │ ── decompose Q into subtasks              │
│  │ Agent        │ ── select tool T_i per subtask            │
│  └──────┬───────┘                                            │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────────────────────────────────────────┐         │
│  │   Retriever Tools (parallel selection)          │         │
│  │  ┌──────┐  ┌──────┐  ┌──────────┐              │         │
│  │  │Visual│  │Audio │  │Entity    │              │         │
│  │  │Search│  │Search│  │Graph     │              │         │
│  │  │Tool  │  │Tool  │  │Search    │              │         │
│  │  └──┬───┘  └──┬───┘  └────┬─────┘              │         │
│  └─────┼─────────┼───────────┼────────────────────┘         │
│        └─────────┼───────────┘                              │
│                  ▼                                          │
│         ┌──────────────────┐                                │
│         │ Analyzer Tool    │ ── filter, extract evidence    │
│         └────────┬─────────┘                                │
│                  ▼                                          │
│         ┌──────────────────┐                                │
│         │ Working Memory M │ ← accumulate cross-modal       │
│         └────────┬─────────┘    evidence                    │
│                  │                                          │
│                  ▼ (loop until all subtasks done)           │
│         ┌──────────────────┐                                │
│         │ VQA Agent        │ ── final answer A              │
│         └──────────────────┘                                │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 Algorithm 1 详解

```
Input: Q, (Video, Audio, Entity Graph)
1. M ← ∅  (initialize working memory)
2. SubtaskList ← PlanningAgent.decompose_and_select(Q)
   SubtaskList = {(S_1, T_1, q_1), ..., (S_N, T_N, q_N)}
3. for each (S, T, q) in SubtaskList:
   a. RetrievedData ← T(q)  
      - Visual: hybrid semantic + attribute search
      - Audio: transcript search (LLM or BM25)
      - Entity Graph: SQL queries (strict-to-relaxed)
   b. Analysis ← AnalyzerTool.analyze(RetrievedData, S)
   c. M ← M ∪ {Analysis}
4. A ← VQAAgent.answer(Q, M)
5. return A
```

关键设计点：
- **Joint decomposition and tool selection**：planner 一次性产出整个 plan（最多 5 个 sub-tasks），而不是 ReAct 风格的 step-by-step。这减少了 LLM 调用次数，也避免了 agent "走偏"。
- **Early exit**：`grade_plan_completion` 检查 working memory 是否已经包含未来 sub-task 的答案，如果是就直接跳到 VQA Agent。这是很实用的工程优化。

### 3.3 三个 Retriever Tools 的技术细节

#### (a) Visual Search Tool ($\text{Tool}_{\text{vis}}$)

每帧用 SigLIP 2 编码：$\phi_I(v_t) \in \mathbb{R}^d$。Query 用 text encoder 编码：$\phi_T(q)$。Retrieval：

$$\cos(\phi_T(q), \phi_I(v_t))$$

支持 attribute filters $f$（如 "kitchen"、"morning"），返回 k-nearest neighbors。

**关键 prompt 设计**：Visual Search 的 system prompt 明确告诉 LLM "不要用 generic common nouns or times of day"、"不要用 specific named entities (non-famous people)"，因为 SigLIP 2 训练时没见过这些。这是 very practical 的工程经验。

#### (b) Audio Transcript Search Tool ($\text{Tool}_{\text{aud}}$)

两种变体：
- **LLM-based search**：把整个 transcript 喂给 LLM，按天并行（绕过 context limits）。质量高但慢。
- **BM25-based lexical search**：传统 IR 方法，快但 recall 低。

Table 4 显示 LLM search 比 BM25 多花 44 秒（169s vs 125s），但准确率高 6.8%。

#### (c) Entity Graph Search Tool ($\text{Tool}_{\text{eg}}$) — 这是核心

Query 是 SQL，schema 是 SQLite table：

```sql
entity_graph_table(
  id INTEGER PRIMARY KEY,
  day INTEGER,           -- 1 to 7
  start_t INTEGER,       -- e.g. 132609 for 13:26:09
  end_t INTEGER,
  transcript TEXT,
  source_id TEXT,
  source_type TEXT,      -- Person/Location/Object
  target_id TEXT,
  target_type TEXT,
  rel_type TEXT          -- TALKS_TO/INTERACTS_WITH/MENTIONS/USES
)
```

**Strict-to-relaxed 查询策略**（这是非常聪明的设计）：

1. **Strict**：exact day + exact timestamp range + exact source_id + exact target_id + exact rel_type
2. **Relax time**：same day, exact entities, same rel_type
3. **Relax day**：all days ≤ query_time day, exact entities, same rel_type
4. **Relax entity match**：same rel_type, use `LIKE` substring for source/target_id（用单词匹配增加召回）
5. **Relax rel_type**：search by entity only

**Intuition**：这个 progressive relaxation 模拟了人类 reasoning —— 先精确找，找不到就逐步放宽。这比单一的 dense retrieval 更可控，也比纯 semantic search 更精确。

---

## 4. 实验数据深度分析

### 4.1 EgoLifeQA 主结果（Table 1）

| Method | VLM | EG | Avg Acc | Tokens |
|--------|-----|----|---------|--------|
| Gemini 2.5 Pro (uniform) | Gemini 2.5 Pro | ✗ | 46.8% | 807K |
| EgoButler (GPT-4o) | GPT-4o | ✗ | 36.2% | 19K |
| EGAgent (GPT-4.1, F+T) | GPT-4.1 | ✗ | 48.6% | 551K |
| **EGAgent (GPT-4.1, EG+F+T)** | GPT-4.1 | ✓ | 50.7% | 571K |
| **EGAgent (Gemini 2.5 Pro, EG+F+T)** | Gemini 2.5 Pro | ✓ | **57.5%** | 880K |

**关键观察**：
- Entity graph 在同一 backbone 上带来 +13.8% (GPT-4.1) 提升，Gemini 上 +10.7%
- 相比 previous SOTA (EgoButler GPT-4o = 36.2%) 提升 **+20.6%**
- Token 用量是 Gemini uniform sampling 的 ~1.1x，但准确率高 10.7%

### 4.2 按任务类型拆解（Figure 4）

5 个 category 中，Entity Graph 收益最大的是：
- **RelationMap**：+20.8% over Gemini 2.5 Pro（需要 multi-hop relational reasoning）
- **TaskMaster**：+22.2% over Gemini 2.5 Pro（需要 entity + action tracking）

这两个 category 之前是 MLLM 的死穴，因为需要跨 entity 推理，单帧 caption 无法捕获。

### 4.3 Video-MME (Long) 对比（Table 2）

| Method | Backbone | Acc | # Frames |
|--------|----------|-----|----------|
| Gemini 2.5 Pro (native) | Gemini 2.5 Pro | **82.0%** | 256 |
| VideoDeepResearch | DeepSeek-r1 + Qwen2.5-VL-7B | 72.4% | 32 |
| **EGAgent (Gemini 2.5 Pro)** | Gemini 2.5 Pro | 74.1% | 50 |
| EGAgent (Qwen2.5-VL-7B) | Qwen2.5-VL-7B | 47.8% | 50 |
| AdaVideoRAG | Qwen2.5-VL-7B | 47.7% | 768 |

**关键洞察**：EGAgent 用 50 frames 匹配 AdaVideoRAG 用 768 frames 的性能 —— **10x 的 frame 效率提升**。这证明 structured graph retrieval 比 dense frame retrieval 更 sample-efficient。

### 4.4 Tool Ablation（Table 5）

| Configuration | Avg Acc | RM | TM |
|---------------|---------|----|-----|
| EG only | 37.6% | 31.2% | 44.4% |
| F only | 34.6% | 28.0% | 34.9% |
| T only | 45.6% | 44.0% | 66.6% |
| F+T | 48.6% | 40.0% | 61.9% |
| **EG+F+T** | **50.7%** | **66.7%** | 50.7% |

**Intuition**：
- T (audio transcript) 单独就很强 —— 说明 egocentric video 中说话信息密度极高
- F (visual) 单独很差 —— RelationMap 只有 28%（接近随机 25%），因为视觉搜索不知道 "谁是谁"
- EG 单独提升 RelationMap 到 31.2%，但加上 F+T 后跳到 **66.7%** —— 说明 cross-modal fusion 才是关键
- 真正的 magic 是 **EG 提供 entity identity，F+T 提供具体 evidence**

### 4.5 Oracle Upper Bound（Table 6）

| Method | Acc |
|--------|-----|
| EGAgent (Gemini 2.5 Pro) | 57.5% |
| Gemini Oracle (F+T, 50 frames) | **68.7%** |

**重要发现**：即使给 perfect temporal localization（ground-truth timestamps），MLLM 也只能到 68.7%。这说明：
1. retrieval 还有 ~11% 的提升空间（57.5% → 68.7%）
2. 但 MLLM reasoning 本身有天花板，需要更强的 backbone 或更好的 reasoning 方法

### 4.6 Retrieval Recall 分析（Table 7）

这是我最喜欢的实验之一，回答了 "为什么 EGAgent 有效"：

recall@10sec：
- Visual search: **0.857**（很强）
- Audio search: 0.218（弱，因为 analyzer 偶尔不显式输出 timestamp）
- Entity graph: 0.127（弱，因为低维投影）
- **EGAgent overall: 0.884**（融合后最强）

recall@1hour：
- Visual: 0.930
- Audio: 0.417
- Entity graph: 0.658（coarse-grained 强）
- EGAgent overall: 0.962

**Intuition**：三个工具是 complementary 的。Visual 强在 fine-grained，Entity graph 强在 coarse-grained + cross-day，Audio 介于中间。组合后从 10 秒到 1 小时窗口都达到 >0.88 recall。

### 4.7 延迟分析（Table 8）

| Component | BM25 setup | LLM setup |
|-----------|-----------|-----------|
| Planning | 3.1s | 3.1s |
| Visual retriever | 4.6s | 4.5s |
| Visual analyzer | 41.1s | 41.8s |
| EG search | 8.4s | 10.2s |
| Audio retriever | 1.7s | — |
| Audio analyzer | 8.2s | 35.4s |
| VQA agent | 6.9s | 6.9s |
| **Total** | **125s** | **169s** |

**关键发现**：MLLM analyzer（处理 retrieved frames）是最大瓶颈（~41s），占 33% 总延迟。Entity graph search 只占 8% 延迟但贡献巨大收益 —— 这是工程上的 sweet spot。

---

## 5. 与现有 Agentic / RAG 方法的对比

| Method | Graph Type | Temporal? | Modality | Limitation |
|--------|-----------|-----------|----------|------------|
| GraphRAG (Edge et al.) | Entity graph | ✗ | Text | 无时间维度 |
| VideoMindPalace | Spatio-temporal | Partial | Video | 依赖 room-level structure |
| GraphVideoAgent | Caption-derived | ✗ | Video | 静态 graph |
| RAVU | Spatio-temporal | ✓ | Video | 一次性构建，无增量 |
| AdaVideoRAG | Adaptive | ✗ | Video | 需 768 frames |
| **EGAgent** | **Entity + temporal** | **✓** | **Cross-modal** | **依赖上游感知质量** |

EGAgent 的独特性在于：**temporal edges + incremental construction + cross-modal reasoning**。其他方法要么没时间维度，要么一次性构建无法增量。

---

## 6. 我的几点 Intuition 总结

### 6.1 为什么 Entity Graph 这么有效？

不是因为 graph 本身有魔力，而是因为：
1. **Discrete symbolic representation**：把连续的 video signal 离散化成 (entity, relation, time) 三元组，让 LLM 可以用 SQL 精确 query，而不是依赖 noisy 的 dense retrieval
2. **Entity identity preservation**：同一个人在不同 caption 里描述不同，但 graph node 是统一的 —— 这解决了 "coreference across time" 问题
3. **Compositional query support**：SQL 天然支持 `WHERE day <= ? AND rel_type = 'TALKS_TO' AND target_id LIKE '%Alice%'` 这种组合查询

### 6.2 为什么 Planner 一次性出 plan 而不是 ReAct？

ReAct (step-by-step) 在 long video 场景有两个问题：
1. 每步都要 LLM call，500 个 query × 5 steps = 2500 LLM calls，太慢
2. Agent 容易 "走偏"，特别是在 50 小时视频里找一个具体事件

Joint decomposition 让 LLM 一次性看全 query，规划好 5 个 sub-tasks，然后 pipeline 化执行。这是 **planning-heavy vs reasoning-heavy** 的 trade-off，在 very long video 场景下前者更合适。

### 6.3 Token Efficiency 的真正来源

Table 2 显示 EGAgent 用 50 frames 匹配 AdaVideoRAG 的 768 frames。原因：
- Entity graph 已经把 "who/what/where" 的 structural information 压缩成 SQL-queryable 的形式
- Visual search 只在需要 fine-grained visual evidence 时才触发
- 这本质上是 **symbolic-neural hybrid** 的胜利

### 6.4 上限在哪里？

Oracle 实验揭示的天花板是 68.7%（用 Gemini 2.5 Pro + perfect retrieval）。剩余的 31.3% 错误来自：
1. MLLM 的 reasoning 能力限制（multi-hop 推理错误）
2. 50 frames 的视觉信息可能不足以回答某些 fine-grained 问题
3. Entity graph 抽取的 recall/precision 上限

**推论**：要突破这个天花板，需要 (a) 更强的 reasoning backbone（如 o3 / R1 类），(b) 动态 frame budget（不是固定 50），(c) iterative graph refinement。

---

## 7. 我会向作者提的几个问题

1. **Entity graph 抽取的 cost**：13,968 条关系用了多少 GPT-4.1 tokens？这个 pipeline 能否用更小的 open-source LLM 替代？这是 deployment 的关键。

2. **Graph 的一致性 / 去重**：同一个人可能在不同 caption 里被抽成不同的 node ID（"Jake"、"he"、"the user"）。Prompt 里有 coreference resolution 指令，但实际效果如何？是否做过 node deduplication 的 ablation？

3. **Dynamic frame budget**：固定 50 frames 是不是 sub-optimal？某些 query（如 "what did I eat for breakfast this week?"）可能需要 200 frames，而 "did I see Alice on Tuesday?" 可能只需要 5 frames。

4. **Iterative graph refinement**：当前 graph 是 one-shot 抽取的。如果 retrieval 失败，能否让 agent 反过来 refine graph（例如重新抽取某些 caption 的关系）？

5. **Cross-video generalization**：EgoLifeQA 只有 Jake 的视角。如果换一个人，graph 结构会完全不同。如何做 transfer learning？

---

## 8. 对未来工作的启示

这篇 paper 给我几个更深的思考：

### 8.1 Symbolic Memory 回归

LLM 时代大家都在堆 dense retrieval / embedding，但 paper 证明：**structured symbolic representation (SQL-queryable graph) + LLM reasoning > pure dense retrieval**。这其实是 classic AI (GOFAI) 的某种回归 —— symbolic layer 提供 compositional reasoning 的脚手架，neural layer 提供 perception 和 language understanding。

### 8.2 Agentic ≠ ReAct

ReAct (Yao et al. 2022) 是 agentic 的经典范式，但在 very long context 下，**plan-then-execute** 更高效。这篇 paper 的 joint decomposition 就是这个思路。我预测未来会有更多 "hierarchical planning" 方案：top-level planner + mid-level tool router + low-level executor。

### 8.3 Temporal 是 video understanding 的第一公民

所有 visual-language model 都在卷 spatial reasoning（OCR、grounding、detection），但 very long video 的核心 challenge 是 **temporal**。这篇 paper 用 $t_{\text{start}}, t_{\text{end}}$ 给每条 edge 打时间戳，让 graph 变成 time-aware。我预期未来会有 **temporal-aware attention mechanisms**、**time-conditioned retrieval**、**event-timeline summarization** 等新方向。

### 8.4 Wearable AI 的真实需求

Ray-Ban Meta glasses 已经出货百万级。真实用户不会问 "describe this video"，他们会问 "我上次在星巴克和谁聊过 blockchain？" 这种 query 需要：
- Location filtering (星巴克)
- Person identification (谁)
- Topic retrieval (blockchain)
- Temporal localization (上次)

这恰好是 EGAgent 设计的四个 search 维度。这篇 paper 不只是 academic exercise，它是 **product-driven research** 的典范。

---

## 9. 总结

EGAgent 的核心贡献可以浓缩成一句话：**在 very long egocentric video 上，用一个 temporally-annotated entity scene graph 作为 symbolic memory，配合 agentic planning + cross-modal retrieval，比 dense frame sampling 或 naive RAG 都更有效**。

工程上值得学习的点：
1. **Strict-to-relaxed SQL querying** —— 处理 noisy extracted graph 的实用策略
2. **Joint decomposition + early exit** —— agentic latency 优化
3. **Cross-modal fusion at memory level** —— 不在 input level fuse，而在 working memory level fuse，更灵活
4. **Temporal annotation as post-hoc step** —— 把 graph 抽取和时间标注解耦，让上游工具（LLMGraphTransformer）可以独立选择

学术上值得 follow up 的点：
1. **Graph quality upper bound**：抽取 graph 的 recall/precision 如何影响下游？目前没量化
2. **Reasoning backbone ceiling**：Oracle 实验揭示 68.7% 天花板，如何突破？
3. **Multi-user graphs**：EgoLife 有 6 个人但只用 Jake 视角，cross-perspective graph fusion 是 open problem
4. **Streaming setting**：当前是 batch 处理 50 小时，real-time streaming（眼镜实时录实时答）如何设计？

这篇 paper 是 very long video understanding 领域的一个 solid milestone，我预期会引发一系列 follow-up work，特别是在 temporal graph construction quality、agentic planning efficiency、cross-video transfer 这几个方向。

参考阅读：
- EgoLife 原始 paper (CVPR 2025): https://arxiv.org/abs/2503.22614
- DrVideo (long video as document): https://arxiv.org/abs/2406.12846
- Video-RAG: https://arxiv.org/abs/2502.01549
- VideoAgent (Wang et al., ECCV 2024): https://arxiv.org/abs/2403.09627
- ReAct 原始 paper: https://arxiv.org/abs/2210.03629
- Project Aria: https://www.projectaria.com/
