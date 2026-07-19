---
source_pdf: Benchmarking Agent Memory in Interdependent Multi-Session Agentic Tasks.pdf
paper_sha256: 379c6eb30b30b2e6a4643ea34c2e9d714a68e7b8ff39860110b38ffeb950d76c
processed_at: '2026-07-18T16:25:18-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# MEMORYARENA: 把 Agent Memory 当作"功能组件"而非"被动存档"来评估

## 1. Paper 的核心直觉

这篇 paper 的 motivation 来自一个非常尖锐的观察：现有关于 LLM agent memory 的 evaluation 存在一个**严重的二分裂**。

- 第一类 benchmark（LoCoMo, LongMemEval, MemoryAgentBench）：测试 **factual recall**。Agent 看一段长 conversation 或长文档，然后回答 QA。问题是 —— agent 根本不需要 **act**，环境也不会给 feedback。最近 SOTA 在这些 benchmark 上已经接近 saturated。
- 第二类 benchmark（WebArena, SWE-Bench, WebShop）：测试 **action**。Agent 在动态环境里 navigation、写代码、买东西。问题是 —— 它们通常是 **single-session**，历史就是扁平 context，不要求 **persistent memory across sessions**。

MEMORYARENA 的 claim：realistic agent deployment 里，memorization 和 action 是 **tightly coupled** 的。Agent 一边和环境交互一边产生 experience，这些 experience 必须被 distill 到 memory 里，然后 memory 必须反过来 conditioning 后续的 decision。论文把这种循环称作 **Memory-Agent-Environment Loop**，并强制要求 subtask 之间有 **causal dependency**（后面 subtask 的 success 必须依赖前面 subtask 的 latent state）。

参考链接：
- Paper 项目主页: https://memoryarena.github.io/
- LoCoMo paper: https://aclanthology.org/2024.acl-long.754/
- LongMemEval: https://arxiv.org/abs/2410.10813
- WebShop: https://arxiv.org/abs/2207.01206
- WebArena: https://arxiv.org/abs/2307.13854

---

## 2. 任务的四个 Domain：为什么这四个？

让我详细讲一下四个 environment 各自的设计意图，这对 build intuition 非常关键。

### 2.1 Bundled Web Shopping（150 tasks，6 subtasks，41.5k trace）

**设计意图**：测 *compatibility tracking*。

数据构造 pipeline：
1. 从 WebShop 的 hierarchy 出发，cluster 共享 penultimate-level category path 的 products（比如 "Electronics > Television & Video > Televisions > TV Mounts" 和 "...> LED & LCD TVs"）。
2. 抽取每个 product 的 key attributes（size, mount type, wattage 等）。
3. 用 commonsense reasoning 构造 **accept-reject map**：75-inch TV accepts 70-inch stand，rejects 50-inch stand。
4. Human annotator 手工 verify 所有 compatibility chain。
5. 每个 session 加入 incompatible distractors（hard negatives）+ compatible distractors + 一个 selection constraint（highest rating / lowest price）确保 unique 解。

**Memory 需求**：第 1 个 subtask 买的 camera body 的 mount type（EF / F / E-mount）必须被记住，否则第 2 个 subtask 选 lens 时无法 filter。这是 **explicit latent constraint** —— environment 不会重述。

### 2.2 Group Travel Planning（270 tasks，5-9 subtasks，40.6k trace，深度 4）

**设计意图**：测 *preference chain + relational reasoning*。

基于 TravelPlanner 扩展。Base itinerary 是 30 个 daily slot（3 meals + accommodation + sightseeing）。然后逐个加入 5-8 个新 traveler，每人有：
- **JOIN constraint**：「我要 day 2 dinner 跟 Rebecca 一起」 —— agent 必须查 Rebecca 当时被 assign 的 dinner。
- **RELATION constraint**：「我要住 rating 比 Rebecca 高至少 2 级的 hotel」 —— 需要 retrieve Rebecca 选的 hotel 的 rating。

**Memory 需求**：dependency chain 深度达 4（A → B → C → D 引用 C，C 引用 B，B 引用 A）。每多一层，agent 都必须正确 recall 上一层具体决策的 attribute。

### 2.3 Progressive Web Search（256 tasks，2-16 subtasks，122.4k trace）

**设计意图**：测 *compositional information accumulation*。

基于 BrowseComp-Plus 830 entries 过滤：
1. 先用 LLM agent + web tool 尝试一次性回答，**移除能直接答的**（这些不需要跨步 memory）。
2. 把剩余每个 query 分解成 ordered subqueries，每个 subquery 引入一个新 constraint。
3. Human annotator verify：subquery 只能依赖前面 subquery 已经获取的信息（**strict causal ordering**）。

**Memory 需求**：trace 单 session 就能达 122k token，long-context agent 直接被 attention saturation 击垮。

### 2.4 Sequential Formal Reasoning（40 math + 20 physics，2-16 subtasks）

**设计意图**：测 *skill / lemma distillation*。

由 PhD-level experts 从真实 paper 中抽取 derivation chain。一个 paper 的 main theorem 被拆成 ordered 的 lemma / proposition sequence。每个 statement 需要 reuse 前面 statement 的结论 + notation + algorithm。

**Memory 需求**：和前三个不同，这里 memory 不是 retrieve fact，而是 **reuse a derived result as a building block**。这是论文相对于 Mem2ActBench / AgentLongBench 这类"从 tool-call trace 里 retrieve 参数"工作的关键区分点 —— MEMORYARENA 要求 **inductive skill transfer**。

---

## 3. 核心公式解析：Memory-Agent-Environment Loop

这是 paper 最 formal 的部分。理解这套公式对 build intuition 至关重要。

### 3.1 Single-session interaction（公式 1）

$$a_{i,t} \sim \pi_{\mathcal{A}}(\cdot \mid s, o_{i,1:t-1}, a_{i,1:t-1}), \quad o_{i,t} \in \mathcal{O}$$

**变量解释**：
- $i$: 第 $i$ 个 subtask 的 index
- $t$: 在 subtask $i$ 内的 action step index，$t = 1, \ldots, T_i$
- $a_{i,t}$: agent 在 step $t$ 选择的 action（比如 `search[...]`, `click[Buy Now]`）
- $\pi_{\mathcal{A}}$: agent 的 policy（即 LLM 的 action sampling 分布）
- $s$: 当前 subtask 的 instruction（注意此处原文写成 $s$，但上下文里应该理解为 $s_i$）
- $o_{i,1:t-1}$: 过去 $t-1$ 步环境返回的 observation 序列
- $a_{i,1:t-1}$: 过去 $t-1$ 步 agent 自己选的 action 序列
- $\mathcal{O}$: observation space

**Intuition**：这是经典的 ReAct loop。Policy 只 condition on 当前 instruction + session 内 history。当 session 结束，所有 $o, a$ 序列**不再可访问**。

### 3.2 Retrieval（公式 2）

$$m_{i,t} = \text{RETRIEVE}\big(\mathcal{M}, s_i, a_{i,1:t-1}, o_{i,1:t-1}\big)$$

**变量解释**：
- $\mathcal{M}$: persistent memory system（可以是 long-context buffer、RAG index、或 memory agent），跨 session 持续存在
- $m_{i,t}$: 在 step $t$ 检索出的 memory content（一段文本、几个 fact、一个 graph substructure 等）
- 其他变量同上

**Intuition**：memory retrieval 不是无脑全文返回，而是 query-conditioned。Query 由「当前 subtask instruction + 当前 session 内已有 history」组成。这模拟了人：你想到「我要买镜头」时，记忆系统主动 surface 出「之前买的机身是 EF mount」。

### 3.3 Memory-conditioned policy（公式 3）

$$a_{i,t} \sim \pi_{\mathcal{A}}(\cdot \mid s_i, o_{i,1:t-1}, a_{i,1:t-1}, m_{i,t})$$

**变量解释**：
- 多了 $m_{i,t}$ 这个 memory context

**Intuition**：agent 的 decision 现在同时 condition on **session-local context**（$o, a$）和 **session-global memory**（$m$）。如果 $m$ retrieval 失败或 retrieval 到错误内容，后续 action 就会偏离正确轨道。

### 3.4 Memory update（公式 4）

$$\mathcal{M} \leftarrow \text{UPDATE}\big(\mathcal{M}, (o_{i,1:T}, a_{i,1:T})\big)$$

**变量解释**：
- $T$: subtask $i$ 的总 step 数
- UPDATE 是 in-place 更新：把整个 subtask 的 trajectory $(o_{i,1:T}, a_{i,1:T})$ 喂给 memory 系统，让它决定怎么 abstract / consolidate / store

**Intuition**：UPDATE 是 **lossy compression**。如果它把所有 raw trace 全留下（0D long-context），那 retrieval 几乎等于直接 paste；如果它把 trace 抽象成几个 fact（1D Mem0 / ReasoningBank），那有 information loss 风险，但 retrieval 噪声小；如果它建成 graph（2D GraphRAG / Mem0-g），那结构化关系可以保留但 query 匹配难度大。

### 3.5 Progress Score（公式 5）

$$\text{PS}_{S_i} = \frac{|s_i^{\text{pass}}|}{|S_i|}, \quad \text{PS} = \frac{1}{N} \sum_i^N \text{PS}_{S_i}$$

**变量解释**：
- $S_i$: 第 $i$ 个 task，它是一个 ordered subtask list $S_i = [s_1, s_2, \ldots, s_{|S_i|}]$
- $|s_i^{\text{pass}}|$: task $S_i$ 中正确完成的 subtask 数
- $N$: test set 中 task 总数

**Intuition**：SR 太严苛 —— 任何一个 subtask fail 就整个 task fail。PS 提供 **partial credit**，能区分「agent 在第一个 subtask 就崩」vs「agent 走到最后一步才崩」。这俩的 failure mode 完全不同。

---

## 4. Memory 系统的 0D / 1D / 2D 分类法

这是 paper 一个很有用的 conceptual tool，灵感来自 Hu et al. 2025a 的 *Memory in the Age of AI Agents*。

| 维度 | 含义 | 代表方法 | 优势 | 劣势 |
|------|------|---------|------|------|
| **0D** | Raw history，无 abstraction | Long-context buffer, BM25, Text-Embedding RAG | 信息无损、in-context learning 友好 | 噪声大、超出 context window 后失效 |
| **1D** | Flat consolidated memory，有 abstraction 但无结构 | MemGPT (Letta), Mem0, ReasoningBank, MemoRAG | 噪声小、可扩展 | 信息有损、retrieval 不一定 align agent 推理 |
| **2D** | 结构化 memory（graph / tree） | Mem0-g, GraphRAG | 保留关系结构、支持 multi-hop query | query 难匹配、update 成本高 |

**关键 insight**：paper 用实验表明，**memory 维度提升不等于性能提升**。在 Group Travel Planning 这种需要精确 attribute recall 的场景，2D GraphRAG 反而比 0D BM25 还差。

参考：
- Mem0: https://arxiv.org/abs/2504.19413
- MemGPT (Letta): https://arxiv.org/abs/2310.08560
- ReasoningBank: https://arxiv.org/abs/2509.25140
- MemoRAG: https://arxiv.org/abs/2409.05591
- GraphRAG: https://arxiv.org/abs/2404.16130

---

## 5. 实验结果深度解读

### 5.1 主表（Table 3）的关键数字

让我把数字摆出来 build intuition：

**All Task Avg SR**（统一 task agent = GPT-5.1-mini）：
- Long-context agents: GPT-5.1-mini 0.16 / GPT-4.1-mini 0.12 / Gemini-3-Flash 0.17 / Claude-Sonnet-4.5 0.19
- Memory agents: Letta 0.15 / Mem0 0.14 / Mem0-g 0.12 / ReasoningBank 0.12
- RAG agents: BM25 0.19 / Text-Embedding-3-Small 0.23 / MemoRAG 0.19 / GraphRAG 0.17

**最 striking 的事实**：连最强的 Text-Embedding RAG 也只有 **23%** SR。在 Group Travel Planning 上，**所有方法 SR 都是 0**。

**PS vs SR 的 gap**：
- Bundled Web Shopping: PS ~0.52, SR ~0.02 → gap = 0.50
- Progressive Web Search: PS ~0.09, SR ~0.23（这里 SR 反而高，因为 SR 只看最后一步成功，PS 看全部 subtask）
- Formal Reasoning: Math PS 0.35, SR 0.22；Phys PS 0.57, SR 0.42

**Interpretation**：agents 能在「局部 subtask」上做得不错，但无法把这些局部 success 整合成一个 **globally consistent solution**。这正是 multi-session interdependency 的核心难度。

### 5.2 SR@k 的衰减曲线（Figure 3）

论文定义 **SR at subtask depth k**：在第 k 个 subtask 上成功的 task 比例。

**Universal observation**：所有方法在所有 environment 上都呈衰减趋势，没有任何方法维持 plateau。这意味着：

$$\text{SR}@k \downarrow \text{ as } k \uparrow, \quad \forall \text{ method, } \forall \text{ environment}$$

**衰减速率的差异**：
- Progressive Web Search：long-context 衰减最快（trace 122k 超出 effective window），external memory 衰减最慢（因为它能 re-surface）。
- Group Travel Planning / Formal Reasoning：需要精确 attribute / lemma recall，RAG-based 衰减慢于 consolidation-based memory（Mem0 / ReasoningBank 把信息压太狠，丢 attribute）。

**Intuition**：这告诉我们一个 design principle —— 当任务需要 **verbatim recall** 时（exact size, exact rating, exact lemma statement），用 0D retrieval；当任务需要 **gist + skill reuse** 时（formal reasoning 里的 high-level idea），用 1D consolidation。

### 5.3 Latency 分析（Table 4 & 5）

每 subtask 平均 latency（秒）：

| Category | Bundled Web | Group Travel | Progressive Search | Math | Phys | Avg |
|----------|-------------|--------------|---------------------|------|------|-----|
| Long-context | 65-95 | 33-119 | 22-180 | 21-83 | 31-65 | 33-82 |
| Memory systems | 83-219 | 125-194 | 76-230 | 40-77 | 50-97 | 99-133 |
| RAG systems | 96-134 | 90-192 | 80-196 | 41-64 | 51-77 | 90-107 |

**关键发现**：
1. Long-context 始终最快 —— 因为没有 retrieval / consolidation 开销。
2. Memory systems（1D/2D）最慢 —— Mem0 在 Progressive Web Search 上 229 秒，因为每次 update 都要 LLM call 抽取 fact。
3. **Memory 复杂度和 latency 不单调相关**：2D GraphRAG (90.2s avg) 比 1D Mem0 (114.8s avg) 还快，因为 graph 构建是 batched，而 Mem0 是 incremental LLM call。

**Takeaway**：design memory system 时不能只看 effectiveness，end-to-end latency 在 multi-session 场景会累积放大。

---

## 6. POMDP 视角：最有意思的理论 framing

Section 4.6 是 paper 最 deep 的部分。让我详细展开。

### 6.1 为什么 multi-session 是 POMDP？

POMDP 的标准定义是 $(S, A, O, T, Z, R, \gamma)$：
- $S$: latent state space（agent 看不到全部）
- $A$: action space
- $O$: observation space
- $T: S \times A \to \Delta(S)$: transition
- $Z: S \to \Delta(O)$: observation function（partial）
- $R$: reward
- $\gamma$: discount

在 MEMORYARENA 里：
- $S$ = 完整的 latent task state（比如：所有已购 product 的 attribute、所有 traveler 的 preference chain、所有已证 lemma 的精确 statement）
- $O_t$ = 当前 subtask instruction + 环境 feedback（部分 observable）
- Agent 在每个 session 看到的 $O_t$ **不包含** $S$ 的全部

### 6.2 Belief drift 假说

在没有 external memory 时，agent 只能用 **truncated context** 或 **parametric knowledge** 作为隐式 belief state $\hat{b}_t$。论文提出：

$$\|\hat{b}_t - b_t^*\| \uparrow \text{ as } t \uparrow$$

其中 $b_t^*$ 是 ground-truth belief。误差累积导致 SR@k 衰减。

### 6.3 External memory = Belief-state approximator

如果 memory 系统 $\mathcal{M}$ 能返回「**all and only** the sufficient statistics for current belief」，那 agent policy 就退化成 fully-observed MDP。形式上：

$$\text{If } m_{i,t} = \text{sufficient statistics}(S_t), \text{ then } \pi_A(\cdot | s_i, o, a, m_{i,t}) \equiv \pi_A^*(\cdot | S_t)$$

**但实验表明**：SOTA memory（Mem0, GraphRAG, MemoRAG）都无法达到这个 ideal。论文识别出两个 bottleneck：

**Memory-side bottleneck**：
- 现有 memory 系统优化目标是 generic recall / compression / semantic similarity，**而不是** task-relevant sufficient statistics。
- 例子：Mem0 在 Group Travel Planning 里把 Rebecca 的 hotel 抽象成 "Rebecca booked a hotel"，丢失了 rating 数值，导致 RELATION constraint 无法 satisfy。

**Agent-side bottleneck**：
- Task agent（GPT-5.1-mini）没有经过训练去 query / interpret / integrate memory output。
- 即使 memory 返回正确信息，agent 也可能 ignore 或 misuse。
- 论文用 case study（Figure 13）展示了：BM25 成功 retrieve 出「Compact」attribute，但 agent 没把它用于 constraint checking，反而选了 "Low Profile" 违反 reject map。

### 6.4 这个 framing 对未来工作的启示

如果接受 POMDP 视角，那 future work 应该：

1. **Memory-side**：设计 memory mechanism 时 explicitly optimize for sufficient statistics preservation。可以用 information bottleneck 的方式：

$$\min_{\mathcal{M}} I(\text{trajectory}; \mathcal{M}) \text{ s.t. } I(\mathcal{M}; S_t^{\text{sufficient}}) \geq H(S_t^{\text{sufficient}}) - \epsilon$$

即：memory 要压缩 trajectory 但保留 sufficient statistics。

2. **Agent-side**：jointly train agent + memory。把 retrieval query 当作可学习 action，把 memory integration 当作 attention 层的额外 input。这和最近 RL-from-search 的工作（如 Search-R1, Toolformer-RL）思路类似。

3. **Belief tracking objective**：把 belief-state estimation 作为 auxiliary loss。类似 Dreamer 在 RL 里 learn recurrent state space model。

参考：
- Dreamer V3: https://arxiv.org/abs/2301.04104
- Search-R1: https://arxiv.org/abs/2503.09516
- Information Bottleneck: https://arxiv.org/abs/physics/0004057

---

## 7. 论文里值得讨论的 Limitations

虽然这篇 paper 写得相当扎实，但有一些可能的问题值得 Andrej 你思考：

### 7.1 Session-level memory retrieval 可能低估了 fine-grained memory

在 Appendix B.1 里作者承认：「we retrieve memory once at the beginning of each subtask (i.e., session-level memory)」。这是一个**重要的 simplification**。

在真实 agentic deployment 里，agent 应该在 **每个 action step** 都 query memory（公式 2 实际上就是这么写的）。只在 session 开始 retrieve 一次意味着：
- 如果 session 中途发现新 constraint，agent 无法重新 retrieve。
- Memory 的 online update 价值被低估。

这可能导致 paper 低估了 Mem0 / ReasoningBank 这类 incremental memory 的真实价值。

### 7.2 Task agent 只用 GPT-5.1-mini，缺乏 model diversity

主表所有结果都基于 GPT-5.1-mini 作为 task agent。虽然 Section 4.2 提到也测了 GPT-4.1-mini, Gemini-3-Flash, Claude-Sonnet-4.5 作为 long-context backbone，但**没有**这些 model 配 external memory 的组合。

如果用 Claude-Sonnet-4.5 + Mem0-g 这种组合，结果可能完全不同。Anthropic 的 model 在 tool use 和 structured reasoning 上有差异，可能对 memory integration 更友好。

### 7.3 Group Travel Planning SR 全 0 的解释可能不充分

论文把 Group Travel Planning 的全 0 SR 归因于「long-horizon reasoning + memorization 都难」。但还有一个可能：**evaluation rubric 太严**。30-slot itinerary 只要一个 slot 错就算 fail。即使 sPS 给出 partial credit（0.44-0.62），也可能掩盖了 agent 实际上的合理 planning 能力。

一个 sanity check 应该是：human baseline 在这个任务上 SR 是多少？如果 human 也只有 30%，那 0% SR 的解读就不同了。

### 7.4 Interdependent subtask 的「causal」定义还可以更紧

论文强调「strict causal ordering」—— 后面 subtask 依赖前面 subtask。但实际上「依赖」的程度可以分等级：
- **Hard dependency**：后面 subtask 完全不可解，除非前面成功。
- **Soft dependency**：后面 subtask 可解但解空间被前面 shrink。
- **Informational dependency**：前面只提供 hint，不影响 feasibility。

Bundled Web Shopping 是 hard dependency（mount type 不对就买不了 lens）。Progressive Web Search 是 soft dependency（前面 constraint 缩小候选集）。这两种的 memory 需求本质不同，paper 没有分开分析。

---

## 8. 和相关工作的 positioning

让我把 MEMORYARENA 放在 landscape 里：

```
                    Memory-focused          Action-focused
                    ┌─────────────────┐    ┌─────────────────┐
  Single-session    │ LoCoMo          │    │ WebArena        │
                    │ LongMemEval     │    │ SWE-Bench       │
                    │ MemoryBench     │    │ WebShop         │
                    │ MemoryAgentBench│    │ Mind2Web        │
                    └─────────────────┘    └─────────────────┘
                    ┌─────────────────┐    ┌─────────────────┐
  Multi-session     │ EvoMem          │    │ VeriGUI         │
  (independent)     │ (串行独立任务)   │    │ AgencyBench     │
                    └─────────────────┘    └─────────────────┘
                    ┌─────────────────┐    ┌─────────────────┐
  Multi-session     │ Mem2ActBench    │    │                 │
  (tool-trace dep.) │ MemTrack        │    │  MEMORYARENA    │
                    │ AgentLongBench  │    │  ★ 第一篇 ★    │
                    └─────────────────┘    └─────────────────┘
                    ┌─────────────────┐    ┌─────────────────┐
  Multi-session     │                 │    │                 │
  (causal dep.)     │   (空白)        │    │  MEMORYARENA    │
                    │                 │    │  ★ 第一篇 ★    │
                    └─────────────────┘    └─────────────────┘
```

**MEMORYARENA 的独特性**：在「multi-session + causal dependency + agentic action」这个 intersection 上是第一篇。

最接近的 prior work：
- **Mem2ActBench**（2026, Shen et al.）：multi-session 但 dependency 是 tool-call parameter retrieval，不是 skill / state 的 causal chain。
- **AgencyBench**（2026, Li et al.）：multi-session 有 dependency，但用 fixed add-and-retrieve tool，不 systematic 评估 memory mechanism。
- **EvoMem**（2025, Wei et al.）：multi-session 但串行独立任务，无 cross-task causal structure。

参考：
- Mem2ActBench: https://arxiv.org/abs/2601.19935
- AgencyBench: https://arxiv.org/abs/2601.11044
- EvoMem: https://arxiv.org/abs/2511.20857
- Memory in the Age of AI Agents (survey): https://arxiv.org/abs/2512.13564

---

## 9. 这篇 paper 对 LLM agent 设计的实用启示

最后总结几条 actionable insight：

### 9.1 不要盲目给 agent 加 memory

实验清楚显示：在 Bundled Web Shopping 和 Group Travel Planning 上，external memory（Mem0, Mem0-g, GraphRAG）的 SR 都是 0，而 long-context GPT-5.1-mini 还有 0.01-0.12。**External memory 不是 free lunch**。

原因（论文称为 **representation mismatch + training mismatch**）：
- **Representation mismatch**：long-context agent 在 verbatim self-consistent trace 上 in-context learn 得很好；external memory 返回的 compressed / segmented / reordered 信息打破了这个 self-consistency。
- **Training mismatch**：task agent 没有和 memory 系统 jointly optimize，不知道怎么 formulate query，也不知道怎么 integrate retrieved content。

**Design principle**：当 trace 在 context window 内时，优先用 long-context；只有 trace 超出 window（Progressive Web Search 的 122k）或需要 skill distillation（Formal Reasoning）时，才上 external memory。

### 9.2 Memory mechanism 应该 task-adaptive

不同 task domain 对 memory 的需求不同：

| Task domain | Memory 需求 | 推荐 mechanism |
|-------------|------------|----------------|
| Bundled Web Shopping | Verbatim attribute recall | 0D RAG (BM25 / embedding) |
| Group Travel Planning | Multi-hop relational reasoning | 2D GraphRAG（理论最佳，但实际 SOTA 还不行） |
| Progressive Web Search | Long-trace compression + retrieval | 1D MemoRAG / Mem0 |
| Formal Reasoning | Skill / lemma distillation | 1D ReasoningBank |

未来工作应该做 **task-conditioned memory selection** —— meta-learner 根据 task 类型动态选 memory mechanism。

### 9.3 Belief-state tracking 是值得探索的 objective

POMDP 视角给出的最大启示：把 memory 当作 belief-state approximator 来 train。

具体地，可以设计一个 self-supervised objective：
- 在 subtask $i$ 完成后，让 memory 系统 predict subtask $i+1$ 需要的 sufficient statistics。
- 用 subtask $i+1$ 的 ground-truth solution 作为 supervision。
- 这样 memory 系统被迫 preserve task-relevant 信息，drop task-irrelevant 噪声。

这个 idea 和 predictive coding / world model learning 很 close，可能能解决论文识别出的 memory-side bottleneck。

### 9.4 Joint agent-memory training 是 open problem

论文识别出 agent-side bottleneck：agent 不会 query / interpret memory。这是 RL 训练 LLM agent 时 memory 没有被 gradient signal 覆盖的体现。

未来方向：
- **Memory-augmented RL**：把 retrieval 当作 differentiable action，用 PPO / GRPO 优化 query policy。
- **Memory-conditioned SFT**：构造 (memory, query, expected action) 三元组，SFT 教 agent 怎么 use memory。
- **Constitutional memory**：用 verifier model 检查 memory output 是否 sufficient，不足时让 memory agent 重 generate。

参考：
- Toolformer: https://arxiv.org/abs/2302.04761
- ReAct: https://arxiv.org/abs/2210.03629
- Reflexion: https://arxiv.org/abs/2303.11366
- Voyager (skill library): https://arxiv.org/abs/2305.16291

---

## 10. 一句话总结

MEMORYARENA 的核心贡献：**把 agent memory evaluation 从「memorize-then-recall」范式推进到「memorize-then-act-under-dependency」范式**，并用四个 domain + 766 个精心设计的人 craft 任务 + Memory-Agent-Environment Loop 公式化 + POMDP 理论 framing，系统地暴露了 SOTA memory 系统（包括 long-context、RAG、Mem0/MemGPT/GraphRAG）在 long-horizon interdependent 场景下的根本性不足 —— **memory 维度的提升不等于 task performance 的提升，joint agent-memory optimization 是未来必经之路**。

Project page: https://memoryarena.github.io/

如果你（Andrej）后续想 build 一个真正能跨 session 持续 learn 的 agent，这篇 paper 提供的 benchmark 和 POMDP framing 是非常好的 testbed 和理论起点。尤其 formal reasoning 那个 domain，PhD-level expert-curated 的 derivation chain，对于测 agent 是否能从「学到 lemma」到「reuse lemma 推 theorem」是难得的高质量数据。
