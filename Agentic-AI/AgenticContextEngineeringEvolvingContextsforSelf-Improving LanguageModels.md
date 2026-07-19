---
source_pdf: AgenticContextEngineeringEvolvingContextsforSelf-Improving LanguageModels.pdf
paper_sha256: c0674b2311d116d66e6d39f473bf9490ced5465ebe049a7b009be77c70d7c941
processed_at: '2026-07-18T05:20:25-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# ACE (Agentic Context Engineering) 深度解析

## 1. 论文核心动机与定位

这篇 paper 来自 Stanford (Qizheng Zhang, Kunle Olukotun, James Zou) 与 SambaNova Systems 合作，解决的是 LLM 应用中 **context adaptation** 这个范式下的根本性问题。

论文主页：https://ace-agent.github.io
代码仓库：https://github.com/ace-agent/ace
AppWorld Leaderboard：https://appworld.dev/leaderboard

### 1.1 为什么 Context Adaptation 重要

现代 AI 系统 (compound AI systems, agents) 的能力提升路径正在从 **weight updates** 转向 **context engineering**。原因包括：
- Context 可解释、可审计
- 可在 runtime 快速注入新知识
- 可跨 model / module 共享
- KV cache reuse 等技术让 long-context serving 成本下降

但是 prior work 存在两个 critical limitations：

### 1.2 Brevity Bias (简洁性偏差)

prompt optimizer 倾向于把 context 压缩成短而通用的 instruction。例如 GEPA 把 brevity 当成优点，但这种 abstraction 会丢失 domain-specific heuristics、tool-use guidelines、common failure modes。Gao et al. [16] 在 test generation 中观察到，iterative optimization 反复产生近乎相同的 instructions (e.g., "Create unit tests to ensure methods behave as expected")。

### 1.3 Context Collapse (上下文坍缩)

这是论文最关键的 empirical 发现。当用 LLM 做 monolithic rewriting (整体重写) accumulated context 时，context 会突然坍缩成非常短的 summary，导致 information 严重丢失。

具体案例 (Figure 2)：
- Step 60：context = 18,282 tokens，accuracy = 66.7
- Step 61：context collapse 到 122 tokens，accuracy 暴跌至 57.1 (低于 baseline 63.7)

这种 collapse 体现了 end-to-end rewriting 的根本风险 —— **accumulated knowledge 会被 abruptly erased 而非 preserved**。

## 2. ACE 框架核心架构

### 2.1 整体设计哲学

ACE 把 context 视作 **evolving playbook** (不断进化的战术手册)，而非 distilled summary。核心设计原则：
- Context 应该 comprehensive 而非 concise
- LLM 擅长从 long detailed context 中自主 distill relevance (与 human 不同)
- 通过 structured, incremental updates 防止 collapse

### 2.2 三角色 Agentic 架构

ACE 借鉴 Dynamic Cheatsheet [41] 的 agentic 设计，将工作流分为三个 specialized components：

```
┌─────────────────────────────────────────────────────────────────┐
│                     ACE Framework Workflow                       │
└─────────────────────────────────────────────────────────────────┘

   New Query
       │
       ▼
┌──────────────┐     reasoning trajectories    ┌──────────────────┐
│              │ ─────────────────────────────▶ │                  │
│   Generator  │                                │    Reflector     │
│              │ ◀───────────────────────────── │                  │
└──────────────┘    bullet feedback (helpful/   └──────────────────┘
       │             harmful markers)                    │
       │                                              lessons
       │                                              (insights)
       ▼                                                   │
┌──────────────┐                                           ▼
│              │ ◀───────────────────────── ┌──────────────────┐
│ Inference    │    merged delta context    │     Curator      │
│ with context │                            │                  │
└──────────────┘                            └──────────────────┘
       │                                              │
       │                                              ▼
       │                                       ┌──────────────┐
       └──────── feedback loop ────────────────│  Context     │
                                              │  (Playbook)  │
                                              │  - bullets[] │
                                              └──────────────┘
```

**Generator**：在新 query 上产生 reasoning trajectories，同时标记哪些 bullets 有用 / 有害。对应 prompt 见 Figure 9。

**Reflector**：critique 这些 traces，extract concrete lessons，可选 multi-round refinement。Reflector 的分离是 ACE 相比 Dynamic Cheatsheet 的关键改进。

**Curator**：synthesize lessons 成 compact delta entries，通过 deterministic, non-LLM 逻辑 merge 到 existing context。

### 2.3 Bullet 数据结构

ACE 的 context 由 itemized bullets 组成，每个 bullet 包含：

```python
Bullet = {
    metadata: {
        id: str,              # unique identifier
        helpful_count: int,   # 被标记为 helpful 的次数
        harmful_count: int    # 被标记为 harmful 的次数
    },
    content: str  # 可复用 strategy / domain concept / failure mode
}
```

这个设计带来三个 property：
1. **Localization**：只更新相关 bullets
2. **Fine-grained retrieval**：Generator 可 focus on most pertinent knowledge
3. **Incremental adaptation**：高效 merge / prune / dedup

## 3. 关键技术创新详解

### 3.1 Incremental Delta Updates

这是 ACE 解决 context collapse 的核心机制。设当前 context 为 $\mathcal{C}_t = \{b_1, b_2, ..., b_n\}$ (bullet 集合)，每个 $b_i$ 含 metadata $m_i$ 和 content $c_i$。

**Update 形式化**：

$$\mathcal{C}_{t+1} = \text{Merge}(\mathcal{C}_t, \Delta_t)$$

其中 $\Delta_t = \{b_{new}^{(1)}, b_{new}^{(2)}, ..., b_{new}^{(k)}\}$ 是 Reflector + Curator 产生的小型 delta set。

**变量解释**：
- $\mathcal{C}_t$：第 $t$ 步的完整 context (bullet 集合)
- $\Delta_t$：第 $t$ 步的增量 delta (compact bullet 集合)
- $\text{Merge}$：deterministic, non-LLM 的合并函数

**Merge 操作细分**：

$$\text{Merge}(\mathcal{C}, \Delta) = \mathcal{C} \cup_{id} \Delta \cup_{update} \Delta$$

- $\cup_{id}$：对于 $\Delta$ 中具有新 id 的 bullet，直接 append 到 $\mathcal{C}$
- $\cup_{update}$：对于 $\Delta$ 中与 $\mathcal{C}$ 中已有 id 匹配的 bullet，in-place 更新 (e.g., 递增 helpful/harmful counter)

**对比 monolithic rewriting**：

Prior 方法 (Dynamic Cheatsheet 等) 的 update 形式为：
$$\mathcal{C}_{t+1} = \text{LLM}_{rewrite}(\mathcal{C}_t, \text{feedback}_t)$$

这里 $\text{LLM}_{rewrite}$ 用 LLM 整体重写，存在信息丢失风险 (context collapse)。ACE 用 deterministic merge 替换 LLM rewrite，把 LLM 的 responsibility 限制在产生 delta $\Delta_t$ 上，从根本上避免 collapse。

**并行性优势**：由于 updates 是 itemized 和 localized 的，multiple deltas 可以并行 merge，支持 batched adaptation at scale。

### 3.2 Grow-and-Refine Mechanism

Context 无限增长会带来 retrieval 困难，ACE 通过 periodic/lazy refinement 控制 redundancy。

**Algorithm 伪代码**：

```
Algorithm: Grow-and-Refine
Input: current context C_t, delta Δ_t, threshold θ
Output: refined context C_{t+1}

# GROW phase
for bullet b in Δ_t:
    if b.id not in C_t:
        C_t.append(b)            # 新 bullet 追加
    else:
        existing = C_t.find(b.id)
        existing.helpful_count += b.helpful_count
        existing.harmful_count += b.harmful_count
        if b.content != existing.content:
            existing.content = merge_content(existing, b)

# REFINE phase (proactive after each delta OR lazy when |C_t| > θ)
if |C_t| > θ OR proactive_mode:
    embeddings = {b.id: embed(b.content) for b in C_t}
    for pair (b_i, b_j) in C_t × C_t where i ≠ j:
        if cosine_sim(embeddings[b_i], embeddings[b_j]) > δ:
            # 去重：保留 helpful_count 更高的
            if b_i.helpful_count >= b_j.helpful_count:
                remove b_j from C_t
                merge_metadata(b_i, b_j)
            else:
                remove b_i from C_t
                merge_metadata(b_j, b_i)
return C_t
```

**变量解释**：
- $\theta$：context window 大小阈值，超过则触发 lazy refinement
- $\delta$：semantic embedding 相似度阈值 (e.g., 0.85)，用于判断 redundancy
- $\text{embed}(\cdot)$：bullet content 的 semantic embedding function
- $\text{cosine\_sim}$：cosine 相似度

**两种触发模式**：
- **Proactive**：每次 delta 后立即 refine，准确度高但 latency 增加
- **Lazy**：仅当 context 超出 window 时才 refine，latency 友好

### 3.3 Multi-Epoch Adaptation

ACE 支持多轮遍历同一组 training samples，progressively 强化 context：

$$\mathcal{C}^{(epoch=e)} = \text{Adapt}(\mathcal{C}^{(epoch=e-1)}, \mathcal{D}_{train})$$

其中 $\mathcal{D}_{train}$ 是 training data，$e \in \{1, 2, ..., E_{max}\}$。论文设 $E_{max} = 5$。

### 3.4 Reflector 的 Iterative Refinement

Reflector 可选 multi-round refinement：

$$L^{(r)} = \text{Reflector}(\text{trace}, L^{(r-1)}, \text{feedback})$$

其中 $L^{(r)}$ 是第 $r$ 轮 refinement 后的 lessons，$r \in \{1, ..., R_{max}\}$，论文设 $R_{max} = 5$。

## 4. 实验结果深度分析

### 4.1 AppWorld Agent Benchmark 结果

AppWorld [43] 是一个 autonomous agent 任务套件，要求 API understanding、code generation、environment interaction。任务分 normal 和 challenge 两个难度。

**Table 1 数据解读** (DeepSeek-V3.1 作为 base LLM)：

| Method | GT Labels | Test-Normal TGC | Test-Normal SGC | Test-Challenge TGC | Test-Challenge SGC | Average |
|---|---|---|---|---|---|---|
| ReAct (baseline) | - | 63.7 | 42.9 | 41.5 | 21.6 | 42.4 |
| ReAct + ICL | ✓ | 64.3 (+0.6) | 46.4 (+3.5) | 46.0 (+4.5) | 27.3 (+5.7) | 46.0 (+3.6) |
| ReAct + GEPA | ✓ | 64.9 (+1.2) | 44.6 (+1.7) | 46.0 (+4.5) | 30.2 (+8.6) | 46.4 (+4.0) |
| **ReAct + ACE** | ✓ | **76.2 (+12.5)** | **64.3 (+21.4)** | **57.3 (+15.8)** | **39.6 (+18.0)** | **59.4 (+17.0)** |
| ReAct + ACE | ✗ | 75.0 (+11.3) | 64.3 (+21.4) | 54.4 (+12.9) | 35.2 (+13.6) | 57.2 (+14.8) |
| ReAct + DC (CU) [online] | ✗ | 65.5 (+1.8) | 58.9 (+16.0) | 52.3 (+10.8) | 30.8 (+9.2) | 51.9 (+9.5) |
| **ReAct + ACE [online]** | ✗ | 69.6 (+5.9) | 53.6 (+10.7) | **66.0 (+24.5)** | **48.9 (+27.3)** | **59.5 (+17.1)** |

**关键 observation**：

1. **Offline + GT labels**：ACE 比 GEPA 高 13% average (59.4 vs 46.4)，比 ICL 高 13.4% (59.4 vs 46.0)。这表明 structured, evolving, detailed contexts 远胜 fixed demonstrations 或 single optimized prompts。

2. **No GT labels (self-improving)**：ACE 无标签也能 work，比 baseline 提升 14.8%。这是 self-improving agent 的关键 —— leverage execution feedback (e.g., code execution success/failure) 作为 signal。

3. **Online adaptation**：ACE 在 online setting 下 average 59.5，比 DC (51.9) 高 7.6%。更 striking 的是在 test-challenge TGC 上 ACE 达到 66.0，比 DC 的 52.3 高 13.7%。

4. **Leaderboard 表现**：ACE (59.4%) matches IBM CUGA (60.3%, GPT-4.1 based production agent)，并在 test-challenge split 上超越 IBM CUGA (TGC +8.4%, SGC +0.7%)，尽管用的是更小的 open-source model (DeepSeek-V3.1)。

### 4.2 Financial Analysis Benchmark 结果

**Table 2 数据解读**：

| Method | GT Labels | FiNER (Acc) | Formula (Acc) | Average |
|---|---|---|---|---|
| Base LLM | - | 70.7 | 67.5 | 69.1 |
| ICL | ✓ | 72.3 (+1.6) | 67.0 (-0.5) | 69.6 (+0.5) |
| MIPROv2 | ✓ | 72.4 (+1.7) | 69.5 (+2.0) | 70.9 (+1.8) |
| GEPA | ✓ | 73.5 (+2.8) | 71.5 (+4.0) | 72.5 (+3.4) |
| **ACE** | ✓ | **78.3 (+7.6)** | **85.5 (+18.0)** | **81.9 (+12.8)** |
| ACE | ✗ | 71.1 (+0.4) | 83.0 (+15.5) | 77.1 (+8.0) |
| DC (CU) [online, no GT] | ✗ | 68.3 (-2.4) | 62.5 (-5.0) | 65.4 (-3.7) |
| **ACE [online, no GT]** | ✗ | 67.3 (-3.4) | 78.5 (+11.0) | 72.9 (+3.8) |

**关键 observation**：

1. ACE offline + GT 比 GEPA 高 9.4% average (81.9 vs 72.5)。Formula 任务上 ACE 提升 18%，说明 financial reasoning 这种需要 specialized XBRL knowledge 的任务特别受益于 comprehensive playbook。

2. **Noisy feedback 的负面影响**：当没有 GT labels 且 execution signals 不可靠时 (FiNER 是 entity tagging，没有 execution feedback)，ACE online 无 GT 反而退化 (-3.4%)。这揭示一个重要 limitation：**context adaptation 严重依赖 feedback quality**。FiNER 这种 task 没有 execution outcome 可供判断，Reflector 容易产生 noisy insights 污染 context。

3. Formula 任务即使无 GT 也能提升 (+11.0)，因为数值计算可以 self-verify (重算检查)。

### 4.3 Ablation Study (Table 3)

| Method | Average |
|---|---|
| ReAct + ACE w/o Reflector or multi-epoch | 55.1 (+12.7) |
| ReAct + ACE w/o multi-epoch | 56.8 (+14.4) |
| **ReAct + ACE (full)** | **59.4 (+17.0)** |
| ReAct + ACE [online] | 56.1 (+13.7) |
| **ReAct + ACE + offline warmup [online]** | **59.5 (+17.1)** |

**Insight**：
- Reflector + multi-epoch 贡献 +4.3% (55.1 → 59.4)
- Multi-epoch 单独贡献 +1.7% (56.8 → 59.4)
- **Offline warmup** 对 online adaptation 极其关键：56.1 → 59.5，提升 +3.4%。这表明 offline 阶段建立的 strong initial context 让 online adaptation 更高效，避免了 cold start 问题。

### 4.4 Cost and Speed Analysis (Table 4)

| Setting | Method | Latency (s) | Rollouts / Token Cost |
|---|---|---|---|
| Offline (AppWorld) | ReAct + GEPA | 53898 | 1434 rollouts |
| Offline (AppWorld) | **ReAct + ACE** | **9517 (-82.3%)** | **357 (-75.1%)** |
| Online (FiNER) | DC (CU) | 65104 | $17.7 |
| Online (FiNER) | **ACE** | **5503 (-91.5%)** | **$2.9 (-83.6%)** |

**为什么 ACE 能同时提升 accuracy 和降低 cost**：

1. **Delta updates 取代 monolithic rewrite**：避免对整个 context 重新生成，只产生 small delta set。
2. **Non-LLM merging**：dedup / merge 用 deterministic 逻辑 (embedding 比较)，不调用 LLM。
3. **Fewer rollouts**：ACE 用 batch size 1，每次 sample 产生一个 delta，而 GEPA 需要大量 rollout 维护 Pareto frontier。

论文声称 ACE adaptation latency 平均降低 86.9%。

## 5. 关键 Design Intuition

### 5.1 为什么 Comprehensive Context 比 Concise Prompt 更好

论文的 core argument：**LLMs 和 humans 学习方式不同**。Humans benefit from concise generalization (因为 working memory 有限)，但 LLMs 更 effective with long, detailed contexts，可以自主 distill relevance [22, 31, 41]。

这个 insight 类似于 retrieval-augmented generation 的哲学 —— 与其 trust model 的 parametric memory，不如 saturate context with abundant information [11, 12, 22]。

### 5.2 三角色 Separation of Concerns

ACE 借鉴 human learning process：**experimenting (Generator) → reflecting (Reflector) → consolidating (Curator)**。

这种 modular 设计避免了 single LLM 同时承担 evaluation、insight extraction、curation 三个不同 cognitive task 的 bottleneck。每个 role 都可以独立 optimize (e.g., Reflector 的 prompt 可以针对 critique quality 调优)。

### 5.3 Deterministic Merge 的稳定性

ACE 用 non-LLM 的 deterministic 逻辑做 context merge，这是 anti-collapse 的关键。LLM rewrite 本质是 stochastic 的 —— 即使 prompt 要求 "preserve all information"，model 仍可能 (在 context 很长时) 自动 compress。Deterministic merge 保证 past knowledge 永远不会被 erase，只会被 update 或 dedup。

### 5.4 Feedback Signal 的依赖性

论文诚实地指出 ACE 的 limitation：**feedback quality 决定 context quality**。

- Agent 任务 (AppWorld)：execution outcomes (code runs or fails) 提供强 signal → ACE 即使无 GT 也能提升
- Financial tagging (FiNER)：无 execution outcome，Reflector 缺乏判断依据 → 可能产生 noisy insights

这表明 ACE 在 **verifiable tasks** (有 execution feedback 或 GT labels) 上最 effective，在 **subjective / non-verifiable tasks** 上需要谨慎。

## 6. 与 Prior Work 的对比

### 6.1 vs. GEPA [4]

GEPA (Genetic-Pareto) 是 sample-efficient prompt optimizer，用 reflective evolution + Pareto search 维护 prompt frontier。

- **GEPA**：optimize 单个 prompt (monolithic)，强调 brevity
- **ACE**：optimize structured bullet collection (incremental)，强调 comprehensiveness

GEPA 在 FiNER 上 +2.8%，ACE 在 FiNER 上 +7.6%。在 AppWorld 上 GEPA +4.0%，ACE +17.0%。差距随 task complexity 增大而增大。

### 6.2 vs. Dynamic Cheatsheet [41]

DC 是 ACE 的直接前作，引入 adaptive external memory。

- **DC**：用 LLM 做 monolithic rewrite of cheatsheet，存在 context collapse 风险
- **ACE**：用 delta updates + deterministic merge，从机制上避免 collapse；加入 dedicated Reflector + multi-epoch

ACE 在 AppWorld online 上比 DC 高 7.6% average。

### 6.3 vs. Reflexion [40]

Reflexion 通过 verbal reinforcement learning 改进 agent planning。

- **Reflexion**：per-episode reflection，不积累跨 episode 的 structured knowledge
- **ACE**：accumulate cross-episode insights into persistent playbook

### 6.4 vs. A-MEM [48]

A-MEM 用 Zettelkasten-inspired 动态组织 memory，每个 entry 有 structured attributes 并自动 link。

- **A-MEM**：focus on memory organization & retrieval
- **ACE**：focus on context engineering 的 broader scope (system prompts + memory + evidence)，并显式解决 brevity bias 和 context collapse

## 7. 工程实现细节

### 7.1 公平性设置

论文特意用同一个 LLM (DeepSeek-V3.1 non-thinking mode) 作为 Generator, Reflector, Curator，避免 strong Reflector/Curator 向 weak Generator 知识 transfer 造成的不公平比较。这 isolated 了 context construction 本身的 benefit。

### 7.2 Hyperparameters

- Batch size: 1 (每个 sample 产生一个 delta)
- Max Reflector refinement rounds: 5
- Max epochs (offline): 5
- Dedup threshold δ: 未明确给出，推测 0.85 左右

### 7.3 Prompts 详解

论文 Appendix D 释放了所有 prompts。关键观察：

**Generator Prompt (Figure 9)**：包含 cheatsheet injection + 结构化 ANALYSIS & STRATEGY / SOLUTION DEVELOPMENT / PROGRAMMING TASKS 三段式 instructions。

**Reflector Prompt (Figure 10)**：critique generated trace，extract lessons。

**Curator Prompt (Figure 11)**：核心 instruction 是 "Identify ONLY the NEW insights MISSING from current playbook"，并强制 "Do NOT regenerate the entire playbook - only provide additions"。输出格式为 PURE JSON，便于 deterministic parsing。

## 8. 对未来方向的 Implications

### 8.1 Long Context ≠ Higher Serving Cost

论文在 Discussion 中强调：虽然 ACE 产生 longer contexts，但现代 serving infrastructure 通过 KV cache reuse [17, 51]、compression [30, 32]、offload [25] 让 long context 的 amortized cost 持续下降。frequently reused context segments 可被 cached，避免重复 prefill。

### 8.2 Online & Continuous Learning

ACE 为 continuous learning 提供了 weight-update 的替代方案：
- Context adaptation 比 fine-tuning 便宜
- Context human-interpretable，支持 selective unlearning (隐私 / 法律需求 [1, 2])
- 适合 distribution shift [19, 24] 场景

### 8.3 局限性

- **Reflector 依赖性**：Reflector 无法 extract meaningful insights 时，context 变 noisy
- **非所有任务都需要 rich context**：HotPotQA [50] 这种 task 可能更适合 concise instruction；Game of 24 [41] 可能只需 single rule
- **Non-verifiable tasks 的挑战**：没有 execution feedback 或 GT 时，ACE 效果受限

## 9. 我的 Intuition 总结

这篇 paper 的核心 insight 可以浓缩为一句话：**Context 应该像 Git repository 一样 incremental evolve，而不是像 document 一样 monolithic rewrite**。

几个 deep intuition：

1. **Information preservation 是 self-improvement 的前提**：如果 adaptation 过程会丢失 information (context collapse)，那么 "self-improving" 就是幻觉 —— 你在 forgetting 而非 learning。ACE 的 delta update + deterministic merge 从机制上 guarantee 了 monotonic information growth (modulo dedup)。

2. **Separation of concerns 对 LLM systems 同样重要**：传统 software engineering 强调 module separation。ACE 把 evaluation (Reflector)、generation (Generator)、curation (Curator) 分开，每个 module 用不同 prompt 优化不同 cognitive function。这比 single-LLM-does-everything 更 robust。

3. **Feedback signal 是 context adaptation 的"梯度"**：Weight training 靠 backprop gradient，context adaptation 靠 feedback signal (execution outcomes / GT labels)。没有可靠 feedback 的 context adaptation 类似于无监督训练 —— 可能 work 但 unreliable。这解释了为什么 ACE 在 agent tasks (有 execution feedback) 上比 entity tagging (无 execution feedback) 上更 robust。

4. **Comprehensive > Concise for LLMs**：这与 prompt engineering 的传统 wisdom 相反。传统 wisdom 认为短 prompt 更好 (less distraction)。但论文 argue：LLM 的 attention 机制足够 robust，可以从 long context 中自主 retrieve relevant parts。短 prompt 丢失 detail 的 cost 超过了 attention 分散的 cost。这个观点与 long-context LLMs [34, 39]、many-shot ICL [3] 的发展趋势一致。

5. **Offline warmup → Online adaptation 的 two-stage 范式**：Table 3 显示 offline warmup 让 online adaptation 从 56.1 提升到 59.5。这暗示一个 general principle：**cold-start online learning 是 suboptimal 的，先用 offline 阶段建立 strong prior 再做 online refinement 更高效**。这与 meta-learning 的 idea 有异曲同工之妙。

参考链接：
- 论文主页: https://ace-agent.github.io
- GitHub: https://github.com/ace-agent/ace
- Dynamic Cheatsheet (前作): https://arxiv.org/abs/2504.07952
- AppWorld Benchmark: https://appworld.dev/leaderboard
- GEPA: https://arxiv.org/abs/2507.19457
- TextGrad: https://arxiv.org/abs/2406.07496
- Reflexion: https://arxiv.org/abs/2303.11366
- A-MEM: https://arxiv.org/abs/2502.12110
- DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437
