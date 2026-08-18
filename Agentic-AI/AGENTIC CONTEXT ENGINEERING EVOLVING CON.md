---
source_pdf: AGENTIC CONTEXT ENGINEERING EVOLVING CON.pdf
paper_sha256: 51050ced82df75c143b151262d5af8763916968ca50374bd8ff778f40552b0ad
processed_at: '2026-08-18T00:08:34-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# ACE 用人话讲

Karpathy, 咱们坐下来聊, 把这篇 paper 的故事用大白话捋一遍.

---

## 故事的起点: 一个尴尬的现象

你训了一个很强的 LLM, 扔到 AppWorld 这种 agent benchmark 上跑, 发现它其实不知道怎么用 API, 不知道 venmo 怎么转账, 不知道 spotify 怎么查 playlist. 这些知识 model weights 里没有, 你也不想 fine-tune.

怎么办? 塞 context. 给它写个 cheatsheet, 告诉它 "venmo 转账要先 login, token 放在 X, amount 要用 float". 

但这个 cheatsheet 怎么来? 两种老办法都有坑.

---

## 坑一: GEPA — 越优化越空

GEPA 的思路是 evolutionary search, 让 LLM 自己改 prompt, 留下得分高的. 听起来很合理, 但实际跑起来 LLM 改来改去, 最后留下的 prompt 长这样:

> "Carefully analyze the task and write correct code."

废话. 真正有用的细节 — 比如 "venmo 的 `payment_memo` 字段长度限制 50 字符, 超了会 silent fail" — 全被压缩掉了. 这就是 **brevity bias**: optimization 天然倾向短的、generic 的 instruction, 因为短的 prompt 在 validation set 上 average 表现还行, 但把 domain-specific 的 gold 都扔了.

Gao et al. 2025 专门 document 了这个现象, iterative prompt optimization 反复产生 near-identical 的空洞 instructions.

---

## 坑二: Dynamic Cheatsheet — 一步清零

DC 的思路更直觉: 让 LLM 边跑边记 cheatsheet, 每一步把 cheatsheet 整个 rewrite 一次, 加上新学到的 lesson.

听起来好, 但 Figure 2 的 case study 真的触目惊心:

```
Step 60:  context = 18,282 tokens,  accuracy = 66.7%
Step 61:  context = 122 tokens,    accuracy = 57.1%
                     ↑ 一步之内塌缩
baseline (不学):    accuracy = 63.7%
```

18K tokens 的 cheatsheet, LLM 重写一次, 直接缩成 122 tokens, accuracy 跌到 baseline 以下. 

为什么? LLM 拿到一大坨 context, 天然倾向 summarize 而非 preserve. 它觉得 "这些大概都差不多意思", 压成一段话. 但这一压, 18K tokens 里那些具体 API 名字、参数类型、error handling pattern 全没了.

这就是 **context collapse**: monolithic rewrite 是个 single point of failure, 一压就没.

---

## ACE 的核心 idea: 别让 LLM 干它不擅长的事

LLM 擅长什么? 生成、推理、诊断. 
LLM 不擅长什么? 精确地管理一个不断增长的 structured memory state.

ACE 的 insight 就是把这个分工切开:

```
Generator  — 跑任务, 产生 trajectory
Reflector  — 看 trajectory, 说 "这里错了, lesson 是 XXX"
Curator    — 把 lesson 整理成一个小 bullet
Merge      — deterministic code, 把 bullet 塞进 playbook
              ↑ 这一步不用 LLM!
```

关键在最后一步. 传统方法 (DC) 是让 LLM 重写整个 cheatsheet, ACE 是让 Curator 只输出一个 delta bullet, 然后 **用普通 Python 代码** 把这个 bullet append 进 playbook. 

这个 merge 是 deterministic 的:

$$\mathcal{P}_{t+1} = \text{Merge}_{\text{code}}(\mathcal{P}_t, \Delta_t)$$

其中 $\mathcal{P}_t$ 是 timestep $t$ 的 playbook (一个 bullet list), $\Delta_t$ 是 Curator 输出的 delta. Merge 逻辑就是: 新 bullet 加进去, 旧 bullet 的 helpful/harmful counter 更新一下, semantic similarity 太高的 dedup 掉. 没有 LLM 参与, 没有信息丢失的风险.

**这就是为什么 ACE 不塌缩**: 信息是 append-only 的, 只有 grow-and-refine 会 dedup, 而 dedup 是 based on embedding similarity 的 code logic, 不是 LLM 的 "我觉得这些差不多" 的 summary.

---

## Bullet 长什么样

paper Figure 3 给了个真实例子 (AppWorld 上 ACE 生成的 playbook 片段):

```
- venmo: login returns token, use it in header for all subsequent calls
- spotify: get_playlist returns tracks, but track order is NOT guaranteed 
  unless you sort by added_at
- common bug: apis.api_docs.show_api_doc(app_name, api_name) — api_name 
  is case-sensitive, 'Login' ≠ 'login'
- file system: reading /home/user/ requires absolute path, relative path 
  silently returns empty
```

每个 bullet 都是一个具体的、可执行的 insight. 对比 GEPA 给你的 "carefully analyze the task", 高下立判.

每个 bullet 还有 metadata: `id`, `helpful_count`, `harmful_count`. Generator 跑任务时会标 "这条 bullet 有用" 或 "这条 bullet 误导我了". 这个 counter 后续可以用来 prune harmful entries.

---

## 三个 design choice 各值多少分

paper 做了 ablation, 量化每个 component 的贡献:

| 配置 | AppWorld Test-Normal Average |
|------|------------------------------|
| ReAct (baseline) | 53.3 |
| ACE w/o Reflector w/o multi-epoch | 55.1 (+2.0 ish) |
| ACE w/o multi-epoch | 56.8 |
| **ACE full** | **70.3 (+17.0)** |

还有 incremental delta update 的 ablation (Table 18):

| 配置 | TGC | SGC |
|------|-----|-----|
| ACE without incremental update (monolithic rewrite) | 67.3 (+3.6) | 46.4 (+3.5) |
| **ACE with incremental update** | **76.2 (+12.5)** | **64.3 (+21.4)** |

看 SGC 那列: +3.5 跳到 +21.4. Delta update 贡献了 18 个点. 这基本就是 context collapse 的代价 — 一旦你用 monolithic rewrite, scenario-level 的细节知识就保不住.

---

## 最大的实验结果: 小模型打平大模型

AppWorld leaderboard (2025年9月快照):

| Agent | Base LLM | Average |
|-------|----------|---------|
| IBM CUGA | GPT-4.1 (巨大) | 60.3 |
| **ReAct + ACE** | **DeepSeek-V3.1 (小得多, open-source)** | **59.4** |

ACE 让一个 open-source 的 DeepSeek 模型, 在 AppWorld test-challenge split 上 **超过** 了 GPT-4.1 驱动的 production agent. 具体数字: test-challenge TGC, ACE online 66.0 vs IBM CUGA 57.6, 高了 8.4 个点.

Intuition: ACE 把 "compute at training time" 换成了 "compute at context accumulation time". 模型不需要在 weights 里 encode 所有 API 知识, 只要在 playbook 里记着就行.

---

## 代价: 又快又便宜

| Metric | GEPA (offline) | ACE (offline) | 省了 |
|--------|-----------------|----------------|------|
| Latency | 53,898 s | 9,517 s | 82.3% |
| # Rollouts | 1,434 | 357 | 75.1% |
| Input tokens | 204M | 39M | 80.8% |

GEPA 贵在哪? 它的 prompt-validation loop. 每个候选 prompt 要在 57 个 validation query 上跑一遍, 光这一步就 139M input tokens (Table 12). ACE 用 deterministic merge 替代了这个 loop, 不需要反复 validate.

| Metric | DC (online) | ACE (online) | 省了 |
|--------|-------------|--------------|------|
| Latency | 65,104 s | 5,503 s | 91.5% |
| Token cost | $17.7 | $2.9 | 83.6% |

---

## 一个反直觉的工程 insight: 长 context 反而更便宜

ACE 的 playbook 比 GEPA 的 prompt 长很多. Evaluation 时 ACE 的 raw input tokens 比 GEPA 多 117%. 但实际花的钱更少, 因为:

- playbook 在多个 query 之间是 **stable** 的
- 现代 serving infra 支持 **KV cache reuse**: 相同的 prefix 只算一次 prefill
- ACE evaluation 时 **91.8% 的 input tokens 从 cache 出**, billed cost 降低 82.6%

公式:

$$\text{billed\_cost} \propto \text{raw\_tokens} \times (1 - \text{cache\_hit\_rate})$$

cache_hit_rate = 0.918, 所以 billed cost 只有 raw 的 8.2%.

**Intuition**: 长 ≠ 贵, 只要长出来的部分是 reusable 的. GEPA 的 prompt 每个 query 都不一样, cache hit rate 低. ACE 的 playbook 跨 query 共享, cache hit rate 极高. 这是 system-level 的胜利.

---

## 没有 ground-truth label 也能学

AppWorld online, 没有 GT label, ACE 仍然达到 59.5 average, 比 DC (51.9) 高 7.6 点.

为什么? 因为 agent task 有 **execution feedback** — 代码跑没跑通、API 调没调成功、task 完没完成, 这些信号本身就是 reward. Reflector 不需要知道正确答案, 只需要看 "这个 trajectory 在 step 3 报了 KeyError, lesson 是 venmo 的 response 里没有这个 field", 就够了.

但 **finance benchmark 上没 GT 就 degrade**: FiNER w/o GT 掉到 67.3 (-3.4). 因为 FiNER 是单轮 NER, 没有 execution feedback, Reflector 没法判断自己答对没. paper §5 诚实承认了这个 limitation.

**Intuition**: ACE 的 self-improvement 依赖 reliable feedback signal. Agent task 天然有 (code execution), NER task 没有. 这是 task-dependent 的, 不是万能的.

---

## 为什么 robust

Table 17 做了个暴力实验: 每 X 步注入一次 harmful reflection:

| X (注入频率) | Accuracy |
|--------------|----------|
| 1 (每步都注入) | 66.7 (-4.0, 低于 base) |
| 5 | 76.1 (+5.4) |
| 10 | 77.0 (+6.3) |
| 100 | 78.2 (+7.5) |
| 不注入 | 78.3 (+7.6) |

每 5 步注入一次垃圾, 仍然 +5.4. 为什么? 因为 bullet 有 helpful/harmful counter, Generator 用了发现没用就标 harmful, grow-and-refine 会 prune 掉. 这是 metadata-based 的自我修复机制.

只有在极端 adversarial (每步都注入) 时才崩. 这是合理边界.

---

## 一句话总结

传统 context adaptation 要么把知识压缩成空洞的 prompt (GEPA), 要么让 LLM 重写整个 memory 然后一步清零 (DC). 

ACE 的做法: **让 LLM 只负责学和想, 让 deterministic code 负责记.** append-only, dedup by code, prune by counter. 信息只增不减 (除了 dedup), context 永远不会 collapse.

然后长 context 在 KV cache reuse 下几乎免费, adaptation cost 比 GEPA 低 80%, 小模型靠 context 积累打平大模型.

核心就这些. 剩下的都是 ablation 和 cross-domain validation.

---

**Links for reference**:
- [Paper code](https://github.com/ace-agent/ace)
- [AppWorld leaderboard](https://appworld.dev/leaderboard)
- [Dynamic Cheatsheet code](https://github.com/suzgunmirac/dynamic-cheatsheet)
- [GEPA paper](https://arxiv.org/abs/2507.19457)
- [Prompt Cache (KV reuse)](https://arxiv.org/abs/2312.03413)
- [Compound AI Systems blog](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/)

---

# ACE: Agentic Context Engineering 深度解析

Karpathy 你好, 这篇 paper 我仔细读完了, 作为一个把 context engineering 当作 "weight update 替代品" 的 framework, ACE 的核心 intuition 其实非常 elegant. 我来把整个故事拆解给你, 从 motivation 到 mechanism 到实验数据.

---

## 1. 背景: Context Adaptation 为什么重要

现代 LLM applications (agents, compound AI systems) 越来越依赖 **context adaptation** — 通过修改 input (instructions, strategies, evidence) 而非 model weights 来改进 performance. 这条路线有几个结构性优势:

- **Interpretability**: context 是人类可读的, 可以 audit ([Wei et al., 2022](https://arxiv.org/abs/2201.11903))
- **Runtime knowledge injection**: 不需要 retraining ([Lewis et al., 2020 RAG](https://arxiv.org/abs/2005.11401))
- **Cross-model portability**: 同一个 context 可以在不同 model 间共享
- **Long-context LLMs 的成熟**: YaRN ([Peng et al., 2024](https://arxiv.org/abs/2309.00071)), KV cache reuse ([Gim et al., 2024 Prompt Cache](https://arxiv.org/abs/2312.03413); [Yao et al., 2025 CacheBlend](https://arxiv.org/abs/2405.16444)) 让长 context 在 serving 层面 affordable

**Key intuition**: context 本质上是模型在 inference time 的 "working memory + playbook", 而非简单的 prompt string.

---

## 2. 两个核心病灶 (这是 paper 最 insightful 的诊断)

### 2.1 Brevity Bias

代表方法 GEPA ([Agrawal et al., 2025](https://arxiv.org/abs/2507.19457)) 把 brevity 当成 strength, 但 Gao et al. ([2025](https://arxiv.org/abs/2501.01329)) 发现 prompt optimization 反复产生 near-identical 的 generic instructions, 例如:

> "Create unit tests to ensure methods behave as expected"

这种 abstraction 会 **drop domain-specific heuristics, tool-use guidelines, common failure modes**. 在 agent 和 knowledge-intensive 场景里, 这些细节恰恰是 performance 的决定因素.

### 2.2 Context Collapse (paper 最关键的发现之一)

Figure 2 的 case study 极具说服力. 在 AppWorld 上用 Dynamic Cheatsheet ([Suzgun et al., 2025](https://arxiv.org/abs/2504.07952)) 做测试, 作者观察到:

| Step | Context Tokens | Accuracy |
|------|----------------|----------|
| 60   | 18,282         | 66.7     |
| 61   | 122 (塌缩!)    | 57.1     |
| baseline (无 adaptation) | — | 63.7 |

**在一步之内, context 从 18K tokens 塌缩到 122 tokens, accuracy 直接跌破 baseline.** 这是因为 LLM 被 task 去 monolithic rewrite context 时, 倾向于 produce 短的 summary, 累积的细节被 abrupt erase.

这是 LLM-as-rewriter 的 fundamental risk: 一次性重写 = 单点 failure, knowledge 不可逆丢失.

---

## 3. ACE 的架构 (核心 contribution)

### 3.1 三角色分工 (Figure 4 解析)

ACE 借鉴 Dynamic Cheatsheet 的 agentic 设计, 但做了 critical 的 **division of labor**:

```
┌─────────────┐     reasoning traces      ┌─────────────┐     lessons       ┌─────────────┐
│  Generator  │ ─────────────────────────→ │  Reflector  │ ────────────────→ │  Curator    │
│             │                            │             │                   │             │
│ 产生 trajectory│                          │ critique &  │                   │ synthesize  │
│ 执行 task    │                            │ extract     │                   │ delta       │
│             │                            │ insights    │                   │ entries     │
└─────────────┘                            └─────────────┘                   └──────┬──────┘
       ↑                                                                           │
       │              deterministic merge (non-LLM logic)                         │
       └───────────────────────────────────────────────────────────────────────────┘
                              Playbook (itemized bullets)
```

**关键设计 choice**: Curator 之后的 merge 是 **lightweight, non-LLM logic**, 而非又一次 LLM rewrite. 这直接规避 context collapse.

### 3.2 Bullet 数据结构

ACE 把 context 表示为 itemized bullets, 每个 bullet:

```
bullet = {
    metadata: {
        id: unique_identifier,
        helpful_count: N_helpful,   # Generator 标记 useful 的次数
        harmful_count: N_harmful    # Generator 标记 misleading 的次数
    },
    content: <strategy | domain_concept | failure_mode>
}
```

可以形式化为一个增量更新过程. 设 timestep $t$ 时 playbook 为 $\mathcal{P}_t = \{b_1, b_2, ..., b_n\}$, 其中 $b_i = (m_i, c_i)$, $m_i$ 是 metadata, $c_i$ 是 content. 一次 delta update $\Delta_t$ 产生后:

$$\mathcal{P}_{t+1} = \text{Merge}(\mathcal{P}_t, \Delta_t)$$

其中 $\Delta_t = \{b^{new}_1, ..., b^{new}_k\} \cup \{(b_i, \delta m_i)\}$, $b^{new}_j$ 是新 bullet, $\delta m_i$ 是对已有 bullet 的 counter 增量. Merge 操作是 deterministic 的 — 这就是 "incremental" 的数学含义.

**对比 monolithic rewrite**: 传统方法做的是 $\mathcal{P}_{t+1} = \text{LLM}(\mathcal{P}_t, \text{feedback})$, 一个 LLM call 全部覆盖, risk 极高.

### 3.3 Generator / Reflector / Curator 的 prompt 解析 (Appendix F)

paper 在 Appendix 里给出了完整的 prompt (Figure 9-14), 我挑几个关键设计点:

**Generator prompt** (AppWorld, Figure 9):
- 明确告诉 model "你有一个 cheatsheet, 里面是 similar problems 的 strategies, patterns, examples"
- 强制 structure: ANALYSIS & STRATEGY → SOLUTION DEVELOPMENT → PROGRAMMING TASKS
- 编码要求 self-contained, 用 `EXECUTE CODE!` 标记

**Reflector prompt** (Figure 10):
> "You are an expert AppWorld coding agent and educator. Your job is to diagnose the current trajectory: identify what went wrong (or could be better), grounded in execution feedback, API usage, unit test report, and ground truth when applicable."

注意 Reflector 是 **post-hoc diagnostician**, 它看的是已经发生的 trace, 提取 lesson. 这和 Reflexion ([Shinn et al., 2023](https://arxiv.org/abs/2303.11366)) 的 verbal reinforcement 思想类似, 但 ACE 把 reflection 和 curation 解耦了.

**Curator prompt** (Figure 11):
把 Reflector 的 output 整理成 compact delta bullets, 控制粒度.

---

## 4. 三大 Innovation 深度讲解

### 4.1 Reflector 解耦 (为什么这是关键)

Dynamic Cheatsheet 是 single-LLM-rewrite, 它既要做 evaluation 又要做 curation. ACE 把这两件事 split:

- **Reflector**: 专注 "what went wrong / what worked", 输出是 lessons (natural language)
- **Curator**: 专注 "how to structure this into the playbook", 输出是 delta bullets

Ablation (Table 3) 直接量化了这个 design:

| 配置 | AppWorld Test-Normal Average |
|------|------------------------------|
| ACE w/o Reflector or multi-epoch | 55.1 (+12.7) |
| ACE w/o multi-epoch | 56.8 (+14.4) |
| **ACE (full)** | **59.4 (+17.0)** |

去掉 Reflector 掉 4.3 个点, 这是 division of labor 的价值.

### 4.2 Incremental Delta Updates

这是 paper 里最 engineering-friendly 的部分. 对比两种 update 范式:

**Monolithic rewrite** (GEPA, DC-CU):
```
P_{t+1} = LLM_rewrite(P_t, feedback_t)
```
- Cost: $O(|P_t|)$ per step (输入整个 context)
- Risk: 信息可能被 erase

**ACE delta update**:
```
delta_t = Curator(Reflector(trace_t))
P_{t+1} = Merge_deterministic(P_t, delta_t)   # non-LLM!
```
- Cost: $O(|delta_t|)$, 且 $|delta_t| \ll |P_t|$
- Risk: 只增不减 (除了 grow-and-refine 的 dedup)

Table 18 的 ablation 极其 dramatic:

| 配置 | Test-Normal TGC | Test-Normal SGC |
|------|-----------------|------------------|
| ReAct (baseline) | 63.7 | 42.9 |
| ACE (no incremental update) | 67.3 (+3.6) | 46.4 (+3.5) |
| **ACE (with incremental update)** | **76.2 (+12.5)** | **64.3 (+21.4)** |

SGC 从 +3.5 跳到 +21.4, 这说明 **delta update 保留了 scenario-level 的细节知识**, 而 monolithic rewrite 把这些细节压缩掉了.

### 4.3 Grow-and-Refine

bullet 会无限增长, 需要 redundancy control. Grow-and-refine 的逻辑:

```
when trigger_condition (proactive per-delta OR lazy on context-overflow):
    for each pair (b_i, b_j) in P:
        sim = cosine(embed(b_i.content), embed(b_j.content))
        if sim > threshold:
            merge b_i, b_j   # 保留 helpful_count 更高的
```

这个机制让 ACE 可以 scale 到 long-horizon, 因为 dedup 是 $O(n^2)$ embedding 比较, 但 $n$ 经过 dedup 后稳定.

**Sensitivity analysis** (Table 20, 21):
- Dedup threshold: 50% → 77.0, 70% → 73.9, 90% → 78.6 (在 FiNER 上, 波动 < 5 点)
- Max context length: 10K/50K/100K → 78.6/78.4/78.3 (几乎无波动)

这说明 ACE **不依赖精细 hyperparameter tuning**, 这是 robustness 的标志.

---

## 5. 实验结果深度解析

### 5.1 AppWorld (Table 1) — Agent Benchmark

AppWorld ([Trivedi et al., 2024](https://arxiv.org/abs/2403.08292)) 是 multi-turn agent benchmark, 涉及 API understanding, code generation, environment interaction. Base LLM 是 DeepSeek-V3.1-671B.

| Method | GT Labels | Test-Normal TGC | Test-Normal SGC | Test-Challenge TGC | Test-Challenge SGC | Average |
|--------|-----------|-----------------|-----------------|---------------------|---------------------|---------|
| ReAct | — | 63.7 | 42.9 | 41.5 | 21.6 | 42.4 |
| ReAct + ICL | √ | 64.3 | 46.4 | 46.0 | 27.3 | 46.0 |
| ReAct + GEPA | √ | 64.9 | 44.6 | 46.0 | 30.2 | 46.4 |
| **ReAct + ACE (offline, w/ GT)** | √ | **76.2** | **64.3** | **57.3** | **39.6** | **59.4** |
| ReAct + ACE (offline, w/o GT) | × | 75.0 | 64.3 | 54.4 | 35.2 | 57.2 |
| ReAct + DC (online) | × | 65.5 | 58.9 | 52.3 | 30.8 | 51.9 |
| **ReAct + ACE (online, w/o GT)** | × | **69.6** | 53.6 | **66.0** | **48.9** | **59.5** |

**几个关键观察**:

1. **ACE vs GEPA**: offline +12.5 TGC / +21.4 SGC on test-normal. GEPA 的 prompt optimization 只能产生 generic instruction, 而 ACE 累积了具体的 tool-use patterns.

2. **无 GT labels 也能 work**: online setting 下 ACE w/o GT 达到 59.5, 比 DC w/o GT (51.9) 高 7.6 点. 这靠的是 **execution feedback** (code 是否成功执行) 而非 ground-truth answer. 这是 self-improving agent 的关键 property.

3. **Test-challenge 的 online ACE 表现惊人**: TGC 66.0 (+24.5), SGC 48.9 (+27.3). online adaptation 在 harder split 上收益更大, 因为 challenge task 暴露更多 failure mode, ACE 能学到更多.

4. **AppWorld Leaderboard** (Figure 5, 截至 2025-09):
   - IBM CUGA (GPT-4.1): 60.3 average
   - ReAct + ACE (DeepSeek-V3.1, open-source, 小得多): 59.4 average
   - 在 test-challenge 上 ACE 反超 IBM CUGA

**Intuition**: ACE 让一个 weaker model 通过 context 积累, 达到 stronger model 的水平. 这是 "compute at inference time" 的胜利, 而非 "compute at training time".

### 5.2 Finance Benchmark (Table 2)

FiNER ([Loukas et al., 2022](https://arxiv.org/abs/2203.07809)): XBRL financial document 的 139 类 entity typing. Formula ([Wang et al., 2025a](https://arxiv.org/abs/2505.19819)): financial concept 应用 + 数值计算.

| Method | GT Labels | FiNER | Formula | Average |
|--------|-----------|-------|---------|---------|
| Base LLM | — | 70.7 | 67.5 | 69.1 |
| ICL | √ | 72.3 | 67.0 | 69.6 |
| MIPROv2 | √ | 72.4 | 69.5 | 70.9 |
| GEPA | √ | 73.5 | 71.5 | 72.5 |
| **ACE (offline, w/ GT)** | √ | **78.3** | **85.5** | **81.9** |
| ACE (offline, w/o GT) | × | 71.1 | 83.0 | 77.1 |
| DC (online, w/ GT) | √ | 74.2 | 69.5 | 71.8 |
| DC (online, w/o GT) | × | 68.3 | 62.5 | 65.4 |
| **ACE (online, w/ GT)** | √ | **76.7** | 76.5 | **76.6** |
| ACE (online, w/o GT) | × | 67.3 | 78.5 | 72.9 |

**关键 insight**: Formula 上 ACE offline +GT 达到 85.5 (+18.0!), 这是 paper 里最大的 single gain. Formula 需要精确的 financial concept 应用, ACE 累积了 XBRL 规则、计算公式、edge case, 这些是 GEPA 的 single prompt 无法承载的.

**但注意 warning**: ACE w/o GT 在 FiNER 上 67.3 (-3.4), DC w/o GT 也 68.3 (-2.4). **没有可靠 feedback signal 时, context adaptation 会 degrade**. paper 在 §5 明确指出这是 limitation — Reflector 需要 reliable signal (execution outcome 或 GT label) 来判断 success/failure.

### 5.3 其他 domain (Table 10, 11)

- **DDXPlus (medical reasoning)**: Base 75.2 → GEPA 76.4 (+1.2) → **ACE 90.2 (+15.0)**. 这个 gain 巨大, 说明 ACE 的 playbook 范式对 multi-step diagnostic reasoning 极其有效.
- **BIRD-SQL**: Base 47.8 → GEPA 52.2 (+4.4) → ACE 52.9 (+5.1). ACE 在 Simple split 上更强 (53.5 vs GEPA 51.6), GEPA 在 Moderate/Challenging 上略胜. 说明 SQL 的复杂 query 可能需要 GEPA 的 prompt-level optimization.

### 5.4 Cross-LLM Generalization (Table 5-9)

ACE 在 4 个 LLM family 上都 work: DeepSeek-V3.1, GPT-OSS-120B, GPT-5.1, Llama-3.3-70B. 

**Intuition**: ACE 是 model-agnostic 的, 因为它操作的是 execution traces 和 contextual deltas, 不依赖 model 的 architectural feature. 但 gain magnitude 受 reflector 质量影响 — Llama-3.3-70B gain 较小, 因为 weaker model 产生 noisier reflection.

---

## 6. Cost & Latency Analysis (工程价值)

这是 paper 最 practical 的部分, 对 production deployment 很关键.

### 6.1 Adaptation Stage (Table 4a, 12)

Offline AppWorld, ACE vs GEPA:

| Metric | GEPA | ACE | Reduction |
|--------|------|-----|-----------|
| Latency (s) | 53,898 | 9,517 | **-82.3%** |
| # Rollouts | 1,434 | 357 | **-75.1%** |
| Total input tokens | 204.1M | 39.3M | -80.8% |
| Total output tokens | 1.87M | 0.31M | -83.6% |

GEPA 贵在它的 **prompt-validation loop**: 每个候选 prompt 要在 57 个 validation query 上 re-evaluate, 产生 139M input tokens (Table 12). ACE 用 deterministic merge 替代了这个 loop.

### 6.2 Online Adaptation (Table 4b)

FiNER online, ACE vs DC:

| Metric | DC | ACE | Reduction |
|--------|-----|-----|-----------|
| Latency (s) | 65,104 | 5,503 | **-91.5%** |
| Token Cost ($) | 17.7 | 2.9 | **-83.6%** |

### 6.3 Evaluation Stage — KV Cache Reuse 的关键 insight (Table 14, 15)

这是 paper 里最 subtle 的工程点. ACE 的 playbook 比 GEPA 的 prompt 长很多, evaluation 时 raw input tokens 多 117.4%. **但实际 billed cost 反而更低**, 因为:

$$\text{billed\_cost} = \text{raw\_tokens} - \text{cached\_tokens}$$

ACE 的 playbook 在多 query 间共享, **91.8% input tokens 从 cache serve**, billed input cost 降低 82.6%.

**Intuition**: 长 context ≠ 高 serving cost, 只要 context 是 stable 的 (ACE 的 playbook 就是). 现代 serving infra ([Prompt Cache](https://arxiv.org/abs/2312.03413), [CacheBlend](https://arxiv.org/abs/2405.16444), [CacheGen](https://arxiv.org/abs/2310.07240), [Continuum](https://arxiv.org/abs/2511.02230)) 让 reused context 几乎免费. 这颠覆了 "prompt 越短越好" 的传统 wisdom.

---

## 7. Robustness Analysis (Section A.4)

### 7.1 Weaker Reflector (Table 16)

FiNER 上, Generator/Curator 固定为 DeepSeek-V3.1, 只换 Reflector:

| Reflector LLM | FiNER Accuracy |
|---------------|----------------|
| GPT-OSS-120B (weaker) | 76.6 (+5.9) |
| DeepSeek-V3.1 (same) | 78.3 (+7.6) |
| GPT-5.1 (stronger) | 78.5 (+7.8) |

Weaker reflector 仍有 +5.9 gain, stronger reflector 只多 +1.9. **ACE 对 reflector 质量不敏感**, 因为 grow-and-refine 的 dedup 和 harmful-count 过滤能 absorb noisy reflection.

### 7.2 Adversarial Reflector (Table 17)

每 X 步注入一次 harmful reflection:

| X (频率) | Accuracy |
|----------|----------|
| 1 (每步) | 66.7 (-4.0) ⚠️ 低于 base |
| 5 | 76.1 (+5.4) |
| 10 | 77.0 (+6.3) |
| 25 | 77.8 (+7.1) |
| 100 | 78.2 (+7.5) |
| No harmful | 78.3 (+7.6) |

**Intuition**: ACE 容忍 moderate noise, 只在极端 adversarial (每步都 corrupt) 时才 degrade. 这是 metadata-based filtering (helpful/harmful counter) 的功劳.

---

## 8. ACE vs GEPA vs DC 的哲学差异

### 8.1 ACE vs GEPA (Appendix C.1)

| 维度 | GEPA | ACE |
|------|------|-----|
| **Objective** | Prompt evolution, maximize evaluator under rollout budget | Accumulate granular reusable insights over long horizons |
| **Representation** | Single monolithic prompt | Itemized Playbook (bullets) |
| **Update** | Evolutionary loop, full prompt rewrite | Incremental delta, deterministic merge |
| **Best for** | Single-turn, prompt-optimizable tasks | Multi-turn agents, knowledge-intensive domains |
| **Bookkeeping** | 无 (每个 candidate 独立) | 有 (helpful/harmful counter, dedup) |

### 8.2 ACE vs Dynamic Cheatsheet (Appendix C.2)

| 维度 | DC | ACE |
|------|-----|-----|
| **Update** | Full cheatsheet rewrite (DC-CU) or retrieved-summary (DC-RS) | Incremental delta |
| **Risk** | Context collapse (§2.2) | 无 (deterministic merge) |
| **Roles** | Single LLM does all | Generator + Reflector + Curator separation |
| **Evaluation domain** | Single-turn (AIME, Game-of-24, GPQA) | Multi-turn agents (AppWorld), domain-specific (FiNER, Formula) |

**核心差异**: DC 依赖 LLM 做 curation, ACE 把 curation 的 merge 部分 hardcode 成 non-LLM logic. 这是一个 **"trust LLM for generation, trust code for state management"** 的设计哲学.

---

## 9. Limitations & Open Questions

paper §5 诚实承认:

1. **依赖 Reflector 质量**: 如果 Reflector 提取不出 meaningful insight, context 会变 noisy. 在 no model can extract useful insights 的 domain, ACE 自然 fail.
2. **并非所有 task 都需要 rich context**: HotPotQA ([Yang et al., 2018](https://arxiv.org/abs/1809.09600)) 这类 task 更受益于 concise high-level instruction. Game of 24 只需 single reusable rule.
3. **无可靠 feedback 时 degrade**: FiNER w/o GT 的 -3.4 说明 context adaptation 本质上是 feedback-driven learning, 没有 signal 就没有 learning.

**我个人觉得的 open questions** (paper 没展开但值得思考):

- **Playbook 的 forgetting**: 长期运行后, 早期 bullet 可能 outdated. grow-and-refine 的 dedup 基于 semantic similarity, 但不基于 temporal relevance. 是否需要 time-decay 或 relevance scoring?
- **Cross-task transfer**: 一个 domain 训练的 playbook 能否 transfer 到另一个 domain? paper 没测.
- **Playbook 的 interpretability vs performance tradeoff**: bullet 太多时, Generator 如何 attend 到 relevant bullet? paper 用 fine-grained retrieval 提了一句, 但没深入.
- **和 RL 的关系**: ACE 本质上是 inference-time RL, reward 是 execution feedback. 是否可以形式化为 policy gradient, 其中 policy 是 context, gradient 是 Reflector 的 textual feedback? 这和 TextGrad ([Yuksekgonul et al., 2025](https://www.nature.com/articles/s41586-024-08394-z)) 的 framework 有 connection.

---

## 10. 对你的 Intuition Building 的总结

Karpathy, 如果我要用一段话总结 ACE 的核心 insight:

**Context 不是 prompt string, context 是 evolving state.** 传统 prompt optimization 把 context 当作 optimization variable (GEPA 的 evolutionary search), 传统 memory 把 context 当作 LLM-rewritable text (DC). ACE 把 context 当作 **structured, append-only, deduplicated log of lessons learned**, 用 LLM 做 generation 和 reflection, 用 deterministic code 做 state merge. 这分离了 "intelligence" (LLM) 和 "memory management" (code), 避开了 LLM-as-rewriter 的 context collapse trap.

**数学上**, 这等价于把 context update 从一个高风险的 function approximation (LLM rewrite) 换成了一个低风险的 incremental update (deterministic merge). variance 大幅降低, signal accumulation 成为可能.

**工程上**, 这让长 context 在 KV cache reuse 下几乎免费, 且 adaptation cost 比 GEPA 低 80%+.

**哲学上**, 这是 "saturate context with abundant information, let LLM distill relevance at inference time" 的胜利, 而非 "compress context into concise summary". 这和人类 learning 的 intuition 相反 — 人类 benefit from concise generalization, LLM benefit from detailed enumeration. ([Jiang et al., 2025](https://arxiv.org/abs/2505.08120); [Liu et al., 2025b Self-Elicit](https://arxiv.org/abs/2504.21146); [Suzgun et al., 2025](https://arxiv.org/abs/2504.07952))

这个 paper 对你一直强调的 "software 2.0 → software 3.0" (prompt as program) 的叙事是一个 strong empirical validation: prompt 不是 static program, 是 self-evolving program, 而 evolution 的 mechanism 应该是 structured + deterministic, 而非 end-to-end LLM rewrite.

---

**Reference links**:
- Paper code: [github.com/ace-agent/ace](https://github.com/ace-agent/ace)
- AppWorld leaderboard: [appworld.dev/leaderboard](https://appworld.dev/leaderboard)
- Dynamic Cheatsheet: [github.com/suzgunmirac/dynamic-cheatsheet](https://github.com/suzgunmirac/dynamic-cheatsheet)
- GEPA: [arxiv.org/abs/2507.19457](https://arxiv.org/abs/2507.19457)
- DSPy (GEPA/MIPROv2 implementation): [dspy.ai](https://dspy.ai)
- TextGrad: [nature.com/articles/s41586-024-08394-z](https://www.nature.com/articles/s41586-024-08394-z)
- Reflexion: [arxiv.org/abs/2303.11366](https://arxiv.org/abs/2303.11366)
- AppWorld: [arxiv.org/abs/2403.08292](https://arxiv.org/abs/2403.08292)
- Prompt Cache: [arxiv.org/abs/2312.03413](https://arxiv.org/abs/2312.03413)
- CacheBlend: [arxiv.org/abs/2405.16444](https://arxiv.org/abs/2405.16444)
- YaRN: [arxiv.org/abs/2309.00071](https://arxiv.org/abs/2309.00071)
- Compound AI Systems blog: [bair.berkeley.edu/blog/2024/02/18/compound-ai-systems](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/)
