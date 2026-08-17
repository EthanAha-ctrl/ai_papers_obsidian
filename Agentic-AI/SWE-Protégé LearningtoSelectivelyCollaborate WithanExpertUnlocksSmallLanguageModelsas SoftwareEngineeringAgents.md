---
source_pdf: SWE-Protégé LearningtoSelectivelyCollaborate WithanExpertUnlocksSmallLanguageModelsas
  SoftwareEngineeringAgents.pdf
paper_sha256: 4cc97155880c895b9c7434909eb7e61cd71a2918ab48acae1ffcc978461a2519
processed_at: '2026-08-12T11:34:26-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# SWE-Protégé 人话版

## 一句话讲清楚

小模型 (7B) 自己一个人修 bug 会陷入死循环——反复 grep 同一个文件，刷屏同样的命令，最后超时失败。SWE-Protégé 教这个小模型学会一件事：**卡住的时候，举手问大模型**。问完之后还得真的照着做，做完汇报。就这一招，7B 模型从 17% 干到 42.4%。

## 问题在哪

先讲清楚 SLM 在 SWE-bench 上的失败模式。

你给一个 7B 模型一个 GitHub issue，让它在一个真实 repo 里找 bug、改代码、跑测试。它用 SWE-agent 这个 scaffold，每一步可以调 bash、edit file、grep 等 tool。问题是 7B 模型跑着跑着就**卡住了**——它会反复执行类似的命令：

```
grep -r "Rational" /testbed/sympy/core/numbers.py
str_replace_editor view /testbed/sympy/core/numbers.py -view_range 1388 1450
str_replace_editor view /testbed/sympy/core/numbers.py -view_range 1450 1500
str_replace_editor view /testbed/sympy/core/numbers.py -view_range 1500 1550
...
```

这种 degenerative loop 在 paper 里被量化了 (Fig. 6)：

| Model | 含 >10 步重复 loop 的 trajectory 比例 |
|-------|------|
| SWE-agent-LM-7B | 33.6% |
| SWE-agent-LM-32B | 24.4% |
| SWE-Protégé-7B post-SFT | 31.0% |
| **SWE-Protégé-7B post-RL** | **0.8%** |

三分之一的 trajectory 都在原地打转。这就是为什么 SLM 在 SWE-bench 上只有 17%——不是它不会写代码，是它**不会 escape 死胡同**。

Reference: [SWE-smith](https://arxiv.org/abs/2504.21798), [SWE-Gym](https://arxiv.org/abs/2412.21139)

## 核心 insight

大模型 (Claude Sonnet) 不容易卡住，因为它有更强的 intuition——看几眼代码就知道 root cause 大概在哪。小模型没这个 intuition，但**小模型不需要自己有这个 intuition，它只需要学会在卡住的时候去借**。

这就像 junior dev + senior dev 的 pair programming。Junior 写大部分代码，做大部分 routine work。Senior 不写代码，只在 junior 卡住的时候点一下："看这个文件，问题在 `_EvaluatorPrinter` class"。Junior 听完就自己去改了。

SWE-Protégé 就是把这个 pattern 训练进 7B 模型。

## 怎么训练的

两阶段 post-training pipeline。

### Phase 1: SFT — 教 mechanics

先用 Claude Sonnet 3.7 在 SWE-agent 环境里跑 ~4.8K 个 task，生成 trajectory。关键 trick：**让 Sonnet 3.7 同时扮演 agent 和 expert**——agent 在卡住时调 `ask_expert`，expert 也是 Sonnet 3.7，但 expert **能看到 ground-truth patch**（被指示不能 verbatim 抄答案）。这产生了 38% more usable trajectories。

然后用这些 trajectory 做 standard next-token SFT：

$$\mathcal{L}_{\mathrm{SFT}}(\theta) = -\mathbb{E}_{(s_i, y_i)}[\log p_\theta(y_i \mid s_i)]$$

- $s_i$: input state（agent history + tool outputs + system prompt）
- $y_i$: target token sequence（下一个 action/thought）
- $\theta$: Qwen2.5-Coder-7B-Instruct 的参数
- $p_\theta(y_i \mid s_i)$: 模型预测 target sequence 的概率

**没有 auxiliary loss**——expert call 稀疏性是 emerge 的，因为 training data 里 expert call 本来就稀疏。

SFT 后效果：30.8% (Sonnet 3.7 expert)。但问题是——**模型会调 expert 了，但经常不调，或者调了之后不 follow**。Loop 还在 31%。

### Phase 2: RL — 教 behavior

这是真正产生 magic 的地方。用 GRPO 做 on-policy RL，核心是 reward design。

每次 rollout 产生一个 trajectory $\tau$，reward 是 composite 的：

$$R_{\mathrm{total}}(\tau, x) = R_{\mathrm{loop}}(\tau) + w_{\mathrm{follow}} R_{\mathrm{follow}}(\tau) + g_{\mathrm{loop}}(\tau) g_{\mathrm{follow}}(\tau) R_{\mathrm{other}}(\tau, x)$$

- $\tau$: full trajectory
- $x$: task metadata (gold patch 等)
- $R_{\mathrm{loop}} \leq 0$: 惩罚 degenerative looping
- $R_{\mathrm{follow}}$: 奖励 follow-through expert guidance
- $R_{\mathrm{other}} = R_{\mathrm{correct}} + w_{\mathrm{sim}} R_{\mathrm{sim}} + w_{\mathrm{expert}} R_{\mathrm{expert}}$: correctness + similarity fallback + expert deferral quality
- $g_{\mathrm{loop}}, g_{\mathrm{follow}} \in \{0, 0.5, 1\}$: gating functions，当 looping 严重或 follow-through 失败时 downweight $R_{\mathrm{other}}$

**关键设计**：gates 的作用是防止 correctness mask pathological behavior。如果 agent 恰好猜对 patch 但 loop 严重，correctness=1 不应该被全权奖励。Gate 在 loop 严重时把 $R_{\mathrm{other}}$ 降到 0 或 0.5×，让 loop penalty 主导。

#### Loop penalty 的具体形式

$$R_{\mathrm{loop}}(\tau) = \max\Big(c_{\mathrm{loop}}, -\lambda_{\mathrm{loop}}\Big(\max(0, s_1 - k_1 + 1) + \sum_{j \geq 2} \max(0, s_j - k_2 + 1)\Big)\Big)$$

- $s_1, s_2, \ldots$: maximal consecutive identical-command streaks 的长度（按时间顺序）
- $k_1 = 15$: 第一次 trigger 的 threshold
- $k_2 = 8$: 后续 trigger 的 threshold（更严格）
- $\lambda_{\mathrm{loop}} = 0.5$: penalty scaling
- $c_{\mathrm{loop}} = -10$: cap

设计很巧妙：第一次容忍长 streak（15 步），后续就更严格（8 步）。这使 stalling sparse but decisive。

#### Warrant score 和 Follow score

每次 expert call $i$，用 expert 本身作为 judge 计算两个 score：

$$u_i := J_{\mathrm{warrant}}(q_i, \tilde{s}_i) \in [0, 1], \quad f_i := J_{\mathrm{follow}}(g_i, \Delta_i(\tau)) \in [0, 1]$$

- $u_i$: 这次 escalation 是否 warranted（防 lazy invocation）
- $f_i$: agent 是否 follow 了 guidance 并 report back
- $q_i$: agent 的 query
- $g_i$: expert 的 guidance
- $\tilde{s}_i$: 给 expert 的 compact context（最近 5 messages）
- $\Delta_i(\tau)$: 接收 $g_i$ 后到下次 expert call 的 agent 行为 segment

这两个 score 通过 hidden judge call 计算，**不返回给 agent**，只 log 给 RL reward 用。这参考了 [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10220) 的思路。

#### GRPO objective

$$J(\theta) = \frac{1}{G} \sum_{i=1}^G \min\Big(r_i(\theta) A_i, \mathrm{clip}(r_i(\theta), 1-\epsilon_{\mathrm{low}}, 1+\epsilon_{\mathrm{high}}) A_i\Big) - \beta \mathrm{KL}\Big(\pi_\theta(\cdot \mid x) \Big\| \pi_{\mathrm{ref}}(\cdot \mid x)\Big)$$

- $G = 6$: 每个 prompt 采样的 completions 数
- $r_i(\theta) = \pi_\theta(a_i \mid x) / \pi_{\theta_{\mathrm{old}}}(a_i \mid x)$: importance sampling ratio
- $A_i$: advantage
- $\epsilon_{\mathrm{high}} = 0.28$, $\epsilon_{\mathrm{low}} = 0.20$: **asymmetric clipping**——参考 [DAPO](https://arxiv.org/abs/2503.14476)，防 premature entropy collapse
- $\pi_{\mathrm{ref}}$: SFT checkpoint

Advantage 用 group normalization：

$$A_i = \frac{r_i^{\mathrm{env}} - \mathrm{mean}(\{r_j^{\mathrm{env}}\}_{j=1}^G)}{\mathrm{std}(\{r_j^{\mathrm{env}}\}_{j=1}^G)}$$

- $r_i^{\mathrm{env}}$: scalar rollout reward for completion $i$

#### Two-stage reward shaping curriculum

**Stage I (steps 1-80): Loop aggressive**
- 强 loop penalty
- 仅 activate $g_{\mathrm{loop}}$ gate
- Follow shaping mild ($w_{\mathrm{follow}}$ 小，$g_{\mathrm{follow}} \equiv 1$)
- **目标**：让模型从 "repeat failed actions" 转向 "seek help when stuck"

**Stage II (steps 81-160): Loop + Follow aggressive**
- 同 loop penalty
- $(w_{\mathrm{expert}}, w_{\mathrm{follow}}) = (0.3, 2.0)$
- $(\tau_{\mathrm{follow}}, p_{\mathrm{follow-low}}) = (0.5, -2.0)$
- Activate $g_{\mathrm{follow}}$ gate
- **Hard -10 penalty when no expert call made**
- **目标**：把 one-shot escalation 转成 multi-turn pair programming

**为什么必须 two-stage**：如果一开始就强 penalize loop 和 follow，模型可能 collapse 到 always-ask 或 never-ask 的 degenerate policy。先单独抑制 loop，建立 "stalled → escalate" 的 basic behavior，再强制 follow-through。

## 结果

### 主结果 (Table 1)

| Model | Pass@1 (%) |
|-------|------------|
| SWE-agent-LM-7B (prior SLM SOTA) | 17.0 |
| SWE-agent-LM-32B | 40.2 |
| Llama3-SWE-RL-70B | 41.0 |
| Claude 3.5 Sonnet + OpenHands | 53.0 |
| Claude 3.7 Sonnet + SWE-agent | 58.2 |
| Claude 4.5 Sonnet + SWE-agent | 72 |
| **SWE-Protégé-7B (Opus 4.1 expert)** | **42.4** |

7B 模型超过 32B open-weight baseline，与 70B Llama3-SWE-RL 相当。

### RL 增益 (Table 2)

| Expert | SFT Pass@1 | Post-RL Pass@1 | ∆ |
|--------|------------|----------------|---|
| Sonnet 3.7 | 30.6 | 30.8 | +1.2 |
| Sonnet 4.5 | 39.4 | 41.0 | +6.2 |
| Opus 4.1 | 39.6 | 42.4 | +2.8 |

RL phase 平均 +3.4%，在更强 expert 上增益更大。

### 成本 (Fig. 4)

| 配置 | Median cost/task |
|------|-------------------|
| Direct Sonnet 3.7 (expert 自己跑) | $0.54 |
| Direct Sonnet 4.5 | $1.24 |
| SWE-Protégé + Sonnet 3.7 | $0.13 (4.2× lower) |
| SWE-Protégé + Sonnet 4.5 | $0.15 (8.2× lower) |
| SWE-Protégé + Opus 4.1 | $0.65 |

Opus 4.1 per-token 比 Sonnet 贵 5×，但 SWE-Protégé 仍 cost-comparable to direct Sonnet 3.7。

### Token 使用 (Fig. 3)

- Expert tokens 仅占 trajectory 的 ~11.9%
- Expert replies: median 500 tokens
- Expert input: median 8,885 tokens
- Total tokens per task: ~3-3.2 × 10^5
- Phase II 减少总 token ~40%（消除 looping）

### Step efficiency (Fig. 10)

- Post-SFT: ~60 steps/task
- Post-RL: ~20 steps/task
- Expert calls: 稳定 ~4 次

### Training dynamics (Fig. 8, 9)

- **Loop reward**: 从 ~-1.5 快速上升到 ~0 by step 80，后维持接近 0
- **Follow reward**: Stage I 波动 0.2-0.5，Stage II 稳定上升至 0.75-0.85
- **Expert calls**: 稳定 3.5-4.5

**关键 insight**：Loop 抑制在 Stage I 完成，但 performance gain 主要来自 Stage II 的 follow-through。单纯停止 looping 不够，必须 follow through on guidance 才能转化为 solve rate。

### Contamination study

Held-out 400 tasks（disjoint from training）：

| Model | No Expert | With Sonnet 3.7 |
|-------|-----------|-----------------|
| SWE-Protégé-7B | 32.0% | 40.3% |
| SWE-Protégé-32B | 41.5% | 43.0% |

**7B 不加 expert 已达 32%**，远超 baseline 17%。SFT 本身已学到 broad capability，expert 只在 hard cases 提供 marginal gain。

## 关键 ablation (Fig. 7)

### Module modifications

| 变体 | Pass@1 |
|------|--------|
| Loop heuristic + full ctx | 33.4% |
| No loop + 5-msg ctx | 29.4% |
| **Fixed-interval invocation** | **19.6%** |
| **Random invocation** | **24.2%** |

被动 invocation (fixed/random) 显著 underperform，即使 expert advice 频率 comparable。**Autonomous invocation 是核心**——模型必须自己学会判断何时 escalate。

### Dataset modifications

| 变体 | Pass@1 |
|------|--------|
| SWE-Protégé SFT data | 30.6% |
| In-place injection of expert calls into existing trajectories | **14.2%** |

In-place injection（在现有 SWE-smith trajectory 里插入 expert call，minimal edit 后续 message）效果**最差**，低于 baseline 17%。这证明 gains 来自 **coherent interactions that protégé meaningfully conditions on**，单纯添加 expert tokens 无用。

## 为什么 work？Intuition

### 1. Hierarchical capability decomposition

软件修复可分解为两类 sub-skills：
- **Routine reasoning + tool use**：大部分 trajectory 是 file navigation、grep、edit。SLM 完全胜任。
- **High-value intuition**：决定哪个 file/class/method 是 root cause。需要 frontier capability。

SWE-Protégé 的 insight：SLM 不需要内化 high-value intuition，只需学会**何时调用 + 如何执行**。

### 2. 为什么 SFT 不足

SFT 教会 mechanics（如何 call expert），但不教 when 与 how to follow through。Post-SFT 模型能 invoke expert 但经常：
- Stalled 时不 escalate
- 收到 guidance 后 partial follow，relapse into looping

Fig. 18 的例子很典型——模型调了 expert，expert 告诉它看 `lambdify.py` 的 `_EvaluatorPrinter`，模型却跑去看 `numbers.py` 的 `Rational` class，然后又开始 looping。

### 3. 为什么 RL 必须 two-stage

Stage I 只 suppress loop——模型学会 "stalled → escalate"。但光 escalate 不够，Fig. 8 显示 Stage I 结束时 loop reward 已接近 0，但 follow reward 还在 0.2-0.5 波动，SWE-bench accuracy 提升有限。

Stage II 强 follow-through penalty——模型被迫真正执行 guidance 并 report back。Follow reward 从 ~0.2 升到 ~0.8，accuracy 才真正提升。

### 4. 为什么 asymmetric information flow

Expert 只看 5 recent messages + agent's query：
- Token efficiency
- Focus
- **Symmetry breaking**：SLM 持有 full state，被迫成为 primary reasoner

### 5. 为什么 self-judge

复用 expert 本身作为 in-trajectory judge：
- No additional model needed
- Distribution match
- Cost：hidden judge call 不返回给 agent，只 log for RL reward

### 6. 为什么 asymmetric clipping

$\epsilon_{\mathrm{high}} = 0.28 > \epsilon_{\mathrm{low}} = 0.20$——参考 [DAPO](https://arxiv.org/abs/2503.14476)。Standard symmetric PPO clipping 在 positive advantage 时 truncate gradient，导致 premature entropy collapse。Asymmetric 允许 positive updates 更大步幅，保持 exploration。

## 与 related work 的区别

### Model routing ([FrugalGPT](https://arxiv.org/abs/2305.05176), [RouteLLM](https://arxiv.org/abs/2406.18665))

Existing routing 是 per-task, single-turn。SWE-Protégé 是 **long-horizon, multi-turn agentic**——per-step routing signal ill-posed。Participating LMs self-determine 何时/如何协作。

### SWE-smith / SWE-Gym / SWE-RL

这些是 data scaling 或 compute scaling。SWE-Protégé 是 **lightweight post-training**，与 scaling 是 complementary paradigm。

| System | Approach | Scale |
|--------|----------|-------|
| [SWE-smith](https://arxiv.org/abs/2504.21798) | Data scaling | 5K trajectories |
| [SWE-Gym](https://arxiv.org/abs/2412.21139) | Open training env | 491 tasks |
| [SWE-RL](https://arxiv.org/abs/2502.18449) | RL on Llama 3 | 273K seed, 512 H100s |
| **SWE-Protégé** | Expert-protégé collaboration | 4.9K SFT + 100 RL tasks |

## Personal take

### Strengths

1. **Clean two-phase pipeline**——SFT 教 mechanics，RL 教 behavior，分工清晰
2. **Sparse expert usage**——11% tokens, ~4 calls，成本极低
3. **Two-stage reward shaping** 是 elegant curriculum design
4. **Asymmetric clipping** 显示对 agentic RL stability 的 careful thinking
5. **Self-judge** 复用 expert，避免额外 model

### Open questions

1. **Expert dependency**：42.4% 依赖 Opus 4.1。Expert 不可用时退化到 32%。
2. **Generalization**：Python-only，未验证其他 SE ecosystems。
3. **Expert cold-start**：SFT 用 Sonnet 3.7 生成 trajectories，inference 用 Opus 4.1，可能 distribution shift。
4. **Reward hacking**：Judge scores 由 expert 计算，可能引入 bias。
5. **Long-horizon scalability**：75-step limit。1000-step tasks 是否仍 effective？

### Broader connections

- **Hierarchical RL / Options framework**：SLM 是 low-level policy，expert 提供 high-level guidance
- **Process reward models**：Expert-as-judge 是 [PRMs in math reasoning](https://arxiv.org/abs/2110.06874) 的 agentic 版本
- **MoE at inference**：Call-level MoE——routing decisions 通过 explicit tool calls
- **Active learning**：SLM 选择 when to query
- **Curiosity-driven exploration**：Loop penalty = anti-curiosity，但 follow reward 鼓励 productive exploration

### Practical implications

对 production agents ([Cursor](https://cursor.com/home), [OpenClaw](https://github.com/openclaw/openclaw))：
- **Cost constraints**：Frontier model token 是 binding constraint 时最 valuable
- **Latency**：SLM 主要 reasoning，expert 仅 sparse calls
- **Quota/rate-limits**：减少 frontier API calls 8.2×
- **Hybrid edge-cloud**：SLM local，expert remote

## 总结

SWE-Protégé 的核心 insight 是：**SLM 在 long-horizon agentic tasks 上的 capability gap，与其说是模型本身能力不足，不如说是它不会在卡住的时候求助**。

通过两阶段 post-training——SFT 教会 mechanics，RL 教会 behavior——7B 模型学会识别 stalled state、sparse escalation、multi-turn follow-through。最终 42.4% Pass@1，超过 prior SLM SOTA +25.4%，与 32B/70B open-weight 模型相当，expert cost 仅 11.9% tokens。

这暗示了一条 SLM 实用化的可行路径——不一定要 scale up SLM，可以教它学会 selective collaboration with frontier capability。就像 junior dev 不一定要变成 senior dev，只要学会在卡住的时候举手问 senior，就能解决大部分问题。

**Key references**:
- [SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified)
- [SWE-agent](https://arxiv.org/abs/2405.15793)
- [SWE-smith](https://arxiv.org/abs/2504.21798)
- [SWE-Gym](https://arxiv.org/abs/2412.21139)
- [SWE-RL](https://arxiv.org/abs/2502.18449)
- [Lingma-SWE-GPT](https://arxiv.org/abs/2411.00622)
- [SWE-Fixer](https://arxiv.org/abs/2501.05040)
- [CWM](https://arxiv.org/abs/2510.02387)
- [Qwen2.5-Coder](https://arxiv.org/abs/2409.12186)
- [GRPO](https://arxiv.org/abs/2510.13786)
- [DAPO (asymmetric clipping)](https://arxiv.org/abs/2503.14476)
- [Self-Rewarding LMs](https://arxiv.org/abs/2401.10220)
- [FrugalGPT](https://arxiv.org/abs/2305.05176)
- [RouteLLM](https://arxiv.org/abs/2406.18665)
- [SkyRL](https://github.com/NovaSky-AI/SkyRL)
- [Torchtune](https://github.com/meta-pytorch/torchtune)
- [ReAct](https://arxiv.org/abs/2210.03629)
- [SLMs survey (Belcak et al.)](https://arxiv.org/abs/2506.02153)
- [Reward shaping (Ng et al.)](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.51.4949)

---

# SWE-Protégé: 让 SLM 学会与 Expert 协作的 Post-Training Framework

## 1. 核心 Motivation 与 Big Picture

这篇 paper 来自 Meta、University of Michigan 与 Stanford，发表于 2026 年 2 月 26 日。核心 insight 是：**SLM (Small Language Model, ≤10B parameters) 在 long-horizon agentic tasks 上失败的根本原因，并非单纯 capability 不足，而是在 stalled states 下无法 escape degenerative loops**。SWE-bench Verified 上的 prior SLM SOTA 仅 ~17% Pass@1，远低于 Claude 3.5 Sonnet (~50%)。SWE-Protégé 重新框架化了问题——将软件修复视为 expert-protégé pair programming，SLM 保留为 primary decision-maker，学习 *何时* 求助 + *如何* 遵循 expert 指导。

最终 SWE-Protégé-7B (Qwen2.5-Coder-7B-Instruct post-trained) 在 SWE-bench Verified 上达到 **42.4% Pass@1**（+25.4% over SWE-agent-LM-7B），expert 调用稀疏（~4 calls/task，11% tokens），成本比 expert-only agent 低 8.2×。

Reference: [SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified), [SWE-agent](https://arxiv.org/abs/2405.15793), [SWE-smith](https://arxiv.org/abs/2504.21798)

---

## 2. 为什么 SLM 在 SWE-bench 上挣扎？Failure Mode 分析

### 2.1 Degenerative Action Looping

SLM 在长 horizon agent tasks 上最典型的失败模式是 **repetitive tool-use sequences**——模型反复调用相同的 base commands（如 `grep`, `find`, `str_replace_editor view`），陷入无 productive exploration。Paper 在 Fig. 6 量化了这一现象：用 repeated tool-use sequence length $L$ 作为 metric，测量 fraction of trajectories containing loops longer than $L$：

- **SWE-agent-LM-32B**: 15-20% trajectories 含 $L > 10$，10% 含 $L > 40$
- **SWE-agent-LM-7B**: 33.6% 含 $L > 10$
- **SWE-Protégé-7B post-SFT (P1)**: 31% 含 $L > 10$，8% 含 $L > 40$
- **SWE-Protégé-7B post-RL (P2)**: 仅 0.8% 含 $L > 10$，0% 含 $L > 20$

这个 gap 极为关键：纯 SFT 不能消除 looping，但 RL phase 用 shaped reward 几乎完全消除。

### 2.2 Data Scaling 的 Plateau

Fig. 2 展示了 training data scaling curves。SWE-agent-LM-7B 在 2.4K trajectories 时 plateau 在 ~17%，继续 scale 到 5K 反而 regression 到 11.8%。SWE-Protégé-7B（Sonnet 3.7 expert）从 1.3K (19.0%) 单调增至 4.8K (33.4%)，+14.4% 净增益；Sonnet 4.5 expert 从 23.6% → 39.4% (+15.8%)。这表明 **expert-augmented trajectories 提供了 SLM 无法独立产出的 high-value reasoning signal**。

Reference: [SWE-Gym](https://arxiv.org/abs/2412.21139), [Lingma-SWE-GPT](https://arxiv.org/abs/2411.00622)

---

## 3. SWE-Protégé 方法详解

### 3.1 Agent System Setup

SWE-Protégé 仅在 SWE-agent 上 **添加一个 tool**: `ask_expert`。状态空间定义为：

- $s \in S$: full agent state，包括 interaction history、tool outputs、system prompts
- Policy $\pi_\theta(a \mid s)$ over actions $a \in \mathcal{A}$
- Extended action space: $\mathcal{A}' = \mathcal{A} \cup \{\text{ask\_expert}\}$

`ask_expert` 是一个 structured tool call，SLM 生成 query，expert 返回 textual guidance，append 到 $s$ 供后续决策。

**关键设计**：Asymmetric information flow
- SLM 看 **完整 state** $s$
- Expert 只看 compact summary $\tilde{s}$，由 most recent $K=5$ messages 组成
- 这避免了 expert token cost 爆炸，同时保持 escalation 的 focus

### 3.2 Phase I: Supervised Induction of Expert Usage

**Goal**: 教 SLM expert 交互的 **mechanics** 与 **local semantics**——如何调用 tool、如何 formulate contextually appropriate query。并不解决 *when to escalate* 与 *how to follow through*。

#### Synthetic Trajectory Generation

用 Claude Sonnet 3.7 作为 trajectory generator，在 SWE-agent 环境中执行 tasks。关键 trick：**expert 也是 Sonnet 3.7 自己**，且 **expert 被允许看到 ground-truth patch**（但被指示不要 verbatim reveal answer）。这产生了 ~38% more usable trajectories vs 不给 ground truth。最终获得 ~4.8K accepted trajectories，与 SWE-smith 训练集 size 相当。

#### SFT Loss

$$\mathcal{L}_{\mathrm{SFT}}(\theta) = -\mathbb{E}_{(s_i, y_i)}[\log p_\theta(y_i \mid s_i)]$$

变量解释：
- $s_i$: input state (agent history + tool outputs + system prompt)
- $y_i$: target token sequence (the next action/thought)
- $\theta$: SLM parameters (Qwen2.5-Coder-7B-Instruct)
- $p_\theta(y_i \mid s_i)$: SLM's predicted probability of target sequence

**No auxiliary loss**——expert call 稀疏性 emerges implicitly through imitating trajectories where expert calls are infrequent。这是一个 elegant design choice，避免 hard-coded sparsity constraints。

**Implementation detail**: Rejection-sampling full SFT with Torchtune，max sequence length 32,762，batch size 32。作者试验了 LoRA/QLoRA，均 underperform full SFT。

### 3.3 Phase II: RL for Expert-SLM Pair Programming

**Goal**: 让 SLM 真正成为 autonomous pair programmer——(i) stalled 时 escalate 而非 loop， 多轮遵循 expert 指导。使用 GRPO-style on-policy RL with composite shaped reward。

#### Collaboration Quality Judge Scores

对于 trajectory $\tau$ 中的 $N$ 次 expert calls，每次 call $i$ 定义：

$$u_i := J_{\mathrm{warrant}}(q_i, \tilde{s}_i) \in [0, 1], \quad f_i := J_{\mathrm{follow}}(g_i, \Delta_i(\tau)) \in [0, 1]$$

变量解释：
- $u_i$: **warranted score**——escalation 是否合理（discouraging lazy invocations）
- $f_i$: **follow-through score**——agent 是否遵循了 guidance 并 report back
- $q_i$: agent 向 expert 发送的 query
- $g_i$: expert 返回的 guidance
- $\tilde{s}_i$: compact context packet 给 expert
- $\Delta_i(\tau)$: 接收 $g_i$ 后到下一次 expert call 或 termination 之间的 agent 行为 segment
- $J_{\mathrm{warrant}}, J_{\mathrm{follow}}$: judge functions，由 expert 本身担任 in-trajectory judge（参考 [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10220)）

这两个 score 通过 hidden judge call 计算，logged per event，不返回给 agent。

#### Composite Reward

$$R_{\mathrm{total}}(\tau, x) = R_{\mathrm{loop}}(\tau) + w_{\mathrm{follow}} R_{\mathrm{follow}}(\tau) + g_{\mathrm{loop}}(\tau) g_{\mathrm{follow}}(\tau) R_{\mathrm{other}}(\tau, x)$$

$$R_{\mathrm{other}}(\tau, x) = R_{\mathrm{correct}}(\tau, x) + w_{\mathrm{sim}} R_{\mathrm{sim}}(\tau, x) + w_{\mathrm{expert}} R_{\mathrm{expert}}(\tau)$$

变量解释：
- $\tau$: full agent trajectory (sequence of actions and tool responses)
- $x$: task instance metadata (gold patch, etc.)
- $R_{\mathrm{loop}} \leq 0$: penalize degenerative looping
- $R_{\mathrm{follow}}$: follow-through reward
- $R_{\mathrm{correct}} \in \{0, 1\}$: 是否通过 unit tests
- $R_{\mathrm{sim}} \in \{-1, 0\}$: similarity fallback（discrete, thresholded）
- $R_{\mathrm{expert}}$: expert deferral quality，使用 $\{u_i\}$
- $g_{\mathrm{loop}}, g_{\mathrm{follow}} \in \{0, 0.5, 1\}$: gating functions
- $w_{\mathrm{follow}}, w_{\mathrm{sim}}, w_{\mathrm{expert}}$: weights

**Key design**: gates $g_{\mathrm{loop}}, g_{\mathrm{follow}}$ downweight $R_{\mathrm{other}}$ 当 severe looping 或 failed follow-through 时，**防止 correctness/similarity mask pathological interaction behavior**。例如，如果 agent 恰好猜对 patch 但 looping 严重，correctness=1 不应被全权奖励。

#### GRPO Objective

$$J(\theta) = \frac{1}{G} \sum_{i=1}^G \min\Big(r_i(\theta) A_i, \mathrm{clip}(r_i(\theta), 1-\epsilon_{\mathrm{low}}, 1+\epsilon_{\mathrm{high}}) A_i\Big) - \beta \mathrm{KL}\Big(\pi_\theta(\cdot \mid x) \Big\| \pi_{\mathrm{ref}}(\cdot \mid x)\Big)$$

变量解释：
- $G$: 每个 prompt 采样的 completions 数 (6 rollouts per prompt)
- $r_i(\theta) = \pi_\theta(a_i \mid x) / \pi_{\theta_{\mathrm{old}}}(a_i \mid x)$: importance sampling ratio
- $A_i$: advantage
- $\epsilon_{\mathrm{high}} = 0.28$, $\epsilon_{\mathrm{low}} = 0.20$: **asymmetric clipping**——参考 [DAPO](https://arxiv.org/abs/2503.14476) 的设计，reduce premature entropy collapse
- $\beta$: KL penalty coefficient
- $\pi_{\mathrm{ref}}$: reference policy (SFT checkpoint)

#### Advantage via Group Normalization

$$A_i = \frac{r_i^{\mathrm{env}} - \mathrm{mean}(\{r_j^{\mathrm{env}}\}_{j=1}^G)}{\mathrm{std}(\{r_j^{\mathrm{env}}\}_{j=1}^G)}$$

- $r_i^{\mathrm{env}}$: scalar rollout reward for completion $i$
- 分子：去 mean 后的 deviation
- 分母：group std，normalize 信号 scale

这是 standard GRPO 的 advantage 计算，参考 [GRPO scaling paper](https://arxiv.org/abs/2510.13786)。

#### Correctness Reward

$$R_{\mathrm{correct}}(\tau, x) = \mathbb{1}\{p(\tau) \text{ passes verification}\}$$

- $p(\tau)$: final patch submitted at end of $\tau$
- 在 fresh environment identical to rollout backend re-run unit tests

#### Similarity Fallback (unresolved only)

$$R_{\mathrm{sim}}(\tau, x) = \begin{cases} 0, & \mathrm{sim}(\tau, x) \geq \theta \\ -1, & \mathrm{sim}(\tau, x) < \theta \end{cases}$$

- $\mathrm{sim}(\tau, x) \in [0, 1]$: string similarity between model patch 和 gold patch (after filtering diff noise)
- $\theta = 0.5$ in practice
- 当 unresolved 时仍提供 stable signal：unrelated patches 被惩罚 $-1$，partial overlap 得到 neutral $0$

#### Stall Penalty (Loop Reward)

$$R_{\mathrm{loop}}(\tau) = \max\Big(c_{\mathrm{loop}}, -\lambda_{\mathrm{loop}}\Big(\max(0, s_1 - k_1 + 1) + \sum_{j \geq 2} \max(0, s_j - k_2 + 1)\Big)\Big)$$

变量解释：
- $s_1, s_2, \ldots, s_M$: lengths of maximal consecutive identical-command streaks（按时间顺序）
- $k_1$: first trigger threshold (initial: 15)
- $k_2 < k_1$: subsequent trigger threshold (subsequent: 8)
- $\lambda_{\mathrm{loop}} = 0.5$: penalty scaling
- $c_{\mathrm{loop}} = -10$: cap on penalty magnitude

**Design**: 第一次 trigger 用 $k_1=15$（容忍 longer initial streak），后续用 $k_2=8$（更严格）。这使 stalling sparse but decisive——short repeats 被容忍，true degeneracy 被严厉惩罚。

Base command normalization 细节：
- Strip leading env-var assignments
- Chain commands (`&&`/`;`) → 保留最后一个 subcommand
- First token 作为 base command (除 `git status`, `git diff` 等两词 base)
- Navigation-like ops (`grep`, `find`, `str_replace_editor view`) collapse into one equivalence class

#### Expert Warrant Reward

$$\phi(u) = \begin{cases} u, & u \geq \tau_{\infty} \\ p_{\mathrm{low}}, & u < \tau_{\infty} \end{cases}, \quad R_{\mathrm{warrant}}(\tau) = \mathrm{Agg}\Big(\{\phi(u_i)\}_{i=1}^N\Big)$$

- $\tau_{\infty}$: warrant threshold
- $p_{\mathrm{low}} \leq 0$: penalty for low-warrant calls
- $\mathrm{Agg}$: 通常 mean (或 min for stricter budgeting)

#### Back-to-Back Penalty

$$R_{\mathrm{b2b}}(\tau) = \max(-1, \lambda_{\mathrm{b2b}} n_{\mathrm{b2b}})$$

- $n_{\mathrm{b2b}}$: number of back-to-back expert calls
- $\lambda_{\mathrm{b2b}} \leq 0$: penalty coefficient

$$R_{\mathrm{expert}}(\tau) = R_{\mathrm{warrant}}(\tau) + R_{\mathrm{b2b}}(\tau) + \mathbb{1}[\text{quota enabled}] R_{\mathrm{quota}}(\tau)$$

#### Follow-Through Reward

$$\psi(f) = \begin{cases} f, & f \geq \tau_{\mathrm{follow}} \\ f_{\mathrm{follow-low}}, & f < \tau_{\mathrm{follow}} \end{cases}, \quad R_{\mathrm{follow}}(\tau) = \mathrm{Agg}\Big(\{\psi(f_i)\}_{i \in T}\Big)$$

- $\tau_{\mathrm{follow}} = 0.5$: follow threshold
- $f_{\mathrm{follow-low}} = -2.0$: penalty
- $T$: indices where follow-through is defined (excluding terminal calls)

#### Gating Functions

$$g_{\mathrm{loop}}(\tau) = \begin{cases} 0, & R_{\mathrm{loop}}(\tau) \leq a_2 \\ 0.5, & R_{\mathrm{loop}}(\tau) \leq a_1 \\ 1, & \text{otherwise} \end{cases}, \quad g_{\mathrm{follow}}(\tau) = \begin{cases} 0, & R_{\mathrm{follow}}(\tau) \leq b_2 \\ 0.5, & R_{\mathrm{follow}}(\tau) \leq b_1 \\ 1, & \text{otherwise} \end{cases}$$

- $a_2 < a_1 \leq 0$, $b_2 < b_1 \leq 0$: thresholds
- $R_{\mathrm{follow}}$ never gated out，只 $R_{\mathrm{other}}$ 被 gate

#### Reward Shaping Curriculum (Two-Stage)

**Stage I (steps 1-80): Loop aggressive shaping** — 诱导 escalation
- $(k_1, k_2, \lambda_{\mathrm{loop}}, c_{\mathrm{loop}}) = (15, 8, 0.5, -10)$
- 仅 activate $g_{\mathrm{loop}}$ gate
- $w_{\mathrm{follow}}$ 小，$g_{\mathrm{follow}} \equiv 1$（follow gate inactive）
- **目标**: 让 policy 从 "repeat failed actions" 转向 "seek help when stuck"

**Stage II (steps 81-160): Loop + Follow aggressive shaping** — 强制 pair programming
- 同 loop penalty
- $(w_{\mathrm{expert}}, w_{\mathrm{follow}}) = (0.3, 2.0)$
- $(\tau_{\mathrm{follow}}, p_{\mathrm{follow-low}}) = (0.5, -2.0)$
- Activate $g_{\mathrm{follow}}$ gate
- **Hard -10 penalty when no expert call made**
- **目标**: 将 one-shot escalation 转化为 multi-turn collaboration

### 3.4 RL Infrastructure

基于 [NovaSky-AI SkyRL](https://github.com/NovaSky-AI/SkyRL)（Ray-based）+ SWE-agent 在 SWE-ReX Docker runtime。关键工程优化：
- Cap concurrent SWE-agent/Docker startups + I/O backoff
- Trajectory-level checkpointing 支持 mid-run reward-shaping updates
- Pipeline inference with multiple in-flight batches (high vLLM utilization)
- 8 A100/H100 80G GPUs，100-task subset，160 total steps

---

## 4. 主要实验结果

### 4.1 SWE-bench Verified 主结果 (Table 1)

| Model | System | Train Size | Pass@1 (%) |
|-------|--------|-----------|------------|
| Claude 4.5 Sonnet | SWE-agent | - | 72 |
| Claude 3.7 Sonnet | SWE-agent | - | 58.2 |
| Claude 3.5 Sonnet | OpenHands | - | 53.0 |
| Llama3-SWE-RL-70B | Agentless | 11M | 41.0 |
| SWE-agent-LM-32B | SWE-agent | 5k | 40.2 |
| SWE-gym-7B | OpenHands | 491 | 10.6 |
| SWE-agent-LM-7B | SWE-agent | 2.4k | 17.0 |
| Lingma-SWE-GPT-7B | SWE-SynInfer | - | 18.2 |
| **SWE-Protégé-7B (Sonnet 3.7)** | SWE-agent | 4.9k | 30.8 |
| **SWE-Protégé-7B (Sonnet 4.5)** | SWE-agent | 4.9k | 41.0 |
| **SWE-Protégé-7B (Opus 4.1)** | SWE-agent | 4.9k | **42.4** |

**Takeaway**: 7B SLM 超越 prior 32B open-weight baseline (40.2%)，与 70B Llama3-SWE-RL (41.0%) 相当，使用 SOTA expert (Opus 4.1) 时还 +1.4%。

### 4.2 Phase II RL 增益 (Table 2)

| Model | Pass@1 (%) | ∆ over SFT |
|-------|------------|------------|
| SWE-Protégé-7B (Sonnet 3.7) | 30.6 | +1.2 |
| SWE-Protégé-7B (Sonnet 4.5) | 41.0 | +6.2 |
| SWE-Protégé-7B (Opus 4.1) | 42.4 | +2.8 |

平均 +3.4% 增益——RL phase 行为塑造效果显著，尤其在更强 expert 上（Sonnet 4.5: +6.2%）。

### 4.3 Cost Analysis (Fig. 4)

| 配置 | Median Cost / Task |
|------|--------------------|
| Direct Sonnet 3.7 | $0.54 |
| Direct Sonnet 4.5 | $1.24 |
| SWE-Protégé + Sonnet 3.7 | $0.13 (4.2× lower) |
| SWE-Protégé + Sonnet 4.5 | $0.15 (8.2× lower) |
| SWE-Protégé + Opus 4.1 | $0.65 |

Opus 4.1 per-token 5× 和 4.54× 更贵，但 SWE-Protégé 仍 cost-comparable to direct Sonnet 3.7。

### 4.4 Token Usage (Fig. 3)

- Expert tokens 仅占 trajectory 的 ~11.9% (Sonnet 4.5)
- Expert replies: median 500 / p95 937 / max 1,657 tokens
- Expert input context: median 8,885 / p95 20,716 / max 43,031 tokens
- Total tokens per task: ~3-3.2 × 10^5 (跨 expert 一致)
- Phase II 减少总 token ~40%（通过消除 looping）

**Insight**: Swapping experts 主要改变 quality 而非 agent-side work amount。

### 4.5 Step Efficiency (Fig. 10)

- Post-SFT: ~60 steps/task
- Post-RL: ~20 steps/task
- Expert calls 稳定在 ~4
- 10.8% instances resolved after ≥40 steps（保持 long-horizon focus）

### 4.6 Loop Reduction (Fig. 6)

| Model | Trajectories with loops > 10 steps |
|-------|------------------------------------|
| SWE-agent-LM-32B | 24.4% |
| SWE-agent-LM-7B | 33.6% |
| SWE-Protégé-7B P1 (post-SFT) | 31.0% |
| SWE-Protégé-7B P2 (post-RL) | **0.8%** |
| Sonnet 3.7 | 1.8% |
| Sonnet 4.5 | 1.8% |

SWE-Protégé-7B post-RL 甚至优于 frontier Claude models 在 loop 抑制上。

### 4.7 Training Dynamics (Fig. 8, 9)

- **Loop reward**: 从 ~-1.5 起步，loop-aggressive shaping 阶段快速上升至 ~0 by step 80，后维持接近 0
- **Follow reward**: Stage I 波动 0.2-0.5，Stage II 稳定上升至 0.75-0.85 by step 160
- **Expert reward**: 稳定在 0.82-0.88 narrow band
- **Mean expert calls**: 稳定 3.5-4.5，mild upward drift
- **Mean step count**: 从 55-65 降至 35-40

**Key insight**: Loop 抑制在 Stage I 完成，但 performance gain 主要来自 Stage II 的 follow-through。这表明单纯停止 looping 不足，必须 follow through on guidance 才能转化为 solve rate。

### 4.8 Failure Mode Shift (Fig. 5)

Post-RL vs Post-SFT：
- Runtime limit aborts 显著减少（之前 common failure even for SWE-agent-LM-32B）
- Agent 不再 stuck，而是 decisively follow guidance 到 end-to-end attempts，即使 guidance imperfect

### 4.9 Ablations (Fig. 7)

**Module modifications**:
- Loop heuristic + full ctx (Loop✓ Ctx×): 33.4%
- Loop heuristic + 5-msg ctx (Loop✓ Ctx✓): 29.4%
- No loop + 5-msg ctx (Loop× Ctx✓): 29.4%
- No loop + full ctx (Loop× Ctx×): 29.0%
- Fixed-interval invocation: **19.6%** (差)
- Random invocation: **24.2%** (差)

**关键 finding**: 被动 invocation (fixed/random) 显著 underperform，即使 expert advice 频率 comparable 或更高。**Autonomous invocation 是核心**。

**Expert modifications**:
- SWE-Protégé-7B expert: 17.0%
- SWE-Protégé-32B expert: 20.8%
- Frontier backends: 显著更强

**Dataset modifications**:
- In-place injection of expert calls into existing SWE-smith trajectories: **14.2%** (最差，低于 baseline 17%)

这证明 gains 来自 **coherent interactions that protégé meaningfully conditions on**，单纯添加 expert tokens 无用。

### 4.10 Contamination Study

Held-out 400 tasks（SWE-smith-style subset, disjoint from training, released after expert models available）：

| Model | No Expert | With Sonnet 3.7 |
|-------|-----------|-----------------|
| SWE-Protégé-7B | 32.0% | 40.3% |
| SWE-Protégé-32B | 41.5% | 43.0% |

**Important**: 7B model 不加 expert 已达 32.0%，远超 baseline 17%。这表明 SFT 本身已学到 broad capability，expert 只在 hard cases 提供 marginal gain。

---

## 5. 为什么 SWE-Protégé Work？Intuition Building

### 5.1 Hierarchical Capability Decomposition

软件修复任务可分解为两类 sub-skills：
1. **Routine reasoning + tool use**: 大部分 trajectory 是 file navigation、grep、edit。SLM 完全胜任。
2. **High-value intuition**: 决定哪个 file/class/method 是 root cause，理解 cross-module 依赖。需要 frontier capability。

SWE-Protégé 的核心 insight：**SLM 不需要内化 high-value intuition，只需学会何时调用 + 如何执行**。这类似于 junior dev + senior dev 的 pair programming——junior 写大部分代码，senior 在 key decision points 提供 guidance。

### 5.2 为什么 SFT 不足

SFT 教会 SLM **mechanics**（如何 call expert），但不教 **when** 与 **how to follow through**。Post-SFT 模型能 invoke expert 但经常：
- Stalled 时不 escalate
- 收到 guidance 后 partial follow，relapse into looping（见 Fig. 18 例子）

### 5.3 为什么 RL 必须 Two-Stage

**Stage I (loop aggressive)**: 如果同时强 penalize loop 和 follow，模型可能 collapse 到 always-ask 或 never-ask 的 degenerate policy。先单独抑制 loop 让模型学会 "stalled → escalate" 的 basic behavior。

**Stage II (follow aggressive)**: 一旦 escalation 行为建立，强 follow-through penalty 强制模型真正执行 guidance 并 report back。这是 multi-turn pair programming 的 emergence。

### 5.4 为什么 Asymmetric Information Flow

Expert 只看 5 recent messages + agent's query：
- **Token efficiency**: 避免 expert 处理 full history 的 cost
- **Focus**: Expert 只需 contextual guidance，不需 understand full trajectory
- **Symmetry breaking**: SLM 持有 full state，被迫成为 primary reasoner

### 5.5 为什么 Self-Judge (Expert as Judge)

复用 expert 本身作为 in-trajectory judge 提供 process supervision：
- **No additional model needed**: Expert 已是 frontier model，judge capability 足够
- **Distribution match**: Judge 评估的正是 expert 自己会怎么想 warranted/follow-through
- **Cost**: Hidden judge call 不返回给 agent，只 log for RL reward

这参考了 [Self-Rewarding LM](https://arxiv.org/abs/2401.10220) 和 process reward models 的思路。

### 5.6 为什么 Asymmetric Clipping

$\epsilon_{\mathrm{high}} = 0.28 > \epsilon_{\mathrm{low}} = 0.20$——参考 [DAPO](https://arxiv.org/abs/2503.14476)。Standard symmetric PPO clipping 在 positive advantage 时 truncate gradient，导致 **premature entropy collapse**。Asymmetric 允许 positive updates 更大步幅，保持 exploration。这对 agentic RL 尤为重要——agent policy 需要保留 diverse action distribution。

---

## 6. 与 Related Work 的对比

### 6.1 Model Routing ([FrugalGPT](https://arxiv.org/abs/2305.05176), [RouteLLM](https://arxiv.org/abs/2406.18665))

Existing routing 工作：
- Per-task routing (single-turn)
- Predictive routing (a priori 选 model)
- Non-predictive routing (sequential execution until quality threshold)

SWE-Protégé 区别：
- **Long-horizon, multi-turn agentic**——per-step routing signal ill-posed
- **Self-determined collaboration**——participating LMs 自己决定何时/如何协作
- SLM 是 primary driver，expert 是 sparse collaborator

### 6.2 SWE-smith / SWE-Gym / SWE-RL

| System | Approach | Scale |
|--------|----------|-------|
| [SWE-smith](https://arxiv.org/abs/2504.21798) | Data scaling (128 repos, synthesized tasks) | 5K trajectories |
| [SWE-Gym](https://arxiv.org/abs/2412.21139) | Open training env | 491 tasks |
| [SWE-RL](https://arxiv.org/abs/2502.18449) | RL on Llama 3 | 273K seed, 512 H100s |
| [Lingma-SWE-GPT](https://arxiv.org/abs/2411.00622) | Dev-process-centric training | 7B/72B |
| [SWE-Fixer](https://arxiv.org/abs/2501.05040) | Specialized retriever + editor | 110K |
| [CWM](https://arxiv.org/abs/2510.02387) | End-to-end + test-time scaling | 32B |
| **SWE-Protégé** | Expert-protégé collaboration | 4.9K SFT + 100 RL tasks |

SWE-Protégé 是 **lightweight post-training**，与 data scaling / compute scaling 是 complementary paradigm。

### 6.3 SLMs in Agentic AI ([Belcak et al. 2025](https://arxiv.org/abs/2506.02153))

Prior SLM 工作：single-turn QA, math reasoning, single-turn coding。SWE-Protégé 是 **first usable SLM on long-horizon agentic coding task**。

---

## 7. Limitations 与 Future Work

1. **Expert treated as black-box**——未做 expert post-training 或 co-adaptation
2. **Single agent framework** (SWE-agent)——未探索其他 scaffold
3. **Python-only** SWE-bench
4. **Phase I/II hyperparameters** 未 exhaustively tuned
5. **Alternate collaboration strategies** 未探索（expert interrupts, bidirectional control）
6. **Broader student model families** 未测试

Ablation 显示 in-house 7B/32B experts underperform frontier——expert post-training 是 orthogonal but non-trivial direction。

---

## 8. Critical Analysis 与 Personal Take

### 8.1 Strengths

1. **Clean two-phase pipeline**——SFT 教 mechanics，RL 教 behavior
2. **Sparse expert usage**——11% tokens, ~4 calls，成本极低
3. **Two-stage reward shaping** 是 elegant curriculum design
4. **Asymmetric clipping** 显示对 agentic RL stability 的 careful thinking
5. **Self-judge** 复用 expert，避免额外 model

### 8.2 Open Questions

1. **Expert dependency**: 42.4% 依赖 Opus 4.1。如果 expert 不可用 (offline, privacy)，performance 退化到 32% (no expert)。这是 production deployment 的考量。
2. **Generalization beyond SWE-bench**: Python-only，未验证其他 SE ecosystems
3. **Expert cold-start**: 7B SFT 阶段用 Sonnet 3.7 生成 trajectories，与 inference-time expert 可能 distribution shift
4. **Reward hacking**: Judge scores 由 expert 计算，可能引入 bias。Paper 未深入分析。
5. **Long-horizon scalability**: 75-step limit。如果 1000-step tasks，loop detection 和 reward shaping 是否仍 effective？

### 8.3 Connections to Broader Themes

- **Hierarchical RL / Options framework**: SLM 是 low-level policy，expert 提供 high-level guidance。这与 [Options framework](https://arxiv.org/abs/1606.01868) 在 RL 中的 hierarchy 类比。
- **Process reward models**: Expert-as-judge 是 [PRMs in math reasoning](https://arxiv.org/abs/2110.06874) 的 agentic 版本
- **Mixture of Experts (MoE) at inference**: 不是参数级 MoE，是 **call-level MoE**——routing decisions 通过 explicit tool calls 而非 learned gating
- **Active learning**: SLM 选择 when to query——主动学习的 spirit
- **Curiosity-driven exploration**: Loop penalty 可视为 anti-curiosity，但 follow reward 鼓励 productive exploration after guidance

### 8.4 Practical Implications

For production agents (Cursor, [OpenClaw](https://github.com/openclaw/openclaw)):
- **Cost constraints**: SWE-Protégé 在 frontier model token 是 binding constraint 时最 valuable
- **Latency**: SLM 主要 reasoning，expert 仅 sparse calls——latency 接近 SLM-only
- **Quota/rate-limits**: 减少 frontier API calls 8.2× 显著缓解 rate limit 压力
- **Local deployment**: SLM 可 local，expert 可 remote——hybrid edge-cloud 设计

### 8.5 Reimplementation Notes

如要复现：
1. **SFT**: Torchtune + Qwen2.5-Coder-7B-Instruct + ~4.8K expert-augmented trajectories (Sonnet 3.7 generated with gold patch access)
2. **RL**: SkyRL + SWE-agent + SWE-ReX Docker + 100 tasks + 160 steps
3. **Expert tool**: 在 SWE-agent YAML 加 `tools/expert_llm` bundle，per-task quota 6
4. **History processing**: 修改 `last_n_observations` 不 elide `<expert_llm_guidance>` 标记
5. **Reward**: 实现所有 reward terms + gates + two-stage curriculum

---

## 9. 总结

SWE-Protégé 的核心贡献是 **将 long-horizon SLM agent 的 capability gap 重新框架化为 collaboration problem**。通过两阶段 post-training，7B SLM 学会：
- 识别 stalled states
- Sparse 但 high-value escalation
- Multi-turn follow-through on guidance

最终达到 42.4% Pass@1 on SWE-bench Verified，超过 prior SLM SOTA +25.4%，与 32B/70B open-weight 模型相当，同时 expert cost 仅 11.9% tokens，成本 8.2× lower than expert-only。这暗示了一条 SLM 实用化的可行路径——不是单纯 scale up SLM，而是学会 selective collaboration with frontier capability。

**Key references**:
- [SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified)
- [SWE-agent](https://arxiv.org/abs/2405.15793)
- [SWE-smith](https://arxiv.org/abs/2504.21798)
- [SWE-Gym](https://arxiv.org/abs/2412.21139)
- [SWE-RL](https://arxiv.org/abs/2502.18449)
- [Lingma-SWE-GPT](https://arxiv.org/abs/2411.00622)
- [SWE-Fixer](https://arxiv.org/abs/2501.05040)
- [CWM](https://arxiv.org/abs/2510.02387)
- [Qwen2.5-Coder](https://arxiv.org/abs/2409.12186)
- [GRPO](https://arxiv.org/abs/2510.13786)
- [DAPO (asymmetric clipping)](https://arxiv.org/abs/2503.14476)
- [Self-Rewarding LMs](https://arxiv.org/abs/2401.10220)
- [FrugalGPT](https://arxiv.org/abs/2305.05176)
- [RouteLLM](https://arxiv.org/abs/2406.18665)
- [SkyRL](https://github.com/NovaSky-AI/SkyRL)
- [Torchtune](https://github.com/meta-pytorch/torchtune)
- [ReAct](https://arxiv.org/abs/2210.03629)
- [SLMs survey (Belcak et al.)](https://arxiv.org/abs/2506.02153)
- [Reward shaping (Ng et al.)](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.51.4949)
