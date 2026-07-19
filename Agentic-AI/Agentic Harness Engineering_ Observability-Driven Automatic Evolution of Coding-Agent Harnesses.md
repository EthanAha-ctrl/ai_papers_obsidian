---
source_pdf: Agentic Harness Engineering_ Observability-Driven Automatic Evolution
  of Coding-Agent Harnesses.pdf
paper_sha256: 3b0642752c1682caec1e8b38591f6da7de5441cd8fe74f1043a3bc42c31577f4
processed_at: '2026-07-18T05:01:52-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Agentic Harness Engineering 深度技术解析

Andrej, 这篇 paper 戳中了一个我一直觉得被低估的痛点：coding agent 的能力上限往往 NOT bottlenecked 在 base model, 而 bottlenecked 在围绕 model 的那一层 "harness" engineering。这篇工作把 harness 从一个手工艺品变成了一个可以 autonomous evolution 的 learnable surface, 而且 core insight 非常 clean —— bottleneck 在 observability, 不在 agent capability。

下面我从 motivation, 三个 observability pillars, Algorithm 1 的形式化, 实验数据, ablation 发现, 一直讲到它对未来 self-evolving systems 的启示。

---

## 1. 为什么 Harness Engineering 是 first-class lever

先回顾一下 coding agent 的实际 stack。一个 deployed coding agent 由以下 layers 组成:

- **Base model** $M$ (e.g., GPT-5.4, Claude, Gemini)
- **System prompt**: shapes work style, reasoning discipline
- **Tools**: file system access, shell execution, search
- **Middleware**: context management, execution hooks, recovery
- **Long-term memory**: persistent cross-session knowledge
- **Skills / Sub-agents**: reusable patterns, delegated execution

把这些 model-external, editable 的 components 合称为 **harness** $H$。一个 coding agent 的 task success 可以粗略建模为:

$$\text{success}(M, H, \text{task}) = f(M) \cdot g(H, \text{task}) + \epsilon$$

其中 $f(M)$ 是 base model 的 intrinsic capability, $g(H, \text{task})$ 是 harness 把 capability 转化为 task success 的 efficiency。Reference: [Harness Engineering at OpenAI](https://openai.com/zh-Hans-CN/index/harness-engineering/), [Anthropic on harness design](https://www.anthropic.com/engineering/harness-design-long-running-apps), [Langchain Deep Agents](https://www.langchain.com/blog/improving-deep-agents-with-harness-engineering).

Paper 的第一个核心 observation: $g(H, \text{task})$ 的 variance 极大。同一个 base model, 用 bash-only minimal seed 还是用 human-engineered Codex-CLI harness, 在 Terminal-Bench 2 上 pass@1 差了 2.2 pp (69.7% vs 71.9%), 在 Hard tier 上差了 5 pp (51.7% vs 56.7%)。这意味着 harness 是一个 first-class optimization lever, AND 它的 optimal point 是 model-specific 的 —— 换 base model 就要 re-tune。

Manual loop 的瓶颈在于:
- Base model 升级太快 (GPT-5.4, Kimi K2.6, MiMo V2.5 Pro 几个月内迭代)
- Trajectory 长度爆炸 (millions of tokens), actionable signal 被 noise 淹没
- Harness 各个 component tight coupling, 改一个 file 风险高

Reference: [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html), [Bitter lesson of agent harnesses](https://browser-use.com/posts/bitter-lesson-agent-harnesses).

---

## 2. AHE 的 Central Insight: Observability Bottleneck

Paper 的 thesis statement 我非常欣赏:

> Once the evolution agent receives structured context over a clear action space, it can reliably converge on better harness designs.

换句话说, 之前 self-evolving agent 的失败 NOT 因为 evolve agent 不够 smart, 而 BECAUSE 它看不见自己该改什么、改了之后发生了什么。这跟 RL 里 reward sparsity 的问题有异曲同工之妙 —— 你需要一个 dense, structured feedback signal 才能 learn efficiently。

AHE 把这个 observability problem 分解成三个 pillars, 每一个 pillar 对应一个具体的 bottleneck:

| Bottleneck | Pillar | Substrate |
|---|---|---|
| Heterogeneous action space across components | Component observability | NexAU decoupled file-level harness |
| Millions of trajectory tokens bury signal | Experience observability | Agent Debugger layered evidence corpus |
| Edits' effect is hard to attribute | Decision observability | Change manifest with falsifiable predictions |

---

## 3. Pillar 1: Component Observability via NexAU

### 3.1 七种 Component Types

NexAU framework (reference: [NexAU GitHub](https://github.com/nex-agi/NexAU), [Nex N1 paper](https://arxiv.org/abs/2512.04987)) 把 harness 暴露成 7 个 orthogonal component types, 每个都是 workspace 下一个独立 file:

```
workspace/
├── systemprompt.md           # System prompt (Jinja template)
├── LongTermMEMORY.md         # Persistent cross-session knowledge
├── ShortTermMEMORY.md        # Session-scoped (read-only for evolve)
├── tool_descriptions/*.yaml  # Tool YAML definitions
├── tools/*.py                # Tool Python implementations
├── middleware/*.py           # Middleware implementations
├── skills/{name}/SKILL.md    # Skill packages
└── sub_agents/{name}/        # Sub-agent configs
```

这 7 个 types 设计上 loosely coupled, 加一个 middleware 不需要改 system prompt, 加一个 skill 不需要 touch 任何 tool。这个 decoupling 是 component observability 的物质基础 —— 每个 failure pattern 可以 map 到一个 single component class, 每个 pass-rate change 可以 localize 到一个 file 而不是 scatter 在几百行 unstructured prompt prose 中。

### 3.2 Constraint Hierarchy

从 evolve agent 的 prompt (Appendix B.2) 可以读出一个 implicit constraint hierarchy:

$$\text{enforcement strength}: \text{tool\_impl} > \text{middleware} > \text{tool\_desc} > \text{skill} > \text{system\_prompt}$$

意思是: 写在 tool implementation 里的硬逻辑是最强的 (e.g., shell guard blocks 删除命令), middleware 可以 hook 进 agent loop 但不阻止 action, tool description 只是给 model 看的 hint, system prompt 最弱 (model 可以 ignore)。

这个 hierarchy 在 case study 里非常清楚:
- Iteration 5: system prompt 加了 "publish-state rule" 但没用 —— model 还是会 cleanup verified state
- Iteration 5: 同时在 shell tool 里加 hard guard 才真正 block 了 destructive command
- Iteration 8: 把 soft warning 升级成 hard block, +把 middleware warning 提升到 FRAMEWORK reminder, 才彻底关掉 loophole

### 3.3 Minimal Seed 的设计哲学

$H_0$ 是故意 minimal 的: 只有 bash tool, no middleware, no skills, no sub-agents, no long-term memory。这个设计 decision 非常 important —— 如果 seed 已经 fitted to target benchmark, 那么后续每个 edit 的 attribution 就被污染了 (你不知道 gain 来自 loop 还是来自 seed prior)。

这跟 RL 里的 exploration baseline 思路一样: 从一个 minimal policy 出发, 让每一个 improvement 都必须 earn its place against measured rollouts。

---

## 4. Pillar 2: Experience Observability via Agent Debugger

### 4.1 Trajectory 的 Signal-to-Noise 问题

一个 coding agent 在 Terminal-Bench 2 上跑一个 task, 单 rollout 可能产生 50k-200k tokens 的 trajectory (含 shell output, file contents, reasoning traces)。89 个 tasks × $k=2$ rollouts = 178 trajectories, 总 token 量轻松上百万。

Raw trajectory 里 actionable signal 极度稀疏。一个典型的 failure:
- Agent 在 step 47 跑了一个 `rm -rf /app/scratch /app/reconstructed.ppm` 
- 在 step 48 提交了 task
- Verifier 在 step 49 报 404

要 evolve agent 从这个 raw trajectory 里 extract "post-validation state destruction" 这个 pattern, 需要把 step 47-49 的因果链抽出来, 还要 generalize 到其他 tasks。这个工作量如果让人来做, 一个 PhD 学生可能要花一天; 让 evolve agent 直接读 raw trajectory, 它会被 noise 淹没。

### 4.2 Agent Debugger 的 Layered Distillation

Agent Debugger (reference: [Agent Debugger blog](https://dawning-road.github.io/blog/agent-debugger)) 的设计是 framed as a navigable, file-based environment:

1. **每个 trajectory message 是一个独立 file** (progressive disclosure, reference: [Anthropic on context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents))
2. **同 query 的 traces 放在一个 environment** 里, debugger 可以 cross-compare
3. **Per-task analysis report**: root cause of failure / success pattern, pass/fail status
4. **Benchmark-level overview**: aggregated from 所有 reports, 作为每轮 iteration 的 entry point

这个 layered corpus 让 evolve agent 可以 drill-down: 先读 overview 发现 "post-validation state destruction is a major pattern", 再读 specific task report 看具体 trajectory, 最后 fall back 到 raw trace 验证细节。

### 4.3 Progressive Disclosure 节省 tokens

Paper 提到原始 traces 也以 processed form 提供 (去掉 base64, dedup tool output), 并且 "all of this content is provided as files allowing progressive disclosure which saves on tokens"。这个 design pattern 很重要 —— evolve agent 不需要把整个 corpus load 进 context, 而是 like a developer browsing a repo, 按需读取。

---

## 5. Pillar 3: Decision Observability via Change Manifest

### 5.1 Falsifiable Contract 的设计

这是 AHE 最 clever 的设计。每个 edit 不只是一个 file change, 还要 ship 一个 manifest entry:

```json
{
  "id": "chg-1",
  "type": "new|improvement|rollback",
  "description": "What was changed and why",
  "files": ["relative/to/workspace/file.py"],
  "failure_pattern": "The failure class this addresses",
  "predicted_fixes": ["task-name-a", "task-name-b"],
  "risk_tasks": ["task-name-c"],
  "constraint_level": "middleware|tool_impl|tool_desc|skill|prompt",
  "why_this_component": "Why this component level was chosen over alternatives"
}
```

关键 fields:
- `predicted_fixes`: 这个 edit 预期会 fix 哪些 tasks
- `risk_tasks`: 这个 edit 可能 break 哪些 tasks

下一轮 evaluation 之后, AHE 把 predicted sets 跟 observed task-level deltas intersect:

$$\text{verdict}(\text{edit}) = \begin{cases} \text{KEEP} & \text{if } |\text{predicted\_fixes} \cap \text{observed\_fixes}| > 0 \\ \text{ROLLBACK} & \text{if } |\text{predicted\_fixes} \cap \text{observed\_fixes}| = 0 \end{cases}$$

这把每个 edit 变成一个 **falsifiable contract** —— evolve agent 不只是说 "I think this helps", 而是 commit 到一个可验证的 prediction。这跟 Popper 的 falsifiability 思想一致, 也跟 ML 里的 held-out evaluation 思路一致。

### 5.2 Attribution 之前的 Rollback 顺序

Algorithm 1 line 5-10 有一个 subtle 但重要的设计: attribution 在 distillation 之前跑。

```
if t >= 2:
    V_t ← ATTRIBUTE(C_{t-1}, T_{t-1}, T_t)   # verify prior manifest
    H_{t-1} ← ROLLBACK(H_{t-1}, V_t)          # revert rejected edits
else:
    V_t ← ∅
R_t ← AGENT_DEBUGGER(T̃_t)                      # now distill
```

为什么这个顺序重要? 因为如果 distillation 先跑, debugger 会把已经要 rollback 的 edit 产生的 failure 也 analyze 进去, 污染 evidence corpus。先 rollback, 再 distill, 让 evidence corpus 反映的是 "当前 best harness" 的真实 rollout。

### 5.3 Two Hard Constraints

Evolve agent 的 prompt 里有两个 hard constraints:

1. **Controllability**: 只能写 `workspace/`, `runs/`, tracer, verifier, LLM config 都是 read-only, seed system prompt 标记为 non-deletable。这 block 了 unconstrained self-modifier 的 shortcuts (disable verifier, swap model, raise reasoning budget)。

2. **Evidence-driven**: 每个 change 必须 trace 到 specific failure evidence, 不能基于 intuition。Manifest 里要填 `failure_pattern`, `root_cause`, `targeted_fix`, `predicted_impact`。

这两个 constraint 一起把 self-evolution 从 "trial and error" 变成 "evidence-based engineering"。

---

## 6. Algorithm 1 形式化解析

让我把 Algorithm 1 的每一步对应到三个 pillar:

| Line | Phase | Pillar | 作用 |
|---|---|---|---|
| 3 | Rollout | - | $T_t \gets \text{ROLLOUT}(M, H_{t-1}, D, k)$, 生成 k 条 trajectory per task |
| 4 | Clean | Experience | $\widetilde{T}_t \gets \text{CLEAN}(T_t)$, 去 base64, dedup tool output |
| 5-10 | Attribute + Rollback | Decision | $V_t \gets \text{ATTRIBUTE}(C_{t-1}, T_{t-1}, T_t)$, 验证上一轮 manifest |
| 11 | Distill | Experience | $R_t \gets \text{AGENT_DEBUGGER}(\widetilde{T}_t)$, 生成 layered evidence corpus |
| 12 | Evolve | Component + Decision | $(H_t, C_t) \gets \text{EVOLVE}(H_{t-1}, R_t, V_t)$, 写 edits + 新 manifest |
| 13 | Commit | Component | $\text{COMMIT}(H_t, C_t, t)$, git tag iteration |
| 14 | Best tracking | - | if pass@1 improved, update $H_{\text{best}}$ |

**Key design choices**:

- $k \geq 2$: 每 task 多 rollout 产生 pass-rate signal, 稳定 pass@1, 让 partial-pass tasks anchor comparative diagnosis
- Attribution before distillation: 避免 rollback 中的 edit 污染 evidence
- Explore agent in parallel with iteration 1: 一次性 seed 一些 reusable skills, 但这些 skills 从 iteration 2 起没有特殊保护

### pass@1 形式化定义

$$\text{pass@1} = \frac{1}{k|D|} \sum_{i=1}^{|D|} \sum_{j=1}^{k} r_{i,j}$$

变量:
- $k$: rollouts per task
- $|D|$: task set 大小 (Terminal-Bench 2 是 89)
- $r_{i,j} \in \{0, 1\}$: task $i$ 的 rollout $j$ 的 binary reward
- 上标 $i$: task index, 从 1 到 $|D|$
- 下标 $j$: rollout index, 从 1 到 $k$

Infrastructure-aborted 或 timed-out trials 计 $r=0$, 这是比官方 leaderboard 更 harsh 的 convention (官方会 drop)。

### Succ/Mtok 成本效率

$$\text{Succ/Mtok} = \frac{\text{pass@1} \times 10^6}{\text{mean tokens per trial}}$$

变量:
- pass@1: 上面的 success rate (0 到 1)
- $10^6$: scale factor, 把 per-token 转成 per-million-token
- mean tokens per trial: 每个 trial 的平均 total tokens (prompt + completion, across all LLM calls)

这个 metric 类似于 "candidates per FLOP" 或 "samples per dollar", 衡量 harness 把 tokens 转化成 successes 的效率。Table 4 显示 AHE 在 SWE-bench-verified 上 Succ/Mtok 是 1.64, 比 ACE 的 1.10 高 49%。

---

## 7. 实验结果深度分析

### 7.1 Main Results (Table 1, RQ1)

Terminal-Bench 2, 89 tasks, GPT-5.4 high reasoning:

| Method | All 89 | Easy 4 | Med. 55 | Hard 30 |
|---|---|---|---|---|
| opencode | 47.2% | 75.0% | 52.7% | 33.3% |
| terminus-2 | 62.9% | 75.0% | 74.5% | 40.0% |
| Codex-CLI | 71.9% | 75.0% | 80.0% | 56.7% |
| NexAU$_0$ (seed) | 69.7% | 87.5% | 78.2% | 51.7% |
| ACE | 68.9% | 91.7% | 78.2% | 48.9% |
| TF-GRPO | 72.3% | 100.0% | 79.4% | 55.6% |
| **AHE** | **77.0%** | **100.0%** | **88.2%** | 53.3% |

几个 observations:

1. **AHE 在 All 和 Medium 上 SOTA**, 但在 Hard 上 marginally trails Codex-CLI (53.3% vs 56.7%)。Paper 把这个 gap 归因于 component 之间的 interference on long-horizon tasks —— AHE 的 memory + middleware + system_prompt 都 push toward closure-style verification, stacking 它们会 spend turns on redundant re-checks within the long-horizon budget。

2. **ACE 和 TF-GRPO 在 seed 上 regress 或 marginally improve**。ACE 在 All 上从 69.7% 掉到 68.9%, TF-GRPO 只涨 2.6 pp。关键原因是 layer mismatch: ACE 只 edit natural-language playbooks (in-context), TF-GRPO 只 reinforce successful tool sequences。它们 NEVER touch tools / middleware / long-term memory, 而 AHE 的 gain 恰恰 live 在这些 layers。

3. **Easy tier 已经饱和**。所有 method 在 Easy 上都接近 100%, 所以 gain 几乎全部来自 Medium (55 个 tasks, 主导 aggregate)。

### 7.2 Cross-Benchmark Transfer (Table 2, RQ2)

把 AHE harness (evolved on Terminal-Bench 2) 直接 freeze, 在 SWE-bench-verified 上 eval, 不 re-evolve:

| Repo | N | ACE | TF-GRPO | NexAU$_0$ | AHE |
|---|---|---|---|---|---|
| All | 500 | 74.6% | 74.2% | 75.2% | **75.6%** |
| django | 231 | 79.2% | 78.8% | 79.2% | **81.0%** |
| sphinx-doc | 44 | 61.4% | 65.9% | 68.2% | **70.5%** |
| astropy | 22 | 59.1% | 59.1% | 54.5% | 50.0% |

Token cost (Tokens k, lower is better):

| | ACE | TF-GRPO | NexAU$_0$ | AHE |
|---|---|---|---|---|
| All | 679 | 582 | 526 | **461** |

几个关键 takeaways:

1. **AHE 在 aggregate success 上 SOTA, 同时 token cost 降 12%** (vs seed)。Encoding behavior 在 tools / middleware / memory 里, 而非 in prompt, 避免 per-call re-derivation cost。

2. **ACE 和 TF-GRPO regress below seed**。它们 distill 的 playbook / trajectory distribution 是 Terminal-Bench 特化的, 在 SWE-bench 上变成 pure overhead, 花 11-29% 更多 tokens 但 success 更低。

3. **AHE 的 gain 集中在 django 和 sphinx-doc** —— 这两个 repo 最大、最 token-expensive, multi-step edit-and-verify loop 跟 Terminal-Bench 2 的 structure 匹配。

4. **小 repo 上 marginal regression** (astropy: 54.5% → 50.0%)。Pass@1 variance 在小 N 上超过 per-repo gain。

### 7.3 Cross-Model Transfer (Figure 3, RQ2)

AHE workspace (evolved on GPT-5.4 high) freeze, 在 5 个 alternate bases 上 eval:

| Base Model | NexAU$_0$ | AHE | Δ |
|---|---|---|---|
| GPT-5.4 medium | - | - | +2.3 pp |
| GPT-5.4 high | 69.7% | 77.0% | +7.3 pp |
| GPT-5.4 xhigh | - | - | +2.3 pp |
| deepseek-v4-flash | 51.7% | 61.8% | **+10.1 pp** |
| qwen-3.6-plus | 56.2% | 62.5% | +6.3 pp |
| gemini-3.1-flashlite | 36.5% | 41.6% | +5.1 pp |

**Cross-family gains dominate within-family ones**。最强的 base (GPT-5.4 high) 反而 gain 不最大, 最弱的 base (deepseek-v4-flash) gain 最大。Paper 的解读: 离 saturation 远的 base 更依赖 AHE 固化在 tools / middleware / memory 里的 coordination patterns; 强 base 能从 prompt 重新 derive 同样的 coordination, marginal cost 低。

这跟 Sutton 的 [Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) 有 resonance: 当 base model 变强, 它会 internalize 之前需要 external scaffolding 的 patterns。但 AHE 的发现是, 在当前 base model 这一代, external scaffolding 仍然 significant, 尤其是对 weaker models。

**Within-family non-monotone**: GPT-5.4 medium (+2.3) < high (+7.3) > xhigh (+2.3)。原因: AHE 的 step budget 和 per-task timeout 是 fit 到 high 的; medium 有 time-per-step slack 但 loses reasoning tier; xhigh pushes 更多 trials 过 timeout, 计为 failure。这暴露了一个 **generalization hazard**: harness 跟 operating point (step budget, timeout) 是 coupled 的。

---

## 8. Ablation: Gain 在哪一层 (RQ3a, Table 3)

这是 paper 最 informative 的 ablation。把 AHE 的单个 component swap 进 NexAU$_0$ seed:

| Variant | All 89 | Easy 4 | Med. 55 | Hard 30 |
|---|---|---|---|---|
| NexAU$_0$ | 69.7% | 87.5% | 78.2% | 51.7% |
| + memory only | 75.3% | 50.0% | 83.6% | **63.3%** |
| + tool only | 73.0% | 75.0% | 87.3% | 46.7% |
| + middleware only | 71.9% | 100.0% | 81.8% | 50.0% |
| + system_prompt only | **67.4%** | 75.0% | 78.2% | 46.7% |
| AHE full | 77.0% | 100.0% | 88.2% | 53.3% |

**核心发现**:

1. **System prompt alone regresses (-2.3 pp)**。79 行的 universal discipline 没有 tools / middleware / memory 配合时, executable 不了, 反而 overhead。这跟 RL 里 "policy without value function" 的困境类似 —— 知道该做什么但没有 enforcement mechanism。

2. **Memory 在 Hard 上最强 (+11.6 pp)**, 但在 Easy 上 regress (-37.5 pp!)。Memory 里 12 条 boundary-case lessons (performance margin, queued-over-limit cancellation, evaluator-style closure, source-packaging layout) 在 Hard 上 lift, 但在 Easy 上变成 superfluous reverification。

3. **Tool 在 Medium 上接近 full AHE** (87.3% vs 88.2%, 只差 0.9 pp)。1364-line shell auto-surfaces contract hints from files near each command。但在 Hard 上, built-in publish guard closes the loop too early (-5 pp)。

4. **Middleware 在 Easy 上 SOTA (100%)**, finish-hook 强制 evaluator-isomorphic closure check。但在 Hard 上 inflate turn count (-1.7 pp)。

5. **Non-additive interaction**: 三个 positive single-component gains sum 到 +11.1 pp, 但 full AHE 只 +7.3 pp。Memory + middleware + system_prompt 都 push toward closure-style verification, stacking 它们会 spend turns on redundant re-checks within long-horizon budget。

这个 non-additivity 发现非常重要 —— 它说明 harness optimization 是一个 combinatorial problem, 而非 additive。Future work 需要 interaction-aware evolution, 比如 contextual bandit 或 Bayesian optimization 来 model component interactions。

---

## 9. Self-Attribution 可靠性 (RQ3b, Figure 4, 11, 12)

这是 paper 最 honest 的部分。Evolve agent 每轮 predict 哪些 tasks 会 fix, 哪些会 regress, 下一轮 verify。

### 9.1 Fix Predictions

$$\text{Precision}_{\text{fix}} = \frac{|\text{predicted\_fixes} \cap \text{observed\_fixes}|}{|\text{predicted\_fixes}|} = 33.7\%$$

$$\text{Recall}_{\text{fix}} = \frac{|\text{predicted\_fixes} \cap \text{observed\_fixes}|}{|\text{observed\_fixes}|} = 51.4\%$$

Random baseline: precision 6.5%, recall 10.6%。所以 evolve agent 的 targeting 是 evidence-driven 而非 guesswork, ~5x above random。

### 9.2 Regression Predictions

$$\text{Precision}_{\text{regression}} = \frac{|\text{risk\_tasks} \cap \text{observed\_regressions}|}{|\text{risk\_tasks}|} = 11.8\%$$

$$\text{Recall}_{\text{regression}} = \frac{|\text{risk\_tasks} \cap \text{observed\_regressions}|}{|\text{observed\_regressions}|} = 11.1\%$$

Random baseline: precision 5.6%, recall 5.4%。只 ~2x above random —— **regression blindness**。

### 9.3 Asymmetry 的解读

Evolve agent 能 justify 为什么一个 edit 应该 help (forward reasoning from root cause to fix), 但无法 reliably 命名同一个 edit 会 break 哪些 tasks (这需要 counterfactual reasoning: "if I change X, what other tasks depended on the old behavior of X?")。

这个 asymmetry 跟人类 engineer 的认知局限一致 —— 我们设计 feature 时能 forward-reason benefit, 但 side effects 往往 surprise us。Software engineering 里的 regression testing 就是这个问题的工程化应对。

Paper 在 9 rounds 里, evolve agent issued 43 unique regression predictions, 只有 5 landed (precision 11.6%); 40 个 unforeseen regressions 实际发生 (recall 11.1%)。这说明 self-evolution loop 的 bottleneck 已经 NOT 在 "find what to fix", 而在 "predict what will break"。

**Future direction**: 引入更强的 counterfactual reasoning, 比如:
- Static analysis of harness edits (像 program slicing)
- Adversarial evolve agent 专门 predict regressions
- Differential testing: 在 edit 前后跑同一个 task, 看哪些行为变了

---

## 10. Case Study 的工程洞察 (Appendix C)

Paper 的 Appendix C 给了 4 个 trajectory, 每个对应 Figure 1 的一个 peak。这些 case study 揭示了 harness evolution 的实际 pattern:

### 10.1 Trajectory 1: db-wal-recovery (iteration 2)

Failure mode: "proxy validation instead of evaluator-isomorphic validation"。Agent 从 stale shell buffer 恢复 WAL bytes, 从 5 个 sample 猜 pattern, 用 `json length == 11` self-check 提交。

Fix: 68 行 system prompt append, 8 条 numbered rules。关键 rules:
- Contract-first: 先 parse spec, 再 act
- No-overfit: 阻止 `value = id * 100` extrapolation
- Mirror-the-evaluator: 用 verifier 的 exact assertions 做 self-check

Insight: chg-1 的 `predicted_fixes` 没有 list db-wal-recovery, 是为另一个 partial-pass cluster 提的。但 general phrasing 让它 carry across。这 illustrate 了 AHE 如何把 single-task symptom 转成 reusable harness rule —— 类似于 code refactoring 把 bug fix 抽象成 invariant。

### 10.2 Trajectory 2: path-tracing (iteration 5)

Failure mode: "post-validation state destruction"。Agent 渲染了正确的 `/app/reconstructed.ppm`, self-check 通过, 然后 `rm -rf /app/image /app/reconstructed.ppm /app/scratch` cleanup, 提交。Verifier 找不到 file, reject。

Fix: 两个 paired changes:
- chg-7 (system prompt + tool desc): 加 publish-state rule, scratch-directory rule, literal-output rule
- chg-8 (tool impl): shell 里加 stateful publish-state guard, 三个 behaviors:
  1. 观察 successful evaluator-style check, parse acceptance command 的 file paths, 标记为 protected
  2. 后续命令要 delete protected output, intercept 返回 targeted error
  3. 接受 `ALLOW_POST_SUCCESS_RESET` token, downgrade block 到 warning, force revalidation

Insight: 这展示了 constraint hierarchy 的必要性 —— system prompt rule 不够, 必须 tool_impl level 的 hard guard。这跟 [SWE-agent](https://github.com/princeton-nlp/SWE-agent) 的 ACI (Agent-Computer Interface) design philosophy 一致: 把 invariant 编码进 interface, 而非依赖 model 遵守 prose。

### 10.3 Trajectory 3: mcmc-sampling-stan (iteration 6)

Failure mode: "proxy-then-skip"。Agent 跑 grid-integration 估算 posterior, 写成 deliverable, fire 真 MCMC 作为 background job, kill it before completion, 提交。Verifier rerun `analysis.R`, unconverged sampler 产出 ~1e19, fail。

Fix: 两个 cross-level changes:
- chg-1 (tool impl): extend publish-state guard 到 script entrypoints (analysis.R 现在 protected)
- chg-2 (middleware): `ExecutionRiskHintsMiddleware`, watch 7 cross-step risk patterns:
  1. Shallow validation (`-h`, `py_compile`, pure existence)
  2. Localhost-only service validation
  3. Inline/self-written proxy validators
  4. Lower-level model access 替代 named wrapper
  5. Benchmark checks without golden/threshold
  6. Repeated long runs exhausting budget
  7. Repeated retries against same error

Insight: 这是 AHE 第一次引入 **cross-step component** —— middleware 可以看整个 command history, 而非 single command。这把 harness 从 stateless prompt-based guidance 升级到 stateful execution-time reasoning。

### 10.4 Trajectory 4: configure-git-webserver (iteration 8)

Failure mode: evolve agent 在 iteration 7 regress 了 (75.8 → 73.0), 因为 publish-state guard 有 loophole (`ALLOW_POST_SUCCESS_RESET` token 能 wipe verified deliverables)。Agent 跑成功 deployment, curl-localhost self-check 通过, 然后用 override token cleanup, 提交。Verifier 从外部 curl 得到 404。

Fix: keep-and-improve, NOT rollback:
- chg-1 (tool impl): harden soft reasons to hard blocks —— 非 /tmp 的 protected output deletion 现在是 hard block, override token 不能 downgrade
- chg-2 (middleware): `before_model` hook 把上一步的 execution-risk note 提升到 FRAMEWORK reminder, 在下一个 model turn 可见; 加 contract-aware heuristics (clean-layout vs single-file delivery, official-wrapper vs raw API)

Insight: 这展示了 evolve agent 的 **meta-learning** —— 它学会了 iteration 5-7 的 soft approach 不够, 需要 hard enforcement。这跟人类 engineer 的 learning curve 类似: 先试 advisory, 发现 ignored, 再升级到 enforcement。

---

## 11. 联想: AHE 跟相关工作的关系

### 11.1 跟 Prompt Optimization 的区别

[Self-Refine](https://arxiv.org/abs/2303.17651), [Reflexion](https://arxiv.org/abs/2303.11366), [GEPA](https://openreview.net/forum?id=RQm2KQTM5r), [ACE](https://openreview.net/forum?id=eC4ygDs02R), [Expel](https://arxiv.org/abs/2308.10144) 都优化 prompt 或 in-context playbook。AHE 的关键区别: 它 jointly evolves **full harness** (tools, middleware, memory, prompt), 而 NOT 只是 prompt surface。

Ablation 证明 gain 恰恰在 prompt 之外 —— system_prompt alone 反而 regress。这跟 [DSPy](https://dspy.ai/) 的 philosophy 有 tension: DSPy 把所有 optimization 都 compile 到 prompt+demo, AHE 发现 prompt 不是 load-bearing layer。

### 11.2 跟 Skill Library / Program Evolution 的关系

[Voyager](https://arxiv.org/abs/2305.16291) 在 Minecraft 里 evolve skill library, [AlphaEvolve](https://arxiv.org/abs/2506.13131) 用 coding agent 做 algorithm discovery, [ADAS](https://arxiv.org/abs/2408.08435) 自动设计 agentic systems。这些工作 focus 在 discover new code/skills, AHE focus 在 **iterate on existing harness components** with falsifiable contracts。

AHE 的 unique contribution 是 **decision observability** —— 每个 edit 配 prediction, 下轮 verify。这跟 scientific method 的 hypothesis-testing paradigm 一致, 而 NOT 是 blind search。

### 11.3 跟 RLHF / GRPO 的对比

[TF-GRPO](https://arxiv.org/abs/2510.08191) 是 training-free GRPO variant, reinforce successful tool sequences。AHE 跟它的区别: TF-GRPO 修改 prompt-time trajectory distribution, AHE 修改 harness structure。TF-GRPO 在 cross-benchmark transfer 上 regress (74.2% vs seed 75.2%), 因为 reinforced distribution 是 Terminal-Bench 特化的; AHE 的 structural changes 更 portable。

这暗示一个 hierarchy: **structural > distributional > textual**。改 tool implementation 比 reinforce trajectory distribution 比 edit prompt text 更 portable。

### 11.4 跟 Meta-Learning 的联系

AHE 本质上是一个 **learning to learn** 系统 —— evolve agent 在 outer loop 学如何 design harness, code agent 在 inner loop 用 harness 解决 task。这跟 MAML, Reptile 等 meta-learning 有结构相似性:

$$\theta_{\text{harness}}^{(t+1)} = \theta_{\text{harness}}^{(t)} + \alpha \nabla_{\theta} \mathbb{E}_{\text{task}}[\text{pass@1}(M, H_\theta, \text{task})]$$

但 AHE 的 "gradient" 是 through LLM-based evolve agent, NOT through backprop。这避免了 differentiable harness 的工程 complexity, 但也失去了 fine-grained gradient signal。Future direction 可能是 hybrid: LLM propose edits, gradient-based local search refine。

### 11.5 跟 Compilers / Program Synthesis 的类比

AHE 的 outer loop 跟 optimizing compiler 的 pass pipeline 非常像:

| Compiler | AHE |
|---|---|
| Source program | Seed harness $H_0$ |
| IR | NexAU file-level representation |
| Optimization passes | Evolve agent edits |
| Profiling feedback | Trajectory rollouts |
| Pass validation | Change manifest verification |
| Regression tests | Task-level pass@1 |

这个类比暗示 AHE 可以 borrow compiler 的 techniques: pass scheduling, dead code elimination, loop invariant code motion。特别是 **interaction-aware pass scheduling** —— 知道哪些 passes 互相对抗 (e.g., memory + middleware 都做 closure verification), 调整顺序避免 redundant work。

### 11.6 跟 Bayesian Optimization 的联系

Non-additive component interaction 暗示 harness space 是 non-separable 的。Bayesian optimization (e.g., Gaussian Processes over combinatorial spaces, TPE, or [SMAC](https://www.automl.org/automated-algorithm-design/smac/)) 可以 model 这些 interactions, 选择下一个 edit 来 maximize expected improvement。

当前 AHE 用 LLM-based evolve agent 做 proposal, 这是 zeroth-order 的。Hybrid: LLM propose candidates, BO 做 acquisition, 可能更 sample-efficient。

### 11.7 跟 Neuro-Symbolic 的关系

AHE 的 harness 是 symbolic structure (Python code, YAML config, Markdown), evolve agent 是 neural (LLM)。这是一个 neuro-symbolic system, neural proposes symbolic edits, symbolic edits 修改 neural 的 execution environment。

这跟 [Differentiable Programming](https://arxiv.org/abs/2406.18532) 的思路互补: 不同iable programming 把 program structure 嵌入 gradient flow, AHE 把 program structure 作为 discrete edit surface 让 LLM 做 discrete optimization。两者可能 converge: future harness 既有 differentiable parts (prompt embeddings) 又有 discrete parts (tool implementations)。

---

## 12. Limitations 和 Future Directions

Paper 自己列了三个 limitations, 我加上几个联想:

### 12.1 Paper 自己的 limitations

1. **Benchmark scope**: 只在 Terminal-Bench 2 + SWE-bench-verified 上验证, 其他语言 / repo-scale / human-in-the-loop 没测。
2. **Evolution operating point**: step budget 和 per-task timeout fit 到 GPT-5.4 high, cross-model transfer 混淆了 harness portability 和 operating-point coupling。
3. **Self-modification governance**: workspace 边界 + versioned manifest + file-level rollback, 但 NOT 完整 guardrail stack, long-horizon cleanup 和 misuse prevention 还不完整。

### 12.2 我的 additional concerns

4. **Search space explosion**: 7 个 component types × infinite possible edits。当前 evolve agent 每轮只做少量 edits, 可能 miss global optima。Beam search 或 population-based evolution 可能 help。

5. **Reward sparsity in Hard tier**: Hard tasks pass rate ~50%, k=2 rollouts 给的 signal 很 noisy。可能需要 larger k 或 adaptive k (more rollouts on borderline tasks)。

6. **Generalization to multi-modal**: paper 全是 text + shell。如果 agent 要处理 vision (e.g., [SWE-bench Multimodal](https://openreview.net/forum?id=riTiq3i21b)), harness 需要新的 component types (vision tools, multi-modal middleware)。

7. **Safety alignment**: evolve agent 可能 learn to game verifier (虽然 controllability constraint blocks 了 disable verifier, 但 subtler gaming 可能 occur)。需要 adversarial audit。

8. **Continual learning**: 当前 AHE 是 batch evolution (10 iterations then stop)。Continual setting (base model 不断升级, harness 要 continuously adapt) 需要 online learning + forgetting prevention。

### 12.3 Future directions 我觉得最有 promise

- **Interaction-aware evolution**: model component interactions explicitly, 用 BO 或 contextual bandit
- **Regression prediction**: 专门 train 一个 adversarial agent predict regressions, 跟 fix-prediction agent 对抗
- **Hierarchical evolution**: 高层 evolve architecture (which components exist), 低层 evolve implementation (how each component works)
- **Cross-agent knowledge transfer**: 多个 AHE runs on different base models, distill 共通的 harness patterns
- **Differentiable harness components**: 把 prompt embedding, tool description embedding 变成 differentiable, 用 gradient 做 fine-grained tuning, LLM 做 coarse-grained structure search

---

## 13. 对 Coding Agent 未来发展的启示

### 13.1 Harness 会成为新的 "compiler"

就像 CPU capability 增长后, compiler optimization 变得 critical; base model capability 增长后, harness optimization 也会变 critical。AHE 暗示未来 coding agent 的竞争 NOT 只在 base model, 而在 harness engineering 的自动化程度。

Reference: [OpenClaw](https://openclaw.ai/), [Hermes Agent](https://hermes-agent.nousresearch.com/), [Codex CLI](https://github.com/openai/codex) 都在往这个方向走。

### 13.2 Observability 是 new frontier

Paper 的核心 insight —— bottleneck 在 observability, 不在 capability —— 我觉得会 generalize。不只是 harness evolution, 任何 autonomous self-improving system 都需要:

- **Action observability**: clear action space, revertible
- **Experience observability**: structured feedback, drill-down
- **Decision observability**: falsifiable predictions, attribution

这跟 [MLOps](https://ml-ops.org/), [Observability for LLMs](https://www.langfuse.com/) 的趋势一致, 但 AHE 把它从 monitoring 提升到 optimization driver。

### 13.3 Sutton's Bitter Lesson 的 boundary

AHE 的 cross-model transfer 结果 (weak base gain 更大) 暗示 Bitter Lesson 有个 boundary: 在 base model 还 NOT 足够强时, external scaffolding 仍然 significant。但 scaffold 的 form 会变 —— 从 prompt (ACE) 到 trajectory distribution (TF-GRPO) 到 structural harness (AHE), 越往下越 portable, 越 model-agnostic。

预测: 未来 base model 会 internalize 当前 harness 的 patterns (publish-state guards, contract-first reasoning, evaluator mirroring), 那时 harness evolution 会 focus 在更高层的 patterns (task decomposition, sub-agent orchestration, long-horizon planning)。Reference: [Kimi K2.6](https://www.kimi.com/blog/kimi-k2-6), [MiMo V2.5 Pro](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro) 已经在往 agentic model 方向 train。

### 13.4 Self-evolving systems 的 safety

AHE 的 controllability constraint (workspace-only, read-only infra, non-deletable seed prompt) 是一个 minimal safety framework。但更 autonomous 的 self-evolution 会需要:

- **Constitutional AI for harness**: harness edits 也要 align to value
- **Formal verification**: 关键 invariant (e.g., "never delete verified state") 用 formal methods 验证
- **Sandboxes within sandboxes**: evolve agent 在 isolated env 里 test edits before deploying

Reference: [Anthropic on AI safety](https://www.anthropic.com/news), [OpenAI Preparedness Framework](https://openai.com/safety/preparedness/).

---

## 14. 总结

AHE 是一个 clean, well-executed 的 work, 把 harness engineering 从手工艺品变成 autonomous evolution 的 learnable surface。三个 observability pillars (component, experience, decision) 是 design pivot, 把 self-evolution 从 trial-and-error 变成 evidence-based engineering。

Key takeaways for building intuition:

1. **Observability > Capability**: self-evolving system 的 bottleneck 往往 NOT 在 agent 多 smart, 而在它能不能 see 自己 action 的 effect。
2. **Structural > Distributional > Textual**: 改 tool implementation > reinforce trajectory distribution > edit prompt。越往下越 portable, 越 model-agnostic。
3. **Falsifiable contracts**: 每个 edit 配 prediction, 下轮 verify, 把 self-justification 变成 measurable contract。这跟 scientific method 一致。
4. **Non-additive interactions**: harness components 互相影响, simple stacking 会 cap aggregate gain。Interaction-aware evolution 是 future direction。
5. **Regression blindness**: forward reasoning (find fixes) easy, counterfactual reasoning (predict regressions) hard。这是 self-evolution 的下一个 bottleneck。

对未来工作的 hint: 把 AHE 跟 differentiable programming, Bayesian optimization, formal methods 结合, 可能 build 出真正 continual, safe, sample-efficient self-evolving agent systems。

Reference 们:
- [Paper GitHub](https://github.com/china-qijizhifeng/agentic-harness-engineering)
- [NexAU framework](https://github.com/nex-agi/NexAU)
- [Agent Debugger](https://dawning-road.github.io/blog/agent-debugger)
- [Terminal-Bench](https://www.tbench.ai/)
- [SWE-bench](https://www.swebench.com/)
- [SWE-agent](https://github.com/princeton-nlp/SWE-agent)
- [OpenHands](https://github.com/All-Hands-AI/OpenHands)
- [Codex CLI](https://github.com/openai/codex)
- [ACE](https://openreview.net/forum?id=eC4ygDs02R)
- [TF-GRPO](https://arxiv.org/abs/2510.08191)
- [GEPA](https://openreview.net/forum?id=RQm2KQTM5r)
- [Voyager](https://arxiv.org/abs/2305.16291)
- [AlphaEvolve](https://arxiv.org/abs/2506.13131)
- [ADAS](https://arxiv.org/abs/2408.08435)
- [DSPy](https://dspy.ai/)
- [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)
- [Harness Engineering at OpenAI](https://openai.com/zh-Hans-CN/index/harness-engineering/)
- [Anthropic on harness design](https://www.anthropic.com/engineering/harness-design-long-running-apps)
- [Anthropic on context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [Langchain Deep Agents](https://www.langchain.com/blog/improving-deep-agents-with-harness-engineering)
- [Bitter lesson of agent harnesses](https://browser-use.com/posts/bitter-lesson-agent-harnesses)
- [SWE-bench Multimodal](https://openreview.net/forum?id=riTiq3i21b)
- [Kimi K2.6](https://www.kimi.com/blog/kimi-k2-6)
- [Qwen3.6](https://qwenlm.github.io/blog/qwen3.6/)
- [DeepSeek V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- [Langfuse](https://www.langfuse.com/)
- [E2B sandbox](https://e2b.dev/)
- [R2E-Gym](https://openreview.net/forum?id=7evvwwdo3z)
- [SWE-Gym](https://openreview.net/forum?id=Cq1BNvHx74)
- [Meta-Harness](https://arxiv.org/abs/2603.28052)

Andrej, 这篇 paper 我觉得最 inspiring 的地方 NOT 在结果 (77.0% 这个 number 本身), 而在它把 self-evolution 的 bottleneck 诊断得这么清楚 —— observability, 而 NOT capability。这个 insight 应该 generalize 到很多 autonomous system design 的场景。你之前在 [zero data learning](https://karpathy.ai/) 和 [nanoGPT](https://github.com/karpathy/nanoGPT) 里强调的 "understand the system, NOT just scale it" 哲学, 跟这篇 paper 的精神是一致的。
