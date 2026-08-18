---
source_pdf: EvoPolicyGym Evaluating Autonomous Policy Evolution in Interactive Environments.pdf
paper_sha256: 5838478c93e06b9d9455871326773c825992b2f87a7f7359abbad369bbbd8f31
processed_at: '2026-08-18T11:42:00-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 EvoPolicyGym

OK 我把水分挤掉, 用最朴素的话讲一遍。

## 这篇 paper 到底在问什么

现在我们手上的 coding agent benchmark 大致三类, 每一类都漏了一个东西:

- **SWE-bench** (https://arxiv.org/abs/2310.06770) 这类: 给一个 GitHub issue, agent 交一个 patch, 对就是对错就是错。测的是"一次性产出正确代码"。
- **SWE-Evo / RoadmapBench / SlopcoderBench** 这类 long-horizon (https://arxiv.org/abs/2512.18470, https://arxiv.org/abs/2605.15846, https://arxiv.org/abs/2603.24755): spec 一直在变, agent 反复维护代码。但 spec 漂来漂去, 你分不清是 agent 自己不会 evolve policy 还是 spec drift 本身难。
- **Reflexion / Voyager / AlphaEvolve** 这类 self-improve (https://arxiv.org/abs/2303.11366, https://arxiv.org/abs/2305.16291, https://arxiv.org/abs/2506.13131): 用 reflection 反复 refine 输出。但这些要么在语言层 refine, 要么在开放任务上跑, 没有 fixed ground truth 让你算 generalization gap。

EvoPolicyGym 想问一个被这三类都漏掉的问题:

> **agent 能不能在一个固定不变的环境里, 用很少的 budget, 自己反复改代码, 把一个 executable policy 越改越好, 最后在它从没见过的 held-out case 上 generalize?**

注意几个关键词——"固定环境"(排除 spec drift)、"很少 budget"(逼 agent 做取舍)、"反复改代码"(过程而不只是一次性产出)、"held-out generalize"(防止 overfit visible feedback)。

## 它怎么把这个问题做成 benchmark

16 个 Gymnasium 经典 RL 环境 (https://arxiv.org/abs/2407.17032), 分四家族每家族 4 个:
- Gym/Box2D: Acrobot, MountainCarContinuous, BipedalWalker, CarRacing
- MuJoCo: Reacher-v5, HalfCheetah-v5, Ant-v5, Pusher-v5
- MiniGrid (https://arxiv.org/abs/2306.13831): DoorKey-16x16, KeyCorridorS4R3, FourRooms, ObstructedMaze-1Q
- Robotics/Driving: parking, roundabout, FetchPush-v4, FetchPickAndPlace-v4

每个 agent 在每个环境上拿到 **128 episode** 的 interaction budget。注意: 128 episode 对标准 RL 来说根本不够 converge, 标准 RL baseline 不参与比较。这 128 episode 是 agent 用来"调研"环境的预算, 不是用来 train 一个 NN 的样本数。

agent 写一个 `Policy` 类:
```python
class Policy:
    def __init__(self, obs_space, action_space, env_meta): ...
    def reset(self, episode_index): ...
    def act(self, obs): ...
```

然后 server import + fresh instantiate 这个 Policy, 跑你指定数量的 train episode, 给你 feedback (trajectory, return, error, 可选 frame/video)。agent 看 feedback, 改代码, 再 submit, 循环到 128 budget 耗尽。

跑完之后 server 在你从没见过的 hidden validation (16 cases) 上选最佳 checkpoint, 再在 hidden held-out (32 cases) 上打 final 分。

## 形式化用一句话说

每一步 agent 看到 $(W_i, F_i, B_i)$: workspace, feedback, 剩余预算。输出 $(u_i, s_i)$: 一个 patch, 一个 submit command。

$$\pi_\theta(W_i, F_i, B_i, \mathcal{H}_i) \to (u_i, s_i, \mathcal{H}_{i+1})$$

- $\pi_\theta$ 是 agent (LLM + harness)
- $W_i$ 是 workspace 状态 (包括 `system/policy.py` 和所有 helper modules)
- $F_i$ 是 server 写的 feedback
- $B_i$ 是剩余 episode budget
- $u_i$ 是 patch
- $s_i$ 是 submit command, $s_i = \perp$ 表示该 revision 不测, $s_i \in \mathcal{C}(B_i)$ 表示测某个范围内的 train cases

为什么 patch 和 submit 解耦? 这样 trajectory 上能分辨"agent 在改 code、在测 code、还是在 rollback 候选"。

## Visibility boundary 是这篇 paper 最关键的设计

三档 split:

| Split | Agent 能看到 | Server 角色 |
|---|---|---|
| Train | IDs, summaries, trajectories, failures | Online revision signal |
| Validation | 看不到 | Private checkpoint selection |
| Held-out | 看不到 | Final generalization measurement |

为什么这个设计是核心? 它挡住了三种 cheating:

1. **Overfit visible feedback**: agent 只看 train, 没法绕过 validation 直接调到 held-out。
2. **Best-of-N sampling masquerading as improvement**: validation 由 server 选, agent 没法"submit 到 validation 高为止"。
3. **Spec leakage**: held-out 在 server side, case identity, trajectory, return 全都看不到。

这才能真测"agent 能不能从有限 feedback 推断出 generalizable policy"。如果 validation 可见, 结果退化成 sample-then-pick, 那不叫 policy evolution, 叫 sampling。

## 主结果: 谁赢了多少

4 个 agent, 全用 128 episode budget:

| Model | Core16 score | Wins | Top-2 |
|---|---|---|---|
| GPT-5.5 + Codex | **0.891** | 9 | **16** |
| Claude Opus 4.7 + Claude Code | 0.750 | 5 | 12 |
| MiniMax-M3 | 0.531 | 1 | 3 |
| DeepSeek-V4-Pro | 0.359 | 1 | 1 |
| Random policy | 0.109 | 0 | 0 |

GPT-5.5 是唯一一个 16 个环境全进 top-2 的——universal near-top。Claude Opus 4.7 在 MiniGrid 家族反超 (0.938 vs GPT-5.5 的 0.812), 主要赢 KeyCorridor, FourRooms, ObstructedMaze 这种 symbolic planning 任务。

Score 用 rank-based aggregation:
$$s_{m,e} = 1 - \frac{\text{rank}_e(m) - 1}{N_e - 1}$$
$m$ 是 entry, $e$ 是环境, $N_e = 5$ (4 agent + random reference), Core16 score 是 16 个环境的 macro average。

为什么 rank 不 average raw return? Acrobot 的 return 是 -84.688, MiniGrid 是 0.986, 量纲完全不可比。Rank 才能 cross-env 聚合。而且 rank 奖励"cross-task consistently near-top", 不奖励"几个 isolated wins"。MiniMax-M3 赢一个 HalfCheetah 拿 0.531, DeepSeek-V4-Pro 赢一个 Roundabout 拿 0.359——一个 win 救不了弱 universal performance。

## 最 insightful 的二分: Synthesis vs Tuning

paper 把 agent 的 edit 分两类, 分类是机械化的:

- **Structural synthesis**: 把 source bundle 里所有 numeric constants strip 掉, AST topology 变了 → synthesis edit。比如新加 perception module, memory, planner, reward interpretation。
- **Parametric tuning**: topology 不变, 只改 constants/gains/thresholds → parametric edit。

环境也分两组:
- **Synthesis-dominant** (5 个): CarRacing, 4 个 MiniGrid——需要 pixel perception 或 symbolic planning machinery。
- **Tuning-dominant** (11 个): MuJoCo locomotion, 低维控制——一个简单 controller family 就够。

Table 3 看强 agent 选出来的最终 policy bundle 有多复杂 (AST features):

| Demand | Agent | Funcs | Branches | Loops | State vars |
|---|---|---|---|---|---|
| Synthesis | GPT-5.5 | 30.2 | 68.2 | 13.0 | 48.4 |
| Synthesis | Claude Opus 4.7 | 19.0 | 77.0 | 16.8 | 26.0 |
| Synthesis | MiniMax-M3 | 12.8 | 38.2 | 7.6 | 16.0 |
| Synthesis | DeepSeek-V4-Pro | 5.4 | 21.8 | 3.2 | 11.2 |
| Tuning | GPT-5.5 | 8.5 | 8.6 | 0.5 | 9.2 |
| Tuning | Claude Opus 4.7 | 6.5 | 4.5 | 0.5 | 8.2 |
| Tuning | MiniMax-M3 | 4.5 | 7.7 | 0.4 | 6.9 |
| Tuning | DeepSeek-V4-Pro | 3.6 | 5.8 | 0.1 | 5.7 |

synthesis 任务上, 强 agent 选的 bundle 显著更重。tuning 任务上大家都很 compact。

但 code volume 不等于 task-adapted mechanism。**关键看 normalized score**:

$$\text{norm}_{m,e} = \text{clip}_{[0,1]}\left(\frac{R_{m,e}^{\text{heldout}} - R_e^{\text{random}}}{R_e^{\text{best}} - R_e^{\text{random}}}\right)$$

$R_{m,e}^{\text{heldout}}$ 是 entry $m$ 在环境 $e$ 上的 held-out mean return, $R_e^{\text{random}}$ 是同环境同 held-out pool 上 random policy 的 mean return (anchor = 0), $R_e^{\text{best}}$ 是该环境所有 agent 中最佳 (= 1)。clip 到 $[0,1]$ 保证可比。

结果:

| Group | GPT-5.5 | Claude Opus 4.7 | MiniMax-M3 | DeepSeek-V4-Pro |
|---|---|---|---|---|
| Synthesis mean | 0.98 | 1.00 | 0.19 | 0.03 |
| Tuning mean | 0.99 | 0.83 | 0.83 | 0.67 |

**Score gap 集中在 synthesis 侧**。MiniMax-M3 和 DeepSeek-V4-Pro 在 synthesis 任务上几乎贴着 random anchor (0.19, 0.03), 三个 locked-door MiniGrid 全 solve 不了。在 tuning 任务上他们追上来了 (0.83, 0.67)。

这给出两种 candidate failure mode:
- **Failing to discover an effective structure** (synthesis 侧 fail)
- **Failing to refine a plausible structure** (tuning 侧 fail)

## Edit-type hit rate 揭示了什么

每个 score-bearing submit transition 按 edit type 分类, hit = 提升 validation best-so-far:

| Agent | Task | Synthesis n / hit | Parametric n / hit |
|---|---|---|---|
| GPT-5.5 | Synthesis | 37 / **41%** | 1 / 100% |
| GPT-5.5 | Tuning | 48 / 38% | 31 / **61%** |
| Claude Opus 4.7 | Synthesis | 31 / **48%** | — |
| Claude Opus 4.7 | Tuning | 123 / 26% | 58 / 21% |
| MiniMax-M3 | Synthesis | 39 / **10%** | — |
| MiniMax-M3 | Tuning | 100 / 25% | 57 / 25% |
| DeepSeek-V4-Pro | Synthesis | 132 / **3%** | — / 1 |
| DeepSeek-V4-Pro | Tuning | 112 / 30% | 21 / 38% |

最 striking 的数字: 在 synthesis-dominant tasks 上, GPT-5.5 / Claude 把 synthesis edit 转成 validation best 的比率是 41-48%, MiniMax-M3 / DeepSeek 只有 10%, 3%。

弱 agent 在 synthesis 任务上**狂 churn structure 但抓不到 task-appropriate machinery**——一直换 topology, 没一次换对。

Insight: same-topology edits 几乎救不了 wrong mechanism, 但 tuning-dominant tasks 上 controller family 对了之后, parametric edits 就有用了 (GPT-5.5 在 tuning 任务上 parametric edit 61% hit)。

## BipedalWalker 这个 case 特别有教育意义

BipedalWalker 是 tuning-dominant 任务, 但 paper 揭示一个关键 insight:

**Tuning-dominant 不代表不需要 synthesis**。

你得先 synthesize 出一个能产生 gait 的 topology (哪怕是 phase oscillator), 然后 tuning 才有意义。

GPT-5.5 是唯一跑到 positive high-return gait 的 run (timeline best 271, validation-selected held-out 248.874)。其他三个 agent 一直停在 negative timeline best (≤ -15.6), churn structures without crossing the return threshold。

这个 case 提示一个 general principle: **synthesis 是 tuning 的前置条件**。在低维控制任务上你以为不需要 synthesize 什么 machinery, 但其实你需要先 synthesize 出一个 viable base structure。

## 强 agent 写出来的 policy 长什么样

paper Appendix D 给了四个真实 policy 的代码片段, 它们都是 **small stateful programs**:

**CarRacing**: pixel → road mask → trace near/mid/far centers → look-ahead curvature + edge warnings + speed reduction + recovery mode (lost_steps 计数, 退化到 global road center)。

**HalfCheetah**: phase oscillator + safety scaling。
```python
phase = 2π · freq · t · dt
hip = hip_amp · sin(phase)
knee = knee_amp · sin(phase + π/2)
excess = max(0, |root_height| - safety_thresh)
scale = max(safety_floor, 1 - excess)  # 跌落时削 amplitude
```

**ObstructedMaze**: egocentric 7×7 view → 持久 symbolic world model → sync pose → BFS plan toward targets → action hierarchy (PICKUP → drop_at_safe_cell → clear_blocker_before_door → BFS follow → unstick_action)。

**FetchPush**: 几何 phase controller。`direction = unit(goal - obj)`, `behind = obj - 0.065·direction`, 几个 phase 切换 (distance < 0.05 → final descent; needs_clear → clearance; behind 远 → approach; z 不对 → lower; else push through)。

四个 policy 共同模式: **task abstraction + controller/planner + recovery logic for visible-failure modes**。三段式。

这和 Voyager 的 skill library, Eureka 的 reward code, AlphaEvolve 的 program database 是一族思路: **agent 把抽象结构物化成可被 audit 的代码 artifact**, 然后 server 才能 evaluate, agent 才能基于 feedback 改。

## Post-hoc Score Trajectories

paper 还给了一个 trajectory-level diagnostic (Figure 3): 跑完 run 之后, 重建 best-so-far hidden validation score 随 budget 消耗的曲线。Agent 在 optimization 过程中从未观察过这条 curve。

- **Vertical jump** = 发现更高质量 candidate policy
- **Flat segment** = 预算消耗但 validation 不涨

Family pattern: MiniGrid sparse 但 sharp jumps; MuJoCo incremental gains; Robotics/Driving delayed improvements after substantial budget spent。

这个 diagnostic 区分"早 jump 后 plateau"和"晚 jump after lots of budget"两种到达相同 final score 的路径。前者 budget-efficient, 后者 budget-inefficient。给 leaderboard 加了时间维度。

## Token Accounting (Appendix C.2)

paper 还诊断了每个 task 的 token 流量 (Table 9), 不进 leaderboard score。GPT-5.5 在 CarRacing 用 0.53M input + 11.36M cache + 0.09M output; Claude Opus 4.7 在同 task 用 0.01M input + 28.90M cache + 0.73M output。Claude 的 cache traffic 远大于 GPT-5.5——可能是因为 Claude Code harness carry long history across revisions。

Insight: **token cost 不能进 leaderboard score**, 因为不同 harness 的 context management 策略不同, normalize 不公平。但作为 diagnostic 它揭示 agent 的"context budget allocation"策略。

## Acknowledgment 里的 origin story

paper 在 acknowledgment 感谢 Jiayi Weng 的 blog post "Learning Beyond Gradients" (https://trinkle23897.github.io/learning-beyond-gradients/, accessed May 18, 2026)。Weng 的核心 insight: **coding agent 可以持续 maintain 和 improve heuristic systems, 而不只是产出一次性 policy file**。

但作者发现 "heuristic" 这个词难 operationalize。传统 hand-written rules 算 heuristic, 但 policy 里包含 tuned numeric parameters, learned components, PPO-optimized baselines 时, 边界就模糊了。

他们的解法很 elegant: **当概念边界模糊时, 不要精修定义, 把被观察对象整体当 benchmark object**。把 executable policy + state + code structure + feedback traces + revision history 一起当成 evaluation unit。Fixed budget 制造 optimization pressure, hidden validation + held-out 防止 leakage。

这个 move 本身很有 methodological insight: 碰到模糊定义的概念, 与其纠缠定义边界, 不如把整个 evolving artifact 当 evaluation 对象。

## 相关联想: 这其实是 programmatic RL 的 revival

传统 RL 是在 parameter space 上做 gradient update。EvoPolicyGym 是在 **source code space** 上做 search。agent 用 LLM 当 proposal distribution, 用 environment feedback 当 reward signal, 迭代改 code。这其实是 old school program synthesis + RL 的混合。

区别在于: 传统 program synthesis 用 genetic programming 或 symbolic regression 当 proposal; 这里用 LLM。LLM 比 GP 强的地方是它能利用 prior knowledge (比如 HalfCheetah 应该用 phase oscillator), 弱的地方是它容易陷入 local mode (一直 churn structure 但抓不到 mechanism)。

References:
- Genetic programming: Koza, "Genetic Programming: On the Programming of Computers by Means of Natural Selection" (1992)
- AlphaEvolve: https://arxiv.org/abs/2506.13131

## 相关联想: The Bitter Lesson 角度

Sutton 的 Bitter Lesson 说 general methods that leverage computation 最终 beat hand-crafted knowledge。EvoPolicyGym 上强 agent 写出来的是 hand-crafted 三段式 policy (CarRacing 的 road mask + curvature + recovery), 这看起来反 bitter lesson。

但仔细想: EvoPolicyGym 测的是 **128 episode budget** regime, 远低于标准 RL 的 converge regime。在 extreme low-data regime, hand-crafted abstraction + LLM prior 确实能 beat end-to-end gradient learning。Sutton 的 bitter lesson 是在 compute-abundant regime 成立的。

如果 budget 从 128 episode 涨到 128M episode, paper 的结论可能会变——agent 可能直接 train 一个 NN policy。这是 paper 没探索的 axis。

## 相关联想: 这和 LLM training 本身的镜像

LLM pre-training 也是 fixed environment (next token prediction) + evolving policy (model weights)。但 LLM training 是 parameter-space update, EvoPolicyGym 是 code-space update。两者其实是同一个抽象的两个 instantiation: fixed task + bounded feedback + evolving policy。

一个有趣的问题: **如果 LLM 在 EvoPolicyGym 上表现好, 是不是说明 LLM 自己 pre-training 时学到的 "policy maintenance" 能力可以 transfer 到 code-space policy evolution?** paper 没直接答, 但 GPT-5.5 的 universal top-2 暗示这种 transfer 存在。

## Build Your Intuition (压成 5 条)

1. **Self-improvement 难评估是因为 improvement 既是 outcome 又是 process**。EvoPolicyGym 的解法: process 物化成 trajectory (edits + submits + feedback), outcome 物化成 hidden held-out return, 中间 hidden validation 做 checkpoint selection 隔离两者。

2. **Bounded budget 是被测能力的一部分**。128 episode 的极端稀缺逼 agent 自己决定 explore vs exploit, 何时改 structure, 何时 tune constant。

3. **AST topology strip 后的变化是 surprisingly good proxy for "是不是引入了新机制"**。它把"改 numeric constant"和"换 control structure"机械地分开, 让你能问"强 agent 是 synthesize 对的 mechanism 还是 tune 对的 parameter"。答案: synthesis-dominant tasks 上前者决定胜负; tuning-dominant tasks 上两者都要但 synthesis 是前置条件。

4. **Useful policies 是 small stateful programs**, 不是巨型 LLM prompt + 大段 reasoning。每个 winning policy 都遵循"task abstraction + controller/planner + recovery logic"三段式。这和 AlphaEvolve 找出来的程序是一族——agent 把抽象物化成可被 audit 的 code artifact。

5. **Heuristic 概念边界模糊时, 把 artifact 整体作为 benchmark object**。这比精修 "heuristic" 定义更 productive, 也更容易 operationalize。

## 我自己看完后的几个开放问题

- **128 episode 是不是太少?** Sutton bitter lesson 在 128M regime 才成立。如果 budget 增大 1000x, leaderboard 可能完全洗牌。paper 没扫这个 axis。
- **Code-space search vs parameter-space search 在什么 regime crossover?** EvoPolicyGym 只测了 code-space, 没直接对比 PPO 等 parametric baseline 在同 128 budget 下的表现。paper 说 "RL baseline 不参与因为 interface 不同", 但其实可以加一个 PPO baseline 跑同样 128 episode, 看它能不能 beat LLM-written hand-crafted policy。我猜 PPO 在 128 episode 下基本废, 但这需要实验。
- **Visibility boundary 在 LLM agent 评估里应该成为 default design 吗?** 现在 SWE-bench 类 benchmark 都没严格 visibility boundary (agent 能看到 unit test 结果然后调到过), 这其实也在 overfit。EvoPolicyGym 的 hidden validation + held-out 模式可以移植到 SWE-bench 类 benchmark 上——把一部分 unit test 设成 hidden, 防止 agent "test until pass"。

References:
- EvoPolicyGym (this paper, https://arxiv.org/abs/2604.12290 - 我推断的 ID, 实际以 paper 为准)
- SWE-bench: https://arxiv.org/abs/2310.06770
- SWE-agent: https://arxiv.org/abs/2405.15793
- CodeAct: https://arxiv.org/abs/2402.01030
- OpenHands: https://arxiv.org/abs/2407.16741
- Reflexion: https://arxiv.org/abs/2303.11366
- Self-Refine: https://arxiv.org/abs/2303.17651
- Voyager: https://arxiv.org/abs/2305.16291
- Eureka: https://arxiv.org/abs/2310.12931
- AlphaEvolve: https://arxiv.org/abs/2506.13131
- Gymnasium: https://arxiv.org/abs/2407.17032
- MiniGrid/MiniWorld: https://arxiv.org/abs/2306.13831
- AgentBench: https://arxiv.org/abs/2308.03688
- OSWorld: https://arxiv.org/abs/2404.07972
- WebArena: https://arxiv.org/abs/2307.13854
- WorkArena: https://arxiv.org/abs/2403.07718
- MLAgentBench: https://arxiv.org/abs/2310.03302
- MLE-bench: https://arxiv.org/abs/2410.07095
- Frontier-eng: https://arxiv.org/abs/2604.12290
- SWE-Evo: https://arxiv.org/abs/2512.18470
- SlopcoderBench: https://arxiv.org/abs/2603.24755
- SpecBench: https://arxiv.org/abs/2605.30314
- RoadmapBench: https://arxiv.org/abs/2605.15846
- AgentLens: https://arxiv.org/abs/2402.08995
- Jiayi Weng "Learning Beyond Gradients": https://trinkle23897.github.io/learning-beyond-gradients/

---

# EvoPolicyGym 详解

这篇 paper 我读了之后觉得它精准踩在一个被既有 benchmark 都没正面回答的空白区域上。我先把核心问题说清楚, 再讲形式化和实验细节。

## 一、它要解决的问题:为什么既有 benchmark 都不够

我们手上的 LLM-based coding agent 现在能做的事, 大致分成三个谱系:

1. **One-shot patch generation** (SWE-bench, https://arxiv.org/abs/2310.06770): 给一个 GitHub issue, agent 一次性提交一个 patch, 用 unit test 判定对错。这里评估的是 "一次性产出正确 artifact 的能力"。
2. **Long-horizon evolving spec** (SWE-Evo https://arxiv.org/abs/2512.18470, SlopcoderBench https://arxiv.org/abs/2603.24755, SpecBench https://arxiv.org/abs/2605.30314, RoadmapBench https://arxiv.org/abs/2605.15846): spec 在变, agent 反复维护代码, 评估的是 "在 evolving specification 下持续维护的质量"。
3. **Feedback-driven self-improvement** (Reflexion https://arxiv.org/abs/2303.11366, Self-Refine https://arxiv.org/abs/2303.17651, Voyager https://arxiv.org/abs/2305.16291, Eureka https://arxiv.org/abs/2310.12931, FunSearch, AlphaEvolve https://arxiv.org/abs/2506.13131): 用 reflection 或外部 feedback 反复 refine 输出。

EvoPolicyGym 想抓的是一个被这三类都漏掉的 setting: **agent 在一个固定的 environment 上, 用极有限的 interaction budget, 反复 edit 一个 executable decision policy, 最终在它从未见过的 held-out cases 上被评估**。

这里有几个关键 design tension:

- **Improvement 既是 outcome 又是 process**: final score 会掩盖 blind retry、overfit to visible feedback、brittle special cases 这些 trajectory-level failure modes。
- **Open-ended engineering 引入 confounder**: 如果 spec 一直变, 你没法分辨是 agent 不会 evolve policy 还是 spec drift 本身就难。
- **Bounded budget 是被测能力的一部分**: 128 episode 的预算逼着 agent 自己决定何时 explore 何时 exploit, 如何把 sparse behavioral evidence 转成 parametric tuning 或 structural synthesis。

所以 EvoPolicyGym 的 setting 选了一条很 specific 的路: **固定 environment + 固定 spec + 固定 episode budget + 严格 train/validation/held-out visibility 边界**。

## 二、Autonomous Policy Evolution 的形式化

每个 run 里有四个核心对象: agent $\pi_\theta$、workspace $W_i$、server-written feedback $F_i$、剩余 episode budget $B_i$。在 observed revision $i$ 处, agent 看到三元组 $(W_i, F_i, B_i)$ 并携带 history $\mathcal{H}_i$。

转移方程:

$$
\pi_\theta(W_i, F_i, B_i, \mathcal{H}_i) \to (u_i, s_i, \mathcal{H}_{i+1})
$$

$$
s_i \in \{\perp\} \cup \mathcal{C}(B_i)
$$

$$
P_{i+1} = \Phi(W_{i+1})
$$

$$
(\Delta F_i, c_i) = S(B_i, P_{i+1}, s_i)
$$

变量解释:

- $\pi_\theta$: harness–model agent, $\theta$ 表示 LLM 参数 + harness tool-use 配置。
- $W_i \in \mathcal{W}$: workspace state, 包含 `system/policy.py`、helper modules、weights、memory files, agent 可见的全部内容。
- $F_i$: server-written feedback from prior submissions, 包括 structured summaries、trajectory records、error reports、可选 frames/videos。
- $B_i \in \mathbb{N}$: remaining episode budget, 整数计数。
- $\mathcal{H}_i$: accumulated conversational and tool-use history, 比如 Claude Code 的 session log。
- $u_i$: workspace patch, 可以 tune constants、add helper modules、introduce memory、replace controller、add diagnostics、restructure policy system。
- $s_i$: server-facing submit command。$\perp$ 表示该 revision 不请求 train evaluation, 否则 $s_i \in \mathcal{C}(B_i)$ 是一个 budget 范围内的 valid train submit (指定 case indices)。
- $\Phi: \mathcal{W} \to \mathcal{P}$: workspace $\to$ executable policy system 的 induction map (server import + fresh instantiation)。
- $P_{i+1} = \Phi(W_{i+1})$: 当前 workspace 诱导出的 executable policy。
- $S$: server operator, 接受 (剩余 budget, 当前 policy, submit command), 返回 $(\Delta F_i, c_i)$。
- $\Delta F_i$: new feedback signal 写到 `feedback/submit_NNN/`。
- $c_i$: charged episode count, $s_i = \perp$ 或 evaluation 不被接受时 $c_i = 0$。

关键设计点: **submit command 不修改 $W_i$ 或 $P_i$**。这意味着 agent 想测一个候选 policy, 必须显式写代码到 workspace, 然后发 submit。这把 "code revision" 和 "evaluation request" 解耦, 让我们能在 trajectory 上分辨 agent 是在改 code、在测 code、还是在 rollback 候选。

## 三、Policy System 的接口设计

Policy system 不是单 file, 而是一个 executable bundle。Judge-facing interface 极简:

```python
class Policy:
    def __init__(self, obs_space, action_space, env_meta): ...
    def reset(self, episode_index): ...   # -> None
    def act(self, obs): ...                # -> action
```

内部状态 $h_t$ 的演化遵循 $\mu(o_t, h_t) \to (a_t, h_{t+1})$ (deterministic) 或 $(a_t, h_{t+1}) \sim \pi(\cdot | o_t, h_t)$ (stochastic)。

这里有一个重要的非平凡选择: **每个 submit 都 fresh import + fresh instantiate Policy**。这逼着 durable state 必须放在 `system/` 下的文件里, 而不能藏在 Python 进程的 global 里。这等于强制 agent 把"学到的"东西外化成 artifact, 而不是依赖隐式 in-memory state。这和 Voyager 的 skill library、AlphaEvolve 的 program database 是同一族思路: 让 learned knowledge 物化成可被 audit 的代码。

## 四、Visibility Boundary: 这是整篇 paper 最关键的设计

Table 6 的三档 split:

| Split | Agent-visible during run | Server role |
|---|---|---|
| Train | IDs, summaries, trajectories, failures | Online revision signal |
| Validation | None | Private checkpoint selection |
| Held-out | None | Final generalization measurement |

具体配置: 每 environment 128 episode train budget, 16 hidden validation cases, 32 hidden held-out cases。

为什么这设计如此重要 — 它直接对抗了几种 cheating mode:

1. **Overfitting to visible feedback**: agent 只看 train trajectory, 没法绕过 validation 直接调参到 held-out 上。
2. **Best-of-N sampling masquerading as improvement**: validation 由 server 选, agent 没法 "submit 直到 validation 高"。
3. **Specification leakage**: held-out 在 server side, agent 完全看不到 case identity、trajectory、return、failure detail。

Checkpoint selection 规则: 128 budget 耗尽后, server 在所有 `status == ok` 的 checkpoint 上跑 hidden validation, 选 mean return 最高的; ties 选 later submit (鼓励 agent 持续改进而非早停)。

## 五、Core16 实验配置与主结果

16 个 environment 来自 Gymnasium (https://arxiv.org/abs/2407.17032) 的四个家族, 每家族 4 个:

- **Gym/Box2D**: Acrobot, MountainCarContinuous, BipedalWalker, CarRacing
- **MuJoCo** (https://arxiv.org/abs/1208.0947, Todorov 2012 IROS): Reacher-v5, HalfCheetah-v5, Ant-v5, Pusher-v5
- **MiniGrid** (https://arxiv.org/abs/2306.13831): DoorKey-16x16, KeyCorridorS4R3, FourRooms, ObstructedMaze-1Q
- **Robotics/Driving**: parking, roundabout, FetchPush-v4, FetchPickAndPlace-v4

4 个 agent 用相同 128-episode budget:

| Agent | Harness | Model string |
|---|---|---|
| GPT-5.5 | Codex | gpt-5.5 |
| Claude Opus 4.7 | Claude Code | claude-opus-4-7-thinking-max |
| MiniMax-M3 | Claude Code | MiniMax-M3 |
| DeepSeek-V4-Pro | Claude Code | deepseek-v4-pro |

主结果 (Table 2):

| Model | Gym/Box2D | MuJoCo | MiniGrid | Robotics/Driving | Core16 | Wins | Top-2 |
|---|---|---|---|---|---|---|---|
| GPT-5.5 | 0.938 | 0.875 | 0.812 | 0.938 | **0.891** | 9 | 16 |
| Claude Opus 4.7 | 0.812 | 0.750 | **0.938** | 0.500 | 0.750 | 5 | 12 |
| MiniMax-M3 | 0.375 | 0.625 | 0.500 | 0.625 | 0.531 | 1 | 3 |
| DeepSeek-V4-Pro | 0.375 | 0.250 | 0.438 | 0.375 | 0.359 | 1 | 1 |
| Random | 0.000 | 0.000 | 0.375 | 0.062 | 0.109 | 0 | 0 |

GPT-5.5 在 16 个环境全进 top-2, 这是唯一一个 universal top-2 entry。Claude Opus 4.7 在 MiniGrid 反超 (0.938 vs GPT-5.5 的 0.812), 主要靠 KeyCorridor、FourRooms、ObstructedMaze、Ant、ContinuousCar 五个 wins。

Scoring 公式 (rank-based, 跨 environment 可比):

$$
s_{m,e} = 1 - \frac{\text{rank}_e(m) - 1}{N_e - 1}
$$

变量: $m$ 是 entry (agent 或 random reference), $e$ 是 environment, $\text{rank}_e(m)$ 是 entry $m$ 在 environment $e$ 上 held-out mean return $y_{m,e}$ 的 descending rank, $N_e = 5$ (4 agents + random reference)。ties 共享 rank score。Category score 在该 category 内 4 个 environment 上 macro-average, Core16 score 在 16 个 environment 上 macro-average。

这个 rank-based 设计 build intuition: 因为 raw return 在 Acrobot 上是 -84.688 但在 MiniGrid 上是 0.986, 量纲完全不可比, rank 才能让 cross-env aggregation 有意义。同时它 reward 的是 "**consistently top across heterogeneous tasks**", 而非 "**few isolated wins**"。MiniMax-M3 在 HalfCheetah 上 win 一个, 但其他 environment 弱, Core16 才 0.531; DeepSeek-V4-Pro win 一个 Roundabout, 其他都偏弱, Core16 0.359。

## 六、Synthesis vs Tuning: 这是我觉得最 insightful 的二分

paper 把 exploration mode 分成两类:

- **Structural synthesis**: 引入 new policy-bundle AST topology (perception module, memory, planner, reward interpretation, state abstraction)。
- **Parametric tuning**: 在 plausible controller 内调整 gains、thresholds、constants、branch-local parameters。

classification 是机械化的: 把 source bundle 的 numeric constants 全部 strip 掉, 然后比较 AST topology。如果 topology 改了 → synthesis edit; topology 保留 → parametric edit。

环境的 demand 侧也分成两组:

- **Synthesis-dominant** (5 个): DoorKey, KeyCorridor, FourRooms, ObstructedMaze, CarRacing — 需要 pixel perception 或 symbolic planning machinery。
- **Tuning-dominant** (11 个): 其余 lower-dimensional control tasks。

Table 3 看 realized computational structure (AST features):

| Demand | Agent | Funcs | Branches | Loops | Depth | State |
|---|---|---|---|---|---|---|
| Synthesis | GPT-5.5 | 30.2 | 68.2 | 13.0 | 4.6 | 48.4 |
| Synthesis | Claude Opus 4.7 | 19.0 | 77.0 | 16.8 | 7.6 | 26.0 |
| Synthesis | MiniMax-M3 | 12.8 | 38.2 | 7.6 | 4.0 | 16.0 |
| Synthesis | DeepSeek-V4-Pro | 5.4 | 21.8 | 3.2 | 3.2 | 11.2 |
| Tuning | GPT-5.5 | 8.5 | 8.6 | 0.5 | 2.1 | 9.2 |
| Tuning | Claude Opus 4.7 | 6.5 | 4.5 | 0.5 | 1.7 | 8.2 |
| Tuning | MiniMax-M3 | 4.5 | 7.7 | 0.4 | 2.3 | 6.9 |
| Tuning | DeepSeek-V4-Pro | 3.6 | 5.8 | 0.1 | 2.5 | 5.7 |

观察: synthesis-dominant tasks 上 strong agents 选 substantially richer source bundles (GPT-5.5 平均 30.2 funcs, 48.4 state vars); tuning-dominant 上所有 agents 都更 compact (GPT-5.5 也降到 8.5 funcs)。但 code volume 不是 sufficient: 复杂代码未必是 task-adapted mechanism, 这是 diagnostic 而非证明。

Figure 4 用 normalized scale 分离两组:

$$
\text{norm}_{m,e} = \text{clip}_{[0,1]}\left(\frac{R_{m,e}^{\text{heldout}} - R_e^{\text{random}}}{R_e^{\text{best}} - R_e^{\text{random}}}\right)
$$

变量: $R_{m,e}^{\text{heldout}}$ 是 entry $m$ 在 environment $e$ 的 held-out mean return, $R_e^{\text{random}}$ 是同 environment 同 held-out pool 上 random policy 的 mean return, $R_e^{\text{best}}$ 是该 environment 上所有 evaluated agent 中最佳 held-out score。clip 到 $[0,1]$ 保证可比。

结果:

- Synthesis-dominant: GPT-5.5 = 0.98, Claude = 1.00, MiniMax-M3 = 0.19, DeepSeek-V4-Pro = 0.03
- Tuning-dominant: GPT-5.5 = 0.99, Claude = 0.83, MiniMax-M3 = 0.83, DeepSeek-V4-Pro = 0.67

Insight: **score gap 集中在 synthesis 侧**。MiniMax-M3 和 DeepSeek-V4-Pro 在 synthesis tasks 上接近 random anchor (0.19, 0.03), 三个 locked-door MiniGrid 全 solve 不了。在 tuning-dominant 上他们和 strong agents 聚拢 (0.83, 0.67)。这提示存在两种 candidate failure mode: **failing to discover an effective structure** 和 **failing to refine a plausible structure**。

## 七、Edit-type 成功率 (Table 4)

把每个 score-bearing submit transition 按 edit type 分类, hit = 提升 validation best-so-far:

| Agent | Task type | Synthesis n | Synthesis hit | Parametric n | Parametric hit |
|---|---|---|---|---|---|
| GPT-5.5 | Synthesis | 37 | 41% | 1 | 100% |
| GPT-5.5 | Tuning | 48 | 38% | 31 | 61% |
| Claude Opus 4.7 | Synthesis | 31 | 48% | — | — |
| Claude Opus 4.7 | Tuning | 123 | 26% | 58 | 21% |
| MiniMax-M3 | Synthesis | 39 | 10% | — | — |
| MiniMax-M3 | Tuning | 100 | 25% | 57 | 25% |
| DeepSeek-V4-Pro | Synthesis | 132 | 3% | — | 1 |
| DeepSeek-V4-Pro | Tuning | 112 | 30% | 21 | 38% |

最 striking: 在 synthesis tasks 上, GPT-5.5 / Claude Opus 4.7 把 synthesis edit 转成 validation best 的比率是 41-48%, 而 MiniMax-M3 / DeepSeek-V4-Pro 是 10%/3%。后两者 **churn structure without traction** — 一直换 topology 但抓不到 task-appropriate machinery。

Insight: same-topology edits 几乎救不了 wrong mechanism, 但在 tuning-dominant tasks 上, 一旦 controller family 对了, parametric edits 就开始有用 (GPT-5.5 在 tuning tasks 上 parametric edit 61% hit)。

## 八、Post-hoc Score Trajectories (Figure 3)

Agent 在 optimization 过程中从未观察过 hidden validation curve。这条 curve 是 run 结束后重建的 best-so-far。

- **Vertical jump** = 发现更高质量 candidate policy。
- **Flat segment** = 预算消耗但 validation 不涨。

Family level pattern:

- MiniGrid: sparse 但 sharp jumps
- MuJoCo: incremental gains
- Robotics/Driving: delayed improvements after substantial budget spent

这个 diagnostic 区分了 "**早 jump 后 plateau**" 和 "**晚 jump after lots of budget**" 两种到达相同 final score 的路径。前者 budget-efficient, 后者 budget-inefficient。这给 leaderboard 加了一个时间维度。

## 九、CarRacing 和 BipedalWalker 的 Case Study

paper 随机抽了两个 case study, 一个来自 synthesis-dominant (CarRacing), 一个来自 tuning-dominant (BipedalWalker)。

**CarRacing timeline (Figure 5)**:
- Claude Opus 4.7: 一直在 synthesis-edit phase
- GPT-5.5: 早期 synthesis 改进, 中间一段 short parametric-edit phase, 再回到 synthesis edits
- 弱 agents: mechanism replacements 和 retests 频繁发生, controller 难以逃出 wrong abstraction

**BipedalWalker timeline (Figure 6)**:
- 环境是 tuning-dominant, 但 **tuning 只在 gait-producing topology 存在之后才有效**
- GPT-5.5 是唯一达到 positive high-return gait 的 run (timeline best 271, validation-selected held-out 248.874, Table 1)
- 其他三个停在 negative timeline best (≤ -15.6), churn structures without crossing the return threshold

这强化了一个关键 insight: **"tuning-dominant" 不意味着不需要 synthesis**, 而是 synthesis 的 milestone 是一个 viable base topology, 之后 parametric refinement 才能起作用。

## 十、实际 Policy 机制 (Appendix D)

paper 给了四个典型 policy 的代码片段, 这些是 "small stateful programs":

**CarRacing**: pixel → road mask → trace near/mid/far centers → look-ahead curvature + edge warnings + speed reduction + recovery mode (lost_steps 计数, 退化到 global road center 或 last_target)。

**HalfCheetah**: periodic gait via phase oscillator
```python
phase = 2π · freq · t · dt
hip = hip_amp · sin(phase)
knee = knee_amp · sin(phase + π/2)
ankle = ankle_amp · sin(phase + π)
```
+ safety scaling: `scale = max(safety_floor, 1 - excess)` where `excess = max(0, |root_height| - safety_thresh)`。本质上是用 root height 作为 fall indicator 削减 amplitude。

**ObstructedMaze**: egocentric 7×7 view → 持久 symbolic world model → sync pose using previous action → BFS plan toward targets (ball/key/door/blocker/frontier) → action hierarchy (PICKUP → drop_at_safe_cell → clear_blocker_before_door → BFS follow → unstick_action)。

**FetchPush**: 几何 phase controller
- direction = unit(goal[:2] - obj[:2])
- behind = obj[:2] - 0.065 · direction
- Phase 切换: distance_xy < 0.05 → final descent; needs_clear → clearance; norm(behind - grip) > 0.03 → approach; |grip.z - push_z| > 0.022 → lower; else push through

Insight: 所有 useful policies 都是 "**task abstraction + controller/planner + recovery logic for failure modes exposed by visible feedback**" 的三段式。这和 Voyager 的 skill library、Eureka 的 reward code 是一族: **agent 把抽象结构物化成可被 audit 的代码**, 然后 tuning 在结构上做局部 refinement。

## 十一、Acknowledgement 里那个被忽视的关键线索

paper 在 acknowledgement 里点名感谢 Jiayi Weng 的 blog post "Learning Beyond Gradients" (https://trinkle23897.github.io/learning-beyond-gradients/, accessed May 18, 2026)。

Weng 的核心 insight 是: **coding agent 可以持续地 maintain 和 improve heuristic systems, 而非仅仅 produce 一次性 policy file**。这个 insight 直接启发了 EvoPolicyGym 的 starting point。

但作者发现 "**heuristic**" 这个词很难 operationalize: 传统 hand-written rules 算 heuristic, 但当 policy 包含 tuned numeric parameters、learned components、PPO-optimized baselines 时, boundary 就模糊了。这个 difficulty 把他们推向一个 elegant reformulation: **把 policy system 本身作为 benchmark object** — executable policy + state + code structure + feedback traces + revision history。Fixed environment-interaction budget 制造 optimization pressure, hidden validation + held-out splits 防止 leakage。

这其实是一个值得玩味的 move: 当一个概念边界模糊时, 不要去精修定义, 而是把被观察的对象整体变成 benchmark 的 evaluation unit。这等价于在说: "**heuristic 不重要, agent 维护的那个 evolving artifact 才是 evaluation unit**"。

## 十二、Limitations

paper 自己承认 (Section 5.3):

- AST topology 是 conservative proxy, 不是 semantic proof。两个 topology 可以 implement similar behavior, 一个 topology 可以混合 useful/harmful ideas。
- policy source-bundle 边界包含 `policy.py` + reachable local helper modules, 但仍排除 generated data files、learned weights、unreferenced experiments。
- synthesis/tuning split 是 lens 不是 taxonomy: Bipedal 仍需 viable gait structure 才能 tuning; CarRacing 在 perception + control 到位后仍从 later parameter choices 受益。

## Build Your Intuition

把整个 paper 压缩成几条核心直觉:

1. **Self-improvement 难评估是因为 improvement 既是 outcome 又是 process**。EvoPolicyGym 的解法: 把 process 物化成 trajectory (workspace edits + submits + feedback), 把 outcome 物化成 hidden held-out mean return, 中间用 hidden validation 做 checkpoint selection 隔离两者。

2. **Bounded budget 是被测能力的一部分**, 不是 confounder。128 episode 的极端稀缺逼着 agent 自己决定何时 explore / exploit, 如何把 sparse behavioral evidence 转成 parametric refinement 或 structural invention。

3. **AST topology strip 后的变化**是一个 surprisingly effective proxy for "是不是引入了新机制"。它把 "改了 numeric constant" 和 "换了 control structure" 机械地分开, 让我们能问 "**strong agent 是因为 synthesize 对的 mechanism 还是 tune 对的 parameter?**"。答案是: 在 synthesis-dominant tasks 上前者决定胜负, 在 tuning-dominant tasks 上两者都重要但需要 viable base topology 先到位。

4. **Useful policies 是 small stateful programs**, 不是巨型 LLM prompt + 大段 reasoning。每个 winning policy 都是 "**task abstraction + controller/planner + recovery logic for visible-failure modes**" 三段式。这呼应了 AlphaEvolve 在算法发现上的观察 — 强 agent 把抽象物化成可被 audit 的代码 artifact。

5. **Heuristic 的概念边界模糊时, 把 artifact 整体作为 benchmark object**。这个 reformulation move 比试图精修 "heuristic" 定义更 productive。

References:
- SWE-bench: https://arxiv.org/abs/2310.06770
- SWE-agent: https://arxiv.org/abs/2405.15793
- CodeAct: https://arxiv.org/abs/2402.01030
- OpenHands: https://arxiv.org/abs/2407.16741
- Reflexion: https://arxiv.org/abs/2303.11366
- Self-Refine: https://arxiv.org/abs/2303.17651
- Voyager: https://arxiv.org/abs/2305.16291
- Eureka: https://arxiv.org/abs/2310.12931
- AlphaEvolve: https://arxiv.org/abs/2506.13131
- Gymnasium: https://arxiv.org/abs/2407.17032
- MiniGrid/MiniWorld: https://arxiv.org/abs/2306.13831
- AgentBench: https://arxiv.org/abs/2308.03688
- OSWorld: https://arxiv.org/abs/2404.07972
- WebArena: https://arxiv.org/abs/2307.13854
- WorkArena: https://arxiv.org/abs/2403.07718
- MLAgentBench: https://arxiv.org/abs/2310.03302
- MLE-bench: https://arxiv.org/abs/2410.07095
- Frontier-eng: https://arxiv.org/abs/2604.12290
- SWE-Evo: https://arxiv.org/abs/2512.18470
- SlopcoderBench: https://arxiv.org/abs/2603.24755
- SpecBench: https://arxiv.org/abs/2605.30314
- RoadmapBench: https://arxiv.org/abs/2605.15846
- AgentLens: https://arxiv.org/abs/2402.08995
- Jiayi Weng blog "Learning Beyond Gradients": https://trinkle23897.github.io/learning-beyond-gradients/
