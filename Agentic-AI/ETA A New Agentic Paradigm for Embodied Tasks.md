---
source_pdf: ETA A New Agentic Paradigm for Embodied Tasks.pdf
paper_sha256: b50ded7501500dc9c5c33aa56793d133f5661eb5b78134a5ec8e8a49f71b23e9
processed_at: '2026-08-18T11:24:51-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

## 核心一句话

这篇 paper 说：别让 LLM 直接预测 robot 的每一步动作了，让 LLM 当"包工头"，只负责发指令，底下有专门的 Tool 干活，每干完一步必须回来汇报，确认没搞砸再发下一条指令。

## 为什么搞这个

现在 embodied AI 最火的是 VLA 路线，比如 OpenVLA、$\pi_0$、GR00T N1 这些。它们做的事情是：输入 image + language instruction，输出 action tokens，直接告诉 robot 怎么动。

这条路线有几个本质问题：

**第一，generalization 靠数据堆。** 你的 training data 覆盖了哪些 object、哪些 scene，model 就只会处理这些。物理世界的组合空间是无限的，你不可能穷举。

**第二，出了错没法查。** VLA 输出一段 action sequence，中间某步 grasp 错了，你根本不知道是 perception 错了还是 planning 错了。整个 thing 是黑盒，debug 无从下手。

**第三，没法从经验里学。** 你想让 robot 记住"上次 grasp 那个杯子失败了，换个角度试试"，但 VLA 的 end-to-end 结构没有显式的 decision point 来 anchor 这种 feedback。

与此同时，digital world 的 coding agent 已经很猛了。ReAct、Reflexion、Voyager 这条线证明了一件事：让 LLM 通过 tool use + feedback loop + experience accumulation，能解决非常复杂的问题。Voyager 在 Minecraft 里能自己探索、写 skill code、积累 skill library，越来越强。

ETA 的核心 insight 就是：**把 coding agent 这套 loop 搬到 physical world，但加上 physical world 必需的 safety constraint。**

## 具体怎么搞的

### 最核心的设计原则

ETA 有一个 runtime invariant，这句话是整个 paper 的灵魂：

> 每次只能执行一个改变 physical world 的 action，执行完必须拿到 fresh observation，才能决定下一步。

为什么这条规则如此关键？在 digital agent 里，你 edit 了 file A，file A 就变了，你可以接着 edit file B，最后再 read 一下。File system 是 deterministic 的。

但 physical world 不一样。你 grasp 一个 object，它可能：
- 成功 grasp 住了
- grasp 空了
- grasp 错了 object
- 碰到东西撞了

如果你在没看结果的情况下连续执行三个 physical actions，你就是在赌场景没变。这在物理世界是不可接受的。

所以 ETA 强制：**每一步 physical action 之后，必须 observe，必须确认结果，才能 decide 下一步。** 这 trade 的是 latency（Planner 要 call 更多轮），换来的是每一步都 grounded in 真实 evidence。

### 三个角色

ETA 把系统拆成三层，有严格的 trust boundary：

**Agent（Planner）**：就是那个 LLM 大脑。它理解 task，维护 working memory，提出下一步要干嘛。但关键——**它没有直接执行 physical action 的权力**。它只能说"我想做 X"，不能直接做。

**Interface**：Agent 和 World 之间的 gatekeeper。它检查 Agent 的命令合不合法、安不安全、前置条件满不满足。Agent 说"我想做 X"，Interface 决定"你可能做"。这层防止 Agent 干蠢事。

**World**：实际执行的地方。Simulator 或 real robot。它执行完返回三个东西——Tool result、environment receipt（trusted 的 reward/termination 信号）、fresh observation。

这个 separation 看起来 over-engineering，但解决一个关键问题：**防止 model 作弊。** 想象 GPT-5.6 这种级别的 model，如果它能直接 read simulator 的 ground-truth object pose，它根本不需要做 perception——直接知道物体在哪。这会让 benchmark 分数 inflated，但 transfer 不到 real robot。Interface 的 trusted receipt 机制确保 reward 只能来自 World 层的 attested receipt，model 不能自己在 return value 里伪造。

### 形式化

在 turn $t$，Agent 产生 command：

$$c_t = \pi_{\text{agent}}(g, o_t, m_t)$$

- $c_t$：turn $t$ 的命令
- $\pi_{\text{agent}}$：Agent 的 policy（就是那个 LLM）
- $g$：task goal
- $o_t$：当前 observation
- $m_t$：working memory（历史 trajectory、obligation、skill 都在这）

Interface 做 gate check：

$$a_t = \begin{cases} \text{dispatch}(c_t), & \text{gate}(c_t, o_t, m_t) = \text{pass} \\ \emptyset, & \text{otherwise} \end{cases}$$

- $a_t$：实际执行的 action
- $\text{gate}(\cdot)$：检查函数，看 structure、authority、provenance、prerequisite 对不对
- $\emptyset$：被拒绝，不执行

World 返回 $(y_t, e_t, o_{t+1})$：
- $y_t$：Tool 的 output
- $e_t$：environment receipt（trusted reward 信号）
- $o_{t+1}$：fresh observation

只有这三条 record 进入 working memory 后，Planning 才能 resume。这就是把 "decision → execution → verification" 强制成一个 atomic unit。

## OpenETA 的两种 configuration

### Full OpenETA（44 个 Tools）

有一个 broad Tool registry，覆盖五大类：

| Category | 代表 Tools | 干什么 |
|----------|-----------|--------|
| Perception | observe, sam3, molmopoint | 看场景、分割、定位 |
| Geometry planning | anygrasp, contact_graspnet, anyplace | 算 grasp pose、placement |
| Physical execution | move_to, gripper_control, ik_preview_check | 实际动 robot |
| Environment | create_simulator_env, close_simulator_env | 管理 environment lifecycle |
| Agent support | save_memory, get_memory, register_skill | 管理 memory、skill |

每个 Tool 声明 side-effect class：`read_only`、`planning`、`bookkeeping`、`world_mutating`。这个 classification 决定 calls 能不能 batch、要不要 fresh observation。Perception calls 可以 batch（不改变 world state），但 `world_mutating` actions 必须一次一个。

### OpenETA for Codex（就 3 个 Tools）

这是最 striking 的配置。只暴露三个 embodied Tools：

**observe**：返回 live images。可以 request 不同视角。

**mark_point**：Agent 在 image 上点一个 2D point，Tool 把它转成 3D World coordinate。有两种 mode——multi-view mode 让 Agent 在两个 view 上各点一个点来 triangulate；single-view mode 直接返回 camera ray 的 first surface intersection。

**move_to**：移动 gripper 到 target pose。Agent 用 approach direction 和 jaw direction 两个 vector 来 specify orientation，比 raw rotation matrix 直观。Close 之前还 render 一个 preview，Agent 可以先看看 target pose 对不对再决定 execute。

设计 motivation 是个 analogy：coding agent 通过 file access、edit、shell、search、version control 这几个 stable operation 就能解决无数 task。同理，embodied control 也可以用极小的 physical interface，把 task decomposition、target selection、spatial reasoning 都留给 multimodal Planner 去 figure out。

## 实验结果

### Full OpenETA on LIBERO（fixed matrix）

LIBERO 是个 manipulation benchmark，4 个 suite（Spatial, Object, Goal, Long/LIBERO-10），每个 10 tasks。Fixed matrix = 40 tasks × 10 seeds = 400 episodes。Planner 是 GPT-5.6 Luna。

| Suite | Success | Rate |
|-------|---------|------|
| Spatial | 8/100 | 8.0% |
| Object | 26/100 | 26.0% |
| Goal | 21/100 | 21.0% |
| Long/LIBERO-10 | 1/100 | 1.0% |
| Overall | 56/400 | 14.0% |

14% 看着 low，但 paper 分析了 failure 原因：

| Failure type | 占比 |
|-------------|------|
| episode_timeout | 62.5% |
| unattended_ask_human | 19.2% |
| max_turns | 10.2% |
| status_report_without_reward | 7.0% |

62.5% 是 timeout。19.2% 是 agent 主动 ask human 但没人 respond（算 failure，不是 assisted）。这说明 bottleneck 不是 physical execution 本身，而是 **long-horizon subgoal tracking、perceptual rechecking、placement relations、remaining-budget coordination**。

Success episode 平均 30.7 turns、1045 秒；Failure episode 平均 45.5 turns、1867 秒。Failure 用了更多 turns——它们确实在 try to recover，但最终 timeout 了。这 hint 了：current general-purpose models 还不能可靠地养成 "observe and correct after every physical action" 的 habit。

### OpenETA for Codex（3 个 Tools，130 tasks）

这个结果非常 striking：

| Planner | PASS@1 | PASS@5 |
|---------|--------|--------|
| GPT-5.6 Luna | 21/130 | 62/130 (47.7%) |
| GPT-5.6 Terra | 58/130 | 83/130 (63.8%) |
| GPT-5.6 Sol | 92/130 | 117/130 (90.0%) |

**只用三个 Tools + GPT-5.6 Sol，在 130 个 LIBERO tasks 上达到 90% PASS@5，没用任何 VLA 或 task-specific policy。**

Sol 在 Spatial 和 Object 上几乎 perfect（10/10）。Goal suite 有 2 个 task 始终 fail。LIBERO-90 还有 11 个 fail。

为什么 Full OpenETA（44 Tools + Luna）只有 14%，而 Codex 配置（3 Tools + Sol）有 90% PASS@5？两个 difference：(1) Planner strength（Sol >> Luna）；(2) Codex 用 PASS@5（5 seeds），Full OpenETA 是 fixed matrix（single seed per cell）。但即便如此，Planner strength 的 effect 巨大。

Sol 相比 Terra 和 Luna 的优势特别在 **precise grasp-point localization**——它产生 fewer empty grasps，并且更 frequently 用 move_to preview 在 act 之前 inspect。这 hint 了 spatial reasoning 和 lookahead 是 key differentiator。

**这个结果说明：在 agentic paradigm 下，Planner reasoning ability 是 dominant bottleneck，而不是 Tool 数量。** 这和 VLA 的 data-driven scaling 是完全不同的 scaling axis。

## Self-evolution（最 honest 的失败实验）

这部分 intellectually 非常 honest。Paper 问：能不能用 recorded experience 改进后续行为？答案是不能——**没有任何 candidate 通过 promotion gate。**

### Promotion Rule

$$\text{Promote}(u) = D(u) \land Q(u) \land R(u) \land H(u)$$

- $u$：proposed update
- $D(u)$：deterministic contract check（schema 和 invariant 验证）
- $Q(u)$：isolated review（在 isolated context 中 review）
- $R(u)$：same-task replay（在原 task 和 seed 上 reproduce trusted success）
- $H(u)$：paired held-out evaluation（在 held-out tasks 上 show objective gain without regression）

四个 gate 全过才能 promote。

### 三种 update form 的结果

**Task-local Skill**（edit guidance for full workflow）：baseline 0/30, candidate 0/30。Candidate 反而 reach attachment 和 placement stage **less often**。

**Exact-task playbook**（从成功 trajectory 提取，exact match 时 load）：baseline 4/30, candidate 1/30。Candidate 用更多 turns、更多 timeouts。你从成功 trajectory 提取 playbook，结果反而 worse。

**Stage-local delta**（对比成功/失败 trajectory）：两个版本都无法 reproduce source success。

### 为什么失败？

Paper 的分析很到位：

> "Most candidates add another visual check or recovery rule; they do not change the underlying perception or control Tools."

核心问题：current update methods 只能在 **textual guidance level** 修改，不能修改 underlying Tool capability。如果 failure 的 root cause 是 grasp estimator 给了 bad candidate，你加一个 "check again" rule 没用——你只会 re-run 同一个 bad grasp estimator。Textual experience 只能 influence model 的 next decision，但 model 的 reasoning ability 和 Tool capability 是 fixed 的。

这 hint 了：未来的 self-improvement 可能需要 **co-evolve model reasoning 和 Tool capability**。如果 Agent 发现某个 Tool 在某类 object 上总 fail，它应该能 trigger 一个 retraining process 来 improve 那个 Tool。Paper roadmap 里提到了这个方向但还没实现。

## Sim2Real 的 honest disclosure

Current release 只有 interface-level integration。两个 demonstration 视频（sponge-to-tray 和 bell-pepper grasp）是 qualitative 的，不能 establish success rate。

Development testing 暴露了两个 bottleneck：

**Depth quality**：RealSense D435i + L515 的 depth 完整性远不如 simulator depth。Depth enhancement 改善了但没彻底 fix，导致 pose estimate 质量不够。

**Motion control**：lateral grasp 有时 trigger acceleration-limit protective stops。可能是 controller PD tuning 或 trajectory shaping 的问题，但 recordings 没 establish causal diagnosis。

这个 honest disclosure 说明：simulation 结果不能直接 imply real-robot capability。从 sim 到 real 的 transfer 仍然需要大量 engineering。

## 这条路线的深层 intuition

### 为什么 Agent 不能直接 control world？

Digital agent 能直接 call shell、edit file，因为 file system 是 deterministic 的。但 physical world 有两个本质 difference：

1. **Stochasticity**：physical action 的 outcome 不确定。你 grasp 一个 object，它可能成功可能失败，必须 observe。
2. **Safety**：physical action 可能造成 damage。需要 gate 来 prevent unsafe actions。

Interface 层解决这两个问题——Agent 有 intelligence 但 no authority，Interface 有 authority 但 no intelligence，World 有 ground truth 但 no initiative。

### 为什么 one-action-at-a-time？

Cost 是 real 的——Full OpenETA 平均 43.4 turns per episode，success episode 平均 1045 秒。但 benefit 是每一步都 grounded in trusted evidence。

Alternative 是生成一段 action sequence 执行。但如果中间某步失败，你需要 detect failure、rollback、replan。在 physical world 你没法 "undo" grasp——你只能 open gripper 重新来。One-action-at-a-time 让 failure detection 和 recovery 变 natural：每步 observe，不对就立即 replan。

### 为什么 Tool contracts stable？

Voyager 里 model 能生成新 skill code 并 execute。Minecraft 里 worst case 就是 avatar 死了 respawn。但 physical world 一个 buggy generated skill 可能导致 robot 撞坏东西或伤人。

ETA 的设计：Tool contracts 由 host 定义，Agent 不能 runtime rewrite。Skill 是 textual guidance，不 execute hidden action sequence。Agent 必须显式 select 每一个 atomic Tool。这限制 flexibility 但保证 safety。

### 为什么 reward 必须是 trusted receipt？

如果 Agent 能直接 declare success，strong model 可能 hallucinate success。LIBERO evaluation 要求 success 只能来自 official positive reward in trusted environment receipt，且 provenance、execution ID、session ID 必须匹配 current turn。这防止 model 在 Tool return value 里伪造 reward 字段，也防止 model inspect simulator hidden state 来 cheat。

## 和 VLA 路线的对比

VLA 的 scaling 主要靠更多 robot training data。ETA 的 scaling 主要靠更强 general-purpose model。这是完全不同的 scaling axis。

如果 general-purpose model 继续 improve（从 GPT-4 到 GPT-5.6 Sol 已经 huge leap），ETA paradigm 的 benefit 会越来越大——你只需要 expose 几个 stable Tools，强 model 就能 figure out 怎么 compose。而 VLA 每扩展一个新 embodiment 或新 task type 都需要大量新 training data。

Paper 的 vision 是两者融合：**strong Planner + VLA as specialist Tool + WAM as predictor。** Planner 在需要时 invoke VLA 做 contact-rich 或 dexterous manipulation，invoke WAM 做未来预测和 plan comparison，但 task-level decision 始终 grounded in observed World evidence。

## 我的 takeaway

这篇 paper 的真正 contribution 不在 performance number（14% 在 LIBERO fixed matrix 上并不 high），而在 **paradigm 层面**。它提出了一个 protocol，把 physical world 的 stochasticity 和 safety 问题转化为可工程化的问题。

几个核心 design choice 值得记住：
1. **One-action-at-a-time with fresh observation** — trade latency for controllability
2. **Agent-Interface-World separation** — intelligence 与 authority 分离
3. **Trusted receipts** — 防止 model 作弊，保证 evaluation integrity
4. **Stable Tool contracts** — 防止 runtime generated code 造成 physical damage
5. **Constrained self-evolution with promotion gates** — safety-first，防止 bad experience 污染 capability library

Self-evolution 的 honest 失败和 Sim2Real 的 honest disclosure 让这篇 paper 非常 credible。它没有 overclaim，明确说了 current limitation 和 future work。这种 intellectual honesty 在 embodied AI 领域不多见。

从更大视角看，这篇 paper 在 bet 一件事：**embodied AI 的 ChatGPT moment 不来自更大的 VLA，而来自把 digital agent 的 loop 结构正确地迁移到 physical world。** 这个 bet 是否 correct 取决于 general-purpose model 的 reasoning ability 能否 continue to scale。如果 GPT-6、GPT-7 级别的 model 能做到更 reliable 的 spatial reasoning 和 long-horizon planning，ETA paradigm 的 leverage 会非常大。

---

# ETA: A New Agentic Paradigm for Embodied Tasks 深度解析

你好 Karpathy！这篇 paper 来自 Shanghai Innovation Institute 和 Fudan University 的团队（邱锡鹏老师组），非常有意思，因为它实际上是在重新思考 embodied AI 的架构层次问题。让我一层一层 build up intuition。

## 1. Motivation：为什么 VLA 路线存在 ceiling

当前 embodied intelligence 的主流路线是 VLA (Vision-Language-Action) models，比如 RT-2, OpenVLA, $\pi_0$, GR00T N1。这些 models 把 observation 和 language instruction 直接 map 到 robot action tokens。

ETA paper 指出这条路线的三个根本瓶颈：

**Generalization 受限于 robot training data 覆盖**。VLA 的 generalization 本质上依赖 training distribution 覆盖了多少 object/environment/embodiment 组合。Open X-Embodiment 已经聚合了大量 heterogeneous data，但物理世界的组合空间是无限的。

**Long-horizon task 难以 control 和 inspect**。当 VLA 输出一段 action sequence，如果中间某一步出错，你很难定位是 perception 错了，还是 planning 错了，还是 control 错了。整个 system 是一个黑盒。

**Self-improvement 缺乏可信的 feedback loop**。要让 robot 从 experience 中学习，你需要知道哪次 execution 成功了、哪次失败了、为什么失败。但 VLA 的 end-to-end 结构没有显式的 decision boundary 来 anchor 这种 feedback。

这里有一个关键 insight：coding agents 已经在 digital world 实现了 emergence——它们能 decompose 高层目标、navigate 复杂环境、execute actions、evaluate results、recover from failures、preserve successful procedures。ReAct, Reflexion, Voyager 这条线已经证明了 "tool-use + feedback loop + experience accumulation" 的范式有效。ETA 的核心 idea 就是 **把 coding agent 的 loop 结构迁移到 physical world**。

## 2. ETA Paradigm 的核心架构

ETA 的核心是重新定义 embodied agent 的信息流。看 Figure 1 描述的 loop：

```
Planner → chooses one Tool call
   ↓
Interface → validates and dispatches
   ↓
World → executes, returns result + fresh observation
   ↓
Planner reads fresh observation, decides next step
```

这个 loop 看起来简单，但有几个非常严格的约束让它本质上不同于 program-centric methods（Code as Policies, VoxPoser, RoboCodeX 等）。

### 2.1 Runtime Invariant（最重要的设计原则）

ETA 的 runtime invariant 是：

> Execute only one world-changing action at a time. Afterward, obtain a fresh observation before executing the next state-dependent action.

这条规则防止两件事：(1) 一个 unobserved sequence of world changes；(2) system 假设 scene 保持不变。

为什么这条规则关键？在 digital agent 中，file system 的 state 是 deterministic 的——你 edit 一个 file，它就变了，你可以连续 edit 多个 file 然后 read。但 physical world 是 stochastic 的：你 grasp 一个 object，它可能滑落、可能 grasp 错 object、可能 collision。如果你在没看到结果的情况下连续执行多个 physical actions，你就在 bet 场景没变——这在 physical world 是不可接受的。

这条 invariant 把 ETA 和 program-centric methods 区分开。Code as Policies 生成一段 program，program 内部可以 query world 和做 reactive control，但下一个 high-level decision 是在 program 执行完或出错时才做。ETA 则在 **每一个 world-changing Tool call 之后** 都强制 fresh observation，让 Planner 重新 decide。代价是 Planner calls 更多、inference latency 更高，但换来的是每一步都可 trace 和 control。

### 2.2 三个角色和 Trust Boundary

ETA 把 system 分成三个有明确 trust boundary 的角色：

**Agent（Planner）**：理解 task，维护 working memory，提出下一个 structured command $c_t$。Agent 有 intelligence 但 **没有 direct physical execution authority**。

**Interface**：验证 command structure, tool contract, provenance, authority, prerequisite evidence，然后 dispatch atomic action。Interface 是 Agent 和 World 之间的 gate，它决定 Agent "may" 做什么。

**World**：执行 actions。Simulator 暴露 ground-truth state, contacts, reward, termination；physical system 暴露 robot state，task relations 和 completion 需要 sensors, checker models, 或 human judgment。

这个 separation 的核心是 **intelligence 与 execution authority 分离**。Agent decides what it wants to do, Interface decides whether it may, World states what actually happened。这听起来像 over-engineering，但它解决了一个关键问题：防止一个强大的 coding model 通过 inspect hidden state 或 bypass physical action path 来 "solve" benchmark。比如 GPT-5.6 这样的强 model 如果能直接 read simulator 的 ground-truth object pose，它就不需要真的做 perception——这会让 benchmark 成绩 inflated 但 transfer 不到 real robot。

### 2.3 形式化定义

ETA 在 turn $t$ 产生 structured command $c_t$：

$$c_t = \pi_{\text{agent}}(g, o_t, m_t)$$

变量含义：
- $c_t$：turn $t$ 的 structured command（tool_call 或 response 类型）
- $\pi_{\text{agent}}$：agent policy（由 foundation model 实现）
- $g$：natural language goal
- $o_t$：turn $t$ 的当前 observation
- $m_t$：turn $t$ 的 working memory（包含历史 trajectories, obligations, skills）

Interface 对 $c_t$ 做 gate check，决定是否 dispatch：

$$a_t = \begin{cases} \text{dispatch}(c_t), & \text{gate}(c_t, o_t, m_t) = \text{pass} \\ \emptyset, & \text{otherwise} \end{cases}$$

变量含义：
- $a_t$：turn $t$ 实际执行的 atomic action
- $\text{gate}(\cdot)$：Interface 的 validation function，检查 structure, authority, provenance, prerequisite
- $\emptyset$：command 被拒绝，不执行

World 返回三元组 $(y_t, e_t, o_{t+1})$：
- $y_t$：Tool result（normalized output）
- $e_t$：environment receipt（trusted 的 reward/termination 信号）
- $o_{t+1}$：fresh observation

Planning 只在这三条 record 进入 working memory 后才 resume。这个 formulation 把 "decision → execution → verification" 强制成一个 atomic unit，每一步都 grounded in trusted evidence。

## 3. OpenETA 系统实现

OpenETA 是 ETA paradigm 的 open-source 实现。它有两个 configuration：

### 3.1 Full OpenETA

包含一个 broad Tool registry（44 个 Tools），覆盖五大 category：

| Category | Representative Tools | Role |
|----------|---------------------|------|
| Perception & localization | observe, retrieve_asset_reference, molmopoint, sam3, enhance_depth | 获取 observation，retrieve target reference，localize 和 segment |
| Geometry & manipulation planning | grasp_pose_estimate, anygrasp, contact_graspnet, graspgenx, anyplace | 生成 grasp/placement proposals |
| Safety & physical execution | ik_preview_check, obstacle_avoidance, move_to, follow_eef_trajectory, gripper_control | 检查 feasibility，change robot state |
| Environment & evidence | create_simulator_env, close_simulator_env, materialize_mcp_images | 管理 environment lifecycle |
| Agent support | save_memory, get_memory, compact_memory, python_exec, web_search, register_skill | 管理 memory, code, retrieval, skills |

每个 Tool 声明 side-effect class：`read_only`, `planning`, `bookkeeping`, `world_mutating`。这个 classification 决定 calls 是否可以 batched，是否需要 fresh observation afterward。这个设计很重要——它让 Interface 知道哪些 call 是 safe to batch（比如 perception calls），哪些必须 one-at-a-time with fresh observation（比如 move_to, gripper_control）。

几个关键概念：

**Tool**：host-registered atomic capability with stable parameter 和 return contracts。Agent 不能在 runtime rewrite 它们。

**Skill**：editable textual guidance describing how model should reason, check, recover。Skill **never executes a hidden action sequence**——Agent 必须仍然显式 select 每一个 atomic Tool。这个设计避免了 Voyager 那种 "skill as executable code" 的风险：在 physical world，一个 buggy skill 可能导致 robot 撞坏东西。

**AtomAction**：execution primitive 可以 physically change World（end-effector motion, trajectory following, gripper control）。它是 `world_mutating` Tools 的 physical subset。Skill text 不能 bypass AtomActions。

**Response 和 Soul**：Response 支持 dialogue, help requests, completion reports。Soul 描述 identity, behavioral boundaries, risk preferences，persist across tasks。两者 influence decisions 但 **没有 physical execution authority**。

### 3.2 Interface 的设计

Interface 做几件关键的事：

**Stable contracts and normalized results**：Planner 只能提交两种 structured command 类型：`tool_call` 和 `response`。Host runtime 定义 Tool names, parameters, handlers, side-effect declarations，Agent 不能 rewrite。每个 return 被 normalized 成 `ToolResult`，分别 record outputs, artifact references, state deltas, diagnostics, environment receipts。

**Execution gates and obligations**：Read-only 或 planning calls 可以在 bounded conditions 下 batched；`world_mutating` actions 一次执行一个。Interface blocks execution 如果 target unconfirmed, safety check fail, 或 prerequisite evidence missing。Explicit mask selection, grasp-candidate switching, post-action observation 成为 working memory 中的 **obligations**——unresolved obligations 约束后续 Tool calls。

**Trusted receipts and backend isolation**：Agent 看到 stable Tool semantics；Interface 通过 MCP (Model Context Protocol) adapter boundary 连接 simulator 或 robot。Low-level joint vectors, controller expansion, internal environment objects 对 planner hidden。Reward 和 termination 只在 host-attested provenance, execution ID, session ID match current turn 时才被接受——**一个普通 Tool handler 不能 mint official reward**。

这一点非常关键。想象一个 scenario：GPT-5.6 是一个非常强的 model，如果它看到 Tool 的 return format 知道 reward 长什么样，它可能在 return value 里伪造 reward 字段说自己成功了。Interface 的 trusted receipt 机制防止这种 attack——reward 必须来自 World 层的 attested receipt，不能来自 Tool handler 的 output。

### 3.3 World 层

World 由两个 MCP service family 实现：

**Simulator MCP**：own environment lifecycle, normalized observations, atomic motion 和 gripper execution, reward, termination。

**Real-robot MCP**：expose camera, robot-state, motion, gripper capabilities 通过 device-specific drivers 和 safety limits。

Interface 把两个 service map 到 same planner-facing Tool contracts，所以 Agent 不需要 simulator- 或 robot-specific control code。这个 separation 也是一个 security 和 evaluation boundary——Agent 不能直接 import simulator object, read privileged ground-truth state, call unregistered controller, 或 fabricate reward。

### 3.4 OpenETA for Codex（极简配置）

这是 paper 中最 striking 的 configuration。它只暴露 **三个 embodied Tools**：

**observe**：返回 live images。Agent request 需要的 views（third-person, wrist, orthographic X/Y/Z）。512×512 resolution。

**mark_point**：把 Agent 选的 2D point 转 3D World coordinate。两种 mode：
- *Multi-view mode*：Agent 在一个 orthographic view 选 pixel，Tool 返回另外两个 views 和 projected ray，Agent 在 ray 上选第二个 point，resolve 3D coordinate。
- *Single-view mode*：对 visible object surface 上的 point，Agent 选一个 pixel，Tool 返回 camera ray 的 first surface intersection。

**move_to**：移动 gripper 到 target position 和 orientation。Agent 用两个 vector（approach direction 和 jaw direction）specify orientation——比 raw rotation angles 更 intuitive。Tool 接受 absolute target 或 relative change。返回 gripper aperture（Agent 可以 check 是否 holds object）。Close 命令前 Tool 还 render target gripper pose preview，Agent 可以 inspect 后 adjust。

设计 motivation 来自一个 analogy：coding agents 通过 small set of stable operations（file access, editing, shell execution, search, version control）solve 很多 tasks。OpenETA for Codex 把这个 idea 应用到 embodied control——保持 physical interface small，把 task decomposition, target selection, spatial reasoning 都留给 multimodal Planner。

这个 configuration 在 LIBERO 上取得了非常 impressive 的结果，我下面详细讲。

## 4. 实验结果深度分析

### 4.1 Full OpenETA on LIBERO（fixed matrix）

LIBERO benchmark 包含 4 个 suite：Spatial, Object, Goal, Long/LIBERO-10，每个 10 tasks。Fixed matrix = 40 tasks × 10 seeds = 400 episodes。

Planner 是 GPT-5.6 Luna with medium reasoning effort。

| Suite | Success/total | Rate | Mean turns |
|-------|--------------|------|-----------|
| Spatial | 8/100 | 8.0% | 47.5 |
| Object | 26/100 | 26.0% | 31.2 |
| Goal | 21/100 | 21.0% | 36.8 |
| Long/LIBERO-10 | 1/100 | 1.0% | 58.5 |
| Overall | 56/400 | 14.0% | 43.4 |

这个 14% 的数字看起来 low，但 paper 非常 transparent 地分析了原因：

**Failure 分布**（mutually exclusive terminal labels）：

| Terminal class | Episodes | Share of failures |
|---------------|----------|-------------------|
| episode_timeout | 215 | 62.5% |
| unattended_ask_human | 66 | 19.2% |
| max_turns | 35 | 10.2% |
| status_report_without_reward | 24 | 7.0% |
| simulator_unknown_handle | 3 | 0.9% (infrastructure) |
| remote_episode_terminated_without_reward | 1 | 0.3% |

62.5% 的 failure 是 timeout。19.2% 是 agent 主动 ask human 但无人 respond（retained as failure, 不是 assisted result）。这说明 system 的 bottleneck 是 **long-horizon subgoal tracking, perceptual rechecking, placement relations, remaining-budget coordination**，而 failure 在 physical execution 本身相对少。

**Resource 对比**（success vs failure）：

| Stratum | Mean turns | Mean Tool calls | Mean wall time |
|---------|-----------|-----------------|----------------|
| Success | 30.7 | 30.7 | 1045.0s |
| Failure | 45.5 | 45.2 | 1866.6s |

Failure 用了更多 turns 和时间——它们确实在 try to recover，但最终 timeout。这 hint 了一个重要问题：current general-purpose models 还不能可靠地养成 "observe and correct after every physical action" 的 habit。

**18/40 tasks 至少成功一次**，但只有 1/40 在 Long/LIBERO-10 成功。Long-horizon multi-object tasks 是 weakest point。

### 4.2 OpenETA for Codex on LIBERO（130 tasks）

这个结果非常 striking。在 130 LIBERO tasks 上（4 个 standard suites + LIBERO-90），用 **只有三个 Tools** 的 configuration：

| Planner | PASS@1 | PASS@5 |
|---------|--------|--------|
| GPT-5.6 Luna | 21/130 | 62/130 (47.7%) |
| GPT-5.6 Terra | 58/130 | 83/130 (63.8%) |
| GPT-5.6 Sol | 92/130 | 117/130 (90.0%) |

Sol 在 PASS@1 就 solve 了 92/130 tasks。这个结果说明：**stronger general-purpose Planner 能用同样的 small physical interface 更 effectively**。

Suite-level breakdown（PASS@k）很有意思。看 Sol：

| Suite | P@1 | P@5 |
|-------|-----|-----|
| Spatial | 10/10 | 10/10 |
| Object | 8/10 | 10/10 |
| Goal | 8/10 | 8/10 |
| Long/LIBERO-10 | 7/10 | 10/10 |
| LIBERO-90 | 59/90 | 79/90 |

Sol 在 Spatial 和 Object 上几乎 perfect。Goal suite 有 2 个 tasks 始终 fail。LIBERO-90 还有 11 个 tasks fail。

为什么 Full OpenETA（44 Tools + Luna）只有 14%，而 OpenETA for Codex（3 Tools + Sol）有 90% PASS@5？这看起来 paradoxical。关键 difference 在两点：(1) Planner strength（Sol >> Luna）；(2) Full OpenETA 的 fixed matrix 是 single seed per cell，而 Codex 的 PASS@5 允许 5 seeds。但即便如此，Planner strength 的 effect 非常显著——这说明在 agentic paradigm 下，**Planner reasoning ability 是 dominant bottleneck**，而不是 Tool 数量。

Sol 相比 Terra 和 Luna 的 advantage 特别在 **precise grasp-point localization**——它产生 fewer empty grasps，并且更 frequently 用 move_to preview 在 act 之前 inspect。这 hint 了 spatial reasoning 和 lookahead 是 key differentiator。

### 4.3 这个结果对 VLA 路线的 implication

OpenETA for Codex 用三个 Tools + GPT-5.6 Sol 在 LIBERO 上达到 90% PASS@5，**without using any VLA or task-specific policy as a Tool**。这意味着对于 LIBERO 这类 manipulation benchmark，task-level planning + geometric tools 可能已经足够，VLA 的 action prediction capability 并非 necessary。

但要注意 caveats：LIBERO 是相对 structured 的 benchmark。对于 contact-rich, deformable object, dexterous manipulation 这类任务，geometric tools 可能不够，VLA 作为 specialist Tool 仍然有价值。Paper 在 Section 8 明确说未来 VLAs like CoRE-VLA 和 Gemini Robotics 可能作为 high-level ETA capabilities 被调用。

## 5. Constrained Agent Self-Evolution（最 interesting 的失败实验）

这部分是 paper 中 intellectually 最 honest 的 section。它 ask：能不能用 recorded experience 改进后续 behavior？答案是不能——**没有任何 candidate 通过 promotion gate**。

### 5.1 Promotion Rule

形式化定义：

$$\text{Promote}(u) = D(u) \land Q(u) \land R(u) \land H(u)$$

变量含义：
- $u$：proposed update（task-local Skill, exact-task playbook, 或 stage-local delta）
- $D(u)$：deterministic contract check（schema 和 invariant 验证）
- $Q(u)$：isolated review（在 isolated context 中 review candidate）
- $R(u)$：same-task replay（在原 task 和 seed 上 reproduce trusted success）
- $H(u)$：paired held-out evaluation（在 held-out tasks 上 show objective gain without regression）

Candidate 必须 pass 全部四个 gate 才能 promote。这个设计借鉴了 Reflexion, ExpeL, Voyager 的 experience reuse idea，但加了 physical system 必需的 checks。

### 5.2 三种 update form

**Task-local Skill**：edit guidance for full pick-and-place workflow。
- 结果：baseline 0/30, candidate 0/30。Candidate reach attachment 和 placement stage **less often**（5→1 attachment passes, 3→0 AnyPlace reaches）。Candidate 增加了 extra visual check 但没 fix failure，反而增加 turns 和 timeouts。

**Exact-task playbook**：从成功 trajectory 提取，只在 environment, suite, task index, task text 全 match 时 load。
- 结果：baseline 4/30, candidate 1/30。Candidate 用更多 turns（49.3 vs 42.8）和更多 timeouts（23 vs 18）。McNemar test $p=0.375$ 不 significant，但 candidate fail "objective gain without regression" gate。
- 这个结果很 counterintuitive——你从成功 trajectory 提取 playbook，结果反而 worse。原因可能是 playbook encode 了 incidental perception 或 contact 细节，在 held-out seeds 上不 generalize。

**Stage-local delta v1 和 v2**：对比成功和失败 trajectory，record 一个 symbolic trigger 和一个 action change。
- v1：baseline 0/3, candidate 0/3，both 都无法 reproduce source success，held-out evaluation 不 run。
- v2（contrastive）：唯一 valid pair，baseline 0/1, candidate 0/1，candidate 增加了 1 个 premature gripper-open violation，没有 change key failure。

### 5.3 为什么 self-evolution 失败？

Paper 的分析非常 insightful：

> "Most candidates add another visual check or recovery rule; they do not change the underlying perception or control Tools. These extra steps can increase Planner turns and timeouts without fixing the failure."

核心问题是：current update methods 只能在 **textual guidance level** 修改，不能修改 underlying perception 或 control Tools。如果 failure 的 root cause 是 grasp estimator 给了 bad candidate，你加一个 "check again" rule 没用——你只会 re-run 同一个 bad grasp estimator。

这 hint 了一个 deeper issue：textual experience 的 effect 是 indirect 的——它通过 influence model's next decision 来起作用，但 model 的 reasoning ability 和 Tool 的 capability 是 fixed 的。如果 bottleneck 在 Tool capability，textual experience 无法 fix。

Paper 的 conclusion 很 honest：

> "The evidence therefore supports the mechanism claim that experience should affect execution only after paired non-regression validation, not a performance-improvement claim."

也就是说，self-evolution 的 value 在于 **防止 bad experience 污染 capability library**，而不是 currently 能 improve performance。这是一个 safety-first 的设计哲学。

## 6. Sim2Real 的 honest disclosure

Paper 在 Sim2Real 部分 非常 transparent。它区分三个 evidence level：

1. **Interface integration**（current level）：code 和 interface contract hashes from clean release commit
2. **Primitive validation**：on-hardware primitive execution with calibration, safety limits
3. **Task validation**：complete task success with continuous video, rollout, verdict evidence

Current release 只达到 level 1。Retained 的两个 demonstration videos（sponge-to-tray 和 bell-pepper grasp）是 qualitative，不能 establish success rate。

Development testing 暴露了两个 Sim2Real bottleneck：

**Depth quality**：RealSense D435i (wrist) + L515 (third-person) 的 depth 完整性远不如 simulator depth。Depth-estimation enhancement 改善了 input 但没 reliably restore geometry sufficient for high-quality pose estimates。

**Motion control**：lateral-grasp target poses 有时 trigger acceleration-limit protective stops。Trace review suggest controller PD tuning 或 trajectory shaping 可能是 contributor，但 retained recordings 不 establish causal diagnosis。

这个 honest disclosure 很重要。它说明了 simulation 结果不能直接 imply real-robot capability。Paper 明确说 simulation 不能 establish real-robot safety, control frequency, emergency-stop behavior, calibration, payload limits。

## 7. 核心设计哲学的深层 intuition

让我总结一下 ETA 的核心 design philosophy，帮你 build intuition：

### 7.1 为什么 Agent 不能直接 control world？

在 digital agent 中，model 可以直接 call shell command, edit file, 这些都是 deterministic 的。但在 physical world，两个 key difference：

1. **Stochasticity**：physical action 的 outcome 是不确定的。你 grasp 一个 object，它可能成功可能失败，你必须在 action 后 observe。
2. **Safety**：physical action 可能造成 damage。你需要 gate 来 prevent unsafe actions。

ETA 通过 Interface 层解决了这两个问题。Interface 是 Agent 和 World 之间的 trust boundary——Agent 有 intelligence 但 no authority, Interface 有 authority 但 no intelligence, World 有 ground truth 但 no initiative。

### 7.2 为什么 one-action-at-a-time？

这个设计 cost 很高——它增加 Planner calls 和 inference latency。但 benefit 是 **每一步都 grounded in trusted evidence**。

考虑 alternative：如果 Agent 生成一段 action sequence 然后执行，中间某步失败，你需要 (1) detect failure, (2) rollback 到某个 checkpoint, (3) replan。这在 physical world 很难——你 grasp 错了 object，你没法 "undo" grasp，你只能 open gripper 重新来。

ETA 的 one-action-at-a-time design 让 failure detection 和 recovery 变得 natural：每一步都 observe，如果发现不对就立即 replan。这 trade latency for controllability。

### 7.3 为什么 Tool contracts 是 stable 的？

在 Voyager 中，model 可以生成新的 skill code 并 execute。这在 Minecraft 中 safe——worst case 你的 avatar 死了 respawn。但在 physical world，一个 buggy generated skill 可能导致 robot 撞坏东西或伤人。

ETA 的设计是：Tool contracts 由 host 定义，Agent 不能在 runtime rewrite。Agent 可以 select registered Tools，可以 compose 它们，但不能 masquerade as new atomic action。Skill 是 textual guidance，不 execute hidden action sequence——Agent 必须显式 select 每一个 atomic Tool。

这个设计限制了 flexibility 但保证了 safety。Paper 在 Section 6 的 self-evolution 实验中 show 了：即使 model propose 了 update，runtime 会 test 它在 reproduce 原始 success 和在 held-out tasks 上有 gain 后才 promote。这防止了 "lesson learned" gaining execution authority without evidence。

### 7.4 为什么 reward 必须是 trusted receipt？

这是 evaluation integrity 的核心。如果 Agent 能直接 declare success，strong model 可能 hallucinate success。Paper 中的 LIBERO evaluation 要求 success 只能来自 official positive LIBERO reward in trusted environment receipt。Reward 和 termination 只在 host-attested provenance, execution ID, session ID match current turn 时被接受。

这个 design 防止了几种 attack：
- Model 在 Tool return value 里伪造 reward 字段
- Model inspect simulator 的 hidden state 来 "cheat"
- Model bypass physical action path 直接 declare task complete

## 8. 和相关工作的 positioning

### 8.1 vs VLA (RT-2, OpenVLA, $\pi_0$, GR00T N1)

VLA 直接 map observation + language → action tokens。ETA 把 VLA 视为 **specialist Tool**——Planner 可以 invoke VLA 当 task 难以用 geometric Tools 表达时。但 Interface 仍然 check action 和 verify result。

### 8.2 vs World Action Models (DreamZero, DreamDojo, WAM)

WAM predict possible future observations 和 actions。ETA 把 WAM 视为 **prediction 或 plan-comparison Tool**——predicted futures 可以 rank plans 或 provide subgoals，但它们不是 deployed World 的 observations。Trusted physical evidence 仍然 determine 下一个 task-level decision。

Figure 2 很好地 summarizes 这个 division of labor：
- VLA proposes actions
- WAM predicts possible futures
- ETA coordinates heterogeneous capabilities around observed World evidence

### 8.3 vs Program-centric methods (Code as Policies, VoxPoser, RoboCodeX)

这些 systems generate 一段 program，program 内部可以 query world 和做 reactive control。下一个 high-level decision 在 program 执行完或出错时才做。

ETA 的 difference 是：在 **每一个 world-changing Tool call 之后** 都强制 fresh observation，让 Planner 重新 decide。Program-centric methods 是 "generate composition then repair/refine around execution"；ETA 是 "select and combine registered capabilities during execution"。

### 8.4 vs Reflexion, ExpeL, Voyager (experience accumulation)

这些方法在 digital world 证明了 experience reuse 的价值。ETA 借鉴了这个 idea 但加了 physical system 必需的 checks：new experience 只能在 reproduce task success 和 pass paired safety check 后 affect 后续 execution。Paper 的 self-evolution 实验显示这个 gate 非常 strict——no candidate passed——但这正是 safety-first 的体现。

## 9. Critical Analysis 和 Open Questions

### 9.1 Latency 问题

One-action-at-a-time design 的 cost 是 real 的。Full OpenETA 在 LIBERO 上 mean 43.4 turns per episode，success episode 平均 1045 秒 wall time。这在 real-time robot control 中可能 acceptable（robot manipulation 本来就慢），但在需要快速 reactive control 的场景（比如 dynamic object catching）可能太慢。

Paper 在 limitations 中承认 "current general-purpose models do not reliably adopt a habit of observing and correcting after every physical action"。这指向一个 fundamental tension：agentic paradigm 的 granularity 和 model 的 reasoning latency 之间的 trade-off。

### 9.2 Self-evolution 的 fundamental limitation

Self-evolution 实验的失败 hint 了一个 deeper issue：textual experience 只能 influence model's next decision，不能 change Tool capability。如果 bottleneck 在 grasp estimator 给 bad candidate，textual "check again" rule 没用。

这 suggest 未来的 self-improvement 可能需要 **co-evolve model reasoning 和 Tool capability**。比如，如果 Agent 发现 grasp estimator 在某种 object 上总 fail，它应该能 trigger 一个 retraining 或 fine-tuning process 来 improve grasp estimator。Paper 在 roadmap（Figure 7）中提到了 "connect trusted rollouts, training, regression tests, and redeployment in a self-improvement loop"，但这是 future work。

### 9.3 Real-robot evidence 的 gap

Current release 只有 interface-level integration，没有 formal real-robot task success。Paper 非常 honest 地说 development recordings 是 qualitative demonstrations。Depth quality 和 motion control 的 bottleneck 是 real engineering challenges。

这 suggest OpenETA 当前在 simulation 中证明的是 **paradigm 的 correctness**（protocol 能 work，evaluation 能 trusted），而不是 **real-world readiness**。从 simulation 到 real robot 的 transfer 仍然需要大量 engineering work。

### 9.4 Planner strength 的 dominant effect

OpenETA for Codex 的结果显示：从 Luna 到 Terra 到 Sol，同样的 3-Tool interface，PASS@5 从 47.7% 到 63.8% 到 90.0%。这说明在 agentic paradigm 下，**Planner reasoning ability 是 dominant bottleneck**。

这和 VLA 路线形成对比。VLA 的 scaling 主要靠更多 robot training data；ETA 的 scaling 主要靠更强 general-purpose model。这两条路线的 scaling law 完全不同。如果 general-purpose model 继续 improve，ETA paradigm 的 benefit 会越来越大——因为你只需要 expose 几个 stable Tools，强 model 就能 figure out 怎么 compose 它们。

### 9.5 LIBERO 的 representativeness

LIBERO 是相对 structured 的 manipulation benchmark。Object 位置已知大致范围，grasp points 相对 standard。ETA 在 LIBERO 上 90% PASS@5 很 impressive，但这不一定 transfer 到更 unstructured 的场景。

Paper 在 limitations 中承认 "bimanual coordination, dynamic contact, and mobile manipulation lack mature Tools and checkers"。这些是 embodied AI 的 frontier，ETA 当前还不 cover。

## 10. 对你（Karpathy）的 research intuition 的 relevance

作为 autoresearch 的作者，你对 "agent 通过 execution feedback 和 accumulated experience improve" 这条路线有 deep understanding。ETA paper 的几个 insight 可能对你 relevant：

1. **Physical world 的 trust boundary**：在 digital agent 中，file system 和 shell 是 deterministic 的，agent 可以 freely act。但在 physical world，你需要在 agent 和 world 之间加一个 Interface 层来做 gate 和 verify。这个 pattern 可能也适用于其他 stochastic environment（比如网络请求, database transaction）。

2. **One-action-at-a-time as runtime invariant**：这个 design principle 可能对任何 stochastic environment 的 agent 都有价值。它 trade latency for controllability，但在高风险场景下是 worth 的。

3. **Self-evolution 的 promotion gate**：ETA 的四 gate（D, Q, R, H）是一个非常 strict 的 experience validation 框架。在 digital agent 中，Voyager 的 skill library 用 simpler validation（code 能 execute 就 promote）。ETA 的 strictness 是因为 physical world 的 failure cost 高。但这个 framework 可能也适用于 digital agent 的 safety-critical applications。

4. **Trusted receipts as evaluation integrity**：ETA 的 trusted environment receipt 机制防止 model 自伪造 success。这在任何 agent evaluation 中都有价值——如果你 evaluate 一个 coding agent，你不应该让它自己 declare success，你需要 run tests 来 verify。

5. **Planner strength as dominant bottleneck**：在 agentic paradigm 下，general-purpose model 的 reasoning ability 是 dominant scaling axis。这和 VLA 的 data-driven scaling 是不同的 bet。如果 general-purpose model 继续 improve，agentic paradigm 的 leverage 会越来越大。

## 参考 links

- Paper PDF 和 project page: https://openmoss.ai/OpenETA/
- GitHub repo: https://github.com/OpenMOSS/OpenETA
- LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO
- ReAct (referenced): https://openreview.net/forum?id=WE_vluYUL-X
- Reflexion (referenced): http://papers.nips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html
- Voyager (referenced): https://openreview.net/forum?id=ehfRiF0R3a
- OpenVLA (referenced): https://proceedings.mlr.press/v270/kim25c.html
- $\pi_0$ (referenced): https://doi.org/10.48550/arXiv.2410.24164
- GR00T N1 (referenced): https://doi.org/10.48550/arXiv.2503.14734
- WAM paper (referenced, same group): https://doi.org/10.48550/arXiv.2605.12090
- CoRE-VLA (referenced, same group): https://arxiv.org/abs/2607.03693
- SAM 3 (referenced): https://doi.org/10.48550/arXiv.2511.16719
- AnyGrasp (referenced): https://doi.org/10.1109/TRO.2023.3281153
- cuRobo (referenced): https://doi.org/10.48550/arXiv.2310.17274
- Anthropic Claude robotics demo (referenced): https://www.anthropic.com/research/claude-plays-robotics
- Your autoresearch repo (referenced [4]): https://github.com/karpathy/autoresearch

---

总结一下我的核心 takeaway：ETA paper 的 contribution 主要在 **paradigm 层面**——它提出了一个 Agent-Interface-World 的三角色 protocol，通过 one-action-at-a-time runtime invariant 和 trusted receipts 把 physical world 的 stochasticity 和 safety 问题转化为可工程化的问题。OpenETA 的实验结果（尤其是 Codex 配置 90% PASS@5）show 了 agentic paradigm 在 LIBERO 上 competitive with 甚至超越 VLA 路线，而 self-evolution 的 honest failure 和 Sim2Real 的 honest disclosure 让 paper 的 claims 非常 credible。这条路线的 scaling 主要 bet on general-purpose model 的 reasoning ability continue to improve，这和 VLA 的 data-driven scaling 是 orthogonal 的 bet——未来可能是两者融合：strong Planner + VLA as specialist Tool + WAM as predictor。
