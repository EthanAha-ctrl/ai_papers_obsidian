---
source_pdf: Harness VLA Steering Frozen VLAs into Reliable Manipulation Primitives
  via Memory-Guided Agents.pdf
paper_sha256: 8c076133567919afd5420dc4a200a388314fec451cb6da1e8d4de93e1a1c0deb
processed_at: '2026-08-19T10:27:57-07:00'
target_folder: Robot-VLA/Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 Harness VLA

## 一句话说清楚

**VLA就像一个只会做特定动作的熟练工，你让它干训练过的活儿特别溜，但换个场景它就懵了。Harness VLA就是给它配一个"工头"（agent），工头负责看图纸、找零件、把工件摆到合适位置，然后喊VLA"过来干这步"，干完再让VLA走，工头接着安排下一步。**

---

## VLA的问题在哪

VLA比如 $\pi_{0.5}$、OpenVLA这些，训练时候看了一堆demo trajectory，学到的是"看到这个画面→手这么动"。问题是它把**语言理解、场景识别、长程规划、低层控制**全揉进一个model了。

这就像你训练一个厨师只学会了"看到案板上有西红柿→拿刀切→放锅里炒"，但如果有一天你跟他说"今天切黄瓜"，案板上也是黄瓜，他可能还是去切西红柿——因为他学的是"画面→动作"的硬绑定，语言那条通道其实是半死不活的 [2][4]。

paper里 Figure 3 展示了两个典型failure：
- OBJECT-PRO：instruction改了target object，画面差不多，RLinf还是重复训练时的behavior
- GOAL-PRO：layout换了，RLinf还是把东西往训练时的区域搬

更糟的是，VLA是stochastic的，物理交互也brittle。**一次抓空了，整个rollout就废了**，因为没有任何mechanism把它 localize 到 subtask。

---

## Coding Agent的问题在哪

另一条路是 Code as Policies [8] 这种，让 LLM 写代码调 API。好处是 LLM 擅长 compositional reasoning，能做 long-horizon 规划。坏处是那些 analytic primitive（IK transport、gripper open/close、base move）在精细操作上不行——你让它抓个不规则形状的物体、塞进抽屉、拧水龙头，它写出来的代码执行起来就拉胯。

之前的工作比如 ASPIRE [66] 想的解法是**不断扩张 skill library**——遇到不会的就写新 skill 加进去。但这在 deployment 时有问题：你怎么知道新写的 skill 是 safe 的、reusable 的、在 changed scene 下还 work？

---

## Harness VLA 的核心 idea

**别扩张 vocabulary，学会怎么用好固定的那几个 primitive。**

把 VLA 当成 primitive library 里的一个特殊成员——VLA_ACT。它和 MOVE_TO、SET_GRIPPER、RELEASE 这些 analytic primitive 平起平坐，都是 planner 可以调的 API。

关键 design choice：
1. VLA_ACT 只负责 contact-rich 的 local 操作（抓取、放置、按按钮、拧东西）
2. Analytic primitives 负责非接触的所有结构化运动（transport、staging、导航、release）
3. Planner 负责语义 grounding、场景 re-binding、long-horizon composition
4. VLA 可以 retry——失败了大不了 re-stage 再来一次

用 Figure 2 的话说：**deployment perturbation 把 task configuration 推到 VLA 训练分布之外，direct rollout 会试图直接 bridge 这个 gap 然后失败。Harness VLA 把 task 拆成 analytic primitive 能 traverse 的部分 + VLA 训练分布仍 informative 的 local contact-rich region。**

---

## 跑起来的样子

整个系统是个 turn-based 的 REPL（Read-Eval-Print Loop）：

```
planner 看 observation + task + memory
  → emit 一个 JSON command（比如 {"action":"vla_act","prompt":"grasp the bowl",...}）
  → environment worker 执行这个 primitive
  → 写回新的 observation + log
  → planner 看新状态，决定下一步
  → 循环直到 success predicate 满足或 budget 耗尽
```

这个 loop 的细节很 software engineering agent 风格（受 SWE-agent [15]、OpenHands [17] 启发）：
- planner 写 `command.json`
- worker 执行完写 `state_NN.json` + `log_NN.json` + RGB-D + `done_NN.flag`
- planner 等这些文件再决定下一步
- **planner 完全不 access simulator 内部 state、object pose 等 oracle 信息**——只能从 RGB + depth map 自己 localize

这就像给 agent 一个真实的 partial observation setting，逼它 ground 所有 spatial reasoning 在 perceptual input 上。

---

## 两个 Memory Module

### Task Specific Memory

一个 task 在一个 reference seed $s_0$ 上探索出成功方案后，存两个文件：

**JSON audit**（记录"为什么这么干 work"）：
```json
{
  "task": "put the black bowl on the wooden tray",
  "strategy": "use VLA for grasping, then analytic transport and release",
  "avoid": ["do not reuse reference xyz values", 
            "verify placement with the benchmark success signal"]
}
```

**JSONL trace**（记录"干了什么、什么顺序"）：
```jsonl
{"action":"vla_act","prompt":"grasp the black bowl","max_chunks":2}
{"action":"move_to","xyz":[0.12,-0.08,0.92],"gripper":null}
{"action":"release"}
```

**关键 insight**：trace 是 skeleton 不是 trajectory replay。坐标是 reference scene 的，deployment 时必须从当前 RGB-D 重新 localize。这就像你记住"先抓碗、再搬到托盘上方、松手"这个 structure，但碗在哪、托盘在哪每次都得现场看。

### Global Memory

跨 task 的通用知识，比如：
- 用 VLA 做 contact-rich phase，稳定 grasp 后用 analytic 做 transport
- gripper 关了但东西没跟着动 = empty grasp，要 re-localize 再 retry
- 别只看视觉接近就判断 success，要查 benchmark 的 success signal

---

## 三阶段生命周期

**Bootstrapping Phase**（只用 seed 0）：
- Agent 有 RESET primitive 可用
- 充足 wall-clock budget
- 反复 trial and error 找到 working composition
- 成功后存 Task Specific Memory + 提取 Global Memory

**Deployment Phase**（held-out seeds）：
- RESET 禁用
- Budget 紧
- Retrieve Task Specific Memory，从 live observation 重新 ground 执行
- 这阶段的 success rate 就是 paper 报的数字

**Zero-shot Setting**（额外 ablation）：
- 完全不 retrieve Task Specific Memory
- 只靠 planner reasoning + Global Memory
- Table 5 显示 LIBERO-Pro GOAL 上 instruction-redirection 79.0% vs Cap-X 16.8%
- **证明 memory 是 enhancer，planner reasoning 本身就能 synthesize 有用 composition**

---

## 实验数字

### Standard LIBERO（Table 2）

Harness VLA (CC) 96.0%，和 frozen RLinf 95.3% 持平。OBJECT suite 100%。**证明 decomposition 没 sacrifice in-distribution 能力**。

### LIBERO-Pro（Table 3）— 这是重头戏

Perturbation 下所有 end-to-end VLA 崩盘：
- OpenVLA: 0.0%
- $\pi_0$: 0.3%
- $\pi_{0.5}$: 11.0%
- RATS（最强 prior baseline）: 43.8%
- RLinf direct: 50.0%
- **Harness VLA (CC): 82.4%**

**比 RATS +38.6 pp**，比 direct RLinf +32.4 pp。Gain 来自 architecture，不来自 VLA backbone 本身。

### RoboCasa365（Table 4）— 厨房场景

- RLDX-1 direct: 30.0% overall
- **Harness VLA (Codex): 55.4% overall**
- **+25.4 pp**

Composite-Unseen 还是难点（13.8% / 15.0%），但比 baseline 翻倍多。

### RoboTwin C2R（Table 6）— 双臂

- LingBot-VLA direct: 50.4%
- **Harness VLA (CC): 58.4%**
- **+8.0 pp**，无 finetune

---

## 三个 Key Finding 讲透

### Finding 1: Planner 负责 Semantic Re-grounding

VLA 的 language channel 基本是 vestigial 的 [2][4]——训练时 instruction 和 trajectory 关联，但部署时 instruction 变了它不一定 follow。

Harness VLA 把 semantic reasoning 显式放在 planner 层：
1. $\Pi$ 解析 task description
2. 从 live RGB-D resolve 当前 contact target
3. Analytic primitives 做 staging + repositioning
4. VLA_ACT 只在 local contact-rich phase 被 invoke

**Intuition**：你跟 VLA 说"去抓那个"——那个是什么、在哪，由 planner 搞定，VLA 只管"到这个 local region 做这个动作"。

### Finding 2: Staging + Retry = 给 VLA 一个 Sandbox

VLA 是 stochastic 的，physical contact 是 brittle 的，单次失败不必废掉整个 rollout。

Planner 把 VLA 当成可 retry 的 local attempt：
```
1. 用 analytic primitive 把 robot 摆到 pre-contact pose
2. invoke VLA_ACT
3. 检查 contact post-condition 满足没
4. 没满足就换 approach pose / viewpoint / staging 再来
```

Figure 4 的实验特别 convincing：cap 每个 episode 的 VLA invocation 次数，发现 curve 快速上升后 saturate——**少量 planner-selected invocation 已经超过 frozen baseline，更多 invocation 在 long-horizon task 上进一步提升**。

**Intuition**：这就像你让一个学徒工去试某个 tricky 的焊接，他站位不对焊不上，你让他挪一挪再试，而不是他第一次失败就开除他整个项目。

### Finding 3: Analytic Primitives 隔离 Non-contact Execution

Analytic primitives 不 replace VLA 在 contact-rich 上的能力。它们做的是 contact 周围的 non-contact structure：
- Free-space transport
- Pre-contact staging
- Wrist/base reorientation
- Retreat
- Post-contact repositioning

这让 VLA 不必承担 long-horizon composition、scene-level grounding、中间运动的所有责任。

Figure 6 的 attribution 分析：
- **LIBERO Pro-family**：大多 task 在 analytic transport/release/reposition 后完成（VLA 建立稳定 contact 后剩下的是搬运+松手）
- **RoboCasa365 + RoboTwin C2R**：更多 task 在 VLA primitive 内完成（terminal contact-rich operation 多，比如拧水龙头、双臂 handover）

**Intuition**：tabletop 任务大致是"抓起来→放下"，contact 完了剩下是搬运。厨房和双臂任务有更多 terminal 的精细操作，所以 VLA 占比更高（Table 18：LIBERO VLA 占 15.8%，RoboTwin 47.4%，RoboCasa365 35.3%）。

---

## Primitive Usage 模式（Table 18）

| Primitive | LIBERO | RoboTwin C2R | RoboCasa365 |
|-----------|--------|--------------|-------------|
| MOVE_TO | 61.8% | 40.9% | 38.7% |
| NAVIGATE_TO | — | — | 9.0% |
| SET_GRIPPER | 11.2% | 4.2% | 4.8% |
| RELEASE | 8.2% | 7.4% | 1.0% |
| MOVE_BASE | — | — | 10.4% |
| **VLA_ACT** | **15.8%** | **47.4%** | **35.3%** |

LIBERO 主要是 tabletop，VLA 做完 grasp 后大量 analytic transport。RoboTwin 是双臂，handover 和 bimanual grasp 多，VLA 占比最高。RoboCasa365 居中，加了 mobile base 的 navigation。

---

## VLA Backend 三件套

| Benchmark | VLA Backend | Architecture |
|-----------|------------|--------------|
| LIBERO / LIBERO-Pro | RLinf (frozen $\pi_{0.5}$-SFT) [22] | Two-stage: high-level subtask $\hat{\ell}$ → low-level action chunk $a_{t:t+H}$ |
| RoboCasa365 | RLDX-1 [23] | Multi-Stream Action Transformer + Qwen3-VL 8B + flow-matching diffusion |
| RoboTwin C2R | LingBot-VLA [24] | Qwen2.5-VL + Mixture-of-Transformers + chunked decoding (T=50) |

关键：**这三个完全不同的 VLA 被 Harness VLA 抽象成统一的 VLA_ACT 接口**。说明这 framework 是 architecture-agnostic 的——不管你 VLA 内部是 flow matching 还是 FAST tokenization 还是 MoT，都可以被"harness"成 contact-rich primitive。

---

## 和 Related Work 的差别

### vs ASPIRE [66]
ASPIRE 不断扩张 skill library，遇到新情况就加新 skill。Harness VLA 故意保持 vocabulary fixed，研究怎么用好固定的那几个。

### vs Voyager [16]  
Voyager 有 persistent memory 但是在 Minecraft 这种 digital sandbox 里，没有 VLA-backed 的 physical contact primitive。Harness VLA 把 memory-centric design 和 VLA contact specialist 结合起来。

### vs Code as Policies [8]、RoboCodeX [61]
这些用 LLM 写代码调 API，但 API 都是 analytic 的，精细操作不行。Harness VLA 加了一个 VLA 作为 contact-rich API，同时用 memory 学习怎么 orchestrate。

---

## 我的 Intuition 总结

这 paper 在我看来是关于 **"承认 VLA 的 locality"**。

VLA 的 competence 是 local 的、contact-rich 的、image-conditioned 的。你硬要它做 long-horizon composition、做 semantic re-grounding、做 layout shift 的 robustness，它就崩——因为这些不是它训练时学的。

Harness VLA 的哲学是：**承认 VLA 只擅长 local contact，把它 isolate 到那个 region，把其他所有事交给 planner + memory + analytic primitives**。

这比"训一个更大的 VLA"或"收集更多 data"要 elegant 得多。你不需要 retrain 任何东西，frozen VLA 直接拿来用，通过 architecture-level 的 decomposition 把它的 operating range 从训练分布扩展到 deployment perturbation space。

而且这 framework 的 ablation 特别 clean：
- Zero-shot 没 memory 也 work（planner reasoning 本身够强）
- Cap VLA invocation 次数的 curve 证明 retry 是关键
- Primitive usage statistics 证明 division of labor 是 emergent 的，不是硬编码的

---

## Limitations

Paper 自己承认：
1. **Planner 和 VLA 之间是 open loop**——没有 gradient flow，没有 joint optimization
2. **没有 RL fine-tuning**（提到 GRPO 作为 future work）
3. **没有 fine-grained image captioning**，cluttered long-horizon 场景的 structural reasoning 受限
4. 没结合 automatic skill discovery（ASPIRE 那条路）

---

## 相关 Links

- **Project Page**: https://harnessvla.github.io/
- **LIBERO**: https://libero-project.github.io/
- **LIBERO-Pro**: https://arxiv.org/abs/2510.03827
- **RoboCasa365**: https://arxiv.org/abs/2603.04356
- **RoboTwin**: https://github.com/RoboTwin-platform
- **RLinf**: https://arxiv.org/abs/2509.15965
- **RLDX-1**: https://arxiv.org/abs/2605.03269
- **LingBot-VLA**: https://arxiv.org/abs/2601.18692
- **OpenVLA**: https://arxiv.org/abs/2406.09246
- **$\pi_0$**: https://physicalintelligence.company/blog/pi0
- **$\pi_{0.5}$**: https://arxiv.org/abs/2504.16054
- **Code as Policies**: https://code-as-policies.github.io/
- **Voyager**: https://arxiv.org/abs/2305.16291
- **SWE-agent**: https://swe-agent.com/
- **OpenHands**: https://github.com/All-Hands-AI/OpenHands
- **ASPIRE**: https://arxiv.org/abs/2607.00272
- **Gemini Robotics**: https://arxiv.org/abs/2503.20020
- **GR00T N1**: https://arxiv.org/abs/2503.14734

---

最后一句话总结：**Harness VLA 教你怎么用一个 frozen 的、不完美的 VLA，通过 agent + memory 把它"驯化"成可靠的 manipulation primitive，而不需要 retrain 任何东西。** 核心 insight 是 asymmetric decomposition——让 VLA 只做它擅长的 contact-rich local operation，其他全部交给 planner + analytic primitives + memory。这比 scale up VLA 更优雅，也更 sample efficient。

---

# Harness VLA：把Frozen VLA"驯化"成Reliable Manipulation Primitive

## 1. 核心问题的Asymmetry Intuition

End-to-end VLA models存在一个尖锐的asymmetry：它们在**contact-rich visuomotor control**上strong（irregular grasping, fixture actuation, tight tolerance placement），但在**language grounding、long-horizon composition、out-of-distribution scenes**上dramatically degrade [1]。

为什么会这样？VLA train在in-distribution task trajectories上。Deployment perturbations包括四类：
- **Semantic retargeting**：instruction指向不同target object
- **Goal re-binding**：goal predicate改变
- **Spatial-layout shifts**：object layout变化
- **Unstable local contacts**：stochastic物理交互失败

经典failure mode：policy在perturbation下"重复training-time behavior"，即使instruction或scene binding已变 [1][2][3]。同时，**一个unstable contact failure就能derail整个monolithic rollout**——因为没有mechanism把failure localize到subtask。

LLM coding agents提供complementary reasoning（Code as Policies, ProgPrompt, RoboCodeX等 [4][5][6]），但purely analytic primitives在irregular grasping、constrained placement、articulated-object interaction上挣扎。

Harness VLA的key insight：**不扩张primitive library，让agent learn how to orchestrate fixed vocabulary**。把VLA从monolithic trajectory policy转换成reusable contact specialist。

---

## 2. Framework架构解析

### 2.1 Problem Formulation

Environment $\mathcal{E}$由rigid-body physics engine驱动（MuJoCo via Robosuite）。每个timestep $t$，environment exposes：

$$o_t = (I_t^{rgb}, I_t^d, q_t)$$

变量解释：
- $o_t$：multimodal observation tuple at timestep $t$
- $I_t^{rgb}$：agent-view RGB image（qualitative scene reasoning）
- $I_t^d$：co-aligned metric depth map（precise spatial localization）
- $q_t$：robot proprioceptive state（end-effector pose + gripper state的concatenation）

Task定义：natural-language description $\ell$ + binary completion predicate $\mathcal{G}$（sparse success signal，only at episode termination）。

### 2.2 Agentic Execution Loop

Turn-based autoregressive interaction between planner $\Pi$ 和 physics engine：

```
while not G(o_t) and step_budget > 0:
    context = retrieve(TaskSpecificMemory, GlobalMemory)
    c_t = Π(o_t, ℓ, context)  // emits JSON invocation
    o_{t+1}, q_{t+1} = engine.execute(c_t)
    t += 1
```

关键：**all low-level control mechanisms**——包括frozen VLA $f_\theta$ 和所有deterministic operational-space controllers——unified成一个predefined primitive library $\mathcal{P}$。VLA不被当作separate hierarchical tier，而是和analytic controllers平起平坐的primitive。

### 2.3 两阶段Lifecycle

**Exploratory Bootstrapping Phase**（单reference seed $s_0$）：
- Agent唯一access RESET primitive
- Generous wall-clock budget
- 探索staging orders、pre-contact poses、VLA ACT invocation timings、early-return thresholds
- 成功后serialize成JSONL，提取heuristics到Global Memory

**Deployment Evaluation Phase**：
- RESET禁用
- Step budget显著缩短
- Retrieve JSONL trace from Task Specific Memory
- Dynamic grounding with live RGB-D
- Reference Global Memory执行
- 这阶段performance直接构成reported benchmark results

### 2.4 Primitive Library (Table 1)

固定vocabulary，分两大类：

**Analytic Primitives**（deterministic, model-based, no training data）：

| Primitive | Type | Role |
|-----------|------|------|
| MOVE_TO | Composite | End-effector到world-frame Cartesian target via embedded solver |
| MOVE_POSE | Composite | Co-vary position + pose (e.g., pitch for reach-limited configs) |
| ROTATE_WRIST | Atomic | Wrist-yaw set-point holding current position |
| ROTATE_PITCH | Atomic | Wrist-pitch set-point holding current position |
| SET_GRIPPER | Atomic | Drive gripper open/close for fixed steps |
| RELEASE | Atomic | Open gripper under release post-condition |
| NAVIGATE_TO | Composite (RoboCasa365) | Mobile base到world-frame location |
| MOVE_BASE | Atomic (RoboCasa365) | Open-loop local base-velocity set-point |

**VLA Primitive**：

```jsonl
{"action": "vla_act", "prompt": <str>, "max_chunks": <int>, "stop": <predicate>}
```

VLA_ACT binds frozen VLA $f_\theta$，emit action chunks直到stop predicate $\tau$ 满足或chunk budget耗尽。$\tau$可以是：
- lift-and-grasp condition
- contact-state condition
- benchmark predicate
- chunk budget

Crucially：**planner不能在deployment时invent新primitive**。vocabulary evaluation前fixed。

---

## 3. Memory Modules：Structure Prior不是Trajectory Replay

### 3.1 Task Specific Memory

包含两个互补文件：

**JSON audit summary**（why）：
```jsonl
{"task":"put the black bowl on the wooden tray",
"success":true,
"strategy":"use VLA for grasping, then analytic transport and release",
"avoid":["do not reuse reference xyz values",
"verify placement with the benchmark success signal"]}
```

**JSONL procedural trace**（what + when）：
```jsonl
{"action":"vla_act","prompt":"grasp the black bowl","max_chunks":2}
{"action":"move_to","xyz":[0.12,-0.08,0.92],"gripper":null}
{"action":"release"}
```

Key principle：**transfer结构不transfer坐标**。Spatial arguments是reference-scene bindings，deployment时必须re-ground from current observation。

### 3.2 Global Memory

Cross-task knowledge，task-independent：

```
Success rule: Use VLA primitives for contact-rich phases such as 
irregular grasping or fixture interaction. After a stable grasp, 
prefer analytic motion for long transport and precise placement.

Failure model: If gripper closes but object does not move with 
end effector, treat as empty grasp. Re-localize and re-stage 
before retrying.

Failure model: Do not terminate from visual proximity alone. 
Check benchmark success signal and latest execution record.
```

### 3.3 Iterative Memory Construction

Memory在interaction中refine而非accumulate：
- 每个primitive后classify outcome：progress / recoverable failure / unrecoverable failure
- 后续attempt可替换procedural trace（若shorter or more reliable）
- Earlier failure observations保留作为future planning constraints

---

## 4. Experiments：四个Benchmark Families

### 4.1 Setup

| Benchmark | Tasks | Eval seeds/task | Reported rollouts |
|-----------|-------|-----------------|-------------------|
| LIBERO | 40 | 10 | 400 |
| LIBERO-Pro | 80 | 10 | 800 |
| RoboCasa365 | 50 | 10/5/5 | 340 |
| RoboTwin C2R | 50 | 5 | 250 |

VLA ACT backends：
- LIBERO/LIBERO-Pro: **RLinf**的pi05 libero130 fullshot $\pi_{0.5}$-SFT checkpoint [22]
- RoboCasa365: frozen **RLDX-1** [23]
- RoboTwin C2R: post-trained **LingBot-VLA** [24]

Planner backends：Codex + Claude Code (CC)

### 4.2 Standard LIBERO (Table 2)

| Method | Spatial | Object | Goal | LIBERO-10 | Overall |
|--------|---------|--------|------|-----------|---------|
| OpenVLA [2] | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| $\pi_0$ [3] | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| RLinf | 99.0 | 96.0 | 97.0 | 89.0 | 95.3 |
| AtomVLA [25] | 96.4 | 99.6 | 97.6 | 94.4 | 97.0 |
| **Harness VLA (CC)** | 97.0 | 100.0 | 94.0 | 93.0 | **96.0** |

Intuition：Harness VLA在in-distribution上保持competitive，OBJECT suite甚至100%。这证明decomposition没有sacrifice原VLA的local capability。

### 4.3 LIBERO-Pro (Table 3) — Perturbation Robustness

| Method | Spat-T | Spat-S | Obj-T | Obj-S | Goal-T | Goal-S | L10-T | L10-S | Overall |
|--------|--------|--------|-------|-------|--------|--------|-------|-------|---------|
| OpenVLA | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| $\pi_0$ | 0.0 | 0.0 | 0.0 | 2.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.3 |
| $\pi_{0.5}$ [4] | 1.0 | 20.0 | 1.0 | 17.0 | 2.0 | 38.0 | 1.0 | 8.0 | 11.0 |
| RATS [13] | 31.0 | 29.0 | 63.0 | 61.0 | 36.0 | 43.0 | / | / | 43.8 |
| RLinf (direct) | 42.0 | 59.0 | 71.0 | 78.0 | 45.0 | 42.0 | 49.0 | 14.0 | 50.0 |
| **Harness VLA (Codex)** | 81.0 | 69.0 | 94.0 | 91.0 | 75.0 | 66.0 | 52.0 | 49.0 | 72.1 |
| **Harness VLA (CC)** | 94.0 | 80.0 | 88.0 | 90.0 | 87.0 | 87.0 | 71.0 | 62.0 | **82.4** |

**改进 +38.6 percentage points over RATS**（最强prior baseline）。

Key observation：direct RLinf只有50.0%。**Gain不是来自VLA backbone本身**，而是来自agentic decomposition。Planner负责re-bind targets、re-stage scene、route through analytic control，VLA只做local contact-rich operation。

### 4.4 RoboCasa365 (Table 4) — Household Kitchen

| Method | Atomic-Seen | Composite-Seen | Composite-Unseen |
|--------|-------------|----------------|------------------|
| RLDX-1 [23] (direct) | 60.0 | 21.3 | 5.0 |
| WorldDreamer [27] | 66.3 | 26.7 | 9.0 |
| $\pi_{0.5}$ | 39.6 | 7.1 | 1.2 |
| $\pi_0$ | 34.6 | 6.1 | 1.1 |
| **Harness VLA (Codex)** | 91.6 | 56.3 | 13.8 |
| **Harness VLA (CC)** | 79.4 | 47.5 | 15.0 |

**改进 +25.4 percentage points over RLDX-1**。Composite-Unseen上仍是难点（13.8%, 15.0%），但比baseline +8.8/+10.0 percentage points。

### 4.5 RoboTwin C2R (Table 6) — Bimanual Clean-to-Randomized

| Benchmark | GR00T-N1.7 | $\pi_{0.5}$ | StarVLA | LingBot-VLA | Harness VLA (Codex) | Harness VLA (CC) |
|-----------|------------|-------------|---------|-------------|---------------------|------------------|
| RoboTwin C2R | 20.7 | 47.9 | 10.6 | 50.4 | 58.0 | **58.4** |

LingBot-VLA baseline 50.4% → Harness VLA 58.4%，**+8.0 percentage points**，无finetune。

### 4.6 Zero-shot LIBERO-PRO GOAL (Table 5) — No Task Specific Memory

| Setting | Method | Average |
|---------|--------|---------|
| Pos (S) | Cap-X | 25.6 |
| Pos (S) | Harness VLA (CC) | **31.0** |
| Task (T) | Cap-X | 16.8 |
| Task (T) | Harness VLA (CC) | **79.0** |

**这组数据尤为重要**：zero-shot（无Task Specific Memory retrieval）下Harness VLA仍大幅超Cap-X，证明**planner reasoning本身可synthesize有用compositions**，memory是enhancer不是crutch。

---

## 5. Key Findings：三个机制分析

### Key Finding 1: Planner-level Semantic Re-grounding

Figure 3的terminal-state frames显示：在task-perturbed OBJECT-PRO case，visual scene相似但instruction redirect target，RLinf重复standard behavior；在swap-perturbed GOAL-PRO case，layout变化但RLinf仍move object到training-time region。

Harness VLA的fix：
1. $\Pi$ parses task description
2. Resolves current contact target from live RGB-D
3. Analytic primitives for staging + repositioning
4. VLA ACT只做local contact-rich phase

**Intuition**：semantic和scene-level reasoning上提到planner，frozen VLA只负责execute contact-rich operation under planner-provided binding。

### Key Finding 2: Planner-staged VLA Invocation + Retry

VLA不是one-shot black box。每次call是planner-chosen local attempt，staging决定frozen policy是否成功：

```
loop:
    stage_to_feasible_pre_contact_config()  # analytic
    invoke_vla(prompt, max_chunks, stop_pred)  # VLA
    if not contact_post_condition_satisfied():
        reframe(approach_pose, viewpoint, local_staging)
        continue
    else:
        break
```

Figure 4的实验：cap maximum VLA invocations per episode。
- 蓝色虚线：frozen-policy baseline
- 灰色虚线：full Harness VLA
- Curve：few VLA calls已超frozen baseline，然后saturate

**Intuition**：VLA sparse使用，但re-stage + retry ability是robustness核心。这相当于给VLA一个"sandbox"——在controlled local region尝试，失败回到analytic control再来。

### Key Finding 3: Analytic Primitives Isolate Non-contact Execution

Analytic primitives处理surrounding non-contact structure：
- Free-space transport
- Pre-contact staging
- Wrist/base reorientation
- Retreat
- Post-contact repositioning

VLA ACT保留给contact-rich phases：grasping, constrained placement, button pressing, faucet turning, drawer manipulation, coffee-machine operation。

Figure 6的task completion attribution：
- LIBERO Pro-family：mostly finished by analytic primitives（transport, release, repositioning after contact established）
- RoboCasa365 + RoboTwin C2R：更多terminal contact-rich operations（fixture actuation, constrained placement, bimanual interaction）

**Intuition**：analytic primitives organize task around contact-rich phases，VLA负责learned visuomotor control需要的phase。这是division of labor，不是replacement。

### Primitive Usage Statistics (Table 18)

| Primitive | LIBERO | RoboTwin C2R | RoboCasa365 |
|-----------|--------|--------------|-------------|
| MOVE_TO | 6263 (61.8%) | 685 (40.9%) | 3004 (38.7%) |
| NAVIGATE_TO | — | — | 701 (9.0%) |
| SET_GRIPPER | 1137 (11.2%) | 71 (4.2%) | 371 (4.8%) |
| RELEASE | 831 (8.2%) | 124 (7.4%) | 76 (1.0%) |
| MOVE_BASE | — | — | 808 (10.4%) |
| **VLA_ACT** | **1598 (15.8%)** | **794 (47.4%)** | **2746 (35.3%)** |
| **Analytic total** | **84.2%** | **52.6%** | **64.7%** |

模式清晰：LIBERO以analytic为主（tabletop structure），RoboTwin C2R的VLA share最高（bimanual contact-heavy），RoboCasa365居中（mobile staging + longer horizon）。

---

## 6. File-Mediated REPL Protocol (Appendix A)

Synchronous file-mediated REPL architecture：
- Long-running environment worker owns live simulator state
- Planner只通过serialized JSON invocations + persisted observations交互
- **Planner不access privileged simulator state, object poses, controller internals**

文件类型：

| File | Role |
|------|------|
| `command.json` | Planner-issued primitive invocation $c_t$ |
| `state_NN.json` | Step-indexed task language, proprioception, success signal |
| `RGB-D / world-map files` | Perceptual evidence for semantic ID + metric re-grounding |
| `log-NN.json` | Diagnostic record (accepted command, status, step counts, failure info) |
| `done_NN.flag` | Synchronization signal |
| `Task Specific Memory trace` | Append-only JSONL procedural memory |
| `Global Memory` | Cross-task success rules + failure models |

Index NN单调递增。Initial observation在NN=00；每个executed primitive产生下一个indexed state。Auditability without oracle access。

---

## 7. Prompt Design (Appendix E)

Shared prompt template + benchmark-specific slots。11个modules：

1. **Role and success signal**：closed-loop controller，optimize $\mathcal{G}$不是visual guess
2. **Perception isolation**：禁止simulator internals，localize from RGB + world maps
3. **File-based REPL**：write one JSON command → wait → read refreshed artifacts → iterate
4. **Primitive vocabulary**：allowed schemas + controller semantics
5. **VLA division of labor**：VLA for contact-rich, analytic for grounding/staging/transport/release/recovery
6. **Task language**：authoritative from state file
7. **Task Specific Memory + Global Memory**：JSON audit for why, JSONL for what+when
8. **Closed-loop recovery**：inspect state/logs/RGB/world maps after every primitive
9. **Budget, reset, termination**：track budget, no reset in strict eval
10. **Output discipline**：write audit JSON + command-trace JSONL
11. **Operating loop**：read → localize → execute → observe → recover → repeat

### Perception Isolation Rule（关键）

```
1. Identify entity from RGB
2. Pick pixels on visible surface
3. Index matching precomputed world map at those pixels
4. Sample multiple stable pixels, use robust statistic (median)
5. Avoid rims, edges, gaps, holes, reflections, background
6. Re-localize after robot/camera/object/base/grasp state changes
```

**Intuition**：这forcing planner ground所有spatial reasoning在perceptual inputs，而非hidden state variables。是partial observation的现实setting。

---

## 8. VLA Backend Details (Appendix D)

### RLinf (LIBERO/LIBERO-Pro)

- 基于$\pi_{0.5}$ architecture
- Two-stage inference：high-level subtask $\hat{\ell}$ prediction → low-level action chunk generation
- Low-level policy produces continuous action chunks $a_{t:t+H}$ via FAST tokenization或flow-based continuous modeling
- Supervised fine-tuned on LIBERO-130
- Frozen evaluation

### RLDX-1 (RoboCasa365)

- Multi-Stream Action Transformer (MSAT)核心
- VLM基于Qwen3-VL 8B + cognition tokens
- Memory module aggregate historical cognition features
- Flow-matching diffusion transformer for continuous action prediction
- Velocity field maps noisy trajectories → clean sequences via iterative denoising
- Chunk-wise inference with partial chunk execution for stable closed-loop control

### LingBot-VLA (RoboTwin C2R)

- Qwen2.5-VL backbone + Mixture-of-Transformers (MoT)
- Separate vision-language reasoning + action generation pathways
- Coupled via shared self-attention
- Action expert module for continuous control signals
- Flow-matching formulation + chunked action decoding (T=50)
- Pre-trained on 9 embodiment teleoperation data + SFT on RoboTwin

Post-training hyperparameters (Table 14)：
- Optimizer: AdamW
- LR: $1 \times 10^{-4}$, Vision encoder LR: $1 \times 10^{-6}$
- Loss: L1 Flow Matching (L1_FM)
- Chunk size: 50, Max seq length: 2048, Flow steps: 10
- Max action/state dim: 75
- Global batch size: 256
- Image: $224 \times 224$, views: top + wrist left + wrist right
- FSDP2 distributed training

---

## 9. 与Related Work的Positioning

### VLA Models谱系
RT-1 → RT-2 → OpenVLA → $\pi_0$ → $\pi_{0.5}$ → GR00T → Gemini Robotics。3D-aware VLAs（3D-VLA）、VLM-based imitation（LIAM）、CLIP-conditioned controllers（CLIPort, Diffusion Policy, ACT）。

Empirical asymmetry激发factorization：VLA delegate planner-selected contact-rich operations，higher-level controller承担language/transport/navigation/release。

### LLM-driven Multimodal Agent
GPT-5.2、Gemini 3、Qwen3-VL、Claude 4 family、Llama 4、Molmo。Frontier VLM已可承担semantic grounding + deterministic manipulation phases。

### Coding Agents for Robot Manipulation
Code-as-Policies → ProgPrompt → Instruct2Act → ChatGPT-for-Robotics → RoboCodeX → ViperGPT → VoxPoser → ASPIRE → SWE-agent → OpenHands → Voyager。

**Harness VLA vs ASPIRE**：ASPIRE expands skill library；Harness VLA deliberately keeps primitive vocabulary fixed，研究memory-guided composition如何extend frozen VLA without deployment-time primitive expansion。

**Harness VLA vs Voyager**：Voyager有persistent memory但未结合VLA-backed contact-rich primitive for physical manipulation。Harness VLA couples两者。

**Two limitations in prior literature**：
1. Task-specific execution traces很少作为reusable, parameterized memory
2. Failure knowledge很少distill成Global Memory防止repeating known pitfalls

---

## 10. Limitations和Future Work

- **Open feedback loop** between high-level planner和low-level VLA
- 缺乏**joint fine-tuning via environmental rewards和human preferences**（GRPO方向）
- 缺乏**fine-grained image captioning**，constrain structural reasoning in cluttered long-horizon tasks
- 结合ASPIRE-style automatic skill-discovery：repeated compositions reveal missing abstraction时，agent可propose/validate/admit新skill

---

## 11. Build Intuition的三个Layer

### Layer 1: Asymmetric Hierarchical Factorization

承认VLA的asymmetry：strong在contact-rich，weak在semantic grounding + long-horizon composition。把strong的地方抽出来作为primitive，把weak的地方提到planner level。

类比：software engineering中Monolithic vs Microservice。Monolithic VLA绑所有职责，failure cascade。Harness VLA"驯化"VLA成reusable component。

### Layer 2: Staging + Retry as Sandbox

VLA是stochastic + contact-rich physically brittle。单次失败不必terminate整个rollout。Harness VLA localize errors到current contact-rich subtask：

```
stage (analytic) → attempt (VLA) → verify → re-stage if fail
```

这相当于给VLA一个sandbox——在controlled local region尝试，失败回到analytic control再来一次。Figure 4的saturation curve直接验证这intuition。

### Layer 3: Memory as Structure Prior

Task Specific Memory是procedural skeleton（what + when），不是trajectory replay（where）。Global Memory是operating knowledge（how + why）。

这种分离让transfer可行：transfer结构不transfer坐标。Zero-shot实验（Table 5）证明planner reasoning本身可synthesize有用compositions，memory是enhancer不是crutch。

### 综合Intuition

Harness VLA本质上是**把VLA的operating range从training trajectory distribution扩展到deployment perturbation space**——通过analytic primitives traverse perturbation space，把VLA invocation限制在local contact-rich regions where training distribution仍然informative（Figure 2的核心图）。

VLA的"competence"是local的、contact-rich的、image-conditioned的。Harness VLA承认这locality，把global reasoning交给planner + memory。

---

## References

[1] Harness VLA Project Page: https://harnessvla.github.io/
[2] OpenVLA: https://arxiv.org/abs/2406.09246
[3] $\pi_0$: https://physicalintelligence.company/blog/pi0
[4] $\pi_{0.5}$: https://arxiv.org/abs/2504.16054
[5] NORA: https://arxiv.org/abs/2504.19854
[6] MolmoAct: https://arxiv.org/abs/2508.07917
[8] Code as Policies: https://code-as-policies.github.io/
[9] ProgPrompt: https://.progress-llm.github.io/
[12] CaP-X: https://arxiv.org/abs/2603.22435
[13] RATS: https://arxiv.org/abs/2606.19419
[15] SWE-agent: https://swe-agent.com/
[16] Voyager: https://arxiv.org/abs/2305.16291
[17] OpenHands: https://github.com/All-Hands-AI/OpenHands
[18] LIBERO: https://libero-project.github.io/
[19] LIBERO-Pro: https://arxiv.org/abs/2510.03827
[20] RoboCasa365: https://arxiv.org/abs/2603.04356
[21] RoboTwin: https://github.com/RoboTwin-platform
[22] RLinf: https://arxiv.org/abs/2509.15965
[23] RLDX-1: https://arxiv.org/abs/2605.03269
[24] LingBot-VLA: https://arxiv.org/abs/2601.18692
[25] AtomVLA: https://arxiv.org/abs/2603.08519
[28] GR00T N1: https://arxiv.org/abs/2503.14734
[34] Gemini Robotics: https://arxiv.org/abs/2503.20020
[42] GPT-5.2: https://openai.com/
[43] Gemini 3: https://blog.google/
[44] Qwen3-VL: https://arxiv.org/abs/2511.21631
[45] Claude Sonnet 4.5: https://www.anthropic.com/
[49] Molmo: https://molmo.allenai.org/
[50] SpatialVLM: https://spatial-vlm.github.io/
[62] ViperGPT: https://vipergpt.github.io/
[63] VoxPoser: https://voxposer.github.io/
[66] ASPIRE: https://arxiv.org/abs/2607.00272

---

## 总结思考

Harness VLA这paper真正解决的是**VLA deployment的"distribution mismatch"问题**——通过asymmetric decomposition而非finetuning或scale-up。

技术contribution三件套：
1. **Frozen VLA as retryable primitive**：VLA_ACT接口 + planner-staged retry
2. **Fixed primitive library + memory-guided orchestration**：不扩张vocabulary，learn operating range
3. **Two-tier memory**：Task Specific (procedural skeleton) + Global (operating knowledge)

实验上+38.6 pp (LIBERO-Pro) / +25.4 pp (RoboCasa365) / +8.0 pp (RoboTwin C2R)的改进，且zero-shot setting仍strong，证明这是**architecture-level的robustness gain**而非data或scale的gain。

对于你的intuition building：这paper本质上是说"别让VLA做所有事"。VLA的强项是local visuomotor contact，weak项是semantic binding和long-horizon composition。承认这asymmetry，用agentic planner + memory把强项isolate出来reuse，weak项提到cognitive layer——这是比"更大VLA"或"更多data"更elegant的path to robustness。
