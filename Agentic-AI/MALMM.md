---
source_pdf: MALMM.pdf
paper_sha256: f2b665acea73572931222d14e142a7551e366a77d68ef5577dac86aabeacfafc
processed_at: '2026-08-05T16:11:41-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# MALMM 用人话讲

Andrej，好，那我把西装脱了，咱们像在咖啡店聊 paper 那样讲一遍。

---

## 这篇 paper 到底在干嘛

一句话：**让一个 LLM 干 robot manipulation 的活，它容易犯糊涂；拆成三个 LLM 各干各的，就清醒多了。**

具体场景是这样的。你给 robot 一个 instruction，比如 "stack four blocks at the green target area"。以前的做法是丢给一个 GPT-4，让它自己想 plan、自己写 Python code、自己处理失败。问题在于，stack 4 个 block 要二十多步，每一步都要 plan + code + 看环境 feedback，context 越堆越长，GPT-4 到后面就开始犯晕：

- 忘了 goal 是啥（明明要 stack 在 green target，stack 到一半跑去别的地方了）
- 不考虑 collision（gripper 直接撞翻已经 stack 好的 block）
- 写 code 漏变量（`position_item_1` 没定义就直接用）
- 调 function 漏参数（`execute_trajectory()` 忘了传 orientation）

这就是 paper 里反复说的 "hallucination in long-horizon tasks"。参考 [Maharana et al. 2024](https://arxiv.org/abs/2402.17753) 对 long-context conversational memory 的系统评测，LLM 在长对话里 factuality 确实会 degrade。

MALMM 的解法很直觉：**既然一个 LLM context 太长会犯晕，那就拆成三个，每个 context 短一点，各自专注一件事。**

---

## 三个 agent 各干啥

### Planner（规划员）

拿到 task instruction + 当前环境状态（哪些 object 在哪、gripper 在哪），输出下一步要干啥。比如 "approach Item 1"、"grasp Item 1"、"move to yellow container"、"release"。

关键设计：planner 的 prompt 里塞了两样东西：
1. **环境坐标系描述** — 因为 LLM 训练数据里 grounded physical interaction 很少，它不知道 gripper 的 "left" 是哪个方向
2. **collision-avoidance rules** — 通用规则，比如 "移动前先抬高 z 轴 0.1m 避免撞东西"

每一步执行完后，planner 还要对比 $s_t$ 和 $s_{t-1}$ 判断上一步成没成功，失败就 replan。

### Coder（程序员）

拿到 planner 给的 sub-goal，翻译成可执行的 Python code。能调的 primitive function 就四个：

```python
execute_trajectory(position, orientation)  # 移动 end-effector
open_gripper()
close_gripper()
check_task_completion()
```

coder 的 prompt 里写死了每个 function 的 signature 和 input/output type，外加 "不要用没定义的变量" 之类的 guideline。

### Supervisor（调度员）

这是 MALMM 相比 "两个 agent" 版本的增量。它的工作是决定下一步该让谁说话。

naive 的 multi-agent 是固定循环：Planner → Coder → Executor → Planner → ...。问题是 coder 偶尔会写错 code（比如只 approach 没 grasp），固定循环要浪费一整个 Planner round 才能发现。

supervisor 看完整 chat history + 所有 agent 的 role description，动态决定下一个该激活谁。coder 写错了？直接扔回 coder 重写，不用麻烦 planner。

---

## 为什么拆开就有用 — context budget 的角度

这个 intuition 我觉得是整篇 paper 最核心的 insight，值得展开讲。

Single Agent 在 step $t$ 的 context 长度大致是：

$$L_{SA}(t) = L_{\text{sys}} + L_g + \sum_{\tau=1}^{t}\left[L_{\text{plan}_\tau} + L_{\text{code}_\tau} + L_{s_\tau}\right]$$

解释每个符号：
- $L_{SA}(t)$：Single Agent 在第 $t$ 步的总 context 长度（token 数）
- $L_{\text{sys}}$：system prompt 长度，固定不变
- $L_g$：task instruction $g$ 的长度，固定
- $L_{\text{plan}_\tau}$：第 $\tau$ 步 planner 输出的 plan 文本长度
- $L_{\text{code}_\tau}$：第 $\tau$ 步生成的 Python code 长度
- $L_{s_\tau}$：第 $\tau$ 步的 environment observation 长度（所有 object 的 3D bounding box、gripper state 等）

这个 sum 是**累加的**，因为 single agent 要记住之前所有步骤才能做 replan。

对 stack 4 blocks 这种 20+ 步的任务，$L_{SA}(20)$ 轻松超过 10K tokens。而 [Liu et al. "Lost in the Middle"](https://arxiv.org/abs/2307.03172) 的实验表明，LLM 对 context 中间位置信息的 retrieval 能力会显著下降。

MALMM 里每个 agent 只 maintain 自己 role 相关的 context：

$$L_P(t) \approx L_{\text{sys}_P} + L_g + L_{s_t} + L_{s_{t-1}} + \sum_{\tau=t-k}^{t} L_{\text{plan}_\tau}$$

Planner 只看最近 $k$ 步的 plan（$k$ 很小），加上当前和上一步的 state。

$$L_C(t) \approx L_{\text{sys}_C} + L_{\text{current\_subgoal}} + L_{s_t} + L_{\text{func\_sig}}$$

Coder 只看当前 sub-goal、当前 state、function signature，几乎不累积。

$$L_S(t) \approx L_{\text{sys}_S} + L_g + L_{\text{condensed\_history}}$$

Supervisor 看的是 condensed history（不是完整的 code 和 plan，是摘要）。

所以三个 agent 各自的 effective context 都比 SA 短很多，hallucination 概率自然降低。这跟人类团队管理的 logic 一样 — 项目经理不需要看每一行 code，程序员不需要看整个 project roadmap，各司其职反而更高效。

---

## 实验结果用大白话讲

### 主实验（Table I）

在 RLBench 的 9 个 task 上，每个 task 跑 25 次：

| 方法 | 平均成功率 |
|------|-----------|
| CAP（Code as Policies） | 0.09 |
| VoxPoser | 0.17 |
| Single Agent | 0.50 |
| MALMM（用 LLaMA-3.3-70B） | 0.70 |
| **MALMM（用 GPT-4-Turbo）** | **0.81** |

CAP 在 zero-shot 下基本废了，因为它本来靠 few-shot examples 才能 work，论文把 examples 拿掉就崩了。VoxPoser 只能在形状规则的物体上 work（put block、rubbish in bin），复杂形状（meat off grill、close jar）就不行。

MALMM 比 SA 高 31 个点，这个 gap 非常大。

### Ablation（Table II）— 这是最有价值的表

把 MALMM 的三个 component 逐个拆掉：

| 配置 | Stack Blocks | Empty Container |
|------|-------------|-----------------|
| SA + 没有 environment feedback | 0.08 | 0.12 |
| SA + 有 feedback | 0.20 | 0.36 |
| 两个 agent（Planner + Coder，没有 Supervisor） | 0.36 | 0.48 |
| **完整 MALMM（P + C + Supervisor）** | **0.56** | **0.64** |

用大白话解读：

**没有 environment feedback** = LLM 一次性生成完整 plan，闭着眼睛执行。任何一步失败（撞了、没抓到）就全盘崩溃。stack blocks 只有 8% 成功率。加了 feedback 后能 open eyes 执行，跳到 20%。

**拆成两个 agent** = planner 和 coder 各自 context 更短，更不容易犯晕。从 20% 跳到 36%。

**加 Supervisor** = coder 偶尔写错 code 时能直接 retry，不用浪费一整个 round。从 36% 跳到 56%。

三个 component 的贡献是 additive 的：

$$0.08 \xrightarrow{+0.12} 0.20 \xrightarrow{+0.16} 0.36 \xrightarrow{+0.20} 0.56$$

$$0.12 + 0.16 + 0.20 = 0.48 = 0.56 - 0.08$$

刚好对上。这说明三个 component 基本没有 interaction effect，各自独立贡献。这是个很 clean 的结果。

### Long-horizon scaling（Figure 5）

stack 2 / 3 / 4 个 block 的对比：

| # blocks | SA | MALMM |
|----------|------|-------|
| 2 | ~0.65 | ~0.85 |
| 3 | ~0.30 | ~0.70 |
| 4 | ~0.10 | ~0.28 |

两个观察：
1. MALMM 在每个 horizon 上都比 SA 好
2. 但 MALMM 自己在 4 blocks 上也只有 28%，说明 **LLM-based manipulation 在 rich physical reasoning 上有硬上限**

这个上限的根源我猜是：LLM 能 plan "把 block 放在 block 上"，但它不会精确推理 stack 的稳定性、contact area、center of mass。这些物理量需要 spatial reasoning，而 LLM 的 spatial reasoning 能力是有限的。要突破这个上限，可能需要 VLA models（比如 [RT-2](https://robotics-transformer2.github.io/)、[OpenVLA](https://openvla.github.io/)）的端到端 physical grounding。

### Real-world（Table IV）

在 Franka Panda 真机上的 5 个 task，每个 10 次：

| Task | SA | MALMM |
|------|------|-------|
| Close Jar | 2/10 | 4/10 |
| Put Block | 3/10 | 6/10 |
| Rubbish in Bin | 2/10 | 6/10 |
| Put Case | 3/10 | 7/10 |
| Jar in Bin | 3/10 | 5/10 |

real-world gap 反而比 simulation 更大，这点有点反直觉。通常 real-world 噪声会缩小方法间差距。我猜原因是 real-world 的 perception 噪声（bounding box 不准、grasp pose 漂移）放大了 SA 的 hallucination 问题 — SA context 里有更多 noisy observation，更容易被带歪。MALMM 的 Supervisor 能更好地 filter 这些噪声。

---

## Vision pipeline 怎么从摄像头拿到 state

simulation 里可以直接拿 ground-truth bounding box。real-world 只能靠摄像头，pipeline 是这样的：

```
RGB-D 拍一帧
    ↓
gpt-4-turbo 看 RGB 图 + task instruction
    → 输出 relevant object 名字（"red jar", "lid"）
    ↓
LangSAM（GroundingDINO + SAM）
    → 输出每个 object 的 2D segmentation mask
    ↓
RGBD → 3D point cloud
    ↓
2D mask 投影到 3D → 每个 object 的 3D point cloud
    ↓
M2T2 model
    → 输出 candidate grasp poses（SE(3) 6D）
    ↓
只保留 top-down 方向的 grasp
    ↓
DBSCAN 聚类 → 选中心 pose
    ↓
Target object 的 3D bounding box（从 point cloud 算）
    ↓
最终 state = {source grasp pose, target bbox}
```

还有个小 trick 防 jitter：如果相邻两帧的 grasp pose 平移 < 0.01m 或旋转 < 30°，就保留上一帧的 pose。因为 gripper 接近物体时视角微小变化会导致 grasp estimate 漂移，这个 stability check 避免了这个问题。

公式表达：

$$\text{if } \|\mathbf{p}_t - \mathbf{p}_{t-1}\| < 0.01 \text{m} \text{ or } \angle(\mathbf{R}_t, \mathbf{R}_{t-1}) < 30°$$

$$\text{then } s_t^{\text{visual}} \leftarrow s_{t-1}^{\text{visual}}$$

- $\mathbf{p}_t$：第 $t$ 帧的 grasp position（$\mathbb{R}^3$）
- $\mathbf{R}_t$：第 $t$ 帧的 grasp orientation（$SO(3)$ 旋转矩阵）
- $\angle(\cdot, \cdot)$：两个旋转之间的角度差
- $s_t^{\text{visual}}$：第 $t$ 步的视觉观测 state

参考：[LangSAM](https://github.com/luca-medeiros/lang-segment-anything)、[M2T2](https://arxiv.org/abs/2311.00976)、[DBSCAN](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf)

Vision-based 比 state-based 掉了约 20-30% 成功率。比如 Close Jar 从 0.84（state）掉到 0.56（vision）。主要原因是 bounding box 估计不准 + grasp estimation 偶尔失败 + single-view occlusion。[Manipulate-Anything](https://arxiv.org/abs/2406.18915) 用 multi-view 缓解 occlusion，是个自然的 follow-up。

---

## 跟你说说我觉得这篇 paper 真正的 contribution 和 limitation

### 真正的 contribution

1. **把 "agent specialization reduces hallucination" 这个 hypothesis 用 clean ablation 证明了**。三个 component 的贡献是 additive 的，这个 result 很 solid。这个 principle 可以 transfer 到其他 long-horizon LLM agent 场景。

2. **Supervisor 的 dynamic routing 比 fixed cyclic sequence 更 robust**。这个 design choice 在 multi-agent 系统里是个通用 lesson — 不要假设每个 agent 都成功，要有 error recovery 机制。跟 [Reflexion](https://arxiv.org/abs/2303.11366) 的 self-reflection 思路类似，但 MALMM 把 reflection 和 act 拆到不同 agent。

3. **Zero-shot 到 novel task 的 demonstration**（Figure 7 的 "reunite brown horse with white one"）说明这个 framework 的 generalization 是真的，靠的是 LLM 的 world knowledge，靠的是 foundation model 的 perception。

### 诚实的 limitation

1. **Prompt engineering 是隐性 supervision**。三个 agent 的 prompt（Figure 9-11）里写了 collision-avoidance rules、coordinate system description、function signatures。这些是 human prior knowledge 的 injection，在 stack blocks 上 tune 好，然后迁移到其他 task。所以 MALMM 的 "zero-shot" 是 zero-shot w.r.t. tasks，zero-shot w.r.t. prompt design 是不算的。这个 distinction 论文没说清楚。

2. **Cost**。三个 GPT-4-Turbo agents，closed-loop，每步至少 3 次 LLM call。Stack 4 blocks 大概 20+ steps → 60+ LLM calls。粗估每次 task execution 成本 $2-5。LLaMA-3.3-70B 的结果（0.70 avg）说明 open-source 是 viable，但仍有 11 point gap。参考 [LLaMA 3 herd](https://arxiv.org/abs/2407.21783)。

3. **Rich physical reasoning 的硬上限**。Stack 4 blocks 只有 28%。LLM 能 plan "放在上面"，但不推理 stack 稳定性、contact area、center of mass。这个上限靠更好的 prompt 或更多 agent 突破不了，需要 fundamentally different approach（VLA models、physics simulator in the loop）。

4. **没有 failure mode 的 quantitative breakdown**。论文说 SA fails due to "forgetting goal / missing collision / variable errors"，但没给每种 failure 的比例。对理解 bottleneck 来说这个分析缺了。

5. **Supervisor 自己也可能 hallucinate**。Supervisor 是个 LLM，它的 routing decision 也会出错。论文没分析 Supervisor 的 error rate。

---

## 我觉得最值得带走 intuition

** specialization reduces hallucination** — 这个 principle 太通用了。

你做 [Eureka Labs](https://eureka-labs.ai/) 的时候，教育场景里 tutor / grader / curriculum-designer 的分工，跟这里的 Planner / Coder / Supervisor 是同构的：
- Tutor 像 Planner，决定下一步教什么
- Grader 像 Coder，执行具体的评估逻辑
- Curriculum-designer 像 Supervisor，协调 tutor 和 grader 之间的 flow

如果一个 LLM 同时干这三件事，context 会很长，容易忘掉 student 之前的学习历史或混淆 role。拆开的话每个 agent 更专注，context 更短，出错概率更低。

同样的 logic 也适用于 coding agent（[Devin](https://devin.ai)、[SWE-agent](https://sWE-agent.com)）、research agent（[AI Scientist](https://github.com/SakanaAI/AI-Scientist)）等场景。只要 task 是 long-horizon 的、需要多种 reasoning mode 的，agent specialization 都是个值得考虑的 design pattern。

---

## 参考链接汇总

- [MALMM project page](https://malmm1.github.io/)
- [AutoGen](https://github.com/microsoft/autogen)
- [RLBench](https://github.com/stepjam/RLBench)
- [LangSAM](https://github.com/luca-medeiros/lang-segment-anything)
- [M2T2](https://arxiv.org/abs/2311.00976)
- [Code as Policies](https://code-as-policies.github.io/)
- [VoxPoser](https://voxposer.github.io/)
- [LM-ZSP](https://arxiv.org/abs/2310.11644)
- [Reflexion](https://arxiv.org/abs/2303.11366)
- [ReAct](https://arxiv.org/abs/2210.03629)
- [Lost in the Middle](https://arxiv.org/abs/2307.03172)
- [Long-term memory eval](https://arxiv.org/abs/2402.17753)
- [Manipulate-Anything](https://arxiv.org/abs/2406.18915)
- [RT-2](https://robotics-transformer2.github.io/)
- [OpenVLA](https://openvla.github.io/)
- [Chain of Thought](https://arxiv.org/abs/2201.11903)
- [DBSCAN](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf)
- [LLaMA 3](https://arxiv.org/abs/2407.21783)
- [Eureka Labs](https://eureka-labs.ai/)

---

一句话总结：**MALMM 告诉我们，robotic manipulation 这种 long-horizon 任务上，与其让一个 LLM 当全栈，不如拆成三个专才，每个 context 短一点、role 清楚一点，hallucination 就少了。这个 lesson 适用于任何需要 long-horizon multi-mode reasoning 的 agent design。**

---

# MALMM: Multi-Agent Large Language Model for Manipulation 深度解析

Andrej, 这篇 paper 我觉得是 LLM-for-robotics 领域里把"agent 分工"这件事做得最干净的一篇。下面我从 motivation、architecture、formulation、experiment、ablation 几个维度拆开讲，尽量把背后的 intuition 挖出来。

---

## 1. Paper 定位与核心 motivation

这篇 paper 攻击的是一个很具体的问题：**Single LLM agent 在 long-horizon manipulation 任务上会 hallucinate**。

具体症状包括：
- 忘记 goal（goal drift）
- 忽略 geometric constraints / collision
- 调用 predefined functions 时漏掉参数（"use variables before initialization"）
- 在 closed-loop replan 过程中,context 越积越长,LLM 在 long-context generation 中产生 factuality degradation（参考 [Maharana et al., 2024](https://arxiv.org/abs/2402.17753)）

而现有 zero-shot 方法比如 [Code as Policies (CAP)](https://arxiv.org/abs/2209.07753)、[VoxPoser](https://arxiv.org/abs/2307.05973)、[LM-ZSP (Kwon et al.)](https://arxiv.org/abs/2310.11644) 都有各自的硬伤:
- CAP 严重依赖 few-shot examples,zero-shot 下基本崩盘(论文 Table I 显示 9 个 task 平均 0.09)
- VoxPoser 需要生成 3D value maps / voxel maps,只能 handle 形状规则物体
- SA (Single Agent) 假设每一步都正确执行,没有 failure recovery

MALMM 的核心 thesis 是: **与其让一个 LLM 同时背 planning + coding + error recovery 三座大山,不如拆成三个 specialized agents,每个 agent 的 context 更短、role 更聚焦,hallucination 自然减少**。

这个动机和软件工程里 "separation of concerns" 完全一致,也呼应了 [Wu et al., AutoGen](https://arxiv.org/abs/2308.08155) 的 multi-agent conversation framework。

---

## 2. Architecture 深度解析

整体系统由 **3 个 LLM agents + 1 个 deterministic tool** 组成:

```
┌─────────────────────────────────────────────────────────┐
│                    SUPERVISOR                           │
│  Input: task g, state s_t, chat history h_t, roles      │
│  Output: next_agent ∈ {Planner, Coder, Code Executor}   │
│  Decision policy: dynamic re-routing based on history   │
└────────────┬────────────────────────────────────────────┘
             │ routes to
     ┌───────┴───────┬──────────────────┐
     ▼               ▼                  ▼
┌─────────┐    ┌─────────┐       ┌──────────────┐
│ PLANNER │───▶│  CODER  │──────▶│ CODE EXECUTOR│
│         │    │         │       │  (Python     │
│ high-   │    │ Python  │       │   interpreter│
│ level   │    │ code    │       │   in RLBench │
│ steps   │    │ gen     │       │   /Franka)   │
│ +replan │    │         │       │              │
└─────────┘    └─────────┘       └──────┬───────┘
                                        │
                                        ▼
                              Environment observation
                              s_{t+1} = f(s_t, code_t)
                                        │
                                        ▼ feedback loop
                              back to Supervisor
```

### 2.1 Planner Agent
- **输入**: task instruction $g$, current environment state $s_t$ (3D bounding boxes + gripper state), previous state $s_{t-1}$
- **输出**: next high-level sub-goal step + 是否需要 replan
- **Prompt 关键设计** (Figure 10):
  - 详细描述 environment coordinate system(让 LLM 能从 gripper perspective 理解方向)
  - 写入 generic collision-avoidance rules(因为 LLM 训练数据里物理交互数据少,不会自发考虑 collision)
  - 要求 Planner 在每一步之后,**比较 $s_t$ 和 $s_{t-1}$** 来判断 action 是否成功,失败则 replan

### 2.2 Coder Agent  
- **输入**: Planner 输出的 sub-goal,environment state $s_t$
- **输出**: 可执行 Python code,调用以下 primitive functions:
  - `execute_trajectory(position, orientation)` — 移动 end-effector 到 3D waypoint
  - `open_gripper()` / `close_gripper()`
  - `check_task_completion()`
- **Prompt 关键设计** (Figure 11): 列出每个 function 的 input/output signature,加上避免 syntactic/semantic error 的 guidelines

### 2.3 Supervisor Agent
- **输入**: task $g$, state $s_t$, **完整 chat history $h_t$ of all active agents**, role descriptions
- **输出**: `next_speaker ∈ {Planner, Coder, Code Executor}`
- **关键区别**: 不是固定 cyclic sequence (Planner → Coder → Executor → Planner ...),而是 **dynamic re-routing**。例如 Coder 漏了变量初始化,Supervisor 可以把控制权再扔回 Coder,而不是傻乎乎地交给 Executor 报错再回来

这个 dynamic routing 是 MALMM 相比 MA (Multi-Agent without Supervisor) 的核心增量。Table II 显示在 stack blocks 上 Supervisor 贡献 +20%,empty container 上 +16%。

---

## 3. Formalization

我尝试把 MALMM 的决策过程形式化,让 intuition 更清晰。

### 3.1 MDP-like formulation

定义:
- $g \in \mathcal{G}$: natural language task instruction
- $s_t \in \mathcal{S}$: environment state at step $t$,包含所有 object 的 3D bounding box $(\mathbf{p}_i, \mathbf{R}_i, \mathbf{d}_i, c_i)$ 和 gripper state $(\mathbf{p}^{ee}_t, \mathbf{R}^{ee}_t, \text{grip\_state})$
  - $\mathbf{p}_i \in \mathbb{R}^3$: object $i$ center position
  - $\mathbf{R}_i \in SO(3)$: orientation (论文里实际只用 z-axis rotation $\theta_i$)
  - $\mathbf{d}_i = (h_i, w_i, l_i)$: dimensions
  - $c_i$: color
- $h_t$: chat history up to step $t$
- $a_t$: action (code snippet executed)

### 3.2 Three policies

**Planner policy**:
$$\pi_P(a_t^{\text{plan}} \mid g, s_t, s_{t-1}, h_t)$$
其中 $a_t^{\text{plan}} \in \{\text{next\_subgoal}, \text{replan}, \text{terminate}\}$

**Coder policy**:
$$\pi_C(c_t \mid a_t^{\text{plan}}, s_t, h_t)$$
其中 $c_t$ 是 Python code string

**Supervisor policy**:
$$\pi_S(\text{next\_agent} \mid g, s_t, h_t, \text{roles})$$
其中 $\text{next\_agent} \in \{P, C, E\}$ (Planner / Coder / Executor)

**Environment transition**:
$$s_{t+1} = f(s_t, \text{execute}(c_t))$$

### 3.3 为什么这个架构 reduce hallucination — context budget argument

Single Agent 在 step $t$ 的 context 长度大致:
$$L_{SA}(t) = L_{\text{sys}} + L_g + \sum_{\tau=1}^{t}\big[L_{\text{plan}_\tau} + L_{\text{code}_\tau} + L_{s_\tau}\big]$$

对 long-horizon 任务(比如 stack 4 blocks,需要 ~20+ steps),$L_{SA}(t)$ 会逼近甚至超过 LLM 的 effective context window,导致 attention dilution 和早期的 plan/code 被"忘记"。

MALMM 中,每个 agent 只需要 maintain 自己 role 相关的 context:
- Planner context: $\approx L_{\text{sys}_P} + L_g + L_{s_t} + L_{s_{t-1}} + \text{recent plans}$
- Coder context: $\approx L_{\text{sys}_C} + L_{\text{current\_subgoal}} + L_{s_t} + \text{function signatures}$
- Supervisor context: $\approx L_{\text{sys}_S} + L_g + \text{condensed history}$

每个 agent 的 effective context 显著更短,这跟 [Liu et al., Lost in the Middle](https://arxiv.org/abs/2307.03172) 的发现一致 — LLM 在长 context 中间位置的信息 retrieval 能力会显著下降。

---

## 4. Environment Observation Pipeline

这是 MALMM 落地到 real-world 的关键工程部分。论文给了两条路径:

### 4.1 State-space observations (simulator)
直接从 RLBench/CoppeliaSim 内部拿 ground-truth:
- 3D bounding boxes (center, orientation, dimensions)
- Object colors
- Gripper position/orientation/open-close state

### 4.2 Vision-based observations (real-world)

```
RGB-D frame (front view)
    │
    ▼
[gpt-4-turbo]  ← input: RGB + task instruction
    │           → output: list of relevant object names (e.g. "red jar", "lid")
    ▼
[LangSAM]      ← GroundingDINO + SAM
    │           → output: 2D segmentation masks for each named object
    ▼
RGBD → 3D point cloud
    │
    ▼
2D mask projection → 3D object point clouds
    │
    ▼
[M2T2]         ← Multi-task Masked Transformer
    │           → output: candidate grasp poses (SE(3) 6D poses)
    ▼
Filter: keep only top-down gripper orientation
    │
    ▼
[DBSCAN clustering]  → select central grasp pose
    │
    ▼
Target object 3D bounding box (from point cloud)
    │
    ▼
Final state s_t = {source_grasp_pose, target_bbox}
```

**History-aware jitter mitigation** (Appendix-D):
为防止视觉估计在 frame 间抖动,引入 stability check:
$$\text{if } \|\mathbf{p}_t - \mathbf{p}_{t-1}\| < 0.01\text{m} \text{ or } \angle(\mathbf{R}_t, \mathbf{R}_{t-1}) < 30°$$
$$\text{then } s_t^{\text{visual}} \leftarrow s_{t-1}^{\text{visual}}$$

这是个很 pragmatic 的小 trick,避免 grasp pose 在 gripper 接近物体时因为视角微小变化而漂移。

参考链接:
- [LangSAM](https://github.com/luca-medeiros/lang-segment-anything)
- [M2T2 paper](https://arxiv.org/abs/2311.00976)
- [DBSCAN original](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf)

---

## 5. Experiments 深度分析

### 5.1 Main results (Table I)

| Method | Avg over 9 tasks |
|--------|------------------|
| CAP | 0.09 |
| VoxPoser | 0.17 |
| Single Agent (SA) | 0.50 |
| MALMM (LLaMA-3.3-70B) | 0.70 |
| **MALMM (GPT-4-Turbo)** | **0.81** |

MALMM 比 SA 高 31 个 absolute point,这是 very sizable gap。

值得注意几个 per-task 模式:
- **Stack Blocks** (long-horizon): SA 0.20 → MALMM 0.56 (提升 180%)
- **Put Block** (simple): SA 0.92 → MALMM 1.00 (提升 8.7%)
- **Open Bottle**: SA 0.80 → MALMM 0.96

直觉: **任务 horizon 越长,MALMM 的相对优势越大**。这跟 Section 5.3 的 ablation 完全吻合。

### 5.2 Ablation study (Table II) — 这是 paper 最有信息量的部分

| Config | Stack Blocks | Empty Container |
|--------|-------------|-----------------|
| SA, no env feedback | 0.08 | 0.12 |
| SA, with env feedback | 0.20 | 0.36 |
| MA (P+C, no Supervisor) | 0.36 | 0.48 |
| MALMM (P+C+S) | 0.56 | 0.64 |

拆解每个 component 的 marginal contribution:

**Environment feedback** (SA no-feedback → SA with-feedback):
- Stack Blocks: +12% (0.08 → 0.20)
- Empty Container: +24% (0.12 → 0.36)
- Intuition: 没有 feedback,LLM 一次性生成完整 plan,任何一步失败就全盘崩溃。有 feedback 后能 detect collision / missed grasp 并 replan

**Agent specialization** (SA → MA):
- Stack Blocks: +16% (0.20 → 0.36)
- Empty Container: +12% (0.36 → 0.48)
- Intuition: Planner 和 Coder 各自 context 更短、role 更聚焦,减少 long-context hallucination

**Supervisor dynamic routing** (MA → MALMM):
- Stack Blocks: +20% (0.36 → 0.56)
- Empty Container: +16% (0.48 → 0.64)
- Intuition: Coder 偶尔会生成 incomplete code(比如只 approach 不 grasp),固定 cyclic sequence 会浪费一个 Planner round 才能发现,Supervisor 可以直接 re-invoke Coder

三个 component 的 contribution stacking:
$$\Delta_{\text{feedback}} + \Delta_{\text{specialization}} + \Delta_{\text{supervisor}} = 12 + 16 + 20 = 48\% \text{ (Stack Blocks)}$$
$$0.08 \xrightarrow{+12} 0.20 \xrightarrow{+16} 0.36 \xrightarrow{+20} 0.56$$

总和 48% 跟实际 0.56 - 0.08 = 0.48 完全吻合,说明三个 component 的影响是 **additive**(至少在这个 task 上),没有强烈的 interaction effect。这是个 clean result。

### 5.3 Long-horizon scaling (Figure 5)

Stack blocks task 的变体:

| # blocks | SA | MALMM |
|----------|------|-------|
| 2 | ~0.65 | ~0.85 |
| 3 | ~0.30 | ~0.70 |
| 4 | ~0.10 | ~0.28 |

关键观察:
- SA 从 2 → 4 blocks 衰减因子 ~6.5×(0.65 → 0.10)
- MALMM 从 2 → 4 blocks 衰减因子 ~3×(0.85 → 0.28)
- MALMM 在 4 blocks 上比 SA 高 ~6×,但绝对值 0.28 仍然不高

这印证了 paper Limitations 部分承认的: **MALMM 在 rich object interactions 上仍有上限**。原因我猜测是 LLM 对 stacked geometry 的 reasoning 能力有限 — 它能 plan "把 block 放在 block 上",但不会精确推理 stack 稳定性、contact area、center of mass 这些物理量。

### 5.4 Vision-based results (Table III)

| Task | SA | MALMM |
|------|------|-------|
| Close Jar | 0.24 | 0.56 |
| Put Block | 0.68 | 0.84 |
| Rubbish in Bin | 0.40 | 0.52 |

对比 state-based (Table I): Close Jar 0.84 → 0.56,Put Block 1.00 → 0.84。Vision-based 平均掉 ~20-30%。

Paper 把这归因于:
- 3D bounding box 检测不准
- Grasp estimation (M2T2) 偶尔失败
- Single-view occlusion

[Manipulate-Anything (Duan et al., 2024)](https://arxiv.org/abs/2406.18915) 用 multi-view 缓解 occlusion,是个自然的 follow-up 方向。

### 5.5 Real-world results (Table IV)

5 个 task,每个 10 trials:

| Task | SA | MALMM | Δ |
|------|------|-------|---|
| Close Jar | 2/10 | 4/10 | +20% |
| Put Block | 3/10 | 6/10 | +30% |
| Rubbish in Bin | 2/10 | 6/10 | +40% |
| Put Case | 3/10 | 7/10 | +40% |
| Jar in Bin | 3/10 | 5/10 | +20% |

Real-world gap 比 simulation 更大,这有点反直觉 — 通常 real-world 噪声会让方法间差距缩小。我猜测原因是 real-world 的 perception 噪声放大了 SA 的 hallucination 问题(因为 SA context 里有更多 noisy observation),而 MALMM 的 Supervisor 能更好地 filter/recover。

Figure 7 展示了 3 个完全 novel 的 instruction("Reunite the brown horse with the white one"、"Save Cow"、"Place Carnivore in Cage"),证明了 zero-shot generalization 到新语义场景。

---

## 6. Execution Log 解析 (Appendix-E)

Appendix-E 给了 empty container task 的完整 execution log,这是理解 MALMM 实际运作的最好材料。

Task: "empty all items from the large container into the yellow container"

Initial state 有 3 个 items + 2 个 containers (yellow + purple) + 1 个 large container。

执行 flow:
1. **Supervisor** → Planner: 生成 step "approach Item 1"
2. **Planner** → Supervisor: 输出 plan
3. **Supervisor** → Coder: 把 plan 转 code
4. **Coder** → Supervisor: 输出 `execute_trajectory(position_above_item_1, orientation)`
5. **Supervisor** → Code Executor: 执行
6. **Code Executor** → Supervisor: 返回 exitcode 0 + new state $s_{t+1}$
7. ... 循环 ...

观察到的一个细节: Item 1 的 center position 在第一次 execute 后从 `[0.25638783, 0.12307772, 0.76729274]` 变成 `[0.26242024, 0.11987246, 0.76251084]` — 物体位置有微小漂移(可能是 simulator 物理引擎的 contact dynamics)。Planner 在下一步用的是 updated position,这就是 environment feedback 的价值。

最后 Planner 总结: "With this, all items from the large container have been successfully relocated to the yellow container. This completes the task." — 明确的 termination signal。

---

## 7. Critical Analysis & Open Questions

### 7.1 Prompt engineering 的隐性 cost
Paper 承认 MALMM "depends on manual prompt engineering"。三个 agent 的 prompt(Figure 9-11)加起来很长,且每个 prompt 里写了 collision-avoidance rules、coordinate system description、function signatures。这些 prompt 是在 stack blocks task 上 tune 出来的,然后 zero-shot 迁移到其他 8 个 task。

这意味着 MALMM 的 "zero-shot" 是 **zero-shot w.r.t. tasks, but NOT zero-shot w.r.t. prompt design**。这个 distinction 在论文里没有明确说清楚。

### 7.2 Cost analysis
三个 GPT-4-Turbo agents,closed-loop,每步至少 3 次 LLM call(Supervisor + Planner/Coder + 可能 re-route)。Stack 4 blocks 大概 20+ steps → 60+ LLM calls。如果每次 call 平均 2K tokens output,GPT-4-Turbo 定价下大概 $2-5 per task execution。这对 research demo 没问题,production 部署需要 cheaper backbone。

LLaMA-3.3-70B 的结果(0.70 avg)说明 open-source backbone 是 viable path,但仍有 11 point gap。

### 7.3 为什么不用 function calling / tool use API?
GPT-4-Turbo 原生支持 function calling,MALMM 却选择让 Coder 生成 raw Python code 再交给 executor。这个 design choice 的 trade-off:
- **Pro**: 更灵活,LLM 可以写复杂逻辑(循环、条件、变量)
- **Con**: 更易出错(syntax error、undefined variable)

如果用 function calling,Supervisor 的 dynamic re-routing 部分可以由 OpenAI 的 tool router 隐式处理。但 paper 选择 explicit Supervisor,可能是因为他们想要 full control over routing logic 和支持 open-source LLM。

### 7.4 与 ReAct / Reflexion 的关系
MALMM 的 Planner replan 机制跟 [Reflexion (Shinn et al., 2023)](https://arxiv.org/abs/2303.11366) 的 self-reflection 很像,跟 [ReAct (Yao et al., 2022)](https://arxiv.org/abs/2210.03629) 的 reason+act 也有神似。但 MALMM 把 reflection 和 act 拆到不同 agent,这是 architectural novelty。

### 7.5 Limitations 没充分讨论的
- **No quantitative analysis on number of LLM calls per task** — 这对 cost estimation 很关键
- **No failure mode taxonomy** — paper 说 SA fails due to "forgetting goal / missing collision / variable errors",但没有 quantitative breakdown
- **No comparison on different LLM backbones beyond GPT-4-Turbo and LLaMA-3.3-70B** — 比如 Claude-3.5-Sonnet、Gemini-1.5-Pro 在 code generation 上有不同 profile
- **Supervisor 的 routing decision 本身可能 hallucinate** — paper 没分析 Supervisor 错误率

---

## 8. Broader Context & Related Work Map

```
LLM for Robotics
├── High-level planning only
│   ├── SayCan (Ahn et al., 2022)        — 需要预训练 skill policies
│   ├── Inner Monologue (Huang et al., 2022)
│   └── LLM+P (Liu et al., 2023)
├── Low-level code generation
│   ├── Code as Policies (Liang et al., 2023)  — few-shot
│   ├── VoxPoser (Huang et al., 2023)          — 3D value maps
│   └── LM-ZSP (Kwon et al., 2023)             — SA baseline
├── Reward / trajectory optimization
│   ├── Language to Rewards (Yu et al., 2023)
│   └── RL-VLM-F (Wang et al., 2024)
├── Multi-agent
│   ├── RoCo (Mandi et al., 2023)              — multi-robot collaboration
│   ├── CoELA (Dasgupta et al., 2023)          — RL+LLM
│   └── MALMM (this paper)                     — first multi-agent for manipulation
└── Vision-language grounding
    ├── BC-Z (Jang et al., 2022)
    ├── RT-2 (Google, 2023)
    └── Physically Grounded VLMs (Gao et al., 2024)
```

MALMM 在这个 map 里的独特位置: **第一个用 multi-agent LLM 做 zero-shot manipulation 的方法,不依赖预训练 skills,不依赖 in-context examples**。

---

## 9. Key Takeaways for Building Intuition

1. **Agent specialization ≈ context budget partitioning**: 把一个长 context LLM call 拆成 N 个短 context LLM call,每个 call 聚焦一个 role,hallucination 显著减少。这个 principle 应该可以 transfer 到其他 long-horizon reasoning 任务。

2. **Environment feedback after each step 是 non-negotiable**: 没有 feedback 的 SA 在 stack blocks 上只有 8% 成功率,有 feedback 后跳到 20%。Feedback 让 LLM 从 open-loop planner 变成 closed-loop controller。

3. **Supervisor 的 dynamic routing 比 fixed sequence 更 robust**: 固定 cyclic sequence 假设每个 agent 都成功,但 LLM 偶尔会 fail(漏变量、incomplete code)。Supervisor 可以局部 retry 而不是浪费整个 round。

4. **Zero-shot generalization 的边界**: MALMM 在 4-block stacking 上只有 28%,说明 LLM-based manipulation 在 rich physical reasoning 上有硬上限。突破这个上限可能需要 VLA (Vision-Language-Action) models 比如 [RT-2](https://robotics-transformer2.github.io/) 或 [OpenVLA](https://openvla.github.io/) 的端到端 physical grounding。

5. **Vision pipeline 是 bottleneck**: State-based 0.81 → Vision-based ~0.64,掉了 20%+。这说明当前 foundation models (LangSAM, M2T2) 在 real-world manipulation 的 perception 还不够 robust。Multi-view (Manipulate-Anything) 或更强的 3D foundation models (比如 [SAM-3D](https://arxiv.org/abs/2406.08629) 方向) 是必要演进。

6. **Prompt engineering 是隐性 supervision**: MALMM 的 prompt 里写了 collision-avoidance rules、coordinate system、function signatures。这些是 human prior knowledge 的 injection。真正的 zero-shot 应该让 LLM 自己 discover 这些 rules,可能需要更强的基础模型或 RL-based prompt optimization。

---

## 10. 参考资源

- [MALMM project page](https://malmm1.github.io/)
- [AutoGen (Microsoft)](https://github.com/microsoft/autogen)
- [RLBench](https://github.com/stepjam/RLBench)
- [LangSAM](https://github.com/luca-medeiros/lang-segment-anything)
- [M2T2](https://arxiv.org/abs/2311.00976)
- [Code as Policies](https://code-as-policies.github.io/)
- [VoxPoser](https://voxposer.github.io/)
- [LM-ZSP (Kwon et al.)](https://arxiv.org/abs/2310.11644)
- [Reflexion](https://arxiv.org/abs/2303.11366)
- [ReAct](https://arxiv.org/abs/2210.03629)
- [Manipulate-Anything](https://arxiv.org/abs/2406.18915)
- [Lost in the Middle](https://arxiv.org/abs/2307.03172)
- [Long-term memory evaluation](https://arxiv.org/abs/2402.17753)
- [LLaMA 3 herd](https://arxiv.org/abs/2407.21783)
- [Chain of Thought](https://arxiv.org/abs/2201.11903)

---

**Final thought**: MALMM 的核心 insight — **specialization reduces hallucination** — 是个很 general 的 principle。在 LLM agent design 里,我们经常陷入 "let one mega-agent do everything" 的诱惑(比如 AutoGPT 早期版本),但 MALMM 用 clean ablation 证明了拆分的价值。这个 lesson 对你正在关注的 [Eureka Labs](https://eureka-labs.ai/) 类型的教育 agent design 也应该有启发 — 教学场景里 tutor / grader / curriculum-designer 的分工,跟这里的 Planner / Coder / Supervisor 是同构的。
