---
source_pdf: ComprehensiveBenchmarking Multi-modal Large Language Models for Vision-Driven
  Embodied Agents.pdf
paper_sha256: 60ebeeaabc00ab17a809b9e5478dd79d803cab89ab29ad931e7b5079dbf281d1
processed_at: '2026-08-03T16:50:54-07:00'
target_folder: Robot-VLA/Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# EMBODIEDBENCH 用人话版

## 一句话总结

这篇paper搞了一个测试平台，看现在那些号称很强的multimodal LLM（GPT-4o, Claude, Gemini这些）到底能不能当robot的大脑用。结论是：**让它们做"把杯子放桌上"这种high-level任务还行，让它们控制机械臂精确抓东西就基本废了**。

---

## 为什么需要这个benchmark

之前大家测试LLM agent，都是在ALFWorld这种纯文字环境里跑——agent看到的是文字描述"你面前有个桌子，桌上有个苹果"，然后输出action。VisualAgentBench虽然加了vision，但也只测high-level planning，相当于"你看着图片，然后说接下来该干啥"。

**没人测过low-level control**——就是让agent看着摄像头画面，直接输出"机械臂往左移动0.3米，旋转15度，闭合夹爪"这种细粒度action。

EMBODIEDBENCH填补了这个空白：4个environment，1128个test task，覆盖从high-level到low-level的spectrum，还按6种能力拆分评估。

Link: https://embodiedbench.github.io

---

## 4个environment到底在测什么

### EB-ALFRED — 家务活指挥官

你给agent一句话："把洗干净的生菜放进冰箱"。Agent看着egocentric视角的图片，从171-298个action里选（find, pick up, open, close, turn on, slice...）。

这是high-level任务——每个action都有明确语义，simulator会自动帮你执行底层运动。相当于你指挥一个人干活："去冰箱那边，打开门，把生菜放进去，关门"。

### EB-Habitat — 房间整理员

跟ALFRED类似但更复杂。70个skill，但navigation只能去receptacle-type object（countertop, fridge这种），不能直接去任何object。意味着你得先navigate到fridge，再navigate到counter，来回跑。

### EB-Navigation — 找东西

Agent在一个房间里，被告诉"找到laptop并靠近它"。只能用8个low-level action：前进0.25m、后退0.25m、左移、右移、左转90°、右转90°、摄像头向上30°、向下30°。

**没有GPS，没有距离传感器，全靠眼睛看**。这跟真实robot navigation很像。

### EB-Manipulation — 控制机械臂

最难的一个。控制7-DoF Franka Panda机械臂，输出7维action vector：

$$a = [X, Y, Z, \text{Roll}, \text{Pitch}, \text{Yaw}, \text{Gripper}]$$

- $(X, Y, Z)$: 夹爪在3D空间的位置坐标
- $(\text{Roll}, \text{Pitch}, \text{Yaw})$: 夹爪的朝向（欧拉角）
- $\text{Gripper}$: 0是闭合，1是张开

为了让MLLM能处理，他们把continuous action space discretize了——position分成100个bin，orientation分成120个bin。这样agent输出整数就行，比如$[57, 61, 20, 10, 60, 25, 1]$。

还额外加了YOLO detection box和object的3D坐标作为hint，不然MLLM根本搞不定。

---

## Agent是怎么设计的

每次planning step，agent做5件事：

1. **描述当前画面**：用文字说"我看到桌上有个红色杯子，左边有个蓝色碗"
2. **反思历史**：回顾之前做了什么，environment反馈了什么
3. **推理**：想清楚怎么从当前state到达goal
4. **制定language plan**：用自然语言写plan
5. **转成executable plan**：输出JSON格式的action序列

**关键设计**：multi-step planning。agent一次可以plan多个action，不是每步只输出一个。好处是省API call，坏处是如果中间某步错了，后面全白做。

这跟model predictive control的思路类似——开环预测一段horizon，执行后根据observation修正。

---

## 核心实验结果

### High-level vs Low-level的鸿沟

| 任务类型 | GPT-4o | Claude-3.5 | InternVL3-78B |
|---------|--------|-----------|---------------|
| EB-ALFRED (high) | 56.3% | 64.0% | 39.0% |
| EB-Habitat (high) | 59.0% | 68.0% | 55.0% |
| EB-Navigation (low) | 57.7% | 44.7% | 53.7% |
| EB-Manipulation (low) | 28.9% | 25.4% | 26.3% |

**High-level任务还能玩，low-level manipulation直接崩**。最好的GPT-4o在manipulation上也只有28.9%成功率。

为什么？因为high-level task有semantic scaffold——"pick up apple"这种action的语义已经很明确，MLLM的language understanding能cover。但low-level task需要**精确的spatial reasoning + continuous control**，MLLM根本没这个能力。

### Vision到底有没有用

这个发现很有意思：

| 设置 | EB-Navigation | EB-ALFRED |
|------|---------------|-----------|
| 有vision | 57.7% | 56.3% |
| 只有language | 17.4% | 58.0% |

**Low-level任务上vision至关重要**——移除vision后navigation掉40个点，long-horizon直接归零。因为navigation全靠看图找路。

**High-level任务上vision几乎没用**——ALFRED上移除vision反而还涨了一点。因为high-level task主要靠text feedback和symbolic reasoning，视觉信息反而可能是noise。

这说明**当前MLLM的vision module在reasoning pipeline里还没真正integrate进去**。Vision encoder把图片变成tokens，但LLM的reasoning还是主要依赖language tokens。

### Long-horizon是最难的

所有model在long-horizon subset上都显著掉分：
- Claude-3.5在Habitat上：Base 96% → Long 58%
- GPT-4o在Habitat上：Base 86% → Long 64%
- GPT-4o-mini在ALFRED上：Base 34% → Long **0%**

为什么long-horizon这么难？因为autoregressive decoding的error会累积。每一步都可能犯小错，15步之后这些error叠加起来基本就废了。而且MLLM没有explicit memory mechanism，不记得自己之前在哪个sub-task上失败了。

这跟你经常说的"LLM的本质limitation来自autoregressive nature"完全吻合。

### Open-source vs Proprietary

InternVL3-78B在low-level task上接近GPT-4o（Navigation 53.7 vs 57.7，Manipulation 26.3 vs 28.9），但high-level task差距明显。

Open-source model有clear scaling trend——size越大performance越好。但最大的open-source model（78B-90B）还是不如最强的proprietary model。

---

## Ablation研究的几个有意思发现

### Camera resolution不是越高越好

300×300 < **500×500** > 700×700

500×500最优。低分辨率缺细节，高分辨率引入太多tokens让MLLM分心。这暗示ViT的patch embedding + attention机制在处理高分辨率时没有learned to selectively focus。

### Detection box对manipulation关键

移除detection box后GPT-4o从39.6%掉到27.1%。因为detection box提供了**explicit visual grounding**——把language里的"object 3"和image里的具体区域关联起来。

这就是Set-of-Mark prompting的思路：在image上画框标号，让MLLM能reference具体object。

Link: https://arxiv.org/abs/2310.11441

### Multi-step images反而有害

给agent看前两步的image，performance反而下降。Current MLLM处理多张sequential image时会confuse，搞不清哪张是current state。

**这是MLLM架构的根本limitation**。Vision encoder + LLM decoder的设计没针对temporal coherence优化。Video-LLM正在尝试解决这个，但还没scale到embodied场景。

### Visual in-context learning是大亮点

把in-context examples从纯文字改成"图片+文字"，Claude-3.5-Sonnet在manipulation上**提升16.7%**。

这个发现非常重要。意思是：**给agent看成功action的visual demonstration，比用文字描述更高效**。

这跟人类学习方式很像——你教小孩抓东西，是给他示范动作，而不是写说明书。Visual ICL可能是未来VLA training的重要方向。

---

## Error Analysis告诉了我们什么

GPT-4o的110个failure episode分析：

| Error Type | EB-ALFRED | EB-Manipulation |
|-----------|-----------|-----------------|
| Perception | 4% | 33% |
| Reasoning | 41% | 23% |
| Planning | 55% | 44% |

**Planning error占主导**。Missing steps（漏掉必要action）、invalid action（执行不了的动作）、wrong termination（以为任务完成了其实没有）是主要问题。

Perception error在manipulation上更高（33%），因为low-level task需要更精确的object识别和位置估计。

**关键insight**：瓶颈不在perception（vision encoder已经相当好），而在reasoning + planning。MLLM能看到东西，但想不清楚该怎么act。

---

## 跟VLA Models什么关系

EMBODIEDBENCH测的是**off-the-shelf MLLM的zero-shot能力**，没有任何robot data训练。VLA models（RT-2, OpenVLA, RDT-1B）是**在robot data上finetune过的**。

| | EMBODIEDBENCH | VLA Models |
|---|---|---|
| 训练数据 | 无robot data | 大量demonstration |
| Generalization | 宽泛任务覆盖 | 限于训练分布 |
| Action space | Discretized | Continuous |
| 样本效率 | 不需要robot data | 需要demo data |

EMBODIEDBENCH衡量的是**foundation model的generalization lower bound**。VLA是**task-specific upper bound**。

真正的突破应该来自两者结合：foundation model pretraining + lightweight task adaptation，可能通过visual ICL或test-time reasoning。

Link: 
- RT-2: https://robotics-transformer2.github.io  
- OpenVLA: https://openvla.github.io
- RDT-1B: https://thu-robomanipulation.github.io

---

## 这篇paper对未来的启示

### 1. Token prediction不足以解决embodied AI

从image tokens到action tokens的mapping需要spatial reasoning和long-horizon planning，纯autoregressive decoding搞不定。需要world model或structured reasoning。

### 2. Vision-language alignment还很shallow

High-level task上vision几乎不贡献，说明visual tokens还没真正integrate到reasoning pipeline。当前的multimodal fusion太shallow了。

### 3. Visual ICL是promising方向

+16.7%的提升暗示agent可以通过visual demonstrations学到task structure，不需要finetuning。这可能是未来embodied AI的关键。

### 4. Long-horizon需要memory

当前agent只用sliding window的recent history。解决long-horizon需要episodic memory或hierarchical planning。

### 5. 3D-aware perception是必须的

当前MLLM在2D image space里reasoning，spatial awareness subset表现差。3D scene representation（NeRF, 3DGS）作为中间layer可能help。

Link:
- NeRF: https://arxiv.org/abs/2003.08934
- 3DGS: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

### 6. World model是根本解

Multi-step image失败 + long-horizon失败都指向同一个方向：agent需要internal model of environment dynamics来plan。这是world model的核心motivation。

Link:
- DreamerV3: https://danijar.com/project/dreamerv3
- JEPA: https://openreview.net/forum?id=BZ5a1r-kVsf
- LeCun on world models: https://www.youtube.com/watch?v=5t1tB8xX1cA

---

## 最后的大白话

这篇paper本质上告诉我们：

**现在的multimodal LLM当robot大脑，high-level指挥还凑合，low-level控制基本废柴**。Vision module虽然接上了但reasoning还没真正用上它。Long-horizon planning是所有model的死穴。

**但visual in-context learning给了希望**——让agent看示范图片比读文字说明书更管用。

**下一步突破**大概来自world model + 3D perception + test-time reasoning的结合。纯靠scale up LLM大概率不够，需要架构创新。

说白了，embodied AI还是个open problem，EMBODIEDBENCH给我们画了张地图，告诉我们现在在哪儿，离目的地还有多远。

---

# EMBODIEDBENCH 深度技术讲解

## 1. Paper的核心动机与定位

EMBODIEDBENCH 是一个为 vision-driven embodied agents 设计的 comprehensive benchmark，填补了 MLLM-based embodied agents 评估领域的空白。在它之前，VisualAgentBench 是第一个 MLLM agent benchmark，但局限于 high-level planning。EMBODIEDBENCH 的核心创新在于同时覆盖 **high-level 和 low-level action tasks**，并引入 **capability-oriented fine-grained evaluation**。

从 Karpathy 你熟悉的视角来看，这个 benchmark 本质上是在回答一个问题：**当前 MLLMs 作为 vision-conditioned policies 的 universal approximator 究竟有多强？** 答案是：high-level symbolic reasoning 还行（GPT-4o 在 EB-ALFRED 56.3%, EB-Habitat 59.0%），但 low-level continuous control 几乎崩溃（EB-Manipulation 仅 28.9%）。

Project page: https://embodiedbench.github.io

---

## 2. POMDP 形式化的细节

Paper 将问题建模为 POMDP augmented with language instructions，定义为元组 (S, A, Ω, T, O, L, R)：

- **S**: complete state space，对 agent 不可观测
- **A**: action space，可以是 high-level 或 low-level
- **Ω**: visual perception space，每个 observation $I_t \in \Omega$ 是 time $t$ 的一帧图像
- **T**: transition dynamics $\tau: S \times A \to S$
- **O**: observation function，将 underlying state 映射到 agent 的 visual observation
- **L**: language instruction，定义 desired goal
- **R**: reward function，评估 task completion：
  $$r_t = \begin{cases} 1 & \text{if } s_t \models L \\ 0 & \text{otherwise} \end{cases}$$

这里 $s_t \models L$ 表示 state $s_t$ 满足 instruction $L$ 定义的逻辑约束（用 PDDL 描述）。

Agent 在 timestep $t$ 维护 history $h_t = (I_0, a_0, ..., I_{t-1}, a_{t-1}, I_t)$，通过 policy $\pi(a_t | L, h_t)$ 选择 action。目标是：

$$\max_\pi \mathbb{E}[r_\tau]$$

其中 $\tau$ 是 terminal timestep（任务完成或达到 max horizon）。

**Intuition building**: 这个 formulation 把 embodied agent 看作一个 partially observable decision-making 问题。关键 challenge 在于 $\pi$ 必须从 2D image $I_t$ + text instruction $L$ 直接映射到 action $a_t$，中间没有显式的 3D scene reconstruction 或 symbolic world model。这正是 MLLM 的 inductive bias 可能不够的地方——它们在 internet-scale image-text pairs 上训练，但缺少 **embodied causal reasoning** 和 **spatial-temporal consistency** 的训练信号。

---

## 3. Action Level Hierarchy

### Low-level action
Low-level action 对应 atomic robot commands。机械臂的 action 通常参数化为 7-dimensional vector：

$$a = [X, Y, Z, \text{Roll}, \text{Pitch}, \text{Yaw}, \text{Gripper}]$$

其中：
- $(X, Y, Z)$: incremental translational displacements（位移增量）
- $(\text{Roll}, \text{Pitch}, \text{Yaw})$: rotational deltas in Euler angles（欧拉角旋转增量）
- $\text{Gripper}$: binary open/closed state of end-effector

类似地，"move forward 0.1 m" 也算 low-level action，因为它 unambiguously 映射到 kinematic transformation。

### High-level action
High-level action 是 low-level primitives 的序列：

$$a^h = [a_1, a_2, ..., a_n]$$

其中每个 $a_i$ 都是可执行的 low-level primitive。例如 "find a HandTowel" 展开为：rotate certain degrees → scan for target → move towards it。

**Intuition**: High-level action space 的 cardinality 远小于 low-level action space，但每个 action 的"语义负担"更重。MLLM 在 high-level 上的优势来自于 LLM 的 symbolic reasoning 能力，但 low-level 失败的本质原因是 **continuous control 需要的 precision 不在 LLM 的 token-level discrete distribution modeling 范围内**。这与你在 Tesla 看到 autonomous driving 中 token prediction 与 continuous trajectory 之间的 gap 是同一个问题。

---

## 4. 四个环境的架构解析

### EB-ALFRED (High-level Household)
- 基于 ALFRED dataset + AI2-THOR simulator
- 8 个 high-level skill types: "pick up", "open", "close", "turn on", "turn off", "slice", "put down", "find"
- Action space **dynamic**：171 到 298 actions（取决于场景物体数量）
- 改进点：
  1. 支持 multi-instance（"find a cabinet 2"）
  2. 合并所有 "put down" 为单 action（因为 robot 一次只能 hold 一个 object）
  3. 修复 simulator bugs + refine instructions
- 300 test instances，6 subsets × 50 instances

### EB-Habitat (High-level Rearrangement)
- 基于 Language Rearrangement benchmark + Habitat 2.0 simulator
- 70 high-level skills，分为 5 类: "navigation", "pick", "place", "open", "close"
- 关键 constraint：navigation 只能到 receptacle-type objects（不像 ALFRED 可以到任何 object）
- 要求 robot visit 多个 location 来 find target
- 300 test instances，6 subsets × 50 instances

### EB-Navigation (Low-level Navigation)
- 基于 AI2-THOR
- 8 个 low-level actions:
  1. Move forward/backward by $\Delta x$
  2. Move rightward/leftward by $\Delta y$
  3. Rotate right/left by $\Delta \theta$ degrees
  4. Tilt camera upward/downward by $\Delta \varphi$ degrees
- Agent 只有 visual observation + textual feedback（无 direct positioning data）
- Success: reach within specified distance of target
- 300 test cases，5 subsets × 60 instances（无 spatial awareness subset）

### EB-Manipulation (Low-level Manipulation)
- 扩展自 VLMbench，使用 CoppeliaSim 控制 7-DoF Franka Emika Panda arm
- 4 task categories: Pick & Place, Stack, Shape Sorter, Table Wiping
- **关键 enhancement**: action space discretization（Yin et al., 2024）
  - Position components (x, y, z): 100 bins
  - Orientation components (roll, pitch, yaw): 120 bins
  - Example: $[x, y, z, \text{roll}, \text{pitch}, \text{yaw}, \text{gripper}] = [57, 61, 20, 10, 60, 25, 1]$
- 额外信息：YOLO detection boxes with index markers + 3D object pose estimation
- 228 instances，5 subsets（无 long-horizon subset）

**Intuition**: Action discretization 是一个 critical trick。它把 continuous control problem 重新转化为 classification problem，让 MLLM 的 token prediction capability 可用。这与 RT-2 等 VLA models 用 action tokens 的思路一致，但 EMBODIEDBENCH 保留了 semantic prompt structure 而不是 end-to-end finetuning。

参考 RT-2: https://robotics-transformer2.github.io

---

## 5. Vision-driven Agent Pipeline 架构

Agent 的输入包括：
1. Language instruction $L$
2. Visual perception $I_t$（当前步图像；少数情况 sliding window 多帧）
3. In-context demonstrations（few-shot examples）
4. Interaction history $h_t$
5. Task-specific information（skill sets / action format）

### Task Planner 的 5 步流程
每个 planning step：
1. Generate textual description of current visual input
2. Reflect on past actions and environmental feedback
3. Reason about how to achieve goal
4. Formulate language-based plan
5. Convert to executable plan in required JSON format

### Multi-step Planning 的关键设计
与 prior work（VisualAgentBench）每个 timestep 只 plan 一个 action 不同，EMBODIEDBENCH 支持 **multi-step planning**：agent 动态决定需要的 action 数量。

优势：
1. Better alignment with in-context examples for sequential decision-making
2. Reduced plan redundancy，尤其 low-level tasks 中 single action 对图像影响小，单步 planning 会导致过多 MLLM API calls

如果 plan failed 或 trigger invalid action，agent 从 latest state 重新 plan。

**Intuition**: Multi-step planning 本质上是在做 **open-loop control with closed-loop correction**。Agent 输出一个 short horizon 的 action sequence（类似 model predictive control），执行后基于 observation error 修正。这是 LLM agent 的一个 efficient design pattern，但本质问题在于 MLLM 无法 **internally simulate** action 的 consequence——它们只能基于 next observation 事后 correction。这与 world model（如 DreamerV3 或 LeCun 的 JEPA）有本质区别。

参考 DreamerV3: https://danijar.com/project/dreamerv3
参考 JEPA: https://openreview.net/forum?id=BZ5a1r-kVsf

---

## 6. 实验数据深度解读

### 6.1 Overall Performance 表格分析

**Table 2 (High-level tasks)**:
| Model | EB-ALFRED Avg | EB-Habitat Avg |
|-------|---------------|----------------|
| Claude-3.5-Sonnet | **64.0** | **68.0** |
| Claude-3.7-Sonnet | 67.7 | 58.7 |
| Gemini-1.5-Pro | 62.3 | 56.3 |
| GPT-4o | 56.3 | 59.0 |
| InternVL3-78B (best open-source) | 39.0 | 55.0 |
| Qwen2.5-VL-72B | 39.7 | 37.7 |
| Llama-3.2-90B-Vision | 32.0 | 40.3 |

**Table 3 (Low-level tasks)**:
| Model | EB-Navigation Avg | EB-Manipulation Avg |
|-------|-------------------|---------------------|
| GPT-4o | **57.7** | **28.9** |
| Claude-3.7-Sonnet | 45.0 | 28.5 |
| Claude-3.5-Sonnet | 44.7 | 25.4 |
| InternVL3-78B | 53.7 | 26.3 |
| Gemini-2.0-flash | 48.7 | 16.7 |

**关键观察**：
1. 不同 model 在不同 task level 各有所长：Claude-3.5 在 high-level 最强，GPT-4o 在 low-level 最强
2. Open-source (InternVL3-78B) 在 low-level 上接近 GPT-4o，但 high-level 仍有 gap
3. Open-source models 展现明确 scaling trend（size 增加 → performance 提升）

### 6.2 Vision 的角色：令人震惊的发现

| Setup | EB-Navigation (GPT-4o) | EB-ALFRED (GPT-4o) |
|-------|------------------------|---------------------|
| With Vision | 57.7 | 56.3 |
| Lang only | 17.4 | 58.0 |

**EB-Navigation 上 vision 移除导致 40%+ 性能下降**，long-horizon subset 直接 collapse 到 0%。但 **EB-ALFRED 上 vision 移除反而略有提升**！

这个现象的 deep reason：high-level tasks 主要依赖 textual feedback（PDDL-style state info）和 symbolic reasoning；low-level tasks 需要 pixel-level perception 来推断 spatial relationship。这暗示 **当前 MLLM 的 visual encoder 在 high-level reasoning 中的作用还没有被充分激活**——它们更多是 attach 一个 vision module 给 LLM，但 LLM 本身的 reasoning 还没学会真正 ground 到 visual features。

### 6.3 Long-Horizon 是最大瓶颈

EB-Habitat:
- Claude-3.5-Sonnet: Base 96% → Long-Horizon 58%
- GPT-4o: Base 86% → Long-Horizon 64%

EB-ALFRED:
- GPT-4o-mini: Base 34% → Long-Horizon 0%

这个 pattern 在所有 model 上一致。**Intuition**: Long-horizon planning 需要 **compositional generalization**——model 必须在 sub-task 之间 maintain intermediate state、infer pre-conditions 和 post-conditions，并且 detect sub-task completion。当前 MLLM 的 autoregressive decoding 没有显式的 memory mechanism，导致长序列 reasoning 的 error 累积。这与你在 tweet 中提到的 "LLMs are fundamentally limited by their autoregressive nature" 完全一致。

---

## 7. Ablation Studies 细节

### 7.1 Language-centric Ablation (Figure 4)

在 EB-ALFRED Base subset 上：
- 移除 environment feedback → GPT-4o -10%, Claude-3.5 -8%
- 0-shot setting（无 in-context examples）→ ~40% success rate
- 减少 in-context examples 显著影响 performance

**Intuition**: 这说明 high-level tasks 的 reasoning 很大程度上 ride on in-context demonstrations 的 pattern matching，而不是真正的 task understanding。这是 LLM 的 known limitation——它们是 impressive pattern completers，但缺乏 causal task model。

### 7.2 Visual-centric Ablations (Figure 5)

#### (a) Camera Resolution
- 300×300: 细节不足
- 500×500: **最优**
- 700×700: 过高分辨率引入 noise，降低 performance

**Intuition**: 这与 ViT 的 patch embedding 设计直接相关。MLLM 通常用 14×14 或 16×16 patch size，更高分辨率意味着更多 tokens，但 attention 机制没有 learned to selectively focus on relevant tokens。这个 finding 强烈暗示需要 **visual token pruning** 或 **hierarchical attention**。

#### (b) Detection Boxes
移除 detection boxes：
- GPT-4o: 39.6% → 27.1%
- Claude-3.5: 37.5% → 29.2%

Detection boxes 帮助 model 把 language instruction 和 visual object 关联起来，相当于一个 **explicit visual grounding**。这与 Set-of-Mark prompting (Yang et al., 2023a) 思路一致。

参考 SoM: https://arxiv.org/abs/2310.11441

#### (c) Multi-step Images
Surprisingly，添加历史图像（前 2 步 + 当前步）导致 performance decline。Current MLLMs 难以理解多张 sequential images 之间的关系，会 confuse current state。

**Intuition**: 这是 MLLM architecture 的 fundamental limitation。Vision encoder + LLM decoder 的 design 没有针对 temporal coherence 优化。Video-LLM 的研究（如 VideoChat, LLaMA-Vid）正在尝试解决这个问题，但还没有 scale 到 embodied agent 的场景。

参考 VideoChat: https://arxiv.org/abs/2305.01000

#### (d) Visual In-context Learning
**这是最 promising 的发现**：在 EB-Manipulation 上，用图像作为 in-context examples（而不是纯文本 examples），Claude-3.5-Sonnet 获得 **+16.7% performance boost**。

**Intuition**: Visual ICL 让 model 看到 "successful low-level action 对应的 object position in image"，建立了 visual feature → action 的直接 mapping。这是少有的 MLLM 能从 visual information 中获益的 case，说明 **structured visual demonstrations 比文字描述更高效**。这可能是未来 VLA training 的一个重要方向——让 model 在 visual demonstration space 内做 in-context generalization。

---

## 8. Error Analysis 深度解析

Paper 对 GPT-4o 在 110 个 failure episodes 上做了 error analysis，分三类：

### Perception Errors
- EB-ALFRED: 仅 4%
- EB-Manipulation: 33%（其中 wrong recognition 22%）

### Reasoning Errors
- EB-ALFRED: 41%
- 主要 sub-error: reflection error (17%) → model 无法从 action history 中识别 planning mistakes

### Planning Errors
- EB-ALFRED: 55%（missing steps 23% + invalid actions 22% + wrong termination 13%）
- EB-Manipulation: 44%（inaccurate actions，difficulty estimating gripper poses）

**Intuition**: 这个 distribution 揭示了一个重要 fact——**当前 MLLM 的瓶颈不在 perception（vision encoder 已经相当好）而在 reasoning + planning**。即使 GPT-4o 在图像中看到了 object，它也无法可靠地推断出正确的 action sequence。这暗示我们需要 **更好的 reasoning architectures**，可能是 system-2 thinking（如 OpenAI o1 的 CoT reasoning）或 **neural-symbolic hybrid** approaches。

参考 OpenAI o1: https://openai.com/o1
参考 AlphaProof (DeepMind): https://deepmind.google/discover/blog/ai-system-advances-frontiers-of-mathematical-reasoning/

---

## 9. 与 VLA Models 的关系（重要联想）

Paper 提到 VLA models（RT-1, RT-2, OpenVLA, Diffusion Policy, RDT-1B）作为相关方向。EMBODIEDBENCH 评估的是 **off-the-shelf MLLMs 作为 zero-shot agents**，而不是 finetuned VLA models。这两者的关系：

| Approach | EMBODIEDBENCH setting | VLA models |
|----------|----------------------|------------|
| Training | Zero-shot inference | End-to-end finetuning on robot data |
| Generalization | Broad task coverage | Limited to training distribution |
| Action space | Discretized / symbolic | Continuous |
| Sample efficiency | No robot data needed | Require demonstration data |

**Intuition**: EMBODIEDBENCH 实际上在衡量 **foundation model 的 generalization lower bound**——没有任何 task-specific training 的情况下能做什么。VLA models 是 upper bound 思路——通过 task-specific training 最大化 performance。**真正的突破**应该来自 **foundation model pretraining + lightweight task adaptation**，可能通过 visual ICL 或 test-time reasoning。

参考 OpenVLA: https://openvla.github.io
参考 Diffusion Policy: https://diffusion-policy.cs.columbia.edu
参考 RDT-1B: https://thu-robomanipulation.github.io

---

## 10. EMBODIEDBENCH 的 Limitations 与 Critical Thoughts

Paper 自己承认：仅 simulator-based evaluation。这是 embodied AI benchmarking 的 common trade-off：

| Approach | Pros | Cons |
|----------|------|------|
| Simulation | Reproducible, safe, cheap | Sim-to-real gap |
| Real-world | Ground truth | Expensive, hard to reproduce |

### 额外 critical points

1. **Action discretization in EB-Manipulation**: Discretizing continuous action space 是 pragmatic trick，但本质上丢掉了 continuous control 的 fine-grained information。真正的 embodied agent 应该输出 continuous actions。

2. **Few-shot ICL dependence**: 0-shot 性能只有 40% 左右，说明 model 主要在做 pattern matching，不是真正 understanding task。

3. **No real vision processing**: EB-ALFRED 上移除 vision 反而提升 performance 暗示 visual encoder 还没真正 integrate 到 reasoning pipeline 中。这是 multimodal model 的 general issue——modality fusion 还是 shallow 的。

4. **No active perception**: Agent 被动接收 observation，不能主动 control camera viewpoint（除了 EB-Navigation 的 tilt）。这与现实 robotic system 不符。

5. **No memory mechanism**: Agent 只用 sliding window 的 recent history。Long-horizon 失败部分是因为没有 episodic memory。这与 LeCun 的 H-JEPA 或 memory-augmented transformers 思路相左。

参考 H-JEPA: https://yann-lecun.com/jepa

---

## 11. 对 Karpathy 思路的 Connection

你近期在 educational content 中提到 **"building AI systems that can act in the world"** 是 next frontier。EMBODIEDBENCH 提供了一些 supporting data points：

1. **Token prediction 不是 sufficient**：从 image tokens 到 action tokens 的 mapping 需要 spatial reasoning 和 long-horizon planning，token-level autoregressive 不足以解决。

2. **Vision-language alignment 仍然是 shallow**：high-level tasks 上 vision 几乎不贡献，说明 visual tokens 还没真正被 LLM 的 reasoning 利用。

3. **In-context learning 是 promising direction**：Visual ICL 的 +16.7% 提升暗示 model 可以通过 visual demonstrations 学到 task structure，不需要 finetuning。

4. **World model 是必要的**：Multi-step image 失败和 long-horizon 失败都暗示 agent 需要 internal model of environment dynamics 来 plan。这是 world model 的核心 motivation。

参考你提到的 world models: https://worldmodels.github.io
参考 Yann LeCun on world models: https://www.youtube.com/watch?v=5t1tB8xX1cA

---

## 12. Future Research Directions (我的联想)

基于 EMBODIEDBENCH 的 findings，以下方向 promising：

1. **Visual ICL at scale**: 在 EB-Manipulation 上的 +16.7% 提升是 striking。可以 explore retrieval-augmented visual demonstrations，让 agent 动态 fetch 最相关的 visual examples。

2. **Hierarchical planning**: 用 LLM 做 high-level task decomposition，用 smaller VLA model 做 low-level execution。EMBODIEDBENCH 的 action level hierarchy 自然支持这个 design。

3. **Test-time compute for embodied reasoning**: 类似 OpenAI o1 的思路，在 action selection 前进行 long-form reasoning。Long-horizon subset 的失败暗示需要更多 reasoning compute。

4. **World model integration**: 让 agent 先在 internal world model 中 simulate action consequences，再 execute。这是解决 multi-step planning 失败的 fundamental approach。

5. **Active perception training**: 训练 agent 主动选择 viewpoint，而不是被动接收 observation。EB-Navigation 中 tilt camera 是一个 simplified version。

6. **Memory-augmented embodied agents**: 引入 episodic memory，让 agent 记住 visited states 和 failed actions，避免 long-horizon 中的重复 errors。

7. **3D scene representation**: 当前 MLLM 在 2D image space 中 reasoning，spatial subset 性能较差。3D neural scene representation（如 NeRF, 3D Gaussian Splatting）作为中间 layer 可能 help。

参考 NeRF: https://arxiv.org/abs/2003.08934
参考 3DGS: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

---

## 13. 论文的 Take-away Summary

| Finding | Implication |
|---------|-------------|
| High-level >> Low-level performance | Current MLLMs 缺乏 continuous control capability |
| Vision matters for low-level, not high-level | Visual grounding 仍未真正 integrate 到 reasoning |
| Long-horizon is biggest bottleneck | Autoregressive decoding 累积 error |
| Visual ICL 提供 +16.7% | Visual demonstrations 是 efficient communication |
| Multi-step images 反而 harmful | Current MLLM 缺乏 temporal reasoning |
| InternVL3-78B 接近 GPT-4o 在 low-level | Open-source 在 catch up |

EMBODIEDBENCH 本质上告诉我们：**off-the-shelf MLLMs 作为 embodied agents 还远远不够**，但 **visual in-context learning 和 hierarchical action space design 是 promising directions**。下一步突破可能来自 **world models**、**3D-aware perception** 和 **test-time reasoning** 的结合。

---

## Reference Links

- EMBODIEDBENCH Project: https://embodiedbench.github.io
- AI2-THOR: https://ai2thor.allenai.org
- Habitat: https://aihabitat.org
- ALFRED: https://askforalfred.com
- VisualAgentBench: https://visualagentbench.github.io
- RT-2: https://robotics-transformer2.github.io
- OpenVLA: https://openvla.github.io
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu
- DreamerV3: https://danijar.com/project/dreamerv3
- Yann LeCun on JEPA: https://openreview.net/forum?id=BZ5a1r-kVsf
- Set-of-Mark Prompting: https://arxiv.org/abs/2310.11441
- OpenAI o1: https://openai.com/o1
- InternVL: https://internvl.github.io
- Qwen2.5-VL: https://qwenlm.github.io/blog/qwen2.5-vl/
- VLABench: https://github.com/OpenRobotLab/VLABench

如果你对某个 specific finding 想深入讨论，特别是 visual ICL、long-horizon planning 的 root cause，或 action discretization 的 trade-offs，我可以进一步展开。
