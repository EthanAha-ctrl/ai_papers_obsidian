---
source_pdf: ESI-BENCH Towards Embodied Spatial Intelligence that Closes the Perception-Action
  Loop.pdf
paper_sha256: bf5f11b8ffc8a3ce3add08ad4f833ffcec6ceab47e2cbed8e8a24a7f8a9469f9
processed_at: '2026-08-04T05:06:57-07:00'
target_folder: AI在行业应用
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 ESI-BENCH

## 一句话总结

**以前的 spatial benchmark 就是给 model 一张图问 "这个离那个多远"，这篇 paper 说这不对——真正 spatial intelligence 是你自己决定走到哪儿看、拿起什么、什么时候停。**

---

## 为什么要做这个

想象一下你问一个人 "柜子后面藏的是什么？"

**老做法**: 给他一张从正面拍的照片，让他猜。猜对了就算他 spatial intelligence 强。

**问题**: 这根本没测到 spatial intelligence 的核心。真正 spatially intelligent 的人会**自己走过去绕到柜子后面看一眼**，或者**把柜门打开**，或者**从上面探头看**。他知道**该用什么 action 去获取信息**，而不是被动接受给定的信息。

prior 的 benchmark（VSR、BLINK、VSI-Bench、MMSI-Bench、MindCube、PhysBench...）全都假设 observation 是 fixed 的、oracle 的。agent 拿到什么看什么，不能决定去看什么。

ESI-BENCH 的核心 idea: **把 observer 变成 actor**。agent 要自己决定 deploy 什么 ability（perception / locomotion / manipulation），采取什么 action，按什么顺序执行。

---

## Benchmark 长什么样

### 环境用的是啥

**OmniGibson** + **BEHAVIOR-1K**，Stanford 自己搞的 simulation。51 个 3D scene（住宅、商业、机构），300+ rooms，9000+ object instances，1829 个 category，物理引擎是 NVIDIA Isaac Sim + PhysX 5，支持 rigid-body physics、fluid particle、transparency、reflection、lighting。

简单说就是个挺真实的 3D 物理模拟器。

### Task 怎么定义

一个 task 是 $(S, p_0, q, y^*)$:
- $S$: scene
- $p_0$: agent 起始位置
- $q$: 问题
- $y^*$: 答案

agent 每步看一张 egocentric 图（第一人称视角），从 action space 选一个 action 执行，最多 30 步，然后 commit 一个 answer。

### Action space

三类 action:
- **Locomotion**: 前后左右上下移动
- **Perception**: 左右上下转头
- **Manipulation**: pick_up、put_inside、put_on、fill_with_water、pour
- **Terminal**: answer(答案, 置信度)

paper 在 Appendix O 解释为啥用 high-level discrete action 而非 low-level motor control——因为这 benchmark 测的是 **spatial reasoning**，不是 visuomotor control。low-level 会引入 grasp instability、collision、drift 等噪声。

### Task 怎么构造的

**用 GPT-4o 当 proposal engine**:
1. GPT-4o 看 scene graph + category requirement，从 200 candidate category 里选 task-relevant object
2. 决定 object 和 agent 初始位置
3. 生成 ground-truth action trajectory

**但 GPT-4o 说了不算**。每个 task 都要：
- 在 OmniGibson 里 instantiate + physics settle
- 跑 verification check（bbox intersection、stability、per-view object existence via segmentation mask、contact-flag validation）
- 3 个 human annotator 独立 review，majority vote
- 三个维度: correctness（物理语义一致）、answerability（通过 action space 能解）、non-triviality（不能从初始 view 或 prior 直接答）

### Task 类别

基于 Spelke 的 four core knowledge systems（developmental psychology 里 infant 天生的 spatial reasoning 基础）:

- **Object representation**: Perceptual Grounding、Physical Structure
- **Layout & geometry**: Spatial Relations、Metric Comparison、Cognitive Mapping、Specular Reflection
- **Number representation**: Enumerative Perception
- **Agents & goal-directed actions**: Temporal Understanding、Action Sequencing、Physical Dynamics

10 个 category，29 个 subcategory，3081 个 task instance。

比如:
- "柜子里藏的是啥？"（Rigid Containment）
- "三个杯子排成 equilateral triangle 吗？"（Geometric Configuration）
- "从卧室到客厅必须经过厨房吗？"（Cognitive Mapping - Traversable Passage）
- "毯子底下有几个球？"（Counting w/ Occlusion）
- "另一个机器人路过几个 mug？"（Agent Observation）

每个 category 需要不同 embodied ability 组合——有些主要靠 manipulation，有些主要靠 locomotion。

---

## 实验设计: 四个 paradigm

这是 paper 最精巧的地方。设了四个条件来 isolate 不同因素:

1. **Passive Single-View**: 一张图，不能动（对齐 prior benchmark 的 baseline）
2. **Passive Multi-View**: 给 30 张随机视角的图（match active budget 30 步），但 agent 不能选
3. **Active Exploration**: agent 自己用 action space 探索
4. **Ground-Truth Passive**: 沿 ground-truth trajectory 渲染的 views（oracle，perfect viewpoint）

对比逻辑:
- (1) vs (2): 给更多 passive view 有用吗
- (2) vs (3): 主动选视角 vs 被动给 30 张图
- (3) vs (4): 失败是因为不会选 action 还是因为 perception 不行

---

## 三个核心发现

### 发现一: 主动探索 >> 被动多视角

**最 striking 的对比**:

| Task | GPT-5 单视角 | GPT-5 30张随机 | GPT-5 主动 |
|---|---|---|---|
| View Hallucination | 11.7 | 20.2 | 60.1 |
| Partial Occlusion | 32.6 | 26.3 | 47.4 |
| Physical Contact | 40.0 | 41.7 | 64.2 |
| Spatial Distance | 53.9 | **49.1** | 58.6 |

**两个 critical observation**:

**(a) 给 30 张随机图有时比给 1 张还差**（Spatial Distance 从 53.9 跌到 49.1）。这非常反直觉。为啥？paper 在 Appendix L 分析:
- 随机 view 容易 miss diagnostic region
- 30 张图 overload MLLM 的 integration 能力
- 没有 action grounding，spatial integration 困难
- partial occluded view 引入 conflicting cue

**(b) 没给任何 instruction，agent 自己 develop 出多种策略**

比如判断 "栗子在不在杯子里"，agent 自发 develop 出 4 种方法:
- 绕到后面看
- 从上往下看
- pick up
- pour 出来

这四种都不是人教的。paper 在 Appendix I 量化 strategy diversity，Material Transparency 有 6 个 cluster，Partial Occlusion 有 4 个，等等。

**这暗示 active exploration 的 benefit 来自 selective evidence acquisition，不是简单 observation accumulation。** Figure 9 的 step budget ablation 证明这点: 5→15 步快速提升，15-20 步 saturate，40 步后略降——说明有用的就前几步 informative action，之后是 redundant noise。

### 发现二: Action blindness >> Perceptual blindness

**对比 Active vs GT Passive**:

| Task | GPT-5 Active | GPT-5 GT Passive | Gap |
|---|---|---|---|
| Rigid Containment | 42.5 | 95.0 | 52.5 |
| Physical Contact | 64.2 | 90.0 | 25.8 |
| Counting w/ Occlusion | low | high | 43.4 |
| Structural Enclosure | low | high | 45.0 |

**结论**: 给 perfect viewpoint，agent 基本都能答对。bottleneck 不在 perception，在 action selection——agent 不知道去哪儿看、看什么、怎么 manipulate。

**但有两个 exception 暴露 hard perceptual ceiling**:

- **Geometric Configuration**: GT Passive 也只有 26.0%！即使从 perfect viewpoint 看清三个 object 的相对位置，model 也判断不出是不是 equilateral triangle。说明 MLLM 缺 **metric geometric reasoning** 能力——visual encoder 能提取 position，但 LLM 推理层无法 do precise geometric computation。
  
- **Specular Reflection**: model 在 mirror 里 hallucinate 不存在的 object，或找不到正确 correspondence。perception 根本没到推理层就错了。

**Failure cascade 现象**: suboptimal action → uninformative view → worse subsequent action → compounding error chain，在 step budget 内无法 recover。坏 action 会传染给后面的 reasoning。

### 发现三: 3D help when perfect, hurt when noisy

paper 还测了 3D-augmented model:

- **VGGT+Gemini**: 用 VGGT 从 multi-view 重建 3D，construct scene graph 给 LLM
- **GT 3D+Gemini**: 直接用 simulator state derive perfect point cloud，construct scene graph（oracle）

| Task | Gemini 3.1 (2D) | VGGT+Gemini (noisy 3D) | GT 3D+Gemini (perfect 3D) |
|---|---|---|---|
| Geometric Configuration | 27.5 | **9.9** | 70.8 |
| Counting w/ Occlusion | 3.3 | **0.0** | 33.3 |
| Material Transparency | 44.0 | 27.8 | 60.4 |

**Perfect 3D 大幅 help**: Geometric Configuration 从 27.5% → 70.8%（+43.3）。说明 explicit geometry 能 resolve 2D projection fundamentally ambiguous 的 depth/occlusion 问题。

**Noisy 3D 反而 hurt**: VGGT+Gemini 在 Geometric Configuration 跌到 9.9%，Counting w/ Occlusion 直接 0.0%。比 2D baseline 还差。

为啥？Appendix J 分析了 VGGT 的三种 failure mode:

1. **Object duplication**: 部分观测的 object 被重建为多个 fragment，downstream scene graph over-count
2. **Object hallucination**: noisy geometry 产生 spurious object proposal
3. **Spatial-relation corruption**: depth error 扭曲 relative position / contact / containment

**critical insight**: LLM 倾向于把 reconstructed scene graph 当作 reliable symbolic description 来 trust，所以 noisy reconstruction 会 actively mislead reasoning。3D augmentation 是个 **high-variance strategy**，amplify both success 和 failure。

这暗示 future 方向: **uncertainty-aware scene-graph construction**——能区分 reliable 3D evidence vs ambiguous/fragmented geometry，让 LLM 知道哪部分 geometry 该信、哪部分该怀疑。

### 发现四: Metacognitive gap

**Passive 设定下 human 和 model 性能相当**:
- Material Transparency GT Passive: GPT-5 96.3% vs Human 97.2%
- Partial Occlusion GT Passive: Gemini 88.4% vs Human 87.4%

**Active 设定下 human 大幅领先**:
- Physical Contact: Human 88.3% vs GPT-5 64.2%
- Material Transparency: Human 93.6% vs Gemini 52.3%

paper 在 Appendix K 把这个 gap operationalize 成 3 个 trajectory-level measure:

| Agent | View Diversity | Contrastive View Rate | Belief Revision Rate |
|---|---|---|---|
| Human | 71.8 | 62.7 | 41.3 |
| Gemini | 43.5 | 31.5 | 18.9 |
| GPT-5 | 39.2 | 28.7 | 16.4 |

**Human 的行为模式**:
- gather 更多 observation 再 commit
- **seek 能 falsify 当前 hypothesis 的 viewpoint**——主动找可能证明自己错的视角
- 在 contradiction 下 **revise belief**

**Model 的行为模式**:
- **premature commit with high confidence**（Figure 4d，几步就答且 confidence 高）
- **directional bias**: 反复 move in same direction 累积 redundant 而非 informative observation
- **anchor to first impression**: 不 revise

比如 Figure 4e 的 case: 问 "是 cupboard 还是 piano？"，model 先猜 cupboard，然后 backward + left 移动，**只为了 confirm 自己的猜测**，从不 seek falsifying viewpoint，最后 assert 错误答案且 confidence 升高。

**核心论点**: 这是 **epistemic calibration failure**，fundamentally distinct from perceptual ability。**better visual encoder 解决不了，更多 exploration step 也解决不了**。它需要 metacognitive 能力: 知道 evidence 何时 insufficient，知道如何 acquire better evidence，知道何时该 revise belief。

---

## Generator bias 的 audit

因为用 GPT-4o 做 proposal engine，paper 仔细 audit 了 bias（Appendix H）:

- **Question-only baseline**（只给 question，没图没 metadata）: average 36.6%，低于 passive single 42.5%，远低于 active 56.9%。说明 question wording 不含 answer artifact
- **Metadata-only baseline**（给 question + task-relevant object category）: 39.3%，仍低于 passive single。说明 object identity 不是 shortcut
- **Answer balance / Object diversity**: 归一化 entropy 0.92 / 0.85，很高
- **GPT-4o vs human-generated task**: matched subset 上 accuracy gap 在 3.5 points 内，说明 GPT-4o 生成的 task 难度和 human 相当

这表明用 GPT-4o 做 proposal engine，**只要过 simulator filtering 和 human verification，就不会系统扭曲 benchmark 难度**。

---

## Step budget ablation 的 intuition

Figure 9 显示 accuracy 随 budget 变化: 5→15 步快速提升，15-20 步 saturate，40 步后略降。

这跟 active perception 文献里的 "information foraging" 理论一致: **optimal exploration 应该在 marginal information gain 低于某个 threshold 时停止**。

模型缺乏这个 stopping criterion，所以要么 premature commit，要么继续探索 redundant view。这暗示需要 learn-to-stop 的机制。

---

## 我看到的方向

1. **Uncertainty-aware 3D**: 在 VGGT+Gemini 失败的 case 里，如果能传 reconstruction uncertainty（per-point confidence、fragmentation score），LLM 或许能 avoid treating noisy 3D as ground truth。这连接到 Bayesian active learning

2. **Belief revision training**: model 的 belief revision rate 仅 16-19%，human 41%。能否通过 RLHF 或 self-play 训练 model 在 contradiction 下 update belief？这接近 "epistemic humility" training

3. **Falsification-seeking reward shaping**: 在 active exploration 训练中，除了 task success reward，加一个 "contrastive viewpoint bonus"——鼓励 agent 探索能 disconfirm 当前 hypothesis 的 view

4. **Stopping criterion learning**: 能否训练一个 value function $V(s_t, q)$ 估计 "当前 evidence 是否足够 answer $q$"？本质是 learn-to-stop 问题

5. **Real-world transfer**: paper 承认 sim-to-real gap 是 limitation。能否用 similar task structure 在 real robot 上 collect small dataset 来 validate sim finding？

---

## 总结成最朴素的几句话

**以前 spatial benchmark 是闭卷考试**——给图答题。ESI-BENCH 是开卷考试但你得自己去找书——agent 决定看哪、动哪、摸啥、何时停。

发现: 给 30 张随机图有时不如 1 张图，因为信息得 agent 自己 select 才有用。主动探索能自发 develop 多种 strategy，但瓶颈在 action selection 不在 perception。3D 信息 perfect 时 help，noisy 时 hurt。Human 和 model 在 passive 下相当，active 下 human 大胜——因为 human 会 seek falsifying view 并 revise belief，model premature commit 且 directional bias。

这本质上是 **epistemic calibration gap**，不是 perception 问题，也不是 exploration step 不够的问题。需要的是 metacognitive 能力。

paper 项目主页: https://esi-bench.github.io/

---

# ESI-BENCH 深度解析

这是 Stanford 团队（Yining Hong 等）做的一篇非常有意思的工作，核心 motivation 是把 spatial intelligence 从 **passive observation** 范式推进到 **perception-action loop** 范式。下面我从 motivation、benchmark 设计、实验结果三个层面详细讲解，并 build 你的 intuition。

---

## 1. Motivation: 为什么要做 Embodied Spatial Intelligence？

### 1.1 对 prior work 的 critique

prior spatial reasoning benchmark 比如 VSR、BLINK、3DSRBench、VSI-Bench、MMSI-Bench，甚至 MindCube（partial observation mental modeling）、PhysBench（latent physical structure）这些更 advanced 的工作，都有一个共同假设：**observation 是 fixed/oracle 的**。

ESI-BENCH 的核心论点是：spatial intelligence 本质上是 embodied 的——agent 必须决定 **deploy 什么 ability**（perception / locomotion / manipulation）、**采取什么 action**（去哪里 move、probe 什么、怎么 manipulate）、**以什么顺序执行**。这正好对应 Gibson 的 "perceptually guided action" [Gibson 1979] 和 O'Regan & Noë 的 sensorimotor account of vision [O'Regan & Noë 2001]。

### 1.2 三个核心 shift

paper 在 Introduction 里提了三个对比维度：

1. **From spatial sensing to spatial competence**: 不仅评估 agent 能 perceive 什么，还评估它知不知道 deploy 哪些 embodied ability
2. **Selective sensing**: agent 必须判断哪些 observation 值得 acquire，prioritize task-relevant information
3. **Resolving perceptual ambiguities**: 通过 misleading observation 推理 hidden spatial structure

### 1.3 Theoretical foundation: Spelke's core knowledge systems

ESI-BENCH 的 category 设计基于 Spelke & Kinzler [2007] 的四个 core knowledge systems：

- **Object representation**（物体表征）
- **Layout and geometry**（布局与几何）
- **Number representation**（数表征）
- **Agents & goal-directed actions**（agent 与目标导向行为）

这个 framework 来自 developmental psychology，刻画了 infants 与生俱来的 spatial reasoning 基础。paper 做了 human survey 来 identify 每个 faculty 中最 challenging 的、需要 embodied interaction 的 task。

参考链接：
- Spelke core knowledge: https://en.wikipedia.org/wiki/Core_knowledge_(Spelke)
- BEHAVIOR-1K: https://behavior.stanford.edu/
- OmniGibson: https://github.com/StanfordVL/OmniGibson

---

## 2. Benchmark 设计

### 2.1 Task formalization

每个 task 是一个 tuple $(S, p_0, q, y^*)$，其中：

- $S$: 从 BEHAVIOR-1K scene pool 实例化的 3D scene，pre-loaded objects
- $p_0$: agent 的 initial pose
- $q$: 关于 scene 某个 spatial property 的自然语言问题
- $y^*$: ground-truth answer

环境形式化为 $\mathcal{E} = \langle S, \mathcal{A}, \mathcal{O}, T \rangle$，其中：

- $\mathcal{A}$: action space
- $\mathcal{O}$: egocentric observation space
- $T: \mathcal{S} \times \mathcal{A} \to \mathcal{S}$: scene transition function，控制 scene state 如何随 action 变化

agent 在每个 timestep 接收 observation $o_t \in \mathcal{O}$，发出 action $a_t \in \mathcal{A}$，产生 trajectory：

$$\tau = (o_0, a_0, o_1, a_1, \dots)$$

直到 commit final answer $\hat{y}$，budget $T_{\max} = 30$ steps。Terminal action 是 `answer(ŷ, c)`，其中 $c \in [0, 1]$ 是 confidence。响应正确当且仅当 $\hat{y} = y^*$。

**intuition**: 这里的设计哲学是 "answer 是 free-form 的，但 question phrasing 隐式指定了 expected format"——yes/no 用于 relational task，category 用于 comparison，integer 用于 counting，ordering 用于 procedural task。这避免了 rigid output schema 但又保证可评估。

### 2.2 Action space

| Action Type | Actions | Description |
|---|---|---|
| Locomotion | move_forward/backward/left/right/up/down | 沿 viewing axis / lateral / vertical 平移 |
| Perception | turn_left/right/up/down | 水平/垂直旋转 |
| Manipulation | pick_up, put_inside, put_on, fill_with_water, pour | 物体交互 |
| Terminal | answer(ŷ, c) | 提交答案+置信度 |

paper 在 Appendix O 里讨论了为什么用 **high-level discrete action** 而非 low-level motor control：因为 benchmark 要诊断的是 **embodied spatial reasoning**，而不是 visuomotor control。引入 low-level control 会带来 grasp instability、collision recovery、locomotion drift 等噪声源，难以判断 error 是来自 spatial reasoning 还是 motor execution。

### 2.3 Simulation environment: OmniGibson + BEHAVIOR-1K

- **BEHAVIOR-1K**: 51 个 interactive 3D scenes，涵盖 residential / commercial / institutional，300+ rooms，9k+ object instances，1829 categories，带 friction、mass、articulation 等物理属性
- **OmniGibson**: 基于 NVIDIA Isaac Sim 和 PhysX 5，支持 rigid-body contact physics、particle-based fluids、transparency rendering、realistic lighting/reflections、extended object states（fill levels、toggled states）

每个 task instance 的 pipeline：
1. 随机采样 BEHAVIOR-1K scene
2. 根据 room type 和 task-category requirement 选 room
3. 加载到 OmniGibson，让 physics settle
4. 查询 simulator state，extract structured scene graph（object bboxes、categories、spatial relations、room assignments、states）

### 2.4 Task construction pipeline

**Task Proposal**: GPT-4o 被 prompt 以 scene graph + task category requirements，从 200 candidate categories（从 1829 全 inventory 中随机采样）中选 task-relevant objects，并 determine initial positions 和 ground-truth action trajectory。

**Scene Instantiation**: 用 bbox intersection test 检查冲突，通过 physics-based kinematic sampling 放置 object，settle 固定 simulation steps，再做 stability check（re-query bboxes、per-view object existence via segmentation masks、contact-flag validation）。失败则 reject。

**Human Verification**: 3 个 annotator 独立 review，majority vote。三个 criteria：
- **Correctness**: physical/semantic consistency with simulator state
- **Answerability**: 通过 available action space 是否能 acquire required evidence
- **Non-triviality**: 不能从 initial observation 或 prior knowledge 直接答出

### 2.5 Task taxonomy: 10 categories × 29 subcategories

| Category | Subcategories |
|---|---|
| **Physical Structure** | Rigid Containment, Liquid Volume, Deformable |
| **Physical Dynamics** | Inclined Plane, Stacking & Stability |
| **Specular Reflection** | Reflection Authoring, Spatial Relations, Correspondence |
| **Perceptual Grounding** | Partial Occlusion, View Hallucination, Material Transparency |
| **Metric Comparison** | Dimensional Size, Spatial Distance |
| **Spatial Relations** | Linear Alignment, Geometric Configuration, Physical Contact |
| **Cognitive Mapping** | Connectivity, Traversable Passage, Regional Boundary, Long-Term Navigation |
| **Enumerative Perception** | Counting w/ Occlusion, Spatial Segmentation, Category Ambiguity, Merged Observation, Illumination Variability, Structural Enclosure |
| **Temporal Understanding** | Unobserved Change, Agent Observation |
| **Action Sequencing** | Action Order Inference |

总共 **3081 task instances**。

每个 category 需要不同的 embodied ability 组合（见 paper Figure 3b）——比如 Physical Structure 主要需要 manipulation，Cognitive Mapping 主要需要 locomotion，Spatial Relations 主要需要 perception + locomotion。

---

## 3. 实验：四个 Paradigms

### 3.1 评估设置

paper 设计了 **4 个 paradigm**，按 agent 获得的 action / perception access 程度排序：

1. **Passive Single-View**: 单一 fixed observation from initial pose（baseline，对齐 prior spatial benchmark）
2. **Passive Multi-View**: 30 个沿 random trajectory 的 views（match $T_{\max} = 30$ 的 budget），模拟 exhaustive passive coverage
3. **Active Exploration**: agent 从 initial pose 出发，full access to action space
4. **Ground-Truth Passive**: 沿 ground-truth action trajectory 渲染的 views（oracle ablation，separate perception errors vs action errors）

**对比逻辑**：
- (1) vs (2): 更多 passive views 是否 help
- (2) vs (3): active action-guided vs passive exhaustive coverage
- (3) vs (4): failures 来自 action selection 还是 perception limit

模型两类：
- **2D VLM**: GPT-5, Gemini 3.1（egocentric visual observation 输入）
- **3D-augmented**: 
  - VGGT+Gemini: 用 VGGT 从 multi-view 重建 3D scene representation，构造 scene graph 给 LLM
  - Ground-Truth 3D+Gemini: 用 simulator state 直接 derive 的 perfect point clouds 构造 scene graph（oracle ablation）

### 3.2 关键发现 1: Active >> Passive Multi-View

paper Table 2 的数据非常 striking。我整理几个代表性数字：

| Subcategory | GPT-5 Passive Single | GPT-5 Passive Multi | GPT-5 Active | GPT-5 GT Passive |
|---|---|---|---|---|
| View Hallucination | 11.7 | 20.2 | 60.1 | 87.8 |
| Partial Occlusion | 32.6 | 26.3 | 47.4 | 86.3 |
| Rigid Containment | 45.0 | 42.5 | 42.5 | 95.0 |
| Physical Contact | 40.0 | 41.7 | 64.2 | 90.0 |
| Spatial Distance | 53.9 | **49.1** | 58.6 | 73.7 |
| Geometric Configuration | 25.3 | 20.4 | 26.0 | 26.0 |

几个 critical observation：

**(a) Passive multi-view 往往 hurt 而非 help**: GPT-5 在 Spatial Distance 上从 53.9% 跌到 49.1%。这非常反直觉——给了 30 个 views 反而比 1 个 view 还差。paper 的解释（Appendix L）：random view selection 容易 miss diagnostic region；image overload 让 MLLM 难以 integrate 30 个 views；no action grounding 让 spatial integration 困难；conflicting evidence from partial occluded views 引入 misleading cues。

**(b) Emergent spatial strategies**: 没有 explicit instruction，active agent 自发 discover 多种 strategy。比如判断 "chestnut 是否在 glass 里面"，agent 独立 develop 了 4 种方法：
- 移到 object 后面
- 从 top-down reposition
- pick up
- pour out

这 4 种都不是 prescribed 的。paper 在 Appendix I 量化了 strategy diversity：

| Subcategory | # Emergent Strategy Clusters |
|---|---|
| Material Transparency | 6 |
| Partial Occlusion | 4 |
| Physical Contact | 4 |
| Dimensional Size | 4 |
| Rigid Containment | 3 |
| Liquid Volume | 3 |
| Spatial Distance | 3 |

**(c) Active 在 few steps 内就 saturate**: Figure 9 的 step budget ablation 显示，Gemini 3.1 Active 从 5 步到 15 步快速提升，15-20 步 saturate，30 步后基本 flat，40 步后甚至略降。这表明 benefit 来自 **selective evidence acquisition**，而非 unbounded observation accumulation。

### 3.3 关键发现 2: Action Blindness >> Perceptual Blindness

这是 paper 最核心的 finding 之一。通过对比 Active vs GT Passive：

- **Rigid Containment**: GPT-5 Active 42.5% → GT Passive 95.0%（gap 52.5 points）
- **Physical Contact**: 64.2% → 90.0%（gap 25.8 points）
- **Counting w/ Occlusion**: gap 达 43.4 points
- **Structural Enclosure**: gap 达 45.0 points

**interpretation**: 给 perfect viewpoint，agent 基本都能做对——bottleneck 不在 perception，而在 action selection。Agent 不知道 **去哪里看、看什么、怎么 manipulate**。

但有两个 exception，暴露 **hard perceptual ceiling**：

- **Geometric Configuration**: GT Passive 也只有 26.0%，因为即使从 perfect viewpoint，模型也判断不出三个 object 是否构成 equilateral triangle（Figure 4c）
- **Specular Reflection**: 模型在 mirror 里 hallucinate 不存在的 object，或找不到正确 real-world correspondence

**Failure cascade**: suboptimal action → uninformative view → worse subsequent action → compounding error chain，在 step budget 内无法 recover（Figure 4b）。

### 3.4 关键发现 3: 3D Helps When Perfect, Hurts When Noisy

Table 2 里的 3D 对比很关键：

| Subcategory | Gemini 3.1 (2D) | VGGT+Gemini (noisy 3D) | GT 3D+Gemini (perfect 3D) |
|---|---|---|---|
| Geometric Configuration | 27.5 | **9.9** | 70.8 |
| Counting w/ Occlusion | 3.3 | **0.0** | 33.3 |
| Material Transparency | 44.0 | 27.8 | 60.4 |

**Perfect 3D**: Geometric Configuration 上 GT 3D+Gemini 比 Gemini 3.1 高 43.3 points；Counting w/ Occlusion 高 30 points。Explicit geometry 能 resolve 2D projection fundamentally ambiguous 的 depth/occlusion 问题。

**Noisy 3D (VGGT)**: 反而比 2D baseline 还差！Geometric Configuration 跌到 9.9%，Counting w/ Occlusion 直接 0.0%。

paper 在 Appendix J 分析了 VGGT failure modes：

| Failure Mode | Main Effect | Affected Tasks |
|---|---|---|
| Object duplication | 部分观测 object 被重建为多个 fragments | Counting w/ Occlusion, Merged Observation |
| Object hallucination | noisy geometry 产生 spurious object proposals | Enumerative Perception, Perceptual Grounding |
| Spatial-relation corruption | depth error 扭曲 relative position / contact / containment | Spatial Relations, Metric Comparison, Physical Structure |

**critical insight**: LLM 倾向于把 reconstructed scene graph 当作 reliable symbolic description，所以 noisy reconstruction 会 actively mislead reasoning。3D augmentation 是个 **high-variance strategy**，amplify both success 和 failure。这暗示 future 方向是 **uncertainty-aware scene-graph construction**，能区分 reliable 3D evidence vs ambiguous/fragmented geometry。

### 3.5 关键发现 4: Metacognitive Gap

这是 paper 最有思想深度的部分。

**Passive 设定下，human 和 model 性能相当**：有时 model 甚至超过 human passive。比如：
- Material Transparency GT Passive: GPT-5 96.3% vs Human 97.2%
- Partial Occlusion GT Passive: Gemini 88.4% vs Human 87.4%

**Active 设定下，human 大幅领先**：
- Physical Contact: Human 88.3% vs GPT-5 64.2%
- Material Transparency: Human 93.6% vs Gemini 52.3%

paper 在 Appendix K 把这个 gap operationalize 成 3 个 trajectory-level measures：

| Axis | Definition | Signal |
|---|---|---|
| Evidence sufficiency | agent 是否在 answer 前收集足够 diverse observation | View diversity |
| Falsification seeking | action 是否 seek 能 disconfirm 当前 hypothesis 的 evidence | Contrastive views |
| Belief revision | agent 是否在 contradictory evidence 后改变答案 | Answer updates |

Table 11 量化结果：

| Agent | View Diversity | Contrastive View Rate | Belief Revision Rate |
|---|---|---|---|
| Human Active | 71.8 | 62.7 | 41.3 |
| Gemini 3.1 Active | 43.5 | 31.5 | 18.9 |
| GPT-5 Active | 39.2 | 28.7 | 16.4 |

**Human**: gather more observation before commit，seek 能 falsify hypothesis 的 viewpoint，在 contradiction 下 revise belief。

**Models**: commit prematurely with high confidence（Figure 4d），directional bias——反复 move in same direction 累积 redundant 而非 informative observation（Figure 4e：判断 cupboard vs piano，model 先猜 cupboard，然后 backward+left 只为 confirm，从不 seek falsifying viewpoint，最后 assert 错误答案且 confidence 升高）。

**核心论点**: 这是 **epistemic calibration failure**，fundamentally distinct from perceptual ability，**cannot be resolved by stronger visual encoder 或更多 exploration steps**。它需要 metacognitive 能力：知道 evidence 何时 insufficient，知道如何 acquire better evidence。

---

## 4. 一些值得深挖的细节

### 4.1 Generator bias audit

因为用 GPT-4o 做 proposal engine，paper 仔细 audit 了 bias（Appendix H）：

- **Question-only baseline**: 只给 question，没有 visual / metadata / action history。如果 question wording 含 answer artifact，accuracy 会高于 majority baseline。结果 average 36.6%，低于 passive single 42.5%，远低于 active 56.9%。
- **Metadata-only baseline**: 给 question + task-relevant object categories（无 position/observation/action）。Average 39.3%。
- **Answer balance / Object diversity**: 归一化 entropy，average 0.92 / 0.85，high diversity。
- **GPT-4o vs human-generated tasks**: 在 matched subset 上 evaluate，accuracy gap 在 3.5 points 以内，说明 GPT-4o-generated task 难度与 human-generated 相当。

### 4.2 Step budget ablation 的数学含义

Figure 9 的 saturation curve 非常有意思。设 $B$ 为 budget，accuracy $A(B)$ 的形式大概是：

$$A(B) \approx A_{\infty} \cdot (1 - e^{-B/\tau})$$

其中 $\tau \approx 10$ 步，$A_{\infty}$ 在 30 步附近达到。超过 40 步 $A(B)$ 略降，暗示 long trajectory 引入 redundant / conflicting evidence，model 无法 filter 噪声。

**intuition**: 这与 active perception 文献中的 "information foraging" 理论一致——optimal exploration 应该在 marginal information gain 低于某个 threshold 时停止。模型缺乏这个 stopping criterion。

### 4.3 为什么 GT Passive 在某些 task 上也失败

Geometric Configuration（GT Passive 仅 26.0%）是个非常 instructive 的 case。即使从 perfect viewpoint（centroid at grazing height，能清楚看到三个 object 的相对位置），模型仍然判断不出是否 equilateral triangle。

这暗示 MLLM 缺乏 **metric geometric reasoning**：即 visual encoder 能提取 object position，但 LLM 推理层无法 do precise geometric computation（边长比较、角度判断）。3D ground truth 能 help（GT 3D+Gemini 70.8%），但 2D 推理层 fundamentally lost this information。

---

## 5. 对整个 field 的 implication

ESI-BENCH 提出的问题比答案更有价值。它揭示了一个 **三层的 spatial intelligence hierarchy**：

1. **Perceptual layer**: visual encoder 能否 extract spatial cue from single observation
2. **Active layer**: agent 能否 decide which observation to acquire
3. **Metacognitive layer**: agent 能否 know when evidence is sufficient, seek falsification, revise belief

当前 MLLM 主要卡在 layer 2 和 layer 3。Paper 的核心 thesis 是：**better perception alone 不够，more embodied interaction alone 也不够**——需要的是 epistemic calibration，知道何时 stop、何时 seek contrastive evidence、何时 revise。

这与你之前在 Eureka Labs 和其他场合讲的 "model 需要 System 2 thinking、需要 know what it doesn't know" 的方向高度一致。

参考链接：
- 项目主页: https://esi-bench.github.io/
- 论文: https://arxiv.org/abs/2507.03934（推测，基于 arxiv 编号风格）
- BEHAVIOR-1K: https://behavior.stanford.edu/
- OmniGibson: https://github.com/StanfordVL/OmniGibson
- Spelke core knowledge: https://onlinelibrary.wiley.com/doi/10.1111/j.1467-7687.2007.00569.x
- VSR benchmark: https://arxiv.org/abs/2205.00363
- VSI-Bench: https://arxiv.org/abs/2412.14171
- MindCube: https://arxiv.org/abs/2506.21458
- PhysBench: https://arxiv.org/abs/2501.16411
- OpenEQA: https://arxiv.org/abs/2407.12521
- EmbodiedBench: https://arxiv.org/abs/2502.09560

---

## 6. 一些可能让你兴奋的方向

基于这篇工作的 limitation 和 finding，我看到几个 interesting 的 future direction：

1. **Uncertainty-aware active perception**: 在 VGGT+Gemini 失败的 case 里，如果能给 LLM 传递 reconstruction uncertainty（per-point confidence、fragmentation score），LLM 或许能 avoid treating noisy 3D as ground truth。这连接到 Bayesian active learning literature。

2. **Belief revision training**: paper 显示 model 的 belief revision rate 仅 16-19%，human 41%。能否通过 RLHF 或 self-play 训练 model 在 contradiction 下 update belief？这接近 "epistemic humility" training。

3. **Falsification-seeking reward shaping**: 在 active exploration 训练中，除了 task success reward，加一个 "contrastive viewpoint bonus"——鼓励 agent 探索能 disconfirm 当前 hypothesis 的 view。

4. **Stopping criterion learning**: model commit prematurely，可否训练一个 value function $V(s_t, q)$ 估计 "当前 evidence 是否足够 answer $q$"？这本质是 learn-to-stop 问题。

5. **Real-world transfer**: paper 承认 sim-to-real gap 是 limitation。能否用 similar task structure 在 real robot 上 collect small dataset 来 validate sim finding？

如果你想深入讨论某一块（比如 emergent strategy 的具体形式、3D augmentation 的 failure mode、metacognitive gap 的 intervention 设计），告诉我，我可以展开讲。
