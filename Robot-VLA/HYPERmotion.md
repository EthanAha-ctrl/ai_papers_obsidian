---
source_pdf: HYPERmotion.pdf
paper_sha256: dfb185143d10cf27774522677c1b50ac3fa92711b5845b2fde13bfa04f924019
processed_at: '2026-08-05T08:55:10-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 HYPERmotion

## 1. 这篇 paper 在干嘛

想象你有一台人形机器人，底下装着 4 条腿，腿底下还有轮子，上半身是 humanoid，有两个 arm 和一个 gripper。然后你跟它说一句：

> "Open the drawer under the desk, find the drill and put it into the tool box, then bring me the box."

机器人要自己听懂这句话，自己决定先干嘛后干嘛，自己选是滚过去还是走过去，自己选是一只手抓还是两只手抱，最后把整串动作执行完。

这就是 paper 要解决的问题。叫 **long-horizon loco-manipulation**，意思是要同时移动身体和操作物体，而且任务由很多步组成。

为什么难？因为：
- 机器人有 **38 个 joint**，直接让神经网络输出 38 维 joint command 很难学
- 任务要先开抽屉再抓东西再放再拿，中间还要走来走去，horizon 太长
- 有时候要一只手（drawer handle），有时候要两只手（大 toolbox）
- 有时候平地滚过去就行，有时候要迈腿（地上有障碍物）
- LLM 不会输出 joint trajectory，它只会输出文字或 code

## 2. 他们怎么做的

他们的核心想法其实非常朴素：**把任务拆成小块，每小块单独学，学完装进一个"技能库"，然后让 LLM 来决定调哪个技能**。

整个 pipeline 分四块：

### 第一块：离线训练技能（Motion Generation）

这台机器人是 38 DoF 的，直接上 end-to-end RL 调不出来。他们用了两个 trick：

**Trick 1：只训上半身**

他们发现要开抽屉、开门、抓东西这种动作，主要靠 arm + torso，腿其实只是负责维持平衡。所以 RL 只训 **上半身的 14-19 维**（6 维 right arm + 6 维 floating base + 1 维 torso yaw + 1 维 gripper，dual-arm 再加 6 维 left arm），腿的 trajectory 暂时不管。

**Trick 2：训完之后用 optimizer 补腿**

RL 在 Isaac Gym 里训，训练的时候故意把 self-collision 关掉（让 RL 探索更自由），所以 RL 输出的 trajectory 可能让 arm 穿过腿。这个 trajectory 拿出来之后，喂给一个叫 **HORIZON** 的 trajectory optimizer（基于 CasADi 的 NLP solver），optimizer 在强制满足 dynamics + joint limit + self-collision + friction cone 这些约束的前提下，找一个最接近 RL trajectory 的 whole-body 解。说白了 optimizer 就是个 **projection operator**，把 RL 在松弛空间里的解投影到真实机器人动力学流形上。

为什么这么做？因为：
- RL 在 reduced space 训，sample efficiency 高
- Optimizer 保证物理可行，sim2real 友好
- 训完存进 library，可以无限复用

他们训了 4 个 skill：drawer opening、door opening、single-arm pick、dual-arm pick。每个 skill 训完都过一遍 optimizer，最后存进 Motion Library。

### 第二块：让 LLM 做高层 planning

用户输入一句英文，LLM 不直接输出 action，而是输出一个 **Behavior Tree（BT）的 XML 文件**。Behavior Tree 你可以理解成一个树状的状态机，每个 node 要么是 action（执行某个技能），要么是 condition（检查某个状态）。LLM 拿到任务描述 + 技能库的目录 + 几个可选 function，输出一棵 BT。

举个例子，对 "Open the drawer and pick the object"，BT 大概长这样：

```
Sequence
├── FindObject
├── MoveTarget (用 AprilTag 导航)
├── ObjectDetect (用 FoundationPose 估 6D pose)
├── WhetherSingleArm (VLM 决定单手还是双手)
│   ├── SinglePick
│   └── DualPick
├── OpenDrawer
└── PickFromDrawer
```

LLM 的作用是 reasoning：分解任务、选择技能、决定调用哪些 function。它不做 control。这就是把 LLM 用在它擅长的语义层，把 control 交给预先训好的 RL policy + optimizer。

### 第三块：VLM 做 morphology selector

机器人有两种 manipulation mode（单手 / 双手）和两种 locomotion mode（轮子 / 腿）。哪个 mode 合适，不是固定的，要看场景：

- 抽屉把手 → 单手够得到 → single_arm
- 大 toolbox → 单手抓不动 → dual_arm
- 平地 → 滚过去 → wheel
- 地上有台阶 → 迈过去 → leg

这个 decision 交给 **GPT-4v**。VLM 收到的输入有：
- 头部摄像头的 RGB 图（看物体）
- 骨盆摄像头的 RGB+depth（看地形）
- 物体的 6D pose（FoundationPose 给的）
- 地形的 voxel grid（点云下采样）
- Robot 的 affordance 描述（什么动作能做、什么够得到）

VLM 输出 mode 选择。这个设计很关键，因为 **同一个 task 在不同 scene 里最优 mode 不同**，硬编码会失败。

### 第四块：BT 执行

LLM 输出的 BT XML 加载进来之后，被一个 BT engine 实例化。BT 每个 tick 都重新 check 所有 condition node，如果 condition 满足就执行对应 action node，action 就是调 Motion Library 里的 RL policy + optimizer。

整个过程：用户说一句话 → LLM 生成 BT → BT 触发技能 → 技能调 RL → RL 给 reference → optimizer 给 feasible trajectory → 机器人执行。

## 3. 为什么这样做 work

我的直觉是这篇 paper 的核心 insight 有三个：

### Insight A：LLM 做 composition，不做 control

LLM 输出 joint trajectory 是灾难（38 维连续控制，LLM 没这个能力）。但 LLM 输出 BT XML 这种结构化符号是非常擅长的。BT 节点又对应预训好的 skill，相当于 LLM 只需要在离散符号空间里做组合，把 dynamics 留给底层。

这跟 SayCan / PaLM-E 是一个思路，但 SayCan 只 grounding affordance，HYPERmotion 多了 whole-body feasibility projection 这一层。

### Insight B：Reduced space RL + projection

直接训 38 DoF humanoid 在 Isaac Gym 里理论上可行，但 sim2real 难。他们的 reduced space RL + QP projection 这套：
- RL 在低维空间快速学到 task-relevant 的 motion
- QP 保证 whole-body physical feasibility
- 两个解耦，可以独立迭代

这个 pattern 在 quadruped manipulator 圈已经有人用 [Ma et al., RA-L 2022]，但 HYPERmotion 是第一个把它推到 38 DoF humanoid + wheel-leg hybrid 上的。

### Insight C：VLM 做 discrete decision，不做 continuous control

VLM 输出 "single_arm" 还是 "dual_arm" 这种 discrete label 是非常擅长的，把它当成 binary classifier 用。但 VLM 直接输出 joint trajectory 同样灾难。这种 "把 VLM 用在 discrete decision 层" 的思路跟 RT-2 那种 end-to-end VLA 形成鲜明对比。

代价是 HYPERmotion 要预先训好一堆 skill，没有 RT-2 那种 zero-shot emergence 的魔力。但在 38 DoF humanoid 这个 scale，end-to-end VLA 的数据需求是天文数字。

## 4. 实验告诉我们什么

### 4.1 RL policy 的成功率（Table A11）

| 任务 | 随机化范围 | 成功率 |
|---|---|---|
| Drawer opening | 0.01-0.10m | 77.8% ± 11% |
| Door opening | 0.02-1.0m | 85.3% ± 19.6% |
| Single-arm pick | 0.2-0.5m | 88.5% ± 6.9% |
| Dual-arm pick | 0.02-1.0m | 73.8% ± 15.7% |

Door 和 dual-arm pick 在 1m 随机化下还能 70%+，policy 泛化性不错。Drawer opening variance 大是因为 handle 位置一点点偏都会导致 grasp 失败。

### 4.2 LLM-Planner vs 传统 BT-Planner（Table A1）

| 指标 | BT-Planner | LLM-BT |
|---|---|---|
| Long-horizon task exec rate | 74% | 86% |
| Long-horizon task success | 68% | 82% |
| Long-horizon avg time | 145s | 121s |

LLM 生成 BT 比人工写 BT 规则更好，因为 LLM 能理解 free-text 指令并推理。

### 4.3 LLM-Planner vs WB-MPC（Table A2）

| 任务 | WB-MPC | LLM-Planner | LLM+MS |
|---|---|---|---|
| Move to target | 84% | 96% | 92% |
| Open door | 64% | 84% | 88% |
| Pick and place | 60% | 72% | 84% |
| **Open drawer + pick** | **0% (TLE)** | 64% | 72% |

最后这一行最关键：传统 whole-body MPC 在 long-horizon 任务上直接 0%，因为 MPC horizon 太短，online recompute 太慢。LLM-Planner 因为把任务拆成 primitive 顺序调用，每段 horizon 短，所以能完成。这就是 modular decomposition 的威力。

### 4.4 Morphology Selector 的提升（Figure 6）

加入 spatial data（depth + 6D pose + voxel grid）后：
- Manipulation 选择准确率从 ~85% → ~92%
- Locomotion 选择准确率从 ~60% → ~90%

Locomotion 提升特别明显，因为单纯 RGB 看不出台阶高度和障碍距离，加了 depth + voxel 才看得出。这跟 SpatialVLM [Chen et al., 2024] 的发现一致：VLM 单靠 2D image 做 spatial reasoning 是不够的，需要 explicit 3D signal。

### 4.5 Error 来源（Figure 8）

实验失败的三大类：
- **Planning error**：LLM 在超复杂任务上 BT 构造错（漏 action 或逻辑错误）
- **Perception error**：AprilTag 在机器人移动时有 2-4cm 误差；FoundationPose 在 occlusion 下不准
- **Execution error**：占大头。预训 policy 遇到 unexpected disturbance（腿撞门框）无法 adapt；long-horizon 上 error 累积（pick 时偏一点，place 时撞到环境掉落）

Execution error 占大头这点很重要：这告诉我们**预训 primitive 在 open-loop 执行下还是脆的**。如果 task 里某一步卡住，没有 closed-loop recovery 机制。Paper 也承认这是 limitation。

## 5. 跟其他路线对比

| 路线 | 代表 | 优点 | 缺点 |
|---|---|---|---|
| End-to-end VLA | RT-2, OpenVLA | zero-shot emergence，data-driven | 在 38 DoF humanoid 上数据量爆炸，sim2real 难 |
| LLM as planner + library | SayCan, HYPERmotion | 模块化，可调试，可解释 | 新 skill 要训，没有 emergence |
| LLM as code generator | Code as Policies | 灵活，可表达任意逻辑 | 在 floating-base humanoid 上 dynamics 不可控 |
| Whole-body MPC | Dadiotis et al. | 物理严谨，closed-loop | long-horizon 失败（horizon 太短） |

HYPERmotion 选了第二条路线，但加了 RL + QP projection 来处理 humanoid 的 dynamics 复杂性，加了 VLM selector 来处理 mode 选择。这是 SayCan 思路在 humanoid scale 上的工程化。

## 6. 我的几点直觉

### 6.1 这套 modular 思路是工业界主流

Tesla Optimus 大概率也是类似分层架构：高层 planner + 中层 skill library + 低层 controller。原因：
- 工业界要可调试、可单元测试、可 blame
- 端到端 VLA 出问题你不知道是 perception 错还是 control 错
- Library 化的 skill 可以独立迭代、独立发布

HYPERmotion 在 academic scale 上完整跑通了这条路线，给了一个 proof-of-concept。

### 6.2 Library size 是最大瓶颈

Paper 自己承认：motion library 大小限制了 task 范围。每加一个 skill 都要：
1. 设计 reward
2. 在 Isaac Gym 训 PPO
3. 调 HORIZON optimizer
4. 写 BT node wrapper
5. 加进 prompt

这个流程太重。未来方向肯定是 **skill 的自动生成**：让 LLM 自动 generate task + reward + 训练脚本，类似 RoboGen [Wang et al., 2023, https://robogen-website.github.io/] 的思路。或者用 task embedding 把所有 skill embed 到 latent space，做 interpolation（Unicorn 的思路）。

### 6.3 Closed-loop replanning 缺失

当前 BT 一旦生成就是 fixed plan，执行中遇到错误只能 retry 同一个 action。没有 "重规划" 能力。这跟 RT-2 那种 closed-loop VLA 是有差距的。

要补这个洞，几个方向：
- 让 VLM 持续监控 task progress，发现偏差就 trigger re-planning
- 给 BT 加 recovery sub-tree，每个 action 失败时有 fallback
- 引入 Reflexion [Shinn et al.] 让 LLM 从失败中学习

### 6.4 Optimizer 是 hidden bottleneck

HORIZON 这种 CasADi NLP solver 在 tens of ms 这个 scale，对 long-horizon 任务的 closed-loop replanning 太慢。如果换成 **differentiable MPC + neural warm-start**，频率提到 100Hz，对 disturbance 鲁棒性会好很多。

### 6.5 VLM 的 confidence 没用上

VLM 输出 single_arm / dual_arm 是个 hard label，没有 confidence。10% 的 morphology 选错率在 long-horizon 上会指数累积。理想情况下应该输出 confidence，低 confidence 时 fallback 到 default mode 或者让 LLM 再 reasoning 一次。

### 6.6 这是 "AI for Robotics" 还是 "Robotics for AI"？

我觉得 HYPERmotion 是 "Robotics for AI" 的路子：用经典 control + RL 解决 dynamics，把 AI 当 planner。这跟 "AI for Robotics"（end-to-end VLA）是两个极端。当前在 38 DoF humanoid 这个 scale，前者更 work，因为 dynamics 太复杂，data 太少。

但随着 data scale 上去（比如 Tesla 的 fleet data），end-to-end VLA 在 humanoid 上的可行性会变高。HYPERmotion 这种 modular 框架可以作为 **end-to-end 模型的 baseline 比较对象**，也可以作为 **数据收集的 teacher**（用 modular pipeline 生成 demonstration，再 distill 成 end-to-end policy）。

## 7. 总结成一句话

**让 LLM 当"项目经理"分派任务，让 RL 当"工人"执行任务，让 VLM 当"调度员"选合适的工作模式，三者通过 skill library + Behavior Tree 接口协作，让 38 DoF 人形机器人从一句人话完成一长串 loco-manipulation。**

这是 SayCan 思路在 humanoid + wheel-leg hybrid 平台上的完整工程实现。技术上没有特别新的 algorithm，但 system integration 做得相当扎实，sim 和 real 都跑通了 long-horizon 任务，这个在 2024-2025 的人形机器人圈里是前列的。

---

参考：
- Project: https://hy-motion.github.io/
- SayCan: https://say-can.github.io/
- RT-2: https://robot-transformer2.github.io/
- SpatialVLM: https://spatial-vlm.github.io/
- VoxPoser: https://voxposer.github.io/
- HORIZON: https://www.frontiersin.org/articles/10.3389/frobt.2022.899025/full
- FoundationPose: https://nvlabs.github.io/FoundationPose/
- HumanoidBench: https://humanoidbench.github.io/
- RoboGen: https://robogen-website.github.io/
- AnyMal Parkour: https://anymal-parkour.github.io/
- Behavior Trees in Robotics: https://www.behavior-robotics.org/
- xbot2: https://github.com/ADVRHumanoids/xbot2
- CartesI/O: https://github.com/ADVRHumanoids/cartesian_interface

---

# HYPERmotion: 详解一篇人形机器人 Hybrid Behavior Planning 的工作

## 1. 总览与定位

这篇 paper 来自 Italian Institute of Technology (IIT) 的 Nikos Tsagarakis 团队，第一作者 Jin Wang。核心目标：让一台 38-DoF 的 wheel-legged humanoid（centauro-like 平台），从 free-text instruction 出发，**zero-shot 完成 long-horizon loco-manipulation 任务**。例如开篇的 example：

> "Open the drawer under the desk, find the drill and put it into the tool box, then bring me the box."

这串 instruction 拆出来至少包含 navigation + perception + drawer opening + object detection + single-arm pick + place + dual-arm pick + locomotion 等十多个 primitive。传统的 whole-body MPC [Dadiotis et al., Humanoids 2023] 在这类任务上基本做不动（Table A2 里 "Open drawer and pick object" 用 WB-MPC 是 **0%** success，TLE）。HYPERmotion 的思路：**把学习到的 motion primitives 装进 library，用 LLM 做高层 task planning 输出 Behavior Tree，用 VLM 做 morphology selector 决定 single/dual arm 和 wheel/leg mode**。

项目主页：https://hy-motion.github.io/

参考相关工作（这些 Karpathy 你应该都熟，我快速串一下）：
- **SayCan / PaLM-E / RT-2 / Code as Policies** —— LLM/VLM 作为 robot planner 的 line of work: https://say-can.github.io/ , https://palm-e.github.io/ , https://robot-transformer2.github.io/
- **RT-1** —— Google 的 Robotics Transformer: https://robotics-transformer.github.io/
- **VoxPoser / SpatialVLM** —— LLM + spatial reasoning for manipulation: https://voxposer.github.io/ , https://spatial-vlm.github.io/
- **HumanoidBench / Humanoid-Gym** —— 人形 RL benchmark + sim2real: https://humanoidbench.github.io/ , https://github.com/RoboTwin/Humanoid-Gym
- **AnyMal Parkour / Robot Parkour Learning** —— 极端 locomotion RL: https://anymal-parkour.github.io/ , https://robotparkour.github.io/
- **CENTAURO / xbot2 / CartesI/O** —— 本文底层用到的硬件和 middleware: https://github.com/ADVRHumanoids/xbot2 , https://github.com/ADVRHumanoids/cartesian_interface
- **HORIZON trajectory optimizer** —— whole-body trajectory optimization framework [Ruscelli et al., Frontiers in Robotics and AI 2022]: https://www.frontiersin.org/articles/10.3389/frobt.2022.899025/full

---

## 2. 核心架构解析（对应 Figure 2）

整条 pipeline 拆成 4 个 sector，training 和 deployment 阶段解耦得很清楚：

### Sector A: Motion Generation（离线）
- 在 **Isaac Gym** 里用 PPO 训练 reduced-dim upper body policy（每条 skill 单独训）
- 把 RL 输出的 20-dim upper-body reference trajectory $\mathbf{q}^*$ 喂给 **whole-body trajectory optimizer (HORIZON)**
- Optimizer 解一个 93-dim state / 58-dim input 的 NLP，得到 full-body feasible trajectory
- 存进 **Motion Library**（XML + Python 控制代码 + 文本描述）

### Sector B: User Input
- Basic Prompts（robot 形态描述、output 格式）
- Motion Library 文本目录
- Function Options（开关：manipulation_mode_selector / locomotion_mode_selector / detection_recovery）

### Sector C: Task Planning（在线，GPT-4o）
- 接收 free-text instruction → 输出 hierarchical task graph
- Task graph 保存为 XML → 实例化为 Behavior Tree (BT) [Colledanchise & Ögren, 2018, https://behavior3.com/]
- BT 节点 = Motion Library 里的 action + condition nodes

### Sector D: Morphology Selector（按需，GPT-4o / GPT-4v）
- 当 BT 走到某个节点需要决定 arm mode / locomotion mode 时触发
- 输入：head/pelvis camera 的 RGB + depth，6D object pose（FoundationPose [Wen et al.]），voxel grid
- 输出：`single_arm` / `dual_arm` / `wheel` / `leg`

**intuition：为什么不直接让 LLM 端到端输出 action sequence？**
因为 38 DoF floating-base humanoid 直接让 LLM 输出 joint trajectory 在工程上灾难性（latency、动力学不可行、self-collision）。把 LLM 限定在 "组合已有 primitives + 决定 mode" 这个层次，是把 LLM 的语义推理能力 用在它擅长的位置 —— symbolic planning，把 dynamics feasibility 交给 RL + QP optimizer。这跟 SayCan 的 motivation 类似，但 SayCan 只 grounding affordance，HYPERmotion 多了一层 **whole-body feasibility projection**。

---

## 3. Whole-body Motion Learning（Section 3.2，关键公式）

### 3.1 Action / Observation Space 的维度切分

这是一个非常关键的工程决策。他们把 robot 拆成 upper body（arm + torso + floating base）和 lower body（legs + wheels）：

- **Single-arm task** action space: $\mathcal{A}_1 \subseteq \mathbb{R}^{14}$
  - 6-DoF right arm
  - 6-DoF floating base（3 translation + 3 Euler）
  - 1 torso yaw
  - 1 gripper
  - left arm 固定

- **Dual-arm task** action space: $\mathcal{A}_2 \subseteq \mathbb{R}^{19}$
  - 在 $\mathcal{A}_1$ 基础上加 6-DoF left arm
  - gripper 默认 closed

为什么这样切？因为 38 DoF 直接训 PPO，sample efficiency 灾难，且 leg 和 arm 的 reward scale 完全不同时序不同。Floating base 写进 action 是为了给 optimizer 一个 "期望 base pose" 的 hint，**heuristic 限制 floating base** 是为了避免 RL 输出腿走不过去的 base trajectory。这种 "reduced space RL + whole-body projection" 的 pattern 在 legged manipulator 圈里 [Ma et al., RA-L 2022] 用过，但 HYPERmotion 是第一个把它扩展到 humanoid + wheel-leg hybrid 的。

### 3.2 Reward 函数（公式 1）

$$
r = \alpha_1 r_{l\_reach} + \alpha_2 r_{r\_reach} + \alpha_3 r_{rot} + \alpha_4 r_{finger} + \alpha_5 r_{task} + \alpha_6 r_{penalty}
$$

逐项解释：

**Reach reward（左/右 end-effector）**：
$$
r_{l\_reach} = \left( \frac{1}{1 + d_l^2} \right)^2, \quad r_{r\_reach} = \left( \frac{1}{1 + d_r^2} \right)^2
$$
- $d_l, d_r$：target object 到 left / right end-effector 的 Euclidean distance
- 这种 $\left( \frac{1}{1+d^2} \right)^2$ 形式叫 "potential-based shaping"，离目标越近 gradient 越大，避免 sparse reward 的问题。平方是让 reward 在近距离时更陡，鼓励 fine-grained approach。

**Orientation alignment**：
$$
r_{rot} = \text{sign}(d_x) \cdot d_x^2 + \text{sign}(d_z) \cdot d_z^2
$$
- $d_x$：gripper forward axis 和 object inward axis 的 dot product（>0 表示朝向正确）
- $d_z$：gripper up axis 和 object up axis 的 dot product
- $\text{sign}(d_x)$ 让 reward 在 dot 为正时正、为负时负，惩罚反方向。平方是为了在已经对齐情况下鼓励更紧的对齐。这是个非常巧妙的设计，避免了单纯用 cosine 的 flat gradient 问题。

**Finger grasping**：
$$
r_{finger} = \beta - (d_t + d_b)
$$
- $d_t, d_b$：gripper top link / bottom link 到 object 的距离
- $\beta$：和 object size 相关的 fine-tuning 参数，paper 里 drawer 是 0.04，door 是 0.02
- 直觉：当 $d_t + d_b < \beta$ 时 gripper 处于 "环绕" object 的状态

**Action penalty**：
$$
r_{penalty} = -\|\mathbf{a}\|^2
$$
- 标准 L2 regularization on action，鼓励 smoothness

**Task-specific reward $r_{task}$**：
每个 task 单独定义，公式 A1-A4 在 Appendix：

- Drawer opening: $r_{task} = \alpha_7 r_{around} + l_{drawer} \cdot r_{around} + l_{drawer}$
  - $r_{around} \in \{0, 0.5\}$：gripper 是否环绕 handle（top link above handle AND bottom link below handle）
  - $l_{drawer}$：drawer 被拉出的长度

- Door opening: $r_{task} = \alpha_7 r_{around} + \text{angle}_{handle} \cdot r_{around} + \text{angle}_{handle} + \text{angle}_{door}$
  - $\text{angle}_{handle}$：门把手被按下的角度
  - $\text{angle}_{door}$：门打开的角度

- Single-arm pick: $r_{task} = \alpha_7 r_{around} + h$，其中 $h \in \{0, 1\}$ 是 object 是否被拿起

- Dual-arm pick: $r_{task} = h$（直接 sparse binary）

参数 $\alpha_1 \dots \alpha_7$ 在 Table A4-A10 里。值得注意 drawer opening 的 $\alpha_5 = 7.5$ 而 $\alpha_6 = 0.01$，说明 task completion 和 penalty 的尺度差 750x，这是 RL 在 robotics 上很常见的 reward tuning 痛点。

### 3.3 Whole-body Optimization（公式 2，A5-A18）

RL 输出的 $\mathbf{q}^* \in \mathbb{R}^{20}$ 是 upper body joint trajectory，但要 deploy 到 38 DoF robot，需要把 leg 和 wheel 也填上。这个 "filling" 是一个 optimal control problem：

$$
\min_{\mathbf{x}(\cdot), \mathbf{u}(\cdot)} \int_0^T L(\mathbf{x}(t), \mathbf{u}(t), t)\, dt
$$

subject to:

- **Dynamics**：$\dot{\mathbf{x}}(t) = f(\mathbf{x}(t), \mathbf{u}(t), t)$
- **Equality constraints** $g_1$：initial state、contact point、...
- **Inequality constraints** $g_2$：joint limits、velocity limits、friction cone、torque limits

变量维度：
- $\mathbf{x}(t) = [\mathbf{q}, \mathbf{v}] \in \mathbb{R}^{93}$ — state（93 维：38 joint position + 38 joint velocity + 6 floating base + ... 实际是 nv 维 generalized velocity 相关）
- $\mathbf{u}(t) = [\dot{\mathbf{v}}, \mathbf{f}_c] \in \mathbb{R}^{58}$ — input（acceleration + contact force）

**Dynamics equation（A7）**：
$$
\dot{\mathbf{x}}_i = \begin{bmatrix} \dot{\mathbf{v}}_i \\ M(\mathbf{q}_i)^{-1}\big( J_c^T(\mathbf{q}_i) \mathbf{f}_c^i - h(\mathbf{q}_i, \mathbf{v}_i) + S\boldsymbol{\tau}_i \big) \end{bmatrix}
$$

- $M \in \mathbb{R}^{n_v \times n_v}$：mass matrix
- $h$：gravity + Coriolis + centrifugal bias
- $J_c$：contact Jacobian
- $\mathbf{f}_c$：contact force（4 legs × 3D = 12 dim，或者根据 contact mode 不同）
- $S \in \mathbb{R}^{n_v \times n_a}$：actuated torque mapping matrix（把 actuated joint torque 嵌入到 full generalized force 向量里）
- $\boldsymbol{\tau}_i$：actuated joint torque

**Cost function（A18）**：
$$
L_i(\mathbf{x}_i, \mathbf{u}_i) = \|\mathbf{q}_i^u - \mathbf{q}_i^*\|^2 + \|\mathbf{u}\|^2
$$
- $\mathbf{q}_i^u$：当前 step 的 upper body joint
- $\mathbf{q}_i^*$：RL 给的 reference
- 第一项：tracking RL reference
- 第二项：energy minimization

**Constraints（A8-A17）的物理意义**：

| 约束 | 含义 |
|---|---|
| $\mathbf{q}^0 = \mathbf{q}_{init}$, $\mathbf{v}^0 = 0$ | 初始静止状态 |
| $\mathbf{q}_{min}^k \leq \mathbf{q}^k \leq \mathbf{q}_{max}^k$ | joint position limit |
| $\mathbf{v}_{min}^k \leq \mathbf{v}^k \leq \mathbf{v}_{max}^k$ | velocity limit |
| $\dot{\mathbf{v}}_{min} \leq \dot{\mathbf{v}} \leq \dot{\mathbf{v}}_{max}$ | acceleration limit |
| $\mathbf{f}_{c,j}^{z,k} \cdot \mathbf{n}_i > 0$ | unilateral contact（foot 不能 "吸" 在地上） |
| $\|(\mathbf{f}_{c,j}^{x,k}, \mathbf{f}_{c,j}^{y,k})\|_2 \leq \mu_i (\mathbf{f}_{c,j}^{z,k} \cdot \mathbf{n}_i)$ | 摩擦锥（leg mode，防滑） |
| $\tau_{fb}^k = 0$, $\tau_{j,min} \leq \tau_j \leq \tau_{j,max}$ | floating base 不施 torque；actuated joint torque limit |

**关键 intuition**：RL 在 Isaac Gym 里不考虑 self-collision（training 时关掉），所以 RL 可以输出"穿过腿"的 trajectory。然后 whole-body optimizer 把这个 trajectory 当 reference，在 enforce self-collision 和 joint limit 的前提下找最近的 feasible trajectory。Optimizer 本质是个 **projection operator**，把 RL 在松弛空间里的解投影到 rigid body dynamics 流形上。这就是为什么 Figure 5 里 real-world trajectory 比 Isaac Gym 的更 smooth —— optimizer 起到了 nonlinear filter 的作用。

---

## 4. Morphology Selector（Section 3.3，公式 3-4）

这是论文我觉得最有意思的设计。Robot 有两种 manipulation mode（single-arm / dual-arm）和两种 locomotion mode（wheel / leg），但**选哪个不是固定的**，而是 VLM 根据当前 scene 决定。

**Manipulation morphology**：
$$
\mathbf{x}_m = \mathcal{V}(\mathbf{s}, \mathbf{I}_{scene}^h, \mathbf{v}_R, \mathbf{p}_V)
$$
- $\mathbf{s}$：task state
- $\mathbf{I}_{scene}^h$：head camera RGB
- $\mathbf{v}_R \in \mathbb{R}^6$：object 在 robot frame 下的 6D pose（FoundationPose 给的）
- $\mathbf{p}_V$：robot 结构和 affordance 的 prompt

**Locomotion morphology**：
$$
\mathbf{x}_l = \mathcal{V}(\mathbf{s}, \mathbf{I}_{scene}^p, \mathbf{V}_g, \mathbf{p}_V)
$$
- $\mathbf{I}_{scene}^p$：pelvis camera RGB
- $\mathbf{V}_g$：pelvis depth → point cloud $\mathbf{P}_c$ → down-sample → voxel grid

为什么 manipulation 用 6D pose 而 locomotion 用 voxel grid？因为 manipulation 决策只关心**一个 object** 的相对位置（决定单臂够得到还是要双臂合抱），而 locomotion 决策关心**整条 path 上的 terrain**（决定能不能 roll 过去还是要 step over）。两种输入对应两种 spatial scale。

Figure 6 的实验结果：spatial data 加入后 manipulation 提升约 5-10%，locomotion 提升更明显（特别是复杂 obstacle 场景，从 ~60% 到 ~90%）。这验证了 2D image 单独不足以 ground 3D affordance 的假设，跟 SpatialVLM [Chen et al., 2024] 的发现一致。

---

## 5. Behavior Tree 作为 LLM 和 Robot 之间的桥梁

LLM 输出的不是 Python 代码（Code as Policies 风格），也不是直接 action sequence，而是 **XML 格式的 BT**。这个选择很聪明：

- BT 是 hierarchical reactive structure，天然支持 re-plan 和 condition check
- XML 是 LLM 训练数据里超大量的格式，输出可靠性高
- 节点 = Motion Library 里的 primitive（pre-trained），所以 LLM 不需要 generate 低层 code

User Interface 的设计：
```
Basic Prompts (robot 描述) + Function Options (开关) + Motion Library (目录) + User Command
```

Function Options 三选一：
1. `manipulation_mode_selector`：插入 `<WhetherSingleArm>` condition node
2. `locomotion_mode_selector`：插入 `<WhetherWheelMove>` condition node
3. `detection_recovery`：插入 `<IsActionSuccess>`，失败就 retry

这种 "function as switchable prompt token" 设计让 instructor 可以控制 planner 的复杂度，避免 over-engineering 简单任务。

---

## 6. 实验数据深度解读

### 6.1 Sim2Real trajectory tracking（Figure 5）

Drawer opening：end-effector 在 x 轴先 increase（approach）后 decrease（pull out），符合物理。Door opening：approach + push down。Trajectory 在 Gazebo 和 real-world 上和 Isaac Gym 的 training trajectory 高度一致，说明 whole-body optimizer 起到了 "filter + projector" 的作用，没有引入显著 distortion。

### 6.2 Skill learning success rate（Table A11）

| Task | Position randomization (m) | Success rate |
|---|---|---|
| Drawer opening | [0.01, 0.10] | 77.8% ± 11.0% |
| Door opening | [0.02, 1.0] | 85.3% ± 19.6% |
| Single-arm pick | [0.2, 0.5] | 88.5% ± 6.9% |
| Dual-arm pick | [0.02, 1.0] | 73.8% ± 15.7% |

注意 door 和 dual-arm pick 的 randomization scope 到 1m，success rate 仍然 >70%，说明 RL policy 有不错的 generalization。Drawer opening 的 variance 大（±11%）因为 handle 位置微小变化都会影响 grip 成功率。

### 6.3 LLM-Planner vs BT-Planner vs WB-MPC（Table A1, A2）

**BT-Planner vs LLM-BT**（Table A1）：

| Method | Exec | Succ | Avg. Time (Loco) | Exec | Succ | Avg. Time (Long) |
|---|---|---|---|---|---|---|
| BT-Planner | 92% | 80% | 39.07s ± 12.4 | 74% | 68% | 145.37s ± 32.6 |
| LLM-BT | 98% | 94% | 32.79s ± 8.6 | 86% | 82% | 121.20s ± 20.4 |

LLM-BT 在 long-horizon 上时间减少 16%，success rate 提升 14%。这是 prompt engineering 给的 benefit。

**WB-MPC vs LLM-Planner vs LLM-Planner+MS**（Table A2）：

| Task | WB-MPC Succ | LLM-Planner Succ | LLM-Planner+MS Succ |
|---|---|---|---|
| Move to target | 84% | 96% | 92% |
| Approach Object | 72% | 88% | **96%** |
| Open door | 64% | 84% | **88%** |
| Pick object | 68% | 80% | 84% |
| Pick and place | 60% | 72% | **84%** |
| Open drawer + pick | **0%** | 64% | **72%** |

WB-MPC 在 long-horizon "Open drawer + pick" 直接 **0%**（TLE），因为 MPC horizon 不够长，online recompute 太慢。LLM-Planner 因为有 library，把问题变成 sequential primitive call，所以能完成。Morphology Selector (MS) 在 approach object、pick and place 上有明显提升，因为这些 task 对 arm mode 和 locomotion mode 敏感。

**Time tradeoff**：LLM-Planner 比 WB-MPC 慢（API call overhead + BT construction），MS 进一步加时间。但 success rate gain 在复杂任务上 justify 了这个 latency。

### 6.4 Error Analysis（Figure 8, Section A2.3）

三类 error：
1. **Planning error**：LLM 在复杂任务上 BT 构造逻辑错或漏 action。解决：把 perception 和 self-state detection 嵌进 primitive 节点，降低 BT 层逻辑复杂度。
2. **Perception error**：AprilTag 在 motion 中有 2-4cm 误差；FoundationPose 受 occlusion 影响。
3. **Execution error**：占比最大。Pre-trained primitive 遇到 unexpected disturbance（如腿撞门框）无法 adapt。Long-horizon 上 error 累积 —— 例如 drawer pick 后 pose 偏一点，place 时撞到环境掉落。还有 actuator overheating 这种硬件问题。

---

## 7. 我的几点 critical observation

### 7.1 RL + Optimization 的 hybrid 是不是 over-engineered？

直接 end-to-end RL 训 38 DoF humanoid 在 Isaac Gym 里其实可行（HumanoidBench 已经做到），但 HYPERmotion 选择 reduced space RL + whole-body QP 这条路有两个 reason：
- Sim2Real gap：end-to-end policy 在 sim 里 work，real 上 sensor noise + actuator dynamics 会让 latent policy 行为不可预测
- Skill reusability：每个 primitive 可以独立迭代、独立 debug、独立 compose

代价是新增 skill 需要单独训 + 单独优化，paper 自己在 Conclusion 里承认这是 limitation。

### 7.2 VLM 做 morphology selector 的可靠性

Figure 6 显示 spatial data 加进去之后 locomotion morphology 选择率从 ~60% 到 ~90%，但 10% 失败率在 long-horizon 任务上会指数累积。Paper 没讨论 failure recovery 在 morphology selector 失败时怎么处理。这是后面可以做的：把 morphology selector 输出 confidence，低 confidence 时 fallback 到 default mode。

### 7.3 跟 RT-2 / OpenVLA 的对比

RT-2 / OpenVLA 走的是 VLA end-to-end 路线，直接从 (image, instruction) → action chunk。HYPERmotion 反其道而行之，**把 LLM 用在 planning 层，action 交给 RL + MPC**。这种 decoupling 的好处：
- LLM 不需要 fine-tune（直接 GPT-4o API）
- Skill 可解释、可调试
- 在 38 DoF 这种 high-dim 上端到端 VLA 数据量爆炸

代价是：
- 新 skill 要训
- LLM latency 高（~1-2s per planning call）
- 没有 closed-loop linguistic interaction

跟 SayCan / VoxPoser / SpatialVLM 这条线对比，HYPERmotion 是把 "LLM as planner" 的思路从 fixed-base arm 推到 floating-base humanoid + wheel-leg hybrid。它的 novelty 不在算法（每个模块都不新），而在 **system integration**：把 PPO + whole-body QP + LLM planner + VLM selector + BT + motion library 缝在一起，在真机器人上 work。

### 7.4 为什么用 Behavior Tree 不用 FSM 或 plain code？

BT 相对 FSM 的优势：
- Hierarchical，可嵌套
- Reactive：每个 tick 都重新 check condition，自然支持 re-plan
- 模块化：condition node 和 action node 可独立 swap

paper 在 Table A1 里专门对比了传统 BT-Planner（基于规则生成 BT）和 LLM-BT，LLM 生成 BT 的 exec rate 98% vs 92%，说明 LLM 对 BT XML 这种结构化输出格式适应得很好。

---

## 8. 可能的延伸方向（build your intuition）

1. **Skill library 的自动扩充**：当前每个 primitive 都要单独训。可以结合 RoboGen [Wang et al., 2023, https://robogen-website.github.io/] 的 generative simulation 思路，让 LLM 自动 generate task + reward + 训练脚本。

2. **Hierarchical RL 替代 BT**：BT 是 symbolic，可以用 diffusion policy 或 ACT 在 primitive 之间学一个 meta-policy。但这样会失去 reactivity。

3. **VLM 的 closed-loop feedback**：当前 VLM 只在 morphology selector 调用。可以让 VLM 持续监控 task progress，做 closed-loop re-planning。Reflexion [Shinn et al.] 的思路可以迁移过来。

4. **World model 的引入**：DreamerV3-style world model 可以让 planner 在 latent space 里 rollout，比 BT tick 更细粒度。

5. **Affordance 的 explicit grounding**：当前 affordance 隐式存在 VLM prompt 里。可以学 Distilled Feature Fields [Shen et al., CoRL 2023] 或 Neural Radiance Field + Language 的方式显式 encode affordance。

6. **从 primitive 到 continuous skill manifold**：Motion library 是离散的，可以用 task embedding 把所有 skill embed 到一个 latent space，然后做 interpolation（Unicorn [Wang et al.] 的思路）。

7. **Whole-body optimizer 的 differentiable 版本**：当前 HORIZON 是 CasADi-based NLP solver，latency 在 tens of ms。可以换成 differentiable MPC + neural network warm-start，online re-plan 频率提升到 100Hz，对 disturbance 鲁棒性会大幅提升。

---

## 9. 总结

HYPERmotion 在 engineering 上是一个相当扎实的 system paper。它的 contribution 不在单个 algorithm 的 SOTA，而在 **把 symbolic LLM planning 和 continuous whole-body control 这两个层级用 motion library + BT + morphology selector 缝起来**，让 38 DoF wheel-leg humanoid 能从 free-text 完成 long-horizon loco-manipulation。论文的实验在 sim 和 real 上都做了，long-horizon 任务 "open drawer + pick drill + put in box + bring to user" 真实机器人跑通了，这在 2024-2025 的人形机器人工作里属于前列。

**核心 takeaway for building intuition**：
- LLM 不要做 control，LLM 做 composition
- RL 不要做 whole-body 直接，RL 做 reduced-space + QP projection
- VLM 不要 end-to-end 输出 action，VLM 做 mode selection 这种 discrete decision
- 三者通过 motion library + BT 这两个 "interface" 解耦

这种 modular 思路 Karpathy 你应该在 Tesla 的工作里也深有体会 —— Optimus 的 stack 大概率也是类似的分层 architecture，只是 industrial scale 更细。HYPERmotion 给了一个 academic 上完整的 proof-of-concept。

---

Reference 链接汇总：
- Project: https://hy-motion.github.io/
- HORIZON optimizer: https://www.frontiersin.org/articles/10.3389/frobt.2022.899025/full
- xbot2 middleware: https://github.com/ADVRHumanoids/xbot2
- CartesI/O: https://github.com/ADVRHumanoids/cartesian_interface
- SayCan: https://say-can.github.io/
- PaLM-E: https://palm-e.github.io/
- RT-2: https://robot-transformer2.github.io/
- VoxPoser: https://voxposer.github.io/
- SpatialVLM: https://spatial-vlm.github.io/
- FoundationPose: https://nvlabs.github.io/FoundationPose/
- HumanoidBench: https://humanoidbench.github.io/
- AnyMal Parkour: https://anymal-parkour.github.io/
- RoboGen: https://robogen-website.github.io/
- Behavior Trees in Robotics: https://www.behavior-robotics.org/
