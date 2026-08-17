---
source_pdf: RoboCOIN.pdf
paper_sha256: a196a83c446a8b12220dea923a34bb6fa72f5629278b446b14433173e8d38ea8
processed_at: '2026-08-12T00:36:48-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# RoboCOIN 用人话说

## 这篇 paper 在干嘛

想象你在训练一个机器人学做饭。它得两只手配合——左手扶碗，右手倒水。这叫 **bimanual manipulation**。

要教机器人这个本事，你得给它看大量人类操作的视频（其实是 teleoperation 录的 trajectory 数据）。问题来了：

1. **数据不够**：双臂机器人的数据远少于单臂
2. **机器种类太多**：有双臂、半人形、全人形，有的用 parallel gripper（夹子），有的用 dexterous hand（灵巧手），DoF 从 12 到 14 不等
3. **标注太浅**：现有 dataset 只给你 "图像→动作" 的配对，没有告诉你 "这一步在干嘛"、"为什么这么动"

RoboCOIN 的核心就一句话：**搞一个大、杂、且有层次标注的双臂机器人数据集，再配一套工具链让大家能复用**。

---

## 数据集长啥样

### 规模

- **180,000+ 条** demonstration
- **421 个 task**（从简单的"把毛巾扔进篮子"到复杂的"把桃子放进抽屉并关上"）
- **15 个 robot platform**（这是关键卖点，别人最多 1-4 个）
- **16 种场景**（厨房、超市、办公室、餐厅...）

### 15 个平台分三类

**Dual-arm（双臂台式）**：12 个。比如 Agilex Cobot Magic，两只 6 DoF 手臂 + 夹子，装在桌面上。便宜，适合实验室。

**Half-humanoid（半人形）**：带升降平台的双臂。比如 Realman RMC-AIDA-L，两只 7 DoF 手臂装在一个能上下移动的柱子上。介于台式和全身人形之间。

**Humanoid（全人形）**：3 个。Unitree G1 这种，两条腿 + 两条 7 DoF 手臂 + 灵巧手。自由度最高，能满地跑，但贵且难控制。

### 遥操作方式四种

| 方式 | 原理 | 谁在用 |
|---|---|---|
| **Isomorphic Arm** | 主从同构臂，你动一根 leader 臂，robot 上的 follower 臂跟着动 | Agilex, Galaxea |
| **Exoskeleton** | 穿外骨骼衣，捕捉人手臂关节角度 | Realman 系列 |
| **VR** | 戴 VR 头显+手柄，在虚拟空间里操作 | Agibot, AI2 AlphaBot, Leju |
| **Motion Capture** | 光学或惯性动捕服，把人动作映射给 robot | Galbot, Unitree G1 |

这四种方式各有利弊，产生的 data 也有不同的 bias。比如 exoskeleton 的 joint angle 直接映射很自然，但 VR 操作的人看不到真实手感，grasp 成功率低。RoboCOIN 把这些混在一起，逼着 model 学到跨 teleoperation 方式的 invariant feature。

---

## 任务怎么分类的——2D Taxonomy

这是 paper 里我最喜欢的设计。他们搞了个二维 grid：

**横轴**：Action Coordination（双臂协调度）
- Low：两条手臂基本是**串行**的，比如先左手拿杯子，再右手倒水
- High：两条手臂**并行**协作，比如双手一起拧瓶盖

**纵轴**：Object Flexibility（物体柔性）
- Rigid：刚体（碗、盘子）
- Articulated：铰接体（抽屉、剪刀）
- Deformable：可变形体（毛巾、面包）

421 个 task 就分布在这个 grid 上。左下角是"单手操作刚体"（最简单），右上角是"双手协同操作变形体"（最难）。

**Intuition**：这个 grid 可以当 curriculum 用。先训左下角的 task，再往右上爬。实验里看到 model 在简单 task 上 80% 成功率，复杂 task 只有 20%，完全符合这个难度梯度。

---

## 核心创新：Hierarchical Capability Pyramid

这是 paper 真正的技术 contribution。一条 trajectory 的标注分三层：

### Layer 1: Trajectory-level（整条轨迹的全局描述）

用自然语言描述整个场景和目标。比如：
> "在厨房台面上，左边有个粉色碗，右边有长条面包和圆形面包，目标是把面包放进碗里"

这一层给 model 一个**全局语境**，相当于 LLM 的 system prompt。

### Layer 2: Segment-level（子任务分解）

把整条轨迹切成几个 subtask，**关键：segment 之间可以时间重叠**，因为双臂是并行的。

比如"pull bowl storage bread"这个 task 被切成 6 个 segment：
1. `move_bowl_right` — 右手把粉色碗挪到桌子中间
2. `grasp_long_bread_left` — 左手抓长面包
3. `place_long_bread_in_bowl` — 把长面包放碗里
4. `grasp_round_bread_left` — 左手抓圆面包
5. `place_round_bread_in_bowl` — 把圆面包放碗里
6. `End`

每个 segment 有 start/end frame index + instruction + 是否有 exception（比如 grasp 失败）。

**Intuition**：这一层教 model **temporal reasoning** 和 **task planning**。Diffusion Policy 或 ACT 这种 model 输出 action chunk 时，其实隐式学到了 segmentation，但 RoboCOIN 把这个 explicit 化了。

### Layer 3: Frame-level（逐帧运动状态）

每一帧用自然语言描述两只手的运动状态：
- Direction（运动方向）
- Velocity $v$（速度，m/s）
- Acceleration $a$（加速度，m/s²）
- Gripper state（open / close / transitioning）

具体怎么算的，用 sliding window：

设 end-effector 位置 $\mathbf{p}(t) \in \mathbb{R}^3$，时间窗 $[t-\Delta, t+\Delta]$：

$$v(t) = \frac{\|\mathbf{p}(t+\Delta) - \mathbf{p}(t-\Delta)\|_2}{2\Delta}$$

- $\mathbf{p}(t)$：时刻 $t$ 的末端执行器 3D 位置
- $\Delta$：半窗口大小（比如 0.1 秒）
- $\|\cdot\|_2$：L2 范数，算两点间欧氏距离
- $v(t)$：时刻 $t$ 的瞬时速度估计（中心差分）

$$a(t) = \frac{v(t+\Delta) - v(t-\Delta)}{2\Delta}$$

然后按 threshold 分类成文本：
- $v < 0.02$ m/s → "stationary"
- $0.02 \leq v < 0.10$ → "moving slowly"
- $v \geq 0.10$ → "moving fast"

**Intuition**：这一层提供 **dense low-level supervision**。VLA model 看图像直接预测 action，中间没有"我现在在干嘛"的 explicit 表示。Frame-level annotation 强制 model 学到"我此刻在 grasp 还是 place"这种状态识别能力。

### 三层之间的时间尺度

- Trajectory-level: $\mathcal{O}(T)$，整条 trajectory 长度（秒到分钟级）
- Segment-level: $\mathcal{O}(T/10)$，每个 segment（秒级）
- Frame-level: $\mathcal{O}(1)$，每帧（10-30 Hz 控制频率）

这跟人脑的双过程理论（System 1 快反应 / System 2 慢思考）很像。NVIDIA GR00T N1 也是 explicit dual-system 架构。RoboCOIN 的 pyramid 给三层都提供 training signal。

---

## CoRobot 框架——数据生产工厂

光有数据集不够，还得有工具链让数据能被生产、清洗、管理。CoRobot 就是这个工厂。

### 组件 1: RTML——轨迹质量约束语言

**RTML = Robot Trajectory Markup Language**，YAML 格式的 DSL。

为什么需要这个？因为人遥操作的数据**质量参差不齐**。有人操作很顺滑，有人很急躁（速度突跳），有人 grasp 总失败。直接喂给 model 学，model 会学到这些坏习惯。

RTML 把"什么是好轨迹"用规则写下来，自动检查。

设计原则三条：
1. **Motion Stability**：轨迹平滑，无突变
2. **Pose Consistency**：关键阶段末端位姿满足 task 约束
3. **Execution Efficiency**：速度与精度平衡

RTML 分两层约束：

**Global constraints**（整条轨迹）：

```yaml
global_constraints:
  velocity:
    linear:
      max: 0.5        # m/s, 硬上限，单帧速度不能超
      mean_max: 0.3    # m/s, 平均速度上限
  acceleration:
    linear:
      max: 12.0       # m/s², 加速度硬上限
```

- `max: 0.5`：任何一帧的速度 $\max_t v(t) \leq 0.5$ m/s
- `mean_max: 0.3`：整条轨迹平均速度 $\bar{v} \leq 0.3$ m/s

**Local stage constraints**（每个 phase 单独约束）：

```yaml
stages:
  - id: "grasp_long_bread_left"
    match_subtask: "Grasp the long bread with left hand"
    constraints:
      workspace:          # 这个阶段末端的 Cartesian 范围
        left:
          min: [0.05, -0.05, -0.05]
          max: [0.25, 0.35, 0.20]
      orientation:        # 朝向约束
        left:
          angular_mean_deviation_max: 0.8   # rad
          std_max: [0.5, 0.5, 0.8]           # rad, 每轴 std 上限
          angular_variance_max: 0.15
      velocity:
        linear:
          mean_max: 0.12
          std_max: 0.10
      idle_arm:            # 另一只手的"闲姿"约束
        arm: "right"
        velocity_linear_mean_max: 0.05
      temporal:
        duration_min: 2.0
        duration_max: 8.0
```

变量逐个解释：
- `workspace.left.min/max`：左手末端位置 $\mathbf{p}_{\text{left}}(t) \in [min, max]$ 的 bounding box
- `angular_mean_deviation_max: 0.8`：该阶段末端朝向相对于基准朝向的平均偏差 $\leq 0.8$ rad（约 46°）
- `std_max: [0.5, 0.5, 0.8]`：roll/pitch/yaw 三个轴的 std 分别 $\leq 0.5, 0.5, 0.8$ rad
- `angular_variance_max: 0.15`：朝向方差 $\leq 0.15$，用来抓抖动
- `idle_arm.velocity_linear_mean_max: 0.05`：左手 grasp 时右手平均速度应 $\leq 0.05$ m/s，超过说明右手在乱动

RTML evaluator 自动跑这些约束，输出每条轨迹的 quality score + 哪个 phase 哪个 metric 违规了。

### 组件 2: Annotation Toolchain

三层 annotation 的生成 pipeline：

**Trajectory-level**：
1. Object detection tool（可能是 GroundingDINO 之类）检测场景中物体位置
2. LLM（可能是 GPT-4V 或 Qwen-VL）把检测结果 + 图像 → 自然语言场景描述

**Segment-level**：
1. Rule-based keyframe detection：基于 velocity 变化点 + gripper state 变化点自动找 segment 边界
2. 人工 refine 这些边界
3. 每个 segment 配一句 instruction

**Frame-level**：
1. Sliding window 算 velocity/acceleration
2. Threshold 分类成 "stationary / moving slowly / moving fast"
3. Gripper state 直接读传感器

**Intuition**：trajectory-level 靠 VLM（贵但稀少），segment-level 靠 rule + human（中等成本），frame-level 纯 rule（便宜但密集）。成本和密度的 trade-off。

### 组件 3: Integrated Robotic Platform

基于 LeRobot 扩展。三个 feature：

**Unified Robot Control**：集成各厂家 SDK + ROS，统一控制接口。

**Fine-Grained Type Extension**：原版 LeRobot 只支持 image + action，这里加了 segment/frame 级 text annotation 的存储。

**Atomic Storage**：数据按 embodiment × task × environment 切成最小子集。

形式化：dataset $\mathcal{D} = \bigcup_i \mathcal{D}_i$，每个子集 $\mathcal{D}_i$ 有 tag set $\mathcal{T}_i = \{e_i, t_i, env_i, ...\}$。

下载时按 query $\mathcal{Q}$ 选：$\{\mathcal{D}_i : \mathcal{T}_i \cap \mathcal{Q} \neq \emptyset\}$。

比如你只想要 "Unitree G1 + 厨房 + 所有 grasp task"，下载这几个 atomic subset 就行，不用拉全量 180K。

---

## HAI 方法——怎么用 pyramid 训 model

光有 annotation 不够，得证明它有用。HAI（Hierarchical Annotation Integration）就是验证方法。

### 核心思路

**不改 VLA model 的架构和参数**，把三层 annotation 作为 additional input token 拼进去。

### 训练时

输入 token sequence：
$$\mathbf{x}_{\text{train}} = [\text{img}, \text{traj\_concept}, \text{seg\_subtask}, \text{frame\_state}, \text{human\_instr}, \text{action\_tokens}]$$

Model 在训练时看到完整的 hierarchical context，学到"怎么利用 context"的能力。

### 推理时

人不会在现场给你标注 segment 和 frame state，所以得**自动生成 context**：
- **Phase change detection**：基于 kinematic state change 自动判断当前 segment
- **State history summarization**：滑窗 summarize 历史 state

推理 token：
$$\mathbf{x}_{\text{infer}} = [\text{img}, \widehat{\text{traj\_concept}}, \widehat{\text{seg\_subtask}}(s_t), \widehat{\text{frame\_state}}(s_t), \text{human\_instr}, \cdot]$$

其中 $\widehat{\cdot}$ 是自动估计的，可能不准。

**Intuition**：这个 trick 类似 teacher-forcing 训练 + free-running 推理。训练时用 ground truth context 让 model 学会"如果有人告诉我现在在 grasp phase，我该怎么动作"，推理时 context 哪怕不完美，model 也已经学会了利用 context 的能力。

这种 train-test gap 是 HAI 的潜在弱点，paper 在 limitation 里承认了。

---

## 实验讲了啥

### RQ1: 多 embodiment 适配性

Platform: Realman RMC-AIDA-L（半人形，2×7 DoF，parallel gripper）
Model: $\pi_0$ + LoRA（r=16, lr=2.5e-5, 30K steps, batch 32）

三个 task 的结果：

| Task | Coordination | Object | Success Rate |
|---|---|---|---|
| Place towel into basket | Low | Deformable | 80% |
| Pass the bowl | High | Rigid | 40% |
| Place peach into drawer and close | High | Articulated | 20% |

**Intuition**：成功率随 coordination + flexibility 单调下降。验证了 taxonomy 的难度梯度是对的。

### RQ2: HAI 有没有用

同样的 $\pi_0$ + Realman 平台，加 HAI：

| Task | $\pi_0$ | $\pi_0$ + HAI | Gain |
|---|---|---|---|
| Place towel into basket | 80% | 90% | +10% |
| Place peach into drawer + close | 20% | **70%** | **+50%** |

**关键 insight**：简单 task 增益小（+10%），复杂 task 增益大（+50%）。

为什么？简单 task model 本身就能搞定，加 context 边际价值小。复杂 task 需要结构化 reasoning 来分解，HAI 恰好提供了这个脚手架。这跟 LLM 上 chain-of-thought 在难题上增益更大的现象一致。

### RQ3: RTML 过滤有没有用

Platform: Unitree G1（humanoid, 2×7 DoF, dexterous hand）
Model: GR00T N1.5（diffusion-based，partial finetune diffusion + projector, lr=1e-4, 10K steps）

两个 task：
- T1: "pick the grape and place into the plate"（单臂）
- T2: "push the bowl and place bread pieces into it"（双臂）

**RTML 过滤掉多少数据**：平均 **35.3%** trajectory 被剔除。

这个数字很惊人——人遥操作的数据里有三分之一是"不达标"的。

**Phase-wise 失败分布**（违规轨迹在哪个 phase 出问题）：

| Phase | Failure Share |
|---|---|
| Grasping | 52.7% |
| Moving | 17.8% |
| Place | (剩余约 30%) |

**Intuition**：grasping 占一半失败。因为 grasp 成功依赖触觉反馈，而遥操作把人的触觉切断了，操作者只能靠视觉猜，很容易失败。

**Metric-wise 失败分布**（违规轨迹违反了什么约束）：

| Metric | Failure Share |
|---|---|
| Velocity violation | 46.2% |
| Duration violation | 24.5% |
| Workspace violation | (剩余) |
| Orientation violation | (剩余) |

**Intuition**：速度违规占主导。人操作 teleoperation 时容易"急躁"，猛拉猛拽，对 robot 的 dynamics 不敏感。Duration 违规次之——要么太快要么太慢，task rhythm 不对。

**四种 fine-tuning 设置对比**（平均成功率）：

| Setting | 描述 | Δ vs Raw |
|---|---|---|
| GR00T-Raw | 原始数据 | 0% |
| GR00T-Coarse | 仅 global 约束过滤 | +3% |
| GR00T-Fine | global + phase 约束过滤 | +16% |
| GR00T-Mine | + 从其他 task 挖高质量 segment | +23% |

**关键 insight**：
- Global 约束（粗过滤）只带来 +3%，边际价值小
- Phase-level 约束（细过滤）带来 +13%（从 +3% 到 +16%），边际价值大
- 跨 task mining 高质量 segment 再加 +7%

**Intuition**：trajectory 的"局部阶段质量"比"全局统计"更能决定 policy 性能。这跟 curriculum learning 的思路一致——关键不是整体数据量，而是每个 subtask 有没有高质量 demonstration。

Mining 跨 task 高质量 segment 的思路很有意思。比如"grasp bread"这个 segment 在很多 task 里都出现，可以从所有 task 里挑出 grasp 质量最高的 segment 拼起来训。这是数据**复用**而不是单纯过滤。

### Boundary Cases 实验

为了验证 RTML 在难场景下也有用，他们挑了三种 challenging initial state：
- **Bread Rotated**：面包朝向极端
- **Bowl at Edge**：碗在 workspace 边缘
- **Bread Together**：面包紧挨着放

| Method | Success Rate |
|---|---|
| GR00T-Raw | 27.5% |
| GR00T-Fine | 35.0% |
| GR00T-Mine | 47.5% |

RTML 牺牲 edge case 覆盖度换取 reliability。在工业部署视角下这是合理的——宁可在极端场景失败，也不要在正常场景出意外。

---

## 我的几点直觉

### 1. 2D Taxonomy 是 Curriculum 的天然 scaffold

Action coordination × Object flexibility 这个 grid 本质上构造了一个 task 难度空间。可以建模为：

$$\text{difficulty}(c, f) \approx \alpha c + \beta f + \gamma c \cdot f$$

- $c \in [0,1]$：coordination 难度
- $f \in [0,1]$：object flexibility 难度
- $\gamma c \cdot f$：interaction term，对应"双臂协同操作柔性物体"这种 hardest case

这给 curriculum learning 提供了明确的路径：从 $(0,0)$ 训到 $(1,1)$。

### 2. 三层 Pyramid 对应三种时间尺度的 reasoning

- Trajectory-level：秒到分钟，对应 System 2 慢思考
- Segment-level：秒级，对应 task planning
- Frame-level：10-30 Hz，对应 System 1 快反应

这种 multi-resolution supervision 跟人脑的双过程理论吻合。NVIDIA GR00T N1 系列 explicit 做 dual-system，RoboCOIN 的 pyramid 给这种架构提供了 training signal。

### 3. RTML 本质是 expert knowledge 的 explicit encoding

把人类专家对"什么是好轨迹"的判断用 YAML 写成可计算约束。这跟 Programming by Demonstration 里的 constraint extraction 思路一致，但 RTML 是**先验定义**的，不从数据学。

未来方向应该是 **learned RTML**：用 VLM 从 demonstration 自动 induce constraint。比如让 GPT-4V 看几百条轨迹，自动总结出"grasp phase 时手腕朝向方差应该小于 0.15"这种规则。

### 4. HAI 的 Open-Loop 问题

推理时用 phase change detection 自动估 segment，这是个**开环**过程。如果 phase detection 错了，segment annotation 就错，可能级联放大。

更 robust 的做法是 closed-loop：让 VLA model 自己输出当前 phase 的 belief $p(\text{phase}=k | s_t)$，再 condition on 这个 belief。类似 Hidden Markov Model 的隐状态推断，让 model 自己当"segment detector"。

### 5. Multi-Embodiment 的真正难题没解决

Paper 在 limitation 里承认：没做 mixed-embodiment training 或 cross-embodiment transfer。这是 multi-embodiment learning 的圣杯。

核心难点：同样 "move-to-pose" 命令，对 Agilex Cobot Magic（2×6 DoF, parallel gripper）和 Unitree G1（2×7 DoF, dexterous hand）的执行特性完全不同。dynamics 差异太大。

可能的方向：
- **Action embedding per embodiment** + shared latent policy
- **Dynamics-aware action representation**（类似 world model）
- **Contrastive cross-embodiment pretraining**（把不同 embodiment 的同一 task 的 trajectory 在 latent space 对齐）

---

## 一句话总结

RoboCOIN 的 thesis：**bimanual generalist policy 需要 structured supervision**，单纯"图像→动作"的模仿学习不够。structured supervision 在三个层面体现：跨 embodiment（统一 data schema + RTML）、跨 task（segment decomposition）、跨时间尺度（frame-level dense signal + trajectory-level goal）。CoRobot 框架把"高质量 robot dataset 生产"从 ad-hoc 工程问题提升为可形式化、可复用、可扩展的系统工程问题。

未来真正的 breakthrough 可能不在于更大的 VLA model，在于更好的 data curation language + automated annotation。RoboCOIN 走的方向比单纯堆 data scale 更接近本质。

参考链接汇总：
- Paper: https://arxiv.org/abs/2507.02029 (BAAI RoboBrain 2.0 同组)
- LeRobot: https://github.com/huggingface/lerobot
- $\pi_0$: https://arxiv.org/abs/2410.24164
- GR00T N1.5: https://research.nvidia.com/labs/gear/gr00t-n1_5/
- Open-X-Embodiment: https://robotics-transformer-x.github.io/
- RDT-1B: https://arxiv.org/abs/2410.07864
- OpenVLA: https://openvla.github.io/
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- ACT/ALOHA: https://tonyzhaozh.github.io/aloha/
- AgiBot World: https://agibot-world.com/
- Galaxea Open-World: https://arxiv.org/abs/2509.00576
- DROID: https://droid-dataset.github.io/
- RoboMIND: https://arxiv.org/abs/2412.13877

---

# RoboCOIN 深度解析

## Paper 核心动机与定位

RoboCOIN 由 Beijing Academy of Artificial Intelligence (BAAI) 联合 15+ 家机器人公司 (Agilex, Galaxea, Unitree, Realman, Galbot, Leju, Agibot 等) 构建，核心目标是解决 bimanual manipulation 数据的 **embodiment heterogeneity** 问题。

关键 insight：现有 dataset 如 Open-X-Embodiment (1.4M trajectories, 22 embodiments) 虽然规模大，但**以单臂为主**；AgiBot World (1M, dual-arm) 和 Galaxea Open-World (50K, dual-arm) 虽然是 bimanual，但**局限于 single embodiment**（商业考量）。$\pi_0$ dataset 也是 closed-source。RoboCOIN 填补了 "multi-embodiment + bimanual + open-source + hierarchical annotation" 这个交叉空白。

参考链接：
- Paper arXiv (搜索): https://arxiv.org/abs/2507.02029 (RoboBrain 2.0 同组)
- LeRobot: https://github.com/huggingface/lerobot
- $\pi_0$: https://arxiv.org/abs/2410.24164
- GR00T N1.5: https://research.nvidia.com/labs/gear/gr00t-n1_5/
- Open-X-Embodiment: https://robotics-transformer-x.github.io/

---

## Dataset 规模与 Diversity 维度

| 维度 | RoboCOIN | Open-X-Embodiment | AgiBot World | Galaxea Open-World | RoboMIND |
|---|---|---|---|---|---|
| Trajectories | **180K+** | 1.4M | 1M | 50K | 107K |
| Embodiments | **15** | 22 | 1 | 1 | 4 |
| Tasks | 421 | 160K | 217 | 150 | 479 |
| Arm config | **Dual** | Single+Dual | Dual | Dual | Single+Dual |
| Annotation | **Hierarchical** | Flat | Flat | Flat | Flat |

15 个 platform 的 morphology 分布：
- **Dual-arm** (12 platforms): Agilex Cobot Magic, Agilex Split ALOHA, Galaxea R1 Lite, Realman RMC-AIDA-L, Agibot G1, AI2 AlphaBot 2/1s, Galbot G1, Tianqing A2, Realman Rs-02/01, Airbot MMK2, Leju Kuavo 4 LB
- **Humanoid** (3 platforms): Leju Kuavo 4 Pro, Unitree G1edu-u3

DoF 分布：2×6 (Agilex, Galaxea, Airbot) 到 2×7 (大多数 humanoid/dexterous 平台)。

Teleoperation methods 多样化：
- **Isomorphic Arm** (leader-follower): Agilex 系列, Galaxea
- **Exoskeleton**: Realman 系列
- **VR**: Agibot G1, AI2 AlphaBot, Tianqing A2, Leju Kuavo
- **Motion Capture**: Galbot G1, Unitree G1

这种 teleoperation method 的多样性本身就是一个 dataset bias source，论文里 RTML 试图量化这个 bias。

---

## Hierarchical Capability Pyramid 技术深度

这是 paper 最核心的 conceptual contribution。三层 annotation 形成多分辨率学习信号：

### (a) Trajectory-level — Global Scene Concepts

描述整个 task 的 scene configuration。具体包含：
- Environment setting (kitchen / office / supermarket / restaurant)
- Object placement (位置、关系)
- Task goal

Intuition：这一层对应 LLM-style "system prompt"，给模型一个全局 task framing。比如 "在厨房台面上，左手边有一个粉色碗，右手边有长条面包和圆形面包，目标是将面包放入碗中"。

### (b) Segment-level — Subtask Decomposition

将 task 分解为 temporally aligned subtasks，**关键点：segments 可重叠**（因为双臂可以并行操作）。每个 segment 包含：
- Start/end frame indices
- Step-by-step instruction
- Exception labels (grasping failure 等)

例如 "pull bowl storage bread" task 被 RTML 分解为 6 个 stages：
1. `move_bowl_right` — "Move the pink bowl to the center of table with right hand"
2. `grasp_long_bread_left` — "Grasp the long bread with left hand"
3. `place_long_bread_in_bowl`
4. `grasp_round_bread_left`
5. `place_round_bread_in_bowl`
6. `End`

### (c) Frame-level — Dense Kinematic State

每帧的自然语言 kinematic 描述。使用 sliding window 对 state sequence 量化 motion：
- Direction (方向)
- Velocity $v$ (m/s)
- Acceleration $a$ (m/s²)
- Gripper state (open/closed/transitioning)

Frame-level labeling 的 rule-based 公式逻辑：

对时间窗 $[t-\Delta, t+\Delta]$ 内的 end-effector position $\mathbf{p}(t) \in \mathbb{R}^3$：

$$v(t) = \frac{\|\mathbf{p}(t+\Delta) - \mathbf{p}(t-\Delta)\|_2}{2\Delta}$$

$$a(t) = \frac{v(t+\Delta) - v(t-\Delta)}{2\Delta}$$

Threshold 分类：
- $v < v_{\text{stationary}}$: "stationary"
- $v_{\text{stationary}} \leq v < v_{\text{slow}}$: "moving slowly"
- $v \geq v_{\text{fast}}$: "moving fast"

这一层提供 **intrinsic feedback control** 的 dense supervision。

---

## CoRobot 框架架构

CoRobot 是数据生产的"factory"，三个组件：

### Component 1: RTML Evaluator

**RTML = Robot Trajectory Markup Language**，YAML 格式的 DSL，定义 trajectory 质量约束。

设计原则三条：
1. **Motion Stability** — 平滑可预测，无突变
2. **Pose Consistency** — 关键 phase 的 end-effector pose 满足 task-specific 约束
3. **Execution Efficiency** — 速度与精度平衡

RTML 结构分两层：

**Global constraints** (整条 trajectory)：
```yaml
global_constraints:
  velocity:
    linear:
      max: 0.5      # m/s, 硬上限
      mean_max: 0.3  # m/s, 平均上限
  acceleration:
    linear:
      max: 12.0    # m/s²
  workspace:        # Cartesian 边界
    min: [x_min, y_min, z_min]
    max: [x_max, y_max, z_max]
  duration:
    min: T_min
    max: T_max
```

**Local stage constraints** (每个 phase)：
```yaml
stages:
  - id: "grasp_long_bread_left"
    match_subtask: "Grasp the long bread with left hand"
    constraints:
      workspace:    # 该阶段的 workspace override
        left:
          min: [0.05, -0.05, -0.05]
          max: [0.25, 0.35, 0.20]
      orientation:  # 朝向容忍度
        left:
          angular_mean_deviation_max: 0.8  # rad
          std_max: [0.5, 0.5, 0.8]
          angular_variance_max: 0.15
      velocity:
        linear:
          mean_max: 0.12
          std_max: 0.10
      idle_arm:    # 另一只手臂的"闲姿"约束
        arm: "right"
        velocity_linear_mean_max: 0.05
      temporal:
        duration_min: 2.0
        duration_max: 8.0
```

变量解释：
- `angular_mean_deviation_max`: 该阶段末端朝向相对于基准朝向的平均偏差上限（rad）
- `std_max`: 朝向 std 上限（每个轴独立）
- `angular_variance_max`: 朝向方差上限，捕捉抖动
- `idle_arm` 约束：当左手 grasp 时，右手 velocity_mean_max 应低于 0.05 m/s，否则视为"乱动"

**RTML Evaluator 输出**：每条 trajectory 的 quality score + phase-wise 违规报告。

### Component 2: Annotation Toolchain

Pipeline 分三步：
1. **Trajectory-level**: Object detection → LLM → scene description
2. **Segment-level**: Rule-based keyframe detection (基于 velocity/state 变化点) → 人工 refine
3. **Frame-level**: Sliding window 状态量化 → threshold → text label

### Component 3: Integrated Robotic Platform

基于 LeRobot 扩展，三个 feature：
- Unified Robot Control (官方 SDK + ROS)
- Fine-Grained Type Extension (支持 segment/frame 级 text annotation)
- **Atomic Storage** — 按 embodiment × task × environment 切分最小子集，通过 tag 动态组合

Atomic storage 公式逻辑：dataset $\mathcal{D} = \bigcup_i \mathcal{D}_i$，每个 $\mathcal{D}_i$ 由 tag set $\mathcal{T}_i = \{e_i, t_i, env_i, ...\}$ 索引。下载时按 query tag set $\mathcal{Q}$ 选 $\{\mathcal{D}_i : \mathcal{T}_i \cap \mathcal{Q} \neq \emptyset\}$。

---

## HAI (Hierarchical Annotation Integration) 方法

HAI 是 paper 用来验证 pyramid 价值的方法，核心 trick：**不改动原 VLA 架构**，把 hierarchical annotation 作为 additional input token 注入。

### 训练时

输入 token sequence：
$$\mathbf{x}_{\text{train}} = [\text{img}, \text{trajectory\_concept}, \text{segment\_subtask}, \text{frame\_state}, \text{human\_instr}, \text{action\_tokens}]$$

### 推理时

不能依赖人工 segment/frame annotation，所以**自动生成 context**：
- **Phase change detection**: 基于 kinematic state change 自动判断当前 segment
- **State history summarization**: 滑窗 summarize 历史 state

推理 token sequence：
$$\mathbf{x}_{\text{infer}} = [\text{img}, \widehat{\text{trajectory\_concept}}, \widehat{\text{segment\_subtask}}(s_t), \widehat{\text{frame\_state}}(s_t), \text{human\_instr}, \cdot]$$

其中 $\widehat{\cdot}$ 表示自动估计。

这个设计的关键 intuition：**用训练时的"完整 annotation 监督"教会模型利用结构化 context，推理时即使 context 不完美，模型也学到了"如何利用 context"的能力**。类似 teacher-forcing + free-running 的 gap。

---

## 实验深度分析

### RQ1: Multi-Embodiment 适配性

Platform: **Realman RMC-AIDA-L** (half-humanoid, 2×7 DoF, parallel gripper)
Model: $\pi_0$ + LoRA (r=16, $\eta=2.5 \times 10^{-5}$, 30K steps, batch 32)

| Task | Coordination | Object Flex. | Success Rate |
|---|---|---|---|
| Place towel into basket | Low | Deformable | **80%** |
| Pass the bowl | High | Rigid | **40%** |
| Place peach into drawer and close it | High | Articulated | **20%** |

Intuition：success rate 随 coordination difficulty + object flexibility 单调下降，验证了 taxonomy 的合理性。

### RQ2: HAI 效果

| Task | $\pi_0$ baseline | $\pi_0$ + HAI | $\Delta$ |
|---|---|---|---|
| Place towel into basket | 80% | 90% | +10% |
| Pass the bowl | 40% | - | - |
| Place peach into drawer + close | 20% | **70%** | **+50%** |

**关键发现**：复杂 task (high coordination + articulated object) 增益最大 (+50%)。Intuition：简单 task 模型本身就能搞定，复杂 task 需要结构化 reasoning 来分解，HAI 恰好提供了这个 scaffold。

### RQ3: RTML 数据质量影响

Platform: **Unitree G1** (humanoid, 2×7 DoF, dexterous hand)
Model: **GR00T N1.5** (diffusion-based, partial finetune on diffusion + projector, $\eta=10^{-4}$, 10K steps)

Tasks:
- T1: "pick the grape and place into the plate" (single-arm)
- T2: "push the bowl and place bread pieces into it" (bimanual)

**RTML 过滤数据量**：平均 **35.3%** trajectory 被剔除（说明 human teleoperation 数据中有约 1/3 是低质量的）。

**Phase-wise 失败分布**：
| Phase | Failure Share |
|---|---|
| Grasping | **52.7%** |
| Moving | 17.8% |
| Place | (剩余) |

Intuition：grasping 是 teleoperation 中最容易失败的 phase，因为 grasp 成功依赖 tactile feedback，而 teleoperation 隔离了这种 feedback。

**Metric-wise 失败分布**：
| Metric | Failure Share |
|---|---|
| Velocity violation | **46.2%** |
| Duration violation | 24.5% |
| Workspace violation | (剩余) |
| Orientation violation | (剩余) |

Intuition：velocity violation 占主导，说明人类 teleoperator 倾向"急躁"地移动，对 robot 的 dynamics 不敏感。

**Four-way fine-tuning 对比** (success rate 平均)：

| Setting | Description | Avg Success Rate | $\Delta$ vs Raw |
|---|---|---|---|
| GR00T-Raw | 原始数据 | baseline | 0% |
| GR00T-Coarse | 仅 global constraint filter | +3% | +3% |
| GR00T-Fine | global + phase-wise filter | +16% | **+16%** |
| GR00T-Mine | + mining 其他 task 的高质量 segment | **+23%** | **+23%** |

Intuition：**phase-wise 约束的边际价值 (+13% over coarse) 远大于 global 约束 (+3%)**。说明"局部阶段质量"比"全局统计"更能决定 policy 性能。Mining 跨 task 高质量 segment 又加 +7%，说明**数据复用**比单纯过滤更高效。

**Boundary cases 实验** (challenging initial states)：
| Method | Success Rate |
|---|---|
| GR00T-Raw | 27.5% |
| GR00T-Fine | 35.0% |
| GR00T-Mine | **47.5%** |

Boundary cases 包含：Bread Rotated (极端朝向), Bowl at Edge (workspace 边缘), Bread Together (紧凑放置)。RTML 牺牲 edge case 覆盖度换取 reliability，工业部署视角合理。

---

## 我的 Intuition Building

### 1. 关于 Dataset Design

RoboCOIN 的 2D taxonomy (action coordination × object flexibility) 是一个非常聪明的 dataset design primitive。它本质上构造了一个**技能难度 grid**，每个 cell 对应一类 manipulation primitive。这让我想到 supervised learning 中"curriculum learning"的思路：可以从 low-coordination + rigid 这类简单 cell 训起，逐步过渡到 high-coordination + deformable。

Formally，设 coordination difficulty 为 $c \in [0, 1]$，object flexibility 为 $f \in [0, 1]$，task 难度可建模为：
$$\text{difficulty} \approx \alpha \cdot c + \beta \cdot f + \gamma \cdot c \cdot f$$
其中 $\gamma \cdot c \cdot f$ 是 interaction term，对应 "需要双臂协调地操作柔性物体" 这种 hardest case。

### 2. 关于 Hierarchical Annotation

三层 annotation 对应三种时间尺度的 reasoning：
- Trajectory-level: $\mathcal{O}(T)$ — 秒级到分钟级
- Segment-level: $\mathcal{O}(T/10)$ — 秒级
- Frame-level: $\mathcal{O}(1)$ — 控制频率 (10-30 Hz)

这跟人脑的 "System 1 / System 2" 双过程理论契合。NVIDIA GR00T N1 系列就是 explicit dual-system：System 2 做慢推理 (planning)，System 1 做快反应 (motor control)。RoboCOIN 的 pyramid 提供了这种 dual-system 的 training signal。

### 3. 关于 RTML 的本质

RTML 本质上是把**人类专家知识显式编码为可计算的 constraint**。这跟 programming by demonstration (PbD) 中"constraint extraction"思路一致，但 RTML 是先验定义的，不是从数据中学的。

未来方向应该是 **learned RTML**：用 VLM 从 demonstration 中自动 induce constraint，类似 "discovering physical constraints from video" 这类工作。可以参考：
- Chain-of-Thought reasoning for constraints
- Differentiable constraint learning

### 4. 关于 HAI 的 Open Loop 问题

HAI 推理时用 phase change detection 自动估计 segment，这是一个**开环**的过程。如果 phase detection 出错，segment annotation 就会错，可能 cascade。更 robust 的做法是 closed-loop：让 VLA model 自身输出当前 phase 的 belief $p(\text{phase}=k | s_t)$，再 condition on 这个 belief。这类似 **Hidden Markov Model** 的隐状态推断。

### 5. Multi-Embodiment 的真正难题

Paper 承认 limitation：没有做 mixed-embodiment training 或 cross-embodiment transfer。这是 multi-embodiment learning 的圣杯。

现有方法如 $\pi_0$ 用 unified action space (relative pose + gripper)，RDT-1B 用 physically interpretable unified action space。但 cross-embodiment transfer 的核心难点是**动力学差异**：同样 move-to-pose command，对 Agilex Cobot Magic (2×6 DoF, parallel gripper) 和 Unitree G1 (2×7 DoF, dexterous hand) 的执行特性完全不同。

可能的解决方向：
- **Action embedding** per embodiment + shared latent policy
- **Dynamics-aware** action representation (类似 world model)
- **Contrastive** cross-embodiment pretraining

---

## 相关扩展阅读

- **Diffusion Policy**: https://diffusion-policy.cs.columbia.edu/
- **ACT (Action Chunking with Transformers)**: https://tonyzhaozh.github.io/aloha/
- **RDT-1B**: https://arxiv.org/abs/2410.07864
- **OpenVLA**: https://openvla.github.io/
- **Octo**: https://octo-model.github.io/
- **DROID dataset**: https://droid-dataset.github.io/
- **BridgeData V2**: https://rail-berkeley.github.io/bridgedata/
- **LIBERO**: https://lifelong-robot-learning.github.io/
- **CALVIN**: https://calvinrobot.github.io/
- **RoboMIND**: https://arxiv.org/abs/2412.13877
- **AgiBot World**: https://agibot-world.com/
- **Galaxea Open-World**: https://arxiv.org/abs/2509.00576
- **RoboBrain-X0**: https://github.com/FlagOpen/RoboBrain-X0

---

## 总结性 Intuition

RoboCOIN 的核心 thesis 可以浓缩为：**bimanual manipulation 的 generalist policy 需要 structured supervision**，单纯 action chunk 的模仿学习不够。这种 structured supervision 在三个层面 manifest：
1. **跨 embodiment** — 统一的 data schema + RTML constraint
2. **跨 task** — segment-level decomposition 提供 compositional structure
3. **跨时间尺度** — frame-level dense signal + trajectory-level goal signal

CoRobot 框架本质上是一个**数据生产 DSL**：RTML 是 constraint language，annotation toolchain 是 semantic compiler，integrated platform 是 runtime。这个 DSL 把"高质量 robot dataset 生产"从 ad-hoc 工程问题提升为可形式化、可复用、可扩展的系统工程问题。

未来真正的 breakthrough 可能不是更大的 VLA model，而是**更好的 data curation language + automated annotation**。在这个意义上，RoboCOIN 走的方向比单纯堆 data scale 更接近本质。
