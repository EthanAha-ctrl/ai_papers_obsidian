---
source_pdf: HOVER Versatile Neural Whole-Body Controller for Humanoid Robots.pdf
paper_sha256: 7ea05e065f5f16f6a93fa2dcbccc2fc07215943fc431934ccda1c50a0b13ce66
processed_at: '2026-08-19T11:30:48-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 HOVER

## 一句话版本

人形机器人干不同活儿要用不同的"遥控器接口"——走路用方向盘，抓东西用手柄，VR遥操作用体感——过去每个接口训一个policy，HOVER用一个网络全搞定，还能在线无缝切换，效果还比专用的好。

---

## 这事儿到底为啥是个问题

想象你买了台Unitree H1，19个关节，51.5公斤，1.8米高。你想让它干活，发现：

- **让它走路**：你得给它发"往前走0.5米/秒"这种命令，这叫root velocity tracking
- **让它挥手**：你得给它发"左肩抬30度、左肘弯60度"这种命令，这叫joint angle tracking  
- **用VR遥操作它**：你得给它发"左手在空间位置(x,y,z)"这种命令，这叫kinematic position tracking

三种命令，三种数学空间，三种reward设计，过去community各自为政：

| 谁干的 | 用啥接口 | 干啥活 |
|---|---|---|
| Radosavovic et al. (Science Robotics) | root velocity | 走路 |
| ExBody (Cheng et al.) | upper joint + lower root | 表演动作 |
| HumanPlus (Fu et al.) | 全身joint + root | 影子模仿 |
| OmniH2O (He et al.) | head+双手位置 | VR遥操作 |
| H2O (He et al.) | 8个keypoint位置 | 全身体感遥操作 |

问题来了：robot走路上突然想抓个东西，你得把root velocity控制器关掉，启动joint angle控制器，两个policy的输出格式都不一样，切换的瞬间robot大概率摔。

**HOVER说：我一个policy全管，你随便切，不摔。**

---

## 核心insight：人类动作是万能"底漆"

HOVER的作者发现了一个关键事实——**不管你用哪种命令接口，robot最终都要做"像人的运动"**。走路要像人走，挥手要像人挥，VR遥操作更是一比一模仿人。

所以与其针对每种命令接口单独训policy，不如先训一个"完美模仿人类动作"的teacher，再让student学怎么从各种命令接口出发去调用这个teacher的skill。

人类动作数据从哪来？AMASS dataset，慕尼黑工大和MPI搞的，几千段mocap数据，跑步跳舞踢球打拳都有。

AMASS: https://amass.is.tue.mpg.de/

---

## 两阶段训练，用做菜打比方

### Stage 1: 训Oracle（大师傅）

Oracle是一个能看到一切信息的"作弊policy"。它训练时能看到：

$$s_t^{\text{p-oracle}} = [\mathbf{p}_t, \boldsymbol{\theta}_t, \dot{\mathbf{p}}_t, \boldsymbol{\omega}_t, \mathbf{a}_{t-1}]$$

逐个翻译：
- $\mathbf{p}_t$：每个身体部位的全局3D坐标（真机上测不准，仿真里白嫖）
- $\boldsymbol{\theta}_t$：每个身体部位朝哪转（quaternion）
- $\dot{\mathbf{p}}_t$：线速度
- $\boldsymbol{\omega}_t$：角速度  
- $\mathbf{a}_{t-1}$：上一帧的动作

Oracle的目标只有一个：**完美模仿AMASS里的人类动作**。reward设计得很丰富（Table II那张大表），核心是tracking reward（body position权重80，root velocity权重100），加上一堆penalty（别摔、别超joint limit、别打滑、别踩太重）和regularization（动作平滑、别抖）。

用PPO训，3层MLP [512, 256, 128]，在IsaacGym里跑。最终oracle survive rate 99.3%——基本什么人类动作都能模仿。

### Stage 2: 蒸馏Student（徒弟）

Student是最终部署的policy，它看不到oracle那些privileged信息，只能看：

$$s_t^{\text{p-student}} = [\mathbf{q}, \dot{\mathbf{q}}, \boldsymbol{\omega}^{\text{base}}, \dot{\mathbf{g}}]_{t-25:t} \cup [\mathbf{a}]_{t-25:t-1}$$

翻译：
- $\mathbf{q}$：19个关节编码器读数（真机直接有）
- $\dot{\mathbf{q}}$：关节速度
- $\boldsymbol{\omega}^{\text{base}}$：躯干IMU角速度
- $\dot{\mathbf{g}}$：重力向量在body frame投影（用来估robot歪没歪）
- 下标$_{t-25:t}$：过去25帧堆叠，50Hz下就是0.5秒，大概一个步态周期

Student看不到全局位置，那它怎么知道自己在哪？靠25帧history推断——这跟人闭眼走路靠本体感觉一个道理。这个trick来自RMA (Robust Motor Adaptation) 那条线。

Student的训练用DAgger：

1. Student在仿真里rollout
2. 每一帧同时算出oracle看到的state
3. Oracle给出"正确动作"$\hat{\mathbf{a}}_t$
4. Student监督学习：$\mathcal{L} = \|\hat{\mathbf{a}}_t - \mathbf{a}_t\|_2^2$

就是让student的动作尽量贴近oracle的动作。

DAgger paper: https://arxiv.org/abs/1011.0686

---

## Mask机制：HOVER真正的魔法

这是整篇paper最聪明的部分。

Student的命令输入长这样：

$$s_t^{\text{g-student}} = M_{\text{sparsity}} \odot [M_{\text{mode}} \odot s_t^{\text{g-upper}}, M_{\text{mode}} \odot s_t^{\text{g-lower}}]$$

$\odot$是逐元素乘，两层mask：

### 第一层：$M_{\text{mode}}$（模式mask）

决定上半身和下半身各自用什么控制模式。三种模式：
- **Kinematic position**：跟踪关键点的3D位置（适合VR遥操作）
- **Joint angle**：跟踪关节角度（适合精细操作）
- **Root tracking**：跟踪躯干速度/高度/朝向（适合走路）

上下半身独立选择。比如上半身用kinematic position（VR控制手），下半身用root tracking（手柄控制走路）——这就是ExBody mode。

### 第二层：$M_{\text{sparsity}}$（稀疏mask）

在选定的模式内部，进一步决定激活哪些维度。比如上半身选了kinematic position模式，但只激活左手位置——这就是"left-hand mode"，专门跟踪一只手。

### Mask怎么采样

每个bit独立从Bernoulli(0.5)采样，**episode开始时采样一次，整个episode不变**。

这模拟了真实场景：你用VR遥操作时，整个session都在用同一个控制模式。但训练时随机采样，让policy见过所有可能的组合。

### 为什么这个设计牛

- **任意组合**：15+种有用mode都是这两层mask的子集，Table I展示了HOVER怎么覆盖ExBody/H20/OmniH2O/HumanPlus全部prior work
- **在线切换**：部署时改mask就行，policy见过的state分布已经覆盖了切换场景
- **天然容错**：VR信号丢了？把对应mask位置0，policy自动用其他信号继续控制

本质上**mask就是RL版的prompt**——LLM见过各种prompt组合能泛化，HOVER见过各种mask组合也能泛化。

---

## 实验结果：反直觉的"通才>专才"

### 仿真结果（Table III）

以ExBody mode为例，ExBody specialist是专门为这个mode训的，HOVER是同一个policy在ExBody mask下的表现：

| 指标 | Specialist | HOVER | 
|---|---|---|
| 全局位置误差 | 275mm | **185mm** |
| 局部位置误差 | 83.1mm | **63.9mm** |
| 上肢关节误差 | 0.166rad | **0.148rad** |
| 下肢关节误差 | 0.243rad | **0.210rad** |

HOVER在12个metric里7个更好。**一个啥都会的policy比专门训的还强**。

为啥？作者的解释是multi-mode训练起了regularization作用。单mode policy容易过拟合到特定reward结构，而HOVER被迫学到更抽象的"怎么像人一样动"，这种共享物理知识让它每个mode都做得更好。

### 跟Multi-Mode RL对比（Figure 4）

更关键的对比：同样multi-mode，distillation vs RL from scratch。

HOVER在32/32个metric-mode组合上全赢。这说明问题不在"multi-mode难"，在"直接用RL训multi-mode会reward conflict"。

不同mode的reward互相竞争梯度，policy找不到满足所有的optimum。Distillation绕开了这个：oracle只管imitation（reward单一），student只管模仿oracle（loss单一），两个stage各自clean。

### 真机结果（Table V, Figure 6）

Unitree H1上测20个standing motion：

| Mode | Specialist | HOVER |
|---|---|---|
| ExBody | 51.3mm | **48.9mm** |
| HumanPlus | 51.0mm | **47.4mm** |  
| OmniH2O | 51.2mm | **47.5mm** |

Figure 6最酷：robot走路走到一半，从ExBody mode在线切到H2O mode，平滑过渡没摔倒。还演示了用Vision Pro遥操作时随机mask掉头和手的位置，robot照样能跟踪剩下的信号。

---

## 直觉总结

### 1. Distillation为啥比multi-task RL强

Multi-task RL失败的原因是reward互相打架。ExBody想要joint accuracy，H2O想要hand position accuracy，同一个reward function没法同时优化——梯度方向不一致。

Distillation把问题拆两半：
- Oracle只管"完美模仿人类"（一个目标）
- Student只管"模仿oracle的动作"（一个loss）

两个stage各自的optimization都很干净，组合起来反而更好。这是"分而治之"在RL里的胜利，跟AlphaZero用MCTS当teacher再distill一个道理。

### 2. 人类动作数据为什么重要

AMASS给了robot一个"运动先验"。平衡、协调、步态周期这些东西，人类经过几百万年进化已经优化好了，robot直接学就行，不用从零探索。

这跟LLM用人类文本当预训练数据一个逻辑——人类语言/运动里蕴含的结构性知识，是很好的inductive bias。

### 3. Mask为啥是universal interface

Mask把command space变成了"约束集合"。policy学到的是"在给定约束下做最合理的运动"。

这跟in-context learning结构相似：LLM见过各种prompt组合能泛化到新prompt，HOVER见过各种mask组合能泛化到新mode组合。**Mask是RL版的prompt engineering**。

### 4. 局限性

- Mode switch是手动的，paper说future work做自动切换
- 真机只测了standing+locomotion，没测complex manipulation
- Student上限是oracle，oracle不行的动作student也不行
- 19-DOF有限，更高DOF robot的command space会爆炸，Bernoulli(0.5)采样可能覆盖不全

---

## 更广的implication

HOVER的范式——**privileged teacher + masked student**——可以推广到很多场景：

- Mobile manipulation：teacher看全局物体位置，student只看RGB，mask决定抓哪个
- Multi-robot：teacher看所有robot状态，student只看邻居，mask决定跟谁协作  
- Autonomous driving：teacher看全场景ground truth，student只看sensor，mask决定关注哪个agent

本质上是把"信息不对称"显式建模到训练pipeline里，用mask控制不对称的程度。

---

## 参考

- HOVER arXiv: https://arxiv.org/abs/2410.21229
- Project page: https://hover-humanoid.github.io/
- OmniH2O: https://omni.human2humanoid.com/
- HumanPlus: https://humanplus.github.io/
- ExBody: https://expressive-humanoid.github.io/
- AMASS: https://amass.is.tue.mpg.de/
- IsaacGym: https://developer.nvidia.com/isaac-gym
- PPO: https://arxiv.org/abs/1707.06347
- DAgger: https://arxiv.org/abs/1011.0686
- DeepMimic: https://xbpeng.github.io/projects/DeepMimic/
- MaskedMimic: https://research.nvidia.com/labs/toronto-ai/maskedmimic/
- Unitree H1: https://www.unitree.com/h1/

---

# HOVER: 详解一个统一的人形机器人全身控制器

## TL;DR — 核心直觉

HOVER 解决的是一个看似简单但本质深刻的问题：**人形机器人不同任务需要不同的"控制接口"**（locomotion 想要 root velocity tracking，tabletop manipulation 想要 upper-body joint angle tracking，teleoperation 想要 kinematic position tracking），过去每个接口都要单独训一个 policy。HOVER 的核心 insight 是——**full-body kinematic motion imitation 是所有这些任务的一个 common abstraction**，它可以作为 motor skill 的"通用先验"，然后通过 distillation 把这些 skill 压到一个支持 multi-mode 的 generalist policy 里。

最终一个 19-DOF Unitree H1 上的单一 neural network 可以无缝切换 15+ 种 control mode，并且在每种 mode 下都比专门为该 mode 训练的 specialist 表现更好——这是一个相当反直觉的 result，因为通常 generalist 会比 specialist 略差。

Paper arXiv: https://arxiv.org/abs/2410.21229
Project page: https://hover-humanoid.github.io/

---

## 1. Problem Setting:为什么这是个难问题

### 1.1 Command space 的碎片化

人形机器人 community 过去几年形成了三大家族的控制接口，每个都对应一类 task：

| Family | 代表工作 | Command 空间 | 适用场景 |
|---|---|---|---|
| Root velocity tracking | Radosavovic et al. (Science Robotics 2024), Berkeley Humanoid | $\dot{p}_{root}$, height, orientation | Locomotion, terrain traversal |
| Joint angle tracking | ExBody (Cheng et al. 2024), HumanPlus (Fu et al. 2024) | $q_{target}$ per motor | Expressive motion, manipulation |
| Kinematic position tracking | OmniH2O (He et al. 2024), H2O (He et al. 2024) | 3D positions of keypoints | Teleoperation (VR, exoskeleton) |

Table I 在 paper 里展示得很清楚：ExBody、H20、OmniH2O、HumanPlus 各自只支持自己那一小块 command space，互不兼容。一个 robot 走路时用 root tracking 控制器，要做 bimanual manipulation 时就得切换到 joint angle 控制器——这在工程上意味着两套 reward、两套训练 pipeline、两套 sim-to-real tuning。

### 1.2 为什么不能简单"拼起来"

直观想法是：把所有 command 维度 concat 起来，训一个 big policy。但有两个问题：

1. **维度不齐**：kinematic position 是 3D×N_keypoints，joint angle 是 N_dof，root velocity 是 6D，三者语义完全不同。
2. **训练信号冲突**：如果同时给所有 reward term，policy 会陷入 local optimum，因为不同 task 的 reward scale、稀疏度都不一样。

HOVER 的解法是 mask + distillation，下面细讲。

---

## 2. 整体架构：两阶段 Teacher-Student

```
┌──────────────────────────────────────────────────────────────┐
│  STAGE 1: Oracle Policy (Teacher)                            │
│  - Input: privileged proprioception + full reference motion  │
│  - Trained with PPO + rich reward (Table II)                 │
│  - Learns human-like motion imitation from AMASS             │
└──────────────────────────────────────────────────────────────┘
                            │
                            │  DAgger distillation
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  STAGE 2: Student Policy (HOVER)                             │
│  - Input: realistic proprioception (25-frame history)        │
│         + masked command (mode mask × sparsity mask)         │
│  - Trained with supervised learning on oracle actions        │
│  - Output: 19-D target joint position → PD controller        │
└──────────────────────────────────────────────────────────────┘
```

这个两阶段范式在 legged locomotion community 已经成熟（参考 AnyMal 的 teacher-student、RMA 等），HOVER 把它用到 whole-body humanoid 上，关键创新在 **Stage 2 的 masking 机制**。

---

## 3. Method 深度解析

### 3.1 Oracle Policy: privileged teacher

Oracle 的目标是"完美模仿人类动作"，它能看到 student 看不到的信息。

**State space**:

$$s_t^{\text{p-oracle}} \triangleq [\mathbf{p}_t, \boldsymbol{\theta}_t, \dot{\mathbf{p}}_t, \boldsymbol{\omega}_t, \mathbf{a}_{t-1}]$$

变量含义：
- $\mathbf{p}_t$：所有 rigid body 的全局 3D 位置（privileged，real robot 测不准）
- $\boldsymbol{\theta}_t$：所有 rigid body 的全局朝向（quaternion 或 rotation matrix）
- $\dot{\mathbf{p}}_t$：rigid body 线速度（仿真里直接拿，真机要估）
- $\boldsymbol{\omega}_t$：rigid body 角速度
- $\mathbf{a}_{t-1}$：上一帧 action（19-D target joint position）

**Goal space**:

$$s_t^{\text{g-oracle}} \triangleq [\hat{\boldsymbol{\theta}}_{t+1} \ominus \boldsymbol{\theta}_t, \hat{\mathbf{p}}_{t+1} - \mathbf{p}_t, \hat{\mathbf{v}}_{t+1} - \mathbf{v}_t, \hat{\boldsymbol{\omega}}_{t+1} - \boldsymbol{\omega}_t, \hat{\boldsymbol{\theta}}_t, \hat{\mathbf{p}}_t]$$

这里设计很关键——**它用的是"差分目标"而不是绝对目标**：
- $\hat{\boldsymbol{\theta}}_{t+1} \ominus \boldsymbol{\theta}_t$：参考下一帧朝向相对当前朝向的增量（$\ominus$ 是 quaternion 差运算）
- $\hat{\mathbf{p}}_{t+1} - \mathbf{p}_t$：参考下一帧位置相对当前位置的位移
- $\hat{\mathbf{v}}_{t+1} - \mathbf{v}_t$, $\hat{\boldsymbol{\omega}}_{t+1} - \boldsymbol{\omega}_t$：速度增量
- $\hat{\boldsymbol{\theta}}_t, \hat{\mathbf{p}}_t$：当前帧的绝对参考（作为 anchor）

这种"relative + absolute"混合表示让 policy 更容易学到"invariant motion pattern"——不管 robot 当前在哪儿，要做的相对运动是一样的。这是 imitation learning 里一个 well-known trick，DeepMimic (Peng et al. 2018) 也用了类似设计。

**Network**: 3-layer MLP [512, 256, 128]，比较 moderate 的 size。

### 3.2 Reward 设计（Table II 详解）

这是 humanoid RL 最难调的部分。Table II 把 reward 分成三类：

**Penalty（负 reward，惩罚不可行行为）**:
- Torque limits (-2): 防止电机烧毁
- DoF position limits (-1.25e2): 防止关节超限
- DoF velocity limits (-5e1): 防止关节速度过大
- Termination (-2.5e2): 摔倒的强惩罚

**Regularization（让运动平滑、自然）**:
- DoF acceleration (-1.1e-5): jerk 最小化
- DoF velocity (-4e-3): 静态时不要乱动
- Lower Action rate (-3) / Upper Action rate (-6.25e-1): action smoothness，注意 lower body 权重比 upper body 大 5 倍——因为腿的动作频率更高、更影响 stability
- Torque (-1e-4): energy efficiency
- Feet orientation (-6.25e1): 脚平放地面
- Feet air time (1e3): **正 reward**，鼓励合理的步态周期
- Feet contact force (-7.5e-1): 不要踩太重
- Stumble (-1.25e3): 不要被绊倒
- Slippage (-7.5e1): 脚不要打滑
- In the air (-2e2): 不要双脚同时离地（除非跳跃动作）
- Max feet height per step (-3e3): 抬腿不要过高

**Task Reward（核心 tracking reward）**:
- DoF position (3.2e1) / DoF velocity (1.6e1): joint tracking
- Body position (8e1) / Body rotation (2e1): rigid body 6D pose tracking
- Body velocity (8) / Body angular velocity (8): velocity tracking
- Root velocity (1e2) / Root rotation (2e1): root tracking

注意 **body position 权重 8e1 远高于 body rotation 2e1**——说明作者认为位置比朝向更重要，这跟 MPJPE 这类 metric 一致。

### 3.3 Motion Retargeting：从 AMASS 到 humanoid

这部分跟 OmniH2O 一样，三步：

1. **Forward kinematics**：把 humanoid 当前 joint configuration 映射到 workspace keypoints
2. **SMPL fitting**：优化 SMPL 参数让它的 keypoints 跟 humanoid keypoints 对齐（SMPL 是人体 mesh 模型，有 24 个 joint + shape parameters）
3. **AMASS retargeting**：用 gradient descent 把 AMASS 里的人类 motion 数据对齐到 fitted SMPL，再映射回 humanoid joint space

最终得到 $\hat{Q}$：humanoid-feasible motion dataset。这步是为了剔除 AMASS 里 humanoid 做不出来的动作（比如过度扭腰、手指动作）。

AMASS: https://amass.is.tue.mpg.de/
SMPL model: https://smpl.is.tue.mpg.de/

### 3.4 Student Policy: HOVER 的核心创新

**Proprioception（realistic, deployable）**:

$$s_t^{\text{p-student}} \triangleq [\mathbf{q}, \dot{\mathbf{q}}, \boldsymbol{\omega}^{\text{base}}, \dot{\mathbf{g}}]_{t-25:t} \cup [\mathbf{a}]_{t-25:t-1}$$

变量含义：
- $\mathbf{q}$：joint position（19-D，encoder 直接读）
- $\dot{\mathbf{q}}$：joint velocity（19-D）
- $\boldsymbol{\omega}^{\text{base}}$：base angular velocity（IMU 3-D）
- $\dot{\mathbf{g}}$：gravity vector 在 body frame 的投影（3-D，IMU 估倾角）
- 下标 $t-25:t$：过去 25 帧的 history stack
- $\mathbf{a}_{t-25:t-1}$：过去 25 帧的 action history

总维度大致是 $(19+19+3+3) \times 25 + 19 \times 24 \approx 1243$ 维。History 是为了替代 oracle 的 privileged state——student 没有 rigid body 全局位置，但可以通过 history 推断 system dynamics。这是 RMA (Robust Motor Adaptation) 范式的延伸。

**Command Masking（核心创新）**:

$$s_t^{\text{g-student}} \triangleq M_{\text{sparsity}} \odot [M_{\text{mode}} \odot s_t^{\text{g-upper}}, M_{\text{mode}} \odot s_t^{\text{g-lower}}]$$

两层 mask：

1. **$M_{\text{mode}}$**：mode-level mask，决定 upper body / lower body 各自走哪种 mode
   - 例：upper = kinematic position, lower = root tracking → 这是 H2O mode
   - 例：upper = joint angle, lower = root tracking → 这是 ExBody mode
   
2. **$M_{\text{sparsity}}$**：dimension-level mask，在 mode 内部进一步选择哪些维度激活
   - 例：upper 在 kinematic position mode 下，只激活 left hand → "left-hand mode"
   - 例：upper 在 kinematic position mode 下，激活 head + 2 hands → OmniH2O mode

每个 bit 独立采样自 $\text{Bernoulli}(0.5)$，**episode 开始时采样，整个 episode 保持不变**——这是为了模拟真实场景：用户在整个 task 期间用同一个 control mode。

这种设计的妙处：
- **Atomicity**：每个 command 维度独立，可以任意组合
- **Generality**：覆盖了 ExBody / HumanPlus / H2O / OmniH2O 所有 prior mode（Table I）
- 训练时随机 mask → policy 见过所有子集组合 → 部署时可以任意切换

### 3.5 DAgger Distillation

Student 用 DAgger (Dataset Aggregation) 训练：

1. 用 student 在仿真里 rollout，得到轨迹 $\{(s_t^{\text{p-student}}, s_t^{\text{g-student}})\}$
2. 同时计算对应的 oracle state $(s_t^{\text{p-oracle}}, s_t^{\text{g-oracle}})$
3. 用 oracle policy 得到 reference action $\hat{\mathbf{a}}_t = \pi^{\text{oracle}}(s_t^{\text{p-oracle}}, s_t^{\text{g-oracle}})$
4. 监督学习：$\mathcal{L} = \|\hat{\mathbf{a}}_t - \mathbf{a}_t\|_2^2$

为什么用 DAgger 而不是 behavioral cloning？因为纯 BC 会有 covariate shift——student 走偏了之后见到的 state 分布跟训练时不一样。DAgger 通过迭代 rollout + re-label 修复这个问题。

DAgger paper: https://arxiv.org/abs/1011.0686

---

## 4. 实验数据深度解读

### 4.1 Q1: HOVER vs Specialists（Table III）

Table III 是全文最重磅的 result。以 ExBody mode 为例：

| Metric | ExBody Specialist | HOVER | 提升 |
|---|---|---|---|
| $E_{\text{g-mpjpe}}$ (mm) | 275±1.65 | **185±1.11** | -32.7% |
| $E_{\text{mpjpe}}$ (mm) | 83.1±0.50 | **63.9±0.38** | -23.1% |
| $E_{\text{upper-j}}$ (rad) | 0.166±0.002 | **0.148±0.002** | -10.8% |
| $E_{\text{lower-j}}$ (rad) | 0.243±0.003 | **0.210±0.004** | -13.6% |
| $E_{\text{root-vel}}$ (m/s) | 0.428±0.007 | 0.452±0.006 | +5.6% (略差) |

HOVER 在 ExBody mode 下，**即使 ExBody specialist 是专门为这个 mode 训的**，HOVER 在 7/12 个 metric 上更好。

作者的解释（也是我觉得最值得思考的）：

> "We hypothesize that this is due to the policy leveraging shared physical knowledge across modes, such as maintaining balance, human-like motion, and precise limb control. These shared skills enhance generalization, leading to better performance across all modes. In contrast, single-mode policies often overfit to specific reward structures and training environments."

换句话说，**multi-mode training 起到了 regularization 作用**。当一个 policy 同时要处理 kinematic position tracking 和 joint angle tracking，它必须学到更抽象的 "how to move human-like" 而不只是 "how to minimize a specific reward"。这跟 multi-task learning 在 NLP/CV 里的发现一致——task diversity 是天然的正则化器。

### 4.2 Q2: HOVER vs Multi-Mode RL Baseline（Figure 4）

这个对比更关键：同样是 multi-mode，**用 distillation vs 用 RL from scratch**？

Figure 4 的 radar chart 显示 HOVER 在 32/32 metrics-modes 组合上都赢。这说明：

- Oracle policy 的"完美 imitation 先验"是关键
- RL from scratch 在 multi-mode 下会陷入 reward conflict——不同 mode 的 reward 互相竞争，policy 找不到满足所有的 optimum
- Distillation 绕过了这个问题：oracle 已经解决了 "how to move"，student 只需要学 "how to map command → action"

这跟 AlphaZero 的 intuition 类似：先用 MCTS 这种 strong search 得到超人类 policy，再 distill 成轻量网络。HOVER 用 privileged-state oracle 当 teacher，逻辑一样。

### 4.3 Q3: Real-World Transfer（Table V, Figure 6）

20 个 standing motion 在 Unitree H1 上测试：

| Mode | Specialist $E_{\text{g-mpjpe}}$ | HOVER $E_{\text{g-mpjpe}}$ |
|---|---|---|
| ExBody | 51.3 mm | **48.9 mm** |
| HumanPlus | 51.0 mm | **47.4 mm** |
| OmniH2O | 51.2 mm | **47.5 mm** |

Real-world 的 gap 比仿真大（仿真里 185mm，真机 48.9mm——因为真机测的是 standing motion，仿真测的是整个 AMASS 包括 dynamic motion），但 HOVER 一致地优于 specialist。

Figure 6 最 interesting：**在线切换 control mode**。Robot 走路走到一半，从 ExBody mode 切到 H2O mode，policy 平滑过渡，没有摔倒。这是 multi-mode training 的直接红利——policy 见过 mode switch 的场景（因为 episode 间 mask 不同），所以对 command space 的"突变"有鲁棒性。

---

## 5. 与相关工作的位置

| 维度 | HOVER | MHC (Dugar et al. 2024) | MaskedMimic (Tessler et al. 2024) |
|---|---|---|---|
| 硬件 | Real humanoid (H1) | Real humanoid | Simulation only (graphics) |
| Mode 覆盖 | 任意子集组合 | joint + root only | kinematic constraints |
| Training | Distillation from oracle | RL from scratch | Distillation |
| 灵感来源 | DeepMimic + DAgger + RMA | Multi-task RL | Masked modeling (MAE-style) |

MaskedMimic 的 idea 跟 HOVER 很像（mask 部分约束，让 policy 推断其余），但 MaskedMimic 在 graphics simulation 里，没有 real robot constraints。HOVER 是第一个把这套 idea 跑通到 real humanoid 上的。

MaskedMimic: https://research.nvidia.com/labs/toronto-ai/maskedmimic/
MHC: https://arxiv.org/abs/2408.07295

---

## 6. 我的 Intuition & Takeaway

### 6.1 为什么 distillation >> multi-task RL

这是这篇 paper 最深的 lesson。Multi-task RL 失败的原因不是"任务太难"，而是**reward landscape 互相冲突**。当 ExBody mode 想要 upper joint accuracy 而 H2O mode 想要 hand position accuracy，同一个 reward function 没法同时优化两者——梯度方向不一致。

Distillation 把这个问题 decompose：
- Stage 1（oracle）：只关心一个 task——perfect imitation。reward 单一、清晰。
- Stage 2（student）：只关心一个 task——模仿 oracle 的 action。loss 单一、清晰。

两个 stage 各自的 optimization 都很 clean，组合起来反而比直接 multi-task RL 更好。这是 "decompose then conquer" 在 RL 里的胜利。

### 6.2 Mask 作为 universal interface

$M_{\text{mode}} \odot$ 和 $M_{\text{sparsity}} \odot$ 的设计本质上把 command space 变成了一个 **"set-of-constraints"** 接口。每个 constraint 是独立的，policy 学会了"在给定 constraint 集合下做最合理的运动"。

这跟 LLM 里的 in-context learning 有结构相似性：LLM 见过各种 prompt pattern，能泛化到新组合；HOVER 见过各种 mask pattern，能泛化到新 mode 组合。**Mask 是 RL 版的 prompt**。

### 6.3 History length 25 的含义

Student stack 25 帧 history。在 50Hz 控制频率下这是 0.5 秒——大致是人类一个 step cycle 的尺度。这不是巧合：history 要足够长让 policy 推断"当前在做什么动作、处于什么 phase"，但太长会引入 noise。25 帧 ≈ 一个 gait cycle 是 legged locomotion community 的经验值。

### 6.4 局限性（paper 没明说但值得注意）

1. **Mode switch 是 manual 的**：paper 里 mode switch 是实验者手动触发的。Section V 提到 "future work will explore automated mode-switching module"——这意味着 HOVER 解决了 "how to execute a mode"，但没解决 "when to switch mode"。后者是个 harder 的 meta-control 问题。

2. **只测了 standing + locomotion**：real-world 评估没有 complex manipulation（比如 pick-and-place、door opening）。Tabletop manipulation 的效果如何还是 open question。

3. **Oracle 的 ceiling**：student 的上限是 oracle。如果 oracle 在某些 extreme motion（跳跃、翻滚）上不行，student 也不行。Paper 里 oracle survive rate 99.3%，已经很高，但 AMASS 里有些 action 可能被 retargeting 剔除了。

4. **19-DOF 限制**：H1 只有 19 个 DOF，upper body expressiveness 有限。如果换到 28-DOF 或更多 finger DOF 的 robot，command space 会爆炸式增长，mask 的 Bernoulli(0.5) 采样可能覆盖不全——可能需要更聪明的 mask scheduling。

### 6.5 更广的 implication

HOVER 的范式（oracle with privileged state → distill to student with realistic observation + random mask）可以推广到很多 robotics 问题：

- **Mobile manipulation**：oracle 看到全局物体位置，student 只看 RGB，mask 决定抓哪个物体
- **Multi-robot swarm**：oracle 看到所有 robot 状态，student 只看邻居，mask 决定跟谁协作
- **Autonomous driving**：oracle 看到全场景 ground truth，student 只看 sensor，mask 决定关注哪个 agent

本质上是 **"privileged teacher + masked student"** 这个 meta-pattern，HOVER 是它在 whole-body humanoid 上的 instantiation。

---

## 7. 总结

HOVER 的贡献可以归纳成一句话：**它把 humanoid control 的 "interface fragmentation" 问题，用 "motion imitation as universal abstraction + masked distillation" 优雅地解决了**。

工程上：一个 policy 顶过去 4-5 个，还能在线切换。
科学上：揭示了 multi-mode training 作为 regularization 的力量，以及 distillation 在解决 reward conflict 上的优越性。

接下来值得 follow 的方向：
1. Automated mode switching（paper 自己提到的 future work）
2. 把 mask 扩展到 time-varying（episode 内切换，模拟更复杂的 task interleaving）
3. 跟 vision-language policy 结合——让 LLM 决定 mask，HOVER 执行
4. 扩展到更高 DOF robot（Fourier GR1, Apptronik Apollo）

Reference links:
- HOVER arXiv: https://arxiv.org/abs/2410.21229
- Project page: https://hover-humanoid.github.io/
- OmniH2O (predecessor): https://omni.human2humanoid.com/
- HumanPlus: https://humanplus.github.io/
- ExBody: https://expressive-humanoid.github.io/
- Unitree H1: https://www.unitree.com/h1/
- AMASS dataset: https://amass.is.tue.mpg.de/
- IsaacGym: https://developer.nvidia.com/isaac-gym
- PPO paper: https://arxiv.org/abs/1707.06347
- DAgger paper: https://arxiv.org/abs/1011.0686
- DeepMimic: https://xbpeng.github.io/projects/DeepMimic/
- MaskedMimic: https://research.nvidia.com/labs/toronto-ai/maskedmimic/
