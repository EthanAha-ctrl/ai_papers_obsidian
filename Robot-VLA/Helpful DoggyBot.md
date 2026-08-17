---
source_pdf: Helpful DoggyBot.pdf
paper_sha256: 0aee9d7061e61629d30da12dcdf8b77a26ce8dd90fc3f9e83529abd57d884534
processed_at: '2026-08-04T23:39:11-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 Helpful DoggyBot

## 一句话总结

Stanford和UCSD的一帮人让一只机器狗学会了听人话，比如你说"去床上把那个毛绒玩具叼过来"，它就能真的爬上床、咬住玩具、再爬下来给你。

---

## 为什么要做这个事？

想象你家养了只狗，你能使唤它帮你拿东西。但现在的quadruped robot有几个尴尬问题：

1. **没手**：背上装个机械臂太重，狗就不灵活了
2. **不认识东西**：在simulation里训练的controller根本不知道"水瓶"和"网球"有什么区别
3. **上不去床**：indoor环境里到处是bed、sofa这种40-50cm高的东西，普通locomotion policy爬不上去

这篇paper的insight其实很简单：**把问题拆成两层，各管各的**。

---

## 怎么拆的？

### 底层：让机器狗学会"跑酷"

用RL在Isaac Gym里训练一个low-level controller，给它三样东西：
- **egocentric depth**（胸前depth camera看到的深度图）
- **proprioception**（关节角度、速度这些自身状态）
- **三个command**：前进速度$v_{cmd}$、转弯角速度$\omega_{cmd}$、身体俯仰角$p_{cmd}$

然后让它学两件事：
1. **能爬**：stairs、bed、sofa这种high obstacle都能翻过去
2. **能歪**：身体能前后tilt 30度，这样gripper能够到不同高度的东西

训练用了two-phase的套路：
- **Phase 1**：用privileged信息（scandots，就是terrain高度的直接采样），让policy先专心学dynamics，不用管perception
- **Phase 2**：用egocentric depth去distill，学一个CNN+GRU的encoder，从depth history推断terrain信息，替换掉scandots

这个套路的intuition：**别让一个network同时学两件难事**。先用perfect信息学好怎么动腿，再学怎么从camera推断terrain。

训练环境也讲究——6144个并行env，stairs高度从0到65cm随机生成，加了domain randomization（friction、mass、motor strength等），还做了curriculum learning。

结果在sim里climb up成功率96%，几乎和Oracle（用privileged信息的）一样好。

### 高层：让机器狗听懂人话

底层controller只知道"往前进、转弯、tilt身体"，不知道什么是"水瓶"也不知道它在哪。高层用VLM来解决这个问题。

**感知pipeline**长这样：
1. 天花板上挂个fisheye RGB camera（这是个assumption，后面说limitation）
2. 用**Florence-2**做open-vocabulary detection——你说"stuff toy"，它就能在画面里框出来
3. 用**SAM2**做segmentation + tracking，10Hz实时跟踪object和robot位置
4. 根据robot和object的相对位置，用simple P-controller算出velocity command往那走

当robot离object大概1米时，切换到**egocentric grasping**：
1. 胸前的RGB+depth camera接管
2. **GroundingDINO**（0.2Hz）做detection
3. **MobileSAM**（0.2Hz）做segmentation  
4. **Cutie**（10Hz）做high-frequency tracking

为什么要用三个model这么麻烦？因为SAM2太重了跑在onboard Jetson上太慢，只好用lightweight的替代品+高频tracker来补上。

从mask里extract object的$(x,y,z)$，然后：
- $x$控制前进速度
- $y$控制转弯
- $z$控制pitch（身体tilt）

当$(x,y,z)$都到位了，触发gripper闭合。

---

## 硬件上的clever设计

**Gripper**：没有用dexterous hand或multi-finger，而是设计了一个**1-DoF的Finray gripper**，装在robot的**底部前方**，用Dynamixel servo通过slider-crank驱动。

Finray effect是个很巧的物理现象：当finger末端被push时，由于柔性结构geometry，finger会反向curve，形成adaptive grasp。这样不管object是圆的（ball）、长的（bottle）、软的（stuffed toy），一口咬住就行。

这个gripper就是dog的"嘴"。

---

## 实验结果怎么样？

### Simulation

对比了5个baseline，在Climb Up、Climb Down、Walk with ±30° pitch这几个task上：
- **Blind**（没vision）和**No GRU**（没memory）：flat地面walk没问题，climbing全军覆没
- **No Distill**（直接学vision policy）：全挂，说明直接从vision学太hard
- **No Waypoint**（没有waypoint引导）：climbing只有14%，因为robot学到的是"绕过去"而不是"爬过去"——reward hacking
- **Ours**：96% climb up，90% climb down，几乎和Oracle的98%一样

### Real World

在两个unseen房间测了三个task，每个task 10次trial：

| | Bed+Toy | Sofa+Bottle | Ground+Ball |
|---|---|---|---|
| Go2 Default | 0% | 0% | 80% |
| No Tracking | 0% | 0% | 0% |
| **Ours** | **60%** | **60%** | **70%** |
| Teleop（人类遥控） | 80% | 70% | 80% |

几个有意思的观察：
- 比Go2 default和open-loop baseline碾压式领先
- 比human teleop只差20%左右——而human teleop第一次也只80%成功
- **时间上反而比human快**——Ground+Ball task，Ours用23秒，human用38秒

---

## 这个approach为什么能work？

核心是**modularity**。

你让一个network同时学"怎么动腿爬床"和"什么是水瓶"，两头都做不好。把它们拆开：
- Locomotion是个well-defined problem，sim能精确模拟，RL能解决
- Semantic understanding是个ill-posed问题，但internet-scale pre-trained VLM已经学得够好了

中间用$v_{cmd}, \omega_{cmd}, p_{cmd}$这三个command当interface，简单clean。

这也是为什么能**zero-shot**——simulation里没见过的房间和物体，VLM见过类似的，就能generalize。

---

## 有什么limitation？

1. **天花板camera不现实**：真实deployment不可能在天花板装camera。需要纯onboard的global localization方案
2. **1-DoF gripper太弱**：只能"咬"，不能做precise manipulation
3. **60%成功率还不够**：离production-ready还远，需要error analysis看哪一步fail最多
4. **只测了2个房间**：true generalization需要更多diverse environments
5. **static environment**：没有moving obstacles和人

---

## 一句话的takeaway

这篇paper告诉我们：**把model-based reasoning（VLM）和model-free motor skill（RL policy）用简单interface连起来，就能做出有意思的zero-shot mobile manipulation**。不需要在sim里render semantic content，也不需要让VLM输出motor torque，各做各擅长的就行。

方向是对的，limitation也是real的，但作为一个"证明concept能work"的paper，它挺solid。

---

# Helpful DoggyBot: Open-World Object Fetching using Legged Robots and VLMs

## Paper Overview

这篇paper来自 Stanford (Qi Wu, Zipeng Fu, Chelsea Finn) 和 UC San Diego (Xuxin Cheng, Xiaolong Wang)，核心目标是让 quadrupedal robot 在unseen indoor environments中zero-shot完成open-vocabulary object fetching任务，比如爬上queen-sized bed去fetch一个stuffed toy。

项目主页：https://helpful-doggybot.github.io/

---

## 1. Motivation与Problem Setting

Quadrupedal locomotion领域已经被learning-based methods dominated（如 [RMA](https://arxiv.org/abs/2107.04034), [ETH's Science Robotics paper](https://www.science.org/doi/10.1126/scirobotics.abc5986)），但要让quadruped在indoor environments中真正"helpful"，存在三个核心gap：

1. **Manipulation capability gap**：传统做法是在back上mount一个robotic arm（如 [ALMA](https://ieeexplore.ieee.org/document/8793744)），但arm的重量和复杂度会损害agility。
2. **Semantic gap between sim and real**：sim里render不出real-world indoor scenes的丰富语义，导致controller无法处理diverse household objects。
3. **Traversability/Reachability gap**：indoor空间cluttered，需要climb up beds/sofas（40-50cm高度），同时whole-body tilting来扩展gripper的workspace。

这篇paper的核心intuition是：把这些问题分层处理——low-level的agile locomotion和whole-body control交给simulation-trained RL policy，high-level的semantic understanding和command generation交给pre-trained VLMs，两者通过velocity/pitch/grasp commands这个interface耦合。

---

## 2. Hardware Setup

### Robot Platform
- **Unitree Go2**：12-DoF quadruped（每条腿3个关节：hip abduction/adduction, hip flexion/extension, knee）
- Onboard compute: **Nvidia Jetson Orin**
- Onboard battery powers both robot和gripper

### Custom 1-DoF Gripper
这是一个非常clever的设计选择。他们没有用dexterous hand或multi-finger gripper，而是设计了一个**Finray effect gripper** mounted on the **bottom front** of the robot，actuated by：
- **Dynamixel XM430-W350-T servo motor**（一个相当strong且precise的servo）
- 通过**slider-crank mechanism**实现fast closing

Finray effect的核心intuition：当finger末端被pushed时，由于柔性结构的special geometry，finger会反而向被pushed的方向curve，形成adaptive grasping。这种design可以pick up各种shape的everyday objects（tennis ball, water bottle, stuffed toy），不需要复杂的perception或grasp planning。

这个gripper充当robot的"mouth"——robot通过"biting"来pick up和firmly hold物体。

### Sensing
- **Egocentric RealSense D435**：mounted on top front，向下倾斜30度，提供depth和RGB。这个angle choice很重要——既要看到ground近处用于locomotion，又要能look up一点看到bed/sofa上的物体。
- **Ceiling-mounted fisheye RGB camera**：提供top-down global view，用于navigation阶段同时track robot和target object。这是一个strong assumption（后面会提到limitation）。
- Onboard Jetson跑learned low-level controller + VLM pipeline（接近物体时）

---

## 3. Learning a General Whole-Body Controller

这是paper的核心technical contribution。他们采用**two-phase training process**：

### Phase 1: Privileged Training with PPO

在Isaac Gym Preview 4中，6144个robots在400个terrains上并行训练。Observation用的是**scandots**——robot周围terrain的height samples。这是privileged information，只能在sim中获得，但能让policy高效学习。

**Whole-body objective**（公式1）：

$$r_{wb} = \exp(-3 \cdot |p_{cmd} - p|)$$

变量解释：
- $p_{cmd}$：commanded pitch angle，uniformly sampled from $[-30°, 30°]$。这个range允许robot向前或向后tilt body，扩展gripper的reach。
- $p$：robot body的actual pitch angle
- 系数3控制reward的sharpness——差距越大，reward衰减越快
- $\exp$ form确保reward始终在$(0, 1]$，且smooth

注意：当robot encountering obstacles时，这个reward被removed，避免pitch tracking和climbing的conflicting objectives。

**Agile locomotion objective**（公式2）：

$$r_{tracking} = \min(\langle v, \hat{d}_{wp}\rangle, v_{cmd}) / v_{cmd}$$

变量解释：
- $v \in \mathbb{R}^2$：robot在world frame下的current velocity（2D，xy平面）
- $v_{cmd} \in \mathbb{R}$：linear velocity command，sampled from $[0, 1]$ m/s
- $\hat{d}_{wp}$：指向next waypoint的unit vector
- $\langle v, \hat{d}_{wp}\rangle$：velocity在waypoint direction上的projection
- $\min(\cdot, v_{cmd})$：cap at commanded velocity，防止robot超速跑得到reward
- 除以$v_{cmd}$：normalize到$[0, 1]$

Waypoint direction计算（公式3）：

$$\hat{d}_{wp} = \frac{x_{wp} - x}{|x_{wp} - x|}$$

- $x_{wp}$：next waypoint的location in world frame
- $x$：robot current position in world frame

然后$\hat{d}_{wp}$被转换为angular velocity command $\omega_{cmd}$作为policy input，计算的是robot current direction和$\hat{d}_{wp}$之间的angular difference。这个设计remove了policy对global information的dependency。

**关键intuition**：velocity tracking在**world frame**计算，防止robot学到"绕过obstacle"这种unintended behavior。如果用robot frame，robot可以通过turn来"假装"forward velocity高。

### Phase 2: Policy Distillation with Egocentric Depth

Phase 1的policy用了scandots（privileged），real world没有scandots。Phase 2通过**Regularized Online Adaptation (ROA)**（来自[Deep Whole-Body Control](https://arxiv.org/abs/2210.10045)）训练一个online estimator，从**history of depth images**恢复environment information。

Architecture：
- **CNN**处理单帧depth image（spatial features）
- **GRU**处理temporal sequence（memory，capture dynamics和terrain context）
- Output替换Phase 1 policy中的scandots input

**关键设计差异**：他们没有像[Extreme Parkour](https://arxiv.org/abs/2309.14341)那样做dual distillation（同时distill heading command和exteroception），而是用VLM来specify heading direction。这避免了dual distillation的out-of-distribution问题——VLM可以handle arbitrary environments，而learned heading policy可能会在unseen场景fail。

### Simulation Environment和Curriculum

- 6144 envs for Phase 1, 384 envs for Phase 2 distillation
- 400 terrains, 10 difficulty levels
- Stair height: $[0, 0.65]$ m（递进curriculum）
- Stairs per env: $[0, 6]$
- Stair width: $[0.8, 3]$ m
- Stair length: $[1.5, 2]$ m
- Friction: $[0.2, 2]$

Curriculum update criteria：基于每个episode robot完成terrain的比例。Up threshold $0.8 \times$ total length, down threshold $0.5 \times$ total length。

### Domain Randomization

| Parameter | Value |
|-----------|-------|
| Push interval | 8s |
| Max push vel xy/z | 0.5 m/s |
| Added mass | $[0, 3]$ kg |
| Added COM offset | $[-0.2, 0.2]$ m |
| Motor strength | $[0.8, 1.2]$ |
| Action delay | $[0, 0.02]$ s |
| Vision delay | 0.1s |
| Vision position rand | 0.005m |
| Vision angle rand | $[24°, 34°]$ |

Vision angle rand的范围对应于camera mount angle的uncertainty，这对sim2real很重要。

### Reward Function细节

除了主要的tracking rewards，还有大量auxiliary rewards（Table V）。几个关键的：

- **Tracking yaw vel**: $\exp(-|\omega_z - \omega_{cmd}|)$，scale 1.0
- **Tracking pitch**: $\exp(-3|p_{cmd} - p|)$，scale 1.5
- **Lin vel z**（walking时）: 惩罚垂直方向速度，scale -9.0（相当大）
- **Ang vel xy**: $\sum \omega_{xy}^2$，惩罚roll/pitch angular velocity，scale -0.05
- **DOF acc**: $\sum((\dot{q}_{t+1} - \dot{q}_t)/\Delta t)^2$，scale -2.5e-7，penalize jerky motions
- **Collision**: $\sum \mathbf{1}(\|f_{contact}\| > 0.1)$，scale -5.0
- **Action rate**: $\|\|a_{t+1} - a_t\|\|$，scale -0.1
- **Delta torques**: $\sum(\tau_{t+1} - \tau_t)^2$，scale -1.0e-7
- **Torques**: $\sum \tau^2$，scale -1e-5
- **Hip pos**: $\sum(q_{hip} - q_{hip,default})^2$，scale -1，鼓励legs回到default position
- **DOF error**: $\sum(q - q_{default})^2$，scale -0.2
- **Feet stumble**: $\mathbf{1}(\|f_{contact,xy}\| > 4 \cdot \|f_{contact,z}\|)$，scale -5，penalize lateral contacts（indicating stumbling）
- **Feet edge**: penalize feet at edge of stairs，scale -1
- **Feet drag**: $\sum(\mathbf{1}(contact) \cdot \|v_{xy}^{feet}\|)$，scale -0.1
- **Energy**: $\|\tau \cdot \dot{q}\|$，scale -1e-3

这些rewards的设计哲学：主要task rewards（tracking）给positive reward，所有"don't do bad things"的rewards给negative。Scale的选择体现了priorities——lin vel z的-9.0特别大，说明他们非常想prevent robot bouncing up and down。

---

## 4. Zero-Shot Deployment using VLMs

这是paper的另一半创新点。整个pipeline分三个阶段：

### Stage 1: Open-Vocabulary Detection, Segmentation, Tracking

**Initial Detection**: 用[Florence-2](https://arxiv.org/abs/2311.16262)做open-vocabulary object detection。Florence-2是Microsoft的unified vision foundation model，可以用natural language描述来detect objects，包括robot itself和target object。

**Segmentation**: 用[SAM2](https://arxiv.org/abs/2408.00714)（Segment Anything Model 2，Meta）生成precise object masks。Florence-2的bounding box作为SAM2的prompt。

**Tracking**: SAM2在10Hz进行object tracking，continuously update object position。

### Stage 2: Navigation

用ceiling-mounted fisheye camera提供global view。这个视角可以同时看到robot和target object，simplify了planning——不需要做SLAM或localization。

Navigation策略很simple：
- Target object position作为single waypoint
- Linear velocity: constant 0.8 m/s towards waypoint
- Angular velocity: proportional controller with $K_p = 0.5$
  $$\omega = K_p \cdot (\theta_{target} - \theta_{robot})$$
  其中$\theta_{target}$是robot到waypoint的direction angle，$\theta_{robot}$是robot当前heading
- Pitch command: 0（保持body level）
- Transition to grasping mode when robot within ~1m of target

**关键assumption**：low-level controller可以traverse most indoor obstacles（beds, sofas），所以不需要obstacle avoidance。这是一个strong assumption但justified by他们的agile controller。

### Stage 3: Grasping

当robot接近target时，切换到egocentric perception。因为SAM2 computationally expensive，onboard inference太慢，他们用一个multi-stage pipeline：

1. **GroundingDINO**（[paper](https://arxiv.org/abs/2303.05499)）：object detection at 0.2Hz
2. **MobileSAM**（[paper](https://arxiv.org/abs/2306.14289)）：lightweight segmentation on RGBD at 0.2Hz
3. **Cutie**（[paper](https://arxiv.org/abs/2303.11342)）：high-frequency tracking at 10Hz

这种architecture的intuition：slow detection提供accurate localization，fast tracking在detection之间interpolate。

从tracked mask中extract object center的$(x, y, z)$ in robot's local frame，然后用proportional controllers生成commands：
- Linear velocity $\propto x$ with $K_p = 0.5$
- Angular velocity $\propto y$ with $K_p = 0.5$
- Pitch $\propto z$ with $K_p = 1$

当所有coordinates都在small threshold内时，trigger grasping action。

---

## 5. Experiments

### Simulation Experiments

**Baselines**:
1. **Blind**: only proprioception, no depth
2. **No GRU**: MLP instead of GRU, no temporal memory
3. **No Distill**: train deployable policy directly with PPO, skip distillation
4. **No Waypoint**: remove waypoint-guided agile locomotion, directly track sampled vel commands
5. **Oracle (Phase 1)**: privileged policy with scandots

**Tasks**: Climb Up, Climb Down, Walk +30° pitch, Walk -30° pitch

**Results** (Table I):

| Method | Climb Up | Climb Down | Walk +30° | Walk -30° | Avg Dist (Up) | Avg Dist (Down) |
|--------|----------|------------|-----------|-----------|---------------|-----------------|
| Blind | 0% | 0% | 100% | 100% | 11% | 10% |
| No GRU | 0% | 0% | 100% | 100% | 12% | 13% |
| No Distill | 0% | 0% | 0% | 0% | 0% | 0% |
| No Waypoint | 14% | 12% | 90% | 100% | 10% | 13% |
| **Ours** | **96%** | **90%** | **100%** | **100%** | **92%** | **84%** |
| Oracle | 98% | 96% | 100% | 100% | 95% | 92% |

**Key takeaways**：
1. Blind和No GRU在simple walking上100% success，但在climbing上complete failure。这说明vision和temporal memory对complex locomotion essential。
2. No Distill完全fail，说明direct learning from vision太hard，two-phase distillation crucial。
3. No Waypoint在climbing上poor performance（14%, 12%），说明waypoint guidance prevent robot从learning "绕过obstacle"的local optima。
4. Ours非常接近Oracle，distillation几乎无损——effectively 96% vs 98% on Climb Up。

### Real-World Experiments

**Tasks**:
1. **Bed + Toy**: climb 40cm queen-sized bed, fetch stuffed toy placed randomly on 1m×1m region
2. **Sofa + Bottle**: climb 44cm sofa, fetch water bottle on 0.2m×1m region
3. **Ground + Ball**: fetch ball on 3m×3m flat region
4. **Bed + Sofa + Toy**: more complex multi-stage task

**Baselines**:
1. **Go2 Default**: Unitree's built-in controller, no exteroception
2. **No Tracking**: open-loop commands from initial pose detection
3. **Teleop**: expert human operator with remote controller

**Results** (Table II):

| Method | Bed+Toy (Nav+Climb) | Pick Up | Climb Down | Total | Sofa+Bottle Total | Ground+Ball Total | Avg Time Toy | Avg Time Bottle |
|--------|---------------------|---------|-------------|-------|--------------------|--------------------|--------------|-----------------|
| Go2 Default | 0% | 0% | 0% | 0% | 0% | 80% | - | - |
| No Tracking | 60% | 0% | 0% | 0% | 40% | 0% | - | 1s* |
| **Ours** | 90% | 78% | 86% | **60%** | **60%** | **70%** | **50s** | **23s** |
| Teleop | 90% | 89% | 100% | **80%** | 70% | 80% | 58s | 38s |

**Key observations**：
1. Ours在Bed+Toy上60% first-attempt success rate，significantly outperforming Go2 Default和No Tracking（both 0%）。
2. Teleop只有80% first-attempt success（即使是expert human），说明这个task本身有难度。
3. Ours在time上actually outperforms teleop——Ground+Ball task上23s vs 38s。这说明VLM的reactive control比human reflexes更快。
4. Sofa+Bottle上demonstrate了robustness on soft deformable surfaces——这是一个non-trivial sim2real challenge。

---

## 6. Architecture Diagram Analysis

虽然paper里Figure 3是system overview，让我verbalize一下整个data flow：

```
[Ceiling Fisheye RGB] 
    ↓
[Florence-2: Open-Vocab Detection]
    ↓
[SAM2: Segmentation + Tracking @ 10Hz]
    ↓
[Robot + Object Position Estimation]
    ↓
[Navigation: P-controller → v_cmd, ω_cmd, pitch=0]
    ↓ (when within 1m of target)
[Egocentric RGB + Depth from D435]
    ↓
[GroundingDINO @ 0.2Hz → MobileSAM @ 0.2Hz → Cutie @ 10Hz]
    ↓
[Object (x,y,z) in robot frame]
    ↓
[Grasp P-controllers: v∝x, ω∝y, pitch∝z]
    ↓
[Low-level Controller: CNN+GRU on depth history + proprioception]
    ↓
[Joint angle commands @ 50Hz → Go2 motors]
    ↓
[Trigger gripper close when (x,y,z) within threshold]
```

整个system是一个**hierarchical structure**：
- VLM作为high-level "brain"：semantic understanding, planning, reactive command generation
- Learned controller作为low-level "spinal cord"：agile locomotion, whole-body control
- Interface是$(v_{cmd}, \omega_{cmd}, p_{cmd})$这个3-DoF command space

这种separation of concerns是关键——VLM不需要知道leg kinematics，controller不需要知道什么是"tennis ball"。

---

## 7. Limitations和Future Directions

Paper自己acknowledge的limitations：

1. **Gripper dexterity有限**：1-DoF Finray gripper只能做simple "biting" grasp，无法做precise manipulation（如按钮、拧瓶盖）。
2. **依赖ceiling-mounted camera**：navigation需要global view，这在real home deployment中impractical。需要onboard sensing only的navigation strategy。
3. **Perception occlusion**：egocentric camera在climbing时可能被occluded，或者object被robot自身遮挡。
4. **No dynamic environments**：假设environment是static的，没有moving obstacles或humans。

Future directions提到：
- Enhanced manipulation capabilities
- Onboard-only navigation
- Cheerful pet behaviors（reference to [Playful DoggyBot](https://arxiv.org/abs/2407.09675)）
- Multi-task sequences
- Online learning和human feedback
- Societal implications

---

## 8. Critical Analysis和Intuition Building

### 为什么这个approach work？

核心intuition是**modularity**。Whole-body locomotion control是一个well-defined MDP，可以在sim中solve。Semantic understanding是一个open-ended problem，pre-trained VLMs已经harness了internet-scale data。把两者通过一个simple command interface连接，避免了在sim中render semantic content的impossible task，也避免了让VLM直接输出motor torques的ill-posed problem。

### 与相关工作对比

- vs [Extreme Parkour](https://arxiv.org/abs/2309.14341)：Extreme Parkour focus on extreme agility但no manipulation，no semantic understanding。DoggyBot adds gripper和VLM layer。
- vs [Robot Parkour Learning](https://arxiv.org/abs/2309.14341)：Similar two-phase training但DoggyBot针对indoor environments而非outdoor parkour。
- vs [Deep Whole-Body Control](https://arxiv.org/abs/2210.10045)（Fu et al., 2022）：使用了类似的ROA distillation，但DoggyBot是quadruped而非humanoid，且加了VLM layer。
- vs [VoxPoser](https://arxiv.org/abs/2307.05973), [Code as Policies](https://arxiv.org/abs/2209.07726)：这些VLM-for-manipulation works focus on table-top static manipulation，DoggyBot是mobile manipulation on legged platform。
- vs [LM-Nav](https://arxiv.org/abs/2207.11944)：wheeled robot navigation with LLMs，DoggyBot是legged且加了manipulation。

### Potential concerns

1. **Ceiling camera assumption**：这是biggest practical limitation。一个deployment-ready system需要onboard sensors做global localization，可能需要visual-inertial odometry（如[Cerberus](https://arxiv.org/abs/2311.10923)）或semantic SLAM。

2. **60% success rate**：虽然在zero-shot setting下impressive，但far from production-ready。Error analysis会很有价值——是navigation fail？climbing fail？grasping fail？

3. **VLM inference latency**：Florence-2 + SAM2在workstation上跑，但real-time性如何？paper没有详细讨论latency budget。

4. **Generalization scope**：只test了2个unseen environments。True generalization需要更多diverse environments。

5. **Object grasping robustness**：1-DoF gripper对object shape/size有限制。Stuffed toy和water bottle形状differences很大，更challenging objects（如keys, credit cards）可能fail。

### Intuition for why two-phase distillation works so well

Phase 1用scandots（perfect terrain knowledge）让policy focus on learning **dynamics**——how to coordinate legs for climbing, tilting, etc.，不需要同时learn perception。

Phase 2只需要learn **perception**——how to extract terrain information from depth history。这个subproblem更structured，因为terrain information是lower-dimensional than raw depth images。

这种"decoupling dynamics learning from perception learning"的paradigm在locomotion learning中被反复validate（[Learning to Walk in Minutes](https://arxiv.org/abs/2109.12098), [RMA](https://arxiv.org/abs/2107.04034), etc.）。

### Intuition for why waypoint guidance is critical

如果只reward local velocity tracking，robot会发现"绕过obstacle"比"climb over"更容易达到high local velocity。这是经典的**reward hacking**现象。

Waypoint在world frame中specify了intended direction，force robot to make progress towards goal而不是arbitrary direction。这把"探索空间"从"任意movement"压缩到"forward progress"。

### Intuition for VLM as high-level planner

VLM的优势是**open-vocabulary**和**zero-shot generalization**。如果用trained perception model，每加一个新object category需要collect data和retrain。VLM从internet image-text pairs中已经learned了rich visual concepts，可以直接transfer到robot场景。

VLM的劣势是**latency和cost**。Florence-2 + SAM2 pipeline不可能跑在onboard Jetson上real-time，需要workstation。这是为什么他们用hierarchical approach——VLM做sparse high-level decisions，learned policy做dense low-level control。

---

## 9. Implementation Details Worth Noting

- **Depth processing**: hole-filling filters, spatial filters, temporal filters, resizing, normalization——mirror sim的process
- **Depth encoder @ 10Hz** via UDP to main process
- **Main policy @ 50Hz** on Jetson
- **Proprioception @ 500Hz** via Cyclone DDS
- **Joint commands** via ROS 2 to Unitree low-level controller，which computes motor torques with internal PD

这个frequency hierarchy很interesting：proprioception fastest (500Hz), policy medium (50Hz), vision slowest (10Hz)。这反映了signal的timescale——body dynamics快，terrain perception慢。

- **Oracle training**: 20k iterations, 10 hours on RTX 4090
- **Distilled policy**: 5k iterations, 6 hours

这relatively fast的training time是Isaac Gym massively parallel simulation的benefit。

---

## 10. Open Questions和Future Research Directions

1. **Onboard global localization**：能否用visual-inertial-leg odometry（[Cerberus](https://arxiv.org/abs/2311.10923)）+ onboard semantic segmentation来replace ceiling camera？

2. **Closed-loop VLM**：能否让VLM observe robot's behavior并correct mistakes？类似[PIVOT](https://arxiv.org/abs/2402.07872)的iterative visual prompting。

3. **Learning from VLM demonstrations**：VLM生成的trajectories能否作为imitation learning的supervision来train better policies？

4. **Multi-object tasks**：能否extend到"fetch me the red ball next to the book on the bed"？需要relational reasoning。

5. **Dynamic environments**：moving obstacles, humans walking around——需要reactive planning。

6. **Force-sensitive grasping**：1-DoF gripper没有force feedback，可能damage fragile objects。能否add tactile sensing？

7. **Energy efficiency**：climbing和tilting很energy-intensive。能否optimize for battery life？

8. **Failure recovery**：当robot fail to climb或drop object，能否autonomously recover？

---

## Reference Links

- Project page: https://helpful-doggybot.github.io/
- Florence-2: https://arxiv.org/abs/2311.16262
- SAM2: https://arxiv.org/abs/2408.00714
- GroundingDINO: https://arxiv.org/abs/2303.05499
- MobileSAM: https://arxiv.org/abs/2306.14289
- Cutie: https://arxiv.org/abs/2303.11342
- Extreme Parkour: https://arxiv.org/abs/2309.14341
- Robot Parkour Learning: https://arxiv.org/abs/2309.14341
- Deep Whole-Body Control: https://arxiv.org/abs/2210.10045
- RMA: https://arxiv.org/abs/2107.04034
- PPO: https://arxiv.org/abs/1707.06347
- HumanPlus: https://arxiv.org/abs/2406.10454
- Open-TeleVision: https://arxiv.org/abs/2407.01512
- OmniH2O: https://arxiv.org/abs/2406.08858
- Playful DoggyBot: https://arxiv.org/abs/2407.09675
- Cerberus (VILO): https://arxiv.org/abs/2311.10923
- Code as Policies: https://arxiv.org/abs/2209.07726
- VoxPoser: https://arxiv.org/abs/2307.05973
- LM-Nav: https://arxiv.org/abs/2207.11944
- RT-2: https://arxiv.org/abs/2307.15818
- OpenVLA: https://arxiv.org/abs/2406.09246
- SayTap: https://arxiv.org/abs/2306.07580
- SpatialVLM: https://arxiv.org/abs/2401.14196
- RoboPoint: https://arxiv.org/abs/2406.10721
- PIVOT: https://arxiv.org/abs/2402.07872

---

## Summary

Helpful DoggyBot是一个elegant system paper，core insight是通过**modular design**来tackle legged mobile manipulation的complexity——learned controller处理agile locomotion，VLM处理semantic understanding，通过简单的command interface连接。Two-phase distillation使sim2real transfer几乎lossless，VLM pipeline使zero-shot generalization到unseen environments成为可能。虽然limitations（ceiling camera, 1-DoF gripper, 60% success rate）prevent immediate deployment，但这个work为future home robots指明了一个promising direction：**combining internet-scale pre-trained models with simulation-trained motor skills**。
