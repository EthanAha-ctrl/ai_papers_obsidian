---
source_pdf: Reconciling Reality through Simulation.pdf
paper_sha256: db7e57f6b298b535e912df1098c33669ffd4366935ea60c422cd161b1e0797ce
processed_at: '2026-08-11T21:48:01-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# RialTo 用人话版

## 一句话故事

拿iPhone扫一下你的厨房，15分钟自动建好sim；你在real world演示15次；robot自己练2天后，碗被挪了、有东西挡着、甚至有人故意捣乱，它还能稳稳把碗放回架子上。

## 为什么这事儿难

想deploy一个robot到家里，你得让它robust——碗位置变了、灯暗了、有人走过、碗在夹爪里slip了，都得能handle。

三条路都有问题：

**Imitation Learning (IL)**
- 演示15次？BC在pose randomization下10%成功，有distractor直接0%
- 演示50次？要2小时人工，也就40%
- 本质问题：BC是open-loop mimic，没见过的情况就崩

**RL in Real**
- Slow、unsafe、reset难、reward难定义
- 摔碗摔盘子怎么训

**RL in Sim**
- 速度快、安全、能parallel
- 但造scene太累——一个带抽屉的柜子要engineer手工写URDF，articulated objects更麻烦

## RialTo的insight

承认三个worlds各有优劣，把它们串起来：

| 来源 | 给什么 | 缺什么 |
|------|--------|--------|
| Real demos | physically correct behavior, real visual | 没state, 没robustness |
| Sim scene | parallel RL, privileged state | 视觉假, physics假 |
| Real-to-sim scan | 几何+articulation | 还是要human切mesh |

核心：用real demos当distribution anchor，用sim当exploration sandbox，用co-training把两边的好处榨干。

## Pipeline 四步走

### Step 1: 扫一下厨房

用iPhone拍视频，选一个工具：
- **Polycam**: 大场景（厨房台面），用LiDAR
- **AR Code**: 单个物体360°扫，精度高
- **NeRFStudio**: 细结构（像dish rack的金属杆），10分钟训个NeRF

拿到raw mesh是一个整体，没法interactive。于是有GUI让你：
- 把抽屉从柜子上"切"下来
- 在抽屉上加joint
- 拖一拖位置

Non-expert用户平均14分40秒搞定一个articulated scene（user study验证）。

### Step 2: Inverse Distillation —— 最clever的一步

你有了15条real demos：$(o, a)$ pairs，$o$是point cloud，$a$是EE delta pose。但sim里RL想要的是带state的demos $(o, a, s)$，$s$是Lagrangian state（object pose, joint angle等）。Real world里拿不到$s$。

**Trick**: 在real demos上训一个IL policy $\pi_{\text{real}}(a|o)$，然后在sim里render point cloud喂给$\pi_{\text{real}}$执行。Sim里执行时，$s$是免费送的——因为sim本来就track着所有object state。

收集15条成功trajectory，自动带上privileged state。

这就是"inverse distillation"——从perception policy反推出state demos。一招解决information asymmetry。

### Step 3: Sim里RL fine-tune

用这15条带state的demos bootstrap PPO。Loss = PPO + value loss + BC loss：

$$\mathcal{L} = \underbrace{\alpha \cdot \mathcal{L}_{\text{PPO}}}_{\text{exploration}} + \underbrace{\beta \cdot \mathcal{L}_{\text{value}}}_{\text{critic}} + \underbrace{\gamma \cdot \mathcal{L}_{\text{BC}}}_{\text{don't go crazy}}$$

BC loss非常关键。Table III显示PPO from scratch在5个task里3个直接0%。剩下的也学到奇怪behavior——比如push toaster底部来开toaster，利用了sim里joint位置的一点小错误。这种policy在real里直接fail。

BC loss把policy拉回到human-demostrated的physically plausible region，同时PPO让它explore pose变化和disturbance下的recovery。

Reward超级simple：
```
success = toaster_joint > 0.65 && gripper_open
```
就这种sparse reward。BC让exploration tractable。

State space = 所有object pose + joint state + robot state拼起来
Action space = 14维discrete（6平移±3cm + 6旋转±0.2rad + 2夹爪开关）

### Step 4: Distill回perception + Co-train with Real

$\pi_{\text{sim}}^*(a|s)$用不了，real world没state。Distill成$\pi_{\text{real}}^*(a|o)$。

Loss = DAgger distillation + Real BC：

$$\mathcal{L} = \alpha \sum_{\text{sim rollouts}} \log \pi_\theta(\pi_{\text{teacher}}(s)|o) + \beta \sum_{\text{real demos}} \log \pi_\theta(a|o)$$

第一项：student模仿teacher在sim里给的action labels
第二项：student同时在real demos上做BC

为什么要第二项？因为sim point cloud和real depth camera point cloud分布不一样（噪声、遮挡模式），sim physics也和real有gap。Real demos 15条虽然少，但是**distribution anchor**——告诉policy "real world长这样"。

Dataset mixing也很精细：
- 15000条 full point cloud（从mesh直接采样，all faces visible）
- 5000条 sim camera视角渲染
- 2000条 sim camera + distractors
- 15条 real demos
四类各1/4采样。然后DAgger iteration再mix。

Point cloud encoder用[Convolutional Occupancy Networks](https://github.com/ltimoner/convolutional_occupancy_networks)：local point net → 3D U-Net → voxel grid → max+avg pool → 128维。再concat robot state (9维) → MLP(256,256) → 14维action distribution。

## 关键实验数字

### 主结果（Table I，Book on Shelf）

| Method | Pose Rand | Distractors | Disturbance |
|--------|----------|------------|-------------|
| BC 15 demos | 10% | 0% | 0% |
| BC 50 demos | 40% | 30% | 20% |
| **RialTo 15 demos** | **90%** | **70%** | **60%** |

时间账：
- BC 50 demos: 1h45min 人工
- RialTo: 30min demos + 15min GUI = 45min 人工

RialTo用1/2时间换2.5x success。

### 8 tasks平均（Figure 5）

- RialTo: 91% / 77% / 75% （pose / distractor / disturbance）
- BC: 25% / 11% / 5%

### RL from scratch vs from demos（Table III）

| Task | RL scratch | RL + 15 real demos | RL + 15 sim demos |
|------|-----------|-------------------|-------------------|
| Open toaster | 62% | 91% | 96% |
| Book on shelf | 0% | 90% | 89% |
| Plate on rack | 2% | 81% | 82% |
| Mug on shelf | 0% | 81% | 82% |
| Open drawer | 0% | 96% | 95% |

关键：real demos和sim demos作为bootstrap几乎一样好——证明inverse distillation真的work。From scratch会exploit sim artifacts，transfer不了。

### Co-training ablation（Figure 6）

Book on shelf with disturbance：
- 只用sim co-training: ~20%
- 加real co-training: ~60%（3x提升）

Plate on rack with disturbance：
- sim only: ~30%
- +real: ~60%（2x提升）

Sim-to-real gap小的task（drawer, mug）co-training提升不明显。

### Real-to-sim scene 必要性（Figure 7）

Train on target drawer reconstruction: 90%
Train on 4 Objaverse drawers (multi-task): 10%

**结论**: 为specific deployment建digital twin，比追求generalist多scene training有效得多——尤其在scene-level manipulation。

### RL from vision vs from state（Appendix Fig 14）

- State-based: 96% in 12h
- Vision-based: 1% in 35h

State-based快~1000x。这就是inverse distillation到state space的必要性。

## 为什么这个组合work

### Real demos的两个作用

1. **Bootstrap sim RL**：通过inverse distillation，变成state demos，引导PPO explore
2. **Distribution anchor**：在distillation时co-train，拉住policy不让它drift to sim-specific features

### Sim的两个作用

1. **Parallel RL substrate**：用state-based PPO快速explore robustness
2. **Privileged info source**：sim自动给state，不需要real state estimator

### BC loss的精妙

PPO from scratch会找sim的漏洞。BC loss相当于一个prior说"human这么做，你别太飘"。但BC又不能太强（weight=0.1），否则explore不动。

### Targeted > Diverse

不要追求"训练一个能在任何厨房工作的policy"。对specific user的specific厨房，targeted digital twin + 少量real demos = 最佳specialization。Generalist要海量data，且可能over-conservative。

## 类比

想象学开车：

- **BC**: 教练演示15次固定路线，你死记硬背。换条路就完蛋。
- **RL in real**: 在马路上随便撞随便试。危险且慢。
- **RL in sim from scratch**: 在VR里随便试，但VR的物理有点假，你学会了"用VR bug开车"。
- **RialTo**: 用无人机扫一条你家到公司的路，建VR路线。教练演示15次。你在VR里随便撞，但有个BC angel说"教练这么开你别差太远"。最后戴AR眼镜把reflex transfer到real车，同时还用教练的15次real demo告诉你"real world长这样"。

## 你能拿这个干什么

如果你想做robot learning research：

1. **Foundation model + RialTo specialization**: 把$\pi_{\text{real}}$的initialization从15 demos BC换成OpenVLA/RT-2这类pretrained model，inverse distillation收集的sim demos质量会高很多
2. **Deformable objects**: 把articulated rigid body scan换成diffable sim of deformables（参考[DiffCloud](https://arxiv.org/abs/2208.01176)）
3. **Auto system ID**: physics params用[ASID](https://arxiv.org/abs/2404.12308)自动infer，进一步缩小dynamics gap
4. **Language conditioning**: 一个scene多task，加language instruction减少per-task engineering
5. **RGB policy**: point cloud信息量有限。RGB能capture texture，但sim-to-real visual gap更大，需要更好domain adaptation

## 局限

- Depth sensor对thin/transparent/reflective物体不行
- 只能sim articulated rigid bodies，deformables不行
- Quasi-static tasks，dynamic tasks需要精确physics ID
- 2天/task，continual learning不可行
- Point cloud信息量有限，texture和fine visual cue丢失

## 个人观感

这篇paper我觉得最value的不是algorithm novelty（每个component单独看都不新：teacher-student distillation、PPO+BC、domain randomization都是已有technique），而是**system engineering的thoughtfulness**：

- 发现information asymmetry（real没state, sim有state）并用inverse distillation解决
- 发现RL from vision太慢，用state-based PPO + distillation绕过
- 发现sim policy会exploit sim artifacts，用BC loss预防
- 发现sim-to-real visual gap，用co-training with real demos弥补
- 发现generalist training太贵，用targeted digital twin specialization

每个design choice都有明确的problem it solves。这是好的system paper的标志——不是堆method，而是把简单method放在正确位置。

参考链接：
- [RialTo Project](https://real-to-sim-to-real.github.io/RialTo/)
- [NeRFStudio](https://github.com/nerfstudio-project/nerfstudio)
- [Convolutional Occupancy Networks](https://github.com/ltimoner/convolutional_occupancy_networks)
- [IsaacSim](https://developer.nvidia.com/isaac-sim)
- [Orbit](https://isaac-orbit.github.io/)
- [DiffCloud](https://arxiv.org/abs/2208.01176)
- [ASID](https://arxiv.org/abs/2404.12308)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [Visual Dexterity (Tao Chen)](https://www.science.org/doi/10.1126/scirobotics.adc9244)
- [RMA](https://arxiv.org/abs/2107.04034)
- [Objaverse](https://objaverse.allenai.org/)

---

# RialTo: Real-to-Sim-to-Real 深度技术解析

## 1. 核心问题与动机

这篇paper解决的是robotic manipulation中一个非常实际的问题：如何让imitation learning policy在real-world deployment时具备robustness，同时avoid prohibitive human supervision和unsafe real-world data collection。

传统的imitation learning存在几个根本性缺陷：
- **Distribution shift brittleness**: BC policy在object pose变化、visual distractors、physical disturbances下表现急剧下降
- **No recovery behavior**: policy无法从mistake中恢复，因为demonstrations里没有这些failure cases
- **Data scaling问题**: 要达到robustness需要大量demos（paper显示50 demos也只能达到40% success rate under pose randomization）

RL in real world则面临：unsafe、slow、reset problem、reward specification困难。Sim中的RL虽然fast且safe，但construct realistic simulation environment非常labor-intensive。

RialTo的核心insight：用real-world video快速construct digital twin simulation，然后用sim中的RL来robustify real-world IL policy。这是一个**complementary**的approach，同时改善IL和RL的缺陷。

Reference: [Project page](https://real-to-sim-to-real.github.io/RialTo/)

## 2. 系统架构总览

RialTo pipeline包含四个sequential steps：

### Step 1: Real-to-Sim Scene Construction
输入：real-world video（用iPhone扫描）
输出：USD/URDF scene $\mathcal{S} = \{\{\mathcal{G}_i\}_{i=1}^M, \mathcal{K}, \mathcal{P}\}$

其中：
- $\mathcal{G}_i$: 第$i$个mesh body的geometry
- $\mathcal{K}$: kinematic relations（joints定义）
- $\mathcal{P}$: physical parameters（mass, friction等）
- $M$: scene中separated bodies的总数

这里用了off-the-shelf 3D reconstruction tools：
- **Polycam**: 适合large scenes（kitchen），用iPhone LiDAR，输出GLTF → 转换为USD
- **AR Code**: 适合single object 360°扫描，精度更高，直接输出USD
- **NeRFStudio**: 用nerfacto模型，适合thin structures（如dish rack的金属杆），训练~10分钟，用Poisson Surface Reconstruction提取mesh

raw mesh $G$是globally-unified geometry，需要进一步处理成separated bodies。这里作者take了human-centric approach，开发了一个GUI让用户：
- Cut mesh分离objects
- Add joints between mesh elements
- Drag/drop reposition objects

User study显示non-expert用户平均14分钟40秒active time就能完成一个articulated scene。

### Step 2: Inverse Distillation (Real-to-Sim Policy Transfer)

这是paper最novel的部分。问题formulation：

给定real-world demonstrations:
$$\mathcal{D}_{\text{real}} = \{(o_1^i, a_1^i), \dots, (o_H^i, a_H^i)\}_{i=1}^N$$

其中：
- $o_t^i$: 第$i$条trajectory第$t$步的observation（3D point cloud）
- $a_t^i$: 对应的action（delta end-effector pose）
- $H$: trajectory长度
- $N$: demonstration数量（论文中~15）

目标：获得带privileged information的sim demos：
$$\mathcal{D}_{\text{sim}} = \{(o_1^i, a_1^i, s_1^i) \dots (o_H^i, a_H^i, s_H^i)\}_{i=1}^M$$

其中$s_t^i$是Lagrangian state（object poses, joint states等）。

**Key insight**: 在real-world中无法获得Lagrangian state，但是当在sim中执行learned perception policy $\pi_{\text{real}}(a|o)$时，sim可以naturally提供privileged state，因为sim中perceptual observation和Lagrangian state的pairing是known a priori的。

具体procedure：
1. 在$\mathcal{D}_{\text{real}}$上用supervised learning训练$\pi_{\text{real}}(a|o)$
2. 在sim中rollout $\pi_{\text{real}}$，用sim rendered point clouds作为input
3. 收集成功trajectories，同时记录privileged state $s_t$
4. 得到$\mathcal{D}_{\text{sim}}$（~15条带privileged info的demos）

这个步骤巧妙地bypass了real-world state estimation的难题。

### Step 3: RL Fine-tuning in Simulation

用$\mathcal{D}_{\text{sim}}$ bootstrap一个state-based policy $\pi_{\text{sim}}(a|s)$的训练。核心loss function（论文公式1）：

$$\max_{\theta, \phi} \alpha \sum_{(s_t, a_t, r_t) \in \tau_{\pi_\theta}} \min\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}\hat{A}_t, \text{clip}\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}, 1-\epsilon, 1+\epsilon\right)\hat{A}_t\right)$$
$$+ \beta \sum_{(s_t, V_t^{\text{targ}}) \in \tau_{\pi_\theta}} (V_\phi(s_t) - V_t^{\text{targ}})^2$$
$$+ \gamma \sum_{(s_t, a_t) \in \mathcal{D}_{\text{sim}}} \frac{\pi_\theta(a_t|s_t)}{\sum_{a_c} \pi_\theta(a_c|s_t)}$$

变量详解：
- $\theta, \phi$: policy network和value function network的参数
- $\pi_\theta(a_t|s_t)$: 当前policy在state $s_t$下采取action $a_t$的概率
- $\pi_{\theta_{\text{old}}}(a_t|s_t)$: 旧policy（PPO中用于ratio计算）的概率
- $\hat{A}_t$: advantage estimator at step $t$（GAE）
- $V_\phi(s_t)$: value function预测
- $V_t^{\text{targ}}$: value target
- $\epsilon$: PPO clip parameter（通常0.2）
- $\alpha, \beta, \gamma$: 三个loss项的权重系数
- $a_c$: 遍历所有可能的discrete actions
- $\tau_{\pi_\theta}$: policy rollout的trajectory

三项loss的作用：
1. **PPO clipped objective**: 标准policy gradient with importance sampling和clip防divergence
2. **Value function loss**: value function的MSE training
3. **BC loss**: 在$\mathcal{D}_{\text{sim}}$上的behavior cloning，用log-likelihood $\log \pi_\theta(a_t|s_t)$ bias policy向demonstrated actions

BC loss有两个作用：
- **Aiding exploration**: 在sparse reward setting下引导policy向promising region探索
- **Biasing toward physically plausible solutions**: 防止policy exploit simulator inaccuracies（如论文Appendix Fig 15所示，PPO from scratch会push toaster底部来打开，利用了joint placement的微小错误）

Reward design非常简单（sparse reward），例如：
- Kitchen toaster: `toaster_joint > 0.65 && gripper_open`
- Plate on rack: `||plate_site - rack_site||_2 < 0.2 && rack_y_axis · plate_z_axis > 0.9 && gripper_open`

State space: 所有objects的pose + joints state + robot state的concatenation
Action space: 14维discrete actions（6 delta position ±0.03m, 6 delta rotation ±0.2rad, 2 gripper open/close）

### Step 4: Teacher-Student Distillation with Co-training

$\pi_{\text{sim}}^*(a|s)$需要privileged state，但real-world部署时只有sensor observations。用teacher-student distillation。

Co-training objective（论文公式2）：

$$\max_\theta \alpha \sum_{(s_i, o_i, a_i) \sim \pi_{\text{teacher}}} \frac{\pi_\theta(\pi_{\text{teacher}}(s_i)|o_i)}{\sum_{a_c} \pi_\theta(a_c|o_i)}$$
$$+ \beta \sum_{(o_i, a_i) \in \mathcal{D}_{\text{real}}} \frac{\pi_\theta(a_i|o_i)}{\sum_{a_c} \pi_\theta(a_c|o_i)}$$

变量详解：
- $\pi_\theta$: student policy（point cloud input）
- $\pi_{\text{teacher}}(s_i)$: teacher policy $\pi_{\text{sim}}^*$在privileged state $s_i$下输出的action
- $o_i$: 对应的point cloud observation
- $a_c$: 遍历所有discrete actions用于normalization
- $\alpha, \beta$: 两个loss项的权重

第一项是DAgger-style distillation：student模仿teacher在sim中给出的actions
第二项是real-world BC：直接在real demos上做BC

Dataset mixing策略很精细：
- 15000 trajectories with full point clouds（all faces visible，直接从mesh采样）
- 5000 trajectories from camera viewpoint matching real-world camera
- 2000 trajectories from sim camera viewpoint with distractors
- 15 real-world trajectories
四类数据各以1/4概率采样

然后做DAgger iteration：rollout student policy，用teacher relabel actions，再mix with distractor data和real data。

Point cloud encoder用的是[Convolutional Occupancy Networks](https://github.com/ltimoner/convolutional_occupancy_networks)的架构：local point net → 3D U-Net → dense voxel grid → max pooling + average pooling → 128维embedding。

## 3. 实验设计与结果

### Tasks
8个tasks覆盖两大类：
- **6-DoF grasping and reorientation**: book on shelf, plate on rack, mug on shelf
- **6-DoF grasping + articulated object interaction**: drawer, cabinet, toaster, cup in trash, plate in kitchen

三个robustness levels（递增难度）：
1. **Pose randomization**: episode开始时randomize object/robot poses
2. **Visual distractors**: 添加clutter
3. **Physical disturbances**: episode中途改变object pose, target location, close drawer, move robot base

### Key Results

**Table I - Book on Shelf对比**:
| Method | Pose Randomization | Distractors | Disturbances |
|--------|-------------------|------------|--------------|
| BC (15 demos) | 10±9% | 0±0% | 0±0% |
| BC (50 demos) | 40±15% | 30±16% | 20±13% |
| RialTo (15 demos) | 90±9% | 70±14% | 60±16% |

RialTo用少于1/3的demos和少于1/2的human time达到2.5倍success rate。

**Figure 5 - 8 tasks平均**：
- RialTo: 91% (pose) / 77% (distractors) / 75% (disturbances)
- BC: 25% / 11% / 5%

**Table III - RL from Scratch vs from Demos**:
RL from scratch在5个tasks中3个完全失败（0%），2个poor performance。关键观察：from scratch policy会exploit simulator inaccuracies，比如push toaster底部而非handle，这种behavior无法transfer到real world。

**Co-training ablation (Figure 6)**:
- Book on shelf with disturbances: real co-training比sim co-training高3.5x
- Plate on rack with disturbances: 高2x
- 在sim-to-real gap小的tasks上，两者相当

**Real-to-sim transfer必要性 (Figure 7)**:
- 在target drawer reconstruction上训练：90% success
- 在4个Objaverse drawers上multi-task训练：10% success
结论：对于specific deployment environment，targeted real-to-sim比diverse procedural generation更effective。

**Inverse distillation ablation (Figure 6)**:
从real demos vs from sim demos启动RL fine-tuning，性能几乎相同。说明inverse distillation成功transfer了demos，且pipeline flexible——可以选择更容易获取的数据源。

**Table X - Demo数量sensitivity**:
- Book on shelf: <15 demos时0% success（因为real policy transfer不到sim，inverse distillation失败）
- Open drawer: >5 demos就能成功
存在step function，取决于task难度。

### User Study

6 users（1 expert, 5 naive），task：scan scene + cut 1 object + scan 1 object + add 1 joint。

平均total time: 25:12，active time: 14:40
Expert user: 10:54 active time

Scaling law公式（论文公式3）：
$$\text{total\_active\_time} = t_{\text{scan\_scene}} + t_{\text{scan\_object}} \cdot N_{\text{objects}} + t_{\text{cut\_object}} \cdot N_{\text{cut\_objects}} + t_{\text{add\_joint}} \cdot N_{\text{joints}}$$

User study测得的平均时间系数：
- $t_{\text{scan\_object}} = 4:50$
- $t_{\text{scan\_scene}} = 3:14$
- $t_{\text{add\_joint}} = 2:54$
- $t_{\text{cut\_object}} = 3:40$

线性scaling，且随expertise提升系数会减小。

### RL from Vision vs from State (Appendix Fig 14)
- RL from compact state: 96% success after 12 hours
- RL from vision: 1% success after 35 hours

原因：vision-based policy batch size小100x，point cloud rendering慢10x，总计~1000x slower。这justifies inverse distillation到state space的必要性。

## 4. Implementation Details

### Network Architectures
**State-based policy** (MLP):
- 2 layers × 256 units
- Input: privileged state
- Output: Categorical distribution over 14 discrete actions
- Value function shares first layer with actor

**Point cloud policy**:
- Point cloud encoder: Convolutional Occupancy Networks → 128-dim embedding
- Concat with robot state (9-dim: EE pose + gripper state)
- MLP 256,256
- Output: Categorical over 14 actions

### Point Cloud Processing (Table VII)
- Total points: 6000
- Arm points: 3000 (sampled from arm mesh)
- Dropout ratio: [0.1, 0.3]
- Jitter ratio: 0.3, noise $\mathcal{N}(0, 0.01)$
- Object mesh points: 1000
- Grid size: 32×32×32

### PPO Hyperparameters (Table VI)
- MLP layers: 256, 256
- PPO n_steps: episode length
- Batch size: 31257
- BC batch size: 32
- BC weight: 0.1
- Gradient clipping: 5

### Distillation Training (Table VIII)
- MLP: 256, 256
- LR: 0.0003
- Optimizer: AdamW
- Batch size: 32-64
- 15000 full pcd traj + 5000 sim pcd traj + 1000 sim pcd with distractors + 15 real traj

### Simulation Details
- Simulator: NVIDIA IsaacSim
- Codebase: Orbit
- Collision mesh: convex decomposition, 64 hull vertices, 32 convex hulls (default)
- Dish rack特殊处理: SDF mesh decomposition, 256 resolution
- Friction: dynamic/static = 0.5, joint = 0.1
- Default mass: 0.41 kg

### Hardware
- Franka Panda arm (两个setup: 固定table + 移动table)
- Camera: Intel RealSense D455 / D435
- Controller communication: Polymetis
- GPU: RTX 2080 / RTX 3090

### Total Training Time
- Vision policy + collect sim demos: 7 hours
- RL fine-tuning: 20 hours
- Teacher-student distillation + DAgger: 24 hours
- **Total: ~2 days 3 hours per task**

## 5. Multi-task Extension (Appendix XI.E)

RialTo可以extend到multi-task：
1. 训练separate single-task state-based policies
2. 从每个task收集trajectories
3. Distill到conditioned on task-id的single multi-task policy
4. Sequential DAgger on each task

Results (Table XII):
- Open drawer: single-task 90% → multi-task 90% (持平)
- Mug on shelf: single-task 100% → multi-task 80% (略降)
但都远超IL baseline (40% / 10%)

## 6. Limitations

作者honest地列了几个：
1. **Depth sensor限制**: thin, transparent, reflective objects检测困难
2. **Simulation可模拟性**: limited to articulated rigid bodies，deformables未支持
3. **Controller速度**: 为了minimize sim-to-real dynamics gap，controller相对慢
4. **Quasi-static assumption**: physics parameters精确identification不必要，但复杂environment会需要
5. **Training time**: 2天/task，continual learning不可行

## 7. 与相关工作的positioning

### Imitation Learning方向
- [Diffusion Policy](https://arxiv.org/abs/2303.04137): visuomotor policy via action diffusion
- [ACT](https://arxiv.org/abs/2304.13705): bimanual manipulation with low-cost hardware
- RT-1: large-scale real-world BC

这些方法在scale data时improve generalization，但RialTo走的是**test-time specialization**路线：针对specific deployment environment做targeted robustification，避免generalist policy的over-conservatism。

### Sim-to-Real方向
- [OpenAI Hand](https://arxiv.org/abs/2008.08738): domain randomization for in-hand manipulation
- [RMA](https://arxiv.org/abs/2107.04034): rapid motor adaptation for locomotion
- [Visual Dexterity](https://www.science.org/doi/10.1126/scirobotics.adc9244): in-hand reorientation via teacher-student

RialTo借鉴了teacher-student distillation + domain randomization，但address的是更challenging的household manipulation（richer visual scenes, sparse reward, minimal engineering）。

### Real-to-Sim方向
- [NeRF in the Palm](https://arxiv.org/abs/2301.08773): corrective augmentation via novel view synthesis
- [Phone2Proc](https://arxiv.org/abs/2305.09756): robust robots from phone scans
- [Ditto](https://arxiv.org/abs/2205.11646): digital twins from interaction

这些工作主要用visual component，RialTo additionally做physical interaction with reconstructed geometry，能discover novel behaviors beyond visual distractors。

## 8. Intuition Building: 为什么这个approach work?

### 为什么不用更多demos？
IL的generalization问题本质是**combinatorial explosion**: 要覆盖所有pose × distractor × disturbance的组合需要指数级demos。RL in sim通过autonomous exploration和domain randomization可以combinatorially explore这些variations。

### 为什么需要inverse distillation？
直接在sim中collect demos需要human在sim中操作，UX差且sim的dynamics可能和real有gap。Inverse distillation让我们用real demos（natural for human）+ sim的privileged state（natural for sim）。这是一个**best of both worlds**的trick。

### 为什么co-training with real data如此重要？
Sim-to-real gap有两个dimension：
1. **Visual gap**: sim rendered point clouds vs real depth camera point clouds（噪声特性、遮挡模式不同）
2. **Dynamics gap**: sim physics vs real physics

Real demos提供了**distribution anchor**，让student policy不会drift to sim-specific visual features，同时bias toward physically safe behaviors（如co-trained policy在grasp前留更多space）。

### 为什么targeted sim > diverse sim?
Generalist policy需要massive data coverage才能handle specific scene，且可能over-conservative。对于特定user的deployment environment，targeted digital twin让policy specialize到这个environment的variations，achieving higher performance with less compute和engineering。

这是一个**pragmatic**的design choice：与其追求one giant generalist model，不如build一个scalable pipeline让non-expert user快速specialize robot到自己的环境。

## 9. 延伸思考与可能的后续方向

### Foundations方向
1. **Automatic system identification**: 论文用default physics params + BC regularization来compensate sim-to-real gap。如果能自动infer physics params（参考[ASID](https://arxiv.org/abs/2404.12308)），可以进一步缩小dynamics gap，enable更dynamic tasks
2. **Deformable objects**: 当前limited to articulated rigid bodies。结合differentiable simulation for deformables（如[DiffCloud](https://arxiv.org/abs/2208.01176)）可以expand适用范围
3. **RGB/RGBD policy**: point cloud虽然sim-to-real友好但信息量有限，RGB可以capture texture和fine-grained visual cues，但sim-to-real visual gap更大。可能需要更好的domain adaptation或generative augmentation

### Scaling方向
1. **Faster training**: 2天/task限制了continual learning。Point cloud encoder是bottleneck，更高效的3D representation（如sparse voxels, implicit representations）可能加速
2. **Foundation model integration**: 论文提到"RialTo can robustify large, expressive pretrained models"。如果能从large-scale pretraining（如RT-2, OpenVLA）出发，inverse distillation + RL fine-tuning可能能bootstrap出更强policy
3. **Language conditioning**: 加language instruction可以让一个digital twin支持多task，减少per-task engineering

### UX方向
1. **Fully automatic scene parsing**: 当前GUI需要human cut mesh + add joint。结合[URDformer](https://arxiv.org/abs/2310.16531)这类work可以自动化更多步骤
2. **Active scanning**: robot自己探索scene，decide哪里需要更详细scan，减少human burden
3. **Incremental scene update**: 部署后scene变化时，只update变化部分而非重新scan整个scene

### RL/IL Theory方向
1. **BC loss weight schedule**: 当前用固定$\gamma$。Curriculum-style decay或adaptive weighting可能更好balance exploration和imitation
2. **Reward learning from demos**: 当前用hand-designed sparse reward。Inverse RL或preference learning可以reduce reward engineering
3. **Off-policy RL**: 当前用PPO（on-policy）。结合offline RL如[CQL](https://arxiv.org/abs/2006.04779)或[IQL](https://arxiv.org/abs/2110.06169)可以更sample-efficient

### 真实部署方向
1. **Safety constraints**: 当前policy可能在disturbance下做出unsafe动作。加constraint layer或safety shield
2. **Failure detection**: policy需要知道何时fail并human介入或re-plan
3. **Multi-robot fleet learning**: 一个robot的digital twin可以share给其他robot在类似环境部署

## 10. 与当前Foundation Model趋势的对比

当前robotic foundation model趋势（RT-2, OpenVLA, Octo等）走的是**generalist**路线：用massive data训练一个能handle many scenes的policy。

RialTo代表的是**specialist + easy specialization**路线：承认generalist model在specific deployment可能underperform，提供一个fast pipeline让user specialize到自己的环境。

两种路线可能converge：
- Foundation model作为$\pi_{\text{real}}$的strong initialization
- RialTo pipeline作为test-time adaptation/finetuning mechanism
- Digital twin作为efficient exploration sandbox

这是一个很有潜力的hybrid方向。

## 11. 代码实现细节与可能的坑

从论文细节推断的一些implementation insights：

### Point Cloud Processing
- 6000 points = 3000 scene + 3000 arm。Arm points从mesh采样，需要forward kinematics获取当前arm mesh
- Normalization: toaster用[0,0,0], 其他用[0.35,0,0.4]作为center
- Scale: toaster用0.625, 其他用1
- Dropout和jitter是关键augmentation，让policy robust to real depth sensor noise

### Inverse Distillation的失败模式
- <15 demos时real policy无法transfer到sim（Table X），导致inverse distillation收集不到成功trajectories
- 这种情况下RL fine-tuning退化为from scratch
- 解决方案：要么collect更多demos，要么用更好的IL方法（如diffusion policy）提高few-shot performance

### Teacher-Student Dataset Mixing
- 15000 full pcd traj + 5000 camera-view traj + 1000 with distractors + 15 real
- 这个比例很关键：太多real data会让policy overfit到specific demos，太少又无法bridge sim-to-real gap
- DAgger iteration后再mix with 1/3 distractor + 1/3 real

### PPO + BC的stability
- BC weight 0.1（Table VI），太大会over-constrain exploration，太小又无法prevent exploiting sim artifacts
- BC batch size 32 vs PPO batch size 31257，比例~0.1%，与BC weight匹配

## 总结

RialTo是一个**工程上非常thoughtful**的系统paper。它没有提出fundamentally new algorithm，但是把real-to-sim scene construction、inverse distillation、RL fine-tuning、teacher-student distillation with co-training这几个component有机组合成一个end-to-end pipeline，每个component都解决了specific bottleneck。

从intuition角度，这篇paper的beauty在于：
- 用real demos作为distribution anchor，避免sim policy drift
- 用sim privileged state作为efficient RL training substrate
- 用inverse distillation桥接两个worlds的information asymmetry
- 用co-training在distillation阶段重新inject real-world information

这是一个**pragmatic engineering**胜过**pure algorithmic novelty**的典范，也符合robot manipulation领域当前的需求——我们需要的不是更fancy的algorithm，而是能让non-expert user真正deploy robust policy到real-world的system。

References:
- [RialTo Project Page](https://real-to-sim-to-real.github.io/RialTo/)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [ALOHA/ACT](https://arxiv.org/abs/2304.13705)
- [Visual Dexterity](https://www.science.org/doi/10.1126/scirobotics.adc9244)
- [RMA](https://arxiv.org/abs/2107.04034)
- [NeRF in the Palm](https://arxiv.org/abs/2301.08773)
- [Phone2Proc](https://arxiv.org/abs/2305.09756)
- [Ditto](https://arxiv.org/abs/2205.11646)
- [Convolutional Occupancy Networks](https://github.com/ltimoner/convolutional_occupancy_networks)
- [NeRFStudio](https://github.com/nerfstudio-project/nerfstudio)
- [Polycam](https://poly.cam)
- [AR Code](https://ar-code.com/)
- [IsaacSim](https://developer.nvidia.com/isaac-sim)
- [Orbit](https://isaac-orbit.github.io/)
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [ASID](https://arxiv.org/abs/2404.12308)
- [DiffCloud](https://arxiv.org/abs/2208.01176)
- [URDformer](https://arxiv.org/abs/2310.16531)
- [CQL](https://arxiv.org/abs/2006.04779)
- [Objaverse](https://objaverse.allenai.org/)
