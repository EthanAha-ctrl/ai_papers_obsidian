---
source_pdf: MANISKILL2.pdf
paper_sha256: 89fb794345d860ee696ca2f4d7a75059b15352b615794eeadb0e6075426136f0
processed_at: '2026-08-05T16:25:14-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# ManiSkill2 用人话讲

## 这paper到底干了啥

一句话：UCSD Hao Su 组做了一个机器人操作的"考场"，里面有20类活、2000多个物件、400万帧示范数据，让机器人学习操作技能的算法能在一个统一的地方比试、验证、迭代。

类比一下：之前的机器人 benchmark 像是各个实验室自己搭的小黑板，每个组画几个题自己测自己。ManiSkill2 是想做一个 standardized 的国家考试中心——题目多、评分严、设备统一、还免费开放。

---

## 为啥要做这个

之前的 benchmark 都有点"作弊"：

- **假抓取 (abstract grasp)**: 像 BEHAVIOR-1K、Habitat 2.0、RLBench 这种，机器人手伸到物体附近就自动算"抓住了"，根本不模拟手指和物体之间的真实接触力。这就像驾考只用鼠标点"刹车"就过关，不用真踩踏板。

- **物体太少**: MetaWorld 才 80 个物体，RoboSuite 才 10 个，RLBench 28 个。你学了 10 个物体的抓取，第 11 个就不灵了。这就像只在 10 个老顾客身上学做菜，遇到新客人就懵。

- **没有软体**: 现实里机器人要处理衣服、水、橡皮泥、面团这些东西，但大多数 benchmark 只有刚体。

- **任务类型单一**: 有的只测 4-DoF 的 pick-and-place，有的只测装配，没有统一的。

ManiSkill2 的 commitment 是：**真物理接触 + 2000+ 物体 + 软体 + 多类型任务**，全都来。

---

## 具体做了啥

### 1. 任务：20 类，分 4 大组

**软体操作 (6 个)**：Fill 把橡皮泥倒进杯子、Hang 把面条挂杆上、Excavate 挖指定量的泥、Pour 倒水到红线、Pinch 捏橡皮泥成指定形状、Write 在泥上写字。

这个分梯度特别妙：Fill 和 Hang 只需要"差不多就行"，Pour 和 Pinch 要"精确到位"。实验结果显示，简单任务能学到 40-60%，精细任务直接 0%。

**精密装配 (3 个)**：PegInsertionSide (3mm 间隙)、PlugCharger (0.5mm 间隙)、AssemblingKits (0.8mm 间隙)。

重点是——以前的 RLBench 测装配只看"是否靠近预定位置"，根本没真的挖洞。ManiSkill2 强制要求 peg 真的插进 hole 里。这把"假装配"还原成"真装配"。

**6-DoF 搬运 (5 个)**：PickCube、StackCube、PickSingleYCB (74 个物体)、PickSingleEGAD (1600 个物体)、PickClutterYCB (杂乱场景)。

一个 subtle 的设计：目标位置在一个 30×50×50 cm 的大工作空间里随机。这意味着你选的抓取姿势不仅要抓得稳，还要够得着——纯视觉评分的 grasp predictor 在这里会败给能考虑运动学可达性的方法。

**articulated object 操作 (5 个)**：推椅子、搬水桶、开柜门、开抽屉、开水龙头。开水龙头特别 tricky，因为 handle 的旋转轴是 3D 的，agent 得从观察里推断该往哪个方向转。

### 2. 软体仿真：自己撸了一个 MLS-MPM

MPM (Material Point Method) 是一种混合方法：用粒子存材料状态，用背景网格算碰撞和动量。MLS-MPM 是 2018 年 SIGGRAPH 的一个改进版，让数学更简洁、GPU 友好。

**关键创新是 2-way coupling**：之前的 PlasticineLab 只能在 MPM 内部算，没法跟外部刚体仿真器交互。ManiSkill2 把刚体形状的 SDF (signed distance function) 复制到软体仿真器，每个时间步评估粒子位置的 SDF、算惩罚力、再把力传回刚体仿真器。

效果：单环境 17-18 FPS，16 个并行环境能跑到 80-84 FPS on RTX Titan——4 倍实时。对一个 MPM 软体仿真器来说是数量级提升。

**人话讲**：以前你想让机器人捏橡皮泥，仿真器要么很慢、要么不能跟机器人耦合。ManiSkill2 用 Nvidia Warp (一个 JIT 框架) 自己撸了一个 GPU 仿真器，跑得飞快，还能跟 SAPIEN 的刚体仿真器双向通信——机器人推橡皮泥，橡皮泥也反推机器人。

### 3. 多控制器 + 动作空间转换：被低估的亮点

控制器是把 policy 输出转成电机命令的接口。ManiSkill2 支持 10+ 种：joint position、joint delta position、end-effector delta pose、joint velocity 等等。

**真正有意思的是动作空间转换**：你用 TAMP 通过 joint position controller 生成了 1000 条 demo，但你想用 end-effector delta pose controller 训 RL agent。怎么把 demo 的动作序列"翻译"过去？

方法：每一步用 forward kinematics 把 source 的 joint position 算成 end-effector pose，再除以 target 环境**实时读出**的 current pose，得到 delta action。这是 closed-loop 的——target 环境真在跑，能纠正累积误差。

成功率：PickSingleYCB 99%、AssemblingKits 98%、Write 100%、TurnFaucet 80%。

**人话讲**：你的示范数据是用"控制关节角度"的方式录的，但你想训"控制手部位移"的 agent。ManiSkill2 给你做个实时翻译，让数据可以跨控制器复用。这就像把英文教材翻译成中文还能保持原意——而且翻译器自己也在听课、随时纠正。

### 4. 渲染服务器架构：2000 FPS 怎么来的

传统多进程环境的渲染流程有个大浪费：CPU 等 GPU 渲染完才能算 reward，然后图片 GPU→CPU→主进程→GPU 一通拷贝。

ManiSkill2 的两招：

**异步渲染**：reward 通常不依赖图像（只看物体位置），所以渲染一开始就并行算 reward，不等。

**渲染服务器**：主进程开个 thread pool 当渲染服务器，所有 worker 进程通过 gRPC 把渲染请求送过去。所有渲染资源（mesh、texture）只存一份，跨进程共享。

结果：Nature CNN 跑到 2532 FPS，Habitat 2.0 才 1224，Isaac Gym 835。

GPU 内存对比更夸张：74 个 YCB 物体放 64 个并行环境，Habitat 直接 OOM，ManiSkill2 只用 5.8GB。

**人话讲**：以前每个进程都自己加载一份物体模型到 GPU，100 个进程就是 100 份重复。ManiSkill2 把渲染集中到一个服务器进程，所有进程共享同一份资源。这就像图书馆从"每人买一本"改成"图书馆借阅"——书还是那么多书，占地少了 99%。

---

## 实验发现了啥

### Sense-Plan-Act (传统模块化方法)

**Contact-GraspNet + 运动规划 在 PickSingleYCB 上 43.24%**

失败原因分两类：
- 27% 是 grasp 预测 confidence 低（小、薄的物体如勺子叉子）
- 30% 是 confidence 高但抓取质量差或机器人够不着

**Transporter Networks 在 AssemblingKits 上 18%**（用 ManiSkill2 严格 metric）；用原 paper 的松散 metric 是 99%。

**人话讲**：现有模块化方法在物体多样性大、精度要求高的场景下根本不够看。模块化方法的好处是可解释，坏处是模块之间不联合优化，每个模块的错误会传递放大。

### 行为克隆

刚体任务几乎全 0，软体任务 Fill 40-60%、Hang 20-35%，Pour/Pinch/Write 全 0。

**为啥这么差**：BC 用 L2 loss 做 regression，对 multi-modal action distribution 会塌缩到 mean。装配任务有多个解（不同插入角度），BC 学了个平均动作，啥也插不进去。软体的精细任务要求 agent 理解"动作如何影响软体形状"，BC 没这个 forward model 能力。

### RL (DAPG+PPO)

**关键结果**：

| 任务 | Point Cloud | RGBD |
|------|-------------|------|
| PickCube | 0.94 | 0.95 |
| PickSingleYCB | 0.51 | 0.18 |
| PegInsertionSide | 0.01 | 0.01 |
| AssemblingKits | 0.00 | 0.00 |
| TurnFaucet | 0.04 | 0.04 |

三个核心发现：

1. **PickSingleYCB 的 point cloud agent (0.51) 比 Contact-GraspNet+运动规划 (0.43) 还高！** 这是端到端 RL 第一次在跨物体泛化上击败模块化方法。说明 end-to-end 学习能学到 grasp quality 和 reachability 的联合优化。

2. **Point cloud 一致 beat RGBD**。在 PickSingleYCB 上 0.51 vs 0.18，差距巨大。3D 信息对 manipulation 是 essential 的。

3. **装配任务全 0**。把间隙扩大 10x，PegInsertionSide 从 0.01 涨到 0.74。说明 PPO+Gaussian policy 在毫米级 contact-rich 任务上 fundamentally 不够。

### 关键 ablation：表示比算法重要

PickSingleYCB point cloud agent：
- 原始设置：0.51
- 换 controller (joint delta 代替 ee delta pose)：0.22
- 换坐标系 (base frame 代替 ee frame)：0.00
- 去掉视觉目标提示 (50 个绿点)：0.16

**人话讲**：换算法可能涨几个点，换 controller 直接腰斩，换坐标系直接归零，加视觉提示涨 3 倍。这告诉我们 manipulation RL 的瓶颈在 representation engineering，不在 algorithm tuning。

---

## Sim2Real

PickCube：仿真 91%，真实 60%。Gap 主要来自 depth sensor 的 noise 分布差异。

Pinch (软体捏橡皮泥)：执行相同动作序列在仿真和真实中，最终形状合理一致。说明 2-way coupled MPM 仿真器已经能复现真实软体动力学。

---

## 大的启示

### 1. 这paper的真正价值是 infrastructure

ManiSkill2 不发明算法，它造考场。但历史上 ImageNet、MuJoCo、Isaac Gym 都是 infrastructure 跃升催生算法爆发。ManiSkill2 的 2000 FPS + 2000 物体 + 4M demo frames 把 manipulation RL 从"玩具规模"推到"真实规模"。

### 2. 暴露的核心 open problem

- **Contact-rich precision**：PPO 在 3mm 间隙装配上全失败。可能需要 off-policy + impedance control learning 或 contact-aware policy parameterization
- **Soft-body forward model**：BC 学不到软体动力学，可能需要 differentiable physics 或 model-based RL
- **Multi-modal action distribution**：BC 在装配上全 0，因为 L2 loss 对多解任务塌缩。这正是 Diffusion Policy (Chi et al. 2023) 后来解决的问题
- **Representation engineering**：换 controller 比换算法影响大得多。Observation frame、action parameterization、visual cues 这些 design choices 才是决定性的

### 3. 哲学层面的联想

这paper某种意义上呼应了 software 2.0 的精神：把数据从特定执行模态中 decouple 出来，让 demonstration 成为可复用的 asset。Controller conversion 就是把"数据"和"执行"分离的工程实现——这跟 ImageNet 让标注数据可被任何 CNN 架构复用是一回事。

更深一层，ManiSkill2 暴露的问题（BC 多模态失败、RL 接触失败）正是 software 2.0 在 robotics 上的 frontier。下一代算法会是 software 2.5 的 hybrid：neural networks + structured priors（diffusion for 多模态、impedance control for 接触、MPM for 可微物理）。

---

## 一句话总结

ManiSkill2 是把机器人操作 benchmark 推到新 scale 的工作，证明了 representation 比 algorithm 重要，暴露了 contact-rich precision 和 soft-body reasoning 是下一个 wave 的主战场，并提供了让社区能 attack 这些问题的开放基础设施。

参考链接：
- 项目主页: https://maniskill.github.io/
- 代码: https://github.com/haosulab/ManiSkill
- Paper: https://arxiv.org/abs/2302.04629

---

# ManiSkill2 深度解析

## 1. Paper 的定位

ManiSkill2 是 UCSD Hao Su 组继 ManiSkill1 之后的 embodied AI manipulation benchmark，主要解决一个核心痛点：**现有 benchmark 在 object-level topology/geometric variation、fully dynamic simulation、multiple task types 三个维度上无法同时满足**。这篇工作的野心很大——想做 manipulation 领域的 "ImageNet"，一个 unified、fast、accessible 的系统，能同时支撑 sense-plan-act、RL、IL 三类算法，且涵盖 rigid/soft body、stationary/mobile-base、single/dual-arm 的全谱系。

从更高视角看，这篇 paper 反映了 embodied AI 领域 2022 年的一个重要拐点：社区从"单 task 单 object 的 toy benchmark"转向"large-scale physically realistic + generalizable"的范式。同期还有 BEHAVIOR-1K (Li et al.)、Habitat 2.0 (Szot et al.) 等，但它们大多采用 abstract grasp 来 bypass 物理接触的复杂性。ManiSkill2 的核心 commitment 是 **fully physical grasp + fully dynamic simulation**，这是它的差异化护城河。

参考链接：
- Paper: https://arxiv.org/abs/2302.04629
- Project page: https://maniskill.github.io/
- GitHub: https://github.com/haosulab/ManiSkill
- SAPIEN: https://sapien.ucsd.edu/

---

## 2. Task Heterogeneity: 20 个 Task Families 的分类学

这是 paper 最有信息量的部分之一。20 个 task families 分为 4 大类：

### 2.1 Soft-body Manipulation (6 tasks)

- **Fill**: 把 clay 从 bucket 倒进 beaker，成功条件是 clay amount > 90%
- **Hang**: 把 noodle 挂到 rod 上，需要 noodle 两端在 rod 不同侧
- **Excavate**: 挖特定 amount 的 clay 到特定 height
- **Pour**: 把 water 从 bottle 倒进 beaker，liquid level 与 red line 偏差 < 4mm
- **Pinch**: 把 plasticine pinch 成 target shape，用 Chamfer distance < 0.3t 衡量
- **Write**: 在 clay 上 write character，IoU > 0.8

这里的 intuition 是：soft-body manipulation 的难度呈梯度分布。**Fill 和 Hang 只需要 coarse control**（达到一个阈值即可），而 **Excavate、Pour、Pinch、Write 需要精细的 deformable object reasoning**——agent 必须理解 action 如何影响 soft body 的 displacement、deformation、最终 shape。从 Table 2 的 BC 结果看，Fill (0.45/0.62) 和 Hang (0.35/0.20) 显著高于 Pour (0.02) 和 Pinch/Write (0.0)，完美验证了这个假设。

### 2.2 Precise Peg-in-hole Assembly (3 tasks)

- **PegInsertionSide**: 3mm clearance，单 peg
- **PlugCharger**: 0.5mm clearance，dual peg
- **AssemblingKits**: 0.8mm clearance，多 shape

这里的关键 contribution 是 **millimeter-level precision with physical contact**。RLBench (James et al.) 的 PlugCharger 只检查 charger 与 receptacle 的 proximity，不 modeling holes；MetaWorld 的 PegInsertionSide 只要求 peg head 接近 hole surface。ManiSkill2 强制要求 **peg 实际进入 hole 并达到 half-insertion 或 fully-fit**，这逼迫算法处理 rich contact dynamics。

**Intuition**: 这是一个很 sharp 的 benchmark 设计——它把 prior work 的"伪 assembly"还原成真 assembly。从 Table 3 可以看到 DAPG+PPO 在所有 assembly tasks 上成功率几乎为 0（PlugCharger 0.01, AssemblingKits 0.00），这证明现有 visual RL 算法在 contact-rich precision insertion 上是 fundamentally broken 的。Table 8 的 ablation 进一步揭示：把 clearance 扩大 10x，PegInsertionSide point cloud 从 0.01 升到 0.74——clearance 是 RL 算法的致命瓶颈。

### 2.3 Stationary 6-DoF Pick-and-place (5 tasks)

- **PickCube / StackCube**: 简单几何
- **PickSingleYCB / PickSingleEGAD / PickClutterYCB**: 74 YCB objects + 1600 EGAD objects + 40 held-out

这里有一个很 subtle 的设计点：**goal position 在 30×50×50 cm³ 的 workspace 内随机**，这意味着 grasp pose selection 必须考虑 **kinematic reachability**，纯粹基于 visual quality 的 grasp scoring 会失败。这正是 Contact-GraspNet (CGN) 在 PickSingleYCB 上只有 43.24% 成功率的根本原因——29.73% 的失败是 high confidence but unreachable。

### 2.4 Mobile/Stationary Articulated Object Manipulation (5 tasks)

继承自 ManiSkill1 的 PushChair、MoveBucket、OpenCabinetDoor、OpenCabinetDrawer，加新的 TurnFaucet。TurnFaucet 的特殊性在于：faucet handle 的 joint axis 是 3D 的，agent 需要从 observation 推断旋转方向，而不是 predefine 一个 axis。

加一个 **AvoidObstacles** task 测试 active perception。

---

## 3. Multi-Controller 与 Action Space Conversion: 一个被低估的创新

这部分我觉得是 paper 里最被低估的 contribution。让我详细展开。

### 3.1 Controller 套件

ManiSkill2 实现了 10+ controllers，分两类：joint-space 和 task-space。

**Arm controllers**:
1. `arm_pd_joint_pos` (7-dim, unnormalized): $a_t = \bar{q}_t$，target joint positions 直接作为 action
2. `arm_pd_joint_delta_pos` (7-dim): $a_t = \bar{q}_t - q_{t-1}$，相对于当前 joint position 的 delta
3. `arm_pd_joint_target_delta_pos` (7-dim): $a_t = \bar{q}_t - \bar{q}_{t-1}$，相对于上一个 desired 的 delta
4. `arm_pd_ee_delta_pos` (3-dim): $a_t = \bar{p}_t - p_{t-1}$，end-effector position delta，内部通过 IK 转换
5. `arm_pd_ee_delta_pose` (6-dim): $\bar{T}_t = T_a \cdot T_{t-1}$，end-effector SE(3) delta pose
6. `arm_pd_ee_target_delta_pose` (6-dim): $\bar{T}_t = T_a \cdot \bar{T}_{t-1}$
7. `arm_pd_joint_vel` (7-dim): $a_t = \bar{\dot{q}}_t$，velocity control，$K_p = 0$
8. `arm_pd_joint_pos_vel` (14-dim): 同时输入 position 和 velocity target
9. `arm_pd_joint_delta_pos_vel` (14-dim): delta 版本

底层 PD controller 公式：
$$\tau(t) = K_p(\bar{q}(t) - q(t)) + K_d(\bar{\dot{q}}(t) - \dot{q}(t))$$

变量说明：
- $\tau(t)$: 电机 torque (generalized force)
- $\bar{q}(t)$: target joint position（controller 的输入）
- $q(t)$: current joint position（从 simulator 读）
- $\bar{\dot{q}}(t)$: target joint velocity
- $\dot{q}(t)$: current joint velocity
- $K_p$: stiffness 增益
- $K_d$: damping 增益

**Intuition**: PD controller 是一个 critically-damped spring-damper system 的离散实现。$K_p$ 决定 spring stiffness，$K_d$ 决定 damping。当 $K_p$ 很大时，joint 几乎瞬间到达 target；当 $K_d$ 适当调谐时，避免 oscillation。

### 3.2 Closed-loop Action Space Conversion

这是整个 paper 的核心算法贡献之一。场景：

- **Source environment**: 用 `arm_pd_joint_pos` controller，通过 TAMP 生成 demonstrations
- **Target environment**: 用 `arm_pd_ee_delta_pose` controller，给 RL/IL agent 用

核心公式：

$$a_{\mathrm{tgt}}(t) = \bar{T}_{\mathrm{tgt}}(t) \cdot T_{\mathrm{tgt}}^{-1}(t) = \bar{T}_{\mathrm{src}}(t) \cdot T_{\mathrm{tgt}}^{-1}(t) = FK(\bar{a}_{\mathrm{src}}(t)) \cdot T_{\mathrm{tgt}}^{-1}(t)$$

变量逐项解释：
- $a_{\mathrm{tgt}}(t)$: target controller 在时间 $t$ 的 desired action（这里就是 delta end-effector pose）
- $\bar{T}_{\mathrm{tgt}}(t)$: target environment 在 $t$ 时刻的 **desired** end-effector pose（在 SE(3) 中）
- $T_{\mathrm{tgt}}(t)$: target environment 在 $t$ 时刻的 **current** end-effector pose
- $T_{\mathrm{tgt}}^{-1}(t)$: current pose 的逆，用于表达"从 current 到 desired"的 relative transform
- $\bar{T}_{\mathrm{src}}(t)$: source environment 的 desired end-effector pose
- $FK(\cdot)$: forward kinematics 函数，把 joint positions 映射到 end-effector pose
- $\bar{a}_{\mathrm{src}}(t)$: source controller 在 $t$ 的 action（这里是 desired joint positions）

**关键 intuition**: 这个公式成立的前提是 $\bar{T}_{\mathrm{tgt}}(t) = \bar{T}_{\mathrm{src}}(t)$，即两个 environment 的 desired end-effector trajectory 一致。Source 的 desired pose 通过 FK 从 joint position 计算出来，target 的 current pose 从 instantiated target environment 实时读取。

**为什么 closed-loop**: 因为 $T_{\mathrm{tgt}}^{-1}(t)$ 是从 **实际运行的 target environment** 读出来的，不是从 source 复制过来的。Open-loop 方法会用 $T_{\mathrm{src}}^{-1}(t)$，但 source 和 target 由于 controller dynamics 不同会累积 execution error。Closed-loop 等价于在每个 timestep 做 trajectory tracking correction。

从 Appendix B.5 的实验数据看，这个方法在 PickSingleYCB 上 99% 成功，AssemblingKits 98%，Write 100%，TurnFaucet 80%（TurnFaucet 低是因为 push-based policy 接触不稳定，而非 conversion 本身的问题）。

**联想**: 这个 idea 跟 robot learning 里的 **action representation learning**（例如 ACT, Diffusion Policy）有共同精神——都是 decouple "what trajectory to execute" 和 "how to execute it"。ManiSkill2 的方法是把 demonstration 数据的 controller 维度 demultiplex 出来，让 demonstrations 可被任何 controller 复用，这是一个 data-centric 的设计哲学。

---

## 4. MLS-MPM Soft Body Simulation: 从 PlasticineLab 走向 Real-time

### 4.1 MLS-MPM 基础

Material Point Method (MPM) 是 hybrid Lagrangian-Eulerian 方法：
- Lagrangian particles: 携带 material state（mass, velocity, deformation gradient $F$, affinity $C$）
- Eulerian background grid: 处理 collision、momentum exchange

MLS-MPM (Hu et al., 2018) 的核心创新是用 **moving least squares (MLS) shape function** 替代 standard B-spline，让 deformation gradient 更新可以解析表达，省去 numerical differentiation。

每个 timestep 的 pipeline（Algorithm 1）：
1. **P2G (Particle to Grid)**: 把 particles 的 mass、momentum、forces 累加到 grid nodes
2. **Grid compute velocity**: 在 grid 上解 momentum equation
3. **G2P (Grid to Particle)**: 把 grid velocity 插值回 particles
4. **Integrate particles**: 更新 particle positions

### 4.2 2-way Rigid-Soft Coupling

这是相对 PlasticineLab 的关键改进。Paper 的方法：

- 把 rigid body 的 collision shapes 复制到 soft-body simulator
- Primitive shapes (box, capsule) 用 analytical SDF
- Mesh shapes 转 SDF volume，存为 3D CUDA texture
- 在 MPM particle positions 上 evaluate SDF，计算 penalty force
- 累积 forces 和 torques 给 rigid body

**与 PlasticineLab 的关键差异**: PlasticineLab 在 MPM **grid nodes** 上 evaluate SDF，把 force 应用到 grid；ManiSkill2 在 **particles** 上 evaluate，apply 到 particles。Paper 报告 particle-based 方法产生 fewer penetration artifacts。

**Intuition**: Particle-based force application 更物理直观——每个 particle 是一个 material sample，penetration 就是 particle 在 SDF 内部，penalty force 应该直接作用在 particle 上。Grid-based 方法等价于把 force "smear" 到 grid 节点，会引入 numerical diffusion，在 boundary 附近尤其 problematic。

### 4.3 性能优化

四个方面：
1. **Warp JIT**: Nvidia Warp 把 Python 翻译成 C++/CUDA，性能接近 native
2. **Host-device transfer 优化**: 扩展 Warp 减少 CPU-GPU data transfer
3. **编译时间优化**: caching 机制
4. **Particle rendering**: screen-space splatting + bilateral filter

性能：single env 17-18 FPS, 16 parallel envs 80-84 FPS on RTX Titan。这是 4x real-time，相对于 PlasticineLab 是数量级提升。

**Sim parameters** (Table 4):
- Grid Length: 0.005-0.015 m
- Particle Volume: 6.2e-8 to 1.2e-7 m³
- Density: 300-3000 kg/m³
- Young's Modulus: 1e4 to 3e5 Pa
- Poisson Ratio: 0.3
- Yield Stress: 2e3 to 1e4 Pa

**Intuition on parameters**: Young's Modulus 1e4 Pa 对应非常软的材料（rubber 约 1e6-1e7 Pa, clay 约 1e5-1e6 Pa）。Yield stress 决定 elastic-plastic 边界——超过这个 stress，deformation 不可逆（这就是 plasticine 可以捏出形状的原因）。

### 4.4 Particle Rendering

Screen-space splatting (Cords & Staadt, 2008):
1. 把每个 particle 渲染为 sphere（screen-space）
2. Bilateral filter smooth depth buffer（Figure 7 显示效果）
3. 在 smoothed depth 上计算 normal 和 lighting

**Trade-off**: Screen-space filter 在不同 view 下可能 inconsistent，paper 通过根据 pixel-to-camera distance scale bilateral filter 来 mitigate。

---

## 5. Parallelization: Asynchronous Rendering + Render Server

这是 paper 的另一个工程亮点。Visual RL 的 throughput 瓶颈通常在 rendering。

### 5.1 Sequential Pipeline 的浪费

传统 pipeline (Figure 2a):
1. Physical simulation (worker process)
2. Update renderer GPU state + draw calls
3. **Wait for GPU render** ← CPU idle
4. Copy image GPU→CPU
5. Compute reward
6. Copy images to main process
7. Copy images to GPU for policy
8. Forward policy network
9. Copy actions to worker

两个浪费：
- Stage 3 CPU 闲置
- Stage 4→6→7 的 GPU→CPU→GPU 数据 copy

### 5.2 Asynchronous Rendering

Stage 2 之后立即开始 reward computation（Stage 5），不等 GPU 渲染完成。这要求 reward 不依赖 visual observation——对大多数 manipulation task 成立，因为 reward 通常基于 object pose（从 simulator 直接读）。

### 5.3 Render Server

关键架构：
- Main Python process 起一个 thread pool 作 render server
- Worker process 通过 gRPC (HTTP 2 + Protocol Buffers) 发渲染请求
- Server side 把 "take picture" 任务放到另一个 thread 并立即 return
- 所有渲染资源由 central resource manager 管理，确保 only one copy per resource

**收益**:
1. **GPU memory sharing**: 74 YCB objects × N envs，Habitat 线性增长 (6.4G→OOM at 64 envs)，ManiSkill2 几乎不变 (4.9G→5.8G)
2. **Throughput**: 2532 FPS (Nature CNN) vs Habitat 1224, Isaac Gym 835
3. **Network capability**: 可以分布式 sim + render
4. **Profiling 友好**: Nsight System 在 single-process 才能正常 profile Vulkan

**Table 1a 数据**:
| Framework | Total FPS (rand) | Total FPS (CNN) | Optimal #Envs |
|-----------|-----------------|-----------------|---------------|
| ManiSkill2 Server | 2487±24 | 2532±63 | 64 |
| ManiSkill2 Sync | 942±19 | 931±4 | 32 |
| Habitat 2.0 | 1275±10 | 1224±13 | 64 |
| RoboSuite 1.3 | 924±3 | 894±15 | 32 |
| Isaac Gym | 865±35 | 835±5 | 512 |

**一个有意思的 observation**: ManiSkill2 在 #Envs > #CPU cores 时表现最好（64 envs on 16 cores）。Paper 的 hypothesis：OS/driver-level scheduler 自然 interleave CPU 和 GPU 执行，当请求远超资源时反而高效。这与 EnvPool 和 Isaac Gym 的 "vectorize within GPU" 哲学不同，更像 **over-subscription 通过 context switching 隐藏 latency** 的经典 OS 思路。

---

## 6. Experiments: SPA, IL, RL

### 6.1 Sense-Plan-Act

**Contact-GraspNet (CGN) on PickSingleYCB**:
- Pretrained on ACRONYM dataset
- Pipeline: CGN 预测 grasp poses with confidence → 按 score 排序尝试 → motion planning 验证 reachability → 执行
- **Success rate: 43.24%** over 74 objects × 5 trials
- Failure modes:
  - 27.03%: low confidence predictions (small/thin objects 如 spoon, fork)
  - 29.73%: high confidence but low grasp quality or unreachable

**Intuition**: 这暴露了 grasp prediction 的两个根本问题：(1) 部分 object geometry 不在 training distribution（thin objects）；(2) grasp quality 和 reachability 是 decoupled 的，purely visual grasp scoring 无法解决。

**Transporter Networks (TPN) on AssemblingKits**:
- 把 rotation bins 从 36 提到 144（高精度需求）
- 用 base camera + hand camera 的 fused point cloud，render top-down orthographic image
- **Success rate: 18%** (ManiSkill2 metric) vs 99% (original 1cm/15° metric)
- 失败主因：imprecise rotation/position prediction for placing

**Intuition**: TPN 在原 paper 上 99% 是因为 metric 太松。ManiSkill2 的 metric 要求 piece 真的 fit 进 hole，这暴露了 TPN 的精度上限。

### 6.2 Imitation Learning: Behavior Cloning

Table 2 关键数据：
- Rigid tasks 几乎全 0（assembly tasks 完全 0）
- Soft tasks: Fill 0.45-0.62, Hang 0.20-0.35, 但 Pour/Pinch/Write 全 0

**为什么 BC 这么差**:
1. Assembly 需要毫米级精度，BC 的 L2 regression loss 在 multi-modal action distribution 上 collapse 到 mean
2. Soft-body task 的 action 影响 soft body 状态是 **highly non-linear** 的，BC 无法 learn 这种 forward model
3. Pinch/Write 需要 **target-conditioned deformation**，BC 没有显式建模 target shape 的 conditioning mechanism

### 6.3 RL with Demonstrations: DAPG+PPO

Training objective:
$$\mathcal{L}_{\rho}^{CLIP}(\theta) = -\mathbb{E}_{(s,a) \sim \rho}\left[\min\left(\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}\hat{A}(s,a), \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}(s,a)\right)\right]$$

$$\mathcal{L}_{\rho}^{1}(\theta) = -\mathbb{E}_{(s,a) \sim \rho}[\pi_\theta(a|s)]$$

$$\mathcal{L}_{DAPG+PPO}(\theta) = \mathcal{L}_{\rho_\tau}^{CLIP}(\theta) + \omega \cdot \mathcal{L}_{\rho_D}^{1}(\theta)$$

变量：
- $\rho$: transition distribution，可以是 $\rho_\tau$（online rollout）或 $\rho_D$（demonstration）
- $\theta$: policy parameters
- $\theta_{old}$: behavior policy（PPO update 前）
- $\hat{A}(s,a)$: GAE advantage estimate
- $r_t(\theta) = \pi_\theta(a|s)/\pi_{\theta_{old}}(a|s)$: probability ratio
- $\epsilon = 0.2$: PPO clip range
- $\omega = 0.1 \cdot 0.995^N$: DAPG loss weight，$N$ 是 PPO epoch count
- $\mathcal{L}_{\rho}^{1}$: DAPG 的 demonstration loss，最大化 demo actions 的 log probability

**Intuition on $\omega$ decay**: DAPG loss 在早期（policy 还差）提供 strong supervision，后期（policy 已经好）让 PPO 主导，避免 demo bias over-regularize policy。0.995^N 在 1000 epoch 后约 0.0067，几乎 disable DAPG loss。

**Table 3 关键结果** (25M steps):
| Task | Point Cloud | RGBD |
|------|-------------|------|
| PickCube | 0.94 ± 0.03 | 0.95 ± 0.02 |
| StackCube | 0.91 ± 0.05 | 0.87 ± 0.04 |
| PickSingleYCB | 0.51 ± 0.05 | 0.18 ± 0.07 |
| PegInsSide | 0.01 ± 0.01 | 0.01 ± 0.01 |
| PlugCharger | 0.01 ± 0.02 | 0.01 ± 0.01 |
| AssemblingKits | 0.00 ± 0.00 | 0.00 ± 0.00 |
| TurnFaucet | 0.04 ± 0.03 | 0.03 ± 0.03 |

**重要发现**:
1. PickSingleYCB point cloud 0.51 > CGN+motion planning 0.43！这证明 end-to-end RL 在 generalization over diverse geometries 上能 beat modular SPA pipeline
2. Point cloud consistently > RGBD，尤其在 PickSingleYCB (0.51 vs 0.18)
3. Assembly tasks 几乎全 0，证明 contact-rich precision 是 RL 的 fundamental limitation
4. Controller ablation: 把 `arm_pd_ee_delta_pose` 换成 `arm_pd_joint_delta_pos`，PickSingleYCB 从 0.51 跌到 0.22 ± 0.18——**controller choice 是 RL 性能的关键 determinant**

### 6.4 Point Cloud Manipulation Learning Ablations (Table 7)

PickSingleYCB point cloud agent:
- Original (ee delta pose + ee frame + visual cues): 0.51
- Delta joint position controller: 0.22
- Robot base frame (instead of ee frame): 0.00
- Remove visual goal cues (50 green points): 0.16

**Intuition**: 
- **End-effector frame** 对 manipulation 至关重要——把 task 表达在 ee frame 等价于把 "reach this point" 变成 "move ee by this delta"，policy 学起来 simple得多
- **Visual goal cues** 对 sparse reward 的 credit assignment 帮助大，类似 reward shaping 的 visual analog
- 这些 ablation 揭示了 manipulation RL 的 representation engineering 比算法本身重要得多

---

## 7. Sim2Real

### 7.1 PickCube

- Simulation: 91.0% success
- Real (ROKAE xMate3Pro + Robotiq 2F-140 + RealSense D415): 60.0% over 50 trials
- Domain gap 来源：depth map 的 noise characteristics（只用 Gaussian noise + random pixel dropout 训练）

### 7.2 Pinch (Soft Body)

通过 motion planning 执行相同 action sequence，对比 simulation 和 real plasticine 最终形状（Figure 4 右）。结果显示 2-way coupled rigid-MPM 能合理再现 multi-grasp deformation。

**Intuition**: Sim2real 的 gap 在 soft body 上反而小一些，因为 plasticine 是 highly damped、low frequency dynamics 的材料，对 small force error 不敏感。相比之下，rigid body contact-rich task（如 assembly）对 force control 精度极敏感，sim2real gap 会大得多。

---

## 8. 与其他 Benchmark 的对比 (Table 10)

| Benchmark | Grasp | #Objects | Multi-controller | Soft-body |
|-----------|-------|----------|------------------|-----------|
| ManiSkill2 | Physical | >2144 | Yes | Warp-MPM |
| BEHAVIOR-1K | Abstract | 3324 | Unknown | Omniverse |
| Habitat 2.0 | Abstract | YCB | Yes | - |
| IsaacGym | Physical | - | No | - |
| MetaWorld | Physical | 80 | No | - |
| RoboSuite | Physical | 10 | Yes | - |
| RLBench | Abstract | 28 | Yes | - |

**ManiSkill2 独特之处**: 唯一同时拥有 **physical grasp + 2000+ objects + multi-controller + native soft-body + fast visual RL** 的 benchmark。

---

## 9. 大图景与未来方向

### 9.1 在 Embodied AI 大图景中的位置

ManiSkill2 代表了 manipulation benchmark 从 **"single task, few objects, abstract grasp"** 到 **"multi-task, large-scale, physical contact"** 的范式转移。这与 computer vision 从 ImageNet 到 LAION 的转移有相似 spirit——scale 和 physical fidelity 是 generalizable skill 的 prerequisite。

### 9.2 关键 open problems（从 paper 结果反推）

1. **Contact-rich precision RL**: DAPG+PPO 在 assembly 上全 0。这暗示 PPO 的 on-policy + Gaussian policy 在 contact-rich task 上 fundamentally insufficient。可能方向：off-policy algorithms (SAC, DDPG)、contact-aware policy parameterization、impedance control learning
2. **Soft-body forward model learning**: BC 在 Pinch/Write 上全 0，说明 policy 无法 implicit 学到 soft-body forward model。可能方向：model-based RL with differentiable physics、latent dynamics model
3. **Long-horizon composition**: Paper 只测 single task。Generalizable manipulation skills 真正考验是 composition——这是 SayCan (Ahn et al.) 和 Code as Policies 的方向
4. **Photorealism vs Speed trade-off**: SAPIEN 有 Kuafu ray-tracing renderer，但牺牲速度。如何在 fast rasterization 和 photorealism 之间找到 sim2real sweet spot

### 9.3 与后续工作的关联

ManiSkill2 之后的关键进展：
- **ManiSkill3** (2024): 加入更多 tasks、better rendering、humanoid robots
- **RT-2, RT-X** (Google DeepMind): VLM-based manipulation，用 large-scale cross-embodiment data
- **Diffusion Policy** (Chi et al.): 把 action 生成建模为 diffusion，解决 multi-modal action distribution 问题（正好回应 BC 在 assembly 上失败的痛点）
- **RDT-1B** (2024): Robotics transformer with diffusion
- **Octo** (2023): Transformer-based generalist robot policy
- **Volumetric Grasp Net / Anygrasp**: CGN 的后继，提升 grasp quality

ManiSkill2 的 unified interface 使其成为这些 generalist policy 方法的 standard evaluation platform。

### 9.4 一个 meta-observation

这篇 paper 在某种意义上是 "engineering-heavy" 工作——它不是提出新算法，而是构建基础设施。但正是这种 infrastructure work 推动了整个 field。回顾 ImageNet、COCO、MuJoCo、Isaac Gym 的历史，每一次 infrastructure 跃升都催生一波算法创新。ManiSkill2 的 2000 FPS、2000+ objects、4M demo frames 把 manipulation RL 从 "toy" 推向 "real scale"，下一个 wave 的算法会在这个 scale 上涌现。

### 9.5 对 Karpathy 的 recall

你之前在 Stanford CS231N 和 OpenAI 的工作中强调过 "software 2.0" 的范式——data-driven learning替代 hand-crafted rules。ManiSkill2 的多 controller action space conversion 某种意义上呼应了这个哲学：**demonstration 数据应该 decoupled from execution modality**，让数据成为一种 reusable asset，可被任意 policy architecture 和 controller 复用。这跟 ImageNet 的数据复用精神一致。

更深一层，ManiSkill2 暴露的问题（BC 在 multi-modal action 上的失败、RL 在 contact-rich 上的失败）正是 software 2.0 在 robotics 上的 frontier。解决这些问题需要 software 2.5 的 hybrid：neural networks + structured priors（如 diffusion for multi-modal、impedance control for contact、MPM for differentiable physics）。

---

## 10. 总结

ManiSkill2 的核心 contribution 是把 manipulation benchmark 推到一个新 scale：**20 task families × 2000+ objects × 4M demo frames × 2000 FPS × physical grasp × soft-body × multi-controller**。它不仅揭示现有算法（SPA, BC, PPO）在 generalizable manipulation 上的根本局限（assembly precision, soft-body reasoning, multi-modal action），还提供了 scalable infrastructure 让 community 能 attack 这些问题。

Paper 的核心 insight 可以归纳为：**generalizable manipulation skill 的瓶颈在 representation（controller, observation frame, action parameterization）而非 algorithm**。Table 7 的 ablation 表明，换 controller 导致 0.51→0.22，换 frame 导致 0.51→0.00——这些是 representation choices，不是 PPO 算法的问题。

参考链接补充：
- PlasticineLab: https://github.com/zeshunzx/PlasticineLab
- Warp: https://github.com/NVIDIA/warp
- MLS-MPM paper: https://zhuanlan.zhihu.com/p/392050285 (Taichi)
- DAPG: https://arxiv.org/abs/1709.10087
- Transporter Networks: https://transporternets.github.io/
- Contact-GraspNet: https://research.nvidia.com/publication/2021-08_Adaconda
- SAPIEN: https://sapien.ucsd.edu/
- ManiSkill2 challenge: https://sapien.ucsd.edu/challenges/

如果你对某个具体 module（例如 MLS-MPM 的数学细节、DAPG loss 的梯度分析、render server 的 gRPC 协议）想 dive deeper，告诉我具体方向，我可以进一步展开。
