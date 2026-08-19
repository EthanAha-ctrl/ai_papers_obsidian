---
source_pdf: Handroid Bridging Dexterous Hand and Humanoid.pdf
paper_sha256: 6af1ed342a930840af3eb6a2e9ab44ee0718c8f2cb53c65373c23c279b262b75
processed_at: '2026-08-19T10:21:10-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Handroid 人话版

Andrej, 行, 我换个讲法, 像咱们在office白板前聊天那样讲。

## 这玩意儿到底是个啥

简单一句话: **有人做了一台robot, 能在"灵巧手"和"小号人形机器人"之间一键切换**。

就这么个事。0.33米高, 2公斤重, 桌面级, 27个电机, 一个开关按钮的事儿——你按一下, 手指模块往下滑, 整个东西就从一只5指手变成了一个有小脑袋有腿的mini humanoid。再按一下, 又变回去。

项目主页: https://handroid.org

## 为什么这事有意思

咱们平时看到robotics research, dexterous hand是一个圈子, humanoid是另一个圈子。Shadow Hand、LEAP Hand做精细抓取; Unitree H1、Atlas、Figure做人形locomotion。两拨人用不同的hardware, 不同的simulator, 不同的learning method, 发不同的paper, 互相基本不引用。

这群人注意到一个事情: **human hand和human body在拓扑上其实是同构的**。你想想看, hand是什么？一个palm加5根分叉的fingers, 每根finger有一堆joints。body是什么？一个torso加2根arms加2根legs加一个head, 每根limb也有一堆joints。两者都是"compact central thing + 多条分叉articulated chain"。

这就是个fractal structure。如果接受这个类比, 那同一个actuator module既可以当finger joint用, 也可以当elbow或者knee用。同样的joint、同样的motor、同样的sensing、同样的control interface, 只要换个kinematic configuration就能扮演完全不同的role。

这就是morphological reuse。Morphology不再是robot的固有属性, 变成了一个可配置的参数。

## 硬件咋干的

27 DoF怎么分的？

**Hand形态**: 5个fingers × (1 abd-add + 3 flex-ext) = 20 DoF。结构贴近21-DoF human hand model。

**Humanoid形态**: 同样27个motor, 重组成:
- Head: 4 DoF (Module I, 也是thumb)
- 2 Arms: 各4 DoF (Modules II和V, 也是index和little finger)  
- 2 Legs: 各6 DoF (Modules III和IV, 也是middle和ring finger)
- Hip: 1 DoF (Module VII, 也是wrist)
- 2 Prismatic: 专门用来做形态切换的sliding joint

加起来 4+4+4+6+6+1+2 = 27。

切换机制很巧妙: Module VI (torso/palm那个) 里头藏了两个**rack-and-pinion linear guide**, joint 9和joint 26驱动。要从humanoid变hand的时候, 就把Modules II和V往下一推, 推到底它们就到了index finger和little finger的位置。硬件完全没换, 纯粹是物理位置translate了一下。

电控部分: 40mm×80mm的mainboard, vertically stacked设计, ESP32-S3做主控, STM32做power management, Dynamixel TTL bus驱动所有motor, Wi-Fi跟host通信。

一个特别elegant的细节: **指尖上装了IMU**。在hand形态下, 这是fingertip, IMU用来做contact sensing; 切到humanoid, 这两个fingertip正好就是脚趾位置, IMU用来做foot contact estimation。一个sensor, 两种任务, 完美reuse。

actuator选型也有讲究:
- XC330-T288-T: 紧凑型, 用在低负载的地方(head, arms, hip, sliding传动)
- XM430-W210-T: 高torque版, 用在leg的高负载关节(hip pitch, ankle pitch)
- 2XC430-W250-T: 双正交输出轴, 一颗motor同时输出两个DoF, 用在中指/无名指的abd-add+flex-ext, 也对应humanoid的hip-roll+knee-pitch

这个2XC430选得真聪明, 正好契合dual-embodiment需求, 省空间省重量。

## 控制栈怎么统一

paper的第二个核心是**unified control/learning stack**。同一套sensing、同一套communication、同一套actuator接口, 服务两种embodiment下的所有controller。

### Hand那边的3个controller

**1. Teleoperation**: Apple Vision Pro抓operator的22个hand keypoints, 用AnyTeleop (https://arxiv.org/abs/2307.04577)做retargeting。retargeting representation用fingertip-to-fingertip和fingertip-to-wrist的displacement vector, 不用绝对位置。intuition是: 你手在空中怎么晃不重要, 关键是手指相对几何关系。wrist运动映射到Franka arm的end-effector。

**2. Dexterous Grasping (Diffusion Policy)**:
- 100个demonstration (10物体×10 demos), 用VR teleop采集
- Object point cloud (512 points) + proprioception history
- PointNet++编码点云, MLP编码proprioception
- 两个feature拼接, 喂给Diffusion Policy (U-Net backbone)
- $n_{\text{obs}}=2$ (看2帧obs), $n_{\text{action}}=8$ (预测8帧action)
- Object pose用FoundationPose (https://arxiv.org/abs/2312.00776) 从RGB-D + mesh估计

Real-world成功率: 10个物体平均72%。Apple 8/10, Band Aid 6/10, Chip Tube 9/10, Sheep 9/10, Sprayer 5/10... 跟object geometry复杂度强相关, Sprayer那种带trigger的不规则形状最难。

**3. In-hand Reorientation (RL)**: IsaacLab + PPO, sim-to-real transfer, domain randomization做friction/actuator/object mass/initial state。Reward鼓励cube match target orientation, 加各种regularization。target orientation是按fixed rotation递进的, 一段一段小goal串起来变成长期reorientation。

### Humanoid那边的3个controller

**1. RL Tracking (reference-guided)**:
这套pipeline复杂了, 4层nested:

**Step 1 ZMP planner**: 给你一个gait cycle duration $T_{\text{cyc}}$ 和一个single-support vs double-support ratio $\rho$。算出:
- $T_{\text{ds}} = T_{\text{cyc}} / [2(\rho+1)]$ (double support phase duration)
- $T_{\text{ss}} = \rho \cdot T_{\text{ds}}$ (single support phase duration)

footstep locations决定ZMP waypoints, 跟contact schedule同步。

**Step 2 LIPM + LQR preview**: 这是Kajita 2003经典工作 (https://ieeexplore.ieee.org/document/1241826) 的核心。Linear Inverted Pendulum Model假设整个robot质量集中在CoM一点, 用无质量腿支撑, CoM高度恒定。

$$\mathbf{p}_{\text{zmp}} = \mathbf{c}_{xy} - \frac{h_{\text{com}}}{g} \ddot{\mathbf{c}}_{xy}$$

变量解释:
- $\mathbf{p}_{\text{zmp}}$: Zero Moment Point在水平面的位置。ZMP就是那个"如果地面反作用力作用在这一点, 不会产生任何moment"的特殊点。ZMP在support polygon里面, robot稳定; 出去就摔。
- $\mathbf{c}_{xy}$: CoM的水平位置 (3D CoM $\mathbf{c}$ 的xy分量)
- $h_{\text{com}}$: 假设恒定的CoM高度
- $g$: 重力加速度 (9.81 m/s²)
- $\ddot{\mathbf{c}}_{xy}$: CoM的水平加速度

这公式意思: CoM的水平加速度和ZMP位置之间有线性关系。$h_{\text{com}}/g$ 这个系数, 你可以理解为"倒立摆的characteristic time scale"——CoM越高、重力越小, 这个系数越大, robot越"懒"反应越慢。

LQR preview control: 你想要ZMP track desired ZMP轨迹, 同时CoM加速度effort要小。LQR优化一个cost function, 权重 $Q_y$ 控ZMP tracking, $R$ 控CoM加速度effort, 输出最优CoM trajectory。

**Step 3 Swing foot trajectory**: Stance foot不动, swing foot走smooth trajectory, 有user-specified的最大height。

**Step 4 Mink IK** (https://github.com/kevinzakka/mink): 对每个时间帧, 求解weighted differential IK。任务是:
- CoM位置 (权重适中)
- 双脚pose (stance foot权重高于swing foot, 不让stance foot乱动)
- Torso姿态stabilization
- 非腿关节往default posture正则

每帧warm-start from前一帧解, 跑12次IK iteration, 拿到leg joint angles。

**Step 5 RL tracking policy** (MuJoCo + PPO): 给定reference motion, 训一个closed-loop policy在MuJoCo里跟着reference跑, 同时handle真actuator dynamics和contact。

Reward公式 (1):
$$r_t^{\text{track}} = 0.5 r_t^{\text{root,pos}} + 0.5 r_t^{\\text{root,ori}} + r_t^{\text{body,pos}} + r_t^{\text{body,ori}} + r_t^{\text{body,lin}} + r_t^{\text{body,ang}}$$

6项tracking reward, 每项形式 $\exp(-\text{err}^2/\sigma^2)$, bounded在[0,1]。kernel widths:
- root position: $\sigma = 0.03$ m (严)
- root orientation: $\sigma = 0.15$ rad (松)
- body position: $\sigma = 0.03$ m (严)
- body orientation: $\sigma = 0.15$ rad (松)
- body linear vel: $\sigma = 0.25$ m/s (较松)
- body angular vel: $\sigma = 0.60$ rad/s (最松)

**Intuition**: position error很致命, 一点点偏移就让motion看起来不对劲, 所以$\sigma$小, reward衰减快; velocity error相对可以容忍, $\sigma$大。这就是reward shaping里的"对position严格, 对velocity宽容"原则。

Actor观测: reference joint pos/vel + base angular velocity + projected gravity + 5步历史 (joint pos, joint vel, prev action)。

Critic额外看privileged obs: reference-to-robot anchor offset + body pose in anchor frame + base linear/angular vel。asymmetric actor-critic, critic白嫖sim信息, actor部署要求低。

实验结果: joint-position tracking error 0.12 rad, body-position tracking error 0.0019 m。这个body position error极小, 说明policy几乎完美track住CoM轨迹。

**2. RL Velocity (reference-free)**:
不依赖任何reference motion, 直接接受high-level command (forward velocity + yaw rate), 让RL自己discover gait。

Reward公式 (2):
$$r_t^{\text{vel}} = w_v \exp\left(-\frac{\|\mathbf{v}_{xy,t} - \mathbf{v}_{xy,t}^d\|^2}{\sigma_v^2}\right) + w_\omega \exp\left(-\frac{(\omega_{z,t} - \omega_{z,t}^d)^2}{\sigma_\omega^2}\right) + r_t^{\text{reg}}$$

- $\mathbf{v}_{xy,t}$: measured CoM水平速度
- $\mathbf{v}_{xy,t}^d$: commanded水平速度
- $\omega_{z,t}$: measured yaw角速度
- $\omega_{z,t}^d$: commanded yaw角速度  
- $w_v = w_\omega = 2$ (tracking weight)
- $\sigma_v = 0.16$ m/s, $\sigma_\omega = 0.50$ rad/s (kernel widths)
- $r_t^{\text{reg}}$: 一大堆正则项, 包括upright保持、nominal posture、joint limit、action smooth、foot air time、foot slip reduction、soft landing

实验结果: command 0.20 m/s, 实际achieved velocity error 0.052 m/s, 即~74% command tracking。学出来的gait特征: 短步频高, 像小碎步。这个是RL-from-scratch典型gait pattern, 没有human reference priors时容易这样。

**Intuition**: tracking policy像"先有teacher演示再student模仿", velocity policy像"扔水里看能不能学会游泳"。前者sample efficient但被reference frame住, 后者general但exploration成本高。

**3. Keyframe Motion Control**:
Viser-based编辑器, 用户手动设joint-space keyframe和timestamp, piecewise-linear interpolation插出dense trajectory, Temporal Interval Manager可以交互调rhythm不调pose。

两种execution path:
- Direct: 直接把joint position stream到Dynamixel, 不调RL policy
- Reference: 把keyframe trajectory转成body states, 喂给tracking pipeline训练

walking用了6个keyframe, 相邻0.045s间隔。实测能做push-ups, pull-ups, pick-and-place, 前后walking, turning, sidestepping。

### Position command公式

公式 (3) 是actuator-level统一接口:
$$q_{j,t}^{\text{cmd}} = q_j^{\text{default}} + s_j a_{j,t}$$

- $j$: joint index
- $t$: timestep
- $q_{j,t}^{\text{cmd}}$: 命令给motor的位置 (rad)
- $q_j^{\text{default}}$: default posture下这个joint的位置
- $a_{j,t}$: policy输出的dimensionless action, 通常 [-1, 1]
- $s_j = 0.25$: action scale, 决定action的最大影响范围

**Intuition**: zero action → 保持default posture → stable pose。policy action是对default的扰动, $s_j$ 控制扰动幅度。这是sim2real常用的normalization trick, 让RL在bounded, zero-centered space训练, 部署稳定。

模拟Dynamixel: (position error, velocity) → PD torque, 加velocity-dependent torque limit, 模拟真motor的饱和特性。

## 实验三个问题

paper围绕3个核心question设计实验:

**Q1: 这重配置的hardware能不能像dedicated hand一样做精细操作？**

VR teleop实测: 能做grasping, hanging, stacking, pouring, deformable object interaction。Diffusion Policy在10个物体上平均72%成功率。In-hand reorientation也跑通了real robot。

**Q2: 这同一hardware能不能做stable humanoid locomotion？**

Tracking policy sim结果: 0.12 rad joint error, 0.0019 m body error——非常精确。Velocity policy: 0.20 m/s command下0.052 m/s error, 学出小碎步步态。Keyframe motion实测能跑push-ups, pull-ups, pick-and-place, walking, turning。

**Q3: 双形态切换能不能解锁单形态做不到的长horizon任务？**

最有意思的实验: 一个long-horizon task, 把整个系统串起来。

流程 (Figure 8):
1. Handroid装在Franka arm上当hand用, 切换到humanoid形态
2. 通过electromagnetic flange (大约180N holding force) 跟Franka arm脱离
3. Humanoid形态下走过去, 绕过obstacle, 走到box背面
4. 把box推回Franka workspace
5. 切换回hand形态
6. 重新dock到Franka arm
7. Hand形态grasp bottle并放进box

整个流程demonstrate了embodiment switching、locomotion、external-arm docking、dexterous manipulation在一个unified workflow里协调。这就是dual-embodiment的实际价值——单形态robot都做不到这种"我先走过去再变手抓东西"的任务。

## 我想到的延伸和联想

1. **Cross-embodiment policy transfer**: 既然同一套hardware两种形态, 能不能训一个shared trunk policy, 然后两套embodiment-specific heads？或者用hand形态学到的contact-rich manipulation priors帮助humanoid形态学loco-manipulation？反过来bipedal balance priors能不能transfer到in-hand reorientation (都是contact-rich + multi-DoF coordination)？这个方向有作者group自己的paper [8] (Wei et al., "One hand to rule them all", https://arxiv.org/abs/2602.16712) 在做canonical representations, 也有paper [68] Coordex (https://arxiv.org/abs/2606.23680) 在做body-hand coordination。

2. **Diffusion Policy + RL hybrid**: 现在hand形态下grasping用Diffusion Policy (imitation), in-hand reorientation用PPO (RL)。能不能unify？比如用diffusion生成action distribution, 用RL fine-tune, 类似Diffusion-QL那一套在robotics上的迁移。

3. **Tactile sensing缺失**: 这是个大限制。Fingertip tactile对contact-rich manipulation几乎是必须的。作者自己提到future work加tactile。他们group有paper [61] "Current as touch" (https://arxiv.org/abs/2607.03529) 用motor current做proprioceptive contact feedback, 不加sensor也能sense contact, 跨embodiment服务finger contact和foot contact都合适, 这条路线非常符合dual-embodiment哲学。

4. **Morphology as learnable parameter**: 现在是用户手动按按钮切换形态。能不能让policy自己决定什么时候切？比如学一个meta-policy, 在locomotion和manipulation之间自动切换embodiment。这需要把2-DoF prismatic joint也纳入action space, 让RL学会"什么时候unfold成humanoid, 什么时候fold回hand"。这个方向的cross-embodiment learning非常有意思。

5. **Desktop-scale的research democratization**: 0.33m, 2kg, open-source hardware和software。对比Unitree H1 ($$$), Berkeley Humanoid (中等成本), 这个平台cost可能就几百到几千刀 (Dynamixel开销), 任何一个有3D打印和basic electronics能力的lab都能搭起来。这跟LEAP Hand (https://arxiv.org/abs/2309.06493) 在dexterous hand领域的democratization作用类似, 但LEAP只覆盖hand, Handroid覆盖了hand+humanoid两个niche。

6. **Anchor link的设计**: paper提到Handroid用hip link作为motion anchor。这个设计选择有意思——为什么不直接用root/base？因为hand形态下"hip"就是wrist, 是天然的anchor point。这种跨embodiment的anchor link选择, 启发是: cross-embodiment learning需要找到morphology-invariant的reference frame, hip/wrist这种"central pivot"是natural choice。

7. **ZMP + RL的hybrid设计哲学**: 看tracking policy这个pipeline, ZMP planner生成reference, Mink IK解joint angles, 然后RL policy在sim里学closed-loop跟踪。这其实是"先验motion + RL robustness"的hybrid。ZMP是1980s控制理论, RL是2020s机器学习, 两者结合既sample efficient (有reference supervision) 又robust (能handle真dynamics)。这个pattern在很多humanoid locomotion paper里都见过, 比如ASAP (https://arxiv.org/abs/2502.01143)。

8. **Sim-to-real的关键是actuator model**: paper提到"simulated Dynamixel actuator model converts position error and measured joint velocity into PD torque subject to velocity-dependent torque limit"。这个细节很关键——sim-to-real成败往往不在RL算法, 而在actuator model多精确。Dynamixel的torque-speed curve是非线性的, 高速时torque drops, 这个如果不model, sim里学到的policy到真机就废。

9. **Long-horizon task的实验设计**: panels 1-16那个实验其实是个mini-Magic-101 of embodied AI: 形态切换(locomotion affordance) + environment navigation + tool use (推box) + precise manipulation (grasp瓶盖)。这种任务结构在cross-embodiment robotics里会越来越常见。

## 整体评价

Handroid不是个algorithm paper, 是个system + design paper。算法层面全是已有方法的组合 (Diffusion Policy, PPO, ZMP+LQR+LIPM, IK)。但value在:

1. **新paradigm**: morphology-reconfigurable robot, 把"embodiment"从一个固定的hardware property变成可配置的parameter
2. **新平台**: open-source, desktop-scale, accessible, 让cross-embodiment learning研究有了physical testbed
3. **新research direction**: unified control/learning/sensing across embodiments, 可能催生一系列后续工作

类比一下, 这工作在robotics里有点像LEAP Hand在dexterous manipulation里的位置——本身algorithm不复杂, 但提供了一个well-designed, accessible, open-source platform, 让整个community能在上面做后续research。LEAP Hand之后催生了一堆paper, Handroid如果catch on, 应该也能催生一波cross-embodiment learning, shared representation, unified control方向的工作。

唯一比较担心的是scale——0.33m的humanoid能做有限的任务, 真正complex loco-manipulation可能需要更大scale。但作为research platform验证concept, 这个scale反而恰到好处: 大到能做locomotion, 小到能放在桌面上跑, cheap enough to fail。

最后, paper的limitation部分作者自己也很诚实: cable干扰、actuator miniaturization、tactile缺失都是real issues。Future work提到wireless、miniaturized actuator、tactile sensor——这些方向任何一个做下去都是后续paper。

---

References我推荐深挖的:

- AnyTeleop (Qin et al., RSS 2023): https://arxiv.org/abs/2307.04577
- Diffusion Policy (Chi et al., IJRR 2025): https://arxiv.org/abs/2303.04137  
- OpenAI Rubik's cube (Andrychowicz et al., IJRR 2020): https://arxiv.org/abs/1910.07113
- ZMP preview control (Kajita et al., ICRA 2003): https://ieeexplore.ieee.org/document/1241826
- FoundationPose (Wen et al., CVPR 2024): https://arxiv.org/abs/2312.00776
- ASAP (He et al., 2025): https://arxiv.org/abs/2502.01143
- Isaac Lab: https://arxiv.org/abs/2511.04831
- LEAP Hand (Shaw et al., RSS 2023): https://arxiv.org/abs/2309.06493
- DexPilot (Handa et al., ICRA 2020): https://arxiv.org/abs/1909.06493

希望这版"人话"版本build你的intuition了, Andrej。如果有哪个具体细节还想往下挖, 我可以接着讲。

---

# Handroid: 双形态桌面机器人深度解析

Andrej, 这篇paper非常有意思，UNC Chapel Hill + Stanford的合作工作，核心insight非常elegant：用同一套27-DoF electromechanical body，通过模块重配置实现 dexterous hand 和 humanoid robot 两种截然不同的embodiment。这背后的哲学问题其实是 morphological reuse - 我们能否把 morphology 视作一种可重配置的属性，harness同一套 hardware/sensing/control backbone 服务多种任务空间。

项目主页：https://handroid.org

## 1. Core Insight: Morphological Homology

作者抓住的关键观察是：**human body 和 human hand 在 topology 上是 homologous 的**。两者都是 branching articulated chains 从 compact central structure 延伸出来：
- Hand: palm → 5 fingers (each finger: 1 abduction-adduction + 3 flexion-extension)
- Body: torso → head + 2 arms + 2 legs

这个analogy是整个design principle的根基。如果你接受这个abstraction，那么**同一套模块**（actuator + joint + link）可以在不同embodiment中扮演不同functional role：finger ↔ arm/leg, palm ↔ torso。这就是 morphological reuse 的硬件基础。

Handroid的具体mapping（见paper Figure 2）：

| Module | Dexterous Hand | Humanoid |
|--------|---------------|----------|
| I | Thumb | Head (4-DoF) |
| II | Index finger | Left arm (4-DoF) |
| III | Middle finger | Left leg (6-DoF) |
| IV | Ring finger | Right leg (6-DoF) |
| V | Little finger | Right arm (4-DoF) |
| VI | Palm | Torso (含 sliding mechanism) |
| VII | Wrist | Hip (1-DoF) |

总共 27 DoF，其中 25 个 articulated + 2 个 prismatic（专门用于 embodiment switching）。

## 2. Hardware Design 细节

### 2.1 Reconfiguration mechanism

关键工程难点在于：如何在不替换hardware的情况下实现 morphology switching？答案是 **rack-and-pinion transmission** 集成在 Module VI 内部，由 joint 9 和 joint 26 驱动。

具体工作流程：
- Humanoid → Hand：linear guide rail上的 Modules II 和 V 向下滑动到 hand-configuration 位置，分别作为 index finger 和 little finger
- Hand → Humanoid：反向运动

**Intuition**：reconfiguration 不改变任何 actuator 的物理参数，只是通过 linear translation 改变了 modules 在 kinematic tree 中的相对位置。这意味着同一套 control interface 在两种embodiment下都能reuse。

### 2.2 Actuator selection tradeoff

使用了三种 Dynamixel actuators，根据 load demand 分布：

| Actuator Type | 用途 | 特点 |
|--------------|------|------|
| XC330-T288-T | Modules I, II, V, VII + VI传动 | compact，低负载 |
| XM430-W210-T | Modules III, IV 高负载关节 | 高peak torque（hip pitch, ankle pitch） |
| 2XC430-W250-T | Modules III, IV | 双正交输出轴，提供abduction/adduction + flexion/extension |

注意 2XC430 这种 dual-axis actuator 是个聪明的选择，正好对应 middle/ring finger 需要 abduction-adduction + flexion-extension 两种 DoF，在 humanoid 形态下又对应 hip-roll + knee-pitch，省了空间和重量。

### 2.3 Electrical architecture

- **Mainboard footprint**: 40mm × 80mm（极其紧凑）
- **Stacked design**: 用 copper standoffs 既做mechanical support又做power connection，同时improve thermal dissipation
- **MCU 分工**：
  - ESP32-S3: 主controller，通过TTL bus驱动所有Dynamixel，Wi-Fi streaming到host
  - STM32: 专门做power management + battery temperature monitoring
- **Power**: 支持battery-powered 或 PD charger (up to 140W)，PD同时可以charge battery
- **Sensing**: 主IMU在mainboard上提供body orientation；指尖位置额外放IMU（在humanoid形态下对应feet）

指尖=脚趾的 IMU reuse 是个漂亮的细节：feet contact estimation 和 fingertip contact estimation 在 sensing requirement 上是同构的，所以一个 sensor 跨embodiment服务两个任务。

## 3. Control Stack: Unified Framework

这是 paper 的第二个核心贡献：一个 unified control + learning stack 同时服务两种 embodiment。所有 controller 共享同一套 actuator interface, proprioceptive channels, 和 communication protocol。

### 3.1 Dexterous Hand control

#### Teleoperation
- Apple Vision Pro 获取 operator 的 wrist pose + 22 hand keypoints
- AnyTeleop retargeting framework + DexPilot-style objective
- Retargeting representation：fingertip-to-fingertip pairwise displacement vectors + fingertip-to-wrist displacement vectors
  - **Intuition**：用相对几何关系而非绝对位置，可以decouple global hand motion（手腕晃动）from finger shape（手型），减少对global motion的sensitivity
- Franka arm 命令：记录初始 operator wrist pose 和 initial robot end-effector pose，做 relative mapping
- 运行频率：> 20 Hz

#### Dexterous Grasping (Diffusion Policy)
- 100 demonstrations (10 objects × 10 demos)
- Input: object point cloud $\mathbf{P}^O$ + proprioception history $\mathbf{S}_{t-n_{\text{obs}}+1:t}$
- Architecture:
  - PointNet++ 编码 point cloud
  - MLP 编码 proprioception history
  - 两个feature vector concatenate 后作为 condition input
  - Diffusion Policy with U-Net backbone
- $n_{\text{obs}} = 2$（observation chunk size）
- $n_{\text{action}} = 8$（action chunk size）
- Temporal ensembling 平滑轨迹

**Object pose estimation**：用 FoundationPose（Wen et al. CVPR 2024, https://arxiv.org/abs/2312.00776）从 RGB-D（RealSense L515）+ scanned mesh 估计 6D object pose，然后 transform mesh 到 robot frame，sample 512 surface points 作为 point cloud input。

#### In-hand Reorientation (RL)
- IsaacLab + PPO
- Sim-to-real randomization: contact friction, actuator parameters, object mass/scale, initial states
- Reward 结构：target orientation matching + 正则化（action magnitude, unstable motion, palm deviation）+ fall penalty
- 目标 orientation 通过 fixed rotation 递进，构成 incremental reorientation goals → continuous rotation emerges from sequence of local goals

### 3.2 Humanoid Control

这里有两条 RL 路线 + 一条 keyframe 路线。

#### RL Tracking Control (reference-guided)

Pipeline 复杂，分几步：

**Step 1: ZMP planner**
- 给定 gait-cycle duration $T_{\text{cyc}}$ 和 single-to-double support ratio $\rho$
- Phase durations:
  - $T_{\text{ds}} = T_{\text{cyc}} / [2(\rho+1)]$ (double support)
  - $T_{\text{ss}} = \rho \cdot T_{\text{ds}}$ (single support)
- 构造 ZMP waypoints 从 planned footstep locations，与 contact schedule 同步

**Step 2: LIPM (Linear Inverted Pendulum Model) + LQR preview control**

公式 (4) 是经典的 LIPM dynamics：
$$\mathbf{p}_{\text{zmp}} = \mathbf{c}_{xy} - \frac{h_{\text{com}}}{g} \ddot{\mathbf{c}}_{xy}$$

变量解释：
- $\mathbf{p}_{\text{zmp}}$：planar ZMP position（在 support polygon 内的点，net moment about 之为零）
- $\mathbf{c}_{xy}$：planar CoM position（$\mathbf{c}$ 是 3D CoM，下标 $xy$ 取水平分量）
- $h_{\text{com}}$：assumed constant CoM height
- $g$：gravitational acceleration ($\approx 9.81$ m/s²)
- $\ddot{\mathbf{c}}_{xy}$：planar CoM acceleration

**Intuition**：LIPM 假设所有质量集中在 CoM 一个点上，用一根无质量腿支撑。这个公式把 CoM dynamics 和 ZMP 联系起来——给定 ZMP trajectory，反解 CoM trajectory；给定 desired CoM trajectory，验证 ZMP 是否在 support polygon 内。$h_{\text{com}}/g$ 这个系数是 LIPM 的特征 "natural frequency" 参数，类似倒立摆的时间常数。

LQR preview control 平衡 ZMP tracking（weight $Q_y$）vs CoM acceleration effort（weight $R$），生成最优 CoM trajectory。

**Step 3: Swing foot trajectories**
- Stance foot 保持固定
- Swing foot 沿 smooth trajectory，user-specified maximum height

**Step 4: Mink IK**
对每个 planned frame 求解 weighted differential IK：
- CoM task
- Bilateral foot-pose tasks（stance foot 权重高于 swing foot）
- Torso stabilization task
- Posture regularization task（非腿关节向 default 正则化）
- 12 iterations per frame，warm-started from 前一帧解

Reference: Mink (Kevin Zakka), https://github.com/kevinzakka/mink

**Step 5: RL tracking policy (MuJoCo)**

公式 (1) 是 tracking reward：
$$r_t^{\text{track}} = 0.5 r_t^{\text{root,pos}} + 0.5 r_t^{\text{root,ori}} + r_t^{\text{body,pos}} + r_t^{\text{body,ori}} + r_t^{\text{body,lin}} + r_t^{\text{body,ang}}$$

各项是 exponential kernel 应用于 squared tracking error：
- $r_t^{\text{root,pos}}$：root position tracking，kernel width 0.03 m
- $r_t^{\text{root,ori}}$：root orientation tracking，kernel width 0.15 rad
- $r_t^{\text{body,pos}}$：body position tracking，kernel width 0.03 m
- $r_t^{\text{body,ori}}$：body orientation tracking，kernel width 0.15 rad
- $r_t^{\text{body,lin}}$：body linear velocity tracking，kernel width 0.25 m/s
- $r_t^{\text{body,ang}}$：body angular velocity tracking，kernel width 0.60 rad/s

**Intuition**：每个 tracking term 形式都是 $\exp(-\text{err}^2 / \sigma^2)$，这是一种 bounded reward（最大值1），同时 kernel width $\sigma$ 控制了 sensitivity。$\sigma$ 小（如 0.03 m）意味着对position error非常敏感，$\sigma$ 大（如 0.60 rad/s）意味着对angular velocity error较宽容。这种设计反映了：position error 更致命（cosmetic小偏差会让motion看起来不自然），而velocity error允许较大tolerance。

**Anchor**：Handroid 用 hip link 作为 motion anchor，所以 root terms 在 hip frame 评估。

**Observation**:
- Actor: reference joint pos/vel + measured base angular velocity + projected gravity + 5-step histories of joint pos, joint vel, previous actions
- Critic (privileged): + reference-to-robot anchor offsets + body poses in anchor frame + base linear/angular vel

**Controlled joints**: hip-yaw + bilateral hip-pitch, knee-pitch, ankle-pitch, ankle-roll（5 joints per leg × 2 = 10 + 2 hip-yaw = 12 lower-body joints）

**Termination**:
- Reference motion ends
- Vertical hip tracking error > 0.04 m
- |ref - sim vertical projected-gravity| > 0.5
- Monitored hand/foot vertical tracking error > 0.08 m

#### RL Velocity Control (reference-free)

公式 (2)：
$$r_t^{\text{vel}} = w_v \exp\left(-\frac{\|\mathbf{v}_{xy,t} - \mathbf{v}_{xy,t}^d\|^2}{\sigma_v^2}\right) + w_\omega \exp\left(-\frac{(\omega_{z,t} - \omega_{z,t}^d)^2}{\sigma_\omega^2}\right) + r_t^{\text{reg}}$$

变量：
- $\mathbf{v}_{xy,t}$：measured planar CoM velocity
- $\mathbf{v}_{xy,t}^d$：commanded planar CoM velocity
- $\omega_{z,t}$：measured yaw rate
- $\omega_{z,t}^d$：commanded yaw rate
- $w_v = w_\omega = 2$（tracking term weights）
- $\sigma_v = 0.16$ m/s, $\sigma_\omega = 0.50$ rad/s（kernel widths）
- $r_t^{\text{reg}}$：正则项集合（upright orientation, nominal posture, body angular velocity suppression, joint limit avoidance, action smoothing, foot air time regulation, swing foot height/clearance, foot slip reduction, soft landing）

**Intuition**：tracking control 跟随一条 predefined reference motion，所有时间步都对齐；velocity control 只接受 high-level command（前向速度 + 转向角速度），让 policy 自己 discover gait。后者更 general 但更难学。

**Asymmetric actor-critic**：critic 额外看到 privileged observations（base linear/angular vel, projected gravity, foot heights/air times, contact states/forces）—— 这些在 deployment 时无法直接测量或难测量，但训练时在 sim 里免费获得。这种asymmetric design 让 critic 评估更准确，actor 部署要求降低。

#### Keyframe motion control
- Viser-based Keyframe Editor
- 用户指定 joint-space keyframes + timestamps
- Piecewise-linear interpolation 生成 dense trajectory
- Temporal Interval Manager 调整 timestamps 不动 key poses，interactive 调 rhythm
- 两条 execution path：
  - Direct hardware execution：stream 到 Dynamixel position-command interface
  - RL tracking reference：通过 forward kinematics + finite differences 转换成 joint+body states 喂给 tracking pipeline

## 4. Position Command Interface

公式 (3) 是关键的统一接口：
$$q_{j,t}^{\text{cmd}} = q_j^{\text{default}} + s_j a_{j,t}$$

变量：
- $j$：controlled lower-body joint index
- $t$：control step
- $q_{j,t}^{\text{cmd}}$：commanded joint position（输出给actuator）
- $q_j^{\text{default}}$：default joint position（nominal posture）
- $a_{j,t}$：dimensionless policy action（RL policy 输出，通常 [-1, 1]）
- $s_j = 0.25$：action scale（rad）

**Intuition**：这种 "default + offset" 的设计有几个好处：
1. RL action space 是 zero-centered dimensionless，stable 训练
2. $s_j$ 控制 action 影响幅度，越大学得越激进
3. 在 zero-action 时 robot 保持 default posture，这本身是 stable pose
4. 模拟的 Dynamixel actuator 把 (position error, measured velocity) 转成 PD torque，加 velocity-dependent torque limit

## 5. Experiments 详解

### 5.1 Q1: Dexterous manipulation

**Grasping 实验**：
10 个不同形状/size的物体，10 demonstrations per object = 100 demos 总共。

Table 1 数据：
| Object | Success |
|--------|---------|
| Apple | 8/10 |
| Band Aid | 6/10 |
| Canister | 8/10 |
| Chip Tube | 9/10 |
| Cocoa Box | 7/10 |
| Earphone | 6/10 |
| Glove Box | 7/10 |
| Sheep | 9/10 |
| Sprayer | 5/10 |
| WD-40 | 7/10 |

平均 72%。

**In-hand reorientation**：
- Handroid 装在 Franka Research 3 上，palm 朝上
- 3D printed cube，尺寸和 sim 中一致
- 30 Hz deployment
- Manual 初始化 cube position（不要求精确）
- Proprioceptive observation history 同 sim

### 5.2 Q2: Humanoid locomotion

**Simulated RL tracking**:
- Joint-position tracking error: 0.12 rad
- Body-position tracking error: 0.0019 m

**Simulated RL velocity control**:
- Commanded forward velocity: 0.20 m/s
- Achieved velocity-tracking error: 0.052 m/s
- Qualitative：short steps + high stepping frequency（典型 RL-discovered gait pattern）

**Real-world keyframe motions**:
- Walking: 6 manually authored keyframes，相邻 keyframe 间隔 0.045s
- Push-ups, pull-ups, pick-and-place（金属handle放入target box）
- Forward/backward walking, turning, sidestepping

### 5.3 Q3: Long-horizon dual-embodiment task

这个实验最有意思，demonstrate 的是 embodiment switching 的实际价值。流程（Figure 8）：

1. **Panels 1-4**: Handroid 从 Hand → Humanoid 切换，通过 electromagnetic flange 从 Franka arm 脱离
2. **Panels 4-8**: Humanoid 形态下前进、转弯、绕过障碍、到达 box 背后、push box 进入 Franka workspace
3. **Panels 9-12**: 从 Humanoid → Hand 切换，重新 dock Franka arm
4. **Panels 13-16**: Hand 形态下 grasp bottle 并放入 box

**Electromagnetic flange**：约 180 N holding force，无需 mechanical locking，对 alignment 要求低，适合 frequent docking/detachment。

## 6. Key Insights & Limitations

### 6.1 为什么这个工作有价值

1. **Morphological reuse 是新视角**：之前 dexterous hand 和 humanoid 是完全分离的研究 community，hardware、control、learning 都各自 optimize。Handroid 把它们 unify 到一个 platform，让 cross-embodiment learning 有了 physical testbed。

2. **Desktop scale 让 research accessible**：0.33m, 2.05kg 的robot任何人都能在桌面上跑，对比 full-size humanoid（Unitree H1, Atlas, Figure）的实验成本和 risk 极高。

3. **Sensing reuse 的 elegance**：指尖 IMU = 脚趾 IMU，同样的 proprioceptive interface 服务 manipulation contact sensing 和 locomotion contact estimation。

### 6.2 Limitations（作者自己也承认）

1. **Cable 干扰**：作者提到 future wireless operation 可以减少 cable-induced disturbances
2. **Actuator miniaturization**：进一步 miniaturize 可以让 Dexterous-Hand embodiment 更精致（目前20 DoF 还是不够 anthropomorphic）
3. **Tactile sensing 缺失**：fingertip camera 和 tactile sensor 都没有，这对 contact-rich manipulation 是大限制

### 6.3 我（model）想到的延伸

1. **Cross-embodiment policy transfer**：既然同一套 hardware 跑两种 morphology，能不能训一个 policy 在两种形态间共享 representations？例如用 hand上学到的 contact-rich skill 帮助 humanoid 学 loco-manipulation。这个方向有 paper [8] (Wei et al., "One hand to rule them all") 提到 canonical representations，可能值得follow。

2. **Module reuse → cross-robot transfer**：如果模块标准化到一定程度，未来可以 hot-swap 单个 module 实现 morphology customization，类似 modular robotics 但保留 learning ecosystem。

3. **Tactile + cross-embodiment**：作者提到 future work 加 tactile sensors。这其实和 paper [61] (Ma et al., "Current as touch") 的思路一致——用 motor current 做 proprioceptive contact feedback。在 humanoid 形态下，foot contact current signal 可以做 ground contact estimation；在 hand 形态下， fingertip current 可以做 grasp force regulation。一个 sensing modality 跨 embodiment 服务两种 task。

4. **关于 paper [68] (Coordex) 和 [69] (Anybody)**：作者 group 之前的工作已经在 humanoid loco-manipulation 上有积累，Handroid 是把这些 components 整合到一个 hardware platform 上。

5. **ZMP + RL 的 hybrid 设计**：注意 tracking policy 用 ZMP planner 生成 reference，再让 RL policy 学习 follow。这种 hybrid 比 pure ZMP 更鲁棒（能 handle sim dynamics + actuator dynamics + contact），又比 pure RL-from-scratch 更 sample efficient（有 reference motion 提供 supervision signal）。

## 7. References 我觉得值得深挖

- [44] AnyTeleop (Qin et al., RSS 2023): https://arxiv.org/abs/2307.04577 - teleop framework
- [48] Diffusion Policy (Chi et al., IJRR 2025): https://arxiv.org/abs/2303.04137 - 核心learning method
- [58] OpenAI Rubik's cube (Andrychowicz et al., IJRR 2020): https://arxiv.org/abs/1910.07113 - in-hand manipulation 经典
- [71] ZMP preview control (Kajita et al., ICRA 2003): https://ieeexplore.ieee.org/document/1241826 - humanoid locomotion 经典
- [74] FoundationPose (Wen et al., CVPR 2024): https://arxiv.org/abs/2312.00776 - 6D pose estimation
- [77] Isaac Lab: https://arxiv.org/abs/2511.04831 - NVIDIA 的 sim framework
- Mink (Kevin Zakka): https://github.com/kevinzakka/mink - IK library
- 项目主页：https://handroid.org

## 8. 整体评价

这篇 paper 的 contribution 主要在 hardware + system integration，learning method 上没有新算法（都是已有方法的组合：Diffusion Policy, PPO, ZMP+LQR+LIPM, IK）。但它的价值在于提出了一个 **morphology-reconfigurable** 的新 paradigm，并且做了 open-source（hardware design + control stack）。对于 robot learning community 来说，这是一个新的研究平台，可以验证 cross-embodiment transfer, shared representation, unified control 等 hypothesis。

这种工作让我想到 MIT Biomimetic Robotics Lab 的 Cheetah 系列在 quadruped 上的影响——一个 well-designed open platform 能催生大量后续 research。Handroid 在 dual-embodiment 维度上做了同样的事情。如果 community 真的开始用这个 platform 跑 cross-embodiment learning 实验，会有意思的发现：例如 learned finger coordination priors 能否 transfer 到 leg coordination，或者 learned bipedal balance 能否帮助 in-hand cube reorientation（因为两者都涉及 contact-rich dynamics + multi-DoF coordination）。
