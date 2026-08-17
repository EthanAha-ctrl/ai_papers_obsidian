---
source_pdf: ToddlerBot Open-Source ML-Compatible.pdf
paper_sha256: 4525a90e6e3d311cd13f15c3b89622c244123165086808f450e2db6a11cd3cf0
processed_at: '2026-08-12T16:32:27-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

这篇 paper 如果用一句话总结，就是 Stanford 造了一个给 AI 研究者玩的“乐高版”人类机器人。它便宜、好修、开源，而且专门为收集机器学习数据量身定制。

下面我用最直觉的方式拆解一下它的精髓。

## 1. 为什么我们需要这么个小东西？

你看现在市面上的人形机器人，Boston Dynamics Atlas、Tesla Optimus、Unitree H1，个头大、力气足、极其精密。但是，如果你是一个搞 AI 和 RL 的研究员，你想拿它们来训模型，会遇到一堆烦人的问题：

- **太贵、太娇气**：搞坏一个零件可能要几万美金，还得等原厂配件。
- **太危险**：一百多斤的铁疙瘩在实验室里摔倒，可能会把人砸进医院，所以测试时要在天花板上装吊车保护它。
- **黑盒太多**：很多时候你不知道它底层的电机控制逻辑是怎么写的，出了 bug 没法调。

所以，Stanford 团队就说：**去他的大块头，我们做一个微型版！**

ToddlerBot 只有 0.56 米高，3.4 公斤重，跟个两三岁小孩差不多。你把它放在桌子上，就算它劈叉摔倒了，顶多也就是摔个塑料件，3D 打印机印个新的，十几分钟换上接着干。最关键的是，它便宜到令人发指，全套成本不到 6000 美金，其中 90% 的钱还是花在买电机和计算芯片（Jetson Orin）上。

而且，他们把**“可复现性”当成死命令**。只要有台一千来块钱的桌面级 3D 打印机，看着他们开源的教程，一个没摸过硬件的计算机系学生，花三天就能把它拼出来。这对搞研究来说太爽了，意味着你可以在实验室里放一排，搞大规模并行实验。

## 2. 为什么塑料壳子也够硬？

你可能会问：3D 打印的塑料，能撑住机器人做运动吗？

这里有个非常 Karpathy-style 的物理直觉。想象你拿一根塑料尺子，如果是一米长，你一掰就弯了；但如果只有一厘米长，你拿钳子都很难掰弯它。

物理公式里，材料的形变程度跟长度的三次方成正比，而它的截面惯性矩（抵抗弯曲的能力）跟长度的四次方成正比。综合下来，相对形变率 $\frac{\delta}{L}$ 是跟长度的平方成反比的。

也就是说：**尺寸越小，它本身就越硬！** 所以，虽然 ToddlerBot 用的是塑料，但因为只有半米高，它的相对强度足以媲美用铝合金做的全尺寸机器人。这就是用“缩小尺寸”换取“材料降级”的巧妙杠杆。

## 3. 让 AI 训练爽到飞起的“数字孪生”

搞机器人 AI 最头疼的就是“Sim-to-Real Gap”——你在电脑模拟器里把机器人训得像武侠小说里的绝世高手，一放到真实世界的机器人上，它立马变成帕金森患者。

为什么会这样？因为模拟器里的电机是理想电机，现实里的电机有摩擦、有齿轮间隙、有发热掉速。为了解决这个，ToddlerBot 团队搞了一套极其硬核的“电机系统辨识”。

直觉是这样的：他们做了一个测试台，专门逼着电机在各种速度下转，然后拿传感器量它到底有多大阻力。通过这些数据，他们拟合出了一个非常精准的电机数学模型。这个模型包含了 9 个参数，比如：
- **摩擦力**：电机刚要动时的那股涩劲。
- **阻尼**：转得越快，阻力越大的效应。
- **力矩衰减**：电机转得越快，能输出的力气越小。

他们把这些参数写进 MuJoCo 模拟器里。结果就是，你在模拟器里生成的走路动作，直接放到真机身上，**零样本直接跑通**。不需要在真机上再微调！这在以前是很难想象的。

更妙的是，同型号的 Dynamixel 电机，参数高度一致。这意味着你在 A 机器人上训好的策略，可以直接拷给 B 机器人跑，完全不用重新调参。

## 4. 像提线木偶一样教机器人干活

光会走还不够，我们还要教它用手干活。怎么教？模仿学习。

你给它做一个一模一样的上半身当“手柄”（Leader arms），你抓着这个手柄动，它就跟着动。同时，你用个游戏手柄控制它的下半身走路。通过这种方式，你可以非常直观地“手把手”教它怎么抓东西。

这里有个极其聪明的细节：当你教它抓东西时，如果还要分心去维持它下半身的平衡，那太累了，而且收集的数据也不干净。于是他们在下半身加了一个**两层 PD 控制器**：
1. 保证它的重心始终在双脚中间。
2. 保证它的躯干始终是竖直的。

这样一来，人类操作者只需要专注于上半身的抓取动作，机器人自己会稳住底盘。他们用这个方法，20 分钟就能采集 60 条高质量的演示数据。然后拿这些数据去训 Diffusion Policy，成功率高达 90%。

## 5. 这玩意儿到底能干啥？

他们在 paper 里秀了一把肌肉：
- **举重**：能举起自身体重 40% 的东西。
- **俯卧撑、引体向上**：纯靠模拟器里设计的关键帧动作，直接迁移到真机执行，这证明了机械结构的精度和仿真的保真度。
- **双手协作**：把地上的玩具捡起来放进小推车。
- **长线任务**：两个 ToddlerBot 配合，一个推车，一个捡东西，收拾满地玩具。

## 6. 最大的 Intuition 收获

看这篇 paper，我最大的感触是：**做 AI 研究，硬件不该是瓶颈，数据基础设施才是。**

ToddlerBot 把自己定位成一个“数据采集机器”，这是很超前的视角。它不是要跟 Optimus 去比谁力气大、谁能进工厂拧螺丝，它是要放在大学实验室的桌子上，让研究员每天生成成千上万条高质量的视觉运动数据。

还有一个很赞的设计是“技能串联”。比如要推小车，先用 Diffusion Policy 抓住车把手，然后无缝切换到 RL 策略往前走。为了让两个策略能接上，他们在训走路策略时，特意从抓把手的数据里采样机器人的初始姿态。这种“让状态分布重叠”的工程小技巧，比硬生生搞一个端到端的巨型模型要实用得多。

总之，ToddlerBot 是一个把“让 AI 研究员爽”作为第一设计原则的产物。它降低了人形机器人研究的门槛，哪怕你没有几百万美金的经费，也能在这个平台上折腾最前沿的 Loco-Manipulation（移动操作）算法。

**参考链接：**
- 项目主页 (含开源代码和组装教程): https://toddlerbot.github.io
- MuJoCo (他们用的物理模拟器): https://mujoco.readthedocs.io
- Diffusion Policy (他们用的模仿学习算法): https://diffusion-policy.cs.columbia.edu/

---

# ToddlerBot: Open-Source ML-Compatible Humanoid Platform 深度解析

## 1. Core Thesis 与设计哲学

这篇paper 来自 Stanford (Shuran Song + C. Karen Liu labs),核心 thesis 非常 Karpathy-friendly: **conventional robot hardware design optimized for actuator strength / sensor accuracy / mechanical precision,但是对 ML-driven robotics research 来说这些都不是 key bottleneck**。真正稀缺的是 (1) affordability + rapid repairability,(2) full-stack ownership without black boxes,(3) innate ability to collect both simulation AND real-world data。

ToddlerBot 把 reproducibility 当成 **hard constraint**(不是 design objective),把 capability 和 ML-compatibility 当成 design objectives。这个 priority ordering 很关键——它解释了为什么选 3D-printed plastic 而不是 metal,选 Dynamixel servo 而不是 custom BLDC,选 0.56m / 3.4kg 而不是 full-scale。

项目主页: https://toddlerbot.github.io

---

## 2. 与现有 Humanoid 平台的定位对比

Table 1 是这篇 paper 的 positioning 核心。让我抽取几个关键维度:

| Platform | Size (m) | Weight (kg) | Compute (TFLOPS) | Active DoFs | Price ($) |
|---|---|---|---|---|---|
| BD Atlas | 1.50 | 89.0 | - | 28 | - |
| Unitree G1 | 1.32 | 35.0 | 2.50 | 29 | 57K |
| Unitree H1 | 1.76 | 47.0 | 1.92 | 19 | 70K |
| Berkeley Humanoid Lite | 0.80 | 16.0 | 0.29 | 22 | 5K |
| BRUCE | 0.70 | 4.8 | 0.1 | 16 | 6.5K |
| NAO H25 | 0.57 | 5.2 | 0.02 | 23 | 14K |
| K-Scale Zeroth | 0.48 | 3.6 | 0.01 | 16 | 1.4K |
| **ToddlerBot (Ours)** | **0.56** | **3.4** | **2.50** | **30** | **6K** |
| Average Adult | 1.73 | 70.9 | - | 32 | - |

几个 observation:
- ToddlerBot 是 **miniature humanoids 中唯一达到 30 DoFs 的** (human body functional approximation 是 32 revolute joints: 6/leg + 7/arm + 3/waist + 3/neck)。NAO H25 / OP3 / Zeroth / BRUCE 这些 prior miniature humanoids 都被 DoF count 限制死了。
- Compute 这一项 ToddlerBot 和 Unitree G1 持平 (Jetson Orin NX 16GB = 2.5 TFLOPS FP32),远超同级 miniature (Zeroth 0.01, BRUCE 0.1)。这对 onboard 跑 diffusion policy 很关键。
- Price $6K,90% 花在 motors + computer 上。

Reference 比较:
- Berkeley Humanoid Lite: https://humanoid-ai.github.io/
- Unitree G1: https://www.unitree.com/g1/
- BRUCE: https://bruce-robotics.github.io/
- K-Scale Zeroth: https://www.kscale.dev/

---

## 3. Mechatronic Design 细节

### 3.1 DoF 分布 (Figure 2)

- **7 DoFs / arm**: shoulder (3) + elbow (2) + wrist (2),用 spur gears 做 axis-aligned transmission
- **6 DoFs / leg**: hip (3) + knee (1, parallel linkage) + ankle (2)
- **2 DoFs neck**: pitch 用 parallel linkage (motor 藏在 head 里),yaw 直接驱动
- **2 DoFs waist**: yaw + roll,用 coupled bevel gears 让两个同 orientation 的 motor 驱动两个 perpendicular axes

### 3.2 三种 transmission primitives (Appendix 8.4, Figure 7)

这是一个 mechanical design 很 elegant 的部分:

**Spur gears** 提供 3 个 advantage:
1. Relocated joint axis (1:1 ratio 把 axis 移到 in-plane 更方便的位置,arm 用得多)
2. Torque modification (ratioed gears 调 final torque output,gripper 用)
3. Load distribution (Dynamixel XC330 output shaft 只有 Teflon bushing 支撑,1:1 spur gear 把 load 转移到 reinforced secondary axis + planar bearings,保护 motor 免受 transverse force,hip yaw 用)

**Coupled bevel gears** 用在 waist:
1. Rotated joint axis (两个同方向 motor 驱动两个 perpendicular DoFs)
2. Combined torque output (每个 axis 两个 motor 同时出力,单个 XC330 不够 drive 整个 upper body)
3. Compact actuation (space-constrained region 塞两个 DoF)

**Parallel linkages** 用在 knee 和 neck pitch:
1. Compact design (neck motor 藏 head 内部)
2. Reduced inertia (knee motor 放高一点)
3. Structural efficiency (knee motor bolt 到 3D-printed structure,load distribution 更好)

在 MuJoCo 里的 simulation modeling: spur gears 用 joint equality constraints,coupled bevel gears 用 fixed tendons,parallel linkages 用 weld constraints。Empirically sim2real gap 很小。

### 3.3 为什么 3D printing 可行 (Appendix 8.1)

这是非常 Karpathy-style 的 scaling argument。从 Euler-Bernoulli beam theory:

$$\delta = \frac{P L^3}{3 E I}$$

其中 $\delta$ 是 tip deflection,$P$ 是 tip point load,$L$ 是 characteristic length,$E$ 是 material elastic modulus,$I$ 是 second moment of area。

Relative deflection:

$$\frac{\delta}{L} \propto \frac{P L^2}{3 E I}$$

Since $I \propto L^4$ (cross-section 也 scale):

$$\frac{\delta}{L} \propto \frac{P}{3 E L^2}$$

**关键 insight**: relative deflection $\frac{\delta}{L}$ 随 $L^2$ 减小而 quadratic 地减小。所以 miniature (small $L$) + 3D-printed plastic (smaller $E$) 可以达到和 full-size aluminum 相当的 strength。这是一个 dimensional analysis 的精彩应用——size 本身就是 strength 的一部分。

---

## 4. Power Factor $\tilde{p}$:一个值得讨论的 metric (Appendix 8.5)

paper 提出一个新的 humanoid capability metric,我觉得很值得讨论。先看 derivation。

### 4.1 出发点:scale-invariant performance comparison

要让一个 1.8m humanoid 和一个 0.5m humanoid "performance 相同",定义是:它们执行 **same sequence of joint motions** over time span $T$,且 total power consumption 是各自 motor 最大功率的相同 fraction:

$$\frac{\int_0^T p(t) dt}{\sum_{i=0}^{N} |\tau_i^{\max} \dot{q}_i|} \approx \frac{\Delta h \cdot m g}{\sum_{i=0}^{N} |\tau_i^{\max} \dot{q}_i|} \approx \frac{h \cdot m g}{\sum_{i=0}^{N} |\tau_i^{\max} \dot{q}_i|}$$

变量解释:
- $p(t)$: instantaneous power output at time $t$
- $\tau_i^{\max}$: 第 $i$ 个 motor 的 maximum torque
- $\dot{q}_i$: 第 $i$ 个 motor 的 joint velocity (取 absolute value 保证 motor 都做正功)
- $N$: motor 总数
- $\Delta h \cdot m g$: gravitational energy gained,approximate 成 humanoid height $h$ 乘 weight $mg$ (因为 $\Delta h$ 与 $h$ proportional)
- $m$: humanoid mass
- $g$: gravity

### 4.2 Power factor 定义

把上面 fraction 翻过来,drop 掉 $\dot{q}$ (因为 same sequence of motion implies same $\dot{q}$ sequence):

$$\tilde{p} = \frac{\sum_{i=0}^{N} |\tau_i^{\max}|}{h \cdot m g}$$

物理含义: **total available torque / (height × weight × gravity)**,即 humanoid 能够产生的总扭矩相对于其 "gravity moment" 的比值。

### 4.3 为什么不 normalize by $N$ (DoF count)

paper 直接 critique 了 Berkeley Humanoid Lite [20] 的 metric。考虑极端 case:两个 humanoid 同 height 同 weight,一个 1 DoF,一个 100 DoFs (equally capable)。如果 normalize by $N$,它们 power factor 相同,这显然不 reflect 真实 actuation capacity。

ToddlerBot 的 $\tilde{p}$ score 是所有 humanoid 中最接近 human 的 (Figure 8)——既不超过太多 (会导致 unnatural motion / battery drain / safety risk),也不低于 human threshold。

Reference: Berkeley Humanoid Lite paper https://arxiv.org/abs/2504.05002

---

## 5. Digital Twin: ML-Compatibility 的基石

这是 paper 最 engineering-heavy 的部分。Digital twin 分两块: zero-point calibration (kinematics) + motor sysID (dynamics)。

### 5.1 Zero-point Calibration (Appendix 8.8, Figure 10)

Dynamixel motors **没有 absolute zero point**(只有 relative encoder),每次 reassembly 后都需要校准。ToddlerBot 设计了一套 3D-printed plug-and-play calibration devices:
- Orange: arm
- Yellow: neck
- Red: hip
- Beige: ankle

插入后 "click into place" 就固定 zero point,1 分钟内完成。定义的 zero pose 是 "standing with both arms parallel to body"。

### 5.2 Motor System Identification Pipeline (Appendix 8.9, 8.10)

这是 paper 最 technical 的部分之一。

**MuJoCo actuator model** 三个核心参数:
- `frictionloss` $\tau_f$: 让 actuator 开始动的 minimum torque (Nm)
- `damping` $d$: backdrive resistance 随 speed 增长的 rate (Nms/rad)
- `armature` $I$: effective rotor inertia (包含 gearbox) (kgm²)

Resistance torque:

$$\tau_r = \tau_f + d \cdot \dot{q}$$

其中 $\dot{q}$ 是 joint velocity。

**测量 protocol**:
1. Damping + friction loss: backdrive motor at constant RPM,torque sensor 记录 resisting torque,linear fit 到 torque-speed data。Intercept = friction loss,slope = damping。
2. Armature inertia: motor free-spin,cut power,观察 spin-down。numerically integrate resistance power 得到 initial stored energy $E$。然后:

$$E = \frac{1}{2} I \omega^2 \implies I = \frac{2E}{\omega^2}$$

其中 $\omega$ 是 initial angular velocity。

### 5.3 Actuation Model (Appendix 8.10)

这是受 Grandia et al. [42] 启发的 actuator model,但做了几个重要 extension。

**PD position control**:

$$\tau_m = k_p (\hat{q} - q) - (k_d^{\min} + k_d) \dot{q}$$

变量:
- $\tau_m$: motor command torque
- $k_p$: proportional gain (Dynamixel 的 unitless $k_p$ 转换到 physical $k_p$ 的 factor ≈ 150)
- $k_d$: derivative gain (user-settable)
- $k_d^{\min}$: **新发现的 parameter**——motor powered on 时即使 $k_d = 0$ 也有 significant additional damping,建模为单独的 active damping
- $\hat{q}$: joint position setpoint
- $q$: actual joint position
- $\dot{q}$: joint velocity

**Velocity-dependent torque limit** (piecewise linear):

$$\tau_{\text{limit}} = \begin{cases} \tau_{\max}, & |\dot{q}| \leq \dot{q}_{\tau_{\max}} \\ \frac{\dot{q}_{\max} - |\dot{q}|}{\dot{q}_{\max} - \dot{q}_{\tau_{\max}}} \cdot \tau_{\max}, & \dot{q}_{\tau_{\max}} < |\dot{q}| \leq \dot{q}_{\max} \\ 0, & |\dot{q}| > \dot{q}_{\max} \end{cases}$$

变量:
- $\tau_{\max}$: low-velocity constant torque limit
- $\dot{q}_{\tau_{\max}}$: 超过此 velocity 后 torque 开始 linear 衰减
- $\dot{q}_{\max}$: 此 velocity 时 torque 降到 0 (max no-load speed)

**和 Grandia 原版的关键区别**: paper 把 **braking torque limit $\tau_{\text{brake}}$** 和 acceleration torque limit 分开,因为 motor 通常能提供更高 braking torque (passive resistance + gearbox inefficiency 帮忙)。

**Passive-active ratio**: 当 motor 被 external torque back-driven 时,$k_p$ 显著增大,因为 external torque 被 gearbox inefficiency 消耗。建模为 $\frac{1}{\eta^2}$,$\eta$ 是 gearbox efficiency。Empirically determined 为 3 (即 $\eta \approx 58\%$)。

**Final joint torque** (带方向依赖的 clamping):

$$\tau = \begin{cases} \text{clamp}_{[-\tau_{\max}, \tau_{\text{brake}}]}(\tau_m) - \tau_r, & \dot{q} \geq 0 \\ \text{clamp}_{[-\tau_{\text{brake}}, \tau_{\max}]}(\tau_m) + \tau_r, & \dot{q} < 0 \end{cases}$$

总共 9 个 parameters per motor type: damping, frictionloss, armature, $\tau_{\max}$, $\dot{q}_{\tau_{\max}}$, $\tau_{\dot{q}_{\max}}$, $\dot{q}_{\max}$, $k_d^{\min}$, $\tau_{\text{brake}}$。Table 3 给出了所有 Dynamixel motor 的 fitted values。

**关键 finding**: 同 model 的 Dynamixel motors 有 nearly identical dynamics parameters。这是 reproducibility 的基石——第二个 ToddlerBot instance 不需要重新 sysID,policy 直接 zero-shot transfer。

Final simulation tracking error: 平均 $1.3°$。

Reference: Grandia et al. RSS 2024 "Design and Control of a Bipedal Robotic Character" https://robotics.sciencemag.org/content/robotics (实际是 RSS 2024)

---

## 6. RL Walking Policy (Section 4.2, Appendix 8.12)

### 6.1 State & Action

Policy $\pi(a_t | s_t)$ 输出 joint position setpoints 给 PD controller。

State:

$$s_t = (\phi_t, c_t, \Delta q_t, \dot{q}_t, a_{t-1}, \theta_t, \omega_t)$$

变量逐一解释:
- $\phi_t$: **phase signal** (周期性 gait phase,编码 walking cycle 中的位置)
- $c_t$: velocity commands (linear + angular,from joystick)
- $\Delta q_t$: position offset relative to neutral pose $q_0$ (joint deviations from default)
- $\dot{q}_t$: joint velocities
- $a_{t-1}$: previous action (smoothness / temporal consistency)
- $\theta_t$: torso orientation (from IMU,大概率 quaternion 或 Euler)
- $\omega_t$: torso angular velocity (from IMU)

### 6.2 Reward decomposition

$$r_t = r_t^{\text{imitation}} + r_t^{\text{regularization}} + r_t^{\text{survival}}$$

**Imitation reward** $r_t^{\text{imitation}}$: 鼓励 imitate reference walking motion,这个 reference 用 closed-form ZMP (Zero Moment Point) solution [47] 生成。Components 见 Table 6:
- Torso quaternion tracking: 1.0
- Linear velocity (XY): 5.0
- Linear velocity (Z): 1.0 (penalize vertical bouncing)
- Angular velocity (XY): 2.0 (penalize roll/pitch)
- Angular velocity (Z): 5.0 (yaw tracking)
- Leg motor position: 5.0
- Feet contact: 1.0

**Regularization reward** $r_t^{\text{regularization}}$:
- Feet air time: 500.0 (encourage proper swing phase)
- Feet clearance: 0.05 (swing foot 抬够高)
- Feet distance: 1.0 (步宽)
- Feet slip: 0.05 (penalize sliding)
- Align with ground: 1.0
- Stand still: 1.0 (no command 时保持稳定)
- Torso roll / pitch: 0.5 each
- Collision: 0.1 (penalty)
- Leg action rate / acceleration: 0.05 each (smoothness)
- Motor torque: 0.01 (energy)
- Energy: 0.05

**Survival reward** $r_t^{\text{survival}}$: 10.0 (prevent early termination)

### 6.3 PPO training setup (Table 5)

- Policy / value network: (512, 256, 128) MLP
- Timesteps: $3 \times 10^8$
- Parallel environments: 1024 (massively parallel via MJX + Brax)
- Episode length: 1000
- Unroll length: 20
- Batch size: 256, 4 minibatches, 4 updates/batch
- Discount $\gamma$: 0.97
- LR: $10^{-4}$
- Entropy cost: 0.0005
- Clip $\epsilon$: 0.2

Inference 在 Jetson Orin NX CPU 上 50 Hz。

Reference: 
- PPO: https://arxiv.org/abs/1707.06347
- MJX: https://mujoco.readthedocs.io/en/latest/mjx.html
- Brax: https://github.com/google/brax
- ZMP closed-form: https://github.com/RobotLocomotion/drake (Tedrake)

---

## 7. Diffusion Policy for Manipulation (Section 4.3, Appendix 8.13)

### 7.1 Data Collection Setup

Teleoperation 设计很 clever:
- **Leader arms**: 第二个 ToddlerBot upper body,FSR (force-sensitive resistors) 嵌入 gripper area 检测 compression force
- **Follower**: ToddlerBot 主体
- **Lower body control**: handheld gaming PC (Steam Deck 或 ROG Ally X),joysticks 发 velocity commands,bottom buttons 触发 policies 或直接控制 neck/waist

**Critical design**: 数据采集时 follower 的 lower body 跑 **two-layer PD controller** 保持 balance:
1. **CoM PD controller**: keep CoM close to center of support polygon (compensate arm movements 引起的 CoM shift)
2. **Torso pitch PD controller**: use IMU 保持 torso upright (compensate heavy lifting)

这层 "balance assist" 让人类 teleoperator 不用管 balance,只管 manipulation,大幅提升 data quality。

Data throughput: **60 trajectories in 20 minutes**。

### 7.2 Diffusion Policy 细节

- Input: cropped + downsampled 96×96 RGB image from fisheye camera
- Visual encoder: ResNet pretrained on ImageNet
- Control freq: 10 Hz (both leader and follower downsampled)
- Actions: leader joint angles
- Observations: follower joint angles + RGB
- Training: 100 diffusion steps
- Inference: **3 DDIM steps** (够用!)
- Parameters: ~300M
- Latency: <100ms on Jetson Orin NX GPU
- Action horizon: 16-step prediction,丢前 3 步 (latency compensation [55]),执行后 5 步

这种 "discard first few actions to compensate latency" 的 trick 来自 UMI (Universal Manipulation Interface)。

Results:
- Bimanual manipulation: 90% success rate (20 trials)
- Full-body manipulation: 75% success rate (20 trials)

Reference:
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- ALOHA 2: https://aloha-2.github.io/
- UMI: https://universal-manipulation-interface.github.io/

---

## 8. Skill Chaining (Section 5, Appendix 8.15)

这是一个我觉得非常 promising 的 direction。Wagon pushing task:
1. Diffusion policy 先 grasp wagon handle
2. **Maintain that pose**,switch to RL walking policy
3. Walk forward while maintaining grip

Challenge: walking policy 训练时 robot 的 end pose 必须和 grasping policy 的 final pose 兼容。Solution: **RL training 时从 grasping policy 的 training data (60 demos) 中 sample robot 的 end pose as initial state**。这是一个 simple but effective 的 distribution-matching trick,让两个 policy 的 state distribution overlap。

这种 skill chaining 是 long-horizon manipulation 的关键 building block。Figure 5 展示了两个 ToddlerBot (Arya + Toddy) 协作清理玩具的 long-horizon scenario,涉及 walking + grasping + wagon pushing + kneeling + bimanual pickup,全部 skill 串联。

---

## 9. Experiments 关键数据

### 9.1 Capability Tests

- **Arm span**: grasp objects up to $27 \times 24 \times 31$ cm³,约 **14× torso volume** ($13 \times 9 \times 12$ cm³)
- **Payload**: lift up to **1484g = 40% body weight** (3484g)
- **Endurance**: walking RL policy stepping in place,**19 minutes** without falling (motor 温度上升导致 OOD,falls 增加)
- **Durability**: withstands **7 falls** before breaking;repair = 21 min print + 14 min assemble (including zero-point calibration)

### 9.2 Velocity Tracking (Appendix 8.14, Table 7)

| Tracking Error | Simulation | Real-World |
|---|---|---|
| Position [m] | 0.082 | 0.133 ± 0.018 |
| Linear Velocity [m/s] | 0.016 | 0.032 ± 0.002 |
| Angular Velocity [rad/s] | 0.056 | 0.113 ± 0.010 |

**关键 observation**: sim-to-real gap (0.133 - 0.082 = 0.051m position) 远小于 tracking gap (0.133 - 0 = 0.133m,即 command 和 actual 的差距)。这说明 digital twin fidelity 已经不是 bottleneck,bottleneck 是 RL policy 本身 (尤其 in-place rotation 不行)。

Position tracking variance: 0.018m, repeatability 良好。

### 9.3 Reproducibility Validation

- CS-major student (no prior hardware experience) 独立组装第二台,**3 天**完成 (含 3D printing 时间)
- 开源社区 1 周内 5 个 independent replication reports
- Policy cross-instance transfer: 在 instance A 上训练的 manipulation policy,直接 deploy 到 instance B,success rate 仍 90% (20 trials)
- RL walking policy 也成功 cross-transfer

---

## 10. Limitations & Future Work (Section 7)

很诚实的 limitations:

1. **Motor performance ceiling**: off-the-shelf Dynamixel 的 max speed / torque / communication speed 限制了 agile tasks
2. **Performance alignment**: 不是 superhuman,而是 average human level
3. **Actuation model limitation**: 不考虑 motor temperature,接近 performance limit 时不准
4. **Scale limitation**: 无法 interact with human-sized objects (但可以用 scaled objects)
5. **3D-printed parts**: 比 metal 更容易 break on impact (虽然 repair 快)
6. **Future sensing**: stereo vision for depth,更多 IMUs for state estimation,tactile sensors for manipulation feedback

---

## 11. 我的 Intuition Building 总结

这篇 paper 的核心 contribution 不是 single algorithmic breakthrough,而是 **system-level integration**。几个 takeaways 对 build intuition 有帮助:

**A. Reproducibility as hard constraint 反向 shapes 所有 design decisions**。一旦把 "single person can replicate at home without specialized equipment" 当 hard constraint,就 forced 出一系列 choices: 3D printing (因为 CNC 不 accessible),Dynamixel (因为 custom BLDC 不 accessible),miniature size (因为 full-scale 需要 gantry crane),open-source everything (因为 black boxes 破坏 reproducibility)。

**B. ML-compatibility 是 first-class design objective,不是 afterthought**。传统 robot design 把 data collection 当 deployment 后的事情。ToddlerBot 把 digital twin fidelity (sysID pipeline) 和 real-world data collection (teleoperation interface) 当 core deliverable。这反映了一个 paradigm shift: **robot 是 data generation infrastructure,不只是 policy execution infrastructure**。

**C. Digital twin 的核心是 actuator model,不是 link geometry**。Link geometry 在 CAD 里就能精确得到;真正的 sim2real gap 在 actuator dynamics。Paper 的 9-parameter actuator model + 自动 sysID test bed 是最有 transferable value 的 engineering contribution。

**D. Skill chaining 的 distribution-matching trick**。从 grasping policy 的 training data sample end pose 作为 RL walking 的 initial state——这种 "make state distributions overlap" 的思路是 chained policies 的通用 pattern,比硬 train end-to-end 更 sample-efficient。

**E. Power factor $\tilde{p}$ 是一个值得讨论的 metric**。它把 "robot 多强" 这个模糊问题 reduce 到 "total torque / gravity moment" 的 dimensionless ratio,且不 normalize by DoF count。Figure 8 显示 ToddlerBot 最接近 human,这解释了为什么它能在 6K budget 下做到 human-like loco-manipulation。

**F. Balance assist during teleoperation 是 data quality 的 hidden lever**。让 lower body 自动 maintain balance,人类只 focus on manipulation,这种 "shared autonomy" 让 60 trajectories in 20 minutes 成为可能,diffusion policy 才能从如此少的数据学到 90% success rate。

整体来看,这篇 paper 在 "democratize humanoid robotics research" 这个方向上做到了极致。6K USD + 3D printer + 一周时间,任何人都可以有一台 30-DoF humanoid + 高保真 digital twin + teleoperation data collection pipeline。对 academic lab 来说这意味着 humanoid loco-manipulation research 不再被 hardware barrier 锁死在少数几个有钱的 lab。

进一步阅读建议:
- OmniH2O / HOVER (whole-body teleoperation + learning): https://omni.human2humanoid.com/
- HumanPlus (Stanford, humanoid shadowing): https://humanplus.github.io/
- ExBody2 (expressive humanoid control): https://github.com/chengxuxin/exbody2
- DROID dataset (large-scale manipulation): https://droid-dataset.github.io/
- Open X-Embodiment: https://robotics-transformer-x.github.io/

如果 Karpathy 你想 hands-on, ToddlerBot 的 GitHub repo (design files + code + assembly tutorials) 在 https://toddlerbot.github.io 链接出去,完全可以自己 replicate 一台来玩。对你 Tesla Optimus 的工作也可能有一些 scale-down prototype 的 value——很多 loco-manipulation policy ideas 可以先在 6K 的 ToddlerBot 上 validate,再 scale up 到 full-size humanoid。
