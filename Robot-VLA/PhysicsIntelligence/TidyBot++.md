---
source_pdf: TidyBot++.pdf
paper_sha256: 51bdc57b0523b46e63796ad8df36dab00038672025aa928f2d9531d71d77d843
processed_at: '2026-08-12T16:10:21-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，用人话讲一遍。

---

## 这帮人干了啥

Stanford 和 Princeton 的一帮人做了个机器人底盘，开源的，五六千美金，你在家一两天就能组装出来。上面可以装任何机械臂。他们用它跑了 6 个家务任务——开冰箱、擦桌子、装洗碗机、倒垃圾、装洗衣、浇花——用手机遥操作录了 50 到 100 段演示，训练 diffusion policy，成功率 6/10 到 10/10 不等。

就这么个事。听起来不复杂，但里面有个非常漂亮的 insight。

---

## 核心问题：为什么 mobile manipulation 的 policy 这么难学

你想想，固定机械臂学一个 task，比如拿杯子，arm 在原地动，policy 只需要学"看到杯子 → 手往那移"。动作空间是关节角度或末端位置，很干净。

一旦加了移动底盘，事情就麻烦了。比如擦桌子，你得一边挪底盘一边动手臂。但问题来了——如果底盘是差速驱动的，像汽车一样不能横着走，那擦桌子这个动作就变成了：**先转个角度，往前开一点，再转回来，再开一点**，也就是你侧方停车那一套。

这意味着 policy 不光要学"擦桌子"这个 task 本身，还要额外学一个"如何用差速驱动挪到旁边"的 control problem。你给它 50 段演示，它得把这两件事一起学，data efficiency 就崩了。

---

## 他们的解法：让底盘能横着走

office chair 你坐过吧？你往后一靠椅子就往后退，你往左推就往左滑，你扭一下就转。四个轮子自己就对齐了，你完全不用操心方向。这就是 **holonomic**——平面三个自由度（前后、左右、转头）任意时刻同时可控。

为什么 office chair 能做到？因为它的轮子是 caster wheel——就是那种安装时轴线偏一点点的万向轮。这个"偏一点点"是关键。它造成了一个杠杆臂，椅子一动，轮子就自动转到行进方向。**不需要电机去转向**，纯被动动力学搞定。

所以他们做的是：用四个带电机的 caster 模块（powered caster），本质上就是一把"电动办公椅"。

---

## 跟普通 swerve drive 有什么区别

FRC 比赛里很流行一种叫 swerve drive 的东西，四个轮子各自由电机驱动和转向，能朝任意方向开。但它的轮子**没有那个偏移**，所以你让它横着走之前，得先转轮子对齐方向，再开，再转轮子，再开。这叫 omnidirectional 但**不** holonomic。

他们的 hack 非常 minimal：拿了 FRC 现成的 MK4 swerve module，3D 打印两个轮子支架，车一根轴，引入 14mm 的 caster offset。就三个定制件，剩下全是现成的。Cost 一下降下来了。

---

## 最有说服力的实验

同一个擦桌子任务，同样 50 段演示，同样训 500 epochs。

**Holonomic 底盘**：9/10 成功，平均路径 2 米，每段 27 秒。
**差速驱动底盘**：4/10 成功，平均路径 4 米，每段 65 秒。

差速驱动的 policy 主要 failure mode 是：**跳过部分桌面不擦**。因为它没学会那个侧方停车式的横向挪动策略——这本身就是个 hard control problem，数据量不够。

还有个隐性 benefit：差速驱动挪的时候摄像头画面在摆，视觉输入 noise 大；holonomic 横着走的时候摄像头一直朝前看，画面稳定，视觉 policy 学起来容易得多。

这个实验说明了一件事——**底盘的运动学性质不只是工程问题，它直接决定了 policy 的可学习性**。同样的算法、同样的数据量，换一个底盘，成功率从 40% 变 90%。这跟你换一个更好的网络架构是同等级别的杠杆。

---

## 数据怎么采

用手机。打开一个网页，手机的 6-DoF 位姿（位置+朝向）通过 WebXR API 传给电脑，电脑把手机动作映射成底盘或机械臂的运动。iPhone 用 ARKit，Android 用 ARCore，WebXR 是统一接口。

好处是：不用买专门的遥操作设备，任何人的手机都行。操作者可以自由走动，精细动作的时候凑近机器人看。相比之下 Mobile ALOHA 把操作者绑在机器人后面，离末端执行器很远，精细操作难。

---

## 为什么这事重要

现在 robot learning 的瓶颈不在算法，在数据。固定臂的数据集已经有了 DROID、Open X-Embodiment 这些，但 mobile manipulation 的数据几乎没有，因为没有一个便宜、好用、标准化的平台。

这个工作的真正贡献不是"又一个机器人"，而是**把 mobile manipulation 的数据采集成本从几十万美金拉到了五六千美金**。如果社区里几十个 lab 都用这个平台采数据，就能拼出一个 mobile manipulation 的大数据集，那时候算法层面的进步才能真正发挥出来。

---

## 一个有意思的 limitation

底盘现在推不动（backdrive 不行），因为转向齿轮比太大（12.8）加上 caster offset 太小（14mm），摩擦力大。这意味着你没法手把手教机器人（kinesthetic teaching），只能遥操作。

作者说拆掉转向齿轮就顺滑了，但那就要更多定制件，违背开源易用原则。这是个 design tradeoff 的好例子。

---

## 一句话总结

**底盘能不能横着走，决定了你的 policy 学不学得会。** 这不是工程细节，是 imitation learning 的 first-order factor。

---

# TidyBot++: Holonomic Mobile Manipulator 深度讲解

## 1. 一句话总结

这篇 paper 的核心论点非常清晰: **mobile manipulation 的 data collection 瓶颈来源于硬件平台设计**, 作者从 ground up 重新设计了一个 \$5–6k 的 open-source holonomic base, 用 powered caster 模块实现 (x, y, θ) 三个 planar DoF 同时独立可控, 配合 WebXR 手机遥操作, 让 diffusion policy 在真实家庭场景下学会 6 个 household task。

项目主页: http://tidybot2.github.io

---

## 2. Holonomic vs Nonholonomic — 物理 intuition

### 2.1 关键区分

| 类型 | 控制维度 | 独立 DoF | 例子 |
|------|---------|---------|------|
| **Nonholonomic** | velocity only, 存在不可积约束 | < 3 | 汽车, 差速 drive, 四足 |
| **Omnidirectional** | 任意方向移动, 但需手动 align wheel | = 3 (但非瞬时) | 普通 swerve 无 caster offset |
| **Holonomic** | (x, y, θ) 同时独立瞬时控制 | = 3, 任意时刻 | office chair, 本文 PCV |

### 2.2 Caster offset 的物理 magic

Figure 2 里那个 caster wheel 的关键 design feature 是: **steer 的 vertical axis 与 wheel 的 roll axis 之间存在一个 offset b**。

直觉构建: 想象你推一把 office chair, chair 朝任意方向移动, caster wheel 会**自动**旋转 align 到运动方向 — 不需要任何 motor 去主动 steer。这是 passive dynamics 的结果:

- offset b 形成了一个 **lever arm**
- 当 chair 速度 v 有一个垂直于 wheel plane 的分量时, 这个分量通过 offset b 产生一个 torque 绕 steer axis
- torque → caster 自然旋转, 直到 wheel plane 与速度方向对齐 (此时 lever arm 不再产生回正 torque)
- 这就是 **trailing wheel** 的稳态行为, 类似超市购物车的前轮

**关键 insight**: 没 offset 的 swerve drive 是 omnidirectional 但**非 holonomic** — 你得先 motor-steer wheel 朝目标方向, 再 drive, 然后再 steer, 再 drive, 这个 sequencing 引入了"非完整约束"的不可积分性。有 offset 的 caster, wheel 永远 trailing, 任意方向可瞬时加速。

参考: 原始 PCV 论文 https://journals.sagepub.com/doi/abs/10.1177/0278364000264 (Holmberg & Khatib, IJRR 2000)

---

## 3. Powered-Caster Vehicle (PCV) Kinematics — 数学细节

### 3.1 单个 caster module 的几何

参考 Figure 4, 每个 caster module 在 base frame 下的安装位置是 $(h_i, \beta_i)$ — i 是 module 索引 (1..4)。每个 module 有:

- **Steer joint** $\phi_i$: wheel 绕 vertical axis 的转向角, 由 absolute encoder 直接测量
- **Roll joint** $\rho_i$: wheel 自身的滚动, 由 motor encoder 通过 gearbox 间接得到
- **Wheel radius** $r$
- **Caster offsets** $b_x$ (longitudinal, 经典 PCV 仅有此分量) 和 $b_y$ (lateral, 本文新增, 来自 design constraint — 想避免 custom machining)

下标含义: $i$ 表示第 i 个 caster module; $x,y$ 表示 offset 在 module-local 坐标系下的分量; $h, \beta$ 是 module 在 base frame 下的极坐标位置 (radial distance + angle)。

### 3.2 速度约束推导

考虑 base 在 ground plane 上的 twist $\xi = (v_x, v_y, \dot\theta)^T$ (base frame 下表达)。第 i 个 caster 的 wheel-ground contact point 在 inertial frame 下速度必须满足 no-slip 约束 (contact 点 ground velocity = 0, 或者 wheel surface velocity = 0)。

Wheel center 在 base frame 下位置 (考虑 steer 后):
$$
\mathbf{p}_i = \begin{bmatrix} h_i \cos\beta_i \\ h_i \sin\beta_i \end{bmatrix} + R_z(\phi_i) \begin{bmatrix} b_x \\ b_y \end{bmatrix}
$$

其中 $R_z(\phi_i)$ 是绕 z 轴旋转 $\phi_i$ 的 2D rotation matrix。

Contact point 速度 (base motion 引起的):
$$
\mathbf{v}_{\text{contact}}^{\text{base}} = \begin{bmatrix} v_x \\ v_y \end{bmatrix} + \dot\theta \, \hat{z} \times \mathbf{p}_i
$$

加上 steer 和 roll joint 的贡献 (caster module 自身运动):
$$
\mathbf{v}_{\text{contact}}^{\text{caster}} = \dot\phi_i \, \hat{z} \times (R_z(\phi_i) [b_x, b_y]^T) + \dot\rho_i \, r \, \hat{e}_{\text{roll}}(\phi_i)
$$

其中 $\hat{e}_{\text{roll}}(\phi_i)$ 是 wheel 朝向的单位向量 (in steer frame 下沿 x 轴, 投回 base frame 是 $\cos\phi_i, \sin\phi_i$)。

No-slip (两个分量: 沿 wheel axis 方向 = 0, 垂直 wheel axis 方向 = caster 自然 align):
$$
\mathbf{v}_{\text{contact}}^{\text{base}} + \mathbf{v}_{\text{contact}}^{\text{caster}} = 0
$$

整理后得到 base twist $\xi$ 与 joint velocity $\dot{\mathbf{q}}_i = (\dot\phi_i, \dot\rho_i)^T$ 之间的 Jacobian 关系:

$$
\xi = J_i(\phi_i) \, \dot{\mathbf{q}}_i
$$

四个 caster stack 起来, 系统是冗余的 (8 joint DoF, 3 task DoF), 给出 4 个 no-slip 约束, 解出 minimal-norm joint velocity 命令:

$$
\dot{\mathbf{q}} = J^{\dagger}(\phi) \, \xi_{\text{desired}}
$$

其中 $J^\dagger$ 是 Moore-Penrose pseudoinverse, $J \in \mathbb{R}^{8 \times 3}$ (4 casters × 2 joints each, 3 base DoF)。

### 3.3 为什么这个公式对 imitation learning 重要

Diffusion policy 输入是 state (image + proprioception), 输出是 action — action 用 **position** 还是 velocity 表达是关键 design choice。Velocity action 在 long horizon 下会 drift, 而 position action 是 "setpoint" 性质的, 更稳定。

非 holonomic base (differential drive) **无法** 直接 position-control (x, y, θ) — 你只能 command 左右轮速度, 然后 base 通过运动学约束被动地集成出 pose。这意味着 policy 输出必须是 velocity, 学习难度上升。Holonomic base 直接 $\xi_{\text{desired}} = (x^*, y^*, \theta^*)$, 控制器内部求解 joint velocities, policy 只需 output end-effector-level setpoint。

参考 robot learning 中关于 action representation 的讨论: Diffusion Policy https://diffusion-policy.cs.columbia.edu/ (Chi et al., RSS 2023)。

---

## 4. Hardware Design 细节

### 4.1 来自 FRC 生态的启发

设计选型来自 **FIRST Robotics Competition (FRC)** 的成熟部件生态 (https://www.firstinspires.org/robotics/frc):
- SDS MK4 Swerve Module (https://www.swervedrivespecialties.com/products/mk4-swerve-module): 每个模块 2 个 motor (drive + steer), 内置 CAN bus controller
- 每年 80,000+ 高中生用同样的部件造 125 lb robot 在高强度 collision 中跑高速 — 可靠性 battle-tested
- Steer gear ratio = 12.8, 这个数字在后面 limitation 章节会回来

### 4.2 从 swerve 到 holonomic 的 minimal 改造

原始 MK4 是 **无 caster offset** 的 swerve module (omnidirectional 但 nonholonomic)。作者用 minimal 改动引入 offset:

- 2 个 3D printed wheel mount (PLA, FDM)
- 1 个 custom machined shaft (可以从 Xometry 这样的 online service 订购)

仅此 3 个 custom parts。Caster offset = 14 mm (这个数字在 limitation 里是 friction 问题的根源)。

### 4.3 Spec table 关键数字

| 项目 | 数值 | 备注 |
|------|------|------|
| Footprint | 50×54 cm | 可穿 household doorway |
| Weight | 34 kg (base) + 12 kg (Kinova Gen3) | |
| Payload (conservative) | 60 kg | 实测加载 122 kg motor 无 struggle |
| Max speed | 1 m/s | |
| Runtime | 8 hours teleop | 768 Wh portable power station |
| Odometry drift | < 1 cm/m translation, < 1°/360° rotation | motion capture 验证 |
| Cost | \$5.4k | vs Tiago \$100k, Fetch \$100k, Stretch \$25k |

### 4.4 Counterweight 设计

便携 power station (8.6 kg) + SLA battery (6 kg) 作为 counterweight 防 tip-over。双 battery 是 design flexibility 优于 electrical efficiency 的取舍 — 避免设计 DC-DC converter 适配各种 arm 和 compute。

---

## 5. WebXR Teleoperation Interface

### 5.1 为什么 WebXR

- iOS 上是 ARKit, Android 上是 ARCore, WebXR 是统一抽象层 (https://www.w3.org/TR/webxr/)
- IMU + camera visual odometry fusion → 6-DoF pose, drift 显著小于纯 IMU (对比 RoboTurk 的 drift 问题)
- 不需要专用 teleoperation hardware (vs DROID 用 Oculus controller 受 IR 视野限制)
- 网页运行, 跨平台, 即开即用

### 5.2 Mapping 策略

Phone 6-DoF pose → 两种 mapping mode:
- **Base mode**: phone 水平 motion 控制 base (x, y), 旋转控制 θ, 实现 "wand-like" 操控
- **Arm mode**: phone 6-DoF 直接映射到 end-effector pose (position + orientation), 类似 VR controller

### 5.3 与 Mobile ALOHA teleop 对比

Mobile ALOHA (https://mobile-aloha.github.io/, Fu et al. 2024) teleoperator 被 strapped 在 base 后部, 远离 end-effector, 精细操作困难。TidyBot++ teleoperator 可以**自由走动**, 精确动作时凑近 robot, ergonomics 更好。

---

## 6. Experiments 拆解

### 6.1 Imitation learning 主实验

| Task | N demos | Success rate |
|------|---------|--------------|
| Open fridge | 100 | 10/10 |
| Wipe countertop | 50 | 9/10 |
| Load dishwasher | 50 | 7/10 |
| Take out trash | 50 | 10/10 |
| Load laundry | 50 | 7/10 |
| Water plant | 50 | 6/10 |

- 每个 policy 训 500 epochs, 10 episodes evaluation
- Diffusion policy 经典配置是 200–300 demos, 这里 50 demos 已足够, 说明 task structure 和 holonomic action space 简化了学习难度

### 6.2 Differential drive 对比实验 (最有意思的部分)

Wipe countertop 任务上 head-to-head:
- 同样 50 demos, 同样 500 epochs
- Holonomic: 9/10 success, 平均路径 2.03 m, 平均 27.4 s/episode
- Differential drive: 4/10 success, 平均路径 4.03 m, 平均 65.2 s/episode

**Failure mode 分析**: differential drive policy 倾向于"skip over portions of countertop rather than wiping"。原因:
1. Policy 必须**额外学习** parallel-parking 式的 sideways maneuver, 这本身就是个 hard control problem
2. 差速 base 在做 sideways maneuver 时 camera view 持续 swerve, 视觉输入 distribution shift 大, 学习信号 noisy
3. Holonomic base 维持 forward-facing camera, 视觉 stable

这个实验直接验证了一个重要 hypothesis: **base 的运动学性质会显著影响 policy 的可学习性**, 而不仅仅影响 execution efficiency。

### 6.3 Odometry 精度

Translation drift < 1 cm/m, Rotation drift < 1°/360° — 这意味着 holonomic base 可以直接 position-control $(x, y, \theta)$ setpoint, 重复性高, 这是 diffusion policy 用 position action 的物理基础。

---

## 7. Limitations 与未来方向

### 7.1 Backdrivability 问题

Steer gear ratio = 12.8 + caster offset 14 mm → 高 steering friction, base 不能 backdrive (即人推不动)。这对于 kinesthetic teaching (手把手教 robot) 是个 loss — 物理上不可行。

作者验证: 拆掉 steer gearing 后 backdrivable 顺滑, 但这违背了 open-source accessibility 原则。

### 7.2 思考延伸 — 与其他 paradigm 对比

| Paradigm | 例子 | 优势 | 劣势 |
|----------|------|------|------|
| Holonomic wheeled (本文) | TidyBot++ | indoor 高效, 学 policy 容易 | 不能爬楼梯, outdoor 弱 |
| Quadruped + arm | See, e.g. https://humanoid-ai.github.io/ | 跨 terrain | nonholonomic, locomotion 与 manipulation 耦合复杂 |
| Humanoid | HumanPlus, Figure | 全场景 | 极其 expensive, balance control |
| Cane/Stick data + deploy | Dobb·E (https://dobb-e.com/) | 极低 data collection 成本 | domain gap, 一次性 stick ≠ robot kinematics |

### 7.3 与 DROID / Open X-Embodiment 生态

DROID (https://droid-dataset.github.io/) 全部是 fixed-arm setups。Open X-Embodiment (https://robotics-transformer-x.github.io/) 数据也以 fixed-arm 为主。TidyBot++ 提供 standardized mobile platform, 可望成为 mobile manipulation 的 "DROID 对应物", 这是 community-level 的影响。

### 7.4 与 TeleMoMa 的关系

TeleMoMa (https://telemoma.github.io/) 是 modular teleoperation framework, 支持多种 interface (包括 mobile phone via ARKit), 但锁定在 3 个商业高 cost robot (Tiago 等)。TidyBot++ 可以看作是 "open-source hardware equivalent" 的 teleop + mobile manipulator, 互补性很强 — 可以想象 TeleMoMa 跑在 TidyBot++ 上。

---

## 8. 我的几个 takeaways

**1. Hardware design 是 policy learning 的 first-class citizen.** 这篇 paper 的核心 contribution 其实**不是**一个新的 learning algorithm, 而是**一个 evidence-based 论证**: 给定相同的 diffusion policy, 50 demos, 500 epochs, base 的运动学性质能从 4/10 拉到 9/10。这是一个非常 strong 的 hardware-software co-design 论据。

**2. Caster offset 是 design 中"看似 trivial 实则 profound" 的细节.** 把 omnidirectional 升级为 holonomic 只需要 ~14 mm 的 offset, 但在 policy learning 上是 qualitative difference — 因为它把 control space 从 SE(2) 的非完整子流形 lift 到完整的 R^3 (position)。

**3. Open-source 部件来源是关键.** 借用 FRC 的成熟供应链是 brilliant 的策略 — 它把 robot hardware 从 research lab hand-crafted 工艺品变成可量产 commodity, 这正是 robot learning data scaling 需要的基础设施。

**4. WebXR 是一个被低估的 interface.** 大多数 lab 还在用 VR controller 或者 dedicated device, 而手机是 universal — 这让 data collection 的 marginal cost 趋近于 0 (任何有手机的人都能 teleop)。结合未来的 cloud teleop + remote operator pool, 可能产生 robot learning 的 "Mechanical Turk"。

**5. Limitation 里的 backdrive 问题反过来指出了下一个方向.** 当前 base 不能 hand-pushed teaching, 只能 teleop。如果要引入 kinesthetic teaching (一般比 teleop 数据更 "natural"), 需要 low-friction steering redesign — 这是 future work 的明确 hook。

---

## 9. 参考链接汇总

- 项目主页: http://tidybot2.github.io
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- Mobile ALOHA: https://mobile-aloha.github.io/
- WebXR API: https://www.w3.org/TR/webxr/
- DROID dataset: https://droid-dataset.github.io/
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- OpenVLA: https://openvla.github.io/
- Dobb·E: https://dobb-e.com/
- TeleMoMa: https://telemoma.github.io/
- FRC: https://www.firstinspires.org/robotics/frc
- SDS MK4 swerve module: https://www.swervedrivespecialties.com/products/mk4-swerve-module
- 原始 PCV 论文 (Holmberg & Khatib 2000): https://journals.sagepub.com/doi/abs/10.1177/0278364000264
- TidyBot 前作: https://tidybot.cs.stanford.edu/
- Hello Robot Stretch: https://hello-robot.com/stretch-3-product
- Universal Manipulation Interface (UMI): https://umi-pipeline.github.io/
- Consistency Policy: https://consistency-policy.github.io/
