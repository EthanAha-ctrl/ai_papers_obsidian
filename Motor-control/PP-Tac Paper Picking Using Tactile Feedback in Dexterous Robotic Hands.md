---
source_pdf: PP-Tac Paper Picking Using Tactile Feedback in Dexterous Robotic Hands.pdf
paper_sha256: d28fddefaf7e0ad86d87b815193bf92e75de35a297c2c553e777910ed7a66370
processed_at: '2026-08-06T05:29:44-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 PP-Tac

## 这篇 paper 到底在干嘛

想象你让机器人去桌上捡起一张 A4 纸。

听起来简单得离谱，对吧？但这件事在 robotics 里面其实是个 nightmare。你想想，纸平贴在桌上，没有边可以抓，视觉根本看不出哪里有"抓手位"，而且一碰它就变形。传统的二指 gripper 基本上完全废掉。

你自己回想一下平时怎么拿桌上的纸——你会用手指肚在纸面上滑一下，让纸微微翘起来一点，然后趁那个翘起来的瞬间捏住。这个动作你做起来完全不费脑子，但它其实是一套非常精细的 sensorimotor coordination。

PP-Tac 就是让机器人学会干这件事。整套系统的核心 idea 就一句话：**给灵巧手的每个指尖装上"电子皮肤"，然后让网络模仿人类的"滑动+捏取"动作**。

---

## 为什么这事之前没人做成

三个原因，都挺 fundamental 的：

**第一，vision 在这里根本没用。** 纸是平的，贴在桌面上，camera 看过去就是一坨颜色，你根本看不出厚度，看不出哪里可以下手。遮挡问题也很严重，手指一过去就把纸盖住了。所以纯 vision-based 方法在这里基本歇菜。

**第二，gripper 的 dexterity 不够。** 你用两指夹子去夹平铺的纸，基本夹不起来，因为没有 edge 可夹。人类能做这件事是因为我们有 5 根手指，可以先滑再捏，建立多个 contact point。两指夹子做不了 sliding 这种动作。

**第三，VBTS（视觉触觉传感器）之前太贵太难部署。** 像 GelSight、DIGIT 这些传感器要标定就得用 CNC 机床，采一大堆数据，RGB 数据传输带宽又高。你想想，一个手 4 个指尖，每个都要装一个，光是 calibration 就够你喝一壶。所以之前的 VBTS + 灵巧手的工作基本都用在 in-hand rotation 这种"已经抓住之后再操作"的场景，没人敢碰 paper picking。

PP-Tac 同时把这三个坑都填了。

---

## R-Tac 这个传感器厉害在哪

你把它理解成一个指尖大小的半球形气球，里面有个小摄像头在拍气球内壁。

气球被压变形 → 内壁的 grayscale 变化 → 摄像头拍到 → 通过标定好的 mapping 函数算出 depth map。

听起来跟 GelSight 差不多，但关键区别是 **R-Tac 用单色光，不是 RGB**。

RGB 传感器在 curved 表面上会有一个致命问题：红绿蓝三个通道在曲面上的亮度分布不均匀，你得采一大堆数据去标定这个不均匀性，所以需要 CNC testbed。R-Tac 用白色 LED + 磨砂漫射片，uniform illumination，intensity 变化直接对应 deformation，30 张图就能标定完，不用 CNC。

这个 trade-off 其实特别聪明。你牺牲了 RGB 的多通道信息（理论上 RGB 能提取更多 surface 信息），换来的是：标定简单、带宽低（单色图像数据量小）、帧率高（120Hz）、成本低（~$60）、3 天就能造一个。

四个指尖都装上，整体延迟 100ms 左右，足够做 closed-loop control。

---

## Slip detection 怎么做的

特别 intuitive 的一个设计。

你去观察 R-Tac 拍到的 tactile image，当纸在手指底下开始打滑的时候，传感器表面会出现肉眼可见的 **wrinkles（褶皱）**。这个物理现象是因为纸和传感器之间在滑动时产生了微小的 shear deformation，在 elastomer 表面 manifest 成皱纹。

所以他们直接训了一个 CNN + MLP，输入是最近 5 帧 tactile image 加 1 帧 no-contact 参考帧，输出是 $P_{slip}$。如果 $P_{slip} > 0.75$，就判定发生 slip，然后触发 force controller 增加按压力度。

20 分钟训练数据，86% accuracy。简单粗暴但有效。

这里有意思的点是，他们没有去显式建模 friction coefficient 或者用 force-based slip detection（那种需要精确的 force/torque 传感器），而是直接让网络从 image pattern 学习。这是典型的 "don't model it, learn it" 思路。

---

## 力学核心：Buckling

整个系统能 work 的物理基础是 paper 的 buckling。

你想，纸平铺在桌上，你用手指压上去，纸不会平白无故翘起来给你抓。但如果你一边压一边稍微往中间滑，纸就会在两指之间鼓起来一个小包，这个就是 buckling。这个鼓起来的小包就是你的 pinch point。

这件事的力学条件其实挺微妙的。手指和纸之间要有足够的 static friction（这样纸会跟着手指走），纸和桌面之间要有 dynamic friction（这样纸在桌面上滑动有阻力），当手指的 static friction 大于纸的 critical buckling resistance，纸就会 buckle。

论文里 Table I 那个实验特别说明问题。1 层纸成功率 90%，3 层 75%，5 层 30%，7 层只有 5%。因为纸越厚，buckling resistance 越大，但你手指 motor 能输出的 force 是有限的，所以 slip 就越来越多，最后就抓不起来。

这个实验直接验证了他们的 force model 是对的，也告诉你这个方法的 physical limit 在哪——材料太硬就没救了。

---

## Policy 怎么训的

这里有个特别实用的工程 trick。

按理说，你要训一个 manipulation policy，最 direct 的办法是 reinforcement learning。但 RL 需要 simulator，你这个 task 的 simulator 得模拟 paper 的 deformable dynamics，还得模拟 VBTS 的 elastomer deformation，sim-to-real gap 巨大，计算也贵得要死。

PP-Tac 的做法是：**完全不模拟 deformable object，只用 rigid-body dynamics**。

具体怎么搞？先 teleop 录一个人类抓取的动作作为 template。然后随机生成各种 terrain（用 cubic spline），把 template 动作的 fingertip (x,y) 投影到 terrain 上得到 z，再通过调整 fingertip 到 terrain 的距离来控制不同的 contact force（因为距离越近 sensor 被压得越深）。

然后用 trajectory optimization 求出 hand joint angles 和 arm pose，loss 是三个项：让 fingertip 跟踪目标轨迹 + 让 joint 角度别偏离初始姿态太远 + 让 wrist 别跑出 workspace。SGD 优化，filter 掉 collision 的轨迹，最后得到 50 万条 grasp motion。

这个 dataset 完全在 rigid-body simulator 里生成的，没有 deformable physics，没有 tactile simulation，所以计算便宜，而且 transfer 到 real robot 直接 work。Sim-to-real 的 gap 通过后面的 domain randomization 来弥补。

---

## Diffusion Policy 的设计

网络架构是 4 层 Transformer encoder，latent dim 512，4 个 attention head。

输入状态维度是 152 维，包含 hand joint positions、joint velocities、joint angles、joint angular velocities、wrist rotation、wrist angular velocity、wrist height、wrist height velocity，还有 4 个 tactile sensor 的 deformation depth。

注意这个状态空间是 **over-parameterized** 的。你想想，joint position 和 joint angle 本质上是 redundant 的（一个能通过 forward kinematics 算出另一个），但他们都塞进去了。这是故意的，让网络自己学习更鲁棒的 latent representation，跟 MDM 那篇 human motion diffusion 的思路一样。

Diffusion 本身是 DDPM 那一套，1000 步训练，但有个关键改动：**预测 $x_0$ 本身，而不是预测 noise $\epsilon$**。

为什么？因为 motion sequence 这种数据，直接预测目标 state 比预测噪声再 denoise 效果更好。这是从 MDM 那里学来的。

还有一个 physics-informed loss 叫 $L_{consist}$，就是让预测出来的 joint angles 经过 forward kinematics 后得到的 positions，和预测出来的 positions 要一致。这是个 sanity check，防止网络预测出 kinematically impossible 的姿态。

推理时加速到 10 步，11ms 出一个 action，RTX 4090 上跑，real-time 没问题。

---

## Domain Randomization 是 sim-to-real 的灵魂

这个我必须强调一下，因为这是整个系统能在 real world work 的关键。

他们在训练时对 historical state 注入 4 种 disturbance：

1. 给 joint angles 加 Gaussian noise，模拟 control error
2. 第一帧加 noise 后逐帧放大，模拟 finger 在起伏 terrain 上移动
3. 随机选几帧保持 static，模拟 finger 卡住的情况
4. 把 frame index 也塞进 input，让网络知道现在第几帧

ablation 实验里，去掉这些 disturbance 之后，kraft paper bag 这种 stiff 材料直接完全失败。加上之后各种 terrain 都能 handle。

代价是训练时间多了 40 万 iteration，但这个 investment 完全值得。

---

## 实验结果的核心 insight

87.5% 整体成功率，这是首次 VBTS + 灵巧手做 paper picking 的结果。

材料维度：cloth 和 plastic bag 容易（软，容易 buckle），paper 中等，kraft paper bag 难（硬，多层，难 initiate buckling）。

Terrain 维度：flat plane 和 slope 容易（friction consistent），book underneath 中等（book edge 提供 initial buckle point，反而有帮助），random terrain 难（force controller 难以 compensate）。

Baseline 对比里最有意思的是：Allegro Hand 本身有 >0.1 rad 的 joint angle error，累积到 fingertip 就是厘米级的位置误差。对于 paper picking 这种需要 millimeter precision 的任务，open-loop 直接废掉。这从侧面证明了 tactile feedback 在这里不是 nice-to-have，是 must-have。

---

## 这个工作的真正贡献

把这件事拆开看：

1. **硬件层面**：R-Tac 证明了 monochrome VBTS 可以 scale 到多指尖部署。这不是什么理论突破，是工程整合的胜利。标定从 days 降到 minutes，成本从 thousands 降到 $60，这才让"4 个指尖都装 VBTS"变成现实。

2. **数据生成层面**：绕过 deformable simulation 是个非常务实的工程决策。与其花大力气去 sim soft body + tactile elastomer，不如直接用 rigid-body + domain randomization。这个思路在 robotics 里值得推广。

3. **Policy 层面**：Diffusion policy 在 robotic control 上的应用越来越成熟，PP-Tac 加了 tactile conditioning 和 consistency loss，把 vision-free 的 tactile-driven manipulation 推到了一个新的 task category。

4. **Task 本身**：paper picking 这个 task 之前基本没人碰，因为它卡在 vision、gripper dexterity、tactile sensing 三个 bottleneck 的交集。PP-Tac 用一套组合拳把这三个都解了。

---

## 我觉得这个工作的局限

**Initial force 是手动设的**。这个很烦，因为你换个材料就得重调。理论上应该让 policy 自己学一个 adaptive initial force，而不是让人 tune。

**完全没用 vision**。我觉得这是个 over-correction。vision 在 paper picking 里确实不能单独 work，但在 "找到纸在哪" 这个 stage 还是有用的。一个 vision for localization + tactile for manipulation 的 two-stage system 可能更 practical。

**Slip detection 的 generalization 存疑**。20 分钟数据训出来的 CNN，对 wet paper、glossy magazine 这种没见过的 material surface 能不能 work？我持保留态度。

**没有 in-hand manipulation after grasp**。抓起来之后呢？能不能 fold、hand over、insert？这个工作只解决到 "pick up" 这一步，后面的 story 还长。

**Tactile sim-to-real 的 gap 没有显式处理**。他们用 rigid-body 生成 data，但 real tactile sensor 的 noise model、lighting variation、elastomer aging 这些都没有 model。完全靠 diffusion policy 的 generalization 能力在扛。长期部署会不会 degrade？不知道。

---

## 一句话总结

PP-Tac 的核心 insight 就是：**paper picking 这个 task 的瓶颈不在单一模态，而在 vision + dexterity + tactile 的三角缺失**。他们用 monochrome VBTS 解决 tactile 部署问题，用 dexterous hand 解决 dexterity 问题，用 diffusion policy + domain randomization 解决 vision 缺失下的 control 问题。三件事拼一起，第一次让机器人能像人一样从桌上捡起一张纸。

工程上非常 solid 的工作，虽然每个单独组件都不是 breakthrough，但整合出来的 system 解决了一个真实的、之前没人解的 task。这种 system paper 我觉得是 robotics 里最 valuable 的贡献类型。

---

# PP-Tac: 基于触觉反馈的灵巧手抓取纸状物体系统

## 1. 核心问题与动机

PP-Tac 解决的是一个机器人领域长期被忽视的难题——**抓取薄、平、可变形的纸状物体**（paper, plastic bag, cloth, kraft paper bag）。这类任务的挑战在于：

- **Vision 失效**：平面物体缺乏 salient features，接触信息难以从视觉获取，且遮挡问题严重
- **Grasp point 缺失**：纸张平贴在桌面上，没有 edge 可抓
- **Deformation 不可预测**：物体形状在 manipulation 过中持续变化

人类解决这个问题的策略非常优雅：先用手指在物体表面滑动（sliding），通过 friction 让纸面产生 **buckling**（屈曲），形成一个可以 pinch 的小区域，再完成抓取。PP-Tac 就是把这个策略复制到机器人上，依靠 **VBTS (Vision-Based Tactile Sensors)** 提供 real-time feedback。

参考相关工作：
- DTact: https://arxiv.org/abs/2303.03739
- 9DTact: https://arxiv.org/abs/2309.18416
- DenseTact: https://arxiv.org/abs/2206.02739
- DIGIT (Meta): https://arxiv.org/abs/2105.04797

---

## 2. R-Tac 传感器：硬件设计的核心创新

### 2.1 设计原则

R-Tac 的设计遵循五个原则，每个都直接对应一个工程痛点：

| Principle | Motivation |
|-----------|-----------|
| Round shape (hemispherical) | 全向触觉感知，配合 sliding motion |
| High resolution | 精确 depth reconstruction 与 slip detection |
| Easy to fabricate & low-cost | ~$60，3天可制造 |
| Efficient calibration | 单色 sensing，30 captures 即可完成 |
| Efficient data transmission | 单色图像数据量小，120Hz 帧率 |

### 2.2 物理结构（Fig.2 exploded view）

从外到内的 layered design：

1. **White LED ring** (LUXEON 2835 4000K SMD LED) + **双面磨砂漫射片**（diffuser）→ 提供 uniform illumination，这是单色 sensing 的关键
2. **PDMS stiff transparent skeleton** (Dow Corning Sylgard 184, Shore 50A) → 提供结构刚性，base:catalyst = 10:1，室温固化 24h
3. **Semitransparent silicone perception layer** (Smooth-On Ecoflex 00-10) → 厚度 2mm 决定 depth sensing range
4. **Opaque protective coating** (Smooth-On Psycho Paint, airbrushed) → 形成不透明表层
5. **Monochrome CMOS camera** (OV9281, 160° lens, 640×480 @ 120Hz, latency ~100ms)

### 2.3 单色 vs RGB 的关键 trade-off

传统 RGB VBTS（如 GelSight, DIGIT）使用 RGB chromatic light，在 curved elastomer 上会产生 uneven light intensity distribution，需要：
- CNC 加工的标定台
- 大量数据采集
- 高带宽传输 chromatic video

R-Tac 改用 monochrome sensing principle（参考 DTact 的思想 https://arxiv.org/abs/2303.03739），intensity 变化直接对应 deformation，capture standard deviation 低至 6。这是它能部署在 4 个指尖的关键。

### 2.4 标定公式详解

完整 mapping 从 pixel coordinates $(u,v)$ 到 sensor coordinates $(x,y,z)$：

$$
\begin{bmatrix} x \\ y \\ z \end{bmatrix} = A^{-1}\left( (D(u,v) - M(I_\Delta(u,v))) K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} - b \right)
$$

**变量解释**：
- $(u,v)$：pixel coordinates in tactile image
- $(x,y,z)$：sensor coordinates, origin 在 elastomer 中心
- $K$：相机内参矩阵（3×3），来自 manufacturer 或 Zhang 标定法 https://arxiv.org/abs/2206.02739
- $A$：外参 rotation matrix（3×3）
- $b$：外参 translation vector（3×1）
- $D(u,v)$：sensor surface reference projection，即未接触时 sensor 表面在 image 上的深度参考
- $M(\cdot)$：depth mapping function，从 grayscale intensity 估算 indentation depth
- $I_\Delta(u,v)$：grayscale intensity image（变形后减去参考帧）

**Intuition**：先用 solvePnP (EPnP https://arxiv.org/abs/1901.05724) 标定相机外参（28次 indentation），再用一个已知尺寸的 ball 一次压痕标定 $M$。整个流程避免 CNC，把标定从 days 降到 minutes。

### 2.5 Force estimation & Slip detection

Force estimation 假设 elasticity theory 线性关系 $F \propto \delta d_{tac}$。

Slip detection 网络（Fig.3）：
- 输入：5 帧 temporal sequence + 1 帧 non-contact reference
- CNN 提取 per-image features → concat → MLP 输出 $P_{slip}$
- 训练数据：20分钟，40% slip / 60% non-slip
- 损失：binary cross-entropy
- 阈值 0.75，slip detection accuracy 86%

**关键 intuition**：slip 发生时，sensor 成像中会出现明显的 **wrinkle features**（褶皱），这是 paper-like object 在 friction 失效时的特征。CNN 直接学习这个 visual signature，比传统 force-based slip detection 更鲁棒。

参考：GelSlim 3.0 slip detection https://arxiv.org/abs/2102.02026

---

## 3. 力学分析：buckling-based grasping 的物理基础

Fig.4 是整个系统的物理核心。当手指施加 normal force 时，paper 会发生 buckling，产生一个 pinchable region。这个过程涉及三类力：

1. **Contact normal force** $F_N$：sensor 施加在 paper 上的法向力
2. **Static friction** $(f_1, f_1')$：finger-paper 间静摩擦，大小 $\leq \mu_s F_N$
3. **Dynamic friction** $(f_2, f_2')$：paper-terrain 间动摩擦，大小 $= \mu_d F_{terrain}$

**成功 buckling 的条件**：

$$f_1 > f_{buckle}^{critical} \quad \text{AND} \quad f_1 > f_2$$

其中 $f_{buckle}^{critical}$ 是 paper 的临界屈曲抗力，取决于材料 stiffness 与 thickness。

**Table I 的实证验证**：

| Paper Layers | No. Slip | Succ. Rate |
|-------------|---------|-----------|
| 1 | 0.2 | 90% |
| 3 | 2.9 | 75% |
| 5 | 13.3 | 30% |
| 7 | 18.2 | 5% |

**Intuition**：随层数增加，$f_{buckle}^{critical}$ 线性增长，但 $f_1$ 受 motor torque 限制 → slip 概率上升 → 成功率骤降。这个实验直接验证了 force model 的正确性。

---

## 4. 数据合成：基于 trajectory optimization 的 expert data 生成

### 4.1 为什么不用 RL 或 teleoperation？

- **RL**：需要 soft-body simulation + VBTS elastomer simulation，sim-to-real gap 巨大，计算昂贵
- **Teleoperation**：需要专用设备，数据多样性受限

PP-Tac 选择 **rigid-body trajectory optimization**，直接 sim-to-real。

### 4.2 轨迹生成 pipeline

1. 生成 randomized terrain profiles（cubic spline, 5 control points, height $\sim U(0, 3cm)$）
2. 预录制一个 teleoperated grasp motion 作为 template
3. 提取 fingertip 的 $(x,y)$ 作为 target，project 到 terrain 得到 $z$
4. 通过控制 finger-terrain 距离控制 $\delta d_{tac}$（Fig.5），实现 different contact forces

### 4.3 优化问题公式化

$$\hat{\gamma} = \arg\min_\gamma (L_{ee} + L_\Delta + L_{R,p_{wrist}})$$

$$L_{ee} = w_{ee} \cdot \text{MSE}(\mathbf{fk}(\gamma), ee_{target})$$

$$L_\Delta = w_\Delta \cdot \text{MSE}(\bar{\gamma}, \gamma)$$

$$L_{R,p_{wrist}} = w_{R,p_{wrist}} \cdot \text{MSE}((\bar{R}, \bar{p}_{wrist}), (R, p_{wrist}))$$

**变量解释**：
- $\gamma$：optimization variable，包含 $N_{data}=100$ 帧的 $[\mathbf{q}, R, p_{wrist}]$
- $\mathbf{q} \in \mathbb{R}^{16}$：16 controllable hand joint angles
- $R \in \mathbb{R}^6$：wrist 6D rotation（来自 6D representation https://arxiv.org/abs/1812.07035）
- $p_{wrist} \in \mathbb{R}$：wrist 沿 z-axis 的 translation
- $\mathbf{fk}(\gamma)$：forward kinematics 计算 4 个 fingertip positions
- $ee_{target}$：fingertip target positions
- $\bar{\gamma}$：initial pose
- $(\bar{R}, \bar{p}_{wrist})$：initial wrist pose
- $w_{ee}, w_\Delta, w_{R,p_{wrist}}$：权重超参

**三项 loss 的 intuition**：
- $L_{ee}$：让 fingertip 沿设计好的 trajectory 移动
- $L_\Delta$：防止 joint 角度发散到不自然姿态（regularization）
- $L_{R,p_{wrist}}$：限制 arm 在 workspace 内，避免奇异姿态

**数据规模**：500,000 samples × 100 frames，用 SGD 优化，过滤掉 collision 序列。

---

## 5. PP-Tac Policy：Diffusion-based motion generation

### 5.1 状态空间设计

State variables per frame：

$$
(p, \dot{p}, q, \dot{q}, R, \Omega, p_{wrist}, \dot{p}_{wrist}, d_{tac})
$$

| Variable | Dimension | Meaning |
|----------|-----------|---------|
| $\mathbf{p}$ | $\mathbb{R}^{17 \times 3} = 51$ | Hand joints position in world frame |
| $\dot{\mathbf{p}}$ | $\mathbb{R}^{17 \times 3} = 51$ | Linear velocity of hand joints |
| $\mathbf{q}$ | $\mathbb{R}^{16}$ | Controllable hand joint angles |
| $\dot{\mathbf{q}}$ | $\mathbb{R}^{16}$ | Angular velocity of hand joints |
| $R$ | $\mathbb{R}^6$ | 6D wrist rotation |
| $\Omega$ | $\mathbb{R}^6$ | Angular velocity of wrist |
| $p_{wrist}$ | $\mathbb{R}$ | Wrist height along z-axis |
| $\dot{p}_{wrist}$ | $\mathbb{R}$ | Velocity of $p_{wrist}$ |
| $\mathbf{d}_{tac}$ | $\mathbb{R}^4$ | 4 sensors' deformation depth |
| **Total** | **$\mathcal{D} = 152$** | |

**Over-parameterized 设计的 intuition**：包含 position + velocity, joint angle + joint position, 让网络学习更鲁棒的 latent features。这和 MDM (Human Motion Diffusion Model https://arxiv.org/abs/2209.14996) 的思想一致。

### 5.2 Diffusion 架构

基于 DDPM（参考 Imagen Video https://arxiv.org/abs/2210.02303 和 Score-based SDE https://arxiv.org/abs/2011.13456），但有几个关键改动：

**Architecture**：
- 4-layer Transformer encoder
- Latent dimension 512
- 4 attention heads
- Input dim: $(1 + N_{prefix} + N_{pred}) \times \mathcal{D}$

**Input composition**：
1. D-dim latent from $(t, i, \bar{d}_{tac})$ via 3-layer MLP
   - $t$：diffusion step
   - $i$：current frame index
   - $\bar{d}_{tac}$：target deformation depth
2. $N_{prefix} \times \mathcal{D}$：historical states
3. $N_{pred} \times \mathcal{D}$：noised future states

### 5.3 关键设计：predict $x_0$ 而非 $\epsilon$

传统 DDPM 预测 noise $\epsilon_t$，PP-Tac 预测 state sequence 本身 $\hat{x}_0^{pred}$（follows MDM https://arxiv.org/abs/2209.14996）。

**Loss function**：

$$L = \|\hat{x}_0^{pred} - x_0^{pred}\|_2^2 + \lambda_{consist} L_{consist}$$

$$L_{consist} = \|\mathbf{fk}(\mathbf{q}_0^{pred}) - \mathbf{p}_0^{pred}\|_2^2$$

**变量解释**：
- $x_0^{pred}$：ground truth future state sequence
- $\hat{x}_0^{pred}$：网络预测的 future state
- $\mathbf{q}_0^{pred}$：预测的 joint angles
- $\mathbf{p}_0^{pred}$：预测的 joint positions
- $\mathbf{fk}$：forward kinematics
- $\lambda_{consist}$：consistency weight

**Intuition**：$L_{consist}$ 强制 joint angles 和 positions 在 forward kinematics 上一致，这是 physics-informed loss，防止网络预测 kinematically impossible states。这比单纯的 MSE loss 在 robotic control 中更合理。

### 5.4 推理加速

训练时 $t \sim U(0, 1000)$，推理时加速：
1. $t=1000$ 直接 denoise 到 $x_0^{pred}$
2. 把 noise 加到 $t=1000-100N_i$ 再 denoise
3. 总共 10 steps，11ms on RTX 4090

设置 $N_{pred} = N_{prefix} = 5$，receding horizon control。

### 5.5 Closed-loop force control

如果 slip detection 触发（$P_{slip} > 0.75$），自动增加目标 deformation depth：

$$\bar{d}_{tac} \leftarrow \bar{d}_{tac} + \Delta d_{tac}$$

这个 closed-loop 让 system 自适应不同 stiffness 的 material。

---

## 6. Domain Randomization：4 种扰动策略

这是 sim-to-real 的关键。在 training 时对 $x^{prefix}$ 注入扰动：

1. **Random Gaussian noise to $\gamma$**：模拟控制误差
2. **第一帧 noise + 后续帧放大**：模拟 finger 沿 rising/descending terrain 移动
3. **Temporal consistent static frames (2~$N_{prefix}$)**：模拟 finger 卡住，同时 $d_{tac}$ 设为 max threshold
4. Frame index 加入 input 避免 stuck 问题

**Ablation 验证**：去除 disturbance 后，stiff objects (kraft paper bag) 几乎完全失败。训练时间增加 ~400,000 iterations，但泛化能力大幅提升。

---

## 7. 实验结果深度解析

### 7.1 Depth Reconstruction 精度

- L1 mean error: 0.35mm
- Median error: 0.28mm  
- 60% errors < 0.3mm
- 处理时间 < 10ms

对比 DenseTact 的 0.1-0.5mm range，R-Tac 在低成本下达到了 comparable 精度。

### 7.2 Materials × Terrains 实验（Fig.9）

4 种 materials × 4 种 terrains × 20 trials = 320 experiments。

**Material 难度排序**（由易到难）：
- Cloth / Plastic bag → 易（low stiffness, 容易 buckle）
- Paper → 中等
- Kraft paper bag → 难（stiff, multilayer, 难以 initiate buckling）

**Terrain 难度排序**：
- Flat plane / 10° slope → 易（consistent friction）
- Book underneath → 中等（edge 提供 initial buckle point）
- Random terrain → 难（force controller 难以 compensate）

**整体 success rate 87.5%**，这是首个用 VBTS-equipped dexterous hand 抓取 paper-like objects 的工作。

### 7.3 Baseline 对比

| Baseline | 设置 | 结果 |
|----------|------|------|
| Bi-finger gripper + teleop | Vision-based | 软材料有时成功，硬材料完全失败 |
| Open-loop | 重放轨迹，无触觉 | 控制误差累积，成功率低 |
| Model-based force tracking | PP-Tac wrist + tactile fingertip control | 结构化 terrain OK，复杂 terrain 失败 |
| Non-disturbance | Full PP-Tac 但无 domain randomization | Stiff 物体完全失败 |
| **PP-Tac (full)** | **完整系统** | **87.5%** |

**关键发现**：Allegro Hand 有 >0.1 rad 的 joint angle 误差（参考 LEAP Hand https://arxiv.org/abs/2309.06440），累积到 fingertip 位置误差是厘米级，对 paper picking 这种 precision-sensitive task 是致命的，所以触觉 feedback 不可缺。

---

## 8. Contact Establishment 算法（Appendix B）

这是一个优雅的几何控制问题：让至少 3 个 fingertip 接触 unknown surface。

**3 步策略**：
1. **First contact**：hand 下降直到任意 finger 接触
2. **Second contact**：以 first contact 为 pivot 旋转（公式11-14）
   - 旋转轴：$v_1 = R_z(90°)(x_n - x_c, y_n - y_c, z_n - z_c)^T$
   - Rodrigues' formula 计算 $R(\theta, v_1)$
3. **Third contact**：以 first-second contact 为 axis 旋转（公式15-16）
   - 旋转轴：$v_2 = \overrightarrow{f_t f_i}$

**Rodrigues' rotation formula**（公式12）：

$$R(\theta, v_1) = I + \sin(\theta)[v_1]_\times + (1-\cos(\theta))[v_1]_\times^2$$

其中 $[v_1]_\times$ 是 $v_1$ 的 skew-symmetric matrix。

这个算法本质上是**通过 finger contact feedback 自适应定位 surface plane**，比预先假设平面位置鲁棒得多。

---

## 9. 与相关工作的对比与联想

### 9.1 VBTS 谱系

| Sensor | Shape | Light | Calibration |
|--------|-------|-------|-------------|
| GelSight (MIT) | Flat | RGB | Complex |
| DIGIT (Meta) | Hemispherical | RGB | Medium |
| DenseTact | Hemispherical | RGB | CNC required |
| DTact | Flat | Monochrome | Simple |
| 9DTact | Cylindrical | Monochrome | Simple |
| **R-Tac** | **Hemispherical** | **Monochrome** | **Simple, 30 captures** |

R-Tac 的核心创新是把 DTact 的 monochrome principle 迁移到 hemispherical form factor，同时简化标定。

### 9.2 Diffusion Policy 谱系

| Method | Application | Key Idea |
|--------|-------------|----------|
| Diffusion Policy (Chi et al. https://arxiv.org/abs/2303.04137) | Visuomotor | CNN-based, predict $\epsilon$ |
| MDM (Tevet et al.) | Human motion | Predict $x_0$ |
| HumanDiffuse (Lin et al. https://arxiv.org/abs/2312.04867) | Two-hand | Consistency loss |
| **PP-Tac Policy** | **Paper picking** | **Transformer + predict $x_0$ + $L_{consist}$** |

PP-Tac 的 closest inspiration 是 MDM，但加入了 robot-specific 的 consistency loss 和 tactile conditioning。

### 9.3 Dexterous Hand + VBTS 组合

| Work | Hand | Task |
|------|------|------|
| Do et al. (DenseTact + Allegro) | Allegro | Small screw manipulation |
| Qi et al. (Fingertip VBTS + DIGIT + Allegro) | Allegro | In-hand rotation |
| NeuralFeels (Suresh et al. https://arxiv.org/abs/2401.03876) | Allegro | Visuotactile in-hand |
| **PP-Tac** | **Custom 16-DoF + R-Tac × 4** | **Paper-like object picking** |

PP-Tac 是首个把 VBTS-equipped dexterous hand 用于 thin deformable object 的工作。

---

## 10. 局限性与未来方向

论文自述的局限：
1. **Initial force 需要手动设置**：$\bar{d}_{tac}$ 的初值是经验参数，太小则 slip 时间长，太大则 motor 过载
2. **Adaptive force 算法对高 stiffness 材料效果有限**：kraft paper bag 在 non-flat surface 上仍困难

**我能想到的额外 limitations 和 extensions**：

1. **Sim-to-real 的 tactile gap**：synthetic data 没有真实 VBTS 的 noise model，diffusion policy 在 real tactile input 上的泛化依赖 domain randomization 的强度
2. **No vision fusion**：完全抛弃 vision 可能限制了在 cluttered scene 中的定位能力。一个 vision + tactile fusion 的版本（类似 NeuralFeels https://arxiv.org/abs/2401.03876）可能更鲁棒
3. **Slip detection 的 generalization**：CNN 在 20 分钟数据上训练，对未见过的 material（如 wet paper, glossy plastic）可能失效
4. **Single-object assumption**：没有处理多张纸叠在一起的情况
5. **In-hand manipulation after grasp**：抓起来之后能不能 in-hand 旋转、fold、hand over？
6. **Geometry-based reasoning**：当前 policy 是 black-box diffusion，缺乏对物理的 explicit reasoning。一个 neuro-symbolic approach 可能更 robust
7. **Multi-modal tactile**：R-Tac 只 sensing normal deformation，没有 shear force sensing。如果加上 3-axis force（类似 GelSlim 3.0），slip detection 可能更准

---

## 11. 系统级 Intuition 总结

PP-Tac 的优雅之处在于它把 **biomimetic strategy + physics-informed learning + cheap hardware** 三者结合：

1. **Biomimetic**：模仿人类 sliding + pinching 的动作序列
2. **Physics**：基于 buckling 力学分析确定 force control 策略
3. **Learning**：Diffusion policy 在 over-parameterized state space 上学习，避免 sim-to-real 的 tactile simulation 难题

**关键设计 trade-off**：
- 单色 sensing 牺牲了 RGB 的多通道信息，换来标定简单、带宽低、120Hz 帧率
- Predict $x_0$ 而非 $\epsilon$ 牺牲了理论上的最优性，换来 motion sequence 的平滑性
- 4 个 sensor 部署在 4 个指尖，每个 USB 独立连接，可能存在 sync 问题，但用 100ms latency 的 closed-loop 容忍了这个 jitter

**最让我印象深刻的部分**：trajectory optimization 完全用 rigid-body dynamics，绕过了 deformable object simulation 的难题，然后用 domain randomization 让 policy 鲁棒到 real-world 的 terrain variations。这种"avoid the hard simulation problem entirely"的思路在 robotics 里非常实用。

参考链接汇总：
- Project page: https://peilin-666.github.io/projects/PP-Tac
- Allegro Hand: https://www.allegrohand.com/ah-v4-main
- DTact: https://arxiv.org/abs/2303.03739
- 9DTact: https://arxiv.org/abs/2309.18416
- DenseTact: https://arxiv.org/abs/2206.02739
- DIGIT: https://arxiv.org/abs/2105.04797
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- MDM: https://arxiv.org/abs/2209.14996
- LEAP Hand: https://arxiv.org/abs/2309.06440
- NeuralFeels: https://arxiv.org/abs/2401.03876
- 6D rotation representation: https://arxiv.org/abs/1812.07035
- EPnP: https://arxiv.org/abs/1901.05724
- Score-based SDE: https://arxiv.org/abs/2011.13456
- Imagen Video: https://arxiv.org/abs/2210.02303
- HandDiffuse: https://arxiv.org/abs/2312.04867
- GelSlim 3.0: https://arxiv.org/abs/2102.02026
