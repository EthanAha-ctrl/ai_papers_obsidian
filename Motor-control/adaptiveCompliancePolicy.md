---
source_pdf: adaptiveCompliancePolicy.pdf
paper_sha256: 79871d41eba042447af678baaebe3449565d5d4e1a22aa5d8e2b45ca477456b7
processed_at: '2026-07-18T02:04:43-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Adaptive Compliance Policy 深度解析

Karpathy 你好，这篇 paper 我详细讲一下。这是 Stanford Shuran Song lab 和 Toyota Research Institute (TRI) 的合作工作，一作 Yifan Hou 是 Matthew Mason 的 PhD student（CMU 时期），现在在 Stanford。这个 lineage 很重要 —— Mason 是 manipulation 领域的奠基人，1981 年那篇 "Compliance and force control" [34] 是这个领域的开山之作，ACP 本质上是 Mason 经典思想的 learning-based 现代化版本。

Project page: https://adaptive-compliance.github.io/

---

## 1. 核心问题与 Motivation

### 1.1 传统 visuomotor policy 的盲点

现在主流的 visuomotor policy（Diffusion Policy [6,7], ACT, RT-1, OpenVLA 等）都在做同一件事：predict position action，让 robot 跟踪这个 position。这背后假设是：只要 position 准，task 就成。但 contact-rich manipulation 完全不是这样。

考虑两个 task:
- **Flipping**: 你要 push 一个物体 pivot against wall，contact force 必须维持在某个范围 —— 太小打滑，太大 wall 滑走
- **Wiping**: 要 maintain contact 同时不能压坏 vase，contact force 需要在 surface normal 方向有 compliance

Position-only policy 在这些 task 上的根本问题：position error 直接通过 robot 的 mechanical stiffness 转换成 force，stiffness 越高，position noise 变成的 force noise 越大。Stiff policy（实验里的 baseline）在第 4 次 wiping test 就把 tool 弄断了（Fig. 8 middle row）—— 这是 position-only policy 在 contact-rich task 上的典型 failure mode。

### 1.2 Compliance 是什么

Mechanical compliance 描述 force 和 motion 之间的 elastic 关系。直观上：
- **Low compliance (high stiffness)**: position 准，但遇到外部 force 容易产生大 internal force
- **High compliance (low stiffness)**: 遇到外部 force 时 position 偏离，但避免大 force

Compliance 的难度在于它有三个维度的 variability:
- **Temporal**: 接触前要 stiff（精确定位），接触后要 compliant
- **Spatial**: pivoting 时 pushing 方向 compliant，arc 方向 stiff
- **Task-dependent**: flipping 和 wiping 的 compliance pattern 完全不同

### 1.3 为什么从 demonstration 学 compliance 难

从单个 demonstration 估计 stiffness 是 ill-conditioned 问题 [4]。原因：stiffness 描述 force-motion 二阶关系，单个 trajectory 里 force 和 position 的 variation 不够。之前的工作 [4, 11, 12, 25] 需要重复同一个 motion 多次来 statistically 估计（用 force covariance 或 position covariance）。这对 visuomotor policy learning 不实用，因为每个 demonstration 都不一样（不同 object、不同 pose）。

Grioli & Bicchi 2011 [13] 提出了 single-demo stiffness estimator，但需要 perfect knowledge of human mass 和 damping，在 kinesthetic teaching 下不现实（人手改变了 effective damping 和 mass）。

之前 pragmatic 的做法：
- **Pre-selected compliance** [20, 24]: 手动调参，不能 adapt
- **Uniform constant stiffness** [26]: Lee et al. 2019 "Making sense of vision and touch" 用 k=500N/m 均匀 stiffness，不能 capture spatial variation

---

## 2. 核心理论：Theorem 1 的 intuition

这是整篇 paper 的关键 insight。我详细讲一下，因为它 elegant 但 paper 里证明比较压缩。

### 2.1 物理建模

公式 (1) 是标准的 spring-mass-damper 模型:

$$f = M \ddot{x} + K(x - x_{ref}) + K_D \dot{x}$$

变量含义：
- $f \in \mathbb{R}^N$: generalized force 作用在 robot 上（N 是 robot 自由度数）
- $x \in \mathbb{R}^N$: generalized position
- $\ddot{x}, \dot{x}$: 对应的 acceleration 和 velocity
- $M \in \mathbb{R}^{N \times N}$: inertia matrix（"virtual mass"）
- $K \in \mathbb{R}^{N \times N}$: stiffness matrix（"virtual spring"）
- $K_D \in \mathbb{R}^{N \times N}$: damping matrix（"virtual damper"）
- $x_{ref}$: reference position（spring force 为零的位置）

这是 admittance control [32] 的标准模型。Low-level controller 接收 $(x_{ref}, K, M, K_D)$，对 UR5e 这种 position-controlled robot 用 admittance（force feedback → position command），对 force-controlled robot 可以用 impedance control [14]。

公式 (2) 和 (3) 是 contact 的建模:
$$J^T \lambda = f \quad \text{(Newton's 2nd law)}$$
$$J \nu \geq 0 \quad \text{(unilateral contact constraint)}$$

变量：
- $J \in \mathbb{R}^{n \times N}$: Contact Jacobian，$n$ 是接触点数
- $\lambda \in \mathbb{R}^n$: contact normal force vector（每个接触点一个法向力）
- $\nu \in \mathbb{R}^N$: generalized velocity

公式 (3) 的 intuition: contact point 不能穿透 environment，所以法向速度非负。

### 2.2 三个 Assumption

- **Assumption I**: Contact force dominates。inertia, friction, gravity 可忽略。通过 slow motion + lightweight objects 保证。注意：这针对的是 manipulation 级别的 force 分析，不是 robot 自身的 compliance control（那个是 fast dynamic）。
  
- **Assumption II**: 所有接触都有非零法向力。$\lambda_i > 0, \forall i$。通过 demonstration 时 clearly make contact 保证。

- **Assumption III**: No pinching contacts。J 的 row 形成的 cone 包含在它的 dual cone $\{x \in \mathbb{R}^N | Jx \geq 0\}$ 内。Fig. 4 给了例子：frictionless 环境下，三个 non-pinching 例子 + 一个 pinching 例子。Pinching intuition: 接触法向互相 "夹住" tool，单方向 compliance 不足以避免 internal force。两指 grasp 是典型 pinching。

### 2.3 Theorem 1 证明 walkthrough

**Theorem 1**: 在 Assumption I, II, III 下，只要 robot 不在 force feedback 方向控制 velocity，就存在 $\nu$ 满足 contact constraint (公式 3)。

**证明**:
不控制 force 方向的 velocity 意味着 velocity 在 force 方向有 free component:

$$\nu = \nu_0 + k f$$

其中 $\nu_0$ 是其他方向的 velocity，$k$ 是任意 scaling factor。

代入 $f = J^T \lambda$ (公式 2):

$$\nu = \nu_0 + k J^T \lambda$$

两边乘 $J$:

$$J \nu = J \nu_0 + k J J^T \lambda$$

由 Assumption II, $\lambda$ 所有分量为正，所以 $J^T \lambda$ 是 J 的 row vectors 的正线性组合，严格在 row cone 内部。

由 Assumption III，这个 cone 包含在 dual cone $\{x | Jx \geq 0\}$ 内，所以:

$$J J^T \lambda > 0$$

(因为 $J^T \lambda$ 在 dual cone 内意味着 $J (J^T \lambda) \geq 0$，且严格在内 means $> 0$)

于是 $J \nu = J \nu_0 + k J J^T \lambda$。对足够大的 $k > 0$，第二项 dominate，使得 $J \nu > 0$，满足 contact constraint (公式 3)。$\square$

### 2.4 Theorem 1 的 intuition

**核心 insight**: 你不需要在所有方向都 compliant，只需要在 force feedback 方向 compliant，就能保证 contact constraint 满足。其他方向可以 high stiffness 来提高 position accuracy。

这是 hybrid force-velocity control [17, 18, 39] 的 learning-based 推广。Mason 的工作需要 explicit contact model，ACP 用 force feedback 直接 identify "force direction"，然后 apply 这个简单规则。

这个 trick 让 stiffness estimation 从 "需要完整 force-motion 变异" 变成 "只需要知道 force 方向"，从 ill-conditioned 变成 well-defined。

### 2.5 Stiffness Matrix 构造

公式 (6):
$$K = S K_0 S^{-1}$$

- $S \in \mathbb{R}^{N \times N}$: 正交基矩阵，列向量组成 $\mathbb{R}^N$ 的一组 orthonormal basis，第一列是 $f/|f|$（force 方向单位向量）
- $K_0$: diagonal matrix $\text{diag}(k_{low}, k_{high}, ..., k_{high})$
- $S^{-1} = S^T$（正交矩阵性质）

构造：在 force 方向（第一列对应）用 $k_{low}$，其他 $N-1$ 个正交方向用 $k_{high}$。

当 $|f|$ 很小时（无接触），所有方向都用 $k_{high}$。

### 2.6 Stiffness magnitude 的连续化

公式 (7):
$$k_{low} = k_{max} - (k_{max} - k_{min}) \frac{|f| - f_{min}}{f_{max} - f_{min}}$$

变量：
- $k_{max}, k_{min}$: stiffness 上下界（hardware 设定）
- $f_{max}, f_{min}}$: force 阈值
- $|f|$: 当前 force magnitude

Intuition:
- $|f| \leq f_{min}$: $k_{low} = k_{max}$（接近无接触，high stiffness）
- $|f| \geq f_{max}$: $k_{low} = k_{min}$（接近零刚度）
- 中间线性插值

为什么需要 continuous: force signal noisy，直接用 $|f|$ 做 hard switch 会导致 stiffness 跳变，对 low-level controller 不友好。

---

## 3. Virtual Target Pose - 巧妙的工程 trick

这是 paper 里我觉得最 elegant 的工程设计。

### 3.1 概念

Policy 输出 19D per arm:
- **Reference pose** (9D): 理想的 end-effector 位姿（3D position + 6D rotation，用 rotation matrix 的 top 2 rows 表示）
- **Virtual target pose** (9D): 实际给 low-level compliance controller 追踪的"虚拟"目标
- **Stiffness scalar** (1D): low stiffness direction 的 magnitude

### 3.2 为什么需要 virtual target

考虑 spring force 公式:
$$f_{spring} = K(x_{actual} - x_{virtual})$$

当 robot 实际位置 $x_{actual} = x_{ref}$（达到 reference pose）时:
$$f_{ref} = K(x_{ref} - x_{virtual})$$

这就是 desired reference force！换句话说，给定 reference pose 和 desired force，可以反解 virtual target:
$$x_{virtual} = x_{ref} - K^{-1} f_{ref}$$

**Intuition**: Virtual target 把 force control 问题转成 position control 问题。Robot 还是只追 position target（virtual target），但通过 spring 力学关系，达到 reference pose 时自然产生 desired force。

### 3.3 好处

1. **跨 robot 平台通用**: impedance-controlled robot 没有 F/T sensor 也能执行（只要追 virtual target）
2. **Diffusion policy 输出统一**: 都是 pose，可以共用 encoder/decoder
3. **Training 时的 hindsight benefit**: 1 秒 moving average 让 virtual target smooth，还给 action label 提供了"即将接触"的 hindsight 信息，这对 smooth contact engaging 重要 —— 否则 policy 不知道下一帧要 contact 了，position 会"撞"上去

### 3.4 Inference 时的 stiffness 重建

在 inference 时，force 方向用 reference pose → virtual target 的方向（不是实测 force 方向），这样 policy 可以主动"选择"compliance direction，而不是被动 react to force。这是个 subtle 但重要的设计 choice。

---

## 4. Policy Architecture 详解

### 4.1 Inputs

- **Fisheye RGB images**: 2 frames, 224×224
- **Robot end-effector poses**: 3 frames
- **Force/torque data**: 6D，1秒 window

### 4.2 Force Encoding - 两种策略

**Strategy 1: Causal Convolution (ACP w.o. FFT)**
- 32 timesteps at 250Hz
- 5-layer causal convolution network
- 输出 768-dim vector
- 参考: WaveNet [37], Lee et al. 2019 [26]

Causal conv 的优势: 严格 causal，能 capture temporal pattern。WaveNet-style dilated conv 可以扩大 receptive field。

**Strategy 2: FFT Encoding (ACP)**
- 6D F/T × 1 秒 at 7000Hz
- 每维做 FFT 转 2D spectrogram (6×30×17)
- ResNet-18 with input channels = 6（修改版）
- 加 CoordConv layer [28] 处理 translational invariance

**为什么 FFT 更好** (实验证实: vase wiping 93.75% vs 81.25%):
- Wiping 是 periodic motion，frequency domain 直接 capture 周期性
- Force frequency content 可能 reveal contact mode（slip/stick/slide）
- 类似 audio processing 用 spectrogram

参考 links:
- WaveNet: https://arxiv.org/abs/1609.03499
- CoordConv: https://arxiv.org/abs/1807.03247

### 4.3 Image Encoding

- CLIP pretrained ViT-B/16 [10]
- 2 frames RGB, 224×224, random crop
- CLIP ViT 在 robotics 上常用作 frozen feature extractor

CLIP link: https://openai.com/research/clip

### 4.4 Fusion

- Image embedding + force embedding → Transformer encoder (self-attention)
- 学习 adaptive visual-force representation
- Concatenate with 3 frames end-effector poses
- 作为 Diffusion Policy head 的 condition

### 4.5 Diffusion Policy Head

继承自 Chi et al. 2023 [6,7]:
- Receding-horizon prediction
- Action chunking (预测 trajectory)
- DDPM 去噪 process

Diffusion Policy link: https://diffusion-policy.cs.columbia.edu/

### 4.6 Training Pipeline

1. 整个 episode 的 wrench data 经过 moving average filter (1 秒 window)
2. 用公式 (6) 计算 stiffness label
3. 用 3D mechanical spring 公式计算 virtual target label
4. Diffusion policy 学 reference pose + virtual target + stiffness scalar

**1 秒 window 的两个好处**:
- Virtual target smooth
- 提供 hindsight: 当前 action label 知道未来 1 秒的 contact 模式，这对 smooth contact engaging 关键

---

## 5. Demonstration Collection - Kinesthetic Teaching

### 5.1 Setup (per arm)

- UR5e manipulator（position feedback）
- RGB camera（visual）
- ATI mini-45 F/T sensor（force feedback）

### 5.2 Demonstration 时的 compliance setting

- **Low stiffness**: 操作者能自由 move robot
- **Low damping, low virtual mass**: 因为 human hand 提供 natural stabilization
- **TCP near handle**: 让 robot 在外力下 rotate 直观

### 5.3 Testing 时的 compliance setting

- **Higher damping, higher virtual mass**: 保证 admittance controller stability（没有 human hand 稳定）

### 5.4 为什么 kinesthetic 比 teleoperation 好

- 操作者直接感受 contact force，能 demonstrate variable compliance behavior
- Haptic feedback 是 compliance learning 的关键信号
- Teleoperation 没有这个，操作者只能 visually estimate force

参考: 类似思想见 Kronander & Billard 2013 [24]

---

## 6. 实验详解

### 6.1 Hardware Setup

- UR5e robot
- GoPro RGB camera: 60Hz
- ATI mini-45 F/T sensor: 7000Hz
- Robot Cartesian control: 500Hz
- Compliance: 3D translational space (N=3)，论文说可扩展到 6D

### 6.2 Task 1: Item Flipping

**Task**: Point finger 翻转物体，通过 pivoting against fixture corner。

**Challenges**:
1. 维持 contact force 不论物体 shape/weight/fixture location
2. 大 contact force 会让 fixture slide

**Data**: 230 episodes, 15 items, 300 epochs

**Test scenarios** (20 tests × 5 = 100 per algorithm):
1. Training items
2. Unseen items（未见过的物体）
3. Push & Flip（物体离 fixture 5cm，先 push 再 flip）
4. Varied Fixture Pose
5. Unstable Fixture（轻 fixture 会动）

**Results**:
| Method | Train | Unseen | Push&Flip | Fixture Pose | Unstable | All |
|--------|-------|--------|----------|--------------|----------|-----|
| **ACP** | 90 | 95 | 95 | 100 | 100 | **96** |
| ACP w.o. FFT | 90 | 100 | 100 | 95 | 90 | 95 |
| Compliant | 80 | 15 | 15 | 5 | 0 | 23 |
| Stiff | 20 | 0 | 5 | 35 | 10 | 14 |

**Analysis**:

- **Baseline 在 train items 上勉强能用**（80%, 20%），靠的是 passive compliance 和 collision-induced force
- **Compliant policy 在 unseen 上崩盘**（15%）—— uniform stiffness 无法 adapt 新物体的 contact geometry
- **Stiff policy 几乎全失败**（14%）—— position error 直接转成 force violation，触发 robot safety stop
- **ACP 在所有 scenario 上 robust**，特别是 unstable fixture（100%）—— adaptive compliance 能 react to fixture 移动
- **ACP vs ACP w.o. FFT**: flipping 任务上差不多（96 vs 95）。论文解释: flipping 的 force signal 高频成分少，FFT 优势不明显

### 6.3 Task 2: Vase Wiping

**Task**: Bimanual robot，左臂 hold vase，右臂 wipe 上面的 mark。

**Tool**: 3D printed tool with kitchen sponges as wipers

**Data**: 200 demonstrations, various vase poses/marking shapes/colors, each demo 1-5 wipes

**Test scenarios**:
1. Small Mark ×5 (one wipe 够)
2. Large Mark ×5 (multiple wipes)
3. Perturbation before contact (PbC) ×4
4. Perturbation after contact (PaC) ×2

**Results**:
| Method | Small | Large | PbC | PaC | All |
|--------|-------|-------|-----|-----|-----|
| **ACP** | 100 | 80 | 100 | 100 | **93.75** |
| ACP w.o. FFT | 100 | 60 | 75 | 100 | 81.25 |
| Compliant | 60 | 20 | 25 | 100 | 43.75 |
| Stiff | broke tool | - | - | - | - |

**Analysis**:

- **Stiff policy broke tool on 4th test** —— position noise 转成过大 force，物理损坏
- **Compliant policy safe but ineffective** —— friction 让 wiping position 偏离，最终 lose contact
- **ACP maintain contact + accurate tracking** —— 这是 simultaneous force-position control 的价值
- **FFT > causal conv on wiping** —— wiping 是 periodic motion，frequency domain 更有效

### 6.4 Fig. 6 - Flipping 中的 stiffness 动态

Top plot: world frame 下 X, Y, Z 三个 axis 的 stiffness 随时间变化
- 接近阶段: X-axis stiffness 下降（force 在 X 方向）
- Pivoting 阶段: compliance 转到 Z-axis（contact normal 变了）
- 这就是 spatial-varying compliance 的实证

Bottom plot: reference pose (yellow), virtual target (orange), compliance direction (yellow line)
- Virtual target 始终在 reference pose "后方"，确保 push force 维持
- Compliance direction 跟着 contact normal 漂移

### 6.5 Fig. 7 - Wiping 中的 compliance

Top: wiping arm 的 stiffness 在 world frame 下变化
Bottom: reference (yellow) 和 virtual (orange) pose
- Wiping arm 的 compliant direction 大致沿 contact normal
- 即使 position estimate 有 error，virtual target 仍把 arm 拉向 vase，保持 contact

---

## 7. 关联工作和扩展思考

### 7.1 与 Diffusion Policy 的关系

ACP 本质上是 Diffusion Policy [6,7] 的扩展。Action space 从纯 position 扩展到 (position, virtual_target, stiffness)。Diffusion 的 multi-modal action distribution 对 contact-rich task 自然支持 —— 同一个 visual-force observation 可以对应多种 compliant action。

### 7.2 与 Mason Hybrid Force-Velocity Control 的历史联系

Mason 1981 [34] 的 hybrid force-velocity control 是经典: 选定 directions 用 force control，其他用 velocity control。需要 explicit contact model 和 contact normal 方向。

ACP 是这个思想的 learning-based 版本:
- 不需要 explicit contact model
- 用 force feedback 自动 identify "force direction"
- 用 learning 推广到 complex task

Yifan Hou 之前的 work [17, 18] 是 model-based 版本，ACP 是 model-free 版本。

### 7.3 与 Operational Space Control 的联系

Khatib 1987 [21] 的 Operational Space Formulation 在 task space 而不是 joint space 设计控制律。Virtual target pose 的设计思想类似 —— 在 task space（end-effector）层面定义 virtual spring，让 low-level controller 实现。

### 7.4 与 recent contact-rich manipulation work

**Concurrent works**:
- **ForceMimic** [29] (2024): 用 force-motion capture system + manual task stage division。比 ACP 更工程化但 less general。
- **FORGE** [36] (2024): Force-guided exploration for RL-based contact-rich manipulation
- **ManiWAV** [30] (2024): Audio-visual data，sound 也是一种 force signal

### 7.5 FFT Encoding 的深层 motivation

Force signal 用 FFT encoding 让我想到几个 parallel:
- **Audio processing**: raw audio → spectrogram → CNN 是经典 pipeline（WaveNet, vocoder, speech recognition）
- **Tactile sensing**: 人类 Pacinian corpuscle 对 200-300Hz 最敏感，说明 frequency content 是 haptic perception 的关键
- **Vibration analysis**: contact mode（slip/stick/slide）在 frequency domain 有 distinct signature

论文只用了 wrist F/T，没 tactile sensor。如果把 tactile sensing (DIGIT, GelSight) 一起 FFT encode，可能 capture 更 fine-grained contact mode。

### 7.6 Foundation Model 角度

用 CLIP ViT 做 image encoder 是 "foundation model as frozen feature extractor" 的 standard 做法。值得思考: 是否能 pretrain 一个 universal compliance predictor？收集多个 task 的 demonstration，学一个 task-conditional compliance foundation model。

### 7.7 Limitations 我自己的思考

**论文没明确讨论但值得关注的**:

1. **Assumption I 限制**: 快速 motion 或重物体时不成立。无法做 fast throwing, heavy pushing。
2. **Assumption III 限制**: 不能做 pinching grasp。两指捏物体这种基本 action 不行。这是 fundamental limitation。
3. **3D translational compliance**: 只用 N=3，没 rotational compliance。对 screwing, insertion 这种需要 rotational compliance 的 task 不够。
4. **1 秒 force window**: 对长 horizon task 够吗？论文没测试 long-horizon。
5. **Cross-robot transfer**: 学的 compliance profile 能 transfer 到不同 arm 吗？比如 UR5e → Kuka。
6. **Tactile sensing 缺失**: 只有 wrist F/T，没 finger tactile。Local contact info 缺失。

### 7.8 Philosophy 的 takeaway

这篇 paper 有个非常实用的工程哲学: **compliance 不需要精确估计，只需要 useful approximation**。

这跟 ML 里很多 trade-off 类似:
- 完整 stiffness estimation 是 ill-conditioned，但 useful approximation 是 well-defined
- "Perfect model" 不存在，但 "good enough model" 能解锁很多 capability
- Theoretical guarantee (Theorem 1) + practical engineering (virtual target, FFT, moving average) 组合拳

这种 philosophy 对 robotics learning 领域很有启发。我们不需要 perfect physical model，需要的是 "useful inductive bias that make learning easier"。

### 7.9 与 RL-based compliance learning 的对比

RL-based 方法 [3, 5, 19, 33, 36] 可以探索 force-motion variation 来学 compliance controller，但需要 per-task retrain，且 sample efficiency 差。ACP 用 human demonstration + physical insight，sample efficient 且 generalize across object variations。

### 7.10 关于 stiffness 矩阵的 SPD 性质

公式 (6) 构造的 K = S K_0 S^T 一定是 symmetric positive definite (SPD):
- S 是正交矩阵
- K_0 是 diagonal with positive entries
- K = S K_0 S^T 是 congruence transformation，保持 SPD

这个 SPD 性质对 compliance controller 稳定性必要。论文没 explicit 提，但是 implicit 在设计里。

### 7.11 关于 moving average filter 的细节

1 秒 window 的 moving average 对 wrench data filtering 是个 strong smoothing。这有几个 implication:
- **Hindsight**: 当前 timestep 的 label "看到"未来 1 秒的 contact pattern
- **Causality 问题**: training 有 hindsight OK，inference 时 policy 必须从 visual-force 输入 predict 这个 hindsight
- **Smoothness**: virtual target 不会跳变，low-level controller 能稳定追踪

这跟 Diffusion Policy 里 action chunking 思想类似 —— 给 policy "future" 信息让它 plan smooth trajectory。

---

## 8. 总结

### 8.1 三个主要 contribution

1. **Theoretical insight (Theorem 1)**: 只要 force 方向 compliant，其他方向 stiff，就能满足 contact constraint。把 ill-conditioned stiffness estimation 变成 well-defined。
2. **Practical implementation (Virtual target pose)**: 把 force control 转 position control，统一 action space，跨 robot 平台通用。
3. **Empirical validation**: 两个 contact-rich task 上 >50% improvement over baselines。

### 8.2 我的整体评价

Strengths:
- Theory + engineering + experiment 三位一体
- Theorem 1 是真正的 intellectual contribution，不是 ad-hoc trick
- Virtual target 设计 elegant
- 实验场景 challenging 且 realistic（unstable fixture, perturbation）

Weaknesses:
- Assumptions 限制应用范围（pinching, fast motion, heavy objects）
- 只在 2 个 task 验证
- 没有 cross-robot transfer 实验
- 与 recent force-aware policy（比如 Aoyama et al. [1]）的对比缺失

### 8.3 对你 (Karpathy) 的 relevance

你之前在 Tesla 做 Autopilot，在 OpenAI 做 language model。这篇 paper 是 robotics + learning 的 hybrid，体现了你常说的 "micro-rules + learning" 的设计哲学:
- Theorem 1 是个 micro-rule (force direction compliant)
- Policy learning 是 data-driven 的 macro-capability
- 两者结合: physical insight 让 learning problem well-defined，learning 让 physical insight generalize across configurations

这跟 language model 里 "grammar + learning" 的 trade-off 类似。Pure learning 不够 sample efficient，pure rule 不能 adapt。Hybrid 最强。

---

## 9. 进一步 reading 的 links

- **Project page**: https://adaptive-compliance.github.io/
- **Diffusion Policy**: https://diffusion-policy.cs.columbia.edu/
- **CLIP**: https://openai.com/research/clip
- **CoordConv**: https://arxiv.org/abs/1807.03247
- **WaveNet**: https://arxiv.org/abs/1609.03499
- **Mason 1981 Compliance and force control**: https://ieeexplore.ieee.org/document/4308479
- **Hogan 1984 Impedance control**: https://ieeexplore.ieee.org/document/4046987
- **Khatib 1987 Operational space**: https://ieeexplore.ieee.org/document/1087184
- **Lee et al. 2019 Making sense of vision and touch**: https://ieeexplore.ieee.org/document/8793606
- **Yifan Hou personal page**: https://www.cs.cmu.edu/~yifanh/
- **Shuran Song lab (Surreal)**: https://surreal.stanford.edu/

---

希望这个 build 你的 intuition。如果你对哪个部分感兴趣（特别是 Theorem 1 的 convex geometry 细节，virtual target 的力学推导，或 diffusion policy 的 condition 机制），我可以再深入展开。
