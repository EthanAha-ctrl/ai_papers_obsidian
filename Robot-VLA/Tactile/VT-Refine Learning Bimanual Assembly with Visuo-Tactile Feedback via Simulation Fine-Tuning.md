---
source_pdf: VT-Refine Learning Bimanual Assembly with Visuo-Tactile Feedback via Simulation
  Fine-Tuning.pdf
paper_sha256: 9397da493bbe5c078ad080a4e866005886036a29ea14974404fbc58737e4318c
processed_at: '2026-08-13T03:27:57-07:00'
target_folder: Robot-VLA/Tactile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej, 咱们用大白话来拆解一下这篇 paper。其实你只要把它类比成我们熟悉的 LLM 训练范式，intuition 立马就出来了：**这基本就是 Robotics 领域的 "Pre-train + RLHF" 范式在双手高精度接触任务上的首次完整跑通。**

### 1. 核心痛点：为什么双手组装这么难搞？

想象一下你闭着眼睛把 USB 插头插进电脑。视觉在最后 2mm 的缝隙插入阶段基本失效，因为手指头把摄像头挡得死死的。这时候人类靠的是触觉，靠手感去“扭一扭、晃一晃”，试探出那个对准的瞬间。

如果你纯靠 Behavior Cloning (BC) 去模仿人类的遥操作数据，会遇到两个致命问题：
1. **Data 缺乏探索性**：遥操作的人也是看着屏幕操作的，他压根没有触觉反馈，所以演示数据里根本没有那种“卡住了-微调-再试”的 wiggle 动作。数据本身就是次优的。
2. **Data scaling 太贵**：如果要让 BC policy 学会这种高精度微调，你需要海量的演示数据，在 real world 收集几千次高精度双手演示成本高到离谱。

### 2. 解法直觉：Real-to-Sim-to-Real + RL Fine-tuning

既然 real world 数据不够好，那就去 Sim 里面靠 RL 自己探索出这些微调动作。整个 pipeline 分两步：

**Stage 1: Pre-train (Behavior Cloning)**
用 30 个 real world 的演示数据，训练一个 Diffusion Policy。这个 policy 很笨，大概只能把零件抓起来，勉强靠近目标位置，成功率只有 30-40% 左右。这就相当于 LLM 的预训练阶段，给模型注入了一个粗糙的先验。

**Stage 2: RL Fine-tune (DPPO)**
把这个预训练好的 policy 扔到高仿真的 Simulator 里，用 PPO 算法进行大规模强化学习微调。Reward 极其简单：插进去了给 1 分，没插进去给 0 分。这就是 Sparse Reward。
因为你已经有了 Stage 1 的先验，RL 不会在大海里捞针，它只会在“接近插入”的状态空间附近探索。Sim 里的物理引擎让它可以并行试错几千次，它自己就把那种“扭一扭”的 wiggle-and-dock 动作给探索出来了。这在表 2 里证明得极其明显：没有 Pre-train 直接从零跑 RL，成功率是 0%！纯靠 RL 探索这个 2mm 的缝隙是不可能的，Pre-train 提供的先验是救命稻草。

### 3. 最硬核的工程突破：让触觉在 Sim 里“算得准”

要让这套跑通，最大的拦路虎是 Sim-to-Real gap。如果你用 GelSight 这种光学触觉传感器，它输出的是高分辨率 RGB 图像，你要在 Sim 里去渲染光照、形变、材质反射，这几乎是不可能做好的，gap 太大。

这篇 paper 的工程取舍极其聪明：他们自研了 **FlexiTac** 传感器，只用压阻材料测法向力。结构就是两层 FPC 夹一层压阻材料，分辨率 2mm，成本 10 美金一个。

在 Sim 里模拟这种触觉极其简单，就是一个 Kelvin-Voigt 弹簧阻尼模型。对于传感器表面上的每一个 taxel $i$：

$$f_{n,i} = -(k_n \cdot d_i + k_d \cdot \dot{d}_i)$$

变量意思：
*   $f_{n,i}$: 第 $i$ 个触点受到的法向力。
*   $k_n = 1.0$: 弹性系数，控制力的大小。
*   $k_d = 3 \times 10^{-3}$: 阻尼系数。
*   $d_i$: 第 $i$ 个点穿透物体的深度，通过 SDF (Signed Distance Field) 查询得到。
*   $\dot{d}_i$: 沿法线方向的相对速度。

整个计算全在 GPU 上并行跑，速度极快。Calibration 的时候，拿个仪器压一下 real sensor，调一下 $k_n$ 和 $k_d$，让 sim 和 real 的受力直方图对齐，Sim-to-Real gap 就直接抹平了。这是让整个 pipeline 成立的基石。

### 4. 网络架构与 Representation

他们没有用图像，而是把 visual 和 tactile 全部统一成了 **Point Cloud**。

*   Visual: 相机拍到的深度点云，每个点 4 维 (x, y, z, 0)。第 4 维补 0 作为 flag。
*   Tactile: 传感器表面的触点，通过 forward kinematics 算出在相机坐标系下的 3D 位置，加上一个 0-1 归一化的压力读数，每个点 4 维。第 4 维放压力值，并打个 1 的 flag 区分模态。

$$\mathbf{P}_t = \left[ (\mathcal{P}_t^v; \mathbf{0}), (\mathcal{P}_t^\tau; \mathbf{1}) \right]$$

把这两坨点云拼在一起，送进 PointNet 提取 64 维特征。然后跟关节状态拼起来，送进 Diffusion U-Net 预测 16 步的 action chunk。用点云的好处就是 sim-to-real 的时候，你只要在 sim 里的点云上加一点乘性高斯噪声 $\mathbf{x}_w \gets \mathbf{x}_w (1 + \mathcal{N}(0, 0.01\sigma))$，domain gap 就基本没了，比去搞 photorealistic rendering 简单太多。

### 5. 实验数据的 Intuition 拆解

你看 Table 1 和 Table 2 的数据，能看出非常有意思的 trend：

**视觉 vs 触觉的增益**
在 Table-Top Setup 上，纯视觉 Fine-tune 后成功率从 34% 涨到 53%，提升了 20%。加了触觉的 VT Fine-tune 从 46% 涨到 85%，提升了 40%。
为什么触觉翻倍增益？因为视觉在最后插入阶段是瞎的。纯视觉 RL 在 sim 里探索的时候，插不进去它不知道该往哪微调，只能瞎撞。加了触觉，RL 突然获得了一个极其 dense 的 gradient signal——手一扭，接触力变了，policy 就知道这个方向是对的，马上顺藤摸瓜插进去。

**DPPO 的 KL Penalty 设计**
他们在 RL Fine-tune 的时候，加了一个极小的 KL penalty $\lambda_{\text{KL}} = 2 \times 10^{-4}$，限制 actor 不要偏离 pre-train 的 manifold 太远。只优化最后几步 denoise steps。这就相当于 LLM 里的 RLHF，你用 reward 去对齐人类偏好，但加个 KL 限制让它别把预训练的语法和常识给忘了。

### 6. 我的一点 Critique 和 联想

这篇 paper 证明了 Robotics 的 "RLHF" 路线在接触密集型任务上绝对走得通。但它的局限也在这儿：它需要 CAD 模型在 sim 里算 SDF，还需要手工对齐 sim 和 real 的物理参数。

**如果往远了联想**，下一步绝对是去掉这个 CAD 依赖。未来可以用 NeRF 或 Gaussian Splatting 直接重建出环境的 SDF，结合 differentiable physics engine，比如 DiffTactile，在 sim 里直接做 reward shaping，而不只是 0/1 的 sparse reward。

另外，触觉数据其实非常像 LLM 里的 token。如果能把全网的触觉数据收集起来，训一个 Tactile Foundation Model，像 CLIP 一样对齐视觉和触觉，那以后 zero-shot 的接触操作就真的可玩了。你当时在 Tesla 搞的那套纯视觉的电机控制其实也是同一个信念：先验越强，下游 fine-tune 越容易。这里只是把视觉换成了触觉。

### 参考链接：
*   **VT-Refine Project Page**: https://binghao-huang.github.io/vt_refine/
*   **Diffusion Policy (Chi et al.)**: https://diffusion-policy.cs.columbia.edu/
*   **DPPO (Diffusion Policy Policy Optimization)**: https://diffusionpolicyppo.github.io/
*   **TacSL (GPU Tactile Simulation Library)**: https://tacsl.github.io/
*   **3D-ViTac (前序工作)**: https://binghao-huang.github.io/3d-vitac/

---

# VT-Refine: Bimanual Assembly with Visuo-Tactile Feedback via Simulation Fine-Tuning

这篇paper来自Columbia University和NVIDIA的合作，CoRL 2025。核心思想非常elegant：把real-world demonstration的behavior cloning policy，通过simulation中的large-scale RL fine-tuning来refine，从而获得precise contact-rich bimanual assembly能力。让我深入讲解。

---

## 1. 核心动机与问题定义

### 1.1 为什么bimanual assembly这么难

考虑一个plug-and-socket assembly task，clearance只有约2mm。这个task的difficulty来自三个层面：

**Perception gap**：当两个parts接近insertion时，visual occlusion非常严重——fingers挡住了contact region，camera根本看不见plug和socket的精确relative pose。人类解决这个问题靠的是tactile feedback，但robot没有。

**Exploration gap**：human在插入受阻时会做"wiggle-and-dock"——小幅度反复调整、感知contact force变化、再调整。这些micro-adjustments在teleoperation demonstration中很难capture，因为teleop interface本身缺乏tactile feedback，operator只能靠vision遥控，无法传递human的tactile exploration策略。

**Data scaling gap**：behavior cloning with diffusion policy [1] 在low-precision task上work well，但precision越高、contact越复杂，需要的demonstration数量explosively增长。30个demonstration对于2mm clearance的assembly远远不够。

VT-Refine的核心insight：**与其collect更多demonstration（昂贵且suboptimal），不如在simulation中通过RL exploration自动discover这些micro-adjustment behaviors**。但前提是simulation要能accurately model tactile sensing，让sim-to-real transfer可行。

### 1.2 与prior work的关键差异

| 维度 | Prior Work | VT-Refine |
|------|-----------|-----------|
| Tactile modality | Optical sensors (GelSight [16]) | Piezoresistive (normal force only) |
| Sim fidelity | Hard to simulate optical | Kelvin-Voigt model, GPU-parallel |
| Bimanual sim-to-real | Mostly vision-only [44-46] | First visuo-tactile bimanual |
| Fine-tuning | IL only or RL from scratch [5] | Diffusion Policy + DPPO fine-tune |
| Representation | Image-based tactile | Point cloud (visual+tactile unified) |

---

## 2. Tactile Sensing System: FlexiTac

### 2.1 Hardware design

FlexiTac的设计哲学是**sacrifice resolution for sim-to-real transferability**。具体规格：

- **Sensing matrix**: 12 × 32 = 384 taxels per pad
- **Spatial resolution**: 2mm (center-to-center)
- **Force range**: 0.2 - 10 N
- **Frequency**: 23 Hz
- **Cost**: ~$10 per pad, ~$30 reading board
- **Structure**: Triple-layer FPC (flexible printed circuit) sandwiching piezoresistive layer

每个robot hand有2个finger，每个finger有1个pad，所以bimanual setup总共有 **4 pads × 384 = 1536 tactile points**。

**为什么选piezoresistive而不是optical？**

Optical tactile sensors (GelSight, GelSlim, DIGIT) 通过internal illumination + camera捕获deformation，能提供sub-mm resolution的RGB image，包含normal force、shear force、surface texture。但：
1. RGB image极难在simulation中accurately reproduce（illumination、material optical properties）
2. Marker-based methods [31-33] 通过marker deviation间接infer force，引入额外compounding error
3. Bulk size受camera focal length限制，难集成到compliant gripper

Piezoresistive只measure normal force，但这个modality在simulation中可以用简单的spring-damper model准确复现，sim-to-real gap极小。这是engineering trade-off的典范。

### 2.2 Tactile Simulation

基于TacSL [12] library（集成在Isaac Gym [48] 中），采用**penetration-based tactile force model** [49]：

核心公式（Kelvin-Voigt viscoelastic model）：

$$f_{n,i} = -(k_n \cdot d_i + k_d \cdot \dot{d}_i)$$

变量含义：
- $f_{n,i}$: taxel $i$ 上的normal contact force（标量，沿contact normal方向）
- $k_n = 1.0$: elastic stiffness（弹性系数），calibration时tune
- $k_d = 3 \times 10^{-3}$: damping coefficient（阻尼系数），calibration时tune
- $d_i$: taxel $i$ 与object的interpenetration depth（穿透深度，米）
- $\dot{d}_i$: relative velocity along contact normal（沿法线的相对速度）

**关键实现细节**：

每个taxel的世界坐标通过forward kinematics计算：

$$\mathbf{x}_i^w = \mathbf{R}_e \cdot \mathbf{x}_i^l + \mathbf{p}_e$$

$$\dot{\mathbf{x}}_i^w = \boldsymbol{\omega}_e \times (\mathbf{R}_e \cdot \mathbf{x}_i^l) + \mathbf{v}_e$$

其中：
- $\mathbf{x}_i^w, \mathbf{x}_i^l$: taxel $i$ 的world / local coordinates
- $(\mathbf{R}_e, \mathbf{p}_e, \boldsymbol{\omega}_e, \mathbf{v}_e)$: sensor pad link的pose和twist（从PhysX获取）
- $\times$: cross product

SDF (Signed Distance Field) query返回每个taxel的 $d_i$, $\hat{\mathbf{n}}_i$, $\dot{d}_i = \hat{\mathbf{n}}_i \cdot \dot{\mathbf{x}}_i^w$。整个pipeline全GPU vectorized over所有environments。

**Taxel sampling procedure**（自动，无需manual annotation）：
1. Detect flat face: 取mesh M最短的bounding box axis作为厚度方向
2. Create planar lattice: 在剩余两axis上铺 $R \times C$ grid，留1mm margin
3. Ray-cast: 从每个lattice node沿 $-\hat{\mathbf{n}}$ 方向ray-cast到mesh，得到3D position

所有taxel assigned固定orientation $q_{\text{taxel}} = \text{Euler}(0, 0, -\pi)$，确保+y axis始终outward。

### 2.3 Calibration

用DMA 850 Dynamic Mechanical Analyzer测量real sensor的force-reading curve，然后iteratively tune $k_n, k_d$ until simulation curve与real curve MSE最小。

Normalization处理noise floor：

$$s^{\text{norm}} = \begin{cases} s / s_{\text{max}}^{\text{fixed}}, & s < \tau \\ s / s_{\text{max}}^{\text{curr}}, & s \geq \tau \end{cases}$$

其中 $\tau$ 是noise threshold，$s_{\text{max}}^{\text{fixed}}$ 来自datasheet，$s_{\text{max}}^{\text{curr}}$ 是frame-wise maximum。这个rule在sim和real中identically应用，确保dynamic range一致。

Fig. 10的ablation说明：uncalibrated simulation会导致RL fine-tuning初期success rate下降（policy需要重新adapt shifted tactile distribution），而calibrated版本monotonically improves。

---

## 3. Visuo-Tactile Policy Architecture

### 3.1 Point-based representation

Observation $o$ 由三个modality组成：

1. **Visual**: $P_t^{\text{visual}} \in \mathbb{R}^{N_{\text{vis}} \times 4}$，colorless point cloud from egocentric RealSense D455
2. **Tactile**: $P_t^{\text{tactile}} \in \mathbb{R}^{N_{\text{tac}} \times 4}$，其中 $N_{\text{tac}} = 384 \times N_{\text{finger}}$，每点是 (XYZ, normalized_pressure)
3. **Proprioception**: joint positions from two arms + two grippers (16-d)

Merged representation：

$$o = P_t^{\text{tactile}} \cup P_t^{\text{visual}}$$

Tactile point的position通过forward kinematics转换到camera 3D frame，**保留visual和tactile之间的spatial relationship**——这是关键设计，让PointNet能learn到"tactile点在visual scene中的哪里"。

具体实现中，为区分modality，append one-hot flag：

$$\mathbf{P}_t = \left[ (\mathcal{P}_t^v; \mathbf{0}), (\mathcal{P}_t^\tau; \mathbf{1}) \right] \in \mathbb{R}^{(N_v + N_\tau) \times 5}$$

第5个channel：0 for visual, 1 for tactile。

### 3.2 网络架构

**Encoder**：PointNet [52] with hidden sizes {64, 128, 256, 512}，LayerNorm after each linear，final projection to 64-d，global max pooling → $\mathbf{f}_t \in \mathbb{R}^{64}$

**State encoder**：16-d joint state → 2-layer MLP {64, 64} → concatenate with $\mathbf{f}_t$

**Diffusion backbone**：U-Net {512, 1024, 2048}, kernel 5, group norm 8

### 3.3 Stage 1: Real-World Pre-training

用DDPM (epsilon-prediction variant [62])：
- Denoising steps: $T = 100$
- Action horizon: $H = 16$ (predict 16-step action chunk)
- Conditioning: first $C = 2$ steps for proprioception, first $C_{\text{img}} = 2$ steps for point clouds (two-stream scheme)

训练数据：30 episodes real-world teleoperation demonstrations。

### 3.4 Stage 2: Simulation Fine-tuning with DPPO

使用DPPO [6] (Diffusion Policy Policy Optimization)，把denoising process formalize为MDP，让reward能backpropagate through denoising chain。

**Actor**: 初始化自pre-trained weights，只optimize最后 $T_{\text{ft}}$ denoising steps
**Critic**: random initialized，asymmetric actor-critic [53]（critic接收low-dim robot+object state）
**KL penalty**: $\lambda_{\text{KL}} = 2 \times 10^{-4}$，keep decoder close to pre-trained manifold

**PPO hyperparameters**:
- $n_{\text{steps}} = 30$ per environment per update
- $\gamma = 0.999$ (discount), $\lambda = 0.95$ (GAE)
- 10 PPO epochs per batch, target KL = 1.0
- Learning rates: $10^{-5}$ (actor), $10^{-3}$ (critic), cosine decay
- $\sigma_{\text{rand}} = 3.0$, $\hat{\sigma}_{\text{min}} = 0.01$, $\tilde{\sigma}_{\text{min}} = 0.10$

**Reward**: sparse binary reward，assemble成功得1，否则0 [54]。这能work是因为pre-trained policy提供了strong prior，guide RL exploration，避免复杂reward engineering。

---

## 4. Experimental Results Analysis

### 4.1 Real-world performance (Table 1)

**Table-Top Bimanual Setup** (WidowX arms + ALOHA 2 teleop):

| Policy | Asset 00081 | 00186 | 00007 | 00446 | 00581 | Avg |
|--------|-------------|-------|-------|-------|-------|-----|
| Visual Pre-Train | 0.35 | 0.40 | 0.40 | 0.20 | 0.35 | 0.34 |
| Visual Fine-Tuned | 0.50 | 0.65 | 0.75 | 0.30 | 0.45 | 0.53 |
| VT Pre-Train | 0.55 | 0.65 | 0.40 | 0.35 | 0.35 | 0.46 |
| VT Fine-Tuned | 0.85 | 0.90 | 0.95 | 0.80 | 0.75 | 0.85 |

关键观察：
- **Visual policy fine-tuning提升 ~20%** (0.34 → 0.53)
- **VT policy fine-tuning提升 ~40%** (0.46 → 0.85)
- **VT比visual-only最终高32%** (0.85 vs 0.53)

**Semi-Humanoid Setup** (Kinova Gen3 + Robotiq 2F-140 + Meta Quest 2 teleop):

| Policy | 00081 | 00186 | 00007 | Avg |
|--------|-------|-------|-------|-----|
| Visual Pre-Train | 0.15 | 0.35 | 0.35 | 0.28 |
| Visual Fine-Tuned | 0.25 | 0.35 | 0.45 | 0.35 |
| VT Pre-Train | 0.30 | 0.30 | 0.60 | 0.40 |
| VT Fine-Tuned | 0.60 | 0.65 | 0.65 | 0.63 |

Semi-humanoid setup整体性能低于table-top，可能因为7-DoF arm的control更难、teleop interface（VR controller vs master arm）precision更低。

### 4.2 Simulation results (Table 2)

| Variant | 00081 | 00186 | 00007 | 00446 | 00581 |
|---------|-------|-------|-------|-------|-------|
| Visual Pre-Train | 0.28 | 0.32 | 0.42 | 0.12 | 0.18 |
| Visual Fine-Tune w/o Pretrain | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Visual Fine-Tune w/ Pretrain | 0.57 | 0.72 | 0.84 | 0.36 | 0.52 |
| VT Pre-Train | 0.45 | 0.48 | 0.54 | 0.34 | 0.31 |
| VT Fine-Tune w/o Pretrain | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| VT Fine-Tune w/ Pretrain | 0.82 | 0.94 | 0.98 | 0.76 | 0.78 |

**Critical insight**: Fine-Tune w/o Pretrain全部0%！说明从scratch用sparse reward RL在这个task上根本explore不到成功state——pre-training提供的prior是必须的。这与DPPO [6] paper的结论一致。

### 4.3 Data scaling (Table 3)

| Demonstrations | Visual Pretrain | Visual Fine-Tune | VT Pretrain | VT Fine-Tune |
|----------------|-----------------|------------------|-------------|--------------|
| 10 | 0.08 | 0.21 | 0.02 | 0.34 |
| 30 | 0.40 | 0.65 | 0.48 | 0.94 |
| 50 | 0.37 | 0.67 | 0.57 | 0.92 |

30 vs 50几乎没有差异——grasp phase已经被30个demo解决，bottleneck是insertion phase的micro-adjustments，这些无法通过增加demo capture（teleop缺乏tactile feedback）。这恰好证明了RL fine-tuning的value：RL能在simulation中explore出demo中缺失的exploratory behaviors。

### 4.4 Sim-to-real transfer loss (Fig. 7)

- Real → Sim: ~5-10% drop
- Sim → Real: smaller, sometimes negligible
- 但RL fine-tuning提升 >30%，所以transfer loss acceptable

---

## 5. 关键技术insight深入解析

### 5.1 为什么VT policy能emerge "wiggle-and-dock"

Fig. 8(a)展示了一个representative success trajectory：imprecise pre-insertion pose后，two arms做rapid cycles of sensing → micro-adjusting → re-sensing。这些back-and-forth maneuvers在demo中从未出现，却在RL fine-tuning中organically emerged。

**机制解释**：

1. Sparse reward只在assembly成功时给1
2. Pre-trained policy把robot带到near-insertion region（接近成功但未成功）
3. 在这个region，RL需要explore small actions来find successful insertion
4. Tactile feedback提供dense signal：contact force的change直接indicate alignment是否improve
5. Vision-only policy在occlusion下blind，只能random探索，效率极低
6. VT policy能基于tactile gradient做informed exploration，快速发现wiggle策略

这就是为什么VT fine-tuning提升40%而visual-only只提升20%——tactile不是简单加了redundant information，而是fundamentally改变了exploration efficiency。

### 5.2 Point cloud representation的sim-to-real优势

为什么point cloud比image-based representation更适合sim-to-real？

**Image domain gap**: rendering photorealistic images很难，lighting、material、texture都要match
**Point cloud domain gap**: 只需geometry + depth noise，容易calibrate

论文在sim中inject multiplicative noise: $\mathbf{x}_w \gets \mathbf{x}_w (1 + \mathcal{N}(0, 0.01\sigma))$，$\sigma = 3$。简单有效。

**Uniform sampling vs FPS**: 论文发现uniform sampling比FPS快10×，且对task performance无measurable impact。这对large-scale RL很重要——FPS的O(N²) cost在每步rollout都执行会bottleneck。

### 5.3 DPPO的design choice

为什么用DPPO而不是直接用PPO fine-tune一个MLP policy？

1. **Pre-trained diffusion policy的manifold preservation**: KL penalty $\lambda_{\text{KL}} = 2 \times 10^{-4}$ 让decoder不偏离pre-trained manifold太远，避免catastrophic forgetting
2. **Action chunking的multi-step coherence**: diffusion自然model action chunk的correlation，比independent action prediction更适合contact-rich task
3. **Denoising as MDP**: DPPO的core insight是把denoising chain视为MDP，每个denoising step是一个action，让reward能credit assign到具体denoising step

只optimize最后 $T_{\text{ft}}$ steps的design避免early denoising steps被破坏（那些steps主要recover coarse structure）。

---

## 6. Limitations & Future Directions

### 6.1 Tactile modality trade-offs

**Resolution**: FlexiTac 2mm vs GelSight <1mm。Fine-grained texture无法capture，但assembly task不需要。

**Shear force**: FlexiTac只measure normal force。Shear对slip detection重要，但paper认为temporal history能implicit infer。这个claim有点weak——in-hand manipulation中slip detection是critical的。

**System integration**: FlexiTac薄、轻、易customize。GelSight bulky，难装compliant gripper。

### 6.2 Methodological limitations

**Manual calibration**: visual domain, tactile distribution, low-level control都要align。这与domain randomization的philosophy相反——paper选择precise calibration而非massive randomization。

**No deformable objects**: Isaac Gym + tactile sim不支持deformable。这限制task scope。

**CAD model required**: sparse reward需要CAD来判断assembly success。CAD-free pipeline是future work方向。

---

## 7. 我的critique与extension thoughts

### 7.1 这篇paper真正贡献了什么

**Conceptual contribution**: 第一个证明bimanual visuo-tactile policy能通过sim-to-real RL fine-tuning大幅提升。这open了新paradigm——不再是"collect more data"而是"explore in sim"。

**Engineering contribution**: FlexiTac + TacSL integration让tactile simulation practical。Cost $10/pad，5分钟fabrication，这是democratization。

**Negative result的value**: Fine-Tune w/o Pretrain = 0% 是非常重要的finding。说明contact-rich assembly的RL exploration问题未被解决，pre-training是必要的prior injection。

### 7.2 我会怎么extend

**Online adaptation in real**: 当前pipeline是offline fine-tune in sim then deploy。如果能在real中持续adapt（with safety constraints），能handle更多distribution shift。

**Tactile-only phase**: 在insertion phase，vision基本occluded。能否自动detect "vision useless" phase并switch to tactile-only policy？这能reduce visual noise干扰。

**Shear force extension**: FlexiTac缺乏shear是真limitation。结合AnySkin [24]的magnetic-based shear sensing可能complement。

**Deformable objects**: 结合DiffTactile [29]的differentiable FEM simulator能enable deformable tactile sim。

**Reward shaping from tactile**: 当前sparse reward。能否用tactile signal的change作为shaping reward？比如contact force increase → potential increase。这可能accelerate RL convergence。

**Foundation model for tactile**: 当前每个task从头train。能否pre-train一个tactile encoder在diverse contact data上，然后transfer？

### 7.3 与LLM/robotics foundation model的connection

VT-Refine的"pre-train then fine-tune" paradigm与LLM training高度analogous：
- Pre-train on demonstrations ≈ pre-train on internet text
- RL fine-tune with sparse reward ≈ RLHF
- KL penalty to pre-trained manifold ≈ KL penalty to reference model in RLHF

这suggests robotics可能走相同的scaling path：large-scale pre-training + RL refinement。但robotics的bottleneck是data collection cost，而simulation是唯一scalable path。VT-Refine指向了sim-based RL fine-tuning作为robotics的"RLHF"。

---

## 8. References & Further Reading

- [Project page](https://binghao-huang.github.io/vt_refine/)
- [Diffusion Policy (Chi et al., RSS 2023)](https://diffusion-policy.cs.columbia.edu/)
- [3D Diffusion Policy (Ze et al., RSS 2024)](https://3d-diffusion-policy.github.io/)
- [3D-ViTac (Huang et al., CoRL 2024)](https://binghao-huang.github.io/3d-vitac/)
- [DPPO (Ren et al., ICLR 2025)](https://diffusionpolicyppo.github.io/)
- [TacSL (Akinola et al., T-RO 2025)](https://_tacsl_.github.io/)
- [AutoMate (Tang et al., RSS 2024)](https://automate.github.io/)
- [ALOHA 2 (Aldaco et al.)](https://aloha-2.github.io/)
- [cuRobo (Sundaralingam et al.)](https://github.com/NVlabs/curobo)
- [Isaac Gym (Makoviychuk et al., NeurIPS 2021)](https://developer.nvidia.com/isaac-gym)
- [PointNet (Qi et al., CVPR 2017)](https://stanford.edu/~rqi/pointnet/)
- [PPO (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [FurnitureBench (Heo et al., RSS 2023)](https://furniture-bench.github.io/)
- [DiffTactile (Si et al., 2024)](https://arxiv.org/abs/2403.08716)
- [AnySkin (Bhirangi et al., ICRA 2025)](https://any-skin.github.io/)

---

## 9. 总结直觉

VT-Refine的核心intuition：**tactile sensing是contact-rich manipulation的"gradient signal"**，而simulation RL是自动discover如何利用这个gradient的mechanism。人类demo因为interface限制丢失了tactile exploration策略，但sim RL能recover这些策略——前提是sim的tactile要足够accurate让transfer可行。

这篇paper的beauty在于engineering trade-off的clarity：放弃optical sensor的high resolution换sim-to-real feasibility，放弃shear force换sim simplicity，放弃dense reward engineering换pre-train prior的exploration guide。每个trade-off都justified by experimental evidence。

如果robotics要走LLM的scaling path，sim-based RL fine-tuning of pre-trained policies很可能是一个core component，而VT-Refine给出了contact-rich bimanual case的第一个complete recipe。
