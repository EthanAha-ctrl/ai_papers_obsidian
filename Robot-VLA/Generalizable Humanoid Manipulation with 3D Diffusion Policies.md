---
source_pdf: Generalizable Humanoid Manipulation with 3D Diffusion Policies.pdf
paper_sha256: 8ea49b295a7d7b41271a45a1d6656d2d82beb4484820295748649ab7a40430c4
processed_at: '2026-08-04T13:32:12-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲这篇 Paper

## 一句话总结

他们让一个 full-sized humanoid robot 在一个 lab scene 里学抓杯子，然后直接拎到 random kitchen / meeting room / office 里用，竟然 work。秘诀是：用 3D point cloud 而不是 2D image 做 policy 的 input，并且把 point cloud 放在 camera frame 里（robot 自己的视角），不用 world frame。

---

## 为什么要做这件事

之前的 humanoid 工作（HumanPlus、OmniH2O、OpenTeleVision、HATO）都卡在一个尴尬的地方：teleoperation demo 很炫，但 learned policy 只能在 training scene 里跑。换个 kitchen，policy 就傻了。

原因很简单——这些工作大多用 2D image 作为 visual input。2D image 对 scene 的 appearance 极度敏感：lighting 变了、background texture 变了、camera angle 变了，pixel distribution 全变，policy 直接崩。

换个角度想：你教一个小孩抓杯子，你不会说"杯子在 world coordinate (1.2, 0.5, 0.8) 的位置"。你会说"你面前那个杯子"。小孩的 reference frame 是自己的眼睛，不是 world origin。iDP3 就是把这个 intuition 落到了 humanoid robot 上。

参考: 人对 egocentric representation 的依赖 https://en.wikipedia.org/wiki/Egocentric_bias

---

## 系统怎么搭起来的

### Robot

Fourier GR1，full-sized humanoid，1.3m 高度级别。enable 了 upper body 25 DoF（head + waist + 2 arms + 2 hands），lower body 用 cart 代替。

为什么不用腿？作者很诚实——current humanoid 的 balance control 还不够 mature，waist 一 lean forward 就可能摔。用 height-adjustable cart 是个 practical workaround。cart 可以调高度，这样不同 kitchen 台面高度不一致时，robot 可以调整到自己 comfortable 的工作高度。

这个 tradeoff 很重要。他们不是做不出来 locomotion，而是选择把精力集中在 manipulation 的 generalization 上。locomotion 的问题让 legged locomotion community 去解，他们先解决 manipulation。

### Sensor

Robot head 上装一个 Intel RealSense L515 solid-state LiDAR。为什么不用普通的 RGB-D camera 比如 D435？因为 D435 的 depth 精度太差，point cloud noise 大，DP3 在这种 noisy point cloud 上表现 suboptimal（DP3 原始 paper 和 RISE 工作都报道过这个 issue https://arxiv.org/abs/2404.12281）。

L515 也不是完美的。作者在 limitation 里说 point cloud 仍然 noisy。他们试过 Livox Mid-360，resolution 和 frequency 不够，做不了 contact-rich manipulation。

这里有个 engineering insight：**depth quality是 3D policy 的 bottleneck**。如果有一个轻量、高精度、高频率的 depth sensor，iDP3 的性能会更好。这也是为什么 Azure Kinect DK 或者未来的 solid-state LiDAR 对 robot learning 很关键。

### Teleoperation

用 Apple Vision Pro（AVP）capture human motion。AVP 给你：
- Head 的 6D pose（position + orientation）
- 两个 wrist 的 6D pose
- Hand joint angles

然后把这些 human data map 到 robot joints：
- **Arm**: 用 Relaxed IK（一个 real-time IK solver https://github.com/uwgraphics/RelaxedIK）从 wrist position 反解 arm joint angles
- **Waist + Head**: 从 human head rotation 直接算。人转头时躯干会跟着动，所以 head rotation 是 waist + head 的合理 proxy
- **Hands**: 直接映射到 Inspire Hands 的 joint positions

有个细节值得注意：teleoperation latency 大约 0.5 秒，因为 LiDAR 把 onboard computer 的 bandwidth/CPU 占满了。他们试过两个 LiDAR（head + wrist），latency 高到没法 collect data。

这意味着**data collection 的 bottleneck 不只是 human 疲劳，还有 hardware bandwidth**。未来如果 sensor 有 dedicated processing unit，或者用 edge AI chip 做 depth estimation，可以大幅降低 latency。

---

## iDP3 到底改了什么

原始 DP3（3D Diffusion Policy，https://arxiv.org/abs/2403.03954）在 tabletop robot arm 上 work 得很好，但直接搬到 humanoid 上完全 fail（Table II 里 0/0）。作者分析了 DP3 fail 的原因，做了四个 key modifications。

### 改动 1：Egocentric 3D Representation（最关键）

原始 DP3 把 point cloud 转到 world frame：

$$\mathbf{P}_{world} = \mathbf{T}_{cam \to world} \cdot \mathbf{P}_{cam}$$

- $\mathbf{P}_{cam}$: camera frame 下的 point cloud，$[X_c, Y_c, Z_c]^T$
- $\mathbf{T}_{cam \to world} \in SE(3)$: camera 到 world 的 transformation matrix，需要通过 hand-eye calibration 获得
- $\mathbf{P}_{world}$: world frame 下的 point cloud

在 world frame 下的好处是：object 的 position 是固定的，容易 segment 出 target object。但问题在于：

1. **Calibration 麻烦**: humanoid 的 camera mount 在 head 上，head 会动。每次 head 动，calibration 就需要更新。而且 real-world calibration 永远有 error。
2. **Segmentation 需要**: world frame 下你看到整个 scene（桌子、墙、地面），需要把 target object segment 出来。通常用 foundation model 或 manual segmentation，但在 humanoid 部署时这不 scalable。

iDP3 的 solution：**直接用 camera frame 的 point cloud**，不做 transformation。

$$\mathbf{P}_{cam} = D(u,v) \cdot \mathbf{K}^{-1} \cdot \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$

- $D(u,v)$: pixel $(u,v)$ 处的 depth value（从 LiDAR 获得）
- $\mathbf{K}$: camera intrinsic matrix，$f_x, f_y$ 是 focal length，$(c_x, c_y)$ 是 principal point
- $(u, v)$: pixel coordinates
- 输出 $\mathbf{P}_{cam} = [X_c, Y_c, Z_c]^T$ 是 camera frame 下的 3D point

**Intuition**: 在 camera frame 下，"robot 看到什么"就是 "policy 输入什么"。当 robot head 转向 cup 时，cup 在 camera frame 里的 position 变化 encode 了 robot 的 intention。policy 学到的是"在我的视角里，cup 在这个 relative position 时，我应该怎么动 arm"。这个 mapping 对 scene 变化是 robust 的。

这跟人很像。你抓杯子时，你的 reference frame 是你的眼睛，不是房间的 world origin。你换到另一个 kitchen 抓杯子，只要你的眼睛看到杯子的 relative position 类似，你的 motor action 就类似。

这也解释了 view invariance（Figure 8）—— 即使 camera viewpoint 大幅度变化，只要 cup 在 camera frame 里的 geometry 类似，iDP3 就能 grasp。而 2D image-based method 直接崩，因为 viewpoint 变化改变了 pixel pattern。

### 改动 2：Scale Up Vision Input

原始 DP3 用 1024 个 sparse points，依赖 segmentation 去掉 background。iDP3 没有 segmentation，直接 scale up 到 4096 points，capture 整个 scene。

Ablation 数据（Table III）：
- 1024 points: 56/129
- 2048 points: 65/128
- 4096 points: 75/139 ← best
- 8192 points: 72/132 ← saturated

**Intuition**: 当你没法 segment 时，就把整个 scene 都给 policy 看，让 policy 自己 learn 哪些 geometry 是 task-relevant 的。这跟 ViT 的思路一样——不要 hand-craft inductive bias，给更多 data 让 model 自己学。

4096 是 sweet spot。8192 反而更差，可能因为：
1. 更多点 = 更多 noise（L515 的 depth error 在每个点上都有）
2. Computational cost 增加，training 可能 unstable
3. 过多的 background points dilute 了 task-relevant signal

### 改动 3：Pyramid Convolutional Encoder

原始 DP3 用 MLP（linear layers）encode point cloud。iDP3 用 pyramid convolutional encoder——多层 conv，每层 receptive field 不同，最后 pyramid fusion。

架构概念：
```
Point Cloud (N=4096, 3)
    ↓ Voxel Sampling
Sampled Points (M, 3)
    ↓ Conv Layer 1 (small receptive field)
Feature 1 (M, C1)  ← capture 局部 geometry（比如 cup 边缘）
    ↓ Conv Layer 2 (medium receptive field)
Feature 2 (M, C2)  ← capture 中等 scale spatial relation
    ↓ Conv Layer 3 (large receptive field)
Feature 3 (M, C3)  ← capture 全局 spatial relation（cup relative to hand）
    ↓ Pyramid Fusion (concat Feature 1, 2, 3)
Final Feature (M, C1+C2+C3)
    ↓ Pooling
Global Feature (C1+C2+C3)
```

Ablation（Table III）：
- Linear (DP3): 58/127
- Conv only: 49/131 ← 比 Linear 还差！
- Linear + Pyramid: 66/134
- Conv + Pyramid (iDP3): 75/139 ← best

Conv alone 比 Linear 差是 counterintuitive 的。可能的解释：Conv 需要 more data 来 learn kernel weights，而 small dataset 下 MLP 更 sample-efficient。但加上 pyramid fusion 后，multi-scale information 弥补了这个 deficit。

**Intuition for "smoother behaviors"**: Conv 的 weight sharing 让 spatially nearby points 产生 similar features，这 induce 了 spatial smoothness。MLP 对每个 point 独立处理，容易 overfit 到 individual point 的 noise。Human demonstration 本身有 tremor/jitter，conv 的 spatial smoothing 能 average out 这些 noise。

### 改动 4：Longer Prediction Horizon

Diffusion Policy 预测一个 action chunk $\mathbf{a}_{t:t+H_p}$，不是单步 $a_t$。$H_p$ 是 prediction horizon。

原始 DP3 用 $H_p = 4$，iDP3 用 $H_p = 16$。

Ablation（Table III）：
- $H_p = 4$: 0/0 ← 完全 fail!
- $H_p = 8$: 33/88
- $H_p = 16$: 75/139 ← best
- $H_p = 32$: 55/130 ← over-smoothing

$H_p = 4$ 完全 fail 是个 striking result。为什么？

**Intuition**: Human demonstration 有 high-frequency jitter（人手天然 tremor，AVP tracking 也有 noise），sensor noise（L515 的 depth error）也会 inject noise。Short horizon 意味着 policy 必须精确 predict 每一步的 action，但 noisy data 让这种 precise prediction 不可能。Longer horizon 让 diffusion model 学一个"average trajectory"——即使每一步都有 noise，overall trajectory 的方向是对的。

这跟 signal processing 里的 low-pass filter 类似。Longer horizon = 更强的 temporal smoothing，filter 掉 high-frequency noise，保留 low-frequency task signal。

但 $H_p = 32$ 又变差，因为太长的 horizon 会 over-smooth 掉 task-relevant 的高频 motion（比如 grasp 瞬间手指的快速 closure）。

### 改动 5：Point Cloud Sampling

原始 DP3 用 Farthest Point Sampling (FPS)，复杂度 $O(N^2)$，对 4096 点很慢。

iDP3 用 cascade：
1. **Voxel Sampling**: 把 3D space 划分成 voxel grid（比如 5cm × 5cm × 5cm），每个 voxel 内取一个 representative point。复杂度 $O(N)$。
2. **Uniform Sampling**: 补充随机点。

**Intuition**: Voxel sampling 保证 spatial coverage——不会所有点都聚集在 dense 区域。Uniform sampling 补充 randomness，避免 voxel grid 的 aliasing artifact。这个 combination 在实践中比 FPS 快很多，效果相当。

---

## 实验结果怎么看

### Main Comparison (Table II)

| Method | Success/Attempts |
|--------|-----------------|
| DP (ResNet18) | 24/106 |
| DP3 (original) | 0/0 |
| DP (frozen R3M) | 62/138 |
| DP (finetuned R3M) | 99/147 ← training scene 最强 |
| iDP3 (DP3 encoder) | 58/127 |
| iDP3 | 75/139 |

几个关键 insight：

1. **Original DP3 完全 fail (0/0)**: 验证了 motivation——DP3 的 world frame + segmentation 依赖在 humanoid 上不可行。

2. **DP + finetuned R3M 在 training scene 最强 (99/147)**: R3M 是在 Ego4D 等大规模 human video 上 pre-trained 的 visual representation（https://arxiv.org/abs/2203.12601），finetuning 后非常 effective。这印证了 "pre-training > from-scratch" 的 common wisdom（参考 https://arxiv.org/abs/2212.05749）。

3. **但 DP 在 new scene 完全崩**（Table IV）: training scene 的高 accuracy 不等于 real-world deployability。这就是 iDP3 的价值——training scene 稍弱，但 generalization 完胜。

### Generalization (Table IV)

| Setting | DP | iDP3 |
|---------|-----|------|
| Training scene | 9/10 | 9/10 |
| New object | 3/10 | 9/10 |
| New view | 2/10 | 9/10 |
| New scene | 2/10 | 9/10 |

iDP3 在所有 generalization settings 下都保持 9/10，DP 从 9/10 暴跌到 2/10。

**这就是 3D representation 的核心价值**：geometry generalizes, appearance doesn't。一个 cup 的 3D shape 在任何 kitchen 里都类似，但 cup 的 2D appearance（color、lighting、background）每个 scene 都不同。

### Attempts Count 的意义

75/139 意味着 139 次 attempts 里 75 次 success。Attempts 数量反映 smoothness——jittery policy 会在 grasp 附近徘徊，少 attempt。iDP3 的 139 次 attempts 说明 policy confidence 高，果断执行。

这个 metric 很 clever。传统 success rate 只反映 accuracy，但 robot deployment 还需要 smoothness。一个 jittery robot 即使偶尔 success，用户体验也很差。

---

## 为什么 Egocentric 3D 能 Generalize：更深一层

让我用一个 thought experiment 来 build intuition。

假设你在 lab 里训练 policy grasp cup。Cup 在 table 上，robot 站在 table 前面，head 朝下看 cup。

**World frame representation**:
- Cup position: $(x_w, y_w, z_w) = (0.5, 0.3, 0.8)$ in world frame
- Policy 学到: "当 cup 在 $(0.5, 0.3, 0.8)$ 时，arm joint 应该是..."

换到 new kitchen:
- Cup position: $(x_w, y_w, z_w) = (1.8, -0.4, 0.75)$ in new world frame
- Policy 从没见过这个 position → **fail**

**Camera frame representation**:
- Cup position in camera frame: $(x_c, y_c, z_c) = (0.1, -0.2, 0.6)$
- Policy 学到: "当 cup 在 camera frame 的 $(0.1, -0.2, 0.6)$ 时，arm joint 应该是..."

换到 new kitchen:
- Robot head 转向 cup（teleoperation 时 human natural 会这样做）
- Cup position in camera frame: $(x_c, y_c, z_c) = (0.12, -0.18, 0.58)$ ← 类似！
- Policy generalize → **success**

**核心 insight**: Camera frame 把 "robot's intention" encode 进了 representation。Robot head 朝哪看，就定义了 policy 的 input space。Humanoid robot 的 embodiment 决定了 camera frame 是 natural reference frame。

这也跟 neuroscience 有联系——human brain 的 spatial representation 也是 egocentric 的（参考 https://www.nature.com/articles/nrn2333）。Parietal cortex 里的 neurons encode object position relative to body/head，不是 relative to world origin。

---

## Limitations 作者自己承认的

1. **AVP teleoperation 累**: 人戴 AVP 几小时就累，data scaling 受限。Fine-grained task（转螺丝）AVP precision 不够。Aloha-style physical interface 可能更适合 dexterous task。

2. **Depth sensor noise**: L515 的 point cloud 不够 accurate。如果用更好的 sensor，iDP3 性能会更好。

3. **No lower body**: 用 cart 绕开 balance，不是真正的 loco-manipulation。

4. **Single scene data**: 虽然 generalization 到了 multiple scenes，但 training data 只来自一个 scene。如果能在 multiple scenes collect data，generalization 会更强。

---

## 这篇 Paper 的 Big Picture

这篇 paper 的 contribution 不仅仅是 iDP3 算法，而是**整个 system demonstration**——证明 3D visuomotor policy 可以在 full-sized humanoid 上实现 cross-scene generalization，只用 single-scene data，onboard computing。

几个 takeaway：

1. **3D > 2D for generalization**: geometry generalizes, appearance doesn't。这是 robot learning 的 fundamental insight。
2. **Egocentric > World frame for general-purpose robots**: 当 robot 是 mobile 的，world frame 是 ill-defined 的。Camera frame 是 natural reference。
3. **Scale up input > Hand-craft segmentation**: 当 segmentation 不可行时，给更多 raw data 让 policy 自己学。
4. **Longer horizon compensates for noisy data**: human demonstration 本质上 noisy，longer prediction horizon 是 implicit denoising。

参考链接汇总：
- Paper 主页: https://humanoid-manipulation.github.io/
- DP3 原始 paper: https://arxiv.org/abs/2403.03954
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- R3M: https://arxiv.org/abs/2203.12601
- DDIM: https://arxiv.org/abs/2010.02502
- EquiBot (Sim(3) equivariance): https://arxiv.org/abs/2407.01479
- Relaxed IK: https://github.com/uwgraphics/RelaxedIK
- Fourier GR1: https://www.fourierintelligence.com/gr1
- RealSense L515: https://www.intelrealsense.com/lidar-camera-l515/
- RISE (3D perception for imitation): https://arxiv.org/abs/2404.12281
- Pre-training for visuo-motor control: https://arxiv.org/abs/2212.05749
- Egocentric spatial representation in neuroscience: https://www.nature.com/articles/nrn2333

如果你想深入某个具体方面——比如 diffusion policy 的 mathematical derivation、point cloud encoder 的 architecture details、或者 teleoperation 的 joint mapping math——我可以进一步展开。

---

# Generalizable Humanoid Manipulation with 3D Diffusion Policies 深度解析

## 1. Paper的Big Picture：从Single-Scene到Cross-Scene Generalization

这篇paper的核心问题非常清晰——humanoid robot的manipulation skill一直被困在training scene里。先前的工作（OmniH2O、HumanPlus、OpenTeleVision、HATO）都展示了whole-body teleoperation或locomotion，但learned policy在新场景下完全fail。作者团队（Stanford主导，SFU、UPenn、UIUC、CMU合作）的目标是**single-scene training → diverse unseen scenes deployment**，并且只用onboard computing。

让我先build一个intuition：为什么3D representation比2D image更generalizable？

2D image-based policy（比如Diffusion Policy + R3M）本质上是在pixel space里学mapping，scene的lighting、background texture、camera viewpoint都会改变pixel distribution。而3D point cloud直接capture了scene的geometry，geometry在新场景里相对invariant（一个cup还是cup形状，不管在哪个kitchen）。这就是为什么iDP3能zero-shot generalize，而DP在new scene里2/10都做不到。

参考链接：
- 3D Diffusion Policy原始paper: https://arxiv.org/abs/2403.03954
- 项目主页: https://humanoid-manipulation.github.io/

---

## 2. System Architecture 全景

系统由四个模块组成（对应Figure 2）：

### 2.1 Humanoid Robot Platform

**Robot**: Fourier GR1，full-sized humanoid，enable了whole upper body {head, waist, arms, hands}共25 DoF。disable了lower body用cart代替——这是个engineering tradeoff。作者在limitation里诚实承认：current humanoid hardware的balance控制还不够mature，所以用height-adjustable cart绕开whole-body control的复杂度。

**LiDAR Camera**: Intel RealSense L515（solid-state LiDAR），mounted在robot head上提供egocentric vision。这里有个重要细节——作者尝试过Livox Mid-360，但resolution和frequency不够支持contact-rich real-time manipulation。RealSense D435的depth精度不够，DP3在D435上表现suboptimal（这在DP3原始paper和RISE工作里都有报道 https://arxiv.org/abs/2404.12281）。

**Height-Adjustable Cart**: 解决tabletop高度差异问题。这是个practical hack——不同kitchen的台面高度差可能有20-30cm，robot waist的lean-forward范围有限，cart的height adjustment让robot能始终在ergonomic pose下操作。

### 2.2 Whole-Upper-Body Teleoperation

用Apple Vision Pro (AVP) capture human motion。AVP提供：
- Head的6D pose (position + orientation)
- 两个wrist的6D pose
- Hand joint angles (通过AVP的hand tracking)

Mapping逻辑：
1. **Arm joints**: 用Relaxed IK (https://github.com/uwgraphics/RelaxedIK) 求解inverse kinematics，track human wrist position。Relaxed IK是个real-time IK solver，能处理redundancy并avoid joint limits。
2. **Waist + Head joints**: 直接从human head的rotation计算。这是个聪明的简化——human转头时自然带动躯干，所以head rotation可以作为waist+head的proxy。
3. **Hands**: 直接映射到Inspire Hands的joint positions。

**Latency问题**: LiDAR sensor占用大量bandwidth/CPU，导致teleoperation latency约0.5秒。作者试过两个LiDAR（head + wrist），latency太高无法collect data。这是个重要的engineering constraint——未来如果用更高效的sensor或dedicated hardware accelerator，可以改善。

### 2.3 Data Format

每个trajectory存observation-action pairs：
- **Observation**: 
  - Visual: point cloud (从L515) + RGB image
  - Proprioception: 25个joint positions
- **Action**: target joint positions (不是end-effector pose)

为什么用joint positions而不是end-effector pose？作者发现real world的noise让end-effector pose计算不准（需要forward kinematics + calibration），直接用joint positions更accurate。这是个反直觉但practical的选择——通常人们觉得task space更interpretable，但在noisy real-world setting下，joint space反而更reliable。

---

## 3. iDP3：Improved 3D Diffusion Policy 核心技术

这是paper的technical core。让我逐个分析每个modification。

### 3.1 Background: Diffusion Policy 数学回顾

Diffusion Policy基于DDPM。给定action sequence $\mathbf{a} \in \mathbb{R}^{H_p \times D_a}$（$H_p$是prediction horizon，$D_a$是action dimension），diffusion process定义：

**Forward process**（加噪）：
$$q(\mathbf{a}_t | \mathbf{a}_0) = \mathcal{N}(\mathbf{a}_t; \sqrt{\bar{\alpha}_t} \mathbf{a}_0, (1-\bar{\alpha}_t) \mathbf{I})$$

其中：
- $\mathbf{a}_0$ 是clean action sequence
- $\mathbf{a}_t$ 是step $t$ 的noisy action
- $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$ 是cumulative product of noise schedule
- $t \in \{1, ..., T\}$，$T$是total diffusion steps

**Reverse process**（去噪）：
$$p_\theta(\mathbf{a}_{t-1} | \mathbf{a}_t) = \mathcal{N}(\mathbf{a}_{t-1}; \mu_\theta(\mathbf{a}_t, t, \mathbf{o}), \Sigma_\theta)$$

其中 $\mathbf{o}$ 是observation（point cloud + proprioception），$\mu_\theta$ 由neural network参数化。

**Training objective**（simplified ε-prediction）：
$$\mathcal{L} = \mathbb{E}_{t, \mathbf{a}_0, \boldsymbol{\epsilon}} \left[ \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{a}_t, t, \mathbf{o}) \|^2 \right]$$

其中 $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$ 是采样的高斯噪声。

iDP3用DDIM做inference（https://arxiv.org/abs/2010.02502），50 training steps，10 inference steps。DDIM是non-Markovian的reverse process，能用更少steps生成高质量样本。

### 3.2 Egocentric 3D Visual Representations

**这是最关键的innovation**。原始DP3在world frame下表示point cloud：

$$\mathbf{P}_{world} = \mathbf{T}_{cam \to world} \cdot \mathbf{P}_{cam}$$

其中 $\mathbf{T}_{cam \to world} \in SE(3)$ 是camera extrinsic matrix。这个formulation的问题是：
1. 需要precise camera calibration（hand-eye calibration）
2. 在world frame下，需要segment出target object（因为world frame包含整个scene）
3. 对于mobile robot/humanoid，camera mount会动，calibration会drift

iDP3直接在camera frame下做：

$$\mathbf{P}_{cam} = D(u,v) \cdot \mathbf{K}^{-1} \cdot \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$

其中：
- $D(u,v)$ 是pixel $(u,v)$ 处的depth value
- $\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$ 是camera intrinsic matrix
- $(u, v)$ 是pixel coordinates
- $\mathbf{P}_{cam} = [X, Y, Z]^T$ 是camera frame下的3D point

**Intuition**: 在camera frame下，robot的"视角"就是policy的input。当robot转头看同一个cup时，cup在camera frame里的位置变化体现了robot的intent（"我要去抓那个cup"），而world frame下cup的位置是固定的，robot的intent需要通过额外的proprioception编码。Egocentric representation天然encode了"robot relative to object"的geometry。

这也解释了为什么iDP3有view invariance（Property 1, Figure 8）——不同viewpoint下，object在camera frame里的位置不同，但policy学到的是"camera frame里object在这个relative position时该如何act"，这个mapping对view change是robust的。

### 3.3 Scaling Up Vision Input

原始DP3用sparse point sampling（1024点），依赖segmentation去除background。iDP3没有segmentation，直接scale up到4096点，capture整个scene。

**Ablation数据** (Table III):
- 1024 points: 56/129
- 2048 points: 65/128
- 4096 points: 75/139 (best)
- 8192 points: 72/132 (saturated, slightly worse)

8192点反而更差——可能因为更多点引入更多noise（L515的depth error），且computational cost增加导致training instability。4096是sweet spot。

**Intuition**: 当没有segmentation时，policy需要从raw scene里"找出"relevant geometry。更多points = 更完整的scene representation = policy能学到哪些geometry是task-relevant的。这有点像Vision Transformer的思路——让model自己学attention，而不是hand-craft segmentation。

### 3.4 Improved Visual Encoder: Pyramid Convolutional

原始DP3用MLP（linear layers）做point cloud encoding。iDP3用pyramid convolutional encoder。

**架构思路**：
```
Point Cloud (N, 3) → Voxel Sampling → Conv1 → Conv2 → Conv3 → Feature
                                      ↓        ↓        ↓
                                      └────────┴────────┘ (pyramid fusion)
```

每一层conv提取不同receptive field的特征，类似FPN (Feature Pyramid Network)。低层capture局部geometry（比如cup的边缘），高层capture全局spatial关系（cup relative to robot hand）。

**Ablation数据** (Table III):
- Linear (DP3): 58/127
- Conv only: 49/131
- Linear + Pyramid: 66/134
- Conv + Pyramid (iDP3): 75/139

Conv alone比Linear差（49 vs 58）——这有点反直觉。可能因为Conv需要更多data来learn kernel，而small dataset下MLP更sample-efficient。但加上pyramid fusion后，Conv+Pyramid最好。

**Intuition for "smoother behaviors"**: Conv的weight sharing让spatially nearby points产生similar features，这induce了spatial smoothness。MLP对每个point独立处理，容易overfit到individual point的noise。Human demonstration本身有jitter，conv的spatial smoothing能average out这些noise。

### 3.5 Longer Prediction Horizon

Diffusion Policy预测action chunk $\mathbf{a}_{t:t+H_p}$而不是单步 $a_t$。原始DP3用 $H_p = 4$，iDP3用 $H_p = 16$。

**Ablation数据** (Table III):
- $H_p = 4$: 0/0 (完全fail!)
- $H_p = 8$: 33/88
- $H_p = 16$: 75/139 (best)
- $H_p = 32$: 55/130 (worse, over-smoothing)

$H_p = 4$完全fail是个striking result。为什么？

**Intuition**: Human demonstration有high-frequency jitter（人手天然tremor），sensor noise（L515的depth error）也会inject noise到observation里。Short horizon意味着policy必须精确predict每一步的action，但noisy data让这种精确prediction不可能。Longer horizon让diffusion model学一个"average trajectory"——即使每一步都有noise，overall trajectory的方向是对的。这有点像temporal smoothing。

但 $H_p = 32$又变差，因为太长的horizon会over-smooth掉task-relevant的高频motion（比如grasping瞬间手指的快速闭合）。

### 3.6 Point Cloud Sampling: Voxel + Uniform

原始DP3用Farthest Point Sampling (FPS)。FPS的复杂度是 $O(N^2)$，对于4096点很慢。

iDP3用cascade：
1. **Voxel sampling**: 把3D space划分成voxel grid（比如5cm × 5cm × 5cm），每个voxel内取一个representative point（centroid或random）。复杂度 $O(N)$。
2. **Uniform sampling**: 如果voxel sampling后点数不够，随机补充。

**Intuition**: Voxel sampling保证spatial coverage——不会所有点都聚集在某个dense区域。Uniform sampling补充randomness，避免voxel grid的aliasing artifact。这个combination在实践中比FPS快很多，且效果相当。

---

## 4. Experiment Results 深度分析

### 4.1 Main Comparison (Table II)

| Method | Total Success/Attempts |
|--------|------------------------|
| DP (ResNet18) | 24/106 |
| DP3 (original) | 0/0 |
| DP (frozen R3M) | 62/138 |
| DP (finetuned R3M) | 99/147 |
| iDP3 (DP3 encoder) | 58/127 |
| **iDP3** | **75/139** |

几个关键观察：

1. **Original DP3完全fail (0/0)**：这验证了作者的motivation——DP3的world frame + segmentation依赖在humanoid上不可行。

2. **DP (finetuned R3M)比iDP3在training scene更强 (99 vs 75)**：这是个honest finding。R3M是pre-trained visual representation（在Ego4D等大规模human video上训练），finetuning后非常effective。作者hypothesize这是pre-training的优势，而3D visual model还没有类似的pre-trained counterpart。

3. **但DP在new scene完全fail** (Table IV, Figure 6)：这是iDP3的价值所在。Training scene的high accuracy不等于real-world deployability。

**Attempts count的意义**: 75/139意味着139次attempts里75次成功。Attempts数量反映smoothness——jittery policy会在grasp附近徘徊，少attempt。iDP3的139次attempts说明policy confidence高，果断执行。

### 4.2 Generalization (Table IV)

| Setting | DP | iDP3 |
|---------|-----|------|
| Training scene | 9/10 | 9/10 |
| New object | 3/10 | 9/10 |
| New view | 2/10 | 9/10 |
| New scene | 2/10 | 9/10 |

iDP3在所有generalization settings下都保持9/10，而DP从9/10暴跌到2/10。这就是3D representation的power——geometry generalizes, appearance doesn't。

### 4.3 Training Efficiency (Figure 7)

iDP3比DP训练更快，即使point cloud数量增加。这是因为：
- 3D representation维度低（4096 × 3 = 12288 dims）vs 2D image (224 × 224 × 3 = 150528 dims)
- Point cloud encoder比CNN encoder轻量
- Diffusion process在低维action space更高效

---

## 5. Why Egocentric 3D Works: Deeper Intuition

让我再深入一下egocentric 3D representation为什么能generalize。

考虑一个grasping task。在world frame下：
- Cup在 $(x_w, y_w, z_w)$
- Robot base在 $(0, 0, 0)$
- Policy需要learn: "当cup在 $(x_w, y_w, z_w)$ 时，arm joint应该..."

换到new scene，cup的world position变了，但policy没见过这个position，fail。

在camera frame下：
- Cup在 $(x_c, y_c, z_c)$ relative to camera
- Robot head看向cup
- Policy学到: "当cup在camera frame的 $(x_c, y_c, z_c)$ 时，arm joint应该..."

换到new scene，只要robot head看向cup（teleoperation时human会natural这样做），cup在camera frame里的relative position是similar的，policy generalize。

**这本质上是一种implicit coordinate frame invariance**。World frame是extrinsic的，camera frame是intrinsic的。Humanoid robot的"embodiment"决定了camera frame是natural的reference frame。

参考Sim(3)-equivariant的工作（EquiBot, https://arxiv.org/abs/2407.01479）显式设计equivariance，而iDP3通过egocentric representation隐式获得了类似的property。

---

## 6. Limitations & Future Directions

作者诚实地列了几个limitations：

1. **AVP teleoperation疲劳**: 人类operator戴AVP几小时就累，data scaling受限。Aloha-style的physical interface可能更适合fine-grained tasks。

2. **Depth sensor noise**: L515的point cloud不够accurate。如果用更高质量的sensor（比如Azure Kinect或工业级LiDAR），iDP3性能会更好。

3. **No lower body**: 用cart绕开了balance问题，但不是真正的humanoid loco-manipulation。未来whole-body control成熟后可以integration。

4. **Fine-grained manipulation难**: 转螺丝这种task，AVP teleoperation的precision不够。

**Future direction猜测**（基于paper的trend）：
- 3D pre-training: 类似R3M但for point cloud，比如在大规模RGBD human video上pre-train point cloud encoder
- Whole-body iDP3: 把locomotion和manipulation统一在一个3D policy里
- Data scaling: 用更ergonomic teleoperation interface（比如exoskeleton）collect更多data

---

## 7. 总结：这篇paper的真正贡献

这篇paper的contribution不只是iDP3算法本身，而是**整个system integration demonstration**——证明3D visuomotor policy可以在full-sized humanoid上实现cross-scene generalization，只用single-scene data。

关键技术insight：
1. **Egocentric 3D > World frame 3D** for general-purpose robots
2. **Scale up input > Hand-craft segmentation** when segmentation不可行
3. **Longer prediction horizon** compensates for noisy human data
4. **3D representation > 2D image** for cross-scene generalization

2000+ episodes的rigorous evaluation让results可信。这不是cherry-picked demo，而是systematic study。

参考链接汇总：
- Paper主页: https://humanoid-manipulation.github.io/
- DP3原始paper: https://arxiv.org/abs/2403.03954
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- R3M: https://arxiv.org/abs/2203.12601
- DDIM: https://arxiv.org/abs/2010.02502
- EquiBot (Sim(3) equivariance): https://arxiv.org/abs/2407.01479
- Relaxed IK: https://github.com/uwgraphics/RelaxedIK
- Fourier GR1: https://www.fourierintelligence.com/gr1
- RealSense L515: https://www.intelrealsense.com/lidar-camera-l515/

如果你想深入某个具体aspect（比如diffusion policy的mathematical derivation、point cloud processing pipeline、或teleoperation的joint mapping），我可以进一步展开。
