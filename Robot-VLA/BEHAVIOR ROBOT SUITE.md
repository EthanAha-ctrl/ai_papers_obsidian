---
source_pdf: BEHAVIOR ROBOT SUITE.pdf
paper_sha256: 828e65325dc70850f10e52c39abd0676e3432dd73c90e4fcfaa80151b729fe4f
processed_at: '2026-07-18T14:26:28-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# BEHAVIOR ROBOT SUITE (BRS) 深度技术解读

你好 Andrej！这篇paper是 Stanford SVL (Stanford Vision Lab) 的工作，发表于 CoRL 2025，作者包括 Yunfan Jiang, Ruohan Zhang, Li Fei-Fei 等人。整体感觉这是一个非常"engineering-complete"的工作，从 hardware 到 data collection 到 algorithm 到 evaluation 闭环都做了，对 robot learning 这一块做了相当务实的推进。我下面从多个层次展开。

---

## 1. Paper 的核心 motivation 和切入点

BRS 的出发点是一个 data-driven 的问题：他们没有凭空设计 task，而是分析了 BEHAVIOR-1K benchmark (https://openreview.net/forum?id=_8DoIe8G3t) 中的 1000 个 everyday household tasks，反推出 robot 真正需要的三种 whole-body control capability：

1. **Bimanual coordination** (B) — 搬重物、抓大物体
2. **Stable and accurate navigation** (N) — 在 house 中长距离移动并精确定位
3. **Extensive end-effector reachability** (R) — 物体分布在不同高度

特别 interesting 的是 Figure 2 中展示的 object 垂直高度分布的多模态结构：modes 出现在 0.09 m (地面附近)、0.49 m (低家具)、0.94 m (桌面高度)、1.43 m (高柜)。这种 multi-modal 分布说明 robot 必须有灵活的 torso 来 cover 整个 vertical range，单纯靠 arm 或单纯靠 lift 都不够。

**Intuition**: 想象一个 household robot 的 workspace 是一个长方体 (ground 到 ~2m)，objects 在这个 volume 内是 non-uniformly distributed 的，集中在几个 ergonomic heights。robot 的 kinematic chain 必须能够 reach 这些 modes，这就要求了 torso (4-DoF) 的存在。

---

## 2. Hardware: Galaxea R1 + JoyLo

### 2.1 Robot 平台 Galaxea R1

R1 是 wheeled (而非 legged) dual-arm manipulator：
- **2 × 6-DoF arms** with parallel jaw gripper, max payload 5 kg per arm
- **4-DoF torso**: 2 waist/hip joints + 2 knee-like joints
- **Omnidirectional mobile base**: 3 wheel motors + 3 steering motors, max 1.5 m/s linear, 3 rad/s yaw
- **Vertical reach**: ground level → 2 m
- **Horizontal reach**: 2.06 m

传感器配置非常 critical：
- 1 × ZED 2 RGB-D (head)
- 2 × ZED-Mini RGB-D (wrist)
- 1 × RealSense T265 (visual odometry, 200 Hz)
- NVIDIA Jetson Orin onboard

Camera 在 60 Hz 出 rectified RGB + aligned depth；robot forward kinematics 在 500 Hz 更新 camera pose，这样就能在 robot base frame 下 fuse 三个 camera 的 point cloud (Figure A.2)。这是 egocentric colored point cloud 的来源。

**Intuition**: 用 wrist camera + head camera 三个 RGB-D 的 fusion 等价于让 robot 拥有一个动态的、ego-centric 的 3D semantic map。Wrist camera 在 manipulation 时提供 close-up 的细节，head camera 提供 global context。

### 2.2 JoyLo: teleoperation interface

JoyLo 是这篇 paper 的一个重要 contribution，我觉得非常 elegant。核心 design：

**Concept**: Puppeteering with kinematic-twin arms + Joy-Con thumbsticks

- Leader arms 是 3D-printed links + Dynamixel XL330 motors，和 robot arms 在运动学上 "twin" (kinematic-twin)
- **Nintendo Joy-Con** 提供 thumbsticks + buttons：
  - Left thumbstick → mobile base velocity
  - Right thumbstick → waist + hips
  - Arrow keys → torso height
  - Triggers → grippers
- 总成本 ~$500 (Table A.IV)

### 2.3 Bilateral teleoperation 的 haptic feedback

这个公式很关键：

$$\boldsymbol{\tau} = \mathbf{K_p}(\mathbf{q}_{\mathrm{robot}} - \mathbf{q}_{\mathrm{JoyLo}}) + \mathbf{K_d}(\dot{\mathbf{q}}_{\mathrm{robot}} - \dot{\mathbf{q}}_{\mathrm{JoyLo}}) - \mathbf{K}$$

变量含义：
- $\boldsymbol{\tau} \in \mathbb{R}^n$: applied to JoyLo arms 的 motor torque
- $\mathbf{q}_{\mathrm{JoyLo}}, \mathbf{q}_{\mathrm{robot}}$: 分别是 JoyLo 和 robot 的 joint positions
- $\dot{\mathbf{q}}_{\mathrm{JoyLo}}, \dot{\mathbf{q}}_{\mathrm{robot}}$: joint velocities
- $\mathbf{K_p}$: proportional gain matrix (position error 的 stiffness)
- $\mathbf{K_d}$: derivative gain matrix (velocity damping)
- $\mathbf{K}$: constant damping/offset term

**Intuition**: 这是一个 PD controller 把 robot arm 的 joint state 拉回 JoyLo arm 的 position。当 robot 撞到东西或者 user 试图 move 到一个 infeasible position，$\mathbf{q}_{\mathrm{robot}}$ 不会跟上 $\mathbf{q}_{\mathrm{JoyLo}}$，差值 $(\mathbf{q}_{\mathrm{robot}} - \mathbf{q}_{\mathrm{JoyLo}})$ 变大，对应 torque 反推到 JoyLo 上，operator 就能 "feel" 到阻力。无需 force sensor，这是非常巧妙的 bilateral teleoperation 实现。

参考：Hannaford 1989 (https://doi.org/10.1109/70.88057), Lawrence 1993 (https://doi.org/10.1109/70.258054) 是 bilateral teleoperation 的经典工作。

**为什么 kinematic-twin 重要**: Kinematic constraints 阻止 operator 生成 infeasible action。IK-based interface (VR, Apple Vision Pro) 经常给出 suboptimal IK solution，singular configuration，导致 jerky motion。User study (Figure 8) 中 JoyLo 的 singularity ratio 比 VR controller 低 78%，比 Apple Vision Pro 低 85%，这个数据非常 striking。

---

## 3. WB-VIMA: 核心算法

这是 paper 的 algorithmic contribution。WB-VIMA = **W**hole-**B**ody **V**Isuo**M**otor **A**ttention policy。

### 3.1 关键问题：为什么 flat 21-DoF 预测会失败

设想 R1 的 kinematic chain: mobile base (3-DoF) → torso (4-DoF) → arms (12-DoF + 2 gripper) = 21-DoF whole-body action。

Paper 给出一个关键的 quantitative observation：**0.17 rad (10°) 的 knee joint 误差** 在 neutral pose 下会导致 end-effector 偏移 **0.14 m**。这是典型的 error amplification 沿 kinematic chain。

**Intuition**: 想象你站在桌边伸手拿杯子，如果你的 hip 错了 10 度，你的手就会偏一大截。如果你只用 arm 去补偿，你得做出不自然的姿势，这就 out-of-distribution 了。所以必须让 arm "知道" torso 已经在哪里，再决定自己怎么 move。

### 3.2 Autoregressive whole-body action decoding

这就是 WB-VIMA 的核心 insight。Action space 被分成三个层次：

$$\mathbf{a}_{\text{base}} \in \mathbb{R}^{T_a \times 3} \rightarrow \mathbf{a}_{\text{torso}} \in \mathbb{R}^{T_a \times 4} \rightarrow \mathbf{a}_{\text{arms}} \in \mathbb{R}^{T_a \times 14}$$

其中 $T_a = 8$ 是 action prediction horizon (从 DDPM 的 chunk-based prediction 借鉴 Diffusion Policy https://arxiv.org/abs/2303.04137 的 idea)。

整个 decoding 是 autoregressive 的，对应三个独立的 denoising diffusion networks $\epsilon_{\text{base}}, \epsilon_{\text{torso}}, \epsilon_{\text{arms}}$，公式 (1)：

$$\mathbf{a}_{\mathrm{base}}^{k-1} \sim \mathcal{N}\left(\mu_k\left(\mathbf{a}_{\mathrm{base}}^{k}, \epsilon_{\mathrm{base}}\left(\mathbf{a}_{\mathrm{base}}^{k} | \mathbf{E}^{a}, k\right)\right), \sigma_k^2 I\right)$$

$$\mathbf{a}_{\mathrm{torso}}^{k-1} \sim \mathcal{N}\left(\mu_k\left(\mathbf{a}_{\mathrm{torso}}^{k}, \epsilon_{\mathrm{torso}}\left(\mathbf{a}_{\mathrm{torso}}^{k} \middle| \mathbf{a}_{\mathrm{base}}^{0}, \mathbf{E}^{a}, k\right)\right), \sigma_k^2 I\right)$$

$$\mathbf{a}_{\mathrm{arms}}^{k-1} \sim \mathcal{N}\left(\mu_k\left(\mathbf{a}_{\mathrm{arms}}^{k}, \epsilon_{\mathrm{arms}}\left(\mathbf{a}_{\mathrm{arms}}^{k} | \mathbf{a}_{\mathrm{torso}}^{0}, \mathbf{a}_{\mathrm{base}}^{0}, \mathbf{E}^{a}, k\right)\right), \sigma_k^2 I\right)$$

变量解析：
- $k$: diffusion timestep，从 $K$ (纯噪声) 递减到 0 (干净 action)
- $\mathbf{a}^{k}$: 第 $k$ 步的 noisy action
- $\mathbf{a}^{0}$: 最终 denoised action (clean sample)
- $\mu_k(\cdot)$: reverse process 的 mean function (依赖 DDPM scheduler)
- $\sigma_k^2$: 第 $k$ 步的 variance，来自 predefined schedule
- $\mathbf{E}^a$: action readout token，从 observation encoder 来的 conditioning
- $\epsilon_{\text{base}}, \epsilon_{\text{torso}}, \epsilon_{\text{arms}}$: 三个独立的 UNet-based denoising network，预测 noise $\epsilon$

**关键差异**: 
- $\epsilon_{\text{base}}$ 只 condition on $\mathbf{E}^a$
- $\epsilon_{\text{torso}}$ condition on $\mathbf{a}_{\text{base}}^0$ (denoised base) + $\mathbf{E}^a$
- $\epsilon_{\text{arms}}$ condition on $\mathbf{a}_{\text{torso}}^0 + \mathbf{a}_{\text{base}}^0 + \mathbf{E}^a$

这里 $\mathbf{a}^0$ 表示已经 denoise 完成的 clean action，作为下游 conditioning。**整个 pipeline 是 sequential 的，必须等 base denoise 完才能 denoise torso**。

**Intuition**: 这就像 hierarchical motion planning — 先决定"脚往哪走"，再决定"腰怎么弯"，最后决定"手怎么伸"。下游 joint 在知道上游 position 的情况下做 adjustment，理论上能 compensate 上游的小 error，避免 error 累积。

### 3.3 Multi-modal observation attention

Observation 来自两个模态：

1. **Egocentric colored point cloud** $\mathbf{P}^{\text{colored}}_{\text{pcd}} \in \mathbb{R}^{N_{\text{pcd}} \times 6}$, $N_{\text{pcd}} = 4096$
   - 6 channels: 3 RGB + 3 XYZ
   - 经过 PointNet (https://arxiv.org/abs/1612.00593) encode 成 $\mathbf{E}^{\text{pcd}} \in \mathbb{R}^{256}$
2. **Proprioception** 21-dim:
   - $v_{\text{mobile base}} \in \mathbb{R}^3$
   - $q_{\text{torso}} \in \mathbb{R}^4$
   - $q_{\text{arms}} \in \mathbb{R}^{12}$
   - $q_{\text{grippers}} \in \mathbb{R}^2$
   - 经过 MLP encode 成 $\mathbf{E}^{\text{prop}} \in \mathbb{R}^{256}$

加上一个 learnable **action readout token** $\mathbf{E}^a$，形成 sequence:

$$\mathbf{S} = [\mathbf{E}_{t-T_o+1}^{\text{pcd}}, \mathbf{E}_{t-T_o+1}^{\text{prop}}, \mathbf{E}_{t-T_o+1}^{\text{a}}, \ldots, \mathbf{E}_t^{\text{pcd}}, \mathbf{E}_t^{\text{prop}}, \mathbf{E}_t^{\text{a}}] \in \mathbb{R}^{3T_o \times E}$$

- $T_o = 2$: observation window (只用过去 2 步)
- $E = 256$: token dimension
- Sequence length = $3 \times 2 = 6$

通过 causal self-attention (transformer decoder, 2 layers, 8 heads, GEGLU activation, dropout 0.1)。Action token 只 attend to previous observation tokens (causality)。

**Intuition**: 三个 modality 通过 attention 而非 concatenation 融合。Concatenation 会让 model 倾向于 overfit 到 proprioception (因为 proprioception 直接和 action 相关)，而 attention 让 model 学到什么时候该看视觉、什么时候该看自己 joint state。Ablation (Figure 6) 证实了这一点：去掉 multi-modal attention 后，model "ignore visual inputs and overfit to proprioception"。

### 3.4 训练和部署

- **训练**: 三个独立的 diffusion loss, $\mathcal{L} = MSE(\epsilon^k, \epsilon_\theta(\cdot | k))$，aggregate 三个 decoder
- **DDPM**: 100 steps (train), DDIM 16 steps (inference) — 参考 DDIM (https://arxiv.org/abs/2010.02502)
- **Hardware**: NVIDIA RTX 4090, 0.02 s effective latency
- **Control freq**: data collected at 10 Hz, robot controller at 100 Hz, 每个新 action repeat 10 次

**Latency trick**: asynchronous policy inference，policy inference 在 background 持续 run，切换新 trajectory 时丢弃前几个 action 补偿 inference latency。这是 robot deployment 常见的 trick，类似 Diffusion Policy 和 UMI 的处理。

---

## 4. 实验：五项 household task

### 4.1 Task 设计

非常 diverse 的 5 个 task，每个 stress test 不同能力 (https://behavior-robot-suite.github.io 有 videos):

| Task | 主要能力 | Sub-tasks | Demonstrations | Avg human time |
|---|---|---|---|---|
| Clean house after party | Navigation | 6 ST | 138 | 210 s |
| Clean the toilet | Reachability | 6 ST | 103 | 120 s |
| Take trash outside | Navigation | 4 ST | 122 | 130 s |
| Put items onto shelves | Reachability | 2 ST | 100 | 60 s |
| Lay clothes out | Bimanual | 4 ST | 98 | 120 s |

### 4.2 主要结果 (Tables A.IX-XIII)

我整理了一下关键数据：

**Sub-task 平均成功率**:
- WB-VIMA: **88%**
- DP3 (https://arxiv.org/abs/2403.03954): ~35%
- RGB-DP (https://arxiv.org/abs/2303.04137): ~15%
- ACT (https://arxiv.org/abs/2304.13705): ~5%

**End-to-end entire task 成功率**:
- WB-VIMA: **58% 平均, 93% peak** (Put items onto shelves)
- DP3: 0% on 4/5 tasks, 20% on Put items
- RGB-DP: 0% on 4/5 tasks, 13% on Put items
- ACT: 0% on all tasks

**安全违规**: WB-VIMA 在所有 task 上几乎为 0；DP3 有 13+9+7=29 次 collision/power loss；ACT 多次撞门框

**一个惊人的数据点**: 在 "clean the toilet" 的 ST-2 (open toilet cover) 和 "lay clothes out" 的 ST-1 (open wardrobe) 这种 contact-rich 任务上，**WB-VIMA 甚至超过 human teleoperation**。Paper 解释为 "training on successful demonstrations enables it to learn precise, coordinated maneuvers" — 也就是 model 学到了 human 偶尔做不到的 consistency。

### 4.3 Ablation (Figure 6, 7)

两个关键 ablation:
1. **Without autoregressive decoding**: 性能掉 53% (Put items task)
2. **Without multi-modal attention**: 性能掉更厉害，导致 model 完全 ignore vision，且出现 4 次 collision

**Roadmap**: 
- Vanilla diffusion policy: baseline
- + Multi-modal attention: +27% (超过 ACT)
- + Autoregressive decoding: +45% (达到 WB-VIMA final 性能)

Simulation ablation (Figure 7, table wiping) 进一步确认这两个 component 在 simulation 中也是关键。

### 4.4 User study (Figure 8)

10 个 participant 对比 JoyLo / VR controllers (https://arxiv.org/abs/2403.07869) / Apple Vision Pro (https://github.com/Improbable-AI/VisionProTeleop)，在 OmniGibson simulator 的 "clean house after wild party" 上。

结果：
- **Success rate**: JoyLo 5× higher than VR controllers; Apple Vision Pro 0 participants 完成 ET
- **Completion time**: JoyLo 比 VR controllers 短 23%；比 Apple Vision Pro 快 71% (navigation) 和 67% (bowl picking)
- **Singularity ratio**: JoyLo 比 VR 低 78%，比 Apple Vision Pro 低 85%
- **User preference**: 70% 一开始以为 IK-based 会更 intuitive，结束后 **100% 都 prefer JoyLo**

**Critical insight**: IK-based interface 在 tabletop manipulation 可能 OK，但在 mobile manipulation 中，IK 无法有效 control mobile base + torso，所以 data quality 差很多。这个 user study 直接打脸了 "IK-based interface 更 intuitive" 的 assumption。

---

## 5. Coordinated whole-body behavior 的 emergent property (Figure 9)

这个我觉得是 paper 中最 cool 的 qualitative result。

**"Take trash outside" ST-3 (open the door)**: robot bends hip forward **while** advancing base to generate enough inertia to推开门。锁住 hip 或 base 的话门打不开，arm joint effort 飙升，可能损坏 hardware。

**"Clean house" ST-2 (open dishwasher)**: robot moves base backward，用整个 body pull 门 open。

**Intuition**: 这些动作不是 hard-coded 的，是从 demonstration 中学到的 emergent coordination。这正是 mobile manipulation 区别于 tabletop manipulation 的核心 — arm 自己干不动的活，需要整个 body 配合。

---

## 6. Limitations

Paper 的 limitations section 写得很 honest：

1. **Camera FOV mismatch**: operator 第三人称视角 vs. robot 第一人称视角，可能导致 partial observable data。Future work 用 active perception (https://arxiv.org/abs/2506.15666)
2. **Compounding errors in long-horizon**: ET 成功率 58% vs. ST 88%，差距来自 multi-stage 累积误差。Future work: correction data (https://arxiv.org/abs/2405.10315), task planning
3. **Point cloud 质量**: 受 lighting / reflective surface 影响。Future work: FoundationStereo (https://arxiv.org/abs/2501.09898)
4. **Robot-specific training**: 只用 R1 的数据，没用 cross-embodiment。Future work: Open X-Embodiment (https://arxiv.org/abs/2310.08864), VLA models (https://arxiv.org/abs/2307.15818, https://arxiv.org/abs/2410.24164)

---

## 7. 我的 takeaways 和 broader context

### 7.1 这个工作在 literature 中的位置

BRS 是一个**综合 system paper**，在几个 trend 的交汇点：

1. **Low-cost teleoperation interfaces**: Mobile ALOHA (https://arxiv.org/abs/2401.02117), ALOHA (https://arxiv.org/abs/2304.13705), GELLO (https://arxiv.org/abs/2309.13037), ACE (https://arxiv.org/abs/2408.11805), AirExo (https://arxiv.org/abs/2309.14975), AnyTeleop (https://arxiv.org/abs/2307.04577)
2. **Diffusion-based policy**: Diffusion Policy (https://arxiv.org/abs/2303.04137), DP3 (https://arxiv.org/abs/2403.03954), One-Step Diffusion (https://arxiv.org/abs/2410.21257)
3. **Whole-body mobile manipulation**: Mobile ALOHA, Tidybot++ (https://arxiv.org/abs/2412.10447), Bunn (https://arxiv.org/abs/2401.14403), SPIN (https://arxiv.org/abs/2405.07991), Harmonic Mobile Manipulation (https://arxiv.org/abs/2310.01822), HumanPlus (https://arxiv.org/abs/2406.10454)
4. **Household benchmarks**: BEHAVIOR-1K, RoboCasa (https://arxiv.org/abs/2406.02523), HomeRobot (https://arxiv.org/abs/2306.02523)

### 7.2 与 Mobile ALOHA 的对比

Mobile ALOHA 是非常 similar 的工作，但有一些关键 difference：
- Mobile ALOHA 用 linear rail 而非 omnidirectional base
- Mobile ALOHA 没有 torso
- Mobile ALOHA 用 ACT (VAE-based)，BRS 用 diffusion + autoregressive
- Mobile ALOHA 的 base 和 arm 是独立预测的，没有 hierarchy

BRS 在 task 复杂度 (articulated objects, deformable objects) 上明显更 ambitious。

### 7.3 Autoregressive decoding 的深层 insight

这个 idea 其实和 LLM 中的 autoregressive token generation 在哲学上是 similar 的 — 序列中的后一个 token 依赖前一个 token 的输出。在 robotics 中，这个 hierarchy 可以是：
- Spatial hierarchy (BRS): base → torso → arms
- Temporal hierarchy: sub-goal → sub-task → primitive action
- Conceptual hierarchy: VLA (https://arxiv.org/abs/2307.15818) → low-level controller

这种 "factor the action space by dependency" 的思路其实是 robot learning 中一个被 underexplored 的方向。类似的 idea 在 HumanPlus (https://arxiv.org/abs/2406.10454) 中也有，但 BRS 通过 diffusion 的 conditioning 实现得更优雅。

### 7.4 Engineering lesson 我觉得最值得学习的

1. **Cost-performance tradeoff**: $500 的 JoyLo 在 user study 中完胜 Apple Vision Pro (~$3500)。这说明 teleoperation interface 的设计哲学不是硬件越贵越好，而是要 fit robot 的 kinematic structure。
2. **Multimodal attention 优于 concatenation**: 这是 ablation 的关键发现，对应 robot learning 中常见的 "proprioception shortcut" 问题 — model 容易 overfit 到 proprioception 因为它和 action 直接相关。
3. **Latency management**: 0.02 s effective latency 是通过 asynchronous inference + DDIM 16 steps + 重复 10 次 action 实现的，这是部署 diffusion policy 到 real robot 的必备 trick。

### 7.5 一些可能的延伸联想

- **Cross-embodiment generalization**: 能否把 WB-VIMA 的 autoregressive structure 扩展到 humanoid (Tesla Optimus, Figure 01, Unitree H1)？Humanoid 的 hierarchy 是 locomotion (legs) → torso → arms，结构上类似。
- **VLM integration**: 现在 observation encoder 是 PointNet，如果换成预训练 3D backbone (如 Point-MAE) 或 VLM feature (CLIP / SigLIP)，generalization 会怎样？
- **Language conditioning**: 当前 task 是 implicit 的，如果能加 language instruction 作为额外 token 进入 attention，是否就变成了 VLA-style whole-body policy？这跟 π0 (https://arxiv.org/abs/2410.24164) 的思路可以结合。
- **Long-horizon composition**: 当前 ET 58% vs. ST 88% 的 gap 提示 long-horizon 累积误差，可以考虑 hierarchical RL (Options framework) 或者 LLM task planner (https://arxiv.org/abs/2302.01560)。
- **Active perception**: operator 视角 vs. robot 视角的 mismatch 是数据 quality 的瓶颈，引入 active vision policy 让 robot 自动调整 camera 姿态可能突破这个 bottleneck。

---

## Reference Links

主项目: https://behavior-robot-suite.github.io

核心算法 reference:
- VIMA (multi-modal prompt robot manipulation): https://arxiv.org/abs/2210.03094
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- DDPM: https://arxiv.org/abs/2006.11239
- DDIM: https://arxiv.org/abs/2010.02502
- PointNet: https://arxiv.org/abs/1612.00593
- U-Net: https://arxiv.org/abs/1505.04597
- Attention is All You Need: https://arxiv.org/abs/1706.03762

Baselines:
- DP3: https://arxiv.org/abs/2403.03954
- ACT / ALOHA: https://arxiv.org/abs/2304.13705
- Mobile ALOHA: https://arxiv.org/abs/2401.02117

Whole-body manipulation:
- Tidybot++: https://arxiv.org/abs/2412.10447
- SPIN: https://arxiv.org/abs/2405.07991
- HumanPlus: https://arxiv.org/abs/2406.10454
- Adaptive Mobile Manipulation: https://arxiv.org/abs/2401.14403
- Bimanual Dexterity: https://arxiv.org/abs/2402.01883

Teleoperation interfaces:
- GELLO: https://arxiv.org/abs/2309.13037
- ACE: https://arxiv.org/abs/2408.11805
- AirExo: https://arxiv.org/abs/2309.14975
- AnyTeleop: https://arxiv.org/abs/2307.04577
- Open-TeleVision: https://arxiv.org/abs/2407.01512
- TelemoMa: https://arxiv.org/abs/2403.07869
- UMI on Legs: https://arxiv.org/abs/2407.10353

Benchmark and datasets:
- BEHAVIOR-1K: https://openreview.net/forum?id=_8DoIe8G3t
- BEHAVIOR: https://proceedings.mlr.press/v164/srivastava22a.html
- OmniGibson: https://arxiv.org/abs/2012.04255
- RoboCasa: https://arxiv.org/abs/2406.02523
- Open X-Embodiment: https://arxiv.org/abs/2310.08864
- DROID: https://arxiv.org/abs/2403.12945
- BridgeData V2: https://arxiv.org/abs/2308.12952

Sim-to-real and correction:
- Transic: https://arxiv.org/abs/2405.10315
- MimicGen: https://arxiv.org/abs/2310.17596
- RoboCook: https://arxiv.org/abs/2306.14447

Foundation models:
- RT-2 (VLA): https://arxiv.org/abs/2307.15818
- OpenVLA: https://arxiv.org/abs/2406.09246
- π0: https://arxiv.org/abs/2410.24164
- GR00T N1: https://arxiv.org/abs/2503.14734
- FoundationStereo: https://arxiv.org/abs/2501.09898

Bilateral teleoperation classic:
- Hannaford 1989: https://doi.org/10.1109/70.88057
- Lawrence 1993: https://doi.org/10.1109/70.258054

Hardware reference:
- Galaxea R1: https://www.galaxea.ai/
- Dynamixel XL330: https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/
- Nintendo Joy-Con: https://www.nintendo.com/joycon

---

## 总结

BRS 这个工作我觉得最值得品味的有三个 layer：

**Hardware layer**: JoyLo 设计哲学 — kinematic-twin + Joy-Con thumbstick 是一个非常 pragmatic 的折衷，既避免了 IK 的 singularity 问题，又提供了 whole-body control 的统一接口。$500 的成本让它有很强的 reproducibility。

**Algorithm layer**: WB-VIMA 的 autoregressive diffusion policy 是一个 simple but powerful 的 architectural prior。它把 robotics 中早就知道的 "hierarchical control" 思想融进了现代 diffusion policy 框架，效果显著。Multi-modal attention 也是 ablation 证实的关键 component。

**System layer**: 5 个 household task + unmodified environments + 不只是 short-horizon sub-task 还包括 long-horizon entire task 评估，让这个工作有真正的 real-world credibility。

对于 build intuition 的关键点：**robot action space 不是 flat 的，是有 internal dependency structure 的，建模这个 structure 比堆 model size 更重要**。这点和 LLM 中 chain-of-thought 的 hierarchical reasoning 在精神上是相通的。

如果你 (Andrej) 想深入了解某个具体方面，比如 JoyLo 的 control loop 实现细节，或者 WB-VIMA 在不同 embodiment 上的扩展性，我可以再展开。
