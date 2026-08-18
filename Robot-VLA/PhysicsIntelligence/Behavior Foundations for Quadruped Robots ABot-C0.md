---
source_pdf: Behavior Foundations for Quadruped Robots ABot-C0.pdf
paper_sha256: 86b3f49a758ea939559df48c4f6393c58efff69c6809138809582c1610964ae8
processed_at: '2026-08-18T02:21:32-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，用更 plain 的语言讲一遍，但关键 design choice 的 why 我还是会讲清楚，因为只讲 what 不讲 why 是没用的。

---

## 1. 这篇 paper 在干嘛

一句话：把 humanoid 那波 Behavior Foundation Model (Sonic、BFM-Zero、OmniH2O、HoloMotion-1) 的范式搬到 quadruped 上，做出来一个能 tracking 任意 motion、能走复杂地形、能跟人握手的 quadruped control stack。代号 ABot-C0，跑在他们自家的 Tutu 机器狗上。

paper 的核心赌注就一个：**quadruped 缺数据，那就用 video generation model 自己造数据**。然后在这个合成数据上 reproduce 出 humanoid 已经看到的 motion tracking scaling law。

链接:
- Sonic: https://arxiv.org/abs/2511.07820
- BFM-Zero: https://arxiv.org/abs/2511.04131
- OmniH2O: https://arxiv.org/abs/2406.08858
- HoloMotion-1: https://arxiv.org/abs/2605.15336

---

## 2. 为什么 quadruped 比 humanoid 难做 BFM

Humanoid 这波能起来，靠两件法宝：
- AMASS 这种大规模人体 MoCap 数据（ https://amass.is.tue.mpg.de ）
- SMPL 拟合把视频变成 3D motion，retargeting 到 humanoid robot 几乎是同构的

quadruped 两件都没有。animal MoCap 数据极少（DogML 才 11 小时，还都是 walk/trot 这种基础步态），dog 到 quadruped robot 的 retargeting 因为骨架差异大很 fragile。 expressive behavior（打滚、跳跃、互动）就更没人采集过。

所以 paper 的逻辑是：既然没数据，就用 generative model 当数据工厂。这个思路跟 humanoid 用 SMPL reconstruction 从 YouTube 视频拉数据是同构的，只是 quadruped 没有 SMPL-equivalent，必须自己造一个 video-to-motion 的 pipeline。

---

## 3. Data Engine：用 video generation 当数据工厂

### 3.1 数据金字塔 16k clips

四类数据拼起来：

| Source | Hours | 角色 |
|---|---|---|
| Animal MoCap | 10.0 | 自然步态 |
| Teleoperation | 3.7 | cold-start demo |
| Artist Design | 0.2 | S-tier 极限动作 |
| **Video Generation** | **18.5** | expressive / acrobatic / diverse |

video generation 单独贡献了 18.5 小时，超过总量 80%。这就是真正让数据 scale 起来的 lever。

### 3.2 怎么从 text/image 生成 robot 能执行的 motion

pipeline 三步：

**Step 1: 用 Wan2.2 生成 robot 视频**
拿一张 canonical standing robot 图当 first frame，让 video diffusion model 在 I2V 模式下接着生成。但有个 bug：video diffusion 会 non-rigid 变形 robot 形态，破坏下游 3D extraction 的 rigid-body 假设。所以他们加了一个 Identity Consistency Loss：

$$\mathcal{L}_{\mathrm{IC}} = \frac{1}{T}\sum_t \max\Bigl(0,\; m_{\mathrm{id}} - \max_j \cos(f_t, f_{\mathrm{ref}}^{(j)})\Bigr)$$

$f_t$ 是第 $t$ 帧的 DINOv2 CLS embedding，$f_{\mathrm{ref}}^{(j)}$ 是 appearance bank 里的 reference embedding。如果当前帧跟最近 reference 的 cosine 相似度低于 margin $m_{\mathrm{id}}$，就产生 loss。这个 loss 只挂在 Wan2.2 的 low-noise expert 上——因为高噪声阶段预测的 clean frame 本身就不可信，DINOv2 打分没意义。

**Step 2: 把视频转成 3D trajectory**
因为 I2V 设定下 camera 是固定已知的，frame-0 就是 URDF canonical pose，所以 3D 重建退化成 temporally constrained kinematic fitting。每帧预测 $K$ 个 body landmark 的 2D 位置，然后最小化重投影误差：

$$L_{\mathrm{reproj}}^{(t)} = \sum_k \|\Pi(\mathrm{FK}(\mathbf{s}_t)_k) - \mathbf{p}_{2D,k}^{(t)}\|^2$$

$\mathbf{s}_t = (\mathbf{p}_t, \phi_t, \boldsymbol{\theta}_t)$ 是待求的 per-frame state（root 位置、Euler 角、12 个 joint angle），$\mathrm{FK}$ 是 URDF forward kinematics，$\Pi$ 是固定 camera 投影。

**Step 3: 三级 quality filter**
- CLIP semantic gate（97% pass）：在 MuJoCo 重渲染 trajectory，跟原视频比 CLIP 相似度
- Geometric gate（70.2% pass）：reprojection error per-clip mean < 20 px。这是 bottleneck，但他们故意 conservative——over-generate 再 discard 比 relax 阈值让 artifact 进训练安全得多
- Physical feasibility gate（97.6% pass）：给每条 trajectory 训一个 specialist tracking policy，rollout 一遍看会不会 fall。用 simulator 当 reward model 来 filter data

参考:
- Wan2.2: https://arxiv.org/abs/2503.20314
- DINOv2: https://arxiv.org/abs/2304.07193
- ViTPose: https://arxiv.org/abs/2204.12484
- QuadFM (他们的 motion generation paper): https://arxiv.org/abs/2603.24021

---

## 4. Motion Tracking：核心方法 + 验证 scaling law

这是 paper 最 humanoid-BFM-风格的部分。

### 4.1 Specialist-to-Generalist 三段式

为什么不能直接训一个 multi-motion RL policy？因为不同 motion clip 的 dynamics 差很大，gradient 会打架。Table 3 里 multi-motion RL 直接训只能拿到 22.18 mm MPJPE，而 per-clip specialist 能到 11.66 mm。差了 2 倍，这就是 gradient interference 的代价。

所以他们走三步：

**Stage 1: 给每个 clip 训一个 specialist**
每条 motion clip $m_k$ 独立训一个 PPO policy $\pi_{\mathrm{expert}}^k$，只看单帧 tracking observation。这是 upper bound——单个 clip 能 fit 得多好。

**Stage 2: 用 DAgger 蒸馏到 Flow-Matching student**
Student 在 simulator 里自己 rollout，在 visited state 上 query 对应 specialist 拿 expert action，再用 conditional Flow-Matching 拟合。关键点：DAgger 是在 student-induced state distribution 上做监督，不是在 reference 固定 state 上——这是经典 covariate shift fix。

为什么用 Flow-Matching 不用 Diffusion？三个原因：
1. Inference 只要 $D=5$ 步 ODE，对 50Hz control loop 友好；diffusion 通常要 20-1000 步
2. Flow-Matching 用 velocity field 表达 multimodal action distribution 比 GMM 灵活
3. Linear interpolation path $\mathbf{a}_t = (1-t)\mathbf{a}_{\mathrm{expert}} + t\epsilon$ 是 optimal transport 解，比 diffusion forward/reverse 加噪更直接

Flow-Matching loss:

$$\mathcal{L}_{\mathrm{FM}}(\theta) = \mathbb{E}\left[\|v_\theta(\mathbf{a}_t, t, \mathbf{o}) - (\epsilon - \mathbf{a}_{\mathrm{expert}})\|^2\right]$$

$v_\theta$ 是要学的 velocity field，target velocity 是 $\epsilon - \mathbf{a}_{\mathrm{expert}}$（从 expert 指向 noise 的方向），inference 反向积分。$t\sim\mathrm{Beta}(1.5, 1.0)$ 偏向小 $t$，让训练更关注接近 expert 的 regime。

**Stage 3: Residual RL 微调**
在 frozen flow policy 上挂一个小 residual actor，bounded 在 $\pm 0.5$ scale 0.2。Table 3 显示这个 residual 主要是 local refinement，seen MPJPE 从 12.04 → 11.98，unseen 几乎不变。设计直觉：flow policy 已经把 simulation tracking 解得很好，长 residual PPO 会 drift 掉 distilled behavior，所以 residual 必须轻。

### 4.2 验证 scaling law

Table 4 是 paper 最关键的实验：

| Samples | Seen MPJPE | Unseen MPJPE | Seen-Unseen Gap | Unseen Succ |
|---|---|---|---|---|
| 30 | 14.44 | 24.61 | 10.17 | 84.30% |
| 100 | 11.70 | 20.27 | 8.58 | 85.26% |
| 1,000 | 12.04 | 16.51 | 4.47 | 86.42% |
| 3,000 | 11.78 | 15.15 | 3.37 | 88.22% |
| **7,076 (full)** | 12.38 | **14.79** | **2.41** | **88.54%** |

unseen MPJPE 从 24.61 单调降到 14.79，seen-unseen gap 从 10.17 缩到 2.41。这是 quadruped motion tracking 第一个 explicit scaling law，跟 humanoid Sonic/GMT 看到的趋势一致。

一个有意思的细节：seen MPJPE 在 full-data 下比 100-motion 时还略差（12.38 vs 11.70）。这是经典的 underfitting-vs-generalization tradeoff——数据规模上去了，model capacity 没同步放大，seen 上 fit 能力下降但泛化提升。下一步应该同步把 transformer layers 或 latent dim 拉大。这是 LLM scaling-style 的 open question。

### 4.3 PRF Motion Curation：fixed budget 下怎么选 data

数据再多，能不能在 fixed budget 下选到更优的子集？他们设计三个互补 score：

- **Physical feasibility** $p(m) = \exp(-\mathbf{w}_p^\top \mathbf{v}^{\mathrm{phys}}(m))$：joint dynamics / root motion / foot sliding 等 violation 的负指数
- **Rollout executability** $r(m) = s(m)\exp(-\bar{e}(m))$：fixed-policy success rate 乘 tracking error 的负指数
- **Flow confidence** $f(m) = \exp(-\gamma_f \sigma_{\mathrm{flow}}(m))$：同一 observation 多次采样的 action variance，variance 小说明 flow policy 有把握

组合 PRF: $S_{\mathrm{cur}}(m) = \lambda_p p + \lambda_r r + \lambda_f f$，权重 $(0.45, 0.35, 0.20)$。

Table 5 ablation：在 70% budget 下，PRF 在 seen 和 unseen 都最好（11.17 / 15.56 mm vs random 14.01 / 16.04 mm）。三个 signal 互补——physical 偏静态可执行，rollout 偏 closed-loop，flow confidence 偏 epistemic uncertainty。这跟 LLM data curation 里 complexity+quality+diversity 选数据是同构的。

### 4.4 MCRC：Manifold-Calibrated Reference Conditioning

这是 motion tracking 部分最 novel 的点。

**问题**：frame-level tracking command 只告诉 policy "现在要做什么动作"，不告诉 "后面 0.4 秒要做什么"。比如同一个 stand-to-sit 当前帧，后面可能接 stand-up，也可能接 roll-over，两种情况需要的预备动作完全不同。

**方法**：训一个 reference-window VAE，把长度 $H=20$ 的 lookahead window 压成 32 维 latent code $\mathbf{z}_t$，作为额外 condition 注入 student policy。

VAE loss:
$$\mathcal{L}_{\mathrm{VAE}} = \|\hat{\mathbf{x}} - \mathbf{x}\|^2 + \beta D_{\mathrm{KL}}(q\|\mathcal{N}(0, I))$$

Student observation 变成：
$$\mathbf{o}_t^{\mathrm{student}} = [\mathbf{o}_t, \mathbf{z}_t(m)] \in \mathbb{R}^{69+32} = \mathbb{R}^{101}$$

Table 6 ablation:

| Observation | Seen MPJPE | Unseen MPJPE | Unseen Succ |
|---|---|---|---|
| $\mathbf{o}_{69}$ base | 12.38 | 14.79 | 88.54% |
| $\mathbf{o}_{69} \oplus \mathbf{z}$ | **11.77** | **12.53** | **91.02%** |
| $\mathbf{o}_{69} \oplus \mathbf{z} \oplus \mathbf{e}_{\mathrm{recon}}$ | 11.91 | 12.86 | 90.76% |

$\mathbf{z}$ 单独加最有效，unseen MPJPE 砍 2.26 mm，unseen success +2.48 pp。reconstruction error $\mathbf{e}_{\mathrm{recon}}$ 单独加有点用，跟 $\mathbf{z}$ 一起加反而略差——说明 reliability cue 在已经 manifold-conditioned 的情况下 redundant，会引入 noise。

这个 idea 跟 LLM 里 prefix/prompt conditioning 是同构的——都是给 generative model 一个 context 来约束输出分布。manifold latent 等于把 lookahead 信息压成 32 维 prompt 注入 policy。

---

## 5. Locomotion：三层 progressive stack

Locomotion 这部分拆成三层递进，每层独立训，避免 reward interference。

### 5.1 Robust Baseline：让 robot 感知自己的物理 state

三个机制叠起来：

**(a) Barlow Twins-style temporal consistency**
从 10-step proprioception history 取两个相邻 5-step views，shared encoder 输出 feature，cross-correlation matrix $C$ 正则化到 identity。对角项大让 feature 信息充分，off-diagonal 小压 redundancy。不用 negative sample，比 contrastive 简单。Actor mask 掉 privileged base velocity，等于让模型从 proprioception 历史里 implicit 推断 base velocity——这是 implicit system identification。

**(b) Explicit state estimation**
挂 regression head 估 base velocity、payload mass、CoM offset，让 robot online 感知自己物理 state，配合 gravity-gated reward 自适应调 stiffness。

**(c) NP3O：硬件安全的关键工程**

这是 CMDP (Constrained MDP) 的 Lagrangian 解法变体。把 joint position、velocity、torque 当 hard constraint：

$$L_{\mathrm{viol}} = \sum_i \lambda_i^{(t)} \max(0, C_{\mathrm{surr}}^{(i)} + \tilde{v}_i)$$

ReLU gating 让 budget 内 violation 梯度为 0，不阻碍 exploration；超 budget 才施加 penalty，$\lambda_i^{(t)}$ 随训练指数增长越来越严。

Table 8 在 3.0 m/s sprint stress test：
- Vanilla PPO: 124.8 次 torque violation，55% fall
- Penalty-based PPO: 17.2 次，15% fall  
- NP3O: **0 次，0% fall**

这个 0 violation 在真实部署里价值巨大——robot aggressive maneuver 下不会烧电机、不会失控。

参考:
- Barlow Twins: https://arxiv.org/abs/2106.04956
- PPO: https://arxiv.org/abs/1707.06347

### 5.2 Diff-CAST：Biomimetic Gait + Omnidirectional

AMP (Adversarial Motion Prior) 是 locomotion imitation 事实标准，但在 quadruped 上有三个 bug：mode collapse、unbounded reward 不稳、forward-biased data 让 lateral/backward 命令 drift。

Diff-CAST 三招同时解：

**(a) Action-Agnostic Diffusion Prior**
不判别 (state, action) pair，直接建模 state transition $\mathbf{x}_t = (s_t, s_{t+1})$。这把 stylistic learning 从 actuator domain 解耦——拿 animal MoCap 训 diffusion model 不需要 torque data，因为 action 根本不进 model。

Classification probability:
$$D_\varphi(\mathbf{x}_t) = \frac{\exp(-L^+(\mathbf{x}_t))}{\exp(-L^+ + \exp(-L^-)}$$

这是把 diffusion ELBO 当 likelihood 估计，再用贝叶斯比率得分类概率——跟最近 diffusion-as-reward 的方法一个 family。

**(b) Bounded Stylistic Reward**
$$r_{\mathrm{diff}} = D_\varphi(\mathbf{x}_t) \in [0, 1]$$
直接用分类概率，不取 $\log D - \log(1-D)$。后者在早期 D→0 时 reward → −∞ 不稳定 PPO value network。bounded 保留 mode-seeking gradient 但消灭 spike。

**(c) SACC: Symmetric Augmented Command Construction**
Animal MoCap 本质是 forward-biased，unconditioned prior 会 override lateral/backward 命令。SACC 在 data 和 architecture 两个 level 注入对称性：
- Kinematic symmetry: sagittal-plane mirror operator $M(\cdot)$ swap contralateral legs，data augmentation 平衡 unilateral bias
- Structural symmetry loss: actor-critic 网络约束 mirrored input 输出 mirrored action
- Yaw invariance: heading-dependent planar features 在 diffusion update 时随机 rotate $\delta \sim U(-\pi, \pi)$，合成 360° 数据

Table 10 结果：
- Forward walk deviation: baseline 25.03 m → Diff-CAST 1.08 m，heading drift 2.07 → 0.13 rad
- Pure backward: baseline **fail (OOD)** → Diff-CAST 0.21 m

w/o SACC 的 FGD 反而更低（348.55 vs 489.13），但 forward deviation 25.03 m——它过拟合 forward-biased data，gait 看上去更 "expert-like" 但完全牺牲了 command tracking。SACC 在 feature level 解耦了 naturalness 和 precision。

参考:
- AMP: https://arxiv.org/abs/2104.02180
- ASE: https://arxiv.org/abs/2205.01906
- Diff-CAST (他们的 paper): https://arxiv.org/abs/2605.08804

### 5.3 All-Terrain Locomotion：三阶段 privileged-to-perceptive

framework 三步：
1. **Privileged teacher**: clean proprioception + height map + dynamics variables (friction, mass, CoM, push)
2. **Clean LiDAR memory distillation**: frozen teacher，只训 LiDAR memory encoder + auxiliary heads。Student input: 8-frame LiDAR memory，每帧 voxelized 成 body-frame 3D occupancy grid + scan-age info。Temporal encoder = per-frame CNN + GRU + ego-motion compensation
3. **Noisy on-policy PPO fine-tune**: student 用自己 action rollout，加 LiDAR domain randomization (sensor noise, point dropout, pose perturbation, scan delay)，teacher regularization (BC, KL, latent matching) 逐渐 decay

这跟 ETH Hutter group 的 perceptive locomotion (Miki et al. Science Robotics) 一脉相承，加了 temporal LiDAR memory 和 terrain-predictive supervision 两个 novel 点。

Table 12 ablation:
- Proprioception-only: 28.0% success, max level 2.2
- Full method: 83.2% success, max level 7.8
- w/o Memory: 72.4% (memory 贡献 ~11 pp)
- w/o Ego-motion Compensation: Bad Impulse 18.5 → 43.0 N·s（最大退化）
- w/o Terrain Reconstruction: 68.6% success

ego-motion compensation 最关键——LiDAR scan 之间 robot 移动了，不补偿的话 grid alignment 错位，直接导致更大冲击。这是很直觉的。

参考:
- Miki et al. Science Robotics: https://www.science.org/doi/10.1126/scirobotics.abk2822
- Extreme Parkour (Pathak): https://arxiv.org/abs/2309.14341

---

## 6. Scene Interaction：Hand-shake 作为 case study

这个 case 体现了 paper 的设计哲学——**不学 monolithic end-to-end VLA policy，复用已有 module 拼起来**。

Pipeline: Perception → Locomotion approach → IK reference generation → Motion tracking execution → Compliant interaction

**为什么这么分？** 一锅炖 end-to-end RL 的话，perception uncertainty、approach behavior、contact timing、balance、compliance 全要塞进一个 reward，需要大规模 vision-based teleoperation data。decompose 的话：locomotion 负责接近，IK 生成 reaching trajectory，motion tracking 负责动态执行，PD gain 调低负责 compliant interaction。每个模块独立可 iterate。

**Locomotion approach command** (公式 25):
$$\mathbf{u}_{\mathrm{loc}} = [k_{xy}(x_h - d_x), k_{xy} y_h, k_\psi \mathrm{atan2}(y_h, \max(x_h, \epsilon))]^\top$$

$x_h, y_h$ 是检测到的 hand 位置在 robot base frame，$d_x$ 是 desired standoff 距离。robot 接近时 yaw 朝向手，$x_h$ 接近 $d_x$ 时停下。

**IK reference generation** (公式 26):
$$\mathbf{q}_{1:T}^{\mathrm{IK}} = \arg\min \sum_t \Bigl(\|W_F(p_F(\mathbf{q}_t) - \mathbf{p}_h^B)\|^2 + \sum_{i\in S} w_i^2 \|p_i - p_i^0\|^2 + \|W_q(\mathbf{q}_t - \mathbf{q}^0)\|^2\Bigr)$$

三项：active foot 追 hand target；support feet + base anchor 保持 nominal；posture 不要离 standing 太远。生成 stand → hand-shake → stand 的 reaching trajectory，喂给 motion tracking policy 训练。

**Adaptive completion detection** (公式 27):
$$s(t) = \max_{j\in\mathcal{A}} |\dot q_j(t) - \dot q_j^{\mathrm{cmd}}(t)|$$
active thigh + calf 关节实际速度跟 command 速度差大，说明有外力作用。warm-up 后 sustained above-threshold 标记 "engaged"，之后 quiet interval 触发 return。期间 PD gain 降到最低让 robot 顺从人手运动。

Table 13: 最终 mean exec error 13.74 cm。听起来大，但这是动态握手过程中的偏差（robot 在顺从人的 motion，不是静态 hold 一个点）。IK reference 本身 mean 1.18 cm 很准，tracking 加 ~5 cm，compliant execution 累积到 13.7 cm，是合理叠加。

---

## 7. Unified Deployment Stack

### 7.1 硬件 (Tutu)

- 3× SENSING cameras, 120°×90° FOV
- 96-channel RoboSense Airy LiDAR
- AISpeech microphone-speaker
- NVIDIA Jetson AGX Orin 64GB
- Cloud offload for heavy reasoning

### 7.2 Runtime 分层

- **Locomotion runner**: 常驻进程，接 velocity command
- **Motion-tracking runner**: 常驻进程，执行 reference motion library / IK trajectory
- **Coordination layer**: 决定哪个 active，blend motor target，safety gating

Motor command 200 Hz，decision/inference 50 Hz。Position-control policy 输出 target joint position，low-level PD:
$$\boldsymbol{\tau}_t = \mathbf{k}_p \odot (\mathbf{q}_t^{\mathrm{tar}} - \mathbf{q}_t) - \mathbf{k}_d \odot \dot{\mathbf{q}}_t$$

**Key insight**: locomotion 和 motion-tracking runner 常驻，切换时不用 reload policy 不用 rebuild observation buffer，interaction latency 显著降低。这是 deployment 工程上很务实的取舍。MuJoCo 和 hardware 共用 robot-state / motor-command 抽象，可以 sim-to-sim verify 再 sim-to-real。

---

## 8. Applications

### 8.1 Interactive Companionship

Cloud-side reasoning + onboard control 分离。Agent 接 multimodal input (audio + video)，重型 reasoning offload 到 cloud，翻译成 compact robot command 发给 onboard stack：
- Goal-directed 指令 ("come here") → locomotion policy
- Expressive request → text-to-motion 模块生成 reference trajectory → motion-tracking policy

可以 real-time interrupt，robot 安全 recover 到 standing 再接下一条。三个 demo: approach-and-handshake / gesture-aware response / exercise companionship (一起做单臂俯卧撑)。

### 8.2 All-Terrain Autonomous Navigation

VLN-based planner 生成 waypoint sequence，local obstacle-avoidance 转 velocity command，control system 基于 3D point-cloud 实时 adapt gait。Figure 9 展示不规则 outdoor slope、auto escalator、不规则楼梯上下都 traverse 成功。

参考 VLN/VLA:
- PaLM-E: https://arxiv.org/abs/2303.03378
- RT-2: https://arxiv.org/abs/2307.15818
- OpenVLA: https://arxiv.org/abs/2406.09246
- π0.5: https://arxiv.org/abs/2504.16054
- SayCan: https://arxiv.org/abs/2204.01691

---

## 9. Karpathy 视角的几个 takeaways

**1. Scaling law 在 robot motion 上是真的，不只是 LLM 现象**
Table 4 那条曲线——30 到 7,076 motion，unseen MPJPE 单调下降，seen-unseen gap 单调缩小。这是 quadruped motion tracking 的第一个 explicit scaling law。Humanoid Sonic/GMT 已经看到，现在 quadruped reproduce 出来。下一步该同步把 model capacity 拉大看 seen fidelity 能不能 hold 住。

**2. Specialist → Generalist + Flow-Matching 蒸馏是 scalable imitation 的有效范式**
Specialist 训练拿 11.66 mm（single-clip upper bound），直接 multi-motion RL 只有 22.18 mm（gradient interference 代价），Flow-Matching distillation 拉回 12.04 mm 同时拿到 generalization。gradient interference 的代价被 distillation 显式买单。这跟 humanoid OmniH2O/GMT/ExBody2 方法论一致。

**3. Video generation 是 quadruped 数据稀缺的破解路径**
Humanoid 靠 SMPL reconstruction 从 YouTube 拉数据，quadruped 没有 SMPL-equivalent，必须自建 I2V + 3D extraction pipeline。Wan2.2 + identity consistency loss + 三级 filter 是一个可行的 data engine 范式。

**4. Manifold latent conditioning 是 lookahead information bottleneck 的 elegant 实现**
MCRC 把 20-frame lookahead window 压成 32 维 latent 注入 policy，unseen success 88.54% → 91.02%。这跟 LLM 里 prefix/prompt conditioning 是同构的——都是给 generative model 一个 context 约束输出分布。reconstruction error 当 reliability cue 反而 redundant，因为 manifold-conditioned policy 自己能 sense OOD。

**5. Multi-policy stack vs unified policy 的取舍**
ABot-C0 选 multi-policy 是为了 stability（每个 sub-policy reward 独立，避免 reward interference），但演进方向一定是 unified transformer architecture（参考 humanoid BFM-Zero / π0.5 / Sonic）。paper 在 Section 7 limitations 里也明确承认这点。

**6. NP3O 的 0 violation 是 deployment 的硬通货**
3 m/s sprint 下 vanilla PPO 124 次 torque violation，penalty PPO 17 次，NP3O 0 次。real robot aggressive maneuver 不烧电机不失控，这是产品级部署必须的。

**7. Decompose interaction 比 monolithic end-to-end 更务实**
hand-shake case 用 perception + locomotion + IK + motion tracking + compliant PD 拼起来，避免一锅炖 end-to-end RL 需要大规模 vision-based teleoperation data。这种 module-reuse 思路在数据稀缺形态下更 scalable。

---

## 10. Limitations & 未来方向

1. Multi-policy stack 而非 unified model——未来应 unified transformer，condition 在 task intent + motion reference + terrain + interaction target
2. Brain-body interface (cloud reasoning + onboard control) 的 command abstraction / latency / uncertainty / safety / recovery 是 open problem
3. Cross-platform evaluation 需要验证到更多 quadruped morphologies
4. Contact-rich interaction 扩展到 push / lift / collaborative carry
5. Self-improving loop——robot 自主 collect real-world experience, identify failures, expand motion repertoire, refine policies (continual skill discovery on hardware)

参考未来方向:
- ENPIRE: https://arxiv.org/abs/2606.19980
- Agentic Skill Discovery: https://arxiv.org/abs/2506.04916
- TWIST2: https://arxiv.org/abs/2511.02832

---

整体直觉：ABot-C0 在 data 上激进（video generation 闯），在 policy 上保守（multi-policy stack 求稳），这种组合在 quadruped 数据稀缺的当下是合理的第一步。scaling law 已经 reproduce，下一步自然是 unified architecture + self-improving loop。

---

# ABot-C0: Quadruped Behavior Foundation Model 深度技术解读

你好 Andrej, 这篇 paper 实际上是把 humanoid 领域刚刚跑通的 Behavior Foundation Model (BFM) 范式 — Sonic, BFM-Zero, HoloMotion-1, OmniH2O 这一波 — 显式地迁移到 quadruped 形态。我下面会按照 "data → policy → deployment" 的逻辑层层展开, 重点讲清楚每个设计选择背后的 intuition。

---

## 1. 定位与 motivation: 为什么 quadruped BFM 不只是 humanoid BFM 减两条腿

Humanoid BFM 之所以能 scaling, 是因为 (a) AMASS / SMPL 这套 motion capture 数据生态非常成熟, (b) retargeting 从 human skeleton 到 humanoid robot 仍是同构的 bipedal 链。Quadruped 两个条件都缺位:
- Animal MoCap 数据稀缺且行为覆盖窄 (DogML 才 11 小时)
- Cross-embodiment retargeting fragile: dog / cat / quadruped-robot 的 joint configuration 不同, 加上 quadruped robot 是 12 DoF 为主, 比 humanoid 7+7+1 的 arm+torso 自由度小很多, 但是逆向地使得 expressive behavior 反而更难设计

所以 ABot-C0 的核心赌注是: 用 generative video model 作为 scalable data 来源, 把 motion tracking scaling law 在 quadruped 上 reproduce 一遍, 同时搭一个能 orchestrate 多个 policy 的 deployment stack 而不是用一个单 monolithic VLA。

参考 humanoid BFM 路线图以建立 intuition:
- Sonic: https://arxiv.org/abs/2511.07820
- BFM-Zero: https://arxiv.org/abs/2511.04131
- HoloMotion-1: https://arxiv.org/abs/2605.15336
- OmniH2O: https://arxiv.org/abs/2406.08858
- ExBody2: https://arxiv.org/abs/2412.13196
- ASAP (sim-to-real alignment for agile humanoid): https://arxiv.org/abs/2502.01143

---

## 2. Data Engine: 一个 video-to-motion 的合成 pipeline

### 2.1 数据金字塔 (16,074 clips, 22.43 hours)

| 来源 | Clips | Hours | 角色 |
|---|---|---|---|
| Motion Capture (basic gaits) | 7,998 | 10.02 | 自然 locomotion pattern |
| Teleoperation | 547 | 3.73 | cold-start demonstration |
| Artist Design | 41 | 0.19 | S-tier extreme maneuvers |
| **Video Generation** | **7,488** | **18.51** | acrobatic / expressive / diverse behaviors |

直觉上的解读: teleoperation 与 artist design 加起来不到 4 小时, video generation 单独贡献了 18.51 hours 的 expressive content — 这是真正让数据 scale 起来的 lever。这点和 humanoid BFM 用 SMPL 重建大规模视频是同构思路, 但 quadruped 没有 SMPL-equivalent, 必须自己造一个 retargeting pipeline。

### 2.2 Identity-Consistent Video Generation (公式 1)

Video diffusion model (他们用 Wan2.2) 在帧之间会出现 non-rigid body deformation, 这会破坏下游 3D extraction 的 rigid-body 假设。他们用 first-frame I2V 固定 camera 和 background, 再额外加一个 identity consistency loss:

$$\mathcal{L}_{\mathrm{IC}} = \frac{1}{T}\sum_{t=1}^{T}\max\Bigl(0,\; m_{\mathrm{id}} - \max_{j\in[N]}\cos(f_t, f_{\mathrm{ref}}^{(j)})\Bigr)$$

变量解释:
- $T$: clip 帧数 (上标时间索引)
- $f_t$: 第 $t$ 帧经过 frozen VAE decode + DINOv2 提取的 CLS embedding
- $f_{\mathrm{ref}}^{(j)}$: appearance bank $\mathcal{B} = \{f_{\mathrm{ref}}^{(j)}\}_{j=1}^{N}$ 中的第 $j$ 个 reference embedding; 这个 bank 通过 greedy coverage-set search 在 DINOv2 CLS embedding 空间用 cosine 阈值 $\tau=0.8$ 构建, 目的是用最少的 reference 覆盖最广的 appearance 子空间
- $m_{\mathrm{id}}$: hinge margin, 当当前帧 embedding 与最近 reference 的 cosine 相似度低于 $m_{\mathrm{id}}$ 时才会产生 loss
- 下标 $\mathrm{IC}$ = Identity Consistency, $\mathrm{FM}$ = Flow-Matching

总训练目标 $\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{FM}} + \lambda\mathcal{L}_{\mathrm{IC}}$, 且 $\mathcal{L}_{\mathrm{IC}}$ 只挂在 Wan2.2 的 low-noise expert 上 — 因为只有 low-noise 区域, predicted clean video 在感知上才足够可靠让 DINOv2 打分。这等于只在 "video diffusion 已经基本知道长什么样" 的 regime 上加 identity 约束, 避免高噪声早期阶段产生不可信的梯度。

### 2.3 Video → 3D Motion Trajectory (公式 2)

I2V 设定让 camera intrinsics/extrinsics 已知且固定, frame-0 pose 等于 URDF canonical standing state。这把一般 ill-posed 的 monocular 3D reconstruction 退化成 temporally constrained kinematic fitting。

每帧预测 $K$ 个 body landmark 的 2D 位置 (用 fine-tuned ViTPose), 然后最小化重投影误差:

$$L_{\mathrm{reproj}}^{(t)} = \sum_k \|\Pi(\mathrm{FK}(\mathbf{s}_t)_k) - \mathbf{p}_{2D,k}^{(t)}\|_2^2$$

- $\mathbf{s}_t = (\mathbf{p}_t, \phi_t, \boldsymbol{\theta}_t)$: 待求的 per-frame state, root 位置 $\mathbf{p}_t\in\mathbb{R}^3$, root Euler 角 $\phi_t\in\mathbb{R}^3$, 12 actuated joint angle $\boldsymbol{\theta}_t\in\mathbb{R}^{12}$
- $\mathrm{FK}(\mathbf{s}_t)_k$: 给定 $\mathbf{s}_t$ 通过 URDF forward kinematics 算出的第 $k$ 个 landmark 的 3D 位置
- $\Pi$: 固定 camera 的投影函数
- $\mathbf{p}_{2D,k}^{(t)}$: ViTPose 预测的第 $k$ 个 landmark 在第 $t$ 帧的 2D 位置

再加 temporal smoothness penalty 和 foot-contact constraints (检测到 stance 的脚锚定到地面), 避免 drift 和 sliding。

### 2.4 三级 quality filter (cost 递增)

| Gate | Pass rate | 作用 |
|---|---|---|
| CLIP semantic gate | 97.0% | 在 MuJoCo 重渲染 trajectory, 计算与原视频帧的 CLIP cosine 相似度, 过低丢弃 |
| Geometric gate | 70.2% | per-clip mean reprojection error < 20 px, max frame < 100 px |
| Physical feasibility gate | 97.6% | 给每条 trajectory 训一个 specialist tracking policy, full-length rollout 不触发 termination (fall / root divergence / velocity explosion) |

第二级 70.2% pass rate 是 bottleneck, 这是 deliberately conservative 的 — over-generate 然后 discard, 比 relax threshold 让 artifact 进入 policy training 安全得多。第三级特别有意思: 用一个小 specialist policy 的 closed-loop rollout 作为 physical feasibility oracle, 等于让 simulator 当 reward model 来 filter data。这个 idea 跟 humanoid 领域的 "physically plausible" filtering 是一致的, 例如 OmniH2O / ASAP 也用 simulator validation。

参考 paper:
- Wan2.2 video generation: https://arxiv.org/abs/2503.20314
- DINOv2: https://arxiv.org/abs/2304.07193 (实际链接: arXiv 2304.07193)
- ViTPose: https://arxiv.org/abs/2204.12484
- QuadFM (他们的 motion generation paper): https://arxiv.org/abs/2603.24021
- Unleashing infinite motion (their video prior paper): https://arxiv.org/abs/2606.28237

---

## 3. Motion Tracking: 把 Flow-Matching 推到 quadruped 上, 并验证 scaling law

### 3.1 Specialist-to-Generalist pipeline 的三个阶段

这是整篇 paper 最 humanoid-BFM-风格的部分。整个 pipeline 画在 Figure 2 里:

**Stage A — Per-motion PPO Specialist**
对每个 motion clip $m_k$ 训一个独立 $\pi_{\mathrm{expert}}^k$, 只看 single-frame tracking observation。单一 clip 训练避开 multi-clip 时不同 dynamics 之间的 gradient interference, 这是 humanoid tracker (OmniH2O, ExBody2, GMT) 都采用的策略。

**Stage B — DAgger Distillation 到 Flow-Matching Student**
Student 在 simulator 里 rollout, 在 visited states 上 query 对应 specialist 得到 expert action, 再用 conditional Flow-Matching 拟合。这等价于把 student-induced state distribution 当作 supervision target, 而不是把 reference 上的固定 state distribution 当 target — 这是 imitation learning 里经典的 covariate shift 修正。

**Stage C — Residual RL**
在 frozen flow policy 上挂一个小 residual actor $\Delta\mathbf{a}$, bounded by clip 阈值 $c=0.5$ 和 scale $s=0.2$。注意这里的 $\Delta\mathbf{a}$ 是加在 flow policy 输出上, 不是加在 reference 上。

### 3.2 Flow-Matching loss 的具体形式 (公式 3-5)

采样路径是 linear interpolation:
$$\mathbf{a}_t = (1-t)\mathbf{a}_{\mathrm{expert}} + t\epsilon,\qquad t\sim\mathrm{Beta}(1.5, 1.0)$$

- $t\in[0,1]$: flow-matching 时间变量, $t=0$ 对应 expert action (干净数据), $t=1$ 对应纯噪声 $\epsilon$
- $\epsilon\sim\mathcal{N}(0, I)$: base noise
- $\mathrm{Beta}(1.5, 1.0)$: 偏向小 $t$ 的采样分布, 让训练更关注接近 expert 的 regime — 类似 diffusion 里 reweight timestep 的 trick
- $\mathbf{a}_t$: 时间 $t$ 处的 noisy action
- $\mathbf{o}$: observation (conditioning)

velocity field 学的目标是 $\epsilon - \mathbf{a}_{\mathrm{expert}}$ (从 expert 指向 noise 的反向), loss:

$$\mathcal{L}_{\mathrm{FM}}(\boldsymbol{\theta}) = \mathbb{E}_{t,\epsilon,\mathbf{a}_{\mathrm{expert}}}\left[\|v_\theta(\mathbf{a}_t, t, \mathbf{o}) - (\epsilon - \mathbf{a}_{\mathrm{expert}})\|^2\right]$$

推理用 reverse Euler integration (从噪声端到 expert 端):
$$\mathbf{a}_{t-1/D} = \mathbf{a}_t - \frac{1}{D}v_\theta(\mathbf{a}_t, t, \mathbf{o}),\qquad t=1, \frac{D-1}{D}, \ldots, \frac{1}{D}$$

注意符号: 上面 $\mathcal{L}_{\mathrm{FM}}$ 里 velocity target 是 $\epsilon - \mathbf{a}_{\mathrm{expert}}$ (从 expert 到 noise 方向), 而 inference 用 $\mathbf{a}_{t-1/D} = \mathbf{a}_t - \frac{1}{D}v_\theta$ 反向积分, 这是因为 forward path $\mathbf{a}_t = (1-t)\mathbf{a}_{\mathrm{expert}} + t\epsilon$ 在 $t$ 增加时是从 expert 走到 noise, inference 要倒着走。这里 $D=5$ 步就够, 这是 Flow-Matching 相对 diffusion 在 robot control 上的关键优势 — 推理步数少, 对 50Hz control loop 友好。

### 3.3 Residual policy (公式 6)

$$\mathbf{a}_{\mathrm{total}} = \mathbf{a}_{\mathrm{flow}} + s\cdot\mathrm{clip}(\Delta\mathbf{a}, \pm c)$$

- $c=0.5$: residual clip 阈值 (joint-position offset 上界)
- $s=0.2$: correction scale, 把 residual 的有效幅度限制在 $\pm 0.1$
- $\mathrm{clip}$ 按 element-wise 算

设计直觉: flow policy 在 simulation 里已经把 tracking 解得很好, 长 residual PPO 训练会 drift away 掉 distilled behavior。Residual 应该是 lightweight local correction, 主要修 robustness 和小 residual gap。

### 3.4 Dynamic-Aware Motion Curation (公式 7-11): 在 fixed budget 下选更好的 data

这个 module 解决的问题是: scaling law 之外, 在 fixed data budget 下能不能选到更优的子集?

第一步, 按 complexity 分 bin 防止选样集中在简单或困难 clip:
$$c(m) := \mathrm{clip}(\mathbf{w}_c^\top\mathbf{u}^{\mathrm{cmp}}(m), 0, 1)$$

- $m$: motion clip
- $\mathbf{u}^{\mathrm{cmp}}(m)$: normalized features, 涵盖 root motion, height variation, joint motion range
- $\mathbf{w}_c$: 权重向量
- $c(m)\in[0,1]$: complexity score

在每个 bin 内, 用三个互补的分数排序:

**Physical feasibility**:
$$p(m) = \exp(-\mathbf{w}_p^\top\mathbf{v}^{\mathrm{phys}}(m))$$
$\mathbf{v}^{\mathrm{phys}}(m)$ 是 normalized violations in joint dynamics, root motion, base tilt, foot sliding, contact consistency。Violation 越大 $p$ 越接近 0。

**Rollout executability**:
$$r(m) = s(m)\exp(-\bar{e}(m))$$
$s(m)$: fixed-policy success rate; $\bar{e}(m)$: normalized tracking error。这是 closed-loop 可执行性指标。

**Flow confidence**:
$$f(m) = \exp(-\gamma_f\sigma_{\mathrm{flow}}(m))$$
$\sigma_{\mathrm{flow}}(m)$: 同一 observation 多次采样的 action variance。variance 越小, flow policy 对这个 clip 越有把握。

最终 PRF score:
$$S_{\mathrm{cur}}(m) = \lambda_p p(m) + \lambda_r r(m) + \lambda_f f(m),\quad \lambda_p+\lambda_r+\lambda_f=1$$

权重 $\lambda_p=0.45, \lambda_r=0.35, \lambda_f=0.20$ 偏向 physical feasibility。这个 PRF 分数和 LLM data curation 里用 complexity + quality + diversity 选数据是同构思路。

### 3.5 MCRC: Manifold-Calibrated Reference Conditioning (公式 12-16)

这是 motion tracking 部分最 novel 的 contribution。问题是: frame-level tracking command 只给当前帧的 target, 没有告诉 policy "这条 reference 在 learned motion manifold 上处在什么位置"。比如同一个 stand-to-sit 当前帧, 可能后面接 stand-up, 也可能接 roll-over, 两种情况需要的预备动作不同。

**VAE 训练**:
对 motion $m$ 在时间 $t$ 取长度 $H$ 的 reference window $\mathbf{x}_{m,t:t+H}$, 拼接 normalized kinematic features (joint pose, root-relative body pose, body velocity)。Encoder 输出 Gaussian posterior $q_\psi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \mathrm{diag}(\boldsymbol{\sigma}^2))$:

$$\mathcal{L}_{\mathrm{VAE}} = \|\hat{\mathbf{x}}_{m,t:t+H} - \mathbf{x}_{m,t:t+H}\|_2^2 + \beta D_{\mathrm{KL}}(q_\psi(\mathbf{z}|\mathbf{x})\|\mathcal{N}(0, I))$$

- $\hat{\mathbf{x}}$: decoder 重建
- $\beta$: KL 重量
- 下标 $m, t:t+H$ 表示 motion $m$ 从 $t$ 到 $t+H$ 的 window

**Manifold coordinate 作为 student observation**:
$$\mathbf{z}_t(m) = \boldsymbol{\mu}_{m,t}$$

取 posterior mean 作为 latent code。靠近 clip 末尾时 repeat 最后一帧填满 window。

**Reconstruction error 作为 reliability signal**:
$$e_{\mathrm{recon},t}(m) = \|\hat{\mathbf{x}}_{m,t:t+H} - \mathbf{x}_{m,t:t+H}\|_2^2$$

如果某个 window 在 manifold 上 reconstruction 误差大, 说明这个 reference segment 是 OOD 的, 可以作为 reliability cue。

最终 student observation:
$$\mathbf{o}_t^{\mathrm{student}} = [\mathbf{o}_t, \mathbf{z}_t(m)] \in \mathbb{R}^{69+32} = \mathbb{R}^{101}$$

- $\mathbf{o}_t$: 69 维 base observation = 24 维 reference command + 45 维 proprioception
- $\mathbf{z}_t(m)$: 32 维 VAE latent, 从 length-20 lookahead window 算出来

policy 输出 12 维 action, 转换为 joint position offset:
$$\mathbf{q}_t^{\mathrm{tar}} = \mathbf{q}_0 + \alpha\mathbf{a}_t,\qquad \alpha=0.25$$

$\mathbf{q}_0$ 是 nominal standing pose, $\alpha=0.25$ 把 action 范围压在合理 joint motion 之内。

### 3.6 实验结果解读

**Specialist-to-generalist baseline (Table 3, 1,000 motions)**:

| Method | Seen MPJPE | Seen Success | Unseen MPJPE | Unseen Success |
|---|---|---|---|---|
| Specialist (per-clip) | 11.66 | 97.34% | — | — |
| Multi-motion RL (single MLP) | 22.18 | 87.24% | 22.20 | 84.86% |
| Flow policy | 12.04 | 95.26% | 16.51 | 86.42% |
| Flow + Residual | 11.98 | 95.49% | 16.50 | 86.43% |

关键 take-aways:
1. Multi-motion RL from scratch 比 specialist 差近 2x (22.18 vs 11.66) — gradient interference 是真实的
2. Flow policy 拿到 specialist 大部分 fidelity (12.04 vs 11.66), 同时 unseen 上 16.51 mm vs multi-motion RL 的 22.20 mm, 改进显著
3. Residual RL 主要是 local refinement: seen 上微改进, unseen 几乎不变 — 印证了 paper 里 "residual should be lightweight" 的设计直觉

**Scaling law (Table 4, 30 → 7,076 motions)**:

| Samples | Seen MPJPE | Unseen MPJPE | Seen-Unseen MPJPE Gap | Unseen Success |
|---|---|---|---|---|
| 30 | 14.44 | 24.61 | 10.17 | 84.30% |
| 100 | 11.70 | 20.27 | 8.58 | 85.26% |
| 300 | 12.15 | 18.84 | 6.70 | 85.90% |
| 1,000 | 12.04 | 16.51 | 4.47 | 86.42% |
| 3,000 | 11.78 | 15.15 | 3.37 | 88.22% |
| **full (7,076)** | 12.38 | **14.79** | **2.41** | **88.54%** |

unseen MPJPE 单调下降 (24.61 → 14.79), seen-unseen gap 单调缩小 (10.17 → 2.41)。这是 quadruped motion tracking 的第一个 explicit scaling law, 跟 humanoid 领域 Sonic / GMT 看到的趋势一致。注意 seen MPJPE 在 full-data 下反而比 100-motion 时略差 (12.38 vs 11.70), 这是典型的 underfitting-vs-generalization tradeoff — 数据规模大了, 模型容量没相应放大的话, seen 上 fit 能力下降但 unseen 泛化提升。这点可以直接联想到 Chinchilla / LLM scaling: 模型 capacity 需要跟 data 同步 scale 才能保持 seen 上的 fit。

**PRF curation ablation (Table 5, 70% budget)**:

| Strategy | Seen MPJPE / Succ | Unseen MPJPE / Succ |
|---|---|---|
| random baseline | 14.01 / 88.47% | 16.04 / 85.53% |
| physical-only $p(m)$ | 13.02 / 91.18% | 15.77 / 85.55% |
| rollout-only $r(m)$ | 11.33 / 94.73% | 15.66 / 85.02% |
| flow-conf-only $f(m)$ | 11.99 / 94.03% | 15.87 / 85.57% |
| **PRF** | **11.17 / 94.81%** | **15.56 / 85.66%** |

在 fixed budget 下 PRF 在 seen 和 unseen 上都最好, 说明三个 signal 是互补的 — physical feasibility 偏静态可执行性, rollout executability 偏 closed-loop, flow confidence 偏 model 自身的 epistemic uncertainty。

**MCRC ablation (Table 6)**:

| Student Observation | Seen MPJPE / Succ | Unseen MPJPE / Succ |
|---|---|---|
| $\mathbf{o}_{69}$ (base) | 12.38 / 92.74% | 14.79 / 88.54% |
| $\mathbf{o}_{69} \oplus \mathbf{e}_{\mathrm{recon}}$ | 12.24 / 93.48% | 13.98 / 89.42% |
| $\mathbf{o}_{69} \oplus \mathbf{z}$ | **11.77 / 94.16%** | **12.53 / 91.02%** |
| $\mathbf{o}_{69} \oplus \mathbf{z} \oplus \mathbf{e}_{\mathrm{recon}}$ | 11.91 / 94.18% | 12.86 / 90.76% |

$\mathbf{z}$ 单独加最有效, unseen MPJPE 从 14.79 砍到 12.53 mm, unseen success 从 88.54% 拉到 91.02%。把 reconstruction error 也加上反而略差 — 说明 reliability cue 在已经 manifold-conditioned 的情况下是 redundant 的, 还会引入一点 noise。

参考 humanoid motion tracking 工作:
- GMT (General Motion Tracking): https://arxiv.org/abs/2506.14770
- Expressive Whole-Body Control (Cheng et al.): https://arxiv.org/abs/2402.16796
- BeyondMimic: https://arxiv.org/abs/2511.07820 (实际 Sonic, 这里可能是 typo)
- From experts to a generalist: https://arxiv.org/abs/2506.12779
- Track any motions under any disturbances: https://arxiv.org/abs/2509.13833

---

## 4. Locomotion: 三层 progressive stack

Locomotion 这部分 paper 拆成三段递进: robust baseline → biomimetic gait → all-terrain。这种 progressive 设计在 RL locomotion 文献里比较少见, 通常会一锅炖。这里分开训的好处是每个 sub-policy 可以独立优化自己的 reward, 避免 reward 之间互相打架。

### 4.1 Robust Baseline (公式 19 + Barlow Twins loss)

三个机制叠在一起:

**Barlow Twins-style temporal consistency**:
从 10-step proprioception buffer 取两个相邻 5-step views (latest history window + one-step-shifted window)。Shared encoder + projector 输出 feature, cross-correlation matrix $C$ 正则化到 identity:

$$\mathcal{L}_{\mathrm{BT}} = \sum_i(1 - C_{ii})^2 + \lambda_{\mathrm{off}}\sum_{i\neq j}C_{ij}^2,\qquad \lambda_{\mathrm{off}}=5\times 10^{-3}$$

- $C_{ii}$: 第 $i$ 个 feature dimension 的自相关 (对角项)
- $C_{ij}$: 不同 dimension 之间的互相关 (off-diagonal)
- 第一项让每个 dimension 信息充分 (variance), 第二项压制冗余
- $\lambda_{\mathrm{off}}$: off-diagonal 权重, 5e-3 是很小的值, 不让 redundancy term 过分主导

这个 trick 的好处是不需要 negative sample, 比 contrastive learning 简单。Actor mask 掉 privileged base linear velocity, 这等于让模型从 proprioception 历史里推断 base velocity — 是 implicit system identification。

**Explicit state estimation**:
在 representation layer 挂 regression head 估 base velocity, payload mass, CoM offset。配合 gravity-gated reward (只在 base 接近水平时给 reward), 这等于让 robot online 感知自己的物理 state, 自适应调节 stiffness。

**NP3O (Normalized Penalized PPO, 公式 19)** — 这是最 deployment-critical 的设计:

$$L_{\mathrm{viol}} = \sum_{i=1}^3 \lambda_i^{(t)}\max\Bigl(0, C_{\mathrm{surr}}^{(i)} + \tilde{v}_i\Bigr)$$

- $i\in\{1,2,3\}$: 三个 constraint — joint position, joint velocity, output torque
- $C_{\mathrm{surr}}^{(i)}$: 第 $i$ 个 constraint 的 surrogate cost, 用 normalized cost advantage 评估
- $\tilde{v}_i$: constraint margin (允许的 violation budget, 通常负值表示允许一定 slack)
- $\lambda_i^{(t)}$: exponentially increasing penalty coefficient, 训练越往后 violation 罚得越重
- $\max(\cdot, 0)$ + ReLU gating: 当 $C_{\mathrm{surr}}^{(i)} + \tilde{v}_i \le 0$ (即还在 budget 内) 时梯度为 0, 不阻碍 exploration; 超出 budget 后才施加 penalty

这是 CMDP (Constrained MDP) 的 Lagrangian 解法的一种变体。Table 8 在 3.0 m/s 高速 sprint 下, vanilla PPO torque violation 124.8 次, penalty-based PPO 17.2 次, NP3O 0 次 — 这是真实的 hardware safety guarantee, 对 deployment 很关键。

参考:
- Barlow Twins: https://arxiv.org/abs/2106.04956
- PPO: https://arxiv.org/abs/1707.06347
- DreamWaQ++ (perceptive quadruped): https://ieeexplore.ieee.org/document/10804919
- Walk These Ways (multiplicity of behavior): https://proceedings.mlr.press/v205/margolis23a.html

### 4.2 Diff-CAST: Biomimetic Gait with Omnidirectional Ability

AMP (Adversarial Motion Prior) 是 locomotion imitation 的事实标准, 但在 quadruped 上有三个 well-known failure mode:
1. Mode collapse on heterogeneous dataset
2. Unbounded adversarial reward 不稳定 PPO value network
3. Forward-biased data 让 lateral/backward command drift

Diff-CAST 用三招同时解:

**(a) Action-Agnostic Diffusion Prior (CC-Diffusion, 公式 20-22)**:
不判别 (state, action) pair, 直接建模 state transition $\mathbf{x}_t = (\mathbf{s}_t, \mathbf{s}_{t+1})$。这把 stylistic learning 从 actuator domain 解耦出来 — 你拿 animal MoCap 来训练 diffusion model 时不需要 torque data, 因为 action 不进 model。这个 idea 跟 humanoid 领域 "用 motion generative model 当 reward" 的思路接近。

conditional denoising MSE:
$$L^+(\mathbf{x}_t) = \mathbb{E}_{k,\varepsilon}[\|\varepsilon - \varepsilon_\varphi(\mathbf{x}_{t,k}, c^+, k)\|^2]$$
$$L^-(\mathbf{x}_t) = \mathbb{E}_{k,\varepsilon}[\|\varepsilon - \varepsilon_\varphi(\mathbf{x}_{t,k}, c^-, k)\|^2]$$

- $\mathbf{x}_{t,k}$: 加噪 step $k$ 的 transition
- $\varepsilon$: noise
- $\varepsilon_\varphi$: diffusion noise predictor, condition 在 concept $c\in\{c^+, c^-\}$ (expert vs agent) 和 velocity command $\mathbf{v}^{\mathrm{cmd}}$
- $c^+$: expert hypothesis, $c^-$: agent hypothesis
- 上标 $+/-$ 指两个 concept label

classification probability 用 softmax 形式导出:
$$D_\varphi(\mathbf{x}_t) = \frac{\exp(-L^+(\mathbf{x}_t))}{\exp(-L^+(\mathbf{x}_t)) + \exp(-L^-(\mathbf{x}_t))}$$

这个推导很有意思: 把 diffusion ELBO 当成 likelihood 估计, 再用贝叶斯比率得到分类概率, 类似把 score-based generative model 当 density estimator 用。这跟 recent 用 diffusion 当 reward 的方法 (DDRG, Diffusion Reward) 是一个 family。

**(b) Bounded Stylistic Reward (公式 23)**:
$$r_{\mathrm{diff}} = D_\varphi(\mathbf{x}_t)\in[0, 1]$$

直接用分类概率当 reward, 不取 $\log D - \log(1-D)$ 的 likelihood ratio。后者在早期 exploration 时容易出 spike (D 接近 0 时 reward → −∞), 不稳定 PPO value network。bounded reward 保留 mode-seeking gradient 但消灭 spike, 这是相当 elegant 的工程选择。

**(c) SACC: Symmetric Augmented Command Construction (公式 24)**:
Kinematic symmetry loss:
$$L_{\mathrm{sym}} = \lambda\Bigl(\|\pi_\theta(\mathbf{s}_t, \mathbf{v}^{\mathrm{cmd}}) - M_a(\pi_\theta(\tilde{\mathbf{s}}_t, \tilde{\mathbf{v}}^{\mathrm{cmd}}))\|^2 + \|V_\Psi(\mathbf{s}_t, \mathbf{v}^{\mathrm{cmd}}) - V_\Psi(\tilde{\mathbf{s}}_t, \tilde{\mathbf{v}}^{\mathrm{cmd}})\|^2\Bigr)$$

- $M(\cdot)$: sagittal-plane mirror operator, swap contralateral legs + negate lateral velocity, roll, yaw rate
- $\tilde{\mathbf{s}}_t = M_s(\mathbf{s}_t)$, $\tilde{\mathbf{v}}^{\mathrm{cmd}} = M_v(\mathbf{v}^{\mathrm{cmd}})$: mirrored state and command
- $M_a(\cdot)$: 对应的 action mirror, 让左右腿 action 对称
- $\pi_\theta, V_\Psi$: actor 和 critic
- $\lambda$: symmetry loss 权重

再加 yaw invariance: 把 heading-dependent planar features 在 diffusion update 时随机 rotate $\delta\sim U(-\pi, \pi)$, 合成 360° 增强数据。

**实验结果 (Table 9, 10)**:
- FGD (Frechet Gait Distance) 从 vanilla AMP 的 4173.66 降到 489.13 (Diff-CAST)
- Forward walk position deviation 从 25.03 m (w/o SACC) 降到 1.08 m (Diff-CAST), heading drift 2.07 → 0.13 rad
- Pure backward command: baseline fail (OOD), Diff-CAST 0.21 m deviation

w/o SACC 的 FGD 反而更低 (348.55), 但是 forward deviation 25.03 m — 说明它过拟合 forward-biased data, gait 看上去更 "expert-like" 但完全牺牲了 command tracking。这是 naturalness vs precision 的 trade-off, SACC 在 feature level 解耦了二者。

### 4.3 All-Terrain Locomotion: 三阶段 privileged-to-perceptive

三阶段 framework:
1. **Privileged teacher training**: clean proprioception + height map + base velocity + contact + dynamics (friction, mass, CoM, push, joint stiffness/damping)
2. **Clean LiDAR memory distillation**: frozen teacher, 只训 LiDAR memory encoder + auxiliary heads。Student input: 8-frame LiDAR memory, 每帧 voxelized 成 body-frame 3D occupancy grid + scan-age info。Temporal encoder = per-frame CNN + GRU + ego-motion compensation
3. **Noisy on-policy PPO fine-tune**: student 用自己的 action rollout, 加 LiDAR domain randomization (sensor noise, point dropout, holes, pose perturbation, scan delay), teacher regularization (BC, KL, latent matching) 逐渐 decay

这个三阶段 pipeline 跟 ETH 哎 Hutter group 的 perceptive locomotion (DreamWaQ, Miki et al. Science Robotics) 一脉相承, 但加了 temporal LiDAR memory 和 terrain-predictive supervision 两个 novel 点。

**Terrain curriculum (Table 11)**:
0-9 level 递增 difficulty, 涵盖 rough flat / obstacles / slope (10°→35°) / stairs (10-step, 0.10-0.30m 高) / platform (0.10-0.60m) / gap (0.10-0.60m 宽)。

**Ablation (Table 12)**:
- Proprioception-only: 28.0% success, max level 2.2
- Full method: 83.2% success, max level 7.8
- w/o Memory: 72.4% success (memory 贡献 ~11 pp)
- w/o Ego-motion Compensation: Bad Impulse 从 18.5 暴涨到 43.0 N·s (最大 ablation 退化)
- w/o Terrain Reconstruction: 68.6% success, Unsafe Foothold 22%

ego-motion compensation 是最关键的 component — 没有 it, historical LiDAR fusion 因为 robot 自身运动 spatial misalignment, 直接导致更大 impulse 冲击。这个 ablation 是非常直觉合理的: LiDAR scan 之间 robot 移动了, 不补偿的话 grid alignment 错位。

参考:
- Miki et al. Science Robotics (learning robust perceptive locomotion): https://www.science.org/doi/10.1126/scirobotics.abk2822
- Extreme Parkour (Pathak): https://arxiv.org/abs/2309.14341
- Robot Parkour Learning: https://arxiv.org/abs/2309.13631
- ASE (Adversarial Skill Embeddings): https://arxiv.org/abs/2205.01906
- AMP (Adversarial Motion Prior): https://arxiv.org/abs/2104.02180
- DreamWaQ: https://arxiv.org/abs/2309.14341

---

## 5. Scene Interaction: Hand-shaking 作为 case study

这个 case study 很值得讲, 因为它体现了 paper 整体的设计哲学 — **不学 monolithic end-to-end policy, 而是复用已有 module 拼起来**。

### 5.1 Pipeline (Figure 4)

Perception (hand landmark + depth + odometry) → Goal-directed Locomotion → IK Reference Generation → Motion Tracking 执行 → Adaptive Compliant Interaction

### 5.2 Goal-directed locomotion command (公式 25)

$$\mathbf{u}_{\mathrm{loc}} = \Bigl[k_{xy}(x_h - d_x),\; k_{xy}y_h,\; k_\psi\mathrm{atan2}(y_h, \max(x_h, \epsilon))\Bigr]^\top$$

- $\mathbf{p}_h^B = (x_h, y_h, z_h)$: 检测到的 hand position 在 robot base frame
- $d_x$: desired forward standoff (停下时距人手的距离)
- $k_{xy}, k_\psi$: 位置和 yaw 比例增益
- $\epsilon$: 防止 atan2 在 $x_h\to 0$ 时奇异
- 输出 $(v_x, v_y, \dot\psi)$ velocity command

这个 controller 接近 robot 的时候 yaw 朝向手, $x_h$ 接近 $d_x$ 时停下。

### 5.3 IK Reference Generation (公式 26)

这是 whole-body IK optimization:

$$\mathbf{q}_{1:T}^{\mathrm{IK}} = \arg\min_{\mathbf{q}_{1:T}} \sum_{t=1}^T \Bigl(\|W_F(p_F(\mathbf{q}_t) - \mathbf{p}_h^B)\|_2^2 + \sum_{i\in S} w_i^2 \|p_i(\mathbf{q}_t) - p_i^0\|_2^2 + \|W_q(\mathbf{q}_t - \mathbf{q}^0)\|_2^2\Bigr)$$

- $\mathbf{q}_{1:T}$: 整条 trajectory 的 joint angle 序列
- $p_F(\mathbf{q}_t)$: 当前 joint configuration 下 active front-foot 的 endpoint position
- $W_F$: active foot 的 weight matrix
- $S$: support feet + base anchor sites (应该保持 nominal 的部位)
- $p_i^0$: 第 $i$ 个 support site 的 nominal position
- $w_i$: 第 $i$ 个 support site 的 weight
- $\mathbf{q}^0$: nominal standing posture
- $W_q$: posture regularization weight

三项: (1) active foot 追 hand target, (2) support feet + base 不动, (3) 整体 posture 不要离 standing 太远。生成 reaching trajectory (stand → hand-shake → stand), 喂给 motion tracking policy 训练。

### 5.4 Adaptive interaction (公式 27)

握住手时, 把 active thigh 和 calf 的 PD gain 降到最低, 让 robot 顺从人的运动。完成检测用 proprioceptive external-stimulus score:

$$s(t) = \max_{j\in\mathcal{A}} |\dot q_j(t) - \dot q_j^{\mathrm{cmd}}(t)|$$

- $\mathcal{A}$: active thigh + calf 关节集合
- $\dot q_j(t)$: 第 $j$ 关节实际速度
- $\dot q_j^{\mathrm{cmd}}(t)$: 第 $j$ 关节 command 速度
- 二者差大说明有外力作用

warm-up 后, sustained above-threshold stimulus 标记 "engaged"; 之后 quiet interval 触发 return segment。Timeout + safety guard 全程开着。

### 5.5 实验结果 (Table 13)

| Error | Mean | Median | P90 |
|---|---|---|---|
| $e_{\mathrm{IK}}$ | 11.8 mm | 9.8 mm | 24.6 mm |
| $e_{\mathrm{track},C}$ (active calf body) | 52.9 mm | 34.4 mm | 125.5 mm |
| $e_{\mathrm{track},B}$ (base anchor) | 59.9 mm | 53.6 mm | 106.0 mm |
| $e_{\mathrm{exec}}$ (final foot-to-target) | 137.4 mm | 121.5 mm | 193.1 mm |

最终 mean error 13.74 cm, 这个数字看起来大, 但 hand-shaking 是 compliant 互动, robot 在握手时是顺从人的 motion 而不是精确 hold 一个点 — 这个误差其实是动态握手过程中的偏差, 不是静态定位精度。IK reference 本身 mean 1.18 cm 很准, tracking 引入 ~5 cm, 整个 compliant execution 累积到 13.7 cm, 这些误差是合理叠加的。

参考:
- Hand reaching for HRI (Prasad et al.): https://arxiv.org/abs/2103.13422
- Non-verbal interaction with quadruped: https://arxiv.org/abs/2408.08393
- Hierarchical loco-manipulation: https://ieeexplore.ieee.org/document/10160523
- Visual whole-body control for legged loco-manipulation: https://arxiv.org/abs/2402.17640

---

## 6. Unified Deployment System

### 6.1 硬件 (Tutu 平台)

- 3× SENSING cameras, 120°×90° FOV (前 / 左 / 右)
- 96-channel RoboSense Airy LiDAR
- AISpeech microphone-speaker module
- NVIDIA Jetson AGX Orin 64GB
- Cloud offload for heavy reasoning

### 6.2 Runtime Stack (Figure 5b)

分层设计:
- **Locomotion runner**: 常驻进程, 接 velocity command
- **Motion-tracking runner**: 常驻进程, 执行 reference motion library / motion generation model / IK trajectory
- **Coordination layer**: 决定哪个 runner 输出 active, blend motor target, 处理 safety gating

低层 motor command 200 Hz, decision / inference 50 Hz。Position-control policies 输出 target joint position, low-level PD:
$$\boldsymbol{\tau}_t = \mathbf{k}_p \odot (\mathbf{q}_t^{\mathrm{tar}} - \mathbf{q}_t) - \mathbf{k}_d \odot \dot{\mathbf{q}}_t$$

- $\boldsymbol{\tau}_t$: 关节 torque
- $\mathbf{k}_p, \mathbf{k}_d$: per-joint PD gain (element-wise 乘 $\odot$)
- $\mathbf{q}_t^{\mathrm{tar}}$: target joint position
- $\mathbf{q}_t, \dot{\mathbf{q}}_t$: 当前 joint position 和 velocity

MuJoCo 和 hardware 共用 robot-state 和 motor-command 抽象, 可以 sim-to-sim verify 再 sim-to-real。

这个设计的核心 insight: locomotion 和 motion-tracking runner 常驻, 切换时不用 reload policy 也不用 rebuild observation buffer, 显著降低 interaction latency。这是 deployment 工程上很务实的取舍。

---

## 7. 下游 Applications

### 7.1 Interactive Companionship (Figure 8)

Cloud-side reasoning + onboard control 分离。Agent 接 multimodal input (audio + video), 重型 reasoning offload 到 cloud, 翻译成 compact robot command 发给 onboard stack:
- Goal-directed 指令 ("come here") → locomotion policy
- Expressive request → text-to-motion 模块生成 reference trajectory → motion-tracking policy

可以 real-time interrupt, robot 安全 recover 到 stable standing 再接下一条 command。三个 demo: approach-and-handshake / gesture-aware response / exercise companionship (一起做单臂俯卧撑)。

### 7.2 All-Terrain Autonomous Navigation (Figure 9)

VLN-based planner 生成 waypoint sequence, local obstacle-avoidance 转 velocity command, control system 基于 3D point-cloud 实时 adapts gait。Figure 9 展示了不规则 outdoor slope、auto escalator、不规则楼梯上下, 都成功 traverse。

参考 VLN/VLA 工作:
- PaLM-E: https://arxiv.org/abs/2303.03378
- RT-2: https://arxiv.org/abs/2307.15818
- OpenVLA: https://arxiv.org/abs/2406.09246
- π0.5: https://arxiv.org/abs/2504.16054
- SayCan (Grounding Language in Robotic Affordances): https://arxiv.org/abs/2204.01691
- CoMaTrack (他们的 VLA tracking paper): https://arxiv.org/abs/2603.22846

---

## 8. 整体直觉与 Karpathy-视角的几个观察

### 8.1 Scaling law 在 quadruped 上首次 explicit 验证

Table 4 是 paper 最关键的实验。30 → 7,076 motions, unseen MPJPE 24.61 → 14.79 mm, unseen success 84.30% → 88.54%, seen-unseen gap 10.17 → 2.41 mm。这条曲线在 humanoid 上 Sonic / GMT 已经看到, 现在 quadruped 也 reproduce 出来。但 paper 自己也指出 seen MPJPE 在 full-data 下比 100-motion 时略差 — 这暗示 model capacity 没跟上 data scale, 下一步应该是同步把 transformer layers 或 latent dim 拉大, 看看 seen fidelity 能不能 hold 住。这是非常 LLM-scaling-style 的 open question。

### 8.2 Specialist → Flow Policy 蒸馏的合理性

Specialist 训练拿 seen 11.66 mm / 97.34% success, 这是 single-clip upper bound。直接 multi-motion RL 训一个 MLP 只有 22.18 mm — 这是 gradient interference 的代价。Flow-Matching distillation 把它拉回到 12.04 mm, 几乎追上 specialist, 同时拿到 generalization。这是非常清晰的 "specialist 训练 + 蒸馏到 generalist" 范式的胜利, 跟 humanoid OmniH2O / GMT / ExBody2 的方法论完全一致。

为什么是 Flow-Matching 而不是 Diffusion? 几个关键原因:
1. **Inference speed**: $D=5$ ODE steps 足够, 对 50 Hz control loop 友好; diffusion 通常要 20-1000 步
2. **Multimodal action distribution**: 不同 motion clip 的 expert action 形成多模态分布, Flow-Matching 用 velocity field 表达比 GMM 更灵活
3. **Optimal transport 性质**: linear interpolation path $\mathbf{a}_t = (1-t)\mathbf{a}_{\mathrm{expert}} + t\epsilon$ 是 OT 解, 比 diffusion 的 forward/reverse 加噪更直接

### 8.3 MCRC 的 intuition: lookahead context 通过 VAE latent 进入 policy

这个 idea 很 deep。当前帧 reference 只告诉 policy "现在要做什么动作", 但不告诉 "后面 0.4 秒要做什么"。Manifold latent $\mathbf{z}_t$ 是 20-frame lookahead window 的 VAE posterior mean, 等于把未来信息压成 32 维 condition 注入 policy。这跟 LLM 里的 prefix/prompt conditioning 是同构的 — 都是给 generative model 一个 context 来约束输出分布。

Ablation 也支持这个直觉: 单加 $\mathbf{z}$ 比 $\mathbf{e}_{\mathrm{recon}}$ 强很多, 说明 manifold coordinate (where on the manifold) 比 reliability cue (how well-explained) 信息量更大。$\mathbf{z} + \mathbf{e}_{\mathrm{recon}}$ 略弱于 $\mathbf{z}$ alone, 说明 reliability cue 反而引入了 noise, 因为已经 manifold-conditioned 的 policy 自己能 sense OOD。

### 8.4 Locomotion 三层 stack 的取舍

把 robust baseline / biomimetic gait / all-terrain 拆成三个 policy, 而不是一个 unified policy 全包, 这是个工程取舍:
- **优点**: 每个 sub-policy reward 独立, 避免 reward interference; 训练 stability 更好; 可以独立 iterate
- **缺点**: deployment 需要 arbitration / smooth blend, system complexity 上升; 不能 cross-task share representation

paper 在 Section 7 limitations 里也明确承认这点: "current system is still a coordinated multi-policy stack rather than a single unified model"。这跟 humanoid 领域向 unified BFM 演进的方向 (BFM-Zero, π0.5, Sonic) 是相反的。下一步应该是把三个 policy 统一到一个 transformer-based policy, 用 task token / instruction conditioning 切换 mode。

### 8.5 NP3O 是 hardware safety 的关键工程

Table 8 那个 0 torque violation 的数字, 在真实部署里价值巨大。Vanilla PPO 在 3 m/s sprint 下 124.8 次 torque violation, 55% fall rate; penalty-based PPO 17.2 次但仍 15% fall; NP3O 全部 0。这意味着 robot 在 aggressive maneuver 下不会烧 motor, 不会失控。这种 hard constraint guarantee 是 deployment 必须的。

### 8.6 与 humanoid BFM 的对照

| 维度 | Humanoid BFM (Sonic / BFM-Zero / OmniH2O) | ABot-C0 (quadruped) |
|---|---|---|
| Data 来源 | AMASS / SMPL 重建, 天然丰富 | 4-source pyramid, video generation 是主 lever |
| Retargeting | 同构 bipedal, 直接 | cross-embodiment fragile, 需自建 I2V pipeline |
| Scaling law | 已验证 | **首次验证** |
| Policy 架构 | Diffusion / Flow-Matching / MLP | Flow-Matching + VAE manifold condition |
| Locomotion | 全身协调, 复杂 | flat / biomimetic / all-terrain 三层 |
| Interaction | arm manipulation | leg reaching + compliant hand-shake |
| Deployment | unified policy 为主 | multi-policy stack + coordination layer |

ABot-C0 在 scaling law 上对齐 humanoid, 在 architecture 上略 conservative (multi-policy), 在 data pipeline 上更 aggressive (video generation pipeline)。这种 "data 闯, policy 稳" 的策略对 quadruped 这种 data-sparse 形态是合理的。

---

## 9. Limitations & 未来方向 (paper 自陈 + 我的推测)

1. **Multi-policy stack 而非 unified model**: 跟 humanoid BFM 演进方向相反。未来应该 unified transformer 架构, condition 在 task intent + motion reference + terrain + interaction target。
2. **Brain-body interface**: 低频 cloud reasoning + 高频 onboard control 的分离是 practical 但留下 command abstraction / latency / uncertainty / safety / recovery 的 open problem。这跟 autonomous driving 的 cloud-edge 分层有类似 design tension。
3. **Cross-platform evaluation**: 现在 Tutu 一个平台, 需要验证到更多 quadruped morphologies。
4. **Contact-rich interaction**: hand-shake 只是 case study, 需要扩展到更多 contact 行为 (push, lift, collaborative carry)。
5. **Self-improving loop**: paper 提到 future direction — robot 自主 collect real-world experience, identify failures, expand motion repertoire, refine policies (continual skill discovery on hardware)。这是 ENPIRE / agentic skill discovery 路线。

参考未来方向工作:
- ENPIRE (agentic self-improvement): https://arxiv.org/abs/2606.19980
- Agentic Skill Discovery (Zhao et al.): https://www.sciencedirect.com/science/article/pii/S0921889025000334
- TWIST2 (humanoid data collection): https://arxiv.org/abs/2511.02832
- HumanPlus: https://arxiv.org/abs/2406.10454
- OmniH2O: https://arxiv.org/abs/2406.08858

---

## 10. 总结

ABot-C0 是 quadruped behavior foundation model 的 system-level step, 不是 single algorithm paper。它的核心贡献:

1. **Data**: 用 video generation (Wan2.2 + identity consistency loss) + multi-stage filter 造出 16k clips 的 quadruped motion library, 是目前最大规模的 quadruped motion 数据集 (对比 DogML 11 小时, ABot-C0 22.43 小时)
2. **Policy**: 首次把 Flow-Matching + specialist-to-generalist distillation 用在 quadruped motion tracking, 验证 scaling law (unseen MPJPE 24.61 → 14.79 mm)
3. **Reference conditioning**: MCRC 用 VAE manifold latent 作为 lookahead context, unseen success 88.54% → 91.02%
4. **Locomotion**: 三层 progressive stack (robust baseline + biomimetic Diff-CAST + all-terrain privileged-to-perceptive)
5. **Deployment**: multi-policy coordination + NP3O hardware safety + 200Hz motor / 50Hz decision 分层 runtime

对 Karpathy 视角最值得 internalize 的几点:
- **Specialist → Generalist + Flow-Matching 蒸馏是 scalable imitation 的有效范式**, gradient interference 的代价被 distillation 显式买单
- **Data scaling law 在 robot motion 上是 real 的**, 不只是 LLM 现象
- **Video generation 作为 data engine 是 quadruped 数据稀缺的破解路径**, humanoid 靠 SMPL 重建, quadruped 得自建 I2V + 3D extraction pipeline
- **Manifold latent conditioning 是 lookahead information bottleneck 的一种 elegant 实现**, 跟 prefix tuning / prompt 是同构思路
- **Multi-policy stack vs unified policy 的取舍**: 现在 quadruped 选择 multi-policy 是为了 stability, 但演进方向一定是 unified transformer architecture

希望这帮到你建立对 quadruped BFM 这条路线的 intuition。如果对某个 module 想更深入 (比如 CC-Diffusion 的 ELBO 推导、NP3O 的 Lagrangian dual、或 Flow-Matching 跟 DDIM/DDPM 的 inference cost 对比), 我可以再展开讲。
