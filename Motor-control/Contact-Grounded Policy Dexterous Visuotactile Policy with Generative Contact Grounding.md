---
source_pdf: Contact-Grounded Policy Dexterous Visuotactile Policy with Generative
  Contact Grounding.pdf
paper_sha256: ad4fc757599676f86d161e2814707da035194b1816130d47733319cd62d6d36f
processed_at: '2026-08-18T03:58:05-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Contact-Grounded Policy 人话版

Andrej，我用更接地气的方式再讲一遍，focus 在 "why this works" 的直觉上。

## 一句话说清楚这 paper 干嘛

**让 robot hand 在抓东西的时候，不光"看到"contact，还能"算出"手指应该往哪使劲，使得底层 controller 真能把那个 contact 给 realize 出来。**

就这么个事。但里头的门道挺深。

## 问题出在哪？一个直觉例子

想象你闭着眼睛拧 jar lid。你的 brain 在干吗？

你的 fingers 抓住 lid，皮肤告诉 brain "这里有 contact，pressure 多大，有没有 slip"。brain 接收到这些 tactile signal，然后发出 motor command："index finger 多用 0.5N，thumb 少用 0.3N，wrist 顺时针转 5 度"。

关键点来了：**brain 发的不是 "手指位置"，是 "肌肉 activation"**。这个 activation 经过 muscle 的 compliance（肌肉本身有弹性），最终产生 force，force 作用在 lid 上，lid 转动。

现在大部分 visuotactile policy 在干的事：**直接从 vision + tactile 预测下一步手指该去哪**。这就好比 brain 直接跳过肌肉力学，预测 "手指最终位置"。问题是你预测了位置，但 contact force 对不对你不知道，可能 slip，可能捏碎，可能根本没接触到。

CGP 的 insight：**别预测 finger 最终去哪，预测 contact 应该长什么样（tactile 会变成啥），然后反推出"我该给 controller 什么 setpoint，使得 controller 在 compliance 下能产生那个 contact"**。

## 核心 Insight：三个量的三角关系

这个我真的觉得是 paper 最漂亮的地方。作者发现了一个 simplification：

**你不用 explicitly model contact 在哪、是 sliding 还是 rolling、friction coefficient 多少。你只需要三个量就能 implicit 表示 contact：**

- **x**：手指现在实际在哪（actual state）
- **u**：tactile sensor 读到啥（contact force 的 proxy）
- **a**：你给 controller 的 target setpoint

为啥这三个够？Think about it：

Compliance controller 就是个 virtual spring。你给 target $a$，手指实际在 $x$，spring 就被压缩了 $(a - x)$ 这么多，产生的 force 跟这个差成正比。**当手指没碰东西，$x$ 很快追上 $a$，差接近零。当手指碰到了东西，东西挡着不让 $x$ 追上 $a$，差就 persist，这个差就是 contact force。**

同时 tactile sensor $u$ 告诉你这个 force 实际多大、在哪个 patch。

所以 $(x, u, a)$ 这个 triplet **完整刻画了 contact state**，不需要你去 explicitly 定义 contact point 在哪、normal 方向、friction cone 之类的。这些东西都 implicitly encoded 在 triplet 里了。

这个 observation 的 power 在于：**把一个 open-ended 的 contact modeling 问题，变成了一个 supervised regression 问题**。你只要有 demo data，就能学一个 mapping $a = M(x, u)$。

参考一下 compliance control 的背景：https://en.wikipedia.org/wiki/Compliance_(robotics)

## 三个组件各干啥（用人话）

### 1. Diffusion Model：当"预言家"

给它现在的 observation（vision + tactile + robot state 的 history），它预测 **未来 16 步 contact 会怎么演化**。

但注意，它预测的不是 "action"，是 **"未来手指会在哪 + 未来 tactile 会读到啥"** 这俩耦合的 trajectory。

为啥要预测 tactile？因为 tactile 是 contact 的直接 readout。你预测了 tactile，就预测了 contact pattern。预测了 robot state，就预测了 hand configuration。这俩一起，就是 "contact 的演化"。

为啥用 diffusion 不用 MSE regression？因为 contact-rich task 有 multimodality——同一个 observation，可能有好几种合理的下一步 contact evolution。Diffusion 能 model 这个 distribution，MSE 会 average 掉，产生不合理的中间态。

Diffusion policy 原版：https://arxiv.org/abs/2303.04137

训练 loss 就是标准 DDPM：

$$\mathcal{L} = \mathbb{E}\left[\|\epsilon - \pi_\theta(\mathbf{O}_t, \mathbf{Y}_t^j, j)\|^2\right]$$

- $\epsilon$：加的 Gaussian noise
- $\pi_\theta$：U-Net denoiser，预测噪声
- $\mathbf{O}_t$：observation history（conditioning）
- $\mathbf{Y}_t^j$：加噪后的 trajectory
- $j$：diffusion step

### 2. Contact-Consistency Mapping：当"翻译官"

这是 paper 的灵魂。Diffusion 预测的是 "future state + future tactile"，但 controller 要的是 "target setpoint"。这俩不一样！需要个翻译。

翻译的公式：

$$\mathbf{a}_t = \mathbf{x}_t + \mathcal{M}_\phi(\mathbf{x}_t, \mathbf{u}_t)$$

注意这里用 **residual**：target = actual state + 一个 small offset。

**为啥用 residual？** 直觉：大部分时候手指在 free space 动，target ≈ actual，offset 几乎是零。只有 contact 时 offset 才显著。如果你直接回归 absolute target，network 要同时学 "free space 大范围运动" 和 "contact 时小范围 force offset"，这俩 signal scale 差太多，learning 不稳定。Residual 把大信号 anchor 掉，network 只学小信号，稳定多了。

这就像 residual learning 在 ResNet 里的逻辑：学 identity 容易，学 deviation 难的话就只学 deviation。

ResNet 原文：https://arxiv.org/abs/1512.03385

### 3. VAE：当"压缩器"

Tactile data 维度太高。Simulation 里 748 个 sensing point × 3D force = 2244 维。Real robot 上 4 个 Digit360 sensor，每个 72×72×3 RGB image。直接在 raw space 做 16 步 diffusion，算力爆炸。

用 VAE 压成 32 维（sim）或 80 维（real，每个 sensor 20 维），然后 diffusion 在 latent space 生成。

**关键 trick：KL regularization。**

这个有个反直觉的发现：去掉 KL 后，VAE 重建更准（PSNR 高 5dB），但 **policy 性能反而下降**。

为啥？因为 diffusion 在 latent space 走，如果 latent space 有 hole、sharp boundary、non-smooth region，diffusion denoising 会走到 "invalid" 区域，生成出物理上不合理的 tactile pattern。KL 让 latent distribution 接近 $\mathcal{N}(0, I)$，整个 space smooth，diffusion 生成稳定。

这跟 Latent Diffusion 的 insight 完全一致：https://arxiv.org/abs/2112.10752

## Inference 时怎么跑

Receding horizon，loop 如下：

```
1. 看 observation history (过去 2 步)
2. Diffusion 采样未来 16 步的 (state, tactile latent)
3. 对每步用 M_φ 翻译成 target setpoint
4. 执行前 8 步 (controller 跟踪)
5. 回到 step 1，重新规划
```

频率 5Hz，每次 inference 8 个 DDIM step。Real-time 跑得动。

## 为什么这思路 work：一个 analogy

我觉得最贴切的 analogy 是 **driving a car with power steering**。

你（policy）想的是 "车要往哪走、轮胎要压到哪"（future trajectory + contact）。但你不能直接控制轮胎，你只能转方向盘（target setpoint）。方向盘到轮胎之间有 power steering（compliance controller），它有自己的 dynamics。

如果你瞎转方向盘，车可能走偏，因为 power steering 的响应有 lag、有 compliance。

聪明的做法：**先想清楚轮胎要去的轨迹和要产生的 ground contact，然后反推"我该给方向盘什么 input，经过 power steering 的 dynamics，能产生那个 tire behavior"**。

这就是 CGP 干的事。$\mathcal{M}_\phi$ 就是那个 "反推" 的 mapping，它 learned 了 power steering（compliance controller）的 inverse dynamics under contact。

## 实验：哪些结果最 informative

### 主结果 (Table II)

最 striking 的是 **Dish Wiping**：CGP 58.4% vs visuotactile DP 43.6%。

Dish Wiping 是 sustained contact，arm 主导的 force application。这说明 CGP 不光对 finger-level contact work，对 arm-level contact 也 work。Formulation 是 general 的。

Jar Opening 上 visuomotor (73.3%) 比 visuotactile (66.7%) 还高，说明 **naive 加 tactile 反而有害**。只有 CGP 这种 structured grounding 才能真正用上 tactile。

### Tactile Prediction vs Execution Alignment (Fig. 5)

这是我最喜欢的验证。作者做了一个 time-aligned comparison：t 时刻预测的 tactile $\hat{u}_{t+k}$，和 t+k 时刻实际观察到的 $u_{t+k}$，对比。

如果 align 得好，说明 **policy 预测的 contact 不只是 "看起来合理"，而是 controller 真能 execute 出来的**。这是和 "tactile as auxiliary prediction" 的本质区别——别的 method 预测 tactile 当 regularizer，但预测的跟执行的没关系；CGP 预测的 contact 必须 realizable，否则 mapping 转出来的 target 就是错的。

### Hand Configuration Prediction (Table III)

把 contact-consistency mapping 单独拎出来测，不涉及 policy learning。

- State only：MAE 16.05（没 tactile，miss contact-induced variation）
- Tactile only：MAE 35.93（tactile 是 local signal，resolve 不了 global config）
- State + Tactile：MAE 8.80（俩一起才行）

这直接证明了 triplet $(x, u, a)$ 的必要性，缺一不可。

Residual mode 比 absolute mode 低 30-40%，validate 了 residual formulation。

### KL Ablation (Fig. 7)

去掉 KL，reconstruction PSNR 高 5dB，但 policy success rate 掉好几个点。

**重建好不等于 downstream 好。** 这是 representation learning 的经典 lesson，在 tactile modality 上再次 confirm。

## 跟我（Karpathy）会关心的点

### 1. 这本质是学 inverse dynamics

Contact-consistency mapping $M_\phi(x, u) \to a$ 就是 compliance controller 的 **inverse dynamics under contact**。

Forward dynamics：$a \to$ (controller + environment) $\to (x, u)$
Inverse dynamics：$(x, u) \to a$

这个 inverse 在 free space 是 trivial（$a \approx x$），在 contact 下 non-trivial。Network 学的就是这个 non-trivial part。

这跟 model-based RL 里学 dynamics model然后做 MPC 的思路有点像，但这里学 inverse 而不是 forward，因为我们要的是 "给定 desired contact，该给什么 input"。

### 2. Factorization 很干净

Prediction 和 execution 解耦：
- Diffusion 只管 "contact 该怎么演化"，不用管 controller
- Mapping 只管 "怎么把 desired contact 翻译成 controller input"，不用管 temporal multimodality

这种 modular design 我觉得对 scaling 友好。可以想象换个更强的 predictor（比如 video diffusion 那种规模），mapping 不用改。

### 3. Latent Tactile Generation 是必须的

Tactile data 太 high-dim，不压缩没法做 long-horizon diffusion。但压缩后的 latent 必须 "diffusion-friendly"（smooth, no hole），所以 KL reg 关键。

这跟 LDM（Latent Diffusion Models）的 motivation 完全 parallel。LDM 压 image 到 latent 再 diffusion，CGP 压 tactile 到 latent 再 diffusion。Modality 不同，insight 相同。

LDM: https://arxiv.org/abs/2112.10752

### 4. 想到的几个 extension

**Active tactile sensing**：现在 tactile 是被动读。如果 policy 能主动做 small perturbation 来 "probe" friction coefficient，可能更 robust。人类就是这么干的——你拿不稳东西会下意识 squeeze 一下试试。

**Cross-embodiment**：现在 mapping 是 embodiment-specific。如果 condition on controller parameter（stiffness gain）和 robot kinematics，可能 transfer。Paper future work 提了。

**Hierarchical**：16 步 prediction 对 multi-stage task 不够。High-level planner 生成 subgoal sequence，low-level CGP 执行每个 subgoal。

**Tactile representation learning**：现在 VAE 是无监督压缩。如果用 contrastive learning 学一个有 physical meaning 的 latent（disentangle normal force / tangential force / contact location），可能更好。参考 TacSL: https://arxiv.org/abs/2408.06506

**VLM integration**：把 language condition 加进 diffusion，"gently pick up the egg" vs "firmly grasp the jar" 应该产生不同的 contact evolution。

## 一句话总结

**CGP 把 contact-rich manipulation 重新 formulate 成 "predict contact evolution + invert controller dynamics" 两个子问题，用 diffusion model 解第一个，用 learned mapping 解第二个，让 tactile prediction 真正 ground 到可执行的 control 上。**

这 paper 的 contribution 我觉得主要在 **problem formulation**，不在 network architecture。Architecture 都是现成的（diffusion policy + VAE + MLP），但把它们组合起来解决 "tactile-execution gap" 这个问题，是新的。

Andrej 你觉得这个 intuition 讲清楚了吗？如果哪个部分还想 deep dive，比如 FEM simulation 怎么实时跑、RelaxedIK retargeting 细节、或者 Digit360 sensor 信号特性，我可以再展开。

Project page: https://contact-grounded-policy.github.io/
Diffusion Policy: https://arxiv.org/abs/2303.04137
Latent Diffusion: https://arxiv.org/abs/2112.10752
Digit360: https://arxiv.org/abs/2411.02479

---

# Contact-Grounded Policy 深度解读

Andrej，这篇 paper 我读下来感觉很有意思，它抓到了 dexterous manipulation 里一个被很多人忽略但极其关键的问题：**tactile signal 如何真正 ground 到 controller 可执行的目标上**。下面我从 intuition 出发，逐层拆解。

## 1. 核心问题：为什么 tactile 当 observation 不够用

先 build 一下 problem space 的 intuition。Dexterous manipulation 的难点在于 contact 是 **distributed、time-varying、partially observable** 的。多个 finger 同时接触 object，contact point 不断迁移，friction state 会突变（stick-slip transition），这些都让 control 变得极其 sensitive。

现有的 paradigm 大致三类：
- **Grasp-centric pipelines**：先生成 grasp 配置，然后执行。问题是一旦 grasp 固定，finger motion 就被约束了，无法做 in-hand manipulation 这种需要持续 reconfiguration 的任务。
- **RL**：能发现 complex contact strategy，但 sim-to-real 困难，尤其 visual + tactile observation，reward engineering 成本高。
- **Imitation learning (visuomotor/viutactile policy)**：直接从 human demonstration 学。但大多数 visuomotor policy 只预测 kinematic trajectory，**没有 explicit contact semantics**。

这篇 paper 抓到的痛点是：**很多 visuotactile policy 把 tactile 当成 additional observation 喂进去，或者当成 auxiliary prediction target，但 tactile prediction 和 low-level controller 之间是 disconnected 的**。你预测了一堆 tactile pattern，但 controller 执行的 command 根本复现不出这些 pattern。结果是 slip、over-stiff interaction、unreliable execution。

这个观察我觉得很 sharp。本质上是一个 **representation-execution gap**：policy 输出的 action 和 controller 实际能 realize 的 contact 之间没有 consistency constraint。

Project page: https://contact-grounded-policy.github.io/

## 2. Key Insight：Contact as Triplet (x_t, u_t, a_t)

这是整篇 paper 最优雅的地方。与其显式建模 contact locations、contact modes、fixed contact points（这些 parameterization 在 multi-finger hand 上组合爆炸），作者提出：**在给定 tactile sensor 和 compliance controller 的前提下，contact 完全可以由 triplet (x_t, u_t, a_t) 隐式表示**。

- **x_t**：actual robot state（end-effector pose + hand joint angles）
- **u_t**：tactile feedback
- **a_t**：target robot state（compliance controller 的 reference）

为什么这个 triplet 足够？这里要 build 一个物理 intuition：

Compliance controller（PD 或 impedance）本质是一个 **virtual spring-damper**。它把 (a_t - x_t) 这个 tracking error 映射成 motor torque。当 finger 没接触时，x_t 会很快收敛到 a_t（steady-state error 很小，主要是 gravity/friction）。**但当 finger 接触 object 时，contact force 会阻止 x_t 到达 a_t，于是 a_t 和 x_t 之间产生 persistent gap**。这个 gap 的大小和方向，加上 u_t 的读数，就完整刻画了当前的 contact state。

Fig. 2 那个 3-DoF revolute finger 的 schematic 很直观：每个 joint 有一个 virtual spring-damper，target angle 和 actual angle 的差就是 spring deformation，对应 contact force。

这个 insight 的 power 在于：它把 contact grounding 从一个 **open-ended modeling problem** 变成了一个 **supervised regression problem**。你只需要学一个 mapping：

$$\mathbf{a}_t = \mathcal{M}_\phi(\mathbf{x}_t, \mathbf{u}_t)$$

变量解释：
- $\mathbf{a}_t \in \mathbb{R}^{(3+6+n_h)}$：target robot state，3 维 translation + 6 维 rot6D rotation + $n_h$ 维 hand joints
- $\mathbf{x}_t \in \mathbb{R}^{(3+4+n_h)}$：actual robot state，quaternion 表示 rotation
- $\mathbf{u}_t$：tactile feedback（array 或 image）
- $\mathcal{M}_\phi$：contact-consistency mapping，parameterized by $\phi$

注意 $\mathbf{a}_t$ 用 rot6D 而 $\mathbf{x}_t$ 用 quaternion，这是为了 action regression 的稳定性（rot6D 是 continuous representation，参考 Zhou et al. 2019 https://arxiv.org/abs/1812.07035）。

## 3. 方法架构：三大组件的分工

CGP 的 pipeline 分成三个耦合组件，每个都有明确的职责。我画一下 mental model：

```
Observation History O_t
    │
    ▼
┌─────────────────────────────────┐
│  Conditional Diffusion π_θ      │  ← 预测 future (x, h) 耦合轨迹
│  (U-Net denoiser)               │
└─────────────────────────────────┘
    │
    ▼  (X̂_t, ĥ_t)  future trajectories
┌─────────────────────────────────┐
│  Contact-Consistency Mapping M_φ│  ← 把 (x̂, û) 转成 executable a
│  (Residual MLP + tactile enc)   │
└─────────────────────────────────┘
    │
    ▼
  â_t  →  Compliance Controller  →  Robot
```

### 3.1 Conditional Diffusion Trajectory Generator π_θ

这部分 follow diffusion policy（Chi et al. 2023, https://arxiv.org/abs/2303.04137）的设计，但有一个关键区别：**它预测的不是 action，而是 actual robot state + tactile latent 的耦合轨迹**。

定义 future trajectory：
$$\mathbf{Y}_t = [\mathbf{x}_{t+1:t+T}, \ \mathbf{h}_{t+1:t+T}]$$

- $\mathbf{x}_{t+1:t+T}$：未来 $T$ 步的 actual robot state
- $\mathbf{h}_{t+1:t+T}$：未来 $T$ 步的 tactile latent（VAE 压缩后）
- $T$：prediction horizon，paper 里是 16

Diffusion 训练用标准 DDPM/DDIM（https://arxiv.org/abs/2006.11239, https://arxiv.org/abs/2010.02502）：

$$\mathbf{Y}_t^j = \alpha_j \mathbf{Y}_t^0 + \sigma_j \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

- $j \in \{1, \ldots, J\}$：diffusion step，$J$ 是总步数
- $\alpha_j, \sigma_j$：fixed noise-schedule coefficients，控制 signal/noise ratio
- $\mathbf{Y}_t^0$：clean trajectory（ground truth）
- $\mathbf{Y}_t^j$：第 $j$ 步加噪后的 trajectory

Loss function：
$$\mathcal{L}_{\text{diff}}(\theta) = \mathbb{E}_{(\mathbf{O}_t, \mathbf{Y}_t^0), \boldsymbol{\epsilon}, j} \left[ \| \boldsymbol{\epsilon} - \pi_\theta(\mathbf{O}_t, \mathbf{Y}_t^j, j) \|^2 \right]$$

- $\mathbf{O}_t = \{\mathbf{o}_{t-T_o+1}, \ldots, \mathbf{o}_t\}$：observation history，$T_o=2$
- $\pi_\theta$：U-Net denoiser，conditioned on $\mathbf{O}_t$ via FiLM（https://arxiv.org/abs/1709.07871）
- $\theta$：denoiser 参数

**为什么用 diffusion 而不是直接回归？** 因为 contact-rich manipulation 有 multimodality——同一个 observation 下可能有多种合理的 contact evolution。Diffusion model 天然能 model multimodal distribution，而 MSE regression 会 average out 这些 mode。

### 3.2 Contact-Consistency Mapping M_φ（Residual Form）

这是 paper 的核心创新。作者没有直接回归 $\mathbf{a}_t$，而是预测 **residual**：

$$\mathbf{a}_t = \mathbf{x}_t + \Delta\mathbf{a}_t, \quad \Delta\mathbf{a}_t = \mathcal{M}_\phi(\mathbf{x}_t, \mathbf{u}_t)$$

- $\Delta\mathbf{a}_t$：target 相对于 actual state 的 offset
- 这个 offset 就是 compliance controller 需要 "push against" 的量

**为什么 residual 这么重要？** 这里有个很深的 intuition。在 free space motion 时，$\Delta\mathbf{a}_t \approx 0$（target ≈ actual）。只有在 contact 时，$\Delta\mathbf{a}_t$ 才显著非零，并且其大小直接对应 contact force。如果直接回归 absolute $\mathbf{a}_t$，network 要同时学习 free-space motion 和 contact-induced offset，这两个 signal 的 scale 和 pattern 差异巨大，learning 会不稳定。Residual form 把 learning "anchor" 到 current state，network 只需要学 contact-induced correction，conditioning 更好，target 更 robust。

Table III 的 ablation 证实了这点：residual mode 的 MAE 普遍比 absolute mode 低 30-40%。

### 3.3 Latent Tactile Generation（VAE 压缩）

这部分解决一个 engineering 难题：tactile observation 维度太高。

Simulation 里是 748 × 3 的 tactile array（748 个 sensing point，每个 3D force），real hardware 里是 4 个 Digit360 sensor，每个 72×72 RGB image。直接在 raw space 做 16-step diffusion prediction 计算量爆炸。

解决方案：先用 VAE（Kingma & Welling 2013, https://arxiv.org/abs/1312.6114）压缩到 compact latent：

$$\mathbf{h}_t = E(\mathbf{u}_t), \quad \hat{\mathbf{u}}_t = G(\mathbf{h}_t)$$

- $E$：encoder
- $G$：decoder
- $\mathbf{h}_t \in \mathbb{R}^M$：latent，simulation 用 $M=32$，Digit360 用 $M=80$（每个 sensor 20 维）

**KL regularization 是关键**。Table IV 和 Fig. 7 的 ablation 显示了一个反直觉但重要的现象：去掉 KL 后 reconstruction MAE 反而更低（因为 VAE 不再被 prior 约束，可以 overfit），但 **policy rollout performance 显著下降**。

为什么？因为 diffusion 在 latent space 生成，如果 latent space 不 structured（有 hole、sharp boundary、non-smooth region），diffusion 的 denoising 过程会走到 "invalid" 区域，生成出来的 tactile latent 对应的 contact pattern 是物理上不合理的。KL regularization 让 latent distribution 接近 $\mathcal{N}(\mathbf{0}, \mathbf{I})$，整个 space smooth 且 well-conditioned，diffusion 生成稳定。这和 Latent Diffusion（Rombach et al. 2022, https://arxiv.org/abs/2112.10752）的 insight 一致。

## 4. Architecture 细节

### Tactile Encoder 的两种设计

**Simulation（tactile array, 748×3）：**
- 1D ResNet，convolution 沿 $N_{\text{tac}}=748$ 维做，$d_{\text{tac}}=3$ 当 channel
- Decoder：latent → 1D feature map → transposed conv 上采样 → residual refinement → final conv

**Real（Digit360, 4×72×72×3）：**
- Per-sensor 2D ResNet，shared weight
- Cross-sensor self-attention 聚合 4 个 sensor 的 feature
- 这种设计让 network 学习 sensor 间的 spatial relationship（比如 index finger 和 thumb 的协同 contact）

### Contact-Consistency Mapping 的两种设计

这里有个 engineering tradeoff：
- **Tactile array**：diffusion 预测 latent → decode 回 raw tactile → re-encode → concat with state → MLP。**保留高频 contact detail**，因为 array decode 轻量。
- **Digit360**：直接用 diffusion 预测的 latent + state → MLP。**避免 image reconstruction 的计算开销**，real-time 友好。

这个区别说明作者对 deployment constraint 有清醒认识——real robot 上 latency 是硬约束，不能为了 fidelity 牺牲 real-time 性。

### Inference 流程

Receding-horizon execution：
1. 采样 $(\hat{\mathbf{X}}_t, \hat{\mathbf{U}}_t) \sim \pi_\theta(\cdot | \mathbf{O}_t)$，预测 16 步
2. 对每步 $k$：$\hat{\mathbf{a}}_{t+k} = \mathcal{M}_\phi(\hat{\mathbf{x}}_{t+k}, \hat{\mathbf{u}}_{t+k})$
3. 执行前 8 步（$T_a=8$），然后 replan
4. Rollout frequency 5Hz，8 DDIM steps per inference

## 5. 实验数据深度分析

### Table II：主结果

| Task | CGP | Visuotactile DP | Visuomotor DP |
|------|-----|-----------------|---------------|
| In-Hand Box Flipping (sim) | **66.0%** | 58.0% | 53.2% |
| Fragile Egg Grasping (sim) | **74.8%** | 70.0% | 53.2% |
| Dish Wiping (sim) | **58.4%** | 43.6% | 42.4% |
| Jar Opening (real) | **93.3%** | 66.7% | 73.3% |
| Real In-Hand Box Flipping | **80.0%** | 60.0% | 60.0% |

几个观察：
1. **Dish Wiping 提升最显著**（+14.8% vs visuotactile）。这个任务是 sustained contact，需要 arm-driven force application，说明 CGP 的 contact grounding 对长 horizon contact 任务特别有效。
2. **Egg Grasping** 这种 delicate force regulation 任务，CGP 比 visuomotor 高 21.6%，说明 tactile 确实关键，而且 CGP 能用对。
3. **Jar Opening** 上 visuomotor (73.3%) 反而比 visuotactile (66.7%) 高，这说明 **naive 加 tactile 不一定有帮助**，甚至可能干扰。CGP 的 structured grounding 才能真正利用 tactile。
4. Real robot 上 CGP 优势更明显（Jar Opening +26.6%），可能因为 real world 的 contact noise 更大，CGP 的 contact-consistency constraint 提供了额外的 robustness。

### Table III：Hand Configuration Prediction Ablation

这个实验设计很 clever——把 contact-consistency mapping 单独拿出来评估，不涉及 policy learning，直接验证 mapping 本身是否学到了 transferable structure。

| Input | Encoder | Abs. Mode | Residual Mode |
|-------|---------|-----------|---------------|
| State + Tactile | ResNet1D | 8.80 | **7.94** |
| State + Tactile | MLP | 12.50 | 8.33 |
| State + Tactile | Transformer | 14.39 | 9.58 |
| State Only | - | 16.05 | 10.64 |
| Tactile Only | ResNet1D | 35.93 | 12.15 |

Key insights：
1. **State + Tactile 缺一不可**。Tactile only 完全不行（35.93），因为 tactile 是 local signal，无法 resolve global hand configuration。State only 也不行（16.05），miss 了 contact-induced variation。两者结合才能 capture contact-consistent structure。
2. **ResNet1D 最好**，Transformer 最差。这个有点反直觉——通常 Transformer 更强。我猜测原因是 tactile array 有 spatial locality（相邻 sensing point 的 force 相关性强），1D conv 能 exploit 这个 locality，而 Transformer 在小数据集（4114 frames）上容易 overfit。
3. **Residual mode 全面优于 absolute mode**，验证了前面的 intuition。

### Fig. 5：Contact Grounding 的直接验证

这是 paper 里我最喜欢的 figure。它做了 **time-aligned comparison**：在 $t$ 时刻预测的 tactile $\hat{\mathbf{u}}_{t+k}$，和 $t+k$ 时刻实际观察到的 $\mathbf{u}_{t+k}$ 对比。如果 CGP 真的 ground 了 contact，两者应该 align。

结果显示 align 得很好，这说明 **CGP 预测的 contact 不只是 plausible forecast，而是 controller 真的能 execute 出来的 contact**。这是和 "tactile as auxiliary prediction" 的根本区别。

Fig. 5 还有一个 subtle 的分析：对比 "align hand target/actual" 和 "align end-effector" 两种 overlay。Dish wiping 任务上两者差异明显（sponge 接触 plate 时 target-actual pose offset 大），说明这是 **arm-driven force application**。而 in-hand box flipping 两者相似，说明是 **hand-dominated interaction**。这证明 CGP 的 formulation 对 hand-driven 和 arm-driven contact 都 generalize。

### Table VI：Digit360 Tactile Compression

| Latent Dim | Model | MAE↓ | KL Loss↓ | PSNR↑ | SSIM↑ |
|------------|-------|------|----------|-------|-------|
| 80 | w/ KL | 9.01 | 0.0694 | 35.21 | 0.987 |
| 80 | w/o KL | 4.55 | 0.7781 | 40.46 | 0.994 |

去掉 KL 后 PSNR 高 5dB，但 KL loss 大 10 倍。**Reconstruction 好不等于 downstream 好**——这是 representation learning 的经典 lesson。CGP 最终用 80 维 latent（每 sensor 20 维），在 fidelity 和 runtime efficiency 间取得平衡。

## 6. 系统集成：Simulation vs Real

### Simulation Setup
- **Physics**：real-time finite element solver + Unreal Engine，用 reduced-order model with data-driven hyperreduction（参考 https://arxiv.org/abs/2509.25392）。这能 simulate compliant contact（比如 sponge deformation），对 dish wiping 这种任务关键。
- **Robot**：UR5 arm + Tesollo DG-5F 5-finger 20-DoF hand
- **Tactile**：748 sensing point，每个 3D force，覆盖所有 finger inner surface + palm
- **Teleoperation**：Meta Quest 3 VR hand tracking → RelaxedIK（http://relaxedik.com/）retargeting

### Real Robot Setup
- **Robot**：Franka Panda arm + Allegro V5 4-finger 16-DoF hand
- **Tactile**：4 × Digit360（https://arxiv.org/abs/2411.02479），fingertip mounted
- **Teleoperation**：OptiTrack mocap + instrumented glove → fingertip retargeting（ignore pinky）

两个 setup 的 teleoperation pipeline 共享 retargeting 和 controller stack，只是 tracking front-end 不同。这种 design 让 sim 和 real 的 demonstration distribution 尽量一致。

## 7. 我的 Intuition 和思考

读完这篇 paper，我 build 出来的核心 intuition 是：

**Contact grounding 本质是 learning the inverse dynamics of compliance control under contact。**

Compliance controller 的 forward dynamics 是：给定 $\mathbf{a}_t$ 和 environment，产生 $\mathbf{x}_t$ 和 $\mathbf{u}_t$。CGP 学的是 inverse：给定 $(\mathbf{x}_t, \mathbf{u}_t)$，反推 $\mathbf{a}_t$。这个 inverse 在 free space 是 trivial 的（$\mathbf{a}_t \approx \mathbf{x}_t$），但在 contact 下变得 non-trivial，因为 contact force 引入了 nonlinearity。

为什么这个 inverse 可以 learn？因为 demonstration data 里，human teleoperator 通过 controller 间接产生了 $(\mathbf{x}_t, \mathbf{u}_t, \mathbf{a}_t)$ triplet，这个 triplet 满足 controller 的物理约束。Network 学到的就是 controller-specific 的 contact-consistency。

**几个我觉得特别 clever 的 design choice：**

1. **Decoupling prediction 和 execution**。Diffusion 预测 "what contact should happen"，mapping 决定 "how to realize it"。这让两个 component 各自专注，diffusion 不用关心 controller dynamics，mapping 不用关心 temporal multimodality。

2. **Residual formulation**。这个 anchoring trick 在 robotics 里很通用（residual policy learning），但在这里特别有意义，因为 contact force 相对于 free-space motion 是 "small signal"。

3. **Latent tactile generation with KL**。这把 Latent Diffusion 的 insight 迁移到 tactile modality，解决了 high-dim tactile prediction 的效率问题。

**我觉得可以 push 的方向：**

1. **Cross-embodiment transfer**。现在 contact-consistency mapping 是 embodiment-specific 的。如果能 condition on controller parameter（impedance gain、update rate）和 robot kinematics，可能实现 cross-robot transfer。Paper 的 future work 也提到这个。

2. **Tactile 的因果模型**。现在 VAE 是无监督压缩，如果用 contrastive learning 或 predictive coding 学一个更有 physical meaning 的 latent（比如 disentangle normal force、friction、contact location），可能进一步提升 grounding quality。参考 Tactile Beyond Pixels（https://arxiv.org/abs/2412.xxxxx）。

3. **Sim-to-real 的 contact alignment**。Paper 明确说没做 sim-to-real，但 CGP 的 formulation 其实很适合——如果 sim 和 real 的 controller 一致，contact-consistency mapping 应该能 transfer。Domain randomization on tactile observation + fixed mapping 可能是一个 path。

4. **Long-horizon planning**。现在 prediction horizon 是 16 步（~3 秒），对于 multi-stage task（比如 pick → regrasp → place）可能不够。可以考虑 hierarchical structure：high-level planner 生成 subgoal，low-level CGP 执行。

5. **Tactile 的 active sensing**。现在 tactile 是被动观察，如果 policy 能主动 "explore" contact（比如 small perturbation 来 probe friction），可能更 robust。这和 active perception 的 idea 类似。

## 8. Limitations 的诚实评估

Paper 自己提的 limitations 我觉得都很 honest：
- **Sensor/controller specificity**：mapping 不能跨 sensor/controller transfer，要 retrain。
- **No sim-to-real**：sim 和 real 分开 evaluate，没做直接 transfer。
- **Single-task training**：每个 task 单独训一个 policy，没测 cross-task transfer。

我额外补充几个：
- **Demonstration efficiency**：45-100 demos per task，对于 dexterous manipulation 不算多，但也不算少。能否 few-shot？
- **Tactile sensor coverage**：real robot 只用 fingertip Digit360，palm 没 sensing。这对 in-hand manipulation 是 limitation，因为 palm contact 很重要。
- **Compliance controller assumption**：整个 framework 依赖 low-level controller 是 compliance 的（PD/impedance）。如果用 position control 或 torque control，formulation 要改。

## 9. 与 Related Work 的定位

Table I 很好地总结了 CGP 的定位：

| Method Category | Executable Contact | Multi-Finger | Distributed Contacts |
|---|---|---|---|
| Adaptive Compliance Policies | ✓ | ✗ | ✗ |
| Sparse Fingertip Force Policies | ✓ | ✓ | ✗ |
| **CGP (Ours)** | ✓ | ✓ | ✓ |

CGP 是第一个同时满足三个条件的：executable contact modeling + multi-finger hand + extensible to distributed contacts。

和最接近的工作对比：
- **Adaptive Compliance Policy**（Hou et al. 2025, https://arxiv.org/abs/2411.xxxxx）：学 compliance reference，但只针对 arm end-effector，single contact point。
- **DexForce**（Chen et al. 2025, https://arxiv.org/abs/2506.xxxxx）：预测 sparse fingertip force，但 multi-finger hand 的 dense contact 建模不了。
- **Reactive Diffusion Policy**（Xue et al. 2025, https://arxiv.org/abs/2504.xxxxx）：slow-fast visual-tactile，但 tactile 还是当 observation，没 grounding 到 controller。
- **ViTacFormer**（Heng et al. 2025, https://arxiv.org/abs/2506.15953）：cross-modal representation，但没 explicit contact-consistency mapping。

CGP 的独特价值在于 **把 contact prediction 和 control execution 显式 coupled 起来**，这个 coupling 通过 learned mapping 实现，既避免了 hand-designed contact model 的局限性，又保证了 physical executability。

---

总结一下，这篇 paper 的 contribution 不只是 engineering trick，而是提出了一个新的 **problem formulation**：把 contact-rich manipulation 看作 contact grounding 问题，用 (x, u, a) triplet 隐式表示 contact，用 learned mapping 实现 prediction-execution coupling。这个 formulation 我觉得有 generalization potential，不限于 dexterous hand，任何 compliance-controlled contact-rich task都可能受益。

如果你想 deep dive 某个部分，比如 FEM simulation 的细节、RelaxedIK 的 retargeting、或者 Digit360 的 signal processing，我可以再展开。
