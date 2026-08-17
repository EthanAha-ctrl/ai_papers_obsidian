---
source_pdf: D-REX DIFFERENTIABLE REAL-TO-SIM-TO-REAL ENGINE.pdf
paper_sha256: fc1a02664aff2023ffd590ded4a5b6c4c8fe43083e64721936ef32ee136e5934
processed_at: '2026-08-03T18:14:34-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# D-REX 用人话讲

## 这个paper到底在干嘛

想象一个场景：你让robot去抓一个ketchup瓶子。Robot在simulation里训练了一万次，到real world一试，ketchup "啪"掉地上了。Why？

Simulation里你假设ketchup bottle重0.1kg，但real world的ketchup bottle重0.7kg。Robot用的force是给0.1kg设计的，抓0.7kg的瓶子当然滑掉。

这是个非常practical的问题。Sim-to-real最大的bottleneck之一就是physical parameters不对。你可以说"那我randomize mass好了"，但randomize会dilute policy performance——你training时见0.1kg和1kg都用力一样，deploy到0.7kg还是不知道该用多大力。

D-REX的idea很直接：**与其盲目randomize，不如直接从real world把mass测出来**。

但问题是怎么测？你不能给robot装个秤。你能观测到的就是video + robot自己的joint encoders。

---

## 核心intuition：从trajectory反推mass

这事儿其实没那么神秘。Newton第二定律 $F=ma$ 大家都懂。如果robot推一个物体，你能看到物体怎么动，你知道robot用了多大力，你就能反推mass。

具体来说：

- Real world里robot推ketchup，你用FoundationPose（一个6-DoF pose estimator，https://arxiv.org/abs/2403.07715）从video里把ketchup每一帧的位置和朝向track下来。这是real trajectory。
- Simulator里robot用同样的action推ketchup，simulator给你sim trajectory。
- 调mass，让sim trajectory跟real trajectory match。

这是system identification的老思路。新意在于：**用differentiable physics让这个匹配过程可以gradient descent**。

---

## 为什么differentiable physics是关键

如果simulator不可微，你要估mass只能用finite difference：试mass=0.1，试mass=0.2，试mass=0.3...看哪个trajectory最match。这在高维参数空间里scale极差。

Differentiable physics让你能直接计算 $\partial \text{trajectory} / \partial \text{mass}$，然后gradient descent。每一步gradient告诉你"mass应该往大调还是往小调"。

他们用了GradSim（https://arxiv.org/abs/2104.02646）的differentiable physics engine + MJX/Brax（https://github.com/google/brax）的kinematics。Contact model用compliant penalty-based——penetration depth是position的continuous function，gradient可以flow。

---

## 一个非常漂亮的数学insight

Paper Appendix A.2.4里有个simplified分析，我觉得是整个工作最精彩的地方。

考虑1D push-down场景，robot施加力 $u(t)$ 推一个mass $m$ 的物体：

$$m\ddot{z}(t) = u(t) - mg$$

二次积分：

$$z(t; m) = z_0 + v_0 t + \frac{1}{m}\int_0^t \int_0^\tau u(s) ds d\tau - \frac{1}{2}gt^2$$

注意这个结构：trajectory $z(t)$ 是 $\frac{1}{m}$ 的affine function。$\alpha(t)$ 和 $\beta(t)$ 都是已知量（initial condition + robot force integral + gravity）。

Loss定义为：

$$\mathcal{L}(m) = \sum_k \|z_k(m) - \hat{z}_k\|^2$$

令 $\theta = 1/m$，loss变成：

$$\mathcal{L}(\theta) = \sum_k \|\alpha_k + \beta_k \theta - \hat{z}_k\|^2$$

这是关于 $\theta$ 的**quadratic function，严格convex**。只有唯一global minimum，gradient descent保证收敛。

这意味着mass identification不是那种nasty non-convex optimization，在simplified场景下其实是个well-conditioned least-squares problem。复杂的contact场景虽然没这么clean，但本质上结构类似。

这个insight让我对方法的robustness有信心。Appendix A.3.2的实验也验证：从2g初始值（比true mass小100-350×）也能converge到正确值。

---

## Real-to-Sim部分：dual Gaussian Splats

重建部分用了个很clever的设计：**两套Gaussian Splats互不干扰**。

一套3D Gaussians负责photometric rendering，optimize appearance。另一套2D surface-aligned Gaussians负责geometry，optimize depth + normal consistency。两套参数完全disjoint，loss也disjoint。

为什么这么做？Appearance和geometry的optimum经常conflict。你想要pretty rendering可能让geometry不准，想要clean geometry可能让rendering变差。Multi-task learning的negative transfer问题。

最后surface Gaussians rasterize成depth maps → TSDF fusion → marching cubes → collision mesh。整个pipeline ~30分钟/object，mesh vertex从13k到67k不等。

参考：3DGS原始paper（https://repo.samkolb.com/3dgaussians/assets/3dgaussians.pdf），2DGS（https://arxiv.org/abs/2403.19632），他们之前的Robo-GS工作（https://arxiv.org/abs/2408.14873）。

---

## 从human video到robot action

你给robot看一段human抓ketchup的video，怎么让robot模仿？

每帧video跑两个model：
- **HaMeR**（https://arxiv.org/abs/2402.09214）：估human hand的6-DoF wrist pose + finger joint angles
- **MCC-HO**（https://arxiv.org/abs/2404.06507）：估object的6-DoF pose

然后用**Dex-Retargeting**（https://arxiv.org/abs/2305.01692）把human hand pose映射到robot hand。Human hand有更多DoF且morphology不同，retargeting解决这个embodiment gap。

输出是robot的16个joint target angles，直接作为demonstration。

---

## Force-aware policy：这是paper的另一个核心contribution

传统grasping policy只predict position：手指应该放在哪。D-REX的policy同时predict position + force。

Policy是个简单MLP（Table 6）：
- Input：object mesh vertices的positional encoding + identified mass
- 3层256-wide MLP + ReLU
- 3个head：16D action + 2D contact constraint + 1D force

Force的公式很physically grounded：

$$\hat{f} = \frac{m \cdot g}{n_{\text{active}}}$$

总抗重力force = $mg$（让物体不掉下去需要的总力），均匀分配到 $n_{\text{active}}$ 个active contact上。1个手指接触就承担全部，4个手指就各承担1/4。

Deployment时，LEAP Hand是direct-drive motor，current和force近似linear，所以直接用predicted normalized force控制motor current。

---

## 为什么force-aware这么重要

Table 3的cross-evaluation实验特别说明问题：

| Train\Eval | Light | Medium | Heavy |
|------------|-------|--------|-------|
| Light policy | 75% | 30% | 15% |
| Medium policy | 40% | 80% | 30% |
| Heavy policy | 15% | 40% | 95% |

Diagonal 75-95%，off-diagonal掉到15-40%。**Policy强烈overfit到training mass**。

Figure 6的qualitative结果更直观：
- Light policy抓heavy object：力太小 → slippage（滑掉）
- Heavy policy抓light object：力太大 → bounce-off（弹飞）
- Mass-matched：稳定

这说明**force不是可以randomize掉的noise，是必须condition on的物理量**。

Appendix A.6给了quantitative分析：

$$F_n \gtrsim \frac{mg}{2\mu}\gamma$$

- $F_n$：需要的normal force
- $\mu$：friction coefficient
- $\gamma$：safety margin

Mass增加，required force线性增加。Baseline policy用固定force，mass超出training range就fail。

---

## 跟Domain Randomization对比

Appendix A.3.5做了跟CrossDex（https://arxiv.org/abs/2410.02479）的对比。CrossDex用mass randomization在[0.5, 1.5]kg训练。

| Method | 117g | 206g | 324g |
|--------|------|------|------|
| CrossDex | 4/10 | 7/10 | 9/10 |
| D-REX | 9/10 | 10/10 | 9/10 |

测试mass全在DR training range外（都更轻）。CrossDex对light object直接fail——DR对OOD mass敏感。D-REX通过explicit mass identification避开了这个限制。

这个对比很有意思。DR的优势是"覆盖一个range"，但对range外的mass无能为力。Mass identification是"测准每个object的mass"，没有range概念。两者complementary，但identification更data-efficient。

---

## 两个hand的分工

工程上有个clever design：mass identification用Allegro Hand，grasping用LEAP Hand。

**Allegro Hand**（https://www.wonikrobotics.com/allegro-hand）：
- 16 DoF
- 内置wiring，结构紧凑
- Low-torque，clean contact dynamics
- 噪音低，differentiable physics的gradient flow稳定

**LEAP Hand**（https://arxiv.org/abs/2309.06440）：
- 16 DoF
- 高torque，能抓重物
- Tendon-driven，human-like
- Current control linearly maps to force
- 但exposed wiring引入mechanical noise，对mass identification不友好

Decoupling roles让每个stage用最合适的hardware。Mass identification需要precise dynamics，用Allegro；grasping需要strong force，用LEAP。

---

## Mass identification的实际数字

Table 1，不同geometry的objects：

| Object | VLM初值 | Identified | Ground Truth | Error |
|--------|---------|------------|--------------|-------|
| Letter U | 500g | 110g | 125g | 12.0% |
| Letter A | 500g | 145g | 134g | 9.0% |
| Lego | 300g | 53g | 59g | 8.6% |
| Domino | 500g | 117g | 106g | 9.3% |
| Cookie | 500g | 200g | 210g | 4.8% |
| Ketchup | 1000g | 667g | 726g | 8.1% |

VLM初值off by 4-10×，优化后误差5-12%。VLM只是给个起点，真正识别靠differentiable physics。

Table 2，同geometry不同density（3D打印infill不同）：

| Density | ρ1 | ρ2 | ρ3 |
|---------|-----|-----|-----|
| Identified | 95g | 129g | 207g |
| Ground Truth | 82g | 125g | 218g |

固定shape变density也能识别，证明方法学的是mass不是geometry。

---

## 跟baselines的grasping对比

Figure 7，8个objects对比DexGraspNet 2.0（https://arxiv.org/abs/2410.23004）和Human2Sim2Robot（https://arxiv.org/abs/2504.12609）：

- DexGraspNet 2.0：大规模sim data训练，固定mass ~0.1kg
- Human2Sim2Robot：从RGBD human demo学习

D-REX在heavy objects上明显胜出。Baselines在heavy object上崩盘，因为它们training时见到的mass都在0.1kg附近，deploy到0.5-1kg object时force不够。

---

## 跟GradSim的区别

GradSim（https://arxiv.org/abs/2104.02646）是predecessor，D-REX的关键区别（Appendix A.9）：

1. **GradSim需要photometric supervision**：要controlled lighting + calibrated cameras + 3D-printed objects。D-REX直接用FoundationPose的6-DoF pose做state-space trajectory loss。
2. **GradSim假设full simulator state**。D-REX从partial noisy real observations估mass。
3. **D-REX把mass identification formulated成constrained optimization**：known robot inputs + accurate boundary conditions → 找mass让FoundationPose trajectory最好reproduced。

GradSim的photometric loss在unstructured real world环境里不practical。D-REX用pose-based loss更robust。

---

## Numerical stability的trick

如果按物理直觉uniform分配mass（每个vertex分摊 $m/N$），高分辨率mesh（50k vertices）+ 小mass会让per-particle mass $< 10^{-6}$ kg，numerical instability + gradient explosion。

他们的fix：**把整个object mass赋给每个particle**，gravity均匀施加，external force按采样vertex数量scaling。这违反物理直觉但math上保持gradient numerically stable。这是个practical engineering choice。

---

## Semi-implicit vs Explicit Euler

Appendix A.2.4的ablation（Table 5）：

| Object | GT | Semi-implicit | Explicit |
|--------|-----|---------------|----------|
| Lego | 59g | 51g | 34g |
| Ketchup | 726g | 667g | 685g |
| Cookie | 210g | 200g | 189g |

Semi-implicit更准确。原因：contact-rich dynamics对small mass或stiff contact，explicit Euler stability region太小，会oscillate甚至diverge。Semi-implicit先update velocity再用新velocity推position，对stiff system更稳定。

这跟Position-Based Dynamics（PBD，https://doi.org/10.1016/j.jvcir.2007.01.005）的思路一致。

---

## Training pipeline

Algorithm 1和2描述的two-stage training：

**Phase 1: Supervised pre-training**
- 数据：human demo retargeted到robot
- Loss：MSE(action) + BCE(contact) + MSE(force)
- Force和contact label都设为1（所有demo假设成功）

**Phase 2: Simulation refinement**
- 用MuJoCo rollout policy
- 计算 $f_{\text{env}} = \text{clip}(\frac{m \cdot g \cdot \text{num\_contacts}}{f_{\max}}, 0, 1)$
- Loss：$0.8 \cdot \text{BCE(contact)} + 0.3 \cdot \text{MSE(force)}$

这是RL-style fine-tuning但在supervised框架里。Phase 2让policy在sim里试，看哪些grasp真的能hold住object，refine force prediction。

Training cost：200-300 demos/object，2分钟训练。复杂object 5000 demos，20分钟。Inference 0.5s/object pose。

---

## Scaling和Generalization

Figure 9的scaling experiment（screwdriver）：
- 1-10 demos：抓不稳
- 20+ demos：稳定
- 5000 demos best

Cross-object generalization（Table 8）：大screwdriver训的policy，换小screwdriver，success rate 90% → 70%。Moderate degradation，within-category generalization可行。

也测了articulated object（stapler）和fine-grained task（computer mouse），都能work。证明框架不限于grasping。

---

## Limitations

- 只学mass，没学friction/stiffness/damping（这些更难实验validate）
- 假设rigid-body dynamics
- Real-to-sim阶段后不能有人在scene里干预
- Object-specific policy，不是fully general
- FoundationPose的z-axis误差有时需要manual post-processing

Paper在A.4解释了为什么focus on mass：mass有clear ground truth，实验容易validate，对grasping影响直接observable。Friction这种参数contact-dependent，spatial-temporal variable，real world validation很难。

---

## 我的take

这个工作的核心贡献是：**把differentiable simulation从"academic demo"变成"practical system identification tool"**。

GradSim证明了differentiable physics能从pixels学参数，但需要lab条件。D-REX让它能在unstructured real world work，靠的是：
1. 用FoundationPose替代photometric supervision
2. 用robot proprioception替代manual force specification
3. 用mass这个well-defined scalar parameter替代high-dimensional physical parameter space

然后mass-conditioned force control是个very clean的idea：总force = $mg$，分配到 $n_{\text{active}}$ 个contact。Physically grounded，computationally trivial，但解决了真实的sim-to-real bottleneck。

两个insight我觉得最有价值：
1. **$1/m$ 参数化让loss变convex**——这是整个方法robustness的theoretical foundation
2. **Force是必须condition on的physical quantity，不是可以randomize掉的noise**——Table 3的cross-evaluation和数据很有说服力

整体来说，D-REX是个systems paper，每个component都不是全新，但组合起来解决了一个concrete problem。Differentiable simulation + Gaussian Splatting + human demo retargeting + mass-conditioned force control，每个component都有prior art，但把它们seamlessly integrate成real-to-sim-to-real pipeline，加上理论分析（convexity）和实验验证（mass accuracy + grasping success），是个很complete的故事。

参考链接汇总：
- D-REX project: https://drex.github.io
- GradSim: https://arxiv.org/abs/2104.02646
- FoundationPose: https://arxiv.org/abs/2403.07715
- 3DGS: https://repo.samkolb.com/3dgaussians/assets/3dgaussians.pdf
- 2DGS: https://arxiv.org/abs/2403.19632
- HaMeR: https://arxiv.org/abs/2402.09214
- MCC-HO: https://arxiv.org/abs/2404.06507
- Dex-Retargeting: https://arxiv.org/abs/2305.01692
- LEAP Hand: https://arxiv.org/abs/2309.06440
- Brax/MJX: https://github.com/google/brax
- Human2Sim2Robot: https://arxiv.org/abs/2504.12609
- DexGraspNet 2.0: https://arxiv.org/abs/2410.23004
- CrossDex: https://arxiv.org/abs/2410.02479
- Robo-GS: https://arxiv.org/abs/2408.14873
- PBD: https://doi.org/10.1016/j.jvcir.2007.01.005

---

# D-REX: Differentiable Real-to-Sim-to-Real Engine 深度讲解

## 1. 核心问题与Motivation

这篇paper要解决一个非常fundamental的问题：**如何从visual observations + robot control signals反推physical parameters（特别是mass），并把这些参数用于force-aware grasping policy的学习**。

传统sim-to-real存在一个chicken-and-egg问题：你要train一个好policy，simulator要accurate；simulator要accurate，physical parameters要对；physical parameters要从real world观测；但观测本身又依赖simulator做forward prediction来对齐。D-REX用differentiable simulation把这个loop闭合起来。

项目page: https://drex.github.io

关键intuition在于：**mass是一个可以从robot-object interaction的trajectory中反推的scalar parameter**。如果robot在real world推一个物体产生trajectory $s_t^{real}$，你在simulator里用同一个action推，调mass让 $s_t^{sim}(m)$ match $s_t^{real}$，这个mass就是真实物理的近似。

---

## 2. 整体架构（4个stage）

```
Real Videos (T_s, T_o, T_t)
        ↓
[1] Real-to-Sim: Gaussian Splatting → collision mesh K + visual P
        ↓
[2] Mass Identification: differentiable physics + robot actions → m*
        ↓
[3] Human Demo Transfer: HaMeR + MCC-HO + Dex-Retargeting → A_t
        ↓
[4] Policy Learning: GraspMLP conditioned on (K, m*) → π_φ
        ↓
Real-world deployment
```

四个组件decoupled设计的好处：每个stage可以独立debug，gradient flow清晰，mass identification的error不会被policy learning放大。

---

## 3. Real-to-Sim Reconstruction（Section 4.1）

这里用了一个**dual Gaussian Splat ensemble**的设计，很关键：

**Rendering set** $\mathcal{P}^{rend}$：3D Gaussians，每个primitive有 $(x_i, y_i, z_i, r_i, g_i, b_i, o_i, s_i, \Sigma_i)$
- $(x,y,z)$: center位置
- $(r,g,b)$: RGB颜色
- $o$: opacity for alpha blending
- $\Sigma \in \mathbb{R}^{3\times 3}$: anisotropic covariance（控制Gaussian的椭球形状）
- $s$: semantic/instance id
- 优化目标：photometric loss

**Surface set** $\mathcal{P}^{surf}$：2D surface-aligned Gaussians，每个primitive有 $(x_j, y_j, z_j, \mathbf{t}_{u,j}, \mathbf{t}_{v,j}, s_{u,j}, s_{v,j})$
- $\mathbf{t}_u, \mathbf{t}_v$: orthonormal tangents（切平面两个方向）
- $s_u, s_v$: 沿切向的standard deviation
- 法向：$\mathbf{n}_j = \mathbf{t}_{u,j} \times \mathbf{t}_{v,j}$
- 优化目标：depth distortion + normal consistency
- **不**受photometric loss影响

为什么分开？因为appearance和geometry的optimum经常conflict——你想要pretty rendering可能损害geometry精度，反之亦然。两套参数disjoint + disjoint loss，互不干扰。最后surface Gaussians rasterize成depth maps → TSDF → marching cubes → collision mesh $\mathcal{K}$。

Runtime: ~300张图，30-35分钟/object，mesh vertex从13k到67k不等（Table 4）。

参考：3DGS paper (https://repo.samkolb.com/3dgs/assets/3dgaussians.pdf)，2DGS (https://arxiv.org/abs/2403.19632)

---

## 4. Mass Identification核心数学（Section 4.2）

这是paper最精彩的部分。让我们一层层剥开。

### 4.1 Objective

$$\min_{m>0} \mathcal{L}_{\text{traj}}(m) := \sum_{t=1}^{T} \|\mathbf{s}_t^{\text{sim}}(m) - \mathbf{s}_t^{\text{real}}\|_2^2 \quad (1)$$

变量：
- $m$: 待优化的object mass（scalar）
- $\mathbf{s}_t = [\mathbf{p}, \mathbf{q}]^\top \in \mathbb{R}^7$：6-DoF pose，其中 $\mathbf{p} \in \mathbb{R}^3$ 是position，$\mathbf{q} \in \mathbb{R}^4$ 是unit quaternion表示orientation
- $T$: trajectory长度

$\mathbf{s}_t^{\text{real}}$ 由 FoundationPose (Wen et al. 2024, https://arxiv.org/abs/2403.07715) 给出。$\mathbf{s}_t^{\text{sim}}(m)$ 由differentiable physics engine跑出来。

### 4.2 Dynamics model

Newton-Euler equation：

$$\mathbf{M}(\mathbf{s}_t, \mathbf{u}_t, m, \theta) \dot{\mathbf{u}}_t = \mathbf{f}(\mathbf{s}_t, \mathbf{u}_t, \theta) \quad (2)$$

变量：
- $\mathbf{u}_t = [\mathbf{v}_t, \omega_t]^\top$：速度，linear $\mathbf{v}_t$ + angular $\omega_t$
- $\mathbf{M}$: mass-inertia matrix（依赖当前pose和mass）
- $\theta$: 其他物理参数（contact stiffness $k_e$, damping $k_d$）
- $\mathbf{f}$: 合力（external + contact + gravity + torque）

Contact model（compliant penalty-based）：

$$\mathbf{f}_n(\mathbf{s}, \mathbf{u}_t, \theta) = -\mathbf{n}\big(k_e C(\mathbf{s}) + k_d \dot{C}(\mathbf{u})\big) \quad (3)$$

- $\mathbf{n}$: contact normal
- $C(\mathbf{s})$: penetration depth（标量，how much物体嵌入contact surface）
- $\dot{C}(\mathbf{u})$: penetration depth的时间导数（contact velocity）
- $k_e, k_d$: stiffness和damping，determine接触多"硬"

这个contact model的好处是differentiable——penetration depth是position的continuous function，没有hard non-penetration constraint，gradient可以flow。

### 4.3 Semi-implicit Euler integration

$$G\Big([\mathbf{s}_t, \mathbf{u}_t], m, \theta\Big) = \begin{bmatrix} \mathbf{s}_t + \Delta t \mathbf{u}_{t+1} \\ \mathbf{u}_{t+1} \end{bmatrix} = \begin{bmatrix} \mathbf{s}_t + \Delta t \Big(\mathbf{u}_t + \Delta t \mathbf{M}^{-1} \mathbf{f}\Big) \\ \mathbf{u}_t + \Delta t \mathbf{M}^{-1} \mathbf{f} \end{bmatrix} \quad (5)$$

变量：
- $\Delta t$: timestep
- 注意velocity先update，再用**new velocity** update position（这就是"semi-implicit"）

为什么不用explicit Euler？Appendix A.2.4有ablation（Table 5）：

| Object | GT (g) | Semi-implicit | Explicit |
|--------|--------|---------------|----------|
| Lego | 59 | 51 | 34 |
| Ketchup | 726 | 667 | 685 |
| Cookie | 210 | 200 | 189 |
| Domino | 106 | 117 | 135 |
| U | 125 | 110 | 98 |
| A | 134 | 145 | 120 |

Semi-implicit更准确（虽然per-iteration慢一点：1.36-1.43s vs 1.17-1.22s）。原因：contact-rich dynamics对小mass或stiff contact来说，explicit Euler stability region太小，会oscillate甚至diverge。Semi-implicit先把velocity update再用新velocity推position，相当于对stiff system更稳定。

### 4.4 Gradient computation

$$\frac{\partial \mathcal{L}_{\text{traj}}}{\partial m} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}_{\text{traj}}}{\partial \mathbf{s}_t^{\text{sim}}} \cdot \frac{\partial \mathbf{s}_t^{\text{sim}}}{\partial \mathbf{M}_t} \cdot \frac{\partial \mathbf{M}_t}{\partial m} \quad (7)$$

这是discrete adjoint method的核心。三步chain rule：
1. Loss对trajectory每个state的偏导（直接）
2. State对mass-inertia matrix的偏导（需要unroll整个dynamics，类似BPTT）
3. Mass-inertia matrix对mass的偏导（analytic）

PyTorch自动做backprop。这里关键挑战是contact event是discontinuous的，但用了compliant model让它变smooth。

### 4.5 **最漂亮的insight**：为什么这个loss在1/m参数化下是convex

Appendix A.2.4给出了simplified push-down场景的分析。考虑1D push：

$$m\ddot{z}(t) = u(t) - mg \quad (21)$$

- $z(t)$: object沿push axis的位置
- $u(t)$: 已知的robot contact force
- $m$: 待估mass

二次积分：

$$z(t; m) = z_0 + v_0 t + \frac{1}{m}\int_0^t \int_0^\tau u(s) ds d\tau - \frac{1}{2}gt^2 \quad (22)$$

写成affine形式：

$$z(t; m) = \alpha(t) + \frac{1}{m}\beta(t) \quad (23)$$

其中 $\alpha(t)$ 只含initial condition和gravity项，$\beta(t)$ 是double integral of $u$——这些**完全已知**，跟$m$无关。

离散化：$z_k(m) = \alpha_k + \frac{1}{m}\beta_k$。

Loss变成：

$$\mathcal{L}_{\text{traj}}(m) = \sum_k \|z_k(m) - \hat{z}_k\|^2 = \sum_k \|\alpha_k + \frac{\beta_k}{m} - \hat{z}_k\|^2$$

令 $\theta = 1/m$：

$$\mathcal{L}_{\text{traj}}(\theta) = \sum_k \|\alpha_k + \beta_k \theta - \hat{z}_k\|^2$$

这是关于 $\theta$ 的**quadratic function，convex**！只有唯一global minimum，gradient descent不会卡local minima。

这个insight非常重要：mass identification不是ill-posed的non-convex problem，在simplified场景下其实是个well-conditioned least-squares。复杂contact场景虽然没这么clean，但本质上也是同样的结构。这给了方法很strong的理论基础。

### 4.6 Numerical stability trick

如果直接用uniform mass distribution（每个vertex分摊 $m/N$），高分辨率mesh（50k+ vertices）+ 小mass会让per-particle mass $<10^{-6}$ kg，numerical instability + gradient explosion。

他们的fix：**把整个object mass赋给每个particle**，gravity均匀施加，external force按采样vertex数量scaling。这违反物理直觉，但math上保持gradient numerically stable。

### 4.7 Mass identification实验结果

Table 1（不同geometry）：

| Object | Letter U | Letter A | Lego | Domino | Cookie | Ketchup |
|--------|----------|----------|------|--------|--------|---------|
| Inferred (VLM) | 500 | 500 | 300 | 500 | 500 | 1000 |
| Identified | 110 | 145 | 53 | 117 | 200 | 667 |
| Ground Truth | 125 | 134 | 59 | 106 | 210 | 726 |
| Error % | 12.0 | 9.0 | 8.6 | 9.3 | 4.8 | 8.1 |

VLM初值off by 4-10×，优化后误差5-12%。注意VLM只是给初始猜测，真正的识别靠differentiable physics。

Table 2（同geometry不同density）：

| Density | $\rho_1$ | $\rho_2$ | $\rho_3$ |
|---------|----------|----------|----------|
| Identified | 95g | 129g | 207g |
| Ground Truth | 82g | 125g | 218g |

固定shape变density也能识别——证明方法确实学到了mass而非geometry。

Convergence通常200 epochs，5-20分钟/object。

---

## 5. Human Demo Transfer（Section 4.3）

每帧 $\mathcal{T}_t$ 处理流程：

1. **HaMeR** (Pavlakos et al. 2024, https://arxiv.org/abs/2402.09214) 重建human hand：$\mathbf{h}_t \in SE(3) \times \mathbb{R}^{J_h}$
   - $SE(3)$: wrist 6-DoF pose
   - $\mathbb{R}^{J_h}$: finger joint angles（$J_h$ = 人类手关节数）

2. **MCC-HO** (Wu et al. 2024, https://arxiv.org/abs/2404.06507) 重建object：$\mathbf{o}_t \in SE(3)$
   - Object 6-DoF pose

3. **Dex-Retargeting** (Qin et al. 2023, https://arxiv.org/abs/2305.01692) 把human hand映射到robot hand：

$$\mathbf{A}_t = \mathcal{R}(\mathbf{h}_t, \mathbf{o}_t) \in \mathbb{R}^{J_r} \quad (9)$$

- $\mathcal{R}$: retargeting function
- $J_r = 16$: robot hand DoF
- $\mathbf{A}_t$: target joint angles

关键假设：object geometry在human demo和robot manipulation之间保持consistent。这是reasonable的因为同一个object被抓。

---

## 6. Force-Aware Policy Learning（Section 4.4）

### 6.1 Policy architecture

$\pi_\phi$ 是multi-head network，输入是object mesh vertices的positional encoding + identified mass $m$，输出三个head：

$$\pi_\phi(\mathbf{o}) = [\hat{\mathbf{A}}, \hat{\mathbf{r}}, \hat{\mathbf{f}}]^\top \in \mathbb{R}^{19} \quad (10)$$

- $\hat{\mathbf{A}} \in \mathbb{R}^{16}$: 16个robot joint positions
- $\hat{\mathbf{r}} \in \mathbb{R}^2$: contact constraint（2维，一个维持contact，一个end-state retention）
- $\hat{\mathbf{f}} \in \mathbb{R}$: normalized grasping force

Force公式：

$$\hat{\mathbf{f}} = \frac{m \cdot g}{n_{\text{active}}}$$

- $m$: identified mass
- $g$: gravity
- $n_{\text{active}}$: active contact数量（手指和object接触点数）

这个公式的物理意义：**总抗重力force = $mg$，均匀分配到 $n_{\text{active}}$ 个contact point上**。Very clean和physically grounded。如果一个手指接触，它要承担全部重量；4个手指接触，每个承担1/4。

具体网络结构（Table 6）：
- Input: positional encoding of N vertices (XYZ)
- Linear 3→256 + ReLU
- Linear 256→256 + ReLU  
- Linear 256→256 + ReLU
- 三个head：Action(Linear→16) / Reward(Linear+Sigmoid→2) / Force(Linear+Sigmoid→1)

### 6.2 Contact constraint

$$\forall t \in [t_0, t_0 + H]: n_{\text{active}}(t) \geq N_{\min}, \quad \mathbb{I}_{\text{in\_hand}}(t) = \begin{cases} 1, & \text{if } n_{\text{active}}(t) \geq 1 \\ 0, & \text{otherwise} \end{cases} \quad (11)$$

- $H$: time horizon
- $N_{\min}$: 阈值，最少多少contact point要维持
- $\mathbb{I}_{\text{in\_hand}}$: indicator，object是否还在手里

### 6.3 Two-stage training（Algorithm 1, 2）

**Phase 1: Supervised pre-training**
- 数据：human demo retargeted到robot
- Loss: $\mathcal{L} = \mathcal{L}_a (\text{MSE}) + \mathcal{L}_r (\text{BCE}) + \mathcal{L}_f (\text{MSE})$
- Force和contact label都设为1（所有demo都假设成功）

**Phase 2: Simulation refinement**
- 用MuJoCo rollout policy
- 计算 $f_{\text{env}} = \text{clip}(\frac{m \cdot g \cdot \text{num\_contacts}}{f_{\max}}, 0, 1)$
- Loss: $\mathcal{L} = 0.8 \mathcal{L}_r + 0.3 \mathcal{L}_f$（注意系数，contact更重要）

这其实是RL-style fine-tuning但在supervised框架里——很巧妙的hybrid。

### 6.4 Force deployment

LEAP Hand是direct-drive brushless motor，motor current → fingertip force近似linear，所以current control直接proxy force control。Allegro也类似。

```
Predicted normalized force ∈ [0,1] → Motor current limit → Torque → Fingertip force
```

---

## 7. 关键实验结果

### 7.1 Mass-conditioned policy的necessity（Figure 5, 6, Table 3）

Table 3 cross-evaluation：

| Train\Eval | $\rho_1$ | $\rho_2$ | $\rho_3$ |
|------------|----------|----------|----------|
| $\rho_1$ | 75% | 30% | 15% |
| $\rho_2$ | 40% | 80% | 30% |
| $\rho_3$ | 15% | 40% | 95% |

Diagonal 75-95%，off-diagonal掉到15-40%。这证明**policy强烈overfit到training mass**，换了mass就fail。这是force-aware control的强证据。

Figure 6定性结果更生动：
- Medium mass policy抓medium object: 稳定
- Medium mass policy抓light object: 力太大 → bounce-off（物体被弹开）
- Medium mass policy抓heavy object: 力太小 → slippage（物体滑落）

### 7.2 Identified mass vs Ground truth mass比较（Figure 5）

Success rate在ground-truth mass和identified mass处都peak，identified mass的policy甚至能match甚至超过ground truth的performance。这意味着mass identification的5-12%误差对policy影响很小。

### 7.3 对比baselines（Figure 7）

vs **DexGraspNet 2.0** (https://arxiv.org/abs/2410.23004): 大规模sim data训练，固定mass ~0.1kg
vs **Human2Sim2Robot** (Lum et al. 2025, https://arxiv.org/abs/2504.12609): 从RGBD human demo学习

8个objects（不同geometry + mass），D-REX在heavy objects上明显胜出，baselines在heavy object上崩盘。原因在Appendix A.6分析：

$$F_n \gtrsim \frac{mg}{2\mu}\gamma$$

- $F_n$: 所需normal force
- $m$: mass
- $g$: gravity
- $\mu$: friction coefficient
- $\gamma \geq 1$: wrench distribution + safety margin

Baseline训练mass ≈ 0.1kg，部署到0.5-1kg object，固定force不够 → slip。D-REX因为conditioned on identified mass，能scale force。

### 7.4 vs Domain Randomization（Table 9, Appendix A.3.5）

CrossDex (https://arxiv.org/abs/2410.02479)用mass randomization在[0.5, 1.5]kg训练。

| Method | 117g | 206g | 324g |
|--------|------|------|------|
| CrossDex | 4/10 | 7/10 | 9/10 |
| D-REX | 9/10 | 10/10 | 9/10 |

测试mass全在[0.5,1.5]kg范围外（都更轻）。CrossDex对light object失败——DR对OOD mass敏感。D-REX通过explicit mass identification避开了这个限制。

### 7.5 Scaling performance（Figure 9）

Screwdriver这个高难度object上：
- 1-10 demos: 抓不稳
- 20+ demos: 稳定
- 5000 demos best

200-300 demos默认，2分钟/object训练。复杂object需要5000 demos，20分钟训练。Inference 0.5s/object pose。

---

## 8. Hardware setup（Appendix A.2.10）

用两个不同的hand，分工很clever：

**Allegro Hand**用于mass identification：
- 16 DoF
- 内置wiring，结构紧凑
- Low-torque actuation，clean contact dynamics
- 噪音低，gradient flow稳定

**LEAP Hand** (https://arxiv.org/abs/2309.06440) 用于grasping：
- 16 DoF
- 高torque，能抓重物
- Modular，3D-printed
- Tendon-driven，human-like
- Current control linearly maps to force
- 但有exposed wiring，对mass identification太noisy

Decoupling roles: Allegro做precise物理参数估计，LEAP做强力抓取。这是工程上的sweet spot。

Arm用Franka Emika Panda（7-DoF），camera是Intel RealSense D435i，第三视角。

---

## 9. 与GradSim的对比（Appendix A.9）

GradSim (https://arxiv.org/abs/2104.02646) 是 predecessors，D-REX的关键区别：

1. **GradSim需要rendering supervision**——photometric loss需要controlled lighting + calibrated cameras + 3D-printed objects。D-REX直接用FoundationPose给的6-DoF pose做state-space trajectory loss。
2. **GradSim假设full simulator state**。D-REX从partial noisy real observations估mass。
3. **D-REX把mass identification formulated成constrained optimization**：known robot inputs + accurate initial/boundary conditions → 找参数让FoundationPose trajectory最好reproduced。
4. D-REX还reuse了GradSim的differentiable rendering机制内部，但物理用MJX (Brax backend, https://github.com/google/brax)。

---

## 10. Cross-object generalization（Appendix A.5）

Table 8: 大screwdriver训出来的policy，换小screwdriver（10×3×3cm 600g → 7×2×2cm 500g），success rate 90% → 70%。Moderate degradation，说明within-category generalization可行。

也测了articulated object（stapler）和fine-grained task（computer mouse），都OK，证明框架不仅限于grasping。

---

## 11. Limitations

- 只学mass，没学friction/stiffness/damping（这些更难实验validate）
- 假设rigid-body dynamics
- Real-to-sim阶段后不能有人在scene里干预（保持Lagrangian dynamics consistency）
- Object-specific policy，不是fully general
- FoundationPose的z-axis误差有时需要manual post-processing

---

## 12. Intuition summary

整个D-REX的核心idea可以这样理解：

**Real world的物理过程**：robot施加action → object按真实physics运动 → 产生trajectory

**Simulator**：robot施加相同action → object按simulator physics运动 → 产生trajectory

如果simulator的physics参数对，两条trajectory应该match。Mass是最关键的scalar parameter，它直接出现在 $F = ma$ 里。Differentiable physics让你能从trajectory error反推到mass。

更妙的是，简化场景下trajectory是 $1/m$ 的affine function，使得loss变成convex quadratic——gradient descent有保证收敛到global minimum。

然后mass-aware force control就让policy能适应各种重量物体，不会因为固定force而对heavy object slip或对light object弹开。

---

## 13. 一些思考

- **为什么不用Force/torque sensor直接测mass？** 因为很多robot没装，且contact force本身也依赖contact geometry，不是简单的称重。Vision-based方法更scalable。
- **为什么push-down而不是grasp做mass identification？** Push-down接触简单，friction影响小（virtual fulcrum assumption），dynamics更clean。Grasp时多指接触复杂，gradient flow不好。
- **为什么VLM初始猜测off by 10×也能work？** 因为convex loss landscape对initial point不敏感。Appendix A.3.2证明从2g初始（100-350× off）也能converge。
- **为什么不直接end-to-end train everything？** Decouple mass identification和policy learning能isolate causal effect of mass（Appendix A.8）。如果end-to-end，你不知道performance gain来自mass accuracy还是policy capacity。
- **这种framework能扩展到friction吗？** 理论上可以，但需要contact-rich interaction让friction effect observable。Mass有gravity这个always-present observable signal，friction需要特定motion（sliding）才显形。
- **dual Gaussian Splat的insight**：appearance和geometry的optimum不align，分开优化避免互相干扰。这其实是multi-task learning的negative transfer问题的具体体现。

整体来说，D-REX是个非常elegant的系统工作：differentiable simulation的practical应用，配合mass-conditioned force control，解决了sim-to-real的一个concrete bottleneck。理论分析（1/m convexity）+ 工程实现（dual Gaussian, semi-implicit Euler, two-hand decoupling）+ 实验验证（mass accuracy + grasping success）都做得很完整。

参考链接：
- D-REX project: https://drex.github.io
- 3DGS: https://repo.samkolb.com/3dgaussians/assets/3dgaussians.pdf
- GradSim: https://arxiv.org/abs/2104.02646
- FoundationPose: https://arxiv.org/abs/2403.07715
- HaMeR: https://arxiv.org/abs/2402.09214
- MCC-HO: https://arxiv.org/abs/2404.06507
- Dex-Retargeting: https://arxiv.org/abs/2305.01692
- LEAP Hand: https://arxiv.org/abs/2309.06440
- Brax/MJX: https://github.com/google/brax
- Human2Sim2Robot: https://arxiv.org/abs/2504.12609
- DexGraspNet 2.0: https://arxiv.org/abs/2410.23004
- CrossDex: https://arxiv.org/abs/2410.02479
- Robo-GS (Lou et al. 2024): https://arxiv.org/abs/2408.14873
