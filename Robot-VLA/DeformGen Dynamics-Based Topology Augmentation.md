---
source_pdf: DeformGen Dynamics-Based Topology Augmentation.pdf
paper_sha256: 527d2f9d1d0ef50e89ebe75cba6ed05ee32f4086cd7dc2a636a783517e465ce9
processed_at: '2026-08-18T05:00:56-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DeformGen 人话版

Andrej，我换个讲法，像在白板上边画边聊那种感觉。

---

## 一句话说清楚这篇paper在干嘛

你有一根绳子、一个布娃娃、一块布，你teleop了一次robot怎么操作它们。你想要1000次这样的demo来训练policy，但不想再teleop了。

**Rigid body时代**的做法：把整个场景转一转、平移一下，robot的trajectory也跟着转一下平移一下——因为rigid body上任意两点距离不变，relative关系preserve，所以新trajectory依然valid。这就是MimicGen [1]、DemoGen [2]干的事。

**Deformable object的问题**：你把一根绳子整体旋转30度，绳子还是那根绳子，shape没变。但你真正想要的是绳子打了个结、弯了个圈、垂下来一截——这种**topological diversity**，global SE(3) transform给不了。

DeformGen的核心idea：**别假装自己比physics simulator更懂deformable object的valid state长什么样，让simulator自己去演化**。

---

## 第一个问题：怎么生成diverse且physically valid的deformed state

### 为什么naive方法不行

想象rope被表示成1000个particles，state space是 $\mathbb{R}^{3000}$。但valid state（绳子不断、不self-intersect、满足mass-spring constraint）只是其中一个非常thin的manifold $\mathcal{S}_{\text{real}}$。

**三种naive尝试及其失败原因**：

**尝试1：整体rigid transform**
把所有particle一起translate + rotate。Reachable set只有6个DoF（3 translation + 3 rotation）。绳子shape一点没变，只是位置变了。Fig. 8证明：non-rigid residual几乎为零。**你aug了个寂寞**。

**尝试2：每个particle独立加噪声**
每个particle $\mathbf{p}_i$ 加 $\epsilon_i \sim \mathcal{P}(\sigma)$。
- $\sigma$ 大：particle四处飞，绳子断了、self-intersect了，掉出 $\mathcal{S}_{\text{real}}$
- $\sigma$ 小：只产生表面小wrinkle [3]，diversity不够

而且即使你事后让simulator"stabilize"一下，也没法验证它真的回到 $\mathcal{S}_{\text{real}}$ 了——simulator可能converge到错误的basin。

**尝试3：用一个continuous deformation field $\phi: \mathbb{R}^3 \to \mathbb{R}^3$**
比如用一个neural network或B-spline去warp整个空间。Topological coherence preserve了（绳子还连着），但 $\phi$ 的构造不考虑material constitutive law——你deform出来的shape可能内部stress不合理，或者和环境interpenetrate。**Structurally coherent but dynamically inadmissible**。

### DeformGen的做法

$$\mathbf{s}_{\text{aug}} = \Phi_{\text{sim}}(\mathbf{s}_0, \mathbf{f}, \Delta t), \quad \mathbf{s}_0 \in \mathcal{S}_{\text{real}} \tag{1}$$

翻译成人话：从一个**已知valid的source state** $\mathbf{s}_0$ 出发，施加一个localized random force $\mathbf{f}$，让simulator $\Phi_{\text{sim}}$ forward simulate $\Delta t$ 时间，得到的新state就是augmented state。

**为什么这个work**：simulator在每个timestep的内部solver同时enforce所有constraint——material elasticity、self-collision、environment contact、gravity settling。所以只要起点valid，终点（在reasonable force和time内）自然stay in $\mathcal{S}_{\text{real}}$。

这有一个**asymmetry**（Appendix A.1的Assumption 1）：
- 从valid state演化 → 大概率stay valid（simulator帮你enforce constraint）
- 从invalid state出发 → simulator不保证能修回来（可能converge到错误basin或不converge）

所以DeformGen的哲学是**"never leave the manifold"**，而不是"leave然后repair"。

### 具体怎么施加force（Appendix C.3）

Simulator不暴露external-force API，所以作者用gripper本身当"搅棒"——让gripper在contact状态下做randomized Cartesian perturbation：

- **Rope/Toy**：180个random step，每step做 ±x/±y translation（magnitude ∈ {0.012, 0.006, 0.003} m）或 ±z rotation（±6°，probability $p_{\text{rot}}=0.45$）
- **Cloth**：260个random step，translation magnitude ∈ {0.018, 0.009, 0.0045} m，rotation ±8°，$p_{\text{rot}}=0.55$
- 之后stabilize 30-40步到quasi-static equilibrium

**Intuition**：这就像你用手去拨弄一根绳子或一块布——拨弄的方式random enough，能产生bending、twisting、folding、draping等各种deformation，但绳子始终是绳子（simulator保证的）。

### Reachable Set

$$\mathcal{R}(\mathbf{s}_0) = \left\{ \Phi_{\text{sim}}(\mathbf{s}_0, \mathbf{f}, \Delta t) \mid \mathbf{f} \in \mathbb{R}^{3N}, \Delta t > 0 \right\} \tag{8}$$

理论上，任何physically plausible configuration都和 $\mathbf{s}_0$ 通过某个physical process相连，所以 $\mathcal{R}(\mathbf{s}_0)$ 可以很大。但paper不claim full coverage，只claim是比rigid transformation explore更广的**practical sampling heuristic**。

---

## 第二个问题：有了deformed state，怎么把source trajectory transfer过去

这是paper的第二个核心contribution。你有了original rope的shape A，和augmented rope的shape B，source trajectory是在shape A上定义的——robot怎么move、怎么grasp。现在要让robot在shape B上做"等价"的事。

### 为什么rigid trajectory transfer不行

Rigid transfer = 对整条trajectory施加一个global SE(3) transform。问题：
- **Grasp misalignment**：source trajectory里gripper对准了rope的某个local region，rigid transform后gripper依然对准那个region的global位置，但rope已经deform了，那个region的local geometry变了，gripper抓空或抓错
- **无法补偿local deformation**：rigid transform只能整体平移旋转，无法让trajectory的不同segment适应object不同部分的deformation

### DeformGen的做法：Deformation-Field Warping

灵感来自Schulman et al. [4]的非rigid registration——从per-particle displacement构造continuous deformation field。

**Step 1：计算per-particle displacement**
- $\mathbf{p}_{\text{orig}} \in \mathbb{R}^{N \times 3}$：source object的N个particle
- $\mathbf{p}_{\text{def}} \in \mathbb{R}^{N \times 3}$：deformed object的N个particle
- $\delta_i = \mathbf{p}_{\text{def},i} - \mathbf{p}_{\text{orig},i}$：第i个particle的displacement

这是discrete的——只有N个点的displacement。但trajectory上的end-effector位置 $x_t$ 可能在这N个点之间，需要interpolate。

**Step 2：Position Warping via KNN + Inverse Distance Weighting**

对trajectory上每个waypoint $x_t$，在 $\mathbf{p}_{\text{orig}}$ 里找K个nearest neighbor，用距离倒数做权重：

$$w_{t,j} = \frac{1}{\|x_t - \mathbf{p}_{\text{orig}, \text{nn}_j(x_t)}\| + \varepsilon} \tag{2}$$

$$\tilde{w}_{t,j} = \frac{w_{t,j}}{\sum_j w_{t,j}}$$

$$d(x_t) = \sum_j \tilde{w}_{t,j} \cdot \delta_{\text{nn}_j(x_t)}$$

变量含义：
- $w_{t,j}$：第 $j$ 个neighbor的weight，越近weight越大
- $\varepsilon$：防止除零，numerical stability
- $\tilde{w}_{t,j}$：normalized后的weight，和为1
- $\text{nn}_j(x_t)$：$x_t$ 在source object里第 $j$ 个最近邻的index
- $d(x_t)$：$x_t$ 处的deformation vector——这是把discrete per-particle displacement **lift**成continuous spatial function的关键一步

**Intuition**：想象你有一堆箭头（per-particle displacement），现在你要在空间任意一点query"这里的deformation是多少"。KNN inverse distance weighting就是一个non-parametric的方式：看周围最近的K个箭头，按距离加权平均。这比fit一个parametric deformation field简单得多，不需要iterative optimization。

**Step 3：加time-dependent decay**

$$\mathbf{x}_t^{\text{warp}} = \mathbf{x}_t + \alpha_t \cdot d(\mathbf{x}_t) \tag{3}$$

$\alpha_t = \text{decay}(t)$ 控制deformation field的影响随时间衰减。三种option：
- **None**：$\alpha_t = 1$，全程uniform
- **Linear**：$\alpha_t = \max(0, 1 - t/T)$，$T$ = trajectory总长
- **Exponential**：$\alpha_t = e^{-\lambda t}$

**为什么需要decay**：trajectory早期（approach + grasp phase），robot要紧贴object的local geometry，deformation field的影响要大。trajectory后期（manipulation phase），object已经被robot manipulates了，再强行跟随initial deformation反而不对——object的状态已经变了，initial deformation field不再represent当前geometry。所以让influence逐渐revert到original path。

**Step 4：Orientation Adaptation via Local Jacobian**

Position搞定了，orientation呢？不能直接对rotation matrix做KNN——因为orientation是object local geometry的函数，不是空间位置的函数。

做法：在 $x_t$ 的KNN neighborhood内，构造local relative coordinates：

$$\ell_{t,j}^{\text{orig}} = \mathbf{p}_{\text{orig}, \text{nn}_j(x_t)} - \mathbf{x}_t \tag{4}$$

$$\ell_{t,j}^{\text{def}} = \ell_{t,j}^{\text{orig}} + \delta_{\text{nn}_j(x_t)}$$

即：source local vectors → deformed local vectors。

然后估计一个local Jacobian $J_t$，使得 $J_t \ell^{\text{orig}} \approx \ell^{\text{def}}$：

$$J_t = \arg\min_J \sum_j \|\ell_{t,j}^{\text{def}} - J \ell_{t,j}^{\text{orig}}\|^2 \tag{5}$$

Closed-form（设 $X_{\text{orig}}, X_{\text{def}}$ 是stacked local vectors）：

$$J_t = X_{\text{def}} X_{\text{orig}}^\top (X_{\text{orig}} X_{\text{orig}}^\top)^+ \tag{6}$$

$(\cdot)^+$ 是Moore-Penrose pseudoinverse。

**Intuition**：$J_t$ 是一个3×3矩阵，描述"local neighborhood从source geometry到deformed geometry的最佳linear approximation"。它capture了local的stretch、shear、rotation。如果rope在某个地方弯了30度，$J_t$ 会反映这个rotation。

**Step 5：投影到SO(3) + SLERP**

$J_t$ 是general 3×3矩阵，不一定是rotation。要把它变成rotation：
- 计算 $J_t R_t$（$R_t$ 是source orientation）
- SVD分解，project到 $SO(3)$ manifold，得到 $R_t'$
- 用SLERP在 $R_t$ 和 $R_t'$ 之间interpolate：

$$R_t^{\text{warp}} = \text{SLERP}(R_t, R_t', \alpha_t) \tag{7}$$

SLERP = Spherical Linear Interpolation，在rotation space上做球面线性插值。$\alpha_t$ 同position warping的decay，控制从original orientation到induced orientation的blend程度。

### KNN Scope的trick（Appendix B.2）

不同phase用不同K：
- **Grasp pose**：小K（5-10）。Grasp只和grasp region附近几个点相关，用local displacement
- **Manipulation trajectory**：$K = N$（全部点）。Manipulation要补偿global shape shift，需要所有particle的information

### Orientation Constraint（Appendix B.3）

Tabletop场景，主要rotation在Z-axis。所以先把 $R_t$ 和 $R_t'$ project到Z-axis rotational component再SLERP，避免其他axis上noisy Jacobian导致的spurious tilting。

---

## 实验讲了什么

### Setup
- Real2Sim-Eval [5] + PhysTwin [6] 做simulation
- xArm7，third-person + wrist camera，848×480，30Hz
- 三个task：rope routing、toy packing、cloth folding
- 四个policy：ACT [7]、Diffusion Policy [8]、SmolVLA [9]、π0 [10]（LoRA fine-tune）
- 每个task生成1200+ augmented state，split成1000 train / 200 test

### 主结果（Table 2）

四个regime对比，巧妙地用ablation隔离两个contribution：

| Policy | 1 Src. | SMG\* | DG\* | DG |
|--------|--------|-------|------|-----|
| ACT avg | 1.33 | 48.17 | 46.83 | **59.00** |
| DP avg | 2.33 | 38.00 | 41.00 | 37.33 |
| SmolVLA avg | 2.50 | 40.33 | 55.00 | **56.50** |
| π0 avg | 2.33 | 27.83 | 51.67 | **56.67** |

- **SMG\*** = rigid state aug + deformation-field warping（SoftMimicGen [11]的reimplementation）
- **DG\*** = topological state aug + local rigid transfer
- **DG** = full method

**两个insight**：
1. SMG\* vs DG → state augmentation的效果（rigid vs topological）。DG更高 → topological diversity重要
2. DG\* vs DG → trajectory transfer的效果（local rigid vs deformation-field warping）。DG often更高 → deformation-aware warping额外benefit

### State Coverage（Fig. 8）

用Procrustes alignment把每个state分解成rigid SE(3) component + non-rigid residual：
- Rigid augmentation（SMG\*）：cluster在source附近，non-rigid residual ≈ 0
- DeformGen：broadly spread，large non-rigid residual

**证明performance gain来自genuine topological diversity，不是more data at similar configurations**。

### Data Scaling（Table 4）

| N | ACT avg | SmolVLA avg |
|---|---------|-------------|
| 100 | 19.50 | 36.83 |
| 250 | 51.50 | 38.00 |
| 500 | 58.67 | 55.83 |
| 750 | 61.50 | 63.17 |

Monotonically increasing → dynamics-based augmentation能从scale中benefit。

### Hard Samples Generalization（Table 5）

最interesting的ablation：在**trajectory synthesis失败**的state上test policy。这些state，policy从未见过successful trajectory。

Rope排除（99.5% synthesis success），只看toy和cloth：

| Task | ACT SMG\* | ACT DG\* | ACT DG |
|------|-----------|----------|--------|
| Toy | 45.50 | 37.50 | 55.50 |
| Cloth | 11.00 | 6.00 | 5.50 |

Policy仍然有non-trivial success → **policy学的是transferable manipulation strategy，不是memorize specific trajectory**。有一定extrapolation能力。

### Synthesis Success Rate（Appendix C.4）

| Task | Generated | Successful | Rate |
|------|-----------|------------|------|
| Rope | 1300 | 1294 | 99.5% |
| Toy | 2200 | 1327 | 60.3% |
| Cloth | 4500 | 1778 | 39.5% |

Cloth最难——deformation更复杂，geometric correspondence更难preserve task semantics。

---

## 我对这篇paper的intuition总结

**Rigid body时代的augmentation是"group-theoretic"的**：用SE(3) equivariance保证transformed data依然valid。数学很漂亮，但只适用于rigid body。

**Deformable object的augmentation应该是"dynamics-based"的**：用physics simulator的forward dynamics作为valid state的implicit定义。Simulator能演化出来的state，by construction physically plausible（在simulator fidelity范围内）。

这个philosophical shift很elegant。你不再需要显式建模 $\mathcal{S}_{\text{real}}$ 长什么样——simulator本身就是 $\mathcal{S}_{\text{real}}$ 的implicit oracle。

Trajectory warping部分则是把discrete per-particle displacement "lift"成continuous spatial function $d(\mathbf{x})$，用KNN inverse distance weighting做non-parametric interpolation。这比fit一个parametric deformation field简单，不需要iterative optimization，而且locality可以tune（K的大小）。

**对policy learning的implication**：augmented data让policy见过更多topology variant，从而在unseen state上generalize。Hard sample ablation尤其reveal——policy甚至能在synthesis失败的state上成功，说明它学到的是更abstract的manipulation strategy，而非reproduce specific trajectory。

---

## References

- [1] MimicGen: https://mimicgen.github.io/
- [2] DemoGen: https://arxiv.org/abs/2502.16932
- [3] Interndata-A1: https://arxiv.org/abs/2511.16651
- [4] Schulman et al., Non-Rigid Registration: https://link.springer.com/chapter/10.1007/978-3-319-28872-7_20
- [5] Real2Sim-Eval: https://arxiv.org/abs/2511.04665
- [6] PhysTwin: https://phystwin.github.io/
- [7] ACT: https://tonyzhaozh.github.io/aloha/
- [8] Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- [9] SmolVLA: https://huggingface.co/blog/smolvla
- [10] π0: https://www.physicalintelligence.company/blog/pi0
- [11] SoftMimicGen: https://arxiv.org/abs/2603.25725
- DeformGen Project: https://zili2002.github.io/DeformGen
- DeformGen Code: https://github.com/Zili2002/DeformGen

---

# DeformGen: Deformable Manipulation的Data Augmentation Framework详解

Andrej，这篇paper解决的是一个很fundamental的问题：**如何把rigid body manipulation里的data augmentation范式扩展到deformable object**。我下面会从intuition、数学formulation、架构细节、实验数据几个维度展开。

---

## 1. Core Problem: 为什么rigid augmentation范式会失效

传统的augmentation方法（MimicGen [1], DemoGen [2], Real2Render2Real [3]）都依赖一个**equivariant assumption**：对rigid body施加一个global SE(3) transformation T，object和end-effector的relative pose保持不变，所以trajectory依然valid。

Deformable object把这个assumption彻底打破，paper识别出两个challenge：

### Challenge 1: State-Space Challenge

Rigid body的state是6-DoF pose，足够抽象。Deformable object被离散成N个particles，configuration space是：

$$\mathcal{S} = \mathbb{R}^{3N}$$

其中每个state $\mathbf{s} = (\mathbf{p}_1, \dots, \mathbf{p}_N) \in \mathcal{S}$ 指定所有particle的position。但**physically plausible subspace** $\mathcal{S}_{\text{real}} \subsetneq \mathcal{S}$，即 $\mathbb{R}^{3N}$ 里绝大多数点对应不了一个物理上可实现的状态（self-intersection、disconnected segments、unrealistic internal stress等）。

**Intuition**: 想象一根rope有1000个particles，state space是 $\mathbb{R}^{3000}$。但rope必须保持拓扑连通性、不能self-penetrate、要满足mass-spring constitutive law——这些constraint把valid state压在一个非常thin的manifold上。naive的per-particle perturbation几乎必然掉出这个manifold，而且simulator无法可靠地把它"修"回来。

### Challenge 2: Trajectory-Transfer Challenge (Non-Equivariance)

Deformable object的material points不再rigidly move together。对整条trajectory施加global rigid transform会导致：
- (i) Grasp pose和object的local geometry misalign，gripper抓不住
- (ii) 只能整体translate/rotate trajectory，无法补偿object的local deformation

---

## 2. Method Overview

DeformGen分两步：

1. **Dynamic Topological Transformation** (Section 3.1)：生成physically plausible的diverse states
2. **Deformation-Field Warping** (Section 3.2)：把source trajectory transfer到augmented state

用PhysTwin [4]做soft-body dynamics + rendering，Real2Sim-Eval [5]做benchmark。

---

## 3. State Augmentation: Dynamics-Based Topological Transformation

### 3.1 Core Insight

**Key insight**: physically plausible states构成 $\mathcal{S}_{\text{real}}$ 这个constrained manifold。不要"perturb然后repair"，而是**从一开始就不离开manifold**——从valid state出发，通过simulator自身的dynamics演化。

形式化：

$$\mathbf{s}_{\text{aug}} = \Phi_{\text{sim}}(\mathbf{s}_0, \mathbf{f}, \Delta t), \quad \mathbf{s}_0 \in \mathcal{S}_{\text{real}} \tag{1}$$

其中：
- $\Phi_{\text{sim}}$：physics simulator，输入state $\mathbf{s}$、external force field $\mathbf{f}$、time interval $\Delta t$，输出evolved state
- $\mathbf{s}_0$：source demonstration的valid初始state
- $\mathbf{f}$：localized force field（在object局部施加随机力）

### 3.2 为什么这个work——Assumption 1

Paper在Appendix A.1给出一个**Approximate conditional closure**假设：

$$\mathbf{s} \in \mathcal{S}_{\text{real}} \implies \Phi_{\text{sim}}(\mathbf{s}, \mathbf{f}, \Delta t) \approx \mathcal{S}_{\text{real}}, \quad \text{for reasonable } \mathbf{f}, \Delta t \tag{A1}$$

$$\mathbf{s} \notin \mathcal{S}_{\text{real}} \not\implies \Phi_{\text{sim}}(\mathbf{s}, \mathbf{f}, \Delta t) \in \mathcal{S}_{\text{real}} \tag{A2}$$

**Intuition**: simulator的内部solver在每个timestep同时enforce所有coupled constraint（material elasticity、self-collision、environmental contact、gravity），所以从valid state演化出来的state自然stay in $\mathcal{S}_{\text{real}}$。但从invalid state出发，simulator可能converge到错误的basin或根本不converge。这是一种**asymmetry**：preservation容易，restoration难。

### 3.3 三种Alternative Strategies的失败原因

Paper详细对比了三种naive方法（Appendix A.2）：

**(i) Global rigid transformation**：
施加uniform $\mathbf{T} \in SE(3)$到所有particle。Preserves plausibility，但reachable set只是 $\mathbf{s}_0$ 的6-DoF subspace，**capture不了任何shape/topology variation**。Fig. 8证明：rigid augmentation的non-rigid residual几乎为零。

**(ii) Per-particle perturbation**：
对每个particle加independent noise $\epsilon_i \sim \mathcal{P}(\sigma)$。能reach整个 $\mathcal{S}$，但面临**coverage-plausibility trade-off**：
- Large $\sigma$：break connectivity，self-intersection，掉出 $\mathcal{S}_{\text{real}}$，stabilization修不回来
- Small $\sigma$：只产生local wrinkles [6]，diversity不够

**(iii) Kinematic topological transformation**：
用continuous deformation field $\phi: \mathbb{R}^3 \to \mathbb{R}^3$。Preserves topological coherence，但 $\phi$ 的构造不考虑material model，deformed state可能violate internal dynamic constraint（unrealistic internal stress、interpenetration）。Structurally coherent但dynamically inadmissible。

**Table 1 总结**：

| Strategy | Coherence | Reachable set | $\subseteq \mathcal{S}_{\text{real}}$ | Recoverable? |
|----------|-----------|---------------|------|--------------|
| (i) Global rigid | ✓ | 6-DoF subspace | ✓ | N/A |
| (ii) Per-particle | ✗ | $\mathcal{S}$ | ✗ | Unreliable |
| (iii) Kinematic | ✓ | $\mathcal{S}$ | ✗ | Unreliable |
| (iv) Dynamics (Ours) | ✓ | $\mathcal{R}(\mathbf{s}_0) \subseteq \mathcal{S}_{\text{real}}$ | ✓ | N/A |

### 3.4 Reachable Set

$$\mathcal{R}(\mathbf{s}_0) = \left\{ \Phi_{\text{sim}}(\mathbf{s}_0, \mathbf{f}, \Delta t) \mid \mathbf{f} \in \mathbb{R}^{3N}, \Delta t > 0 \right\} \tag{8}$$

理论上任何physically plausible configuration都和 $\mathbf{s}_0$ 通过某个physical process相连，所以 $\mathcal{R}(\mathbf{s}_0)$ 可以很大。但paper不claim full coverage，只claim是一个**practical sampling heuristic**，比rigid transformation探索更广。

### 3.5 实现细节（Appendix C.3）

因为simulator不暴露external-force API，作者用gripper执行randomized Cartesian perturbation来transmit force：

- **Rope/Toy**: 180 random steps，translation magnitude ∈ {0.012, 0.006, 0.003} m，rotation ±6°，$p_{\text{rot}} = 0.45$
- **Cloth**: 260 random steps，translation magnitude ∈ {0.018, 0.009, 0.0045} m，rotation ±8°，$p_{\text{rot}} = 0.55$
- 每step：±x/±y translation 或 z-axis rotation
- 之后stabilize 30-40 steps到quasi-static equilibrium

---

## 4. Trajectory Augmentation: Deformation-Field Warping

这是paper的第二个核心contribution。给定source trajectory $\{(x_t, R_t)\}_{t=1}^T$ 和augmented object state，如何transfer trajectory使其适应deformed geometry。

灵感来自Schulman et al. [7]的非rigid registration思想——从per-particle displacement构造continuous deformation field，避免iterative optimization。

### 4.1 Position Warping

记source和deformed point cloud：
- $\mathbf{p}_{\text{orig}} \in \mathbb{R}^{N \times 3}$：source object particles
- $\mathbf{p}_{\text{def}} \in \mathbb{R}^{N \times 3}$：deformed object particles

Per-point displacement：
$$\delta_i = \mathbf{p}_{\text{def},i} - \mathbf{p}_{\text{orig},i}$$

对每个end-effector position $x_t$（timestep $t$），从 $\mathbf{p}_{\text{orig}}$ 中retrieve K nearest neighbors，用**inverse distance weighting**插值：

$$w_{t,j} = \frac{1}{\|x_t - \mathbf{p}_{\text{orig}, \text{nn}_j(x_t)}\| + \varepsilon} \tag{2}$$

$$\tilde{w}_{t,j} = \frac{w_{t,j}}{\sum_j w_{t,j}}$$

$$d(x_t) = \sum_j \tilde{w}_{t,j} \cdot \delta_{\text{nn}_j(x_t)}$$

变量含义：
- $w_{t,j}$：第 $j$ 个neighbor对 $x_t$ 的权重（距离的倒数）
- $\varepsilon > 0$：numerical stability，防止除零
- $\tilde{w}_{t,j}$：normalized权重
- $\text{nn}_j(x_t)$：$x_t$ 在 $\mathbf{p}_{\text{orig}}$ 中的第 $j$ 个nearest neighbor的index
- $d(x_t)$：$x_t$ 处的deformation vector（continuous deformation field在 $x_t$ 处的取值）

**Warped position**（带time-dependent decay）：

$$\mathbf{x}_t^{\text{warp}} = \mathbf{x}_t + \alpha_t \cdot d(\mathbf{x}_t) \tag{3}$$

其中 $\alpha_t = \text{decay}(t)$。三种decay option（Appendix B.1）：
- **None**: $\alpha_t = 1$，全程uniform应用
- **Linear**: $\alpha_t = \max(0, 1 - t/T)$，$T$是trajectory总长
- **Exponential**: $\alpha_t = e^{-\lambda t}$，$\lambda > 0$控制decay rate

**Intuition**: decay让trajectory在grasp phase（早期）紧贴local deformation，到manipulation phase后期逐渐revert到original path。这是因为后期object已经被manipulated，再强行跟随initial deformation反而不对。

### 4.2 Orientation Adaptation

Position warping只解决了"在哪里"，还要解决"朝哪个方向"。

构造local relative coordinates（在 $x_t$ 的KNN neighborhood内）：

$$\ell_{t,j}^{\text{orig}} = \mathbf{p}_{\text{orig}, \text{nn}_j(x_t)} - \mathbf{x}_t \tag{4}$$

$$\ell_{t,j}^{\text{def}} = \ell_{t,j}^{\text{orig}} + \delta_{\text{nn}_j(x_t)}$$

即：source local vectors → deformed local vectors。

**Local Jacobian estimation** via least squares：

$$J_t = \arg\min_J \sum_j \|\ell_{t,j}^{\text{def}} - J \ell_{t,j}^{\text{orig}}\|^2 \tag{5}$$

**Closed-form solution**：设 $X_{\text{orig}}$ 和 $X_{\text{def}}$ 是stacked local vectors的矩阵，则：

$$J_t = X_{\text{def}} X_{\text{orig}}^\top (X_{\text{orig}} X_{\text{orig}}^\top)^+ \tag{6}$$

其中 $(\cdot)^+$ 是Moore-Penrose pseudoinverse。

**Intuition**: $J_t$ 是一个affine map的最佳线性近似，描述local neighborhood如何从source geometry变换到deformed geometry。它捕获了local的stretch、shear、rotation。

**Induced rotation**：把 $J_t R_t$ 投影到 $SO(3)$ manifold（via SVD），得到 $R_t'$。

**Final warped orientation** via SLERP：

$$R_t^{\text{warp}} = \text{SLERP}(R_t, R_t', \alpha_t) \tag{7}$$

SLERP = Spherical Linear Interpolation，在 $SO(3)$ 上做球面线性插值，参数 $\alpha_t$ 控制从original orientation到induced orientation的blend程度。

### 4.3 KNN Scope策略（Appendix B.2）

不同phase用不同K：

- **Grasp pose**: 小K（5-10）。因为grasp pose只和grasp region附近的object points强相关，用local displacement就够了。
- **Manipulation trajectory**: $K = N$（所有points）。因为manipulation要补偿global shape shift，需要globally weighted deformation field。

### 4.4 Orientation Constraint（Appendix B.3）

Tabletop场景下，主要orientation变化在Z-axis（perpendicular to table）。所以把 $R_t$ 和 $R_t'$ 先project到Z-axis rotational component再做SLERP，避免其他axis上noisy Jacobian估计导致的spurious tilting/flipping。

### 4.5 Task-specific Hyperparameters（Appendix C.5）

| Task | Grasp KNN | Manipulation KNN | Decay |
|------|-----------|------------------|-------|
| Rope | K=5 | K=N | Linear |
| Toy | K=5 | K=N | None |
| Cloth | K=10 | K=N | Exponential (λ=0.02) |

---

## 5. Architecture Pipeline

整个pipeline（对应Fig. 1c, 1d）：

```
[Single Demo] 
    ↓
[State Augmentation] — dynamics rollout with randomized forces
    ↓ (1200+ augmented states per task)
[Trajectory Synthesis] — Deformation-Field Warping
    ↓ (attempt trajectory for each augmented state)
[Simulation Execution] — verify success in Real2Sim-Eval
    ↓ (filter successful episodes)
[Data Split] — 1000 train / 200 test / 200 hard samples
    ↓
[Policy Training] — ACT / DP / SmolVLA / π0 (LoRA)
    ↓
[Evaluation] — held-out states, including synthesis-failure states
```

---

## 6. Experiments

### 6.1 Setup

- **Simulator**: Real2Sim-Eval + PhysTwin
- **Robot**: xArm7，两个RGB camera（third-person + wrist，848×480，30Hz）
- **Tasks**:
  - **Rope routing**: 把rope穿过clip。Success criterion：最后100帧中至少30帧满足rope和clip上下平面有≥100个spring-segment crossing
  - **Toy packing**: 把stuffed toy放进container。Success：final frame，至少3050 object points落在scaled OBB（×1.05）内
  - **Cloth folding**: 把cloth折成三角形。Success：3-4 vertices，IoU≥0.72，coverage≥0.80
- **Policies**: ACT [8], Diffusion Policy [9], SmolVLA [10], π0 [11]（LoRA fine-tune）

### 6.2 Main Results (Table 2)

4个training regime对比：
- **1 Src.**: 单条source demo
- **SMG\***: SoftMimicGen reimplementation——rigid state aug + deformation-field warping（隔离state augmentation效果）
- **DG\***: topological state aug + local rigid transfer（隔离trajectory transfer效果）
- **DG**: full method

| Policy | 1 Src. | SMG* | DG* | DG |
|--------|--------|------|-----|-----|
| ACT avg | 1.33 | 48.17 | 46.83 | **59.00** |
| DP avg | 2.33 | 38.00 | 41.00 | 37.33 |
| SmolVLA avg | 2.50 | 40.33 | 55.00 | **56.50** |
| π0 avg | 2.33 | 27.83 | 51.67 | **56.67** |

DG在3/4 architecture上最高。

**两个insight**：
1. **Topological state diversity → generalization**：SMG* vs DG，都用deformation-field warping，但SMG*用rigid state perturbation，DG用dynamics-based。DG更高，说明broader deformable-state coverage重要。
2. **Deformation-field warping → complementary gains**：DG* vs DG，state aug相同但trajectory transfer不同。DG often > DG*，说明deformation-aware warping额外benefit。

### 6.3 State Coverage Analysis (Fig. 8)

用Procrustes alignment把每个augmented state分解为rigid SE(3) component + non-rigid residual：
- Rigid augmentation（blue）：cluster在source附近，non-rigid residual ≈ 0
- DeformGen（orange）：broadly spread，large non-rigid residual

这证明performance gain来自**genuine topological diversity**，不只是more data at similar configurations。

### 6.4 Ablation: Rigid-only Test (Table 3)

测试state只涉及rigid transformation时，DG是否hurt performance。SMG*在ACT/DP/SmolVLA上最高（符合预期，因为training分布match），但DG在π0上最高，且整体competitive。**说明topologically diverse training不会substantially compromise rigid场景的性能**。

### 6.5 Ablation: Data Quantity Scaling (Table 4)

| N | ACT avg | SmolVLA avg |
|---|---------|-------------|
| 100 | 19.50 | 36.83 |
| 250 | 51.50 | 38.00 |
| 500 | 58.67 | 55.83 |
| 750 | 61.50 | 63.17 |

ACT从19.50% → 61.50%，SmolVLA从36.83% → 63.17%，**monotonically increasing**，说明dynamics-based augmentation能从increased data scale中benefit。

### 6.6 Ablation: Hard Samples Generalization (Table 5)

测试policy在**trajectory synthesis失败**的state上的表现——这些state从未见过successful trajectory。Rope排除（99.5% synthesis success）。

| Task | ACT SMG* | ACT DG* | ACT DG |
|------|----------|---------|--------|
| Toy | 45.50 | 37.50 | 55.50 |
| Cloth | 11.00 | 6.00 | 5.50 |

Policy仍然能achieve non-trivial success，**说明policy学的是transferable manipulation strategy而非memorize individual demonstrations**，有一定extrapolation能力。

### 6.7 Synthesis Success Rate（Appendix C.4）

| Task | Generated states | Successful trajectories | Success rate |
|------|------------------|-------------------------|-------------|
| Rope | 1300 | 1294 | 99.5% |
| Toy | 2200 | 1327 | 60.3% |
| Cloth | 4500 | 1778 | 39.5% |

Cloth最难——deformation更复杂，geometric correspondence更难preserve task semantics。

### 6.8 Failure Modes (Fig. 9)

- Grasp misalignment on extreme deformations（visual appearance偏离training data太多）
- Premature release due to unstable contact under large deformations

---

## 7. 与SoftMimicGen [12]的关系

SoftMimicGen是closest prior work，也尝试解决trajectory-transfer challenge，但**state augmentation仍然是rigid的**——它的state distribution是"typically one with a larger set of possible placements for objects in the scene"，即SE(3) perturbation。所以SMG*在paper里作为ablation baseline，隔离state augmentation的效果。DeformGen的contribution是把state augmentation也从rigid提升到topological。

---

## 8. Limitations (Appendix D)

1. **Single-arm only**：没extend到bimanual/multi-robot
2. **Limited task diversity**：只测了rope/toy/cloth，没测dough shaping、surgical tissue、cable routing in clutter
3. **Sim-to-real gap**：全在simulation里，real transfer可能需要domain adaptation
4. **Trajectory synthesis不universal successful**：complex contact dynamics、large topological change、kinematic constraint会导致warped trajectory失败

---

## 9. 我的Intuition总结

这篇paper的核心贡献是把augmentation的**"equivariance assumption"**替换成**"dynamics consistency"**：

- Rigid body: 用group theory（SE(3) equivariance）生成valid augmented data
- Deformable body: 用physics simulator的dynamics演化生成valid augmented data

这个思路非常elegant，因为simulator本身就是 $\mathcal{S}_{\text{real}}$ 的implicit定义——任何simulator能演化出来的state都by construction physically plausible（在simulator的fidelity范围内）。

Trajectory warping部分则是把discrete per-particle displacement "lift"成continuous spatial function $d(\mathbf{x})$，用KNN inverse distance weighting做interpolation。这个idea来自Schulman et al. [7]的非rigid registration，但paper把它adapt到manipulation trajectory的position + orientation两个维度，并加了time-dependent decay。

**对policy learning的implication**：augmented data让policy见过更多topology variant，从而在unseen state上generalize。Hard sample ablation尤其interesting——policy甚至能在synthesis失败的state上成功，说明它学到的不是"reproduce specific trajectory"，而是更abstract的manipulation strategy。

---

## References (Web Links)

- [1] MimicGen: https://mimicgen.github.io/
- [2] DemoGen: https://arxiv.org/abs/2502.16932
- [3] Real2Render2Real: https://arxiv.org/abs/2505.09601
- [4] PhysTwin: https://phystwin.github.io/
- [5] Real2Sim-Eval: https://arxiv.org/abs/2511.04665
- [6] Interndata-A1: https://arxiv.org/abs/2511.16651
- [7] Schulman et al., Learning from Demonstrations through Non-Rigid Registration: https://link.springer.com/chapter/10.1007/978-3-319-28872-7_20
- [8] ACT: https://tonyzhaozh.github.io/aloha/
- [9] Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- [10] SmolVLA: https://huggingface.co/blog/smolvla
- [11] π0: https://www.physicalintelligence.company/blog/pi0
- [12] SoftMimicGen: https://arxiv.org/abs/2603.25725
- DeformGen Project Page: https://zili2002.github.io/DeformGen
- DeformGen GitHub: https://github.com/Zili2002/DeformGen

---

如果你想深入某个部分——比如Jacobian的SVD投影到SO(3)的细节、Procrustes alignment怎么做state decomposition、或者为什么cloth的synthesis success rate只有39.5%而rope有99.5%——我可以再展开。
