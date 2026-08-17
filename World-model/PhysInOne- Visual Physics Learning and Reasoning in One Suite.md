---
source_pdf: PhysInOne- Visual Physics Learning and Reasoning in One Suite.pdf
paper_sha256: 091b6c5cd221e197896813cdfe109d13365e535d7461726b515b3eb661c7d6cd
processed_at: '2026-08-06T03:33:41-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# PhysInOne 人话版

## 一句话概括

这帮人用Unreal Engine + Taichi + Blender造了一个超大的物理视频数据集，153K个3D场景，2M个视频，覆盖71种日常物理现象，然后拿来fine-tune现有的video generation模型，发现物理真实度确实提升了，但复杂场景还是搞不定。

---

## 为什么需要这个东西

现在的AI生成视频经常违反物理常识，ball往天上飞、物体突然变速、碰撞不守恒。为什么？因为训练数据里压根没有高质量的物理标注视频。以前的数据集要么只有几千个视频(Physion 24K, PhysTwin 66)，要么只有单一现象(ShapeStacks只管stacking)，要么物体都是简单几何体放在干净背景上。这种数据训练出来的模型当然不懂物理。

PhysInOne的思路很直接: 把scale拉满，2 million videos，比之前最大的数据集大两个数量级，同时保证每个视频都有完整的ground truth annotation (3D mesh、trajectory、物理参数、文字描述)。

---

## 数据怎么造的

整个pipeline其实不复杂，四步:

**第一步**: 从教材里挑71个basic physical phenomena，比如collision、gravity、buoyancy、magnetic attraction、laser reflection这些。然后把它们组合起来，单个的71个，两两组合943个，三三组合2270个，总共3284个"physical activities"。

**第二步**: 从Sketchfab、FAB、BlenderKit收集2231个3D objects (锤子、砖头、椅子、风扇、玻璃杯等)，623种materials (塑料、金属、木头、石头、布料)，528个backgrounds (客厅、卧室、工厂、游泳池)。

**第三步**: 在Unreal Engine里搭scene。每个activity平均用46.84个不同的scene来实现，换不同的物体、材质、背景。物体分solid、interactable (带关节)、destructible (能碎)、deformable (能变形)、granular (颗粒)、liquid六类。

**第四步**: 跑物理引擎模拟，12个固定相机 + 1个移动相机拍视频，同时生成depth、segmentation、trajectory等annotation。

---

## 三个物理引擎各管一摊

这个数据集最实在的地方在于用了三套不同的simulation:

**Chaos Physics (UE5自带)**: 管rigid body collision、fracture destruction。像ball落地弹跳、玻璃碎裂、跷跷板平衡这些。还额外实现了wind force (力随距离平方衰减) 和magnetic force (用dipole field公式 $\mathbf{B}(\mathbf{p}) = \frac{\mathbf{r}_N}{\|\mathbf{r}_N\|^3} - \frac{\mathbf{r}_S}{\|\mathbf{r}_S\|^3}$)。

**Doriflow (Blender插件)**: 专管液体，用SPH方法。pressure force和viscous force都是标准SPH公式，smoothing kernel那套。主要做Newtonian fluid和non-Newtonian fluid。

**Taichi MPM**: 管deformable和granular，这套最硬核。实现了五种本构模型: neo-Hookean (elastic solid)、StVK+von Mises (plasticine)、J-based viscous (Newtonian fluid)、viscoplastic (non-Newtonian)、Drucker-Prager (granular)。每种模型都有自己的yield criterion和return mapping。整套P2G → grid update → boundary handling → G2P的MPM标准流程，用voxelized SDF做复杂mesh的collision。

---

## PMF这个metric是这篇paper的灵魂

以前评估生成视频用FVD，基于Inception feature算Fréchet distance。问题是FVD主要看semantic content，两个motion完全不同但content相似的视频FVD可能很低。用VLM做评判也不靠谱，VLM本身就不懂物理。

PMF的insight是: **motion dynamics的本质信息在Fourier domain的energy spectrum里**。对视频做3D DFT (空间x、y + 时间t)，得到spatiotemporal frequency domain，然后算normalized energy distribution。

关键数学性质: 如果把视频在spatiotemporal上shift一下，或者改变brightness，normalized energy spectrum完全不变。只有phase受shift影响。所以比较两个视频的energy spectrum，就是在比较它们的motion pattern，自动decouple掉了initial position和brightness这些无关因素。

具体公式: $\text{PMF} = -\ln d_{TV}(E^{\text{gen}}, E^{\text{ref}})$，$d_{TV}$是total variation distance。

paper里那个toy example很直观: 两个ball一个从左边落一个从右边落，PMF是正无穷(完美匹配)；一个上一个下，PMF掉到0.158；一个快落一个慢落，PMF是0.322。red sphere和blue cube做相同motion，PMF依然高分。

---

## 四个应用实验

### 1. Video Generation Fine-tuning

拿SVD、CogVideoX、Wan2.2三个模型，用83K video-text pair做fine-tuning，试了LoRA、SFT、FLT三种方法。

结果: SFT最有效。SVD的PMF从2.753提到3.147，Wan2.2从2.041提到2.978，human rating从2.26提到5.95。说明PhysInOne确实能给模型注入物理知识。

有意思的是按物理类型分: magnetism和fluid的PMF普遍比mechanics和optics高，说明不同物理现象的建模难度差异很大。

### 2. Future Frame Prediction

分long-term (预测2.6秒后，78帧) 和short-term (实时预测下一10帧)。

4D方法 (TiNeuVox、DefGS、FreeGave、TRACE) 在seen view上PSNR约22-23，novel view掉到15-18，5-7dB的gap。这说明3D物理动力学建模的view generalization是个大问题。

MAGI-1作为video generation model反而最强，PMF 4.086，PSNR 23.14。可能因为它是大规模pretrained的，泛化能力更好。

### 3. Physical Properties Estimation

用PAC-NeRF和GIC估计Young's modulus、Poisson ratio、viscosity、yield stress这些物理参数。

结果: 误差很大。Elastic solids的$\log_{10}(E)$误差PAC-NeRF是117±68，GIC是16±1.36。Newtonian fluid的$\log_{10}(\mu)$误差GIC是70±53。这些数字说明在复杂背景+复杂物体下，现有方法基本估不准。

Resimulation (用估计参数重新模拟) 的PSNR: PAC-NeRF 24.12，GIC 26.90。GIC (3DGS-based)比PAC-NeRF (NeRF-based)好约2-3dB，3DGS在dynamic scene重建上确实有优势。

### 4. Motion Transfer

用MotionPro和GoWithTheFlow把一个视频的motion transfer到另一个scene的static image上。

结果: PMF只有3.3-3.5，比fine-tuned generation model (3.0-4.0)还低。LPIPS反而高 (0.41-0.47)。说明现有motion transfer方法虽然能保持visual fidelity，但无法准确transfer复杂的multiphysics multiobject motion pattern。

---

## 核心takeaway

**数据scale确实重要**。从20K到2M，两个数量级的提升，让fine-tuning foundation model成为可能，物理fidelity确实提升了。

**合成数据是可行路径**。虽然simulator不完美，但error是controlled的，而且能提供真实视频不可能获得的精确annotation。这对物理property estimation这种task特别关键。

**PMF这个metric设计得很好**。Fourier energy spectrum的shift-invariance和brightness-invariance恰好对应了"motion pattern vs initial state"的decoupling，比FVD合理多了。

**现有方法离solved还很远**。novel view的5-7dB PSNR gap，physical property estimation的大误差，motion transfer的低PMF，都说明physics-grounded world model还是open problem。PhysInOne的价值在于提供了足够challenging的benchmark来暴露这些limitation。

**MPM那套本构模型实现是真功夫**。五种材料的yield criterion和return mapping，从neo-Hookean到Drucker-Prager，覆盖了连续介质力学的主要模型，这个工程量不小。

---

# PhysInOne: Visual Physics Learning and Reasoning in One Suite 深度技术解析

## 一、核心定位与motivation

PhysInOne是一个大规模合成数据集，核心目标是解决visual physics learning领域长期存在的**数据稀缺瓶颈**。现有数据集如Physion(24K videos)、CLEVRER(10K)、ComPhy(12K)、Physion++(18K)、VideoPhy(11.3K)等，规模普遍在数千到数万量级，且局限于单一物理现象、简化物体形状、干净背景。PhysInOne将规模提升到**2 million videos / 153,810 scenes / 71 phenomena**，是之前最大数据集的两个数量级以上。

paper引用了Richard Feynman的名言 "What I cannot create, I do not understand" 作为future frame prediction task的哲学基础。核心insight在于：要让AI真正理解物理世界，必须能够预测、生成、推理符合物理规律的视频内容，而非仅仅生成视觉上plausible的像素序列。

参考文献链接：
- PhysInOne项目主页: https://vlar-group.github.io/PhysInOne.html
- 相关baseline: [Physion](https://arxiv.org/abs/2111.00116), [PAC-NeRF](https://arxiv.org/abs/2306.10721), [GIC](https://arxiv.org/abs/2407.16561)

---

## 二、数据集构建pipeline详解

### 2.1 物理现象分类 (71 phenomena)

基于Halliday《Fundamentals of Physics》教材，覆盖四大领域：

| 领域 | 典型现象举例 | 涉及的物理定律 |
|------|-------------|--------------|
| Mechanics (1-32, 40-50) | collision, gravity, friction, spring, seesaw, buoyancy | Newton's Laws, Momentum Conservation, Hooke's Law, Torque |
| Optics (33-39) | laser reflection, mirror sweep, light obstruction | Law of Reflection |
| Fluid Dynamics (51-66, 69-70) | buoyancy, fluid transfer, viscous flow, surface tension | Navier-Stokes, Mass Conservation |
| Magnetism (13-14) | magnetic attraction/repulsion | Dipole field formulation |

组合逻辑：71个basic phenomena作为"原子"，组合成single/double/triple-physics activities:
- Single: 71个
- Double: C(71,2)中筛选出943个有物理意义的组合
- Triple: 筛选出2270个
- **总计3284个multiphysics activities**

### 2.2 3D Asset收集

| 类别 | 数量 | 来源 |
|------|------|------|
| 3D Objects | 2,231 objects / 163 categories | Sketchfab, FAB, BlenderKit |
| Materials | 623 (plastic/metal/wood/stone/fabric) | ShareTextures, FAB |
| Backgrounds | 528 (indoor+outdoor) | - |

物体类型细分:
- **Solid objects**: hammer, brick等刚性物体
- **Interactable objects**: 带关节的物体(风扇、门、跷跷板)，用UE Blueprint + Physical Constraint实现，支持Linear/Angular Limit/Motor四种约束
- **Destructible objects**: 可破碎物体(玻璃)，用UE Geometry Collection + Apex Destruction，通过procedural fracture分chunk
- **Deformable objects**: 可变形物体，用MPM模拟
- **Granular objects**: 颗粒物(沙子)，用MPM
- **Liquid**: 液体，用SPH

### 2.3 物理引擎与模拟方法

#### Chaos Physics (UE5)
负责rigid body dynamics、fracture destruction。特殊效果实现:
- **Wind Force**: rectangular force field，力大小随距离二次衰减 $\|\mathbf{F}\| \propto 1/d^2$
- **Laser Interaction**: 通过collision detection确定intersection，提取surface normal后按反射定律计算反射方向
- **Magnetic Force**: 用dipole field formulation建模:

$$\mathbf{B}(\mathbf{p}) = \frac{\mathbf{r}_N}{\|\mathbf{r}_N\|^3} - \frac{\mathbf{r}_S}{\|\mathbf{r}_S\|^3}$$

其中$\mathbf{r}_{N,S} = \mathbf{p} - \mathbf{p}_{N,S}$，$\mathbf{p}$是空间任意点，$\mathbf{p}_N, \mathbf{p}_S$是磁铁南北极位置。通过在target magnet的poles采样B场，聚合得到translational force和rotational torque。

#### Doriflow (Blender + SPH)
用于Newtonian/non-Newtonian fluid simulation。核心SPH公式:

**Pressure Force**:
$$\mathbf{F}_{press,i} = -\sum_j m_i m_j \frac{p_i + p_j}{2\rho_i \rho_j} \nabla W(\mathbf{r}_{ij}, h)$$

- $i, j$: particle indices
- $m_i, m_j$: particle masses
- $p_i, p_j$: pressures at particles
- $\rho_i, \rho_j$: densities
- $W(\mathbf{r}_{ij}, h)$: smoothing kernel，$h$是smoothing length
- $\mathbf{r}_{ij}$: particle间距离向量

**Viscous Force**:
$$\mathbf{F}_{visc,i} = \sum_j \frac{\mu}{2}\left(\frac{\mathbf{v}_i - \mathbf{v}_j}{\rho_i + \rho_j}\right) \cdot \nabla^2 W(\mathbf{r}_{ij}, h)$$

- $\mu$: dynamic viscosity coefficient
- $\mathbf{v}_i, \mathbf{v}_j$: velocity vectors
- $\nabla^2 W$: Laplacian of smoothing kernel

#### Taichi MPM
实现完整的Material Point Method pipeline，支持五类材料的本构模型:

**Elastic Solids (neo-Hookean)**:
$$J\mathbf{T}(\mathbf{F}) = \mu(\mathbf{F}\mathbf{F}^\top) + (\lambda \log J - \mu)\mathbf{I}$$

- $\mathbf{F}$: deformation gradient (3×3 matrix)
- $J = \det(\mathbf{F})$: Jacobian determinant (体积变化率)
- $\mathbf{T}$: Cauchy stress tensor
- $\mu, \lambda$: Lamé parameters, 与Young's modulus $E$ 和Poisson ratio $\nu$的关系:

$$\mu = \frac{E}{2(1+\nu)}, \quad \lambda = \frac{\nu E}{(1+\nu)(1-2\nu)}$$

**Plasticine (StVK + von Mises)**:
通过SVD分解 $\mathbf{F} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$，定义Hencky strain $\boldsymbol{\epsilon} = \log\boldsymbol{\Sigma}$:

$$J\mathbf{T}(\mathbf{F}) = \mathbf{U}(2\mu\boldsymbol{\epsilon} + \lambda\text{tr}(\boldsymbol{\epsilon})\mathbf{I})\mathbf{U}^\top$$

von Mises yield measure:
$$\delta\gamma = \|\hat{\boldsymbol{\epsilon}}\| - \frac{\tau_Y}{2\mu}$$

- $\hat{\boldsymbol{\epsilon}}$: normalized Hencky strain
- $\tau_Y$: yield stress
- $\delta\gamma \leq 0$: elastic state, $Z(\mathbf{F}) = \mathbf{F}$
- $\delta\gamma > 0$: plastic state, return mapping:

$$Z(\mathbf{F}) = \mathbf{U}\exp\left(\boldsymbol{\epsilon} - \delta\gamma\frac{\hat{\boldsymbol{\epsilon}}}{\|\hat{\boldsymbol{\epsilon}}\|}\right)\mathbf{V}^\top$$

**Newtonian Fluids (J-based + viscous)**:
$$J\mathbf{T}(\mathbf{F}) = \frac{1}{2}\mu(\nabla\mathbf{v} + \nabla\mathbf{v}^\top) + \kappa(J - J^{-6})$$

- $\mathbf{v}$: velocity field
- $\kappa$: bulk modulus
- $J^{-6}$项用于体积惩罚

**Non-Newtonian Fluids (viscoplastic)**:
引入plastic viscosity $\eta$:
$$\hat{\mu} = \mu\frac{\text{tr}(\boldsymbol{\Sigma}^2)}{d}, \quad \mathbf{s} = 2\mu\hat{\boldsymbol{\epsilon}}, \quad \hat{s} = \|\mathbf{s}\| - \frac{\delta\gamma}{1 + \frac{\eta}{2\hat{\mu}\Delta t}}$$

- $d$: spatial dimension
- $\Delta t$: time step
- 当$\eta = 0$退化为plasticine模型

**Granular (Drucker-Prager)**:
yield criterion:
$$\text{tr}(\boldsymbol{\epsilon}) > 0 \quad \text{or} \quad \delta\gamma = \|\hat{\boldsymbol{\epsilon}}\|_F + \alpha\frac{(d\lambda + 2\mu)\text{tr}(\boldsymbol{\epsilon})}{2\mu} > 0$$

$$\alpha = \sqrt{\frac{2}{3}}\frac{2\sin\theta_{\text{fric}}}{3 - \sin\theta_{\text{fric}}}$$

- $\theta_{\text{fric}}$: friction angle

MPM pipeline核心步骤:
1. **P2G (Particle to Grid)**: 粒子属性(质量、动量、APIC affine terms)通过quadratic interpolation转移到grid nodes
2. **Grid Update**: normalize velocities, apply external forces (gravity)
3. **Boundary Handling**: 支持axis-aligned bounding box, analytic primitives, complex mesh collider (voxelized SDF)
4. **G2P (Grid to Particle)**: 更新粒子位置、速度、affine fields

### 2.4 相机配置

| 配置 | 参数 |
|------|------|
| Static cameras | 12个，均匀分布在upper hemisphere, elevation 30°~60° |
| Moving camera | 1个，三种trajectory策略 |
| Resolution | 1120×1120 |
| FPS | 30 (laser scenes: 60) |
| Average duration | 5.2 seconds |

三种monocular camera trajectory采样策略:
1. **Linear Drift Sampling**: 沿longitude均匀间隔180°，latitude带random perturbation
2. **Sinusoidal Interpolation Sampling**: 起止点之间用sin函数插值，更平滑
3. **Circular Loop Trajectory**: 球面上的圆形loop轨迹

### 2.5 Annotations五大类

1. **Geometry**: depth maps (1120×1120, meters), 3D meshes
2. **Semantics**: segmentation maps (object ID from 1, background=0)
3. **Motion**: 3D trajectories (position + rotation per frame), dense optical flow for rigid objects
4. **Physical Properties**: JSON格式存储dynamic friction, static friction, density, restitution
5. **Text**: 人工标注+Qwen3-VL-235B-A22B-Thinking润色，平均64 English words/scene

---

## 三、PMF (Physical Motion Fidelity) Metric - 数学推导

这是paper最重要的技术贡献之一。传统FVD只评估visual realism，VLM-based metrics缺乏物理理解能力，QA-based metrics无法定量评估motion correctness。

### 3.1 3D DFT定义

对视频 $\mathcal{V}(c; h, w, t)$ (color channel $c \in \{0,1,2\}$, spatial $h \in [0,H-1]$, $w \in [0,W-1]$, temporal $t \in [0,T-1]$):

$$\tilde{\mathcal{V}}(c; u, v, s) = \sum_{h,w,t} \mathcal{V}(c; h, w, t) \cdot e^{-2\pi i\left(\frac{uh}{H} + \frac{vw}{W} + \frac{st}{T}\right)}$$

- $u, v$: spatial frequency indices
- $s$: temporal frequency index
- $(u, v, s)$共同构成spatiotemporal frequency domain

### 3.2 Amplitude, Phase, Energy定义

$$A_{c;u,v,s} = |\tilde{\mathcal{V}}(c; u, v, s)|$$
$$\psi_{c;u,v,s} = \arctan\left(\frac{\text{Im}(\tilde{\mathcal{V}}(c;u,v,s))}{\text{Re}(\tilde{\mathcal{V}}(c;u,v,s))}\right)$$
$$E_{u,v,s} = \frac{\sum_{c=0}^{2}|\tilde{\mathcal{V}}(c;u,v,s)|^2}{\sum_{u',v',s'}\sum_{c=0}^{2}|\tilde{\mathcal{V}}(c;u',v',s')|^2}$$

$E_{u,v,s}$是**normalized energy spectrum**，所有频率分量能量之和为1。

### 3.3 不变性证明

**Spatiotemporal shift invariance**: 对shifted video $\mathcal{V}'(c;h,w,t) = \lambda\mathcal{V}(c;h+\delta h, w+\delta w, t+\delta t)$ (假设duplicate padding):

$$\tilde{\mathcal{V}}'(c;u,v,s) = \lambda e^{i\delta\psi}\tilde{\mathcal{V}}(c;u,v,s)$$

其中 $\delta\psi = 2\pi(\frac{u\delta h}{H} + \frac{v\delta w}{W} + \frac{s\delta t}{T})$。

由此:
- Amplitude: $A' = \lambda A$ (受brightness scaling影响，不受shift影响)
- **Normalized Energy: $E' = E$** (完全invariant to shift和brightness)
- Phase: $\psi' = \psi + \delta\psi$ (受shift影响，不受brightness影响)

### 3.4 PMF定义

$$\text{PMF}(\mathcal{V}_{\text{gen}}, \mathcal{V}_{\text{ref}}) = -\ln d_{TV}(E^{\text{gen}}, E^{\text{ref}})$$

$$d_{TV}(E^{\text{gen}}, E^{\text{ref}}) = \frac{1}{2}\sum_{u,v,s}|E^{\text{gen}}_{u,v,s} - E^{\text{ref}}_{u,v,s}|$$

- $d_{TV}$: total variation distance，衡量两个概率分布差异
- PMF越大，motion pattern越相似
- PMF对初始spatiotemporal state不敏感，专注于motion dynamics本身

### 3.5 PMF的intuition (Figure 19 toy examples)

| 场景 | PMF值 | 物理含义 |
|------|-------|---------|
| 两个ball分别从左右落下 | $+\infty$ | 相同motion pattern，不同initial position → 完美匹配 |
| 一个下落，一个上升 | 0.158 | 完全反向的motion → 能量谱严重mismatch |
| 一个fast fall，一个slow fall | 0.322 | 相同方向不同速度 → temporal frequency分布不同 |
| Red sphere vs Blue cube (相同motion) | 高分 | 视觉appearance不同但motion一致 → 高相似度 |

---

## 四、四大应用任务实验详解

### 4.1 Physics-aware Video Generation

**Models**:
- SVD-XT (U-Net based I2V)
- CogVideoX-1.5-5B (Transformer TI2V, concat conditioning)
- Wan2.2-5B (Transformer + flow matching, cross-attention conditioning)

**Fine-tuning strategies**:
- LoRA: rank=32, $\alpha=1.0$
- SFT: full parameter tuning
- FLT: 只调最后2个transformer blocks (~400MB trainable)

**Training**: 83,650 text-video pairs, lr=1e-4, batch=64

**Results (Table 2)**:

| Model | PMF↑ | FVD↓ | Rating↑ |
|-------|------|------|---------|
| SVD | 2.753 | 203 | 6.09 |
| SVD_sft | **3.147** | 143 | 6.08 |
| CogVideoX | 2.877 | 165 | 2.98 |
| CogVideoX_lora | 2.869 | 149 | 2.95 |
| Wan2.2-5B | 2.041 | 258 | 2.26 |
| Wan2.2-5B_sft | **2.978** | 190 | 5.95 |

**Key findings**:
- SFT普遍比LoRA/FLT更有效，特别是在Wan2.2上rating从2.26提升到5.95
- 各模型在magnetism和fluid上PMF较高，mechanics和optics较低 → 反映不同物理类型难度差异
- Human rating与PMF高度correlated，验证metric validity

### 4.2 Future Frame Prediction

#### 4.2.1 Long-term (~2.6s, 78 frames ahead)

**4D Modeling methods** (scene-specific training):
- TiNeuVox, DefGS, FreeGave, TRACE
- 使用10个view训练，测试seen/novel view

**Video Prediction methods**:
- ExtDM (训练在83K videos)
- MAGI-1 (pretrained, 16 steps, 32 condition frames)

**Results (Table 3)**:

| Method | PMF (seen/novel) | PSNR (seen/novel) |
|--------|-----------------|------------------|
| TiNeuVox | 3.710/2.885 | 21.49/15.20 |
| DefGS | 3.980/3.347 | 22.85/17.95 |
| FreeGave | 3.897/3.265 | 22.57/17.75 |
| TRACE | 3.869/3.242 | 22.42/17.44 |
| ExtDM | 3.363/- | 19.55/- |
| MAGI-1 | **4.086**/- | **23.14**/- |

**Insight**: 4D methods在seen view上PSNR约22-23，novel view下降到15-18，gap约5-7dB，反映3D物理动力学建模的view generalization困难。

#### 4.2.2 Continuous Short-term (next 10 frames, real-time)

**Results (Table 4)**:

| Method | PMF (seen/novel) | PSNR (seen/novel) |
|--------|-----------------|------------------|
| DefGS | 4.536/3.728 | 26.02/20.92 |
| FreeGave | **4.742**/3.706 | **27.09**/20.80 |
| ExtDM | 3.774/- | 22.14/- |
| MAGI-1 | 4.696/- | 26.75/- |

Short-term比long-term PSNR高约4dB (26 vs 22)，符合预期。

### 4.3 Physical Properties Estimation

**Baselines**: PAC-NeRF (NeRF-based), GIC (3DGS-based)
**Test set**: 20 scenes (每类材料4个), test-tiny

**Estimated parameters accuracy (Table 5)**:

| Material | Method | Parameters |
|----------|--------|-----------|
| Elastic Solids | PAC-NeRF | $\log_{10}(E)$: 117.18±68.44, $\nu$: 14.26±7.94 |
| | GIC | $\log_{10}(E)$: 16.35±1.36, $\nu$: 3.32±1.34 |
| Plasticine | PAC-NeRF | $\log_{10}(E)$: 68.38±44.14, $\tau_Y$: 25.51±18.34 |
| Newtonian Fluids | GIC | $\log_{10}(\mu)$: 70.07±53.44, $\log_{10}(\kappa)$: 287.56±194.98 |
| Granular | PAC-NeRF | $\theta_{\text{fric}}$: 16.87±27.36 |

**Resimulation结果 (Table 6)**:

| Method | PMF↑ | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|------|-------|-------|--------|
| PAC-NeRF | 5.617 | 24.12 | 0.942 | 0.086 |
| GIC | **5.938** | **26.90** | 0.950 | **0.074** |

GIC (3DGS-based)整体优于PAC-NeRF (NeRF-based)，特别在Newtonian/Non-Newtonian fluids上PSNR gap约4-5dB。Granular substances两类方法都较差 (SSIM 0.86)，反映离散颗粒物质的建模挑战。

### 4.4 Motion Transfer

**Baselines**: MotionPro (optical flow analyzer + I2V), GoWithTheFlow (noise warping in diffusion)

**Results (Table 7)**:

| Method | PMF↑ | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|------|-------|-------|--------|
| GoWithTheFlow | 3.309 | 18.98 | 0.691 | 0.410 |
| MotionPro | 3.484 | 20.28 | 0.775 | 0.467 |

PMF值约3.3-3.5，远低于video generation任务的fine-tuned model (3.0-4.0)，说明现有motion transfer方法无法处理multiphysics multiobject的复杂motion pattern。LPIPS反而较高(0.41-0.47)，反映visual fidelity与physical fidelity的trade-off。

---

## 五、与现有数据集的对比 (Table 1)

| Dataset | Phenomena | Scenes | Videos | Multi-physics | Multi-object | Complex BG |
|---------|-----------|--------|--------|---------------|--------------|------------|
| Physics101 | 4 | - | 17K | ✗ | ✗ | ✗ |
| ShapeStacks | 1 | 20K | 20K | ✗ | ✓ | ✗ |
| CLEVRER | 1 | 10K | 10K | ✗ | ✓ | ✗ |
| Physion | 8 | - | 24K | ✗ | ✓ | ✗ |
| Physion++ | 9 | - | 18K | ✓ | ✓ | ✗ |
| PAC-NeRF | 1 | 9 | 99 | ✗ | ✗ | ✗ |
| VideoPhy | 1 | 4 | 11.3K | ✓ | ✓ | ✓ |
| PhysTwin | 1 | 22 | 66 | ✗ | ✓ | ✗ |
| IntPhys2 | 1 | 344 | 1.4K | ✗ | ✓ | ✓ |
| Physics-IQ | 1 | 66 | 396 | ✗ | ✓ | ✓ |
| NewtonGen | 12 | - | 1.2K | ✗ | ✗ | ✓ |
| **PhysInOne** | **71** | **153K** | **2M** | **✓** | **✓** | **✓** |

PhysInOne在所有维度上都是orders of magnitude larger，唯一同时满足multiphysics + multiobject + complex background + comprehensive annotations的数据集。

---

## 六、技术细节的几个关键intuitions

### 6.1 为什么用合成数据而非真实视频

paper引用[32, 60]论证现有simulators的errors被thoroughly studied and controlled，且合成数据能提供精确ground truth (3D geometry, physical properties, trajectories)，这些在真实视频中几乎不可能获得。reference [100] (Synthetic Video Enhances Physical Fidelity)也验证了合成数据提升物理fidelity的有效性。

### 6.2 为什么PMF优于FVD

FVD基于Fréchet distance在Inception-V3 feature space计算，主要捕获semantic content而非motion dynamics。两个motion完全不同但content相似的视频FVD可能很低，PMF通过能量谱分析避免了这个问题。Figure 19的toy example清晰展示了PMF对motion pattern的敏感性。

### 6.3 数据划分策略

train/val/test = 8:1:1，关键约束:
- 每个3D asset只出现在一个partition (防止data leakage)
- 每个physical phenomenon在三个partition中比例近似保持
- Test sets进一步细分: test-small (772), test-mini (103), test-tiny (20)

### 6.4 物理参数的diversity (Table 9)

| Parameter | Range |
|-----------|-------|
| Friction Coefficient | 0~0.9 |
| Restitution | 0.1~0.8 |
| Density | 0.1~21.3 |
| SPH Viscosity | 0~5.0 |
| Yield Stress | 3~10 |
| Friction Angle | 15~60° |

---

## 七、Limitations与future directions

1. **Simulator fidelity**: Chaos Physics/MPM/SPH并非完美物理，存在known errors
2. **4 domains only**: 省略thermodynamics (需要temperature)和acoustics (需要sound)
3. **Scale of fine-tuning**: 只用83K subset，全量2M training仍需探索
4. **Novel view generalization**: 所有4D methods在novel view上PSNR下降5-7dB，gap明显
5. **Complex physical reasoning**: Magnetism (PMF 3.4)和Fluid (PMF 3.3)相对容易，Mechanics (2.5)和Optics (2.6)更难
6. **Motion transfer**: 现有方法PMF约3.3-3.5，远未solved

---

## 八、相关参考链接

- **Project page**: https://vlar-group.github.io/PhysInOne.html
- **UE Chaos Physics**: https://docs.unrealengine.com/5.0/en-US/chaos-physics-in-unreal-engine/
- **Taichi Lang**: https://taichi-lang.org/
- **Doriflow**: https://www.doriflow.com/
- **PAC-NeRF**: https://arxiv.org/abs/2306.10721
- **GIC**: https://arxiv.org/abs/2407.16561
- **SVD**: https://arxiv.org/abs/2311.15127
- **CogVideoX**: https://arxiv.org/abs/2408.06091
- **Wan**: https://arxiv.org/abs/2503.20314
- **Physion**: https://arxiv.org/abs/2106.06644
- **VideoPhy**: https://arxiv.org/abs/2410.02327
- **FreeGave**: https://arxiv.org/abs/2503.06528
- **MAGI-1**: https://arxiv.org/abs/2505.13211
- **MotionPro**: https://arxiv.org/abs/2503.07111
- **GoWithTheFlow**: https://arxiv.org/abs/2501.08806
- **Fundamentals of Physics (教材)**: Halliday, Resnick, Walker, 12th Edition

---

## 九、总结性思考

PhysInOne的核心价值在于将visual physics learning从"toy problem"推向"realistic scale"。通过组合71个basic phenomena成3284个multiphysics activities，再用平均46.84个scene实例化每个activity，最终得到153,810个scenes，每个scene有13个camera views = 2 million videos。这个scale使得fine-tuning foundation models成为可能。

PMF metric的设计哲学值得深思: **motion dynamics = energy spectrum distribution**，这个等价关系通过Fourier transform建立，且energy spectrum对spatiotemporal shift和brightness rescaling的invariance恰好对应了我们关心的"motion pattern vs initial state"的decoupling。这种基于频域的物理fidelity评估方法可能成为未来video generation评估的新范式。

paper暴露的critical gaps同样重要: novel view generalization的5-7dB gap，granular substances的低SSIM (0.86)，motion transfer的PMF仅3.3，都说明physics-grounded world model仍是open problem。PhysInOne作为benchmark的价值在于提供了足够challenging且well-annotated的testbed来揭示这些limitation。
