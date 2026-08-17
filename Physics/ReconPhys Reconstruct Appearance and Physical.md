---
source_pdf: ReconPhys Reconstruct Appearance and Physical.pdf
paper_sha256: 8c3d07c38d2f447452f6a4ac4f8e03dbf37e67ecdc530f580b2caf51aee50953
processed_at: '2026-08-11T21:51:48-07:00'
target_folder: Physics
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好的，用人话讲讲这篇paper到底在干什么。

## 一句话概括

你拿手机拍一个软东西掉地上弹两下的视频，喂给这个系统，一秒钟之内它就告诉你：这玩意儿长啥样（3D模型）、有多重、多硬、阻尼多少、地面摩擦多少。然后这堆参数可以直接扔进模拟器里，让机器人去练习捏它、拽它、揉它。

## 以前为啥这事难做

以前的思路是**per-scene optimization**：对每一个新物体，把视频和多视角拍的东西塞进物理引擎，然后一点点调参数（mass、stiffness、damping、friction），让模拟出来的画面和真实视频尽量像。调一个物体要一个多小时，每来个新东西都要重来。Spring-Gaus 就是这个套路，还要求多相机视角。

这就像你每见一个新物体都要从头研究它的物理属性，没法把"我见过枕头、汉堡、鸡蛋掉地上的经验"迁移到新东西上。

## ReconPhys 的核心思路

换成 **amortized inference**：训一个 feedforward neural network，让它在成千上万个"虚拟物体掉地上"的视频上看会了，以后见到新视频直接一次前向就吐出物理参数。

这就像你训练一个小孩看了几万个东西掉地上的视频，他以后见到一个新的软玩具，扔一次就能大概判断"这玩意儿挺软、挺轻、不太弹"。

## 关键技术 trick：让物理引擎和渲染器都可微分

这是整篇 paper 最精髓的地方。

你要让神经网络学会"从视频反推物理参数"，得有一个 supervision signal。但他们没有 ground-truth 物理参数标签（现实中你不可能给每个物体标注真实 stiffness）。所以他们用了一个很巧妙的 self-supervised 套路：

1. 网络看视频 → 预测 physical attributes (mass, stiffness, damping, friction)
2. 把这些参数塞进 **可微分的 spring-mass simulator**，模拟出 anchor points 的运动轨迹
3. 通过 IDW 插值把 sparse anchor 的运动传递到 dense 3D Gaussians 的中心点上
4. 用 **可微分的 3DGS renderer** 把这些 deformed Gaussians 渲染成画面
5. 算渲染画面和真实视频的 L2 loss
6. gradient 从 loss 一路反传，穿过 renderer → binding → simulator，最后回到 physical attributes

整个 pipeline 每一步都可微，所以 photometric loss 就能监督物理参数学习。**完全不需要物理标签**。

intuition 是：如果网络预测的 stiffness 是对的，模拟出来的形变就匹配视频；如果预测错了，渲染画面就不像，loss 就大。所以网络必须去"理解"视频里的运动模式来反推参数。

## 为什么 3DGS predictor 是 frozen 的

这是个很关键的 design choice。他们用了一个预训练好的 3DGS predictor 重建 canonical geometry（物体静止时的 3D 形状），然后**冻结它**，不参与训练。

为什么？因为如果 geometry 也能被 gradient 更新，网络就会"作弊"——直接调 geometry 去 fit 视频画面，而懒得去学物理参数。冻结 geometry 后，网络**必须**通过调 physical attributes 来解释运动，这才是真正的 system identification。

这就像考学生，你得把答案（geometry）盖住，只让他根据题目（视频里的运动）推出解题过程（物理参数），不然他就直接抄答案了。

## Spring-Mass Model 的物理细节

每个物体被表示成一组 sparse 的 mass points（anchors），每个 anchor 和它最近的 K 个邻居用弹簧连起来。每根弹簧有 stiffness $k$ 和 damping $d$，每个点有 mass $m$，整体和地面之间有 friction $f$。

弹簧力用的是 **generalized Hooke's law**，多了个指数 $p_k$：
$$\mathbf{F} = -k \cdot (\|\mathbf{x}_i - \mathbf{x}_j\| - l_{ij})^{p_k} \cdot \hat{u}$$

$p_k=1$ 就是普通线性弹簧，$p_k>1$ 越拉越硬（像生物组织），$p_k<1$ 越拉越软。

积分用 semi-implicit Euler，比 explicit Euler 在弹簧系统上更稳定，不会能量爆炸。

## Anchor 怎么驱动 Gaussian

物体表面和内部用 volume sampling 撒一些 anchor points（数量 $N_A$ 远小于 Gaussian 数 $N$），然后每个 Gaussian 的中心位置通过 **Inverse Distance Weighting** 从最近的 $n_b$ 个 anchor 插值得到：

$$\mu_i^{t+1} = \frac{\sum_j \mathbf{x}_j^{t+1} / r_j^{p_b}}{\sum_j 1/r_j^{p_b}}$$

intuition：anchor 越近的 Gaussian 受它影响越大。这和 skeletal animation 里 bones 驱动 mesh vertices 的思路一样，只不过这里用连续的 IDW 替代离散的 skinning weights。

## Self-Forcing 是干嘛的

autoregressive 模拟时，每一步从**网络自己上一步的预测状态**出发，不从 ground-truth 状态出发。

为什么？因为训练时如果有 GT 状态喂进去，网络学到的是"给定完美状态推下一步"；但推理时没有 GT，只能用自己的预测，分布不匹配，累积误差爆炸。Self-forcing 让训练分布匹配推理分布。

同时用 truncated BPTT 防止长序列反传时梯度爆炸——每步 detach 一下 input state，让 gradient 只从当前帧的 loss 流回 physical attributes，不穿越整条 trajectory。

## 数据怎么来的

这是 paper 里很扎实的部分。他们建了一个全自动 pipeline：

1. 从 Objaverse-XL 里用 Qwen3-8B 做语义筛选，挑出 500 个适合做软体动画的物体
2. 用 TRELLIS 从 4 个正交视角重建 high-fidelity 3DGS
3. 对每个物体，用 hash(物体ID) 作为 seed 做 anchor sampling（保证可复现）
4. 从连续分布里采样 physical parameters：$m \in [0.2, 6.0]$，$k \in [10, 1200]$，$d \in [0.1, 5.0]$，$f \in [0, 1.0]$
5. 模拟 30 帧 free-fall + ground collision
6. 渲染成 512×512 的单目视频

关键：**物理参数是连续分布而非离散类别**，所以网络学的是 regression 而非 classification，这对 generalization 很重要。

## 实验结果有多炸

Table 1 是重头戏：

| Method | Recon PSNR | Recon CD | Future Pred PSNR | Time |
|--------|-----------|----------|------------------|------|
| 4DGS | 30.33 | 0.593 | — | >1h |
| Spring-Gaus | 22.26 | 0.466 | 13.27 | >1h |
| **Ours** | **33.84** | **0.001** | **21.64** | **<1s** |

几个亮瞎眼的点：

- **Chamfer Distance 从 0.593 降到 0.001，593 倍改进**。这个数字夸张到有点反常。原因是 ReconPhys 用 frozen 的 pretrained 3DGS predictor，canonical geometry 质量有保证；而 4DGS 是 deformation-based，canonical geometry 容易漂。

- **Future Prediction PSNR 21.64 vs 13.27，提升 8.37 dB**。这是真正衡量"网络有没有学到物理"的指标。Spring-Gaus 用多视角 + 1 小时优化都打不过 ReconPhys 的单目 + 1 秒推理。说明 amortized inference 学到的 prior 比 per-scene optimization 更 generalizable。

- **推理速度 <1s vs >1h**。这是 paradigm shift 的直接证据。

## Physical Disentanglement 验证

Table 2 和 Table 3 验证一个关键问题：网络是不是真的学到了物理，还是只是记住了 motion pattern？

他们给同一个物体（比如同一个 Cylinder 几何）配两套不同物理参数，看网络能不能区分。

结果：ReconPhys 在两套配置下都能高 fidelity 重建（PSNR 36.86 vs 36.35），并且 future prediction 的轨迹明显不同（Figure 6）。证明网络确实把"形变模式"和"物理参数"解耦了——同一个形状，不同材质，给出不同的未来预测。

物理参数误差 Table 3 里 stiffness error 从 827 降到 297，明显改善但绝对值仍大。这说明 inverse physics 本质是 ill-posed 的（多组参数能产生相似的视觉动力学），网络学到的可能是某种"等效参数"而非真实物理参数。这其实没那么糟——对 simulation 应用来说，参数能复现观察到的动力学就够了，不一定要对应真实物理量。

## 为啥对 Robotics 重要

Section 4.5 是这篇 paper 的 ultimate motivation。

机器人抓软东西难，因为 sim2real gap 大，而模拟器里又缺好的 deformable asset。传统方法人工建模一个软物体资产要花很久。

ReconPhys 的 pathway：
```
拍 1 秒视频 → SAM 分割 → ReconPhys <1s 推理 → 3DGS + spring-mass asset → PhysTwin 模拟器 → 机器人 policy training
```

Figure 7 展示了 4 个 manipulation 场景：stretch pencil bag、squeeze pillow、stretch hamburger、squash toy。这就是把真实世界物体快速"翻译"成模拟器能用的资产，scalable 地生成训练数据。

## 我觉得最巧妙的几个点

1. **Frozen 3DGS predictor 强制网络学物理**，而不是调 geometry 作弊。这个 inductive bias 非常对。

2. **Differentiable simulation-rendering loop 让 photometric loss 监督物理参数**，完全不需要物理标签。这是把 inverse physics 变成可端到端学习的问题。

3. **Continuous physical parameter space**，让网络学 regression 而非 classification，这才可能 generalize 到训练分布外的材质。

4. **Hash-based anchor sampling** 保证 reproducibility，这个 engineering 细节看起来不起眼但其实关键——不然每次 anchor 配置不同，预测的参数没法比较。

5. **Self-forcing + truncated BPTT** 处理 autoregressive simulation 训练的标准 trick，借鉴自 video prediction 领域。

## 局限性

老实说几个明显的 gap：

1. **Shared parameters**：所有 anchor 共享同一个 mass、stiffness、damping。真实物体是异质的（带骨头的肉、带夹层的枕头），这个简化会限制应用。

2. **Spring-mass 的表达力**：只能模拟弹性形变，没法做塑性形变、断裂、流体。PAC-NeRF 用 MPM 更 expressive 但也更复杂。

3. **Synthetic → Real gap** 没量化：训练全在合成数据，real-world 验证只是 qualitative（Figure 5、7）。Sim-to-real 的 domain gap 没被严肃对待。

4. **Friction 估计不准**：Table 3 显示 friction error 比 Spring-Gaus 还大。因为 friction 只在 ground contact 时可观察，信号稀疏。

5. **单目 depth ambiguity**：单目视频本身 ill-posed，网络依赖 frozen 3DGS predictor 提供 canonical geometry 来缓解，但如果 predictor 在某物体上失败，physics 也会跟着失败。

## 更广的 context

这篇 paper 其实是几个趋势的交汇点：

- **3DGS 的 explicit representation** 让 differentiable rendering 变得高效（比 NeRF 快得多）
- **Differentiable physics** 让 gradient 能流过物理模拟（DiffTaichi、Brax、Genesis 方向）
- **Amortized inference** 取代 per-scene optimization（LRM、Zero-1-to-3、InstantID 等）
- **Self-supervised learning via differentiable pipeline**（Dreamer、World Models 路线）

ReconPhys 把这几个趋势拧到一起，做 inverse physics 的 amortized inference。我觉得这个 paradigm 会被广泛复制——differentiable physics + differentiable rendering + feedforward estimator + self-supervised photometric loss，这是个可扩展的 template。

类似工作脉络可以追踪：
- PhysTwin: https://arxiv.org/abs/2503.17973
- Embodiedreamer（同一 group）: https://arxiv.org/abs/2507.05198
- Spring-Gaus: https://arxiv.org/abs/2403.09534
- PhysGaussian: https://arxiv.org/abs/2311.04306
- PAC-NeRF: https://arxiv.org/abs/2303.05512
- 4DGS: https://arxiv.org/abs/2402.07138
- Self-Forcing: https://arxiv.org/abs/2506.08009

未来可能的扩展方向：用 GNN-based learned simulator 替代 hand-designed spring-mass（Sanchez-Gonzalez et al. 2020 那条线），predict per-point heterogeneous material properties，加入 plasticity/fracture 等更丰富物理现象，用 real-world data fine-tune 缩小 sim-to-real gap。

---

# ReconPhys 深度技术解析

## 1. Core Intuition：为什么这个工作重要

ReconPhys 解决的核心问题是：**给定一个 monocular video（单目视频）捕获一个 deformable object 的 free-fall + collision + rebound 动力学过程，如何在 <1 second 内同时恢复 appearance、geometry 和 physical attributes**，无需 per-scene optimization、无需 ground-truth physics labels。

关键 insight 在于：**用 feedforward neural estimator 替代 scene-specific physics tuning**。通过 conditioning physical attributes prediction on reconstructed shape + observed motion dynamics，学习一个 generalizable mapping from visual cues → material properties，从而实现 zero-shot transfer。

这本质上是把 inverse physics problem 从 "per-scene optimization" paradigm 转移到 "amortized inference" paradigm，类似于 LRM (Large Reconstruction Model) 对单图 3D 重建做的事情，只不过这里多了一个 physics dimension。

Project page: https://chuanshuogushi.github.io/ReconPhys

---

## 2. Problem Formulation 详解

### 2.1 Object Model

Object $O$ 被建模为 deformable body，由 spring-mass system 支配：

- **Mass points**: $\{\mathbf{x}_i\}_{i=1}^{N_A}$，共 $N_A$ 个 anchor points
  - $\mathbf{x}_i \in \mathbb{R}^3$：第 $i$ 个 mass point 的 3D 位置
  - $N_A$：anchor 总数（远小于 Gaussian 数 $N$，即 $N_A \ll N$）
  
- **Connectivity**: 通过 KNN 在 initial shape 上一次性确定，之后保持固定
  - 这意味着拓扑结构是 static 的，只有 positions 随时间变化

- **Physical attributes**: $\mathbf{p} = (\{m_i\}, \{k_{ij}\}, \{d_{ij}\}, f)$
  - $m_i$：第 $i$ 个 mass point 的质量
  - $k_{ij} \geq 0$：连接 point $i$ 和 $j$ 的 spring 的 stiffness（刚度）
  - $d_{ij} \geq 0$：spring 的 damping coefficient（阻尼系数）
  - $f \geq 0$：object 与 ground 之间的 global friction coefficient（摩擦系数）

### 2.2 3DGS Representation

Geometry + appearance 用 $N$ 个 Gaussian kernels 表示。每个 Gaussian $\mathbf{g}_j$ 包含：
- 3D center $\boldsymbol{\mu}_j$（**crucially，这是被 bind 到 spring-mass system 的**）
- Orientation $\theta_j$（rotation）
- RGB color $\mathbf{c}_j$
- Scale $\boldsymbol{\sigma}_j$
- Opacity $\alpha_j$

### 2.3 Feedforward Model

Equation (1):
$$(\hat{\mathbf{p}}, \hat{\mathbf{g}}) = \mathcal{M}(\mathcal{V})$$

- $\mathcal{V}$：input monocular video
- $\mathcal{M}$：feedforward prediction model
- $\hat{\mathbf{p}}$：predicted physical attributes
- $\hat{\mathbf{g}}$：predicted 3DGS（canonical state）

**简化处理**：论文中采用 shared parameters across points/springs，即 $\hat{m}_i = \hat{m}$（所有 mass point 共享一个 mass 值），$\hat{k}_{ij} = \hat{k}$（所有 spring 共享一个 stiffness），$\hat{d}_{ij} = \hat{d}$。这降低了 prediction 的维度，但牺牲了 per-point 物理异质性。这是一个 engineering trade-off。

---

## 3. Physical Dynamics Model 细节

### 3.1 Spring-Mass System Construction

Equation (2):
$$\mathcal{L} = \{l_{i,j}\}_{i=1,j=1}^{N_A, K} = \mathrm{knn}(\mathcal{A}, \mathcal{A}, K)$$

- $\mathcal{A} = \{\mathbf{x}_i\}_{i=1}^{N_A}$：anchor set
- $K$：每个 anchor 的最近邻数量（论文中应该是个 hyperparameter，未明确给值）
- $l_{i,j}$：anchor $\mathbf{x}_i$ 与其第 $j$ 个最近邻 $\mathbf{x}_{i,j}$ 之间的 rest length（初始距离）
- 这个 rest length 在 simulation 过程中是固定的，作为 spring 的 equilibrium state

### 3.2 Force Computation

**Net force** on anchor $\mathbf{x}_i^t$ at timestep $t$（Equation 3）：
$$\mathbf{F}_i^t = \sum_{j=1}^{K} \mathbf{F}_{i,j}^{k,t} + \sum_{j=1}^{K} \mathbf{F}_{i,j}^{d,t} + m_i \mathbf{g}_{\mathrm{grav}}$$

- $\mathbf{F}_{i,j}^{k,t}$：spring force from neighbor $j$
- $\mathbf{F}_{i,j}^{d,t}$：damping force from neighbor $j$
- $m_i \mathbf{g}_{\mathrm{grav}}$：gravity force，$\mathbf{g}_{\mathrm{grav}}$ 是 gravity acceleration vector（通常 $[0, 0, -9.81]$）

**Nonlinear Spring Force**（Equation 4）—— 这是 generalized Hooke's law：
$$\mathbf{F}_{i,j}^{k,t} = -k_{i,j} \left(\lVert \mathbf{x}_i^t - \mathbf{x}_{i,j}^t \rVert - l_{i,j}\right)^{p_k} \cdot \frac{\mathbf{x}_i^t - \mathbf{x}_{i,j}^t}{\lVert \mathbf{x}_i^t - \mathbf{x}_{i,j}^t \rVert}$$

变量解析：
- $k_{i,j}$：stiffness coefficient
- $\lVert \mathbf{x}_i^t - \mathbf{x}_{i,j}^t \rVert$：当前时刻两 anchor 之间的实际距离
- $l_{i,j}$：rest length（零力距离）
- $p_k$：nonlinearity exponent，控制 spring 的非线性程度
  - $p_k = 1$：standard linear Hooke's law
  - $p_k > 1$：strain-stiffening（越拉越硬，类似 biological tissue）
  - $p_k < 1$：strain-softening
- $\frac{\mathbf{x}_i^t - \mathbf{x}_{i,j}^t}{\lVert \mathbf{x}_i^t - \mathbf{x}_{i,j}^t \rVert}$：unit direction vector from $j$ to $i$
- 负号：force 指向恢复 rest length 的方向

**Damping Force**（Equation 5）：
$$\mathbf{F}_{i,j}^{d,t} = -d_{i,j} (\mathbf{v}_i^t - \mathbf{v}_{i,j}^t) \cdot \frac{\mathbf{x}_i^t - \mathbf{x}_{i,j}^t}{\lVert \mathbf{x}_i^t - \mathbf{x}_{i,j}^t \rVert} \cdot \frac{\mathbf{x}_i^t - \mathbf{x}_{i,j}^t}{\lVert \mathbf{x}_i^t - \mathbf{x}_{i,j}^t \rVert}$$

变量解析：
- $d_{i,j}$：damping coefficient
- $\mathbf{v}_i^t - \mathbf{v}_{i,j}^t$：relative velocity between two anchors
- 这个公式有点复杂，本质上是把 relative velocity 投影到 spring direction 上，只对沿 spring 方向的相对运动施加阻尼
- 两个相同的 unit vector 相乘：$\hat{u} \cdot \hat{u}^T$ 形成的是 projection matrix $\hat{u}\hat{u}^T$，作用是把 velocity vector 投影到 spring axis 上

### 3.3 Numerical Integration

**Semi-implicit (Symplectic) Euler**（Equations 6, 7）：
$$\hat{\mathbf{v}}_i^{t+1} = \mathbf{v}_i^t + \frac{\mathbf{F}_i^t}{m_i} \Delta t$$
$$\hat{\mathbf{x}}_i^{t+1} = \mathbf{x}_i^t + \hat{\mathbf{v}}_i^{t+1} \Delta t$$

- $\Delta t$：timestep
- 注意 velocity 用 **updated** velocity $\hat{\mathbf{v}}_i^{t+1}$ 来更新 position，这就是 "semi-implicit" 的含义
- 对比 explicit Euler：$\hat{\mathbf{x}}_i^{t+1} = \mathbf{x}_i^t + \mathbf{v}_i^t \Delta t$（用 old velocity）
- Semi-implicit Euler 的优势：能量保守性更好，对 spring system 更稳定，长时间 simulation 不会能量爆炸

**Boundary Condition**（Equation 8）：
$$(\mathbf{x}_i^{t+1}, \mathbf{v}_i^{t+1}) = \mathcal{B}(\hat{\mathbf{x}}_i^{t+1}, \hat{\mathbf{v}}_i^{t+1})$$

- $\mathcal{B}(\cdot)$：boundary function，处理 ground collision
- 通常实现为：如果 $\mathbf{x}_i^{t+1}$ 的 z 坐标 < ground plane，则 clamp position 到 ground，并 reflect/damp velocity 的 z 分量
- Friction $f$ 在这里发挥作用：切向 velocity 乘以 $(1 - f)$ 或类似衰减

### 3.4 Two-Stage Binding Mechanism

这是连接 physical simulation 与 visual representation 的关键桥梁。

**Stage 1: Anchor Sampling**（Equation 9）：
$$\mathcal{A} = \mathcal{V}_{\mathrm{vol}}(\mathcal{X})$$

- $\mathcal{X} = \{\mu_i\}_{i=1}^N$：所有 Gaussian centers
- $\mathcal{V}_{\mathrm{vol}}(\cdot)$：volume sampling function
- 关键点：**anchors 分布在 object volume 内部，而非仅表面**
  - 这很重要：表面-only 的 anchor 会导致 simulation 不稳定，因为 deformation 在体积内部也有
  - $N_A \ll N$：用 sparse anchors 驱动 dense Gaussians，类似 control lattice / cage deformation 的思想

**Stage 2: Position Interpolation via IDW**（Equation 10）：
$$\mu_i^{t+1} = \frac{\sum_{j=1}^{n_b} \mathbf{x}_{i,j}^{t+1} \cdot (1/r_{i,j}^{p_b})}{\sum_{j=1}^{n_b} (1/r_{i,j}^{p_b})}$$

- $n_b$：每个 Gaussian 使用的 nearest anchor 数量
- $r_{i,j} = \lVert \boldsymbol{\mu}_i^0 - \mathbf{x}_{i,j}^0 \rVert$：Gaussian center $\mu_i$ 与 anchor $\mathbf{x}_{i,j}$ 的 **initial** 距离
- $p_b > 0$：distance falloff exponent，控制插值的 localness
  - $p_b$ 大：更 local，只有最近的 anchor 起作用
  - $p_b$ 小：更 global，远处 anchor 也有影响
- IDW = Inverse Distance Weighting，经典插值方法

**Intuition**：这相当于把 sparse anchor motion "upsample" 到 dense Gaussian motion，类似 skinning 中 bones 驱动 vertices 的机制，只不过这里是 continuous 的 IDW 而非 discrete 的 skinning weights。

---

## 4. Model Architecture 深度解析

### 4.1 Dual-Branch Design

```
Input Video V = {I_t}_{t=1}^T
        |
        +---→ [3DGS Predictor] (FROZEN, pretrained)
        |         |
        |         +---→ canonical Gaussians ĝ^0 (centers, scale, rotation, color, opacity)
        |                   |
        |                   +---→ Volume Sampling → Anchors A
        |                                      |
        +---→ [Physics Predictor] (TRAINABLE)  |
        |         |                            |
        |         +---→ InternViT (per-frame)  |
        |         +---→ ResNet + Self-Attention (temporal aggregation)
        |         +---→ MLP Decoder
        |                   |                  |
        |                   +---→ p̂ = (m̂, k̂, d̂, f̂)
        |                                      |
        +---→ [Differentiable Spring-Mass Simulator] ← (p̂, A)
        |                   |
        |                   +---→ anchor trajectories {x_i^t, v_i^t}
        |                                      |
        +---→ [IDW Binding] ← (anchor traj, canonical Gaussians)
        |                   |
        |                   +---→ deformed Gaussians ĝ^t (only centers updated)
        |                                      |
        +---→ [Differentiable 3DGS Renderer]
        |                   |
        |                   +---→ rendered frames Î_t
        |
        +---→ Photometric Loss L = Σ ||I_t - Î_t||^2
```

### 4.2 3DGS Predictor（Frozen）

- 使用 off-the-shelf pretrained weights
- **始终保持 frozen**，不参与 training
- 这是个关键 design choice：
  - 好处：training 只需 focus on physics identification，减少参数空间
  - 好处：canonical geometry 质量有保证
  - 代价：如果 3DGS predictor 对某个 object 重建不好，physics estimation 也受限

### 4.3 Physics Predictor

**Visual Encoder**: InternViT-300M (Chen et al. 2024)
- 从每帧提取 dynamic features
- InternViT paper: https://arxiv.org/abs/2307.06935

**Temporal Aggregation**: ResNet backbone (Koonce 2021) + Self-Attention (Vaswani et al. 2017)
- ResNet 处理 spatial features
- Self-Attention 聚合 spatio-temporal context
- 这里的设计思路：先 per-frame encode，再 cross-frame aggregate

**Decoder**: MLP → physical attributes $\hat{\mathbf{p}}$

**为什么这个 architecture 能 work**：
- Physics predictor 本质上是在做 **system identification from visual observation**
- 它需要从 "how the object deforms over time" 推断 "what material properties cause this deformation"
- 这是个 inverse problem，需要理解 deformation pattern（stiff → small deformation, soft → large deformation; high damping → quick settle, low damping → oscillating）

### 4.4 Self-Forcing Training Strategy

借鉴 Huang et al. 2025 的 Self Forcing（https://arxiv.org/abs/2506.08009）：

**核心思想**：在 autoregressive simulation rollout 中，每一步从 **model 自己的 previous predicted state** 出发，而非从 ground-truth / proxy state 出发。

**为什么需要**：
- Training 时如果用 GT state 作为每步 input，model 学到的是 "given perfect state, predict next state"
- Inference 时没有 GT state，只能用自己的 prediction，导致 **train-test mismatch** / exposure bias
- Self-forcing 让 training distribution 匹配 inference distribution

**Truncated Backpropagation**：
- 长序列 unrolled simulation 会导致 gradient explosion/vanishing
- 解决方案：每个 rollout step 之前 detach input state
- $\mathbf{x}_{t+1} = \mathrm{simulate}(\mathrm{detach}(\mathbf{x}_t), \hat{\mathbf{p}})$
- 这样 gradient 只从 current step 的 rendering loss 流回 $\hat{\mathbf{p}}$，而非通过整个 trajectory

---

## 5. Differentiable Simulation-Rendering Loop

这是整个 framework 的核心，让 self-supervised training 成为可能。

### 5.1 Forward Pass

```
p̂ → [Simulator] → {x_i^t, v_i^t} → [IDW Binding] → μ^t → [Renderer] → Î_t
```

每个环节都可微：
1. **Simulator**: semi-implicit Euler + force computation 都是用 differentiable operations 构成的
2. **IDW Binding**: weighted average，天然可微
3. **3DGS Renderer**: differentiable rasterization (Kerbl et al. 2023)

### 5.2 Backward Pass

Gradient flow:
```
∂L/∂Î_t → ∂Î_t/∂μ^t (via renderer) → ∂μ^t/∂x_i^t (via IDW) → ∂x_i^t/∂p̂ (via unrolled simulator)
```

**关键 insight**：因为 3DGS predictor 是 frozen 的，gradient pathway 专门用于 identifying physically meaningful attributes，而非 improving 3DGS reconstruction。这是个很重要的 inductive bias：**让 physics 解释 observation，而非让 geometry 去 overfit observation**。

### 5.3 Training Objective

Equation (11):
$$\mathcal{L} = \sum_{t=1}^{T} \lVert I_t - \hat{I}_t \rVert_2^2$$

- $I_t$：ground-truth frame at time $t$
- $\hat{I}_t$：rendered predicted frame
- 纯 photometric L2 loss，**没有 physics supervision**
- 这意味着：如果 model 能 reconstruct 所有 frames，那它的 physics parameters 一定是 "correct" 的（至少在 observation-equivalent 意义下）

**Intuition**：这是个 **inverse problem via differentiable simulation**。Model 必须找到一组 physical parameters，使得 simulated dynamics 在 rendering 后匹配 observation。这类似于 model-based RL 中的 system identification，只不过这里的 "sensor" 是 differentiable renderer。

---

## 6. Synthetic Data Pipeline

### 6.1 Asset Generation

1. **Source**: Objaverse-XL (Deitke et al. 2023) - https://arxiv.org/abs/2307.05638
2. **Semantic Filtering**: Qwen3-8B (Yang et al. 2025) 筛选适合 non-rigid dynamics 的 objects，得到 500 个 eligible objects
3. **3DGS Reconstruction**: 渲染 4 个 orthogonal views → TRELLIS (Xiang et al. 2025) 重建 high-fidelity 3DGS
   - TRELLIS: https://trellis3d.github.io/

### 6.2 Consistent Anchor Sampling

**关键 engineering detail**：anchor sampling seed 由 object 的 unique identifier 的 hash code 生成。

为什么重要：
- 如果 random sampling，每次运行 anchor configuration 不同
- 则 predicted physical attributes 无法 cross-run 比较
- Hash-based seeding 保证 reproducibility + interpretability

### 6.3 Physical Parameter Distribution

Continuous sampling ranges:
- Mass $m \in [0.2, 6.0]$
- Stiffness $k \in [10, 1200.0]$
- Damping $d \in [0.1, 5.0]$
- Friction $f \in [0.0, 1.0]$

**关键 design choice**：continuous distribution 而非 discrete categories。这让 model 学到的是 regression 而非 classification，更 generalizable。

### 6.4 Simulation Specs

- 30-frame free-fall trajectory under gravity + ground collision
- Rendered at $512 \times 512$ resolution
- Dataset: $(\mathcal{V}, \mathbf{g}, \mathbf{p})$ triplets

---

## 7. Experimental Results 深度分析

### 7.1 Cross-Object Generalization (Table 1)

| Method | Recon PSNR↑ | Recon CD↓ | Future Pred PSNR↑ | Future Pred CD↓ | Time |
|--------|-------------|-----------|-------------------|-----------------|------|
| 4DGS | 30.33 | 0.593 | - | - | >1h |
| Spring-Gaus | 22.26 | 0.466 | 13.27 | 0.349 | >1h |
| **Ours** | **33.84** | **0.001** | **21.64** | **0.004** | **<1s** |

**关键观察**：

1. **Reconstruction PSNR**: 33.84 vs 30.33 (4DGS) vs 22.26 (Spring-Gaus)
   - ReconPhys 超越 4DGS +3.51 dB，尽管 4DGS 用 4 views 而 ReconPhys 只用 1 view
   - 这看似 counterintuitive，但解释是：4DGS 是 per-scene optimization，在 unseen configuration 下可能 overfit；ReconPhys 的 frozen 3DGS predictor 是大规模 pretrained 的，quality 更稳定

2. **Chamfer Distance**: 0.001 vs 0.593 (4DGS) —— **593x 改进**
   - 这是个惊人的数字。CD 衡量 3D geometry accuracy
   - 解释：ReconPhys 的 canonical geometry 来自 pretrained 3DGS predictor，geometry quality 极高
   - 4DGS 作为 deformation-based method，canonical geometry 可能漂移

3. **Future Prediction PSNR**: 21.64 vs 13.27 (Spring-Gaus) —— **+8.37 dB**
   - 这是最重要的 metric，衡量 model 是否真正学到 physics
   - Spring-Gaus 用 per-scene optimization 但 future prediction 仍然差，说明它的 physics identification 不 robust
   - ReconPhys 的 amortized inference 反而更 generalizable

4. **Inference Time**: <1s vs >1h —— **3600x+ 加速**
   - 这是 paradigm shift 的直接体现

### 7.2 Physical Disentanglement (Table 2)

测试：相同 geometry，不同 physical attributes，能否区分？

Table 2 显示：
- 同一 object 的两个物理配置（e.g., Cylinder-1 vs Cylinder-2），ReconPhys 都能 high-fidelity 重建（PSNR 36.86 vs 36.35）
- Spring-Gaus 的 PSNR 明显更低且不稳定

Figure 6 可视化：同一 object 不同 physical attributes 产生 **divergent future trajectories**，证明 model 学到的是 physics 而非 memorized motion pattern。

### 7.3 Physical Attribute Errors (Table 3)

| Method | Stiffness | Damp | Mass | Friction |
|--------|-----------|------|------|----------|
| Spring-Gaus | 827.67 | 2.546 | 2.276 | 1.082 |
| Ours | 297.3 | 1.151 | 1.337 | 1.508 |

观察：
- Stiffness error 大幅降低（827 → 297），但绝对值仍较大
- 这说明 physics identification 本质上是 ill-posed 的（multiple parameter combinations can produce similar visual dynamics）
- Friction 方面 Spring-Gaus 略好，可能因为 friction 主要影响 ground contact，观察信号较弱

### 7.4 Real-World Results (Figure 5, 7)

定性验证 real-world non-rigid assets：
- Figure 5: 两个 object 的 dropping + deformation
- Figure 7: 4 个 manipulation 场景（stretch pencil bag, squeeze pillow, stretch hamburger, squash toy）
- 使用 SAM (Kirillov et al. 2023) 做 segmentation: https://arxiv.org/abs/2304.02643
- 集成到 PhysTwin (Jiang et al. 2025) framework 做 interactive simulation: https://arxiv.org/abs/2503.17973

---

## 8. 与 Related Works 的精细对比

### 8.1 vs Spring-Gaus (Zhong et al. 2024)

Spring-Gaus paper: https://arxiv.org/abs/2403.09534

| 维度 | Spring-Gaus | ReconPhys |
|------|-------------|-----------|
| Input | Multi-view | Monocular |
| Inference | Per-scene optimization (>1h) | Feedforward (<1s) |
| Generalization | Per-scene only | Cross-object zero-shot |
| Training | Per-scene | Self-supervised on synthetic dataset |
| Binding | Spring-mass + 3DGS | Same（ReconPhys 借鉴了）|

ReconPhys 本质上是 Spring-Gaus 的 **amortized inference** 版本。

### 8.2 vs PAC-NeRF (Li et al. 2023)

PAC-NeRF paper: https://arxiv.org/abs/2303.05512

- PAC-NeRF 用 MPM (Material Point Method) + NeRF
- 假设 known material families（e.g., elastic, plastic, sand）
- 使用 coarse global parameters
- ReconPhys 用 spring-mass（更简单但 less expressive），但 feedforward + generalizable

### 8.3 vs PhysGaussian (Xie et al. 2024)

PhysGaussian paper: https://arxiv.org/abs/2311.04306

- PhysGaussian: continuum mechanics + 3DGS for **generative dynamics**
- 不做 per-scene attribute inference from observation
- ReconPhys 是 inverse direction: from observation → attributes

### 8.4 vs 4DGS (Wu et al. 2024)

4DGS paper: https://arxiv.org/abs/2402.07138

- 4DGS: real-time dynamic rendering，但 **no explicit physics**
- 无法 do future prediction beyond training frames
- ReconPhys: 牺牲一点 recon quality 换取 physics awareness + future prediction

---

## 9. Limitations & Open Questions

1. **Shared physical parameters**: $\hat{m}_i = \hat{m}$ for all $i$。真实 object 内部 material 是 heterogeneous 的（e.g., 带骨头的肉）。Future work 应该 predict per-point attributes。

2. **Spring-mass expressiveness**: Spring-mass 是简化的物理模型，无法捕捉 continuum mechanics 的全部现象（e.g., volume preservation, plastic deformation, fracture）。

3. **Synthetic → Real gap**: 训练在合成数据上，real-world 验证仅 qualitative。Sim-to-real transfer 的 domain gap 未量化。

4. **Monocular ambiguity**: 单目视频本身是 ill-posed 的（depth ambiguity）。Model 依赖 3DGS predictor 的 canonical geometry 来缓解，但如果 predictor 失败，physics 也会失败。

5. **Friction estimation**: Table 3 显示 friction error 反而比 Spring-Gaus 大。Friction 只在 ground contact 时可观察，信号弱。

6. **Nonlinearity exponent $p_k$**: 论文未说明 $p_k$ 是 predicted 还是 fixed。如果是 fixed，则 model 无法捕捉 strain-stiffening/softening 的差异。

---

## 10. 更广的 Context：为什么这是 Robotics 的重要工作

这篇 paper 的 ultimate vision 在 Section 4.5：**为 robotic manipulation 生成 simulation-ready assets**。

当前 robotics 的痛点：
- Sim2real gap 很大
- Non-rigid object manipulation 缺乏 good simulation
- 手动创建 deformable asset 费时费力

ReconPhys 的 pathway：
```
Real video (1s capture) → ReconPhys (<1s inference) → 3DGS + spring-mass asset 
→ PhysTwin simulation → Robot policy training → Real deployment
```

这对应了 Karpathy 你经常强调的 **"software 2.0 for physics"** 思路：用 learned model 替代 hand-engineered physics solvers，用 data 驱动 system identification。

类似工作脉络：
- PhysTwin: https://arxiv.org/abs/2503.17973
- Embodiedreamer (同一作者): https://arxiv.org/abs/2507.05198
- DreamerV3 系列的 model-based RL
- Genesis simulator (Lin et al.) 的 generative physics 方向

---

## 11. 个人 Technical Takeaways

1. **Frozen 3DGS predictor 是关键**：这让 physics predictor 的 gradient 不会 "cheat" by modifying geometry，必须真正解释 dynamics。

2. **Self-forcing + truncated BPTT**：这是处理 autoregressive simulation training 的标准技巧，类似 video prediction 中的 scheduled sampling。

3. **Continuous physics parameter space**：让 model 学 regression 而非 classification，这是 generalization 的关键。

4. **Synthetic data pipeline 的自动化程度**：用 LLM (Qwen3-8B) 做 semantic filtering，用 TRELLIS 做 3DGS reconstruction，用 spring-mass 做 simulation —— 这是一个 scalable data engine 的 template。

5. **Inverse physics via differentiable simulation**：这个 paradigm 会被广泛复制。Differentiable physics + differentiable rendering + amortized inference = scalable physics understanding。

6. **Open question**: 能否把这个 framework 扩展到 more complex physics（fluids, granular materials, plasticity）？Spring-mass 的 expressiveness 是瓶颈。Future work 可能用 Graph Neural Network-based learned simulators (Sanchez-Gonzalez et al. 2020) 替代 hand-designed spring-mass。

---

## 12. References

- ReconPhys Project Page: https://chuanshuogushi.github.io/ReconPhys
- 3DGS: https://repo.samuelgarcia.ga/kerbl3dgaussians
- Spring-Gaus: https://elastic-gaussians.github.io/
- PhysGaussian: https://physgaussian.github.io/
- PAC-NeRF: https://sites.google.com/view/PAC-NeRF
- 4DGS: https://github.com/fudan-zvg/4d-gaussian-splatting
- InternViT: https://github.com/OpenGVLab/InternVL
- TRELLIS: https://trellis3d.github.io/
- Objaverse-XL: https://objaverse-xl.github.io/
- SAM: https://segment-anything.com/
- PhysTwin: https://arxiv.org/abs/2503.17973
- Self-Forcing: https://arxiv.org/abs/2506.08009
- Embodiedreamer (同一 group): https://arxiv.org/abs/2507.05198
