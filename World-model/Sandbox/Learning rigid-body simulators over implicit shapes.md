---
source_pdf: Learning rigid-body simulators over implicit shapes.pdf
paper_sha256: d98ef2422b89a1b1303096de909f3c68963b128a6c43e33a3173f9537cade556
processed_at: '2026-08-05T13:42:10-07:00'
target_folder: World-model/Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

用最直白的人话讲，这篇 paper 解决了一个非常现实的工程瓶颈：**之前的 AI 物理模拟器在处理大量物体时，内存会瞬间爆炸。**

传统基于 mesh 的 learned simulator（比如 FIGNet）工作起来像是一个社交恐惧症患者。假设场景里有 1000 个 mesh nodes，为了检测碰撞，它要把这 1000 个 nodes 和场景里其他所有 nodes 两两配对检查。这种 $O(K^2)$ 的全连接模式导致只要物体稍微多一点，或者 mesh 稍微精细一点，GPU 显存就直接 OOM 了。同时，为了在 mesh 上找最近的点，还得依赖 CPU 上跑的 BVH tree search，速度慢且难以整合进 deep learning pipeline。

SDF-Sim 的核心思路极其优雅：**抛弃显式的 mesh，用 learned SDF（一个 MLP）把物体的几何形状“捏”成一个连续的场。**

想象一下，SDF 就像是物体周围自带的一个“力场发生器”。无论你在 3D 空间的哪个角落，只要给这个 MLP 输入你的坐标，它瞬间就能吐出一个数字告诉你距离物体表面有多远，再跑一次 backward pass 就能告诉你表面的法线方向。这意味着，当你想知道物体 A 上的一个点有没有撞到物体 B 时，直接问物体 B 的 SDF MLP 就行。距离够近的话，直接在这个点和物体 B 的中心连一条边。这就是一个 $O(I \cdot K)$ 的 linear scaling 操作，$I$ 是物体数量，远小于总节点数 $K$。

这就带来了一种极强的 **compositionality**（组合性）。因为 GNN 学到的是 local pairwise interaction 的物理规律，只要局部碰撞力学预测得准，全局场景无论堆叠多少个物体，都可以通过 message passing 像搭积木一样组合起来。所以，一个只在 10 个物体小场景上训练的模型，能够直接 generalize 到 512 个球互撞，甚至 380 个复杂形状物体（1.1 million nodes）堆叠的宏大场景，不仅没 OOM，还跑出了 200 步的 realistic rollout。

更有意思的是实验中的 intuition：
1. **网络对 SDF 的 noise 极度 robust**。作者尝试了不同 size 的 SDF MLP，发现即使用一个很粗糙的 SDF，simulator 的 accuracy 依然很高。这说明 GNN 在 message passing 中，能够隐式地纠正掉 implicit shape representation 带来的微小几何误差。只要碰撞的大方向和大概距离对，网络就能算出正确的 impulse。
2. **Vision 到 Simulation 的打通**。因为 SDF 可以直接从 multi-view RGB images 通过 VolSDF 蒸馏出来，这意味着你可以拿手机绕着桌子拍一圈，重建出桌子和大花瓶的 SDF，直接扔进 SDF-Sim 里，让一个虚拟鞋子掉下来和花瓶发生复杂的碰撞。这套 pipeline 完全不需要人工去建 mesh。

**Reference Web Links:**
- SDF-Sim Project Page: https://sites.google.com/view/sdf-sim
- FIGNet Baseline Paper: https://openreview.net/forum?id=IbmjGDnY6-
- VolSDF (Vision to SDF): https://github.com/lioryariv/volsdf
- Kubric Dataset: https://github.com/google-research/kubric

---

这篇 paper 介绍了 **SDF-Sim**，一种基于 learned **Signed Distance Functions (SDFs)** 和 **Graph Neural Networks (GNNs)** 的 rigid-body simulator。作者包括 DeepMind 的 Yulia Rubanova, Tatiana Lopez-Guevara, Kelsey R. Allen 等人。这篇工作的核心动机在于解决传统基于 mesh 的 learned simulators 在大规模场景下因 collision detection 产生 $\mathcal{O}(K^2)$ 复杂度而导致的 memory 与 runtime bottleneck。SDF-Sim 通过引入 implicit shape representation，将 inter-object collision edges 的复杂度降至 linear scaling $\mathcal{O}(I \cdot K)$，成功模拟了包含多达 1.1 million nodes 和数百个 objects 的超大规模场景，并首次打通了从 multi-view RGB images 重建 SDF 并直接用于 physical simulation 的 pipeline。

以下是详细的技术讲解与 intuition building。

### 1. Background 与 Motivation: Mesh-based GNNs 的 Bottleneck

在 learned physics simulation 领域，**Graph Network-based Simulators (GNS)** 通常将 object 的 triangle mesh 转化为 graph：mesh vertices 转化为 graph nodes，mesh edges 转化为 graph edges。物体内部的力（如弹力、速度传递）通过 intra-object edges 传播；物体间的碰撞通过 inter-object collision edges 传播。

传统方法（如 FIGNet, MeshGraphNets）在构建 collision edges 时，需要在所有属于不同 objects 的 mesh nodes 或 triangles 之间计算距离。如果距离小于阈值 $\mathcal{D}$，就建立一条 collision edge。当场景中有 $K$ 个 surface nodes 和 $I$ 个 objects 时，最坏情况下 potential collision pairs 的数量是 $\mathcal{O}(K^2)$。这种 quadratic scaling 导致在处理具有精细 meshes 或大量 objects 的场景时，GPU memory 迅速耗尽（OOM）。同时，为了寻找 closest point on mesh，通常需要依赖 CPU-based 的 **Bounding Volume Hierarchy (BVH)** tree search，这严重阻碍了 deep learning pipeline 的端到端 GPU 加速。

### 2. 核心技术：SDF-Sim 架构与 Graph 构建

SDF-Sim 的核心创新在于利用 **learned SDF** 替代 explicit mesh 来进行 distance query 和 closest point computation。

#### 2.1 Learned Signed Distance Functions (SDFs)
SDF 定义了一个 continuous field $f(\mathbf{y}; \boldsymbol{\theta}): \mathbb{R}^3 \to \mathbb{R}$，其中 $\mathbf{y}$ 是 3D space 中的任意 query point，输出是该点到 object surface 的 signed distance。Surface 被 implicit 地定义为 zero level set $\{\mathbf{y} \in \mathbb{R}^3 | f(\mathbf{y}) = 0\}$。Sign 为正表示在 external，为负表示在 internal。

在 SDF-Sim 中，每个 object 的 SDF 由一个 pre-trained MLP 参数化（通常为 8 layers, 128 hidden units）。对于任意点 $\mathbf{y}$，可以通过一次 forward pass 获取 distance，通过一次 backward pass 获取 gradient。SDF 的一个关键属性是可以直接计算 surface 上的 closest point $\mathbf{y}^*$：

$$ \mathbf{y}^* = \mathbf{y} - f_{\theta}(\mathbf{y}) \nabla f_{\theta}(\mathbf{y}) $$

**公式变量解析：**
- $\mathbf{y}^*$: Query point $\mathbf{y}$ 在 object surface 上的 closest point。
- $\mathbf{y}$: 3D 空间中的查询点。
- $f_{\theta}(\mathbf{y})$: SDF MLP 输出的 signed distance。它度量了 $\mathbf{y}$ 到 $\mathbf{y}^*$ 的直线距离。
- $\nabla f_{\theta}(\mathbf{y})$: SDF 在 $\mathbf{y}$ 点的 gradient。它是一个 unit vector，指向远离 surface 的最短路径方向。
- Intuition: 想象你站在空间中某点 $\mathbf{y}$，SDF 告诉你距离表面有多远 ($f_{\theta}$)，gradient 告诉你表面在哪个方向 ($\nabla f_{\theta}$)。你只要沿着 gradient 的反方向走 $f_{\theta}$ 的距离，就能精准落到表面上。这个计算完全在 GPU 上并行执行，速度极快。

#### 2.2 Graph Construction 与 Inter-object Edges
SDF-Sim 的 graph 包含两类 nodes：
1. **Object nodes** $\mathcal{V}_O = \{\mathbf{o}_i\}$: 代表每个 object 的 center of mass，存储 global motion state。
2. **Surface nodes** $\mathcal{V}_S = \{\mathbf{n}_{ik}\}$: 位于 object surface 上的采样点。

Intra-object connectivity 很简单：surface nodes 连接到对应的 object node。

关键创新在于 **inter-object collision edges** 的构建。对于 object $O_i$ 上的 surface node $\mathbf{n}_{ik}$，SDF-Sim 直接 query 相邻 object $O_j$ 的 SDF $f_{\theta_j}$。如果 distance $d_{ik}^j \le \mathcal{D}$，则在 $\mathbf{n}_{ik}$ 和 $O_j$ 的 object node $\mathbf{o}_j$ 之间建立一条 edge。注意这里连接的是 node 到 object center，避免了 surface node 之间的全连接。这使得 collision edges 数量在最坏情况下仅为 $\mathcal{O}(I \cdot K)$。

为了提供足够的 collision 信息给 GNN，edge features 中包含了 closest point 的位置。因为 SDF 是在 object 的 canonical pose (静止参考姿态) 下训练的，计算 closest point 时需要进行坐标系转换：

$$ \mathbf{c}_{ik}^j = \mathbf{n}_{ik} - d_{ik}^j \mathcal{T}_j \left( \nabla f_{\theta_j} \left( \mathcal{T}_j^{-1}(\mathbf{n}_{ik}) \right) \right) $$

**公式变量解析：**
- $\mathbf{c}_{ik}^j$: Object $O_i$ 的 node $\mathbf{n}_{ik}$ 在 object $O_j$ surface 上的 closest point。
- $\mathbf{n}_{ik}$: Object $O_i$ 上的第 $k$ 个 surface node 在 world space 的位置。
- $d_{ik}^j$: 从 $\mathbf{n}_{ik}$ 到 $O_j$ 的 signed distance。
- $\mathcal{T}_j$: Object $O_j$ 在当前 timestep 的 rigid transformation (包含 translation $\mathbf{p}_j^t$ 和 rotation $\mathbf{R}_j^t$)。
- $\mathcal{T}_j^{-1}$: Inverse transformation。将 world space 中的 $\mathbf{n}_{ik}$ 转换到 $O_j$ 的 local reference space。
- $f_{\theta_j}$: Object $O_j$ 的 SDF MLP。
- Intuition: 必须先用 $\mathcal{T}_j^{-1}$ 把 world coordinate 转回到 object 自己的坐标系，才能 query 它的 SDF MLP。算出 gradient 后，gradient 是在 local space 里的，所以需要用 $\mathcal{T}_j$ 把这个 gradient vector 旋转回 world space。

#### 2.3 Network Architecture 与 Dynamics Prediction
Graph 编码后，经过 10 步 message passing（MLP 含 2 hidden layers, 128 units, LayerNorm）。网络预测每个 surface node 的 acceleration $\hat{\mathbf{a}}_{ik}^{t+1}$。

Node features 包含 finite-difference velocity：$\mathbf{v}_{ik} = (\mathbf{n}_{ik}^t - \mathbf{n}_{ik}^{t-1}, \mathbf{n}_{ik}^{t-1} - \mathbf{n}_{ik}^{t-2})$。这相当于用过去 3 个 timestep 的位置估算速度。Loss function 采用 per-node acceleration 的 L2 loss。Ground truth acceleration 通过 finite difference 计算：

$$ \mathbf{a}_k^{t+1} = \mathbf{n}_k^{t+1} - 2\mathbf{n}_k^t + \mathbf{n}_k^{t-1} $$

这里 $\mathbf{a}_k^{t+1}$ 是下一时刻的加速度，上标 $t+1, t, t-1$ 代表相邻的三个时间步。这本质上是 Verlet integration 的离散化形式。预测出 node accelerations 后，使用 Euler integration 更新 node positions，最后通过 **Shape Matching** 算法将形变的 nodes 拟合回 rigid transformation (translation 和 rotation)，保证物体的刚体属性。

### 3. 实验结果与 Intuition 分析

#### 3.1 小规模场景的 Accuracy vs. Efficiency
在 Kubric Movi-B/C datasets 上（3-10 个 objects），SDF-Sim 与 SOTA baseline FIGNet* 进行了对比。

**数据解析 (Movi-C dataset):**
| Model | Translation RMSE (m) | Rotation Err (deg) | # Collision Edges | Peak Memory (MiB) | Runtime/step (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| FIGNet* | 0.18 ± 0.0097 | 20.68 ± 0.6483 | 2220.96 ± 469 | 75.87 ± 9.587 | 0.30 ± 0.0163 |
| SDF-Sim | 0.24 ± 0.0151 | 23.15 ± 0.668 | **300.46 ± 34** | **43.35 ± 0.7466** | **0.17 ± 0.0038** |

**Intuition:** SDF-Sim 在纯 accuracy 上略微逊色于 FIGNet*（0.24m vs 0.18m translation error）。这非常 reasonable：SDF MLP 的 query 存在微小的 approximation error，且 SDF-Sim 丢失了 mesh triangles 的 explicit topological connectivity。然而，SDF-Sim 的 collision edges 数量减少了近 7 倍，runtime 减少了 43%，memory 减少了 42.8%。在 rigid-body dynamics 中，微小的 position error 会在多次 collision 后被放大，但 0.24m 的 error 仅占 Movi-C 中物体平均移动距离 (4.92m) 的 4.9%，在视觉合理性上完全可以接受。

#### 3.2 超大规模场景的 Scaling
在 Spheres-in-Bowl 实验中，作者将 sphere 数量从 1 增加到 512。
- FIGNet 在 140 个 objects 时 OOM，FIGNet* 在 160 个 objects 时 OOM。
- SDF-Sim 成功模拟了 512 个 spheres 互撞的场景。Runtime 比 FIGNet* 快 5 倍。
- 更震撼的是，在 "Falling shoes" (300 objects, 851k nodes) 和 "Heaps of Stuff" (380 objects, 1.1M nodes) 场景中，SDF-Sim 跑出了 200 步的 realistic rollouts。这是 learned simulator 首次达到 million-node 级别。

**Intuition:** 这里体现了 generalization 的威力。SDF-Sim 仅在 10 个 objects 的场景上训练，却能 scale 到数百个 objects。原因在于 SDF-Sim 学到的是 local pairwise interaction 的物理规律。只要 local collision mechanics 学得准，global scene 的规模无论多大，都可以通过 message passing 组合起来。这种 **locality** 和 **compositionality** 是 GNN 的核心 inductive bias。

#### 3.3 Ablation: SDF Quality 的影响
作者训练了不同 size 的 SDF MLPs (32, 64, 128 hidden units)。
- 128 units 的 SDF MSE 最低，shape 最精细。
- 但令人惊讶的是，无论 SDF size 多大，SDF-Sim 的 translation/rotation error 几乎一样。

**Intuition:** 这说明 learned simulator 对 SDF 的 noise 具有极强的 robustness。GNN 能够在 message passing 中隐式地 correct 掉 SDF approximation 带来的微小距离误差。只要 SDF 提供的大体几何轮廓和 gradient 方向是对的，网络就能预测出正确的 collision impulse。这也解释了为什么用 vision (VolSDF) 重建出来的带 noise 的 SDF，依然能跑出不错的 simulation。

#### 3.4 Vision Integration
作者使用了 Mip-NeRF 360 的 Garden scene，用 VolSDF 从 multi-view RGB images 中 distill 出 table 和 vase 的 mesh，然后训练成 SDF，丢进 SDF-Sim 中与虚拟 shoe 进行碰撞。尽管 mesh 有 80k vertices（FIGNet 会 OOM），SDF-Sim 依然成功模拟了复杂的碰撞细节。

### 4. 拓展联想与未来方向

1. **Amortized SDF Training:** Paper 目前的 limitation 是需要为每个 object 单独 pre-train 一个 SDF MLP (18 hours on V100)。未来可以引入 **DeepSDF** 或 generative models (如 3D Shape Diffusion)，用一个 conditional generative model 一次性表征所有 shapes，实现 zero-shot shape acquisition 和 simulation。
2. **Differentiable Physics Pipeline:** SDF-Sim 目前的 simulator 是 learned GNN。若能将 SDF query 直接嵌入到 differentiable physics engine（如 DiffTaichi 或 Brax）中，用 analytic collision solver 结合 SDF gradient，就能实现端到端的 system identification 和 model-based RL，这会对 robotics 产生巨大影响。
3. **Dynamic Topology & Deformable Objects:** 当前工作局限于 rigid-body。SDF 的 extension（如 Neural Impulse Fields）有望处理 deformable objects 或 fluids。想象一下，将 fluid simulation 也用 implicit neural field 表达，结合 GNN，或许能模拟大量 fluid-solid interaction。

**Reference Web Links:**
- SDF-Sim Project Page & Videos: https://sites.google.com/view/sdf-sim
- FIGNet (Baseline) Paper: https://openreview.net/forum?id=IbmjGDnY6-
- Kubric Dataset: https://github.com/google-research/kubric
- VolSDF: https://github.com/lioryariv/volsdf
- DeepSDF: https://github.com/facebookresearch/DeepSDF
- Shape Matching for Rigid Bodies: https://dl.acm.org/doi/10.1145/1073204.1073216

总结来看，SDF-Sim 通过引入 implicit shape representation，彻底解除了 learned simulator 在 mesh density 上的 curse of dimensionality，将物理模拟的 scale 提升了几个数量级，同时展示了 vision-to-physics 的巨大潜力。这是一项极具启发性的工作。
