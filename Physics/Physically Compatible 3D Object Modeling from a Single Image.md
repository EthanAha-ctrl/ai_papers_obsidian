---
source_pdf: Physically Compatible 3D Object Modeling from a Single Image.pdf
paper_sha256: 3046b361c4e4ec64a88dae07509a6c66a35ec6df85c602968e7a05104c5633cf
processed_at: '2026-08-06T03:23:47-07:00'
target_folder: Physics
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

## 一个生活场景的类比

你拍了一张猫咪跳起来的照片，猫咪在空中弓着背、四肢张开、尾巴甩到一边。然后你想用橡皮泥捏一个一模一样的猫咪——但你捏出来的是照片里那个"跳跃中的姿态"，结果橡皮泥猫就保持那个怪异姿势摆在那儿，不会动，看起来非常别扭。

正确的做法应该是：捏一个"猫咪放松站着的姿势"（rest shape），然后想个办法让它跳到照片里那个姿态（施加力让它变形）。

这篇 paper 干的就是这件事，只不过把橡皮泥换成了 3D 模型，把"跳跃"换成了"gravity 下自然下垂"。

## 现有方法的问题

你给一张图片，[Wonder3D](https://github.com/xxlong0/Wonder3D)、[LGM](https://github.com/3DTopia/LGM)、[TripoSR](https://github.com/VAST-AI-Research/TripoSR) 这些 single-view reconstruction model 都能帮你生成一个 3D shape。它们生成的 shape 看起来跟 image 很像，确实漂亮。

但问题在于：image 里你看到的那朵花，花瓣是微微下垂的——因为 gravity 把它拉下来了。而 reconstruction model 不知道这一点，它以为"花本来就长这样"，直接把下垂的花瓣当成 rest shape。

后果就是你 3D print 出来之后，gravity 又把这个花瓣往下拉一次，于是花瓣下垂得更厉害，完全偏离了 image。或者更惨，整个 object 质心不对，直接 topple 倒下。

[Project page](https://gmh14.github.io/phys-comp/) 上 Figure 1 的对比图就是讲这个——上面那行是 baseline 的结果，花会倒、椅子会塌；下面那行是这篇 paper 的结果，稳稳的。

## 核心思路

image 其实是一个"物理 snapshot"——物体在 gravity 作用下达到 static equilibrium 的状态。物体的 geometry 由三个东西共同决定：

1. **Material properties**：这个物体多硬、多软、密度多大（Young's modulus E、Poisson's ratio ν、density ρ）
2. **External forces**：主要是 gravity，可能还有固定 attachment 点的力
3. **Rest shape**：没有任何力作用时的"放松状态"

这三个东西组合起来，经过 physics simulation，就得到你 image 里看到的 geometry。

现有方法只看到 image geometry，忽略了 1 和 2，直接拿它当 rest shape——这就是错的。

这篇 paper 说：用户告诉我 material 和 external forces（3D printing 时这俩本来就要指定），我来求 rest shape，让这个 rest shape 在 gravity 下自然 settle 到的 state 正好匹配 image。

## 怎么求 rest shape

直接拿 mesh vertex 位置当优化变量有个麻烦：优化过程中四面体可能"翻面"（inversion），变成 inside-out，物理就崩了。传统做法要加一堆 nonlinear inequality constraint，优化很麻烦。

作者借用了 [Bohan Wang 之前 anatomy modeling 的工作](https://jamesyanhua.github.io/) 的一个聪明 trick：不要直接动 vertex 位置，而是定义一个 "plastic deformation field" F_p——每个 element 一个 3×3 矩阵，描述"这个 element 被永久捏成了什么样"。把 F_p 应用到初始 mesh 上，就得到 rest shape。

好处是：
- **Translation invariance**：mesh 摆在哪不影响结果
- **防 inversion 很简单**：只要 F_p 的 singular values 都是正的就行，一行代码搞定
- **Isotropic material 下自由度减半**：F_p 可以简化成 symmetric matrix，9 个数变 6 个数
- **平滑正则化**：用 graph Laplacian 让相邻 element 的 plastic strain 连续，不会东捏一块西捏一块

然后把它套进一个 constrained optimization：

$$\min_{\mathbf{F}_p, \mathbf{x}_{\mathrm{static}}} \mathcal{L}(\mathbf{x}_{\mathrm{static}}) + \mathcal{L}_{\mathrm{reg}}(\mathbf{F}_p) \quad \text{s.t.} \quad \mathbf{f}_{\mathrm{int}} = \mathbf{f}_{\mathrm{ext}}$$

loss 部分是让 static shape 匹配 image geometry，加上让质心稳定不要 topple。Constraint 是 **hard constraint**——static equilibrium 必须严格满足。

## 为什么 hard constraint

很多人遇到 constrained optimization 的第一反应是：把 constraint 当 penalty 加到 loss 里就完事了，比如 $\min \mathcal{L} + \lambda \|\mathbf{f}_{\mathrm{int}} - \mathbf{f}_{\mathrm{ext}}\|^2$。

但这样有问题：λ 大了优化 ill-conditioned 走不动，λ 小了 constraint 永远不严格满足，物理还是不兼容。永远在 numerical stability 和 physical accuracy 之间妥协。

作者用 **implicit differentiation** 来算 gradient，保证每一步优化都在"严格满足 equilibrium 的 manifold"上走。这是 [Deep Equilibrium Models](https://arxiv.org/abs/2006.08695) 在 physics simulation 上的应用，也等价于 differentiable physics 的 adjoint state method，参考 [Burczynski 1997](https://doi.org/10.1016/S0045-7825(96)01116-4)。

具体推导在 Appendix C，但实际计算时不需要真的 invert 那个巨大的 stiffness matrix——用 sparse linear solver 解一个 adjoint 形式的线性系统就行，avoid 显式 invert 大矩阵。

## 实验

作者从 [Objaverse](https://objaverse.allenai.org/) 选了 100 个 plant/animal/character 类的 shape（这些类别对物理兼容性要求高），render 成 image，喂给 5 个 baseline reconstruction model，然后对比 baseline 输出 vs 加上本方法优化后的输出。

5 个 metric 全部针对 physics:
- **#CC**：连通分量数，物体不应该有 floating parts
- **Mean Stress**：von Mises 应力，高说明有易断的薄结构
- **Standability**：gravity 下能保持稳定不倒的比例
- **Img. Loss**：gravity 下 rendered image 和 input image 的 L1 距离
- **Fracture Rate**：超过应力 threshold 的 element 比例

结果表里最 dramatic 的是 Wonder3D：Standability 从 6.9% 提到 72.4%，Mean Stress 从 10.68 kPa 降到 0.45 kPa。LGM 的 Standability 从 20.3% 飙到 85.5%。

[TetSphere](https://arxiv.org/abs/2411.06046)（作者前一作，直接输出 volumetric tetrahedral mesh）效果最好，因为不需要从 surface mesh 转 tetrahedral mesh 这个 lossy 步骤。

## 一个最 elegant 的 ablation

Figure 5 我觉得最惊艳：同一张 input image，用不同的 Young's modulus（5e4 到 5e5 Pa）去优化，得到 6 个不同的 rest shape。但这 6 个 rest shape 在 gravity 下全部 settle 到同一个 static shape——完美匹配 image。

这恰好展示了问题的 ill-posed 性：多组 (material, rest shape) 等价同一个 image。

然后给每个物体施加一个相同的 compression force（黄色 box 压一下），因为 material 不同，deformation 行为完全不同。这就实现了"用 material 控制物体 dynamic 行为，rest shape 自动调整匹配 image"的 controllability——用户只要说"我要橡胶的"或者"我要硬塑料的"，framework 自动算出对应的 rest shape。

## Application

### Dynamic simulation
出来的 rest shape 直接塞进 dynamic FEM simulator，用 implicit backward Euler time discretization（Appendix F 的 Eq 10-11）。Attachment force 用 spring force，contact force 用 [IPC](https://ipc-sim.github.io/)。最后还加 4D [Perlin noise](https://en.wikipedia.org/wiki/Perlin_noise) 风力让植物摇晃。

### 3D printing
用 [Formlabs Form3](https://formlabs.com/3d-printers/form-3/) stereolithography：
- Flexible 80A 树脂（tensile modulus < 3 MPa）打印软花
- White Resin V4（1.6 GPa）打印硬鹅

实物照片跟 computation 结果对得上，未优化的对照组实物 sag 或 topple 了。

## 我觉得最 cool 的几个点

### 1. 问题 formulation 本身的优雅
把 reconstruction 从"几何拟合"重新定义成"inverse physics"——不是在复制 image，而是在反推产生 image 的物理 setup。这跟 [Pac-NeRF](https://arxiv.org/abs/2305.02222)、[PhysGaussian](https://arxiv.org/abs/2311.12198) 那条线思路一致，但首次用在 single image + static equilibrium 上。

### 2. Plastic strain 作为 parameterization 的工程价值
[Bohan Wang 2021 anatomy paper](https://jamesyanhua.github.io/) 那个 plastic strain trick 在这里找到了新的应用场景。把一个困难的 nonlinear inequality constraint 问题转化成 singular value 约束，工程上特别干净。

### 3. Implicit differentiation 让它能 plug 进 deep learning pipeline
用 adjoint state method 算 gradient，跟 [NEq](https://arxiv.org/abs/1909.01377)、[DEQ](https://arxiv.org/abs/2006.08695) 思路一致，可以接 Adam optimizer。这意味着未来可以 fine-tune LRM 让它直接输出 physically compatible shape。

### 4. Native volumetric representation 的胜利
TetSphere 在所有 metric 上都 best 不是巧合——它在 representation 层面就避免了 surface-to-volume 转换的 lossy 过程。这呼应了 [differentiable iso-surfacing](https://arxiv.org/abs/2305.02283) 那条线的趋势。

## Limitations 和我的联想

1. **Material 和 external forces 还得用户指定**。未来可以从 image 自动 estimate material（参考 [image2mass](https://arxiv.org/abs/1709.04805)、[Intrinsic images in the wild](https://opensurfaces.cs.cornell.edu/intrinsic/)），或者从 LLM 估计（"植物应该是软的"、"石头应该是硬的"）。

2. **依赖 tetrahedral mesh**，需要 [TetWild](https://github.com/Yixin-Hu/TetWild) 转换，lossy。如果有个 differentiable converter，就能把 physical compatibility loss 直接 backprop 到 LRM weights，端到端训练 physically-aware reconstruction model。这是 future work 里作者自己提的，也是最有想象空间的方向。

3. **只处理 static equilibrium**，dynamics（video input）是自然扩展。

4. **没 model contact**：如果 image 里物体正坐在椅子上，static equilibrium 其实包含 contact force。本方法只 model gravity + attachment，会让 rest shape 过度补偿。

### 我想到的延伸方向

**Constrained 3D diffusion**：把这套 framework 套到 generative model 上——diffusion 不直接生成 image geometry，而是生成 rest-shape geometry 的分布，然后 physical compatibility 作为 [classifier guidance](https://arxiv.org/abs/2105.05233) 或 [constrained diffusion](https://arxiv.org/abs/2206.00964) 的 guidance term。这样 generation 出来的 shape 天生 physically plausible。

**Neural plastic strain field**：把 F_p 换成 neural network（implicit neural representation of plastic deformation），让 rest shape 的 prior 来自 learned generative model 而不是 hand-crafted bi-harmonic regularizer。类似 [NeRF](https://arxiv.org/abs/2003.08934) 把 geometry 用 MLP encode 的思路。

**Material discovery from image**：让 E, ν, ρ 也成为优化变量，给定 image 和一类 candidate materials，自动选最 plausible 的那一个。这其实就是把 Equation 2 的 Θ 也变成 variable，再用一个 material prior regularize。

**Scene-level physical compatibility**：现在只处理 single object，但很多 image 是 multi-object scene with contact、support、stacking。把 framework 扩到 scene level，处理 object-object interaction，contact force 从 environment 来——这就接近 [real-world scene reconstruction](https://arxiv.org/abs/2401.05677) 的物理版了。

## 一句话总结

之前所有 single-view reconstruction 方法都在做"几何复制"，这篇 paper 重新定义成"物理反演"——image 是物理过程的结果，要反推 setup。用 plastic strain parameterization + implicit differentiation 把 inverse elastic design 接到现代 reconstruction pipeline 上，让出来的 3D shape 真的能 3D print 出来、能 simulate 出来、还稳。

---

# Physically Compatible 3D Object Modeling from a Single Image — 深入解析

## 1. 一句话核心

Existing single-view reconstruction methods（Wonder3D、LGM、MeshLRM、TripoSR、TetSphere）只复制 image 里的 visual geometry，但 image 是物体在 **static equilibrium** 下的 snapshot，本身已经被 external force（gravity）deform 过了。所以如果你直接把 reconstructed geometry 当 rest shape 去 3D print / simulate，它会继续 deform 一次，要么 topple 要么 sag，完全偏离 input image。本文把 reconstruction 重构为 inverse elastic design 问题：给定 mechanical properties Θ 和 external forces f_ext，去优化 rest-shape geometry X_rest，使得 X_rest 在 gravity 下 settle 到的 static geometry x_static 正好匹配 image。Project page: https://gmh14.github.io/phys-comp/

## 2. 核心 insight：三个 orthogonal attributes

物体在 image 中呈现的 geometry = 三个 orthogonal 物理 attributes 的合成：
- **Mechanical properties** Θ = {Young's modulus E, Poisson's ratio ν, density ρ}
- **External forces** f_ext = {gravity, fixed attachment forces}
- **Rest-shape geometry** X_rest = 没有任何力作用时的形状

这是一个 ill-posed inverse problem：同一个 static geometry x_static 可以由无穷多组 (Θ, f_ext, X_rest) 组合产生。现有方法 implicit 假设物体 rigid 且无外力，于是直接拿 image geometry 当 rest shape——这就是问题根源。作者把 Θ 和 f_ext 视为 user-specified input（这是 3D printing / simulation 用户实际能给出的信息），求解 X_rest。

## 3. 公式逐一解析

### Equation 1: Static equilibrium (Newton 第三定律在 deformable body 上的形式)

$$\mathbf{f}_{\mathrm{int}}(\mathbf{x}_{\mathrm{static}}, \mathbf{X}_{\mathrm{rest}}; \Theta) = \mathbf{f}_{\mathrm{ext}}(\mathbf{x}_{\mathrm{static}})$$

- **f_int**: 内部弹性力，把物体从 rest shape X_rest 拉到 current shape x_static 产生的恢复力。维度映射 $\mathbb{R}^{3N} \times \mathbb{R}^{3N} \to \mathbb{R}^{3N}$，N 是 mesh 顶点数。
- **f_ext**: 外部力（这里主要是 gravity $\mathbf{M}\mathbf{g}$）。$\mathbb{R}^{3N} \to \mathbb{R}^{3N}$
- **Θ**: material 参数集
- **x_static**: image 中观测到的形状（deformed state）
- **X_rest**: 无外力 rest state

物理直觉：图像里的"杯子边缘下垂"不是"杯子本来长这样"，而是"杯子本来更直 + gravity 把边缘拉下垂 + 内部弹性力刚好和 gravity 抵消"。

### Equation 2: Constrained optimization

$$\begin{array}{rl}
\min_{\mathbf{X}_{\mathrm{rest}}, \mathbf{x}_{\mathrm{static}}} & \mathcal{I}(\mathbf{X}_{\mathrm{rest}}, \mathbf{x}_{\mathrm{static}}) = \mathcal{L}(\mathbf{x}_{\mathrm{static}}) + \mathcal{L}_{\mathrm{reg}}(\mathbf{X}_{\mathrm{rest}}) \\
\mathrm{s.t.} & \mathbf{f}_{\mathrm{int}}(\mathbf{x}_{\mathrm{static}}, \mathbf{X}_{\mathrm{rest}}; \Theta) = \mathbf{f}_{\mathrm{ext}}(\mathbf{x}_{\mathrm{static}})
\end{array}$$

**Loss $\mathcal{L}(\mathbf{x}_{\mathrm{static}})$** 有两种 task:
1. **Image matching**: $\mathcal{L} = \|\mathbf{x}_{\mathrm{static}} - \mathbf{X}_{\mathrm{target}}\|_2^2$ — point-wise Euclidean 距离，让 static shape 对齐 target geometry $\mathcal{M}_{\mathrm{target}}$（来自 baseline reconstruction model 的输出）
2. **Stability**: $\mathcal{L} = \|\mathrm{proj}_z(\mathcal{C}(\mathbf{x}_{\mathrm{static}})) - \hat{\mathcal{C}}\|$ — $\mathcal{C}(\cdot)$ 计算质心，$\mathrm{proj}_z$ 投影到地面 z-plane，$\hat{\mathcal{C}}$ 是 desired 质心位置（保证物体不会 topple）

**Regularizer $\mathcal{L}_{\mathrm{reg}}$**: 后面替换为 plastic strain 的 bi-harmonic smoothness。

关键设计是 hard constraint——绝不能转成 penalty 加到 loss 里，否则 constraint 永远不会完美满足，物理兼容性就崩了。

### Equation 3 & 4: Plastic strain field parameterization

直接优化 X_rest 会有 **inversion problem**（四面体翻转、inside-out element），传统做法要加 nonlinear inequality constraints，优化不可 tractable。作者借用 [Wang et al. 2021 plastic strain for anatomy](https://jamesyanhua.github.io/) 的思路：把 X_rest 写成对初始 mesh 施加 plastic deformation 的结果：

$$\mathbf{X}_{\mathrm{rest}} := \phi(\mathbf{F}_p; \mathbf{X}_{\mathrm{init}}), \quad \mathrm{with} \quad \mathbf{f}_{\mathrm{int}}(\mathbf{X}_{\mathrm{rest}}, \mathbf{X}_{\mathrm{init}}; \Theta) = \mathbf{0}$$

- **F_p**: plastic strain field，每个 material point 一个 $3\times 3$ 矩阵；向量化为 $\mathbb{R}^{9Z}$，Z 是 element 数
- **φ**: 把 F_p 应用到 X_init 得到 X_rest 的映射
- 约束 $\mathbf{f}_{\mathrm{int}} = \mathbf{0}$ 表示 plastic 过程没有外力，rest shape 自己 settle

替换后得到 Equation 4:
$$\begin{array}{rl}
\min_{\mathbf{F}_p, \mathbf{x}_{\mathrm{static}}} & \mathcal{I}(\mathbf{F}_p, \mathbf{x}_{\mathrm{static}}) = \mathcal{L}(\mathbf{x}_{\mathrm{static}}) + \mathcal{L}_{\mathrm{reg}}(\mathbf{F}_p) \\
\mathrm{s.t.} & \mathbf{f}_{\mathrm{int}}(\mathbf{x}_{\mathrm{static}}, \phi(\mathbf{F}_p; \mathbf{X}_{\mathrm{init}}); \Theta) = \mathbf{f}_{\mathrm{ext}}(\mathbf{x}_{\mathrm{static}})
\end{array}$$

其中 $\mathcal{L}_{\mathrm{reg}}(\mathbf{F}_p) = \|\mathbf{L}\mathbf{F}_p\|_2^2$，$\mathbf{L} \in \mathbb{R}^{9Z \times 9Z}$ 是 graph Laplacian（volumetric mesh 上的 bi-harmonic energy）。

**F_p 的三个关键优势**（Appendix B 推导）:

1. **Translation invariance**: X_init 平移不影响 deformation gradient F = ∂x/∂X_init，所以 F_p 自动与位置无关，无需担心 X_init 摆放位置。

2. **Non-inversion 通过 singular values 简单约束**: 因为 det(F_p) > 0 即可避免 inversion。

3. **Isotropic material 简化**: F_p = RS（polar decomposition），R 在 isotropic 下 redundant（因 $\Phi(\mathbf{F}_e; \theta) = \Phi(\mathbf{F}_e \mathbf{R}; \theta)$），所以 F_p 是 symmetric matrix，只需约束 eigenvalues > 0。

### Appendix B: 弹性能和弹性力

$$\mathcal{E}(\mathbf{F}_e, \mathbf{F}_p; \Theta) = V(\mathbf{F}_p) \Phi(\mathbf{F}_e; \Theta)$$
$$V(\mathbf{F}_p) = V_{\mathrm{init}} \det(\mathbf{F}_p), \quad \mathbf{F}_e = \mathbf{F}\mathbf{F}_p^{-1}, \quad \mathbf{F} = \partial \mathbf{x}/\partial \mathbf{X}_{\mathrm{init}}$$

- **V(F_p)**: plastic strain 之后的 element 体积
- **V_init**: 初始体积
- **F_e**: elastic deformation gradient（elastic 部分）
- **F**: total deformation gradient（current x 相对 init）
- **Φ**: elastic energy density function (isotropic Neo-Hookean, [Smith et al. 2018](https://arxiv.org/abs/1710.00599))

弹性力（对 x 求导）:
$$\mathbf{f}_{\mathrm{int}}(\mathbf{x}, \phi(\mathbf{F}_p; \mathbf{X}_{\mathrm{init}}); \Theta) = V(\mathbf{F}_p) \frac{\partial \Phi}{\partial \mathbf{F}_e} : \frac{\partial \mathbf{F}}{\partial \mathbf{x}} \mathbf{F}_p^{-1}$$

注意 $\partial \mathbf{F}/\partial \mathbf{x}$ 是常数（F 对 x 线性），所以求解 static equilibrium 是一个 Newton-Raphson 解非线性方程。

### Equation 5: Implicit differentiation 的梯度

$$\frac{\partial \mathcal{I}}{\partial \mathbf{F}_p} = -\left(\frac{\partial \mathcal{L}}{\partial \mathbf{x}_{\mathrm{static}}}\right) \left[\frac{\partial \mathbf{f}_{\mathrm{net}}}{\partial \mathbf{x}_{\mathrm{static}}}\right]^{-1} \frac{\partial \mathbf{f}_{\mathrm{net}}}{\partial \mathbf{F}_p} + \frac{\partial \mathcal{L}_{\mathrm{reg}}}{\partial \mathbf{F}_p}$$

其中 $\mathbf{f}_{\mathrm{net}} = \mathbf{f}_{\mathrm{int}} - \mathbf{f}_{\mathrm{ext}}$ 是净力。

推导（Appendix C）：对 hard constraint 关于 F_p 求全微分:
$$\frac{\partial \mathbf{f}_{\mathrm{net}}}{\partial \mathbf{F}_p} + \frac{\partial \mathbf{f}_{\mathrm{net}}}{\partial \mathbf{x}_{\mathrm{static}}} \frac{\partial \mathbf{x}_{\mathrm{static}}}{\partial \mathbf{F}_p} = 0$$
$$\Rightarrow \frac{\partial \mathbf{x}_{\mathrm{static}}}{\partial \mathbf{F}_p} = -\left[\frac{\partial \mathbf{f}_{\mathrm{net}}}{\partial \mathbf{x}_{\mathrm{static}}}\right]^{-1} \frac{\partial \mathbf{f}_{\mathrm{net}}}{\partial \mathbf{F}_p}$$

然后 chain rule 代入 objective 得到 Eq 5。

**实际计算 trick**:
1. $\partial \mathbf{f}_{\mathrm{net}}/\partial \mathbf{x}_{\mathrm{static}}$ 是 sparse 矩阵（stiffness matrix）
2. 先用 sparse linear solver 解 $(\partial \mathcal{L}/\partial \mathbf{x}_{\mathrm{static}}) [\partial \mathbf{f}_{\mathrm{net}}/\partial \mathbf{x}_{\mathrm{static}}]^{-1}$ 得到 dense vector size 3N（adjoint 形式，避免显式 invert 大矩阵）
3. 再乘 sparse 矩阵 $\partial \mathbf{f}_{\mathrm{net}}/\partial \mathbf{F}_p$

这其实是经典 adjoint state method / reverse-mode sensitivity analysis，见 [Burczynski et al. 1997](https://doi.org/10.1016/S0045-7825(96)01116-4)。等价于 differentiable simulation 的反向传播。

## 4. Pipeline 架构（Figure 2）

```
Input Image ──┐
              ▼
   [Single-view reconstruction model] ──► M_target (surface/point/NeuS/...)
              │
              ▼
   [TetWild tetrahedralization] ──► M_init (volumetric tet mesh)
              │
              │  + Material Θ (E, ν, ρ)
              │  + External forces f_ext (gravity, attachment)
              ▼
   [Plastic strain optimization loop]
        │
        ├─ F_p → φ → X_rest
        ├─ Newton-Raphson solve: f_int(x_static, X_rest) = f_ext(x_static) → x_static
        ├─ L(x_static) = image match + stability
        ├─ Implicit differentiation backward → ∂I/∂F_p
        └─ Adam update F_p
              │
              ▼
   Rest-shape geometry + stress heatmap
```

Material 用 isotropic Neo-Hookean（[Smith 2018](https://arxiv.org/abs/1710.00599)），Young's modulus E ∈ {5e4, 5e5 Pa}、Poisson ν = 0.45、density ρ = 1000 kg/m³。

Newton-Raphson 解 static equilibrium 用 line search，max 200 iterations；F_p 优化用 gradient descent (Adam) 最多 1000 iterations；总耗时约 80 秒/shape on AMD Ryzen 9 5950X + 64GB RAM。

## 5. Evaluation metrics

现有 metrics（PSNR、SSIM、Chamfer distance、Volume IoU）只看 visual/geometric 质量，看不到 physical compatibility。作者从 FEM 借鉴了 5 个新 metrics:

| Metric | 含义 | 越低/高好 |
|---|---|---|
| #CC (Number of Connected Components) | 结构完整性，floating parts 多说明差 | ↓ |
| Mean Stress (von Mises, kPa) | 平均应力，高说明薄结构 / 易断 | ↓ |
| Standability (%) | 重力下保持稳定的 shape 比例 | ↑ |
| Img. Loss (l1) | rendered image vs input image 距离 | ↓ |
| Fracture Rate | stress > threshold 的 element 比例（曲线下面积） | ↓ |

[von Mises stress](https://en.wikipedia.org/wiki/Von_Mises_yield_criterion) 是 1913 年的 yield criterion，工程上判定 ductile material 是否 yield。

## 6. Baselines 和实验结果

五个 baseline，每个对应一种 geometry representation:

| Method | Geo. Rep | #CC | Mean Stress | Standable | Img. Loss |
|---|---|---|---|---|---|
| Wonder3D | NeuS | 2.54 ± 2.64 | 10.68 → 0.45 kPa | 6.9 → 72.4 % | 0.073 → 0.069 |
| LGM | Gaussian splatting | 2.67 ± 2.13 | 1.14 → 1.01 kPa | 20.3 → 85.5 % | 0.121 → 0.116 |
| MeshLRM | surface mesh | 1.55 ± 2.13 | 0.54 → 0.38 kPa | 29.6 → 74.5 % | 0.065 → 0.064 |
| TripoSR | NeRF triplane | 1.43 ± 1.12 | 0.29 → 0.22 kPa | 24.2 → 80.6 % | 0.066 → 0.059 |
| TetSphere | tet-sphere | **1.00 ± 0.00** | 0.22 → **0.19** kPa | 32.8 → **92.2** % | 0.061 → **0.057** |

[Baselines](https://github.com/MrNeRF/awesome-3D-generation):
- [Wonder3D](https://arxiv.org/abs/2310.15072): multi-view diffusion + NeuS reconstruction
- [LGM](https://arxiv.org/abs/2402.06054): Large Gaussian Model, multi-view → Gaussian splatting
- [MeshLRM](https://arxiv.org/abs/2404.12385): LRM 直接输出 mesh
- [TripoSR](https://arxiv.org/abs/2403.02151): Feedforward, NeRF triplane
- [TetSphere](https://arxiv.org/abs/2411.06046): 作者前一作，tetrahedral spheres（volumetric，原生 best）

**关键观察**:
1. **Geometry representation 强烈影响 structural integrity**: LGM (point cloud) 因为分不清内外，#CC 高、有 floating parts；TetSphere 因为 volumetric，#CC 始终 1。
2. **MeshLRM 和 TripoSR** Standability 较好但 Img. Loss 大（gravity 下 sagging 偏离 image）。
3. **本方法在所有 baseline 上都改善所有 metric**，最显著的是 Wonder3D 和 MeshLRM（这两个常出 thin structures）。
4. **Fracture rate**（Figure 3）显示 MeshLRM 和 TripoSR 的 thin structure 大量 element 超应力，本方法显著降低。

## 7. Ablation: Young's modulus 的可控制性

Figure 5 是我最喜欢的 ablation：同一 image，用不同 E（5e4 ~ 5e5 Pa）优化得到不同 X_rest，但所有 X_rest 在 gravity 下 settle 到同一个 x_static（匹配 image）。这验证了问题的 ill-posed 性质——多组 (Θ, X_rest) 等价。

但加上额外 compression force 后，不同 E 的物体表现不同 deformation。这意味着 user 可以通过指定 material 来控制物体的 dynamic 行为，rest shape 自动调整来匹配 image。这是非常 elegant 的 controllability。

## 8. Applications

### Dynamic simulation
用 implicit backward Euler time discretization（Appendix F, Eq 10-11）:
$$\mathbf{M}\ddot{\mathbf{x}} + \mathbf{D}(\mathbf{x})\dot{\mathbf{x}} + \mathbf{f}_{\mathrm{elastic}}(\mathbf{x}) + \mathbf{f}_{\mathrm{attachment}}(\mathbf{x}) + \mathbf{f}_{\mathrm{contact}}(\mathbf{x}) = \mathbf{M}\mathbf{g}$$

- M: mass matrix
- D: Rayleigh damping
- f_attachment = k_a(Sx - x̄(t)): spring force，S 是 selection matrix，x̄(t) 是 keyframe-animated 的 attachment target
- f_contact = k_c·d: contact penalty force，d 是 penetration depth；friction 用 [IPC](https://ipc-sim.github.io/)

每个 timestep 用 Newton's method 解非线性方程。End of sequence 加 4D Perlin noise 风力。

### 3D printing (Appendix E)
Formlabs Form3 stereolithography, 100 μm layer thickness:
- **Flexible 80A**: tensile modulus < 3 MPa, 100% strain to failure（柔性花，55/65 mm）
- **White Resin V4**: tensile modulus 1.6 GPa（刚性 goose，50 mm）

验证优化后的实物确实匹配 image，未优化的实物 sag/topple。

## 9. Limitations 和未来方向

1. **Material properties 和 external forces 需要用户预设**: 未来方向是 from single image 自动 estimate Θ 和 f_ext（类似 [image2mass](https://arxiv.org/abs/1709.04805)、[Intrinsic images in the wild](https://opensurfaces.cs.cornell.edu/intrinsic/)）。

2. **依赖 tetrahedral mesh**: TetWild 转换 lossy，未来需要 differentiable converter 把 any geometry representation 直接转 tetrahedral，从而能把 physical compatibility loss 反传到 LRM 的 weights 做 fine-tuning，端到端训练 physically compatible reconstruction model。

3. **只处理 static equilibrium**: Video 输入捕获 dynamics 是自然扩展（参考 [Pac-NeRF](https://arxiv.org/abs/2305.02222)、[PhysGaussian](https://arxiv.org/abs/2311.12198)、[Spring-Gaus](https://arxiv.org/abs/2403.09434)）。

## 10. 我对这篇文章的 intuition build

### 类比
想象你在画一个跳跃的篮球运动员。你画的是"跳跃到最高点的那一刻"——腿弯曲、衣服飘起、肌肉拉伸。这是 dynamics 的某个 frame。如果你把这张画当成"rest pose"去做一个雕塑，雕塑自己当然不会跳跃，它就是站成那个怪异姿势。正确的做法是 sculpt 一个 "rest pose"（直腿、衣服下垂、肌肉放松），然后再设计一个能让它在某个瞬间达到画中姿势的"力"。

Single-view reconstruction 之前都在做"把画当成 rest pose 雕塑"的错事。

### 更深的技术 intuition

这个 work 本质上是 **inverse elastic shape design**（[Chen et al. 2014](https://arxiv.org/abs/1404.4267)）的"image-conditioned"版本。Chen 2014 需要用户手工 design target deformed shape + 一个 manually created rest geometry 作为 initial guess，然后通过 asymptotic numerical method 调整 rest shape 使 equilibrium 状态匹配 target deformed shape。

本文把"target deformed shape"自动来自 single-view reconstruction model 的输出，把"initial guess"设成 target shape 本身（合理：因为 rest shape 通常和 image shape 不差太多，plastic strain 是小 deform）。技术上最大的升级是从 asymptotic numerical method 换成 implicit differentiation + Adam，可以和 deep learning pipeline 无缝集成。

### Implicit differentiation vs Penalty method
为什么 hard constraint 必须用 implicit differentiation 而不是 penalty？

Penalty: $\min \mathcal{L}(\mathbf{x}_{\mathrm{static}}) + \lambda \|\mathbf{f}_{\mathrm{net}}\|^2$
- λ 大 → stiff optimization，ill-conditioned
- λ 小 → constraint 不严格满足，物理不兼容
- 总在 numerical 和 physical 之间妥协

Implicit differentiation 保证每一步 F_p 的 gradient 都是在"严格满足 equilibrium 的 manifold"上的 tangent direction。这是 [DEQ](https://arxiv.org/abs/2006.08695) 思想在 physics simulation 上的应用。

### Plastic strain vs 直接优化 vertex positions

直接优化 X_rest 的 vertex positions 有 inversion problem。Plastic strain F_p 的关键观察:
- F_p 是 element-wise 的，每个 element 一个 3×3 矩阵，比 vertex 位置更"物理"
- Translation invariance: 不需要担心 mesh 在 world coordinate 哪里
- Singular value > 0 一行代码即可防 inversion（vs 复杂的 nonlinear inequality constraints）
- Isotropic material 下退化成 symmetric，自由度从 9 降到 6
- Bi-harmonic regularizer on F_p 自动生成 spatially smooth 的 plastic strain，避免 element 之间的 discontinuity

### 为什么 TetSphere 在本方法下 best?

因为 TetSphere 直接输出 volumetric tetrahedral mesh，没有 surface → volume 的 lossy 转换（TetWild 步骤）。其他方法（surface mesh、point cloud、NeuS、NeRF）都得通过 TetWild 转成 tetrahedral，过程中可能丢失细节或引入 mesh quality 问题。这也提示了"native volumetric reconstruction model"是 physical compatibility 的正确方向。

## 11. 相关工作和未来联想

### 类似的 inverse physics 思路
- [Inverse Elastic Shape Design (Chen 2014)](https://arxiv.org/abs/1404.4267): 手工 initial shape，asymptotic numerical method
- [Sag-free initialization (Hsu 2022)](https://arxiv.org/abs/2204.10364): deformable simulation 初始化问题
- [Pac-NeRF](https://arxiv.org/abs/2305.02222): Physics-augmented continuum NeRF，from multi-view 视频
- [PhysGaussian](https://arxiv.org/abs/2311.12198): 3D Gaussians + 物理仿真
- [Spring-Gaus](https://arxiv.org/abs/2403.09434): spring-mass 3D Gaussians
- [PiE-NeRF](https://arxiv.org/abs/2311.13099): Physics-based interactive elastodynamics NeRF
- [Physical property understanding from language-embedded fields](https://arxiv.org/abs/2404.04242): 从 radiance field 提取 material properties

### Differentiable physics simulation
- [DiffTaichi](https://arxiv.org/abs/1910.00935): Differentiable physics engine
- [Brax / JAX-MD](https://github.com/jax-md/jax-md): JAX-based differentiable molecular dynamics
- [PhiFlow](https://github.com/tum-pbs/PhiFlow): Differentiable fluid simulation
- 这篇 paper 走 implicit differentiation + adjoint，是 differentiable physics 的另一种风格，更适合 hard constraint

### Single-view 重建的未来
- LRM ([Large Reconstruction Model](https://arxiv.org/abs/2310.04400), [OpenLRM](https://github.com/3DTopia/OpenLRM)) feedforward + 本方法的 hard constraint → fine-tuning，端到端 physically-aware LRM
- Differentiable TetWild（[Differentiable Iso-Surfacing](https://arxiv.org/abs/2305.02283) 类）能让 surface-to-volume 转换也参与 backward
- Diffusion prior + physical guidance: 把 static equilibrium 作为 SDE/ODE 的 guidance term，类似 [DreamFusion](https://arxiv.org/abs/2209.14988) 的 SDS 但是加上 physical constraint

### Failure mode 我能想到的
- 物体如果有 contact（比如人坐椅子），image 中的 static equilibrium 其实包含 contact force，本方法只 model gravity + attachment，没 model contact from environment → 会过度调整 rest shape 来补偿
- Cloth/hair 这种 highly deformable 部分，single E, ν 不足以描述
- Material anisotropy（木头有纹理方向）目前只支持 isotropic
- 大型 scene（多 object 互相 contact）本方法单 object 处理

### 我的延伸想法
把这套 framework 套到 generative 3D model 上：先 generate 一个 candidate shape，然后用 physical compatibility optimization "fix" 它，使 generation 变成"physically plausible distribution"。这等价于 [constrained diffusion](https://arxiv.org/abs/2206.00964) 在 3D 生成上的应用——可以训练 diffusion 在 rest-shape manifold 上而不是 deformed-shape manifold 上。

或者更激进：把 F_p parameterization 换成 neural network（neural plastic strain field），让 rest shape 的 prior 来自 learned generative model 而不是 bi-harmonic regularizer。

### 一个有意思的细节
作者把 X_init 设为 M_target 本身。这意味着 rest shape 通常和 image shape "长得像"，但稍微"反方向"补偿 gravity。如果你 print 出来拿着，它会"低头一点、垂一点"刚好和 image 一致——但 X_rest 本身（无 gravity 状态）会比 image 更"挺"。这跟人手工 sculpting 时的直觉相反（雕塑家直接雕"看到的样子"），但物理上是对的。

---

总结一句话：这篇文章把 single-image 3D reconstruction 从"几何复制"重新定义成"物理反演"问题，通过 plastic strain parameterization + implicit differentiation 把 inverse elastic design 跟现代 reconstruction pipeline 干净地接起来。工程实现紧凑、实验充分、application 闭环（simulation + 3D printing 都验证），是 reconstruction → fabrication pipeline 上的一块重要拼图。
