---
source_pdf: Gaussian Frosting Editable Complex Radiance.pdf
paper_sha256: bfdcd4677d794c0b7065c7436f52204243fe4ea9a40b3e8aad934202d75449a2
processed_at: '2026-08-04T12:26:56-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Frosting 人话版

## 一句话说清楚

把 3DGS 拆成两层：**骨架是 mesh（可 edit），肉是 Gaussians（可渲染 hair/grass）**。骨头上贴多厚的肉，看材质 —— 毛发多的地方肉厚，桌面就贴一层皮。

## 为什么需要这个东西

先看现状有多尴尬：

**3DGS** 渲染吊打所有人，速度快得离谱。但 Gaussians 是一堆散兵游勇，你没法跟 artist 说"把这堆 Gaussians 里代表 Buzz 手臂的那 3 万个选中然后旋转 30 度"。artist 要的是 mesh，mesh 有 vertex、有 topology、能 rig、能 skin、能 UV unwrap，是 CG 行业几十年的基础设施。

**SuGaR** 想了个招：把 Gaussians 压扁贴到 mesh 上。这下能 edit 了。但问题来了 —— 你把猫毛、草丛这种本质是体积的东西压成一张皮，渲染就糊了。猫毛没法正确遮挡后面的腿，因为 SuGaR 把毛砍了只剩皮。

所以你面对一个 trade-off：
- 想要渲染好 → 得让 Gaussians 自由飘着当体积
- 想要能 edit → 得把 Gaussians 钉死在 mesh 表面上

Frosting 说：**我全都要**。

## 核心思路

想象你有一块蛋糕（mesh），蛋糕表面凹凸不平。你往上抹一层奶油（Gaussians）：
- 桌面那种光滑地方，奶油抹薄薄一层就行
- 猫毛那种毛茸茸地方，奶油要堆厚厚一堆
- 厚度不是人调的，是自动根据材质算出来的

这层奶油本身就是 Gaussians，用 3DGS 的 rasterizer 渲染，所以速度很快。但每个 Gaussian 被参数化绑在 mesh 的三角形上，mesh 动它就跟着动，所以能 animate。

## 厚度怎么自动算出来

这是 paper 最聪明的地方，思路很 tricky：

你手上同时有两套 Gaussians：
1. **自由版**（unconstrained）：没加约束，Gaussians 自由分布，但充满噪点和 outlier
2. **压扁版**（regularized）：SuGaR 约束过，对齐到 surface，但 fuzzy 区域对齐失败

关键 insight：**压扁版在 fuzzy 区域对齐失败，这个"失败"本身就是信号**。猫毛那里压扁版的 Gaussian 被压不下去，说明那里需要厚度。

具体做法分三步：

**第一步**：对每个 mesh vertex，看最近的压扁 Gaussian 有多"胖"（沿法线方向的标准差 $\sigma$），取 $[-3\sigma, +3\sigma]$ 作为搜索范围。胖的 Gaussian 说明附近是 fuzzy 的，搜索范围大。

**第二步**：在这个范围内找压扁版的等值面（density $\geq 0.01$ 的点），确定一个中间区间 $J$。这一步是用压扁版"圈地" —— 把搜索范围限制在 surface 附近，挡住远处的 outlier。

**第三步**：在圈好的地盘里，找自由版的等值面，最终厚度由自由版决定。因为自由版保留了 fuzzy 区域的真实体积信息，但被压扁版"框住"了，不会被远处的 floater 污染。

用一句话说：**压扁版当导航员指路，自由版当测量员量厚度**。

## Gaussians 怎么绑到 mesh 上

每个 mesh 三角形，沿法线方向往外推、往里推，得到六个顶点，形成一个三棱柱（prism）。Gaussian 就住在这个三棱柱里，位置用 barycentric coordinates 表示。

好处：
- Gaussian **数学上不可能跑出三棱柱**，不用加 penalty
- mesh 变形时，Gaussian 跟着 vertex 走，自动更新
- 用户指定要多少个 Gaussians，就 sample 多少个，数量可控（vanilla 3DGS 是自动 densify，没法控制）

采样时一半均匀撒（照顾 flat 区域要纹理细节），一半按体积撒（照顾 fuzzy 区域要体积效果）。

## 实验结果怎么说

**Shelly 数据集**（专门测 fuzzy material）：

Frosting 39.84 dB，比 vanilla 3DGS 还高 2.18 dB。这很反直觉 —— 加了约束怎么还比没约束的好？因为 Frosting 的 Gaussians budget 花在了刀刃上（fuzzy 区域），而 3DGS 的 densification 是全局 gradient-based 的，会在很多没用的地方也造 Gaussians。

**Mip-NeRF 360**（真实场景）：

比 3DGS 低 0.31 dB，但在所有能 edit 的方法里排第一。Paper 解释当 3DGS 有很好的 COLMAP 初始化时，Frosting 的 densification 优势就不明显了。

**Ablation 里一个有意思的点**：

固定厚度在 fuzzy-only 场景反而比自适应厚度好（40.00 vs 39.84），但在真实场景（有 flat surface）就崩了。这证明厚度必须自适应 —— 不能一刀切。

## 我觉得最值得关注的设计 pattern

1. **把 failure 当 signal**：SuGaR 在 fuzzy 区域对齐失败，本来是 bug，Frosting 拿来当 feature。这种"垃圾数据里淘金"的思路在很多地方有用。

2. **Bottleneck + Residual 结构**：mesh 是 information bottleneck（低频结构），Frosting layer 是 residual capacity（高频细节），thickness 决定 residual budget。这跟 ResNet、Transformer FFN、U-Net skip connection 的哲学一脉相承 —— 都是在 structured backbone 上加 adaptive capacity。

3. **两阶段搜索**：先用 high-precision low-recall 的方法（regularized）圈范围，再用 high-recall low-precision 的方法（unconstrained）填内容。这跟 detection 里 region proposal + classification 的两阶段思路一模一样。

4. **Poisson depth 自动选择**：用 Gaussian 间最近邻距离的 0.1 分位数当 complexity score，而不是用 mean。因为只有"挤在一起 encode 细节的 Gaussians"才是真信号，大量稀疏背景 Gaussians 会把 mean 拉偏。用 quantile 当 robust estimator，很实用的工程经验。

## 一句话总结

Frosting = mesh 骨架 + 自适应厚度 Gaussian 肉层 + barycentric 绑定，三件套让 3DGS 既保持渲染质量又能 rig/animate/composite，而且代码和浏览器 viewer 都会开源。

---

# Gaussian Frosting Paper 深度讲解

Andrej, 这篇 paper 我觉得是 2024 年 Gaussian Splatting 这个方向里非常 elegant 的一篇,因为它解决了一个根本性的 tension: **volumetric methods (3DGS, NeRF) 渲染质量好但 unstructured 无法 edit,而 surface-based methods (mesh) 可 edit 但渲染 fuzzy materials (hair, grass, fur) 时崩掉**。Frosting 的核心 insight 是用一个 adaptive thickness 的 Gaussian layer "包裹" 在 mesh 表面上,相当于在 mesh 上涂一层"糖霜",fuzzy 的地方糖霜厚,flat 的地方糖霜薄。

## 1. Motivation 和 Problem Statement

**Vanilla 3DGS** (Kerbl et al., SIGGRAPH 2023, https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) 用百万级 unstructured Gaussians 表示场景,渲染速度极快 (real-time via differentiable rasterization),但是这些 Gaussians 之间没有 topology 关联,artist 没法 rig/animate/sculpt。

**SuGaR** (Guédon & Lepetit, CVPR 2024, https://anttwo.github.io/suGaR/) 通过 regularization term 把 Gaussians 强制 align 到真实 surface 上,然后 Poisson reconstruction 出 mesh,Gaussians 被 flatten 并 pin 到 mesh surface 上。这样可 edit 了,但牺牲了 fuzzy material 的渲染质量 — 因为 hair/fur 这种东西本质上是 volumetric 的,flatten 到 surface 上就丢失了 occlusion 和 depth-varying appearance。

**Adaptive Shells** (Wang et al., SIGGRAPH 2023, https://research.nvidia.com/labs/toronto-ai/adaptive_shells/) 用 SDF + 两个 explicit meshes (inner/outer bound) bounding 一个 thin volumetric layer,渲染质量好,但依赖 NeuS-style SDF optimization,一个 synthetic scene 要 8 小时,而且 SDF 对 sharp geometry 重建有局限。

Frosting 想要同时拿到: **3DGS 的速度 + Adaptive Shells 的 hybrid volume-surface 质量 + mesh 的 editability**,而且只依赖一个 mesh (不像 Adaptive Shells 依赖两个 mesh)。

## 2. Method Overview — 双向过程

Paper 的核心设计是一个 **forward then backward** 的 pipeline:

### Forward Process (Volume → Surface)

1. 先做 unconstrained 3DGS optimization 7000 iterations,让 Gaussians 自由 position (这些叫 **unconstrained Gaussians**,保存备用)
2. 然后加 SuGaR 的 surface alignment regularization,继续 optimize 到 15000 iterations (这些叫 **regularized Gaussians**)
3. 从 regularized Gaussians 用 Poisson reconstruction 提取 base mesh M

**Key intuition**: regularized Gaussians 在 flat surface 上 align 得很好,但在 fuzzy materials 上 align 失败 — 这种 misalignment 正是 fuzzy 区域的信号!Paper 把这个"失败"当作 feature 用,而不是 bug。

### Backward Process (Surface → Volume)

给定 base mesh M,在每个 vertex $v_i$ 上沿 normal $n_i$ 方向定义两个 shift $\delta_i^{in}$ 和 $\delta_i^{out}$,构成一个 thickness 可变的 layer。然后在 layer 里 sample 新的 Gaussians 并 optimize。

## 3. Frosting Layer 厚度估计 — 核心技术贡献

这是 paper 最 tricky 的部分。直觉上你想用 unconstrained Gaussians 的 isosurface 来定义 thickness,但问题是 unconstrained Gaussians 含大量 floaters 和 outliers,而且即使在 flat surface 上也会形成不必要的厚 layer。

直接用 regularized Gaussians 又不行,因为 regularization 把它们压扁了,会 miss 掉 fuzzy 区域。

**Paper 的解决方案 — 两阶段搜索**:

#### Step 1: 定义 confidence interval $I_i$

对每个 vertex $v_i$,找最近的 regularized Gaussian,取它在 normal 方向上的标准差 $\sigma_i$:

$$I_i = [-3\sigma_i, 3\sigma_i]$$

这是 1D Gaussian 的 99.7% confidence interval。Fuzzy 区域 $\sigma_i$ 大,$I_i$ 也大。

#### Step 2: 在 $I_i$ 内搜索 regularized Gaussians 的 isosurface

定义集合 $T$:

$$T = \{t \in I_i \mid d_r(v_i + t n_i) \geq \lambda\}$$

变量含义:
- $d_r$: regularized Gaussians 的 density function (Eq. 1 形式,即 $\sum_g \alpha_g \exp(-\frac{1}{2}(p-\mu_g)^T \Sigma_g^{-1}(p-\mu_g))$)
- $v_i + t n_i$: 沿 normal 方向偏移 $t$ 的点
- $\lambda = 0.01$: isosurface level,接近零

然后取:
$$\epsilon_i^{in} = \inf(T), \quad \epsilon_i^{out} = \sup(T)$$

并定义扩展 interval:
$$J_i = [\epsilon_i^{mid} - k \epsilon_i^{half}, \epsilon_i^{mid} + k \epsilon_i^{half}]$$

其中 $\epsilon_i^{mid} = (\epsilon^{in} + \epsilon^{out})/2$, $\epsilon_i^{half} = (\epsilon^{out} - \epsilon^{in})/2$, $k=3$。

**Intuition**: $J_i$ 是 regularized Gaussians "承认存在"的区域,但用 $k=3$ 扩展,既包含大部分 unconstrained Gaussians 又 reject outliers。

#### Step 3: 在 $J_i$ 内搜索 unconstrained Gaussians 的 isosurface

$$V = \{t \in J_i \mid d_u(v_i + t n_i) \geq \lambda\}$$

$$\delta_i^{in} = \inf(V), \quad \delta_i^{out} = \sup(V)$$

这里 $d_u$ 是 unconstrained Gaussians 的 density。最终 thickness 由 unconstrained Gaussians 决定,但搜索范围被 regularized Gaussians 限制住了 — 这是一个非常聪明的 "regularized as prior, unconstrained as evidence" 的设计。

## 4. Gaussian Parameterization — Prismatic Cells 和 Barycentric Coordinates

这是让 Frosting **editable** 的关键。对每个 mesh triangle (vertices $v_0, v_1, v_2$ with normals $n_0, n_1, n_2$),six vertices 形成一个 **prismatic cell** (不规则三棱柱):

- 三个 outer vertices: $(v_i + \delta_i^{out} n_i)_{i=0,1,2}$
- 三个 inner vertices: $(v_i + \delta_i^{in} n_i)_{i=0,1,2}$

每个 Gaussian $g$ 的 mean $\mu_g$ 用六组 barycentric coordinates 参数化:

$$\mu_g = \sum_{i=0}^{2} \left( b_g^{(i)} (v_i + \delta_i^{out} n_i) + \beta_g^{(i)} (v_i + \delta_i^{in} n_i) \right)$$

约束:
$$\sum_{i=0}^{2} (b_g^{(i)} + \beta_g^{(i)}) = 1$$

变量含义:
- $b_g^{(i)}, \beta_g^{(i)} \geq 0$: Gaussian $g$ 相对于 outer/inner vertices 的 barycentric weights
- softmax activation 保证 non-negative 且 sum to 1

**为什么这个参数化重要**: 
1. **Hard constraint**: Gaussian 数学上不可能跑出 prismatic cell,optimization 时不需要额外 penalty
2. **Edit propagation**: 当 mesh deformation 时,$\mu_g$ 自动跟着变,因为它是 vertices 的线性组合
3. **Deterministic count**: 用户指定 budget N (5M real / 2M synthetic),不像 vanilla 3DGS 的 adaptive density control 是 emergent 的

## 5. Animation 时的 Gaussian 参数自动调整

当 mesh 变形时,光调整 $\mu_g$ 不够 — Gaussian 的 rotation $q_g$ 和 scaling $s_g$ 也得跟着变,否则会出现 stretching artifacts。

Paper 的方案 (Section 9 of supplementary): 对每个 prismatic cell,先算 cell center $c$ 和 6 个 vertices $v_i$。对每个 vertex,计算向量 $(c - v_i)$ 从原状态到变形状态的 rotation (axis-angle 表示,cross product 给 axis) 和 rescaling (沿 $(c-v_i)$ 方向 scale,其他 axis 不变)。

然后对 Gaussian $g$,用它的 barycentric coordinates 把 6 个 vertex 的 transformation 加权平均,apply 到 Gaussian 的三个主轴上,最后 orthonormalize。

**Intuition**: 这其实是一个 piecewise linear 的 deformation field — 每个 cell 内部用 vertex transformations 的 barycentric interpolation。Paper 自己承认这是 limitation,说可以换成 physics-based deformation model。这让我想到 Linear Blend Skinning (LBS) 在 character animation 里的角色 — 简单但 work,advanced 版本可以用 Dual Quaternions 或 physics simulation。

## 6. Octree Depth 自动选择 — 一个小但关键的 trick

SuGaR 默认用 Poisson reconstruction 的 octree depth $D=10$,但对简单场景这会 over-resolve,导致 mesh 上出现 ellipsoidal bumps (Gaussian 形状 leakage) 和 holes。

Paper 提出基于 **Gaussian 间最近邻距离** 的 complexity score:

$$CS = Q_{0.1}\left(\left\{ \min_{g' \neq g} \frac{\|\mu_g - \mu_{g'}\|_2}{L} \right\}_{g \in \mathcal{G}}\right)$$

变量:
- $\mathcal{G}$: 所有 Gaussians 集合
- $L$: point cloud bounding box 最长边
- $Q_{0.1}$: 0.1-quantile (用 quantile 而非 mean 是为了 robust to outliers,同时 0.1 比 min 更稳定)

然后:
$$\bar{D} = \lceil -\log_2(\gamma \times CS) \rceil$$

其中 $\gamma = 100$ 是 scene-independent hyperparameter。

**Intuition**: octree cell size 是 $2^{-D}$ (normalized),我们希望 cell size 略大于最近邻 Gaussian 间距 — 这样 Poisson reconstruction 既不会 over-resolve (出现 bumps) 也不会 under-resolve (丢失 detail)。用 quantile 而非 mean 是因为只有"细节 Gaussians"(互相靠近的)才真正 encode geometry,大量 sparse background Gaussians 会污染 mean。

Table 4 的 ablation 显示: NeRFSynthetic 上 PSNR 从 31.63 → 33.03 (+1.4 dB),triangle count 从 >1M → 863K;Shelly 上 triangle count 从 939K → 203K (4.6x reduction) 而 PSNR 基本持平。这是一个 free lunch。

## 7. Sampling Strategy — 体积与纹理的平衡

给定 budget N,采样策略:
- N/2 Gaussians: **uniform** over prismatic cells (保证 flat 区域有足够 Gaussians 恢复 texture)
- N/2 Gaussians: **proportional to cell volume** (保证 fuzzy 区域有足够 Gaussians 做 volumetric rendering)

对于 unbounded scenes (Mip-NeRF 360),distant cells 体积大但贡献小,paper 用 Mip-NeRF 360 风格的 contraction:

$$f(x) = \begin{cases} x & \text{if } \|x - c\| \leq l \\ c + l \times (2 - \frac{l}{\|x-c\|}) \frac{x-c}{\|x-c\|} & \text{if } \|x - c\| > l \end{cases}$$

变量:
- $c$: camera positions bounding box 中心
- $l$: bounding box 对角线一半

这是把远处空间"压缩"到单位球附近的经典手法,和 Mip-NeRF 360 的 contraction 一样 (https://jonbarron.info/mipnerf360/)。

## 8. Self-intersection 避免策略

Prismatic cells 之间可能 intersect (特别是 concave 区域),导致 animation 时 Gaussians 不跟随正确的 cell 移动。

Paper 的简单方案: 从 $\delta = 0$ 开始,逐步增大,一旦检测到某 vertex 的 cell 与其他 cell 相交,就 freeze 这个 vertex 的 shift。这是一个 greedy 的 constraint satisfaction,虽然 suboptimal 但 effective。

## 9. 实验结果分析

### Table 1 — Synthetic scenes

**Shelly dataset** (fuzzy materials 专门 benchmark):
| Method | PSNR | SSIM | LPIPS |
|--------|------|------|-------|
| 3DGS | 37.66 | 0.958 | 0.066 |
| Adaptive Shells | 36.02 | 0.954 | 0.079 |
| SuGaR | 36.33 | 0.954 | 0.059 |
| **Frosting** | **39.84** | **0.977** | **0.033** |

Frosting 比 unconstrained 3DGS 高 **+2.18 dB**!这是非常反直觉的 — 通常约束会降低渲染质量。Paper 的解释:Frosting 的 densification 比 vanilla 3DGS 的 adaptive density control 更高效,因为它 targeted 在 fuzzy areas 而非全局。3DGS 的 densification 是 gradient-based 的,会在很多不必要的区域也 create Gaussians。

**NeRFSynthetic**:
| Method | PSNR |
|--------|------|
| 3DGS | 33.32 |
| SuGaR | 32.40 |
| Frosting | 33.03 |

这里 Frosting 略低于 3DGS (0.29 dB),因为这些场景大多是无 fuzzy material 的 hard-surface objects,Frosting 的优势发挥不出来。

### Table 2 — Mip-NeRF 360 (real, unbounded)

| Method | Avg PSNR |
|--------|----------|
| 3DGS | 28.69 |
| Mip-NeRF 360 | 29.09 |
| SuGaR | 27.27 |
| Adaptive Shells | 26.61 |
| Frosting | 28.38 |

Frosting 是 editable methods 里最好的,但比 3DGS 低 0.31 dB。Paper 解释:当 3DGS 有很好的 SfM initialization (大量 points) 时,Frosting densification 的优势减弱。这暗示在弱 initialization 场景下 Frosting 可能反超。

### Table 5 — Thickness ablation

| Strategy | Shelly | Mip-NeRF 360 Avg |
|----------|--------|------------------|
| Constant (small) | 39.03 | 28.28 |
| Constant (medium) | 39.67 | 28.20 |
| Constant (large) | 40.00 | 28.10 |
| Regularized only | 39.34 | 28.34 |
| **Full method** | **39.84** | **28.38** |

有趣的是 constant large thickness 在 Shelly 上反而最高 (40.00 > 39.84),但在 Mip-NeRF 360 上最低 — 因为 real scenes 有 flat surfaces,厚 layer 引入 artifacts。这正证明了 **adaptive thickness 的必要性**:不能 one-size-fits-all。

而且 regularized-only 比 full method 低 0.5 dB,证明用 unconstrained Gaussians refine 是必要的。

## 10. 和你 (Karpathy) 工作的潜在联系

Andrej, 几个我觉得你可能感兴趣的点:

1. **Densification as learned allocation**: Frosting 的 densification 本质上是一个 **adaptive computation allocation** 问题 — 在哪里 spend Gaussian budget。这和你在 nanoGPT (https://github.com/karpathy/nanoGPT) 里讨论的 MoE / sparse attention 的思想有结构性相似:都是 "where to allocate capacity"。

2. **Barycentric parameterization as differentiable binding**: 这个 prismatic cell + barycentric coords 的设计,本质上是把 Gaussians "绑定" 到 mesh 的 local frame 上。这和 NeRF 里 position encoding 的作用类似 — 都是 inject structure into otherwise free optimization。

3. **Piecewise linear deformation field**: Paper 的 animation 参数调整其实就是 LBS 的 3D analogue。你之前在 CS231n 讲过 LBS 的 differentiable 版本 (https://cs231n.github.io/),这里的 vertex transformation averaging 是一个 simplified 版本。

4. **Complexity score 用 quantile 而非 mean**: 这是一个 robust estimation 的好例子,让我想到你在 "Software 2.0" (https://karpathy.medium.com/software-2-0-a64152b37c35) 里提到的 — engineering heuristics 在 ML pipeline 里仍然关键。

5. **Contraction for unbounded scenes**: Mip-NeRF 360 的 contraction $f(x)$ 其实和 NeRF 的 positional encoding 在哲学上是对偶的 — 一个 compress far field,一个 expand near field。

## 11. Limitations 和 Future Directions

Paper 自己提到:
- **Piecewise linear deformation** — 升级到 physics-based (FEM, Mass-spring) 会更好
- **Model size** 比 vanilla 3DGS 大 (要存 barycentric coords + mesh vertices) — 可以用 3DGS compression 工作 (https://ynamitaras.github.io/Compressed3DGS/) 缓解

我觉得还可以延伸的方向:
- **Differentiable mesh editing**: 现在 mesh edit 是 exogenous (Blender),如果能 end-to-end learn mesh deformation for task (比如 inverse rendering for pose estimation)
- **Frosting for dynamic scenes**: 现在 static scene,如果结合 4D Gaussian Splatting (https://github.com/hustvl/4DGaussians) 做 dynamic Frosting
- **Multi-layer frosting**: 对 translucent materials (skin, wax),单层 frosting 可能不够,multi-layer 可以 model subsurface scattering
- **Neural frosting thickness**: 现在 thickness 是 deterministic estimated,可以用一个小 MLP 学习 thickness field,让它 condition on viewing direction 或 lighting
- **Frosting + diffusion priors**: 用 stable diffusion 作为 rendering loss 的 regularizer (类似 DreamFusion, https://dreamfusion3d.github.io/),可能帮助 sparse-view 场景

## 12. Code 和资源

- Paper project page: https://anttwo.github.io/frosting/
- SuGaR (前作): https://anttwo.github.io/suGaR/
- 3DGS original: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- Adaptive Shells: https://research.nvidia.com/labs/toronto-ai/adaptive_shells/
- Poisson Surface Reconstruction: https://hhoppe.com/proj/poissonrecon/
- Mip-NeRF 360 contraction: https://jonbarron.info/mipnerf360/
- COLMAP: https://colmap.github.io/
- PyTorch: https://pytorch.org/

---

**总结一下 intuition**: Frosting 的优雅之处在于它把 SuGaR 的 "failure" (fuzzy 区域 alignment 失败) 转化为 "signal" (thickness 估计),然后用一个简单但 mathematically sound 的 parameterization (barycentric in prismatic cells) 把 volumetric freedom 重新注入到一个 editable structure 里。这其实是一个 **bottleneck design** — mesh 是 information bottleneck,Frosting layer 是围绕 bottleneck 的 residual capacity,thickness 决定 residual 的 budget。

这和 deep learning 里很多 design pattern 呼应: ResNet 的 residual connections (https://arxiv.org/abs/1512.03385),Transformer 的 FFN capacity,U-Net 的 skip connections — 都是在 structured backbone 上加 adaptive capacity。Frosting 在 3D representation 上做了类似的事。
