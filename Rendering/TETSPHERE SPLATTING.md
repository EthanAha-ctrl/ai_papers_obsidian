---
source_pdf: TETSPHERE SPLATTING.pdf
paper_sha256: f9a1657b722bb2a208634977e7d34d77a16db3700f6ad9a2064a3f4cf368a9f9
processed_at: '2026-08-12T13:40:50-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej, 既然你需要“用人话说说”，那我们就抛开那些学术包装，直接切入这套系统的物理直觉和工程逻辑。同时按照你的要求，我会把核心的数学公式、变量定义和实验数据嵌在其中，帮你构建底层的 intuition。

### 1. 核心痛点：为什么现有 3D 生成的东西“没法用”？

现在的 3D 重建或者生成模型（比如 NeRF, 3D Gaussian Splatting, FlexiCubes），如果只看渲染出来的图片，效果非常惊艳。但凡你想把生成的 3D 资产导出成 obj 格式，放进 Blender、Unity 或者物理仿真引擎里，就会立刻崩溃。
提取出来的 mesh 往往惨不忍睹：三角形要么极长极扁，要么出现不该有的翻转，或者表面飘着一堆没用的“灰尘”碎片。这种 mesh 在传统图形学里属于“废品”，根本没法用来做有限元分析（FEM）或者高质量的 texture mapping。

TetSphere Splatting 就是为了解决这个问题：**在生成或者重建 3D 形状的时候，顺便把 mesh 质量给保住了。**

### 2. 核心直觉：用“内部有支架的橡皮泥球”捏形状

想象一下我们是怎么捏泥人的。
*   **Eulerian 方法（比如 NeRF, FlexiCubes）**：像在一个固定的 3D 网格盒子里填沙子。你想刻画细节，就必须把盒子切得极度细。这极度耗费内存，且最后从沙堆里提取表面的过程必然产生不规则的三角形。
*   **传统 Lagrangian 方法（比如 3DGS, DMesh）**：像撒一把毫无关联的弹珠或者随便扔几张纸片在空中。它们虽然轻便，但在空间中随意乱飘，缺乏整体的结构约束，很容易乱七八糟。
*   **TetSphere Splatting 的做法**：它给你发了一盒“高级橡皮泥球”。每个球内部不是实心的，而是用“四面体”搭好了内部支架。你只需要拉扯这些球的外皮，内部的支架会跟着一起拉伸。因为有内部支架的约束，你怎么捏，外表都不会变得坑坑洼洼，更不会发生穿透或者翻转。把很多个这样捏好的球拼在一起，就成了最终的 3D 形状。

### 3. 技术拆解：这套“支架”是怎么通过数学起作用的？

这篇 paper 的精髓全在它的 optimization objective 里面。为了让这些橡皮泥球在变形时保持高质量，他们定义了如下的能量函数：

$$
\underset{\mathbf{x}}{\operatorname{min}} ~ \Phi(R(\mathbf{x})) + w_1 \vert\vert \mathbf{L}\mathbf{F}_{\mathbf{x}} \vert\vert_2^2 + w_2 \sum_{i,j} (\operatorname*{min}\{0, \det(\mathbf{F}_{\mathbf{x}}^{(i,j)})\})^2
$$

我们来逐个拆解这些变量和符号，看看它们在直觉上对应什么：

**变量解释：**
*   $\mathbf{x} \in \mathbb{R}^{3NM}$：这是唯一的优化变量。$\mathbf{x}$ 是所有橡皮泥球顶点的空间坐标。$M$ 是球的数量，$N$ 是单个球内部的顶点数，$3$ 代表三维空间的 xyz。
*   $\mathbf{F}_{\mathbf{x}}^{(i,j)} \in \mathbb{R}^{3 \times 3}$：第 $i$ 个球里第 $j$ 个四面体支架的 deformation gradient。计算方式是当前形状的边矩阵乘以初始形状边矩阵的逆。直觉上，它记录了这块小支架被怎么拉扯和旋转了。
*   $\mathbf{L} \in \mathbb{R}^{9MT \times 9MT}$：Laplacian matrix。$T$ 是单个球内的四面体个数，$9$ 来自 $3 \times 3$ 矩阵的展平。它用来比较一个支架和它周围相邻支架的差异。

**三项能量的直觉：**
1.  **$\Phi(R(\mathbf{x}))$ - 渲染损失**：$R(\cdot)$ 是可微渲染器，把当前的球渲染成图片，$\Phi(\cdot)$ 计算和目标图片的差异（比如 RGB 的 $l_1$ loss，或者 depth 的 MSE）。它负责告诉你：“往左捏一点，图片才像”。
2.  **$w_1 \vert\vert \mathbf{L}\mathbf{F}_{\mathbf{x}} \vert\vert_2^2$ - 双谐波能量**：这是这篇 paper 的神来之笔。注意，它优化的对象不是顶点坐标 $\mathbf{x}$，是变形梯度 $\mathbf{F}_{\mathbf{x}}$。如果是优化坐标，那整个球会被抹平，丢失细节。优化变形梯度意味着：**它允许整个球体做巨大的刚性旋转或者均匀缩放（此时 $\mathbf{L}\mathbf{F} \approx 0$），但严厉惩罚局部相邻支架之间的变形不连贯。** 这样就能在保留锐利边缘的同时，消除表面那种坑坑洼洼的噪声。
3.  **$w_2 \sum (\operatorname*{min}\{0, \det(\mathbf{F}_{\mathbf{x}}^{(i,j)})\})^2$ - 防翻转约束**：$\det(\mathbf{F}_{\mathbf{x}}^{(i,j)})$ 是矩阵的行列式。如果行列式小于 0，说明这个四面体被捏得“内翻外”了，就像把手套的指尖从里面戳出来一样。这会导致法线反向，渲染崩坏。这里通过一个 soft penalty 惩罚所有小于 0 的情况，强行把顶点推回正向。

### 4. Pipeline 的细节：怎么放这些球？

你不能随便把球扔在空间里。Paper 里设计了一个 Silhouette Coverage 算法。
1.  先建一个 $300 \times 300 \times 300$ 的粗略 voxel grid。
2.  把 voxel 投影到多视角图片里，如果在所有视角里都在前景，就标记为候选点。
3.  用 mixed-integer linear programming 求解一个集合覆盖问题：
    $$ \operatorname*{min}_{\mathbf{v}} \vert\mathbf{v}\vert \quad \mathrm{s.t.} \quad \mathbf{D}\mathbf{v} \geq \mathbb{1} $$
    $\mathbf{D}$ 是覆盖矩阵，如果 voxel $j$ 能被放在候选点 $i$ 的球覆盖，就是 1。$\mathbf{v}$ 是二值选择向量。直觉上，就是用最少的球，把所有候选点包住。这个步骤通常选出 20 个左右的球，1 分钟就能跑完。

### 5. 实验数据：到底好在哪里？

让我们看看 Table 1 里多视角重建的数据，重点看 mesh 质量指标：

| Method | Geo. Rep. | Cham. ↓ | Vol. IoU ↑ | ALR ↑ | MR(%) ↑ | CC Diff. ↓ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FlexiCubes | Eulerian | 0.0247 | 0.5887 | 0.0722 | 45.5 | 201.3 |
| DMesh | Lagrangian| 0.0136 | 0.5616 | 0.1193 | 9.09 | 3.75 |
| **Ours** | **Lagrangian**| 0.0184 | 0.6844 | **0.6602** | **100** | **0.0** |

*   **Reconstruction Accuracy (Cham. & Vol. IoU)**：TetSphere 的 Chamfer Distance (0.0184) 处于第一梯队，Volume IoU (0.6844) 是全场最高。这说明形状捏得很准。
*   **ALR (Area-Length Ratio)**：衡量三角形是不是接近等边。越高越好。TetSphere 达到了 0.6602，而 DMesh 只有 0.1193。**直觉上，TetSphere 生成的三角形极其规整，完美适合下游的 FEM 仿真。**
*   **MR (Manifoldness Rate)**：TetSphere 是 100%。生成的所有 mesh 全是闭合流形。DMesh 只有可怜的 9.09%，没法用。
*   **CC Diff (Connected Component Discrepancy)**：TetSphere 是 0.0。完全没有漂浮的碎片。

### 6. 更进一步的联想与未来方向

这篇 paper 展示了一种将经典几何处理和现代可微渲染结合的范式。你可以联想到：
*   **与单球变形的对比**：以前 Nicolet et al. 2021 也想过用一个球去捏，但拓扑一复杂，单球必须剧烈变形，就会发生 self-intersection，只能靠中途不断 remesh 来救。TetSphere 用多个球拼起来，每个球只做轻度变形，完美避开了 remeshing 带来的 texture parameterization 失效问题。
*   **局限性与拓扑保证**：Paper 在 Limitations 里很坦诚，多个球的 Union 操作在理论上并不能绝对保证整体 manifold。两个球如果在交界处没对齐，内部依然可能存在 interface。未来的工作可能需要引入 level-set fusion 或者 boolean union 的可微版本，来在球与球之间做真正的拓扑融合。
*   **Thin Shell Problem**：对于极薄的结构（比如树叶），体积表示的四面体容易塌缩，导致 $\det(\mathbf{F}) \to 0$ 引发数值不稳定。这种情况下，DMesh 这种纯表面表示反而更灵活。

### 相关 Web Links 供参考

如果你想深入跑一下或者对比相关工作，这里有些有用的链接：
*   **TetSphere Splatting 官方代码库:** [https://github.com/gmh14/tssplat](https://github.com/gmh14/tssplat)
*   **DMesh (对比组):** [https://arxiv.org/abs/2404.13445](https://arxiv.org/abs/2404.13445)
*   **FlexiCubes (对比组):** [https://research.nvidia.com/labs/toronto-ai/flexicubes/](https://research.nvidia.com/labs/toronto-ai/flexicubes/)
*   **2DGS (对比组):** [https://surh.github.io/2d-gaussian-splatting/](https://surh.github.io/2d-gaussian-splatting/)
*   **Coverage Axis (初始化灵感来源):** [https://onlinelibrary.wiley.com/doi/10.1111/cgf.14570](https://onlinelibrary.wiley.com/doi/10.1111/cgf.14570)
*   **Locally Injective Mappings (防翻转约束理论基础):** [https://igl.ethz.ch/projects/LIM/](https://igl.ethz.ch/projects/LIM/)
*   **nvdiffrast (底层可微渲染器):** [https://nvlabs.github.io/nvdiffrast/](https://nvlabs.github.io/nvdiffrast/)

---

Andrej, 这是一篇非常有意思的 paper, 它的核心贡献在于提出了一种全新的 Lagrangian 几何表示方法: TetSphere Splatting。在当下 3D Gaussian Splatting 和 Neural Implicit representations (如 NeRF, SDF) 占据主导的生态中, 这篇文章回归了计算机图形学中最经典的 FEM (Finite Element Method) 和 volumetric mesh 理念, 将其与 differentiable rendering 和 score distillation 无缝结合, 旨在解决一个在 3D 生成与重建中极其痛点的问题: **mesh quality**。

现在很多 3D 生成工作能生成视觉上很好看的结果, 但提取出来的 mesh 质量极差, 充满 degenerated triangles, non-manifold edges 和 floating artifacts, 根本无法直接用于物理仿真或传统的 CG pipeline。TetSphere Splatting 通过引入由 tetrahedra 组成的 volumetric sphere 作为 primitive, 在保证计算效率的同时, 从根本上约束了 mesh 的 topological 和 geometric quality。

下面我为你进行深度的技术拆解, 试图 build your intuition。

### 1. 核心直觉: Eulerian vs Lagrangian vs TetSphere

要理解这篇 paper, 我们先看 representation 的 spectrum:
*   **Eulerian representations** (NeRF, NeuS, VolSDF, FlexiCubes, DMTet): 场是固定在空间网格上的。你通过 querying 一个神经网络或一个 grid 来获取 density 或 SDF。缺点是计算昂贵, 且提取 mesh 需要 Marching Cubes 或 Marching Tetrahedra, 这一步往往会破坏 mesh 质量, 产生 irregular triangles。
*   **Lagrangian representations** (3DGS, DMesh): 质点在空间中自由移动。计算很快, 但缺乏全局结构。3DGS 产生的是点云, 常有噪声; DMesh 产生的是面片, 常出现 non-manifold 或 self-intersections。
*   **TetSphere Splatting**: 引入了一个中间态的 primitive。每个 primitive 是一个由 tetrahedra 填充的 volumetric sphere。它属于 Lagrangian, 因为这些 spheres 在空间中移动和变形; 但它又具有 Eulerian 的局部结构特性, 因为每个 sphere 内部的 vertices 被 tetrahedral connectivity 严格绑定。这种 local connectivity 使得施加 FEM 级别的 geometric regularization 成为可能, 从而保证了 mesh 质量。

### 2. 数学公式与能量函数解析

TetSphere splatting 的核心是将 shape reconstruction 变成一个 deformation optimization 问题。假设我们有 $M$ 个 TetSpheres, 每个 TetSphere 有 $N$ 个 vertices 和 $T$ 个 tetrahedra。所有顶点的位置记为 $\mathbf{x} \in \mathbb{R}^{3NM}$。

优化的 objective function 包含三个部分:

$$
\underset{\mathbf{x}}{\operatorname{min}} ~ \Phi(R(\mathbf{x})) + w_1 \vert\vert \mathbf{L}\mathbf{F}_{\mathbf{x}} \vert\vert_2^2 + w_2 \sum_{i,j} (\operatorname*{min}\{0, \det(\mathbf{F}_{\mathbf{x}}^{(i,j)})\})^2
$$

**变量与符号解释:**
*   $\mathbf{x}$: 所有 TetSphere 顶点的位置向量, 维度为 $3NM$。这是优化变量。
*   $R(\cdot)$: Differentiable rendering function (如基于 nvdiffrast 的 mesh rasterizer)。
*   $\Phi(\cdot)$: Rendering loss, 可以是 RGB 的 $l_1$ loss, depth 的 MSE loss, 或 normal 的 cosine embedding loss。
*   $\mathbf{F}_{\mathbf{x}}^{(i,j)} \in \mathbb{R}^{3 \times 3}$: 第 $i$ 个 TetSphere 中第 $j$ 个 tetrahedron 的 **deformation gradient**。它由公式 $\mathbf{F} = \mathbf{D}_s \mathbf{D}_m^{-1}$ 计算, 其中 $\mathbf{D}_s$ 是 deformed tetrahedron 的 edge matrix, $\mathbf{D}_m$ 是 rest-pose tetrahedron 的 edge matrix。它描述了局部的旋转与拉伸。
*   $\mathbf{L} \in \mathbb{R}^{9MT \times 9MT}$: Laplacian matrix。它作用于 deformation gradient field 上。如果两个 tetrahedra 共享一个 face, 则在对应的 $9 \times 9$ block 上设置为 $-\mathbf{I}$; 对角线 block $\mathbf{L}_{pp}$ 设置为 $k\mathbf{I}$, $k$ 是该 tetrahedron 的邻居数量。
*   $w_1, w_2$: 动态调整的权重, 使用 cosine scheduler。
*   $\det(\mathbf{F}_{\mathbf{x}}^{(i,j)})$: Deformation gradient 的行列式。

**深入直觉: 为什么是 Bi-harmonic Energy on Deformation Gradient?**
这是这篇 paper 最关键的设计。传统的 mesh smoothing 往往是 minimizing Laplacian energy on vertex positions, 即 $||\mathbf{L}\mathbf{x}||^2$, 这会导致 mesh 严重收缩和 over-smoothing, 丢失 sharp features。
本文 minimizing 的是 $||\mathbf{L}\mathbf{F}_{\mathbf{x}}||_2^2$, 即对 **deformation gradient** 求 bi-harmonic energy。直觉上, 这是在惩罚 *变形的不平滑性*。如果一个顶点移动了, 相邻的顶点也需要以相似的 *方式* (旋转/缩放) 移动。它允许整个球体做大范围的 rigid transformation 或 uniform scaling (此时 $\mathbf{L}\mathbf{F} \approx 0$, 梯度为零), 同时严厉惩罚局部的剧烈扭曲。这保证了 mesh 可以自由贴合 target shape 的同时, 内部结构依然光滑, 不会出现 bumpy surface。

**深入直觉: Local Injectivity Constraint**
$\det(\mathbf{F}) > 0$ 是 FEM 中的经典约束, 保证 tetrahedron 不会发生翻转 (inverted)。如果发生翻转, mesh 的 outward normal 方向会反转, 导致 rendering 出现灾难性的 artifacts。公式中通过 $(\operatorname*{min}\{0, \det(\mathbf{F})\})^2$ 作为一个 soft penalty, 只有当 $\det(\mathbf{F}) < 0$ 时才会产生梯度把顶点推回正确方向。

### 3. 架构与 Pipeline 解析

Paper 的整体 pipeline 如 Figure 3 所示:
1.  **Initialization (Silhouette Coverage):** 给定 multi-view images, 先构建一个 coarse voxel grid (分辨率 $300 \times 300 \times 300$)。投影这些 voxels 到图像得到 foreground mask。然后使用类似 Coverage Axis (Dou et al., 2022) 的思想, 将初始的 TetSphere 中心放置在这些 candidate voxels 上。
    这是一个 Set Cover 问题, paper 使用了 mixed-integer linear programming 求解:
    $$
    \operatorname*{min}_{\mathbf{v}} \vert\mathbf{v}\vert \quad \mathrm{s.t.} \quad \mathbf{D}\mathbf{v} \geq \mathbb{1}
    $$
    其中 $\mathbf{D}$ 是 coverage matrix (voxel $j$ 是否被 sphere $i$ 覆盖), $\mathbf{v}$ 是 binary selection vector。通常选 $M=20$ 个 spheres, 求解时间约 1 分钟。
2.  **Deformation Optimization:** 使用上述的 objective function, 通过 gradient descent 优化顶点位置 $\mathbf{x}$。CUDA 实现的 PyTorch extension 用于加速 geometric energy 的计算。
3.  **Texture & PBR Material (Optional):** 由于 TetSphere 的 topology 在 deformation 过程中保持不变, 不像 DMTet 每步需要提取 isosurface 导致 UV parameterization 失效, TetSphere 可以在初始化时做一次 texture parameterization, 然后直接优化一个 $2048 \times 2048$ 的 texture map 或使用 MLP 输出 Disney BRDF 参数。

### 4. 实验数据与 SOTA 对比分析

Paper 在 multi-view 和 single-view reconstruction 上与多种 SOTA 方法进行了对比。我们来看看 Table 1 的数据:

| Method | Geo. Rep. | Cham. ↓ | Vol. IoU ↑ | ALR ↑ | MR(%) ↑ | CC Diff. ↓ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FlexiCubes | Eulerian | 0.0247 | 0.5887 | 0.0722 | 45.5 | 201.3 |
| NeuS | Eulerian | 0.0192 | 0.6182 | 0.0573 | 72.3 | 8.1 |
| 2DGS | Lagrangian | 0.0322 | 0.4923 | 0.0209 | 27.3 | 25.1 |
| DMesh | Lagrangian | 0.0136 | 0.5616 | 0.1193 | 9.09 | 3.75 |
| **Ours** | **Lagrangian** | 0.0184 | 0.6844 | **0.6602** | **100** | **0.0** |

*   **Reconstruction Accuracy (Cham. & Vol. IoU):** TetSphere 在重建精度上具有竞争力。它的 Chamfer Distance (0.0184) 优于 FlexiCubes 和 2DGS, 稍逊于 NeuS 和 DMesh, 但 Vol. IoU (0.6844) 是最高的, 说明它对 volume 的占有率最好, 很好地捕捉了整体结构。
*   **Mesh Quality (ALR, MR, CC Diff):** 这是 TetSphere 统治级的表现。
    *   **ALR (Area-Length Ratio):** 衡量三角形是否接近等边。TetSphere 达到了 0.6602, 而其他方法最高只有 0.1193 (DMesh)。这说明 TetSphere 生成的 mesh 极其规整, 完美适合 FEM 仿真。
    *   **MR (Manifoldness Rate):** TetSphere 是 100%, 所有生成的 mesh 都是 closed manifold。相比之下, DMesh 只有 9.09%, 2DGS 只有 27.3%。
    *   **CC Diff (Connected Component Discrepancy):** TetSphere 是 0.0, 完全没有 floating artifacts。

Table 3 的计算成本测试也值得关注。在 Image-to-3D 的 SDS optimization 中, TetSphere 在 40GB A100 上可以处理 batch size 120, 速度达到 6.59 iter/s, 显著优于 Make-it-3D, Magic123 等 Eulerian 方法, 甚至优于同为 Lagrangian 的 DreamGaussian。这得益于 explicit representation 在 SDS 优化中不需要 querying dense grid 或庞大的 MLP。

### 5. 延伸联想与未来方向

*   **与传统的 Remeshing 对比:** 像 Nicolet et al. (2021) 和 Palfinger (2022) 也尝试过 deformable mesh, 但它们使用 single surface sphere。当 target shape topology 复杂时, single sphere 必须经历 extreme deformation, 导致 self-intersection 或 degeneracy, 必须依赖 intermediate remeshing。Remeshing 会破坏 texture parameterization。TetSphere 用 multiple volumetric spheres 绕过了这个问题, 每个 sphere 只需 moderate deformation, 无需 remeshing。这是一种用 *数量* 换取 *局部质量* 的经典工程哲学。
*   **局限性与 Topology Guarantee:** Paper 在 Limitations 中坦诚, TetSphere 的最终 mesh 是多个 spheres 的 union, 它在理论上不保证 global manifoldness, 也不保证 topology 与 ground truth 完全一致 (例如, 如果两个 spheres 没有完全融合, 可能会产生 internal interfaces)。未来的工作可能需要引入 boolean operations 或 level-set fusion 来在 TetSphere level 做真正的 topology merge。
*   **Thin Shell Problem:** 在 Table 5 之后的讨论中提到, 对于 plant leaves 这种 thin shell 结构, TetSphere 表现稍弱于 DMesh。直觉上, 体积表示在塌缩成极薄的面时, 容易触发 $\det(\mathbf{F}) \to 0$ 的边界, 导致数值不稳定, 此时 surface-based 方法 (如 DMesh) 反而更灵活。

### 6. 相关参考链接

为了方便你进一步深挖, 这里提供一些相关的 link:

*   **TetSphere Splatting GitHub:** [https://github.com/gmh14/tssplat](https://github.com/gmh14/tssplat)
*   **FlexiCubes (Baseline):** [https://research.nvidia.com/labs/toronto-ai/flexicubes/](https://research.nvidia.com/labs/toronto-ai/flexicubes/)
*   **DMesh (Baseline):** [https://arxiv.org/abs/2404.13445](https://arxiv.org/abs/2404.13445)
*   **2DGS (Baseline):** [https://surh.github.io/2d-gaussian-splatting/](https://surh.github.io/2d-gaussian-splatting/)
*   **Coverage Axis (Initialization inspiration):** [https://dl.acm.org/doi/10.1111/cgf.14570](https://dl.acm.org/doi/10.1111/cgf.14570)
*   **Locally Injective Mappings (Non-inversion constraint):** [https://igl.ethz.ch/projects/LIM/](https://igl.ethz.ch/projects/LIM/)
*   **nvdiffrast (Differentiable Rendering backend):** [https://nvlabs.github.io/nvdiffrast/](https://nvlabs.github.io/nvdiffrast/)
*   **Geometry Processing (Botsch & Sorkine):** [https://www.cse.wustl.edu/~taoju/cse554/lectures/lecture10_deformation.pdf](https://www.cse.wustl.edu/~taoju/cse554/lectures/lecture10_deformation.pdf)

总而言之, TetSphere Splatting 是一次非常精彩的 “Back to Basics” 的尝试。它告诉我们, 在追求神经网络的万能拟合能力的同时, 计算机图形学几十年来积累的 FEM, mesh processing 和 geometric energy minimization 中蕴含着极其强大的 inductive bias。将这些 classic primitives 重新包装为 differentiable, optimization-friendly 的 modules, 依然是通往 high-fidelity 3D world model 的一条极具潜力的路径。
