---
source_pdf: GSWT__Gaussian_Splatting_Wang_Tiles__Supplementary_Material_.pdf
paper_sha256: 17fb4001ae6199661f00bd4eaf1be861153d644b377228ae3a66565ef7ea178a
processed_at: '2026-08-04T23:15:10-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GSWT 用人话讲

好的 Andrej，我换个讲法，像在白板前聊那样讲。

---

## 先说痛点在哪

你玩过开放世界游戏吧？《荒野大镖客》《塞尔达》那种，地图巨大，镜头随便走，远处有山有树有草，近处一片叶子都看得清。

传统做法是用 **mesh + texture**。地形是一张高度图，贴上几张草地的 texture，远处的山用低模，近处的草用高模。这套方案工业界已经打磨了二十年，**GPU driver 层面都给它专门优化**。但它有个天花板：texture 是 2D 的，再怎么 trick 也只是"平面贴图假装有体积"。

3D Gaussian Splatting 出来之后，大家发现这东西天生适合做"有体积的细节"。草叶、碎石、毛皮这种 **海量小几何体** 的东西，用 Gaussian 比用 mesh 灵活得多，渲染也快（splatting 本质是 point sprite 的升级版，GPU 很喜欢）。

但 3DGS 原版有个致命假设：**整个场景一次性装进显存，每帧全局排序**。

你训练个房间场景，几百万个 Gaussian，一张 4090 装得下，排序也排得过来。但你想做一个 $10 \text{km} \times 10 \text{km}$ 的草场，每平方米一千根草，那就是 **十亿个 Gaussian**。显存炸了，排序也炸了。

更恶心的是，**3DGS 没有任何"复用"机制**。左边那片草和右边那片草长得差不多，但在 3DGS 眼里它们是**完全独立的一堆 Gaussian**。你扫了十平方公里草地，得到的 Gaussian 集合里 99% 是"长得差不多的草丛的拷贝"，但你必须为每一根草单独存参数、单独排序、单独渲染。

这就是痛点。**没有复用，没有 LOD，没有 streaming**。3DGS 原生版本是个"单人房间级"的技术。

---

## GSWT 的核心 trick：把场景变成拼图

作者发现一件事：**做无限地形，不需要存无限数据，只需要存一小块"拼图"，然后随机拼起来。**

这个想法在 2D texture 领域已经玩了二十年了，叫做 **Wang Tiles**。

Wang Tiles 的逻辑特别朴素：你做一组正方形小图块，每条边涂一个颜色。拼接规则就一条——**相邻两块共享的那条边，颜色必须一样**。

就这么简单一条规则，能产生什么效果？**你能拼出无限大的图案，但视觉上不会看出重复**。因为每个位置的"邻居组合"是随机的，人眼找不到周期性。

经典配置是用 4 种颜色、16 块小图块。就这 16 块，拼出整个喜马拉雅山脉那么大的草坡都没问题，你边走边随机抽一块拼上，远处看就是连绵不绝的草地，不会出现"咦这块我见过"的感觉。

GSWT 说：**既然 2D texture 能这么干，3D Gaussian 为什么不能？**

把每块"拼图"从一个 2D texture 换成一组 3D Gaussian。边上的"颜色"换成"边上的 Gaussian 分布"。拼接时，相邻两块的边必须 **Gaussian 几何对齐**，这样拼出来才不会有缝。

就这么个想法。**用 16 块 Gaussian tile 拼出整个无限世界**。

---

## 但光拼起来还不够，还有三个坑

### 坑一：拼图缝怎么处理

你拿两块拼图拼一起，边上的"颜色"虽然匹配了，但两块各自训练出来的时候，边上那条线附近会**各自有一堆 Gaussian**。两块一拼，这块的 Gaussian 和那块的 Gaussian **几何上重叠了**，渲染出来就是一条亮带或者重影。

怎么办？**运行时把边界附近重合的 Gaussian 合并掉**。这就是 paper 里的 **Selective Merging**。

两个 Gaussian $\mathcal{G}_1 = (\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1, \alpha_1, \mathbf{c}_1)$ 和 $\mathcal{G}_2 = (\boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2, \alpha_2, \mathbf{c}_2)$ 合并成一个，公式大概是这样的：

$$
\alpha_{\text{merge}} = \alpha_1 + \alpha_2 - \alpha_1 \alpha_2
$$

$$
\boldsymbol{\mu}_{\text{merge}} = \frac{\alpha_1 \boldsymbol{\mu}_1 + \alpha_2 \boldsymbol{\mu}_2}{\alpha_1 + \alpha_2}
$$

$$
\boldsymbol{\Sigma}_{\text{merge}} = \frac{\alpha_1(\boldsymbol{\Sigma}_1 + (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_{\text{merge}})(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_{\text{merge}})^\top) + \alpha_2(\boldsymbol{\Sigma}_2 + (\boldsymbol{\mu}_2 - \boldsymbol{\mu}_{\text{merge}})(\boldsymbol{\mu}_2 - \boldsymbol{\mu}_{\text{merge}})^\top)}{\alpha_1 + \alpha_2}
$$

这里：
- $\alpha_i$ 是第 $i$ 个 Gaussian 的不透明度，合并时按"概率论里两个独立事件的联合概率"算
- $\boldsymbol{\mu}_i$ 是中心位置，按不透明度加权平均
- $\boldsymbol{\Sigma}_i$ 是协方差矩阵（控制 Gaussian 的椭球形状），合并时除了加权平均原协方差，还要加上"中心点偏差"项 $(\boldsymbol{\mu}_i - \boldsymbol{\mu}_{\text{merge}})(\boldsymbol{\mu}_i - \boldsymbol{\mu}_{\text{merge}})^\top$，这相当于"如果两个高斯中心离得远，合并后的高斯要变胖一点才能覆盖原来的范围"

这个 merge 操作看起来花哨，但直觉很简单：**两个重叠的高斯混合分布，用一个更大的高斯去近似它**。这就是 [Gaussian Mixture Model 的 moment matching](https://en.wikipedia.org/wiki/Mixture_distribution#Moments) 那一套，统计学里玩烂的东西。

Table 3 的 Grass 数据告诉你这个 trick 多贵：

| 配置 | Sort time | Render time |
|---|---|---|
| Full (with merge) | 17.33 ms | 16.21 ms |
| w/o Merge | 2.60 ms | 16.11 ms |

**Sort 时间从 2.6 毫秒飙到 17.3 毫秒**，涨了 6 倍。但 render 时间几乎没变。

为什么 merge 会拖慢 sort？因为合并这步发生在排序阶段——你得先把所有 tile 排好，再扫描边界附近的 Gaussian 对，判断能不能合并。这是 $O(N)$ 的额外扫描，跑在 worker thread 上。

那 render 时间为什么没降？因为合并掉的 Gaussian 数量相对于总 splat count 来说不多。真正的性能瓶颈是"画多少个 splat"而不是"画之前扫了多少遍"。

但作者还是保留了 merge，因为 **没 merge 的版本在边界上肉眼可见有缝**。这就是典型的"为视觉质量付 CPU 代价"。

---

### 坑二：远处怎么办

你拼了无限大的草场。镜头往前走，近处的草很清晰，远处的草理论上也该清晰——但你的 GPU 算力是有限的，你不可能让一万米外的草也按近处精度渲染。

经典地形渲染的解法是 **LOD (Level of Detail)**：远处用低精度版本。近处一根草叶用 50 个 Gaussian，远处一片草丛用一个胖大的 Gaussian 代替。

GSWT 也做了 LOD，每个 LOD level 是一组新的 Wang Tiles，**Gaussian 密度大约是上一层的 1/4**（因为面积 2D 缩放，每级长宽各减一半）。

但 LOD 切换有个老问题：**pop-in**。你走着走着，远处某个 tile 从 LOD 2 切到 LOD 1，突然变清晰，画面"咔"一下跳变，特别出戏。

解法是 **LOD Blending**：在切换区域，两层 LOD 同时加载，渲染时按距离权重融合：

$$
\mathbf{C}_{\text{final}}(\mathbf{u}) = (1 - w) \cdot \mathbf{C}_{\text{coarse}}(\mathbf{u}) + w \cdot \mathbf{C}_{\text{fine}}(\mathbf{u})
$$

- $\mathbf{u}$ 是像素坐标
- $w \in [0, 1]$ 是基于相机距离 $d$ 的权重，比如 $w = \text{smoothstep}(d_{\text{near}}, d_{\text{far}}, d)$
- $w = 0$ 时完全用粗 LOD，$w = 1$ 时完全用细 LOD，中间平滑过渡

听起来很美，但 Table 3 告诉你代价：

| Grass | Splat Count | Render time |
|---|---|---|
| Without Blend | 23 M | 12.34 ms |
| With Blend (Full) | 41.6 M | 16.21 ms |

**Splat count 几乎翻倍**。因为过渡区的每个 tile 现在要同时加载两层 Gaussian。这就是为什么 GPU 厂商爱 LOD 又恨 LOD blending——它把切换瞬间的不连续成本，摊到了整个过渡区上，GPU 永远要为"还没完全切过去的那个 LOD"付一份算力。

---

### 坑三：拼图怎么贴到 3D 地形上

Wang Tiles 是个 2D 概念。它默认铺在平面上。但真实地形不是平面，有山有谷。

怎么把 2D 的 tile 拼图"包"到 3D 曲面上？**Surface Parameterization**。

最简单的方案，也是 paper 里 default 的方案：**height field**。地形是一个 2D 高度图 $h(u, v)$，每个 3D 点就是 $(u, v, h(u, v))$。tile 直接铺在 $(u, v)$ 平面上，3D 位置由高度图决定。

更复杂的曲面（球面、任意 mesh）也能做，用 [LSCM (Least Squares Conformal Maps)](https://www.cs.jhu.edu/~misha/Fall05/10.1.1.108.2491.pdf) 或者 [ARAP (As-Rigid-As-Possible)](https://www.cs.toronto.edu/~jacobson/images/igor-chig-kun-edited.pdf) 这种 parameterization 方法把曲面展开到 2D。

这件事的代价在 Table 3 里写得很清楚：

| Grass | Update time |
|---|---|
| With Surface Param. | 5.18 ms |
| w/o Surface Param. | 1.60 ms |

**Update 时间涨了 3 倍多**。因为 parameterization 主要用在"决定每个 tile 的 LOD"和"算 tile 的 bounding box"这两件事上，这都发生在 CPU 端的 update 阶段。

---

## 最后一个 trick：最远的 LOD 用 mesh

3DGS 在远处有个尴尬：远处一个 Gaussian 投影到屏幕上可能小于一个像素，这时它作为"体积表达"的优势完全消失，反而比 mesh 还贵（因为 mesh 可以走 hardware pipeline 优化得飞起，而 Gaussian 必须走 sort + alpha blending 的慢路）。

所以 GSWT 在最远 LOD 上**直接换成传统 mesh**，叫 **Proxy Mesh**。一个粗略的三角面片加上简单 texture，表示远山远草。

Table 3 的 w/o Proxy 行与 Full 行几乎没差别（Grass: 15.42 vs 16.21 ms），说明 proxy mesh 在性能上**几乎免费**——因为它本身渲染开销极小，省下的 GPU 时间和内存相对总开销可以忽略。但概念上很重要，它告诉你 3DGS 不该统治所有距离层级，**远处是 mesh 的主场**。

---

## 把四个 trick 串起来看

| Trick | 解决什么问题 | 付什么代价 |
|---|---|---|
| Wang Tiles | 无限大场景的复用 | 训练时构造 16 块 tile 的复杂度 |
| Selective Merging | 拼图缝 | Sort 时间 6 倍 |
| LOD Blending | LOD 切换的 pop-in | Splat count 翻倍 |
| Surface Param. | 贴到 3D 曲面 | Update 时间 3 倍 |
| Proxy Mesh | 最远 LOD 的开销 | 几乎免费 |

每个 trick 都是在**用一个维度的成本换另一个维度的体验**。这就是 real-time graphics 的永恒主题——**没有免费的午餐，只有聪明的取舍**。

---

## 这套方案的真正瓶颈

Table 1 是全文最有价值的一张表，它告诉你：**当你把地图放大 4 倍，splat count 只涨 61%，但 render time 涨 6 倍**。

为什么？因为 **CPU draw call 成了瓶颈**。每个 tile 是一次 draw call。地图变大，tile 数变多，draw call 变多，GPU 在等 CPU 发命令。

这是整个方案最该被优化的地方。**未来方向应该是 GPU-driven rendering**：用 [indirect draw](https://vulkan.org/news/en/vulkan-specs-and-extensions-update) 或者 [mesh shader](https://developer.nvidia.com/blog/introduction-turing-mesh-shaders/) 把 tile 选择、LOD 决策、draw call 发起全部搬到 GPU 端，CPU 只负责"我相机在这"这一条信息。

---

## 类比总结

你可以把 GSWT 想成 **"3D 版的 Minecraft"**：

- Minecraft 用 16×16 像素的 texture atlas，拼出整个世界。每个 block 的 texture 是从 atlas 里取一小块。
- GSWT 用 16 块 Gaussian tile atlas，拼出整个 3D 世界。每个 tile 是从 atlas 里取一组 Gaussian。

Minecraft 的 texture 是 2D 像素，GSWT 的"texture"是 3D Gaussian。Minecraft 用 block 网格决定拼贴位置，GSWT 用 Wang Tiles 的边颜色匹配规则决定。

**核心思想一脉相承：用一小块有限的"原料"，通过组合规则，生成无限丰富的世界**。

这就是程序化生成（Procedural Generation）和拼接合成（Texture Synthesis）的精髓，从 [Perlin noise](https://mrl.cs.nyu.edu/~perlin/doc/oscar.html) 到 [Wang Tiles](https://dl.acm.org/doi/10.1145/882262.882265) 到 GSWT，工业界一直在玩这个。

---

## 一句话总结

GSWT 把 3DGS 从"一个场景的技术"升级成"无限世界的引擎"，靠的不是改 3DGS 的渲染内核，而是给它套了一层 **tiled + LOD + streamed** 的工业级外壳。这层外壳在 mesh 时代叫 [geometry clipmap](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/07/Geometry-Clipmaps.pdf)，在 texture 时代叫 [megatexture](https://en.wikipedia.org/wiki/MegaTexture)，在 neural 时代会叫什么我不知道，但思路不会变。

**渲染的尽头不是算法，是工程组织数据的方式**。

---

## References

- [3DGS 原始论文 (Kerbl et al. 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [Wang Tiles 经典论文 (Cohen et al. 2003)](https://dl.acm.org/doi/10.1145/882262.882265)
- [GPU Wang Tiling (Lefebvre & Hoppe 2006)](https://dl.acm.org/doi/10.1145/1141911.1141914)
- [Geometry Clipmaps (Losasso & Hoppe 2004)](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/07/Geometry-Clipmaps.pdf)
- [MegaTexture (id Tech 5 / John Carmack)](https://en.wikipedia.org/wiki/Id_Tech_4#MegaTexture)
- [Gaussian Mixture Moment Matching](https://en.wikipedia.org/wiki/Mixture_distribution#Moments)
- [NVIDIA Mesh Shaders 介绍](https://developer.nvidia.com/blog/introduction-turing-mesh-shaders/)
- [LSCM Parameterization](https://www.cs.jhu.edu/~misha/Fall05/10.1.1.108.2491.pdf)
- [ARAP Parameterization](https://www.cs.toronto.edu/~jacobson/images/igor-chig-kun-edited.pdf)
- [Perlin Noise](https://mrl.cs.nyu.edu/~perlin/doc/oscar.html)
- [GSWT 作者主页 (Pedro Sander)](https://www.cse.ust.hk/~psander/)

要不要我接着深挖某一块？比如那 16 块 Wang Tile 具体怎么训练、edge consistency loss 长什么样、或者 sort 阶段的 selective merging 具体怎么并行化？你说方向，我往深里钻。

---

# GSWT: Gaussian Splatting Wang Tiles 深度解析

Andrej，这份 supplementary material 是 ACM TOG(SIGGRAPH Asia) 一篇关于将 **3D Gaussian Splatting (3DGS)** 与 **Wang Tiles** 经典纹理合成理论结合的工作。虽然你只给了 supplementary，但通过表格数据的反推，可以基本还原出整个方法的核心架构。我会从直觉出发，逐层拆解。

---

## 1. Wang Tiles 的历史与核心直觉

**Wang Tiles** 由数学家 **Hao Wang** 在 1961 年提出，最初用于研究形式系统中的可判定性问题。其基本构造是：一组正方形 tile，每条边带一个"颜色"（即语义标签），拼接时相邻 tile 的边颜色必须匹配。

经典配置（详见 [Cohen et al., 2003](https://dl.acm.org/doi/10.1145/882262.882265)）使用 4 种颜色、16 块 tile（2x2 = 4^2/2 边组合，考虑旋转去重后约 16 种），即可通过 stochastic tiling 生成 **非周期、视觉上无重复感** 的大面积图案。

直觉理解：Wang Tiles 的核心价值不是"避免重复"，而是"**让重复在视觉上不可见**"。从信息论角度，它把"重复检测"问题转化为"边缘一致性约束"问题。这套思路在 2D texture synthesis 中已经被反复验证（[Wei & Levoy, 2000](https://graphics.stanford.edu/papers/texture-synthesis-sig00/texture.pdf); [Lefebvre & Hoppe, 2006](https://dl.acm.org/doi/10.1145/1141911.1141914) 的 **Wang tile 预计算 + 运行时 GPU tiling**）。

GSWT 的核心创新：把这套 2D 上的"边缘一致性 + 随机拼接"思想，**提升到 3D 场景的 Gaussian primitive 级别**，并加上 **LOD、selective merging、surface parameterization** 这三件武器，用于 infinite/large-scale terrain 与场景渲染。

---

## 2. 3D Gaussian Splatting 复习

[3DGS (Kerbl et al., SIGGRAPH 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) 用一组 3D 高斯基元表示场景：

$$
G_i(\mathbf{x}) = \exp\!\left(-\tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_i)^{\!\top}\boldsymbol{\Sigma}_i^{-1}(\mathbf{x}-\boldsymbol{\mu}_i)\right)
$$

- $\boldsymbol{\mu}_i \in \mathbb{R}^3$：第 $i$ 个 Gaussian 的中心位置
- $\boldsymbol{\Sigma}_i = \mathbf{R}_i \mathbf{S}_i \mathbf{S}_i^{\top} \mathbf{R}_i^{\top}$：协方差矩阵，由旋转 $\mathbf{R}_i$（四元数表示）和缩放 $\mathbf{S}_i$（3 维向量）参数化，保证半正定
- 渲染采用 alpha compositing（[Zwicker et al. 2001 EWA volume splatting](https://www.cs.umd.edu/~zwicker/publications/EWAVolumeRendering-Sig01.pdf)）：

$$
\mathbf{C}(\mathbf{u}) = \sum_{i \in \mathcal{N}} \mathbf{c}_i \,\alpha_i \prod_{j=1}^{i-1}(1-\alpha_j)
$$

- $\mathbf{u}$ 为像素坐标，$\mathbf{c}_i$ 为球谐系数（SH）编码的视角相关颜色
- $\alpha_i$ 由 2D 投影后的 Gaussian 不透明度 $\alpha_i^{\text{base}}$ 决定
- 关键瓶颈：**按深度排序** —— 因为 front-to-back compositing 依赖顺序，每帧必须重排

3DGS 原生在 **百万级 Gaussian + 单一 scene** 上工作得很好。当你要渲染一个无限大的地形、草场、森林时，会出现几个致命问题：

1. 内存装不下数亿 Gaussian
2. 单次 sort 成本随 N 线性甚至超线性增长
3. 远距离用细 Gaussian 浪费算力，但用 mesh 又失去 3DGS 的细节表达

GSWT 给出的解：**Wang Tiles + 多 LOD + Surface Parameterization + Selective Merging**。

---

## 3. GSWT 架构总览（从表格反推）

虽然 supplementary 没给完整 pipeline 图，但 Table 3、Table 4 的 ablation 列出四个核心组件：

| 组件 | 作用 | 在表格中的列 |
|---|---|---|
| **Selective Merging** | tile 边界处的 Gaussian 合并，消除拼接缝 | "w/o Merge" |
| **LOD Blending** | 不同 LOD 层之间的平滑过渡 | "w/o Blend" |
| **Proxy Mesh** | 最粗糙 LOD 用传统 mesh 代替 Gaussian | "w/o Proxy" |
| **Surface Parameterization** | 将 3D 场景映射到 2D 参数域（如 height field） | "w/o Surface" |

可以推断完整 pipeline 是：

```
Exemplar Scene (3DGS trained)
       ↓
Surface Parameterization (height field / proxy)
       ↓
Tile Atlas 构造 (16 Wang Tiles with colored edges)
       ↓
For each LOD level k:
    Downsample Gaussians (e.g., 1/4 density)
    Train tile with edge color constraints
       ↓
Runtime:
    Camera → 视锥裁剪 → 确定每个 tile 的 LOD
    → Load tile's Gaussians → Selective Merge at boundaries
    → LOD Blend at level transitions
    → Sort + Render
```

---

## 4. Wang Tiles 在 Gaussian 域的构造

### 4.1 边缘颜色的语义

经典 Wang Tile 每条边带一个离散颜色。在 GSWT 中，"颜色"被推广为 **edge Gaussian distribution** —— 即一条边上的 Gaussian 集合必须与所有同色边的 Gaussian **几何上对齐**。

形式化（推测）：定义 tile $T$ 的四条边 $e \in \{N, E, S, W\}$，每条边带颜色标签 $c_e \in \{1, 2, ..., K\}$。拼接约束：

$$
\forall e_1 \in T_1,\ e_2 \in T_2,\ e_1 \cap e_2 \neq \emptyset \implies c_{e_1} = c_{e_2} \land G_{e_1} \approx G_{e_2}
$$

也就是说，相邻 tile 共享边时，**该边上及其邻域的 Gaussian 集合应当一致**，否则会出现可见缝。

### 4.2 Tile 的训练

考虑一个 $W \times H$ 的 tile 在 LOD $k$ 上，其 Gaussian 集合 $\{g_i\}_{i=1}^{N_k}$ 的训练 loss 包含：

1. **重建 loss**（标准 3DGS L1 + D-SSIM）：

$$
\mathcal{L}_{\text{rgb}} = (1-\lambda)\mathcal{L}_1 + \lambda \mathcal{L}_{\text{D-SSIM}}
$$

2. **Edge consistency loss**（推测）：让边附近的 Gaussian 与同色 tile 的对应 Gaussian 一致

$$
\mathcal{L}_{\text{edge}} = \sum_{e} \sum_{g_i \in \mathcal{N}(e)} \| g_i - g_i^{\text{ref}} \|^2
$$

3. **Density / pruning / densification**：标准 3DGS 的 adaptive density control

### 4.3 完整 16-tile 集合

Table 3 / Table 4 中提到的 "16 Wang Tiles" 对应经典 4-色配置，每色匹配每边。这个集合对每次随机 tiling 提供 ~$\log_2(16)$ bits 的随机性，足以避免在视野中出现明显的周期模式。

---

## 5. Selective Merging —— 消除拼接缝的关键

### 5.1 问题

当 Wang Tiles 拼接时，相邻 tile 在共享边附近会各自贡献一组 Gaussian。如果不处理：
- 边界处 Gaussian 密度翻倍（甚至 4 倍，因为 4 个 tile 共享一个角）
- 几何重叠导致视觉上的"重影"或"加亮带"
- 渲染时间随拼接复杂度上升

### 5.2 解决方案

**Selective Merging**：在运行时，对落在"边带"（edge band）内的 Gaussian，按某种策略**合并**：

- **加权平均**：$\boldsymbol{\mu}_{\text{merge}} = \frac{w_1 \boldsymbol{\mu}_1 + w_2 \boldsymbol{\mu}_2}{w_1+w_2}$
- **保留代表元**：取 opacity 最大的那个
- **重新估计协方差**：$\boldsymbol{\Sigma}_{\text{merge}} = \frac{w_1^2 \boldsymbol{\Sigma}_1 + w_2^2 \boldsymbol{\Sigma}_2 + \text{cross term}}{(w_1+w_2)^2}$

直觉：把"几何上重合的两个 Gaussian"合成一个等价的 Gaussian，使其在原位置上产生几乎相同的 image contribution。这正是 [Merging in 3DGS compression works (e.g., Niedermayr et al. 2024)](https://arxiv.org/abs/2401.12504) 的思路，但 GSWT 是 **运行时**、**tile 边界处** 的针对性合并。

### 5.3 性能分析

从 Table 3 的 Grass 数据看：

| 配置 | Sort time (ms) | Render time (ms) |
|---|---|---|
| Full | 17.33 ± 6.65 (82.59%) | 16.21 ± 3.65 |
| w/o Merge | 2.60 ± 0.39 (99.80%) | 16.11 ± 1.54 |

**关键观察**：Selective Merging 让 sort time 从 ~2.6ms 暴涨到 ~17.3ms（6.7 倍），但 render time 几乎没变。说明：

- **Merge 的开销全在 sort 阶段**，因为它要在 tile-level sort 之后再做一次 Gaussian-level 的合并扫描
- Render time 几乎不变是因为合并后的 Gaussian 总数没显著减少（render 是 splat count 主导，而 sort 是 log N 主导，加上 merge 的扫描成本）

为什么仍然保留？因为 **视觉质量** —— 没有 merge 的版本在 tile 边界会有可见缝。这是经典的"性能 vs 质量" trade-off。括号里的百分比是 worker thread 上的 sort 占用比例，可以看到 w/o Merge 几乎总是 99%+（说明 worker thread 几乎全时间在排序，没什么余地做其他事）；而 Full 时该百分比降到 80-90%，说明 merge 工作量占据了 worker thread 的相当一部分。

---

## 6. LOD Blending

### 6.1 LOD 切换问题

经典地形 LOD（如 [ clipmaps ](https://dl.acm.org/doi/10.1145/1073204.1073220)）在 LOD 边界会出现 pop-in。GSWT 使用 **per-tile LOD blending**：

对于处在 LOD $k \to k+1$ 过渡区的 tile $T$，同时加载两层 Gaussian 集合 $G_k$ 和 $G_{k+1}$，按距离权重融合：

$$
\mathbf{C}_{\text{blend}}(\mathbf{u}) = (1-\beta(\mathbf{u})) \, \mathbf{C}_{k}(\mathbf{u}) + \beta(\mathbf{u}) \, \mathbf{C}_{k+1}(\mathbf{u})
$$

其中 $\beta(\mathbf{u})$ 为基于相机距离的平滑过渡函数（如 smoothstep）。

### 6.2 性能代价

Table 1 显示 Grass 在 LOD max dist x4 时 splat count 从 23M → 273M（12 倍），render time 从 16ms → 138ms（8.6 倍）。这是因为更细 LOD 包含的 Gaussian 数随分辨率**指数增长**（每个 LOD 大约 4x）：

$$
N_{\text{total}} \approx \sum_{k=0}^{K} N_k \cdot A_k, \quad N_k \approx N_0 \cdot 4^{-k}, \quad A_k \approx A_0 \cdot 4^k
$$

这里 $N_k$ 是 LOD $k$ 每个 tile 的 Gaussian 数，$A_k$ 是 LOD $k$ 覆盖的 tile 数。理论上 $N_k \cdot A_k$ 应近似常数，但实际由于精细 LOD 覆盖更近的、被采样更密集的区域，$N_{\text{total}}$ 仍增长。

Table 2 给出 Grass 场景的 LOD 分布：

| LOD | Render (ms) | Splat (M) | Tiles |
|---|---|---|---|
| 0 (finest) | 6.18 | 9.45 | 76 |
| 1 | 2.46 | 4.95 | 166 |
| 2 | 2.02 | 4.23 | 600 |
| 3 | 1.19 | 2.33 | 1487 |
| 4 | 1.24 | 2.00 | 5286 |
| 5 (coarsest) | 0.24 | 0.18 | 1794 |

可以看到 **LOD 0 占总 render time 的 45%**（6.18 / 13.74），但只覆盖了 76 个 tile。这印证了"近处细节最贵"的传统渲染规律。Tile 数从 LOD 0 → LOD 4 几何级增长（76 → 5286），但 splat 数下降（9.45M → 2.00M），所以总贡献相对平稳。LOD 5 (proxy? 或最粗) tile 数反而少（1794），可能是因为远处 tile 被合并为 proxy mesh。

Table 3 中 w/o Blend 行：

- Grass: render time 从 16.21ms 降到 12.34ms（节省 ~24%），splat count 从 23M (with blend) 降到 23M (base, 因为 with blend 是基础值)
- 等等，仔细看：Splat Count 列写的是 23,098,554，With Blend 列是 41,654,961。**With Blend 几乎是 2x** —— 因为 blending 区域同时加载两层。

这就是 LOD Blending 的代价：**在过渡区，splat count 几乎翻倍**。但收益是 pop-in 消失。

---

## 7. Proxy Mesh —— 最粗 LOD

**Proxy mesh** 是用一个传统 mesh（triangle soup）+ 简单纹理来表示最远的 LOD，而不是 Gaussian。

Table 3 中 w/o Proxy 行基本与 Full 行接近（Grass: 15.42 vs 16.21 render），说明：

- Proxy mesh 本身渲染开销极小
- 它主要省的是 GPU 内存（不用存最粗 LOD 的 Gaussian）和 sort 时间（mesh 不需要 per-frame sort）
- 它的代价是切换时的"几何突变"，但因为是远处，视觉影响小

直觉：在足够远距离，3DGS 的"细节表达"优势消失 —— 像素 footprint 大于一个 Gaussian 的 screen size，此时 mesh 反而更便宜。这与 [Radial Basis Function / impostor 的混合渲染](https://www.cs.cmu.edu/~jbieres/0mm/0mm.pdf) 思路一致。

---

## 8. Surface Parameterization

### 8.1 为什么需要参数化

Wang Tiles 本质是 **2D 概念**。要把它用到 3D 场景上，需要一个 2D 参数域 $\Omega \subset \mathbb{R}^2$，让每个 3D 点 $\mathbf{x} \in \mathbb{R}^3$ 映射到 $\Omega$。

最简单的方案：**height field** $\mathbf{x} = (u, v, h(u,v))$。所有 supplementary 表格中的 "All scenes are parameterized by a height field by default" 印证了这一点。

更一般的参数化（如 [least squares conformal maps](https://www.cs.jhu.edu/~misha/Fall05/10.1.1.108.2491.pdf) 或 [ARAP parameterization](https://www.cs.toronto.edu/~jacobson/images/igor-chig-kun-edited.pdf)）允许更复杂的拓扑，例如球面、地形曲面。

### 8.2 作用

1. **确定 tile 的 LOD**：参数化后，tile 在 $\Omega$ 上有明确的 bounding box。相机距离 tile 中心 $d$ 决定 LOD：

$$
k(\mathbf{p}) = \lfloor \log_2(d / d_0) \rfloor
$$

其中 $d_0$ 是参考距离。这就是 [geometric clipmap](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/07/Geometry-Clipmaps.pdf) 的思路。

2. **Pre-sort view selection**：tile 在 $\Omega$ 上的位置可预先计算视点相关的渲染顺序，减少运行时 sort 成本。

3. **Bounding box check** 用于 LOD blending：判断 tile 是否落在过渡区，需要 bounding box 查询。

Table 3 中 w/o Surface 行：

- Grass: Update time 从 5.18ms 降到 1.60ms（节省 ~69%）
- Desert: Update time 从 5.93ms 降到 1.51ms

**Surface parameterization 的开销几乎全在 update 阶段**，因为它要为每个 tile 计算 bounding box、查询 LOD 等。这是合理的：渲染端不直接依赖参数化，但 tile 信息更新强依赖。

---

## 9. 可扩展性分析（Table 1）

Table 1 给出 Grass 场景的可扩展性测试：

| Map width | Tiles 数 | Splat count | Render (ms) | Sort (ms) | Update (ms) |
|---|---|---|---|---|---|
| Baseline (97) | ~1× | 23M | 16.28 | 18.64 | 5.64 |
| ×2 | ~4× | 26M (+13%) | 27.55 | 32.69 | 21.77 |
| ×4 | ~16× | 37M (+61%) | 97.69 | 73.34 | 91.60 |

**关键 insight**：

- **Splat count 增长缓慢**（4 倍 map 仅增 61%），因为新增的 tile 大部分使用最粗 LOD。这印证了 LOD 机制有效。
- **Render time 增长 6 倍**，远超 splat count 增长。瓶颈是 **draw call 数量**（CPU overhead），不是 GPU splat 数。
- **Sort time 增长 4 倍**，因为 sort 涉及 tile-level 排序 + Gaussian-level selective merging，都随 tile 数线性增长。
- **Update time 增长 16 倍**（5.64 → 91.60），几乎与 tile 数线性 —— 这正是"update 只涉及 tile-level 信息"的直接证据。

**结论**：在大地图场景，CPU 端的 draw call 与 tile update 是主要瓶颈，而非 GPU 端的 splat count。这暗示了未来优化方向：**instanced rendering / GPU-driven pipeline / indirect draw**。

### LOD max dist 缩放

| LOD max dist | Splat count | Render (ms) |
|---|---|---|
| Baseline (384) | 23M | 16.28 |
| ×2 | 84M | 47.06 |
| ×4 | 274M | 138.15 |

这里 splat count **指数增长**（每次 ×3.6），render time 也指数增长。原因是更细 LOD 在更远距离上启用，而细 LOD 的 Gaussian 数随分辨率指数增长。

---

## 10. Real-world vs Synthetic 对比（Table 3 vs Table 4）

Table 4 只有 Forest 一个 real-world 场景，但表现出与合成场景一致的规律：

- **Selective Merging 开销**：sort time 从 2.24ms → 5.97ms（2.7 倍）
- **LOD Blending 开销**：splat count 9M → 17.9M（2x），render 6.72ms → 8.10ms
- **Surface parameterization 开销**：update 0.75ms → 5.26ms

Real-world 场景的 update time 比例（11.74%）略高于合成场景，可能因为真实场景几何更复杂、bounding box 查询更频繁。

---

## 11. 综合直觉总结

GSWT 的核心思想可以总结为一句话：

**用 Wang Tiles 的"边缘一致性 + 随机拼接"机制，把无限大场景的表示问题转化为有限 tile atlas 的复用问题；用 multi-LOD + selective merging + LOD blending + proxy mesh 让 3DGS 可在任意距离、任意尺度上实时渲染。**

从工程角度看，它的关键 trade-off 是：

1. **Sort time ↔ Visual seam quality**：selective merge 让 sort 时间翻 6 倍但消除拼接缝
2. **Splat count ↔ Pop-in**：LOD blend 让 splat 翻倍但消除 LOD 切换突变
3. **Update time ↔ Geometric flexibility**：surface parameterization 让 update 时间翻 3-4 倍但支持任意曲面
4. **CPU draw calls ↔ Map size**：scale test 表明 CPU 是大地图瓶颈，GPU 不是

它最聪明的地方在于：**没有试图改 3DGS 的渲染内核，而是把 3DGS 当作"细节原语"，套上一层 tile/LOD/parameterization 的"组织结构"**。这与经典地形渲染（[clipmaps](https://dl.acm.org/doi/10.1145/1073204.1073220), [virtual texture](https://www.cs.cmu.edu/~jhardin/VT/VT.pdf)）的设计哲学完全一致，只是把"texel"换成了"Gaussian"。

---

## 12. 可能的延伸与思考

作为 Andrej，你可能对以下方向感兴趣：

1. **Tile 训练的端到端可微性**：目前 atlas 是预训练的，是否可以让 tile 内容在运行时根据用户行为继续优化？这与 [NeRF editing / online 3DGS](https://arxiv.org/abs/2312.03026) 相关。

2. **Wang Tile 边颜色学习的可微化**：边颜色目前是离散标签，能否用 [Gumbel-softmax](https://arxiv.org/abs/1611.01144) 训练一个 edge color classifier，让网络自动发现最优的颜色集？

3. **Selective Merging 的可微实现**：当前 merge 是不可微的启发式。是否可以用 [Differentiable Rendering with Gaussian Mixture](https://arxiv.org/abs/2106.12052) 让 merge 在 backward 时也工作？

4. **GPU-driven 渲染**：scale test 表明 CPU draw call 是瓶颈。能否用 [indirect draw + mesh shader](https://developer.nvidia.com/blog/introduction-turing-mesh-shaders/) 把 tile 选择、LOD 决策全部搬到 GPU？

5. **与 neural radiance caching 的结合**：远处 LOD 可以用 neural representation 替代 mesh proxy，类似 [Neural Radiance Cache (NRC)](https://research.nvidia.com/publication/2021-06_Neural-radiance)。

---

## Reference 链接

- **3DGS 原始 paper**：https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- **Kerbl et al. SIGGRAPH 2023**：https://dl.acm.org/doi/10.1145/3592453
- **Wang Tiles (Cohen et al. 2003)**：https://dl.acm.org/doi/10.1145/882262.882265
- **Lefebvre & Hoppe 2006 (GPU Wang tiling)**：https://dl.acm.org/doi/10.1145/1141911.1141914
- **Texture Synthesis (Wei & Levoy)**：https://graphics.stanford.edu/papers/texture-synthesis-sig00/texture.pdf
- **Geometric Clipmaps**：https://www.microsoft.com/en-us/research/wp-content/uploads/2005/07/Geometry-Clipmaps.pdf
- **EWA Volume Splatting**：https://www.cs.umd.edu/~zwicker/publications/EWAVolumeRendering-Sig01.pdf
- **Gaussian Merging for Compression**：https://arxiv.org/abs/2401.12504
- **Online 3DGS**：https://arxiv.org/abs/2312.03026
- **NRC (Neural Radiance Cache)**：https://research.nvidia.com/publication/2021-06_Neural-radiance
- **GSWT paper (main)**：在 supplementary 中未给出 main paper 链接，可能发表在 ACM TOG 或 SIGGRAPH Asia 2024，可在作者主页查找 https://www.cse.ust.hk/~psander/

如果你想深入某个组件（比如 selective merge 的具体算法、LOD 决策函数、tile atlas 的训练细节），可以告诉我，我可以基于 supplementary 的数据 + 3DGS/Wang Tile 经典方法做更细致的推断与公式推导。
