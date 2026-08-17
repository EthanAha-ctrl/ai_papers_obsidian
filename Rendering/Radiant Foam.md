---
source_pdf: Radiant Foam.pdf
paper_sha256: 57181465ec018e8b8af4b9f98e9ed92a6decb62753ccedf4ce001cff72b06053
processed_at: '2026-08-11T20:48:18-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Radiant Foam 人话版

## 一句话总结

**用一锅"肥皂泡"来表示3D场景，光线穿过泡泡一个接一个走，不需要专门硬件就能实时做ray tracing，质量接近3DGS。**

---

## 背景：为什么需要这个东西？

现在的3D场景表示方法主要有两个阵营：

### 阵营一：NeRF（ray marching）
- 想象你在3D空间里发射一条光线，沿路采样很多点
- 每个点问一个小神经网络："这里有啥？什么颜色？"
- 把沿路信息积攒起来得到最终像素颜色
- **优点**：光线怎么走就怎么算，reflection、refraction、镜头扭曲都天然支持
- **缺点**：每个像素要query几百次MLP，慢得要死，<1 FPS

参考：https://www.matthewtancik.com/nerf

### 阵营二：3DGS（rasterization）
- 把场景塞满一堆椭圆形"果冻"（Gaussian）
- 每个果冻有位置、大小、颜色
- 渲染时把果冻按深度排序，一个个"压扁"到屏幕上
- **优点**：GPU rasterization 硬件天生擅长这个，300+ FPS
- **缺点**：rasterization 是个"近似"，把光线的复杂行为简化掉了。想加reflection？想搞rolling shutter？想处理鱼眼镜头？都得写一堆hack

参考：https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

### 历史在重复

作者观察到一个有趣的pattern：

**Classical graphics** 早期用 ray tracing，后来为了real-time改用 rasterization，再后来NVIDIA出了RTX，ray tracing又回来了。

**NeRF领域**也是这个剧本：早期ray-based（NeRF），后来转rasterization（3DGS），现在是不是也该回来了？

但问题是：ray tracing 通常需要专门硬件（RT cores）才能real-time。NVIDIA自己的3DGRT [Moenne-Loccoz et al. 2024] 用OptiX+RTX才跑到78-119 FPS。

参考：https://research.nvidia.com/labs/toronto-ai/3DGRT/

**这篇论文说：其实不用专门硬件也能real-time ray tracing，秘诀在于representation的结构。**

---

## 核心idea：Voronoi Foam

### 什么是Voronoi diagram？

想象你在地板上撒一把钉子。每个钉子"统治"离它最近的那片地板区域。所有钉子的领地拼起来就是Voronoi diagram。

每个区域是一个凸多面体（3D情况），叫做一个"cell"。

参考：https://en.wikipedia.org/wiki/Voronoi_diagram

### 为什么用Voronoi而不是Delaunay？

Delaunay triangulation 是 Voronoi 的对偶：把每个钉子和它领地相邻的钉子连起来，得到一堆tetrahedra。

但 Delaunay 有个讨厌的性质：你稍微动一下某个钉子位置，connectivity 可能突然"flip"——本来A和B连着，瞬间变成A和C连。这种离散跳变让gradient descent崩溃。

**Voronoi的妙处**：虽然它也有这种flip，但flip发生的瞬间，受影响的那个cell face的面积**恰好为零**！

打个比方：你在调音台推fader，某个声道突然mute了，但mute的瞬间音量恰好是0，所以你根本听不到"啪"的一声。对优化算法来说，这就像是"什么都没发生"。

所以虽然topology在离散变化，但**渲染出来的图像**相对于钉子位置是完全连续可微的。gradient descent完全不知道发生了flip，照样能work。

### "Foam"是什么意思？

物理上的stable foam（比如啤酒泡、肥皂泡）满足：
- 每个bubble内部pressure相等
- bubble之间的interface是平面
- 整体结构类似Voronoi diagram

所以作者把他们的representation叫"foam"——一锅数学肥皂泡，每个泡泡会发光，所以叫"Radiant Foam"。

参考：https://en.wikipedia.org/wiki/Plateau%27s_laws

---

## Ray Tracing怎么这么快？

### 传统ray tracing为什么慢

一般ray tracing要在场景里找"光线撞到什么了"。场景可能有几百万个triangle，不能一个个试，所以要先建一棵BVH树（bounding volume hierarchy）。每次query是 $O(\log n)$。

NVIDIA RTX硬件专门加速这个BVH query，所以能real-time。没有这个硬件就只能干瞪眼。

### Weiler et al. 2003 的老办法

这篇论文的核心发现是：**20年前有个叫Weiler的德国人发了一个paper** [Weiler et al. 2003]，提出在volumetric mesh上做ray tracing不需要acceleration structure。

原理特别简单：既然mesh的cells是space-filling的（把整个空间填满，无缝隙），那光线从cell A出去必然进cell B。你只要知道A的邻居是谁，一个一个试过去就行。

伪代码就是：
```
找到起点所在的cell
while 光还在场景里:
    for 每个邻居cell:
        算光线和共享face的交点
    选最近的那个交点，进入那个邻居cell
    累积这个cell的颜色和密度
```

**关键**：这个loop的步数等于光线穿过的cell数，和总cell数无关。如果场景有100万cell但光线只穿过50个，那只需要50步。

参考：https://ieeexplore.ieee.org/document/1196223

### 为什么这个老paper被遗忘了？

因为2003年那会儿大家不搞differentiable rendering，这种算法主要用于科学可视化（医学CT、流体仿真post-processing）。后来NeRF火了，大家都在想MLP、hash grid、Gaussian这些，没人回头看graphics老paper。

作者团队里有人有graphics背景（Andrea Tagliasacchi），翻出来了这个gem。

---

## 训练过程怎么搞？

### 初始化

从COLMAP得到sparse point cloud，作为初始Voronoi sites。

参考：https://colmap.github.io/

### Densification

训练过程中，看哪些cell的gradient norm特别大（乘以cell半径normalize一下），说明那里"underfitting"，需要更多capacity。

用Multinomial sampling选cell来densify（借鉴3DGS-MCMC思路）。被选中的cell分裂成两个，新的site位置在原site附近perturb一下。

参考：https://arxiv.org/abs/2404.09105

### Pruning

删除满足两个条件的cell：
1. 自己density很低
2. 所有邻居density也很低

为什么要加第二个条件？因为Voronoi cell的geometry由邻居决定。哪怕某个cell自己density=0，它可能还在"定义"邻居cell的边界，删了它邻居的形状就变了，surface会破。

### Quantile Loss

为了减少"floater"（悬在空中的鬼影），加了个regularization。

Mip-NeRF 360用的是distortion loss，需要 $O(N^2)$ 的nested sum。这篇paper换成sample-based的quantile loss，便宜很多。

intuition：希望density集中在surface上，weight distribution应该是"spiky"的，少数几个cell吃掉绝大部分weight。quantile loss惩罚"weight分布太平"的情况。

---

## 实验结果到底怎么样？

### 质量

| Method | Mip-NeRF 360 PSNR | FPS |
|--------|-------------------|-----|
| 3DGS | 28.69 | 293 |
| Mip-NeRF 360 | 29.23 | <1 |
| 3DGRT (NVIDIA RTX) | 28.71 | 78 |
| **Radiant Foam** | 28.47 | **200** |

比3DGS低0.2-0.5 dB，肉眼几乎看不出。比Mip-NeRF 360低0.7 dB。

### 速度

200 FPS（Mip-NeRF 360 scenes）和301 FPS（Deep Blending scenes），比3DGRT快2-3倍，**而且不用RTX硬件**。

这个结果挺震撼的。NVIDIA自己用专门硬件做ray tracing，被一个20年前的算法+一些engineering trick给秒了。

### Ablation

最重要的component是densification（去掉掉10 dB），其次是quantile loss（去掉掉1.25 dB）和SfM initialization（去掉掉2 dB）。Pruning影响很小。

---

## 为什么这篇论文重要？我的几个直觉

### 1. Representation的结构可以替代硬件

这是个很深的insight。大家一直觉得ray tracing必须用BVH+硬件加速。但这篇paper说：如果你的representation本身有good structure（space-filling tessellation），ray traversal就是 $O(\text{ray length})$，根本不需要加速结构。

这启发我们去想：还有什么representation有类似的"built-in acceleration"性质？

### 2. 离散拓扑变化可以"藏"起来

Voronoi diagram的edge flip是个离散事件，但因为它发生在measure-zero的区域，对continuous field没有影响。

这是个generalizable的idea。比如：
- Mesh representation里，connectivity change可以设计在degenerate configuration发生
- Discrete optimization的continuous relaxation可能不需要Gumbel-softmax这种，只要discrete change在"silent"时刻发生就行

### 3. Ray-based + Real-time是个新空间

3DGS之后的很多工作都在打补丁：fix popping、支持distortion、加reflection。每个都hack一堆。

Radiant Foam给了一个principled框架：只要你能写出ray的行为，就能渲染。reflection就是ray bouncing，refraction就是Snell law在cell boundary处apply，rolling shutter就是per-row改ray origin/time，motion blur就是time-varying ray。

这对sensor modeling、VR/AR、robotics vision都很有用。

### 4. 历史的轮回

Classical graphics：ray tracing → rasterization → ray tracing (with RTX)
Differentiable rendering：ray marching (NeRF) → rasterization (3DGS) → ray tracing (Radiant Foam?)

如果这个pattern成立，未来可能更多work回归ray-based方法。

---

## Limitations 和open questions

### 作者承认的

- Voronoi要求boundary是sites的perpendicular bisector，导致很多小empty cell来定义surface，浪费capacity
- 只支持static scene
- 没有relighting
- 没有streaming
- 没有generative

### 我推测的

**Memory**：Voronoi cells可能比Gaussian占更多memory（empty cells的开销）。论文没报memory数字，可疑。

**Outdoor scenes表现差**：Table 3看Bicycle、Stump比indoor差很多，可能unbounded background对Voronoi不友好。

**Sharp features**：piecewise constant在每个cell内，sharp edges需要很多cell才能逼近。

**Training stability**：incremental Delaunay对coincident points敏感，论文自己也提了。大场景可能容易崩。

### 我好奇的方向

**Power Diagram**：给每个site加个weight，cell boundary不再要求等距。这能解耦cell size和density，可能减少empty cells。

参考：https://en.wikipedia.org/wiki/Power_diagram

**Hierarchical foam**：类似hierarchical 3DGS，搞LOD。远处用大cell，近处用小cell。Voronoi的递归subdivision可能work。

**Foam + MLP**：每个cell内不再constant，而是small MLP。增加expressiveness，可能追上Mip-NeRF 360的质量。

**Differentiable physics on foam**：Voronoi天然适合FEM，能不能做differentiable fluid sim？

**Generative foam**：diffusion model直接生成Voronoi sites + per-cell attributes。比生成Gaussian更structured。

**Optimal transport connection**：Voronoi和OT关系很深，能不能用OT工具优化foam的capacity分配？

---

## 一句话收尾

这篇paper挖出了一个被遗忘的graphics算法，配合Voronoi的优雅数学性质，证明了"ray-based + real-time + differentiable"三者可以同时拥有。质量还差一点，但打开了一个新空间——representation的structure本身就是一种"硬件加速"。

---

# Radiant Foam: Real-Time Differentiable Ray Tracing 深度讲解

## 1. 论文的核心 motivation 与历史定位

这篇论文的核心 insight 可以追溯到 computer graphics 的历史发展脉络。作者提出了一个非常深刻的观察：**history is repeating itself**。

### 1.1 从 classical graphics 到 NeRF 的轮回

在 classical computer graphics 中，rasterization 是为了 real-time rendering 而引入的 approximation of the rendering equation [Kajiya 1986]。但 ray tracing 能 trivially 处理的 effects（reflection, refraction, transparency），在 rasterization 下需要大量 clever tricks（参考 Graphics Gems 系列）。

NeRF [Mildenhall et al. 2020] 的发展重复了这个故事：
- 早期 NeRF：ray-based volume rendering，质量高但慢
- 3DGS [Kerbl et al. 2023]：用 rasterization 换取 real-time 性能
- 但 3DGS 的 rasterization 近似带来了 limitations：camera distortion、secondary lighting、rolling shutter 等难以建模

Radiant Foam 的核心 question：**能否同时拥有 ray tracing 的表达力和 rasterization 的速度？**

参考链接：
- NeRF: https://www.matthewtancik.com/nerf
- 3DGS: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- Rendering Equation: https://en.wikipedia.org/wiki/Rendering_equation

---

## 2. 核心技术：Voronoi Diagram 作为 Differentiable Volumetric Mesh

### 2.1 为什么不用 Delaunay Triangulation？

Delaunay triangulation [Delaunay 1934] 是经典的 mesh 构造方法：给定一组点 $\{\mathbf{p}_i\}$，找到所有满足 "circumsphere 内部不包含其他点" 的 tetrahedra。

但这里有两个致命问题：

**问题 1：Discrete "edge flips"**
当某个 vertex 进入另一个 cell 的 circumsphere 时，connectivity 会发生离散变化。如图 6 所示，两个 neighboring simplices 的 circumsphere 重合时，triangulation 的 edge 会 "flip"。这导致 optimization landscape 出现 discontinuities，gradient descent 无法收敛。

**问题 2：Cell 数量不固定**
Delaunay triangulation 的 tetrahedra 数量随 configuration 变化，无法为每个 cell 关联可优化的 $\sigma$ 和 $\mathbf{c}$ 值。

### 2.2 Voronoi Diagram 的妙处

Voronoi diagram [Voronoi 1908] 是 Delaunay 的 dual graph。每个 cell $\mathbf{c}_i$ 定义为：

$$\mathbf{c}_i = \{x \in \mathbb{R}^3 : \arg\min_j ||x - \mathbf{p}_j|| = i\}$$

其中：
- $x$ 是 3D 空间中的任意点
- $\mathbf{p}_j$ 是第 $j$ 个 primal vertex（Voronoi site）
- $\arg\min_j ||x - \mathbf{p}_j|| = i$ 表示 $x$ 到所有 sites 的最近邻是第 $i$ 个 site

**关键的 continuity argument**：

虽然 Voronoi diagram 也有 discrete connectivity changes（与 Delaunay 同步），但这些 discrete changes 恰好发生在 **cell face area 为零** 的时刻！

让我详细解释这个 observation 的深刻含义：

考虑 Figure 6 中的 edge flip 过程：
- Left：两个 neighboring simplices 的 circumsphere 不同
- Center：circumsphere 重合，Voronoi cell 的某个 face 面积恰好为零
- Right：flip 完成后，新的 face 出现

在 volume rendering 中，cell boundary 对 ray 的贡献正比于 face area。当 face area = 0 时，这个 face 对 rendering 没有贡献，因此 discrete change 被 "hidden" 在 zero-volume region 中。

**这意味着**：虽然 topology 是离散变化的，但 **field representation**（即 $\sigma(\cdot)$ 和 $\mathbf{c}(\cdot)$ 作为空间函数）相对于 primal vertex positions 是完全连续的。这是 gradient-based optimization 的关键。

参考链接：
- Delaunay Triangulation: https://en.wikipedia.org/wiki/Delaunay_triangulation
- Voronoi Diagram: https://en.wikipedia.org/wiki/Voronoi_diagram
- DMTet (相关 work): https://research.nvidia.com/labs/toronto-ai/DMTet/

---

## 3. Volume Rendering 的精确形式

### 3.1 连续形式

标准 volume rendering integral（公式 1 和 2）：

$$\mathbf{c_r} = \int_{t_{\min}}^{t_{\max}} T(t) \cdot \sigma(\mathbf{r}(t)) \cdot \mathbf{c}(\mathbf{r}(t)) \, dt$$

$$T(t) = \exp\left(-\int_{t_{\min}}^{t} \sigma(\mathbf{r}(u)) \, du\right)$$

变量解释：
- $\mathbf{c_r}$：ray $r$ 观察到的 color
- $t_{\min}, t_{\max}$：ray 与 scene bounding volume 的交点参数
- $\mathbf{r}(t)$：ray 上距离 $t$ 处的 3D 点
- $T(t)$：transmittance，从 $t_{\min}$ 到 $t$ 的累积透过率
- $\sigma(\cdot)$：density field（体积密度）
- $\mathbf{c}(\cdot)$：radiance field（颜色/辐射度）

### 3.2 Piecewise Constant 离散形式

当 $\sigma$ 和 $\mathbf{c}$ 在每个 cell 内 piecewise constant 时（公式 3 和 4）：

$$\mathbf{c_r} = \sum_{n=1}^{N} T_n \cdot (1 - \exp(-\sigma_n \delta_n)) \cdot \mathbf{c}_n$$

$$T_n = \prod_{j=1}^{n} \exp(-\sigma_j \delta_j)$$

变量解释：
- $N$：ray 经过的 cell 数量
- $T_n$：到达第 $n$ 个 cell 的累积 transmittance
- $\sigma_n$：第 $n$ 个 cell 的 density
- $\delta_n$：ray 在第 $n$ 个 cell 内的行进距离（segment width）
- $\mathbf{c}_n$：第 $n$ 个 cell 的 color

**关键 insight**：对 NeRF 来说，公式 (3) 是公式 (1) 的近似（需要 importance sampling）。但对 Radiant Foam 来说，**这两个公式是 exactly equivalent**！因为每个 Voronoi cell 内部 $\sigma$ 和 $\mathbf{c}$ 真的是 constant，且 cell boundary 是精确的 polyhedral faces。

这避免了 NeRF 中复杂的 sampling schemes（如 Mip-NeRF 的 conical frustums、Zip-NeRF 的 grid-based sampling）。

参考链接：
- Volume Rendering Digest: https://arxiv.org/abs/2203.15906
- Mip-NeRF: https://jonbarron.info/mipnerf/
- Zip-NeRF: https://arxiv.org/abs/2304.06706

---

## 4. Ray Tracing Algorithm 深度解析

### 4.1 Algorithm 1 逐行解析

```
1: procedure RENDER(o, d)          // o: ray origin, d: ray direction
2:   t_0 ← 0
3:   i ← nn(o)                     // 找到 origin 最近的 Voronoi site
4:   T ← 1                         // 初始 transmittance
5:   C ← 0                         // 累积 color
6:   while T > ε do
7:     x ← v_i                     // v_i: cell i 的 primal vertex
8:     t_1 ← ∞
9:     i' ← ∅
10:    for all j ∈ N(i) do          // N(i): cell i 的邻居集合
11:      x' ← v_j
12:      (t_j, front) ← INTERSECT(o, d, x, x')
13:      if front and (t_j < t_1) then
14:        t_1 ← t_j
15:        i' ← j
16:      end if
17:    end for
18:    c ← c_i                      // cell i 的 color
19:    σ ← σ_i                      // cell i 的 density
20:    α ← 1 - exp(-σ(t_1 - t_0))  // alpha compositing
21:    C ← C + T·α·c
22:    T ← T·(1 - α)
23:    t_0 ← t_1
24:    i ← i'
25:  end while
26:  return C
27: end procedure
```

### 4.2 算法的关键设计

**1. Cell face 的表示**
每个 Voronoi cell 的 face 是两个 neighboring sites $\mathbf{v}_i$ 和 $\mathbf{v}_j$ 之间的 perpendicular bisector plane。这个 plane 的方程为：

$$(\mathbf{v}_j - \mathbf{v}_i) \cdot \mathbf{x} = \frac{1}{2}(\|\mathbf{v}_j\|^2 - \|\mathbf{v}_i\|^2)$$

ray $\mathbf{o} + t\mathbf{d}$ 与这个 plane 的交点：

$$t_j = \frac{\frac{1}{2}(\|\mathbf{v}_j\|^2 - \|\mathbf{v}_i\|^2) - (\mathbf{v}_j - \mathbf{v}_i) \cdot \mathbf{o}}{(\mathbf{v}_j - \mathbf{v}_i) \cdot \mathbf{d}}$$

**2. Front-facing 判断**
`front` flag 表示这个 face 是 ray 的出口（normal 与 ray direction 夹角 < 90°）。具体来说：

$$\text{front} = \left((\mathbf{v}_j - \mathbf{v}_i) \cdot \mathbf{d} > 0\right) \text{ and } (t_j > 0)$$

这确保我们找到的是 ray 真正离开 cell 的 face，而非 back-facing face（如图 4 中蓝色标记的）。

**3. 无需 acceleration structure**
传统 ray tracing（如 BVH）需要 $O(\log n)$ 的 query。但这个算法直接在 mesh topology 上 "walking"，每个 step 只需遍历当前 cell 的 neighbors。由于 Voronoi cells 是 space-filling 的，ray 必然穿过相邻 cells，因此 walking 是 complete 的。

这是 **Weiler et al. 2003** 算法的核心 insight，被 modern computer vision 遗忘了 20 年。

参考链接：
- Weiler et al. 2003: https://ieeexplore.ieee.org/document/1196223
- Fast Ray Traversal of Tetrahedral Meshes: https://mgarland.org/files/papers/rt-triangle.pdf

---

## 5. Optimization 策略详解

### 5.1 Densification

类似 3DGS，Radiant Foam 需要 adaptive densification。关键 measure：

$$\text{densification\_score}_i = \left\|\frac{\partial \mathcal{L}}{\partial \mathbf{p}_i}\right\| \cdot r_i$$

其中：
- $\frac{\partial \mathcal{L}}{\partial \mathbf{p}_i}$：loss 对第 $i$ 个 Voronoi site 位置的 gradient
- $r_i$：cell $i$ 的 approximate radius
- 乘以 $r_i$ 是为了 normalize 掉 cell 大小的影响

**采样策略**：用 Multinomial distribution 采样，probability mass function 正比于 densification_score。这借鉴了 3DGS-MCMC [Kheradmand et al. 2024] 的思路。

### 5.2 Pruning

pruning 的条件：
1. cell density $\sigma_i$ 很低
2. **且** 所有 neighbors 的 density 也很低

第二个条件很关键！因为 Voronoi cell 的 geometry 由 neighbors 决定。即使某个 cell 的 density 为零，它可能还在 **定义 surface boundary**。如果贸然删除，会破坏相邻 cells 的 geometry。

### 5.3 Quantile Loss

为了减少 "floater" artifacts，作者提出了 quantile loss（公式 6）：

$$\mathcal{L}_{\text{quantile}} = \mathbb{E}_{t_1, t_2 \sim \mathcal{U}[0,1]} \left[|W^{-1}(t_1) - W^{-1}(t_2)|\right]$$

变量解释：
- $W^{-1}(\cdot)$：volume rendering weight distribution 的 quantile function（inverse CDF）
- $t_1, t_2$：从 $[0,1]$ uniform 采样的两个值
- $W^{-1}(t_1)$：表示 "累积 weight 达到 $t_1$ 比例时对应的 ray 参数"

**intuition**：如果 density 集中在 surface 上，weight distribution 应该是 "spiky" 的，即大部分 weight 集中在少数几个 cells。此时 $W^{-1}(t_1)$ 和 $W^{-1}(t_2)$ 会很接近（除非 $t_1, t_2$ 跨越了 spike）。

**与 Mip-NeRF 360 distortion loss 的对比**：
Mip-NeRF 360 的 distortion loss 是：

$$\mathcal{L}_{\text{dist}} = \sum_{i,j} w_i w_j \left|\frac{t_i + t_{i+1}}{2} - \frac{t_j + t_{j+1}}{2}\right|$$

这需要 $O(N^2)$ 的 nested sum。Quantile loss 通过 sampling 避免了 quadratic cost。

### 5.4 总 loss

$$\mathcal{L} = \mathcal{L}_{\text{rgb}} + \lambda \mathcal{L}_{\text{quantile}}$$

其中 $\mathcal{L}_{\text{rgb}}$ 是 standard L2 photometric loss。

参考链接：
- 3DGS-MCMC: https://arxiv.org/abs/2404.09105
- Mip-NeRF 360: https://arxiv.org/abs/2111.12021

---

## 6. Implementation Details

### 6.1 Voronoi Optimization 的工程挑战

**Incremental Delaunay Triangulation**：
每次 primal vertex 位置变化后，需要更新 Delaunay triangulation（从而更新 Voronoi）。完全 rebuild 太慢，因此用 incremental update。

**Rebuild 频率的 trade-off**：
- 早期（1:1 ratio）：每步都 rebuild，保证 mesh 准确
- 后期（1:100 ratio）：每 100 步 rebuild 一次，因为 optimization 收敛后 discrete changes 减少

### 6.2 Training 超参数

- Optimizer：Adam [Kingma 2014]
- Position learning rate：$2 \times 10^{-4} \to 2 \times 10^{-6}$（cosine annealing）
- Density learning rate：$1 \times 10^{-1}$，decay 0.1×
- SH learning rate：$5 \times 10^{-3}$，decay 0.1×
- Density activation：softplus with $\beta = 10$
- SH degree：3（即 16 个 coefficients）
- Total iterations：20k，最后 2k 只优化 radiance/density，冻结 positions
- Bonsai scene 训练时间：70 分钟（RTX 4090）

### 6.3 SH Warmup

前 25% iterations 只优化 zero-order SH（即 DC component），之后才优化 high-order SH。这避免了 high-frequency color 早期干扰 geometry learning。

参考链接：
- Adam Optimizer: https://arxiv.org/abs/1412.6980
- Spherical Harmonics: https://en.wikipedia.org/wiki/Spherical_harmonics

---

## 7. 实验结果深度分析

### 7.1 Table 1：主结果

| Method | Mip-NeRF 360 PSNR↑ | SSIM↑ | LPIPS↓ | FPS↑ | Deep Blending PSNR↑ | SSIM↑ | LPIPS↓ | FPS↑ |
|--------|-------------------|-------|--------|------|---------------------|-------|--------|------|
| 3DGS* | 28.69 | 0.87 | 0.22 | 293 | 29.41 | 0.90 | 0.32 | 319 |
| Mip-Splatting | 29.39 | 0.88 | 0.20 | 241 | 29.47 | 0.90 | 0.32 | 260 |
| 3DGS-MCMC | 29.72 | 0.89 | 0.19 | 302 | 29.71 | 0.90 | 0.32 | 662 |
| MipNeRF360 | 29.23 | 0.84 | 0.21 | <1 | 29.40 | 0.90 | 0.25 | <1 |
| 3DGRT** | 28.71 | 0.85 | 0.25 | 78 | 29.23 | 0.90 | 0.32 | 119 |
| **Radiant Foam** | 28.47 | 0.83 | 0.21 | **200** | 28.95 | 0.89 | 0.26 | **301** |

**关键观察**：

1. **Quality**：Radiant Foam 的 PSNR 比 3DGS 低约 0.2-0.5 dB，比 Mip-NeRF 360 低约 0.7 dB。这是 quality 的 trade-off，但换来 ray tracing 的 flexibility。

2. **Speed**：Radiant Foam 达到 200-301 FPS，远超 3DGRT 的 78-119 FPS。这很 surprising，因为 3DGRT 用了 NVIDIA RTX hardware acceleration。

3. **为什么 Radiant Foam 比 3DGRT 快？**
   - 3DGRT 基于 3DGS 的 overlapping primitives，导致 BVH degradation
   - 3DGRT 每条 ray 需要 multiple intersections with overlapping Gaussians
   - Radiant Foam 的 Voronoi cells 是 **non-overlapping** 的 space partition
   - 无需 secondary acceleration structure，直接 mesh walking

### 7.2 Table 2：Ablation Study

| Config | Bonsai | Garden | Playground | Mean |
|--------|--------|--------|-----------|------|
| Full | 32.15 | 26.58 | 29.59 | 29.15 |
| No SfM | 29.65 | 25.83 | 26.34 | 27.00 |
| No Densify + SfM | 20.23 | 18.88 | 19.55 | 19.36 |
| No Prune | 32.25 | 26.58 | 29.46 | 29.15 |
| No Quantile | 29.62 | 25.35 | 29.59 | 27.90 |

**关键 insight**：
- **Densification 最关键**：去掉后 PSNR 掉 10 dB！这说明 adaptive capacity allocation 是核心。
- **SfM initialization**：去掉掉 2 dB，主要影响 background 和 sparse-view regions。
- **Quantile loss**：去掉掉 1.25 dB，主要引起 floaters。
- **Pruning**：几乎无影响（0.1 dB），因为 prunable points 本来就少。

### 7.3 Per-Scene 分析（Table 3 & 4）

从 per-scene 结果看，Radiant Foam 在 **indoor scenes**（Room, Counter, Bonsai, Kitchen）表现更好，在 **outdoor unbounded scenes**（Bicycle, Garden, Stump）表现稍差。

**Hypothesis**：Outdoor scenes 有大量 background（sky、distant terrain），Voronoi cells 的 piecewise constant 假设可能不够 expressive。而 indoor scenes 的 surfaces 更明确，Voronoi cells 能更好捕捉。

参考链接：
- Mip-NeRF 360 dataset: https://jonbarron.info/mipnerf360/
- Deep Blending dataset: https://hdr-2014.github.io/
- 3DGRT: https://research.nvidia.com/labs/toronto-ai/3DGRT/

---

## 8. 与相关 Work 的深度对比

### 8.1 vs. 3DGS

| Aspect | 3DGS | Radiant Foam |
|--------|------|--------------|
| Primitive | Anisotropic Gaussian | Voronoi cell |
| Rendering | Rasterization | Ray tracing |
| Connectivity | Fixed (from SfM) | Dynamic (Voronoi) |
| Overlapping | Yes | No |
| Camera distortion | Hard | Easy |
| Reflection/Refraction | Hard | Easy |

### 8.2 vs. DMTet / Tet-Splatting

DMTet [Shen et al. 2021] 用 Delaunay triangulation 构造 tetrahedral mesh，但：
- DMTet 优化 vertex positions，不优化 connectivity
- DMTet 用 marching tetrahedra 提取 surface，不是 volume rendering
- Tet-Splatting [Gu et al. 2024] 是 DMTet 的 differentiable rasterizer

Radiant Foam 的优势：
- Connectivity 通过 Voronoi implicit 定义，可 differentiable 优化
- Volume rendering 而非 surface extraction

### 8.3 vs. Tetra-NeRF

Tetra-NeRF [Kulhanek & Sattler 2023] 用 Delaunay tetrahedral mesh + ray marching。但：
- 每条 ray 需要 many MLP evaluations（slow）
- 不优化 connectivity
- 用 barycentric interpolation 而非 piecewise constant

### 8.4 vs. DeRF

DeRF [Rebain et al. 2021] 用 Voronoi decomposition 分解 scene，但：
- 每个 Voronoi cell 内部还是用 NeRF MLP
- 没有利用 Voronoi 的 ray tracing efficiency
- 只是 spatial decomposition，不是 representation 本身

参考链接：
- DMTet: https://research.nvidia.com/labs/toronto-ai/DMTet/
- Tetra-NeRF: https://tetra-nerf.github.io/
- DeRF: https://derf.github.io/

---

## 9. 为什么这个方法 Important？我的 Intuition

### 9.1 "Foam" 的物理直觉

作者用 "closed-cell foam" 作类比非常精妙。物理上的 stable foam（如肥皂泡）满足：
- 每个 bubble 内部 pressure 相等
- Interfaces 是平面（Gauss 的 Plateau laws）
- 结构类似 Voronoi diagram

Radiant Foam 的 Voronoi cells 就是这种 foam 结构，每个 cell "emits" view-dependent radiance。

### 9.2 "Hidden Discontinuities" 的数学直觉

这个方法最深刻的 insight 是：**discrete changes 可以被 hidden 在 zero-measure regions 中**。

更形式化地：设 $f: \mathbb{R}^n \to \mathbb{R}$ 是我们想优化的函数。如果 $f$ 在某个 measure-zero set 上有 discontinuities，但这个 set 的 gradient contribution 为零，那么 gradient descent 仍然可以工作。

Voronoi diagram 的 edge flips 恰好满足这个条件：flip 发生时，affected face 的 area = 0，因此对 ray intersection 的贡献 = 0。

这个 insight 可以推广到其他 representation：
- **Meshes**：如果能找到一种 connectivity parameterization，使得 discrete changes 发生在 zero-contribution regions，就能 differentiable 优化 connectivity。
- **Tessellations**：Power diagrams、Laguerre diagrams 等广义 Voronoi 可能也满足类似性质。

### 9.3 为什么 Ray Tracing 不需要 Hardware Acceleration？

传统 wisdom：ray tracing 需要 BVH + hardware（RTX）才能 real-time。

但 Weiler et al. 的算法打破了这个 wisdom：
- BVH 的 $O(\log n)$ query 是为了 **arbitrary geometry**
- 但对于 **space-filling tessellation**（如 Voronoi），ray 必然穿过相邻 cells
- 因此 mesh walking 是 $O(\text{ray length})$，与 cell 数量无关

这启发我们：**representation 的结构可以 substitute hardware acceleration**。

### 9.4 与 Modern ML Trends 的联系

Radiant Foam 的设计哲学与几个 modern ML trends 呼应：

1. **Sparse mixture of experts**：Voronoi cells 类似 experts，每个 cell 负责 space 的一个 region。Densification 就是 expert allocation。

2. **Continuous embeddings of discrete structures**：Voronoi diagram 是 discrete structure，但通过 primal vertex positions 实现 continuous parameterization。类似 Gumbel-Softmax、straight-through estimators。

3. **Differentiable physics**：Foam 的物理结构（Plateau laws）与 Voronoi 的数学结构对应，这暗示了 differentiable simulation 的可能性。

参考链接：
- Plateau's Laws: https://en.wikipedia.org/wiki/Plateau%27s_laws
- Differentiable Physics: https://geomtech.github.io/diffphys/
- Mixture of Experts: https://arxiv.org/abs/1701.06538

---

## 10. Limitations 与 Future Work

### 10.1 论文承认的 limitations

1. **Voronoi 限制**：cell boundary 必须是两个 sites 的 perpendicular bisector，导致很多 small empty cells 用来定义 surface。
2. **Composition**：如何 compose 多个 foam models？
3. **Dynamic content**：目前只支持 static scenes。
4. **Illumination**：没有 relighting 能力。
5. **Generative modeling**：如何与 diffusion models 结合？

### 10.2 我推测的 additional limitations

1. **Memory overhead**：Voronoi cells 比 Gaussians 更多（因为需要 empty cells 定义 boundary），可能 memory footprint 更大。

2. **Background modeling**：unbounded scenes 的 background（sky）需要 infinite cells，或特殊处理。Table 3 显示 outdoor scenes 表现差，可能与此有关。

3. **Sharp features**：piecewise constant 假设可能难以捕捉 sharp edges，除非 densification 非常 aggressive。

4. **Training stability**：incremental Delaunay 可能 numerically unstable（论文提到 "very close points" 会导致 failure）。

### 10.3 Future Directions 的联想

1. **Power Diagrams**：generalize Voronoi 到 weighted sites，允许 cell size 与 density 解耦。

2. **Adaptive refinement**：类似 finite element methods 的 h-refinement，在 surface 附近 adaptively refine cells。

3. **Neural foam**：每个 cell 内部用 small MLP 而非 constant，增加 expressiveness。

4. **Differentiable rendering of effects**：
   - Reflection：ray bouncing 在 mesh walking 中自然实现
   - Refraction：Snell's law 在 cell boundary 处 apply
   - Caustics：bidirectional ray tracing

5. **Dynamic foam**：每个 cell 有 velocity，用 differentiable simulation 演化 foam 结构。

6. **Generative foam**：用 diffusion model 生成 Voronoi site positions + per-cell attributes。

7. **Compression**：Voronoi 的 regularity 可能允许更好的 compression（类似 mesh compression）。

8. **Streaming**：类似 SMERF [Duckworth et al. 2024]，用 Voronoi 的 spatial partition 实现 streaming rendering。

参考链接：
- Power Diagrams: https://en.wikipedia.org/wiki/Power_diagram
- SMERF: https://smerf-3d.github.io/
- Differentiable Simulation: https://arxiv.org/abs/2104.10539

---

## 11. 公式补充：Voronoi Cell 的 Geometry 计算

### 11.1 Cell Volume

Voronoi cell $\mathbf{c}_i$ 的 volume 可以通过 Delaunay triangulation 计算。每个 Delaunay tetrahedron $T_k$ 对应一个 Voronoi vertex（circumcenter）$\mathbf{v}_k$。Cell $\mathbf{c}_i$ 的 volume 是：

$$V_i = \sum_{T_k \ni \mathbf{p}_i} \text{Volume}(\text{pyramid from } \mathbf{p}_i \text{ to face opposite } \mathbf{p}_i \text{ in } T_k)$$

### 11.2 Cell Face Area

Cell $\mathbf{c}_i$ 和 $\mathbf{c}_j$ 之间的 face area：

$$A_{ij} = \text{Area}(\text{polygon formed by circumcenters of tetrahedra containing edge } (\mathbf{p}_i, \mathbf{p}_j))$$

### 11.3 Gradient Flow

当 $\mathbf{p}_i$ 移动时，所有含 $\mathbf{p}_i$ 的 Delaunay tetrahedra 的 circumcenters 移动，从而影响 cell geometry。Gradient 通过 ray intersection points $\delta_n$ 流回 $\mathbf{p}_i$。

具体来说，ray 与 face $(i,j)$ 的交点 $t_{ij}$ 对 $\mathbf{p}_i$ 的 gradient：

$$\frac{\partial t_{ij}}{\partial \mathbf{p}_i} = \frac{\partial}{\partial \mathbf{p}_i} \frac{\frac{1}{2}(\|\mathbf{v}_j\|^2 - \|\mathbf{v}_i\|^2) - (\mathbf{v}_j - \mathbf{v}_i) \cdot \mathbf{o}}{(\mathbf{v}_j - \mathbf{v}_i) \cdot \mathbf{d}}$$

这个 gradient 是 well-defined 的，除了 degenerate cases（ray 平行于 face）。

---

## 12. 总结：Radiant Foam 的核心贡献

1. **Representation**：Voronoi diagram 作为 differentiable volumetric mesh，解决了 connectivity optimization 的 continuity 问题。

2. **Rendering**：Revive 了 Weiler et al. 2003 的 mesh ray tracing，证明无需 hardware acceleration 也能 real-time。

3. **Optimization**：Densification + pruning + quantile loss 的组合，使 Voronoi sites 能 adaptive 分配。

4. **Applications**：Ray-based rendering 天然支持 reflection、refraction、camera distortion 等 effects。

5. **Philosophical contribution**：证明 representation 的结构可以 substitute hardware acceleration，为 differentiable rendering 开辟新方向。

---

## 13. 个人思考与 Open Questions

1. **Voronoi vs. Power Diagram**：如果用 Power Diagram（weighted Voronoi），能否减少 empty cells？weights 可以作为 additional optimizable parameters。

2. **Hierarchical foam**：类似 hierarchical 3DGS [Kerbl et al. 2024]，能否构造 hierarchical Voronoi？Quadtree-like structure 可能实现 LOD。

3. **Foam + Neural fields**：每个 cell 内部用 small MLP（而非 constant）能否提升 quality？类似 Tri-MipRF 的思路。

4. **Differentiable Delaunay**：能否完全 differentiable 地构造 Delaunay（而非 incremental rebuild）？这可能是 training speedup 的关键。

5. **Foam for physics simulation**：Voronoi cells 天然适合 FEM-like simulation。能否用 Radiant Foam 做 differentiable physics？

6. **Connection to optimal transport**：Voronoi diagram 与 optimal transport 有深刻联系。能否用 OT 的工具优化 foam？

7. **Generative foam**：能否训练 diffusion model 直接生成 foam parameters？类似 3DGS 的 generative works。

参考链接：
- Hierarchical 3DGS: https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/
- Tri-MipRF: https://arxiv.org/abs/2304.03607
- Optimal Transport: https://arxiv.org/abs/1803.00567

---

这篇论文的 core insight 是 **representation structure can substitute hardware acceleration**，并通过 Voronoi diagram 的 beautiful mathematical property（discrete changes hidden in zero-measure regions）实现了 differentiable ray tracing。虽然 quality 略逊于 3DGS，但它开辟了 ray-based differentiable rendering 的新范式，未来 work 有很大空间。
