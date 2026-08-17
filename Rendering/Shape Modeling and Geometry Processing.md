---
source_pdf: Shape Modeling and Geometry Processing.pdf
paper_sha256: 7a4f3cde7ea4999da0e6853750a38b4a889fa9d1fb31c9cebaec292f892de705
processed_at: '2026-08-12T05:31:56-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 Remeshing & Smoothing

好，我把前面那篇 paper 用大白话再过一遍，同时把里面的数学和直觉揉在一起讲。

---

## 一、Remeshing 到底在干嘛

你有一个 mesh，可能是扫描仪扫出来的，可能是 marching cubes 重建的，也可能是美术在 Blender 里随手建的。这个 mesh 长得歪七扭八：有的三角形细得像针，有的 vertex 邻居数是 3，有的 vertex 邻居数是 30。

**Remeshing 就是把同一个 3D 形状，换一套更整齐的三角形来表示。**

就像你写代码，同样的功能可以写得很乱也可以写得很整洁。Remeshing 就是给 mesh 做 "refactor"。

### 什么样的 mesh 算"好"

Paper 里列了一堆标准，核心就几条：

1. **边长尽量一样** — 不要有的边特别长有的特别短
2. **三角形尽量等边** — 等边三角形是各向同性的，没有偏好方向
3. **Valence 尽量是 6** — 每个 vertex 周围最好是 6 个邻居
4. **高曲率地方多采样，低曲率地方少采样** — adaptive
5. **Sharp edge 要保留** — 比如桌子的棱角不能被磨圆
6. **各向异性时边要沿主曲率方向对齐**

#### 为什么 valence 6 这么特殊？

这个用 Euler formula 一算就明白了。对一个大 triangulated surface：

$$V - E + F = \chi = 2 - 2g$$

其中 $V$ 是 vertex 数，$E$ 是 edge 数，$F$ 是 face 数，$g$ 是 genus（亏格，洞的数量）。

每个三角形有 3 条边，每条边被 2 个三角形共享，所以 $3F = 2E$。
每个 edge 连 2 个 vertex，所以 $2E = V \cdot \bar{v}$，其中 $\bar{v}$ 是平均 valence。

把这三个关系一凑：

$$\bar{v} = \frac{2E}{V} = \frac{6V - 6\chi}{V} = 6 - \frac{6\chi}{V}$$

当 mesh 越来越大（$V \to \infty$），$\bar{v} \to 6$。

**所以 6 是大自然规定的最优 valence。** 偏离 6 就是"不自然"，paper 里用 $\sum_i (\text{valence}(v_i) - 6)^2$ 来衡量 mesh 有多"不自然"。

Intuition：valence 6 的 vertex 在局部看起来就像一个正六边形铺砖，这是最均匀的 2D 连接方式。

参考：https://www.pmp-book.org/ Chapter 6

---

## 二、两条技术路线

### 路线 A：Parameterization-Based

**核心想法：把 3D 曲面摊平到 2D，在 2D 上做事情，再 lift 回 3D。**

Pipeline：
1. 找一个 conformal mapping 把 mesh 摊到 2D
2. 在 2D 上 sampling（决定放多少点、放哪里）
3. 在 2D 上 triangulate（Delaunay 之类）
4. Lift 回 3D

**Conformal** 的意思是保角——3D 上的小圆映射到 2D 还是小圆，只是大小变了，形状没变。所以只引入 area distortion，不引入 angle distortion。

#### 采样怎么放？——CVT (Centroidal Voronoi Tessellation)

在 2D 上随便撒点不行，点之间距离会不均匀。Paper 用了一个能量：

$$E(x_1, \dots, x_k, R_1, \dots, R_k) = \sum_{i=1}^{k} \int_{R_i} \|x - x_i\|^2 \, dx$$

变量解释：
- $x_i$：第 $i$ 个采样点的位置（site）
- $R_i$：第 $i$ 个 site "管辖" 的区域（Voronoi cell）
- $k$：采样点总数
- $\|x - x_i\|^2$：点 $x$ 到 site $x_i$ 的距离平方

这个能量的意思就是：**整个平面上每个点都指派给最近的 site，然后惩罚它离 site 的距离。** 这其实就是 k-means 的 objective function。

两个条件同时满足时能量最小：
- 区域 $R_i$ 是 site $x_i$ 的 Voronoi cell
- site $x_i$ 是区域 $R_i$ 的重心（centroid）

这叫 **Centroidal Voronoi Tessellation**。

#### Lloyd 算法

```
重复：
  1. 算当前所有 site 的 Voronoi diagram
  2. 把每个 site 移到它 Voronoi cell 的重心
直到收敛
```

Lloyd 收敛很慢（线性收敛）。Liu et al. TOG 2009 用 L-BFGS 直接优化能量，快了 100 倍：
https://dl.acm.org/doi/10.1145/1516522.1516536

#### 加 density 做自适应

$$E = \sum_{i=1}^{k} \int_{R_i} \rho(x) \|x - x_i\|^2 \, dx$$

$\rho(x)$ 是密度函数。$\rho$ 大的地方采样就密。

怎么定 $\rho$？用 conformal map 的 area distortion：3D 上面积小的地方在 2D 上被放大了，所以 2D 上要稀疏采样；反之 2D 上要密集采样。

#### Parameterization 路线的坑

- 闭合 mesh 必须先 cut open，cut graph 的选择是 NP-hard 级问题
- Cut 完 lift 回去时 seam 处两边三角形接不上，要 stitching
- 像人体手臂腿这种细长突出部分，parameterization distortion 极大，数值上 ill-conditioned
- Boundary 处理麻烦

参考 Alliez et al. 2003: https://hal.inria.fr/inria-00302666/

---

### 路线 B：Surface-Oriented (Botsch et al. 2004)

**核心想法：不做 parameterization，直接在 3D mesh 上用局部操作改来改去。**

四个基本局部操作：

1. **Edge Collapse** — 把一条边的两个 vertex 合并成一个，少一个三角形
2. **Edge Split** — 在边中点插入新 vertex，多一个三角形
3. **Edge Flip** — 把两个三角形共享的边翻一下，vertex 数和 face 数不变，只改 connectivity
4. **Vertex Shift** — 把 vertex 在 tangent plane 上挪一挪

每个操作只影响局部 1-ring，$O(1)$ 的。100k 三角形 5 秒搞定。

参考：https://cg.cs.uni-bielefeld.de/publications/paper/2004_Polygon-Remeshing.pdf

#### 核心算法

给定目标边长 $L$，算出：
$$L_{\max} = \frac{4}{3} L, \quad L_{\min} = \frac{4}{5} L$$

然后循环：

```
1. 把所有比 L_max 长的 edge 从中点 split
2. 把所有比 L_min 短的 edge collapse 掉
3. Flip edges 让 valence 趋近 6
4. Tangential Laplacian smoothing
5. 把 vertex 投影回原始 reference mesh
```

#### 为什么是 4/3 和 4/5？

这个数字不是拍脑袋的。Paper 里有推导，核心想法是让 split 和 collapse 的"偏差"对称，不会震荡：

Split 一条长 $L_{\max}$ 的边，变成两条 $L_{\max}/2$ 的边。我们希望这两条新边"偏离 $L$ 的程度"和原来那条长边"偏离 $L$ 的程度"一样：

$$|L_{\max} - L| = \left|\frac{L_{\max}}{2} - L\right|$$

解出来 $L_{\max} = \frac{4}{3} L$。

Collapse 一条短边 $L_{\min}$，合并后新边大约 $\frac{3}{2} L_{\min}$（假设相邻边也是 $L_{\min}$ 级别）。同理要求偏差对称：

$$|L_{\min} - L| = \left|\frac{3}{2} L_{\min} - L\right|$$

解出来 $L_{\min} = \frac{4}{5} L$。

**Intuition：如果阈值不对称，split 完立刻被 collapse，collapse 完立刻被 split，死循环。对称阈值保证算法稳定收敛。**

#### Edge Flip 怎么决定翻不翻

看 flip 前后 valence excess 的变化：

$$\text{cost} = \sum_{i=1}^{4} (\text{valence}(v_i) - \text{opt}(v_i))^2$$

- interior vertex 最优 valence = 6
- boundary vertex 最优 valence = 4

flip 一下，两个 vertex valence +1，另两个 valence -1。如果 flip 后总 cost 降低了，就 flip。

Intuition：这是在最小化 valence 的方差，让所有 vertex valence 都尽量接近 6。

#### Vertex Shift (Tangential Relaxation)

$$\mathbf{c}_i = \frac{1}{\text{valence}(v_i)} \sum_{j \in \mathcal{N}(v_i)} \mathbf{p}_j$$

就是把 vertex 挪到它所有邻居的重心位置。但**只在 tangent plane 内挪**，不能沿 normal 挪（否则形状会变形）。挪完之后第 5 步投影回 reference mesh 保持几何。

#### Adaptive 版本

高曲率地方边要短，低曲率地方边可以长：

$$L_{\text{local}} = \min\left(L, \frac{c}{\kappa_{\max}}\right)$$

$\kappa_{\max}$ 是局部最大主曲率，$c$ 是用户参数（比如 0.5）。

**Intuition：曲率半径 $r = 1/\kappa_{\max}$ 是局部几何的"自然尺度"。边长应该和曲率半径同量级。边比曲率半径长很多，离散就失真；边比曲率半径短很多，浪费。**

---

## 三、Smoothing 在干嘛

Mesh 上有噪声——扫描仪扫出来有 jitter，marching cubes 重建有 staircase。噪声是高频，几何是低频。Smoothing 就是做低通滤波。

### 3.1 Smoothness 和 Curvature 的关系

光滑的曲面曲率变化平缓，粗糙的曲面曲率变化剧烈。所以 smoothing 本质上就是在减小曲率或者让曲率变化更平缓。

但用哪个曲率？

| 曲率 | 问题 |
|---|---|
| Principal curvatures $\kappa_{\min}, \kappa_{\max}$ | min/max operator 非线性，不连续 |
| Gauss curvature $K = \kappa_{\min} \kappa_{\max}$ | intrinsic，对 cylinder 这种 developable surface 是 0，区分不了 |
| Mean curvature $H = \frac{\kappa_{\min} + \kappa_{\max}}{2}$ | 线性，而且可以用 Laplace-Beltrami 直接算 |

**Mean curvature 最好用**，因为有这个漂亮公式：

$$\Delta_{\mathcal{M}} \mathbf{p} = -2 H \mathbf{n}$$

变量解释：
- $\Delta_{\mathcal{M}}$：Laplace-Beltrami operator（曲面上的 Laplacian）
- $\mathbf{p}$：曲面上点的位置向量
- $H$：mean curvature
- $\mathbf{n}$：surface normal

**Intuition：在曲面上对位置场做 Laplacian，得到的是一个向量，方向沿 normal，大小是 $2H$。** 这就把曲率和 Laplacian 直接联系起来了。

### 3.2 从 1D 曲线开始直觉

1D 曲线上 Laplacian 就是二阶导：

$$L(\mathbf{p}_i) = \frac{1}{2}(\mathbf{p}_{i-1} - \mathbf{p}_i) + \frac{1}{2}(\mathbf{p}_{i+1} - \mathbf{p}_i)$$

就是把当前点挪到左右邻居的中点。

Smoothing flow：

$$\tilde{\mathbf{p}}_i = \mathbf{p}_i + \lambda L(\mathbf{p}_i)$$

$\lambda \in (0, 1)$ 是步长。

Paper 里展示 $\lambda = 0.5$ 迭代 1000、5000、10000、50000 步：曲线越来越缩，最后缩成一个点。

**这就是 explicit mean curvature flow 的致命问题：体积不守恒。**

### 3.3 Mesh 上的 Mean Curvature Flow

连续 PDE：

$$\frac{\partial \mathbf{p}}{\partial t} = \lambda \Delta_{\mathcal{M}} \mathbf{p} = -2\lambda H \mathbf{n}$$

**Intuition：曲面上每个点沿 normal 方向移动，速度正比于 mean curvature。曲率大的地方移动快，弯的地方被拉直。** 这就是 mean curvature flow，等价于曲面上的 heat equation。

#### Forward Euler（显式）

$$\mathbf{p}^{(n+1)} = (\mathbf{I} + dt \cdot \lambda \mathbf{L}) \mathbf{p}^{(n)}$$

问题：$dt$ 大了就爆。CFL 条件限制 $dt < \frac{1}{2\lambda \cdot \text{valence}_{\max}}$。

### 3.4 Taubin 的 $\lambda|\mu$ Smoothing (SIGGRAPH 1995)

Taubin 的洞察：**能不能只滤掉高频，保留低频？**

策略：交替做 smooth 和 inflate

$$\tilde{\mathbf{p}} = (\mathbf{I} + \lambda \mathbf{L}) \mathbf{p}, \quad \lambda > 0$$
$$\tilde{\tilde{\mathbf{p}}} = (\mathbf{I} + \mu \mathbf{L}) \tilde{\mathbf{p}}, \quad \mu < 0$$

第一步 $\lambda > 0$ 是正常 smooth（低通 + 收缩），第二步 $\mu < 0$ 是"反 Laplacian"（膨胀回来）。

**Intuition：Laplacian 的每个 eigenvector 对应一个频率 $\omega_k$。两步组合后每个 mode 乘以 $(1 + \lambda \omega_k)(1 + \mu \omega_k)$，这是一个双零点 low-pass filter。** 高频被压掉，低频保留。

Taubin 推荐 $\lambda = 0.6307$，$\mu = -0.6732$，来自 signal processing 的 filter design。

缺点：要迭代几百次，慢。

参考：https://dl.acm.org/doi/10.1145/218380.218449

### 3.5 Implicit Fairing (Desbrun et al. 1999)

**用 backward Euler 代替 forward Euler：**

$$\frac{\mathbf{p}^{(n+1)} - \mathbf{p}^{(n)}}{dt} = \lambda \mathbf{L} \mathbf{p}^{(n+1)}$$

整理：

$$(\mathbf{I} - dt \cdot \lambda \mathbf{L}) \mathbf{p}^{(n+1)} = \mathbf{p}^{(n)}$$

每步要解一个 sparse linear system（用 PCG 之类）。

**为什么 implicit 稳定？**

- 显式：高频模式放大倍数 $|1 - dt \cdot \omega_k|$，$dt$ 大了就 $>1$ 爆炸
- 隐式：高频模式放大倍数 $\left|\frac{1}{1 + dt \cdot \omega_k}\right| < 1$，永远衰减

所以 **$dt$ 可以随便大，无条件稳定。** 一步就能做大量 smoothing。

Paper 里 Figure 4 的实验：
- Explicit $\lambda dt = 1$，10 步：几乎没效果（CFL 限制）
- Implicit $\lambda dt = 10$，1 步：7 次 PCG iteration，比 explicit 快 30%，效果还好
- Taubin $\lambda|\mu$，20 步：慢，效果一般

参考：https://dl.acm.org/doi/10.1145/311535.311576

### 3.6 关键：权重选择

Paper 里有个对比图非常重要：**uniform Laplacian 的 smoothing 结果依赖 mesh 密度。**

即使 underlying surface 对称，如果 mesh 一边密一边稀，uniform Laplacian smoothing 会给出不对称结果。

为什么？因为 uniform Laplacian $\frac{1}{\text{valence}} \sum (\mathbf{p}_j - \mathbf{p}_i)$ 只看 connectivity，不看几何。

**正确做法是用 cotangent weights：**

$$w_{ij} = \frac{1}{2}(\cot \alpha_{ij} + \cot \beta_{ij})$$

- $\alpha_{ij}$：edge $(v_i, v_j)$ 对面一个 vertex 处的角
- $\beta_{ij}$：edge $(v_i, v_j)$ 对面另一个 vertex 处的角

**Intuition：cotangent weights 来自 Poisson equation 在 triangle mesh 上的 finite element 离散化，是真正的 geometric Laplacian。** 它考虑了三角形的形状，所以 mesh 密度变化时仍然给出同一结果。

Paper 里展示：uniform Laplacian 在不对称 mesh 上给不对称结果，cotangent Laplacian 给对称结果。

参考 Meyer et al. 2002: https://dl.acm.org/doi/10.1007/978-3-662-05189-4_2

---

## 四、一句话总结直觉

**Remeshing** 就是给 mesh 做 refactor——同样的形状换一套更整齐的三角形来表示，要么在 2D 参数域上做（precise 但 fragile），要么直接在 3D 上用局部操作（fast 且 robust）。

**Smoothing** 就是把 mesh 当信号做低通滤波——用 Laplace-Beltrami 这个曲面上的二阶导算子，让高频噪声衰减，核心 trade-off 是 explicit（快但不稳）vs implicit（要解线性系统但无条件稳定），以及 uniform weights（依赖 mesh）vs cotangent weights（mesh-independent）。

两者本质上都是 **用 PDE 视角看离散几何**，这是几何处理的核心哲学，也是后来 geometric deep learning 的祖先。

---

# Shape Modeling and Geometry Processing — Remeshing & Smoothing 深入解析

这篇 paper 是 ETH Zürich 的 Roi Poranne 与 Bielefeld 大学 Mario Botsch 风格 lecture notes 的合集，本质上对应 Botsch 等人的经典著作 *Polygon Mesh Processing* (2010, AK Peters) 的核心章节。它讨论的是几何处理中最基础也最重要的两个问题：**Remeshing** (改善离散表示的质量) 和 **Smoothing / Fairing** (去噪、平滑几何)。下面我从直觉到公式逐层展开。

参考链接：
- Mario Botsch 的主页: https://cg.cs.uni-bielefeld.de/people/botsch/
- *Polygon Mesh Processing* 配套资源: https://www.pmp-book.org/
- Roi Poranne (现为 CMU): https://www.cs.cmu.edu/~rporanne/
- Botsch et al. 2004 Polygon Remeshing: https://cg.cs.uni-bielefeld.de/publications/paper/2004_Polygon-Remeshing.pdf
- Alliez et al. 2003 Anisotropic Remeshing: https://hal.inria.fr/inria-00302666/
- Liu et al. 2009 CVT (TOG): https://dl.acm.org/doi/10.1145/1516522.1516536
- Taubin 1995 (SIGGRAPH): https://dl.acm.org/doi/10.1145/218380.218449
- Desbrun et al. 1999 Implicit Fairing: https://dl.acm.org/doi/10.1145/311535.311576
- Gabriel Taubin HPQM course: https://hpqm.cs.brown.edu/taubin.html

---

## Part I — Remeshing: 哲学与目标

### 1.1 为什么需要 remeshing？

输入的 mesh 来源五花八门：3D 扫描仪、marching cubes 重建、CAD tessellation、艺术家手工建模的 mesh 等。这些 mesh 经常携带**离散质量缺陷**：sliver triangles (细长三角)、valence 极端的 vertex (例如 valence 30)、不规则连接、孤立边界等。

**离散几何质量直接决定后续所有几何处理算法的行为**：
- 求解 PDE (e.g. diffusion equation on mesh) 时，element quality 影响 condition number；
- 物理仿真时 sliver triangles 导致 stiffness matrix 病态；
- Parameterization 时 poor valence 会引入 distortion；
- 渲染时某些 shading artifacts 与 triangle size 强相关。

直觉上，我们可以把 mesh 看作对隐式 underlying surface $\mathcal{S}$ 的**采样+连接图**。Remeshing 就是重新选 sample 位置并重连，使得在 sample 数不变甚至变少的情况下，离散表示更"忠实"。

### 1.2 什么是"好 mesh"？——六大判据

Paper 列出的判据：

1. **Equal edge lengths** — 所有边长尽量相同
2. **Equilateral triangles** — 三角形接近等边
3. **Valence close to 6** — 每个 interior vertex 邻居数为 6
4. **Uniform vs. adaptive sampling** — 在平坦区域少采样，高曲率区域多采样
5. **Feature preservation** — sharp edges、material boundaries 保留
6. **Alignment to curvature lines** — anisotropic remeshing 中边的方向沿主曲率方向
7. **Isotropic vs. anisotropic** — 元素各向同性 / 各向异性

#### Intuition: 为什么 valence = 6 是"神圣"的数字？

考虑 Euler formula: $V - E + F = \chi$，其中 $\chi = 2 - 2g$ 是 Euler characteristic。对一个 triangulation，有 $3F = 2E$ (每 face 三条边但每边被两 face 共享)。平均 valence 满足：

$$\bar{v} = \frac{2E}{V} = \frac{6V - 6\chi}{V} = 6 - \frac{6\chi}{V} \to 6 \text{ (当 } V \to \infty\text{)}$$

对一个 genus 0 大三角网格，平均 valence 严格趋近于 6。所以 6 是"自然界钦定"的最优 valence。Vertex valence 偏离 6 越多，total excess $\sum (v_i - 6)^2$ 越大，mesh 质量越差。这就是后面 edge flip 的优化目标来源。

#### Intuition: 为什么 equilateral？

Equilateral triangle 是 isotropic 的——没有 preferred direction。各向同性意味着 mesh 上所有方向都能均匀表达几何信号，处理时不会出现 direction-dependent artifacts。这对接下来要做的 parameterization、simulation、subdivision 等都至关重要。

### 1.3 Local Structure 分类轴

- **Element type**: triangle / quad / polygon
- **Element shape**: isotropic vs anisotropic
- **Element distribution**: sizing (uniform vs adaptive) + grading (size transitions 平缓还是突变)
- **Element orientation**: 各向异性情况下是否对齐 principal curvature directions

Quad mesh 之所以被 animation 偏爱，是因为 quad 形成**regular grid**，便于 skeleton binding 和 loop subdivision；triangle mesh 之所以通用，是因为任意 topology 都能 triangulate。

---

## Part II — Remeshing 的两大路线

### 2.1 总体哲学分歧

| 路线 | 核心想法 | 优点 | 缺点 |
|---|---|---|---|
| Parameterization-Based | 把曲面解开到 2D，在 2D 做 sampling / triangulation，再 lift 回 3D | 2D 算法成熟、容易引入 anisotropy、容易做 coarse remeshing | 闭曲面要 cut，seam 处有 stitching 问题，数值敏感 |
| Surface-Oriented | 直接在 3D 曲面上用 local operators 操作 | 不需要 global cut，速度快，robust | 难引入 anisotropy，对参数化-依赖特性较弱 |

直觉上，2D 路线是把 3D 问题的"几何"和"组合"分离；surface-oriented 则是直接在流形上"修炼"。

### 2.2 Parameterization-Based Remeshing [Alliez et al. '03]

pipeline：
1. Compute a **conformal** parameterization $\phi: \mathcal{S} \to \Omega \subset \mathbb{R}^2$
   - Conformal 意味着局部保角，只引入 area distortion 而不引入 angle distortion
   - 等价于 metric 满足 Cauchy-Riemann：在 surface 上 local 的圆映射到 2D 后仍是圆
2. 在 2D domain $\Omega$ 上做 **density-based sampling**，density 取决于 area distortion (Jacobian $|\partial \phi / \partial u|$)
   - intuition: 3D 上 unit area，2D 上对应 area 大 → 2D 上要稀疏采样；反之稠密
3. 在 2D 上 triangulate (e.g. Delaunay)
4. Lift 回 3D surface

#### Sampling 中的 Centroidal Voronoi Tessellation (CVT)

直接 random sampling 在 2D 会导致点间距不均匀。Paper 提出最小化 **quantization energy**：

$$E(x_1, \dots, x_k, R_1, \dots, R_k) = \sum_{i=1}^{k} \int_{R_i} \| x - x_i \|^2 \, dx$$

公式变量解释：
- $x_i \in \mathbb{R}^2$: 第 $i$ 个 site (sample) 的位置
- $R_i$: 第 $i$ 个 site 对应的区域 (Voronoi cell)
- $k$: site 总数
- $\|x - x_i\|^2$: 点 $x$ 到 site $x_i$ 的 squared Euclidean distance

直觉：对每个点 $x$，把它"指派"给最近的 site $x_i$，并惩罚其与 site 的距离平方。这就是经典的 **k-means** objective！

**两阶段优化**：
- 若 $\{x_i\}$ 固定，$R_i$ 应该取 **Voronoi cell** $\{x : \|x - x_i\| \le \|x - x_j\|, \forall j\}$
- 若 $\{R_i\}$ 固定，$x_i$ 应取 $R_i$ 的 **centroid** (mass center)：
$$x_i = \frac{\int_{R_i} x \, \rho(x) \, dx}{\int_{R_i} \rho(x) \, dx}$$

两个条件同时满足 → **Centroidal Voronoi Tessellation**。

#### Lloyd's Algorithm

```
repeat:
    1. Compute Voronoi diagram for current {x_i}
    2. Move each x_i to centroid of its Voronoi cell
until convergence
```

Lloyd 收敛慢 (线性 convergence)。Paper 引用了 Liu et al. TOG '09 的工作，用 **quasi-Newton** (L-BFGS) 直接优化 CVT energy，得到 100x 速度提升。

#### Varying Density: Anisotropic / Adaptive CVT

加入 density function $\rho(x)$：

$$E = \sum_{i=1}^{k} \int_{R_i} \rho(x) \|x - x_i\|^2 \, dx$$

- 高 $\rho$ 区域 → 那里 sampling 密度更大 (因为惩罚权重更大)
- $\rho(x)$ 取决于 parameterization 的 area distortion: $\rho \propto 1/J(x)$

这样 uniform density in 3D ↔ varying density in 2D, 这是 conformal map 的核心 trade-off。

### 2.3 Parameterization-based 的 limitations

paper 列出的麻烦：

1. **Closed meshes**: 必须 cut open → 选择 cut graph 是个 NP-hard 级问题
2. **Free boundary**: 若允许 boundary 自由变形，2D domain 形状不确定
3. **Protruding parts**: 像人体手臂腿这种拓扑，parameterization distortion 极大
4. **Seams**: cut 之后 lift 回去时 seam 处两边的 triangulation 不连续，需要 stitching
5. **Numerical problems**: 比如腿伸出的区域，parameterization 矩阵严重 ill-conditioned

为了避开这些问题，转向 surface-oriented 方法。

### 2.4 Direct Surface Remeshing [Botsch et al. '04]

key insight: 不做 global parameterization，也不做 local parameterization (太贵)，而是用 **local operators + projections to reference mesh**。100k triangles 处理只要 <5s。

#### 四个 local operators

1. **Edge Collapse**: 把两个 vertex 合并成一个，减少 triangle 数量
2. **Edge Split**: 在 edge 中点插入新 vertex，增加 triangle 数量
3. **Edge Flip**: 翻转共享边的两个三角形（保持 vertex 数 + face 数不变，但改 connectivity）
4. **Vertex Shift**: 移动 vertex 位置到 tangent plane 上一个更好的位置

每个 operator 都是 $O(1)$ 操作 (只影响局部 1-ring)。

#### Isotropic Remeshing Algorithm (核心算法)

输入 target edge length $L$，计算 $L_{\min} = \frac{4}{5} L$, $L_{\max} = \frac{4}{3} L$。

**为什么 4/3 和 4/5？**

直觉：我们要保证 split 和 collapse 是**互补不冲突**的。若把 $L_{\max}$ 设成 $L$，split 之后两条新边长度都是 $L_{\max}/2$，立刻小于 $L_{\min}$ → 又被 collapse → 死循环。

对 split 后两段新边各为 $L_{\max}/2$，必须满足 $L_{\max}/2 \ge L_{\min}$，即 $L_{\max} \ge 2 L_{\min}$。

对 collapse 后新边长约为 $L_{\min} + L_{\text{neighbor}} \approx \frac{3}{2} L_{\min}$ (若两个相邻边都接近 $L_{\min}$)，必须满足 $\frac{3}{2} L_{\min} \le L_{\max}$，即 $L_{\min} \le \frac{2}{3} L_{\max}$。

Paper 给出的对称推导：要求 split 和 collapse 的"误差对称":

$$|L_{\max} - L| = |\tfrac{1}{2} L_{\max} - L| \Rightarrow L_{\max} = \tfrac{4}{3} L$$
$$|L_{\min} - L| = |\tfrac{3}{2} L_{\min} - L| \Rightarrow L_{\min} = \tfrac{4}{5} L$$

详细解释：
- 左式 $|L_{\max} - L|$: 一条长 $L_{\max}$ 的边超过目标的距离
- 右式 $|\frac{1}{2}L_{\max} - L|$: split 后两条 $L_{\max}/2$ 的边距目标的距离
- 令两者相等 → split 与不 split 是同样的"坏程度" → 不会震荡
- 同理 collapse: $L_{\min}$ 与 collapse 后大约 $\frac{3}{2} L_{\min}$ 的边，距 $L$ 的偏差相等

**Algorithm iterate:**

```
1. Split edges longer than L_max at midpoint
2. Collapse edges shorter than L_min (collapse to midpoint or 1/3-2/3)
3. Flip edges to reduce valence excess
4. Tangential Laplacian smoothing (vertex shift)
5. Project vertices back to original reference mesh
```

第 5 步**关键**：因为 smoothing 会让 vertex 偏离原始 surface，必须 projection 回去。这意味着保留原 mesh 作为 reference，新 vertex 通过 ray casting 或 closest point query 投影回 reference。这就避开了 parameterization 但仍然保持 surface fidelity。

#### Edge Flip 优化目标

```
flip if sum_i (valence(v_i) - optimal_valence(v_i))^2 decreases
```

- interior vertex optimal valence = 6
- boundary vertex optimal valence = 4

直觉：每 flip 一次，两个 vertex 的 valence +1，另两个 vertex 的 valence -1。我们要让 valence 越接近 optimal 越好。

$$\text{cost} = \sum_{i=1}^{4} (\text{valence}(v_i) - \text{opt\_valence}(v_i))^2$$

比较 flip 前 flip 后的 cost，若减少则 flip。

#### Vertex Shift (Tangential Relaxation)

Uniform Laplacian smoothing:

$$\mathbf{c}_i = \frac{1}{\text{valence}(v_i)} \sum_{j \in \mathcal{N}(v_i)} \mathbf{p}_j$$

公式变量：
- $\mathbf{c}_i$: vertex $v_i$ 的 1-ring 邻居的**重心**
- $\mathcal{N}(v_i)$: $v_i$ 的 1-ring 邻居集合
- $\mathbf{p}_j$: 邻居 vertex 的位置
- valence$(v_i) = |\mathcal{N}(v_i)|$

移动 $v_i$ 到 $\mathbf{c}_i$，但**只在 tangent plane 内移动** (不沿 normal 滑动)，然后再 projection 到 reference surface 保持几何。

### 2.5 Feature Preservation

Sharp edges 和 material boundaries 必须保留。具体处理：
- **Don't move corners**: vertex 在 sharp edge 上不允许 tangential relaxation 移出该 edge
- **Collapse only along features**: collapse 只能沿 feature edge 方向 collapse (不能跨越)
- **Don't flip feature edges**: feature edge 永不 flip
- **Project to feature curves**: 移动后必须 projection 到 feature curve 上

直觉：feature edges 像是 "guide rails"，vertex 必须沿 rails 移动。这样最终 mesh 既有好的 element quality，又能保留几何特征的 topology。

### 2.6 Adaptive Remeshing

uniform target $L$ 不够好——高曲率处应该用更短的边，低曲率处可以用更长的边。策略：

$$L_{\text{local}} = \min\left(L, \frac{c}{\kappa_{\max}}\right)$$

公式变量：
- $L_{\text{local}}$: 局部 target edge length
- $L$: global upper bound
- $\kappa_{\max}$: 局部 max principal curvature
- $c$: 用户控制的常数 (e.g. 0.5)

intuition: 曲率半径 $r = 1/\kappa_{\max}$ 是局部几何的"自然尺度"。若 edge 比 $r$ 长很多，离散误差就大；若 edge 远短于 $r$，浪费。设 $L \sim c \cdot r$ 是经验上的好选择。

---

## Part III — Smoothing / Fairing

### 3.1 Motivation

- 扫描数据有 noise (硬件误差、registration 误差)
- Marching cubes 输出的 mesh 有 staircase artifacts
- 噪声 mesh 在 shading、simulation、下游 geometry processing 都出问题

intuition: noise 是高频信号，几何是低频信号。我们要做的是**低通滤波** surface。

### 3.2 Smoothness 与 Curvature

几何上，smooth surface 是 **curvature 变化平缓**的 surface。我们想 minimize 的目标自然涉及 curvature。

三种 curvature:

1. **Principal curvatures $\kappa_{\min}, \kappa_{\max}$**
   - Nonlinear operator (min, max)
   - "discontinuous" 因为排序切换

2. **Gauss curvature $K = \kappa_{\min} \cdot \kappa_{\max}$**
   - Intrinsic: 只依赖 Riemannian metric，不依赖 ambient embedding
   - 缺点: developable surface (cylinder) 上 $K = 0$ → 不能 distinguish

3. **Mean curvature $H = \frac{\kappa_{\min} + \kappa_{\max}}{2}$**
   - 通过 Laplace-Beltrami operator 直接获得:
     $$\Delta_{\mathcal{M}} \mathbf{p} = -2 H \mathbf{n}$$
   - $\Delta_{\mathcal{M}}$: Laplace-Beltrami operator (surface 上 Laplacian)
   - $\mathbf{p}$: surface 上的位置 vector (在 $\mathbb{R}^3$)
   - $H$: mean curvature
   - $\mathbf{n}$: surface normal

intuition: 在 surface 上 Laplacian of position field 既有方向 (normal 方向) 也有大小 ($2H$)。这是为什么 mean curvature flow 等价于 surface 上的 heat diffusion。

#### Discrete Laplace-Beltrami

$$\Delta_{\mathcal{M}}(\mathbf{p}_i) = \delta_i = \frac{1}{W_i} \sum_{j \in \mathcal{N}(i)} w_{ij} (\mathbf{p}_j - \mathbf{p}_i)$$

- $\delta_i$: vertex $v_i$ 处的 discrete Laplacian (vector)
- $W_i = \sum_j w_{ij}$: 归一化 weight
- $w_{ij}$: edge weight (uniform / cotangent / etc.)
- $\mathbf{p}_j - \mathbf{p}_i$: vertex $v_j$ 相对 $v_i$ 的位移

$\delta_i$ 的方向 → approximates normal $\mathbf{n}_i$
$\delta_i$ 的大小 → approximates $2H_i$

### 3.3 1D Intuition: Smoothing a Curve

#### 连续版本

考虑曲线 $C(s)$，$s$ 是弧长参数。Laplace 1D 是 second derivative w.r.t. arc length:
$$L(\mathbf{p}(s)) = \frac{d^2 \mathbf{p}}{ds^2}$$

直觉：曲线在某点的二阶导指向 curvature center，大小为 curvature $\kappa$。所以"沿 Laplacian 流动"= 减小曲率=smooth。

#### 离散 1D (uniform sampling)

$$L(\mathbf{p}_i) = \frac{1}{2}(\mathbf{p}_{i-1} - \mathbf{p}_i) + \frac{1}{2}(\mathbf{p}_{i+1} - \mathbf{p}_i)$$

- $\mathbf{p}_{i-1}, \mathbf{p}_i, \mathbf{p}_{i+1}$: 相邻三个 sample
- 系数 1/2: 因为每个 sample 离 $\mathbf{p}_i$ 弧长为 $\Delta s$
- $\frac{1}{\Delta s^2} [(\mathbf{p}_{i+1} - \mathbf{p}_i) - (\mathbf{p}_i - \mathbf{p}_{i-1})] \cdot (\Delta s)^2$ → 这里假设 $\Delta s = 1$ 简化

#### Matrix-vector form

$$\mathbf{L} \mathbf{p}$$

$\mathbf{L}$ 是 1D Laplacian matrix (三对角):
$$\mathbf{L} = \frac{1}{2} \begin{pmatrix} \ddots & & \\ -1 & 2 & -1 \\ & -1 & 2 & -1 \\ & & & \ddots \end{pmatrix}$$

对 interior vertices, $\mathbf{L}$ 的每行 = $(-1, 2, -1)/2$，这是经典的 discrete Laplacian。

#### Flow to reduce curvature

迭代更新：
$$\tilde{\mathbf{p}}_i = \mathbf{p}_i + \lambda L(\mathbf{p}_i) = \mathbf{p}_i + \lambda \frac{d^2 \mathbf{p}}{ds^2}$$

公式变量：
- $\tilde{\mathbf{p}}_i$: 新位置
- $\lambda \in (0, 1)$: smoothing 参数 (类似 learning rate / step size)
- $L(\mathbf{p}_i)$: 当前点处 Laplacian

Matrix form:
$$\tilde{\mathbf{p}} = \mathbf{p} + \lambda \mathbf{L} \mathbf{p} = (\mathbf{I} + \lambda \mathbf{L}) \mathbf{p}$$

intuition: $\mathbf{I} + \lambda \mathbf{L}$ 是一个 **low-pass filter matrix**。它的 eigenvalues 在 $[1 - 4\lambda, 1]$ 区间，将 Laplacian 的 high-frequency eigenvectors (高频模式) 衰减。

#### Drawback

paper 中展示 λ=0.5, 迭代 1000, 5000, 10000, 50000 步: curve 越来越收缩，最终退化为一点。这是 explicit mean curvature flow 的本质缺陷——**体积不守恒**。

---

### 3.4 在 Mesh 上的 Mean Curvature Flow

#### 连续 PDE

$$\frac{\partial \mathbf{p}}{\partial t} = \lambda \Delta_{\mathcal{M}} \mathbf{p} = -2 \lambda H \mathbf{n}$$

- $\mathbf{p}(\mathbf{x}, t)$: surface 在时间 $t$ 处的位置 field
- $\Delta_{\mathcal{M}}$: Laplace-Beltrami
- $H$: mean curvature
- $\mathbf{n}$: normal

直觉：surface 上的每个点沿 normal 方向移动，速度正比于 local mean curvature。曲率大的地方移动快 → 弯曲被消除 → smooth。这就是著名的 **mean curvature flow** (MCF)，等价于 surface 上 heat equation，所以有极强 smoothing 作用。

#### Forward Euler (explicit)

$$\frac{\mathbf{p}^{(n+1)} - \mathbf{p}^{(n)}}{dt} = \lambda \mathbf{L} \mathbf{p}^{(n)}$$
$$\mathbf{p}^{(n+1)} = (\mathbf{I} + dt \cdot \lambda \mathbf{L}) \mathbf{p}^{(n)}$$

- $\mathbf{p}^{(n)}$: 第 $n$ 步所有 vertex 位置 stacked vector
- $dt$: 时间步长
- $\mathbf{L}$: discrete Laplacian matrix
- $\mathbf{I}$: identity

**Unstable** 当 $dt \cdot \lambda$ 大时。Largest eigenvalue of $\mathbf{L}$ 在 uniform Laplacian case 上界为 $\sim 2 \cdot \text{valence}$，所以 $dt$ 必须满足 CFL-like condition $dt < 1/(2\lambda \cdot \text{valence}_{\max})$。

#### Taubin's λ|μ Smoothing (SIGGRAPH '95)

Taubin 的洞察：纯 smoothing 会让 mesh 缩水，能否做 **low-pass filter** 而**保留低频内容**?

策略：交替做 smoothing 和 inflation:
$$\tilde{\mathbf{p}} = (\mathbf{I} + \lambda \mathbf{L}) \mathbf{p}$$
$$\tilde{\tilde{\mathbf{p}}} = (\mathbf{I} + \mu \mathbf{L}) \tilde{\mathbf{p}}$$

- $\lambda > 0$: smooth step (低通 + 收缩)
- $\mu < 0$: "inflate" step (逆向 Laplacian)

intuition: 这是一个 **band-pass filter**。Laplacian 的 eigenvectors $\mathbf{v}_k$ 对应 frequency $\omega_k = -\lambda_k$ (eigenvalue 是负的)。组合 filter 后每 mode 乘以 $(1 + \lambda \omega_k)(1 + \mu \omega_k)$，这是一个**双零点 low-pass filter**。

Taubin 推荐 $\lambda = 0.6307$, $\mu = -0.6732$ (来自 signal processing 设计)。

**优点**: simple, 不收缩 (理论上能量守恒于 low-freq modes)
**缺点**: 收敛慢 (数百次 iteration)，结果仍依赖 mesh connectivity

---

### 3.5 Implicit Fairing [Desbrun et al., SIGGRAPH '99]

#### Backward Euler

Forward Euler 不稳定，问题来自用 $\mathbf{p}^{(n)}$ 算 Laplacian 而更新到 $\mathbf{p}^{(n+1)}$。改用 $\mathbf{p}^{(n+1)}$ 算 Laplacian (backward Euler):

$$\frac{\mathbf{p}^{(n+1)} - \mathbf{p}^{(n)}}{dt} = \lambda \mathbf{L} \mathbf{p}^{(n+1)}$$

整理:
$$\mathbf{p}^{(n+1)} - dt \lambda \mathbf{L} \mathbf{p}^{(n+1)} = \mathbf{p}^{(n)}$$
$$\boxed{(\mathbf{I} - dt \lambda \mathbf{L}) \mathbf{p}^{(n+1)} = \mathbf{p}^{(n)}}$$

变量解释：
- $\mathbf{I} - dt \lambda \mathbf{L}$: 线性系统 matrix
- $\mathbf{p}^{(n)}$: 当前已知位置
- $\mathbf{p}^{(n+1)}$: 求解下一步位置

每步需要解一个 sparse linear system (e.g. via CG 或 BiCG)。

**Unconditionally stable** for any $dt > 0$.

intuition 为什么 implicit 稳定：
- 显式: 高频模式被 $|1 - dt \omega_k|$ 放大若 $dt \omega_k > 2$
- 隐式: 高频模式被 $|1/(1 + dt \omega_k)| < 1$ 永远衰减

#### 实验结果 (paper 中 Figure 4)

- Stanford bunny 原图: noisy
- 10 步 explicit with $\lambda dt = 1$: 几乎没效果 (CFL 限制)
- 1 步 implicit with $\lambda dt = 10$: 7 iterations PCG (preconditioned bi-conjugate gradient), 比 explicit 快 30%
- 20 步 Taubin $\lambda|\mu$: 速度慢，效果中等

Implicit fairing 不仅快，**还支持大 $dt$**，可以一步完成大量 smoothing。

参考 https://dl.acm.org/doi/10.1145/311535.311576

---

### 3.6 Mesh Independence: 权重选择的重要性

paper 中关键观察：uniform Laplacian 的 smoothing 结果**依赖 mesh 密度**。即使 underlying surface 是对称的，若 mesh 在一侧更密，uniform smoothing 会产生不对称结果。

原因：uniform Laplacian $\frac{1}{\text{valence}} \sum (\mathbf{p}_j - \mathbf{p}_i)$ 是 connectivity-based 的，它不知道每条边的几何意义。

#### Cotentangent Weights (cotangent Laplacian)

正确的 discrete Laplace-Beltrami 应该用 cotangent weights:
$$w_{ij} = \frac{1}{2}(\cot \alpha_{ij} + \cot \beta_{ij})$$

- $\alpha_{ij}$: edge $(v_i, v_j)$ 对面 vertex $v_k$ 处的角
- $\beta_{ij}$: edge $(v_i, v_j)$ 对面另一个 vertex $v_l$ 处的角 (若 edge 是 boundary 只有一个)

intuition: cotangent weights 来自 Poisson equation 在 triangle mesh 上的 finite element 离散化，是真正的 **geometric Laplacian**。它考虑了 triangle 的形状，所以 mesh 密度变化时仍然给出同一结果。

参考: Meyer, Desbrun, Schröder, Barr "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds" (2002)
https://dl.acm.org/doi/10.1007/978-3-662-05189-4_2

---

## Part IV — 高阶联想

### 4.1 CVT 与 Optimal Transport

CVT energy 实际上是 **2-Wasserstein distance** 在 quantization 形式下的特殊情形。最近几年 neural network + OT 的方向 (e.g. PointNet++, shape reconstruction) 都用到这个能量。

可以参考 Liu et al. 2009 的 L-BFGS 加速:
https://dl.acm.org/doi/10.1145/1516522.1516536

### 4.2 Mean Curvature Flow 与 Willmore Energy

MCF 实际上是 $L^2$ gradient flow of area functional. 若想 minimize bending energy (Willmore energy $\int H^2 dA$) 则需更高阶 PDE (Willmore flow)。这是后面 surface fairing 文献的进阶主题:
- Clarenz et al. "Willmore flow" (2000)
- Bobenko & Schröder "Discrete Willmore flow"

### 4.3 Bilateral / Nonlinear Smoothing

Linear Laplacian smoothing 在保留 sharp features 时会失败——它会磨平所有特征。改进:
- Bilateral filtering (Jones et al. 2003, Fleishman et al. 2003): 用 normal + position 双 kernel 保留 sharp edges
- Anisotropic diffusion (例如 Perona-Malik 的 mesh 版本): 根据 gradient magnitude 调整 conductivity

### 4.4 Modern Learning-based Approaches

近 5 年 (2020-2025) 几何处理的 deep learning 方向:
- **DGCNN** (Wang et al. 2019): EdgeConv 用 k-NN graph
- **PointNet++** (Qi et al. 2017): Multi-scale sampling 类似 adaptive remeshing
- **Neural Subdivision** (Liu et al. 2020): 学习 subdivision rules
- **DeepMesh** (2024): Diffusion model 生成 mesh

### 4.5 Quad Meshing & Directional Fields

paper 里提到 triangles vs quads 没深入展开。Quad meshing 经典工作:
- Bommes, Zimmer, Kobbelt "Mixed-integer quadrangulation" (SIGGRAPH 2009)
- "Frame field" 或 "cross field" 决定 quad orientation

参考 https://www.graphics.rwth-aachen.de/publication/03175/

### 4.6 Numerical Linear Algebra

Implicit fairing 解的是 $(\mathbf{I} - dt \lambda \mathbf{L}) \mathbf{p} = \mathbf{p}_n$。
- $\mathbf{L}$ 是 symmetric positive semi-definite (cotangent)
- 矩阵 sparse, 用 sparse Cholesky / PCG
- 大 mesh (millions of vertices) 用 hierarchical preconditioner (e.g. HLibPro, https://www.hlibpro.com/)

---

## Part V — 实战 Intuition 总结

| 任务 | 选择 | 为什么 |
|---|---|---|
| 想要 uniform isotropic remeshing | Surface-oriented local operators (Botsch 04) | 快，robust，不需要 cut |
| 想要 anisotropic / curvature-aligned | Parameterization-based (Alliez 03) | 2D 各向异性 sampling 容易 |
| 想做 fast smoothing | Implicit fairing | 1 步可以做大 smoothing |
| 想做 feature-preserving smoothing | Bilateral filter 或 Taubin + feature lock | Linear Laplacian 会磨平 |
| 想做 coarse remeshing (大大简化) | Parameterization-based + CVT | Surface-oriented 在 face 数减少很多时不稳 |
| Mesh 上有 sharp edges | 保留 feature edges, 禁止 flip, only along collapse | feature 像导轨 |

---

## 参考资料汇编

**Remeshing:**
- Alliez, Meyer, Desbrun "Interactive Geometry Remeshing" (SIGGRAPH 2002) https://dl.acm.org/doi/10.1145/566654.566659
- Alliez et al. "Anisotropic Polygonal Remeshing" (SIGGRAPH 2003) https://hal.inria.fr/inria-00302666/
- Botsch, Bommes, Kobbelt "Efficient Linearly Constrained Remeshing" (2004) https://cg.cs.uni-bielefeld.de/publications/paper/2004_Polygon-Remeshing.pdf
- Liu et al. "On Centroidal Voronoi Tessellation—Energy Smoothness and Fast Computation" (TOG 2009) https://dl.acm.org/doi/10.1145/1516522.1516536
- Dunyach et al. "Adaptive remeshing for real-time mesh deformation" (2013)

**Smoothing:**
- Taubin "A Signal Processing Approach to Fair Surface Design" (SIGGRAPH 1995) https://dl.acm.org/doi/10.1145/218380.218449
- Desbrun, Meyer, Schröder, Barr "Implicit Fairing of Irregular Meshes using Diffusion and Curvature Flow" (SIGGRAPH 1999) https://dl.acm.org/doi/10.1145/311535.311576
- Meyer, Desbrun, Schröder, Barr "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds" (2002) https://dl.acm.org/doi/10.1007/978-3-662-05189-4_2
- Jones, Durand, Desbrun "Non-Iterative, Feature-Preserving Mesh Smoothing" (SIGGRAPH 2003) https://dl.acm.org/doi/10.1145/1201775.882367
- Fleishman, Drori, Cohen-Or "Bilateral Mesh Denoising" (SIGGRAPH 2003) https://dl.acm.org/doi/10.1145/1201775.882370

**Books:**
- Botsch, Kobbelt, Pauly, Alliez, Levy "Polygon Mesh Processing" (2010) https://www.pmp-book.org/
- Crane, "Discrete Differential Geometry: An Applied Introduction" (2019 修订版) https://www.cs.cmu.edu/~kmcrane/

**Code & Demos:**
- PMP (Polygon Mesh Processing) Library https://www.pmp-library.org/
- libIGL https://libigl.github.io/
- OpenMesh https://www.openmesh.org/
- Geometry Central https://geometry-central.net/

---

## 最后总结一句话 Intuition

Remeshing = 在固定 underlying surface 的几何信息下，重新设计离散表示的 combinatorial + positional 结构，使得 mesh 在离散计算意义下"各向同性 / 自适应 / feature-preserving"。

Smoothing = 把 surface 当作信号，用 Laplace-Beltrami 这个 surface 上的"二阶导数算子"做低通滤波，可以在 surface 上做 heat equation 的离散积分 (explicit 或 implicit)，关键在于选对 weights (cotangent) 让结果 mesh-independent。

两者本质都是 **用 PDE 视角看待离散几何**，是 geometric deep learning 的祖先之一。
