---
source_pdf: Algorithms for Generalized Signed Distance and Winding Numbers.pdf
paper_sha256: d69fb4d13ebb8b1687b32e5f360c9baae4b17bb63bae3b494983d80df141ba3a
processed_at: '2026-08-18T00:42:31-07:00'
target_folder: Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

这篇论文虽然充满了复杂的数学公式，但核心思想其实非常直观。如果用大白话来概括，就是在讲**“如何从一堆破烂的几何数据中，猜出它原本完整的形状，并算出空间里任意一点到这个形状的精确距离”**。

想象一下你用3D扫描仪扫了一个杯子，但因为技术限制，扫出来的数据全是洞、有噪点、甚至有些面连错了。这时候，连最基本的问题都很难回答：“这个点是在杯子里面还是外面？”、“这个点离杯子的表面有多远？”

这篇论文的核心主张是：**不要试图去修补那些破烂的三角形网格或点云，而是把它们转换成一个在整个空间中平滑分布的“数学场”（就像温度场或磁场）。这种平滑的场天生对瑕疵免疫，能自动把漏洞补齐。**

论文把这个想法拆解成了三个递进的算法：

### 1. 核心发现：“里外”和“距离”本来就是一家人
以前人们觉得，“判断一个点在不在形状内部”和“算一个点到形状的距离”是两个不同的问题。作者通过数学发现，它们其实是同一个数学方程的两个极端。
- 如果你把方程里的一个参数（叫 $\lambda$）调到最小，方程就会尽力去“补窟窿”，猜出完整的形状，但它算不准距离（只能告诉你里外）。
- 如果你把 $\lambda$ 调到最大，方程能算出极其精确的距离，但它会死死盯着最近的那个数据点，根本无法把破缺的形状补全。
- **传统方法的困境就在于：你没办法同时既要补窟窿，又要算准距离。** 尤其是当你手里只有一堆离散的点（点云）时，如果强行算距离，算出来的只是“到最近那个点的距离”，而不是“到真实杯子表面的距离”。

### 2. 解决方案一：散播箭头
为了打破上面的困境，作者提出了第一种方法（Signed Heat Method，SHM）。
与其直接去平滑“位置”或“距离”（这会导致上述的死盯着最近点的问题），不如去平滑“方向”。
想象一下，你在原本破碎的表面撒上无数根小箭头（法向量），表示表面朝哪个方向。然后你让这些箭头像热量一样向周围空间扩散。神奇的是，当箭头扩散开后，即使在有窟窿的地方，箭头也会自动顺着趋势排列好，把窟窿的走向猜出来。
最后，你只要顺着这些排列好的箭头走，就能自然地量出精确的距离，同时还能自动补全形状。

### 3. 解决方案二：给每个点套上一个甜甜圈
第一种方法很好，但它需要把整个空间切分成网格（像我的世界那样）才能计算，如果场景太大就太慢了。如果你只有一堆稀疏的点云，想要极快地查出任意一点的距离怎么办？作者提出了第二种方法（Points as Tori, PAT）。
既然我们不能只看孤立的点，那不如给每个点拟合一个微小的局部曲面。作者发现，“甜甜圈（圆环）”是一种极其完美的万能局部形状——把它压扁就是平面，拉直就是圆柱，缩胖就是球面，甚至还能表示马鞍面。
所以，算法利用一点点机器学习，观察每个点周围的邻居，猜出这里原本的曲面长什么样，然后给这个点套上一个虚拟的“甜甜圈”。查询距离时，只需要把附近几个甜甜圈的预测结果平滑地混合一下就行了。这种方法不需要网格，速度极快（几毫秒查一次），还能直接放到GPU shader里渲染。

### 4. 解决方案三：画在曲面上的迷宫
前面两种方法都默认形状能分出“里外”。但如果你是在一个本身就很复杂的表面上（比如画在一个甜甜圈形状的曲面上），有些线条是绕着中间的洞转圈的，它们根本就不包围任何区域，也就没有所谓的“里面”。
为了处理这种情况，作者提出了第三种方法（Surface Winding Numbers, SWN）。
算法同样借用了“平滑场”的思想。它先把杂乱的线条变成一个平滑的函数，然后通过分析这个函数的“流向”（Hodge分解），自动把线条分成两类：一类是真正能围成一个圈的（有里外），另一类只是穿过表面的（无里外）。过滤掉后者后，就能完美地给曲面上的区域打标签了。

### 总结
这篇论文本质上是在做“几何数据的去噪与补全”。它没有用现在流行的暴力堆叠深度学习网络的方式，而是利用了深刻的数学物理直觉（扩散过程、调和场、同调论），把残缺不全的几何数据转化为平滑的场来处理。用极小的计算代价和极强的数学可解释性，解决了计算机图形学里最基础也最头疼的“破烂数据处理”问题。

---

# Algorithms for Generalized Signed Distance and Winding Numbers 论文讲解

## 论文全景

这篇是 CMU 的 Nicole Feng 在 2026 年的 PhD 论文（CMU-CS-26-104），导师是 Keenan Crane（[Geometry Collective](https://geometrycollective.github.io/)）。论文的统一主题是**"generalized"几何推断**——从不完美、不完整、有噪声、自相交或拓扑错乱的观测数据中，反推底层曲线/曲面的 inside/outside 与 signed distance。这篇论文的核心贡献是把三个看似不同的问题——winding numbers、Poisson surface reconstruction、signed distance——用 screening 参数 λ 串成一个连续谱，并据此设计出三种算法：

- **Signed Heat Method (SHM)** — 基于短时 vector diffusion（[Feng & Crane, SIGGRAPH 2024](https://doi.org/10.1145/3658220)）
- **Points as Tori (PAT)** — 基于学习局部 torus SDF 的 pointwise formula（[Feng et al., SIGGRAPH 2026](https://doi.org/10.1145/3811385)）
- **Surface Winding Numbers (SWN)** — 任意拓扑表面上的 inside/outside（[Feng, Gillespie & Crane, SIGGRAPH 2023](https://doi.org/10.1145/3592401)）

项目主页: [nzfeng.github.io/research](https://nzfeng.github.io/research)

---

## 1. 核心动机：为什么"generalized"问题难

数字几何几乎总是 broken 的——scanner 出 missing data, modeler 出 self-intersection, orientation flipped, nonmanifold edges。对于**干净几何**，inside/outside 与 distance 是 well-defined 的；一旦数据 broken，这两个问题都变得 ambiguous，更糟糕的是**"compute unsigned distance then sign" 与 "compute signed distance directly" 不再等价**（见 Figure 2.4 inset）。比如 fast marching 在 broken geometry 上会 propagate sign error；pseudonormal test 在 broken geometry 上完全失效。

论文的核心主张是：**与其直接修补 broken geometry，不如去处理一个 globally-defined smooth function，让 function 的结构吸收掉局部的瑕疵**。这是 Keenan Crane 组一向的方法论——"work with functions, not geometry"。

---

## 2. Chapter 3: 理论基础——三个问题的统一

### 2.1 起点：eikonal equation 与 Hopf-Cole 变换

Signed distance function (SDF) $\phi$ 形式上满足 **signed eikonal equation**:

$$
\begin{cases}
\|\nabla u(x)\|^2 = 1 & x \notin \Omega \\
u(x) = 0 & x \in \Omega \\
\frac{\partial u}{\partial n}(x) = 1 & x \in \Omega
\end{cases}
\tag{3.5}
$$

变量解释：
- $\Omega$：source geometry（曲线/曲面），submanifold of codimension 1
- $u(x)$：signed distance function
- $n(x)$：$\Omega$ 上 outward-pointing normal
- 关键差异 vs unsigned 版本：Neumann 条件 $\partial u/\partial n = 1$ **连续穿越** $\Omega$（无跳），而 unsigned 版本 $\partial u^\pm/\partial n = \pm 1$ **两侧反向**

Viscosity 形式（Crandall & Lions [1983](https://www.ams.org/tran/1983-277-01/S0002-9947-1983-0690039-0/)）：

$$
\text{sign}_\Omega(x)(\|\nabla u\|^2 - 1) = \frac{1}{\lambda}\Delta u
\tag{3.6}
$$

加入右端 $\frac{1}{\lambda}\Delta u$ 这个 viscosity term 把非线性 eikonal 正则化为一个非线性 elliptic 问题。$\lambda \to \infty$ 时退化为纯 eikonal。

### 2.2 Hopf-Cole 变换与 jump screened Laplace equation

经典 Hopf-Cole 变换（[Hopf 1950](https://doi.org/10.1002/cpa.3160030302); [Cole 1951](https://doi.org/10.1090/qam/42600)）通过 $w(x) = e^{-\lambda u(x)}$ 把 viscous Burgers 转成 linear heat equation。Lipman 在 [2021 ICML](https://proceedings.mlr.press/v139/lipman21a.html) 把这套搬到了 signed distance：

$$
w(x) = \text{sign}_w(x) \exp\left(-\lambda\, \text{sign}_w(x)\, u(x)\right)
\tag{3.7}
$$

其中 $\text{sign}_w(x)$ 是 $w$ 的符号（与 $u$ 同号）。代入 Equation 3.6 得到 **jump screened Laplace equation**：

$$
\begin{cases}
\Delta w(x) - \lambda^2 w(x) = 0 & x \notin \Omega \\
w^\pm(x) = \pm 1 & x \in \Omega \\
\partial w^+/\partial n = \partial w^-/\partial n & x \in \Omega
\end{cases}
\tag{3.8}
$$

变量与含义：
- $w^\pm(x) := w(x \pm s n(x))$ 当 $s \to 0$，即从 $\Omega$ 正/负侧逼近 $x$ 的 one-sided limit
- $\lambda$：screening amount，控制 Laplace 的阻尼强度
- 边界条件 1：Dirichlet-ish 但 **跳变** ±1（保留 signed 信息）
- 边界条件 2：normal derivative 跨 $\Omega$ **连续**（保留 Neumann 信息）
- 逆变换：$u(x) = -\frac{1}{\lambda}\text{sign}_w(x)\log|w(x)|$，即 Equation 3.9

### 2.3 三个方法连续相连：λ 的角色

这是论文 Figure 3.1 所展示的**关键洞察**：

- **λ → 0**：Equation 3.8 退化为 jump Laplace equation（Equation 2.2）——这就是 **generalized winding number (GWN)** 和 unregularized **Poisson surface reconstruction (PSR)**！
- **λ → ∞**：通过对数变换 Equation 3.9，$w$ 的指数衰减被反演，得到 **signed distance function**
- **中间 λ**：得到一个 between reconstruction 和 distance 的"hybrid" function

**所以 winding numbers / Poisson reconstruction / signed distance 是同一族 PDE 的两个极端**——加一个 screening term $\lambda^2 w$ 就把 occupancy method 升级成 distance method。

也可以写成右端集中分布的 screened Poisson 方程：

$$
\Delta w(x) - \lambda^2 w(x) = -2(\nabla \cdot n(x))\,\mu_\Omega(x)
\tag{3.10}
$$

其中 $\mu_\Omega$ 是集中在 $\Omega$ 上的 measure，$n(x)$ 是 $\Omega$ 法向。这正是 Poisson surface reconstruction 写出的形式——把 source term 替换成 $\nabla \cdot n \,\mu_\Omega$——而 Kazhdan 2006 PSR 等价于把这个右端用 Gaussian 卷积正则化然后取 λ→0（[Chen et al. 2024a](https://doi.org/10.1145/3687914)）。

### 2.4 Green's function视角：Yukawa double-layer potential

Equation 3.8 的 free-space Green's function（Yukawa/screened Coulomb potential）：

$$
G^\lambda(x,y) := \frac{\exp(-\lambda \|x-y\|)}{4\pi \|x-y\|}
\tag{3.11}
$$

边界积分公式：

$$
w(x) = \int_\Omega \frac{(\lambda\|x-z\|+1)\langle x-z, n(z)\rangle}{2\pi\|x-z\|^3} \exp(-\lambda\|x-z\|)\,dz
\tag{3.13}
$$

当 $\lambda \to 0$，kernel $\frac{\langle x-z, n(z)\rangle}{2\pi\|x-z\|^3}$ 就是经典的 winding number / solid angle kernel（[Jacobson et al. 2013](https://doi.org/10.1145/2461912.2461916)）。

### 2.5 Convolutional distance formulas：Laplace's method

更一般地，对任意函数 $h, \varphi$，**Laplace's method**（[Evans 1998 §4.5](https://www.ams.org/books/amstexttext/019/)）给出当 $\lambda \to \infty$：

$$
\int_\Omega h(z)\exp(-\lambda\varphi(z))\,dz \sim \left(\frac{2\pi}{\lambda}\right)^{d/2} \det(\nabla^2\varphi(x^*))^{-1/2} h(x^*) \exp(-\lambda\varphi(x^*))
\tag{3.14}
$$

- $x^* := \arg\min_{z\in\Omega}\varphi(z)$ — 唯一最小点
- $d$ — domain 维数
- 渐近近似到 $O(\lambda^{-1})$ 项

两边取 $-\frac{1}{\lambda}\log(\cdot)$ 后，**最右边的指数项成为 dominant contribution**，其他项衰减到 0。所以：

$$
\tilde{d}^\lambda(x) = -\frac{1}{\lambda}\log\left(\int_\Omega h_x(z)\exp(-\lambda\|x-z\|)\,dz\right) \to \min_{z\in\Omega}\|x-z\|
\tag{3.15}
$$

这就是 **convolutional distance formula** 的一般形式，subsumes LogSumExp distance ([Madan & Levin 2022](https://doi.org/10.1145/3528223.3530093)), Kreisselmeier-Steinhauser, Varadhan, Schrödinger distance transform 等。

**自归一化版本**（self-normalized，partition-of-unity 风格）：

$$
\hat{d}^\lambda(x) = \frac{\int_\Omega g_x(z)\exp(-\lambda\|x-z\|)\,dz}{\int_\Omega \exp(-\lambda\|x-z\|)\,dz}
\tag{3.16}
$$

→ $\lambda\to\infty$ 时退化为 $g_x(x^*)$，**即 closest point 处 g 值的估计**。

### 2.6 Section 3.3 的 fundamental limitation：为什么 naive 公式在点云上崩

这是论文的一个核心结论，需要建立直觉：**naive convolutional distance 公式在 point cloud 上注定失败**。

原因：当 $\lambda\to\infty$，公式完全由 **minimizer $x^*$ 处的局部行为**决定（Laplace method 的本质）。对 dense image contours，sample grid 与 distance grid 同分辨率，这没问题；但对 point cloud，$x^*$ 是 **最近采样点本身**，所以 $\tilde{d}^\lambda(x) \to \min_i \|x-p_i\|$ — 你拿到的是到离散点集的距离，而不是到 underlying surface 的距离。

更糟糕的是 λ 是一个**矛盾双耦**：
- 大 λ → 好 eikonality（距离梯度模 ≈ 1）→ 但 overfits 到点集
- 小 λ → 好插值，逼近 underlying surface → 但距离梯度远离 1，不是 distance function

正则化 kernel 没用——Figure 3.2 展示了 isotropic / anisotropic Gaussian regularization 都被 asymptotic regime 消除掉了，因为正则化等价于选 $h$，而 $h$ 在 $\lambda\to\infty$ 下被指数 dominant 项淹没。

### 2.7 与 diffusion/flow matching 的深刻联系（Section 3.4）

论文很 clever 地指出这套 asymptotic 与 generative modeling 的 flow matching / diffusion model 的关系：

- Diffusion model 的 score function $\nabla\log\rho_t$ 用 Gaussian mixture 闭式表达后，第一项就是 $\sum_i t x_i \exp(-\|z-tx_i\|^2/2(1-t)^2) / \sum_i \exp(-\cdot)$ —— 这就是 Equation 3.16 形式！
- 等价于把 sample 推向 **nearest training point**
- 所以 **flow matching 本质上是在做 kernel density estimation**，注定 overfit training set（[Liu et al. 2022](https://arxiv.org/abs/2209.03003), [Somepalli et al. 2023](https://doi.org/10.1109/CVPR52729.2023.00586), [Carlini et al. 2023](https://www.usenix.org/conference/usenixsecurity23)）

正如论文所说："generalization doesn't come from the flow model, but rather from the neural network (over)regularizing the score function." 这是把 SCALAR diffusion 改为 **VECTOR diffusion 才能突破**的根本动机——Bamberger et al. [2025](https://arxiv.org/abs/2510.05930) 也独立得到了类似结论。

---

## 3. Chapter 4: Signed Heat Method (SHM)

### 3.1 核心思想：vector diffusion 提供 parallel transport

短时 vector heat diffusion 的渐近行为（[Berline, Getzler & Vergne 1992, Thm 2.30](https://link.springer.com/book/10.1007/978-3-662-03580-2)）：

$$
k_t^\nabla \sim (4\pi t)^{-d/2} \exp\left(-\frac{\text{dist}(x,y)^2}{4t}\right) c(x,y) \sum_{i=0}^\infty t^i \Phi_i^\Delta(x,y)
\tag{4.4}
$$

- $k_t^\nabla$ — vector heat kernel on $M$
- $\Phi_0^\Delta(x,y)$ — leading term，是 **parallel transport map** $P_{\gamma_{xy}}$ along shortest geodesic
- $c(x,y)$ — scalar surface-related correction
- $t$ — diffusion time

**直觉**：当 $t\to 0$，diffused vector $X_t(x)$ ≈ parallel transport of the normal $N(\bar{x})$ at the closest point $\bar{x}\in\Omega$ along the shortest geodesic $\gamma_{x\bar{x}}$。Parallel transport along geodesics 保持切性，所以 $X_t(x)$ 切于 $\gamma$，从而平行于 unsigned distance gradient $\nabla\phi$；并且因为 transport 的是 **oriented** normal，**符号也对了**。

### 3.2 三步算法

**Step 1**：扩散 normals，解 vector diffusion equation

$$
\frac{d}{dt}X_t = \Delta^\nabla X_t, \quad X_0 = N\mu_\Omega
\tag{4.5}
$$

- $\Delta^\nabla$ — connection Laplacian on vectors
- $N\mu_\Omega$ — 集中在 $\Omega$ 上的 vector-valued measure，沿 $\Omega$ 法向 $N$ 方向

**Step 2**：normalize 得到 SDF gradient 的近似

$$
Y_t := X_t / \|X_t\|
$$

**Step 3**：求一个 gradient 最小二乘匹配 $Y_t$ 的 scalar function $\phi$

$$
\min_{\phi:M\to\mathbb{R}} \int_M \|\nabla\phi - Y_t\|^2
\tag{4.6}
$$

变分给出 Poisson equation：

$$
\Delta\phi = \nabla\cdot Y_t \quad \text{on } M, \quad \frac{\partial\phi}{\partial n} = n\cdot Y_t \quad \text{on } \partial M
\tag{4.7}
$$

### 3.3 离散化细节

时间离散用一步 backward Euler：

$$
(\text{id} - t\Delta^\nabla)X_t = X_0
\tag{4.8}
$$

**Crouzeix-Raviart (CR) basis** on edges — 每个 edge $ij$ 关联一个 facewise linear function $\varphi_{ij}$ 在 edge midpoint 取 1，其他 midpoint 取 0。好处是曲线源可以直接在 face 内部 discretize，不需要从曲线切空间映射到顶点切空间。

**Complex encoding** of connection Laplacian：用 complex number 表示 face 内切向量（1 对应 $\hat{e}_{ij}$, $i$ 对应 $\hat{e}_{ij}^\perp$），2×2 实 block (4 floats) 换成 1 个 complex (2 floats) — 一半存储。

**时间步长**：$t = h^2$，$h$ 是 edge midpoint 之间平均距离，约等于半平均 edge length。这个选择使方法 scale-invariant（Laplacian 是二阶的）。

最终离散方程组：

$$
(M + tL^\nabla)X = X_0 \quad \text{然后} \quad C\phi = b
\tag{4.9, 4.10}
$$

$C$ 是 cotan Laplacian，$b$ 是离散 divergence of $Y_t$。

### 3.4 边界行为：为什么 SHM 优于 Unsigned Heat Method

[Crane et al. 2013b](https://doi.org/10.1145/2516971.2516977) 的 Unsigned Heat Method (UHM) 在 domain boundary 附近有 bias — 因为 UHM 的 $Y$ 是标量 heat distribution $q$ 的 gradient，而 $q$ 用 Dirichlet/Neumann 边界条件时 $\nabla q$ 在 $\partial M$ 必须正交或切向，无法正确指向最近的 source point。

SHM 没这个问题——vector diffusion 直接给出 closest point 的 normal，与 SDF 真正的 gradient 一致；connection Laplacian 自动 encode 向量场的 zero-Neumann BC（[Gelfand et al. 2000 I.6](https://www.coursera.org/lecture/calculus-on-manifolds)），不需要特殊处理。

### 3.5 精确插值：saddle point formulation

可以把 "$\phi$ 在 $\Omega$ 上为常数" 编码成线性约束 $A\phi = 0$，引入 Lagrange multiplier $\mu$，解 saddle-point 系统：

$$
\begin{bmatrix} C & A^T \\ A & 0 \end{bmatrix} \begin{bmatrix} \phi \\ \mu \end{bmatrix} = \begin{bmatrix} \nabla\cdot Y_t \\ 0 \end{bmatrix}
\tag{4.11}
$$

支持多个 disjoint level set 约束（每条曲线取一个未知常数）。

### 3.6 Distance sharpening：从 SHM 起步做精确 distance

可以利用 signed convex 优化作为后处理。Unsigned distance 等价于：

$$
\max_\phi \int_M \phi \, dx \quad \text{s.t.} \quad |\nabla\phi|\le 1, \; \phi=0\,\text{on}\,\Omega
\tag{4.12}
$$

([Belyaev & Fayolle 2020](https://doi.org/10.1007/s11075-019-00789-5) 用 ADMM)

把 objective 替换成 $\int_M \text{sign}(\phi_0)\phi$，用 SHM 的 $\phi_0$ 作为 warm-start，得到 sharpened signed distance（Figure 4.13）。ADMM 慢但准确；PDGH 一阶便宜但参数难调。

### 3.7 实验数据（Section 4.5）

| Method | 平面 mesh mean error | 时间 | 备注 |
|---|---|---|---|
| SHM (this work) | 接近 UHM | 略慢于 UHM | 自动正确 BC |
| UHM (Crane 2013) | 基准 | 基准 | 需 boundary heuristic |
| ADMM-BF | 最低 | 4-5x 慢于 SHM | 只能 unsigned |
| FMM | 中等 | 快 | broken geometry 严重失败 |

在 44 个 mesh 的 closed curve 基准上，SHM 给出 approximately **linear convergence** in mean edge length (Figure 4.15)，median order of accuracy 约 0.83。对 corrupted curve（拓扑/几何/orientation 错误），SHM fail gracefully — Figure 4.16 显示 error 随 corruption 增长平滑而不是 catastrophic jump。

---

## 4. Chapter 5: Points as Tori (PAT)

### 4.1 动机：SHM 的 FEM 局限性

SHM 作为 FEM 方法，需要 spatial discretization 和 large coupled linear solves。在 isolated pointwise query 上不 efficient。理想的需求：
- output-sensitive（不预先 mesh 整个 domain）
- 避免 spatial discretization（尤其大型场景）
- 支持 differentiable / shader-friendly 表达

直接把 SHM 的 Poisson 方程转成 boundary integral 不行——球面例子的 $\nabla\cdot Y(x) = 2/\|x\|$ 衰减太慢，distance 函数到 infinity 发散，Green 公式不适用。所以必须换思路。

### 4.2 PAT 的核心公式：自归一化 kernel density estimator

回到 Equation 3.16，但关键是用 **per-point torus SDF** 作为 $g_i$：

$$
\phi(x) = \frac{\sum_{i=1}^{|P|} g_i(x)\exp(-\lambda\|x-p_i\|)}{\sum_{i=1}^{|P|}\exp(-\lambda\|x-p_i\|}
\tag{5.1}
$$

- $P = \{p_i\}_{i=1}^{|P|}$ — input point cloud
- $g_i(x)$ — signed distance to torus $\mathbb{T}_i$ fitted at point $p_i$
- $\lambda$ — screening，自动选择

**为什么这能 generalize 而 Equation 3.15/3.16 失败**：因为 $g_i$ 不再是 naive planar distance $\langle x-p_i, n_i\rangle$（pseudonormal），而是 **second-order local surface estimate**——一个 torus 的 SDF。所以即使 $\lambda\to\infty$，公式仍然从最近点取值，但这个值是基于 $k$-neighborhood 的 robust 二阶表面估计，而不是孤立的 planar approximation。

### 4.3 为什么选 tori？

In $\mathbb{R}^2$，曲线局部是圆；in $\mathbb{R}^3$，曲面局部是 **torus** (圆推广到两个曲率方向)。Torus SDF 有 closed-form 表达：

$$
\phi_{\mathbb{T}}(x) = \|d\| - r, \quad d := \big(\|(x-c)\times u\| - R, \, \langle x-c, u\rangle\big)
\tag{5.3}
$$

- $c$ — torus center
- $u$ — axis of revolution (unit vector)
- $R$ — major radius
- $r$ — minor radius
- $\|d\|$ 是 2D 向量 $d$ 的 Euclidean 范数 — 等价于把"圆环面 → 内圈圆环曲线，再 → 查询点"两次嵌套 distance to circle

特殊极限：
- $R \to \infty$ → 平面 / 圆柱
- $R = r$ → horn torus（鞍面附近）
- $r \to 0$ → circle (退化为曲线)

Tori 能 capture 所有二阶 local surface behavior：球面、椭球面、鞍面、圆柱、平面。

### 4.4 Torus fitting：从局部多项式到 torus 参数

每个 $p_i$ 用局部多项式 surface：

$$
Q_i^*(s,t) = p_i + s\,s_i + t\,t_i + Q_i(s,t)\,n_i
\tag{5.2}
$$

$$
Q_i(s,t) = \sum_{n=0}^2\sum_{m=0}^2 a_{n,m}\,s^n t^m
$$

- $s_i, t_i$ — 局部 frame 的两个切向
- $n_i$ — 法向
- $a_{n,m}$ — 多项式系数

实际只需要 6 个系数：$a_{0,0}, a_{0,1}, a_{1,0}, a_{1,1}, a_{0,2}, a_{2,0}$ 就足以决定：
- $a_{0,0}$ — 沿法向的 **shift**（surface 在 $p_i$ 处不在 $p_i$，而在 $p_i + a_{0,0}n_i$）
- $a_{1,0}, a_{0,1}$ — tilt of normal
- $a_{2,0}, a_{0,2}$ — 两个主曲率（沿 principal directions）
- $a_{1,1}$ — principal directions 之间的混合，影响主方向估计

由这 6 个系数用标准 Monge patch 公式（论文 Appendix C）求出 principal curvatures $\kappa_\text{max}, \kappa_\text{min}$ 和 principal directions。torus 的 major/minor radii 和 axis direction由 principal curvatures/directions 直接给出。

**Sign 决定**：torus osculates surface from interior or exterior (Figure 5.4 右)，决定 $g_i = \text{sign}(\mathbb{T}_i)\phi_{\mathbb{T}_i}$。

### 4.5 用小型 transformer 学习 per-neighborhood 系数

为什么不手工 fit？Section 5.5.2 论证：classical point set 方法（MLS, RBF）的 neighborhood bandwidth 选择极 brittle（Figure 5.17：没有任何 $\sigma$ 能给出好结果）。Learning 解决：用 fixed-size neighborhoods + 神经网络拟合大量数据上的最佳系数。

网络架构（Figure 5.6）：
- 输入：$N_k(p_i)$ — $p_i$ 与 $k$ 最近邻的局部点集（$k=64$）
- Transformer blocks（[Vaswani et al. 2017](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)）学习哪些 neighbor 重要
- 输出：6 个 polynomial 系数 $a_{0,0}, a_{0,1}, a_{1,0}, a_{1,1}, a_{0,2}, a_{2,0}$

**Loss**:

$$
\mathcal{L} = \mathcal{L}_\text{distance} + \mathcal{L}_\text{eikonal}
\tag{5.5}
$$

$$
\mathcal{L}_\text{distance} = \frac{1}{Q}\sum_{j=1}^Q |\phi(q_j) - \phi_\text{true}(q_j)|, \quad \mathcal{L}_\text{eikonal} = \frac{1}{Q}\sum_{j=1}^Q |1 - \|\nabla\phi(q_j)\||
$$

eikonal loss 隐式鼓励 neighborhoods 平滑 blend 在一起。训练数据：CAD 模型 ([Koch et al. 2019 ABC dataset](https://deep-geometry.github.io/abc-dataset/)) + procedurally generated shapes, 共 40M training examples, 45 小时 on RTX 3090. 网络仅 6.4MB on disk.

### 4.6 设置 λ：数值稳定 + exponent shift

直接用 Equation 5.1 在 far-from-cloud 区域会有 catastrophic underflow。用 shifted formula：

$$
\phi(x) = \frac{\sum_{i=1}^{|P|} g_i(x)\exp(-\lambda_x(\|x-p_i\| - \sigma_x))}{\sum_{i=1}^{|P|}\exp(-\lambda_x(\|x-p_i\| - \sigma_x))}
\tag{5.6}
$$

$$
\sigma_x := \frac{1}{2}\max_{p_i\in P}\|x-p_i\|, \quad \lambda_x := \frac{C}{\sigma_x}, \quad C = 64
\tag{5.7}
$$

- $\sigma_x$ — per-query shift, 使 exp argument 在 0 附近居中
- $C=64$ — 使得 $\exp(-C)$ 不超过 single-precision 机器精度（最大约 87）
- 数值稳定且不改变结果（[Blanchard et al. 2020](https://doi.org/10.1093/imanum/draa038)）

### 4.7 性能 vs accuracy 对比

Figure 5.12 - 5.14 给出 4 个数据集上的结果（512-point point clouds）：

| Method | Precompute | Per-query | 平均 SDF error | 备注 |
|---|---|---|---|---|
| **PAT** | 一次 per-cloud | $7.4\times10^{-5}$ s (MacBook) | **最低** across datasets | 一次性预训练 |
| SHM (grid) | 全局 meshing + solve | 全局 | 较高 (sparse data) | FEM 限制 |
| SHC (Eq 5.8) | 无 | 快 | brittle | tangent-plane 失败 |
| SSPD (Eq 5.9) | 无 | 快 | brittle | pseudonormal 失败 |
| SPSR + mesh-SDF | expensive | fast | 略低于 PAT (除 ABC) | 多步 pipeline |
| NN-VIPSS | expensive | 中 | 高质量但贵 | [Xia & Ju 2025](https://doi.org/10.1145/3731191) |

29M 点的 David scan: 12.5 min precompute, **4ms per query**。

### 4.8 PAT 的局限

- **Out-of-distribution**: 训练数据 2048 points, 在 [−1, 1]³ 内。极密点云上 fitted tori 可能塌缩为小球（Figure inset），呈现"葡萄串"。
- **Noisy point clouds**: 网络训练时**不能带 noise**（otherwise 训练崩溃），因为 curvature estimation 是 ill-conditioned 问题。这是 PAT 的真正短板。
- **Non-conservative for perfect geometry**: 自归一化公式给出 distance overestimate 而非 underestimate（与 LogSumExp 相反）。
- **不能保证 exact interpolation**: 因为公式是 $\mathcal{C}^\infty$，无法精确 enforce $\phi=0$ on $P$。

### 4.9 PAT 与 Neural Fields 的对比

PAT 不像 DeepSDF ([Park et al. 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Park_DeepSDF_Learning_Continuous_Signed_Distance_Functions_for_Shape_Representation_CVPR_2019_paper.pdf)) 或 IGR ([Gropp et al. 2020](https://arxiv.org/abs/2002.10099)) 那样 per-shape fit 一个 neural field。PAT 的网络是 **per-neighborhood** 共享，预训练一次后任何 point cloud 都直接 forward pass。这避免了 eikonal 的非线性优化陷阱、per-shape training cost、和 scalability 限制——这是论文 Section 5.5.3 的核心论点。Section 5.5.5 提出 "small, judicious use of learning" philosophy：data-driven 用于补 classical formula 缺的 representational power，classical formula 提供收敛性和结构。

---

## 5. Chapter 6: Surface Winding Numbers (SWN)

### 5.1 问题：surface 上 "inside/outside" 的根本模糊

在 $\mathbb{R}^d$，closed curves/surfaces 总是 bound region (Jordan-Brouwer)。在一般 surface $M$ 上**不一定**——存在 **nonbounding loops** (non-nullhomologous in $H_1(M)$)。例如 torus 上绕洞一圈的 loop 不 bound 任何 region (Figure 2.1)。

所以 winding number 的标准定义 (Equation 6.1) 不能直接推广到 surfaces——integral 在 nonbounding loops 上对任意 closed curve $\Gamma'$ 给 0 而不是 ±1。

### 5.2 SWN 核心思路：jump harmonic + Hodge decomposition

利用 winding number = jump harmonic function (Section 2.3.3)。先解 **jump Laplace equation**:

$$
\Delta u = 0 \text{ on } M\setminus\Gamma, \quad u^+ - u^- = 1 \text{ on }\Gamma, \quad \partial u^+/\partial n = \partial u^-/\partial n \text{ on }\Gamma
\tag{6.2}
$$

但若 $M$ 不 simply-connected，$u$ 不一定是 piecewise constant 区域标签 (Figure 6.1)。需要**区分**：
- **Bounding components** of $\Gamma$ — nullhomologous in $H_1(M)$
- **Nonbounding components** — encoded in harmonic 1-form

### 5.3 Darboux derivative：forgetting jumps

对 jump harmonic function $u$，**Darboux derivative** $\mathscr{D}u$ 是 "*du modulo jumps"——考虑 $u$ 在每个连续区域内的标准 exterior derivative $du$，跨过 $\Gamma$ 时**忽略 jump**。

1D 例子（论文 inset）：
$$
f'(x) = \omega(x) + \sum_i \Lambda_i \delta_{x_i}
$$
- $\omega$ — piecewise smooth 部分（连续 part）
- $\Lambda_i\delta_{x_i}$ — Dirac delta at jumps，magnitude $\Lambda_i$

Darboux derivative 只保留 $\omega = \mathscr{D}f$。**关键性质**：jump harmonic 的 Darboux derivative 全局连续（harmonicity forces derivatives to match across jumps）。

逆 Darboux 是不唯一的——可以"pick jumps"。这个自由度就是 SWN 用来识别 nonbounding components 的工具。

### 5.4 SWN 算法

1. 解 Equation 6.2 得到 $u$
2. 计算 Darboux derivative $\omega := \mathscr{D}u$ — 一个 1-form
3. **Hodge decomposition** (Equation 2.1): $\omega = d\alpha + \delta\beta + \gamma$
   - 实际只需解 $\Delta_2\beta = d\omega$，然后 $\gamma = \omega - \delta\beta$
   - $\gamma$ 是 harmonic 1-form，编码 nonbounding components
4. **最小 L¹ 跳跃 integration**：找 $v$ 使得 Darboux derivative 看起来像 $\gamma$ — 即 $v$ 跨过 $\Gamma$ 的 nonbounding 部分。这是 |F| 个 DOF 的 sparse linear program
5. $\mathcal{T}v$ 给出 nonbounding components 的完成（locus of jumps），$\Gamma \setminus \mathcal{T}v$ 是 bounding components
6. 用 $\Gamma \setminus \mathcal{T}v$ 上的 jumps 重新解 Equation 6.2，得到 final winding number function $w$，round 到整数

### 5.5 算法的精度与速度

934 个 test cases, 451 在非 simply-connected 表面：

| Surface type | Mean error | Max error | 时间 |
|---|---|---|---|
| Simply-connected | 0.14% | 5% | < 2 s |
| Non-simply-connected | < 0.5% (80% of models) | — | up to few minutes (full LP) |

Naive rounding of $u$ (不过滤 nonbounding components) 在 ~10% examples 错分 > 50% 面积 (Figure 6.8)。

### 5.6 Reduced-size LP

用 Dijkstra 把 domain partition 成几个 connected components，只在 components 之间做 LP（DOFs 从 |F| 降到 components 数）。**100× speedup** with 可比 accuracy (Figure 6.12)。

### 5.7 与 SHM 的整合：piecewise continuous distance

当 curve 根本没有 inside/outside（例如 genus>0 surface 上的 nonbounding loop），SHM 的 L² integration 产生 "extremely warped isolines" (Figure 6.13)。解决方案：用 **L¹ integration** — linear program 允许 $\phi$ 在 edge 上 jump，penalize jump length。这个 LP 与 SWN 的 LP 结构相同，统一了两个算法。

---

## 6. Chapter 7: 实用对比与 corruption 实验

### 6.1 三种方法的 completion priors

| Method | 完成策略 | 在 hole 上表现 |
|---|---|---|
| GWN | 调和延拓，cap off 用 circular arcs | saddle-shaped patches，normal 不连续 |
| SHM | 调和延拓 + normal continuation | flat patches 但可能 "pointy" at large holes |
| PAT | 纯局部，无 global 补全 | 大 hole 不补；neighbor 不一致处 discontinue |

Figure 7.5 是 systematic study of 三种 prior 在 growing hole 上的行为。

### 6.2 Cylinder/Sphere corruption experiments

Figures 7.7-7.8 给出 signed distance error landscape over four corruption types (density, noise, outliers, orientation) × aspect ratio / radius。**所有方法对 outliers 最敏感**——error landscape 在 outliers > 0 时有 sharp discontinuity。**SPSR 是唯一 outlier 平滑增长的**——其他方法 cliff-edge。

| Method | Gaussian noise 容忍 | Orientation flips 容忍 |
|---|---|---|
| SHM | up to ~10% of radius | ~30% |
| SPSR | up to ~10% of radius | ~30% |
| GWN | degrade progressively | ~20% |
| PAT | degrade progressively (但退化为"葡萄串") | brittle |

### 6.3 Recommendation

干净几何用 [FCPW](https://github.com/rohan-sawhney/fcpw) + winding number；broken geometry 用 SHM 或 SPSR；sparse point cloud 用 PAT（带 GPU）；naive convolutional formulas (Equations 3.15, 3.16) **只在 dense image contour 上才有意义**。

---

## 7. 我对这篇论文的几点观察

### 7.1 真正的核心贡献是 §3.1 的 unification

许多人把 winding numbers 和 signed distance 当作不同问题。论文用 screening 参数 λ 把它们绑成同一族 PDE——这是 conceptual framework 的贡献。**真正难的不是这个 unification 本身，而是 §3.3 的 negative result**：naive Hopf-Cole 在点云上完全失败。这个失败（"λ 既耦合 regression 又耦合 distance，方向相反"）才是 SHM 与 PAT 设计的根本驱动力。

### 7.2 SHM 与 vector diffusion 的深刻联系

SHM 的真正聪明之处：用 **vector diffusion 替代 scalar diffusion** 把 Equation 3.15 的失败 escape 掉。Laplace 方法让 scalar diffusion 在 $\lambda\to\infty$ 下塌缩到 nearest point，但 vector diffusion 保留 orientation 信息——orientations 即使 magnitude 衰减也仍然有意义，可以被 normalize 后重新用作 SDF gradient。这是论文 Section 4.2 的"eureka moment"——parallel transport 的渐近行为（Berline et al. Thm 2.30）提供了 rigorous justification。

### 7.3 PAT 与 generative ML 的呼应

Section 3.4 把 naive convolutional formula 的失败与 flow matching 的 overfitting/memory 联系起来，这个 connection 是论文最 visionary 的部分。**Flow matching 用高斯核时就是 kernel density estimator**，注定 memorize training set——现在 SOTA 是用巨大神经网络 overregularize score function。PAT 提供另一种 path：**用 learned local 表达 replace 标量 kernel**，类似 Bamberger et al. 2025 用 anisotropic Gaussian 估 tangent space。这种"local geometric structure beats global regularization"的哲学在 generative modeling 中可能也成立——值得 attention。

### 7.4 SWN 与 cohomology 的处理

SWN 的 elegant 处在于把 "这个 curve 是 bounding 还是不 bounding" 这个**离散拓扑问题**转化为 **Hodge decomposition 的 harmonic component** —— 一个 linear algebra 问题。Darboux derivative 是"forget jumps" 这个 topological/differential trick 的具体实现，inverse Darboux 的 non-uniqueness 正好对应 curve decomposition 的 ambiguity，用 $L^1$ minimization 挑选最短非 bounding 部分。

### 7.5 几个开放方向

- **N-ary distance** (Section 8.2): 现有 SDF 都 binary。多个 region 需要独立 SDFs。
- **Closed-form screened winding numbers**: 像 [Liu et al. 2025](https://doi.org/10.1145/3730886) 那样对 rational parametric curves 推 closed-form。
- **Statistical geometry processing**: 用 Gaussian process 给 winding number uncertainty quantification（[Sellán & Jacobson 2022](https://doi.org/10.1145/3550454.3555441)），但 SDF / PAT 因非线性 normalization 难做。
- **Geometric vs data-driven**: PAT 用了 minimum learning，提供 "small, judicious use of learning" template——这对 foundation model 时代是反向信号，但很 sound。

---

## 8. 参考链接

主要 paper:
- 论文 PDF: [CMU-CS-26-104](https://www.cs.cmu.edu/~kmcrane/) (将上传 thesis 服务器)
- Signed Heat Method (SHM): [nzfeng.github.io/research/SignedHeatMethod](https://nzfeng.github.io/research/SignedHeatMethod/index.html), [DOI 10.1145/3658220](https://doi.org/10.1145/3658220)
- Surface Winding Numbers (SWN): [nzfeng.github.io/research/WNoDS](https://nzfeng.github.io/research/WNoDS/index.html), [DOI 10.1145/3592401](https://doi.org/10.1145/3592401)
- Points as Tori (PAT): [nzfeng.github.io/research/PointsAsTori](https://nzfeng.github.io/research/PointsAsTori/index.html), [DOI 10.1145/3811385](https://doi.org/10.1145/3811385)
- Code: [github.com/nzfeng/signed-heat-python](https://github.com/nzfeng/signed-heat-python)

背景与相关工作:
- Keenan Crane's Geometry Collective: [geometrycollective.github.io](https://geometrycollective.github.io/)
- Heat Method for Geodesics: [DOI 10.1145/2516971.2516977](https://doi.org/10.1145/2516971.2516977)
- Vector Heat Method: [DOI 10.1145/3243651](https://doi.org/10.1145/3243651)
- Discrete Exterior Calculus: [DOI 10.1145/2504435.2504442](https://doi.org/10.1145/2504435.2504442)
- Generalized Winding Number: [DOI 10.1145/2461912.2461916](https://doi.org/10.1145/2461912.2461916)
- Fast Winding Numbers: [DOI 10.1145/3197517.3201337](https://doi.org/10.1145/3197517.3201337)
- Screened Poisson Surface Reconstruction: [DOI 10.1145/2487228.2487237](https://doi.org/10.1145/2487228.2487237)
- Lipman 2021 Phase Transitions: [proceedings.mlr.press/v139/lipman21a](https://proceedings.mlr.press/v139/lipman21a.html)
- Berline, Getzler, Vergne: [Springer book](https://link.springer.com/book/10.1007/978-3-662-03580-2)
- IQ distance functions reference: [iquilezles.org/articles/distgradfunctions3d](https://iquilezles.org/articles/distgradfunctions3d/)
- FCPW (fast closest points): [github.com/rohan-sawhney/fcpw](https://github.com/rohan-sawhney/fcpw)
- ABC dataset: [deep-geometry.github.io/abc-dataset](https://deep-geometry.github.io/abc-dataset/)
- ADMM distance (Belyaev-Fayolle): [DOI 10.1007/s11075-019-00789-5](https://doi.org/10.1007/s11075-019-00789-5)
- Varadhan 1967: [DOI 10.1002/cpa.3160200206](https://doi.org/10.1002/cpa.3160200206)
- NKSR (Huang 2023): [arXiv 2302.13835](https://arxiv.org/abs/2302.13835)
- DeepSDF (Park 2019): [arXiv 1901.05103](https://arxiv.org/abs/1901.05103)
- IGR (Gropp 2020): [arXiv 2002.10099](https://arxiv.org/abs/2002.10099)
- NN-VIPSS (Xia & Ju 2025): [DOI 10.1145/3731191](https://doi.org/10.1145/3731191)
- Carré du champ flow matching (Bamberger 2025): [arXiv 2510.05930](https://arxiv.org/abs/2510.05930)
- Lipman 2021: [proceedings.mlr.press/v139/lipman21a.html](https://proceedings.mlr.press/v139/lipman21a.html)
- Vaswani 2017 Attention: [NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

这篇论文体现了 CMU Geometry Collective 一贯的 "mathematical structure first, algorithm follows" 美学——先把 winding number / Poisson / SDF 在 λ 上统一起来，发现 asymptotic 失败模式，再用 vector diffusion 和 learned local kernels 跳出失败区域。SHM 是"换 PDE 数据类型 (scalar→vector)"的解决方案；PAT 是"换 kernel 复杂度 (constant→learned local)"的解决方案；SWN 是"换 algebraic framework (real→homology class)"的解决方案。三个算法都用了**"function 比 geometry 友好"**这个中心主张，diffusion 出现在三处都非偶然——它 connect 了 parallel transport、Varadhan/Hopf-Cole、和 de Rham cohomology，是整篇论文的潜在主旋律。
