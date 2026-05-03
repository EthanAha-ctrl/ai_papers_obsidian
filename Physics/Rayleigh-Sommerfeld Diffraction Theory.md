# Rayleigh-Sommerfeld Diffraction Theory 详解

## 一、历史背景与核心问题

### 为什么需要这个理论？

当 light 通过 aperture（孔径）或遇到 obstacle（障碍物）时，会发生 **diffraction（衍射）** 现象——light 不再沿直线传播，而是"绕弯"进入 geometric shadow（几何阴影）区域。

早在 17 世纪，**Christiaan Huygens** 就提出了一个直觉性的想法：
> 波前上的每一点都可以看作是一个 secondary wavelet（次级子波）的 source，这些子波的 envelope（包络面）形成新的 wavefront。

这个想法很美，但有一个致命问题：**没有数学形式化**。它无法精确预测 diffraction pattern（衍射图样）中各点的 light intensity（光强）分布。

后来 **Augustin-Jean Fresnel** 用数学语言重新表述了 Huygens 的想法，加入 **interference（干涉）** 的概念，并引入 **obliquity factor（倾斜因子）** 来修正子波强度的角度依赖性。这成为了著名的 **Huygens-Fresnel Principle**。

但 Fresnel 的理论仍然有两个问题：
1. **Obliquity factor 是"凑"出来的**，没有物理推导
2. **Backward propagating wave（后向传播波）** 无法被合理消除

这就引出了我们需要严格数学框架的需求——**Kirchhoff Diffraction Theory** 以及后来的 **Rayleigh-Sommerfeld Diffraction Theory**。

---

## 二、第一性原理：从 Maxwell Equations 到 Scalar Wave Equation

### 2.1 Maxwell Equations 出发

在 free space（自由空间）中，**Maxwell equations** 的 differential form（微分形式）为：

$$\nabla \cdot \mathbf{E} = 0$$

$$\nabla \cdot \mathbf{B} = 0$$

$$\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}$$

$$\nabla \times \mathbf{B} = \mu_0 \epsilon_0 \frac{\partial \mathbf{E}}{\partial t}$$

其中：
- $\mathbf{E}$ = electric field vector（电场矢量）
- $\mathbf{B}$ = magnetic field vector（磁场矢量）
- $\mu_0$ = permeability of free space（真空磁导率）
- $\epsilon_0$ = permittivity of free space（真空介电常数）
- $t$ = time（时间）

### 2.2 推导 Wave Equation

对第三个方程取 curl（旋度）：

$$\nabla \times (\nabla \times \mathbf{E}) = -\frac{\partial}{\partial t}(\nabla \times \mathbf{B})$$

利用 vector identity（矢量恒等式）：
$$\nabla \times (\nabla \times \mathbf{E}) = \nabla(\nabla \cdot \mathbf{E}) - \nabla^2 \mathbf{E}$$

由于 $\nabla \cdot \mathbf{E} = 0$，且代入第四个方程：

$$-\nabla^2 \mathbf{E} = -\mu_0 \epsilon_0 \frac{\partial^2 \mathbf{E}}{\partial t^2}$$

整理得：
$$\nabla^2 \mathbf{E} - \frac{1}{c^2}\frac{\partial^2 \mathbf{E}}{\partial t^2} = 0$$

其中 $c = \frac{1}{\sqrt{\mu_0 \epsilon_0}}$ 是 speed of light（光速）。

### 2.3 Scalar Approximation（标量近似）

如果 light 的 polarization（偏振）效应在 diffraction 过程中不显著（这在大多数 aperture 尺寸远大于 wavelength 时成立），我们可以用一个 **scalar field（标量场）** $U(\mathbf{r}, t)$ 来描述光的振动：

$$\nabla^2 U - \frac{1}{c^2}\frac{\partial^2 U}{\partial t^2} = 0$$

对于 **monochromatic wave（单色波）**，采用 time-harmonic form（时间谐振形式）：

$$U(\mathbf{r}, t) = \text{Re}\{u(\mathbf{r}) e^{-i\omega t}\}$$

其中：
- $u(\mathbf{r})$ = complex amplitude（复振幅），是位置 $\mathbf{r}$ 的函数
- $\omega$ = angular frequency（角频率）
- $i$ = imaginary unit（虚数单位）

代入后得到 **Helmholtz equation**：

$$\boxed{(\nabla^2 + k^2)u(\mathbf{r}) = 0}$$

其中 $k = \frac{\omega}{c} = \frac{2\pi}{\lambda}$ 是 **wave number（波数）**，$\lambda$ 是 **wavelength（波长）**。

---

## 三、Green's Function Method：数学工具的核心

### 3.1 Green's Theorem（格林定理）

这是整个 diffraction theory 的数学基石。设 $u(\mathbf{r})$ 和 $g(\mathbf{r})$ 是两个在 volume $V$ 内有连续二阶导数的 scalar functions，则：

$$\iiint_V (u\nabla^2 g - g\nabla^2 u) \, dV = \oiint_S \left(u\frac{\partial g}{\partial n} - g\frac{\partial u}{\partial n}\right) dS$$

其中：
- $S$ = surface enclosing volume $V$（包围体积 $V$ 的闭合曲面）
- $\frac{\partial}{\partial n}$ = normal derivative outward（外法向导数）
- $dV$ = volume element（体积元）
- $dS$ = surface element（面积元）

### 3.2 Green's Function 的定义

**Green's function** $G(\mathbf{r}, \mathbf{r}')$ 是 Helmoltz equation 在 point source 下的解：

$$(\nabla^2 + k^2)G(\mathbf{r}, \mathbf{r}') = -\delta(\mathbf{r} - \mathbf{r}')$$

其中 $\delta(\mathbf{r} - \mathbf{r}')$ 是 **Dirac delta function（狄拉克δ函数）**，表示在 $\mathbf{r}'$ 处有一个 point source。

### 3.3 Free-space Green's Function

在 free space 中，Green's function 的解是著名的 **spherical wave（球面波）**：

$$\boxed{G(\mathbf{r}, \mathbf{r}') = \frac{e^{ik|\mathbf{r} - \mathbf{r}'|}}{4\pi|\mathbf{r} - \mathbf{r}'|}}$$

这个形式非常重要，它表示：
- 从 $\mathbf{r}'$ 发出的 spherical wave
- Amplitude 以 $1/r$ 衰减（能量守恒的要求）
- Phase 随距离增加而累积

---

## 四、Kirchhoff Diffraction Theory：先驱工作的成功与缺陷

### 4.1 Kirchhoff Integral Formula（基尔霍夫积分公式）

利用 Green's theorem 和 Green's function，可以推导出著名的 **Kirchhoff integral formula**：

$$u(P) = \frac{1}{4\pi}\oiint_S \left[\frac{e^{ikr}}{r}\frac{\partial u}{\partial n} - u\frac{\partial}{\partial n}\left(\frac{e^{ikr}}{r}\right)\right] dS$$

其中：
- $u(P)$ = field at observation point $P$（观测点 $P$ 的场）
- $r$ = distance from surface element to $P$（从面积元到 $P$ 的距离）
- $\frac{\partial}{\partial n}$ = normal derivative（法向导数）

这个公式告诉我们：**任意闭合曲面上的场分布，可以完全决定曲面内任意一点的场**。这就是 **Huygens' Principle 的严格数学表述**。

### 4.2 Kirchhoff Boundary Conditions（基尔霍夫边界条件）

对于实际的 diffraction problem，我们考虑一个 **opaque screen（不透明屏幕）** 上有一个 aperture $\Sigma$。Kirchhoff 提出：

1. **在 aperture 内（$\Sigma$ 上）**：场 $u$ 和其法向导数 $\frac{\partial u}{\partial n}$ 与没有 screen 时完全相同
2. **在 screen 的 opaque 部分**：场 $u = 0$ 且 $\frac{\partial u}{\partial n} = 0$

### 4.3 Kirchhoff Theory 的数学矛盾

这看似合理的边界条件，却存在一个严重的数学问题：

**问题核心**：如果 $u$ 和 $\frac{\partial u}{\partial n}$ 在一个非零面积的 surface 上都为零，根据 **uniqueness theorem（唯一性定理）** of Helmholtz equation，则整个空间中的 $u$ 必须恒为零！

这意味着：
- 数学上，Kirchhoff boundary conditions 是 **over-specified（过定）** 的
- 你不能同时指定 $u$ 和 $\frac{\partial u}{\partial n}$ 的值

### 4.4 另一个问题：Backward Wave

Kirchhoff formula 中使用的是 **free-space Green's function**：
$$G(\mathbf{r}, \mathbf{r}') = \frac{e^{ik|\mathbf{r} - \mathbf{r}'|}}{4\pi|\mathbf{r} - \mathbf{r}'|}$$

这个函数在 source point $\mathbf{r}'$ 处有 **singularity（奇点）**。当 $\mathbf{r}'$ 位于 surface $S$ 上时，需要特殊处理。

更重要的是，这个公式不能自然地消除 **backward propagating wave（后向传播波）**——也就是说，理论上存在一个向 screen 后方传播的波，这与物理直觉不符。

---

## 五、Rayleigh-Sommerfeld Diffraction Theory：解决方案

**Lord Rayleigh（瑞利勋爵）** 和 **Arnold Sommerfeld（阿诺德·索末菲）** 各自独立地提出了两种不同的方法来消除 Kirchhoff theory 的矛盾，这就是 **Rayleigh-Sommerfeld Diffraction Theory**。

### 5.1 核心思想：使用不同的 Green's Function

关键 insight：与其使用 free-space Green's function，不如构造一个 **满足特定边界条件的 Green's function**。

我们要求新的 Green's function $G_{RS}$ 满足：

$$(\nabla^2 + k^2)G_{RS}(\mathbf{r}, \mathbf{r}') = -\delta(\mathbf{r} - \mathbf{r}')$$

并且在 screen surface $S_0$（假设为 $z = 0$ 平面）上满足：

- **First kind（第一类）**：$G_{RS}^{(1)} = 0$ on $S_0$
- **Second kind（第二类）**：$\frac{\partial G_{RS}^{(2)}}{\partial n} = 0$ on $S_0$

这样，在应用 Green's theorem 时，公式中对应项会自动消失，从而避免了 over-specification 的问题！

---

## 六、Rayleigh-Sommerfeld First Kind Formula

### 6.1 构造 Green's Function $G^{(1)}$

利用 **method of images（镜像法）**，构造：

$$G^{(1)}(\mathbf{r}, \mathbf{r}') = \frac{e^{ik|\mathbf{r} - \mathbf{r}'|}}{4\pi|\mathbf{r} - \mathbf{r}'|} - \frac{e^{ik|\mathbf{r} - \tilde{\mathbf{r}}'|}}{4\pi|\mathbf{r} - \tilde{\mathbf{r}}'|}$$

其中 $\tilde{\mathbf{r}}'$ 是 $\mathbf{r}'$ 关于 screen plane $z = 0$ 的 **mirror image（镜像点）**：

$$\tilde{\mathbf{r}}' = (x', y', -z')$$

**验证边界条件**：
在 $z = 0$ 平面上，$|\mathbf{r} - \mathbf{r}'| = |\mathbf{r} - \tilde{\mathbf{r}}'|$，所以：

$$G^{(1)}|_{z=0} = \frac{e^{ikr}}{4\pi r} - \frac{e^{ikr}}{4\pi r} = 0$$

边界条件自动满足！

### 6.2 First Kind Diffraction Formula

将这个 Green's function 代入 Green's theorem，经过推导得到：

$$\boxed{u(P) = -\frac{1}{2\pi}\iint_{\Sigma} u(x_0, y_0, 0) \cdot \frac{\partial}{\partial z}\left(\frac{e^{ikr}}{r}\right) dx_0 dy_0}$$

其中：
- $u(P)$ = complex amplitude at observation point $P$（观测点的复振幅）
- $\Sigma$ = aperture area（孔径区域）
- $u(x_0, y_0, 0)$ = field on aperture plane（孔径平面上的场）
- $r = \sqrt{(x - x_0)^2 + (y - y_0)^2 + z^2}$ = distance from aperture point to observation point（从孔径点到观测点的距离）
- $z$ = perpendicular distance from aperture to observation plane（从孔径到观测平面的垂直距离）

### 6.3 计算法向导数

$$\frac{\partial}{\partial z}\left(\frac{e^{ikr}}{r}\right) = \frac{e^{ikr}}{r}\left(ik - \frac{1}{r}\right)\frac{\partial r}{\partial z}$$

其中：
$$\frac{\partial r}{\partial z} = \frac{z}{r} = \cos\theta$$

$\theta$ 是 propagation direction 与 $z$-axis 的夹角。

因此：
$$\frac{\partial}{\partial z}\left(\frac{e^{ikr}}{r}\right) = \frac{e^{ikr}}{r}\left(ik - \frac{1}{r}\right)\cos\theta$$

### 6.4 物理意义解读

First kind formula 的边界条件是：
- **指定 aperture 上的 field $u$**
- **不指定 $\frac{\partial u}{\partial n}$**

这对应于一个 **soft aperture（软孔径）** 或 **透明 aperture**，其中 field 可以自由"穿过" aperture 而不改变其 amplitude。

---

## 七、Rayleigh-Sommerfeld Second Kind Formula

### 7.1 构造 Green's Function $G^{(2)}$

使用 method of images，但这次用 **加法** 而非减法：

$$G^{(2)}(\mathbf{r}, \mathbf{r}') = \frac{e^{ik|\mathbf{r} - \mathbf{r}'|}}{4\pi|\mathbf{r} - \mathbf{r}'|} + \frac{e^{ik|\mathbf{r} - \tilde{\mathbf{r}}'|}}{4\pi|\mathbf{r} - \tilde{\mathbf{r}}'|}$$

**验证边界条件**：

计算法向导数（注意法向是 $-z$ 方向，因为 screen 在 $z > 0$ 区域）：

$$\frac{\partial G^{(2)}}{\partial n}\bigg|_{z=0} = -\frac{\partial G^{(2)}}{\partial z}\bigg|_{z=0}$$

在 $z = 0$ 平面上：
$$\frac{\partial}{\partial z}\left[\frac{e^{ikr}}{r} + \frac{e^{ikr'}}{r'}\right]\bigg|_{r=r'}$$

其中 $r = \sqrt{(x-x')^2 + (y-y')^2 + z^2}$, $r' = \sqrt{(x-x')^2 + (y-y')^2 + (-z)^2}$。

由于 $z = 0$ 时，$\frac{\partial r}{\partial z} = \frac{z}{r} = 0$ 且 $\frac{\partial r'}{\partial z} = \frac{-z}{r'} = 0$，所以：

$$\frac{\partial G^{(2)}}{\partial z}\bigg|_{z=0} = 0$$

因此 $\frac{\partial G^{(2)}}{\partial n} = 0$，边界条件满足！

### 7.2 Second Kind Diffraction Formula

$$\boxed{u(P) = \frac{1}{2\pi}\iint_{\Sigma} \frac{\partial u}{\partial n}\bigg|_{z=0} \cdot \frac{e^{ikr}}{r} dx_0 dy_0}$$

其中：
- $\frac{\partial u}{\partial n}\big|_{z=0}$ = normal derivative of field on aperture（孔径上场的法向导数）

### 7.3 物理意义解读

Second kind formula 的边界条件是：
- **指定 aperture 上的 $\frac{\partial u}{\partial n}$**
- **不指定 $u$**

这对应于一个 **hard aperture（硬孔径）** 或 **反射型 aperture**。

---

## 八、两种 Rayleigh-Sommerfeld Formula 的比较

### 8.1 数学形式的对比

| Property | First Kind ($G^{(1)}$) | Second Kind ($G^{(2)}$) |
|----------|------------------------|-------------------------|
| Green's Function | $\frac{e^{ikr}}{4\pi r} - \frac{e^{ikr'}}{4\pi r'}$ | $\frac{e^{ikr}}{4\pi r} + \frac{e^{ikr'}}{4\pi r'}$ |
| Boundary Condition on $S_0$ | $G^{(1)} = 0$ | $\frac{\partial G^{(2)}}{\partial n} = 0$ |
| What is specified | $u$ on $\Sigma$ | $\frac{\partial u}{\partial n}$ on $\Sigma$ |
| Physical interpretation | Soft aperture | Hard aperture |
| Kernel in integral | $\frac{\partial}{\partial n}\left(\frac{e^{ikr}}{r}\right)$ | $\frac{e^{ikr}}{r}$ |

### 8.2 Obliquity Factor 的自然导出

这是 Rayleigh-Sommerfeld theory 最美妙的成果之一。

对于 **normal incidence plane wave** 照明 aperture：
$$u(x_0, y_0, 0) = A \quad \text{(constant amplitude)}$$

代入 first kind formula：

$$u(P) = -\frac{A}{2\pi}\iint_{\Sigma} \frac{\partial}{\partial z}\left(\frac{e^{ikr}}{r}\right) dx_0 dy_0$$

展开法向导数：

$$\frac{\partial}{\partial z}\left(\frac{e^{ikr}}{r}\right) = \frac{e^{ikr}}{r}\left(ik - \frac{1}{r}\right)\cos\theta$$

当 $r \gg \lambda$（far field approximation，即 $kr \gg 1$）：

$$\frac{\partial}{\partial z}\left(\frac{e^{ikr}}{r}\right) \approx ik\frac{e^{ikr}}{r}\cos\theta = \frac{ik}{r}e^{ikr}\cos\theta$$

因此：

$$u(P) \approx -\frac{ikA}{2\pi}\iint_{\Sigma} \frac{e^{ikr}}{r}\cos\theta \, dx_0 dy_0$$

我们自然地得到了 **obliquity factor**：

$$\boxed{K(\theta) = \cos\theta}$$

这正是 Fresnel 当年"凑"出来的结果！现在它有了严格的数学推导。

### 8.3 Forward vs Backward Propagation

**Rayleigh-Sommerfeld theory 的另一个重要优势**：它自然地消除了 backward wave。

在 first kind formulation 中，由于 $G^{(1)} = 0$ 在 screen plane 上，backward propagating wave 的贡献自动为零。这符合物理直觉——aperture 后方不应该有来自 aperture 的波传播。

---

## 九、与 Fresnel 和 Fraunhofer Diffraction 的关系

### 9.1 层次结构

```
                    Rayleigh-Sommerfeld Theory
                            (Exact, Scalar)
                                   |
                    ┌──────────────┴──────────────┐
                    |                             |
            Paraxial Approximation          Near-field
            (kr ≫ 1, small θ)              (Fresnel Region)
                    |                             |
                    ↓                             ↓
            Fresnel Diffraction            Fresnel Diffraction
            (Quadratic phase)              (Full formulation)
                    |
            Far-field Approximation
            (z ≫ aperture size)
                    |
                    ↓
            Fraunhofer Diffraction
            (Fourier Transform)
```

### 9.2 Fresnel Approximation

当 observation point 距离 aperture 远大于 aperture 尺寸时，对 distance $r$ 进行展开：

$$r = \sqrt{(x-x_0)^2 + (y-y_0)^2 + z^2} = z\sqrt{1 + \frac{(x-x_0)^2 + (y-y_0)^2}{z^2}}$$

使用 binomial expansion：
$$r \approx z + \frac{(x-x_0)^2 + (y-y_0)^2}{2z} - \frac{[(x-x_0)^2 + (y-y_0)^2]^2}{8z^3} + ...$$

保留到 quadratic term：
$$r \approx z + \frac{(x-x_0)^2 + (y-y_0)^2}{2z}$$

代入 Rayleigh-Sommerfeld first kind formula：

$$u(x,y,z) \approx -\frac{ikA\cos\theta}{2\pi z}e^{ikz}\iint_{\Sigma} e^{ik\frac{(x-x_0)^2+(y-y_0)^2}{2z}} dx_0 dy_0$$

这就是 **Fresnel diffraction integral**。

### 9.3 Fraunhofer Approximation

当 observation point 更远，以至于：

$$z \gg \frac{k(x_0^2 + y_0^2)_{max}}{2}$$

即 **Fresnel number（菲涅尔数）** $N_F = \frac{a^2}{\lambda z} \ll 1$（$a$ 是 aperture 半径）：

可以忽略 quadratic phase term in $x_0, y_0$，只保留 linear term：

$$(x-x_0)^2 + (y-y_0)^2 \approx x^2 + y^2 - 2(xx_0 + yy_0)$$

此时 diffraction integral 变成：

$$u(x,y,z) \approx C e^{ik(z + \frac{x^2+y^2}{2z})}\iint_{\Sigma} u(x_0, y_0, 0) e^{-i\frac{k}{z}(xx_0 + yy_0)} dx_0 dy_0$$

这正是 **2D Fourier Transform** of aperture function！

---

## 十、完整数学推导：从 Helmholtz Equation 到 Rayleigh-Sommerfeld Integral

### 10.1 Setup

考虑一个 opaque screen 在 $z = 0$ 平面，其上有一个 aperture $\Sigma$。Observation point $P$ 位于 $z > 0$ 半空间。

### 10.2 选择闭合曲面

选择闭合曲面 $S$ 由三部分组成：
1. $S_1$: aperture plane $z = 0$（包括 aperture $\Sigma$ 和 opaque part）
2. $S_2$: hemispherical surface at infinity（无穷远处半球面）
3. $S_3$: small sphere around observation point $P$（包围观测点的小球面）

### 10.3 应用 Green's Theorem

设 $u(\mathbf{r})$ 是我们要求解的场，$G(\mathbf{r}, \mathbf{r}_0)$ 是 Green's function。

在 $S$ 内部（不包括 $P$ 点），有：
$$\iiint_V (u\nabla^2 G - G\nabla^2 u) dV = \oiint_S \left(u\frac{\partial G}{\partial n} - G\frac{\partial u}{\partial n}\right) dS$$

由于 $u$ 和 $G$ 都满足 Helmholtz equation（在 $V$ 内），左边为零：
$$\oiint_S \left(u\frac{\partial G}{\partial n} - G\frac{\partial u}{\partial n}\right) dS = 0$$

### 10.4 分析各曲面贡献

**$S_2$ 的贡献**（Sommerfeld radiation condition）：
在无穷远处，满足 **radiation condition**：
$$\lim_{r\to\infty} r\left(\frac{\partial u}{\partial r} - iku\right) = 0$$
这保证了只有 outward propagating wave。$S_2$ 的贡献为零。

**$S_3$ 的贡献**（围绕 $P$ 点的小球）：
当小球半径 $\epsilon \to 0$：
$$\oiint_{S_3} \left(u\frac{\partial G}{\partial n} - G\frac{\partial u}{\partial n}\right) dS = -u(P)$$
负号是因为法向指向球心。

### 10.5 得到 Integral Formula

综合以上：

$$u(P) = \iint_{S_1} \left(u\frac{\partial G}{\partial n} - G\frac{\partial u}{\partial n}\right) dS$$

### 10.6 应用 Rayleigh-Sommerfeld First Kind Green's Function

使用 $G^{(1)}$ 满足 $G^{(1)}|_{z=0} = 0$：

$$u(P) = \iint_{S_1} u\frac{\partial G^{(1)}}{\partial n} dS = -\iint_{S_1} u\frac{\partial G^{(1)}}{\partial z} dS$$

负号来自法向定义。

代入：
$$G^{(1)} = \frac{e^{ikr}}{4\pi r} - \frac{e^{ikr'}}{4\pi r'}$$

在 $z = 0$ 平面上，$r = r'$，但对 $z$ 的导数：
$$\frac{\partial G^{(1)}}{\partial z}\bigg|_{z=0} = 2 \cdot \frac{\partial}{\partial z}\left(\frac{e^{ikr}}{4\pi r}\right)\bigg|_{z=0}$$

因此：
$$u(P) = -\frac{1}{2\pi}\iint_{\Sigma} u(x_0, y_0, 0) \frac{\partial}{\partial z}\left(\frac{e^{ikr}}{r}\right) dx_0 dy_0$$

**Q.E.D.**

---

## 十一、实验验证与数值计算

### 11.1 经典实验：Single Slit Diffraction

**Setup**：
- Wavelength: $\lambda = 632.8$ nm (He-Ne laser)
- Slit width: $a = 100$ $\mu$m
- Distance to screen: $z = 1$ m

**Fresnel number**：
$$N_F = \frac{a^2}{\lambda z} = \frac{(100 \times 10^{-6})^2}{632.8 \times 10^{-9} \times 1} = 0.0158$$

由于 $N_F \ll 1$，可以使用 Fraunhofer approximation。

**Intensity distribution**：
$$I(\theta) = I_0 \left(\frac{\sin\beta}{\beta}\right)^2$$

其中 $\beta = \frac{\pi a \sin\theta}{\lambda}$。

### 11.2 Near-field Case: Fresnel Diffraction

当 $N_F \sim 1$ 或更大时，必须使用完整的 Rayleigh-Sommerfeld 或 Fresnel integral。

**Example calculation** ($N_F = 5$):

对于 circular aperture of radius $a$，在 distance $z$：

$$N_F = \frac{a^2}{\lambda z} = 5$$

需要计算：
$$u(\rho, z) = \int_0^a \int_0^{2\pi} u_0(r_0) \frac{e^{ik\sqrt{r_0^2 + \rho^2 + z^2 - 2r_0\rho\cos\phi'}}}{\sqrt{r_0^2 + \rho^2 + z^2 - 2r_0\rho\cos\phi'}} kr_0 dr_0 d\phi'$$

其中 $\rho = \sqrt{x^2 + y^2}$ 是 radial coordinate。

### 11.3 数值方法：Angular Spectrum Method

对于计算 Rayleigh-Sommerfeld diffraction，一种高效的方法是 **Angular Spectrum Method（角谱法）**。

**Fourier domain representation**：
$$\tilde{u}(k_x, k_y, z) = \tilde{u}(k_x, k_y, 0) e^{ik_z z}$$

其中：
$$k_z = \sqrt{k^2 - k_x^2 - k_y^2}$$

**Evanescent waves（倏逝波）**：
当 $k_x^2 + k_y^2 > k^2$ 时，$k_z$ 变为虚数：
$$k_z = i\sqrt{k_x^2 + k_y^2 - k^2}$$

这些 waves 指数衰减，只在 near-field 有贡献。

---

## 十二、Vector Diffraction Theory：超越标量近似

### 12.1 标量近似的局限

Rayleigh-Sommerfeld theory 是 scalar theory，假设 polarization effect 可以忽略。但在以下情况失效：

1. **Aperture size comparable to wavelength**（$a \sim \lambda$）
2. **High-NA focusing**（高数值孔径聚焦）
3. **Strongly focused beams**（强聚焦光束）

### 12.2 Vector Kirchhoff Diffraction

需要使用 **vector diffraction theory**，从 Maxwell equations 直接出发。

**Electric field**：
$$\mathbf{E}(\mathbf{r}) = \frac{i}{\omega\epsilon_0}\oiint_S \left[(\mathbf{n} \times \mathbf{H})\nabla G + (\mathbf{n} \times \mathbf{H}) \cdot \nabla G\right] dS$$

这涉及 **Stratton-Chu formula** 或 **Jones vector** 方法。

### 12.3 Richards-Wolf Integral

对于 high-NA focusing，使用 **Richards-Wolf integral**：

$$\mathbf{E}(x,y,z) = \frac{ikf}{2\pi}\iint_{\Omega} \mathbf{a}(\theta,\phi) e^{ik(z\cos\theta + r\sin\theta\cos(\phi-\phi'))} \sqrt{\cos\theta}\sin\theta d\theta d\phi$$

其中：
- $f$ = focal length（焦距）
- $\theta, \phi$ = angular coordinates in pupil（光瞳上的角坐标）
- $\mathbf{a}(\theta,\phi)$ = amplitude vector（振幅矢量）

---

## 十三、现代应用

### 13.1 Digital Holography

在 **digital holography（数字全息）** 中，需要从 hologram 重建 object wave。Rayleigh-Sommerfeld integral 提供了精确的 reconstruction algorithm。

**Reconstruction formula**：
$$u(x,y,z) = -\frac{1}{2\pi}\iint_{\text{hologram}} I(x_0, y_0) \frac{\partial}{\partial z}\left(\frac{e^{ikr}}{r}\right) dx_0 dy_0$$

### 13.2 Optical Tweezers

**Optical tweezers（光镊）** 使用 tightly focused laser beam 来 trap 微粒。计算 focus point 的 field 需要精确的 diffraction integral。

### 13.3 Metasurface Design

**Metasurfaces（超表面）** 是人工设计的 2D 结构，用于控制 wavefront。设计过程需要精确计算 sub-wavelength structures 的 diffraction pattern。

### 13.4 Computational Lithography

在 semiconductor manufacturing 中，**optical lithography（光刻）** 的 resolution 受 diffraction 限制。Rayleigh-Sommerfeld theory 用于计算 resist 上的 aerial image。

---

## 十四、Summary：建立你的 Intuition

### 14.1 核心要点总结

| Concept | Intuition |
|---------|-----------|
| **Huygens' Principle** | 每个 wavefront 上的点都是 secondary source |
| **Green's Function** | Point source 的响应函数，是 propagator |
| **Rayleigh-Sommerfeld vs Kirchhoff** | RS 用特殊 Green's function 避免过定边界条件 |
| **Two kinds of RS** | First kind: specify $u$; Second kind: specify $\frac{\partial u}{\partial n}$ |
| **Obliquity factor** | $\cos\theta$，natural result from RS theory |
| **Fresnel/Fraunhofer** | Approximations of RS under different conditions |

### 14.2 Mathematical Hierarchy

```
Exact solution: Rayleigh-Sommerfeld (scalar)
       ↓ (paraxial: kr ≫ 1)
Fresnel integral (quadratic phase)
       ↓ (far-field: z ≫ a²/λ)
Fraunhofer integral (Fourier transform)
```

### 14.3 关键公式速查

**Rayleigh-Sommerfeld First Kind**:
$$u(P) = -\frac{1}{2\pi}\iint_{\Sigma} u_0 \cdot \frac{\partial}{\partial z}\left(\frac{e^{ikr}}{r}\right) dS$$

**Rayleigh-Sommerfeld Second Kind**:
$$u(P) = \frac{1}{2\pi}\iint_{\Sigma} \frac{\partial u}{\partial n} \cdot \frac{e^{ikr}}{r} dS$$

**Fresnel Diffraction**:
$$u(x,y,z) = \frac{e^{ikz}}{i\lambda z}\iint u_0(x',y') e^{i\frac{\pi}{\lambda z}[(x-x')^2+(y-y')^2]} dx'dy'$$

**Fraunhofer Diffraction**:
$$u(x,y,z) = \frac{e^{ikz}e^{i\frac{k}{2z}(x^2+y^2)}}{i\lambda z}\tilde{u}_0\left(\frac{kx}{z}, \frac{ky}{z}\right)$$

---

## 参考文献与延伸阅读

### 经典教材

1. **Goodman, J. W.** *Introduction to Fourier Optics*, 3rd ed. Roberts & Company Publishers, 2005.
   - Fourier optics 的经典教材，详细讲解 scalar diffraction theory

2. **Born, M. & Wolf, E.** *Principles of Optics*, 7th ed. Cambridge University Press, 1999.
   - 光学"圣经"，Chapter 8 详细讨论 Kirchhoff 和 Rayleigh-Sommerfeld theory

3. **Hecht, E.** *Optics*, 5th ed. Pearson, 2016.
   - 本科光学教材，Chapter 10 介绍 Fresnel 和 Fraunhofer diffraction

### 原始论文

4. **Sommerfeld, A.** "Optics: Lectures on Theoretical Physics, Vol. IV." Academic Press, 1954.
   - Sommerfeld 的原始推导

5. **Kirchhoff, G.** "Zur Theorie der Lichtstrahlen." *Sitzungsberichte der Königlich Preußischen Akademie der Wissenschaften zu Berlin*, 1882.
   - Kirchhoff 的原始论文

### 现代综述

6. **Nieto-Vesperinas, M.** *Scattering and Diffraction in Physical Optics*. World Scientific, 2006.
   - 现代观点的 diffraction theory

7. **Paganin, D. M.** *Coherent X-Ray Optics*. Oxford University Press, 2006.
   - 包含 rigorous diffraction theory 和应用

### Online Resources

8. **MIT OpenCourseWare - Optics**
   https://ocw.mit.edu/courses/mechanical-engineering/2-71-optics-spring-2009/
   - MIT 的光学课程，有 diffraction 的详细 lecture notes

9. **Stanford University - Fourier Optics**
   https://web.stanford.edu/class/ee361/
   - Stanford 的 Fourier optics 课程

10. **HyperPhysics - Diffraction**
    http://hyperphysics.phy-astr.gsu.edu/hbase/phyopt/diffrac.html
    - Georgia State University 的物理概念网页

### 进阶话题

11. **Wolf, E.** "Electromagnetic Diffraction in Optical Systems I: An Integral Representation of the Image Field." *Proc. R. Soc. Lond. A* 253, 349-357 (1959).
    - Vector diffraction theory

12. **Richards, B. & Wolf, E.** "Electromagnetic Diffraction in Optical Systems II: Structure of the Image Field in an Aplanatic System." *Proc. R. Soc. Lond. A* 253, 358-379 (1959).
    - High-NA focusing 的经典论文

---

希望这个详细的解释能够帮助你建立对 Rayleigh-Sommerfeld diffraction theory 的深刻 intuition！如果需要更深入探讨某个具体方面（比如 numerical implementation 或 vector extension），请告诉我。