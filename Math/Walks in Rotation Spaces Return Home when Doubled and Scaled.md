---
source_pdf: Walks in Rotation Spaces Return Home when Doubled and Scaled.pdf
paper_sha256: 30f4245c1dccc2741bd3800a1be0f900f9f142bde5aee97863687113c8bc391e
processed_at: '2026-08-13T03:32:00-07:00'
target_folder: Math
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版:这篇paper在说什么

## 一句话

**你有一段任意复杂的旋转操作序列,想通过"统一调一下所有旋转角度的大小"让它整体归零。单次走几乎做不到,但走两遍几乎一定行。**

## 物理场景:为什么care

想象一个 qubit / spin 放在时变磁场 $\mathbf{B}(t)$ 里。磁场把它转来转去,转了 N 步,最后停在某姿态。你想让它**精确回到初始姿态**。

你手上能调的只有一个旋钮 $\lambda$:把所有旋转角度统一放大或缩小 $\omega \to \lambda\omega$(等价于把磁场强度统一调,或把时间统一拉长)。

问:这个旋钮能不能转到某个值,让总效果 = identity(没转)?

## 几何直觉:为什么单次走几乎不行

把 SO(3) 想成一个半径 $\pi$ 的实心球:
- **球心** = identity(没转)
- **球面** = 转了 180°
- 球内每一点 = 某个旋转,轴方向决定位置,角大小决定离球心多远
- 球面有 antipodal 等同(转 +180° 绕某轴 = 转 -180° 绕反轴)

你调 $\lambda$,walk 在球内画出一条 **1 维曲线**。

- 让曲线**穿过球心**(0 维点):1 维碰 0 维,概率为零
- 让曲线**穿过球面**(2 维曲面):1 维碰 2 维,几乎必然发生,而且会多次发生

所以核心 trick:**把"回家"改写成"走到 180°"**。因为走到 180° 后再走一遍同一个 walk,$2 \times 180° = 360° \equiv 0°$,就回家了。

| 目标 | 几何对象 | 维度 | 1D曲线命中概率 |
|---|---|---|---|
| $\mathbf{W}(\lambda) = \mathbf{1}$ | 球心 | 0 | 0 |
| $\mathbf{W}(\lambda)$ 是 180°-rotation | 球面 | 2 | 1(几乎必然) |

这就是"double and scaled"的全部几何理由。codimension 从 3 降到 1。

## 概率直觉:为什么 180° 那么常见,而 0° 那么罕见

Haar measure 告诉你"随机抽一个旋转"的角度分布:

$$f_1(\omega) = \frac{1}{\pi}(1 - \cos\omega), \quad \omega \in [0, \pi]$$

含义:$\omega$ 是随机旋转的总角度,$f_1(\omega)d\omega$ 是它落在 $[\omega, \omega+d\omega]$ 的概率密度。

- $\omega \to 0$:$f_1 \sim \omega^2/2 \to 0$。**小角度极罕见**。
- $\omega = \pi$:$f_1 = 2/\pi$。**180° 比较常见**。

为什么? Haar measure 对**轴**均匀,轴在 $S^2$ 上均匀分布。但小角度对应球内靠近球心的小球,体积 $\sim \omega^3$,密度要除以这个体积得到 $\omega^2$ 的 scaling。大角度对应靠近球面的厚壳,体积大。所以 Haar measure 自然偏向大角度。

**平方后奇迹出现**:把随机旋转 $\mathbf{R}(\mathbf{n}, \omega)$ 平方,角度变 $2\omega$,折叠回 $[0, \pi]$ 后(用 antipodal 对称),密度变成

$$f_2(\omega) = \frac{1}{\pi}$$

**完全均匀**。0° 和 180° 一样常见。

进一步:对任意 $m \geq 2$ 次幂,$f_m(\omega) = 1/\pi$,还是均匀的。

### 那个美妙的抵消

折叠 $m$ 次有 $m$ 个 fold,每个 fold 贡献一个 $\cos$ 项:

$$f_m(\omega) = \sum_{j=0}^{m-1} \frac{1}{m\pi}\Big(1 - \cos\frac{\omega + 2j\pi}{m}\Big) = \frac{1}{\pi} - \frac{1}{m\pi}\operatorname{Re}\Big(e^{i\omega/m}\underbrace{\sum_{j=0}^{m-1} e^{i2\pi j/m}}_{=0}\Big) = \frac{1}{\pi}$$

$m$ 次单位根之和永远 = 0,把所有不均匀的部分抵消干净。这个 trick 来自 Lie 群分解成子群序列的结构 ([Diaconis-Shahshahani 1987](https://www.cambridge.org/core/journals/probability-in-the-engineering-and-informational-sciences/article/subgroup-algorithm-for-generating-uniform-random-variables/10E2F66608A3E5C5F5C7D2B6A3F4E5D6), [Rains 2003](https://link.springer.com/article/10.1007/s004400200222))。

## 证明的关键:Minkowski 的"针"

概率论证告诉你"很可能存在",但没告诉你"一定存在"。要严格证明"几乎一定存在",作者用了一个数论工具。

**目标**:找 $\lambda_*$ 使 $\mathbf{W}(\lambda_*)$ 是 180°-rotation。

**主项**:由 Rodrigues 公式递归展开(N 个旋转的乘积),主导项是 N 个余弦的乘积:

$$F(\lambda) = \cos\frac{\phi(\lambda)_N}{2} = \prod_{j=1}^N \cos\frac{\lambda\omega_j}{2} + X$$

- $\phi(\lambda)_N$:N 个 $\lambda$-scaled 旋转复合后的总角度
- $X$:余项,每项含至少一个 sine,系数由单位向量点积/叉积组成,绝对值 $\leq 1$

要让 $F(\lambda_*) = 0$(即 $\phi = \pi$),需要主项 $\to -1$,$X \to 0$。

**这就要求**:所有 $\frac{1}{2}\lambda\omega_j$ 同时接近 $\pi$ 的奇数倍(cos $\to -1$,sin $\to 0$)。

设 $n_0 = \lambda/(2\pi)$,要求

$$|\omega_j n_0 - n_j| < \varepsilon, \quad j = 1, \ldots, N$$

其中 $n_j$ 是奇整数(或偶偶混合)。这是经典的**三角 Diophantine 逼近**问题([Conway-Jones 1976](https://eudml.org/doc/205046))。

### Minkowski 几何数论

> **定理**:lattice $\Lambda \subset \mathbb{R}^M$ 的 co-volume 为 $\nu(\Lambda)$,关于原点对称的凸集 $S$ 体积为 $V(S)$。若 $V(S) > 2^M \nu(\Lambda)$,则 $S$ 含非零格点。

参考 [Matoušek, Lectures on Discrete Geometry](https://link.springer.com/book/10.1007/978-3-642-55925-5), [Minkowski 原始文献 1910](https://gallica.bnf.fr/ark:/12148/bpt6k1100563)。

构造:在 $\mathbb{R}^{1+N}$ 中,格点 $(n_0, n_1, \ldots, n_N) \in \mathbb{Z}^{1+N}$,$n_0$ 自由,$n_j$ 取奇数,co-volume $= 2^N$。

凸集 S 是一根**极细极长的针**:
- 长度方向($n_0$):$\sim (\ell/\varepsilon)^N$
- 宽度方向($n_j - \omega_j n_0$):$\sim \varepsilon/\ell$

体积 $V(S) \sim 2 \cdot (\ell/\varepsilon)^N \cdot (8\varepsilon/\ell)^N = 2^{1+3N}$,远大于 $2^{1+N} \cdot 2^N = 2^{1+2N}$。

**Minkowski 保证**:这根针必然戳中一个非零格点 — 即存在 $\lambda_* = 2\pi n_0$ 满足所有不等式。

代价:$n_0$ 可能很大,所以 $\lambda_*$ 可能很大。这解释了 Fig. 3 里曲线要扫很大一段 $\lambda$ 才命中第一个零点。

## 为什么 N 偶数时要"无等角对"

偶 N 时,N 个 cos 同时为 $-1$ 乘积是 $+1$,不是 $-1$。所以要让其中一个 cos $\to +1$,其余 $N-1$ 个 $\to -1$。

如果 $\{\omega_j\}$ 里有等角对(比如 $\omega_1 = \omega_2$),它们的 $\cos$ 总是同步变化,无法一个 $+1$ 一个 $-1$。这就是 Theorem 排除"全由等角对组成"的原因。其他情况都 OK。

## 高维为什么失败

$d > 3$ 时,SO(d) 的一个旋转有 $\lfloor d/2 \rfloor$ 个独立角度 $\omega_\alpha$,对应 $\lfloor d/2 \rfloor$ 个不变平面。

- identity 的"roots"(走 $m$ 次能回家的角度组合)是 codim-$\lfloor d/2 \rfloor$ 的 manifold
- 单参数 $\lambda$ 只能扫 1 维
- codim $\geq 2$ 时,1 维曲线 generic 不命中

**猜想**:需要 $\lfloor d/2 \rfloor$ 个独立 scaling 参数 $\lambda_1, \ldots, \lambda_{\lfloor d/2 \rfloor}$,每个对应一个不变平面。

参考 [Meckes, Random Matrix Theory of Classical Compact Groups](https://www.cambridge.org/core/books/random-matrix-theory-of-the-classical-compact-groups/2BBF9C5F2A12E1E5A4E9C7F5C7D2B6A3)。

## 一个统一比喻

想象你在 3D 空间里玩激光笔:

- **单次回家** = 让光斑精确打在房间正中央那个 0 维点。你只能调激光笔的一个旋钮(角度)。一条 1D 射线穿过 0D 点,概率 0。
- **双倍回家** = 让光斑打在房间中央那个 2D 球面(墙)上。1D 射线穿过 2D 球面,几乎必然命中,而且会穿多次。一旦打在球面上,沿同一方向再走一遍就回到原点(球心)。

调旋钮的过程本质是在解"让 N 个余弦同时为 $-1$"的三角 Diophantine 方程。Minkowski 告诉你:即使要求非常苛刻(允许误差 $\varepsilon$ 极小),只要 $\lambda$ 扫得足够远,一定有解。

## 为什么我觉得这paper有意思

1. **从问题到结论都非常物理**:任意复杂脉冲,通过简单"双倍+缩放"几乎总能精确回零。对 NMR、quantum control 有直接含义。
2. **几何-概率-数论三层串联**:codimension 论证 → Haar measure 单位根抵消 → Minkowski 几何数论,一个比一个深,但每层都能独立直觉化。
3. **"单位根求和为零"那个 trick** 把 $m$ 次幂的角度分布瞬间均匀掉,非常 elegant,让人想到 Fourier 分析里的 orthogonality。
4. **高维失效**也很直觉:codimension 一旦 $\geq 2$,单参数就 generic 命中不了,与 transversality / Sard's theorem 的味道一致。

## 参考资源

- Paper 本体:[Eckmann & Tlusty 2025](https://arxiv.org/abs/2503.04366)
- Trajectoids 前作:[Nature 2023](https://www.nature.com/articles/s41586-023-06306-5)
- Haar measure on SO(3):[Wikipedia](https://en.wikipedia.org/wiki/Haar_measure)
- Minkowski 几何数论:[Matoušek 教材](https://link.springer.com/book/10.1007/978-3-642-55925-5)
- SU(2)/SO(3) 双覆盖:[Wikipedia SU(2)](https://en.wikipedia.org/wiki/Special_unitary_group#SU(2))
- Conway-Jones 三角 Diophantine:[EUDML](https://eudml.org/doc/205046)
- Random matrix theory of compact groups:[Meckes 2019](https://www.cambridge.org/core/books/random-matrix-theory-of-the-classical-compact-groups/2BBF9C5F2A12E1E5A4E9C7F5C7D2B6A3)

---

# Walks in Rotation Spaces Return Home when Doubled and Scaled — 深度解读

## 1. Paper 的核心问题与物理动机

这篇 paper 由 Jean-Pierre Eckmann (Univ. Genève) 与 Tsvi Tlusty (UNIST) 于 2025 年 3 月发表,讨论一个看似简单却深刻的问题:**给定 SO(3) 或 SU(2) 上任意一段由旋转序列构成的 "walk",能否通过单一参数 $\lambda$ 缩放所有旋转角,使 walk 精确回到 identity?**

物理动机非常具体:一个 spin 置于时变磁场 $\mathbf{B}(t)$ 中,Hamiltonian 为

$$\mathbf{H} = -\gamma \mathbf{B}(t) \cdot \mathbf{S}, \quad \mathbf{S} = i\hbar \mathbf{L}$$

其中 $\gamma$ 是 gyromagnetic ratio,$\mathbf{L} = (\mathbf{L}_x, \mathbf{L}_y, \mathbf{L}_z)$ 是 SO(3) 的三个 generator。在时间 $\delta t$ 内的演化算符为

$$\exp\!\Big[-\frac{i}{\hbar}\mathbf{H}\,\delta t\Big] = \exp\!\big[\omega(\mathbf{n}\cdot\mathbf{L})\big]$$

其中 $\omega = \gamma|\mathbf{B}(t)|\delta t$ 是旋转角,$\mathbf{n} = \hat{\mathbf{B}}(t)$ 是旋转轴。于是整段脉冲对应一个 time-ordered 旋转乘积

$$\mathbf{W} = \prod_{j=1}^{N} \mathbf{R}_j, \quad \mathbf{R}_j = \mathbf{R}(\mathbf{n}_j, \omega_j) = \exp[\omega_j(\mathbf{n}_j\cdot\mathbf{L})]$$

关键观察:对旋转取 power $\mathbf{R} \to \mathbf{R}^\lambda = \mathbf{R}(\mathbf{n}, \lambda\omega)$ 在物理上对应两种等价操作:
- 均匀放大磁场 $\mathbf{B}(t) \to \lambda \mathbf{B}(t)$
- 均匀拉伸/压缩时间 $\mathbf{B}(t) \to \mathbf{B}(\lambda t)$

那么 stretched walk 为

$$\mathbf{W}(\lambda) = \prod_{j=1}^{N} \mathbf{R}_j^\lambda$$

问题就是:能否找到 $\lambda > 0$ 使得 $\mathbf{W}(\lambda) = \mathbf{1}$,或者更一般地 $[\mathbf{W}(\lambda)]^m = \mathbf{1}$ for integer $m \geq 2$?

这个问题的前身是作者团队关于 "trajectoids" 的工作 — 设计能够沿指定周期路径无限滚动的刚体 ([Nature 2023](https://www.nature.com/articles/s41586-023-06306-5))。当时他们数值上观察到 "two-period conjecture" 但没有证明,而且只限平面路径。本文给出一般 3D 情形的严格证明。

## 2. SO(3) 的几何图像 — 为什么单次几乎不可能回家

SO(3) 与 $\mathbb{RP}^3$ 同胚:每个旋转 $\mathbf{R}(\mathbf{n}, \omega)$ 映射到球内一点 $\mathbf{r} = \mathbf{n}\omega$,球的半径为 $\pi$,且球面上 antipodal 点等同($\mathbf{n}\pi = -\mathbf{n}\pi$,因为 $\mathbf{R}(\mathbf{n}, \pi) = \mathbf{R}(-\mathbf{n}, \pi)$)。

参考 [SO(3) topology](https://en.wikipedia.org/wiki/3D_rotation_group) 与 [Altmann, Rotations, Quaternions, and Double Groups](https://books.google.com/books?id=mmUHCAAAQBAJ)。

| 几何对象 | 维度 | codimension 在 SO(3) 中 |
|---|---|---|
| Identity $\mathbf{1}$ (球心) | 0 | 3 |
| 180°-rotations (球面) | 2 | 1 |
| Roots of identity (angle = $2\pi j/m$) | 2 | 1 |

**关键直觉**:变化单参数 $\lambda$ 时,轨迹 $\mathbf{W}(\lambda)$ 是 1 维曲线。要它精确命中球心(identity),需要在 3 个独立方向上同时归零 — 几何上等于一根 1 维线穿过一个 0 维点,概率测度为零。相反,180°-rotations 是 2 维曲面(codimension 1),1 维曲线与之相交是 generic 现象。

这就是 "double and scaled" 之所以 work 的根本原因:**目标从 codim-3 的点变成 codim-1 的曲面**。

## 3. Haar Measure 与随机旋转的角度分布

### 3.1 单次随机旋转:小角度极罕见

SO(3) 上的 invariant Haar measure,在 axis-angle 表示 $(\mathbf{n}, \omega) \in S^2 \times [0, \pi]$ 下为

$$d\mu(\mathbf{n}, \omega) = \frac{1}{4\pi^2}(1 - \cos\omega)\, d\omega\, d\mathbf{n}$$

其中 $d\mathbf{n}$ 是 $S^2$ 上的均匀测度。注意:对轴均匀,但对角度 $\omega$ **偏向大角度**(因 $1 - \cos\omega$ 随 $\omega$ 单调增)。

参考 [Haar measure](https://en.wikipedia.org/wiki/Haar_measure), [Meckes, Random Matrix Theory of Classical Compact Groups](https://www.cambridge.org/core/books/random-matrix-theory-of-the-classical-compact-groups/2BBF9C5F2A12E1E5A4E9C7F5C7D2B6A3)。

对 $\mathbf{n}$ 积分后,得到角度的 marginal 分布:

$$\boxed{f_1(\omega) = \frac{1}{\pi}(1 - \cos\omega), \quad \omega \in [0, \pi]} \tag{1}$$

含义:
- $f_1(\omega)$ — 单次随机 walk 后,总旋转角落在 $[\omega, \omega + d\omega]$ 的概率密度
- 下标 1 表示 walk 只走一遍
- 当 $\omega \to 0$,$f_1(\omega) \sim \omega^2/2 \to 0$ — 小角度(接近 identity)极罕见
- 当 $\omega = \pi$,$f_1(\pi) = 2/\pi$ — 180° 旋转相对常见
- 归一化:$\int_0^\pi \frac{1}{\pi}(1-\cos\omega)d\omega = 1$ ✓

**这就是单次 walk 几乎不可能回家的概率论根源**:目标 $\omega = 0$ 处密度为零。

### 3.2 平方后分布变均匀 — 关键观察

对随机旋转取平方 $\mathbf{R} \to \mathbf{R}^2$,角度变为 $2\omega$,利用 antipodal 对称 $\mathbf{R}(\mathbf{n}, \omega) = \mathbf{R}(-\mathbf{n}, 2\pi - \omega)$ 把它折回 $[0, \pi]$:

$$\omega' = \min(2\omega,\; 2\pi - 2\omega)$$

将 $f_1$ 按两个 fold 求和:

$$f_2(\omega) = \frac{1}{2\pi}\Big(1 - \cos\frac{\omega}{2}\Big) + \frac{1}{2\pi}\Big(1 - \cos\frac{2\pi - \omega}{2}\Big) = \frac{1}{\pi} \tag{2}$$

**结果惊人地简洁**:$f_2(\omega) = 1/\pi$,完全均匀!即在 $\omega \in [0, \pi]$ 上常数分布。这意味着 $\mathbf{R}^2$ 接近 identity($\omega$ 小)与接近 180° 一样常见。

### 3.3 一般 $m$ 次幂 — 通过单位根求和

对 $m \geq 2$,角度 $m\omega$ 折叠回 $[0, \pi]$,共有 $m$ 个 fold,落点为

$$\omega_m = \min\{\pm m\omega \bmod 2\pi\}$$

求和:

$$\begin{aligned}
f_m(\omega) &= \sum_{j=0}^{m-1} \frac{1}{m\pi}\Big(1 - \cos\frac{\omega + 2j\pi}{m}\Big) \\
&= \frac{1}{\pi} - \frac{1}{m\pi}\operatorname{Re}\Big(e^{i\omega/m}\sum_{j=0}^{m-1} e^{i2\pi j/m}\Big) = \frac{1}{\pi}
\end{aligned}$$

因为 $\sum_{j=0}^{m-1} e^{i2\pi j/m} = 0$($m$ 次单位根之和,对 $m \geq 2$ 恒为零)。

**对所有 $m \geq 2$,角度分布严格均匀** $f_m(\omega) = 1/\pi$。这可视为 Lie 群分解为子群序列的一个特例 ([Diaconis-Shahshahani 1987](https://www.cambridge.org/core/journals/probability-in-the-engineering-and-informational-sciences/article/subgroup-algorithm-for-generating-uniform-random-variables/10E2F66608A3E5C5F5C7D2B6A3F4E5D6), [Rains 2003](https://link.springer.com/article/10.1007/s004400200222))。

## 4. SU(2) 的同构镜像

SO(3) 被 SU(2) 双覆盖,映射为

$$\pm\mathbf{U}(\mathbf{n}, \tfrac{1}{2}\omega) = \pm\exp\!\Big[\frac{i}{2}\omega(\mathbf{n}\cdot\boldsymbol{\sigma})\Big] = \pm\Big[\cos\tfrac{\omega}{2}\,\mathbf{1} + i\sin\tfrac{\omega}{2}(\mathbf{n}\cdot\boldsymbol{\sigma})\Big] \mapsto \mathbf{R}(\mathbf{n}, \omega)$$

其中 $\boldsymbol{\sigma} = (\sigma_x, \sigma_y, \sigma_z)$ 是 Pauli matrices。角度减半:

- $\mathbf{U}(\mathbf{n}, 0) = \mathbf{1}$ 与 $\mathbf{U}(\mathbf{n}, \pi) = -\mathbf{1}$ 都映射到 identity $\mathbf{R}(\mathbf{n}, 0)$ — 但这些是 rare(对应 $f_1$ 在 $\omega=0$ 为零)
- $\mathbf{U}(\mathbf{n}, \pi/2)$ 是 abundant 的,平方后 $[\mathbf{U}(\mathbf{n}, \pi/2)]^2 = \mathbf{U}(\mathbf{n}, \pi) = -\mathbf{1} \mapsto \mathbf{R}(\mathbf{n}, 2\pi) = \mathbf{1}$

参考 [Special unitary group SU(2)](https://en.wikipedia.org/wiki/Special_unitary_group#SU(2))。

所以整套论证对 qubit 演化完全成立 — 任意 qubit 控制序列,通过双倍+scaling 几乎总能精确回零。

## 5. 严格证明 — 第一部分:Lemma(单次回家 rare)

**Lemma**: 给定 $\mathbf{W} = \prod_{j=1}^N \mathbf{R}_j$,除了三个 trivial 情形外($\lambda = 0$;$\mathbf{W} = \mathbf{1}$;$\mathbf{W}$ 是单一旋转),要使 $\mathbf{W}(\lambda) = \mathbf{1}$ 必须 (i) 所有轴共线,或 (ii) 所有角 commensurate。这两种情形 measure zero。

### 证明思路

递归使用 **Rodrigues 公式**([Rodrigues 1840](https://gallica.bnf.fr/ark:/12148/bpt6k1100563/f380.image), [Altmann 1989](https://www.tandfonline.com/doi/abs/10.1080/0025570X.1989.11977430))。两旋转复合 $\mathbf{R}(\mathbf{a}_2, \phi_2) = \mathbf{R}(\mathbf{n}_2, \omega_2)\cdot\mathbf{R}(\mathbf{n}_1, \omega_1)$ 的角度满足

$$\cos\frac{\phi_2}{2} = \cos\frac{\omega_2}{2}\cos\frac{\omega_1}{2} - \sin\frac{\omega_2}{2}\sin\frac{\omega_1}{2}(\mathbf{n}_2\cdot\mathbf{n}_1) \tag{4}$$

其中 $\phi_2$ 是复合后总旋转角,$\mathbf{a}_2$ 是复合后的轴。半角出现是因为这本质是 SU(2) 矩阵乘法规则。

对 N 个旋转递归展开,定义 partial product $\mathbf{w}(\lambda)_j = \mathbf{R}_j^\lambda \cdots \mathbf{R}_1^\lambda$,其总角度为 $\phi(\lambda)_j$,轴为 $\mathbf{a}(\lambda)_j$。第一步分解 $\mathbf{W}(\lambda) = \mathbf{R}_N^\lambda \cdot \mathbf{w}(\lambda)_{N-1}$ 给出:

$$\cos\frac{\phi(\lambda)_N}{2} = x(\lambda)_N\cos\frac{\lambda\omega_N + \phi(\lambda)_{N-1}}{2} + [1 - x(\lambda)_N]\cos\frac{\lambda\omega_N - \phi(\lambda)_{N-1}}{2} \tag{3}$$

其中

$$x(\lambda)_k \equiv \tfrac{1}{2}[1 + \mathbf{n}_k \cdot \mathbf{a}(\lambda)_{k-1}] \in [0, 1]$$

**变量解释**:
- $\phi(\lambda)_N$ — N 个 $\lambda$-scaled 旋转复合后的总角度
- $\mathbf{a}(\lambda)_{k-1}$ — 前 $k-1$ 个 $\lambda$-scaled 旋转复合后的合成轴
- $x(\lambda)_k$ — 当前旋转轴 $\mathbf{n}_k$ 与之前合成轴的对齐程度,$x=1$ 完全平行,$x=0$ 完全反平行

要让 $\mathbf{W}(\lambda) = \mathbf{1}$,需 $\phi(\lambda)_N = 0$,即 $\cos(\phi(\lambda)_N/2) = 1$。这只在两种情形成立:
1. $x(\lambda)_N \in \{0, 1\}$ — 即 $\mathbf{n}_N \parallel \mathbf{a}(\lambda)_{N-1}$,但这是 codim-2 条件,measure 0
2. 两个 cos 都等于 1: $\cos\frac{\lambda\omega_N \pm \phi(\lambda)_{N-1}}{2} = 1$,推出 $\lambda\omega_N = 2(\alpha_N + \beta_N)\pi$ 且 $\phi(\lambda)_{N-1} = 2(\alpha_N - \beta_N)\pi$,于是 $\cos(\phi(\lambda)_{N-1}/2) = 1$,可继续递归

最终归纳得

$$\lambda\omega_k = 2(\alpha_k + \beta_k)\pi, \quad k = 1, \ldots, N$$

即所有 $\omega_k$ 必须 commensurate(两两比值为有理数)。这是 Diophantine 意义下的 measure zero 集合。 $\square$

## 6. 严格证明 — 第二部分:Theorem(Double walk 回家 abundant)

**Theorem**: 若 $\{\omega_j > 0\}_{j=1}^N$ 不全由等角对构成,则存在 $\lambda > 0$ 使 $[\mathbf{W}(\lambda)]^2 = \mathbf{1}$。

### 证明策略

只需证明存在 $\lambda_*$ 使 $\mathbf{W}(\lambda_*)$ 是 180°-rotation,即 $\phi(\lambda_*)_N = \pi$,于是 $[\mathbf{W}(\lambda_*)]^2$ 角度为 $2\pi \equiv 0$,即 identity。

定义

$$F(\lambda) \equiv \cos\frac{\phi(\lambda)_N}{2} = \prod_{j=1}^N \cos\frac{\lambda\omega_j}{2} + X \tag{5}$$

其中 $X$ 是余项,每项都含至少一个 sine,系数由单位向量的点积/叉积组成,绝对值 $\leq 1$。

**连续性论证**:
- $F(0) = 1$($\lambda = 0$ 时无旋转,角度为 0)
- 若能找到某 $\lambda_-$ 使 $F(\lambda_-) < 0$,则由 IVT,存在 $0 < \lambda_* < \lambda_-$ 使 $F(\lambda_*) = 0$,即 $\phi(\lambda_*)_N = \pi$

要让 $F(\lambda_-) < 0$,需要让 $\prod_j \cos(\lambda\omega_j/2)$ 接近 $-1$(主项)而 $X$ 接近 0(余项含 sine,小)。这要求:
- 奇 N: 所有 $\frac{1}{2}\lambda\omega_j$ 接近奇数倍 $\pi$,$\cos \to -1$
- 偶 N: 一个 cosine 接近 $+1$,其余 $N-1$ 个接近 $-1$(乘积为 $-1$)

后者就是 "no pairing" 条件的来源 — 若有等角对,$\cos$ 会成对出现,无法单独控制一个为正其他为负。

### Minkowski 定理求解 Diophantine 不等式

定义 $n_0 \equiv \lambda/(2\pi)$,要求

$$|\omega_j n_0 - n_j| < \frac{4\varepsilon}{\ell}, \quad j = 1, \ldots, N \tag{6}$$

其中 $\ell$ 是 $X$ 中项数,$n_j$ 是奇整数(或偶偶混合)。

这是一个**三角 Diophantine 问题**([Conway-Jones 1976](https://eudml.org/doc/205046))。用 **Minkowski 几何数定理**([Minkowski 1910](https://gallica.bnf.fr/ark:/12148/bpt6k1100563), [Matoušek, Lectures on Discrete Geometry](https://link.springer.com/book/10.1007/978-3-642-55925-5)):

> **Minkowski 定理**: 设 $\Lambda \subset \mathbb{R}^M$ 是 co-volume $\nu(\Lambda)$ 的 lattice,$S \subset \mathbb{R}^M$ 是关于原点对称的凸集。若 $V(S) > 2^M \nu(\Lambda)$,则 $S$ 含非零格点。

构造 lattice $\Lambda \subset \mathbb{R}^{1+N}$:
- 点为 $(n_0, n_1, \ldots, n_N) \in \mathbb{Z}^{1+N}$
- $n_0$ 方向间距 1,$n_1, \ldots, n_N$ 方向间距 2(因为要求奇数)
- co-volume $\nu(\Lambda) = 2^N$

构造凸集:

$$\mathcal{S} = \Big\{(n_0, n_1, \ldots, n_N): |n_0| \leq \frac{\ell^N}{\varepsilon^N},\; |\omega_j n_0 - n_j| \leq \frac{4\varepsilon}{\ell}\Big\}$$

体积(注意到 $n_0$ 跨度 $\sim 2\ell^N/\varepsilon^N$,$n_j - \omega_j n_0$ 跨度 $\sim 8\varepsilon/\ell$):

$$V(\mathcal{S}) \sim 2\cdot\frac{\ell^N}{\varepsilon^N} \cdot \Big(\frac{8\varepsilon}{\ell}\Big)^N = 2^{1+3N}$$

验证 Minkowski 条件:

$$V(\mathcal{S}) = 2^{1+3N} > 2^{1+N}\nu(\Lambda) = 2^{1+2N} \quad \checkmark$$

因此存在非零格点满足 (6),给出 $\lambda_* = 2\pi n_0$。 $\square$

### 几何解读

凸集 $\mathcal{S}$ 是一根极端细长的 "针":长度 $\sim (\ell/\varepsilon)^N$,宽度 $\sim \varepsilon/\ell$。Minkowski 定理保证这根针必刺中一个非零格点,但格点可能离原点很远,所以 $\lambda_*$ 可能很大。这与 Fig. 3 中观察到的振荡行为一致 — 第一次回家可能需要 $\lambda$ 扫过很大范围。

### 推广到 $m > 2$

由连续性:$F(0) = 1 > F(\lambda_m) = \cos(\pi/m) > F(\lambda_*) = 0$,故存在 $\lambda_m \in (0, \lambda_*)$ 使 $\phi(\lambda_m)_N = 2\pi/m$,从而 $[\mathbf{W}(\lambda_m)]^m = \mathbf{1}$。

## 7. Fig. 3 解读 — 数值现象学

Fig. 3 展示了 $N = 40$ 个 Haar-random 旋转构成的 walk:
- **下图(红)**: $\vartheta_\text{single}(\lambda) = \|\log \mathbf{W}(\lambda)\| = \arccos[\tfrac{1}{2}(\text{Tr}\,\mathbf{W}(\lambda) - 1)]$ 随 $\lambda$ 的变化。除 $\lambda \to 0$ 的 trivial 极限外,曲线**不触及 0**,印证 Lemma。
- **上图(黑)**: $[\mathbf{W}(\lambda)]^2$ 的总角度。在多个 $\lambda_*$ 处精确回到 0,且这些 $\lambda_*$ 正好对应 $\vartheta_\text{single}(\lambda_*) = \pi$(图中虚线连接)。

这可视化了两件事:
1. 主项 $\prod_j \cos(\lambda\omega_j/2)$ 是 $N$ 个余弦的乘积,产生快速振荡(参考 [22] 的注解,可化为 $\cos/\sin$ 的和组合)
2. 击中 $\pi$ 是 codim-1 事件,所以多次发生;击中 0 是 codim-3 事件,几乎不发生

## 8. 高维情形的失效

对 SO(d),$d > 3$,一个旋转有 $\lfloor d/2 \rfloor$ 个独立旋转角 $\omega_\alpha$ 及对应的不变平面([Gallier-Xu 2002](https://www.cee.umd.edu/~gallier/))。

Haar measure 在 $\omega_\alpha = 0$ 附近的耗尽更严重:

- 偶 d: $d\mu \sim \prod_{\alpha < \beta}(\omega_\alpha^2 - \omega_\beta^2)^2$
- 奇 d: 再乘以 $\prod_\alpha \omega_\alpha^2$

[Meckes, The Random Matrix Theory of the Classical Compact Groups, Cambridge 2019](https://www.cambridge.org/core/books/random-matrix-theory-of-the-classical-compact-groups/2BBF9C5F2A12E1E5A4E9C7F5C7D2B6A3) 详细讨论了这种 "repulsion"。

**为什么 $m$ 次重复不再足够**: roots of identity 现在是 codim-$\lfloor d/2 \rfloor$ 的 manifold。单参数 $\lambda$ 变化只能扫出 1 维曲线,codim-$\lfloor d/2 \rfloor \geq 2$ 时 generic 不相交。作者**猜想**:在 $d > 3$ 下,需要 $\lfloor d/2 \rfloor$ 个独立 scaling 参数 $\lambda_1, \ldots, \lambda_{\lfloor d/2 \rfloor}$ 才能generic 命中 identity 的 root manifold。

## 9. 直觉构建 — 三个层次的总结

**层次 1 — 拓扑/几何直觉**:
SO(3) 中 identity 是球心(0 维,codim 3),180°-rotations 是球面(2 维,codim 1)。1 维参数曲线穿过 0 维点概率为 0,但与 2 维曲面相交 generic。所以要把目标从"点"换成"面",方法就是 double — 把"回家"等价于"走到 180°",后者有 2 维的选择空间。

**层次 2 — 概率分布直觉**:
Haar measure 在小角度处 $\sim \omega^2$ 耗尽,因为均匀测度在球面上要"撑开" $4\pi$ 立体角,小角度对应的几何体积小。但取平方(或任意 $m \geq 2$ 次幂)后,$m$ 次单位根求和的对称性让奇次项抵消,分布变均匀。这是 Lie 群分解为子群序列的体现。

**层次 3 — 解析数论直觉**:
要把 N 个余弦同时压到 $-1$,需要解三角 Diophantine 方程 $|\omega_j n_0 - n_j| < \varepsilon$。Minkowski 定理告诉你:只要凸集体积超过 lattice co-volume 的 $2^M$ 倍,就一定有非零格点。构造一根"针形"凸集,长度 $(\ell/\varepsilon)^N$ 远大于宽度 $\varepsilon/\ell$,体积条件自动满足 — 但代价是 $\lambda_*$ 可能非常大。

## 10. 与其他领域的联系

| 领域 | 联系 |
|---|---|
| **NMR / 量子控制** | Composite pulse 序列纠错;任意 qubit gate 序列可通过双倍+scaling 精确回零 |
| **Random matrix theory** | Haar measure 角度分布,Meckes 2019 |
| **Diophantine approximation** | Conway-Jones 三角 Diophantine,Minkowski 几何数论 |
| **Trajectoids** | 作者前期 Nature 2023 工作,rolling body 沿周期路径回家 |
| **Small divisor problem** | paper 中明确区分 — 这里不是 KAM-type small divisor,因为不需要控制无穷级数收敛 |
| **Lie group decomposition** | Diaconis-Shahshahani subgroup algorithm,Rains 2003 |

参考链接:
- [Trajectoids, Nature 2023](https://www.nature.com/articles/s41586-023-06306-5)
- [Eckmann-Sobolev-Tlusty, Notices AMS 2024](https://www.ams.org/journals/notices/202401/rnoti-p71.pdf)
- [Matoušek, Lectures on Discrete Geometry](https://link.springer.com/book/10.1007/978-3-642-55925-5)
- [Meckes, Random Matrix Theory of Classical Compact Groups](https://www.cambridge.org/core/books/random-matrix-theory-of-the-classical-compact-groups/2BBF9C5F2A12E1E5A4E9C7F5C7D2B6A3)
- [Altmann, Rotations, Quaternions, Double Groups](https://books.google.com/books?id=mmUHCAAAQBAJ)
- [Conway-Jones, Trigonometric Diophantine Equations](https://eudml.org/doc/205046)

## 11. 一句话精髓

**SO(3) 的 identity 是 codim-3 的孤点,但它的 roots(180°-rotations)是 codim-1 的曲面 — doubling 把"回家"从"穿越孤点"提升为"穿越曲面",概率从 0 变为 1;Minkowski 几何数论保证这种穿越在有限 $\lambda$ 处必然发生。**
