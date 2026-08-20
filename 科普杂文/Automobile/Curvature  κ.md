Curvature $\kappa$ (Kappa) 的计算取决于 curve 的表达形式。衡量了 curve 在某一点处偏离 straight line (tangent) 的程度，即单位弧长上切向量的转角。

以下是基于不同 representation 的详细计算方法：

### 1. Parametric Curves (参数方程)

这是最通用的表达形式。假设一条 curve 由 vector-valued function $\mathbf{r}(t)$ 定义，其中 $t$ 是 parameter。

**Formula:**
$$ \kappa(t) = \frac{\|\mathbf{r}'(t) \times \mathbf{r}''(t)\|}{\|\mathbf{r}'(t)\|^3} $$

**变量解析:**
*   $\mathbf{r}(t)$: Position vector (位置向量)，描述 curve 上的点。
*   $\mathbf{r}'(t)$: First derivative，代表 **Velocity vector** (速度向量) 或 Tangent vector $\mathbf{T}$。其模长 $\|\mathbf{r}'(t)\|$ 即为 speed $v$。
*   $\mathbf{r}''(t)$: Second derivative，代表 **Acceleration vector** (加速度向量)。
*   $\times$: Cross product (叉积)。在 2D 情况下，cross product 的模长等价于 determinant $\left| \begin{matrix} x' & y' \\ x'' & y'' \end{matrix} \right|$。
*   $\|\cdot\|$: Euclidean norm (欧几里得范数)，即向量的长度。

**Intuition Building:**
分子 $\|\mathbf{r}'(t) \times \mathbf{r}''(t)\|$ 计算的是由 velocity 和 acceleration 张成的平行四边形的面积。
*   如果 acceleration 与 velocity 平行（即只改变速度大小，不改变方向，如直线运动），cross product 为 0，$\kappa = 0$。
*   如果 acceleration 垂直于 velocity（即 purely changing direction，如匀速圆周运动），curvature 达到最大。

---

### 2. Explicit Functions (显函数)

当 curve 表示为 $y = f(x)$ 时，这是 parameterization 的一个特例（令 $x=t$, $y=f(t)$）。

**Formula:**
$$ \kappa(x) = \frac{|y''|}{(1 + (y')^2)^{3/2}} $$

**变量解析:**
*   $y'$: First derivative of $y$ with respect to $x$，即 slope of the tangent line (切线斜率)。
*   $y''$: Second derivative，描述 slope 的变化率。
*   分母中的 $(1 + (y')^2)^{1/2}$ 实际上是 arc length parameter 的导数项 $\frac{ds}{dx}$。

**Simplification:**
在 curve 非常平坦（$y'$ 很小）的情况下，$(y')^2 \approx 0$，分母趋近于 1。此时 Formula 退化为：
$$ \kappa \approx |y''| $$
这就是许多 Engineering approximations 中常用的形式。

---

### 3. Arc Length Parameterization (弧长参数化)

如果我们使用 Arc length $s$ 作为 parameter（这是最 natural 的 parameterization），公式会极度简化。

**Formula:**
$$ \kappa(s) = \left\| \frac{d\mathbf{T}}{ds} \right\| = \|\mathbf{T}'(s)\| $$

或者涉及 Principal Normal vector $\mathbf{N}$:
$$ \mathbf{T}'(s) = \kappa(s) \mathbf{N}(s) $$

**变量解析:**
*   $s$: Arc length parameter。
*   $\mathbf{T}$: Unit tangent vector (单位切向量)。
*   $\mathbf{N}$: Unit normal vector (单位法向量)。

**Intuition:**
这是 Curvature 最本质的定义。$\kappa$ 就是 Unit tangent vector 随 arc length 变化的速率。想象你沿着 curve 以单位速度前进，你转动方向盘的速率就是 curvature。

---

### 4. Implicit Curves (隐式方程)

对于由 $F(x, y) = 0$ 定义的 curve。

**Formula:**
$$ \kappa = \frac{|2F_x F_y F_{xy} - F_x^2 F_{yy} - F_y^2 F_{xx}|}{(F_x^2 + F_y^2)^{3/2}} $$

**变量解析:**
*   $F_x, F_y$: Partial derivatives of $F$ with respect to $x$ and $y$ (一阶偏导)。
*   $F_{xx}, F_{yy}, F_{xy}$: Second partial derivatives (二阶偏导)。
*   分母 $(F_x^2 + F_y^2)^{1/2}$ 是 Gradient $\nabla F$ 的模长。

---

### 5. Discrete Curvature (离散曲率)

在 Computer Graphics 和 Geometry Processing 中，curve 通常由 polyline (折线) 表示。

**Method: Turning Angles**
$$ \kappa_i \approx \frac{\theta_i}{\frac{l_{i-1} + l_i}{2}} $$
或者更简单的 approximation:
$$ \kappa_i \approx \frac{2 \tan(\theta_i/2)}{l} $$

**变量解析:**
*   $\theta_i$: Turning angle (转角)，即 vertex $i$ 处两条 incident edges 的夹角的外角（$\pi$ 减去内角）。
*   $l$: Average length of adjacent edges。

**Intuition:**
离散情况下，我们无法定义真正的 derivative。我们通过计算线段转折的角度除以线段长度来 estimate curvature。这符合 First Principles：单位长度上的方向改变量。

---

### 6. Geometric Interpretation: Radius of Curvature

Curvature 与 Radius of Curvature $R$ 互为倒数。

$$ R = \frac{1}{\kappa} $$

**Intuition:**
Curve 在某一点局部可以由一个圆来近似，这个圆叫做 **Osculating Circle** (密切圆)。
*   Circle 的半径越小，curve 弯曲得越厉害，$\kappa$ 越大。
*   Line 可以看作 radius 无穷大的 circle，$\kappa = 0$。

### Reference Links:
*   **Wolfram MathWorld: Curvature**: [https://mathworld.wolfram.com/Curvature.html](https://mathworld.wolfram.com/Curvature.html)
*   **Wikipedia: Curvature**: [https://en.wikipedia.org/wiki/Curvature](https://en.wikipedia.org/wiki/Curvature)
*   **Desmos Visualization**: [https://www.desmos.com/calculator/cajaxwnlua](https://www.desmos.com/calculator/cajaxwnlua) (Interactive demo)