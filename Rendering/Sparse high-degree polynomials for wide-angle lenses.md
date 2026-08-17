---
source_pdf: Sparse high-degree polynomials for wide-angle lenses.pdf
paper_sha256: 065226313c506384ab4c5eba56b6bf561e025bb03049a1f62236c73791bedb00
processed_at: '2026-08-12T08:50:43-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Hi Andrej, 我用人话结合底层 technical intuition 来拆解这篇 paper。

简单来说，这篇 paper 解决了一个非常实际的问题：**如何用极低的计算成本，在渲染器里高精度地模拟那些结构极其复杂的相机镜头（特别是视角接近 180 度的鱼眼镜头）？**

在 physically-based rendering 中，如果每次 ray 都要老老实实地和十几个 lens elements 求交、计算 refraction，计算量会爆炸。所以以前的人尝试用 Taylor series 来把整个 lens system 拟合成一个 polynomial。但是 Taylor expansion 有致命弱点：它是对光轴中心点做的局部展开。一旦到了鱼眼镜头的边缘（光线偏折极大），Taylor polynomial 就完全失真了；而且为了勉强提高精度，把 degree 提到 5 或者 7，会导致 coefficient 数量组合爆炸，根本没法用。

这篇 paper 的核心 insight 在于：抛弃 Taylor series，把 lens system 当作一个 black box，直接在极高维度（degree 高达 11）的 monomial space 里进行 sparse regression。这就像是你有 4368 种调料（对应各种 $x^a y^b dx^c \dots$ 的组合），做这道菜根本不需要全放，只需要用 OMP (Orthogonal Matching Pursuit) 算法精挑细选出 40 种最关键的调料，就能完美还原出 lens 的光学畸变。

下面我为你拆解其中的核心技术细节，帮你 build intuition。

### 1. 坐标系的重构：如何避免鱼眼镜头的奇点

传统的光线表示法是 plane/plane parametrization（用光线穿过两个平行平面的交点来表示一条 ray）。这种表示在 free space 里是完美的 linear polynomial。但是，对于 fisheye lens，光线出射角接近 90 度甚至 180 度。在 90 度时，平面参数化会产生数学奇点（除以 0）。

为了解决这个问题，authors 提出了 sphere/sphere parametrization。对于 outer pupil（镜头最外侧的镜片，假设是球面的），我们将 ray 的 position $(x_o, y_o)$ 投影到垂直于 optical axis 的平面上，而将 ray 的 direction $(dx_o, dy_o)$ 投影到该交点处的局部 tangent plane 上。

为了构建这个局部 tangent frame，paper 给出了转换矩阵 $\mathbf{T}(\mathbf{n})$ (Eq. 3)：

$$
\mathbf{T}(\mathbf{n}) = \left[ \begin{array}{ccc} n_z / l & -n_x n_y / l & n_x \\ 0 & l & n_y \\ -n_x / l & -n_y n_z / l & n_z \end{array} \right], \quad l := \sqrt{n_x^2 + n_z^2}
$$

**变量解释：**
*   $\mathbf{n} = (n_x, n_y, n_z)$ 是光线击中球面处的法向量。
*   $l$ 是一个归一化因子，等于法向量在 $x-z$ 平面上的投影长度。
*   矩阵的三列分别代表：local tangent vector（在 $x-z$ 平面内且垂直于 $\mathbf{n}$）、bitangent vector（通过 cross product 得到）和 normal vector $\mathbf{n}$。

**Intuition：** 想象你在地球仪上标点。如果用经纬度，南北极点的经度是无定义的。这个矩阵的作用就是，无论光线打在镜头球的哪个位置，都能建立一个以该点法向量为 Z 轴的局部坐标系。巧妙的是，authors 把极点设在了 $\pm y$ 方向，避开了 optical axis，保证了正对镜头中心的光线在计算时数值绝对稳定。

### 2. Sparse Polynomial 构建：OMP 算法的精妙之处

我们想要拟合一个 5D input 到 5D output 的 mapping。对于一个 degree $d=11$ 的 polynomial，有 $N(d) = \binom{5+11}{11} = 4368$ 种可能的 monomial terms。我们要在这个巨大的空间里找出一组极稀疏的系数向量 $\mathbf{c}'$。

Paper 构建了一个 overdetermined linear system (Eq. 6)：

$$
\hat{\Phi} = \left( \begin{array}{ccccc} x_1 & y_1 & \dots & \lambda_1^{d-1} dy_1 & \lambda_1^d \\ x_2 & y_2 & \dots & \lambda_2^{d-1} dy_2 & \lambda_2^d \\ \vdots & & & & \vdots \\ x_M & y_M & \dots & \lambda_M^{d-1} dy_M & \lambda_M^d \end{array} \right)
$$

**变量解释：**
*   $\hat{\Phi}$ 是设计矩阵，维度是 $M \times N(d)$。
*   $M$ 是我们通过 ray tracing 预先计算好的 ground truth 光线数量（取 $10 \cdot N(d)$）。
*   矩阵的每一行是一根 ray 的 5D 输入 $(x, y, dx, dy, \lambda)$ 计算出的各种 monomial 的值。
*   我们的目标是最小化误差 $\|\hat{\Phi} \cdot \mathbf{c}' - \mathbf{b}\|_2^2$ (Eq. 5)，其中 $\mathbf{b}$ 是 ray traced 得到的真实 output light field。

因为我们要限制 $\mathbf{c}'$ 只有 $s$ 个非零元素（比如 40 个），这个问题是 NP-hard 的。Authors 用了 OMP 的变体来贪心地求解。

**Intuition 与改进：** 传统的 OMP 就像是一个只会加人、不会开人的团队。一旦选了 40 个人，就算后来发现某个人是混子，也不能踢掉。Authors 做了两个关键改进：
1.  **Exact error calculation：** 选人时不靠简单的点积打分，老老实实做 least squares 拟合，看真实的 squared error 降了多少。
2.  **Replacement：** 团队满员（达到 $s$ 个）后，每次考察一个新人，遍历团队里现有的所有人，看替掉谁能让总 error 最小。这就保证了最终的 40 个 terms 是全局近乎最优的组合。

我们看 Table 1 的数据，对于 fisheye-aspherical lens，同样是 28 个 coefficients，Taylor polynomial 的误差是 $1.52 \times 10^{-2}$，而 Sparse polynomial 的误差是 $6.26 \times 10^{-3}$，精度提升了一个数量级。

### 3. Aperture Sampling 与 Two-Stage Newton Method

在 Monte Carlo path tracing 中，从 camera 发射 ray 很容易，但在 bidirectional path tracing 中，我们需要从 scene 里的 light source 连接 ray 到 camera。这就面临一个难题：在哪里连接？

如果在 outer pupil 随机采样，对于鱼眼镜头来说，由于 internal aperture（光圈）的物理遮挡，90% 以上的光线会被挡住，效率极低。Authors 提出直接在 aperture 上采样，然后反向求解 sensor 上的 position 和 direction。

这需要解一个非线性的 root-finding 问题。Authors 设计了一个非常优雅的 two-stage Newton iteration (Algorithm 2)。给定 scene 里的点 $\hat{\mathbf{o}}$ 和采样的 aperture point $\hat{\mathbf{A}}_{xy}$，我们要找到 sensor state $\mathbf{S} = (x_s, y_s, dx_s, dy_s)$。

*   **Stage 1 (更新 direction)**：当前的光线可能穿不过目标 $\hat{\mathbf{A}}_{xy}$。利用解析 Jacobian 矩阵 $J_a = \mathrm{d}\mathbf{A}_{xy}/\mathrm{d}\mathbf{S}_{\omega}$，求出需要修正的 sensor direction。
    $$\mathbf{S}_{\omega} \gets \mathbf{S}_{\omega} + J_a^{-1} \cdot \Delta\mathbf{A}_{xy}$$
    **变量解释：** $\mathbf{S}_{\omega}$ 是 sensor direction $(dx_s, dy_s)$，$\Delta\mathbf{A}_{xy}$ 是当前穿过的 aperture 位置与目标位置的 2D 误差。

*   **Stage 2 (更新 position)**：光线虽然穿过了 aperture，但出了镜头后可能指不到 scene 里的 $\hat{\mathbf{o}}$。算出 outer pupil 处的方向误差 $\Delta\mathbf{O}_{\odot}$，利用另一个解析 Jacobian $J_o = \mathrm{d}\mathbf{O}_{\omega}/\mathrm{d}\mathbf{S}_{xy}$，修正 sensor position。
    $$\mathbf{S}_{xy} \gets \mathbf{S}_{xy} + J_o^{-1} \cdot \Delta\mathbf{O}_{\odot}$$
    **变量解释：** $\mathbf{S}_{xy}$ 是 sensor position $(x_s, y_s)$，$\Delta\mathbf{O}_{\odot}$ 是出射方向与目标方向的偏差。

为了保证物理正确性，这种采样方式的改变必须在 Monte Carlo estimator 中补偿 Jacobian。Light tracing 的贡献公式变为 (Eq. 8)：

$$c_{lt} = \frac{f}{p} = W(\mathbf{S}) / \left( p(\mathbf{A}_{xy}) \cdot \|J_{\nu}\| \cdot \left\| \frac{\mathrm{d}\mathbf{O}_{xy}}{\mathrm{d}\mathbf{A}_{xy}} \right\| \right)$$

**变量解释：**
*   $W(\mathbf{S})$ 是 sensor responsivity。
*   $p(\mathbf{A}_{xy})$ 是在 aperture 上的采样 PDF。
*   $\|J_{\nu}\| = \cos \theta = \sqrt{R - \mathbf{O}_x^2 - \mathbf{O}_y^2 / R}$，其中 $R$ 是 outer pupil 球面的曲率半径。这个 term 是将 projected disk area measure 转换为球面上的 vertex area measure。
*   $\|\mathrm{d}\mathbf{O}_{xy}/\mathrm{d}\mathbf{A}_{xy}\|$ 则是通过共享的 $\mathrm{d}\mathbf{S}_{\omega}$ 空间链式求导得到的。

### 总结

这篇 paper 的精妙之处在于把光学镜头的复杂几何与折射，完全压缩进了一个高度定制化的 sparse polynomial 中。通过 Sphere/sphere parametrization 解决了鱼眼镜头的坐标系奇点，通过带有 Replacement 机制的 OMP 解决了高阶多项式的特征选择，通过 Two-stage Newton method 解决了反向光线追踪的采样难题。最终，每根光线只需计算几十次乘加，就能极其逼真地模拟出极其昂贵的物理光学效果，甚至能直接放进 GPU shader 里做交互式预览。

**References for further reading:**
*   原始 Polynomial Optics paper (Hullin et al. 2012): https://dl.acm.org/doi/10.1111/j.1467-8659.2012.03126.x
*   Hanika 的前作，关于 realistic lenses 的 Monte Carlo rendering (HD14): https://jo.dreggn.de/home/2014-lens.pdf
*   Orthogonal Matching Pursuit 基础理论 (Tropp & Gilbert 2007): https://arxiv.org/abs/math/0506403

---

Hi Andrej, 这篇 paper 是关于 realistic camera modeling 的非常扎实的工作。它的核心 intuition 在于: 将 geometric optics 中复杂的 ray-lens interactions 视作一个 high-dimensional, smooth, non-linear mapping。传统的做法是使用 Taylor series 来近似这个 mapping，但这在 wide-angle (fisheye) lenses 和 aspherical elements 面前会崩溃，因为 high-degree Taylor expansion 会导致 combinatorial explosion of coefficients。这篇 paper 的核心贡献是抛弃了 Taylor series，引入了 sparse approximation (Orthogonal Matching Pursuit, OMP) 直接在 high-degree monomial space 中寻找最 critical 的少数 terms，同时设计了一套全新的 sphere/sphere parametrization 来处理 180 度的 extreme light bending。

下面我为你详细拆解这篇 paper 的技术细节。

### 1. Light Field Parametrization 与 Coordinate Spaces

要将一根 ray 从 sensor 传到 lens 外部，我们需要描述 5D light field：2D position, 2D direction, 以及 1D wavelength ($\lambda$)。Paper 中定义了两个 mapping $P_a(\mathbf{S})$ 和 $P_o(\mathbf{S})$：

$$P_a(\mathbf{S}) : (x_s, y_s, dx_s, dy_s, \lambda) \mapsto (x_a, y_a, dx_a, dy_a, \tau_a) \quad \text{(Eq. 1)}$$
$$P_o(\mathbf{S}) : (x_s, y_s, dx_s, dy_s, \lambda) \mapsto (x_o, y_o, dx_o, dy_o, \tau_o) \quad \text{(Eq. 2)}$$

这里，下标 $s, a, o$ 分别代表 sensor, aperture, outer pupil。变量 $x, y$ 是 position，$dx, dy$ 是 direction。$\tau$ 是 Fresnel transmittance（衡量光在 lens surface 反射导致的能量损失）。$\mathbf{S}$ 是输入的 5D sensor ray。

这里有一个非常关键的 design choice：对于 sensor 和 aperture，paper 使用了 plane/plane parametrization。这意味着一个 ray 被定义为它与相距 $dz=1$ 的两个平行平面的交点。这种 parametrization 的巨大好处是 free space propagation 可以用一个 linear polynomial 精确表达，使得我们可以通过简单地平移 sensor 来实现 refocus，而完全不需要重新计算多项式。

但是对于 outer pupil（lens 的最外层表面），由于 fisheye lens 的 field of view 接近 180 度，传统的 plane/plane parametrization 会在 90 度处产生 singularity。为了解决这个问题，authors 引入了 sphere/sphere parametrization。

在 sphere/sphere parametrization 中，position $(x_o, y_o)$ 是 outer pupil 球面上的点在 optical axis 垂直面上的投影。Direction $(dx_o, dy_o)$ 是 ray 在球面交点处的局部 tangent plane 上的半球投影。为了构建这个局部 tangent frame，paper 给出了转换矩阵 $\mathbf{T}(\mathbf{n})$ (Eq. 3)：

$$
\mathbf{T}(\mathbf{n}) = \left[ \begin{array}{ccc} n_z / l & -n_x n_y / l & n_x \\ 0 & l & n_y \\ -n_x / l & -n_y n_z / l & n_z \end{array} \right], \quad l := \sqrt{n_x^2 + n_z^2}
$$

这里的 $\mathbf{n} = (n_x, n_y, n_z)$ 是球面交点处的法向量。矩阵的三列分别是 tangent vector (在 x-z plane 内且垂直于 $\mathbf{n}$)、bitangent vector (通过 cross product 得到) 和 normal vector $\mathbf{n}$ 本身。$l$ 是一个归一化因子。为了避开 optical axis 上的 singularity，authors 将极点设置在了 $\pm y$ 轴方向，确保了正对镜头中心的光线在参数化时不会产生数值不稳定。

### 2. Sparse Polynomial Construction 与 OMP Algorithm

这是这篇 paper 最具启发性的部分。我们想要拟合 high-degree polynomial 来捕捉 aspherical lenses 的复杂 aberrations。一个 degree $d$ 的 5 变量 polynomial，其 term 的数量 $N(d) = \binom{n+d}{d}$。当 $d=11$ 时，$N(d) = 4368$。如果对于 5 个 output variable 都拟合，会有几万个 coefficients。这在 Monte Carlo path tracing 中每条 ray 都要计算一次，是不可接受的。

Paper 的核心 insight 是：在这个巨大的 monomial space 中，只有极少数的 terms 对最终的误差贡献最大。这类似于 neural network 中的 pruning 或者 sparse regression。

构建 polynomial 的 term 形式如下：
$$c \cdot x_s^{d_0} y_s^{d_1} dx_s^{d_2} dy_s^{d_3} \lambda_s^{d_4} \quad \text{with degree} \sum_{i=0}^4 d_i \leq d \quad \text{(Eq. 4)}$$
这里 $c$ 是系数，$d_0 \dots d_4$ 是各个输入变量的幂次。

为了找到 sparse solution $\mathbf{c}'$，authors 构建了一个 $M \times N(d)$ 的设计矩阵 $\hat{\Phi}$ (Eq. 6)。$M$ 是 ray tracing 采样得到的 ground truth rays 的数量（通常设置为 $10 \cdot N(d)$ 以保证 overdetermined）。矩阵的每一行是一个 sample，每一列是一个 monomial term 的求值。

目标是最小化 $\|\hat{\Phi} \cdot \mathbf{c}' - \mathbf{b}\|_2$，其中 $\mathbf{b}$ 是 ray traced 的真实 output light field。Authors 使用了 Orthogonal Matching Pursuit (OMP) 的变体，如 Algorithm 1 所示。

标准的 OMP 是一种 greedy 算法，每次迭代选择与当前 residual 最相关的一个 column (monomial term) 加入 active set，然后重新做 least squares 拟合。但这篇 paper 做了两个关键改进：
1.  **Exact error calculation**: 在选择新 column 时，计算加入该 column 后进行 least squares 拟合的真实 squared error，依赖于 dot product 进行近似估计。
2.  **Replacement (替换机制)**: 当 active set 达到预设的 sparsity $s$ (比如 40 个 terms) 后，不再单纯添加新 term，而是寻找 active set 中最不重要的 term，测试将其替换为新的 candidate term 是否能降低 total error。这防止了算法在早期锁定了一个看似好但实际上 globally suboptimal 的 term。

**Intuition building**: 想象你在用一堆 basis functions 拟合一个复杂的波形。Taylor series 就像是你必须按照 $1, x, x^2, x^3 \dots$ 的顺序使用 basis，哪怕 $x^7$ 完全没用你也得带着它。OMP 就像是一个 sparse autoencoder，你有一个巨大的 dictionary of basis functions (所有 degree 11 的组合)，你只挑出那些能 reconstruct signal 的最关键的 40 个 basis。

从 Table 1 的实验数据可以看出，对于 fisheye-aspherical lens，degree 4 的 Taylor polynomial (28 terms) 误差是 $1.52 \times 10^{-2}$，而他们的 Sparse polynomial (同样 28 terms) 误差降到了 $6.26 \times 10^{-3}$，甚至比包含 126 个 terms 的 Complete degree 4 polynomial 效果还要好。这证明了 sparse selection 在 high-degree space 中找到了远比 Taylor truncaction 优秀的 basis representation。

### 3. Aperture Sampling 与 Light Tracing

在 Monte Carlo rendering 中，path tracing (从 camera 向 scene 发射 ray) 很简单：sample sensor point，sample aperture point，算出 sensor direction，然后 evaluate polynomial 即可。但是 light tracing (从 light source 向 camera 连接) 非常困难。因为如果像之前的工作 [HD14] 那样 sample outer pupil，很多 ray 会被内部的 aperture 物理遮挡，对于 fisheye lens 这种现象极其严重（见 Figure 5，aperture 在 outer pupil 上的投影非常小）。

为了解决这个问题，paper 提出了直接 sample aperture 的算法 (Algorithm 2)。给定一个 scene 中的点 $\hat{\mathbf{o}}$ 和一个 sample 出的 aperture point $\hat{\mathbf{A}}_{xy}$，我们需要反向求出 sensor 上的 position 和 direction $\mathbf{S} = (x_s, y_s, dx_s, dy_s)$，使得这根 ray 刚好穿过 $\hat{\mathbf{A}}_{xy}$ 并击中 $\hat{\mathbf{o}}$。

这是一个 root-finding 问题。由于 polynomial 是非线性的，authors 使用了一个 two-stage Newton iteration：

*   **Stage 1**: 固定 sensor position，更新 sensor direction。
    我们需要算出 sensor direction $\mathbf{S}_{\omega}$ 的修正量。利用 polynomial 的解析 Jacobian $J_a = \mathrm{d}\mathbf{A}_{xy}/\mathrm{d}\mathbf{S}_{\omega}$，通过 $\mathbf{S}_{\omega} \gets \mathbf{S}_{\omega} + J_a^{-1} \cdot \Delta\mathbf{A}_{xy}$ 来更新。这里 $\Delta\mathbf{A}_{xy}$ 是当前 ray 穿过的 aperture point 与目标 $\hat{\mathbf{A}}_{xy}$ 的误差。

*   **Stage 2**: 固定 sensor direction，更新 sensor position。
    将当前 ray 传到 outer pupil 得到 $\mathbf{O}$，计算它指向目标 $\hat{\mathbf{o}}$ 的方向误差 $\Delta\mathbf{O}_{\odot}$。利用 Jacobian $J_o = \mathrm{d}\mathbf{O}_{\omega}/\mathrm{d}\mathbf{S}_{xy}$，通过 $\mathbf{S}_{xy} \gets \mathbf{S}_{xy} + J_o^{-1} \cdot \Delta\mathbf{O}_{\odot}$ 来更新 sensor position。

这个 two-stage Newton method 利用了 polynomial 的解析导数，收敛极快。为了在 path tracing 和 light tracing 之间保持 consistency，必须对 sampling 密度进行 Jacobian 变换 (Eq. 7, 8, 9)。

对于 path tracing，estimator 贡献是：
$$c_{pt} = \frac{f}{p} = W(\mathbf{S}) / \left( p(\mathbf{A}_{xy}) \left\| \frac{\mathrm{d}\mathbf{S}_{\omega}}{\mathrm{d}\mathbf{A}_{xy}} \right\| \right) \quad \text{(Eq. 7)}$$
这里 $W(\mathbf{S})$ 是 sensor responsivity，$p(\mathbf{A}_{xy})$ 是 aperture 上的采样 PDF。Jacobian $\|\mathrm{d}\mathbf{S}_{\omega}/\mathrm{d}\mathbf{A}_{xy}\|$ 将 aperture area measure 转换为 sensor solid angle measure。

对于 light tracing，因为现在是在 aperture 上采样并连接到 outer pupil，estimator 变成：
$$c_{lt} = \frac{f}{p} = W(\mathbf{S}) / \left( p(\mathbf{A}_{xy}) \cdot \|J_{\nu}\| \cdot \left\| \frac{\mathrm{d}\mathbf{O}_{xy}}{\mathrm{d}\mathbf{A}_{xy}} \right\| \right) \quad \text{(Eq. 8)}$$
这里多了一个 $\|J_{\nu}\|$，这是球面几何的 Jacobian。$\|J_{\nu}\| = \cos \theta = \sqrt{R - \mathbf{O}_x^2 - \mathbf{O}_y^2 / R}$，其中 $R$ 是 outer pupil 最后一个 lens element 的曲率半径。这个 term 将 outer pupil 上的 projected disk area measure 转换为球面上的 vertex area measure。

$\mathrm{d}\mathbf{O}_{xy}/\mathrm{d}\mathbf{A}_{xy}$ 的计算则是通过 shared measurement space $\mathrm{d}\mathbf{S}_{\omega}$ 传递的 (Eq. 9)：
$$\left\| \mathrm{d}\mathbf{O}_{xy} / \mathrm{d}\mathbf{A}_{xy} \right\| = \left\| \mathrm{d}\mathbf{O}_{xy} / \mathrm{d}\mathbf{S}_{\omega} \right\| \cdot \left\| \mathrm{d}\mathbf{S}_{\omega} / \mathrm{d}\mathbf{A}_{xy} \right\|$$
这两个 Jacobian 分量可以直接从 $P_o(\mathbf{S})$ 和 $P_a(\mathbf{S})$ 的解析微分中提取出 $2 \times 2$ 的 sub-matrices 得到。这种数学上的严谨性保证了 bidirectional path tracing 中 measurement 的正确性。

### 4. Interactive Preview 与 Implementation Details

Sparse polynomial 的另一个巨大优势是计算极快，可以在 GPU 上做 interactive preview (Algorithm 3)。给定一个 RGB-D texture (包含深度信息)，直接在 GLSL shader 中 evaluate sparse polynomial。利用 min/max mipmap 结构，ray marching 可以大步跨越空白区域，极大地减少了 texture fetches。在 AMD Radeon R9 390 上，1080x720 分辨率 144 samples per pixel 只需要 137 ms。

在 implementation 方面，由于 polynomial degree 高达 11，设计矩阵 $\hat{\Phi}$ 会非常庞大。Authors 强调必须使用 double precision (单精度 float 会导致数值崩溃，因为不同 monomials 之间的 dynamic range 极大)。Table 2 对比了不同 OMP 变体的耗时与误差。带有 replacement 和 exact fit 的 OMP 虽然单线程拟合耗时较长 (长达近 7 分钟)，但这只是一次性的 precompute，换来的是渲染时极低的 error 和极少的 evaluation cost。

### 总结与 Intuition 升华

这篇 paper 的精髓在于它把光学系统的 ray tracing 抽象为了一个 high-dimensional non-linear regression problem。

*   **Taylor series** 是一种 "what if" 式的局部展开，它强依赖于展开点（光轴原点），一旦远离原点（fisheye 边缘）就会 diverge。
*   **Sparse polynomial (OMP)** 是一种 global feature selection，它直接探索整个 function space，挑出那些最能描述光学畸变的 "eigenterms"。

结合 sphere/sphere parametrization 解决了大角度的奇点问题，以及精巧的 two-stage Newton iteration 解决了 light tracing 的采样难题，这使得基于物理的复杂镜头模拟能够真正在 production rendering 中落地。

**References for further reading:**
*   Original Polynomial Optics paper (Hullin et al. 2012): https://dl.acm.org/doi/10.1111/j.1467-8659.2012.03126.x
*   Hanika's previous work on efficient Monte Carlo rendering with realistic lenses (HD14): https://jo.dreggn.de/home/2014-lens.pdf
*   Orthogonal Matching Pursuit basics (Tropp & Gilbert 2007): https://arxiv.org/abs/math/0506403
*   OMP with replacement (Jain et al. 2011): https://proceedings.neurips.cc/paper/2011/hash/f229a6e8e308e2c8fade3affdbbf5c5f-Abstract.html
