---
source_pdf: Transforming Normals.pdf
paper_sha256: b59359914869ffe206af3d164b449044773696e0351d5ed42c12029566d193ce
processed_at: '2026-08-12T18:10:48-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

## 一句话版本

Normal 看着像一根棍子（vector），其实它是一块小面积的方向（antivector），所以变换它要用 adjugate transpose，传统用的 inverse transpose 只是个"省事但不稳"的近似。

---

## 为什么 normal 和 tangent 变换方式不同

拿一张纸放在桌上，纸面上画一根箭头沿着纸面走，那根箭头就是 tangent vector。再拿一根牙签垂直扎进纸面，牙签的方向就是 normal。tangent 和 normal 看起来都是"一根方向"，但它们本质是两种东西：

- tangent 是"沿着某个方向走多远"——本质是长度量，grade 1；
- normal 是"这块小纸面朝哪边"——本质是面积量，grade 2（在 3D 里）。

你拿手把纸横向拉长两倍，纸上画的箭头（tangent）横向变长两倍，很自然。但扎在纸上的牙签方向要怎么变？如果你用同一个矩阵直接乘它，会发现牙签和纸面不再垂直了——因为纸被拉长，几何关系变了，normal 必须用另一套规则"补偿"才行。

这就是为什么 normal 需要特殊变换。它和 tangent 用同一个矩阵 $\mathbf{m}$ 直接乘会出错，得换一个矩阵。

参考：https://www.reedbeta.com/blog/mathematical-roots-of-the-normal-matrix/

---

## 经典解法：inverse transpose

图形学教科书里满天飞的说法：用 $\mathbf{m}^{-\mathrm{T}}$（inverse transpose）。

推导很短：只要 normal 变换后还跟 tangent 垂直就行，列出方程 $(\mathbf{x}\mathbf{n})\cdot(\mathbf{m}\mathbf{t})=0$，解出来 $\mathbf{x}=\mathbf{m}^{-\mathrm{T}}$。

这套在 99% 的场景能用，但有两个坑：

1. **$\mathbf{m}$ 不可逆时直接挂**。比如把 mesh 投影到一个平面（rank-deficient matrix）、或某个轴 scale 成 0，inverse 不存在，shader 里 `inverse()` 返回 NaN，整个 lighting 崩掉。
2. **$\mathbf{m}$ 含 reflection 时方向会"骗人"**。inverse transpose 把 $\det\mathbf{m}$ 这个因子除掉了，你看不到 reflection 信号，normal 会指向错误的那一面，double-sided lighting 判断出错。

参考：https://computergraphics.stackexchange.com/questions/1502/why-does-my-normal-matrix-transform-not-work

---

## Lengyel 的更好解法：adjugate transpose

Lengyel 换了一个出发点，不要"保持垂直"这个间接条件，直接强制"保持叉积"。

逻辑很自然：normal 的定义就是 $\mathbf{n}=\mathbf{s}\times\mathbf{t}$，两个 tangent 叉积出来。变换完之后，normal 应该等于变换完的两个 tangent 的叉积：
$$\mathbf{x}\mathbf{n}=\mathbf{m}\mathbf{s}\times\mathbf{m}\mathbf{t}$$

把这个叉积展开（paper 里的方程 1），会自然冒出来一个矩阵，它的三列是 $\mathbf{m}$ 的列两两叉积。这个矩阵的转置满足 $\mathbf{x}^{\mathrm{T}}\mathbf{m}=(\det\mathbf{m})\mathbf{I}$，线性代数里这个 $\mathbf{x}^{\mathrm{T}}$ 有专门的名字——**adjugate**，记作 $\mathrm{adj}(\mathbf{m})$。

所以正确的变换矩阵是：
$$\mathbf{x}=\mathrm{adj}(\mathbf{m})^{\mathrm{T}}$$

**它和 inverse transpose 差什么？** 一个标量因子 $\det\mathbf{m}$：
$$\mathrm{adj}(\mathbf{m})=(\det\mathbf{m})\,\mathbf{m}^{-1}$$

normal 变换完通常要 renormalize，乘一个标量不影响方向，所以两个矩阵在大部分场景"看起来一样"。但 adjugate transpose 有三个工程优势：

- **不除 det，永远算得出**：adjugate 是 9 个 2×2 余子式加正负号拼出来的，纯多项式，不需要除法。$\mathbf{m}$ singular 时返回 0，不会 NaN。
- **暴露 reflection 信号**：$\det\mathbf{m}<0$ 时 adjugate transpose 会带这个负号，shader 里检查 `dot(n_transformed, n_original)` 的符号就知道是不是翻面了，inverse transpose 把这个信号抹掉了。
- **跟 Plücker line、plane 的变换在同一个框架里**：见下面。

参考：https://en.wikipedia.org/wiki/Adjugate_matrix

---

## 为什么 normal "其实不是 vector"——exterior algebra 的视角

这是 paper 真正想推的洞察。

在 3D 空间里，除了 scalar 和 vector，还有更"高阶"的几何对象：

- grade 0：scalar（一个数，比如温度）
- grade 1：vector（一个方向 + 长度，比如力、速度、tangent）
- grade 2：bivector（一块带方向的面积，比如一小块纸面、力矩、磁场 B）
- grade 3：trivector（一块带方向的体积，比如一小块流体微元）

这就是 exterior algebra（外代数，也叫 Grassmann algebra）。任意 $n$ 维空间有 $n+1$ 种 grade，总共有 $2^n$ 个基元素。3D 是 $1+3+3+1=8$ 个，4D 是 $1+4+6+4+1=16$ 个。

**关键观察**：normal 是一块面积的朝向，它是 bivector（3D 下 grade 2，一般 $n$ 维下 grade $n-1$），叫 antivector。它根本不是 vector。

那为什么我们在代码里一直拿它当 vector 处理？因为 3D 有个"巧合"：$\binom{3}{2}=3$，bivector 的维数和 vector 的维数都是 3，可以一一对应。这个对应叫 Hodge dual。**cross product 本质是 wedge product 加 Hodge dual 的组合**，它把本来是 bivector 的结果"翻译"成 vector 给你看。

所以 3D 里 `s × t` 表面上吐出来一个 vector，其实底下是个 bivector。一旦你去 2D 或 4D，cross product 就不存在了（只有 3D 和 7D 有这种"乘出来还是同维向量"的好事），但 wedge product 永远在。normal 的本质就暴露了——它是 wedge product 的产物，是 bivector。

参考：https://en.wikipedia.org/wiki/Exterior_algebra, https://en.wikipedia.org/wiki/Hodge_star_operator

---

## Exomorphism：把 $\mathbf{m}$ 扩展成一个大矩阵 $\mathbf{M}$

既然 vector 是 grade 1，bivector 是 grade 2，trivector 是 grade 3……那一个普通的 $n\times n$ 变换矩阵 $\mathbf{m}$（只管 grade 1）能不能扩展成一个"管所有 grade"的大矩阵？

能。要求很简单：变换得是 wedge product 的 homomorphism，也就是说
$$\mathbf{M}(\mathbf{a}\wedge\mathbf{b})=(\mathbf{M}\mathbf{a})\wedge(\mathbf{M}\mathbf{b})$$
"先 wedge 再变换"等于"先变换再 wedge"。这一个约束就把整个大矩阵 $\mathbf{M}$ 唯一确定了。

$\mathbf{M}$ 是个 $2^n\times 2^n$ 的块对角矩阵，对角线上有 $n+1$ 个块，每个块对应一个 grade。第 $k$ 个块叫 $C_k(\mathbf{m})$，叫 **$k$-th compound matrix**，大小 $\binom{n}{k}\times\binom{n}{k}$：

- $C_0(\mathbf{m})=[1]$：scalar 不变
- $C_1(\mathbf{m})=\mathbf{m}$：vector 按 $\mathbf{m}$ 变
- $C_2(\mathbf{m})$：bivector 按"列两两 wedge"变
- $C_{n-1}(\mathbf{m})=\mathrm{adj}(\mathbf{m})^{\mathrm{T}}$：**antivector，就是 normal 的变换矩阵**
- $C_n(\mathbf{m})=[\det\mathbf{m}]$：体积元素按 det 缩放

3D 下的 8×8 $\mathbf{M}$ 长这样（对角块）：
$$\mathbf{M}=\mathrm{diag}\big([1],\ \mathbf{m},\ \mathrm{adj}(\mathbf{m})^{\mathrm{T}},\ [\det\mathbf{m}]\big)$$

paper 最核心的一句话就藏在里面：**antivector 的变换矩阵 $C_{n-1}$ 永远是 adjugate transpose**，不管几维。这就是 Lengyel 那个解法不是"凑出来的 trick"，而是 exterior algebra 框架里的自然结果。

参考：https://en.wikipedia.org/wiki/Compound_matrix

---

## 4D 投影空间：点、线、面全统一

图形学实际用 4D 齐次坐标：
- 点 $(x,y,z,w)$ 是 grade 1（4 维）
- 线 Plücker 坐标 $(l_{vx},l_{vy},l_{vz},l_{mx},l_{my},l_{mz})$ 是 grade 2（6 维）
- 平面 $(g_x,g_y,g_z,g_w)$ 是 grade 3（4 维）
- volume 是 grade 4（1 维）

正好对应 4D exterior algebra 的 1, 4, 6, 4, 1 维结构。所以 $\mathbf{M}$ 是 16×16 块对角矩阵，5 个块：

| grade | 对象 | 维度 | 变换矩阵 |
|---|---|---|---|
| 0 | scalar | 1 | $[1]$ |
| 1 | point | 4 | $\mathbf{m}$ |
| 2 | line | 6 | $C_2(\mathbf{m})$ |
| 3 | plane | 4 | $C_3(\mathbf{m})$ |
| 4 | volume | 1 | $[\det\mathbf{m}]$ |

paper 给了两个具体例子，验证这套公式跟经典图形学公式完全一致。

**Translation 例子**：$\mathbf{m}$ 是平移矩阵。算出 $C_2(\mathbf{m})$ 之后会发现——line 的 direction 不变，moment 加上 $\mathbf{t}\times\mathbf{v}$。这跟 Plücker 坐标在平移下的经典公式一模一样。算出 $C_3(\mathbf{m})$ 之后会发现——plane 的 normal 不变，offset 减去 $\mathbf{t}\cdot\mathbf{n}$。这也是经典公式。

**Nonuniform scale 例子**：$\mathbf{m}=\mathrm{diag}(s_x,s_y,s_z,1)$。算出 $C_2$ 后发现 line direction 按 $(s_x,s_y,s_z)$ 缩放（grade 1，长度量），moment 按 $(s_y s_z, s_z s_x, s_x s_y)$ 缩放（grade 2，面积量）。$C_3$ 平面 normal 按两两乘积缩放（面积量），offset 按 $s_x s_y s_z$ 缩放（体积量）。

**直觉**：grade 就是"长度量纲的次数"。grade 1 缩一次，grade 2 缩两次（面积），grade 3 缩三次（体积）。nonuniform scale 下不同 grade 的分量按不同方式缩放，这就是为什么 normal 和 tangent 行为不同。

参考：https://en.wikipedia.org/wiki/Pl%C3%BCcker_coordinates

---

## 这套框架给你什么直觉

1. **Normal 是面积，不是棍子**。3D 里 cross product 假装成 vector，本质是 bivector。一般维度下 cross 不存在，wedge 才是普适语言。

2. **Adjugate transpose 是 normal 的"正确"变换矩阵**，inverse transpose 是它在 $\det\neq 0$ 且无 reflection 时的简化。singular 和 reflection 是真实场景（shadow projection、mirror、scale 接近 0），用 adjugate 更稳。

3. **Reflection 检测免费送**：$\det\mathbf{m}<0$ 时 adjugate transpose 自带负号，shader 里检查一下就知道要不要 flip normal。inverse transpose 把这个信号除掉了。

4. **Compound matrix 是"嵌套的 adjugate"**：$C_k(\mathbf{m})$ 每个元素是 $\mathbf{m}$ 某个 $k\times k$ 子矩阵的 det。$k=n-1$ 时就是 adjugate，$k=1$ 就是 $\mathbf{m}$，$k=0$ 是 1，$k=n$ 是 det。所以 adjugate transpose 是这套结构里的一个特例。

5. **所有"特殊变换"都在同一个框架里**：normal、Plücker line、plane、volume element、电磁场的 $\mathbf{B}$（bivector）、流体的 vorticity（bivector）……都用同一个 $\mathbf{M}$ 的不同对角块。物理仿真里用 bivector 表达"绕轴的旋转量"时，变换就要用 $C_2$，不能直接用 $\mathbf{m}$。

6. **Grade = 量纲 = 在 nonuniform scale 下的缩放规律**。这是判断一个量属于哪个 grade 最直观的方法。长度量按一个 $s$ 缩，面积量按两个 $s$ 乘积缩，体积量按三个 $s$ 乘积缩。

---

## 工程上换一行代码的事

shader 里把
```glsl
mat3 normalMatrix = transpose(inverse(mat3(modelMatrix)));
```
换成
```glsl
mat3 normalMatrix = transpose(adjugate(mat3(modelMatrix)));
```
`adjugate` 自己写一个函数，9 个 2×2 余子式 + checkerboard 符号，比 `inverse` 还简单（不除法），且 singular/reflection 都安全。

完整推导和代码：https://projectivegeometricalgebra.org, https://terathon.com/blog/transforming-normals/

---

## 一句话再总结

Normal 在 3D 里被 cross product "伪装"成 vector，本质是 grade-2 的 bivector（一块面积的方向）。它要按 antivector 的规则变换，正确的矩阵是 adjugate transpose，传统用的 inverse transpose 只是 adjugate transpose 除掉一个 det 之后的近似，在 singular 和 reflection 时会出问题。这件事放在更大的 exterior algebra 框架里看：$\mathbf{m}$ 扩展成 $2^n\times 2^n$ 的 exomorphism matrix $\mathbf{M}$，对角线上的 $C_{n-1}$ 块永远等于 $\mathrm{adj}(\mathbf{m})^{\mathrm{T}}$。同一个框架里 $C_2$ 管 Plücker line，$C_3$ 管 plane，$C_n$ 管体积。所有"特殊变换"在 exterior algebra 下统一成一个故事——grade 决定量纲，量纲决定变换规律。

---

# Transforming Normals —— Eric Lengyel 的 paper 详解

这篇 paper 来自 Eric Lengyel（terathon.com / C4 Engine 作者）2024 年 12 月的 blog，本质是他书 *Projective Geometric Algebra Illuminated* 第 2.7 节 "Exomorphisms" 的摘录和动机说明。核心论点非常 sharp：**normal 向量之所以要"特殊变换"，根因是它根本不是 vector，而是 exterior algebra 里的 antivector (grade n−1)**；正确的变换矩阵是 **adjugate transpose** $\mathrm{adj}(\mathbf{m})^{\mathrm{T}}$，而不是图形学教科书里满天飞的 inverse transpose $\mathbf{m}^{-\mathrm{T}}$。后者只是前者在 $\det\mathbf{m}\neq 0$ 时差一个标量因子的特例，在 singular matrix 或 reflection 情况下会出问题。

下面我按 paper 的逻辑链逐步展开，尽量把每个公式的符号、上标下标含义、以及背后的几何直觉都拆出来。

---

## 1. 问题设置与经典解法

设 $\mathbf{t}$ 是 surface 上某点处的 tangent vector（列向量），$\mathbf{m}$ 是 $3\times3$ 的变换矩阵，普通向量按 $\mathbf{t}'=\mathbf{m}\mathbf{t}$ 变换。$\mathbf{n}$ 是该点处的 normal，与 $\mathbf{t}$ 垂直，即 $\mathbf{n}\cdot\mathbf{t}=0$，等价写成 $\mathbf{n}^{\mathrm{T}}\mathbf{t}=0$（行向量 × 列向量 = 标量）。

如果直接算 $\mathbf{n}'=\mathbf{m}\mathbf{n}$，当 $\mathbf{m}$ 含 nonuniform scale 或 skew（非正交）时，$\mathbf{t}'$ 和 $\mathbf{n}'$ 的点积不再为零，垂直性破坏。Paper 中那个水平方向放两倍的三角形例子就是经典反例：normal 还指"斜上方"，但斜边已经变陡，二者不再垂直。

经典解法：找一个矩阵 $\mathbf{x}$ 使得
$$(\mathbf{x}\mathbf{n})\cdot(\mathbf{m}\mathbf{t})=(\mathbf{x}\mathbf{n})^{\mathrm{T}}\mathbf{m}\mathbf{t}=\mathbf{n}^{\mathrm{T}}\mathbf{x}^{\mathrm{T}}\mathbf{m}\mathbf{t}=0$$
对任意 $\mathbf{n}^{\mathrm{T}}\mathbf{t}=0$ 成立。最直接的取法是 $\mathbf{x}^{\mathrm{T}}\mathbf{m}=\mathbf{I}$，即 $\mathbf{x}=\mathbf{m}^{-\mathrm{T}}$。这就是 inverse transpose。

**两个 bug：**
- $\mathbf{m}$ singular（$\det\mathbf{m}=0$，例如 projection、退化 scale）时 $\mathbf{m}^{-1}$ 不存在，公式直接挂掉；
- $\mathbf{m}$ 含 reflection（$\det\mathbf{m}<0$）时，inverse transpose 的方向会出问题，paper 没有展开，但下一节会看到 adjugate transpose 自然带 $\det\mathbf{m}$ 因子，reflection 会让法线"翻面"，这点必须在外部检测处理。

---

## 2. Lengyel 的 derivation：从 cross product 不变性出发

Lengyel 换了一条更结实的路：**强制 cross product 在变换下保持**。

设 surface 上有两个不平行 tangent $\mathbf{s},\mathbf{t}$，则 $\mathbf{n}=\mathbf{s}\times\mathbf{t}$。无论用什么矩阵 $\mathbf{x}$ 变换 $\mathbf{n}$，都必须满足
$$\mathbf{x}\mathbf{n}=\mathbf{m}\mathbf{s}\times\mathbf{m}\mathbf{t}$$
即"变换后的 normal"等于"变换后的两个 tangent 的叉积"。这是 normal 的几何定义本身。

把 $\mathbf{m}\mathbf{s}$ 和 $\mathbf{m}\mathbf{t}$ 按列展开。记 $\mathbf{m}_{[j]}$ 为 $\mathbf{m}$ 的第 $j$ 列（zero-based），$\mathbf{s}=(s_x,s_y,s_z)^{\mathrm{T}}$，则
$$\mathbf{m}\mathbf{s}=s_x\mathbf{m}_{[0]}+s_y\mathbf{m}_{[1]}+s_z\mathbf{m}_{[2]}$$
同理 $\mathbf{m}\mathbf{t}=t_x\mathbf{m}_{[0]}+t_y\mathbf{m}_{[1]}+t_z\mathbf{m}_{[2]}$。

把 cross product 分配进去：
$$\mathbf{x}\mathbf{n}=(s_x\mathbf{m}_{[0]}+s_y\mathbf{m}_{[1]}+s_z\mathbf{m}_{[2]})\times(t_x\mathbf{m}_{[0]}+t_y\mathbf{m}_{[1]}+t_z\mathbf{m}_{[2]})$$

展开化简（叉积 $\mathbf{a}\times\mathbf{a}=0$，且 $\mathbf{a}\times\mathbf{b}=-\mathbf{b}\times\mathbf{a}$）后得到 paper 中的方程 (1)：

$$
\begin{aligned}
\mathbf{x}\mathbf{n}=&(s_y t_z-s_z t_y)(\mathbf{m}_{[1]}\times\mathbf{m}_{[2]})\\
&+(s_z t_x-s_x t_z)(\mathbf{m}_{[2]}\times\mathbf{m}_{[0]})\\
&+(s_x t_y-s_y t_x)(\mathbf{m}_{[0]}\times\mathbf{m}_{[1]})
\end{aligned}\tag{1}
$$

**直觉拆解：**
- 系数 $(s_y t_z-s_z t_y)$、$(s_z t_x-s_x t_z)$、$(s_x t_y-s_y t_x)$ 恰好就是 $\mathbf{s}\times\mathbf{t}$ 的三个分量，也就是原 normal $\mathbf{n}=(n_x,n_y,n_z)^{\mathrm{T}}$ 的三个分量；
- 右侧三个向量 $\mathbf{m}_{[1]}\times\mathbf{m}_{[2]}$、$\mathbf{m}_{[2]}\times\mathbf{m}_{[0]}$、$\mathbf{m}_{[0]}\times\mathbf{m}_{[1]}$ 就是 $\mathbf{x}$ 的三列。

所以
$$\mathbf{x}=\big[\,\mathbf{m}_{[1]}\times\mathbf{m}_{[2]}\ \ \mathbf{m}_{[2]}\times\mathbf{m}_{[0]}\ \ \mathbf{m}_{[0]}\times\mathbf{m}_{[1]}\,\big]$$

这个矩阵的转置 $\mathbf{x}^{\mathrm{T}}$ 满足
$$\mathbf{x}^{\mathrm{T}}\mathbf{m}=(\det\mathbf{m})\mathbf{I}$$
因为对任何 $\{i,j,k\}$ 是 $\{0,1,2\}$ 的偶排列，$(\mathbf{m}_{[i]}\times\mathbf{m}_{[j]})\cdot\mathbf{m}_{[k]}=\det\mathbf{m}$（这就是 triple product 公式 $\det\mathbf{m}=\mathbf{m}_{[0]}\cdot(\mathbf{m}_{[1]}\times\mathbf{m}_{[2]})$ 的全部偶置换）。

满足 $\mathbf{A}^{\mathrm{T}}\mathbf{m}=(\det\mathbf{m})\mathbf{I}$ 的矩阵 $\mathbf{A}=\mathbf{x}^{\mathrm{T}}$ 定义上就叫 **adjugate** $\mathrm{adj}(\mathbf{m})$。所以
$$\boxed{\mathbf{x}=\mathrm{adj}(\mathbf{m})^{\mathrm{T}},\quad \mathbf{n}'=\mathrm{adj}(\mathbf{m})^{\mathrm{T}}\mathbf{n}}$$

**关键关系：** 当 $\mathbf{m}$ 可逆时
$$\mathrm{adj}(\mathbf{m})=(\det\mathbf{m})\mathbf{m}^{-1}\;\;\Longrightarrow\;\;\mathrm{adj}(\mathbf{m})^{\mathrm{T}}=(\det\mathbf{m})\mathbf{m}^{-\mathrm{T}}$$
两者差一个标量 $\det\mathbf{m}$。对 normal 而言我们一般事后会 renormalize，所以方向不变，结果一致。这就是为什么 inverse transpose 在大多数场景"看起来对"。

**但 adjugate transpose 更稳：**
- $\mathbf{m}$ singular：$\mathbf{m}^{-1}$ 不存在，但 $\mathrm{adj}(\mathbf{m})$ 永远存在（它由 minors 的代数余子式组成，是多项式，不需要除法）。这是工程上的实际意义，比如把一个 mesh 投影到平面后再算 lighting，adjugate 不会爆 NaN。
- $\det\mathbf{m}<0$（reflection）：inverse transpose 仍给出一个"看起来正确"的方向，但与变换后的几何体的手性已经反了。adjugate transpose 带 $\det\mathbf{m}$ 因子，符号会暴露这个 reflection，调用方可以据此 flip normal 或决定是否启用 double-sided lighting。

paper 还顺带吐槽：Graphics Gems Vol.1 (1990) Ron Goldman 的 "Matrix Inversion" 章节里最早提到 adjugate 可变换 normal，但 (1) 没解释为什么对，(2) 把 $M^*=(1/\det M)M^{-1}$ 写反了，应该是 $M^*=\det(M)\,M^{-1}$。这是早期文献一个流传甚广的笔误。

---

## 3. 更大的图景：normal 不是 vector，是 antivector

这是 paper 真正想推的视角。

在 $n$ 维 vector space 上构建 **exterior algebra**（ Grassmann algebra），有 $\binom{n}{k}$ 个 grade-$k$ 基，总共 $2^n$ 个基元素。grade 分布：
- grade 0：scalar（数量，1 维）
- grade 1：普通 vector（$n$ 维）
- grade 2：bivector（$\binom{n}{2}$ 维，即 oriented area）
- …
- grade $k$：$k$-vector（$\binom{n}{k}$ 维，oriented $k$-volume）
- grade $n-1$：**antivector**（$n$ 维，oriented $(n-1)$-volume，也就是 hyperplane 法向）
- grade $n$：antiscalar（1 维，pseudo-scalar / 体积元素）

3D 中 cross product 之所以"恰好"产生一个 vector，是因为 $\binom{3}{2}=3$，bivector 和 vector 维数凑巧相等，可以用 Hodge dual 把 bivector $\mathbf{e}_{23},\mathbf{e}_{31},\mathbf{e}_{12}$ 与 vector $\mathbf{e}_1,\mathbf{e}_2,\mathbf{e}_3$ 一一对应。**cross product 本质上是 wedge product + Hodge dual**，伪装成 vector 的 bivector。

normal 在物理/几何意义上代表"无穷小面积元的方向"，本质是 bivector（3D 中）、$(n-1)$-vector（一般 $n$ 中）。所以它要按 grade-$n-1$ 的规则变换，而不是 grade-1 的规则。这就解释了"为什么 normal 和 tangent 变换不同"——它们根本不是同一个 grade。

wedge product 在任意维度都有定义，cross product 只在 3D（和 7D 有特殊形式）成立，所以一般化必须用 wedge product。

---

## 4. Exomorphism matrix：把 m 扩展为 $2^n\times 2^n$ 的块对角矩阵

记大写 $\mathbf{M}$ 为 exomorphism matrix（外同态矩阵），它是把 grade-1 变换 $\mathbf{m}$ 扩展到整个 exterior algebra 后的 $2^n\times 2^n$ 矩阵。要求 $\mathbf{M}$ 对 wedge product 是 homomorphism：

$$\mathbf{M}(\mathbf{a}\wedge\mathbf{b})=(\mathbf{M}\mathbf{a})\wedge(\mathbf{M}\mathbf{b})\tag{2.48}$$

这个单一约束（对基元素成立即可，由线性扩展到所有 multivector）就完全决定了 $\mathbf{M}$。

**构造方式：**
- grade-0：$\mathbf{M}\mathbf{1}=\mathbf{1}$，所以 $C_0(\mathbf{m})=[1]$（$1\times 1$ 单位）；
- grade-1：$C_1(\mathbf{m})=\mathbf{m}$ 本身（$n\times n$）；
- grade-2：对每个基 bivector $\mathbf{e}_{ij}$，要求 $\mathbf{M}\mathbf{e}_{ij}=(\mathbf{M}\mathbf{e}_i)\wedge(\mathbf{M}\mathbf{e}_j)=\mathbf{m}_{[i]}\wedge\mathbf{m}_{[j]}$，得到 $\binom{n}{2}\times\binom{n}{2}$ 子矩阵 $C_2(\mathbf{m})$；
- grade-$k$：类似取 $k$ 列的 wedge product，得到 $C_k(\mathbf{m})$；
- grade-$n$：$\mathbf{M}\mathbb{1}=\mathbf{m}_{[1]}\wedge\cdots\wedge\mathbf{m}_{[n]}=(\det\mathbf{m})\mathbb{1}$，所以 $C_n(\mathbf{m})=[\det\mathbf{m}]$。

子矩阵 $C_k(\mathbf{m})$ 在线性代数里就叫 **$k$-th compound matrix**（也叫 $k$-th exterior power 的矩阵表示、或 $k$-th adjugate / minors matrix）。它们沿对角线排成块对角矩阵 $\mathbf{M}$，非对角块全 0。

**重要观察：**
- $C_{n-1}(\mathbf{m})$ 总是 $\mathrm{adj}(\mathbf{m})^{\mathrm{T}}$，因为它每一列由"除一列外其余 $n-1$ 列的 wedge product"组成，这正是 antivector 的变换。这就是 paper 的核心定理在 exterior algebra 框架下的自然结果。
- $C_n(\mathbf{m})=\det\mathbf{m}$ 也不是巧合：det 就是 $n$ 个列的 wedge product 的系数。

---

## 5. 3D 的完整例子（$2^3=8$ 维 multivector）

paper 式 (2.47) 定义 3D 的完整 multivector
$$\mathbf{u}=s\,\mathbf{1}+v_x\mathbf{e}_1+v_y\mathbf{e}_2+v_z\mathbf{e}_3+b_x\mathbf{e}_{23}+b_y\mathbf{e}_{31}+b_z\mathbf{e}_{12}+t\,\mathbb{1}$$

变量含义：
- $s$：scalar 系数（grade 0）
- $(v_x,v_y,v_z)$：vector 系数（grade 1）
- $(b_x,b_y,b_z)$：bivector 系数（grade 2），按 $\mathbf{e}_{23},\mathbf{e}_{31},\mathbf{e}_{12}$ 顺序
- $t$：trivector / pseudo-scalar 系数（grade 3）

记 $\mathbf{m}=[\mathbf{a}\ \mathbf{b}\ \mathbf{c}]$，即三列分别为 $\mathbf{a},\mathbf{b},\mathbf{c}$（每个是 3D 列向量，下标 $a_x,a_y,a_z$ 等）。这是 $\mathbf{e}_1,\mathbf{e}_2,\mathbf{e}_3$ 变换后的像。

$C_2(\mathbf{m})$ 的第一列对应基 $\mathbf{e}_{23}$，要算 $\mathbf{b}\wedge\mathbf{c}$（因为 $\mathbf{e}_2\wedge\mathbf{e}_3=\mathbf{e}_{23}$）：
$$\mathbf{b}\wedge\mathbf{c}=(b_y c_z-b_z c_y)\mathbf{e}_{23}+(b_z c_x-b_x c_z)\mathbf{e}_{31}+(b_x c_y-b_y c_x)\mathbf{e}_{12}$$

这三个系数就是 $C_2(\mathbf{m})$ 的第一列。第二列、第三列同理，分别对应 $\mathbf{c}\wedge\mathbf{a}$（基 $\mathbf{e}_{31}$）和 $\mathbf{a}\wedge\mathbf{b}$（基 $\mathbf{e}_{12}$）。

paper 式 (2.52) 写出来的 $C_2(\mathbf{m})$ 就是一个 $3\times 3$ 矩阵，每列是某两个列向量的 cross product 的分量。**它正是第 2 节方程 (1) 中那个 $\mathbf{x}$，也即 $\mathrm{adj}(\mathbf{m})^{\mathrm{T}}$**——cross product 那条路和 wedge product 那条路在 3D 完全会合。

8×8 exomorphism matrix 的对角块依次是：
$$\mathbf{M}=\mathrm{diag}\big(C_0(\mathbf{m}),\,C_1(\mathbf{m}),\,C_2(\mathbf{m}),\,C_3(\mathbf{m})\big)=\mathrm{diag}\big([1],\,\mathbf{m},\,\mathrm{adj}(\mathbf{m})^{\mathrm{T}},\,[\det\mathbf{m}]\big)$$

---

## 6. 4D 投影空间的例子（$2^4=16$ 维 multivector）

实际图形学用 4D 齐次坐标 $(x,y,z,w)$ 表 point，用 6 维 Plücker 坐标表 line，用 4 维 $(g_x,g_y,g_z,g_w)$ 表 plane。这三者刚好对应 4D exterior algebra 的 grade 1, 2, 3（其中 grade 2 的 $\binom{4}{2}=6$ 维有两个分量：spatial bivector 和 moment bivector，物理上对应 line direction 和 line moment）。

16×16 exomorphism matrix 有 5 个对角块 $C_0,C_1,C_2,C_3,C_4$，大小 $1,4,6,4,1$，加起来 16。其中：
- $C_0=[1]$
- $C_1=\mathbf{m}$（$4\times 4$ 齐次变换矩阵，paper 默认第四行 $[0,0,0,1]$）
- $C_2$：$6\times 6$，变换 Plücker line
- $C_3$：$4\times 4$，变换 plane
- $C_4=[\det\mathbf{m}]$（齐次 affine 下恒为 1，因为 det 是 $1\cdot 1\cdot 1\cdot 1=1$，但若 w 分量有缩放就会变）

### 6.1 Translation 例子（式 2.53–2.55）

平移矩阵
$$\mathbf{m}=\begin{bmatrix}1&0&0&t_x\\0&1&0&t_y\\0&0&1&t_z\\0&0&0&1\end{bmatrix}$$

它的 $C_2$（line 的变换，6×6）：
$$C_2(\mathbf{m})=\begin{bmatrix}
1&0&0&0&0&0\\
0&1&0&0&0&0\\
0&0&1&0&0&0\\
0&-t_z&t_y&1&0&0\\
t_z&0&-t_x&0&1&0\\
-t_y&t_x&0&0&0&1
\end{bmatrix}$$

这里 line 写成 $(l_{vx},l_{vy},l_{vz},l_{mx},l_{my},l_{mz})$，前 3 是 direction $\mathbf{v}$，后 3 是 moment $\mathbf{m}=\mathbf{p}\times\mathbf{v}$。变换后：
- direction 不变（translation 不改方向，合理）；
- moment 变成 $\mathbf{m}'=\mathbf{m}+\mathbf{t}\times\mathbf{v}$，即 paper 说的"t 与 direction 的 cross 加到 moment"。这与经典 Plücker 平移公式 (1.42) 完全一致，证明 exomorphism 框架在数值上就是 Plücker 几何。

它的 $C_3$（plane 的变换，4×4）：
$$C_3(\mathbf{m})=\begin{bmatrix}
1&0&0&0\\
0&1&0&0\\
0&0&1&0\\
-t_x&-t_y&-t_z&1
\end{bmatrix}$$

plane 写成 $(g_x,g_y,g_z,g_w)$（normal $+$ offset），变换后 $g_w\to g_w-\mathbf{t}\cdot\mathbf{g}_{xyz}$，与公式 (1.38) 一致：translation 把平面"往原点反方向推"。

### 6.2 Nonuniform scale 例子（式 2.56–2.57）

$$\mathbf{m}=\mathrm{diag}(s_x,s_y,s_z,1)$$

$C_2(\mathbf{m})=\mathrm{diag}(s_x,s_y,s_z,s_y s_z,s_z s_x,s_x s_y)$

这里前 3 对应 line direction（按 $s_x,s_y,s_z$ 缩放，因为 direction 是长度量），后 3 对应 line moment（按两两乘积缩放，因为 moment 是面积量）。**直觉：grade-2 量在每一对被缩放的方向上都按面积缩放。**

$C_3(\mathbf{m})=\mathrm{diag}(s_y s_z,s_z s_x,s_x s_y,s_x s_y s_z)$

前 3 对应 plane normal（按面积缩放，因为 normal 本质是 oriented area 的 Hodge dual），第 4 个 $g_w$ 按 $s_x s_y s_z$ 缩放，因为它是体积量。

**paper 在这里的几何 punchline**：物理量按"长度的多少次方"缩放，就对应 exterior algebra 的 grade。grade-$k$ 量在 nonuniform scale 下按 $k$ 个方向的乘积缩放，这是 grade 作为"维度"的最直接体感。

---

## 7. 工程实践要点

把上面的内容收口到实际代码：

1. **CPU/GPU 端的 normal matrix 选择**：传统写法 `mat3 normalMatrix = transpose(inverse(mat3(modelMatrix)))` 在 reflection、singular 矩阵下会出问题。Lengyel 主张用 adjugate transpose。
   - GLM/HLSL 没有现成的 `adjugate`，但可以写一个 `adjugate3x3` 函数（9 个 2×2 余子式 + checkerboard sign），比 `inverse` 更稳健，且对 singular 矩阵返回 0 而不是 NaN。
   - 对 4×4 affine 矩阵 $\begin{bmatrix}\mathbf{R}&\mathbf{t}\\0&1\end{bmatrix}$，左上 3×3 是 $\mathbf{R}$，adjudate transpose 的左上 3×3 只依赖 $\mathbf{R}$（因为 $C_3$ 的右下角 = det $\mathbf{R}$，cross 项都涉及 $\mathbf{t}$ 但落在 $C_2$ 的 off-block，对 normal 无影响）。所以代码上仍然取左上 3×3 的 adjugate transpose 即可。

2. **Reflection 检测**：检查 $\det\mathbf{m}<0$ 后 flip normal，这是 adjugate transpose 给你"暴露的"信号，inverse transpose 因为除掉了 det 而把这个信号抹掉，需要单独再算 det 检测。

3. **Plücker line 和 plane 的变换**：直接复用 paper 给的 $C_2$（6×6）和 $C_3$（4×4）。如果只是 affine（第四行 $[0,0,0,1]$），有 closed form：
   - line: $\mathbf{v}'=\mathbf{R}\mathbf{v}$，$\mathbf{m}'=\mathbf{R}\mathbf{m}+\mathbf{t}\times(\mathbf{R}\mathbf{v})$
   - plane: $\mathbf{n}'=\mathrm{adj}(\mathbf{R})^{\mathrm{T}}\mathbf{n}$，$d'=d-\mathbf{t}\cdot\mathbf{n}'$（注意 $\mathbf{n}'$ 已经按 adjugate 变换，不是 $\mathbf{R}\mathbf{n}$）

4. **更高阶量**：如果你做物理仿真（vorticity 是 bivector）、流体（涡度）、电磁学（$\mathbf{E}$ 是 vector，$\mathbf{B}$ 是 bivector），同样需要 $C_2$，不能用 $\mathbf{m}$ 直接乘。Lengyel 的框架把所有"特殊变换"统一成一个 compound matrix 的查询。

---

## 8. 我自己的几点直觉补充

- **Grade = 量纲**：grade-$k$ 对象在物理上对应"$k$ 阶面积/体积"。nonuniform scale 例子最清楚：direction 是 grade-1（长度），moment 是 grade-2（面积），所以它们缩放规律不同。这件事在传统线性代数里被"藏起来"，因为线性代数只关心 grade-0 和 grade-1。
- **Cross product 是"作弊"**：3D 之所以能用一个普通向量表示 normal，是 Hodge dual 的巧合。一旦到了 2D 或 4D，cross product 不存在，但 wedge product 仍工作，normal 的本质（grade $n-1$）就暴露了。
- **Adjugate = "n−1 阶 exterior power 的矩阵表示"**：这正是 $C_{n-1}(\mathbf{m})$。在 3D 是 cross product 列，在 4D 是 4×4 plane 变换，在 $n$D 是 $(n-1)\times(n-1)$ minors。
- **为什么 inverse transpose 在 singular 时"挂"**：因为 inverse 要除以 det，det=0 时除法爆炸；adjugate 是 det 乘以 inverse，把除法消掉了，多项式形式永远可计算。这是"先乘 det 再除 det"在中间步骤避免除法的典型技巧，类似 Cayley-Hamilton、Schur complement 等。
- **Compound matrix 是"嵌套的 adjugate"**：$C_k(\mathbf{m})$ 的每个元素是 $\mathbf{m}$ 的某个 $k\times k$ 子矩阵的 det。当 $k=n-1$ 时就是 adjugate；$k=1$ 时就是 $\mathbf{m}$；$k=0$ 时是 1；$k=n$ 时是 det。所以 adjugate transpose 是"一阶 compound matrix 的转置"中的特例 $k=n-1$。

---

## 9. 参考资源

- Eric Lengyel 个人主页与 PGA 中心：https://projectivegeometricalgebra.org
- 原文 blog（这篇 paper 的来源）：https://terathon.com/blog/transforming-normals/ （paper 标题 "Transforming Normals", 2024-12-12）
- 配套书 *Projective Geometric Algebra Illuminated*：https://projectivegeometricalgebra.org/book.html
- 早期 *Graphics Gems* (1990) Ron Goldman "Matrix Inversion" 章节，ADJ 的最早图形学引用：https://www.sciencedirect.com/science/article/pii/B9780080507539500188
- Adjugate matrix（classical adjoint）Wikipedia：https://en.wikipedia.org/wiki/Adjugate_matrix
- Exterior algebra / Grassmann algebra 维基：https://en.wikipedia.org/wiki/Exterior_algebra
- Compound matrix Wikipedia：https://en.wikipedia.org/wiki/Compound_matrix
- Plücker coordinates 维基（line 的 grade-2 表示）：https://en.wikipedia.org/wiki/Plücker_coordinates
- Hodge dual 维基（cross product = wedge + Hodge）：https://en.wikipedia.org/wiki/Hodge_star_operator
- Multivector / Geometric algebra 综述：https://en.wikipedia.org/wiki/Geometric_algebra
- "Why do we transform normals by the inverse transpose?" 经典讨论（StackExchange）：https://computergraphics.stackexchange.com/questions/1502/why-does-my-normal-matrix-transform-not-work 和 https://www.reedbeta.com/blog/mathematical-roots-of-the-normal-matrix/
- Alex Heitt 整理的 normal transform 笔记：http://www.glprogramming.com/red/appendixf.html (OpenGL Red Book Appendix F)

---

## TL;DR 给你的直觉

1. Normal 不是 vector，是 grade $n-1$ 的 antivector（本质是一块 oriented area，3D 中靠 Hodge dual 假装成 vector）。
2. Cross product 是 wedge + Hodge 的 3D 特化；general 情况用 wedge。
3. Antivector 在线性变换 $\mathbf{m}$ 下的正确变换矩阵是 $\mathrm{adj}(\mathbf{m})^{\mathrm{T}}$，不是 $\mathbf{m}^{-\mathrm{T}}$。后者是前者乘 $\det\mathbf{m}$ 后再除掉的版本，对 singular / reflection 不稳。
4. 把 $\mathbf{m}$ 扩展到整个 exterior algebra，得到 $2^n\times 2^n$ 块对角 exomorphism matrix $\mathbf{M}$，对角块是 $k$-th compound matrix $C_k(\mathbf{m})$。$C_{n-1}=\mathrm{adj}(\mathbf{m})^{\mathrm{T}}$ 是其中一个特例。
5. 4D 齐次空间下，$C_2$ 就是 Plücker line 的 6×6 变换，$C_3$ 是 plane 的 4×4 变换。Translation 和 nonuniform scale 的特例与经典公式完全自洽，证明这套框架就是图形学几何的"母公式"。
6. 工程：在 shader 里把 `transpose(inverse(M))` 换成 `transpose(adjugate(M_upper_3x3))`，对 singular/reflection 更鲁棒，且 expose 出 det 符号让你判断 reflection。
