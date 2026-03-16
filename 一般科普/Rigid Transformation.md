# Rigid Transformation（刚体变换）详解

## 一、基本概念与直观理解

### 1.1 什么是 Rigid Transformation?

根据你上传的文件，**Rigid transformation**（也称为 **Euclidean transformation** 或 **Euclidean isometry**）是一种保持欧氏空间中任意两点之间 **Euclidean distance**（欧氏距离）不变的几何变换。

**直观理解**：想象一个刚性物体（如一块石头），你可以在空间中移动它、旋转它，但物体本身的形状和大小不会改变——这就是刚体变换的本质。

### 1.2 Rigid Transformation 的分类

| 类型 | 英文名称 | 特征 | det(R) |
|------|---------|------|--------|
| **Proper Rigid Transformation** | Rigid Motion / Euclidean Motion | 保持手性 | +1 |
| **Improper Rigid Transformation** | 含有反射 | 改变手性 | -1 |

**手性**：左手 vs 右手，反射会将左手变成右手。

---

## 二、数学形式化定义

### 2.1 核心公式

刚体变换 $T$ 作用在任意向量 $\mathbf{v}$ 上，产生变换后的向量 $T(\mathbf{v})$：

$$\boxed{T(\mathbf{v}) = \mathbf{R} \mathbf{v} + \mathbf{t}}$$

**变量解释**：
- $\mathbf{v} \in \mathbb{R}^n$：输入向量（原始点的坐标）
- $\mathbf{R} \in \mathbb{R}^{n \times n}$：**Rotation matrix**（旋转矩阵）或 **Orthogonal matrix**（正交矩阵）
- $\mathbf{t} \in \mathbb{R}^n$：**Translation vector**（平移向量）
- $T(\mathbf{v})$：变换后的向量

### 2.2 正交矩阵的条件

矩阵 $\mathbf{R}$ 必须满足：

$$\mathbf{R}^T \mathbf{R} = \mathbf{I}$$

或等价地：

$$\mathbf{R}^T = \mathbf{R}^{-1}$$

**推导过程**：
根据文件中的距离公式，对于线性变换 $[\mathbf{L}]$：

$$d([\mathbf{L}]\mathbf{v}, [\mathbf{L}]\mathbf{w})^2 = ([\mathbf{L}]\mathbf{v} - [\mathbf{L}]\mathbf{w}) \cdot ([\mathbf{L}]\mathbf{v} - [\mathbf{L}]\mathbf{w})$$

利用点积的矩阵表示：$\mathbf{a} \cdot \mathbf{b} = \mathbf{a}^T \mathbf{b}$

$$d([\mathbf{L}]\mathbf{v}, [\mathbf{L}]\mathbf{w})^2 = (\mathbf{v} - \mathbf{w})^T [\mathbf{L}]^T [\mathbf{L}] (\mathbf{v} - \mathbf{w})$$

要保持距离不变，即 $d([\mathbf{L}]\mathbf{v}, [\mathbf{L}]\mathbf{w})^2 = d(\mathbf{v}, \mathbf{w})^2 = (\mathbf{v} - \mathbf{w})^T (\mathbf{v} - \mathbf{w})$，必须有：

$$[\mathbf{L}]^T [\mathbf{L}] = \mathbf{I}$$

### 2.3 Determinant 条件

计算行列式：

$$\det([\mathbf{L}]^T [\mathbf{L}]) = \det[\mathbf{L}]^2 = \det[\mathbf{I}] = 1$$

因此：

$$\det[\mathbf{L}] = \pm 1$$

- **$\det(\mathbf{R}) = +1$**：**Rotation**（旋转），保持手性，称为 **Proper Rigid Transformation**
- **$\det(\mathbf{R}) = -1$**：**Reflection**（反射），改变手性，称为 **Improper Rigid Transformation**

---

## 三、Euclidean Distance（欧氏距离）公式

### 3.1 距离平方公式

对于 $\mathbb{R}^n$ 中的两点 $\mathbf{X} = (X_1, X_2, \ldots, X_n)$ 和 $\mathbf{Y} = (Y_1, Y_2, \ldots, Y_n)$：

$$d(\mathbf{X}, \mathbf{Y})^2 = (X_1 - Y_1)^2 + (X_2 - Y_2)^2 + \cdots + (X_n - Y_n)^2$$

或使用向量形式：

$$d(\mathbf{X}, \mathbf{Y})^2 = (\mathbf{X} - \mathbf{Y}) \cdot (\mathbf{X} - \mathbf{Y}) = \|\mathbf{X} - \mathbf{Y}\|^2$$

这是 **Pythagorean theorem**（毕达哥拉斯定理）的推广。

### 3.2 刚体变换的距离保持性

刚体变换 $g: \mathbb{R}^n \to \mathbb{R}^n$ 满足：

$$d(g(\mathbf{X}), g(\mathbf{Y}))^2 = d(\mathbf{X}, \mathbf{Y})^2$$

**证明**（对于 Translation）：
$$d(\mathbf{v} + \mathbf{d}, \mathbf{w} + \mathbf{d})^2 = (\mathbf{v} + \mathbf{d} - \mathbf{w} - \mathbf{d}) \cdot (\mathbf{v} + \mathbf{d} - \mathbf{w} - \mathbf{d}) = (\mathbf{v} - \mathbf{w}) \cdot (\mathbf{v} - \mathbf{w}) = d(\mathbf{v}, \mathbf{w})^2$$

---

## 四、变换分解与特殊定理

### 4.1 2D 情况

在二维空间中，**Proper Rigid Motion** 只有两类：
1. **Translation**（平移）
2. **Rotation**（旋转）

### 4.2 3D 情况

在三维空间中：

**分解定理**：每个刚体运动都可以分解为一个 **Rotation** 和一个 **Translation** 的组合，因此有时称为 **Rototranslation**。

### 4.3 Chasles' Theorem（沙勒定理）

这是刚体变换中最重要的定理之一：

> **Chasles' Theorem**：在三维空间中，任何刚体运动都可以表示为 **Screw Motion**（螺旋运动）。

**螺旋运动**：绕某条轴的旋转 + 沿同一轴的平移。

**数学表示**：
$$T(\mathbf{v}) = \mathbf{R}(\theta, \hat{\mathbf{n}}) \mathbf{v} + d \cdot \hat{\mathbf{n}}$$

其中：
- $\hat{\mathbf{n}}$：螺旋轴的单位方向向量
- $\theta$：绕轴的旋转角度
- $d$：沿轴的平移距离

**几何直观**：想象一个螺丝钉拧入木头——它同时在旋转和前进。

---

## 五、群论视角

### 5.1 Euclidean Group（欧氏群）

所有刚体变换（proper 和 improper）构成的数学群称为 **Euclidean Group**，记为 $E(n)$。

**群公理验证**：
1. **封闭性**：两个刚体变换的复合仍是刚体变换
2. **结合律**：$(T_1 \circ T_2) \circ T_3 = T_1 \circ (T_2 \circ T_3)$
3. **单位元**：Identity transformation $T(\mathbf{v}) = \mathbf{v}$
4. **逆元**：$T^{-1}(\mathbf{v}) = \mathbf{R}^T(\mathbf{v} - \mathbf{t})$

### 5.2 Special Euclidean Group（特殊欧氏群）

所有 Proper Rigid Transformation（刚体运动）构成的群称为 **Special Euclidean Group**，记为 $SE(n)$。

$$SE(n) = \left\{ \begin{pmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0}^T & 1 \end{pmatrix} : \mathbf{R} \in SO(n), \mathbf{t} \in \mathbb{R}^n \right\}$$

### 5.3 Orthogonal Group 与 Special Orthogonal Group

| 群 | 符号 | 定义 | 维度 |
|---|------|------|------|
| **Orthogonal Group** | $O(n)$ | $\{\mathbf{R} : \mathbf{R}^T \mathbf{R} = \mathbf{I}\}$ | 包含旋转和反射 |
| **Special Orthogonal Group** | $SO(n)$ | $\{\mathbf{R} : \mathbf{R}^T \mathbf{R} = \mathbf{I}, \det(\mathbf{R}) = 1\}$ | 仅包含旋转 |

**Lie Group 结构**：$SO(n)$ 是一个 **Lie Group**（李群），因为它具有流形结构。

---

## 六、齐次坐标表示

### 6.1 为什么需要齐次坐标？

刚体变换 $T(\mathbf{v}) = \mathbf{R}\mathbf{v} + \mathbf{t}$ 包含线性部分 $\mathbf{R}\mathbf{v}$ 和仿射部分 $\mathbf{t}$。使用齐次坐标可以将刚体变换表示为**单一的矩阵乘法**。

### 6.2 齐次坐标表示

在齐次坐标中，$n$ 维向量 $\mathbf{v} \in \mathbb{R}^n$ 扩展为 $(n+1)$ 维向量：

$$\tilde{\mathbf{v}} = \begin{pmatrix} \mathbf{v} \\ 1 \end{pmatrix} \in \mathbb{R}^{n+1}$$

刚体变换的矩阵形式：

$$\begin{pmatrix} T(\mathbf{v}) \\ 1 \end{pmatrix} = \begin{pmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0}^T & 1 \end{pmatrix} \begin{pmatrix} \mathbf{v} \\ 1 \end{pmatrix} = \begin{pmatrix} \mathbf{R}\mathbf{v} + \mathbf{t} \\ 1 \end{pmatrix}$$

**完整矩阵形式**（以 3D 为例）：

$$\mathbf{T} = \begin{pmatrix} r_{11} & r_{12} & r_{13} & t_x \\ r_{21} & r_{22} & r_{23} & t_y \\ r_{31} & r_{32} & r_{33} & t_z \\ 0 & 0 & 0 & 1 \end{pmatrix} \in SE(3)$$

### 6.3 变换的复合

两个刚体变换的复合：

$$\mathbf{T}_1 \cdot \mathbf{T}_2 = \begin{pmatrix} \mathbf{R}_1 & \mathbf{t}_1 \\ \mathbf{0}^T & 1 \end{pmatrix} \begin{pmatrix} \mathbf{R}_2 & \mathbf{t}_2 \\ \mathbf{0}^T & 1 \end{pmatrix} = \begin{pmatrix} \mathbf{R}_1 \mathbf{R}_2 & \mathbf{R}_1 \mathbf{t}_2 + \mathbf{t}_1 \\ \mathbf{0}^T & 1 \end{pmatrix}$$

### 6.4 逆变换

$$\mathbf{T}^{-1} = \begin{pmatrix} \mathbf{R}^T & -\mathbf{R}^T \mathbf{t} \\ \mathbf{0}^T & 1 \end{pmatrix}$$

---

## 七、旋转矩阵的具体形式

### 7.1 2D Rotation Matrix

绕原点旋转角度 $\theta$：

$$\mathbf{R}(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

**验证正交性**：
$$\mathbf{R}^T \mathbf{R} = \begin{pmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$

**Determinant**：
$$\det(\mathbf{R}) = \cos^2\theta + \sin^2\theta = 1$$

### 7.2 3D Rotation Matrices

绕坐标轴的旋转：

**绕 X 轴旋转**（角度 $\alpha$）：
$$\mathbf{R}_x(\alpha) = \begin{pmatrix} 1 & 0 & 0 \\ 0 & \cos\alpha & -\sin\alpha \\ 0 & \sin\alpha & \cos\alpha \end{pmatrix}$$

**绕 Y 轴旋转**（角度 $\beta$）：
$$\mathbf{R}_y(\beta) = \begin{pmatrix} \cos\beta & 0 & \sin\beta \\ 0 & 1 & 0 \\ -\sin\beta & 0 & \cos\beta \end{pmatrix}$$

**绕 Z 轴旋转**（角度 $\gamma$）：
$$\mathbf{R}_z(\gamma) = \begin{pmatrix} \cos\gamma & -\sin\gamma & 0 \\ \sin\gamma & \cos\gamma & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

### 7.3 Euler Angles（欧拉角）

任意 3D 旋转可以表示为三次绕坐标轴旋转的组合。常见的有：

**Z-Y-X Euler Angles**：
$$\mathbf{R} = \mathbf{R}_z(\psi) \mathbf{R}_y(\theta) \mathbf{R}_x(\phi)$$

其中：
- $\phi$：**Roll**（滚转角）
- $\theta$：**Pitch**（俯仰角）
- $\psi$：**Yaw**（偏航角）

**注意**：Euler Angles 存在 **Gimbal Lock**（万向节锁）问题。

### 7.4 Rotation Vector（旋转向量/轴角表示）

任意旋转可以用旋转轴 $\hat{\mathbf{n}}$（单位向量）和旋转角度 $\theta$ 表示：

$$\mathbf{R}(\theta, \hat{\mathbf{n}}) = \cos\theta \cdot \mathbf{I} + (1 - \cos\theta) \hat{\mathbf{n}} \hat{\mathbf{n}}^T + \sin\theta \cdot [\hat{\mathbf{n}}]_\times$$

其中 $[\hat{\mathbf{n}}]_\times$ 是 **Skew-symmetric Matrix**（反对称矩阵）：

$$[\hat{\mathbf{n}}]_\times = \begin{pmatrix} 0 & -n_z & n_y \\ n_z & 0 & -n_x \\ -n_y & n_x & 0 \end{pmatrix}$$

**Rodrigues' Formula**：上述公式也称为 Rodrigues 公式。

---

## 八、应用领域

### 8.1 Computer Vision（计算机视觉）

**应用场景**：
- **Camera Pose Estimation**（相机位姿估计）
- **3D Reconstruction**（三维重建）
- **Object Tracking**（目标跟踪）
- **Image Registration**（图像配准）

**经典问题**：给定两组对应点 $\{(\mathbf{p}_i, \mathbf{q}_i)\}_{i=1}^N$，求解最优刚体变换 $(\mathbf{R}, \mathbf{t})$ 使得：

$$\min_{\mathbf{R}, \mathbf{t}} \sum_{i=1}^N \|\mathbf{R}\mathbf{p}_i + \mathbf{t} - \mathbf{q}_i\|^2$$

**解决方案**：**Kabsch Algorithm** / **ICP (Iterative Closest Point)**

### 8.2 Robotics（机器人学）

**应用场景**：
- **Forward Kinematics**（正运动学）
- **Inverse Kinematics**（逆运动学）
- **SLAM (Simultaneous Localization and Mapping)**
- **Motion Planning**（运动规划）

### 8.3 Medical Imaging（医学影像）

**应用场景**：
- **Image Registration**（图像配准）
- **Surgical Navigation**（手术导航）

### 8.4 Computer Graphics（计算机图形学）

**应用场景**：
- **3D Modeling**（三维建模）
- **Animation**（动画）
- **Virtual Reality / Augmented Reality**

---

## 九、参数化方法比较

| 方法 | 参数数量 | 优点 | 缺点 |
|------|---------|------|------|
| **Rotation Matrix** | 9 (3×3) | 直观，无奇异点 | 冗余，需约束 |
| **Euler Angles** | 3 | 直观 | Gimbal Lock |
| **Axis-Angle** | 4 (3+1) | 物理意义明确 | 轴归一化约束 |
| **Unit Quaternion** | 4 | 无奇异点，插值平滑 | 需归一化 |
| **Twist Coordinates** | 6 | 线性化，适合优化 | 仅局部有效 |

### 9.1 Unit Quaternion（单位四元数）

单位四元数 $\mathbf{q} = (q_w, q_x, q_y, q_z)$ 满足 $\|\mathbf{q}\| = 1$：

$$\mathbf{q} = \left(\cos\frac{\theta}{2}, \hat{\mathbf{n}}\sin\frac{\theta}{2}\right)$$

对应的旋转矩阵：

$$\mathbf{R} = \begin{pmatrix} 1-2(q_y^2+q_z^2) & 2(q_x q_y - q_w q_z) & 2(q_x q_z + q_w q_y) \\ 2(q_x q_y + q_w q_z) & 1-2(q_x^2+q_z^2) & 2(q_y q_z - q_w q_x) \\ 2(q_x q_z - q_w q_y) & 2(q_y q_z + q_w q_x) & 1-2(q_x^2+q_y^2) \end{pmatrix}$$

**四元数乘法**对应旋转的复合：

$$\mathbf{q}_1 \otimes \mathbf{q}_2 = \mathbf{R}(\mathbf{q}_1) \cdot \mathbf{R}(\mathbf{q}_2)$$

---

## 十、Lie Algebra 视角

### 10.1 Lie Algebra se(3)

$SE(3)$ 对应的 Lie Algebra 是 $\mathfrak{se}(3)$，其元素为：

$$\boldsymbol{\xi} = \begin{pmatrix} \boldsymbol{\omega} \\ \mathbf{v} \end{pmatrix} \in \mathbb{R}^6$$

其中 $\boldsymbol{\omega} = \theta \hat{\mathbf{n}}$ 是角速度向量，$\mathbf{v}$ 是线速度向量。

### 10.2 Exponential Map

从 Lie Algebra 到 Lie Group 的映射：

$$\mathbf{T} = \exp(\boldsymbol{\xi}^\wedge)$$

其中 $\boldsymbol{\xi}^\wedge$ 是 **Twist** 的矩阵表示：

$$\boldsymbol{\xi}^\wedge = \begin{pmatrix} [\boldsymbol{\omega}]_\times & \mathbf{v} \\ \mathbf{0}^T & 0 \end{pmatrix} \in \mathbb{R}^{4 \times 4}$$

**Exponential Map 公式**：

$$\exp(\boldsymbol{\xi}^\wedge) = \begin{pmatrix} \exp([\boldsymbol{\omega}]_\times) & \mathbf{V}\mathbf{v} \\ \mathbf{0}^T & 1 \end{pmatrix}$$

其中：
$$\mathbf{V} = \mathbf{I} + \frac{1 - \cos\theta}{\theta^2}[\boldsymbol{\omega}]_\times + \frac{\theta - \sin\theta}{\theta^3}[\boldsymbol{\omega}]_\times^2$$

### 10.3 应用：优化问题

在优化中，使用 **Local Parametrization**：

$$\mathbf{T}(\boldsymbol{\xi}) = \mathbf{T}_0 \cdot \exp(\boldsymbol{\xi}^\wedge)$$

这样可以用 $\boldsymbol{\xi} \in \mathbb{R}^6$ 作为参数，在 $\mathbf{T}_0$ 附近进行局部线性化。

---

## 十一、实验数据示例

### 11.1 正交性验证

对于旋转矩阵 $\mathbf{R}$：

| 性质 | 公式 | 验证值 |
|------|------|--------|
| 正交性 | $\mathbf{R}^T \mathbf{R} = \mathbf{I}$ | 单位矩阵 |
| 行列式 | $\det(\mathbf{R}) = 1$ | +1 |
| 列向量范数 | $\|\mathbf{r}_i\| = 1$ | 1 |
| 列向量正交 | $\mathbf{r}_i \cdot \mathbf{r}_j = 0, i \neq j$ | 0 |

### 11.2 自由度

| 维度 | Rotation DOF | Translation DOF | Total DOF |
|------|--------------|-----------------|-----------|
| 2D | 1 ($\theta$) | 2 ($t_x, t_y$) | 3 |
| 3D | 3 ($\phi, \theta, \psi$) | 3 ($t_x, t_y, t_z$) | 6 |

---

## 十二、总结

**Rigid Transformation** 的核心要点：

1. **定义**：保持 Euclidean distance 不变的变换
2. **形式**：$T(\mathbf{v}) = \mathbf{R}\mathbf{v} + \mathbf{t}$
3. **约束**：$\mathbf{R}^T \mathbf{R} = \mathbf{I}$，$\det(\mathbf{R}) = \pm 1$
4. **群结构**：构成 $SE(n)$ 群，是 Lie Group
5. **分解**：3D 中可分解为 Screw Motion
6. **表示**：齐次坐标、四元数、轴角等多种参数化方式

---

## 参考资料

1. **Wikipedia - Rigid Transformation**: https://en.wikipedia.org/wiki/Rigid_transformation
2. **Wikipedia - Euclidean Group**: https://en.wikipedia.org/wiki/Euclidean_group
3. **Wikipedia - Chasles' Theorem**: https://en.wikipedia.org/wiki/Chasles%27_theorem_(kinematics)
4. **Wikipedia - Rotation Matrix**: https://en.wikipedia.org/wiki/Rotation_matrix
5. **Wikipedia - Quaternions and Spatial Rotation**: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
6. **Multiple View Geometry in Computer Vision** (Hartley & Zisserman): http://szeliski.org/Book/
7. **Lie Groups for Computer Vision**: https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf
8. **3D Math Primer for Graphics and Game Development**: https://gamemath.com/
9. **SE(3) Lie Group Tutorial**: https://ethaneade.com/lie_groups.pdf
10. **Kabsch Algorithm**: https://en.wikipedia.org/wiki/Kabsch_algorithm