---
source_pdf: Rigid Body Motion and the Euclidean Group.pdf
paper_sha256: 53c64d93782a4307e5e32a25c6d558249bc829c8fe08a5ccad35c6045ec11ef8
processed_at: '2026-08-11T23:52:00-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

这篇 paper 讲的是一件特别朴素的事：**怎么用数学语言描述刚体在空间里的运动**。但它的厉害之处在于，它告诉你运动和力其实是同一套数学结构，就像电场和磁场其实是同一个电磁张量的不同分量。

下面我尽量不用公式，用类比和画面来讲。

---

## 1. 先搞清楚研究对象：rigid body

Rigid body 就是"硬邦邦、不变形的物体"。你想象一块铁块在空间里飞，它的运动有两部分：

- **位置**变了（translation，平移）
- **朝向**变了（rotation，旋转）

这两件事是独立的，不能合并成一个。比如你把杯子从桌左移到桌右，位置变了但朝向没变；你把杯子原地转 90 度，朝向变了但位置没变。所以完整描述一个 rigid body 在 3D 空间的姿态需要 **6 个数字**：3 个管位置，3 个管朝向。

数学上把"所有可能的姿态"组成的集合记为 **SE(3)**，读作 Special Euclidean Group。这是个 Lie group——你可以暂时把它想成"一个 6 维的空间，里面的每个点代表刚体的一种姿态"。

类比：SE(3) 像一个 6 维的地图，地图上每个点都是刚体可能的一种摆法。

---

## 2. 任何"挪一下"都能当成拧螺丝

这是 paper 里最美的一个结论，叫 **Chasles 定理**。

你想把一个刚体从姿态 A 挪到姿态 B，可以怎么做？最朴素的方式是：先平移一段，再转一下。但 Chasles 在 1830 年发现：**总存在一根轴，让你绕着它转某个角度、同时沿它滑一段距离，就能完成这个位移**。

这就是螺丝钉的运动！你拧螺丝的时候，螺丝既在转圈又在往里钻，转和滑是绑在一起的。

所以 SE(3) 里任何一个"挪一下"（finite displacement）都能看成 **绕某个 screw axis 转一定角度 + 滑一定距离**。这个 screw 由三样东西确定：
1. 一条空间里的直线（axis）
2. 一个 pitch（每转一圈滑多远）
3. 转多少角度

Paper 里用 homogeneous transformation matrix 来描述这种位移，但本质就是这个几何图像。

---

## 3. 把"拧螺丝"推广到瞬时：twist

刚体运动不是一步到位的，是连续的。比如机械臂关节在转，每一刻都有一个瞬时速度。

Paper 的核心观察是：**任何瞬时刚体运动，也都对应一个"瞬时螺丝"**。这就是 Chasles 定理的无穷小版本。

具体说，刚体在某一瞬间的运动可以分解成：
- 绕某根轴转（angular velocity $\boldsymbol{\omega}$）
- 沿这根轴滑（linear velocity 的轴向分量）

二者合起来就是一个 **twist**，用 6 个数字表示：3 个角速度 + 3 个线速度。

这根轴叫做 **Instantaneous Screw Axis (ISA)**。形象点：刚体上每个点的速度都切于一个螺旋线，所有螺旋线共用同一根中心轴，所有点的"旋转分量"垂直于轴，"滑动分量"沿轴方向。

举个具体例子：
- 门绕铰链转：ISA 就是铰链轴，pitch = 0（没有滑动）
- 抽屉往外拉：ISA 在无穷远，pitch = ∞（纯平移，没有转动）
- 拧螺丝：ISA 就是螺丝的中轴，pitch = 螺距

paper 公式 (17) 给出了从 twist 反算 ISA 和 pitch 的具体公式，核心就两步：
1. 用 $\boldsymbol{\omega} \times \mathbf{v}$ 算出轴的位置
2. 用 $\boldsymbol{\omega} \cdot \mathbf{v}$ 算出 pitch

---

## 4. 力系统也是拧螺丝：wrench

到这儿只是运动学，paper 的妙处在于下一半：**力也是同一种几何对象**。

想象刚体被一堆力和力矩作用。这堆乱七八糟的力，能不能化简？答案是能，而且化简后的形式和 twist 一模一样：

**任何力系统都能化简为：一个力沿某根轴 + 一个绕这根轴的力矩**。

这就是 **wrench**，也用 6 个数字表示：3 个力分量 + 3 个力矩分量。

- pure force（纯力）：pitch = 0
- pure couple（纯力矩）：pitch = ∞
- 一般力 + 力矩：pitch 介于两者之间

为什么力和运动竟然是同一种几何对象？因为它们的代数结构完全一样：都是一个 6 维向量 + 一条轴 + 一个 pitch。

这个对偶性叫 **kinematic-static duality**，是 Ball 在 1900 年系统化的。

---

## 5. 把运动和力接起来：reciprocity

现在两样东西都有了：twist（运动）和 wrench（力）。它们怎么互动？

答案在 **功率**：力对运动做功 = 力 × 速度。

但 twist 和 wrench 都是 6 维向量，怎么乘？paper 公式 (31) 给出：
$$P = \mathbf{F} \cdot \mathbf{v} + \mathbf{M} \cdot \boldsymbol{\omega}$$

也就是：力乘线速度 + 力矩乘角速度。这就是 wrench 作用在 twist 上的瞬时功率。

**Reciprocity** 的定义就是：当这个功率等于零，我们说这个 wrench 和这个 twist 互相 reciprocal。

什么情况下力不做功？三种 paper 给出的典型情形：

### 情形 1：revolute joint（铰链）
你推门，如果推力的方向通过铰链轴，门不动——因为力矩为零。
更一般地：wrench 和 revolute joint 允许的 twist reciprocal，当 wrench 的轴穿过铰链轴。

### 情形 2：prismatic joint（滑轨）
抽屉轨道只能沿一个方向滑。你垂直于轨道方向推，抽屉不动——力被轨道约束抵消了。
更一般地：wrench 和 prismatic joint 允许的 twist reciprocal，当 wrench 轴垂直于滑轨。

### 情形 3：frictionless contact
无摩擦接触只能提供沿法线的力。刚体如果运动方向沿切面，接触力不做功。
更一般地：twist 和 contact normal 的 wrench reciprocal，当 twist 轴穿过接触法线。

这就是 reciprocity 的威力：**约束力总是和允许运动 reciprocal**。这是整个机构学的基础。

---

## 6. 为什么这套数学值得学

你可能会问：为什么不用简单的 $F=ma$ + 矢量力学，非得搞 Lie group？

因为 screw theory 让你同时处理三件事：

### (1) Kinematics（运动学）
机械臂每个关节都有一个 screw。把所有 screw 加权求和就是末端速度——这是 manipulator Jacobian 的几何意义。

### (2) Statics（静力学）
机械臂被外力作用时，每个关节需要多大扭矩来抵抗？只要算 wrench 在每个 joint screw 上的"投影"（reciprocal product）。

### (3) Dynamics（动力学）
Lagrangian 动力学里 kinetic energy 是 $\frac{1}{2}\boldsymbol{\omega}^T \mathbf{I} \boldsymbol{\omega}$。在 SE(3) 上写 inertia tensor，整个 rigid body dynamics 可以写成统一形式。

这三件事在 screw theory 里是**同一套公式**，只是把 twist 换成 wrench，把运动换成力。这就是它的优雅。

---

## 7. 几个直白的例子

### 例子 A：开门
门装在铰链上，铰链就是 ISA，pitch = 0。门上每一点的速度垂直于"该点到铰链轴的连线"，大小正比于距离。这就是公式 (16) 的几何含义。

### 例子 B：拧螺丝
你拿螺丝刀拧螺丝，螺丝刀施加的力矩让螺丝转，同时螺丝的螺纹把它往里拉。这是一个 pitch 等于螺距的 screw motion。如果螺丝是右旋的，pitch > 0；左旋的，pitch < 0。

### 例子 C：汽车方向盘
方向盘转动 → 前轮转向。这里的 ISA 是方向盘的转轴，pitch = 0。但方向盘通过 linkage 把这个 twist 转换成前轮的另一个 twist。整个转换就是 SE(3) 元素的复合。

### 例子 D：机器人手臂
6 自由度机械臂的每个关节对应一个 screw。末端执行器的速度 = 6 个 joint screw 加权和，权重是各关节速度。这就是 forward velocity kinematics。公式里那个 6×6 矩阵 Jacobian 的每一列就是一个 joint screw。

### 例子 E：抓杯子
手指捏杯子，接触面提供法向力（无摩擦假设）。杯子能不能被"形封闭"地固定？条件是所有接触法线的 wrenches 张成整个 6 维 wrench 空间。这就是 grasp analysis。

---

## 8. 历史彩蛋

Screw theory 是少数几个"被工程重新激活的古典数学"：

- **1806 Poinsot**：把力系统化简为 wrench
- **1830 Chasles**：证明任何刚体位移 = screw displacement
- **1876 Ball**：出版 *A Treatise on the Theory of Screws*，系统化
- **1900 Ball 第二版**：paper 引用的就是这版
- **20 世纪中期**：screw theory 沉寂，因为矢量力学够用
- **1990s**：robotics 兴起，发现 6 维 vector 刚好处理运动 + 力，screw theory 复活
- **1994 MLS94**：Murray, Li, Sastry 把它纳入现代 robotics 数学框架
- **2010s**：SLAM、IMU preintegration、pose graph optimization 都在 SE(3) 上做

所以你现在看 robotics 论文里的 $\mathrm{Ad}_g$、$\mathrm{ad}_\xi$、$\exp(\hat{\xi})$ 这些符号，全是 screw theory 的现代化身。

---

## 9. 一句话总结

这篇 paper 在说：

**刚体的运动可以用"绕某根轴转 + 沿这根轴滑"来描述（twist），刚体受到的力系统也可以用"沿某根轴的力 + 绕这根轴的力矩"来描述（wrench），两者通过"做功为零"这个条件连接起来（reciprocity），构成运动学和静力学的统一数学框架。**

记住三个词就够了：
- **Twist** = 运动的 screw
- **Wrench** = 力的 screw  
- **Reciprocal** = 力对运动不做功

剩下的所有公式都是为了把这三件事在不同坐标系之间、不同时间尺度下、不同力学情境下严格地算出来。

---

## 参考阅读

入门顺序建议：
1. **Lynch & Park, Modern Robotics** 第 3-6 章：https://hades.mech.northwestern.edu/index.php/Modern_Robotics
2. **MLS94 第 2-3 章**：https://www.cds.caltech.edu/~murray/mlswiki/
3. **Solà, Micro Lie Theory**：https://arxiv.org/abs/1812.01537
4. **Ball 原著（历史感）**：https://archive.org/details/treatiseontheory00ballrich

读完 Lynch & Park 的前几章，再回头看这篇 paper，会发现它讲的每一句话都对应一个具体的机器人学应用。Screw theory 真正"开窍"的瞬间，是你发现所有机械臂教材里的公式，都可以用"twist + wrench + reciprocal"这三把刀切出来。

---

# Rigid Body Motion and the Euclidean Group — 详解

这篇 paper 是 screw theory 的经典章节，讨论如何用 Lie group 的语言来描述刚体运动。核心思想在于：把刚体位移看作 Lie group SE(3) 的元素，把瞬时运动看作 Lie algebra se(3) 的元素，把力系统看作对偶空间 se*(3) 的元素，并用 reciprocity 把它们串起来。下面我按章节展开，并尽量讲清公式里每个符号的物理意义。

---

## 1. Euclidean Group SE(3) 的代数结构

### 1.1 Homogeneous transformation matrix

刚体 B 相对参考系 {A} 的位姿用 4×4 homogeneous transformation matrix 描述：

$$
{}^{A}\mathbf{A}_{B} = \begin{bmatrix} {}^{A}\mathbf{R}_{B} & {}^{A}\mathbf{r}^{O'} \\ \mathbf{0}_{1\times 3} & 1 \end{bmatrix}
$$

变量解释：
- 左上标 ${}^{A}$：表示量在 frame {A} 中表达
- ${}^{A}\mathbf{R}_{B} \in \mathbb{R}^{3\times 3}$：rotation matrix，将 {B} 中向量分量变换到 {A} 中分量
- ${}^{A}\mathbf{r}^{O'} \in \mathbb{R}^{3}$：{B} 原点 $O'$ 在 {A} 中的位置向量
- 最后一行 $[0,0,0,1]$ 是为了使矩阵乘法能同时处理 rotation 和 translation

### 1.2 SE(3) 的 group axioms

集合
$$
SE(3) = \{(\mathbf{R}, \mathbf{r}) \mid \mathbf{R}\in\mathbb{R}^{3\times 3}, \mathbf{r}\in\mathbb{R}^{3}, \mathbf{R}^{T}\mathbf{R}=\mathbf{R}\mathbf{R}^{T}=\mathbf{I}, \det\mathbf{R}=+1\}
$$

满足四个 group axioms：

| Axiom | 内容 |
|-------|------|
| Closure | 若 $A, B \in SE(3)$，则 $AB \in SE(3)$ |
| Associativity | $(AB)C = A(BC)$ |
| Identity | 存在 $4\times 4$ 单位阵 $\mathbf{I}$ |
| Inverse | 每个 $A$ 都有 $A^{-1} \in SE(3)$，使 $AA^{-1}=\mathbf{I}$ |

由于乘法和求逆都是连续操作，且 SE(3) 局部同胚于 $\mathbb{R}^6$，所以 SE(3) 是一个 6 维 differentiable manifold。一个 group 同时是 differentiable manifold，就称为 **Lie group**（以 Sophus Lie 命名）。

**Intuition**: SE(3) 既是 group 又是 manifold，意味着我们可以在它上面做"微积分"。Lie algebra se(3) 就是 manifold 在 identity 处的 tangent space，它给出瞬时速度（twist）的线性化描述。

### 1.3 重要 subgroups

paper Table 1 列出 SE(3) 的子群与机构学中运动副的对应：

| Subgroup | Notation | 自由度 | 对应运动副 |
|----------|----------|--------|------------|
| 球面旋转群 | SO(3) | 3 | S-pair (spherical joint) |
| 平面欧氏群 | SE(2) | 3 | E-pair (planar joint) |
| 平面旋转群 | SO(2) | 1 | R-pair (revolute joint) |
| n 维平移群 | T(n) | n | — |
| 一维平移群 | T(1) | 1 | P-pair (prismatic joint) |
| 圆柱群 | SO(2)×T(1) | 2 | C-pair (cylindrical joint) |
| 螺旋群 | H(1) | 1 | H-pair (helical joint) |

这个对应非常关键：**机构中的每类运动副就是 SE(3) 的一个 subgroup**。这是 Klein Erlanger program 在运动学中的体现——用对称群分类几何对象。

参考链接：
- Lie group 基础: https://en.wikipedia.org/wiki/Lie_group
- SE(3) 详解: https://en.wikipedia.org/wiki/Euclidean_group
- Screw theory 历史: https://en.wikipedia.org/wiki/Screw_theory

---

## 2. SO(3) 与 Euler Angles

### 2.1 SO(3) 的定义

$$
SO(3) = \{\mathbf{R} \mid \mathbf{R}\in\mathbb{R}^{3\times 3}, \mathbf{R}^{T}\mathbf{R}=\mathbf{R}\mathbf{R}^{T}=\mathbf{I}, \det\mathbf{R}=+1\}
$$

"special" 一词指排除 $\det\mathbf{R}=-1$（即排除反射，只保留 proper rotation）。

### 2.2 Euler angle 分解

任意旋转可分解为三次基本旋转：

$$
{}^{A}\mathbf{R}_{B} = {}^{A}\mathbf{R}_{M}(\psi) \, {}^{M}\mathbf{R}_{N}(\phi) \, {}^{N}\mathbf{R}_{B}(\theta)
$$

变量解释：
- $\psi$：绕 {A} 的 x 轴旋转角
- $\phi$：绕中间 frame {M} 的 y 轴旋转角
- $\theta$：绕中间 frame {N} 的 z 轴旋转角
- {M}, {N} 是构造出来的 intermediate frames

对应矩阵连乘（公式 5）：
$$
\mathbf{R}_{B} = \mathbf{R}_{x}(\psi)\,\mathbf{R}_{y}(\phi)\,\mathbf{R}_{z}(\theta)
$$

**关键 caveat**: Euler 角在 $\phi = \pm 90°$ 时奇异（gimbal lock），所以 Euler 角只在 SO(3) 的一个开邻域上有效，整体上 SO(3) 不能被一个 $\mathbb{R}^3$ 坐标图覆盖——它是非平凡的 3-manifold。

**Intuition**: SO(3) ≅ $\mathbb{RP}^3$（实射影 3 维空间）。把 SO(3) 想象成半径为 $\pi$ 的实心球，且对径点等价（旋转 $\pi$ 与旋转 $-\pi$ 是同一个旋转）。这就是为什么 SO(3) "拓扑上不是 $\mathbb{R}^3$"。

参考：
- SO(3) topology: https://en.wikipedia.org/wiki/Rotation_group_SO(3)
- Gimbal lock: https://en.wikipedia.org/wiki/Gimbal_lock

---

## 3. 平移群 T(3) 与平面群 SE(2)

### 3.1 T(3)

当 $\mathbf{R} = \mathbf{I}$，homogeneous transform 退化为：
$$
{}^{A}\mathbf{A}_{B} = \begin{bmatrix} \mathbf{I}_3 & {}^{A}\mathbf{r}^{O'} \\ \mathbf{0} & 1 \end{bmatrix}
$$

两个平移的合成就是向量相加：${}^{A}\mathbf{r}^{O'}_{\text{total}} = {}^{A}\mathbf{r}^{O'}_1 + {}^{A}\mathbf{r}^{O'}_2$。

所以 T(3) 同构于 $(\mathbb{R}^3, +)$，是 abelian Lie group。

### 3.2 SE(2)

平面运动 = 平面旋转 + 平面平移，3 个参数 $(\theta, r_x, r_y)$：
$$
{}^{A}\mathbf{A}_{B} = \begin{bmatrix} \cos\theta & \sin\theta & 0 & r_x \\ -\sin\theta & \cos\theta & 0 & r_y \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}
$$

SE(2) 是 3 维 Lie group，常见于 mobile robot、planar mechanism。

---

## 4. Screw 位移与 One-Parameter Subgroup

### 4.1 Cylindrical motion (C-pair)

绕 z 轴同时旋转和平移，两个独立参数 $(\theta, k)$：
$$
\mathbf{A}_{B} = \begin{bmatrix} \cos\theta & \sin\theta & 0 & 0 \\ -\sin\theta & \cos\theta & 0 & 0 \\ 0 & 0 & 1 & k \\ 0 & 0 & 0 & 1 \end{bmatrix}
$$

### 4.2 Helical / Screw motion (H-pair)

当 $k = h\theta$，旋转与平移线性耦合：
$$
\mathbf{A}_{B} = \begin{bmatrix} \cos\theta & \sin\theta & 0 & 0 \\ -\sin\theta & \cos\theta & 0 & 0 \\ 0 & 0 & 1 & h\theta \\ 0 & 0 & 0 & 1 \end{bmatrix}
$$

- $h$：**pitch**（螺距），单位 [长度/弧度]
- $h = 0$ → pure rotation (R-pair)
- $h \to \infty$ → pure translation (P-pair)

关键性质：$\mathbf{A}(\theta_1)\mathbf{A}(\theta_2) = \mathbf{A}(\theta_1+\theta_2)$，这就是 **one-parameter subgroup**，同构于 $(\mathbb{R}, +)$。

**Intuition**: one-parameter subgroup 是 Lie group 中的一条"直线"——它经过 identity，沿固定方向"匀速"前进。在 SE(3) 中，这条"直线"对应一个 screw（一条轴 + 一个 pitch）。Chasles 定理说 SE(3) 中任何有限位移都可以由某个 screw 的有限转动实现，正是这一构造的离散版本。

参考 Chasles 定理: https://en.wikipedia.org/wiki/Chasles%27_theorem_(kinematics)

---

## 5. Twist 与瞬时运动学

### 5.1 Twist matrix 的推导

设 ${}^{A}\mathbf{A}_{B}(t)$ 随时间变化，刚体上点 P 在 {A} 中的位置：
$$
{}^{A}\mathbf{r}^{P}(t) = {}^{A}\mathbf{A}_{B}(t) \, {}^{A}\mathbf{r}^{P}(t_0)
$$

求导得速度：
$$
{}^{A}\mathbf{v}^{P}(t) = {}^{A}\dot{\mathbf{A}}_{B}(t) \, [{}^{A}\mathbf{A}_{B}(t)]^{-1} \, {}^{A}\mathbf{r}^{P}(t)
$$

定义 **twist matrix**：
$$
{}^{A}\mathbf{T}_{B} = {}^{A}\dot{\mathbf{A}}_{B} [{}^{A}\mathbf{A}_{B}]^{-1} = \begin{bmatrix} {}^{A}\mathbf{W}_{B} & {}^{A}\mathbf{v}^{\hat{O}} \\ \mathbf{0} & 0 \end{bmatrix}
$$

变量解释：
- ${}^{A}\mathbf{W}_{B} = {}^{A}\dot{\mathbf{R}}_{B} [{}^{A}\mathbf{R}_{B}]^{T}$：angular velocity matrix，3×3 skew-symmetric
- ${}^{A}\mathbf{v}^{\hat{O}} = {}^{A}\dot{\mathbf{r}}^{O'} - {}^{A}\mathbf{W}_{B} {}^{A}\mathbf{r}^{O'}$：body B 上某点 $\hat{O}$ 的速度，该点瞬时与 {A} 原点 $O$ 重合

### 5.2 Skew-symmetry 的来源

由 $\mathbf{R}^{T}\mathbf{R} = \mathbf{I}$ 求导：
$$
\dot{\mathbf{R}}\mathbf{R}^{T} + \mathbf{R}\dot{\mathbf{R}}^{T} = 0 \;\Rightarrow\; \mathbf{W} + \mathbf{W}^{T} = 0
$$

所以 $\mathbf{W}$ 是 skew-symmetric，可由 3 个独立参数 ${}^{A}\boldsymbol{\omega}_{B} = [\omega_1, \omega_2, \omega_3]^T$ 参数化：
$$
{}^{A}\mathbf{W}_{B} = \begin{bmatrix} 0 & -\omega_3 & \omega_2 \\ \omega_3 & 0 & -\omega_1 \\ -\omega_2 & \omega_1 & 0 \end{bmatrix} = [{}^{A}\boldsymbol{\omega}_{B}]^{\times}
$$

上标 $\times$ 表示把 3×1 向量变成 3×3 skew-symmetric 矩阵的 hat operator。

### 5.3 Twist vector

把 6 个独立参数组装成 6×1 vector：
$$
{}^{A}\mathbf{t}_{B} = \begin{bmatrix} {}^{A}\boldsymbol{\omega}_{B} \\ {}^{A}\mathbf{v}^{\hat{O}} \end{bmatrix} \in \mathbb{R}^{6}
$$

**这就是 Lie algebra se(3) 的元素**。Twist matrix 是 se(3) 在 4×4 matrix representation 下的像，twist vector 是其 6×1 坐标表示。

**Intuition**: Twist 是"瞬时 screw"——它告诉我们这一瞬间刚体在哪个轴上、以多大速度、带多大 pitch 旋转+滑动。把它积分一段时间就得到 finite displacement（即 SE(3) 元素）。exponential map $\exp : se(3) \to SE(3)$ 把 twist 映成 finite screw displacement。

参考：
- Murray, Li, Sastry 书 (MLS94): https://www.cds.caltech.edu/~murray/mlswiki/
- Twist in robotics: https://en.wikipedia.org/wiki/Screw_theory#Twist

---

## 6. Instantaneous Screw Axis (ISA)

这是 Chasles 定理的"无穷小版本"。

### 6.1 构造

给定一般 twist $[{}^{A}\boldsymbol{\omega}_{B}, {}^{A}\mathbf{v}^{\hat{O}}]^T$：

1. 令 $\mathbf{u} = {}^{A}\boldsymbol{\omega}_{B}/\omega$（单位向量，沿角速度方向）
2. 把线速度分解为平行 + 垂直分量：
   $$ {}^{A}\mathbf{v}^{\hat{O}} = h\,{}^{A}\boldsymbol{\omega}_{B} + \mathbf{r} \times {}^{A}\boldsymbol{\omega}_{B} $$
3. 求 $\mathbf{r}_n$（垂直于 $\boldsymbol{\omega}$ 的轴上点）和 $h$（pitch）：

$$
\omega = \|{}^{A}\boldsymbol{\omega}_{B}\|, \quad \mathbf{u} = \frac{{}^{A}\boldsymbol{\omega}_{B}}{\omega}
$$
$$
\mathbf{r}_n = \frac{{}^{A}\boldsymbol{\omega}_{B} \times {}^{A}\mathbf{v}^{\hat{O}}}{\omega^{2}}, \quad h = \frac{{}^{A}\boldsymbol{\omega}_{B} \cdot {}^{A}\mathbf{v}^{\hat{O}}}{\omega^{2}}
$$

变量解释：
- $\omega$：amplitude，角速度大小
- $\mathbf{u}$：ISA 方向单位向量
- $\mathbf{r}_n$：从 {A} 原点到 ISA 上最近点的位置向量
- $h$：pitch，每弧度平移距离

### 6.2 几何意义

刚体上任何点 Q 的速度由公式 (16) 给出：
$$
{}^{A}\mathbf{v}^{Q} = {}^{A}\mathbf{v}^{P} + {}^{A}\mathbf{W}_{B}\,\overrightarrow{PQ}
$$

- ${}^{A}\mathbf{v}^{P}$ 沿 ISA 方向（滑动分量），ISA 上所有点速度相同
- ${}^{A}\mathbf{W}_{B}\,\overrightarrow{PQ}$ 垂直于 ISA，正比于到轴距离（旋转分量）

整体上每个点的速度都切于一个 right circular helix，轴是 ISA，螺距是 $h$。这就把"瞬时刚体运动"几何化为一个 screw。

### 6.3 Plücker 坐标表示

Twist 可以写成（公式 18）：
$$
{}^{A}\mathbf{t}_{B} = \omega \begin{bmatrix} \mathbf{u} \\ h\mathbf{u} + \mathbf{r}_n \times \mathbf{u} \end{bmatrix}
$$

里面 $[\mathbf{u}; \mathbf{r}_n \times \mathbf{u}]$ 正是 ISA 这条直线的 **Plücker coordinates**。Twist vector 的 6 个分量也称作 screw coordinates，类比于直线的 Plücker 坐标。

参考：
- Plücker coordinates: https://en.wikipedia.org/wiki/Pl%C3%BCcker_coordinates
- Screw axis: https://en.wikipedia.org/wiki/Screw_axis

---

## 7. Wrench：力系统的对偶

### 7.1 力系统的化简

作用于刚体上的 n 个力 + m 个 pure couples 可合成：
$$
\mathbf{F} = \sum_{i=1}^{n}\mathbf{F}_i, \quad \mathbf{M}^{O} = \sum_{i=1}^{m}\mathbf{C}_i + \sum_{i=1}^{n}\mathbf{r}_i \times \mathbf{F}_i
$$

定义 wrench vector（公式 22）：
$$
{}^{A}\mathbf{w} = \begin{bmatrix} \mathbf{F} \\ \mathbf{M}^{O} \end{bmatrix} \in \mathbb{R}^{6}
$$

### 7.2 化简为 wrench about an axis

把 $\mathbf{M}^O$ 分解为平行于 $\mathbf{F}$ 的 $\mathbf{C}$ 和垂直于 $\mathbf{F}$ 的 $\mathbf{C}'$。垂直分量可由移动力的作用线产生，求出 $\mathbf{r}_n$：
$$
\mathbf{r}_n = \frac{\mathbf{F} \times \mathbf{M}^O}{\mathbf{F}\cdot\mathbf{F}}
$$

平移力作用线过 $\mathbf{r}_n$ 后，剩下的 couple $\mathbf{C}$ 平行于 $\mathbf{F}$，组合即为 **wrench**：force + parallel couple along same axis。

**Pitch of wrench**:
$$
\lambda = \frac{|\mathbf{C}|}{|\mathbf{F}|}
$$

- $\lambda = 0$：pure force（沿轴的纯力）
- $\lambda \to \infty$：pure couple（纯力偶）

### 7.3 对偶关系

Wrench 与 twist 在形式上完全对偶：

| | Twist | Wrench |
|--|-------|--------|
| 上半 3 维 | 角速度 $\boldsymbol{\omega}$ | 力 $\mathbf{F}$ |
| 下半 3 维 | 线速度 $\mathbf{v}^{\hat{O}}$ | 力矩 $\mathbf{M}^O$ |
| Pitch | $h = \boldsymbol{\omega}\cdot\mathbf{v}/\omega^2$ | $\lambda = \mathbf{F}\cdot\mathbf{M}/F^2$ |
| Axis | ISA | Wrench axis |
| 群作用 | 运动（kinematics） | 力（statics） |

这种对偶是 **kinematic-static duality**，由 Ball 在 1900 年系统化。在现代 robotics 里，它就是 $se(3)$ 与 $se^*(3)$ 之间的 natural pairing。

参考 Ball 的原著: https://archive.org/details/treatiseontheory00ballrich

---

## 8. Transformation Laws for Twists and Wrenches

### 8.1 Similarity transform

同一瞬时运动在 frame {F} 中描述（公式 25）：
$$
{}^{F}\mathbf{T}_{G} = {}^{F}\mathbf{A}_{A}\,{}^{A}\mathbf{T}_{B}\,({}^{F}\mathbf{A}_{A})^{-1}
$$

特别地，取 {F} = {B}（body frame），得到 body velocity：
$$
{}^{B}[{}^{A}\mathbf{T}_{B}] = ({}^{A}\mathbf{A}_{B})^{-1}\,{}^{A}\dot{\mathbf{A}}_{B}
$$

对比 spatial velocity ${}^{A}\mathbf{T}_{B} = {}^{A}\dot{\mathbf{A}}_{B}({}^{A}\mathbf{A}_{B})^{-1}$，区别在于 inverse 放在左边还是右边。

**Intuition**: 
- Spatial velocity: "在 space frame {A} 中看 body 的速度"
- Body velocity: "在 body frame {B} 中看 body 自己的运动"

二者表达同一个物理量，只是坐标系不同。

### 8.2 6×6 Adjoint transform

对 6×1 twist vector，变换关系为（公式 30）：
$$
{}^{F}[{}^{A}\mathbf{t}_{B}] = {}^{F}\mathbf{G}_{A}\,{}^{A}[{}^{A}\mathbf{t}_{B}]
$$

其中
$$
{}^{F}\mathbf{G}_{A} = \begin{bmatrix} {}^{F}\mathbf{R}_{A} & \mathbf{0} \\ [{}^{F}\mathbf{r}^{O}]^{\times}\,{}^{F}\mathbf{R}_{A} & {}^{F}\mathbf{R}_{A} \end{bmatrix}
$$

变量解释：
- ${}^{F}\mathbf{R}_{A}$：{A} 到 {F} 的 rotation
- $[{}^{F}\mathbf{r}^{O}]^{\times}$：{A} 原点 O 在 {F} 中位置向量的 hat matrix
- 这正是 SE(3) 上的 **Adjoint representation** $\mathrm{Ad}_g$，$g \in SE(3)$

同一变换也作用于 wrenches，验证了 twist/wrench 的对偶性。

参考 Adjoint representation: https://en.wikipedia.org/wiki/Adjoint_representation

---

## 9. Reciprocity：连接 Kinematics 与 Statics

### 9.1 Power pairing

Wrench $\mathbf{w} = [\mathbf{F}^T, \mathbf{M}^T]^T$ 作用于 twist $\mathbf{t} = [\boldsymbol{\omega}^T, \mathbf{v}^T]^T$ 上的瞬时功率：
$$
P = \mathbf{F}\cdot\mathbf{v} + \mathbf{M}\cdot\boldsymbol{\omega}
$$

写成矩阵形式（公式 31）：
$$
P = {}^{A}\mathbf{t}_{B}^{T}\,\mathbf{D}\,{}^{A}\mathbf{w} = {}^{A}\mathbf{w}^{T}\,\mathbf{D}\,{}^{A}\mathbf{t}_{B}
$$

其中
$$
\mathbf{D} = \begin{bmatrix} \mathbf{0}_{3\times 3} & \mathbf{I}_{3\times 3} \\ \mathbf{I}_{3\times 3} & \mathbf{0}_{3\times 3} \end{bmatrix}
$$

是交换上下 3 维块的对称矩阵。这正是 $se(3)$ 与 $se^*(3)$ 之间的 **Klein form** / reciprocal product。

### 9.2 两个 screw 的 reciprocal 条件

设两 screw $\mathbf{S}_1$（pitch $h_1$）、$\mathbf{S}_2$（pitch $h_2$），轴线间最短距离 $d$，夹角 $\phi$，则（公式 32）：
$$
(h_1 + h_2)\cos\phi - d\sin\phi = 0
$$

**几何解读**:
- $d\sin\phi$ 项：两轴偏离造成的"非共面"程度
- $(h_1+h_2)\cos\phi$ 项：pitch 提供的"补偿"
- 二者相等 → wrench on $\mathbf{S}_1$ 不对 twist about $\mathbf{S}_2$ 做功

### 9.3 三个关键推论

paper 给出三类常见 reciprocal 情形：

**情形 1 — Revolute joint**（pitch $h_1 \to \infty$ 即 twist 是 pure rotation）：
- Wrench 是 pure force（pitch 0）且作用线过转动轴
- 或 wrench pitch $h_2 = d\tan\phi$

**情形 2 — Frictionless point contact**（wrench 是沿法线的 pure force）：
- Twist 是 pure rotation（pitch 0）且过接触法线
- 或 twist pitch $h_1 = d\tan\phi$

**情形 3 — Prismatic joint**（twist 是纯平移，pitch $\to \infty$）：
- Wrench 是 pure couple（pitch $\infty$）
- 或 wrench 轴线垂直于 prismatic 轴

这些 reciprocal 关系是 **机构约束分析的基础**：理想运动副允许的 twist 与其约束 wrench 互为 reciprocal。这是 Hunt、Ball screw theory 在现代 robotics 里的核心应用。

参考：
- Reciprocal screw system: https://en.wikipedia.org/wiki/Screw_theory#Reciprocal_screws
- 机构约束分析: https://www.geometricfoundation.org/

---

## 10. 整体 Intuition Map

把全章串起来，可以画一张概念地图：

```
SE(3) (Lie group, 6-dim manifold)
  ├── elements: finite rigid displacements
  ├── subgroups: SO(3), SE(2), T(n), H(1), ...  ← 运动副分类
  └── tangent at identity: se(3) (Lie algebra)
            ├── elements: twists (瞬时运动)
            ├── matrix rep: T_B = A_dot A_inv
            ├── vector rep: t = [ω; v]^T
            └── 每个非零 twist ↔ 一条 ISA + pitch h + amplitude ω
                                    (Chasles 微分版)

Dual space: se*(3)
  └── elements: wrenches (力系统)
            └── 每个非零 wrench ↔ 一条 wrench axis + pitch λ + intensity F

Pairing: <w, t> = F·v + M·ω  (power)
  └── = 0  ⇔  reciprocal (做功为零)
              └── 用于机构约束、接触摩擦、合成系统分析
```

**核心信息**: 刚体运动学（kinematics）和静力学（statics）通过 screw theory 共享同一套代数结构。Twist 是运动 screw，wrench 是力 screw，二者通过 reciprocity 配对。这就是为什么 robot kinematic equations 和 static force-closure 分析可以"镜像式"地处理——它们在 $se(3) / se^*(3)$ 上是同一个数学对象。

---

## 11. 延伸与联想

### 11.1 与现代控制理论的连接

Twist $\mathbf{t} \in se(3)$ 是 SE(3) 上的"速度"。Lie group 上的运动方程可写为：
$$
\dot{g} = g\,\hat{\xi}, \quad g \in SE(3), \xi \in se(3)
$$

这正是机器人学中 manipulator dynamics 的标准形式。MLS94 第 5 章用它来推导 manipulator Jacobian。

### 11.2 与微分几何的连接

SE(3) 是非 compact、非交换 Lie group。其 Lie algebra se(3) 的 bracket 是：
$$
[\xi_1, \xi_2] = \text{ad}_{\xi_1}\xi_2, \quad \text{ad}_\xi = \begin{bmatrix} [\boldsymbol{\omega}]^\times & \mathbf{0} \\ [\mathbf{v}]^\times & [\boldsymbol{\omega}]^\times \end{bmatrix}
$$

Killing form 在 se(3) 上是退化的，所以 SE(3) 不是 semisimple——它有 abelian ideal（即 T(3) 的平移部分）。这就是为什么刚体动力学比 SO(3) 上的旋转动力学更微妙。

### 11.3 与机器人 grasp 分析

- 完整约束 grasp 的 force closure 等价于：grasp wrenches 张成整个 $se^*(3)$
- 形闭（form closure）等价于：接触法线 twists 的 reciprocal wrenches 张成整个 wrench 空间

paper 第 3.6 节末尾的三个推论正是这些现代理论的种子。

### 11.4 与 GPS/SLAM 的连接

SE(3) 是 6-DoF pose 估计的标准 state space。SLAM 中的 pose graph optimization、IMU pre-integration、ESKF（Error-State Kalman Filter）都在 SE(3) / se(3) 上做运算。Forster 的 IMU pre-integration paper 把 $\Delta\mathbf{R}, \Delta\mathbf{p}, \Delta\mathbf{v}$ 都放在 Lie group 上处理，正是 screw theory 的现代化。

参考：
- Forster et al. IMU preintegration: https://rpg.ifi.uzh.ch/docs/RSS15_forster.pdf
- Solà et al. "Micro Lie theory": https://arxiv.org/abs/1812.01537
- Lynch & Park, Modern Robotics (在线教材): http://hades.mech.northwestern.edu/index.php/Modern_Robotics

### 11.5 与 Poinsot/Ball 历史脉络

- 1806 Poinsot：力系统化简为 wrench
- 1830 Chasles：任何刚体位移 = screw displacement
- 1876 Ball: A Treatise on the Theory of Screws 系统化
- 1900 Ball 第二版（paper 引用 [1]）
- 1994 MLS94 把它纳入现代 robotics 数学框架
- 2000s Featherstone, Selig, Lynch-Park 等继续推进

这条脉络里 screw theory 几经沉浮，直到 robotics 兴起才被"激活"——它是少数几个由工程需求重新发现的古典几何理论。

---

## 12. 一些容易踩的坑

1. **Twist 的表达 frame**：${}^{A}\mathbf{t}_{B}$ 中的线速度是"瞬时与 {A} 原点重合的 body 点"的速度，不是 body 原点的速度。两者差一个 $\boldsymbol{\omega}\times\mathbf{r}^{O'}$。这点 paper 公式 (14)-(15) 强调过。

2. **Spatial vs Body velocity**：两者数值不同（差一个 adjoint），但描述同一个运动。规划算法里混用会出 bug。

3. **Pitch 的符号**：$h = (\boldsymbol{\omega}\cdot\mathbf{v})/\omega^2$ 可正可负，决定螺旋是右旋还是左旋。

4. **Reciprocal 的对称性**：$\langle \mathbf{w}, \mathbf{t}\rangle = 0 \Leftrightarrow \langle \mathbf{t}, \mathbf{w}\rangle = 0$。所以"wrench 对 twist 不做功"自动蕴含"twist 对 wrench 不接收能量"。

5. **Singular configurations**：Euler 角、ISA 计算在 $\omega = 0$ 时退化（这时运动是纯平移，ISA 在无穷远）。

---

## 参考资源汇总

- **Ball, Treatise on Screws (1900)**: https://archive.org/details/treatiseontheory00ballrich
- **MLS94 (Murray, Li, Sastry)**: https://www.cds.caltech.edu/~murray/mlswiki/
- **Lynch & Park, Modern Robotics**: http://hades.mech.northwestern.edu/index.php/Modern_Robotics
- **Selig, Geometric Fundamentals of Robotics**: https://link.springer.com/book/10.1007/978-0-387-28970-8
- **Featherstone, Rigid Body Dynamics Algorithms**: http://royfeatherstone.org/
- **Micro Lie theory (Solà)**: https://arxiv.org/abs/1812.01537
- **SE(3) on Wikipedia**: https://en.wikipedia.org/wiki/Euclidean_group
- **Screw theory on Wikipedia**: https://en.wikipedia.org/wiki/Screw_theory
- **Plücker coordinates**: https://en.wikipedia.org/wiki/Pl%C3%BCcker_coordinates
- **Forster IMU preintegration**: https://rpg.ifi.uzh.ch/docs/RSS15_forster.pdf

---

如果你想深入，下一步建议是：(a) 把 MLS94 第 2-3 章做完习题；(b) 用 Python/NumPy 实现 $\exp: se(3) \to SE(3)$ 并验证 Chasles 定理；(c) 读 Selig 的书把 Riemannian metric、Killing form 这些几何对象在 SE(3) 上具体算一遍。Screw theory 真正"开窍"的瞬间，是你发现 robot kinematics、statics、dynamics 都可以用同一套 6×1 vector 和 6×6 matrix 处理——这就是 Lie group 的力量。
