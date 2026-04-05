在光滑无摩擦平面上，两个匀速直线运动的 Box 发生碰撞，你调用 MuJoCo 计算下一帧状态。MuJoCo 并不会像简单的游戏引擎那样直接修改速度向量，而是基于一个严密的 **Pipeline**，依次执行 **Collision Detection**（碰撞检测）、**Constraint Formulation**（约束构建）、**Inverse Dynamics**（逆动力学，用于受力平衡）或 **Forward Dynamics**（前向动力学，求解加速度），以及最后的 **Time Integration**（时间积分）。

对于你的 Box 场景，核心在于如何求解非穿透约束。以下是 MuJoCo 计算下一时刻状态的详细技术流程：

### 1. 整体计算流程 Pipeline

当你调用 `mj_step` 或者分步调用 `mj_forward` 时，MuJoCo 执行以下阶段：

1.  **mj_position**: 更新关节位置和场景中的物体几何坐标。
2.  **mj_velocity**: 计算关节速度和物体空间速度。
3.  **mj_sensor**: (可选) 计算传感器数据。
4.  **mj_forward**: 这是核心。它通过 **Constraint-based** 方法求解加速度。
    *   子步骤 A: **Collision Detection**。
    *   子步骤 B: **Active Constraints Selection**。
    *   子步骤 C: **Constraint Solver**（求解 KKT 系统或优化问题）。
5.  **mj_integrate**: 使用求解出的加速度更新速度和位置。

### 2. 碰撞检测

在你的场景中，两个 Box 都是 **Geom**。MuJoCo 的碰撞检测模块并不仅仅是检查重叠，而是为了生成接触约束。

*   **Broad Phase**: MuJoCo 使用空间划分或 Bounding Volume Hierarchy (BVH) 快速排除不可能碰撞的物体。
*   **Narrow Phase**: 对于两个 Box，MuJoCo 使用 **SDF (Signed Distance Field)** 或者解析几何算法（如 SAT 分离轴定理的变体）来确定接触情况。
*   **Contact Generation**: 一旦检测到重叠，MuJoCo 会生成一个或多个 `mjContact` 结构体。
    *   **Variables (变量)**:
        *   `pos`: 接触点在世界坐标系中的位置 $p_c \in \mathbb{R}^3$。
        *   `frame`: 接触坐标系，其中 $z$ 轴是接触法线方向 $n_c$。
        *   `dist`: 穿透深度 $d$。如果 $d < 0$，表示未接触或刚好接触；如果 $d > 0$，表示发生穿透。对于光滑平面上的 Box，MuJoCo 允许微小的“软穿透”来定义接触力。
        *   `geom1`, `geom2`: 涉及碰撞的两个几何体索引。

### 3. 约束建模与 Jacobian 映射

MuJoCo 不使用 Penalty Method（纯弹簧阻尼模型）来处理刚体碰撞（虽然可以设置弹簧属性），而是使用 **Complementarity**（互补性）条件将接触处理为带不等式约束的动力学问题。

#### 3.1 广义坐标与动力学方程
整个系统的状态由 **Generalized Coordinates (广义坐标)** $q$ 和 **Generalized Velocities (广义速度)** $v$ 描述。
对于两个 Box：
*   $q = [x_1, y_1, z_1, q_{w1}, q_{x1}, q_{y1}, q_{z1}, x_2, y_2, z_2, q_{w2}, ...]^T$。

前向动力学方程为：
$$ M(q) \dot{v} + C(q, v) = \tau_{app} + \tau_{constraint} $$
其中：
*   $M(q) \in \mathbb{R}^{n_v \times n_v}$ 是 **Mass Matrix (质量矩阵)**，包含质量和转动惯量。
*   $\dot{v} \in \mathbb{R}^{n_v}$ 是 **Generalized Acceleration (广义加速度)**。
*   $C(q, v) \in \mathbb{R}^{n_v}$ 是 **Bias Forces (科里奥利力、离心力和重力)**。在你的光滑平面场景中，如果没有重力，这一项主要由速度的二次项构成；如果有重力，它包含 $G(q)$。
*   $\tau_{app}$ 是外部施加的广义力（这里是 0）。
*   $\tau_{constraint}$ 是通过 Jacobian 映射的接触力。

#### 3.2 Jacobian 矩阵 $J_c$
接触发生在笛卡尔空间，但动力学求解是在广义坐标空间。需要通过 Jacobian 将接触力 $f_c$ 映射为广义力 $\tau_{c}$。
$$ \tau_{c} = J_c(q)^T f_c $$

$J_c \in \mathbb{R}^{3 \times n_v}$（对于无摩擦的单点接触，若考虑摩擦则是 $6 \times n_v$）。
*   $J_c$ 的前 3 行对应接触点法线方向的平移速度对广义速度的偏导数。
*   具体计算公式：接触点速度 $v_{contact} = J_c v$。

### 4. 求解器 - 核心 Magic

这是最关键的步骤。MuJoCo 并不直接“弹开”盒子，而是求解下一时刻的加速度 $\dot{v}$ 和接触力 $f_c$，使得它们满足互补性条件。

该问题被构建为一个 **Convex Optimization（凸优化）** 问题，具体来说是求解 **KKT (Karush-Kuhn-Tucker)** 条件。

#### 4.1 定义约束条件
为了保证物体不穿透（或者是软穿透），满足下述条件：
$$ v_{normal} = J_c v \ge 0 \quad (\text{分离速度} \ge 0) $$
$$ f_c \ge 0 \quad (\text{接触推力} \ge 0) $$
$$ f_c \cdot v_{normal} = 0 \quad (\text{互补性}) $$

实际上，MuJoCo 使用 **Soft Constraint (软约束)** 模型，即允许极小的 penetration 以获得数值稳定性。这通过引入 **Regularization (正则化)** 参数 $R$ 来实现。

#### 4.2 构建求解系统
MuJoCo 的求解器（通常是 `mjCG` - Conjugate Gradient 或 `mjNewton` - Newton 方法）求解以下线性系统（类似于 QP 的 KKT 系统）：

$$
\begin{bmatrix}
M & -J_c^T \\
J_c & -R
\end{bmatrix}
\begin{bmatrix}
\dot{v} \\
f_c
\end{bmatrix}
=
\begin{bmatrix}
-C \\
-\epsilon v_{contact}
\end{bmatrix}
$$

*   **Variables Explanation (变量解释)**:
    *   $\dot{v}$: 我们需要的广义加速度，用于积分下一帧。
    *   $f_c$: 拉格朗日乘子，即接触点在法线方向上的力。
    *   $R$: 对角矩阵，表示 **Softness**。如果 $R=0$，则是完全刚体；$R > 0$ 允许微小穿透，类似弹簧系数。
    *   $\epsilon$: **Damping** 参数，防止持续震荡，对应接触阻尼。

#### 4.3 求解过程
由于 Box 是无摩擦的（Frictionless），$J_c$ 只有法线方向的一行。如果是多点接触（比如 Box 角落刮擦），矩阵会变大。
求解器会迭代寻找一组 $(\dot{v}, f_c)$，使得：
1.  动力学方程 $M\dot{v} + C = J_c^T f_c$ 平衡。
2.  接触约束 $J_c \dot{v} \approx - v_{prev}$ （将接近速度变为分离速度或零）。

这是一个 **Primal-Dual Interior Point** 类型的优化过程。MuJoCo 专门针对这种稀疏矩阵进行了高度优化。

### 5. 时间积分

一旦求解器算出了 $\dot{v}$（广义加速度），MuJoCo 使用 **Semi-implicit Euler (半隐式欧拉法)** 进行积分更新状态。

$$ v_{t+1} = v_t + \dot{v} \cdot dt $$
$$ q_{t+1} = q_t + v_{t+1} \cdot dt $$

*   **Why Semi-implicit?**: 它比标准的 Symplectic Euler 更好地保持了能量守恒，且对弹簧-阻尼系统（即接触约束）更加稳定。
*   $dt$ 是 `mjModel.opt.timestep`。

在你的 Box 碰撞案例中：
*   碰撞前：$C=0, f_c=0 \implies \dot{v}=0$，速度不变。
*   接触时刻：$f_c$ 瞬间产生（在模型离散时间片内），导致 $v_{t+1}$ 法向分量反转或变为零，完成“反弹”或“停止”。

### 6. 详细公式与变量含义解析

为了建立更深刻的 Intuition，我们看接触力的具体计算公式。

MuJoCo 计算的法向接触力 $f_n$ 实际上是基于 Penetration depth $d$ 和 Penetration velocity $v_p$ 的近似函数，但被封装在 QP 求解中。其物理效果等同于：

$$ f_n = k \cdot d - b \cdot v_p $$

但在求解器内部，这是对 $R^{-1}$ 和 $\epsilon$ 的隐式求解。对于**无摩擦**盒子：

1.  **Collision Cost Function (代价函数)**:
    求解器实际上在最小化：
    $$ \text{Cost} = \frac{1}{2} \dot{v}^T M \dot{v} + \frac{1}{2} f_c^T R f_c $$
    受限于约束。
    这解释了为什么系统倾向于寻找需要最小“力”的解，同时也符合牛顿定律。

2.  **Friction Cone (摩擦锥)**:
    由于是无摩擦平面，摩擦锥退化为一个点，即切向力 $f_t = 0$。这大大简化了求解器的工作量，不需要处理 Coulomb friction inequality $|f_t| \le \mu f_n$。

### 7. 数据结构细节

当你获取 `mjData` 时，关注以下字段来理解碰撞发生了什么：

*   **`ncon`**: 当前时刻检测到的 Contact 对数量。如果是 0，说明没有算碰撞。
*   **`contact`**: 一个 `mjContact` 数组，长度 `ncon`。
    *   `contact[pos]`: 世界坐标下的接触点。
    *   `contact[frame]`: 接触法线矩阵。第 2 列（索引 2, z轴）就是法线向量 $n$。
    *   `contact[dist]`: 穿透深度。
*   **`efc_J`**: **Constraint Jacobian**。这是一个稀疏矩阵，存储了 $J_c$ 的非零元素。这是连接几何碰撞和动力学方程的桥梁。
*   **`efc_force`**: 求解器计算出的约束力 $f_c$。

### 8. 总结与直觉建立

MuJoCo 处理两个 Box 碰撞，并不是直接写 `if collision then reflect_velocity`。
它做的实际上是：

1.  **Detect**: 发现两个 Box 穿透了。
2.  **Map**: 算出这个穿透点的法线方向 $n$，并算出这个点的速度怎么和 Box 的整体速度挂钩（算 Jacobian $J$）。
3.  **Optimize**: 设定一个目标——“我要让这两个点在下一帧不穿透，或者按某种弹性系数分离，且所需的力要符合物理定律（牛顿第二定律）”。
4.  **Solve**: 解一个大型线性方程组（KKT System），算出必须施加的力 $f_c$ 和由此产生的系统加速度 $\dot{v}$。
5.  **Update**: 用 $\dot{v}$ 修改速度，用新速度修改位置。

所以，你看到的下一帧状态，是**优化解**的结果，而不是简单的几何反射。这也解释了为什么在 MuJoCo 中，如果 Solver iteration（迭代次数）设置得太低，物体看起来会像“果冻”一样软，因为优化还没收敛到刚体解。

### References and Web Links

*   **MuJoCo Documentation (Official)**: [MuJoCo Documentation](https://mujoco.readthedocs.io/en/stable/)
*   **MuJoCo Overview (Todorov et al.)**: [MuJoCo: A physics engine for model-based control](https://doi.org/10.1109/MRA.2012.2206314)
*   **Theory of Soft Contacts**: [MuJoCo Contact Modeling](https://mujoco.readthedocs.io/en/stable/modeling.html#contact)
*   **Complementarity Problems in Physics**: [Complementarity Formulation for Rigid Body Dynamics](https://www.cs.cmu.edu/~baraff/papers/sig96.pdf) (David Baraff's seminal work, which MuJoCo draws heavily from in spirit).
*   **Semi-implicit Euler Integration**: [Semi-implicit Euler Method](https://en.wikipedia.org/wiki/Semi-implicit_Euler_method)