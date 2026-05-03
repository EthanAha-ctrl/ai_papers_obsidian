

# GJK 算法详解：从第一性原理出发构建直觉

---

## 1. 问题本质：两个凸集之间的最小距离

**第一性原理思考**：给定空间中两个凸体 $A$ 和 $B$，我们想回答一个最基本的问题——**它们之间的最短距离是多少？** 如果距离为零，则它们相交（碰撞）。

这个看似简单的问题，当形状复杂时（多面体可能有成千上万个顶点），暴力枚举所有点对的复杂度是 $O(|A| \cdot |B|)$，完全不可行。GJK 的核心洞察是：**我们可以将双集问题转化为单集问题，然后用迭代方法逼近答案**。

---

## 2. 数学基石：Minkowski Difference

### 2.1 定义

两个凸集 $A$ 和 $B$ 的 **Minkowski difference** 定义为：

$$A - B = \{ \mathbf{a} - \mathbf{b} \mid \mathbf{a} \in A, \mathbf{b} \in B \}$$

其中：
- $\mathbf{a}$ 是集合 $A$ 中的任意点
- $\mathbf{b}$ 是集合 $B$ 中的任意点
- 结果是**所有可能的差向量**构成的集合

### 2.2 为什么这个变换如此重要？——三个关键性质

| 性质 | 数学表述 | 直觉含义 |
|------|----------|----------|
| **凸性保持** | $A, B$ 凸 $\Rightarrow$ $A - B$ 凸 | 单集问题仍可用凸优化方法 |
| **距离等价** | $\min_{\mathbf{a}\in A, \mathbf{b}\in B}\|\mathbf{a}-\mathbf{b}\| = \min_{\mathbf{c}\in A-B}\|\mathbf{c}\|$ | 两集距离 = 原点到 $A-B$ 的最短距离 |
| **相交判定** | $\mathbf{0} \in A - B \iff A \cap B \neq \emptyset$ | 原点在 $A-B$ 内 ⟺ 两集相交 |

> **直觉构建**：想象将 $B$ 的每个点平移到原点，则 $A$ 也跟着"相对移动"。$A - B$ 就是"从 $B$ 的视角看，$A$ 在哪"的所有可能位置。如果原点在这个差集内部，说明存在 $\mathbf{a} = \mathbf{b}$，即碰撞。

### 2.3 几何直觉

$A - B$ 可视化为 $A$ 与 $-B$（$B$ 关于原点的镜像）的 **Minkowski sum**：

$$A - B = A + (-B)$$

你可以想象把 $-B$ 的原点放在 $A$ 的边界上"滑动"，扫过的所有区域就是 $A - B$。

---

## 3. 核心原语：Support Function（支撑函数）

### 3.1 定义

对于凸集 $C$ 和非零方向向量 $\mathbf{d}$，支撑函数定义为：

$$s_C(\mathbf{d}) = \sup_{\mathbf{x} \in C} \mathbf{x} \cdot \mathbf{d}$$

对应的 **支撑点** $\mathbf{x}^* \in C$ 满足：

$$\mathbf{x}^* \cdot \mathbf{d} = s_C(\mathbf{d})$$

> **直觉**：支撑函数就是"沿方向 $\mathbf{d}$ 看，这个集合能伸多远"。支撑点就是最远的那个边界点。

### 3.2 不同形状的计算

| 形状 | 支撑函数计算 | 时间复杂度 |
|------|------------|-----------|
| **多面体**（顶点 $\{\mathbf{v}_i\}$） | $s_C(\mathbf{d}) = \max_i(\mathbf{v}_i \cdot \mathbf{d})$，取最大点积对应的顶点 | $O(n)$，$n$ 为顶点数 |
| **球**（中心 $\mathbf{c}$，半径 $r$） | $\mathbf{x}^* = \mathbf{c} + r \cdot \frac{\mathbf{d}}{\|\mathbf{d}\|}$ | $O(1)$ |
| **胶囊体**（两端球心 $\mathbf{s}_1, \mathbf{s}_2$，半径 $r$） | $\mathbf{x}^* = \text{closest\_point\_on\_segment}(\mathbf{d}) + r \cdot \frac{\mathbf{d}}{\|\mathbf{d}\|}$ | $O(1)$ |
| **OBB** | 沿各轴方向取极值点的线性组合 | $O(1)$ |

### 3.3 Minkowski Difference 的支撑函数

**关键性质**（无需显式构造 $A - B$）：

$$s_{A-B}(\mathbf{d}) = s_A(\mathbf{d}) - s_B(-\mathbf{d})$$

> 这就是 GJK 高效的根本原因！你**永远不需要真正构建 $A - B$**，只需要分别对 $A$ 和 $B$ 查询支撑点，然后相减即可得到 $A - B$ 上的支撑点。

---

## 4. 算法核心：Simplex 迭代

### 4.1 Simplex 是什么？

**Simplex**（单纯形）是 $\mathbb{R}^n$ 中 $n+1$ 个仿射独立点构成的凸包：

| 维度 | Simplex 类型 | 顶点数 | 形象 |
|------|------------|--------|------|
| 0 | Point (0-simplex) | 1 | 一个点 |
| 1 | Edge (1-simplex) | 2 | 线段 |
| 2 | Triangle (2-simplex) | 3 | 三角形 |
| 3 | Tetrahedron (3-simplex) | 4 | 四面体 |

> **直觉**：GJK 的思路是用一个越来越大的单纯形"逼近"原点在 $A - B$ 中的位置。如果单纯形能"包住"原点，说明 $A$ 和 $B$ 碰撞；否则，单纯形上离原点最近的点就给出了最小距离的估计。

### 4.2 算法流程（伪代码详解）

```
function GJK_Distance(A, B, ε):
    d ← 初始方向（如 (1,0,0) 或 A.center - B.center 的归一化）
    
    // === 步骤1：获取初始支撑点 ===
    p ← support(A-B, d)   // 即 s_A(d) - s_B(-d)
    S ← {p}               // 初始 0-simplex
    
    if ‖p‖ < ε:
        return 0           // 几乎为零，已相交
    
    d ← -p                // 方向指向原点
    
    // === 步骤2：主循环 ===
    while true:
        // 2a. 获取新支撑点
        p ← support(A-B, d)
        
        // 2b. 分离测试（关键终止条件！）
        if p · d ≤ 0:
            // 新点没有越过"原点侧"
            // 说明沿此方向已无法更靠近原点 → 确认分离
            closest ← closest_point_to_origin(S)
            return ‖closest‖
        
        // 2c. 扩展单纯形
        S ← S ∪ {p}
        
        // 2d. 包含测试：原点在 S 内？
        if contains_origin(S):
            return 0       // 碰撞！
        
        // 2e. 缩减：只保留离原点最近的特征
        closest ← closest_point_to_origin(S)
        S ← minimal_simplex_containing(closest, S)
        
        // 2f. 更新搜索方向
        d ← -closest
        
        if ‖d‖ < ε:
            return ‖closest‖
```

### 4.3 流程图（架构级）

```
┌─────────────────────────────────────────────┐
│           初始化                              │
│  d = 初始方向 (e.g. 归一化的中心差)            │
│  p₀ = s_A(d) - s_B(-d)                      │
│  S = {p₀}                                   │
│  d = -p₀                                    │
└──────────────────┬──────────────────────────┘
                   │
          ┌────────▼────────┐
          │   获取新支撑点    │
          │  p = s_A(d)     │
          │    - s_B(-d)    │
          └────────┬────────┘
                   │
          ┌────────▼────────┐    p · d ≤ 0
          │  分离测试        │──────────────► 返回 ‖closest(S)‖
          │  p · d > 0 ?    │               （分离，距离已知）
          └────────┬────────┘
                   │ Yes
          ┌────────▼────────┐
          │  S ← S ∪ {p}   │
          │  扩展单纯形      │
          └────────┬────────┘
                   │
          ┌────────▼────────┐    原点 ∈ S
          │  包含测试        │──────────────► 返回 0
          │  原点在S内？     │               （碰撞！）
          └────────┬────────┘
                   │ No
          ┌────────▼────────┐
          │  缩减单纯形      │
          │  只保留最近特征   │
          │  更新 d = -closest│
          └────────┬────────┘
                   │
                   └──────────────► 回到"获取新支撑点"
```

---

## 5. Simplex 缩减：Johnson's Sub-algorithm 与 Voronoi Region

### 5.1 核心问题

在扩展单纯形后（最多 $n+1$ 个点），我们需要找到**单纯形上离原点最近的点**，并**丢弃不必要的顶点**，保持单纯形最小化。

### 5.2 Voronoi Region 方法

以 3D 四面体为例，其 Voronoi 图将空间分为 **15 个区域**：
- 4 个顶点区域（0-simplex）
- 6 条边区域（1-simplex）  
- 4 个面区域（2-simplex）
- 1 个内部区域（3-simplex）

**判定流程**：

```
原点在哪个 Voronoi Region？
│
├── 在四面体内部 → 碰撞！距离 = 0
│
├── 在某面区域 → 缩减为该三角形（3 点）
│   │
│   ├── 在三角形内部 → 原点到面的投影
│   ├── 在某边区域 → 缩减为线段（2 点）
│   └── 在某点区域 → 缩减为点（1 点）
│
├── 在某边区域 → 缩减为线段（2 点）
│
└── 在某点区域 → 缩减为点（1 点）
```

### 5.3 具体计算：线段（1-simplex）上的最近点

给定线段 $\mathbf{ab}$（两个支撑点），原点 $\mathbf{0}$ 在其上的最近点参数为：

$$t = \frac{-\mathbf{a} \cdot (\mathbf{b} - \mathbf{a})}{|\mathbf{b} - \mathbf{a}|^2}$$

其中：
- $t$ 是参数，最近点为 $\mathbf{a} + t(\mathbf{b} - \mathbf{a})$
- 若 $t \leq 0$：最近点是 $\mathbf{a}$，缩减为点
- 若 $t \geq 1$：最近点是 $\mathbf{b}$，缩减为点
- 若 $0 < t < 1$：最近点在线段内部，保留两个点

### 5.4 具体计算：三角形（2-simplex）上的最近点

三角形顶点 $\mathbf{a}, \mathbf{b}, \mathbf{c}$，原点的重心坐标：

$$\lambda_a = \frac{|\triangle(\mathbf{0}, \mathbf{b}, \mathbf{c})|}{|\triangle(\mathbf{a}, \mathbf{b}, \mathbf{c})|}, \quad \lambda_b = \frac{|\triangle(\mathbf{a}, \mathbf{0}, \mathbf{c})|}{|\triangle(\mathbf{a}, \mathbf{b}, \mathbf{c})|}, \quad \lambda_c = \frac{|\triangle(\mathbf{a}, \mathbf{b}, \mathbf{0})|}{|\triangle(\mathbf{a}, \mathbf{b}, \mathbf{c})|}$$

其中 $|\triangle(\cdot)|$ 表示三角形面积。

- 若所有 $\lambda_i \geq 0$：原点在三角形内部 → 可能碰撞
- 若某个 $\lambda_i < 0$：原点在该顶点对侧 → 缩减为对面的边

### 5.5 具体计算：四面体（3-simplex）上的最近点

四面体顶点 $\mathbf{a}, \mathbf{b}, \mathbf{c}, \mathbf{d}$，使用**有符号体积**：

$$V(\mathbf{p}_1, \mathbf{p}_2, \mathbf{p}_3, \mathbf{p}_4) = \frac{1}{6}\det\begin{pmatrix} \mathbf{p}_2 - \mathbf{p}_1 \\ \mathbf{p}_3 - \mathbf{p}_1 \\ \mathbf{p}_4 - \mathbf{p}_1 \end{pmatrix}$$

重心坐标：

$$\lambda_a = \frac{V(\mathbf{0}, \mathbf{b}, \mathbf{c}, \mathbf{d})}{V(\mathbf{a}, \mathbf{b}, \mathbf{c}, \mathbf{d})}, \quad \lambda_b = \frac{V(\mathbf{a}, \mathbf{0}, \mathbf{c}, \mathbf{d})}{V(\mathbf{a}, \mathbf{b}, \mathbf{c}, \mathbf{d})}, \quad \ldots$$

- 所有 $\lambda_i > 0$：原点在内部 → 碰撞
- 否则：缩减为使 $\lambda$ 为正的子单纯形

---

## 6. 终止条件详解

| 条件 | 数学表述 | 含义 |
|------|----------|------|
| **碰撞** | $\mathbf{0} \in \text{conv}(S)$ | 原点在当前单纯形内，距离 = 0 |
| **分离确认** | $\mathbf{p}_{new} \cdot \mathbf{d} \leq 0$ | 新支撑点未越过原点侧，无法更近 |
| **收敛** | $\mathbf{p}_{new} \in S$（或 $\|\mathbf{d}_{new} - \mathbf{d}_{old}\| < \varepsilon$） | 无进展，当前估计已足够 |
| **数值精度** | $\|\mathbf{d}\| < \varepsilon$ | 方向向量接近零，距离估计稳定 |

> **关键直觉**：分离测试 $\mathbf{p}_{new} \cdot \mathbf{d} \leq 0$ 的意思是——我们沿着方向 $\mathbf{d}$ 去找 $A - B$ 中最远的点，如果这个最远点都没能到达原点那一侧（$\mathbf{p} \cdot \mathbf{d}$ 度量了沿 $\mathbf{d}$ 方向的投影），那原点一定在 $A - B$ 外部。

---

## 7. 完整示例：2D 两个圆的距离

**设定**：
- 圆 $A$：中心 $\mathbf{c}_1$，半径 $r_1$
- 圆 $B$：中心 $\mathbf{c}_2$，半径 $r_2$
- 不相交条件：$\|\mathbf{c}_1 - \mathbf{c}_2\| > r_1 + r_2$

**支撑函数**：$s(\mathbf{d}) = \mathbf{c} + r \cdot \frac{\mathbf{d}}{\|\mathbf{d}\|}$

**迭代过程**：

| 迭代 | 方向 $\mathbf{d}$ | $s_A(\mathbf{d})$ | $s_B(-\mathbf{d})$ | 支撑点 $\mathbf{p} = s_A - s_B$ | 操作 |
|------|-----------|-----------|------------|------------------|------|
| 0 | $\mathbf{c}_2 - \mathbf{c}_1$（归一化） | $\mathbf{c}_1 + r_1\mathbf{d}$ | $\mathbf{c}_2 - r_2\mathbf{d}$ | $(\mathbf{c}_1 - \mathbf{c}_2) + (r_1 + r_2)\mathbf{d}$ | 初始 0-simplex |
| 1 | $-\mathbf{p}_0$ | 同上 | 同上 | **与 $\mathbf{p}_0$ 相同** | 无新点，收敛！ |

**结果**：$\|\mathbf{p}_0\| = \|\mathbf{c}_1 - \mathbf{c}_2\| - r_1 - r_2$（圆的间距减去半径和）

> **为什么这么快？** 因为圆的支撑函数是闭式解，支撑点唯一，GJK 一步就找到了最近的特征对。

---

## 8. 数值稳定性问题与解决方案

### 8.1 退化情况

| 退化类型 | 原因 | 后果 |
|----------|------|------|
| **共面点** | 四面体四个点近似共面 | 行列式≈0，重心坐标计算溢出 |
| **支撑点不唯一** | 多面体的边/面方向与查询方向平行 | 不同迭代可能选到不同等距顶点 |
| **方向向量退化** | $\mathbf{d} \approx \mathbf{0}$ | 支撑查询无意义 |

### 8.2 Johnson's Sub-algorithm 的问题

原始的 Johnson 子算法通过求解线性方程组来计算重心坐标：

$$\begin{pmatrix} \mathbf{p}_2 - \mathbf{p}_1 & \mathbf{p}_3 - \mathbf{p}_1 & \mathbf{p}_4 - \mathbf{p}_1 \end{pmatrix} \begin{pmatrix} \lambda_2 \\ \lambda_3 \\ \lambda_4 \end{pmatrix} = -\mathbf{p}_1$$

当矩阵接近奇异时（近退化），**灾难性抵消**（catastrophic cancellation）导致结果严重失真。

### 8.3 Signed Volume 方法（2017年改进）

Montanari, Petrinic, Barbieri (2017) 提出用**有符号体积比**替代直接行列式求逆：

$$\lambda_i = \frac{V(\mathbf{0}, \text{face opposite to } \mathbf{p}_i)}{V(\mathbf{p}_1, \mathbf{p}_2, \mathbf{p}_3, \mathbf{p}_4)}$$

**优势**：
- 仿射不变性：对坐标缩放不敏感
- 数值稳定性提升至机器精度
- 执行时间减少 **15–30%**

> 参考：[Montanari et al., 2017, "A new algorithm for computing the closest point on a simplex"](https://doi.org/10.1016/j.compgeom.2017.06.002)

### 8.4 实用技巧

| 技巧 | 做法 | 效果 |
|------|------|------|
| **ε-tolerance** | dot product < 1e-6 视为零 | 避免噪声敏感 |
| **Double precision** | 使用 float64 | 防止迭代累积误差 |
| **Warm starting** | 缓存上一帧的 simplex | 利用时间连贯性，平均 1-3 次迭代 |
| **小扰动** | 对退化顶点加微小随机偏移 | 避免零行列式 |

---

## 9. 与相关算法的对比

### 9.1 GJK vs SAT（Separating Axis Theorem）

| 特性 | GJK | SAT |
|------|-----|-----|
| **适用范围** | 任意凸体（通过支撑函数） | 仅多面体 |
| **时间复杂度** | 每次迭代 $O(n)$，平均 3-6 次 | $O(k)$，$k$ 为轴数（OBB: 15轴） |
| **穿透深度** | 需配合 EPA | 需额外计算 |
| **动态场景** | 天然支持 warm starting | 每帧重新计算 |
| **精度控制** | ε-tolerance 灵活 | 确定性 |

### 9.2 GJK vs Linear Programming

GJK 本质上是求解如下凸优化问题的**Frank-Wolfe 方法**（条件梯度法）的特例：

$$\min_{\mathbf{x} \in A - B} \|\mathbf{x}\|^2$$

与通用 LP 求解器相比，GJK 利用了问题的特殊结构（距离到原点的投影），达到了接近常数时间的性能。

### 9.3 复杂度分析

| 指标 | 值 |
|------|-----|
| 每次迭代复杂度 | $O(n)$（$n$ 为顶点数，支撑查询） |
| 3D 中最大迭代数 | 理论上 $O(1)$（最多 4 次扩展到四面体），实际 5-10 次 |
| 总复杂度（平均） | 近似 $O(n)$ |
| 空间复杂度 | $O(1)$（只存储最多 4 个支撑点） |

---

## 10. 扩展：EPA（Expanding Polytope Algorithm）

当 GJK 检测到碰撞（原点在 $A - B$ 内部）时，**EPA** 接手计算**穿透深度**（penetration depth）：

### 10.1 算法思路

```
┌──────────────────────────────────────┐
│  GJK 终止：原点在单纯形内部           │
│  → 碰撞确认，但穿透深度未知           │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  EPA：从 GJK 的最终单纯形开始         │
│  迭代扩展多面体，直到接触 A-B 的边界   │
│  穿透深度 = 原点到多面体边界的最短距离 │
└──────────────────────────────────────┘
```

**步骤**：
1. 从 GJK 最终 simplex 出发
2. 找到离原点最近的面
3. 沿该面的法线方向查询新支撑点
4. 将新点加入多面体，移除被"看到"的旧面
5. 重复直到收敛

**输出**：穿透深度 $\delta$ 和碰撞法线 $\mathbf{n}$

> 参考：[Wikipedia - Expanding Polytope Algorithm](https://en.wikipedia.org/wiki/Expanding_polytope_algorithm)

---

## 11. 最新进展：GJK++（2024）

Louis Montaut 等人在 2024 年提出 **GJK++**，将凸优化中的 **Polyak 动量** 和 **Nesterov 加速** 引入 GJK 迭代：

### 11.1 核心思想

标准 GJK 的搜索方向更新为：

$$\mathbf{d}_{k+1} = -\text{closest}_k$$

GJK++ 引入动量项：

$$\mathbf{d}_{k+1} = -\text{closest}_k + \beta_k \cdot (\mathbf{d}_k - \mathbf{d}_{k-1})$$

其中 $\beta_k$ 是动量系数，借鉴了 Nesterov 加速梯度法的收敛率理论。

### 11.2 性能提升

| 场景 | 标准 GJK 迭代次数 | GJK++ 迭代次数 | 加速比 |
|------|-------------------|----------------|--------|
| 近距圆形 | 3-4 | 2-3 | ~1.5× |
| 复杂多面体 | 6-10 | 3-5 | ~2× |
| 连续碰撞查询 | 8-15 | 4-8 | ~2× |

> 参考：[Montaut et al., 2024, "GJK++: Accelerated GJK"](https://arxiv.org/abs/2403.07649)

---

## 12. 代码实现要点（C++ 伪代码）

```cpp
struct Simplex {
    Vec3 points[4];
    int count; // 1-4
};

float gjk_distance(const Shape& A, const Shape& B) {
    Vec3 d = normalize(A.center() - B.center()); // 初始方向
    Simplex S = {};
    
    // 初始支撑点
    Vec3 p = support(A, d) - support(B, -d);
    S = {p};  // 0-simplex
    
    if (length(p) < EPSILON) return 0.0f;
    d = -p;
    
    for (int iter = 0; iter < MAX_ITER; iter++) {
        p = support(A, d) - support(B, -d);
        
        // 分离测试
        if (dot(p, d) <= 0.0f) {
            return length(closest_point_on_simplex(S));
        }
        
        // 扩展
        S = add_point(S, p);
        
        // 缩减 + 最近点
        Vec3 closest;
        S = reduce_simplex(S, closest);  // Johnson's sub-algorithm
        d = -closest;
        
        if (length(d) < EPSILON) return 0.0f;
    }
    return length(closest_point_on_simplex(S));
}
```

---

## 13. 应用场景总览

| 领域 | 应用 | 框架/库 |
|------|------|---------|
| **游戏物理** | 碰撞检测、接触点计算 | Bullet Physics, Box2D, PhysX |
| **机器人** | 路径规划、障碍物规避 | OMPL, FCL |
| **CAD** | 实体干涉检查 | OpenGJK |
| **分子模拟** | 分子间距离计算 | HOOMD-blue |
| **离散元法 (DEM)** | 颗粒碰撞检测 | GPU-accelerated GJK |

---

## 14. 直觉总结：GJK 的本质

把 GJK 理解为 **"在一个越来越精确的笼子里找原点"**：

1. **问题转化**：两集距离 → 原点到 Minkowski difference 的距离（单集问题）
2. **迭代逼近**：每次沿"朝原点方向"在 Minkowski difference 上取一个新点，扩大"笼子"
3. **精简笼子**：每次只保留与原点最近的特征（点/边/面/体），丢弃多余点
4. **终止判定**：
   - 原点进了笼子 → 碰撞！
   - 笼子不增长了 → 分离，当前距离就是答案

这个框架的美妙之处在于：**你永远不需要真正构造 Minkowski difference**——支撑函数的线性性质让你只需查询两个原始集的边界点，就足以迭代逼近答案。

---

## 参考文献

1. Gilbert, E.G., Johnson, D.W., & Keerthi, S.S. (1988). *A fast procedure for computing the distance between complex shapes in three-dimensional space*. IEEE Journal of Robotics and Automation, 4(2), 193-203. [DOI: 10.1109/70.1203](https://doi.org/10.1109/70.1203)
2. Ericson, C. (2004). *Real-Time Collision Detection*. Morgan Kaufmann. [Link](https://www.sciencedirect.com/book/9781558607323)
3. Gilbert & Johnson (1985). *Distance functions and their application to robot path planning*. [DOI: 10.1109/JRA.1985.1087007](https://doi.org/10.1109/JRA.1985.1087007)
4. Ong & Gilbert (1997). *GJK with incremental motions*. [DOI: 10.1109/70.580932](https://doi.org/10.1109/70.580932)
5. Montanari, Petrinic, & Barbieri (2017). *Improved GJK with signed volume*. [DOI: 10.1016/j.compgeom.2017.06.002](https://doi.org/10.1016/j.compgeom.2017.06.002)
6. OpenGJK: [https://github.com/MattiaMontanari/openGJK](https://github.com/MattiaMontanari/openGJK)
7. Montaut et al. (2024). *GJK++: Accelerated GJK*. [arXiv:2403.07649](https://arxiv.org/abs/2403.07649)
8. Wikipedia - GJK Algorithm: [https://en.wikipedia.org/wiki/Gilbert%E2%80%93Johnson%E2%80%93Keerthi_distance_algorithm](https://en.wikipedia.org/wiki/Gilbert%E2%80%93Johnson%E2%80%93Keerthi_distance_algorithm)
9. Wikipedia - Convex Set: [https://en.wikipedia.org/wiki/Convex_set](https://en.wikipedia.org/wiki/Convex_set)
10. Wikipedia - Expanding Polytope Algorithm: [https://en.wikipedia.org/wiki/Expanding_polytope_algorithm](https://en.wikipedia.org/wiki/Expanding_polytope_algorithm)