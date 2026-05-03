[https://zone.dog/braindump/spline_fields/](https://zone.dog/braindump/spline_fields/)

这是一篇由 **Aeva Palecek** 于 2024年12月31日 发布的技术博客，介绍了一种名为 **"Spline Distance Fields"（样条距离场）** 的创新地形生成技术。该技术用于作者正在开发的 CPU 光线追踪渲染器 **Star Machine** 和赛车游戏 **Rainy Road**。

---

## 🎯 问题背景

### 项目需求
作者需要一个地形渲染系统，满足以下约束：

| 需求 | 说明 |
|------|------|
| Spline-based | 道路等地形特征必须由样条曲线定义 |
| Composable | 地形特征可组合，支持运行时快速切换 |
| Procedural-friendly | 便于程序化物体放置 |
| Fast processing | 快速处理为渲染中间产物 |
| Compact data | 数据紧凑，偏好隐式表示 |
| Rapid iteration | 快速迭代设计 |
| Open tools | 不使用专有工具 |

### 现有工具的局限性
作者调研后发现几乎所有通用地形编辑器都至少违反其中一个约束。

---

## 💡 核心思想：Spline Distance Fields

### 第一性原理

> **"Everything Affects Everything"** — 设计与构图的基本法则

在现实世界中：
- 景观影响道路的选址决策
- 道路反过来改变景观

**关键洞察**：如果我们知道某些东西**必须在哪里**，就可以推断**周围是什么**。可以完全抛弃输入高度图，仅从**点和样条曲线**生成地形（甚至可以实时生成）。

### 技术本质

这个方法本质上是 **Voronoi Diagram（沃罗诺伊图）** 的变体：

```
对于空间中任意点 P:
    1. 找到最近的样条曲线 S
    2. 在 S 上找到最近的点 C (closest point)
    3. C 点的 binormal 向量 B 定义一个平面
    4. 该平面决定 P 点的局部高度
```

---

## 📐 数学原理详解

### Frenet-Serret Frame（弗雷奈-塞雷坐标系）

在曲线的每个点上定义一个局部坐标系：

```
T = Tangent Vector (切向量)
N = Normal Vector (法向量)  
B = Binormal Vector (副法向量) = T × N
```

#### 各向量含义：

| 向量 | 定义 | 在本技术中的作用 |
|------|------|------------------|
| **T (Tangent)** | 曲线的切线方向单位向量 | 位于我们要定义的平面上 |
| **N (Normal)** | 垂直于切向量，指向曲率中心方向 | 位于我们要定义的平面上 |
| **B (Binormal)** | **B = N × T** (叉积) | **这是我们要的"平面法向量"！** |

### 叉积公式

$$\vec{B} = \vec{N} \times \vec{T} = \begin{pmatrix} N_y T_z - N_z T_y \\ N_z T_x - N_x T_z \\ N_x T_y - N_y T_x \end{pmatrix}$$

其中：
- $\vec{N} = (N_x, N_y, N_z)$ 是法向量
- $\vec{T} = (T_x, T_y, T_z)$ 是切向量
- $\vec{B}$ 是副法向量，长度为1（因为 N 和 T 都是单位向量且正交）

### 平面方程

给定副法向量 $\vec{B}$ 和曲线上一点 $\vec{C}$，定义的平面方程为：

$$\vec{B} \cdot (\vec{P} - \vec{C}) = 0$$

或展开为：

$$B_x(P_x - C_x) + B_y(P_y - C_y) + B_z(P_z - C_z) = 0$$

### 有符号距离函数

对于任意点 $\vec{P}$ 到该平面的有符号距离：

$$d(\vec{P}) = \vec{B} \cdot (\vec{P} - \vec{C}) = B_x(P_x - C_x) + B_y(P_y - C_y) + B_z(P_z - C_z)$$

**物理意义**：
- $d > 0$：点在平面"上方"
- $d < 0$：点在平面"下方"
- $d = 0$：点在平面上

---

## 🛠️ 实现架构

### 总体流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                        Blender Geometry Nodes                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐      ┌──────────────┐      ┌──────────────────┐  │
│  │  Splines │ ───▶ │  Surfel      │ ───▶ │  Distance Field  │  │
│  │ Collection│     │  Generation  │      │  Calculation     │  │
│  └──────────┘      └──────────────┘      └──────────────────┘  │
│                            │                      │              │
│                            ▼                      ▼              │
│                     ┌──────────────┐      ┌──────────────┐      │
│                     │ Position     │      │ Grid Mesh    │      │
│                     │ Binormal     │      │ Deformation  │      │
│                     └──────────────┘      └──────────────┘      │
│                                                  │               │
│                                                  ▼               │
│                                           ┌──────────────┐      │
│                                           │  Heightmap   │      │
│                                           │  (Terrain)   │      │
│                                           └──────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

### 核心步骤

#### Step 1: Curves to Surfels（曲线转表面元素）

**Surfel 定义**：
```
Surfel = {
    position: vec3,    // 样条曲线上的点
    binormal: vec3     // 该点的副法向量
}
```

操作：
1. 将曲线采样为离散点
2. 计算每个点的 T, N 向量
3. 叉积得到 B 向量
4. 投影到 XY 平面（保留原始 Z 值）

#### Step 2: Surfel Sampling（Surfel 采样）

```python
# 伪代码
for each vertex V in Grid:
    # 找最近的 Surfel
    nearest_index = SampleNearest(V.position, surfels)
    nearest_surfel = surfels[nearest_index]
    
    # 获取参数
    C = nearest_surfel.position  # 最近点
    B = nearest_surfel.binormal  # 副法向量
    
    # 计算有符号距离
    d = dot(B, V.position - C)
```

#### Step 3: Heightmap Generation（高度图生成）

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Distance   │ ──▶ │    Blur     │ ──▶ │  Z Offset   │
│  Field      │     │  (多次迭代)  │     │  Applied    │
└─────────────┘     └─────────────┘     └─────────────┘
```

**Blur 的作用**：解决"Liminality Problem"（过渡区间问题），即不同样条曲线之间的边界区域可能出现的高度突变。

---

## 🔬 深入技术细节

### Liminality Problem（过渡区间问题）

当采样点位于两条或多条样条曲线的"中间区域"时：

```
        Spline A                    Spline B
           │                           │
           │    ←── Liminal Zone ──→   │
           │        (过渡区间)          │
```

问题表现：
- 高度突变
- 不自然的过渡
- 类似地质学中的 **Subduction/Obsuction**（俯冲/仰冲）现象

解决方案：
1. **调整网格密度**：较低的密度会产生更平滑的过渡
2. **Blur 迭代**：模糊距离场值
3. **Inverse Distance Weighting**（逆距离加权，高级变体）

### Inverse Distance Weighting (IDW) 变体

由 **Danpiker** 提出的改进方案：

$$h(\vec{P}) = \frac{\sum_{i=1}^{n} w_i(\vec{P}) \cdot h_i(\vec{P})}{\sum_{i=1}^{n} w_i(\vec{P})}$$

其中权重函数：

$$w_i(\vec{P}) = \frac{1}{d(\vec{P}, S_i)^p}$$

- $d(\vec{P}, S_i)$：点 $\vec{P}$ 到第 $i$ 条样条曲线的距离
- $p$：幂指数（控制影响衰减速度，通常 $p = 2$）

**优点**：
- 完美解决 liminality 问题
- 产生非常平滑的过渡效果

**缺点**：
- 计算成本较高（需要考虑多条样条曲线）
- 可能过度平滑，失去"earthiness"（土地质感）

---

## 🎮 实际应用案例

### 1. Rainy Road（赛车游戏）
- 作者自己的项目
- 用于程序化地形生成

### 2. Bitmap 的游戏项目
- 完全抛弃 Godot 内置地形工具
- 改用基于 Spline Distance Fields 的新工作流
- 对于 2D 插画师背景的艺术家特别友好

### 3. Sculptor（Godot 非破坏性建模工具）
- 开发者：Dbat
- 扩展版本：支持自由形态 3D 网格（非高度图）
- 使用光线追踪迭代变形网格

---

## 📊 与相关技术的对比

| 技术 | 数据表示 | 优势 | 劣势 |
|------|----------|------|------|
| **Heightmap** | 2D 图像 | 简单、快速 | 无法表示洞穴/悬垂 |
| **Voxel** | 3D 体素 | 任意拓扑 | 内存消耗大 |
| **SDF (Signed Distance Field)** | 隐式函数 | 精确、紧凑 | 计算复杂 |
| **Spline Distance Fields** | 样条曲线 | 直观编辑、紧凑 | 需要特殊处理过渡区 |

---

## 🔗 相关数学领域

作者在开发此技术后才发现已有相关数学研究：

1. **Medial Axis Transform（中轴变换）**
2. **Voronoi Diagrams on Curves（曲线上的沃罗诺伊图）**
3. **Implicit Surface Modeling（隐式曲面建模）**
4. **Frenet-Serret Formulas（弗雷奈-塞雷公式）**

---

## 📚 参考资源

### 博客原文
- [Spline Distance Fields](https://www.aevapalecek.net/posts/spline-distance-fields/) - Aeva Palecek 的博客

### 相关数学
- [Frenet-Serret Formulas - Wikipedia](https://en.wikipedia.org/wiki/Frenet%E2%80%93Serret_formulas)
- [Voronoi Diagram - Wikipedia](https://en.wikipedia.org/wiki/Voronoi_diagram)
- [Inverse Distance Weighting - Wikipedia](https://en.wikipedia.org/wiki/Inverse_distance_weighting)

### 工具与项目
- [Blender Geometry Nodes](https://docs.blender.org/manual/en/latest/modeling/geometry_nodes/index.html)
- [Godot Engine](https://godotengine.org/)

### 相关技术
- [Signed Distance Fields in Computer Graphics](https://iquilezles.org/articles/distfunctions/)
- [Procedural Terrain Generation](https://www.informit.com/articles/article.aspx?p=2130322)

---

## 💭 总结

这篇博客展示了一个**非常实用且创新**的地形生成技术。其核心价值在于：

1. **第一性原理思维**：从"道路影响地形，地形影响道路"的现实观察出发
2. **隐式表示的优雅**：仅用曲线定义复杂地形
3. **实用主义**：不追求完美数学，而是解决实际问题
4. **开放分享**：提供完整的 Blender 源文件

这是一个典型的**艺术家友好**的程序化生成技术，特别适合那些不擅长传统3D雕塑但擅长2D绘制的创作者。正如作者朋友 Bitmap 所说——现在她可以"直接画出地形"了！