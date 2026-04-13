**当你面对一个数学/数值计算函数时，如何在"可能不知道正确输出"的情况下，依然写出有意义的单元测试？** 作者以 `ecef_to_lla`（ECEF笛卡尔坐标 → 大地经纬高坐标）为例，层层递进地提出了 **四个测试层级**。

## 一、问题背景：为什么数值例程的测试难？

### 1.1 ECEF 与 LLA 坐标系基础

**ECEF (Earth-Centered, Earth-Fixed)** 是一个固连在地球上的笛卡尔坐标系：
- 原点 $O$ 在地球质心
- $x$ 轴指向赤道与本初子午线的交点
- $z$ 轴指向北极（沿地球自转轴）
- $y$ 轴与 $x, z$ 构成右手系

**LLA (Latitude, Longitude, Altitude)** 是大地坐标系：
- $\phi$ (latitude)：纬度，$[-90°, 90°]$
- $\lambda$ (longitude)：经度，$[-180°, 180°]$
- $h$ (altitude/height)：椭球高度，沿椭球法线方向

### 1.2 核心数学：ECEF → LLA 变换

这**不是**简单的球面坐标变换！地球不是球体，而是一个**旋转椭球体** (oblate ellipsoid)，WGS84定义的参数为：

$$
\begin{aligned}
a &= 6378137.0 \text{ m} \quad &\text{(semi-major axis, 赤道半径)} \\
f &= 1/298.257223563 \quad &\text{(flattening, 扁率)} \\
b &= a(1-f) \approx 6356752.314 \text{ m} \quad &\text{(semi-minor axis, 极半径)} \\
e^2 &= 2f - f^2 \approx 6.694 \times 10^{-3} \quad &\text{(first eccentricity squared, 第一偏心率的平方)}
\end{aligned}
$$

其中偏心率 $e$ 描述椭球偏离球体的程度：

$$
e^2 = \frac{a^2 - b^2}{a^2}
$$

ECEF到LLA的变换涉及**迭代求解**（因为 $\phi$ 和 $h$ 之间互为隐函数），常用算法有 **Bowring's method**、**Zhu's method**、**Ferrari's solution** 等。正因为算法复杂且可能迭代，测试才变得尤其重要——你无法"一眼看出"结果对不对。

> 参考算法论文：[Zhu, J. (1993). "Exact Transformation of Geocentric to Geodetic Coordinates..."](https://doi.org/10.1007/s001900050034)

---

## 二、四层测试策略详解

### Level 1: Magic Numbers（魔法数字）

**做法**：从外部可信来源获取已知输入-输出对，硬编码到测试中。

```cpp
const Eigen::Vector3d pos{1318610., -4504041., 4306196.};
const auto [lat, lon, alt] = ecef_to_lla(pos);
REQUIRE_THAT(lat, WithinAbs(42.730, 0.001));
```

**优点**：
- 简单直接，零学习成本
- 如果数据源可信且稳定，能快速验证

**致命缺陷**：
1. **数据溯源问题**：你用的哪个版本的脚本？`c_ECEF2geo_final_v2.m` 还是 `v3`？脚本本身的bug会被复制到测试中
2. **脆弱性**：算法换了实现（比如从迭代法换成闭式解），精度特征可能变化，但magic numbers是写死的
3. **反向验证陷阱**：最糟糕的做法是——先运行代码，把输出复制到assert里，测试"永远通过"——但测的是**错误行为**

**第一性原理反思**：Magic numbers 本质是在做 **regression testing**（回归测试），而不是 **correctness testing**（正确性测试）。它告诉你"输出没变"，但不告诉你"输出是对的"。

---

### Level 2: Trivial Points（平凡点）

**做法**：寻找输出可由定义**直接推导**的特殊输入点。

文章给出最经典的例子——**$x$ 轴与椭球面的交点**：

$$
\text{输入: } \vec{P} = (a, 0, 0)
$$

由定义：
- 此点在赤道面上 → $\phi = 0°$
- 此点在本初子午线上 → $\lambda = 0°$
- 此点在椭球面上 → $h = 0$

因此：
$$
ecef\_to\_lla(a, 0, 0) = (0°, 0°, 0\text{ m})
$$

**更多平凡点的推导**：

| 输入点 | ECEF坐标 | 理论LLA | 推导依据 |
|--------|----------|---------|----------|
| 北极点 | $(0, 0, b)$ | $(90°, \text{undefined}, 0)$ | $z$轴正方向=北极，在椭球面上 |
| 南极点 | $(0, 0, -b)$ | $(-90°, \text{undefined}, 0)$ | 同上 |
| $y$轴交点 | $(0, a, 0)$ | $(0°, 90°, 0)$ | 赤道+东经90° |
| 北极上方1000m | $(0, 0, b+1000)$ | $(90°, \text{undefined}, 1000)$ | 沿法线（此时=z轴）升高 |
| $x$轴上方1000m | $(a+1000, 0, 0)$ | $(0°, 0°, 1000)$ | ⚠️ 这其实**不对**！|

**⚠️ 关键陷阱**：$(a+1000, 0, 0)$ 的椭球高度**不是** 1000m！因为椭球法线方向不沿径向（除赤道和极点外）。在赤道处恰好法线沿径向，所以 $(a+1000, 0, 0)$ 确实是 $h=1000$，但在中纬度就不是了。

椭球面上点的法线方向与径向方向的偏差角称为 **垂线偏差** (deflection of the vertical)，这是大地测量的核心概念之一。

**第一性原理反思**：Trivial points 利用了系统的 **对称性**（symmetry）和 **边界条件**（boundary conditions）。这是物理学家思考的方式——在最简单的构型下验证理论。

---

### Level 3: Property-Based Testing（基于性质的测试）

**核心思想**：不检查具体输出值，而是检查输出必须满足的 **数学/物理性质**。

#### 性质1：纬度符号与Z坐标符号一致

$$
\text{sign}(\phi) = \text{sign}(P_z)
$$

这是显然的：$z > 0$ → 北半球 → $\phi > 0$，反之亦然。

```cpp
REQUIRE(std::signbit(lat) == std::signbit(z));
```

#### 性质2：Round-trip（往返一致性）

这是**最强大**的性质之一：

$$
ecef\_to\_lla \circ lla\_to\_ecef = \text{id}
$$

即：
$$
\vec{P}_{ECEF} \xrightarrow{ecef\_to\_lla} (\phi, \lambda, h) \xrightarrow{lla\_to\_ecef} \vec{P}'_{ECEF} \implies \vec{P}' = \vec{P}
$$

代码：
```cpp
const auto [lat, lon, alt] = ecef_to_lla(pos);
const Eigen::Vector3d round_trip = lla_to_ecef(lat, lon, alt);
REQUIRE_THAT(round_trip.x(), WithinAbs(pos.x(), 1e-6));
REQUIRE_THAT(round_trip.y(), WithinAbs(pos.y(), 1e-6));
REQUIRE_THAT(round_trip.z(), WithinAbs(pos.z(), 1e-6));
```

#### 更多可用性质

| 性质 | 数学表达 | 物理含义 |
|------|----------|----------|
| 纬度范围 | $\|\phi\| \leq 90°$ | 地球表面纬度不会超过极点 |
| 经度范围 | $\|\lambda\| \leq 180°$ | 经度的周期性 |
| 高度连续性 | 附近点的高度差很小 | 地球表面光滑 |
| 等高面性质 | 同一 $\phi, \lambda$，$h$ 增加 $\Delta h$，ECEF沿法线方向移动 $\Delta h$ | 法线定义 |
| 赤道对称 | $ecef\_to\_lla(x, y, z)_\phi = -ecef\_to\_lla(x, y, -z)_\phi$ | 南北半球对称 |

**关于 Catch2 的 GENERATE 机制**：

`GENERATE(take(N, random(min, max)))` 的组合语义：
- `random(min, max)`：一个**无限生成器**，持续产出 $[min, max]$ 内的伪随机数
- `take(N, gen)`：截取前 $N$ 个值，变成有限生成器
- `GENERATE(...)`：为每个生成的值创建一个独立的测试用例

这意味着如果10个随机值中有一个失败，你能精确知道是哪一个，而不是"10次中有1次失败"这种模糊信息。

**第一性原理反思**：Property-based testing 本质上在验证 **不变量**（invariants）——那些无论输入如何都成立的数学真理。这与形式化验证（formal verification）的思想一致，只是退化为统计采样。

> 相关工具：[Haskell QuickCheck](https://hackage.haskell.org/package/QuickCheck)（原创）、[Python Hypothesis](https://hypothesis.readthedocs.io/)、[Rust Proptest](https://github.com/proptest-rs/proptest)

---

### Level 4: Checking Against a Specification（对照规范验证）

**最高可信度**的测试方式：对照**权威标准文档**中的参考数据。

WGS84 规范 **NGA.STND.0036_1.0.0_WGS84**（前身为 NIMA TR8350.2）中列出了多个**参考监测站**的精确坐标，同时给出了 ECEF 和 LLA 值。

文章中使用的6个监测站：

| 站名 | ECEF X (m) | ECEF Y (m) | ECEF Z (m) | Lat (°) | Lon (°E) | Alt (m) |
|------|-----------|-----------|-----------|---------|----------|---------|
| Colorado Springs | -1248599.695 | -4819441.002 | 3976490.117 | 38.80293817 | 255.47540411 | 1911.778 |
| Ascension | 6118523.866 | -1572350.772 | -876463.909 | -7.95132931 | 345.58786964 | 106.281 |
| Diego Garcia | 1916196.855 | 6029998.797 | -801737.183 | -7.26984216 | 72.37092367 | -64.371 |
| Kwajalein | -6160884.028 | 1339852.169 | 960843.154 | 8.72250188 | 167.73052378 | 39.652 |
| Hawaii | -5511980.264 | -2200246.752 | 2329481.004 | 21.56149239 | 201.76066695 | 425.789 |
| Cape Canaveral | 918988.062 | -5534552.894 | 3023721.362 | 28.48373823 | 279.42769502 | -24.083 |

**关键技术细节——经度约定**：

NGA 规范使用 **东经** (Longitude East)，范围 $[0°, 360°)$。例如 Colorado Springs 的经度是 255.475°E，即：

$$
\lambda = 255.475° - 360° = -104.525° \quad (\text{西经})
$$

代码中的处理：
```cpp
const auto lon_spec_wrapped = lon_spec > 180 ? lon_spec - 360 : lon_spec;
```

**容差选择的意义**：
- 纬度/经度：$1 \times 10^{-6}$ 度 ≈ 0.11 mm（在赤道），这是亚毫米级精度
- 高度：0.1 m — 注意高度精度要求远低于角度，这是因为 **椭球高度的计算对输入误差更敏感**

关于高度误差的敏感性，可以用误差传播公式理解。LLA到ECEF的变换中，高度的偏导数为：

$$
\frac{\partial z}{\partial h} = \sin\phi \cdot N(\phi)
$$

其中 $N(\phi)$ 是 **卯酉圈曲率半径** (radius of curvature in the prime vertical)：

$$
N(\phi) = \frac{a}{\sqrt{1 - e^2 \sin^2\phi}}
$$

> WGS84规范文档：[NGA.STND.0036](https://earth-info.nga.mil/index.php?dir=wgs84&action=wgs84)

---

## 三、四层策略的哲学关系

从第一性原理出发，这四层构成了一个 **可信度金字塔**：

```
        Level 4: Specification
       /                      \
      /    Level 3: Properties  \
     /                            \
    /   Level 2: Trivial Points     \
   /                                  \
  /     Level 1: Magic Numbers          \
 /________________________________________\
```

| 层级 | 验证的是什么 | 信任基础 | 覆盖度 | 维护成本 |
|------|------------|---------|--------|---------|
| L1 | 某个特定输入的输出 | 外部数据源 | 极低 | 低（但脆弱） |
| L2 | 边界/对称点的正确性 | 数学定义 | 低 | 低 |
| L3 | 不变量的普遍成立 | 数学/物理定律 | 高（统计意义上） | 中 |
| L4 | 对权威标准的符合性 | 规范文档 | 中（取决于参考数据量） | 中 |

**关键洞察**：每一层都回答不同的问题——

- L1 回答："这个特定case对了吗？"
- L2 回答："在边界条件下行为合理吗？"
- L3 回答："在任意输入下性质成立吗？"
- L4 回答："与权威标准一致吗？"

**它们是互补的，不是互斥的。**

---

## 四、延伸思考与扩展

### 4.1 LLA → ECEF 的精确公式

正向变换（LLA → ECEF）是闭式解：

$$
\begin{aligned}
N(\phi) &= \frac{a}{\sqrt{1 - e^2 \sin^2\phi}} \\
x &= (N(\phi) + h) \cos\phi \cos\lambda \\
y &= (N(\phi) + h) \cos\phi \sin\lambda \\
z &= \left(N(\phi)(1-e^2) + h\right) \sin\phi
\end{aligned}
$$

其中：
- $N(\phi)$：卯酉圈曲率半径，表示椭球面在纬度 $\phi$ 处沿东西方向的曲率半径
- $e^2$：第一偏心率的平方
- $h$：椭球高度（沿法线，非径向！）

**注意**：$z$ 的公式中因子是 $(1-e^2)$ 而不是 $(1-e)$，这是因为极半径 $b = a\sqrt{1-e^2}$。

### 4.2 ECEF → LLA 的迭代算法（Bowring）

反向变换没有闭式解（虽然Ferrari给出了闭式近似），常用迭代法：

$$
\begin{aligned}
\phi_0 &= \arctan\left(\frac{z}{p(1-e^2)}\right) \\
\phi_{i+1} &= \arctan\left(\frac{z + e^2 N(\phi_i) \sin\phi_i}{p}\right)
\end{aligned}
$$

其中 $p = \sqrt{x^2 + y^2}$（到z轴的距离）。

通常2-3次迭代即达到 $10^{-10}$ 弧度精度。

### 4.3 性质测试在更广泛数值计算中的应用

这套方法论远不止坐标变换：

| 领域 | 函数 | 可用性质 |
|------|------|---------|
| 线性代数 | 矩阵求逆 | $A \cdot A^{-1} = I$ |
| 优化 | 梯度下降 | 损失单调递减 |
| 插值 | 样条函数 | 过控制点、二阶导连续 |
| FFT | 傅里叶变换 | Parseval定理 $\|f\|_2 = \|\hat{f}\|_2$ |
| 统计 | PDF积分 | $\int_{-\infty}^{\infty} f(x)dx = 1$ |
| 微分方程 | ODE求解器 | 对称性守恒、能量守恒（Hamilton系统） |
| 三角函数 | $\sin, \cos$ | $\sin^2 + \cos^2 = 1$, 周期性, 奇偶性 |

### 4.4 浮点精度与容差选择的艺术

文章中不同层级用了不同容差，这不是随意的：

| 测试层级 | 角度容差 | 高度容差 | 原因 |
|---------|---------|---------|------|
| L1 Magic | 0.001° | 1 m | 外部工具有限精度 |
| L2 Trivial | 0.001° | 1.0 m | 定义精确，误差仅来自浮点 |
| L3 Property | N/A | N/A | 检查性质，非具体值 |
| L4 Spec | $10^{-6}$° | 0.1 m | 参考站数据高精度 |

**0.001° 对应的距离**：

$$
\Delta s_{\text{lat}} = R \cdot \Delta\phi = 6371\text{km} \times 0.001° \times \frac{\pi}{180} \approx 111\text{m}
$$

所以 0.001° 的角度容差 ≈ 111m 的位置容差，对于Level 1测试来说是比较宽松的。

而 Level 4 的 $10^{-6}$°：

$$
\Delta s = 6371\text{km} \times 10^{-6}° \times \frac{\pi}{180} \approx 0.11\text{m}
$$

即亚毫米级，这已经接近 WGS84 参考站数据的内在精度了。

### 4.5 CI/CD 中的数值测试策略

文章最后提到将 V&V（Verification & Validation）尽可能多地自动化为单元测试并集成到 CI/CD 中，这是现代工程的关键实践：

- **Verification**：代码是否正确实现了算法？（代码层面）
- **Validation**：算法的输出是否反映了真实世界？（需求层面）

Property-based tests 特别适合 CI/CD，因为：
1. 每次运行可能生成不同的随机种子，增加覆盖
2. 不依赖外部数据，不会因数据源变化而中断
3. 性质是永恒的数学真理，不存在"版本"问题

但要注意设置 **随机种子**（seed）以保证可重现性——这在调试时至关重要。

---

## 五、总结

这篇文章的核心洞见可以凝练为一句话：

> **测试数值例程的正确姿势，是从"验证特定输出"升级到"验证普遍性质"。**

四层策略的本质是从 **经验信任** → **定义信任** → **逻辑信任** → **权威信任** 的递进。每一层都有不可替代的价值，真正的工程实践应该**四层并用**，形成纵深防御（defense in depth）。

这种思维方式完全可以用第一性原理来理解：**数学系统的真理分为两类——偶然真理和必然真理。** Level 1 验证的是偶然真理（碰巧这个输入对应那个输出），Level 2-3 验证的是必然真理（由定义和定律决定），Level 4 验证的是社会约定的真理（规范说是什么就是什么）。优秀的测试策略应该尽可能多地覆盖必然真理。

---

### 参考资源

- [WGS84 规范 - NGA](https://earth-info.nga.mil/index.php?dir=wgs84&action=wgs84)
- [Catch2 文档 - GENERATE](https://catch2.temporal-mirage.com/doc/generators.html)
- [ECEF-LLA 在线转换工具](https://www.oc.nps.edu/oc2902w/coord/llhxyz.htm)
- [Zhu's closed-form algorithm](https://doi.org/10.1007/s001900050034)
- [Haskell QuickCheck - 性质测试的起源](https://hackage.haskell.org/package/QuickCheck)
- [Python Hypothesis - Python生态的性质测试](https://hypothesis.readthedocs.io/)
- [Borkowski's closed-form ECEF-to-geodetic](https://doi.org/10.1007/s00190-007-0212-3)