# 这篇 Blog 的深度解析：Approximate First Principal Component 的简化算法

## 📌 一句话总结

这篇 blog 介绍了一种**无需计算 covariance matrix 和 eigenvector** 的近似方法来估计数据集的第一主成分方向，来源于 Dennis Ranke 的 exoquant-rs 颜色量化库，本质上是一个"向量翻转累加归一化"的 trick，计算量远低于完整 PCA，但在某些数据分布下会失效。

---

## 1. 背景：PCA 与颜色量化

### 1.1 第一主成分

给定一组 $\mathbb{R}^d$ 中的点 $\{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n\}$，第一主成分是使得数据投影后方差最大的方向。形式化：

$$\mathbf{v}_1 = \arg\max_{\|\mathbf{v}\|=1} \sum_{i=1}^{n} \left( \mathbf{v}^\top (\mathbf{x}_i - \bar{\mathbf{x}}) \right)^2$$

其中：
- $\mathbf{v}_1 \in \mathbb{R}^d$ 是第一主成分方向（单位向量）
- $\bar{\mathbf{x}} = \frac{1}{n}\sum_{i=1}^{n} \mathbf{x}_i$ 是均值向量
- 目标函数就是投影后的方差（未除以 $n$ 的版本）

等价地，$\mathbf{v}_1$ 是 covariance matrix $\mathbf{C} = \frac{1}{n}\sum_{i=1}^{n}(\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^\top$ 的最大特征值对应的特征向量。

### 1.2 颜色量化

把像素颜色看作 RGB 立方体中的 3D 点，颜色量化就是把这个 3D 空间分割成互斥子空间，每个子空间代表一个调色板颜色。层级式自顶向下方法（如 **median cut**）递归地用一个平面将空间一分为二，类似 BSP-tree。

### 1.3 三种精度等级

| 方法 | 计算量 | 精度 |
|------|--------|------|
| Axis-aligned（按最大 variance 的坐标轴切分） | 低 | 粗糙 |
| **Approximate PC（本文方法）** | **中** | **中等** |
| Full PCA（计算 covariance matrix + eigenvector） | 高 | 精确 |

---

## 2. 核心算法详解

### 2.1 原始 Rust 代码逻辑

```rust
let mut dir = Colorf::zero();           // 累加器，初始为零向量
for entry in &histogram {
    let mut tmp = (entry.color - avg) * entry.count as f64;  // 偏移向量 × 权重
    if tmp.dot(&dir) < 0.0 {            // 如果与累加器方向相反
        tmp *= -1.0;                     // 翻转！
    }
    dir += tmp;                          // 累加
}
dir = dir / dir.dot(&dir).sqrt();        // 归一化
```

### 2.2 逐步拆解

**Step 1: 排序** — 数据按最大 variance 的坐标轴排序

$$\text{sort\_axis} = \arg\max_{j \in \{0,1,2\}} \text{Var}(\{x_{1j}, x_{2j}, \ldots, x_{nj}\})$$

这使得累加过程是确定性的（deterministic），也可能帮助估计质量。

**Step 2: 计算均值**

$$\bar{\mathbf{x}} = \frac{1}{n}\sum_{i=1}^{n} \mathbf{x}_i$$

**Step 3: 逐点累加（核心！）**

对每个点 $\mathbf{x}_i$：

$$\boldsymbol{\delta}_i = \mathbf{x}_i - \bar{\mathbf{x}}$$

$$\boldsymbol{\delta}_i' = \begin{cases} \boldsymbol{\delta}_i & \text{if } \boldsymbol{\delta}_i \cdot \mathbf{d}_{acc} \geq 0 \\ -\boldsymbol{\delta}_i & \text{if } \boldsymbol{\delta}_i \cdot \mathbf{d}_{acc} < 0 \end{cases}$$

$$\mathbf{d}_{acc} \leftarrow \mathbf{d}_{acc} + \boldsymbol{\delta}_i'$$

其中 $\mathbf{d}_{acc}$ 是累加器方向向量，初始为 $\mathbf{0}$。

**Step 4: 归一化**

$$\hat{\mathbf{d}} = \frac{\mathbf{d}_{acc}}{\|\mathbf{d}_{acc}\|_2}$$

### 2.3 直觉：为什么这样能近似第一主成分？

**第一性原理推导：**

PCA 的目标是找到使投影方差最大的方向。等价地说，我们希望所有偏移向量 $(\mathbf{x}_i - \bar{\mathbf{x}})$ 在该方向上的投影尽量大。

如果我们简单地把所有偏移向量加起来：

$$\mathbf{s} = \sum_{i=1}^{n} (\mathbf{x}_i - \bar{\mathbf{x}}) = \mathbf{0}$$

**结果为零！** 因为均值点定义上使得偏移之和为零。

**关键 insight：翻转操作。** 如果我们在累加时，把与当前累加方向相反的偏移向量翻转过来，那么：

- 沿着主成分方向一致的偏移 → 直接累加（**正反馈**）
- 与主成分方向相反的偏移 → 翻转后也沿主方向累加（**正反馈**）
- 沿着次成分方向的偏移 → 翻转与否大致随机，累加后**互相抵消**

这就像一个**投票机制**：每个偏移向量都在"投票"决定主方向，但只投给当前共识方向的正半边。最终，主成分方向上的投票会系统性地累积，而其他方向上的投票会随机化地抵消。

**数学上的直觉联系：**

这与 **power iteration** 有微妙的联系。Power iteration 求最大特征向量的迭代是：

$$\mathbf{v}^{(k+1)} = \frac{\mathbf{C} \cdot \mathbf{v}^{(k)}}{\|\mathbf{C} \cdot \mathbf{v}^{(k)}\|}$$

展开 $\mathbf{C} \cdot \mathbf{v}$：

$$\mathbf{C} \cdot \mathbf{v} = \frac{1}{n}\sum_{i=1}^{n} (\mathbf{x}_i - \bar{\mathbf{x}}) \cdot \big((\mathbf{x}_i - \bar{\mathbf{x}})^\top \mathbf{v}\big)$$

每一项是 $\boldsymbol{\delta}_i \cdot (\boldsymbol{\delta}_i^\top \mathbf{v})$，即偏移向量乘以它在 $\mathbf{v}$ 方向上的投影标量——如果投影为负，这一项就自然指向 $-\boldsymbol{\delta}_i$ 方向。

而本文的方法是：如果 $\boldsymbol{\delta}_i \cdot \mathbf{d}_{acc} < 0$，则取 $-\boldsymbol{\delta}_i$——相当于把投影为负的项翻转。**区别在于**：PCA 中是乘以投影标量（连续权重），这里只是翻转（离散 ±1 权重）。所以这是一种"符号化"的 power iteration 的单次 pass 近似。

也可以联系到 **Incremental PCA** 和 **Oja's Rule**：

$$\Delta \mathbf{w} = \eta \cdot \mathbf{x}_i \cdot (\mathbf{x}_i^\top \mathbf{w}) - \eta \cdot \mathbf{w} \cdot (\mathbf{w}^\top \mathbf{x}_i \mathbf{x}_i^\top \mathbf{w})$$

Oja's rule 是一种在线 PCA 学习规则，而本文的方法可以看作一种极度简化的变体——只用符号（sign）而非投影值来决定更新方向。

---

## 3. 实验验证与局限性

### 3.1 在 Gaussian 数据上的表现

作者测试了在 2D 随机 Gaussian 上与 `sklearn.decomposition.PCA` 的对比，结果表明近似效果良好，添加 Gaussian noise 后两者表现大致相当。

### 3.2 失效案例：不相关的均匀/非 Gaussian 分布

当 X 和 Y 通道**完全无关**时：

- PCA 正确检测到 X 轴有更大的 variance → 第一主成分沿 X 轴
- 近似方法给出一个**错误的平均方向**（比如 45° 对角线）

**为什么失效？** 因为当两个方向独立时，偏移向量在第一象限和第三象限（沿 X 方向一致）的分布与第二、四象限对称，翻转操作会系统性地把次成分方向的贡献也保留下来，导致"折中"到一个对角方向。

这是一个**根本性的局限**：翻转操作假设了数据在主方向上的偏移有"极性"（polarity），即大部分偏移集中在一侧或两侧但方向一致。对于不相关的均匀分布，这个假设不成立。

### 3.3 可能的 bias

作者提到近似结果可能倾向于指向零方向——即估计的第一主成分方向可能系统性偏向原点方向（相对于均值）。这是因为排序使得负值先处理，初始累加方向被早期数据点影响。

---

## 4. 计算复杂度对比

| 操作 | Full PCA | Approximate PC | Axis-aligned |
|------|----------|----------------|--------------|
| 计算均值 | $O(nd)$ | $O(nd)$ | $O(nd)$ |
| 计算 covariance matrix | $O(nd^2)$ | **不需要** | **不需要** |
| Eigenvector 分解 | $O(d^3)$ | **不需要** | **不需要** |
| 排序 | 可选 | $O(n \log n)$ | $O(n \log n)$ |
| 主循环 | — | $O(nd)$ | $O(n)$ |
| **总计** | $O(nd^2 + d^3)$ | $O(nd + n\log n)$ | $O(nd)$ |

对于 $d = 3$（RGB 颜色），$d^3 = 27$ 很小，但 covariance matrix 的计算和 eigenvector 提取的常数因子不小（参考 libsquish 的实现），而近似方法只是一个简单循环。

---

## 5. 应用场景

### 5.1 颜色量化
Median cut 的改进版——用近似主成分方向代替 axis-aligned 切分平面，获得更好的分割质量。

### 5.2 纹理压缩（BC1/DXT1）
如 libsquish 库中，需要找到颜色端点（color endpoints），这本质上就是找点集的主方向。

### 5.3 Oriented Bounding Box (OBB)
在 mesh 上拟合方向对齐的包围盒，需要知道主轴方向。ShaderToy 上的 Tuan Kuranes 实现就是用此近似方法计算主轴，然后用 Duff et al. 的方法构建任意正交基来得到另外两个轴。

参考：Duff et al., "Building an Orthonormal Basis, Revisited", JCGT 2017
- http://jcgt.org/published/0006/01/01/

---

## 6. 更广泛的联系

### 6.1 Power Iteration
经典的求最大特征向量方法，反复乘以 covariance matrix。本文方法可以视为一种 "sign-power iteration" 的单步变体。

### 6.2 Oja's Rule / Incremental PCA
在线学习第一主成分的 Hebbian 学习规则。本文方法是一种极其简化的版本。

### 6.3 Random Projection + Sign
与 **random projection** 中使用 sign 来减少维度有精神上的相似——用离散化（sign）代替连续值来降低计算。

### 6.4 Weighted Median / Weiszfeld Algorithm
求几何中位数的方法也是一种迭代重加权，与本文的"逐步修正方向"有结构上的类比。

### 6.5 Spherical K-means
将数据投影到单位球面上做 k-means，也涉及方向的累加和归一化，与本文的翻转+累加操作有相似的几何直觉。

---

## 7. 总结

这篇 blog 的核心贡献是：

1. **发现并解释**了 exoquant 中一个"school level vector math"的近似主成分算法
2. **建立了直觉**：翻转累加 = 偏移向量对主方向的"投票"
3. **指出了局限**：对不相关分布的数据失效
4. **给出了实用定位**：介于 axis-aligned 和 full PCA 之间的轻量级方案

**原文链接**（推测）：作者 Fabian Giesen（rygorous），博客可能是 https://fgiesen.wordpress.com/ 上的文章。

**相关资源：**
- exoquant-rs: https://github.com/exoticorn/exoquant-rs
- exoquant.c 原始版本 (2004): https://github.com/exoticorn/exoquant
- libsquish PCA 实现: https://github.com/svn2github/libsquish
- Tuan Kuranes 的 OBB ShaderToy: https://www.shadertoy.com/ （具体链接在原文中）
- PCA 维基百科: https://en.wikipedia.org/wiki/Principal_component_analysis
- Power Iteration: https://en.wikipedia.org/wiki/Power_iteration