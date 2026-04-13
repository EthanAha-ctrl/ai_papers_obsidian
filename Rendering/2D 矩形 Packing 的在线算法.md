# Skyline 算法：2D 矩形 Packing 的在线算法详解

## 一、问题背景与动机

这篇文章讨论的是**2D 矩形装箱问题 (2D Rectangle Bin Packing)**，这是一个在多媒体开发、GPU 编程、游戏开发乃至工业领域都有广泛应用的问题。

### 1.1 为什么需要 2D Packing？

核心动机来自 **GPU 纹理绑定开销**：

- 在 GPU 渲染中，切换纹理 是非常昂贵的操作
- 如果每个 glyph 都单独一张纹理，渲染文字时会产生大量 bind 操作
- **解决方案**：将多个小矩形打包到一张大纹理中，称为 **atlas**
  - 字体场景 → **Glyph Atlas**
  - 游戏 sprite → **Spritesheet**
  - 网页图标 → 单个大文件下载比多个小文件更高效

### 1.2 工业应用

该问题不仅限于计算机图形学：
- 📰 报纸排版：一页能放多少广告？
- 🪵 木材切割：一块木板能切出多少零件？
- 📦 物流装载：货车能装多少包裹？

### 1.3 离线 vs 在线 Packing

参考来源：**Jukka Jylänki 的经典综述**（[A Thousand Ways to Pack the Bin - A Practical Approach to Two-Dimensional Rectangle Bin Packing](https://github.com/juj/RectangleBinPack)）

| 特性 | Offline Packing | Online Packing |
|------|----------------|----------------|
| **是否预知所有矩形** | ✅ 是 | ❌ 否 |
| **可排序优化** | ✅ 是 | ❌ 否 |
| **最优性** | 更优 | 次优 |
| **代表算法** | **MAXRECTS** | **Skyline** |
| **典型场景** | Spritesheet 生成 | 字体 glyph 缓存 |

作者的场景是 **字体 glyph 缓存**——运行时才知道需要哪些 glyph（字体 + 粗体/斜体/大小组合），因此必须用 Online 算法。这也是 **stb_rect_pack.h**、**fontstash**、**nanovg** 等知名库的选择。

> 🔗 Jukka Jylänki 的论文：[A Thousand Ways to Pack the Bin](https://github.com/juj/RectangleBinPack/blob/master/RectangleBinPack.pdf)

---

## 二、API 设计

作者追求极简主义，整个 API 由 **一个结构体 + 一个函数** 组成：

```c
struct JvPack2d {
    uint16_t* pSkyline;       // skyline 数据数组，容量需 ≥ 2 * maxWidth
    uint16_t maxWidth;        // atlas 宽度
    uint16_t maxHeight;       // atlas 高度
    bool _bInitialized;       // 私有状态，需零初始化
    uint16_t _skylineCount;   // skyline 中的顶点数
};

bool jvPack2dAdd(JvPack2d* pP2D, uint16_t const size[2], uint16_t pos[2]);
```

### 关键设计决策

1. **`uint16_t` 坐标系**：GPU 纹理单维度上限为 16384 像素，`uint16_t` 范围 [0, 65535] 足够
2. **用户管理内存**：`pSkyline` 由调用者分配，最大需要 `2 * maxWidth` 个 `uint16_t`
   - 最坏情况：1 像素宽的矩形形成锯齿状 skyline
   - **优化技巧**：如果所有矩形宽度是 4 的倍数，可将所有宽度/`maxWidth` 除以 4，输出位置乘以 4，节省 75% 内存
3. **Freestanding**：不依赖任何标准库

---

## 三、Skyline 算法核心原理

### 3.1 核心思想

Skyline 算法**从底向上**逐行放置矩形，只追踪每一列的**最高位置**，形成一个类似城市天际线 的轮廓。

$$\text{skyline}[x] = \text{第 } x \text{ 列的最高已占位置}$$

这本质上是一个 **1D 高度图 (1D Height Map)**。

### 3.2 数据结构

Skyline 用**有序顶点数组**表示，每个顶点是水平线段的左端点，按 $x$ 坐标从左到右排序：

$$\text{skyline} = [(x_0, y_0), (x_1, y_1), \ldots, (x_{n-1}, y_{n-1})]$$

其中每个 $(x_i, y_i)$ 表示：从 $x_i$ 开始到 $x_{i+1}$（或 `maxWidth`）的这一段，高度为 $y_i$。

**示例**（文中图示）：

$$\text{skyline} = [(0, 8), (2, 5), (5, 3), (7, 4)]$$

表示：
- 列 0–1：高度 8
- 列 2–4：高度 5  
- 列 5–6：高度 3
- 列 7+：高度 4

### 3.3 浪费空间问题

⚠️ Skyline 只维护轮廓，**不追踪轮廓下方的空隙**。矩形层叠可能产生无法利用的 "死角"。

```
┌──┐
│  │   ← 空隙：skyline 不追踪这里
│  ┌──┐
│  │  │
└──┘  │
   └──┘
```

**缓解方案**：可用 **freelist** 追踪这些空隙，但每次插入需要线性搜索 freelist，**牺牲性能换利用率**。作者选择不实现 freelist，保持简洁高效。

---

## 四、算法两步骤详解

### Step 1: 搜索最佳插入位置

**贪心策略**：找到 **最低行** → **最左列** 的位置。

#### 形式化描述

给定待插入矩形尺寸 $(w, h)$，对 skyline 中每个顶点 $(x_i, y_i)$ 作为候选：

1. **右边界检查**：若 $w > \text{maxWidth} - x_i$，矩形超出 atlas，跳过
2. **碰撞提升**：遍历所有满足 $x_i \leq x_j < x_i + w$ 的顶点 $(x_j, y_j)$：
   - 若 $y_j > y_i$，则提升：$y_i \leftarrow y_j$（矩形必须放在更高的位置以避免碰撞）
3. **上边界检查**：若 $h > \text{maxHeight} - y_i$，矩形超出 atlas 高度，跳过
4. **更新最优**：若 $y_i < \text{bestY}$（或 $y_i = \text{bestY}$ 且 $x_i < \text{bestX}$），更新最优候选

#### 为什么只需遍历顶点而非每列？

关键观察：**"最左" 偏好**。如果某个列位置可行，那么向左移动直到遇到 skyline 顶点，位置只会更好或等价（不会更差）。因此最优位置一定在某个顶点的 $x$ 坐标处。

#### 内循环：碰撞检测

对于候选位置 $(x, y)$ 和矩形宽度 $w$，内循环遍历 $x_j \in [x, x+w)$ 范围内的顶点：

$$y \leftarrow \max(y, y_j) \quad \forall j : x \leq x_j < x + w$$

同时记录被覆盖的顶点范围 $[\text{idxBest}, \text{idxBest2})$，用于 Step 2 的数据结构更新。

```c
// 搜索最佳位置的核心代码
for (uint16_t idx = 0; idx < skylineCount; ++idx) {
    uint16_t x = pSkyline[idx].x;
    uint16_t y = pSkyline[idx].y;

    if (width > maxWidth - x)
        break;                    // 超出右边界
    if (y >= bestY)
        continue;                 // 不可能比当前最优更好

    uint16_t xMax = x + width;
    uint16_t idx2;
    for (idx2 = idx + 1; idx2 < skylineCount; ++idx2) {
        if (xMax <= pSkyline[idx2].x)
            break;                // 不再与后续顶点重叠
        if (y < pSkyline[idx2].y)
            y = pSkyline[idx2].y; // 提升 y 避免碰撞
    }

    if (y >= bestY)
        continue;                 // 提升后仍不够好
    if (height > maxHeight - y)
        continue;                 // 超出上边界

    idxBest = idx;
    idxBest2 = idx2;
    bestX = x;
    bestY = y;
}
```

**时间复杂度**：外循环 $O(n)$，内循环 $O(n)$，理论上单次插入 $O(n^2)$，其中 $n = \text{skylineCount}$。

---

### Step 2: 更新 Skyline 数据结构

找到位置后，需要：
1. **删除**被新矩形"遮挡"的顶点
2. **插入**1–2 个新顶点

#### 新增的两个顶点

- **TL (TopLeft)**：$(x, y + h)$ — 新矩形的顶边左端
  - 始终插入
- **BR (BottomRight)**：$(x + w, y_{\text{last}})$ — 新矩形的右边下端
  - $y_{\text{last}}$ = 被遮挡的最后一个顶点的高度
  - **条件插入**：仅当 BR 不会与下一个顶点同列且不会超出 atlas 边界时

#### BR 不插入的三种情况

| 情况 | 条件 | 原因 |
|------|------|------|
| 矩形右边界 = atlas 右边界 | $x + w = \text{maxWidth}$ | 不需要记录 |
| BR 与下一顶点同列 | $\text{BR}.x = \text{next}.x$ | 避免同列多顶点，保证 $O(\text{maxWidth})$ 上限 |
| 正常情况 | 上述都不满足 | ✅ 插入 BR |

#### 数组操作

删除 $[\text{idxBest}, \text{idxBest2})$ 范围的顶点，插入 $\text{insertedCount} \in \{1, 2\}$ 个新顶点：

$$\Delta = \text{insertedCount} - \text{removedCount}$$

- $\Delta > 0$：数组**扩张**，从后向前移位
- $\Delta < 0$：数组**收缩**，从前向后移位
- $\Delta = 0$：直接覆盖

```c
if (insertedCount > removedCount) {
    // 扩张：从后向前移
    for (idx = skylineCount-1; idx >= idxBest2; --idx)
        pSkyline[idx + Δ] = pSkyline[idx];
} else if (insertedCount < removedCount) {
    // 收缩：从前向后移
    for (idx = idxBest2; idx < skylineCount; ++idx)
        pSkyline[idx - |Δ|] = pSkyline[idx];
}
pSkyline[idxBest] = newTL;
if (bBottomRightPoint)
    pSkyline[idxBest + 1] = newBR;
```

**不变量**：$\text{skylineCount} \leq \text{maxWidth}$（保证内存安全）

---

## 五、复杂度分析

### 5.1 空间复杂度

$$S(n) = O(\text{maxWidth})$$

Skyline 顶点数上限为 `maxWidth`（每列一个顶点，如锯齿状城堡轮廓），因此 `pSkyline` 数组需要 `2 * maxWidth` 个 `uint16_t`。

### 5.2 时间复杂度

| 场景 | 每次插入 | 总体（n 个矩形） |
|------|---------|-----------------|
| **理论最坏** | $O(\text{maxWidth}^2)$ | $O(n \cdot \text{maxWidth}^2)$ |
| **宽矩形为主** | $O(\text{maxWidth})$ | $O(n \cdot \text{maxWidth})$ |
| **窄矩形为主** | 内循环短，外循环长 | 见下方直觉分析 |

### 5.3 作者的直觉分析

> "The algorithm contains two nested loops, thus the cost per added rectangle could theoretically be quadratic in ATLAS_SIZE, but I did not measure it. Without doing a mathematical analysis, my intuition is that the inner loop checking collisions with the skyline is proportional to the rectangle's width; but the bigger the width is, the less points there will be in the skyline, thus the outer loop does less work next time."

这本质上是一个**摊还分析 (Amortized Analysis)** 的直觉：

$$\sum_{i=1}^{n} (\text{外循环迭代数}_i \times \text{内循环迭代数}_i)$$

- 宽矩形 → 内循环长，但合并大量顶点 → skyline 变短 → 后续外循环短
- 窄矩形 → 内循环短 → skyline 变长 → 后续外循环长

两者相互制约，**总工作量可能并非简单的二次方**。

---

## 六、实验数据深度解读

### 6.1 四个 Worst-Case 场景

| 场景 | Atlas 尺寸 | 矩形尺寸 | 矩形数 | Skyline 峰值点数 |
|------|-----------|---------|--------|-----------------|
| **WorstCaseWidth** | $(S, 2)$ | $(1, 1)$ | $2S$ | $S$ |
| **WorstCaseHeight** | $(2, S)$ | $(1, 1)$ | $2S$ | $2$ |
| **WorstCaseDiagonalV** | $(S, S)$ | $1$-宽矩形 | $2S-1$ | $S$ |
| **WorstCaseDiagonalH** | $(S, S)$ | $1$-高矩形 | $2S-1$ | $2$ |

### 6.2 数据趋势分析

**WorstCaseWidth**（每次插入 1×1 矩形，skyline 有 S 个点）：

| ATLAS_SIZE | 时间 | 比率 | 每矩形时间 |
|-----------|------|------|-----------|
| 512 | 0.7 ms | — | 0.68 µs |
| 1024 | 2.9 ms | 4.1× | 1.42 µs |
| 2048 | 10.0 ms | 3.4× | 2.44 µs |
| 4096 | 37.2 ms | 3.7× | 4.54 µs |
| 8192 | 122.3 ms | 3.3× | 7.45 µs |
| 16384 | 524.4 ms | 4.3× | 16.0 µs |
| 32768 | 2322.1 ms | 4.4× | 35.4 µs |
| 65535 | 7935.0 ms | 3.4× | 60.5 µs |

当 `ATLAS_SIZE` 翻倍时，时间约 ×4 → **$O(S^2)$ 行为**。

**解释**：1×1 矩形宽度为 1，内循环几乎不执行（最多检查 1 个点），但 skyline 始终保持 ~S 个点，外循环每次 $O(S)$，总共 $2S$ 个矩形 → 总 $O(S^2)$。

**WorstCaseHeight**（skyline 只有 2 个点）：

| ATLAS_SIZE | 时间 | 每矩形时间 |
|-----------|------|-----------|
| 512 | 0.0 ms | ~0 µs |
| 65535 | 2.4 ms | 0.018 µs |

时间几乎不增长 → **$O(S)$ 行为**。外循环只遍历 2 个点，内循环也极短。

### 6.3 关键洞察

$$\boxed{T_{\text{total}} \approx \sum_{k=1}^{n} \text{skylineCount}_k}$$

**总时间约等于所有插入时刻的 skyline 顶点数之和**。这解释了为什么 WorstCaseWidth 是二次的而 WorstCaseHeight 是线性的。

---

## 七、与 MAXRECTS 算法的对比

### 7.1 MAXRECTS 核心思想

MAXRECTS 维护**所有空闲矩形的列表**，而非仅维护轮廓：

$$\mathcal{F} = \{R_1, R_2, \ldots, R_m\}$$

每次插入时：
1. 遍历所有空闲矩形，找到最佳拟合
2. 插入矩形后，将重叠的空闲矩形**分裂**为最多 4 个子矩形
3. 合并完全包含的矩形，去除冗余

### 7.2 对比

| 维度 | Skyline | MAXRECTS |
|------|---------|----------|
| **数据结构** | 有序顶点数组 | 空闲矩形列表 |
| **空间复杂度** | $O(W)$ | $O(W \cdot H)$ 或更高 |
| **碎片处理** | 不追踪内部空隙 | 完全追踪 |
| **利用率** | 较低 (~85-90%) | 较高 (~90-95%) |
| **Online 支持** | ✅ 天然支持 | ⚠️ 可以但非最优 |
| **实现复杂度** | 低 | 中等 |
| **典型用途** | 运行时 glyph 缓存 | 离线 spritesheet 生成 |

### 7.3 Skyline 的变体

Jukka Jylänki 的论文中还讨论了 Skyline 的几种**启发式选择策略**：

1. **Bottom-Left (BL)**：最低最左（本文采用的策略）
2. **MinWaste**：选择浪费空间最小的位置
3. **Best Area Fit**：选择剩余面积最紧凑的位置

> 🔗 参考：[stb_rect_pack.h](https://github.com/nothings/stb/blob/master/stb_rect_pack.h) 实现了 Bottom-Left 和 Best Area Fit 两种策略

---

## 八、第一性原理思考

### 8.1 为什么 Skyline 有效？

从第一性原理出发，2D Packing 的本质困难是**二维约束的组合爆炸**。Skyline 的核心洞察是：

$$\text{2D 问题} \xrightarrow{\text{只追踪轮廓}} \text{1D 问题}$$

通过**降维**，将二维的空间管理简化为一维的高度图，用 $O(W)$ 空间代替 $O(W \times H)$ 空间。

### 8.2 降维的代价

降维必然**丢失信息**——轮廓下方的空隙信息被丢弃。这是 **精确性 vs 效率** 的经典权衡：

$$\text{利用率} \uparrow \iff \text{信息量} \uparrow \iff \text{复杂度} \uparrow$$

Freelist 是一种折中：追踪部分空隙，但增加了搜索成本。

### 8.3 贪心策略的理论保证

Skyline 采用的 Bottom-Left 贪心策略对 Online Packing 没有**竞争比** 保证——存在构造性反例使得贪心解与最优解的比值任意大。但在实践中，对于"合理"的输入分布，表现通常可接受。

> 🔗 理论参考：[Online Bin Packing - First Fit 竞争比为 ~1.7](https://en.wikipedia.org/wiki/Bin_packing_problem#Online_algorithms)

---

## 九、实现细节补充

### 9.1 初始化

首次调用 `jvPack2dAdd` 时检测 `_bInitialized == false`，初始化 skyline 为单个点 $(0, 0)$——表示整个 atlas 从第 0 行开始都是空的。

### 9.2 零尺寸矩形

```c
if (width == 0 || height == 0)
    return false;
```

0 宽/高矩形无意义，直接拒绝。

### 9.3 内存安全断言

```c
_jvASSERT(skylineCount + insertedCount - removedCount, <=, maxWidth);
```

确保 skyline 顶点数始终不超过 `maxWidth`，这是内存安全的根本保证。

---

## 十、总结与延伸

### 核心要点

1. **Skyline = 1D 高度图**，通过降维简化 2D Packing
2. **Online 算法**，不需要预知所有矩形，适合运行时 glyph 缓存
3. **时间复杂度**：最坏 $O(W^2)$，实践中摊还后通常更好
4. **空间复杂度**：$O(W)$，远优于 MAXRECTS
5. **利用率**：因不追踪内部空隙而偏低，可用 freelist 弥补

### 延伸阅读

- 🔗 [Jukka Jylänki 的综述论文](https://github.com/juj/RectangleBinPack)
- 🔗 [stb_rect_pack.h](https://github.com/nothings/stb/blob/master/stb_rect_pack.h) — 最广泛使用的开源实现
- 🔗 [fontstash](https://github.com/memononen/fontstash) — 基于 Skyline 的字体缓存
- 🔗 [nanovg](https://github.com/memononen/nanovg) — 使用 fontstash 的 2D 绘图库
- 🔗 [Rectangle Bin Packing on Wikipedia](https://en.wikipedia.org/wiki/Bin_packing_problem)
- 🔗 [作者源码](https://jvns.ca/) (UNLICENSE 公共领域)
- 🔗 [Bin Packing 问题的近似算法理论](https://www.sciencedirect.com/science/article/pii/S0022000004001178)

### 改进方向

| 方向 | 方法 | 代价 |
|------|------|------|
| 提高利用率 | 添加 freelist | 插入变慢 |
| 提高利用率 | 多 atlas 策略 | 管理复杂度 |
| 加速搜索 | 用 segment tree / 优先队列维护 skyline | 实现复杂度 |
| 减少碎片 | Waste Map Skyline 变体 | 额外数据结构 |
| 并行化 | 批量插入 + 排序 | 丧失 Online 性质 |