

# MMD, COV, 1-NNA: 3D Point Cloud Generation 的核心评估 Metrics

让我先搜索一下最新的相关资料。





好，现在我有足够的信息来详细解释这三个 metrics 了。

---

## 背景：为什么需要这三个 Metrics？

这三个 metrics 最早由 **Achlioptas et al. (ICML 2018)** 在论文 *"Learning Representations and Generative Models for 3D Point Clouds"* 中系统提出，用于评估 **3D point cloud generative model** 的质量。

核心问题是：给定一个 **reference set** $S_r$（真实的 point clouds）和一个 **generated set** $S_g$（模型生成的 point clouds），如何量化 $S_g$ 的质量？

这与 2D image generation 中用 FID/IS 评估 GAN 类似，但 point cloud 是无序的 3D 点集合，需要专门的 distance measure。

---

## 前置知识：Point Cloud 之间的 Distance

在计算这三个 metrics 之前，首先要定义两个 **individual point cloud** 之间的距离 $d(X, Y)$，常用两种：

### 1. Chamfer Distance (CD)

$$d_{CD}(X, Y) = \frac{1}{|X|} \sum_{x \in X} \min_{y \in Y} \|x - y\|_2^2 + \frac{1}{|Y|} \sum_{y \in Y} \min_{x \in X} \|x - y\|_2^2$$

- $X, Y$：两个 point cloud，每个都是 $\mathbb{R}^3$ 中的点集合
- $|X|$：point cloud $X$ 中的点数
- $\|x - y\|_2^2$：两点之间的 Euclidean distance 的平方
- 第一项：对 $X$ 中每个点，找 $Y$ 中最近的点，求平均距离（衡量 $X \to Y$ 的 coverage）
- 第二项：对称地衡量 $Y \to X$
- **直觉**：CD 像"双向最近邻搜索"的平均值，计算复杂度 $O(|X| \cdot |Y|)$

### 2. Earth Mover's Distance (EMD)

$$d_{EMD}(X, Y) = \min_{\phi: X \to Y} \frac{1}{|X|} \sum_{x \in X} \|x - \phi(x)\|_2$$

- $\phi$：一个从 $X$ 到 $Y$ 的 **bijection**（一一对应映射），要求 $|X| = |Y|$
- 在所有可能的 bijection 中找使总距离最小的那个
- **直觉**：想象把一堆泥土（$X$）搬运到目标位置（$Y$），EMD 是最小搬运代价
- 计算复杂度 $O(n^3)$（Hungarian algorithm），比 CD 贵得多

---

## 1. MMD (Minimum Matching Distance) — 衡量 **Fidelity（保真度）**

### 定义

$$\text{MMD}(S_g, S_r) = \frac{1}{|S_r|} \sum_{Y \in S_r} \min_{X \in S_g} d(X, Y)$$

- $S_r$：reference set（真实 point cloud 的集合），包含 $|S_r|$ 个 point cloud
- $S_g$：generated set（生成的 point cloud 的集合），包含 $|S_g|$ 个 point cloud
- $d(X, Y)$：用 CD 或 EMD 计算的两个 point cloud 之间的距离
- 对 **reference set 中的每一个** point cloud $Y$，找 generated set 中与它**最相似**的 $X$
- 然后对所有这些最小距离取**平均**

### 直觉

> **MMD 回答的问题是：生成的样本是否能"高质量地覆盖"每一个真实样本？**

- **越低越好**（Lower is better）
- 如果生成的 point cloud 质量很差（形状不像真实的），即使数量再多，每个真实样本的最近匹配距离都会很大 → MMD 高
- **弱点**：MMD 对 **mode collapse** 不敏感。假设 generated set 只生成了一种极其精确的形状，只要这个形状恰好与某些 reference 很接近，那些 reference 的匹配距离就低。但如果很多 reference 找不到匹配，MMD 会高。不过如果 reference set 中形状多样性高而 generated set 只有一种形状，MMD 会很高。

### 关键细节

注意方向性：MMD 是从 $S_r$ **向** $S_g$ 匹配的。每个 reference 找 generated 中最近的。如果 generated set 极其丰富但质量都差，MMD 高；如果 generated set 只有少量高质量样本，某些 reference 可能匹配不到 → MMD 也高。

---

## 2. COV (Coverage) — 衡量 **Diversity（多样性）**

### 定义

$$\text{COV}(S_g, S_r) = \frac{|\{ Y \in S_r \mid \exists X \in S_g, \text{arg}\min_{Y' \in S_r} d(X, Y') = Y \}|}{|S_r|}$$

更直白地说：

1. 对 generated set 中的**每一个** $X \in S_g$，找 reference set 中与它最近的 $Y^* = \text{arg}\min_{Y \in S_r} d(X, Y)$
2. 把所有这些被"匹配到"的 $Y^*$ 收集起来，得到一个子集
3. COV = 这个子集的大小 / $|S_r|$

### 直觉

> **COV 回答的问题是：生成的样本"覆盖"了多少种不同的真实样本？**

- **越高越好**（Higher is better），范围 $[0, 1]$
- 如果 generative model 发生 **mode collapse**（只生成几种形状），那大多数 reference point cloud 永远不会被匹配到 → COV 低
- 如果 generative model 很 diverse，能覆盖所有 mode → COV 高
- **弱点**：COV 对 **quality** 不敏感。即使生成的 point cloud 质量很差（但足够 diverse），每个 reference 可能都被至少一个 generated sample 匹配到（虽然距离很大），COV 仍然可以很高

### MMD 与 COV 的互补性

| 场景 | MMD | COV |
|------|-----|-----|
| 高质量 + 高多样性 | **低** ✓ | **高** ✓ |
| 高质量 + mode collapse（只生成一种形状） | 高 ✗ | **低** ✗ |
| 低质量 + 高多样性 | **高** ✗ | 高 ✓ |
| 低质量 + 低多样性 | 高 ✗ | 低 ✗ |

正如 Achlioptas 原文所说：*"The complementary nature of MMD and Coverage directly follows from their definitions."*

---

## 3. 1-NNA (1-Nearest Neighbor Accuracy) — **同时**衡量 Fidelity 和 Diversity

### 定义

这是一个基于 **two-sample test** 的 metric，来自 statistical hypothesis testing。

设 $S_r$ 和 $S_g$ 各有 $N$ 个 point cloud（通常要求等大小）。将它们合并成一个集合 $S_{r} \cup S_{g}$，共 $2N$ 个样本。

对合并集中的每个样本 $X$，找**除自身之外**的 1-nearest neighbor $N_X$，然后看 $N_X$ 来自哪个集合：

$$\text{1-NNA}(S_g, S_r) = \frac{\sum_{X \in S_g} \mathbb{1}[N_X \in S_g] + \sum_{Y \in S_r} \mathbb{1}[N_Y \in S_r]}{|S_g| + |S_r|}$$

其中：
- $\mathbb{1}[\cdot]$ 是 indicator function（条件为真时返回 1，否则返回 0）
- $N_X$ 是在 $S_g \cup S_r \setminus \{X\}$ 中与 $X$ 距离最近的样本（用 CD 或 EMD 作为距离）
- 分子第一项：generated 样本中，其最近邻也来自 generated set 的数量
- 分子第二项：reference 样本中，其最近邻也来自 reference set 的数量

### 直觉（第一性原理）

> **核心思想：如果 $S_g$ 和 $S_r$ 来自完全相同的分布，那么一个样本的最近邻来自 $S_g$ 或 $S_r$ 的概率应该各是 50%。**

- **理想值 = 50%**（$S_g$ 和 $S_r$ 无法区分）
- **1-NNA > 50%**：说明两个集合是 **可区分的**，即 generated 样本和 real 样本在分布上有差异
  - 如果 generated 样本质量差 → generated 样本倾向于与其他 generated 样本更近 → 1-NNA 趋近 100%
  - 如果 mode collapse → generated 样本聚集在一起 → generated 的最近邻都是 generated → 1-NNA 高
- **1-NNA < 50%**：理论上不太可能，但如果 generated set 中某些样本恰好"填充"了 reference set 之间的空隙，可能出现

### 为什么 1-NNA 更 robust？

来自 **Yang & Huang (PointFlow, 2019)** 的分析：

> *"Unlike COV and MMD, 1-NNA directly measures distributional similarity and takes both diversity and quality into account."*

| Failure mode | MMD 能检测？ | COV 能检测？ | 1-NNA 能检测？ |
|---|---|---|---|
| 低质量 (bad fidelity) | ✓ | ✗ | ✓ |
| Mode collapse (low diversity) | 部分 | ✓ | ✓ |
| 分布不匹配 | 部分 | 部分 | ✓ |

---

## 完整的评估 Pipeline 示意

```
Reference Set S_r (N real point clouds)
    │
    ├──── 每个 point cloud 采样 2048 个点
    │
Generated Set S_g (N generated point clouds)
    │
    ├──── 每个 point cloud 采样 2048 个点
    │
    ▼
┌─────────────────────────────────────────────┐
│  Distance Matrix D[i][j] = d(S_g[i], S_r[j]) │
│  d 可以是 CD 或 EMD                          │
└─────────────────────────────────────────────┘
    │
    ├─→ MMD: 对 S_r 的每行取 min，再求 mean → 标量
    ├─→ COV: 对 S_g 的每行取 argmin，统计被匹配到的 S_r unique 个数 / |S_r|
    └─→ 1-NNA: 合并两个集合，Leave-one-out 1-NN 分类 → accuracy
```

---

## 实验数据示例（来自 PointFlow 论文）

| Model | 1-NNA-CD ↓ | 1-NNA-EMD ↓ | COV-CD ↑ | COV-EMD ↑ | MMD-CD ↓ | MMD-EMD ↓ |
|---|---|---|---|---|---|---|
| r-GAN | 83.53 | 76.14 | 33.18 | 20.27 | 5.384 | 7.294 |
| l-GAN (CD) | 68.45 | 67.61 | 41.82 | 36.59 | 3.735 | 6.494 |
| PointFlow | **46.98** | **46.59** | **46.83** | **46.98** | **2.419** | **5.938** |

**观察**：
- PointFlow 的 1-NNA 最接近理想值 **50%** → 分布匹配最好
- r-GAN 的 1-NNA 高达 83%，说明生成样本与真实样本很容易区分
- COV 和 MMD 的趋势一致但信息互补

---

## 深层直觉总结

从**第一性原理**来看，评估一个 generative model 需要回答三个问题：

1. **"生成的东西像不像真的？"** → **Fidelity / Quality** → 用 **MMD** 衡量（每个真实样本都能找到一个像它的生成样本吗？匹配距离有多小？）

2. **"生成的东西多不多样？"** → **Diversity / Coverage** → 用 **COV** 衡量（生成样本覆盖了多少种真实样本的 mode？）

3. **"生成的整体分布和真实分布是否一致？"** → **Distribution Match** → 用 **1-NNA** 衡量（你能分辨哪个是真的哪个是假的吗？如果不能 → 1-NNA ≈ 50% → 完美匹配）

这三个 metrics 合在一起就像一个"三角校验"：MMD 和 COV 各有盲点，而 1-NNA 提供了一个更全面的 two-sample test。

---

## 参考资料

- **Achlioptas et al. (2018)**: [Learning Representations and Generative Models for 3D Point Clouds](http://proceedings.mlr.press/v80/achlioptas18a/achlioptas18a.pdf) — 原始定义
- **Yang et al. (2019)**: [PointFlow: 3D Point Cloud Generation with Continuous Normalizing Flows](https://graphics.stanford.edu/courses/cs348n-22-winter/PapersReferenced/PointFlow_1906.12320.pdf) — 引入 1-NNA 的深入分析
- **Zeng et al. (2022)**: [LION: Latent Point Diffusion Models](https://openreview.net/forum?id=tHK5ntjp-5K) — 使用这些 metrics 的现代 diffusion model
- **Rethinking Metrics (2025)**: [Rethinking Metrics and Diffusion Architecture for 3D Point Cloud Generation](https://arxiv.org/html/2511.05308v1) — 对这些 metrics 的局限性进行系统性反思