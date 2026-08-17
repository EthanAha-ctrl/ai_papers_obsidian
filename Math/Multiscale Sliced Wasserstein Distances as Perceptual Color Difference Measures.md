---
source_pdf: Multiscale Sliced Wasserstein Distances as Perceptual Color Difference
  Measures.pdf
paper_sha256: e8a1463913f3a69f07411c1876787c3941ebb36997f8e8dc905382dd155fac0d
processed_at: '2026-08-05T21:37:49-07:00'
target_folder: Math
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说这篇 paper

## 一句话说清楚

**怎么判断两张照片颜色"看起来"差多少？以前的办法要求两张照片像素对齐，对不齐就崩了。这篇 paper 用一个不用训练、计算很快的方法，让颜色相似的色块自动找对应，管你照片对不对齐。**

---

## 问题到底是啥

你拿 iPhone 和 Samsung 拍同一棵树。两张照片颜色有点不一样 - iPhone 偏暖，Samsung 偏冷。你想量化"颜色差多少"，这就是 color difference (CD) 评估。

老办法 (CIELAB ΔE, CIEDE2000) 怎么做？拿对应位置的像素一个一个比，pixel (100, 200) 跟 pixel (100, 200) 比。问题是：你手抖了一下，两张照片有 5 个 pixel 的平移。于是 pixel (100, 200) 比的其实是树的边缘和天空，**全是假的 color difference**。

更极端的例子：把一张照片水平翻转。对人来说颜色完全一样，但所有 co-located 方法都说"差异巨大"。这显然不对。

人的视觉系统根本不这么干活。你看两张照片，不会盯着 (100, 200) 这个坐标比，你会自动把"左边那块树叶"和"左边那块树叶"对应起来，不管它精确落在哪个 pixel 上。**color 和 structure 是一起感知的**，这是 Gestalt psychology 几十年前就讲清楚的。

---

## 核心思路 - 三件事拼起来

### 第一件：别比 pixel，比 patch 分布

不要问"这个像素对应那个像素"，而是问"**这张图里色块的统计分布，和那张图里色块的统计分布，差多少**"。

举例：图 A 里有 30% 是天空蓝 patch，20% 是树叶绿 patch，10% 是树干棕 patch... 图 B 也有类似的分布。如果两图颜色一样，这些 patch 分布应该重合。这个想法把"位置对齐"这件事彻底扔掉了。

但怎么算两个分布的差距？用 Wasserstein distance - 想象把一堆泥土从分布 A 搬到分布 B，最少花多少力气。这个距离比 KL divergence 好在：分布完全不重叠时它也能给出有意义的值。

### 第二件：Wasserstein 太贵，用 "切西瓜" 技巧降维

直接算 Wasserstein 在高维上计算量爆炸 - image patch 展开是几百维，几十万个 patch，根本算不动。

Sliced Wasserstein 的 trick：**随机选一个方向，把所有 patch 投影到这条线上变成一个数字，在 1D 上算 Wasserstein，重复几百次取平均**。1D Wasserstein 好算 - 把两边数字排个序，对应位置相减就行，$\mathcal{O}(M \log M)$。

这里的 magic：投影 + 排序之后，**两张图里色块相似的 patch 会落到差不多的排名位置**。比如投影方向偏"蓝色"，那么最蓝的 patch 在两图都排第一，次蓝的都排第二... 于是排序匹配就自动建立了 non-local 对应。不需要显式找 nearest neighbor，sort() 一把搞定。

### 第三件：单尺度不够，必须 multiscale

只在一个 scale 上做，会出现一个问题：你可以把一张图的所有色块随机打乱位置重新拼，**SWD 完全不变**。因为分布没变。但人眼看这张图就是乱的。

解决：建 Gaussian pyramid，从原图到不断模糊下采样的低分辨率图，每一层都算 SWD 取平均。粗层约束全局颜色分布，细层约束局部结构。**五层叠加起来，patch 分布匹配才能锁死 pixel-level 的图像内容**。

Paper 里 Fig. 4 这个实验特别直观 - 从 noise 出发最小化 MS-SWD 去恢复原图：
- 1 层：还是 noise，啥也看不出来
- 3 层：出现模糊的色块
- 5 层：完整恢复原图

所以 multiscale 不是"锦上添花"，是 **metric property 的硬性要求**。

---

## 为什么用 CIELAB 不用 sRGB

sRGB 是 perceptually non-uniform - 同样 Euclidean 距离，暗处的颜色差一点和亮处的颜色差一点，人感知差很多。CIELAB 设计目标就是让 Euclidean distance 近似 perceptual difference，虽然不完美但够用。

也不用 VGG 这种 deep features，因为：1) deep features 把 color 信息丢了不少 (ReLU 砍掉负值)；2) deep features 是 co-located 比较，对 misalignment 敏感 (实验里 LPIPS 在 non-aligned PLCC 掉到 0.272)；3) 不想训练。

---

## 实验结果一句话总结

**SPCD dataset 上 30,000 对图，non-aligned 部分 MS-SWD PLCC=0.841，碾压所有 baselines，包括 60M 参数的 CD-Flow (trained)。而且 horizontal flip 这种杀手场景 PLCC 还是 0.836，其他方法基本归零。**

关键对比：
- CIEDE2000 aligned 上 0.827，non-aligned 掉到 0.653
- LPIPS aligned 上 0.767，non-aligned 崩到 0.272
- MS-SWD aligned 上 0.778，non-aligned 反而稳定在 0.841
- MS-SWD 完全 training-free，加 0.05M 参数 learnable projection 版本，overall PLCC 到 0.884

translation 5%、scale 1.1x、horizontal flip 三种几何变换，MS-SWD 几乎不掉点，其他全崩。**因为 patch 分布跟 spatial arrangement 无关，你怎么变换图像分布都不变。**

---

## 它还做了什么

1. **CD map 可视化** - Fig. 5 显示其他方法在 misaligned pair 上沿 object boundary 产生假的 high CD，MS-SWD 准确指出了真正颜色差异的位置 (云、楼、树)
2. **Metric property 验证** - 在 100,000 个 image triplet 上验证 triangle inequality 没违反，所以这玩意能当 loss 用
3. **Color transfer 应用** - 给一张 grayscale 图，从 color source 图 transfer 颜色，最小化 MS-SWD 就行。video 也能做，temporal consistency 还不错，因为 SWD 是统计性指标，frame 间平滑

---

## 为什么这 paper 有意思

几个点我觉得你会 appreciate：

**1. 在 deep learning 时代，random projection + sort 这种 training-free 方法能在特定 problem 上 beat 60M 参数 trained model。** 这说明 problem 的 inductive bias 抓对了，简单算法就够。这种 work 让人想到 nanoGPT 的哲学 - 用 minimal implementation 揭示 mechanism。

**2. sort() 这个 operator 把 non-local patch matching 从 $\mathcal{O}(M^2)$ 的 nearest neighbor search 降到 $\mathcal{O}(M \log M)$，且 implicitly 实现了 bidirectional 对应。** 这是 algorithm 设计上很漂亮的 move，不是 brute force，是利用了 problem structure。

**3. 整个 pipeline 没 invent 任何新东西** - Gaussian pyramid (1990s)、SWD (2012)、CIELAB (1976) 全是现成的。组合方式对了，效果就出来。这种 work 是对 problem 本质的理解，不是堆 model。

---

## 一句话给你带走

**Color perception 不靠 pixel 对齐，靠 patch 分布匹配。Multiscale + Sliced Wasserstein + sort，training-free 就能干翻 trained model，尤其对 misalignment 鲁棒。Inductive bias 比参数量重要。**

---

# Multiscale Sliced Wasserstein Distances as Perceptual Color Difference Measures - 深入解析

## 1. Problem Setting 与 Motivation

Karpathy 你看这篇 paper 的核心 problem statement 非常 clean。**Color difference (CD) assessment** 是 evaluate 两张 photographic image 在 human perception 上 color 差异的任务。传统方法 (CIELAB ΔE, CIEDE2000, S-CIELAB, CD-Net, CD-Flow) 都 implicit 假设两图是 pixel-aligned 的，做 **co-located comparison**。但 real-world 场景下，比如不同 smartphone 拍同一场景，image pairs 是 **misaligned** 的 (global motion, local object displacement, viewpoint variation, 甚至 horizontal flip)。这时 co-located comparison 就崩溃了，因为 human perceptual system 实际上做的是 **non-local** 的 color/structure 对应。

Fig. 1 那 4 个 cases 很直观地揭示了这件事 - CIEDE2000 这种 metric 在 misaligned pair 上完全失效。这其实呼应了 Gestalt psychology 的现代理解：color 和 structure 在 visual cortex 中是 **inextricably interdependent** 的 unitary process (参考文献 [2, 22, 42] - Kanizsa 的 Gestalt perception 和 Shapley & Hawken 关于 cortex 中 color 编码)。

这个 problem 我觉得和你在 Tesla 做 vision 的直觉很像 - human perceptual system 不靠 pixel grid 对齐，而靠 **semantic correspondence**。这篇 paper 的核心 insight 是：**patch distribution matching** 在 multiscale 下能 implicitly 建立 non-local 对应，而且 **training-free**。

---

## 2. Wasserstein Distance 的直觉与数学

### 2.1 1-Wasserstein Distance (Earth Mover's Distance)

公式 (1):
$$\mathrm{WD}(\mu, \nu) = \inf_{\gamma \in \Gamma(\mu, \nu)} \mathbb{E}_{(\mathbf{x}, \mathbf{y}) \sim \gamma} \|\mathbf{x} - \mathbf{y}\|_1$$

变量解释：
- $\mu, \nu$：两个 probability distributions
- $\Gamma(\mu, \nu)$：所有 marginal 是 $\mu$ 和 $\nu$ 的 **joint distributions (couplings)** 的集合
- $\gamma$：一个 specific coupling，即一个 transportation plan，告诉 how much mass 从 $\mathbf{x}$ 搬到 $\mathbf{y}$
- $\mathbf{x}, \mathbf{y}$：从 $\gamma$ 采样的 paired points
- $\|\mathbf{x} - \mathbf{y}\|_1$：transportation cost (用 $\ell_1$-distance)
- $\inf$：在所有 coupling 上取 infimum

直觉：imagine $\mu$ 是一堆泥土分布，$\nu$ 是要堆成的目标形状，$\gamma$ 是 "从哪里搬到哪里" 的 plan，WD 就是最小化 total transport cost。这就是 Earth Mover's Distance 的来源。

### 2.2 为什么 Wasserstein 比 KL/JS 好

Paper section 3.1 列了 4 个 advantages，我详细展开一下，因为这对你 build intuition 关键：

1. **Intuitive interpretation**: KL divergence 是 log-likelihood ratio，没有几何意义；WD 直接是搬运成本
2. **Sensitivity to distribution shape**: 两个不重叠的 distributions，KL 会爆炸 (无穷大)，JS 会饱和到 $\log 2$，但 WD 会给出 finite 且 meaningful 的 distance，反映 **几何距离**
3. **Robustness to support differences**: support 不重叠时 WD 依然 well-defined
4. **Smooth gradients**: KL divergence 在 GAN training 中有 vanishing gradient 问题，这正是 Arjovsky et al. (Wasserstein GAN, [4]) 解决的问题 - https://arxiv.org/abs/1701.07875

### 2.3 计算复杂度的 nightmare

直接计算 1-Wasserstein 在 empirical distributions 上需要 solving linear programming，复杂度 $\mathcal{O}(M^{2.5})$ (ref [36], Pitié et al.) - 这里 $M$ 是 sample 数。对于 image patch，比如 $256 \times 256$ image 有 $M \approx 60K$ patches，这个复杂度 prohibitive。

---

## 3. Sliced Wasserstein Distance - 精妙的降维

### 3.1 公式 (2) 详解

$$\mathrm{SWD}(U, V) = \mathbb{E}_{\mathbf{w} \sim \mathcal{U}(\mathbb{S}^{N-1})} \mathrm{WD}(U\mathbf{w}, V\mathbf{w})$$

变量解释：
- $U, V \in \mathbb{R}^{M \times N}$：两个 empirical distributions 的 sample matrices，$M$ 个 $N$-dim samples
- $\mathbb{S}^{N-1} := \{\mathbf{w} \in \mathbb{R}^{N \times 1} | \|\mathbf{w}\|_2^2 = 1\}$：$N$ 维空间中的 unit hyper-sphere
- $\mathcal{U}(\mathbb{S}^{N-1})$：hyper-sphere 上的 uniform distribution
- $\mathbf{w}$：random unit vector，是 projection direction
- $U\mathbf{w}$：把 $U$ 投影到 $\mathbf{w}$ 上，得到 $M$ 个 scalars (1D distribution)
- $\mathrm{WD}(U\mathbf{w}, V\mathbf{w})$：1D Wasserstein distance

### 3.2 直觉：Radon Transform 视角

SWD 本质上是 **Integral over Radon transform**。Radon transform 把 $N$-dim function 通过 projection 累积成 1D function。SWD 可以理解为：**沿所有方向投影，然后计算 1D Wasserstein，再平均**。

这正是 Rabin et al. [38] 的 key insight - https://link.springer.com/chapter/10.1007/978-3-642-31293-5_39

为什么 1D Wasserstein 可以高效计算？因为 1D 下，optimal coupling 就是 **sorted order matching**：
$$\mathrm{WD}_{1D}(U\mathbf{w}, V\mathbf{w}) = \frac{1}{M} \|\mathrm{sort}(U\mathbf{w}) - \mathrm{sort}(V\mathbf{w})\|_1$$

sort 的复杂度 $\mathcal{O}(M \log M)$，所以 SWD 总复杂度 $\mathcal{O}(P \cdot M \log M)$，其中 $P$ 是 projection 数量。这是 massive speedup。

### 3.3 sort() 的几何含义 - 这是 paper 的灵魂

Fig. 3 的 visualization 是关键。每个 random projection $\mathbf{w}$ 把 patch 投到一个 scalar，sort 后 **rank 相同的 patches** 就建立了 correspondence。

如果两图只在 color 上有差异，patch 的 structure 信息相近，那么投影到 random direction 后，**相似 patch 会落在相似的 rank 位置**，于是 sort-based matching 自然实现 **non-local patch correspondence**。

这其实和 Elnekave & Weiss [17] (https://arxiv.org/abs/2204.00073) 的 GPDM (Generative Patch Distribution Matching) 一脉相承，但目的不同：他们是要 generate natural images，这里是用同样机制度量 CD。

---

## 4. MS-SWD 完整 Pipeline

### 4.1 Algorithm 1 逐步分解

Input: 图像对 $(X, Y) \in \mathbb{R}^{H \times W \times 3}$，scale 数 $K=5$，projection 数 $P=128$

Step 3: **Gaussian pyramids** $\{X^{(i)}\}_{i=1}^K, \{Y^{(i)}\}_{i=1}^K$，downsample factor $R=2$
- $X^{(i)}, Y^{(i)} \in \mathbb{R}^{\lfloor H/2^{i-1} \rfloor \times \lfloor W/2^{i-1} \rfloor \times 3}$
- 这和 MS-SSIM (Wang, Simoncelli, Bovik [54] - https://ieeexplore.ieee.org/document/1292216) 的 multiscale 设计一致

Step 4: **sRGB → CIELAB conversion** - 这个很关键，后面会讲

Step 7-12: 对每个 scale $i$ 和每个 projection $j$:
- 采样 $\mathbf{w} \sim \mathcal{U}(\mathbb{S}^{N \times 3 - 1})$ - 注意这里是 $N \times 3$ 维，因为 patch 是 $\sqrt{N} \times \sqrt{N} \times 3$ (patch shape × 3 color channels)
- `unflat(w)`: 把 vector reshape 成 convolution kernel
- `Conv2d(X^{(i)}, w, 'reflect')`: **用 $\mathbf{w}$ 做卷积** - 这等价于 matrix multiply $X_{\mathrm{col}}^{(i)} \mathbf{w}^{(j)}$，但用 conv 更高效
- `flat()`: flatten 结果
- $\Delta E \leftarrow \Delta E + \frac{1}{M} \|\mathrm{sort}(\mathbf{x}) - \mathrm{sort}(\mathbf{y})\|_1$

Step 15: $\Delta E(X, Y) = \frac{1}{KP} \Delta E$ - 跨 scale 和 projection 的平均

### 4.2 公式 (3) 完整形式

$$\Delta E(\mathbf{X}, \mathbf{Y}) = \frac{1}{K} \sum_{i=1}^K \mathrm{SWD}\left(X_{\mathrm{col}}^{(i)}, Y_{\mathrm{col}}^{(i)}\right) = \frac{1}{KP} \sum_{i=1}^K \sum_{j=1}^P \mathrm{WD}\left(X_{\mathrm{col}}^{(i)} \mathbf{w}^{(j)}, Y_{\mathrm{col}}^{(i)} \mathbf{w}^{(j)}\right)$$

变量：
- $X_{\mathrm{col}}^{(i)} \in \mathbb{R}^{M \times (N \times 3)}$：第 $i$ 层的 patch matrix，$M$ 个 patch，每个 patch 是 $\sqrt{N} \times \sqrt{N} \times 3$ 拉平成 $N \times 3$ 维
- $\mathbf{w}^{(j)}$：第 $j$ 个 random unit projection
- $K=5$：scale 数 (from MS-SSIM)
- $P=128$：projection 数 (from GPDM [17])

### 4.3 img2col 的巧妙

$X_{\mathrm{col}}^{(i)}$ 通过 `img2col()` 操作得到，是卷积的标准实现技巧。但这里有更深含义：它把图像 patches 转化为 sample matrix，于是 SWD 可以直接套用。**Convolution = matrix multiply** 这件事让 random projection 实现得极其高效 - paper 强调 "the matrix multiplication $X_{\mathrm{col}}^{(i)} \mathbf{w}^{(j)}$ can be implemented by a single convolution"。

这个 trick 让我想到你 build micrograd 时的直觉 - 概念上简单的事情要落到 efficient implementation，往往需要利用 deep 数学结构的等价性。

---

## 5. Multiscale 的必要性 - Fig. 4 的消融

### 5.1 Reference Image Recovery 实验

Fig. 4 是 paper 中 build intuition 最关键的实验。从 Gaussian noise $Y_{\mathrm{init}}$ 出发，最小化 $\Delta E(X, Y)$ w.r.t. $Y$，看能否 recover reference $X$。

结果震撼：
- $K=1$ (single scale): 几乎是 noise，无结构
- $K=2, 3$: 出现一些模糊的 color blob
- $K=4$: 开始有粗略 structure
- $K=5$ (default): 完整 recover reference
- $K=6$: 几乎 identical

### 5.2 为什么 single scale 不够？

Single scale SWD 只关心 patch distribution，不关心 patch 的 spatial arrangement。**任何 spatial permutation** 都不会改变 single scale SWD 的值 - 这正是为什么 $K=1$ 几乎是 noise。

Multiscale 的作用：通过 downsample，**larger patch 在更粗 scale 上对应于更大 spatial context**。当 $K=5$ 时，最粗 scale 的 patch 几乎覆盖整张图像，强约束 global color distribution；最细 scale 约束 local structure。这种 **hierarchical constraint stacking** 才能锁死 pixel-level fidelity。

这和 Laplacian pyramid 的思想一致 (Burt & Adelson [8] - https://ieeexplore.ieee.org/document/1095851) - 多分辨率分析能 capture 不同 scale 的信息。也和你 build nanoGPT 时 stack transformer layers 锁住不同 abstraction level 的感觉有共鸣。

### 5.3 与 GAN 的联系

Karras et al. progressive growing GAN [23] (https://arxiv.org/abs/1710.10196) 也用 SWD 作为 evaluation metric - 他们把 SWD 用在 wavelet domain 上评估生成图像质量。这篇 paper 把 SWD 直接用作 distance metric 本身，且在 pixel/CIELAB domain。两个工作的共通 insight：**multiscale + SWD** 是 capture image perceptual structure 的 powerful combination。

---

## 6. CIELAB 的选择 - 为什么不是 sRGB 或 deep features

Paper 在 section 3.2 简短说 "we observe significant performance gains over the sRGB color space"，但没给详细 ablation。这里我从 background 补充：

**CIELAB 设计**：CIE 在 1976 提出，意图是 perceptually uniform color space。$L^*$ 是 lightness (0=黑, 100=白)，$a^*$ 是 green-red opponent，$b^*$ 是 blue-yellow opponent。设计上 CIELAB 的 Euclidean distance 近似对应 perceptual color difference。

**为什么不用 sRGB**：sRGB 是 gamma-encoded，perceptually non-uniform，相同 Euclidean distance 在 dark region 和 bright region 对应的 perceived difference 差很多。

**为什么不用 deep features (VGG, like LPIPS [56])**：
- LPIPS (https://arxiv.org/abs/1801.03924) 在 co-located 比较，对 misalignment 极敏感 (Table 1 中 non-aligned PLCC=0.272)
- Deep features encode semantic content，但 color 信息会被部分丢弃 (ReLU activation 把 negative 全砍掉)
- Training-free 的 CIELAB 反而更 generalizable

**S-CIELAB 的失败**：S-CIELAB 是 spatial-extended CIELAB (Zhang & Wandell [57] - https://sid.onlinelibrary.wiley.com/doi/abs/10.1889/1.1837695)，用 CSF-based lowpass filter 作 preprocessing。但它仍然是 co-located comparison，对 misalignment 敏感 (Table 1 中 non-aligned PLCC=0.627)。说明 spatial filter 解决了 "single pixel comparison is too local" 的问题，没解决 misalignment。

---

## 7. 实验深度分析

### 7.1 SPCD Dataset

SPCD (Wang et al. [52] - https://ieeexplore.ieee.org/document/10037906) 是这个领域的 game-changer：
- 30,000 photographic image pairs
- 10,005 non-perfectly aligned (6 flagship smartphones 实拍)
- 20,000 perfectly aligned (simulated color alteration)
- Diverse: foreground, background complexity, lighting, weather, camera modes

### 7.2 Table 1 - 主实验结果

| Method | Aligned PLCC | Non-aligned PLCC | All PLCC |
|---|---|---|---|
| CIEDE2000 | 0.827 | 0.653 | 0.725 |
| S-CIELAB | 0.824 | 0.627 | 0.699 |
| LPIPS | 0.767 | **0.272** | 0.448 |
| CD-Flow (60.49M params) | - | - | 0.871 (从 ref [9] 知) |
| **MS-SWD** | **0.778** | **0.841** | **0.794** |

**关键观察**：
1. LPIPS 在 non-aligned 上几乎崩溃 (0.272)，证明 deep features 对 misalignment 极敏感
2. CIEDE2000 在 aligned 上比 MS-SWD 好 (0.827 vs 0.778)，但在 non-aligned 上差很多 (0.653 vs 0.841)
3. MS-SWD 是 training-free，但 overall PLCC=0.794 已经超过绝大多数 trained methods
4. Table 4 中 learned MS-SWD (just 0.05M params, 比 CD-Flow 少 1000x) 达到 overall PLCC=0.884

**Build intuition**: 这个结果表明，对于 color perception，**正确的 inductive bias (multiscale + distribution matching + CIELAB)** 比 brute-force deep learning 更 data-efficient。这其实呼应了你关于 neural network 设计中 inductive bias 重要性的观点。

### 7.3 Table 2 - Geometric Transformation Robustness

这个实验更狠：
- Translation (5% pixels): MS-SWD PLCC=0.836 vs CIEDE2000 PLCC=0.377
- Dilation (1.1x): MS-SWD PLCC=0.833 vs CIEDE2000 PLCC=0.362
- Flipping (horizontal): **MS-SWD PLCC=0.836** vs CIEDE2000 PLCC=0.170

注意 Flipping 是非常苛刻的 test - semantic content 完全保留，但所有 spatial correspondence 反转。Co-located methods 全部崩溃，但 MS-SWD 因为是基于 patch distribution (与 spatial arrangement 无关)，几乎没影响。甚至 MS-SWD 在 Flipping 上 (PLCC=0.836) 比 translation (PLCC=0.836) 一致。

### 7.4 Table 3 - Random Projection 数量消融

| P | STRESS↓ | PLCC↑ | SRCC↑ | Time (ms) |
|---|---|---|---|---|
| 4 | 31.849 | 0.804 | 0.779 | 3.7 |
| 16 | 29.186 | 0.833 | 0.799 | 4.2 |
| 64 | 28.425 | 0.841 | 0.805 | 6.2 |
| **128** | **28.363** | **0.841** | **0.805** | 9.5 |
| 256 | 28.318 | 0.842 | 0.806 | 15.3 |

从 $P=64$ 到 $P=128$ 性能 gain 已 marginal，但 $P$ 太小会破坏 metric property (不能保证 identity of indiscernibles)。这和 Monte Carlo approximation 的 variance-$\sqrt{P}$ scaling 一致。

---

## 8. Metric Property 的 Empirical Verification

数学上 SWD 是 metric，但 multiscale extension 严格说是否还是 metric 需要 verify。Paper 用 computational experiment 验证四条：

1. **Non-negativity**: $\Delta E(X, Y) \geq 0$ - 从公式 (3) 直接成立
2. **Symmetry**: $\Delta E(X, Y) = \Delta E(Y, X)$ - 公式对称
3. **Identity of indiscernibles**: $\Delta E(X, Y) = 0 \iff X = Y$ - 用 Fig. 4 的 reference recovery 验证
4. **Triangle inequality**: $\Delta E(X, Y) \leq \Delta E(X, Z) + \Delta E(Z, Y)$ - 在 100,000 random triplets 上测试，no violation

这是 paper 的 strong claim - 不仅好用，还是 mathematical metric，可以作为 optimization loss。这与 KL divergence (not metric) 形成对比，给 perceptual optimization 带来稳定收敛保证。

---

## 9. Color Transfer 应用

### 9.1 公式 (6) - Optimization Formulation

$$Y^\star = \arg\min_{\mathbf{Y}} \Delta E(X, Y)$$

给定 source color image $X$，把 color appearance transfer 到 target $Y_{\mathrm{init}}$。这是 pure gradient descent，不需要 GAN。

### 9.2 Fig. 6, 7 结果

Image color transfer (Fig. 6) 和 video color transfer (Fig. 7) 都很 compelling。对比：
- CIELAB transfer: 把 unwanted structure 也 transfer 了
- DISTS: 经常不动 target
- MS-SWD: color 准确 transfer，structure 保留，video 时 temporal consistency 也好

**为什么 MS-SWD 在 video 上稳定？** 因为每个 frame 独立优化，但 SWD 是 statistical distance，frame 间的统计性变化比 pixel-level 变化更平滑。这其实是 implicit 的 temporal regularization。

### 9.3 与 Style Transfer 的对比

这个应用让我想到 Gatys et al. neural style transfer (https://arxiv.org/abs/1508.06576) - 用 Gram matrix matching 实现 style transfer。MS-SWD 用 patch distribution matching，conceptually similar 但更轻量 (no VGG, no Gram matrix computation)。这种 **distribution matching 替代 feature matching** 的思路在 generative modeling 中很 powerful。

---

## 10. 联想与更深讨论

### 10.1 与 Optimal Transport 的更深联系

SWD 是 **entropic-regularized optimal transport** 的特殊形式。Cuturi 2013 (Sinkhorn distance) 把 OT 用 entropy regularization 让计算可微。SWD 用 slicing 避免 regularization，但牺牲一些 OT 的几何精度换取计算效率。

最近几年 neural OT (e.g., Neural OT by Makkuva et al. - https://arxiv.org/abs/1910.00118) 用 neural network parameterize transport plan。MS-SWD 的反方向：用 random projection + sort 完全避免 learning，但依然 capture distribution geometry。

### 10.2 与 Diffusion Models 的潜在联系

Diffusion model 的 forward/reverse process 本质是 OT 问题的 stochastic 实现。SWD 评估 patch distribution 相似性，可以看作 score matching 在 sliced space 的近似。如果有 diffusion-based color transfer，用 MS-SWD 作为 guidance term 应该会 work。

### 10.3 Karpathy 你应该感兴趣的设计哲学

这篇 paper 有几个 design choices 我觉得你会 appreciate:

**1. Training-free 的力量**: 在 deep learning era，这篇 paper 用 random projection + sort 实现了 SOTA (在 non-aligned 场景)。这印证了你的观点 - 好的 inductive bias + 简单 algorithm 能 beat 复杂 model。这让人想到 nanoGPT 的哲学：用最简洁 implementation 揭示 core mechanism。

**2. Multiscale as inductive bias**: $K=5$ 直接从 MS-SSIM 继承，没有任何 tuning。这种 "borrow established hyperparameters" 比 grid search 更 principled，因为它们是经过 human visual perception research 验证的。

**3. Sort-based correspondence**: 用 sort() 替代 explicit patch matching (PatchMatch [5] - https://dl.acm.org/doi/10.1145/1531326.1531330) 是 brilliant move。Sort 把 nearest neighbor search 复杂度从 $\mathcal{O}(M^2)$ 降到 $\mathcal{O}(M \log M)$，且 implicitly 实现了 **bidirectional correspondence** (sort 后两端 rank 对应，天然 symmetric)。

### 10.4 可能的改进方向

Paper 自己列出几个 future work：
1. **Strict metric proof**: 现在是 empirical verification，但 strict mathematical proof 缺失
2. **Alternative pyramids**: Laplacian, steerable pyramid, VGG feature hierarchy
3. **Extension to other perceptual aspects**: image quality, texture similarity

我加几个 Karpathy-flavored 的：
1. **Self-supervised learned projections**: 用 contrastive learning 在 large image corpus 上学习 projection directions，可能比 random 更 efficient
2. **Diffusion-guided CD**: 用 diffusion model 的 score function 修正 SWD 在 high-frequency detail 上的盲区
3. **Video temporal SWD**: extend 到 3D patches (spatial + temporal)，应该能更好 capture video CD

### 10.5 Limitations 我看到的

Paper 没强调的几个 limitation:
1. **Patch size 固定 $\sqrt{N} = 11$**: 这是 MS-SSIM 的继承，但 photographic image 的 perceptual scale 应该 content-adaptive
2. **CIELAB 在 wide gamut 上不够**: HDR, wide gamut content 需要 CIECAM02-16 或 new color appearance models
3. **No learned color space**: 最新 research (如 Wang et al. [52]) 表明 learned color space 可以 beat hand-crafted

---

## 11. 总结 - 核心 Takeaways

1. **核心 insight**: Photographic image 的 perceptual CD 评估需要 **non-local patch distribution matching**，因为 human color perception 是 unitary process 把 color + structure 绑定
2. **方法 elegant**: Multiscale + Sliced Wasserstein + CIELAB，三件事组合，全部 training-free
3. **Sort operator 是灵魂**: 把 high-dim non-local matching 通过 random projection + sort 实现成 $\mathcal{O}(M \log M)$ 复杂度
4. **Empirical metric**: 不仅 useful，还是 mathematical metric，可以作 loss
5. **对 misalignment 鲁棒**: horizontal flip 都 PLCC=0.836，碾压所有现有方法

---

## References

- Paper GitHub: https://github.com/real-hjq/MS-SWD
- Wasserstein GAN (Arjovsky et al.): https://arxiv.org/abs/1701.07875
- Sliced Wasserstein (Rabin et al.): https://link.springer.com/chapter/10.1007/978-3-642-31293-5_39
- GPDM (Elnekave & Weiss): https://arxiv.org/abs/2204.00073
- SPCD Dataset (Wang et al.): https://ieeexplore.ieee.org/document/10037906
- MS-SSIM (Wang, Simoncelli, Bovik): https://ieeexplore.ieee.org/document/1292216
- LPIPS (Zhang et al.): https://arxiv.org/abs/1801.03924
- Progressive GAN (Karras et al.): https://arxiv.org/abs/1710.10196
- Neural Style Transfer (Gatys et al.): https://arxiv.org/abs/1508.06576
- PatchMatch (Barnes et al.): https://dl.acm.org/doi/10.1145/1531326.1531330
- CIEDE2000 (Luo et al.): https://onlinelibrary.wiley.com/doi/abs/10.1002/1520-6378(200110)26:5%3C340::AID-COL6%3E3.0.CO;2-7
- S-CIELAB (Zhang & Wandell): https://sid.onlinelibrary.wiley.com/doi/abs/10.1889/1.1837695
- Neural OT (Makkuva et al.): https://arxiv.org/abs/1910.00118
- SinGAN (Shaham et al.): https://arxiv.org/abs/1903.07222
- InGAN (Shocher et al.): https://arxiv.org/abs/1812.00231
- CD-Flow (Chen et al., 2023): https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Learning_a_Deep_Color_Difference_Metric_for_Photographic_Images_CVPR_2023_paper.pdf

Karpathy, 这篇 paper 的 beauty 在于：它把一个看似需要 deep learning 的 perceptual problem，用 optimal transport + multiscale analysis 优雅地降维成 training-free algorithm，且效果超过 60M 参数的 CD-Flow (在 robustness 场景)。这种 work 让我想到你说的 "看一个方法是不是真的理解了 problem" - 这篇 paper 真的理解了 color perception 在 misalignment 下的本质。
