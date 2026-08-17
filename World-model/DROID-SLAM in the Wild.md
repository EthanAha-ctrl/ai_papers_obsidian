---
source_pdf: DROID-SLAM in the Wild.pdf
paper_sha256: 105e2595d9496ec51c71c8ae31a62197a2fac7ee75a3028fa4bfd9f95612592e
processed_at: '2026-08-04T00:16:31-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

嘿 Andrej，既然你要我用大白话把这篇 paper 捋一遍，那我们就抛开那些学术八股文，直接切进它的灵魂深处。我会把里面的数学公式和工程实现细节全部拆解成你可以直接在脑子里跑通的直觉。

## 1. 这个系统到底在解决什么痛点？

传统的 SLAM (比如 DROID-SLAM) 跑在一个很 core 的假设上：**世界是静止的**。
一旦画面里走过一个人，或者你拿着相机在街上拍，画面里全是车在动，这个假设就崩了。相机会把移动的人当成静止的路标去算几何，结果就是算出来的 camera pose 乱飞，建出来的 map 全是重影。

之前的人怎么解决这个问题？两条路，都有致命缺陷：
- **暴力擦除法**：用神经网络检测人或车，直接把这些 pixel 从画面里抠掉。缺陷是，万一跑过一只狗、一片被风吹动的树叶、或者一个没见过的变形金刚呢？你的 detector 没学过这个，它就不抠，系统就崩了。
- **建图对比法**：像 WildGS-SLAM 那样，先建一个超精细的 3D Gaussian 地图，然后把当前画面和地图渲染出来的画面对比，哪个 pixel 对不上，哪个就是动态的。缺陷是，在极度混乱的场景下，地图本身就建得乱七八糟，你怎么拿一个错误的地图去鉴定谁在动？

## 2. DROID-W 的核心 Idea：用 Semantic Consistency 解耦 Geometric Mapping

这篇 paper 最漂亮的地方就在于它的 insight：**放弃几何，拥抱语义。**

作者发现，如果一个 pixel 属于静态背景，无论相机怎么动，这个 pixel 在不同帧里的 DINOv2 feature 肯定是极度一致的。如果一个 pixel 属于动态物体，它在不同帧里的 feature 肯定会变（因为视角变了、形变了、或者被遮挡了）。

所以，DROID-W 压根就不去建什么完美的 3D 地图来辅助找动态物体。它直接提取相邻帧的 DINOv2 features，算它们的 cosine similarity。如果 features 对不上，这个 pixel 的 "uncertainty" 就高。然后在算 camera pose 的时候，直接把 uncertainty 高的 pixel 权重打到极低。

这就把一个极难的几何问题，转化成了一个极其鲁棒的语义 feature 比对问题。

## 3. 数学公式直击：大白话拆解

为了让你 build intuition，我们把核心公式拆开来嚼。

### 3.1 Uncertainty 如何影响 Bundle Adjustment？

在 DROID-SLAM 里，优化的目标是让 reprojection residual 最小：
$$\mathbf{E} = \sum \big\| \mathbf{p}_{ij}^* - \mathbf{p}_{ij} \big\|_{\Sigma_{ij}}^2$$

DROID-W 在这里加了点料，改成了 Uncertainty-aware 的 Mahalanobis distance（公式4）：
$$\pmb{\Sigma}_{ij}^{\text{uncer}} = \text{diag} \Big( \mathbf{w}_{ij} \cdot \frac{1}{\mathbf{u}_i'} \Big)$$

**变量拆解：**
- $\mathbf{w}_{ij}$: DROID-SLAM 自己预测的 confidence map（原版就有的）。
- $\mathbf{u}_i'$: 帧 $i$ 当前 pixel 的 dynamic uncertainty。
- 取倒数 $\frac{1}{\mathbf{u}_i'}$ 意味着：uncertainty 越大，权重越小。如果这个 pixel 在动，它的 $\mathbf{u}_i'$ 就会很大，导致它在 BA 里基本没贡献，不会把相机位姿带偏。

### 3.2 Uncertainty 本身怎么更新？（公式6）

这是全文的灵魂。作者用 DINOv2 feature 的 cosine similarity 来驱动 uncertainty 更新：
$$\mathbf{E}_{\text{sim}}(\mathbf{u}') = \sum_{(i,j) \in \mathcal{E}} \frac{1 - \frac{\mathbf{F}_i \cdot \mathbf{F}_{ij}}{\|\mathbf{F}_i\|_2 \|\mathbf{F}_{ij}\|_2}}{\mathbf{u}_i' \cdot \mathbf{u}_{ij}'}$$

**变量拆解：**
- $\mathbf{F}_i$: 帧 $i$ 在某个 pixel 上的 DINOv2 feature。
- $\mathbf{F}_{ij}$: 把帧 $i$ 的 pixel 通过当前的 pose 和 depth 投影到帧 $j$ 上，用 bilinear interpolation 采出来的对应 feature。
- 分子 $1 - \cos(\mathbf{F}_i, \mathbf{F}_{ij})$: 两个 frame 之间 feature 的不一致度。静态背景这个值接近 0，动态物体这个值很大。
- 分母 $\mathbf{u}_i' \cdot \mathbf{u}_{ij}'$: 两个 frame 在这个对应位置的 uncertainty 乘积。这就是 **bidirectional decoupling**。

**大脑模拟运行过程：**
假设画面里有一面墙和一辆飞驰的车。
- **对于墙**：分子 $1 - \cos$ 接近 0。这时候优化器如果给墙很高的 uncertainty（让分母变大），对降低总 loss 没什么帮助，徒增惩罚（看下面的 prior）。
- **对于车**：分子 $1 - \cos$ 很大。这时候 loss 很大。优化器为了让 loss 降下来，就会拼命把分母 $\mathbf{u}_i'$ 和 $\mathbf{u}_{ij}'$ 推大。车被成功标记为高 uncertainty。

**为什么要 bidirectional decoupling（双向解耦）？**
如果分母只用 $\mathbf{u}_i'^2$（见公式10 ablation），那么如果帧 $i$ 的车被标记为高 uncertainty，连带着帧 $j$ 的对应 pixel 也被强制认为是高 uncertainty。但这可能不对。比如帧 $i$ 的车被挡住了，但帧 $j$ 的车没被挡住。用 $\mathbf{u}_i' \cdot \mathbf{u}_{ij}'$ 就允许两帧的 uncertainty 独立生长，更精确地刻画了动态关系。

### 3.3 防止“偷懒”的 Logarithmic Prior（公式7）

优化器是很狡猾的。如果只有上面的 $\mathbf{E}_{\text{sim}}$，优化器会发现一个捷径：把所有 pixel 的 uncertainty 都推到无穷大，这样分母无穷大，整个 loss 直接变成 0。

所以必须加一个 prior 来按住它：
$$\mathbf{E}_{\text{prior}}(\mathbf{u}') = \sum_i \log(\mathbf{u}_i' + 1.0)$$

**变量拆解：**
- $\mathbf{u}_i'$: 当前 uncertainty。
- $+1.0$: 防止 $\log(0)$ 爆炸，也防止 loss 变成负数。

**直觉：** $\log$ 函数增长得越来越慢。你想把 uncertainty 推到 100？可以，你要付出 $\log(101) \approx 4.6$ 的代价。推到 1000？要付出 $\log(1001) \approx 6.9$ 的代价。这就像一个弹簧，uncertainty 越大，往回拉的力就越强。只有当分子（feature 不一致度）大到足以证明这个代价是值得的时候，系统才会把 uncertainty 推高。

## 4. 工程优化细节：Affine Mapping 与 Weight Decay

如果直接对每个 pixel 的 $\mathbf{u}$ 进行优化，画面会变成雪花点，一个 pixel 说自己动，隔壁 pixel 说自己没动，这在几何上是不合理的。

作者搞了个极其聪明的 trick：**Local Affine Mapping**。
$$\mathbf{u} = \text{Softplus}(\boldsymbol{\theta} \cdot \mathbf{F})$$

这其实就是一个没有非线性隐藏层的单层线性变换。
- $\mathbf{F}$: DINOv2 feature。它本身带有极强的 spatial smoothness。
- $\boldsymbol{\theta}$: 可学习参数。
- 因为是线性变换，feature 的 spatial smoothness 就直接传递给了 uncertainty。画面上的 uncertainty 边界会很自然，不会出现椒盐噪声。
- 优化 $\boldsymbol{\theta}$ 的时候，还加了 weight decay：$\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \lambda \cdot \mathbf{g}_t - \eta \cdot \boldsymbol{\theta}_{t-1}$。防止参数数值爆炸导致系统不稳定。

## 5. Depth Prior 救命稻草（公式10）

在极度动态的场景下，比如画面 90% 都是人，相机一开始就懵了，根本找不到 rigid-motion correspondence 怎么初始化？

作者引入了 Metric3D v2 预测的 monocular metric depth $\mathbf{D}_i$ 作为 anchor：
$$\mathbf{E}^+(\mathbf{G}', \mathbf{d}') = \sum \left\| \mathbf{p}_{ij}^* - \mathbf{p}_{ij} \right\|_{\Sigma_{ij}^{\text{uncer}}}^2 + \gamma_d \sum_i \left\| \mathbf{d}_i' - \mathbf{D}_i \right\|^2$$

**直觉：** 当画面全在动，reprojection residual 就全废了。这时候 BA 层就会被 $\gamma_d \sum \left\| \mathbf{d}_i' - \mathbf{D}_i \right\|^2$ 这一项接管。它告诉系统：“既然你看不懂物体在怎么动，那就先相信单目深度网络给你的深度估计，起码别让 depth 乱飘。” 这个设计在初始化阶段简直就是保命的。

## 6. 实验数据表解读：它到底赢在哪里？

看 Table 1 (Bonn RGB-D Dynamic Dataset)：

| Method | Balloon | Crowd | Person | Moving | Avg |
|--------|---------|-------|--------|--------|-----|
| DROID-SLAM | 7.5 | 5.2 | 4.3 | 2.3 | 4.91 |
| DynaMoN (MS&SS) | 2.8 | 3.5 | 14.8 | 1.3 | 4.10 |
| WildGS-SLAM | 2.8 | 1.6 | 3.9 | 1.7 | 2.52 |
| **DROID-W** | **2.6** | **1.3** | **3.3** | **1.6** | **2.30** |

- 对比 DROID-SLAM，误差直接腰斩。证明这套 uncertainty 机制把动态干扰剔除得极其干净。
- 对比 DynaMoN (依赖 motion segmentation)，在 Person 序列上 DynaMoN 崩了 (14.8 cm)，因为人的动作变形太诡异，segmentation 网络分割不好。DROID-W 稳稳的 3.3 cm，说明 feature similarity 完全无视了物体类别，只看一致性。

看 Table 5 (Runtime)：
- WildGS-SLAM: 0.22 FPS (基本等于跑不动，算一帧要 4-5 秒)。
- DROID-W: 10.57 FPS (勉强 Real-time)。
速度快了 40 倍。原因很简单：WildGS-SLAM 要不停地去渲染、更新庞大的 3D Gaussian 地图来算 loss。DROID-W 只是做 matrix 乘法算 cosine similarity，用 gradient descent 更新一个小小的 affine 矩阵，这计算量根本不是一个量级的。

## 7. 给你的 Intuition 总结与发散

1. **Geometric SLAM 的天花板已经被 semantic features 打破了。** 以前我们觉得 SLAM 是纯几何问题，要算极线、算三角化。现在看来，只要你的 feature 足够强（DINOv2/FiT3D），feature consistency 可以直接替代几何 consistency 来做 outlier rejection。
2. **Decoupling 是关键。** 之前 dynamic SLAM 失败的根因是 coupling 太深。Mapping 烂 -> Uncertainty 烂 -> Tracking 烂 -> Mapping 更烂，死循环。DROID-W 把 Uncertainty 直接挂在 Feature 上，Tracking 挂在 Uncertainty 上，Map 挂在 Tracking 上，哪怕最后建出来的 Map 有点烂，前面的 Tracking 和 Uncertainty 依然是自洽且准确的。
3. **下一步可以做啥？** 你在搞自动驾驶和具身智能，这套东西直接就能用。比如机器人在人群中导航，或者车在拥挤的街道上行驶。如果在这套 UBA 的基础上，把 DINOv2 换成更具有时序感知的 feature（比如 VideoMAE 或者直接用 SAM2 的 mask feature），uncertainty 的估计甚至可以提前预测（物体还没动，系统就知道它要动了），那机器人就真的具有了“预判”能力。

**参考链接：**
- DROID-SLAM (基础架构): https://github.com/princeton-vl/DROID-SLAM
- DINOv2 (Semantic Backbone): https://arxiv.org/abs/2304.07193
- Metric3D v2 (Depth Prior): https://arxiv.org/abs/2404.15506
- FiT3D (Refined DINOv2): https://arxiv.org/abs/2407.20229
- Bonn RGB-D Dynamic Dataset: https://www.ipb.uni-bonn.de/data/dynamic-rgbd-benchmark/
- 这篇论文的官方代码: https://github.com/MoyangLi00/DROID-W.git

---

# DROID-SLAM in the Wild 深度技术解析

## 一、核心问题定位

这篇paper解决的核心问题是：**DROID-SLAM在动态环境下的失效**。DROID-SLAM的differentiable bundle adjustment (BA) layer基于rigid-motion假设，当场景中存在dynamic objects时，reprojection residuals变得不可靠，导致pose estimation和depth estimation同时崩塌。

传统解决方案有两类：
1. **Detection/segmentation-based**：DynaSLAM, DS-SLAM, Co-Fusion, MaskFusion等，依赖predefined motion priors，对unseen object classes失效
2. **Uncertainty-aware mapping-based**：WildGS-SLAM, UP-SLAM等，使用shallow MLP从DINO features预测uncertainty，但optimization tightly coupled with Gaussian Splatting mapping，mapping quality差时uncertainty estimation也崩

DROID-W的核心insight：**将uncertainty estimation从geometric mapping中解耦，直接利用multi-view visual feature similarity来驱动uncertainty optimization**。

## 二、系统架构解析

### 2.1 整体pipeline（对应Figure 2）

```
RGB sequence {I_t}
      ↓
┌─────────────────────────────────────────┐
│  Feature Extraction (FiT3D/DINOv2)      │
│  Metric Depth Prediction (Metric3D)     │
└─────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────┐
│  Iterative Optimization Loop:            │
│  ┌─────────────────┐  ┌──────────────┐  │
│  │ Pose-Depth Refine│←→│ Uncertainty  │  │
│  │ (UBA + depth reg)│  │ Optimization │  │
│  └─────────────────┘  └──────────────┘  │
└─────────────────────────────────────────┘
      ↓
Camera poses {G_t}, inverse depths {d_t}, uncertainties {u_t}
      ↓
Global BA (freeze uncertainty)
      ↓
Dynamic point cloud + camera trajectory
```

### 2.2 状态变量定义

对input sequence $\{I_t\}_{t=0}^N$，系统维护：
- **Camera pose**: $\mathbf{G}_t \in SE(3)$ — 每帧的6-DoF位姿
- **Inverse depth**: $\mathbf{d}_t \in \mathcal{R}^{\frac{H}{8} \times \frac{W}{8}}$ — 在1/8分辨率下的逆深度图（计算效率与精度权衡）
- **Per-pixel uncertainty**: $\mathbf{u}_t \in \mathcal{R}^{\frac{H}{8} \times \frac{W}{8}}$ — 动态uncertainty map
- **Frame graph**: $(\mathcal{V}, \mathcal{E})$ — co-visibility graph，edge $(i,j) \in \mathcal{E}$ 表示帧i和j有overlap

## 三、DROID-SLAM Preliminaries

### 3.1 Rigid-motion correspondence (公式1)

$$\mathbf{p}_{ij} = \Pi_c \Big( \mathbf{G}'_{ij} \circ \Pi_c^{-1} \big( \mathbf{p}_i, \mathbf{d}'_i \big) \Big)$$

变量含义：
- $\Pi_c$: camera projection function（3D点→2D像素）
- $\Pi_c^{-1}$: inverse projection（2D像素+depth→3D点）
- $\mathbf{G}'_{ij}$: 帧$i$到帧$j$的relative pose（current estimate）
- $\mathbf{p}_i \in \mathcal{R}^{\frac{H}{8} \times \frac{W}{8} \times 2}$: 帧$i$的pixel grid坐标
- $\mathbf{d}'_i$: 帧$i$的current inverse depth estimate

物理含义：给定帧$i$的pixel和depth，先反投影到3D空间，再用relative pose变换到帧$j$坐标系，再投影到帧$j$的image plane。这就是rigid-motion假设下的correspondence prediction。

### 3.2 Differentiable BA (公式2-3)

Energy function：
$$\mathbf{E}(\mathbf{G}', \mathbf{d}') = \sum_{(i,j) \in \mathcal{E}} \big\| \mathbf{p}_{ij}^* - \mathbf{p}_{ij} \big\|_{\Sigma_{ij}}^2$$

其中：
- $\mathbf{p}_{ij}^*$: DROID-SLAM的update operator预测的dense correspondence（learned）
- $\mathbf{p}_{ij}$: 公式1推导的rigid-motion correspondence
- $\Sigma_{ij} = \text{diag}(\mathbf{w}_{ij})$: Mahalanobis distance的weight matrix
- $\mathbf{w}_{ij}$: DROID-SLAM预测的confidence map

Gauss-Newton更新通过Schur complement求解：
$$\begin{bmatrix} \mathbf{B} & \mathbf{E} \\ \mathbf{E}^\top & \mathbf{C} \end{bmatrix} \begin{bmatrix} \Delta \boldsymbol{\xi} \\ \Delta \mathbf{d} \end{bmatrix} = \begin{bmatrix} \mathbf{v} \\ \mathbf{w} \end{bmatrix}$$

- $\Delta \boldsymbol{\xi}$: pose update (se(3) tangent space)
- $\Delta \mathbf{d}$: inverse depth update
- $\mathbf{B}$: pose-pose block (Hessian)
- $\mathbf{C}$: depth-depth block（**对角矩阵**，因为每个depth residual只依赖一个depth值）
- $\mathbf{E}$: pose-depth cross block

由于$\mathbf{C}$是对角的，可以直接element-wise求逆 $\mathbf{C}^{-1} = 1/\mathbf{C}$，然后用Schur complement：
$$\Delta \boldsymbol{\xi} = [\mathbf{B} - \mathbf{E}\mathbf{C}^{-1}\mathbf{E}^\top]^{-1} (\mathbf{v} - \mathbf{E}\mathbf{C}^{-1}\mathbf{w})$$
$$\Delta \mathbf{d} = \mathbf{C}^{-1} (\mathbf{w} - \mathbf{E}^\top \Delta \boldsymbol{\xi})$$

这个设计的计算复杂度关键在于$\mathbf{C}$的对角性，避免了full Hessian inversion。

## 四、Uncertainty-aware Bundle Adjustment (UBA)

### 4.1 关键insight

Dynamic objects违反rigid-motion假设，导致$\mathbf{p}_{ij}^* - \mathbf{p}_{ij}$的residual不可靠。解决方案：引入per-pixel uncertainty $\mathbf{u}_t$，对dynamic regions的residual降权。

### 4.2 Uncertainty-aware Mahalanobis distance (公式4)

$$\pmb{\Sigma}_{ij}^{\text{uncer}} = \text{diag} \Big( \mathbf{w}_{ij} \cdot \frac{1}{\mathbf{u}_i'} \Big)$$

变量含义：
- $\mathbf{w}_{ij}$: DROID-SLAM原生的correspondence confidence
- $\mathbf{u}_i'$: 帧$i$的current uncertainty estimate
- 取$1/\mathbf{u}_i'$意味着**uncertainty越大，weight越小，residual贡献越低**

### 4.3 Interleaved optimization

为什么不jointly optimize？因为joint Gauss-Newton on (pose, depth, uncertainty)的Hessian矩阵规模巨大，计算prohibitive。所以采用alternating optimization：

**Step 1: Pose-depth refinement** (固定uncertainty)
$$\hat{\mathbf{E}}(\mathbf{G}', \mathbf{d}') = \sum_{(i,j) \in \mathcal{E}} \left\| \mathbf{p}_{ij}^* - \mathbf{p}_{ij} \right\|_{\Sigma_{ij}^{\text{uncer}}}^2$$

**Step 2: Uncertainty optimization** (固定pose-depth)，详见下节。

## 五、Uncertainty Optimization（核心创新）

### 5.1 为什么不用reprojection residual？

在dynamic scenes中，dynamic objects的reprojection error本身就可信度低。用reprojection residual去optimize uncertainty会陷入chicken-and-egg问题：uncertainty错→residual不可靠→uncertainty更新错。

**Solution**: 使用DINOv2 features的multi-view consistency，因为：
1. DINOv2 features对appearance variation robust
2. 提供semantic-level信息
3. 不依赖geometric reconstruction quality

### 5.2 Feature extraction

使用**FiT3D**（refined DINOv2 model，来自[61] Yue et al. ECCV 2024）提取2D visual features $(\mathbf{F}_i, \mathbf{F}_j)$。FiT3D通过3D-aware fine-tuning改进了DINOv2的feature quality用于几何任务。

### 5.3 Similarity loss (公式6)

$$\mathbf{E}_{\text{sim}}(\mathbf{u}') = \sum_{(i,j) \in \mathcal{E}} \frac{1 - \frac{\mathbf{F}_i \cdot \mathbf{F}_{ij}}{\|\mathbf{F}_i\|_2 \|\mathbf{F}_{ij}\|_2}}{\mathbf{u}_i' \cdot \mathbf{u}_{ij}'}$$

变量含义：
- $\mathbf{F}_i$: 帧$i$在pixel $\mathbf{p}_i$处的DINOv2 feature
- $\mathbf{F}_{ij}$: 通过bilinear interpolation从帧$j$的$\mathbf{p}_{ij}$位置采样的feature
- 分子: $1 - \cos(\mathbf{F}_i, \mathbf{F}_{ij})$，feature dissimilarity
- 分母: $\mathbf{u}_i' \cdot \mathbf{u}_{ij}'$，**bidirectional uncertainty**

**Bidirectional decoupling的关键意义**：
- 如果只用$\mathbf{u}_i'^2$（公式10的ablation版本），则两个帧的uncertainty强耦合
- 使用$\mathbf{u}_i' \cdot \mathbf{u}_{ij}'$允许两个帧的dynamic regions独立建模
- 例如：帧$i$中静态物体在帧$j$中被动态物体遮挡，应当$\mathbf{u}_j'$高而$\mathbf{u}_i'$可以低

**物理直觉**：
- 静态区域的DINOv2 features在multi-view间应该consistent（cosine similarity高）
- 动态物体由于motion、occlusion、view-dependent效应，features inconsistent
- 分子高（inconsistent）时，为了让loss小，优化器会推高分母（uncertainty）
- 这就是"高uncertainty压低loss contribution"的机制

### 5.4 Logarithmic prior (公式7)

$$\mathbf{E}_{\text{prior}}(\mathbf{u}') = \sum_i \log(\mathbf{u}_i' + 1.0)$$

**防止trivial solution** $\mathbf{u}' \to +\infty$：
- 如果没有这个prior，优化器会推所有uncertainty到无穷大，使similarity loss→0
- $\log(\mathbf{u}_i' + 1.0)$是concave function，增长慢于线性，惩罚大uncertainty
- 加1.0 bias防止log(0)和负值
- Ablation（Table 10）：w/o prior term → 5.18 cm vs Full 2.30 cm，差距巨大

### 5.5 Total uncertainty cost (公式8)

$$\mathbf{E}_{\text{uncer}}(\mathbf{u}') = \mathbf{E}_{\text{sim}}(\mathbf{u}') + \gamma_{\text{prior}} \mathbf{E}_{\text{prior}}(\mathbf{u}')$$

$\gamma_{\text{prior}}$是平衡系数。

### 5.6 Affine mapping regularization（重要细节）

直接per-pixel优化uncertainty会导致spatial inconsistency和noise overfitting。Solution：

$$\mathbf{u} = \text{Softplus}(\boldsymbol{\theta} \cdot \mathbf{F})$$

- $\boldsymbol{\theta}$: learnable affine parameters
- $\mathbf{F}$: DINOv2 features
- Softplus: $\text{Softplus}(x) = \log(1 + \exp(x))$，保证uncertainty > 0

**关键区别**：与WildGS-SLAM/UP-SLAM的MLP decoder不同，这里用**local affine mapping**，作用是sliding window内的spatial regularization。Affine是线性变换，参数量小，避免overfitting，且保证local smoothness。

### 5.7 Gradient descent with weight decay (公式9)

不用Newton方法（避免Hessian inversion），用gradient descent：

$$\mathbf{g}_t = \sum_{i=0}^N \frac{\partial \mathbf{E}_{\text{uncer}}}{\partial \mathbf{u}_i'} \cdot \frac{\partial \mathbf{u}_i'}{\partial \boldsymbol{\theta}_{t-1}}$$

$$= \sum_{i=0}^N \frac{\partial \mathbf{E}_{\text{uncer}}}{\partial \mathbf{u}_i'} \cdot \frac{1}{1 + \exp(-\boldsymbol{\theta}_{t-1} \cdot \mathbf{F}_i)} \cdot \mathbf{F}_i$$

$$\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \lambda \cdot \mathbf{g}_t - \eta \cdot \boldsymbol{\theta}_{t-1}$$

变量含义：
- $\mathbf{g}_t$: gradient w.r.t. affine parameters $\boldsymbol{\theta}$
- $\frac{1}{1 + \exp(-\boldsymbol{\theta} \cdot \mathbf{F})}$: Softplus的导数（即sigmoid）
- $\lambda$: learning rate
- $\eta \cdot \boldsymbol{\theta}_{t-1}$: **weight decay**，防止$\boldsymbol{\theta}$发散，stabilize optimization

Ablation验证：
- w/o affine mapping: 2.47 cm（vs Full 2.30）
- w/o weight decay: 2.34 cm

### 5.8 Jacobian推导（补充材料公式11-13）

完整的Jacobian：
$$\frac{\partial \mathbf{e}_{ij}}{\partial \mathbf{u}_i'} = -\frac{1 - \cos(\mathbf{F}_i, \mathbf{F}_{ij})}{(\mathbf{u}_i')^2 \cdot \mathbf{u}_{ij}'} = -\frac{\mathbf{e}_{ij}}{\mathbf{u}_i'}$$

$$\frac{\partial \mathbf{e}_{ij}}{\partial \mathbf{u}_j'} = -\frac{\mathbf{e}_{ij}}{\mathbf{u}_j'} \cdot \boldsymbol{\alpha}_{ij}$$

其中 $\boldsymbol{\alpha}_{ij} \in \mathcal{R}^{(\frac{H}{8} \times \frac{W}{8}) \times (\frac{H}{8} \times \frac{W}{8})}$ 是bilinear interpolation weight matrix，每行只有4个非零元素（2D bilinear的4个邻居）。

## 六、SLAM System设计

### 6.1 Initialization

DROID-SLAM用12个keyframes初始化，disparity初始化为1（很粗糙）。DROID-W的改进：

**Metric depth regularization**（公式10）：
$$\mathbf{E}^+(\mathbf{G}', \mathbf{d}') = \sum_{(i,j) \in \mathcal{E}} \left\| \mathbf{p}_{ij}^* - \mathbf{p}_{ij} \right\|_{\Sigma_{ij}^{\text{uncer}}}^2 + \gamma_d \sum_i \left\| \mathbf{d}_i' - \mathbf{D}_i \right\|^2$$

- $\mathbf{D}_i$: Metric3D v2 [17]预测的metric depth（下采样到H/8 × W/8）
- $\gamma_d$: depth regularization权重
- 作用：高动态场景下，rigid-motion residual不可靠时，metric depth提供anchor防止tracking发散

Ablation: w/o monocular depth → 3.30 cm (vs Full 2.30 cm)，提升明显。

### 6.2 Incremental tracking

新keyframe加入后，sliding window内执行local BA + depth regularization。同时优化poses, disparities, uncertainties。

### 6.3 Global BA

Frontend tracking结束后，对所有keyframes执行global BA refine poses和disparities。**关键设计：freeze dynamic-uncertainty parameters**。

原因：affine transformation是sliding window内的local regularizer，global scale下意义不明，且会引入instability。

### 6.4 非keyframe pose recovery

DROID-SLAM只优化keyframes，non-keyframe poses通过SE(3)插值 + pose graph update恢复。

### 6.5 Trajectory alignment

用Sim(3) Umeyama alignment [50]与ground truth对齐，处理scale ambiguity。

## 七、实验数据深度分析

### 7.1 Bonn RGB-D Dynamic (Table 1)

| Method | Balloon | Balloon2 | Crowd | Crowd2 | Person | Person2 | Moving | Moving2 | Avg |
|--------|---------|----------|-------|--------|--------|---------|--------|---------|-----|
| DROID-SLAM | 7.5 | 4.1 | 5.2 | 6.5 | 4.3 | 5.4 | 2.3 | 4.0 | 4.91 |
| WildGS-SLAM | 2.8 | 2.4 | 1.6 | 2.2 | 3.9 | 3.1 | 1.7 | 2.5 | 2.52 |
| DynaMoN (MS&SS) | 2.8 | 2.7 | 3.5 | 2.8 | 14.8 | 2.2 | 1.3 | 2.7 | 4.10 |
| **DROID-W** | **2.6** | **2.5** | **1.3** | **1.8** | **3.3** | **2.9** | **1.6** | **2.3** | **2.30** |

观察：
- 相比DROID-SLAM提升52% (4.91→2.30)
- 相比WildGS-SLAM提升9% (2.52→2.30)
- 相比DynaMoN提升44% (4.10→2.30)
- 在Person序列上，DynaMoN特别差（14.8 cm），DROID-W显著好（3.3 cm），说明不依赖motion segmentation的优势

### 7.2 TUM RGB-D (Table 2)

DROID-W在低动态序列(f3/sr, f3/shs)上与DROID-SLAM持平，说明uncertainty机制在静态场景下不损害性能。

WildGS-SLAM在低动态序列上反而比DROID-SLAM差（f3/sr: 2.4 vs 2.2, f3/shs: 2.0 vs 1.4），印证paper的论述：mapping质量差时uncertainty estimation失效。

### 7.3 DyCheck (Table 3)

DROID-W平均0.034，相比DROID-SLAM 0.044提升23%。在haru序列上稍差（0.093 vs 0.005），因为moving dog占据大部分view，static features不足以支撑tracking。

### 7.4 DROID-W Dataset (Table 4)

| Method | DT1 | DT2 | DT3 | DT4 | DT5 | DT6 | DT7 | Avg |
|--------|-----|-----|-----|-----|-----|-----|-----|-----|
| TTT3R | 4.64 | 11.25 | 4.30 | 7.35 | 11.28 | 5.09 | 7.26 | 7.309 |
| Splat-SLAM | 0.10 | 6.44 | 0.89 | 0.66 | 0.91 | 2.11 | 0.07 | 1.597 |
| DROID-SLAM | 0.26 | 7.84 | 1.05 | 0.33 | 0.64 | 0.06 | 0.05 | 1.460 |
| WildGS-SLAM | 0.10 | 0.95 | 0.43 | 0.36 | 0.87 | 1.22 | 0.53 | 0.637 |
| **DROID-W** | 0.15 | **0.25** | **0.15** | **0.32** | **0.24** | 0.43 | **0.07** | **0.230** |

DT2是所有方法的难点（Splat-SLAM 6.44, DROID-SLAM 7.84），DROID-W取得0.25的突破。

### 7.5 Runtime (Table 5)

| Method | Dynamic Bonn | TUM | DyCheck |
|--------|--------------|-----|---------|
| DROID-SLAM | 19.89 FPS | 26.97 | 17.50 |
| WildGS-SLAM | 0.22 FPS | 0.32 | 0.18 |
| **DROID-W** | **10.57 FPS** | **14.92** | **11.06** |

**40× speedup over WildGS-SLAM**，同时保持real-time。

## 八、DROID-W Dataset详情

### 8.1 硬件配置

- Livox Mid-360 LiDAR + RGB camera rigid mounting
- RGB: 1200×1600 resolution, 20 FPS
- RTK ground truth: 10 Hz
- IMU synchronized

### 8.2 Ground truth策略

Downtown 1-2: 无RTK，使用FAST-LIVO2 [65] trajectories作为pseudo ground truth
- FAST-LIVO2在DT3-7上ATE仅0.071m（Table 8），证明可作为reliable GT
- DROID-W在DT3-7上0.242m，与FAST-LIVO2的0.071m差距合理（FAST-LIVO2用了LiDAR+IMU+RGB）

### 8.3 序列统计 (Table 7)

| Sequence | Frames | Trajectory Length [m] |
|----------|--------|----------------------|
| Downtown 1 | 1427 | 90.83 |
| Downtown 2 | 2200 | 122.25 |
| Downtown 3 | 1438 | 62.33 |
| Downtown 4 | 1794 | 85.19 |
| Downtown 5 | 2157 | 129.93 |
| Downtown 6 | 1900 | 104.99 |
| Downtown 7 | 1900 | 109.35 |

平均轨迹长度约100m，远超indoor benchmarks的几米，是真正的大scale outdoor evaluation。

### 8.4 YouTube视频 (Table 9)

9个视频，时长8秒到30分钟不等。超过5分钟的视频被分割成5分钟segments（GPU内存限制）。Camera intrinsics用MonST3R [62]从20帧估计。

## 九、关键Ablation Study分析

### 9.1 主ablation (Table 6)

| Config | ATE [cm] | 增量贡献 |
|--------|----------|---------|
| w/o Uncertainty-aware BA | 5.13 | +2.83 |
| w/o monocular depth | 3.30 | +1.00 |
| w/o uncertainty decouple | 2.57 | +0.27 |
| w/o affine mapping | 2.47 | +0.17 |
| w/o weight decay | 2.34 | +0.04 |
| Full | 2.30 | - |

关键insight：
1. **UBA是最核心组件**，去掉后误差增加123%
2. **Monocular depth是第二大贡献**，去掉了增加43%
3. **Bidirectional decoupling**贡献明显（11%）
4. **Affine mapping**和**weight decay**贡献较小但consistent

### 9.2 Prior term ablation (Table 10)

| Config | ATE [cm] |
|--------|----------|
| w/o prior term | 5.18 |
| Full | 2.30 |

**Prior term至关重要**，去掉后所有uncertainty趋于无穷大，等同于w/o UBA配置。

## 十、Qualitative Results分析

### 10.1 Uncertainty estimation (Figure 3, 5)

DROID-W的uncertainty maps特点：
- **Spatially coherent**: 边缘锐利，区域一致
- **Semantically consistent**: 正确识别dynamic objects
- **Robust to view-dependent effects**: 处理reflections (Taylor 22, Tokyo Walking 2)、shadows
- **Sensitive to small dynamic objects**: 即使小物体也能准确标注
- **Robust to motion blur**: Tomyum 1/2序列验证

WildGS-SLAM失败模式：mapping质量差→Gaussian reconstruction error→uncertainty MLP学到错误信号→不稳定预测

MonST3R失败模式：binary mask预测依赖pretrained model generalizability，对challenging scenes检测不全

### 10.2 静态→动态过渡处理 (Figure 6)

Tomyum 1的door opening场景：
- Door静止时（keyframe 280之前）：低uncertainty，feature用于tracking
- Door开始移动：uncertainty立即升高
- **Multi-view信息利用**：frame graph连接多帧，door在280/281帧uncertainty升高（即使本帧door还没动，但相邻帧动了）

这是multi-view consistency-based方法的优势：能从graph topology中aggregate dynamics信息。

### 10.3 3D Reconstruction (Figure 4, 8)

DROID-SLAM失败模式：
- Scale drift (St. Moritz 1)
- Erroneous geometry (St. Moritz 3)
- Noisy distractors (Tokyo Walking 1 & 2)
- Taylor 22: 单走廊重建为两个separate结构（pose tracking错误导致）

DROID-W重建质量：
- Geometrically accurate
- Temporally consistent
- Static regions保留细节（如Tokyo Walking 1的white lane marking清晰可见）

## 十一、Limitations & Future Work

### 11.1 论文承认的局限

**Initialization阶段问题**：uncertainty optimization基于frame-to-frame alignment，初始化时pose估计还不准，导致uncertainty estimation不准。可能的解决：incorporate reconstruction priors。

### 11.2 推测的其他局限

1. **DINOv2 feature计算开销**：每帧需要forward pass through DINOv2，对边缘部署不友好
2. **Affine mapping的locality**：只在sliding window内regularize，可能无法捕捉long-range dynamics
3. **Monocular depth scale**: Metric3D的metric depth在scale上可能与实际有偏差，依赖$\gamma_d$调参
4. **Reflection handling**：虽然实验显示对reflection robust，但extreme view-dependent effects可能仍出问题
5. **Single GPU memory**：>5分钟视频需要分段，可能丢失long-range consistency

## 十二、技术贡献总结与个人评价

### 12.1 核心贡献

1. **Decoupling uncertainty from mapping**：最关键的insight，让uncertainty estimation不依赖Gaussian/NeRF mapping质量
2. **Bidirectional uncertainty decoupling**：$\mathbf{u}_i' \cdot \mathbf{u}_{ij}'$的设计巧妙解耦两帧dynamics
3. **Affine mapping + Softplus**：轻量级regularizer，避免MLP的overfitting
4. **Logarithmic prior**：简洁有效的trivial solution避免机制
5. **DROID-W dataset**：填补outdoor dynamic SLAM benchmark空白

### 12.2 与同类工作对比

| 特性 | WildGS-SLAM | UP-SLAM | DROID-W |
|------|-------------|---------|---------|
| Uncertainty source | Rendered image loss | Rendered + feature sim | Multi-view feature sim only |
| Mapping dependency | Strong (Gaussian) | Strong (Gaussian) | Weak (point cloud) |
| Decoder | MLP | MLP | Affine + Softplus |
| Optimization | Joint with mapping | Joint with mapping | Interleaved with BA |
| Real-time | 0.22 FPS | ~1 FPS | 10+ FPS |
| Outdoor capability | Poor | Poor | Strong |

### 12.3 Intuition building

这篇paper的核心insight可以用一句话概括：**Dynamic objects破坏multi-view consistency，而DINOv2 features能semantic感知这种inconsistency，且不依赖几何重建质量**。

这个insight的深层意义在于：传统SLAM的"几何先验"和现代deep features的"语义先验"应当complement，而非sequential依赖。WildGS-SLAM是"先建好地图再估uncertainty"，DROID-W是"用语义一致性直接估uncertainty，再去stabilize几何"。

从optimization角度，DROID-W选择了**alternating minimization**而非joint optimization，这在non-convex problems中往往更稳定，且每个subproblem可以选用最适合的solver（Gauss-Newton for BA, Gradient Descent for uncertainty）。

### 12.4 参考链接

- 论文代码: https://github.com/MoyangLi00/DROID-W.git
- DROID-SLAM原始论文: https://proceedings.neurips.cc/paper/2021/hash/e8796234b34f1c3f2c5f3c5c5c5c5c5c-Abstract.html
- DINOv2: https://arxiv.org/abs/2304.07193
- Metric3D v2: https://arxiv.org/abs/2404.15506
- FiT3D: https://arxiv.org/abs/2407.20229
- WildGS-SLAM: https://arxiv.org/abs/2411.16593
- MonST3R: https://arxiv.org/abs/2410.03825
- FAST-LIVO2: https://github.com/hku-mars/FAST-LIVO2
- Bonn RGB-D Dynamic: https://www.ipb.uni-bonn.de/data/dynamic-rgbd-benchmark/
- TUM RGB-D: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/
- DyCheck: https://dynamic-video-check.github.io/

### 12.5 个人思考延伸

这篇paper让我想到几个更深层的问题：

1. **Feature consistency vs geometric consistency**: 这两者本质上是multi-view correspondences的不同modalities。DROID-W证明了在dynamic scenes下feature consistency更robust。能否设计一个unified framework自动权衡两种consistency？

2. **Uncertainty的时序建模**: 当前affine mapping是spatial regularizer，但uncertainty在时间维度也应当smooth。能否引入temporal consistency loss？比如uncertainty的变化率penalty。

3. **Graph topology的影响**: Frame graph的edge selection直接影响multi-view consistency measurement。如何设计edge selection策略以最大化uncertainty estimation的informative-ness？

4. **Foundation model的role**: DINOv2作为semantic feature extractor，但更近期的foundation models（如DINOv3, SAM2, VGGT）能否进一步提升？特别是VGGT [52]本身就有geometry-aware能力。

5. **Active uncertainty**: 能否根据当前uncertainty map主动决定下一帧keyframe selection或view direction？这引向active SLAM领域。

6. **Generative priors**: Metric3D作为depth prior，能否用video diffusion models作为temporal dynamics prior来约束uncertainty？这是generative AI与SLAM的交叉点。

7. **4D reconstruction**: 当前输出static point cloud + dynamic point cloud分离，能否直接输出4D scene representation（如4D Gaussians）？这需要将uncertainty作为4D建模的signal而非mask。

这些方向都值得探索，DROID-W的decoupling思想为后续工作提供了清晰的baseline和思想基础。
