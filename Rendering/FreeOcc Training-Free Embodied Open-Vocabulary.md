---
source_pdf: FreeOcc Training-Free Embodied Open-Vocabulary.pdf
paper_sha256: 796bf356a75fb1bb7e9123191cd30a0e8e359d65778be1aa74f18aebc0ac5815
processed_at: '2026-08-04T10:31:24-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 FreeOcc

## 一句话说清楚

**以前要让机器人理解周围 3D 空间的形状和语义，必须先花大价钱标注几十万个 voxel，然后 train 一个 model。FreeOcc 说：不用 train，拿现成的 SLAM + 3D Gaussian + VLM 拼起来就行，效果还更好。**

---

## 为什么这事难

想象一个机器人在房间里走。它每秒看到 30 张 RGB/D 图片，要回答两个问题：

1. **前面 1 米处有没有东西挡着？**（geometry，用来避障）
2. **那个东西是椅子还是桌子？**（semantic，用来执行指令）

传统做法是把这个空间切成小方块（voxel，每边 8cm），每个方块标 "occupied/free" + 类别。问题在于：

- 标注一个房间几万个 voxel，人工 + 半自动 pipeline 成本极高
- ScanNet 标了 1513 个房间已经是极限了
- Train 出来的 model 只认得 ScanNet 里的 11 种东西，换到 Replica 数据集上直接哑火（Table III: mIoU 从 42.53 掉到 0.00）

**核心矛盾**：你 train 得越狠，overfit 越严重；你 train 得越轻，performance 不够。

---

## FreeOcc 的思路：不 train，直接拼

把任务拆成四步，每步都用现成工具：

### Step 1: 我在哪、周围有什么点

用 DROID-SLAM 吃进 RGB 图片流，吐出：
- 每一帧的 camera pose（机器人在哪、朝哪看）
- 一堆稀疏 3D 点（哪些地方有东西）

这就是经典 SLAM 干的事，几十年技术积累，很稳。

**直觉**：SLAM 给你一个「骨架」，告诉你空间的大致结构。骨架越准，后面铺的肉越准。

### Step 2: 把骨架填成肉

SLAM 给的点太稀疏，不能直接用。要变成 dense 的 3D Gaussian map。

这里有个**坑**：传统 3DGS-SLAM（如 SplaTAM、MonoGS）优化 Gaussian 的时候只看「render 出来的图对不对」，不管 3D 位置对不对。结果就是同一个像素可以由「近处小 Gaussian」或「远处大 Gaussian」解释，loss 一样。这就叫 geometric ambiguity。

**FreeOcc 的 fix 很简单粗暴**：

1. **Gaussian 的中心位置死死钉在 SLAM 给的点上**，优化过程中不许动
2. **初始化时把 Gaussian 拉成沿光线的长条**（深度不确定，横向确定）

效果立竿见影：
- IoU 从 22.34（DROID-Splat）涨到 39.34（FreeOcc mono），**涨了 76%**
- FPS 从 10.7 涨到 25.3，因为参数空间小了，优化快了

**直觉**：SLAM 已经告诉你点在哪了，你为什么还要让 Gaussian 乱跑？钉死就完事了。这个 insight 看起来 trivial，但之前所有 3DGS-SLAM 都没这么做，因为大家都在追求 novel view synthesis 的画质，没人在乎 voxel-level 的几何精度。

### Step 3: 给每个 Gaussian 贴标签

用 VLM（默认是一个基于 DINOv2+CLIP 的 training-free segmentation model）对每张图提取 per-pixel 的 language embedding。

然后把这个 2D embedding 通过 depth back-project 到 3D，attach 到对应的 Gaussian 上。

现在每个 Gaussian 长这样：
- 位置 $\mu$（SLAM 给的）
- 形状 $\mathbf{s}, \mathbf{r}$（优化出来的）
- 颜色 $\mathbf{c}$（rendering 用的）
- 不透明度 $o$
- **Language feature $\mathbf{f}$**（VLM 给的）

**直觉**：这一步相当于把 2D 的语义信息「粘」到 3D 结构上。粘合剂是 SLAM 的 depth。

### Step 4: 把 Gaussian 转成 voxel grid

这一步要把连续的 Gaussian 转成离散的 voxel。方法是对每个 voxel 中心 $\mathbf{x}$：

1. 找到附近的 Gaussian
2. 用 Gaussian 概率密度算每个 Gaussian 对这个 voxel 的「占用程度」$\alpha_k(\mathbf{x})$
3. 用概率独立性公式合并：$\alpha(\mathbf{x}) = 1 - \prod(1 - \alpha_k)$（至少一个 Gaussian 占用的概率）
4. 语义同理：用 posterior responsibility 做加权平均，得到 voxel 的 language feature
5. 查询时：把 text prompt encode 成 embedding，和 voxel feature 算 cosine similarity

**直觉**：想象每个 Gaussian 是一团雾，越靠近中心越浓。Voxel 要判断自己被雾覆盖多少，以及是哪种雾（语义）。多个雾团叠加用概率公式合并。

---

## 实验结果说了什么

### EmbodiedOcc-ScanNet（训练集上）

| 方法 | 需要 train 吗 | 需要 pose GT 吗 | IoU | mIoU |
|------|-------------|---------------|------|------|
| EmbodiedOcc++ (supervised) | 是，需要 voxel annotation | 是 | 52.2 | 43.60 |
| GaussTR (self-sup) | 是 | 是 | 15.63 | 4.95 |
| **FreeOcc (mono)** | **不用** | **不用** | 31.29 | 13.86 |

FreeOcc 不 train 不标 pose，还能碾压 self-supervised 方法 2 倍以上。和 supervised 有差距，但差距主要来自 pose 精度（Table VII: 给 GT pose 能涨 10 个点）。

### ReplicaOcc（zero-shot，没见过的场景）

| 方法 | IoU | mIoU |
|------|------|------|
| EmbodiedOcc (supervised) | 22.91 | **0.00** |
| GaussTR (self-sup) | 15.01 | 0.01 |
| **FreeOcc (mono)** | 46.81 | 16.93 |
| **FreeOcc (rgbd)** | 55.65 | 20.90 |

**这是最震撼的结果**。所有 learning-based 方法在新场景上 mIoU 归零，而 FreeOcc 依然坚挺。

为什么？因为 FreeOcc 没学到任何 dataset-specific 的东西。SLAM 是通用的几何工具，VLM 是通用的语义工具，换个场景照常工作。

**直觉**：Learning-based 方法像背答案的学生，换题型就懵；FreeOcc 像会做题的学生，虽然每道题做得慢一点，但什么题型都能做。

---

## 几个有意思的细节

### 1. Fig.1 里的 labeling 问题

EmbodiedOcc-ScanNet 的 GT label 把 window 标成 wall，把 chair 标成 floor。FreeOcc 用 VLM 反而能正确识别 window 和 chair，但在 benchmark 上被「惩罚」了——因为和 GT 不一致。

这说明 **fixed-label benchmark 评估 open-vocabulary 方法有根本缺陷**。GT 不一定对，尤其当 GT 是 coarse taxonomy 的时候。

### 2. Real-world 部署的 voxel 分辨率问题

桌上的红黄纸杯用 0.08m voxel 根本分不开，要降到 0.005m 才行。但 0.005m 对整个房间来说 memory 爆炸。

**暗示**：未来需要 adaptive resolution——物体附近用细 voxel，空旷区域用粗 voxel。这是个很有前途的方向。

### 3. GAGU 的加速效应

通常加约束会变慢，但 GAGU 反而让 FPS 从 10.7 涨到 25.3。原因是 Gaussian 参数空间维度大幅缩小（$\mu$ 不优化了），优化收敛更快。

**直觉**：你把搜索空间从「所有参数」缩到「只剩 scale 和 rotation」，当然快了。

### 4. VLM 可以随便换

Table VI 显示换 SEEM 或 DINOv2 做 VLM，IoU 几乎不变（31.29 vs 31.18），但 mIoU 大跌（13.86 → 8.35）。

说明 geometry 和 semantics 完全 decoupled。geometry 由 SLAM + GAGU 决定，semantics 由 VLM 决定。换个更好的 VLM，语义就更好，几何不受影响。

---

## 对整个领域的 implication

### Foundation model 时代的 system design 哲学

FreeOcc 体现了一种思路：**与其 train 一个 end-to-end model，不如 design 一个 pipeline 把多个 foundation model 串起来**。

好处：
- 每个 component 独立进步，SLAM 更好 → occupancy 更准，VLM 更好 → 语义更丰富
- Zero-shot generalization 天然具备
- 可解释、可 debug
- 部署灵活，换个 sensor 不用 retrain

坏处：
- Pipeline latency 比 end-to-end 高
- Integration complexity
- 每个 component 的 error 会 propagate

### 对 embodied AI 的意义

机器人需要 lifelong 在真实世界工作，不可能预先标注所有场景。FreeOcc 这种 training-free + open-vocabulary 的范式更接近真实部署需求。

想象一下：机器人装个 RealSense，开机就能建图，你说「把桌上那个红杯子拿给我」，它就能在 occupancy map 里 query "red cup" 定位到物体。不需要预先训练，不需要标注，不需要 closed-set 类别。

---

## 一句话总结

**FreeOcc 把 occupancy prediction 从「学习问题」重新定义成「拼装问题」——用 SLAM 拼几何，用 VLM 拼语义，用 Gaussian 当粘合剂。不 train 不标，效果还不错，generalization 秒杀所有 learning-based 方法。核心技术创新是 GAGU（把 Gaussian 中心钉死在 SLAM 点上），简单粗暴但有效。**

Project page: https://the-masses.github.io/freeocc-web/

---

# FreeOcc 技术深度解析

Andrej, 这篇 paper 是 occupancy prediction 领域一个相当有意思的范式转换。我从几个角度来 build intuition。

## I. 核心问题定位：为什么需要 training-free

现有 occupancy prediction 的三个痛点：

1. **Voxel-level annotation 极度昂贵**：EmbodiedOcc-ScanNet 这类 dataset 需要 dense 3D reconstruction + 半自动 labeling，scale 不起来
2. **Pose dependency**：即便 self-supervised 方法如 GaussianOcc、GaussTR，仍然需要 GT camera poses during both train 和 infer
3. **Generalization 崩溃**：Table III 显示 EmbodiedOcc 在 ReplicaOcc 上 zero-shot 时 mIoU 从 42.53 直接掉到 0.00，这是典型的 dataset overfitting

FreeOcc 的关键 insight 在于：occupancy prediction 本质上是个 geometric reconstruction + semantic grounding 的复合任务，而 SLAM + 3DGS + VLM 这三个 component 已经各自成熟，只需要一个 tight coupling 的 pipeline 把它们串起来，完全 bypass 学习阶段。

参考链接：
- EmbodiedOcc: https://arxiv.org/abs/2502.20237
- GaussianOcc: https://arxiv.org/abs/2406.06985
- GaussTR: https://arxiv.org/abs/2501.10126

---

## II. 四层 Pipeline 架构详解

### Layer 1: SLAM Backbone (DROID-SLAM)

选择 DROID-SLAM 而不是 MASt3R-SLAM 或 VGGT-SLAM 作为 default，关键原因是 DROID-SLAM 的 optical flow network 不依赖 SfM 3D supervision，避免了 MASt3R/VGGT 这类 feed-forward SLAM 在训练阶段引入的 inductive bias。

DROID-SLAM 通过 long temporal window 的 joint optimization，输出 globally consistent poses $\mathcal{T}_{1:T} = \{\mathbf{T}_1, \dots, \mathbf{T}_T\}$ 和 sparse 3D points $\mathcal{P}_{1:T} = \{\mathbf{p}_i \in \mathbb{R}^3\}_{i=1}^{N_T}$。

**关键直觉**：global consistency 是后续所有 layer 的 anchor。如果 SLAM 有 drift，downstream Gaussian map 会累积误差，最终 occupancy 的 IoU 直接崩溃。这也是为什么 Table VII 中 GT Pose vs estimated pose 的 IoU 差距是 34.40 → 45.06（约 31% 提升）。

参考：DROID-SLAM https://arxiv.org/abs/2101.10881

### Layer 2: Geometrically Consistent 3D Gaussian Construction

这是 paper 的技术核心。我先讲清楚 **Geometric Ambiguity Problem**。

#### 2.1 Geometric Ambiguity 的数学根源

Standard 3DGS-SLAM 的 objective：

$$\min_\theta \sum_{t=1}^{T} \left( \|\hat{I}_t - I_t\|_2^2 + \beta \|\hat{D}_t - D_t\|_2^2 \right) \tag{2}$$

变量解释：
- $\theta$: 所有 Gaussian 参数的集合 $\{(\mu_i, \mathbf{s}_i, \mathbf{r}_i, o_i, \mathbf{c}_i, \mathbf{f}_i)\}$
- $\hat{I}_t, \hat{D}_t$: rendered image 和 depth
- $I_t, D_t$: observed image 和 depth
- $\beta$: RGB-depth loss 的平衡权重（hyperparameter）
- $T$: frame 数量

**Linearize** rendering operator $F$ 在 $\theta^\star$ 附近：$F(\theta^\star + \delta\theta) \approx F(\theta^\star) + J\delta\theta$，其中 $J = \frac{\partial F}{\partial \theta}|_{\theta^\star}$ 是 Jacobian。

对于 unobservable 或 weakly observable 方向，存在 $\delta\theta \neq 0$ 使得 $J\delta\theta = 0$。这意味着 **loss landscape 在某些方向上是 flat 的**，optimal solution 不是 isolated point 而是 parameter space 上的一个 manifold。

#### 2.2 单 pixel ray 的 ambiguity 具体化

对像素 $\mathbf{u}$ 的 alpha compositing：

$$\hat{I}(\mathbf{u}) = \sum_k w_k(\theta; \mathbf{u}) \mathbf{c}_k, \quad \hat{D}(\mathbf{u}) = \sum_k w_k(\theta; \mathbf{u}) z_k \tag{3}$$

变量：
- $w_k(\theta; \mathbf{u}) \geq 0$: 第 $k$ 个 Gaussian 沿 ray $\mathbf{u}$ 的 compositing weight（依赖 transmittance 和 opacity）
- $\mathbf{c}_k$: 第 $k$ 个 Gaussian 的 color
- $z_k$: 第 $k$ 个 Gaussian 沿 ray 的 depth

**直觉**：Eq.(3) 只约束了 ray 上的一阶矩（weighted sum）。不同的 $\{(w_k, z_k)\}$ 配置可以产生 **identical** $(\hat{I}(\mathbf{u}), \hat{D}(\mathbf{u}))$。比如把 weight 集中在近处的小 Gaussian 或分散到远处的大 Gaussian，rendered 结果可能相同。

这就是为什么 3DGS-SLAM 的 geometry 经常出现 floaters、depth distortion——loss 没有唯一解。

#### 2.3 Geometrically Anchored Gaussian Updates (GAGU)

解决方案分两步：

**Step 1: Geometry-aware Initialization (G-ini)**

对 frame $t$ 的 pixel $\mathbf{u}$，计算 normalized ray direction $\mathbf{d}_{t,\mathbf{u}} \in \mathbb{R}^3$，构造 local rotation $R_{t,\mathbf{u}}$ 使其 local $+Z$ 轴对齐 $\mathbf{d}_{t,\mathbf{u}}$。初始化 anisotropic scale：

$$\mathbf{s}_{t,\mathbf{u}} = (s_\perp, s_\perp, s_\parallel), \quad s_\parallel = \gamma s_\perp \tag{4}$$

变量：
- $s_\perp$: 垂直于 viewing ray 的 Gaussian extent（两个相同值，表示 cross-section 是圆形）
- $s_\parallel$: 平行于 viewing ray 的 extent
- $\gamma$: user-controlled elongation ratio（$\gamma > 1$ 时 Gaussian 是沿 ray 拉长的椭球）

**直觉**：这个初始化建模了「深度不确定但横向位置确定」的物理先验。沿 ray 方向 uncertainty 大（因为 monocular depth 估计有误差），横向 uncertainty 小（因为 pixel 投影位置精确）。这直接减少了 ambiguity 的维度。

**Step 2: Anchored Optimization**

$$\min_\theta \sum_{t=1}^{T} \left( \|\hat{I}_t - I_t\|_2^2 + \beta \|\hat{D}_t - D_t\|_2^2 \right), \quad \text{s.t. } \mu_t = \mathcal{P}_t \tag{5}$$

**关键改动**：Gaussian centers $\mu$ **固定** 到 SLAM 估计的 3D points $\mathcal{P}_t$，不参与 optimization。这把 SLAM 的 geometric consistency 直接 hard-coded 进 Gaussian map。

#### 2.4 Ablation 数据分析

Table V 的 ablation 非常 informative：

| Setting | IoU | mIoU | FPS |
|---------|-----|------|-----|
| w/o GAGU, G-ini (mono) | 19.88 | 10.53 | 10.7 |
| w/o G-ini (mono) | 31.20 | 12.06 | 26.8 |
| Ours (mono) | 39.05 | 15.40 | 25.3 |

**关键观察**：
1. GAGU 贡献 IoU +11.32，**同时 FPS 从 10.7 提升到 26.8（2.5×）**。这个 speedup 来自 unconstrained Gaussian 参数空间维度大幅缩小，optimization 收敛更快。
2. G-ini 在 GAGU 基础上再贡献 IoU +7.85，但 FPS 几乎不变（26.8 → 25.3）。说明 initialization 主要影响收敛点的 quality，不影响 cost。

#### 2.5 与其他 3DGS-SLAM 对比

Table IV 的对比很 striking：

| Method | Replica IoU | ScanNet-mini IoU | Average |
|--------|-------------|------------------|---------|
| DROID-Splat (mono) | 26.27 | 18.41 | 22.34 |
| Ours (mono) | 46.81 | 31.87 | 39.34 |
| SplaTAM (rgbd) | 31.11 | 17.91 | 24.51 |
| Ours (rgbd) | 55.65 | 34.82 | 45.24 |

FreeOcc (mono) 甚至超过所有 RGB-D 3DGS-SLAM 方法。这验证了 decoupled optimization 是 3DGS-SLAM 的系统性问题，不只是 input modality 的问题。

参考：
- SplaTAM: https://arxiv.org/abs/2403.02751
- MonoGS: https://arxiv.org/abs/2312.06741
- Photo-SLAM: https://arxiv.org/abs/2405.19686

---

### Layer 3: Open-Vocabulary Semantic Association

使用 pretrained OV segmentation model（default 是 [60]，一个 training-free OV segmentation 方法基于 DINOv2 + CLIP）提取 per-pixel language-aligned embedding $\mathbf{z}_t(\mathbf{u}) \in \mathbb{R}^D$。

**Pipeline**：
1. 对 input image $\mathcal{T}_t$ 提取 dense per-pixel embeddings $\mathbf{z}_t(\mathbf{u})$
2. 用 SLAM depth back-project 到 3D
3. 每个 lifted 3D point 关联到最近的 geometrically anchored Gaussian
4. 把 language feature $\mathbf{f}_i$ attach 到 Gaussian $G_i$

最终 Gaussian 参数化为 $G_i = (\mu_i, \mathbf{s}_i, \mathbf{r}_i, o_i, \mathbf{c}_i, \mathbf{f}_i)$，其中 $\mathbf{f}_i$ 是 language-aligned feature。

**Table VI 的 ablation** 显示替换 VLM：
- SEEM: mIoU 13.86 → 8.35（下降 40%）
- DINOv2: mIoU 13.86 → 8.18（下降 41%）

但 IoU 几乎不变（31.29 vs 31.18/31.59）。这说明 **geometry 和 semantics 是 decoupled 的**，VLM 的 quality 只影响 mIoU，geometry 完全由 SLAM + GAGU 决定。

参考：
- Training-free OV segmentation (paper [60]): https://arxiv.org/abs/2411.09219
- CLIP: https://arxiv.org/abs/2103.00020
- SEEM: https://arxiv.org/abs/2304.06718
- DINOv2: https://arxiv.org/abs/2304.07193

---

### Layer 4: Gaussian-to-Occupancy Projection

这是把 continuous Gaussian 表示转成 discrete voxel grid 的关键步骤，基于 GaussianFormer2 的 probabilistic superposition 思路，但扩展到 open-vocabulary。

#### 4.1 几何 occupancy

对 query 3D location $\mathbf{x}$，retrieve neighboring LE-Gaussians $\mathcal{H}(\mathbf{x}) = \{G_k\}_{k=1}^{P(\mathbf{x})}$，其中 $P(\mathbf{x}) = |\mathcal{H}(\mathbf{x})|$。

每个 neighbor 的 covariance：
$$\Sigma_k = R(\mathbf{r}_k) \text{diag}(\mathbf{s}_k^2) R(\mathbf{r}_k)^\top$$

变量：
- $R(\mathbf{r}_k)$: 由 quaternion $\mathbf{r}_k$ 构造的 rotation matrix
- $\mathbf{s}_k^2$: scale vector 的元素平方（$\mathbf{s}_k \in \mathbb{R}^3_+$，所以 $\mathbf{s}_k^2$ 是 element-wise square）
- $\Sigma_k \in \mathbb{R}^{3\times3}$: Gaussian ellipsoid 的 covariance matrix

单个 Gaussian 的 spatial support：
$$\alpha_k(\mathbf{x}) = \exp\left( -\frac{1}{2} (\mathbf{x} - \mu_k)^\top \Sigma_k^{-1} (\mathbf{x} - \mu_k) \right) \tag{6}$$

这是 standard multivariate Gaussian density（无 normalization constant，因为只关心 relative support）。

**Probabilistic exclusion** 组合：
$$\alpha(\mathbf{x}) = 1 - \prod_{G_k \in \mathcal{H}(\mathbf{x})} \big(1 - \alpha_k(\mathbf{x})\big) \tag{7}$$

**直觉**：这是 $P(\text{at least one Gaussian occupies } \mathbf{x})$ 的概率公式，假设各 Gaussian 独立。如果任一 $\alpha_k \to 1$，则 $\alpha(\mathbf{x}) \to 1$；如果所有 $\alpha_k$ 都小，则 $\alpha(\mathbf{x})$ 接近它们的 union。

#### 4.2 语义 propagation

Posterior responsibility under local GMM：
$$p(G_k | \mathbf{x}) = \frac{p(\mathbf{x} | G_k) \pi_k}{\sum_{G_j \in \mathcal{H}(\mathbf{x})} p(\mathbf{x} | G_j) \pi_j} \tag{8}$$

变量：
- $p(\mathbf{x} | G_k) = \mathcal{N}(\mathbf{x}; \mu_k, \Sigma_k)$: Gaussian $G_k$ 在 $\mathbf{x}$ 处的 likelihood
- $\pi_k = o_k$: mixture weight 用 opacity（直觉：更 opaque 的 Gaussian 对 surface 贡献更大）
- $p(G_k | \mathbf{x})$: posterior responsibility，即 $\mathbf{x}$ 处的 occupancy 由 $G_k$ 贡献的比例

Feature propagation via posterior expectation：
$$\mathbf{f}(\mathbf{x}) = \sum_{G_k \in \mathcal{H}(\mathbf{x})} p(G_k | \mathbf{x}) \mathbf{f}_k, \quad \hat{\mathbf{f}}(\mathbf{x}) = \frac{\mathbf{f}(\mathbf{x})}{\|\mathbf{f}(\mathbf{x})\|_2} \tag{9}$$

**关键设计**：normalize $\hat{\mathbf{f}}$ 到 unit sphere，这样后续 cosine similarity 计算更稳定。

#### 4.3 Open-vocabulary query

给定 query category set $\mathcal{C}$，用 text encoder 提取 text embeddings $\{\mathbf{t}_c\}_{c \in \mathcal{C}}$：

$$\hat{\mathbf{t}}_c = \frac{\mathbf{t}_c}{\|\mathbf{t}_c\|_2}, \quad s(\mathbf{x}, c) = \hat{\mathbf{f}}(\mathbf{x})^\top \hat{\mathbf{t}}_c \tag{10}$$

变量：
- $\hat{\mathbf{t}}_c$: normalized text embedding for category $c$
- $s(\mathbf{x}, c) \in [-1, 1]$: cosine similarity between voxel feature 和 text embedding

最终输出：$\alpha(\mathbf{x})$ 作为 occupancy probability，$s(\mathbf{x}, c)$ 作为 open-vocabulary semantic score（仅对 occupied voxels 报告）。

**直觉**：这是一个 soft assignment——每个 voxel 不被 forced 到某个 fixed class，而是 maintain 一个 continuous language embedding，可以 query 任意 text prompt。这是 open-vocabulary 的本质。

参考：GaussianFormer2 https://arxiv.org/abs/2412.10395

---

## III. 实验结果深度分析

### 3.1 EmbodiedOcc-ScanNet (Table II)

| Method | Annotation | IoU | mIoU |
|--------|-----------|-----|------|
| EmbodiedOcc++ (supervised) | Occ+Sem+Pose+D+R | 52.2 | 43.60 |
| RoboOcc (supervised) | 同上 | 53.3 | 44.05 |
| GaussianOcc (self-sup) | Pose+R | 10.17 | 4.34 |
| GaussTR (self-sup) | Pose+R | 15.63 | 4.95 |
| **FreeOcc (mono)** | None | 31.29 | 13.86 |
| **FreeOcc (rgbd)** | None | 34.40 | 15.84 |

**关键观察**：
1. FreeOcc **超过 self-supervised 方法 2× 以上**，且不需要 pose annotation
2. 与 supervised 方法差距约 18-20 IoU。Table VII 分析了差距来源：
   - GT Pose: IoU 34.40 → 45.06（+10.66），pose accuracy 是主要 bottleneck
   - Closed-set semantics: mIoU 15.84 → 20.42（+4.58），benchmark taxonomy mismatch
   - 两者结合: mIoU 27.39，仍低于 supervised 的 43.60，说明 voxel-level supervision 仍有不可替代的作用

**Andrej 的 intuition**：这里有个有意思的现象——supervised 方法把很多 visually distinct 的 object（如 window、chair）coarse classify 成 "wall"、"floor"（见 Fig.1）。这是 dataset annotation 的 coarseness 问题。FreeOcc 的 OV 能力反而能正确识别这些 object，但在 benchmark 上被「惩罚」了。这暴露了 fixed-label benchmark 评估 OV 方法的根本 limitation。

参考：EmbodiedOcc++ https://arxiv.org/abs/2410.02254, RoboOcc https://arxiv.org/abs/2504.14604

### 3.2 ReplicaOcc Zero-shot (Table III)

| Method | IoU | mIoU |
|--------|-----|------|
| EmbodiedOcc (supervised) | 22.91 | **0.00** |
| GaussianOcc (self-sup) | 8.71 | 0.00 |
| GaussTR (self-sup) | 15.01 | 0.01 |
| **FreeOcc (mono)** | 46.81 | 16.93 |
| **FreeOcc (rgbd)** | 55.65 | 20.90 |

**这是 paper 最 striking 的结果**：所有 learning-based 方法的 mIoU 几乎为 0！Supervised EmbodiedOcc 的 IoU 也从 51.52 暴跌到 22.91。

**原因分析**：
1. **Appearance shift**: ScanNet 和 Replica 的 visual characteristics 差异大
2. **Camera/scale shift**: 模型 overfit 到 ScanNet 的 intrinsics 和 metric scale
3. **Label space shift**: ReplicaOcc 有 44 类，EmbodiedOcc-ScanNet 只有 11 类

FreeOcc 完全 immune 到这些 shift，因为：
- SLAM 是 generic geometric estimator，不依赖 dataset-specific priors
- VLM 的 open-vocabulary 能力天然支持任意 label space
- 没有 learned occupancy prior 需要 generalize

**Table VIII 的 OV validation**：top-10 categories mIoU 31.06，top-40 降到 12.01。低频类别（小物体、部分观测）更难，符合直觉。

### 3.3 ReplicaOcc Benchmark 设计

**Dataset construction 三阶段**：

1. **Sparse labeled voxel extraction**: back-project depth pixels（stride $s_{\text{pix}} = 4$ subsample）到 3D，quantize 到 voxel size $v = 0.08\text{m}$，majority voting 决定 semantic label

2. **Regular global voxel grid**: 对齐 sparse voxels 到 axis-aligned grid，dimensions $N_x \times N_y \times N_z$ 由 scene extent 决定。每个 grid cell 取 nearest sparse voxel 的 label（距离 < 1 voxel），否则 empty

3. **Observability mask**: frame stride $s_{\text{frm}} = 2$ 采样，project voxel centers 到 camera view，检查 (a) 在 camera 前方 (b) 在 image boundary 内 (c) 不被 measured depth occluded（tolerance 正比于 $v$）。Union 后 optional 3D binary dilation。

**直觉**：这个 observability mask 是评估 embodied occupancy 的关键——只评估 agent 实际观察过的 voxel，避免「未观察区域」的 false negative 惩罚。

---

## IV. Real-world Deployment (Section VIII)

部署配置：
- **Hardware**: Intel i9-14900KF + RTX 5090
- **Sensor**: Intel RealSense D435i
- **Resolution**: 1920×1080 (RGB + depth synchronized)
- **Depth conversion**: raw sensor units × $10^{-3}$ → metric scale
- **VLM**: Qwen3-VL（生成 per-frame visible object categories）
- **Warm-up**: 短暂 auto-exposure 收敛期

**Temporal aggregation**：跨帧累计 predicted words，形成 scene-level open-vocabulary semantic space。这解决了 single-frame VLM 预测的 temporal inconsistency 问题（Section VI 提到的 limitation）。

**Desktop widget experiment**（Fig. 6）：标准 voxel size 0.08m 无法区分桌上小物体，需要减到 0.005m 才能正确区分红黄纸杯。这暗示 **adaptive dynamic resolution** 是未来 embodied occupancy 的重要方向。

参考：Qwen3-VL https://arxiv.org/abs/2511.21631

---

## V. Limitations 与 Future Directions

1. **SLAM backbone dependency**: long-term drift 仍会影响 consistency。Future work 可以把 occupancy representation 作为 SLAM factor graph 的 optimization objective，形成闭环
2. **VLM temporal inconsistency**: 连续帧的 semantic prediction 不稳定。Confidence-aware filtering 或 temporal consistency constraints 可能解决
3. **Voxel resolution fixed**: 0.08m 对小物体不够，0.005m 对大场景浪费 memory。Adaptive resolution 是 promising direction
4. **Pose accuracy gap**: Table VII 显示 GT pose 能提升 31% IoU，说明 SLAM 仍是 bottleneck

---

## VI. 更广的 context 与联想

### 6.1 与 Neural Radiance Field SLAM 的对比

FreeOcc 选择 3DGS 而不是 NeRF 的原因：
- 3DGS 是 explicit representation，直接 query 任意点无需 network forward
- Gaussian-to-occupancy projection 是 analytic 的，NeRF 需要 density field sampling
- 3DGS 的 editability 更好（attach language feature 到 specific primitive）

但 NeRF-SLAM（如 NICE-SLAM、iMAP）在 continuous geometry 上可能更 smooth。FreeOcc 的 GAGU 某种程度上弥补了 3DGS 的离散性。

参考：NICE-SLAM https://arxiv.org/abs/2112.03089

### 6.2 与 Foundation Model-based SLAM 的关系

Table VI 显示 MASt3R-SLAM 和 VGGT-SLAM 都能提升 FreeOcc 性能：
- MASt3R-SLAM: IoU 31.29 → 33.80, mIoU 13.86 → 15.66
- VGGT-SLAM: IoU 31.29 → 33.09, mIoU 13.86 → 15.90, FPS 25.30 → 45.17

**直觉**：Foundation model 提供的 3D priors 能改善 pose estimation 和 sparse geometry，直接 benefit downstream。这暗示 future FreeOcc 的 performance ceiling 会随 SLAM foundation model 进步而提升。

参考：
- MASt3R-SLAM: https://arxiv.org/abs/2503.21731
- VGGT-SLAM: https://arxiv.org/abs/2412.04663 (VGGT 原文)
- VGGT: https://vggt.github.io/

### 6.3 Embodied AI 的更广 implication

FreeOcc 的 training-free 特性对 embodied AI 有重要意义：
1. **Deployment flexibility**: 任意 robot platform，任意 sensor configuration，无需 retraining
2. **Lifelong learning**: 可以持续更新 map 而不会 catastrophic forgetting（因为没有 learned weights）
3. **Human-robot interaction**: open-vocabulary 支持自然语言指令（"把桌上的红杯子拿给我"）

### 6.4 与 occupancy prediction 理论的 connection

Eq.(7) 的 probabilistic exclusion 公式实际上连接到了 **occupancy grid mapping** 的经典理论（Thrun, Burgard, Fox 的 Probabilistic Robotics）。经典方法用 inverse sensor model，FreeOcc 用 Gaussian superposition——本质都是 Bayesian fusion，只是 sensor model 从 ray-cast 变成了 Gaussian density evaluation。

参考：Probabilistic Robotics http://probabilistic-robotics.org/

### 6.5 3DGS 的 geometric ambiguity 与其他工作的呼应

FreeOcc 指出的 ambiguity problem 在其他 3DGS 工作中也有体现：
- **SAGS** (Structure-Aware 3DGS) 用 structural priors 约束 Gaussian
- **PINGS** 把 Gaussian 和 distance field 结合
- **Feature 3DGS** 在 Gaussian 上 attach feature field

FreeOcc 的独特之处在于 **把 SLAM points 作为 hard constraint**（Eq.5 的 $\mu_t = \mathcal{P}_t$），而不是 soft regularization。这更激进但更有效。

参考：
- SAGS: https://arxiv.org/abs/2407.19040
- PINGS: https://arxiv.org/abs/2412.17693
- Feature 3DGS: https://arxiv.org/abs/2311.17801

---

## VII. 总结性 Intuition

FreeOcc 的核心贡献可以浓缩为一句话：**把 occupancy prediction 从 learned prediction problem 重新定义为 geometric reconstruction + semantic grounding 的 composition problem**。

这个 reframe 的关键 insight 是：
1. Occupancy 的 geometry 部分已经被 SLAM + 3DGS 解决得很好，只需要 fix geometric ambiguity
2. Semantic 部分已经被 VLM 解决得很好，只需要 lift 到 3D
3. 两者 decoupled，可以独立改进（Table VI 验证）

这种 modular design 的好处是 **每个 component 的进步都能直接 benefit 整个 system**——SLAM foundation model 更强，occupancy 就更准；VLM 更强，semantic 就更丰富。这比 end-to-end learned method 的 update cost 低得多。

**Andrej 可能会感兴趣的 meta-point**：这篇 paper 体现了「foundation model 时代」的 system design 哲学——与其 train 一个 end-to-end model，不如 design 一个 pipeline 把多个 foundation model 串起来。这与 Tesla 的 vision pipeline、Waymo 的 modular autonomy 思路一脉相承。Trade-off 是 latency 和 integration complexity，但 benefit 是 interpretability、modularity、和 zero-shot generalization。

Project page: https://the-masses.github.io/freeocc-web/
