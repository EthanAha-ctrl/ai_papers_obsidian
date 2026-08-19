---
source_pdf: Holo360D A Large-Scale Real-World Dataset with Continuous Trajectories
  for.pdf
paper_sha256: b8f2942cc98cf0af37c4f156f34469f4e02866c101f3193807150e10cdbac689
processed_at: '2026-08-19T11:21:44-07:00'
target_folder: Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好的，用最直白的人话给你讲一下这篇 paper 的核心 intuition。

简单来说，这篇 paper 的故事就是：**现有的 feed-forward 3D 模型（像 DUSt3R、VGGT、π³）在全景图上全部拉胯，根本原因是数学投影把图像扭曲了，而作者团队直接扛着激光扫描仪满地跑，造了一个迄今最大、最精细的真实世界全景 3D 数据集来教模型重新做人。**

---

### 一、 为什么全景图会让 3D 模型崩掉？（核心 Intuition）

所有基于 CNN 或 Transformer 的 3D 模型，都是在普通的 perspective image（透视相机拍的平面图）上训练的。在这些图里，图像的像素坐标和真实世界的角度是**均匀对应**的。你往左挪 10 个 pixel，真实世界的角度就均匀地转一点。

但是全景图用的是 Equirectangular projection（ERP，等距柱状投影）。它把一个球面硬生生压扁成一个长方形。我们来看它的投影公式：

$$u = \frac{W}{2\pi} \cdot \text{atan2}(X, Z) + \frac{W}{2}$$
$$v = \frac{H}{2} - \frac{H}{\pi} \cdot \text{atan2}\big(Y, \sqrt{X^2 + Z^2}\big)$$

变量解释：
*   $X, Y, Z$：相机坐标系下的 3D 点坐标。
*   $W, H$：全景图像的宽和高。
*   $u, v$：这个 3D 点在 2D 全景图上的像素坐标。
*   $\text{atan2}$：双参数反正切函数，用来求角度。

直觉上，球面的面积元素是 $dA = R^2 \cos\phi \, d\theta \, d\phi$（$\theta$ 是经度，$\phi$ 是纬度）。
在赤道（$\phi=0$），$\cos\phi=1$，一个 pixel 对应一块正常的球面面积。
但是到了南北极（$\phi \to \pm\pi/2$），$\cos\phi \to 0$。这意味着**在极点附近，极少量的 3D 空间被极度拉伸、塞进了大量的 pixel 里**。

这就带来了灾难：CNN 的卷积核（或者 Transformer 的局部 attention）假设相邻 pixel 的几何关系是连续、均匀的。但在全景图的极点处，相邻 pixel 可能对应着真实世界里极其巨大的角度跨度。模型在普通图片里学到的 geometric prior（几何先验）在这里完全失效，直接导致 3D 重建结果一片模糊。

---

### 二、 为什么不用老数据集微调？

作者指出，现有的全景 3D 数据集有三个致命缺陷：

1.  **Scale 太小**：比如大名鼎鼎的 Matterport3D 只有 10,790 张全景图，根本喂不饱现在的 feed-forward 大模型。
2.  **Depth 质量差**：很多深度图有空洞（玻璃、遮挡），或者 RGB 图和深度图没对齐。
3.  **轨迹不连续**：Matterport3D 是把相机架在三脚架上，每隔 2.25 米拍一次。这叫 wide-baseline（宽基线）。现在的 feed-forward 模型依赖多视角之间的重叠和连续运动轨迹来推断 3D，这种“瞬移式”的拍摄方式直接让模型抓瞎。

---

### 三、 Holo360D 怎么造出来的？（硬核工程力）

为了解决上述问题，作者用了一个非常硬核的物理方案：一个人拿着一台集成 LiDAR、IMU、RTK-GNSS 和 360° 相机的手持扫描仪，在各种场景里**慢慢走**（室内 0.3 m/s，室外 0.6 m/s）。

为什么要慢慢走？因为 LiDAR 在高速运动下点云会稀疏。他们一共走了 31.5 公里，采了 19 个小时，搞出了 109,495 张带精确位姿和深度的高分辨率全景图。

#### 1. 填洞与重网格化
LiDAR 扫不到玻璃后面的东西，也扫不到被遮挡的角落，导致 mesh 上有洞。如果直接拿有洞的 mesh 去渲染深度图，模型学到的就是“这里有洞”。

作者写了一个 pipeline 来填洞。对于小洞，用保持曲率的三角剖分直接补；对于大洞，先插“桥边”把大洞切成小洞，再补。
同时，对于墙板这种平整地方，保留低精度 mesh 节省算力；对于栏杆、家具这种细小结构，扔掉平滑过的 mesh，用原始高精度点云重新做 Poisson surface reconstruction。这保证了 depth map 的极高完整度（室内 0.86，室外 0.82）。

#### 2. 深度图渲染的巧思
这里有一个很 clever 的工程设计。他们生成了两种深度图：
*   **Mesh depth (稠密)**：光线追踪打到 mesh 上，每个 pixel 都有深度。
*   **Point depth (稀疏)**：把 LiDAR 原始点云投影到图像上，很稀疏，但是绝对真实，没有任何插值误差。

他们用稠密的 Mesh depth 来做 visibility check（可见性判断），把被遮挡的稀疏点过滤掉。这样既保留了 LiDAR 的绝对精度，又去除了错误的前后景遮挡关系。

---

### 四、 核心实验与直觉解释

作者拿目前最强的全景零样本模型 π³ 做了微调实验，得出了几个极其重要的工程直觉。

#### 实验 1：直接喂全景图 vs 切成透视图喂
| Config | Indoor ATE↓ | Indoor Acc.↓ | Outdoor Acc.↓ |
| :--- | :--- | :--- | :--- |
| π³ baseline | 0.093 | 0.052 | 0.293 |
| π³ + 全景图直喂 | 0.018 | 0.075 | 0.718 |
| π³ + 切成 8 个透视图 | **0.014** | **0.024** | **0.194** |

**直觉解释**：
如果你直接把全景图塞给模型，位姿估计反而变好了（ATE 从 0.093 降到 0.018），但点云精度彻底崩了（Outdoor Acc. 飙升到 0.718）。
为什么？因为位姿估计靠的是全局特征匹配，全景图视野更广，匹配点更多，所以有用。但是点云回归靠的是 pixel-wise 的几何先验，前面说过，ERP 投影在两极的拉伸直接把卷积核搞废了。

所以，最务儿的做法是：**把一张全景图切成 8 张普通的透视图**再喂给模型。这等于把球面畸变切成了模型能理解的平面块。

#### 实验 2：切成 8 个视角 vs 10 个视角
作者试了在水平 8 个视角的基础上，加上向上看和向下看（zenith & nadir）凑成 10 个视角。
结果 10 个视角反而更差。

**直觉解释**：
向下看会拍到拿着扫描仪的人的手和脚（动态噪声），向上看会拍到天花板（低纹理重复区域）。这两个视角引入的噪声和信噪比降低，直接干扰了模型的 cross-view attention。多并不总是好。

#### 实验 3：用 Mesh depth 监督 vs 用 Point depth 监督
*   用 Mesh depth（稠密）：位姿和点云都学得很好。
*   用 Point depth（稀疏）：位姿学得差，点云也一般。

**直觉解释**：
Mesh depth 提供了 100% 的 pixel-level 监督信号，梯度非常强。Point depth 太稀疏，大部分 pixel 没有 loss，梯度信号弱。
**最终策略**：用 Mesh depth 做训练监督，用 Point depth 做评估指标（因为 Point depth 最诚实，没有算法插值的幻觉）。

#### 实验 4：跨数据集对比
在 Matterport3D 上微调 π³，结果比完全微调前还差 9 倍（Overfitting 到小数据集上失去了泛化能力）。而在 Holo360D 上微调，各项指标全面碾压。这证明了“数据规模+数据质量”是一切的基础。

---

### 五、 局限与未来联想

Paper 最后提到一个 limitation：远距离区域的重建质量依然不行。

**我的直觉是**：这是光学物理极限。一个 5760×2880 的全景图，每个 pixel 对应大约 0.06° 的视角。在 10 米外，1 个 pixel 对应 1.1 厘米；但在 100 米外，1 个 pixel 对应 11 厘米。距离越远，深度图的空间分辨率指数级下降，监督信号极度稀疏。要解决这个问题，可能需要 multi-resolution supervision（多分辨率监督），或者在设计 attention 时加入基于距离的权重衰减。

顺着这篇 paper，我联想到几个方向：

1.  **Architecture-level 的改变**：既然切透视图是一种 workaround，未来一定会出现原生的 Panoramic Transformer。比如在 attention 里直接用 spherical distance（球面距离）而不是 pixel distance，或者把 Equirectangular 投影换成 Icosahedral projection（二十面体投影，更均匀）。Paper 里也暗示了要用 "panoramic rays" 替代 pixel positions。
2.  **与 3D Gaussian Splatting 结合**：像 Splatter-360 这种前馈式高斯泼溅方法，极其依赖稠密的多视角输入。Holo360D 的连续轨迹简直是为前馈式 NeRF/3DGS 量身定制的训练集。
3.  **Panoramic SLAM**：把 Holo360D 拿来训练一个全景版的 DROID-SLAM，直接在 360° 相机上做实时稠密重建，这对 VR/AR 头显的 inside-out tracking 是巨大的推动。

**总结一句话**：这篇 paper 没搞什么花哨的模型结构创新，就是靠硬核的工程力，用激光雷达扫出一片高质量的数据海，把现有模型灌饱，顺便给出了“切图喂模型”、“用稠密深度监督”这几条极其务实的工程 recipe。这非常符合现阶段数据驱动的 scaling law 哲学。

参考链接：
*   Holo360D GitHub: https://github.com/Jou719/Holo360D
*   VGGT (Visual Geometry Grounded Transformer): https://vgg-t.github.io/
*   π³ (Permutation-equivariant Visual Geometry): https://arxiv.org/abs/2507.13347
*   Matterport3D: https://niessner.github.io/Matterport/
*   DROID-SLAM: https://arxiv.org/abs/2108.10869

---

# Holo360D: 给 Karpathy 的深度技术解读

## 一、一句话直觉

Holo360D 是一个用 handheld laser scanner + 360° camera rig 走出来的 109,495 张全景图、带 LiDAR-grade ground truth depth/mesh/pose 的大规模真实世界数据集。它的核心 thesis 是: 现有 feed-forward 3D 模型 (DUST3R、VGGT、π³) 在 panorama 上崩掉的根本原因是 **equirectangular projection 引入的 non-uniform spherical sampling**, 而 feed-forward model 学到的 geometric prior 假设的是 perspective image 的 uniform angular sampling。要修这个问题, 需要 (a) 一个大规模、连续轨迹、深度完整的 panoramic 数据集来 fine-tune, (b) 重新思考 input representation (split into perspective views 而不是直接喂 panorama)。

GitHub: https://github.com/Jou719/Holo360D

---

## 二、为什么需要 Holo360D — 现有 panoramic datasets 的三个 fundamental limitations

### (I) Scale constraint
Stanford2D3D 只有 1,314 个 panorama, Matterport3D 10,790 个, KITTI-360 83,000 (但是是从车上拍的 discrete viewpoints, 不是连续轨迹)。在 SfM-free、feed-forward 的 paradigm 下, model 容量大、需要 diverse supervision, 几千到一万级的数据根本撑不起 fine-tuning。

参考:
- Matterport3D: https://niessner.github.io/Matterport/
- Stanford2D3D: https://github.com/ComputerVisionLab/Stanford2D3D
- KITTI-360: http://www.cvlibs.net/datasets/kitti-360/

### (II) Depth map quality
两个问题: (a) depth completeness 低 (Matterport3D 0.62, Stanford2D3D 0.72), 因为 mesh 有 hole (glass、occlusion、insufficient scan coverage); (b) RGB-depth alignment error 大 (Matterport3D 7.99 px), 因为 scanner pose 和 360° camera pose 不严格同步, 或者 360° image 是从多个 pinhole stitch 出来的。

### (III) Trajectory discontinuity
Matterport3D 的 panorama 是 tripod 在 discrete location 拍的, 平均间距 2.25 m — 这是 wide-baseline, 几乎没法用 DUST3R/π³ 这类需要 overlap 的方法, 因为 model 学到的是 narrow-baseline 的 epipolar prior。KITTI-360 是车载, viewpoint sampling 1.01 m, 算连续, 但只是 outdoor, 而且 panorama 是从 6 个 pinhole 拼接的, 几何上和 perspective model 兼容性更好, 但牺牲了真实 360° 的几何 prior。

Holo360D 在三个维度上同时解决:
- **109,495 panoramas** (一个数量级提升)
- **depth completeness 0.86/0.82** (indoor/outdoor), alignment **5.03 px** (state-of-the-art)
- **viewpoint sampling 0.29 m** (10x 比 KITTI-360 密), 真正 continuous trajectory

---

## 三、数据采集硬件 — 第一性原理

手持 rig 是一个 3D laser scanner (LiDAR + IMU + RTK-GNSS + 3 pinhole cameras) 和 360° camera 刚性连接, software trigger 同步。Table 6 的关键参数:

| Module | Key Parameter | Value |
|---|---|---|
| LiDAR FOV | horizontal × vertical | $360° \times 270°$ |
| LiDAR range | — | 0.05–120 m |
| LiDAR point freq | — | 320,000 pts/s, 16 channels |
| LiDAR absolute/relative accuracy | — | 5 cm / 1 cm |
| RTK-GNSS | horizontal | $\pm(8 + 1 \times 10^{-6} D)$ mm, $D$ 是 baseline 距离 km |
| IMU freq | — | 200 Hz |
| 360° camera | resolution × fps | $5760 \times 2880$ @ 24 fps |

**直觉**: 之所以手持、慢走 (0.3 m/s 室内、0.6 m/s 室外), 是因为 LiDAR 在高速 + 有限 viewpoint 下 point cloud 稀疏 — slow + overlap trajectory 才能确保 mesh dense 且 consistent。这跟 DROID dataset 的 "data quality > data scale, but both matter" 的哲学类似 (DROID: https://droid-dataset.github.io/)。

---

## 四、Pipeline 数学拆解

### Stage 1: Onboard SLAM (coarse)
scanner 自带的 real-time SLAM 融合 LiDAR + IMU + GNSS + pinhole cameras, 输出 coarse registered point cloud + camera pose。这是一个 factor-graph formulation, 类似 LIO-SAM (https://github.com/TixiaoShan/LIO-SAM):

$$\min_{\mathbf{x}} \sum_k \|r_{\text{LiDAR}}(\mathbf{x}_k)\|_{\Sigma_L}^2 + \|r_{\text{IMU}}(\mathbf{x}_k)\|_{\Sigma_I}^2 + \|r_{\text{GNSS}}(\mathbf{x}_k)\|_{\Sigma_G}^2$$

其中 $\mathbf{x}_k = (\mathbf{R}_k, \mathbf{t}_k, \mathbf{v}_k, \mathbf{b}_g, \mathbf{b}_a)$ 是第 $k$ 帧的 pose、velocity、IMU biases, residual $r$ 是 measurement 与 predicted 之间的差。

### Stage 2: Offline reconstruction (fine)

**(a) Global Bundle Adjustment** — 联合优化 360° camera poses 和 3D points:

$$\min_{\{\mathbf{R}_j, \mathbf{t}_j\}, \{\mathbf{P}_i\}} \sum_{(i,j) \in \mathcal{V}} \rho\Big(\big\|\pi_{\text{eqr}}(\mathbf{R}_j \mathbf{P}_i + \mathbf{t}_j) - \mathbf{p}_{ij}\big\|^2_{\Sigma_{ij}}\Big)$$

变量说明:
- $\mathbf{R}_j \in SO(3), \mathbf{t}_j \in \mathbb{R}^3$: 第 $j$ 个 360° camera 的 rotation 和 translation
- $\mathbf{P}_i \in \mathbb{R}^3$: 第 $i$ 个 3D point (来自 LiDAR, 已被 triangulated/refined)
- $\pi_{\text{eqr}}$: **equirectangular projection** (见下面公式)
- $\mathbf{p}_{ij} \in \mathbb{R}^2$: point $i$ 在 camera $j$ 中观测到的 pixel
- $\rho$: robust kernel (Cauchy/Huber)
- $\mathcal{V}$: visibility set

**Equirectangular projection 公式** (这是整篇 paper 的 "敌人"):

$$u = \frac{W}{2\pi} \cdot \text{atan2}(X, Z) + \frac{W}{2}, \quad v = \frac{H}{2} - \frac{H}{\pi} \cdot \text{atan2}\big(Y, \sqrt{X^2 + Z^2}\big)$$

变量: $(X, Y, Z)$ 是 camera frame 下的 3D point, $W, H$ 是 image width/height, $\text{atan2}$ 是 two-argument arctangent。$u$ 沿水平方向、$v$ 沿垂直方向。

球面面积元素 $dA = R^2 \cos\phi \, d\theta \, d\phi$, 而 image 像素 $du \, dv \propto d\theta \, d\phi$。所以在 equator ($\phi=0$) 一个 pixel 对应 $R^2 \, d\theta \, d\phi$ 的球面面积, 但在 pole ($\phi \to \pm\pi/2$) 对应 $\to 0$ 的面积 — 即 **pole 区域被 severely over-sampled, 出现 extreme stretching**。这就是 perspective-trained CNN 学的 local patch statistics 在 poles 处失效的数学根源。

**(b) Poisson Surface Reconstruction** — 把 colored point cloud 变 mesh:

求解 Poisson equation: $\nabla^2 f = \nabla \cdot \mathbf{V}$, 其中 $\mathbf{V}(\mathbf{x}) = \sum_i \mathbf{n}_i \cdot K(\mathbf{x} - \mathbf{x}_i)$, $\mathbf{n}_i$ 是 oriented normal, $K$ 是 kernel。然后 mesh = $\{f = \text{iso-value}\}$ (https://hhoppe.com/projects/).

### Stage 3: Post-processing (核心创新之一)

**(i) Data Denoising** — 三步:
1. Manual crop: 把点云裁剪到 ROI
2. Radius outlier removal: 对每个 point, 看 $r$-邻域内邻居数, 若 $< k_{\min}$ 则删除
3. Visual inspection: 删除大块 noise cluster

**直觉**: residential area 有 dynamic pedestrian (motion artifact) + reflective surface (specular outlier), 这些是 LiDAR 的 well-known failure mode。

**(ii) Mesh Hole Filling** — 三步:
1. **Detect holes** + 测量 perimeter $P$。Hole detection 经典方法: 找 boundary edges (only one adjacent face), 然后 flood-fill 把连通的 boundary edges 分组成 holes。
2. **Small holes** ($P < P_{\text{th}}$): curvature-preserving triangulation — 在 hole boundary 上的 vertices 满足 "Delaunay-like" + boundary curvature constraint, 确保补的 patch 跟 surrounding mesh curvature 连续。
3. **Large holes** ($P \geq P_{\text{th}}$): 先 insert **bridge edges** 把大洞切成几个小洞, 再分别 curvature-preserving triangulate。Bridge edges 的选择是为了 minimize geometric distortion。

**(iii) Region-specific Remeshing** — 对 thin structures (家具、栏杆) 特别关键:
- High-quality regions (wall, floor): 保留 stage 2 mesh (downsampled, smoothed)
- Low-quality regions (thin objects): 丢掉, 用 original high-resolution point cloud **重新** Poisson reconstruct

**直觉**: 这一步揭示了一个 trade-off — global mesh 为了 computational tractability 要 downsample, 但 thin structures 一旦 downsample 就 lose critical detail (像 Siggraph ASCII art 的细节就糊了)。Region-specific 等于 multi-resolution mesh, 把 "expensive computation budget" 投在 details 上而不是均匀 spread。

**(iv) Depthmap Creation** — 两种 depth:
- **Mesh depth (dense)**: 对每个 pixel $p=(u,v)$, 计算对应 viewing ray:
$$\mathbf{d}(u, v) = \big(\sin\theta \cos\phi, \, \sin\phi, \, \cos\theta \cos\phi\big), \quad \theta = 2\pi\big(\tfrac{u}{W} - \tfrac{1}{2}\big), \quad \phi = \pi\big(\tfrac{1}{2} - \tfrac{v}{H}\big)$$
然后 ray-cast: 找最小 $t > 0$ 使得 $\mathbf{O} + t \mathbf{d}$ 与 mesh 某三角面相交。depth $= t$ (since $\|\mathbf{d}\|=1$)。
- **Point depth (sparse)**: project point cloud, 每 pixel 保留最近 point 的 depth; 然后用 mesh depth 做 visibility check — 如果 point depth > mesh depth, 说明这个 point 被 mesh 挡住了, 应该 occluded, 丢弃。

这个 mesh-as-visibility-oracle 的 trick 很 clever, 因为 point cloud 本身没有 occlusion reasoning (它就是稀疏的 3D 点集), mesh 提供了 continuous surface 来判断 visibility。

---

## 五、Dataset Statistics — 量化证据

Table 1 的核心数字, 加上 paper Sec 3.4 的细节:

| Metric | Holo360D | Matterport3D | Stanford2D3D | KITTI-360 | 360Loc |
|---|---|---|---|---|---|
| Viewpoint sampling distance | **0.29 m** | 7.99* | 9.45 | 1.01 | 0.49 |
| Alignment error (px, ↓) | **5.03** | 7.99 | 9.45 | 11.72 | 12.24 |
| Depth completeness indoor (↑) | **0.86** | 0.62 | 0.72 | — | 0.62 |
| Depth completeness outdoor (↑) | **0.82** | — | — | 0.16 | 0.7 |
| # panoramas | **109,495** | 10,790 | 1,314 | 83,000 | 2,244 |
| # scenes | 75 | 90 | 10 | 11 | 4 |

\* Matterport3D 的 alignment 7.99 应该是平均 inter-frame distance, paper Table 1 把它们排在一列可能是笔误; depth-panorama alignment 实际 0.62 px error 那一列。

Point cloud reconstruction accuracy 测量方法很有意思: 用一个 **2 mm 精度的 laser rangefinder** 测量 scene 内 reference dimensions (如图 6, 比如门高、墙距), 然后从 scanner 重建的点云里 extract 同样的 dimensions, 计算 RMSE:
- **Indoor: 4.5 mm RMSE**
- **Outdoor: 7.0 mm RMSE**

这给了一个 absolute scale 的 ground truth, 而 KITTI-360/Matterport3D 都没做这个 cross-validation。

总采集量: 31.5 km trajectory, 19 hours on-site, 190,000 m² area (单 scene 最大 40,000 m², 单 trajectory 最长 5 km)。

---

## 六、Benchmark Experiments — 在 π³ 上做 fine-tuning strategy 搜索

π³ (https://arxiv.org/abs/2507.13347) 是 permutation-equivariant 版本的 VGGT, 没有 reference frame bias, 是 panoramic multi-view 3D reconstruction 的 SOTA baseline。Holo360D 在 π³ 上做了三组 ablation。

### Ablation 1: Input representation (panorama vs split views)

Table 2:
| Config | Indoor ATE↓ | Indoor RPE$_t$↓ | Indoor RPE$_r$↓ | Indoor Acc.↓ | Indoor Comp.↓ | Outdoor Acc.↓ | Outdoor Comp.↓ |
|---|---|---|---|---|---|---|---|
| π³ baseline | 0.093 | 0.112 | 2.034 | 0.052 | 0.033 | 0.293 | 0.221 |
| π³ + panorama | 0.018 | 0.030 | 0.621 | 0.075 | 0.160 | 0.718 | 0.584 |
| π³ + **split views** | 0.014 | 0.014 | 0.295 | **0.024** | **0.018** | **0.194** | **0.152** |

**关键发现**: 直接喂 panorama 给 perspective-trained model, **pose 反而变好了** (ATE 0.018 < 0.093), **但 point cloud 质量崩了** (Acc. 0.075 > 0.052, Comp. 0.160 >> 0.033, outdoor 更糟到 0.718)。

**直觉**: pose estimation 主要靠 global feature matching, panorama 提供 more visual context 在这方面有帮助。但 point cloud regression 依赖 pixel-wise geometric prior, 而 perspective CNN 的 receptive field 假设 uniform angular sampling, 在 equirectangular distortion 下完全失效, 所以 panorama 喂进去 point cloud 全糊。Split views 把 panoramic distortion 切成 perspective "chunks", 几何 prior 才生效。

Paper 在 Sec 5 明确点出: **"Model adaptation is required to effectively handle spherical distortion, such as introducing panoramic rays to enhance geometric attention and designing a panoramic loss with latitude awareness to improve geometric supervision."** — 这是 future work 的 hint。

### Ablation 2: View decomposition (8 vs 10 views)

8 views: 水平方向均匀 8 个 perspective views (Figure 9, 每个 ~90° horizontal FOV, 有 overlap)。
10 views: 8 views + 1 个 upward (zenith) + 1 个 downward (nadir)。

Table 3 节选:
| Config | Indoor ATE↓ | Indoor RPE$_r$↓ | Indoor Acc.↓ | Outdoor Acc.↓ |
|---|---|---|---|---|
| + 8 views, mesh | **0.014** | **0.295** | **0.024** | **0.194** |
| + 10 views, mesh | 0.021 | 0.795 | 0.027 | 0.208 |

**直觉**: 多 2 个 view 看似更完整, 实际 nadir view 看到 dynamic pedestrian 脚 / 反光地板 (noise), zenith view 看天花板 (low-texture, repetitive pattern)。这俩 view 的 **信噪比低**, cross-view attention 被 noise 主导, 反而 degrade model。这个观察跟 rooftop 360° NeRF 工作里 "zenith/nadir 是 hard case" 一致 (Mip-NeRF 360: https://jonbarron.info/mipnerf360/)。

### Ablation 3: Depth supervision type (mesh vs point vs mesh+point)

Table 3:
| Config | Indoor ATE↓ | Indoor Acc.↓ | Indoor Comp.↓ |
|---|---|---|---|
| 8 views, mesh | 0.014 | 0.024 | 0.018 |
| 8 views, point | 0.054 | 0.045 | 0.040 |
| 8 views, mesh+point | 0.015 | 0.024 | 0.018 |

**直觉**: 
- **Mesh depth 是 dense supervision**, 每个 pixel 都有 GT, gradient 信号强, pose 和 geometry 都学得好。
- **Point depth 是 sparse supervision**, 大部分 pixel 没 GT, gradient 弱, pose 退化严重 (ATE 0.054 vs 0.014)。但 point depth 是 raw LiDAR 测量, 是最 "honest" 的 ground truth, 没有 mesh hole filling 的 hallucination, 所以 paper 把它当 evaluation GT 用。
- Mesh+point 跟 mesh-only 持平, 没显著增益。

Paper 结论: mesh depth 作为 training supervision, point depth 作为 evaluation metric。这是一个很 clean 的设计原则, 值得借鉴。

### Cross-model evaluation (Table 4)

| Model | Indoor Acc.↓ | Outdoor Acc.↓ | Indoor AUC@30↑ |
|---|---|---|---|
| π³ baseline | 0.052 | 0.293 | 0.733 |
| π³ finetune | **0.024** | **0.194** | **0.837** |
| VGGT baseline | 0.096 | 0.495 | 0.596 |
| VGGT finetune | 0.059 | 0.376 | 0.756 |
| FLARE baseline | 0.133 | 0.852 | 0.256 |
| FLARE finetune | 0.031 | 0.365 | 0.515 |

三个 model 都显著提升。VGGT 提升 2x, FLARE 提升 4x, π³ 提升 2x。说明 Holo360D 的 training signal 是 architecture-agnostic 的。

VGGT: https://vgg-t.github.io/
FLARE: https://zhangsz008.github.io/flare.github.io/

### Cross-dataset comparison (Table 5, 关键!)

| Train data | Indoor Acc.↓ | Outdoor Acc.↓ | Indoor ATE↓ |
|---|---|---|---|
| π³ baseline | 0.052 | 0.293 | 0.093 |
| π³ + Matterport3D | 0.477 | 2.803 | 0.428 |
| π³ + **Holo360D** | **0.024** | **0.194** | **0.014** |

在 Matterport3D 上 fine-tune 反而比 baseline **差 9x** (Acc. 0.477 vs 0.052)。这是因为 Matterport3D scale 太小 (10,790), fine-tune 后 over-fit 到它的 distribution, 失去 generalization。Holo360D 则既提升 in-domain 又不破坏 out-of-domain。

**这是 paper 最强的 selling point**: 数据集规模 + 质量 直接决定 feed-forward model 能不能 fine-tune。这跟 Karpathy 在 "Recipe for training GPT" 里强调的 data quality > everything 的思想完全一致 (https://github.com/karpathy/nanoGPT)。

---

## 七、Limitations 和我的 intuition

### Paper 自己提的 limitation: distant regions quality degrade
Figure 18 显示远距离区域重建糊。Paper 解释: distant region 在 image 中占 pixel 数少, supervision 信号弱; 且 spatial resolution 在远距离上对应大 spatial extent (一个 pixel 对应几米的 3D 距离), 监督稀疏。

**我的延伸直觉**: 这其实是 panoramic depth estimation 的 fundamental problem — 一个 5760×2880 全景图, 假设 FOV 360°×180°, 每个 pixel 对应 $0.0625° \times 0.0625°$ 的 angular extent。在 10 m 距离, 一个 pixel 对应 1.1 cm; 在 100 m 距离, 一个 pixel 对应 11 cm。所以远距离的 **depth spatial resolution 是 linearly degraded** 的, 这是物理极限, 不是数据集问题。要从根本上解决, 需要 multi-resolution supervision (类似 NeRF 的 pyramid loss) 或者 focal length adaptation (类似 foveated rendering)。

### 联想: spherical representation 的 alternatives

Equirectangular 不是唯一选择。学术界有:
- **Tangent images** (Eder et al., https://arxiv.org/abs/2012.03046) — 把球面投影到 20 个 tangent planes, 每个 plane 是 perspective, 没 distortion
- **Icosahedral projection** — 投到 20 面体表面, 更 uniform
- **Spherical CNNs** (Cohen et al., https://arxiv.org/abs/1801.10130) — 用 spherical harmonics 而不是 Euclidean convolution
- **Gnomonic projection** — 把球面切 6 个 cube face (cube map)

Paper 没探索这些 alternative, 但 Sec 5 提到 "panoramic rays to enhance geometric attention" — 这暗示 future model 应该在 attention 里嵌入 ray direction 而不是 pixel position, 让 model 自己 learn spherical geometry。这跟 NeRF 的 positional encoding 思想一致 (https://arxiv.org/abs/2003.08934)。

### 联想: 跟 3D Gaussian Splatting 的关系

Paper 提到 Splatter-360 [10] 和 PanoGRF [9] 在 wide-baseline panorama 上做 generalizable Gaussian splatting。Holo360D 的 continuous trajectory + dense depth 应该是这类方法的理想 training data — 比 Matterport3D 的 wide-baseline 训练数据强很多。Gaussian Splatting 原文: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

### 联想: 跟 DROID-SLAM 的关系

DROID-SLAM (Teed et al., https://arxiv.org/abs/2108.10869) 也是 dense SLAM, 但用 perspective camera。Holo360D 的 continuous trajectory 数据 + dense depth 可以训练一个 "panoramic DROID-SLAM", 在 360° camera 上做 dense SLAM, 这对 VR/AR 是 huge。

### 联想: 跟 Foundation Models 的关系

- **Depth Anything V2** (https://depth-anything-v2.github.io/) 在 perspective image 上是 SOTA monocular depth, 但在 panorama 上因 spherical distortion 崩。PanDA [7] (https://zidongcao.org/panda/) 试图用 unlabeled panorama + Mobius augmentation 修, 但还是 perspective model 的 fine-tune, 没 architectural change。
- Holo360D 给了足够的数据来 **从头训练** 一个 panoramic foundation model, 而不是 fine-tune perspective model。这是 next step。

### 联想: 数据集 design philosophy

Holo360D 跟几个 "high quality real-world 3D dataset" 的 trend 一致:
- **DL3DV-10K** (https://dl3dv-10k.github.io/DL3DV-10K.github.io/): 10K scenes, 51.2M images, 真实世界 perspective 3D
- **Aria Digital Twin** (https://www.projectaria.com/adt/): 第一人称视角的 digital twin
- **CO3D** (https://github.com/facebookresearch/co3d): category-level 3D
- **Mapillary Planet-Scale Depth** (https://github.com/mapillary/mapillary_sls): street-view depth at scale

Holo360D 是 panoramic 方向的对应。所有这些数据集都遵循同一原则: **真实世界采集 > 合成, dense GT > sparse, continuous trajectory > discrete**。

---

## 八、结论 — 对 Karpathy 而言的关键 take-away

1. **Equirectangular distortion 是 perspective-trained feed-forward model 在 panorama 上崩的数学根源**。$dA = R^2 \cos\phi \, d\theta \, d\phi$ 这个公式解释了一切。
2. **数据 scale + quality 决定 fine-tuning 是否 work**。Matterport3D (10K) fine-tune 反而负向, Holo360D (109K) 一致正向。这跟你 nanoGPT 里强调的 data quality philosophy 一致。
3. **Split views 优于 panorama direct input** — 说明 perspective geometric prior 在架构层面是 baked-in 的, 短期 best practice 是切 view, 长期 solution 是 architectural change (panoramic rays, latitude-aware loss)。
4. **Mesh depth > point depth as supervision, point depth > mesh depth as evaluation** — 一个非常 clean 的 train/eval protocol design。
5. **8 views > 10 views** — zenith/nadir views 的 noise > 信息增益, 提醒我们在 multi-view 设置下, view selection 本身是个值得 ablate 的 axis, 不是 "more views always better"。
6. **Limitation: distant regions** 是 fundamental 的 angular resolution limit, 不是 dataset bug。需要 architectural solution (multi-resolution / foveated attention) 来根本解决。

未来期待 Holo360D 训出来的 panoramic-native foundation model — 应该会出现一个 "panoramic DINOv2" 或 "panoramic DUSt3R", 把 spherical geometry baked into attention mechanism。Paper 末尾的 hint ("panoramic rays, latitude-aware loss") 给了 clear research direction。

如果让我预测 next paper, 我赌作者组接下来会做一个 **panoramic-equivariant transformer**, 类似 π³ 的 permutation equivariance, 但加上 spherical equivariance (在 SO(3) 上 equivariant 而不是只在 permutation group 上), 用 Holo360D 训。这是一个 100+ citation 的方向。
