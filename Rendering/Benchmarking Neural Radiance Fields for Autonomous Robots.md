---
source_pdf: Benchmarking Neural Radiance Fields for Autonomous Robots.pdf
paper_sha256: 63a5dc8508abfd36bc80726e6547356ebca4f8f94ec1175b1c8fc87a9e4ded17
processed_at: '2026-07-18T16:35:16-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Benchmarking Neural Radiance Fields for Autonomous Robots — 深度技术讲解

你好 Andrej！这篇paper是Yuhang Ming等人写的一篇comprehensive survey，系统性地review了Neural Radiance Fields (NeRF)在autonomous robots中的应用。我尝试build your intuition about为什么NeRF对robotics如此有吸引力，以及各个sub-task的技术核心。

---

## 1. Paper的核心Thesis

这篇survey的核心论点：**NeRF作为一种implicit neural representation，能够从sparse, unstructured sensor data中重建high-fidelity 3D scenes，从而显著提升autonomous robots的perception, localization, navigation, 和decision-making能力**。

作者将autonomous robot system分解为六个core modules：
- **Perception**
- **Localization & Navigation**
- **Decision-Making**
- **Execution**
- **Communication**
- **State-monitoring & Fault-handling**

其中前三个modules是NeRF应用最密集的区域。Survey覆盖了六个具体tasks：
1. 3D Reconstruction (§3)
2. Segmentation (§4)
3. Pose Estimation (§5)
4. SLAM (§6)
5. Planning & Navigation (§7)
6. Interaction (§8)

Reference: [arxiv paper](https://arxiv.org/abs/2405.01333) (相近的Wang et al. survey)

---

## 2. NeRF技术Background — 公式与Intuition

### 2.1 Scene Representation

NeRF的核心是用一个MLP $F_\Theta$ 将5D input mapping到4D output：

$$F_\Theta(\mathbf{x}, \mathbf{d}) = (\mathbf{c}, m)$$

变量解释：
- $\mathbf{x} = (x, y, z)^T$：3D spatial position（空间位置坐标）
- $\mathbf{d} = (\theta, \phi)^T$：2D viewing direction（观察方向，用球坐标的theta和phi表示）
- $\mathbf{c} = (r, g, b)^T$：emitted RGB color（该点向观察方向发射的RGB颜色）
- $m$：implicit surface representation，可选自 $\{\sigma, s, o\}$
  - $\sigma$：volume density（体积密度，原始NeRF的选择）
  - $s$：signed distance function value（SDF值，到最近表面的有符号距离）
  - $o$：occupancy（占用概率，0到1之间）
- $\Theta$：MLP的weights

**Intuition building**：为什么这个representation如此powerful？传统的3D representation（point cloud, voxel grid, mesh）都是explicit的，每个element单独存储。NeRF把整个scene "压缩"进一个MLP的weights里。这意味着scene的几何和appearance信息以continuous, differentiable的形式存在，可以任意分辨率query。对robotics来说，这意味着从few noisy images就能得到一个dense, continuous的scene model。

### 2.2 Positional Encoding

直接用$(\mathbf{x}, \mathbf{d})$作为input，MLP会suffer from "spectral bias"——倾向于学习low-frequency functions，导致rendered views over-smoothed。解决方案是positional encoding：

**Sinusoidal encoding** (原始NeRF)：

$$\gamma(\mathbf{p}) = (\sin(2^0 \pi \mathbf{p}), \cos(2^0 \pi \mathbf{p}), \dots, \sin(2^{L-1} \pi \mathbf{p}), \cos(2^{L-1} \pi \mathbf{p}))$$

变量解释：
- $\mathbf{p}$：input coordinate，可选自$\{\mathbf{x}, \mathbf{d}\}$
- $L$：maximum frequency level，控制encoding的"resolution"，原始paper用$L=10$ for position, $L=4$ for direction

**Fourier mapping** (Tancik et al.)：

$$\gamma(\mathbf{p}) = (\sin(2\pi \mathbf{B}\mathbf{p}), \cos(2\pi \mathbf{B}\mathbf{p}))$$

变量解释：
- $\mathbf{B} \in \mathbb{R}^{L \times d}$：random matrix，每个entry从zero-mean Gaussian采样
- $d$：input dimension（position是3，direction是2）

**Intuition**：positional encoding本质上是一个kernel trick——把low-dimensional input映射到high-dimensional space，使得MLP能在high-dimensional space中拟合high-frequency functions。这与Fourier analysis中的频率分解对应。

Reference: [Tancik et al. NeurIPS 2020](https://arxiv.org/abs/2006.10739)

### 2.3 Volume Rendering

给定camera ray $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$，其中$\mathbf{o}$是camera center，$t_n$和$t_f$是near/far bounds。

**Volume density rendering** (原始NeRF)：

$$\hat{C}(\mathbf{r}) = \sum_{i=1}^{N} T_i (1 - \exp(-\sigma_i \delta_i)) \mathbf{c}_i$$

$$T_i = \exp\left(-\sum_{j=1}^{i-1} \sigma_j \delta_j\right)$$

变量解释：
- $N$：沿ray采样的点数
- $t_i$：第$i$个采样点对应的ray参数
- $\delta_i = t_{i+1} - t_i$：相邻采样点之间的距离
- $\sigma_i$：第$i$个点的volume density
- $\mathbf{c}_i$：第$i$个点的color
- $T_i$：transmittance，从near bound到第$i$个点的累积"透明度"
- $1 - \exp(-\sigma_i \delta_i)$：在第$i$个小区间内ray被absorbed的概率（类比Beer-Lambert law）

**SDF-based rendering** (NeuS风格)：

$$\hat{C}(\mathbf{r}) = \frac{1}{\sum_{i=1}^{N} w_i} \sum_{i=1}^{N} w_i \mathbf{c}_i, \quad w_i = \sigma\left(\frac{s_i}{tr}\right) \cdot \sigma\left(-\frac{s_i}{tr}\right)$$

变量解释：
- $s_i$：第$i$个点的SDF值
- $tr$：truncation distance，控制SDF的有效范围
- $\sigma(\cdot)$：sigmoid function
- $w_i$：weight，在surface附近（$s_i \approx 0$）达到最大

**Occupancy-based rendering** (UNISURF风格)：

$$\hat{C}(\mathbf{r}) = \sum_{i=1}^{N} w_i \mathbf{c}_i, \quad w_i = o_i \prod_{j=1}^{i-1}(1 - o_j)$$

变量解释：
- $o_i$：第$i$个点的occupancy probability
- $w_i$：第$i$个点是ray"hit"第一个surface的概率

**Intuition**：三种rendering equation的物理直觉都是"ray marching"——沿着ray采样，计算每个采样点对pixel color的贡献。Volume density用exponential decay modeling occlusion；SDF用surface附近的bell-shaped weight；Occupancy用第一个surface的probabilistic hit。对robotics，SDF通常给出更clean的surface，这对manipulation和grasping很重要。

Reference: [NeuS paper](https://arxiv.org/abs/2106.10689), [UNISURF paper](https://arxiv.org/abs/2104.10078)

---

## 3. 3D Reconstruction — 技术分类与Benchmark

### 3.1 Rigid Reconstruction的三大方向

作者把rigid reconstruction分为：
- **Object-Level**：单个物体，DTU-MVS dataset
- **Larger-Scene**：room-scale或city-scale，Replica, ScanNet
- **Sequential-Input**：从video stream增量重建

#### Object-Level的关键methods：

| Method | 核心创新 | Representation |
|--------|---------|---------------|
| **UNISURF** [25] | 首次将occupancy field与NeRF结合，去除per-pixel mask需求 | Occupancy |
| **NeuS** [6] | SDF + volume rendering的unbiased formulation | SDF |
| **VolSDF** [28] | Volume density作为transformed SDF的function | SDF |
| **Geo-NeuS** [29] | 用point clouds提供explicit SDF supervision | SDF |
| **HF-NeuS** [31] | 分解SDF为base + displacement，coarse-to-fine | SDF |
| **Voxurf** [35] | Hybrid voxel grid + dual decoder，20x加速 | SDF |
| **SparseNeuS** [36] | 从image features学generalizable priors，sparse views | SDF |
| **NeRO** [44] | 两阶段：geometry → BRDF recovery | SDF |

**关键benchmark on DTU-MVS** (Table 1)：

DTU-MVS有15个challenging scans，metric是Chamfer-$L_1$ distance [mm]（越低越好）。从Table 1的关键数字：

- COLMAP baseline: 1.36 mm (avg)
- IDR: 0.90 mm
- NeuS: 0.77 mm
- **Geo-NeuS: 0.51 mm** (利用point cloud prior)
- Discrete-NeuS: 0.73 mm (coordinate quantization trick)
- Voxurf: 0.72 mm (voxel grid acceleration)
- SparseNeuS (3 views): 1.64 mm → finetuned: 1.27 mm

**Intuition**：可以看出几个趋势：
1. Pure NeRF (NeuS, 0.77) 已经显著优于传统MVS (COLMAP, 1.36)
2. 加入geometric priors (Geo-NeuS, 0.51)能进一步提升，因为point cloud提供了surface location的强约束
3. Sparse view是open challenge——3 views时误差翻倍(1.64 vs 0.77)
4. Voxel-based hybrid representations (Voxurf)在保持quality的同时大幅加速

#### Larger-Scene的关键methods：

| Method | 关键技术 |
|--------|---------|
| **NeuralRGB-D** [7] | SDF + depth supervision + pose refinement |
| **Vox-Surf** [48] | Dense voxel grid + small MLP |
| **Instant-NGP** [49] | Multi-resolution hash encoding，大幅加速 |
| **GO-Surf** [50] | 4-level feature grid + SDF regularization |
| **ManhattanSDF** [51] | Manhattan-world assumption for planar regions |
| **NeuRIS** [52] | Adaptive normal supervision with pre-trained normal estimator |
| **MonoSDF** [56] | Omnidata monocular depth/normal prediction |
| **PC-NeRF** [61] | Parent-child NeRF for city-scale |

**Benchmark on Replica & ScanNet** (Table 2)：

Replica metrics (threshold $\tau = 5$cm):
- **Go-Surf**: F-score 0.990, C-$L_1$ 0.012 cm (SOTA on Replica)
- Du-NeRF: F-score 0.991
- Neural RGB-D: F-score 0.847

ScanNet metrics:
- **H2O-SDF**: F-score 0.801, C-$L_1$ 0.035 cm (SOTA)
- NeuRIS: F-score 0.794
- MonoSDF-MLP: F-score 0.733

**Intuition**：Replica和ScanNet的性能差异很大。Replica是synthetic high-quality数据，F-score能到0.99；ScanNet是real-world noisy data，最好也只到0.80。这说明real-world reconstruction仍有巨大improvement空间。ManhattanSDF的Manhattan-world assumption（室内场景主要由axis-aligned planes组成）在planar regions能提供强prior，但对non-planar场景不适用。

### 3.2 Deformable Reconstruction

分为两类：
- **Physics-based**：结合elastic deformation, kinematics, dynamics
  - Explicit approach (Neural Impostor, Pie-NeRF)：先重建mesh/particles，再做物理仿真
  - Implicit approach (Chen et al., PAC-NeRF)：将physical simulation嵌入loss function，需要differentiable simulator
  
- **Deformation with dynamic sequence**：从video学deformation field
  - **D-NeRF** [74]：canonical space + deformation network
  - **Nerfies** [75]：template NeRF + deformation code + appearance code
  - **HyperNeRF** [76]：扩展到5D hyper-canonical space
  - **K-Planes, HexPlane**：用feature planes explicit表示canonical space和时间

**Intuition**：deformable reconstruction的核心思路是"canonical space + deformation field"。所有时间步的scene都deform回canonical space，在那里用一个standard NeRF建模。这种decoupling让模型能处理topology changes和large motions。对robotics，physics-based方法尤其有价值——它们能预测interaction后的deformation，这对manipulation planning至关重要。

Reference: [D-NeRF](https://arxiv.org/abs/2011.13961), [PAC-NeRF](https://arxiv.org/abs/2303.05512)

---

## 4. Segmentation — 三大类别

### 4.1 Segmentation Interpolation

核心idea：利用NeRF的interpolation能力，从sparse labels传播到dense segmentation。

**Semantic-NeRF** [103]是开创性工作：
- 在NeRF MLP上append segmentation head
- 训练loss = photometric loss + cross-entropy semantic loss
- 能从~1% labeled pixels学到dense segmentation
- 支持denoising, super-resolution, label propagation

**NeSF** [106]：
- 先训练standard NeRF，提取density grid
- 用3D U-Net将density grid转换为semantic feature grid
- Semantic MLP从feature grid输出semantic rendering
- 优势：能generalize到unseen scenes

**JacobiNeRF** [107]：
- Encoding semantic correlations between scene points
- 用Jacobian alignment做regularization
- 优于Semantic-NeRF在label propagation task上

### 4.2 Multi-view Consistent Segmentation

解决2D segmentation across views的inconsistency问题。

**SUDS** [81]：factorize scene into static, dynamic, far-field三个hash tables
**Instance-NeRF** [111]：用NeRF-RCNN从pre-trained NeRF提取3D masks，纠正2D predictions
**Panoptic Lifting** [114]：从machine-generated 2D masks lift到consistent 3D panoptic segmentation
**Contrastive Lift** [115]：用slow-fast clustering objective，去除upper bound on object count

### 4.3 Benchmark Analysis (Table 3, 4)

**Outdoor (KITTI-360)**:
- NeRF + PSPNet: mIoU 53.01
- Panoptic NeRF: mIoU 81.1, PQ 64.4 (all classes)
- 3D-2D CRF: mIoU 79.5

**Room-level (Replica)**:
- Mask2Former baseline: mIoU 52.4
- Panoptic Lifting: mIoU 67.2, PQ_scene 57.9
- Contrastive Lift (SF): PQ_scene 59.1
- Semantic Ray (finetuning): mIoU 76.0, Acc 80.8

**Intuition**：segmentation task的核心挑战是multi-view consistency。Single-view 2D segmentation methods (Mask2Former)在不同views给出inconsistent labels。NeRF-based methods通过3D scene structure的约束，自然获得consistency。但pure NeRF方法(18.7 mIoU)远不如combined approaches (76.0 mIoU)，说明geometry alone不够，需要strong 2D priors。

Reference: [Semantic-NeRF](https://arxiv.org/abs/2104.04521), [Panoptic Lifting](https://arxiv.org/abs/2212.00824)

---

## 5. Pose Estimation — 三大paradigm

### 5.1 Inverse Optimization

核心idea：固定NeRF weights，优化camera pose。

**iNeRF** [140]：
- 假设NeRF已pre-trained
- 固定network weights $\Theta$
- 优化pose $T = [R|t]$，minimize rendering loss

$$\min_{R, t} \sum_{\mathbf{r}} \|\hat{C}(\mathbf{r}; R, t) - C_{gt}(\mathbf{r})\|^2$$

**BARF** [145]：Bundle-adjusting NeRF，同时优化network weights和poses
**SPARF** [147]：多view feature point correspondence约束，sparse views
**L2G-NeRF**：Local-to-global alignment with 3D point clouds

### 5.2 Data Augmentor

用NeRF合成novel views训练pose regressor。

**LENS** [153]：
1. 训练NeRF-W在outdoor scene
2. 用real + rendered images训练CoordiNet pose regressor

**LATITUDE** [156]：Two-stage
- Stage 1: Mega-NeRF合成views训练pose regressor
- Stage 2: iNeRF-style pose refinement

**Loc-NeRF** [163]：Particle filter + NeRF rendering for Monte Carlo localization

### 5.3 Feature Matching

用3D latent code做direct matching。

**Nerfels** [164]：locally dense, globally sparse 3D representation，PnP求解pose
**NeuMap** [167]：Scene coordinate regression with scene-agnostic transformers
**CROSSFIRE** [171]：学习descriptor field，establish 2D-3D correspondences
**NeRFMatch** [174]：直接学NeRF features和image features之间的match

### 5.4 Benchmark (Table 5)

**7-Scenes (indoor)** — median translation/rotation error [m/deg]:
- LENS: 0.08 / 3.0
- IMA: 0.04 / 1.1
- NeuMap: 0.03 / 1.1
- CROSSFIRE: 0.04 / 1.1
- **NeRFMatch: 0.02 / 0.7** (SOTA)

**Cambridge Landmarks (outdoor)**:
- LENS: 0.39 / 1.2
- NeuMap: 0.14 / 0.3
- **NeRFMatch: 0.08 / 0.4** (SOTA)

**Intuition**：pose estimation的三个paradigm反映了不同trade-off：
1. **Inverse optimization** (iNeRF)：accurate但slow，需要pre-trained NeRF
2. **Data augmentor** (LENS)：fast inference但需要大量合成数据
3. **Feature matching** (NeRFMatch)：兼顾accuracy和speed，是current SOTA

NeRFMatch的成功在于它learn了NeRF intermediate features和CNN image features之间的direct correspondence，避免了expensive rendering-based optimization。

Reference: [iNeRF](https://arxiv.org/abs/2012.05877), [BARF](https://arxiv.org/abs/2104.00152), [NeRFMatch](https://arxiv.org/abs/2403.09577)

---

## 6. SLAM — NeRF-based SLAM的三大类

### 6.1 RGB-D SLAM

**iMAP** [183]是开创性工作：
- Single MLP表示整个scene
- 增量优化NeRF for mapping
- Inverse optimization for tracking
- **Limitation**：single MLP的memory issue，forgetting problem

**NICE-SLAM** [8]：
- Voxel grid features + pretrained MLP decoders
- Multi-resolution (coarse, mid, fine)
- **Limitation**：fixed pretrained MLPs，generalization受限

后续工作的两大方向：

**Mapping innovations**:
- **MeSLAM** [184]：segmented map regions across multiple networks
- **Point-SLAM** [193]：neural point clouds，dynamically anchor features
- **iSDF** [194]：first SDF-based neural implicit SLAM
- **ESLAM** [201]：multi-scale feature planes + shallow decoders → TSDF
- **Co-SLAM** [202]：hybrid coordinate + sparse parametric encodings

**Tracking innovations**:
- 大多数采用joint learning
- 部分用conventional VO initialization (ORB-SLAM3, Droid-SLAM)

### 6.2 Monocular SLAM

Two approaches:
- **End-to-end joint mapping + tracking** (NICER-SLAM, Dense RGB SLAM)
- **Decoupled tracking + mapping** (Orbeez-SLAM, NeRF-VO, NeRF-SLAM, HI-SLAM, GO-SLAM)

Decoupled方法通常用mature VO system (ORB-SLAM, Droid-SLAM)做tracking，NeRF只做mapping。

### 6.3 LiDAR SLAM

**LONER** [223]：First neural implicit LiDAR SLAM
- Point-to-plane ICP for tracking
- Hierarchical feature grid encoding
- Dynamic margin loss + depth/sky losses

**NeRF-LOAM** [225]：Octree-based voxels for large-scale outdoor
**PIN-SLAM** [228]：Point-based implicit neural representation for global consistency

### 6.4 Benchmark Analysis (Table 6, 7, 8)

**Replica Mapping Quality** (Table 6):
| Method | Acc.[cm]↓ | Comp.[cm]↓ | C.R.[%]↑ | Dep.L1↓ |
|--------|----------|----------|---------|--------|
| iMAP | 6.95 | 5.33 | 66.60 | 7.64 |
| NICE-SLAM | 2.85 | 3.00 | 89.33 | 3.53 |
| Point-SLAM | 1.79 | 2.39 | 92.99 | 0.44 |
| **ESLAM** | **0.97** | **1.05** | **98.60** | 0.77 |
| Co-SLAM | 1.94 | 1.70 | 96.62 | 0.77 |
| NeRF-SLAM (RGB) | 3.10 | 3.59 | 81.30 | 4.49 |

**Replica Tracking RMSE [cm]** (Table 7):
- DNS-SLAM: 0.45 (RGB-D SOTA)
- CP-SLAM: 0.46
- Point-SLAM: 0.52
- Li et al. (RGB): 0.46
- NeRF-VO (RGB): 0.43

**Intuition**：
1. **ESLAM在mapping上SOTA**，得益于multi-scale feature planes和TSDF representation
2. **Tracking上RGB方法已经接近RGB-D**，这说明monocular depth estimation的进步
3. **iMAP → NICE-SLAM → ESLAM**的演进展示了从single MLP到multi-resolution grid的architecture进步
4. **C.R. (Completion Ratio)**衡量reconstruction的completeness，从iMAP的66.6%到ESLAM的98.6%是巨大提升

Reference: [iMAP](https://arxiv.org/abs/2104.00604), [NICE-SLAM](https://arxiv.org/abs/2112.12230), [ESLAM](https://arxiv.org/abs/2211.00503), [Co-SLAM](https://arxiv.org/abs/2304.14377)

---

## 7. Planning & Navigation

### 7.1 Navigation Methods

**NeRF-Nav** [236]：
- Pre-trained NeRF作为environment model
- Trajectory optimization avoiding high-density regions
- Optimization-based filtering for pose/velocity estimation

**CATNIPS** [237]：
- 将NeRF转换为Poisson Point Process (PPP)
- Rigorous uncertainty quantification for collision
- Probabilistic Unsafe Robot Region (PURR) voxel representation

**RNR-Map** [239]：
- Latent-code-based NeRF
- Grid map存储latent codes而非occupancy
- Enable visual localization + navigation

**CBF-NeRF** [241]：
- NeRF提供single-step visual foresight
- Control Barrier Functions for safe control

### 7.2 Active Mapping

Core idea：机器人主动选择views来maximize information gain。

**UGP-NeRF** [243]：
- Ray-based volumetric uncertainty estimator
- Entropy of weight distribution along each ray
- Next-best-view selection policy

**NeruAR** [244]：
- Color prediction as Gaussian distribution
- Variance connected to PSNR by linear relationship
- Variance as PSNR proxy for image quality

**ActiveNeRF** [245]：
- Select samples with most information gain
- Uncertainty reduction given new inputs

**NARUTO** [251]：
- Multi-resolution hash grid as mapping backbone
- Uncertainty aggregation for goal searching + path planning

### 7.3 Metrics

Two evaluation goals：
1. **Navigation performance**: Success Rate (SR), Success weighted by Path Length (SPL)
2. **Reconstruction quality**: F-score, IoU, Chamfer distance

**Intuition**：NeRF在navigation中的核心价值是提供dense, continuous的environment representation。Traditional occupancy grids是binary且discrete；NeRF的density field是continuous的，能model "soft" obstacles和uncertainty。Active mapping特别有意思——机器人能主动探索uncertain regions，这对应人类"curiosity-driven" exploration的intuition。

Reference: [NeRF-Nav](https://arxiv.org/abs/2110.00168), [CATNIPS](https://arxiv.org/abs/2302.12931)

---

## 8. Interaction — Grasping & Manipulation

### 8.1 Open-loop Grasping

"Perceive-Plan-Execute" pipeline：
1. **Observation**: NeRF从multi-view images重建3D geometry
2. **Planning**: 从NeRF提取mesh/depth/normal，生成grasp proposals
3. **Execution**: Robot执行grasping

**Special cases**:
- **Transparent objects** (Dex-NeRF [260], GraspNeRF [261]): NeRF的view-dependent modeling特别适合透明物体
- **Reflective objects** (NFL [264]): 用normal cues代替RGB
- **Unknown objects** (Chen et al. [265]): Poking strategy生成reconstructions

### 8.2 Closed-loop Control

Two methodologies：
- **Imitation Learning** (behavior cloning)：NeRF生成visual-action datasets
- **Reinforcement Learning**：NeRF作为state encoder或sim-to-real transfer

**Byravan et al.** [270]：NeRF of real scenes → realistic novel views for policy learning
**Driess et al.** [271]：NeRF作为feature encoder，freeze后train policy network

### 8.3 Intuition

NeRF在manipulation中的价值：
1. **Transparent/reflective objects**：传统depth sensor失败，NeRF的view-dependent rendering能正确建模
2. **Sim-to-real transfer**：NeRF能生成photorealistic synthetic data，缩小sim-real gap
3. **Latent state representation**：NeRF的intermediate features是compact, view-invariant的state representation，适合RL

Reference: [Dex-NeRF](https://arxiv.org/abs/2110.14217), [NeRF-Supervision](https://arxiv.org/abs/2203.01913)

---

## 9. Future Directions

### 9.1 3D Gaussian Splatting (3DGS)

3DGS用一组unordered points表示scene，每个point有：
- Position $\mathbf{p}$
- Color $\mathbf{c}$
- Opacity $o$
- Covariance $\Sigma$

Gaussian shape：

$$G(\mathbf{x}) = \exp\left(-\frac{1}{2}\mathbf{x}^T \Sigma^{-1} \mathbf{x}\right)$$

Screen space projection：

$$\Sigma' = JW \Sigma W^T J^T$$

变量解释：
- $J$：projective transformation的Jacobian
- $W$：world transform matrix
- $\Sigma'$：screen-space covariance

**Advantages over NeRF**:
1. **Training speed**: 几分钟vs几小时
2. **Rendering quality**: 通常优于NeRF
3. **Convergence basin**: 更大，对pose estimation更robust

**Current limitations**:
1. Geometry reconstruction质量不如SDF-based NeRF
2. Training time仍对real-time robotics不够（2-3小时tracking in [288]）

**Intuition**：3DGS本质上是explicit representation（points with attributes），但用differentiable rasterization训练。它trades off了NeRF的compactness（MLP weights）for speed和quality。对robotics，这意味着可以实时更新map，但memory consumption更高。

Reference: [3DGS paper](https://arxiv.org/abs/2308.14539), [SplaTAM](https://arxiv.org/abs/2312.02126), [Gaussian-SLAM](https://arxiv.org/abs/2312.10070)

### 9.2 Large Language Models (LLMs) Integration

**LERF** [291]：Distill CLIP features into NeRF，enable textual queries
**LERF-TOGO** [292]：+ DINO for zero-shot semantic grasping
**OV-NeRF** [293]：+ SAM for open-set semantic understanding

**Intuition**：LLMs提供open-vocabulary scene understanding。传统semantic segmentation受限于predefined classes；CLIP distillation让NeRF能respond to arbitrary text queries。这对human-robot interaction至关重要——用户可以说"pick up the red mug"而不是predefined class ID。

Reference: [LERF](https://arxiv.org/abs/2303.09553), [CLIP](https://arxiv.org/abs/2103.00020)

### 9.3 Generative AI

**Challenges**:
- NeRF是fitting method，每个scene从scratch训练
- 缺乏3D general knowledge
- Few-shot reconstruction困难

**Approaches**:
- **GAN-based** (pi-GAN [295], GRAF [296]): 3D-aware generation
- **VAE-based** (NeRF-VAE [297]): Variational autoencoder for OOD generation
- **Diffusion-based** (DiffRF [298], PoseDiff [299], Shap-E [300]): Photorealistic generation from text

**Intuition**：Generative AI能provide 3D priors that NeRF lacks。Diffusion models特别有前景——它们能complete scenes from sparse observations，这对robotics的few-shot scenario非常重要。但computational cost是巨大challenge。

Reference: [Shap-E](https://arxiv.org/abs/2305.02463), [GRAF](https://arxiv.org/abs/2007.02442)

---

## 10. 总体Intuition Building

让我总结几个high-level insights：

### 10.1 Representation Trade-offs

| Representation | Compactness | Quality | Speed | Editability |
|---------------|------------|---------|-------|-------------|
| MLP (vanilla NeRF) | High | Medium | Slow | Hard |
| Voxel Grid | Low | High | Fast | Medium |
| Multi-resolution Hash (Instant-NGP) | Medium | High | Fast | Hard |
| Point-based (3DGS) | Low | High | Fast | Easy |
| SDF-based | Medium | High (geometry) | Medium | Medium |

### 10.2 Robotics-specific Challenges

1. **Real-time requirement**: 大多数NeRF methods仍offline training，对real-time robotics不够
2. **Memory constraints**: Robot hardware有限，large MLPs或dense voxel grids不practical
3. **Generalization**: 每个scene从scratch训练，无法leverage prior knowledge
4. **Dynamic environments**: Real world是dynamic的，static NeRF assumption常被违反
5. **Sensor fusion**: Robots有multiple sensors (RGB, depth, LiDAR, IMU)，如何fuse进neural representation是open question

### 10.3 最有前景的方向

基于这篇survey的分析，我认为最有前景的方向是：

1. **3DGS for SLAM**：实时mapping + tracking，已show SOTA results
2. **LLM-embedded NeRF**：Open-vocabulary scene understanding for natural human-robot interaction
3. **Generative priors**：Diffusion models提供3D priors，enable few-shot reconstruction
4. **Active perception**：Robot主动探索uncertain regions，结合uncertainty quantification
5. **Differentiable physics**：PAC-NeRF风格的physics-augmented NeRF，enable predictive manipulation

### 10.4 Benchmark标准化需求

Survey指出LiDAR SLAM部分缺乏standardized metrics across papers。这是一个重要的gap——需要统一的benchmark来fairly compare methods。类似地，deformable reconstruction也"no unified dataset and metrics are currently available"。

---

## Web Reference汇总

- [Original NeRF paper (Mildenhall et al. ECCV 2020)](https://arxiv.org/abs/2003.08934)
- [NeuS (Wang et al. NeurIPS 2021)](https://arxiv.org/abs/2106.10689)
- [UNISURF (Oechsle et al. ICCV 2021)](https://arxiv.org/abs/2104.10078)
- [Instant-NGP (Müller et al. SIGGRAPH 2022)](https://arxiv.org/abs/2201.05989)
- [iMAP (Sucar et al. ICCV 2021)](https://arxiv.org/abs/2104.00604)
- [NICE-SLAM (Zhu et al. CVPR 2022)](https://arxiv.org/abs/2112.12230)
- [ESLAM (Johari et al. CVPR 2023)](https://arxiv.org/abs/2211.00503)
- [Co-SLAM (Wang et al. CVPR 2023)](https://arxiv.org/abs/2304.14377)
- [3DGS (Kerbl et al. SIGGRAPH 2023)](https://arxiv.org/abs/2308.14539)
- [Semantic-NeRF (Zhi et al. ICCV 2021)](https://arxiv.org/abs/2104.04521)
- [Panoptic Lifting (Siddiqui et al. CVPR 2023)](https://arxiv.org/abs/2212.00824)
- [iNeRF (Yen-Chen et al. IROS 2021)](https://arxiv.org/abs/2012.05877)
- [BARF (Lin et al. ICCV 2021)](https://arxiv.org/abs/2104.00152)
- [Dex-NeRF (Ichnowski et al. CoRL 2021)](https://arxiv.org/abs/2110.14217)
- [LERF (Kerr et al. 2023)](https://arxiv.org/abs/2303.09553)
- [PAC-NeRF (Li et al. 2023)](https://arxiv.org/abs/2303.05512)
- [SplaTAM (Keetha et al. CVPR 2024)](https://arxiv.org/abs/2312.02126)
- [DTU Dataset](https://roboimagedata.compute.dtu.dk/?page_id=36)
- [Replica Dataset](https://arxiv.org/abs/1906.05797)
- [ScanNet](http://www.scan-net.org/)
- [KITTI](http://www.cvlibs.net/datasets/kitti/)
- [TUM RGB-D](https://vision.in.tum.de/data/datasets/rgbd-dataset)

---

这篇survey的价值在于它systematic地categorize了NeRF在robotics中的应用，并提供了extensive benchmarks。希望这个technical walkthrough帮你build intuition about NeRF在autonomous robots中的potential和limitations。如果你对某个specific topic（比如SLAM的tracking algorithms，或3DGS的future potential）想深入讨论，随时告诉我！
