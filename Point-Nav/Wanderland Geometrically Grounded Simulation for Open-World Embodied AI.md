---
source_pdf: Wanderland Geometrically Grounded Simulation for Open-World Embodied AI.pdf
paper_sha256: 0f21258b27da793b93daa47d14c2e9819a494c5794c5622048a4d80412ab7a1f
processed_at: '2026-08-13T03:37:10-07:00'
target_folder: Point-Nav
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 Wanderland

Andrej，咱们把这篇 paper 翻译成人话。

## 一句话讲完

想用随手拍的视频建个 virtual world 来训练 robot navigation？不行，视频天生不知道"一米有多长"，建出来的世界是歪的。robot 在歪世界里学走路，学出来的 policy 也是歪的。作者搞了个带 LiDAR 的手持扫描仪，把真实世界精准地搬进 simulation，这样训练和评测 robot 才靠谱。

---

## 为什么视频不行——三个要命的问题

### 问题 1: 视频不知道"一米有多长"

你拿手机拍一段街景视频，丢给 COLMAP 或者 VGGT 这种 vision model 去 reconstruct，它能告诉你"camera 在哪、朝哪看"，但这个坐标系是 arbitrary scale 的。可能是米，可能是英尺，可能是某个 unknown unit。这叫 **scale ambiguity**。

Table 2 的数据特别直观。作者把 8 个最 SOTA 的 vision-only reconstruction 方法跑在同一批 data 上，取 "Best of All"（每个 metric 挑所有方法里最好的那个），结果：

- **T-ATE^S = 0.30 m**（scale 对齐后仍有 30cm 误差）
- **R-ATE = 5°**（旋转误差 5 度）

在一个不到 100 米长的 scene 里，camera pose 误差 30cm + 5°。对 navigation 来说这意味着什么？意味着 robot 以为自己在门口，其实撞墙上了。

**公式讲讲**（T-ATE^S）：

$$\text{T-ATE}^S = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\|s \cdot R_{SIM3}\mathbf{t}_i^{pred} + \mathbf{t}_{SIM3} - \mathbf{t}_i^{gt}\|^2}$$

- $N$: camera pose 总数
- $\mathbf{t}_i^{pred}$: 第 $i$ 个 predicted camera 的平移
- $\mathbf{t}_i^{gt}$: ground truth 平移
- $R_{SIM3}, \mathbf{t}_{SIM3}, s$: SIM(3) alignment 求出来的 rotation、translation、scale

SIM(3) 就是 SE(3) 加上一个 scale freedom，等于"我承认你不知道绝对尺度，让我帮你 best-effort 对齐一下"。对齐完还剩 30cm，说明 vision-only 在 metric accuracy 上根本没法用。

### 问题 2: 建出来的"墙"是碎的

Embodied navigation 需要 collision mesh——robot 得知道哪能走哪不能走。Video-3DGS 方法从 Gaussian opacity field 里 extract mesh，结果是碎片化的、充满 noise 的。

Figure 5 里直观对比了三家的 mesh：
- **Vid2Sim** [27]: mesh 像被炸过一样，碎片横飞
- **GaussGym** [29]: mesh 大面积缺失，根本连不上
- **Wanderland (ours)**: 从 LiDAR point cloud 直接 Marching Cubes，干净光滑

为什么 vision-only 的 mesh 这么烂？因为 3DGS 的 opacity 本来就不是一个明确的 surface 表示——它是一堆 ellipsoidal blob 的叠加，每个 blob 有一个 opacity 值。你想从里面切出一个 surface，就像从一团云里切出墙的形状，本身就 ill-posed。

而 LiDAR 直接给你每个点的 3D 坐标，你 voxelilze 之后 Marching Cubes 就是干净的 occupancy-based mesh。

### 问题 3: 偏离拍摄路线就糊了

YouTube touring videos 的 camera trajectory 是"一条线走过去"，view diversity 极差。3DGS 在训练 view 附近还行，稍微偏一点就糊成一坨——因为 3DGS 本质上是在"记住"看到过的视角，没看到的地方它就瞎编。

这对 navigation 是致命的：robot 需要在 training trajectory 之外的地方看世界，因为 evaluation 时的 path 不会和 capture path 完全重合。

---

## Wanderland 怎么解决的

### 硬件：MetaCam 扫描仪

Figure 2 那个手持设备，集成了：
- **Livox Mid-360 LiDAR** + 内置 IMU
- **RTK-GNSS**（厘米级 GPS）
- **两个 4K fisheye camera**，180°+ FOV

LiDAR 倾斜安装，为了多扫地面（robot 走的地方）和让 FOV 跟 camera 重叠。

**采集策略**很讲究：
- 不是按固定 frame rate 拍，而是**走了一定距离或转了一定角度才 trigger**——这样 viewpoints 在空间上均匀分布
- **Training trajectory** 是 closed-loop dense coverage（把所有能走的地方密集扫一遍）
- **Extrapolation trajectory** 是模拟自然 navigation path，view overlap 很小——专门留出来 evaluation

这就是 Figure 3 的设计，很 elegant：training 和 evaluation split 在采集时物理分开。

**Location**: NYC + Jersey City，530 个 scene，3.8M m²，420K frames，覆盖住宅、商业区、街道、广场、校园。

### 重建：LIV-SLAM

MetaCam Studio 跑一个 LiDAR-Inertial-Visual-GNSS fusion pipeline（基于 Fast-LIVO2 [55]、R3LIVE [56]、VINS-Mono [49] 这类方法的扩展）。

产出：
1. **Dense, metric-scale point cloud**（5-10mm spacing）
2. **Globally consistent camera poses**（metric accuracy）

为什么 multi-sensor fusion 是 sweet spot？

| Sensor | 提供 | 缺什么 |
|--------|------|--------|
| LiDAR | metric geometry, depth | 无 appearance |
| RGB | appearance, semantics | 无 scale, 室外易失败 |
| IMU | high-rate motion | 长期 drift |
| GNSS | global consistency | 室内无信号、精度有限 |

单独任何一个都不够。Fusion 起来互补。

### 3DGS Training 的几个 trick

**Trick 1: 从 LiDAR point cloud 初始化 Gaussian**

原始 point cloud 10-50M 个点，downsample 到 ~5M，每个点一个 Gaussian。

初始化时 opacity 设成：
$$\alpha_i \propto \frac{1}{V_i}$$
- $\alpha_i$: 第 $i$ 个 Gaussian 的初始 opacity
- $V_i$: 该 Gaussian 的体积（KNN heuristic 估计）

**Intuition**: 体积大的 Gaussian 初始 opacity 低。为什么？因为大 Gaussian 往往是 transient obstacle（路人、车）留下的 spurious floater，你不想让它在 training 早期 dominate rendering。这是个 anti-floater trick。

**Trick 2: Depth regularization——但不 freeze Gaussian centers**

作者尝试过"既然 LiDAR 这么准，干脆把 Gaussian 中心 freeze 住只学 appearance"——结果 Table II 显示反而更差。

- Frozen: extrapolated SSIM 0.558
- Depth loss (不 freeze): extrapolated SSIM 0.591

为什么 freeze 反而差？因为 LiDAR point cloud 虽然 metric 准，但它是离散采样，缺 high-frequency details。Freeze 住等于强加一个 uniform geometric prior，模型没法 fit 图像里的细节。Depth loss 是软约束——"大致别离 LiDAR 太远，但允许微调"。

Depth loss 公式：
$$\mathcal{L}_{depth} = \sum_{rays} \|D_{pred}(r) - D_{init}(r)\|^2$$
- $D_{pred}(r)$: 当前 3DGS 沿 ray $r$ 渲染出的 depth
- $D_{init}(r)$: 从 initialized (LiDAR) Gaussians 渲染出的 depth，当 pseudo-GT

注意这里没用 monocular depth estimation（Depth Anything 之类），而是用 LiDAR initialized Gaussian 自己渲染出来的 depth 当 GT。这是 Wanderland 跟 Vid2Sim 的关键区别——Vid2Sim 用 monocular depth [87] 当 supervision，引入了 depth prediction 的 noise。

**Trick 3: Difix3D+ 做 view augmentation**

用 Difix3D+ [77]（一个 single-step diffusion model，专门修 3D reconstruction artifacts）来 augment training views：

1. Sample 一个 extrapolated camera pose
2. 从当前 3DGS rasterize 一张图（可能很糊）
3. 把这张糊图 + 最近的 training views 喂给 Difix3D+
4. Difix3D+ 输出一张 cleaned + geometrically accurate 的 novel view
5. 把这张图加回 training set，但用 lower loss weight（不抢 original observation 的戏）

**Curriculum**: 早期只 sample 训练轨迹附近的 novel view，随 training step 渐进扩大半径。这稳定了 large-scale 3DGS training。

Table III 的 ablation 显示 view augmentation 让 extrapolated PSNR 从 16.97 → 17.92，提升明显。

### Mesh Extraction

从 global point cloud 出发：
1. Filter 掉离 trajectory 太远的点
2. Voxelize 成 occupancy grid
3. **Marching Cubes** [78] 提取 triangle mesh
4. 移除 <50 faces 的 fragment

关键 design choice: mesh 只管 collision，不管 appearance（appearance 归 3DGS）。所以不需要 watertight，不需要 fine-grained detail，只要 occupancy-accurate。

然后 mesh + 3DGS 一起塞进 **USD (Universal Scene Description)** 格式，直接 load 进 Isaac Sim。

---

## 实验数据讲讲

### RL Training 的对比（Table 4）

这是最 striking 的结果。作者把三个 navigation model（NoMaD, CityWalker, MBRA）在两种 env 里做 RL post-training：

**在 Vid2Sim env 里**：
- NoMaD: SR +0%, SPL **-4%**
- CityWalker: SR **-21%**, SPL **-19%**
- MBRA: SR +0%, SPL **-5%**

**在 Wanderland env 里**：
- NoMaD: SR **+8%**, IR **-13%**
- CityWalker: SR **+14%**, SPL **+14%**, IR **-23%**
- MBRA: SR **+5%**, IR **-7%**

Vid2Sim env 里 RL 训练**让 model 变差**了。这听起来反直觉——RL 不是应该让 policy 更好吗？

Intuition: 在 inaccurate geometry 里，RL reward 鼓励 agent "抄近路"。但抄的近路可能穿过了 mesh 的 gap（mesh 碎片化导致的虚假通道），或者被 phantom obstacle（noise 产生的虚假障碍）挡住。Agent 学会了 exploit 这些 sim artifact，eval 时在真实 env 里就 crash。

这就是 **sim-to-real gap 的几何版**——不只是纹理不像，连空间结构都不对。

### Navigation Benchmark（Table 5）

在 Wanderland 整个 dataset 上评测各家 model：

| Method | Indoor SR | Outdoor SR |
|--------|----------|------------|
| NoMaD [90] | 0.22 | 0.24 |
| CityWalker [21] | 0.39 | 0.21 |
| MBRA [91] | 0.35 | 0.22 |
| NaVid [92] | 0.29 | 0.15 |
| NaVILA [71] (VLN) | 0.47 | 0.31 |

三个观察：
1. **VLN 比 point-goal/image-goal 容易**——NaVILA (VLN) 最好，因为 LLM 帮忙理解 instruction
2. **Outdoor 比 Indoor 难**——trajectory 更长、topology 更复杂、有 elevation change
3. **没有 model SR > 50%**——open-world navigation 还差得远

### NVS 质量（Table 3）

| Method | Interp PSNR | Extrap PSNR | Extrap LPIPS |
|--------|-----------|------------|-------------|
| 3DGS (COLMAP) | 18.27 | 16.90 | 0.559 |
| Vid2Sim | 17.20 | 16.49 | 0.371 |
| GaussGym | 12.17 | 12.63 | 0.725 |
| **Wanderland** | **20.37** | **17.92** | **0.445** |

GaussGym 的 PSNR 12.17 基本是糊的，因为 VGGT [30] reconstruction 本身就烂。Vid2Sim 的 LPIPS 反而比 Wanderland 好，但 PSNR/SSIM 差，说明它在 perceptual 层面"看着像"但 pixel-level 不准。

### Semantic Consistency（Figure 6）

这图特别 informative。作者不只看 PSNR，还看渲染出来的图能不能被 perception model 正确理解：

- **Grounded SAM 2** [86] 在 GaussGym 渲染图上 segmentation 直接失败——因为渲染太碎，SAM 根本分不出 object boundary
- **DINOv3** [85] features 在 Vid2Sim 渲染图上跟 GT 严重偏离（PCA colors 完全不同）——这对用 DINO features 做 end-to-end navigation 的 policy [21] 是致命的

Wanderland 渲染出来的图，SAM 和 DINOv3 都能正常工作。这说明 **rendering quality 不只是 PSNR，还有 semantic fidelity**——perception model 依赖的那些 mid-level features 要跟真实图像一致。

---

## 对你的直觉有什么启发

Andrej，你一直在思考 vision-only 的极限。这篇 paper 用实验数据论证了一个观点：

**Vision-only reconstruction 的 metric accuracy 天花板比我们以为的低**。即使是 2025 年最 SOTA 的 foundation model（VGGT, MapAnything, DA3），在 <100m scene 里 scale-aligned 后还有 30cm/5° 误差。这不是"再多数据/更大模型"能填的 gap，是 modality limitation。

为什么？因为 monocular depth 本质是 ill-posed——从 2D 推 3D 尺度，信息量不够。Scale 只能从运动或已知 size 物体推断，这两个 source 都 fragile。

LiDAR 为什么要存在？不是因为 vision 做不到"看起来对"，而是因为 vision 做不到"metrically 对"。而 embodied AI 恰恰需要 metrically 对——robot 撞墙差 30cm 就是撞上和没撞上的区别。

这跟你在 Tesla 看到的情况有 mirror image 的意味：Tesla 走纯 vision 路线是因为量产约束，但在 simulation benchmarking 这种"可以奢侈用更多 sensor"的场景，multi-sensor fusion 仍然有明显优势。

**Limitations**:
- 1 FPS capture rate（硬件限制）→ viewpoint 稀疏
- 只处理 static environment（没建模 dynamic agent）

这俩都是 future work 的明显方向。可以想象下一个版本会上 higher-rate capture + dynamic object modeling（4D reconstruction？）。

---

**相关 links**:
- Project page: https://ai4ce.github.io/wanderland/
- Paper (arXiv 应该会放): 关注 NYU AI4CE Lab https://ai4ce.github.io/
- Vid2Sim: https://www.cs.columbia.edu/~bzhao/vid2sim/
- GaussGym: https://arxiv.org/abs/2510.15352
- 3DGS original: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- VGGT: https://vgg-t.github.io/
- Difix3D+: https://github.com/NVlabs/DIFIX3D
- CityWalker（同 lab 前作）: https://ai4ce.github.io/citywalker/
- Fast-LIVO2: https://github.com/hku-mars/FAST-LIVO2
- gsplat library: https://github.com/nerfstudio-project/gsplat
- Isaac Sim: https://developer.nvidia.com/isaac-sim

---

# Wanderland: Geometrically Grounded Simulation for Open-World Embodied AI

## 1. 核心 Intuition: 为什么需要 Geometric Grounding

Andrej, 这篇paper的核心 thesis 可以一句话概括：**当前基于 video 的 3DGS simulation (比如 Vid2Sim, GaussGym) 在 embodied navigation benchmarking 中是不可信的，因为 vision-only reconstruction 无法提供 metric-scale 的 geometry**。作者用 LiDAR+IMU+RGB 多传感器 fusion 来解决这个问题。

让我先 build 一下 intuition。Embodied AI 的 closed-loop evaluation 需要：
- **Photorealistic rendering** (用于 perception)
- **Geometrically grounded collision/physics** (用于 interaction)

经典 indoor datasets (Matterport3D [9], ScanNet [11], HM3D [7]) 用 tripod-based RGB-D 提供 metric scale 但只限 indoor。Video-3DGS 方法扩展到 outdoor 但丢失了 metric accuracy。这是一个 fundamental trade-off，Wanderland 试图兼得两者。

Project page: https://ai4ce.github.io/wanderland/

---

## 2. Video-3DGS Pipelines 的三大缺陷

作者系统地诊断了 vision-only pipelines 的 failure modes：

### 2.1 Inaccurate 3D Reconstruction
基于 SfM [28, 36] 或 deep reconstruction models [30, 37] 产生 non-metric camera poses 和 depth。Table 2 显示即使取 "Best of All" vision-only methods，T-ATE^S (scale-aligned translation error) 仍有 0.30m，R-ATE 仍有 5°。

### 2.2 Unreliable Geometry
从 3DGS opacity fields 提取的 collision meshes 是 fragmented 和 metrically ungrounded [38, 39]。Figure 5 显示 Vid2Sim 和 GaussGym 的 mesh 充满 noise 和 fragmentation。

### 2.3 Extrapolated View Degradation
City touring videos 的 trajectories 是 uni-directional 的 (Figure 1 第一列)，导致 training views 不够 diverse，rendering 在 off-trajectory viewpoints 急剧退化 [40]。

---

## 3. WANDERLAND Framework 详解

### 3.1 Data Collection — MetaCam Device

硬件配置 (Figure 2):
- **Livox Mid-360 LiDAR** (non-repetitive scanning) + built-in IMU
- **RTK-GNSS antenna** (global positioning)
- **两个 4K fisheye cameras** (>180° FOV, synchronized)
- LiDAR 倾斜安装以优化 ground-level capture 和与 cameras 的 FOV overlap

**关键设计决策**：
- **Distance/angular threshold trigger** 而非 fixed frame rate — 确保 viewpoints 在空间上均匀分布
- **Closed-loop training trajectories** + **extrapolation trajectories** (Figure 3) — 前者 dense multi-view coverage 用于 reconstruction，后者 simulate natural navigation 用于 evaluation
- **Real-time point cloud preview** — operator 可以主动 fill gaps

**Limitation**: 1 FPS due to hardware constraints，导致 viewpoint sampling 稀疏。

### 3.2 Data Processing — LIV-SLAM

MetaCam Studio 实现 **LiDAR-Inertial-Visual-GNSS sensor fusion**，基于 [49, 55, 72] 的方法论扩展。产出：
- Dense, metric-scale point clouds
- Globally consistent camera trajectories

**为什么 LIV-SLAM 优于纯 visual SLAM？**
- Visual SLAM [44-49]: 无 absolute scale，长 trajectory drift
- LiDAR SLAM [50-54]: metric accuracy 但无 RGB semantics
- LIV-SLAM [55-57]: 互补 — LiDAR 给 metric geometry，visual 给 appearance，IMU 给 high-rate motion，GNSS 给 global consistency

### 3.3 Image Processing

**Two-stage masking**:
1. **Egoblur** [73]: mask 人脸和车牌 (privacy)
2. **YOLOv11** [74]: mask 动态物体 (people, animals, vehicles)

**Fisheye → Pinhole undistortion**:
Crop to 120° FOV 然后 undistort。原因：低阶 distortion model 在大 FOV 下近似不准，且多数 3DGS models 在 pinhole images 上训练 [75, 76]。

### 3.4 3DGS Training

#### Initialization
从 dense colorized point cloud (5-10mm spacing, ~10-50M raw points) **downsample 到 ~5M points**，每个 point 一个 Gaussian。

**Opacity parameterization 的关键 insight**:
$$\alpha_i \propto \frac{1}{V_i}$$
其中 $\alpha_i$ 是第 $i$ 个 Gaussian 的初始 opacity，$V_i$ 是其体积 (通过 KNN heuristic 估计)。

**Intuition**: 体积大的 Gaussian (可能来自 transient obstacles 的 spurious floaters) 初始 opacity 低，避免在 training 早期 dominate rendering。这是一个 anti-floater trick。

#### Depth Regularization
不同于用 monocular depth 作为 pseudo GT，作者 **直接 project initialized Gaussians 到每个 camera pose** 作为 GT depth。

**为什么不 freeze Gaussian centers？** Table II 显示 frozen Gaussians 虽然保持 multi-view geometric consistency，但 visual quality 下降 (PSNR 20.39 vs 20.37，但 extrapolated SSIM 0.558 vs 0.591)。Frozen centers 施加过于 uniform 的 geometric prior，限制了 model 捕获 high-frequency details 的能力。

**Depth loss**:
$$\mathcal{L}_{depth} = \sum_{rays} \| D_{pred}(r) - D_{init}(r) \|^2$$
其中 $D_{pred}(r)$ 是渲染的 depth，$D_{init}(r)$ 是从 initialized Gaussians 渲染的 depth (作为 GT)。

#### Training View Augmentation — Difix3D+
使用 pretrained **Difix3D+** [77] 增强 extrapolated views:
1. 对 sampled extrapolated camera pose，从当前 3DGS rasterize 一张 image
2. 连同 nearest neighboring training views 作为 reference 喂给 Difix3D+
3. 获得 cleaned + geometrically accurate novel view
4. 加入 training set，用 lower loss weight

**Curriculum strategy**: 早期只 sample 训练轨迹附近的 novel views，随 training steps 渐进扩大 sampling radius。这稳定了 large-scale 3DGS training。

#### Hyperparameters (Table I)
- Training steps: 15,000
- SH degree: 3
- Perceptual loss weight: 0.2
- Depth loss weight: 0.02
- Means learning rate: $1.6 \times 10^{-5}$
- Scales learning rate: $1.0 \times 10^{-3}$
- Opacity learning rate: $2.0 \times 10^{-2}$

### 3.5 Mesh Extraction & Scene Integration

**Marching Cubes** [78] 从 voxelized point cloud 提取 mesh:
1. Filter 远离 trajectory 的 points
2. Voxelize → occupancy grid
3. Marching Cubes → triangle mesh
4. 移除 <50 faces 的 fragments

**Design philosophy**: Mesh 用于 collision checking 而非 visual detail，所以追求 occupancy-accurate 而非 visually detailed。不需要 watertight (3DGS 负责外观)。

**USD integration**: Mesh (physics layer) + 3DGS (renderer) 共享 MetaCam world coordinate frame (meters)，直接加载到 Isaac Sim。

### 3.6 Navigation Tasks

**Expert trajectories**:
- Unity NavMesh baking → triangulated navigable surface
- Pathfinding → collision-free shortest paths
- Start/goal 在 camera poses 附近采样

**VLN instructions** (two-stage prompting):
1. Gemini 2.5 Flash [84] 输出 structured JSON (segmented subinstructions with landmarks/actions)
2. Condense 成 single fluent instruction
3. Human spot-check

---

## 4. 实验结果深度解析

### 4.1 3D Reconstruction (Table 2)

评估的 vision-only methods:
- **DUSt3R** [79], **MUSt3R** [80]: pairwise/multi-view stereo
- **VGGT** [30]: Visual Geometry Grounded Transformer
- **π3** [81]: permutation-equivariant geometry learning
- **MapAnything** [82]: universal feed-forward metric reconstruction
- **DA3** [83]: Depth Anything 3
- **COLMAP** [36] (with/without GT intrinsic calibration)

**Key metrics definitions**:

**Translation ATE - Raw** (T-ATE^R): SE(3) alignment 后的 translation RMSE
$$\text{T-ATE}^R = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\|(R_{SE3} \cdot \mathbf{t}_i^{pred} + \mathbf{t}_{SE3}) - \mathbf{t}_i^{gt}\|^2}$$
- $R_{SE3}, \mathbf{t}_{SE3}$: SE(3) alignment 的 rotation 和 translation
- $\mathbf{t}_i^{pred}, \mathbf{t}_i^{gt}$: predicted 和 ground truth 的 camera translation

**Translation ATE - Scaled** (T-ATE^S): SIM(3) alignment 后的 translation RMSE
$$\text{T-ATE}^S = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\|s \cdot R_{SIM3}\mathbf{t}_i^{pred} + \mathbf{t}_{SIM3} - \mathbf{t}_i^{gt}\|^2}$$
- $s$: scale factor from SIM(3) alignment

**Rotation ATE** (R-ATE):
$$\text{R-ATE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\Delta\theta_i^2}, \quad \Delta\theta_i = \cos^{-1}\left(\frac{\text{tr}(\mathbf{R}_i^{gt\top}\mathbf{R}_i^{pred})-1}{2}\right)$$

**AUC@30**:
$$\text{AUC@30} = \int_0^{30} P(\max(\text{R-RTE}, \text{T-RTE}_{deg}) < \theta) d\theta$$

**关键结论**: 即使 "Best of All" vision-only methods，T-ATE^S = 0.30m, R-ATE = 5°。在 <100m 场景中，这是 navigation 不可接受的误差。

### 4.2 Novel View Synthesis (Table 3)

| Method | Interp PSNR | Extrap PSNR | Extrap LPIPS |
|--------|------------|------------|--------------|
| 3DGS (COLMAP) | 18.27 | 16.90 | 0.559 |
| Vid2Sim | 17.20 | 16.49 | 0.371 |
| GaussGym | 12.17 | 12.63 | 0.725 |
| **Ours** | **20.37** | **17.92** | **0.445** |

**Semantic consistency** (Figure 6): GaussGym 的 fragmented rendering 导致 Grounded SAM 2 [86] 检测失败；Vid2Sim 的 DINOv3 [85] features 与 GT 严重偏离 (PCA colors 不同)。这对依赖 DINO features 的 end-to-end navigation policies [21] 是致命的。

### 4.3 RL Training (Table 4)

在 Vid2Sim env 中 RL post-training:
- NoMaD [90]: SR 0% change, SPL -4%
- CityWalker [21]: SR -21%, SPL -19%
- MBRA [91]: SR 0%, SPL -5%

在 WANDERLAND env 中:
- NoMaD: SR +8%, IR -13%
- CityWalker: SR +14%, SPL +14%, IR -23%
- MBRA: SR +5%, IR -7%

**Intuition**: 不准确的 geometry 让 RL 鼓励 locally shorter 但 globally unreliable behaviors — 因为 agent 可以 "穿过" 不存在的 gaps (mesh fragmentation) 或被 phantom obstacles 阻挡。

### 4.4 Navigation Benchmark (Table 5)

| Method | Indoor SR | Outdoor SR |
|--------|----------|------------|
| NoMaD | 0.22 | 0.24 |
| CityWalker | 0.39 | 0.21 |
| NaVILA [71] (VLN) | 0.47 | 0.31 |

**观察**:
1. VLN (NaVILA) > point-goal/image-goal (受 LLMs 帮助)
2. Outdoor 比 Indoor 更难 (longer trajectories, complex topology, elevation changes)
3. 没有 model SR > 50% — open-world navigation 仍有巨大 research gap

---

## 5. Ablation Studies (Table III)

3DGS components 的影响:
- Diverse views alone: PSNR 15.26 (interp), 12.99 (extrap)
- + GT camera pose: 18.27 / 16.90
- + LiDAR init: 15.54 / 15.19 (pose + init 但无 depth loss 反而更差!)
- + Depth loss: 20.95 / 16.97
- + View augmentation: 20.37 / **17.92** (extrap 最佳!)

**关键 insight**: GT pose 和 dense LiDAR init 只是好的起点，**必须配合 depth loss regularization** 才能转化为好的 NVS quality。View augmentation 对 extrapolated views 至关重要。

---

## 6. Limitations & Broader Impact

**Limitations**:
1. 1 FPS capture rate → sparse viewpoints → 影响 complex scenes 渲染
2. 只处理 static environments (无 dynamic pedestrians/vehicles)

**Broader Impact**:
- Metric camera poses + dense LiDAR = vision geometry models 的 GT
- Extrapolated views = NVS benchmark under realistic off-trajectory conditions
- Outdoor 3D vision 的大规模 metric benchmark 稀缺问题

---

## 7. 我的 Intuition 总结

Andrej, 这篇 paper 的核心 message 对你的工作很 relevant：**embodied AI 的 sim-to-real gap 不只是 visual fidelity 问题，更是 geometric fidelity 问题**。

为什么 vision-only 不够？因为：
- SfM/depth prediction 本质上是 ill-posed inverse problem
- Metric scale 在 monocular 设定下不可恢复
- 即使 foundation models (VGGT, DA3) 也只能达到 0.3m/5° 误差

为什么 LiDAR+Visual 是 sweet spot？
- LiDAR: metric accuracy, robustness to texture-less regions
- Visual: appearance, semantics
- IMU: high-rate motion interpolation
- GNSS: global consistency

这让我想到你们在 Tesla 做的——纯 vision 的极限 vs multi-sensor fusion 的 trade-off。这篇 paper 用实验数据论证：**对于 closed-loop embodied AI benchmarking，geometric grounding 不是 optional 的**。

**Related links**:
- Project: https://ai4ce.github.io/wanderland/
- Vid2Sim (CVPR 2025): https://www.cs.columbia.edu/~bzhao/vid2sim/
- GaussGym: https://arxiv.org/abs/2510.15352
- 3DGS (Kerbl et al.): https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- VGGT: https://vgg-t.github.io/
- Difix3D+: https://github.com/NVlabs/DIFIX3D
- Isaac Sim: https://developer.nvidia.com/isaac-sim
- gsplat: https://github.com/nerfstudio-project/gsplat
- CityWalker (作者团队前作): https://ai4ce.github.io/citywalker/
- NYU AI4CE Lab (Chen Feng): https://ai4ce.github.io/
