---
source_pdf: VINGS-Mono Visual-Inertial Gaussian Splatting Monocular SLAM in Large
  Scenes.pdf
paper_sha256: 2a78eb7d9ff34167aa453fa34d7be6e0305010d3451f9e0d02c04283ed3d6355
processed_at: '2026-08-13T01:18:58-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# VINGS-Mono 人话版

我换一种讲法，假设你刚听到这篇 paper 的名字，想 5 分钟搞清楚它到底干了啥。

---

## 一句话总结

**用一部手机的单目相机 + IMU，边走边建一张公里级、千万高斯点的高斯泼溅地图，还能跑闭环、躲动态物体，最后在手机上实时显示。**

听起来不刺激，但你想想：之前所有 GS-SLAM 方法要么得有 depth camera，要么只能在十几米的房间里玩，要么得背个 LiDAR。这哥们说"我把这些都干掉了，而且能跑 8 公里"，这是一个非常硬的工程里程碑。

---

## 为什么难？

你要在大街上用单目相机做 GS-SLAM，会立刻撞上 5 个坑：

**坑 1：单目没尺度**
单目相机天生不知道"这堵墙离我 3 米还是 30 米"，每段轨迹的 scale 都会漂。室内还好，跑 5 公里就彻底飞了。

**坑 2：前向运动退化**
开车时相机基本沿光轴前进， triangulation 条件很差，feature 匹配的 baseline 也不够，传统 ORB-SLAM 直接跟丢。

**坑 3：高斯点爆炸**
公里级场景至少需要几千万个 Gaussian ellipsoid，单张 4090 24GB 显存根本塞不下，光存这些 Gaussian 的参数就够呛。

**坑 4：累积误差必须闭环**
跑几公里累积个几十米误差很正常。但闭环这事在 GS map 上极难 —— 传统 SLAM 改一下 sparse landmark 就完事，GS map 是几千万个点，闭环之后是不是要 retrain 全图？那要跑几天。

**坑 5：街上有车有人**
车和人是动的，GS 的 static 假设一坏，地图全是 floaters，结果就是 MonoGS 在 SmallCity 上直接给你画一片黑。

VINGS-Mono 就是要把这 5 个坑一个个填上。它的填坑思路如下。

---

## 模块 1：前端 VIO（解决坑 1 + 坑 2）

不用 ORB 这种稀疏特征点，用 **DROID-SLAM 那套 Dense Bundle Adjustment**：对每对相邻帧用 RAFT 算 dense optical flow，然后用 GRU 反复迭代，最后同时优化每像素的 inverse depth 和所有关键帧的 pose。

这么做的好处是**前向运动也能撑住**：稀疏点在前向运动下 baseline 不够，dense flow 不一样，每个像素都贡献约束，退化姿态被拉回来。

公式 (1) 给的就是这个意思：把第 $i$ 帧的像素 $\mathbf{u}_i$ 加上 inverse depth $\mathbf{d}_i^{-1}$ 反投影到 3D，用相对 pose $\mathbf{T}_{ij}$ 变到 frame $j$ 再投影，应该等于 GRU 算出的光流 $\mathbf{u}_{ij}$。整个能量函数 (2) 就是让这两个东西尽量对上。

然后做一件很聪明的事：**用 Schur Complement 把 depth 消掉，只留 pose 约束塞进 GTSAM factor graph**（公式 4）。

$$(\mathbf{B}_i - \mathbf{E}_i\mathbf{C}_i^{-1}\mathbf{E}_i^T)\Delta\xi = \mathbf{v}_i - \mathbf{E}_i\mathbf{C}_i^{-1}\mathbf{z}_i$$

- $\mathbf{B}_i$：pose-pose 块
- $\mathbf{E}_i$：pose-depth 耦合块
- $\mathbf{C}_i$：depth-depth 对角块
- $\Delta\xi$：pose 增量

把百万像素的 depth 信息压成一个 compact 的位姿约束，然后和 IMU 预积分（公式 8 那套 VINS-Mono 标准）一起进 GTSAM 优化。这就是为什么它能跑 100Hz KITTI-360 unsync 数据 —— factor graph 维度被压住了。

还有一个被忽视的小创新：**从 DBA 的 Hessian 直接算 depth uncertainty**（公式 9）：

$$\Sigma_{d^{-1}} = \mathbf{C}^{-1} + \mathbf{C}^{-1}\mathbf{E}^T\Sigma_T\mathbf{E}\mathbf{C}^{-1}$$

- $\Sigma_T$：位姿 marginal covariance
- $\Sigma_{d^{-1}}$：每像素逆深度 covariance

这个 $\Sigma_{d^{-1}}$ 后面会被 Dynamic Eraser 拿来用，是 paper 里一个串联前后端的小细节。

---

## 模块 2：2D Gaussian Map（解决坑 3 + 一部分坑 1）

它选 **2DGS 而不是 3DGS**。这是个明智选择：2DGS 把每个 Gaussian 当 surface element，法向明确，几何上贴 surface 而不是体积，normal loss 才有意义，对 SLAM 这种重视几何一致性的场景天然友好。

mapping 的核心策略是**先大量加，再剪枝**，放弃 3DGS 原版的 clone-split + reset opacity。原因很直接：在 driving 这种 forward-view 场景下，reset opacity 行为不稳定，反复触发 densify 但几何上没必要。

每来一帧新 keyframe：渲染当前帧 → 删掉 frustum 内 RGB/depth loss 高的、投影半径过大的 Gaussian → 在 low accumulation 区域按比例加点 → 训练。Loss 用 2DGS 那套：

$$\mathcal{L} = \lambda_{rgb}\mathcal{L}_{rgb} + \lambda_d\mathcal{L}_d + \lambda_n\mathcal{L}_n + \lambda_{acc}\mathcal{L}_{acc}$$

权重 $\lambda_{rgb}=1.0, \lambda_d=0.5, \lambda_n=0.1, \lambda_{acc}=0.1$。

但真正让这套 scale 到 51.7M Gaussians 的是 **Score Manager**。

### Score Manager —— 这是这篇 paper 最核心的工程贡献

每个 Gaussian 维护三件东西：

$$S_C(g) = \sum_{t=0}^{K}\sum_{u=0}^{P} f_i\prod_{j=1}^{i-1}(1-f_j) \quad \text{(contribution, 求和)}$$
$$S_E(g) = \max\Big(\big\{\sum_{u}\mathcal{L}_{rgb}(u)f_i\prod_{j=1}^{i-1}(1-f_j)\big\}\Big) \quad \text{(error, 取最大)}$$
$$ID(g) = \arg\max_t(\text{per-frame contribution})$$

- $S_C$：contribution score，**用 sum 累积**
- $S_E$：error score，**用 max 而不是 sum**
- $ID(g)$：贡献最大的那一帧 ID

为什么 $S_C$ 用 sum、$S_E$ 用 max？作者自己解释：contribution 大的 Gaussian 一定要留（不管是单帧贡献大还是多帧累积贡献大都该留），所以用 sum；error 用 sum 会冤枉那些"被多帧看到的普通 Gaussian"，用 max 只惩罚真正在某帧造成大误差的 Gaussian。

这种 **asymmetric aggregation** 是个非常有意思的工程直觉，跟 attention 里 max-pooling 抑噪 vs sum-pooling 保留长尾是一个道理。

然后三个机制：

**Status Control**：Gaussian 在 stable / unstable 两态间切换
- unstable 且 $S_C < 10^{-4}$ → stable（冻结出优化池）
- stable 且 $S_E > 0.5$ → unstable 并 reset 分数（重激活）

stable 状态保护历史 Gaussian：转弯或跨房间时历史 Gaussian 会重新进 frustum，按距离 prune 会灾难性丢失，stable 标记保留它们。同时 sparse Adam 优化时 mask 掉 stable Gaussians，加速训练。

**Storage Control**：unstable 且 $S_C < 0.5$ → 直接 prune

**GPU-CPU Transfer**：每 8 个 keyframe，按 $ID(g)$ 算 Gaussian 关联的 pose 距当前 pose 的距离，距离 $< \tau$ 调入 GPU，$> \tau$ 调回 CPU。

注意：**用 $ID(g)$ 算 pose 距离而不是 Gaussian 中心距离**。算百万 Gaussian 中心距离本身就贵，用 $ID(g)$ 直接查 pose 距离就 O(1) 一查完事。

Ablation 数据很漂亮：ScanNet-0106 上 Gaussian 数从 4.04M 降到 1.96M，PSNR 反而从 22.98 升到 23.02。剪枝剪到 PSNR 升高 —— 这说明原 map 里有一堆 Gaussian 在贡献噪声，剪掉反而干净。

### Sample Rasterizer —— backward 加速 273%

原版 2DGS 的 backward：每个 GPU block 负责 16×16=256 个 pixel，每个 thread 处理 1 个 pixel，迭代数 = 该 pixel 关联的 Gaussian 数。瓶颈 = 整个 tile 中"最重 pixel"的迭代数。某些 pixel 有几百个 Gaussian 覆盖，其它 thread 都得等它。

VINGS-Mono 的改动：forward 时每 32 个 Gaussian 存一次中间状态（warp 对齐），backward 时把 GPU 划成 warp，每 warp 处理一组 Gaussian；每个 tile 内挑 loss 比例最高的 $r=0.5$ 部分 pixel 做反向，每 thread 迭代数降到 $256 \times 0.5 = 128$。

backward 时间 11.55ms → 4.23ms，**加速 273%**，PSNR 只降 0.56 dB。

直觉：高 loss pixel 主导梯度，按 loss 做重要性采样等价于 importance sampling on gradient。这跟 NeRF 早期的 pixel error sampling 一脉相承，只是 3DGS 之前没人这么改 rasterizer。

### Single-to-Multi Pose Refinement

公式 (14) 是关键：

$$\hat\mu_k = \mathbf{T}_{c_k}^w \mathbf{T}_{c_k}^{\hat c_k} \mathbf{T}_w^{c_k}\mu_k$$
$$\min_{\{\mathbf{T}_{c_k}^{\hat c_k}\}} \mathcal{L}_{rgb}(\hat I_k, I_k)$$

- $\mathbf{T}_{c_k}^{\hat c_k}$：第 $k$ 帧位姿的扰动（优化变量）
- $\mu_k$：与该帧关联的 Gaussian 中心（由 $ID(g)$ 决定）
- $\hat I_k$：渲染图像

因为 score manager 已经给每个 Gaussian 绑了 $ID(g)$，渲染单帧时 gradient 流到 Gaussian 位置后再传到 **该 Gaussian 所属 keyframe 的 pose**，一次渲染同时优化视野内所有 keyframe 的 pose。

实验：ScanNet-0106 ATE
- 不做 pose refine：0.25m
- 只优化当前帧（SplaTAM / MonoGS 做法）：0.19m
- 优化可见帧（本文）：0.16m

直觉：当前帧的 Gaussians 大多是从其它 keyframe 来的，这些 Gaussians 的位置依赖历史 keyframe 的 pose。把 gradient 同时回传到那些 keyframe，等于用一个 dense photometric error 同时约束多个 pose，信息量比单帧高得多。

---

## 模块 3：NVS Loop Closure（解决坑 4）

这是 paper 最有意思的一个 idea。

### Loop Detection 三步

1. **特征匹配**：和距离阈值内、frame ID 差 > 10 的历史帧做 LightGlue 匹配，匹配点 > 50 的帧按数量降序排
2. **Render Depth + Solve PnP**：渲染当前帧 depth $\hat D_{t_n}$，用匹配点 + depth 做 PnP 解相对 pose $\hat T_{t_n}^w = T_{t_m}^w T_{t_n}^{t_m}$。**只用 depth 低于阈值的点**（远处深度不稳）
3. **Novel View Synthesis 验证**：用 $\hat T_{t_n}^w$ 在 Gaussian map 上渲染，和真实 $I_{t_n}$ 算 L1 loss。loss 低于阈值 **或** 低于其它候选帧中位 loss 的 1/10 → loop detected

这个第 3 步是核心 idea：**"两张图是不是同一场景"  →  "这张新图能不能在 GS map 里被当作合法 novel view 渲染出来"**。GS 天生就是 NVS 引擎，直接用渲染 loss 当 loop 验证，省掉 BoW vocabulary 维护，也不用维护 covisibility matrix。

### Loop Correction：避免 retrain 全图

这是大场景必须解决的工程问题。GO-SLAM 那种 incremental global retraining 在 8km 上根本跑不动。

VINGS-Mono 的做法：

1. **Pair Gaussian with Pose**：所有历史 keyframe forward 一遍，每个 Gaussian 选 $S_C$ 最大的 pose 作为 matched pose。1000 帧约 2 秒。

2. **Pose Graph Optimization + Gaussian 刚体变换**：

公式 (15)：

$$\mu_i' = T_{c_k}^{w\prime} T_w^{c_k}\mu_i$$
$$r_i' = R^{-1}(T_{c_k}^{w\prime} T_w^{c_k} R(r_i))$$

- $k = ID(g_i)$：该 Gaussian 所属 keyframe
- $T_{c_k}^{w\prime}$：pose graph 优化后的新位姿
- $\mu_i, r_i$：原 Gaussian 中心与旋转
- $\mu_i', r_i'$：校正后

3. **再 train 100 iter fine-tune + 按 $S_C$ prune**

整个 loop correction 几秒级，相对 GO-SLAM 那种 global retraining 快了几个数量级。

为什么 work？因为每个 Gaussian 已经和 keyframe pose 绑定（通过 $ID(g)$），pose graph 优化后相当于给每个 Gaussian 做了一次刚体变换（scale + rotation），几何结构基本保留。然后 100 iter 微调修补局部。这种 "刚性变换 + 微调" 模式比 retrain 全图高效得多。

---

## 模块 4：Dynamic Object Eraser（解决坑 5）

街上必然有车有人。4D Gaussian / dynamic NeRF 那套需要 offline 训练时序模型，SLAM 这种 incremental 场景用不了。

VINGS-Mono 的 heuristic：

1. Fast-SAM 给所有可能动态的语义 mask $\{M_k\}$
2. 渲染当前帧 color，和真实帧算 **re-rendering loss**：

$$\mathcal{L}_{re} = \mathcal{L}_{SSIM} \cdot \mathcal{L}_{L1} \cdot \Sigma_{d^{-1}}$$

- $\mathcal{L}_{SSIM}$：纹理敏感
- $\mathcal{L}_{L1}$：颜色敏感
- $\Sigma_{d^{-1}}$：前端 DBA 给的 depth uncertainty

这里乘 $\Sigma_{d^{-1}}$ 很聪明：动态物体边缘 depth 估不准，$\Sigma_{d^{-1}}$ 大，re-rendering loss 被放大 → 边缘更容易被判为动态。

3. 公式 (16)：

$$M_{dyn,k} = \Big(\frac{\text{高 loss 像素数}}{\text{mask 像素数}} > 20\%\Big) \wedge \Big(\overline{\mathcal{L}_{re}}(M_k) > \mathcal{L}_{re}^{th}\Big)$$

两个条件 AND 才判动态。停车场里的静止车辆 re-rendering loss 低，不被误删；街上跑的车 re-rendering loss 高 + 边缘 depth 不稳，被删。

BONN 数据集 ATE：从 30.25cm（无 eraser）降到 4.34cm（有 eraser），击败 ReFusion 27.65cm 和 RodynSLAM 12.2cm。

---

## 实验数据快速过一遍

### Localization

**室内 ScanNet + BundleFusion（ATE cm）**：VINGS-Mono 15-92cm，普遍 SOTA。MonoGS 60-190cm，PhotoSLAM 150-360cm，ORB-SLAM3 25-243cm。

**户外 Waymo + Hierarchical（ATE m）**：VINGS-Mono SmallCity 2.82m、Campus 1.03m、Waymo 0.91-2.67m。MonoGS 和 PhotoSLAM 在 SmallCity 直接 fail（一片黑 / 跟丢回原点），只能跑前 50 帧。

**VIO KITTI / KITTI-360**：低频 KITTI 02 上 VINGS 2.64% / 0.44° vs VINS 2.08% / 1.68°，iSLAM 2.08% / 0.53°。高频 KITTI-360 unsync 上 VINGS 几乎全面领先，02 序列仅 0.58% vs iSLAM 38.46%（iSLAM 在 unsync 上崩了）。

### Rendering

室内 ScanNet 平均 PSNR 22.43dB / SSIM 0.79 / LPIPS 0.22。MonoGS 17-21dB。

户外 KITTI-360 08：VINGS 24.52dB vs MonoGS 16.08dB vs PhotoSLAM 15.81dB。**差 8-9 dB 是巨大差距**，主要来自 score manager 控制 floaters + depth uncertainty 抑制天空。

MegaNeRF Building / Rubble：VINGS 25.45 / 25.21 dB，大幅领先 GO-SLAM 20.71 / 20.81 dB。

### Runtime

| Dataset | Frames | Total | Tracking/frame | FPS | Model |
|---|---|---|---|---|---|
| Waymo-01 | 198 | 117s | 214ms | 1.69 | 386MB |
| SmallCity | 877 | 739s | 247ms | 1.18 | 1817MB |
| KITTI-08 | 5177 | 4560s | 273ms | 1.13 | 10366MB |

Tracking 4-5 FPS，Mapping 慢但并行跑。10GB 模型存 8km KITTI 51.73M Gaussians，单张 4090 装下。

### Mobile App

Flutter app 跨平台，手机收 480×720 @30Hz + IMU 传服务器跑 VINGS-Mono，输出 BEV Gaussian map / rendered color / depth / normal。自采校园 1.02km × 0.4km 数据，自行车 10km/h 采集，20Hz RGB + 1Hz GPS 验证，单目几乎无 scale drift。

---

## 我的吐槽和思考

### 1. 它真正的贡献不是"单模块 SOTA"，而是"系统性工程"

VINGS-Mono 没有哪一个 trick 单独拿出来是革命性的：DBA 来自 DROID-SLAM，IMU 融合来自 VINS-Mono，2DGS 来自 SIGGRAPH 2024，Sample Rasterizer 来自 Taming 3DGS，LightGlue 来自 CVG，Fast-SAM 来自 Meta，pose graph 优化来自 GTSAM。

它的贡献是**把 GS-SLAM 系统里每一处可能卡住的地方都做了具体工程优化，然后让它们协同**。Score Manager 那个 $S_C$ 用 sum / $S_E$ 用 max 的细节，stable/unstable 二态保护历史 Gaussian 的设计，$ID(g)$ 用于 pose-Gaussian 绑定做 loop correction —— 这些都是踩坑之后悟出来的工程直觉，光看公式体会不到，得跑实验才知道为什么必须这么设计。

### 2. Loop Closure 用 NVS 是个非常漂亮的 framing

传统 visual loop closure 是 "image A 和 image B 长得像不像"。VINGS-Mono 把它重新定义为 "image B 能不能在 GS map 上被当作 image A 视角的合法 novel view"。

这是一个 representation-level 的认知转换：GS 不只是 dense map，它本身承担了 place recognizer 的角色。如果未来把 L1 loss 换成 DINOv2 feature loss 或者 LPIPS，对光照 / 季节变化的鲁棒性应该能再上一个台阶。最近 SALAD (https://github.com/SaladVision/SALAD) 已经证明 self-supervised feature 比 BoW 强很多，这条线值得继续做。

### 3. 2DGS vs 3DGS 这个选择被低估了

2DGS 把每个 Gaussian 当 surface element，天然 normal-aware。这让 depth uncertainty、normal loss、loop correction 里的刚体变换都有了 surface 几何意义。3DGS 是 volumetric 表示，没有明确的 normal 概念，在 SLAM 里反而吃亏。我猜后续 GS-SLAM 工作会越来越多用 2DGS 或类似的 surface-aligned 表示。

### 4. Limitations 说得很诚实

Paper 自己承认：
- **极高速度运动**：DBA 在大 frame interval 下恢复几何困难，GS 多次 iter 限制重建速度
- **未 on-device**：仍依赖 server

未来方向提到要给 DBA 加 prior（比如 diffusion-based depth: Marigold, GeoWizard）、用 Point Transformer V3 / Large Spatial Model 直接输出 Gaussian 属性减少 iter 数、部署到 edge device。

### 5. 我额外想到的几个方向

- **Depth uncertainty 的进一步使用**：现在只在 mask 和 dynamic eraser 用了 $\Sigma_{d^{-1}}$。其实可以拿它做 per-Gaussian anisotropic regularization，比如对高不确定区生成的 Gaussian 加更强 prior pull 到 frontier surface。NeRF-SLAM (Rosinol et al. 2023) 已经做过这种 uncertainty-aware mapping，2DGS-SLAM 还没充分利用。

- **Score Manager 与 recently active "Gaussian lifecycle" 文献的连接**：Compact-3DGS (Lee et al. 2024)、Self-organizing Gaussians (Chen et al. 2024)、EAGLES、Mip-Splatting 都是同一波思路 —— 把 GS 当资源池而非稠密表达。Score-based 管理可以和这些 anchor-based 或 SH-pruning 方法叠加。

- **Pose Refinement 的理论性质**：把 Gaussians 按 $ID(g)$ 绑到 keyframe 后，pose 优化变成 "每个 Gaussian 是其 anchor pose 的相对量"。这跟 neural field 里 canonical frame + per-frame deformation 思路接近（Nerfies / HyperNeRF / 4DGS）。理论上可以推出这套优化的 Jacobian 结构、可观测性条件，做 SLAM 风格的 observability analysis。

- **Mobile 端压缩**：10366 MB for 8km KITTI 太大。结合 SOG (Sort-Free Gaussian Splatting, arXiv:2410.18931) 替代 radix sort、压缩 SH 到 RGB，应该能压到 1-2GB。

- **Meta 观察**：VINGS-Mono 的设计哲学是 "前端用学习的稠密方法获取几何先验 + IMU 因子；后端把 GS 既当 map 又当 place recognizer"。这把传统 SLAM 里 frontend (VO/VIO) - backend (loop closure / pose graph) - mapper (dense reconstruction) 三件套重新融合：**GS 不只是被动的 dense representation，它本身承担了 place recognition、loop correction、dynamic detection 三重身份**。这跟 recent "Foundation Model as SLAM" (DPVO, DROID-SLAM, iSLAM, Large Spatial Model) 是平行发展的两个流派：VINGS 偏 explicit-Gaussian，LSM 偏 implicit-neural。我倾向于 explicit 这条线在工程上更容易 scale，因为可解释、可裁剪、可 cache 到 disk，而 neural field 的 inductive bias 难以控制。

---

## 一句话再总结一次

VINGS-Mono 不是某一个 single trick 的胜利，而是**把 GS 在 SLAM 系统里每一处可能卡住的地方（init、densify、pruning、memory、loop、dynamic）都做了具体工程优化，然后让它们协同**。读 paper 时建议重点看 Algorithm 1（Score Manager）、公式 13/14/15 这三处，是整套设计的压舱石。

参考：
- 项目主页：https://vings-mono.github.io
- 2DGS: https://2d-gaussian-splatting.github.io/
- DROID-SLAM: https://github.com/princeton-vl/DROID-SLAM
- Taming 3DGS: https://taming3dgs.github.io/
- SOG: https://arxiv.org/abs/2410.18931
- Large Spatial Model: https://arxiv.org/abs/2410.18956
- DINOv2: https://dinov2.metademolab.com/
- SALAD: https://github.com/SaladVision/SALAD
- NeRF-SLAM: https://github.com/MIT-SPARK-Kimera/NeRF-SLAM

---

# VINGS-Mono: Visual-Inertial Gaussian Splatting Monocular SLAM in Large Scenes — 一份深度讲解

下面按 paper 的逻辑链梳理，并夹带我自己读出的直觉与联想。公式我会把每个符号拆开讲清楚，方便你 build intuition。

参考资源：
- Project page: https://vings-mono.github.io
- 2DGS (基础表示): https://2d-gaussian-splatting.github.io/
- 3DGS: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- DROID-SLAM (DBA 来源): https://github.com/princeton-vl/DROID-SLAM
- RAFT (correlation volume): https://github.com/princeton-vl/RAFT
- GTSAM (factor graph): https://gtsam.org/
- VINS-Mono (IMU 预积分): https://github.com/HKUST-Aerial-Robotics/VINS-Mono
- LightGlue (特征匹配): https://github.com/cvg/LightGlue
- Metric3D (depth prior): https://github.com/YvanYin/Metric3D
- Fast-SAM (语义分割): https://github.com/CASIA-IVA-Lab/FastSAM
- evo (评测): https://github.com/MichaelGrupp/evo

---

## 1. 论文核心动机：把 GS-SLAM 从"室内 + RGB-D / LiDAR"推到"单目 + IMU + 公里级户外"

现有 Gaussian Splatting SLAM 的痛点：
- SplaTAM / MonoGS / PhotoSLAM / Gaussian-SLAM 基本只在 Replica、TUM-RGBD、ScanNet 这类小室内场景里玩，且大多依赖 depth camera 初始化。
- LIVGaussMap / MMGaussian 虽然能到几百米，但需要 LiDAR，consumer 不友好。
- 单目 GS-SLAM (MonoGS、PhotoSLAM、MGS) 用 ORB 特征点或 random 初始化 Gaussians，在 driving 这种 forward-view + 快速运动下极易漂移、产生 floaters。
- 大场景累积误差要靠 loop closure，但传统 BoW 在公里级会膨胀，GO-SLAM 维护 co-visibility matrix 是 O(N²) 存储，稠密 GS map 的 loop correction 普遍需要 retrain 全部历史帧，根本不可行。

VINGS-Mono 的 claim 是：第一个能在公里级 driving / aerial / 室内 通吃，仅用单目相机 + 可选低频 IMU 的 GS-SLAM。KITTI-360 上跑 8.05 km，地图含 51.73 million Gaussians，单张 RTX 4090 就能跑 online。这是个非常硬的工程目标。

---

## 2. System Overview：四模块流水线

整条 pipeline 我画成脑子里这个图：

```
RGB + IMU (optional)
    │
    ▼
┌────────────────────────────────────────────────────────┐
│ VIO Front End                                          │
│   • RAFT correlation volume → GRU → optical flow        │
│   • Dense Bundle Adjustment (DBA) 解 inverse depth + T  │
│   • Schur Complement 把 depth 消掉 → 姿态约束           │
│   • GTSAM factor graph 融合 IMU 预积分                  │
│   • 从 Hessian marginal covariance 算 depth uncertainty │
└────────────────────────────────────────────────────────┘
    │ {T_t, D_t, U_t, I_t}
    ▼
┌────────────────────────────────────────────────────────┐
│ 2D Gaussian Map (在线增量)                             │
│   • Online Mapping (add-then-prune, 而非 clone-split)  │
│   • Score Manager: S_C / S_E / ID(g), 三态管理 + CPU/GPU 交换 │
│   • Sample Rasterizer: pixel-parallel → Gaussian-parallel │
│   • Single-to-Multi Pose Refinement: 一帧 loss 优化多帧 │
└────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────┐
│ NVS Loop Closure                                       │
│   • 特征匹配 (LightGlue) → PnP 用 rendered depth      │
│   • Novel view synthesis 当作"看是不是同一个场景"的验证 │
│   • Pose graph optimization + Gaussian-pose pairing 一次性校正 │
└────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────┐
│ Dynamic Object Eraser                                  │
│   • Fast-SAM 出语义 mask                               │
│   • Re-rendering Loss = SSIM × L1 × Σ_d^{-1}           │
│   • heuristic: 高 loss 占比 + 平均 loss 阈值 → 动态 mask │
└────────────────────────────────────────────────────────┘
```

核心 intuition：**前端给"姿态 + 带不确定度的稠密深度"作为监督信号，让 2DGS 在线重建；后端用 GS 自己的 NVS 能力做 loop detection，再用 pose-Gaussian 绑定关系做 loop correction 而不必 retrain**。整套设计把"GS SLAM 大场景扩展性"这件事从多个角度同时压下来。

---

## 3. Visual-Inertial Front End

### 3.1 Dense BA & Vision Factor (来自 DROID-SLAM 思路)

对相邻 RGB 帧，先用 RAFT 的 correlation volume 编码 + GRU 反复迭代得到 optical flow 残差 $\mathbf{r}_{ij}$ 与对应权重 $\mathbf{w}_{ij}$。下游做 1/8 分辨率（实时性需要）。

公式 (1)：把逆深度 $\mathbf{d}_i^{-1}$ 通过反投影 $\Pi_C^{-1}$ 拿到 3D 点，再用相对姿态 $\mathbf{T}_{ij}$ 变到 frame $j$，再投影回去，得到预测光流 $\mathbf{u}_{ij}$：

$$\mathbf{u}_{ij} = \Pi_C(\mathbf{T}_{ij} \circ \Pi_C^{-1}(\mathbf{u}_i, \mathbf{d}_i^{-1}))$$

- $\Pi_C$：相机投影函数
- $\Pi_C^{-1}$：反投影（给像素坐标 + 逆深度 → 3D 射线上的点）
- $\mathbf{T}_{ij}$：frame $i$ 到 frame $j$ 的位姿
- $\mathbf{d}_i^{-1}$：每像素逆深度图（1/8 分辨率）
- $\mathbf{u}_i$：像素坐标

GRU 输出修正流场后，定义 corrected correspondence $\mathbf{u}_{ij}^* = \mathbf{r}_{ij} + \mathbf{u}_{ij}$。

公式 (2) 是 DBA 的能量函数：

$$E(\mathbf{T}, \mathbf{d}^{-1}) = \sum_{(i,j)\in\epsilon} \|\mathbf{u}_{ij}^* - \Pi_C(\mathbf{T}_{ij}\circ\Pi_C^{-1}(\mathbf{u}_i, \mathbf{d}_i^{-1}))\|^2_{\Sigma_{ij}}$$

- $\Sigma_{ij}$：由 $\mathbf{w}_{ij}$ 构成的对角权重矩阵
- $\epsilon$：图里所有 active 边

用 Gauss-Newton 同时优化 $\mathbf{T}_i, \mathbf{T}_j, \mathbf{d}^{-1}$。

公式 (3) 是把 frame $i$ 锚定、投影到 N 个共视 frame 的 stacked Hessian：

$$\begin{bmatrix} \Sigma\mathbf{v}_{ii} \\ \mathbf{v}_{i1} \\ \vdots \\ \mathbf{v}_{iN} \\ \Sigma\mathbf{z}_{ii} \end{bmatrix} = \begin{bmatrix} \Sigma\mathbf{B}_{ii} & \mathbf{B}_{i1} & \cdots & \mathbf{B}_{iN} & \Sigma\mathbf{E}_{ii} \\ \mathbf{B}_{i1}^\top & \mathbf{B}_{11} & & & \mathbf{E}_{i1} \\ \vdots & & \ddots & & \vdots \\ \mathbf{B}_{iN}^\top & & & \mathbf{B}_{NN} & \mathbf{E}_{iN} \\ \Sigma\mathbf{E}_{ii}^\top & \mathbf{E}_{i1}^\top & \cdots & \mathbf{E}_{iN}^T & \Sigma\mathbf{C}_{ii} \end{bmatrix} \begin{bmatrix} \Delta\xi_i \\ \Delta\xi_1 \\ \vdots \\ \Delta\xi_N \\ \Delta\mathbf{d}_i^{-1} \end{bmatrix}$$

- $\Delta\xi_i$：第 $i$ 帧位姿在 Lie algebra 上的增量
- $\mathbf{B}_{ij}$：位姿对位姿的 Hessian block
- $\mathbf{E}_{ij}$：位姿对逆深度的 Hessian block
- $\mathbf{C}_{ii}$：逆深度对逆深度的对角 block（每像素独立，所以是 diag）
- $\mathbf{v}, \mathbf{z}$：对应的 gradient vector

公式 (4) Schur Complement 消掉深度，得到帧间 pose 约束：

$$(\mathbf{B}_i - \mathbf{E}_i\mathbf{C}_i^{-1}\mathbf{E}_i^T)\Delta\xi_{i,1,\dots,N} = \mathbf{v}_i - \mathbf{E}_i\mathbf{C}_i^{-1}\mathbf{z}_i$$

这一步把稠密像素几何信息压成一个 compact 的姿态约束，可以塞进后续 factor graph 里。GPU 上高度并行。

公式 (5)：pose 更新完后回写深度：

$$\Delta(\mathbf{d}_i^{-1}) = \mathbf{C}_i^{-1}(\mathbf{z}_i - \mathbf{E}_i^T\Delta\xi_{i,1,\dots,N})$$

直觉上：**深度残差 = 本身梯度 - 位姿更新"吃掉"的那部分**。Schur 把"密集深度"和"稀疏姿态"解耦，使得每次只更新小维度姿态，再 cheap 地反推深度。

之后用 RAFT 的 convex upsampling 把 1/8 深度上采样到全分辨率，权重由 GRU 学出来。

### 3.2 Visual-Inertial Factor Graph (GTSAM)

IMU 状态向量定义：

$$\mathbf{b}_k = [\mathbf{b}_{a,k} \; \mathbf{b}_{g,k}], \quad \mathbf{x}_k = [\mathbf{T}_{b_k}^w \; \mathbf{v}_{b_k}^w \; \mathbf{b}_k]$$

- $\mathbf{T}_{b_k}^w$：IMU body frame 在 world 中的位姿
- $\mathbf{v}_{b_k}^w$：速度
- $\mathbf{b}_{a,k}, \mathbf{b}_{g,k}$：accelerometer / gyroscope bias

视觉因子从 DBA Schur 后出来，$\xi_w^{c_k}$ 是 $\mathbf{T}_w^{c_k}$ 的 Lie algebra，$\mathbf{H}_c, \mathbf{v}_c$ 是 DBA 的 information matrix 和 vector。

公式 (8) IMU 预积分残差（VINS-Mono 那套）：

$$r_b(\mathbf{x}_k, \mathbf{x}_{k+1}) = \begin{bmatrix} \mathbf{R}_w^{b_k}(\mathbf{p}_{b_{k+1}}^w - \mathbf{p}_{b_k}^w + \tfrac12\mathbf{g}^w\Delta t_k^2 - \mathbf{v}_{b_k}^w\Delta t_k) - \hat\alpha_{b_k}^{b_{k+1}} \\ \mathbf{R}_w^{b_k}(\mathbf{v}_{b_{k+1}}^w + \mathbf{g}^w\Delta t_k - \mathbf{v}_{b_k}^w) - \hat\beta_{b_k}^{b_{k+1}} \\ \mathrm{Log}((\mathbf{R}_{b_k}^w)^{-1}\mathbf{R}_{b_{k+1}}^w(\hat\gamma_{b_{k+1}}^{b_k})^{-1}) \\ \mathbf{b}_{a,k+1} - \mathbf{b}_{a,k} \\ \mathbf{b}_{g,k+1} - \mathbf{b}_{g,k} \end{bmatrix}$$

- $\hat\alpha, \hat\beta, \hat\gamma$：IMU 预积分的平移/速度/旋转量（用 bias 重传播时只需要一阶近似）
- $\mathbf{g}^w$：重力向量（世界系）
- $\Delta t_k$：两关键帧之间时间间隔

注意 paper 这里写的是 optional IMU —— 如果没有 IMU，factor graph 退化成纯视觉姿态约束，对应 real-world app 用手机 1Hz GPS 那种极限场景。

### 3.3 Depth Uncertainty Estimation（我觉得这是这篇论文最 under-appreciated 的点）

从 DBA 的 Hessian 算 marginal covariance：

$$\Sigma_T = (\mathbf{H}/\mathbf{C})^{-1}$$
$$\Sigma_{d^{-1}} = \mathbf{C}^{-1} + \mathbf{C}^{-1}\mathbf{E}^T\Sigma_T\mathbf{E}\mathbf{C}^{-1}$$
$$= \mathbf{C}^{-1} + (\mathbf{L}^{-1}\mathbf{E}\mathbf{C}^{-1})^T(\mathbf{L}^{-1}\mathbf{E}\mathbf{C}^{-1})$$

- $\mathbf{H}/\mathbf{C}$：对 $\mathbf{C}$ 做 Schur 后的 reduced Hessian
- $\mathbf{L}$：$\Sigma_T$ 的 Cholesky 下三角分解
- $\Sigma_{d^{-1}}$：每像素逆深度的 marginal covariance

Intuition：DBA 输出"深度 + 这个深度有多可信"。**这种不确定度直接喂给后续 Gaussian 添加与 Dynamic Eraser**。比如天空、远景 depth 算不准的地方，$\Sigma_{d^{-1}}$ 很大，map 时直接 mask 掉，避免 SLAM 的经典 floaters。

---

## 4. 2D Gaussian Map（论文的核心模块）

### 4.1 Online Mapping Process

关键设计：**先大量添加，后剪枝**，舍弃了 3DGS 原版 clone-and-split + reset opacity 的策略。原因：在 GS-SLAM 这种 incremental 场景下，reset opacity 表现不稳定，前向视图（driving）会反复触发 densify 但又没真正几何需要。

新 keyframe 来时：
1. 渲染当前帧 color 和 depth
2. 在 frustum 内删除：高 RGB loss、高 depth loss、投影半径过大的 Gaussians
3. 重渲染 accumulation map，在低 accumulation 区域按比例加点（与低 acc 像素面积成正比）
4. 从最近 keyframe list 随机抽帧训练

Loss（沿用 2DGS）：

$$\mathcal{L} = \lambda_{rgb}\mathcal{L}_{rgb} + \lambda_{depth}\mathcal{L}_d + \lambda_{norm}\mathcal{L}_n + \lambda_{acc}\mathcal{L}_{acc}$$

实验配置：$\lambda_{rgb}=1.0, \lambda_{depth}=0.5, \lambda_{normal}=0.1, \lambda_\alpha=0.1$。

2DGS 渲染公式 (11)：

$$f(p) = \alpha\cdot\exp(-\tfrac12(p-\mu)^T(p-\mu))$$
$$(C, D, N, A) = \sum_{i=1}^{N}(c_i, z_i, n_i, 1)f_i\prod_{j=1}^{i-1}(1-f_j)$$

- $f_i$：第 $i$ 个 Gaussian 的"沿 ray 的 footprint 权重"，由 2D Gaussian 在 ray 上的积分给出
- $\alpha$：opacity
- $\mu$：Gaussian 中心在 ray 上的投影
- $c_i, z_i, n_i$：颜色、相机系深度、法向
- $A$：accumulation（沿途 alpha 累积）

为什么选 2DGS 而非 3DGS？2DGS 把每个 Gaussian 当成 surface element (surflet)，几何上更贴 surface，normal 更准，更利于 SLAM 几何一致性。这条选择很关键，让后面 normal loss、depth uncertainty、loop correction 都有 surface 概念。

### 4.2 Score Manager（我觉得是论文最核心的工程贡献）

每个 Gaussian 维护两个分数 + 一个 frame ID：

$$S_C(g) = \sum_{t=0}^{K}\sum_{u=0}^{P} f_i\prod_{j=1}^{i-1}(1-f_j)$$
$$S_E(g) = \max\Big(\big\{\sum_{u=0}^{P}\mathcal{L}_{rgb}(u)f_i\prod_{j=1}^{i-1}(1-f_j)\big\}_{t=0,\dots,K}\Big)$$
$$ID(g) = \arg\max_t\Big(\big\{\sum_{u=0}^{P} f_i\prod_{j=1}^{i-1}(1-f_j)\big\}_{t=0,\dots,K}\Big)$$

- $S_C$：contribution score，对 K 个 keyframe、对它触及的 P 个像素累积权重，**用求和**
- $S_E$：error score，对 K 个 keyframe 取**最大值**（不是求和）
- $ID(g)$：贡献最大的那一帧的 frame ID

为什么 contribution 用 sum、error 用 max？作者讲得很直白：
- contribution 高的 Gaussian 有三种情况：单帧贡献大、影响多帧、两者皆大 —— 都该保留
- error 高的 Gaussian 如果用 sum，一个被多帧观察到的"普通 Gaussian"会累积出大 error 被误删；用 max 则只惩罚真正在某帧造成大 loss 的 Gaussian

这种 asymmetric aggregation 是个很漂亮的工程直觉，类似 attention 里 "max-pooling 抑制噪声 vs sum-pooling 保留长尾"。

之后三个机制：

**(1) Status Control（stable / unstable 二态）**
每 $\Delta n_{status}=400$ iter（一次 keyframe list 全替换）：
- unstable 且 $S_C < 10^{-4}$ → stable（被"冻结"出优化池）
- stable 且 $S_E > 0.5$ → unstable 并 reset $S_C, S_E$（重新进入优化）

**Stable 状态的妙用**：sparse Adam 优化时 mask 掉 stable Gaussians，加速；同时保留它们以便 revisit 时重新激活。在转弯、跨房间时历史 Gaussian 会重新进入 frustum，如果按距离 prune 就会灾难性丢失，stable 标记避免这种情况。

**(2) Storage Control**
每 $\Delta n_{storage}=200$ iter：unstable 且 $S_C < 0.5$ 的 Gaussian 直接 prune。

**(3) GPU-CPU Transfer**
每 $\Delta K=8$ keyframe：用 $ID(g)$ 算 Gaussian 关联的 pose 距当前位姿的距离，距离 $< \tau$ 的从 CPU 调入 GPU，$> \tau$ 的从 GPU 调到 CPU。

这里用 $ID(g)$ 而非直接算 Gaussian 中心距离的原因：算百万 Gaussian 中心距离本身就要花时间，而每个 Gaussian 已经记录了贡献最大帧的 ID，直接查 pose 距离即可，复杂度从 O(#Gaussians) 降到 O(#Gaussians × 常数查表)。

Ablation 表 VII 数据：
- ScanNet-0106：score manager 把 Gaussian 数从 4.04M 降到 1.96M（102.4 周期），PSNR 反而升了（22.98 → 23.02）
- Waymo-Scene13：从 1.78M 降到 1.06M，PSNR 仅降 1.2 dB

我自己的联想：这套 score-based 管理和 recent works 比如 "Reducing the Memory Footprint of 3D Gaussian Splatting" (Hou et al. 2024, arXiv:2410.18931) 以及 "Taming 3DGS" 是同一波思路 —— 把 GS 当成资源池而非稠密表达，按"贡献 vs 误差"做生命周期管理。这其实是把 NeRF-SLAM 的"在 keyframe 间挑 anchor"思想迁移到了 explicit representation 上。

### 4.3 Sample Rasterizer（backward 加速 273%）

原版 2DGS：每个 GPU block 负责 16×16 tile = 256 pixels，每个 thread 处理 1 个 pixel 的 backward，迭代数 = 该像素关联的 Gaussian 数。瓶颈 = 全 tile 中"最重像素"的迭代数，backward 时间被 worst-case 主导。

VINGS-Mono 的改动：
1. forward 时每 32 个 Gaussian 存一次中间状态到 buffer（warp 对齐）
2. backward 时把 GPU 划成 warp（32 thread 一组），每个 warp 处理一个 Gaussian 集
3. 在每个 tile 内挑 loss 比例最高的 $r$ 部分 pixel 做反向，每 thread 迭代数 = $256 \times r$
4. 取 $r = 0.5$，backward 从 11.55ms → 4.23ms，加速 273%

Ablation 数据（KITTI）：
- Full-Tile backward：7.21ms
- Sample-based（$r=0.5$）：4.23ms，PSNR 仅降 0.56 dB

直觉：**高 loss 像素主导梯度，低 loss 像素梯度信息量小，按 loss 排序做 top-k sampling 等价于 importance sampling on gradient**。这跟 NeRF 早期 "pixel sampling by error" 思路一致，但 GS 的 rasterizer 之前没这么改，主要因为 forward/backward 对称的 thread 结构难以打破 —— 作者用 forward 存中间状态打破对称，这个 hack 很关键。

参考 Taming 3DGS: https://taming3dgs.github.io/

### 4.4 Single-to-Multi Pose Refinement

公式 (14)：

$$\hat\mu_k = \mathbf{T}_{c_k}^w \mathbf{T}_{c_k}^{\hat c_k} \mathbf{T}_w^{c_k}\mu_k$$
$$\hat{\mathbf{T}}_{c_k}^{\hat w} = \mathbf{T}_{c_k}^w(\mathbf{T}_{c_k}^{c_k})^{-1}$$
$$\hat I_k = \mathcal{R}(\{\hat\mu_k, s_k, c_k, r_k\}, \hat{\mathbf{T}}_{c_k}^w)$$
$$\min_{\{\mathbf{T}_{c_k}^{c_k}\}} \mathcal{L}_{rgb}(\hat I_k, I_k)$$

- $\mathbf{T}_{c_k}^{c_k}$：作为优化变量的相对姿态扰动
- $\mu_k$：Gaussian 中心
- $s_k, c_k, r_k$：scale、color、rotation（保持不变，只优化 pose）

关键：因为 score manager 已经给每个 Gaussian 绑了 $ID(g)$（贡献最大 keyframe），渲染单帧 $\hat I_k$ 时，gradient 流回到 $\{\hat\mu_k\}$ 后再传到该 Gaussian 所属 keyframe 的 $\mathbf{T}_{c_k}^{c_k}$。**一次渲染同时优化视野内所有 keyframe 的 pose**。

实验对比：
- 不做 pose refine：ScanNet-0106 ATE 0.25m
- 只优化当前帧 pose（SplaTAM/MonoGS 做法）：0.19m
- 优化可见帧 pose（本文）：0.16m

直觉：**GS-SLAM 之前是"用当前帧 loss 训当前帧 pose"，但当前帧的 Gaussians 大多是从其它 keyframe 来的 —— 这些 Gaussians 的位置依赖于历史 keyframe 的 pose。把 gradient 同时回传到那些 keyframe，等于用一个 dense photometric error 同时约束多个 pose**，比单帧优化信息量高得多。

---

## 5. NVS Loop Closure（创新点，把 GS 自身当 loop detector）

### 5.1 Loop Detection 三步走

1. **特征匹配**：和距离阈值内、frame ID 差 > 10 的历史帧做 LightGlue 匹配，匹配点数 $> N_{match}^{th}=50$ 的帧组成 $\{I_{t_k}\}^{filt}$，按匹配数降序排
2. **Render Depth + Solve PnP**：渲染当前帧深度 $\hat D_{t_n}$，用匹配点 + 深度做 PnP 求相对 pose $T_{t_n}^{\hat t_m}$，进而 $\hat T_{t_n}^w = T_{t_m}^w T_{t_n}^{t_m}$。**注意 PnP 只用深度低于阈值的点**，远处深度不稳。
3. **Novel View Synthesis 验证**：用 $\hat T_{t_n}^w$ 在 Gaussian map 上渲染 $\mathcal{R}(T_{t_n}^w)$，和 $I_{t_n}$ 算 L1 loss。loss 低于阈值 **或** 低于 $\{I_{t_k}\}^{filt}$ 中其它帧中位 loss 的 1/10 → loop detected。

Intuition：**"两张图是不是同一场景" → "这张新图能不能在 GS map 里被当作合法 novel view 渲染出来"**。GS 天生就是 NVS 引擎，直接用渲染 loss 当 loop 验证，省去 BoW 的 vocabulary 维护。这种 framing 很漂亮。

### 5.2 Loop Correction：避免 retrain 全部历史帧

1. **Pair Gaussian with Pose**：所有历史 keyframe forward 一遍，每个 Gaussian 选 $S_C$ 最大的 pose 作为 matched pose。1000 帧约 2 秒，因为 GS 渲染快。

2. **Correct Pose & Gaussians**：对历史 frame $\{T_{c_k}^w\}$ 加 loop constraint 做 pose graph optimization 得 $\{T_{c_k}^{w\prime}\}$，按平移向量 norm 比例算 scale：

公式 (15)：

$$\mu_i' = T_{c_k}^{w\prime} T_w^{c_k}\mu_i$$
$$r_i' = R^{-1}(T_{c_k}^{w\prime} T_w^{c_k} R(r_i))$$

- $\mu_i, r_i$：原 Gaussian 中心与旋转（quaternion）
- $\mu_i', r_i'$：校正后
- $R(\cdot)$：quaternion → rotation matrix
- $k = ID(g_i)$：该 Gaussian 所属 keyframe

校正后只再 train 100 iter 做精修，再按 $S_C$ prune。整个 loop correction 几秒级，相对 GO-SLAM 那种"global retraining"快了几个数量级。

为什么这套能 work？因为每个 Gaussian 已经和 keyframe pose 绑定（通过 $ID(g)$），pose graph 优化后相当于给每个 Gaussian 做了一次刚体变换（scale + rotation），几何结构基本保留。然后 100 iter fine-tune 修补局部。这种"刚性变换 + 微调"模式比 retrain 全图高效得多，是大场景 loop closure 必备。

---

## 6. Dynamic Object Eraser

Open-set 语义分割 (Fast-SAM) 给语义 mask $\{M_k\}$，但语义"可能动态"的 mask 不一定真在动。比如停车场里静止的车。需要时间维度验证：

Re-rendering Loss：

$$\mathcal{L}_{re} = \mathcal{L}_{SSIM} \cdot \mathcal{L}_{L1} \cdot \Sigma_{d^{-1}}$$

- $\mathcal{L}_{SSIM}$：对纹理敏感
- $\mathcal{L}_{L1}$：对颜色敏感
- $\Sigma_{d^{-1}}$：depth uncertainty（前面 front end 算的）

这里乘上 $\Sigma_{d^{-1}}$ 的目的是：**对深度不准的边缘区域放大 re-rendering loss**，让动态物体的边缘（depth 不稳）更容易被识别。

公式 (16)：

$$M_{dyn,k} = \Big(\frac{\sum\mathbf{1}(\mathcal{L}_{re}(M_k) > \mathcal{L}_{re}^{90\%})}{\sum\mathbf{1}(M_k)} > \gamma\Big) \wedge \Big(\overline{\mathcal{L}_{re}}(M_k) > \mathcal{L}_{re}^{th}\Big)$$
$$M_{dyn} = \bigcup_{k=0}^{K} M_{dyn,k}$$

- $\mathcal{L}_{re}^{90\%}$：pixel-level re-rendering loss 的 90 分位数
- $\gamma = 20\%$：mask 内超过 90 分位 loss 的像素占比阈值
- $\mathcal{L}_{re}^{th}$：mask 平均 loss 阈值
- 两个条件 AND 才判动态

实验 BONN dataset：ATE 从 30.25cm（无 eraser）降到 4.34cm（有 eraser），击败 ReFusion (27.65cm) 和 RodynSLAM (12.2cm)。

直觉：**用"渲染 loss 高 + 深度边缘不确定"作为动态的代理**。动态物体移动后，map 在原位置还是旧外观，新观测和旧 map 不一致 → loss 高。结合语义先验，避免误判静态纹理复杂区域。

---

## 7. 实验结果梳理

### 7.1 Localization

**室内（ScanNet + BundleFusion，ATE cm）**：
- ORB-SLAM3：常 25-243cm
- DROID-SLAM：11-265cm
- MonoGS：62-191cm
- PhotoSLAM：151-359cm
- VINGS-Mono：15-92cm，**普遍 SOTA 或 near-SOTA**

**户外（Waymo + Hierarchical，ATE m）**：
- 大多数 baseline 在 SmallCity 直接 fail（MonoGS 全黑、PhotoSLAM 跟丢回原点）
- VINGS-Mono：SmallCity 2.82m、Campus 1.03m、Waymo 0.91-2.67m

**VIO（KITTI / KITTI-360）**：$t_{rel}\%$ / $r_{rel}°/100m$
- KITTI 02：VINGS 2.64% / 0.44° vs VINS 2.08/1.68, ORB-SLAM3 3.51/1.42, iSLAM 2.08/0.53
- KITTI360 unsync 高频 IMU 上：VINGS 在 00/02/05/06/10 序列几乎全面领先，比如 02 仅 0.58% vs iSLAM 38.46%

值得注意：iSLAM 在 KITTI360 unsync 上崩了（38% 误差），可能因为 unsync + 高频 IMU 处理不稳，VINGS 反而最稳。

### 7.2 Rendering（PSNR/SSIM/LPIPS）

室内 ScanNet 平均：VINGS PSNR 22.43dB, SSIM 0.79, LPIPS 0.22。MonoGS、PhotoSLAM 都明显落后。

户外 5 个 dataset：
- KITTI-360 08：VINGS 24.52dB vs MonoGS 16.08dB vs PhotoSLAM 15.81dB
- Waymo 14：23.76dB vs MonoGS 23.00dB
- MegaNeRF Building/Rubble：25.45 / 25.21 dB，大幅领先 GO-SLAM 的 20.71/20.81

PSNR 差距 5-10 dB 是巨大差异，主要来自 score manager 控制 floaters + depth uncertainty 抑制天空。

### 7.3 Runtime

| Dataset | Frames | Total | Tracking/frame | FPS | Model size |
|---|---|---|---|---|---|
| Waymo-01 | 198 | 117s | 214ms | 1.69 | 386 MB |
| Hiera-SmallCity | 877 | 739s | 247ms | 1.18 | 1817 MB |
| KITTI-Odom08 | 5177 | 4560s | 273ms | 1.13 | 10366 MB |

Tracking 模块约 4-5 FPS，Mapping 慢于 tracking 但并行运行。10GB 模型存 8 km KITTI 全程 51.73M Gaussians。

### 7.4 Mobile App & Real-world

- Flutter app，iOS/Android/Windows 跨平台
- 手机收 480×720 @30Hz + IMU，传到 server 跑 VINGS-Mono
- 自采校园 1.02km × 0.4km 数据，自行车 10km/h 采集，20Hz RGB + 1Hz GPS 验证：几乎无 scale drift

---

## 8. 我的理解与延伸联想

### 8.1 为什么 VINGS-Mono 能"在大场景活下来"？

把它的成功拆解成几条**正交但相互支撑**的设计：

1. **稠密前端 → 几何先验**：DROID-SLAM 风格 DBA 给出稠密深度 + uncertainty，避免了 MonoGS / PhotoSLAM 那种 sparse ORB 点初始化的脆弱性。driving 场景中 texture 弱、forward motion 退化严重，稠密光流约束是救命稻草。
2. **Schur Complement 给出 compact 视觉因子**：把密集深度信息塞进 GTSAM，使得 visual-inertial factor graph 规模可控，可以公里级运行。
3. **2DGS 而非 3DGS**：surface-aligned 表示让 normal loss、depth uncertainty 都有几何意义。
4. **Score Manager 是大场景 scalable 的关键**：没有这套，51M Gaussian 一个 RTX 4090 装不下。CPU-GPU transfer + stable/unstable 二态让历史信息不丢、当前活跃集小。
5. **Loop Closure 不 retrain**：通过 Gaussian-pose pairing 用 $ID(g)$ 把 pose 优化结果直接映射到 Gaussian 上，再微调 100 iter。这是把"pose 优化"和"dense map 优化"做了 decoupling，否则公里级 retrain 不可行。
6. **Dynamic Eraser 用 re-rendering + depth uncertainty**：巧妙利用 GS 已经渲染好这个"副作用"，免去了 4D Gaussian / dynamic NeRF 那种 offline 训练时序模型的负担。

### 8.2 与其它方法的对比直觉

- **vs GO-SLAM**：GO-SLAM 用 covisibility matrix + 全局 incremental 优化，O(N²) 存储；VINGS 用 GS NVS 做 loop detection，存储线性，且 dense map 是 explicit Gaussian 不是隐式 NeRF。
- **vs SplaTAM**：SplaTAM 用深度相机，depth-supervised Gaussians，pose 优化用 single-frame photometric；VINGS 单目 + DBA，single-to-multi refinement，量级差几个数量级。
- **vs MonoGS**：MonoGS 用 ORB-SLAM3 front + 3DGS map，大场景 floaters 严重；VINGS 用 dense DBA front + 2DGS + score manager，几何更鲁棒。
- **vs LoopSplat / Loopy-S-LAM**：submap-based loop correction，但单目 scale drift 在 submap 内未解决；VINGS 直接在 Gaussian-pose 绑定上做刚性变换，能处理单目尺度漂移。
- **vs LIVGaussMap**：要 LiDAR；VINGS consumer-friendly。
- **vs NEWTON**：view-centric submap + NeRF，难以应对户外快运动；VINGS 用 2DGS 显式表达，运动鲁棒。

### 8.3 仍存在的限制（paper 自己也承认）

- **极高速度运动**：DBA 在大 frame interval 下恢复几何困难，GS 多次 iter 也限制了重建速度。作者 future work 提到要给 DBA 加 prior、用 point transformer / Large Spatial Model 直接出 Gaussian 属性。
- **未 on-device**：仍依赖 server，未来要部署到 edge。

### 8.4 我额外想到的几个方向

- **Depth uncertainty 的进一步使用**：现在只在 mask 和 dynamic eraser 用了 $\Sigma_{d^{-1}}$。其实可以拿它做 per-Gaussian isotropic/anisotropic regularization，比如对高不确定区生成的 Gaussian 加更强的 prior pull 到 frontier surface。这个方向在 NeRF-SLAM 里 (Rosinol et al. 2023, NeRF-SLAM) 已经做过，2DGS-SLAM 还没充分利用。
- **Score Manager 与 recently active "Gaussian lifecycle" 文献的连接**：例如 Compact-3DGS (Lee et al. 2024)、Self-organizing Gaussians (Chen et al. 2024)、EAGLES、Mip-Splatting。Score-based 管理可以和这些"anchor-based"或"SH-pruning"的方法叠加。
- **Loop Detection 用 NVS 这个思路本身**：这其实把"视觉 place recognition"重新定义成"view synthesis consistency"。如果再把 rendering loss 升级成 LPIPS/DINOv2 feature loss，对光照变化、季节变化的鲁棒性应该更强。最近 DINOv2 + SALAD 这条线 (https://github.com/SaladVision/SALAD) 已经证明 self-supervised feature 比 BoW 强很多。
- **DBA + 2DGS 的几何耦合**：DBA 给出的 marginal covariance 实际上是"前端认为哪里几何可信"，而 2DGS 的 normal loss 是"渲染一致性认为哪里 surface 平滑"。这两者可以做信息融合 —— 比如把 $\Sigma_{d^{-1}}$ 作为 normal loss 的加权权重。
- **Mobile 端的进一步压缩**：10366 MB model for 8km KITTI 太大。结合 SOG (Sort-Free Gaussian Splatting, arXiv:2410.18931) 或 radix sort 替代、压缩 SH 到 RGB 的方法，应该能压到 1-2GB。
- **Pose Refinement 的理论性质**：把 Gaussians 按 $ID(g)$ 绑到 keyframe 后，pose 优化变成了"每个 Gaussian 是其 anchor pose 的相对量"。这和 neural field 里 "canonical frame + per-frame deformation" 思想接近 ( Nerfies / HyperNeRF / 4DGS )。理论上可以推出这套优化的 Jacobian 结构、可观测性条件，做 SLAM 风格的 observability analysis。
- **与 diffusion-based pose / depth prior 结合**：Metric3D 已经用了。未来用 diffusion-based monocular depth (Marigold, GeoWizard) + DBA 联合优化，可能在弱纹理室内、地下停车场这种 GPS-denied 场景表现更好。

### 8.5 一个 meta 观察

VINGS-Mono 的整体设计哲学是"**前端用学习的稠密方法获取几何先验 + IMU 因子；后端把 GS 既当 map 又当 place recognizer**"。这把传统 SLAM 里 frontend (VO/VIO) - backend (loop closure / pose graph) - mapper (dense reconstruction) 三件套重新融合：**GS 不只是被动的 dense representation，它本身承担了 place recognition、loop correction、dynamic detection 三重身份**。这种"用一个 representation 干多件事"的思路，跟 recent "Foundation Model as SLAM" (DPVO, DROID-SLAM, iSLAM, Large Spatial Model) 是平行发展的两个流派：VINGS 偏 explicit-Gaussian，LSM 偏 implicit-neural。我倾向于 explicit 这条线在工程上更容易 scale，因为可解释、可裁剪、可 cache 到 disk，而 neural field 的 inductive bias 难以控制。

---

## 9. 总结

VINGS-Mono 把 monocular GS-SLAM 从室内推到公里级户外，核心是四个工程模块协同：

1. **DROID-style DBA + GTSAM**：稠密几何先验 + IMU 融合，单目尺度可观测
2. **2DGS + Score Manager + Sample Rasterizer**：scalable 到 50M+ Gaussians，单卡可跑
3. **NVS Loop Closure**：把 GS 当 place recognizer，loop correction 用 Gaussian-pose pairing 避免全量 retrain
4. **Dynamic Eraser**：re-rendering loss + depth uncertainty + 语义 mask，heuristic 但有效

它不是某一个 single trick 的胜利，是**把 GS 在 SLAM 系统里每一处可能卡住的地方（init、densify、pruning、memory、loop、dynamic）都做了具体工程优化**。这种系统性工程反而很难在 paper 里完全讲清楚 —— 比如 $S_C$ 用 sum、$S_E$ 用 max 这种细节，以及 stable/unstable 二态管理对历史 Gaussian 保护，都是踩了很多坑才悟出来的。读 paper 时建议重点看 Algorithm 1、公式 (13)/(14)/(15) 这三处，是整套设计的"压舱石"。

参考链接汇总：
- 项目主页：https://vings-mono.github.io
- 2DGS: https://2d-gaussian-splatting.github.io/
- 3DGS: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- DROID-SLAM: https://github.com/princeton-vl/DROID-SLAM
- RAFT: https://github.com/princeton-vl/RAFT
- VINS-Mono: https://github.com/HKUST-Aerial-Robotics/VINS-Mono
- ORB-SLAM3: https://github.com/UZ-SLAMLab/ORB_SLAM3
- LightGlue: https://github.com/cvg/LightGlue
- Metric3D: https://github.com/YvanYin/Metric3D
- Fast-SAM: https://github.com/CASIA-IVA-Lab/FastSAM
- GTSAM: https://gtsam.org/
- evo: https://github.com/MichaelGrupp/evo
- KITTI: http://www.cvlibs.net/datasets/kitti/
- KITTI-360: http://www.cvlibs.net/datasets/kitti-360/
- Waymo Open: https://waymo.com/open/
- MegaNeRF: https://megenerf.github.io/
- Hierarchical 3DGS: https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/
- ScanNet: http://www.scan-net.org/
- BundleFusion: https://graphics.stanford.edu/projects/bundlefusion/
- BONN RGB-D dynamic: https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/
- SplaTAM: https://splatam.github.io/
- MonoGS: https://rmurai.co.uk/projects/GaussianSplattingSLAM/
- PhotoSLAM: https://huajianup.github.io/projects/Photo-SLAM/
- GO-SLAM: https://youmi-zhy.github.io/go-slam/
- LoopSplat: https://loopsplat.github.io/
- Loopy-SLAM: https://lukaskoestler.github.io/loopy-slam/
- Taming 3DGS: https://taming3dgs.github.io/
- SOG (sort-free): https://arxiv.org/abs/2410.18931
- Large Spatial Model: https://arxiv.org/abs/2410.18956
- DINOv2: https://dinov2.metademolab.com/
- SALAD: https://github.com/SaladVision/SALAD
