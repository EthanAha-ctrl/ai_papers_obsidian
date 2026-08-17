---
source_pdf: VisionPAD.pdf
paper_sha256: d4fd819883b5d82e1eda06861ddc2932d3a1686a15d96a7211750735f6ac4cc4
processed_at: '2026-08-13T01:50:20-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，用白板讨论的语气来讲。

---

## 这帮人想干嘛

Autonomous driving 现在主流是纯摄像头方案，6 个相机拍一圈，用神经网络搞出 BEV 图或者 occupancy grid，然后做 detection、规划。问题是训这些东西需要海量 3D 标注——每个 frame 都要标 box、occupancy、map，贵得要死。

所以大家都在想：能不能先无监督地 pre-train 一个 encoder，让它先"看懂"世界，再拿少量标注 fine-tune 下游任务？

## 之前的人怎么做的，问题在哪

UniPAD 是最近最猛的一个：把 BEV features 喂给一个 NeRF renderer，让它重建出 depth map 和 RGB image，用 LiDAR 投影的 depth 和 camera 的 RGB 做 supervision。encoder 为了重建得准，被迫把 3D geometry 编进 features 里。

问题：**它需要 LiDAR depth 当监督信号**。纯 image 监督的话基本学不动，performance 甚至会掉。那就尴尬了——我要是装了 LiDAR 还费这劲干啥？全 camera 方案的意义就在于没有 LiDAR。

## VisionPAD 的核心 idea

三个 trick 组合拳：

**Trick 1：把 NeRF 换成 3D Gaussian Splatting**

NeRF 渲染慢是因为它要对每个像素 cast 一条 ray，沿 ray 采样很多点，每个点过 MLP 算 density 和 color。分辨率越高，ray 越多，越慢。所以 UniPAD 每次只能采少量 ray，geometry 学不细。

3D-GS 不一样，它是把 3D 空间里的一堆 Gaussian 椭球直接"拍"到 2D 图像上，类似 rasterization，跟分辨率关系不大。同样算力下能 render 更清晰的图，gradient 更密集，geometry 学得更好。

实测：NeRF 1973MB 显存、900ms 一帧；3D-GS 134MB、70ms。**显存降 13 倍，速度提 13 倍**。这个差距太大，意味着可以用更大 batch、更高分辨率、更长时序——pre-training scaling 的几个关键维度全打开了。

**Trick 2：让网络预测每个 voxel 的速度，warp 到下一帧做重建**

加一个小的 velocity head，对每个 voxel 预测一个 3D 速度向量。然后把这个 voxel 按速度 × 时间间隔挪到下一帧的位置，用 warped voxel render 下一帧的 image，跟真实的下一帧 RGB 比 L1 loss。

为什么这招有用？因为静态 voxel 速度 = 0，动态 voxel 速度 ≠ 0，网络为了重建准，必须把"哪些 voxel 在动、动多快"学对。这就是把 motion 信息注入 encoder 的过程。

而且这本质上是 world model 的简化版——"当前状态 + 速度 → 下一帧状态 → 观测"，只是假设匀速运动。简单但够用。

**Trick 3：Photometric consistency（光度一致性）**

这是 self-supervised depth estimation 领域几十年的老把戏：用当前帧 render 出来的 depth，把每个 pixel 反投影到 3D，再用 ego pose 变换到相邻帧坐标系，再投到相邻帧的 pixel 坐标，grid sample 相邻帧的 RGB。如果 depth 准、pose 准，warp 出来的图应该跟当前帧一致。

Loss 是 SSIM + L1 的组合。

这一步给网络一个直接的几何反馈信号：depth 错了 → warp 出来的图糊了 → loss 大 → 网络调 depth。整个过程不需要 LiDAR。

## 为什么这三招组合起来能 work

关键 insight 是：**单靠当前帧 RGB 重建，网络可以"作弊"**——比如把 background 涂对、object 位置乱标，RGB loss 也能很低。但加上后两招就不行了：

- Trick 2 逼网络学 motion，光涂对当前帧不够，得知道物体怎么动
- Trick 3 逼网络学 depth，光 color 对不够，几何关系也得对

三个 signal 互相约束，网络被逼着学到了真正的 3D scene structure。

Ablation 也很说明问题：单独上 3D-GS 替换 NeRF，performance 反而掉了（mAP 18.2 vs baseline 19.4）。必须三个 trick 一起上，performance 才起飞。

## 结果如何

在 nuScenes 上：

- 3D detection：纯 image pre-train 比 UniPAD 纯 image 版好 3.8 mAP，接近 UniPAD 加 LiDAR 的版本
- Occupancy：比 UniPAD 纯 image 版好 5.4 mIoU
- Map segmentation：好 5.4 IoU
- 数据效率：只给 25% 下游标注数据时，优势更大（mAP 高 ~6 个点）

最后一点很关键——**数据越少，pre-training 越值**。这跟 Tesla 的逻辑完全一致：大量无标注驾驶数据 pre-train，少量人工标注 fine-tune。

## 我觉得最值得琢磨的点

1. **Renderer 选对了，self-supervised learning 的 signal efficiency 就上来了**。NeRF 这种 implicit + slow 的 representation 不适合 pre-training，3D-GS 这种 explicit + fast 的才适合。这可能是个 general 的 insight：pre-training 的 auxiliary task 必须 compute-efficient，否则 gradient signal 不够密集。

2. **World model 路线可能是 pre-training 的终局**。Trick 2 就是简化版 world model。未来大概率会把"velocity × Δt"这个 linear transition 换成 learned transition（transformer 或 diffusion），变成真正的 generative world model pre-training。

3. **三个 inductive bias 全是 geometric/physical 的**：3D voxel structure、multi-view consistency、temporal velocity smoothness。这些是物理先验，不需要学，直接 encode 进网络结构。在 driving 这种 structured environment 下，物理先验比 learned prior 更 sample efficient。

4. **Limitations 没说但值得注意**：velocity 假设匀速（转弯刹车会错）、photometric consistency 假设 Lambertian（玻璃金属反光会错）、occlusion 处理没提。这些都是 self-supervised depth estimation 老问题了，VisionPAD 没完全解决，只是把问题 wrap 进了一个更大的 framework。

## 一句话总结

**把 NeRF 换成 3D-GS 解决了"渲染太慢学不动"的问题，加 velocity estimation 和 photometric consistency 解决了"纯 image 没有几何监督学不准"的问题，三者合起来让纯摄像头 pre-training 第一次真正 work 了。**

核心直觉就是：pre-training 的 auxiliary task 必须**算得快**（3D-GS）+ **信号够硬**（多帧几何 + 运动一致性），两者缺一不可。

---

# VisionPAD 深度解析

Andrej 你好！这篇 paper 确实是 2024-2025 年 self-supervised pre-training for autonomous driving 这条线里很值得琢磨的一篇。我会从 your "build intuition" 的角度，把这篇 paper 的设计逻辑、技术细节、与整个领域的关系都讲透。

## 1. 这篇 paper 到底在解决什么问题

Autonomous driving 的 vision-centric perception (BEVFormer [1], UVTR [2], BEVDet [3], OccNet 等) 都依赖昂贵的 3D annotations (boxes, occupancy, HD maps)。Pre-training 显然是出路，但之前的 rendering-based pre-training (UniPAD [4], ViDAR [5], MIM4D [6]) 都隐含一个依赖：**需要 LiDAR-projected depth maps 作为 supervision**，否则 NeRF-style volume rendering 训不动 geometry。

VisionPAD 的核心论断是：**只要换一种更高效的 differentiable renderer (3D-GS 替代 NeRF)，再加两个 self-supervised signal (voxel velocity + photometric consistency)，纯 image supervision 就能学到 strong 3D geometric + motion representations**，甚至能超越有 LiDAR depth supervision 的 UniPAD。

这其实是 self-supervised depth estimation [7] 那条老路 (Monodepth2, FSRE) 在大规模 BEV pre-training 上的 "回归"，但配上 3D-GS 这种更高效的 renderer 之后，scaling 变得可行。

Ref:
- UniPAD: https://arxiv.org/abs/2312.13726
- ViDAR: https://arxiv.org/abs/2312.17655
- 3D-GS: https://repo.akm.tugraz.at/~kerbl/kerbl_gaussian_splatting.pdf
- Monodepth2: https://arxiv.org/abs/1806.01260

## 2. 整体架构直觉

VisionPAD 的 pipeline (Fig. 2) 可以拆成四个模块：

```
Multi-view multi-frame images 
    → Image encoder (ConvNeXt-S) → 2D features F_I ∈ R^{N×H×W×C}
    → View transformer (BEV lifting) → 3D voxel features V ∈ R^{X×Y×Z×C}
    → 3D-GS Decoder (anchor-based) → Gaussian primitives {g_k}
    → Differentiable rasterization → rendered images C_t, depth D_t
    → 三个 supervision:
        1) L_img: 当前帧 RGB reconstruction (L1)
        2) L_vel: velocity warp 后的 adjacent frame RGB reconstruction
        3) L_pc: photometric consistency via re-projection
```

关键 insight：**整个 pre-training 阶段 backbone 输出的 voxel features V 是 "真实可用的 representation"，下游 fine-tune 时直接复用 backbone + view transformer，丢掉 3D-GS decoder 和所有 auxiliary head**。这和 MAE [8] 的思路一脉相承——一个 non-trivial 的重建任务能让 encoder 学到有用的 features。

## 3. 为什么是 3D-GS 而不是 NeRF (这是 paper 的第一个 key insight)

UniPAD 用 NeRF 的 volume rendering，需要 ray sampling：每次 iteration 只能 sample 有限数量的 rays，image resolution 越高，采样比例越低，geometry 学习越差。这就是为什么 UniPAD 在纯 image supervision 下会掉点 (NDS 从 45.0 → 44.8)。

3D-GS [9] 用 splat-based rasterization，把 3D Gaussians 直接 splat 到 2D 像素上，**渲染成本对 image resolution 不敏感** (因为是 rasterization，不是 per-ray MLP query)。同样 compute budget 下，3D-GS 能 render 更高分辨率图像，捕捉更细粒度颜色，geometry 也学得更好。

Paper Table 6 给了一个很硬的对比：
- UniPAD-C (NeRF): 1973 MB memory, 900 ms latency
- VisionPAD (3D-GS): 134 MB memory, 70 ms latency

Memory 降 93.2%，latency 降 92.2%。这是一个数量级的差距，意味着 pre-training 可以 scale 到更长 temporal window、更大 batch size、更高分辨率——这些都是 self-supervised learning scaling 的关键变量。

我个人觉得这里有个深层 intuition：**NeRF 是 implicit scene representation (一个 MLP 编码一切)，3D-GS 是 explicit scene representation (一组 primitives 直接 store 属性)**。在 self-supervised pre-training 这种 "需要 dense gradient flow 到 3D space" 的场景里，explicit representation 因为有直接的 spatial anchor，gradient path 更短，更容易学到 geometry。

## 4. 公式细节拆解

### 4.1 3D Gaussian 参数化 (Eq. 1, 2)

$$\Sigma = \mathbf{R}\mathbf{S}\mathbf{S}^T\mathbf{R}^T$$

- $\Sigma \in \mathbb{R}^{3\times 3}$: 3D Gaussian 的 covariance matrix
- $\mathbf{R} \in \mathbb{R}^{4}$: 用 quaternion 表示的 rotation (4维，方便插值，避免 gimbal lock)
- $\mathbf{S} \in \mathbb{R}^3_+$: scaling vector (各向异性 scale)
- $\mathbf{S}\mathbf{S}^T$: diagonal scaling matrix 的 outer product，保证 positive semi-definite

直觉：covariance 必须正定，直接学 $\Sigma$ 不可行，所以 decompose 成 R·S·S^T·R^T。这等价于先 scale 再 rotate，是 SO(3)×Diag+ 的参数化。

投影到 2D:
$$\Sigma' = \mathbf{J}\mathbf{W}\Sigma\mathbf{W}^T\mathbf{J}^T$$

- $\mathbf{W}$: view transformation (世界 → 相机)
- $\mathbf{J}$: projective transformation 的 Jacobian (透视投影的局部 affine approximation)
- $\Sigma'$: 2D image plane 上的 covariance

直觉：透视投影是非线性的，但在每个 Gaussian 中心附近可以 affine linearize (用 first-order Taylor expansion)，J 就是这个 linearization 的 Jacobian。这是 3D-GS 比 NeRF 快的核心原因之一——不需要 per-point MLP，只需要一个解析的 Jacobian。

### 4.2 Alpha blending rendering (Eq. 3, 4)

$$\mathbf{C}(p) = \sum_{i \in K} c_i \alpha_i \prod_{j=1}^{i-1}(1-\alpha_j)$$

- $p$: 像素位置
- $K$: 沿 ray 击中的 Gaussian 集合 (按 depth 排序)
- $c_i$: 第 i 个 Gaussian 的 spherical harmonics color
- $\alpha_i$: 第 i 个 Gaussian 在 pixel p 处的 opacity × 2D Gaussian response
- $\prod_{j=1}^{i-1}(1-\alpha_j)$: 前面所有 Gaussian 的 transmittance

这是 classic volume rendering 的 front-to-back compositing，新视角合成领域几十年沿用 [10]。

$$\mathbf{D}(p) = \sum_{i \in K} d_i \alpha_i \prod_{j=1}^{i-1}(1-\alpha_j)$$

- $d_i$: 第 i 个 Gaussian 中心到相机的距离

同样的 alpha blending 用于 depth map rendering，这是 VisionPAD 引入 photometric consistency 的前提——需要 differentiable rendered depth。

### 4.3 Anchor-based Gaussian prediction (这是从 PixelSplat [11] 借来的)

每个 voxel 中心作为 anchor point，用 MLP 预测若干 Gaussian primitives 的属性：
- offset from voxel center: $\Delta\mu_k$
- SH coefficients: $c_k$
- opacity: $\alpha_k$ (用 tanh activation, <0 时直接 filter 掉)
- scale: $\mathbf{S}_k$
- rotation: $\mathbf{R}_k$

Paper Table 5 给了一个有意思的 ablation：
- Predict offset (相对 anchor) + scale learnable + rotation learnable: 27.3 NDS / 26.5 mAP
- Predict absolute mean: 26.8 NDS (掉了 0.5)
- Scale fixed 为 0.4 (voxel size 的一半): 25.3 mAP (掉了 1.2)
- Scale 范围 [0.1, 0.5] vs [0.2, 0.8]: 前者更好

直觉：**offset prediction 给了网络一个 residual learning 的 inductive bias**，类似 Faster R-CNN 的 anchor box offset regression——绝对位置太难学，相对 anchor 的 offset 更容易优化。Scale learnable 重要是因为不同物体 (远处的车 vs 近处的树) 需要不同 Gaussian 半径，fixed scale 等于强行让网络用同一分辨率描述所有尺度——这违反 multi-scale representation 的基本原则。

### 4.4 Voxel velocity estimation (Eq. 5 隐含, Algorithm 1)

这是 paper 最 novel 的部分。Auxiliary head 预测 per-voxel velocity $\mathcal{F}_t \in \mathbb{R}^{X\times Y\times Z\times 3}$ (world coordinate 下的 3D velocity vector)。

Warp 过程：
```
1. F_t = flow_decoder(V_t)  // 预测绝对 velocity per voxel
2. F̂_{t+n} = Δt · n · F_t   // 乘以时间间隔得到位移
3. F̃_{t+n} = M · F̂_{t+n}    // 用 pose matrix 转换到 future frame ego coordinate
4. V_{t+n} = grid_sample(V_t, F̃_{t+n})  // warp 当前 voxel features
```

- $\Delta t = 1/12$ 秒 (nuScenes 12Hz)
- $n$: 预测未来第几帧
- $M$: current → future ego coordinate 的 pose transformation

然后用 warped voxels 通过 3D-GS decoder render future frame 的 multi-view images，用真实 future frame RGB 做 L1 supervision。

**关键设计：backprop 时只更新 velocity head 的参数，不更新 backbone**。这是个很有意思的 choice——它的目的可能是让 velocity head 学到 motion-specific features 而不污染 backbone (backbone 已经被 L_img 和 L_pc 优化了)。这一点 paper 写得不够清楚，我倾向于认为这是一个 "gradient routing" 策略，避免 velocity head 的 noise 干扰 geometry learning。

直觉：**这一步本质上是一个 world model 的简化版**——给定当前 voxel state，预测下一帧 voxel state (通过 velocity 推演)，然后 render 下一帧 image。这跟 ViDAR [5] 的 next-frame LiDAR prediction 是同一思路，但用 velocity-as-state 替代直接 next-frame prediction。Velocity 这种 parameterization 的好处是：static voxel velocity = 0, dynamic voxel velocity ≠ 0，网络自动学到 dynamic-static decomposition，这对下游 detection 的 moving object 识别直接有用。

### 4.5 Photometric consistency (Eq. 5, 6)

$$\mathbf{I}_{t'\to t} = \langle\mathbf{I}_{t'}\rangle(\text{proj}(\mathbf{D}_t, \mathbf{T}_{t\to t'}, \mathbf{K}))$$

- $\mathbf{I}_{t'}$: source frame (相邻帧) 的 RGB
- $\mathbf{D}_t$: 当前帧 rendered depth (从 3D-GS decoder 来)
- $\mathbf{T}_{t\to t'}$: current → source 的 relative pose
- $\mathbf{K}$: camera intrinsics
- $\text{proj}(\cdot)$: 把 3D 点 (depth + pixel coord) 投影到 source frame 的 2D 坐标
- $\langle\cdot\rangle$: differentiable grid sampling (bilinear)

直觉：用当前帧 depth 把每个 pixel 反投影回 3D，再用 relative pose 变换到 source frame 坐标系，再投影到 source frame 的 pixel，最后 grid sample source image 得到 re-projected image。如果 depth 准确 + pose 准确，re-projected image 应该和 current frame image 一致 (除了 occlusion 区域)。

Loss:
$$\mathcal{L}_{pc} = \alpha(1 - \text{SSIM}(\mathbf{I}_t, \mathbf{I}_{t'\to t})) + (1-\alpha)\|\mathbf{I}_t - \mathbf{I}_{t'\to t}\|_1$$

- $\alpha$: 平衡 SSIM 和 L1 的权重 (Monodepth2 用 0.85)
- SSIM: structural similarity, 关注 patch-level 结构
- L1: pixel-level 强度

这是 Monodepth2 [7] 的经典 self-supervised depth loss。VisionPAD 把它移植到 pre-training 阶段，关键是**用 3D-GS rendered depth 而不是 LiDAR depth**——这就让整个 pipeline 不依赖 LiDAR。

直觉：photometric consistency 给了网络一个 "如果 depth 错了，warp 后的 image 就糊了" 的直接 signal，强迫 depth 预测准确。这是 self-supervised depth estimation 几十年的核心 insight，VisionPAD 的贡献是把它 wrap 进一个 end-to-end 可微的 3D-GS pipeline 里。

### 4.6 Total loss (Eq. 8)

$$\mathcal{L} = \omega_1 \mathcal{L}_{img} + \omega_2 \mathcal{L}_{vel} + \omega_3 \mathcal{L}_{pc}$$

- $\omega_1 = 0.5$ (current frame RGB reconstruction)
- $\omega_2 = 1.0$ (velocity warp 后的 future frame RGB reconstruction)
- $\omega_3 = 1.0$ (photometric consistency)

直觉：$\omega_2 = \omega_3 = 1$ 表明 paper 把 motion learning 和 geometric consistency 视为同等重要；$\omega_1 = 0.5$ 略低可能是因为 current frame reconstruction 本身相对简单 (信息没丢失)，而 future frame 和 cross-frame consistency 才是 harder signal。

## 5. 实验结果深度解读

### 5.1 3D Object Detection (Table 1)

最 striking 的对比：
- UVTR baseline: 45.0 NDS / 37.2 mAP (no pre-train)
- UVTR + UniPAD (image only, no LiDAR depth): 44.8 NDS ↓0.2 / 38.5 mAP ↑1.3
- UVTR + UniPAD (C+L, with LiDAR depth): 50.2 NDS ↑1.4 / 42.8 mAP ↑3.6
- UVTR + VisionPAD (image only): 49.7 NDS ↑0.9 / 41.2 mAP ↑2.0
- UVTR + VisionPAD (C+L): 50.4 NDS ↑1.6 / 43.1 mAP ↑3.9

**这里有一个非常重要的 insight**：UniPAD 在纯 image supervision 下基本没学到东西 (NDS 反而下降)，必须加 LiDAR depth supervision 才有用。但 VisionPAD 在纯 image supervision 下就接近 UniPAD + LiDAR 的水平。这说明 **3D-GS + velocity + photometric consistency 这套组合，让 image supervision 的 signal efficiency 提升了一个量级**。

CS (camera sweeps, two historical frames) 输入下，VisionPAD 达到 49.7 NDS / 41.2 mAP，**no TTA, no ensemble, no LiDAR**，这是当时 SOTA 级别。

### 5.2 Semantic Occupancy (Table 2)

- UVTR baseline: 30.1 mIoU
- UVTR + UniPAD (image only): 31.0 mIoU ↑0.9
- UVTR + VisionPAD: 35.4 mIoU ↑5.4
- BEVDet-Occ (8 frames): 39.3 mIoU
- BEVDet-Occ + VisionPAD: 42.0 mIoU ↑2.7

Occupancy 是 dense prediction，对 geometry 极敏感，VisionPAD +5.4 mIoU 这个 gain 非常大，证明 3D-GS 学到的 voxel geometry 是真有信息量的。

### 5.3 Map Segmentation (Table 3)

- UVTR: 15.0 IoU
- UVTR + UniPAD: 16.3 ↑1.3
- UVTR + VisionPAD: 20.4 ↑5.4

Map segmentation 也大幅提升。这表明 VisionPAD 学到的 representation 不只对 3D 任务有用，对 BEV 2D 任务也有效——representation 的 generalization 很强。

### 5.4 Data Efficiency (Fig. 4)

这是一个 Karpathy 你应该特别关注的图。当 fine-tuning 用 25% 数据时：
- VisionPAD vs UniPAD: mAP 提升 ~6 个点

**这个 scaling behavior 说明 VisionPAD 学到的 representation 信息密度高**。在 label-scarce regime 下 (这是 real-world autonomous driving 的常态)，好的 pre-training 价值更大。这跟你在 Tesla 时反复强调的 "data is the bottleneck" 的论点完全契合。

### 5.5 Ablation (Table 4)

从 baseline (22.8 NDS / 19.4 mAP) 一路加组件：
- + 3DGS Decoder (Model A): 22.8 NDS / 18.2 mAP (mAP 反而掉了 1.2！)
- + Gaussian Filter (Model B): 23.4 / 18.9
- + V.V. Est (Model C): 23.6 / 20.1
- + P.C. (Model D): 26.0 / 24.5 (这里 P.C. 单独加，提升巨大)
- + V.V. Est + P.C. (VisionPAD): 27.3 / 26.5

**注意 Model A**：单独换 3DGS decoder 反而掉了 mAP。这说明 3DGS 本身不够，必须有 velocity 和 photometric consistency 才能让 image supervision 真正学到 geometry。这是一个非常重要的 negative result，paper 没足够强调。

**P.C. 是最大的 single-component gain** (+2.4 NDS / +4.4 mAP from Model C to Model D)。这印证了我前面的分析：photometric consistency 是 image-only self-supervised geometry learning 的核心 signal。

## 6. Karpathy 视角的一些深层思考

### 6.1 这本质上是 world model 的简化形态

VisionPAD 的 voxel velocity estimation 其实是 latent world model 的一个特例：
- State: voxel features V_t
- Transition: V_{t+n} = warp(V_t, velocity × Δt × n)
- Observation: render(V) → image
- Loss: observation prediction error

这跟 Sora [12]、Genie [13]、DayDreamer [14] 的 world model 思路是相通的，只是 VisionPAD 用了 explicit 3D representation (voxels + Gaussians) 而不是 latent tokens。Explicit representation 的好处是 interpretability 和 sample efficiency (人类先验直接 encode 进 representation structure)。

Ref:
- Sora technical report: https://openai.com/research/video-generation-models-as-world-simulators
- Genie: https://arxiv.org/abs/2402.15391

### 6.2 Self-supervised learning 的 inductive bias 讨论

VisionPAD 用了三个 strong inductive biases：
1. **3D structure prior**: voxel grid + 3D Gaussians 直接 encode 3D scene structure
2. **Multi-view consistency**: 同一 3D 点在不同视角下 color 一致
3. **Temporal consistency**: 同一 object 在相邻帧 velocity 平滑

这三个 prior 都是 geometric / physical 的，不是 learned。这跟 MAE 的 "natural image patch naturalness" prior 不同，更接近 physics-informed neural network 的思路。

我倾向于认为对于 autonomous driving 这种 structured physical environment，**显式的 geometric prior 比 learned prior 更 sample efficient**，这也是为什么 VisionPAD 能用纯 image supervision 就 beat UniPAD + LiDAR。

### 6.3 与 BEVWorld / OccWorld 的关系

最近的 BEVWorld [15] 和 OccWorld [16] 走的是 "learn a generative world model in BEV space" 的路线，用 diffusion 或 autoregressive transformer 预测 future BEV tokens。VisionPAD 的 voxel velocity warp 可以看作一个 "linear dynamics" world model——假设 velocity 恒定，下一帧 voxel state 由当前 state + velocity × Δt 给出。

更复杂的 world model (OccWorld 用 transformer 学 transition) 能捕捉 non-linear dynamics (e.g. 转弯、加速)，但需要更多数据。VisionPAD 的简化方案在 pre-training 阶段够用，因为 goal 是 learn representation 不是 accurate future prediction。

Ref:
- BEVWorld: https://arxiv.org/abs/2407.05679
- OccWorld: https://arxiv.org/abs/2312.03741

### 6.4 Compute efficiency 的 scaling 含义

Table 6 的 13× memory reduction + 13× latency reduction 是巨大的。这意味着：
- 同样的 GPU budget，可以用更长 temporal window (e.g. 4 frames → 16 frames)
- 同样的 GPU budget，可以用更高 resolution (e.g. 1600×900 → 1920×1080)
- 同样的 GPU budget，可以用更大 batch size (更稳定的 gradient)

这三者都是 self-supervised pre-training scaling 的关键维度。Karpathy 你应该最懂这个——pre-training scaling laws 对这些 hyper-parameter 极度敏感。

### 6.5 Limitations (paper 没说但你应该问的)

1. **Velocity 假设恒定**：对匀速运动 OK，对加减速 / 转弯场景不准。一个改进方向是学 acceleration 或直接学 non-linear transition function。
2. **Photometric consistency 假设 Lambertian**：玻璃、金属、镜面反射物体违反这个假设，会造成 depth noise。
3. **Occlusion 处理**：re-projection 时被遮挡的区域 photometric loss 会失效，paper 没说怎么 handle (Monodepth2 用 auto-masking 但这里没提)。
4. **Dynamic object 在 photometric consistency 里是 confounder**：相邻帧的 moving car 会被 warp 错位置，产生 wrong depth signal。Paper 用 velocity estimation 区分 static/dynamic，但没明确说 photometric loss 只对 static voxel 计算——这可能是 implicit 的 (dynamic voxel 的 opacity 低，自动被 down-weight)。

### 6.6 与 Occupancy-as-World-Model 的方向

最近 OccWorld [16], Cam4DOcc [17], ViDAR [5] 都在往 "occupancy prediction as world model" 方向走。VisionPAD 可以看作这个方向的 pre-training 版本——它的 voxel velocity warp 是 occupancy world model 的简化 transition，加 3D-GS 是 observation model。这跟 RBM / VAE / world model 的 state-space formulation 完全对应。

如果顺着这个思路，下一步很自然的是：把 VisionPAD 的 linear velocity transition 换成 learned transition network (e.g. 一个 transformer 或 diffusion model)，把 3D-GS observation model 保留，那就是一个完整的 generative world model for autonomous driving pre-training。

Ref:
- Cam4DOcc: https://arxiv.org/abs/2309.13544

## 7. Implementation 细节 (你可能会问的)

- **Backbone**: ConvNeXt-S [18], 1600×900 input (跟 UniPAD 对齐)
- **Volume**: 180×180×5, range [-54m, 54m] × [-54m, 54m] × [-5m, 3m], 256 channels
- **Voxel size**: 0.6m (main) / 0.8m (ablation)
- **Pre-training**: 12 epochs, AdamW, lr=2e-4, batch_size=4
- **Augmentation**: random scale + rotation + partial input masking (size=32, ratio=0.3)
- **Anchors per voxel**: 1 (paper 测了 2/3/4 都不更好，因为更多 Gaussian 会增加 compute 但 representation 没显著提升)

Voxel 通道数 256 是 standard BEVFormer 配置，paper 用 128 做 ablation 是为了省 compute。

## 8. 总结：VisionPAD 给我们的 intuition

1. **Renderer 选择对 self-supervised geometry learning 至关重要**：3D-GS 比 NeRF 适合 pre-training，因为 explicit + efficient + resolution-agnostic。
2. **Image-only supervision 可以学 geometry，但需要 multi-frame signal**：单帧 reconstruction 不够，必须有 cross-frame consistency (photometric + velocity)。
3. **Pre-training 是 world model 的简化版**：velocity warp + render 是 state transition + observation 的特例，linear 假设够用。
4. **Compute efficiency 决定 scaling**：3D-GS 的 13× efficiency gain 让 pre-training 能 scale 到更长 temporal、更高 resolution、更大 batch。
5. **Inductive bias 重要**：3D voxel + Gaussian 这种 explicit geometric prior 比 learned latent representation 在 sample-scarce regime 下更 efficient。

这篇 paper 的真正贡献是把 self-supervised depth estimation (Monodepth2 line) + 3D-GS (computer graphics line) + BEV perception (autonomous driving line) 三个领域的 insights 整合到一个统一 framework 里，并且证明了纯 image supervision 可以达到 LiDAR supervision 的水平。这对真正 scalable 的 vision-only autonomous driving 系统是很重要的 milestone。

---

Related papers 你可能想看：
- UniPAD: https://arxiv.org/abs/2312.13726
- ViDAR: https://arxiv.org/abs/2312.17655
- MIM4D: https://arxiv.org/abs/2403.08760
- OccWorld: https://arxiv.org/abs/2312.03741
- BEVWorld: https://arxiv.org/abs/2407.05679
- PixelSplat: https://arxiv.org/abs/2312.12375
- Monodepth2: https://arxiv.org/abs/1806.01260
- 3D-GS: https://repo.akm.tugraz.at/~kerbl/kerbl_gaussian_splatting.pdf
- GaussianFormer: https://arxiv.org/abs/2405.17429
- GaussianOcc: https://arxiv.org/abs/2408.11447
- Street Gaussians: https://arxiv.org/abs/2401.01339
- DrivingGaussian: https://arxiv.org/abs/2406.02507
- HUGS: https://arxiv.org/abs/2401.06897

如果你想深入某个具体方面（比如 3D-GS rasterization 的 CUDA 实现细节、photometric consistency 的 auto-masking 策略、velocity head 的 gradient routing 设计），可以再展开聊。
