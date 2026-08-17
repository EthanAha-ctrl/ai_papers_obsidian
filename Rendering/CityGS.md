---
source_pdf: CityGS.pdf
paper_sha256: c88d15a66aa8030d39e5bd9d924a50b4bda55bccc7ae664fd18c2d9a753d491b
processed_at: '2026-08-03T15:39:14-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 CityGS-X

## 这篇 paper 到底在解决什么问题？

一句话：**怎么用 3DGS 把一个城市级别（5K+ 张图）的场景重建出来，又快又准又不爆显存。**

听起来简单，实际是 3DGS 的噩梦。你去查 CityGaussian、VastGS 这些 prior work，它们的套路都是：

1. 把城市切成几个 block
2. 每个 block 丢给一个 GPU 训练
3. 训完把所有 block merge 到一张卡上做渲染

这个 pipeline 有三个死穴：

**死穴一：Merge 回单卡就 OOM。** 4090 只有 24GB 显存，一个城市搞下来轻松上千万个 Gaussian，merge 到一张卡直接 crash。所以之前的方法最多也就搞定几百万 Gaussian 的场景。

**死穴二：Block 边界处不连续。** 每个 block 独立训练，block A 和 block B 的边界处没有 multi-view constraint 约束，拼起来会有缝。你增大 overlap area 能缓解，但训练时间又上去了。

**死穴三：几何不准。** Vanilla 3DGS 的 anisotropic Gaussian 本质上是"体积"，不是 surface。重建出来的 mesh 满是 floaters、有洞、surface 噪声大。

CityGS-X 的目标就是一次性干掉这三个问题。

---

## CityGS-X 的核心 idea：把 merge-and-partition 干掉

之前的方法为什么必须 merge？因为每个 block 是显式存的 Gaussian attribute（position, scale, rotation, opacity, SH color，每个 Gaussian 要 59 个 float）。这种 explicit representation 没法跨 GPU 直接用——你在 GPU 0 上训练的 Gaussian，GPU 1 不知道。

CityGS-X 的 key insight：**换 representation**。

它不直接存 Gaussian 了，改存 **voxel + 一个 shared MLP decoder**。具体讲：

- 把 scene 用八叉树切成 K 层 LoD 的 voxel grid
- 每个 voxel 只存一个 32 维的 feature embedding + 几个 offset
- 真正的 Gaussian attribute 由一个 MLP decoder 从 voxel feature 解码出来
- 这个 MLP decoder **所有 GPU 共享**，gradient 通过 DDP all-reduce 同步

这样一来，**voxel 可以跨 GPU 分 shard 存**，但 decoder 是共享的。每个 GPU 只需要存自己负责的那部分 voxel embedding，通信只发生在 decoder 的 gradient 同步上（这个量很小）。

这就彻底干掉了 merge 步骤——根本不需要 merge，因为 Gaussian attribute 是 implicit 的，渲染时按需 decode 出来就行。

---

## Voxel 怎么跨 GPU 分配？

公式 (3) 看着唬人，其实超简单：**round-robin spatial sampling**。

想象 voxel list 按空间位置排序后，第 1 个给 GPU 1，第 2 个给 GPU 2，...，第 M+1 个又给 GPU 1。这样每个 GPU 拿到的 voxel 在空间上是均匀散开的。

为什么这么干？因为如果随机分配，可能 GPU 0 拿到一堆空 voxel（大场景里大部分 voxel 是空的），GPU 1 拿到全是密集区域的 voxel，GPU 0 训练 1 秒就完事等 GPU 1 等 10 秒。spatial average 保证负载均衡。

这个跟 Grendel-GS 的思路类似，但粒度从 Gaussian-level 提到 voxel-level。Voxel 数远少于 Gaussian 数，所以分配更稳定。

---

## 渲染怎么并行？

这是工程上最难的部分。流程：

1. 把待渲染的 image 切成 16×16 pixel 的小 patch
2. 这些 patch 通过 adaptive load balancing 分发到不同 GPU
3. **每个 patch 需要 render 哪些 Gaussian？** 跨所有 GPU、跨所有 LoD 做 voxel intersection search
4. 找到的 voxel 在各自 GPU 上 decode 成 Gaussian attribute
5. 把这些 Gaussian attribute transfer 到负责该 patch 的 GPU
6. 该 GPU 用经典 tile-based rasterization 渲染

关键 trade-off：transfer 的 Gaussian 数被 patch 大小（16×16）控制，所以通信量可控。

RGB / Depth / Normal 三个任务同时 render，公式 (5)(6)(7) 都是 alpha-blending 变体，只是 blending 的量不同：
- RGB blending 用 color
- Normal blending 把每个 Gaussian 的 normal 也加权融合
- Depth blending 用 ray-Gaussian plane 交点距离（PGSR 的 unbiased depth，避免 Gaussian 厚度造成的 bias）

---

## 三个 loss 怎么搭？

这是 paper 最 intuitive 的部分。

**Step 1 - Batch-Level RGB Training**：普通 3DGS 一次 render 一张图（batch=1），这里一次 render B 张图（实验里到 32）。Gradient 从多视角同时回传到 shared decoder，天然 regularize，避免 single-view overfitting。Loss 就是 L1。

**Step 2 - Enhanced Depth-Prior Training**：用 Depth Anything 估单目深度，但单目深度没有真实 scale 且多视角不一致。CityGS-X 干两件事：
- 用 least squares 把 mono depth 对齐到 SfM sparse point cloud 上恢复 scale
- 对每个 target view，找邻近 view 做 re-projection，re-projection error 大的区域 mask 掉（说明 mono depth 在那里不靠谱）

然后用 filtered 后的 depth 做 L1 supervision。这步让场景的大平面先对齐，提供 smooth initialization。

**Step 3 - Batch-Level Geometric Training**：把 batch 内的 image 两两配对，在每个 pair 里随机采 7×7 patch，用 homography warping 找到对应 patch，算 NCC（normalized cross correlation）作为 photometric loss。Stop gradient 防止 degenerate solution。

这三步是 progressive 的：先用 RGB 学 appearance，再加 depth prior 把 geometry 拉个粗略形状，最后用 photometric constraint refine 细节。Ablation 里 Step 2 直接跳过去用 Step 3，PSNR 反而降——证明 depth prior 提供的 smooth initialization 是必要的。

---

## 结果到底多好？

看 Table 2 的 MatrixCity（5000+ 图）：

| Method | PSNR | F1 |
|---|---|---|
| NeuS | 16.76 | FAIL |
| 2DGS | 21.35 | 0.270 |
| CityGS-V2 | 27.23 | 0.556 |
| **CityGS-X** | **27.58** | **0.581** |

Recall 从 0.752 提到 0.840，意味着 mesh 上的洞少了。你看 Fig. 5 的 mesh 对比就很直观：CityGS-V2 在 Residence 底部和 Rubble 地面有明显空洞，CityGS-X 没有。

资源消耗（Table 5）：

| Method | Building 时间 | Building 显存 |
|---|---|---|
| Mega-NeRF | 19:49 | 5.84 GB |
| 3DGS | 21:37 | 4.62 GB |
| CityGS-X | **03:00** | **2.00 GB** |

5000+ 图用 4×4090 五小时训完，单卡显存 2GB。其他方法在 4K 渲染时直接 OOM，CityGS-X 还能跑。

---

## 为什么这套设计 work？三个底层 reason

**Reason 1：Representation 和分布式通信 co-design。**

Vanilla 3DGS 直接 DDP 会爆——所有 Gaussian attribute 都要 all-reduce。CityGS-X 把 Gaussian attribute 隐式化进 shared decoder，每个 GPU 只存 voxel embedding（32 维），通信量极小。这是 representation 层面就为 distributed 训练铺好路。

**Reason 2：Multi-view constraint 在 optimization 层面 natural 出现。**

之前 partition-merge 的 multi-view constraint 是 block 间缺失的，只能 post-merge 时用 overlap area 弥补。CityGS-X 的 batch-level training 让同一 batch 内不同 view 的 gradient 同时作用到 shared decoder 和分布式 voxel feature，multi-view constraint 是 implicit 出来的，不用 post-processing。

**Reason 3：Geometry prior 做 initialization basin。**

Step 2 depth prior 给一个粗略几何，Step 3 photometric constraint 才能在这个 basin 里 refine。这是 curriculum learning 的思路——直接上 photometric constraint 没有粗略初始化会发散。Ablation ID 5（跳过 Step 2 直接 Step 3）PSNR 22.10，ID 4（三步全用）22.76，证明这点。

---

## 这篇 paper 的 limitation

**Limitation 1：Progressive training schedule 是手工的。** Step 2 在 10k 步引入，Step 3 在 30k 步引入，这些是固定数。不同场景需要的 transition point 可能不同。没有自适应机制。

**Limitation 2：View-Dependent Gaussian Transfer 通信开销没分析。** Patch 数 × 每 patch Gaussian 数的 transfer 量在 16+ GPU 规模下可能成为瓶颈，但 paper 没给 communication time breakdown。

**Limitation 3：Depth prior filter 阈值 $\tau_d = 1$ 没给 sensitivity analysis。** 这个阈值对场景依赖很强，MatrixCity（几何简单）和 Rubble（废墟复杂几何）的合理阈值可能差很多。

**Limitation 4：没 report 4K 渲染的 FPS。** 训练能跑不代表 inference 顺畅，没给 throughput 数据。

**Limitation 5：Homography-based photometric loss 的 7×7 patch 偏小。** 经典 MVS 方法用 11×11 或更大，7×7 对噪声敏感。

---

## 一句话总结

CityGS-X 的核心 trick：**把 3DGS 的 explicit Gaussian 换成 voxel + shared MLP decoder 的 implicit 表示，从而天然 fit DDP 分布式训练，干掉 merge 步骤；再用 batch-level multi-view training 让 multi-view constraint 在 optimization 层面自动出现，配 progressive depth prior + photometric loss 把几何做准。**

不是单点创新，是 system-level 的工程整合：Scaffold-GS 的 hybrid representation + Octree-GS 的 LoD + Grendel-GS 的 distributed + PGSR 的 geometry + Depth Anything 的 prior，全都串起来做 city-scale scene。如果开源且工程实现扎实，会成为 3DGS large-scale reconstruction 的新 baseline。

Project page: https://lifuguan.github.io/CityGS-X/

你如果还想 dig 哪个 module 的实现细节，比如 Gaussian Transfer 的 PCIe 通信模式、enhanced depth filter 的具体算法、或者 voxel intersection search 的并行实现，可以继续问。

---

# CityGS-X: Large-Scale Scene Reconstruction 的 Scalable Architecture 深度解读

## 一、这篇 Paper 要解决的核心 Pain Point

Karpathy 你应该对 3DGS 的 scalability 问题很敏感。当前 large-scale scene reconstruction 落到 3DGS 上的三大死结：

1. **Partition-and-Merge paradigm 的 inherent flaw**：传统方法（VastGS, CityGaussian, CityGS-V2）将 scene 切成 block 用多 GPU 训练，然后 merge 回 single GPU 做渲染。block 之间缺乏 multi-view constraint，导致 boundary inconsistency（Fig. 2a）。增大 overlap area 或者 post-merge fine-tune 又要额外时间成本。
2. **Single-GPU rendering ceiling**：RTX 4090 上最多跑 ~5M Gaussians，scene 一大就 OOM。即使做 pruning 也撑不住 city-scale scene（5K+ images）。
3. **Geometry accuracy trade-off**：原始 3DGS 的 anisotropic Gaussian 无法很好捕捉 surface，导致 floaters 和 noisy mesh。PGSR/2DGS 改善了，但 large-scale 下计算代价太高。

CityGS-X 的核心 claim：用一个 **DDP-like（Distributed Data Parallel）的 PH²-3D 表示**彻底替换 merge-and-partition，并行训练 + 并行渲染，同时 batch-level 多视角约束修复 geometry。5000+ images 用 4×4090 仅需 5 小时，且能跑 4K 渲染。

---

## 二、PH²-3D Architecture 架构解析

### 2.1 Hierarchical Hybrid Representation 的底层逻辑

回顾一下 Scaffold-GS [9] 和 Octree-GS [8] 的设计动机：
- **Scaffold-GS**：用 anchor voxel + MLP decoder 生成 Gaussians，比 vanilla 3DGS 省内存但缺少 LoD 机制
- **Octree-GS**：在 Scaffold-GS 基础上加 octree 结构做 LoD 渲染

CityGS-X 把这两个 idea 串起来并分布式化。核心公式 (2)：

$$\mathbf{X}_k = \left\{ \left\lfloor \frac{\mathbf{P}}{\delta / 2^k} \right\rfloor \cdot \delta / 2^k \right\}, \quad 0 \leq k \leq K-1$$

变量解读：
- $\mathbf{P}$：SfM 初始化得到的 sparse point cloud（ COLMAP 输出的 3D 点）
- $\delta$：第 0 层 LoD 的 voxel 初始尺寸（最粗糙层）
- $k$：LoD 层级 index，$k=0$ 最粗糙，$k=K-1$ 最精细
- $2^k$：每升一层 voxel 尺寸减半（八叉树式细分）
- $\lfloor \cdot \rfloor$：floor operation，把 point P 量化到 voxel grid

Intuition：相当于在每个 LoD 层级上做 point-to-voxel hashing。同一个 3D 点在 $K$ 层的 voxel index 都被预计算好，渲染时根据 observation distance 动态选 $k$。

### 2.2 Cross-GPU Voxel 分配：Spatial Average Sampling

公式 (3)：

$$\mathbf{X}_k^{(m)} = \{X_{k,i}^{(m)} \mid i = m + jM\}, \quad j \leq \frac{v}{M}, \quad m \in \{1, \cdots, M\}$$

变量：
- $M$：GPU 总数
- $m$：第 $m$ 个 GPU 的 index
- $v$：当前 LoD 层 voxel 总数
- $\mathbf{X}_k^{(m)}$：分配给第 $m$ 个 GPU 的 voxel 子集

这是一个 **stride-$M$ 的 spatially uniform sampling**。本质上是把 voxel list 在空间上重排后做 round-robin 分配。为什么用 spatial average 而不是 random？因为如果随机分配，某个 GPU 可能拿到全是空 voxel（大场景中 voxel 稀疏度高），导致某些 GPU 空转等同步。spatial average 保证每个 GPU 拿到的 voxel 在空间上均匀分布，从而 view-dependent rendering 时激活的 voxel 数大致相等。

这让我联想到 Grendel-GS [17] 中的 load balancing 思路，但 Grendel-GS 主要处理 Gaussian-level 分布，CityGS-X 处理 voxel-level 分布，粒度更粗更稳定。

### 2.3 Shared Gaussian Decoder 的 DDP 机制

公式 (4)：

$$\{\mu_n^{(m)}\} = \{x_v^{(m)}\} + O_n^{(m)} \cdot l_v^{(m)}$$

$$\{\{\alpha_n^{(m)}\}, \{c_n^{(m)}\}, \{\Sigma_n^{(m)}\}\} = \mathrm{F}_{de}(F_v^{(m)}, \Delta_{vc}, \tilde{\mathrm{d}}_{vc})$$

变量：
- $x_v^{(m)}$：voxel center position（在 m 号 GPU 上的第 v 个 voxel）
- $O_n^{(m)} \in \mathbb{R}^{n \times 3}$：n 个 learnable offset（类似 Scaffold-GS 的 anchor offset）
- $l_v^{(m)} \in \mathbb{R}^3$：scaling factor，约束 offset 在 voxel 局部范围内
- $F_v^{(m)} \in \mathbb{R}^{32}$：voxel 的 learnable feature embedding
- $\Delta_{vc}$：view-dependent relative viewing distance（voxel-to-camera）
- $\tilde{\mathrm{d}}_{vc}$：voxel-to-camera direction（unit vector）
- $\mathrm{F}_{de}(\cdot)$：shared Gaussian Decoder（一个小 MLP）

Intuition 解读：
- $\mathrm{F}_{de}$ 是 **所有 GPU 共享的**（DDP 风格），只需要在 backward 时 all-reduce gradient
- 每个 GPU 持有自己的 voxel embedding $F_v$、offset $O_v$、scaling $l_v$，这部分 state 是 sharded 的
- View-dependent input $(\Delta_{vc}, \tilde{\mathrm{d}}_{vc})$ 让 decoder 输出 view-adaptive Gaussian，类似 Scaffold-GS 的 view-aware 设计

这比 vanilla 3DGS 显式存所有 Gaussian 属性省内存太多：3DGS 一个 Gaussian 存 59 float（position 3 + scale 3 + rotation 4 + color 3 + opacity 1 + SH 45），这里每个 voxel 只存 32 + 3 + 3n + 3 ≈ 38 + 3n，而 decoder 参数全 GPU 共享。

---

## 三、Batch-Level Multi-Task Rendering 细节

### 3.1 View-Dependent Gaussian Transfer

这是 CityGS-X 的关键 engineering 点。流程：
1. 把待渲染的 batch image 切成 16×16 pixel patch
2. 用 Grendel-GS 风格的 adaptive load balancing 把 patch 分发到不同 GPU
3. 每个 patch 所在 GPU 需要哪些 Gaussians？跨所有 LoD 跨所有 GPU **parallel voxel intersection search**
4. 找到的 voxel 通过公式 (4) decode 成 Gaussians
5. 把这些 Gaussians **transfer 到负责渲染该 patch 的 GPU**
6. 经典 tile-based rasterization 渲染

关键 trade-off：transfer 的 Gaussians 数量受 patch 大小控制（16×16），所以通信量可控。这是 Grendel-GS 的核心 trick，CityGS-X 把它从 Gaussian-level 提到 voxel-level，因为 voxel 数远小于 Gaussian 数。

### 3.2 RGB/Depth/Normal 三任务渲染

公式 (5)、(6)、(7) 都是 alpha-blending 变体。

公式 (5) - RGB：

$$\hat{\pmb{C}} = \sum_{i=1}^{N} \pmb{c}_{\pi(i)} \alpha_{\pi(i)} \prod_{j=1}^{i-1}(1 - \alpha_{\pi(j)})$$

- $N$：transferred 到当前 patch 的 Gaussian 数
- $\pi(\cdot)$：按 depth 排序的 reorder function（front-to-back）
- $\pmb{c}_{\pi(i)}$：第 i 个 Gaussian（排序后）的 color
- $\alpha_{\pi(i)}$：opacity × 2D projection of covariance

公式 (6) - Normal（来自 PGSR [51]）：

$$\hat{N} = \sum_{i=1}^{N} R_c^T n_{\pi(i)} \alpha_{\pi(i)} \prod_{j=1}^{i-1}(1-\alpha_{\pi(j)})$$

- $R_c^T$：camera-to-world rotation matrix 的转置，把 world-frame normal 变到 camera frame
- $n_{\pi(i)}$：第 i 个 Gaussian 的 normal（PGSR 用 Gaussian 的 shortest axis 作为 normal 方向）

公式 (7) - Unbiased Depth（来自 PGSR [51]）：

$$D = \sum_{i=1}^{N} d_{\pi(i)} \alpha_{\pi(i)} \prod_{j=1}^{i-1}(1-\alpha_{\pi(j)})$$

$$\hat{D} = \frac{D}{\hat{N} K^{-1} \tilde{\pmb{p}}}$$

- $d_{\pi(i)}$：ray-Gaussian plane intersection distance（不是 Gaussian center 距离）
- $\tilde{\pmb{p}}$：pixel homogeneous coordinate $[u, v, 1]^T$
- $K^{-1}$：camera intrinsics 的逆
- $\hat{N} K^{-1} \tilde{\pmb{p}}$：normal-weighted ray direction normalization

Intuition：PGSR 的 unbiased depth 公式把 alpha-blended expected depth 除以 normal-weighted ray length，避免 Gaussian 厚度造成的 depth bias。CityGS-X 直接复用这个 trick，把它 batch 化。

---

## 四、Progressive Training 三步走

### 4.1 Step 1: Batch-Level RGB Training

公式 (8)：

$$\mathcal{L}_{bl-rgb} = \frac{1}{B}\sum_{b=1}^{B}\|\hat{I}_b - I_b\|_1$$

- $B$：batch size（实验中 up to 32）
- $\hat{I}_b, I_b$：rendered vs GT image

关键 insight：传统 3DGS batch size = 1（一次只渲染一个 view），shared Gaussian Decoder 在 batch=B 下训练，gradient 来自多视角聚合，类似 batch normalization 的效果，缓解 single-view overfitting。

Karpathy 你应该会注意到这个 design choice 跟你的 nanoGPT / minGPT 教程里讲 large batch 训练的 motivation 是一脉相承的——multi-view gradient aggregation 提供 natural regularization。

### 4.2 Step 2: Enhanced Depth-Prior Training

这是 paper 最有想法的部分。先用 Depth Anything [6] 出 monocular depth $D_p$，但 mono depth 缺乏 absolute scale 且多视角不一致。CityGS-X 用 **least squares 对齐** sparse SfM point cloud $P$ 来恢复 scale 和 shift：

$$\tilde{D}_p = s \cdot D_p + t$$

其中 $s, t$ 通过最小化 $\sum_{p \in P}\|s D_p(p) + t - D_{SfM}(p)\|^2$ 求得。

然后 **Enhanced Depth-Prior Regularization**：对每个 target view，找 nearby views 做 re-projection，计算 re-projection error $E$，超过阈值 $\tau_d = 1$ 的区域 mask 掉（Fig. 4 中黑色区域）。

公式 (9)：

$$\mathcal{L}_{e-depth} = \frac{1}{B}\sum_{b=1}^{B}\|\hat{D_b} - \tilde{D}_{f,b}\|_1$$

- $\tilde{D}_{f,b}$：filter 后的 enhanced pseudo depth
- $\hat{D_b}$：渲染的 unbiased depth

Intuition：mono depth estimator 在大平面、远处天空表现很好，但在 thin structure、repetitive texture（如玻璃幕墙）会失败。filter 掉 high re-projection error 区域等于 "trust estimator only where multi-view geometry agrees"。

这让我想起 RobustNeRF、NeRF-W 的 robustness design，但 CityGS-X 的 filtering 是 batch-level 的，比 per-view 更稳定。

### 4.3 Step 3: Batch-Level Geometric Training

公式 (10)：

$$\mathcal{L}_{bl-geo} = \frac{1}{B/2}\sum_{i=0}^{B/2}\Big(1 - \mathrm{NCC}(\hat{C}_{2i}(\pmb{p}_{2i}), \mathrm{sg}[\hat{C}_{2i-1}(\pmb{H}_{i,2i-1}\pmb{p}_{2i})])\Big)$$

变量：
- batch 被分成 $B/2$ 个 image pair
- $\pmb{p}_{2i}$：在 image $2i$ 上随机采样的 7×7 patch center
- $\pmb{H}_{i,2i-1}$：image $2i$ 到 image $2i-1$ 的 homography matrix（由 depth 和 camera pose 算出）
- $\mathrm{NCC}(\cdot)$：normalized cross correlation [89]，对 intensity scale 和 shift 不变的 patch 相似度
- $\mathrm{sg}[\cdot]$：stop gradient（只通过 image $2i$ 的 patch 反传，image $2i-1$ 的 patch 作为 target）

Intuition：这是 multi-view photometric consistency 的经典套路（参考 MVSNet, DeepVideoMVS），但在 3DGS 训练里很少见。NCC 比 L1/L2 对 lighting variation 鲁棒。Stop gradient 防止两个 patch 互相 pull 导致 degenerate solution。

---

## 五、实验数据解读

### 5.1 Novel View Synthesis (Table 1)

四个 scene（Building, Rubble, Residence, Sci-Art）的 PSNR/SSIM/LPIPS。CityGS-X 在 w/ Geometric Optimization 组里全面 SOTA：

| Scene | CityGS-X PSNR | 第二名 | Δ |
|---|---|---|---|
| Building | 22.76 | SuGaR 17.76 / PGSR 16.12 | +5.00 / +6.64 |
| Rubble | 26.15 | PGSR 23.09 | +3.06 |
| Residence | 22.44 | PGSR 20.57 | +1.87 |
| Sci-Art | 22.77 | PGSR 19.72 | +3.05 |

跟 w/o Geometry 组比，CityGS-X 在 Rubble 上 +0.2dB over Momentum-GS（25.93），Sci-Art 上 LPIPS 0.179 vs Momentum-GS 0.205。说明 geometry 约束不损失 appearance quality，反而提升（因为 depth prior 帮助 disambiguate texture）。

### 5.2 Surface Reconstruction (Table 2 - MatrixCity)

| Method | PSNR | P | R | F1 |
|---|---|---|---|---|
| NeuS | 16.76 | FAIL | FAIL | FAIL |
| Neuralangelo | 19.22 | 0.080 | 0.083 | 0.081 |
| SuGaR | OOM | - | - | - |
| GOF | 17.42 | FAIL | FAIL | FAIL |
| 2DGS | 21.35 | 0.207 | 0.390 | 0.270 |
| CityGS | 27.46 | 0.362 | 0.637 | 0.462 |
| CityGS-V2 | 27.23 | 0.441 | 0.752 | 0.556 |
| **CityGS-X** | **27.58** | **0.444** | **0.840** | **0.581** |

Recall (R) 0.840 比 CityGS-V2 的 0.752 高 11.7%，说明 CityGS-X 重建出的 surface 覆盖更全（fewer holes），这个在 Fig. 5 的 mesh 对比里也能直观看到——Residence 底部和 Rubble 地面 CityGS-V2 有明显空洞。

### 5.3 Batch Size / GPU Number Ablation (Table 3)

| B.S./GPU | PSNR | SSIM | LPIPS | Storage (MB) | Mem (GB) |
|---|---|---|---|---|---|
| 2/2 | 25.58 | 0.787 | 0.244 | 1540×2 | 14.76 |
| 4/4 | 25.70 | 0.812 | 0.234 | 775×4 | 14.31 |
| 8/4 | 26.15 | 0.823 | 0.210 | 788×4 | 20.16 |
| 16/8 | 26.25 | 0.825 | 0.210 | 379×8 | 17.57 |

观察：
- B.S. 16 vs B.S. 2：PSNR +0.67, LPIPS -0.034，证实 multi-view gradient aggregation 有效
- Storage 几乎恒定（~3GB total），说明 voxel 数固定，只是 decoder 被多 GPU 共享
- B.S./GPU = 8/4 时单 GPU mem 飙到 20.16 GB（接近 4090 的 24GB 上限），因为 per-GPU batch 大导致 Gaussian transfer 多

### 5.4 Progressive Training Ablation (Table 4)

| ID | Step1 | Step2 | Step3 | PSNR | Time |
|---|---|---|---|---|---|
| 1 | ✗ | ✗ | ✗ | 22.41 | 4h21m |
| 2 | ✓ | ✗ | ✗ | 23.24 | 1h48m |
| 3 | ✓ | ✓ | ✗ | 22.46 | 2h10m |
| 4 | ✓ | ✓ | ✓ | 22.76 | 3h04m |
| 5 | ✓ | ✗ | ✓ | 22.10 | 3h11m |

关键 insight：
- ID 2 vs ID 1：+0.83 dB PSNR, 时间反而 -2.5 小时。说明 batch RGB 训练比 single-view 训练不仅效果好还更快（多 GPU 充分利用）
- ID 3 vs ID 2：PSNR 反而降了 0.78。因为 depth prior 在初期可能过度 smooth 掉 fine detail
- ID 4 vs ID 5：Step 2 不能跳过。直接上 Step 3 几何约束，没有 depth prior 提供 smooth 初始化，PSNR 反而比 ID 2 低。这是 progressive training 的 evidence——depth prior 提供 "good initialization basin"，photometric constraint 才能 refine
- 时间上 ID 4 比 ID 5 还快 7 分钟，因为 depth prior 让 Step 3 更快收敛

### 5.5 Training Time / Memory (Table 5)

CityGS-X 在所有 scene 上都是 fastest + lowest memory：
- Building: 3:00 / 2.00 GB（vs Mega-NeRF 19:49 / 5.84 GB）
- Rubble: 2:15 / 2.29 GB
- Residence: 2:40 / 2.61 GB
- Sci-Art: 3:30 / 1.40 GB

4090 上 2GB 内存跑 5K+ image 的 scene，这是非常 impressive 的 memory efficiency，主要归功于 PH²-3D 把 Gaussian attribute 隐式化进 decoder。

---

## 六、与相关工作的联想对比

### 6.1 跟 Grendel-GS 的关系
Grendel-GS [17] 是 first distributed 3DGS system，但仍是 vanilla 3DGS 的 explicit Gaussian。CityGS-X 借用了 Grendel-GS 的 patch-level load balancing（公式 3 后的 adaptive load balancing）和 Gaussian transfer 机制，但把表示换成 hybrid hierarchical，进一步省内存。

Grendel-GS: https://arxiv.org/abs/2406.18533

### 6.2 跟 CityGaussian / CityGS-V2 的演进
- **CityGaussian** (ECCV 2024)：引入 block-wise training + global refinement
- **CityGS-V2**：加 PGSR-style geometry，但仍是 partition-merge
- **CityGS-X**：彻底去 partition-merge，用 DDP

CityGaussian: https://arxiv.org/abs/2404.01133  
CityGS-V2: https://arxiv.org/abs/2411.00771

### 6.3 跟 Octree-GS / Scaffold-GS 的关系
PH²-3D 直接脱胎于 Scaffold-GS 的 anchor + MLP decoder 思路，再用 Octree-GS 的 LoD 八叉树结构。但 Octree-GS 是 single-GPU 的，CityGS-X 把它分布式化。

Octree-GS: https://arxiv.org/abs/2403.17898  
Scaffold-GS: https://arxiv.org/abs/2311.14075

### 6.4 跟 PGSR / 2DGS 的 geometry 思路
CityGS-X 复用了 PGSR 的 normal rendering (公式 6) 和 unbiased depth (公式 7)。但 PGSR 是 single-scene 的，CityGS-X 把它 batch-level 化并加 enhanced depth filter。

PGSR: https://arxiv.org/abs/2406.06521  
2DGS: https://arxiv.org/abs/2403.17888

### 6.5 跟 Depth Anything 的整合
Depth Anything v1 (CVPR 2024) 输出 relative depth，CityGS-X 用 least squares 对齐到 SfM sparse cloud 恢复 scale。这是 mono depth + multi-view fusion 的常见 pipeline（参考 MiDaS + NeRF 的 RobustNeRF）。

Depth Anything: https://arxiv.org/abs/2401.10891

### 6.6 跟 Federated 3DGS 的对比
Fed3DGS [41] 用 federated learning 思路做分布式，但 communication cost 高且需要 model distillation。CityGS-X 用 DDP 直接同步 gradient，更简单高效。

Fed3DGS: https://arxiv.org/abs/2403.11460

### 6.7 跟 VastGS / DoGaussian 的 partition-merge 对比
VastGS [37] 用 decoupled appearance modeling + progressive data partitioning。DoGaussian [48] 用 Gaussian consensus 做 distributed training。两者都需要 post-merge，CityGS-X 不需要。

VastGS: https://arxiv.org/abs/2402.17427  
DoGaussian: https://arxiv.org/abs/2405.13943

---

## 七、Karpathy 你可能会问的几个 Critical Questions

### Q1: View-Dependent Gaussian Transfer 的通信开销瓶颈在哪里？

每次 patch rendering 需要 cross-GPU voxel intersection search + Gaussian attribute transfer。如果 batch=B，patch 数 = B × H × W / 256，比如 B=4, 1080p 图像 = 4 × 1920 × 1080 / 256 ≈ 32,400 patches。每个 patch 平均要 1k-10k Gaussians，PCIe 4.0 带宽 ~32 GB/s，每 step 通信开销可能 100ms+。这可能是 scaling 到 16+ GPU 时的瓶颈。Paper 没有给 communication time breakdown，是个遗憾。

### Q2: Shared Gaussian Decoder 会不会成为 optimization 死锁？

所有 GPU 共享 decoder，backward 时 all-reduce gradient。如果 voxel feature $F_v$ 在不同 GPU 上分布差异大，decoder 的 gradient 来自不同 GPU 的不同 voxel 集合，这可能造成 optimization 震荡。Paper 里 batch RGB training 的 +0.83 dB 增益可能正是 multi-view gradient aggregation 抵消了这种震荡。

### Q3: Depth prior filter 的阈值 $\tau_d = 1$ 怎么定？

Paper 没给 sensitivity analysis。这个阈值对场景依赖很强——建筑场景（大量平面）和自然场景（复杂几何）的合理阈值差异可能很大。Enhanced Depth-Prior 在 MatrixCity 表现好可能是因为场景几何简单，mono depth 估计准。在 Rubble 这种废墟场景可能需要更激进的 filter。

### Q4: Homography-based photometric loss (公式 10) 的 7×7 patch 是不是太小？

7×7 = 49 pixel，NCC 在这么小 patch 上对噪声敏感。MVS 经典方法（如 PatchMatchNet）通常用 11×11 或更大。CityGS-X 选 7×7 可能是为了计算效率，但可能损失 texture-rich 区域的 matching robustness。

### Q5: 4K rendering 的实际 throughput？

Supp. Fig. 10 显示 4K 训练用 8 GPU + B.S. 8，但没给 FPS 数据。如果 CityGS-X 真 wants 立刻用于 production，需要报告 4K@30FPS 的可行性。猜测应该在 10-20 FPS 范围（基于 4×4090 跑 5K image scene 5h 训练推算）。

---

## 八、Intuition 总结：为什么这套设计 work？

Karpathy 你喜欢 first-principles 思考，我尝试总结 CityGS-X work 的三个底层 reason：

1. **Representation-Computation Co-design**：PH²-3D 把 explicit Gaussian attribute 压缩进 implicit decoder，每个 GPU 只存 voxel embedding（32 维），decoder 全局共享。这天然 fit DDP paradigm，因为通信量只涉及 decoder gradient（小）+ transferred Gaussian attribute（patch 级别可控）。如果直接 DDP vanilla 3DGS，所有 Gaussian attribute 都要 all-reduce，通信量爆炸。

2. **Multi-View Constraint at Optimization Level**：传统 partition-merge 在 block 训练时各 block 独立，boundary 处没有 multi-view constraint。CityGS-X 的 batch-level training 让同一 batch 内不同 view 的 gradient 同时作用到 shared decoder 和分布式的 voxel feature，相当于 multi-view constraint 在 optimization 层面被隐式 enforce。这比 post-merge refinement 优雅得多。

3. **Geometry Prior as Initialization Basin**：Step 2 depth prior 提供一个 smooth、approximate 几何，让 Step 3 photometric constraint 不至于在 noisy 初始化上发散。这跟 curriculum learning / pretraining 的 motivation 同构——先学粗的再学细的。Ablation ID 5 vs ID 4 证实了这一点。

这套设计哲学其实可以推广到其他 large-scale neural field system：representation 要 fit 分布式系统的 communication pattern，optimization 要 fit 多视角约束的本质，prior 要 fit coarse-to-fine 的 optimization landscape。

---

## 九、可能延伸的研究方向

1. **Dynamic Scene Extension**：CityGS-X 假设 static scene。把它扩展到 dynamic large-scale（如 City street with moving cars）需要 4D version of PH²-3D，类似 4DGS [98] 的思路但 distributed 化。4DGS: https://arxiv.org/abs/2310.08528

2. **Gaussian Decoder Capacity Scaling**：当前 $\mathrm{F}_{de}$ 是个轻量 MLP，如果 scale 到 transformer-based decoder，会不会进一步提升 geometry 表达力？类似 GRM, LRM 思路。GRM: https://arxiv.org/abs/2403.11321

3. **End-to-End SLAM Integration**：CityGS-X 需要 SfM 预先算 pose 和 sparse cloud。如果跟 SPLATAM [94] / CG-SLAM [95] 这样的 online dense SLAM 集成，可以做 large-scale online reconstruction。SplaTAM: https://arxiv.org/abs/2312.02126

4. **Mono Depth Filter 的 Self-Supervised 替代**：现在用 Depth Anything + SfM scale alignment + re-projection error filter。如果用 self-supervised multi-view depth（如 MiDaS 风格），可以避免 SfM 依赖。

5. **Gaussian Pruning 在 PH²-3D 下的形式化**：vanilla 3DGS 有 dense pruning, LightGaussian [15] 等。CityGS-X 的 voxel-level representation 如何 prune？可以基于 voxel feature 的 information gain 来 rank。LightGaussian: https://arxiv.org/abs/2311.17245

---

## 十、Overall 评价

CityGS-X 是一个 **system-level 的 contribution**，不像纯算法 paper 有一个 single novel module。它的价值在于：
- 把 Scaffold-GS / Octree-GS 的 hybrid hierarchical representation 通过 DDP 机制 scale 到 multi-GPU
- 用 batch-level multi-task rendering 让 multi-view constraint 在 optimization 层面 natural 出现
- 用 enhanced depth prior + photometric constraint 两阶段 progressive refine geometry

Engineering 工程量大，insight 相对直接，但对 large-scale 3DGS 的 production 部署有实际价值。如果它开源且代码质量过关，会成为 CityGaussian 系列的 SOTA baseline。Code 在 https://lifuguan.github.io/CityGS-X/ （应该会 release）。

唯一让我个人觉得可以质疑的点：Progressive training 三步是手工设计 schedule，没有自适应机制。如果不同 scene 需要不同的 transition point，固定 step 数（10k, 30k, 50k）可能 suboptimal。一个 self-distillation 或 auto-curriculum 的版本会更 elegant。

希望这些拆解能帮到你 build intuition。如果你后续想 dig deeper 哪个 module（特别是 Gaussian Transfer 的通信机制或 enhanced depth filter 的实现细节），可以继续追问。
