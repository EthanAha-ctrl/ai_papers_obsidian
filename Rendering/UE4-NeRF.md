---
source_pdf: UE4-NeRF.pdf
paper_sha256: b5ec4a153f4ba91f4cc3f824e5154a2547988550ca32872a46f2acb4c5a261c7
processed_at: '2026-08-12T18:58:28-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版 UE4-NeRF

好，咱们抛开公式，用大白话把这 paper 的逻辑捋一遍。

---

## 一句话总结

这帮人干的事就一句话：**把 NeRF 重建出来的场景，"烤"成游戏引擎能直接跑的 mesh，然后在 UE4 里像玩游戏一样实时渲染和编辑。**

听起来简单，但其实 NeRF 的人憋了好几年都做不到。为什么？因为 NeRF 本质上是"每帧每个 pixel 都要 query 几百次神经网络"的 brute-force 方法，这种东西跟 real-time 天生犯冲。你要让游戏跑 60 FPS，你不能再 query 网络了，你得走 GPU 里那套几十年打磨好的 rasterization 管线——就是把三角形投到屏幕上、查 texture、着色。

所以核心问题就变成：**怎么把一个 implicit neural field，变成一堆三角形 + texture，还能在 GPU 上跑飞快。**

---

## NeRF 原版为什么慢

原始 NeRF 干的事其实很傻。你拿相机拍 100 张照片，它学一个 MLP $F_\theta(\mathbf{x}, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma)$，里面 $\mathbf{x}$ 是 3D 点位置，$\mathbf{d}$ 是 viewing direction，输出 RGB color $\mathbf{c}$ 和 density $\sigma$。

渲染一帧的时候，它对屏幕上每个 pixel 打一条 ray，沿 ray 上采 100~200 个点，每个点都要 forward 一次 MLP。1920×1080 的图，就是 200 万条 ray × 200 个采样点 = 4 亿次 MLP forward。一张图要几秒到几分钟。这玩意儿能 real-time 才怪。

于是大家开始想：能不能把这玩意儿**bake**一下？像烤蛋糕一样，把训练好的 network 拆开，存成能快速查询的东西？

---

## Mobile-NeRF 先走了一步，但卡壳了

Mobile-NeRF (https://arxiv.org/abs/2208.00277) 是第一个真正在手机上跑 NeRF 的工作。思路是：**用一堆三角形 mesh 表示场景，每个 mesh 上挂一个小 texture 存 feature vector，fragment shader 里跑一个小 MLP 把 feature 转成 color。**

这思路听起来很美好，但落地时有两个大坑：

**坑一：训练巨慢。** 它要 8 张 V100 训好几天才能训一个小 scene。因为它还在用原始的 positional encoding，而且 mesh 的优化从随机初始化开始，loss landscape 很崎岖。

**坑二：opacity 被二值化了。** 它为了让 mesh 要么"在"要么"不在"，把 opacity 硬性变成 0 或 1。结果你拍一个有玻璃、水、塑料棚的 outdoor 场景，那些半透明的东西全废了。要么显示成实心，要么干脆消失。

UE4-NeRF 这帮人一看：这两个坑都是可以绕过去的。

---

## UE4-NeRF 的三个关键 trick

### Trick 1：用 Instant-NGP 的 hash encoding 加速训练

Instant-NGP (https://nvlabs.github.io/instant-ngp/) 那篇 paper 已经证明，用一个 multi-resolution hash table 把位置 $\mathbf{x}$ 编码成高维 feature，比原始 NeRF 的 sinusoidal positional encoding 快一个数量级。UE4-NeRF 直接借过来用。

公式上就是：

$$\text{hash}(\mathbf{x}) = \bigoplus_{l=0}^{L-1} \text{trilinear\_interp}(\mathbf{x}, \text{grid}_l)$$

$L$ 是 level 数，每个 level 是一个不同分辨率的 voxel grid，用 hash table 存 feature。低 level 抓大尺度结构，高 level 抓细节。最后 concat 起来喂给 MLP。

这一招让训练时间从 Mobile-NeRF 的几天掉到几十分钟。

### Trick 2：Regular octahedron mesh 初始化

这个是真有意思。Mobile-NeRF 在每个 voxel cell 中心放一个 cube（6 个面），cube 的面只能 align 到 axis。你想想，一个倾斜的屋顶、一个山坡，用 axis-aligned cube 去贴，肯定会出现锯齿和缝隙。Mesh vertex 想优化都没法优化到正确位置，因为初始几何太受限。

UE4-NeRF 改用 **regular octahedron，20 个面**（8 外 + 12 内）。这玩意儿接近球形，从哪个方向看都对称，相当于给 mesh vertex 一个 rotation-invariant 的起点。Vertex 可以往任意方向 drift，去贴 power line、树冠、屋顶斜面，都不会卡死。

直觉上就是：**cube 是个固定姿势的盒子，octahedron 是个松散的球，球更容易被推到正确的形状。**

### Trick 3：两段式 photometric loss 把 opacity "挤"到少数 mesh 上

这是 paper 最聪明的地方。

你想，最终渲染时只保留 $\alpha > 0.3$ 的 mesh face（paper 统计只占 5%）。所以训练时就得想办法让 opacity 集中到少数 face 上，别弥散得到处都是。

它分两个阶段：

**前 10000 epoch（part 1）**：跟普通 NeRF 一样，$\|\hat{\mathbf{C}} - \mathbf{C}\|^2$，让网络先学个大概的几何和颜色。Opacity 想在哪儿就在哪儿。

**10000 epoch 之后（part 2）**：加一个 threshold $f$，opacity 低于 $f$ 的采样点直接 ignore；累积 opacity 超过 0.8 就提前停止 ray marching。这等于告诉网络："你给我把 opacity 收紧，别浪费在不重要的地方。"

阈值 $f$ 的 schedule 是：

$$f = \min\{0.3, \lfloor \frac{epoch - 10000}{10000}\rfloor^2 \times 0.012\}$$

平方增长，慢慢收紧，最高 0.3。这个值正好对应后面 export 时的阈值。

两段用 $\beta = 0.8^{\lfloor epoch/10000 \rfloor}$ 加权，$\beta$ 随 epoch 指数衰减，慢慢从 part1 切到 part2。这是 curriculum learning：先学容易的（稀疏 opacity），再学难的（塌缩到 surface）。

---

## 训练完后，怎么 bake 进 UE4

训练完，每个 block 有：network 权重 + 5 个 LOD 的 mesh。接下来三步：

**第一步：提取最终 mesh。** 不光从训练相机视角发射 ray，还从上方多个角度发射 parallel light。任何 triangle face 上所有 intersection 的 opacity 都 < 0.3，就 clip 掉。剩下 5% 的 mesh 就是最终场景。

**第二步：UV mapping。** 每个 triangle 的 vertex 映射到一个 32×32 的 texture 上。注意这 texture 不存 color，存的是 8 维 feature vector + 1 维 opacity。

**第三步：BC4 压缩。** 9 个 single-channel map 用 BC4 (https://docs.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-block-compression) 压缩，4×4 pixel block 压成 8 byte，省 2/3 显存。

---

## 渲染时怎么工作

这是整个 system 的"魔法"所在，但其实也简单：

1. **UE4 rasterizer** 拿 mesh，用传统 polygon pipeline rasterize 到屏幕，每个 pixel 得到 8D feature + 1D opacity
2. **Neural Shader**（HLSL fragment shader）在 GPU 上跑一个 3 层 MLP，输入 8D feature + 9D SH 编码的 viewing direction = 17D，输出 RGB
3. **Alpha dithering + TAA** 处理半透明

你看，这就是 Mobile-NeRF 那套思路，但工程上塞进了 UE4 这么个工业级渲染引擎。UE4 自带 TAA、自带 LOD 切换、自带 .obj/.fbx 导入、自带物理碰撞、自带第一人称控制器——这些 NeRF 的人做梦都想要的 feature，UE4 全免费送你。

所以 paper 才敢说 "users can use a broomstick to fly or drive a car within the rendered environment"。这是 NeRF 圈子第一次能在一个 reconstructed scene 里开开车玩玩。

---

## LOD 怎么解决 large-scale 的帧率

这里抄的是游戏圈几十年的 Level of Detail (LOD) 技术。同一个 block 训练 5 个 mesh 精度：128³、64³、32³、16³、8³。

- 观察者离得远（无人机飞 120 米高）→ 用粗 mesh（8³），三角面少，FPS 高
- 观察者离得近 → 用细 mesh（128³），细节多，FPS 低

关键设计：**所有 LOD 共享同一个 Encoder-Decoder network，但 coarse LOD 的 loss 只更新 mesh vertex，不回传到 network 权重。**

为什么？因为粗 mesh 表示不了 power line、表示不了树叶这种 thin structure，如果你让它监督 network，network 会被逼着学一个模糊的平均，反而毁掉 fine LOD 的细节。所以粗 LOD 只优化 vertex 位置，不动 network。这是个 stop-gradient trick。

---

## Pseudo-depth：免费的 supervision

Outdoor 场景 RGB-D 相机几乎没用（光照太强干扰、范围有限）。但 NeRF 训练本来就要 COLMAP (https://colmap.github.io/) 做 SFM 估计相机位姿，COLMAP 顺便就给你一个 sparse 3D point cloud。

UE4-NeRF 直接把这 sparse point cloud project 回每张训练图，拿到稀疏 depth map，作为 supervision。Loss 加一项 $\mathcal{L}_D$：

$$\mathcal{L} = \mathcal{L}_{rgb} + \mathcal{L}_D$$

这一招让收敛快很多，而且 outdoor 场景特别合适。Depth-supervised NeRF (https://arxiv.org/abs/2107.02791) 早证明过这个有效，UE4-NeRF 只是把 sparse COLMAP 点云拿来复用，不增加任何传感器依赖。

---

## Transient object：人、车直接 mask 掉

UAV 拍的场景里，行人、车辆是 dynamic object，会让不同视角看到不一致。NeRF-W (https://arxiv.org/abs/2008.02268) 当年给 transient object 单独学一个 head 解决。UE4-NeRF 更粗暴：直接用 Panoptic-DeepLab (https://arxiv.org/abs/1911.10164) 和 PSPNet (https://arxiv.org/abs/1612.01105) 把人、车 semantic segment 出来，训练时 mask 掉。

这 trade-off 合理。Large-scale aerial scene 里 dynamic object 就是噪声，不值得花 model capacity 学。

---

## 实验数据告诉你什么

我挑几个关键数字解读：

| 指标 | UE4-NeRF | Instant-NGP | Mobile-NeRF |
|------|----------|-------------|-------------|
| PSNR | **25.03** | 22.00 | 16.92 |
| 训练时间 | 40 min | 15 min | 2880 min |
| GPU 显存 | 3 GB | 32 GB | — |
| Real-time FPS (2K) | 55 | 11 | ✗ |

几个 takeaway：

**PSNR 25 vs 22 (Instant-NGP)**：3 dB gap 很大。来自三件事：pseudo-depth supervision、octahedron mesh 比 voxel grid 表示能力强、curriculum loss 让 opacity 后期 focus 在 surface 上而不是弥散。

**GPU 3 GB vs 32 GB (Instant-NGP)**：差一个数量级。因为 UE4-NeRF 已经 bake 进 texture atlas + mesh，inference 时几乎不存 network feature grid。Instant-NGP 在 inference 时还要 load 整个 hash table。

**训练 40 min vs 2880 min (Mobile-NeRF)**：1000× speedup。主要来自 hash encoding（10×）、mesh 初始化好（5×）、curriculum loss 让后期 refine 不浪费时间（20×）。乘起来差不多 1000×。

**Mobile-NeRF FPS = ✗**：Mobile-NeRF 在 large-scale outdoor 根本跑不起来，因为 opacity 二值化让半透明物体全废，而且训练成本太大没法 scale 到几个平方公里。

---

## 为什么这套东西能 work：核心 intuition

我给你提炼三个底层 intuition：

### Intuition 1：Mesh 既是采样支架又是可学习 geometry

传统 NeRF 的采样点沿 ray 均匀分布，geometry 完全由 density field 隐式表示。UE4-NeRF 把采样点绑到 mesh surface，mesh vertex 又能通过 backprop 移动。所以 mesh 一开始是个 scaffold（决定在哪里采样），训练中又变成 geometry（被 photometric loss 推到正确位置）。

这种 bi-level optimization 比 NeRF 那种 "什么都 implicit" 要好收敛，因为 mesh 给了你一个 inductive bias：**surface 在哪里**。NeRF 完全靠 density field 慢慢 emerge 出来 surface，UE4-NeRF 一开始就告诉你 surface 大概在哪，让你 refine。

### Intuition 2：Texture 存 feature 不存 color，是 view-dependent 的关键

为什么不直接 bake RGB 进 texture？因为同一个点从不同角度看，color 不一样（specular、reflection）。存 RGB 等于把 view-dependent 信息丢了。

存 8D feature，把 viewing direction 在 fragment shader 里再喂进去算 color，这样既享受 rasterizer 的速度，又保留 NeRF 的 view-dependent realism。这是 SNeRG (https://arxiv.org/abs/2103.14619) 早提出来的"bake feature not color"哲学，UE4-NeRF 接着用。

### Intuition 3：Curriculum loss 让 mesh 自然 sparse

如果你想 export mesh，你希望 mesh 数量少。如果训练时只让 opacity 出现在 surface 附近，mesh 自然稀疏。但一开始就强迫稀疏，网络还没学好 geometry，会塌掉。

所以分两段：前 10000 epoch 让 opacity 自由弥散，网络先学个大概；后期加 threshold + early stopping，逼 opacity 塌缩到少数 face 上。最后 export 时按 $\alpha > 0.3$ 筛 face，刚好留 5%。**这是 curriculum 在 implicit-to-explicit 转换里的妙用。**

---

## 这 paper 的真实价值在哪

学术上，这 paper 没有全新理论。Hash encoding 是 Instant-NGP 的，mesh + neural shader 是 Mobile-NeRF 的，spatial decomposition 是 Block-NeRF / Mega-NeRF 的，depth supervision 是 Depth-supervised NeRF 的，transient handling 是 NeRF-W 的。

它的真正贡献是 **engineering synthesis**：把这些 trick 缝在一起，加一个 curriculum loss 解决 opacity sparsity，再加 UE4 这个工业级渲染引擎做后端，第一次让 "用 NeRF 重建一个城市、在 UE4 里开车飞 broomstick" 这件事变成现实。

说白了，这 paper 证明了一件事：**NeRF-as-asset 这条路走通了**。你拍几百张照片，几十分钟训完，得到一个能在游戏引擎里跑、能编辑、能加 .obj 模型的真实场景。这是 VR、Metaverse、digital twin 这些概念真正落地的一步。

---

## 如果让我挑刺

我自己觉得这 paper 有几个没解决的硬骨头：

**1. Relighting 没解决。** Neural Shader 输入是 viewing direction，没有 light direction。所以太阳位置变了，高光不会跟着动。这是 baked lighting。要解决得引入 PhySG (https://arxiv.org/abs/2109.09017) 或者 TensoIR (https://arxiv.org/abs/2205.04675) 那种 relightable NeRF 思路。

**2. 显存 scaling 还是会爆。** 600×600 m² 就 14 GB GPU memory，一个城市 10 km × 10 km 需要 ~700 GB。单卡 3090 装不下，得做更激进的 streaming，比如 NVMe 直接 streaming feature map 进 GPU。

**3. Pre-rendering 视角盲点。** Paper 说 "from any viewing angles" 提取 mesh 很难，会有 hole。Mesh 表示从 inside 看永远不如从 outside 看，underground、tunnel、indoor dense 场景是死穴。

**4. Geometry 细节其实一般。** Figure 9 的 shading mesh 看挖掘机都辨识不出来。能 render 出细节靠 texture + neural shader 补救，但 silhouette 还是粗 mesh。如果场景里有大量 thin structure（铁丝网、栅栏），会出问题。

**5. 如果换成 Gaussian Splatting 可能更简单。** Gaussian Splatting (https://arxiv.org/abs/2308.14737) 是 2023 中后期出来的，UE4-NeRF 投稿时还没火。回头看，Gaussian Splatting 对 large-scale + real-time 用 3D Gaussians + tile-based rasterizer 直接解决，path 更简洁。但 Gaussian 难做传统 3D 编辑，editable 性差，所以 UE4-NeRF 的 mesh + UE4 集成这条路在 "可编辑" 这个维度上仍然有优势。

---

## 一句话总结的人话版

UE4-NeRF = **"用 NeRF 训练，把场景烤成 mesh + feature texture，塞进 UE4 用 rasterizer 跑，顺便解决了半透明、大场景、可编辑三件事"**。学术上没新理论，工程上把一堆 trick 缝在一起，第一次让 NeRF 重建的真实场景能在游戏引擎里实时跑、能编辑、能加 object。这就是这 paper 的价值。

---

## 参考链接

- UE4-NeRF: https://jamchaos.github.io/UE4-NeRF/
- NeRF: https://arxiv.org/abs/2003.08934
- Instant-NGP: https://nvlabs.github.io/instant-ngp/
- Mobile-NeRF: https://arxiv.org/abs/2208.00277
- Block-NeRF: https://arxiv.org/abs/2202.05309
- Mega-NeRF: https://arxiv.org/abs/2207.02535
- Mip-NeRF 360: https://arxiv.org/abs/2111.12077
- NeRF-W: https://arxiv.org/abs/2008.02268
- SNeRG: https://arxiv.org/abs/2103.14619
- Depth-supervised NeRF: https://arxiv.org/abs/2107.02791
- Gaussian Splatting: https://arxiv.org/abs/2308.14737
- Panoptic-DeepLab: https://arxiv.org/abs/1911.10164
- PSPNet: https://arxiv.org/abs/1612.01105
- PhySG: https://arxiv.org/abs/2109.09017
- TensoIR: https://arxiv.org/abs/2205.04675
- DMET: https://arxiv.org/abs/2110.01154
- Nerfstudio: https://docs.nerf.studio/
- COLMAP: https://colmap.github.io/
- BC4 compression: https://docs.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-block-compression

---

# UE4-NeRF: 为 Large-Scale Scene 的 Real-Time Rendering 而生的 Hybrid Representation

Andrej 你好。这篇 paper 2023 年发出来, 我觉得它走的路线跟同期 Mobile-NeRF、BakedNeRF、SNeRG 这一类 "把 NeRF 蒸馏成 rasterizable asset" 的工作是同一脉络，但是它的工程取舍明显偏向**生产可用性**而非学术 benchmark。下面我尽量把它的设计逻辑、公式含义、实验数据和它与你熟悉的几条 NeRF 主线的关系讲透，帮你 build intuition。

---

## 1. 这篇 paper 的真正动机

NeRF 的 original formulation 用一个 MLP $F_\theta: (\mathbf{x}, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma)$ 把一个 continuous 5D field 嵌进网络权重里。它的渲染公式：

$$\hat{\mathbf{C}}(\mathbf{r}) = \sum_{i=1}^{N} T_i \alpha_i \mathbf{c_i}, \quad T_i = \prod_{j=0}^{i-1}(1-\alpha_j)$$

这里 $\mathbf{r} = \mathbf{o} + t\mathbf{d}$ 是 ray, $\alpha_i = 1 - \exp(-\sigma_i \delta_i)$ 是 discretized alpha, $T_i$ 是 accumulated transmittance。每帧每个 pixel 都要 query MLP 几百次, 所以 throughput 一直是 NeRF 的死穴。

UE4-NeRF 想解决的是三个**叠加在一起**的问题：
1. **Large-scale**: 几平方公里, UAV 拍出来的场景, 像素 6000×4000, 含 GPS
2. **Real-time**: 4K @ 43 FPS, 必须走传统 rasterization pipeline 而非 volume rendering
3. **Editable**: 渲出来之后还要能在 UE4 里加车、加 broomstick、做 composition

Mobile-NeRF (Chen et al. 2022, https://arxiv.org/abs/2208.00277) 已经证明了 NeRF-to-mesh + GLSL fragment shader 的可行性, 但它的两个痛点被 UE4-NeRF 直接打到:

- **训练成本**: Mobile-NeRF 训一个小 scene 要 8×V100 训几天, 因为它要从头把 mesh 和 texture atlas 都学出来, 而且 hash encoding 之外的 opacity binarization 让 optimization landscape 很崎岖
- **半透明物体**: Mobile-NeRF 把 opacity 二值化, 所以水、玻璃、塑料棚这种 outdoor 常见 material 直接废掉。这对于 farm、construction site 这种场景是致命的

UE4-NeRF 用了三个核心 tricks 一起解决这些: (a) **multi-resolution hash encoding 借自 Instant-NGP** (https://nvlabs.github.io/instant-ngp/) 加速 convergence; (b) **regular octahedron mesh 初始化 + vertex optimization** 避免 Mobile-NeRF 的 binarization; (c) **alpha-dithered + TAA** 让 mesh pipeline 也能模拟半透明。这三件事合在一起把训练时间从 ~2880 min 砍到 ~40 min, 1000× 的 speedup (Table 3)。

---

## 2. 整体架构 (对应 Figure 2)

UE4-NeRF 是一个三段式系统:

```
Training ────► Pre-rendering ────► Rendering
   │                │                  │
   │                │                  ├── UE4 submodule (rasterizer + Neural Shader)
   │                │                  └── Inference submodule (feature map cache)
   │                │
   │                ├── Mesh extraction (opacity > 0.3 face clip)
   │                └── UV mapping + BC4 compression
   │
   ├── Block partition (with overlap, coarse model 先训)
   ├── Encoder-Decoder MLP (multi-resolution hash + SH)
   ├── Mesh vertex optimization (regular octahedron 20-face init)
   └── LOD training (128³ → 64³ → 32³ → 16³ → 8³)
```

这个 partition + per-block NeRF + 共享 inference cache 的设计, 让我想到 Block-NeRF (https://arxiv.org/abs/2202.05309) 和 Mega-NeRF (https://arxiv.org/abs/2207.02535) 的 spatial decomposition, 但区别在于 UE4-NeRF 把每个 block 直接 bake 成 polygon mesh, 这是 Block-NeRF/Mega-NeRF 没做的——后者依然停留在 volume rendering 范式。

---

## 3. 训练阶段的数学细节

### 3.1 Mesh 初始化: 为什么是 regular octahedron?

这个选择很有意思。原始 Mobile-NeRF 是在 voxel grid 的每个 cell 中心放一个 cube, 每个面是一个 binary-opacity quad。问题在于 cube 的 6 个面只能 align 到 axis, 倾斜 surface (山坡、屋顶) 会出现严重的 aliasing 和 convergence 不稳定。

UE4-NeRF 选择 **20-face regular octahedron**: 8 个 exterior face + 12 个 interior face。这其实是把每个 voxel cell 用一个接近球形的 polyhedron 包起来。直觉上是: spherical-ish mesh 提供了一个 rotation-invariant 的初始化, vertex 可以往任何方向 drift, 所以 steep slopes、tree canopy、电力线这种 thin structure 都能被 mesh 顶到。

它的 ray-mesh intersection 用标准 ray-triangle test, 然后所有 intersection 点作为 sampling points (而不是 NeRF 那种 stratified sampling along ray)。这点很关键: **采样点的位置由 mesh 决定, 同时 mesh 的 vertex 又由 photometric loss 反传回去优化**。这是一个 bi-level optimization, mesh 既 是采样支架, 也是可学习 geometry。

### 3.2 Encoder-Decoder 分离

公式 (1) 和 (2):

$$\mathcal{E}(\text{hash}(\mathbf{p_i}); \theta_\mathcal{E}) \rightarrow \mathbf{M_i}, \alpha_i$$
$$\mathcal{D}(\mathbf{M_i}, \mathcal{SH}(\mathbf{d_i}); \theta_\mathcal{D}) \rightarrow \mathbf{c_i}$$

变量含义:
- $\mathbf{p_i} \in \mathbb{R}^3$: 第 $i$ 个 sampling point 的 3D 位置
- $\text{hash}(\cdot)$: Instant-NGP 风格的 multi-resolution hash encoding, $L$ 个 level, 每个 level $N_l$ 个 grid, 输出 $\mathbf{F}$ 维 feature
- $\theta_\mathcal{E}$: Encoder MLP 权重, 结构是 $32\times 64, 64\times 64, 64\times 64, 64\times 8$ (4 层)
- $\mathbf{M_i} \in \mathbb{R}^8$: 8 维 feature vector, **会写进 texture map**, 后面给 UE4 用
- $\alpha_i \in [0,1]$: 直接预测 alpha (而不是 NeRF 那种 $\sigma$ 再 discretize), 这是为了和 mesh rasterization pipeline 对齐
- $\mathcal{SH}(\mathbf{d_i})$: viewing direction 用 Spherical Harmonics 编码, 9 维 (degree 2 截断)
- $\theta_\mathcal{D}$: Decoder MLP 权重, $17\times 16, 16\times 16, 16\times 3$ (3 层)。输入 17 = 8 (feature) + 9 (SH direction)
- $\mathbf{c_i} \in \mathbb{R}^3$: 最终 RGB color

这个分离是直接抄了 Mobile-NeRF 的设计哲学: **view-independent 的东西 (geometry, albedo-ish feature) 写进 texture atlas; view-dependent 的东西 (specular, reflection) 在 fragment shader 里实时算**。这样 texture map 可以被 GPU 缓存、被 rasterizer 直接采样, 不需要每帧 re-evaluate MLP。

### 3.3 两段式 photometric loss: 这是 paper 最聪明的地方

普通 NeRF loss 就是 $\|\hat{\mathbf{C}} - \mathbf{C}\|_2^2$。但 UE4-NeRF 后期要把 mesh 提取出来 bake, 只保留 $\alpha > 0.3$ 的 face, 所以训练阶段必须**强迫 opacity 集中到少数 mesh 上**, 否则提取出来的 mesh 到处都是, rasterization 慢且占显存。

它设计了两个 part:

**Part 1** (Eq. 4): 标准 photometric loss
$$\mathcal{L}_{rgb}^{part1}(\theta, V_p) = \sum_{r \in \mathcal{R}} \|\hat{\mathbf{C}}(\mathbf{r}) - \mathbf{C}(\mathbf{r})\|_2^2$$
其中 $V_p$ 是当前 LOD level 下 polygon mesh 的 vertex 位置, $\mathcal{R}$ 是 batch 里的 ray 集合, $\mathbf{C}(\mathbf{r})$ 是 GT color。

**Part 2** (Eq. 5 + 6): 带 opacity threshold 和 early-stopping 的渲染
$$f = \min\{0.3, \lfloor \frac{epoch - 10000}{10000}\rfloor^2 \times 0.012\}$$
$$\hat{\mathbf{C}}'(\mathbf{r}) = \frac{\sum_{\phi=1}^N T_\phi \alpha_\phi \mathbf{c_\phi}}{1 - T_\phi'}$$
$$\mathcal{L}_{rgb}^{part2} = \sum_{r \in \mathcal{R}} \|\hat{\mathbf{C}}'(\mathbf{r}) - \mathbf{C}(\mathbf{r})\|_2^2$$

关键变量:
- $f$: 动态阈值, epoch 10000 之前 $f = 0$ (退化为普通 NeRF 训练); 10000 之后逐渐爬升, 但不超过 0.3
- $T_\phi'$: early-stopping 时的剩余 transmittance, 论文里说当 accumulated opacity 超过 0.8 (即 $T < 0.2$) 就停
- 分母 $1 - T_\phi'$ 是个 normalization: 因为我们提前终止了 ray marching, 后面的 contribution 被砍掉, 所以把现有的 accumulated color scale 上去 cover 整个 color range

这个设计的 intuition 是: **训练前期允许 opacity 弥散, 让网络先学会大致的几何和颜色; 训练后期逐渐收紧, 把 opacity 压缩到少数几个 mesh face 上**, 这样 export 时直接按 $\alpha > 0.3$ 筛 face, 5% 的 mesh 就能保留所有可见 surface (paper 5% 数据来自 Section 3.2)。

两段用 $\beta$ 加权 (Eq. 7):
$$\beta \mathcal{L}^{part1} + (1-\beta) \mathcal{L}^{part2}, \quad \beta = 0.8^{\lfloor epoch / 10000 \rfloor}$$

$\beta$ 是 part1 的权重, 随 epoch 指数衰减。$\lfloor \cdot \rfloor$ 是 floor 函数, 每 10000 epoch 跌一档。早期 $\beta \approx 1$, 全靠 part1; 后期 $\beta \to 0$, 全靠 part2。这其实是个**curriculum**, 从 "soft NeRF" 慢慢过渡到 "hard mesh"。

### 3.4 LOD 多分辨率训练

它对每个 block 训练 5 个 LOD level, mesh grid 从 128³ → 64³ → 32³ → 16³ → 8³。**重要细节**: 同一个 block 的所有 LOD 共享同一个 Encoder-Decoder 网络, 但**只有 level-1 (最精细) 的 loss 会回传到 network 权重**, 粗 LOD 的 loss 只更新 vertex 坐标。

公式 (8) 写成:
$$\mathcal{L}_{rgb} = \sum_{l=1}^{5} \beta \mathcal{L}_{rgb}^{part1}(\theta, V_p) + (1-\beta)\mathcal{L}_{rgb}^{part2}(\theta, V_p)$$

但 paper 在文字里说明: coarse level 的 gradient 只 flow 到 $V_p$, 不 flow 到 $\theta$。这个 stop-gradient 设计是为了**防止粗 mesh 给共享 network 错误的 supervision**——比如 coarse mesh 表示不了 power line, 就不该逼 network 学一个模糊的 power line feature。这个 trick 在 coarse-to-fine 3D reconstruction 里很常见, 类似 Mip-NeRF 360 (https://arxiv.org/abs/2111.12077) 的 online distillation 思路。

### 3.5 Pseudo-depth supervision

公式 (9): $\mathcal{L} = \mathcal{L}_{rgb} + \mathcal{L}_D$

paper 没在主文给 $\mathcal{L}_D$ 的具体形式, 只说 "see supplementary"。但它的来源很有意思: **不**用 RGB-D camera (因为 outdoor 光照下深度不准, 范围也有限), 而是**复用 COLMAP 的 sparse point cloud**。

具体流程:
1. COLMAP SFM 估计 camera pose (这步本来 NeRF 也要做)
2. 顺便拿到 sparse 3D point cloud $\{\mathbf{P}_k\}$
3. 对每张训练图, 把 sparse point cloud project 回 image, 拿到 sparse depth map
4. 这些 sparse depth 当作 supervision, 监督 NeRF 在对应 ray 上的 depth $\hat{t} = \sum T_i \alpha_i t_i$

这跟 Depth-supervised NeRF (https://arxiv.org/abs/2107.02791) 一脉相承, 但巧妙在**完全复用已有 COLMAP 输出**, 没有 extra sensor 依赖。对 large-scale UAV 场景特别合适, 因为 drone 的 RGB-D 几乎不实用。

### 3.6 Transient objects 处理

paper 用 Panoptic-DeepLab (https://arxiv.org/abs/1911.10164) 和 PSPNet (https://arxiv.org/abs/1612.01105) 做 semantic segmentation, 把行人、车这些 dynamic object mask 掉, 不参与 training。这是抄 NeRF-W (https://arxiv.org/abs/2008.02268) 的 transient object handling 思路, 但更暴力——NeRF-W 是给 transient object 学一个 separate head, 这里直接 mask 掉。

我觉得这处 trade-off 是合理的: 对 large-scale aerial scene, dynamic object 本来就是 noise, 不值得花 model capacity 去学。

---

## 4. Pre-rendering 阶段

训练完之后, 每个 block 有: network 权重 + 5 个 LOD 的 mesh (vertex 已优化)。

Pre-rendering 干三件事:

### 4.1 Mesh 提取

不只从训练视角发射 ray, 还在 block 上方加一束 **parallel lights** (多角度), 用这些 ray 跟 mesh 求 intersection。当某个 triangle face 上所有 intersection 的 opacity 都 $< 0.3$, 就 clip 掉这个 face。

paper 统计: 最终保留下来的 mesh 只占原始的 ~5%。这个 5% 是 UE4-NeRF 实际能 real-time 的关键原因之一——100 万个 triangle 跟 5 万个 triangle 在 rasterizer 里完全是两个世界。

### 4.2 坐标对齐

把 mesh 的坐标从 NeRF local frame 转到 UE4 world frame。UE4 坐标系: +X 东, +Y 南, +Z 上, scale 100:1 (即 1 unit = 1 cm, 100 unit = 1 m)。GPS 给的地理坐标先转到 ENU (East-North-Up) 再 scale。

### 4.3 UV mapping

每个 triangle 的 vertex 坐标映射到 texture atlas 的 UV 坐标。这里用的是 per-mesh 32×32 texture (Section 3.3 "Texture map")。

---

## 5. Rendering 阶段: UE4 + Neural Shader

这是整个 system 的核心 trick。流程:

1. **UE4 rasterizer** 走传统 polygon pipeline, 把每个 mesh face rasterize 成 pixels
2. 每个 pixel 从 texture map 采样得到 8 channel feature (即 $\mathbf{M_i}$) + 1 channel opacity
3. **Neural Shader** (HLSL fragment shader) 在 GPU 上跑 Decoder MLP, 输入是 8 channel feature + 9 channel SH(direction) = 17D, 输出 RGB
4. **Alpha dithering + TAA** 模拟半透明

### 5.1 BC4 压缩

Texture map 是 9 个 single-channel map, 用 **BC4 compression** (https://docs.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-block-compression) 压缩。BC4 是 block compression, 4×4 pixel block 压成 8 byte (单通道), 压缩比 4:1, 所以 9 个 channel 用 BC4 后只占原始的 1/3。这就是 paper 说的 "reduces 2/3 GPU memory"。

注意 **texture map 不存 color**, 存的是 spatial feature + opacity。color 由 Neural Shader 实时算。这跟传统 game 的 texture 完全不同, 倒是跟神经纹理 (neural texture) 概念一致, 类似 NSVF (https://arxiv.org/abs/2103.06952)。

### 5.2 动态 inference cache

每个 block 都给所有 mesh 存 texture map 太大了 (几个平方公里场景)。所以 UE4-NeRF 用 **dynamic inference**:
- UE4 submodule 告诉 inference submodule "现在要看 block $k$ 的 level $l$"
- Inference submodule 算 (或从 cache 取) 这个区域 mesh 的 feature map, 推给 UE4
- 同一区域反复看时不重算

这是 NeRF 的 "lazy baking" 思路。和 SNeRG (https://arxiv.org/abs/2103.14619) 的 full baking 不同, UE4-NeRF 只 bake 当前需要看的部分。这跟 streaming open-world game 的 asset streaming 几乎一模一样。

### 5.3 半透明处理

普通 mesh rasterization 是 opaque, Z-buffer 一次写入即可。但 outdoor 场景有树叶、水、玻璃。UE4-NeRF 不走 order-independent transparency (太贵), 而用 **alpha dithering + TAA**:

- Alpha dithering: 用 stochastic dithering pattern (比如 Bayer matrix 或 blue noise) 把 alpha 转成 binary mask, 只保留对应位置的 pixel
- TAA (Temporal Anti-Aliasing): 跨帧累积, 把抖动的高频 noise 平均掉

这是 graphics 圈很老的 trick, Williams 等人 2003 年的 LLNL tech report (https://www.osti.gov/biblio/15005875) 早就讨论过。UE4 内置 TAA, 所以基本 free。代价是 dynamic object 可能有 ghosting, 但对 static scene 影响很小。

---

## 6. 实验数据深度解读

### 6.1 数据集 (Table 1)

| Scene | Area (m²) | Host Mem (4k) | GPU Mem (4k) | FPS@2K | FPS@4K | Camera Altitude |
|-------|-----------|---------------|--------------|--------|--------|-----------------|
| FL (Farmland) | 180×240 | ~12 GB | ~5 GB | 55 | 36 | 120 m |
| CS (Construction) | 420×240 | ~25 GB | ~11 GB | 66 | 43 | 70 m |
| IP (Industrial Park) | 600×600 | ~40 GB | ~14 GB | 58 | 35 | 120 m |

观察:
- **GPU 显存**几乎是 linearly scale with area, 4 GB/100k m² 量级。这个 slope 其实不算特别好——意味着一个 1 km² 城市需要 ~70 GB GPU memory, 单卡 3090 (24GB) 装不下
- **FPS 不严格随 area 下降**, 因为 IP (360k m²) 反而比 CS (100k m²) FPS 高一点点。paper 解释是 **CS 有更多半透明物体 (construction 临时棚、塑料布)**, alpha dithering 开销大
- **Camera altitude**: 70~120 m, 这是 UAV 航拍典型高度, 也是 LOD 选择的依据

### 6.2 跟其他方法对比 (Table 2)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | FPS(2K) | Host Mem | GPU Mem | Train Time |
|--------|-------|-------|--------|---------|----------|---------|------------|
| NeRFacto | 20.99 | 0.663 | 0.389 | 0.5 | ~14 GB | ~3 GB | ~45 min |
| Instant-NGP | 22.00 | 0.631 | 0.426 | 11 | ~10 GB | ~32 GB | ~15 min |
| Mega-NeRF | 17.37 | 0.23 | 0.546 | ✗ | — | — | ~2160 min |
| Mobile-NeRF | 16.92 | 0.23 | 0.419 | ✗ | — | — | ~2880 min |
| **UE4-NeRF** | **25.03** | **0.704** | **0.287** | **55** | ~19 GB | ~3 GB | ~40 min |

几个关键 takeaways:

1. **PSNR 25.03 vs Instant-NGP 22.00**——3 dB gap 非常显著。这个 gain 主要来自三点: pseudo-depth supervision、regular octahedron mesh 表示能力比 voxel 强、curriculum loss 让网络后期 focus 在 surface 上
2. **GPU 显存只 ~3 GB**——这跟 Instant-NGP 的 32 GB 形成鲜明对比。原因是 UE4-NeRF 在 inference 阶段几乎不存 network feature grid, 因为已经 bake 进 texture atlas + mesh
3. **训练时间 40 min** vs Mobile-NeRF 2880 min = 1000× speedup。这个 1000× 主要是两个因素: (a) hash encoding 加速 (Instant-NGP 也用, 但 Mobile-NeRF 用的是 positional encoding), (b) Mesh 表示收敛快 (octahedron 初始化给了一个好起点)
4. **Mobile-NeRF 和 Mega-NeRF 在 large-scale 上 FPS = ✗**, 是因为它们根本跑不了完整 large scene, Mobile-NeRF 因为 binary opacity 退化, Mega-NeRF 是 volume rendering 慢

### 6.3 训练时间分解 (Table 3)

| Method | Per-block (1×3090) | Whole scene (4×3090) | FPS |
|--------|---------------------|----------------------|------|
| Mobile-NeRF | 48h | n×48h (串行) | 35 @ 2K (block only) |
| Mega-NeRF | 36h | 12h + 36h×n/4 | 60 @ 800×800 (scene) |
| **UE4-NeRF** | **40 min** | **1h + 40min×n/4** | **50 @ 2K (scene)** |

注意 Mega-NeRF 的 "12h + 36h×n/4" 是因为 12h 用于 scene-level coarse model, n 个 block 用 4 卡并行训。UE4-NeRF 同理, 1h 用于 coarse block partition model, 后面 block 并行训练。**这个公式告诉我们: UE4-NeRF 对 block 数量的 scaling 几乎是线性的**, 加卡就能加 block。

---

## 7. 设计直觉总结 (这是给你的核心 takeaways)

### Intuition 1: **Mesh 既 是 sampling scaffolding 又 是可学习 geometry**

传统 NeRF 的 sampling 是 ray 上的均匀/stratified 点, geometry 完全由 volume density 隐式表示。UE4-NeRF 把 sampling 点绑到 mesh surface, 让 mesh vertex 通过 backprop 移动。这样 geometry 变成 explicit + differentiable, 后续 bake 不需要 marching cubes 这种额外步骤。

这让我想起 **DeepSDF** (https://arxiv.org/abs/1901.05103) 和 **DMTet** (https://arxiv.org/abs/2110.01154) 的可微 mesh 思路。区别是 UE4-NeRF 不用 SDF, 直接用 mesh vertex 当 free parameter。

### Intuition 2: **Curriculum loss 是核心 contribution**

如果没有 part2 loss, mesh vertex optimization 会让 opacity 弥散到所有 face 上 (因为 part1 loss 是稠密的 RGB 监督, 任何 face 上的 opacity 都能 contribute)。Part2 loss 通过 threshold + early stopping 强制 opacity collapse 到少数 face, 这是 export 出来 mesh 能 sparse 的根本原因。

$0.8^{\lfloor epoch/10000\rfloor}$ 这个衰减 schedule 其实挺激进的, 每 10000 epoch 跌 20%, 意味着 30000 epoch 后 part2 主导。80k epoch 总训练的话, 后 50k epoch 都在 refine mesh sparsity。

### Intuition 3: **Why Neural Shader instead of baked RGB?**

为什么不直接 bake RGB color 进 texture, 让 rasterizer 一遍过? 因为 NeRF 的 view-dependent appearance (specular、reflection) 是和 viewing direction 强相关的。如果存 RGB, 同一个 pixel 从不同角度看都是同一个颜色, 高光就废了。

Neural Shader 保留 8D view-independent feature + 9D view-dependent SH encoding, 在 fragment shader 里实时算 color, 这样既享受 rasterizer 的速度, 又保留 NeRF 的 view-dependent realism。

这跟 NeRF 的 "bake feature, not color" 哲学一致, SNeRG 也是这么做的。

### Intuition 4: **UE4 集成的工程价值**

paper 在 Section 4.4 提到一个反直觉的现象: **加 object 之后 FPS 反而略微提升**, 因为遮挡部分 NeRF 不需要 Neural Shader 计算。这说明 UE4-NeRF 的 rasterization pipeline 跟传统 mesh 一样支持 early-Z, 这对后续 game-style 场景非常重要。

UE4 还天然支持 .obj / .fbx 导入, 物理碰撞, 第一人称控制, broomstick flight (paper Figure 1c), 这些都是 volume-rendering NeRF 永远做不到的。

---

## 8. 局限性 (paper 自己承认的)

1. **CUDA 依赖**: 必须 NVIDIA GPU。这其实是 BC4 压缩 + custom fragment shader 的副产物。要跨 vendor 需要重写压缩 path
2. **显存 scaling**: paper 的 IP 场景 600×600 m² 已经 14 GB GPU memory。如果要重建一个城市 (10 km × 10 km), 显存会爆炸。需要更激进的 streaming, 比如 NVMe 直接 streaming feature map
3. **Pre-rendering 视角盲点**: paper 说 "from any viewing angles" 提取 mesh 难, 会有 hole。这其实是 mesh 表示的 fundamental limitation——inside-out 不可能。所以 underground、tunnel、dense indoor 是它的死穴

我自己补充几个 paper 没提的 limitations:

4. **Dynamic lighting**: Neural Shader 输入是 9D SH direction, 但**没有 light direction**。所以重建出来的 scene 是 baked lighting, 太阳位置变了之后高光不会动。要解决这个得引入 relightable NeRF 思路 (比如 NeRF-W 的 appearance embedding 扩展, 或者 PhySG https://arxiv.org/abs/2109.09017)
5. **Geometry 细节**: Figure 9 的 shading mesh 看上去非常粗糙, 挖掘机都辨识不出来。能 render 出细节是因为 texture + neural shader 补救, 但 silhouette 仍然是粗 mesh。这意味着 paper 看到的 PSNR 25 dB 主要靠 texture 细节, 不是几何细节
6. **训练 ray-mesh intersection 的成本**: 每个 batch 要算几千条 ray 跟几十万 mesh triangle 求 intersection。Paper 没给训练时的 ray-mesh 加速结构 (BVH? KD-tree?), 但 40 min/block 训练时间暗示这块用了 CUDA 优化得很彻底

---

## 9. 跟其他工作的关系图

```
NeRF (2020) ─────► Volume rendering 范式
   │
   ├──► Instant-NGP (2022): hash encoding 加速训练
   │       │
   │       └──► UE4-NeRF (本 paper): 用 hash encoding + mesh 表示
   │
   ├──► Block-NeRF / Mega-NeRF (2022): spatial decomposition for large-scale
   │       │
   │       └──► UE4-NeRF: 同样 partition, 但 bake 成 mesh
   │
   ├──► Mobile-NeRF (2022): mesh + GLSL shader for mobile real-time
   │       │
   │       └──► UE4-NeRF: 同样 mesh 思路, 但解决 opacity binarization 和训练慢
   │
   ├──► NeRF-W (2020): transient / appearance handling
   │       │
   │       └──► UE4-NeRF: 借用 transient mask 思路
   │
   ├──► Depth-supervised NeRF (2021): depth 监督加速收敛
   │       │
   │       └──► UE4-NeRF: pseudo-depth from COLMAP
   │
   └──► SNeRG / PlenOctrees (2021): bake NeRF 进 sparse voxel / octree
           │
           └──► UE4-NeRF: 进一步 bake 进 polygon mesh, 用 GPU rasterizer
```

可以看出, UE4-NeRF 真正的 contribution 是 **engineering synthesis**: 把 Instant-NGP 的 hash encoding、Mobile-NeRF 的 mesh+bake 思路、Block-NeRF 的 spatial decomposition、NeRF-W 的 transient handling、Depth-supervised NeRF 的 pseudo-depth 全部缝在一起, 再用 UE4 的 rasterization pipeline 把 render 跑通。

学术上没有全新理论, 但 system-level 是一次漂亮的 integration, 而且验证了 **"NeRF-as-asset" 这条 production 路径的可行性**。

---

## 10. 你可能会问的几个问题 (我想象 Karpathy 视角)

**Q: 为什么不直接用 Gaussian Splatting?**

Gaussian Splatting (https://arxiv.org/abs/2308.14737) 是 2023 中后期出来的, UE4-NeRF 投稿时还没流行。但回头看, Gaussian Splatting 对 large-scale + real-time 的解决路径完全不同: 它用 3D Gaussians 作为 primitive, 用 tile-based rasterizer 直接 render。相比之下 UE4-NeRF 的 mesh + neural shader 路径更复杂, 但**editable 性更好** (Gaussian 难做传统 3D 编辑)。另外 Gaussian Splatting 对 very thin structure (power line) 表现也未必好。

**Q: 1000× speedup 主要来自哪里?**

我估算: hash encoding vs positional encoding 大概 10× (参考 Instant-NGP paper); mesh 表示让 supervision 更直接, 大概 5×; octahedron 初始化避免 binarization 的不稳定, 大概 5×; 剩余来自 curriculum loss 减少了后期 refine 的时间。乘起来 ~250-1000×, 跟 paper 报告吻合。

**Q: PSNR 25 dB 是不是作弊 (因为 UE4 渲出来 exposure 不一致)?**

Paper 自己承认 "actual rendering quality is expected to surpass the metric's performance"。PSNR 25 在 large-scale outdoor 算合理但不出彩 (Mip-NeRF 360 能做到 27+ on bounded scene)。但 Mip-NeRF 360 不能 real-time, 也不能 large-scale。这是 Pareto front 上的不同点。

**Q: 这套东西能 port 到 Unreal Engine 5 吗?**

理论上可以, 而且 UE5 的 Nanite (virtualized geometry) 正好能解决 mesh 数量爆炸的问题, Lumen (global illumination) 可能还能给 Neural Shader 提供 light direction 作为额外 input, 解决 relighting 问题。如果有人做 UE5-NeRF, 把 Nanite + Lumen 接进来, 应该比 UE4 版本更强。这是我猜的下一步方向。

---

## 11. 参考链接汇总

- UE4-NeRF project page: https://jamchaos.github.io/UE4-NeRF/
- NeRF (Mildenhall et al. 2020): https://arxiv.org/abs/2003.08934
- Instant-NGP (Müller et al. 2022): https://nvlabs.github.io/instant-ngp/ 
- Mobile-NeRF (Chen et al. 2022): https://arxiv.org/abs/2208.00277
- Mega-NeRF (Turki et al. 2022): https://arxiv.org/abs/2207.02535
- Block-NeRF (Tancik et al. 2022): https://arxiv.org/abs/2202.05309
- Mip-NeRF 360 (Barron et al. 2022): https://arxiv.org/abs/2111.12077
- NeRF-W (Martin-Brualla et al. 2020): https://arxiv.org/abs/2008.02268
- SNeRG (Hedman et al. 2021): https://arxiv.org/abs/2103.14619
- Depth-supervised NeRF (Deng et al. 2021): https://arxiv.org/abs/2107.02791
- Gaussian Splatting (Kerbl et al. 2023): https://arxiv.org/abs/2308.14737
- Panoptic-DeepLab (Cheng et al. 2020): https://arxiv.org/abs/1911.10164
- DMET (Shen et al. 2021): https://arxiv.org/abs/2110.01154
- Nerfstudio (Tancik et al. 2023): https://docs.nerf.studio/
- COLMAP (Schönberger et al. 2016): https://colmap.github.io/
- BC4 compression docs (MSDN): https://docs.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-block-compression

---

总结一下 build intuition 的核心: **UE4-NeRF 是把 NeRF 的 implicit field 转译成 explicit mesh 的过程, 用 curriculum loss 让 opacity 自然 collapse 到 mesh face 上, 用 neural shader 保留 view-dependent appearance, 用 UE4 rasterizer 实现 real-time, 用 LOD 和 dynamic inference cache 解决 large-scale**。它的价值不在单个 trick, 而在这一套 stack 串起来后第一次让 "用 NeRF 重建出来的城市能像 3A 游戏一样跑、一样编辑" 这件事变成现实。后续如果有人把 Nanite + Lumen 接进来, 或者把这套思路迁移到 Gaussian Splatting, 应该会很有意思。
