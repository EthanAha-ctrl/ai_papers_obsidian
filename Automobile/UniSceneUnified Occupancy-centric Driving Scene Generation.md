---
source_pdf: UniSceneUnified Occupancy-centric Driving Scene Generation.pdf
paper_sha256: e1e9b3bb8d200b745c4745c0efb51d38b8811a48fc17fdc897e5030acb985fe1
processed_at: '2026-08-12T20:11:44-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 UniScene

## 这篇论文到底在干嘛

自动驾驶公司缺数据，采集真实数据又贵又慢，所以大家想用AI生成假数据来训练模型。但问题是你要生成的数据种类很多：

- 给摄像头看的视频
- 给LiDAR看的点云  
- 给occupancy网络看的3D体素

之前的方法都是"单打独斗"——做一个专门生成视频的模型，做一个专门生成LiDAR的模型。而且都是从2D的鸟瞰图直接硬生成，这个mapping太难学了。

UniScene说：我做一个模型，三个都能生成，而且质量还比专门做的更好。

## 核心思路：找个中间人

打个比方。你让一个厨师直接从"食材清单"做出一桌完整的菜（有炒菜、有汤、有甜点），这个要求太变态了，厨师得同时精通所有菜系。

UniScene的思路是：先从食材清单做一个"半成品料包"，然后从料包分别做炒菜、做汤、做甜点。每个步骤都简单了。

这个"半成品料包"就是 **semantic occupancy** ——一个3D的体素网格，每个voxel知道自己的semantic类别（路面、车、建筑等）和位置。

为什么选occupancy当中间人？

1. 它有3D geometry信息（比BEV强）
2. 它有semantic信息（比纯mesh强）
3. 它是view-invariant的（所有sensor都从它rendering出来，物理上保证一致性）
4. 从BEV生成它相对容易（都是3D structure）

## 三个生成任务怎么做的

### Task 1: BEV → Occupancy

这是第一步，也是最关键的一步。

**痛点**：之前的OccSora、OccWorld用VQVAE压缩occupancy，压缩狠了重建质量崩盘。OccSora在512倍压缩下mIoU只有27.4，惨不忍睹。

**UniScene的解法**：用continuous VAE替代VQVAE。同样是512倍压缩，mIoU干到72.9，提升45个点。这个gap直接决定了后面所有生成质量的上限。

具体架构：
- **VAE Encoder**：只做spatial，把3D occupancy压成2D BEV-style latent
- **VAE Decoder**：做spatial + temporal，把latent sequence还原成occupancy sequence
- **DiT**：标准的diffusion transformer，BEV layout和noise在input层面concat，一起patchify后过spatial + temporal attention

为啥encoder不做temporal而decoder做？因为encoder要保持灵活性，单帧也能encode；decoder才需要考虑帧间一致性。

### Task 2: Occupancy → Video

这里有个很巧的bridge。

**痛点**：occupancy是3D voxels，video是2D pixels，怎么让video model理解occupancy？

**UniScene的解法**：用3D Gaussian Splatting把occupancy render成两个东西：
1. **Depth map**：每个camera view的深度图
2. **Semantic map**：每个camera view的语义图

这两个东西喂给video diffusion model当condition。

**为什么这个比spatial-temporal attention好？**

以前的方法（MagicDrive等）用attention机制让多个camera view之间互相看，强行对齐。但attention是软约束，学不好就会cross-view不一致。

UniScene的rendering是硬约束：4个camera view的depth和semantic都来自同一个3D occupancy，物理上不可能不一致。结果FVD从110降到70，提升巨大。

还有个细节trick叫 **geometric-aware noise prior**：SVD默认把reference frame的latent直接加到noise里。UniScene用rendered depth做warping，把reference frame的appearance根据geometry重投影到target frame的位置。这样moving car这种dynamic region也能正确建模motion。

### Task 3: Occupancy → LiDAR

这里的关键是 **efficiency**。

**痛点**：occupancy是dense的，LiDAR ray要采样很多点，如果每个voxel都算neural rendering，太慢。

**UniScene的解法**：
1. **Sparse UNet**：只在occupied voxels上做卷积，empty space跳过
2. **Prior-guided sampling**：LiDAR ray上只在occupancy标记为occupied的地方采样，empty space不浪费算力
3. **Ray-dropping head**：预测哪些ray会被drop掉（真实LiDAR有这个现象）

结果：质量比LiDARDM好31% MMD，速度快100倍（0.47s vs 45.12s）。

## 效果总结

| Task | Metric | UniScene | 之前SOTA | 提升 |
|------|--------|----------|----------|------|
| Occupancy Gen | F3D | 158.55 | 145.65 (OccWorld) | - |
| Video Gen | FVD | 71.94 | 112.65 (Vista*) | -36% |
| LiDAR Gen | MMD | 2.40 | 3.51 (LiDARDM) | -32% |
| LiDAR Speed | Time | 0.47s | 45.12s | 100x |

下游任务（用生成的数据训练感知模型）也全面提升：
- Occupancy prediction: +3.7 mIoU
- BEV segmentation: +5.62 mIoU  
- 3D detection: +3.41 NDS

## 我的take

这篇paper最核心的insight其实就一句话：**找对intermediate representation比设计复杂architecture重要得多**。

Occupancy这个representation好在哪里？它把"场景是什么"（semantic）和"场景长什么样"（geometry）打包在一起，而且和sensor无关。所以它能当所有sensor数据的共同祖先。

这个philosophy其实挺普遍的：
- LLM用token当所有模态的中间表示
- NeRF用radiance field当scene的中间表示
- UniScene用occupancy当driving scene的中间表示

未来的方向我觉得是把这个framework做成 **action-conditioned**：你不只能生成scene，还能说"车往左转"，然后scene跟着变。这就是真正的world model了，不只是data generator。

另外一个有意思的点：Figure 11显示DiT的scaling law有效，意味着这个框架还有很大headroom。现在用的是中等大小的DiT，换成大模型occupancy generation质量还能继续涨，整个pipeline的天花板还没摸到。

---

# UniScene: 统一Occupancy-centric的自动驾驶场景生成

## 1. Core Motivation 与 Problem Formulation

这篇论文解决的是一个自动驾驶数据生成的根本问题。当前方法如MagicDrive、DriveDreamer、LiDARDM等都只生成单模态数据，从BEV layouts或者3D boxes直接生成RGB video或LiDAR。这种single-step的建模方式存在两个核心问题:

**问题1: Distribution Complexity Mismatch**
直接从coarse BEV layout建模到dense multi-view video/LiDAR的mapping $P(\text{Data}|\text{BEV})$ 是个ill-posed problem。BEV只有2D semantic + 2D geometric information, 而要生成的是完整的3D dynamic scene with viewpoint-dependent appearance。这个gap太大, 导致现有方法的FVD仍然在110+ (Vista*) 水平。

**问题2: Multi-modality Gap**
不同传感器数据有完全不同的representations: 
- Video: dense 2D pixels with view-dependent appearance
- LiDAR: sparse 3D points with reflectance
- Occupancy: dense 3D semantic voxels

没有shared intermediate representation, 每种模态都要单独建模, 计算资源浪费。

UniScene的核心insight: 用semantic occupancy作为meta representation, 把任务decompose为:

$$P(\text{Vid}, \text{Lid}|\text{BEV}) = P(\text{Vid}, \text{Lid}|\text{Occ}) \cdot P(\text{Occ}|\text{BEV})$$

这个factorization的关键在于: occupancy既包含semantic information (类别labels) 又包含geometric information (3D voxel structure), 是一种view-invariant的representation。从BEV生成occupancy相对容易(都是3D structure), 从occupancy生成video/LiDAR也相对容易(都是传感器观测), 分解后每一步的distribution complexity都降低了。

Reference: [UniScene Project Page](https://arlo0o.github.io/uniscene/)

## 2. 整体Architecture深度解析

### 2.1 Occupancy-centric Hierarchy

整个pipeline分为两个stage, 训练时也是两阶段:

**Stage 1: Train Occupancy Generation Model**
- 输入: BEV layout sequences + noise
- 输出: semantic occupancy sequences
- 模型: Occupancy VAE + Occupancy DiT

**Stage 2: Joint Train Video & LiDAR Generation**
- 固定Stage 1模型, 用它生成occupancy作为condition
- Video model: 基于SVD初始化, 加occupancy渲染条件
- LiDAR model: Sparse UNet + ray-based volume rendering

这种stage-wise训练的好处: Stage 1学的是geometry+semantics的abstract distribution, Stage 2学的是sensor-specific的rendering distribution。两个分布的性质不同, 分开学更stable。

### 2.2 数据预处理细节

这里有个重要细节: NuScenes原始的occupancy是2Hz的key-frame annotation。但video和LiDAR都是12Hz, 频率不匹配。作者的处理方式:

1. 用ASAP的12Hz interpolated annotations
2. Concatenate全scene的LiDAR points
3. 用NKSR算法reconstruct mesh
4. 提取mesh vertices, 根据LiDARseg和12Hz annotations assign semantic labels
5. 转换成occupancy格式

这个interpolation pipeline其实是个很好的工程trick, 可以reference [NKSR](https://github.com/nv-tlabs/nksr) 和 [ASAP benchmark](https://arxiv.org/abs/2212.08914)。

## 3. Occupancy Generation深度解析

### 3.1 Temporal-aware Occupancy VAE

这是整个系统最关键的component之一。之前的工作如OccSora, OccWorld, OccLLama都用VQVAE做discrete compression, 但VQVAE在高compression ratio下reconstruction质量急剧下降。UniScene用continuous VAE。

**VAE Encoder结构:**
- 输入: $O \in \mathbb{R}^{H \times W \times D}$ (3D occupancy)
- 转换成BEV: $\hat{O} \in \mathbb{R}^{H \times W \times D \cdot C'}$, 其中$C'$是learnable class embedding dimension (设为8)
- 2D CNN + 2D axial attention → downsampled latent $z_{occ} \in \mathbb{R}^{C \times h \times w}$
- $h = H/d$, $w = W/d$, $d$是downsampling factor

注意这里有个细节: encoder只用2D操作, 把3D occupancy展平成BEV-style的representation。这样在encoder阶段不考虑temporal信息, 保持flexibility。

**VAE Decoder结构:**
- 输入: $z_{occ}^{seq} \in \mathbb{R}^{T \times C \times h \times w}$ (时序latent sequence)
- 3D CNN + 3D axial attention → 重建$\hat{O}^{seq} \in \mathbb{R}^{T \times H \times W \times D \cdot C'}$
- Reshape成$\mathbb{R}^{THW \times D \times C'}$, 与class embedding做dot product得到logits
- argmax得到最终occupancy

这里temporal信息只在decoder中出现, 这是个借鉴自Align-Your-Latents的设计哲学: encoder保持spatial flexibility, decoder处理temporal correlation。

**VAE Loss (Eq. 1):**
$$\mathcal{L}_{occ}^{vae} = \mathcal{L}_{CE} + \lambda_1 \mathcal{L}_{LS} + \lambda_2 \mathcal{L}_{KL}$$

- $\mathcal{L}_{CE}$: Cross-entropy loss, 衡量voxel分类准确性
- $\mathcal{L}_{LS}$: Lovász-softmax loss, 这是IoU的surrogate loss, 解决class imbalance问题
- $\mathcal{L}_{KL}$: KL divergence, regularize latent space
- $\lambda_1, \lambda_2$: loss weights

[Lovász-softmax reference](https://arxiv.org/abs/1705.23587)

**Reconstruction Performance (Table 2):**
| Method | Compression Ratio | mIoU | IoU |
|--------|-------------------|------|-----|
| OccSora (VQVAE) | 512 | 27.4 | 37.0 |
| Ours (VAE) | 512 | 72.9 | 64.1 |
| OccWorld (VQVAE) | 16 | 65.7 | 62.2 |
| Ours (VAE) | 32 | 92.1 | 87.0 |

这个结果非常impressive: 在512x compression下, VAE比VQVAE高了45.5 mIoU。这解释了为什么后续generation质量能显著超越OccSora——VAE reconstruction bottleneck直接限制了generation upper bound。

Table 11里还有个有趣的comparison: 同样架构下, VQVAE在32x compression只有59.8 mIoU, 而VAE达到92.1。这证明continuous representation在occupancy这种structured data上确实有优势。

### 3.2 Latent Occupancy DiT

**Input Alignment:**
- BEV layout: $\mathbf{B}^i \in \mathbb{R}^{C_b \times H \times W}$ → downsample到$\mathbf{B}_{down}^i \in \mathbb{R}^{C_b \times h \times w}$
- Noise latent: $\mathbf{Z}_{occ}^i \in \mathbb{R}^{C_o \times h \times w}$
- Concatenate: $\mathbf{Z}_{cat} \in \mathbb{R}^{(C_o + C_b) \times h \times w}$
- Patchify: $\mathbf{Z} \in \mathbb{R}^{L \times E_d}$, $L$是patch数, $E_d$是embedding dimension

这种explicit alignment策略(early fusion, 把BEV和noise在input层面concat)比cross-attention条件化更直接, 模型更容易学习spatial relationships。

**DiT Backbone:**
- Spatial-Temporal Latent Diffusion Transformer
- Stacked spatial blocks + temporal blocks
- Spatial blocks: aggregate features across positions within same frame
- Temporal blocks: aggregate features across frames at same position
- 2D positional embeddings + 1D temporal embeddings

**DiT Loss (Eq. 2):**
$$\mathcal{L}_{occ}^{dit} = \mathbb{E}\left[\sum_{i=1}^{T} \|f_{dit}(z_{occ}^i, \mathbf{B}^i) - \epsilon_n^i\|^2\right]$$

- $z_{occ}^i$: $i^{th}$ frame的noisy latent
- $\mathbf{B}^i$: $i^{th}$ frame的BEV layout condition  
- $\epsilon_n^i \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$: 添加的Gaussian noise
- $f_{dit}$: model预测的noise

这是标准的epsilon-prediction形式。注意这里sum over $T$ frames, 所以是sequence-level的diffusion, 不是independent frame diffusion。

**Training tricks:**
- Random drop BEV condition with probability 0.1, enable unconditional generation (for editing)
- CFG scale默认1.0 (因为BEV condition已经explicit concat了)
- 8 frames during training
- DiT learning rate $1e-4$, 600 epochs
- VAE learning rate $1e-3$, 200 epochs

**Ablation Study (Table 8):**
| Method | mIoU | F3D | MMD |
|--------|------|-----|-----|
| Ours | 19.44 | 158.55 | 10.60 |
| w/o VAE 3D Axial Attention | 18.77 | 167.91 | 17.10 |
| w/o DiT Temporal Attention | 17.63 | 176.74 | 11.35 |
| w/o DiT Spatial Attention | 10.29 | 261.03 | 18.59 |

可以看到:
- Spatial attention最重要 (去掉F3D涨65%), 因为需要建模spatial coherence
- VAE 3D axial attention对MMD影响最大 (降低38%), 因为temporal consistency主要由decoder保证
- DiT temporal attention影响中等 (F3D涨10%)

### 3.3 Occupancy Forecasting Variant

为了和OccWorld/OccLLama对比, 作者改造成forecasting model:
- 训练: conditional frames (no noise) + future frames (with noise) 一起过DiT, loss只算future frames
- 推理: $T_c = 1$ 或 $2$ reference frames + $T_f = 6$ future frames
- 比OccWorld的$T_c = 5$少很多reference frames

**Results (Table 12):**
| Method | 1s mIoU | 2s mIoU | 3s mIoU | Avg |
|--------|---------|---------|---------|-----|
| OccWorld | 25.75 | 15.14 | 10.51 | 17.13 |
| OccLLama | 25.05 | 19.49 | 15.26 | 19.93 |
| Ours (1 ref) | 30.93 | 24.87 | 20.75 | 27.33 |
| Ours (2 ref) | 35.37 | 29.59 | 25.08 | 31.76 |

即使只用1个reference frame, mIoU avg都比OccWorld用5个reference frame高10.2 points。这说明continuous VAE + better DiT design的representation power远超VQVAE-based方法。

### 3.4 Scalability Analysis

Figure 11显示随着FLOPs增加, F3D持续下降, mIoU持续上升。这是DiT架构的scaling law特性, 和Peebles & Xie的Scalable Diffusion Models with Transformers结论一致。这暗示UniScene的occupancy generation headroom还很大, 只要用更大模型就能继续提升。

[DiT reference](https://arxiv.org/abs/2212.09748)

## 4. Video Generation深度解析

### 4.1 Gaussian-based Joint Rendering

这是我最喜欢的component之一。核心idea: occupancy是voxel-based representation, video是pixel-based representation, 中间需要bridge。作者用3D Gaussian Splatting做这个bridge。

**Gaussian Construction:**
- 输入: semantic occupancy $\in \mathbb{R}^{H \times W \times D}$
- 转换成Gaussian primitives: $\mathcal{G} = \{G_i\}_{i=1}^N$
- 每个Gaussian: position $\mu$ (voxel center), semantic label $s$ (voxel category), opacity $\alpha$ (set to 1), covariance $\Sigma$ (default rotation + voxel size scaling)

这是个simplification: 不像原始3DGS那样学习每个Gaussian的参数, 这里直接用voxel的structural prior定义Gaussian。所以rendering几乎是free的, 不需要training。

**Rendering Equations:**

Depth map (Eq. 3):
$$\mathbf{D} = \sum_{i \in N} d_i \alpha_i' \prod_{j=1}^{i-1}(1-\alpha_j')$$

Semantic map (Eq. 4):
$$\mathbf{S} = \text{argmax}\left(\sum_{i \in N} \text{onehot}(s_i) \alpha_i' \prod_{j=1}^{i-1}(1-\alpha_j')\right)$$

- $d_i$: $i^{th}$ Gaussian的depth value (from camera)
- $\alpha_i'$: 投影到2D后的opacity, 由3D opacity $\alpha$和projected 2D Gaussian共同决定
- $\prod_{j=1}^{i-1}(1-\alpha_j')$: front-to-back compositing的transmittance
- $\text{onehot}(s_i)$: semantic label的one-hot encoding

这里用了tile-based rasterization (来自3DGS), 所以渲染速度很快。注意semantic map用argmax而不是weighted sum, 这保证了semantic label的discrete性质。

[3DGS reference](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

**Why this works so well:**
传统multi-view video generation用spatial-temporal attention保证cross-view consistency, 计算量巨大且容易出错。这里通过共享的3D occupancy → 4个camera view的rendering, 物理上保证了cross-view consistency。深度和semantic都来自同一个3D representation, 不可能不一致。

### 4.2 ControlNet-style Conditioning

Rendered maps通过ControlNet-style encoder注入到video diffusion UNet:
- 两个encoder branch: 一个for depth, 一个for semantic
- 每个branch: 2 ResNet blocks + zero convolution
- Zero conv初始化为0, 保留SVD预训练能力
- 与SVD的cross-attention一起作为condition

这种设计借鉴自ControlNet, 保留pre-trained SVD的generation power, 同时加入spatial-precise conditioning。

[ControlNet reference](https://arxiv.org/abs/2302.05543)

### 4.3 Geometric-aware Noise Prior

这是个很巧妙的trick。SVD默认的image-to-video生成里, condition frame的latent直接channel-wise concat到noise input。UniScene进一步用depth information做geometric warping。

**基础形式 (Eq. 5):**
$$\epsilon_{vid}^i = \lambda z_c + \epsilon_n^i$$

- $z_c$: conditional reference frame的VAE latent
- $\epsilon_n^i \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$: random noise
- $\lambda$: balancing coefficient

这相当于把reference frame的appearance prior注入到noise中, 但忽略了geometric change。

**Geometric-aware form (Eq. 6-7):**
$$\epsilon_{vid}^i = \lambda \text{Warp}(z_c, \mathbf{D}^i, \mathbf{K}, [\mathbf{R}_{0,i}|\mathbf{t}_{0,i}]) + \epsilon_n^i$$

Warp operation (Eq. 7):
$$p_0 = \mathbf{K} \cdot (\mathbf{R}_{0,i} \cdot (\mathbf{K}^{-1} \cdot p_i \cdot \mathbf{D}^i(u,v)) + \mathbf{t}_{0,i})$$

- $p_i = (u, v, 1)^T$: $i^{th}$ framelatent feature中的pixel coordinate
- $p_0$: 对应到conditional latent $z_c$中的coordinate
- $\mathbf{K}$: camera intrinsics (scaled to latent space)
- $[\mathbf{R}_{0,i}|\mathbf{t}_{0,i}]$: 从target $z_i$到conditional $z_c$的transform matrix
- $\mathbf{D}^i(u,v)$: $i^{th}$ frame的rendered depth at $(u,v)$

这个warp本质是: 给定target view的depth和relative pose, 把target view的每个pixel reproject回reference view, 找到对应的appearance prior。这样dynamic regions也能正确建模motion。

**Video Training Loss (Eq. 8):**
$$\mathcal{L}_{vid} = \mathbb{E}\left[\sum_{i=1}^{T}(1-m^i) \cdot \|f_{vid}(z_{vid}^i, t, z_c, \mathbf{D}^i, \mathbf{S}^i) - z_0^i\|^2\right]$$

- $f_{vid}$: video generation model
- $z_{vid}^i$: noisy input latent
- $t$: text prompt (weather, time of day等)
- $z_c$: randomly selected conditional reference frame
- $\mathbf{D}^i, \mathbf{S}^i$: rendered depth and semantic maps
- $z_0^i$: ground truth clean latent
- $m$: one-hot mask选择condition frames (这些frames不参与loss计算)
- $(1-m^i)$: 只对non-condition frames计算loss

Random selection of $z_c$很重要: 防止模型over-rely on特定frame, 增加robustness。

**Ablation Study (Table 9):**
| Method | FID | FVD |
|--------|-----|-----|
| Ours | 6.12 | 70.52 |
| w/ Spatial-temporal Attention | 12.72 | 110.87 |
| w/o Rendered Semantic Map | 11.72 | 107.92 |
| w/o Rendered Depth Map | 10.17 | 102.42 |
| w/o Depth-based Noise Prior | 7.23 | 87.52 |

关键发现:
- Occupancy-based conditional guidance比spatial-temporal attention好太多 (FVD: 70.52 vs 110.87)
- Semantic map比depth map更重要 (FVD: 102.42 vs 107.92), 但两者都重要
- Geometric-aware noise prior贡献了17 FVD improvement

### 4.4 Video Generation Results

**Main Results (Table 4):**
| Method | Multi-view | Video | FID | FVD |
|--------|-----------|-------|-----|-----|
| MagicDrive | ✓ | ✓ | 16.20 | - |
| Drive-WM | ✓ | ✓ | 15.80 | 122.70 |
| Vista* (multi-view impl) | ✓ | ✓ | 13.97 | 112.65 |
| Ours (Gen Occ) | ✓ | ✓ | 6.45 | 71.94 |
| Ours (GT Occ) | ✓ | ✓ | 6.12 | 70.52 |

Ours (Gen Occ)用生成的occupancy, Ours (GT Occ)用ground truth occupancy。两者差距很小 (FVD: 71.94 vs 70.52), 说明occupancy generation质量已经很高, 不是bottleneck。

FVD比Vista*低40+, 这是巨大提升。考虑到Vista是single-view SOTA, 这里multi-view version用spatial-temporal attention impl, 性能下降是expected的。

## 5. LiDAR Generation深度解析

### 5.1 Prior Guided Sparse Modeling

LiDAR和occupancy都是3D representation, 但LiDAR是sparse points, occupancy是dense voxels。这里的关键是efficiency: 不能对每个voxel都做neural rendering。

**Sparse UNet:**
- 输入: semantic occupancy grids
- 用submanifold sparse convolution (只在occupied voxels计算)
- 输出: sparse voxel features

这借鉴自Part-A2 net的设计, 极大减少计算量。Table 10显示去掉Sparse UNet, JSD从0.072涨到0.097, 内存只省0.11GB, 说明Sparse UNet的cost-effective很高。

[Sparse UNet reference](https://arxiv.org/abs/1907.03670)

### 5.2 Ray-based Sparse Sampling

**Core Idea:**
对每条LiDAR ray, 不需要在整个ray上均匀采样, 而是根据occupancy prior只在可能hit surface的地方采样。

- 对LiDAR ray $r$ uniform采样生成候选points $s$
- 用occupancy定义PDF: occupied voxel内概率=1, 其他=0
- 基于PDF resample $n$ points $\{\mathbf{r}_i = o + s_i v\}_{i=1}^n$
  - $o$: ray origin
  - $v$: normalized ray direction

这避免了在empty space浪费计算。Table 10显示这个策略降低58.94%时间消耗, 还略微提升JSD。

### 5.3 Volume Rendering for LiDAR

每个采样point feature $u_i$通过MLP预测SDF value $f(s)$, 然后计算weights:

**Weight computation (Eq. 9-10):**
$$\beta_i = \max\left(\frac{\Phi_s(f(\mathbf{r}(s_i))) - \Phi_s(f(\mathbf{r}(s_{i+1})))}{\Phi_s(f(\mathbf{r}(s_i)))}, 0\right)$$

$$w(s_i) = \prod_{j=1}^{i-1}(1-\beta_j)\beta_i, \quad h = \sum_{i=1}^{n} w(s_i)s_i$$

- $\Phi_s(x) = (1 + e^{-sx})^{-1}$: sigmoid function with sharpness parameter $s$
- $f(\mathbf{r}(s_i))$: SDF value at point $s_i$ on ray $\mathbf{r}$
- $\beta_i$: 类似NeuS的density, 但基于SDF differential
- $w(s_i)$: 渲染weight, front-to-back compositing
- $h$: 最终depth

这是NeuS的volume rendering formulation, 用SDF的差分近似density, 适合solid surface reconstruction。

[NeuS reference](https://arxiv.org/abs/2106.10689)

**Ray feature aggregation (Eq. 19):**
$$v_r = \sum_{i=1}^{n} w_i \cdot u_i$$

- $u_i$: $i^{th}$ point的feature (from Sparse UNet)
- $w_i$: rendering weight
- $v_r$: aggregated ray feature

$v_r$通过MLP预测intensity和ray-drop probability。

### 5.4 Ray-dropping Head

真实LiDAR有ray-dropping现象: 由于material reflectance, distance, incidence angle等原因, 部分ray没有return。这是LiDAR realism的关键。

UniScene加一个ray-dropping head预测每条ray被drop的概率, 训练时作为binary classification。Table 10显示这个head降低25.92% MMD和25.00% JSD, 是LiDAR realism的关键component。

**LiDAR Loss (Eq. 11):**
$$\mathcal{L}_{lid} = \mathcal{L}_{depth} + \lambda_1 \mathcal{L}_{inten} + \lambda_2 \mathcal{L}_{drop}$$

- $\mathcal{L}_{depth}$: depth prediction loss
- $\mathcal{L}_{inten}$: intensity prediction loss
- $\mathcal{L}_{drop}$: ray-drop classification loss
- $\lambda_1, \lambda_2$: balancing coefficients

### 5.5 LiDAR Generation Results

**Main Results (Table 5):**
| Method | MMD ($10^{-4}$) | JSD | Time (s) |
|--------|----------------|-----|----------|
| LiDARDM | 3.51 | 0.118 | 45.12 |
| Open3D | 8.15 | 0.149 | 2.39 |
| Ours (Gen Occ) | 2.40 | 0.108 | 0.47 |
| Ours (GT Occ) | 1.53 | 0.072 | 0.25 |

UniScene在质量上超越LiDARDM, 速度上快100倍。这是prior-guided sparse modeling的胜利。

**Diverse Beam Scanning (Figure 16):**
训练时只用32-beam, 但推理时可以生成16/32/64-beam, 不需要retrain。这是ray-based formulation的天然优势: ray的数量和pattern不影响model architecture。

## 6. Downstream Task Evaluation

### 6.1 Occupancy Prediction (Table 6)

用UniScene生成数据augment训练集, 测试CONet baseline:
| Setting | Baseline mIoU | w/ UniScene | Δ |
|---------|----------------|-------------|---|
| Camera (C) | 12.8 | 16.5 | +3.7 |
| LiDAR (L) | 15.8 | 19.3 | +3.5 |
| Camera+LiDAR | 20.1 | 23.9 | +3.8 |

Camera modality提升最大, 因为生成的video质量最高。Multi-modal fusion也获得提升, 因为occupancy-based guidance同时提升video和LiDAR质量。

### 6.2 BEV Segmentation & 3D Detection (Table 7)

| Method | Road mIoU | Vehicle mIoU | mAP | NDS |
|--------|-----------|--------------|-----|-----|
| Baseline | 74.30 | 36.00 | 32.88 | 37.81 |
| w/ MagicDrive | 79.56 | 40.34 | 35.40 | 39.76 |
| w/ Ours | 81.69 | 41.62 | 36.50 | 41.22 |

UniScene在所有metric上都超过MagicDrive, 这是end-to-end advantage: 不只是video质量好, 下游任务也benefit更多。

## 7. 我的Insights和Further Thoughts

### 7.1 关于Decomposed Generation Philosophy

UniScene的success本质上验证了一个重要principle: **复杂生成任务应该用intermediate representation分解**。这个idea在别处也有体现:
- LDM用VAE latent space做diffusion (pixel→latent)
- ControlNet用spatial condition做precise control
- DALL-E 3用GPT生成prompt (intent→prompt)
- UniScene用occupancy做3D intermediate (BEV→Occ→Sensors)

这个pattern值得generalize: 找一个semantically meaningful, geometrically informative, modality-agnostic的intermediate representation是关键。

### 7.2 关于VQVAE vs VAE for Structured Data

UniScene证明在structured data (occupancy)上, continuous VAE远胜VQVAE。这有意思, 因为在natural images上VQVAE也很work (DALL-E, Stable Diffusion早期尝试)。

可能的原因:
- Occupancy是categorical, 但spatial structure是continuous
- VQVAE的codebook collapse问题在长尾categories上更严重
- Continuous latent保留更多spatial detail

但VAE的drawback是generation时无法用categorical sampling tricks。也许VQ-VAE + continuous refinement是future方向。

### 7.3 关于Gaussian Splatting as Bridge

UniScene用simplified 3DGS做occupancy→images的bridge。这让我想到一个更ambitious的方向: 如果occupancy generation直接output learnable Gaussians (rather than discrete voxels), 可以直接rendering到任意sensor modality。

最近有些工作如[GS-LiDAR](https://arxiv.org/abs/2402.03974)已经探索Gaussian-based LiDAR synthesis。UniScene + GS-LiDAR的组合可能是更unified的framework。

### 7.4 关于Temporal Consistency

UniScene的temporal consistency主要由VAE decoder的3D axial attention保证。但有个potential issue: 长horizon generation (8 frames)的drift问题。Figure 5显示coherent generation, 但没quantify长sequence的quality degradation。

Future work可能需要:
- Autoregressive generation with sliding window
- Hierarchical temporal modeling (coarse-to-fine)
- Test-time guidance with physical constraints

### 7.5 关于Cross-dataset Generalization

Figure 19显示UniScene可以直接transfer到Waymo和nuPlan。这是occupancy作为intermediate representation的另一个advantage: occupancy的distribution比raw video更universal。

但需要注意: Waymo和nuPlan的camera setup和NuScenes不同, 所以需要conditional reference image做appearance guidance。这暗示UniScene的generalization还有改进空间。

### 7.6 关于Scaling

Figure 11显示DiT scaling有效。这意味着:
- 更大的DiT可能进一步提升occupancy generation quality
- 但VAE的reconstruction capacity可能是上限
- VAE本身也应该consider scaling (更深的encoder/decoder)

### 7.7 关于Embodied AI Application

论文最后提到extending to embodied intelligence和robotics。我认为这是对的方向:
- Robotics同样需要multi-modal synthetic data
- Occupancy是robot manipulation的natural representation
- 但robotics scene更complex (articulated objects, deformable objects)

可能需要extend occupancy representation到part-level或者object-centric。

### 7.8 关于Limitations

论文提到unifying system资源消耗大。具体来说:
- Stage 1训练: 16 batch size, 200 + 600 epochs
- Stage 2训练: 96 batch size, 90 epochs
- 8 frames per sequence
- 需要A100 GPUs

这个compute budget对academic lab不friendly。Lightweight deployment (e.g., distillation, pruning)是future work。

### 7.9 与World Models的关系

UniScene严格说是generative model, 不是world model。但occupancy forecasting variant (Ours-Fore.)其实是个world model: 给定past occupancy, 预测future occupancy。

这和DriveDreamer, WoVoGen, GAIA-1等world model方向有intersection。区别是: UniScene用occupancy作为state representation, 而非latent video features。这是个更好的choice: occupancy physical meaningful, 可以用planning algorithms直接interact。

### 7.10 关于Action Conditioning

UniScene目前conditioning是BEV layout, 这是static snapshot。如果要extend到action-conditioned generation (e.g., "turn left"), 需要action representation。

一个natural extension: 把ego action (trajectory)作为conditioning, 让occupancy generation考虑action-dependent scene evolution。这和world model的action-conditioned prediction更接近。

## 8. Reference Links

- [UniScene Project Page](https://arlo0o.github.io/uniscene/)
- [Stable Video Diffusion](https://arxiv.org/abs/2311.15127)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [NeuS](https://arxiv.org/abs/2106.10689)
- [Diffusion Transformer (DiT)](https://arxiv.org/abs/2212.09748)
- [ControlNet](https://arxiv.org/abs/2302.05543)
- [OccWorld](https://arxiv.org/abs/2311.16038)
- [OccSora](https://arxiv.org/abs/2405.20337)
- [MagicDrive](https://arxiv.org/abs/2310.02601)
- [LiDARDM](https://arxiv.org/abs/2404.02903)
- [Vista](https://arxiv.org/abs/2405.17398)
- [NuScenes](https://www.nuscenes.org/)
- [OpenOccupancy](https://github.com/JeffWang987/OpenOccupancy)
- [NKSR](https://github.com/nv-tlabs/nksr)
- [Lovász-Softmax Loss](https://arxiv.org/abs/1705.23587)
- [Sparse Convolution](https://arxiv.org/abs/1907.03670)
- [Align-Your-Latents](https://arxiv.org/abs/2304.08877)
- [DriveDreamer-2](https://arxiv.org/abs/2403.06845)
- [Drive-WM](https://arxiv.org/abs/2311.16038)
- [BEVFusion](https://arxiv.org/abs/2205.05495)

## 9. Final Thoughts

UniScene是个非常well-executed的工作。它的核心contribution其实不是novel architecture, 而是novel **problem decomposition philosophy**: 用semantic occupancy作为unified intermediate representation, 把multi-modal driving scene generation从一个monolithic hard problem分解成两个tractable sub-problems。

这个philosophy可以extend到其他领域:
- Medical imaging: 用anatomical structures作为intermediate
- Indoor scene: 用room layout + object placements作为intermediate
- Robotics: 用object-centric states作为intermediate

我个人觉得最exciting的方向是把UniScene和action-conditioned world model结合, 实现action-controllable scene generation。这样不只是data generation, 而是真正的embodied simulation环境。

另外一个值得探索的方向是implicit occupancy representation, 比如NeRF-style或者occupancy field, 这样可以摆脱voxel resolution的限制。但这需要重新设计VAE和DiT architecture。

总之, UniScene是个solid的work, 提供了一个unified framework的blueprint, 后续工作可以在它的基础上做很多extension。关键insight是: **找对intermediate representation是multi-modal generation的关键**。
