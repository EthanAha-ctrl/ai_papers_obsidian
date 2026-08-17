---
source_pdf: UniScene.pdf
paper_sha256: e1e9b3bb8d200b745c4745c0efb51d38b8811a48fc17fdc897e5030acb985fe1
processed_at: '2026-08-12T20:10:14-07:00'
target_folder: Automobile
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说 UniScene

## 核心问题

自动驾驶需要训练数据，但采集真实数据贵、标注更贵。所以大家想用生成模型合成数据。问题是：

**现有方法都是单打一**——要么生成video，要么生成LiDAR，要么生成occupancy。而且直接从BEV layout（2D的road layout + vehicle boxes这种粗略信息）跳到最终输出，步子太大，生成质量一般。

## UniScene的核心idea

与其一步到位，不如分两步走：

**Step 1**: 先从BEV layout生成semantic occupancy（3D voxel grid，每个voxel有semantic label）

**Step 2**: 再从occupancy生成video和LiDAR

### 为什么这么拆？

用概率的话说：
$$P(Vid, Lid | BEV) = P(Vid, Lid | Occ) \cdot P(Occ | BEV)$$

左边是从粗到细的复杂mapping，右边是两个更简单的mapping的乘积。

用大白话说：
- **BEV → Occupancy**：从粗略2D layout到3D semantic structure，信息量增加但还在geometric domain
- **Occupancy → Video/LiDAR**：从3D structure到不同sensor的output，本质是rendering + appearance synthesis

每个step都更tractable，模型不用一次学太多东西。

### 为什么选occupancy当中间桥梁？

- 比BEV map信息多：有完整3D结构 + semantics
- 比video/LiDAR更abstract：没有appearance和sensor noise，更容易生成
- 天然bridge两个output：可以render成2D maps（给video用），可以sample成sparse points（给LiDAR用）

## 三个生成模块怎么work的

### Module 1: Occupancy Generation

**挑战**：3D occupancy维度高，直接diffusion计算量大。

**解法**：
1. **VAE压缩**：先把occupancy压到latent space。关键选择是用continuous VAE而不用discrete VQVAE——因为VQVAE在高压缩比下codebook会collapse，信息损失严重。Table 2显示同样512x压缩，VAE的mIoU 72.9 vs VQVAE 55.8，差距巨大。

2. **DiT生成**：在latent space用Diffusion Transformer做生成，条件是BEV layout。用spatial + temporal attention同时建模空间和时间关系。

**Key design**：VAE的encoder是per-frame的，但decoder是3D的——这样temporal信息只在decode阶段引入，encoder保持simple。

### Module 2: Video Generation

**挑战**：如何用3D occupancy引导2D video生成？如何保证multi-view consistency？

**传统做法**：用spatial-temporal attention让不同view之间互相attend（MagicDrive, Drive-WM）。计算贵，效果一般。

**UniScene做法**：用Gaussian Splatting把occupancy render成multi-view的depth map + semantic map，作为ControlNet-style条件喂给SVD。

**为什么work**：
- Depth map告诉model每个pixel的3D位置 → geometric guidance
- Semantic map告诉model每个pixel是什么 → semantic guidance  
- 多个view从同一个3D occupancy render出来 → 天生consistent

**额外trick - Geometric-aware noise prior**：sampling时，把reference frame的appearance warp到target frame（用depth做reprojection），这样moving object的prior能正确对齐。

### Module 3: LiDAR Generation

**挑战**：LiDAR point cloud很sparse，大部分3D空间是空的。直接dense computation浪费。

**解法**：
1. **Sparse UNet**：只在occupied voxel上算，用submanifold sparse convolution
2. **Prior-guided sampling**：沿每条LiDAR ray采样点时，只在occupied voxel附近采，不浪费时间在空旷区域
3. **Volume rendering**：用NeuS-style的SDF + volume rendering预测每个ray的depth
4. **Physics modeling**：额外预测reflection intensity和ray-dropping（模拟真实LiDAR beam没反射回来的情况）

**效果**：推理速度0.25s vs LiDARDM的45s，快180倍，质量还好。

## 实验结果说了什么

### Generation质量

| 任务 | UniScene | 之前SOTA | 提升 |
|------|----------|----------|------|
| Video (FVD) | 70.52 | 112.65 (Vista*) | -37% |
| LiDAR (MMD) | 1.53 | 3.51 (LiDARDM) | -56% |
| Occupancy (mIoU) | 31.76 | 25.75 (OccWorld) | +23% |

### 关键发现：Gen Occ ≈ GT Occ

用生成的occupancy vs ground truth occupancy做video/LiDAR生成，质量几乎一样：

- Video FVD: 71.94 (Gen Occ) vs 70.52 (GT Occ) → 差距1.4
- LiDAR MMD: 2.40 vs 1.53 → 差距0.87

这说明occupancy generation质量足够好，不成为bottleneck。这是整个decomposed learning方案成功的最关键证据。

### Downstream task提升

用生成的数据augment训练：
- Occupancy prediction: IoU +8.5 (camera), +5.9 (multi-modal)
- BEV segmentation: road mIoU +7.4, vehicle mIoU +5.6
- 3D detection: mAP +3.6, NDS +3.4

全面超过用其他方法（BEVGen, Vista*, MagicDrive）做augmentation的效果。

## Controllable能力

1. **几何编辑**：修改BEV layout（比如删掉某辆车），occupancy、video、LiDAR都会相应变化。用DDIM Inversion保持unchanged部分的consistency。

2. **属性控制**：text prompt控制weather（sunny/rainy）、time of day（day/night）。

3. **泛化**：在Waymo、nuPlan上直接transfer也能work。

## 一句话总结

**把复杂的driving scene生成拆成两步：先从BEV生成3D occupancy作为"中间语言"，再用这个中间语言分别生成video和LiDAR。每个步骤都更简单，最终三个输出都比之前SOTA好，而且中间产物质量足够高不拖后腿。**

关键的技术创新：
- Continuous VAE instead of VQVAE for occupancy compression
- Gaussian Splatting rendering把3D occupancy转成2D条件
- Sparse modeling把LiDAR的sparsity变成computational advantage
- Depth-based noise prior保证temporal consistency

---

# UniScene: 统一驾驶场景生成框架深度解析

## 一、核心Intuition: 为什么需要Occupancy-centric的层次化建模？

这篇paper的核心insight可以用一个简单的概率分解来理解。直接从BEV layout生成video/LiDAR需要建模的分布是：

$$P(Vid, Lid | BEV)$$

这个分布极其复杂——BEV layout只是2D的粗略几何信息（road layout, vehicle boxes），而video是高维的pixel空间，LiDAR是3D点云。直接学这个mapping，模型需要同时encode geometry、appearance、dynamics、sensor physics，burden太重。

UniScene的key idea是把这个complex distribution分解成两个更tractable的step：

$$P(Vid, Lid | BEV) = P(Vid, Lid | Occ) \cdot P(Occ | BEV)$$

这里 $Occ$ 是semantic occupancy——一个3D voxel grid，每个voxel有semantic label。为什么选occupancy作为intermediate representation？三个reasons：

1. **Rich semantics + geometry**: 相比BEV map只有2D layout，occupancy有完整的3D结构 + 每个voxel的semantic class（road, vehicle, building等）
2. **Controllable**: 通过编辑BEV layout可以controllable地生成不同occupancy
3. **Transferable**: Occupancy可以render成2D maps（video guidance），也可以sample成sparse points（LiDAR guidance）

这种decomposed learning的思路类似于Latent Diffusion——把高维生成问题压缩到更compact的latent space，降低优化难度。

## 二、Stage I: Controllable Occupancy Generation

### 2.1 Temporal-aware Occupancy VAE

首先要解决occupancy的compression问题。3D occupancy $O \in \mathbb{R}^{H \times W \times D}$ 维度很高，直接在voxel space做diffusion计算量巨大。

**Encoder设计**：
- 把3D occupancy $O \in \mathbb{R}^{H \times W \times D}$ reshape成BEV表示 $\hat{O} \in \mathbb{R}^{H \times W \times DC'}$，其中 $C'$ 是learnable class embedding维度（设为8）
- 用2D CNN + 2D axial attention下采样得到latent $Z_{occ} \in \mathbb{R}^{C \times h \times w}$，$h = H/d$, $w = W/d$，$d$ 是down-sampling factor
- 注意encoder是per-frame的，**temporal信息只在decoder引入**

**Decoder设计**：
- 输入temporal latent sequence $z_{occ}^{seq} \in \mathbb{R}^{T \times C \times h \times w}$
- 用3D CNN + 3D axial attention上采样回BEV representation $\hat{O}^{seq} \in \mathbb{R}^{T \times H \times W \times DC'}$
- reshape成 $\mathbb{R}^{THW \times D \times C'}$，与class embedding做dot product得到logits
- argmax得到最终occupancy sequence $O^{seq} \in \mathbb{R}^{T \times H \times W \times D}$

**VAE Loss**：
$$\mathcal{L}_{occ}^{vae} = \mathcal{L}_{CE} + \lambda_1 \mathcal{L}_{LS} + \lambda_2 \mathcal{L}_{KL}$$

- $\mathcal{L}_{CE}$: Cross-entropy，标准分类loss
- $\mathcal{L}_{LS}$: Lovász-softmax loss，直接优化IoU的surrogate，对occupancy这种dense prediction任务很重要
- $\mathcal{L}_{KL}$: KL divergence正则化latent space到 $\mathcal{N}(0, I)$
- $\lambda_1, \lambda_2$: balancing weights

**为什么用VAE不用VQVAE？** Table 2的对比很说明问题：

| Method | Compression Ratio | mIoU | IoU |
|--------|------------------|------|-----|
| OccSora (VQVAE) | 512 | 27.4 | 37.0 |
| Ours (VQVAE) | 512 | 55.8 | 56.8 |
| Ours (VAE) | 512 | **72.9** | **64.1** |

VQVAE的discrete codebook在high compression ratio下会损失大量信息（codebook collapse问题），而VAE的continuous latent space能更好preserve spatial details。这对后续generation quality至关重要——如果reconstruction都不行，generation的上限就被限死了。

### 2.2 Latent Occupancy DiT

**核心架构**：基于DiT (Diffusion Transformer)做latent diffusion。

**Input处理**：
1. BEV layout $B^i$ 下采样到 $B_{down}^i \in \mathbb{R}^{C_b \times h \times w}$，与latent shape对齐
2. 与noisy latent $Z_{occ}^i \in \mathbb{R}^{C_o \times h \times w}$ concatenate得到 $Z_{cat} \in \mathbb{R}^{(C_o+C_b) \times h \times w}$
3. Unified patch embedder转成tokens $Z \in \mathbb{R}^{L \times E_d}$，$L$ 是patch数，$E_d$ 是embedding dim

**Backbone**: Spatial-Temporal Latent Diffusion Transformer
- Spatial blocks: 聚合同一帧内不同position的feature
- Temporal blocks: 捕获不同帧同一position的feature
- 2D positional embeddings + 1D temporal embeddings

**DiT Loss**:
$$\mathcal{L}_{occ}^{dit} = \mathbb{E}\left[\sum_{i=1}^{T} \| f_{dit}(z_{occ}^i, B^i) - \epsilon_n^i \|^2\right]$$

- $f_{dit}(z_{occ}^i, B^i)$: model预测的noise
- $z_{occ}^i$: 第$i$帧的noisy latent
- $B^i$: 第$i$帧的BEV layout condition
- $\epsilon_n^i \sim \mathcal{N}(0, I)$: ground truth noise
- $T$: 总帧数（训练时固定为8）

**Classifier-Free Guidance**: 训练时以0.1概率随机drop掉BEV condition，inference时CFG scale=1.0（即不做guidance，可能是因为layout condition已经足够strong）

**Scalability**: Figure 11显示F3D和mIoU都随FLOPs增加而improve，符合DiT的scaling law（Peebles & Xie, ICCV 2023）。

### 2.3 Ablation Insights (Table 8)

| Design | Effect |
|--------|--------|
| VAE 3D Axial Attention | MMD -38.01% (temporal consistency) |
| DiT Temporal Attention | F3D -10.29% |
| DiT Spatial Attention | F3D -39.26% (最关键) |

Spatial attention最重要——因为occupancy的spatial structure很complex，需要attention建模long-range dependency。

## 三、Stage II: Video Generation via Gaussian-based Joint Rendering

### 3.1 从Occupancy到条件信号

核心challenge：如何把3D occupancy转换成video diffusion model能用的2D条件？

**Naive approach**: 用spatial-temporal attention保证multi-view consistency（MagicDrive, Drive-WM的做法）。但computation expensive且效果一般。

**UniScene approach**: 用Gaussian Splatting把occupancy render成multi-view的semantic map和depth map，作为ControlNet-style的条件。

### 3.2 Gaussian-based Joint Rendering详解

**Step 1: Occupancy → 3D Gaussians**
每个occupancy voxel转换成一个Gaussian primitive $G_i$，属性包括：
- Position $\mu_i$: voxel center
- Semantic label $s_i$: voxel的semantic class
- Opacity $\alpha_i$: 默认设为occupied=1
- Covariance $\Sigma_i$: 由default rotation + voxel size scaling计算

**Step 2: 3D → 2D Projection**
Gaussian Splatting的核心公式，3D covariance变换到camera coordinate：

$$\Sigma_i' = J W \Sigma_i W^T J^T$$

- $W$: viewing transformation（world to camera）
- $J$: Jacobian of affine approximation of projective transformation
- $\Sigma_i'$: 2×2 covariance in camera coordinates

Projected 2D Gaussian的opacity：
$$\alpha' = \alpha \exp\left(-\frac{1}{2}(x-\mu)^T (\Sigma')^{-1} (x-\mu)\right)$$

- $x \in \mathbb{R}^{2\times 1}$: pixel position in camera coordinates
- $\mu \in \mathbb{R}^{2\times 1}$: projected Gaussian center

**Step 3: Tile-based Rasterization**

**Depth Map**:
$$\mathbf{D} = \sum_{i \in N} d_i \alpha_i' \prod_{j=1}^{i-1}(1-\alpha_j')$$

- $d_i$: depth value of $i$-th Gaussian
- $\alpha_i'$: projected opacity
- $\prod_{j=1}^{i-1}(1-\alpha_j')$: transmittance（前面所有Gaussian都没挡住的概率）
- 按depth排序累加

**Semantic Map**:
$$\mathbf{S} = \arg\max\left(\sum_{i \in N} \text{onehot}(s_i) \alpha_i' \prod_{j=1}^{i-1}(1-\alpha_j')\right)$$

- $\text{onehot}(s_i)$: semantic label的one-hot encoding
- 累加weighted one-hot vectors后取argmax
- 这样能处理多个Gaussian重叠的情况

**Key insight**: Depth map提供explicit geometric guidance，semantic map提供explicit semantic guidance。这两个map对每个camera view都render一遍，保证multi-view consistency通过shared 3D occupancy自然achieved。

### 3.3 ControlNet-style Conditioning

Rendered maps通过encoder branch注入SVD (Stable Video Diffusion)：
- 两个encoder module：一个处理depth，一个处理semantic
- 每个module: 2个ResNet blocks + zero convolution
- Zero conv初始化保证训练初期不破坏pre-trained SVD的generation能力

### 3.4 Geometric-aware Noise Prior

这是一个sampling-time的trick，进一步enhance temporal consistency。

**Naive version**:
$$\epsilon_{vid}^i = \lambda z_c + \epsilon_n^i$$

- $z_c$: conditional frame的latent feature（通过3D video VAE encoder得到）
- $\epsilon_n^i \sim \mathcal{N}(0, I)$: random noise
- $\lambda$: balancing coefficient
- 直接把conditional frame的appearance prior加到noise里

**问题**: 动态区域（moving cars等）在不同帧变化大，简单复制appearance prior不correct。

**Geometric-aware version**:
$$\epsilon_{vid}^i = \lambda \text{Warp}(z_c, \mathbf{D}^i, \mathbf{K}, [\mathbf{R}_{0,i}|\mathbf{t}_{0,i}]) + \epsilon_n^i$$

**Depth-based Warp**:
$$p_0 = \mathbf{K} \cdot (\mathbf{R}_{0,i} \cdot (\mathbf{K}^{-1} \cdot p_i \cdot \mathbf{D}^i(u,v)) + \mathbf{t}_{0,i})$$

- $p_i = (u, v, 1)^T$: 第$i$帧latent中的像素坐标
- $p_0$: 对应到conditional frame的像素坐标
- $\mathbf{K}$: camera intrinsics（scaled to latent resolution）
- $[\mathbf{R}_{0,i}|\mathbf{t}_{0,i}]$: 从frame $i$到conditional frame的extrinsic
- $\mathbf{D}^i(u,v)$: 第$i$帧rendered depth在$(u,v)$处的值

**Intuition**: 用depth做geometric reprojection，把conditional frame的appearance warp到target frame的视角，这样moving object的appearance prior就能正确对齐。

### 3.5 Video Training Loss

$$\mathcal{L}_{vid} = \mathbb{E}\left[\sum_{i=1}^{T} (1-m^i) \cdot \| f_{vid}(z_{vid}^i, t, z_c, \mathbf{D}^i, \mathbf{S}^i) - z_0^i \|^2\right]$$

- $f_{vid}$: video diffusion UNet输出
- $t$: text prompt（weather, time of day等attribute control）
- $z_c$: conditional reference frame latent
- $\mathbf{D}^i, \mathbf{S}^i$: 第$i$帧的rendered depth和semantic map
- $z_0^i$: ground truth clean latent
- $z_{vid}^i$: noisy input latent
- $m^i$: one-hot mask选择conditional frames（不计算loss）
- 随机选$z_c$避免对特定帧的over-fitting

### 3.6 Ablation Insights (Table 9)

| Setting | FID↓ | FVD↓ |
|---------|------|------|
| Ours | 6.12 | 70.52 |
| w/ Spatial-temporal Attention | 12.72 | 110.87 |
| w/o Rendered Semantic Map | 11.72 | 107.92 |
| w/o Rendered Depth Map | 10.17 | 102.42 |
| w/o Depth-based Noise Prior | 7.23 | 87.52 |

Occupancy-based conditional guidance比spatial-temporal attention好得多（FVD 70.52 vs 110.87），证明了3D representation作为intermediate的value。

## 四、Stage II: LiDAR Generation via Prior-guided Sparse Modeling

### 4.1 为什么LiDAR生成需要special design？

LiDAR point cloud的特殊性：
1. **Sparse**: 大部分3D空间是空的
2. **Ray-based**: 每个点对应一条从sensor出发的ray
3. **Physics**: 有reflection intensity, ray dropping（beam没反射回来）等现象

如果直接在dense voxel space做生成，computation wasteful且不能model ray physics。

### 4.2 Sparse UNet + Prior-guided Sampling

**Step 1: Sparse UNet**
- 输入semantic occupancy
- 用submanifold sparse convolution（Graham & van der Maaten）提取spatial features
- 只在occupied voxel计算，效率高

**Step 2: Ray-based Sampling**
对每条LiDAR ray $r$，uniform sampling得到points sequence $s$。

**Prior-guided PDF**:
- Occupied voxel内的point: probability = 1
- 其他point: probability = 0
- 这样resample时只在有物体的区域采样，避免waste computation

**Resample**: 从PDF采样 $n$ 个points $\{\mathbf{r}_i = o + s_i v\}_{i=1}^n$
- $o$: ray origin (LiDAR位置)
- $v$: normalized ray direction
- $s_i$: 沿ray的距离

### 4.3 LiDAR Head: Volume Rendering + Physics Modeling

**SDF Prediction**:
每个sampled point的feature $u_i$（从Sparse UNet采样）经过MLP预测signed distance function $f(\mathbf{r}(s_i))$。

**Volume Rendering Weights** (NeuS-style):
$$\beta_i = \max\left(\frac{\Phi_s(f(\mathbf{r}(s_i))) - \Phi_s(f(\mathbf{r}(s_{i+1})))}{\Phi_s(f(\mathbf{r}(s_i)))}, 0\right)$$

- $\Phi_s(x) = (1+e^{-sx})^{-1}$: sigmoid function with temperature $s$
- $f(\mathbf{r}(s_i))$: SDF value at point $s_i$ on ray $\mathbf{r}$
- $\beta_i$: "density" at point $i$，表示surface在这里的概率

**Weighted Depth**:
$$w(s_i) = \prod_{j=1}^{i-1}(1-\beta_j)\beta_i, \quad h = \sum_{i=1}^{n} w(s_i) s_i$$

- $w(s_i)$: 每个点的渲染权重
- $h$: 渲染深度（LiDAR测距值）

**Ray Feature Aggregation**:
$$v_r = \sum_{i=1}^{n} w_i \cdot u_i$$

- $u_i$: 第$i$个点的feature
- $w_i$: 渲染权重
- $v_r$: 聚合后的ray feature

**Intensity Head**: $v_r$ 经过MLP预测reflection intensity

**Ray-dropping Head**: $v_r$ 经过MLP预测ray被drop的概率（模拟real LiDAR的miss detection）

### 4.4 LiDAR Loss

$$\mathcal{L}_{lid} = \mathcal{L}_{depth} + \lambda_1 \mathcal{L}_{inten} + \lambda_2 \mathcal{L}_{drop}$$

- $\mathcal{L}_{depth}$: depth预测loss
- $\mathcal{L}_{inten}$: intensity预测loss
- $\mathcal{L}_{drop}$: ray-dropping分类loss

### 4.5 Ablation Insights (Table 10)

| Setting | MMD↓ | JSD↓ | Time(s)↓ | Memory(GB)↓ |
|---------|------|------|----------|-------------|
| Ours | 1.53 | 0.072 | 0.25 | 6.84 |
| w/o Sparse UNet | 2.88 | 0.097 | 0.21 | 6.73 |
| w/o Sparse Sampling | 1.69 | 0.075 | 0.30 | 16.66 |
| w/o Ray-dropping Head | 3.25 | 0.100 | 0.25 | 5.05 |

**Sparse Sampling**带来巨大的memory savings（16.66GB → 6.84GB，-58.94%），且性能略好。Ray-dropping head对MMD/JSD影响最大（+25.92% MMD），说明modeling LiDAR physics很important。

## 五、实验结果全景分析

### 5.1 Generation Quality对比

**Occupancy Generation (Table 3)**:
- Ours-Gen (CFG=1): mIoU 19.44, F3D 158.55, MMD 10.60
- OccWorld: mIoU 17.13, F3D 145.65, MMD 9.89
- Ours-Fore (forecasting variant): mIoU 31.76, F3D 43.13, MMD 2.86

Forecasting variant表现最好，因为用了conditional frames。Generation variant在mIoU上比OccWorld好，但F3D/MMD略差（可能因为generation task更难，没有reference frames）。

**Video Generation (Table 4)**:
- Ours (Gen Occ): FID 6.45, FVD 71.94
- Ours (GT Occ): FID 6.12, FVD 70.52
- Vista*: FID 13.97, FVD 112.65
- Drive-WM: FID 15.80, FVD 122.70

**Gen Occ vs GT Occ差距很小**（FVD 71.94 vs 70.52），说明occupancy generation的质量足够好，没有成为bottleneck。这是decomposed learning成功的关键证据。

**LiDAR Generation (Table 5)**:
- Ours (Gen Occ): MMD 2.40, JSD 0.108, Time 0.47s
- Ours (GT Occ): MMD 1.53, JSD 0.072, Time 0.25s
- LiDARDM: MMD 3.51, JSD 0.118, Time 45.12s

**推理速度快100倍**（0.47s vs 45.12s），这是因为sparse modeling避免了dense computation。

### 5.2 Downstream Task Benefits

**Occupancy Prediction (Table 6)**:
- Camera-only: CONet baseline 20.1 IoU → w/ Ours 28.6 (+8.5)
- LiDAR-only: 30.9 → 33.1 (+2.2)
- Multi-modal: 29.5 → 35.4 (+5.9)

Camera-only的提升最大，因为generated video提供了diverse training data。

**BEV Segmentation & 3D Detection (Table 7)**:
- Road mIoU: 74.30 → 81.69 (+7.39)
- Vehicle mIoU: 36.00 → 41.62 (+5.62)
- mAP: 32.88 → 36.50 (+3.62)
- NDS: 37.81 → 41.22 (+3.41)

全面surpass其他augmentation方法（BEVGen, Vista*, MagicDrive）。

## 六、与相关工作对比

### 6.1 Generation方法对比

| 类别 | 代表方法 | 局限 | UniScene优势 |
|------|----------|------|--------------|
| Video only | MagicDrive, Drive-WM | 缺LiDAR, 缺3D consistency | Occupancy提供3D prior |
| LiDAR only | LiDARDM, LiDARGen | 缺video, 慢 | Sparse modeling快100x |
| Occupancy only | OccSora, OccWorld | VQVAE损失大, 不可控 | VAE + BEV controllable |
| World model | WoVoGen, Vista | 单modal或weak 3D | Unified 3 modalities |

### 6.2 技术对比

**vs VQVAE-based occupancy generation**:
- OccSora: compression 512, mIoU 27.4 (codebook collapse)
- UniScene: compression 512, mIoU 72.9 (continuous latent)

**vs Spatial-temporal attention for multi-view**:
- MagicDrive/Drive-WM: cross-view attention, expensive
- UniScene: occupancy-based rendering, natural consistency

**vs Mesh-based LiDAR generation**:
- LiDARDM: mesh diffusion + ray casting, 45s/frame
- UniScene: sparse sampling + volume rendering, 0.25s/frame

## 七、Controllable Generation能力

### 7.1 BEV Editing

通过修改BEV layout $B_{ori} \to B_{new}$，然后用DDIM Inversion：
1. Original occupancy $O_{ori}$ → noise latent $\epsilon_{ori}$ (guided by $B_{ori}$)
2. 用 $\epsilon_{ori}$ 作为starting noise + $B_{new}$ 作为condition → new occupancy $O_{new}$
3. $O_{new}$ → new video $V_{new}$ + new LiDAR $L_{new}$

这样能保持大部分scene不变，只修改edited region。

### 7.2 Attribute Control

Text prompt控制weather（sunny, cloudy, rainy）和time of day（day, night），因为SVD本身支持text conditioning。

### 7.3 Generalization

Figure 19显示在Waymo和nuPlan上直接transfer也能work（虽然back view unavailable需要特殊处理）。

## 八、Limitations和Future Directions

1. **Computational cost**: 训练整个unified system resource-intensive
2. **Lightweight deployment**: 未探索
3. **Extension**: Embodied intelligence, robotics

## 九、Key Takeaways

1. **Decomposed learning wins**: 把complex generation分解成hierarchical steps，每个step更tractable
2. **Occupancy is the right intermediate**: Rich semantics + geometry + controllable + transferable
3. **Gaussian Splatting bridges 3D and 2D**: 用differentiable rendering把occupancy转成video guidance
4. **Sparse modeling for efficiency**: LiDAR的sparsity应该被exploit而不是ignored
5. **Physics modeling matters**: Ray-dropping, intensity等LiDAR physics对realism很重要
6. **Gen Occ ≈ GT Occ**: 中间representation的质量足够好，不成为bottleneck

## 参考链接

- **Project page**: https://arlo0o.github.io/uniscene/
- **DiT (Peebles & Xie)**: https://arxiv.org/abs/2212.09748
- **Gaussian Splatting**: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- **Stable Video Diffusion**: https://arxiv.org/abs/2311.15127
- **ControlNet**: https://github.com/lllyasviel/ControlNet
- **NeuS**: https://arxiv.org/abs/2106.10689
- **NuScenes**: https://www.nuscenes.org/
- **OccWorld**: https://arxiv.org/abs/2311.16038
- **MagicDrive**: https://gaoruiyuan.com/magicdrive/
- **LiDARDM**: https://arxiv.org/abs/2404.02903
- **Submanifold Sparse Conv**: https://arxiv.org/abs/1706.01307
- **Lovász-Softmax**: https://arxiv.org/abs/1705.08790
- **OccSora**: https://arxiv.org/abs/2405.20337
- **Vista**: https://arxiv.org/abs/2405.17398
- **NKSR**: https://huggingface.co/papers/2301.10243
