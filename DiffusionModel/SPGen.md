---
source_pdf: SPGen.pdf
paper_sha256: 2811165e562d8c3c55e26387d125004a018cca626a1c0e0dc9bb0b0f0941597d
processed_at: '2026-08-12T09:54:47-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# SPGen 人话版

## 这 paper 到底在干嘛

你给它一张 2D 照片（比如一张椅子的正面图），它给你吐出一个 3D 椅子模型，全程 6–10 秒。

就这件事。听起来简单，但难点在"中间用什么格式表示这个 3D 椅子"。

---

## 3D 生成的两难

现在做 single image to 3D 主要两条路，各有各的毛病：

**路线一：直接在 3D 空间里生成**（CLAY、Trellis、Hunyuan3D-2 这些）
- 直接预测 point cloud、SDF、或者 mesh faces
- 问题：3D 数据太少了，而且不像 2D 图像有几十亿张可以预训练。你要从头教会 model "什么是对称"、"什么是局部平滑"，这些 2D model 早就免费学到的东西
- 代价：256 张 A800 训两周，烧钱

**路线二：生成多个视角的 2D 图再拼回 3D**（Wonder3D、Zero123 这些）
- 让 model 生成 6–24 个角度的照片，然后用 differentiable rendering 拼成 3D
- 问题：不同角度之间对不齐。model 画正面觉得椅子腿在左，画侧面又觉得在右，拼起来就错位、糊、有 artifact
- 根本原因：同一个 3D 点会出现在多个 view 里，model 必须学会跨 view 一致，但这是个"软约束"（靠 cross-attention），随时可能翻车

---

## SPGen 的核心 idea：换一种 2D 表示

paper 的 insight 是：**与其生成多个 perspective views 再拼，不如生成一张"全景深度图"**。

想象一下：把椅子放在一个透明气球的正中间，你站在气球中心，往四面八方看。每个方向你都能看到椅子的某个点，记录下"这个方向椅子离我多远"。

- 气球表面有经纬度（azimuth θ、polar φ），就像地球仪
- 把气球像世界地图一样剪开摊平，横轴是经度（360°）、纵轴是纬度（180°），变成一张 256×512 的图
- 每个像素存一个数字：那个方向上物体离原点的距离 d

这就是 **Spherical Projection map (SP map)**。

### 为什么这个表示妙

**第一，天然一致，没有歧义**。
SP map 上每个像素对应一条唯一的光线，每个 3D surface 点只出现在一个像素位置。model 生成什么就是什么，不存在"正面和侧面要一致"这种麻烦事。用数学行话说是 injective function（单射）。

**第二，可以直接用 SDXL**。
SP map 虽然存的是 depth 不是颜色，但它长得就是一张 2D 图，有 locality（相邻像素对应相邻方向）、有 symmetry（旋转物体对应 SP map 平移）、有重复 pattern。这些恰好是 SDXL 在几十亿张图上学到的 prior。所以你只要 finetune 一下，model 就能从"画 RGB 图"转行"画 depth 全景图"，之前学的构图直觉全都能用上。

**第三，多层可以处理复杂结构**。
单层 SP map 的问题是：如果光线穿过了杯子的外壁、内壁、对面的内壁、外壁，你只能记一个距离，丢了内部结构。Solution 是记每条光线的所有 intersection，从外到内排好序，存到 4 张 SP map 里。

实测在 160k Objaverse 物体上：
- 1 层覆盖 92.0% 的 surface
- 4 层覆盖 99.9%
- 再多意义不大

所以 4 层就够了，大部分物体就是个 shell，少数有内腔。

---

## 训练 pipeline 三个 stage

### Stage 1：Finetune AutoEncoder

SDXL 原本的 VAE 是为 RGB 图设计的，你拿它来压缩 depth map 效果不好，必须 finetune。

输入一张 SP map，encode 成 latent，再 decode 回来，要求还原得准。同时加 KL regularization 让 latent 分布别跑偏太远（不然后面 UNet 接不住）。

**这里有个关键观察**：如果只用 L1 loss 训，还原出来的 SP map 边缘是糊的。paper 做了个很漂亮的诊断——把预测和 GT 的 error map 画出来，发现 error 几乎全集中在 SP map 的"几何边缘"上（对应 3D 物体的 silhouette 和 depth discontinuity）。在频域上看，error 集中在 FFT 的高频区（四角）。

原因很朴素：L1 loss 把所有像素一视同仁。边缘像素可能只占 5%，它们的 loss 信号被 95% 的平滑区域稀释了，model 自然倾向于把平滑区做好、边缘糊掉。

**Paper 的两个 trick**：

1. **Edge loss**：用 Sobel 算子标出边缘像素，再 dilate 一下形成 band，对这些像素加大 loss 权重。本质就是"考试给难题加权，逼学生练难题"。

2. **Spectral loss**：对预测和 GT 都做 FFT，用 high-pass filter 只留高频分量，分别比较 phase（相位，决定边缘在哪）和 magnitude（幅值，决定边缘多 sharp）。这比直接比较复数值更稳定。

两个 trick 加上之后，train loss 从 $3.05 \times 10^{-4}$ 降到 $1.47 \times 10^{-4}$，收敛快了一倍，边缘明显锐利。

### Stage 2：Finetune Diffusion UNet

AE 训好后，用它把所有 SP map encode 成 latent，然后训 SDXL 的 UNet 做 conditional generation（输入单张图，输出 SP map latent）。

**Condition encoder 的选择**：paper 对比了 CLIP 和 DINOv2。CLIP 是 contrastive learning，学到的是全局 semantic（"这是一只狗"），spatial 信息弱；DINOv2 是 self-distillation，加了各种 data augmentation，spatial detail 更丰富。对"要忠实还原输入图像几何"这个任务，DINOv2 明显更合适。paper 用 PCA 可视化 token map，CLIP 的主成分几乎看不出空间结构，DINOv2 的能清楚看到物体轮廓。

**Layer-wise Self-Attention (LSA)**：这是处理 multi-layer 的关键。4 层 SP map 共享同一组 ray grid，layer 1 的 pixel (θ,φ) 和 layer 2 的 pixel (θ,φ) 是同一条光线的两个 intersection，有强相关。LSA 把 4 层的 hidden states concat 起来跑 self-attention，让每个 pixel 能看到其他层对应位置，学到"外层 depth 必须 ≤ 内层 depth"这种约束。

Ablation 显示去掉 LSA，F-Score 从 95.57% 暴跌到 60.75%，CD 涨 8.5 倍。可视化里各层出现 scale/orientation mismatch，unproject 回 3D 就一堆 floating artifact。

**Circular padding**：SP map 横轴是 360° 周期的，最左和最右是同一条经线。普通 zero padding 会让这里断开，用 circular padding 让卷积"绕一圈"接上。实测边界误差 0.23%。

### Stage 3：SP map → Mesh

生成完 SP map 后，用公式 $\mathcal{P} = [\sin\phi\cos\theta, \sin\phi\sin\theta, \cos\phi]^\top \cdot d$ 把每个 pixel unproject 回 3D 点，得到几十万个点的 point cloud。

- **Watertight 物体**（Objaverse、GSO 这些）：训了个轻量 3D-UNet 预测每个点的 normal，然后跑 Poisson reconstruction 拿 mesh
- **Open surface**（DeepFashion3D 衣服这些）：SDF 不行（只能 watertight），用 SurfD 的 point-cloud-to-UDF AutoEncoder 预测 UDF，再 MeshUDF 提 mesh

---

## 结果有多猛

### GSO 数据集

vs InstantMesh（之前最强 baseline）：
- Chamfer Distance：0.0120 → 0.0051，好 57.5%
- Volume IoU：0.4310 → 0.5407，好 25.4%
- F-Score：88.84% → 95.57%，好 7.6%
- 速度：35s → 6–10s，快 3.5–6 倍

### 训练效率（最夸张的对比）

| 方法 | GPU | 训练时间 | 数据量 |
|---|---|---|---|
| CLAY | 256 A800 (10TB) | ~2 周 | 527k |
| Trellis | 64 A100 (2.5TB) | - | 500k |
| TripoSG | 160 A100 (6.25TB) | ~3 周 | 2M |
| **SPGen** | **2 GPU (0.09TB)** | **~1 周** | **160k** |

SPGen 用不到 5% 的资源，F-Score 达到 98.28%（Trellis 98.35%、Hunyuan3D-2 98.43%），基本打平。CD/IoU 略差但仍然 competitive。

这就是 image-domain representation 的威力——你把 3D 问题降维成 2D 问题，直接白嫖 SDXL 在几十亿张图上学到的 prior。

---

## 几个直觉性总结

**为什么 SP map 比 multi-view 好**：multi-view 是"从外面拍 6 张照片拼"，SP map 是"站中心拍一张全景"。前者各照片有 overlap，要学一致；后者单张全覆盖，天生一致。

**为什么 SP map 比 geometry image / UV mapping 好**：后两者要把 mesh 剪开摊平，剪法不唯一，剪开的边要 model 学缝合，很别扭。SP map 用球面投影，固定 mapping，no cut，no stitch。

**为什么 SP map 比 Matryoshka 好**：Matryoshka 是 6 个方向各拍深度图再 fuse 成 voxel，surface detail 被 voxel 分辨率 $N^3$ 限制。SP map 直接 $N^2$，同分辨率下细节更好。

**为什么 finetune SDXL 而不从 scratch 训**：SDXL 学过"对称"、"局部平滑"、"重复 pattern"、"物体 part 的典型布局"。这些在 SP map 上照样成立（旋转对称对应 SP map 平移对称，part 布局在 SP map 上有固定 pattern）。从 scratch 训就要重新学这些，所以 CLAY 要 256 A800 两周，SPGen 只要 2 GPU 一周。

**为什么 edge loss 重要**：3D 几何质量 90% 取决于 silhouette 和 depth discontinuity 是否 sharp。L1 loss 让 model 把这些高频细节糊掉，相当于"画了个像但轮廓不清"。Edge + spectral loss 强迫 model 把精力放在刀刃上。

---

## 一个类比

把 SPGen 想象成一个翻译任务：

- 输入：一张椅子的正面照片（英文）
- 输出：一张"球面全景深度图"（一种新的语言）

这种新语言虽然语法特殊（存的是距离不是颜色），但书写格式跟英文一样（都是 2D grid，都有 locality 和 pattern）。所以你找一个会英文的翻译（SDXL），让他学学新语言的语法（finetune），就能干活。他之前学的"对称句式"、"段落结构"全都能迁移。

而 CLAY 那种 geometry-based 方法相当于从零教一个婴儿学全新语言，而且这种语言没有大量文本可读（3D data 少），所以要用 256 个老师（A800）教两周。

---

## 局限

1. **极区 distortion**：equirectangular projection 在两极会 over-sampling（世界地图上格陵兰岛被放大同理），但实测误差 0.20%，可控
2. **光线与面平行时数值不稳**：比如 hemisphere 的 flat boundary 正好与 ray 平行，Möller-Trumbore 算法会出问题，paper 用 perturbation 绕过
3. **Texture 生成只做了小规模 demo**：2k subset 上验证可行，没 scale up
4. **固定 4 层**：对极少数超复杂物体可能不够，但 99.9% 覆盖率够用了

---

## 我的 takeaway

这篇 paper 给 3D generation 领域一个重要 lesson：**representation 是 leverage 点**。与其堆算力硬训 3D native model，不如找一个"既能完整表达 3D 几何、又能 plug-in 2D pretrained prior"的中间表示。SP map 就是这样一个 sweet spot——injective 保证一致、multi-layer 保证拓扑灵活、2D 格式保证能蹭 SDXL。

如果后续有人用 DiT 替代 SDXL、用 rectified flow 替代 DDPM、用 cube2sphere 解决极区 distortion，这个方向可能还有 2–3 倍的提升空间。

---

# SPGen 深度讲解：Spherical Projection 作为 3D Shape Generation 的一致且灵活表示

## 1. Paper Overview 与核心 Motivation

这篇 paper 是 SIGGRAPH Asia 2025 的工作，来自 Texas A&M University (Wenping Wang 组、Xin Li 组) + LightSpeed Studios + HKUST + Waymo。核心 idea 用一句话概括：**把 3D mesh 投影到 unit sphere 上展开成 multi-layer equirectangular depth maps，然后直接 finetune SDXL 做 image-conditioned generation**，最后 unproject + Poisson/UDF reconstruction 拿到 mesh。

paper 想解决三个痛点：
1. **View inconsistency**：multi-view diffusion 方法（Wonder3D、Zero123、SyncDreamer）靠 cross-attention 软约束对齐 views，overlapping 区域常出现几何冲突。
2. **Topology inflexibility**：SDF 类方法只能处理 watertight mesh，open surface（衣服、纸张）表达不了；mesh autoregressive 方法（MeshGPT、MeshAnything）受 face 数量限制。
3. **Scalability / Efficiency**：geometry-based 方法（CLAY、Trellis、Hunyuan3D-2、TripoSG）动辄 64–256 A100 训练数周，527k–2M data；SPGen 只用 2 GPU、160k data、1 week 达到 competitive performance。

paper 链接与相关资源：
- arXiv (推测): https://arxiv.org/abs/2506.XXXX (该 paper 属于 SIGGRAPH Asia 2025，arXiv 可能尚未放出)
- ACM DOI: https://doi.org/10.1145/3757377.3763959
- Project page (推测): https://spgen.github.io/ 或 https://tamu.edu/... 
- SDXL: https://arxiv.org/abs/2307.01952
- DINOv2: https://arxiv.org/abs/2304.07193

---

## 2. Spherical Projection (SP) 表示：从 3D mesh 到 2D depth panorama

### 2.1 几何构造

把 normalized 3D object（scale 到 [−0.5, 0.5]）放在 unit sphere 中心。对 sphere 上每个点 $(\theta, \varphi)$，从 origin 沿 radial direction cast 一条 ray，与 mesh surface 求交，记录 depth $d = \|\mathcal{P}\|_2$（即交点到 origin 的欧氏距离）。Sphere 再通过 **equirectangular projection** 展平为 256×512 的 2D map，横轴 azimuth、纵轴 polar。

### 2.2 公式 (1) 解析

$$\mathcal{P} = \mathrm{F}^{-1}(\theta, \phi) = \begin{bmatrix} \sin\phi \cos\theta \\ \sin\phi \sin\theta \\ \cos\phi \end{bmatrix} d$$

变量含义：
- $\mathcal{P} \in \mathbb{R}^3$：3D 点的 Cartesian 坐标
- $\theta \in [-\pi/2, 3\pi/2)$：azimuth angle，水平绕轴角度（覆盖 360°，2π 范围）
- $\phi \in [0, \pi)$：polar angle，从北极（z+ 轴）开始的倾斜角（覆盖 180°，π 范围）
- $d \in \mathbb{R}^+$：depth，从 origin 到 surface 点的距离
- $[\sin\phi\cos\theta, \sin\phi\sin\theta, \cos\phi]^\top$：单位球面上的方向向量

**Intuition**：这本质就是球坐标 → 直角坐标转换。SP map 上每个像素 $(\theta, \varphi)$ 唯一确定一条 ray，存一个 scalar $d$，反过来用此公式即可 unproject 回 3D 点。所以 SP map 到 3D surface 是 **injective function**（每个 valid pixel 对应唯一 surface point），这就从根本上消除了 multi-view 间的歧义。

### 2.3 为什么是 injective 这么关键

Multi-view 方法里，同一个 3D 点会出现在多个 view 的不同像素位置，generator 必须学会跨 view 一致地预测它——这是个 soft constraint（cross-attention），violations 不可避免。SP map 是 single panorama view，每个 surface point 唯一映射到一个 pixel，generator 输出什么就是什么，没有 conflict 可能。

### 2.4 Multi-layer SP Map：处理 self-occlusion 与 internal structure

单层 SP map 在 ray 多次穿过 surface 时只能记一个 depth，无法表达 internal layer（如杯子内壁、双层壳体、多孔结构）。Solution 是 trace ray 上所有 intersection，按从外到内顺序存到多个 layer。

Algorithm 1 的逻辑：
```
For each ray R_i:
  Find all intersections {P_i^0, P_i^1, ..., P_i^k} via Ray-MeshIntersection
  For j in 0..k:
    if P_i^{k-j} is not NULL:
      M^step[θ,φ] = ||P_i^{k-j}||_2  (从最外层往里填)
      step += 1
```

注意它从最外层 intersection 开始记录（`P_i^{k-j}` 反向遍历），保证 layer 1 是最外层、layer k 是最内层，这为后续 Layer-wise Self-Attention 提供了天然的顺序先验。

Empirical 实验显示在 160k Objaverse 上：
- 1 layer: IoU 92.0%
- 2 layers: 98.7%
- 3 layers: 99.8%
- 4 layers: 99.9%
- 5 layers: 99.9% (无提升)

因此 paper 选 4 layers 作为默认配置。这个曲线告诉我们 3D 物体的 "shell 复杂度" 分布是长尾的，4 层覆盖 99.9%，再多意义不大。

---

## 3. Generation Pipeline：基于 SDXL 的 finetuning

整体 pipeline 见 Fig. 2，包含三个 stage：AE training、Diffusion training、Surface extraction。

### 3.1 Preliminaries（公式 2–4）

**AE reconstruction + KL regularization**（公式 2）：

$$L_{recon} = \mathbb{E}_M \big[ \|M - \Psi_\mathcal{D}(\Psi_\mathcal{E}(M))\| \big] + \lambda \cdot \mathbb{E}_M \big[ D_{KL}(Q(z|M) \| \mathcal{N}(0, I)) \big]$$

变量：
- $M$：input SP map（单通道 depth）
- $\Psi_\mathcal{E}, \Psi_\mathcal{D}$：encoder, decoder
- $z_0 \sim Q(z|M)$：latent code
- $Q(z|M)$：encoder 输出分布
- $\lambda$：KL 权重（paper 用 $10^{-8}$，保持 latent 接近 standard normal，否则 SDXL 的 UNet 无法继承）
- $D_{KL}$：KL 散度

**Forward diffusion**（公式 3）：

$$z_t = \sqrt{\alpha_t} z_0 + \sqrt{1 - \alpha_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

- $z_t$：time step $t$ 时的 noisy latent
- $\alpha_t$：noise schedule coefficient（cumulative product of $1-\beta_t$）
- $\epsilon$：标准 Gaussian noise

**Reverse diffusion training**（公式 4）：

$$L_{diff} = \mathbb{E}_{z_0, \epsilon, t} \bigg[ \| \epsilon - \epsilon_\Theta(z_t, t) \|^2 \bigg]$$

- $\epsilon_\Theta$：UNet 参数化的 noise predictor
- $\Theta$：UNet 参数
- 这是标准 $\epsilon$-prediction 形式

### 3.2 单图 conditioning：DINOv2 vs CLIP

paper 在 supplementary A.1 里有个很重要的 ablation：对比 CLIP 和 DINOv2 的 visual embedding 质量（Fig. 10 通过 PCA 可视化）。CLIP 主要 capture 全局 semantic（contrastive learning），spatial 信息弱；DINOv2 (self-distillation) 通过 data augmentation 学到更 fine-grained 的 spatial detail。对 SP map generation 这种需要 pixel-level faithful to input 的任务，DINOv2 更合适。所以 SPGen 用 DINOv2 作为 condition encoder，再 cross-attention 注入 UNet。

相关 links：
- CLIP: https://arxiv.org/abs/2103.00020
- DINOv2: https://arxiv.org/abs/2304.07193

### 3.3 关键 trick：Circular padding

SP map 在 azimuth 方向是周期性的（$\theta = -\pi/2$ 与 $\theta = 3\pi/2$ 是同一条线），如果用普通 zero padding 会导致 border discontinuity。paper 借鉴 panoramic image generation 工作（[Wang et al. 2023](https://arxiv.org/abs/2308.14686)），在 UNet conv 层用 **circular padding** 处理 azimuth 方向，保证 border 一致。实测 azimuth border 的 absolute-relative-error 只有 0.23%。

---

## 4. Layer-wise Self-Attention (LSA)：让多层 SP 对齐

### 4.1 公式 (5)、(6) 解析

公式 (5)：

$$\bar{m} = \mathrm{Concat}\big([\mathrm{Flat}(m^1), \ldots, \mathrm{Flat}(m^k)], \text{dim}=-1\big)$$

- $m^i \in \mathbb{R}^{C \times h \times w}$：UNet 中对应第 $i$ 层 SP map 的 hidden state
- $\mathrm{Flat}(\cdot)$：把空间维 flatten 为 $\mathbb{R}^{C \times (hw)}$
- $\bar{m}$：所有 layers 沿 spatial 维 concat，变成 $\mathbb{R}^{C \times (k \cdot hw)}$

公式 (6)：

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\bigg(\frac{QK^T}{\sqrt{C_a}}\bigg) \cdot V$$

- $Q, K, V$：由 $\bar{m}$ 经线性投影得到
- $C_a$：attention 的 head dimension（除以 $\sqrt{C_a}$ 做 scaled dot-product）

### 4.2 Intuition

这个设计的关键在于：**多层 SP map 共享同一组 ray（同一 $(\theta, \varphi)$ 网格），只是记录了不同 depth 的 intersection**。所以 layer $i$ 的某个 pixel 与 layer $i+1$ 同 pixel 之间存在强相关（同一 ray 的连续 intersection）。LSA 把所有 layer 的 hidden states concat 起来跑 self-attention，让每个 pixel 都能 attend 到其他 layer 的对应位置，从而学到 "outer layer 的 depth 必须 ≤ inner layer 的 depth" 这种空间约束。

Ablation（Table 4）显示去掉 LSA 后 F-Score 从 95.57% 暴跌到 60.75%，CD 从 0.0051 涨到 0.0436（约 8.5x）。Fig. 6 可视化显示没有 LSA 时各 layer 出现 scale 或 orientation mismatch，unproject 后产生 floating artifacts 和 self-intersection。

---

## 5. Geometry Regularization：解决 high-frequency error 集中问题

这是 paper 最 elegant 的部分之一，我觉得有 building intuition 价值。

### 5.1 观察：error 集中在 edge

paper 在训练 AE 时发现：用 L1 loss 训出来的 SP map 整体轮廓 OK，但边缘模糊、细节丢失。Fig. 3 给了三组可视化：
- **Edge map**（Sobel）：标出 SP map 上的几何边界
- **Spectrum map**（FFT）：频谱图，中心是低频，四角是高频
- **Error map**：预测 vs GT 的 pixel-wise error

可以清楚看到 error map 的空间分布与 edge map 高度重合，频谱上 error 也集中在四个角（高频区）。原因：L1 loss 把所有 pixel 一视同仁，edge pixel 在总像素中占比极小（比如 5%），它们的 loss 信号被大量 smooth area 稀释，model 自然倾向于优化大面积的 smooth 区域。

类比：这也是 super-resolution 任务里常见的现象，[Focal Frequency Loss (Jiang et al. 2021)](https://arxiv.org/abs/2012.08114) 提出过类似观察。

### 5.2 Edge loss（公式 7）

$$L_{edge} = \mathbb{E}_M \big[ \mu \mathcal{B} \cdot \|\mathcal{M} - \Psi(\mathcal{M})\| + (1-\mu)(1-\mathcal{B}) \cdot \|\mathcal{M} - \Psi(\mathcal{M})\| \big]$$

变量：
- $\mathcal{B} = \mathrm{Dilateate}(\mathrm{Sobel}(\mathcal{M}))$：boundary mask，先用 Sobel 算 gradient 找 edge，再 dilation 扩大边缘区域
- $\Psi = \Psi_\mathcal{D} \circ \Psi_\mathcal{E}$：整个 AE
- $\mu$：edge 区域的 loss 权重（>0.5 倾向 edge，paper 中应该是较大值）

**Intuition**：这是个加权 L1 loss，强制让 model 在 edge pixel 上多花 "注意力"。Dilation 是必要的，因为单纯 Sobel 得到的 edge 太细（1 pixel），加 dilate 后形成 band，给 model 一些容差空间。

### 5.3 Spectral loss（公式 8）

$$L_{spec} = \mathbb{E}_M \big[ \mathcal{H} \cdot \|\mathrm{Arg}(\mathcal{M}_s) - \mathrm{Arg}(\tilde{\mathcal{M}}_s)\| + \zeta \mathcal{H} \cdot \big| \|\mathcal{M}_s\|_2 - \|\tilde{\mathcal{M}}_s\|_2 \big| \big]$$

变量：
- $\mathcal{M}_s = \mathrm{FFT}(\mathcal{M})$：对 SP map 做 2D FFT 得到频谱
- $\tilde{\mathcal{M}}_s = \mathrm{FFT}(\Psi(\mathcal{M}))$：预测 SP map 的频谱
- $\mathcal{H}$：high-pass filter，圆形 mask 中心区域置 0、外围置 1，只允许高频通过
- $\mathrm{Arg}(\cdot)$：复数的 argument（phase / 相位）
- $\|\cdot\|_2$：复数的 modulus（magnitude / 幅值）
- $\zeta$：phase 与 magnitude 的权重平衡

**Intuition**：高频分量同时有 phase 和 magnitude，phase 决定 "edge 在哪"，magnitude 决定 "edge 有多 sharp"。两者分开惩罚比直接比较 complex value 更稳定。这比 [Focal Frequency Loss](https://arxiv.org/abs/2012.08114) 更精细，因为后者用 weighted MSE，这里分开处理 phase。

### 5.4 Total loss（公式 9）

$$L = L_{recon} + \alpha L_{edge} + \beta L_{spec}$$

- $\alpha, \beta$：两个 regularizer 的权重

Ablation（Fig. 4）：在 10k 子集上训练，加 geometry regularization 后 train loss 从 $3.05 \times 10^{-4}$ 降到 $1.47 \times 10^{-4}$，test loss 从 $3.59 \times 10^{-4}$ 降到 $1.61 \times 10^{-4}$，9k iteration 收敛。可视化显示加 regularizer 后 SP map 边缘更锐利，reconstructed mesh 表面噪声大幅减少。

---

## 6. Surface Extraction：从 SP map 到 mesh

paper 设计了两条路径：

### 6.1 Watertight mesh：Poisson reconstruction

1. Unproject 所有 SP map 的 pixel 到 3D 得到 dense point cloud（4 layers × 256×512 ≈ 524k points，会去重）
2. 训练一个 lightweight 3D-UNet (based on ConvONet, [Peng et al. 2020](https://arxiv.org/abs/2007.12484)) 作为 normal estimator，输入 point cloud 输出每个 point 的 unit normal
   - 配置：192 hidden dimensions, 5-level encoder with 128 feature dims
   - 训练数据：从 GT mesh 上 sample 25600 oriented points per object
3. 用 oriented point cloud 跑 Poisson reconstruction（标准 [Kazhdan et al.](https://www.cs.jhu.edu/~misha/MyPapers/PoissonRecon.pdf)）拿 watertight mesh

### 6.2 Open surface：UDF reconstruction

对于 DeepFashion3D 这种 garment 类 open surface：
1. 用 [SurfD (Yu et al. 2024)](https://arxiv.org/abs/2403.04132) 预训练的 point-cloud-to-UDF AutoEncoder 预测空间中的 UDF (Unsigned Distance Field)
2. 用 [MeshUDF (Guillard et al. 2022)](https://arxiv.org/abs/2206.08797) 从 UDF 提取 mesh

这种 dual-pathway 设计让 SPGen 同时支持 watertight 和 open surface，比纯 SDF 方法灵活。

---

## 7. Experiments 详解

### 7.1 Datasets & Metrics

- **Objaverse** (filter 后 160k objects for training, 1993 for validation)：[Deitke et al. 2023](https://arxiv.org/abs/2212.01622)
- **GSO (Google Scanned Objects)** 30 shapes for test：[Downs et al. 2022](https://arxiv.org/abs/2209.08133)
- **DeepFashion3D** for open surface：[Zhu et al. 2020](https://arxiv.org/abs/2003.12742)
- Metrics: Chamfer Distance (CD, ↓), Volume IoU (↑), F-Score with threshold 0.1 (↑)
- Alignment：brute-force rotation search + center/scale normalize to [−1, 1]

### 7.2 GSO 结果（Table 1）

| Method | Latency | CD↓ | Vol.IoU↑ | F-Score%↑ |
|---|---|---|---|---|
| Point-E | ~25s | 0.0690 | 0.1953 | 52.23 |
| Shape-E | ~20s | 0.0418 | 0.2785 | 64.83 |
| Wonder3D | ~10min | 0.0398 | 0.2930 | 68.82 |
| CRM | ~18s | 0.0264 | 0.3374 | 74.43 |
| OpenLRM | ~15s | 0.0344 | 0.3770 | 71.50 |
| LGM | ~40s | 0.0212 | 0.4220 | 78.41 |
| InstantMesh | ~35s | 0.0120 | 0.4310 | 88.84 |
| **SPGen (Ours)** | **6–10s** | **0.0051** | **0.5407** | **95.57** |

vs InstantMesh（最强 baseline）：
- CD: 0.0120 → 0.0051，**+57.5% relative gain**
- IoU: 0.4310 → 0.5407，**+25.4%**
- F-Score: 88.84% → 95.57%，**+7.6%**
- Latency: 35s → 6–10s，约 **3.5–6× 加速**

### 7.3 DeepFashion3D 结果（Table 2）

| Method | CD↓ | IoU↑ | F-Score%↑ |
|---|---|---|---|
| Wonder3D | 0.0223 | 0.3370 | 73.52 |
| OpenLRM | 0.0237 | 0.3680 | 78.25 |
| LGM | 0.0244 | 0.3110 | 71.60 |
| InstantMesh | 0.0314 | 0.2890 | 68.72 |
| SurfD | 0.0136 | 0.3860 | 82.31 |
| **Ours (rgb)** | 0.0099 | 0.4200 | 87.16 |
| **Ours (sketch)** | **0.0092** | **0.4480** | **89.35** |

vs SurfD：CD +32.4%, IoU +16.1%, F-Score +8.6%。这是 open surface setting，证明 SP multi-layer 对 garment 类拓扑有效。

### 7.4 与 image-based 表示的对比（Table 3）

在 GT mesh 上做 reconstruction（非生成，纯表示能力测试）：

| Resolution | 32 | 32 | 64 | 64 | 128 | 128 | 256 | 256 |
|---|---|---|---|---|---|---|---|---|
| Method | CD | Stor | CD | Stor | CD | Stor | CD | Stor |
| Matryoshka | 7.59 | 10 | 2.43 | 25 | 1.43 | 78 | 0.95 | 261 |
| UV Mapping | 6.28 | 8 | 2.29 | 32 | 1.16 | 128 | 0.88 | 512 |
| **Ours** | **2.66** | **5** | **1.58** | **16** | **0.96** | **56** | **0.85** | **194** |

CD ×10⁻³，Storage 单位应该是 MB 或类似。SP 在低 resolution 时优势更明显（32 时 CD 是 Matryoshka 的 35%）。原因是 SP 没有 boundary stitching 问题，而 Matryoshka 受 voxel bottleneck 限制（$N^3$ vs $N^2$），UV Mapping 虽然也是 $N^2$ 但有非唯一 cut。

### 7.5 与 large foundation model 的对比（Table 6，关键 efficiency 数据）

| Method | GPU | Time | Iter | Data | Latency | CD↓ | IoU↑ | F-Score%↑ |
|---|---|---|---|---|---|---|---|---|
| CLAY | 256 A800 (10 TB) | ~2 weeks | - | 527k | ~15s | 0.0046 | 0.6355 | 96.95 |
| Trellis | 64 A100 (2.5 TB) | - | 400k | 500k | ~40s | 0.0030 | 0.6495 | 98.35 |
| Hunyuan3D-2 | - | - | - | - | ~15s | 0.0028 | 0.7440 | 98.43 |
| TripoSG | 160 A100 (6.25 TB) | ~3 weeks | 700k | 2m | ~50s | 0.0030 | 0.7381 | 99.08 |
| TripoSF | 64 A100 (2.5 TB) | - | - | 400k | - | - | - | - |
| **Ours** | **2 GPU (0.09 TB)** | **~1 week** | **80k** | **160k** | **6–10s** | 0.0034 | 0.6208 | 98.28 |

**SPGen 用 <5% 的训练资源达到与 large foundation models 相当的 F-Score (98.28% vs 98.35% Trellis)**，CD/IoU 略差但仍 competitive，latency 显著更优。这是 image-based representation + pretrained prior 的胜利——把 3D 问题降维到 2D，直接利用 SDXL 数十亿图像的 prior。

相关 links：
- CLAY: https://arxiv.org/abs/2406.10163 (CLAY paper)
- Trellis: https://arxiv.org/abs/2412.01506
- Hunyuan3D-2: https://arxiv.org/abs/2501.12202
- TripoSG: https://arxiv.org/abs/2502.06608

---

## 8. Ablations 深度分析

### 8.1 Component ablation (Table 4)

| Variant | CD↓ | IoU↑ | F-Score%↑ |
|---|---|---|---|
| Full | 0.0051 | 0.5407 | 95.57 |
| w.o. LSA | 0.0436 | 0.2568 | 60.75 |
| w.o. finetuning AE | 0.0610 | 0.2072 | 52.04 |
| w.o. finetuning UNet | 0.1742 | 0.1034 | 27.42 |

Insights：
- 不 finetune UNet 几乎完全失效（F-Score 27%），说明 SDXL 的 RGB prior 无法直接迁移到 SP depth map，必须 finetune
- 不 finetune AE 也掉很多，因为 SDXL 的 VAE 是为 RGB 设计的，SP depth 分布完全不同
- LSA 对 multi-layer 一致性至关重要

### 8.2 SD prior 的作用 (Fig. 14, supplementary C.1)

对比从 scratch 训练 vs 从 SDXL 预训练 weight finetune：
- 15k step test loss: $4.37 \times 10^{-5}$ (with prior) vs $3.08 \times 10^{-4}$ (no prior)，约 **7x 加速收敛**
- Step 3k 的可视化显示，with SD prior 的 SP map 已经有清晰结构，no prior 的还全是噪声

这印证了 paper 核心 thesis：SP map 的 image-domain formulation 让 model 能直接继承 SD 的 locality、semantic、implicit symmetry 等 prior。

### 8.3 Border consistency (C.2)

Circular padding 后 azimuth border 的 absolute-relative-error 仅 0.23%，证明 SP map 的 360° 周期性被 model 正确学到。

### 8.4 Distortion analysis (C.4)

Polar area 0.20%, equator 0.25%, 平均 0.22% 的 absolute-relative-error。Equirectangular projection 在极区确实有 over-sampling distortion，但 SPGen 的 model 学到了处理这种 distortion，所以 3D 重建误差可控。

---

## 9. Texture Generation 扩展 (Supplementary A.2)

paper 还展示了 SP map 可扩展到其他 surface attribute（texture、normal、curvature）。具体做法：

1. **Cascade pipeline** (Fig. 11)：
   - Stage 1：用前述 SPGen 生成 SP depth maps
   - Stage 2：把 SP depth latents 加噪作为 condition，再加 single-view image embedding，condition 另一个 UNet 生成 SP color map
   
2. **Cross-Domain Self-Attention** (借鉴 [MVDream](https://arxiv.org/abs/2308.16512))：
   - 先把 noisy depth latent 喂进 color UNet 记录 hidden states
   - 再 sample 纯噪声进同一 UNet，与 depth hidden states 做 cross-domain self-attention
   - 保证 color 与 depth 在 SP map 上对齐

3. 为什么不 shared UNet？
   - RGB 和 depth 在 perspective image 上还共享 contour/shape 结构
   - 但在 SP map 上，texture complex 的 simple shape 和 simple texture 的 complex shape 会让两个 domain 分布完全不同
   - 共享会导致 negative transfer，所以 decouple

小规模实验（2k Objaverse subset）显示可行性，但 paper 没大规模训练 due to 资源限制。

---

## 10. 与 Related Work 的深度对比

### 10.1 vs Multi-view diffusion (Wonder3D, Zero123, SyncDreamer, Zero-1-to-G)

- Multi-view：6–24 perspective views，每个 view partial coverage，overlap 区域靠 cross-attention 软对齐
- SP：单一 panorama view 全覆盖，injective mapping 保证 no conflict
- Multi-view 无法表达 self-occlusion 后的 internal layer（如杯子内壁），SP multi-layer 可以

### 10.2 vs Geometry Image / UV Mapping

- Geometry Image ([Gu et al. 2002](https://arxiv.org/abs/2308.14686)) 和 UV Mapping ([Yan et al. 2024b](https://arxiv.org/abs/2408.03178)) 都需要把 mesh cut 后展开
- Cut 是 non-unique：同一个 mesh 有无数种 cut 方式，model 要学 cut invariant 很难
- Genus > 0 物体的 cut 很复杂，boundary stitching 是大问题
- SP map 用固定 mapping（球面投影），no cut，no stitching，scalable

### 10.3 vs Matryoshka Network ([Richter & Roth 2018](https://arxiv.org/abs/1803.06072))

- Matryoshka 用 6 个 axis-aligned stack 的 nested depth map，最后 fuse 成 voxel grid
- Voxel bottleneck：surface detail 受 $N^3$ 限制，SP 直接 $N^2$
- 6 个 stack 之间无 explicit consistency constraint，容易产生 hole 或 jagged artifact
- SP 是 single panorama + LSA，naturally consistent

### 10.4 vs GenRe ([Zhang et al. 2018](https://arxiv.org/abs/1709.05517))

- GenRe 是早期 spherical map 工作，single layer
- 用 cascaded inpainting + voxel refinement
- 依赖外部 depth estimator，error accumulation
- SPGen 是 end-to-end，multi-layer，无 voxel bottleneck

### 10.5 vs Geometry-based (CLAY, Trellis, Hunyuan3D-2, TripoSG, MeshGPT, Direct3D)

- 这些方法直接在 3D latent / point cloud / mesh face 上做 diffusion
- 优势：3D native，不需要 projection 转换
- 劣势：
  - 需要 3D native data preprocessing（SDF computation, point sampling, mesh tokenization），noisy
  - 没有 2D image 的 billion-scale prior，要从头学 symmetry、locality
  - 训练资源巨大（256 A800, 2 weeks for CLAY）
- SPGen 把问题降维到 2D，直接复用 SDXL prior，efficiency 优势巨大

### 10.6 vs LRM-based (InstantMesh, LGM, CRM, OpenLRM)

- LRM 类用 transformer feed-forward 单图 → triplane / Gaussians / voxel
- Fast inference but 需要 large 3D training data
- 没 image diffusion 的生成多样性，更像 reconstruction
- SPGen 保持 generation nature（diffusion sampling）+ 更高质量

---

## 11. Limitations

1. **Faces 平行于 ray 方向**：当 mesh face 正好与 sphere radius 平行时（如 hemisphere 的 flat boundary），ray-face intersection 数值不稳定。paper 用 [Möller-Trumbore algorithm](https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm) 检测并 discard 这种 case，必要时加 random perturbation。

2. **Polar distortion**：Equirectangular projection 在极区 over-sampling，理论上损失精度。但 ablation C.4 显示 0.20% error，controllable。

3. **Texture extension 未大规模验证**：A.2 只在 2k subset 上做了 feasibility study。

4. **依赖 SDXL 架构**：换 backbone 可能效果不同，paper 没探索 DiT-based 2D model。

5. **Layer 数固定**：4 layers 是 empirical 选择，对极复杂物体（如 fractal 结构）可能不足。

---

## 12. Intuition 总结：为什么 SPGen work

让我尝试 build 一个完整的 mental model：

### 12.1 表示层面：SP map 是 3D surface 的"完美 2D 编码"

3D surface 是 2D manifold，理论上可以 injective 映射到 2D domain（拓扑学保证）。但大多数 mapping（geometry image、UV atlas）需要 cut，破坏 injectivity。SP map 利用 unit sphere 作为中间媒介，把 surface 包在内部，从 origin cast ray 投影——只要 surface 是 star-shaped-ish（对原点可见），就是 injective。Multi-layer extension 处理 non-star-shaped 部分（self-occlusion），把 ray 的所有 intersection 都记录。

关键点：**SP map 在 representation 层面就消除了 multi-view 的不一致可能**。Generator 只需要学一个 mapping from image to SP map，不需要学跨 view consistency。

### 12.2 生成层面：复用 2D prior

SP map 是 structured 2D image，有 locality（相邻 pixel 对应相邻 ray）、implicit symmetry（rotation 对应 SP map shift）、semantic patterns（chair腿、桌面在 SP map 上的位置可预测）。这些恰好是 2D diffusion model 在 billion-scale image 上学到的 prior。所以 finetune SDXL 几乎是 "免费" 拿到这些 prior。

对比：geometry-based 方法（CLAY 等）要学 3D 的 locality、symmetry，没有现成 prior，必须从头训，所以需要 256 A800 × 2 weeks。

### 12.3 Geometry Regularization：弥补 L1 loss 的偏见

L1/L2 loss 是 pixel-uniform 的，但 SP map 上的几何信息分布是 non-uniform 的——edge 是 high information density 区域。Sobel + FFT dual regularizer 强迫 model 关注 edge 和高频，本质是 **importance-weighted reconstruction loss**，但 importance 来自几何 prior 而非任务 prior。

### 12.4 LSA：multi-layer 的 "soft geometric constraint"

4 layers 的 SP map 之间有强约束：layer $i$ 的 depth ≤ layer $i-1$ 的 depth（outer 要近，inner 要远）。LSA 通过 self-attention 让 model 隐式学到这个约束，比 hard rule 更灵活（允许 model 处理无 inner intersection 的 ray）。

### 12.5 Efficiency：image-domain 是 free lunch

整个 pipeline 的 efficiency 来自三处：
- 表示：$N^2$ vs voxel 的 $N^3$
- Prior：SDXL 的 billion-image 知识免训
- Finetune：只调 AE + UNet，不动 SDXL 主架构

合计：2 GPU × 1 week vs 256 A800 × 2 weeks，~100x 资源节省。

---

## 13. 个人联想与延伸

1. **Spherical CNNs 的复兴**：早期 [Cohen et al. Spherical CNNs](https://arxiv.org/abs/1801.10130)、[U-GCN](https://arxiv.org/abs/1904.02419) 等工作在 sphere 上做 equi-invariant CNN，但没 scale 起来。SPGen 用 equirectangular + circular padding 绕过了 spherical convolution 的复杂度，直接用平面 CNN，是个 pragmatic 的选择。

2. **类似 panoramic generation**：[Diffusion360](https://arxiv.org/abs/2311.13141)、[HORIZON](https://arxiv.org/abs/2403.12918)、[MVDream](https://arxiv.org/abs/2308.16512) 等都用过 equirectangular representation，但都做 scene generation，SPGen 是第一个用 SP 做 object-level 3D shape generation 并 multi-layer 化的。

3. **Layered Depth Images (LDI)**：[LDI (Shade et al. 1998)](https://graphics.stanford.edu/papers/ldi/ldi.pdf) 是 1998 年 SIGGRAPH 的经典工作，single view + multiple depth per ray。SPGen 的 multi-layer SP map 本质是 LDI 的 spherical 版本。这个 idea 等了 25 年才被结合到 diffusion pipeline。

4. **Possible extension：sparse view SP**：如果输入是 sparse multi-view 而非 single view，可以联合 condition 多 view embedding → generate 更高保真 SP map，可能击败 LRM 类方法。

5. **Possible extension：video-to-4D**：用 video frame sequence condition SP map sequence，每帧一个 SP map，自然支持 4D dynamic shape generation。

6. **Rectified Flow 替代 DDPM**：[TripoSG](https://arxiv.org/abs/2502.06608) 用 rectified flow 训 3D generation 效果很好。SPGen 用 DDPM/Euler Ancestral，换成 flow matching 可能 inference 更快、质量更高。

7. **SDXL vs DiT**：SDXL 是 UNet-based，[PixArt](https://arxiv.org/abs/2310.00426)、[Stable Diffusion 3](https://arxiv.org/abs/2403.03206) 用 DiT。DiT 在 image generation 上已经超过 UNet，SPGen 换成 DiT backbone 可能进一步提升。

8. **与 3D Gaussian Splatting 结合**：SP color map + SP depth map 可以直接 unproject 到 3D Gaussians，可能比 mesh 更适合 rendering。参考 [Zero-1-to-G](https://arxiv.org/abs/2501.05427)。

9. **Polar distortion 的根本解决**：可以用 [Cube2Sphere projection](https://en.wikipedia.org/wiki/Cube_mapping) 替代 equirectangular，6 个 cube face 上均匀采样，无 polar distortion。但破坏了 single 2D image 的简洁性。

10. **Generative vs Reconstruction 的 trade-off**：SPGen 保持 diffusion sampling 的生成多样性，但 inference 6–10s 比 LRM feed-forward 慢。可以探索 consistency model 或 distillation 把 SP diffusion 蒸馏成 feed-forward。

---

## 14. 总结

SPGen 的核心贡献是 **找到了一个"刚刚好"的 2D 表示**——既保留了 3D 的完整几何信息（multi-layer + injective），又能直接 plug-in 到成熟 2D diffusion pipeline（SDXL），还避免了 multi-view、geometry image、UV mapping 的各种 stitching/consistency 问题。配合 geometry regularization（edge + spectral loss）和 LSA，在 2 GPU 上达到 SOTA 几何质量。

这个工作让我想到一个 general lesson：**representation design 是 generative model 的 leverage point**。在 2D image generation 上大家已经接受了 VAE latent + diffusion 的范式，3D 的 representation 还在战国时代。SPGen 提供了一个 elegant 的 image-domain representation 选项，可能成为后续 3D foundation model 的重要参考。

Reference links 汇总：
- Paper: https://doi.org/10.1145/3757377.3763959
- SDXL: https://arxiv.org/abs/2307.01952
- DINOv2: https://arxiv.org/abs/2304.07193
- Wonder3D: https://arxiv.org/abs/2310.15008 (开源: https://github.com/xxlong0/Wonder3D)
- InstantMesh: https://arxiv.org/abs/2404.07191
- LGM: https://arxiv.org/abs/2402.05063
- CRM: https://arxiv.org/abs/2403.07414 (推测)
- SurfD: https://arxiv.org/abs/2403.04132
- CLAY: https://arxiv.org/abs/2406.10163
- Trellis: https://arxiv.org/abs/2412.01506
- Hunyuan3D-2: https://arxiv.org/abs/2501.12202
- TripoSG: https://arxiv.org/abs/2502.06608
- MeshUDF: https://arxiv.org/abs/2206.08797
- ConvONet: https://arxiv.org/abs/2007.12484
- Focal Frequency Loss: https://arxiv.org/abs/2012.08114
- Matryoshka: https://arxiv.org/abs/1803.06072
- GenRe: https://arxiv.org/abs/1709.05517 (推测)
- Geometry Image: https://dl.acm.org/doi/10.1145/566570.566594
- Möller-Trumbore: https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
- Stable Diffusion 3 / DiT: https://arxiv.org/abs/2403.03206
- PixArt: https://arxiv.org/abs/2310.00426
- MVDream: https://arxiv.org/abs/2308.16512
- Zero-1-to-G: https://arxiv.org/abs/2501.05427
