---
source_pdf: DifFRelight.pdf
paper_sha256: a8680f4abbf01d1a1cdbab36975b1718e6e7e3aa05acbdfeaca2a7cd52dac77c
processed_at: '2026-08-03T21:15:25-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej, 咱们坐下来用最直白的话聊聊 Netflix 这帮人到底在 DifFRelight 里干了什么。抛开那些学术包装，这篇 paper 的核心就是：**怎么把一个在 flat lighting 下拍出来的 boring 的 3D 活人视频，后期打上任意帅气的电影级灯光，而且连一根头发丝的 shadow 和眼球的高光都不穿帮。**

Volumetric performance capture 很牛，一圈摄像机把人拍成 3D，但是拍的时候为了看清细节，灯光必须是 flat 的 (全场打亮)。如果要放进赛博朋克街道或者火星表面，这种 flat lighting 就完蛋了。传统做法是极快地交替切换灯光拍视频 (time-multiplexed)，演员得像机器人一样不动，设备巨贵且反人类。

DifFRelight 的思路极度 pragmatic：既然我有 Stable Diffusion 这个见过几十亿张图、什么光照都见过的超级大脑，我能不能用一个小 dataset，教会它把特定的 flat-lit 人脸翻译成特定灯光下的人脸？

### 1. 数据采集：暴力但有效的 OLAT
他们在 stage 里摆了一圈 LED 面板。演员做一个表情，他们只亮第一块面板，拍一圈 75 个视角的照片；然后亮第二块，再拍一圈。这就是 One-Light-At-a-Time (OLAT)。123 块面板，30 个表情，建出了一个 subject-specific 的专属 dataset。

在这里，每一帧数据就变成了一个 triplet: `{I_FlatLit, I_OLAT, d}`。`I_FlatLit` 是全场打亮的图，`I_OLAT` 是只有面板 `d` 方向亮着的图，`d` 是个 3D 向量。

### 2. 把 Diffusion 模型改造成 Relighting 引擎
他们拿了 Stable Diffusion v2.1。原本它的 U-Net 是接收一个 random noise 的 latent，加上一段 text prompt 的 embedding，去预测 noise。

**怎么把 flat-lit 图像喂进去？**
他们绕开了 ControlNet 的架构，直接采用最暴力的通道拼合。把 flat-lit 图像用 VAE encoder 压成 latent，和 random noise 的 latent 直接在 channel 维度拼起来。U-Net 的第一层 conv layer 的输入通道数直接 doubling (从 4 通道变 8 通道)。后面的 weights 全部复用 pre-trained 的 SD。这样做最大限度地保留了 SD 对自然图像结构的 prior，避免了从头训练。

**怎么告诉模型灯光在哪？**
灯光方向 `d` 是个 3 维向量，信息量太少。他们用 Spherical Harmonics (SH) 把它展开成 16 维 (degree=3)，增加频率带宽。然后补零 zero-pad 到 text embedding 的长度。
公式这样写：
$$\mathbf{s}_d = \mathbf{0} \oplus \mathcal{Y}(\mathbf{d})$$
- `s_d`: 最终送进 cross-attention 的 condition embedding。
- `0`: 零矩阵，纯粹为了凑长度，让 pre-trained 的 projection layer 不用改维度。
- `Y(d)`: SH encoding，把 3D 向量映射到高维空间。
- `⊕`: 拼接符号 concatenation。

模型训练的 loss 就是标准的 diffusion noise prediction loss：
$$\mathcal{L}_{\text{diffusion}} = \| \hat{\epsilon}(\mathbf{z}_{OLAT}^{(t)} \oplus \mathcal{E}(\mathbf{I}_{FlatLit}); \mathbf{s}_d, t) - \epsilon \|_2^2$$
- `ε̂`: U-Net 预测出来的 noise。
- `z_OLAT^(t)`: 在 timestep `t` 加了噪声的 ground truth OLAT latent。
- `E(I_FlatLit)`: flat-lit 图像的 latent。
- `ε`: 实际加进去的 random noise。
- `t`: diffusion 的当前时间步。

**一个超级关键的工程细节：Pyramid Noise**
这帮人发现，如果用标准的 zero-meaned Gaussian noise，预测出来的图像会发生严重的 color shift。因为 OLAT 图像中，大面积的区域是纯黑的（没有被那个 LED 面板照到的地方）。U-Net 有个 inductive bias，倾向于把事情拉向平均值，这就导致暗部区域预测不准。
为了解决这个，他们使用了 Pyramid Noise。这种噪声在不同 spatial resolution 上叠加了结构化的低频偏移。它强行让网络学会预测 dark pixels 的绝对 intensity，而不是只去预测相对的均值。
实验数据很夸张：用了 Pyramid Noise，PSNR 是 30.04；不用，PSNR 掉到 26.55。整整差了 3 个多 dB。

### 3. 解决长视频的 3DGS 漂移问题
视频有几百帧，你不能对每一帧独立做 relighting，否则帧与帧之间会疯狂闪烁。你需要一个 temporal-consistent 的 flat-lit input。他们用了 deformable 3D Gaussian Splatting (3DGS)。但是长序列里，一套全局的 canonical Gaussians 根本表达不了所有的微表情和头发飞舞。

他们搞了个 Two-stage 的分段法：
**Stage 1**: 在整个视频里均匀挑 K 个 keyframes，只用这 K 帧训练一个 deformable 3DGS 当作 "Base Model"。
**Stage 2**: 把视频按 keyframes 切成几段。每段单独训练一个 3DGS，但是初始化用 Base Model 在这段第一帧时刻的变形状态。这样保证每段的起点都对齐了。

为了防止段与段之间的接缝处不连续，他们在 warm-up 阶段加了一个 L2 正则：
$$\mathcal{L}_{reg} = \| \delta'_{t_{k_0}} - \delta_{t_{k_0}} \|_2 + \| \delta'_{t_{k_1}} - \delta_{t_{k_1}} \|_2$$
- `δ'_{t_k0}` 和 `δ'_{t_k1}`: 当前段网络预测的在第 0 个和第 1 个 keyframe 时刻的 deformation offset。
- `δ_{t_k0}` 和 `δ_{t_k1}`: 初始化时从 Base Model 拿来的固定 offset。
这个正则项强迫新训练的段在边界处必须乖乖待在 Base Model 规定的位置，不能乱跑。

### 4. 模拟面光源和 HDRI
OLAT 只是一个小 LED 面板，相当于点光源，打出来的高光很硬。如果想要电影里的柔光怎么办？
他们引入了一个 light size factor `β`。`β=0` 就是点光源，`β=1` 就是 flat-lit。训练时，他们把多个 OLAT 图像加权混合，模拟出一个 Spherical Gaussian (SG) 面光源。

SG 的 sharpness `λ` 和 size 有关：
$$\lambda = \frac{-\cos(\theta)}{\cos(\theta)^2 - 1}$$
- `λ`: 面光源的锐度。`λ` 越大光越集中，`λ` 越小光越散。
- `θ`: 拐点角度，在 `1°` 到 `89°` 之间线性插值。

到了 HDRI 环境光，思路就更暴力了：把一张 HDRI 贴图拆成 123 个方向的 OLAT，或者拆成 15 个 Spherical Gaussian。每个方向单独跑一次模型推理，然后把这些生成的图在 linear space 里按照 HDRI 的亮度权重加起来。

### 5. 最终总结一下 Intuition
这篇 paper 成功的原因，在于它巧妙地把一个困难的物理渲染问题（subsurface scattering, 各向异性头发反射），转化成了一个纯 data-driven 的图像翻译问题，并且利用了 Stable Diffusion 的强大结构先验。

它最聪明的设计有三点：
1. 不用 ControlNet，直接 channel concat flat-lit latent 和 noise，最大限度保护了 SD 的 pre-trained weights。
2. 把 light direction 塞进 SH encoding 然后走 cross-attention，完美匹配了原本 text prompt 的作用机制。
3. Pyramid noise 解决了 relighting 任务里 dark pixels 的均值回归问题，这是一个极度贴合 task 特性的工程 trick。

**Web Links for Reference:**
- DifFRelight Paper on arXiv: https://arxiv.org/abs/2410.05863
- Stable Diffusion 2.1 Model: https://huggingface.co/stabilityai/stable-diffusion-2-1
- Original 3D Gaussian Splatting: https://repo.sam-lab.ethz.ch/sam/3dgaussian_splatting
- Deformable 3DGS Paper: https://arxiv.org/abs/2312.15059
- Pyramid Noise for Diffusion Models: https://wandb.ai/johnowhitaker/multires_noise/reports/Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2
- Offset Noise (Color shift issues): https://www.crosslabs.org/blog/diffusion-with-offset-noise

---

# DifFRelight 深度技术讲解

 Andrej, 这篇来自 Netflix Eyeline Studios 的 SIGGRAPH Asia 2024 paper 是一个把 latent diffusion prior 跟 volumetric performance capture 做深度耦合的 production-grade 系统。我尽量把每个技术细节都拆开讲，让你 build 起直觉。

Paper 链接: https://doi.org/10.1145/3680528.3687644
Project / arXiv: https://arxiv.org/abs/2410.05863 (DifFRelight)

---

## 1. 问题背景与 motivation

Volumetric performance capture (volcap) 用一圈向内的 cameras 捕捉动态 human performance,绕开了 traditional 3D character modeling / rigging / animation 的 pipeline。痛点:大多数 capture 都在单一 flat lighting 下完成,导致 rendered performance 无法融入 cinematic lighting 场景。

Legacy 解决方案比如 [Guo et al. 2019] (The Relightables) 用 time-multiplexed lighting,逐帧交替不同 lighting condition,再通过 ML inference [Meka et al. 2020] 做 relighting。这种方案要求 high-speed global-shutter cameras,演员不舒服,数据量爆炸。

DifFRelight 的核心 insight:**把 flat-lit performance 当作"先验几何 + albedo 表达",用 subject-specific 的 OLAT 数据 fine-tune 一个 diffusion model 学会 flat→directional 的 reflectance field 转换**。这绕开了物理渲染对 subsurface scattering / anisotropic hair 之类的难题,直接 data-driven。

---

## 2. 数据采集系统

### 2.1 Capture stage 硬件

- **LED panel dome**: 16 columns × 5 rows = 80 wall panels,每 panel 50cm × 50cm,型号 ROE BP2v2
- **Ceiling**: 6 × 5 = 30 panels
- **Floor**: 10 panels 提供额外 lighting angle
- **Volume 尺寸**: 高 250cm,直径 276cm
- **Cameras**: 75 台同步 4K Z-CAM e2 cinema cameras,透过 5cm gaps 窥视
- **Capture**: 24 fps 同步,LED panel 顺序点亮做 OLAT

每个 LED panel 从 stage center 看大约 20° cone,这比 point light 产生更少 aliasing 的 shadow,同时仍能捕捉 eye highlights 和 sharp specular reflection。这是数据采集端的关键设计选择:用 area light 的"软"特性换取 shadow quality。

### 2.2 数据规格

- **4 subjects**,每人:
  - 30 expressions (30E)
  - 123 OLATs (123L) + 13 flat-lit tracking frames interspersed
  - 75 views (75V)
- **Training split**: 27E × 115L × 69V 的笛卡尔积
- **Validation split**: 剩余的 E/L/V 组合
- 还拍了一个 clean plate 用于 background matting (3DGS 重建时用)

注意:OLAT sequence 期间演员会有微小运动,所以他们用 optical flow 在 image space 把每帧对齐到 flat-lit frame [Meka et al. 2019]。

### 2.3 颜色空间处理

Linear → sRGB 转换。原因是 pre-trained Stable Diffusion 是在 sRGB 域训练的,domain alignment 对 fine-tuning 收敛很关键。这点很多 relighting 工作都忽略了。

---

## 3. Diffusion Relighting 架构 (核心创新)

### 3.1 整体 formulation

数据集定义为 {I_FlatLit, I_OLAT, d},其中 d 是从 stage center 指向 panel center 的方向向量 (camera space)。

任务是学一个 mapping: I_FlatLit, d → I_OLAT,并 generalize 到 novel (view, expression, lighting)。

### 3.2 Stable Diffusion 基础回顾

Stable Diffusion v2.1 包含:
- **VAE encoder E**: image I → latent z = E(I)
- **VAE decoder D**: z → I ≈ D(z)
- **U-Net ε̂(z^(t); s, t)**: 预测 noisy latent z^(t) 在 timestep t 的 noise,conditioned on text embedding s

Latent space 通常 spatial downsample 8× (例如 512×512 → 64×64)。

### 3.3 Spatial conditioning 设计

这里有个非常聪明的设计。考虑两种选择:
- 用 ControlNet 把 flat-lit 当 condition
- 直接 concatenate flat-lit latent + noise 作为 U-Net input

DifFRelight 选了后者,只把 first conv layer 的 input channels **doubling**(原 4 channels = 1 latent noise,变成 8 channels = 1 latent noise + 1 flat-lit latent)。

**为什么这么做**:
1. 最大保留 pre-trained weights,只改一层 conv 的 weight
2. Spatial alignment 通过 concatenation 直接传给 U-Net 早期层,而不是通过 cross-attention 间接传递
3. ControlNet 在 ablation 里表现最差 (PSNR=20.30 vs Ours=30.29),因为它只能用 sparse spatial info (flat-lit image + diffuse shading map),而 diffusion U-Net 内部 expectation 是 dense spatial structure

对比 ablation 里 ControlNet baseline,他们 feed 的是 3-channel flat-lit + 1-channel diffuse shading map (light direction · photometric normal)。ControlNet 在这里有点"杀鸡用牛刀",它的设计假设是 condition signal 是 sparse hint,但 flat-lit 实际上是 dense full image,信息密度不匹配。

### 3.4 Global lighting conditioning:SH encoding 替换 text embedding

公式 (1):
$$\mathbf{s}_d = \mathbf{0} \oplus \mathcal{Y}(\mathbf{d})$$

- **s_d**: 最终送进 cross-attention 的 embedding
- **0**: zero-padding,长度补到 text embedding 维度 (Stable Diffusion 的 text encoder 输出长度,通常是 77 tokens × 1024 dims)
- **⊕**: concatenation
- **Y(d)**: Spherical Harmonics encoding of light direction d
- **SH degree = 3** (实测最佳,见 ablation)

直觉:light direction 是 3D 单位向量,只有 3 个自由度,但通过 SH 展开到 degree 3 能到 16 维 (1+3+5+7=16),频率足够表达 20° cone 的 angular variation。degree 2 (9 维) 太低,degree 5/7 反而过拟合或与 dataset angular sampling rate 不匹配。

**为什么 zero-pad 到 text embedding 长度**:cross-attention 的 key/value projection 是 pre-trained 的,要保持 sequence length 一致才能复用 weights。zero-padding 部分相当于"无意义的占位 token",fine-tuning 会让 attention 学会忽略它们。这点我从 Marigold [Ke et al. 2023] 那里见过类似手法。

### 3.5 训练 objective

公式 (2):
$$\mathcal{L}_{\text{diffusion}} = \| \hat{\epsilon}(\mathbf{z}_{OLAT}^{(t)} \oplus \mathcal{E}(\mathbf{I}_{FlatLit}); \mathbf{s}_d, t) - \epsilon \|_2^2$$

变量解析:
- **ε̂**: U-Net noise predictor
- **z_OLAT^(t)**: noisy ground truth latent at timestep t
- **E(I_FlatLit)**: VAE-encoded flat-lit latent (spatial condition)
- **⊕**: channel-wise concatenation → 形成 8-channel input
- **s_d**: light direction 的 SH embedding (global condition,通过 cross-attention)
- **t**: diffusion timestep
- **ε**: target noise

公式 (3):
$$\mathbf{z}_{OLAT}^{(t)} = \sqrt{\alpha_t} \mathcal{E}(\mathbf{I}_{OLAT}) + \sqrt{1-\alpha_t} \epsilon$$

- **α_t**: predefined scheduler value, 控制保留 ground truth 信号 vs noise 的比例
- **E(I_OLAT)**: ground truth OLAT image 的 VAE latent
- **ε**: random noise

### 3.6 Pyramid noise 的关键作用

这是个非常 practical 的发现。Standard DDPM 用 zero-meaned Gaussian noise ε ~ N(0, I)。但 DifFRelight 发现这会导致 substantial color shift (预测图像偏色)。

原因分析 [Guttenberg 2023]: diffusion U-Net 在 denoise 时,如果 noise 是 zero-meaned,网络的 inductive bias 倾向于预测 zero-meaned residual,导致 dark regions (低 intensity pixels) 难以准确恢复。对于 relighting 来说,OLAT 图像大区域是 dark (没被点亮的 panel 对应区域),这个问题尤其严重。

**Pyramid noise** [Whitaker 2024] 是 multi-resolution noise,在不同 spatial frequency 上叠加 noise。直觉上,它给 dark regions 加了 structured low-frequency offset,让 network 容易预测 dark pixel 的 absolute level。

Ablation (Tab. 2):
- With pyramid noise: PSNR=30.04
- Without pyramid noise: PSNR=26.55 (**下降 3.49 dB**)

这是非常显著的差距。Fig. 5 的可视化显示不用 pyramid noise 时颜色偏移明显。

### 3.7 VAE freezing

训练时冻结 VAE encoder E 和 decoder D。理由:flat-lit 和 OLAT 都在 natural image distribution 内,应该在 VAE latent manifold 上。解冻 VAE 在小数据集上会破坏 pre-trained representation。

---

## 4. Scalable Dynamic 3DGS (第二核心创新)

### 4.1 为什么需要这个

Diffusion relighting 是 image-based 的,如果输入 flat-lit frames 时序不一致,输出的 relit frames 也会闪烁。所以需要 temporal-consistent flat-lit rendering。

Deformable 3DGS [Jung et al. 2023] 用 canonical 3D Gaussians + deformation field δ(x, t) → (δx, δr, δs)。问题:canonical Gaussians 全局共享,长序列 (10-15 秒 = 240-360 帧) 下表达力不够,要么 Gaussians 数量爆炸,要么细节丢失。

### 4.2 Two-stage training

**Stage 1**: 均匀采样 K keyframes,只在这 K frames 上训练一个 deformable 3DGS,得到 pretrained K-frame model。

**Stage 2**: 用 K keyframes 作为 transition points,把整个 sequence 分成 K-1 segments (每个 segment 包含相邻 2 keyframes)。每个 segment 独立训练一个 deformable 3DGS,但初始化来自 K-frame model 在该 segment 第一帧时刻 t_{k_0} 的 deformation。

具体地:Stage 1 模型在时间 t_{k_0} 输出 deformation offset,把 canonical Gaussians 变形到 t_{k_0} 时刻的状态,作为 segment 的初始 Gaussians。这保证不同 segment 的初始 state 在 transition point 上对齐。

**Warm-up phase** (3000 iters): 固定 Gaussians 不动,只训练 deformation network,加 L_reg 正则:

公式 (4):
$$\mathcal{L}_{reg} = \| \delta'_{t_{k_0}} - \delta_{t_{k_0}} \|_2 + \| \delta'_{t_{k_1}} - \delta_{t_{k_1}} \|_2$$

- **δ_{t_{k_0}}, δ_{t_{k_1}}**: 初始 deformation offset (从 K-frame model 来)
- **δ'_{t_{k_0}}, δ'_{t_{k_1}}**: 当前 learned offset,包括对 position x, rotation r, scaling s 的 offset

这个正则确保 segment 在两端 transition point 上的 deformation 不会偏离 K-frame model 的预测太多。

**Densification phase** (10000 iters): 放松约束,允许 Gaussians clone/split/prune 做细节重建。

**Joint optimization**: 最后固定 Gaussians 数量,联合训练 deformation network + Gaussians。

### 4.3 Ablation 数据 (Tab. 3)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|
| Baseline deformable 3DGS [Jung 2023] | 30.202 | 0.876 | 0.254 |
| Partition w/o K-frame | 31.717 | 0.893 | 0.239 |
| Partition w/ K-frame (full method) | 31.114 | 0.901 | 0.218 |

有意思的是 "Partition w/o K-frame" 在 PSNR 上比 full method 高 (31.717 vs 31.114),但 LPIPS 反而差 (0.239 vs 0.218)。这说明 K-frame + L_reg 的价值在于 perceptual consistency (LPIPS 低),代价是 single-frame 的 reconstruction 略有 trade-off。

### 4.4 Temporal blending

即便 3DGS 输出 temporal-consistent flat-lit,diffusion inference 仍可能产生 high-frequency temporal inconsistency (因为是 image-based,没有 video prior)。

他们采用 [Jamriška et al. 2019] 的 temporal blending,keyframe step size = 4 (每 4 帧一个 keyframe,中间用 optical flow warp 插值)。这是 post-processing 补救,不是 video diffusion。Limitation 里提到 fast motion 会有 artifact (Fig. 9 双 shirt stripes)。

---

## 5. Unified Lighting Control

### 5.1 Area light representation

引入 light size factor β:
- β = 0: point light (single OLAT)
- β = 1: flat-lit (diffuse, all directions equal)

训练时:
1. Light direction d 乘以 (1 - β) 调制 intensity
2. Ground truth 用多个 OLAT weighted sum 模拟 isotropic Spherical Gaussian illumination

Spherical Gaussian sharpness λ 与 size factor 的关系:

$$\lambda = \frac{-\cos(\theta)}{\cos(\theta)^2 - 1}$$

$$\theta = \alpha \cdot (\theta_{max} - \theta_{min}) + \theta_{min}$$

- **λ**: SG sharpness parameter (越大越尖锐)
- **θ**: SG 的 inflection point (角度参数)
- **α**: 0~1 之间,对应 β 的连续化 (这里我没看到 paper 明确写 α 和 β 的关系,推断 α = 1 - β)
- **θ_min = 1° = π/180**
- **θ_max = 89° = 89π/180**

直觉:λ 越大,SG lobe 越窄,光照越像 point light;λ 越小,SG 越宽,光照越 diffuse。β 从 0 到 1 的扫描对应 SG 从尖锐到平坦。

### 5.2 HDRI composition

两种方式:
1. **OLAT-based**: 把 HDRI lat-long 投影到 123 个 panel 方向上,每个方向取 average intensity 作为 weight,inference 出 123 张 relit image,linear space 加权和
2. **Area-light-based**: 用 15 个 SG 拟合 HDRI lat-long,每个 SG 有 (direction, sharpness, intensity),inference 出 15 张加权和

Animation HDRI: 旋转 HDRI 时同步调整 conditioned directions,保持 subject-camera 相对关系。

Fig. 16 显示 real-world HDRI relighting 结果,需要 color chart calibration 补偿 LED vs real-world spectrum 差异 (3-channel scaling factor)。

---

## 6. 实验结果深度解析

### 6.1 Quantitative comparison (Tab. 1)

四个 validation configuration:
- **Novel Light**: 训练见过的 expression/view,新 lighting direction
- **Novel Expression**: 新 expression,见过 L/V
- **Novel View**: 新 viewpoint
- **Novel LEV**: 三个维度全部 novel (最难)

| Config | Method | PSNR↑ | SSIM↑ | LPIPS↓ | FLIP↓ |
|---|---|---|---|---|---|
| Novel Light | Ours | 30.29 | 0.8212 | 0.1750 | 0.0825 |
| | U-Net | 23.66 | 0.7971 | 0.2988 | 0.1586 |
| | ControlNet | 20.30 | 0.6708 | 0.2281 | 0.2304 |
| Novel LEV | Ours | 30.98 | 0.8229 | 0.1729 | 0.0754 |
| | U-Net | 24.55 | 0.8010 | 0.2961 | 0.1440 |
| | ControlNet | 20.51 | 0.6490 | 0.2301 | 0.2425 |

观察:
- ControlNet LPIPS (0.2281) 比 U-Net (0.2988) 好,但 PSNR/SSIM 差很多 → ControlNet 生成 perceptual OK 但 spatial 错位严重
- U-Net baseline spatial alignment 好 (因为直接 conditioned on flat-lit) 但 blur → PSNR 看起来 OK,LPIPS 差
- Ours 在所有指标上 dominate,diffusion prior 提供了 high-frequency detail recovery

注意 U-Net baseline 用了 foreground mask 训练 (背景 black),而 diffusion-based 没有 mask,背景由 3DGS 给出。Metrics 计算时都做了 background removal 保证公平。

### 6.2 Ablation 关键发现 (Tab. 2)

| Variant | PSNR↑ | SSIM↑ | LPIPS↓ | FLIP↓ |
|---|---|---|---|---|
| Final model | 30.04 | 0.8141 | 0.1719 | 0.0859 |
| w/ shading map | 30.32 | 0.8164 | 0.1686 | 0.0836 |
| w/o pyramid noise | 26.55 | 0.7683 | 0.1776 | 0.1433 |
| w/o pre-trained | 28.26 | 0.8014 | 0.1937 | 0.1041 |

**关键 insight**:
1. **Shading map condition** 略好 (PSNR +0.28) 但需要 photometric normal,要求 time-multiplexed gradient illumination capture。Paper 选 SH encoding 是为了 production 友好,不需要额外硬件。
2. **Pyramid noise 是 critical**: -3.49 dB PSNR,这是颜色保真度的关键
3. **Pre-trained weights 是 critical**: -1.78 dB PSNR,diffusion prior 在小数据集上必需

### 6.3 SH degree ablation (Fig. 19)

degree 2 (9 维) PSNR/SSIM 显著差,degree 3 最佳,degree 5/7 略降。Paper 解释:degree 3 大致匹配 dataset 中 lighting directions 的 angular sampling rate。

直觉:SH degree 决定 conditioning 的 angular frequency bandwidth。Dataset 有 123 个 panel,从 stage center 看 angular resolution 大概 15-20°,对应 SH degree ≈ 3 (因为 SH degree l 能表达 angular frequency l,180°/20° ≈ 9 = degree 2 max angular resolution,但实际需要 oversampling)。Degree 5/7 反而过拟合到 specific lighting configuration。

---

## 7. 模型 generalization

### 7.1 Subject-specific 是 trade-off

Subject-specific 训练最好保留 identity 和 fine details (skin texture, hair, lip color)。但限制了 generalization。

Ablation (Fig. 8):
- **Subject-specific model** 应用到 unseen subject: 能 relight 但继承 training subject 的 skin texture / lip color
- **Three-subject model** 应用到 unseen subject: identity preservation 更好,但仍有 color shift

Paper 建议未来用 diverse multi-person dataset + identity disentanglement。

### 7.2 跟 zero-shot method 对比

Appendix A.5 跟 SwitchLight [Kim et al. 2024] 和 IC-Light [Zhang et al. 2024] 对比:
- **SwitchLight**: physically-based rendering,泛化到任意 subject,但 skin color tone in HDRI reflection 不够自然
- **IC-Light**: Stable Diffusion + background conditioning,temporal inconsistent,无法精确 directional lighting,eye highlight 完全缺失 (Fig. 21, 22)

IC-Light 失败的原因:它用 background image 隐式 conditioning lighting,没有显式 light direction signal。对 360° HDRI 旋转场景无法 work (因为它需要 background image,旋转 HDRI 不会改 background)。

---

## 8. Training / Inference 细节

- **Model**: Stable Diffusion v2.1 + Diffusers library
- **GPUs**: 8 × NVIDIA A100 40GB
- **Iterations**: 100K
- **Batch size**: 8
- **Optimizer**: Adam, lr = 3e-5
- **Training scheduler**: DDPM 1000 steps
- **Inference scheduler**: DDIM 30 steps
- **Inference time**: 5-6 秒/帧 on single A100
- **3DGS**: 20 frames/segment, 40K iterations/segment, full HD resolution

注意 inference 用 DDIM 30 steps 而不是 DDPM 1000 steps,这是 diffusion 加速标准做法。30 步在 relighting task 上质量足够。

---

## 9. 局限性

1. **Temporal consistency**: 没有 video diffusion 训练,依赖 post-processing optical flow blending。Fast motion 会有 artifact。
2. **Subject generalization**: subject-specific 训练,新 subject 有 identity drift。
3. **Full-body**: 当前只 train 了 face,需要 full-body 数据扩展。

---

## 10. 我的整体 intuition 总结

这个 paper 最值得学习的几个点:

1. **Diffusion prior 的"低成本"利用**: 只改 first conv layer 的 input channels (doubling),所有其他 weights 复用 pre-trained Stable Diffusion。这种 minimal modification strategy 在 fine-tuning 小数据集任务上非常实用。

2. **Spatial vs Global conditioning 分离**: Spatial (flat-lit latent) 用 concatenation 进 early layers,Global (light direction SH) 用 cross-attention。这符合信号本质 - spatial structure 应该早期融合,global modulating signal 用 attention。

3. **Pyramid noise 解决 dark region color shift**: 这是个非常 specific 到 relighting 的发现。Relighting task 的 ground truth 有大块 dark region (OLAT 没被照亮的部分),standard zero-meaned Gaussian noise 难以恢复 dark pixel 的 absolute intensity。Pyramid noise 给 dark region 加了 structured low-frequency offset,有效解决这个 inductive bias 问题。

4. **Two-stage 3DGS 把"全局 canonical"问题转化为"分段 canonical + 全局 init"**: 长序列 dynamic 3DGS 的核心矛盾 - canonical space 表达力 vs temporal consistency。K-frame model 提供 global consistency prior,每段独立优化提供表达力,L_reg 在 transition point 锁定一致性。

5. **SH degree = dataset angular sampling rate**: 这是个非常物理的直觉。Conditioning encoding 的 frequency bandwidth 应该匹配数据本身的 frequency。degree 3 对应 123 OLATs 的 angular density,过高过低都差。

6. **Production-aware 的 trade-off**: Paper 主动选了不需要 photometric normal 的方案,虽然 shading map condition 略好 (PSNR +0.28)。这反映了工业界对 capture cost 的敏感性。

Reference links:
- Paper: https://arxiv.org/abs/2410.05863
- Stable Diffusion v2.1: https://huggingface.co/stabilityai/stable-diffusion-2-1
- Deformable 3DGS: https://arxiv.org/abs/2312.15059
- 3DGS original: https://repo.sam-lab.ethz.ch/sam/3dgaussian_splatting
- Marigold (depth estimation with diffusion): https://arxiv.org/abs/2312.02145
- Pyramid noise: https://wandb.ai/johnowhitaker/multires_noise/reports/Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2
- Offset noise (Guttenberg): https://www.crosslabs.org/blog/diffusion-with-offset-noise
- The Relightables (Guo et al. 2019): https://doi.org/10.1145/3355089.3356571
- DifusionRig: https://openaccess.thecvf.com/content/CVPR2023/papers/Ding_DiffusionRig_Learning_Personalized_Priors_for_Facial_Appearance_Editing_CVPR_2023_paper.pdf
- DiFaReli: https://openaccess.thecvf.com/content/ICCV2023/papers/Ponglertnapakorn_DiFaReli_Diffusion_Face_Relighting_ICCV_2023_paper.pdf
- DiLightNet: https://arxiv.org/abs/2402.11929
- IC-Light: https://github.com/lllyasviel/IC-Light
- SwitchLight: https://arxiv.org/abs/2402.18848

如果你想深挖某个细节(比如 SG sharpness 公式推导、3DGS deformation network 的具体 architecture、或 SH encoding 在 cross-attention 里如何与 pre-trained K/V projection 交互),告诉我,我可以展开。
