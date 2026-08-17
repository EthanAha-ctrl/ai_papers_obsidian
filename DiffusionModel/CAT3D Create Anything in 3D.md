---
source_pdf: CAT3D Create Anything in 3D.pdf
paper_sha256: a1207fbb8e2f7bf3f252e4d954706ad7db82d059721fa5b7fcdd5ba7a1c818f0
processed_at: '2026-08-03T15:06:13-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# CAT3D 用人话讲

## 核心思路一句话

你想从一张照片搞出 3D，传统方法都失败。CAT3D 说，那我就用 AI 帮你"拍"几百张不同角度的照片，拍完之后丢给成熟的 3D 重建算法就行了。

就这么简单。

## 为什么这个思路 work

NeRF 这类 3D 重建算法其实很强大，你给它几百张照片它就能还原出极其精细的 3D scene。问题是你手里就一张图，或者三张图，这个 math problem 就 under-determined，无数个 3D scene 都能解释这三张图，算法选哪个都不一定对。

之前所有人的做法都是想办法给 3D 重建加各种 prior，加 regularization，搞 SDS distillation，用 feed-forward network 直接预测 3D。每个方法都只针对一个特定 setting，又慢又复杂。

CAT3D 的 insight 是：**瓶颈根本不在重建算法，瓶颈在 observation 数量不够**。那我就把数量补齐不就完了。

补齐的方式就是让一个 multi-view diffusion model 帮你 generate 几百张 novel view。这个 model 就是"虚拟摄影师"，给它一个视角，它能想象出从其他视角看这个 scene 长什么样。

## Multi-View Diffusion Model 长什么样

拿一个现成的 text-to-image model，类似 Stable Diffusion，往里面塞多张图。

具体怎么塞：原本 model 处理一张图，现在处理 8 张图（1 张你给的 conditional + 7 张要 generate 的 target）。每张图先 VAE encode 成 latent，然后把 camera pose 信息拼到 channel 维度上。关键操作是把原来 2D self-attention 膨胀成 3D self-attention，让 8 张图的 latent 互相能 attend 到。这样 model 就能学到"这张图左边的物体在另一张图里应该在右边"这种 cross-view correlation。

Camera pose 怎么喂进去很讲究。不用一个低维向量（那样 generalize 不好），用 raymap——每个 pixel 对应一条从 camera 出发的 ray，记录 origin 和 direction。这个 raymap 直接告诉你"这个 pixel 在 3D 空间对应哪个方向"，model 不用自己猜，直接学 appearance 就行。Rays 相对于第一张 conditional image 算，这样整个系统对 3D 世界的 rigid 变换 invariant。

训练的时候 conditional images 保持 clean，target images 加 noise，loss 只在 target 上算。8 张图一起训，随机选 1 个 conditional 或者 3 个 conditional，一个 model 同时搞定单图和少视角两个 setting。

## 怎么从 8 张变 960 张

Model 训练时只见过 8 张一组，但推理时要生成几百张。怎么办？

分组采样。把几百个目标视角按位置聚类，每 5 个近的放一组，每组独立 generate。这样每组内部 consistency 强，model 能 hold 住。

单图场景特别麻烦，因为没有任何 anchor 来 ground 整个 scene。所以先用 k-means++ 贪心选 7 个 anchor 视角，这 7 个视角尽量 spread 开覆盖整个 scene。先 generate 这 7 个 anchor，然后把 anchor 当 conditional 去生成剩下的。这就是 block-wise autoregressive，先搭骨架再填肉。

Camera trajectory 也很关键，不是随便绕一圈就行。RealEstate10K 用 spline 拟合 input views 再 offset，LLFF 用 forward-facing circle，CO3D 用 spline 加 scale factor，Mip-NeRF 360 用椭圆 path。每个数据集的 scene 结构不同，trajectory 设计要 match。

## 重建阶段怎么处理 inconsistency

生成的几百张图虽然大体一致，但不是完美 3D consistent。这是 generated views 的通病，连 SOTA video model 都做不到完美 consistent。

所以标准 NeRF 训练不能直接用，要改造：

1. **加 LPIPS loss**：原始 Zip-NeRF 只用 photometric MSE loss，对 pixel 级别不一致敏感。LPIPS 用 deep feature 算 perceptual distance，忽略 high-frequency 细节不一致，保留 semantic 一致。生成的图之间 high-frequency 对不上但 semantic 对得上，正好适合 LPIPS。

2. **Distance-based weighting**：离 captured view 近的 generated view 更可信，远的更不可信。用 Gaussian kernel $w \propto \exp(-b \cdot s^2)$，$s$ 是到最近 captured view 的距离。训练初期 $b=0$ 所有 view 权重一样，后期退火到 $b=15$，近的 view 权重高。直觉是：初期先用所有 generated view 搭出大致 geometry，后期让 captured view 主导细节。

3. **Zip-NeRF 配置缩小**：view-dependence network 缩小，iteration 只跑 1000 次，防 overfitting 也加速。

整个 reconstruction 在 16 张 A100 上跑 55 秒到 4 分钟。

## 为什么快一个数量级

之前 ReconFusion、ZeroNVS 要 1 小时，因为它们要 iterative distillation——在 3D representation 里反复 backprop，每次都要 query diffusion model。

CAT3D 的 prior 用一次 forward pass 就 amortize 完了，generate 完图之后就是个标准 NeRF optimization，没有 gradient through diffusion model 的开销。Amortized inference 替代 iterative inference，这就是快一个数量级的根本原因。

## 结果怎么样

Few-view 重建在 5 个数据集上几乎全 SOTA。DTU 数据集 3-view PSNR 从 ReconFusion 的 20.74 提到 22.02。时间从 1 小时降到几分钟。

单图到 3D，CLIP score competitive，1 分钟出结果。IM-3D 在 object 上略好但要 3 分钟，而且 IM-3D 不支持 scene。CAT3D 同时支持 object 和 scene。

## 几个关键 ablation 的 intuition

**多 output view 比单 output view 好**：即使你独立评估每张 output 图，5 张一起 generate 的质量都比逐张 generate 高。因为 joint modeling 强制 model 学 cross-view consistency prior，这个 prior 反哺单图质量。

**Raymap 比 pose vector 好**：pose vector 是 global latent，model 要自己学会"pose 怎么映射到每个 pixel 的 ray"。Raymap 直接 per-pixel 告诉 model ray 方向，把这个 geometric prior hard-code 进 input，model 专注学 appearance-geometry correlation。

**3D attention 比 temporal attention 好**：不同 view 之间不仅时间相关，空间上每个 pixel 都要 cross-view reasoning。纯 temporal attention 只在 image level 交流信息，不够。

**Pretrained init 至关重要**：2D web image 数据量远超 3D 数据，pretrained init 把这个 2D prior 带进来大幅提升 OOD generalization。

## 局限

同一个 scene 的 intrinsics 要大致一致，不同 camera 拍的混合不行。Base text-to-image model 的表达能力是天花板。8 张一组的限制意味着生成的几百张图内部未必完美 consistent。Camera trajectory 要手动设计，大型开放场景不好搞。

## 最核心的 take-away

Sparse-view 3D creation 的瓶颈不在 representation，不在 reconstruction algorithm，不在什么花式 regularization。瓶颈就是 observation 数量不够。你把 observation 补齐到 dense capture 水平，标准 reconstruction pipeline 就能 work。

这个 insight 很可能指导未来一堆 3D 工作——不用在 reconstruction 端绞尽脑汁，把 prior 用在 generate observation 上就完事了。

---

# CAT3D: Create Anything in 3D — 完整技术解析

## 1. Core Intuition: 把 ill-posed generation 转化为 over-constrained reconstruction

3D创建任务的真正瓶颈在于**observations数量不足**。NeRF、Instant-NGP、Gaussian Splatting 这些 photogrammetry 方法在 dense capture (几百到上千张照片) 下能产生极高质量 3D 内容,而 sparse input (单图、少视角、文本) 下整个 reconstruction problem 就 ill-posed,导致 geometry 和 appearance 不准。

CAT3D 的核心 insight 极其简单粗暴:**与其为每种 sparse-input 场景设计专门方法 (DreamFusion 的 SDS, PixelNeRF 的 feed-forward, regularization tricks),不如让一个 multi-view diffusion model 帮你 "拍出" 几百张虚拟照片,然后丢给标准 3D reconstruction pipeline**。

这把"3D creation"问题坍缩到"3D reconstruction"问题。Generation prior 和 reconstruction 完全解耦,这是为什么 CAT3D 比 ReconFusion、ZeroNVS 快一个数量级 (1 分钟 vs 1 小时)、方法更简单、质量更高。

项目主页: <https://cat3d.github.io/>

---

## 2. Pipeline 拆解:两阶段架构

```
Input (1~many views + poses)
        ↓
[Stage 1] Multi-View Diffusion Model
        ↓ 生成 80~960 个 novel views (含 pose)
[Stage 2] Robust 3D Reconstruction (Zip-NeRF + LPIPS + distance weighting)
        ↓
NeRF (interactive rendering)
```

Stage 1 用 generative prior 填补 unseen regions;Stage 2 用 robust reconstruction 处理 generated views 之间的不一致性。

---

## 3. Multi-View Diffusion Model (Section 3.1) — 技术细节

### 3.1 学习的目标分布

模型学的是 conditional joint distribution:

$$p\big(\mathbf{I}^{\mathrm{tgt}} \mid \mathbf{I}^{\mathrm{cond}}, \mathbf{p}^{\mathrm{cond}}, \mathbf{p}^{\mathrm{tgt}}\big) \tag{1}$$

变量说明:
- $\mathbf{I}^{\mathrm{cond}} = \{I_1^{\mathrm{cond}}, \dots, I_M^{\mathrm{cond}}\}$: M 张 conditional images (observed views)
- $\mathbf{p}^{\mathrm{cond}}$: 这 M 张图的 camera parameters (intrinsics + extrinsics)
- $\mathbf{I}^{\mathrm{tgt}} = \{I_1^{\mathrm{tgt}}, \dots, I_N^{\mathrm{tgt}}\}$: N 张要生成的 target images
- $\mathbf{p}^{\mathrm{tgt}}$: N 张 target 的指定 camera poses (用户指定想从哪些角度看到场景)

注意 $\mathbf{p}^{\mathrm{tgt}}$ 是显式输入,这就是为什么 model 能做 "任意指定视角" 的 synthesis。这一点比 SVD 这类只支持 smooth orbital trajectory 的 video model 强很多。

### 3.2 Architecture (Figure 7)

Backbone 是一个 text-to-image latent diffusion model (类 Stable Diffusion 架构,Stable Diffusion 论文: <https://arxiv.org/abs/2112.10752>)。

具体 inflation 步骤:

1. **VAE encoding**: 每张 512×512×3 image → 64×64×8 latent。参考 Kingma VAE: <https://arxiv.org/abs/1312.6114>
2. **Camera raymap concatenation**: 每张图的 latent 在 channel 维 concat 一个 raymap (与 latent 同 H,W)。Raymap 编码每个 spatial location 的 ray origin (3 维) + ray direction (3 维),共 6 维。Rays 相对于第一张 conditional image 的 camera 计算,保证 SE(3) invariance。这个 design 来自 SRT (Scene Representation Transformer, <https://arxiv.org/abs/2111.13152>) 和 Novel View Synthesis with Diffusion Models (NVS-Diff, <https://arxiv.org/abs/2210.04628>)。
3. **Binary mask concatenation**: 一个 binary mask (conditional vs target) 也 concat 到 channel 维,告诉 model 哪张是 clean 哪张要 denoise。
4. **3D self-attention inflation**: 原始 2D U-Net 中每个 2D residual block 后面的 2D spatial self-attention 被膨胀为 3D self-attention (2D in space + 1D across images)。思路来自 MVDream (<https://arxiv.org/abs/2308.01829>)。Inflate 时直接继承 2D 预训练权重,新增参数极少。
5. **Text embedding 移除**: 因为 CAT3D 不做 text condition,所以把 cross-attention to text 的那部分丢掉。
6. **3D attention 只在 resolution ≤ 32×32**: 64×64 上做 3D attention 序列长度 64×64×8 = 32k,FlashAttention (<https://arxiv.org/abs/2205.14135>) 都吃力,而增益 marginal (Table 3 显示 32→64 PSNR 14.63→14.64)。所以只在 32×32 及更小分辨率用 3D attention,更大分辨率退化为普通 2D spatial attention。

模型总参数量 850M,显著小于 IM-3D 的 4.3B (<https://arxiv.org/abs/2403.07991>) 和 SV3D 的 1.5B (<https://arxiv.org/abs/2403.12002>)。

### 3.3 Noise Schedule Shift (关键 trick)

从 pretrained 2D image diffusion model 改造成 multi-view diffusion model 时,数据维度从 2D 变成更高维度。Simple Diffusion (<https://arxiv.org/abs/2301.11093>) 指出高分辨率扩散需要 shift log SNR。

CAT3D 把 log signal-to-noise ratio shift by $\log(N)$,$N$ 是 target images 数量。直觉是: 生成 N 张图比生成 1 张图信息量更大,需要更高 noise level 才能让 model 在 training 时充分 explore joint distribution。

### 3.4 训练设置

- **联合训练 8 frames**: $N + M = 8$,随机选 $N = 1$ (1 cond + 7 tgt) 或 $N = 3$ (3 cond + 5 tgt)。这样一个 model 同时 handle 单图和少视角两个 setting。
- **Conditional latents clean, target latents noisy**: diffusion loss 只在 target images 上算。
- **Classifier-Free Guidance (CFG)**: 训练时以 0.1 概率 drop conditional images + poses (<https://arxiv.org/abs/2207.12598>)。Sampling 时 CFG weight = 3。
- **DDIM 50 steps** (<https://arxiv.org/abs/2010.02502>)
- **1M iterations** with 1+7 setting,**0.4M iterations** joint mixture,batch size 128,lr $5 \times 10^{-5}$

### 3.5 训练数据

四个带 camera pose annotation 的数据集:
- Objaverse (<https://arxiv.org/abs/2212.01860>): synthetic 3D objects
- CO3D (<https://arxiv.org/abs/2109.00505>): real object-centric videos
- RealEstate10K (<https://arxiv.org/abs/1805.09817>): real estate walkthroughs
- MVImgNet (<https://arxiv.org/abs/2303.06531>): multi-view object scans

按 [7] ReconFusion (<https://reconfusion.github.io/>) 的做法等概率采样。

---

## 4. Generating Novel Views (Section 3.2) — 从 8 views 扩展到 960 views

训练时 model 只见过 8 个 views 一组,但 inference 时要生成几百张图。怎么解决?

### 4.1 Grouped Sampling

把目标 viewpoints 按 camera position 聚类成小的 groups (5 views/group),每组独立 generate。这本质上是 **block-wise autoregressive**。距离近的 views 放一组是因为它们之间的 multi-view consistency 最强,8-view model 一次性 handle 5 个 near views 比较稳。

### 4.2 Anchor-based Autoregressive Sampling (单图场景)

单图场景特别棘手,因为没有 long-range consistency 的 anchor。流程:

1. 用 k-means++ 贪心 initialization (<https://theory.stanford.edu/~sergei/papers/kmeansICML06.pdf>) 选 7 个 anchor views。贪心准则: 每次选离已选 cameras 距离最远的那个。
2. 给定 1 个 input view,生成 7 个 anchor views (一次 model forward)。
3. 给定 1 input + 7 anchors (共 8 views),从其中选 3 nearest 作为 conditional,grouped sample 出剩余的所有 views。

### 4.3 Camera Trajectories

四类 paths (Figure 8):
1. **Orbital paths**: 不同 scale 和 height 围绕 center scene 的轨道
2. **Forward-facing circle**: 不同 scale 和 offset 的 forward-facing 圆
3. **Spline paths**: 拟合 input views 的 spline + 不同 offset
4. **Spiral cylindrical**: 沿圆柱面 spiral,move into/out of scene

数据集对应:
- RealEstate10K: spline + xz-plane offset, 800 views
- LLFF/DTU: forward-facing circle + z-axis offset, 960/480 views
- CO3D: spline + scale factors, 640 views
- Mip-NeRF 360: elliptical + z-axis offset, 720 views

---

## 5. Robust 3D Reconstruction (Section 3.3) — 处理 generated views 的 inconsistency

Generated views 不完美 3D consistent。即使 SOTA video diffusion model 也有这个问题 (<https://arxiv.org/abs/2311.17138> "Shadows Don't Lie and Lines Can't Bend")。所以 standard NeRF 训练 pipeline 需要改造。

### 5.1 Base: Zip-NeRF

Zip-NeRF (<https://zipnerf.github.io/>) loss:
$$\mathcal{L} = \mathcal{L}_{\mathrm{photo}} + \lambda_d \mathcal{L}_{\mathrm{distort}} + \lambda_i \mathcal{L}_{\mathrm{interlevel}} + \lambda_w \mathcal{L}_{\mathrm{weight}}$$

- $\mathcal{L}_{\mathrm{photo}}$: photometric reconstruction loss (RGB MSE)
- $\mathcal{L}_{\mathrm{distort}}$: distortion loss,鼓励 ray sampling 分布集中在 surface 附近
- $\mathcal{L}_{\mathrm{interlevel}}$: interlevel loss,保证 multi-resolution hash grid 不同 level 一致
- $\mathcal{L}_{\mathrm{weight}}$: normalized L2 weight regularizer

### 5.2 加 LPIPS Perceptual Loss

$$\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{Zip-NeRF}} + \lambda_p \mathcal{L}_{\mathrm{LPIPS}}$$

LPIPS (<https://arxiv.org/abs/1801.03924>) 用 AlexNet/VGG feature 计算 perceptual distance,忽略 low-level high-frequency detail 不一致 (这正是 generated views 之间的不一致点),保留 high-level semantic consistency。

$\lambda_p = 0.25$ for single image-to-3D + RealEstate10K/LLFF/DTU,$\lambda_p = 1.0$ for CO3D/Mip-NeRF 360 (更难数据集需要更强 perceptual supervision)。

### 5.3 Distance-based View Weighting (关键 trick)

Generated views 离 observed views 越近,uncertainty 越小,consistency 越好,越应该被 trust。距离越远,越发散。

设计 Gaussian kernel weighting:

$$w \propto \exp\big(-b \cdot s^2\big)$$

- $s$: 该 generated view 到最近的 captured view 的距离
- $b$: scaling factor,**linearly annealed from 0 to 15**

训练初期 $b = 0$,所有 views 权重相同,等于标准 NeRF;训练后期 $b = 15$,近 observed views 权重高,远 views 权重低。这给模型先用 generated views 拼出大致 geometry,后期再让 observed views 主导细节。

Intuition: 训练初期需要 generated views 才能填补 unseen region,平等对待;训练后期 geometry 已经稳定,observed views 的 detail 应该 dominate,generated views 退到 "辅助 regularization" 的角色。

### 5.4 Hyperparameters

- View-dependence network: width 32, depth 1 (smaller than Zip-NeRF default,防 overfitting)
- Training iterations: 1000 (极少,加速)
- Few-view: 128×128 patches, batch size 1M rays, 4 minutes
- Single-image: 32×32 patches, batch size 65k, 55 seconds
- lr log-decay from 0.04 to $10^{-3}$
- 16 A100 GPUs

整个 CAT3D pipeline (生成 + reconstruction) 大约 1 分钟。

---

## 6. Experiments — 量化结果

### 6.1 Few-View 3D Reconstruction (Table 1)

数据集: RealEstate10K, LLFF (<https://arxiv.org/abs/1905.00817>), DTU (<https://roboimagedata.compute.dtu.dk/?idf=11>), CO3D, Mip-NeRF 360 (<https://jonbarron.info/mipnerf360/>)

3-view setting 几个数据集 PSNR:

| Dataset | Zip-NeRF (no prior) | ZeroNVS | ReconFusion | CAT3D |
|---|---|---|---|---|
| RealEstate10K | 20.77 | 19.11 | 25.84 | **26.78** |
| LLFF | 17.23 | 15.91 | 21.34 | **21.58** |
| DTU | 9.18 | 16.71 | 20.74 | **22.02** |
| CO3D | 14.34 | 17.13 | 19.59 | **20.57** |
| Mip-NeRF 360 | 12.77 | 14.44 | 15.50 | **16.62** |

CAT3D 在几乎所有 setting 都 SOTA,且推理时间从 1 小时降到几分钟。

### 6.2 Single Image to 3D (Table 2)

CLIP image score 评估 (semantic fidelity):

| Model | Time (min) | CLIP (Image) |
|---|---|---|
| ImageDream | 120 | 83.77 ± 5.2 |
| One2345++ | 0.75 | 83.78 ± 6.4 |
| IM-3D (NeRF) | 40 | 87.37 ± 5.4 |
| IM-3D | 3 | 91.40 ± 5.5 |
| CAT3D | 1 | 88.54 ± 8.6 |

CAT3D 在 1 分钟内达到 competitive CLIP score,IM-3D 用更长时间在 object-centric 数据上更好,但 IM-3D 不能做 scenes,不能做 object-in-context。

---

## 7. Ablations (Table 3) — Build Your Intuition

这是 paper 最有信息量的部分,逐项分析:

### 7.1 Target Views 数量

| Setting | In-domain PSNR | OOD PSNR | NeRF PSNR |
|---|---|---|---|
| 3 cond + 1 tgt (ReconFusion-like) | 18.85 | 14.12 | 16.17 |
| 3 cond + 5 tgt | 21.66 | 14.63 | 16.29 |

**Insight**: 把单 output 改成 5 outputs joint model,即使你独立评估每一张 output,质量都更好。原因: joint modeling 强制 model 学到 multi-view consistency 的 prior,这个 prior 反过来让单张图也更"3D 合理"。

### 7.2 Camera Conditioning

| Setting | OOD PSNR |
|---|---|
| Low-dim vector (8-dim, cross-attention) | 14.19 |
| Raymap (channel concat) | 14.63 |

**Insight**: Low-dim pose embedding 在 in-domain 表现 OK,但 OOD generalize 差。Raymap 是 dense per-pixel pose encoding,model 容易 generalize 到没见过的 camera configurations,因为 ray map 本身直接告诉你 "这个像素对应 3D 空间哪个方向"。

### 7.3 Attention Layer 类型

| Setting | OOD PSNR |
|---|---|
| Temporal attention only (1D) | 13.41 |
| 3D attention until 16×16 | 14.23 |
| 3D attention until 32×32 | 14.63 |
| 3D attention until 64×64 (full) | 14.64 |

**Insight**: Temporal-only attention 不够,因为不同 views 之间不仅时间维度相关,空间上每个像素都需要 cross-view reasoning (比如 view A 的某个物体在 view B 对应像素)。3D attention 让每个 spatial location 都能 attend 到其他 views 的对应 spatial location。64×64 上 3D attention 增益 marginal (0.01 PSNR) 但贵很多,所以只在 32×32 及以下用。

### 7.4 Pretrained Initialization

| Setting | OOD PSNR |
|---|---|
| From scratch | 13.88 |
| From pretrained (1M iter) | 14.63 |
| From pretrained (1M) + joint (0.4M) | 15.19 |

**Insight**: Web-scale 2D image prior 通过 pretrained init 大幅改善 OOD generalization。这正是为什么 2D prior 比 3D native prior 更强的核心原因 — 3D 数据集再大也比不上 web image 数据规模。Joint training (1+7 和 3+5 混合) 进一步提升。

### 7.5 LPIPS Loss & View Count (Figure 6)

- 720 views (9 orbits) vs 80 views (1 orbit): 720 让 central object geometry 更好,但 background 可能更糊 (inconsistency 累积)
- No LPIPS: texture 和 geometry 都明显劣化

---

## 8. Limitations

1. **Intrinsics 一致性假设**: 训练数据每 scene 内 intrinsics 大致一致,test 时跨 camera intrinsics 不行
2. **Base model 表达能力限制**: scene 内容如果 OOD for base text-to-image model,CAT3D 也差
3. **8 views 限制**: model 只见 8 views 一组,generated 大集合内部未必 fully consistent
4. **Manual camera trajectories**: 需要 manually 设计 path,大型开放环境难设计

---

## 9. 与 Karpathy 直觉相关的几个 deep insights

### 9.1 Generation Prior 和 Reconstruction 解耦的力量

DreamFusion (SDS, <https://dreamfusion3d.github.io/>) 把 generation prior 绑在 reconstruction 里,distillation 极慢且需要 gradient through 3D representation。CAT3D 把 prior 用一次 (generate views),然后丢给 standard reconstruction。这种 "amortize prior then reconstruct" 思路本质是把 amortized inference (model forward) 替代 iterative inference (gradient descent with prior as regularizer)。这也解释了为什么快一个数量级。

### 9.2 Multi-view Joint Modeling 即 "Implicit 3D Prior"

3D self-attention 让 model 在 latent space 学到 view 间 correlations。这相当于学了一个 implicit 3D prior,但 representation 是 2D images 而不是 explicit NeRF/Gaussian。Reconstruction stage 才把这个 implicit prior 变成 explicit 3D。这种 implicit→explicit 的两阶段做法有 generic value。

### 9.3 为什么 Raymap > Pose Vector

Pose vector 是 global latent,每个 pixel 看到的都是同一个 pose 信息,需要 model 自己学会 "怎么从 pose 推导每个 pixel 的 ray"。Raymap 直接给 per-pixel ray,把这部分 prior 直接 hard-coded 进 input,让 model 集中精力学 appearance 和 geometry correlation。这种 "explicit geometric prior as input feature" 的做法在 NVS 历史上反复出现 (NeRF 的 positional encoding 也是类似思路)。

### 9.4 Noise Schedule Shift 的深层含义

$\log(N)$ shift 对应 "增加数据维度需要更高 noise level"。这跟 video diffusion 也用类似 trick 一样。Karpathy 你可能会想到: 这是 diffusion model 在不同 intrinsic dimensionality 下 optimal forward process 不同的体现。$N$ 张图联合分布的 entropy 大约是单图的 $\log N$ 倍,noise level 也需要相应提高。

### 9.5 Distance Weighting 是 "Annealed Trust Schedule"

训练初期所有 views 平等 (用 generated views 填洞),后期信任度按到 observed view 距离衰减。这相当于 Bayesian 框架下,observed views 的 likelihood 是确定的,generated views 的 likelihood 不确定性随距离增加,$w \propto \exp(-b s^2)$ 是 Gaussian uncertainty model 的精确体现。$b$ 从 0 退火到 15 对应 "uncertainty 逐渐显式化"。

---

## 10. 与相关工作的对比脉络

- **vs DreamFusion/Magic3D/Fantasia3D**: 不用 SDS,iterative distillation 换成单次 generate + standard recon
- **vs ReconFusion**: ReconFusion 用 PixelNeRF + CLIP embedding conditioning,单 output view,需要 3D distillation fine-tune。CAT3D 用纯 attention conditioning,multi-view joint output,不需要 distillation
- **vs MVDream/ImageDream**: MVDream 4 views,做 text-to-3D object。CAT3D 8 views,支持 1~many input,做 scenes + objects
- **vs IM-3D**: IM-3D 也是 "iterative multi-view diffusion + recon" 思路,4.3B params,只做 object。CAT3D 850M params,做 scene + object + few-view recon
- **vs SV3D**: SVD-based,1.5B params,orbital trajectory only。CAT3D 支持 arbitrary trajectories
- **vs LRM/GS-LRM/Splatter Image**: Feed-forward methods 几秒出结果,但 quality 差。CAT3D 1 分钟,quality 高

---

## 11. 重要 References

- CAT3D project: <https://cat3d.github.io/>
- NeRF: <https://www.matthewtancik.com/nerf>
- Instant-NGP: <https://nvlabs.github.io/instant-ngp/>
- Gaussian Splatting: <https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/>
- ReconFusion: <https://reconfusion.github.io/>
- DreamFusion: <https://dreamfusion3d.github.io/>
- MVDream: <https://mvdream.github.io/>
- ImageDream: <https://imagen-3d.github.io/>
- ZeroNVS: <https://kylesargent.github.io/zeronvs/>
- Zero123: <https://zero123.cs.columbia.edu/>
- Zip-NeRF: <https://zipnerf.github.io/>
- Mip-NeRF 360: <https://jonbarron.info/mipnerf360/>
- Stable Video Diffusion: <https://stability.ai/news/stable-video-diffusion-open-ai-video-model>
- Sora (Video generation models as world simulators): <https://openai.com/research/video-generation-models-as-world-simulators>
- IM-3D: <https://arxiv.org/abs/2403.07991>
- SV3D: <https://arxiv.org/abs/2403.12002>
- Objaverse: <https://objaverse.allenai.org/>
- CO3D: <https://facebookresearch.github.io/co3d/>
- RealEstate10K: <https://google.github.io/realestate10k/>
- MVImgNet: <http://gvlab-pku.github.io/MVImgNet/>
- LPIPS: <https://richzhang.github.io/PerceptualSimilarity/>
- FlashAttention: <https://github.com/Dao-AILab/flash-attention>
- LDM (Stable Diffusion): <https://ommer-lab.com/research/latent-diffusion-models/>
- Simple Diffusion (noise schedule shift): <https://arxiv.org/abs/2301.11093>
- Classifier-Free Guidance: <https://arxiv.org/abs/2207.12598>
- DDIM: <https://arxiv.org/abs/2010.02502>
- LRM: <https://yiconghong.me/LRM/>
- Splatter Image: <https://szymanowiczs.github.io/splatter-image>
- DreamCraft3D: <https://mrtorrent24.github.io/DreamCraft3D/>
- RealmDreamer: <https://realmdreamer.github.io/>
- MotionCtrl: <https://wzx0800.github.io/projects/AniClipart-project-page/> 与 <https://arxiv.org/abs/2312.03641>
- AnimateDiff: <https://github.com/guoyww/AnimateDiff>
- PixelNeRF: <https://alexyu.net/pixelnerf/>
- SRT: <https://arxiv.org/abs/2111.13152>
- k-means++: <https://theory.stanford.edu/~sergei/papers/kmeansICML06.pdf>
- Score Distillation Sampling 引入 (DreamFusion): <https://dreamfusion3d.github.io/>
- ProlificDreamer (VSD): <https://ml.cs.tsinghua.edu.cn/~prolificdreamer/>
- Att3D (Amortized text-to-3D): <https://att3d.github.io/>

---

## 12. 总结式 Intuition

CAT3D 把 "3D creation from sparse input" 重新表述为 "novel view generation from sparse input"。这个 reformulation 的力量在于:

1. **降维打击**: 把 ill-posed 3D 问题 (有无数解) 压成 well-posed reconstruction (单一解)
2. **Amortized prior**: 把 generative prior 用一次 forward pass 而非 iterative gradient descent
3. **模块化**: generation model 和 reconstruction model 各自可以独立改进 (换个更好的 base model 立即受益)
4. **Architecture minimalism**: 在 2D image diffusion model 上 inflate 出 3D attention,加 raymap,加 mask,加 noise shift — 改动极少
5. **Robustness through loss design**: LPIPS + distance weighting 把 generated views 的 inconsistency 吸收掉

核心 take-away:**sparse-view 3D creation 的瓶颈不在 representation 或 reconstruction algorithm,而在 observations 数量**。一旦你能用 generative model 把 observations 数量补齐到 dense capture 水平,剩下交给标准 reconstruction pipeline 就够了。这个 insight 应该会被未来 3D creation 工作反复借用。
