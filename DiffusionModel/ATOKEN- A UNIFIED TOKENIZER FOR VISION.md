---
source_pdf: ATOKEN- A UNIFIED TOKENIZER FOR VISION.pdf
paper_sha256: 5aa48c6a090845e29c7a0fc73fcdb0554a8a678320538c7313d25eac1623325b
processed_at: '2026-07-18T10:31:40-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# ATOKEN 深度讲解：Apple 的 Unified Visual Tokenizer

Andrej, 这篇 paper 我读完之后直觉上觉得它在做一件 vision community 早就该做的事 —— 把 image / video / 3D 三种 modality 塞进同一个 latent space，并且同时支持 reconstruction 和 understanding。下面我把核心 idea、架构 trick、训练 recipe、实验数据全部展开讲，并尽量 build 你的 intuition。

---

## 1. 为什么这件事难：visual tokenization 的"碎片化"困境

LLM 之所以能一个 model 干所有事，关键在于 BPE tokenizer 把 code、英文、中文、表格全部映射到统一 token space（[Sennrich et al., 2015, BPE](https://arxiv.org/abs/1508.07909)）。Vision 这边一直没做到这件事，paper 在 Section 2 列了三层 fragmentation：

1. **Task specialization**：reconstruction tokenizer（SD-VAE、VQGAN、Cosmos、GigaTok）只管像素细节；understanding encoder（CLIP、SigLIP2、VideoPrism）只管语义。两边互不兼容。
2. **Modality fragmentation**：image tokenizer 处理不了 video 的时间维度；video tokenizer（Hunyuan、Wan、TAE）处理不了 3D；3D tokenizer（Trellis-SLAT）又用不了 image/video 大规模预训练数据。
3. **Architectural trade-offs**：convolutional tokenizer 在 scaling 上收益递减（[GigaTok, Xiong et al., 2025](https://arxiv.org/abs/2504.08736)）；pure transformer tokenizer（[ViTok, Hansen-Estruch et al., 2025](https://arxiv.org/abs/2501.09755)）scaling 好但 GAN 训练不稳定。

ATOKEN 的目标就是同时解决这三件事，Table 1 给了一张很清晰的对比表 —— 它是唯一一个在 reconstruction/understanding × image/video/3D × continuous/discrete × GAN-free × native resolution × temporal compression 所有维度都打勾的方法。

---

## 2. 核心洞察：Sparse 4D Latent Space

这是整篇 paper 最 elegant 的 idea。作者观察到 image、video、3D 其实都可以放进同一个 4D 坐标系 $(t, x, y, z)$：

- **Image**：占据 $(x, y)$ 平面，$t = z = 0$
- **Video**：在 $(t, x, y)$ 上延伸，$z = 0$
- **3D asset**：在 $(x, y, z)$ 空间作为 surface voxels，$t = 0$

形式化定义在 Eq (1)：

$$z = \{(z_i, p_i)\}_{i=1}^{L}, \quad z_i \in \mathbb{R}^C, \quad p_i \in \{0, 1, \dots, N-1\}^4$$

变量含义：
- $z_i$：第 $i$ 个 active location 的 latent feature，channel 数为 $C$（Stage 1 是 32，Stage 2/3 升到 48）
- $p_i = [t, x, y, z]$：4D 坐标
- $N$：每个轴的 resolution（3D 用 $64^3$ grid）
- $L$：active location 数量（sparse，不是 dense grid）

**Intuition**：这相当于在 4D 空间里放一块"显示器"来表示 image/video（grid index 当坐标用），而 3D asset 的坐标是真实物理 occupancy。作者特别说明这种 dual interpretation 不会破坏 generalization，因为 4D RoPE 学的是 relative position，不依赖 absolute coordinate 的语义。

这种 sparse 表示有个巨大好处：**encoder 不需要为不同 modality 改架构**，只需要让 active location 集合不同。这也直接 enable 了 native resolution（不像 conv 必须固定 spatial size）和 temporal compression（video 的 active location 沿 $t$ 轴排布）。

---

## 3. Architecture：Pure Transformer + 4D RoPE

### 3.1 Space-Time Patch Embedding

输入 $x \in \mathbb{R}^{T \times H \times W \times 3}$ 切成 $t \times p \times p$ 的 non-overlapping space-time patch（默认 $t=4, p=16$）。Image ($T=1$) 通过 temporal zero-padding 变成 4-frame patch，保证和 video 维度一致。

3D 的处理是 paper 里比较 hacky 的部分（Figure 3）：从球面采样相机渲染 multi-view image → 对每个 view 做 space-time patchify → 把 $64^3$ voxel grid 中的每个 voxel back-project 到相关 view，gather + average patch feature。这里和 [Trellis-SLAT, Xiang et al., 2024](https://arxiv.org/abs/2412.01506) 的区别是：Trellis 用 DINOv2 feature，ATOKEN 直接用 raw RGB patch（这样能 end-to-end 训练，不依赖外部 feature）。

### 3.2 Encoder：从 SigLIP2 扩展到 4D

Encoder 初始化自 [SigLIP2-SO400M-patch16-naflex, Tschannen et al., 2025](https://arxiv.org/abs/2502.14786)，做两个改动：

1. **Patch embedding 扩展**：从 2D patch 变成 $t \times p \times p$ space-time block，temporal weight zero-init（保留原 image feature）。
2. **Position embedding**：SigLIP2 原本用 learnable 2D absolute PE，ATOKEN 保留它，并在每个 attention layer 加 4D RoPE（[Lu et al., 4M, 2024](https://arxiv.org/abs/2312.06647)）。

4D RoPE 的好处：relative position 在 $(t, x, y, z)$ 四个轴上独立旋转，对任意 resolution / temporal length 都能 generalize，这正是 NaFlex（NaViT-style）训练需要的。

### 3.3 Decoder：Two Task-Specific Heads

Decoder 从 scratch 训练（不 pretrain），和 encoder 同样架构（27 层 transformer，$d=1152$，16 heads）。

- **Image/Video head** $\mathcal{D}_P$（Eq 2）：直接 decode 到 pixel space，image 当 $T=1$ video 处理，discard temporal padding（参考 [TAE, Polyak et al., 2024](https://arxiv.org/abs/2410.13720)）。
- **3D head** $\mathcal{D}_{GS}$（Eq 3）：decode 到 Gaussian splatting 参数，每个 voxel 生成 $K$ 个 Gaussian：

$$\mathcal{D}_{GS}: \{(z_i, p_i)\}_{i=1}^L \to \{\{(o_i^k, c_i^k, s_i^k, \alpha_i^k, r_i^k)\}_{k=1}^K\}_{i=1}^L$$

变量含义：
- $o_i^k$：position offset（被 $\tanh$ 约束在 voxel 附近，$x_i^k = p_i + \tanh(o_i^k)$）
- $c_i^k$：color
- $s_i^k$：scale
- $\alpha_i^k$：opacity
- $r_i^k$：rotation
- $K$：每个 voxel 的 Gaussian 数量

这种 local constraint 是从 Trellis 抄来的，保证 Gaussian 不会乱跑。

### 3.4 Dual Projection：Reconstruction + Understanding 共享 encoder

这是统一 task 的关键设计（Figure 2）：

- **Reconstruction 路径**：$z^r = W_r(z)$，低维 projection + KL regularization（VAE-style），可选 FSQ 离散化 $\tilde{z}^r = \text{FSQ}(z^r)$
- **Understanding 路径**：attention pooling 得到 global representation $\bar{z}$，再 $z^s = W_s(\bar{z})$ 和 text embedding 对齐

两个 projection 共享同一个 encoder feature $z$，所以没有架构 duplication —— 这是和 VILA-U、UniTok 思路类似但更彻底的设计。

---

## 4. Adversarial-Free Training：Gram Loss 的洞察

这是 paper 里我个人觉得最漂亮的一段分析。

### 4.1 GAN 在 transformer tokenizer 上失败

Figure 4(a) 显示：在 pure transformer 架构下，discriminator 很快 dominate generator，logit 发散，rFID 退化。这解释了为什么 [ViTok](https://arxiv.org/abs/2501.09755) 这类 transformer tokenizer 一直打不过 conv tokenizer —— GAN 训练 dynamic 在 transformer 上更脆弱。

### 4.2 rFID 分解：covariance 占 86.6%

作者把 rFID 误差分解成 mean 和 covariance 两部分（Figure 4(b)）：

- **Covariance component**：≈ 86.6%（捕捉 texture、style 等 second-order statistics）
- **Mean component**：≈ 13.4%

这个观察非常关键 —— 它说明 reconstruction error 主要不是"像素均值偏了"，而是"feature 分布的二阶统计量没对齐"。

### 4.3 Gram Matrix Loss 直接打 covariance

[Gram matrix loss, Gatys et al., 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html) 定义：

$$\mathcal{L}_{Gram}(x, \hat{x}) = \sum_l \|G(\Phi_l(x)) - G(\Phi_l(\hat{x}))\|_F^2$$

变量含义：
- $\Phi_l$：pretrained 网络（这里是 VGG）的第 $l$ 层
- $G(F) = FF^\top$：feature map $F$ 的 Gram matrix，捕捉 channel 之间的 correlation
- $\|\cdot\|_F$：Frobenius norm
- 求和 over 多层

**Intuition**：Gram matrix 就是 feature 的 uncentered covariance，直接优化它就等于直接打 rFID 的 covariance 项。Figure 4(c) 显示 Gram loss 训练全程稳定，rFID 持续下降。

### 4.4 完整 Reconstruction Loss（Eq 6）

$$\mathcal{L}_{rec}^I = \lambda_1 \mathcal{L}_1 + \lambda_{LPIPS} \mathcal{L}_{LPIPS} + \lambda_{GRAM} \mathcal{L}_{GRAM} + \lambda_{CLIP} \mathcal{L}_{CLIP}$$

- $\mathcal{L}_1 = \|x - \hat{x}\|_1$：pixel-level supervision
- $\mathcal{L}_{LPIPS}$（[Zhang et al., 2018](https://arxiv.org/abs/1801.03924)）：perceptual similarity
- $\mathcal{L}_{GRAM}$：texture/style（核心创新）
- $\mathcal{L}_{CLIP}$：semantic consistency

权重：$\lambda_1 = 1.0, \lambda_{LPIPS} = 10.0, \lambda_{GRAM} = 10^3, \lambda_{CLIP} = 1.0$。Gram loss 权重给到 $10^3$ 是因为它的 magnitude 本身很小。

Video 和 3D 只用 $\mathcal{L}_1$（节省 compute），靠 cross-modal transfer 从 image 那边把 detail 学过来。这个简化是合理的 —— video/3D 数据量小，加 perceptual loss 收益有限。

### 4.5 Semantic Loss（Eq 7）

Image 用 knowledge distillation from frozen SigLIP2：

$$\mathcal{L}_{sem}^I = \text{KL}\left(\text{softmax}(\tau^{-1} s^{teacher}) \| \text{softmax}(\tau^{-1} s^{student})\right)$$

变量含义：
- $s^{teacher}$：frozen SigLIP2 给出的 vision-text similarity score
- $s^{student}$：ATOKEN 给出的 score（和同一个 frozen text encoder 配对）
- $\tau = 2.0$：temperature

Video 和 3D 用 [SigLIP 的 sigmoid loss, Zhai et al., 2023](https://arxiv.org/abs/2303.15387)，因为 batch size 小，softmax 对比学习不稳定。

总 loss（Eq 4）：$\mathcal{L} = \lambda_{rec} \mathcal{L}_{rec} + \lambda_{sem} \mathcal{L}_{sem} + \lambda_{KL} \mathcal{L}_{KL}$，权重 $0.2 : 1.0 : 10^{-8}$。注意 $\lambda_{sem} = 1.0$ 远大于 $\lambda_{rec}$，因为 understanding 是从 pretrained SigLIP2 来的强 prior，不能让它 collapse。

---

## 5. Progressive Curriculum：四阶段训练

这是工程上很关键的设计（Figure 5, Table 2）。

| Stage | Modalities | Image Res | Video Res | 3D Size | Steps |
|---|---|---|---|---|---|
| 1: Image Foundation | I only | 64–512 | — | — | 200k |
| 2: Video Dynamics | I + V | 64–1024 | 64–512 | — | 200k |
| 3: 3D Geometry | I + V + 3D | 64–2048 | 64–1024 | $64^3$ | 50k |
| 4: Discrete Tokenization | + FSQ | 同上 | 同上 | 同上 | 100k |

几个细节：

1. **Stage 1** latent dim = 32，**Stage 2** 升到 48（accommodate motion complexity，参考 [Seawead, 2025](https://arxiv.org/abs/2504.08685)）
2. **Round-robin sampling**：用 gradient accumulation 平衡 image-text distillation 和其他 task，保证 semantic alignment 在所有 stage 都不丢
3. **Stage 2 的 temporal tiling**：16–32 frames → 4–8 latent frames，stride 1–3 保 consistency / 4–12 保 diversity；understanding 用 1 FPS up to 64 frames
4. **KV-caching across temporal tiles**（Figure 6）：消除 redundant computation，比 overlapping tile 方法高效很多 —— 这是 transformer tokenizer 的天然优势
5. **Stage 4 的 FSQ**（[Mentzer et al., 2023](https://arxiv.org/abs/2309.15505)）：48 维 latent 切成 8 组 × 6 维，每组 quantize 到 4 levels，codebook size 4096

**最 surprising 的发现**：multimodal training 不但没损害 image reconstruction，反而提升它 —— Stage 1 rFID 0.258 → Stage 3 rFID 0.209，提升 19%。作者的解释是 video 的 temporal dynamics 和 3D 的 geometric understanding 给 image reconstruction 提供了 complementary signal。

---

## 6. 实验结果：数据表精读

### 6.1 Image Reconstruction（Table 4）

ATOKEN-So/C 在 ImageNet 256×256 上：
- PSNR 29.72, SSIM 0.848, LPIPS 0.085, **rFID 0.209**
- COCO 上 rFID 2.026

对比 baseline：
- SD-VAE（[Rombach et al., 2022](https://arxiv.org/abs/2112.10752)）：rFID 0.606
- FLUX.1 dev（[Labs et al., 2025](https://arxiv.org/abs/2503.16393)）：rFID 0.176（但 8×8 压缩，ATOKEN 是 16×16）
- Hunyuan（[Kong et al., 2024](https://arxiv.org/abs/2412.03603)）：rFID 0.670
- Wan2.2（[Wan, 2025](https://arxiv.org/abs/2503.20314)）：rFID 0.749

ATOKEN 在 16×16 压缩下 rFID 0.21 是非常强的数字 —— 注意 16×16 比 8×8 难很多，Cosmos-CI16×16 在同样 setting 下 rFID 0.959，VAVAE 用 32 维 latent 才到 0.279。

### 6.2 Image Understanding（Table 5）

ATOKEN-So/C Stage 3 在 ImageNet zero-shot：
- 224 resolution: 82.2%（vs SigLIP2 83.4%）
- 384 resolution: 82.9%
- 512 resolution: 82.9%

只比 SigLIP2 低 1.2%，但 ATOKEN 同时还要做 reconstruction + video + 3D。UniTok 是 78.6%，VILA-U 是 78.0%，ATOKEN 显著领先。

### 6.3 Video Reconstruction（Table 6）

DAVIS 1080p:
- ATOKEN-So/C Stage 3: PSNR 33.11, rFVD 10.76
- Wan2.1: 33.50 / 17.75
- Wan2.2: 33.06 / 12.65
- Hunyuan: 32.33 / 22.94

TokenBench 720p:
- ATOKEN: 36.07 / 3.01
- Wan2.2: 36.39 / 3.19
- Hunyuan: 36.37 / 3.78

ATOKEN 在 16×16 压缩下接近 Wan2.2（也是 16×16），明显超过 8×8 的 Cosmos 和 Hunyuan。Stage 2 → Stage 3 的提升（35.63 → 36.07 PSNR on TokenBench）证明 3D 加入对 video 有正向 transfer。

### 6.4 Video Understanding（Table 7）

MSRVTT R@1: 40.2%（SigLIP2 是 41.9%，VideoPrism 是 52.7%）
MSVD R@1: 53.5%

这里 ATOKEN 明显比 dedicated video encoder 低，作者诚实承认是 video-text pair 数据量不够。但作为 unified tokenizer 这个数字已经 reasonable。

### 6.5 3D（Table 8）

Toys4k:
- ATOKEN-So/C: PSNR 28.28, LPIPS 0.062, Acc 90.9%
- Trellis-SLAT: PSNR 26.97, LPIPS 0.054

ATOKEN 在 PSNR 上超过 specialized Trellis-SLAT，LPIPS 略差。Figure 11 显示 ATOKEN 的 color consistency 更好（因为从 image/video transfer 了 color prior），Trellis-SLAT 有 color shift。

### 6.6 Scaling Analysis（Figure 7）

这是 paper 里最重要的 ablation 之一。Base model（192M）vs So400m（800M）：

- Stage 1: Base rFID 0.323, So rFID 0.258 —— 都还 OK
- Stage 2（加 video）: Base rFID 退化 49% 到 0.483，So 改善 19% 到 0.209

**Intuition**：multimodal tokenization 有 capacity threshold。小 model 会 catastrophic interference，大 model 反而受益于 cross-modal learning。这和 LLM scaling 的观察一致 —— 多任务在足够 capacity 下是 synergistic 而非 competitive。

### 6.7 Representation Structure（Figure 8）

T-SNE 可视化显示：dense feature 有清晰 class clustering，但 project 到 48 维 latent 后 class 边界变模糊。作者提出一个有意思的问题：**显式 semantic clustering 在低维空间里真的必要吗？** VAVAE 强调 semantic alignment，但 ATOKEN 的实验表明大 model 能 leverage 看似 intermixed 的 representation。这个点我觉得值得深挖 —— 可能 semantic information 编在了 2D projection 看不到的高阶结构里。

---

## 7. Downstream Applications

### 7.1 Multimodal LLM（Table 9, 10）

把 ATOKEN-So/C 塞进 [SlowFast-LLaVA-1.5, Xu et al., 2025](https://arxiv.org/abs/2503.18943) 替换 [Oryx-ViT, Liu et al., 2024](https://arxiv.org/abs/2409.12961)，frozen tokenizer 只训 projector + LLM。

Image understanding（7B LLM）：
- ATOKEN: RW-QA 68.8, AI2D 81.2, SQA 92.1, MMMU 48.7, MathV 61.2, OCRBench 74.5, TextVQA 77.7
- Oryx-ViT: 67.5 / 80.4 / 91.1 / 49.0 / 62.5 / 76.4 / 76.4

ATOKEN 在大多数 benchmark 上略胜 Oryx-ViT。这说明 unified tokenizer 在 understanding 端不输 dedicated encoder。

Video understanding（7B LLM）：
- ATOKEN: VideoMME 64.5, PercepTest 70.3, NExT-QA 83.7, LongVideoBench 60.6, MLVU 69.8, LVBench 44.8
- Oryx-ViT: 63.9 / 69.6 / 83.3 / 62.5 / 71.5 / 45.3

ATOKEN 在 general video QA 更强，Oryx-ViT 在 long-form（MLVU）更强 —— 作者归因于 Oryx 专门为 long video 训过 retrieval task。

### 7.2 Image Generation with Continuous Tokens（Table 11）

用 [Lightning-DiT, Yao & Wang, 2025](https://arxiv.org/abs/2501.01423) 框架，ImageNet 256×256 class-conditional：

- ATOKEN-So/C Stage 3: gFID 1.56, IS 260.0, Pre 0.79, Rec 0.63
- VAVAE: gFID 1.35（image-specific 优化）
- REPA（[Yu et al., 2024](https://arxiv.org/abs/2410.06940)）: gFID 1.42

ATOKEN 在 unified setting 下 gFID 1.56 已经接近 specialized 方法。CFG scale 1.65（48 channel）vs 1.5（32 channel），符合 Lightning-DiT 的观察 —— wider latent 需要更强 guidance。

### 7.3 Image Generation with Discrete Tokens（Table 12）

用 [TokenBridge, Wang et al., 2025](https://arxiv.org/abs/2503.16430) 框架：

- ATOKEN-So/D: gFID 2.23, IS 274.5
- TokenBridge: gFID 1.76（但 vocab 只有 8，ATOKEN 是 4096）
- UniTok: gFID 2.51

ATOKEN 的 vocab 4096 比 TokenBridge 的 8 大很多，建模难度高，但仍然 beat UniTok。

### 7.4 Text-to-Video Generation（Table 13）

用 MMDiT backbone，controlled setting（256×256 image, 192×336 video）：

- ATOKEN-So/C Stage 3: CLIP 32.50, Pick 21.74, GenEval 64.61%, VBench Total 78.46%
- Wan2.1: 32.45 / 21.62 / 65.57% / 78.60%
- Hunyuan: 32.49 / 21.66 / 66.11% / 78.02%

ATOKEN 和 Wan2.1 几乎打平，超过 Hunyuan 和 Cosmos。这是 unified tokenizer 第一次在 T2V 上和 specialized video tokenizer 持平。

### 7.5 Image-to-3D Synthesis（Figure 14）

用 Trellis-SLAT 的 diffusion 框架，替换 tokenizer 为 ATOKEN-So/C。能生成 3D asset 但 fidelity 还没达到原版 Trellis。作者归因于 48 维 latent vs 8 维 latent，diffusion model 在高维空间需要重新调 hyperparameter。这个 limitation 诚实写了。

---

## 8. 我的 Intuition 和联想

### 8.1 为什么 4D sparse 表示 work

我觉得这个设计本质上是把 vision 的 modality 差异"几何化"了。Image / video / 3D 不是三种不同的 data type，而是 4D 时空中的不同 low-dimensional slice。这和 [4M, Mizrahi et al., 2023](https://arxiv.org/abs/2312.06647) 的思路一脉相承，但 ATOKEN 更进一步 —— 它把 reconstruction 和 understanding 也统一了。

类比 LLM 的 BPE：BPE 之所以 work 是因为所有 text 共享 character-level 的 sub-structure。ATOKEN 的 4D space 类似 —— 所有 visual data 共享 spatial 的 sub-structure，只是 temporal / depth 维度激活与否。

### 8.2 Gram loss 的更深层意义

rFID 分解的 86.6% / 13.4% 这个数字我觉得很值得 re-implement 验证。如果 covariance 真的占这么大比重，那 GAN 之所以 work 本质上也是因为 discriminator 在 match feature distribution 的 second-order statistics。Gram loss 是个 analytic 的替代品 —— 不需要训 discriminator，直接 closed-form 优化 covariance。

这个 insight 可能对其他 generative model 也有启发。比如 diffusion model 的 training loss 也可以考虑加 Gram term 来改善 texture 生成质量。事实上 [REPA, Yu et al., 2024](https://arxiv.org/abs/2410.06940) 已经在做类似的事（用 DINOv2 feature alignment），但用的是 first-order feature matching，不是 second-order。

### 8.3 Cross-modal transfer 的 mechanism

Stage 1 → Stage 3 image rFID 提升 19% 这件事，我觉得机制可能是：

- Video 训练让 encoder 学到 temporal invariance，这等价于一个 strong augmentation，强迫 image representation 抓住 content 而非 view-specific detail
- 3D 训练让 encoder 学到 view invariance，similar augmentation effect
- 两者都让 image feature 更 robust，reconstruction 时更 focus on "真正重要的"信息

这和 [Data2Vec, Baevski et al., 2022] 或者 multi-view self-supervised learning 的 mechanism 类似 —— 不同 view/perspective 的 supervision 起到 regularization 作用。

### 8.4 Capacity threshold 的 implication

Figure 7 的 Base vs So400m 对比我觉得是 paper 最重要的 finding 之一。它说明 unified tokenizer 不是"加 modality 就一定好"，而是有 minimum capacity requirement。这对社区是个 warning —— 小实验室复现 unified tokenizer 可能会失败，不是 idea 不 work，是 model 不够大。

这和 LLM 的 emergent ability 现象类似 —— 某些 capability 只在 scale 超过 threshold 后才出现。Unified visual tokenization 可能就是这样一个 emergent property。

### 8.5 和 LLM tokenizer 的类比与差距

ATOKEN 在 vision 里做的事很像 LLM 的 BPE，但还有几个 gap：

1. **Discrete token 的 codebook size**：BPE 是 ~50k vocab，ATOKEN 是 4096。差距还大。
2. **Token 的 semantic compositionality**：BPE token 有 clear semantic meaning（"un" + "happy"），ATOKEN 的 visual token 还没显示出这种 compositionality。
3. **Cross-modal alignment**：BPE token 跨语言有 cognate（"computer" / "computadora"），ATOKEN 跨 modality 的 token 对应关系还不明确。

未来 work 可能要往这些方向推。

### 8.6 限制和未来方向

Paper 诚实承认的 limitation：
- T2V 数据量受限，结果在 controlled setting 下做的
- Image-to-3D 还没达到 Trellis fidelity
- Long video understanding 不如 Oryx-ViT
- 没构建 comprehensive omnimodel（只 separate downstream 验证）

我额外想到的：
- 4D 表示对 audio 没有自然位置 —— 如果要加 audio modality，可能要扩展到 5D 或用单独的 token type
- FSQ 的 codebook 是 fixed 的，没有 EMA update，长期 scaling 可能不如 VQ
- KV-cache 的 temporal tiling 对 very long video（>1000 frames）可能还是有 memory pressure

---

## 9. 相关工作的 web links

核心 reference：
- [SigLIP2](https://arxiv.org/abs/2502.14786) — encoder 初始化
- [Trellis-SLAT](https://arxiv.org/abs/2412.01506) — 3D pipeline 基础
- [4M](https://arxiv.org/abs/2312.06647) — 4D RoPE 思路来源
- [FSQ](https://arxiv.org/abs/2309.15505) — discrete quantization
- [VILA-U](https://arxiv.org/abs/2409.04429) — 早期 unified tokenizer
- [UniTok](https://arxiv.org/abs/2502.20321) — 主要对比方法
- [VAVAE / Lightning-DiT](https://arxiv.org/abs/2501.01423) — image generation baseline
- [GigaTok](https://arxiv.org/abs/2504.08736) — scaling tokenizer
- [Cosmos](https://arxiv.org/abs/2501.03575) — video tokenizer
- [Wan](https://arxiv.org/abs/2503.20314) — video tokenizer
- [Hunyuan](https://arxiv.org/abs/2412.03603) — video tokenizer
- [TokenBridge](https://arxiv.org/abs/2503.16430) — discrete generation
- [SlowFast-LLaVA-1.5](https://arxiv.org/abs/2503.18943) — MLLM 验证
- [Oryx-ViT / Oryx-MLLM](https://arxiv.org/abs/2409.12961) — MLLM 对比
- [REPA](https://arxiv.org/abs/2410.06940) — generation alignment
- [LPIPS](https://arxiv.org/abs/1801.03924) — perceptual loss
- [Gram matrix / Gatys style transfer](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html) — style loss 来源
- [NaViT](https://arxiv.org/abs/2307.06304) — native resolution
- [DFN](https://arxiv.org/abs/2309.17425) — image data
- [Panda70M](https://arxiv.org/abs/2406.06022) — video data
- [Objaverse](https://arxiv.org/abs/2307.05663) — 3D data
- [SigLIP](https://arxiv.org/abs/2303.15387) — sigmoid loss
- [VQ-VAE](https://arxiv.org/abs/1711.00937) — discrete tokenization 经典
- [SD-VAE / LDM](https://arxiv.org/abs/2112.10752) — VAE tokenizer
- [ViTok](https://arxiv.org/abs/2501.09755) — transformer tokenizer scaling

---

## 10. 总结

ATOKEN 的核心 contribution 我觉得有三层：

1. **Conceptual**：把 vision modality 的差异几何化为 4D space 的不同 slice，这个 abstraction 让统一架构自然落地。
2. **Technical**：Gram loss 替代 GAN 是 transformer tokenizer 的关键 enabler，rFID 分解的 86.6% covariance observation 是个 actionable insight。
3. **Empirical**：cross-modal positive transfer（image rFID 在加 video/3D 后反而提升）是反直觉但 robust 的 finding，说明 unified training 在足够 capacity 下是 free lunch。

如果让我预测下一步，我觉得 community 会沿着这条路做：
- 更大 codebook 的 discrete tokenizer（接近 BPE scale）
- 真正的 omnimodel（一个 model 同时做 understanding + generation across modality）
- 把 audio / depth / tactile 加进 5D/6D 表示
- Gram loss 思路推广到 diffusion training loss

Andrej，这篇 paper 我觉得是 vision tokenizer 走向 LLM-style unification 的一个 milestone。它不一定最终形态，但 4D sparse + Gram loss + progressive curriculum 这套组合很可能会被后续工作继承。
