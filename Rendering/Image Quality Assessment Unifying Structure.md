---
source_pdf: Image Quality Assessment Unifying Structure.pdf
paper_sha256: bb82eab96af10d622780f95166b0379442741b90f83030a80dc3d0ebbd6a81ad
processed_at: '2026-08-05T09:06:20-07:00'
target_folder: Rendering
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 DISTS

## 痛点:现有 metric 都有个盲点

你拿一片草地拍两张照片,像素层面完全不一样(草叶位置全不同),但你看起来就是同一片草地。PSNR 给这俩打很低分,SSIM 也好不到哪去,LPIPS 一样翻车。整个 IQA 圈子几十年的 metric 都栽在这个地方。

原因特别简单:所有这些 metric 都默认两张图是"对齐"的,然后逐像素或者逐 feature 比较差异。texture 根本不满足这个假设。texture 的本质是**统计上的一致性**,具体每根草在哪根本无所谓。

这个缺陷有实际后果。做 image compression 的人早就想:texture 区域我统计地合成一个差不多的就好了,干嘛非得 bit-exact 重建?但只要你的 quality metric 一遇 texture resampling 就崩溃,你就没法用这个 metric 做 codec 的 loss,整个 workflow 卡死。

DISTS 就是第一个**明确容忍 texture resampling** 的 full-reference IQA metric。Reference: https://arxiv.org/abs/2004.07728

---

## 核心 idea:SSIM 的两件套搬到 VGG feature space

先回忆 SSIM 在干嘛。SSIM [3] 把 image quality 拆成两件事:
- **luminance**:两张图平均亮度差不差
- **structure**:两张图对应位置的局部协方差差不差

公式分别是:
$$l(x,y) = \frac{2\mu_x\mu_y + C_1}{\mu_x^2 + \mu_y^2 + C_1}$$
$$s(x,y) = \frac{2\sigma_{xy} + C_2}{\sigma_x^2 + \sigma_y^2 + C_2}$$

$\mu$ 是 mean,$\sigma^2$ 是 variance,$\sigma_{xy}$ 是 covariance,$C_1, C_2$ 是防止除零的小常数。

SSIM 的致命伤:这两个统计量都是在 **11×11 的 local sliding window** 里算的。一旦你的图整体平移几个 pixel,local mean 对得上,但 local covariance 直接错位,structure term 立刻崩。

DISTS 干了三件事:

### 1. 把 SSIM 的两个 term 搬到 VGG feature space

VGG16 在 ImageNet 上预训练好,抽 conv1_2 / conv2_2 / conv3_3 / conv4_3 / conv5_3 这 5 层的 feature map,加上 raw image 作为"第 0 层"。每层通道数 (3, 64, 128, 256, 512, 512),一共 1475 个 channel。

对每一个 channel,算这个 channel 的 global mean 和 global variance。注意:是**整张 feature map 一个值**,不是 local window。

然后对每一对对应 channel,直接套 SSIM 的 $l$ 和 $s$ 公式:

$$l(\tilde{x}_j^{(i)}, \tilde{y}_j^{(i)}) = \frac{2\mu_{\tilde{x}_j}^{(i)} \mu_{\tilde{y}_j}^{(i)} + c_1}{(\mu_{\tilde{x}_j}^{(i)})^2 + (\mu_{\tilde{y}_j}^{(i)})^2 + c_1}$$

$$s(\tilde{x}_j^{(i)}, \tilde{y}_j^{(i)}) = \frac{2\sigma_{\tilde{x}_j\tilde{y}_j}^{(i)} + c_2}{(\sigma_{\tilde{x}_j}^{(i)})^2 + (\sigma_{\tilde{y}_j}^{(i)})^2 + c_2}$$

上标 $(i)$ 是第几层 conv layer,$i \in \{0,1,...,5\}$。下标 $j$ 是这层里第几个 channel。$\mu$ 是这个 channel 整张 feature map 的 spatial mean(一个标量)。$\sigma_{xy}$ 是两个 channel 拉平之后的 covariance(也是一个标量)。

为什么搬到 VGG 就 work?pixel space 里距离和 perceptual distance 不均匀 [35][36],VGG feature space 经过 ImageNet 训练后更接近 perceptually uniform 的空间 [7]。

为什么用 global 而不是 local?texture 的特征由 statistics 决定,global statistics 天然对 spatial layout 不敏感。同一片草地的两块 patch,global mean 和 global variance 几乎一样,但 local window 对应位置上的 covariance 完全乱。

### 2. 用 channel mean 作为 texture descriptor

这部分是 paper 里最 surprising 的发现。Gatys 当年做 neural texture synthesis [32],用的是 Gram matrix,大概 30 万个统计量。Portilla & Simoncelli 2000 年的经典 wavelet texture model [15] 用 710 个统计量。

DISTS 发现:只用 1475 个 channel means,就能合成视觉上和原图几乎不可区分的 texture。Texture synthesis 的 objective 就是:

$$y^* = \arg\min_y \sum_{i,j} \left(\mu_{\tilde{x}_j}^{(i)} - \mu_{\tilde{y}_j}^{(i)}\right)^2$$

$x$ 是目标 texture,$y$ 是从 random noise 开始优化的 synthesized image。$i$ 是 layer index,$j$ 是 channel index。这个公式让 $y$ 在每一层的每个 channel 的平均 activation 等于 $x$ 的。

Paper Fig.3 做了 layer-wise ablation:只用 conv1_2 means 合成出来的是模糊颜色块,只用 conv5_3 合成出来的是大结构形状,累积到 conv5_3 的所有 means 合成出来视觉上和 reference 几乎一样。

为什么 1475 个 means 这么强?我的直觉:VGG 的每个 channel 在 ImageNet 上学到的 filter,大致对应自然图像的某种 pattern detector。一个 texture patch 在这些 filter 下的平均激活,大致相当于"这个 texture 在各种 semantic 维度上的 energy 分布"。这其实和 Heeger-Bergen 1995 [51] 用 steerable pyramid 做 texture analysis 的思路同源,只是 VGG filter 比 steerable pyramid 更 expressive。

Reference: https://www.cns.nyu.edu/~heeger/papers/heeger-bergensiggraph95.pdf

### 3. 加权求和,只训权重

最终的 DISTS distance:

$$D(x,y; \alpha, \beta) = 1 - \sum_{i=0}^m \sum_{j=1}^{n_i} \left(\alpha_{ij} l(\tilde{x}_j^{(i)}, \tilde{y}_j^{(i)}) + \beta_{ij} s(\tilde{x}_j^{(i)}, \tilde{y}_j^{(i)})\right)$$

$\alpha_{ij} \geq 0$ 是第 $i$ 层第 $j$ channel 的 texture 项权重,$\beta_{ij} \geq 0$ 是 structure 项权重,约束 $\sum_{i,j}(\alpha_{ij} + \beta_{ij}) = 1$。

VGG 的卷积核**完全固定不动**,只训这 2950 个标量权重。参数量极小,过拟合风险低。

---

## 训练:两个 loss 在打架

DISTS 的训练特别有意思,同时优化两个互相矛盾的目标。

**目标 1:对 human quality rating 拟合**

用 KADID-10k [59],81 张 reference × 25 种 distortion × 5 个 level = 10125 张图,带 human MOS。Loss 是:

$$E_1(x, y; \alpha, \beta) = |D(x, y; \alpha, \beta) - q(y)|$$

$q(y)$ 是 normalized MOS,$|\cdot|$ 是 L1。这个 loss 让 DISTS 对 blur / noise / JPEG 这些传统 distortion 敏感。

**目标 2:对 texture resampling 不敏感**

用 DTD dataset [60],5640 张 texture 图(47 类)。从同一张 texture 裁两个不同的 $256\times256$ patch $z_1, z_2$,loss 是:

$$E_2(z; \alpha, \beta) = D(z_1, z_2; \alpha, \beta)$$

就是让 DISTS 对同一张 texture 的不同 crop 给低距离。这个 loss 直接 push 模型去关注 statistics 而非 spatial layout。

**总 loss**:

$$E = \frac{1}{|\mathcal{Q}|}\sum_{x,y \in \mathcal{Q}} E_1 + \lambda \frac{1}{|\mathcal{T}|}\sum_{z \in \mathcal{T}} E_2$$

$\lambda = 1$。$\mathcal{Q}$ 是 KADID-10k 的 batch,$\mathcal{T}$ 是 DTD 的 batch。Adam,batch 32,lr 1e-4,每 1K iter 减半,5K iter 收敛,RTX 2080 上一小时训完。

$\lambda=1$ 这个选择背后有含义:quality matching 和 texture invariance 是两种不同的"什么算相似"的 prior,等权让模型自己学到在什么 channel 上 focus structure、什么 channel 上 focus texture。从 ablation 看,早期 conv layer 学到的 $\alpha$ 大(更关注 texture mean),后期 layer 学到的 $\beta$ 大(更关注 structure correlation)。

---

## metric property:injective 的重要性

这个很多人会跳过,但 Karpathy 你做 optimization 一定懂这个痛点。

大多数 IQA metric 在用作 optimization loss 时会卡在 spurious minimum。原因是它们对应的 feature transformation 是 **surjective** 的——不同的 image 在 feature space 可能 collapse 到同一点,所以 loss=0 不蕴含 $y=x$。

GMSD [12] 只取 gradient,丢掉 luminance,明显 surjective。GTI-CNN [19] 故意用 surjective mapping 换取 geometric invariance,代价是丢掉大量 structure 信息。LPIPS [7] 基于 VGG 但没强制 injective,优化时残留 visible artifacts(见 paper Fig.2)。

DISTS 的解决方法特别简单粗暴:把 raw image 直接 concat 到 representation 里当第 0 层。这样如果两个 image 的所有 feature 都相同,那 raw pixel 也必须相同,所以 $D(x,y)=0 \Leftrightarrow x=y$。

Paper Lemma 1 证明 $d(x,y) = \sqrt{D(x,y)}$ 是 proper metric:non-negativity、symmetry、triangle inequality、identity of indiscernibles 全满足。证明用到 Brunet et al. [58] 关于 SSIM-motivated distance 是 metric 的结果,加 Cauchy-Schwarz 推 triangle inequality。

实际好处:用 DISTS 做 loss 跑 image restoration / neural compression,优化不会 drift 到 spurious solution。这点在 Ding et al. 后续 paper [80] 里有验证:https://arxiv.org/abs/2005.01338

---

## 还有一个 trick:换 pooling

VGG 原版用 max pooling。Henaff & Simoncelli [42] 发现 max pooling 在做 geodesic interpolation 时会产生 visible aliasing artifacts。原因:max pooling 不满足 Nyquist criterion——下采样之前必须 low-pass filter 到 cutoff < π/2,否则高频混叠。

DISTS 把所有 max pooling 换成 weighted ℓ₂ pooling:

$$P(x) = \sqrt{g * (x \odot x)}$$

$x$ 是 feature map,$g$ 是 5×5 Hanning window,$\odot$ 是 pointwise product,$*$ 是 2D 卷积。这个操作先平方、再 blur、再开方,本质上是个 energy-type pooling。

这操作有三个 deep reasons:
1. 满足 Nyquist,de-aliasing
2. ℓ₂ pooling 对应 V1 complex cell 的 energy model [44],和 HVS early vision 对齐
3. 和 scattering transform [45] 的 complex modulus 同源

References:
- Henaff-Simoncelli geodesics: https://arxiv.org/abs/1511.04680
- V1 complex cells: https://www.jneurosci.org/content/35/44/14829
- Scattering transform: https://arxiv.org/abs/1203.1003

---

## 实验数字背后的直觉

### 标准 IQA 库(Table 1)

DISTS 在 LIVE / CSIQ / TID2013 上 PLCC 大约 0.93-0.95,和 MAD / FSIM / GMSD 这些 SOTA 差不多。注意 DISTS 没在这三个库上训练过(只在 KADID-10k 训),而 MAD/FSIM/GMSD 这帮老 metric 在这三个库上被 IQA 社区反复调了十几年,有 over-adapting 风险。

### BAPPS(Table 2)

这是 Zhang et al. [7] 的大规模 patch similarity 数据集,包含传统 distortion 和 real-world algorithm distortion(denoising autoencoder / super-resolution / deblurring / colorization / frame interpolation)。

DISTS 没在 BAPPS 训练过,overall 2AFC score 0.689,和 LPIPS 0.692 几乎一样。LPIPS 是直接在 BAPPS 上训练的,DISTS 是 zero-shot 还能持平,generalization 强。

2AFC score 公式:$p\hat{p} + (1-p)(1-\hat{p})$,$p$ 是人类选某张图的比例,$\hat{p} \in \{0,1\}$ 是模型 binary preference。

### Texture similarity(Table 3)

这才是 DISTS 的主场。SynTEX 上 PLCC 0.901,作者自建 TQD 上 PLCC 0.903。其他所有方法(SSIM 0.619 / LPIPS 0.674 / IGSTQA 0.816)都明显落后。

TQD 的构造:10 张 Pixabay texture × 7 种传统 distortion × 3 级别 + 4 个 synthesis 算法结果 [15][32][72][73] + 4 张 original crop。10 个 subject 做 ranking,用 reciprocal rank fusion 聚合:

$$r(x) = \sum_{k=1}^K \frac{1}{\gamma + r_k(x)}$$

$r_k(x)$ 是第 $k$ 个 subject 给 $x$ 的 rank,$\gamma$ 是控制 outlier 影响的常数。

### Texture classification & retrieval(Table 4)

Brodatz [75] 112 类 texture,每类取 9 个 $256\times256$ patch。DISTS 用作 k-NN 的距离度量:
- Color Brodatz classification acc 0.995
- Color Brodatz retrieval mAP 0.988
- Grayscale 略低(0.968 / 0.951),说明 color 信息重要

LPIPS 也能做(CBT acc 0.960),但 DISTS 明显更强。SIFT BoW baseline 在 classification 上 0.924,在 retrieval 上 0.859。DISTS 完胜。

mAP 公式:
$$\text{mAP} = \frac{1}{Q}\sum_{q=1}^Q \left(\frac{1}{K}\sum_{k=1}^K P(k) \cdot \text{rel}(k)\right)$$

$Q$ 是 query 数,$K$ 是 relevant 图数,$P(k)$ 是 cutoff $k$ 处的 precision,$\text{rel}(k) \in \{0,1\}$ 是 relevance indicator。

ALOT [77] 这种 lighting/viewpoint 变化大的库上,DISTS zero-shot classification acc 0.926,不及 supervised CNN [79] 的 0.993。DISTS 只 capture visual appearance,不学习 lighting/viewpoint invariance。

### Geometric invariance(Table 5)

LIVE augmented with 5% shift / 3° rotation / 1.05 dilation / mixed。DISTS SRCC 总分 0.928,完爆所有方法。第二名 GTI-CNN 0.875(但 GTI-CNN 丢了 perceptual 信息)。

DISTS 的几何不变性来自三个 source:
1. ℓ₂ pooling de-aliasing,下采样不混叠
2. Global statistics 对 spatial shift 鲁棒
3. Texture invariance training 顺带学到 mild geometric robustness

### Ablation(Table 6)

这是最有信息量的表。从 LPIPS 逐步加 modification:

(a) LPIPS baseline:SynTEX PLCC 0.591
(b) + ℓ₂ pooling:0.594(微弱提升,核心是 de-aliasing)
(c) + input image:0.582(几乎无影响,但保 unique minimum)
(d) + local SSIM 11×11:0.738(local 仍受 alignment 影响)
(e) + global SSIM:0.868(全球统计大幅提升)
(f) (c) + E₂ training:0.780
(g) (d) + E₂:0.774
(h) (e) + E₂ = DISTS:0.901

最关键的 two observations:
- **global SSIM 比 local SSIM 强很多**(SynTEX 0.868 vs 0.738)。Local sliding window 仍受 spatial alignment 影响,global 自然鲁棒。
- **E₂ training 显著改善 texture 任务**(0.868 → 0.901),代价是标准 IQA 库上略降(0.957 → 0.954 on LIVE PLCC)。

---

## 把直觉 build 起来

1. **HVS 对 texture 是 statistics-based perception,对 structure 是 correlation-based perception**。这两件事在 VGG feature space 里可以分别用 channel mean 和 channel covariance 来 quantify。

2. **VGG 的 channel 在 ImageNet 上学到的 filter,大致对应自然图像的 semantic pattern detector**。一个 texture 在这些 filter 下的平均激活,就是这个 texture 的"semantic fingerprint"。

3. **SSIM 的两个 term 在 pixel space 太弱,搬到 VGG channel space 后变得 expressive**。而且把 local window 换成 global,直接解决 alignment 假设。

4. **Texture invariance training 通过 push 同一 texture 的不同 crop 接近,强迫模型关注 statistics 而非 layout**。这是把 Julesz 1962 的 conjecture 用 data-driven 方式 encode 进 loss。

5. **Injective 保证 unique minimum**。对 perceptual optimization 至关重要。直接 concat raw image 是简单粗暴但有效的方案。

6. **ℓ₂ pooling 既是 engineering fix(de-aliasing)又是 biological alignment(V1 complex cells)**。这种改动一般一个方法最多占一头,DISTS 同时占两头。

7. **双 loss 训练是 implicit multi-task regularization**。Quality matching 提供 structure sensitivity,texture invariance 提供 statistics tolerance。模型在 channel 维度上学到"什么时候 focus structure、什么时候 focus texture"的自动分配。

8. **Proper metric + injective + differentiable**。这三个性质加起来使 DISTS 成为 ideal perceptual optimization loss。LPIPS 缺 injective,SSIM 缺 texture awareness,DISTS 两边都补上了。

---

## 适用场景与限制

**适用**:
- Texture synthesis / generation 的 evaluation
- Neural image compression 的 loss(尤其 texture 区域统计重建)
- Image restoration 的 perceptual loss
- Style transfer 评估
- Texture retrieval / classification

**限制**:
- Global measure,不给 per-pixel quality map
- 大角度 rotation / 大尺度 dilation 仍 fail
- Lighting / viewpoint 变化大的场景不如 supervised 方法
- Non-convex,optimization 收敛慢于 SSIM
- 没在 KADID-10k 之外的库上做 cross-validation,可能有 dataset bias

---

## 一句话总结

DISTS = VGG feature space + global SSIM + texture invariance training + injective guarantee。这四件事拼起来,把 SSIM 从 pixel-level local similarity 升级到 feature-level global similarity,顺便解决了 texture perception 和 perceptual optimization 的 unique minimum 问题。代码:https://github.com/dingkeyan93/DISTS

---

# DISTS: Deep Image Structure and Texture Similarity 深度解析

## Paper 基本信息

**Title**: Image Quality Assessment: Unifying Structure and Texture Similarity  
**Authors**: Keyan Ding, Kede Ma, Shiqi Wang, Eero P. Simoncelli  
**Venue**: IEEE Transactions on Image Processing (TIP)  
**Code**: https://github.com/dingkeyan93/DISTS  
**ArXiv**: https://arxiv.org/abs/2004.07728  
**Author page (Kede Ma)**: https://kedema.org/  
**Simoncelli lab**: https://www.cns.nyu.edu/~eero/  

---

## 1. 核心动机:为什么需要新metric

传统 full-reference IQA methods(PSNR、SSIM、FSIM、VIF、GMSD、LPIPS、PieAPP、DeepIQA)都依赖 **point-by-point alignment**。这个假设在 texture region 完全崩溃:同一片草地裁出两块patch,像素层面差异巨大,但人类视觉几乎无法区分(参见 Fig.1)。

这暗示了一个 deep insight:**HVS 对 texture 的感知由 statistics 而非 spatial layout 决定**。这要追溯到 Julesz 1962 年的 visual texture discrimination 理论 [16],以及 Portilla & Simoncelli 2000 年的 parametric texture model [15]。

实际应用价值:如果 IQA metric 能容忍 texture resampling,那么新一代 compression engine 可以"统计性地合成"texture region 而非逐像素重建 [9][10]——这是 JPEG XL、AV1、neural compression 都想做的事。

相关 link:
- Julesz 1962: https://ieeexplore.ieee.org/document/1057698
- Portilla-Simoncelli texture model: https://www.cns.nyu.edu/~eero/ABSTRACTS/portilla-iijcv00.html
- Gatys neural texture synthesis: https://arxiv.org/abs/1505.07376
- LPIPS: https://arxiv.org/abs/1801.03924

---

## 2. 方法架构总览

DISTS 的 pipeline 可以拆成 4 个模块:

### 2.1 Initial Transformation f: R^n → R^r

基于 **VGG16** pretrained on ImageNet [14][41],选 5 个 conv layer: conv1_2, conv2_2, conv3_3, conv4_3, conv5_3。但做了两个关键 modification:

**(a) 替换 max pooling 为 weighted ℓ₂ pooling (Eq.1)**

$$P(x) = \sqrt{g * (x \odot x)}$$

变量解释:
- $x$: input feature map
- $g(\cdot)$: Hanning window blurring kernel,approximate Nyquist cutoff at π/2 radians/sample
- $\odot$: pointwise (Hadamard) product
- $*$: 2D convolution
- stride 2 实现 downsampling

**为什么这么改?** Max pooling 在 geodesic interpolation 中间会产生 visible aliasing artifacts(Henaff & Simoncelli [42] 的发现)。Nyquist 定理要求 sub-sample by 2 之前必须用 cutoff 频率 < π/2 的 low-pass filter。Hanning window ≈ $\frac{1}{2}(1-\cos(2\pi n/N))$ 满足这个要求。

更深的 connection: ℓ₂ pooling 描述了 V1 complex cells 的 energy model [44],也和 scattering transform [45] 的 complex modulus 同源。所以这不仅是工程 trick,而是和 HVS early vision 对齐。

Ref:
- Henaff-Simoncelli geodesics: https://arxiv.org/abs/1511.04680
- V1 complex cell model (Vintch et al.): https://www.jneurosci.org/content/35/44/14829
- Scattering transform (Bruna-Mallat): https://arxiv.org/abs/1203.1003

**(b) 加入 input image 作为 "zeroth" layer,保证 injective**

$$f(x) = \{\tilde{x}_j^{(i)}; i=0, \ldots, m; j=1, \ldots, n_i\}$$

变量:
- $i$: layer index,$i=0$ 对应原始 RGB image, $i \in \{1,\ldots,5\}$ 对应 conv1_2 到 conv5_3
- $j$: channel index within layer $i$
- $n_i$: 通道数,$(n_0, n_1, n_2, n_3, n_4, n_5) = (3, 64, 128, 256, 512, 512)$,总通道数 1475
- $m=5$: 选的 conv layer 数
- $\tilde{x}^{(0)} = x$: 第零层就是 raw image

**为什么需要 injective?** Ma et al. [50] 证明: 在 Gaussian random weights + ReLU 下,2-layer CNN 当 output dimension 增加 logarithmic factor 时是 injective 的。但这个结果不能直接推广到 deep VGG。最简单粗暴的解决: 把 raw pixel 直接 concat 到 representation 里,这样如果两个 image 的 feature representation 完全相同,那 pixel 也相同 ⇒ injective。

**Injective 的实际好处** (Fig.2 演示): 给定 reference $x$ 和 noise initialization $y_0$,通过 $y^* = \arg\min_y D(x,y)$ 用 gradient descent recover $x$。
- GTI-CNN [19]、GMSD [12](surjective mapping)→ 完全失败
- LPIPS [7](无 injective 保证)→ 残留 artifacts
- DISTS → 完美恢复

这是 perceptual optimization 的关键:loss landscape 必须有 **unique minimum**,否则 image restoration 会卡在 spurious solution。

Ref:
- Ma et al. invertibility of CNN: https://arxiv.org/abs/1807.00913
- Image quality models for optimization (Ding et al. follow-up): https://arxiv.org/abs/2005.01338

### 2.2 Texture Representation: Channel Means 即 Texture

这是 paper 最 surprising 的发现。Gatys 用 Gram matrix (~306K 参数)做 texture synthesis [32]。本文发现:**只用 1475 个 channel means 就够了**。

**Texture synthesis objective (Eq.4)**:

$$y^* = \arg\min_y \sum_{i,j} \left( \mu_{\tilde{x}_j}^{(i)} - \mu_{\tilde{y}_j}^{(i)} \right)^2$$

变量:
- $x$: target texture image
- $y$: 当前 synthesized image(从 random noise 初始化)
- $\mu_{\tilde{x}_j}^{(i)}$: spatial average over spatial dimensions of channel $j$ in layer $i$ for reference
- $\mu_{\tilde{y}_j}^{(i)}$: 同上 for current $y$

**Fig.3 的 layer-wise ablation**:
- 只用 conv1_2 means → 捕捉 intensity / color
- 只用 conv5_3 means → 捕捉 shape / structure
- 累积到 conv5_3 → 与 reference 视觉几乎不可区分

**Fig.4 与经典对比**:
| Method | # Statistics | Quality |
|---|---|---|
| Portilla & Simoncelli [15] | ~710 | 中等 |
| Gatys et al. [32] | ~306K | 最好 |
| DISTS texture model | 1475 | 中间偏上 |

这里有个 **subtle point**: Gatys 的 306K 统计量比 image pixel 数还多,synthesis 的"多样性"其实反映的是 **optimization 的 local minima**,而非 implicit probability distribution 的 entropy (Ustyuzhaninov et al. [54] 的发现)。DISTS 的 1475 个 means 远小于 pixel 数,所以 implicit distribution 有真实 entropy。

为什么 channel means 这么强? 我个人的解读: VGG 训练在 ImageNet 上学到了自然图像的 marginal activation distributions。对一个 texture patch,各 channel 的 mean activation 大致对应"该 texture 在各 semantic filter bank 下的 energy"。这其实呼应了 Heeger-Bergen 1995 [51] 的 pyramid texture analysis——只不过 VGG filter 比 steerable pyramid 更 expressive。

Ref:
- Gatys neural texture: https://arxiv.org/abs/1505.07376  
- Ustyuzhaninov "what does it take": https://arxiv.org/abs/1701.04340
- Heeger-Bergen texture: https://www.cns.nyu.edu/~heeger/papers/heeger-bergensiggraph95.pdf

### 2.3 Perceptual Distance Measure: SSIM-style 在 VGG feature 上

对每一对 channel $\tilde{x}_j^{(i)}, \tilde{y}_j^{(i)}$,定义两个 similarity:

**Texture / luminance similarity (Eq.5)**:

$$l(\tilde{x}_j^{(i)}, \tilde{y}_j^{(i)}) = \frac{2 \mu_{\tilde{x}_j}^{(i)} \mu_{\tilde{y}_j}^{(i)} + c_1}{\left(\mu_{\tilde{x}_j}^{(i)}\right)^2 + \left(\mu_{\tilde{y}_j}^{(i)}\right)^2 + c_1}$$

变量:
- $\mu_{\tilde{x}_j}^{(i)}, \mu_{\tilde{y}_j}^{(i)}$: scalar global means
- $c_1 = 10^{-6}$: 稳定常数

这正好是 SSIM [3] 中的 luminance comparison term $l(x,y) = (2\mu_x\mu_y + C_1)/(\mu_x^2 + \mu_y^2 + C_1)$,只是 domain 从 raw pixel 换到 VGG channel。

**Structure similarity (Eq.6)**:

$$s(\tilde{x}_j^{(i)}, \tilde{y}_j^{(i)}) = \frac{2 \sigma_{\tilde{x}_j \tilde{y}_j}^{(i)} + c_2}{\left(\sigma_{\tilde{x}_j}^{(i)}\right)^2 + \left(\sigma_{\tilde{y}_j}^{(i)}\right)^2 + c_2}$$

变量:
- $\sigma_{\tilde{x}_j}^{(i)})^2$: global variance over spatial positions of channel $\tilde{x}_j^{(i)}$
- $\sigma_{\tilde{x}_j \tilde{y}_j}^{(i)}$: global covariance between two flattened channels
- $c_2 = 10^{-6}$

注意这里 $s$ 用的是 **global** covariance over 全图,不是 local sliding window。这是与原 SSIM 的关键区别:global statistics 自然对 spatial shift 不敏感。原 SSIM 是 11×11 Gaussian window local statistics,所以要求 pixel alignment。

**最终 DISTS distance (Eq.7)**:

$$D(x, y; \alpha, \beta) = 1 - \sum_{i=0}^{m} \sum_{j=1}^{n_i} \left( \alpha_{ij} \, l(\tilde{x}_j^{(i)}, \tilde{y}_j^{(i)}) + \beta_{ij} \, s(\tilde{x}_j^{(i)}, \tilde{y}_j^{(i)}) \right)$$

变量:
- $\alpha_{ij} \geq 0$: learnable weight for texture term,channel $j$ at layer $i$
- $\beta_{ij} \geq 0$: learnable weight for structure term
- 约束: $\sum_{i=0}^m \sum_{j=1}^{n_i} (\alpha_{ij} + \beta_{ij}) = 1$
- VGG kernels 固定,**只训练 $\{\alpha, \beta\}$**(只 2950 个参数,极小)

**Proper Metric (Lemma 1)**:

$$d(x,y) = \sqrt{D(x,y)}$$

满足:non-negativity、symmetry、triangle inequality、identity of indiscernibles。

**Triangle inequality 证明思路**:
$$d_{ij}(x,y) = \sqrt{\alpha_{ij}(1 - l) + \beta_{ij}(1 - s)}$$
Brunet et al. [58] 证明 $d_{ij}$ 是 metric。然后:
$$d(x,y) = \sqrt{\sum_{i,j} d_{ij}^2(x,y)}$$
是 Hilbertian norm of metric components,通过 Cauchy-Schwarz:
$$\sqrt{\sum_{ij}(d_{ij}(x,z) + d_{ij}(z,y))^2} \leq \sqrt{\sum_{ij}d_{ij}^2(x,z)} + \sqrt{\sum_{ij}d_{ij}^2(z,y)}$$

Ref:
- Original SSIM (Wang et al. 2004): https://ieeexplore.ieee.org/document/1284395
- Brunet et al. SSIM metric properties: https://ieeexplore.ieee.org/document/6034630
- CW-SSIM (complex wavelet SSIM): https://www.cns.nyu.edu/~eero/ABSTRACTS/wangZ-Fri05.html

### 2.4 训练:双目标优化

**Loss (Eq.16)**:

$$E(\mathcal{Q}, \mathcal{T}; \alpha, \beta) = \frac{1}{|\mathcal{Q}|} \sum_{x, y \in \mathcal{Q}} E_1(x, y; \alpha, \beta) + \lambda \frac{1}{|\mathcal{T}|} \sum_{z \in \mathcal{T}} E_2(z; \alpha, \beta)$$

其中:

**Quality matching term (Eq.14)**:
$$E_1 = |D(x, y; \alpha, \beta) - q(y)|$$
- $q(y)$: normalized MOS from KADID-10k [59] (81 ref × 25 distortions × 5 levels = 10125 images)
- $|\cdot|$: L1 distance to ground-truth MOS

**Texture invariance term (Eq.15)**:
$$E_2 = D(z_1, z_2; \alpha, \beta)$$
- $z_1, z_2$: 两个 $256 \times 256 \times 3$ patches 裁自同一张 DTD [60] texture image
- DTD: 5640 张,47 类

**超参数**:
- $\lambda = 1$(权衡 quality vs texture invariance)
- Adam [68],batch size 32,lr=1e-4
- 每 1K iter lr 减半,共 5K iter(~1 hour on RTX 2080)
- Zeroth stage weights 投影到 [0.02, 1] 保 unique minimum
- $c_1 = c_2 = 10^{-6}$
- 输入 re-scale 到 short side 256 pixel(沿用 SSIM 习惯)

Ref:
- KADID-10k: http://database.mmsp-kn.de/kadid-10k.html
- DTD: https://www.robots.ox.ac.uk/~vgg/data/dtd/

---

## 3. 与其它 IQA 方法的联系(Section 2.5 详解)

| Method | 与 DISTS 关系 |
|---|---|
| SSIM [3] / MS-SSIM [63] | DISTS 是 multi-scale hierarchical 版本,cross-scale 权重由 human-rated natural images 校准 |
| CW-SSIM [25] | CW-SSIM 用 complex wavelet phase pattern 做 translation invariance;DISTS 通过 texture invariance training 顺带获得 |
| Adaptive linear system [18] | DISTS 可视为 adaptive **nonlinear** system,structure 比较=structural distortion, texture 比较=non-structural distortion |
| Gatys style/content [55] | style Gram matrix 与 content MSE 互相冗余,合并非唯一极小;DISTS 用 SSIM-style mean/cov 代替,得到 unique minimum |
| Johnson perceptual loss [67] | 直接 $\ell_p$ norm on VGG features,用 late layer 引入 semantic,无 texture invariance |

---

## 4. 实验详解

### 4.1 标准 IQA 数据库(Table 1)

| Method | LIVE PLCC | CSIQ PLCC | TID2013 PLCC |
|---|---|---|---|
| PSNR | 0.865 | 0.819 | 0.677 |
| SSIM | 0.937 | 0.852 | 0.777 |
| MS-SSIM | 0.940 | 0.889 | 0.830 |
| MAD | 0.968 | 0.950 | 0.827 |
| VIF | 0.960 | 0.913 | 0.771 |
| FSIMc | 0.961 | 0.919 | 0.877 |
| GMSD | 0.957 | 0.945 | 0.855 |
| DeepIQA | 0.940 | 0.901 | 0.834 |
| PieAPP | 0.908 | 0.877 | 0.859 |
| LPIPS | 0.934 | 0.896 | 0.749 |
| **DISTS** | **0.954** | **0.928** | **0.855** |

**关键直觉**: DISTS 没有在 LIVE/CSIQ/TID2013 上训练,但 competitive。MAD/FSIM/GMSD 在这三个老库上的高分数可能有 **over-adapting** 风险——这些库被 IQA 社区用了十几年,模块选择可能无意中针对它们。

PLCC fitting 用 4 参数 logistic (Eq.17):
$$\hat{D} = \frac{\eta_1 - \eta_2}{1 + \exp\left(-(D - \eta_3)/|\eta_4|\right)} + \eta_2$$
$\eta_1, \eta_2$: upper/lower asymptote
$\eta_3$: inflection point
$\eta_4$: slope at inflection

### 4.2 BAPPS(Table 2)

| Method | Traditional | CNN-based | All synthetic | All real-world | All |
|---|---|---|---|---|---|
| Human | 0.808 | 0.844 | 0.826 | 0.695 | 0.739 |
| LPIPS (trained on BAPPS) | 0.760 | 0.828 | 0.794 | 0.641 | 0.692 |
| **DISTS (not trained on BAPPS)** | **0.772** | **0.822** | **0.797** | **0.651** | **0.689** |

**DISTS 没在 BAPPS 训练过**却与 LPIPS 持平(0.689 vs 0.692),generalization 强。

BAPPS 2AFC score 公式:
$$\text{score} = p\hat{p} + (1-p)(1-\hat{p})$$
- $p$: 人类选择某张 image 的比例
- $\hat{p} \in \{0, 1\}$: model 的 binary preference

Ref:
- BAPPS dataset: https://richzhang.github.io/PerceptualSimilarity/

### 4.3 Texture Similarity(Table 3)

| Method | SynTEX PLCC | TQD PLCC |
|---|---|---|
| SSIM | 0.619 | 0.330 |
| LPIPS | 0.674 | 0.402 |
| STSIM | 0.650 | 0.422 |
| IGSTQA | 0.816 | 0.804 |
| **DISTS** | **0.901** | **0.903** |

在 SynTEX [71] 和作者自建 TQD 上 DISTS 大幅领先。TQD 包含: 10 张 Pixabay texture × 7 种传统 distortion × 3 级别 × 4 个 synthesis 算法 [15][32][72][73] + 4 张 original crop。

10 个 subject 用 **reciprocal rank fusion (Eq.18)** 聚合 ranking:
$$r(x) = \sum_{k=1}^K \frac{1}{\gamma + r_k(x)}$$
- $r_k(x)$: 第 $k$ 个 subject 给 $x$ 的 rank
- $\gamma$: 常数,缓解 outlier 影响

### 4.4 Texture Classification & Retrieval(Table 4)

Brodatz [75]: 112 类 texture,每类取 9 个 $256\times256$ patch。

| Method | CBT class acc | GBT class acc | CBT mAP | GBT mAP |
|---|---|---|---|---|
| SSIM | 0.397 | 0.210 | 0.371 | 0.145 |
| LPIPS | 0.960 | 0.861 | 0.951 | 0.839 |
| SIFT BoW + k-NN | 0.924 | 0.928 | 0.859 | 0.865 |
| **DISTS** | **0.995** | **0.968** | **0.988** | **0.951** |

mAP (Eq.19):
$$\text{mAP} = \frac{1}{Q}\sum_{q=1}^Q \left( \frac{1}{K}\sum_{k=1}^K P(k) \cdot \text{rel}(k) \right)$$
- $Q$: # queries
- $K$: # relevant images
- $P(k)$: precision at cutoff $k$
- $\text{rel}(k) \in \{0, 1\}$: relevance indicator

ALOT dataset [77](250 textures × 100 viewing angles) DISTS k-NN classification acc=0.926,虽不及 supervised CNN [79] (0.993),但 zero-shot 已经 strong。

Ref:
- Brodatz: http://multibandtexture.recherche.usherbrooke.ca/
- ALOT: https://aloi.science.uva.nl/

### 4.5 Geometric Invariance(Table 5)

LIVE augmented with: 5% horizontal shift / 3° rotation / 1.05 dilation / mixed。

| Method | Trans | Rot | Dil | Mixed | Total |
|---|---|---|---|---|---|
| PSNR | 0.159 | 0.153 | 0.152 | 0.146 | 0.195 |
| SSIM | 0.171 | 0.168 | 0.177 | 0.166 | 0.190 |
| NLPD | 0.062 | 0.074 | 0.083 | 0.066 | 0.112 |
| LPIPS | 0.811 | 0.908 | 0.893 | 0.861 | 0.779 |
| GTI-CNN | 0.864 | 0.906 | 0.904 | 0.890 | 0.875 |
| **DISTS** | **0.948** | **0.939** | **0.946** | **0.937** | **0.928** |

DISTS 几何不变性来自三点:(a) ℓ₂ pooling 抗 aliasing,(b) global statistics 抗 local shift,(c) texture invariance training 间接学到 mild geometric robustness。

### 4.6 Ablation Study(Table 6)

从 LPIPS 逐步加 modification:

| Variant | LIVE PLCC | TID2013 PLCC | SynTEX PLCC | TQD PLCC | LIVE_Aug PLCC |
|---|---|---|---|---|---|
| (a) LPIPS baseline | 0.934 | 0.850 | 0.591 | 0.403 | 0.801 |
| (b) + ℓ₂ pooling | 0.937 | 0.851 | 0.594 | 0.410 | 0.807 |
| (c) + input image | 0.935 | 0.851 | 0.582 | 0.410 | 0.795 |
| (d) + local SSIM (11×11) | 0.950 | 0.853 | 0.738 | 0.664 | 0.798 |
| (e) + global SSIM | 0.955 | 0.859 | 0.868 | 0.780 | 0.899 |
| (f) (c) + E₂ training | 0.934 | 0.791 | 0.780 | 0.680 | 0.830 |
| (g) (d) + E₂ | 0.929 | 0.801 | 0.774 | 0.672 | 0.820 |
| **(h) (e) + E₂ = DISTS** | **0.954** | **0.855** | **0.901** | **0.903** | **0.931** |

**关键 takeaways**:
1. ℓ₂ pooling 对标准 IQA 微弱帮助,核心价值是 de-aliasing
2. 加 input image 几乎无影响,但保证 unique minimum
3. **global SSIM 大幅优于 local SSIM**(SynTEX 0.868 vs 0.738)——local 仍受 alignment 影响
4. E₂ training 改善 texture & geometric,略微伤 standard IQA
5. 完整 DISTS = (e) + (f) 的 best of both worlds

---

## 5. Intuition 总结:为什么 DISTS work

1. **Texture = global statistics, Structure = local correlations**。HVS 对 texture 的 preattentive discrimination 由 first/second order statistics 驱动(Julesz conjecture 的现代版本)。

2. **VGG feature space 是 perceptually uniform 的**。Pixel space 中距离与 perceptual distance 不均匀 [35][36];VGG 经过 ImageNet training 学到了对自然图像 marginal 的 normalization,使得 Euclidean-like distance 在 feature space 更 meaningful。

3. **Global SSIM = texture-aware SSIM**。SSIM 的 luminance term $l$ 用 global mean ⇒ texture 项;SSIM 的 contrast-structure term $s$ 用 global covariance ⇒ structure 项,但这个 covariance 在 channel 上是整图一个值,对 spatial shift 自然鲁棒。

4. **Channel means 既是 texture descriptor 也是 luminance proxy**。这统一了 SSIM 的"luminance comparison"和 texture statistics 的"marginal activation"——它们在 VGG channel 上是同一回事。

5. **Injectivity = 唯一最小**。这点在 deep image prior、neural compression、perceptual loss based restoration 里至关重要。Loss 必须 uniquely minimized at $y=x$,否则优化会 drift 到 spurious solution。

6. **双任务 training 是 implicit multi-task regularization**。Quality matching 提供结构敏感性,texture invariance 提供统计容忍度,$\lambda=1$ 的平衡让模型学到何时关注 structure、何时关注 texture 的"自动切换"。

---

## 6. Limitations & Follow-up

Paper 自己提到:
- DISTS 是 global measure,不能给 spatial quality map(不像 SSIM 给 per-pixel quality map)
- 几何不变性对大角度 rotation / 大尺度 dilation 仍 fail
- 在 ALOT 这种 viewpoint/lighting 变化大的库上不如 supervised CNN
- Highly non-convex, optimization 收敛慢于 SSIM

后续工作:
- Ding et al. "Comparison of image quality models for optimization" [80]: https://arxiv.org/abs/2005.01338 ——验证 DISTS 作为 perceptual optimization loss 用于 denoising/deblurring/SR/compression 的效果
- DISTS 后续被广泛用作 neural image compression 的 loss,例如在 learned JPEG/AOM codec 中

---

## 7. 相关延伸阅读(自行联想)

- **MAD competition** [35]: Simoncelli 提出 max differentiation 作为 model comparison 方法论,与 DISTS 评测精神一致。https://www.cns.nyu.edu/~wangZ/MAD/
- **EigenDistortions** [36]: Berardino et al. 用 Fisher information 在 hierarchical representation 上找 perceptual 最重要的 distortion direction。这给"DISTS 应该测什么"提供理论支持。https://arxiv.org/abs/1710.01113
- **NLPD (Normalized Laplacian Pyramid Distance)** [39]: Laparra et al. 的 divisive normalization pyramid,与 DISTS 的 multi-scale 哲学相通,但用 fixed wavelet 而非 learned VGG。https://arxiv.org/abs/1701.05595
- **DeepIQA / WaDIQaM**: Bosse et al. 把 quality map 用 CNN 学出来。https://arxiv.org/abs/1707.09934
- **PieAPP**: Prashnani et al. pairwise preference learning,与 LPIPS 互补。https://arxiv.org/abs/1806.02067
- **End-to-end optimized image compression** [47]: Balle, Laparra, Simoncelli。DISTS 可直接用作这类 codec 的 loss。https://arxiv.org/abs/1608.05159
- **Style transfer 域**: Gatys et al. https://arxiv.org/abs/1508.06576
- **JPEG AI / NNVC**: MPEG 的 neural compression standard,正在用 DISTS 之类 metric 评测。https://jpeg.ai/

---

## 8. 公式速查卡(完整变量定义)

| 公式 | 含义 | 关键变量 |
|---|---|---|
| Eq.1 | ℓ₂ pooling | $x$ feature map, $g$ Hanning filter, $\odot$ pointwise |
| Eq.2 | f(x) 表示 | $i$ layer idx, $j$ channel idx, $n_i$ #channels, $m=5$ |
| Eq.4 | texture synthesis | $\mu$ spatial mean, 1475 维 |
| Eq.5 | luminance sim | $c_1=10^{-6}$ 稳定项 |
| Eq.6 | structure sim | $\sigma^2$ variance, $\sigma_{xy}$ covariance, $c_2=10^{-6}$ |
| Eq.7 | DISTS distance | $\alpha_{ij}, \beta_{ij} \geq 0$, 归一化和=1 |
| Eq.8 | proper metric | $d=\sqrt{D}$ |
| Eq.9-10 | metric 分解 | $d_{ij}$ = per-channel metric (Brunet 2011) |
| Eq.14 | quality loss | $q(y)$ KADID-10k MOS |
| Eq.15 | texture invariance | $z_1, z_2$ from DTD same image |
| Eq.16 | total loss | $\lambda=1$ trade-off |
| Eq.17 | logistic fit | $\eta_{1..4}$ for PLCC |
| Eq.18 | reciprocal rank fusion | $\gamma$ outlier control |
| Eq.19 | mAP | $P(k)$ precision, $\text{rel}(k)$ relevance |

---

## 9. 最终 takeaway

DISTS 的本质是把 **SSIM 的"luminance + structure"分解** 从 pixel space lift 到 pretrained VGG feature space,并改用 global statistics 而非 local window。这个改动看似简单,但抓住了一个 fundamental insight:**texture perception is statistics-based, structure perception is correlation-based**。通过在 KADID-10k 上学 human quality rating + 在 DTD 上学 texture invariance 的 dual objective,DISTS 在 standard IQA 库 competitive、texture 库 SOTA、geometric invariance SOTA、texture retrieval SOTA,且是 proper metric + injective + differentiable。

对于做 image restoration、neural compression、generative model 评估的人来说,DISTS 是比 LPIPS 更"texture-friendly"的选择。代码: https://github.com/dingkeyan93/DISTS

注:LPIPS 在 neural compression 社区流行的原因是 early adoption;DISTS 在 texture-related 任务(尤其是 synthesizer / texture generation)上明显更合适,而且 perceptual optimization 时 unique minimum 保证不会 drift。
