---
source_pdf: Back to Basics Let Denoising Generative Models Denoise.pdf
paper_sha256: 8c265e4adfd488ccbce035a53ad9055edddd02f5c0393738f54da583ee832b8f
processed_at: '2026-08-18T02:11:58-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说这篇paper

---

## 一句话版本

今天的diffusion model其实不"denoise"——它们predict noise，不predict clean image。这篇paper说：**直接predict clean image，用最简单的plain ViT on pixels，就能work得很好**，而且在高分辨率高维度情况下，predict noise会惨败。

---

## 问题是什么

### Diffusion model的"denoising"名不副实

Diffusion model叫"denoising diffusion"，听起来像是"从noisy image恢复clean image"。但实际操作中，**network预测的不是clean image，而是noise**。

为什么会这样？历史原因。DDPM [[23]](https://arxiv.org/abs/2006.11239) 2020年发现predict noise (ε-prediction) 比predict clean image (x-prediction) 效果好得多。于是整个community就接受了ε-prediction作为default，后来又发展出v-prediction [[52]](https://arxiv.org/abs/2202.00512)。

大家都觉得这三种prediction target是"等价的"——数学上确实可以通过reweighting互相转换。**但等价性有个前提：network有足够capacity。**

### 什么时候capacity不够

用latent diffusion (LDM) [[49]](https://arxiv.org/abs/2112.10752) 的时候，input维度很小（比如32×32×4 = 4096维），network hidden dimension (通常768+) 远大于input，capacity充足，ε/v-prediction都work。

但如果你直接在pixel space用plain ViT，patch size设大一点（比如16×16×3 = 768维），情况就不同了。Network的hidden dimension可能只有768，**和patch dimension一样大甚至更小**。这时候predict noise就出问题了。

---

## 核心insight：Manifold Assumption

### 什么是manifold assumption

[[4]](https://mitpress.mit.edu/9780262033589/semi-supervised-learning/) Natural image虽然活在high-dimensional pixel space里（比如256×256×3 = 196608维），但实际只占据一个low-dimensional manifold。你可以想象成：所有可能的natural image只是这个196608维空间里一张很薄的"曲面"上的点。

这个"曲面"的维度（intrinsic dimension）可能只有几千维，远小于196608。

### 关键不对称性

这就造成了一个关键不对称：

- **Clean image x**: on-manifold，实际只需要low-dim information就能描述
- **Noise ε**: off-manifold，isotropic Gaussian均匀分布在full high-dim space里
- **Velocity v = x - ε**: 也off-manifold，因为ε的high-dim贡献dominate

用通俗的话说：**predict clean image只需要找到manifold上的一个点（容易），predict noise需要描述整个high-dim空间里的一个点（难）。**

### Toy experiment的intuition

[[Figure 2]](https://arxiv.org/abs/2509.20894) 的toy experiment把这个intuition展示得很清楚。

假设真实的underlying data是2维的（d=2），但被random projection matrix"埋"在D维空间里。Network是一个256维hidden的MLP。

当D=2时，三种prediction都work（network capacity远超target dimension）。

当D=512时：
- **x-prediction**: 仍然work。因为真正的target是2维manifold上的点，256维hidden >> 2维target，sufficient。
- **ε-prediction / v-prediction**: 惨败。因为target是512维的noise/velocity，256维hidden < 512维target，information loss inevitable。

**这就是paper的核心insight：network capacity不够的时候，predict clean data容易，predict noise难。**

---

## 实验证据

### Table 2: 9种combination的系统对比

[[Table 2]](https://arxiv.org/abs/2509.20894) 是最关键的实验。作者systematically地enumerate了所有(prediction space × loss space) = 3×3 = 9种combination。

**ImageNet 256×256, JiT-B/16, patch dim = 768 = hidden dim:**

| | x-pred | ε-pred | v-pred |
|---|---|---|---|
| x-loss | 10.14 | 379.21 | 107.55 |
| ε-loss | 10.45 | 394.58 | 126.88 |
| v-loss | **8.62** | 372.38 | 96.53 |

**x-prediction在所有loss space下都work（FID 8-10），ε/v-prediction在所有loss space下都惨败（FID 96-394）。**

对比 **ImageNet 64×64, JiT-B/4, patch dim = 48 << hidden dim = 768:**

| | x-pred | ε-pred | v-pred |
|---|---|---|---|
| x-loss | 5.76 | 6.20 | 6.12 |
| ε-loss | 3.56 | 4.02 | 3.76 |
| v-loss | 3.55 | 3.63 | 3.46 |

**所有combination都work**，因为patch dimension太小，network capacity充足。

**Critical insight**: 这解释了为什么之前LDM的工作没有expose这个问题——它们的input dimensionality太小了，capacity bottleneck没有暴露。

### Loss weighting不是sufficient的

有人说"三种prediction可以通过reweighting互相转换"。Paper回应：**reweighting解决不了capacity问题**。

从Table 2(a)看：
- ε/v-prediction在所有loss space（即所有weighting scheme）下都fail
- x-prediction在所有loss space下都work

**Loss weighting影响training dynamics，但无法弥补network capacity不足。** 这就像给一个太小的人穿多大的衣服都不合适——问题在人太小，不在衣服。

### Noise level shift也不是sufficient的

[[Table 3]](https://arxiv.org/abs/2509.20894) 调整noise level (通过logit-normal distribution的μ参数):

| μ | x-pred | ε-pred | v-pred |
|---|---|---|---|
| 0.0 | 14.44 | 464.25 | 120.03 |
| -0.4 | 9.79 | 372.91 | 109.93 |
| **-0.8** | **8.62** | 372.36 | 96.53 |
| -1.2 | 8.99 | 355.25 | 106.85 |

适当高噪声对x-prediction有益，但**无法挽救ε/v-prediction的fundamental capacity issue**。

### Bottleneck反而有益

[[Figure 4]](https://arxiv.org/abs/2509.20894) 最counterintuitive：把768-dim patch通过一个更小的bottleneck（比如32-dim）再expand回768-dim hidden，FID反而更好了！

**Intuition**: 这直接验证了manifold assumption。Natural image patch的intrinsic dimension远小于768。Bottleneck forces network学习low-dim representation，这恰好是manifold learning的目标。

这跟classical autoencoder [[68]](https://icml2008.cs.helsinki.fi/papers/592.pdf) 的philosophy完全一致：bottleneck structure encourages learning low-dimensional manifolds。

### Table 5: 分辨率scalability

| Resolution | Model | Patch dim | Params | FID |
|---|---|---|---|---|
| 256×256 | JiT-B/16 | 768 | 131M | 4.37 |
| 512×512 | JiT-B/32 | 3072 | 133M | 4.64 |
| 1024×1024 | JiT-B/64 | 12288 | 141M | 4.82 |

**Patch dim从768增加到12288（16倍），FID只worsen从4.37到4.82。** Network hidden还是768，远小于patch dim，但仍然work。

这直接反证了"network width必须match patch dim"的假设。**因为manifold的intrinsic dimension不随resolution线性scale——1024×1024 image的intrinsic dim可能只比256×256多几倍。**

---

## 为什么JiT这么简单

### Architecture: "Just Image Transformer"

[[Figure 3]](https://arxiv.org/abs/2509.20894) JiT的architecture就是standard ViT [[13]](https://arxiv.org/abs/2010.11929) 直接applied to patches of pixels:

1. **Patchify**: 把image分成non-overlapping patches (e.g., 16×16×3 = 768-dim per patch)
2. **Linear embedding**: patch dim → hidden dim
3. **Positional embedding** + **Transformer blocks** (with adaLN-Zero for conditioning)
4. **Linear predictor**: hidden dim → patch dim

**就这么简单。没有tokenizer，没有pre-training，没有extra loss，没有domain-specific architecture。**

### 最终算法

公式（Eq. 6）:

$$\mathcal{L} = \mathbb{E}_{t, \mathbf{x}, \boldsymbol{\epsilon}} \|\mathbf{v}_\theta(\mathbf{z}_t, t) - \mathbf{v}\|^2$$

$$\text{where } \mathbf{v}_\theta(\mathbf{z}_t, t) = \frac{\text{net}_\theta(\mathbf{z}_t, t) - \mathbf{z}_t}{1 - t}$$

变量解释:
- $\mathbf{z}_t = t\mathbf{x} + (1-t)\boldsymbol{\epsilon}$: noised sample at time $t$，$t$从0（pure noise）到1（pure data）
- $\mathbf{x}$: clean data
- $\boldsymbol{\epsilon}$: Gaussian noise
- $\mathbf{v} = \mathbf{x} - \boldsymbol{\epsilon}$: target velocity (flow direction)
- $\text{net}_\theta(\mathbf{z}_t, t)$: network直接output clean image prediction $\mathbf{x}_\theta$
- $\mathbf{v}_\theta$: 通过公式从$\mathbf{x}_\theta$计算出的predicted velocity
- $1/(1-t)$: reweighting factor，在$t \to 1$（low noise）时emphasize

**Network直接predict clean image (x-prediction)，但loss在velocity space计算 (v-loss)。** 这结合了x-prediction的capacity advantage和v-loss的better weighting。

Training step伪代码:
```python
t = sample_t()                    # logit-normal sampling
epsilon = randn_like(x)           # Gaussian noise
z = t * x + (1 - t) * epsilon     # Flow interpolation
v = (x - z) / (1 - t)             # Target velocity
x_pred = net(z, t)                # Network predicts x (x-prediction!)
v_pred = (x_pred - z) / (1 - t)   # Transform to v-space
loss = l2_loss(v - v_pred)        # v-loss
```

### "Just Advanced" Transformer improvements

[[Table 4]](https://arxiv.org/abs/2509.20894) 还show了incorporate LLM community的advances能进一步提升:

| Component | Source | Effect on FID |
|-----------|--------|---------------|
| Baseline (SwiGLU, RMSNorm) | [[54]](https://arxiv.org/abs/2002.05202) [[75]](https://arxiv.org/abs/1910.07467) | 7.48 |
| + RoPE, qk-norm | [[62]](https://arxiv.org/abs/2104.09864) [[19]](https://arxiv.org/abs/2010.04245) | 6.69 |
| + in-context class tokens | [[35]](https://arxiv.org/abs/2406.11838) | 5.49 |

**Plain Transformer的好处: 因为架构和task解耦，能直接benefit from LLM community的architectural advances。**

---

## 对比和意义

### 对比latent diffusion

LDM通过两阶段approach:
1. Train VAE压缩pixel → latent (e.g., 256×256×3 → 32×32×4 = 4096-dim)
2. Diffusion in latent space

Latent space维度小，ε/v-prediction能work。但:
- **依赖pre-trained tokenizer**: VAE需要adversarial + perceptual loss training
- **Lossy compression**: VAE重建不是perfect

JiT直接在pixel space做x-prediction，self-contained，no information loss。

### 对比其他pixel-space diffusion

[[Table 7]](https://arxiv.org/abs/2509.20894):

| Method | Architecture | FID@256 | GFLOPs |
|---|---|---|---|
| ADM-G [[12]](https://arxiv.org/abs/2105.05233) | U-Net | 4.59 | 1120 |
| SiD2 [[26]](https://arxiv.org/abs/2412.12406) | UViT/1 | 1.38 | 653 |
| PixelFlow [[6]](https://arxiv.org/abs/2504.07963) | XL/4 | 1.98 | 2909 |
| PixNerd [[70]](https://arxiv.org/abs/2507.23268) | XL/16 + DINOv2 | 2.15 | 134 |
| **JiT-G/16** | **Plain ViT** | **1.82** | **383** |

**JiT用最少的compute (383 GFLOPs) 达到了competitive的结果 (1.82 FID)，而且no pre-training, no extra loss。**

### 为什么这很重要

1. **Self-contained**: 不依赖任何external component，容易reproduce，容易adapt到新domain
2. **Generalizable**: 对proteins、molecules、weather等domain-specific tokenizer难设计的情况特别有价值
3. **Conceptually clean**: 回归到Denoising Autoencoder [[68]](https://icml2008.cs.helsinki.fi/papers/592.pdf) 的原始philosophy——predict clean data for manifold learning

### "Obvious in hindsight" quality

最好的paper往往有"obvious in hindsight" quality。JiT的核心insight——predict clean data, not noise——在retrospect是obvious的。这正是Denoising Autoencoder [[68]](https://icml2008.cs.helsinki.fi/papers/592.pdf) 的原始idea。但community被DDPM的empirical success seduced，忘了这个principle。

Kaiming He的style一贯如此：ResNet (identity mappings)、MoCo (momentum encoder)、MAE (masking) 都是"obvious in hindsight"的insights，executed with rigor。

---

## 总结

这篇paper的message可以用一句话概括：

**Under the manifold assumption, predicting clean data (on-manifold) is fundamentally easier than predicting noise (off-manifold), especially when network capacity is limited relative to observed dimensionality.**

Practically, 这让一个极简的架构（plain ViT on pixels, no tokenizer, no pre-training, no extra loss）能work得很好。Conceptually, 它reconnects modern diffusion models to classical manifold learning and denoising autoencoders。

**Reference**: [Back to Basics: Let Denoising Generative Models Denoise (arXiv:2509.20894)](https://arxiv.org/abs/2509.20894)

---

# Back to Basics: Let Denoising Generative Models Denoise — 深度解读

这篇来自Tianhong Li和Kaiming He (MIT)的paper，表面上是一个简单的"回归x-prediction"的主张，实际上触及了modern diffusion models一个被长期忽视的根本性问题：**neural network capacity与prediction target的dimensionality之间的mismatch**。让我从first principles出发，帮你build intuition。

---

## 1. 核心论点：Manifold Assumption下的prediction target asymmetry

### 1.1 Manifold Assumption的数学表述

Manifold assumption [[4]](https://mitpress.mit.edu/9780262033589/semi-supervised-learning/) 假设natural data $\mathbf{x} \in \mathbb{R}^D$ 实际lie在一个低维manifold $\mathcal{M}^d$ 上，其中 $d \ll D$。对于ImageNet 256×256×3，$D = 196608$，但intrinsic dimension $d$ 可能只有几千维。

关键insight：**clean data $\mathbf{x}$ 是on-manifold的，而noise $\boldsymbol{\epsilon}$ 和velocity $\mathbf{v} = \mathbf{x} - \boldsymbol{\epsilon}$ 是off-manifold的**。

- $\mathbf{x} \in \mathcal{M}^d$: 受manifold结构约束，effective dimensionality = $d$
- $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$: isotropic Gaussian，均匀分布在$\mathbb{R}^D$上，effective dimensionality = $D$
- $\mathbf{v} = \mathbf{x} - \boldsymbol{\epsilon}$: $\mathbf{x}$的low-dim贡献被$\boldsymbol{\epsilon}$的high-dim贡献dominate，effective dimensionality ≈ $D$

### 1.2 为什么这导致prediction target不对称

考虑一个hidden dimension为 $h$ 的network。Information theory视角：

**x-prediction**: 网络需要output一个on-manifold的点。即使 $h < D$，只要 $h \geq d$，网络就能encoding足够信息来reconstruct manifold上的点。Network像一个**projector**，将high-dim input project到low-dim manifold。

**ε-prediction / v-prediction**: 网络需要output一个off-manifold的full-dimensional quantity。如果 $h < D$，network的bottleneck会丢失部分noise information，导致prediction error。Network像一个**identity function with denoising**，必须保留所有information。

这就是为什么作者说："a limited-capacity network can still predict the clean data, as it only needs to retain the low-dimensional information while filtering out the noise."

### 1.3 Toy Experiment的深层intuition

[[Figure 2]](https://arxiv.org/abs/2509.20894) 的toy experiment非常enlightening。设underlying data $\hat{\mathbf{x}} \in \mathbb{R}^d$ (d=2)，通过random column-orthogonal matrix $P \in \mathbb{R}^{D \times d}$ (满足 $P^\top P = I_{d \times d}$) 嵌入到 $D$-dim space: $\mathbf{x} = P\hat{\mathbf{x}}$。

网络是5-layer ReLU MLP with 256-dim hidden。当 $D = 512$:
- **x-prediction**: 网络只需要学习 $P \hat{\mathbf{x}}$ 的2-dim structure。256-dim hidden >> 2-dim target，sufficient。
- **ε-prediction**: 网络需要output 512-dim noise。256-dim hidden < 512-dim target，information loss inevitable。
- **v-prediction**: 同理，fail。

这个experiment的精妙之处在于：**$P$ 是unknown to the model**，所以model必须从data中infer manifold structure。对于x-prediction，这正好是manifold learning的目标；对于ε-prediction，model被forced去preserve它不需要的information。

---

## 2. Diffusion Formulation的完整推导

### 2.1 Flow Matching的起点

采用flow matching [[37]](https://arxiv.org/abs/2210.02747) [[38]](https://arxiv.org/abs/2209.03003) [[1]](https://arxiv.org/abs/2209.15571)的linear schedule:

$$\mathbf{z}_t = t\mathbf{x} + (1-t)\boldsymbol{\epsilon} \quad (1)$$

变量含义:
- $\mathbf{z}_t$: noised sample at time $t$
- $t \in [0, 1]$: time parameter
- $\mathbf{x}$: clean data
- $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$: Gaussian noise
- $a_t = t, b_t = 1-t$: noise schedule coefficients

Boundary conditions:
- $t = 1$: $\mathbf{z}_1 = \mathbf{x}$ (pure data)
- $t = 0$: $\mathbf{z}_0 = \boldsymbol{\epsilon}$ (pure noise)

### 2.2 Flow velocity的定义

$$\mathbf{v} = \frac{d\mathbf{z}_t}{dt} = a_t'\mathbf{x} + b_t'\boldsymbol{\epsilon} = \mathbf{x} - \boldsymbol{\epsilon} \quad (2)$$

这里 $a_t' = \frac{da_t}{dt} = 1$, $b_t' = \frac{db_t}{dt} = -1$。

### 2.3 v-loss的标准formulation

$$\mathcal{L} = \mathbb{E}_{t, \mathbf{x}, \boldsymbol{\epsilon}} \|\mathbf{v}_\theta(\mathbf{z}_t, t) - \mathbf{v}\|^2 \quad (3)$$

变量:
- $\mathbf{v}_\theta$: network parameterized by $\theta$
- $\mathbf{v} = \mathbf{x} - \boldsymbol{\epsilon}$: target velocity
- $\|\cdot\|^2$: squared L2 norm

### 2.4 Sampling via ODE

Generation通过solving ODE [[37]](https://arxiv.org/abs/2210.02747):

$$\frac{d\mathbf{z}_t}{dt} = \mathbf{v}_\theta(\mathbf{z}_t, t) \quad (4)$$

从 $\mathbf{z}_0 \sim p_{\text{noise}}$ 开始，积分到 $t=1$。Default solver: 50-step Heun [[20]](https://de.wikipedia.org/wiki/Heun-Verfahren)。

### 2.5 九种组合的systematic enumeration

[[Table 1]](https://arxiv.org/abs/2509.20894) 最关键的贡献是systematic地enumerate了所有9种(prediction space × loss space)组合。给定三个unknowns $\{\mathbf{x}, \boldsymbol{\epsilon}, \mathbf{v}\}$ 和一个network output，需要两个constraints:

**Constraints:**
1. Flow definition: $\mathbf{z}_t = t\mathbf{x} + (1-t)\boldsymbol{\epsilon}$
2. Velocity definition: $\mathbf{v} = \mathbf{x} - \boldsymbol{\epsilon}$

**x-prediction的equation system (Eq. 5):**
$$\begin{cases}
\mathbf{x}_\theta = \text{net}_\theta \\
\mathbf{z}_t = t\mathbf{x}_\theta + (1-t)\boldsymbol{\epsilon}_\theta \\
\mathbf{v}_\theta = \mathbf{x}_\theta - \boldsymbol{\epsilon}_\theta
\end{cases}$$

Solving:
- $\boldsymbol{\epsilon}_\theta = \frac{\mathbf{z}_t - t\mathbf{x}_\theta}{1-t}$
- $\mathbf{v}_\theta = \frac{\mathbf{x}_\theta - \mathbf{z}_t}{1-t}$

**Key insight**: 9种combination都是legitimate generators，因为都可以transform到v-space来solve ODE。但它们的training dynamics和capacity requirements不同。

### 2.6 Loss reweighting的mathematics

考虑x-prediction + v-loss的组合 (Table 1, row 3, col a):

$$\mathcal{L} = \mathbb{E}\|\mathbf{v}_\theta - \mathbf{v}\|^2 = \mathbb{E}\left\|\frac{\mathbf{x}_\theta - \mathbf{z}_t}{1-t} - \frac{\mathbf{x} - \mathbf{z}_t}{1-t}\right\|^2 = \mathbb{E}\left[\frac{1}{(1-t)^2}\|\mathbf{x}_\theta - \mathbf{x}\|^2\right]$$

这是一个reweighted x-loss，weight $\frac{1}{(1-t)^2}$ 在 $t \to 1$ 时blow up，强调low-noise (接近clean) region。

---

## 3. JiT Architecture的detailed解析

### 3.1 "Just Image Transformer"的minimalist design

[[Figure 3]](https://arxiv.org/abs/2509.20894) 展示了JiT的architecture，本质上是standard ViT [[13]](https://arxiv.org/abs/2010.11929) 直接applied to patches of pixels:

**Pipeline:**
1. **Patchify**: 将 $H \times W \times 3$ image分成non-overlapping patches of size $p \times p$
   - JiT/16 at 256: 256 patches, each 768-dim (16×16×3)
   - JiT/32 at 512: 256 patches, each 3072-dim (32×32×3)
   - JiT/64 at 1024: 256 patches, each 12288-dim (64×64×3)

2. **Linear embedding** (optional bottleneck): patch dim → hidden dim
3. **Positional embedding**: 加上spatial position information
4. **Transformer blocks**: standard ViT blocks with adaLN-Zero conditioning [[46]](https://arxiv.org/abs/2212.09748)
5. **Linear predictor**: hidden dim → patch dim

### 3.2 Bottleneck embedding的surprising benefit

[[Figure 4]](https://arxiv.org/abs/2509.20894) 是paper最counterintuitive的result之一。将768-dim patch通过bottleneck $d'$:

$$\text{patch} \xrightarrow{W_1 \in \mathbb{R}^{d' \times 768}} d'\text{-dim} \xrightarrow{W_2 \in \mathbb{R}^{768 \times d'}} 768\text{-dim hidden}$$

Results:
- $d' = 768$ (no bottleneck): baseline
- $d' = 16$: FID still reasonable!
- $d' = 32 \sim 512$: FID improved by ~1.3

**Intuition**: 这直接验证了manifold assumption。Natural image patches的intrinsic dimension远小于768。Bottleneck forces network学习low-dim representation，这恰好是manifold learning的目标 [[64]](https://arxiv.org/abs/physics/0004057) [[48]](https://www.cs.toronto.edu/~hinton/absps/contractive-autoencoder.pdf)。

这与classical autoencoders [[68]](https://icml2008.cs.helsinki.fi/papers/592.pdf) [[69]](https://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf) 的philosophy完全一致：bottleneck structure encourages learning low-dimensional manifolds。

### 3.3 "Just Advanced" Transformer improvements

[[Table 4]](https://arxiv.org/abs/2509.20894) 展示了incorporating LLM community的advances:

| Component | Source | Effect on JiT-B/16 FID |
|-----------|--------|----------------------|
| Baseline (SwiGLU, RMSNorm) | [[54]](https://arxiv.org/abs/2002.05202) [[75]](https://arxiv.org/abs/1910.07467) | 7.48 |
| + RoPE, qk-norm | [[62]](https://arxiv.org/abs/2104.09864) [[19]](https://arxiv.org/abs/2010.04245) | 6.69 |
| + in-context class tokens | [[35]](https://arxiv.org/abs/2406.11838) | 5.49 |

**Key design choice**: In-context class tokens prepended at later blocks (not input). For JiT-B: start at block 4; JiT-L: block 8; JiT-H/G: block 10。32个repeated class tokens with different positional embeddings。

这个设计的intuition: early blocks专注于spatial feature extraction，class information在later blocks inject更effective。

---

## 4. Experimental Results的deep analysis

### 4.1 Table 2: 9种combination的对比

**Table 2(a): ImageNet 256×256, JiT-B/16, patch dim = 768 = hidden dim**

| | x-pred | ε-pred | v-pred |
|---|---|---|---|
| x-loss | 10.14 | **379.21** | **107.55** |
| ε-loss | 10.45 | **394.58** | **126.88** |
| v-loss | **8.62** | **372.38** | **96.53** |

**Observations:**
1. x-prediction works across all loss spaces (FID 8-10)
2. ε-prediction catastrophically fails (FID ~370-395)
3. v-prediction catastrophically fails (FID ~96-127)
4. v-loss is preferable for x-prediction (8.62 < 10.14 < 10.45)

**Table 2(b): ImageNet 64×64, JiT-B/4, patch dim = 48 << hidden dim = 768**

| | x-pred | ε-pred | v-pred |
|---|---|---|---|
| x-loss | 5.76 | 6.20 | 6.12 |
| ε-loss | 3.56 | 4.02 | 3.76 |
| v-loss | 3.55 | 3.63 | 3.46 |

**All combinations work reasonably** because patch dim (48) << hidden dim (768)，网络有ample capacity保留noise information。

**Critical insight**: 这解释了为什么之前LDM (latent dim ~4-8) 的工作没有expose这个问题——它们的input dimensionality太小了。

### 4.2 Figure 7: Training loss的quantitative evidence

[[Figure 7]](https://arxiv.org/abs/2509.20894) top显示，在相同v-loss下:
- x-prediction: ~25% lower training loss than v-prediction
- ε-prediction: ~3× higher loss, unstable

这直接证明x-prediction是inherently easier task，因为data on low-dim manifold。

Bottom的denoised images可视化显示v-prediction有noticeable artifacts，而x-prediction干净。Single-step error在multi-step ODE solver中accumulate，导致catastrophic failure。

### 4.3 Table 3: Noise-level shift的analysis

通过调整logit-normal distribution [[15]](https://arxiv.org/abs/2403.03206)的$\mu$来shift noise level:

$$\text{logit}(t) \sim \mathcal{N}(\mu, \sigma^2)$$

- $\mu = 0$: balanced
- $\mu = -0.8$ (default): higher noise, smaller $t$

Results:
| $\mu$ | x-pred | ε-pred | v-pred |
|---|---|---|---|
| 0.0 | 14.44 | 464.25 | 120.03 |
| -0.4 | 9.79 | 372.91 | 109.93 |
| **-0.8** | **8.62** | 372.36 | 96.53 |
| -1.2 | 8.99 | 355.25 | 106.85 |

**Intuition**: 适当高噪声对x-prediction有益（更多training signal从high-noise region），但**无法挽救ε/v-prediction的fundamental capacity issue**。

### 4.4 Table 5: Cross-resolution scalability

| Resolution | Model | Patch dim | Params | GFLOPs | FID |
|---|---|---|---|---|---|
| 256×256 | JiT-B/16 | 768 | 131M | 25 | 4.37 |
| 512×512 | JiT-B/32 | 3072 | 133M | 26 | 4.64 |
| 1024×1024 | JiT-B/64 | 12288 | 141M | 30 | 4.82 |

**Astonishing**: patch dim从768增加到12288 (16×)，FID只worsen从4.37到4.82。这直接反证了"network width must match patch dim"的assumption。

**Mechanism**: x-prediction只需要output on-manifold的点。Manifold的intrinsic dimension不随resolution linearly scale——1024×1024 image的intrinsic dim可能只比256×256多几倍，远小于16×的pixel increase。

### 4.5 Table 6: Model scaling

| Model | 256 (200ep) | 256 (600ep) | 512 (200ep) | 512 (600ep) |
|---|---|---|---|---|
| JiT-B | 4.37 | 3.66 | 4.64 | 4.02 |
| JiT-L | 2.79 | 2.36 | 3.06 | 2.53 |
| JiT-H | 2.29 | 1.86 | 2.51 | 1.94 |
| JiT-G | 2.15 | 1.82 | 2.11 | **1.78** |

**JiT-G at 512 (1.78) < JiT-G at 256 (1.82)**! 这是counterintuitive的——通常更高resolution更难。

**Explanation**: For very large models on ImageNet, FID largely depends on overfitting. 512 resolution poses harder task → less overfitting → better FID。这暗示ImageNet的"天花板"可能在256 resolution已经hit了。

### 4.6 Table 12: Cross-resolution generation

| Model | FID@256 | FID@512 |
|---|---|---|
| JiT-G/16@256 | 1.82 | 2.45 (↑512) |
| JiT-G/32@512 | 1.84 (↓256) | 1.78 |

**Insight**: 512 model downsample到256几乎无损 (1.84 vs 1.82)，但256 model upsample到512明显差 (2.45 vs 1.78)。High-res model包含更多information，low-res是lossy projection。

---

## 5. Pre-conditioner的critical analysis (Appendix B.2)

### 5.1 EDM pre-conditioner的formulation

EDM [[29]](https://arxiv.org/abs/2206.00364) 的pre-conditioner:

$$\mathbf{x}_\theta(\mathbf{z}_t, t) = c_{\text{skip}} \cdot \mathbf{z}_t + c_{\text{out}} \cdot \text{net}_\theta(\mathbf{z}_t, t) \quad (7)$$

变量:
- $c_{\text{skip}}$: skip connection coefficient (从input直接passthrough)
- $c_{\text{out}}$: output scaling coefficient
- $\text{net}_\theta$: raw network output

**Critical observation**: 除非 $c_{\text{skip}} \equiv 0$，否则network的direct output不是pure $\mathbf{x}_\theta$。Network实际上predict的是 $\mathbf{x}_\theta - c_{\text{skip}}\mathbf{z}_t$，这是a mix of data and noise。

### 5.2 EDM-style coefficients的conversion

由于EDM用variance-exploding schedule ($\mathbf{z}_t = \mathbf{x} + \sigma_t \boldsymbol{\epsilon}$)，而JiT用variance-preserving ($\mathbf{z}_t = t\mathbf{x} + (1-t)\boldsymbol{\epsilon}$)，需要conversion:

$$\sigma_t = \frac{1-t}{t}$$

EDM coefficients:
- $c_{\text{skip}} = \frac{1}{t} \cdot \frac{\sigma_{\text{data}}^2}{\sigma_{\text{data}}^2 + \sigma_t^2}$
- $c_{\text{out}} = \frac{\sigma_{\text{data}} \sigma_t}{\sqrt{\sigma_{\text{data}}^2 + \sigma_t^2}}$

其中 $\sigma_{\text{data}} = 0.5$ [[29]](https://arxiv.org/abs/2206.00364)。

**Asymptotic behavior**: 当 $t \to 0$:
- $\sigma_t \to +\infty$
- $c_{\text{skip}} \to 0$
- $c_{\text{out}} \to 1$
- 接近pure x-prediction

### 5.3 Table 10: Pre-conditioner results

| | x-pred ($c_{\text{skip}}=0$) | EDM-style | Linear ($c_{\text{skip}}=t$) |
|---|---|---|---|
| x-loss | 10.14 | 28.94 | 39.50 |
| ε-loss | 10.45 | 72.05 | 67.56 |
| v-loss | 8.62 | 35.49 | 46.25 |

**Key findings:**
1. Pure x-prediction ($c_{\text{skip}}=0$) 最好
2. Pre-conditioned versions都fail (FID 28-72)
3. Pre-conditioned比bare ε/v-prediction好 (因为接近x-prediction when $t \to 0$)

**Intuition**: Pre-conditioner在 $t \to 0$ (high noise) region接近x-prediction，但在 $t \to 1$ (low noise) region偏离。Low noise region对final image quality至关重要，deviation here causes failure。

---

## 6. 与相关工作的contextualization

### 6.1 DDPM的历史accident

DDPM [[23]](https://arxiv.org/abs/2006.11239) 原本code [[24]](https://github.com/hojonathanho/diffusion) 中支持x-prediction，但empirically发现ε-prediction更好。这established了ε-prediction的dominance。

**Why DDPM's observation might be misleading:**
1. DDPM实验在CIFAR-10 (32×32×3 = 3072-dim total, but with U-Net dense convolutions)
2. U-Net的dense convolutions + long skip connections implicitly avoid information bottleneck
3. Low resolution + dense architecture → capacity sufficient for ε-prediction

JiT的large-patch ViT explicit地creates bottleneck，expose了这个问题。

### 6.2 Latent Diffusion的"hidden" solution

LDM [[49]](https://arxiv.org/abs/2112.10752) 通过两-stage approach:
1. **Stage 1**: Train autoencoder (VAE) to compress pixel → latent (e.g., 256×256×3 → 32×32×4 = 4096-dim)
2. **Stage 2**: Diffusion in latent space

Latent space的dimensionality (4096) 远小于pixel space (196608)，所以ε/v-prediction能work。

**Problem**: 这"隐藏"了问题，依赖pre-trained tokenizer。Tokenizer本身需要adversarial loss + perceptual loss (VGG [[56]](https://arxiv.org/abs/1409.1556))，不是self-contained。

### 6.3 Pixel-space diffusion的prior work

| Method | Architecture | Key design | FID@256 |
|---|---|---|---|
| ADM-G [[12]](https://arxiv.org/abs/2105.05233) | U-Net | Dense conv, long skip | 4.59 |
| SiD2 [[26]](https://arxiv.org/abs/2412.12406) | UViT/2 | Hierarchical, small patches | 1.38 |
| PixelFlow [[6]](https://arxiv.org/abs/2504.07963) | XL/4 | Multi-scale, small patches | 1.98 |
| PixNerd [[70]](https://arxiv.org/abs/2507.23268) | XL/16 | NeRF head + DINOv2 | 2.15 |
| **JiT-G/16** | **Plain ViT** | **x-prediction only** | **1.82** |

**JiT的优势:**
- No tokenizer, no pre-training, no extra loss
- Compute-friendly: 383 GFLOPs (JiT-G) vs 653-2909 GFLOPs (others)
- General-purpose Transformer (benefit from LLM advances)

### 6.4 Connection to Denoising Autoencoders

DAE [[68]](https://icml2008.cs.helsinki.fi/papers/592.pdf) [[69]](https://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf) 的original philosophy:
- Input: corrupted data
- Output: clean data
- Objective: learn manifold structure

DAE预测clean data是natural的，因为goal是manifold learning。Score matching [[67]](https://www.mitpressjournals.org/doi/10.1162/NECO_a_00142) 的ε-prediction是indirect的——预测score function ≈ 预测noise (up to scaling)。

**JiT回归到DAE的原始philosophy**: 直接predict clean data，让network做manifold learning。

---

## 7. Algorithm的practical implementation

### 7.1 Training step (Algorithm 1)

```python
# net(z, t): JiT network
# x: training batch
t = sample_t()                    # logit-normal sampling
epsilon = randn_like(x)           # Gaussian noise
z = t * x + (1 - t) * epsilon     # Flow interpolation (Eq. 1)
v = (x - z) / (1 - t)             # Target velocity (Eq. 2)
x_pred = net(z, t)                # Network predicts x (x-prediction)
v_pred = (x_pred - z) / (1 - t)   # Transform to v-space (Table 1)
loss = l2_loss(v - v_pred)        # v-loss (Eq. 6)
```

### 7.2 Sampling step (Algorithm 2, Euler)

```python
# z: current samples at t
x_pred = net(z, t)                # Predict clean x
v_pred = (x_pred - z) / (1 - t)   # Transform to v-space
z_next = z + (t_next - t) * v_pred  # Euler step
```

**Numerical stability**: 为了防止 $1/(1-t)$ 在 $t \to 1$ 时blow up，clip denominator to 0.05。

### 7.3 Final algorithm (Eq. 6)

$$\mathcal{L} = \mathbb{E}_{t, \mathbf{x}, \boldsymbol{\epsilon}} \|\mathbf{v}_\theta(\mathbf{z}_t, t) - \mathbf{v}\|^2$$

$$\text{where } \mathbf{v}_\theta(\mathbf{z}_t, t) = \frac{\text{net}_\theta(\mathbf{z}_t, t) - \mathbf{z}_t}{1 - t}$$

**Key design choices:**
- **Prediction**: x-prediction (network outputs clean image)
- **Loss**: v-loss (reweighted x-loss with $1/(1-t)^2$)
- **Rationale**: x-prediction解决capacity issue；v-loss提供better weighting emphasizing low-noise region

---

## 8. Deep intuitions和broader implications

### 8.1 Information Bottleneck视角

JiT的success可以用information bottleneck [[64]](https://arxiv.org/abs/physics/0004057) theory理解:

$$\min_{\theta} I(\mathbf{z}_t; \text{net}_\theta(\mathbf{z}_t)) - \beta \cdot I(\text{net}_\theta(\mathbf{z}_t); \mathbf{x})$$

- 第一项: compression (network output应该compress input)
- 第二项: relevance (network output应该retain information about clean data)

**x-prediction**: 直接optimize这个objective。Network learns to compress noise (irrelevant) and retain manifold structure (relevant).

**ε-prediction**: Network forced to retain $I(\text{net}_\theta; \boldsymbol{\epsilon})$，violating compression principle.

### 8.2 为什么latent diffusion work但"cheating"

LDM的autoencoder stage implicitly learns a projection $\mathbf{z} = E(\mathbf{x})$ where $\mathbf{z}$ lives in a lower-dim space closer to manifold. 然后diffusion在 $\mathbf{z}$-space做ε-prediction，capacity sufficient。

**但这"cheating"在两方面:**
1. **依赖external tokenizer**: VAE需要adversarial + perceptual loss training
2. **Lossy compression**: VAE的reconstruction不是perfect，high-freq detail可能丢失

JiT直接在pixel space做x-prediction，self-contained，no information loss.

### 8.3 Generalization到other domains

Paper的vision是general "Diffusion + Transformer" paradigm适用于:
- **Proteins**: 高dim amino acid / 3D structure space，tokenizer难设计
- **Molecules**: SMILES tokenizer有limitation，直接在molecular graph上做diffusion
- **Weather/Climate**: High-dim grid data，physical constraints

对于这些domains，x-prediction的principle同样适用: natural data on low-dim manifold，predict clean data easier than predicting noise.

### 8.4 Connection to classical denoising

Classical denoising methods [[9]](https://ieeexplore.ieee.org/document/4289163) [[14]](https://ieeexplore.ieee.org/document/1709123) [[47]](https://ieeexplore.ieee.org/document/1263213) (BM3D, sparse coding, wavelets) 都基于:
- Natural images有sparse / low-dim representation
- Denoising = projecting noisy image onto this representation

JiT的x-prediction是neural network版的classical denoising: predict clean image directly, leveraging learned manifold structure.

### 8.5 Why "Back to Basics"

Paper title的"Back to Basics"有多层含义:
1. **Back to x-prediction**: 回归DDPM之前的原始denoising formulation
2. **Back to manifold**: 回归manifold assumption作为first principle
3. **Back to plain Transformer**: 回归ViT的minimalist design，no domain-specific architecture
4. **Back to self-contained**: 回归no pre-training, no extra loss的pure approach

### 8.6 Open questions和future directions

1. **Why does v-loss help x-prediction?** Weight $1/(1-t)^2$ emphasizing low-noise region。但optimal weighting仍是open problem [[30]](https://arxiv.org/abs/2301.13788)。

2. **Bottleneck dim的optimal choice?** Figure 4显示 $d' = 32 \sim 512$ 都好，但如何principled地选择 $d'$ relative to intrinsic manifold dimension?

3. **Interaction with CFG?** JiT用CFG [[22]](https://arxiv.org/abs/2207.12598) + CFG interval [[33]](https://arxiv.org/abs/2403.12596)。x-prediction如何影响guidance的dynamics?

4. **Extension to text-to-image?** JiT只在class-conditional ImageNet验证。Text-to-image需要text encoder + cross-attention，如何保持self-contained?

5. **Video generation?** Temporal dimension增加dimensionality，x-prediction的advantage可能更significant。

### 8.7 Critique和limitations

**Potential concerns:**
1. **ImageNet-specific?** ImageNet是single-class conditional generation。Unconditional或text-to-image setting可能different。
2. **FID作为唯一metric?** FID有known limitations [[32]](https://arxiv.org/abs/1904.06991)。Precision/recall (Table 13) 显示JiT-G的recall (0.62) 不如RAE (0.67)。
3. **Long training?** 600 epochs是significant compute。Efficiency对比应该考虑total training cost。
4. **Why v-loss > x-loss for x-prediction?** Paper没有fully解释。可能是v-loss的weighting恰好better matches manifold structure的learning dynamics。

---

## 9. Personal reflections (Karpathy视角)

作为曾经的OpenAI Director和Tesla AI Director，我会这样思考这篇paper的significance:

### 9.1 "Simpler is better"的vindication

Deep learning history反复证明: simple, principled methods往往eventually win。从AlexNet到ViT，从RNN到Transformer，complexity often comes from working around fundamental issues rather than addressing them.

JiT的minimalist design (plain ViT + x-prediction, no tokenizer, no pre-training) is in this tradition。它removes the "patches" (VAE, perceptual loss, DINOv2 alignment) that LDM community has accumulated。

### 9.2 Capacity vs. Formulation的false dichotomy

Community经常assume: "if model doesn't work, make it bigger"。Table 5反证: JiT-B at 1024×1024 (patch dim 12288 >> hidden 768) 仍work，FID 4.82。

**Right question**: 不是"how to make model bigger"，而是"what should model predict"。Formulation的选择决定capacity requirements。

### 9.3 Manifold assumption的re-emergence

Manifold assumption是deep learning的foundation，但modern diffusion community某种程度上forgot它。我们accept了ε-prediction作为"default"，没有question为什么。

JiT的contribution是re-connect diffusion models to classical manifold learning。Bottleneck的beneficial effect (Figure 4) 是这种connection的quantitative validation。

### 9.4 Self-contained AI systems的philosophy

Current trend towards increasingly complex pipelines (VAE + DINOv2 + diffusion + text encoder + ...) 创造了fragile, hard-to-reproduce systems。JiT代表counter-trend: self-contained, understandable, generalizable。

对于scientific applications (proteins, molecules, climate)，self-contained approach is not just nice-to-have but essential——domain-specific tokenizers may not exist.

### 9.5 The "obvious in hindsight" quality

最好的papers往往有"obvious in hindsight" quality。JiT的核心insight (predict clean data, not noise) 在retrospect是obvious——这正是Denoising Autoencoder的original idea。但community被DDPM的empirical success seduced，forgot this principle。

Kaiming He的style一贯如此: ResNet (identity mappings), MoCo (momentum encoder), MAE (masking) 都是"obvious in hindsight"的insights，executed with rigor。

---

## 10. References和further reading

- **Paper**: [Back to Basics: Let Denoising Generative Models Denoise](https://arxiv.org/abs/2509.20894)
- **ViT**: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- **DDPM**: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- **Flow Matching**: [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- **EDM**: [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)
- **DiT**: [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
- **LDM**: [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- **Manifold Learning book**: [Semi-Supervised Learning (Chapelle et al.)](https://mitpress.mit.edu/9780262033589/semi-supervised-learning/)
- **Denoising Autoencoders**: [Extracting and Composing Robust Features with DAE](https://icml2008.cs.helsinki.fi/papers/592.pdf)
- **Information Bottleneck**: [The Information Bottleneck Method](https://arxiv.org/abs/physics/0004057)
- **v-prediction**: [Progressive Distillation for Fast Sampling](https://arxiv.org/abs/2202.00512)
- **REPA**: [Representation Alignment for Generation](https://arxiv.org/abs/2410.06924)
- **SiD2**: [Simpler Diffusion (SiD2)](https://arxiv.org/abs/2412.12406)
- **MAR**: [Autoregressive Image Generation without Vector Quantization](https://arxiv.org/abs/2406.11838)

---

## 总结

JiT的core message可以distill为一句话: **Under the manifold assumption, predicting clean data (on-manifold) is fundamentally easier than predicting noise (off-manifold), especially when network capacity is limited relative to observed dimensionality.**

这个insight的power在于它的generality: 任何high-dimensional data with low-dim manifold structure (i.e., all natural data) 都benefit from x-prediction。Paper通过rigorous experiments (9种combination, toy experiment, bottleneck analysis, cross-resolution) 验证了这一principle。

Practically, JiT enables self-contained pixel-space diffusion with plain ViT, competitive with LDM while avoiding tokenizer dependency。Conceptually, it reconnects modern diffusion models to classical manifold learning and denoising autoencoders。

作为community，我们需要重新审视"defaults" (ε-prediction, VAE tokenizer, perceptual loss) 是否truly necessary，还是working around deeper issues。JiT提供了一个compelling alternative: 回归first principles, 让simple方法work。

Reference: [Back to Basics: Let Denoising Generative Models Denoise (arXiv:2509.20894)](https://arxiv.org/abs/2509.20894)
