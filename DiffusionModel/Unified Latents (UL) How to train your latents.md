---
source_pdf: Unified Latents (UL) How to train your latents.pdf
paper_sha256: 7d7491db68020b232398f340ac7c07bcb4299ffe985c8d23aebf0935d769bb3b
processed_at: '2026-08-12T19:34:41-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Unified Latents 用人话讲

Andrej, 我换一种方式给你讲，像咱们在 whiteboard 前面聊天那种感觉。

## 先说这个故事 about 什么

你看现在的 image generation 领域，大家都在做 latent diffusion。Stable Diffusion 是鼻祖，它的做法是：先训一个 VAE autoencoder 把 512×512 的 image 压成 64×64×4 的 latent，然后在这个 latent space 上训一个 diffusion model。

听起来合理，但有个尴尬的问题：**这个 VAE 怎么训？**

Stable Diffusion 的原始 recipe 是 kl weight 调到很小很小（1e-6 那种级别），然后加一个 GAN loss、加一个 perceptual loss、加一个 L1 loss... basically 就是一个 kitchen sink。结果就是 latent 里到底有多少 bits 的信息？谁也说不清楚。kl weight 大了 latent 就变成 garbage noise，小了就信息量太大 base model 学不动。这个 weight 是手调的，没 principle。

后来 2024-2025 这波人想了个别的办法：**直接用 DINO 的 features 当 latent**。DINO 是自监督训出来的 semantic features，已经是 "好" 的 representation 了。结果 FID 非常漂亮，但是 PSNR 很烂（<20），reconstruction 出来跟原图差很远，high-frequency 细节全丢了。为什么？因为 DINO 本来就不是为 reconstruction 训的。

所以 you see the tension：

- **信息量大的 latent** → reconstruction 好 → 但 diffusion model 学起来痛苦
- **信息量小的 latent** → diffusion model 喜欢但 → reconstruction 烂

这篇 paper 的核心 claim 是：**这个 trade-off 可以 principled 地 navigate**，用 diffusion prior 自己去 measure latent 里有多少 information。

## 思想实验：information 怎么 measure？

我们先 think hard 一下，什么叫 "latent 里有 N bits 信息"？

Information theory 告诉你：如果 latent $z$ 来自 distribution $p(z)$，那 encode 一个 sample 需要的 bits 就是 $-\log p(z)$ 的 expectation。所以如果你知道 $p(z)$ 长什么样，你就知道 bitrate。

那 $p(z)$ 怎么 estimate？**用 diffusion model 来 estimate！** 这就是 paper 的核心 insight。Diffusion model 本质上就是个 density estimator，它的 ELBO 就是 log-likelihood 的 lower bound。

So instead of 用一个 hand-coded $\mathcal{N}(0, I)$ 当 prior（像传统 VAE），let's 用一个 *learned* diffusion model 当 prior。这个 diffusion prior 的 ELBO loss 就是 latent 的 bitrate 的 tight upper bound。

很 elegant 对吧？

## 但是这里有个 tricky 的地方

如果你直接拿一个 diffusion prior 训 VAE，会遇到一个 numerical stability 问题。这个问题 LSGM (Vahdat et al., 2021) 已经踩过坑了。

传统 VAE 里 encoder 输出 $q(z|x) = \mathcal{N}(\mu(x), \sigma^2(x))$，learned mean 和 learned variance。KL term 是：

$$KL[q(z|x) \| p(z)] = \mathbb{E}_{q}\left[\log \frac{q(z|x)}{p(z)}\right]$$

这里 $p(z)$ 是 diffusion prior。展开来会包含一个 encoder entropy term $-\mathbb{E}_q[\log q(z|x)]$，这个 term 涉及 $\log \sigma(x)$，**高方差**，训练特别不稳。LSGM paper 自己都抱怨这个。

UL 的解决方案是：**不要让 encoder 输出 variance 了**。

Encoder 只输出一个 deterministic point $z_{\text{clean}}$，然后我们手动加一个 *fixed* 的高斯噪声 $\mathcal{N}(0, \sigma_0^2 I)$ 把它变成 $z_0$。这个 $\sigma_0$ 是 hyperparameter，paper 里设 $\lambda(0) = 5$ 也就是 $\sigma_0 \approx 0.08$。

数学上发生了什么？**这个 fixed noise 把 encoder 的 stochasticity 吸收进了 diffusion forward process 里**。原来 LSGM 里 encoder distribution $q(z|x) = \mathcal{N}(\mu, \sigma^2)$ 的 forward process 是在 $z$ 上加噪，现在变成：先 deterministic 得到 $z_{\text{clean}}$，再 forward 加噪到 $z_0$。这两个 process 在数学上是等价的，但实现上不再需要 encoder entropy term，loss 简化到只有两项：decoder loss + prior loss。

Table 6 ablation D 验证了这个选择：让 encoder 学习 variance 反而 gFID 更差（1.81 vs 1.54），而且训练不稳定（variance collapse to 0）。

## 再说 forward process 的关键参数

Diffusion 的 forward process 是 $z_t = \alpha_t z_{\text{clean}} + \sigma_t \epsilon$，paper 用 variance-preserving schedule，意思是 $\alpha_t^2 + \sigma_t^2 = 1$ 总是成立。

定义 log-SNR $\lambda(t) = \log(\alpha_t^2 / \sigma_t^2)$：
- $t=1$ → $\lambda = -\infty$ → 纯噪声
- $t=0$ → $\lambda = 5$ → 几乎干净（但还有一点点噪声 $\sigma_0 \approx 0.08$）
- $t=0.5$ → $\lambda$ 中间值

注意这里 $t=0$ **不是** 完全 clean！这是 paper 的关键 design choice。$t=0$ 对应 $\lambda_z(0) = 5$，意味着 latent 里还残留一点 noise。这点 noise 就是 "bitrate 的 hard cap"。

为什么要这样做？因为这个 $\sigma_0$ 决定了 encoder 能塞多少 information 进 latent。$\sigma_0$ 越小，理论上能塞的 bits 越多（noise 越小，signal-to-noise ratio 越高）。$\sigma_0 = 0.08$ 对应大约每 latent dimension $5 / \ln 2 \approx 7.2$ bits 的 cap。

Ablation B 把 $\lambda_z(0)$ 调到 10（$\sigma \approx 0.007$），结果 latent 几乎无噪声，encoder 把所有信息塞进 latent，diffusion prior 根本学不到 bitrate bound，rFID 暴跌到 28.27。这说明 noise floor 不能太低。

## Prior loss 公式逐项拆解

Eq. 3 是 prior 的核心 loss：

$$\mathcal{L}_z = \mathbb{E}_t\left[-\frac{d\lambda_z(t)}{dt}\frac{e^{\lambda_z(t)}}{2} w(\lambda_z(t)) \|z_{\text{clean}} - \hat{z}(z_t, \theta)\|^2\right] + KL[p(z_1|x) \| \mathcal{N}(0,I)]$$

逐个讲：

- $\|z_{\text{clean}} - \hat{z}(z_t, \theta)\|^2$：network 试图从 noisy $z_t$ 预测 clean $z_{\text{clean}}$，标准 MSE
- $e^{\lambda_z(t)}/2$：从 MSE 转 likelihood 的 scaling，log-SNR 越大（noise 越小）这个 coefficient 越大，因为 low-noise prediction 错了代价更大
- $-d\lambda_z(t)/dt$：把 $t$ 上的 uniform sampling 转成 $\lambda$ 上的 sampling weighting。如果 schedule 让 $\lambda$ 在某些区域变化快，这个区域就被 down-weight
- $w(\lambda_z(t))$：**weighting function**，paper 强调必须是 1（unweighted ELBO）才能保证 tight bitrate bound。这就是 prior 跟 decoder 的关键差异
- $KL[p(z_1|x) \| \mathcal{N}(0,I)]$：在 $t=1$（纯噪声）处的 KL，因为 $z_1 \approx \mathcal{N}(0,I)$，这个 term 通常很小可忽略

**为什么 prior 必须 $w=1$？** 这是 paper 的一个关键 subtle 点。如果 prior 用 reweighted ELBO（比如 $w = \text{sigmoid}(\lambda - b)$），那 high-noise levels 被打折，意味着 latent 在这些 levels 编码 information "便宜"。Encoder 会学会把 information 塞进 discounted noise levels 来逃避 prior 的 regularization。结果 bitrate bound 不再 tight，latent 实际 information 比 bound 显示的高，base model 会发现 latent 比 expected 难学。

所以 prior 必须 unweighted。Bitrate 要 faithful measure，秤不能被动手脚。

## Decoder loss 可以 reweighted

Decoder loss (Eq. 4) 是另一回事：

$$\mathcal{L}_x = \mathbb{E}_t\left[\frac{d\lambda_x(t)}{dt}\frac{e^{\lambda_x(t)}}{2} w_x(\lambda_x(t)) \|x - \hat{x}(x_t, z_0, \theta)\|^2\right]$$

这里 $w_x(\lambda_x(t)) = \text{sigmoid}(\lambda_x(t) - b)$ 是 reweighted 的，**可以打折**。

为什么 decoder 可以打折而 prior 不行？因为 decoder 不参与 bitrate 的 measure。Decoder 只负责把 $z_0$ decode 成 image，loss 权重怎么设都行，反正不 measure information content。

实际上 reweighted decoder loss 有好处：high-frequency imperceptible 细节对应 low noise levels，sigmoid weighting 把这些 down-weight，让 decoder "不用太在意" 这些细节，generation quality（FID）反而更好。这就是 Kingma & Gao (2023) 的工作里讲过的 "image quality friendly weighting"。

https://arxiv.org/abs/2303.00848

Paper 引入了第二个 hyperparameter $c_{\text{lf}}$（loss factor），就是 decoder loss 乘个 scalar：

$$\mathcal{L}_x \to c_{\text{lf}} \cdot \mathcal{L}_x$$

这等效于 down-weight KL term。在 VAE 文献里这是经典做法，防止 "posterior collapse"（decoder 太强导致 latent 不被使用）。Paper 里 $c_{\text{lf}} \in [1.3, 1.7]$ 这个窄区间就够。

## 两阶段训练的本质

Stage 1：joint train encoder + prior + decoder，prior loss 用 unweighted ELBO，得到 bitrate-faithful 的 latent。

Stage 2：冻 encoder，重训 base model with sigmoid weighting。

为什么要分两阶段？我反复读了几遍，理解是：

Stage 1 的 prior 必须用 unweighted ELBO 才能保证 bitrate bound tight。但 unweighted ELBO 对 generation 不友好，因为它平等对待 low-freq 和 high-freq 信息，而 generation 主要关心 low-freq 的 semantic content。如果直接用 stage 1 的 prior 做 sampling，FID 不行（附录 B 说只能到 4 左右）。

Stage 2 换 sigmoid weighting 重训 base model，相当于把 prior 当成 "密度测量器"（stage 1 用），把 prior 当成 "生成模型"（stage 2 用）分离开来。

这是个有点 ugly 的设计，因为理论上你希望 end-to-end。但 paper 试了 single-stage（附录 B），用 truncated logistic distribution randomize base model 的 max log-SNR，结果 FID 4，远不如 two-stage 的 1.4。所以 practical 上 two-stage 更好。

我个人觉得这暗示 stage 1 的 bitrate measurement 和 stage 2 的 generation quality 有内在 conflict。Stage 1 的 unweighted ELBO 是 measure 任务，stage 2 的 reweighted ELBO 是 generation 任务，硬要 joint 训练两者互相拉扯。可能需要更 clever 的 objective 才能 unify。

## Loss Factor 怎么调

Table 2 是 paper 最 informative 的实验。横轴是 loss factor，纵轴是各种 metric：

| LF | bits/pixel | rFID | PSNR | gFID (small) | gFID (medium) |
|----|------------|------|------|--------------|----------------|
| 1.3 | 0.035 | 0.79 | 25.7 | 1.42 | 1.37 |
| 1.5 | 0.059 | 0.47 | 27.6 | 1.54 | 1.31 |
| 1.7 | 0.083 | 0.36 | 28.9 | 1.77 | 1.38 |
| 1.9 | 0.101 | 0.31 | 29.6 | 2.02 | 1.45 |
| 2.1 | 0.116 | 0.27 | 30.1 | 2.38 | 1.58 |

人话讲：

- LF 调大 → decoder loss 权重大 → encoder 被鼓励塞更多 info 进 latent → bits/pixel 涨 → rFID 降（reconstruction 更好）→ PSNR 涨
- 但是 gFID 的变化不一样：
  - **Small base model**：LF 涨 gFID 一直变差（1.42 → 2.38）。因为 small model capacity 有限，塞太多 info 它学不动
  - **Medium base model**：LF 涨 gFID 几乎不变（1.37 → 1.58）。大 model 有 capacity 处理更多信息

**这是 paper 的核心 scaling insight**：小 model 配低 bitrate latent，大 model 配高 bitrate latent。这跟 LLM 里 chinchilla scaling 的精神类似——不同 component 之间要 balance。

但 paper 没给出 explicit scaling law，只给了一个 trend。Discussion section 里也说 "future work is to establish scaling laws for Unified Latents that predict the optimal bitrate given a training budget"。这是个 open question，也是个 great research direction。

## Latent shape 几乎无关

Table 3 的 channel ablation 和 Table 4 的 spatial ablation 都很 striking。

| # channels | rFID | gFID |
|------------|------|------|
| 4 | 7.19 | N/A |
| 8 | 1.53 | N/A |
| 16 | 0.54 | 1.76 |
| 32 | 0.42 | 1.60 |
| 64 | 0.48 | 1.77 |

Channels 从 16 到 64 变化，gFID 几乎不变（1.6-1.77）。但是 SD 的 latent shape 是 critical 设计——4 channels、8x downsampling 都是 hyper-tuned 的。UL 这里几乎无关。

为什么？因为 UL 的 bitrate 不是由 shape 控制的，是由 **noise floor $\sigma_0$ 和 loss factor** 控制的。Shape 给的是 *capacity*，noise floor 给的是 *utilization*。Encoder 会自动学会怎么利用 capacity——channel 多了每个 channel 上的有效 noise floor 会被自动 adjust。

这跟传统 VAE 完全不同。传统 VAE 里 channel 数和 spatial 大小是 *bottleneck*——你硬性限制 channel 数来限制 information flow。UL 里 channel 数只是 "管线粗细"，bitrate 是 prior measure 出来的。

对 practical 工程来说这是 huge win：你不用为了 4 channel vs 8 channel 调几个月实验。

## 跟其它工作的对比

### vs Stable Diffusion VAE

SD VAE 训练 recipe：
- VAE encoder/decoder (GAN + perceptual + L1 + tiny KL)
- 主要靠 channel bottleneck（4 channels）限制 info
- KL 到 $\mathcal{N}(0, I)$ 但权重 tiny（1e-6）
- 然后冻 VAE 训 latent diffusion

UL 的优势：
- Bitrate 是 principled 的 bound，不是 ad-hoc
- Latent shape 不敏感
- Diffusion decoder 比 GAN decoder 更强大（predict distribution）

UL 的劣势：
- Diffusion decoder 采样慢（要 diffusion sampling），GAN decoder 一次 forward pass
- 需要 distillation 才能 practical

### vs DINO latents (Show-o / MAR / RAE)

DINO-based latents 用 pre-trained DINO 当 encoder，diffusion/autoregressive model 当 base model。

DINO latent 的好处：semantic representation 已经 pre-trained 好，base model 学起来 easy，FID 漂亮。

DINO latent 的坏处：
- PSNR ≤20，high-freq 细节丢失
- Reconstruction 视觉上跟原图差很远
- DINO 是 external pre-trained，不能 end-to-end 调 latent

UL 能同时拿到 semantic + high-freq (PSNR 28-30)，且 end-to-end trainable。

https://arxiv.org/abs/2510.15301
https://arxiv.org/abs/2510.11690

### vs LSGM

LSGM (Vahdat et al., 2021) 是最接近的工作，也用 diffusion prior + VAE framework。

差异：
- LSGM encoder 输出 distribution (mean + variance)，有高方差的 entropy term
- LSGM 训练不稳，需要 special tricks
- UL 用 deterministic encoder + fixed noise，把这个 instability 完全消掉

我觉得这是 paper 最大的 methodological contribution——一个 *simplification*（不要 learned variance）反而让事情 *work better*。深度学习里反复出现的 theme。

https://arxiv.org/abs/2111.09295

### vs SiD2 (Simpler Diffusion 2)

SiD2 (Hoogeboom et al., 2024) 是 paper 里引用的 pixel-space diffusion baseline，在 ImageNet-512 上做到 1.5 FID。

UL 比 SiD2 更 efficient（同样 FLOPs 下 FID 更低），因为 latent space 降低了 computational cost。

但 SiD2 没有 VAE，没有 latent space 的 trade-off 问题。所以 UL 是 SiD2 的 latent-space 延伸，加上了 latent learning 这个 dimension。

https://arxiv.org/abs/2410.19324

## 我的几个 takeaways

### 1. "Information Bound as Regularization" 这个 idea 是 transferable

UL 的核心 idea 是：用 density estimator 的 ELBO 当 information measure，从而 regularize latent。这个 idea 在 diffusion 之外也 work。比如你可以想象用 autoregressive model 的 ELBO 当 discrete latent 的 bitrate measure。Paper 自己也在 Discussion 提了 "discrete diffusion decoder" 可能扩展到 text。

### 2. Fixed noise 是 deep learning 里的 universal pattern

让 model 学 variance 听起来更 expressive，但实践中 fixed noise 更稳定、更好。这跟 dropout、data augmentation 的 random crop 一样——stochastic 是 regularization，但 stochastic 的 *schedule* 要固定，让 model learn 来适应它，而不是 model 自己决定 stochasticity 大小。

### 3. Two-stage 训练暴露了 ELBO 的 dual role

ELBO 既是 likelihood bound（用来 measure），又是 generation objective（用来 sample）。Stage 1 用前者，Stage 2 用后者。这是 VAE 长期存在的 tension——max likelihood 跟 sample quality 不一定 align。UL 通过 two-stage 把这两个 role 分离。理论上更 elegant 的做法是 single objective 同时满足两个 role，但 paper 试了不行。

### 4. Bitrate 作为 scaling law 维度

UL 给了我们一个新的 scaling 维度：latent bitrate。原 LDM scaling law 主要看 model size × data，UL 这里加了 bitrate 作为第三个 dimension。Table 2 已经显示 small vs medium model 的 optimal bitrate 不同。预测 large model 的 optimal bitrate 还更高。如果能 fit 出 bitrate vs model size 的 power law，会是个非常有意义的 contribution。

### 5. Latent shape insensitivity 是 robust 的 sign

UL 对 channel 数和 spatial 大小不敏感，说明信息流不是被 shape 硬性 bottleneck 的，是被 prior 的 noise floor 软性 regulate 的。这种 robustness 是 model 设计得对的好 sign。对比 SD 的 VAE 你调一下 channel 数立刻崩。

## 几个我觉得值得 follow-up 的方向

1. **Bitrate scaling law**：Table 2 只有 small/medium，加 large model 看 trend。我的猜测是 optimal LF 跟 model size 是 power law 关系
2. **Flow matching version**：UL 用 VP schedule，换 flow matching 的 linear schedule 应该也 work，理论上更简单
3. **Discrete latents**：如果 prior 换成 discrete diffusion (D3PM)，可以压缩到 discrete tokens。这跟 TiTok 的方向有点像
4. **Single-stage training**：附录 B 试了 single-stage 到 FID 4，离 two-stage 的 1.4 还差很远。需要更 clever 的 objective
5. **Cross-modal latents**：UL 在 image 和 video 上 work，但能不能扩展到 audio、text、multimodal？Discussion section 提了 "discrete data like text" 是 future direction

## 关键 reference links

- **Variational Diffusion Models** (Kingma et al., 2021) - ELBO 推导基础
  https://arxiv.org/abs/2107.00630
- **Understanding Diffusion Objectives** (Kingma & Gao, 2023) - weighting 的理论分析
  https://arxiv.org/abs/2303.00848
- **LSGM** (Vahdat et al., 2021) - diffusion prior in VAE 的早期工作
  https://arxiv.org/abs/2111.09295
- **Latent Diffusion Models** (Rombach et al., 2022) - Stable Diffusion 基础
  https://arxiv.org/abs/2112.10752
- **Simpler Diffusion (SiD2)** (Hoogeboom et al., 2024) - pixel-space baseline
  https://arxiv.org/abs/2410.19324
- **DDPM** (Ho et al., 2020) - diffusion models 经典 paper
  https://arxiv.org/abs/2006.11239
- **Diffusion Autoencoders** (Preechakul et al., 2022) - diffusion encoder + decoder
  https://arxiv.org/abs/2111.15640
- **DINO** (Caron et al., 2021) - 自监督 representation baseline
  https://arxiv.org/abs/2104.14294
- **Deep Compression Autoencoder** (Chen et al., 2024) - 高压缩 autoencoder
  https://arxiv.org/abs/2410.10733
- **TiTok** (Yu et al., 2024) - 1D token latent
  https://arxiv.org/abs/2406.07550
- **DiffuseVAE** (Pandey et al., 2022)
  https://arxiv.org/abs/2201.00308
- **DiVAE** (Shi et al., 2022)
  https://arxiv.org/abs/2206.00386
- **ε-VAE** (Zhao et al., 2025)
  https://arxiv.org/abs/2410.04081
- **SWYCC** (Birodkar et al., 2024)
  https://arxiv.org/abs/2409.02529
- **Latent Diffusion without VAE** (Shi et al., 2025) - DINO latents
  https://arxiv.org/abs/2510.15301
- **Diffusion Transformers with RAE** (Zheng et al., 2025)
  https://arxiv.org/abs/2510.11690

Hope this gives you the full intuition, Andrej. 这篇 paper 不是那种 fundamentally new idea 的工作，而是一种 *unification* 和 *simplification*——把 VAE、diffusion prior、diffusion decoder 这三件事用一个 coherent math framework 串起来，且 empirical 上 work。我很 appreciate 这种 elegance。

---

# Unified Latents (UL): How to train your latents - 深度技术讲解

Andrej, 这是一篇非常 elegant 的 paper，来自 Google DeepMind Amsterdam (Hoek, Hoogeboom, Mensink, Salimans)。它解决了 latent diffusion models 中的核心痛点：**如何 principled 地 regularize latent representations**。让我从 motivation、math、architecture、experiments 四个层面来 build 你的 intuition。

## 1. Motivation & Problem Framing

现有的 latent space 设计存在三种困境：

| 方法 | Latent 信息密度 | Reconstruction 质量 | Base model 学习难度 |
|------|-----------------|---------------------|---------------------|
| Stable Diffusion VAE (KL to $N(0,I)$) | KL 权重手动调，难解释 bitrate | 中等 (PSNR ~24) | 容易 |
| DINO/SigLIP-based latents | 低 (semantic only) | 差 (PSNR ≤20) | 容易 |
| Unregularized autoencoder | 高 | 极好 | 非常难 |

核心 trade-off：**latent 信息越多 → reconstruction 越好 → base model 越难学**。这篇 paper 的核心 insight 是：**既然 latent 最终要用 diffusion model 来建模，那就让 diffusion prior 自己去 regularize 它**，而不是用一个 ad-hoc 的 KL 到 Gaussian 的方式。

## 2. Core Math: 三层 ELBO 的分解

### 2.1 基础 VAE ELBO (Equation 1)

$$-\log p_\theta(\pmb{x}) \leq \underbrace{\mathbb{E}_{z_0 \sim p_\theta(z_0|\pmb{x})}\left[-\log p_\theta(\pmb{x}|z_0)\right]}_{\text{decoder term}} + \underbrace{KL\left[p_\theta(z_0|\pmb{x}) \middle| p_\theta(z_0)\right]}_{\text{prior/encoder term}}$$

变量解释：
- $\pmb{x}$: input image (e.g. 512×512×3)
- $z_0$: latent at "clean" time $t=0$
- $p_\theta(\pmb{x}|z_0)$: **diffusion decoder** $D_\theta$
- $p_\theta(z_0|\pmb{x})$: **encoder** $E_\theta$ 输出的 posterior
- $p_\theta(z_0)$: **diffusion prior** $P_\theta$ 学到的分布

### 2.2 Diffusion KL Bound (Equation 2)

这是关键推导，来自 VDM (Kingma et al., 2021) 和 DDPM (Ho et al., 2020)：

$$KL[p(x_0|x) | p(x_0)] \leq \mathbb{E}_{t \sim \mathcal{U}(0,1)}\left[-\frac{d\lambda(t)}{dt}\frac{e^{\lambda(t)}}{2}w(\lambda_t) \|x - \hat{x}(x_t, \theta)\|^2\right] + KL[p(x_1|x) | p(x_1)]$$

变量详解：
- $\lambda(t) = \log(\alpha_t^2 / \sigma_t^2)$: **log signal-to-noise ratio** (log-SNR)
- $\alpha_t, \sigma_t$: forward process 系数，满足 $\alpha_t^2 + \sigma_t^2 = 1$ (variance preserving)
- $x_t = \alpha_t x + \sigma_t \epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$: noisy version
- $\hat{x}(x_t, \theta)$: network 预测的 clean data
- $-\frac{d\lambda(t)}{dt}$: log-SNR 随时间变化率，将 time-domain 积分转成 log-SNR domain
- $\frac{e^{\lambda(t)}}{2}$: 从 MSE 转到 likelihood 的 scaling factor
- $w(\lambda_t)$: weighting function，**必须是 1** 才能得到 tight bound
- $KL[p(x_1|x)|p(x_1)]$: 在 $t=1$ (纯噪声) 处的 KL，通常可忽略因为 $p(x_1) \approx \mathcal{N}(0,I)$

**Intuition**: 任意 distribution $p(x_0)$ 都可以用 diffusion 来 fit，KL 上界就是 diffusion ELBO。

### 2.3 Latent Prior Loss (Equation 3)

UL 的关键设计：encoder 输出 deterministic $z_{\text{clean}} = E(x, \theta)$，然后加固定噪声到 $t=0$：

$$p(z_0 | z_{\text{clean}}) = \mathcal{N}(\alpha_0 z_{\text{clean}}, \sigma_0^2 I)$$

其中 $\alpha_0 = \sqrt{\text{sigmoid}(+5)} \approx 1.0$, $\sigma_0 = \sqrt{\text{sigmoid}(-5)} \approx 0.08$，对应 $\lambda_z(0) = 5$。

$$KL[p(z_0|x) | p_\theta(z_0)] \leq \mathbb{E}_t\left[-\frac{d\lambda_z(t)}{dt}\frac{e^{\lambda_z(t)}}{2}w(\lambda_z(t)) \|z_{\text{clean}} - \hat{z}(z_t, \theta)\|^2\right] + KL[p(z_1|x) | \mathcal{N}(0,I)]$$

**为什么 fixed noise 是关键？**  
如果 encoder 输出 distribution $p(z|x) = \mathcal{N}(\mu_z, \text{diag}(\sigma_z^2))$ (像 LSGM, Vahdat et al., 2021)，会多出 encoder entropy term：

$$\mathcal{L}_e = -\frac{1}{2}\log[\sigma_z^2 e^{\lambda_z(0)} + 1]$$

这个 term 高方差且训练不稳定，ablation D 显示 learned variance 会 collapse 到 0。**Fixed noise 把 encoder 的 stochasticity 吸收到 forward process 里**，数学等价但训练稳定。

### 2.4 Decoder Loss (Equation 4)

$$-\log p_\theta(\pmb{x}|z_0) \leq \mathbb{E}_{t \sim \mathcal{U}(0,1)}\left[\frac{d\lambda_x(t)}{dt}\frac{e^{\lambda_x(t)}}{2} w_x(\lambda_x(t)) \|\pmb{x} - \hat{\pmb{x}}(\pmb{x}_t, z_0, \theta)\|^2\right]$$

这里 $w_x(\lambda_x(t)) = \text{sigmoid}(\lambda_x(t) - b)$ 是 **reweighted ELBO**，和 prior 不同的核心是：
- **Prior 必须 $w=1$** (unweighted)：否则 encoder 会"作弊"把信息塞到 discount 最大的 noise levels
- **Decoder 可以 reweighted**：high-frequency 信息即使 imperceptible 也让 decoder 处理，因为 cost/bit 更低

## 3. Architecture & Training Pipeline

### 3.1 三组件架构 (Figure 1)

```
Image x ──► Encoder E_θ (ResNet) ──► z_clean ──(add noise)──► z_0
                                                              │
                                                              ▼
              Diffusion Prior P_θ (ViT) ◄── z_t (从 z_0 加噪) 
                                                              │
                                                              ▼
Image x_t ──► Diffusion Decoder D_θ (UViT) ◄── z_0 (conditioning) ──► x̂
```

具体配置：
- **Encoder**: ResNet with channels [128, 256, 512, 512], 2 residual blocks for downsampling, 3 blocks final stage, 2×2 patching
- **Prior** (Stage 1): single-level ViT, 8 blocks, 1024 channels
- **Base model** (Stage 2): 2-stage ViT with [512, 1024] channels, [6, 16] blocks, dropout 0.1
- **Decoder**: UViT architecture with conv down/up [128, 256, 512] + transformer (8 blocks, 1024 channels)

### 3.2 Two-Stage Training

**Stage 1**: 训练 $E_\theta$ + $P_\theta$ + $D_\theta$ 同时优化 $\mathcal{L}_z + \mathcal{L}_x$ (Algorithm 1)

```
1. Sample x ~ p_data
2. z_clean = E(x, θ)                          # deterministic encoding
3. Sample t ~ U(0,1), ε ~ N(0,I)
4. z_t = α_z(t) z_clean + σ_z(t) ε            # forward noise latent
5. L_z(θ) = -dλ_z/dt · e^{λ_z(t)}/2 · ||z_clean - ẑ(z_t,θ)||² + KL[z_1|x || N(0,I)]
6. z_0 = α_z(0) z_clean + σ_z(0) ε_z          # noisy latent for decoder
7. x_t = α_x(t) x + σ_x(t) ε
8. L_x(θ) = dλ_x/dt · e^{λ_x(t)}/2 · w_x(λ_x) · ||x - x̂(x_t, z_0, θ)||²
9. Optimize L_z + L_x
```

**Stage 2**: 冻结 $E_\theta$，重训 base model with sigmoid weighting。因为 stage 1 的 prior 用 unweighted ELBO，对 low-frequency 和 high-frequency 平等加权，生成质量差。Stage 2 换成 $w = \text{sigmoid}(\lambda - b)$ 显著提升 FID (App. B 显示单阶段训练只能达到 FID 4，双阶段能到 1.4)。

## 4. Key Hyperparameters: 控制信息流

### 4.1 Loss Factor (LF) 和 Sigmoid Bias (b)

Decoder loss 乘以 loss factor $c_{\text{lf}}$ (等效于 down-weighting KL)：

$$\mathcal{L}_x(\theta) = c_{\text{lf}} \cdot \frac{d\lambda_x}{dt} \frac{e^{\lambda_x(t)}}{2} \text{sigmoid}(\lambda_x(t) - b) \|\pmb{x} - \hat{\pmb{x}}\|^2$$

**Effect of LF (Table 2)**:

| LF | bits/pixel | rFID@50k | PSNR | gFID (small) | gFID (medium) |
|----|------------|----------|------|--------------|----------------|
| 1.3 | 0.035 | 0.79 | 25.7 | 1.42 | 1.37 |
| 1.5 | 0.059 | 0.47 | 27.6 | 1.54 | 1.31 |
| 1.7 | 0.083 | 0.36 | 28.9 | 1.77 | 1.38 |
| 1.9 | 0.101 | 0.31 | 29.6 | 2.02 | 1.45 |
| 2.1 | 0.116 | 0.27 | 30.1 | 2.38 | 1.58 |

**Intuition**: 
- LF ↑ → latent bits ↑ → rFID/PSNR 改善 → gFID 变差 (对小 model)
- **Small models 偏好低 bitrate** (1.3-1.5)，因为容量有限
- **Large models 对 bitrate 不敏感**，甚至偏好稍高 bitrate

### 4.2 Latent Bitrate Upper Bound

这是 paper 声称的"interpretable bound"的本质。由于 prior 用 unweighted ELBO，KL 等于 noise levels 上的积分，每个 noise level $\lambda$ 贡献的 bits 是确定的。在 $\lambda_z(0) = 5$ 时，每个 latent channel 最多编码 ~5 bits (modulo 量化效应)，乘以 channel 数和 spatial 维度就是 latent 的 bitrate 上界。

## 5. Experiments & Scaling

### 5.1 ImageNet-512 (Figure 4)

UL 在 training FLOPs vs FID 上全面超越：
- **UL**: FID 1.4 @ ImageNet-512
- 比 SD latents 训练的相同架构 (small SD, medium SD) 都好
- 比 pixel-space diffusion (SiD2, Hoogeboom et al., 2024) 也好

### 5.2 Latent Shape Insensitivity (Table 3, 4)

**Channel ablation (固定 32×32 spatial)**:

| # channels | rFID | gFID@50K |
|------------|------|----------|
| 4 | 7.19 | (不能训) |
| 8 | 1.53 | (不能训) |
| 16 | 0.54 | 1.76 |
| 32 | 0.42 | 1.60 |
| 64 | 0.48 | 1.77 |

**Spatial ablation (固定 32 channels)**:

| Latent shape | rFID | gFID |
|--------------|------|------|
| 64×64×32 | 0.40 | 2.12 |
| 32×32×32 | 0.41 | 1.63 |
| 16×16×32 | 1.41 | 1.74 |

**Insight**: UL 对 latent shape 几乎不敏感 (除了 channel 太少 reconstruction 不行)，因为 bitrate 是被 diffusion prior 的 noise floor 控制的，而不是 channel count。这是相比 SD VAE 的巨大优势 —— SD 必须精细调 channels 和 spatial downsampling。

### 5.3 Text-to-Image (Table 1)

| Latents | gFID@30K | CLIP |
|---------|----------|------|
| UL (LF=1.5) | 4.1 | 27.1 |
| Pixel (no latents) | 5.0 | 27.0 |
| StableDiffusion | 6.8 | 27.0 |

### 5.4 Kinetics-600 (Figure 9)

- **Small UL**: FVD 1.7
- **Medium UL**: **FVD 1.3 (SOTA)**
- 超越 MAGVIT, W.A.L.T. 等 token-based 方法

## 6. Ablations (Table 6) - 哪些是 essential

| Variant | bits/pixel | rFID | gFID |
|---------|------------|------|------|
| UL baseline (LF=1.5) | 0.059 | 0.47 | 1.54 |
| A. Stop-gradient on prior input (用 KL to N(0,I) regularize) | 0.121 | 1.81 | 7.80 |
| B. Less noise: λ_z(0)=10 (σ≈0.007) | 0.008 | 28.27 | — |
| C. Train AE on ImageNet (in-distribution) | 0.034 | 1.37 | 1.63 |
| D. Learned encoder variance | 0.060 | 0.69 | 1.81 |

**关键发现**:
- **A**: 没有 prior gradient，必须减 channels 到 8 才能工作，gFID 暴跌到 7.8 → prior regularizer 是 essential
- **B**: 几乎无噪声 → encoder 把所有信息塞进 latent，prior 学不到 bitrate，rFID 28.27 太烂训不动 base model
- **D**: Learned variance 不稳定，gFID 1.81 > baseline 1.54 → fixed noise 更好

## 7. 与 Related Work 的关键差异

### 7.1 vs LSGM (Vahdat et al., 2021)

LSGM 联合训练 diffusion prior 和 VAE，但需要 encoder entropy term $\mathbb{E}_{q(z_0|x)}\log q(z_0|x)$，高方差不稳定。UL 用 **fixed noise** 吸收这个 term 进 forward process，目标函数只剩两项 (decoder + prior)，训练稳定。

Paper: https://arxiv.org/abs/2111.09295

### 7.2 vs Stable Diffusion / Latent Diffusion (Rombach et al., 2022)

SD 用 KL to $\mathcal{N}(0,I)$ + GAN-trained decoder + channel bottleneck。问题：
- KL 权重手动调，bitrate 不可解释
- GAN decoder 无 likelihood，mode collapse 但 rFID 看着好
- Latent shape 选择 critical (4 channels, 8× downsampling)

UL 用 diffusion decoder (更强大，predicts distribution)，diffusion prior regularizer (principled bitrate)，对 shape 不敏感。

Paper: https://arxiv.org/abs/2112.10752

### 7.3 vs DINO-based latents (Show-o, MAR, etc., Shi et al., 2025; Zheng et al., 2025)

DINO latents semantic 好但 PSNR ≤20，高频细节丢失。UL 可以同时获得 semantic + high-freq (PSNR 27-30)。

- https://arxiv.org/abs/2510.15301
- https://arxiv.org/abs/2510.11690

### 7.4 vs DiVAE / DiffuseVAE / ϵ-VAE

- DiffuseVAE: 先训 MSE autoencoder，再 finetune diffusion decoder (两阶段不联合)
- DiVAE: discrete VQ-VAE tokens + diffusion decoder
- ϵ-VAE: diffusion decoder + channel bottleneck (没有 learned prior)
- SWYCC: 类似 ϵ-VAE

UL 是第一个 jointly train encoder + diffusion prior + diffusion decoder 并获得 principled bitrate bound 的工作。

- DiffuseVAE: https://arxiv.org/abs/2201.00308
- DiVAE: https://arxiv.org/abs/2206.00386
- ϵ-VAE: https://arxiv.org/abs/2410.04081
- SWYCC: https://arxiv.org/abs/2409.02529

## 8. Limitations & Open Questions

1. **Diffusion decoder 采样贵**: 比 GAN decoder 慢一个量级，需要 distillation
2. **为什么 weakly informative latents 更易建模？** Paper 没完全回答：是不是把建模难度推给 decoder 了？Diffusion decoder 比 GAN 严格更强大 (predicts distribution)，但 GAN 的 mode collapse 可能反而帮 rFID
3. **Scaling laws 未建立**: 大 base model 偏好更多 informative latents，需要预测 optimal bitrate vs training budget 的关系
4. **Single-stage training 不 work**: 附录 B 显示 end-to-end 训练只能到 FID 4，必须分两阶段

## 9. Personal Takeaways (给你的 intuition)

1. **统一性之美**: prior 和 decoder 都用 diffusion，同一套数学 (KL bound, ELBO) 同时适用于 latent space 和 pixel space。这让 hyperparameter (LF, bias) 在两个空间有可比语义。

2. **Fixed noise ≠ learned variance**: 这是个反直觉的设计。传统 VAE 教科书说 learned variance 更 expressive。但实践上 fixed noise 让 encoder entropy term 消失，训练稳定。这是 deep learning 里反复出现的 theme —— 约束更少的 expressive 但更稳定的版本往往更好。

3. **Bitrate 可解释性的代价**: UL 用 $\lambda_z(0) = 5$ 硬性限制了 latent precision。这意味着 high-freq 细节完全靠 decoder 补。如果未来想做更高保真 (e.g. 4K, medical imaging)，这个 noise floor 需要重新设计。

4. **Stage 2 retraining 的本质**: Stage 1 用 ELBO weight=1 (符合 bitrate bound)，Stage 2 用 sigmoid weight (image quality friendly)。这是把"度量任务"和"生成任务"分离 —— ELBO 用来定义 latent，生成质量单独优化。

5. **与 Flow Matching / Rectified Flow 的关系**: UL 用 variance-preserving noise schedule ($\alpha^2 + \sigma^2 = 1$)，但理论上换成 flow matching 的 linear schedule 也应该 work。Paper 没探索这个。

## Reference Links

- **Paper PDF**: 这篇论文暂未在 arxiv 公开，目前只有 DeepMind 内部版本
- **Variational Diffusion Models** (Kingma et al., 2021): https://arxiv.org/abs/2107.00630
- **Understanding Diffusion Objectives** (Kingma & Gao, 2023): https://arxiv.org/abs/2303.00848
- **Simpler Diffusion (SiD2)** (Hoogeboom et al., 2024): https://arxiv.org/abs/2410.19324
- **LSGM** (Vahdat et al., 2021): https://arxiv.org/abs/2111.09295
- **Latent Diffusion Models** (Rombach et al., 2022): https://arxiv.org/abs/2112.10752
- **DDPM** (Ho et al., 2020): https://arxiv.org/abs/2006.11239
- **Diffusion Autoencoders** (Preechakul et al., 2022): https://arxiv.org/abs/2111.15640
- **TiTok** (Yu et al., 2024): https://arxiv.org/abs/2406.07550
- **DINO** (Caron et al., 2021): https://arxiv.org/abs/2104.14294
- **Deep Compression Autoencoder** (Chen et al., 2024): https://arxiv.org/abs/2410.10733

希望这给你 build 起了完整的 intuition —— UL 的核心创新是**把 diffusion prior 当成 bitrate 定义器**，通过 fixed noise 消除 encoder entropy 项，从而得到一个 principled、stable、interpretable 的 latent learning framework。整个方法的 elegance 在于三个组件 (encoder/prior/decoder) 用同一套 diffusion math 串起来，没有任何 ad-hoc 的 loss weight。
