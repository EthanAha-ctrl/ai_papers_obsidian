---
source_pdf: Flow to the Mode.pdf
paper_sha256: c6a1d6c640175e2a6334f344c953cff88399b2c70fd5c68e025b204332c30b83
processed_at: '2026-08-04T09:44:53-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 FlowMo

## 一句话总结

现有的 image tokenizer 都是 CNN + GAN loss 的套路，FlowMo 说：我用纯 transformer + diffusion，不用 adversarial loss，不用 distillation，照样能打到 SOTA，关键秘诀是 **"别想把所有 mode 都 match，主动去 seek 你想要的那个 mode"**。

---

## 为什么现有 tokenizer 有问题

先说背景。生成图片的 pipeline 一般分两步：

第一步，把 256×256 的图片压缩成一小串 token（比如 256 个 token，每个 18 bits，总共才 4608 bits）。这一步叫 tokenization，或者叫 image compression。

第二步，在这串 token 上训一个 generative model（MaskGiT 或者 LlamaGen 之类的）。

第一步的 tokenizer 至关重要——压缩得不好，后面生成质量再高也白搭。

现有的 SOTA tokenizer，从 VQGAN 到 OpenMagViT-V2 到 LlamaGen 到 TiTok，基本都长一个样：

- Encoder 用 CNN 把图片 downsample 成 2D spatial latent
- Decoder 用 CNN 把 latent upsample 回图片
- Training loss = MSE + perceptual + adversarial（GAN loss）
- 有时候还要从别人训好的 tokenizer distill

这套路有几个问题：

**Adversarial loss 很难搞**。GAN training 的老问题了——mode collapse、training instability、需要 LeCam regularization、需要 adaptive gradient scale、需要精心调 loss weight。每个 trick 都是为了 patch 一个 instability。

**依赖 CNN**。现在 transformer 在 scaling 上更 predictable，硬件也更 optimize for transformer，但 tokenizer 这块还 stuck in CNN era。

**依赖 distillation**。TiTok 虽然用了 1D latent 和 transformer，但先要从 pretrained CNN tokenizer distill，不是 end-to-end 学出来的。

FlowMo 想做的事：把这些都扔掉，用纯 transformer + diffusion autoencoder，end-to-end 训，看能不能到 SOTA。

---

## Diffusion Autoencoder 为什么之前没 work

Diffusion autoencoder 这个概念 2022 年就有了（Preechakul et al. CVPR 2022, https://arxiv.org/abs/2111.15640）。思路很简单：

- Encoder 把图片压成一个 latent code c
- Decoder 是个 conditional diffusion model，学 p(x|c)
- End-to-end 训

听起来很自然，但之前一直没在 ImageNet-1K 上打到 SOTA。为什么？

FlowMo 的核心 insight 在这里。让我用直觉来解释。

---

## 核心问题：Multimodal Reconstruction Distribution

假设你有一张人脸图片，你用 encoder 压成 256 个 token。这 256 个 token 能装多少信息？4608 bits。一张 256×256×3 的图片有 196608 个 pixel，每个 pixel 值域 256。原始信息量是 196608 × 8 = 1,572,864 bits。

你把 1.5 million bits 压成 4608 bits，丢了 99.7% 的信息。

这意味着 latent code c 只能 specify 图片的"大意"——"这是一个微笑的白人女性，脸在中间，背景是绿色"。但具体每根头发丝怎么弯、每个毛孔长哪、每条皱纹多深，c 里根本没存。

所以 p(x|c) 是 **multimodal** 的。给定 c，有无数种 hair / pore / wrinkle 的组合都 match 这个 c。每一种组合就是一个 mode。

---

## 传统 Diffusion Autoencoder 的 Failure Mode

你训一个 diffusion decoder 去 model p(x|c) 的 velocity field。Training 用的是 flow matching loss（本质是 MSE）：

$$\mathcal{L}_{flow} = \mathbb{E}\left[\|x - z - d_\theta(x_t, c, t)\|_2^2\right]$$

这里：
- $x$ = clean image
- $z$ = Gaussian noise  
- $x_t = t \cdot z + (1-t) \cdot x$ = noised image
- $t$ = noise level，$t \in [0,1]$，$t=0$ 是 clean，$t=1$ 是 pure noise
- $d_\theta$ = decoder，output 是 velocity
- $c$ = quantized latent code

target 是 $x - z$，这其实是 true velocity 的 negative（即 denoising direction）。

MSE loss 的本质是：network 会去学 **所有 modes 的 conditional expectation**，也就是所有 modes 的 weighted average。

问题来了。Sampling 的时候你解 ODE，deterministic 地走。你 follow 的是 average velocity field。但 average of all modes 不等于任何一个 mode！它落在 modes 之间的"valley"里——一个 low-density region。

结果就是：**blurry, washed-out reconstruction**。你看到的人脸是所有可能细节的平均，没有 sharpness。

这就是 diffusion autoencoder 之前打不过 GAN-based tokenizer 的根本原因。GAN 有 adversarial loss，它逼着 decoder 只 sample 一个 sharp mode，不许 average。

---

## FlowMo 的解决方案：Mode-Seeking

FlowMo 的 insight 一句话：**与其试图 match 所有 modes，不如主动 seek 那些和原图 perceptually close 的 modes，drop 掉其他的。**

具体怎么做？两个 ingredient：

### Ingredient 1: Mode-Seeking Post-Training (Stage 1B)

Stage 1A 先正常训 diffusion autoencoder，end-to-end，用 flow loss + perceptual loss + LFQ losses。这就是 standard recipe，训完你会得到一个 average-mode 的 decoder。

然后 Stage 1B 做一件关键的事：**freeze encoder，只训 decoder，但这次直接 backprop through 整个 sampling chain**。

具体来说，从 pure noise $z$ 开始，跑 $n$ 步 Euler integration：

$$d_{t_i}(x_t) = x_t + (t_{i+1} - t_i) \cdot d_\theta(x_t, c, t_i)$$

变量解释：
- $d_{t_i}$ = 第 $i$ 步的 sample update function
- $x_t$ = 当前 noised image
- $t_i$ = 当前 noise level
- $t_{i+1}$ = 下一步 noise level  
- $d_\theta(x_t, c, t_i)$ = decoder 输出的 velocity
- $(t_{i+1} - t_i)$ = 步长

跑 $n$ 步后得到一个 sample，然后算这个 sample 和原图 $x$ 的 perceptual distance：

$$\mathcal{L}_{sample} = \mathbb{E}\left[d_{perc}\left(x, d_{t_n} \circ d_{t_{n-1}} \circ \cdots \circ d_{t_1}(z)\right)\right]$$

这里 $d_{t_n} \circ \cdots \circ d_{t_1}(z)$ 就是 $n$ 步 Euler 积分后的最终 sample，$d_{perc}$ 是 perceptual distance。

**整个 chain 都 differentiable**，gradient 从 final sample 一路 backprop 到 decoder 的所有参数。这逼着 decoder 学会：在 sampling 过程中，主动 navigate 到和原图 perceptually close 的 mode，drop 掉不 close 的 mode。

这个和 Stage 1A 的 $\mathcal{L}_{perc}$ 本质不同。Stage 1A 的 perceptual loss 只监督 **1-step prediction**：

$$\mathcal{L}_{perc} = \mathbb{E}\left[d_{perc}\left(x, x_t + t \cdot d_\theta(x_t, c, t)\right)\right]$$

这里 $x_t + t \cdot d_\theta$ 只是从 $x_t$ 走一步的结果。1-step prediction 好不等于 final sample 好，因为 accumulated error 会 snowball。

Table 7 的 ablation 很说明问题：
- $\mathcal{L}_{sample}$（8步，weight=0.01）：rFID = 1.28
- 各种 weight 的 $\mathcal{L}_{perc}$：最好 rFID = 1.60

直接 optimizing final sample 远胜过 optimizing 1-step prediction。

---

### Ingredient 2: Shifted Sampler

Sampling 的时候，timestep spacing 用一个 shift parameter $\rho$：

$$(t_1, \ldots, t_n) = \left(\left(\frac{n}{n}\right)^\rho, \left(\frac{n-1}{n}\right)^\rho, \ldots, \left(\frac{1}{n}\right)^\rho\right)$$

变量：
- $n$ = 总步数
- $\rho$ = shift hyperparameter
- $t_i$ = 第 $i$ 个 timestep

$\rho = 1$ 是 linear spacing，标准的 rectified flow sampler。$\rho = 4$ 是 FlowMo 用的。

$\rho = 4$ 的效果：timestep 集中在 low noise levels（$t \to 0$），high noise levels 用大步跨过。

为什么这样好？直觉：

**High noise levels（$t \to 1$）**：这时 $x_t \approx z$（pure noise），decoder 完全靠 $c$ 来 predict velocity。由于 $c$ 的 conditional signal 极强，$p(x|c)$ 被强 constrain，modes 集中。可以 take large steps，不需要 fine-grained sampling。

**Low noise levels（$t \to 0$）**：这时 $x_t \approx x$（clean image），细节决定成败。Modes 可能分散，需要 small steps 来 carefully navigate 到正确的 mode。

所以把 FLOPs 集中在 low noise levels 是高效的——大步跨过 "modes 集中" 的 high noise region，小步精雕 "modes 分散" 的 low noise region。

Ablation（Table 4）：unshifted sampler（$\rho=1$）→ rFID 3.42 vs 2.87 baseline。

---

## 两个 Stage 之外的关键细节

### 细节 1: Thick-Tailed Noise Schedule

Stable Diffusion 3 用 logit-normal noise schedule 来 sample $t$。Logit-normal 给 $t=0$ 和 $t=1$ 分配 0 probability mass。

但对 FlowMo，$t=1$（pure noise）的 velocity estimate **极其 critical**。因为 $x_1 = z$（pure noise），$d_\theta(z, c, 1)$ 完全靠 $c$。如果 network 在 $t=1$ 没 training signal，输出 garbage velocity，后续 sampling 全走偏。

后果不光是 rFID 崩，还有 **discoloration**——重建图片颜色不正常（Figure 7）。

FlowMo 的 fix 很简单：10% 的时间从 uniform $[0,1]$ sample $t$，而不是 logit-normal。这给 $t=0$ 和 $t=1$ 分配了 non-zero probability mass。

Ablation：logit-normal → rFID 4.08, PSNR 16.45（baseline 2.87, 20.71）。这是 second worst ablation，仅次于 no perceptual loss。

---

### 细节 2: Perceptual Loss on 1-Step Prediction in Stage 1A

Stage 1A 除了 flow loss，还有一个 perceptual loss on 1-step denoised prediction：

$$\mathcal{L}_{perc} = \mathbb{E}\left[d_{perc}\left(x, x_t + t \cdot d_\theta(x_t, c, t)\right)\right]$$

$x_t + t \cdot d_\theta$ 是从 $x_t$ 走一步步长 $t$ 的结果。在 $t=1$ 时，$x_1 = z$，$z + 1 \cdot d_\theta(z, c, 1) = z + (x - z) = x$。所以这是 1-step denoised prediction。

没有这个 loss：rFID 从 2.87 崩到 13.86（Table 4）。最 catastrophic 的 degradation。Yang & Mandt (NeurIPS 2023, https://arxiv.org/abs/2305.18231) 理论上证明了：diffusion autoencoder 的 end-to-end objective 对应一个 modified variational lower bound with non-Gaussian decoder，但单靠 flow loss 不够，需要 perceptual loss 来 inject perceptual priors。

---

### 细节 3: Different Perceptual Networks for Different Stages

- Stage 1A：LPIPS-VGG
- Stage 1B：ResNet

MaskBit (https://arxiv.org/abs/2409.16211) 发现 ResNet perceptual loss 对 tokenizer training 有效，但 FlowMo 在 Stage 1A 用 ResNet 效果不好。Stage 1B 用 ResNet 更好——可能是 LPIPS-VGG 的 gradient 在 long sampling chain 中不够 informative。

---

### 细节 4: Classifier-Free Guidance in Tokenizer

FlowMo 在 tokenizer training 时 dropout latent code $c$ 10% 的时间，inference 时用 classifier-free guidance（CFG weight 1.5）+ guidance interval (0.17, 1.02)。

Guidance 让 sample 更"confident"地靠近某个 mode，避免 average-out。Guidance interval 限制了 guidance 的应用范围，避免 over-saturation。

Ablation：no guidance → rFID 3.28 vs 2.87。在 tokenizer 里用 CFG 是 non-obvious 的发现——以前觉得 CFG 是 generative model 的事，tokenizer 不需要。

---

## 架构长什么样

基于 MMDiT（Stable Diffusion 3 的 architecture, https://arxiv.org/abs/2403.03206），从 Flux (https://github.com/black-forest-labs/flux) 改造。

**Encoder**：
- 输入：patchified image $x$ + initial latent $c_0$（全 0）
- 把 latent tokens 和 image tokens concatenate
- Self-attention across concatenated sequence
- 两个 modality 的 streams 独立（MMDiT 的设计）
- 输出：$\hat{c} = e_\theta(x, c_0)$

**Quantization**：
用 LFQ (Lookup-Free Quantization, https://arxiv.org/abs/2310.07141)：

$$c = q(\hat{c}) = 2 \cdot \mathbb{1}[\hat{c} \geq 0] - 1$$

每个 element 的 sign 作为 bit。如果 latent sequence length = 256，token size = 18，codebook = $2^{18} = 262144$。

为什么 LFQ 不用 FSQ？Ablation（Table 4）：FSQ rFID = 3.14 vs LFQ 2.87。LFQ 稍好。

**Decoder**：
- 输入：noised image $x_t$ + quantized latent $c$ + time $t$
- 和 encoder 一样的 MMDiT architecture
- Time $t$ 通过 AdaLN modulation condition 每个 block
- Decoder 比 encoder 大且深（encoder depth = 8, decoder depth = 16）

**μP Parameterization**：
所有模型用 μP (https://arxiv.org/abs/2203.03466)，让 hyperparameters 从 small exploratory configs "transfer" 到 large configs。避免在大模型上做 hyperparameter search。

---

## 实验结果

### Tokenization SOTA（Table 1）

| BPP | Model | rFID↓ | PSNR↑ | SSIM↑ | LPIPS↓ |
|-----|-------|-------|--------|--------|---------|
| 0.070 | OpenMagViT-V2 | 1.17 | 21.63 | 0.640 | **0.111** |
| 0.070 | **FlowMo-Lo** | **0.95** | **22.07** | **0.649** | 0.113 |
| 0.219 | LlamaGen-32 | 0.59 | 24.44 | 0.768 | **0.064** |
| 0.219 | **FlowMo-Hi** | **0.56** | **24.93** | **0.785** | 0.073 |

FlowMo 在 rFID、PSNR、SSIM 三个指标 SOTA。LPIPS 稍逊——可能因为 mode-seeking 倾向于 sharp dominant modes，drop 掉一些 LPIPS-sensitive 的 fine details。

---

### Stage 1B 的效果（Table 5）

| Model | rFID↓ | PSNR↑ | LPIPS↓ |
|-------|-------|--------|---------|
| FlowMo-Lo (no post-train) | 1.10 | 21.38 | 0.134 |
| FlowMo-Lo (post-trained) | **0.95** | **22.07** | **0.113** |
| FlowMo-Hi (no post-train) | 0.73 | 24.02 | 0.086 |
| FlowMo-Hi (post-trained) | **0.56** | **24.93** | **0.073** |

**三个指标同时提升**。这非常 non-trivial——通常 mode-seeking 会 trade rFID for PSNR。这说明 mode-seeking 和 naive mode-matching 的区别在于完全不同的 optimization landscape，而非同一 tradeoff curve 上的不同位置。

---

### Generation（Table 2）

| Tokenizer | FID↓ | IS↑ | sFID↓ | Prec.↑ | Rec.↑ |
|-----------|------|-----|--------|---------|--------|
| OpenMagViT-V2 | **3.73** | 241 | 10.66 | 0.80 | **0.51** |
| FlowMo-Lo | 4.30 | **274** | **10.31** | **0.86** | 0.47 |

有趣的是：tokenizer 更好了，但 generation 没有全面更好。FID 稍差，Precision 更好，Recall 稍差。

直觉解释：mode-seeking 让每个 $c$ 的 $p(x|c)$ 更 concentrated（precision 高），但整个 marginal $p(x) = \int p(x|c) p(c) dc$ 的 diversity 下降（recall 低）。

Paper 诚实地说："There is an interesting and complicated interplay between tokenizer quality and generation quality"。这是 open question。

---

## Mode-Seeking 的精确机制（C.4）

通过 Precision/Recall 分析：

| Comparison | Prec.↑ | Rec.↑ |
|------------|--------|--------|
| vs val stats (1:1), no post-train | 0.974 | 0.988 |
| vs val stats (1:1), post-trained | **0.993** | **0.991** |
| vs train stats (no correspondence), no post-train | 0.734 | 0.660 |
| vs train stats, post-trained | **0.766** | 0.634 |

**1:1 correspondence（每个 $c$ 对应一张 specific 原图）**：post-training 同时提升 precision 和 recall。说明 mode-seeking 让 $p(x|c)$ 对于每个 $c$ 都更 concentrated around the true mode（原图），所以 1:1 下更好。

**No correspondence（和 train set 比较）**：precision 提升，recall 下降。Expected tradeoff——sacrificing global diversity for fidelity。

---

## Scaling（C.3）

| μP width | # Params (×10⁶) | rFID↓ | PSNR↑ | LPIPS↓ |
|----------|------------------|-------|--------|---------|
| 2 | 260 | 7.77 | 20.84 | 0.169 |
| 3 | 367 | 5.31 | 21.28 | 0.160 |
| 4 | 517 | 4.45 | 21.60 | 0.155 |
| 5 | 710 | 3.84 | 21.70 | 0.152 |

所有指标 monotonically improve with width。Transformer-based architecture scaling well-defined——和 CNN-based tokenizer 相比，更 predictable。

---

## Limitations

1. **Inference time**：需要 25 步 forward pass（vs CNN tokenizer 的 1 步）。可以 distill 5-10x speedup，但 paper 没做
2. **Generation metrics**：tokenizer 更好但 generation 没有全面更好
3. **LPIPS 劣势**：mode-seeking 可能 drop 掉一些 LPIPS-sensitive 的 details
4. **Stage 1B compute**：backprop through full sampling chain 很 expensive，只 train 了 1 epoch
5. **Longer training**：rFID 在 Stage 1A 没 saturate

---

## 对 Build Intuition 有帮助的联想

### 联想 1: DDPO / AlignProp / Reward Backpropagation

Stage 1B 的 backprop through sampling chain 和 AlignProp (https://arxiv.org/abs/2310.03739) 的思路很像。AlignProp 是 text-to-image diffusion model 的 post-training，通过 backprop through 整个 sampling chain 来 optimize a reward function。DDPO (https://arxiv.org/abs/2305.13301) 用 RL objective 做 post-training。

FlowMo 的 $\mathcal{L}_{sample}$ 可以看作一种 reward：reward = perceptual similarity to original image。通过 backprop through sampling chain 来 maximize 这个 reward。

区别在于：AlignProp/DDPO optimize 生成质量（aesthetic、prompt alignment），FlowMo optimize reconstruction quality。但 mechanism 是一样的——直接优化 final sample quality。

### 联想 2: Rate-Distortion-Perception Tradeoff

Blau & Michaeli (ICML 2019, https://arxiv.org/abs/1906.06841) 提出了 rate-distortion-perception tradeoff。在 lossy compression 中，你可以 optimize distortion（MSE/PSNR）或者 perception（rFID/realism），但不能同时 maximize 两者——除非你增加 rate（BPP）。

FlowMo 的 mode-seeking 似乎同时提升了 distortion 和 perception，没有 trade off。这看起来违反了 RDP tradeoff？

实际上没有。RDP tradeoff 是说在给定 rate 下，distortion 和 perception 有 tradeoff。但 FlowMo 通过更高效的利用 rate（更好的 encoder + 更好的 decoder）来提升整个 Pareto frontier，不是在同一 frontier 上移动。

Figure 9 的实验验证了这一点：naive likelihood maximization（通过 modulate $x_1$ norm）只能 trade off rFID 和 PSNR，而 mode-seeking 同时提升两者。

### 联想 3: Truncation Sampling in GANs

Paper 里提到 modulating $x_1$ norm 来调整 sample likelihood，类比了 truncation sampling in GANs (Brock et al. ICLR 2019, https://arxiv.org/abs/1809.11096)。Truncation sampling 是从 truncated Gaussian sample $z$，减少 diversity 来提升 quality。

但 truncation sampling 只能 trade quality for diversity，不能同时提升。Mode-seeking 不同——它通过 active selection of modes，同时提升 fidelity 和 sharpness。

### 联想 4: Classifier-Free Guidance 的深层含义

CFG 的本质是：在 sampling 时，把 conditional velocity $v_c$ 和 unconditional velocity $v_{\emptyset}$ 的 difference amplify：

$$v_{guided} = v_{\emptyset} + w \cdot (v_c - v_{\emptyset})$$

这里 $w$ 是 guidance weight。直觉：$v_c - v_{\emptyset}$ 是"latent code $c$ 带来的信息增量"。Amplify 它 = 更 confident 地走向 $c$ 指定的 mode。

这在 tokenizer context 下的意义：让 reconstruction 更 faithful 地反映 $c$ 的信息，避免 average-out 到 modes 之间的 valley。

Guidance interval (https://arxiv.org/abs/2404.07724) 限制了 guidance 的应用范围（只在某些 timesteps guidance），避免 over-saturation——即过度 confident 导致的 artifacts。

### 联想 5: VQ-VAE 的 Posterior Collapse 问题

VQ-VAE 训练时经常遇到 posterior collapse——codebook 只有少数几个 code 被使用。LFQ 的 entropy loss 就是来解决这个问题的。

FlowMo 用 LFQ + entropy loss + commitment loss 来确保 codebook well-utilized。Table 6 显示 commitment loss weight = 0.000625，相当小。Paper 还提到 bounding commitment loss with tanh 会 degrade performance——unbounded 更好。

这个反直觉的发现可能和 diffusion decoder 的 training dynamics 有关：diffusion loss 本身就有 regularization effect，额外的 strong commitment loss 会 over-constrain encoder，限制其 expressivity。

### 联想 6: EDM2 的 Weight Normalization

FlowMo 用了 EDM2 (https://arxiv.org/abs/2312.02696) 的 weight normalization trick：在 MLP blocks 中 per-step normalize weight matrices。这是 Karras et al. 发现的训练 diffusion model 的 critical trick——counteract exploding activations and weight matrices。

这个 trick 在大 transformer 训练中尤其重要，因为 long sequence length + bfloat16 precision 容易导致 numerical instability。FlowMo 用 bfloat16，sequence length 长（256 latent tokens + 1024 image patches = 1280 tokens），所以这个 trick 很必要。

### 联想 7: 从 2D Latent 到 1D Latent 的转变

传统 tokenizer（VQGAN, LlamaGen, OpenMagViT-V2）都用 2D spatially-aligned latent。比如 16×16 grid，每个位置一个 code。这个 inductive bias 很自然——图片是 2D 的，latent 也 2D 很合理。

TiTok 首先用 1D latent，但靠 distillation。FlowMo 是第一个 end-to-end 学 1D latent 的 SOTA tokenizer。

1D latent 的好处：更 flexible 的 token allocation（不受 spatial grid constraint），更容易 variable-length tokenization（生成时可以 early stop），更 compatible with LLM-style autoregressive modeling。

但 1D latent 失去了 spatial inductive bias，需要 model 自己 learn spatial structure。Transformer 的 capacity 足够做到这一点，但需要更多 data 和 compute。

### 联想 8: 和 VAE 的 ELBO 联系

Yang & Mandt (NeurIPS 2023) 证明了 diffusion autoencoder 的 end-to-end objective 对应一个 modified variational lower bound with non-Gaussian decoder $p(x|c)$。

标准 VAE 的 ELBO：

$$\log p(x) \geq \mathbb{E}_{q(c|x)}[\log p(x|c)] - D_{KL}(q(c|x) \| p(c))$$

这里 $p(x|c)$ 是 Gaussian decoder。Diffusion autoencoder 用 diffusion model 作为 decoder，$p(x|c)$ 是 multimodal 的，expressivity 远强于 Gaussian。

Flow loss $\mathcal{L}_{flow}$ 对应于 $\mathbb{E}_{q(c|x)}[\log p(x|c)]$ 的一个 variational lower bound（通过 noise score matching）。Perceptual loss $\mathcal{L}_{perc}$ 是额外的 regularization，inject perceptual priors that the pure flow loss misses。

### 联想 9: Stage 2 Generation 的 Interplay

为什么更好的 tokenizer 没带来更好的 generation？

一个 hypothesis：FlowMo 的 mode-seeking 让 latent space 更 "compact"——每个 $c$ 的 $p(x|c)$ 更 concentrated。这对 reconstruction 好，但对 generation 可能不好——因为 generative model 需要 model $p(c)$，如果 $p(c)$ 的 manifold 更 complex（因为 encoder 学到了更 fine-grained 的 features），generative model 更难学。

另一个 hypothesis：MaskGiT 的 capacity 不足以 utilize FlowMo 的 richer latent space。Paper 只训了 300 epochs（SOTA generative model 训 1000-1080 epochs），所以 generative model 可能 undertrained。

这个 interplay 值得深入研究——可能需要更大的 generative model 和更长 training 来 fully exploit FlowMo 的 latent space。

### 联想 10: 和 DINOv2 / Self-Supervised Learning 的联系

FlowMo 的 encoder 学到的 features 是为了 help velocity prediction at all noise levels。这和 self-supervised learning（DINOv2, https://arxiv.org/abs/2304.07193）的目标不同——DINOv2 学的是 invariant features for recognition。

但两者有联系：都是 end-to-end 训一个 encoder，通过一个 proxy task（diffusion decoding vs contrastive/distillation）来 learn useful representations。FlowMo 的 encoder 可能学到了和 DINOv2 不同的、更 suitable for generation 的 features。

值得探索：FlowMo encoder 的 features 能否用于 downstream recognition tasks？如果能，说明 diffusion autoencoder 学到的 representations 也有 recognition value。

### 联想 11: Video Tokenization 的潜力

FlowMo 是 image tokenizer，但 architecture 是 transformer-based，天然 extendable to video。Video tokenization 的 challenge 是 temporal consistency——每帧的 reconstruction 要和相邻帧 consistent。

Diffusion autoencoder 的 multimodal $p(x|c)$ 在 video context 下更 problematic——如果每帧独立 sample mode，会 flicker。Mode-seeking 可能自然地 help temporal consistency，因为每帧都 seek 和原图 close 的 mode，而相邻帧的原图是 similar 的。

这可能是一个 promising 的 future direction。

### 联想 12: 和 Consistency Models 的关系

Consistency model (Song et al. 2023, https://arxiv.org/abs/2303.01469) 是 diffusion model 的 distillation 产物，能 1-step sample。如果 FlowMo 的 decoder 被 distill 成 consistency model，可以大幅加速 inference（25步 → 1步）。

Paper 提到 decoding 可以 distill 5-10x speedup，但没具体说用什么方法。Consistency model 是一个 natural choice。

但 distillation 可能影响 mode-seeking 的效果——consistency model 的 1-step sample 是否还能 maintain mode-seeking 的 sharpness？这需要实验验证。

---

## 总结：FlowMo 的核心贡献

1. **Conceptual**：mode-seeking insight——diffusion autoencoder 的 failure mode 是 averaging over modes，solution 是 active mode selection
2. **Methodological**：两阶段 training（mode-matching pre-training + mode-seeking post-training）+ shifted sampler
3. **Engineering**：thick-tailed noise schedule、perceptual loss on 1-step prediction、different perceptual networks for different stages、CFG in tokenizer、EDM2 weight normalization、μP parameterization
4. **Empirical**：ImageNet-1K 上多个 BPP regime 的 SOTA tokenization，scaling well-defined，generalize 到其他 dataset 和 resolution

---

## Reference Links

- FlowMo: https://kylesargent.github.io/flowmo
- MMDiT (SD3): https://arxiv.org/abs/2403.03206
- Rectified flow: https://arxiv.org/abs/2209.03003
- Flow matching: https://arxiv.org/abs/2210.02747
- Diffusion Autoencoders: https://arxiv.org/abs/2111.15640
- Yang & Mandt: https://arxiv.org/abs/2305.18231
- AlignProp: https://arxiv.org/abs/2310.03739
- DRAFT: https://arxiv.org/abs/2305.18781
- DDPO: https://arxiv.org/abs/2305.13301
- Classifier-free guidance: https://arxiv.org/abs/2207.12598
- Guidance intervals: https://arxiv.org/abs/2404.07724
- LFQ: https://arxiv.org/abs/2310.07141
- FSQ: https://arxiv.org/abs/2309.15505
- μP: https://arxiv.org/abs/2203.03466
- EDM2: https://arxiv.org/abs/2312.02696
- TiTok: https://arxiv.org/abs/2406.07550
- LlamaGen: https://arxiv.org/abs/2406.06525
- OpenMagViT-V2: https://arxiv.org/abs/2409.04410
- VQGAN: https://arxiv.org/abs/2101.08407
- MaskBit: https://arxiv.org/abs/2409.16211
- Rate-Distortion-Perception tradeoff: https://arxiv.org/abs/1906.06841
- DiTo: https://arxiv.org/abs/2501.18593
- FlexTok: https://arxiv.org/abs/2502.13967
- Consistency models: https://arxiv.org/abs/2303.01469
- DINOv2: https://arxiv.org/abs/2304.07193
- Large Scale GAN Training (truncation): https://arxiv.org/abs/1809.11096

---

# FlowMo: Mode-Seeking Diffusion Autoencoders 深度解析

## 1. Big Picture: 这篇paper想解决什么问题

当前SOTA的image tokenizer（如VQGAN, LlamaGen, OpenMagViT-V2, TiTok）几乎都遵循一个固定recipe: **CNN encoder + 2D spatially-aligned latent + adversarial loss + perceptual loss**。这套recipe虽然好用，但有四个痛点:

1. **Adversarial loss不稳定** - 需要LeCam regularization、adaptive gradient scale、careful loss weight tuning
2. **依赖CNN** - 难以享受transformer的scaling efficiency和well-understood scaling behavior
3. **依赖distillation** - TiTok要先从pretrained CNN tokenizer distill
4. **2D spatially-aligned latent** - 在大model scale时可能不够flexible

FlowMo想做的事情: 用**纯transformer-based diffusion autoencoder**，不用adversarial loss，不用distillation，不用CNN，不用2D spatial latent，达到SOTA。

Diffusion autoencoder其实不是新概念（Preechakul et al. CVPR 2022, SODA, Sample What You Can't Compress, Yang & Mandt NeurIPS 2023），但之前一直没在ImageNet-1K reconstruction上达到SOTA。FlowMo的核心贡献在于: **找到了让diffusion autoencoder真正work的关键insight**。

paper link: https://kylesargent.github.io/flowmo

---

## 2. 核心Insight: Mode-Seeking

这是全文最critical的一段。让我仔细剖析。

**问题的本质**: 给定一个quantized latent code c，由于c的信息量有限（比如256 tokens × 18 bits = 4608 bits），重建分布 p(x|c) 必然是multimodal的。比如一张模糊的人脸照片，c可能只编码了"这是一个微笑的白人女性，face在中间"，但具体皱纹、毛孔、头发丝的细节，c的信息不够specify，所以p(x|c)是multimodal的——有无数种皱纹、毛孔、头发丝的搭配都match这个c。

**传统diffusion autoencoder的做法**: 训练一个network去match the velocity field of p(x|c)。问题是，training用的是MSE-like loss on velocity，network会去学**所有modes的mixture**。Sampling时用ODE，由于ODE是deterministic的，最终sample到的是某种"average" mode——结果就是blurry、washed-out reconstruction。

**FlowMo的insight**: 对于perceptual reconstruction任务，我们care的是**reconstruction和original image的perceptual距离**。与其试图match p(x|c)的所有modes，不如主动**seek那些perceptually close to original image的modes**，drop掉其他modes。

这个insight的实现有两个ingredient:
1. **Mode-seeking post-training (Stage 1B)** - 直接backprop through sampling chain，optimization目标是sample的perceptual quality
2. **Shifted sampler** - sampling时timestep concentration towards low noise levels，bias towards mean of p(x|c)

为什么这两个design选择能work？直觉是:

- Stage 1A训练完，decoder学会了approximate the velocity field，但这个velocity field是mixture of all modes的average。Sampling时如果deterministically follow这个average velocity field，会去到modes之间的"valley"——即blurry region
- Stage 1B通过直接优化sample quality，强迫decoder学会**drop掉那些不perceptually relevant的modes**，把mass集中在perceptually close的modes附近
- Shifted sampler让大部分FLOPs花在low noise levels (t→0)，因为这是perceptual details决定性的阶段。High noise levels (t→1)可以take large steps，因为这时c的conditional signal最强，p(x|c)已经被strongly constrained，mode比较集中

这个insight对应rate-distortion-perception tradeoff (Blau & Michaeli ICML 2019, https://arxiv.org/abs/1906.06841)。FlowMo的精妙之处: 同时提升了rFID和PSNR（见Table 5），没有trade off，说明mode-seeking的mechanism比naive的likelihood-based方法更高效。

---

## 3. Architecture详解

基于MMDiT (Multimodal Diffusion Image Transformer, Stable Diffusion 3 architecture, https://arxiv.org/abs/2403.03206)，从Flux (https://github.com/black-forest-labs/flux) 改造。

### Encoder

输入: 
- Patchified image x ∈ R^n (n是patch数 × patch dim)
- Initial latent code c_0 ∈ R^d (全0向量)

Processing:
- Concatenate latent tokens和image tokens
- Self-attention across the concatenated sequence
- 但保持**两个modality的streams独立**（类似MMDiT的设计，hidden states分开但attention是shared的）

Output: 
- ĉ = e_θ(x, c_0)，即latent token sequence

### Quantization

用LFQ (Lookup-Free Quantization, https://arxiv.org/abs/2310.07141):
```
c = q(ĉ) = 2·1[ĉ ≥ 0] - 1
```

直觉: 把每个element的sign作为bit，所以每个element是1 bit。如果latent sequence length是256，token size是18（这里token size指的是一组bits的长度，可以factorize成9 bits×2或者18 bits×1），那codebook是2^18 = 262144。

为什么用LFQ而不是FSQ？Table 4 ablation: FSQ rFID=3.14 vs LFQ rFID=2.87。LFQ稍好，但commitment loss的处理更微妙——bounding commitment loss with tanh会degrade performance。

### Decoder

输入:
- Noised image x_t
- Quantized latent c
- Time/noise level t

Processing:
- 和encoder一样的MMDiT architecture，但接受额外的time parameter t
- t通过AdaLN modulation (https://arxiv.org/abs/2212.09748) condition每个MMDiT block
- Decoder比encoder**大且深**（Table 6: encoder depth=8, decoder depth=16; final configs decoder hidden=1152, encoder保持768）

输出: 
- v = d_θ(x_t, c, t)，velocity field

**关键设计**: encoder和decoder architecturally symmetric但differently sized。这是合理的——encoder只需要把image压缩成信息sufficient的code，decoder要做更复杂的generation task。

### μP parameterization

所有模型用μP (https://arxiv.org/abs/2203.03466)，目的是让hyperparameters能从small exploratory configs "transfer"到large scaled-up configs。这避免了在大模型上做hyperparameter search的高昂成本。

---

## 4. Stage 1A: Mode-Matching Pre-training

### Rectified Flow Formulation

核心公式 (Eq. 4):
$$x_t = t \cdot z + (1-t) \cdot x$$

变量解释:
- x: clean image, x ~ p_x
- z: pure Gaussian noise, z ~ N(0, I)
- t: noise level, t ∈ [0, 1], t=0对应clean data, t=1对应pure noise
- x_t: noised image at level t

注意这个和DDPM的formulation不同。DDPM是 x_t = √(α_bar_t)·x + √(1-α_bar_t)·ε，是signal和noise的weighted sum。Rectified flow是**linear interpolation** x_t = t·z + (1-t)·x，更简洁。

### Velocity field

Rectified flow学的是velocity field:
$$v(x_t, t) = \frac{dx_t}{dt} = z - x$$

从t=0到t=1，velocity指向"add more noise"的方向；从t=1到t=0，velocity指向"remove noise, go to data"的方向。

Decoder学的就是这个velocity:
$$v = d_\theta(x_t, c, t)$$

### L_flow loss

Eq. 5:
$$\mathcal{L}_{flow} = \mathbb{E}\left[\left\|x - z - d_\theta(x_t, q(e_\theta(x)), t)\right\|_2^2\right]$$

变量:
- x: clean image
- z: noise
- x_t = t·z + (1-t)·x: noised image
- e_θ(x): encoder output
- q(): quantization
- d_θ: decoder
- t: noise level

直觉: target是 x - z = -v（true velocity的负方向，即denoising direction）。让decoder output match这个target。这就是标准的flow matching loss (Lipman et al. ICLR 2023, https://arxiv.org/abs/2210.02747)。

### L_perc loss

Eq. 6:
$$\mathcal{L}_{perc} = \mathbb{E}\left[d_{perc}\left(x, x_t + t \cdot d_\theta(x_t, q(e_\theta(x)), t)\right)\right]$$

直觉: x_t + t·d_θ(...) 是从x_t走一步（步长t）后的image。在t=1时，x_1 = z (pure noise)，z + 1·d_θ(z, c, 1) = z + velocity = z + (x - z) = x。所以这其实是**1-step denoised prediction**的perceptual loss。

为什么需要这个loss？Yang & Mandt (NeurIPS 2023, https://arxiv.org/abs/2305.18231) 证明了diffusion autoencoder的end-to-end objective对应于一个modified variational lower bound with non-Gaussian decoder。但单靠flow loss，reconstruction质量不够sharp，需要perceptual loss来inject perceptual priors。

Ablation (Table 4): no perceptual loss → rFID从2.87崩到13.86！这是最catastrophic的degradation，证明这个loss是必要的。

### LFQ losses

Entropy loss (Eq. 7):
$$\mathcal{L}_{ent} = \mathbb{E}[H(q(\hat{c}))] - H(\mathbb{E}[q(\hat{c})])$$

直觉: 
- 第一项 E[H(q(ĉ))] 是average per-token entropy，鼓励每个token的codebook usage high entropy（即每个位置用多样化的code）
- 第二项 H(E[q(ĉ)]) 是batch-averaged code的entropy，鼓励batch level的code集中
- 合起来: 鼓励per-token diversity但batch-level concentration，即codebook被well-utilized

Commitment loss (Eq. 8):
$$\mathcal{L}_{commit} = \mathbb{E}\left[\|\hat{c} - q(\hat{c})\|_2^2\right]$$

直觉: 让continuous encoder output ĉ靠近它的quantized version q(ĉ)，梯度能从decoder flow back to encoder。

### Total Stage 1A loss

Eq. 9:
$$\mathcal{L}_{flow} + \lambda_{perc}\mathcal{L}_{perc} + \lambda_{commit}\mathcal{L}_{commit} + \lambda_{ent}\mathcal{L}_{ent}$$

Table 6 weights:
- λ_ent = 0.0025
- λ_commit = 0.000625 (注意比较小，相对于GAN-based tokenizer)
- λ_perc (lpips) = 0.1

有意思的细节: bounding commitment loss with tanh会degrade performance。这个比较反直觉——通常我们觉得bounded gradient更stable，但这里unbounded更好。

---

## 5. Stage 1B: Mode-Seeking Post-training

这是全文最novel的部分。

### Setup

- Freeze encoder e_θ
- Co-train decoder d_θ with L_flow和L_sample
- L_sample inspired by AlignProp (https://arxiv.org/abs/2310.03739) 和 DRAFT (https://arxiv.org/abs/2305.18781)

### Flow sample update function

Eq. 10:
$$d_{t_i}(x_t) = x_t + (t_{i+1} - t_i) \cdot d_\theta(x_t, c, t_i)$$

变量:
- d_{t_i}: 第i步sample update
- x_t: current noised image
- t_i: current noise level
- t_{i+1}: next noise level
- d_θ: decoder
- c: quantized latent

这是Euler method的update rule。从x_1=z (pure noise)开始，按照t_1, t_2, ..., t_n的顺序积分，最终得到x_0 ≈ reconstructed image。

### L_sample loss

Eq. 11:
$$\mathcal{L}_{sample} = \mathbb{E}\left[d_{perc}\left(x, d_{t_n} \circ d_{t_{n-1}} \circ \cdots \circ d_{t_1}(z)\right)\right]$$

直觉: 
- z: pure noise
- d_{t_1}(z): first Euler step
- d_{t_n} ∘ ... ∘ d_{t_1}(z): n步Euler积分后的sample (≈ reconstructed image)
- d_perc: perceptual distance between sample和original image x

关键: **整个sampling chain都differentiable**，loss能backprop through all n steps to update decoder d_θ。

这是和Stage 1A的L_perc最本质的区别:
- L_perc只监督1-step prediction（t·d_θ in Eq. 6）
- L_sample监督n步Euler积分后的最终sample

这个区别非常关键，Table 7的ablation:
- L_sample (n=8, weight=0.01): rFID = 1.28
- L_perc with ResNet (各种weights): 最好rFID = 1.60

直接optimizing the final sample quality远比optimizing 1-step prediction更effective。这背后的intuition: 我们care的是final sample的perceptual quality，不是1-step prediction的quality。1-step prediction可能很好但accumulated error让final sample变差；反之，optimizing final sample能避免accumulated error。

### Timestep sampling for L_sample

从supplementary:
$$t_i = \frac{\sum_{j=i}^{n} u_j}{\sum_{j=1}^{n} u_j}$$

其中 u_1, ..., u_n ~ Unif(0, 1)。

直觉: 这是randomized timestep spacing。每次training step用不同的spacing，让network generalize to various sampling schedules。这很重要，因为test time可能用不同的n和不同的spacing。

### Total Stage 1B loss

Eq. 12:
$$\mathcal{L}_{flow} + \lambda_{sample} \cdot \mathcal{L}_{sample}$$

λ_sample = 0.01（critical）。

为什么这个weight这么critical?
- Too small: network forgets perceptual features acquired in Stage 1A (rFID degrades)
- Too large: "reward hacking"——decoder overfits to d_perc metric，或training diverges

### Perceptual network choice

- Stage 1A: LPIPS-VGG
- Stage 1B: ResNet

为什么不同？MaskBit (https://arxiv.org/abs/2409.16211) 发现ResNet perceptual loss对tokenizer training effective，但FlowMo在Stage 1A用ResNet效果不好，LPIPS-VGG更好。Stage 1B用ResNet更好——可能是LPIPS-VGG的gradient在long sampling chain中不够informative。

### Training details for Stage 1B

- Batch size: 64 (reduced from 128 in Stage 1A)
- LR: 5e-5 (reduced from 1e-4)
- n = 8 (Euler steps during training)
- Gradient checkpointing + gradient accumulation
- Train approximately 1 epoch (both FlowMo-Lo and FlowMo-Hi)
- Early stopping to counteract reward hacking

为什么只train 1 epoch？因为computationally expensive（backprop through full chain）。但即使1 epoch，效果显著（Table 5: FlowMo-Hi rFID 0.73→0.56）。

---

## 6. Shifted Sampler

### Timestep spacing

Eq. 14:
$$(t_1, ..., t_n) = \left(\left(\frac{n}{n}\right)^\rho, \left(\frac{n-1}{n}\right)^\rho, ..., \left(\frac{1}{n}\right)^\rho\right)$$

变量:
- n: total number of sampling steps
- ρ: shift hyperparameter
- t_i: 第i个timestep

分析:
- ρ = 1: linear spacing, t_i = i/n。标准的rectified flow sampler
- ρ > 1: timestep concentration towards t=1（large steps near t=1, small steps near t=0）
- ρ → ∞: single large step at t=1, corresponds to regressing x given c (Chan et al. ICCV 2023, https://arxiv.org/abs/2301.07734)

FlowMo用ρ = 4。

### 为什么ρ = 4 work?

直觉:
- t→1 (high noise): c的conditional signal最强，p(x|c)被strongly constrained，modes集中。可以take large steps，不需要fine-grained sampling
- t→0 (low noise): 这是perceptual details决定性的阶段。Modes可能分散，需要small steps来carefully navigate到正确的mode

ρ = 4的effect: 大部分sampling FLOPs花在low noise levels，high noise levels用大步跨过。这bias towards "mean of p(x|c)"，即towards perceptually relevant modes，同时保持sufficient resolution in critical region。

Ablation (Table 4): unshifted sampler (ρ=1) → rFID 3.42 vs 2.87 baseline。

### Classifier-Free Guidance

FlowMo还能用classifier-free guidance (https://arxiv.org/abs/2207.12598) + guidance intervals (https://arxiv.org/abs/2404.07724):
- Training时dropout latent code c 10% of the time
- Inference时用guidance interval (0.17, 1.02) in EDM2 convention, CFG weight 1.5

直觉: guidance让sample更"confident"地靠近某个mode，避免average-out。Guidance interval限制了guidance的应用范围（不在所有timesteps都guidance），避免over-saturation。

Ablation: no guidance → rFID 3.28 vs 2.87。

---

## 7. Noise Schedule: 一个容易忽略但critical的细节

Stable Diffusion 3用logit-normal noise schedule for rectified flow training。但FlowMo发现这个schedule导致问题。

### Logit-normal的问题

Logit-normal density给t=0和t=1分配0 probability mass。这意味着network在t=1 (pure noise)时**没有training signal**。

但对FlowMo来说，t=1的estimate很critical:
- 当t=1时，x_t = z (pure noise)
- Velocity estimate d_θ(z, c, 1) 完全依赖于c
- 由于c的conditional signal极强，这个estimate对reconstruction quality至关重要

如果network在t=1缺乏training，会输出garbage velocity，导致后续sampling走偏。后果: PSNR collapse, **discoloration** (Figure 7的reconstructed image颜色不正常)。

### Thick-tailed logit-normal

FlowMo的fix: 10% of the time, sample t from uniform [0, 1] 而不是logit-normal。这给t=0和t=1分配了non-zero probability mass，确保network在extremes也有training signal。

Ablation (Table 4): logit-normal → rFID 4.08, PSNR 16.45 (vs 2.87, 20.71 baseline)。这是second worst ablation after no perceptual loss。证明这个细节极其critical。

---

## 8. 实验结果分析

### Table 1: Tokenization SOTA

| BPP | Model | Tokens | Vocab | rFID↓ | PSNR↑ | SSIM↑ | LPIPS↓ |
|-----|-------|--------|-------|-------|--------|--------|---------|
| 0.070 | OpenMagViT-V2 | 256 | 2^18 | 1.17 | 21.63 | 0.640 | 0.111 |
| 0.070 | **FlowMo-Lo** | 256 | 2^18 | **0.95** | **22.07** | **0.649** | 0.113 |
| 0.219 | LlamaGen-32 | 1024 | 2^14 | 0.59 | 24.44 | 0.768 | **0.064** |
| 0.219 | **FlowMo-Hi** | 1024 | 2^14 | **0.56** | **24.93** | **0.785** | 0.073 |

观察:
1. FlowMo在rFID, PSNR, SSIM三个指标上SOTA，多个BPP regime
2. 但LPIPS指标FlowMo稍逊色（FlowMo-Hi 0.073 vs LlamaGen-32 0.064）
3. 这个LPIPS劣势可能因为: FlowMo的mode-seeking策略倾向于sharp reconstructions of dominant modes，可能drop掉一些LPIPS-sensitive的fine details

### Table 2: Generation (MaskGiT)

| Tokenizer | FID↓ | IS↑ | sFID↓ | Prec.↑ | Rec.↑ |
|-----------|------|-----|--------|---------|--------|
| OpenMagViT-V2 | **3.73** | 241 | 10.66 | 0.80 | **0.51** |
| FlowMo-Lo | 4.30 | **274** | **10.31** | **0.86** | 0.47 |

观察:
1. FlowMo的FID稍差（4.30 vs 3.73），但Precision更好（0.86 vs 0.80）
2. Recall稍差（0.47 vs 0.51）
3. 这暗示一个interesting的tradeoff: tokenizer的mode-seeking让reconstruction更sharp（generation precision高），但可能牺牲了generation diversity（recall低）

paper里诚实地说: "There is an interesting and complicated interplay between tokenizer quality and generation quality"。这是open question。

### Table 3: vs DiTo (concurrent work)

| Model | rFID↓ | PSNR↑ | SSIM↑ | LPIPS↓ |
|-------|-------|--------|--------|---------|
| DiTo | 0.78 | 24.10 | 0.706 | 0.102 |
| FlowMo (continuous version) | **0.65** | **26.61** | **0.791** | **0.054** |

为了和DiTo (https://arxiv.org/abs/2501.18593) fair comparison，FlowMo做了continuous latent version。FlowMo全面碾压。

### Table 5: Stage 1B ablation (核心)

| Model | rFID↓ | PSNR↑ | LPIPS↓ |
|-------|-------|--------|---------|
| FlowMo-Lo (no post-train) | 1.10 | 21.38 | 0.134 |
| FlowMo-Lo (post-trained) | **0.95** | **22.07** | **0.113** |
| FlowMo-Hi (no post-train) | 0.73 | 24.02 | 0.086 |
| FlowMo-Hi (post-trained) | **0.56** | **24.93** | **0.073** |

观察:
1. Post-training让所有三个指标都提升！这不是trivial的——通常mode-seeking会trade rFID for PSNR
2. 这暗示mode-seeking和naive mode-matching的difference不在"tradeoff曲线上不同位置"，而在"完全不同的optimization landscape"

### C.1: Rate-Distortion-Perception Tradeoff分析

paper做了一个interesting的实验: 通过modulate x_1的norm来调整sample likelihood（类似truncated sampling in GANs）。结果: 只能tradeoff rFID和PSNR，不能同时提升两者。

这证明了FlowMo的mode-seeking不是简单的likelihood maximization，而是更深的mechanism。

### C.4: Mode-seeking的精确机制

通过Precision/Recall分析mode-seeking:

| Comparison | Prec.↑ | Rec.↑ |
|------------|--------|--------|
| vs val stats (1:1 correspondence), no post-train | 0.974 | 0.988 |
| vs val stats, post-trained | **0.993** | **0.991** |
| vs train stats (no correspondence), no post-train | 0.734 | 0.660 |
| vs train stats, post-trained | **0.766** | 0.634 |

观察:
1. vs val (1:1): post-training同时提升precision和recall。这说明mode-seeking让reconstructed distribution更接近true per-image distribution
2. vs train (non-correspondence): precision提升, recall下降。这是expected tradeoff——sacrificing global diversity for fidelity

直觉: mode-seeking让 p(x|c) 对于每个c都更concentrated around the true mode（即original image），所以1:1 correspondence下更好。但每个c的distribution变窄，意味着整个marginal p(x) = ∫p(x|c)p(c)dc的diversity下降。

---

## 9. Table 4 Stage 1A Ablation深度分析

| Variant | rFID↓ | PSNR↑ | LPIPS↓ |
|---------|-------|--------|---------|
| FlowMo (fewer params, baseline) | 2.87 | 20.71 | 0.15 |
| Doubled patch size | 6.39 | 19.94 | 0.17 |
| MSE-trained encoder | 3.82 | 21.40 | 0.15 |
| Without perceptual loss | **13.86** | 22.11 | 0.21 |
| FSQ quantization | 3.14 | 21.31 | 0.14 |
| Logit-normal schedule | 4.08 | 16.45 | 0.21 |
| Unshifted sampler | 3.42 | 20.25 | 0.16 |
| No guidance | 3.28 | 20.67 | 0.16 |

### 各个ablation的intuition

**Doubled patch size (patch 8 → 16)**: sequence length减少4x，model capacity不足。所有指标恶化。和Simpler Diffusion (https://arxiv.org/abs/2410.19324) 的发现一致: pixel-space diffusion需要small patch size。

**MSE-trained encoder**: 先用MSE训练encoder+regression decoder, freeze encoder features, 再train diffusion decoder。这种两阶段训练boost PSNR（21.40 vs 20.71），但rFID差（3.82 vs 2.87）。为什么？因为MSE-trained encoder features不necessarily contain velocity-prediction-relevant info at all noise levels。End-to-end training让encoder学会extract features that help velocity prediction across all t。

**Without perceptual loss (catastrophic)**: rFID从2.87崩到13.86。这印证了Yang & Mandt的发现: perceptual loss on 1-step denoised prediction是diffusion autoencoder的critical component。没有它，flow loss alone学不出perceptually sharp reconstructions。

**FSQ vs LFQ**: FSQ (https://arxiv.org/abs/2309.15505) 的pairwise metric稍好，但rFID差。FSQ的codebook是固定的grid，LFQ是binary sign，LFQ的rFID advantage可能来自更大的effective codebook expressivity。

**Logit-normal schedule**: 已经讨论过，导致discoloration。这是第二大degradation。

**Unshifted sampler (ρ=1)**: rFID 3.42 vs 2.87。证实了timestep concentration towards low noise levels的必要性。

**No guidance**: rFID 3.28 vs 2.87。Classifier-free guidance即使在tokenizer stage也有用，这是一个non-obvious的发现。

---

## 10. 与Concurrent Work比较

### DiTo (https://arxiv.org/abs/2501.18593)

- 也是diffusion autoencoder for image tokenization
- 但focus on **continuous** latent space
- FlowMo focus on **discrete** latent space，compare against SOTA discrete tokenizers

为了fair comparison, FlowMo做了continuous version，结果全面碾压（Table 3）。

### FlexTok (https://arxiv.org/abs/2502.13967)

- Diffusion tokenizer learned **on top of** a continuous VAE latent space
- VAE用perceptual + adversarial + reconstruction losses
- FlowMo**不依赖** auxiliary VAE，完全end-to-end

关键difference: FlexTok还依赖traditional VAE training recipe，FlowMo完全摆脱。

---

## 11. Training细节 (来自Supplementary)

### Hyperparameters (Table 6)

| Hyperparameter | FlowMo (fewer params) | FlowMo-Lo | FlowMo-Hi |
|----------------|----------------------|-----------|------------|
| Learning rate | 0.0001 | same | same |
| Batch size | 128 | same | same |
| Num. epochs | 40 | 130 | 80 |
| Hidden size (μP width) | 768 | 1152 | 1152 |
| Encoder patch size | 8 | 4 | 4 |
| Encoder depth | 8 | same | same |
| Decoder depth | 16 | same | same |
| Latent sequence length | 256 | same | same |
| Latent token size | 18 | same | same |
| Total params (×10^6) | 517 | 945 | 945 |

观察:
1. Final configs (FlowMo-Lo/Hi)只比exploratory config多train一些epochs和稍大hidden dim
2. FlowMo-Lo和FlowMo-Hi的架构**完全一样**，只BPP不同（通过vocab size和num tokens控制）
3. 945M params算是中等规模，没有overly large

### Optimizer

- Adam with (β1, β2) = (0.9, 0.95)
- Higher β2 unstable（可能bfloat16精度或长sequence length导致）
- Weight decay = 0
- EMA rate 0.9999
- bfloat16 precision
- Encoder LR set to 0 after 200K steps (encoder frozen relatively early)

### EDM2-style weight normalization

Forcibly normalize weight matrices in MLP blocks per step, following EDM2 (https://arxiv.org/abs/2312.02696)。Counteract exploding activations and weight matrices。

### Wall clock times (C.6)

| Model | Encode (s) | Decode (s) |
|-------|------------|------------|
| LlamaGen-32 | 0.021 | 0.038 |
| FlowMo-B | 0.050 | 2.391 |
| FlowMo-B (small) | 0.011 | 0.322 |

FlowMo的encoding speed和LlamaGen-32 comparable，所以不bottleneck generation。主要slowdown是diffusion decoder（2.391s vs 0.038s），但作者指出可以distill 5-10x speedup。

---

## 12. Scaling Behavior (C.3)

| μP width | # Params (×10^6) | rFID↓ | PSNR↑ | LPIPS↓ |
|----------|------------------|-------|--------|---------|
| 2 | 260 | 7.77 | 20.84 | 0.169 |
| 3 | 367 | 5.31 | 21.28 | 0.160 |
| 4 | 517 | 4.45 | 21.60 | 0.155 |
| 5 | 710 | 3.84 | 21.70 | 0.152 |

观察: 所有指标monotonically improve with width。这是transformer-based architecture的好sign——scaling well-defined。和CNN-based tokenizers的scaling behavior相比，更predictable。

---

## 13. Dataset Generalization (C.2)

FlowMo只在ImageNet-1K 256×256训练，但能generalize到:
- 更高resolution (512×512) - 通过patchwise diffuse-and-blend strategy (Hoogeboom et al. https://arxiv.org/abs/2305.18231)
- 其他dataset (OpenImages, Food-101)

Table 8结果显示: FlowMo-Hi在不同dataset和resolution上都strong performance，即使BPP比Cosmos DI-8x8低（Cosmos DI-8x8是0.249 BPP, FlowMo-Hi是0.219 BPP）。

---

## 14. Limitations和Open Questions

1. **Inference time**: 需要25步forward pass。可以distill但paper没做
2. **Generation metrics**: tokenizer quality更好但generation metrics没有全面更好。这个interplay需要更多研究
3. **LPIPS劣势**: FlowMo在LPIPS上稍逊色于一些baseline。可能因为mode-seeking的specific perceptual metric choice
4. **Training compute**: Stage 1B的backprop through full sampling chain很expensive，只train了1 epoch
5. **Longer training**: rFID在Stage 1A没有saturate，更长training可能进一步提升

---

## 15. 核心Take-aways for Building Intuition

1. **Diffusion autoencoder的failure mode是averaging over modes**: 训练时MSE-like loss让network去match mixture of all modes，sampling时deterministic ODE走到modes之间的valley
2. **Mode-seeking是关键**: 主动select perceptually relevant modes，drop irrelevant ones
3. **实现mode-seeking的两个ingredient**:
   - 直接optimizing final sample quality（不是1-step prediction）
   - Biased sampler spending FLOPs in critical low-noise region
4. **细节决定成败**:
   - Thick-tailed noise schedule (avoid logit-normal's 0 prob at t=1)
   - Perceptual loss on 1-step prediction in Stage 1A
   - Different perceptual networks for different stages
5. **Transformer-only + μP** allows clean scaling without CNN inductive bias
6. **Tokenizer quality ≠ generation quality**: 这是open question，值得深入研究

---

## 16. 个人思考

这篇paper有几个让我impressed的地方:

**Conceptual elegance**: mode-seeking这个insight非常clean。一旦理解了multimodal p(x|c)的问题，mode-seeking就是obvious的solution。但之前的工作没系统性地做这件事。

**Engineering honesty**: paper坦率承认generation metrics没有全面更好，并指出这是complicated interplay。没有over-claim。

**Ablation thoroughness**: Table 4, 5, 7的ablation都很informative。特别是Table 7证明了L_sample远胜过各种weight的L_perc，这是关键设计选择的strong justification。

**μP的使用**: 用μP让hyperparameter search在small model上做，然后transfer到large model。这是practical工程的好例子。

但也有一些值得探讨的points:

- **Why mode-seeking improves PSNR simultaneously with rFID?** Table 5显示这个simultaneous improvement，但paper没给出完全satisfying的theoretical explanation。是否mode-seeking让network学到更"faithful"的encoder representations？
- **Generation quality的mystery**: 为什么更好的tokenizer没带来更好的generation？可能因为generation model (MaskGiT)的capacity不足以utilize richer latent space？或者tokenizer的mode-seeking让latent space更难model？
- **Mode-seeking的generalization**: 这个insight是否能推广到其他diffusion model fine-tuning tasks？比如text-to-image的aesthetic improvement？

总之，FlowMo是一个elegant且effective的工作，给image tokenization带来新的paradigm。它的核心insight (mode-seeking for diffusion autoencoders) 可能会inspire更多工作——比如把这个idea应用到video tokenization、3D generation等领域。

---

## Reference Links

- FlowMo project page: https://kylesargent.github.io/flowmo
- MMDiT (SD3): https://arxiv.org/abs/2403.03206
- Rectified flow (Liu et al.): https://arxiv.org/abs/2209.03003
- Flow matching (Lipman et al.): https://arxiv.org/abs/2210.02747
- Flux: https://github.com/black-forest-labs/flux
- DiT (Peebles & Xie): https://arxiv.org/abs/2212.09748
- ViT: https://arxiv.org/abs/2010.11929
- Diffusion Autoencoders (Preechakul): https://arxiv.org/abs/2111.15640
- Yang & Mandt (lossy compression with diffusion): https://arxiv.org/abs/2305.18231
- Sample What You Can't Compress: https://arxiv.org/abs/2409.02529
- High-Fidelity Image Compression with Score-based Models: https://arxiv.org/abs/2305.18231
- AlignProp: https://arxiv.org/abs/2310.03739
- DRAFT: https://arxiv.org/abs/2305.18781
- DDPO: https://arxiv.org/abs/2305.13301
- Classifier-free guidance: https://arxiv.org/abs/2207.12598
- Guidance intervals: https://arxiv.org/abs/2404.07724
- LPIPS: https://arxiv.org/abs/1801.03924
- μP: https://arxiv.org/abs/2203.03466
- EDM2: https://arxiv.org/abs/2312.02696
- LFQ: https://arxiv.org/abs/2310.07141
- FSQ: https://arxiv.org/abs/2309.15505
- TiTok: https://arxiv.org/abs/2406.07550
- LlamaGen: https://arxiv.org/abs/2406.06525
- OpenMagViT-V2: https://arxiv.org/abs/2409.04410
- VQGAN: https://arxiv.org/abs/2101.08407
- ViT-VQGAN: https://arxiv.org/abs/2112.01527
- MaskGiT: https://arxiv.org/abs/2112.04126
- MaskBit: https://arxiv.org/abs/2409.16211
- Cosmos: https://arxiv.org/abs/2501.03575
- DiTo (concurrent): https://arxiv.org/abs/2501.18593
- FlexTok (concurrent): https://arxiv.org/abs/2502.13967
- Rate-Distortion-Perception tradeoff (Blau & Michaeli): https://arxiv.org/abs/1906.06841
- GeNVS (Chan et al.): https://arxiv.org/abs/2301.07734
- Simpler Diffusion (SiD2): https://arxiv.org/abs/2410.19324
- Hourglass Diffusion Transformers: https://arxiv.org/abs/2401.05152
- Improved Precision/Recall: https://arxiv.org/abs/1804.06991
- ADM (DDPM): https://arxiv.org/abs/2105.05233
