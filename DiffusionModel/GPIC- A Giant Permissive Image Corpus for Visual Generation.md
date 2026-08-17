---
source_pdf: GPIC- A Giant Permissive Image Corpus for Visual Generation.pdf
paper_sha256: 38fb0c838a67409875850037aff87341937c3c510adfdf571e792bf1c83df6e5
processed_at: '2026-08-04T22:07:51-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 GPIC 这件事

Andrej，我重新用更口语的方式讲一遍，把技术细节和 intuition 都融在叙事里。

---

## 故事的起点：为什么我们现在需要一个新的 image benchmark

过去十几年，做 image generation 的研究，大家都在同一个擂台上比武——**ImageNet-1K 256×256 class-conditional generation**。从 BigGAN 到 DiT 到 SiD2 到 VAR，所有 paper 的 headline number 都是 "FID on ImageNet-256"。

但这个擂台现在有两个大问题。

### 问题一：擂台和真实战场脱节了

现代真实的 image generation 模型，不管是 Stable Diffusion、PixArt、FLUX，还是 Midjourney、DALL-E，它们都不是 class-conditional 的。它们都是 **text-to-image**，输入是一段自由文本，输出是一张图。训练数据是几亿到几十亿的 web image-text pairs。

你在 ImageNet-1K 上做 class-conditional "生成 1000 类里的 golden retriever"，得出的结论——比如用多大的 CFG scale、用什么 tokenizer、怎么 schedule learning rate——很难直接搬到一个真实 T2I 模型上。这是一个 paradigm gap，擂台上练的招式在战场上用不上。

### 问题二：FID 已经被打穿了

FID 这个 metric 是 2017 年 Heusel et al. 提出的，用 Inception-v3 提 feature，然后算两个 distribution 之间的 Fréchet Distance。公式长这样：

$$\text{FID} = \|\mu_r - \mu_g\|_2^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$

变量含义：
- $\mu_r$: real images 的 feature 均值向量
- $\mu_g$: generated images 的 feature 均值向量
- $\Sigma_r$: real images 的 feature 协方差矩阵
- $\Sigma_g$: generated images 的 feature 协方差矩阵
- $\|\cdot\|_2^2$: L2 范数平方，衡量中心漂移
- $\text{Tr}(\cdot)$: 矩阵迹，衡量 spread 差异

这个 metric 本意是"生成图和真实图在 Inception feature space 里越像，FID 越低"。但过去几年，SOTA 模型的 FID 已经**低于** "holdout real ImageNet images vs. train ImageNet images" 的 FID 了。

这事儿荒谬在哪？你拿一批真实图、模型从没见过的真实图，算它们和 train set 的 FID，结果这个 FID 比 model 生成的图的 FID 还高。意思是模型生成的图在 Inception feature space 里比真实 holdout 图更像 train set。

这就是经典的 **Goodhart's law**：当一个 measure 变成 target，它就不再是 good measure 了。

为什么会这样？Inception-v3 是 supervised trained on ImageNet-1K 的，它的 feature space 和 ImageNet-1K label space 共线。模型 overfit 到 ImageNet-1K train set 的 Inception feature 流形里，生成的图就钻进了 train set feature 流形的"内部"，FID 自然低。这不是真的生成质量高，是 metric 被打穿了。

GPIC 这篇 paper 的核心 motivation 就是：**擂台该换了**。

参考链接：
- Goodhart's law 在 ML benchmark 的讨论: https://arxiv.org/abs/2403.18885
- Stein et al., "Exposing flaws of generative eval metrics", NeurIPS 2023: https://arxiv.org/abs/2311.16512
- FID 原始 paper (Heusel et al.): https://arxiv.org/abs/1706.08500

---

## GPIC 要做成什么样：四个 criteria

作者提出一个新的 benchmark 数据集必须同时满足四条：

1. **Permissive**: 每张图都有清晰 license，允许 commercial use 和 redistribution。用 trained model 不会被 license 污染，能直接 release weights。
2. **Stable**: 数据集 frozen 在某个状态，不能用 URL index（因为 link rot 会让半年后的复现性归零）。
3. **Large**: 100M images 量级，配 rich text captions，足够训现代 T2I model。
4. **Accessible**: 能直接 download tar shards streaming，不需要自己爬数据、不需要 memory-intensive resharding。

作者用 Table 1 评估了现有 dataset：

| Property | ImageNet-1K | YFCC100M | OpenImages | DataComp | **GPIC** |
|---|---|---|---|---|---|
| Permissive | ✗ | ✗ | ? | ✗ | ✓ |
| Stable | ? | ✗ | ? | ? | ✓ |
| Large | ✗ | ✓ | ✓ | ✓ | ✓ |
| Accessible | ? | ✗ | ✗ | ✗ | ✓ |

ImageNet-1K license 实际上不 permissive（research-only），YFCC100M 和 DataComp 都是 URL index（unstable），OpenImages license 模糊。没有一个全打勾的。GPIC 要做的就是全打勾。

参考链接：
- YFCC100M: https://multimediacommons.wordpress.com/yfcc100m-core-dataset/
- DataComp: https://arxiv.org/abs/2304.14108
- LAION-5B: https://arxiv.org/abs/2210.08402

---

## GPIC 是怎么造出来的：四阶段 pipeline

GPIC 从 110M 源图池最终清理出 100M clean 图。整个 pipeline 分四步。

### Stage 1: 收集源图

从 **Flickr** 和 **Wikimedia** 收图，只保留 **CC BY、CC0、Public Domain、No-Known-Restrictions** 四类 license 的图。最终 87.7% 来自 Flickr，12.3% 来自 Wikimedia，总 110,569,761 张。

为什么只选这两个 source？因为它们有 structured license metadata，能程序化验证 license。其他平台（如 Reddit、Pinterest）虽然图多，但 license 不清。

一个 engineering 细节：GPIC release 时不暴露 URL，只保留 attribution string、license name、license URL。这样既满足 attribution 要求，又避免 release 大规模 URL index 引发 privacy 问题。

### Stage 2: 过滤低质量和有害图

三层 cascade filter：

1. **Resolution / aspect ratio filter**: 移除极端分辨率、极端长宽比、longest side < 256px 的图。约移除 0.01%。
2. **VLM quality filter (Qwen3-VL-4B)**: 移除 near-blank、严重 blur、under/over-exposed 的图。约移除 0.3%。
3. **VLM safety filter (Qwen3-VL-4B)**: 移除 flagged unsafe 的图。约移除 0.35%。

为什么用 VLM 做 filter 而不是 CLIP-score 或 aesthetic predictor？CLIP-score 会偏好 caption-aligned 的图，丢掉很多视觉上有意义但 caption 难以描述的图（比如抽象艺术、纹理、纹理特写）。VLM 的语义判断更接近人眼。

参考链接：
- Qwen3-VL: https://arxiv.org/abs/2511.21631
- vLLM (用于大规模 VLM inference): https://arxiv.org/abs/2309.06180
- SGLang: https://arxiv.org/abs/2312.07104

### Stage 3: 去重——这是整个 paper 最 elegant 的工程部分

去重的核心矛盾：permissive license 数据很稀缺，**aggressive 去重会浪费宝贵数据**，但 conservative 去重又会留下 burst photography、repost、meme edit 这些 duplicates。

#### 用 SSCD features 做 similarity 比较

作者用 SSCD (Self-Supervised Copy Detection) feature 而不是 perceptual hash (pHash)。SSCD 是 Meta 提出的专门为 copy detection 训练的 self-supervised descriptor，对 JPEG compression、resize、crop、color jitter 鲁棒。直觉上，SSCD feature space 的几何结构是：**同一张图的所有视觉变体（resize、compress、crop）在 SSCD feature space 中接近共线**，而**视觉相关但内容不同的图有较高但不极端的 similarity**。

参考链接：
- SSCD paper: https://arxiv.org/abs/2202.10261

#### FAISS 近似最近邻搜索

110M images 的全量 pairwise comparison 是 O(N²) = 1.2 × 10^16，必须用 ANN search。FAISS 在 100M 量级的 SSCD feature (512 dim, L2-normalized) 上可以 tractable 跑完。

参考链接：
- FAISS: https://arxiv.org/abs/2401.08281

#### Power law extrapolation 估算——最漂亮的 trick

直接在 110M 上跑 dedup 很贵。作者先在 6 个小子集（108K → 3.4M）上跑 SSCD dedup，每个子集用 5 个 threshold θ ∈ {0.75, 0.80, 0.85, 0.90, 0.95}，然后拟合 power law 预测 full-scale removals：

$$D(N) = A \cdot N^{\beta}$$

变量含义：
- $D(N)$: 子集大小为 $N$ 时被移除的图片数量
- $N$: 子集大小（number of images）
- $A$: 拟合系数，反映基准 removal rate
- $\beta$: 幂指数，反映 removal rate 随数据规模如何增长

为什么是 power law？直觉是 duplicate 出现频率受"图像创作模态的长尾分布"驱动。同一个 scene 被不同人拍、上传、edit 的过程近似遵循 preferential attachment，所以 duplicates 在子集中的密度随 N 缓慢上升（sublinear power law）。

外推到 110M，θ=0.95 估计移除 9.62M images，剩余 101M。

#### Two-tier dedup rule

单一 threshold 会误杀很多视觉相关但不同的图（不同 pose、不同 viewpoint）。作者用两级规则：

1. **High-confidence duplicates**: SSCD cosine similarity > 0.9625 → 移除 lower-resolution image
2. **Repeated near-copy clusters**: 在 similarity > 0.90 构建的 graph 上，connected components 大小 ≥ 5 → 只保留 highest-resolution image

阈值靠 manual inspection 校准。Figure 5 展示了不同相似度区间内 nearest-neighbor pairs 的 qualitative 例子：
- 0.75-0.85: 类别相似（都是 cat），但视觉差异大
- 0.85-0.90: 同一 scene 的不同视角
- 0.90-0.95: 同一图的不同 crop / edit
- 0.95-0.9625: near-duplicate，但仍有可见差异
- >0.9625: high-confidence duplicate

最终保留 101.3M images，再用 SHA-256 hash 验证没有 exact duplicates。

### Stage 4: Captioning——不用 alt text，全部用 VLM 重 caption

这是 GPIC 区别于 YFCC100M 和 OpenImages 的另一个关键设计。Alt text 经常缺失、噪声大、和图像内容 weakly aligned，而且可能包含 PII 或 toxic 内容。VLM captioning 保证 caption quality 一致性。

#### 四种 caption 格式

| Format | Proportion | 特点 |
|---|---|---|
| Tag | 1% | 无序 keyword list |
| Short | 45% | 1-2 句话简洁描述 |
| Medium | 45% | 2-4 句话，含 spatial relations、attributes |
| Long | 9% | 完整 scene description，含 OCR、counting、fine-grained details |

caption distribution 的多模态性可以训练模型适应不同 inference 时的 prompt 风格。这避免了"模型只懂 detailed prompt，不会处理 short user query"的问题。

#### Captioning model 选择：microbenchmark + Pareto frontier

captioning 100M images 要在 quality 和 throughput 之间 trade-off。作者构造了 microbenchmark：1520 images, 720 short + 640 medium + 160 long captions。Human annotators refine VLM-generated captions 作为 reference。5 个 evaluation axes：overall summary, counting accuracy, spatial understanding, attribute binding, OCR。0-2 scale, LLM-as-a-judge。

Qwen3-VL-Instruct 4 个 scale 的 Pareto 比较：

| Model | Throughput (img/s, 1×H100) | 结论 |
|---|---|---|
| Qwen3-VL-2B | 最高 | 质量不够 |
| **Qwen3-VL-4B** | 56.10 (short) / 49.31 (medium) | **Best Pareto** |
| Qwen3-VL-8B | 中等 | throughput 折半 |
| Qwen3-VL-30B-A3B (MoE) | 最低 | summary 1.73 vs 4B 的 1.68，gain 不大 |

最终选 Qwen3-VL-4B-Instruct，1500 H100 hours caption 完整 corpus。

为什么不用 GPT-4V / Gemini / Claude？closed-source VLM 在 100M scale 不可行。粗估：100M images × $0.01/image = $1M，远超学术 budget。

参考链接：
- Qwen3-VL: https://arxiv.org/abs/2511.21631
- Qwen3 (text): https://arxiv.org/abs/2505.09388

#### Split construction

- 100M train / 200K val / 1M test
- 每个 split 都保持 Flickr vs Wikimedia 比例和 caption-type 分布
- 提供 nested tiers：GPIC-Nano (1M, 80 shards) / GPIC-Lite (10M, 800 shards) / GPIC-Full (100M, 8000 shards)
- 切换 tier 只需选 shard range，对 ablation 友好
- 打包成 8,000 个 tar shards (12.9 TB) on Hugging Face，每个 shard ~12,500 images，caption-type 比例平衡

---

## 新的 evaluation protocol：FD-DINOv2 替换 FID

GPIC 不只是 release 数据集，还重新设计了 evaluation protocol。

### 为什么用 DINOv2 features 而不是 Inception-v3

FID 用 Inception-v3 features，Inception-v3 是 supervised trained on ImageNet-1K，feature space 和 ImageNet-1K label space 共线，所以会被 Goodharted。

FD-DINOv2 用 DINOv2 ViT-L/14 features（1024-dim CLS token），DINOv2 是 self-supervised trained on LVD-142M，feature space 与 ImageNet-1K label space 不共线，更接近人眼感知。

Fréchet Distance 公式不变：

$$\text{FD} = \|\mu_r - \mu_g\|_2^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$

只换了 feature extractor。

参考链接：
- DINOv2: https://arxiv.org/abs/2304.07193
- Stein et al., "Exposing flaws of generative eval metrics": https://arxiv.org/abs/2311.16512

### Oracle references：benchmark saturation 的标尺

这是 GPIC evaluation 设计中最聪明的部分。作者计算了 real-vs-real FD 作为 reference point：

| GPIC Subset | FD↓ | Precision↑ | Recall↑ | Density↑ | Coverage↑ |
|---|---|---|---|---|---|
| Full (100M) | 1.19 | 0.947 | 0.950 | 1.000 | 0.972 |
| Lite (10M) | 1.25 | 0.951 | 0.947 | 1.010 | 0.973 |
| Nano (1M) | 1.60 | 0.946 | 0.946 | 1.002 | 0.968 |
| Val (200K) | 2.37 | 0.948 | 0.949 | 0.993 | 0.966 |
| **Test-50K** | **7.44** | 0.949 | 0.953 | 0.997 | 0.967 |

**Test-50K vs Test-1M 的 FD = 7.44** 这个数字是 saturation 的下界。当模型 FD 接近或低于 7.44 时，benchmark 开始 saturate。

### 多维度 metrics

GPIC 不只看 FD，还报告：
- **Precision**: fidelity，generated samples 有多少落入 real manifold
- **Recall**: diversity，real samples 有多少能被 generated manifold 覆盖
- **Density**: fidelity 的连续版，不 upper bound by 1
- **Coverage**: diversity 的连续版
- **MMD**: non-parametric alternative to FD，不需要 Gaussian 假设

参考链接：
- Improved Precision & Recall (Kynkäänniemi et al.): https://arxiv.org/abs/1904.06991
- Assessing generative models via precision and recall (Sajjadi et al.): https://arxiv.org/abs/1806.00035
- Reliable fidelity and diversity metrics (Naeem et al.): https://arxiv.org/abs/2002.09797

### Anti-gaming policy

GPIC protocol 明确禁止用 DINOv2 features 作为 training loss，因为这会直接优化 metric 的 feature space，构成 metric-specific optimization 而非 generative capability improvement。同时鼓励披露是否用了 larger auxiliary models（如 DINOv3、SigLIP），因为这些 model 见过的 data 远超 GPIC，会造成 unfair comparison。

直觉：**evaluation feature space 的几何不能 leak 进 training objective**。一旦 leak，模型直接学 feature manifold 的低维投影，generation quality 没真的提升，FD 数字却在 drop。这就是 FID 被 Goodharted 的根本机制。

### Held-out test set vs. training set

ImageNet-1K FID 的另一个问题是 reference statistics 算在 train set 上，模型可以 memorize train set 的 feature 分布获得低 FID，holdout 检测不出 overfitting。

GPIC 把 reference statistics 算在 **1M held-out test set** 上，generated samples 用 50K fixed test captions 生成。如果模型 memorize train set，FD 会暴露（因为 test set 的 feature 分布和 train set 不同）。

---

## Reference baseline: JiT-T2I pixel-space flow matching

GPIC 不只 release dataset，还训练了一个 reproducible baseline：JiT-T2I，1.1B params，用 Qwen3-1.7B 做 text encoder。

### 为什么选 pixel-space flow matching 而不是 latent diffusion

1. **No tokenizer pretraining**: latent diffusion 需要先训 VAE，引入额外 hyperparameter 和 compute。pixel-space 直接 end-to-end。
2. **No auxiliary losses**: VAE 通常配 KL loss、perceptual loss、LPIPS。pixel-space 简化到 single flow matching objective。
3. **Single-stage training**: reproducibility 更好。

代价是 compute cost 更高，所以 baseline 只训 1 epoch on 256×256。

### Flow matching 的目标函数

给定 noise $x_0 \sim \mathcal{N}(0, I)$ 和 data $x_1 \sim p_{\text{data}}$，定义 flow：

$$x_t = (1-t) x_0 + t x_1$$

变量含义：
- $x_t$: 在 time $t$ 的中间状态
- $t$: flow 的时间参数，$t=0$ 是 noise，$t=1$ 是 data
- $x_0$: 标准正态噪声
- $x_1$: 真实图像

对应 velocity field 是 $u_t = x_1 - x_0$。训练目标：

$$\mathcal{L}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,1],\, x_0 \sim \mathcal{N}(0,I),\, x_1 \sim p_{\text{data}},\, c} \left[ \left\| v_\theta(x_t, t, c) - (x_1 - x_0) \right\|_2^2 \right]$$

变量含义：
- $v_\theta$: 神经网络预测的 velocity field
- $\theta$: 网络参数
- $c$: conditioning signal (text caption)
- $x_1 - x_0$: 目标 velocity（从 noise 指向 data 的直线方向）

直觉：flow matching 学一个 vector field，让所有从 noise 出发的"流线"汇聚到 data manifold。和 DDPM 的区别是，flow matching 用 linear interpolation path（直线），trajectory 更短、更可解释、更易用 ODE solver。

参考链接：
- Flow Matching (Lipman et al., ICLR 2023): https://arxiv.org/abs/2210.02747
- Rectified Flow (Liu et al.): https://arxiv.org/abs/2209.03003
- JiT (Li & He, 2025): https://arxiv.org/abs/2511.13720
- PixelGen: https://arxiv.org/abs/2602.02493

### Classifier-free guidance 的 flow matching 版本

inference 时用 CFG：

$$\hat{v} = v_\theta(x_t, t, c) + w \cdot \left( v_\theta(x_t, t, c) - v_\theta(x_t, t, \varnothing) \right)$$

变量含义：
- $\hat{v}$: guidance-adjusted velocity
- $v_\theta(x_t, t, c)$: conditional velocity（带 text）
- $v_\theta(x_t, t, \varnothing)$: unconditional velocity（null text）
- $w$: guidance scale

当 $w > 0$，强化 conditional direction 远离 unconditional direction。$w$ 太小生成质量低，$w$ 太大则 mode collapse、diversity 下降。

### 实验结果

JiT-T2I (1.1B params) 在 GPIC-Full 训 1 epoch, 256×256, batch size 256, AdamW lr=1e-4, betas (0.9, 0.95), constant schedule with 0.1% warmup, 40 hours on 8×H100。

| CFG | FD↓ | Precision↑ | Recall↑ | Density↑ | Coverage↑ |
|---|---|---|---|---|---|
| 1.75 | 204.01 | 0.917 | 0.530 | 1.034 | 0.806 |
| 4.00 | 87.80 | 0.933 | 0.765 | 1.012 | 0.906 |
| **6.25** | **76.25** | 0.942 | 0.792 | 1.014 | 0.908 |

观察：
- FD 随 CFG scale 增大单调下降（204 → 76），CFG 提升 fidelity 显著
- Recall 也单调上升（0.530 → 0.792），在此 regime 内 CFG 没引起 mode collapse
- CFG=6.25 在 ImageNet-1K 上偏高（常见 1.5-3.0），但在 GPIC 上更高才 optimal，因为 text-conditioned generation 比 class-conditioned 需要 stronger guidance align 文本语义

距离 saturation (Test-50K vs Test-1M FD=7.44) 还有 10x 空间，说明 GPIC benchmark 还有大量爬坡空间。

---

## 这个 paper 的几个深层直觉

### 1. Evaluation feature space 必须独立于 training signal

任何评测器如果和 training loss 在同一个 representation space，都会被 Goodharted。这不只影响 image generation。在 LLM eval 里，MMLU/HumanEval 等 benchmark 的 distribution 和 training distribution 重叠越多，benchmark 寿命越短。GPIC 用 self-supervised DINOv2 feature（与 ImageNet-1K label space 不共线）延长 benchmark 寿命。

### 2. Permissive data 的稀缺性

power law extrapolation 显示 θ=0.95 在 110M source pool 上只能去 9.6M duplicates，相对保守的 8.7% 去重率。对比 LAION-5B 这种 unrestricted license pool，dedup 移除率可以到 30-50%。这说明 CC BY / CC0 数据已经稀缺，aggressive dedup 会浪费宝贵数据。

CC license 是 2002 年提出的，Flickr 用户上传图基本 default CC BY-NC。后来 Flickr 改 default，但 CC BY 数据存量有限。如果 community 想持续做 open generative model，需要新的 permissive data production 机制。

### 3. Pixel-space 是否会回潮

从 2022 年 LDM/Stable Diffusion 开始，latent-space diffusion 几乎是 industry consensus，因为 compute efficiency 巨大。但 2024-2025 出现反向 trend：

- **SiD2** (Hoogeboom et al., 2024) 在 ImageNet-512 达到 1.5 FID with pixel-space diffusion，beat latent methods
- **PixelGen** (Ma et al., 2026) 用 perceptual loss 让 pixel diffusion beat latent diffusion
- **JiT** (Li & He, 2025) 简化 pixel diffusion

直觉：VAE 的 reconstruction error 是 latent diffusion 的 fundamental ceiling。一旦 model scale 和 data scale 大到某个 threshold，VAE bottleneck 变成限制因素，pixel-space 反而能学得更好。

参考链接：
- SiD2: https://arxiv.org/abs/2410.19324
- Reconstruction vs. Generation: https://arxiv.org/abs/2501.01423
- VAR (next-scale prediction): https://arxiv.org/abs/2404.02905

### 4. FD-DINOv2 的 residual risks

虽然 FD-DINOv2 暂时解决 Inception-v3 的 saturation 问题，但 DINOv2 也可能在 LVD-142M 上见过 GPIC test images（因为 LVD-142M 是 web crawled），存在 implicit leakage。如果 community 直接用 DINOv2 features 做 training loss，FD-DINOv2 也会被打穿。

更 robust 的方向可能是 self-supervised feature trained 在 controlled corpus 上（例如只在 GPIC train set 上 trained 的 SSL model）。这是 chicken-and-egg：先有 GPIC 才能训 controlled SSL model，反过来才能 stable eval。这是值得后续工作探索的方向。

### 5. DINOv2 backbone size 和 register tokens 对 FD 的影响

Appendix B.3 ablate 了 4 个 backbone (ViT-S/B/L/g) × 2 (with/without registers)。结论：
- Small/Base/Large backbone 上 with-registers 比 without-registers FD 更低
- Giant backbone 上两者接近
- 8 个配置的 Pearson correlation = 0.847，Kendall's W = 0.795

所以作者用 default = DINOv2 ViT-L/14 without registers，与 Stein et al. 2023 一致。

Register tokens 是 Darcet et al. 提出的，用于减少 ViT high-norm artifacts。Register 改变 feature 分布，对 FD 绝对值有影响。相对排序仍然 stable。

参考链接：
- Vision Transformers Need Registers: https://arxiv.org/abs/2309.16588

### 6. Image preprocessing 的一致性

Appendix B.1 专门提到用 Pillow bicubic downsample，不同 Python 库（PIL vs OpenCV vs torchvision）的 bicubic kernel 不同，会导致 ±0.5 FID 差异。在饱和 benchmark 上这是 noise floor 量级，会让结果不可复现。

参考链接：
- Parmar et al., "On aliased resizing and surprising subtleties in GAN evaluation": https://arxiv.org/abs/2104.11222

### 7. Caption format distribution 的设计

1% / 45% / 45% / 9% 这个 mix 值得讨论。Long caption 占比小可能因为：
- Long caption token 长，KV cache 大，throughput 低
- Long caption 在真实 user prompt distribution 里占比小
- 留 9% 训练模型处理 detailed prompt 的能力即可

但这也意味着 long caption 的样本 diversity 不够，模型在 long prompt 上的 OOD 性能可能受限。后续工作可以 ablate 这个 mix。

### 8. GPIC-Nano (1M) 作为 ImageNet-1K 替代品的潜力

ImageNet-1K train set 1.28M images，GPIC-Nano 1M images，scale 可比。但 GPIC-Nano 是 text-conditioned，caption 多样，没有 class label。这能否成为下一个十年的 ImageNet-1K？

直觉：能，但需要时间。需要 reference method 在 GPIC-Nano 上建立 Pareto frontier（不同 param count / training FLOPs 的 model），让 community 有可比 baseline。GPIC paper 给的 JiT-T2I baseline 是 1.1B 模型，太大了，nano benchmark 还需要 50M-200M scale 的小 baseline。

### 9. Broader impact 的诚实

paper 最后一句说 "despite our deduplication efforts, some near-duplicates may remain in GPIC, although their prevalence is estimated to be small"。任何 100M scale 的 dataset 都会有 noise，关键是 noise rate 和影响是否可控。GPIC 通过 SSCD similarity inspection + power law extrapolation + conservative threshold 来 bound duplicate rate，是 reasonable 的工程妥协。

---

## 一句话总结

GPIC 是 Stanford 团队针对 ImageNet-1K benchmark 被打穿、open research 无法 reproducible、permissive license data 稀缺这三个问题，造的一个 100M 规模、license 清晰、frozen tar shard hosted on Hugging Face 的新 benchmark 数据集。技术上最 elegant 的部分是 power law extrapolation 估算 dedup removals，以及 FD-DINOv2 + held-out test set + anti-gaming policy 组合解决 evaluation leakage。JiT-T2I baseline 提供了 reproducible 起点，但离 saturation 还有 10x 空间，说明 GPIC 是面向未来 3-5 年的 benchmark，不是 2026 年的 SOTA。

---

## 相关工作的 web links

- GPIC project page: https://gpic.stanford.edu
- DINOv2: https://arxiv.org/abs/2304.07193
- DINOv3: https://arxiv.org/abs/2508.10104
- SigLIP: https://arxiv.org/abs/2303.15343
- SSCD: https://arxiv.org/abs/2202.10261
- FAISS: https://faiss.ai
- vLLM: https://arxiv.org/abs/2309.06180
- SGLang: https://arxiv.org/abs/2312.07104
- Qwen3-VL: https://arxiv.org/abs/2511.21631
- Qwen3: https://arxiv.org/abs/2505.09388
- DataComp: https://arxiv.org/abs/2304.14108
- LAION-5B: https://arxiv.org/abs/2210.08402
- YFCC100M: https://multimediacommons.wordpress.com/yfcc100m-core-dataset/
- OpenImages: https://storage.googleapis.com/openimages/web/index.html
- ImageNet: https://image-net.org
- DiT: https://arxiv.org/abs/2212.09748
- SiD2: https://arxiv.org/abs/2410.19324
- VAR: https://arxiv.org/abs/2404.02905
- Flow Matching: https://arxiv.org/abs/2210.02747
- Rectified Flow: https://arxiv.org/abs/2209.03003
- JiT: https://arxiv.org/abs/2511.13720
- PixelGen: https://arxiv.org/abs/2602.02493
- Improved Precision & Recall: https://arxiv.org/abs/1904.06991
- Reliable fidelity and diversity metrics: https://arxiv.org/abs/2002.09797
- FID: https://arxiv.org/abs/1706.08500
- Inception-v3: https://arxiv.org/abs/1409.4842
- Vision Transformers Need Registers: https://arxiv.org/abs/2309.16588
- Exposing flaws of generative eval metrics: https://arxiv.org/abs/2311.16512
- img2dataset: https://github.com/rom1504/img2dataset
- Wan: https://arxiv.org/abs/2503.20314
- Sora: https://openai.com/research/video-generation-models-as-world-simulators
- Parmar et al. aliased resizing: https://arxiv.org/abs/2104.11222

---

# GPIC: 一个为视觉生成时代重新设计的大规模 Permissive 图像基准

Andrej 你好，这篇 GPIC 我读完后第一感觉是，它做的事情看起来平淡——又是一个 image-text dataset——但仔细想其实是对当前 generative image modeling 评测生态的一次结构性修正。它要修的不只是"数据集不够大"的问题，而是"benchmark 已经被 Goodharted"以及"open research 无法 reproducible"这两个更隐蔽的问题。下面我从动机、pipeline、benchmark protocol、baseline 实验四个层面来拆解，并尽量把直觉和细节都铺开。

---

## 1. 为什么需要重新做这件事：ImageNet-1K benchmark 的"死亡螺旋"

过去十年，class-conditional ImageNet-1K generation (256×256) 是事实上的标准 benchmark，从 BigGAN → VQVAE → VQGAN → DiT → SiD2 → VAR 全部都在这个坐标系上爬坡。但作者指出两个关键问题：

### 1.1 Distribution shift: 现代 generative model 已经不在 class-conditional 范式下了

现代 T2I 模型（Stable Diffusion、PixArt、FLUX、Sora、Wan 等）几乎都是 free-form text conditioned，训练语料是几百 M 到几个 B 的 web image-text pairs。在 ImageNet-1K 上做 class-conditional generation 得到的结论（架构选择、guidance scale、tokenizer、loss 设计）已经很难迁移到实际 production T2I 设定。这是一个 paradigm gap。

### 1.2 Goodharting: FID 已经被"打穿"了

这是论文 Figure 9 想表达的直觉。在 ImageNet-1K 上，多个 SOTA 模型的 FID **低于** "held-out real ImageNet images vs. ImageNet train set" 的 FID。换句话说，模型生成的图像在 Inception feature space 里看起来比真实的 holdout 图像更像 train set。这本身就是一个清晰的 Goodhart's law 案例：

> "When a measure becomes a target, it ceases to be a good measure."

为什么 FID 会被打穿？我的理解是：
- FID 用 Inception-v3 features，而 Inception-v3 本身就在 ImageNet-1K 上 supervised-trained。feature space 的几何结构对 ImageNet-1K train set 高度敏感。
- 当模型 overfit 到 ImageNet-1K train set 的 Inception-feature 流形上，generated images 落在 train set 的 feature 流形内部，于是 ||μ_g - μ_r|| 很小、Tr(Σ_r + Σ_g - 2(Σ_rΣ_g)^{1/2}) 也很小。
- Holdout real images 因为是 ImageNet-1K 训练分布外的样本，离 train set 的 feature 流形中心更远，反而被模型"超过"了。

这是一个非常深刻的教训：**evaluation feature space 不能和 training distribution 的 supervised feature space 共线**。否则 evaluation metric 的"信号"会被 training objective 系统性吸收。

GPIC 提出用 **FD-DINOv2**（Fréchet Distance over DINOv2 features）来替代，因为 DINOv2 是 self-supervised trained，feature space 与 ImageNet-1K 的 label space 不共线，更接近人眼感知。

参考链接：
- Stein et al., "Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models", NeurIPS 2023: https://arxiv.org/abs/2311.16512
- DINOv2 (Oquab et al., 2023): https://arxiv.org/abs/2304.07193
- 关于 Goodhart's law 与 ML benchmark 的讨论: https://arxiv.org/abs/2403.18885

---

## 2. 数据集设计：四个 criteria 的结构性论证

GPIC 提出 benchmark 数据集要同时满足 **Permissive、Stable、Large、Accessible** 四个 criteria。论文 Table 1 把 ImageNet-1K、YFCC100M、OpenImages、DataComp 都按这四个 criteria 打分，结论是它们都"瘸腿"：

| Property | ImageNet-1K | YFCC100M | OpenImages | DataComp | **GPIC** |
|---|---|---|---|---|---|
| Permissive | ✗ | ✗ | ? | ✗ | ✓ |
| Stable | ? | ? | ? | ? | ✓ |
| Large | ✗ | ✓ | ✓ | ✓ | ✓ |
| Accessible | ? | ✗ | ✗ | ✗ | ✓ |

让我把这四个 criteria 的工程含义讲清楚：

### 2.1 Permissive

每张 image 必须有已知 license 且允许 commercial use 和 redistribution。这是 GPIC 与 LAION-5B / DataComp 的核心区别——后者是 URL index 或者非 permissive license mix。GPIC 只收 **CC BY、CC0、Public Domain、No-Known-Restrictions** 四类 license。

这个 design choice 的工程后果是：**derivative artifacts（即 trained model weights）可以自由 release，不会被 license 污染**。当前很多企业模型无法开源就是因为训练数据 license 不清。

### 2.2 Stable

数据集必须 frozen 在某个 fixed state，否则 link rot 会让 6 个月后的复现性归零。LAION-5B、YFCC100M 都是 URL index，image hosting sites 经常会 404、改图、被删除。论文把 GPIC 打包成 8,000 个 tar shards (12.9 TB) centrally hosted on Hugging Face，这样 snapshot 永久不变。

这里有一个 subtle 的点：dataset drift 不只是 URL rot，还包括 source website 改图（例如有人在 Flickr 重新编辑了图）。Frozen tar shard 同时规避了这两种 drift。

### 2.3 Large

GPIC-Full 100M training images，平均 height 479px、width 587px，total 28 trillion pixels。是 ImageNet-1K train set 的 ~88x，足够训练 modern T2I baseline。同时提供 nested tiers：

- **GPIC-Nano**: 1M images (80 shards)
- **GPIC-Lite**: 10M images (800 shards)
- **GPIC-Full**: 100M images (8000 shards)

三层是嵌套关系，切换只需选择 shard range，这对 ablation 实验和 scaling law study 都很友好。

### 2.4 Accessible

不需要自建 crawler（img2dataset），不需要 memory-intensive resharding（YFCC100M 的 parquet 格式对 streaming training 不友好）。GPIC 用 webdataset 风格的 tar shards，可以直接用 torch DataLoader 或 HF datasets 的 streaming mode 拉取。

---

## 3. Pipeline 细节：从 110M source pool 到 100M clean corpus

GPIC 的 pipeline 分四个 stage，每个 stage 都有可学习的工程细节。

### 3.1 Stage 1: Source pool

110,569,761 images，其中 87.7% Flickr + 12.3% Wikimedia。这两个 source 的选择是因为它们都有 structured license metadata。最终 release 不暴露 URL，只保留 attribution string、license name、license URL 作为 metadata。

### 3.2 Stage 2: Image filtering

三层 cascade：

1. **Resolution / aspect ratio filter**：移除极端分辨率和极端长宽比的图（约 0.01% 移除）。同时丢弃 longest side < 256px 的图。
2. **VLM quality filter (Qwen3-VL-4B)**：移除 near-blank、严重 blur、under/over-exposed 图（约 0.3% 移除）。
3. **VLM safety filter (Qwen3-VL-4B)**：移除 flagged unsafe 图（约 0.35% 移除）。

这里有个工程直觉：用 VLM 做 filter 比 CLIP-score 或 aesthetic predictor 更"通用"。CLIP-score 偏好 caption-aligned 的图，会丢掉很多视觉上有意义但 caption 难以描述的图（如抽象艺术、纹理）。Qwen3-VL 的语义判断更接近人眼。

参考链接：
- Qwen3-VL technical report: https://arxiv.org/abs/2511.21631
- vLLM (用于大规模 VLM 推理): https://arxiv.org/abs/2309.06180
- SGLang: https://arxiv.org/abs/2312.07104

### 3.3 Stage 3: Deduplication —— 这是整篇 paper 我觉得最有意思的部分

去重的核心矛盾是：**aggressive dedup 会浪费 permissive license 数据，而 conservative dedup 又会留下 burst photography 和 repost**。作者选择用 copy detection features 而不是 perceptual hash（pHash）来做。

#### 3.3.1 SSCD features

SSCD = Self-Supervised Copy Detection，是 Meta 提出的一种专门为 image copy detection 训练的自监督 descriptor。它基于 ResNet50 + SimCLR-style self-supervised learning，但加了 L2 normalization 和 copy detection loss。SSCD feature 的优点是对 JPEG compression、resize、crop、color jitter 等常见 image transformation 鲁棒。

直觉上，SSCD feature space 的几何结构是：**同一张图的所有视觉变体（resize、compress、crop）在 SSCD feature space 中接近共线**，而**视觉相关但内容不同的图（不同视角、不同 pose）有较高但不极端的相似度**。

参考链接：
- SSCD paper (Pizzi et al., CVPR 2022): https://arxiv.org/abs/2202.10261

#### 3.3.2 FAISS approximate nearest-neighbor search

110M images 上的全量 pairwise comparison 是 O(N²) = 1.2 × 10^16，必须用 ANN。FAISS 的 IVF-PQ 或 IVF-HNSW 可以在合理时间内完成。论文没明说用了哪个 index，但对 100M 量级的 SSCD feature (512 dim, L2-normalized)，常见做法是 IVF4096,PQ32 或 IVF16384 + HNSW refinement。

参考链接：
- FAISS: https://arxiv.org/abs/2401.08281

#### 3.3.3 Power law extrapolation（这是最漂亮的工程 trick）

直接在 110M 上跑 dedup 很贵，作者先在 6 个小子集（108K → 3.4M）上跑 SSCD dedup，每个子集跑 5 个 threshold θ ∈ {0.75, 0.80, 0.85, 0.90, 0.95}，然后拟合 power law 预测 full-scale removals：

$$D(N) = A \cdot N^{\beta}$$

变量含义：
- $D(N)$: 在子集大小为 $N$ 时被移除的图片数量
- $N$: 子集大小
- $A$: 拟合系数，反映基准 removal rate
- $\beta$: 幂指数，反映 removal rate 随数据规模 sublinear/superlinear 增长

为什么是 power law？我的直觉是：duplicate 出现的频率受"图像创作模态的长尾分布"驱动。同一个 scene 被不同人拍、上传、edit 的过程近似遵循 preferential attachment，所以 duplicates 在子集中的密度会随 $N$ 缓慢上升（sublinear power law）。

外推到 110M，θ=0.95 估计移除 9.62M images，剩余 101M。最终论文用更 conservative 的 two-tier rule 实际保留 101.3M images。

#### 3.3.4 Two-tier dedup rule

直觉是：单一 threshold 会误杀很多视觉相关但不同的图（不同 pose、不同 viewpoint）。所以用两级规则：

1. **High-confidence duplicates**: SSCD cosine similarity > 0.9625 → 移除 lower-resolution image
2. **Repeated near-copy clusters**: 在 similarity > 0.90 构建的 graph 上，connected components ≥ 5 → 只保留 highest-resolution image

阈值校准靠 manual inspection。Figure 5 展示了不同相似度区间内 nearest-neighbor pairs 的 qualitative 例子。直觉上：
- 0.75-0.85: 类别相似（都是 cat），但视觉差异大
- 0.85-0.90: 同一 scene 的不同视角
- 0.90-0.95: 同一图的不同 crop / edit
- 0.95-0.9625: near-duplicate，但仍有可见差异
- >0.9625: high-confidence duplicate

最后用 SHA-256 hash over image bytes 验证没有 exact duplicates。

### 3.4 Stage 4: Captioning

这是 GPIC 区别于 YFCC100M 和 OpenImages 的另一个关键设计：**不用 source metadata 或 alt text，全部用 VLM 重新 caption**。原因：
- Alt text 经常缺失、噪声大、和图像内容 weakly aligned
- Alt text 可能包含 PII 或 toxic 内容
- VLM captioning 保证 caption quality 一致性

#### 3.4.1 四种 caption 格式

| Format | Proportion | 特点 |
|---|---|---|
| Tag | 1% | 无序 keyword list |
| Short | 45% | 1-2 句话简洁描述 |
| Medium | 45% | 2-4 句话，包含 spatial relations、attributes |
| Long | 9% | 完整 scene description，含 OCR、counting、fine-grained details |

直觉是：caption distribution 的多模态性可以训练模型适应不同 inference 时的 prompt 风格。这避免了"模型只懂长 detailed prompt，不会处理 short user query"的问题。

#### 3.4.2 Captioning model 选择：microbenchmark + Pareto frontier

captioning 100M images 需要在 quality 和 throughput 之间做 trade-off。作者构造了一个 microbenchmark：

- 1520 images, 720 short + 640 medium + 160 long captions
- Human annotators refine VLM-generated captions 作为 reference
- 5 个 evaluation axes：overall summary, counting accuracy, spatial understanding, attribute binding, OCR
- 0-2 scale, LLM-as-a-judge

Qwen3-VL-Instruct 4 个 scale 的 Pareto 比较（Figure 7）：

| Model | Throughput (img/s, 1×H100) | 简评 |
|---|---|---|
| Qwen3-VL-2B | 最高 | 质量不够 |
| **Qwen3-VL-4B** | 56.10 (short) / 49.31 (medium) | **Best Pareto** |
| Qwen3-VL-8B | 中等 | throughput 折半 |
| Qwen3-VL-30B-A3B (MoE) | 最低 | summary 1.73 vs 4B 的 1.68，gain 不大 |

最终选 Qwen3-VL-4B-Instruct，1500 H100 hours caption 完整 corpus。简单估算：100M images / 50 img/s ≈ 2M sec ≈ 555 GPU hours 单卡，考虑到 caption 不同 format、KV cache 等 overhead，1500 hours 是 reasonable budget。

#### 3.4.3 为什么不用 GPT-4V / Gemini / Claude

closed-source VLM 的成本对 100M scale 不可行。粗估：1M images × $0.01/image = $1M，100M 就是 $100M，远远超出学术 budget。即使 $0.001/image 也是 $100K+，而且 throughput 受 API rate limit 限制。

---

## 4. Benchmarking protocol：FD-DINOv2 替换 FID

这是 paper 的核心 contribution 之一。让我把 evaluation pipeline 完整写出来。

### 4.1 Fréchet Distance 公式

无论 FID 还是 FD-DINOv2，底层都是 Fréchet Distance between two multivariate Gaussians in feature space：

$$\text{FD} = \|\mu_r - \mu_g\|_2^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2\left(\Sigma_r \Sigma_g\right)^{1/2}\right)$$

变量含义：
- $\mu_r \in \mathbb{R}^d$: reference set (1M GPIC test images) 的 feature 均值向量
- $\mu_g \in \mathbb{R}^d$: generated set (50K generated images) 的 feature 均值向量
- $\Sigma_r \in \mathbb{R}^{d \times d}$: reference set 的 feature 协方差矩阵
- $\Sigma_g \in \mathbb{R}^{d \times d}$: generated set 的 feature 协方差矩阵
- $\|\cdot\|_2^2$: L2 范数的平方，衡量均值漂移
- $\text{Tr}(\cdot)$: 矩阵的迹，衡量协方差结构差异
- $(\Sigma_r \Sigma_g)^{1/2}$: matrix square root，通过 SVD 或 Schur decomposition 计算

第一项衡量"生成分布的中心是否对齐真实分布中心"，第二项衡量"分布的 spread 是否一致"。FD 越小越像真实分布。

FID vs FD-DINOv2 的唯一区别是 feature extractor：
- FID: Inception-v3 (pool3 features, 2048-dim)，supervised on ImageNet-1K
- FD-DINOv2: DINOv2 ViT-L/14 (CLS token, 1024-dim)，self-supervised on LVD-142M

### 4.2 Oracle references：benchmark saturation 的标尺

这是 GPIC evaluation 设计中最聪明的部分。作者计算了 real-vs-real FD 作为 reference point：

| GPIC Subset | FD↓ | Precision↑ | Recall↑ | Density↑ | Coverage↑ |
|---|---|---|---|---|---|
| Full (100M) | 1.19 | 0.947 | 0.950 | 1.000 | 0.972 |
| Lite (10M) | 1.25 | 0.951 | 0.947 | 1.010 | 0.973 |
| Nano (1M) | 1.60 | 0.946 | 0.946 | 1.002 | 0.968 |
| Val (200K) | 2.37 | 0.948 | 0.949 | 0.993 | 0.966 |
| **Test-50K** | **7.44** | 0.949 | 0.953 | 0.997 | 0.967 |

**Test-50K vs Test-1M 的 FD = 7.44** 这个数字非常重要——它是 saturation 的下界。当模型的 FD 接近或低于 7.44 时，说明模型已经接近"真实样本之间的差距"，benchmark 开始 saturate。

对比 JiT-T2I baseline 的 FD = 76.25，离 saturation 还有 10x 空间。这就是作者说的"FD-DINOv2 remains unsaturated"的 quantitative 含义。

### 4.3 多维度 metrics

GPIC 不只看 FD，还报告：

- **Precision** ([Kynkäänniemi et al., 2019](https://arxiv.org/abs/1904.06991))：fidelity，衡量 generated samples 有多少落入 real manifold
- **Recall**：diversity，衡量 real samples 有多少能被 generated manifold 覆盖
- **Density** ([Naeem et al., 2020](https://arxiv.org/abs/2002.09797))：fidelity 的连续版，不 upper bound by 1
- **Coverage**：diversity 的连续版
- **MMD** (Maximum Mean Discrepancy)：non-parametric alternative to FD，不需要 Gaussian 假设

直觉：FD 是 single scalar，会被 mean shift 和 covariance shift 混淆。Precision/Recall 拆开 fidelity 和 diversity，但离散化。Density/Coverage 是 Precision/Recall 的连续版，更 stable。MMD 是 sanity check。

### 4.4 Anti-gaming policy

GPIC protocol 明确禁止用 DINOv2 features 作为 training loss，因为这会直接优化 metric 的 feature space，构成 metric-specific optimization 而非 generative capability improvement。同时鼓励披露是否用了 larger auxiliary models（如 DINOv3、SigLIP），因为这些 model 见过的 data 远超 GPIC，会造成 unfair comparison。

这个 design 的直觉非常清晰：**evaluation feature space 的"几何"不能 leak 进 training objective**。一旦 leak，模型直接学 feature manifold 的低维投影，generation quality 没真的提升，FD 数字却在 drop。这就是 FID 被 Goodharted 的根本机制。

### 4.5 Held-out test set vs. training set

ImageNet-1K FID 的另一个问题是 reference statistics 算在 train set 上，所以模型可以 memorize train set 的 feature 分布来获得低 FID，holdout 检测不出 overfitting。

GPIC 把 reference statistics 算在 **1M held-out test set** 上，generated samples 用 50K fixed test captions 生成。这样如果模型 memorize train set，FD 会暴露（因为 test set 的 feature 分布和 train set 不同）。

---

## 5. Reference baseline：JiT-T2I pixel-space flow matching

GPIC 不只 release dataset，还训练了一个 reproducible baseline。这个 baseline 的设计选择反映了作者对 generative modeling 的偏好。

### 5.1 为什么选 pixel-space flow matching 而不是 latent diffusion

作者选 JiT（pixel-space flow matching + Transformer），参考 Ma et al. 的 PixGen 架构。直觉理由：

1. **No tokenizer pretraining**: latent diffusion 需要先训 VAE，VAE 本身就是另一个 training stage，引入额外 hyperparameter 和 compute。pixel-space 直接训 end-to-end。
2. **No auxiliary losses**: VAE 通常配 KL loss、perceptual loss、LPIPS 等。pixel-space 简化到 single flow matching objective。
3. **Single-stage training**: 不需要两阶段（VAE → diffusion），reproducibility 更好。

代价是 compute cost 更高（pixel space 比 latent space 大很多），所以 baseline 只训了 1 epoch on 256×256。

### 5.2 Flow matching 的目标函数

Flow matching 是 diffusion 的连续泛化。给定 noise distribution $x_0 \sim \mathcal{N}(0, I)$ 和 data distribution $x_1 \sim p_{\text{data}}$，定义 flow：

$$x_t = (1-t) x_0 + t x_1$$

其中 $t \in [0, 1]$ 是 time variable。对应 velocity field 是：

$$u_t(x_t | x_0, x_1) = x_1 - x_0$$

训练目标：

$$\mathcal{L}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,1],\, x_0 \sim \mathcal{N}(0,I),\, x_1 \sim p_{\text{data}},\, c} \left[ \left\| v_\theta(x_t, t, c) - (x_1 - x_0) \right\|_2^2 \right]$$

变量含义：
- $v_\theta$: 神经网络预测的 velocity field，参数为 $\theta$
- $x_t$: 在 time $t$ 的中间状态
- $t$: flow 的时间参数，0 = noise，1 = data
- $c$: conditioning signal (text caption)
- $x_0$: 标准正态噪声
- $x_1$: 真实图像
- $x_1 - x_0$: 目标 velocity（从 noise 指向 data 的直线方向）

直觉：flow matching 学一个 vector field，让所有从 noise 出发的"流线"汇聚到 data manifold。和 DDPM 的区别是，flow matching 用 linear interpolation path（直线），DDPM 用 forward SDE 的 marginal；前者 trajectory 更短、更可解释、更易用 ODE solver。

参考链接：
- Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023: https://arxiv.org/abs/2210.02747
- Liu et al., "Rectified Flow": https://arxiv.org/abs/2209.03003
- Ma et al., "Back to basics: Let denoising generative models denoise" (JiT paper): https://arxiv.org/abs/2511.13720
- PixelGen (perceptual loss for pixel diffusion): https://arxiv.org/abs/2602.02493

### 5.3 Classifier-free guidance 的 flow matching 版本

inference 时用 CFG：

$$\hat{v} = v_\theta(x_t, t, c) + w \cdot \left( v_\theta(x_t, t, c) - v_\theta(x_t, t, \varnothing) \right)$$

变量含义：
- $\hat{v}$: guidance-adjusted velocity
- $v_\theta(x_t, t, c)$: conditional velocity（带 text）
- $v_\theta(x_t, t, \varnothing)$: unconditional velocity（null text）
- $w$: guidance scale，等价于 CFG scale 的扩展形式（实际 CFG scale = $w + 1$）

直觉：当 $w > 0$，强化 conditional direction 远离 unconditional direction，类似 classifier guidance 但不需要外部 classifier。$w$ 太小生成质量低，$w$ 太大则 mode collapse、diversity 下降。

### 5.4 实验结果

JiT-T2I (1.1B params, PixGen-XXL/16, Qwen3-1.7B text encoder) 在 GPIC-Full 训 1 epoch, 256×256, batch size 256, AdamW lr=1e-4, betas (0.9, 0.95), constant schedule with 0.1% warmup, 40 hours on 8×H100。

| CFG | FD↓ | Precision↑ | Recall↑ | Density↑ | Coverage↑ |
|---|---|---|---|---|---|
| 1.75 | 204.01 | 0.917 | 0.530 | 1.034 | 0.806 |
| 4.00 | 87.80 | 0.933 | 0.765 | 1.012 | 0.906 |
| **6.25** | **76.25** | 0.942 | 0.792 | 1.014 | 0.908 |

观察：
- FD 随 CFG scale 增大单调下降（1.75→6.25: 204→76），说明 CFG 提升 fidelity 显著
- Recall 也单调上升（0.530→0.792），说明在此 regime 内 CFG 没引起 mode collapse
- Coverage 同步上升

注意 CFG=6.25 在 ImageNet-1K class-conditional 上已经偏高（常见 1.5-3.0），但在 GPIC 上更高才 optimal。直觉解释：text-conditioned generation 比 class-conditioned 需要 stronger guidance 来 align 文本语义。

距离 saturation (Test-50K vs Test-1M FD=7.44) 还有 10x 空间，说明 GPIC benchmark 还有大量爬坡空间，不像 ImageNet-1K 已经打穿。

---

## 6. 更细的 appendix notes

### 6.1 DINOv2 backbone size 和 register tokens 对 FD 的影响

Appendix B.3 ablate 了 4 个 backbone (ViT-S/B/L/g) × 2 (with/without registers)。结论：
- Small/Base/Large backbone 上 with-registers 比 without-registers FD 更低（因为 register 减少 high-norm artifacts，feature 值范围更小）
- Giant backbone 上两者接近
- 8 个配置的 Pearson correlation = 0.847，Kendall's W = 0.795

所以作者用 default = DINOv2 ViT-L/14 without registers，与 Stein et al. 2023 一致。

直觉：register tokens 是 Darcet et al. (2401.09192) 提出的，用于减少 ViT high-norm artifacts。但 register 改变 feature 分布，对 FD 的绝对值有影响。相对排序仍然 stable。

参考链接：
- Darcet et al., "Vision Transformers Need Registers": https://arxiv.org/abs/2309.16588

### 6.2 Image preprocessing: ImageNet-256 和 GPIC-256

统一用 Pillow bicubic downsample。作者专门提到不同 Python 库的 bicubic kernel 不同（PIL vs OpenCV vs torchvision），会导致 ±0.5 FID 差异。这种"细节差异"在饱和 benchmark 上是 noise floor 量级，会让结果不可复现。

参考链接：
- Parmar et al., "On aliased resizing and surprising subtleties in GAN evaluation", CVPR 2022: https://arxiv.org/abs/2104.11222

---

## 7. 我的几个 take-aways 和联想

### 7.1 关于 evaluation metric 设计的一般化教训

GPIC 的 anti-gaming policy 让我想到 ML benchmark 的一个 general principle：**evaluation feature space 必须独立于 training signal**。任何"评测器"如果和"训练 loss"在同一个 representation space，都会被 Goodharted。

这影响的不只是 image generation。在 LLM eval 里，我们用 MMLU/HumanEval 这些 benchmarks 也面临类似问题：评测基准的 distribution 和训练 distribution 重叠越多，benchmark 寿命越短。GPIC 通过用 self-supervised DINOv2 feature（与 ImageNet-1K label space 不共线）来延长 benchmark 寿命。

### 7.2 关于 permissive data 的稀缺性

power law extrapolation 显示 θ=0.95 在 110M source pool 上只能去 9.6M duplicates，这个相对保守的去重率（8.7%）说明 permissive license data 已经很稀缺，aggressive dedup 会浪费宝贵数据。对比 LAION-5B 这种 unrestricted license pool，dedup 移除率可以到 30-50%。

这暗示一个数据生态问题：**CC BY / CC0 数据的 supply 增长率跟不上 generative model 的 data demand 增长率**。CC license 是 2002 年 Creative Commons 提出的，那个时候 Flickr 用户上传图基本都 default CC BY-NC。后来 Flickr 改了 default，但 CC BY 数据存量就那么多。 Wikimedia 虽然持续增长但增长率有限。如果社区真的想持续做 open generative model，需要新的 permissive data production 机制（如 synthetic data with verifiable permissive origin？）。

### 7.3 关于 pixel-space 是否会回潮

JiT baseline 选择 pixel-space 是一个有意思的 signal。从 2022 年 LDM/Stable Diffusion 开始，latent-space diffusion 几乎是 industry consensus，因为 latent space 的 compute efficiency 巨大。但 2024-2025 出现两个反向 trend：

1. **SiD2 (Simpler Diffusion, Hoogeboom et al., 2024)** 在 ImageNet-512 上达到 1.5 FID with pixel-space diffusion，beat latent methods
2. **PixelGen (Ma et al., 2026)** 用 perceptual loss 让 pixel diffusion beat latent diffusion
3. **JiT (Li & He, 2025)** 提出 "let denoising generative models denoise"，简化 pixel diffusion

直觉：VAE 的 reconstruction error 是 latent diffusion 的 fundamental ceiling。一旦 model scale 和 data scale 大到某个 threshold，VAE bottleneck 变成限制因素，pixel-space 反而能学得更好。但 compute cost 仍然是问题——GPIC baseline 只训 1 epoch on 256×256，比 latent diffusion 的 budget 多很多但效果还不在 SOTA。

参考链接：
- SiD2 paper: https://arxiv.org/abs/2410.19324
- Reconstruction vs. Generation (Taming optimization dilemma): https://arxiv.org/abs/2501.01423
- VAR (next-scale prediction, alternative to diffusion): https://arxiv.org/abs/2404.02905

### 7.4 关于 benchmark 数据集的"宪法"

GPIC 强调用 frozen tar shards 而非 URL index。这个 idea 不新（ImageNet 当年也是 frozen tar），但近 5 年随着 webdataset 的兴起重新流行。我注意到 DataComp 和 LAION 都开始提供 frozen snapshot 版本。

HF datasets streaming + tar shards 现在基本是 large scale training 的 de-facto standard。GPIC 把它做成 8,000 个 ~12,500 images/shard 的 balanced shard，每个 shard 都保持 caption-type 比例（1% tag, 45% short, 45% medium, 9% long），这避免了某些 shard 全是 short caption、某些全是 long caption 的 shard-level bias。

### 7.5 关于 caption format distribution 的设计

1% / 45% / 45% / 9% 这个 mix 我觉得值得讨论。Long caption 占比小可能因为：
- Long caption token 长，KV cache 大，throughput 低
- Long caption 在真实 user prompt distribution 里占比小（用户多半输入 1-2 句）
- 留 9% 训练模型处理 detailed prompt 的能力即可

但这也意味着 long caption 的样本 diversity 不够，模型在 long prompt 上的 OOD 性能可能受限。后续工作可以 ablate 这个 mix。

### 7.6 GPIC-Nano (1M) 作为 ImageNet-1K 替代品的潜力

ImageNet-1K train set 1.28M images，GPIC-Nano 1M images，scale 可比。但 GPIC-Nano 是 text-conditioned，caption 多样，没有 class label。这能否成为下一个十年的 ImageNet-1K？

我的直觉：能，但需要时间。需要一些 reference method 在 GPIC-Nano 上建立 Pareto frontier（不同 param count / training FLOPs 的 model），让 community 有可比的 baseline。GPIC paper 给的 JiT-T2I baseline 是 1.1B 模型，太大了，nano benchmark 还需要 50M-200M scale 的小 baseline。

### 7.7 关于 FD-DINOv2 的 residual risks

虽然 FD-DINOv2 暂时解决了 Inception-v3 的 saturation 问题，但作者自己警告：DINOv2 也可能在 LVD-142M 上见过 GPIC test images（因为 LVD-142M 是 web crawled），存在 implicit leakage。如果 community 直接用 DINOv2 features 做 training loss，FD-DINOv2 也会被打穿。

更 robust 的方向可能是 self-supervised feature trained 在 controlled corpus 上（例如只在 GPIC train set 上 trained 的 SSL model）。但这个 chicken-and-egg：先有 GPIC 才能训 controlled SSL model，反过来才能 stable eval。这是一个值得后续工作探索的方向。

### 7.8 关于 broader impact 的诚实

最后一句说 "despite our deduplication efforts, some near-duplicates may remain in GPIC, although their prevalence is estimated to be small"。这种"我们不完美"的声明在 dataset paper 里少见，但很重要。任何 100M scale 的 dataset 都会有 noise，关键是 noise rate 和影响是否可控。GPIC 通过 SSCD similarity inspection + power law extrapolation + conservative threshold 来 bound duplicate rate，是 reasonable 的工程妥协。

---

## 8. 相关工作的 web links 汇总

- GPIC project page: https://gpic.stanford.edu
- GPIC Hugging Face (推测): https://huggingface.co/datasets/StanfordGPIC
- DINOv2: https://arxiv.org/abs/2304.07193
- DINOv3 (2025): https://arxiv.org/abs/2508.10104
- SigLIP: https://arxiv.org/abs/2303.15343
- SSCD copy detection: https://arxiv.org/abs/2202.10261
- FAISS: https://faiss.ai
- vLLM: https://arxiv.org/abs/2309.06180
- SGLang: https://arxiv.org/abs/2312.07104
- Qwen3-VL: https://arxiv.org/abs/2511.21631
- Qwen3 (text): https://arxiv.org/abs/2505.09388
- DataComp: https://arxiv.org/abs/2304.14108
- LAION-5B: https://arxiv.org/abs/2210.08402
- YFCC100M: https://multimediacommons.wordpress.com/yfcc100m-core-dataset/
- OpenImages: https://storage.googleapis.com/openimages/web/index.html
- ImageNet (Deng et al., 2009): https://image-net.org
- DiT (Peebles & Xie): https://arxiv.org/abs/2212.09748
- Simpler Diffusion (SiD2): https://arxiv.org/abs/2410.19324
- VAR (Tian et al.): https://arxiv.org/abs/2404.02905
- Flow Matching (Lipman et al.): https://arxiv.org/abs/2210.02747
- Rectified Flow (Liu et al.): https://arxiv.org/abs/2209.03003
- JiT (Li & He, 2025): https://arxiv.org/abs/2511.13720
- PixGen (Ma et al., 2026): https://arxiv.org/abs/2602.02493
- Improved Precision & Recall (Kynkäänniemi et al.): https://arxiv.org/abs/1904.06991
- Assessing generative models via precision and recall (Sajjadi et al.): https://arxiv.org/abs/1806.00035
- Reliable fidelity and diversity metrics (Naeem et al.): https://arxiv.org/abs/2002.09797
- FID (Heusel et al., NeurIPS 2017): https://arxiv.org/abs/1706.08500
- Going deeper with convolutions (Inception-v3): https://arxiv.org/abs/1409.4842
- Vision Transformers Need Registers: https://arxiv.org/abs/2309.16588
- Exposing flaws of generative eval metrics (Stein et al.): https://arxiv.org/abs/2311.16512
- img2dataset: https://github.com/rom1504/img2dataset
- Wan video generation models: https://arxiv.org/abs/2503.20314
- Sora (OpenAI): https://openai.com/research/video-generation-models-as-world-simulators
- Parmar et al. aliased resizing: https://arxiv.org/abs/2104.11222

---

## TL;DR

GPIC 的核心 contribution 是把"permissive, stable, large, accessible"作为四个 explicit design axes，然后系统地解决 ImageNet-1K 时代遗留下来的 benchmark saturation、Goodharting、irreproducibility 三个问题。技术上最 elegant 的部分是 power law extrapolation 估算 dedup removals，以及 FD-DINOv2 + held-out test set + anti-gaming policy 组合解决 evaluation leakage。JiT-T2I baseline 提供了 reproducible 起点，但离 saturation 还有 10x 空间，说明 GPIC 是面向未来 3-5 年的 benchmark，而不是 2026 年的 SOTA。
