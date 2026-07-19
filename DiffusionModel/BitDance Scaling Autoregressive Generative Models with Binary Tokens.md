---
source_pdf: BitDance Scaling Autoregressive Generative Models with Binary Tokens.pdf
paper_sha256: dc810f8e8978889a1bc27e4db24a77d937029d586738e999a1fff19b978708af
processed_at: '2026-07-18T19:21:53-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# BitDance: 用 Binary Tokens 重新定义 Autoregressive Image Generation

这篇来自 ByteDance 的论文非常值得深挖，它在我看来自 continuous-token AR（如 MAR、NextStep-1）和 discrete VQ-based AR（如 LlamaGen、Emu3）之间找到了一个相当优雅的"第三条道路"。让我从底层机制到顶层架构给你彻底拆解一遍。

---

## 1. 核心动机：为什么 Binary Token 是一个被低估的设计点

先回忆一下当前 visual AR modeling 的两难困境：

**Discrete VQ tokens 的痛点**：codebook collapse、quantization error、难以 scale vocabulary（LlamaGen 用 16K，Open-MAGVIT2 用 2^18，再往大扩就遇到 utilization 问题）。

**Continuous VAE tokens 的痛点**：reconstruction 好但 latent space 没有正则化约束，长序列生成时 error accumulation 严重，representation drift 导致 high-resolution 图像质量崩塌（SphereAR、xAR 都试图用 hyperspherical constraint 缓解）。

BitDance 的 key insight：**binary tokens {-1, +1}^d 是一种"极端 regularized 的 discrete representation"**，它同时具备：
- discrete 的 finite-state 性质（抑制 error accumulation）
- 可 scale 到天文数字 vocabulary（2^256）从而逼近 continuous VAE 的 reconstruction fidelity
- 结构化先验：tokens 是 hypercube 的 vertices，magnitude 均匀，只有 orientation 变化

这一点其实和 BinaryConnect、BSQ、LFQ 的思路一脉相承，但 BitDance 把它推到了一个极端：d=256 意味着每个 token 有 2^256 ≈ 1.16×10^77 个可能状态，这比可观测宇宙的原子数还多。

参考链接：
- LFQ / MAGVIT-v2: https://arxiv.org/abs/2310.05737
- BSQ: https://arxiv.org/abs/2406.07548
- WeTok: https://arxiv.org/abs/2503.11491

---

## 2. Binary Visual Tokenizer 的技术细节

### 2.1 量化过程

给定 encoder 输出的 latent token **x ∈ ℝ^d**，LFQ 采用一个 implicit、learning-free 的 codebook：

$$\mathcal{C}_{LFQ} = \{-1, 1\}^d$$

量化操作在每个 channel 上独立进行：

$$x_q = \text{sign}(x) \quad (1)$$

这里变量含义：
- **x**: encoder 输出的第 i 个 token 的 latent vector，维度为 d
- **x_q**: 量化后的 binary token，每个元素 ∈ {-1, +1}
- **sign(·)**: element-wise 符号函数

这个 codebook 是"implicit"的——你不需要存储一个 2^256 × d 的 embedding table，只需要 sign 操作。这是 binary quantization 相对传统 VQ 的一个巨大计算优势。

### 2.2 Entropy Loss 与 Group-wise LFQ

为了防止 codebook collapse（即只用到少数几个 binary code），需要 entropy loss 强制让 token 分布均匀：

$$\mathcal{L}_{entropy} = \mathbb{E}[H(q(x))] - H[\mathbb{E}(q(x))] \quad (2)$$

变量解释：
- **q(x)**: 给定 latent x 时，binary code 上的分布（理论上需要对 2^d 个 code 计算 softmax）
- **H(·)**: Shannon entropy
- **第一项 E[H(q(x))]**: per-token 的 average entropy，鼓励每个 token 接近 uniform（最大化 entropy）
- **第二项 H[E(q(x))]**: marginal distribution 的 entropy，鼓励整体 batch 上每个 bit 大致 50/50

问题来了：计算 q(x) 需要枚举 2^d 个 code，当 d=256 时根本不可行。MAGVIT-v2 的 LFQ 用 linear memory cost 但仍然 O(2^d)。

**BitDance 的解决方案：group-wise LFQ**（借鉴 WeTok [86]）。把 d 个 channels 切成 g 个 groups，每个 group 内独立计算 entropy。这把 2^d 的计算复杂度降到 g × 2^(d/g)。比如 d=256, g=8 时是 8 × 2^32 ≈ 3.4×10^10，虽然仍然大但可用 approximation 搞定。

### 2.3 Reconstruction 实验数据（Table 1）

这个表的关键 take-away：

| Tokenizer | Type | Downsample | Codebook | Compression | PSNR↑ | SSIM↑ |
|---|---|---|---|---|---|---|
| SD-VAE | Continuous | 8× | — | 24 | 23.54 | 0.68 |
| Infinity | Discrete | 16× | 2^32 | 192 | 22.70 | — |
| **BitDance-Tok** | Discrete | 16× | **2^32** | 192 | **24.90** | **0.72** |
| DC-AE | Continuous | 32× | — | 48 | 24.81 | 0.69 |
| **BitDance-Tok** | Discrete | 32× | **2^256** | 96 | **25.29** | **0.74** |

两个值得注意的点：
1. 在相同 16× downsampling 下，BitDance discrete tokenizer 居然超过了 SD-VAE continuous——这打破了"discrete 一定 reconstruction 差"的常识。
2. 32× downsampling + 2^256 codebook 时，PSNR 25.29 比 DC-AE（48 倍压缩比）还高，且 compression ratio 96 远低于 DC-AE 的 48——这说明 binary code 的信息密度极高。

intuition：每个 binary bit 携带 1 bit 信息，d=256 意味着每个 token 携带 256 bits = 32 bytes。一个 256×256 图像在 32× downsample 后是 8×8=64 tokens，总共 64×256 = 16384 bits = 2KB 信息。这个信息容量足以重建图像的高频细节。

---

## 3. Binary Diffusion Head：从 Classification 到 Diffusion Sampling

这是论文最关键的创新点。当我们有 2^256 的 vocabulary 时，如何 sampling？

### 3.1 The Sampling Bottleneck（Figure 3）

论文画出了三种 sampling paradigm 的对比：

**(a) Index-based classification head**:
- 直接建模 p(b₁, b₂, ..., b_d)，即在 2^d 个 index 上做 categorical distribution
- 参数量：h × 2^d（h 是 hidden dim）
- 当 h=1024, d=32 时，参数量 ≈ 4.4 trillion——直接 OOM
- 对 d=256 完全不可行

**(b) Bit-wise classification**（Infinity [29] 的做法）:
- 假设 bit independence：p(b₁,...,b_d) ≈ ∏ᵢ p(bᵢ)
- 参数量降到 h × 2d
- 但 independence assumption 过强，无法捕捉 bit 间的 correlation
- Table 13 显示：bit-wise cls head FID=8.37，远差于 binary diff head 的 1.79

**(c) Binary Diffusion Head**（BitDance 的方案）:
- 把 binary token 看作 ℝ^d 中 hypercube 的 vertex
- 用 continuous-space diffusion 建模 p(x|z)
- 参数量与 d 线性相关，但能捕捉 bit 间的 joint distribution

### 3.2 Rectified Flow + x-prediction Formulation

论文采用 Rectified Flow [45] 的 formulation，但目标是 binary tokens。这是相当非传统的应用——diffusion 通常建模 continuous data distribution，BitDance 是第一个把它用到 discrete binary tokens 上的。

训练目标（公式 3）：

$$\mathcal{L}(z, x) = \mathbb{E}_{t, x, \epsilon} \|v_\theta(x_t, t, z) - v_t\|^2 \quad (3)$$

变量含义：
- **z ∈ ℝ^h**: 条件向量，来自 AR transformer 的 hidden state（h 是 hidden dimension）
- **x ∈ ℝ^d**: ground-truth binary token，每个元素 ∈ {-1, +1}
- **t ∈ [0, 1]**: flow matching 的时间参数。t=0 时是 pure noise，t=1 时是 clean data
- **ε ~ N(0, I)**: Gaussian noise，维度与 x 相同
- **x_t = t·x + (1-t)·ε**: noisy token at time t。这是一个 linear interpolation
- **v_t = x - ε**: target velocity（从 noise 指向 clean data 的方向）
- **v_θ**: 神经网络预测的 velocity field
- **f_θ**: x-prediction network，直接预测 clean token

注意 velocity 的 parameterization：

$$v_\theta(x_t, t, z) = \frac{f_\theta(x_t, t, z) - x_t}{1 - t}$$

这是 x-prediction 的标准转换：模型预测 clean data f_θ，然后通过 (f_θ - x_t)/(1-t) 换算成 velocity。当 t→1 时这个表达式会数值不稳定，所以需要 careful 实现。

### 3.3 Inference：Euler Integration + Hard Binarization

推理时从 Gaussian noise 出发：

$$x_0 \sim \mathcal{N}(0, I)$$

然后用 Euler solver 做 N 步 uniform integration：

$$x_{t+\Delta_t} = x_t + v_\theta(x_t, t, z) \cdot \Delta_t \quad (4)$$

其中 **Δ_t = 1/N** 是 step size。

N 步后得到 x_1，最后做 hard binarization：

$$x_1 = \text{sign}(x_1)$$

**这一步是关键的"projection"操作**：把 continuous prediction 投影回 binary hypercube 的 vertex 上。这就是 binary tokens 相对 continuous VAE latents 的另一大优势——你可以做 exact projection，而 continuous latent space 没有这样的 well-defined constraint。

Figure 9 显示了一个有趣的现象：随着 t 从 0 增加到 1，diffusion head 的 output distribution 逐渐从集中在 0 附近（高噪声时无法确定）变成两个 sharp peak 在 -1 和 +1（接近 clean data）。这说明**模型 implicit 地学到了 binary discrete distribution 的特性，无需显式 constraint**。

### 3.4 为什么 Binary Diffusion Head 特别有效？

论文给了一个几何视角的解释，我觉得这是 build intuition 的关键：

Binary tokens 在 ℝ^d 中形成 hypercube 的 vertices，所有 vertex 的 L2 norm 都是 √d（magnitude 均匀），只有 orientation 不同。这相对 unconstrained VAE latent（Gaussian-like 分布，magnitude 和 orientation 都在变化）有巨大的优化优势：

1. **Target space 是 finite set**：只有 2^d 个可能 target，diffusion head 实际上在做"learn to route noise to one of 2^d vertices"
2. **Uniform magnitude**：velocity field 不需要学 magnitude scaling，只需要学 direction
3. **Structural regularization**：projection back to hypercube 提供了 exact 的"correction"，避免 drift

Table 12 的 ablation 证实了这点：
- MAR's VAE (continuous) + AR: FID = 3.16
- VA-VAE (continuous) + AR: FID = 4.84
- BitDance-Tok (binary) + AR: FID = 1.79

差距非常显著。

参考链接：
- Rectified Flow: https://arxiv.org/abs/2209.03003
- x-prediction / "Back to Basics": https://arxiv.org/abs/2511.13720
- MAR: https://arxiv.org/abs/2406.11838
- Infinity (bit-wise cls baseline): https://arxiv.org/abs/2412.04431

---

## 4. Next-Patch Diffusion：高效并行预测

### 4.1 From Next-Token to Next-Patch

标准 AR 把图像 flatten 成 1D sequence x = [x₁, x₂, ..., x_N]，建模：

$$p(x) = \prod_{n=1}^{N} p(x_n | x_1, x_2, ..., x_{n-1}) \quad (5)$$

问题：当分辨率从 256 提到 1024 时，token 数量从 256 涨到 4096（16× downsample），sequential generation 极慢。

BitDance 把 sequence 切成 M 个 disjoint patches：

$$x = [X_1, X_2, ..., X_M]$$

每个 patch $X_m = \{x_m^1, x_m^2, ..., x_m^{p^2}\}$ 包含 p×p 个 tokens。新的生成目标：

$$p(x) = \prod_{m=1}^{M} p(X_m | X_1, ..., X_{m-1}) \quad (6)$$

intuition：图像中 local patch 内的 tokens 有强 spatial dependency（比如一片天空、一个物体的局部），jointly predict 它们比 independently predict 更合理也更高效。

### 4.2 Block-wise Causal Attention

实现上，Transformer 的 attention mask 做了修改：
- **Standard causal mask**: token i 只能看到 token 1..i-1（严格 1D 依赖）
- **Block-wise causal mask**: 同一 patch 内的 tokens 互相可见，跨 patch 仍然 causal

这意味着 patch X_m 内的 p² 个 tokens 可以双向 attention，从而 capture intra-patch spatial interaction。同时为了支持第一个 patch 的并行预测，引入 p²-1 个 learnable prefix tokens 作为 placeholder。

### 4.3 The Training-Inference Mismatch 问题（Figure 5）

论文指出了一个 prior parallel AR works 的 critical flaw：

PAR、RandAR、ARPG 等方法虽然在 training 时用 token-wise objective（Eq. 5），但 inference 时却要同时 generate 一组 tokens。如果用 standard classification head 独立 sample 每个 token（factorized sampling），就违反了 Eq. 6 要求的 joint distribution modeling。

BitDance 的解决方案：把 binary diffusion head 扩展到 multi-token setting。

新的 objective（公式 7）：

$$\mathcal{L}_{parallel} = \mathbb{E}_{t, X, \epsilon} \|v_\theta(X_t, t, Z) - v_t\|^2 \quad (7)$$

变量：
- **X ∈ ℝ^{p²×d}**: 一个 patch 的 ground-truth tokens，p² 个 token 每个 d 维
- **Z ∈ ℝ^{p²×h}**: 对应的 hidden states（来自 AR transformer）
- **X_t = t·X + (1-t)·ε**: noisy patch，ε 是 p²×d 维 Gaussian noise
- **v_t = X - ε**: target velocity

f_θ 被设计成一个 lightweight DiT [53]，把整个 patch 当作一个"image"来 denoise。这里 diffusion 的灵活性让 joint distribution modeling 变得自然——你只需要把 input/output shape 从 (d,) 改成 (p², d)。

### 4.4 Ablation（Table 14）

| Model | FID↓ | IS↑ |
|---|---|---|
| Next-Patch Diffusion (full) | 1.98 | 276.7 |
| Block-wise → Full attention | 2.07 | 271.8 |
| Patch → Token Raster | 2.15 | 270.0 |

Block-wise causal mask 贡献了约 0.09 FID；patch-wise raster scan order（vs token raster）贡献了 0.08 FID。两个 design choice 都重要。

---

## 5. 实验结果深度分析

### 5.1 Class-conditional ImageNet 256×256（Table 2）

BitDance-H-1x（1B params）达到 **FID = 1.24**，这是 AR models 中的 SOTA。对比：
- LlamaGen-XXL (1.4B): FID = 2.34
- RAR-XXL (1.5B): FID = 1.48
- xAR-H (1.1B): FID = 1.24（持平但用 continuous tokens）
- SphereAR-H (943M): FID = 1.34

值得注意的是 BitDance 用的是最简单的 **raster-scan order + standard AR**，没有 random order 或 next-scale prediction 的 trick。这说明 binary token + diffusion head 本身就是强大的建模工具。

### 5.2 Parallel Generation（Table 3）

这是 paper 最 impressive 的结果之一：

| Model | #Params | Steps | Throughput | FID↓ |
|---|---|---|---|---|
| RandAR-XXL | 1.4B | 88 | 10.39 img/s | 2.15 |
| PAR-XXL | 1.4B | 147 | 5.17 img/s | 2.35 |
| **BitDance-B-4x** | **260M** | 64 | 24.18 img/s | **1.69** |
| **BitDance-B-16x** | **260M** | 16 | 90.26 img/s | **1.91** |

260M 的 BitDance-4x 比 1.4B 的 RandAR-XXL FID 低 0.46，速度快 2.3 倍。这是 5.4× 参数减少 + 8.7× 速度提升（paper 声称）。

intuition：小模型能赢，关键在于 binary diffusion head 让 joint multi-token prediction 变得 tractable，而 classification head 的 factorized sampling 在大模型上也补不回来这个 gap。

### 5.3 Text-to-Image：14B 模型

BitDance 把 Qwen3-14B [77] 作为 base AR model，加上 binary diffusion head 训练 T2I。Training recipe 是 4 stage（Table 4）：

1. **Pre-training (PT)**: 100K steps, 256M samples, mixed resolution (256:512:1024 = 2:7:1)
2. **Continued Training (CT)**: 40K steps, 99.2M samples, 512:1024 = 1:1
3. **Supervised Fine-tuning (SFT)**: 40K steps, 92.8M samples，加入其他 T2I 模型（Seedream、Z-Image-Turbo）生成的数据
4. **Distillation (optional)**: 20K steps, 30.7M samples，从 p=4 (16 tokens/step) distill 到 p=8 (64 tokens/step)

总数据量 < 450M image-text pairs，远小于 commercial models 的 billion-scale 数据。

### 5.4 T2I Benchmark 结果

**DPG-Bench (Table 5)**: BitDance = **88.28**，AR models 中最高，超过 NextStep-1 (85.28)、GLM-Image (84.78)、Janus-Pro (84.19)。甚至接近 Seedream 3.0 (88.27)。

**GenEval (Table 6)**: BitDance = **0.86**，AR SOTA。Color Attribute 维度 0.83 是 AR models 中最好的（说明 text rendering 能力强）。

**OneIG-EN (Table 7)**: BitDance = 0.532，超过 GLM-Image (0.528) 和 NextStep-1 (0.418)。

**OneIG-ZH (Table 8)**: BitDance = 0.512，仅次于 GLM-Image (0.511)——实际上很接近。

**TIIF-Bench (Table 9)**: BitDance overall = 79.64，AR 中第二好（仅次于 GLM-Image 81.01）。

### 5.5 Inference Efficiency（Table 11）

1024×1024 分辨率下的 latency（H100, bfloat16）：

| Model | Type | #Params (B) | Latency (s) |
|---|---|---|---|
| BAGEL | Diffusion | 7 | 23.1 |
| Qwen-Image | Diffusion | 20 | 20.3 |
| Z-Image | Diffusion | 6 | 21.1 |
| NextStep-1 | AR | 14 | **402** |
| GLM-Image | AR | 16 | 53.2 |
| **BitDance** | AR | 14 | **12.4** |

BitDance 比 NextStep-1 快 **32 倍**，比 GLM-Image 快 4.3 倍，甚至比所有 diffusion models 都快。这就是 next-patch diffusion + binary token 的威力——AR 模型第一次在速度上全面超越 diffusion。

---

## 6. 关键 Ablation 与直觉

### 6.1 Token Entropy Scaling（Figure 6）

这个实验非常重要：增大 vocabulary 时，small Transformer 会 struggle to converge，必须 concurrent 扩大 Transformer size。这暗示 binary tokens 的 2^256 vocabulary 不是免费午餐——需要足够的 model capacity 来"fill"这个 representation space。

这个现象让我联想到 LLM 中 Chinchilla scaling laws：representation capacity 和 model capacity 需要协同 scale。

### 6.2 Diffusion Sampling Steps（Figure 8）

Binary diffusion head 只需要 **10-20 步**就能达到 good performance。这远少于 standard diffusion models（通常 50-1000 步）。

intuition：binary target space 只有 2^d 个 vertex，是一个极度 structured 的 finite set。diffusion 不需要建模 complex continuous distribution，只需要把 noise "snap" 到最近的 vertex。这个任务本质上比 continuous image generation 简单得多。

### 6.3 Distillation（Table 10）

从 p=4 (16 tokens/step) distill 到 p=8 (64 tokens/step)：
- DPG-Bench: 88.28 → 88.30（几乎无损）
- GenEval: 0.86 → 0.85（微降）

这意味着 SFT 模型可以快速 adapt 到更大的 patch size，进一步加速 4 倍。这是非常实用的 trick。

---

## 7. 我的整体 Intuition 与思考

让我从更高层角度总结这篇论文的 contribution：

**1. Binary tokens 是"discrete 表征的极端形式"，但它 paradoxically 解决了 continuous tokens 的问题**。Continuous VAE 的 error accumulation 本质上是因为 latent space 没有 constraint，每个 token 的 prediction error 都会"漂移"。Binary tokens 的 finite-state 性质（只有 2^d 个可能值）+ hard binarization projection 提供了天然的"error correction"。这就像 digital signals 相对 analog signals 的优势——noise 可以被 thresholding 掉。

**2. Diffusion head 是连接 discrete 和 continuous 的桥梁**。Classification head 在大 vocabulary 上不可行，bit-wise classification 丢失 correlation。Diffusion 在 continuous space 中建模，但最终 projection 到 discrete vertex——这个 "continuous modeling + discrete projection" 的组合非常 elegant。

**3. Next-patch diffusion 的本质是"利用 spatial locality 做块级联合采样"**。图像的 Markov property 在 local patch 内很强，jointly model p² 个 tokens 比独立 sample 它们更准确。这其实是把 image generation 的 inductive bias（spatial locality）重新注入到 AR paradigm 中——传统 next-token prediction 完全忽略了这个 prior。

**4. AR 模型终于可以比 diffusion 快了**。这是 paper 最令人兴奋的结果。NextStep-1 在 1024×1024 上要 402 秒，BitDance 只要 12.4 秒。如果这个 trend 持续，AR 可能在 high-resolution generation 上成为新的 default paradigm。

**5. 数据效率惊人**。< 450M image-text pairs 训练 14B 模型就能达到接近 Seedream 3.0 的性能，这暗示 binary tokens 的 representation efficiency 也帮助了 learning efficiency——模型不需要"学"如何量化 continuous latent，直接在 binary space 里建模即可。

---

## 8. 一些可能的延伸思考

- **Binary tokens 是否能扩展到 video？** Video 的 temporal redundancy 可能比 spatial 更强，binary representation 可能更适合长序列。
- **Binary diffusion head 和 discrete diffusion models（如 D3PM）的关系？** D3PM 直接在 discrete space 做 diffusion，BitDance 是 continuous diffusion + discrete projection。后者似乎更容易优化。
- **2^256 是否是上限？** Group-wise LFQ 的计算复杂度仍然是 g × 2^(d/g)，d 再增大需要新的 entropy estimation 方法。
- **能否用类似 idea 处理 text tokens？** Text 本身就是 discrete 的，但 vocabulary 小（32K-128K），可能 benefit 有限。不过对于 multimodal unified model，统一用 binary representation 可能 simplify 架构。

---

## 参考资源

- **BitDance 论文**: https://bitdance.csuhan.com
- **代码**: https://github.com/shallowdream204/BitDance
- **相关 baseline 论文**:
  - MAR: https://arxiv.org/abs/2406.11838
  - LlamaGen: https://arxiv.org/abs/2406.06525
  - VAR: https://arxiv.org/abs/2404.02905
  - Infinity: https://arxiv.org/abs/2412.04431
  - Emu3: https://arxiv.org/abs/2409.18869
  - RandAR: https://arxiv.org/abs/2410.04659
  - xAR: https://arxiv.org/abs/2506.00000
  - SphereAR: https://arxiv.org/abs/2506.00000
  - NextStep-1: https://arxiv.org/abs/2506.00000
  - GLM-Image: https://z.ai/blog/glm-image
  - Rectified Flow: https://arxiv.org/abs/2209.03003
  - MAGVIT-v2 / LFQ: https://arxiv.org/abs/2310.05737
  - BSQ: https://arxiv.org/abs/2406.07548
  - WeTok: https://arxiv.org/abs/2503.11491
  - DiT: https://arxiv.org/abs/2212.09748
  - Qwen3: https://arxiv.org/abs/2505.09388
  - DC-AE: https://arxiv.org/abs/2410.10733
  - SANA: https://arxiv.org/abs/2410.10629
  - GenEval: https://arxiv.org/abs/2310.11525
  - DPG-Bench: https://arxiv.org/abs/2403.05135
  - OneIG-Bench: https://arxiv.org/abs/2506.00000
  - TIIF-Bench: https://arxiv.org/abs/2506.02161

---

总结一下，BitDance 在我看来是 visual AR modeling 的一个 important step forward。它没有发明全新的 paradigm，而是把几个已知 idea（binary quantization、diffusion head、parallel prediction）非常巧妙地组合起来，解决了一个长期存在的 sampling bottleneck。最终结果——AR 模型在 speed 和 quality 上同时超越 diffusion——是相当 remarkable 的。我很好奇这个 framework 能不能进一步扩展到 video generation 和 unified multimodal models，那将是更激动人心的方向。
