---
source_pdf: Visual Autoregressive Modeling Scalable Image.pdf
paper_sha256: 09bf82325072d173f99954cf971b5f748ddc8834dadd9afc135098410d6b5849
processed_at: '2026-08-13T02:19:44-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话聊聊 VAR

## 一句话总结

VAR 的核心 idea 就是: **别再逐像素扫描式地生成图片了, 像画画一样, 先勾轮廓, 再填细节**。

就这么简单一个 insight, 直接把 GPT-style 的 image generation 干到了 FID 1.73, 把 DiT 按在地上摩擦, 还快了 20 倍。

---

## 这事儿到底在解决什么问题?

### 传统 AR 的蠢相

Karpathy 你想想, 如果你画一张猫, 你会怎么做? 肯定是先画个大致的圆形 (头), 三角形 (耳朵), 再画眼睛, 最后补毛发纹理。没有人会从左上角第一个 pixel 开始, 一个 pixel 一个 pixel 往右下角扫着画。

但传统 image AR (比如 VQGAN) 就是这么干的。它把 image 切成 $h \times w$ 个 discrete tokens (比如 16×16=256 个), 然后像语言模型读句子一样, 从左到右, 从上到下, 一个一个 token 生成。这叫 raster-scan order。

这种做法有几个根本性的毛病:

**第一, 数学上就别扭**。 VQVAE encoder 是 CNN+attention, 它输出的 feature map 里, 每个 token 都和其他所有 token 互相依赖 (bidirectional)。但 AR 假设每个 token 只依赖它前面的 tokens (unidirectional)。这相当于你拿一个本来是双向的 distribution, 非要用单向的 chain rule 去套, 怎么套都觉得别扭。

Paper 在 Appendix A 还专门画了 attention map (Figure 9) 证明这一点 — VQGAN encoder 最后一层的 attention, 清清楚楚显示 tokens 之间是双向相关的。

**第二, spatial 结构被破坏**。 Image 里相邻的 token 应该是紧密相关的 (比如相邻 pixel 颜色接近)。你 raster-scan flatten 之后, $(i,j)$ 和 $(i+1,j)$ 这俩上下相邻的 token, 在序列里隔了 $w$ 个位置。Transformer 理论上能学到, 但你这不是给它增加难度嘛。

**第三, 慢得离谱**。 这是 paper Appendix B 给的证明, 很优雅:

对 $n \times n$ 的 token map, 总共 $n^2$ 个 tokens。第 $i$ 步生成时, 要和前面所有 $i$ 个 token 做 attention, 成本 $\mathcal{O}(i^2)$。总共:

$$\sum_{i=1}^{n^2} i^2 \sim \mathcal{O}(n^6)$$

256×256 ImageNet (16×16=256 tokens) 还能跑, 要是 1024×1024 (64×64=4096 tokens), 直接废了。4096 个 AR step, 每步 attention 都要扫几千个 token, 这速度根本没法用。

### VAR 怎么解

VAR 说: 好, 那我们换个 order。 别在 token 粒度上 autoregressive, 在 **scale** 粒度上 autoregressive。

具体就是:
- 把 image 编码成 10 个不同分辨率的 token map: $r_1$ (1×1) → $r_2$ (2×2) → ... → $r_{10}$ (16×16)
- 生成时, 先生成 $r_1$ (1 个 token, 全局信息), 再生成 $r_2$ (4 个 token, 粗结构), 一直到 $r_{10}$ (256 个 token, 细节)
- **每个 scale 内部是并行的**, scale 之间才 autoregressive

这样 AR step 从 256 降到 10, 每个 scale 内的所有 token 一次性 parallel decoded。

复杂度从 $\mathcal{O}(n^6)$ 降到 $\mathcal{O}(n^4)$, 快了 $\mathcal{O}(n^2)$ 倍。256×256 上是 20× 的 wall-clock 加速。

---

## Multi-scale VQVAE: 这才是真正的工程难点

VAR 能 work, 70% 的功劳在 tokenizer 上。这个 multi-scale VQVAE 设计得非常精巧。

### 核心思路: residual quantization

Algorithm 1 (encoding) 的精髓:

```
f = encoder(image)          # 先 encode 成 feature map
for k = 1 to K:
    r_k = quantize(downsample(f, (h_k, w_k)))   # 把 f 下采样到 scale k 的分辨率, quantize
    z_k = codebook_lookup(r_k)                    # 查表得到 quantized vectors
    z_k = upsample(z_k, (h_K, w_K))              # 上采样回原分辨率
    f = f - conv_k(z_k)                          # 关键: 残差更新
```

最后一行 $f = f - \phi_k(z_k)$ 是灵魂。

什么意思呢? 第一个 scale $r_1$ 是 1×1, 它只能捕捉全局信息 (整体颜色, 类别)。 把这个全局信息从 $f$ 里减掉, 剩下的 $f$ 就是"全局信息解释不了的细节"。然后第二个 scale $r_2$ (2×2) 来解释这些细节。再把 $r_2$ 能解释的减掉, 交给 $r_3$。如此往复。

这和 RQ-Transformer 的 residual 思想类似, 但 RQ 是在同一个 spatial 位置堆叠多个 code, VAR 是在 spatial 维度上做 multi-scale decomposition。

### Shared codebook

一个关键 design choice: 所有 10 个 scale 共享同一个 codebook $Z \in \mathbb{R}^{4096 \times C}$。这样 VAR transformer 的 prediction head 是统一的, 不用每个 scale 学一套 vocabulary。

这其实挺反直觉的 — 1×1 scale 和 16×16 scale 看到的是完全不同的信息 (全局 vs 局部), 用同一套 codebook 真的 OK 吗? 但实验证明 work, 可能是因为 codebook 够大 (4096), 而且每个 code 是一个 abstract feature vector, 不限于 specific scale。

### Decoder (Algorithm 2)

Reconstruction 就是 encoding 的逆过程:

```
f_hat = 0
for k = 1 to K:
    z_k = codebook_lookup(r_k)
    z_k = upsample(z_k, (h_K, w_K))
    f_hat = f_hat + conv_k(z_k)        # 累加每个 scale 的贡献
image_hat = decoder(f_hat)
```

简单粗暴地把所有 scale 的信息加起来, 再 decode 成 image。

---

## VAR Transformer: 简单到令人发指

### 架构

就是 GPT-2 + AdaLN。Paper 特意强调没用 RoPE, 没用 SwiGLU, 没用 RMSNorm。就是为了 isolate VAR algorithm 本身的 contribution, 不让 fancy architecture 抢功。

参数 scaling rule (公式 7, 8):
- width $w = 64d$ (d 是 depth)
- heads $h = d$
- dropout $dr = 0.1 \cdot d/24$
- 总参数 $N(d) = 73728 d^3$

所以 depth 30 的 model 大约是 2B 参数。

### Block-wise causal attention

训练时用 block-wise causal mask:
- Scale $k$ 内部的 tokens 互相 fully visible (bidirectional)
- Scale $k$ 的 tokens 可以看到 scale $1, 2, ..., k-1$ 的所有 tokens
- 但不能看到 scale $k+1, ..., K$ 的 tokens

这样一次 forward pass 就能算所有 scale 的 loss, 训练效率很高。

Inference 时用 KV-cache, 逐步生成: scale 1 → scale 2 → ... → scale 10。每个 scale 内部 parallel decoded。

### QK Normalization

一个小 trick: 把 attention 的 query 和 key 都 normalize 到 unit norm:
$$\hat{q} = q / \|q\|_2, \quad \hat{k} = k / \|k\|_2$$

这个在 Stable Diffusion 3 和 ViT-22B 里也用了, 主要解决 deep transformer 训练时 attention logits 爆炸的问题。Ablation 显示 FID 从 3.60 降到 3.30。

---

## Scaling Laws: 这才是最 exciting 的部分

Karpathy 你对 LLM scaling laws 很熟, VAR 展示出的 behavior 几乎一模一样。

### Power-law 形式

$$L = (\beta \cdot X)^\alpha$$

$X$ 可以是 params $N$, compute $C_{\min}$, 或 tokens $T$。取 log 后是线性关系: $\log L = \alpha \log X + \alpha \log \beta$。

### 实测结果

**Model size**:
$$L_{\text{last}} = (2.0 \cdot N)^{-0.23}$$

LLM 的 Kaplan scaling law 给的是 $\alpha \approx -0.076$, Chinchilla 大约 $-0.1$。VAR 的 $-0.23$ **比 LLM 还陡**。这意味着 VAR 在 vision domain 的 scaling efficiency 更高。

**Compute**:
$$L_{\text{last}} = (2.2 \times 10^{-5} \cdot C_{\min})^{-0.13}$$

Pearson correlation $-0.998$ — 这个数字太强了, 几乎完美 linear in log-log space。

**Token error rate**:
$$Err_{\text{last}} = (4.9 \times 10^2 N)^{-0.016}$$

Error rate 下降很慢 ($-0.016$), 但 loss 下降快。这说明 model 主要在 calibrate probability distribution (让正确 token 的 prob 更高), 而不是改变 top-1 prediction。

### 为什么这很重要

Scaling laws 是 LLM 成功的基石 — 它让你能用小 model 预测大 model 的性能, 从而合理分配 compute。Vision 之前一直没有这么 clean 的 scaling law。Diffusion model 的 scaling behavior 是 messy 的, DiT 在 675M 之后就 saturate 了 (Figure 3 最 striking 的信息)。

VAR 显示出 LLM-grade 的 scaling law, 这意味着:
1. Vision AR 终于可以像 LLM 一样 "scale up and move closer to AGI"
2. 可以用小 model 做实验, 预测大 model 性能, 省 compute
3. 证明了 "next-token/next-scale prediction + scaling" 这个 paradigm 可以 transfer 到 vision

---

## 实验结果: 数字说话

### ImageNet 256×256 (Table 1)

| Model | Params | FID | IS | Steps | Wall-clock |
|-------|--------|-----|-----|-------|------------|
| DiT-XL/2 | 675M | 2.27 | 278 | 250 | 45× |
| L-DiT-3B | 3B | 2.10 | 304 | 250 | >45× |
| L-DiT-7B | 7B | 2.28 | 316 | 250 | >45× |
| VQGAN | 1.4B | 15.78 | 74 | 256 | 24× |
| **VAR-d30** | 2B | **1.92** | 323 | 10 | **1×** |
| **VAR-d30-re** | 2B | **1.73** | **350** | 10 | **1×** |

几个关键 takeaway:

1. **VAR beat DiT**。 2B params 的 VAR 干过了 3B 和 7B 的 L-DiT。这是 GPT-style AR 首次超越 diffusion transformer。

2. **DiT 在 675M 后就 saturate 了**。 L-DiT-7B (2.28) 比 L-DiT-3B (2.10) 还差! 这说明 diffusion 在 vision 上的 scaling 已经碰到瓶颈。VAR 则持续提升 (d16→d20→d24→d30 单调下降)。

3. **20× 速度提升**。 VAR 10 步生成, DiT 250 步。Wall-clock 上 VAR 快 45× (相对 DiT-XL/2)。

4. **数据效率**。 VAR 训练 350 epoch, DiT 训练 1400 epoch。VAR 用 1/4 的数据就能达到更好效果。

### Ablation (Table 3)

| 配置 | FID |
|------|-----|
| AR baseline (VQGAN) | 18.65 |
| 只把 AR 换成 VAR, 其他不变 | 5.22 |
| + AdaLN | 4.95 |
| + Top-k sampling | 4.64 |
| + Classifier-free guidance | 3.60 |
| + QK normalization | 3.30 |
| + Scale to 2B | 1.73 |

最 striking 的是第二行: **仅仅把算法从 AR 换成 VAR, 其他什么都不改, FID 从 18.65 飙到 5.22**。13.43 的 absolute improvement, 全靠 next-scale prediction 这一个 idea。

---

## Zero-shot Generalization

VAR 在 in-painting, out-painting, class-conditional editing 上都能 zero-shot 做, 不用 fine-tune。

方法很简单: inference 时把已知区域的 tokens teacher-force, 只让 model 生成 mask 内的 tokens。

这能 work 是因为 VAR 的 bidirectional-within-scale 结构。传统 AR 做 in-painting 很尴尬 — 如果 mask 在 image 中间, 你 raster-scan 扫到 mask 时, 前面的 tokens 是未知的, 你没法 condition on 未知的东西。

VAR 没这个问题: 每个 scale 内 tokens 互相可见, 只要给 prefix scales + mask 外的 tokens 就行。

这呼应了 paper Section 3.1 提的 AR 第二个 weakness: unidirectional 导致无法做某些 zero-shot task。

---

## 我的一些 deeper thoughts

### VAR 为什么能 beat DiT?

Karpathy, 我的 hypothesis:

**1. Discrete vs continuous representation**。 VAR 用 VQ codes (discrete), DiT 用 continuous latent。Discrete 的好处是 model 学 categorical distribution, loss 是标准 cross-entropy, gradient well-behaved。Diffusion 的 noise prediction 在不同 noise level 上 gradient behavior 变化大, 需要精心设计 noise schedule。

**2. Multi-scale inductive bias**。 Coarse-to-fine 是很强的 prior, 符合自然图像的 hierarchical structure。Diffusion 理论上也能 learn 这个, 但需要 model 自己 discover。VAR 直接 hard-code 了。

**3. Loss landscape 更 friendly**。 Cross-entropy 是 convex 的 (在 logit 空间), gradient 很 stable。Diffusion 的 loss 涉及 noise level 的加权积分, optimization landscape 更复杂。

**4. KV-cache 让 inference 更高效**。 10-step AR + KV-cache 天然适合高效推理。Diffusion 的 250-step reverse process 无法 KV-cache。

### 与 LLM 的深度类比

VAR 和 LLM 的 similarity 是 structural 的:
- Tokenization (VQVAE ↔ BPE)
- Discrete codes
- Cross-entropy loss
- Next-token/next-scale prediction
- KV-cache
- Top-k sampling
- Classifier-free guidance (类似 conditional generation)

这种 similarity 意味着 VAR 很容易和 LLM 整合。Paper Section 8 提到 text-to-image 是 future work。如果 VAR 接到 LLM 上当 image head, 多模态 unified model 就水到渠成了。

### Video generation 的想象空间

Paper Section 8 说 video 可以 extend 成 "3D next-scale prediction":
- 把 video 看作 $(T, H, W)$ 3D pyramid
- 每个 scale 是 $(T_k, H_k, W_k)$
- Coarse-to-fine 在时间维度也 apply

如果这个 work, 它可能解决 SORA 等 diffusion video model 的 temporal consistency 问题 — 因为 VAR 的每个 scale 在时间维度是 global 的, 天然有 temporal coherence。

### Limitations

1. **VQVAE 是 bottleneck**。 Paper 用的 vanilla VQGAN, FID reconstruction 可能 ~5。换上 MagViT-2 的 LFQ 或 FSQ 应该能进一步提升。

2. **Rejection sampling**。 FID 1.73 用了 rejection sampling, raw FID 1.92。实际部署可能有问题。

3. **只在 ImageNet class-conditional 上验证**。 Text-to-image 还没做。

4. **Inference 时 parallel decoding 的细节**。 Paper 说每个 scale 内 parallel sample, 但没说怎么处理 token 之间的 spatial correlation。如果是 independent sampling, 可能产生 spatially inconsistent tokens。这个细节需要查 code。

---

## 公式速查 (给你 Karpathy 做参考)

### VAR likelihood (公式 6)

$$p(r_1, r_2, \ldots, r_K) = \prod_{k=1}^{K} p(r_k \mid r_1, r_2, \ldots, r_{k-1})$$

- $r_k \in [V]^{h_k \times w_k}$: 第 $k$ 个 scale 的 token map
- $K$: scale 数量 (10)
- 每个 $p(r_k | r_{<k})$ 是 $h_k \times w_k$ 个 tokens 的 joint distribution, inference 时 parallel 采样

### 参数量 (公式 8)

$$N(d) = \underbrace{d \cdot 4w^2}_{\text{attn}} + \underbrace{d \cdot 8w^2}_{\text{FFN}} + \underbrace{d \cdot 6w^2}_{\text{AdaLN}} = 18dw^2 = 73728d^3$$

- $d$: depth, $w = 64d$: width
- Attn: Q/K/V/O, 4 个 $w \times w$
- FFN: expand to $4w$ + contract, 共 $8w^2$
- AdaLN: 6 个 modulation (scale+shift for attn, FFN, residual)

### Scaling laws (公式 11-16)

$$L_{\text{last}} = (2.0 \cdot N)^{-0.23}, \quad L_{\text{avg}} = (2.5 \cdot N)^{-0.20}$$

$$L_{\text{last}} = (2.2 \times 10^{-5} \cdot C_{\min})^{-0.13}$$

Pearson $-0.998$, 非常 clean 的 power-law。

---

## 最终总结

Karpathy, VAR 这篇 paper 的意义我觉得有三层:

**第一层: 一个很好的 engineering insight**。 把 image AR 从 token-level 改成 scale-level, 解决了 efficiency 和 spatial locality 问题。简单, 有效, elegant。

**第二层: 首次让 AR beat DiT**。 这打破了 "diffusion is king for image generation" 的迷思。GPT-style AR 在 vision 上也能做到 SOTA, 甚至更好。

**第三层: 证明了 LLM paradigm 可以 transfer 到 vision**。 Scaling laws, zero-shot generalization — 这两个 LLM 的标志性 feature, 在 VAR 上都看到了。这暗示着 vision 可能可以走 LLM 的路: 一个 unified autoregressive model, scale up, emergent ability。

对我来说, 最 exciting 的是第三层。如果这个 paradigm 继续发展 (更好的 tokenizer, text-to-image, video extension), 它可能成为 vision generation 的新 standard, 甚至推动 multimodal AGI。

你 Karpathy 当年写 nanoGPT, 讲 "Let's build GPT" 的时候, 强调的是 AR + scaling 这个 paradigm 的简洁和 power。VAR 把这个 insight 带到了 vision, 而且 initial validation 很成功。这值得关注。

References:
- Paper PDF: https://arxiv.org/abs/2404.02905
- Project page: https://var.vision
- Code: https://github.com/FoundationVision/VAR
- Online demo: https://var.vision
- DiT (Peebles & Xie, ICCV 2023): https://arxiv.org/abs/2212.09748
- VQGAN (Esser et al., CVPR 2021): https://arxiv.org/abs/2012.09841
- RQ-Transformer (Lee et al., CVPR 2022): https://arxiv.org/abs/2203.01941
- MagViT-2 (Yu et al., 2023): https://arxiv.org/abs/2310.05737
- SORA (OpenAI, 2024): https://openai.com/research/video-generation-models-as-world-simulators
- Kaplan scaling laws: https://arxiv.org/abs/2001.08361
- Chinchilla (Hoffmann et al., 2022): https://arxiv.org/abs/2203.15556

---

# VAR (Visual Autoregressive Modeling): Next-Scale Prediction 的深度解析

## 1. 核心直觉: 从 "next-token" 到 "next-scale"

Karpathy 你看这篇 paper 的核心 insight 其实非常 elegant: 人类画画的时候从来不会逐像素 raster-scan 地画,而是先打草稿(全局结构),再逐步细化(局部细节)。VAR 把这个 intuition 形式化为 **next-scale prediction**。

传统 image AR (VQGAN, DALL-E, Parti) 的问题是:它们把 2D token map 强行 flatten 成 1D 序列,然后用语言模型的方式 left-to-right, top-to-bottom 生成。这有几个根本性的问题:

**1.1 Mathematical premise violation**

VQVAE encoder 是 CNN + self-attention,它是 bidirectional 的。所以编码出的 token map $q \in [V]^{h \times w}$ 中,每个 token $q^{(i,j)}$ 与所有其他 token 都有 mutual dependency。但 AR 模型假设 $x_t$ 只 depend on prefix $x_{<t}$。这本质上是把一个 bidirectional distribution 强行用 unidirectional chain rule 分解。

Paper 在 Appendix A 用 attention map 给了 empirical evidence (Figure 9): VQGAN encoder 的最后一层 self-attention 显示 tokens 之间有 strong bidirectional dependency。这告诉我们 AR 在 image 上的 mathematical 假设就是 broken 的。

**1.2 Spatial locality 被破坏**

Token map 中 $q^{(i,j)}$ 与 4-neighbors $q^{(i±1,j)}, q^{(i,j±1)}$ 有强 spatial correlation。Flatten 成 raster-scan 序列后,$q^{(i,j)}$ 和 $q^{(i+1,j)}$ 之间隔了 $w$ 个 tokens。虽然理论上 transformer 能学到,但实际上 unidirectional attention 让这种 bidirectional neighbor relation 变得难学。

**1.3 Computational inefficiency**

这个是 paper Appendix B 给的证明,非常关键:

对 $n \times n$ 的 token map,AR 需要生成 $n^2$ 个 tokens,每个 token 第 $i$ 步需要 $\mathcal{O}(i^2)$ 计算(因为 attention with all previous tokens):

$$\sum_{i=1}^{n^2} i^2 = \frac{1}{6} n^2 (n^2+1)(2n^2+1) \sim \mathcal{O}(n^6)$$

也就是说,256×256 ImageNet (token map 16×16 = 256 tokens) 已经是 256 个 AR steps;如果是 1024×1024 (token map 64×64 = 4096 tokens),那就是 4096 steps,而且 attention cost 是 cubic 增长。这根本不可扩展。

VAR 的解法是:每个 scale 内**并行**生成所有 tokens,scale 之间才 autoregressive。对于 $K$ 个 scales $n_1 < n_2 < ... < n_K = n$,第 $k$ 步并行生成 $n_k^2$ 个 tokens,attention 的 total tokens 是 $\sum_{i=1}^k n_i^2$。论文 Lemma B.2 证明 total complexity 是 $\mathcal{O}(n^4)$ - 注意这其实是 paper 里写得稍微乐观了点,严格说每个 scale 的 attention cost 是 $O((\sum_{i=1}^k n_i^2)^2)$,求和后 dominant term 是 $n_K^4 = n^4$。

## 2. Multi-scale VQVAE: Residual Quantization 的精妙之处

这是 VAR 能 work 的 foundation。Paper 用了一个看起来简单但很 elegant 的设计。

### 2.1 Algorithm 1 (Encoding) 详解

```
Input: raw image im
f = E(im)                              # encoder 输出 feature map, shape h×w×C
R = []                                 # multi-scale token maps

for k = 1, ..., K:
    r_k = Q(interpolate(f, h_k, w_k))  # 把 f 下采样到 (h_k, w_k) 然后 quantize
    R.push(r_k)
    z_k = lookup(Z, r_k)               # 用 shared codebook Z 查表得到 quantized vectors
    z_k = interpolate(z_k, h_K, w_K)   # 上采样回原分辨率 h_K×w_K
    f = f - φ_k(z_k)                   # 残差更新! 关键步骤
```

这里 $\phi_k$ 是第 $k$ 个 scale 的 conv layer (paper 说有 K 个额外的 conv,共 0.03M params)。

**为什么是 residual?** 因为第一个 scale $r_1$ (1×1) 只能捕捉全局信息 (比如整体色调、class)。从 $f$ 中减掉 $\phi_1(z_1)$ 后,剩下的 $f$ 就是"全局信息解释不了的部分",需要 $r_2$ (2×2) 来解释。这个 residual 结构让每个 scale 都专注于"前一个 scale 解释不了的信息"。

这其实和 RQ-Transformer [50] 的 residual 思想类似,但 RQ 是在同一个 spatial 位置堆叠多个 code,VAR 是在 spatial 维度上做 multi-scale。

### 2.2 Algorithm 2 (Decoding/Reconstruction)

```
Input: multi-scale token maps R = (r_1, r_2, ..., r_K)
f_hat = 0
for k = 1, ..., K:
    r_k = R.pop()
    z_k = lookup(Z, r_k)               # shared codebook
    z_k = interpolate(z_k, h_K, w_K)   # 上采样到原分辨率
    f_hat = f_hat + φ_k(z_k)           # 残差累加
im_hat = D(f_hat)
```

Reconstruction 就是 encoding 的逆过程,累加 $\phi_k(z_k)$ 即可。这保证了 encode-decode 的可逆性。

### 2.3 关键设计 choice

- **Shared codebook** across all scales: $Z \in \mathbb{R}^{V \times C}$, $V = 4096$。这避免了每个 scale 学独立的 codebook 导致 vocabulary 爆炸,也让 VAR transformer 的 prediction head 是统一的。
- **K = 10 scales**: paper 用了 $\{(1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (8,8), (10,10), (13,13), (16,16)\}$,从 1×1 到 16×16 (downsample ratio 16×)。Total tokens = 1+4+9+16+25+36+64+100+169+256 = 680,而传统 AR 是 256。所以 VAR 实际上 token 数量更多!但每个 scale 内并行,所以 AR step 数从 256 降到 10。
- **VQGAN compound loss** (公式 5): $\mathcal{L} = \|im - \hat{im}\|_2 + \|f - \hat{f}\|_2 + \lambda_P \mathcal{L}_P + \lambda_G \mathcal{L}_G$。包含 L2 reconstruction、feature L2、LPIPS perceptual loss、StyleGAN discriminator loss。和原 VQGAN [30] 完全一致,只在 quantizer 上做了修改。

## 3. VAR Transformer 架构

### 3.1 整体设计

很简洁 - 就是 GPT-2 style decoder-only transformer + AdaLN。Paper 特意强调 "We do not use advanced techniques in large language models, such as RoPE, SwiGLU MLP, or RMS Norm"。这是为了 isolate VAR algorithm 的 contribution。

**Input format**: $([s], r_1, r_2, ..., r_{K-1}) \to \text{predict} (r_1, r_2, ..., r_K)$

其中 $[s]$ 是 class embedding (作为 start token 和 AdaLN condition)。

### 3.2 Block-wise causal attention

这是 VAR 训练的关键。不是 standard lower-triangular causal mask (那样的话每个 token 只能看到前面的 token),而是 **block-wise**:第 $k$ 个 scale 内部 tokens 互相 fully visible (bidirectional),但只能 attend 到 scale $\leq k$。

具体来说,对于 $r_k$ 中的某个 token,它的 attention 范围是:
- 所有 $r_1, r_2, ..., r_{k-1}$ 中的 tokens (前缀)
- $r_k$ 中的所有 tokens (包括它自己和同 scale 的其他 tokens)

这种 mask 让 training 时一次 forward pass 就能计算所有 scales 的 loss,效率很高。

Inference 时用 KV-cache,逐步生成:scale 1 (1 token) → scale 2 (4 tokens) → ... → scale 10 (256 tokens)。每个 scale 内的 tokens 是一次性 parallel decoded。

### 3.3 Positional embedding

Paper 说每个 $r_k$ 有 "associated k-th position embedding map"。我推测是给每个 scale 内的每个 spatial position 加 2D positional embedding。具体实现细节可能要查 code:https://github.com/FoundationVision/VAR

### 3.4 Parameter scaling rule

公式 (7) 和 (8) 给了 model scaling rule:
- width $w = 64d$
- heads $h = d$
- dropout $dr = 0.1 \cdot d/24$
- params $N(d) = 18dw^2 = 73728d^3$

其中 $d$ 是 depth。这个 cubic scaling ($N \propto d^3$) 是因为 width 和 depth 同步增长。Paper 训练了 $d \in \{6, 8, 10, 12, 16, 20, 24, 30, 36\}$ 等不同 size,从 18M 到 2B 参数。

分解公式 (8):
- Self-attention: $d \cdot 4w^2$ (Q, K, V, O 4 个 linear layer,每个 $w \times w$)
- Feed-forward: $d \cdot 8w^2$ (两个 linear layer,hidden = 4w,所以 $w \times 4w + 4w \times w = 8w^2$ per layer)
- Adaptive LayerNorm: $d \cdot 6w^2$ (AdaLN 通常有 6 个 modulation parameters: scale 和 shift for attention, FFN, 和 residual)

### 3.5 QK Normalization

"We found normalizing queries and keys to unit vectors before attention can stabilize the training."

这个 trick 很重要。具体是把 $q$ 和 $k$ 都 normalize 到 unit norm:
$$\hat{q} = q / \|q\|_2, \quad \hat{k} = k / \|k\|_2$$

然后 attention score 是 $\hat{q} \cdot \hat{k} / \tau$,其中 $\tau$ 是 learnable temperature。

这个 technique 在 stable diffusion 3 [29] 和 ViT-22B 等大模型中也被采用,主要解决 deep transformer 训练时 attention logits 爆炸的问题。

Ablation study Table 3 显示 QK norm 让 FID 从 3.60 降到 3.30。

## 4. Scaling Laws: VAR 模仿 LLM 的关键证据

这是 paper 最 exciting 的部分。Karpathy 你对 LLM scaling laws 很熟悉,VAR 显示出几乎 identical 的 power-law behavior。

### 4.1 公式形式

公式 (9): $L = (\beta \cdot X)^\alpha$

其中 $X$ 可以是 $N$ (params), $T$ (tokens), 或 $C_{\min}$ (optimal compute)。$\alpha$ 是 power-law exponent,$\beta$ 是 prefactor。

取 log: $\log L = \alpha \log X + \alpha \log \beta$ - 这就是 log-log 空间的线性关系。

### 4.2 实测结果

**Model size scaling** (公式 11):
- $L_{\text{last}} = (2.0 \cdot N)^{-0.23}$
- $L_{\text{avg}} = (2.5 \cdot N)^{-0.20}$

对比 LLM 的 scaling law (Kaplan et al. [43] 给出 $\alpha \approx -0.076$ for test loss vs params,Chinchilla [38] 给出更好的 exponents),VAR 的 $\alpha \approx -0.2$ 实际上**比 LLM 更陡峭**。这意味着 VAR 在 vision domain 的 scaling efficiency 更高。

**Compute scaling** (公式 13-16):
- $L_{\text{last}} = (2.2 \times 10^{-5} \cdot C_{\min})^{-0.13}$
- $L_{\text{avg}} = (1.5 \times 10^{-5} \cdot C_{\min})^{-0.16}$

Pearson correlation 接近 $-0.998$ - 这是 paper 最强的 claim 之一。说明 VAR 真的有 LLM-style 的 scaling laws。

### 4.3 Token error rate 也 scale

$$Err_{\text{last}} = (4.9 \times 10^2 N)^{-0.016}$$

虽然 exponent 很小 ($-0.016$),但仍然是 power-law decrease。这表明随着 model 增大,token prediction accuracy 也在提升。

## 5. 实验结果分析

### 5.1 ImageNet 256×256 (Table 1)

VAR-d30 (2B params):
- **FID = 1.73** (with rejection sampling)
- IS = 350.2
- 10 steps (vs DiT 250 steps, VQGAN 256 steps)
- 1× wall-clock (vs DiT-XL/2 45×, VQGAN 24×)

对比:
- DiT-XL/2 (675M): FID 2.27
- L-DiT-3B: FID 2.10
- L-DiT-7B: FID 2.28 (!!! 7B 反而比 3B 差,说明 DiT 在 675M 后就不 scale 了)
- VQGAN (1.4B, with rejection): FID 5.20
- VAR-d30 (2B): FID 1.92 (无 rejection), 1.73 (有 rejection)

关键 observation: **DiT 在 675M 后 scaling saturate 甚至 negative**。这是 Figure 3 最 striking 的信息。VAR 则持续提升。

### 5.2 ImageNet 512×512 (Table 2)

VAR-d36-s: FID 2.63, IS 303.2, 1× wall-clock
DiT-XL/2: FID 3.04, 81× wall-clock

VAR 在更高分辨率上优势更明显。

### 5.3 Ablation (Table 3)

| Component | FID |
|-----------|-----|
| AR baseline | 18.65 |
| AR → VAR (only change algorithm) | 5.22 |
| + AdaLN | 4.95 |
| + Top-k sampling | 4.64 |
| + CFG (ratio 2.0) | 3.60 |
| + QK Norm | 3.30 |
| + Scale to 2B | 1.73 |

最重要的 row 是第 2 行:仅仅把 AR 换成 VAR,其他什么都不变,FID 从 18.65 降到 5.22。这是 **13.43 的 absolute improvement**,证明了 next-scale prediction 这个 idea 本身的 power。

## 6. Zero-shot Generalization

Paper 展示了 VAR 在 in-painting, out-painting, class-conditional editing 上的 zero-shot 能力。方法很简单:在 inference 时把已知区域的 tokens teacher-force,只让 model 生成 mask 内的 tokens。

这能 work 是因为 VAR 的 bidirectional-within-scale + coarse-to-fine 结构。传统 AR (raster-scan) 做 in-painting 很 awkward,因为生成顺序固定 - 如果 mask 在 image 中间,前面的 tokens 是未知的。VAR 没这个问题:每个 scale 内 tokens 互相可见,只要给 model prefix scales + mask 外的 tokens 就行。

这其实呼应了 paper Section 3.1 提到的 AR 的第二个 weakness:"Inability to perform some zero-shot generalization"。

## 7. 与 Diffusion 的对比 - 我的深入思考

Karpathy,我觉得这篇 paper 最 deep 的 contribution 不仅是 FID 数字,而是它揭示了 vision generation 的一个新方向。

### 7.1 为什么 VAR 能 beat DiT?

我的 hypothesis:

**1. Discrete vs continuous representation**。VAR 用 VQ codes (discrete),DiT 用 continuous latent。Discrete representation 的好处是 model 可以学习 categorical distribution,而 continuous 需要 Gaussian 或复杂的 noise schedule。

**2. Multi-scale inductive bias**。VAR 的 coarse-to-fine 是很强的 prior,符合自然图像的 hierarchical structure。Diffusion 虽然理论上也能 learn 这个,但需要 model 自己 discover,而 VAR 是 hard-coded 的。

**3. Better loss landscape**。VAR 是 standard cross-entropy loss,gradient 是 well-behaved 的。Diffusion 的 loss 涉及 noise prediction,gradient 在不同 noise level 上变化很大,需要 carefully design noise schedule。

**4. KV-cache efficiency**。VAR 的 10-step inference 比 DiT 的 250-step 快 25×。这不仅是 wall-clock 优势,也意味着 VAR 的 "information bottleneck" 更小 - 每个 step 要一次性决定整个 scale 的所有 tokens,而不是慢慢 denoise。

### 7.2 与 LLM 的类比

VAR 真的很像 LLM:
- Tokenization (VQVAE ↔ BPE)
- Discrete codes
- Cross-entropy loss
- Next-token/next-scale prediction
- KV-cache for fast inference
- Top-k sampling
- Classifier-free guidance (类似 LLM 的 conditional generation)

这种 similarity 让 VAR 很容易和 LLM 整合。Paper 在 Section 8 提到 text-to-image 是 future work,我推测应该是把 VAR 当作 LLM 的 "image generation head"。

### 7.3 Limitations

1. **VQVAE 是 bottleneck**。Paper 用的是 vanilla VQGAN (OpenImages 训练,FID reconstruction 可能 ~5)。如果用更好的 tokenizer (比如 MagViT-2 [95] 的 LFQ,或 FSQ [59]),VAR 应该还能提升。

2. **Class-conditional only**。目前只做了 ImageNet class-conditional,没有 text-to-image。但架构上 extend 应该不难。

3. **256×256 token map 是 16×16**。更高分辨率需要更多 scales 或更大 token map。Paper 在 512×512 上用了 d36-s (single AdaLN),说明 compute 仍然有限制。

4. **Rejection sampling 依赖**。FID 1.73 用了 rejection sampling,raw FID 是 1.92。这在实际部署时可能有问题。

## 8. Formula 详细解读

### 公式 (1): Standard AR likelihood

$$p(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_1, x_2, \ldots, x_{t-1})$$

- $x_t \in [V]$: 第 $t$ 个 token,$V$ 是 vocabulary size
- $T$: sequence length
- 等式左边:整个序列的 joint probability
- 等式右边:chain rule 分解,每个 conditional probability 只 depend on prefix

### 公式 (3): VQ Quantization

$$q^{(i,j)} = \arg\min_{v \in [V]} \|\text{lookup}(Z, v) - f^{(i,j)}\|_2$$

- $f^{(i,j)} \in \mathbb{R}^C$: encoder 输出的 feature map 在位置 $(i,j)$ 的 feature vector
- $Z \in \mathbb{R}^{V \times C}$: codebook,$V$ 个 code,每个 code 是 $C$ 维
- $\text{lookup}(Z, v)$: 取 codebook 第 $v$ 个 code
- $q^{(i,j)} \in [V]$: 量化后的 token index,是 nearest neighbor in Euclidean space

### 公式 (6): VAR likelihood

$$p(r_1, r_2, \ldots, r_K) = \prod_{k=1}^{K} p(r_k \mid r_1, r_2, \ldots, r_{k-1})$$

- $r_k \in [V]^{h_k \times w_k}$: 第 $k$ 个 scale 的 token map,$h_k \times w_k$ 个 tokens
- $K$: scale 数量 (paper 用 10)
- 每个 $p(r_k | r_{<k})$ 是 $h_k \times w_k$ 个 tokens 的 joint distribution,但 inference 时 parallel 采样

### 公式 (8): Parameter count

$$N(d) = \underbrace{d \cdot 4w^2}_{\text{self-attn}} + \underbrace{d \cdot 8w^2}_{\text{FFN}} + \underbrace{d \cdot 6w^2}_{\text{AdaLN}} = 18dw^2 = 73728d^3$$

- $d$: transformer depth (number of layers)
- $w = 64d$: hidden width
- Self-attention: 4 个 linear projections (Q, K, V, output),每个 $w \times w$
- FFN: 2 个 linear (expand to $4w$, contract back),总 $2 \times w \times 4w = 8w^2$
- AdaLN: 6 个 modulation (scale+shift for attn, FFN, residual),每个 $w$ 维,总 $6w^2$ per layer

代入 $w = 64d$:$18d \cdot (64d)^2 = 18 \cdot 4096 \cdot d^3 = 73728 d^3$。

对于 $d = 30$: $N \approx 73728 \times 27000 \approx 2.0 \times 10^9 = 2B$。验证通过。

### 公式 (11)-(12): Scaling laws with N

$$L_{\text{last}} = (2.0 \cdot N)^{-0.23}, \quad L_{\text{avg}} = (2.5 \cdot N)^{-0.20}$$

- $L_{\text{last}}$: 最后一个 scale (16×16) 的 test cross-entropy loss
- $L_{\text{avg}}$: 所有 scales 平均的 test loss
- $N$: parameter count
- Exponent $-0.23$ vs LLM 的 $-0.076$ (Kaplan) 或 $-0.1$ (Chinchilla): VAR 的 scaling **更高效**

$$Err_{\text{last}} = (4.9 \times 10^2 N)^{-0.016}$$

- $Err$: token prediction error rate (top-1 accuracy 的 1 - acc)
- Exponent 很小 $-0.016$,说明 error rate 下降慢,但 loss 下降快 - 这意味着 model 主要在 calibrate probability distribution,而不是改变 top-1 prediction

## 9. 架构图深度解析 (Figure 4)

Figure 4 展示了 VAR 的 two-stage training:

**Stage 1: Multi-scale VQVAE training**
- Input: raw image $im$
- Encoder $\mathcal{E}$: CNN + attention,输出 feature map $f$
- Multi-scale quantization (Algorithm 1): 把 $f$ 编码成 10 个 token maps $(r_1, ..., r_{10})$
- Decoder $\mathcal{D}$: 从 token maps reconstruct $\hat{im}$
- Loss: compound loss (5),包括 L2, perceptual, adversarial
- 训练数据: OpenImages,与 VQGAN baseline 相同

**Stage 2: VAR transformer training**
- Input: $([s], r_1, r_2, ..., r_{K-1})$
- Target: $(r_1, r_2, ..., r_K)$ (shifted by one scale)
- Architecture: GPT-2 + AdaLN,block-wise causal mask
- Loss: standard cross-entropy
- Training data: ImageNet (class-conditional)

注意 Stage 1 和 Stage 2 用不同的数据集!Stage 1 在 OpenImages 训练 tokenizer,Stage 2 在 ImageNet 训练 transformer。这是 common practice (tokenizer 要 generalize,transformer 专注 downstream task)。

## 10. 与最新工作的关联

### 10.1 与 SORA 的关系

SORA [14] 是 Diffusion Transformer for video。Paper 在 abstract 和 introduction 都提到 SORA,暗示 VAR 可以作为 alternative。Section 8 "Video generation" 明确说:

"By considering multi-scale video features as 3D pyramids, we can formulate a similar '3D next-scale prediction' to generate videos via VAR."

这个想法很有意思:把 video 看作 $(T, H, W)$ 3D pyramid,每个 scale 是 $(T_k, H_k, W_k)$。Coarse-to-fine 在时间维度也 apply,可能能解决 long video generation 的 temporal consistency 问题。

### 10.2 与 MagViT-2 / FSQ 的关系

Paper Section 8 提到 "advancing VQVAE tokenizer [99, 59, 95] as another promising way"。MagViT-2 的 LFQ (Lookup-Free Quantization) 和 FSQ (Finite Scalar Quantization) 都能提供更好的 codebook utilization。如果 VAR 换上更好的 tokenizer,FID 应该能进一步下降。

### 10.3 与 LlamaGen 的对比

LlamaGen (Sun et al., 2024, Apple) 是同期工作,用 Llama-style AR + improved VQVAE,也 beat 了 DiT。但 LlamaGen 仍然是 next-token (raster-scan),没有 VAR 的 next-scale 设计。VAR 在 efficiency 上应该有优势 (10 steps vs LlamaGen 的 256+ steps)。

## 11. 我的批判性思考

### 11.1 VAR 真的 "first time" beat DiT 吗?

Paper abstract 说 "for the first time, makes GPT-style AR models surpass diffusion transformers"。这个 claim 要小心:
- VAR-d30-re 用了 rejection sampling
- 不用 rejection 的 VAR-d30 FID = 1.92,仍然 beat DiT-XL/2 (2.27) 但和 L-DiT-3B (2.10) 接近

不过即使 raw FID 1.92 也是 strong result,claim 基本成立。

### 11.2 Scaling law 的 $\alpha$ 真的比 LLM 大吗?

VAR 的 $\alpha \approx -0.2$ 看起来比 LLM 的 $-0.076$ 大,但要注意:
- LLM 的 loss 是 over thousands of tokens averaged,VAR 是 over 680 tokens
- LLM vocabulary ~30K,VAR vocabulary 4096
- 不同 loss scale 可能影响 $\alpha$ 的比较

所以直接比较 $\alpha$ 可能不公平。但 VAR 的 Pearson $-0.998$ 确实是 strong evidence of power-law scaling。

### 11.3 Inference 的 parallel decoding 细节

Paper 说 "all distributions over the $h_k \times w_k$ tokens in $r_k$ will be generated in parallel"。但 inference 时怎么 sample 呢?是 independent sample 每个 token,还是有 correlation?

我推测是 independent sample (从 categorical distribution),但这可能产生 spatially inconsistent tokens。也许需要 iterative refinement 或 mask prediction 在 scale 内部?Paper 没有详细说明,可能需要查 code:https://github.com/FoundationVision/VAR

实际上从 Top-k sampling 的 ablation 看,VAR 用了 top-k categorical sampling,应该是 independent。

### 11.4 为什么不 bidirectional across scales?

VAR 是 scale 之间 unidirectional ($r_k$ only depends on $r_{<k}$)。但理论上可以想象 bidirectional multi-scale model (类似 diffusion 的 bidirectional Markov chain)。为什么不呢?

我的猜测:
1. Unidirectional 才能 KV-cache,实现 fast inference
2. Coarse-to-fine 的 prior 本身就符合 causal interpretation
3. Bidirectional 会变成 masked prediction (像 MaskGIT),失去 AR 的 simplicity

## 12. 实现细节的疑问

读完 paper 我还有几个疑问,可能需要查 code:

1. **Positional embedding 怎么处理不同 scale?** 每个 scale 的 token map 大小不同 (1×1 到 16×16),怎么 encode spatial position?
2. **Class embedding 怎么 inject?** 是只作为 start token,还是每个 scale 都加?
3. **CFG 怎么实现?** Classifier-free guidance 需要conditional 和 unconditional 两个 forward pass。VAR 是怎么做的?
4. **Top-k sampling 在每个 scale 是独立的吗?** 还是有什么 correlation mechanism?

这些细节可以查:https://github.com/FoundationVision/VAR

## 13. 总结

VAR 是一个 elegant 且 effective 的工作。它的核心贡献:

1. **Conceptual**: 把 image AR 从 "next-token" 重新定义为 "next-scale",解决了 AR 在 image 上的 mathematical 和 efficiency 问题
2. **Empirical**: 首次让 GPT-style AR 在 image generation 上 beat DiT,FID 1.73 with 20× speedup
3. **Scientific**: 展示了 VAR 的 scaling laws,with Pearson $-0.998$,这是 vision AR 朝 LLM-style scaling 迈进的重要 evidence
4. **Practical**: 开源 code 和 model,推进 community 发展

对于你 Karpathy 来说,VAR 的意义可能更深层:它证明了 LLM 的 success paradigm (next-token prediction + scaling laws + zero-shot generalization) 可以 transfer 到 vision。这不是简单的 architecture copy,而是理解了 AR 的本质 - "define a meaningful order on data" - 然后为 vision 找到正确的 "order" (multi-scale coarse-to-fine)。

这种 conceptual insight 比 FID 数字更重要,因为它指明了 vision AR 的未来方向。如果 VAR 的 paradigm 继续 scale (比如换上更好的 tokenizer,扩展到 text-to-image,扩展到 video),它有可能成为 vision generation 的新 standard。

References:
- Paper: https://var.vision
- Code: https://github.com/FoundationVision/VAR
- Online demo: https://var.vision
- DiT (Peebles & Xie, ICCV 2023): https://arxiv.org/abs/2212.09748
- VQGAN (Esser et al., CVPR 2021): https://arxiv.org/abs/2012.09841
- RQ-Transformer (Lee et al., CVPR 2022): https://arxiv.org/abs/2203.01941
- MagViT-2 (Yu et al., 2023): https://arxiv.org/abs/2310.05737
- SORA (OpenAI, 2024): https://openai.com/research/video-generation-models-as-world-simulators
- Kaplan scaling laws: https://arxiv.org/abs/2001.08361
- Chinchilla: https://arxiv.org/abs/2203.15556
- LlamaGen: https://arxiv.org/abs/2310.05737 (实际上这是 MagViT-2,LlamaGen 是 https://arxiv.org/abs/2310.05737 - 需要核实)
