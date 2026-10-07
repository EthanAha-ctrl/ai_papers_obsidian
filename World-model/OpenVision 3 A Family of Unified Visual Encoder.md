---
source_pdf: OpenVision 3 A Family of Unified Visual Encoder.pdf
paper_sha256: b3a0ee36ca88571f6db0b7571fd6d3795881f6cbf55601abd9c07ee0e5a23ac9
processed_at: '2026-08-06T01:02:21-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

咱们用大白话来拆解一下 OpenVision 3 这篇 paper 的核心 intuition。这篇 paper 其实就在回答一个根本问题：**能不能用一个 visual encoder 同时搞定图像理解与图像生成？**

### 1. 问题的本质：为什么 traditionally 要分开做？

以前大家觉得这俩任务是矛盾的。
Image understanding（比如 CLIP）要把图像压缩成一个高度抽象的 semantic label，比如看到一张狗的图，最好输出一个“狗”的 concept，丢掉所有毛发纹理、背景细节，这样方便跟 text 对齐。
Image generation（比如 VAE / Diffusion）反过来，必须死死记住所有的 pixel-level 细节、高频纹理，否则画出来的图就是糊的。

因为目标冲突，主流的 Unified Multimodal Models (UMMs) 比如 BAGEL 或者 Uni-Fluid，通常采用 dual tokenizer：拿 SigLIP 提取 semantic tokens 给 LLM 看图说话，拿 FLUX-VAE 提取 pixel tokens 给 Diffusion 画画。这样做 system 很复杂，而且两套特征很难真正融合。另一派用 VQ-GAN 把图像变成离散 token，像文字一样处理，但 quantization 一定会丢信息，画出来的图质量上限很低。

OpenVision 3 给了一个非常 elegant 的解法：用 continuous representation，且只用一个 encoder。

### 2. OpenVision 3 的架构直觉：在“好底片”上学语义

最核心的 insight：**直接在 VAE 的 latent space 里面训练 ViT**。

咱们可以把 FLUX-VAE 想象成一个顶级摄影助理，它把原图拍成了一张极高画质的“底片”（$z_{vae}$）。这张底片已经把冗余的 pixel 压缩掉了，但保留了所有重建图像必需的结构和纹理，并且这个 latent space 天生就是 perceptually aligned 的。

OpenVision 3 的操作就是拿一个 randomly initialized 的 ViT，直接吃这张底片：

$$z_u = \mathcal{E}_{vit}(z_{vae}) \in \mathbb{R}^{\frac{H}{16} \times \frac{W}{16} \times D_u}$$

- $z_{vae}$: VAE 输出的 latent feature
- $\mathcal{E}_{vit}$: 从 scratch 训练的 ViT encoder
- $z_u$: 输出的 unified representation
- $\frac{H}{16}$: 因为 VAE downsample 8x，ViT patch size 是 2x2，所以总 downsample 是 16x

得到 $z_u$ 之后，它分出两个 branch：
**Reconstruction branch**：给 $z_u$ 加点高斯噪声（模拟 diffusion 的 forward process），然后用 ViT decoder 尝试还原回 VAE latent，再过 frozen VAE decoder 还原图像。这迫使 $z_u$ 保留所有的 pixel 细节。
**Understanding branch**：拿 $z_u$ 去跟 text 做 contrastive learning，并且让 text decoder 去 autoregressive 地预测 caption。这迫使 $z_u$ 学会 high-level semantic。

由于两个 branch 共享同一个 $z_u$，ViT encoder 被迫学出一个同时包含 semantic concept 和 pixel detail 的表征。

### 3. 最反直觉的发现：Mutual Synergy

这篇 paper 最精彩的 part 是 Section 5.1 的 ablation。通常我们以为 joint training 是一种 trade-off，懂语义的模型画画会变差，懂画画的模型语义会变差。但作者发现这俩任务居然 **互相促进**。

实验发现：
如果你把 reconstruction loss 拿掉，只训练 understanding loss，模型的 pixel recon loss 居然自己就降下去了！
如果你把 understanding loss 拿掉，只训练 reconstruction loss，模型的 captioning loss 也会有轻微下降（虽然 contrastive loss 停滞了）。

**直觉解释**：
Semantic supervision 给了模型一个“世界模型”的 prior。如果 ViT 知道这张图里是一只猫坐在草地上，它在做 reconstruction 的时候，就有了搜索空间的约束，知道该往猫和草的方向还原细节，这比盲目重建 pixel 要容易得多。反之，reconstruction 任务强迫 ViT 去看懂图像的细微结构，这种对细节的把握反过来帮助模型更准确地 caption 出图里的内容。这印证了 Platonic Representation Hypothesis：理解与生成在底层共享同一种计算逻辑。

### 4. 为什么一定要用 VAE？（Tables 6, 7, 8）

作者做了一个 ablation，去掉 VAE，直接让 ViT 处理 raw image patches（类似于普通的 CLIP）。

| Model | rFID↓ (Reconstruction) | gFID↓ (Generation) |
|---|---|---|
| w/o VAE (Raw ViT) | 0.980 | 9.68 |
| OpenVision 3 (VAE + ViT) | 0.216 | 8.45 |

**直觉解释**：
如果没有 VAE，ViT 要从 raw pixel 一步一步学到高频纹理和语义，这个跨度太大，很容易顾此失彼。
有了 VAE，VAE 已经把 raw pixel 里那些难以预测的高频噪声滤掉了，变成了一个高度结构化的 latent space。ViT 在这个已经“清洗”过的空间里做 semantic 对齐，阻力极小，同时保留了生成所需的底子。并且，理解任务也从这个 clean latent space 中获益了，在 LLaVA-NeXT framework 下，带 VAE 的版本在所有 6 个 understanding benchmark 上全面领先无 VAE 版本。

### 5. 实验数据直觉解读

#### Reconstruction (Table 2)
- 专门做生成的 **FLUX-VAE**: rFID 0.176
- 之前的 unified tokenizer **UniTok**: rFID 0.362
- **OpenVision 3**: rFID 0.187

这就很离谱了。OpenVision 3 作为一个 unified tokenizer，重建质量几乎逼近了专门做生成的 FLUX-VAE，直接碾压了所有其他 unified tokenizer。这说明 semantic loss 完全没有拖累 reconstruction，反而帮了忙。

#### Generation (Table 3)
- 语义 encoder **CLIP + RAE**: gFID 2.54
- **OpenVision 3 + RAE**: gFID 1.87

用同样的 RAE 生成框架，OpenVision 3 的 token 质量远超 CLIP。因为 CLIP 输出的 feature 太过 abstract，丢掉了生成所需的细节，而 OpenVision 3 保留了这些。

#### Understanding (Tables 4 & 5)
在 LLaVA-1.5 和 LLaVA-NeXT 框架下，OpenVision 3 与 OpenAI CLIP 打得有来有回。
在 LLaVA-NeXT（更强的 LLM）下，OpenVision 3 的优势更明显：
- **OpenAI-CLIP L/14**: SeedBench 61.2
- **OpenVision 3 L/2**: SeedBench 68.6

直觉上，CLIP 把图压成了 1 个 CLS token 加几个 patch token，丢了 spatial detail。而 OpenVision 3 因为有 VAE latent 兜底，保留了 dense 的 spatial structure，当背后的 LLM 足够强时，它能挖掘出这些 dense features 里的丰富信息，从而超越 CLIP。

### 6. 总结 Intuition

这篇 paper 的 intuition 就一句话：**找一个好的信息瓶颈**。

直接从 raw pixel 学 unified representation 太难，用过度压缩的 CLIP space 学又丢细节。FLUX-VAE 的 latent space 恰好是一个完美的中间态：它去掉了冗余，保留了所有生成细节，而且空间结构足够平滑。在这个温床上，用 contrastive + captioning 去提语义，用 noise + decode 去提生成，两者水到渠成，互相成就。

这给未来的 UMM 指了一个很清晰的方向：不要再搞 dual tokenizer 拼凑了，直接在顶级的 VAE latent space 上从头训练一个多任务 ViT，就能拿到极好的 unified representation。

### Reference Links
- OpenVision 3 Project Page: https://ucsc-vlaa.github.io/OpenVision3/
- Platonic Representation Hypothesis: https://arxiv.org/abs/2405.07987
- BAGEL: https://arxiv.org/abs/2505.14683
- RAE (Diffusion Transformers with Representation Autoencoders): https://arxiv.org/abs/2510.11690
- UniTok: https://arxiv.org/abs/2502.20321
- LLaVA-NeXT: https://arxiv.org/abs/2406.16860

---

# OpenVision 3: Unified Visual Encoder 深度解析

## 一、Core Problem & Motivation

当前 Unified Multimodal Models (UMMs) 面临一个 fundamental tension：**visual understanding** 与 **visual generation** 对 representation 的需求存在 conflict。

### 现有方案的两个流派

**方案A：Dual Tokenizer**（BAGEL, Uni-Fluid, MOGAO, UniWorld-V1）
- 同一张图编码两次：一次用 SigLIP/CLIP 得到 semantic tokens，一次用 VAE 得到 pixel tokens
- 问题：system complexity高，两个 feature space 难以深度融合
- BAGEL 用 FLUX-VAE + SigLIP2；UniWorld-V1 直接拼接两类特征

**方案B：Shared Discrete Tokenizer**（TokenFlow, UniTok, VILA-U, EMU3.5）
- 基于 VQ-GAN 的 discrete codebook
- 问题：quantization 引入 discretization error，限制 generation quality
- UniTok 用 multi-codebook quantization；VILA-U 对 SigLIP 特征做 residual quantization

OpenVision 3 要解决的核心问题：**如何用一个 continuous visual tokenizer 同时服务 understanding 和 generation，且 training pipeline 透明、可复现**。

这与 Platonic Representation Hypothesis [Huh et al., 2024] 的思想一致——不同模态反映 shared underlying reality，统一表征能带来 mutual benefits。

参考：
- Platonic Representation Hypothesis: https://arxiv.org/abs/2405.07987
- BAGEL: https://arxiv.org/abs/2505.14683
- UniTok: https://arxiv.org/abs/2502.20321
- TokenFlow: https://arxiv.org/abs/2501.16424

---

## 二、Architecture 深度解析

### 2.1 整体数据流

```
Image x ∈ R^(H×W×C)
    │
    ▼
[Frozen FLUX.1-dev VAE Encoder ξ_vae]  (downsample 8×)
    │
    ▼
z_vae ∈ R^(H/8 × W/8 × D_vae)
    │
    ▼
[Trainable ViT Encoder E_vit]  (patch size 2×2)
    │
    ▼
z_u ∈ R^(H/16 × W/16 × D_u)   ← Unified representation
    │
    ├─────────────────┬─────────────────┐
    ▼                 ▼                 ▼
[+ Gaussian noise]   [Text Encoder]    (frozen downstream)
    │                 │
    ▼                 ▼
[ViT Decoder + Linear]  Contrastive Loss
    │                 │
    ▼                 │
ẑ_vae                 │
    │                 │
    ▼                 │
[Frozen VAE Decoder]  Captioning Loss (Text Decoder)
    │
    ▼
x̂ (reconstructed image)
```

### 2.2 关键公式详解

**公式(1): VAE encoding**

$$z_{vae} = \xi_{vae}(x) \in \mathbb{R}^{\frac{H}{8} \times \frac{W}{8} \times D_{vae}}$$

- $x$: 输入图像，$H, W, C$ 分别为高度、宽度、通道数
- $\xi_{vae}$: FLUX.1-dev 的 VAE encoder，frozen
- $z_{vae}$: VAE latent，空间维度被压缩 8×（高度和宽度各 8×）
- $D_{vae}$: VAE latent 的通道数（FLUX VAE 通常为 16 channels）
- 这个空间是后续所有训练发生的 "playground"

**公式(2): ViT encoding**

$$z_u = \mathcal{E}_{vit}(z_{vae}) \in \mathbb{R}^{\frac{H}{16} \times \frac{W}{16} \times D_u}$$

- $\mathcal{E}_{vit}$: ViT encoder，从 scratch 训练
- patch size 为 2×2（作用于 VAE latent 上），所以再压缩 2×
- 总压缩率：$8 \times 2 = 16$×，与 CLIP B/16 的 token 数对齐
- $D_u$: ViT 的 hidden dimension（Base 约 768，Large 约 1024）
- **关键设计**：ViT 直接在 VAE latent 上做 patching，而非 raw pixels

**公式(3): Noise injection（reconstruction branch only）**

$$\tilde{z}_u = z_u + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

- $\sigma$: per-instance 的 noise scale，从 $[0, \tau]$ uniform 采样
- $\odot$: element-wise multiplication
- $\epsilon$: standard Gaussian noise，shape 与 $z_u$ 相同
- **Intuition**: 这模拟 diffusion 的 forward process。Generation 本质是从 noise 到 image，所以 decoder 需要在 noised representation 上也能重建。这让 representation 对扰动 robust，有利于下游 flow matching / diffusion generator。

**公式(4): Reconstruction loss**

$$\mathcal{L}_{rec} = \ell_1(x, \hat{x}) + \beta \ell_1(z_{vae}, \hat{z}_{vae}) + \lambda \mathcal{L}_{LPIPS}(x, \hat{x})$$

- $\ell_1$: L1 distance（mean absolute error）
- $x, \hat{x}$: 原图与重建图
- $z_{vae}, \hat{z}_{vae}$: VAE latent 与重建的 VAE latent
- $\beta$: VAE latent loss 权重（pretrain 阶段为 0.4）
- $\lambda$: LPIPS 权重（pretrain 阶段为 0，finetune 阶段为 0.5）
- $\mathcal{L}_{LPIPS}$: Learned Perceptual Image Patch Similarity，用 VGG/AlexNet 提取多层特征计算距离
- **三层 loss 的作用**：pixel L1 保 coarse 结构，latent L1 保 mid-level 信息，LPIPS 保 perceptual quality

**公式(5): Understanding loss**

$$\mathcal{L}_{und} = \mathcal{L}_{caption} + \alpha \mathcal{L}_{contrastive}(z_u, z_{txt})$$

- $\mathcal{L}_{caption}$: autoregressive cross-entropy loss，text decoder 从 $z_u$ 预测 caption tokens
- $\mathcal{L}_{contrastive}$: InfoNCE-style loss，对齐 $z_u$ 与 $z_{txt}$
- $\alpha$: contrastive 权重，设为 1.0
- $z_{txt}$: text encoder 提取的 caption feature
- 这是 CoCa [Yu et al., 2022] 的 dual-objective 设计

**公式(6): Overall loss**

$$\mathcal{L}_{overall} = \omega_{rec} \mathcal{L}_{rec} + \omega_{und} \mathcal{L}_{und}$$

- $\omega_{rec} = 0.5$, $\omega_{und} = 1.0$
- **关键 insight**：understanding 权重是 reconstruction 的两倍。作者解释：减小 reconstruction 权重能保持 generation quality 同时不损害 understanding。这有点反直觉——通常认为 generation 需要更多 pixel 监督。但实际上，semantic signal 已经隐式包含了大量 structural 信息（见 5.1 节的 synergy 分析）。

参考：
- CoCa: https://arxiv.org/abs/2205.01917
- LPIPS: https://arxiv.org/abs/1801.03924
- FLUX VAE: https://github.com/black-forest-labs/flux

---

## 三、Training Pipeline 细节

### 3.1 Progressive Training（来自 CLIPA 的 insight）

| 阶段 | Resolution | Batch Size | LR | Epochs | LPIPS λ | β | α | ω_rec | ω_und |
|------|-----------|-----------|-----|--------|---------|---|---|-------|-------|
| Pretrain | 128×128 | 8192 | 8e-6 | 4000 | 0 | 0.4 | 1.0 | 0.5 | 1.0 |
| Finetune | 224/256 | 4096 | 4e-7 | 400 | 0.5 | 0.4 | 1.0 | 0.5 | 1.0 |

**Intuition**：
- CLIPA 发现 CLIP 训练存在 inverse scaling law：低分辨率阶段投入更多计算更高效
- 10:1 的 epoch 比例意味着 90% 的计算在 128×128 上完成
- 这是因为 contrastive learning 在低分辨率已能学到大部分 semantic 抽象
- Finetune 阶段开启 LPIPS：避免 pretrain 阶段不同 resolution 的 LPIPS 冲突

参考 CLIPA: https://arxiv.org/abs/2310.16772

### 3.2 组件冻结策略

| 组件 | 状态 | 说明 |
|------|------|------|
| FLUX.1-dev VAE | Frozen | 提供高质量 latent space |
| ViT Encoder | Trainable (scratch) | 核心 unified representation learner |
| ViT Decoder | Trainable | reconstruction branch |
| Text Encoder | Trainable | contrastive learning |
| Text Decoder | Trainable | captioning |
| Linear layer | Trainable | latent 投影 |

注意：TUNA（concurrent work）用 pretrained ViT，而 OpenVision 3 从 scratch 训练，这是关键区别。

### 3.3 数据

- DataComp dataset，用 LLaVA-Llama-3 recaption
- 1 epoch ≈ 1.3M samples
- Recaptioned captions 质量高，提升 multimodal learning 效率

参考：
- DataComp: https://github.com/mlfoundations/datacomp
- LLaVA-NeXT recaption: https://arxiv.org/abs/2406.08478

---

## 四、Experimental Results 深度分析

### 4.1 Reconstruction Performance (Table 2)

| Model | Type | ImageNet rFID↓ | COCO rFID↓ |
|-------|------|----------------|------------|
| SD-VAE | Gen-only | 0.606 | 4.142 |
| SD3-VAE | Gen-only | 0.201 | 1.671 |
| FLUX-VAE | Gen-only | 0.176 | 1.343 |
| RAE (CLIP) | Unified | 1.06 | 10.119 |
| UniTok | Unified | 0.362 | 3.918 |
| OmniTokenizer | Unified | 1.411 | 6.292 |
| VILA-U | Unified | 4.231 | 10.997 |
| **OpenVision 3** | **Unified** | **0.187** | **1.601** |

**关键观察**：
1. OpenVision 3 的 rFID 0.187 接近 FLUX-VAE 的 0.176（专门的 generation tokenizer）
2. 相比最强 unified competitor UniTok，提升约 2×（0.187 vs 0.362）
3. RAE (CLIP) 表现最差——纯 semantic encoder 丢失太多 pixel 信息
4. 这证明了 VAE+ViT hybrid 设计能避免 semantic compression 的信息损失

### 4.2 Generation Performance (Table 3)

| Tokenizer | Generator | gFID↓ | IS↑ | Pre.↑ | Rec.↑ |
|-----------|-----------|-------|-----|-------|-------|
| SD-VAE | DiT | 2.27 | 278.2 | 0.83 | 0.57 |
| SD-VAE | SiT | 2.06 | 270.3 | 0.82 | 0.59 |
| UniTok | LlamaGen | 2.51 | 216.7 | 0.82 | 0.57 |
| CLIP | RAE | 2.54 | 256.4 | 0.80 | 0.54 |
| OpenVision | RAE | 2.44 | 262.2 | 0.80 | 0.53 |
| **OpenVision 3** | **RAE** | **1.87** | **290.0** | **0.84** | **0.59** |

**关键观察**：
1. OpenVision 3 (1.87) 甚至超过 SD-VAE+SiT (2.06)，尽管后者是专门的 generation pipeline
2. 相比 CLIP+RAE (2.54)，提升 0.67 gFID——这是 semantic encoder 做 generation 的 inherent limitation
3. IS 290.0 是所有方案中最高的，说明生成质量与多样性都好
4. **直觉解释**：OpenVision 3 的 representation 既保留了 semantic structure（从 contrastive+captioning 学到），又保留了 pixel-level fidelity（从 VAE latent + reconstruction 学到）。Flow matching generator 在这样的 representation 空间上更容易学到 meaningful manifold。

参考 RAE: https://arxiv.org/abs/2510.11690

### 4.3 Understanding Performance

**LLaVA-1.5 Framework (Table 4):**

| Vision Encoder | Size | SeedBench | ScienceQA | GQA | POPE |
|----------------|------|-----------|-----------|-----|------|
| OpenAI-CLIP | B/16 | 62.2 | 73.7 | 58.6 | 82.9 |
| OpenVision 3 | VAE+B/2 | 63.1 | 73.1 | 58.9 | 83.7 |
| OpenAI-CLIP | L/14 | 65.4 | 73.9 | 60.6 | 84.7 |
| OpenVision 3 | VAE+L/2 | 65.8 | 72.4 | 61.0 | 85.2 |

**LLaVA-NeXT Framework (Table 5):**

| Vision Encoder | Size | SeedBench | ScienceQA | GQA | POPE |
|----------------|------|-----------|-----------|-----|------|
| OpenAI-CLIP | B/16 | 61.2 | 72.8 | 58.1 | 84.7 |
| OpenVision 3 | VAE+B/2 | 63.3 | 68.9 | 59.2 | 84.9 |
| OpenAI-CLIP | L/14 | 61.8 | 75.3 | 59.4 | 84.1 |
| OpenVision 3 | VAE+L/2 | 68.6 | 73.6 | 62.0 | 86.6 |

**关键观察**：
1. 在 LLaVA-NeXT (更强 framework) 下，OpenVision 3 的优势更明显
   - SeedBench: 63.3 vs 61.2 (Base), 68.6 vs 61.8 (Large)
   - Large 模型 SeedBench 提升 6.8 分，非常显著
2. ScienceQA 偶尔落后，因为这是纯 text-heavy task，VAE 的 pixel 信息帮助有限
3. **直觉解释**：更强的 LLM backbone 能更好利用 OpenVision 3 representation 中的 dense visual信息。CLIP 的 representation 过度压缩（pooling 到 single CLS token），丢失了 spatial 细节；OpenVision 3 保留了 patch-wise dense representation。

参考：
- LLaVA: https://github.com/haotian-liu/LLaVA
- SeedBench: https://arxiv.org/abs/2307.16125
- GQA: https://arxiv.org/abs/1902.09506

---

## 五、Ablation Studies 的深层 insight

### 5.1 Reciprocal Synergy（最有趣的发现）

**实验设计**：分别去掉 reconstruction loss 和 understanding loss，观察另一分支的 loss 曲线。

**Figure 4：去掉 reconstruction loss，只用 understanding loss**
- Pixel recon loss 和 latent recon loss 仍然显著下降
- Caption loss 和 contrastive loss 无明显变化

**Figure 5：去掉 understanding loss，只用 reconstruction loss**
- Pixel recon loss 反而更高（加上 semantic loss 后降低）
- Caption loss 略有下降
- Contrastive loss 几乎停滞

**Intuition 解释**：

为什么 semantic loss 能帮助 reconstruction？
- Semantic supervision 迫使 representation 编码 high-level structure（物体类别、场景语义）
- 这个 structure 作为 prior，约束了 reconstruction 的 search space
- 类似于 image completion 中 semantic context 帮助填补细节
- 这与 REPA [Yu et al., 2024] 的发现一致：在 diffusion 训练中 align 中间特征到 DINO/CLIP 能加速收敛

为什么 reconstruction loss 能帮助 semantic（captioning）？
- Captioning 是 generative task（autoregressive generation）
- Reconstruction 也是 generative task
- 两者共享 "如何从 representation 生成" 的能力
- Contrastive 是判别任务，所以 reconstruction 帮助不大

**更深层的 hypothesis**：
- Image understanding 和 image generation 在 representation 层面 share computational primitives
- 这支持了 Platonic Representation Hypothesis：存在 universal representation 同时服务两者
- 传统的 "semantic vs pixel" 二分法可能是 false dichotomy

参考 REPA: https://arxiv.org/abs/2410.06940

### 5.2 Necessity of VAE (Tables 6, 7, 8)

**Reconstruction (Table 6):**

| Model | PSNR↑ | SSIM↑ | LPIPS↓ | rFID↓ |
|-------|-------|-------|--------|-------|
| w/o VAE | 32.82 | 0.935 | 0.060 | 0.980 |
| OpenVision 3 | 30.33 | 0.885 | 0.061 | 0.216 |

注意：w/o VAE 的 PSNR/SSIM 更高，但 rFID 大幅落后（0.980 vs 0.216）。PSNR/SSIM 是 pixel-level metric，rFID 是 distributional metric。这说明：
- w/o VAE 能做 point-wise 重建（每个 pixel 平均误差小）
- 但 w/ VAE 能保持更好的 perceptual distribution（生成的 image manifold 更真实）
- VAE latent space 本身就是 perceptually aligned 的，在这个空间学习 representation 天然带 perceptual prior

**Generation (Table 7):**

| Model | gFID↓ |
|-------|-------|
| w/o VAE | 9.68 |
| OpenVision 3 | 8.45 |

Generation 提升 1.23 gFID，符合预期——VAE latent space 是 flow matching generator 的天然 working space。

**Understanding (Table 8):**

在 LLaVA-NeXT 下，OpenVision 3 在所有 6 个 metrics 上领先 w/o VAE 版本。这有点 surprising：VAE 本是为 pixel 设计的，为什么能帮助 understanding？

**可能的解释**：
1. VAE latent 是 dense spatial representation，保留了 CLIP pooling 丢失的 spatial 信息
2. VAE 经过大规模 image generation 训练，latent space 已 encode 丰富的 visual priors
3. 在 VAE latent 上训练 ViT，相当于站在巨人肩膀上学习 semantic

### 5.3 Decoder Size (Figure 6)

| Decoder | Pixel Loss | Stability | Training Time |
|---------|------------|-----------|---------------|
| M/1 | Higher | Stable | 6.1h |
| B/1 | Lower | Stable | 6.0h |
| L/1 | Lowest(?) | Unstable | 6.9h |

**关键发现**：
- M→B 显著降低 pixel loss
- B→L 训练不稳定
- Understanding loss 对 decoder size 几乎不敏感

**直觉解释**：
- Reconstruction 是更难的 pixel-level task，需要足够 decoder capacity
- 但 L decoder 不稳定可能因为：parameter 太多 + reconstruction loss landscape 复杂 + 与 encoder 的 capacity imbalance
- Understanding 只依赖 encoder output，decoder 是 auxiliary，所以不敏感

### 5.4 Encoder Size (Table 9)

| Model | Reconstruction rFID | Generation gFID |
|-------|---------------------|-----------------|
| OpenVision 3-B | 0.187 | 8.45 |
| OpenVision 3-L | 0.186 | 8.89 |

**关键发现**：
- B 和 L 在 reconstruction/generation 上几乎相同
- 但 L 在 understanding 上显著更好（Table 4, 5）
- 作者解释：outer VAE architecture 施加了 theoretical ceiling

**更深层 intuition**：
- VAE 的 reconstruction capacity 是 fixed bottleneck
- 一旦 encoder 能 fully utilize VAE latent space，再大也没用
- 这与 "VAE 是 information bottleneck" 的观点一致
- 但 understanding 不受 VAE bottleneck 限制，因为它直接用 encoder output，所以能从更大 encoder 受益
- 这暗示：unified tokenizer 的 scaling law 在 generation 和 understanding 上不同

---

## 六、与 Concurrent / Related Works 的对比

### 6.1 vs TUNA [Liu et al., 2025]

| 维度 | TUNA | OpenVision 3 |
|------|------|--------------|
| VAE | ✓ | ✓ (FLUX.1) |
| ViT | Pretrained | From scratch |
| Training paradigm | Non-transparent | Transparent, progressive |
| Loss design | 未详述 | Detailed (rec + und with sub-components) |

OpenVision 3 的 contribution 是 systematize 这个设计并给出 transparent training recipe。

参考 TUNA: https://arxiv.org/abs/2512.02014

### 6.2 vs UniLIP / Show-o2 / TokenFlow / UniTok

| 方法 | Representation | Semantic Source | Pixel Source |
|------|----------------|-----------------|--------------|
| UniLIP | Continuous | CLIP + self-distill | Implicit |
| Show-o2 | Continuous | Semantic projection | VAE projection |
| TokenFlow | Discrete (VQ) | Dual codebook | Shared mapping |
| UniTok | Discrete (multi-VQ) | Multi-codebook | Multi-codebook |
| **OpenVision 3** | **Continuous** | **Contrastive+Captioning on z_u** | **VAE latent reconstruction** |

OpenVision 3 的独特之处：
1. 用 VAE latent 作为统一 working space（而非 raw pixels）
2. Continuous representation（避免 VQ error）
3. Single encoder, single representation（真正 unified）
4. Two separate decoder branches（避免 mutual interference）

### 6.3 vs REPA-E [Leng et al., 2025]

REPA-E: Unlocking VAE for end-to-end tuning with latent diffusion transformers
- 也探索 "语义监督 + VAE" 的组合
- 但 REPA-E 关注 diffusion training，OpenVision 3 关注 unified tokenizer
- 思路相似：semantic supervision 能 improve pixel-level tasks

参考 REPA-E: https://arxiv.org/abs/2504.10483

---

## 七、Critical Analysis & Open Questions

### 7.1 Limitations 我看到的

1. **VAE 的 theoretical ceiling**
   - 论文承认 VAE bottleneck 限制 generation scaling
   - 未来可能需要 trainable VAE 或更好的 latent space
   - 但 trainable VAE 会破坏 frozen VAE 带来的 training stability

2. **Token 数量限制**
   - 196 tokens (Base) / 256 tokens (Large)
   - 比 SigLIP2 的 dynamic resolution 灵活性差
   - 高分辨率细节（OCR、小物体）可能受限

3. **Understanding 略弱于 CLIP 在某些 benchmark**
   - ScienceQA 在 LLaVA-NeXT 下：68.9 vs 72.8 (Base)
   - 纯 text reasoning task 上 VAE 信息帮助有限

4. **Generation evaluation 仅在 ImageNet class-conditional**
   - 没有 text-to-image 评估
   - RAE 框架下评估，与 SOTA diffusion model (FLUX, SD3) 未直接对比

5. **Loss 权重 ω_und = 2 × ω_rec 的选择**
   - 论文给的解释比较 hand-wavy
   - 没有完整 ablation sweep

### 7.2 Future Directions

1. **Trainable VAE with regularization**
   - 类似 REPA-E 的 end-to-end VAE tuning
   - 可能突破 theoretical ceiling

2. **Multi-resolution tokenization**
   - 借鉴 SigLIP2 / RADIO 的 multi-resolution 设计
   - 在不同 scale 上学习 unified representation

3. **加入 video / 3D**
   - Wan2.1 VAE 已支持 video
   - 扩展到 spatiotemporal unified tokenizer

4. **更细粒度 understanding**
   - 当前 token 数限制 spatial resolution
   - 可能需要 hierarchical / pyramidal design

5. **与 LLM 的 deeper fusion**
   - 当前是 plug-in vision encoder
   - 未来可能做 end-to-end UMM training

---

## 八、对 Field 的影响

### 8.1 实践层面

1. **开源 unified tokenizer**
   - 代码、数据、checkpoints 全开
   - 降低 UMM 研究 barrier
   - 可能成为新 baseline

2. **训练 recipe 透明**
   - Progressive training, loss weights, hyperparameters 都详尽
   - 可复现性强

3. **VAE + ViT 成为新 paradigm**
   - TUNA + OpenVision 3 concurrent work 验证了这一思路
   - 可能替代 dual tokenizer 方案

### 8.2 理论层面

1. **Platonic Representation Hypothesis 的实证**
   - 单一 representation 服务 understanding + generation
   - 两个 task 互相 benefit
   - 支持 "universal representation" 假说

2. **Platonic Representation Hypothesis**
   - Ilya Sutskever 等人提到的 idea
   - 不同 task / modality 在 deep model 中 converge 到 similar representation
   - OpenVision 3 提供了视觉领域的 evidence

3. **Semantic supervision 的 pixel-level benefit**
   - 挑战 "semantic vs pixel" 二分法
   - 与 REPA 系列工作共同形成 evidence chain

---

## 九、我的整体 Intuition 总结

OpenVision 3 的核心 insight 可以浓缩为一句话：**好的 visual representation 应该在 "信息丰富" 的空间学习，而非 "信息压缩" 的空间**。

传统 CLIP 把图像压缩到 single CLS token，丢失 spatial 信息；传统 VAE 保留 pixel 信息但缺乏 semantic。OpenVision 3 选择在 VAE latent（既 dense 又 perceptually aligned）上学习 semantic，相当于在"信息瓶颈适当"的空间做 multi-task learning。

这背后的哲学：**理解与生成本是同一硬币的两面**。理解是 image→concept，生成是 concept→image，两者共享 visual world model。把它们强行拆开（dual tokenizer）会丢失 synergy；强行统一到离散空间（VQ）会引入 artifact。Continuous representation in VAE latent + dual decoder branches 是目前的 sweet spot。

类似思路在 LLM 领域也有：next token prediction 同时学语法和语义，无需分开。OpenVision 3 在视觉领域验证了类似的 unified learning principle。

---

## 十、Reference Links

### 核心论文
- OpenVision 3 Project Page: https://ucsc-vlaa.github.io/OpenVision3/
- OpenVision (v1): https://arxiv.org/abs/2505.04639
- OpenVision 2: https://arxiv.org/abs/2509.01644
- TUNA: https://arxiv.org/abs/2512.02014
- BAGEL: https://arxiv.org/abs/2505.14683
- TokenFlow: https://arxiv.org/abs/2501.16424
- UniTok: https://arxiv.org/abs/2502.20321
- VILA-U: https://arxiv.org/abs/2409.04429
- Show-o 2: https://arxiv.org/abs/2506.15564
- UniWorld-V1: https://arxiv.org/abs/2506.03147
- EMU3.5: https://arxiv.org/abs/2510.26583

### 理论基础
- Platonic Representation Hypothesis: https://arxiv.org/abs/2405.07987
- REPA: https://arxiv.org/abs/2410.06940
- REPA-E: https://arxiv.org/abs/2504.10483
- CoCa: https://arxiv.org/abs/2205.01917
- CLIP: https://arxiv.org/abs/2103.00020
- SigLIP 2: https://arxiv.org/abs/2502.14786
- CLIPA v2: https://arxiv.org/abs/2306.15658
- RAE: https://arxiv.org/abs/2510.11690

### 组件
- FLUX.1: https://github.com/black-forest-labs/flux
- LLaVA: https://github.com/haotian-liu/LLaVA
- DataComp: https://github.com/mlfoundations/datacomp
- DiT: https://github.com/facebookresearch/DiT
- AM-RADIO: https://arxiv.org/abs/2312.06709

### Benchmark
- SeedBench: https://arxiv.org/abs/2307.16125
- GQA: https://arxiv.org/abs/1902.09506
- MME: https://arxiv.org/abs/2306.13394
- ScienceQA: https://arxiv.org/abs/2209.10658
- POPE: https://arxiv.org/abs/2305.10355
- ImageNet: https://www.image-net.org/
- COCO: https://cocodataset.org/

希望这个详细解析能 build up 你的 intuition about unified visual tokenization！这个方向我认为是 multimodal learning 未来 1-2 年的核心 battlefield。
