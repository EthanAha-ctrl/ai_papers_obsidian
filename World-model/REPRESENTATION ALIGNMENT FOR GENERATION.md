---
source_pdf: REPRESENTATION ALIGNMENT FOR GENERATION.pdf
paper_sha256: cc5be5a56cc9bbc9cdf3f1e4299af894f3e1548e36838863b405e98cd4382806
processed_at: '2026-08-11T22:47:56-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好的，Andrej，我们抛开那些复杂的公式，用最直白的直觉来聊聊这篇 paper 到底在干嘛。

这篇 paper 的核心可以用一句话总结：**与其让 diffusion model 从零开始瞎摸索怎么“理解”图片，不如直接拿现成的学霸（DINOv2）的笔记给它抄。**

### 1. 痛点在哪？

现在最火的生成模型架构是 Diffusion Transformer (DiT / SiT)。它的任务是拿一张全是噪点的图，一步步去噪，最后还原出清晰的图。

在这个去噪过程中，model 的中间层会形成一个 hidden state（我们叫它 $\mathbf{h}_t$）。这个 $\mathbf{h}_t$ 本质上就是 model 对当前图片的“理解”。

问题在于：让 model 通过“预测噪点”或者“预测 velocity”这种纯粹的 pixel-reconstruction 任务去自发学出一个高质量的 semantic representation，实在太慢了，而且天花板很低。作者拉出已经训了 7M steps 的 SiT-XL 一看，发现它学到的 representation 跟顶级的 SSL encoder（DINOv2）比起来，依然是个“学渣”。两者在对齐度（CKNNA）和 linear probing 上差了一大截。

直觉上很好理解：如果只是逼着模型还原像素，它的 attention 就会被高频的纹理细节填满，根本没空去提取“哦，这是一只狗”这种高级语义。

### 2. REPA 的“抄作业”方案

既然 diffusion model 自己学不好 representation，而生成图片又极度依赖好的 representation 作为底座，那怎么办？

REPA (REPresentation Alignment) 的做法极其粗暴有效：**直接加一个 auxiliary loss，强迫 diffusion model 内部的 representation 去对齐 DINOv2 的 representation。**

具体怎么做的？流程非常清晰：
1. 拿一张干净的图片 $\mathbf{x}_*$，喂给 DINOv2，得到一组极度优秀的 semantic features：$\mathbf{y}_*$。
2. 把同一张图片加上噪点，变成 $\mathbf{z}_t$，喂给正在训练的 diffusion transformer。
3. 在 transformer 内部比如第 8 层，把此时的 hidden state $\mathbf{h}_t$ 抠出来。
4. 接一个小的 MLP projection head，把这个 $\mathbf{h}_t$ 转换到跟 DINOv2 一样的维度空间。
5. 算一个简单的 cosine similarity loss，强迫这个 projected hidden state 去逼近 DINOv2 输出的 $\mathbf{y}_*$。

最终训练的 loss 就是原来的 diffusion loss 加上这个 alignment loss：
$\mathcal{L} = \mathcal{L}_{\mathrm{velocity}} + \lambda \mathcal{L}_{\mathrm{REPA}}$

### 3. 这里有两个极度巧妙的直觉

这也是我觉得这篇 paper 最让人拍大腿的地方：

**第一，为什么 target 是 clean image 的 feature，却拿去跟 noisy input 的 feature 对齐？**
DINOv2 是吃干净图片的，你没法把一张充满噪点的图直接喂给它。REPA 的做法是：不管输入多噪，我都强迫 diffusion model 在内部“脑补”出那张干净图片的 semantic feature。
这等于是在告诉 model：你别管外面那些噪点，你在第 8 层的时候，必须在心里明白这画的是啥。这就把“去噪”这个任务在 semantic 层面给 explicit 化了。

**第二，为什么只在早期层（比如 layer 8 / 28）对齐？**
作者发现，如果在很深的层做对齐，FID 反而变差了，但 linear probing accuracy 会变高。
直觉上，一个生成模型需要分工：早期层负责搞懂“这是什么”（semantic），后期层负责搞定“纹理和高频细节”。如果你强迫后期层也去 align DINOv2 的 semantic space，就等于逼着画细节的笔去写哲学论文，把 capacity 浪费了，生成质量自然下降。只 align 前 8 层，等于给 model 打好坚实的语义地基，后面的层就可以毫无顾忌地去雕花。

### 4. 效果有多炸裂？

因为有了 DINOv2 这个外部“外挂”提供强大的 representation inductive bias，diffusion model 训练得飞快。

在 ImageNet 256×256 上：
原本的 SiT-XL/2 要训练 7M iterations 才能达到 FID = 8.3。
加了 REPA 之后，只训了 400K iterations，FID 就达到了 7.9！
直接 **17.5× 的加速**，而且最终质量还更好。配合上 classifier-free guidance 的一些 trick，直接刷到了 SOTA (FID = 1.42)。

### 5. 为什么这件事本质上很深？

这其实触及了目前 AI 圈一个很底层的哲学讨论：**判别模型和生成模型的 representation 是不是殊途同归的？**

Yann LeCun 一直强调，生成模型做 pixel-level reconstruction 是学不到好 representation 的，应该去做 JEPA 那种 latent space 的 prediction。而 REPA 等于是在说：diffusion model 确实学不好 representation，但我们不用推翻它，只要把它中间的 representation 硬拉到 DINOv2 那个 Platonic representation space 里去就行了。

更强 encoder 训出来的 REPA，生成的图片就更好（Table 2 里的 scaling 关系极其漂亮）。这意味着：**生成的质量瓶颈，其实就是理解能力的瓶颈。**只要把理解能力拉满，生成自然就通了。

这跟大语言模型 (LLM) 的直觉一模一样：LLM 通过 next-token prediction 自然就能学到极好的 representation，因为 token 本身就是高度浓缩的 semantic unit。而图像的 pixel 太过底层，diffusion 等于是在底层泥潭里挣扎，REPA 就是强行把它拽到 semantic 的层面上来。

### 6. 可能的联想与延伸

顺着这个直觉，我们能开很多脑洞：

- **视频生成（Sora-like models）：** Sora 用的也是 DiT 架构。如果我们在训视频 DiT 的时候，引入一个强大的 video SSL encoder（比如 VideoMAE-v2 或者未来的某个 DINOv2-Video），强迫视频 diffusion model 的早期层去 align 视频的时空 semantic feature，大概率也能获得惊人的加速。
- **多模态注入：** 既然 DINOv2 work，那 CLIP/SigLIP 的 text feature 也能作为 target。事实上作者在 Appendix 里试了 T2I，效果也很好。这说明 model 其实不在乎 align 的是什么，只要是一个高度压缩、信息稠密的 representation 就行。
- **跟 LLM distillation 的相似性：** REPA 本质上就是 feature-level distillation。类似于把 GPT-4 的中间层 hidden states 蒸馏给小模型。这里的绝妙之处在于，teacher 是判别模型，student 是生成模型，跨越了任务边界。

总结来说，这篇 paper 简单得让人觉得“这也能发顶会？”，但它的效果又好得让人没法忽略。它用最简单的方式证明了一件事：**生成模型之所以难训，往往只是因为它还不懂自己在生成什么。只要帮它跨过“理解”这道坎，剩下的它自己就能搞定。**

Reference:
- REPA 原文: https://arxiv.org/abs/2410.06940
- SiT 原文: https://arxiv.org/abs/2401.08715
- DINOv2 原文: https://arxiv.org/abs/2304.07193
- Platonic Representation Hypothesis: https://arxiv.org/abs/2405.10318

---

# REPA: 让 Diffusion Transformer 训练变简单的 Representation Alignment

## 1. 论文的 core insight

这篇 paper 的标题就在挑衅："Training Diffusion Transformers Is Easier Than You Think"。作者们 (Sihyun Yu, Jinwoo Shin, Saining Xie 等) 想说的核心是：**diffusion model 训练困难的真正 bottleneck 不是生成任务本身,而是模型还要同时学一个好的 internal representation**。如果把这个 representation learning 的 burden 分担给一个已经训好的外部 SSL encoder (如 DINOv2),训练效率和质量都会大幅提升。

这个想法本身在 Karpathy 的直觉里非常自然——它呼应了几个我长期关注的现象：

- **Platonic Representation Hypothesis** (Huh et al., 2024, https://arxiv.org/abs/2405.10318): 不同的 model、不同 modality、不同 objective,在 scale 上去之后都会收敛到相似的 representation space。如果 convergence 是必然的,为什么不直接 shortcut?
- **LeCun 一直强调的**: pixel-level reconstruction 不是好的 SSL objective,因为它没法丢掉不必要的 detail (JEPA 的 motivation, https://arxiv.org/abs/2301.08243)。
- **MAE vs DINO 的差距**: 都是 ViT-based SSL,MAE 做 reconstruction、DINO 做 self-distillation,但 DINOv2 的 features 显著更强。这暗示 reconstruction objective 在 representation learning 上是次优的。

REPA 的逻辑就是: diffusion model 本质上是 denoising autoencoder (Vincent, 2011, https://www.mitpressjournals.org/doi/10.1162/NECO_a_00142),它的 representation h 是 implicit 学出来的,但这个 representation 学习过程是被 reconstruction 间接驱动的,所以学不好。Solution? 直接 inject 高质量 external representation。

---

## 2. 三个关键的 empirical observations (Section 3.2)

作者用 SiT-XL/2 (训了 7M iterations 的 checkpoint) 和 DINOv2-g 做对比,得到三个观察。这是整个工作的 motivation 基础。

### Observation 1: Semantic gap 存在 (Figure 2a)

Linear probing on ImageNet,用 diffusion transformer 的 layer-wise hidden states (globally pooled):

- SiT-XL/2 在 layer 20 左右达到 peak (大约 50-55% top-1)
- DINOv2-g 大约 85%+
- **gap 显著**: diffusion model 学到了 discriminative features (跟 Xiang et al. 2023 的发现一致,https://arxiv.org/abs/2303.00548),但远不如专门的 SSL encoder。
- Peak 之后 linear probing 快速下降——后续 layers 从 semantic 转向 high-frequency details,为了生成。

### Observation 2: 弱 alignment 已存在 (Figure 2b)

用 CKNNA (Centered Kernel Nearest-Neighbor Alignment, Huh et al., 2024) 测量 SiT 和 DINOv2 之间的 representation alignment:

**CKNNA 公式回顾** (Appendix C.1):

$$
\mathrm{CKA}(\mathbf{K}, \mathbf{L}) = \frac{\mathrm{HSIC}(\mathbf{K}, \mathbf{L})}{\sqrt{\mathrm{HSIC}(\mathbf{K}, \mathbf{K}) \mathrm{HSIC}(\mathbf{L}, \mathbf{L})}}
$$

其中 $\mathbf{K}_{ij} = \kappa(\phi_i, \phi_j)$, $\mathbf{L}_{ij} = \kappa(\psi_i, \psi_j)$ 是两个网络在数据集上的 kernel matrices (内积 kernel),HSIC 是 Hilbert-Schmidt Independence Criterion:

$$
\mathrm{HSIC}(\mathbf{K}, \mathbf{L}) = \frac{1}{(n-1)^2} \sum_{i,j} (\langle \phi_i, \phi_j \rangle - \mathbb{E}_l[\langle \phi_i, \phi_l \rangle])(\langle \psi_i, \psi_j \rangle - \mathbb{E}_l[\langle \psi_i, \psi_l \rangle])
$$

CKNNA 是 CKA 的 relaxed 版,只在 k-nearest neighbors 上计算 (k=10):

$$
\alpha(i, j; k) = \mathbb{1}[i \neq j \text{ and } \phi_j \in \mathrm{knn}(\phi_i; k) \text{ and } \psi_j \in \mathrm{knn}(\psi_i; k)]
$$

- SiT 和 DINOv2 的 CKNNA 比 MAE 和 DINOv2 高 (说明 SiT 的 representation 已经有点接近 DINOv2 的 " Platonic " 空间)
- 但绝对值低,远不如 MoCov3 vs DINOv2 之间的 alignment。

### Observation 3: Alignment 随 scale 改善但慢 (Figure 2c)

更大数据 + 更长训练 → CKNNA 更高,但即使 7M iterations 仍然低。这意味着 alignment 是 "natural" 趋势,但 diffusion objective 本身驱动力很弱,需要外力 push。

**Intuition**: 这三个观察连起来就是 paper 的 thesis: diffusion model 在往 DINOv2 那个 representation space 靠,但太慢。我们直接给它一个 alignment loss 当 shortcut。

---

## 3. REPA 方法的数学形式 (Section 3.3)

### 背景公式: flow matching / stochastic interpolants

Eq (1) 的前向过程:

$$
\mathbf{x}_t = \alpha_t \mathbf{x}_* + \sigma_t \epsilon, \quad \alpha_0 = \sigma_T = 1, \alpha_T = \sigma_0 = 0
$$

- $\mathbf{x}_* \sim p(\mathbf{x})$: clean data (这里实际上是 latent $\mathbf{z} = E(\mathbf{x})$, E 是 Stable Diffusion VAE encoder)
- $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$: Gaussian noise
- $\alpha_t, \sigma_t$: schedule,论文用 **linear interpolant**,即 $\alpha_t = 1-t, \sigma_t = t$, $T = 1$
- $t \in [0, T]$: time index

Eq (3): velocity field 是两个条件期望的加权和:

$$
\mathbf{v}(\mathbf{x}, t) = \dot{\alpha}_t \mathbb{E}[\mathbf{x}_* | \mathbf{x}_t = \mathbf{x}] + \dot{\sigma}_t \mathbb{E}[\epsilon | \mathbf{x}_t = \mathbf{x}]
$$

- $\dot{\alpha}_t = d\alpha_t/dt$: 对 linear interpolant 是 -1
- $\dot{\sigma}_t$: 对 linear interpolant 是 1
- $\mathbb{E}[\mathbf{x}_* | \mathbf{x}_t = \mathbf{x}]$: "去噪" 的方向 (恢复 clean data)
- $\mathbb{E}[\epsilon | \mathbf{x}_t = \mathbf{x}]$: "加噪" 的方向 (predict noise)

Eq (4) velocity training objective:

$$
\mathcal{L}_{\mathrm{velocity}}(\theta) := \mathbb{E}_{\mathbf{x}_*, \epsilon, t} \big[ ||\mathbf{v}_\theta(\mathbf{x}_t, t) - \dot{\alpha}_t \mathbf{x}_* - \dot{\sigma}_t \epsilon||^2 \big]
$$

注意 target 是 $\dot{\alpha}_t \mathbf{x}_* + \dot{\sigma}_t \epsilon$,对 linear interpolant 就是 $-\mathbf{x}_* + \epsilon$。这跟 $\epsilon$-prediction 和 v-prediction 都可以互相转换 (https://arxiv.org/abs/2202.00512)。

### REPA 的核心 loss

设 $f$ 是 pretrained encoder (如 DINOv2),clean image $\mathbf{x}_*$,encoder 输出 $\mathbf{y}_* = f(\mathbf{x}_*) \in \mathbb{R}^{N \times D}$:
- $N$: patch 数 (在 SiT-XL/2 中,latent 32×32,patch size 2,所以 N=16×16=256)
- $D$: embedding dimension of $f$ (DINOv2-B 是 768,L 是 1024,g 是 1536)

Diffusion transformer 把 noisy latent $\mathbf{z}_t$ 编码为 hidden state $\mathbf{h}_t = f_\theta(\mathbf{z}_t)$。这里 $f_\theta$ 是 diffusion transformer 的 "encoder" 部分 (隐式的)。

REPA 加一个 trainable projection head $h_\phi$ (3 层 MLP with SiLU),把 $\mathbf{h}_t$ 投到 $\mathbb{R}^{N \times D}$,然后 patch-wise 对齐:

$$
\mathcal{L}_{\mathrm{REPA}}(\theta, \phi) := -\mathbb{E}_{\mathbf{x}_*, \epsilon, t} \Big[ \frac{1}{N} \sum_{n=1}^{N} \mathrm{sim}(\mathbf{y}_*^{[n]}, h_\phi(\mathbf{h}_t^{[n]})) \Big]
$$

- $n \in \{1, \ldots, N\}$: patch index
- $\mathbf{y}_*^{[n]}$: 第 $n$ 个 patch 的 DINOv2 输出
- $\mathbf{h}_t^{[n]}$: 第 $n$ 个 patch 的 diffusion transformer hidden state (在某个 layer 的输出)
- $\mathrm{sim}$: similarity function。论文比较了:
  - **Cosine similarity** (negative): $\mathrm{sim}(\mathbf{a}, \mathbf{b}) = \cos(\mathbf{a}, \mathbf{b})$
  - **NT-Xent** (SimCLR-style, https://arxiv.org/abs/2002.05709): contrastive loss with temperature

总 loss:

$$
\mathcal{L} := \mathcal{L}_{\mathrm{velocity}} + \lambda \mathcal{L}_{\mathrm{REPA}}
$$

- $\lambda = 0.5$ 是默认值 (Table 5 ablation: 0.25 → 8.6 FID, 0.5 → 7.9, 0.75 → 7.8, 1.0 → 7.8,基本饱和)

### 一个微妙的点: target 是 clean image 的 DINOv2 features

注意 $\mathbf{y}_*$ 是从 **clean image** $\mathbf{x}_*$ 计算的,而 $\mathbf{h}_t$ 是从 **noisy latent** $\mathbf{z}_t$ 计算的。这意味着 diffusion transformer 要学的不是"对应这个 noisy 输入的 DINOv2 features",而是"这个 noisy 输入对应的 clean image 的 DINOv2 features"。

这跟 denoising autoencoder 的思想一致,但 target 不是 pixel 而是 semantic feature。这个设计避免了 SSL encoder 必须处理 noisy input 的问题——DINOv2 只需要看 clean image。

**Intuition**: 我觉得这是 REPA 最聪明的设计选择。如果你想让 DINOv2 直接吃 noisy input,得 fine-tune 它,可能破坏 representation 质量。这里把 noise 的 "denoising" 工作完全留给 diffusion transformer,而 DINOv2 只提供 "what semantic content is here" 的 supervision。

### 另一个微妙的点: alignment depth = 8

在 SiT-XL/2 (28 layers) 中,只在 layer 8 的 hidden states 上加 REPA loss。这看似反直觉——一般 ViT 的 deeper layers 才是 semantic 的。但作者发现:

| Depth | FID ↓ | Acc ↑ |
|-------|-------|-------|
| 6     | 10.3  | 66.2  |
| **8** | **10.0** | 68.1 |
| 10    | 10.5  | 68.6  |
| 12    | 11.2  | 69.4  |
| 14    | 11.6  | 70.0  |
| 16    | 12.1  | 71.1  |

注意 trade-off: 更深的 layer alignment 给出 **更高** 的 linear probing accuracy (semantic更强),但 **更差** 的 FID (生成质量下降)。

**Intuition**: 我理解这是 division of labor。早期 layers 学 semantic representation,late layers 学 high-frequency details 用于生成。如果强迫 late layers 也 align 到 DINOv2 (semantic),就压抑了它们做 reconstruction 的工作。这也呼应 Observation 1 里 linear probing 在 layer 20 之后快速下降——diffusion transformer 的后期 layers 本来就不该是 semantic 的。

### 架构 (Appendix B, Figure 9)

DiT block:
- Input: noisy latent $\mathbf{z}_t \in \mathbb{R}^{32 \times 32 \times 4}$ (SD VAE)
- Patchify: patch size 2 → 256 patches,每个 patch 是 2×2×4 = 16 维,linear project 到 hidden dim
- 加 positional embedding
- AdaIN-zero modulation: timestep $t$ 和 condition (class label) 通过 MLP 输出 scale/shift 参数,modulate 每个 attention/MLP block
- LayerNorm → Attention → LayerNorm → MLP
- Output: velocity prediction $\mathbf{v}_\theta(\mathbf{z}_t, t)$

REPA 在 layer 8 的输出 (hidden state,不是 final output) 上加一个 MLP projection,然后跟 DINOv2 features 做 similarity loss。这个 MLP projection head 只在训练时用,inference 时丢弃。

模型配置 (Table 1):
| Config | #Layers | Hidden dim | #Heads |
|--------|---------|------------|--------|
| B/2    | 12      | 768        | 12     |
| L/2    | 24      | 1024       | 16     |
| XL/2   | 28      | 1152       | 16     |

---

## 4. 实验结果与数据解读

### Main result (Table 3, no CFG)

| Model | Iter. | FID ↓ |
|-------|-------|-------|
| SiT-XL/2 (vanilla) | 7M | 8.3 |
| SiT-XL/2 + REPA | **400K** | **7.9** |
| SiT-XL/2 + REPA | 1M | 6.4 |
| SiT-XL/2 + REPA | 4M | 5.9 |

400K iterations 的 REPA 已经超过 vanilla 7M 的结果——这就是 **17.5× speedup** 的来源。然后继续训还能进一步改进到 5.9 (no CFG)。

### System-level comparison (Table 4, with CFG)

| Model | Epochs | FID ↓ | sFID ↓ | IS ↑ | Pre. ↑ | Rec. ↑ |
|-------|--------|-------|--------|------|--------|--------|
| DiT-XL/2 | 1400 | 2.27 | 4.60 | 278.2 | 0.83 | 0.57 |
| SiT-XL/2 | 1400 | 2.06 | 4.50 | 270.3 | 0.82 | 0.59 |
| SiT-XL/2 + REPA | 200 | 1.96 | 4.49 | 264.0 | 0.82 | 0.60 |
| SiT-XL/2 + REPA | 800 | 1.80 | 4.50 | 284.0 | 0.81 | 0.61 |
| SiT-XL/2 + REPA + guidance interval | 800 | **1.42** | 4.70 | 305.7 | 0.80 | 0.65 |

1.42 是新的 SOTA on ImageNet 256×256 (with CFG)。guidance interval 来自 Kynkäänniemi et al. 2024 (https://arxiv.org/abs/2404.07724),只在 $t \in [0, 0.7]$ 范围内 apply CFG,w=1.80。

### Component analysis (Table 2)

**Target representation effect** (SiT-L/2, 400K iters):

| Target Repr. | FID ↓ | Acc ↑ |
|--------------|-------|-------|
| MAE-L | 12.5 | 57.3 |
| DINO-B | 11.9 | 59.3 |
| MoCov3-L | 11.9 | 63.0 |
| I-JEPA-H | 11.6 | 62.1 |
| CLIP-L | 11.0 | 67.2 |
| SigLIP-L | 10.2 | 68.8 |
| DINOv2-L | 10.0 | 68.1 |
| DINOv2-B | 9.7 | 65.7 |
| DINOv2-g | 9.8 | 65.7 |

**关键发现**: target encoder 越 strong (linear probing acc 越高),REPA 后 diffusion model 的 generation FID 越低,且自身的 linear probing 也越高。这是一个非常 clean 的 scaling law-like 关系 (Figure 5a 是 "Linear probing vs FID" 的散点图,呈现明显的负相关)。

值得注意: **MAE** 作为 target 是最差的——这正好印证了 paper 的 thesis: reconstruction-based SSL 学不到好 representation,把它蒸馏到 diffusion model 也没用。

CLIP 和 SigLIP (有 text supervision) 也很好,虽然它们没在 ImageNet 上训过 (CLIP 用 WIT-400M,SigLIP 用 WebLi-4B)。这表明 improvement 不是来自 dataset leakage。

### Scalability (Figure 5b/c)

模型越大,REPA 的相对改进越大:
- SiT-B/2: vanilla vs REPA 在某个 FID 上的 gap 较小
- SiT-L/2: gap 更大
- SiT-XL/2: gap 最大

**Intuition**: 这跟 scaling laws 的直觉一致——大模型 capacity 高,但需要合适的 inductive bias 来 guide 它。REPA 提供的 representation prior 在大模型上 leverage 得更充分。

### ImageNet 512×512 (Table 11, Appendix J)

| Model | Epochs | FID ↓ |
|-------|--------|-------|
| SiT-XL/2 | 600 | 2.62 |
| SiT-XL/2 + REPA | 80 | 2.44 |
| SiT-XL/2 + REPA | 100 | 2.32 |
| SiT-XL/2 + REPA | 200 | **2.08** |

这里 DINOv2 用 448×448 输入,positional embedding 插值到 64×64 latent grid 上。说明 REPA 在更高分辨率上也 work。

### Text-to-Image (Appendix K, Table 12)

在 MS-COCO 上,作者用 MMDiT (Stable Diffusion 3 的 backbone, Esser et al. 2024, https://arxiv.org/abs/2403.03206) 从头训:

| Method | FID |
|--------|-----|
| MMDiT (ODE; NFE=50) | 6.05 |
| MMDiT + REPA (ODE; NFE=50) | 4.73 |
| MMDiT (SDE; NFE=250) | 5.30 |
| MMDiT + REPA (SDE; NFE=250) | 4.14 |

T2I 也 work,即使有 text representation 通过 cross-attention 提供,visual representation alignment 仍然有用。

---

## 5. Ablations 与细节

### Time-step 上的 alignment (Figure 7)

REPA 在所有 noise level 上都缩小了 representation gap (linear probing 更高,CKNNA 更高),不只是 low-noise 区域。这有点 surprising——理论上 high-noise 区域 (t 接近 1) 应该很难从 noisy input 预测 clean semantic。但模型 apparently 学会了。

### 不同 encoder 的 alignment (Figure 8)

如果 target 是 MoCov3 或 MAE,训练后 diffusion model 的 features 跟相应 encoder 的 CKNNA 也提高。说明 REPA 是通用的 alignment scheme,不是 DINOv2-specific。

### λ 的 robustness (Table 5)

| λ | 0.25 | 0.5 | 0.75 | 1.0 |
|---|------|-----|------|-----|
| FID ↓ | 8.6 | 7.9 | 7.8 | 7.8 |
| IS ↑ | 118.6 | 122.6 | 124.4 | 124.8 |

0.5 之后基本饱和,说明 alignment 和 denoising 不是 trade-off,而是协同的——给 representation learning 多一点 weight 不会损害生成质量。这个 robustness 让我觉得 REPA 的成功不是巧合。

### Feature map visualization (Figure 38, Appendix L)

PCA visualization 显示 REPA 训出来的 model 有 coarse-to-fine 的 feature map (低 t 是 fine details,高 t 是 coarse semantic),而 vanilla model 在高 t 时 feature map 很 noisy。这说明 alignment 让模型在高 noise 时也能保持 semantic structure。

---

## 6. 跟其他工作的联系与我的联想

### 6.1 Knowledge distillation 视角

REPA 本质上是一种 **feature distillation**: teacher = DINOv2,student = diffusion transformer 的早期 layer。这跟 DreamTeacher (Li et al. 2023, https://arxiv.org/abs/2304.09095) 方向相反——DreamTeacher 是用 diffusion model 蒸馏到其他 model,REPA 是反过来。

更近的对比是 **FitNet** (Romero et al. 2014, https://arxiv.org/abs/1412.6550) 那种 intermediate layer hint,但这里 hint 来自不同 modality (clean image) 而不是相同输入。

### 6.2 与 MaskDiT, SD-DiT 的对比

MaskDiT (Zheng et al. 2024, https://arxiv.org/abs/2312.00052) 在 diffusion training 中加 MAE-style mask reconstruction。SD-DiT (Zhu et al. 2024) 加 MoCo-style contrastive loss。这些都是在 diffusion model 内部加 SSL auxiliary loss,而不是从外部 inject representation。REPA 的优势是直接用 best-in-class SSL encoder (DINOv2),不需要 diffusion model 自己 relearn 这些 features。

### 6.3 与 RCG (Li et al. 2024) 的对比

RCG (Representation-Conditioned Generation, https://arxiv.org/abs/2312.03701) 用两个 diffusion model:一个生成 1D representation vector,另一个 conditional on 这个 vector 生成 image。REPA 不需要两个 model,直接把 representation 作为 hidden state supervision。RCG 是 "modular" approach,REPA 是 "integrated" approach。

### 6.4 与 Platonic Representation Hypothesis 的关系

Huh et al. 2024 (https://arxiv.org/abs/2405.10318) 提出:不同 model 在 scale 上去后会收敛到同一个 representation space。REPA 的 Observation 2 (SiT 跟 DINOv2 已经 weakly aligned) 正是这个 hypothesis 的 evidence。REPA 做的事就是加速这个 convergence——把 7M iterations 才能达成的 alignment,用 400K iterations 完成。

### 6.5 跟 JEPA 的哲学相似性

JEPA (LeCun, https://arxiv.org/abs/2301.08243) 强调在 representation space 而不是 pixel space 做 prediction,因为 pixel-level reconstruction 浪费 capacity 在 irrelevant details 上。REPA 在 representation space 做 alignment,跟这个哲学一致。Diffusion model 本身是 pixel/latent-space reconstruction,但 REPA 给它加了一个 representation-level supervision,某种程度上让 model 不必 "all in" on pixel reconstruction。

### 6.6 关于 "encoder-decoder" 视角

论文把 diffusion model 看作 $g_\theta \circ f_\theta$,其中 $f_\theta$ 是 implicit encoder 学 representation $\mathbf{h}_t$,$g_\theta$ 是 decoder 输出 velocity。这跟 denoising autoencoder (Bengio et al. 2013, https://ieeexplore.ieee.org/document/6472238) 的 framing 一致。

REPA 本质是给 $f_\theta$ 加 supervision,而 $g_\theta$ 仍然 free to learn 像素生成。这个 decoupling 让两部分各司其职。

### 6.7 与 Classifier-Free Guidance 的协同

CFG (Ho & Salimans, https://arxiv.org/abs/2207.12598) 在 inference 时用 conditional 和 unconditional prediction 的差作为 guidance。REPA 改善了 conditional prediction 的 quality,所以 CFG 的 baseline 更好。Guidance interval (Kynkäänniemi et al. 2024) 进一步只在 $t \in [0, 0.7]$ apply CFG——因为 high-noise 区域的 conditional/unconditional 差异主要是 semantic,REPA 已经把这个搞定了。

### 6.8 一些可能的 extension (作者在 Appendix M 提到)

- **Time-varying λ**: 根据 noise schedule 调整 alignment strength,可能在 high-noise 时更需要 alignment
- **Pixel diffusion 而不是 latent diffusion**: latent diffusion 的 VAE 已经 compressed 了信息,pixel diffusion 上 REPA 效果可能不同
- **Video diffusion**: DINOv2 主要是 image encoder,video 上需要 DINOv2-Video 之类的
- **Theoretical analysis**: 为什么 instance discrimination objective 和 denoising objective 的 representations 可以 align?跟 score matching 的 connection 是什么?

### 6.9 我自己的一些 speculation

1. **REPA 跟 next-token prediction 的关系**: 训 LLM 时我们不需要额外的 representation alignment——next-token prediction 本身就 implicitly 学到好 representation (因为 token 是离散 semantic unit)。Diffusion 训在 continuous pixel/latent space,reconstruction objective 被 high-frequency detail 主导,所以 representation 学不好。REPA 相当于给 diffusion model 一个 "semantic token prediction" 的 auxiliary signal。

2. **为什么 layer 8 而不是 layer 0**: layer 0 的 hidden state 就是 patch embedding,还没有 context mixing。需要几层 attention 才能让 patches 之间互相 "see",这时候 align 才有意义。但太深了又会让 representation 过度 contextualized,失去跟 patch-level DINOv2 features 的对应。Layer 8 是 sweet spot。

3. **REPA 跟 Diffusion + Discriminative head 的 hybrid**: 一些工作 (Yang et al. 2022, https://arxiv.org/abs/2208.07791; Deja et al. 2023) 训一个 model 同时做 classification 和 generation。REPA 跟这些不同——它不在 output 加 discriminative head,而是 align intermediate features。后者更 "soft",不会干扰 generation。

4. **REPA 跟 Sora 的潜在 connection**: Sora 用 DiT 做 video generation。如果 REPA 在 video 上也 work (用 video SSL encoder),可能能加速 Sora-like model 的训练。Paper 里 Appendix K 的 T2I 实验是 small-scale,但效果显著。

5. **REPA 跟 distillation in LLMs 的类比**: 把 DINOv2 features 蒸馏到 diffusion transformer,跟 "distill GPT-4 logits into smaller model" 类似。但 REPA 是 feature-level 不是 logit-level,而且是 cross-modal (encoder → diffusion) 不是同模态。

---

## 7. 局限与 open questions

### 7.1 作者承认的 limitation

- 主要在 latent diffusion 上验证,pixel diffusion 没试
- 主要在 image 上,video 没试
- 没有理论解释为什么 alignment 能加速 training

### 7.2 我自己想到的 issues

- **DINOv2 的 inductive bias 被 baked in**: 用 DINOv2 作为 target 等于把它的 bias (e.g., 对 texture vs shape 的偏好) 注入 diffusion model。如果 DINOv2 有 systematic bias,diffusion model 也会继承。
- **Patch correspondence 问题**: 论文假设 diffusion transformer 第 $n$ 个 patch 对应 DINOv2 第 $n$ 个 patch。在 latent diffusion 中,diffusion patch 是 2×2 latent,DINOv2 patch 是 14×14 pixel。这个 correspondence 不是天然的,需要 positional embedding interpolation。可能有 misalignment 风险。
- **Inference 时的 representation drift**: 训练时 REPA loss 推动 representation 接近 DINOv2,但 inference 时是 free-running ODE/SDE,representation 可能 drift。论文没分析这个。
- **多模态 extension**: T2I 已经 work,但 audio、3D 等其他模态需要相应的 SSL encoder。REPA 的 generalization 取决于目标模态有没有 strong SSL representation。

### 7.3 跟 Karpathy 的 "Software 2.0" 视角

从 Software 2.0 的角度,REPA 是一种 "transfer learning at the representation level"。传统 transfer learning 是 fine-tune 整个 pretrained model,REPA 是把 pretrained representation 作为 auxiliary supervision,让 diffusion model 在保持自己 architecture 和 objective 的同时利用 external representation。这跟 LLM 里 "logit distillation from larger model" 类似,但更 fine-grained。

---

## 8. 一些相关链接

- **DiT**: https://arxiv.org/abs/2212.09748
- **SiT**: https://arxiv.org/abs/2401.08715
- **DINOv2**: https://arxiv.org/abs/2304.07193
- **MAE**: https://arxiv.org/abs/2111.06377
- **I-JEPA**: https://arxiv.org/abs/2301.08243
- **MoCov3**: https://arxiv.org/abs/2104.02057
- **CLIP**: https://arxiv.org/abs/2103.00020
- **SigLIP**: https://arxiv.org/abs/2303.15343
- **Stable Diffusion / LDM**: https://arxiv.org/abs/2112.10752
- **Flow Matching**: https://arxiv.org/abs/2210.02747
- **Stochastic Interpolants**: https://arxiv.org/abs/2303.08797
- **Classifier-Free Guidance**: https://arxiv.org/abs/2207.12598
- **Guidance Interval**: https://arxiv.org/abs/2404.07724
- **Platonic Representation Hypothesis**: https://arxiv.org/abs/2405.10318
- **CKA**: https://arxiv.org/abs/1905.00414
- **DreamTeacher**: https://arxiv.org/abs/2304.09095
- **RCG**: https://arxiv.org/abs/2312.03701
- **MaskDiT**: https://arxiv.org/abs/2312.00052 (实际链接: https://arxiv.org/abs/2312.00052)
- **Vincent 2011 (denoising autoencoder connection)**: https://www.mitpressjournals.org/doi/10.1162/NECO_a_00142
- **DreamTeacher / REPA original**: https://arxiv.org/abs/2410.06940 (REPA 论文)
- **MMDiT (SD3)**: https://arxiv.org/abs/2403.03206
- **U-ViT**: https://arxiv.org/abs/2209.12152
- **MDTv2**: https://arxiv.org/abs/2303.14389
- **DiffiT**: https://arxiv.org/abs/2404.01657
- **EDM2 (Karras et al. 2024)**: https://arxiv.org/abs/2312.02696
- **SimCLR (NT-Xent)**: https://arxiv.org/abs/2002.05709
- **Bengio et al. 2013 (Representation Learning Review)**: https://ieeexplore.ieee.org/document/6472238

---

## 9. Take-away intuitions

总结几个我从中提取的 intuition:

1. **Diffusion model 训练困难不是生成难,是 representation learning 难**。生成的 "decoding" 部分模型很容易学,但 "encoding" 出 semantic representation 被 reconstruction objective 拖累。

2. **External representation 可以做 shortcut**。Platonic convergence 是 natural trend,直接 align 加速 17.5×。

3. **Layer-wise division of labor**: 早期 layers 做 semantic,后期 layers 做 high-frequency details。REPA 只 align 早期 layers (layer 8/28),保留后期 layers 的 generation capacity。

4. **Clean target, noisy input**: 把 DINOv2 当 teacher 不需要 fine-tune 它处理 noisy input,让 student (diffusion transformer) 自己 learn "denoise in representation space"。

5. **Strong encoder → stronger diffusion model**: DINOv2 比 MAE 好,MoCov3 居中——encoder quality 直接 transfer 到 generation quality。这暗示 representation quality 是 generation quality 的 bottleneck。

6. **Scalable inductive bias**: 大模型从 REPA 获益更多——这跟 "inductive bias helps more when capacity is high" 的直觉一致。

7. **Generalizes across objectives**: DDPM、stochastic interpolants、SiT、DiT、MMDiT 上都 work,说明 REPA 不是 objective-specific trick,而是 representation-level 的 fundamental 加速。

这 paper 给我的最大启发是: **生成模型和判别模型在 representation 层面是同一个东西的两个 facet,把它们 explicit 桥接起来比让生成模型自己摸索要高效得多**。这跟 LLM 里 "next-token prediction 自动学 representation" 的高效性形成 contrast——continuous generation 比 discrete generation 更需要这种 explicit alignment。

如果让我推测 next steps,我觉得最 promising 的方向是:
- Time-varying REPA (根据 noise schedule 调 λ)
- Video REPA with video SSL encoder (e.g., VideoMAE-v2, DINOv2-Video)
- 跟 distillation 结合,用 REPA-trained model 作为 teacher 蒸馏更小的 student
- Cross-modal REPA (e.g., align image diffusion model with text encoder features for stronger T2I)

这是一个 "obvious in hindsight" 的工作——所有 ingredients (DiT, DINOv2, CKA, contrastive loss) 都已经存在,但把它们组合起来得到 17.5× speedup 是非常 clean 的贡献。
