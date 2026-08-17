---
source_pdf: Improved Baselines with Representation Autoencoders.pdf
paper_sha256: 8aff665f128a822068f93c9b703e10cd66e18f32339dfa0a47206847640084fa
processed_at: '2026-08-05T09:21:46-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Hey Andrej, 我用最直白的大白话，结合底层的数学直觉，给你拆解一下这篇 RAEv2。

## 1. 背景故事：为什么要搞 RAE？

在 diffusion model 生成图片的 pipeline 里，通常先用 VAE 把 256x256 的图片压缩成 16x16 的 latent tokens，然后让 DiT (Diffusion Transformer) 在这个 latent space 里做去噪。

传统 VAE 的毛病在于，它只关注像素级的重建（reconstruction），latent space 缺乏 high-level 的 semantic。这就导致 DiT 需要从头学习“什么是狗”、“什么是猫”这种全局概念。

同时，业界有一堆极其强大的 pretrained vision encoder（比如 DINOv2, DINOv3），它们已经具备了极好的 semantic understanding，但是因为被训练成做分类/对比学习，其 feature 缺乏低级纹理细节，没法直接用来重建图片。

所以之前的 RAE (Representation Autoencoders) 提出了一个大胆的想法：直接把 DINOv2 当作 VAE 的 encoder，冻结它，只在后面接一个轻量 decoder 让它学着把图片拼回来。这样 DiT 就直接在一个充满 semantic 的 latent space 里工作了。

但是原始 RAE 有三个痛点：重建差、收敛慢、且传统的 classifier-free guidance (CFG) 在它的 latent space 里直接失效。

RAEv2 就是来收拾这个烂摊子的。它用了三个极其简单但精妙的 insight，把训练速度提升了 10 倍。

---

## 2. Insight 1: 别只用最后一层，把最后 K 层加起来

**人话直觉**：
在 DINOv3 这种深层 transformer 里，最后一层包含了“这是一只狗”的全局语义，但是低级几何和纹理信息（比如狗毛的走向、背景的窗户）在反向传播中被 wash 掉了。如果你只用最后一层，你的 reconstruction 一定很差，因为细节都丢了。但是，这些细节其实保留在倒数第 2 层、倒数第 3 层里。

怎么融合？直接拼接太贵，会让 latent 维度爆炸。这篇论文发现了一个极其简单的方法：直接做 element-wise addition。

**技术细节与公式**：
论文提出了 Multi-Layer Sum (MLS)，公式如下：

$$\mathbf{x} = \sum_{\ell = L - K + 1}^{L} \mathbf{z}_\ell \in \mathbb{R}^{N \times d}$$

变量解释：
*   $\mathbf{x}$: 最终输入给 DiT 的 latent representation。
*   $L$: pretrained encoder 的总层数（DINOv3-L 是 24 层）。
*   $K$: 我们要相加的最后 $K$ 层的数量（超参数）。
*   $\ell$: 层的索引下标。
*   $\mathbf{z}_\ell$: 第 $\ell$ 层输出的 feature map。
*   $N$: spatial token 的数量（16x16 = 256）。
*   $d$: feature 的隐藏维度（DINOv3-L 是 1024）。

**为什么简单的 addition 能行？**
直觉在于高维空间的几何特性。在 1024 维这种极高维的空间里，不同层提取的特征 subspace 几乎是正交的。语义层和纹理层的特征向量方向不同，element-wise addition 相当于把这些不同频率的信号直接叠加，而不会产生严重的 destructive interference。这跟 Feature Pyramid Networks (FPN) 里面用 element-wise sum 融合不同 level 特征的逻辑一脉相承，同时也契合 Johnson-Lindenstrauss 引理关于高维空间保距的特性。

**实验数据**：
通过调 $K$，可以直接控制 reconstruction 和 generation 的 trade-off：
*   $K=1$ (原版 RAE): rFID = 0.60, PSNR = 18.93
*   $K=7$ (Generation 最佳): rFID = 0.29, PSNR = 22.57, guided gFID = 1.06
*   $K=23$ (Reconstruction 最佳): rFID = 0.18, PSNR = 27.03

可以看到，仅仅通过把最后 7 层加起来，rFID 降了一半，gFID 达到了 SOTA。这完全 training-free，连 encoder 参数都没动一下。

---

## 3. Insight 2: RAE 和 REPA 是互补的，别互斥

**人话直觉**：
REPA (Representation Alignment) 是另一种加速 DiT 的技术。它的做法是在 DiT 的中间层（比如第 8 层）接一个 projection head，强行让 DiT 中间层的 feature 去逼近 DINOv2 的 feature。

很多人觉得：我已经用 RAE 把 DINOv2 当作 latent 输入给 DiT 了，DiT 内部再去 distill DINOv2，这就相当于做了一次 skip connection，纯属浪费模型容量。Saining Xie 之前甚至在 Twitter 上发过 "RIP REPA"。

但这篇论文通过 27 个 encoder 的大规模实验发现：**RAE 和 REPA 的工作机制完全正交，同时用效果巨幅提升。**

怎么理解这种正交？
*   **RAE 解决了 "What to generate" (全局语义)**：它给 DiT 提供了一个极好的起跑线，DiT 一开始就知道要生成狗。
*   **REPA 解决了 "How to spatially arrange" (空间结构)**：它约束了 DiT 中间层 token 之间的 self-similarity 结构，告诉 DiT “狗的头和尾巴在 spatial 上应该怎么 attend to each other”。

**技术细节与实验数据**：
论文用 Linear Probing (LP) 衡量全局语义，用 Local Distance Similarity (LDS) 衡量空间结构。计算它们与 gFID 的 Pearson 相关系数 $r$（负值代表负相关，绝对值越大越相关）：

| Method | LP ($r$) | LDS ($r$) | Average ($r$) |
| :--- | :--- | :--- | :--- |
| REPA alone | +0.34 (反相关) | -0.89 (强相关) | -0.56 |
| RAE alone | -0.81 (强相关) | -0.13 (几乎无相关)| -0.55 |
| RAE + REPA | -0.64 | -0.53 | **-0.83** |

这说明什么？单独用 REPA，全局语义不仅没用反而有反作用；单独用 RAE，空间结构根本学不到。两者结合后，相关性达到了 -0.83 的极强负相关。因为它们在处理不同的子问题，叠加在一起产生了 super-additive 的效果。这也解释了为什么像 DINOv3-L 这种在 LP 和 LDS 上都很强的 encoder，在 RAEv2 下能吊打之前所有的 encoder。

参考链接：
*   REPA 原论文: https://arxiv.org/abs/2410.06940
*   iREPA (讲 spatial structure 的重要性): https://arxiv.org/abs/2512.10794

---

## 4. Insight 3: REPA 头可以白嫖当 Guidance 用

**人话直觉**：
在 diffusion 采样时，CFG 需要跑一次 conditional 和一次 unconditional 前向传播，NFE 翻倍。而 RAE 的 latent space 空间太结构化，CFG 直接失效。之前的解法是 AutoGuidance (AG)，即另外训练一个小号的 DiT 当作 weak baseline 来做对比。

这篇论文发现了一个极度聪明的白嫖方法：我们在 Insight 2 里给 DiT 第 8 层接的 REPA head，本质上就是一个只用了一半 transformer 层的 "小号 DiT"！只要我们把它预测的目标重新参数化为 x-prediction，它就可以直接充当 AutoGuidance 里的那个 weak baseline。

**技术细节与公式**：
假设 full model (跑完 28 层) 输出为 $\hat{\mathbf{x}}_\text{full}$，REPA head (只跑到第 8 层) 输出为 $\hat{\mathbf{x}}_\text{repa}$。Guidance 公式为：

$$\hat{\mathbf{x}}_\text{guided} = \hat{\mathbf{x}}_\text{full} + w \cdot (\hat{\mathbf{x}}_\text{full} - \hat{\mathbf{x}}_\text{repa}) \tag{3}$$

变量解释：
*   $\hat{\mathbf{x}}_\text{full}$: 完整 DiT 预测的 clean latent。
*   $\hat{\mathbf{x}}_\text{repa}$: 第 8 层浅层 head 预测的 clean latent。
*   $w$: guidance scale 超参数，控制偏离程度。
*   $\hat{\mathbf{x}}_\text{guided}$: 最终用来推导 velocity 的 guided latent。

采样时再用 $\boldsymbol{\nu} = (\mathbf{x}_t - \hat{\mathbf{x}}_\text{guided}) / t$ 转回 velocity。

**为什么这很牛？**
因为在同一个 forward pass 里，第 8 层的输出早就算出来了，算 $\hat{\mathbf{x}}_\text{repa}$ 几乎零开销。这直接干掉了额外的 guidance model，并且把推理时的 NFE 减半。这本质上是对 Deep Supervision 思想的极限压榨：中间层的监督信号在推理时变成了自我纠偏的 guidance 信号。

参考链接：
*   AutoGuidance: https://arxiv.org/abs/2410.02391
*   CFG 原论文: https://arxiv.org/abs/2207.12598

---

## 5. 衡量效率的新指标：EPFID@k

**人话直觉**：
gFID 从 1.13 降到 1.06 其实没多大实际意义，因为这里面噪声很大。但是从 800 epochs 降到 80 epochs 达到同样的 gFID，意义极其重大。这就像 LLM 领域的 nanoGPT speedrun，大家比的是“达到某个 loss 需要多少分钟”。

所以作者提出了 EPFID@k (Epochs to reach unguided gFID $\leq k$)：

| Method | Epochs | EPFID@2 ↓ | gFID ↓ |
| :--- | :--- | :--- | :--- |
| REPA-E | 800 | 480 | 1.12 |
| RAE-XL (原版) | 800 | 177 | 1.13 |
| **RAEv2 (Ours)** | 80 | **35** | **1.06** |

达到 gFID $\leq 2$ 所需的 epochs，从 480 直接降到 35，效率提升了 13 倍。这种效率提升让你可以在 12 小时内用 32 张 H100 跑出一个 SOTA 图像生成模型，这对业界的工程落地是颠覆性的。

参考链接：
*   modded-nanogpt speedrun: https://github.com/KellerJordan/modded-nanogpt

---

## 6. 泛化到 World Models 的直觉

这点我觉得 Karpathy 你会特别喜欢。他们把 RAEv2 用到了 Navigation World Models (NWM) 上，也就是给模型看过去 4 帧视角，让它自回归预测未来 16 秒的帧。

在 RECON dataset 上，RAEv2 把 FVD 从 NWM 的 200.97 直接干到了 105.61。

**为什么降这么多？**
因为在 world model 里，时间一致性极度依赖低级几何和纹理的连续性。原版 RAE 只用最后一层，丢掉了低频纹理，导致预测的下一帧窗户数量都会变（论文里提到的 flickering 现象）。RAEv2 用最后 K 层相加，保留了这些 low-level detail，使得长程 autoregressive rollout 时画面不再闪烁。

这从侧面印证了一个很深刻的直觉：**对于控制、导航、视频生成这类需要极强时空连续性的任务，global semantic 只是入场券，low-level spatial structure 才是长程稳定性的基石。**

参考链接：
*   NWM (Yann LeCun 组的): https://arxiv.org/abs/2412.03572
*   RECON dataset: https://arxiv.org/abs/2310.07896

---

## 7. 总结与更多联想

RAEv2 的成功可以归结为一种“回到 basics”的 engineering 胜利。没有发明新架构，没有搞复杂的 loss，就是通过严格的大规模 ablation 找到了 latent space 的正确使用方式。

我甚至可以联想到这和 LLM 里的 residual stream 概念非常像。在 LLM 里，不同 layer 的 residual stream 存储了不同频率的信息，最后 unembedding matrix 把它们一起读出来。RAEv2 的 MLS 其实就是显式地在 vision encoder 里做了类似的事情，通过相加让 decoder 能够同时读取高频和低频信息。

未来可以联想的方向：
1. **Learnable MLS**: 目前是等权相加，如果每一层给一个 learnable scalar，或者用 attention pooling 去动态融合不同层，在 reconstruction 上应该还能进一步提升。
2. **Unified Any-to-Any Tokenization**: 既然 RAE latent space 同时保持了 linear probing 的准确度（85.1%）和生成质量（1.06 gFID），这意味着 LLM 的 text token、Vision encoder 的 image token、Diffusion 的 latent token 完全可以统一在一个空间里，这对真正的 multimodal AGI 系统架构是一个极强的信号。

希望能帮你 build up 这个直觉！如果对其中某些公式的梯度流或者具体实现细节有更进一步兴趣，我们可以继续深挖。

---

# Improved Baselines with Representation Autoencoders (RAEv2) 详细讲解

## 1. 背景与动机

Representation Autoencoders (RAE) 是最近由 Zheng et al. 提出的一种 framework，用 pretrained vision encoder（如 DINOv2）的 features 直接作为 latent space，替换 traditional VAE。这样做的好处是 unified tokenization——understanding 和 generation 共享同一套 representation。原始 RAE paper 在 ImageNet-256 上取得了不错的结果，但存在三个核心问题：

1. **Reconstruction 不够好**：相比 SDVAE、Flux-VAE 这类专门训练的 VAE，RAE 的 rFID 较差
2. **CFG 不兼容**：standard classifier-free guidance 在 RAE latent space 上失效，需要训练一个 secondary weaker model（AutoGuidance, AG），增加 compute 和 complexity
3. **Encoder 层次未被充分利用**：之前的 RAE 直接用 final layer output，但不同层包含互补信息

RAEv2 这篇 paper 通过三个 insight 解决了这些问题，在 ImageNet-256 上 80 epochs 就达到 gFID 1.06，相比 RAE 的 800 epochs 训练效率提升 10× 以上。

参考链接：
- 原始 RAE paper: https://arxiv.org/abs/2510.11690
- REPA paper: https://arxiv.org/abs/2410.06940
- iREPA paper: https://arxiv.org/abs/2512.10794
- DINOv3: https://arxiv.org/abs/2508.10104

---

## 2. 三个核心 Insight

### 2.1 Insight 1: Generalized Representation Encoder

**核心问题**：Pretrained vision encoder 的不同层捕获不同类型的信息。Figure 15 显示：
- 早期-middle 层：保留 finer spatial structure、low-level texture、geometry
- 后期层：强调 global semantics、object-level 概念

如果只用 final layer，就丢失了 low-level 信息，导致 reconstruction 质量差。但直接 concatenation 又会导致 latent space 维度爆炸（N×Ld 或 LN×d），让 diffusion model 训练困难。

**两种 parameter-free 的方案**：

#### 方案 A: Multi-Layer Sum (MLS) - 公式 (1)

$$\mathbf{x} = \sum_{\ell = L - K + 1}^{L} \mathbf{z}_\ell \in \mathbb{R}^{N \times d}$$

变量解释：
- $\mathbf{x}$：聚合后的 encoder output，作为 RAE 的 latent representation
- $L$：pretrained encoder 的总层数（DINOv3-L 为 24 层）
- $K$：要聚合的最后 K 层数量（hyperparameter）
- $\ell$：层索引下标
- $\mathbf{z}_\ell$：第 $\ell$ 层的 feature map
- $N$：token 数（patch grid 大小，16×16=256）
- $d$：每个 token 的 feature 维度（DINOv3-L 为 1024）

这个公式背后的几何直觉来自 Johnson-Lindenstrauss 类的 insight：在 high-dimensional space 中，addition 能 preserve underlying subspace 的 geometric structure。所以简单相加不会破坏各层的 representation 信息，反而能实现"互补融合"。

#### 方案 B: Multi-Layer Random projection (MLR) - 公式 (2)

$$\mathbf{x} = [\mathbf{z}_{L-K+1} \| \cdots \| \mathbf{z}_L] \mathbf{R} \in \mathbb{R}^{N \times d}$$

变量解释：
- $\mathbf{R} \in \mathbb{R}^{Kd \times d}$：固定 random matrix（i.i.d. Gaussian，初始化后 freeze）
- $[\cdot \| \cdots \| \cdot]$：channel 维度 concatenation
- 这个 projection 期望意义上 preserve pairwise distances（standard random projection 思想）

**关键 ablation 结果（Table 3, 15）**：MLS 和 MLR 在 Stage-1 reconstruction 上几乎打平，但 MLS 在 Stage-2 generation（gFID）上全面胜出。比如 K=2 时 MLR 的 gFID=3.085，MLS=2.586；K=8 时 MLR=3.580，MLS=2.688。

**直觉解释**：Random projection 虽然理论上 distance-preserving，但在 vision feature 这种 structured data 上，simple addition 更能保留 "task-relevant" 的几何关系。Addition 在 high-dimensional space 中起到了类似于 "spectral filtering" 的作用——语义层和细节层的 representation 在 subspace 中近似正交，加起来反而同时保留了两者。

**K 的选择（Figure 7）**：
- K=1（原 RAE）：rFID=0.60，PSNR=18.93
- K=7：rFID=0.29，PSNR=22.57，guided gFID=1.06（最佳 generation）
- K=23（full MLS）：rFID=0.18，PSNR=27.03，guided gFID=1.25

有一个非常有趣的现象：reconstruction 单调随 K 增加变好，但 unguided generation 在 K=1 附近最好（1.50），guided generation 在 K=7 附近最好。这说明 guidance 容量为 K 提供了额外的 "headroom"，可以利用更深的 representation 信息。

**对 understanding 的影响（Table 5, 18）**：linear probing 在不同 K 下基本保持不变（85.10-85.39），说明 generalized formulation 在改进 generation/reconstruction 的同时不损害 understanding，这对 unified tokenization 是关键验证。

---

### 2.2 Insight 2: RAE 和 REPA 的互补机制

**主流 assumption**：RAE 已经用 pretrained encoder 作为 latent，再 distill 同样的 representation 到中间 diffusion layers（REPA）就是 wasteful skip connection。Saining Xie 本人在 Twitter 上也表达过类似观点 [58]。

**Empirical finding**：实验推翻了这个 assumption。在 27 个 vision encoder 上做 ablation，无论哪个 encoder，RAE + REPA 都比单独的 RAE 或 REPA 更好。

**为什么互补？** 通过两种指标分析：

1. **Linear Probing (LP)**：测量 global semantic quality
2. **Local Distance Similarity (LDS)**（来自 iREPA [48]）：测量 token 间的 spatial self-similarity structure

**Correlation analysis（Figure 6, Table 在 Figure 6d）**：
- REPA alone (with VAE)：LP 的 Pearson r=+0.34（**反相关**），LDS 的 r=-0.89（**强相关**）
- RAE alone：LP 的 r=-0.81（**强相关**），LDS 的 r=-0.13（**几乎无相关**）
- RAE + REPA：LP=-0.64，LDS=-0.53，average r=-0.83（**最强**

注：这里 r 是与 gFID 的相关，所以负值越强代表 representation 越好则 gFID 越低。

**工作机制的直觉**：
- RAE 提供 semantically rich 的 latent space——diffusion model 在 "what to generate" 上有强先验
- REPA regularize 中间 diffusion features 的 token-token 相似性结构——解决 "how spatial structure should look"

Figure 5 的可视化非常说明问题：加上 REPA 后，diffusion features 的 linear probing 几乎不变，但 spatial self-similarity 结构显著改善。这印证了 REPA 主要影响 spatial structure 而非 global semantics。

**这个互补性的重要推论**：之前 RAE 选择 DINOv2-B 作为 encoder（在 original RAE recipe 下 generation 最好），但 RAEv2 下 DINOv3-L（同时有强 LP 和强 LDS）反而最好（Table 2）。Table 12 显示了一个清晰的 trend：综合 score `Avg(LP', LDS')` 越高，generation 越好。

**Self-REPA**：REPA 的 target encoder 和 RAE 的 encoder 是同一个，这样既简化了 pipeline 又能持续受益。

---

### 2.3 Insight 3: REPA as x-prediction for Self-Guidance

**问题**：原始 RAE 用 AutoGuidance（AG）训练第二个 weaker model，开销大。CFG 在 RAE latent space 上失效（Table 1：CFG gFID=3.86 比 w/o guidance 的 3.75 还差）。

**Key observation**：REPA 的 projection head $h_\phi$ 接收 early-layer features $h$，predicts clean latent $\hat{x}_\text{repa} = h_\phi(h)$。在 RAE 设定下 clean latent 就是 encoder representation $x = E(I)$。所以 **REPA head 就是在 RAE latent space 上做 x-prediction**！

**重要 reformulation**：把 full model output 也改成 x-prediction（不是默认的 velocity prediction），然后两个输出处于同一空间，可以直接做 internal guidance：

$$\hat{\mathbf{x}}_\text{guided} = \hat{\mathbf{x}}_\text{full} + w \cdot (\hat{\mathbf{x}}_\text{full} - \hat{\mathbf{x}}_\text{repa}) \tag{3}$$

变量解释：
- $\hat{\mathbf{x}}_\text{full}$：full model（所有 transformer layers）的 x-prediction
- $\hat{\mathbf{x}}_\text{repa}$：REPA head（只用 early-layer features 的轻量 MLP）的 x-prediction
- $w$：guidance scale，超参数
- $\hat{\mathbf{x}}_\text{guided}$：最终 guided prediction

采样时再转换回 velocity：$\boldsymbol{\nu} = (\mathbf{x}_t - \hat{\mathbf{x}}_\text{guided}) / t$，其中 $\mathbf{x}_t$ 是 noisy latent，$t$ 是 time step。

**为什么这等同于 AutoGuidance？** 因为 REPA head 只访问 early-layer features，capacity 远小于 full model，本质上就是一个 "weak baseline"。但这个 weak baseline 是在同一个 forward pass 中免费得到的——不需要训练第二个 model，也不需要额外的 unconditional forward pass（CFG 那样）。

**NFE 节省**：CFG 需要 conditional + unconditional 两次 forward，NFE 翻倍。RAEv2 在同一 forward 中拿到 $\hat{x}_\text{full}$ 和 $\hat{x}_\text{repa}$，NFE 减半。

**Ablation（Table 4, 16）**：
- 无 guidance：K=7 gFID=1.65，K=23 gFID=3.01
- CFG：K=7 gFID=1.49，K=23 gFID=2.83
- AG：K=7 gFID=1.14，K=23 gFID=1.37
- **REPA Guidance**：K=7 gFID=1.06，K=23 gFID=1.25（最好）

**x-prediction 的重要性（Table 17）**：internal guidance without x-prediction reparameterization 只有 1.87 gFID，加上 x-prediction 达到 1.65。这个 reparameterization 让 REPA head 和 full head 真正在同一空间，才能做有意义的 linear combination。

**与 deep supervision 的联系**：作者提到这本质上相当于 deeply-supervised network（Lee et al. 2015 [32]）的 modern 版本——在不同深度监督同一个目标，但通过 reparameterization 让中间 head 同时承担 guidance 功能。

---

## 3. 架构与训练细节

### 3.1 整体架构（Table 10, 11）

**Backbone**：DiTDH-XL（来自 DDT [54]）：
- Encoder：28 blocks，hidden dim 1152，16 attention heads
- Decoder：2 blocks，hidden dim 2048，16 heads
- MLP ratio 4.0，SwiGLU，RoPE + APE，RMSNorm
- Latent patch size 1，产生 16×16=256 tokens

**RAE Encoder**：
- ImageNet & World Models：DINOv3-L（256×256 输入，patch 16，输出 1024×16×16）
- T2I：SiGLIP2-B（输出 768×16×16）
- Decoder 单独预训练 16 epochs，然后 freeze

**REPA 配置**：
- Alignment layer depth = 8（在 transformer 的第 8 个 block 加 REPA loss）
- Projection：linear，从 1152 维 mapping 到 encoder 维度
- REPA coefficient λ = 0.5
- Target encoder 与 RAE encoder 相同（self-REPA）

**Training**：
- LR = 2e-4，linear decay 到 2e-5（epoch 50）
- 25 epochs warmup，batch size 1024
- bfloat16 混合精度，gradient clip max norm 1.0
- EMA decay 0.9995
- 80 epochs（ImageNet）

**Flow matching**：continuous-time flow matching，x-prediction，logit-normal time sampling

**Conditioning**：in-context conditioning（不用 adaLN-Zero），4 个 timestep token + 8 个 class token（ImageNet）或 256 个 text token（T2I）或 1024+4+1 个 token（NWM）

### 3.2 EPFID@k 指标

论文提出新指标 EPFID@k：达到 unguided gFID ≤ k 所需的 epochs 数。这类似于 language model 领域的 speedrun（参考 modded-nanogpt [28]）。

**Table 7 结果**：
- SiT-XL/2：>800 epochs
- DDT-XL：>800
- LightningDiT：>800
- REG：560
- REPA-E：480
- RAE-XL：177
- **RAEv2：35**

这个改进是惊人的——从 RAE 的 177 降到 35，超过 5× 提速。从原始 SiT 的 >800 到 RAEv2 的 35，超过 20× 提速。

参考链接：modded-nanogpt speedrun: https://github.com/KellerJordan/modded-nanogpt

---

## 4. 实验结果深度分析

### 4.1 Reconstruction vs Generation Trade-off（Table 13）

对比最新的 representation-based autoencoders：

| Method | Training-free | rFID↓ | gFID↓ |
|--------|--------------|-------|-------|
| DINO-Tok [26] | ✗ | 0.32 | 5.94 |
| DINO-SAE [8] | ✗ | 0.37 | 3.07 |
| VFM-VAE [5] | ✗ | 0.52 | 3.41 |
| AlignTok [9] | ✗ | 0.26 | 3.71 |
| RPiAE [20] | ✗ | 0.50 | 2.25 |
| RAE [69] | ✓ | 0.602 | 2.23 |
| **RAEv2 (K=7)** | ✓ | 0.29 | 1.65 |
| **RAEv2 (K=23)** | ✓ | 0.18 | 3.02 |

关键观察：RAEv2 同时拿到 **最佳 generation (K=7)** 和 **最佳 reconstruction (K=23)**，且是 training-free 的。其他方法需要 finetune encoder、auxiliary loss、架构修改。这种"调一个 K 就能在 recon-gen spectrum 上滑动"的设计非常有用。

### 4.2 FD^r 评估（Table 7）

Representation Fréchet Distance [61] 在 6 个 feature space 上计算：Inception, ConvNeXt, DINOv2, MAE, SigLIP, CLIP。RAEv2 80 epochs 在所有 6 个上都达到 best，包括 FD 几何平均 2.17，远超 RAE 的 3.26（800 epochs）。

参考：FD^r paper: https://arxiv.org/abs/2604.28190

### 4.3 Monge Distance（Table 19）

新指标 MDr [4]（基于 optimal transport）进一步验证 RAEv2 的优越性，在 5/6 个 feature space 上达到 best。

参考：MIND/Monge distance: https://arxiv.org/abs/2605.06797

### 4.4 模型尺度（Table 6）

RAEv2 在 B (165M)、L (470M)、XL (839M) 三个 scale 上都比 RAE 有显著改进：
- B：5.48 → 3.37
- L：3.80 → 2.76
- XL：3.75 → 2.61

### 4.5 Encoder 选择（Table 12）

完整的 encoder sweep（按 Avg(LP', LDS') 排序）显示一个清晰 trend：composite score 越高，generation 越好。DINOv3-L（LP=87.0, LDS=0.42, Avg=0.65）是最佳选择。

---

## 5. 对其他任务的 Generalization

### 5.1 Text-to-Image（Table 8, 20）

设置：
- JourneyDB + BLIP3o long/short caption 预训练 150K iter，batch 1024
- BLIP3o-60k finetune 50 epochs
- Text encoder：Qwen3-0.6B
- RAE encoder：SiGLIP2-B

结果：

| Method | Pretraining GenEval | Pretraining DPG | FT GenEval | FT DPG |
|--------|---------------------|-----------------|------------|--------|
| Flux-VAE | 41.7 | 77.6 | 78.3 | 79.2 |
| RAE | 58.4 | 80.1 | 81.5 | 80.6 |
| **RAEv2** | **62.4** | **81.7** | **82.7** | **82.3** |

注意 RAEv2 finetune 后 GenEval 82.7 已经接近 SOTA T2I models，仅用 0.9B 参数模型。

参考链接：
- JourneyDB: https://arxiv.org/abs/2307.13767
- BLIP3o: https://arxiv.org/abs/2505.09568
- GenEval: https://arxiv.org/abs/2310.11525

### 5.2 Navigation World Models（Table 9, 21）

设置：
- Backbone：DiTDH-XL
- 4 个过去 frames（256×256）作为 context，每个 frame 编码成 16×16=256 tokens，共 1024 context tokens
- 4 个 action tokens（Δx, Δy, Δψ）+ 1 个 Fourier-embedded time token
- Total 1029 conditioning tokens
- Dataset：RECON
- Training：100K iter，batch 256
- Evaluation：autoregressive rollout 1/2/4/8/16 秒

FVD 结果：

| Method | FVD↓ |
|--------|------|
| DIAMOND | 762.73 |
| NWM | 200.97 |
| RAE | 312.01 |
| **RAEv2-NWM** | **105.61** |

这个结果非常震撼——RAEv2-NWM 比 NWM 提升 2×。作者指出大量增益来自 generalized RAE（K>1），因为早期层保留了 low-level texture 和 geometry，对 temporally consistent rollout 至关重要。

Figure 13 的可视化很有意思：RAE 会在连续帧之间产生 flickering（比如窗户数量在相邻帧变化），RAEv2 保留了 scene structure。这个观察在 video generation 领域是经典问题——latent space 的 "flicker" 通常源于缺乏 low-level consistency。

参考链接：
- NWM: https://arxiv.org/abs/2412.03572
- DIAMOND: https://arxiv.org/abs/2412.14131
- RECON dataset: https://arxiv.org/abs/2310.07896

---

## 6. 直觉与联想

### 6.1 为什么 simple addition 在 vision representation 上这么有效？

这让我想到几条相关的 thread：

1. **Feature pyramid networks (FPN)** 的 multi-level fusion 也是用 element-wise addition，背后的直觉类似——不同 level 的 features 在 channel 维度上接近正交，加法等于"soft concatenation without cost"

2. **CLIP / SigLIP 的 multi-layer ensemble**：一些工作发现对最后几层做 weighted sum 能提升 downstream 性能

3. **High-dimensional geometry**：在 1024 维这种 high-dim space，两个 random vectors 几乎正交，所以 addition 不会产生 interference。但 semantically 相关的 layers（比如 DINOv3 的最后 K 层都是 high-level semantic）会有一定 alignment，相加相当于 "blend" 不同 abstraction level

4. **DINOv2/DINOv3 的最后一层信息密度**：self-supervised model 的 last layer 通常被训练成高度 invariant，这破坏了 spatial details。倒数几层保留了更多 spatial context，加进去能 "rescue" 这些细节

### 6.2 RAE 和 REPA 的互补性意味着什么？

这是一个非常 deep 的 finding。让我尝试 formulate 一个直觉：

- **Diffusion transformer 需要解决两个 problem**：
  - "What to generate"（global semantic）——这由 conditioning signal 提供
  - "How to spatially arrange it"（token-token structure）——这需要从 data 中 learn

- RAE 直接把 "what" 嵌入 latent space，所以 diffusion 不需要重新 discover 语义
- REPA 通过 alignment loss 告诉 intermediate features "如何 attend to each other"，提供 spatial scaffolding

这两个 axes 几乎 orthogonal，所以同时用没有 redundancy。这也解释了为什么 LP 和 LDS 是 uncorrelated 的 encoder property——它们 measure encoder 的两个独立维度。

### 6.3 x-prediction 和 self-guidance 的深层含义

这个 insight 让我觉得非常巧妙：

- **CFG 的本质**：用一个 weak baseline（unconditional model）来 estimate "prior direction"，full model 偏离这个 prior 的方向就是 "signal"
- **AutoGuidance**：把 unconditional 换成 weak conditional model，效果更好（因为 weak conditional 已经包含了大部分 prior signal，对比更精细）
- **RAEv2**：REPA head 是一个 "structural weak learner"——它只看 early-layer features，看到的是 "大致的 semantic + 部分 spatial structure"，而 full model 看到的是 "refined everything"

这本质上类似于 **bootstrap aggregation 的思想**——一个 model 和它的 "weakened version" 的差值就是这个 model 的 "specific" 信息。在 RAE latent space 中，这个 specific 信息正是 guidance 想要 amplify 的。

参考链接：
- AutoGuidance: https://arxiv.org/abs/2410.02391
- CFG: https://arxiv.org/abs/2207.12598
- Internal dynamics guidance: https://arxiv.org/abs/2512.24176

### 6.4 与 LLM speedrun 的类比

EPFID@k 这个指标启发自 modded-nanogpt speedrun（Keller Jordan 等人）。在 LLM 领域，大家发现 incremental benchmark 提升很难衡量 practical training efficiency，所以转向 "达到某个 loss threshold 的时间"。这篇 paper 把同样的哲学引入 diffusion：

- gFID 从 1.13 提升到 1.06 几乎没意义（噪声级别）
- 但 EPFID@2 从 177 降到 35 意味着 "practical usability threshold" 提前 5×

这种思维方式可能改变 future diffusion model 评估 paradigm。

### 6.5 Unified Tokenization 的 implications

如果 RAEv2 真的能在同一 latent space 上同时做好 understanding（LP=85.10）和 generation（gFID=1.06），这暗示了一个 unified architecture 的可能：

- Vision encoder + diffusion model + decoder 共享同一 representation
- LLM 用的 visual token 就是 diffusion 用的 latent token
- Any-to-any generation（text-image-video-action）可能在一个 token space 内完成

这和 JanusFlow [38]、BLIP3o 等工作的方向一致，但 RAEv2 提供了一个 training-free 的简化路径——不需要重新训练 encoder。

参考链接：
- JanusFlow: https://arxiv.org/abs/2411.18220
- Scaling T2I with RAE: https://arxiv.org/abs/2601.16208

### 6.6 Limitations 与未来方向

作者在 Section E 提到几个 limitation：

1. 只用 simple addition 和 random projection，没有探索 learnable aggregation（如 attention pooling、gated sum）
2. Encoder 选择仍靠 empirical search
3. 没有尝试 end-to-end optimize encoder（如 REPA-E [33]）

潜在的相关方向：

- **Learnable K per layer**：每层有一个 learnable scalar weight，让模型自己决定融合比例
- **Cross-encoder fusion**：用多个 encoder（DINOv3 + SigLIP）的不同层
- **Hierarchical diffusion**：不同 denoising stage 用不同 K（前期用大 K 恢复 spatial details，后期用小 K 关注 semantic refinement）

### 6.7 关于 "REPA is x-prediction" 的更深思考

让我再深挖一下这个 observation。在 RAE 设定下：
- Clean latent $x = E(I)$ 是 encoder 输出
- DiT 在每个 timestep 预测 $\hat{x}_\theta(x_t, t, c)$（如果用 x-prediction）
- REPA 在第 8 层加一个 head：$h_\phi(h_8)$，要求它预测 $x$

这相当于在 transformer 中间加了一个 shortcut，但 shortcut 的 target 是 input latent。这种 shortcut 有两个作用：
1. **Regularizer**：让中间 features 包含 encoder representation 的 information
2. **Weak predictor**：由于只用第 8 层 features（vs. 第 28 层的 full prediction），它是一个 "lossy" 预测

这两个作用的 duality 是 RAEv2 的精髓。在 standard DiT 中，中间层 supervision 早就被尝试过（deep supervision），但只有在 RAE 框架下，supervision target 同时是 latent，supervision head 才能"免费"作为 guidance baseline。这个 "lucky coincidence" 让 self-guidance 变得自然。

### 6.8 可能的 failure modes

虽然 paper 没讨论，我推测一些可能的 limitation：

1. **K 太大时的 generation degradation**：K=23 时 unguided gFID 是 3.01（比 K=1 的 1.50 差），说明太深的 representation 让 diffusion "harder to denoise"。可能是因为 low-level details 增加了 noise level 的 entropy
2. **Encoder 选择 overfitting**：27 个 encoder 上做 ablation 选择了 DINOv3-L，可能在其他 dataset 上不是最优
3. **Self-REPA 的 capacity overlap**：用同一个 encoder 做 RAE 和 REPA target 可能让 representation 过于 "easy to predict"，限制 diffusion 的 expressiveness

---

## 7. 总结

RAEv2 的三个 insight 可以一句话概括：

1. **Generalized encoder**：用最后 K 层 sum 代替 final layer，training-free，Pareto-optimal recon-gen
2. **RAE + REPA complementary**：semantic（RAE）+ spatial structure（REPA）orthogonal，combination is super-additive
3. **REPA as x-prediction**：reformulate REPA head 为 guidance signal，省去 separate model 和 extra forward pass

合起来达到 10× convergence speedup，gFID 1.06 in 80 epochs，FD^r 2.17，state-of-the-art on ImageNet-256 / RECON navigation world models / competitive T2I。

这篇 paper 的核心 contribution 不在 architecture（依然用 DiTDH-XL），不在新 loss（依然 flow matching），而在 **recipe engineering**——通过系统性的 ablation 和 mechanism understanding 找到最优 training recipe。这种"回到 basics，找到 simple 但 effective 的组合"的风格让我想到 ResNet、ViT 时代的精神。作者自己也说希望这能 "provide useful insights for practical adoption"——这正是一个 good baseline paper 的价值。

参考链接汇总：
- 主 paper: https://arxiv.org/abs/2510.11690 (RAEv2)
- 原始 RAE: https://arxiv.org/abs/2510.11690
- REPA: https://arxiv.org/abs/2410.06940
- iREPA: https://arxiv.org/abs/2512.10794
- DINOv3: https://arxiv.org/abs/2508.10104
- DINOv2: https://arxiv.org/abs/2304.07193
- DiT: https://arxiv.org/abs/2212.09748
- SiT: https://arxiv.org/abs/2410.02113
- REPA-E: https://arxiv.org/abs/2504.10483
- AutoGuidance: https://arxiv.org/abs/2410.02391
- Internal dynamics guidance: https://arxiv.org/abs/2512.24176
- NWM: https://arxiv.org/abs/2412.03572
- Perception Encoder: https://arxiv.org/abs/2504.13181
- BLIP3o: https://arxiv.org/abs/2505.09568
- FD^r: https://arxiv.org/abs/2604.28190
- modded-nanogpt speedrun: https://github.com/KellerJordan/modded-nanogpt
- Back to basics (x-pred): https://arxiv.org/abs/2511.13720
- WebSSL: https://arxiv.org/abs/2504.01017
