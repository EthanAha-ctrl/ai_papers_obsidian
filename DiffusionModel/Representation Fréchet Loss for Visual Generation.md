---
source_pdf: Representation Fréchet Loss for Visual Generation.pdf
paper_sha256: eadbfa2f5e4431dab81fa20857bfc54c8ba9e97704e6048757503c5e1d5f1529
processed_at: '2026-08-11T22:49:36-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

## 一句话总结

十年来大家都在用 FID 评价生成模型好坏，但从来没人敢直接拿它当 training loss 用——因为算 FID 需要 5 万张图，反传梯度会爆炸。这篇 paper 说：**别傻了，统计量用 5 万张算，梯度只走当前 batch 就行了**。结果发现这个 trick 不但能 fine-tune 出更强的 generator，还能把一个 multi-step 模型直接变成 one-step 的，不需要 distillation 也不需要 adversarial training。

---

## 这事为什么之前没人做成

FID 这个 metric 大家都熟。它本质上就干一件事：把 real 图和 fake 图都塞进 Inception-v3 网络提特征，假设两组特征都服从 multivariate Gaussian，然后算这两个 Gaussian 之间的 2-Wasserstein distance。

公式长这样：

$$
\text{FD}_\phi(\mathcal{R}, \mathcal{G}) = \|\mu_r - \mu_g\|_2^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)
$$

- $\mu_r, \mu_g$：real 和 fake 在特征空间的均值向量
- $\Sigma_r, \Sigma_g$：对应的协方差矩阵，$\Sigma \in \mathbb{R}^{2048 \times 2048}$（Inception 特征是 2048 维）
- $\|\cdot\|_2^2$：L2 范数平方
- $\text{Tr}(\cdot)$：矩阵的 trace
- $(\Sigma_r \Sigma_g)^{1/2}$：矩阵乘积的平方根

这公式里每一项都是 differentiable 的，理论上完全可以当 loss 用。那为什么之前没人做？

**核心障碍是 covariance 估计需要大 sample**。要稳定估计一个 $2048 \times 2048$ 的 full-rank 协方差矩阵，至少要几千张图，实际操作中大家用 5 万张。但你的 training batch 通常就 1024，拿 1024 张图估出来的 $\Sigma_g$ 噪声极大，gradient 全是噪声，模型反而越训越烂。

之前有人试过用 batch-wise FD 当 loss（[Mathiasen & Hvilshøj, 2020](https://arxiv.org/abs/2009.14075)），结果就是模型退化。这篇 paper 的 Table 1a 第一行也复现了这一点：batch-only（queue size = 0k）FID 从 3.31 恶化到 3.84。

所以整个社区默认 "FID 当 evaluator 可以，当 loss 不行"。

---

## Paper 的 key trick

**Decouple**：统计量用大窗口算，gradient 只走当前 batch。

具体两种实现：

### 1. Queue 版

维护一个大小 $N$ 的 feature queue（比如 $N = 50{,}000$）。每 iteration 生成 $B = 1024$ 张图，把它们的 features 塞进 queue，弹掉最老的 1024 个。然后用整个 queue 算 mean 和 covariance，再算 FD。反传时只对当前 batch 的 features 留梯度，queue 里历史的 features 全部 detach 当常数。

这跟 [MoCo](https://arxiv.org/abs/1911.05722) 的 queue 思路一模一样——contrastive learning 里也是需要大量 negative sample 但 batch 放不下，就用 queue 解决。

### 2. EMA 版（更优雅）

干脆不存 features，只存 first moment 和 second moment 的 EMA。

每 step 先算当前 batch 的 moments：

$$
\mu_{\text{batch}}^{(t)} = \frac{1}{B}\sum_{i=1}^B \phi(\hat{\mathbf{x}}_i)
$$

$$
\mathbf{M}_{\text{batch}}^{(t)} = \frac{1}{B}\sum_{i=1}^B \phi(\hat{\mathbf{x}}_i)\phi(\hat{\mathbf{x}}_i)^\top
$$

- $\mu_{\text{batch}}^{(t)} \in \mathbb{R}^d$：当前 batch 特征均值
- $\mathbf{M}_{\text{batch}}^{(t)} \in \mathbb{R}^{d \times d}$：当前 batch 特征的 second moment $E[xx^\top]$（未中心化）
- $B$：batch size
- $\phi(\hat{\mathbf{x}}_i)$：第 $i$ 张生成图的 feature
- $t$：iteration index

然后 EMA 更新：

$$
\mu_g^{(t)} = \beta \mu_g^{(t-1)} + (1-\beta) \mu_{\text{batch}}^{(t)}
$$

$$
\mathbf{M}_g^{(t)} = \beta \mathbf{M}_g^{(t-1)} + (1-\beta) \mathbf{M}_{\text{batch}}^{(t)}
$$

- $\beta \in (0, 1)$：EMA decay rate
- $\mu_g^{(t)}, \mathbf{M}_g^{(t)}$：第 $t$ step 的 running estimate
- 反传时上一 step 的 $\mu_g^{(t-1)}, \mathbf{M}_g^{(t-1)}$ 需要 detach

最后用 $\Sigma_g = \mathbf{M}_g - \mu_g \mu_g^\top$ 还原协方差（标准的 $\text{Cov}(X) = E[XX^\top] - E[X]E[X]^\top$）。

EMA 的 effective population 大约是 $\frac{1}{1-\beta}$。$\beta = 0.999$ 对应 effective 1000 个 sample 的加权平均，但因为 EMA 天然 upweight recent sample，比简单的 1000 sample 平均更 on-policy 也更稳定。

**EMA 比 queue 好在哪**：不用存 features，只存两个矩阵（mean 是 $d$ 维向量，second moment 是 $d \times d$ 矩阵）。对 $d = 2048$ 来说总共也就 8.4M 个 number，几 MB 内存。而且更 on-policy，效果还略好（FID 0.81 vs queue 的 0.89，见 Table 1b vs 1a）。

---

## 还有几个工程细节值得提

### Matrix square root 的高效计算

FD 里的 $\text{Tr}((\Sigma_r \Sigma_g)^{1/2})$ 涉及矩阵平方根，直接算很贵。Paper 用了一个 trace 的 cyclic property trick：

$$
\text{Tr}((\Sigma_r \Sigma_g)^{1/2}) = \text{Tr}((\Sigma_r^{1/2} \Sigma_g \Sigma_r^{1/2})^{1/2})
$$

设 $A = \Sigma_r^{1/2} \Sigma_g \Sigma_r^{1/2}$，这是 symmetric PSD 矩阵，eigenvalues $\lambda_1, \dots, \lambda_d \geq 0$。那么 $\text{Tr}(A^{1/2}) = \sum_i \sqrt{\lambda_i}$。

$\Sigma_r^{1/2}$ 只需算一次（real statistics 固定），用 eigendecomposition 预存。训练时每 step 只需要 `torch.linalg.eigvalsh` 算 $A$ 的 eigenvalues（不算 eigenvectors），取 $\sqrt{\lambda}$ 求和。这个函数比 `eigvals` 快很多，paper 里有提到。

### Multi-representation loss normalization

不同 representation 的 FD 数值差异巨大。MAE feature 空间 FD 在 0.04 量级，DINOv2 在 14 量级，CLIP 在 5.6 量级（看 Table F.7）。直接相加会让数值大的 dominate。Paper 用：

$$
\mathcal{L}_{\phi_i} = \frac{\text{FD}_{\phi_i}(\mathcal{R}, \mathcal{G})}{\text{sg}(\text{FD}_{\phi_i}(\mathcal{R}, \mathcal{G})) + c}
$$

- $\text{sg}(\cdot)$：stop-gradient，分子有梯度，分母 detach
- $c = 0.01$：避免除零
- 这样每项 loss 都自动 rescale 到 $O(1)$ 附近

这个 `loss / loss.detach()` trick 在 RLHF 里也常用，叫 self-normalization。

**Default 组合 "SIM"** = SigLIP2 + Inception + MAE，覆盖 vision-language / supervised / self-supervised reconstructive 三种 pretraining 目标。

---

## 实验结果中最有意思的几个发现

### 发现 1：FD-loss 是强 post-training objective

pMF-H 这个 one-step generator，post-train 100 epochs 后：

| 指标 | Base | + FD-Inception | + FD-SIM |
|------|------|----------------|----------|
| FID ↓ | 2.29 | 0.72 | 0.77 |
| FDr^6 ↓ | 6.87 | 4.86 | **1.89** |
| IS ↑ | 267.2 | 298.8 | 310.1 |

FID 0.72 在 ImageNet 256×256 上是当时 SOTA。

### 发现 2：FID 已经 saturated，会 misrank quality

Table 1c 是 paper 里最 thought-provoking 的实验。用不同 representation 当 loss 训 pMF-B：

| Loss | FID ↓ | FDr^6 ↓ |
|------|-------|---------|
| Base | 3.31 | 13.70 |
| FD-Inception | **0.81** | 10.81 |
| FD-ConvNeXt | 1.64 | 8.46 |
| FD-DINOv2 | 4.89 | 8.47 |
| FD-MAE | 6.42 | 6.63 |
| FD-SigLIP | 7.71 | 5.85 |
| FD-SIM | 0.94 | **4.20** |

**奇怪的事**：用 MAE 或 SigLIP 当 loss，FID 反而恶化（从 3.31 到 6.42、7.71），但 FDr^6 大幅改善（13.70 到 6.63、5.85）。视觉上看 Figure 4，用 modern representation 优化的模型 object structure 明显比 Inception-optimized 模型好，尽管 FID 高得多。

更夸张的：FD-Inception 能让 FDr 在 Inception 那一列降到 0.48，**低于 1.0**——意思是生成图像在 Inception 特征空间下比 real validation 还 "靠近" training set。这明显是 reward hacking 的征兆。

Appendix A 的 stress test 直接 demo 了这一点：用 100× learning rate 狂优化 Inception，能得到 IS 660、FID 2.09，但 FDr^6 = 50.66，图像明显 artifact（Figure A.1）。这就是 Goodhart's law 的 visual demo。

### 发现 3：FID 说 ImageNet 已 solved，FDr^6 说远未 solved

Figure 3 显示一个 paradox：

- **左图 FID**：近年来 MAR-H、LightningDiT、REPA-E 等 SOTA 已经做到 FID 1.4–1.6，**低于 ImageNet validation set 自己的 FID 1.68**。按这个 metric，ImageNet 生成已经 "solved"；
- **右图 FDr^6**：SOTA 还在 3–6 区间，validation set 是 1.0，**离 real 还差 3-6 倍**。

这就是 FDr^k metric 的动机。它的定义：

$$
\text{FDr}_{\phi_i}(\mathcal{G}) = \frac{\text{FD}_{\phi_i}(\mathcal{G}, \mathcal{T})}{\text{FD}_{\phi_i}(\mathcal{V}, \mathcal{T})}
$$

$$
\text{FDr}^K(\mathcal{G}) = \frac{1}{K}\sum_{i=1}^K \text{FDr}_{\phi_i}(\mathcal{G})
$$

- $\mathcal{T}$：ImageNet training set
- $\mathcal{V}$：ImageNet validation set
- $\mathcal{G}$：generated images
- $\phi_i$：第 $i$ 个 representation model
- $K$：representation 总数，论文用 6

**直观含义**：生成图像在 $\phi_i$ 空间下距离 training set 是 validation set 的多少倍。Validation set 自己 FDr = 1.0。FDr = 2.0 意思是生成图像感知距离是 real validation 的 2 倍。

6 个 representation 覆盖：Inception-v3（supervised CNN）、ConvNeXt-v2（self-supervised CNN）、DINOv2（contrastive ViT）、MAE（reconstructive ViT）、SigLIP2（vision-language ViT）、CLIP（vision-language ViT）。Architecture 和 training objective 都多样化。

### 发现 4：FD-loss 能把 multi-step 模型直接 repurpose 成 one-step

JiT-L/16 原本是 50-step denoising model。Naive 用 1-NFE 调用直接 collapse（FID 291.59）。但 FD-loss post-train 50 epochs 后：

| Setting | NFE | FID ↓ | FDr^6 ↓ |
|---------|-----|-------|---------|
| JiT-L (50-step) | 200 | 2.59 | 10.73 |
| JiT-L (1-step, naive) | 1 | 291.59 | 214.75 |
| + FD-Incep. | 1 | 0.77 | 12.86 |
| + FD-SIM | 1 | 0.85 | **3.29** |

**这事很神奇**：没有任何 teacher distillation，没有 adversarial loss，没有 per-sample target。就一个 FD-loss，硬生生把一个从没学过 one-step 的 multi-step 模型变成了 FID 0.85 的 one-step generator。

这暗示 one-step generation 这个问题本质上就是 distribution matching——你不需要 "教" 模型怎么做单步预测，只需要让它的 output distribution 匹配 real distribution。

对比一下其他 1-NFE 路线：
- [Consistency Models](https://arxiv.org/abs/2303.01469)：需要 teacher score function
- [DMD](https://arxiv.org/abs/2311.18828)：需要 teacher + adversarial
- [Score Identity Distillation](https://arxiv.org/abs/2404.04057)：需要 teacher score

FD-loss 完全不需要 teacher，只需要 pre-computed statistics $(\mu_r, \Sigma_r)$。这跟 LLM 里的 RLHF 形成有趣对照——RLHF 需要 reward model，FD-loss 用 frozen representation model 当 "reward model"。

### 发现 5：Text-to-Image 也能这么搞

Section 4.5 把 SD3.5 Medium（2.5B params 的 MMDiT）repurpose 成 1-NFE text-to-image generator。原本 28 step × 2 CFG = 56 NFE，post-train 后 1 NFE，**56× 速度提升**。

训练细节（Table B.3）：
- 15k iterations，batch 1024，lr 1e-5
- EMA β = 0.999，warm-start 50k base model samples
- Reference statistics 从 BLIP3o 数据集（3M 或 60k 子集）算
- 同样用 SIM (SigLIP2 + Inception + MAE)

Figure G.1 显示 1-NFE 模型仍能保留 prompt 内容，且 reference distribution 会 "shape" 生成的 aesthetic——用 photographic 数据 train 就偏写实，用 GPT-4o stylized 数据 train 就偏插画风。这暗示 FD-loss 不只是改善 quality，还能 transfer reference 的 style distribution。

---

## 这个 work 在更大 context 里的位置

### 跟 distribution matching 历史的关系

Distribution-level objective 不是新概念。GAN ([Goodfellow et al., 2014](https://arxiv.org/abs/1406.2661)) 本质就是 distribution matching，用 discriminator 区分 real 和 fake。MMD-based generator（[GMMN, ICML 2015](https://arxiv.org/abs/1502.02761)；[MMD GAN, ICLR 2018](https://arxiv.org/abs/1801.01401)）用 kernel trick 直接 match distribution。Moment matching GAN（[MCGAN, ICML 2017](https://arxiv.org/abs/1702.04159)）match mean + covariance。Sliced Wasserstein（[CVPR 2018](https://arxiv.org/abs/1710.07075)）用投影简化 Wasserstein。

但 prior work 都在 batch 内估计 distribution，被高维 covariance 噪声淹没。FD-loss 的 contribution 是 **decoupling** 这个 trick，让 distribution-level objective 在 batch = 1024 的情况下也能稳定优化。

### 跟 contrastive learning 的同构

MoCo ([He et al., CVPR 2020](https://arxiv.org/abs/1911.05722)) 解决 contrastive learning 的 negative sample 数量问题，用 momentum encoder + queue。FD-loss 用 EMA + queue 解决 distribution estimation 的 sample 数量问题。**两者本质都是 decoupling estimation scale from optimization scale**。

### 跟 RLHF 的类比

Paper 在 Appendix A.1 自己点出了这个类比。FD-loss 在概念上类似 RLHF：
- Pre-training：next-token prediction / denoising
- Post-training：reward model optimization / FD-loss optimization
- 都面临 reward hacking / Goodhart's law 问题
- 都用 ensemble（multi-representation SIM / reward model ensemble）来 mitigate

这跟 LLM 里的 [reward model over-optimization](https://arxiv.org/abs/2210.10760) 是同构问题。Figure A.1 的 stress test 就是 reward hacking 的 visual demonstration。

### 跟 Goodhart's law 的关系

[Goodhart's law](https://en.wikipedia.org/wiki/Goodhart%27s_law) 说 "当一个 measure 变成 target 时，它就不再是好的 measure"。FID 被 community implicit gradient descent 近十年，已经 over-fit 到 Inception feature 的 blind spot。这篇 paper 用 FDr^k 来纠正这个 bias，跟 [Jayasumana et al., CVPR 2024](https://arxiv.org/abs/2401.09603) "Rethinking FID"、[Kynkäänniemi et al., ICLR 2023](https://arxiv.org/abs/2201.08269) "Role of ImageNet in FID"、[Stein et al., NeurIPS 2023](https://arxiv.org/abs/2310.05170) "Exposing flaws of generative eval metrics" 这一系列批评 FID 的工作形成 constructive dialog——不光批评，还给替代方案。

---

## 用 Karpathy 的直觉框架理解

### Intuition 1：Simple objective > complex algorithm

你过去常说 "softmax regression 直接做 next-token prediction 就行，不需要复杂算法"。FD-loss 同样体现了这个哲学——它就是一个 FD + EMA，总共 30 行 code（Algorithm 1）。没有 adversarial training 的不稳定，没有 distillation 的复杂 pipeline，没有 per-sample regression target 的需求。Simple objective 选对了，复杂算法就不需要。

### Intuition 2：Optimizing the proxy breaks the proxy

你讲 RLHF 时强调过这个。FID 这个 proxy 被社区 implicit gradient descent 近十年，已经 break 了。Paper 的 stress test 直接 demo 这点——Inception 优化的极端版本 FID 看起来很好，图像实际 garbage。FDr^k 就是 multi-proxy ensemble 来 mitigate。

### Intuition 3：Distribution-level > sample-level for generation

主流生成模型都是 sample-level loss（diffusion 的 $\|\hat{x}_0 - x\|^2$、flow matching 的 velocity regression、distillation 的 per-sample match）。但 generation 本质是 distribution matching 问题。FD-loss 直接在 distribution level 优化，跳过 sample-level 的中间步骤。这跟 one-step generation 能 work 有深刻联系——你不需要 "教" 模型每一步怎么走，只需要让最终 output distribution 匹配 real。

### Intuition 4：Decoupling is a powerful engineering pattern

BatchNorm 用 running statistics decouple inference 和 training 的 statistics estimation。MoCo 用 queue decouple negative sample 数量和 batch size。FD-loss 用 EMA decouple population size 和 batch size。**这是 deep learning 里反复出现的 pattern**——当某个 estimation 需要大 sample 但 gradient 只需要小 batch 时，decoupling 就能解锁新的可能性。

### Intuition 5：Post-training 是新范式

Pre-training 用简单 objective（next-token / denoising）+ 大数据 + 大算力。Post-training 用 refined objective（reward model / FD-loss）+ 小数据 + 小算力。这是 LLM 和 visual generation 共同的范式。FD-loss paper 把这个范式拓展到 visual generation，且 distribution-level objective 比 sample-level 更自然适合 post-training。

---

## 几个 open question 我觉得值得思考

1. **Higher-order moment matching**：FD 只匹配 first two moments，假设 Gaussian。能不能用 MMD 或 sliced Wasserstein 同时匹配 higher moments？这跟 FD-loss 的 decoupling trick 完全兼容。
2. **Learned representation for FD**：能否 jointly learn representation 和 generator？类似 GAN discriminator 但 distribution-level。这会引入 adversarial 的不稳定，但可能解锁更强的 representation。
3. **Mode collapse 分析**：FD-loss 直接优化 distribution matching，理论上比 GAN 更不容易 mode collapse（因为 explicit 匹配 covariance），但 paper 没深入分析 diversity vs. fidelity trade-off。
4. **Beyond image**：3D / video / audio generation 的 FD-loss？需要对应的 representation model。DINOv3 with temporal、3D feature、audio feature 都是 candidate。
5. **FD-loss 跟 diffusion pre-training 的联合训练**：能否从头训练用 FD-loss，而不是只做 post-training？这会需要更稳定的 covariance 估计，但可能解锁新的 training paradigm。

---

## 总结

这篇 paper 真正的贡献有三层：

**Technical 层**：decouple population from batch 的 FD 优化框架，EMA 实现优雅且 memory-efficient，30 行代码搞定。

**Empirical 层**：pMF-H + FD-loss 在 ImageNet 256×256 上 FID 0.72（1-NFE SOTA），且能 repurpose multi-step JiT 和 SD3.5 Medium 成 1-NFE generator（56× 加速）。

**Conceptual 层**：揭示 FID 已 saturated，提出 FDr^k multi-representation metric，把 generation 从 sample-level loss 推向 distribution-level loss，跟 LLM post-training / RLHF 形成同构对照。

最美的部分还是 **simplicity**。Long-standing assumption（"FID 不能直接优化"）往往只是 practical 限制，principle 上没阻碍。Decoupling estimation scale from optimization scale 这个 pattern 可能还有大量 application 没被探索——任何需要 population-level statistics 但 batch-level gradient 的场景都值得重新审视。

Paper 链接和 code：https://github.com/Jiawei-Yang/FD-loss

---

# Representation Fréchet Loss for Visual Generation - 深度解析

## 1. 核心 insight: 重新定位 Fréchet Distance 在生成建模中的角色

这篇 paper 的核心 thesis 极其简洁——FID 长期以来只活在 evaluation 那一边，从未直接作为 training loss 使用。作者证明这其实是 **历史性的 practical 限制，而 principle 上没有任何阻碍**。FD 本身 fully differentiable，定义里没有任何东西限制它只能用于 evaluation。一旦把 "population scale for estimation" 和 "batch scale for gradient" decouple 开，FD 立刻就变成一个 scalable、effective 的 post-training objective。

这个 insight 一旦成立，paper 揭示出三个非常深刻的发现：

1. **FD-loss 是强 post-training objective**——pMF-H + FD-loss 在 ImageNet 256×256 上做到 FID 0.72；
2. **FD-loss 可以把 multi-step generator 直接 repurpose 成 one-step generator**——不需要 teacher distillation、不需要 adversarial training、不需要 per-sample regression targets；
3. **FID 本身已经 saturated 作为 quality signal**——它会 misrank visual quality，作者提出 FDr^k 这个 multi-representation metric 来纠正。

Repo: https://github.com/Jiawei-Yang/FD-loss

---

## 2. 为什么直接优化 FD "被认为不可能"

先看 Fréchet Distance 的定义。给定 feature extractor $\phi(\cdot)$，real images $\mathcal{R} = \{\mathbf{x}_i\}$ 和 generated images $\mathcal{G} = \{\hat{\mathbf{x}}_i\}$，它们在特征空间被建模为 multivariate Gaussians：

$$
\mu_r = \mathbb{E}[\phi(\mathbf{x})], \quad \Sigma_r = \text{Cov}[\phi(\mathbf{x})], \quad \mu_g = \mathbb{E}[\phi(\hat{\mathbf{x}})], \quad \Sigma_g = \text{Cov}[\phi(\hat{\mathbf{x}})]
$$

变量含义：
- $\mu_r, \mu_g \in \mathbb{R}^d$：real 和 generated features 的均值向量，$d$ 是 feature dimension（Inception 是 2048）
- $\Sigma_r, \Sigma_g \in \mathbb{R}^{d \times d}$：协方差矩阵
- $\mathbb{E}[\cdot]$：对集合取经验均值
- $\text{Cov}[\cdot]$：经验协方差

FD 公式（Eq. 2）：

$$
\text{FD}_\phi(\mathcal{R}, \mathcal{G}) = \|\mu_r - \mu_g\|_2^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)
$$

- $\|\mu_r - \mu_g\|_2^2$：均值差异的 squared L2 norm，捕获 first-order moment mismatch
- $\text{Tr}(\cdot)$：矩阵 trace
- $(\Sigma_r \Sigma_g)^{1/2}$：矩阵乘积的 square root（unique PSD square root）
- 当 $\phi$ = Inception-v3，这就是 FID

第二项 $\text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$ 等价于 **两个 Gaussian 之间的 2-Wasserstein distance 在 covariance 上的部分**，这就是 FD 的几何含义：它就是两个 multivariate Gaussian 之间的 2-Wasserstein distance 的闭式解。

**直接优化 FD 的两个核心障碍**：

1. **Population scale 障碍**：估计一个 full-rank 的 $\Sigma_g \in \mathbb{R}^{2048 \times 2048}$ 需要至少 2048 个独立 sample，但实际中需要 50k 才能稳定。一个 training batch（64–1024）远不够；
2. **梯度计算障碍**：如果要严格保证 on-policy，每 step 需要 backprop 通过 50k 个 sample，这计算上 prohibitive。

Table 1a 的 0k 那行直接验证了这一点：仅用 batch（1024 个 sample）估计 FD 并优化，FID 反而从 3.31 恶化到 3.84。这是因为高维 covariance 估计在小 sample 下噪声太大，gradient 被噪声主导。

---

## 3. FD-loss: Decoupling Principle

**核心 idea**：FD 估计用大窗口，梯度只走当前 batch。这是 contrastive learning 中 MoCo queue 和 BatchNorm EMA 思想的迁移。

### 3.1 Queue-based estimator

维护一个大小为 $N$ 的 feature queue（如 $N=100{,}000$）。每 iteration：

1. Generator 生成 $B$ 张 image（$B \ll N$，如 $B=1024$）；
2. 用 $\phi$ 提取 features，enqueue 这 $B$ 个，dequeue 最旧的 $B$ 个；
3. 用整个 queue 的 empirical mean 和 covariance 计算 FD；
4. **Backprop 只对当前 batch 的 features 流梯度**——queue 中之前 iteration 的 features 当作常数（detach）。

这个 trick 跟 MoCo 的 momentum encoder + queue 思想完全一致（参考 [He et al., MoCo, CVPR 2020](https://arxiv.org/abs/1911.05722)）：用大集合估计 distribution，但梯度只在 mini-batch 上流。

### 3.2 EMA-based estimator

更优雅的版本——不存 queue，只存 first 和 second moment 的 EMA。给定当前 batch features $\{\phi(\hat{\mathbf{x}}_i)\}_{i=1}^B$，定义 batch moments：

$$
\mu_{\text{batch}}^{(t)} = \frac{1}{B}\sum_{i=1}^B \phi(\hat{\mathbf{x}}_i), \quad \mathbf{M}_{\text{batch}}^{(t)} = \frac{1}{B}\sum_{i=1}^B \phi(\hat{\mathbf{x}}_i)\phi(\hat{\mathbf{x}}_i)^\top
$$

- $\mu_{\text{batch}}^{(t)} \in \mathbb{R}^d$：当前 batch 的 mean
- $\mathbf{M}_{\text{batch}}^{(t)} \in \mathbb{R}^{d \times d}$：当前 batch 的 second moment（**未中心化**的协方差，即 $E[xx^\top]$）

EMA 更新规则（Eq. 4）：

$$
\mu_g^{(t)} = \beta \mu_g^{(t-1)} + (1-\beta) \mu_{\text{batch}}^{(t)}, \quad \mathbf{M}_g^{(t)} = \beta \mathbf{M}_g^{(t-1)} + (1-\beta) \mathbf{M}_{\text{batch}}^{(t)}
$$

- $\beta \in (0, 1)$：EMA decay rate
- $\mu_g^{(t)}, \mathbf{M}_g^{(t)}$：第 $t$ step 的 running estimate
- $\mu_g^{(t-1)}, \mathbf{M}_g^{(t-1)}$：上一 step 的 estimate（在反传时需要 detach）

恢复 covariance（Eq. 5）：

$$
\Sigma_g^{(t)} = \mathbf{M}_g^{(t)} - \mu_g^{(t)} \mu_g^{(t)\top}
$$

这就是经典的 $\text{Cov}(X) = \mathbb{E}[XX^\top] - \mathbb{E}[X]\mathbb{E}[X]^\top$。

**为什么 EMA 比 queue 更优雅**：
- 不需要存 features，只存两个统计量矩阵（$d + d^2$ 个 number），对于 $d=2048$ 也就 8.4M 参数；
- EMA 自然 upweight recent samples，更 on-policy；
- 不会因为 queue 太大而变得 stale。

EMA 的 effective population size 大致是 $\frac{1}{1-\beta}$。所以 $\beta = 0.999$ 对应 effective population ≈ 1000，但实际上由于 EMA 是 weighted average，效果比简单的 1000 sample 平均更稳定。Table 1b 显示 $\beta = 0.999$ 最优（FID 0.81），$\beta = 0.9999$ 又开始退化（FID 0.98）——太大了反而 stale。

### 3.3 矩阵 square root 的高效计算

FD 里的 $\text{Tr}((\Sigma_r \Sigma_g)^{1/2})$ 需要矩阵 square root，这是 computational bottleneck。论文用了一个巧妙的 trick：

预计算 $\Sigma_r^{1/2}$（real statistics 是固定的，只需要一次），然后利用 trace 的循环性质：

$$
\text{Tr}((\Sigma_r \Sigma_g)^{1/2}) = \text{Tr}\left((\Sigma_r^{1/2} \Sigma_g \Sigma_r^{1/2})^{1/2}\right)
$$

设 $A = \Sigma_r^{1/2} \Sigma_g \Sigma_r^{1/2}$（这是 symmetric PSD），它的 eigenvalues $\lambda_1, \dots, \lambda_d$ 都非负。那么 $\text{Tr}(A^{1/2}) = \sum_i \sqrt{\lambda_i}$。

用 `torch.linalg.eigvalsh`（只算 eigenvalues，不算 eigenvectors，比 `eigvals` 快很多）直接拿到 $A$ 的 eigenvalues，再求 $\sqrt{\lambda}$ 之和。这避免了显式做 matrix square root。

---

## 4. Multi-Representation FD-Loss

不同 representation 的 FD 数值差异巨大（Table F.7 显示 MAE feature 空间下 FD 在 0.04 量级，DINOv2 在 14 量级，CLIP 在 5.6 量级）。直接相加没有意义。论文用 normalization（Eq. 6）：

$$
\mathcal{L} = \sum_i w_i \cdot \mathcal{L}_{\phi_i}, \quad \mathcal{L}_{\phi_i} = \frac{\text{FD}_{\phi_i}(\mathcal{R}, \mathcal{G})}{\text{sg}(\text{FD}_{\phi_i}(\mathcal{R}, \mathcal{G})) + c}
$$

- $w_i$：每个 representation 的权重（论文用 $w_i = 1$）
- $\text{sg}(\cdot)$：stop-gradient——分子用当前 FD 值，分母用 detach 的 FD 值。这让每个 loss 项大约在 unit scale 附近，避免某个 representation 因数值大就 dominate
- $c$：小常数（论文 $c = 0.01$）避免数值不稳定

这个 trick 类似于 `loss / loss.detach()`，相当于 self-normalization——gradient 大小被自动 rescale 到 $O(1)$，跟具体 representation 的 scale 解耦。这种 normalization 在 RLHF 的 reward model 训练中也常见。

**Default "SIM" 组合**：SigLIP2 + Inception + MAE。覆盖：
- **SigLIP2**：vision-language，捕获 semantic + perceptual
- **Inception**：传统 supervised，对 FID 友好
- **MAE**：self-supervised reconstructive，捕获 texture/structure

---

## 5. FDr^k Metric: 揭示 FID 的盲点

### 5.1 Metric paradox

Figure 3 展示了一个震撼现象：截至 2025 年，state-of-the-art generator 的 FID 已经 **低于 real validation set 的 FID**（validation set 自己 vs. training set 的 FID 是 1.68，但 MAR-H、LightningDiT 等已经达到 1.42、1.42）。直观上这暗示 "ImageNet generation 已经 solved"，但人类视觉判断明确否认这一点。

这是经典的 Goodhart's law 现象：当 Inception 这个 metric 被社区集体 "gradient descent" 近十年后，已经 over-fit 到它的特征空间，optimization 找到了 Inception feature space 的 reward hacking 路径。

### 5.2 FDr 定义

为了跨 representation 比较，论文定义 normalized FD ratio（Eq. 7）：

$$
\text{FDr}_{\phi_i}(\mathcal{G}) = \frac{\text{FD}_{\phi_i}(\mathcal{G}, \mathcal{T})}{\text{FD}_{\phi_i}(\mathcal{V}, \mathcal{T})}
$$

- $\mathcal{T}$：ImageNet training set
- $\mathcal{V}$：ImageNet validation set
- $\mathcal{G}$：generated images
- $\phi_i$：某个 representation model

**直观含义**：generated images 在 $\phi_i$ 空间下距离 training set 是 validation set 的多少倍。Validation set 自己的 FDr = 1.0。FDr = 2.0 意味着生成图像在 $\phi_i$ 空间下感知上比 validation 远 2 倍。

### 5.3 FDr^k

Average over $K$ 个 representation（Eq. 8）：

$$
\text{FDr}^K(\mathcal{G}) = \frac{1}{K}\sum_{i=1}^K \text{FDr}_{\phi_i}(\mathcal{G})
$$

论文用 6 个 representation：Inception-v3、ConvNeXt-v2、DINOv2、MAE、SigLIP2、CLIP。这覆盖：
- CNN + ViT 架构
- supervised + self-supervised + reconstructive + vision-language 训练目标

**关键数据对比**（Table 4 摘录）：

| Method | NFE | FDr^6 ↓ | FID ↓ |
|--------|-----|---------|-------|
| Validation set | N/A | 1.00 | 1.68 |
| MAR-H (multi-step, 256×2×100) | 51200 | 5.61 | **1.56** |
| LightningDiT-XL | 500 | 4.57 | 1.42 |
| REPA-E | 500 | 3.04 | 1.17 |
| RAE-XL | 100 | 3.26 | 1.16 |
| **pMF-H + FD-loss (SIM, 1-NFE)** | **1** | **1.89** | 0.77 |

注意 FID 上 REPA-E 比 MAR-H "更好"（1.17 vs. 1.56），但 FDr^6 上 MAR-H 反而更差（5.61 vs. 3.04）。这种 ranking 不一致就是 paper 要揭示的核心：**单一 representation（Inception）已经 saturated，无法反映真正的 perceptual quality**。

pMF-H + FD-loss 在 FDr^6 = 1.89 下，意味着在 6 个 representation 综合判断下，生成图像距离 training set 是 validation set 的 1.89 倍——已经很接近 real 了，但仍有 gap。这跟 human preference study（Figure 6）一致：FD-loss 后的 pMF-H 仍输给 real validation images。

---

## 6. 关键实验详解

### 6.1 Population size ablation (Table 1a, 1b)

**Setup**：pMF-B/16 + FD-Inception，50 epochs post-training。

| Queue size | FID ↓ | IS ↑ | FDr^6 ↓ |
|------------|-------|------|--------|
| Base (no post-train) | 3.31 | 254.6 | 13.70 |
| 0k (batch only) | 3.84 | 250.9 | 17.06 |
| 5k | 1.05 | 280.0 | 11.89 |
| 10k | 0.93 | 283.9 | 11.71 |
| **50k** | **0.89** | 288.3 | **10.91** |
| 100k | 0.93 | 288.8 | 11.15 |
| 500k | 1.22 | 294.4 | 17.67 |

**关键 insight**：
1. 0k（仅 batch statistics）：模型 **退化**。这证实了 batch-wise FD 优化不可行，正是 prior work 失败的原因（如 [Mathiasen & Hvilshøj, 2020](https://arxiv.org/abs/2009.14075)）；
2. 5k–100k 都有显著改善，50k 是 sweet spot；
3. 500k 反而恶化——queue 太大，统计量过于 stale，off-policy 问题 dominate；
4. **500k 那行揭示 FID 的盲点**：FID 还在改善（1.22 < 3.31），但 FDr^6 退化到 17.67（比 base 的 13.70 还差）。这意味着 Inception feature space 下能 hack，但其他 representation 看不下去了。

| β (EMA decay) | FID ↓ | IS ↑ | FDr^6 ↓ |
|---------------|-------|------|--------|
| Base | 3.31 | 254.6 | 13.70 |
| 0.0 (= batch only) | 3.84 | 250.9 | 17.06 |
| 0.9 | 0.98 | 283.6 | 11.19 |
| 0.99 | 0.84 | 291.8 | 10.74 |
| **0.999** | **0.81** | 294.5 | **10.81** |
| 0.9999 | 0.98 | 287.7 | 11.63 |

EMA 比 queue 略好（0.81 vs. 0.89），且不需要存 feature buffer，更 scalable——后续所有实验都用 EMA, β=0.999。

### 6.2 Representation model ablation (Table 1c)

这是 paper 最 thought-provoking 的 table：

| Loss | FID ↓ | FDr^6 ↓ |
|------|-------|--------|
| Base | 3.31 | 13.70 |
| FD-Inception | **0.81** | 10.81 |
| FD-ConvNeXt | 1.64 | 8.46 |
| FD-DINOv2 | 4.89 | 8.47 |
| FD-MAE | 6.42 | 6.63 |
| FD-SigLIP | 7.71 | 5.85 |
| FD-SigLIP+Incep. | 0.89 | 5.95 |
| **FD-SIM** (SigLIP+Incep.+MAE) | 0.94 | **4.20** |

**关键 insight**：
1. **对角线现象**：每个 representation 在自己的 feature space 下都最优（on-diagonal）。这本身是 trivial 的——你优化什么 metric，那个 metric 就降下来；
2. **Off-diagonal 揭示 trade-off**：
   - Inception 优化能压低 FID，但 FDr^6 改善有限（10.81）；
   - MAE / SigLIP 优化让 FID 反而恶化（6.42, 7.71），但 FDr^6 大幅改善（6.63, 5.85）；
   - **Inception-optimal 模型 ≠ perceptual optimal 模型**——Figure 4 视觉对比直接验证这一点，Inception-optimized 模型 FID 最低但 object structure 不如 modern representation 优化的；
3. **CNN vs. ViT 倾向**：ConvNeXt（现代 CNN）介于 Inception 和 ViT 之间。CNN-based representation 倾向改善 FID，ViT-based 倾向改善 FDr^6；
4. **Saturation 现象**：FD-Inception 能让 FDr（Inception 那列）降到 0.48，**低于 1.0**——意味着生成图像在 Inception 空间下比 real validation 还 "靠近" training set。这强烈暗示 Inception 已被 hack。

**SIM 组合**效果最好——兼顾 FID（0.94，几乎不恶化）和 FDr^6（4.20，最佳）。

### 6.3 Repurposing multi-step generator (Table 2)

**Setup**：JiT-L/16（原本 50-step denoising model），强行用 1-NFE 调用（terminal step 直接 predict $\hat{x}_0$），然后 FD-loss post-train。

| Setting | NFE | FID ↓ | FDr^6 ↓ |
|---------|-----|-------|--------|
| JiT-L (50-step) | 200 | 2.59 | 10.73 |
| JiT-L (1-step, naive) | 1 | 291.59 | 214.75 |
| + FD-Incep. | 1 | 0.77 | 12.86 |
| + FD-MAE | 1 | 6.52 | 9.30 |
| + FD-SigLIP | 1 | 5.10 | 9.04 |
| + FD-SigLIP+MAE | 1 | 4.67 | 3.83 |
| **+ FD-SIM** | 1 | **0.85** | **3.29** |

**震撼点**：
1. Naive 1-step 调用直接 collapse（FID 291.59）——multi-step model 从未被训练做 1-step prediction；
2. FD-loss 单独优化（无 distillation、无 adversarial、无 per-sample target）直接把 1-step FID 拉到 0.85，FDr^6 拉到 3.29；
3. **FD-Inception 单独在 repurposing 场景下 insufficient**——FID 0.77 看似最好，但 FDr^6 12.86（基本没改善）。需要 modern representation 才能产生视觉可接受的 sample；
4. 这暗示 FD-loss 是真正的 **distribution-matching objective**——它在 distribution level 把 generated 推向 real，自然就学到了 one-step mapping，不需要 explicit distillation。

这跟 consistency models ([Song et al., ICML 2023](https://arxiv.org/abs/2303.01469))、distribution matching distillation ([Yin et al., CVPR 2024](https://arxiv.org/abs/2311.18828))、score identity distillation ([Zhou et al., ICML 2024](https://arxiv.org/abs/2404.04057)) 等路线不同——FD-loss 完全不用 teacher，不用 score function，只用 pre-computed statistics。

### 6.4 Scalability (Table 3)

FD-loss 在所有 settings 下都 work：

| Family | Size | Base FDr^6 / FID | +SIM FDr^6 / FID | +Incep. FDr^6 / FID |
|--------|------|------------------|------------------|----------------------|
| pMF (pixel) | B | 13.70 / 3.31 | 3.50 / 0.85 | 10.66 / 0.77 |
| pMF | L | 9.09 / 2.72 | 2.09 / 0.78 | 6.19 / 0.73 |
| pMF | H | 6.87 / 2.29 | 1.89 / 0.77 | 4.86 / 0.72 |
| iMF (latent) | B | 15.29 / 3.45 | 5.56 / 0.88 | 11.34 / 0.79 |
| iMF | L | 9.06 / 1.93 | 2.74 / 0.79 | 6.63 / 0.75 |
| iMF | XL | 8.39 / 1.82 | 2.45 / 0.76 | 6.01 / 0.72 |
| JiT (pixel, repurposed) | B | 15.65 / 3.71 | 5.53 / 1.00 | 22.48 / 0.76 |
| JiT | L | 10.73 / 2.59 | 3.24 / 0.77 | 12.75 / 0.73 |
| JiT | H | 7.66 / 1.97 | 2.65 / 0.75 | 10.18 / 0.72 |

观察：
- **Same hyperparameter 跨 family / size / resolution 都 work**——说明 FD-loss 是非常 robust 的 objective；
- **Larger model 起点更好，FD-loss 增益持续**；
- **JiT-B + Inception 是少数 FDr^6 反而恶化的 case**（22.48 > 15.65），但 +SIM 仍然改善（5.53 < 15.65）。这强化了 SIM 的必要性。

### 6.5 Text-to-Image 上的扩展 (Section 4.5, Figure 7, G.1)

把 SD3.5 Medium（2.5B params 的 MMDiT）repurpose 成 1-NFE text-to-image generator。**56× NFE reduction**（原本 28 step × 2 CFG = 56 NFE → 1 NFE）。

Training：
- Reference statistics 从 BLIP3o-Pretrain-Long-3M 或 BLIP3o-GPT4o-60k 计算
- 15k iterations，batch 1024，lr 1e-5
- 同样 SIM (SigLIP2 + Inception + MAE)

观察：
- 即使 56× 速度提升，post-trained 1-NFE 模型仍保留 prompt 内容；
- Reference distribution 影响 aesthetic——3M photographic 数据 → photorealistic；60k GPT-4o stylized 数据 → illustration 风格；
- 这显示 FD-loss 可以 "shape" 生成的 distribution aesthetic，因为它直接匹配 reference statistics。

---

## 7. 与 Karpathy 视角的联系：Build Intuition

Andrej，这篇 paper 在多个层面触动你经常强调的 intuition：

### 7.1 Softmax 致命的诱惑 → Metric Goodhart's Law

你常讲的 "optimizing the proxy breaks the proxy" 在这里得到完美验证。Inception FID 被社区 implicit gradient descent 近十年，已经 over-fit 到 Inception feature 的 blind spot。Figure A.1 的 stress test 极其震撼——100× learning rate 优化 Inception，得到 IS 660, FID 2.09，但 FDr^6 = 50.66，图像明显 artifact。这就是 reward hacking 的 visual demo。

这跟 RLHF over-optimization ([Gao et al., ICML 2023](https://arxiv.org/abs/2210.10760)) 是同构问题：reward model 是 proxy，直接优化它会 destroy true preference。FD-loss 用 multi-representation（SIM）+ FDr^k 来 mitigate，这跟用 ensemble of reward models 的思想类似。

### 7.2 Distribution-level vs. Sample-level Objective

Generative modeling 主流是 sample-level loss：
- Diffusion / flow matching：$\|\hat{x}_0 - x\|^2$ 或 velocity regression
- GAN：adversarial，但 discriminator 仍 sample-level 判别
- Distillation：per-sample matching to teacher

FD-loss 是纯 distribution-level，只匹配 first two moments（mean + covariance）在某个 feature space。这让人想起：
- **Moment matching networks** ([Li et al., ICML 2015](https://arxiv.org/abs/1502.02761))——MMD-based generator
- **MCGAN** ([Mroueh et al., ICML 2017](https://arxiv.org/abs/1702.04159))——mean + covariance feature matching
- **Sliced Wasserstein** ([Deshpande et al., CVPR 2018](https://arxiv.org/abs/1710.07075))

但 prior work 都在 batch 内估计 distribution，被噪声淹没。FD-loss 的 key insight 是 **decoupling**——这跟 contrastive learning 用 memory bank / queue 解决 negative sample 数量的思路异曲同工。

### 7.3 "ImageNet generation 已 solved" 的 debunk

你过去 tweet 过 GAN 已 "solve" ImageNet 的话。这篇 paper 的 Figure 3 直接挑战这个 narrative：

- 在 FID 上，确实 solved——validation set 自己 FID 1.68，SOTA 已经 < 1.5；
- 在 FDr^6 上，远未 solved——SOTA 还在 3-6，validation set 是 1.0；
- **Human preference**（Figure 6）证实 FDr^6 比 FID 更接近人类判断——最强 generator 仍输给 real validation images。

这跟 [Jayasumana et al., CVPR 2024](https://arxiv.org/abs/2401.09603) "Rethinking FID" 的结论一致，但 paper 给出了 constructive 解决方案（FDr^k + FD-loss）。

### 7.4 1-NFE 是 distribution matching 的自然结果

Consistency model / DMD / score identity distillation 路线都需要 teacher。FD-loss 完全不需要 teacher——它只看 **distribution-level statistics**。这暗示一个深刻观点：

**One-step generation 不需要"教"模型怎么做单步预测，只需要让它的 output distribution 匹配 real distribution**。

这跟你讲 "softmax regression 直接做 next-token prediction 就行，不需要复杂算法" 类似——把 objective 选对，简单方法就 work。FD-loss 的 simplicity 让 1-NFE generation 退化为 distribution matching 问题。

### 7.5 EMA 的角色

EMA 在这里起双重作用：
1. **Numerical stability**：稳定 covariance 估计，避免 batch noise；
2. **On-policy regularizer**：upweight recent samples，避免 stale statistics。

这跟 BatchNorm 的 running statistics ([Ioffe & Szegedy, ICML 2015](https://arxiv.org/abs/1502.03167)) 完全同构。Paper 也明确提到这个联系。β = 0.999 是 sweet spot，跟 BatchNorm 默认 momentum 0.1（等价 EMA decay 0.9）相比要大得多，因为 FD 需要更稳定的估计。

---

## 8. 与你近期关注点的联系：Post-training 范式

这篇 paper 实际上定义了一个新的 **post-training 范式**——用 distributional distance 做 post-training，类似 RLHF 之于 LLM。

| Pre-training | Post-training | Objective |
|--------------|---------------|-----------|
| LLM next-token | RLHF / DPO | reward model from human prefs |
| Diffusion denoising | FD-loss | feature-space distribution matching |
| Multi-step generator | FD-loss repurposing | same |

这跟当前 LLM 的 test-time scaling / RLVR 路线遥相呼应：pre-train 用简单 objective（next-token / denoising），post-train 用更 refined 的 distribution-level objective（reward model / FD-loss）。

Appendix A.1 明确讨论了这个类比——FD-loss 是 "distribution-level reward supplied by representation model"。Reward hacking 风险也类似，所以 multi-representation（SIM）就是 reward model ensemble。

---

## 9. Limitations 和 Open Problems

Paper 自己承认的：
1. **Gaussian assumption**：FD 假设 feature distribution 是 Gaussian，只匹配 first two moments。Higher-order moment 信息丢失。这跟 MMD 用 kernel 可以匹配 infinite moment 不同；
2. **Representation 依赖**：FD-loss 效果强依赖 representation 选择。6 个 representation 也不一定穷尽 perceptual quality。作者明确说 "the representation problem remains open"；
3. **ImageNet 局限**：实验主要在 ImageNet 256/512。更大的分辨率、其他 domain（face, medical, video）未验证；
4. **No sampling diversity 分析**：FD-loss 直接优化 distribution matching，可能 collapse mode。Paper 没深入分析 diversity vs. fidelity trade-off。

我加的几个 open question：
- **Higher-order moment matching**：能否用 sliced Wasserstein 或 MMD 同时匹配 higher moments？这是 paper 的自然扩展；
- **Learned representation for FD**：能否 jointly learn representation 和 generator？类似 GAN 的 discriminator 但 distribution-level；
- **3D / Video FD-loss**：3D-aware representation（如 DINOv3 with temporal、3D feature）+ FD-loss 是否能扩展到 video generation post-training？

---

## 10. References

主 paper：
- [FD-loss GitHub](https://github.com/Jiawei-Yang/FD-loss)

关键 prior work：
- [Heusel et al., FID/GANs TTUR, NeurIPS 2017](https://arxiv.org/abs/1706.08500)
- [He et al., MoCo, CVPR 2020](https://arxiv.org/abs/1911.05722)
- [Ioffe & Szegedy, BatchNorm, ICML 2015](https://arxiv.org/abs/1502.03167)
- [Mathiasen & Hvilshøj, Backprop through FID, 2020](https://arxiv.org/abs/2009.14075)
- [Doan et al., FD in discriminator, 2020](https://arxiv.org/abs/2003.11774)
- [Li et al., GMMN, ICML 2015](https://arxiv.org/abs/1502.02761)
- [Mroueh et al., MCGAN, ICML 2017](https://arxiv.org/abs/1702.04159)
- [Deshpande et al., Sliced Wasserstein, CVPR 2018](https://arxiv.org/abs/1710.07075)
- [Jayasumana et al., Rethinking FID, CVPR 2024](https://arxiv.org/abs/2401.09603)
- [Kynkäänniemi et al., Role of ImageNet in FID, ICLR 2023](https://arxiv.org/abs/2201.08269)
- [Gao et al., Scaling laws for reward overopt, ICML 2023](https://arxiv.org/abs/2210.10760)
- [Song et al., Consistency Models, ICML 2023](https://arxiv.org/abs/2303.01469)
- [Yin et al., DMD, CVPR 2024](https://arxiv.org/abs/2311.18828)
- [Zhou et al., Score Identity Distillation, ICML 2024](https://arxiv.org/abs/2404.04057)
- [Goodhart's Law, 1984](https://en.wikipedia.org/wiki/Goodhart%27s_law)

Modern representation models：
- [Oquab et al., DINOv2](https://arxiv.org/abs/2304.07193)
- [He et al., MAE, CVPR 2022](https://arxiv.org/abs/2111.06377)
- [Tschannen et al., SigLIP2, 2025](https://arxiv.org/abs/2502.14786)
- [Radford et al., CLIP, ICML 2021](https://arxiv.org/abs/2103.00020)
- [Liu et al., ConvNeXt, CVPR 2022](https://arxiv.org/abs/2201.03545)

Generators 用到的：
- [Lu et al., pMF (pixel MeanFlow), 2026](https://arxiv.org/abs/2601.22158)
- [Geng et al., MeanFlow, NeurIPS 2025](https://arxiv.org/abs/2505.13479)
- [Geng et al., iMF, CVPR 2026]
- [Li & He, JiT, CVPR 2026]
- [Esser et al., SD3.5, ICML 2024](https://arxiv.org/abs/2403.03206)
- [Li et al., MAR, NeurIPS 2024](https://arxiv.org/abs/2406.11838)

---

## 11. 总结：Paper 的真正贡献

我把它放在四个层次：

1. **Technical**：decouple population from batch 的 FD 优化框架，EMA 实现 elegant，跟 MoCo queue 同构；
2. **Empirical**：FD-loss 跨 generator family / size / resolution / modality (class-cond + text-cond) 都 work，1-NFE FID 0.72 是当时最强；
3. **Diagnostic**：FDr^k 揭示 FID 已 saturated，会 misrank quality。这跟 Kynkäänniemi、Jayasumana、Stein 等批评 FID 的工作给出 constructive 替代；
4. **Conceptual**：把 generative modeling 从 sample-level loss 推向 distribution-level loss。Post-training 用 distribution distance 跟 LLM post-training 用 reward model 同构。这给社区一个新的 paradigm。

Paper 最美的部分是 **simplicity**——整个 method 就 30 行 code（Algorithm 1）。没有任何 trick：没有 adversarial、没有 distillation、没有 per-sample target、没有 curriculum、没有 auxiliary loss。就一个 FD + EMA + multi-representation。这跟 Karpathy 经常强调的 "simplicity is strength" 完全共鸣。

它也提醒我们：long-standing assumption（"FD 不能直接优化"）往往只是 practical 限制，不是 principle 阻碍。**Decoupling estimation scale from optimization scale** 这个 pattern 可能还有很多 application——任何需要 population-level statistics 但 batch-level gradient 的场景（contrastive learning、distribution matching、moment matching）都值得重新审视。
