---
source_pdf: What Matters for Diffusion-Friendly Latent Manifold Prior-Aligned Autoencoders
  for Latent Diffusion.pdf
paper_sha256: c993b2a17be71f327c77ce05c1b0a626f9bf125c83cf65cc3266207a7baff788
processed_at: '2026-08-13T04:09:23-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# PAE 用人话说

## 一句话总结

Latent diffusion model 的 tokenizer 不要只盯着 reconstruction，要主动把 latent space 的"形状"捏好，diffusion 学起来才轻松。

## 核心问题在哪

Latent diffusion model（LDM）的工作流程是：image 经过 tokenizer 压缩成 latent，diffusion model 在这个 latent space 里做生成，最后 decoder 把 latent 还原成 image。

问题在于：tokenizer 传统上只被训练做一件事——**把 image 压成 latent 再还原回去，还原得越像越好**。这个 objective 叫 reconstruction fidelity。

但是 paper 在 Figure 2(a) 做了一个特别扎心的实验：把 tokenizer 的 bottleneck channel 从 32 一路加到 128，rFID（reconstruction error）一路下降，但是 gFID（generation quality）根本不跟着降，甚至有时还变差。

这就好比：你给一个学生一套特别清晰的教科书，他考试反而考砸了。教科书清晰不等于学生学得好——**教科书的内容组织方式**才是关键。

PAE 的核心 insight 就是：**latent space 的几何组织方式（manifold geometry）决定了 diffusion model 学得好不好，而不是 reconstruction 做得多精确**。

## 三个 Manifold Properties

Paper 把"latent space 长什么样"拆成了三个可测量的维度，每个维度对应 diffusion 学习的一个痛点。

### 1. Spatial Structure Coherence (SSC) — 空间结构对不对

想象 latent 是一个 16×16 的 token grid。每个 token 对应 image 的一个 patch。SSC 衡量的是：**这些 token 之间的 spatial grouping 是否跟 image 里的 object boundary 对应**。

测量方法：对 latent tokens 算一个 affinity matrix $A_{ij}$，做 spectral clustering，看 cluster 结果跟 COCO panoptic ground truth 对不对得上。用 normalized mutual information 量化：

$$\text{SSC}(\mathbf{y}, \hat{\mathbf{y}}) = \frac{I(\mathbf{y}; \hat{\mathbf{y}})}{\sqrt{H(\mathbf{y}) H(\hat{\mathbf{y}})}}$$

- $\mathbf{y}$: ground truth object labels，从 COCO panoptic mask downsample 到 16×16
- $\hat{\mathbf{y}}$: latent token clustering 结果
- $I(\cdot;\cdot)$: mutual information
- $H(\cdot)$: Shannon entropy

**Intuition**：如果 latent token grouping 已经跟 object structure 对齐了，diffusion model 的 self-attention 就不需要花精力去"修正"空间错位，直接 focus 学生成 pattern 就行。

Figure 2(b) 的 controlled experiment 验证了这点：只改 SSR 的 weight，SSC 越高，gFID 越低，monotonic 的。

### 2. Local Perceptual Continuity (LPC) — 局部连续不连续

LPC 回答：**latent 稍微扰动一下，decode 出来的 image 变化有多大**。

$$\text{LPC}_\epsilon = \mathbb{E}_{\mathbf{x}, \mathbf{u}}\left[\frac{d_{\text{LPIPS}}(D(\mathbf{z} + \epsilon \mathbf{u}), D(\mathbf{z})) + d_{\text{LPIPS}}(D(\mathbf{z} - \epsilon \mathbf{u}), D(\mathbf{z}))}{2}\right]$$

- $\mathbf{z} = E(\mathbf{x})$: encoded latent
- $\mathbf{u} \sim \text{Unif}(\mathbb{S}^{d-1})$: random unit direction
- $\epsilon$: perturbation scale
- $D(\cdot)$: decoder
- $d_{\text{LPIPS}}$: LPIPS perceptual distance

**Intuition**：Diffusion 本质上是 local prediction——给定 noisy latent，预测少一点 noise 的 latent。如果 latent space 局部很 rough，稍微动一下 decode 出来的 image 就天差地别，diffusion 的 prediction target 就很不稳定，训练难收敛，inference 需要很多步。

Figure 2(c) 验证：LPC 越小，gFID 越好，而且是三个 metric 里 cross-tokenizer correlation 最强的。这说明 local continuity 是最 fundamental 的 property。

### 3. Global Semantic Quality (GSQ) — 语义聚类纯不纯

GSQ 看：**latent space 里，每个 sample 的 nearest neighbor 是不是同类**。

$$\text{GSQ} = \frac{1}{N_c}\sum_{i=1}^{N_c} \mathbf{1}[y_{j^\star(i)} = y_i]$$

- $j^\star(i) = \arg\max_{j \neq i} \langle \tilde{\mathbf{f}}_i, \tilde{\mathbf{f}}_j \rangle$: cosine similarity 最近的 neighbor
- $y_i$: class label
- $\tilde{\mathbf{f}}_i$: globally pooled latent feature，$\ell_2$ normalized

**Intuition**：class-conditional generation 里，diffusion model 要学的是"给定 class label，生成这个 class 的 image"。如果 latent space 里同类 sample 已经聚在一起，generator 学起来就容易；如果同类 sample 散落各处，generator 就很痛苦。

## PAE 怎么把这三个 Property 捏出来

### Architecture Overview

```
Image x
  → Frozen VFM (DINOv2-L) → H_vfm [语义 backbone，不动]
  → DAM [注入 pixel detail]
  → Projector → z̃ [压成 16×16×32]
  → RMS Norm → z [球面 latent]
  → Deprojector + Decoder → x̂ [重建]
```

关键设计：**frozen VFM 提供语义骨架，DAM 注入细节，projector 压到 compact latent**。

### Detail-Aware Modulator (DAM)

Frozen VFM feature 语义强但缺细节。直接 finetune VFM 会破坏 pretrained structure。DAM 的做法：

```python
# K=6 层 transformer block
H_p = patchify(x)  # pixel tokens
for l in range(K):
    H_p = SelfAttn(H_p)       # pixel tokens 内部交互
    H_p = CrossAttn(H_p, H_vfm)  # attend 到 frozen VFM
    H_p = MLP(H_p)

# Zero-init scale-shift fusion
gamma_p, beta_p = split(W @ H_p)  # W 初始化为 0
H_z = LayerNorm(H_vfm * (1 + gamma_p) + beta_p)
```

**关键 trick**：$\mathbf{W}$ zero-initialized。训练开始时 $\gamma_p = 0, \beta_p = 0$，所以 $\mathbf{H}_z = \text{LayerNorm}(\mathbf{H}_{\text{vfm}})$，完全由 VFM 主导。随着训练，detail 逐渐注入。

为什么用 scale-shift 而不是 residual concat？Table 3(b) 的 ablation：

| Method | gFID |
|--------|------|
| Finetuning VFM | 2.13 |
| Residual Add | 2.46 |
| Residual Concat | 2.38 |
| **DAM** | **1.80** |

Residual concat 引入 uncontrolled mixing，scale-shift 的 zero-init 保证 controlled injection。

### Sphere Manifold via RMS Norm

$$z = \tilde{z} / \sqrt{\text{mean}(\tilde{z}^2) + \epsilon}$$

Latent 被投影到 unit sphere 上。好处：
1. MCR 的 random direction sampling 在 sphere 上 natural
2. 去掉 magnitude redundancy，channel 利用更均匀
3. 匹配 diffusion 的 isotropic Gaussian noise assumption

### 三个 Regularization

#### SSR: Spatial Structure Regularization

$$\mathcal{L}_{\text{SSR}} = \|\mathbf{G}_z - \mathbf{G}_T\|_F^2$$

- $\mathbf{G}_z = \mathbf{Z}^\top \mathbf{Z}$: latent 的 Gram matrix
- $\mathbf{G}_T$: refined VFM target 的 Gram matrix
- $\|\cdot\|_F$: Frobenius norm

Gram matrix 的 $(i,j)$ entry 是 token $i$ 和 token $j$ 的 inner product，捕捉 patch 间的 spatial correlation。

**为什么用 Gram 而不是直接 feature alignment**：Gram matrix 是 rotation-invariant，只约束 relative structure 不约束 absolute orientation，避免 over-constrain。

#### MCR: Manifold Continuity Regularization — 最精妙的设计

这个设计解决一个 dilemma：怎么 regularize local continuity 不破坏 reconstruction。

Naive 方法：让 decoder 从 perturbed latent 重建 original image。问题：perturbation 大了破坏 reconstruction，小了 regularization 太弱。

PAE 的 cascaded perturbation：

```python
z_r = encoder(x)                    # anchor
Delta = random_direction()          # random direction on sphere
z_m = z_r + alpha_m * Delta         # medium perturbation (42.5°)
z_l = z_r + alpha_l * Delta         # large perturbation (85°)

x_r = decoder(z_r)
x_m = decoder(z_m)
x_l = decoder(z_l)

# Cascaded consistency, 不是全部对齐到 anchor
L_mcr = ||x_m - sg(x_r)||_1 + LPIPS(x_m, sg(x_r))     # medium → recon
       + ||x_l - sg(x_m)||_1 + LPIPS(x_l, sg(x_m))    # large → medium
```

`sg()` 是 stop-gradient，anchor 不被更新。

**Intuition**：不要求 large perturbation 直接回到 anchor，而是 large→medium→recon 逐级对齐。每一级只要求 local consistency，通过 transitivity 获得全局连续性。

这跟 diffusion 的 coarse-to-fine process 天然契合：diffusion 也是一小步一小步 denoise。

Table 7 的 ablation 直接验证：

| Method | LPC | rFID | gFID |
|--------|-----|------|------|
| No Perturb | 0.258 | 0.25 | 2.10 |
| Small (42.5°) | 0.219 | 0.26 | 2.00 |
| Large (85°) | 0.205 | **0.28** | 2.04 |
| **Cascaded** | **0.170** | 0.26 | **1.80** |

Large perturbation 单独用 LPC 最好但 rFID 受损；cascaded 同时拿 best LPC 和 best gFID 且不伤 rFID。

#### SCR: Semantic Consistency Regularization

$$\mathcal{L}_{\text{SCR}} = \left(1 - \cos(\bar{\mathbf{z}}_{T,g}, \bar{\mathbf{z}}_g)\right) + \left(1 - \cos(\bar{\mathbf{z}}_T, \bar{\mathbf{z}})\right)$$

- 第一项：pooled token 的 cosine similarity（global semantic）
- 第二项：patch-level token 的 cosine similarity（dense semantic direction）
- $\bar{\cdot}$: $\ell_2$ normalization

Table 3(c) 显示 patch-level supervision 很重要：

| Target | gFID |
|--------|------|
| Pooling Token only | 2.14 |
| Feature Tokens only | 1.87 |
| **Full Tokens** | **1.80** |

## VFM Prior Refinement — 容易被忽略但重要

Raw VFM features 不能直接用作 alignment target：
1. DINOv2-L 是 D=1024，tokenizer bottleneck 是 d=32，dimension mismatch
2. VFM patch feature 在 16×16 resolution 上 spatial structure 不够 clean

所以先单独训一个 refine stage：

```
Raw VFM feature H_vfm
  → Projector P_t → Z_T [compact, d=32]
  → Deprojector Q_t → H_vfm_hat [重建回高维]
  
Loss: ||H_vfm_hat - H_vfm||^2  +  ||Gram(Z_T) - Gram(H_ref)||^2
```

$\mathbf{H}_{\text{ref}}$ 是 raw VFM feature 经过 AnyUp upsample → low-pass filter → downsample 回 16×16，suppress 噪声保留 coarse structure。

Refine 之后 target 更 bottleneck-matched，Table 2(b) 显示全 metric 都有 modest improvement。

## 实验结果

### Main Results (Table 1)

**80 epochs（convergence efficiency）**：

| Method | rFID | gFID (guided) |
|--------|------|---------------|
| REPA | 0.61 | 1.42 |
| GAE | 0.44 | 1.48 |
| VTP | 0.36 | 1.44 |
| RAE (DiTDH-XL, 839M) | 0.57 | — |
| **PAE (DINOv2)** | **0.26** | **1.27** |

PAE 用 675M params + 普通 CFG，打败了用 839M params + AutoGuidance 的 RAE。

**800 epochs（final performance）**：

| Method | gFID (guided) |
|--------|---------------|
| SFD | 1.06 |
| RAE (DiTDH-XL) | 1.13 |
| Send-VAE + REPA | 1.21 |
| **PAE (DINOv2)** | **1.03** |

1.03 是 ImageNet 256×256 的 new SOTA。

### Convergence Speed

PAE 比 RAE 快 **13×** 达到 comparable quality。Intuition：latent manifold 已经 organized 好，diffusion 不需要 waste early epochs 学 latent space structure。

### Few-Step Sampling (Table 6)

| Steps | gFID |
|-------|------|
| 10 | 1.88 |
| 15 | 1.28 |
| 45 | 1.06 |
| 250 | 1.03 |

15 steps 就匹配 FAE 的 250-step performance，**16.7× fewer inference steps**。这是 MCR 的直接效果——locally continuous manifold 让 diffusion trajectory 更容易快速 traverse。

### Ablation: 三个 Regularization 的互补性 (Table 2a)

| SSR | MCR | SCR | SSC | LPC | GSQ | gFID |
|-----|-----|-----|-----|-----|-----|------|
| × | × | × | 0.18 | 0.320 | 0.19 | 7.18 |
| ✓ | × | × | 0.29 | 0.296 | 0.26 | 2.74 |
| × | ✓ | × | 0.23 | 0.221 | 0.24 | 2.53 |
| × | × | ✓ | 0.21 | 0.286 | 0.39 | 2.63 |
| ✓ | ✓ | × | 0.33 | 0.187 | 0.33 | 2.02 |
| ✓ | × | ✓ | 0.31 | 0.258 | 0.46 | 2.10 |
| × | ✓ | ✓ | 0.24 | 0.176 | 0.45 | 2.08 |
| **✓** | **✓** | **✓** | **0.35** | **0.170** | **0.50** | **1.86** |

每个 objective most strongly 改善对应的 metric：SSR→SSC, MCR→LPC, SCR→GSQ。三个一起用全面最优。

### Generic Regularization 对比 (Table 3a)

| Method | gFID |
|--------|------|
| No regularization | 7.79 |
| KL Reg | 5.17 |
| Diffusion-loss Reg | 4.22 |
| **PAE (geometry-targeted)** | **1.80** |

Generic regularizer 有帮助但远不如 targeted manifold shaping。这证明 gain 来自 **manifold geometry 的 explicit design**，不只是 regularization 本身。

### Cross-Encoder Generalization (Table 8)

| VFM | w/o $\mathcal{L}_p$ | w/ $\mathcal{L}_p$ |
|-----|---------------------|--------------------|
| DINOv2 | 7.79 | 1.80 |
| SigLIP2 | 6.89 | 2.32 |
| DINOv3 | 6.62 | 1.81 |
| MAE | 7.97 | 3.65 |

Prior alignment 在所有 VFM 上都带来 huge improvement，证明方法 generality。DINO 系列最好因为 self-supervised pretraining 产生 spatially 和 semantically balanced features；SigLIP semantic 强但 spatial 弱；MAE spatial 还行但 semantic 弱。

## 为什么这个 Framework Work

传统 tokenizer 只约束 observed data points 的 reconstruction，对 neighborhood 和 global organization 几乎无约束。Latent manifold 的 geometry 是**被动 emerge**的，受限于 reconstruction objective 的 inductive bias。

PAE 主动 shape 三个 dimension：
- **SSR** 填补 spatial topology gap（reconstruction 不保证 token relation survive bottleneck）
- **MCR** 填补 local neighborhood gap（reconstruction 只约束 anchor point）
- **SCR** 填补 global semantic gap（reconstruction 无 semantic supervision）

这三个 regularization 对应 diffusion 学习的三个痛点，所以 targeted shaping 比 generic regularization 效果好得多。

## Broader Perspective

PAE 提供了一个 rethinking tokenizer design 的 framework：**用 manifold geometry 而非 reconstruction fidelity 来 evaluate tokenizer quality**。这个 perspective 可能影响：

1. **Tokenizer benchmarking**：SSC, LPC, GSQ 作为新的 diagnostic metrics，Figure 12 的 cross-tokenizer validation 显示这些 metric 在 diverse tokenizer families 上都 correlate with generation quality
2. **Tokenizer-generator co-design**：显式考虑 manifold properties 而非只 optimize reconstruction
3. **Unified perspective**：spectral methods（SER, EQ-VAE）可以理解为 biasing manifold toward smoother structure；structural methods（SSVAE, DC-AE 1.5）理解为 simplifying local geometry；representation methods（RAE, GAE）理解为 improving semantic organization——都是 manifold organization 的不同 facet

## References

- [PAE Paper](https://arxiv.org/abs/2512.19862)
- [RAE - Zheng et al. 2025](https://arxiv.org/abs/2510.11690)
- [REPA - Yu et al. 2024](https://arxiv.org/abs/2410.06940)
- [VA-VAE - Yao et al. 2025](https://arxiv.org/abs/2501.09755)
- [GAE - Liu et al. 2026](https://arxiv.org/abs/2603.10365)
- [AlignTok - Chen et al. 2026](https://arxiv.org/abs/2509.25162)
- [DINOv2 - Oquab et al. 2023](https://arxiv.org/abs/2304.07193)
- [DINOv3 - Siméoni et al. 2025](https://arxiv.org/abs/2508.10104)
- [SigLIP2 - Tschannen et al. 2025](https://arxiv.org/abs/2502.14786)
- [Latent Diffusion - Rombach et al. 2022](https://arxiv.org/abs/2112.10752)
- [EQ-VAE - Kouzelis et al. 2025](https://arxiv.org/abs/2502.09509)
- [SER - Skorokhodov et al. 2025](https://arxiv.org/abs/2502.14831)
- [Manifold Hypothesis Survey - Loaiza-Ganem et al. 2024](https://arxiv.org/abs/2404.02954)
- [LPIPS - Zhang et al. 2018](https://arxiv.org/abs/1801.03924)
- [Spectral Clustering - von Luxburg 2007](https://arxiv.org/abs/0711.0187)
- [AnyUp - Wimmer et al. 2026](https://arxiv.org/abs/2510.12764)
- [VTP - Yao et al. 2026](https://arxiv.org/abs/2512.13687)
- [UniFlow - Yue et al. 2025](https://arxiv.org/abs/2510.10575)
- [TokenFlow - Qu et al. 2025](https://arxiv.org/abs/2412.03069)
- [SFD - Pan et al. 2025](https://arxiv.org/abs/2512.04926)
- [Send-VAE - Page et al. 2026](https://arxiv.org/abs/2601.05823)
- [FAE - Gao et al. 2025](https://arxiv.org/abs/2512.07829)
- [SVG - Shi et al. 2025](https://arxiv.org/abs/2510.15301)
- [MAE - He et al. 2021](https://arxiv.org/abs/2111.06377)
- [CRT - Ramanujan et al. 2025](https://arxiv.org/abs/2412.16326)
- [SSVAE - Liu et al. 2025](https://arxiv.org/abs/2512.05394)
- [DC-AE - Chen et al. 2025](https://arxiv.org/abs/2410.10733)
- [DDT - Wang et al. 2025](https://arxiv.org/abs/2504.05741)
- [REPA-E - Leng et al. 2025](https://arxiv.org/abs/2504.10483)

---

# PAE: Prior-Aligned Autoencoders for Latent Diffusion — 深度技术解析

## 1. 核心问题的动机与Intuition

这篇paper直击latent diffusion model (LDM)中一个被长期忽视的问题：**tokenizer定义的latent space几何性质决定了downstream diffusion的学习难度**，但是现有tokenizer设计objective（reconstruction fidelity或feature inheritance）与generation quality之间存在systematic mismatch。

Figure 2(a)的pilot experiment揭示了一个关键事实：当bottleneck channel dimension从32增加到128时，rFID单调改善，但gFID并不随之改善——这说明reconstruction-oriented objective无法保证latent space对diffusion友好。这个observation与VA-VAE ([Yao et al. 2025](https://arxiv.org/abs/2501.09755))和CRT ([Ramanujan et al. 2025](https://arxiv.org/abs/2412.16326))的发现一致，但PAE把这个insight推到了更深的层次。

## 2. 三个Manifold Properties的Formalization

### 2.1 Spatial Structure Coherence (SSC)

SSC衡量latent tokens是否按照object-aware spatial structure进行grouping。这个metric基于一个关键intuition：transformer-based generator通过self-attention建模token-token relations，如果latent tokenization本身已经spatially coherent，那么generator就不需要"补偿"spatial incoherence。

具体计算（Eq 7-8）：

$$A_{ij} = \exp\left(\frac{\langle \hat{\mathbf{z}}_i, \hat{\mathbf{z}}_j \rangle}{\sigma}\right), \quad \hat{\mathbf{z}}_i = \frac{\mathbf{z}_i}{\|\mathbf{z}_i\|_2}, \quad A_{ii} = 0$$

这里 $\mathbf{z}_i \in \mathbb{R}^C$ 是第 $i$ 个latent token，$\hat{\mathbf{z}}_i$ 是其 $\ell_2$-normalized形式，$\sigma > 0$ 是temperature parameter，$A \in \mathbb{R}^{N \times N}$ 是token affinity matrix（$N = HW$）。

通过normalized spectral clustering ([von Luxburg 2007](https://arxiv.org/abs/0711.0187))得到predicted labels $\hat{\mathbf{y}}$，与COCO Panoptic的ground truth $\mathbf{y}$比较：

$$\text{SSC}(\mathbf{y}, \hat{\mathbf{y}}) = \frac{I(\mathbf{y}; \hat{\mathbf{y}})}{\sqrt{H(\mathbf{y}) H(\hat{\mathbf{y}})}}$$

$I(\cdot;\cdot)$是mutual information，$H(\cdot)$是Shannon entropy。使用geometric mean normalization的NMI使得SSC对cluster数量不敏感。

### 2.2 Local Perceptual Continuity (LPC)

LPC测量decoder在local latent perturbation下的perceptual stability。这个metric直接对应diffusion的local prediction nature：flow matching和denoising都是local prediction problems，nearby latent points应该map到nearby prediction targets。

公式（Eq 9-10）：

$$\text{LPC}_\epsilon = \mathbb{E}_{\mathbf{x}, \mathbf{u}}\left[\frac{d_{\text{LPIPS}}(D(\mathbf{z} + \epsilon \mathbf{u}), D(\mathbf{z})) + d_{\text{LPIPS}}(D(\mathbf{z} - \epsilon \mathbf{u}), D(\mathbf{z}))}{2}\right]$$

- $\mathbf{z} = E(\mathbf{x})$: encoder output
- $\mathbf{u} \sim \text{Unif}(\mathbb{S}^{d-1})$: 随机unit perturbation direction在latent sphere上
- $\epsilon > 0$: perturbation scale
- $d_{\text{LPIPS}}$: LPIPS perceptual distance ([Zhang et al. 2018](https://arxiv.org/abs/1801.03924))
- $D(\cdot)$: full decoder

Multi-scale加权（Eq 10）：

$$\text{LPC} = \sum_{s=1}^{|\mathcal{R}|} w_s \text{LPC}_{\epsilon_s}, \quad w_s = \frac{\rho_s^{-1}}{\sum_{r=1}^{|\mathcal{R}|} \rho_r^{-1}}$$

$\mathcal{R} = \{0.1, 0.5, 1.0, 2.0\}$，$\epsilon_s = \rho_s \|\mathbf{z}\|_2$。权重 $w_s$ 使得smaller scales获得更大权重，强调infinitesimal neighborhood的continuity，同时用moderately larger scales增强robustness。

### 2.3 Global Semantic Quality (GSQ)

GSQ测量latent space中nearest neighbor的label purity。这个metric不同于class-centroid compactness，它evaluates whether local semantic neighborhoods are label-consistent，这对class-conditional generation特别重要。

公式（Eq 11-14）：

$$\mathbf{f}_i = \text{GAP}(\mathbf{Z}_i) = \frac{1}{HW}\sum_{h=1}^H \sum_{w=1}^W \mathbf{Z}_i[:, h, w] \in \mathbb{R}^C$$

$$\bar{\mathbf{f}} = \frac{1}{N_c}\sum_{i=1}^{N_c} \mathbf{f}_i, \quad \tilde{\mathbf{f}}_i = \frac{\mathbf{f}_i - \bar{\mathbf{f}}}{\|\mathbf{f}_i - \bar{\mathbf{f}}\|_2}$$

$$j^\star(i) = \arg\max_{j \neq i} \langle \tilde{\mathbf{f}}_i, \tilde{\mathbf{f}}_j \rangle$$

$$\text{GSQ} = \frac{1}{N_c}\sum_{i=1}^{N_c} \mathbf{1}[y_{j^\star(i)} = y_i]$$

$\mathbf{1}[\cdot]$是指示函数，$y_i$是class label。实践中取100个ImageNet classes的random subset计算，report 5次的mean和std。

### 2.4 Supplementary: Effective Rank (eRank)

$$\text{erank}(\mathbf{F}) = \exp\left(-\sum_{i=1}^C \bar{\sigma}_i \log \bar{\sigma}_i\right), \quad \text{eRank} = \frac{\text{erank}(\mathbf{F})}{C}$$

$\bar{\sigma}_i = \sigma_i / \sum_j \sigma_j$是normalized singular values，$\mathbf{F} \in \mathbb{R}^{N \times C}$是pooled feature matrix。eRank是entropy-based effective rank ([Roy & Vetterli](https://ieeexplore.ieee.org/document/1576978))，衡量channel utilization的均匀程度。

### 2.5 Cross-Tokenizer Validation (Appendix D.1, Figure 12)

值得强调的是，作者在diverse tokenizer families（VAE-based, VQ-based, masked-token, representation AE）上验证了这些metric的generality。Pearson correlation显示：
- LPC与gFID的monotonic relation最清晰（across tokenizer families）
- GSQ和eRank有meaningful positive correlation
- SSC在cross-tokenizer setting下correlation较弱（因为architecture差异太大）

这说明**local continuity是跨tokenizer family最robust的indicator**，而spatial structure更依赖于specific architecture。

## 3. PAE Architecture深度解析

### 3.1 Overall Pipeline

```
Input x (B×3×256×256)
    ↓
Frozen VFM E (DINOv2-L)
    → H_vfm (B×N×D), N=1024 patch tokens, D=1024
    ↓
DAM (K=6 transformer blocks)
    + pixel tokens H_p (from patchify)
    → H_z (modulated representation)
    ↓
Projector P_θ (attention + conv)
    → z̃ (B×32×16×16)
    ↓
RMS Normalization
    → z (sphere-like latent)
    ↓
[For downstream diffusion]
    ↓
Deprojector Q_θ + Pixel Decoder D_θ
    → x̂ (reconstruction)
```

### 3.2 Detail-Aware Modulator (DAM) — 关键设计

DAM解决一个核心dilemma：frozen VFM features有强semantic structure但missing fine-grained detail，而直接finetuning VFM会destroy pretrained structure。

DAM的computation（K层迭代）：

$$\mathbf{H}_p^{(l)} = \text{MLP}\left(\text{CrossAttn}\left(\text{SelfAttn}(\mathbf{H}_p^{(l-1)}), \mathbf{H}_{\text{vfm}}\right)\right)$$

- $\mathbf{H}_p^{(0)}$: patchified pixel tokens
- $\mathbf{H}_{\text{vfm}}$: frozen VFM features作为cross-attention的keys/values
- 第 $l$ 层：self-attention先在pixel tokens内部交互，再cross-attend到VFM features

Zero-initialized scale-shift fusion（Eq 1）：

$$\gamma_p, \beta_p = \text{split}(\mathbf{W} \Delta\mathbf{H}), \quad \mathbf{H}_z = \text{LayerNorm}(\mathbf{H}_{\text{vfm}} \odot (1 + \gamma_p) + \beta_p)$$

- $\Delta\mathbf{H} = \mathbf{H}_p^{(K)}$: DAM output
- $\mathbf{W}$: **initialized to zero**，所以训练初始时 $\gamma_p = 0, \beta_p = 0$，即 $\mathbf{H}_z = \text{LayerNorm}(\mathbf{H}_{\text{vfm}})$
- $\odot$: element-wise multiplication

这个zero-init设计非常关键：训练开始时latent完全由VFM主导，随着训练progressively inject pixel detail。Table 3(b)的ablation显示：
- Finetuning VFM: gFID 2.13
- Residual Add: 2.46
- Residual Concat: 2.38
- DAM: **1.80**

DAM的zero-init scale-shift比简单residual concat好很多，因为residual concat引入uncontrolled mixing。

### 3.3 Low-Dimensional Sphere Manifold

Projector output经过RMS normalization：

$$z = \tilde{z} / \sqrt{\text{mean}(\tilde{z}^2) + \epsilon}$$

- $\tilde{z} \in \mathbb{R}^{B \times d \times H' \times W'}$，$d=32$, $H'=W'=16$
- $\epsilon$: small constant for numerical stability

这把latent投影到unit sphere上，有三个好处：
1. **去除channel redundancy**：sphere上的distribution更compact
2. **稳定MCR的perturbation**：在sphere上sampling direction有natural geometry
3. **匹配diffusion的isotropic noise assumption**：Gaussian noise在sphere上的行为更uniform

Figure 8(a)显示latent dimension在32附近performance peak，过小capacity不足，过大channel utilization下降（eRank降低）。

## 4. Prior Alignment Regularizations — 方法核心

### 4.1 Spatial Structure Regularization (SSR)

$$\mathcal{L}_{\text{SSR}} = \|\mathbf{G}_z - \mathbf{G}_T\|_F^2$$

- $\mathbf{G}_z = \mathbf{Z}^\top \mathbf{Z} \in \mathbb{R}^{N \times N}$: latent的Gram matrix
- $\mathbf{G}_T = \mathbf{Z}_T^\top \mathbf{Z}_T$: refined VFM target的Gram matrix
- $\|\cdot\|_F$: Frobenius norm

Gram matrix的 $(i,j)$ entry是token $i$ 和token $j$ 的inner product，捕捉patch-wise spatial correlation。这个设计的巧妙之处在于：Gram matrix是**rotation-invariant**的，所以只约束relative structure而不约束absolute orientation，避免了over-constraining。

### 4.2 Manifold Continuity Regularization (MCR) — 最精妙的设计

MCR是这篇paper最technical sophisticated的部分。核心问题是：如何regularize local Lipschitz continuity而不破坏reconstruction？

Naive方法的问题：直接让decoder从perturbed latent reconstruct original image，会有trade-off——large perturbation损害fidelity，small perturbation regularization太弱。

MCR的cascaded perturbation design（Eq 4）：

构造两个perturbed latents：

$$\mathbf{z}_m = \mathbf{z}_r + \alpha_m \Delta, \quad \mathbf{z}_l = \mathbf{z}_r + \alpha_l \Delta, \quad \alpha_l > \alpha_m > 0$$

- $\mathbf{z}_r \sim q(\mathbf{z}|x)$: reconstruction latent
- $\Delta$: 随机direction（在RMS-normalized sphere上）
- $\alpha_m, \alpha_l$: 两个perturbation scale，对应42.5°和85° angular deviation

Cascaded consistency loss：

$$\mathcal{L}_{\text{MCR}} = \underbrace{\|\hat{x}_m - \text{sg}(\hat{x}_r)\|_1 + \text{LPIPS}(\hat{x}_m, \text{sg}(\hat{x}_r))}_{\text{medium} \to \text{recon}} + \underbrace{\|\hat{x}_l - \text{sg}(\hat{x}_m)\|_1 + \text{LPIPS}(\hat{x}_l, \text{sg}(\hat{x}_m))}_{\text{large} \to \text{medium}}$$

- $\hat{x}_r = D(\mathbf{z}_r)$, $\hat{x}_m = D(\mathbf{z}_m)$, $\hat{x}_l = D(\mathbf{z}_l)$
- $\text{sg}(\cdot)$: stop-gradient，关键！stop-gradient使得anchor不被更新

这个cascaded design的intuition非常漂亮：**不要求large perturbation直接回到anchor，而是通过medium作为intermediate**。这创造了progressive regularization：large→medium→recon，每个step只要求local consistency。

Table 7的ablation直接验证了这个设计：
- No Perturb: LPC 0.258, gFID 2.10
- Small Perturb (42.5°): LPC 0.219, gFID 2.00
- Large Perturb (85°): LPC 0.205, rFID 0.28 (reconstruction受损!)
- **Cascaded (42.5° + 85°)**: LPC 0.170, rFID 0.26, gFID 1.80

Cascaded design同时获得best LPC和best gFID，且不牺牲rFID。

### 4.3 Semantic Consistency Regularization (SCR)

$$\mathcal{L}_{\text{SCR}} = \left(1 - \cos(\bar{\mathbf{z}}_{T,g}, \bar{\mathbf{z}}_g)\right) + \left(1 - \cos(\bar{\mathbf{z}}_T, \bar{\mathbf{z}})\right)$$

- $\bar{\cdot}$: $\ell_2$ normalization
- $\mathbf{z}_{T,g}$: pooled target token (global semantic)
- $\mathbf{z}_g$: pooled latent
- $\mathbf{z}_T$: patch-level target tokens
- $\mathbf{z}$: patch-level latent

第一项：concept-level alignment through pooling
第二项：dense semantic direction preservation

Table 3(c)显示：
- Pooling Token only: gFID 2.14
- Feature Tokens only: 1.87
- **Full Tokens (pooling + patch)**: 1.80

这说明patch-level semantic direction对diffusion很重要，因为它preserves dense semantic structure而不仅仅是global concept。

### 4.4 Total Objective

$$\mathcal{L}_p = \lambda_{ssr}\mathcal{L}_{\text{SSR}} + \lambda_{mcr}\mathcal{L}_{\text{MCR}} + \lambda_{scr}\mathcal{L}_{\text{SCR}}$$

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \mathcal{L}_p$$

$$\mathcal{L}_{\text{recon}} = \mathcal{L}_{\ell_1} + \lambda_{\text{lpips}}\mathcal{L}_{\text{LPIPS}} + \lambda_{\text{gan}}\mathcal{L}_{\text{GAN}}$$

Loss weights: $\lambda_{\text{lpips}}=1.0, \lambda_{\text{gan}}=0.5, \lambda_{ssr}=0.2, \lambda_{mcr}=0.5, \lambda_{scr}=1.0$

注意SCR权重最大（1.0），MCR其次（0.5），SSR最小（0.2）。这暗示global semantic organization是最重要的，而spatial structure可以通过Gram matrix的soft constraint自然emerge。

## 5. VFM Prior Refinement — 被低估的关键步骤

### 5.1 为什么Raw VFM Features不适合作为Alignment Target

Raw VFM features有两个问题：
1. **Channel-redundant**: DINOv2-L的D=1024，而tokenizer bottleneck是d=32，直接distill会有dimensionality mismatch
2. **Spatially imperfect at tokenizer resolution**: VFM的patch features在16×16 resolution上spatial structure不够clean

### 5.2 Refine Stage Pipeline

Refine stage独立训练一个lightweight projector-deprojector pair：

$$\mathbf{Z}_T = \mathcal{P}_\theta^t(\mathbf{H}_{\text{vfm}}) \in \mathbb{R}^{N \times d}, \quad d=32$$

$$\hat{\mathbf{H}}_{\text{vfm}} = \mathcal{Q}_\theta^t(\mathbf{Z}_T)$$

$\mathcal{Q}_\theta^t$是4-layer ViT，hidden dim 1024。

Refine objective：

$$\mathcal{L}_{\text{refine}} = \lambda_{\text{rep}}\mathcal{L}_{\text{rep}} + \lambda_{\text{gram}}\mathcal{L}_{\text{gram}}$$

$$\mathcal{L}_{\text{rep}} = \|\hat{\mathbf{H}}_{\text{vfm}} - \mathbf{H}_{\text{vfm}}\|_2^2$$

$$\mathcal{L}_{\text{gram}} = \|\text{Gram}(\mathbf{Z}_T) - \text{Gram}(\mathbf{H}_{\text{ref}})\|_F^2$$

- $\mathbf{H}_{\text{ref}}$: raw VFM经过AnyUp ([Wimmer et al. 2026](https://arxiv.org/abs/2510.12764)) upsample到256×256，再low-pass spatial normalization（strength 0.4），再bilinear downsample回16×16

### 5.3 Refinement的效果

Table 2(b)的ablation显示refined vs raw targets：
- SSC: 0.33 → 0.35
- LPC: 0.171 → 0.170
- GSQ: 0.48 → 0.50
- rFID: 0.27 → 0.26
- gFID: 1.95 → **1.86**

Improvement modest but consistent across all metrics。Figure 4的visualization显示refined structural target有clearer patch-wise spatial correlation，compressed semantic target在embedding space仍然well-clustered。

## 6. 实验结果深度分析

### 6.1 Main Results (Table 1)

**80 epochs (convergence efficiency)**:
| Method | rFID | gFID (guided) | gFID (unguided) |
|--------|------|---------------|-----------------|
| REPA [Yu et al. 2024](https://arxiv.org/abs/2410.06940) | 0.61 | 1.42 | 7.90 |
| SFD | 0.26 | 1.30 | 3.53 |
| RAE (DiTDH-XL) | 0.57 | — | 2.16 |
| GAE | 0.44 | 1.48 | 1.82 |
| VTP | 0.36 | 1.44 | 2.62 |
| **PAE (DINOv2)** | **0.26** | **1.27** | **1.80** |

PAE在80 epochs就达到1.27 gFID，比所有representation-native baselines好，甚至超过使用839M params + AutoGuidance的RAE。

**800 epochs (final performance)**:
| Method | gFID (guided) | gFID (unguided) |
|--------|---------------|-----------------|
| SFD | 1.06 | — |
| Send-VAE + REPA | 1.21 | 1.75 |
| RAE (DiTDH-XL) | 1.13 | 1.51 |
| **PAE (DINOv2)** | **1.03** | **1.43** |

**1.03 gFID是ImageNet 256×256的new SOTA**。

### 6.2 Convergence Speed

Figure 1显示PAE比RAE快13× convergence达到comparable quality。这个加速的intuition：PAE的latent manifold已经well-organized，diffusion model不需要"waste" early epochs来learn latent space structure。

### 6.3 Few-Step Sampling (Table 6)

| Steps | gFID | IS |
|-------|------|-----|
| 10 | 1.88 | 277.0 |
| 15 | 1.28 | 282.0 |
| 25 | 1.20 | 289.4 |
| 45 | 1.06 | 296.4 |
| 250 | 1.03 | 296.9 |

PAE在15 steps就达到FAE 250-step的gFID，**16.7× fewer inference steps**。这是MCR的直接效果：locally continuous manifold使得diffusion trajectory更容易被快速traverse。

### 6.4 Ablation: Prior Alignment的互补性 (Table 2a)

| SSR | MCR | SCR | SSC | LPC | GSQ | gFID |
|-----|-----|-----|-----|-----|-----|------|
| × | × | × | 0.18 | 0.320 | 0.19 | 7.18 |
| ✓ | × | × | 0.29 | 0.296 | 0.26 | 2.74 |
| × | ✓ | × | 0.23 | 0.221 | 0.24 | 2.53 |
| × | × | ✓ | 0.21 | 0.286 | 0.39 | 2.63 |
| ✓ | ✓ | × | 0.33 | 0.187 | 0.33 | 2.02 |
| ✓ | × | ✓ | 0.31 | 0.258 | 0.46 | 2.10 |
| × | ✓ | ✓ | 0.24 | 0.176 | 0.45 | 2.08 |
| **✓** | **✓** | **✓** | **0.35** | **0.170** | **0.50** | **1.86** |

关键observations：
1. 每个objective most strongly improves其intended dimension（SSR→SSC, MCR→LPC, SCR→GSQ）
2. 三个objective互补，full model在所有dimension都best
3. 任何两个组合都比single好，但full model显著优于pairwise（1.86 vs 2.02-2.10）

### 6.5 Regularization Strategy对比 (Table 3a)

| Method | gFID | IS |
|--------|------|-----|
| Baseline (no Lp) | 7.79 | 117.2 |
| KL Reg (1e-6 weight) | 5.17 | 132.4 |
| Diff Reg (auxiliary diffusion loss) | 4.22 | 148.3 |
| **Ours (geometry-targeted)** | **1.80** | **218.3** |

Generic regularizers有帮助但远不如geometry-targeted alignment。KL penalty和auxiliary diffusion loss只改善~3-4 gFID，而PAE改善~6 gFID。这证明gain来自**targeted manifold shaping**而非generic regularization。

### 6.6 Cross-Encoder Generalization (Table 8)

| VFM | w/o Lp gFID | w/ Lp gFID | Improvement |
|-----|-------------|------------|-------------|
| DINOv2 | 7.79 | 1.80 | 5.99 |
| SigLIP2 | 6.89 | 2.32 | 4.57 |
| DINOv3 | 6.62 | 1.81 | 4.81 |
| MAE | 7.97 | 3.65 | 4.32 |

Prior alignment在所有4个VFM上都带来huge improvement，证明方法的generality。DINOv2和DINOv3最好，SigLIP2次之，MAE最弱但仍有4.32 improvement。

Figure 6(c)的geometry profile解释了这个ordering：
- **DINO-based**: 最balanced across SSC, LPC, GSQ
- **SigLIP2**: strongest GSQ (semantic)但weaker SSC和LPC
- **MAE**: reasonable SSC但clearly weaker LPC和GSQ

这与VFM的pretraining objective一致：DINO系列self-supervised learning产生spatially和semantically balanced features；SigLIP的contrastive language-image pretraining强化global semantic但weak spatial detail；MAE的reconstruction objective保留spatial structure但semantic相对weak。

## 7. 与Related Work的深层对比

### 7.1 Representation-Guided DiT vs Representation-Native DiT

PAE属于Representation-Native paradigm：通过tokenizer构造representation-rich latent space。

**Representation-Guided**方法（如REPA [Yu et al. 2024](https://arxiv.org/abs/2410.06940), DDT [Wang et al. 2025](https://arxiv.org/abs/2504.05741)）在generator side注入VFM priors，但latent space本身仍然是reconstruction-oriented。这些方法improve how generator models a given space，但not how that space should be constructed。

PAE的insight：**在tokenizer stage就organize manifold，比在generator stage compensate更高效**。Figure 1的13× convergence speedup直接evidence这个claim。

### 7.2 与GAE, AlignTok, RAE的对比

- **GAE** ([Liu et al. 2026](https://arxiv.org/abs/2603.10365)): 使用VFM features作为alignment target，但align to raw frozen features，bottleneck-mismatched
- **AlignTok** ([Chen et al. 2026](https://arxiv.org/abs/2509.25162)): 类似teacher alignment，但同样不explicitly shape manifold geometry
- **RAE** ([Zheng et al. 2025](https://arxiv.org/abs/2510.11690)): 直接使用pretrained encoder features作为latent，semantic强但high-frequency detail弱，且latent dimension高（utilization低）

PAE的关键差异：
1. **Refined targets**而非raw VFM features（bottleneck-matched）
2. **三个complementary manifold objectives**而非single alignment
3. **Explicit geometry shaping**而非implicit inheritance

### 7.3 与Spectral/Structure-Centric Methods的对比

- **EQ-VAE** ([Kouzelis et al. 2025](https://arxiv.org/abs/2502.09509)): scale-equivariant regularization
- **SER** ([Skorokhodov et al. 2025](https://arxiv.org/abs/2502.14831)): low-pass spectral regularization
- **SSVAE** ([Liu et al. 2025](https://arxiv.org/abs/2512.05394)): local correlation regularization

这些methods从spectral或structural角度improve diffusability。PAE的manifold perspective提供了unified language：
- Spectral methods → biasing manifold toward smoother coarse structure
- Structural methods → simplifying local geometry
- PAE → explicit three-dimensional manifold organization

## 8. Design Intuition总结

### 8.1 为什么这个framework work？

PAE的成功源于一个deep insight：**diffusion model的学习难度由latent manifold的geometry决定，而geometry可以explicitly shaped**。

传统tokenizer只优化reconstruction，让manifold geometry emerge implicitly。但implicit emergence受限于reconstruction objective的inductive bias——reconstruction只约束observed data points，对neighborhood和global organization几乎无约束。

PAE的三个regularization分别填补这个gap：
- **SSR**: 约束instance-level spatial topology（reconstruction无法保证token间relation survive bottleneck）
- **MCR**: 约束local neighborhood（reconstruction只约束anchor point）
- **SCR**: 约束global semantic organization（reconstruction无semantic supervision）

### 8.2 为什么sphere manifold重要？

RMS normalization把latent投影到sphere上有三个critical benefits：
1. **Isotropic perturbation**: MCR的random direction sampling在sphere上natural
2. **Channel efficiency**: 去除magnitude redundancy，所有channel被uniformly利用
3. **Diffusion compatibility**: Gaussian noise assumption在sphere上更consistent

### 8.3 为什么cascaded MCR比single perturbation好？

Cascaded design的数学intuition：**local Lipschitz continuity可以通过chain of local consistencies来approximate global continuity**。

如果直接要求large perturbation回到anchor，需要decoder在large neighborhood都robust，这会smooth out detail。Cascaded设计只要求每一步local consistency，允许decoder在medium scale保留更多detail，同时通过transitivity获得large-scale continuity。

这与diffusion的coarse-to-fine generation process天然契合：diffusion也是通过small denoising steps逐步refine。

## 9. Limitations和Future Directions

Paper承认的limitations：
1. 只在ImageNet 256×256验证（higher resolution和diverse domains未测）
2. Fixed-resolution latent diffusion（variable resolution未探索）
3. 依赖refined VFM-derived supervision和handcrafted regularization terms

从我的perspective，几个interesting future directions：
1. **Self-supervised manifold organization**: 能否不用VFM teacher，通过self-supervised objective自然emerge这些properties？
2. **Dynamic token allocation**: 结合ElasticTok ([Yan et al. 2025](https://arxiv.org/abs/2410.08368))或FlexTok ([Bachmann et al. 2025](https://arxiv.org/abs/2502.13967))，在variable token budget下shape manifold
3. **Video extension**: MCR的cascaded design天然适合video的temporal continuity
4. **Multimodal extension**: UniFlow ([Yue et al. 2025](https://arxiv.org/abs/2510.10575))和TokenFlow ([Qu et al. 2025](https://arxiv.org/abs/2412.03069))的unified tokenizer + PAE的manifold organization

## 10. 对社区的 broader impact

PAE提供了一个principled framework来rethink tokenizer design：**tokenizer quality应该用manifold geometry而非reconstruction fidelity来evaluate**。这个perspective可能影响：
1. **Tokenizer benchmarking**: SSC, LPC, GSQ可以作为新的diagnostic metrics
2. **Tokenizer-generator co-design**: 显式考虑manifold properties而非只optimize reconstruction
3. **Theoretical understanding**: manifold hypothesis ([Loaiza-Ganem et al. 2024](https://arxiv.org/abs/2404.02954))在latent diffusion中的concrete operationalization

## References

- [PAE Paper (this work)](https://arxiv.org/abs/2512.19862)
- [RAE - Zheng et al. 2025](https://arxiv.org/abs/2510.11690)
- [REPA - Yu et al. 2024](https://arxiv.org/abs/2410.06940)
- [VA-VAE - Yao et al. 2025](https://arxiv.org/abs/2501.09755)
- [GAE - Liu et al. 2026](https://arxiv.org/abs/2603.10365)
- [AlignTok - Chen et al. 2026](https://arxiv.org/abs/2509.25162)
- [DINOv2 - Oquab et al. 2023](https://arxiv.org/abs/2304.07193)
- [DINOv3 - Siméoni et al. 2025](https://arxiv.org/abs/2508.10104)
- [SigLIP2 - Tschannen et al. 2025](https://arxiv.org/abs/2502.14786)
- [Latent Diffusion - Rombach et al. 2022](https://arxiv.org/abs/2112.10752)
- [DiT - Peebles & Xie 2023](https://arxiv.org/abs/2212.09748)
- [VAE - Kingma & Welling 2013](https://arxiv.org/abs/1312.6114)
- [EQ-VAE - Kouzelis et al. 2025](https://arxiv.org/abs/2502.09509)
- [SER - Skorokhodov et al. 2025](https://arxiv.org/abs/2502.14831)
- [DC-AE - Chen et al. 2025](https://arxiv.org/abs/2410.10733)
- [Manifold Hypothesis Survey - Loaiza-Ganem et al. 2024](https://arxiv.org/abs/2404.02954)
- [LPIPS - Zhang et al. 2018](https://arxiv.org/abs/1801.03924)
- [Spectral Clustering - von Luxburg 2007](https://arxiv.org/abs/0711.0187)
- [AnyUp - Wimmer et al. 2026](https://arxiv.org/abs/2510.12764)
- [VTP - Yao et al. 2026](https://arxiv.org/abs/2512.13687)
- [UniFlow - Yue et al. 2025](https://arxiv.org/abs/2510.10575)
- [TokenFlow - Qu et al. 2025](https://arxiv.org/abs/2412.03069)
- [ElasticTok - Yan et al. 2025](https://arxiv.org/abs/2410.08368)
- [FlexTok - Bachmann et al. 2025](https://arxiv.org/abs/2502.13967)
- [DDT - Wang et al. 2025](https://arxiv.org/abs/2504.05741)
- [CRT - Ramanujan et al. 2025](https://arxiv.org/abs/2412.16326)
- [SSVAE - Liu et al. 2025](https://arxiv.org/abs/2512.05394)
- [SFD - Pan et al. 2025](https://arxiv.org/abs/2512.04926)
- [REPA-E - Leng et al. 2025](https://arxiv.org/abs/2504.10483)
- [Send-VAE - Page et al. 2026](https://arxiv.org/abs/2601.05823)
- [FAE - Gao et al. 2025](https://arxiv.org/abs/2512.07829)
- [SVG - Shi et al. 2025](https://arxiv.org/abs/2510.15301)
- [VFM-VAE - Bi et al. 2025](https://arxiv.org/abs/2510.18457)
- [MAE - He et al. 2021](https://arxiv.org/abs/2111.06377)
- [RMSNorm - Zhang & Sennrich 2019](https://arxiv.org/abs/1910.07467)

---

这篇paper的核心贡献是把"diffusion-friendly latent space"从一个模糊的intuition转化为三个可测量、可优化的manifold properties，并通过targeted regularization explicit shape它们。MCR的cascaded perturbation design尤其elegant，巧妙解决了local continuity regularization与reconstruction fidelity的trade-off。1.03 gFID的SOTA和13× convergence speedup提供了strong empirical evidence that **manifold organization matters more than reconstruction fidelity for latent diffusion**。
