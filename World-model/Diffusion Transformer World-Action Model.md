---
source_pdf: Diffusion Transformer World-Action Model.pdf
paper_sha256: e91dae0327fa6fb1072506ffcd933c8b0a07a60041edbc72044527c141f0a092
processed_at: '2026-08-18T05:41:04-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲这篇 Paper

## 一句话版本

他们想造一个能"看现在 + 知道你要怎么开车 → 预测未来8秒画面"的小模型，结果发现**你怎么评估它，决定了你觉得它好不好**。用老办法评估，死板的回归模型赢；用更合理的办法评估，diffusion model 完胜。这中间的故事就是整篇 paper。

---

## 背景：这事儿为什么难

想象你坐在副驾驶，看到一个路口。你问我"2秒后画面长啥样"。这个问题天然是 multi-modal 的——可能右边来一辆红车，可能来一辆白车，可能没人。future 是一坨可能性，不是一个确定的答案。

传统的 regression model 用 MSE 训练，它会做一件特别蠢的事：**把所有可能性平均掉**。红色车 + 白色车 + 没车，平均一下，得到一辆半透明的灰车。这辆车离任何一个真实 future 都"差不多远"，所以 CosSim 看着还凑合，但实际画面糊得没法看。

Diffusion model 不这么干。它每次 forward pass 采样一个具体的 future——这次是红车，下次可能白车。每个 sample 离 GT 都远（因为它是"某一种"未来，不是"平均"未来），但它属于真实 future distribution。

这就是 paper 的核心 tension：**distortion metric 奖励平均（blurry），distribution metric 奖励 realism**。

参考：[Blau & Michaeli 的 perception-distortion tradeoff 原始 paper (CVPR 2018)](https://openaccess.thecvf.com/content_cvpr_2018/html/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.html)

---

## Step 1: 先搞清楚在哪 latent space 做 prediction

他们试了 6 个 frozen encoder，想看哪个 encode 出来的 feature 最适合做 action prediction（预测 steering 和 accel）。

结果很有意思：**V-JEPA2 用 16 帧视频版本**（rep64）比单帧版本（rep1）steering RMSE 低 40%。

这个 ablation 特别干净——同一个 architecture，同一个 checkpoint family，唯一区别是 video encoder 处理 1 帧还是 16 帧。所以这 40% 完全归因于 temporal context。

**直觉**：单帧 encoder 看不到车在动，看不到弯道曲率怎么变化，它只能从一张静态图猜 action。视频 encoder 见过"车往前开时画面怎么变"，所以它 encode 的 feature 里藏着 ego-motion dynamics。

| Encoder | Steer RMSE |
|---|---|
| V-JEPA2 rep64 (16-frame) | **0.058** |
| V-JEPA2 rep1 (1-frame) | 0.097 |
| DINOv2 | 0.104 |
| CLIP | 0.117 |
| ViT-S/16 supervised | 0.121 |
| VQ-VAE | 0.126 |

还有个小发现：self-supervised（DINOv2, V-JEPA2 rep1）比 supervised ViT 好，说明 self-supervised 学到更 rich 的几何结构。VQ-VAE 最差，因为它优化的是 reconstruction，encode 的是 appearance，不是 dynamics。

参考：[V-JEPA2](https://arxiv.org/abs/2506.09985), [DINOv2](https://arxiv.org/abs/2304.07193)

---

## Step 2: 用 SD-VAE 把画面压到 compact latent

不直接在 pixel space 做 diffusion（太贵），也不在 pooled 384-d vector 上做（丢掉 spatial structure）。用 Stable Diffusion 的 VAE：

- 256×256×3 RGB → 32×32×4 latent（压缩 64× spatial, channel 4）
- patchify（patch size=4）→ 8×8=64 个 tokens，每个 dim 64
- scaling factor 0.18215（standard SD 的）

这样 transformer 在 64 个 token 上做 prediction，计算 tractable，又保留了 spatial structure。

参考：[Latent Diffusion (Rombach et al., CVPR 2022)](https://arxiv.org/abs/2112.10752)

---

## Step 3: DiT Architecture 的关键 choice

### 3.1 adaLN-Zero

每个 transformer block 用 adaptive layer norm：
$$
\gamma, \beta, \alpha = \text{MLP}(\boldsymbol{c}), \quad h = \alpha \odot \text{Attn}(\boldsymbol{\gamma} \cdot \text{LN}(\boldsymbol{x}) + \beta)
$$

- `γ` (gamma): scale
- `β` (beta): shift  
- `α` (alpha): gate，**初始化为 0**

α 初始化为 0 意味着每个 block 一开始就是 identity function，signal 直接 pass through。这让深层网络从"什么都不做"开始，慢慢学习有用的 transformation，训练稳定得多。

`c` 是 conditioning vector，sum 了三个东西：
1. Sinusoidal timestep embedding（diffusion 的 τ）
2. Pooled present latent `z_t`
3. Per-token Fourier action embedding

参考：[DiT paper (Peebles & Xie, ICCV 2023)](https://peebles.github.io/DiT.html)

### 3.2 Fourier Action Embedding

action 是 2D 连续 vector `(steer, accel)`。直接喂给 MLP 学不好高频 mapping（MLP 有 spectral bias，倾向于学低频）。所以用 Fourier features：

$$
\text{FourierEmbed}(a) = \left[\sin(2\pi f_j a), \cos(2\pi f_j a)\right]_{j=1}^{N_f}
$$

- `a = (steer, accel)`: 2D action
- `f_j`: 64 个 learned frequencies per dimension
- 输出 shape `(B, H, D)`：per-token，每个 horizon step 的 embedding 不同

per-token 这个点很重要——每个 prediction step 对应不同 action，adaLN modulation 每步都不同，self-attention 能 exploit per-step temporal structure。

参考：[Tancik et al., Fourier features NeurIPS 2020](https://arxiv.org/abs/2006.10739)

### 3.3 Residual Anchoring

$$
\hat{z}_{t+k} = z_t + \Delta_k(z_t, \{a_{t+i}\}, \tau)
$$

model 预测的不是绝对 latent，是相对于 present `z_t` 的 delta。`z_t` 在所有 H 个位置 broadcast。

好处：early training 时 model 即使啥都不会，也至少能 copy present frame，不会输出 random noise。graceful degradation。

**坏处（后面会讲）**：这个 anchoring 导致了 motion 问题。

---

## Step 4: Diffusion Objective — x0 而非 ε

### 4.1 Forward Process

$$
q(\tilde{z}_\tau | z_0) = \mathcal{N}\left(\sqrt{\bar{\alpha}_\tau} z_0, (1 - \bar{\alpha}_\tau) I\right)
$$

- `z_0`: clean future latent
- `τ`: timestep (1 到 T=1000)
- `ᾱ_τ`: cumulative noise schedule（cosine schedule）
- `√ᾱ_τ`: signal 保留比例
- `(1 - ᾱ_τ)`: noise variance

τ=0 时 ᾱ=1，就是 clean latent；τ=T 时 ᾱ=0，纯 noise。

### 4.2 Loss: x0-prediction

$$
\mathcal{L}_{\text{diff}} = \mathbb{E}_{\tau, \epsilon}\left[\left\| \hat{z}_0(\tilde{z}_\tau, c, \tau) - z_0 \right\|_2^2\right]
$$

- `ẑ_0`: model 直接预测的 clean latent
- `z̃_τ`: noised input
- `c`: conditioning
- `τ`: timestep

**关键 design choice: 预测 x0，不预测 ε（noise）**

这是 paper 的一个核心 finding。在 compact latent space（64 tokens × dim 64 = 4096 dim），ε-prediction 会 collapse 到 near-copy。

**直觉**：ε-prediction 让 model 在高维 Gaussian noise 空间里"猜 noise direction"。在 pixel space（比如 256×256×3 = 196608 dim），相对 noise 的"方向感"强，因为 dimension 多，noise 占主导。但在 compact latent space，signal-to-noise ratio 不同，ε-prediction 学不到有用的东西，model 直接 copy input 算了。

x0-prediction 直接预测目标，信号路径短，supervision 直接。**switching from ε to x0 恢复了 88.5% 的 performance gap**。

参考：[DDPM (Ho et al.)](https://arxiv.org/abs/2006.11239), [Improved DDPM (cosine schedule)](https://arxiv.org/abs/2102.09672)

### 4.3 Classifier-Free Guidance

训练时以 `p=0.1` 把 action embedding 置零，让 model 同时学 conditional `p(action|z)` 和 unconditional `p(z)`。推理时做 guidance（虽然 paper 没明说 scale，推测是标准 CFG）。

参考：[Classifier-Free Guidance (Ho & Salimans)](https://arxiv.org/abs/2207.12598)

### 4.4 DDIM Sampling

推理从 pure Gaussian noise 开始，DDIM 50 步 deterministic refine。deterministic 很重要——同 noise input 产生同 output，这保证了 action controllability 实验的干净性（固定 noise, sweep steering, 看 output 怎么变）。

参考：[DDIM (Song et al., ICLR 2021)](https://arxiv.org/abs/2010.02502)

---

## Step 5: 四个 Ingredients — 全部必要，jointly sufficient

他们做了 hypothesis-driven diagnostic chain：

| Hypothesis | 结论 | 解释 |
|---|---|---|
| H1: capacity | **rejected** | DiT-direct（无 diffusion）match MLP，architecture 不是 bottleneck |
| H2: objective | **confirmed** | ε→x0 恢复 88.5% gap |
| H3: longer horizon | **rejected** | 2Hz driving conditioned on actions 时 posterior 近 unimodal |
| H4: per-token action seq | **partial** | DiT 比 MLP 更受益（+0.007~+0.020 CosSim），self-attention exploit per-step structure |

最后加上 spatial tokens + residual anchoring 后，DiT 在 ViT (+0.020) 和 DINOv2 (+0.023) 上 beat matched-parameter MLP。

**四个 ingredients**：
1. Spatial tokens（64 个 token，有 spatial structure）
2. x0 objective
3. Residual anchoring
4. Sampling matched to target uncertainty（diffusion sampling 匹配 multi-modal future）

---

## Step 6: 核心实验 — Perception-Distortion Frontier

### 6.1 现象

在 SD-VAE pipeline 里跑，Direct regression 和 Diffusion 的对比：

| Model | KID↓ | FID↓ | CosSim↑ |
|---|---|---|---|
| Direct (regression) | 0.375 | 370.8 | 0.471 |
| Diffusion (raw) | 0.294 | 341.9 | 0.233 |
| Interp (α=.5) | 0.084 | 166.6 | 0.316 |
| **Diffusion (calibrated)** | **0.078** | **162.5** | 0.260 |
| VAE-GT ceiling | ≈0 | ≈0 | 1.000 |

看 CosSim：direct 0.471 vs diffusion 0.260，direct 赢。
看 KID：direct 0.375 vs diffusion 0.078，**diffusion 4.8× 赢**。

### 6.2 为什么 metric 差这么多

**CosSim / L2 / SSIM 是 distortion metric**：测量每个 sample 离 GT 多近。regression model 输出 conditional mean，离所有可能 future 都"中等近"，所以 CosSim 高。但它不是任何 plausible future。

**FID / KID 是 distribution metric**：测量 generated frame distribution 和 real frame distribution 的统计距离。在 Inception-v3 feature space 里算。

#### FID 公式

$$
\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)
$$

- `μ_r, Σ_r`: real frames 在 Inception feature space 的 mean vector 和 covariance matrix
- `μ_g, Σ_g`: generated frames 的对应统计量
- `Tr`: trace（矩阵对角线之和）
- `(Σ_r Σ_g)^{1/2}`: 矩阵平方根（via SVD）
- 整体: Frechet distance between two multivariate Gaussians

#### KID 公式

$$
k(x, y) = \left(\frac{1}{d}\langle x, y\rangle + 1\right)^3
$$

KID = squared MMD with polynomial kernel degree 3。

- `x, y`: Inception feature vectors
- `d`: feature dimension
- `<x, y>`: 内积
- MMD: Maximum Mean Discrepancy

KID 在 small N 时比 FID 更 robust（这里只有 150 scenes），所以是 paper 的 primary metric。

参考：[FID (Heusel et al., NeurIPS 2017)](https://arxiv.org/abs/1706.08500), [KID (Binkowski et al., ICLR 2018)](https://arxiv.org/abs/1711.06044)

### 6.3 Deployable Calibration — 让 advantage 落地

Diffusion raw 的 KID 是 0.294，不够好。原因：VAE encoder/predictor 引入了 per-channel color offset（diffusion sample 有轻微 tint）。

他们 estimate 一个 per-channel mean 和 scale shift，**只用 training split** 估计，test time 应用。结果：

- Raw diffusion: KID 0.294
- Calibrated diffusion: KID 0.078
- Oracle（用 test stats，不可 deploy）: KID 0.086

Train calibration 几乎追上 oracle，所以这个 advantage 是 **deployable 的**，不是 test-time artifact。

**直觉**：calibration 就像给 diffusion sample 戴眼镜——它本身已经 capture 了真实 distribution 的 shape，只是 color space 偏了一点。calibration 把这个偏移纠正，不需要 test-time GT。

### 6.4 Frontier 可 scale

2-point probe（3.0M vs 5.4M params，1 seed each）显示 larger model 在每个 diffusion point 都更好（KID 0.078 vs 0.089）。frontier 随 capacity 往右下移。虽然只是 2 点 1 seed，但方向 clean。

---

## Step 7: Motion 问题 — 单 pass 模型不会"往前开"

### 7.1 现象

虽然 diffusion 在 appearance 上好，但它生成的 16 步视频"画面不怎么动"。把 consecutive frame difference 分解：

- **Low-frequency**（Gaussian blur σ=8）：coherent scene motion，ego-driving 引起的整体 shift
- **High-frequency**（residual）：texture variation

结果：
- Diffusion: high-freq 0.98× GT（texture 很好），low-freq 0.44× GT（scene motion 很差）
- Direct regression: low-freq 0.56× GT（blur 反而 capture 更多 mean motion）
- Diffusion 的 image-plane displacement **near zero**！

加了 temporal-difference loss fine-tune 也没用，说明这是 **structural problem**，不是 loss surface 问题。

### 7.2 Diagnosis: Shared-Present Anchor 是元凶

回顾公式 (3):
$$
\hat{z}_{t+k} = z_t + \Delta_k(z_t, \{a_{t+i}\}, \tau)
$$

**每个 future token 都 anchor 到同一个 `z_t`**。model 学到的是"在 present 基础上加点 texture variation"，而不是"让 scene 往前 shift"。

在 2Hz driving 下，每帧间隔 500ms，ego 车开挺远，scene 应该明显往前移。但 shared anchor 让 model 倾向于 re-render current layout，只是换换 texture。

### 7.3 Jump Model — Compact Fix

reparameterize：不预测 16 步 future，而是预测 Δ=4 jump（2s），然后 chain 4 次：

$$
z_{t+4j} = f_\theta\left(z_{t+4(j-1)}, \bar{a}_{t+4(j-1):t+4j}\right), \quad j=1, \dots, 4
$$

- `f_θ`: compact 1.7M params network（n_blocks=2, dim 192），**3× smaller** than 5.4M baseline
- `z_{t+4(j-1)}`: 上一个 jump 的 output（**re-anchor on own prediction**）
- `ā`: mean-pooled Fourier-embedded actions over 4 steps
- `j`: jump step (1 到 4)

关键：每一步 re-anchor 到自己的 prediction，不是总 anchor 到 present。这逼 model 累积 motion。

训练：teacher-forced（用 GT anchor 训）
测试：open-loop chain（用自己 prediction 做 anchor）——这才是真正的 motion fidelity test

### 7.4 Jump Model 结果

- Motion direction correlation: **0.48**（超过 5.4M single-pass baseline 的 0.41）
- Low-frequency motion magnitude: **1.02× GT**（几乎完整 capture motion magnitude）
- 用了 3× 更少参数

**直觉**：motion limitation 是 anchoring design 的锅，不是 capacity 不够。换个 parameterization，小模型也能恢复 motion direction。但 appearance 还 blurry，这是 scale 问题，不是 design 问题。

---

## Step 8: Action Controllability — 模型真的"听"action 吗

### 8.1 实验设计

固定 diffusion noise，sweep steering input（5th 到 95th percentile），看 t+15 帧的 horizontal scene displacement。在 40 个 held-out windows 上测。

### 8.2 结果

- **Diffusion**: Spearman ρ(steer, shift) = **+0.81**，100% sign-correct on 18/40 scenes with detectable shift
- **Direct regression**: ρ = **-0.18**（uncorrelated）

更狠的实验——**inverse-control probe**（non-circular，从预测的 future 反推 steering）：
- Diffusion: 0.67× chance error（能反推）
- Direct: 1.24× chance error（比随机猜还差）

**直觉**：diffusion model 学到的 latent dynamics 里真的 encode 了 action semantics。给它一个 future，它能反推"这是左转还是右转产生的"。regression model 完全没这个能力，因为它输出的是 average future，把 action 信息 average 掉了。

这是 planning-relevant 的 property：如果 world model 连"这个 future 是哪个 action 产生的"都答不出，那它没法支持 downstream planner。

---

## Step 9: 为什么要用 compact scale

5.4M params（vs GAIA-1 5B, Cosmos 多 B）听起来"小"，但这是 feature：

- 5B model 跑一次 ablation 要多少 GPU hour？5.4M 模型可以跑 4 个 hypothesis × 3 seeds × 多个 encoder
- 能 isolate 单个 design factor（objective, anchoring, sampling, metric）
- 四个 ingredients 是 design principle，可 transfer to larger scale
- Scaling probe 初步 evidence 这些 principle 在更大 scale 下 hold

参考：[GAIA-1](https://arxiv.org/abs/2309.17080), [Cosmos](https://arxiv.org/abs/2501.03575)

---

## Intuition Building: 这篇 Paper 真正的 Story

### 三层 Tradeoff 叠加

1. **Encoder level**: temporal context >> single-frame（V-JEPA2 16-frame 比 1-frame 低 40% RMSE）
2. **Objective level**: x0 >> ε in compact latent（88.5% gap recovery）
3. **Evaluation level**: distribution >> distortion（4.8× KID advantage）

这三层都不是"大模型更大 data 就能解决"的问题，是 design choice 问题。每层都有"看起来对、实际错"的陷阱：
- Single-frame encoder 看着够用，但 temporal context 是 dynamics 的载体
- ε-prediction 是 standard choice，但在 compact latent 下 collapse
- Distortion metric 是 AV 领域 default，但它系统性奖励 blurry mean

### Anchor 是 Motion 的开关

shared-present anchor 让 model 偷懒——不用累积 motion，只要 re-render present。re-anchor on own prediction 逼 model 往前走。

这个 finding 特别 actionable：你想让 world model 有真实 motion？别让它总 anchor 到 present，让它 chain 起来，每步 anchor 到上一步的 output。

### Calibration 让 Lab Result 落地

diffusion model 在 raw 状态下 KID 0.294，看起来没那么 impressive。但加一个 train-derived per-channel shift，KID 掉到 0.078，几乎追上 oracle（0.086）。

这意味着 diffusion 的 advantage 不是 test-time trick，是 production-deployable 的。很多 paper 报告 raw metric 就停了，这篇 paper 多走了一步，证明这个 advantage 在 production pipeline 里能拿到。

### Controllability 是 World Model 的试金石

world model 不是 video generator。它必须 encode action → future 的 causal mapping。这篇 paper 用 inverse-control probe 证明 diffusion model 有这个能力，regression 没有。

**直觉**：regression model 输出 average future，average 把 action 信息洗掉了。diffusion model 每次 sample 一个具体 future，这个 future 是由 conditioning action 生成的，所以 action 信息保留在 output 里。这是 generative model 相对 discriminative model 的本质优势——它 model 的是 conditional distribution，不是 conditional expectation。

---

## Web Links 汇总

### 核心 methodology papers
- [DiT (Peebles & Xie, ICCV 2023)](https://peebles.github.io/DiT.html)
- [Latent Diffusion / SD-VAE (Rombach et al., CVPR 2022)](https://arxiv.org/abs/2112.10752)
- [DDPM (Ho et al., NeurIPS 2020)](https://arxiv.org/abs/2006.11239)
- [DDIM (Song et al., ICLR 2021)](https://arxiv.org/abs/2010.02502)
- [Improved DDPM / cosine schedule (Nichol & Dhariwal, ICML 2021)](https://arxiv.org/abs/2102.09672)
- [Classifier-Free Guidance (Ho & Salimans, 2022)](https://arxiv.org/abs/2207.12598)
- [Fourier Features (Tancik et al., NeurIPS 2020)](https://arxiv.org/abs/2006.10739)

### Evaluation metrics
- [Perception-Distortion Tradeoff (Blau & Michaeli, CVPR 2018)](https://openaccess.thecvf.com/content_cvpr_2018/html/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.html)
- [FID (Heusel et al., NeurIPS 2017)](https://arxiv.org/abs/1706.08500)
- [KID (Binkowski et al., ICLR 2018)](https://arxiv.org/abs/1711.06044)

### Encoders benchmarked
- [V-JEPA2](https://arxiv.org/abs/2506.09985)
- [V-JEPA](https://arxiv.org/abs/2404.08471)
- [DINOv2](https://arxiv.org/abs/2304.07193)
- [CLIP](https://arxiv.org/abs/2103.00020)
- [VQGAN](https://arxiv.org/abs/2012.09841)
- [ViT](https://arxiv.org/abs/2010.11929)

### AV world model 对比
- [GAIA-1](https://arxiv.org/abs/2309.17080)
- [Cosmos (NVIDIA)](https://arxiv.org/abs/2501.03575)
- [DriveDreamer](https://arxiv.org/abs/2309.09777)
- [GenAD](https://arxiv.org/abs/2405.09349)
- [DIAMOND](https://arxiv.org/abs/2405.12399)
- [UniSim](https://arxiv.org/abs/2310.06114)

### Dataset
- [nuScenes](https://www.nuscenes.org)

### 项目代码
- [GitHub: latent-world-models-av](https://github.com/dlcv-team/latent-world-models-av)

---

## 最后的直觉总结

这篇 paper 的真正贡献，不是"我们造了一个 SOTA AV world model"，而是"我们用 controlled experiment 拆解了 AV world model 的 design space，发现了三个被 default choice 掩盖的陷阱"。

每个陷阱都是"看起来对、实际错"的：
1. **用 single-frame encoder** —— 省事，但 temporal context 是 dynamics 的载体
2. **用 ε-prediction** —— standard，但在 compact latent 下 collapse
3. **用 distortion metric 评估** —— AV 领域 default，但它系统性奖励 blurry regression mean

每个 fix 都是 paper 主动 diagnose 出来的：
1. 换 V-JEPA2 16-frame encoder
2. 换 x0-prediction
3. 换 FID/KID distribution metric
4. 加 train-derived calibration
5. 换 chain-anchor jump parameterization 恢复 motion

这些 design principle 在 compact scale 下验证，应该 transfer 到 GAIA-1/Cosmos scale。这是 compact scale 研究的价值——能做 controlled ablation，能 isolate factor，能 build intuition。然后再用更大 scale 把这些 principle 放大。

---

# Diffusion Transformer World-Action Model for AV Scene Prediction — 深度技术讲解

## 一、核心问题与 motivation

这篇 paper 的核心 tension 在于：**AV world model 的 evaluation metric 选择会决定哪个 model "看起来更好"**。当用 distortion metrics（CosSim, SSIM, L2）评估时，deterministic regressor 会"赢"，因为它 collapse 到 blurry conditional mean；但是用 distribution metrics（FID, KID）评估时，diffusion model 远更接近真实 frame distribution。这是一个被 AV latent prediction 领域长期忽视的 perception-distortion tradeoff 问题。

参考链接：
- [Perception-Distortion Tradeoff (Blau & Michaeli, CVPR 2018)](https://openaccess.thecvf.com/content_cvpr_2018/html/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.html)
- [DiT (Peebles & Xie, ICCV 2023)](https://peebles.github.io/DiT.html)
- [V-JEPA2 (Bardes et al., 2025)](https://arxiv.org/abs/2506.09985)

---

## 二、整体 Pipeline 详解

### 2.1 Encode-Predict-Decode 架构

```
Front-camera RGB (256×256×3)
        ↓ frozen SD-VAE encoder
Latent grid (32×32×4, scale=0.18215)
        ↓ patchify (patch size=4)
64 spatial tokens (dim=64 each)
        ↓ Anchored VAE DiT (4 blocks, ~5.4M params)
Predicted future latents {z_{t+1}, ..., z_{t+H}}
        ↓ frozen SD-VAE decoder
Rendered frames 256×256
```

输入：
- 当前帧 latent `z_t`（present anchor）
- ego-action 序列 `{a_{t+1}, ..., a_{t+H}}`，每个 action 是 2D vector `a_t = (steer_t, accel_t)`，z-score normalized

输出：
- 未来 H 步的 latent（H ∈ {4, 8, 16}，对应 2s/4s/8s @ 2Hz）

### 2.2 为什么用 SD-VAE latent space？

SD-VAE 把 256×256×3 压到 32×32×4（spatial compression 64×，channel 4），然后 patchify 成 8×8=64 个 tokens。这让 transformer 能在 compact token 序列上做 prediction，避免 pixel-space diffusion 的计算爆炸。

参考：[Latent Diffusion Models (Rombach et al., CVPR 2022)](https://arxiv.org/abs/2112.10752)

---

## 三、Encoder Benchmark — 为什么 temporal context 这么重要

### 3.1 六个 frozen encoders 对比

Table 1 数据：

| Encoder | Steer RMSE↓ | Accel RMSE↓ |
|---|---|---|
| V-JEPA2 rep64 (16-frame) | **0.058** | **0.055** |
| V-JEPA2 rep1 (single-frame) | 0.097 | 0.059 |
| DINOv2-S/14 | 0.104 | 0.072 |
| CLIP ViT-B/32 | 0.117 | 0.067 |
| ViT-S/16 | 0.121 | 0.071 |
| VQ-VAE Tracker | 0.126 | 0.063 |

V-JEPA2 rep64 比 rep1 降低 40% steering RMSE。

### 3.2 关键 insight

这是 controlled ablation：rep64 和 rep1 是**同一 architecture、同一 checkpoint family**，唯一区别是 video encoder 处理 16 帧 vs 1 帧。所以这 40% 的 gap 完全归因于 **temporal context**。

为什么 single-frame encoders 不行？因为它们看不到：
- Frame-to-frame ego-motion patterns
- Lane curvature dynamics
- 驾驶场景的 motion cue（光流、车辆相对位置变化）

self-supervised methods（DINOv2, V-JEPA2 rep1）优于 supervised ViT-S/16 和 CLIP，说明 self-supervised features 捕获更 rich 的 geometric structure。VQ-VAE Tracker 最差，因为 reconstruction-oriented features 编码 appearance，不编码 dynamics。

Acceleration 的 gap（0.055 vs 0.059）比 steering 小，因为 acceleration 在单帧中更可预测（车辆速度相对静态）。

参考：[DINOv2](https://arxiv.org/abs/2304.07193), [CLIP](https://arxiv.org/abs/2103.00020), [VQGAN](https://arxiv.org/abs/2012.09841)

---

## 四、Anchored VAE DiT Architecture 细节

### 4.1 adaLN-Zero Conditioning

公式 (1)：
$$
\gamma, \beta, \alpha = \text{MLP}(\boldsymbol{c}), \quad h = \alpha \odot \text{Attn}(\boldsymbol{\gamma} \cdot \text{LN}(\boldsymbol{x}) + \beta)
$$

变量解释：
- `c`：conditioning vector（sum of timestep embedding + pooled present latent + per-token Fourier action embedding）
- `γ` (gamma)：scale modulation，对 attention output 做 scaling
- `β` (beta)：shift modulation，对 attention output 做 shifting
- `α` (alpha)：gate，**初始化为 0**，让每个 block 一开始是 identity function（关键 trick！让深层网络训练稳定）
- `⊙`：element-wise multiplication（Hadamard product）
- `LN`：LayerNorm
- `Attn`：multi-head self-attention

这个 design 来自 DiT（Peebles & Xie），是 ViT 和 conditional generation 的关键桥梁。

### 4.2 Fourier Action Embedding

公式 (2)：
$$
\text{FourierEmbed}(a) = \left[\sin(2\pi f_j a), \cos(2\pi f_j a)\right]_{j=1}^{N_f}
$$

变量：
- `a = (steer, accel)`：2D action vector
- `f_j`：learned frequencies（每个 dimension 学 64 个）
- `N_f = 64`
- 输出 shape：`(B, H, D)`，H 是 horizon，D 是 model dimension

为什么用 Fourier features？因为 MLP 对 low-dimensional input 学高频函数困难（spectral bias），Fourier features 让网络能表示 action 的非线性 mapping。参考 [Tancik et al., NeurIPS 2020](https://arxiv.org/abs/2006.10739)。

这个 embedding 是 **per-token** 的：每个 horizon step 的 action embedding 不同，所以 adaLN modulation 在每个 prediction step 都不同。这让 transformer 能利用 per-step temporal structure（这是 H4 partially confirmed 的原因）。

### 4.3 Residual Anchoring

公式 (3)：
$$
\hat{z}_{t+k} = z_t + \Delta_k(z_t, \{a_{t+i}\}, \tau)
$$

变量：
- `z_t`：present latent（anchor）
- `Δ_k`：residual function（model 实际预测的）
- `a_{t+i}`：action sequence
- `τ`：diffusion timestep
- `k`：horizon step（1 到 H）
- `z_t` 在所有 H 个 position 上 **broadcast**

为什么 residual？这 stabilizes early training，让 model 优雅地 degrade 到 copying present，而非 random noise。这也是后面 jump model 改进的起点。

---

## 五、Diffusion Objective 和 Sampling

### 5.1 Forward Process

$$
q(\tilde{z}_\tau | z_0) = \mathcal{N}\left(\sqrt{\bar{\alpha}_\tau} z_0, (1 - \bar{\alpha}_\tau) I\right)
$$

变量：
- `z_0`：clean future latent
- `τ`：timestep（1 到 T=1000）
- `ᾱ_τ`：cumulative product of noise schedule（cosine schedule from [Nichol & Dhariwal, ICML 2021](https://arxiv.org/abs/2102.09672)）
- `√ᾱ_τ`：signal scaling
- `(1 - ᾱ_τ)`：noise variance
- `I`：identity covariance

### 5.2 x0-prediction Loss（关键 design choice）

公式 (4)：
$$
\mathcal{L}_{\text{diff}} = \mathbb{E}_{\tau, \epsilon}\left[\left\| \hat{z}_0(\tilde{z}_\tau, c, \tau) - z_0 \right\|_2^2\right]
$$

变量：
- `ẑ_0`：model 预测的 clean future latent
- `z̃_τ`：noised latent
- `c`：conditioning
- `τ`：timestep
- `ε`：sampled Gaussian noise
- `||·||_2^2`：squared L2 norm

**这里用 x0-prediction，而非 ε-prediction**（即不预测 noise）。诊断结果：ε-prediction 在 compact latent space 中会 collapse 到 near-copy，而 x0-prediction 恢复 88.5% 的 performance gap。

直觉解释：在 low-dimensional latent space（64 个 tokens × dim 64 = 4096 dim），直接预测 noise ε 等于在高维 Gaussian 中"猜"noise direction，这很难。而预测 x0 是直接预测目标 latent，信号更直接。在 pixel space 中 ε-prediction 更稳定，因为 pixel dim 极大，相对 noise 的"方向感"更强。

### 5.3 Classifier-Free Guidance

训练时 action dropout `p=0.1`：以概率 p 将 action embedding 置零。这让 model 同时学习 conditional 和 unconditional 分布。推理时用 guidance（虽然 paper 没明确写 guidance scale，应该是标准 CFG）。

### 5.4 DDIM Sampling

推理用 DDIM（[Song et al., ICLR 2021](https://arxiv.org/abs/2010.02502)）50 步 deterministic sampling，从 pure Gaussian noise 开始 refine。这是 deterministic 版本（vs DDPM 的 stochastic），保证同 noise input 产生同 output，对 action controllability 实验重要。

---

## 六、DiT Diagnosis — 四个必要 Ingredients

### 6.1 Hypothesis-Driven Diagnostic Chain

H1（capacity）: **rejected**。DiT-direct（no diffusion）match MLP，所以 architecture 不是 bottleneck。

H2（objective）: **confirmed**。switching from ε- to x0-prediction 恢复 88.5% 的 gap。ε-prediction 在 compact latent regime 下 collapse 到 near-copy。

H3（horizon）: **rejected**。longer horizon 并不 favor DiT；2Hz driving conditioned on logged actions 时 posterior 接近 unimodal。

H4（action-seq）: **partially confirmed**。per-token action-sequence conditioning 让 DiT 比 MLP 受益更多（+0.007 to +0.020 CosSim on 3 encoders）。self-attention 能 exploit per-step temporal structure，而 MLP 不能。

### 6.2 四个 Ingredients（all necessary, jointly sufficient）

1. **Spatial tokens**：64 个 spatial tokens（8×8 grid），让 transformer 有 spatial structure
2. **x0 objective**：直接预测 clean latent，避免 ε collapse
3. **Residual anchoring**：predict Δ from present anchor
4. **Sampling matched to target uncertainty**：diffusion sampling 匹配 multi-modal future distribution

恢复 spatial tokens 后，DiT 在 ViT (+0.020±0.002) 和 DINOv2 (+0.023±0.002) 上 beat matched-parameter MLPs（12M params, 3 seeds）。

---

## 七、Perception-Distortion Frontier — 核心 Finding

### 7.1 为什么 Distortion Metrics 奖励错误的东西

在 SD-VAE latent space 中，appearance ambiguity under point losses 变得 pronounced。考虑 multi-modal future：下一帧可能有车、可能没车，可能在左、可能在右。

**Deterministic regressor** 用 MSE 训练，会 collapse 到 **conditional mean**：平均所有可能的 future，得到一个 blurry image。这个 blurry image 与每个具体 GT 的 CosSim 都"还行"（0.471），但是它不是任何 plausible future。

**Diffusion model** 采样具体 instance：每次 forward pass 采样一个 plausible future。这个 specific sample 与 GT 的 CosSim 较低（0.260），但是它属于真实 frame distribution。

### 7.2 Distribution Metrics

#### FID (Frechet Inception Distance)

公式 (5)：
$$
\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)
$$

变量：
- `μ_r, Σ_r`：real frame distribution 在 Inception-v3 feature space 的 mean 和 covariance
- `μ_g, Σ_g`：generated frame distribution 的 mean 和 covariance
- `Tr`：trace
- `(Σ_r Σ_g)^{1/2}`：matrix square root（计算 via SVD）
- 整体：Frechet distance between two Gaussians in feature space

#### KID (Kernel Inception Distance)

$$
k(x, y) = \left(\frac{1}{d}\langle x, y\rangle + 1\right)^3
$$

KID 是 squared MMD with polynomial kernel（degree 3）。比 FID 更 **robust at small N**（这里 N=150 scenes）。这是 paper 的 primary metric。

变量：
- `x, y`：Inception-v3 feature vectors
- `d`：feature dimension
- `<x, y>`：inner product
- 整体：polynomial kernel of degree 3 with bias 1

### 7.3 Deployable Calibration

VAE encoder/predictor 引入 per-channel offset in predicted latents。他们 estimate 一个 **train-derived calibration**（per-channel mean and scale shift），只用 training split 估计，test time 应用。

结果：train calibration（KID 0.078）几乎 recover post-hoc oracle calibration（KID 0.086），所以这个 advantage 是 **deployable**，不是 test-time artifact。

### 7.4 Frontier 数据（Table 2）

| Model | KID↓ | FID↓ | CosSim↑ |
|---|---|---|---|
| Direct (regression) | 0.375 | 370.8 | 0.471 |
| Diffusion (raw) | 0.294 | 341.9 | 0.233 |
| Interp (α=.5) | 0.084 | 166.6 | 0.316 |
| **Diffusion (calib.)** | **0.078** | **162.5** | 0.260 |
| VAE-GT ceiling | ≈0 | ≈0 | 1.000 |

**4.8× KID improvement**（0.078 vs 0.375）。Latent interpolation（α=0.5 blend direct and diffusion）提供 intermediate operating point。

3 seeds 的 KID 是 0.076±0.005，gap 远超 seed variance。Capacity probe（3.0M vs 5.4M, 1 seed each）显示 larger model 在每个 diffusion point都更好（KID 0.078 vs 0.089），frontier shifts favorably with scale。

参考：[FID (Heusel et al., NeurIPS 2017)](https://arxiv.org/abs/1706.08500), [KID (Binkowski et al., ICLR 2018)](https://arxiv.org/abs/1711.06044)

---

## 八、Motion Fidelity Diagnosis 和 Jump Model

### 8.1 Motion Decomposition

把 consecutive frame difference 分解：
- **Low-frequency**：Gaussian-blurred（σ=8, kernel 31）consecutive L2 → coherent scene structure change（ego-motion 引起的整体 shift）
- **High-frequency**：residual（original minus blurred）consecutive L2 → texture variation

还计算 image-plane displacement（horizontal/vertical profile cross-correlation）。

### 8.2 Single-pass Model 的 Motion 问题

结果（Figure 6）：
- Diffusion: texture 0.98× GT（很好），coherent motion 0.44× GT（很差）
- Regression mean: scene-level motion 0.56× GT（blur 反而 capture 更多 motion，因为 mean shift 包含 average motion）
- Diffusion 的 image-plane displacement **near zero**！

Diagnosis：**shared-present anchor** 是罪魁祸首。公式 (3) 中每个 future token 都 anchor 到同一个 `z_t`，model 倾向于 re-rendering current layout with varied texture，而非 accumulating ego-motion。

### 8.3 Chain-Anchor Jump Model

公式 (6)：
$$
z_{t+4j} = f_\theta\left(z_{t+4(j-1)}, \bar{a}_{t+4(j-1):t+4j}\right), \quad j=1, \dots, 4
$$

变量：
- `f_θ`：compact 1.7M 参数 network（n_blocks=2, dim 192）—— **3× smaller than 5.4M baseline**
- `z_{t+4(j-1)}`：previous jump 的 output（**re-anchor on own prediction**）
- `ā`：mean-pooled Fourier-embedded actions over 4 steps
- `j`：jump step（1 到 4）
- 每 jump 是 Δ=4 transition（2s @ 2Hz）
- 4 步 chain 到达 `z_{t+16}`（8s）

训练：teacher-forced ground-truth anchors
测试：open-loop chain with model's own predictions（真正的 motion fidelity test）

### 8.4 Jump Model 结果

- Motion direction correlation：**0.48**（超过 larger single-pass baseline 0.41）
- Low-frequency motion magnitude：**1.02× GT**（capture 全部 motion magnitude）
- 即使 3× smaller

这证明 limited motion 是 **objective and anchoring choice**，非 capacity limitation。

---

## 九、Action Controllability

### 9.1 实验设计

Sweep steering input across training distribution range（5th to 95th percentile），fixed diffusion noise，measure induced horizontal scene displacement at t+15 on 40 held-out windows。

### 9.2 结果

- **Diffusion model**: Spearman `ρ(steer, shift) = +0.81`，100% sign-correct on 18/40 scenes where shift exceeds detection threshold
- **Direct regression**: `ρ = -0.18`（uncorrelated，39/40 valid）

**Inverse-control probe**（non-circular，predict held-out steering from predicted future）：
- Diffusion: 0.67× chance error
- Direct: 1.24× chance error（worse than random）

这证明 diffusion model 的 latent dynamics 编码 action semantics，可支持 downstream planning。Regression model 完全缺乏这个能力。

---

## 十、Intuition 总结

### 10.1 三层 Tradeoff

1. **Encoder level**: temporal context >> single-frame（V-JEPA2 rep64 vs rep1 的 40% gap）
2. **Objective level**: x0-prediction >> ε-prediction in compact latent（88.5% gap recovery）
3. **Evaluation level**: distribution metrics >> distortion metrics（4.8× KID advantage）

### 10.2 关键 design lesson

- **Anchor design 决定 motion behavior**：shared-present anchor → static rendering；per-step re-anchor → motion recovery
- **Sampling 匹配 uncertainty**：deterministic regression 适合 unimodal posterior；diffusion sampling 适合 multi-modal future
- **Calibration makes it deployable**：train-derived per-channel shift 让 diffusion advantage 从 lab 走向 production

### 10.3 Compact Scale 的价值

5.4M params（vs GAIA-1 5B, Cosmos）能做 controlled ablations：
- 单独 isolate objective、representation、anchoring、sampling 的影响
- 四个 ingredients 可 transfer to larger scale
- Positive scaling probe（3.0M vs 5.4M）初步 evidence

### 10.4 Future Work 方向

- Scale（capacity, data, frame rate）
- Stronger temporal supervision（perceptual losses, autoregressive scheduled sampling）
- Closed-loop evaluation with predicted actions
- Multi-camera setups

---

## 参考链接汇总

1. **DiT**: https://peebles.github.io/DiT.html
2. **V-JEPA2**: https://arxiv.org/abs/2506.09985
3. **Perception-Distortion Tradeoff**: https://openaccess.thecvf.com/content_cvpr_2018/html/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.html
4. **Latent Diffusion (SD-VAE)**: https://arxiv.org/abs/2112.10752
5. **DDIM**: https://arxiv.org/abs/2010.02502
6. **Classifier-Free Guidance**: https://arxiv.org/abs/2207.12598
7. **Improved DDPM (cosine schedule)**: https://arxiv.org/abs/2102.09672
8. **Fourier Features (Tancik)**: https://arxiv.org/abs/2006.10739
9. **FID**: https://arxiv.org/abs/1706.08500
10. **KID**: https://arxiv.org/abs/1711.06044
11. **nuScenes**: https://www.nuscenes.org
12. **GAIA-1**: https://arxiv.org/abs/2309.17080
13. **Cosmos (NVIDIA)**: https://arxiv.org/abs/2501.03575
14. **DIAMOND**: https://arxiv.org/abs/2405.12399
15. **DINOv2**: https://arxiv.org/abs/2304.07193
16. **CLIP**: https://arxiv.org/abs/2103.00020
17. **VQGAN**: https://arxiv.org/abs/2012.09841
18. **项目代码**: https://github.com/dlcv-team/latent-world-models-av

---

这篇 paper 的核心贡献是建立了一个 **metric-aware 的 design framework**：在 compact scale 下 isolate 出哪些 design choice 真正 matter（objective, anchoring, sampling, evaluation metric），并证明这些 choice 的 advantage 在更大 scale 下应该 hold。Perception-distortion frontier 的 characterization 和 jump model 的 motion recovery 是两个最 actionable 的发现，直接指向 production AV world model 的下一步 iteration。
