---
source_pdf: Normalizing Trajectory Models.pdf
paper_sha256: 179d237faa61e3ecb71979bfe2a04ca8e0e5ad04b30d0d8ab4f6e9299f4cf2b7
processed_at: '2026-08-05T22:41:30-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# NTM 用人话讲

## 1. 问题:Gaussian 假设在 few-step 时崩了

先讲清楚 diffusion models 在干嘛。

想象你有一张 clean image $x_0$,你往里加 noise 得到 $x_t$:
$$x_t = (1-t)x_0 + t\epsilon$$

$t=0$ 是 clean,$t=1$ 是 pure noise。生成就是从 $t=1$ 往 $t=0$ 走,每步预测一个 reverse conditional $p(x_s | x_t)$ — 给我 noisy image,告诉我 clean 一点点的 image 分布长啥样。

**Standard diffusion 假设**:这个 $p(x_s | x_t)$ 是 Gaussian。

这个假设在 **step 多** 的时候是对的。你走 50 步,每步跨度小,local 的 transition 确实像 Gaussian。但压到 **4 步**,每步跨度大,真实的 $p(x_s | x_t)$ 是什么样?

看这个积分:
$$p(x_s | x_t) = \int p(x_s | x_t, x_0) p(x_0 | x_t) dx_0$$

- $p(x_s | x_t, x_0)$:给定 clean image $x_0$,这是 Gaussian (forward process 的构造保证)
- $p(x_0 | x_t)$:一个 noisy image 可能对应 **很多** plausible clean images,这个是 multimodal 的

所以 $p(x_s | x_t)$ 是 **mixture of Gaussians** — 你 integrate over 所有 plausible $x_0$,每个贡献一个 Gaussian component。一个 Gaussian 根本 capture 不了。

这就是 paper 说的 "Gaussian bottleneck":few-step 时,单 Gaussian 是 fundamental 限制。

现有 few-step 方法怎么处理?
- **Consistency models** (https://arxiv.org/abs/2303.01469), **DMD** (https://arxiv.org/abs/2311.18828):学 deterministic mapping,扔掉 likelihood
- **DDGAN** (https://arxiv.org/abs/2206.03511):用 GAN 替代 Gaussian,但 mode-seeking + unstable

**Gap**:没人能在 few-step 下保留 exact likelihood。NTM 填这个 gap。

---

## 2. 核心 Insight:Warp 空间让 Gaussian 重新 Work

NTM 的核心想法用一句话讲:**找一个 invertible mapping $f_T$ 把 $x_s$ warp 到一个 latent space $u_s$,在那个 space 里 conditional 是 Gaussian**。

### 2.1 类比:你有一堆复杂的分布,与其直接建模,不如变换坐标系

想象 $p(x_s | x_t)$ 在 x-space 是个 weird 形状 (multimodal, heavy tail)。你不直接建模它。你找一个 nonlinear 的坐标系变换 $u_s = f_T(x_s)$,在 u-space 里分布变成 Gaussian。

这就够了?不够 — 你还需要 $f_T$ **invertible**,这样:
1. 你能算 likelihood (change of variables)
2. 生成时能 invert 回 x-space

这正是 normalizing flow 的核心思想 (https://arxiv.org/abs/1410.8516, https://arxiv.org/abs/1605.08803)。

### 2.2 NTM 的两个组件

NTM 把每一步 reverse conditional 拆成两个组件:

**Transporter** $f_T$:spatial autoregressive flow,shallow (8 layers),把 $x_s, x_t$ warp 到 $u_s, u_t$。这是 invertible 的,有 tractable Jacobian。

**Predictor** $f_P$:deep Transformer (24 layers),从 noisy $u_t$ 预测 $\hat{u}_s$。这是 affine coupling:
$$\hat{u}_s = \mu_P(u_t, t, s, y) + \sigma_P(u_t, t, s, y) \cdot z$$

- $\mu_P$:predicted mean
- $\sigma_P$:predicted scale
- $z \sim \mathcal{N}(0, I)$:noise
- $y$:condition (text, class label)

Predictor 在 u-space 是 Gaussian,但因为 $f_T$ 是 nonlinear invertible,x-space 里是 non-Gaussian:
$$p(x_s | x_t) = \mathcal{N}(f_T(x_s); \mu_P, \text{diag}(\sigma_P^2)) \cdot |\det J_{f_T}(x_s)|$$

那个 $|\det J_{f_T}|$ 就是 change of variables 的 Jacobian 修正。

### 2.3 关键 Insight:Conditional 比 Marginal 简单

这是 NTM 最深的 insight,值得多讲几句。

STARFlow (https://arxiv.org/abs/2505.20041) 直接建模 $p(x)$ — 整个 image distribution。这需要 deep spatial AR flow (256+ sequential steps),因为 $p(x)$ 极其复杂。

NTM 不建模 $p(x)$,而是建模 $p(x_s | x_t)$ — **conditional on 一个 noisy version of the image**。

为什么这简单多了?因为你 condition on $x_t$,你已经知道 image 大概长啥样 (颜色、layout、object 位置都隐含在 $x_t$ 里)。$p(x_s | x_t)$ 只需要建模 "given 这个 noisy image,clean 一点点的版本分布在哪" — 这比建模整个 image distribution 简单得多。

所以 NTM 的 transporter 可以 shallow (8 layers vs STARFlow 的 24+ layers × multiple blocks × 256 AR steps),因为每个 step 的 conditional 比较简单。

### 2.4 Division of Labor

这就形成了一个优雅的 division of labor:

| 组件 | 深度 | 作用 |
|------|------|------|
| Predictor | Deep (24 layers) | Cross-timestep reasoning,捕获 denoising signal 主体 |
| Transporter | Shallow (8 layers) | Local non-Gaussian residual within each step |

Predictor 是 deep 的因为它要 reasoning across timesteps — 这需要大容量。Transporter 是 shallow 的因为它只处理 local residual — 每个步骤内的 non-Gaussian 结构有限 (因为 $x_t$ 已经 constrain 了)。

Predictor 用 non-causal full attention,所以 spatial positions 并行处理 — inference 高效。Transporter 是 spatial AR,但 shallow,所以 AR cost 可控。

---

## 3. 公式直觉

### 3.1 NTM Loss

最关键的公式是 Eq (3.3):
$$\mathcal{L}_{\text{NTM}} = -\log p(x_s | x_t) = -\log p_P(u_s | u_t) - \log|\det J_{f_T}|$$

- 第一项 $-\log p_P(u_s | u_t)$:u-space 里的 Gaussian NLL
- 第二项 $-\log|\det J_{f_T}|$:transporter 的 Jacobian 修正,account for the warping

这是 **exact NLL** of $p(x_s | x_t)$,通过 change of variables 推出。

Expanded form (Eq 3.4):
$$\mathcal{L}_{\text{NTM}} = \sum_{k=1}^T \left[\frac{1}{2}\|z_k\|^2 + \sum_n \left(\log \sigma_P^{(k,n)} + \sum_\ell \log \sigma_T^{(k,\ell,n)}\right)\right]$$

- $z_k$:第 k 步的 noise latent
- $\sigma_P^{(k,n)}$:第 k 步,位置 n 的 predictor scale
- $\sigma_T^{(k,\ell,n)}$:第 k 步,第 $\ell$ 个 transporter block,位置 n 的 scale
- $\frac{1}{2}\|z_k\|^2$:Gaussian prior 的 NLL
- $\log \sigma$ 项:Jacobian 修正

### 3.2 Forward Process 保持 Marginals

Eq (2.3) 定义 Markov forward:
$$x_t = \alpha_{s,t} x_s + \sigma_{s,t} \epsilon$$

- $\alpha_{s,t} = (1-t)/(1-s)$:缩放系数
- $\sigma_{s,t} = \sqrt{t^2 - \alpha_{s,t}^2 s^2}$:noise std

Proposition 1 (Appendix A.4) 证明这个 transition 保持 marginal $q(x_t | x_0) = \mathcal{N}((1-t)x_0, t^2 I)$。关键 trick:
$$\mathbb{E}[x_t | x_0] = \alpha_{s,t} (1-s) x_0 = \frac{1-t}{1-s} (1-s) x_0 = (1-t) x_0 \checkmark$$
$$\text{Var}[x_t | x_0] = \alpha_{s,t}^2 s^2 + \sigma_{s,t}^2 = \alpha_{s,t}^2 s^2 + t^2 - \alpha_{s,t}^2 s^2 = t^2 \checkmark$$

这让你构造一个 correlated Markov trajectory,而不是 independent marginal samples。

---

## 4. Architecture 细节

Table 4 给出 architecture spec:

| 组件 | From scratch | Finetuned |
|------|--------------|-----------|
| Hidden dim | 3072 | 3072 |
| Blocks | 3 | 3 |
| Layers per block | [4, 4, 24] | [4, 4, 24] |
| Transporter | Blocks 1-2 (4 layers each) | Blocks 1-2 (4 layers each) |
| Predictor | Block 3 (24 layers) | FLUX.2-klein (4B) |
| Patch size | 1 | 2 |
| Position encoding | 2D RoPE | 2D RoPE |
| Transporter scan | Alternating L→R, R→L | Alternating L→R, R→L |

### 4.1 Transporter

- 2 个 TarFlow-style (https://arxiv.org/abs/2412.06329) causal AR flow blocks
- 每个 block 4 层
- Alternating scan directions (L→R 然后 R→L) 改善 spatial mixing
- At highest noise level ($t \approx 1$),transporter 被 skip (identity transform),因为 input 接近 isotropic Gaussian,AR coupling 没信息

### 4.2 Predictor

- From scratch: 24 层 non-causal full attention Transformer,处理 all timestep levels in parallel
- Finetuned: wraps pretrained FLUX.2-klein (4B) backbone,用 zero-init projection output residual corrections
- Timestep conditioning: additive sinusoidal embeddings
- Text conditioning: Qwen-2.5-VL encoder (https://arxiv.org/abs/2502.02381)

### 4.3 与 STARFlow / Flow Matching 的对比 (Figure 8)

这是 spectrum:
- **STARFlow**:1 个 step,deep spatial AR (256+ steps within that step) 建模 $p(x)$
- **Flow Matching**:many steps (50-250),每步 Gaussian,建模 velocity field
- **NTM**:few steps (4-8),每步 shallow NF + cross-timestep deep predictor,建模 $p(x_s | x_t)$

NTM 是 STARFlow 和 flow matching 的 interpolation。

---

## 5. Finetuning:从 Pretrained Model 启动

这是 paper 的一个 major practical contribution。

### 5.1 问题

你有一个 pretrained flow matching model (FLUX.2-klein, https://blackforestlabs.ai/),它在 x-space 预测 velocity $v$。你想把它变成 NTM。

### 5.2 初始化策略

**Step 1**:用 pretrained velocity derive Gaussian posterior
$$\hat{u}_0 = u_t - t \cdot v \quad \text{(预测 clean sample)}$$

**Step 2**:用 closed-form coefficients 算 posterior
$$\mu_{\text{post}} = A(t,s) u_t + B(t,s) \hat{u}_0, \quad \sigma_{\text{post}} = C(t,s)$$

这些 coefficients 是 forward process 的 reverse posterior (Proposition 2, Appendix A.5):
$$A(t,s) = \frac{s^2(1-t)}{t^2(1-s)}, \quad B(t,s) = \frac{(t-s)(t+s-2ts)}{t^2(1-s)}, \quad C(t,s)^2 = \frac{s^2(t-s)(t+s-2ts)}{t^2(1-s)^2}$$

- $A(t,s)$:$u_t$ 的权重
- $B(t,s)$:$\hat{u}_0$ 的权重
- $C(t,s)$:posterior std

Table 3 给 4-step schedule 的数值:

| Step | t | s | A(t,s) | B(t,s) | C(t,s) |
|------|---|---|--------|--------|--------|
| 1 | 1.000 | 0.754 | 0.140 | 0.614 | 0.371 |
| 2 | 0.754 | 0.509 | 0.271 | 0.496 | 0.362 |
| 3 | 0.509 | 0.263 | 0.470 | 0.362 | 0.297 |
| 4 | 0.263 | 0.020 | 0.948 | 0.049 | 0.049 |

注意最后一步 (t=0.263 → s=0.020):A 接近 1, B 接近 0, C 很小 — 几乎 identity,只需要去小量 noise。

**Step 3**:Initialize predictor 为这个 posterior,加 zero-init residual correction
$$\mu_P = \mu_{\text{post}}, \quad \sigma_P = \sigma_{\text{post}} \cdot \exp(\delta_\sigma), \quad \delta_\sigma = \text{proj}_{\text{out}}(h)$$

$\text{proj}_{\text{out}}$ 初始化为 0,所以 $\sigma_P = \sigma_{\text{post}}$ at init。

**Step 4**:Initialize transporter 为 identity ($f_T = \text{id}$)

**结果**:at initialization,整个模型 = pretrained Gaussian posterior in x-space。你从 pretrained quality 启动,然后 NLL objective 慢慢 drift 到 non-Gaussian。

### 5.3 Mean-Alignment Auxiliary Loss

问题:NLL objective 会让 predictor 和 transporter 一起 drift,可能 unstable。

解决:加 auxiliary loss align predictor mean with frozen pretrained prediction
$$\mathcal{L}_{\text{aux}} = \|\mu_P - \mu_{\text{FM}}\|^2$$

- $\mu_{\text{FM}}$:frozen pretrained backbone 直接从 x-space 预测的 mean
- Total loss: $\mathcal{L} = \mathcal{L}_{\text{NTM}} + \lambda \mathcal{L}_{\text{aux}}$, $\lambda$ anneal (开始 full,后来 decay)

**作用** (Appendix A.3):
1. 防止 catastrophic forgetting from pretrained solution
2. 稳定 u-space coordinate system,让 transporter 学有意义的 spatial mapping
3. 让 non-Gaussian 表达来自 $\sigma_P$ 和 $f_T$,而 $\mu_P$ 保持接近 pretrained

**Ablation** (Figure 7a, 7b):
- $\lambda = 0$:finetuning 早期 diverge
- $\lambda > 0$:稳定

实验用 $\lambda = 2.5$,MSE variant,cosine decay。

---

## 6. Trajectory Score Denoising — 重大创新

### 6.1 背景

TarFlow 和 STARFlow 有个 trick:因为 natural images 在 low-dim manifold 上,normalizing flow 需要数据 dense,所以加小 noise 然后用 score function denoise:
$$\hat{x}^{\text{den}} = \hat{x} - \Sigma \nabla_{\hat{x}} \log p(\hat{x})$$

这是 per-sample denoising,独立处理每个 sample。

### 6.2 NTM 的扩展

NTM 有个 natural advantage:生成的 trajectory $\hat{x} = (\hat{x}_{t_0}, \ldots, \hat{x}_{t_T})$ **本身就是 noisy sequence** (from Markov forward process),不需要加 noise!

更重要的:trajectory elements 是 **correlated** 的 (Markov property),不是 independent。NTM loss 给 joint score over 整个 trajectory。

Eq (3.5):
$$\hat{x}^{\text{den}} = \frac{1}{1-t}\left(\hat{x} - S \cdot \nabla_{\hat{x}} \mathcal{L}_{\text{NTM}}\right)$$

- $\hat{x}$:整个 trajectory
- $S$:trajectory covariance matrix
- $\nabla_{\hat{x}} \mathcal{L}_{\text{NTM}}$:NTM loss 的 gradient,这是 joint score
- $1/(1-t)$:normalize 到 clean domain

### 6.3 Covariance Matrix $S$

Proposition 3 (Appendix A.6) 推导:
$$[S]_{ij} = \frac{\min(t_i, t_j)^2 (1-\max(t_i, t_j))}{1-\min(t_i, t_j)}$$

推导:
$$x_{t_j} = \alpha_{t_i, t_j} x_{t_i} + \sigma_{t_i, t_j} \epsilon$$
$$\text{Cov}(x_{t_i}, x_{t_j} | x_0) = \alpha_{t_i, t_j} \cdot \text{Var}(\xi_i) = \frac{1-t_j}{1-t_i} \cdot t_i^2 \cdot I$$

- 对角元 $S_{ii} = t_i^2$:第 i 步的 noise variance
- 非对角元:cross-timestep covariance

### 6.4 为什么这比 per-sample Denoising 强?

$S$ 是 dense matrix。修正一个 timestep 的 error 会 **propagate 到所有其他 timestep**,因为它们 correlated。

Per-sample denoising 只用 diagonal ($\text{Var}(x_{t_i}) = t_i^2$),没有 cross-timestep 信息。

Trajectory denoising 用 full covariance matrix,所以你用 **整个 trajectory 的信息** refine 每一个点。这是 massive efficiency gain。

### 6.5 Learned Denoiser:Self-Distillation

Test-time backprop-based denoising 慢。NTM 的 trick:训练 lightweight denoiser $g_\phi$ amortize 这成 single forward pass。

$$\mathcal{L}_{\text{den}} = \|g_\phi(u_{t_0}, y) - \hat{x}_0^{\text{den}}\|^2$$

- $g_\phi$:non-causal attention Transformer
- Input:predictor output $u_{t_0}$ at cleanest level + text y
- Target:score-based denoising 的输出
- Training:MSE,post-train after main model converges

因为 trajectory 是 Markov,$u_{t_0}$ 包含 deterministic predict clean output 所需的所有信息。

**Table 2**:
| Method | img/s ↑ | LPIPS ↓ |
|--------|---------|---------|
| Full NF + Traj. denoise | 0.20 | - |
| Predictor + Denoiser | 1.88 | 0.121 |

**9× speedup**,高保真 (LPIPS 0.121)。

这是 **self-distillation**:NTM 用自己的 score function 教 denoiser,不需要额外数据。

---

## 7. 实验结果

### 7.1 Text-to-Image (Table 1)

| Type | Method | GenEval↑ | DPG↑ |
|------|--------|----------|------|
| DM | SDXL (50 steps) | 0.55 | 74.65 |
| DM | SD3-Medium (50 steps) | 0.62 | 84.08 |
| DM | FLUX.1-dev (50 steps) | 0.66 | 83.84 |
| DM | Janus-Pro-7B | 0.80 | 84.19 |
| DM | HiDream-I1-Full | 0.83 | 85.89 |
| DM | Seedream 3.0 | 0.84 | 88.27 |
| DM | Qwen-Image | 0.87 | 88.32 |
| DM | Nucleus-Image | 0.87 | 88.79 |
| NF | STARFlow (256 AR steps) | 0.56 | - |
| **NF** | **NTM (scratch, 256×256, 4 steps)** | **0.82** | **79.64** |
| **NF** | **NTM (finetune, 512×512, 4 steps)** | **0.76** | **83.38** |

关键观察:
- **NTM from scratch 4 步** 达到 GenEval 0.82,显著超过 STARFlow (0.56, 256 AR steps)
- 接近强 diffusion baselines (FLUX.1-dev 0.66, Janus-Pro 0.80)
- **NTM 是唯一保留 exact likelihood 的方法**
- Finetune 版本在 512×512 达到 DPG 83.38,接近 FLUX.1-dev (83.84, 50 steps) — 但只用 4 步

### 7.2 Class-Conditional ImageNet (Table 6)

| Method | Type | #Params | Steps | FID↓ |
|--------|------|---------|-------|------|
| DiT-XL/2 | DM | 675M | 250 | 2.27 |
| SiT-XL | DM | 675M | 250 | 2.06 |
| VAR | AR | 2.0B | 10 | 1.73 |
| STARFlow (FAE) | NF | 1.4B | 256 | 2.67 |
| **NTM** | **NF** | **1.4B** | **4** | **3.83** |
| **NTM** | **NF** | **1.4B** | **8** | **3.24** |
| **NTM** | **NF** | **1.4B** | **16** | **2.80** |

- NTM 16 步 FID 2.80,接近 STARFlow 256 步 FID 2.67
- **只用 NLL objective,没有 adversarial / perceptual losses**
- Step 数从 256 降到 16,质量损失很小

### 7.3 Ablation: Transporter 作用 (Figure 2)

- 没有 transporter,flow matching 4 步产生严重模糊
- 有 transporter,4 步恢复 50 步质量
- Invertible mapping 提供 latent space 让 affine predictor 变 expressive

### 7.4 Ablation: T=1 失败 (Figure 9, Section 5)

T=1 时输出严重 degraded。为什么?

T=1 时整个 non-Gaussian structure 必须由 shallow transporter (8 layers) 单独捕获。Predictor 退化为 single-step Gaussian coupling。这容量不够 — 这本质是 STARFlow-like 架构但 transporter 远浅于 STARFlow (24+ layers × multiple blocks)。

**Implication**:NTM sweet spot 是 T=4-8。要 push 到 single-step exact-likelihood generation,需要 fundamentally 不同 architecture (adaptive-depth transporters, progressive capacity allocation)。

### 7.5 Multi-Trajectory Training (Figure 6)

Finetuned 模型用不同 T 训练,与 FLUX (50 steps) baseline 比较:
- T=4:最佳 quality-speed trade-off for finetuning
- T=8, T=16:更细粒度 denoising,但 inference 更慢

---

## 8. 训练细节

### 8.1 Algorithm 1 (Training)

```
1. Sample per-example minimum noise: t_min ~ Uniform[t_min_lo, t_min_hi]
2. Compute shifted timestep schedule: (t_0, ..., t_T) with t_0 = t_min
3. Forward trajectory: x_{t_{k+1}} = α_{t_k, t_{k+1}} x_{t_k} + σ_{t_k, t_{k+1}} ε_k
4. Transporter: u_{t_k} = f_T(x_{t_k}), accumulate log|det J_{f_T}|
5. Predictor: (μ_P, σ_P) = DeepBlock(u_{t_{k+1}}, t_{k+1}, t_k, y)
6. z_k = (u_{t_k} - μ_P) / σ_P
7. NTM loss: L_NTM = Σ [½||z_k||² + log σ_P] - log|det J_{f_T}|
8. (Optional) FM aux loss: L_aux = λ ||μ_P - μ_FM||²
9. Update θ via ∇(L_NTM + L_aux)
```

### 8.2 Algorithm 2 (Sampling)

```
1. Sample initial noise: û_{t_T} ~ N(0, I)
2. For k = T, T-1, ..., 1:
   - z_k ~ N(0, I)
   - (μ_P, σ_P) = DeepBlock(û_{t_k}, t_k, t_{k-1}, y)
   - (Optional CFG)
   - û_{t_{k-1}} = z_k · σ_P + μ_P
3. Transporter inverse: x̂_{t_0} = f_T^{-1}(û_{t_0}) (sequential AR with KV-cache)
4. (Optional) Self-refinement or learned denoiser
5. Decode: image = VAE.decode(x̂_{t_0})
```

### 8.3 Hyperparameters (Table 5)

| | From Scratch (ImageNet) | Finetuned |
|---|---|---|
| Optimizer | AdamW (0.9, 0.95) | AdamW (0.9, 0.95) |
| Weight decay | 10^-4 | 10^-4 |
| Peak LR | 10^-4 | 5×10^-5 |
| LR schedule | Cosine with warmup | Cosine with warmup |
| Precision | bfloat16 | bfloat16 |
| Distributed | FSDP2 | FSDP2 |
| Steps (T) | 4 | 4 |
| t_min range | Uniform[0.0, 0.05] | Uniform[0.0, 0.05] |
| CFG dropout | 10% | 10% |
| FM aux λ | - | 2.5 |
| FM aux type | - | MSE |
| λ annealing | - | Cosine decay |
| Batch size | 1024 (64 H100) | 1024 (64 H100) |

### 8.4 CFG (Appendix C.5)

Logits-guided CFG,作用于 coupling parameters:
$$s = \left(\frac{\sigma_c}{\sigma_u}\right)^2 \text{ (clipped to [0,1])}$$
$$\sigma_{\text{eff}} = \frac{\sigma_c}{\sqrt{1 + w - w \cdot s}}$$
$$\mu_{\text{eff}} = \frac{(1+w)\mu_c - w \cdot s \cdot \mu_u}{1 + w - w \cdot s}$$

- $w$:guidance scale
- 对应 $(1+w)\log p_c - w \log p_u$ 应用到 Gaussian coupling
- 同时调整 mean 和 scale,比 standard linear guidance (只改 mean) 更合理

### 8.5 Timestep Schedule (Appendix C.4)

Shifted schedule (SD3, https://arxiv.org/abs/2403.03206):
$$\sigma_k = \frac{e^\mu}{e^\mu + 1/\tilde{\sigma}_k - 1}, \quad \mu = 0.5 + 0.65 \cdot \frac{L_{\text{seq}} - 256}{4096 - 256}$$

长序列用更大 shift,适应分辨率。

---

## 9. 用人话总结 NTM 的故事

1. **问题**:Diffusion models few-step 时不 work,因为 Gaussian 假设崩了 — 真实 reverse conditional 是 mixture of Gaussians

2. **Insight**:与其直接建模 non-Gaussian,不如 warp 空间让 Gaussian 重新 work。找一个 invertible mapping (transporter) 把 $x_s$ 变到 $u_s$,在 u-space 里 conditional 是 Gaussian

3. **关键 trick**:因为 condition on $x_t$ 已经 constrain 了 plausible images,conditional $p(x_s | x_t)$ 比 marginal $p(x)$ 简单,所以 transporter 可以 shallow (8 layers vs STARFlow 的 deep AR)

4. **架构**:Deep predictor (24 layers) 处理 cross-timestep reasoning,shallow transporter 处理 local non-Gaussian residual — division of labor

5. **Likelihood**:Invertibility 保证 exact log-likelihood via change of variables,$\mathcal{L}_{\text{NTM}} = -\log p_P(u_s|u_t) - \log|\det J_{f_T}|$

6. **Finetuning**:从 pretrained flow matching model 启动,identity init + zero-init residual + mean-alignment aux loss,稳定训练

7. **Trajectory denoising**:生成的 trajectory 是 correlated Markov sequence,$S$ matrix 编码 cross-timestep covariance。修正一个 timestep 的 error propagate 到所有 timestep,比 per-sample denoising 更有效。可以 self-distill 成单 forward pass 的 denoiser,9× speedup

8. **结果**:T2I GenEval 0.82 (4 steps, from scratch),接近 SOTA diffusion baselines,远超 prior NF (STARFlow 0.56)。**唯一保留 exact likelihood 的 few-step 方法**

9. **限制**:T=1 失败是 capacity constraint。Sweet spot 是 T=4-8。Push 到 single-step exact-likelihood 需要 fundamentally 不同 architecture

10. **意义**:NTM 在 STARFlow (1 step, deep AR) 和 flow matching (many steps, Gaussian) 之间找到 sweet spot,提供 principled framework 让 exact-likelihood generation 走向 few-step regime

---

## 10. 相关工作链接

**Normalizing Flows**:
- NICE: https://arxiv.org/abs/1410.8516
- RealNVP: https://arxiv.org/abs/1605.08803
- Glow: https://arxiv.org/abs/1807.03039
- TarFlow: https://arxiv.org/abs/2412.06329
- STARFlow: https://arxiv.org/abs/2505.20041

**Diffusion/Flow Matching**:
- DDPM: https://arxiv.org/abs/2006.11239
- Score SDE: https://arxiv.org/abs/2011.13456
- Flow Matching: https://arxiv.org/abs/2210.02747
- Rectified Flow: https://arxiv.org/abs/2209.03003
- SD3: https://arxiv.org/abs/2403.03206

**Few-step Generation**:
- Progressive Distillation: https://arxiv.org/abs/2202.00509
- Consistency Models: https://arxiv.org/abs/2303.01469
- DMD: https://arxiv.org/abs/2311.18828
- Latent Consistency Models: https://arxiv.org/abs/2310.04378
- DDGAN: https://arxiv.org/abs/2206.03511
- Mean Flows: https://arxiv.org/abs/2505.13447
- FlowMaps: https://arxiv.org/abs/2505.18825
- NFM: https://arxiv.org/abs/2603.09014

**Self-supervised Learning** (predictor-encoder 类比):
- BYOL: https://arxiv.org/abs/2006.07733
- JEPA: https://arxiv.org/abs/2301.08243
- VICReg: https://arxiv.org/abs/2105.04906
- DINO: https://arxiv.org/abs/2104.14294

**T2I Baselines**:
- SDXL: https://arxiv.org/abs/2307.01952
- PixArt-α: https://arxiv.org/abs/2310.00426
- FLUX.1: https://blackforestlabs.ai/
- Janus-Pro: https://arxiv.org/abs/2501.17811
- HiDream-I1: https://arxiv.org/abs/2503.04287
- DART: https://arxiv.org/abs/2410.08159
- Qwen-Image: https://arxiv.org/abs/2505.20704

**Evaluation**:
- GenEval: https://arxiv.org/abs/2310.17525
- DPG-Bench: https://arxiv.org/abs/2403.05135

**Text Encoder**:
- Qwen-2.5-VL: https://arxiv.org/abs/2502.02381

**Latent Space**:
- FAE: https://arxiv.org/abs/2512.07829

---

# Normalizing Trajectory Models (NTM) 深度讲解

## 1. 问题动机:Gaussian Bottleneck

这篇 paper 直击 diffusion models 和 flow matching models 在 few-step generation 时的一个 fundamental bottleneck。让我先解释清楚这个 bottleneck 的本质。

Diffusion 和 flow matching 的核心假设是:reverse conditional $p(x_s | x_t)$ 在 step size 小的时候近似 Gaussian。当 step 数多 (50-250 步) 时,每一步跨度小,这个 Gaussian 近似准确。但压缩到 4 步时,每步跨度大,真实的 $p(x_s | x_t)$ 变成 mixture of Gaussians,是 multimodal 且 heavy-tailed 的。

为什么是 mixture of Gaussians?看 Eq (2.4):
$$p(x_s | x_t) = \int p(x_s | x_t, x_0) p(x_0 | x_t) dx_0$$

- $p(x_s | x_t, x_0)$:给定 clean image $x_0$ 的 posterior conditional,这是 Gaussian (Markov forward process 的构造保证)
- $p(x_0 | x_t)$:posterior over clean images,这个是 complex multimodal 的 (因为一个 noisy $x_t$ 可能对应多种 plausible clean images)
- 积分后:$p(x_s | x_t)$ 是 mixture of Gaussians,单 Gaussian 无法 capture

现有 few-step 方法都牺牲了 likelihood framework:
- **Distillation methods** (progressive distillation, DMD, consistency models):学习 deterministic mapping,没有 tractable density
- **DDGAN** (Xiao et al., 2022, https://arxiv.org/abs/2206.03511):用 GAN 代替 Gaussian,但 mode-seeking + training instability + 不可扩展

**Gap**:没有任何方法能在 few-step generation 的同时保留 exact likelihood。NTM 填补这个 gap。

---

## 2. 核心 Insight:Warp the Space so Gaussian Works

NTM 的核心 insight 非常优雅:**学一个 latent space,在这个 space 里 reverse conditional 足够简单,可以用 Gaussian 建模,同时通过 invertibility 保证 exact log-likelihood**。

这本质上是 normalizing flow 思想的 conditional 版本。让我详细讲:

### 2.1 Predictor-Encoder 架构 + Invertibility = Normalizing Flow

NTM 的架构像一个 self-supervised representation learning 的 predictor-encoder (类似 BYOL https://arxiv.org/abs/2006.07733, JEPA https://arxiv.org/abs/2301.08243, VICReg https://arxiv.org/abs/2105.04906):

- **Transporter** $f_T$:把 $x_s$ 和 $x_t$ 都映射到 u-space,得到 $u_s = f_T(x_s, s)$ 和 $u_t = f_T(x_t, t)$
- **Predictor** $f_P$:从 noisy $u_t$ 和 noise $z$ 生成 $\hat{u}_s = f_P(u_t, z, y)$

关键区别:**transporter 是 invertible, same-dimension 的 mapping,而不是 compressive encoder**。这个 invertibility constraint 把 representation learning objective 变成了 normalizing flow 的 exact log-likelihood 优化。

### 2.2 为什么这比纯 Gaussian 强?

看 Eq (A.4) 在 Appendix A.2:
$$p(x_s | x_t) = \mathcal{N}(f_T(x_s); \mu_P, \text{diag}(\sigma_P^2)) \cdot |\det J_{f_T}(x_s)|$$

- u-space 里是 Gaussian (predictor 是 affine coupling)
- x-space 里由于 nonlinear invertible mapping $f_T$,变成 non-Gaussian (multimodal, heavy-tailed, skewed)
- Jacobian determinant $|\det J_{f_T}|$ 重新加权 density 来 account for nonlinear warping

这是一个 division of labor:
- **Predictor (deep, 24 layers)**:捕获 denoising signal 主体,做 cross-timestep reasoning
- **Transporter (shallow, 8 layers = 2 blocks × 4 layers)**:处理 local non-Gaussian residual structure within each step

---

## 3. 公式详细讲解

### 3.1 Forward Process (Eq 2.1, 2.3)

Flow matching 的 forward interpolation:
$$x_t = (1-t)x_0 + t\epsilon, \quad q(x_t | x_0) = \mathcal{N}((1-t)x_0, t^2 I), \quad t \in [0,1]$$

- $x_0$:clean data
- $\epsilon \sim \mathcal{N}(0, I)$:Gaussian noise
- $t \in [0,1]$:t=0 是 clean,t=1 是 pure noise
- $q(x_t | x_0)$ 的 mean 是 $(1-t)x_0$,variance 是 $t^2$

Markovian forward transition (Eq 2.3) 保证 marginal 一致性:
$$x_t = \alpha_{s,t} x_s + \sigma_{s,t} \epsilon, \quad \alpha_{s,t} = \frac{1-t}{1-s}, \quad \sigma_{s,t} = \sqrt{t^2 - \alpha_{s,t}^2 s^2}$$

- $\alpha_{s,t}$:缩放系数,保证 $\mathbb{E}[x_t | x_0] = (1-t)x_0$
- $\sigma_{s,t}$:噪声标准差,保证 $\text{Var}[x_t | x_0] = t^2$
- 关键:构造 correlated Markov trajectory,不是 independent marginal samples

Proposition 1 (Appendix A.4) 证明了这个 transition 保留 marginals。

### 3.2 Normalizing Flow 基础 (Eq 2.5, 2.6)

Change-of-variables formula:
$$\log p(x) = \log p_0(f_\theta(x)) + \log|\det J_{f_\theta}(x)|$$

- $f_\theta$:invertible mapping
- $p_0$:simple prior (Gaussian)
- $J_{f_\theta}$:Jacobian matrix

TarFlow 的 autoregressive flow (Eq 2.6):
$$z_n = \frac{x_n - \mu_\theta(x_{<n})}{\sigma_\theta(x_{<n})}, \quad \log|\det J| = -\sum_n \log \sigma_\theta^{(n)}$$

- $x_n$:第 n 个 spatial token
- $x_{<n}$:前面所有 token (causal)
- $\mu_\theta, \sigma_\theta$:由 causal Transformer 预测
- Jacobian 是 triangular (autoregressive),所以 determinant 容易算

### 3.3 NTM 核心定义 (Eq 3.1, 3.3, 3.4)

NTM 模型定义:
$$\hat{u}_s = f_P(u_t, z, y), \quad u_s = f_T(x_s, s), \quad u_t = f_T(x_t, t)$$

Predictor 是 affine map: $\hat{u}_s = \mu_P(u_t, t, s, y) + \sigma_P(u_t, t, s, y) \cdot z$

这定义了 $p_P(u_s | u_t, y) = \mathcal{N}(\mu_P, \text{diag}(\sigma_P^2))$。

**NTM loss (Eq 3.3)** — 这是关键:
$$\mathcal{L}_{\text{NTM}} = -\log p(x_s | x_t) = -\log p_P(u_s | u_t) - \log|\det J_{f_T}|$$

- 第一项:u-space 里的 Gaussian log-likelihood
- 第二项:transporter 的 Jacobian 修正 (change of variables)
- 这就是 $p(x_s | x_t)$ 的 exact negative log-likelihood

**Expanded form (Eq 3.4)**:
$$\mathcal{L}_{\text{NTM}} = \sum_{k=1}^T \left[\frac{1}{2}\|z_k\|^2 + \sum_n \left(\log \sigma_P^{(k,n)} + \sum_\ell \log \sigma_T^{(k,\ell,n)}\right)\right]$$

- $z_k$:第 k 步的 latent noise
- $\sigma_P^{(k,n)}$:第 k 步位置 n 的 predictor scale
- $\sigma_T^{(k,\ell,n)}$:第 k 步,第 $\ell$ 个 transporter block,位置 n 的 scale
- 这是整个 trajectory 的 exact NLL

第一项 $\frac{1}{2}\|z_k\|^2$ 是 Gaussian prior 的负对数,$\log \sigma$ 项是 Jacobian 修正。

### 3.4 Trajectory Score Denoising (Eq 3.5) — 重大创新

$$\hat{x}^{\text{den}} = \frac{1}{1-t}\left(\hat{x} - S \cdot \nabla_{\hat{x}} \mathcal{L}_{\text{NTM}}\right)$$

- $\hat{x} = (\hat{x}_{t_0}, \ldots, \hat{x}_{t_T})$:生成的整个 trajectory
- $S$:trajectory covariance matrix, $[S]_{ij} = \frac{\min(t_i, t_j)^2 (1-\max(t_i, t_j))}{1-\min(t_i, t_j)}$
- $\nabla_{\hat{x}} \mathcal{L}_{\text{NTM}}$:NTM loss 对整个 trajectory 的 gradient,这是 joint score
- $1/(1-t)$:把 noisy domain 映射到 clean domain

**为什么这是创新?**

TarFlow 和 STARFlow 的 score-based denoising 是独立 per-sample 的。NTM 的 trajectory 是 correlated Markov sequence,$S$ 矩阵编码了 cross-timestep correlations。Proposition 3 (Appendix A.6) 推导:

$$\text{Cov}(x_{t_i}, x_{t_j} | x_0) = \alpha_{t_i, t_j} \cdot \text{Var}(\xi_i) = \frac{1-t_j}{1-t_i} \cdot t_i^2 \cdot I$$

修正一个 timestep 的 error 会 propagate 到所有其他 timestep (因为 $S$ 是 dense matrix),比独立 denoising 更有效。

### 3.5 Trajectory Covariance 直觉

$S$ 矩阵的 intuition:
- 对角元 $S_{ii} = t_i^2$:第 i 步的 noise variance
- 非对角元:cross-timestep covariance,正比于较小 timestep 的 variance,反比于较大 timestep 距离 1 的差

这告诉我们:相近 timestep 之间 correlation 强,timestep 越大 correlation 越弱 (越接近 pure noise)。

---

## 4. 架构详解

### 4.1 Deep-Shallow Architecture (Table 4)

NTM 采用 STARFlow (https://arxiv.org/abs/2505.20041) 的 deep-shallow 设计,但关键修改:

| 组件 | From scratch | Finetuned |
|------|--------------|-----------|
| Hidden dim | 3072 | 3072 |
| Blocks | 3 | 3 |
| Layers per block | [4, 4, 24] | [4, 4, 24] |
| Transporter | Blocks 1-2 (4 layers each) | Blocks 1-2 (4 layers each) |
| Predictor | Block 3 (24 layers) | FLUX.2-klein (4B) |
| Patch size | 1 | 2 |
| KV heads | 8 | 8 |
| Positional encoding | 2D RoPE | 2D RoPE |

**关键设计**:
- **Transporter**:TarFlow-style causal autoregressive flow,alternating scan directions (L→R, R→L),8 layers total,浅但 spatial AR
- **Predictor**:non-causal full-attention Transformer,24 layers,深但 parallel over spatial,operates over trajectory dimension

为什么这样分配容量?
- Transporter 浅 → 处理 local non-Gaussian residual,每个 step 内的 spatial 复杂度有限 (因为 conditional on $x_t$ 已经 constrain 了 plausible images)
- Predictor 深 → 处理 cross-timestep reasoning,捕获 trajectory 上的 denoising signal 主体
- Predictor 的 non-causal attention 让 spatial positions 并行处理,inference 高效

### 4.2 与 STARFlow 的对比 (Figure 8, Section 5)

这是一个 spectrum:

| 维度 | STARFlow | Flow Matching | NTM |
|------|----------|---------------|-----|
| 模型对象 | marginal $p(x)$ | velocity field | conditional $p(x_s | x_t)$ |
| Steps | 1 (single pass) | many (T=50-250) | few (T=4-8) |
| Within-step | 256+ AR steps | 1 Gaussian | shallow NF |
| Depth 位置 | 集中在单 step 内 | 每步浅 | 分布在 trajectory 上 |

NTM 的 architectural trade-off:用 shallow transporter per step + deep predictor across trajectory,trade per-step expressiveness for multi-step structure。

---

## 5. Training from Scratch

### 5.1 两种训练模式

Algorithm 1 (Appendix B) 描述训练过程:

1. **End-to-end**:计算 Eq (3.4) over 所有 T conditional factors
2. **Pair-wise**:每个 batch element 随机采样一个 consecutive pair $(t, s)$

每个 batch element 独立采样 $T \in \{4, 8, 16\}$,让单一模型支持不同 step counts 而不需 retraining。$f_T$ 接受 $T$ 作为额外输入来 adapt 到 local timestep spacing。

### 5.2 Sampling (Algorithm 2)

1. 采样初始 noise: $\hat{u}_{t_T} \sim \mathcal{N}(0, I)$
2. Predictor reverse (sequential over k, parallel over spatial):
   - 采样 $z_k \sim \mathcal{N}(0, I)$
   - 计算 $(\mu_P^{(k)}, \sigma_P^{(k)}) = \text{DeepBlock}(\hat{u}_{t_k}, t_k, t_{k-1}, y)$
   - 应用 CFG
   - $\hat{u}_{t_{k-1}} = z_k \cdot \sigma_P^{(k)} + \mu_P^{(k)}$
3. Transporter inverse: $\hat{x}_{t_0} = f_T^{-1}(\hat{u}_{t_0})$ (sequential AR decoding with KV-cache)
4. Optional: self-refinement (Algorithm 3) or learned denoiser
5. Decode: image = VAE.decode($\hat{x}_{t_0}$)

### 5.3 Classifier-Free Guidance (Appendix C.5)

NTM 用 logits-guided CFG,作用于 coupling parameters 而不是 predicted sample:

$$s = \left(\frac{\sigma_c}{\sigma_u}\right)^2 \text{ (clipped to } [0,1]\text{)}$$
$$\sigma_{\text{eff}} = \frac{\sigma_c}{\sqrt{1 + w - w \cdot s}}$$
$$\mu_{\text{eff}} = \frac{(1+w)\mu_c - w \cdot s \cdot \mu_u}{1 + w - w \cdot s}$$

- $w$:guidance scale
- $\mu_c, \sigma_c$:conditional predictions
- $\mu_u, \sigma_u$:unconditional predictions
- 对应 $(1+w)\log p_c - w \log p_u$ 应用到 Gaussian coupling in u-space
- 同时调整 mean 和 scale,比 standard linear guidance (只改 mean) 更合理

### 5.4 Timestep Schedule (Appendix C.4)

Shifted schedule (来自 SD3, https://arxiv.org/abs/2403.03206):
$$\tilde{\sigma}_k = \frac{k}{T}, \quad \sigma_k = \frac{e^\mu}{e^\mu + 1/\tilde{\sigma}_k - 1}, \quad \mu = 0.5 + 0.65 \cdot \frac{L_{\text{seq}} - 256}{4096 - 256}$$

- $L_{\text{seq}}$:spatial sequence length
- $\mu$ 根据序列长度调整,长序列用更大的 shift

---

## 6. Finetuning from Pretrained Models

### 6.1 初始化策略 (Eq 3.6, 3.7)

Pretrained flow matching backbone 训练来预测 velocity $v$ in x-space。NTM 重用这个:

$$\hat{u}_0 = u_t - t \cdot v \quad \text{(预测 clean sample)}$$
$$\mu_{\text{post}} = A(t,s) u_t + B(t,s) \hat{u}_0, \quad \sigma_{\text{post}} = C(t,s)$$

**Closed-form coefficients** (Proposition 2, Appendix A.5):

$$A(t,s) = \frac{s^2(1-t)}{t^2(1-s)}, \quad B(t,s) = \frac{(t-s)(t+s-2ts)}{t^2(1-s)}, \quad C(t,s)^2 = \frac{s^2(t-s)(t+s-2ts)}{t^2(1-s)^2}$$

这是 reverse posterior $q(x_s | x_t, x_0)$ 的参数,通过 Bayes 推导。

**Table 3** 给出 4-step schedule 的具体数值:

| Step | t | s | A(t,s) | B(t,s) | C(t,s) |
|------|---|---|--------|--------|--------|
| 1 | 1.000 | 0.754 | 0.140 | 0.614 | 0.371 |
| 2 | 0.754 | 0.509 | 0.271 | 0.496 | 0.362 |
| 3 | 0.509 | 0.263 | 0.470 | 0.362 | 0.297 |
| 4 | 0.263 | 0.020 | 0.948 | 0.049 | 0.049 |

注意:最后一步 (t=0.263 → s=0.020) $A$ 接近 1, $B$ 接近 0, $C$ 很小 — 几乎是 identity,只有小量噪声需要去除。

**Zero-init scale correction** (Eq 3.7):
$$\mu_P = \mu_{\text{post}}, \quad \sigma_P = \sigma_{\text{post}} \cdot \exp(\delta_\sigma), \quad \delta_\sigma = \text{proj}_{\text{out}}(h)$$

- $\text{proj}_{\text{out}}$ 初始化为 0,所以 $\sigma_P = \sigma_{\text{post}}$ at initialization
- Transporter 初始化为 identity ($f_T = \text{id}$)
- 整个模型初始时 = pretrained Gaussian posterior in x-space

### 6.2 Mean-Alignment Auxiliary Loss (Eq 3.8)

$$\mathcal{L}_{\text{aux}} = \|\mu_P - \mu_{\text{FM}}\|^2$$

- $\mu_{\text{FM}}$:frozen pretrained backbone 直接从 x-space 预测的 mean
- 总 loss: $\mathcal{L} = \mathcal{L}_{\text{NTM}} + \lambda \mathcal{L}_{\text{aux}}$, $\lambda$ anneal during training

**三个作用** (Appendix A.3):
1. 防止 catastrophic drift from pretrained solution
2. $\mu_{\text{FM}}$ 定义 meaningful u-space,$\mathcal{L}_{\text{aux}}$ 确保 transporter 学会把 per-step predictions 连成 coherent trajectory
3. 让 transporter 和 predictor 可以联合优化而 drift

**Ablation** (Figure 7a, 7b):
- $\lambda = 0$:finetuning 早期 diverge — NLL objective 不足以保持 predictor 接近 pretrained solution
- $\lambda > 0$:训练稳定

**Limiting case** $\lambda \to \infty$: $\mu_P$ forced to match $\mu_{\text{FM}}$,non-Gaussian 表达只能来自 $\sigma_P$ 和 $f_T$。比 standard diffusion 强 (有 spatial flow),但比 full NTM 弱 (没学 $\mu_P$)。

---

## 7. Fast Generation via Learned Denoiser (Section 3.4)

Standard NTM sampling 需要:
- T sequential predictor steps
- AR decoding at each step
- Test-time backprop-based trajectory score denoising

为了消除这个 cost,训练 lightweight denoiser $g_\phi$:

$$\mathcal{L}_{\text{den}} = \|g_\phi(u_{t_0}, y) - \hat{x}_0^{\text{den}}\|^2$$

- $g_\phi$:non-causal attention Transformer
- 输入:predictor output $u_{t_0}$ at cleanest level + text embeddings y
- 输出:denoised image $\hat{x}_0^{\text{den}}$ in patch space
- Target:score-based denoising (Eq 3.5) 的输出
- 训练:MSE,在 main model converge 后 post-train

因为 trajectory 是 Markov,$u_{t_0}$ 包含 deterministic predict clean output 所需的所有信息。

**Inference pipeline**:
1. Run predictor over T steps → $u_{t_0}$
2. Run $g_\phi$ in single forward pass → $\hat{x}_0$

这 bypasses transporter AR decoding 和 backprop-based denoising。

**Table 2 结果**:
| Method | img/s ↑ | LPIPS ↓ |
|--------|---------|---------|
| Full NF + Traj. denoise | 0.20 | - |
| Predictor + Denoiser | 1.88 | 0.121 |

~9× speedup,高保真 (LPIPS 0.121)。

这是 self-distillation:NTM 用自己的 score function 教 denoiser,不需要额外数据。

---

## 8. 实验结果分析

### 8.1 Text-to-Image (Table 1)

| Type | Method | GenEval↑ | DPG↑ |
|------|--------|----------|------|
| DM | SDXL | 0.55 | 74.65 |
| DM | SD3-Medium | 0.62 | 84.08 |
| DM | FLUX.1-dev | 0.66 | 83.84 |
| DM | Janus-Pro-7B | 0.80 | 84.19 |
| DM | HiDream-I1-Full | 0.83 | 85.89 |
| DM | Seedream 3.0 | 0.84 | 88.27 |
| DM | Qwen-Image | 0.87 | 88.32 |
| DM | Nucleus-Image | 0.87 | 88.79 |
| NF | STARFlow (256 AR steps) | 0.56 | - |
| **NF** | **NTM (from scratch, 256×256, 4 steps)** | **0.82** | **79.64** |
| **NF** | **NTM (finetune, 512×512, 4 steps)** | **0.76** | **83.38** |

关键观察:
- NTM from scratch 4 步达到 GenEval 0.82,显著超过 STARFlow (0.56, 256 AR steps)
- NTM 接近强 diffusion baselines (FLUX.1-dev 0.66, Janus-Pro 0.80, HiDream 0.83)
- **NTM 是唯一保留 exact likelihood 的方法**
- Finetune 版本在 512×512 达到 DPG 83.38,接近 FLUX.1-dev (83.84, 50 steps)

### 8.2 Class-Conditional ImageNet (Table 6, Appendix D.3)

| Method | Type | #Params | Steps | FID↓ |
|--------|------|---------|-------|------|
| DiT-XL/2 | DM | 675M | 250 | 2.27 |
| SiT-XL | DM | 675M | 250 | 2.06 |
| LlamaGen | AR | 3.1B | 256 | 2.18 |
| VAR | AR | 2.0B | 10 | 1.73 |
| DART | AR | 820M | 16 | 3.82 |
| TarFlow | NF | 1.4B | 1024 | 5.56 |
| STARFlow (VAE) | NF | 1.4B | 1024 | 2.40 |
| STARFlow (FAE) | NF | 1.4B | 256 | 2.67 |
| **NTM** | **NF** | **1.4B** | **4** | **3.83** |
| **NTM** | **NF** | **1.4B** | **8** | **3.24** |
| **NTM** | **NF** | **1.4B** | **16** | **2.80** |

- NTM 16 步 FID 2.80,接近 STARFlow (FAE) 256 步 FID 2.67
- **只用 NLL objective,没有 adversarial 或 perceptual losses**
- Step 数从 256 降到 16,质量损失很小

### 8.3 Multi-Trajectory Training (Figure 6)

Finetuned 模型用不同 T 训练,与 FLUX (50 steps) baseline 比较:
- T=4: 最佳 quality-speed trade-off for finetuning
- T=8, T=16: 更细粒度 denoising,但 inference 更慢

### 8.4 Transporter 作用 (Figure 2)

- 没有 transporter,flow matching 4 步产生严重模糊
- 有 transporter,4 步恢复 50 步质量
- Invertible mapping 提供 latent space 让 affine predictor 变得 expressive

### 8.5 T=1 失败 (Figure 9, Section 5)

T=1 时:
- 整个 non-Gaussian structure 必须由 shallow transporter (8 layers) 单独捕获
- Predictor 退化为 single-step Gaussian coupling
- 容量不足 — 这本质上是 STARFlow-like 架构但 transporter 远浅于 STARFlow (24+ layers × multiple blocks)

**Implication**:NTM sweet spot 是 T=4-8。要 push 到 single-step exact-likelihood generation,需要 fundamentally 不同 的 architecture,比如 adaptive-depth transporters 或 progressive capacity allocation。

---

## 9. 理论分析 (Appendix A)

### 9.1 NTM 作为 Conditional Normalizing Flow (A.1)

NTM 把 $x_s$ 映射到 $z$ 通过两个 invertible transformations 的 composition:
1. Transporter: $u_s = f_T(x_s)$,spatial autoregressive flow,triangular Jacobian
2. Predictor: $z = (u_s - \mu_P(u_t, t, s))/\sigma_P(u_t, t, s)$,diagonal affine coupling

Composition $z = f_P(f_T(x_s); u_t, t, s)$ 是 invertible,exact log-likelihood:
$$\log p(x_s | x_t) = \underbrace{\log p_0(z)}_{\text{Gaussian prior}} + \underbrace{\log|\det J_{f_P}|}_{\text{predictor}} + \underbrace{\sum_{\ell=1}^L \log|\det J_{f_T^{(\ell)}}|}_{\text{transporter}}$$

### 9.2 Decomposition (A.2)

**Without transporter** ($f_T = \text{id}$):
$$p(x_s | x_t) = \mathcal{N}(\mu_P(x_t, t, s), \text{diag}(\sigma_P^2))$$

这是 diagonal Gaussian,就是 standard diffusion。NTM loss 退化为:
$$\mathcal{L} = \sum_k \left[\frac{1}{2}\left\|\frac{x_{s_k} - \mu_P^{(k)}}{\sigma_P^{(k)}}\right\|^2 + \sum_n \log \sigma_P^{(k,n)}\right]$$

这是 heteroscedastic Gaussian regression 的 NLL。如果 $\sigma_P$ 固定,最小化 $\mu_P$ 就是 weighted MSE,恢复 standard diffusion/flow matching objective。

**With transporter**:nonlinear invertible mapping 让 x-space 分布变 non-Gaussian,Jacobian 重新加权 density。

### 9.3 FM Auxiliary Loss 分析 (A.3)

- $\lambda = 0$:predictor 和 transporter 自由分配 work,可能 early-training instability
- $\lambda > 0$:$\mu_P \approx \mu_{\text{FM}}$ 稳定 u-space,non-Gaussian 来自 $\sigma_P$ 和 $f_T$
- $\lambda \to \infty$:$\mu_P$ forced match $\mu_{\text{FM}}$,只有 transporter 提供 non-Gaussian

实验中 $\lambda$ anneal:开始 full strength 稳定初始化,然后 decay 让 NLL fine-tune mean beyond Gaussian approximation。

---

## 10. 我的理解和 Intuition

### 10.1 核心 Intuition:Warp Then Gauss

NTM 的本质是:**找一个 warp $f_T$ 使得 warped space 里 conditional 是 Gaussian**。这比直接在原 space 建模 non-Gaussian 容易,因为:
- Gaussian predictor 容易训练 (affine coupling)
- Warp 通过 Jacobian 获得 exact likelihood
- Warp 可以 shallow (因为 conditional 比 marginal 简单)

这和 normalizing flow 的本质一脉相承,但是 conditional 而不是 marginal,而且 warp 是 shallow 的。

### 10.2 为什么 Conditional 比 Marginal 简单?

STARFlow 直接建模 $p(x)$,需要 deep spatial AR (256+ steps)。NTM 建模 $p(x_s | x_t)$,因为 conditioning on $x_t$ 已经 constrain 了 plausible images 的空间,所以 per-step flow 简单,只需 shallow transporter。

这是 NTM 的 fundamental insight:把 capacity 从 single-step AR 转移到 multi-step trajectory,trade per-step expressiveness for multi-step structure。

### 10.3 Trajectory Denoising 的优美之处

普通 normalizing flow denoising 是 per-sample,利用 score function $\nabla \log p(x)$。NTM 的 trajectory 是 Markov correlated sequence,$S$ 矩阵编码 cross-timestep covariance。

修正一个 timestep 的 error 会 propagate 到所有其他 timestep (因为 $S$ dense),这比独立 denoising 更有效 — 你在用整个 trajectory 的信息 refine 每一个点。

这个 trajectory-level score denoising 可以蒸馏成单 forward pass 的 denoiser,实现 9× speedup。

### 10.4 Finetuning Recipe 的工程智慧

Finetuning 的设计很巧妙:
- Identity init + zero-init residual:保证初始时 = pretrained model
- Mean-alignment aux loss:防止 drift,稳定 u-space
- Closed-form posterior coefficients:用 pretrained velocity 直接 derive Gaussian posterior

这让你能从 FLUX.2-klein (4B) 等 SOTA flow matching 模型 finetune,而不是从头训。这是一个 practical 的贡献。

### 10.5 Open Questions

Paper 自己提到:
- T=1 失败是 capacity constraint,不是 training issue
- Push 到 single-step exact-likelihood 需要 fundamentally 不同 architecture
- Distribution-level post-training (adversarial, perceptual) 可以进一步提升 few-step quality

NTM 提供了一个 principled framework,后续工作可以在:
- Adaptive-depth transporters
- Progressive capacity allocation
- Distribution-level post-training

---

## 11. 相关工作链接

让我整理一下相关工作的链接供参考:

**Normalizing Flows**:
- NICE: https://arxiv.org/abs/1410.8516
- RealNVP: https://arxiv.org/abs/1605.08803
- Glow: https://arxiv.org/abs/1807.03039
- TarFlow: https://arxiv.org/abs/2412.06329
- STARFlow: https://arxiv.org/abs/2505.20041

**Diffusion/Flow Matching**:
- DDPM: https://arxiv.org/abs/2006.11239
- Score SDE: https://arxiv.org/abs/2011.13456
- Flow Matching: https://arxiv.org/abs/2210.02747
- Rectified Flow: https://arxiv.org/abs/2209.03003
- SD3 (Rectified Flow Transformers): https://arxiv.org/abs/2403.03206

**Few-step Generation**:
- Progressive Distillation: https://arxiv.org/abs/2202.00509
- Consistency Models: https://arxiv.org/abs/2303.01469
- DMD: https://arxiv.org/abs/2311.18828
- Latent Consistency Models: https://arxiv.org/abs/2310.04378
- DDGAN: https://arxiv.org/abs/2206.03511
- Mean Flows: https://arxiv.org/abs/2505.13447
- FlowMaps: https://arxiv.org/abs/2505.18825
- NFM (Normalizing Flow Matching): https://arxiv.org/abs/2603.09014

**Self-supervised Learning**:
- BYOL: https://arxiv.org/abs/2006.07733
- JEPA: https://arxiv.org/abs/2301.08243
- VICReg: https://arxiv.org/abs/2105.04906
- DINO: https://arxiv.org/abs/2104.14294

**T2I Baselines**:
- SDXL: https://arxiv.org/abs/2307.01952
- PixArt-α: https://arxiv.org/abs/2310.00426
- FLUX.1: https://blackforestlabs.ai/
- Janus-Pro: https://arxiv.org/abs/2501.17811
- HiDream-I1: https://arxiv.org/abs/2503.04287
- DART: https://arxiv.org/abs/2410.08159

**Evaluation**:
- GenEval: https://arxiv.org/abs/2310.17525
- DPG-Bench: https://arxiv.org/abs/2403.05135

---

## 12. 总结

NTM 是一个 elegant 的 framework,把 normalizing flow 的 expressiveness 和 diffusion/flow matching 的 multi-step structure 桥接起来。核心贡献:

1. **理论**:把 $p(x_s | x_t)$ 建模为 conditional normalizing flow with exact log-likelihood,解决 Gaussian bottleneck
2. **架构**:shallow transporter per step + deep predictor across trajectory 的 division of labor
3. **Finetuning**:identity init + zero-init residual + mean-alignment aux loss 的稳定 recipe
4. **Self-distillation**:trajectory score denoising 利用 Markov covariance,蒸馏成 4-step sampler
5. **实验**:T2I GenEval 0.82 (4 steps, from scratch),唯一保留 exact likelihood 的 few-step 方法

NTM 在 STARFlow (1 step, deep AR) 和 flow matching (many steps, Gaussian) 的 spectrum 上找到 sweet spot T=4-8,提供了一个 principled 的 framework 让 exact-likelihood generation 走向 few-step regime。
