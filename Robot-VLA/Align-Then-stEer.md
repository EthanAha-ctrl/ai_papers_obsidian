---
source_pdf: Align-Then-stEer.pdf
paper_sha256: c771963b858c5c5c4d4bd7ae64dfe188776c9fff9181214522411d89bd507e73
processed_at: '2026-07-18T07:32:29-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Align-Then-stEer (ATE): 深度技术讲解

## TL;DR

ATE 解决 VLA 跨 embodiment/跨 task adaptation 的核心痛点：pre-training dataset 里都是 6-DoF 单臂（Open X-Embodiment、DROID、ALOHA），adaptation 时却可能遇到 7-DoF 双臂 RealMan，action distribution 完全不匹配。ATE 的核心招式两步走：

1. **Stage 1 (Align)**: 用两个 InfoVAE 把 pre-training action 和 adaptation action 都嵌入到同一个 latent space $\mathcal{Z}$。关键在于 reverse KL 让 adaptation latent 被"塞进"pre-trained latent 的某个 mode 里，构造 hierarchical 结构。
2. **Stage 2 (Steer)**: 借用 classifier guidance（Dhariwal & Nichol 2021 的思路）在 latent space 里施加 gradient guidance，引导 diffusion/flow-based VLA 的 fine-tuning 朝目标 distribution 收敛。

整个 framework 是 plug-and-play 的，不动 VLA 架构，只额外训练两个轻量级 VAE。

Project page: https://align-then-steer.github.io/
Code: https://github.com/TeleHuman/Align-Then-Steer

---

## 1. Motivation: 为什么直接 fine-tune 不行

VLA 的标准两阶段范式：
- **Pre-train**: 在 $\mathcal{D}_{\text{pretrain}}$ 上学 $p(\bar{\mathbf{a}}_{t:t+H-1} | \mathbf{o}_t, l)$，捕捉 $\mathcal{M}_{\text{pretrain}} = \{\mathcal{E}_1, ..., \mathcal{E}_n\}$ 多种 embodiment 的 general visuomotor prior。
- **Adapt**: 在 $\mathcal{D}_{\text{adaptation}}$ 上 fine-tune，target embodiment $\mathcal{E}_{\text{target}} \notin \mathcal{M}_{\text{pretrain}}$。

问题出在 action distribution 不一致：
- DoF 不同（单臂 6-DoF vs 双臂 16-DoF）
- 物理表示不同（joint angles、end-effector poses、torques）
- 任务不同导致 multimodality 不同

直接 fine-tune 时，新的 action distribution 离 pre-trained prior 太远，需要大量数据才能 shift。近期工作（OpenVLA 的 LoRA、FAST 的 DCT 压缩、VLA-Cache、MoLE-VLA）大多只优化 parameter efficiency，没有触及 action distribution mismatch 的根本。

**ATE 的洞察**: 既然分布不匹配是核心矛盾，那就显式地构造一个统一 latent space，让 guidance 在这个 space 里拉扯 policy。

---

## 2. Stage 1: Unified Action Latent Space

### 2.1 Pre-training Action VAE $\mathcal{V}_{\text{pretrain}} = \{\mathbf{E}_\phi, \mathbf{D}_\phi\}$

**架构**: Transformer encoder + Transformer decoder with cross-attention
- Encoder 输入: action chunk $\bar{\mathbf{a}}_{t:t+H-1}$ (长度 H)
- 前置两个 learnable tokens: $\mu_{\text{token}}$ 和 $\Sigma_{\text{token}}$，输出对应位置被用作 Gaussian 分布的均值和方差
- 用 reparameterization trick 采样 $z \in \mathbb{R}^d$
- Decoder 用 H 个 zero embeddings 做 query，z 做 key/value，重构出 H 长度的 action chunk

这个设计借鉴自 motion sequence VAE (Petrovich et al. 2021, "Action-Conditioned 3D Human Motion Synthesis with Transformer VAE"，CVPR 2021, https://openaccess.thecvf.com/content/CVPR2021/papers/Petrovich_Action-Conditioned_3D_Human_Motion_Synthesis_With_Transformer_VAE_CVPR_2021_paper.pdf)，把整段序列压成一个 chunk-level latent。

**损失函数（InfoVAE variant, Eq. 5）**:

$$
\mathcal{L}(\phi; \mathcal{D}_{\text{pretrain}}) = \underbrace{\mathbb{E}_{p_{\mathcal{D}}(\bar{\mathbf{a}})} \mathbb{E}_{q_\phi(z|\bar{\mathbf{a}})} [\log p_\phi(\bar{\mathbf{a}}|z)]}_{\text{(I) Reconstruction}} - (1-\alpha) \underbrace{\mathbb{E}_{p_{\mathcal{D}}} [D_{\text{KL}}(q_\phi(z|\bar{\mathbf{a}}) \| p(z))]}_{\text{(II) Amortized KL}} - (\alpha - \lambda - 1) \underbrace{D_{\text{KL}}(q_\phi(z) \| p(z))}_{\text{(III) Marginal KL}}
$$

变量含义：
- $\phi$: VAE 参数
- $\bar{\mathbf{a}}_{t:t+H-1}$: 长度为 H 的 pre-training action chunk
- $z$: latent vector, $z \in \mathbb{R}^d$（论文里 $d=512$）
- $p(z) = \mathcal{N}(0, I)$: fixed latent prior
- $\alpha, \lambda$: hyperparameters 控制各 term 强度
- 项 (I): standard VAE reconstruction log-likelihood
- 项 (II): amortized inference 的 KL，鼓励 posterior 接近 prior
- 项 (III): marginal KL，鼓励聚合分布 $q_\phi(z) = \int q_\phi(z|\bar{\mathbf{a}}) p_{\mathcal{D}}(\bar{\mathbf{a}}) d\bar{\mathbf{a}}$ 接近 prior

为什么不用 vanilla VAE（Kingma & Welling 2013, https://arxiv.org/abs/1312.6114）？vanilla VAE 有两个著名问题：
1. **Posterior collapse**: decoder 太强时 posterior 退化成 prior，z 不携带信息
2. **Amortized inference gap**: encoder 用单一神经网络近似所有 posterior，capacity 不足

InfoVAE (Zhao et al. 2017, https://arxiv.org/abs/1706.02262) 通过显式优化 marginal KL + 最大化 mutual information $\mathbb{I}(\bar{\mathbf{a}}; z)$ 来缓解。论文里用 **MMD (Maximum-Mean Discrepancy)** 替代项 (III)，因为高维 Gaussian 的 KL 优化不稳定。MMD 来自 Li et al. 2015 "Generative Moment Matching Networks" (http://proceedings.mlr.press/v37/li15.html) 和 Dziugaite et al. 2015 (http://auai.org/uai2015/proceedings/papers/230.pdf)。

### 2.2 Adaptation Action VAE $\mathcal{V}_{\text{adaptation}} = \{\mathbf{E}_\psi, \mathbf{D}_\psi\}$

架构完全一样，关键在 prior 改成已学好的 $q_\phi(z)$：

$$
\mathcal{L}(\psi; \mathcal{D}_{\text{adaptation}}) = \mathbb{E}_{p_{\mathcal{D}}(\tilde{\mathbf{a}})} \mathbb{E}_{q_\psi(z|\tilde{\mathbf{a}})} [\log p_\psi(\tilde{\mathbf{a}}|z)] - (1-\alpha) \mathbb{E}_{p_{\mathcal{D}}} [D_{\text{KL}}(q_\psi(z|\tilde{\mathbf{a}}) \| q_\phi(z))] - (\alpha - \lambda - 1) D_{\text{KL}}(q_\psi(z) \| q_\phi(z))
$$

其中 $\tilde{\mathbf{a}}_{t:t+L-1}$ 是 adaptation action chunk（长度 L，跟 pre-training 长度 H 可以不同），$q_\phi(z)$ 通过对 pre-trained VAE 在所有 pre-training action chunk 上 forward 算 running mean $\mu_\phi$ 和 variance $\Sigma_\phi$，近似为 $\mathcal{N}(\mu_\phi, \Sigma_\phi)$。

### 2.3 Reverse KL 与 Mode-Seeking: 直觉

这是 ATE Stage 1 的核心 trick。要 minimize 的是 $D_{\text{KL}}(q_\psi(z|\tilde{\mathbf{a}}) \| q_\phi(z))$，注意 KL 的方向：$q_\psi$ 在前，$q_\phi$ 在后，这是 **reverse KL**。

**为什么 reverse KL 是 mode-seeking？** 写开定义：
$$
D_{\text{KL}}(q \| p) = \int q(x) \log \frac{q(x)}{p(x)} dx
$$
当 $p(x) \to 0$ 而 $q(x) > 0$ 时，$\log(q/p) \to +\infty$，惩罚无穷大。所以 $q$ 必须避开 $p=0$ 的区域。但 $q$ 可以只覆盖 $p>0$ 的一个 mode，不必覆盖所有 mass。这就是 mode-seeking。

对比 forward KL $D_{\text{KL}}(p \| q)$：当 $p(x) > 0$ 而 $q(x) \to 0$ 时惩罚无穷大，所以 $q$ 必须覆盖 $p$ 的所有 support，是 mass-covering（mode-covering）。

直观理解：pre-training latent $q_\phi(z)$ 是个 multimodal distribution（不同 embodiment 占不同 mode），adaptation latent $q_\psi(z)$ 通过 reverse KL "挤进"其中一个 mode，保留了 pre-training 的 hierarchical structure。这个嵌入过程相当于让新 embodiment 在 latent space 里"认领"一个区域，不破坏其他 mode 的结构。

参考资料：
- Bishop "Pattern Recognition and Machine Learning" Chapter 10 (https://www.microsoft.com/en-us/research/people/cmbishop/#!prml-book)
- Minka 2005 "Divergence measures and message passing" (https://tminka.github.io/papers/divergence/minka-divergence.pdf)

### 2.4 Algorithm 1 解析

```
Input: D_pretrain, D_adaptation, VAE_φ, VAE_ψ, epochs N
1. Train VAE_φ on D_pretrain by maximizing Eq. 5
2. Compute running μ_φ, Σ_φ over all pretrain chunks via E_φ
3. Train VAE_ψ on D_adaptation by maximizing Eq. 6 with prior N(μ_φ, Σ_φ)
```

实际开销：
- Step 1 在 RDT 上用 Open X-Embodiment subset + DROID + Kuka + ALOHA，3000 episodes，1000 epochs，~12 小时
- Step 2 在 RoboTwin/ManiSkill/real-robot 上，每个 task 100 episodes，200 epochs，~0.5 小时
- Latent dim = 512, Adam optimizer, lr = 1e-4, batch size = 64

---

## 3. Stage 2: Classifier Guidance for Steering Adaptation

### 3.1 Diffusion Model 背景

DDPM (Ho et al. 2020, https://arxiv.org/abs/2006.11239) 的前向过程：
$$
q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{\alpha_t} \mathbf{x}_{t-1}, (1-\alpha_t) I)
$$

变量：
- $\alpha_t \in (0, 1)$: predefined noise scheduler
- $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$: cumulative product
- $\mathbf{x}_t$: 在第 t 步加噪后的样本
- $\mathbf{x}_0$: 原始 clean 数据

训练 loss（Eq. 2）:
$$
\mathcal{L}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \epsilon}[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t)\|^2]
$$

反向采样（Eq. 3）:
$$
\mathbf{x}_{k-1} = \frac{1}{\sqrt{\alpha_k}} \left( \mathbf{x}_k - \sqrt{1 - \bar{\alpha}_k} \epsilon_\theta(\mathbf{x}_k, k) \right) + \sigma_k \epsilon_k
$$

### 3.2 Flow Matching 背景

Flow Matching (Lipman et al. 2023, https://arxiv.org/abs/2210.02747; Liu et al. 2023, https://arxiv.org/abs/2209.03023) 把生成建模成 source distribution 到 target distribution 的 probability path。

线性 Gaussian path: $\mathbf{x}_\tau \sim \mathcal{N}(\tau \mathbf{x}_0, (1-\tau) I)$，$\tau \in [0, 1]$

Conditional Flow Matching loss（Eq. 4）:
$$
\mathcal{L}(\theta) = \mathbb{E}_{\tau, \mathbf{x}_0, \epsilon} [\| v_\theta(\tau \mathbf{x}_0 + (1-\tau)\epsilon, \tau) - (\mathbf{x}_0 - \epsilon) \|^2]
$$

变量：$\tau$ 是 flow 时间步（区别于 diffusion 的 $t$）；$v_\theta$ 是 velocity field；$\epsilon \sim \mathcal{N}(0, I)$ 是噪声。

### 3.3 Tweedie's Formula 导出 Score Function

Eq. 7 是 ATE 的关键推导之一。根据 Luo 2022 "Understanding Diffusion Models: A Unified Perspective" (https://arxiv.org/abs/2208.11970) 和 Tweedie's formula：

对于 $\mathbf{a}^k_t = \sqrt{\bar{\alpha}_k} \mathbf{a}^0_t + \sqrt{1 - \bar{\alpha}_k} \epsilon$，其期望和方差已知，score function 是：
$$
\nabla_{\mathbf{a}^k} \log p_\theta(\mathbf{a}^k) = -\frac{1}{\sqrt{1 - \bar{\alpha}_k}} \epsilon_\theta(\mathbf{a}^k, k)
$$

变量：$\nabla_{\mathbf{a}^k}$ 是对 noisy action 的 gradient；左式是 score function (log density 的 gradient)。

### 3.4 Conditional Score via Bayes' Rule

要 sample from $p_\theta(\mathbf{a}^k | y)$（y 是 conditioning label），用 Bayes:
$$
\log p_\theta(\mathbf{a}^k | y) = \log p_\theta(\mathbf{a}^k) + \log p(y | \mathbf{a}^k) - \log p(y)
$$

对 $\mathbf{a}^k$ 求 gradient，第三项消失：
$$
\nabla \log p_\theta(\mathbf{a}^k | y) = \nabla \log p_\theta(\mathbf{a}^k) + \nabla \log p(y | \mathbf{a}^k)
$$

把 Eq. 7 代入：
$$
\nabla \log p_\theta(\mathbf{a}^k | y) = -\frac{1}{\sqrt{1 - \bar{\alpha}_k}} \epsilon_\theta(\mathbf{a}^k, k) + g
$$

其中 $g = \nabla_{\mathbf{a}^k} \log p(y | \mathbf{a}^k)$ 是 classifier guidance gradient。再反解出"calibrated noise prediction"（Eq. 8）:
$$
\hat{\epsilon}(\mathbf{a}^k, k) := \epsilon_\theta(\mathbf{a}^k, k) - \sqrt{1 - \bar{\alpha}_k} \cdot g
$$

### 3.5 Guidance for Diffusion VLAs（Eq. 9）

把 calibrated noise prediction 嵌入 denoising loss：
$$
\mathcal{L}(\theta) = \mathbb{E}_{k, \epsilon, (\mathbf{o}_t, \mathbf{a}^0, l) \sim \mathcal{D}_{\text{adapt}}} \left[ \left\| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_k} \mathbf{a}^0 + \sqrt{1-\bar{\alpha}_k} \epsilon, k, \mathbf{o}_t, l) + \sqrt{1-\bar{\alpha}_k} \cdot \lambda \cdot g \right\|^2 \right]
$$

变量：
- $k$: diffusion timestep
- $\epsilon \sim \mathcal{N}(0, I)$: 训练时采样的 noise
- $\mathbf{a}^0_{t:t+h}$: clean action chunk from adaptation data
- $\mathbf{o}_t$: observation (RGB + proprioception)
- $l$: language instruction
- $\lambda$: guidance scale（标量超参）
- $g$: classifier guidance gradient

### 3.6 Guidance for Flow-based VLAs（Eq. 10-12）

类似推导，但要用 flow matching 的 score relation（Eq. 10, 来自 Lipman et al. 2024 "Flow Matching Guide and Code", https://arxiv.org/abs/2412.06264）:
$$
v_\theta(\mathbf{x}^\tau) = \frac{1}{\tau} \mathbf{x}^\tau + \frac{1-\tau}{\tau} \nabla_{\mathbf{x}^\tau} \log p_\theta(\mathbf{x}^\tau)
$$

变量：$v_\theta$ 是 velocity field；$\tau \in [0,1]$ 是 flow 时间；右式第一项来自 linear Gaussian path 的解析 velocity，第二项是 score 修正。

Conditional version（Eq. 11）:
$$
\hat{v}_\theta(\mathbf{a}^\tau | y) = v_\theta(\mathbf{a}^\tau) + \frac{1-\tau}{\tau} \nabla_{\mathbf{a}^\tau} \log p(y | \mathbf{a}^\tau)
$$

最终训练目标（Eq. 12）:
$$
\mathcal{L}(\theta) = \mathbb{E}_{\tau, \epsilon, ...} \left[ \left\| v_\theta(\tau \mathbf{a}^0 + (1-\tau)\epsilon, \tau, \mathbf{o}_t, l) + \frac{1-\tau}{\tau} \lambda g - (\mathbf{a}^0 - \epsilon) \right\|^2 \right]
$$

### 3.7 Guidance Function 的设计（Eq. 13）

这是 ATE 最有创意的一步。借鉴 Carvalho et al. 2023 "Motion Planning Diffusion" (https://arxiv.org/abs/2310.07144) 和 Liang et al. 2025 "DexHandDiff" (https://arxiv.org/abs/2412.10687) 的 energy-based classifier 思路：

$$
p_\psi(y | \hat{\mathbf{a}}^k) = \frac{1}{Z_\psi} \exp(-\| \mathbf{E}_\psi(\hat{\mathbf{a}}^k) - \mathbf{E}_\psi(\mathbf{a}^0) \|^2)
$$

变量：
- $\mathbf{E}_\psi$: adaptation VAE 的 encoder（参数 frozen 在 Stage 1 训好后）
- $\hat{\mathbf{a}}^k$: noisy action chunk during denoising
- $\mathbf{a}^0$: ground truth clean action chunk（仅在训练时可得）
- $Z_\psi$: normalizing constant，求 gradient 时可忽略

求 gradient:
$$
g = \nabla_{\hat{\mathbf{a}}^k} \log p_\psi(y | \hat{\mathbf{a}}^k) \propto -\nabla_{\hat{\mathbf{a}}^k} \| \mathbf{E}_\psi(\hat{\mathbf{a}}^k) - \mathbf{E}_\psi(\mathbf{a}^0) \|^2
$$

**直觉**: guidance 在 latent space 里度量 noisy action 和 ground truth 的距离，gradient 拉着 noisy action 让其 latent 接近 ground truth 的 latent。这相当于在 unified latent space 里加一根"弹簧"连着 noisy prediction 和 ground truth，每次 denoising step 都把弹簧能量算进 loss。

**为什么这是 latent guidance 而不是直接 action space guidance？** 因为 raw action space 在不同 embodiment 间根本不可比（DoF 都不一样），但在 unified latent space 里 distance 是 well-defined 的。这正是 Stage 1 构造 unified latent space 的意义所在。

**为什么用 InfoVAE 的 encoder 不用 vanilla VAE？** vanilla VAE 的 posterior collapse 会让 encoder 输出退化成 prior，那 $\mathbf{E}_\psi(\hat{\mathbf{a}}^k)$ 和 $\mathbf{E}_\psi(\mathbf{a}^0)$ 可能都坍塌到原点，guidance 信号消失。InfoVAE 的 mutual information 项保证 latent 编码了有意义的 action 信息。

### 3.8 Algorithm 2 & 3 解析

**Algorithm 2 (Flow-based VLAs)**:
```
repeat:
  Sample (o_t, l, a_{t:t+L-1}) from D_adaptation
  Sample ε ~ N(0,I), τ ~ U([0,1])
  Corrupt: a^τ = τ·a + (1-τ)·ε
  Get latents: ẑ = E_ψ(a^τ), z = E_ψ(a)
  Compute g = -∇_{a^τ} ||z - ẑ||²
  Ground truth velocity: v_t = a - ε
  Update: θ ← θ - η ∇_θ || v_θ(a^τ; τ, o_t, l) + (1-τ)/τ · λ·g - v_t ||²
until converged
```

**Algorithm 3 (Diffusion-based VLAs)**:
```
repeat:
  Sample (o_t, l, a_{t:t+L-1}) from D_adaptation
  Sample ε ~ N(0,I), k ~ U({1,...,T})
  Corrupt: a^k = √(ᾱ_k)·a + √(1-ᾱ_k)·ε
  Get latents: ẑ = E_ψ(a^k), z = E_ψ(a)
  Compute g = -∇_{a^k} ||z - ẑ||²
  Update: θ ← θ - η ∇_θ || ε_θ(a^k; k, o_t, l) - √(1-ᾱ_k)·λ·g - ε ||²
until converged
```

注意 guidance 只在训练时用（因为依赖 ground truth $\mathbf{a}^0$），推理时是 standard VLA。这跟传统 classifier guidance 在采样时计算 gradient 不一样，ATE 是把 guidance 烧进训练 loss 里，相当于一种特殊的 score distillation。

---

## 4. 架构图解析（Figure 2）

### Figure 2(a): Stage 1 - Unified Action Latent Space

左侧：pre-training actions $\bar{\mathbf{a}}_{t:t+H-1}$ → $\mathbf{E}_\phi$ → latent $z$ with prior $\mathcal{N}(0, I)$，训练好后得到 $q_\phi(z)$（multimodal distribution，每个 mode 对应一个 pre-training embodiment）

右侧：adaptation actions $\tilde{\mathbf{a}}_{t:t+L-1}$ → $\mathbf{E}_\psi$ → latent $z$ with prior $q_\phi(z) \approx \mathcal{N}(\mu_\phi, \Sigma_\phi)$，reverse KL 让 adaptation latent 嵌入 pre-trained latent 的某个 mode

可视化上，论文展示 pre-trained latent distribution 是个多峰形状，adaptation latent cluster 被"吸引"到其中一个峰里。

### Figure 2(b): Stage 2 - Steering Adaptation

Diffusion VLA $\epsilon_\theta$ 接收 noisy action $\mathbf{a}^k$ + observation $\mathbf{o}_t$ + language $l$，预测 noise $\epsilon_\theta(\mathbf{a}^k, k, \mathbf{o}_t, l)$。

Parallel 路径上，$\mathbf{E}_\psi$ 把 noisy action $\mathbf{a}^k$ 和 ground truth $\mathbf{a}^0$ 编码到 latent space，计算 latent distance，gradient backpropagate 到 $\mathbf{a}^k$ 得到 $g$。

最终 calibrated noise prediction $\hat{\epsilon} = \epsilon_\theta - \sqrt{1 - \bar{\alpha}_k} \cdot \lambda \cdot g$ 用于 denoising loss。

---

## 5. 实验数据深度解析

### 5.1 Simulation: RoboTwin 1.0（Table 2）

17 个 task，比较 RDT-1B 和 π0 with/without ATE。

| Method | RDT-1B | RDT-1B + ATE | π0 | π0 + ATE |
|--------|--------|--------------|----|-----------|
| Average | 31.8% | 41.6% (+9.8) | 36.1% | 44.8% (+8.7) |

亮点 gain:
- **Empty Cup Place**: RDT 22% → 61% (+39)，π0 32% → 36% (+4)
- **Put Apple Cabinet**: RDT 20% → 45% (+25)，π0 34% → 55% (+21)
- **Dual Bottles Pick (Easy)**: π0 48% → 85% (+37)
- **Blocks Stack (Easy)**: π0 30% → 50% (+20)
- **Block Handover**: RDT 69% → 91% (+22)

也有少数 negative case:
- Bottle Adjust (RDT): 53% → 37% (-16)
- Tool Adjust (RDT): 54% → 42% (-12)
- Pick Apple Messy (π0): 18% → 11% (-7)

这些 task 通常 baseline 已经很高或者 task 本身 multimodality 强（多种抓取方式），guidance 可能 over-constrain。

### 5.2 Simulation: ManiSkill3（Table 3）

| Task | RDT-1B | RDT-1B + ATE |
|------|--------|--------------|
| Push Cube | 65.2% | 78.4% (+13.2) |
| Pick Cube | 7.6% | 14.8% (+7.2) |
| Average | 36.4% | 46.6% (+10.2) |

Push Cube 是 contact-rich task，ATE 提升 13.2 个百分点，说明 guidance 对需要精细 force control 的 task 有效。

### 5.3 Real-World: Dual-Arm RealMan 7-DoF

每个 task 160 trajectories，长 horizon (>1 min)，bimanual coordination。RealMan 是 7-DoF 单臂 × 2，跟 pre-training data 里的 6-DoF 单臂完全不同 embodiment。

| Task | Baseline (120k) | ATE (120k) | ATE (90k) | ATE (60k) |
|------|-----------------|------------|-----------|-----------|
| Cook Bun | 15% | 100% | 100% | 70% |
| Pick Bun | 20% | 70% | 50% | 40% |
| Make Sandwich | 20% | 55% | 40% | 30% |
| Use Toaster | 12% | 47.5% | 30% | 25% |
| **Average** | **16.7%** | **58.1%** | **40%** | **41.25%** |

**32% 的 absolute gain**，这是论文最 striking 的结果。重点观察：
- Cook Bun 在 60k steps 就 70%，90k 就 100%，baseline 120k 才 15%。说明 ATE 不只是 final performance 高，convergence 也快得多。
- Use Toaster 用 graded metric（完全插入 1.0，倾斜 0.25，部分 0.5），因为 task 难度太高二元 success 不够 informative。

### 5.4 Generalization（Tables 4, 5）

- **Illumination**: 低/高/闪烁光照下，ATE 在 Cook Bun 上仍有 60% success，baseline 0%
- **Spatial Generalization**: 物体位置偏移 6.5 cm radius，ATE Cook Bun 40% / Pick Bun 60% / Make Sandwich 40%
- **Visual Distractor**: 随机放置无关物体（水果、puzzle），ATE Cook Bun 40% / Pick Bun 80%
- **Human Disturbance**: 人中途拿走物体，ATE Pick Bun 40% / Make Sandwich 60%，baseline 几乎全 fail

ATE 在 disturbance 鲁棒性上的优势来自 latent space 约束：policy 被限制在 structured latent manifold 内，不会因扰动跳出 prior 之外。

### 5.5 Ablation: Two-stage vs Single-stage VAE（Figure 6）

Single-stage VAE 只在 adaptation data 上训，没有 pre-training latent prior。Two-stage 在 multiple task 上 consistently 更好，证实 aligned latent space 的关键作用。

### 5.6 Force Analysis（Appendix G）

Figure 13 展示 Cook Bun 任务 6-axis force 数据。Baseline 经常施加过大下压力导致 steamer 变形，ATE 的 force 曲线更平滑、幅值更小。这是 ATE "意外"的副效益：latent manifold 约束让 motion 更安全 smoother。

---

## 6. 训练超参数细节

### InfoVAE Pre-training（Table 9）:
- Latent dim: 512
- Temporal input length: 64 (RDT) / 50 (π0) / 14 (DP)
- Input dimensionality: 128 (RDT) / 32 (π0) / 14 (DP)
- Optimizer: Adam, lr = 1e-4, weight decay = 1e-4
- Batch size: 64

### Fine-tuning（Table 10）:
- RDT: chunk 64, batch 64, lr 1e-4, 100k steps
- π0: chunk 50, batch 24, lr 2.5e-5, 60k steps  
- DP: chunk 8, batch 128, lr 1e-4, 300 epochs

### Real Robot:
- π0: chunk 20, batch 48, lr 2.5e-5, 120k steps

### 训练时长:
- Stage 1 Step 1 (pre-train VAE): 12 小时
- Stage 1 Step 2 (adapt VAE): 0.5 小时
- 硬件: 4× A100 (RDT/π0) 或 1× A100 (DP)

---

## 7. 关联技术联想

### 7.1 Latent Action 系列

- **UniVLA** (Bu et al. 2025, https://arxiv.org/abs/2502.14420): 用 VQ-VAE 从 cross-embodiment videos 提取 task-centric latent actions for pre-training
- **UniACT** (Zheng et al. 2025, https://arxiv.org/abs/2501.10181): 学习 discrete atomic behavior codebook 形成 universal latent action
- **LAPA** (Ye et al. 2025, https://arxiv.org/abs/2410.11758): Latent Action Pretraining from Videos，ICLR 2025
- **Latent Action Diffusion** (Bauer et al. 2025, https://arxiv.org/abs/2506.14608): cross-embodiment manipulation with VAE + contrastive learning + hand motion retargeting

ATE 跟这些工作的本质区别在于：它们都试图构造 universal latent space for pre-training，但 ATE 显式处理 pre-training 和 adaptation 之间的 distribution gap，用 reverse KL mode-seeking 嵌入 adaptation latent 到 pre-training latent 的 mode 里。

### 7.2 Classifier Guidance 系列

- **Diffusion Models Beat GANs** (Dhariwal & Nichol 2021, https://arxiv.org/abs/2105.05233): classifier guidance 的开山之作，在 sample time 用 classifier gradient 引导 diffusion sampling
- **Universal Guidance** (Bansal et al. 2023, https://arxiv.org/abs/2305.12266): 把 guidance 推广到任意 guidance function
- **Motion Planning Diffusion** (Carvalho et al. 2023, https://arxiv.org/abs/2310.07144): ATE guidance function 的直接 inspiration，用 energy-based model 构造 classifier
- **DexHandDiff** (Liang et al. 2025, https://arxiv.org/abs/2412.10687): dexterous manipulation 的 diffusion planning with latent guidance

ATE 把 classifier guidance 从 sampling time 搬到 training time，因为 ground truth action $\mathbf{a}^0$ 只在训练时可得。这本质上是一种 **score distillation**，类似 SDS (Score Distillation Sampling, Poole et al. 2022 https://arxiv.org/abs/2205.15191) 的逻辑：用一个 frozen encoder/guidance 提供 gradient signal，让 student model 学到符合某些 constraint 的 distribution。

### 7.3 VLA 主流架构

- **Diffusion Policy** (Chi et al. 2023, https://arxiv.org/abs/2303.04137): diffusion 在 robotic manipulation 的最早 application
- **RDT-1B** (Liu et al. 2024, https://arxiv.org/abs/2410.07864): 1B 参数 bimanual diffusion VLA
- **π0** (Black et al. 2024, https://arxiv.org/abs/2410.24164): flow matching VLA, Physical Intelligence
- **π0.5** (Intelligence et al. 2025, https://arxiv.org/abs/2504.16054): π0 的 open-world generalization 版本
- **OpenVLA** (Kim et al. 2024, https://arxiv.org/abs/2406.09246): open-source 7B VLA
- **FAST** (Pertsch et al. 2025, https://arxiv.org/abs/2501.09747): DCT-based action tokenization 提速 fine-tuning
- **DexVLA** (Wen et al. 2025, https://arxiv.org/abs/2502.05855): plug-in diffusion expert
- **DiVLA / DiffusionVLA** (Wen et al. 2025, https://arxiv.org/abs/2505.02503): unified diffusion + autoregression
- **DITA** (Hou et al. 2025, https://arxiv.org/abs/2503.19757): scaling diffusion transformer VLA
- **SmolVLA** (Shukor et al. 2025, https://arxiv.org/abs/2506.01844): affordable efficient VLA
- **TinyVLA** (Wen et al. 2025, https://arxiv.org/abs/2409.12514): data-efficient VLA
- **GROOT N1** (NVIDIA 2025, https://arxiv.org/abs/2503.14734): humanoid foundation model
- **H-RDT** (Bi et al. 2025, https://arxiv.org/abs/2503.04136): human manipulation enhanced bimanual

### 7.4 Robotic Datasets

- **Open X-Embodiment** (OXE Collaboration 2023, https://arxiv.org/abs/2310.08864): 多 embodiment 联合 dataset
- **DROID** (Khazatsky et al. 2024, https://arxiv.org/abs/2403.12945): in-the-wild robot manipulation
- **BC-Z** (Jang et al. 2021, https://arxiv.org/abs/2202.05810): zero-shot task generalization
- **RoboTwin 1.0** (Mu et al. 2025, https://arxiv.org/abs/2504.02049): dual-arm benchmark with digital twins
- **ManiSkill3** (Tao et al. 2024, https://arxiv.org/abs/2410.00425): GPU-parallelized robotics simulation, contact-rich

### 7.5 VLM Backbones

- **PaLI-X** (Chen et al. 2023, https://arxiv.org/abs/2305.18565): scaling multilingual VLM
- **PaLIGemma** (Beyer et al. 2024, https://arxiv.org/abs/2407.07726): 3B versatile VLM
- **Prismatic** (Anschütz & Le Bras 2023, https://arxiv.org/abs/2402.09449): OpenVLA backbone
- **BLIP-2** (Li et al. 2023, https://arxiv.org/abs/2301.12597): bootstrapping language-image pre-training
- **Qwen2-VL** (Wang et al. 2024, https://arxiv.org/abs/2409.12191): any resolution VLM

### 7.6 Teleoperation Hardware

Appendix H 描述了 dual-operator Vision Pro teleoperation:
- 用 TrackingStreamer 提取 Vision Pro 的 hand keypoint estimation
- Nintendo Joy-Con 控制 gripper open/close（避免视觉识别误差）
- Pinocchio library 做 inverse kinematics（Pinocchio: https://stack-of-tasks.github.io/pinocchio/）
- Dual operator 的优点：处理 simultaneous bimanual actions、多视角视觉、操作员分工

这个 teleop setup 类似 ALOHA 和 Gello 的思路，但用 Vision Pro 替代 leader arm，更轻量。

### 7.7 Energy-Based Model 视角

ATE 的 guidance function 本质是个 energy-based model (EBM):
$$
p_\psi(y | \mathbf{a}) \propto \exp(-E_\psi(\mathbf{a}, y))
$$

其中 $E_\psi(\mathbf{a}, y) = \|\mathbf{E}_\psi(\mathbf{a}) - \mathbf{E}_\psi(\mathbf{a}^0)\|^2$ 是 energy function。

EBM 在 diffusion guidance 中的应用参见:
- **Dual Diffusion Distillation** 系列
- **EBM 早期工作**: Du & Mordatch 2019 "Implicit Generation and Modeling with Energy-Based Models" (https://arxiv.org/abs/1903.08689)
- **Score-based generative models**: Song et al. 2021 "Score-Based Generative Modeling through SDEs" (https://arxiv.org/abs/2011.13456)

### 7.8 MMD 替代 KL

Maximum-Mean Discrepancy 替代 KL 的好处:
- 不需要 parametric form
- 在高维空间里 optimization tractable
- kernel-based，可以用 Gaussian kernel 或其他 RKHS kernel

参考:
- Li et al. 2015 "Generative Moment Matching Networks" (http://proceedings.mlr.press/v37/li15.html)
- Dziugaite et al. 2015 "Training Generative Neural Networks via MMD" (http://auai.org/uai2015/proceedings/papers/230.pdf)

### 7.9 VLA Fine-tuning Efficiency 系列

ATE 跟这一系列工作互补:
- **VLA-Cache** (Xu et al. 2025, https://arxiv.org/abs/2502.02175): adaptive token caching
- **FAST** (Pertsch et al. 2025, https://arxiv.org/abs/2501.09747): DCT 压缩 action tokens
- **MoLE-VLA** (Zhang et al. 2025, https://arxiv.org/abs/2503.20384): dynamic layer skipping via mixture-of-layers
- **OpenVLA fine-tuning** (Kim et al. 2025, https://arxiv.org/abs/2502.19645): speed/success 优化

这些工作主要在 parameter 和 token 层面优化，ATE 在 distribution 层面优化，可以叠加使用。

---

## 8. 整体 Intuition

让我把整条逻辑链重新梳理一遍 build intuition:

1. **问题本质**: pre-trained VLA 学到了一个 action distribution $p_{\mathcal{D}_{\text{pretrain}}}(\bar{\mathbf{a}})$，adaptation 时要 shift 到 $p_{\mathcal{D}_{\text{adapt}}}(\tilde{\mathbf{a}})$。如果两个 distribution 在 raw action space 完全不可比（DoF 不同），直接 fine-tune 的 gradient signal 不 inform 改变方向，需要海量数据。

2. **Stage 1 智慧**: 不在 raw action space 算 distance，而在 latent space 算。先训一个 VAE 把 pre-training actions 压成 multimodal latent $q_\phi(z)$（每个 mode 对应一类 embodiment 的 action style）。再训 adaptation VAE，用 reverse KL 把 adaptation latent 挤进 $q_\phi(z)$ 的某个 mode。这样不同 embodiment 的 action 在 latent space 里就有了 well-defined 的 distance。

3. **Stage 2 智慧**: 直接 fine-tune 时 policy 不知道 latent space 这回事，可能跑到 latent manifold 之外，丢失 pre-trained prior。ATE 把 latent distance 算成 energy-based classifier 的 gradient，inject 到 diffusion/flow 训练 loss 里，相当于每一步都在 latent space 里给 policy 一个"拉力"，让它别离 pre-trained prior 太远。

4. **训练时 vs 推理时**: ATE 的 guidance 只在训练时生效（因为需要 ground truth $\mathbf{a}^0$），推理时是标准 VLA。这点跟传统 classifier guidance（Dhariwal & Nichol 2021）不同——传统方法在 sample time 算 gradient，每步 forward classifier。ATE 是把 guidance 烧进 policy 参数里，推理 zero overhead。

5. **为什么 work**: 三个原因叠加
   - Reverse KL mode-seeking 让 latent space 有 hierarchical 结构，adaptation 不破坏 pre-trained prior
   - Latent guidance 提供了 action distribution level 的 supervision signal，比纯 raw action L2 loss 信息量大
   - Latent manifold 约束相当于 regularizer，防止 fine-tuning 灾难性遗忘

6. **为什么 efficient**: 只训练两个轻量 VAE（参数量远小于 VLA），不动 VLA 架构，加少量 gradient computation。Stage 1 Step 1 是一次性 offline 训练，Step 2 是 0.5 小时 fast tuning。

7. **隐含 bonus**: 因为 policy 被约束在 latent manifold 内，motion 自动 smoother，force control 自动更稳（Appendix G 的 6-axis force 数据证明）。

---

## 9. 局限与潜在问题（hallucination 的联想）

从 paper 读出的潜在 issue:

1. **Reverse KL 的 mode collapse 风险**: mode-seeking 是双刃剑。如果 adaptation data 本身 multimodal（多种抓取 style），reverse KL 可能让 adaptation latent 只 collapse 到一个 mode，丢失多样性。从 Table 2 看 Bottle Adjust 和 Tool Adjust 的 negative gain 可能就是这个原因。

2. **Guidance scale λ 调参敏感**: 太大会 over-constrain，太小没效果。Paper 没给出 ablation on λ，可能需要 per-task tuning。

3. **Stage 1 VAE pre-training 的 embodiment coverage**: 如果 pre-training dataset 没覆盖某类 action（比如 humanoid 高 DoF），$q_\phi(z)$ 可能没有合适 mode 给 adaptation 嵌入，效果打折。

4. **Encoder 对 noisy action 的 robustness**: guidance 在所有 diffusion step $k$ 都计算，包括高噪声 step。论文 claim InfoVAE 比 vanilla VAE robust，但没有 ablation 验证 encoder 对 $\mathbf{a}^k$（噪声大）vs $\mathbf{a}^0$（clean）的 latent 一致性。

5. **Cross-task 任务的 multimodality**: latent space 的 mode 是按 embodiment 还是 task 组织？如果 task 间共享 embodiment，mode-seeking 可能错误地 merge 不同 task 的 action。

6. **Long-horizon coordination**: real-world experiment 显示 ATE 在长 horizon 任务（>1 min）有效，但 latent space 是否能 capture temporal structure 没显式讨论。Paper 用 chunk-level latent（一个 chunk 一个 z），如果 chunk 间依赖强可能不够。

7. **Generalization 评估的样本量**: Tables 4, 5 每个 condition 只 5 trials，统计意义有限。

---

## 10. 可能的扩展方向

1. **Hierarchical latent space**: 在 chunk-level latent 之上再加 trajectory-level latent，capture long-horizon structure。类似 hierarchical VAE 或 VQ-VAE two-level。

2. **Cross-task embedding**: 用 task embedding 把不同 task 的 latent mode organize，让新 task 可以 query 已有 mode。

3. **Online adaptation**: 现在 ATE 是 offline fine-tuning，可以扩展成 online RL setting，guidance 在 inference 时也用（用 demonstration 或 human feedback 作为 ground truth）。

4. **Adversarial guidance**: 用 discriminator 代替 latent distance（类似 GAN），可能更 sharp 的 distribution matching，但要小心 instability。

5. **Multi-embodiment 同时 adaptation**: 现在 ATE 只支持 single target embodiment，可以扩展到 multi-target，让一个 pre-trained VLA 同时 fine-tune 多个 embodiment（multi-task formulation）。

6. **VLM-based guidance**: 用 VLM 的 captioning 或 VQA 作为 guidance function（"is this action reasonable for the task?"），更 semantic 的 constraint。

7. **Diffusion Distillation**: ATE 训练时用 guidance，推理时是 standard VLA。可以考虑 distillation，把 guidance 蒸馏成 inference-time 的 simpler policy。

8. **Connection 到 RLHF**: latent guidance 跟 RLHF 的 KL penalty 思想类似——都是用 reference distribution 约束 policy 不要离太远。可以借鉴 RLHF 的 optimization theory (PPO + KL penalty)。

---

## 11. 总结

ATE 的核心贡献是把 VLA adaptation 问题重新 frame 成 latent distribution alignment + guidance 问题，避开了 raw action space 不可比的难题。技术上把 InfoVAE、reverse KL mode-seeking、Tweedie's formula score function、classifier guidance、energy-based model 这几条线缝合在一起，plug-and-play 不动 VLA 架构。实验上 +9.8% simulation、+32% real-world dual-arm 的 gain 是 significant 的。

值得 take-away 的 intuition:
- **Distribution mismatch 是 VLA adaptation 的根本瓶颈**，不是 parameter count 问题
- **Latent space 算 distance 比原始 action space 算 distance 更 meaningful** for cross-embodiment
- **Reverse KL mode-seeking 是 elegant trick** 构造 hierarchical structured latent
- **Guidance 烧进训练 loss**（vs sampling time）让 inference zero overhead
- **Latent manifold 约束自动带来 motion smoothness**，意外 bonus

跟整个 VLA 生态的关系：ATE 是 plug-and-play 的 adaptation 层，可以跟 FAST 的 tokenization、OpenVLA 的 LoRA、VLA-Cache 的 caching 等正交叠加，是 foundation model 时代 fine-tuning recipe 的一个 useful primitive。

完整代码和 project: https://align-then-steer.github.io/ / https://github.com/TeleHuman/Align-Then-Steer
