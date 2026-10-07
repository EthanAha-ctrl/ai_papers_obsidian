---
source_pdf: DIFFUSIONBLOCKS BLOCK-WISE NEURAL NETWORK TRAINING VIA DIFFUSION INTERPRETATION.pdf
paper_sha256: eb109a523b6ee1e014fb6091c53067ac10d3f4f5f210ca0679d713629673d885
processed_at: '2026-08-03T21:55:33-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# DiffusionBlocks 人话版

Andrej，我用最直觉的方式再讲一遍。

---

## 一句话概括

训练大模型时 backprop 要存所有层的中间结果，memory 随深度线性涨。这篇论文说：**residual network 的每一层其实就是 diffusion model denoising 的一步**，所以把网络切成几个 block，每个 block 只管一段 noise level，独立训练就行——memory 直接降 B 倍，效果跟 end-to-end 几乎一样，有时候还更好。

---

## 核心直觉：Residual = Denoise

你写过 ResNet, 你知道 residual block 长这样：

$$\mathbf{z}_\ell = \mathbf{z}_{\ell-1} + f_{\theta_\ell}(\mathbf{z}_{\ell-1})$$

变量解释：$\mathbf{z}_\ell$ 是第 $\ell$ 层的输出, $f_{\theta_\ell}$ 是这一层的 transform, $\theta_\ell$ 是参数。

Neural ODE 那帮人早就说过这是 ODE 的 Euler step。但那只是"它像 ODE"这种 fuzzy 的说法，没告诉你具体是哪个 ODE, 也没给实际的 training recipe。

这篇论文做了一个更 sharp 的 observation: 这个 update rule **就是** VE diffusion 的 reverse process 的 Euler discretization。

VE diffusion 的 reverse ODE (Eq. 1):

$$\frac{d\mathbf{z}_\sigma}{d\sigma} = -\sigma \nabla_\mathbf{z} \log p_\sigma(\mathbf{z}_\sigma)$$

变量解释：$\sigma$ 是 noise level (越大越噪), $p_\sigma$ 是 noise level $\sigma$ 处的 marginal 分布, $\nabla_\mathbf{z} \log p_\sigma$ 是 score function。

用 Tweedie's formula 把 score 换成 denoiser $D_\theta$（预测 clean data 的网络），然后做 Euler step on $\sigma$，你就得到 (Eq. 4):

$$\mathbf{z}_{\sigma_\ell} = \mathbf{z}_{\sigma_{\ell-1}} + \frac{\Delta\sigma_\ell}{\sigma_{\ell-1}} \bigl(\mathbf{z}_{\sigma_{\ell-1}} - D_\theta(\mathbf{z}_{\sigma_{\ell-1}}, \sigma_{\ell-1})\bigr)$$

下标 $\ell-1$ 指"对应上一层 noise level 的量", $\Delta\sigma_\ell = \sigma_{\ell-1} - \sigma_\ell > 0$ 是 noise decrement。

**对比一下 ResNet update**：$\mathbf{z}_\ell = \mathbf{z}_{\ell-1} + f_{\theta_\ell}(\mathbf{z}_{\ell-1})$

结构上几乎一样：都是 skip + transform。差别只是 transformer block 的 output $f$ 现在被 reinterpret 成 denoiser $D$，混入系数从 $(1, 1)$ 变成 $(\alpha, \beta) = (1 + \Delta\sigma/\sigma, -\Delta\sigma/\sigma)$。

**这意味着你几乎不用改网络**。每个 transformer block 的 attention/MLP 该咋算咋算，只是输出乘个 scalar $\beta$, skip 乘个 scalar $\alpha$。再加一个 noise level $\sigma$ 的 conditioning（通过 AdaLN 之类的）。Done。

---

## 为什么独立训练 OK？

这是 paper 的 soul。一般的 block-wise training（Forward-Forward, greedy layer-wise 等）失败的原因是：local objective 跟 global objective 之间没有数学联系。你给每层设个 ad-hoc local loss，optimize 它，能不能 assemble 出 end-to-end 的工作能力？理论上没人能保证。

Diffusion 模型有一个 deep property: **不同 noise level $\sigma$ 上的 denoiser 在 training 时数学独立**。$\sigma=5$ 处的 score 学得好不好，跟 $\sigma=1$ 处学得好不好一点关系都没有。这就是为什么 eDiff-I 那种用 ensemble of expert denoisers (一个 denoiser 只负责一段 $\sigma$) 是 OK 的——理论上是 sound 的。

DiffusionBlocks 把这个 property 直接拿来用。每个 block $b$ 负责一段 $\sigma$ 范围 $[\sigma_b, \sigma_{b-1}]$，独立训练 (Eq. 6):

$$\mathcal{L}_b(\theta_b) = \mathbb{E}_{(\mathbf{x},\mathbf{y}) \sim p_{\text{data}},\, \sigma \sim p_{\text{noise}}^{(b)},\, \boldsymbol{\epsilon} \sim \mathcal{N}(0, I)} \bigl[ w(\sigma) \cdot \text{Loss}\bigl(\bar{f}_{\theta_b \mid \sigma}(\mathbf{x}, \mathbf{y} + \sigma\boldsymbol{\epsilon}), \mathbf{y}\bigr) \bigr]$$

变量解释：
- $p_{\text{noise}}^{(b)}$：把 $p_{\text{noise}}$ 限制在 block $b$ 的 noise interval 然后 renormalize
- $w(\sigma)$：weighting, EDM 给的是 $w(\sigma) = (\sigma^2 + \sigma_{\text{data}}^2)/(\sigma \cdot \sigma_{\text{data}})^2$, 用来平衡 gradient magnitude
- $\mathbf{y}$：clean target
- $\boldsymbol{\epsilon}$：标准高斯 noise

**训练时**：random 选一个 block $b$，从 $p_{\text{noise}}^{(b)}$ 采一个 $\sigma$，给 clean target 加 noise $\mathbf{z}_\sigma = \mathbf{y} + \sigma\boldsymbol{\epsilon}$，forward 只跑这一个 block 的 $L/B$ 层，loss 算 denoising error, backward 只跑这 $L/B$ 层。其他 block 完全不参与——既不传 activation 也不传 gradient。

**Inference 时**：从 $\sigma_{\max}$ 走到 $\sigma_{\min}$，每个 $\sigma$ 调用对应 block 一次, 按 Euler step 更新。

---

## Equi-Probability Partitioning: 别均匀切

最 naive 的做法是把 $[\sigma_{\min}, \sigma_{\max}]$ 均匀切 B 段。这有个问题：denoising difficulty 不是均匀的。中间 noise level（结构开始 emerging 但还没清晰）最难，纯噪声或几乎 clean 都容易。

Karras et al. (2022) 的 EDM 用 log-normal $p_{\text{noise}}$ 把训练概率质量集中在中间：

$$\log\sigma \sim \mathcal{N}(P_{\text{mean}}, P_{\text{std}}^2), \quad P_{\text{mean}} = -1.2, P_{\text{std}} = 1.2$$

如果你 uniform 切 B 段，中间 block 会"超载"——它那段虽然宽度跟别的 block 一样, 但 probability mass 密集, 训练 sample 多。

Equi-probability partitioning 的做法：让每个 block 处理相同 probability mass $1/B$:

$$\int_{\sigma_b}^{\sigma_{b-1}} p_{\text{noise}}(\sigma)\, d\sigma = \frac{1}{B}$$

解析解：

$$\sigma_b = \exp\bigl(P_{\text{mean}} + P_{\text{std}} \cdot \Phi^{-1}(q_b)\bigr)$$

变量解释：$\Phi^{-1}$ 是 inverse standard normal CDF, $q_b = q_{\min} + (b/B)(q_{\max} - q_{\min})$, $q_{\min/\max} = \Phi\bigl((\log\sigma_{\min/\max} - P_{\text{mean}})/P_{\text{std}}\bigr)$。

效果（Figure 4）：中间 block 的 interval 在 log-σ 空间很窄（mass 密集），边缘 block 的 interval 很宽（mass 稀疏）。每个 block 工作量相同。

Ablation (Table 7, CIFAR-10, 12 层分 3 block):

| Partitioning | Layer Dist | FID ↓ |
|---|---|---|
| Uniform | [4,4,4] | 43.53 |
| Equi-Probability | [4,4,4] | **38.03** |

5 分的差距，单纯靠 partitioning strategy。

---

## 怎么应用到不同架构

Paper 给的 3-step recipe 适用于 ViT, DiT, Llama-style AR, masked diffusion LM, recurrent-depth model。对每种 task, $\mathbf{x}$ (conditioning) 和 $\mathbf{y}$ (target) 含义不同（Appendix A 有表）：

- **Image classification**：$\mathbf{x}$ = image patches, $\mathbf{y}$ = class label embedding. Noise 加在 label embedding 上。Loss 用 cross-entropy。
- **Image generation (DiT)**：$\mathbf{x}$ = (noisy latents, optional class label), $\mathbf{y}$ = clean latents. 标准 diffusion，没什么特殊。
- **AR text generation**：$\mathbf{x}$ = previous clean tokens, $\mathbf{y}$ = next token embedding. Loss 用 cross-entropy on token。需要 causal mask trick 维持 autoregressive 性质（用 sequence concatenation 让 noisy future tokens attend 到 clean past tokens）。
- **Masked diffusion LM**：partition 的是 masking schedule $\alpha(t)$ 不是 noise level $\sigma$。证明 (Appendix D) training mass 在 $\alpha$ 上 uniform, 所以按 $\alpha_b = 1 - b/B$ 切。
- **Recurrent-depth (Huginn)**：把 K 次 recurrence 当成 K-step diffusion, single-pass 训练, 干掉 BPTT。

---

## 实验结果速览

| 任务 | 架构 | baseline | DiffusionBlocks | 备注 |
|---|---|---|---|---|
| CIFAR-100 分类 | ViT-12, B=3 | 60.25% | 59.30% | FF 只有 7.85% |
| CIFAR-10 生成 | DiT-S/2, B=3 | FID 39.83 | FID 37.20 | test split 上更好 |
| ImageNet-256 生成 | DiT-L/2, B=3 | FID 12.09 | FID 10.63 | test split 上更好 |
| text8 MDM | DiT-12, B=3 | BPC 1.56 | BPC 1.45 | discrete 也 work |
| LM1B AR | Llama-2 style, B=4 | MAUVE 0.50 | MAUVE 0.71 | AR 也能转 |
| OWT AR | Llama-2 style, B=4 | MAUVE 0.85 | MAUVE 0.82 | 持平 |
| LM1B recurrent | Huginn | MAUVE 0.49 | MAUVE 0.70 | 省 10x compute |

**最 striking 的 result**: ImageNet-256, B=2 (12 layers/block) 比 B=1 (24 layers, end-to-end) FID 还好（9.90 vs 12.09）。

| B (blocks) | FID ↓ | L/B |
|---|---|---|
| 1 (end-to-end) | 12.09 | 24 |
| **2** | **9.90** | 12 |
| 3 | 11.11 | 8 |
| 4 | 11.90 | 6 |
| 6 | 14.43 | 4 |

Moderate block count 比 end-to-end 还好，这是个 surprising result。论文 hypothesize 两个 factor：
1. 每个 block 直接通过 denoising objective 连到 target, gradient path 短, signal 跟 end-to-end 的长链 backprop 不同
2. 不同 block 在不同 noise range 上 specialization, 类似 implicit curriculum learning (Bengio 2009)

---

## 跟 Activation Checkpointing 的对比

Karpathy 你肯定会问。设 $L$ 层, 每层 parameter size $P$, activation size $A$, Adam optimizer 需要额外 $2P$:

| Pattern | Memory |
|---|---|
| Standard | $(4P + A) \cdot L$ |
| Activation checkpointing | $4PL + A$ (只省 activation) |
| DiffusionBlocks | $(4P + A) \cdot L/B$ (所有 components 都省) |
| Combined | $\approx (4P + A) \cdot L/B$ |

关键区别：**activation checkpointing 只省 activation memory**, 对 parameter/gradient/optimizer states 无效。大模型上 optimizer states 占大头, DiffusionBlocks 把这些也降 B 倍。

Compute: activation checkpointing 增加训练时间 ~4/3 倍 (recompute activation), DiffusionBlocks 保持这个比例。但 DiffusionBlocks 还允许 embarrassingly parallel block training——B 个 block 在 B 个 GPU 上完全独立训, 零通信开销。

---

## Limitations

1. 要求每层 input/output 同维度, U-Net 这种 down/up-sampling 的不行
2. Block granularity 怎么选没 principled 方法
3. 为什么 B=2 比 B=1 好没理论解释
4. 只试了 from-scratch training, 预训练模型 fine-tune 成 DiffusionBlocks 没做
5. 只用 Euler sampler, DPM-Solver++ / UniPC 等没试

---

## Open Questions 值得深挖

**为什么 moderate block count 反超 end-to-end?** 如果能 rigor 证明这是 denoising regularization 的某种 explicit form, 可能给 deep network optimization 一个新 angle。论文 hypothesize specialization + curriculum, 但没证明。

**Pre-trained model fine-tune 成 DiffusionBlocks 可行吗?** 现有 LLM/Diffusion model 已经有合理的"层次化 representation", fine-tune 时把 layer 切 block, 每个 block 负责一段 noise, 可能比 from-scratch 更高效。

**Recurrence = Diffusion 这个同构还能用到别处吗?** Huginn 那个 case 暗示任何 recurrent refinement (looped transformer, universal transformer 等) 都可能用 single-pass diffusion training 代替 BPTT。

---

## Reference Web Links

- Paper: https://arxiv.org/abs/2507.06184
- Sakana AI project page: https://sakana.ai/diffusionblocks
- Code: https://github.com/SakanaAI/DiffusionBlocks
- EDM (Karras et al. 2022): https://arxiv.org/abs/2206.00364
- Neural ODE (Chen et al. 2018): https://arxiv.org/abs/1806.07366
- Score-based diffusion SDE (Song et al. 2021): https://arxiv.org/abs/2011.13456
- DiT (Peebles & Xie 2023): https://arxiv.org/abs/2212.09748
- Forward-Forward (Hinton 2022): https://arxiv.org/abs/2212.13345
- NoProp (Li et al. 2025): https://arxiv.org/abs/2503.24322
- Huginn (Geiping et al. 2025): https://arxiv.org/abs/2502.05171
- MD4 (Shi et al. 2024): https://arxiv.org/abs/2412.18578
- Masked diffusion LM (Sahoo et al. 2024): https://arxiv.org/abs/2406.07524
- SEDD (Lou et al. 2024): https://arxiv.org/abs/2310.16834
- Block Diffusion (Arriola et al. 2025): https://arxiv.org/abs/2504.01055
- eDiff-I (Balaji et al. 2023): https://arxiv.org/abs/2211.01324
- Curriculum learning (Bengio et al. 2009): https://dl.acm.org/doi/10.1145/1553374.1553380
- Universal Transformers (Dehghani 2019): https://arxiv.org/abs/1807.03819
- Haber & Ruthotto 2017 (stable architectures): https://iopscience.iop.org/article/10.1088/0266-5611/34/1/014004

---

## Intuition for Karpathy

你做过 micrograd, 你写 ResNet, 你训 transformer。DiffusionBlocks 的 thesis 如果用一句话说就是：**residual update 跟 diffusion denoise 在数学结构上是同一个操作**，所以网络深度 = diffusion step 数，block-wise training = noise-level-wise training, 而 noise-level-wise training 在 score matching 理论里本来就是 independent 的。

这件事为什么 important: 它给 layer-wise training 提供了第一个 *principled* 的 local objective。每个 block 知道自己要做什么 (denoise 一段 σ range)，collectively 这些 block assemble 出一个 faithful approximation of 全 reverse process。Forward-Forward / greedy layer-wise 之类的方法没这个性质, 所以效果差。

最 interesting 的 open question 不是"能不能 work" (显然能), 是"为什么 B=2 比 B=1 更好"。这暗示 deep network 的某些 generalization 性质可能来自把 long backprop chain 切成短的 direct-to-target path, 而非传统的"端到端最 optimal"信仰。如果这点能 rigor 化, 可能打开一类新的 optimization principle。

---

# DiffusionBlocks: 深度技术讲解

Andrej, 这篇 Sakana AI 的工作有一个非常优美的核心 thesis：**residual network 的每一步 update，可以重新解读为 VE (Variance Exploding) diffusion 的 probability flow ODE 的 Euler discretization step**。一旦完成这个 reinterpretation，整个 transformer 的 stack 就变成了一个 B-step denoising process，每个 block 在自己的 noise interval 内独立训练，不需要其他 block 的 activations 或 gradients——memory 直接降 B 倍，且训练效率与 end-to-end 几乎相同。

下面我从 intuition、公式、架构、实验、相关工作五个层面拆解。

---

## 1. 关键 Conceptual Jump: 从 Neural ODE 到 Reverse Diffusion ODE

Chen et al. (2018) 的 Neural ODE 已经把 ResNet 的 update rule `z_ℓ = z_{ℓ-1} + f_θ_ℓ(z_{ℓ-1})` 诠释为 ODE `dz/dt = f(z,t)` 的 Euler step。Haber & Ruthotto (2017) 也讨论过 stability。这些工作给了一个"residual 是连续动力系统离散化"的模糊陈述。

DiffusionBlocks 做的是 **更具体** 的事：它指出 ResNet 的 update rule 在数学上对应于 *某个 specific* ODE——VE score-based diffusion 的 probability flow ODE——的 Euler step。这个具体性使得每个 layer 的 update 有了一个 well-defined 的 "noise level σ"，于是 layer-wise training 可以转化为 noise-level-wise training，而后者在 score matching 理论里天然 *是* independent 的。

VE diffusion 的 probability flow ODE (Eq. 1):

$$
\frac{d\mathbf{z}_\sigma}{d\sigma} = -\sigma \, \nabla_{\mathbf{z}} \log p_\sigma(\mathbf{z}_\sigma)
$$

变量含义：
- $\mathbf{z}_\sigma$：noise level $\sigma$ 下的 noisy state，$\sigma \in \mathbb{R}_{\geq 0}$
- $p_\sigma$：在 noise level $\sigma$ 处的 marginal 数据分布
- $\nabla_{\mathbf{z}} \log p_\sigma$：score function（对数密度对 z 的梯度）
- 符号：注意 $\sigma$ 在 forward process 是递增的（加噪），reverse process 中 $d\sigma/dt < 0$，所以这个 ODE 沿 $\sigma$ 递减方向走

Tweedie's formula 给出 score ≈ $(D_\theta(\mathbf{z}_\sigma, \sigma) - \mathbf{z}_\sigma)/\sigma^2$，其中 $D_\theta$ 是 denoiser（预测 clean data）。代入 ODE 并做 Euler discretization on noise levels $\sigma_0 > \sigma_1 > \cdots > \sigma_T$，定义 $\Delta\sigma_\ell := \sigma_{\ell-1} - \sigma_\ell > 0$：

$$
\mathbf{z}_{\sigma_\ell} = \mathbf{z}_{\sigma_{\ell-1}} - \Delta\sigma_\ell \cdot \sigma_{\ell-1} \cdot \nabla_{\mathbf{z}} \log p_{\sigma_{\ell-1}}(\mathbf{z}_{\sigma_{\ell-1}}) \quad \text{(Eq. 3)}
$$

$$
= \mathbf{z}_{\sigma_{\ell-1}} + \frac{\Delta\sigma_\ell}{\sigma_{\ell-1}} \bigl(\mathbf{z}_{\sigma_{\ell-1}} - D_\theta(\mathbf{z}_{\sigma_{\ell-1}}, \sigma_{\ell-1})\bigr) \quad \text{(Eq. 4)}
$$

下标 $\ell-1$ 表示"对应上一层 noise level 的量"。这个 update 的形式是：

$$
\mathbf{z}_{\text{new}} = \alpha \cdot \mathbf{z}_{\text{old}} + \beta \cdot D_\theta(\mathbf{z}_{\text{old}})
$$

其中 $\alpha = 1 + \Delta\sigma_\ell/\sigma_{\ell-1}$, $\beta = -\Delta\sigma_\ell/\sigma_{\ell-1}$。

对比 Transformer residual update: $\mathbf{z}_\ell = \mathbf{z}_{\ell-1} + f_{\theta_\ell}(\mathbf{z}_{\ell-1})$，结构上有 skip connection（系数 1）+ 一个 transform。DiffusionBlocks 把 transformer block 的输出 $f_{\theta_\ell}(\cdot)$ 重新解释为 denoiser $D_{\theta_\ell}$，然后用 $\alpha, \beta$ 这两个 scalar 系数把它的混入方式改成 Euler reverse step 的形式。**网络结构基本不变，只是 attention/MLP 的输出乘了一个 $\beta$，skip 改成 $\alpha$ 倍**。这是一个 minimal modification。

---

## 2. 三步 Conversion Recipe (Section 3.1, Figure 2)

给定 $L$-layer 网络 $\mathcal{F} = \{f_{\theta_\ell} \mid \ell \in [L]\}$，每个 block 是 set-to-set map $\mathbb{R}^d \to \mathbb{R}^d$。

**Step 1: Block partitioning** — 把 $L$ 层分成 $B$ 个 blocks: $\mathcal{F} = \bigsqcup_{b=1}^B \mathcal{F}_b$，第 $b$ 个 block 包含层 $\{\ell_{b-1}+1, \ldots, \ell_b\}$，组合算子 $\bar{f}_{\theta_b} := f_{\theta_{\ell_b}} \circ \cdots \circ f_{\theta_{\ell_{b-1}+1}}$。

**Step 2: Noise range assignment** — 取 log-normal $p_{\text{noise}}$: $\log\sigma \sim \mathcal{N}(P_{\text{mean}}, P_{\text{std}}^2)$, Karras 默认 $P_{\text{mean}}=-1.2, P_{\text{std}}=1.2$。把 $[\sigma_{\min}, \sigma_{\max}] = [0.002, 80]$ 分成 $B$ 个 intervals $\{[\sigma_b, \sigma_{b-1}]\}_{b=1}^B$，用 equi-probability partitioning (下节)。

**Step 3: Noise conditioning + input extension** — 每个 block 的输入从 $\mathbf{x}$ 变成 $\tilde{\mathbf{x}} = (\mathbf{x}, \mathbf{z}_\sigma)$，其中 $\mathbf{z}_\sigma = \mathbf{y} + \sigma\boldsymbol{\epsilon}$。block 还要接收 $\sigma$ 的 conditioning（通过 AdaLN、FiLM 或类似机制）。最终 update：

$$
\mathbf{z}_b = \mathbf{z}_{b-1} + \frac{\Delta\sigma_b}{\sigma_{b-1}} \bigl(\mathbf{z}_{b-1} - [\bar{f}_{\theta_b \mid \sigma_{b-1}}(\mathbf{x}, \mathbf{z}_{b-1})]_{\mathbf{z}}\bigr) \quad \text{(Eq. 5)}
$$

其中 $[\bar{f}(\cdot)]_{\mathbf{z}}$ 是 block 输出中对应 noisy state $\mathbf{z}$ 那部分 token 的值（因为现在输入是 $(\mathbf{x}, \mathbf{z})$ 的拼接）。

不同任务里 $\mathbf{x}, \mathbf{y}$ 的具体含义不同（见 Appendix A 的 table）：

| Task | $\mathbf{x}$ (conditioning) | $\mathbf{y}$ (clean target) |
|---|---|---|
| Image classification | input image patches | class label embedding |
| Image generation (DiT) | noisy image latents (+ class label) | clean image latents |
| AR text generation | previous clean tokens | next token embedding |
| Masked diffusion | sequence with masks | unmasked sequence |
| Recurrent-depth | input prompt | clean continuation embedding |

---

## 3. Block-Independent Training: 为何数学上 OK

每个 block 的 objective (Eq. 6)：

$$
\mathcal{L}_b(\theta_b) := \mathbb{E}_{(\mathbf{x},\mathbf{y}) \sim p_{\text{data}},\, \sigma \sim p_{\text{noise}}^{(b)},\, \boldsymbol{\epsilon} \sim \mathcal{N}(0,I)} \Bigl[ w(\sigma) \cdot \text{Loss}\bigl(\bar{f}_{\theta_b \mid \sigma}(\mathbf{x}, \mathbf{y} + \sigma\boldsymbol{\epsilon}), \mathbf{y}\bigr) \Bigr]
$$

变量：
- $p_{\text{noise}}^{(b)}$：把 $p_{\text{noise}}$ 限制在 block $b$ 的 interval $[\sigma_b, \sigma_{b-1}]$ 并 renormalize
- $w(\sigma)$：EDM weighting $w(\sigma) = (\sigma^2 + \sigma_{\text{data}}^2) / (\sigma \cdot \sigma_{\text{data}})^2$，$\sigma_{\text{data}}=0.5$，用于 counteract $p_{\text{noise}}$ 的 sampling bias，使得所有 $\sigma$ 上 gradient magnitude 平衡
- $\text{Loss}$：L2 loss (公式 2, 连续 diffusion) 或 cross-entropy (离散 token tasks)

**为什么独立训练可行？** Denoising score matching 的一个 deep property: 不同 noise level $\sigma$ 上的 score $\nabla_{\mathbf{z}} \log p_\sigma(\mathbf{z}_\sigma)$ 在训练时 *互相独立*。Denoising 在 $\sigma=\sigma_1$ 处学得好，不需要知道 $\sigma=\sigma_2$ 处的 denoiser 长什么样。这是 diffusion 模型架构本身就允许 per-noise-level 专家化（e.g., eDiff-I 用 ensemble of expert denoisers）的根源。DiffusionBlocks 直接利用这点：每个 block 是一个 noise-level 专家，独立训练，但 assembled together 又 faithful approximates 全 reverse process。

这跟 Forward-Forward (Hinton 2022)、greedy layer-wise training (Bengio 2006)、Local loss (Nøkland & Eidnes 2019)、Block-SSL (Siddiqui 2024) 的 fundamental 区别：那些方法的 local objective 与 global objective 之间 *没有 principled 的联系*，每个 block 只知道自己 local 的 ad-hoc 目标，无法保证 collectively 实现 end-to-end 的功能。Table 1 显示 FF 在 ViT 上只到 7.85% (基本随机), 是典型例子。

---

## 4. Equi-Probability Partitioning (Section 3.3, Figure 4)

朴素 uniform partitioning: $\sigma_b = \sigma_{\min} + b(\sigma_{\max}-\sigma_{\min})/B$。问题：denoising difficulty 不是 uniform in $\sigma$，中间 noise levels 是 image structure emerging 的地方，最难。Karras et al. (2022) 把 $p_{\text{noise}}$ 集中在中间 noise 也正是为此。如果 block 也按 uniform 划分，中间 block 会"超载"，边缘 block "闲置"。

Equi-probability partitioning: 让每个 block 处理 *相同概率质量* 1/B：

$$
\int_{\sigma_b}^{\sigma_{b-1}} p_{\text{noise}}(\sigma)\, d\sigma = \frac{1}{B}
$$

实现上：

$$
\sigma_b = \exp\bigl(P_{\text{mean}} + P_{\text{std}} \cdot \Phi^{-1}(q_b)\bigr)
$$

其中：
- $\Phi^{-1}$: inverse standard normal CDF
- $q_b = q_{\min} + \frac{b}{B}(q_{\max} - q_{\min})$
- $q_{\min/\max} = \Phi\bigl(\frac{\log\sigma_{\min/\max} - P_{\text{mean}}}{P_{\text{std}}\bigr)}$

直观效果（Figure 4）：中间 block 的 interval 在 log-σ 空间很窄（denoising 难、probability mass 密集），边缘 block 的 interval 很宽（容易但 mass 稀疏）。每 block "工作量" 相同。

Ablation (Table 7, CIFAR-10, 12 层分 3 blocks):

| Partitioning | Layer Dist | FID ↓ |
|---|---|---|
| Uniform | [4,4,4] | 43.53 |
| Uniform | [2,4,6] | 42.37 |
| **Equi-Probability** | **[4,4,4]** | **38.03** |
| Equi-Probability | [2,4,6] | 40.40 |

Equi-probability 在所有 layer distribution 上都显著好。值得注意的是 equi-probability + uniform layer distribution [4,4,4] 是最优的——这意味着 practitioner 只需 *等分 layer 数*，noise 划分自动平衡 difficulty。这非常 user-friendly。

---

## 5. Masked Diffusion 的特殊适配 (Appendix D)

对于离散 token 的 masked diffusion model (MD4, Shi et al. 2024)，连续 noise level 概念不直接适用，但可以 partition masking schedule $\alpha(t): [0,1] \to [1,0]$，$\alpha(t)$ 是 $t$ 时刻保持 unmasked 的概率。

NELBO 目标 (Eq. 8):

$$
\mathcal{L}(\theta) = \mathbb{E}_{\mathbf{x}_0} \int_0^1 \frac{-\alpha'(t)}{1-\alpha(t)} \mathbb{E}_{\mathbf{x}_t \sim q(\mathbf{x}_t \mid \mathbf{x}_0)} \Bigl[\sum_{i: x_{ti} = [\text{MASK}]} \text{CE}\bigl(f_\theta(\mathbf{x}_t, t)_i, x_{0i}\bigr)\Bigr] dt
$$

变量：
- $\alpha'(t) := d\alpha/dt < 0$ (随时间更多 token 被 mask)
- $q(\mathbf{x}_t \mid \mathbf{x}_0)$: forward masking kernel (Eq. 7)
- $f_\theta(\mathbf{x}_t, t)_i$: 模型对第 $i$ 个 mask 位置的预测

**关键 observation**: 在 $t$ 处，被 mask 的 token 数期望是 $n(1-\alpha(t))$，所以单位 $t$ 的 contribution density:

$$
\frac{-\alpha'(t)}{1-\alpha(t)} \cdot (1-\alpha(t)) = -\alpha'(t)
$$

那么任意 interval $[t_a, t_b]$ 的 total contribution:

$$
\int_{t_a}^{t_b} -\alpha'(t)\, dt = \alpha(t_a) - \alpha(t_b)
$$

这意味着 training mass 在 $\alpha$ 上是 uniform 分布的，*不在* $t$ 上 uniform。所以 partition 边界应该取 $\alpha$ 的等距：$\alpha_b = 1 - b/B$，对应 $t_b = \alpha^{-1}(1 - b/B)$。对 linear schedule $\alpha(t) = 1-t$，恰好 $t_b = b/B$，但 general schedule 不一定。

每个 block 在自己的 $\alpha$ interval 上训练：

$$
\mathcal{L}_b(\theta_b) = \mathbb{E}_{\mathbf{x}_0} \int_{t_{b-1}}^{t_b} \frac{-\alpha'(t)}{1-\alpha(t)} \mathbb{E}_{\mathbf{x}_t} \Bigl[\sum_{i: x_{ti}=[\text{MASK}]} \text{CE}\bigl(D_{\theta_b}(\mathbf{x}_t, t)_i, x_{0i}\bigr)\Bigr] dt \quad \text{(Eq. 13)}
$$

Global loss 加性分解 $\mathcal{L} = \sum_b \mathcal{L}_b$。这与连续 diffusion 的 equi-probability partitioning *完全 analogous*——都是 partition "工作量" 而非 "interval 长度"。

---

## 6. Recurrent-Depth Models 的 special case (Section 5.5)

Huginn (Geiping et al. 2025) 是 recurrent-depth 模型：同一个 4-layer core 网络 repeated ~32 次，用 truncated BPTT (8 steps) 训练。每次 iteration 是 $\mathbf{z}_k = \mathbf{z}_{k-1} + f_\theta(\mathbf{z}_{k-1})$ 形式——直接是 diffusion update 的形式。

DiffusionBlocks 在这里的用法不同：把 *整个 recurrence* 解释为一个 diffusion process，把网络当作 single denoiser $D_\theta(\mathbf{z}_\sigma, \sigma)$。训练时 sample $\sigma \sim p_\sigma$，做 *single* forward pass 把 noisy input 映到 clean target——*完全不需要 BPTT*。Inference 时仍然用 32 次 iteration（这是 Huginn 的 design），但训练时 computational cost 大约降 10×（论文估算）。

Table 5 (LM1B)：

| Method | MAUVE ↑ | PPL (Llama-2) ↓ | PPL (GPT2-XL) ↓ |
|---|---|---|---|
| Huginn | 0.49 | 17.04 | 46.73 |
| + DiffusionBlocks | **0.70** | **16.08** | **42.43** |

Performance 更好 + training cost 大幅降低。这显示 recurrent refinement 与 diffusion denoising 在 *mathematical structure* 上是同一件事，只是 training algorithm 不同。

---

## 7. 实验数据汇总与分析

### Image classification (CIFAR-100, Table 1)

12-layer ViT, B=3:

| Method | Accuracy ↑ |
|---|---|
| ViT baseline | 60.25 |
| + Forward-Forward | 7.85 |
| + DiffusionBlocks | **59.30** |

DiffusionBlocks 几乎保留 accuracy (loss ~1%)，同时只训 4 层/block。Forward-Forward 完全失败——contrastive local objective 在 ViT 上不 work，这本身就是 FF 的 a well-known limitation.

### Image generation (Table 2)

DiT-S/2 on CIFAR-10, DiT-L/2 on ImageNet-256, B=3:

| Dataset | Method | FID (train/test) ↓ |
|---|---|---|
| CIFAR-10 | DiT | 32.84 / 39.83 |
| | + DiffusionBlocks | **30.59 / 37.20** |
| ImageNet-256 | DiT | 9.01 / 12.09 |
| | + DiffusionBlocks | 9.00 / **10.63** |

DiffusionBlocks 在 test split 上 *比 end-to-end 更好*。Inference 时每个 denoising step 只跑 1 个 block (4 或 8 层 vs 24 层)，50 step diffusion 是 50× speedup on inference compute。

### Masked diffusion (Table 3, text8)

12-layer DiT-based MDM, B=3:

| Method | BPC ↓ |
|---|---|
| MD4 baseline | 1.56 |
| + DiffusionBlocks | **1.45** |

### Autoregressive (Table 4, LM1B/OWT)

12-layer Llama-2 style, B=4:

| Dataset | Method | MAUVE ↑ | PPL (Llama-2) ↓ | PPL (GPT2-XL) ↓ |
|---|---|---|---|---|
| LM1B | AR | 0.50 | 14.58 | 38.87 |
| | + DiffusionBlocks | **0.71** | **12.32** | **30.99** |
| OWT | AR | **0.85** | **15.05** | **25.24** |
| | + DiffusionBlocks | 0.82 | 14.99 | 26.33 |

LM1B 全面更好，OWT 基本持平。这相当 stunning：AR 模型本来不是 denoising 架构，DiffusionBlocks 把它强制 reinterpret 成 denoising 也 work。

### Block count trade-off (Table 8, ImageNet-256, 24 层)

| B | FID ↓ | L/B | Relative Speed |
|---|---|---|---|
| 1 (end-to-end) | 12.09 | 24 | 1.0× |
| 2 | **9.90** | 12 | 2.0× |
| 3 | 11.11 | 8 | 3.0× |
| 4 | 11.90 | 6 | 4.0× |
| 6 | 14.43 | 4 | 6.0× |

**Moderate block count (B=2,3) 比 end-to-end (B=1) 更好**！这是 surprising result。论文 hypothesize 两个 factor：
1. 每个block直接连到target via denoising objective, gradient signal与 end-to-end 不同（更"短"的 gradient path）
2. 不同 block 在不同 noise range 上的 specialization 类似 implicit curriculum learning (Bengio 2009)

### Wall-time (Table 12, ViT)

| Method | Wall time (sec/iter) |
|---|---|
| ViT end-to-end | 0.0507 |
| DiffusionBlocks per block (4 layers) | 0.0181 |
| DiffusionBlocks aggregated (×3) | 0.0543 |

基本持平，~7% overhead 来自 noise conditioning。这验证了 Section H 的理论：DiffusionBlocks 总 compute 与 standard training 相同 (L/B × B × K = L×K layer evaluations)，但 memory 降 B 倍。

---

## 8. 与 Activation Checkpointing 的对比 (Appendix G)

Karpathy 你肯定会问这个。设 $L$ 层，每层 parameter size $P$, activation size $A$, Adam 需要 $2P$ (momentum + variance)：

| Pattern | Memory |
|---|---|
| Standard | $(4P + A) \cdot L$ |
| Activation checkpointing | $4PL + A$ (只省 activation, 不省 optimizer states) |
| DiffusionBlocks | $(4P + A) \cdot L/B$ |
| Both combined | $\approx 4P \cdot L/B + A \cdot L/B$ |

关键区别：activation checkpointing 只省 *activation* memory，对 parameter/gradient/optimizer states 完全无效。DiffusionBlocks 同时减少 *所有* memory components $B$ 倍。对大模型 (optimizer states 常占大头) 这是决定性优势。两者可以叠加。

Compute cost：standard 3F per iter, checkpointing 4F (recompute activations), DiffusionBlocks 保持这个比例。但 DiffusionBlocks 还允许 *embarrassingly parallel* block training——多个 block 在多个 GPU 上完全无通信地训练。

---

## 9. Limitations & Open Questions

1. **Input/output 同维度要求**：framework 依赖每层是 set-to-set map $\mathbb{R}^d \to \mathbb{R}^d$，所以 U-Net (有 down/up-sampling) 不能直接应用。论文 future work 提到这点。
2. **Block granularity 选择**：实验里把整个 ViT/DiT block 当作 unit，但没有 principled 的选择方法。
3. **为什么 B=2 比 B=1 更好**：论文只 hypothesize (specialization + curriculum)，没有理论解释。这可能是理解 deep network 优化的一个 interesting direction。
4. **Pre-trained model 转换**：论文训 from scratch，预训练 large model fine-tune 成 DiffusionBlocks 没试。
5. **Other samplers**：实验只用 Euler。DPM-Solver++、UniPC 等可以替换 inter-block connection。

---

## 10. 与相关工作的关系

- **Neural ODE (Chen 2018)**：建立了 residual = ODE discretization，DiffusionBlocks 把它 *specialize* 到 VE diffusion 的 reverse ODE。
- **EDM (Karras 2022)**：DiffusionBlocks 直接用 EDM 的 log-normal $p_\sigma$、weighting $w(\sigma)$、preconditioning。Equi-probability partitioning 也是基于 EDM $p_\sigma$ 的 cumulative mass。
- **eDiff-I (Balaji 2023), Remix-DiT (Fang 2024), Switch Diffusion (Park 2024)**：这些都是 expert denoiser ensembles，但 training 是 joint 或 fine-tune from shared params。DiffusionBlocks 完全 independent, no shared params, no joint fine-tune。
- **Block Diffusion (Arriola 2025)**：AR + diffusion interpolation。DiffusionBlocks 借用其 sequence concatenation trick 维持 causal consistency。
- **Forward-Forward (Hinton 2022)**：block-wise 用 contrastive objective，无法 extend 到 generation。DiffusionBlocks 的 denoising score matching 是 generative objective，自然涵盖 generation。
- **NoProp (Li et al. 2025)**：concurrent work，也是 backprop-free diffusion training。但 NoProp 用 custom CNN arch，只在 classification 上 demo，没 systematize。DiffusionBlocks 提供 conversion recipe 适用于现代 transformers。Table 6 显示 DiffusionBlocks (46.88%) 在 CIFAR-100 上超 NoProp 所有 variants (最高 46.06%)，且是唯一同时 continuous-time + block-wise 的方法。
- **Huginn (Geiping 2025)**：recurrent-depth model。DiffusionBlocks 把它的 K-iteration training 转成 single-pass，eliminate BPTT。
- **Block-SSL (Siddiqui 2024), Local loss (Nøkland 2019), Greedy layer-wise (Bengio 2006)**：之前的 block-wise training 方法，ad-hoc local objectives，no theoretical guarantee of global coherence。DiffusionBlocks 提供 principled replacement。

---

## Reference Web Links

- Paper (arXiv): https://arxiv.org/abs/2507.06184 (DiffusionBlocks)
- Sakana AI project page: https://sakana.ai/diffusionblocks
- Code: https://github.com/SakanaAI/DiffusionBlocks
- EDM framework (Karras et al. 2022): https://arxiv.org/abs/2206.00364
- Neural ODE (Chen et al. 2018): https://arxiv.org/abs/1806.07366
- Stable architectures (Haber & Ruthotto 2017): https://iopscience.iop.org/article/10.1088/0266-5611/34/1/014004
- Score-based diffusion SDE (Song et al. 2021): https://arxiv.org/abs/2011.13456
- DiT (Peebles & Xie 2023): https://arxiv.org/abs/2212.09748
- Forward-Forward (Hinton 2022): https://arxiv.org/abs/2212.13345
- NoProp (Li et al. 2025): https://arxiv.org/abs/2503.24322
- Huginn / Recurrent-depth (Geiping et al. 2025): https://arxiv.org/abs/2502.05171
- MD4 (Shi et al. 2024): https://arxiv.org/abs/2412.18578
- Masked diffusion LM (Sahoo et al. 2024): https://arxiv.org/abs/2406.07524
- SEDD (Lou et al. 2024): https://arxiv.org/abs/2310.16834
- Block Diffusion (Arriola et al. 2025): https://arxiv.org/abs/2504.01055
- eDiff-I (Balaji et al. 2023): https://arxiv.org/abs/2211.01324
- Curriculum learning (Bengio et al. 2009): https://dl.acm.org/doi/10.1145/1553374.1553380
- Universal Transformers (Dehghani 2019): https://arxiv.org/abs/1807.03819
- Looped Transformers (Fan et al. 2025): https://arxiv.org/abs/2402.05171

---

## Intuition 总结 (for Karpathy)

1. **Residual networks are *literally* (discretized) reverse-diffusion samplers**——不只是"连续动力系统的离散化"这种 fuzzy 的 Neural ODE interpretation。这个 specific 的 interpretation 给了 layer 一个 well-defined noise level $\sigma$，于是 layer-wise training $\Leftrightarrow$ noise-level-wise training，而后者在 score matching 里 *数学上是 independent 的*。这是 framework 能 work 的根本 reason。

2. **Block 之间没有 gradient/activation flow**：训练时 random sample 一个 block $b$，sample $\sigma \sim p_{\text{noise}}^{(b)}$，给 $\mathbf{y}$ 加噪，forward 只跑这 $L/B$ 层，backward 也只跑这 $L/B$ 层。其他 blocks 完全不参与。Memory 严格降 B 倍。

3. **Equi-probability partitioning 是核心 trick**：让每个 block 处理相同的 probability mass，相当于让每个 block 的 "denoising 工作量" 相同，避免中间 block 超载、边缘 block 闲置。Table 7 显示这比 uniform partitioning FID 好近 5 分。

4. **Moderate block count 反超 end-to-end (B=2 比 B=1 更好)**：可能是 specialization 类似 curriculum learning 的 regularization effect，也可能是 denoising objective 给 block 一个 *direct* gradient path 到 target，与 end-to-end 长链 backprop 的 gradient signal 不同。这需要进一步理论分析。

5. **Recurrence 与 Diffusion 是同一件事**：Huginn 的 K 次 iteration 与 K-step diffusion denoising 是 mathematical 同构的。这给出一个把 BPTT-based recurrent training 转成 single-pass diffusion training 的 recipe，10× training cost reduction 且 performance 更好。

6. **适用于 modern transformer 系**：ViT, DiT, Llama-style AR, masked diffusion LM, recurrent-depth——都能用同一个 3-step recipe 转换。这是与 NoProp 等 concurrent work 的关键 differentiator。

我觉得最值得深挖的 open question 是为什么 B=2 会比 B=1 好——如果能 rigorous 证明这是 denoising regularization 的某种 explicit form，可能会给 deep learning optimization 一个新的 angle。
