---
source_pdf: Latent Forcing Reordering the Diffusion Trajectory for Pixel-Space Image
  Generation.pdf
paper_sha256: fa28a00bdbc067db4057c1736da0b85a55bf2be32f39bc60636aadb7d1894451
processed_at: '2026-08-05T12:08:40-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好嘞，用大白话重新捋一遍。

## 一句话总结

**Latent diffusion 的核心优势其实不是"压缩到低维"，而是"先生成 latent 再生成 pixel"这个顺序。只要把这个顺序做到位，在 pixel space 也能获得 latent diffusion 的效率，而且不用 decoder、不丢信息、端到端。**

---

## 现状的两难

目前做 image generation 有两条路：

**Latent diffusion**（Stable Diffusion 那套）：先把 256×256 图像用 VAE 压到 32×32 的 latent，然后在 latent space 做 diffusion，最后用 decoder 解回 pixel。好处是维度低、好训、收敛快。坏处是 encoder 丢信息（人脸、文字这种高频细节经常糊掉），还要单独训练 decoder，整个 pipeline 是分段的，不是端到端。

**Pixel diffusion**：直接在原图上做 diffusion。好处是端到端、无损、不用 decoder。坏处是维度高（256×256×3），训练慢，而且天然是"先低频后高频"的生成顺序（Sander Dieleman 那篇 spectral autoregression 讲过：因为 noise 是白的、信号是低频强的）。

业界主流都押 latent diffusion，觉得压缩是关键。

---

## 关键 Insight

这篇 paper 说：等等，你们搞错了重点。

Latent diffusion 真正的 benefit 不是"把 256×256 压成 32×32"这个降维本身。真正的 benefit 是：**latent diffusion 隐式地让生成按"先粗后细"的顺序发生** —— diffusion model 先生成 coarse latent structure，然后 decoder 再 render 出 pixel 的高频细节。这个 ordering 才是收敛快的原因。

如果你能在 pixel space 里强行 enforce 一个类似的 ordering —— 先生成 DINOv2 这种 self-supervised semantic latent（代表 high-level structure），再生成 pixel（代表细节），那就能同时拿到 latent diffusion 的效率和 pixel diffusion 的端到端无损。

DINOv2 features 在这里相当于一个 **scratchpad / 草稿纸** —— model 先在草稿纸上画好大致结构，然后照着草稿纸填 pixel。最后草稿纸扔掉，只留 pixel 输出。

---

## 怎么实现

很简单：一个 transformer，两套 time variable。

- 把图像 patchify 成 256 个 patch token
- 用 DINOv2 提取同样空间对齐的 256 个 latent token
- 这两组 token 直接相加（element-wise add），token 数不变
- transformer 有两个 time 输入：$t_{\text{latent}}$ 和 $t_{\text{pixel}}$，各自控制 latent 和 pixel 的 noise 水平
- 训练时按特定 schedule 让 latent 先 denoise，pixel 后 denoise
- 推理时：先用 25 步把 latent 从 noise denoise 到 clean，再用 25 步把 pixel 从 noise denoise 到 clean（此时 latent 已经 clean，作为强 condition）
- 输出时只用 pixel，latent 扔掉

架构改动极小：参数只增加 0.5%（多一个 time embedding MLP），FLOPs 几乎不变。

---

## 为什么 ordering 这么重要

核心实验在 Figure 4 和 Table 1。

Figure 4 做了一个特别直观的对比：同样 PSNR（同样 MSE 意义下的"信息量"）下，看 DINOv2 features 和 pixel features 各自保留了什么。

- DINOv2 在低 PSNR 时还能保留住物体的大致轮廓和位置 —— 因为它本身训练目标就是学 semantic structure
- Pixel 在低 PSNR 时几乎是糊的 noise —— 因为像素层面没有 prior，必须 noise 一点点去掉才能看出结构

所以：**先从 noise 里 denoise 出 DINOv2，比先 denoise 出 pixel，要容易得多**。因为 DINOv2 这个 representation 本身就是"高频细节被压掉、只留 structure"的压缩，denoise 它的 search space 小。

然后有了 clean 的 DINOv2 之后，pixel 生成变成"根据已知 structure 填细节"的条件生成问题，比无条件从 noise 生成 pixel 容易几个数量级。

Table 1 量化了这个差距：latent 先 vs pixel 先，FID 差 **3 倍**（DINOv2 上 18.65 vs 55.35）。

---

## Cascaded 训练的小 trick

实际部署用 cascaded schedule（latent 先完全 denoise，再 denoise pixel）。但直接这么做有两个坑：

**坑 1**：传统 diffusion 在 high noise 时基本是 unconditional（输出 class 平均脸）。但 Latent Forcing 在 $t_{\text{pixel}}=0$ 时 latent 已经 clean，model 被 strongly conditioned —— 这个 transition 很难学。解决：10% 的时间从均匀分布采样 early pixel timestep，给 model 练习这个 hard transition。

**坑 2**：Cascaded error accumulation —— latent 生成有误差会传到 pixel。解决：训练时给 latent 加一点点 noise（最多 25%），像 data augmentation，防止 model 过拟合到 latent 的高频细节。inference 时不加 noise。

这个 augmentation 效果很明显：不加 noise 训练 200 epoch 反而比 80 epoch 差（overfit，16.47 vs 13.97）；加 25% noise 后 200 epoch 到 10.93，训练越久越好。

---

## 跟 REPA 的对比 —— 真正的核心论证

REPA 是去年 ICLR 的工作，做法是：训练 diffusion model 时，加一个 alignment loss 让中间层 hidden representation 去匹配 DINOv2 features。效果很好，加速训练。

但 REPA 有个问题：后期训练会失效（Wang et al., 2025 报告过）。

Latent Forcing 的解读：**REPA 的收益其实部分来自 ordering**，不只是 distillation。因为 REPA 让 hidden state 逼近 DINOv2，相当于在 network 内部"隐式地先生成了 DINOv2-like representation 再生成 pixel" —— 这是一个隐式的 ordering。但因为是 soft constraint，ordering 不严格，后期训练会退化。

Latent Forcing 把 ordering 显式化、硬性化 —— 直接分两个 modality、两个 time variable、硬性 schedule。这样 scaling 更稳。

关键证据是 **unconditional generation**（Table 10）：这时没有 class condition，ordering 纯粹由 tokenizer 提供。

- JiT：FID 53.26
- JiT + REPA：35.04  
- **LF-DiT (DINOv2)：20.44**

LF-DiT 比 REPA 好 1.8 倍。更厉害的是，用 Data2Vec2（一个比 DINOv2 弱很多、只在 ImageNet 上训了 150 epoch 的模型）做 Latent Forcing，也比 DINOv2 distillation 好。这说明 **order 本身才是 driver，不是 representation 质量本身**。

---

## 结果

ImageNet 256×256 上：

- 之前 pixel diffusion SOTA 是 JiT-L：FID 16.21 (unguided) / 2.79 (guided)
- Latent Forcing：FID **7.2** (unguided) / **2.48** (guided)
- 接近 latent diffusion SOTA（LightningDiT 2.17/1.35，REPA-E 1.69/1.12）

而且注意：Latent Forcing 的 input space 是 **6 floats/pixel**（3 个 RGB + 3 个 DINOv2 channel）—— ImageNet-256 历史上"压缩最少"的生成模型。传统 wisdom 说越压缩越好生成（Yao et al. 的 reconstruction-generation tradeoff），Latent Forcing 直接反驳：无损也能生成得好，关键是 ordering。

---

## 为什么我觉得这个工作有意义

它不是又一个把 SOTA 刷高 0.1 的工作。它是一个 **conceptual reframe**：

1. **Tokenizer 的作用重新理解**：tokenizer 不是为了"压缩维度"或"去掉 perceptually irrelevant 信息"，而是为了"提供 ordering 信号"。PSNR 不是 diffusability 的好 proxy，ordering trajectory 才是。

2. **REPA 的成功原因重新归因**：不是 distillation 本事，是隐式 ordering。

3. **Pixel diffusion vs latent diffusion 的对立消解**：这俩不是对立的，latents 的 benefit 可以用 ordering 的视角搬到 pixel space。最终可能 pixel diffusion 因为端到端、无损、不用 decoder，会 outscale latent diffusion。

4. **类似 LLM 里的 chain-of-thought**：latent 就是 model 的"思考过程"，pixel 是"答案"。先思考再答题比直接答题好。这个类比很深，论文也引了 Hao et al. 2024 的工作。

---

## 可能的局限

- 依赖外部 DINOv2，不是完全 from scratch。如果目标 domain 没 good self-supervised model，效果可能打折
- Hyperparameter 多（$p_{\text{latent}}$、两个 logit-normal 的 $\mu$、noise $\beta$、time shift $\alpha$），调参空间大
- 只在 ImageNet class-conditional 上验证，text-to-image 这种更复杂的 conditional generation 还没试
- 推理 50 步（25+25）在 pixel space 跑，可能比 latent diffusion 的 25 步在低维 latent 跑慢

但作为一篇 rethinking 性质的 paper，它给的视角足够有价值 —— 让你下次设计 diffusion model 时，会先想"我的生成 ordering 是什么"而不是"我的 tokenizer 压缩率是多少"。

---

# Latent Forcing: Reordering the Diffusion Trajectory for Pixel-Space Image Generation 深度解析

## Paper 概览

这篇 paper 来自 Stanford (Alan Baade, Eric Ryan Chan, Kyle Sargent, Changan Chen, Ehsan Adeli, Li Fei-Fei) 和 University of Michigan (Justin Johnson), 核心贡献是提出 **Latent Forcing (LF)** —— 一个看似简单但极其有效的想法: 把 latent diffusion 的 efficiency 优势和 pixel diffusion 的 end-to-end 优势结合起来, 通过**重新排列 diffusion 轨迹的生成顺序**实现。

Project page: https://latentforcing.github.io/
arXiv: https://arxiv.org/abs/2505.00546
Code (可能): https://github.com/alanbaade/latent-forcing

## 1. 核心直觉: 生成顺序的重要性

人类创作图像时, 先 sketch 大结构, 再填充细节。Image diffusion 也应该遵循这个顺序: 先 high-level semantics, 再 low-frequency structure, 最后 high-frequency detail。

现有方法的隐式顺序:
- **Latent Diffusion (LDM)**: diffusion model 生成 coarse latent → decoder (GAN/diffusion) 渲染 pixel details。但是 encoder 损失信息, 需要单独训练 decoder, 而且 reconstruction-generation tradeoff 限制了 latent space 设计 (Yao et al., 2025: https://arxiv.org/abs/2410.01866)
- **Pixel Diffusion (JiT)**: end-to-end, 但 high-dimensional, 而且频率域 autoregression (Dieleman, 2024: https://sander.ai/2024/09/02/spectral-autoregression.html) 强制先低频后高频, 灵活性差

**Latent Forcing 的 insight**: LDM 的好处其实不是 "压缩到低维", 而是 "latent 先生成, pixel 后生成" 这个 **ordering**。如果显式地把这个 ordering 应用到 pixel-space diffusion 上, 就能既保留 end-to-end 又获得 latent diffusion 的 convergence benefits。

这个 insight 让我想到 Hopfield networks 和 energy-based models 里的 "scratchpad" 概念, 还有 CoT (chain-of-thought) 在 LLM 里的作用 (Hao et al., 2024: https://arxiv.org/abs/2412.06769) —— latents 就像 scratchpad, 帮助 model 在生成 final answer (pixels) 之前先做 intermediate computation。

## 2. 方法: Flow-Based Diffusion on Multiple Modalities

### 2.1 单 modality Flow 回顾

Flow matching / rectified flow (Liu et al., 2022: https://arxiv.org/abs/2209.03003):
$$z_t = t \mathbf{x} + (1-t) \boldsymbol{\epsilon}$$

变量解释:
- $\mathbf{x} \sim p_{\text{data}}$: clean data distribution
- $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$: standard Gaussian noise
- $t \in [0,1]$: time variable, $t=1$ 为 pure data, $t=0$ 为 pure noise (注意这里和 EDM/Karras 的方向是反的)
- $z_t$: noised latent

训练目标:
$$\mathbb{E}_{\mathbf{x}, \boldsymbol{\epsilon}} \| \mathbf{v}_\theta(z_t, t) - (\mathbf{x} - \boldsymbol{\epsilon}) \|^2$$

其中 $\mathbf{v}_\theta$ 是 velocity prediction network。这里 follow JiT (Li & He, 2025: https://arxiv.org/abs/2511.13720) 用 **x-prediction with v-loss weighting**, 因为高维空间直接预测 $\epsilon$ 或 $\mathbf{v}$ 会有 information capacity bottleneck。

### 2.2 多 modality 扩展

对 $k$ 个 modality (实践中 $k=2$: pixel + latent), 每个有自己的 time variable $t_i$:

$$\mathcal{L} = \sum_{i=1}^{k} \lambda_i \mathbb{E} \| \mathbf{v}_{\theta,i}(\mathbf{z}_{1,t_1}, \ldots, \mathbf{z}_{k,t_k}, t_1, \ldots, t_k) - (\mathbf{x}_i - \boldsymbol{\epsilon}_i) \|^2 \quad (1)$$

变量解释:
- $\mathbf{x}_i \sim p_{\text{data}_i}$: 第 $i$ 个 modality 的 clean data
- $\boldsymbol{\epsilon}_i$: 第 $i$ 个 modality 的 noise
- $t_i \in [0,1]$: 第 $i$ 个 modality 的 time variable (可以是 correlated)
- $\mathbf{z}_{i,t_i} = t_i \mathbf{x}_i + (1-t_i)\boldsymbol{\epsilon}_i$: 第 $i$ 个 modality 的 noised latent
- $\mathbf{v}_{\theta,i}$: model 输出的第 $i$ 个 modality 的 velocity
- $\lambda_i$: loss weight, 用于平衡不同 modality 的 loss magnitude

### 2.3 Inference: Global Time + Per-Modality Schedules

定义 global time $t_{\text{global}} \in [0,1]$, per-modality schedule $t_i = f_i(t_{\text{global}})$, 要求 $f_i$ non-decreasing, $f_i(0)=0$, $f_i(1)=1$。

Euler step 从 $t$ 到 $s$:
$$\mathbf{z}_{i, f_i(s)} = \mathbf{z}_{i, f_i(t)} + (f_i(s) - f_i(t)) \cdot \mathbf{v}_{\theta,i}(\cdot) \quad (2)$$

**生成顺序 (Generation Order)** 用 SNR trajectory 定义:
$$\mathcal{O}(t_{\text{global}}) = \left( \frac{f_i(t_{\text{global}})^2 \mathbb{V}[\mathbf{x}_i]}{(1 - f_i(t_{\text{global}}))^2} \right)_{i=1}^k \quad (3)$$

变量解释:
- $\mathbb{V}[\mathbf{x}_i]$: 第 $i$ 个 modality 的 variance
- SNR = signal variance / noise variance, 衡量信息量
- SNR 越高, 该 modality 在该时刻 "越 clean"

这里 mutual information $I(\mathbf{x}_i; \mathbf{z}_{i,t_i})$ 被 Gaussian channel capacity $\frac{1}{2}\log_2(1 + \text{SNR}_i)$ 上界控制, 所以 SNR 是 information reveal rate 的 proxy。

### 2.4 Scaling ⟺ Time Scheduling (关键 corollary)

由 Hoogeboom et al., 2023 (Simple Diffusion: https://arxiv.org/abs/2301.11093) 启发: scaling data magnitude 等价于 shifting noise schedule。

由 SD3 (Esser et al., 2024: https://arxiv.org/abs/2403.03206) 推导: scaling latent by $\alpha$ 等价于 time shift:
$$f_{\alpha\text{-shift}}(t) = \frac{t\alpha}{1 + (\alpha-1)t} \quad (4)$$

变量解释:
- $\alpha$: scaling factor ($\alpha > 1$ 加快生成, $\alpha < 1$ 减慢)
- 当 $\alpha = 1$, $f(t) = t$ (无 shift)
- 当 $t \to 0$, $f \approx \alpha t$ (early timesteps 被加速)
- 当 $t \to 1$, $f \approx 1$ (最终 clean)

**重要推论**: 任何 multi-modality combination 都隐含了 per-modality time schedule, 即使只用一个 time variable, 因为 scaling 就是 schedule。这解释了为什么 SD3 的 shift scaling 有效 —— 它实际上在 reorder 生成顺序。

### 2.5 Joint Distribution 分解

当 latent 是 pixel 的 deterministic function (如 DINOv2 features), $P(Y|X) = 1$, 则:
$$P(X, Y) = P(Y|X) P(X) = P(X)$$

所以无论 ordering 如何, 都在优化 raw data likelihood。当 schedules 不重叠时:
$$P(X, Y) = P(Y) P(X|Y)$$

这正是 cascaded generation。这从理论上论证了 Latent Forcing 的 lossless 性质 —— 与 LDM 不同, 它直接优化 pixel 分布 likelihood, 不需要 VAE lower bound。

## 3. 架构细节

### 3.1 Tokenization

- Input image $I \in \mathbb{R}^{256 \times 256 \times 3}$
- **Pixel tokens**: patchify 成 256 tokens, patch size 16: $x_{\text{pixel}} \in \mathbb{R}^{16 \times 16 \times 768}$, 其中 $768 = 16 \times 16 \times 3$
- **Latent tokens**: 用 DINOv2-B+Registers (Oquab et al., 2023: https://arxiv.org/abs/2304.07193; Darcet et al., 2023: https://arxiv.org/abs/2309.16588), 输入 resize 到 224×224, patch 14×14, 然后通过 linear projection 得到 $x_{\text{latent}} \in \mathbb{R}^{16 \times 16 \times D}$
- **关键**: latent patches 和 pixel patches spatially align (像 REPA, Yu et al., 2025: https://arxiv.org/abs/2410.06938)
- Pixels 归一化到 $[-1, 1]$, latents rescale 到 match pixel global variance

### 3.2 DiT 改动 (Figure 2)

**改动 1: Token embedding addition**
- Pixel 和 latent 各自经过 128-dim linear bottleneck, 然后 **直接 add 在一起** (token 数不变)
- 这意味着每个 token 同时承载 pixel 和 latent 信息

**改动 2: 两个 time variables**
- 传统 DiT 用 adaLN-Zero (Peebles & Xie, 2023: https://arxiv.org/abs/2212.09748), time embedding + class embedding 通过 2-layer MLP → scale/shift
- Latent Forcing 增加第二个 time embedding MLP, 参数增加 ~0.5%

**改动 3: Output Experts (optional, 但 ablation 显示有用)**
- 取 last 4 transformer layers, split 成两个 M/2-layer experts
- 一个负责 latent output, 一个负责 pixel output
- 参数量不变, FLOPs 不变
- 动机: 单一 linear projection 输出多 modality 可能 strain network capacity

### 3.3 x-prediction with v-loss

由 JiT 启发, 用 x-prediction (预测 clean target) 配合 v-loss weighting:
$$\mathcal{L} = \left\| \frac{(\mathbf{x}_{\text{pred}} - \mathbf{z}_t)}{\max(1-t, t_{\text{clip}})} - \frac{(\mathbf{x} - \mathbf{z}_t)}{\max(1-t, t_{\text{clip}})} \right\|^2 \quad (5)$$

变量解释:
- $\mathbf{x}_{\text{pred}}$: model 直接预测 clean data
- $t_{\text{clip}} = 0.05$ (Multi-Schedule) 或 $1/3$ (统一 scale): 防止除零
- 分母 $\max(1-t, t_{\text{clip}})$ 是 v-loss 权重, 在 $t \to 1$ (clean) 时小, $t \to 0$ (noise) 时大

## 4. Multi-Schedule Model: 探索 Ordering 空间

### 4.1 训练 schedule

- 传统: shifted logit-normal (Karras et al., 2022: https://arxiv.org/abs/2206.00364)
- 问题: 多变量 logit-normal 是 product distribution, cascaded trajectory 会 receive zero training signal
- 解决: sample from **uniform**, 然后 apply time shift (Eq. 4)
- $t_{\text{clip}} = 1/3$ 平衡 low-noise timestep 的 gradient magnitude
- Loss weights $\lambda_i$ 让 pixel 和 latent loss magnitude 相等

### 4.2 结果 (Table 1, Figure 3)

不同 $\alpha$ shift 的 FID-10K (无 guidance):

| Latent Model | $\alpha=1/64$ | $1/16$ | $1/4$ | $1$ (eq. SNR) | $4$ | $16$ | $64$ |
|---|---|---|---|---|---|---|---|
| 64×64 pixels | 44.51 | 44.45 | 44.35 | 44.57 | 44.20 | 42.35 | 42.31 |
| Data2Vec2 | 55.19 | 50.24 | 38.24 | 27.69 | 24.26 | **23.61** | 24.44 |
| DINOv2-B+REG | 55.35 | 50.64 | 37.63 | 24.39 | 18.99 | **18.65** | 18.90 |

$\alpha > 1$ (latent 早生成) 比 $\alpha < 1$ (pixel 早生成) FID 差距巨大 (DINOv2: 18.65 vs 55.35, **~3× 改进**)。

**直觉解释**: Figure 4 显示, 在相同 PSNR (MSE) 下, DINOv2 features 保留了显著更多 spatial structure, 而 pixel features 在低 SNR 时表现出 large-scale structural uncertainty。Model 先生成结构清晰的 latent, 然后用 latent condition pixel 生成, 比 "先模糊地猜 pixel, 再 refine" 容易得多。

### 4.3 Ordering 在早期 timestep 最关键 (Figure 3 右)

线性 schedule 让 DINOv2 先 denoise 到 $t_{\text{DINO}} = 0.15$ 已经 capture 大部分 ordering gain。这意味着 ordering 的作用主要集中在 coarse-to-fine 转换点附近。

### 4.4 跨 modality 交互 (Figure 5)

- Pixel features 在没有 DINOv2 信息时 sharply increase PSNR at early timesteps (但生成的是 "平均脸")
- Pixel features 在几乎所有 noise levels 都 improve DINOv2 predictions (mutual benefit)
- DINOv2 features 在 $t_{\text{pixel}} \geq 0.75$ 时不再 inform pixel generation (此时 pixel 已经基本 clean)

## 5. Single-Schedule Model: 实际 Baseline

### 5.1 Cascaded Schedule (最佳)

- 以概率 $p_{\text{latent}}$ 选择 latent step (否则 pixel step)
- Latent step: $t_{\text{pixel}} = 0$ (full noise), 禁用 pixel loss
- Pixel step: $t_{\text{latent}} = 1.0$ (no noise), 禁用 latent loss
- 分别用 logit-normal schedule 采样 (Zheng et al., 2025 RAE: https://arxiv.org/abs/2503.13458 和 Li & He, 2025)

**Ablation (Table 2)**: $p_{\text{latent}} = 0.4$ 最优 (FID 12.42), 太高 (0.8) 则 16.13

**Logit-normal 参数 (Tables 3, 4)**:
- Latent: $\mu_{\text{LATENT}} = 1.0$, FID 14.11 (vs 1.2 → 13.97, 1.4 → 14.41)
- Pixel: $\mu_{\text{PIXEL}} = 0.4$, FID 16.13 (vs 0.0 → 18.28, 0.8 → 16.38)

### 5.2 改进 Cascaded Generation

**问题 1**: 传统 diffusion 在 high noise 时 weakly conditioned (输出 class average), 但 Latent Forcing 在 $t=0$ 时 strongly conditioned on denoised latent, 使 logit-normal (zero mass at $t_{\text{pixel}} = 0$) 不利。

**解决**: 10% 时间从 $U[0, 0.5]$ 采样 pixel timestep。Table 5 显示这改善 12.42 → 12.98 (如果移除的话)。

**问题 2**: Cascaded error (像 cascaded diffusion, Ho et al., 2022: https://arxiv.org/abs/2106.15738) —— latent 错误会传到 pixel。

**解决**: 训练时 pixel steps 给 latent 加少量 noise: $t_{\text{latent}} \in U[1-\beta, 1]$。

Table 6 的关键发现:
- $\beta = 0\%$, 80 epochs: FID 13.97
- $\beta = 0\%$, 200 epochs: FID **16.47** (overfitting 恶化!)
- $\beta = 25\%$, 80 epochs: 13.48
- $\beta = 25\%$, 200 epochs: **10.93** (训练更长反而更好)

**直觉**: noise 像 augmentation, 防止 model 过拟合到 latent space 的高频 difficult-to-generate details。

Table 7: 训练用 noise, inference 不用 noise 最好 (10.93), 加 noise 反而有害 (5% → 11.07)。这支持了 "noise as augmentation" 解释 —— 测试时不应有 augmentation。

### 5.3 Output Experts (Table 5)

- 有 output experts: 12.42
- 无 output experts: 12.93

零参数代价, 0.5 FID 改进。

### 5.4 不同 schedule 比较 (Table 8)

| Time Schedule | FID Unguided | FID Guided |
|---|---|---|
| **Cascaded** | **12.42** | **6.60** |
| Lin. Offset $o=0.1$ | 20.98 | 10.73 |
| Var. Shift $\alpha=9$ | 13.48 | 8.16 |

Cascaded 最好, 但 variance shift 也不错 —— 这意味着 **不一定要 hard cascaded**, 也可以 soft overlap。

### 5.5 Guidance

- AutoGuidance (Karras et al., 2024: https://arxiv.org/abs/2411.18689) 优于 CFG, 因为 DINOv2 features probe 到 class label, 使 class conditioning redundant
- System-level (Table 11): CFG-Interval (Kynkäanniemi et al., 2024: https://arxiv.org/abs/2404.07759) 用于 DINOv2 timesteps, AutoGuidance 用于 pixel timesteps

## 6. 与 REPA 的关系: Ordering vs Distillation

REPA (Yu et al., 2025: https://arxiv.org/abs/2410.06938) 通过 aligning diffusion model 的 hidden representations 与 DINOv2 features, 极大加速 diffusion training。但 Wang et al., 2025 (https://arxiv.org/abs/2505.16792) 显示 REPA 在 late-stage training 失效。

Latent Forcing 的视角: REPA 的 gain 部分来自 **ordering**, 不只是 distillation。

### 6.1 Conditional Generation (Table 9, 80 epochs)

| Model | FID Unguided | FID Guided |
|---|---|---|
| JiT | 25.18 | 5.64 |
| JiT + REPA | 18.60 | 4.57 |
| **LF-DiT DINOv2** | **9.76** | **4.18** |
| LF-DIT D2V2 | 12.46 | 5.45 |

LF-DiT vs JiT+REPA: unguided 1.9× 改进, guided 略好。

### 6.2 Unconditional Generation (Table 10) —— **关键实验**

Unconditional 时, class conditioning 不在, ordering 纯粹由 tokenizer 决定:

| Model | FID Unguided | FID Guided |
|---|---|---|
| JiT | 53.26 | 44.80 |
| JiT + REPA | 35.04 | 24.40 |
| **LF-DiT DINOv2** | **20.44** | **13.36** |
| LF-DIT D2V2 | 20.99 | 15.56 |

Guided FID 1.8× 改进 vs REPA。这强力论证 **ordering 是核心, distillation 不是主要 driver**。

更惊人的: Data2Vec2-Large (只训练 150 epochs, 比 DINOv2 弱) 的 Latent Forcing 也比 DINOv2 distillation 好。这意味着 **任何 deterministic latent representation, 只要 order 在前, 都能 benefit** —— representation 不一定要是最好的。

### 6.3 个人思考: REPA 失效的另一种解释

Wang et al., 2025 报告 REPA 在 late stage 失效。从 Latent Forcing 视角看: REPA 的 alignment loss 让 hidden representations approximate DINOv2, 这相当于隐式让 model 在内部 "生成" DINOv2 features, 然后 use 它们 —— 但因为 alignment 是 soft constraint, ordering 不严格。Latent Forcing 显式 enforce ordering, 所以 scaling 更好。

类似 LLM 里 CoT vs implicit latent reasoning: latent reasoning (Hao et al., 2024: https://arxiv.org/abs/2412.06769) vs explicit token CoT。显式通常更易训练但 cost 高。

## 7. System-Level Comparison (Table 11)

| Model | Params | Dec Params | Epochs | PSNR↑ | FID(U)↓ | FID(G)↓ |
|---|---|---|---|---|---|---|
| **Latent Diffusion** | | | | | | |
| RAE | 839M | 415M | 800 | 18.09 | 1.51 | 1.13 |
| DiT-XL/2+SD-VAE | 675M | 49M | 1400 | 23.40 | 9.62 | 3.04 |
| LightningDiT | 675M | 41M | 800 | 25.29 | 2.17 | 1.35 |
| REPA-E | 675M | 41M | 1480 | 26.25 | 1.69 | 1.12 |
| MAR + Uniflow | 479M | 300M | 400 | 32.48 | 2.45 | 1.85 |
| **Pixel Diffusion** | | | | | | |
| ADM | 554M | 0 | 400 | ∞ | 10.94 | 3.94 |
| SiD UViT/2 | 2B | 0 | - | ∞ | 2.77 | 2.44 |
| SiD2 UViT/2 | - | 0 | - | ∞ | - | 1.73 |
| **ViT Pixel Diffusion** | | | | | | |
| JiT-L | 459M | 0 | 200 | ∞ | 16.21 | 2.79 |
| **LF-DiT-L** | **465M** | **0** | **200** | **∞** | **7.2** | **2.48** |

LF-DiT-L 是 pixel diffusion SOTA, FID 7.2 unguided / 2.48 guided, 比 JiT 改进 2.25× / 1.12×, 接近 latent diffusion SOTA (REPA-E 1.69/1.12, LightningDiT 2.17/1.35)。

**Compression 视角**: LF-DiT 用 **6 floats/pixel** (3 RGB + 3 DINOv2 channels after normalization) —— 是 ImageNet-256 上**有史以来 least compressed 的 input space**, 却超过所有 pixel diffusion 接近 latent diffusion SOTA。这挑战了 "more compression → better generation" 的传统 wisdom (Yao et al., 2025)。

## 8. 关键 Insights 提炼 (Intuition Building)

### 8.1 为什么 ordering 重要? 

从 information theory: Gaussian channel capacity $C = \frac{1}{2}\log_2(1+\text{SNR})$ bits/dim。Diffusion 是渐进 reveal information。如果两个 modality 都 reveal, model 可以 leverage 先 reveal 的作为 condition, 降低后 reveal 的 entropy。这是 conditional generation 的本质: $H(X|Y) < H(X)$ 当 $Y$ informative。

Cascaded 生成 $P(Y)P(X|Y)$ 在 deterministic latent 时 $I(X;Y)$ 最大化, 所以 $H(X|Y)$ 最小, 生成最容易。如果反序 $P(X)P(Y|X)$, model 还是要先生成 $X$ (难), $Y$ 没帮助。

### 8.2 为什么 pixel diffusion 难?

JiT 表明 high-dim prediction 可行 (图像在 low-dim manifold)。但 frequency-domain autoregression 强制 low-freq 先 high-freq 后, 这可能不是最 informative ordering —— semantic structure 不一定在 lowest frequencies。

Dieleman (2024: https://sander.ai/2024/09/02/spectral-autoregression.html) 的观察是: pixel diffusion 自然 spectral autoregression 因为 noise 是 white (all frequencies), signal 是 colored (low-freq dominant)。但 Latent Forcing 通过外部 latent 提供更好的 "first signal"。

### 8.3 REPA 是 implicit Latent Forcing?

REPA 的 alignment loss $\|\phi(h_\theta) - \text{DINOv2}\|^2$ 让 hidden $h$ approximate DINOv2 features at certain layers。Diffusion 过程中 $h$ 从 random → meaningful, 相当于内部生成 DINOv2。但这是 **soft constraint**, ordering 严格度低。Latent Forcing 显式分离 modality, hard ordering, 所以 late-stage training 不退化。

### 8.4 6 floats/pixel 为何能 work?

传统 wisdom: 压缩去除 "perceptually irrelevant" 信息, 让 model focus on "diffusable" dimensions。Latent Forcing 的 answer: 重要的是 **早期 timestep 的 effective dimensionality**, 不是整体。早期只生成 latent (低维), 后期 pixel (高维但 strongly conditioned), 整个 trajectory 的 effective complexity 低。

类似 LLM 的 KV cache: 总 tokens 多, 但每步 attention 计算量受 query 数控制。

## 9. 实现细节 (Appendix A.1, Table 12)

- ViT-L: depth 24, hidden 1024, 16 heads, patch 16
- Pixel bottleneck 128, latent bottleneck 128
- In-context CLS (JiT style) 在 block 32 之后插入
- Optimizer: Adam, $\beta_1, \beta_2 = 0.9, 0.95$
- Batch 1024, LR 2e-4, constant schedule
- EMA decay 0.9999
- Latent sampler: logit-normal $\mu=-1.2, \sigma=1.0$
- Pixel sampler: logit-normal $\mu=-0.8, \sigma=0.8$
- Class token drop 0.1 (for CFG)
- Heun ODE solver, 50 steps total (25 latent + 25 pixel for cascaded)
- Loss weights: $\lambda_{\text{pixel}} = 1.0$, $\lambda_{\text{DINOv2}} = 0.333$, $\lambda_{\text{D2V2}} = 0.25$
- DINOv2: ViT-B+registers, layer 12 pre-norm, following RAE
- Data2Vec2: Large, 150 epochs ImageNet, layer 12

## 10. 个人延伸思考

### 10.1 与 Diffusion Forcing (Chen et al., 2025a: https://arxiv.org/abs/2411.04528) 对比

Diffusion Forcing 用 multiple time schedules 做 autoregressive video prediction。Latent Forcing 借鉴了 multiple time variables 的思想, 但用在 **non-autoregressive** setting, joint generate 同一 input 的不同 representations。

### 10.2 与 MAR / Masked Generative Models

MAR (Li et al., 2024: https://arxiv.org/abs/2406.11838) 用 autoregressive 生成 tokens (VAE 编码后)。Latent Forcing 也在某种意义上 autoregressive: latent 先 pixel 后, 但用 continuous diffusion 而非 discrete AR, 且 end-to-end。

### 10.3 与 RAE (Zheng et al., 2025: https://arxiv.org/abs/2503.13458) 的关系

RAE 用 DINOv2 features 作为 tokenizer 的辅助 supervision。Latent Forcing 用 DINOv2 作为生成 modality。两者都利用 self-supervised features, 但 RAE 仍需 decoder, LF 不需。

### 10.4 未来方向

1. **更多 modalities**: 加入 text embeddings, depth, segmentation —— 多 modality Latent Forcing。
2. **Hierarchical**: 多层 latent (DINOv2 CLS → DINOv2 patches → pixels), 类似 Matryoshka Diffusion (Gu et al., 2024: https://arxiv.org/abs/2310.15111)。
3. **Text-to-image**: DINOv2 替换为 CLIP text features。可能让 pixel-space T2I 与 Stable Diffusion 竞争。
4. **Video**: latent (semantic frames) → pixel (all frames)。Diffusion Forcing 的视频版本。
5. **Learned latents**: 不依赖外部 DINOv2, 而是 jointly train 一个 self-supervised encoder + Latent Forcing diffusion。可能达到完全 end-to-end。
6. **Diffusability 的重新理解**: PSNR 不是 diffusability 的好 proxy, ordering 才是。这能指导未来 tokenizer 设计 (Yang et al., 2025: https://arxiv.org/abs/2507.15856; Kouzelis et al., 2025: https://arxiv.org/abs/2506.07344)。

### 10.5 与 Scaling Laws 的关系

如果 ordering 是核心 driver, 那 scaling 不只依赖 model size 和 data size, 还依赖 **trajectory 设计**。Latent Forcing 在 fixed compute 下 2× 改进, 这是 algorithmic scaling。

类似 Chinchilla (Hoffmann et al., 2022) 重新定义 data/params ratio。Latent Forcing 重新定义 "compute per bit of information generated in order"。

### 10.6 潜在 issue

1. **External dependency**: 依赖 DINOv2 等 pretrained model, 不是完全 from scratch。如果 DINOv2 在某 domain 不好 (e.g., 医学影像), Latent Forcing 也会受限。
2. **Inference latency**: 50 steps (25+25) 比 latent diffusion (典型 25-50 steps on low-dim latent + decoder) 可能慢, 因为 pixel 维度高。
3. **Generalization**: ImageNet 上验证, 但 ImageNet 类别 bias 重。Text-to-image 上的效果待验证。
4. **Multiple time variables 的 hyperparameter**: $p_{\text{latent}}$, logit-normal $\mu$, $t_{\text{clip}}$, $\beta$ 等都需要调, search space 大。

## 11. 总结

Latent Forcing 是一个 elegant 的 idea: 把 latent diffusion 的 efficiency 来源从 "compression" 重新归因于 "ordering", 然后用 multi-time-variable diffusion 显式实现 ordering, 在 pixel space 上达到 latent diffusion 的 convergence speed 同时保持 end-to-end 和 lossless。

核心公式 (Eq. 1, 2, 3, 4) 简单, 架构改动 minimal (+0.5% params), 但 FID 改进显著 (pixel diffusion SOTA, 接近 latent SOTA)。更重要的是, 它提供了一个新视角来理解 diffusion model 的 design choices —— 从 tokenizer compression 到 REPA distillation, 都可以重新用 ordering 来分析。

Key references:
- Paper: https://arxiv.org/abs/2505.00546
- JiT: https://arxiv.org/abs/2511.13720
- REPA: https://arxiv.org/abs/2410.06938
- RAE: https://arxiv.org/abs/2503.13458
- LightningDiT: https://arxiv.org/abs/2502.04847
- DINOv2: https://arxiv.org/abs/2304.07193
- Diffusion Forcing: https://arxiv.org/abs/2411.04528
- EDM2: https://arxiv.org/abs/2312.02657
- Dieleman spectral autoregression: https://sander.ai/2024/09/02/spectral-autoregression.html
- Reconstruction vs Generation tradeoff: https://arxiv.org/abs/2410.01866
- REPA Works Until It Doesn't: https://arxiv.org/abs/2505.16792
- Latent denoising tokenizers: https://arxiv.org/abs/2507.15856
- EQ-VAE: https://openreview.net/forum?id=UWhW5YYLo6
- AutoGuidance: https://arxiv.org/abs/2411.18689
- CFG-Interval: https://arxiv.org/abs/2404.07759
- Simple Diffusion: https://arxiv.org/abs/2301.11093
- SD3: https://arxiv.org/abs/2403.03206
- DiT: https://arxiv.org/abs/2212.09748
- SiD2: https://arxiv.org/abs/2410.19324
- Diffuse and Disperse: https://arxiv.org/abs/2506.09027
