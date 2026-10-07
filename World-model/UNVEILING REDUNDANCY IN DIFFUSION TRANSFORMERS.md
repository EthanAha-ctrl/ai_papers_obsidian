---
source_pdf: UNVEILING REDUNDANCY IN DIFFUSION TRANSFORMERS.pdf
paper_sha256: 900f27182a4ad2ed22f636ded78dc94d1d6549e84ab85ed8fc1ff43b2111351b
processed_at: '2026-08-12T20:29:38-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

## 一句话总结

**DiT 模型推理时，相邻 step 算出来的中间结果（attention 的 K、V、output）长得很像，可以 cache 复用省算力。但这篇 paper 发现：每个模型的"像不像"pattern 完全不一样，之前大家各看各的模型，得出互相打架的结论。**

---

## 问题从哪来

Diffusion model 生成图片/视频，本质就是反复跑同一个神经网络 $T$ 次（比如 28 步、50 步）。每次输入 $x_t$，模型预测 noise $\epsilon_t$，然后 update 出 $x_{t-1}$。

$$x_{t-1} = \mathrm{Update}(x_t, t, \epsilon_t), \quad \epsilon_t = \epsilon_\theta(x_t, t, c)$$

这里 $x_t$ 是第 $t$ 步的 noisy latent，$t$ 是 timestep，$c$ 是 text condition，$\epsilon_\theta$ 就是 DiT backbone。$\mathrm{Update}$ 是 sampler（DDIM / Flow Match Euler）做的事。

问题：$T$ 次 forward pass 太慢了。生成一张高清图要分钟级，video 更久。

**Key observation**：相邻 step 的 $x_t$ 差异很小（特别是后期），所以模型内部的 activations 也差不多。如果能 cache 上一步的中间结果，这步直接复用，就省一次 forward。

之前有三组人做了研究：
- **L2C**：发现后期 steps 的 activation 差异比前期大
- **TGATE**：发现 cross-attention 在 step 5-10 后差异趋于零
- **PAB**：发现 U 形——中间差异小、两端大

三个结论互相矛盾。有人看到"后期冗余高"，有人看到"后期冗余低"。到底谁对？

---

## 这篇 paper 做了什么

作者说：你们都只看了自己那一个模型，以偏概全了。我把 7 个主流 DiT 模型全跑一遍，看看到底是怎么回事。

7 个模型：

| Model | 参数量 | Attention Layers | Diffusion Steps |
|-------|--------|------------------|-----------------|
| Flux.1-dev | 12B | 57 | 28 |
| Pixart-Alpha | 0.6B | 28 | 20 |
| Stable-Diffusion-3 | 2B | 24 | 28 |
| CogVideoX-5B | 5B | 30 | 50 |
| Open-Sora | 1.1B | 56 | 30 |
| Latte-1 | 0.7B | 56 | 50 |
| Mochi-1-preview | 10B | 48 | 64 |

测量的东西：每个 attention layer 的 $K$、$V$、$A$（output），在相邻 step 之间的 L1 distance：

$$\Delta X_{\mathcal{M}(p)}^{i,j} = \| X_{\mathcal{M}(p)}^{i,j} - X_{\mathcal{M}(p)}^{i+1,j} \|_1$$

这里 $X$ 是 $K$ 或 $V$ 或 $A$，$i$ 是 diffusion step，$j$ 是 layer index。L1 越小 = 越冗余 = 越适合 cache。

---

## 核心发现：每个模型都有自己的"指纹"

跑完 7 个模型，发现 redundancy pattern 完全是 **model-specific** 的，没有一个 universal 规律。

### Flux.1-dev 的 pattern

前期 step 的 L1 大（变化剧烈），后期小（稳定）。但 **step 16 附近突然跳一下**——所有 layer 都在这个位置出现 spike。深层 layer 的 L1 比浅层大。

直觉：Flux 用 Rectified Flow，从纯 noise 到 clean image。step 16 大概对应 noise level $\sigma \approx 0.5$，是 image 大致结构成型的临界点。过了这个点，model 主要在做 detail refine，变化就小了。

### Pixart-Alpha 的 pattern

完全不一样。浅层 layer 的 L1 几乎为零（全程冗余），深层 layer 在 step 16-10 之间有个 bell shape。而且 $A$ 的变化远大于 $K$ 和 $V$。

直觉：浅层只做 spatial processing，很稳定。深层融合 text 和 image feature，在 structure formation 期间变化剧烈。$A$ 变化大但 $K,V$ 变化小，说明是 query $Q$ 在主导变化——相同的 key/value 被重新加权了。

### Stable-Diffusion-3 的 pattern

U 形——开头和结尾 L1 大，中间小。$K,V,A$ 同 pattern。

直觉：SD-3 是 MMDiT，text 和 image 双向交互。前期从 noise 出发变化大，中期 structure 稳定，后期做 final detail sharpening 又开始折腾 attention pattern。这跟 PAB 在 Open-Sora 上看到的 U 形吻合，但 PAB 错以为这是 universal。

### CogVideoX-5B 的 pattern

三段式：先降（step 50-45）→ 升（45-40）→ 降（40-0）。$K$ 和 $A$ 比 $V$ 冗余。

直觉：video 模型要处理 temporal dimension。前 5 步是 noise settle，step 5-10 建立 temporal coherence，后期稳定。$V$ 变化大可能因为 video 的 motion 信息主要在 value 里流动。

### Latte-1 的 pattern

大部分 step 高冗余（L1 很低），但 **末尾几个 step 突然 L1 上升**。

直觉：Latte-1 大部分时间在"磨"细节，最后几步突然 critical refinement。

### Open-Sora 的 pattern

$A$ 的 L1 显著高于 $K,V$。$K,V$ 是前期高后期低，$A$ 是两端高中间低。

直觉：$A$ 和 $K,V$ 解耦了——相似的 $K,V$ 产生差异大的 $A$，因为 softmax 非线性放大了 query 的差异。

### Mochi-1-preview 的 pattern

前期 L1 在 0 和 0.1-0.5 之间剧烈跳变，后期先 peak 再降。

直觉：10B 大模型在前期"探索"处理哪部分信息，attention pattern 不稳定。这种 chaotic early behavior 让 caching 很难做。

---

## 关键 ablation：到底什么影响 pattern

### Prompt 的影响

测了 1000 个 MS-COCO prompt 的 coefficient of variation（标准差除以均值）：

| Target | Flux | Pixart | SD-3 | CogVideoX | Open-Sora | Latte-1 |
|--------|------|--------|------|-----------|-----------|---------|
| K | 10.0% | 10.4% | 10.5% | 13.9% | 11.6% | 12.6% |
| V | 10.0% | 11.4% | 11.1% | 14.1% | 13.7% | 12.7% |
| A | 10.8% | 12.7% | 11.3% | 14.9% | 15.0% | 14.1% |

只有 10-15% 的波动。意思是：换 prompt，redundancy pattern 基本不变。

直觉：prompt 决定"生成什么内容"，redundancy pattern 反映"模型怎么处理信息流"。前者是 content-level，后者是 architecture-level。所以 prompt 只贡献一点点 noise。

### Step Count 的影响

Flux 默认 28 步，作者跑了 56 步和 14 步对比。

发现：56 步时 sharp transition 出现在 step 34（34/56 ≈ 0.61），28 步时在 step 16（16/28 ≈ 0.57），14 步时 plot 更平滑但 transition 相对位置差不多。

**关键 insight**：transition 的相对位置稳定，说明它对应的是某个 noise level 的临界值，跟 step 数没关系。redundancy 是 noise level $\sigma_t$ 的函数，step index 只是 $\sigma_t$ 的离散采样。

### Scheduler 的影响

把 Flux 的 scheduler 换成 SD-3 的，pattern 几乎不变。

直觉：scheduler 决定采样轨迹的精确形状，但相同 architecture 对相同 noise level 的 input 产生相似的 internal representation。所以 scheduler 换了也没事。

---

## 说了这么多，结论就两条

1. **每个 DiT 模型的 redundancy distribution 都有自己的 pattern，没有 universal 规律。**之前 L2C、TGATE、PAB 的矛盾结论是因为各看各的模型。

2. **同一个模型内，pattern 对 prompt、step count、scheduler 都很稳定。**一旦 profile 一次，cache 策略就可以固化到 model release 里。

因果关系的直觉：

$$\text{Redundancy Pattern} = F(\text{Architecture}) + \epsilon(\text{Prompt, Steps, Scheduler})$$

$F$ 是 architecture 决定的 dominant factor，$\epsilon$ 是 10-15% 量级的小扰动。

---

## 对做 cache 的人意味着什么

不能指望一个 universal cache 策略。得 per-model profile，per-layer per-step design cache policy。

| Model | Cache 策略直觉 |
|-------|----------------|
| Flux.1-dev | 避开 step 16 附近，后期可以激进 cache |
| Pixart-Alpha | 浅层全程 cache，深层在 middle steps 谨慎 |
| SD-3 | Middle steps 激进 cache，两端谨慎 |
| CogVideoX-5B | 注意 step 5-10 的上升区 |
| Latte-1 | 末尾几步不能 cache |
| Open-Sora | $A$ 要谨慎，$K,V$ 可以激进 |
| Mochi-1 | 前期波动大，需要 adaptive cache |

---

## 我的 commentary

这篇 paper 的价值在于"把 field 里的 confusion 整理清楚"。它没提出新 method，但阻止了大家继续在错误 generalization 上浪费时间。

它 implicit 揭示了一个 deep insight：**redundancy 是 architecture 的 invariant**。这跟 scaling law 类似——某种 architecture-level 的 emergent property。

但它也有不足：
- 没解释 *为什么* 不同 architecture 有不同 pattern（缺 mechanistic explanation）
- 没讨论 redundancy 和 generation quality 的 trade-off（cache 多激进开始掉点？）
- Mochi-1 数据不完整（只跑 100 prompt）
- 没有理论框架把 L1 distance 跟 noise level 关联起来

Future work 方向里，"dynamic cache adaptation" 最有意思——运行时根据 attention output 的 running statistics 决定要不要 reuse。这需要 lightweight redundancy estimator，可能是个小 neural network 或者 statistical test。

另外一个潜在方向：**noise-level-conditional caching**。既然 pattern 是 $\sigma_t$ 的函数，那 cache policy 应该 condition 在当前 noise level 上，step index 只是 proxy。如果直接用 $\sigma_t$ 做 condition，可能跨 scheduler 都能 generalize。

---

## 相关链接

- Project repo: https://github.com/xdit-project/DiTCacheAnalysis
- L2C: https://arxiv.org/abs/2406.01733
- TGATE: https://arxiv.org/abs/2404.02747
- PAB: https://arxiv.org/abs/2408.12588
- DeepCache (U-Net 时代): https://arxiv.org/abs/2312.00958
- DiTFastAttn: https://arxiv.org/abs/2406.08552
- ∆-DiT: https://arxiv.org/abs/2406.01125
- xDiT inference engine: https://github.com/xdit-project/xDiT
- PipeFusion: https://arxiv.org/abs/2405.14438
- Flux.1-dev: https://huggingface.co/black-forest-labs/FLUX.1-dev
- Pixart-Alpha: https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS
- Stable-Diffusion-3: https://huggingface.co/stabilityai/stable-diffusion-3-medium
- CogVideoX: https://huggingface.co/THUDM/CogVideoX-5b
- Open-Sora: https://github.com/hpcaitech/Open-Sora
- Latte-1: https://huggingface.co/maxin-cn/Latte-1
- Mochi-1: https://huggingface.co/genmo/mochi-1-preview

---

# 论文详解：UNVEILING REDUNDANCY IN DIFFUSION TRANSFORMERS

## 1. 论文核心问题与Motivation

这篇 paper 来自 Tencent 团队，核心聚焦于 Diffusion Transformers (DiTs) inference latency 优化中的一个关键 phenomenon——**input temporal redundancy**。在 diffusion model 采样过程中，相邻 diffusion steps 的 activations 之间存在高度相似性，这为 caching 机制提供了理论基础。然而，之前的研究 (L2C、TGATE、PAB) 观察到的 redundancy pattern 互相冲突，导致这个领域出现 confusion。

作者提出的核心 question：**redundancy distribution 究竟是 model-intrinsic 的属性，还是由 prompt / step count / scheduler 等外部因素决定？**

参考链接：
- Project: https://github.com/xdit-project/DiTCacheAnalysis
- L2C paper: https://arxiv.org/abs/2406.01733
- TGATE paper: https://arxiv.org/abs/2404.02747
- PAB paper: https://arxiv.org/abs/2408.12588

---

## 2. Diffusion Model 背景公式解析

### 2.1 Denoising Process 公式 (1)

论文给出的核心 update 公式：

$$\boldsymbol{x}_{t-1} = \mathrm{Update}(\boldsymbol{x}_t, t, \epsilon_t), \quad \epsilon_t = \epsilon_\theta(\boldsymbol{x}_t, t, c)$$

变量含义：
- $\boldsymbol{x}_t$：第 $t$ 步的 noisy latent representation，shape 通常为 $\mathbb{R}^{B \times C \times H \times W}$（image）或 $\mathbb{R}^{B \times F \times C \times H \times W}$（video，多一维 frame $F$）
- $t$：当前 diffusion timestep，$t \in \{T, T-1, \dots, 1, 0\}$，注意 diffusion 过程是从 $t=T$ 走向 $t=0$
- $\epsilon_t$：模型预测的 noise
- $\epsilon_\theta$：参数为 $\theta$ 的 noise-prediction network（即 DiT backbone）
- $c$：condition，如 text embedding（来自 CLIP/T5 encoder）
- $\mathrm{Update}$：sampler 特定的 update function，对于 DDIM：

  $$x_{t-1} = \sqrt{\alpha_{t-1}} \cdot \hat{x}_0 + \sqrt{1-\alpha_{t-1}} \cdot \epsilon_t$$

  其中 $\hat{x}_0 = (x_t - \sqrt{1-\alpha_t} \cdot \epsilon_t) / \sqrt{\alpha_t}$ 是 predicted clean image，$\alpha_t$ 是 cumulative noise schedule。对于 Flow Match Euler（Flux.1-dev 和 SD-3 使用的），Update 是 ODE solver step：

  $$x_{t-1} = x_t + (t - t') \cdot v_\theta(x_t, t, c)$$

  其中 $v_\theta$ 是 velocity prediction，$(t - t')$ 是 timestep difference。

### 2.2 为什么 Redundancy 存在

直观理解：在 diffusion 采样过程中，相邻 timestep 的 $x_t$ 与 $x_{t+1}$ 之间只差一个小的 update step。当 noise schedule 比较平滑时（特别是 late stage，$t \to 0$），$x_t$ 的变化幅度小，导致通过 $\epsilon_\theta$ 后内部 activations 的变化也小。这就是 **input temporal redundancy** 的物理来源。

---

## 3. Redundancy 度量方法详解

### 3.1 核心度量定义

对于 model $\mathcal{M}$、prompt $p$、layer $j$、diffusion step $i$，定义：

$$X_{\mathcal{M}(p)}^{i,j}, \quad X \in \{K, V, A\}$$

- $K = X W_K$：attention 的 Key 矩阵，$X$ 是 layer $j$ 的 input，$W_K \in \mathbb{R}^{d \times d_k}$
- $V = X W_V$：Value 矩阵
- $A = \mathrm{softmax}(\frac{Q K^T}{\sqrt{d_k}}) V$：attention layer 的 output

L1 distance 度量相邻 step 的差异：

$$\Delta X_{\mathcal{M}(p)}^{i,j} = \| X_{\mathcal{M}(p)}^{i,j} - X_{\mathcal{M}(p)}^{i+1,j} \|_1$$

注意：在 diffusion 过程中，**Step $i+1$ 出现在 Step $i$ 之前**（因为是从 $T$ 走到 $0$）。L1 distance 越小，redundancy 越高，越适合 caching。

为什么选 L1 而非 L2？L1 对 outlier 更鲁棒，对于 attention activation 中可能存在的 sparse high-magnitude values，L1 提供更稳定的统计。

### 3.2 Coefficient of Variation 用于 Prompt 稳定性分析

为评估 prompt 对 redundancy 的影响，定义：

$$\mathrm{CV}(X) = \frac{\mathrm{std}([\Delta X_{\mathcal{M}(p_1)}^{i,j}, \Delta X_{\mathcal{M}(p_2)}^{i,j}, \dots, \Delta X_{\mathcal{M}(p_N)}^{i,j}])}{\mathrm{mean}([\Delta X_{\mathcal{M}(p_1)}^{i,j}, \dots, \Delta X_{\mathcal{M}(p_N)}^{i,j}])}$$

CV 是无量纲的相对离散度度量，可以跨不同 layer / step 比较。CV ~ 10-15% 意味着 prompt 带来的变化只是 baseline 的十分之一量级，远小于 model 本身带来的差异。

---

## 4. 之前研究观察的 Conflicts

| 研究 | 模型 | 观察 Pattern |
|------|------|--------------|
| L2C | DiT (small) | Late steps 的 difference 显著高于 early steps；MHA > FFN |
| TGATE | SD-2.1 | Cross-attention 在 step 5-10 后趋于零 |
| PAB | Open-Sora | U-shaped，中间低两端高 |

这三个观察互相矛盾——L2C 说 late steps difference 高（redundancy 低），而 TGATE 说 late steps 趋于零（redundancy 高）。这种矛盾说明：**每个研究只看了自己 model 的局部 pattern，没有 generalization**。这是本文要解决的核心问题。

---

## 5. 实验设置

### 5.1 模型集合（Table 1 解析）

| Model | #Params | #Attn Layers | #Diffusion Steps |
|-------|---------|--------------|------------------|
| Flux.1-dev | 12B | 57 | 28 |
| Pixart-Alpha | 0.6B | 28 | 20 |
| Stable-Diffusion-3 | 2B | 24 | 28 |
| CogVideoX-5B | 5B | 30 | 50 |
| Open-Sora | 1.1B | 56 | 30 |
| Latte-1 | 0.7B | 56 | 50 |
| Mochi-1-preview | 10B | 48 | 64 |

观察：
- 参数量跨度极大（0.6B 到 12B），保证 architecture diversity
- Attention layers 数量差异显著（24 到 57）
- Diffusion steps 也各异（20 到 64），让 step count 影响可以被 ablate

数据集：MS-COCO 前 1000 个 captions 作为 prompts（Mochi-1-preview 因 latency 高只用前 100）。

### 5.2 Architecture 差异直觉

- **Flux.1-dev**：Rectified Flow Transformer，使用 double-stream 和 single-stream block 组合， conditioning 通过 extra input tokens（T5 + CLIP）
- **Pixart-Alpha**：纯 transformer blocks，cross-attention 做 text conditioning
- **SD-3**：MMDiT，text 和 image 在 shared attention 中交互
- **CogVideoX-5B**：3D attention，专门为 video 设计
- **Open-Sora / Latte-1 / Mochi-1**：都是 video DiT，但 spatial-temporal attention 的组织方式不同

这些架构差异是导致 redundancy pattern 不同的根本原因。

---

## 6. Prompt 影响实验（Table 2 详解）

### 6.1 数据表

| Target | Flux.1-dev | Pixart-Alpha | SD-3 | CogVideoX-5B | Open-Sora | Latte-1 | Mochi-1-preview |
|--------|------------|--------------|------|--------------|-----------|---------|-----------------|
| K | 10.0% | 10.4% | 10.5% | 13.9% | 11.6% | 12.6% | - |
| V | 10.0% | 11.4% | 11.1% | 14.1% | 13.7% | 12.7% | - |
| A | 10.8% | 12.7% | 11.3% | 14.9% | 15.0% | 14.1% | - |

### 6.2 分析

- **Image models**（Flux, Pixart, SD-3）：CV 约 10-13%，比较稳定
- **Video models**（CogVideoX, Open-Sora, Latte-1）：CV 略高，13-15%。原因可能是 video 的 temporal dimension 引入更多 variance，且 video prompts 复杂度更高
- **A 的 CV > K 和 V 的 CV**：Attention output 经过 softmax 归一化后，对 input perturbation 更敏感（softmax 的非线性放大效应）
- **Mochi-1-preview** 缺失：因为只用了 100 prompts，统计样本不够，且 compute 限制没跑完

**Intuition**：prompt 决定的是「生成什么内容」，而 redundancy pattern 反映的是「模型如何处理信息流」，后者是 architecture 的属性。所以 prompt 只贡献 ~10-15% 的 variability，而 model 切换会带来量级上的根本变化。

---

## 7. 各 Model 的 Redundancy Pattern 深度分析

### 7.1 Flux.1-dev (Figure 2)

Pattern 特征：
- **整体下降趋势**：early steps（high noise region）L1 distance 高，late steps 低
- **Step 16 附近 sharp transition**：所有 layers 在此处出现 L1 distance 骤增
- **Deep layers > shallow layers**：深层 difference 更大

直觉解释：
- Flux 用 Rectified Flow，从 $t=1$（noise）到 $t=0$（clean）。Early steps 处理 high-frequency noise 结构，每步 information change 大
- Step 16 对应 noise schedule 中约 $\sigma = 0.5$ 附近——这是 image structure 形成的关键时刻（coarse-to-fine transition）
- 深层网络处理 high-level semantic，对 input 变化更敏感

### 7.2 Pixart-Alpha (Figure 3)

Pattern 特征：
- **Shallow layers L1 ~ 0**：浅层几乎完全冗余
- **Deep layers bell-shaped**：在 step 16-10 之间 L1 distance 上升
- **A 的 L1 >> K 和 V 的 L1**：attention output 变化远大于 K, V

直觉解释：
- Pixart 用 cross-attention 做 text conditioning。Shallow layers 主要做 spatial processing，stable across steps
- Deep layers 融合 text 和 image features，在 middle steps（structure 形成期）变化最剧烈
- A 变化大但 K, V 变化小说明：相似 query 在重新加权 attention pattern，Q (query) 是主导变化的因素

### 7.3 Stable-Diffusion-3 (Figure 4)

Pattern 特征：
- **U-shaped**：两端高，中间低
- **Deep layers 更显著**
- **K, V, A 同 pattern**

直觉解释：
- SD-3 是 MMDiT，text 和 image 在 shared attention 中双向交互
- Early steps：从纯 noise 出发，每步变化大
- Middle steps：coarse structure 已稳定，主要 refine 细节
- Late steps：final detail sharpening，attention pattern 重新聚焦，导致 L1 distance 上升
- U-shaped 与 PAB 在 Open-Sora 上的发现吻合，但 PAB 误以为这是 universal pattern

### 7.4 CogVideoX-5B (Figure 5)

Pattern 特征：
- **三段式**：先下降（steps 50-45）→ 上升（45-40）→ 下降（40-0）
- **K 和 A 冗余 > V**

直觉解释：
- CogVideoX 用 expert transformer 处理 video，有专门的 temporal attention
- 前 5 步快速降冗余：noise 初始化后的快速 settle
- Steps 5-10 上升：temporal coherence 建立阶段
- 后期下降：temporal structure 稳定
- V 变化大可能与 Value 的「信息载体」角色相关——video 的 motion 信息主要在 V 中流动

### 7.5 Latte-1 (Figure 6)

Pattern 特征：
- **大部分 steps 高冗余**（L1 低）
- **末尾几个 steps 突然 L1 上升**

直觉解释：
- Latte-1 用 spatial-temporal factorized attention，大部分 time step 处理稳定
- 末尾突增：final detail enhancement 阶段，model 重新调整 attention pattern 做 detail sharpening
- 这是典型的 "stuck in refinement mode 然后突然 critical refinement" pattern

### 7.6 Open-Sora (Figure 7)

Pattern 特征：
- **A 的 L1 显著高于 K 和 V**：A 是 dominant 变化源
- **K, V：early 高 late 低**
- **A：两端高中间低**

直觉解释：
- Open-Sora 也是 spatial-temporal 分解
- K, V 反映 key/value representation，early steps 处理 noise 抽取，变化大
- A 反映 attention pattern，在 structure formation（early）和 detail refinement（late）都需要重排
- A 与 K, V 解耦说明：相似 K, V 可以产生差异大的 A，因为 softmax 的非线性放大 query 差异

### 7.7 Mochi-1-preview (Figure 8)

Pattern 特征：
- **Initial stages 剧烈波动**：L1 在 0 和 0.1-0.5 之间跳变
- **Late stages：先 peak 后下降**
- **不同 layer 行为差异大**

直觉解释：
- Mochi-1 是 10B 大模型，capacity 足够处理 complex dynamics
- Initial fluctuation 可能反映 model 在「选择」处理哪部分 information，是一种 attention pattern 的探索行为
- 这种 chaotic early behavior 给 caching 带来挑战——不能简单 cache 整个 phase

---

## 8. Step Count Ablation (Figures 9, 10)

### 8.1 实验设计

Flux.1-dev 默认 28 步。做两组对比：
- **56 步**：步数加倍
- **14 步**：步数减半

### 8.2 结果

| Step Count | Sharp Transition 位置 | 相对位置 |
|------------|------------------------|----------|
| 28 (default) | Step 16 | 16/28 ≈ 0.57 |
| 56 | Step 34 | 34/56 ≈ 0.61 |
| 14 | (smoother，但仍可见) | ~0.57 |

### 8.3 关键 insight

**Sharp transition 的相对位置稳定**，说明这不是 step 离散化的 artifact，而是与 noise schedule 中某个绝对 noise level 对应——大致对应 image structure formation 的临界 $\sigma$ 值。这意味着 redundancy pattern 是 **noise level 的函数**，而非 step index 的函数。

公式直觉：

$$\Delta X^{i,j} \approx f(\sigma_t, \text{layer } j, \text{model } \mathcal{M})$$

其中 $\sigma_t$ 是 step $i$ 对应的 noise level。当 step count 变化时，$\sigma_t$ 的取值集合变化，但 $f$ 函数形状不变。

### 8.4 14 步时的 smoothing

14 步时 plot 更平滑的原因：
- 每步 update 更大，相邻 step 的 $x_t$ 差异更大，单个 step 测得的 L1 噪声相对小
- 离散采样点稀疏，sharp features 被平均化

---

## 9. Scheduler Ablation (Figure 11)

### 9.1 实验

Flux.1-dev 默认用 Flow Match Euler Discrete Scheduler（带 specific sigma schedule）。SD-3 也用 Flow Match Euler Discrete 但配置不同。作者把 Flux 的 scheduler 替换为 SD-3 的，观察 L1 distance 分布。

### 9.2 结果

Figure 11 与 Figure 2 几乎相同——scheduler 切换不改变 redundancy pattern。

### 9.3 解释

Scheduler 决定的是「采样路径在 latent space 中的轨迹形状」，但相同 model architecture 对相同 noise level 的 input 产生相似的 internal representation。Scheduler 主要影响 trajectory 的精确位置，不影响 representation 的 intrinsic redundancy property。

---

## 10. 总结：Redundancy 的 Causal Model

基于所有实验，可以建立这样的因果直觉：

$$\text{Redundancy Pattern} = F(\text{Model Architecture}) \oplus \epsilon(\text{Prompt}, \text{Step Count}, \text{Scheduler})$$

其中 $F$ 是 architecture 决定的 dominant factor，$\epsilon$ 是小幅扰动（10-15% 量级）。这意味着：

1. **Architecture-specific caching is necessary**：不能指望 universal cache strategy
2. **Once profiled, stable**：profile 一次即可，cache strategy 可以固化到 model release
3. **Per-layer, per-step strategy**：不同 layer 在不同 step 有不同 redundancy，cache policy 应该 fine-grained

---

## 11. 对 Cache 策略设计的 Implications

基于论文的发现，cache 策略应该是：

$$\text{Cache}(i, j) = \begin{cases} \text{reuse} & \text{if } \Delta X^{i,j} < \tau \\ \text{recompute} & \text{otherwise} \end{cases}$$

其中 $\tau$ 是 redundancy threshold。但 threshold 选择需要 model-specific：
- Flux.1-dev：避开 step 16 附近，late steps 可激进 cache
- Pixart-Alpha：shallow layers 全程 cache，deep layers 在 middle steps 谨慎
- SD-3：middle steps 激进 cache，early/late 谨慎
- CogVideoX-5B：注意 5-10 step 之间的上升区
- Latte-1：末尾几个 steps 不能 cache
- Open-Sora：A 的 cache 要谨慎，K, V 可以激进
- Mochi-1：early stage 波动大，需要 adaptive cache

---

## 12. Future Work 方向

论文最后提出两个方向：

1. **减少 inference steps 而非近似**：通过更好的 ODE solver 或者 distillation
2. **Dynamic cache adaptation**：运行时检测 redundancy 并自适应调整 cache 策略，而非 static profile

第二个方向特别有意思——可以想象一种 online 的 cache controller，基于 attention output 的 running statistics 来决定是否 reuse。这需要 lightweight 的 redundancy estimator，可能是某个小 neural network 或者 statistical test。

---

## 13. 个人 Commentary

这篇 paper 的价值不在于提出新 method，而在于做一个 **clarifying systematic study**。在 ML research 中，这种 "把别人互相矛盾的 observation 整理清楚" 的工作极其重要——它阻止了 field 在错误的 generalization 上浪费时间。

**亮点**：
- 实验设计严谨，ablation 完整
- 提供 open-source tool，对未来研究有 enabling 价值
- Implicit 地揭示了 redundancy 是 architecture 的 invariant——这是个 deep insight

**不足**：
- 没有深入分析 **为什么** 不同 architecture 有不同 pattern（缺乏 mechanistic explanation）
- 没有讨论 redundancy 与 generation quality 的 trade-off
- Mochi-1 数据不完整
- 没有理论框架解释 L1 distance 与 noise level 的关系

**潜在扩展**：
- 研究 redundancy 与 model scale 的 scaling law
- 跨 model 的 redundancy transfer learning（是否能找到某些 architecture family 的共性）
- Redundancy 在 distillation / fine-tuning 后的变化
- 与 quantization、pruning 的交互

---

## 14. 相关工具与资源

- **xDiT**：作者团队的 DiT inference engine，支持 pipeline parallelism，https://github.com/xdit-project/xDiT
- **PipeFusion**：patch-level pipeline parallel for DiT，https://arxiv.org/abs/2405.14438
- **DiTFastAttn**：attention compression for DiT，https://arxiv.org/abs/2406.08552
- **∆-DiT**：training-free DiT acceleration，https://arxiv.org/abs/2406.01125
- **DeepCache**：U-Net 时代的 caching，https://arxiv.org/abs/2312.00958

---

## 15. 关键 Takeaways

1. **Redundancy distribution 是 model architecture 的 intrinsic property**，对 prompt、step count、scheduler 鲁棒（10-15% variability）
2. **不同 DiT model 的 pattern 差异显著**，无法 universal generalization
3. **Per-model profiling + tailored caching** 是正确路径
4. **Sharp transitions** 通常对应 noise level 的 critical value（如 structure formation threshold），与 step count 无关
5. **A vs K/V 的差异** 揭示了 attention 的非线性放大效应——相似 K, V 可以产生差异大的 A
6. **Video models 比 image models 略不稳定**，prompt 对 video redundancy 影响更大

这篇论文让我对 DiT inference 的 caching 优化有了新 intuition：cache 策略必须 model-specific，且应该基于 noise level（而非 step index）来 design cache policy。未来的方向可能是 **noise-level-conditional caching**——根据当前 $\sigma_t$ 动态决定 cache 哪些 layer。
