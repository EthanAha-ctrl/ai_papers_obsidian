---
source_pdf: stablediffusion.pdf
paper_sha256: caec405ec5d24dbfd505a51ad91d7ad7490dcfde6aa63893f9448894c37e331f
processed_at: '2026-08-12T10:40:22-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrew 老哥这篇博客写得非常扎实，但里面充斥着 ODE solver、SDE、ancestral 等数学术语。Let's build the intuition from scratch. 我们抛开那些复杂的常微分方程公式，直接从直觉出发，看看 Stable Diffusion 的 sampler 到底在干什么。

### 1. 核心比喻：在泥潭里捞雕像

想象一下，你在潜在空间 这个 65,536 维的高维空间里，有一座完美的雕像（你想要的最终图像）。但是，这座雕像被一层又一层厚厚的泥巴（高斯噪声 Gaussian noise）完全包裹住了。

- **Diffusion Process (扩散过程)**：训练时，我们不断往雕像上扔泥巴，直到它变成一个完全随机的泥球，并记录下扔泥巴的过程。
- **Denoising/Sampling (去噪/采样)**：生成图像时，我们拿一个随机的泥球，开始一点点把泥巴洗掉，试图还原出里面的雕像。
- **U-Net (噪声预测器)**：这就是你的洗泥巴工人。你给他看一个沾了泥巴的雕像，他能告诉你“这块泥巴长什么样”，然后你把泥巴 subtract 掉。
- **Sampler (采样器)**：这就是你洗泥巴的**策略**。用高压水枪冲？用小刷子慢慢刷？还是洗两步退一步看看？

### 2. Noise Schedule：洗泥巴的节奏控制

你不可能用同样的力气从头洗到尾。Noise schedule 就是控制每一步要洗掉多少泥巴的计划表。

通常，第一步泥巴最厚，我们大刀阔斧地洗；越往后，雕像细节越清晰，我们就得小心翼翼地洗。Karras noise schedule 的核心 intuition 就是：在接近终点（泥巴很少）的时候，把洗泥巴的步长切得极小。因为如果在最后阶段步子迈大了，高压水枪很容易把雕像的精细纹理（比如眼睛、头发丝）直接冲毁，这就叫 **truncation error (截断误差)**。

从数学上看，如果我们增加 sampling steps (比如从 15 步加到 30 步)，其实就是在细化这个时间切分：
$$ \Delta \sigma = \sigma_{t} - \sigma_{t-1} $$
当 steps 足够大时，$\Delta \sigma \to 0$，U-Net 计算出的梯度方向 $\epsilon_\theta(x_t, t)$ 就能完美贴合真实的概率流流形，截断误差趋于零，图像就能完美收敛。

### 3. Sampler 门派大乱斗：到底该选哪个？

理解了洗泥巴的过程，我们来看看文章里提到的几类 sampler：

#### A. 老古董派：Old-School ODE Solvers
- **Euler**: 最简单的直男。他看一眼泥巴，算出方向，闭着眼睛走固定的一步，走到哪算哪。公式就是 $x_{t-1} = x_t - \Delta t \cdot \epsilon_\theta(x_t, t)$。纯确定性，算力便宜，但精度一般。
- **Heun**: Euler 的小心眼版本。他走一步之后，会在新位置再看一眼，然后退回来，取两步的平均值。公式上相当于梯形积分。这叫二阶求解器 (2nd order solver)。因为要算两次 U-Net，所以速度慢了一倍，但更准。

#### B. 搞破坏派：Ancestral Samplers (名字里带 "a" 的)
- 比如 **Euler a**, **DPM2 a**。
- 这些家伙有强迫症。每洗掉一点泥巴，他们非要再往雕像上撒一把沙子（注入随机噪声 SDE term）。
- 直觉上，这让雕像一直在轻微抖动。结果就是：**永远无法收敛**。你跑 20 步和跑 40 步，出来的根本是两张不同的图。如果你想要可重复的结果，千万别碰它们。

#### C. 现代主力军：DPM 和 DPM++
- 这是 2022 年后为 diffusion model 专门设计的新算法。它们利用了扩散 ODE 的半线性结构，能用大步长快速跨越高噪声区域，且不偏离流形。
- **DPM++ 2M Karras**: 这是目前的“甜点区”。2M 代表 Multi-step 2nd order，它会利用上一步的梯度信息，所以每步只需算一次 U-Net，但达到了接近二阶的精度。配合 Karras schedule，20-30 步就能出极好的图。
- **DPM++ SDE Karras**: 质量最好的之一，但它带了 SDE (随机微分方程) 特性。相当于它既用了高级的洗泥技术，又撒了点沙子。画面细节极其丰富，但同样不收敛。

#### D. 特殊流派
- **DDIM**: 2020 年的老祖宗。它假设每一步的图像是由“最终图像 + 当前方向 + 随机噪声”线性组合而成的。虽然老，但在低步数 (8-10步) 下感知质量出奇地好。
- **UniPC**: 2023 年的新秀。用 Predictor 预测一步，再用 Corrector 修正误差。能在 5-10 步内出图，但文章实测在常规 20-30 步下，并没有比 Euler 好多少。

### 4. Flux 为什么用最简单的 Euler？

文章评论区有个非常敏锐的问题：现在的巨无霸模型 **Flux** 为什么默认只用最简单的 Euler + Simple scheduler，不用那些花里胡哨的 DPM++？

Andrew 回答得很到位，但让我们再深入挖一下 intuition。Flux dev 是经过了 guidance distillation (引导蒸馏) 的 student model。蒸馏的过程就像是把崎岖的山路填平了。原来的 SD 1.5 里，因为 classifier-free guidance (CFG) 的存在，梯度的 landscape 非常尖锐，U-Net 预测的噪声方向在流形上抖动剧烈，所以需要高阶 solver (比如 DPM++) 去平滑这些抖动。

而 Flux dev 在训练时，已经把 CFG 的梯度整合进了模型内部。它的 score landscape 被打磨得异常平滑。在平滑的曲面上，你用一阶的 Euler 直接顺着坡走，就已经非常精准了。用高阶 solver 反而会因为过度校正或者引入截断误差而弄巧成拙。

### 5. 总结与实操建议

用大白话总结，怎么选 sampler：

1. **日常快速出图，又要稳定可复现**：用 **DPM++ 2M Karras** (20-30 steps)。又快又好，步数调高调低图像基本不变。
2. **追求极致细节，不在乎跑 10 步和 40 步图不一样**：用 **DPM++ SDE Karras** (10-15 steps)。它会给你带来最丰富的质感，但要靠运气抽卡。
3. **玩 Flux 等新一代蒸馏模型**：老老实实用 **Euler** + Simple。模型太好了，不需要复杂的数值求解器来兜底。
4. **绝对不要用**：任何带 **"a"** 的 ancestral sampler (如果你需要精确控制的话)，以及 PLMS (确实过时了)。

### Web Links & References

如果你想深入数学细节，这里是这篇博客引用的核心论文：

- **Karras Schedule (解决噪声分布设计)**: *Elucidating the Design Space of Diffusion-Based Generative Models* (2022) - https://arxiv.org/abs/2206.00364
- **DDIM (经典非线性采样)**: *Denoising Diffusion Implicit Models* (2020) - https://arxiv.org/abs/2010.02502
- **DPM-Solver (极速ODE求解)**: *DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps* (2022) - https://arxiv.org/abs/2206.00927
- **DPM-Solver++ (带Guidance的改进版)**: *DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models* (2022) - https://arxiv.org/abs/2211.01095
- **Probability Flow ODE (SDE转ODE的基石)**: *Score-based generative modeling through stochastic differential equations* (2020) - https://arxiv.org/abs/2011.13456
- **UniPC (Predictor-Corrector机制)**: 官方主页 https://unipc.ivg-research.xyz/
- **K-Diffusion (AUTOMATIC1111背后的代码库)**: Katherine Crowson 的 GitHub https://github.com/crowsonkb/k-diffusion

---

这篇 article 详尽梳理并测试了 Stable Diffusion (潜在扩散模型 Latent Diffusion Models) 中的 sampling 机制。Stable Diffusion 的生成过程本质上是求解一个常微分方程 (ODE) 或者随机微分方程 (SDE)。这篇文章从数学原理、数值方法分类以及实证评估三个维度，详细解析了 AUTOMATIC1111 GUI 中常见的 19 种 samplers。

### Mathematical Foundations: Diffusion ODE/SDE

Stable Diffusion 训练了一个 U-Net $\epsilon_\theta(x_t, t, c)$ 来预测噪声。根据 score-based generative modeling 的理论，反向扩散过程可以写成连续时间的 SDE：

$$ dx = [f(x,t) - g^2(t) \nabla_x \log p_t(x)] dt + g(t) d\bar{w} $$

其中:
- $x_t$: 在时间 $t$ 的 latent state（下标 $t$ 表示时间步，空间维度通常为 $4 \times 64 \times 64$ 或更高维）。
- $f(x,t)$: drift term（漂移项），决定系统的确定性演化方向。
- $g(t)$: diffusion coefficient（扩散系数），控制随机噪声添加的尺度。
- $\nabla_x \log p_t(x)$: score function，可以通过训练的 noise predictor $\epsilon_\theta$ 近似，其关系为 $\nabla_x \log p_t(x) \approx -\frac{\epsilon_\theta(x_t, t)}{\sigma(t)}$。
- $d\bar{w}$: reverse-time Brownian motion（反向布朗运动）。

通过 Probability Flow ODE formulation (Song et al., 2020)，可以将上述 SDE 转换为一个确定性的 ODE，这两者具有相同的边缘概率分布：

$$ dx = \left[ f(x,t) - \frac{1}{2} g^2(t) \nabla_x \log p_t(x) \right] dt $$

Samplers 就是用来数值求解这个 ODE 的离散化算法。文章将这其中的映射关系总结为：
- Time $t \rightarrow$ noise level $\sigma$
- Time quantization $\rightarrow$ noise schedule
- Position $x \rightarrow$ latent image
- Velocity $\rightarrow$ predicted noise $\epsilon_\theta$

### Noise Schedule

Noise schedule 决定了离散化时间步 $\Delta t$ 以及对应的 $\sigma$ 变化轨迹。
- 默认调度：线性或余弦分布。
- Karras schedule (Karras et al. 2022)：$\sigma$ 在接近 0（最后几步）时步长更小。其数学形式可以表示为 $\sigma(t) = (\sigma_{\max}^{\frac{1}{\rho}} + t (\sigma_{\min}^{\frac{1}{\rho}} - \sigma_{\max}^{\frac{1}{\rho}}))^\rho$，其中 $\rho$ 控制曲线的非线性弯曲程度（通常取 7 或 8）。这种调度在高噪声时大步走，低噪声时小步走，有效减少了截断误差 (truncation error)，提升了精细细节的收敛效果。

### Sampler Taxonomy 采样器分类

#### 1. Old-School ODE Solvers 老式 ODE 求解器
这些是传统的数值积分方法，直接应用于扩散模型的 ODE。
- **Euler**: 最简单的一阶求解器。更新公式为 $x_{t-\Delta t} = x_t - \Delta t \cdot \epsilon_\theta(x_t, t)$。这里 $x_{t-\Delta t}$ 是下一步的 latent state，$\Delta t$ 是步长。完全确定性，速度快，但一阶截断误差较大。
- **Heun**: 二阶 Runge-Kutta (RK2) 或梯形法。每步需要两次 U-Net 评估，因此速度慢 2x。公式包括预测步和校正步：
  1. Predictor: $\tilde{x}_{t-\Delta t} = x_t - \Delta t \cdot \epsilon_\theta(x_t, t)$
  2. Corrector: $x_{t-\Delta t} = x_t - \frac{\Delta t}{2} [\epsilon_\theta(x_t, t) + \epsilon_\theta(\tilde{x}_{t-\Delta t}, t-\Delta t)]$
- **LMS (Linear Multistep Method)**: 利用前面 $k$ 步的梯度信息来计算当前步，类似 Adams-Bashforth 方法。公式：$x_{t-\Delta t} = x_t + \sum_{i=1}^{k} \alpha_i \Delta t \cdot \epsilon_\theta(x_{t_i}, t_i)$。其中 $\alpha_i$ 是多步法权重。精度提高，但只需一次 U-Net 评估。

#### 2. Ancestral Samplers 祖先采样器
带有 "a" 后缀（如 Euler a, DPM2 a）。它们在每一步去噪后会重新注入随机噪声（对应于 SDE 的 $g(t) d\bar{w}$ 项）。以 Euler a 为例，公式（概念上）：
$$ x_{t-\Delta t} = \text{ODE\_Step}(x_t, t) + \eta \cdot \sigma_{t-\Delta t} \cdot z $$
其中 $z \sim \mathcal{N}(0, I)$，$\eta$ 为控制随机性的参数。由于每一步都在注入新的随机噪声，图像永远无法稳定在某个确定的流形上，表现为不收敛。改变 step 数量会导致最终的 latent distribution 发生显著变化，图像内容随之剧烈改变。

#### 3. DPM-Solver Family (DPM, DPM++, UniPC)
这些是专门为扩散模型设计的指数积分器。扩散 ODE 具有半线性结构，可以通过指数积分精确求解线性部分，从而允许更大的步长。
- **DPM-Solver (Lu et al. 2022)**: 一阶或二阶求解器。二阶 DPM2 需要两次 U-Net 评估。
- **DPM-Solver++ (DPM++)**: 改进版，更稳定，常用于 guided sampling。
- **DPM++ 2M**: Multi-step version。利用前一步的信息，只需一次 U-Net 评估即可达到接近二阶的精度。文章指出 DPM++ 2M Karras 是最佳选择之一：收敛快，质量好。
- **DPM++ SDE**: 结合了 SDE 和 DPM++。虽然感知质量高，由于注入了噪声，依然不收敛。
- **UniPC (Unified Predictor-Corrector, 2023)**: 基于预测-校正机制。Predictor 预测下一步，Corrector 修正当前步。能在 5-10 步内生成高质量图像。

#### 4. DDIM (Denoising Diffusion Implicit Models)
DDIM 是最早的采样器之一，基于非马尔可夫过程。其核心公式将每一步的图像表示为最终去噪图像、指向当前步的方向以及随机噪声的线性组合：
$$ x_{t-1} = \sqrt{\alpha_{t-1}} \underbrace{\left( \frac{x_t - \sqrt{1-\alpha_t} \epsilon_\theta(x_t, t)}{\sqrt{\alpha_t}} \right)}_{\text{predicted } x_0} + \sqrt{1-\alpha_{t-1} - \sigma_t^2} \cdot \epsilon_\theta(x_t, t) + \sigma_t z $$
其中 $\alpha_t$ 是前向过程累积方差，$z$ 是高斯噪声（当 $\eta > 0$ 时存在）。

### Empirical Evaluation 实验评估

文章通过三个维度评估了这些 samplers：
1. **Image Convergence**: 计算不同 step 数下生成的图像与 40-step 参考图像的 latent distance。Ancestral samplers 曲线震荡，永远不收敛。DPM++ 2M Karras 和 Euler 收敛极快。
2. **Speed**: 分为两档。一阶求解器（Euler, DPM++ 2M）每步只需 1 次 U-Net 评估，耗时为 1x。二阶求解器（Heun, DPM2, DPM++ SDE）需要 2 次评估，耗时为 2x。
3. **Quality**: 使用 BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) 评估无参考图像的感知质量。DPM++ SDE Karras 和 DDIM 在此指标上表现最佳。

### Intuition Building

在 65,536 维的 latent space 中，轨迹的初始阶段步幅大，勾勒出图像的全局构图 (global composition)，接近终点时步幅小，精细化局部细节 (local details)。由于是在极低维度的流形上移动，传统的 ODE solver 容易因为 truncation error 偏离流形。DPM-Solver 利用了扩散 ODE 的半线性特性，通过指数积分减小了这种偏差。

在文章的评论区，作者也讨论了 Flux 模型为何倾向于使用 Euler + Simple scheduler。由于 Flux dev model 经过了 guidance distillation，其 score landscape 相对平滑，不需要高阶的 corrector 来补偿 classifier-free guidance 带来的尖锐梯度，使用简单的 Euler solver 即可高效获得优质结果。

### References
- Karras 2022 (Elucidating the Design Space): https://arxiv.org/abs/2206.00364
- DDIM: https://arxiv.org/abs/2010.02502
- DPM-Solver: https://arxiv.org/abs/2206.00927
- DPM-Solver++: https://arxiv.org/abs/2211.01095
- Score-based SDE / Probability Flow ODE: https://arxiv.org/abs/2011.13456
- UniPC: https://unipc.ivg-research.xyz/
