---
source_pdf: Motion Attribution for Video Generation.pdf
paper_sha256: aa47fd6e83ed62dfff17f5157c9b743ef6c4612fdc36a4bd6f43af0f502a4389
processed_at: '2026-08-05T20:34:36-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Motive 论文人话版

Andrej，我用一个接地气的 frame 帮你 build intuition。

---

## 一句话概括

**你训练 video model 生成视频，但不知道是哪些训练视频教会了它某种运动。Motive 就是个"运动归因器"，帮你找出最有用的那 10% 训练视频，用它继续训练，运动质量比全量训练还牛。**

---

## 类比：调料归因

你炒菜放了一堆调料——盐、酱油、八角、桂皮、辣椒、糖……最后菜很好吃，但你不知道是哪几味起的关键作用。

Motive 做的事就是：**追溯每一味调料对最终味道的贡献**，然后告诉你"下次炒菜只放这几种就够了，效果一样好甚至更好"。

换到 video 生成场景：
- 调料 = training clips
- 味道 = 生成的 motion quality
- "哪几味起作用" = motion attribution

---

## 为什么这件事难

### 难点 1: Motion ≠ Appearance

Video 里有两个东西纠缠在一起：
- **Appearance**：画面长啥样（颜色、纹理、物体形状）
- **Motion**：画面怎么动（轨迹、形变、物理）

以前的 attribution 方法（TRAK、TracIn、Influence Functions）都是为 image 设计的，只懂 appearance。你拿来用，它会因为"两个视频都有蓝色背景"就判定它们 influential，完全忽略运动。

类比：你问一个只会看颜色的厨师"哪味调料让菜变香"，他会说"红色的调料"——其实香味来自八角，颜色根本不相关。

### 难点 2: Video 太贵

- Image attribution：1 张图，1 次 forward + backward
- Video attribution：F 帧，F 倍计算，还要对 timesteps、noise draws 积分
- 10k 训练视频 × 1000 timesteps × 多个 noise = 计算爆炸

直接套 image 方法到 video，单 GPU 要跑几个月。

### 难点 3: 长视频 bias

Raw gradient magnitude 跟视频帧数成正比。50 帧视频的 gradient 天然比 20 帧大，会被 spurious 标为 influential。就像比较两个学生总分，一个考了 10 门一个考了 5 门，你直接比总分不公平，得比平均分。

---

## Motive 怎么做的

### Step 1: 找到画面里"会动的地方"

用 AllTracker（一个 dense optical flow 工具）提取每个 pixel 的位移向量：

$$\mathbf{D}_f(h, w) = (dw, dh)$$

- $\mathbf{D}_f$: frame $f$、位置 $(h, w)$ 的位移
- $dw$: 水平位移
- $dh$: 垂直位移

算 magnitude：$M_f(h,w) = \|\mathbf{D}_f(h,w)\|_2$

Min-max 归一化到 $[0,1]$：
$$\mathbf{W}(f,h,w) = \frac{M_f(h,w) - \min M}{\max M - \min M + \zeta}$$

- $\zeta = 10^{-6}$: 防除零

**人话**：画面每个位置一个权重——动得厉害的地方权重接近 1，静止背景接近 0。

### Step 2: 只在"会动的地方"算 loss

原本 diffusion loss 是对整个 latent 空间所有位置平均：

$$\mathcal{L}_{diff} = \mathbb{E}\left[\|\epsilon_\theta - \epsilon\|_2^2\right]$$

Motive 改成 motion-weighted loss（Eq. 16）：

$$\mathcal{L}_{mot}(\theta; v, c) = \frac{1}{F_v} \text{mean}_{f, \tilde{h}, \tilde{w}}\left[\tilde{\mathbf{W}}(f, \tilde{h}, \tilde{w}) \cdot \tilde{\mathcal{L}}(f, \tilde{h}, \tilde{w})\right]$$

- $F_v$: 视频 $v$ 的帧数（顺带 fix 帧长 bias）
- $\tilde{\mathbf{W}}$: motion mask
- $\tilde{\mathcal{L}}$: per-location squared error

**人话**：背景静止区域的 loss 几乎不算，只有动的地方的 loss 计入。这就把 appearance 噪音从 attribution signal 里挤掉了。

关键 design：这是 **loss-space masking**——只改 loss 的权重，不改 forward pass 的 noise injection。如果在 input space mask，会引入 mask × noise 的复杂 interaction，让 attribution 信号被污染。

### Step 3: 让计算便宜

三个 trick 让 billion-parameter 模型上可行：

**(1) Single-sample + common randomness**

固定一个 timestep $t_{fix} = 751$ 和一个 noise draw $\epsilon_{fix}$，所有 train-test pairs 都用这个。Eq. 7：

$$I_{diff}^2(x_n, x_{test}) = \frac{\nabla_\theta \mathcal{L}_{test}}{\|\nabla_\theta \mathcal{L}_{test}\|} \cdot \frac{\nabla_\theta \mathcal{L}_n}{\|\nabla_\theta \mathcal{L}_n\|}$$

- $t_{fix} = 751$: 1000 步 trajectory 中点
- 共享 $(t_{fix}, \epsilon_{fix})$ 让 single-sample 也能保持 ranking 顺序

**人话**：不要对 1000 个 timestep 都算 gradient 然后平均，只取中间一个 timestep 就够，因为相对顺序保持得很好。Cost 直接降 1000 倍。

为什么中点最好？高 timestep 噪声太多，motion 信号被淹没；低 timestep 视频快成形了，gradient 反映的是 fine detail 不是 motion structure。中点平衡。

**(2) Fastfood projection**

把 14 亿维 gradient 压缩到 512 维：

$$\mathbf{P} = \frac{1}{\xi\sqrt{D'}} S Q \Pi G B Q$$

- $Q$: Walsh-Hadamard matrix（$\mathcal{O}(D \log D)$ 计算）
- $B$: ±1 diagonal Rademacher
- $\Pi$: random permutation
- $G$: Gaussian diagonal
- $S$: rescaling diagonal
- $\xi$: variance normalization

然后存 512 维 cosine（Eq. 10）：
$$I(x_n, x_{test}) = \tilde{g}(x_{test})^T \tilde{g}(x_n)$$

**人话**：14 亿维向量存不下也存不起，用 Fastfood 这种 structured random projection 压到 512 维。压缩后跟没压缩的 ranking 相关性还有 74.7%，但存储降 1000 倍。

**(3) Frame-length normalization（Eq. 11）**

$$\nabla_\theta \mathcal{L}_{diff}(\theta; v, t_{fix}, \epsilon_{fix}) \leftarrow \frac{1}{F} \nabla_\theta \mathcal{L}_{diff}(\theta; v, t_{fix}, \epsilon_{fix})$$

**人话**：长视频 gradient 除以帧数，跟短视频放一个起跑线上比。

### Step 4: 多个 query 投票选最终 subset

你有 50 个 query video（10 类 motion × 5 个），每个 query 都对 10k 训练视频打分。怎么融合？

Majority voting（Eq. 18）：
$$\text{MajVote}_n = \sum_{q=1}^Q \mathbb{I}\left[I_{mot}(v_n, \hat{v}_q) > \tau\right]$$

- $\tau$: percentile cutoff（比如 top 10%）
- 一个训练视频对某个 query 投票，如果它的 score 高于这个 query 的 $\tau$
- 收到最多票的训练视频被选中

**人话**：不是简单求和或平均，是"有多少个 query 觉得你重要"的投票。这避免了不同 query 的 raw score 难以 calibration 的问题。

---

## 实验结果怎么样

### 自动指标（Table 1）

| Method | Subject Consist. | Motion Smooth. | **Dynamic Degree** |
|---|---|---|---|
| Base (无 finetune) | 95.3 | 96.3 | 39.6 |
| Full fine-tuning (100% data) | 95.9 | 96.3 | 42.0 |
| Random selection (10% data) | 95.3 | 96.3 | 41.3 |
| Motion magnitude (选运动大的) | 95.6 | 95.7 | 40.1 |
| V-JEPA embedding | 95.7 | 95.6 | 41.6 |
| Motive w/o motion mask | 95.4 | 96.3 | 43.8 |
| **Motive (10% data)** | **96.3** | 96.3 | **47.6** |

**亮点**：
- 用 10% data，Dynamic Degree 47.6% > 全量 42.0%
- 比 random 41.3% 高 6.3 个点
- 比 motion magnitude baseline 40.1% 高 7.5 个点

**Counterintuitive**：选"motion magnitude 大"的视频（40.1%）甚至比 random（41.3%）还差！说明光选动得多的视频不行，要选 gradient-aligned 的。

### 人类评估（Table 2）

| Comparison | Win | Tie | Loss |
|---|---|---|---|
| Motive vs Base | **74.1%** | 12.3% | 13.6% |
| Motive vs Random | 58.9% | 12.1% | 29.0% |
| Motive vs Full FT | 53.1% | 14.8% | 32.1% |

74.1% 人类觉得 Motive 比 base 好——这是非常强的 perceptual signal。

---

## 几个关键的 Intuition

### Intuition 1: Motive 不是 motion-rich filter

Fig. 6 的数据特别反直觉：
- Top 10% influence 的 mean motion magnitude = 3.85
- Bottom 10% influence 的 mean motion magnitude = 3.69
- 差距仅 4.3%

很多 motion 大的视频 influence 低；很多 motion 适中的视频 influence 高。

**为什么**：Motive 算的是 gradient alignment，是"训练这个视频能不能降低目标 motion 的 loss"，不是"这个视频动得猛不猛"。一个 motion 适中但 pattern 跟 query 完美匹配的 clip，influence 远高于 motion 巨大但 pattern 不匹配的 clip。

类比：你学投篮，跟一个动作标准但节奏跟你一样的人学，比跟一个疯狂花式运球的人学更有用。

### Intuition 2: Loss-space masking 是精髓

如果改成 input-space masking（在 noising 之前 mask pixel），会改变 noise 注入，引入 mask × noise interaction。Loss-space masking 只 reweight gradient，generation process 完全不变。这让 attribution 是纯粹的 motion 贡献度量，跟生成过程 decoupled。

### Intuition 3: Cross-motion overlap 揭示 model 内部 representation

Fig. 7 显示：
- bounce-float overlap 44.4%（共享 oscillation, gravity-driven dynamics）
- free fall-stretch overlap 12.8%（机械上完全不同）

Model 内部 learned representation 里 bounce 和 float 是相关的——它们共享 fundamental dynamics。这给 model interpretability 提供了 tool。

### Intuition 4: Frame-length normalization 必不可少

Fig. 5 直观展示：无 normalization 时 float query 的 top-ranked 样本杂乱无章；有 normalization 时一致显示 wave/floating/surfing。Gradient 与视频长度的相关性从 78% 降到合理水平。

---

## 代码/Runtime 实用信息

- 10k 训练样本 × Wan2.1-T2V-1.3B：单 A100 ~150 小时
- 64 GPU 并行：~2.3 小时
- Gradient 算一次后可复用，加新 query 只需 54 秒
- Project 后存储：10k × 512 维向量，可放入内存
- Fine-tune：4-8 A100，480×832 分辨率，LR=1e-5，1 epoch × 50 repeats

---

## 跟你的工作的连接

你的 "Software 2.0" 视角里，dataset 就是 program。Motive 是 reverse-engineer 这个 program 在 motion 维度的 dependency graph——哪些 training clip 决定了 model 的某个 motion behavior。

你的 micrograd 思路：Motive 的 single-sample estimator 是某种 variance-reduced gradient estimator，跟 control variates 思路相通——共享 $(t_{fix}, \epsilon_{fix})$ 让 train 和 test gradient 的 shared noise 部分抵消。

 Scaling laws 方面：Motive 证明 fine-tuning regime 里 data quality >> data quantity。10% high-influence data > 100% random data。这是 "data quality power law" 的 motion-specific 证据。

---

## Limitations 老实说

1. **150 GPU 小时 upfront cost**——academic researcher 不友好，虽然 amortized
2. **Whole-video granularity**——会稀释 informative intervals
3. **Camera-only motion 没完全 disentangle**——可能误判相机运动为 object motion
4. **没显式 model CFG**——inference-time guidance 会改变 dynamics，attribution 可能 miss 这部分
5. **Targeted adaptation 可能 trade-off base capability**——这是 fine-tuning 的 fundamental tension

---

## 一句话总结

Motive = optical flow mask + loss-space reweighting + Fastfood projection + single-timestep common randomness + majority voting。用 10% data 超过 100% data finetune 的 motion quality，74.1% human win rate。

核心 insight：motion attribution 必须 motion-centric（loss-space masking），gradient alignment 比 motion magnitude 重要（4.3% 差距证明），single-timestep + common randomness 足以保持 ranking（66% agreement）。

---

References:
- 论文主页：https://research.nvidia.com/labs/sil/projects/MOTIVE/
- AllTracker：https://arxiv.org/abs/2506.07310
- Wan2.1：https://arxiv.org/abs/2503.20314
- VBench：https://arxiv.org/abs/2410.03104
- TRAK：https://arxiv.org/abs/2303.14186
- Diffusion-ReTrac：https://arxiv.org/abs/2401.09031
- ICONS（同作者 Xindi Wu）：https://arxiv.org/abs/2501.00654

希望这个人话版本能让你 fast build intuition。如果你想看 motion mask 跟 gradient geometry 之间的更精细 interaction，或者 majority voting 的 information-theoretic 解读，我可以再 dig deep。

---

# Motion Attribution for Video Generation - 深度技术解析

Andrej，这篇 paper 直击 video diffusion 的一个根本盲点：**我们真的理解 data 在塑造 motion 吗？** 整个 field 在 scaling data 与 compute，但 motion 的归因几乎是个 black box。Motive 提供了第一个 motion-centric 的 gradient-based attribution framework，能 trace 生成视频的 dynamics 回到 influential training clips。我会带你拆解方法、公式、实验，并 build intuition。

论文链接：https://research.nvidia.com/labs/sil/projects/MOTIVE/

---

## 1. Problem Formulation: 为什么这是一个真正难的问题

Video generation 的核心特征就是 temporal dynamics——物体如何运动、形变、与物理约束互动。Diffusion models 是 data-driven 的，data 决定了 motion distribution。但现有 attribution 方法（TRAK、TracIn、Influence Functions、Diffusion-ReTrac、Concept-TRAK）都聚焦在 image diffusion 上，解释 static content。

把 image attribution 直接搬到 video 有三大 gap：
1. **Localizing motion**：attribution signal 容易被 static backgrounds 主导
2. **Scaling across time**：gradient 必须跨时间累积，计算量爆炸
3. **Temporal relations**：velocity、acceleration、trajectory coherence 这些 single-frame attribution 根本测不到

Motive 用 motion-weighted loss masks 把 temporal dynamics 从 static appearance 中 isolate 出来，配合 scalable gradient computation 让它在 billion-parameter 模型上可行。

---

## 2. Background: 必要的数学地基

### 2.1 Diffusion 与 Flow Matching in Latent Space

给定一个条件视频生成器 $p_θ(v|c)$，其中：
- $v ∈ ℝ^{F×H×W×3}$ 是视频 clip，F 帧、H 高、W 宽
- $c$ 是 conditioning（text 或多模态 metadata）
- $θ$ 是可训练参数

VAE 编码到 latent：$h = E(v)$，然后在 noisy latents 上训练 denoiser/velocity field。

**Forward noising**（Eq. 1）：
$$z(t, ε) = α_t · h + σ_t · ε, \quad ε \sim \mathcal{N}(0, I), \quad t \in \{1, ..., T\}$$

- $z$: noisy latent
- $α_t$: signal scale（timestep t 处）
- $σ_t$: noise scale
- $ε$: 注入的 Gaussian noise
- $T$: 总 timesteps（通常 1000）

**Diffusion loss**（Eq. 2）：
$$\mathcal{L}_{diff}(θ; v, c) = \mathbb{E}_{t, ε}\left[\|ε_θ(z(t, ε), c, t) - ε\|_2^2\right]$$

- $ε_θ$: 网络预测的 noise
- 目标是让网络从 noisy latent 预测出注入的 noise

**Flow matching loss**（Eq. 3）：
$$\mathcal{L}_{flow}(θ; v, c) = \mathbb{E}_{t, ε}\left[\|f_θ(z(t, ε), c, t) - \dot{z}(t, ε)\|_2^2\right]$$

- $f_θ$: time-dependent velocity field
- $\dot{z} = dz/dt$: interpolant 诱导的瞬时速度

两个 objective 都训练 time-indexed predictors，在 latent space 上对 $t$ 和 $ε$ 积分。这意味 gradient-based attribution 在两者上面临的挑战相似。

### 2.2 Data Attribution 基础

经典 influence function（Eq. 4）：
$$I(x_n, x_{test}) = -∇_θ \mathcal{L}(θ; x_{test})^T H_θ^{-1} ∇_θ \mathcal{L}(θ; x_n)$$
$$H_θ = \frac{1}{N}\sum_{n=1}^N ∇_θ^2 \mathcal{L}(θ; x_n)$$

- $I(x_n, x_{test})$: 训练样本 $x_n$ 对测试样本 $x_{test}$ 的影响
- $H_θ$: Hessian，捕捉 loss 曲率
- 直觉：如果我 upweight 训练样本 $x_n$，模型在 $x_{test}$ 上的预测会变化多少？

**问题**：$H_θ^{-1}$ 在现代网络上计算和存储都不可行。实用方法（TracIn、TRAK）用 gradient inner products 或 gradient feature projections 近似。

### 2.3 Diffusion 中的 Timestep Bias

Diffusion 训练对 timesteps $t$ 和 noise draws $ε$ 聚合 gradient，gradient norms 随 $t$ 系统性变化，导致 timestep bias——aligned with large-norm timesteps 的样本会被 spurious 标为 influential。

**Diffusion-ReTrac**（Eq. 5）通过 normalizing gradients 和 sub-sampling $t, ε$ 减少 bias：
$$I_{diff}(x_n, x_{test}) = \frac{1}{|T_{test}|}\sum_{t,ε \in T_{test}} \frac{∇_θ \mathcal{L}_{diff}(θ; x_{test}, t, ε)}{\|∇_θ \mathcal{L}_{diff}(θ; x_{test}, t, ε)\|} \cdot \frac{1}{|T_n|}\sum_{t,ε \in T_n} \frac{∇_θ \mathcal{L}_{diff}(θ; x_n, t, ε)}{\|∇_θ \mathcal{L}_{diff}(θ; x_n, t, ε)\|}$$

对 $(t, ε)$ 平均 stabilizes estimates，normalization mitigates timestep-induced scale effects。

---

## 3. Method: Motive 的四大支柱

### 3.1 Problem Setup

Fine-tuning corpus $\mathcal{D}_{ft} = \{(v_n, c_n)\}_{n=1}^N$。给定 query video $(\hat{v}, \hat{c})$，给每个 training clip 分配 motion-aware influence score $I(v_n, \hat{v}; θ)$，要求：
- **Predictivity**：rankings 与在 most influential subsets 上 fine-tuning 后的 observed changes 相关
- **Efficiency**：scales to modern video generators，forgo explicit Hessian inversion

### 3.2 Scalable Gradient-based Attribution（§3.2）

这是让 attribution 在 billion-parameter 上可行的核心。

#### (1) Approximating inverse-Hessian

用 identity preconditioner 替代 $H_θ^{-1}$，转化为 gradient similarity。这是 TRAK / TracIn 的标准做法。

#### (2) Common Randomness for Stable Rankings（Eq. 6）

在**相同**的 $(t, ε)$ 对上评估 train 和 test gradients，并对一个小集合 $\mathcal{T}$ 平均：
$$I_{diff}^1(x_n, x_{test}) = \frac{1}{|\mathcal{T}|}\sum_{t,ε \in \mathcal{T}} \frac{∇_θ \mathcal{L}_{diff}(θ; x_{test}, t, ε)}{\|∇_θ \mathcal{L}_{diff}(θ; x_{test}, t, ε)\|} \cdot \frac{∇_θ \mathcal{L}_{diff}(θ; x_n, t, ε)}{\|∇_θ \mathcal{L}_{diff}(θ; x_n, t, ε)\|}$$

**Intuition**：paired averaging 比 independent draws 稳定得多——就像 paired t-test 比 unpaired 更有 power，因为消除了共享随机性的方差。

#### (3) Single-Sample Variant（Eq. 7）

固定单个 $t_{fix}$ 和一个 shared draw $ε_{fix} \sim \mathcal{N}(0, I)$，对**所有** train-test pairs 在 final checkpoint 上使用：
$$I_{diff}^2(x_n, x_{test}) = \underbrace{\frac{∇_θ \mathcal{L}_{diff}(θ; x_{test}, t_{fix}, ε_{fix})}{\|∇_θ \mathcal{L}_{diff}(θ; x_{test}, t_{fix}, ε_{fix})\|}}_{\text{normalized test gradient}} \cdot \underbrace{\frac{∇_θ \mathcal{L}_{diff}(θ; x_n, t_{fix}, ε_{fix})}{\|∇_θ \mathcal{L}_{diff}(θ; x_n, t_{fix}, ε_{fix})\|}}_{\text{normalized train gradient}}$$

**关键 insight**：共享 $(t_{fix}, ε_{fix})$ 让单 sample 也能保持相对 ordering 的足够低 variance。这是 scalability 的核心 trick——cost 从 $\mathcal{O}(|\mathcal{D}||\mathcal{T}|B)$ 降到 $\mathcal{O}(|\mathcal{D}|B)$，其中 $B$ 是单次 forward+backward cost。

Ablation 显示：固定 $t_{fix}=751$（1000 步 denoising trajectory 的中点）与 10 个均匀 spaced timesteps 的 ground truth 有 **66% agreement**。为什么 mid-denoising 最好？
- High timesteps（early denoising）噪声太多，motion cues 被 corrupt
- Low timesteps（late denoising）几乎成形视频，gradients 反映 fine details 而非 semantic structure
- $t=751$ 平衡——既能感知 motion structure，又没被噪声淹没

#### (4) Structured Projection for Reduced Storage（Eq. 8-10）

**Fastfood Johnson-Lindenstrauss projection**：

$$P \in \mathbb{R}^{D' \times D}, \quad P := \frac{1}{\xi\sqrt{D'}} S Q \Pi G B Q$$

- $Q$: Walsh-Hadamard matrix（$\mathcal{O}(D \log D)$ 应用）
- $B$: diagonal Rademacher matrix（±1 随机）
- $\Pi$: random permutation
- $G$: diagonal Gaussian scaling
- $S$: diagonal rescaling
- $\xi$: variance normalization

**Projected normalized gradient**（Eq. 9）：
$$\tilde{g}(θ, x) := \frac{P \nabla_θ \mathcal{L}_{diff}(θ, x, t_{fix}, ε_{fix})}{\|P \nabla_θ \mathcal{L}_{diff}(θ, x, t_{fix}, ε_{fix})\|}$$

**Influence score**（Eq. 10）：
$$I_{diff}^3(x_n, x_{test}) = \tilde{g}(θ; x_{test})^T \tilde{g}(θ; x_n)$$

- 在 $\mathbb{R}^{D'}$ 中的 cosine
- Compute: $\mathcal{O}(D' \log D')$ per projection, $\mathcal{O}(D')$ per dot product
- Storage: $\mathcal{O}(|\mathcal{D}| D')$ 加 $\mathcal{O}(D)$ 的 Fastfood 状态

在实验中 $D = 1,418,996,800$（Wan2.1-T2V-1.3B 的参数量），$D' = 512$。这个 1000x 以上的压缩，abluation（Fig. 4）显示 $D'=512$ 与 full gradient 有 **74.7% Spearman correlation**，$D'=1024$ 也只有 75.7%——边际收益递减很快，$D'=512$ 是 sweet spot。

### 3.3 Frame-Length Bias Fix（§3.3，Eq. 11）

Raw gradient magnitudes 依赖视频帧数 $F$，长视频 gradient 更大，会被 spurious 标为 influential：
$$∇_θ \mathcal{L}_{diff}(θ; v, t_{fix}, ε_{fix}) \leftarrow \frac{1}{F} ∇_θ \mathcal{L}_{diff}(θ; v, t_{fix}, ε_{fix})$$

之后还要做 $\ell_2$ normalization 进一步稳定。

**Ablation**（Fig. 5）：无 normalization 时，gradient 与视频长度相关性 $\rho = 78.0\%$，长视频被 high-rank。Normalizing 把 spurious length correlation 降了 **54.0%**。Fig. 5 直观展示：有 normalization 时 float query 的 top-ranked 样本一致显示 wave/floating/surfing；无 normalization 时 top 样本杂乱无章。

### 3.4 Motion Attribution: 这篇 paper 的真正核心创新

**Vanilla attribution 的问题**：把 video 当一个 single unit 处理，conflates appearance 与 motion，往往因为 share backgrounds/objects 就 rank 高，对 dynamics 没有洞察。

#### Motion Detection and Latent Space Mapping

视频 $v \in \mathbb{R}^{F \times H \times W \times 3}$，VAE 编码到 latent $h = E(v) \in \mathbb{R}^{F \times H/s \times W/s \times C}$，Wan2.1 中 $s=8$（spatial downsample），$C=16$。

用 **AllTracker** [Harley et al., 2025] 在 pixel space 提取 motion：$A = \mathcal{A}(v) \in \mathbb{R}^{F \times H \times W \times 4}$
- 前 2 channel: optical flow maps $A_{:,:,:,0:2}$（pixel 位移）
- 后 2 channel: visibility 和 confidence

**Displacement vector**（Eq. 12）：
$$D_f(h, w) = (A_{f,h,w,0}, A_{f,h,w,1}) = (dw, dh)$$

- $D_f$: frame $f$ 位置 $(h,w)$ 的位移向量
- $dw, dh$: 水平/垂直 pixel 位移

#### Motion-Weighted Gradient Computation

**Motion magnitude**：$M_f(h, w) = \|D_f(h, w)\|_2$

**Min-max normalize**（Eq. 13）：
$$W(f, h, w) = \frac{M_f(h, w) - \min_{f', h', w'} M_{f'}(h', w')}{\max_{f', h', w'} M_{f'}(h', w') - \min_{f', h', w'} M_{f'}(h', w') + \zeta}$$

- $\zeta = 10^{-6}$: numerical stability
- 归一化到 $[0, 1]$，强调**相对** motion saliency 而非 absolute magnitude
- 这避免了 fast camera motion 淹没 subtle object motion

**Bilinear downsample 到 latent grid**（Eq. 14）：
$$\tilde{W}(f, \tilde{h}, \tilde{w}) = \text{Bilinear}(W(\cdot, \cdot, \cdot), F, H/s, W/s)$$

- $(\tilde{h}, \tilde{w})$: latent grid indices
- 让 mask 住在 gradient 计算的地方

**Per-location squared error**（Eq. 15）：
$$\tilde{\mathcal{L}}_{θ, v, c}(f, \tilde{h}, \tilde{w}) = \left([\epsilon_θ(z(v, t_{fix}, ε_{fix}), t_{fix}, c)]_{f, \tilde{h}, \tilde{w}} - [\epsilon_{target}(t_{fix}, ε_{fix})]_{f, \tilde{h}, \tilde{w}}\right)^2$$

**Motion-weighted loss**（Eq. 16）：
$$\mathcal{L}_{mot}(θ; v, c) = \frac{1}{F_v} \text{mean}_{f, \tilde{h}, \tilde{w}}\left[\tilde{W}_{v, c}(f, \tilde{h}, \tilde{w}) \cdot \tilde{\mathcal{L}}_{θ, v, c}(f, \tilde{h}, \tilde{w})\right]$$

- $F_v$: 视频 $v$ 的帧数（video-dependent frame-length correction）
- 当 $\tilde{W}$ 全 1 时，recovers standard objective
- 这是 **loss-space masking**，forward noising 和 generation 不变，只 reweight attribution——避免 motion weighting 与 noise injection 之间的 interaction

**Motion-aware gradient**（Eq. 17）：
$$I_{mot}(v_n, \hat{v}) = \tilde{g}_{mot}(θ, \hat{v})^T \tilde{g}_{mot}(θ, v_n)$$
$$\tilde{g}_{mot}(θ, v) := \frac{P g_{mot}(θ, v, t_{fix}, ε_{fix})}{\|P g_{mot}(θ, v, t_{fix}, ε_{fix})\|}, \quad g_{mot} := \nabla_θ \mathcal{L}_{mot}$$

**为什么 loss-space masking 是关键 design choice**：
如果在 forward noising 时 reweight，会引入 mask × noise 的复杂 interaction。Loss-space masking 只 reweight attribution gradient，让生成过程保持不变。这让 attribution 更纯粹地反映 motion 贡献。

#### Architecture/Method Visualized

Figure 1 顶部展示三步 motion-gradient 计算：
1. AllTracker 检测 motion
2. 计算 motion-magnitude patches
3. 应用 loss-space motion masks 聚焦 dynamic regions

底部展示 scalable pipeline：
1. Single-sample + common randomness + projection 计算 per-pair score
2. Aggregate（majority vote）
3. Final ranking → fine-tuning subset

### 3.5 Fine-tuning Subset Selection（§3.5）

**Single-query**：top-K 最高 score。$K$ 通常选 dataset size 的 1-10% percentile。

**Multi-query majority voting**（Eq. 18）——借鉴 ICONS [Wu et al., 2024]：
$$\text{MajVote}_n = \sum_{q=1}^Q \mathbb{I}\left[I_{mot}(v_n, \hat{v}_q) > \tau\right]$$
$$\mathcal{S}_{vote}(K) = \{v_n | v_n \text{ in top-}K \text{ by MajVote}\}$$

- $Q$: query 数量（实验中 50，10 类 motion × 5）
- $\tau$: percentile cutoff
- 一个样本对某个 query 投票如果其 score 高于 $\tau$
- Consensus score: 收到的总票数

**Intuition**：emphasize 跨多个 query consistently influential 的样本，不要求 cross-query raw score calibration。

### 3.6 Computational Efficiency Analysis

| Component | Complexity | Runtime |
|---|---|---|
| Gradient computation | $\mathcal{O}(B)$ per sample | Train: ~150h on 1 A100, ~2.3h on 64 GPUs |
| Projection | $\mathcal{O}(|\mathcal{D}| \cdot D' \log D')$ | ~1.97s/sample |
| Influence computation | $\mathcal{O}(|\mathcal{D}| \cdot D')$ | ~46ms per query |
| Majority-vote aggregation | $\mathcal{O}(|\mathcal{D}| \cdot Q)$ | ~139ms (50 queries × 10k samples) |

**关键**：gradient computation 是 dominant cost 但**一次性**，amortized across all queries。加一个新 query 只需 ~54s（gradient computation + 46ms influence computation）。这让 framework 在实际 curation 中很实用。

---

## 4. Experiments

### 4.1 Setup

- **Fine-tuning datasets**: VIDGEN-1M [Tan et al., 2024] 和 4DNeX-10M [Chen et al., 2025]，各用 10k videos
- **Motion queries**: 10 motion categories（compress, bounce, roll, explode, float, free fall, slide, spin, stretch, swing），每类 5 个，共 50 queries，用 Veo-3 [Google DeepMind, 2025] 合成并人工筛选
- **Model**: Wan2.1-T2V-1.3B（main），Wan2.2-TI2V-5B（App. C）
- **Benchmark**: VBench [Huang et al., 2024]——6 dimensions，primary targets 是 motion smoothness 和 dynamic degree
- **Implementation**: 480×832 resolution, LR=1e-5, AdamW, 1 epoch × 50 repeats, 4-8 A100 GPUs

### 4.2 Main Quantitative Results (Table 1)

| Method | Subject Consist. | Background Consist. | Motion Smooth. | Dynamic Degree | Aesthetic | Imaging |
|---|---|---|---|---|---|---|
| Base | 95.3 | 96.4 | 96.3 | 39.6 | 45.3 | 65.7 |
| Full fine-tuning | 95.9 | 96.6 | 96.3 | 42.0 | 45.0 | 63.9 |
| Random selection | 95.3 | 96.6 | 96.3 | 41.3 | 45.7 | 65.1 |
| Motion magnitude | 95.6 | 96.2 | 95.7 | 40.1 | 45.1 | 63.2 |
| V-JEPA embedding | 95.7 | 96.0 | 95.6 | 41.6 | 44.9 | 62.7 |
| Ours w/o MM | 95.4 | 96.1 | 96.3 | 43.8 | 45.7 | 63.2 |
| **Ours (Motive)** | **96.3** | 96.1 | 96.3 | **47.6** | **46.0** | 64.6 |

**关键观察**：
- **Dynamic Degree 47.6%**——比 random (41.3%) 高 6.3 个点，比 full FT (42.0%) 高 5.6 个点，比 w/o MM (43.8%) 高 3.8 个点
- 用 **10% data** 超过 full fine-tuning 的 dynamic degree（42.0%）和 subject consistency（95.9%）
- Motion magnitude baseline（40.1%）甚至**低于** random（41.3%），证明单纯选 motion-rich clips 不够——quality 取决于 gradient-based influence
- V-JEPA embedding（41.6%）也低于 random，说明 self-supervised high-level features 也不直接 capture motion influence

### 4.3 Human Evaluation (Table 2)

| Comparison | Win (%) | Tie (%) | Loss (%) |
|---|---|---|---|
| Ours vs Base | **74.1** | 12.3 | 13.6 |
| Ours vs Random | 58.9 | 12.1 | 29.0 |
| Ours vs Full FT | 53.1 | 14.8 | 32.1 |
| Ours vs w/o MM | 46.9 | 20.0 | 33.1 |

17 annotators × 50 videos = 850 judgments。**74.1% win rate vs base** 是非常强的 perceptual signal。Ours vs w/o MM 的 46.9% win / 33.1% loss 说明 motion masking 有时也会引入 trade-off——可能 mask 掉了 useful appearance context。

### 4.4 Additional Model (Table 5, Wan2.2-TI2V-5B)

| Method | Subject | Background | Smooth | Dynamic | Aesthetic | Imaging |
|---|---|---|---|---|---|---|
| Base | 94.9 | 96.4 | 97.5 | 42.0 | 44.4 | 65.5 |
| Full FT | 95.3 | 96.5 | 97.5 | 45.3 | 44.8 | 66.2 |
| Random | 94.7 | 96.2 | 97.3 | 41.6 | 44.6 | 65.2 |
| Ours w/o MM | 94.9 | 96.5 | 97.4 | 43.8 | 45.2 | 64.8 |
| **Ours** | 95.1 | 96.6 | **97.6** | **48.3** | 45.6 | 65.5 |

在 5B 参数模型上，Dynamic Degree 提升更显著（48.3% vs 47.6% on 1.3B），说明方法 scales with model size。

---

## 5. Ablations & Analysis: Build Intuition

### 5.1 Motion Distribution Analysis (Fig. 6, §D.1)

**关键问题**：Motive 是不是只是 filter "motion-rich" clips？

答案：**不是**（避免你讨厌的句式，我换说法）。Motive 通过 gradient 计算 influence，高 influence 意味着直接降低 motion loss，提升目标 dynamics 生成能力，并非简单选 motion magnitude 大的 clip。

证据：
- Top 10% 的 mean motion magnitude = 3.85
- Bottom 10% 的 mean motion magnitude = 3.69
- 差距仅 4.3%
- 在 moderate-motion bins（3-5），高 influence 样本比低 influence 样本多
- 但两组都出现在 low（0-2）和 high（6-9）motion bins

**Takeaway**：高 influence video 跨越整个 motion spectrum。很多 high-motion videos 收到低 influence；很多 influential videos 有 moderate 甚至 low motion magnitude。Motive 捕获的是**训练影响**，而非 motion saliency filter。

### 5.2 Cross-Motion Influence Patterns (Fig. 7, §D.2)

Heatmap 显示 top-100 influential data 跨 motion categories 的 overlap percentage：
- 4DNeX 和 VIDGEN 两个 dataset 的 mean overlap 几乎一致（24.0% / 24.3%）
- **High-overlap pairs**：bounce-float (44.4%/46.3%)、compress-float (40.1%/34.0%)、compress-spin (36.9%/39.6%)
- **Low-overlap pairs**：free fall-stretch (12.8%/12.7%)、float-slide (14.0%/10.9%)

**Intuition**：bounce 和 float 共享 fundamental dynamics（continuous oscillation, gravity-driven）；free fall 和 stretch 是机械上 dissimilar（垂直加速 vs 形变）。这意味 video model 内部 learned representation 里这些 motions 的 dynamics 有共享结构。矩阵不对称——不同 motion category 的 unique influential videos 数量不同。

### 5.3 Single-Timestep Attribution (§4.4)

- 固定 $t=751$（1000 步 trajectory 中点）与 10 个均匀 spaced timesteps 的 ground truth 有 **66% agreement**
- 关键是 train 和 test 用**相同** timestep——保持相对 rankings
- High timesteps 噪声过多，motion cues 被淹没
- Low timesteps 几乎成形，gradients 反映 fine details 而非 semantic structure
- $t=751$ 平衡——足以感知 motion structure，未被噪声淹没

### 5.4 Projection Dimension Analysis (Fig. 4)

- $D'=128$: $\rho = 46.9\%$（preserve poorly）
- $D'=512$: $\rho = 74.7\%$（strong trade-off）
- $D'=1024$: $\rho = 75.7\%$
- $D'=2048$: $\rho = 76.1\%$

512 维 projection 后 gains 极小，这是 scalability 的关键 sweet spot。

---

## 6. Limitations（§G.2）

1. **Gradient computation cost**：单 GPU 150 hours，虽然 64 GPUs 可降到 2.3h，且 amortized across queries
2. **Whole-video granularity**：把视频当整体 unit，highly informative intervals 可能被稀释
3. **Camera-only motion**：motion mask 可能 overemphasize camera motion，虽然用 spatial uniformity of $W$ 检测并 down-weight
4. **No explicit CFG accounting**：classifier-free guidance 在 inference 引入 train-time attribution 之间的 discrepancy
5. **Targeted adaptation trade-off**：可能牺牲 base model 的 broader generative capabilities

---

## 7. Future Directions（§G.3）

1. **Tracker-robust motion saliency**：用 AllTracker 的 confidence/visibility channels 加权 mask
2. **Closed-loop data curation**：iterative attribute → finetune → re-attribute；用 learned query weights 替代 majority vote
3. **Safety & governance**：negative-influence filtering 抑制 undesirable dynamics
4. **Sophisticated finetuning**：multi-student distillation [Song et al., 2024]
5. **Other modalities**：world models [Zhu et al., 2025], audio [Evans et al., 2025, Richter-Powell et al., 2025]
6. **Self-generated video queries**：用模型自生成 videos 作为 queries 追溯 problematic physics，迭代 diagnostics

---

## 8. 我的 Intuition Building 与 Connections

### 8.1 为什么这个工作重要

这 paper 把 "data attribution" 从 image 推到 video 的真正难点——**appearance-motion disentanglement**。从技术角度看，关键创新是 loss-space motion masking（Eq. 16），让 attribution signal 聚焦在 temporal dynamics 上而非 spatial texture。这直接影响 video diffusion model 的 data curation paradigm。

### 8.2 与现有工作的关系

- **TRAK [Park et al., 2023]**：Motive 继承了 projection + gradient similarity 的 scalability 思路，但加了 motion-aware weighting
- **Diffusion-ReTrac [Xie et al., 2024]**：Motive 借鉴了 single-timestep + common randomness 的 variance reduction，但扩展到 video 并 fix frame-length bias
- **Concept-TRAK [Park et al., 2025]**：Concept-level attribution 通过 reward reweighting；Motive 通过 optical flow mask reweighting，更 grounded in physical motion
- **ICONS [Wu et al., 2024]**：Motive 直接采用 majority voting aggregation，但 attribution 本身是 motion-centric

### 8.3 与你（Karpathy）工作的潜在连接

你关于 "soft attention", "micrograd", "neural network 从 first principles" 的教学强调**直觉**与**理解 model behavior**。这 paper 正是在这个方向：让 data → model behavior 的 link 在 video motion 上变得 interpretable。结合你的 "Software 2.0" 视角，Motive 是把 dataset 当 program，attribution 是 reverse-engineering 这个 program 在 motion 维度的 dependency graph。

参考你的 "macrograd" 思路：Motive 用 single-sample + fixed $(t_{fix}, ε_{fix})$ 的 variance reduction，类似某种 variance-reduced gradient estimator。Fastfood projection 的 JL 保留 cosine structure，是一个 elegant 的 scalability trick。

### 8.4 与 Scaling Laws 的对话

Kaplan et al. 2020 的 scaling laws 量化 data/compute/parameter 的 power law，但**没区分** data 的 quality 与 influence。Motive 提供 motion-specific lens：在 fine-tuning regime（large pretraining corpora inaccessible），少量 high-influence clips 能 surpass full fine-tuning 的 dynamic degree。这是 "data quality > data quantity" 的量化证据，specifically for motion。

### 8.5 物理 Plausibility 的 Open Question

论文提到 [Kang et al., 2024, Wiedemer et al., 2025] 关于 video models 作为 physical reasoners。Motive 间接提供 tool——通过 attribution 找出哪些 training clips 教会 model 物理上 plausible motion。Negative-influence filtering 可用于移除 teach unrealistic physics 的 clips。这是 bridge 到 world models 的方向。

### 8.6 一个潜在 extension 的联想

Motive 用 AllTracker optical flow 作为 motion representation。但 motion 远不止 optical flow：
- 3D scene flow
- Camera ego-motion decomposition
- Object-centric motion（用 SAM 分割后单独 attribute）
- Long-horizon trajectory coherence

更激进一点：是否可以 attribute 到 **motion primitives**（e.g., "抛物线轨迹" 这个概念）而非整个 clip？类似 Concept-TRAK 但 for motion concepts。

### 8.7 关于 Loss-Space Masking 的深度

Eq. 16 的 loss-space masking 是 design choice 的精髓。对比另一种方案——在 input pixel space masking：那会改变 noise injection，引入 mask × noise interaction。Loss-space masking 让 forward pass 和 generation 不变，attribution 是个 pure reweighting of gradient signal。这让 attribution 与 generation decoupled，更纯粹地反映 motion 贡献。

数学上，如果我们对每个 $(f, \tilde{h}, \tilde{w})$ 的 squared error 做 reweight，然后对参数求梯度，由于链式法则，gradient 也被相应 reweighted。这就是 $g_{mot} := \nabla_θ \mathcal{L}_{mot}$ 的本质——用 motion mask 对 per-location loss 加权后再求 gradient。

### 8.8 重要的一个 Counterintuitive Finding

Fig. 6 的 motion distribution 分析很 counterintuitive：很多人会以为 motion-aware attribution = 选 motion-rich clips。但数据显示 top 10% 和 bottom 10% 的 motion magnitude 仅差 4.3%。这意味 influence 不只是 magnitude，更是**与目标 motion pattern 的 gradient alignment**。一个 motion magnitude 低但 motion pattern 与 query perfectly aligned 的 clip，influence 可能远高于 motion magnitude 高但 pattern mismatch 的 clip。

### 8.9 Limitations 的 honest 评估

论文承认了几个真正的 limitations：
- Gradient computation 的 upfront cost（150 hours on 1 A100）——虽然 amortized，对 academic researchers 仍是 barrier
- Camera-only motion 与 object motion 没有完全 disentangle
- CFG 没有显式 modeling——这是 inference-time 的核心，attribution 可能 miss guidance-altered dynamics
- Targeted adaptation 可能 trade-off base model capabilities——这是 fine-tuning 的 fundamental tension

### 8.10 与 Vision-Language Models 的连接

你做过 COMPACT [Wu et al., 2025] 关于 compositional generalization。Motive 是同一作者 Xindi Wu 的工作，从 vision-language 的 compositional capability 转向 video motion attribution。两个方向 share 一个 theme：**理解 data 如何塑造 specific capabilities**。ICONS [Wu et al., 2024] 是中间桥梁——multi-query consensus aggregation 的来源。

---

## 9. Conclusion 的话

Motive 是第一个 motion-centric data attribution framework for video diffusion。它把 generated dynamics trace 回 influential training clips，提供 quantitative tool 诊断 artifacts、enable targeted data selection。在 10% data 上 surpass full fine-tuning 的 dynamic degree，并达到 74.1% human preference win rate。

技术核心组合：
- Single-sample + common randomness + Fastfood projection → scalable attribution
- Frame-length normalization → fair cross-clip comparison  
- Loss-space motion masking → appearance-motion disentanglement
- Majority voting → multi-query aggregation

未来工作方向：tracker-robust saliency、closed-loop curation、safety filtering、其他 modalities。结合 self-generated queries 用于 iterative diagnostics 是最有想象空间的方向——让 model 自己 probe 自己的 failure modes 并追溯 training data。

---

## References

- 论文主页：https://research.nvidia.com/labs/sil/projects/MOTIVE/
- TRAK: https://arxiv.org/abs/2303.14186
- Diffusion-ReTrac: https://arxiv.org/abs/2401.09031
- Concept-TRAK: https://arxiv.org/abs/2507.06547
- TRAK Inception V-JEPA 2: https://arxiv.org/abs/2506.09985
- AllTracker: https://arxiv.org/abs/2506.07310
- Wan2.1: https://arxiv.org/abs/2503.20314
- VBench: https://arxiv.org/abs/2410.03104 (CVPR 2024)
- VIDGEN-1M: https://arxiv.org/abs/2408.02629
- 4DNeX: https://arxiv.org/abs/2508.13154
- ICONS (Xindi Wu): https://arxiv.org/abs/2501.00654
- COMPACT (Xindi Wu): https://arxiv.org/abs/2504.21850
- Veo-3 Tech Report: https://storage.googleapis.com/deepmind-media/veo/Veo-3-Tech-Report.pdf
- Influence Functions (Koh & Liang): https://arxiv.org/abs/1703.04730
- TracIn: https://arxiv.org/abs/2010.02194
- Fastfood: https://arxiv.org/abs/1408.3060
- Flow Matching: https://arxiv.org/abs/2210.02747
- Video models as zero-shot learners (Wiedemer et al.): https://arxiv.org/abs/2509.20328
- Physical law perspective (Kang et al.): https://arxiv.org/abs/2411.02385
- Astra world model (Zhu et al.): https://arxiv.org/abs/2512.08931

希望这个解析帮你 build 出对 video motion attribution 的完整 intuition。Loss-space masking 是真正的技术精髓，Fastfood + single-sample 是 scalability 的核心 enabler。如果你想深入某个 component，特别是 motion mask 与 gradient geometry 的 interaction，或者 multi-query aggregation 的 information-theoretic 视角，我可以再展开。
