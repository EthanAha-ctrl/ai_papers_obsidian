---
source_pdf: Do Video Foundation Models Understand Intuitive Physics.pdf
paper_sha256: 6541d93cf7bf1c84e5ff402b31b52e1aff8af27d97ae7101cdf6306a804c8a20
processed_at: '2026-08-18T06:37:12-07:00'
target_folder: Physics
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 Paper

## 这帮人到底在问什么

很简单一句话：**现在的 video model 真的"懂"物理吗？**

你看 V-JEPA、VideoMAE 这些模型，benchmark 分数刷得很高。但高分不代表真懂，可能只是学会了 visual shortcut —— 比如看到球就猜"会弹"，看到人就猜"会走"，根本没在 reasoning physical dynamics。

这帮人说：别 fine-tune，别改模型，直接把 pretrained model 冻住，往中间 layer 插一个小 probe（就一个小 classifier），看能不能读出 physics 信号。如果能读出，说明 pretraining 本身就把物理知识 encode 进去了。如果读不出，那 benchmark 上的高分就是 fake news。

## 三个选手，三种训练哲学

他们选了三个 model family，代表三种完全不同的 pretraining 思路：

**V-JEPA（LeCun 阵营）**：不重建 pixel，在 latent space 预测未来。给它看视频前半段，让它预测后半段的 representation。philosophy 是"我不关心 pixel 长什么样，我关心 world state 怎么 evolve"。这本质上就是在 train world model。

**VideoMAE**：mask 掉 90% 的 spatio-temporal patch，让 model 重建 missing pixel。philosophy 是"如果你能 fill in 被遮挡的部分，说明你理解了 object continuity 和 temporal structure"。但问题在于 pixel-level loss 容易被 low-level texture shortcut 满足。

**LTX-Video**：diffusion model，train 来生成 video 的，根本不为 representation learning 设计。但他们好奇：denoising 过程中间会不会顺带 encode 一些 physics structure？

## 两个 Benchmark 的设计哲学

**IntPhys2**：用合成视频测四条物理原则 —— object permanence（遮挡后还在）、immutability（属性不变）、spatio-temporal continuity（不瞬移）、solidity（不穿透）。每个 scene 有 4 个 clip（2 possible + 2 impossible），metric 要求所有 possible clip 的 score 都高于所有 impossible clip。极其严苛，错一个就整个 scene 判错。random baseline 是 16.67%。

**MVP**：更狠。构造 minimal video pair —— 两个视频长得几乎一样，问同一个问题，但答案相反。比如"球会穿过墙吗"，pair A 答案是 yes（球穿墙了，impossible），pair B 答案是 no（球弹回来了，possible）。你必须两个都对才算分。这彻底堵死了 visual shortcut —— 你不能靠"看到墙"就猜答案，因为两个 video 都有墙。

## 核心发现，一个一个说

### 发现 1：V-JEPA 完胜

在 temporal attentive probe 下：
- IntPhys2：V-JEPA family 56-67% VOE，VideoMAE 59%（v2 只 16%，疑似欠训练），LTX 47%
- MVP：V-JEPA 94%，VideoMAE 92%，LTX 84%

V-JEPA family 几乎全面领先。直觉解释：latent space prediction 本质上逼 model 学习 "如果物体这样运动，下一步 state 会怎样"，这就是 physical reasoning。pixel reconstruction 可以靠 texture pattern matching 偷懒，diffusion generation 优化的是 visual quality 不是 understanding。

但注意 confound：V-JEPA 2.1 用 ViT-Gigantic（48 层），比 VideoMAE-v2 大。所以领先可能 partly 来自 scale。paper 诚实承认这点。

### 发现 2：物理信息在中间偏深层最丰富

把每个 model 按 depth 切成 25%、50%、75%、100% 四个点，分别 probe：

**MVP**：越深越好，late layer 最强。V-JEPA 从 0.25 depth 的 59% 涨到 final layer 的 87%。直觉：minimal pair 要求 fine-grained physical discrimination，这种 abstract 区分只在深层 representation 里才 accessible。

**IntPhys2**：中间偏深最 rich，但 non-monotonic。很多 model 在 75% depth 达到 peak，然后 final layer 反而掉。直觉：IntPhys2 的 violation 类型（比如 object 瞬移）在中间层被 encode 为 object-level structure，但 final layer 可能把信息 compress 成 abstract semantic（"这是球"），反而丢了 physical plausibility 的细节。

**LTX 特殊**：diffusion model 多一个维度 —— denoising noise level。他们发现中间 noise level（0.4-0.7）的 representation 最 informative。直觉：高噪声时几乎纯噪声没 signal；低噪声时 representation 已 collapse 到具体 pixel；中间阶段 model 在做最 abstract 的 structure prediction，这时候 physics skeleton 最 exposed。

### 发现 3：信息 encoding 深度因 benchmark 而异

用三种 probe 测，从弱到强：linear → MLP → temporal attentive。

**MVP**：linear probe 几乎完全失败（pair consistency 37-50%，接近 chance）。MLP 大幅恢复（64-87%）。Temporal attentive 进一步提升（84-94%）。说明 MVP 的 physics signal 在 representation 中是 high-dimensional nonlinear encoding，必须用能 model temporal interaction 的 probe 才能解开。

**IntPhys2**：linear probe 已经能恢复不少 signal（35-51% VOE）。说明部分 physics-violation cue 在 V-JEPA representation 中被 organize 成 explicit linear direction。但 temporal attentive 仍 often 最强。

直觉：MVP 的 minimal pair 设计强迫 fine-grained 对偶区分，representation 中是 nonlinear boundary。IntPhys2 的 possible/impossible 是 broader distinction，更容易被 linear hyperplane 分开。

### 发现 4：最 revealing 的 temporal control

这是 paper 最 insightful 的部分。他们做了两个 control：

**Frame-shuffled**：把视频帧顺序随机打乱。破坏 causal order，保留所有 static visual appearance。

**Single-frame**：随机取一帧重复 16 次。彻底消灭 temporal variation。

**MVP 上结果**：两个 control 都让性能崩溃。Single-frame 掉 70-100%，shuffle 掉 39-61%。LTX 在 attentive probe 下 shuffle 掉 96%。这说明 MVP 上的 90%+ 分数是真实的，确实依赖 multi-frame ordered temporal evidence，不是 visual shortcut。

**IntPhys2 上结果**：surprising。Single-frame 总是大幅掉分（38-100%），说明确实需要 multi-frame evidence。但 frame-shuffled 在某些 config 下几乎不掉分！比如 V-JEPA temporal attentive 在 shuffle 下 Δ=0.00% —— 打乱帧顺序后表现完全不变。

这意味着什么？在那个 config 下，probe 其实没在 reasoning causal physics trajectory，而是在 exploit unordered multi-frame bag of features。比如"frame 1 有球 + frame 5 没球"这种 presence/absence pattern，不需要 frame order 就能判断 object permanence violation。

这是个重要的 validity check：**probing paper 没有 temporal control 就不可信**。Shuffle Δ=0% 是 red flag，意味着 probe 在走 dataset shortcut。

## 串起来的 Intuition

如果我要把整篇 paper 压缩成几个 mental model：

**Pretraining objective 决定 representation 的 abstractness 层级**。Latent prediction (V-JEPA) 最 abstract，最 world-model-like。Pixel reconstruction (VideoMAE) 中等。Generation (LTX) 中间 denoising stage 有 abstract structure，但 final stage collapse 到 pixel。

**Physics 信息在网络中的分布是 anvil-shaped**。Early layer 太 low-level（edge, color, texture），middle-to-late 最 rich（object structure, event dynamics），final layer 有时反而不行（collapse 到 semantic category）。

**Probe expressivity 是 information depth 的 diagnostic tool**。Linear decodable = 浅层 explicit encoding。需要 MLP = nonlinear encoding。需要 temporal attentive = 跨 token temporal pattern encoding。不同 benchmark 的 physics signal encoding depth 不同。

**Benchmark 设计决定 measurement validity**。MVP 的 minimal pair 设计抗 shortcut，测出来的 signal 是真的。IntPhys2 的 VOE metric 严苛但允许 unordered multi-frame shortcut，需要 temporal control 才能区分真假 reasoning。

## 对你的可能 relevance

你一直在关注 world model 和 video understanding。这篇 paper 给几个 hint：

- V-JEPA 在 latent space 预测胜过 pixel space，这对 autonomous driving world model 设计有直接指导意义 —— predict in latent representation space
- Middle layer 的 physics signal 比 final layer 更 accessible，fine-tune downstream task 时考虑从中间层取 feature
- Temporal attentive probe 在 token sequence 上 work better，暗示 video LLM 设计中保留 token-level temporal attention 的重要性
- Diffusion model 中间 denoising step 的 representation 最 informative，对 diffusion + understanding joint training 有 design hint

核心 message：**pretrained video model 确实 encode 了一些 physics knowledge，但这个 knowledge 的 accessibility 强烈依赖于 pretraining paradigm、layer depth、readout mechanism 三者交互**。V-JEPA family 在当前 generation 的 large video model 中，physics-relevant feature 最 accessible。

---

# Do Video Foundation Models Understand Intuitive Physics? — 深度解析

## 1. Paper 核心问题与动机

这篇 paper 想回答一个非常 fundamental 的问题：pretrained video foundation models 在 frozen representations 中，是否真的编码了 intuitive-physics 信息，或者只是通过 superficial visual statistics 和 benchmark-specific shortcuts 伪装成"懂物理"。和之前 Garrido et al. (2025) [https://arxiv.org/abs/2502.11831] 和 Joseph et al. (2026) [https://arxiv.org/abs/2602.07050] 的相关工作相比，这篇 paper 的核心贡献是 **在同一种 frozen-feature probing protocol 下横向对比三个 pretraining paradigms**，同时做 layerwise 分析、probe expressivity 分析和 temporal controls，以 disentangle 几个 confounding factor。

四个 Research Questions：
1. 不同 pretraining objective 编码的物理信息 accessibility 是否相等？
2. 在网络哪个 depth 上 physics-relevant information 最 accessible？
3. 这些信息是 linearly decodable 还是需要更强 probe 才能恢复？
4. 性能是否真的依赖 temporal dynamics，还是可以用 static appearance 蒙混？

## 2. 三种 Pretraining Paradigms 的 Objective 公式

### 2.1 V-JEPA (Predictive Joint-Embedding)

V-JEPA [https://arxiv.org/abs/2404.08471] 基于 LeCun 的 JEPA 思想 [https://arxiv.org/abs/2301.08243]，核心是在 latent representation space 做预测，避开 pixel-level reconstruction：

$$\mathcal{L}_{\text{V-JEPA}} = \frac{1}{|\mathcal{T}|} \sum_{i \in \mathcal{T}} \left\| \bar{\phi}(x_i^{\text{target}}) - \phi_\theta(x_i^{\text{target}}; x^{\text{context}}) \right\)_2^2$$

变量含义：
- $x^{\text{context}}$：context video tube，即未被 mask 的 spatio-temporal region
- $x_i^{\text{target}}$：第 $i$ 个被 mask 的 target tube
- $\bar{\phi}(\cdot)$：EMA target encoder 的输出，stop-gradient，作为 prediction target
- $\phi_\theta(\cdot)$：context encoder + predictor 在 context 上预测 target 表示
- $\mathcal{T}$：target tube 集合
- $\|\cdot\|_2^2$：squared L2 范数

关键 insight：因为 loss 在 abstract representation space 而非 pixel space，模型不会被惩罚微小 pixel noise，所以有动力去 encode 高层 structure（运动方向、object permanence、spatio-temporal continuity）。

V-JEPA 2 [https://arxiv.org/abs/2506.09985] 和 V-JEPA 2.1 [https://arxiv.org/abs/2603.14482] 在 scaling 上做更大，并且强化 dense temporally-grounded feature。

### 2.2 VideoMAE (Masked Reconstruction)

VideoMAE [https://arxiv.org/abs/2203.12602] 和 VideoMAE-v2 [https://arxiv.org/abs/2303.16727] 使用 pixel-level masked reconstruction：

$$\mathcal{L}_{\text{VideoMAE}} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \left\| x_i - \hat{x}_i \right\|_2^2$$

变量含义：
- $\mathcal{M}$：masked spatio-temporal patch 集合
- $x_i$：第 $i$ 个 masked patch 的原始 pixel
- $\hat{x}_i$：decoder 重建的 pixel
- $|\mathcal{M}|$：masked patch 数量

VideoMAE 用 extreme masking ratio (≈90% tube masking)，强迫模型用 long-range temporal context 推理 missing content。理论上这能强迫 model 学习 object continuity 和 occlusion reasoning，但 pixel-level loss 也容易被 low-level texture shortcut 满足。

### 2.3 LTX-Video (Diffusion-based Generation)

LTX-Video [https://arxiv.org/abs/2501.00103] 是 video latent diffusion，训练目标是 standard DDPM-style noise prediction：

$$\mathcal{L}_{\text{diffusion}} = \mathbb{E}_{t, x_0, \epsilon}\left[ \left\| \epsilon - \epsilon_\theta(x_t, t, c) \right\|_2^2 \right]$$

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

变量含义：
- $x_0$：clean video latent
- $x_t$：在时间步 $t$ 加噪后的 latent
- $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$：cumulative noise schedule
- $\epsilon$：标准高斯噪声
- $\epsilon_\theta(x_t, t, c)$：neural network 预测的噪声，$c$ 是 text conditioning
- $t$：diffusion 时间步，$t \in [0, T]$

LTX 的特殊性：它的目标函数优化 generation quality 而非 representation quality，但 paper 想测试 denoising trajectory 中间是否隐含 physics-relevant structure。

## 3. 两个 Benchmark 的核心 Metric 公式

### 3.1 IntPhys2 — Violation of Expectation (VOE)

IntPhys2 [Bordes et al. 2025] 围绕四条 intuitive physics 原则构造：
- **Permanence**：物体被遮挡后仍存在
- **Immutability**：物体属性（颜色、形状）不无故改变
- **Spatio-temporal continuity**：物体不瞬移
- **Solidity**：两个物体不互相穿透

每个 scene 是一个 quadruplet：2 个 possible clip + 2 个 impossible clip。

VOE accuracy 的严格定义：

$$\text{VOE(scene)} = \mathbb{1}\left[\min_{i \in \mathcal{P}} s_i > \max_{j \in \mathcal{I}} s_j\right]$$

变量含义：
- $\mathcal{P}$：scene 中 possible clip 的 index 集合（2 个）
- $\mathcal{I}$：impossible clip 的 index 集合（2 个）
- $s_i$：probe 输出的 scalar plausibility score for clip $i$
- $\mathbb{1}[\cdot]$：indicator function，满足条件返回 1

这个 metric 非常严苛：即使 4 个 clip 分类全对，只要一个 possible clip 的 score 低于某个 impossible clip，整个 scene 就被判错。所以 paper 报告的 IntPhys2 数字普遍低 (15.69%–66.67%)，random baseline 是 16.67%（连续 4 个 score 排序中 all-possible 排在 all-impossible 之前的概率）。

### 3.2 MVP — Pair Consistency

MVP [Krojer et al. 2025] 的核心设计是 **minimal video pairs**：两个视频视觉上几乎相同，配对同一个 question，但答案是相反的（yes/no 对偶）。这构造了一个 anti-shortcut constraint：

$$\text{Pair-consistency} = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} \mathbb{1}\left[\hat{y}_q^{(a)} = y_q^{(a)} \land \hat{y}_q^{(b)} = y_q^{(b)}\right]$$

变量含义：
- $\mathcal{Q}$：minimal pair 的集合
- $y_q^{(a)}, y_q^{(b)}$：pair $q$ 中两个 video 的 ground-truth answer（互为反义）
- $\hat{y}_q^{(a)}, \hat{y}_q^{(b)}$：model 的 binary 预测

Pair-level 划分（论文 Section 4.2）保证 training/validation/test 中 minimal pair 不会跨 split 泄漏。

因为 backbone 没有 language alignment，paper 把 MVP 的 text-conditioned QA 转换为 binary plausibility 判别任务（Figure 1）：每个 sample 被映射为 plausible/implausible 标签，pair-consistency 在 binary 预测的层面计算。这种 adaptation 让 frozen probing 不引入 vision-language alignment 训练阶段，保持评测纯净。

## 4. Probing Methodology 详解

### 4.1 三种 Probe 的架构与公式

**Linear probe** (Alain & Bengio 2018 [https://arxiv.org/abs/1610.01644])：

$$\hat{y} = \sigma(W \cdot \text{Pool}(H_\ell) + b)$$

变量含义：
- $H_\ell \in \mathbb{R}^{N_t \times N_s \times d}$：layer $\ell$ 输出的 token embedding 序列，$N_t$ temporal tokens，$N_s$ spatial tokens，$d$ embedding dim
- $\text{Pool}(\cdot)$：spatio-temporal average pooling，输出 $\mathbb{R}^d$ 向量
- $W \in \mathbb{R}^{1 \times d}$, $b \in \mathbb{R}$：可训练参数
- $\sigma$：sigmoid

线性 probe 测试的是 "信息是否 explicitly linearly decodable"。

**MLP probe**：

$$h_1 = \text{GeLU}(\text{LayerNorm}(W_1 \cdot \text{Pool}(H_\ell) + b_1))$$
$$\hat{y} = \sigma(W_2 \cdot h_1 + b_2)$$

可以扩展到多层 hidden layers（Optuna search 中最大到 [1024, 512, 1024]）。MLP 测试的是 "信息 present but not linearly decodable"。

**Temporal Attentive probe**：

这个 probe 不在 pooled 表示上工作，直接在 token sequence 上操作：

$$\text{SelfAttn}(H_\ell) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

其中 $Q = H_\ell W_Q$, $K = H_\ell W_K$, $V = H_\ell W_V$，$d_k = d / n_{\text{heads}} = d / 16$。

然后通过 cross-attention 做 classification：

$$\text{CrossAttn}(q_{\text{cls}}, H'_\ell) = \text{softmax}\left(\frac{q_{\text{cls}} K'^T}{\sqrt{d_k}}\right) V'$$

变量含义：
- $H'_\ell$：self-attention 输出的 token sequence
- $q_{\text{cls}} \in \mathbb{R}^d$：learnable classification query
- $K', V'$：从 $H'_\ell$ 投影得到
- 输出 $\hat{y} = \sigma(\text{Linear}(\text{CrossAttn}))$

这种 probe 可以 explicit model temporal interaction，是测试 "physics info 是否依赖 token-level temporal structure" 的关键。

### 4.2 Hyperparameter Search

Optuna (TPE sampler + median pruner)：
- Learning rate: $\log\mathcal{U}(10^{-5}, 10^{-2})$
- Weight decay: $\log\mathcal{U}(10^{-8}, 10^{-2})$
- Batch size: $\{32, 64, 128, 256\}$
- Epochs: $\{20, 50, 100, 500, 1000, 2000\}$
- 每层独立 20-trial study
- Temporal attentive probe 因 memory 限制只 grid search learning rate，batch size = 1

## 5. Temporal Control Conditions 与其物理含义

两个 control 都在 input 层面操作，backbone 和 probe 都 frozen：

**Frame-shuffled control**：

$$H_\ell^{\text{shuffled}} = f_\theta(\text{Permute}_\pi(X))$$

其中 $\pi$ 是 random permutation over frame indices。这保留了所有静态 visual appearance，但破坏了 causal event structure 和 motion trajectory。如果性能不掉，说明 model 其实没用 temporal order，只用 frame bag 的 visual statistics。

**Single-frame control**：

$$X^{\text{single}} = [x_{t_0}; x_{t_0}; \dots; x_{t_0}]$$

随机采样一个 frame 重复 16 次。这彻底消灭 temporal variation，但保持 input format 不变。这是 "static appearance upper bound"。

控制指标计算：

$$\Delta_\% = 100 \times \frac{\text{Score}_{\text{control}} - \text{Score}_{\text{main}}}{\text{Score}_{\text{main}}}$$

负值越大表示越依赖 temporal dynamics。

## 6. 实验结果深度解读

### 6.1 Analysis 1: Best-case Benchmark Comparison

| Model | IntPhys2 VOE (Temp. Attn.) | MVP Pair (Temp. Attn.) |
|---|---|---|
| V-JEPA | 56.86 | **94.03** |
| V-JEPA 2 | 58.82 | 93.33 |
| V-JEPA 2.1 | **66.67** | 93.73 |
| VideoMAE | 58.82 | 92.01 |
| VideoMAE-v2 | 15.69 ⚠ | 91.10 |
| LTX-Video | 47.06 | 84.33 |

几个关键观察：

1. **V-JEPA family 在两个 benchmark 上都领先**。这强烈支持 latent-space predictive learning 比 pixel reconstruction 更适合 encode abstract physical structure。原因：V-JEPA 的 loss function 强迫 model 在 representation space 预测 future state，这本质上就是 "world model" 训练；pixel reconstruction loss 可以被 low-level texture shortcut 满足。

2. **VideoMAE-v2 在 IntPhys2 上的 15.69% 是异常值**，接近 random (16.67%)。Appendix A.6 解释为 undertraining：extending 60 epochs + patience 30 后能恢复到 31.37%。这提醒我们：在 layer-wise probing 中，small dataset (IntPhys2 只有 604 training clips) 配合 large backbone 容易欠拟合。

3. **LTX-Video 落后但非 trivial**（MVP 84.33%）。Diffusion model 不为 representation learning 优化，但 denoising trajectory 中间阶段确实 encode 一些 physics-relevant 信息，paper 后续专门分析。

### 6.2 Analysis 2: Layerwise 深度分析 (MLP probe)

Figure 2 + Figure 6 的 layerwise profile 揭示两个 benchmark 有截然不同的 depth signature：

**MVP — late-layer dominated**：
- V-JEPA: 0.25 depth → 59.45% pair → 1.00 depth → 87.26% (Δ=+27.81)
- V-JEPA 2: 0.25 → 52.07% → 1.00 → 86.75% (Δ=+34.68)
- V-JEPA 2.1: 0.25 → ? → 0.75 → 70.78% (peak) → 1.00 → 68.45%
- VideoMAE: 0.25 → 50.46% → 0.75 → 64.41% (peak) → 1.00 → 62.59%

**IntPhys2 — intermediate-to-late, non-monotonic**：
- V-JEPA: 0.25 → 15.69% → 0.50 → 45.10% → plateau
- V-JEPA 2: peak at 0.75 (56.86%) → drop to 47.06% at 1.00
- V-JEPA 2.1: peak at 0.75 (39.22%)
- VideoMAE: peak at 0.75 (35.29%)
- VideoMAE-v2: peak at 0.50 (47.06%)

为什么 MVP 和 IntPhys2 depth signature 不同？

我的 intuition：MVP 是 minimal pair，要求 model 对 subtle physics difference 做 discriminating decision。这种 abstract discrimination 在 late layer 的 abstract representation 中最强；early layer 偏 low-level feature（color, texture, edge），无法支持 fine-grained physical reasoning。

IntPhys2 的 VOE 要求 4 个 clip 之间的 ranking。一些 possible/impossible 区别（比如 object occlusion 时的 continuity violation）可能在中层 representation 中就被明确编码（因为中层处理 object-level structure），但 late layer 可能 compress 这些细节到 abstract semantic（"这是一个 ball"），反而损失 physical plausibility 信号。这是为什么 IntPhys2 在 late layer 反而 sometimes drop。

### 6.3 LTX-Video Denoising Trajectory 的特殊分析

Figure 7 + 8 显示 LTX 在 (noise_level × block_index) 二维 grid 上的 performance。MVP 上的 MLP probe 最佳 cell 是 (block=24, noise=0.7) → 69.2% pair。

噪声水平的影响：
- 高噪声 (0.9-1.0)：x_t 几乎是纯噪声，几乎没有 signal → 0-30% pair
- 中噪声 (0.4-0.7)：denoising process 已经 partially restore 结构，但还未 collapse 到 final generation，此时 physics structure 最 accessible → 60-69%
- 低噪声 (0.1)：denoising 接近完成，representation 变得 image-specific，physics 抽象度下降 → 60%

这给我一个 strong intuition：**diffusion model 的中间 noise level 含有最 rich 的 "physics skeleton"**，因为此时模型在做最抽象的 structure prediction；early noise 太混沌，late noise 已 collapse 到 pixel-level rendering。

### 6.4 Analysis 3: Probe Expressivity 揭示的信息可访问性

MVP 上 probe 容量影响巨大（Appendix Figure 9）：

| Model | Linear | MLP | Temp. Attn. |
|---|---|---|---|
| V-JEPA | 48.74 | 87.26 | 94.03 |
| V-JEPA 2 | 47.32 | 86.75 | 93.33 |
| V-JEPA 2.1 | 43.88 | 70.78 | 93.73 |
| VideoMAE | 37.51 | 64.41 | 92.01 |
| VideoMAE-v2 | 38.62 | 66.13 | 91.10 |
| LTX-Video | 49.95 | 69.16 | 84.33 |

Linear probe 几乎完全 fail（接近 chance 50% for pair-consistency if random）。MLP 大幅恢复，temporal attentive 进一步提升。这说明 MVP 的 physics signal 在 representation 中是 **nonlinearly encoded**，并且 **跨 token temporal structure** 才能解开。

IntPhys2 上 probe 影响更小：
- V-JEPA Linear: 50.98 VOE，MLP: 45.10，Temp.Attn: 56.86
- LTX Linear: 49.02，MLP: 47.06，Temp.Attn: 47.06

IntPhys2 中部分 signal linearly decodable，意味着 V-JEPA encoder 把一些 physics-violation cue 直接 organize 在 representation 的某个 linear direction 上。

Intuition：MVP 的 minimal pair 设计强迫 fine-grained 对偶区分，这种 distinction 在 representation 中是 high-dimensional nonlinear boundary；IntPhys2 的 possible/impossible 是 broader binary distinction，更容易被 linear hyperplane 分开。

### 6.5 Analysis 4: Temporal Controls 的诊断价值

Table 5 是 paper 最 informative 的 table 之一。

**MVP 上 controls 几乎完全崩溃**：
- Single-frame control 让所有 model 下降 70-100%
- LTX-Video temporal attentive 在 single-frame 下掉 96.52% — 完全 collapse

这证明 MVP 上的 90%+ pair-consistency 不是 spurious correlation，而是真实依赖 multi-frame temporal evidence。

**IntPhys2 上 controls 揭示 surprising 现象**：

| Model | Probe | Shuffle Δ | Single Δ |
|---|---|---|---|
| V-JEPA | Linear | -57.69% | -84.62% |
| V-JEPA | Temp.Attn | **0.00%** | -75.86% |
| V-JEPA 2.1 | Linear | **0.00%** | -72.22% |
| VideoMAE-v2 | MLP | -70.83% | -75.00% |
| LTX-Video | Linear | -60.00% | -100.00% |

注意 V-JEPA temporal attentive 在 shuffle 下 Δ=0.00%。这意味着 model 在 frame-shuffled 输入下表现和 main task 一样好！这说明在这个特定 config 下，probe 已经 collapse 到 exploit dataset bias / unordered appearance cue，而非真正的 causal physics reasoning。Paper 诚实地指出这一点，是 probing 文献中常见的 "probe can shortcut" 问题 [Hewitt & Liang 2019, https://arxiv.org/abs/1909.03368]。

而 single-frame control 在 IntPhys2 上总是大幅掉分（最低 -38.10%，多数 -60% 到 -100%），说明 IntPhys2 至少需要 multi-frame evidence，但**不需要 frame order**——只要看到多个 frame 的 bag of features 就足够判断 plausibility。

这个 distinction 非常 deep：
- MVP 需要 **causal temporal trajectory**
- IntPhys2 (有时) 只需要 **multi-frame unordered evidence**

我的 hypothesis：IntPhys2 的 violation 类型（permanence, solidity 等）有些可以通过 "object present in frame 1 + absent in frame 5" 这种 unordered presence/absence pattern 判断，不需要精确 motion。MVP 的 minimal pair 强迫区分微小 motion 差异（比如 "ball goes through wall" vs "ball bounces off wall"），这种 distinction 必须依赖 ordered motion。

## 7. Intuition Building：把所有 finding 串起来

如果我把整篇 paper 的 finding 浓缩成几个 mental model：

1. **Pretraining objective 决定 representation 在 abstract-to-concrete 谱系中的位置**：
   - Latent prediction (V-JEPA) → 鼓励 abstract, world-model-like representation
   - Pixel reconstruction (VideoMAE) → 部分 abstract 但仍保留 pixel-level cue
   - Generation (LTX) → 中间 denoising stage 有 abstract structure，但 final stage collapse 到 pixel

2. **Physics 信息分布是 "anvil-shaped"**：early layer 弱（low-level feature），middle-to-late 最强（abstract object/event structure），late layer (有时) 反而弱（collapse 到 semantic classification）。

3. **Probe expressivity 揭示 "信息编码深度"**：
   - Linear decodable = 浅层 abstraction，explicit feature
   - 需 MLP = 高维 nonlinear encoding
   - 需 Temporal attentive = 跨 token 的 temporal interaction pattern

4. **Temporal controls 是 probing 的 validity check**：没有 control 的 probing paper 不可信。Shuffle Δ=0% 是 red flag，意味着 probe 在做 dataset shortcut。

5. **Benchmark 设计哲学影响 measurement sensitivity**：
   - IntPhys2 严苛 VOE metric + 容易 shortcut → 表面 high score 不等于真懂物理
   - MVP minimal pair + pair-consistency → 抗 shortcut 但需要 binary adaptation for frozen probing

## 8. 关键 limitations 与 future work

Paper 诚实承认：
1. **没有 disentangle objective from scale/recency**。V-JEPA 2.1 (ViT-Gigantic, 48 layers) 比 VideoMAE-v2 (ViT-G, 40 layers) 大，且 V-JEPA 2.1 更新。所以 V-JEPA 领先可能 partly 来自 scale 而非 objective。
2. **MVP adaptation 引入 binary plausibility 转换**，丢失 original QA semantics。这可能 understate 某些 model 的真实 capability。
3. **IntPhys2 仅 204 test clip**，small dataset 导致 probe 训练不稳定（VideoMAE-v2 例子）。
4. **没 probe representation 的 specific physical property**（比如 object permanence 单独 vs solidity 单独）。Layerwise breakdown by condition 会让结果更 interpretable。

未来方向：
- 在 matched-scale 模型上重复实验，isolate objective effect
- Probing individual physical principles (permanence/solidity/etc.) 而非 aggregate VOE
- 测试 probe 之外的反事实方法（causal mediation analysis [Joseph et al. 2026, https://arxiv.org/abs/2602.07050]）

## 9. 与更大图景的连接

这篇 paper 在三个 active research line 的交叉点：

1. **World model debate**：V-JEPA 系列被 LeCun 团队明确 framing 为 world model。Paper 提供证据支持 latent-predictive paradigm 在 encode physical structure 上确实胜过 reconstruction/generation，这间接支持 JEPA 的 world-model 假说 [Bardes et al. 2024, https://arxiv.org/abs/2404.08471]。

2. **Probing methodology evolution**：从 Alain & Bengio 2018 的 linear probing 到 attentive probing [Psomas et al. 2026, https://arxiv.org/abs/2506.10178]，paper 展示 probe expressivity 影响结论。Single-probe study 不可信。

3. **Diffusion model 内部 mechanism**：LTX 的 denoising trajectory 分析 align with recent work showing diffusion models encode scene structure in intermediate denoising stages [Xiao et al. 2024, https://arxiv.org/abs/2405.14864; Zhu et al. 2024, https://arxiv.org/abs/2403.12037]。这启发我们：diffusion model 不只是 generator，它的中间 representation 可被 harvest 做 understanding。

## 10. 给你 (Karpathy) 的 takeaway

考虑到你之前在 Tesla 和 OpenAI 的工作关注 world model 和 video understanding，这篇 paper 给你几个可能 useful 的 angle：

- V-JEPA 在 latent space 预测胜过 pixel space 这个结论，对设计 next-gen autonomous driving world model 有指导意义——predict in latent representation space, not in pixel space。
- Layerwise profile 显示物理信息在 intermediate-to-late depth 最 accessible，但 late layer 有时 collapse 到 abstract semantic。这暗示在 fine-tuning 时，从 middle layer 取 feature 可能比 final layer 更适合 physics-aware downstream。
- Temporal attentive probe 在 token sequence 上 work better，对 video LLM 设计有启示：保留 token-level temporal attention 而非过早 pool。
- LTX 的 denoising trajectory 中段最 informative，对 diffusion-based video generation + understanding joint training 有 hint：在中间 denoising step 上 attach understanding head 可能更 efficient。

### Key References

- V-JEPA: https://arxiv.org/abs/2404.08471
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- V-JEPA 2.1: https://arxiv.org/abs/2603.14482
- I-JEPA: https://arxiv.org/abs/2301.08243
- VideoMAE: https://arxiv.org/abs/2203.12602
- VideoMAE-v2: https://arxiv.org/abs/2303.16727
- LTX-Video: https://arxiv.org/abs/2501.00103
- Garrido et al. 2025 (intuitive physics emergence): https://arxiv.org/abs/2502.11831
- Joseph et al. 2026 (physics in video world models): https://arxiv.org/abs/2602.07050
- Probing original (Alain & Bengio): https://arxiv.org/abs/1610.01644
- Control tasks in probing (Hewitt & Liang): https://arxiv.org/abs/1909.03368
- Attentive probing (Psomas et al.): https://arxiv.org/abs/2506.10178
- Diffusion as motion interpreter (Xiao et al.): https://arxiv.org/abs/2405.14864
- Diffusion for RVOS (Zhu et al.): https://arxiv.org/abs/2403.12037
- El Banani et al. 3D awareness probing: CVPR 2024
- Optuna: https://dl.acm.org/doi/10.1145/3292500.3330701
