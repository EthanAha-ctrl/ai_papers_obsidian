---
source_pdf: Joint Reward Modeling Internalizing Chain-of-Thought for Efficient Visual
  Reward Models.pdf
paper_sha256: 10af32800cca099cef00fb20355b57cf13f2ee09ba6d0d6b49f23a2af21a752b
processed_at: '2026-08-05T10:52:02-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Karpathy，如果用最直白的人话来拆解这篇 paper，它其实就是在讲一个 **把 System 2 的思考过程，硬生生压进 System 1 的直觉里** 的故事。

---

### 1. 痛点：现有的 Reward Model 都有残缺

在 RLHF 里，Reward Model (RM) 是给生成结果打分的裁判。对于 image editing 这种复杂任务，裁判得看懂全局语义（比如指令是“把猫换成狗”，背景必须保留）。

现在的主流 RM 有两种，都有致命问题：
- **Discriminative RM** (比如 EditReward, HPSv2): 像个死记硬背的无脑质检员。直接用 pairwise ranking loss 学“图 A 比图 B 好”。速度极快，打分稳，但**没任何推理能力**。为了拟合 loss，它会走捷径，只盯着边缘锐不锐、颜色饱不饱和这种 shallow visual cues，根本不懂跨区域的逻辑一致性。
- **Generative RM** (比如 EditScore, VIEScore): 像个话痨评论家。用 MLLM 生成一大段 Chain-of-Thought (CoT) reasoning text，再从文字里派生出分数。语义理解强，但**太慢且抓不住人类偏好**。因为它优化的 loss 是 next-token cross-entropy，它学会的是“说话通顺”，不是“打分准”。而且 online RL 里每个 step 都要等它写几百个 token，根本跑不起来。

### 2. 核心绝招：Training 时自言自语，Inference 时闭嘴直觉打分

JRM 的想法特别直觉：人类专家一开始做诊断，得显式地一条条对照 checklist 推理（System 2）；熟练之后，扫一眼就能凭直觉判断（System 1）。

JRM 用一个 shared backbone (Qwen3-VL-8B) 加两个 head：
- **Discriminative head $f_\theta$**: 输出 scalar reward score。
- **Language head $g_\phi$**: 输出 structured reasoning text。

训练时，强行让两个 head 共享同一个 hidden state $\mathbf{h} = E(x, c)$，其中 $x$ 是 image，$c$ 是 instruction。Loss function 是：

$$ \mathcal{L}_{\text{total}} = (1-\alpha)\mathcal{L}_{\text{rank}} + \alpha\mathcal{L}_{\text{LM}} $$

这里 $\alpha = 0.7$。重点在于，LM loss 的权重很大，意味着模型必须把大部分精力花在“生成结构化评估理由”上。评估理由里强制要求用 `<|bbox k|>` 标注具体的 edit region，用 `<|global|>` 讨论全局一致性。

但到了 inference 时，直接把 Language head 砍掉，只过 Discriminative head。模型根本不输出任何 token，直接出分。所谓的 **Latent CoT**，就是那段被迫生成的 reasoning text，它的逻辑结构已经被“烤”进了 hidden state 的特征空间里，Discriminative head 直接从这个高维空间里读取 reward。

### 3. 为什么这招管用？对抗 Representation Collapse

要 build intuition，得从 representation space 角度看。传统的 Discriminative RM 为什么蠢？因为 ranking loss 监督信号太弱（只有 A 好于 B），模型为了偷懒，会把 high-dimensional 的 hidden state $\mathbf{h}$ 压缩到一个极低维的子空间里，只保留“好/坏”这一个维度的信息。这就是 **Representation Collapse** (https://aclanthology.org/2019.emnlp-main.415/)。

JRM 加了 LM loss，情况就变了。模型要想生成连贯的 reasoning text（比如指出 region 0 纹理不对，region 1 颜色对了，background 保持得很好），它的 hidden state 就**必须**编码足够多的 semantic factors（区域身份、指令满足度、背景一致性等）。这就像用一根绳子强行把高维空间撑开，不让它塌缩。

Paper 里用 Singular Value Decomposition (SVD) 拿到了铁证。他们提取了 hidden state 矩阵 $H \in \mathbb{R}^{N \times d}$ 做分解 $H = U \Sigma V^\top$。
- Baseline (只有 ranking loss) 的有效秩只有 46.86。
- JRM 的有效秩飙升到 91.77，几乎翻倍。

有效秩的定义是奇异值的熵：$r_{\text{eff}} = \exp(-\sum_i p_i \log p_i)$，其中 $p_i = \sigma_i^2 / \sum_j \sigma_j^2$，$\sigma_i$ 是第 $i$ 个 singular value。秩翻倍意味着 singular value spectrum 变得非常平坦，信息均匀分布在很多维度上。这就是 Latent CoT 成功注入的硬核证据。

### 4. 细节技术点：Uncertainty-aware Ranking Loss

JRM 的 ranking loss 也不是随便写的，它借用了 HPSv3 的 uncertainty-aware ranking。公式如下：

$$ P(x_i \succ x_j \mid c) = \iint \text{sigmoid}(r_i - r_j) \mathcal{N}(r_i|\mu_i, \sigma_i) \mathcal{N}(r_j|\mu_j, \sigma_j) dr_i dr_j $$

这里 $x_i, x_j$ 是两个 candidate editing 结果。
- $r_i, r_j$: 是从 Gaussian 分布里采样的 reward scores。
- $\mu_i, \mu_j$: 是 Discriminative head 预测的 mean reward。
- $\sigma_i, \sigma_j$: 是模型自己学出来的 uncertainty (标准差)。
- $c$: 是 instruction (条件)。

这个积分本质上是把 Bradley-Terry model 扩展到了概率空间。如果数据标注有矛盾，或者图像质量实在太接近，模型就会学一个很大的 $\sigma$，算出来的 preference probability 就会接近 0.5，避免模型在噪声数据上 overfitting。这种做法比死板的 binary classification 要 robust 得多。

### 5. 实验结果与 RL 联想

在 EditReward-Bench 上，JRM 拿了 85.1% 的准确率，GPT-5 只有 75.5%。但 JRM 推理时连一个 token 都不生成，速度跟最基础的 discriminative model 一模一样。

更炸裂的是在 downstream online RL 里的表现。用 Flow-GRPO (https://arxiv.org/abs/2505.05470) 训 OmniGen2 (https://arxiv.org/abs/2506.18871)：
- 用 GPT-4.1 当裁判，GEdit-Bench 提升 +0.45。
- 用 JRM 当裁判，GEdit-Bench 提升 +1.00，直接翻倍。

这说明 JRM 的 latent reasoning 不光 benchmark 分高，作为 reward signal 的 stability 和 accuracy 极其强。Online RL 最怕 reward hacking 和 reward variance，JRM 因为 hidden state 编码了全局逻辑，policy 想通过局部 trick 骗分非常难。

### 6. 给你的联想：System 2 到 System 1 的计算图

这篇 paper 给我的最大启发是关于 **compute allocation** 的。你之前在 Deep Learning 讲义里提过，LLM 的 forward pass 本质是一个固定深度的计算图。Explicit CoT 是通过生成 token 来增加计算图的深度。

JRM 提供了另一种思路：**把深度藏在宽度里**。不需要在时间维度上展开生成 token，而是在单个 forward pass 的 hidden state 的高维空间里，用 LM loss 强行刻画出复杂的逻辑流形。这跟 Implicit CoT distillation (https://arxiv.org/abs/2311.01460) 和 Pause Tokens (https://arxiv.org/abs/2310.02226) 的精神是相通的。

顺着这个思路联想，既然单 token 的 hidden state 能编码这么多 reasoning，那能不能用几个 latent tokens 组成一个隐式的 reasoning chain？比如在 hidden state 里做几步 internal recurrence (类似 latent reasoning: https://arxiv.org/abs/2412.06869)，然后再接 reward head？甚至，能不能用 RL (像 DeepSeek-R1: https://arxiv.org/abs/2501.12948 那样) 在 latent space 里直接优化，让模型自己探索出最佳的 latent reasoning trajectory，完全绕开自然语言的束缚？

JRM 证明了 supervised 的 LM loss 可以把 reasoning 压进 representation，那 RL 在 latent space 的探索可能就是下一个突破口。

---

# JRM 深度讲解：把 CoT 内化进 Discriminative Reward Model 的 hidden state

Karpathy 你好，这篇 paper 的核心 thesis 很有意思，我把它拆成几层来讲，build 你对 "latent CoT via joint training" 的 intuition。

---

## 1. Motivation 层：Reward Modeling 的根本矛盾

Reward model 在 RLHF / RLAIF 里是把 human preference 映射成 scalar reward 的桥梁。对于 image editing 这种 task，reward model 必须捕获 **cross-region semantic consistency** 和 **implicit logical constraints** (e.g. 指令是 "把猫换成狗并让草地变蓝"，必须同时验证替换 + 颜色 + 背景保留)。现有 paradigm 有两条路：

- **Discriminative RM** (e.g. EditReward, HPSv2, PickScore)：用 pairwise / listwise ranking loss 直接拟合 human preference。优点是 latency 低、variance 低、preference alignment 准。缺点是 supervision signal 只有 "A 比 B 好" 这种 binary ranking，representation 容易塌缩到 shallow visual cues (texture, color statistics, edge density)，学不到 global semantic。
- **Generative RM** (e.g. EditScore, VIEScore)：用 MLLM 生成 CoT-style reasoning text，再派生出 score。优点是 semantic understanding 强。缺点是 inference 慢 (要生成几百 tokens)、preference alignment 弱 (cross-entropy 优化的是 next-token coherence 而非相对偏好)、reward 信号 unstable。

JRM 的 insight 是：**reasoning 能力不需要在 inference 时显式输出 token**，关键是 **representation space 里有没有编码足够 semantic structure**。如果训练时强迫 backbone 同时支持 ranking 和 reasoning-text generation，那么 reasoning trajectory 就会被 "压缩" 进 hidden state，discriminative head 可以直接从 latent CoT 中读 reward。

这个思想类比 human expert：先显式学 logical reasoning，然后 internalize 成 fast intuition。System 2 → System 1。

---

## 2. Architecture 层：Shared Backbone + Two Heads

Backbone 用 Qwen3-VL-8B-Instruct。输入是 (image $x$, instruction $c$)，输出 shared representation:

$$\mathbf{h} = E(x, c) \tag{1}$$

这里 $\mathbf{h}$ 是 appended learnable special token 的 final hidden state (类似 CLS token)。

两个 head：

- **Reward head** $f_\theta(\mathbf{h}) \in \mathbb{R}^2$：一个 lightweight MLP，输出 2 维 score (instruction following + visual quality)，用于 ranking。
- **Language head** $g_\phi(\mathbf{h})$：复用 LLM 的 LM head (tied weights)，autoregressive 生成 evaluation text $y$。

关键：训练时两个 head 共享 backbone，inference 时只保留 reward head，language head 完全丢弃。这就是 "training-inference decoupling"。

---

## 3. Loss 层：Uncertainty-aware Ranking + Cross-Entropy

### 3.1 Ranking loss (公式 4-5)

JRM 借鉴 HPSv3 的 **uncertainty-aware ranking**。把 reward score 建模成 Gaussian：

$$r \sim \mathcal{N}(\mu, \sigma)$$

其中 $\mu$ 是 reward head 输出的 mean，$\sigma$ 是 learned uncertainty (annotation noise 越大 $\sigma$ 越大)。Preference probability 是二重积分：

$$P(x_i \succ x_j \mid c) = \iint \text{sigmoid}(r_i - r_j) \, \mathcal{N}(r_i | \mu_i, \sigma_i) \, \mathcal{N}(r_j | \mu_j, \sigma_j) \, dr_i \, dr_j \tag{4}$$

变量解释：
- $x_i, x_j$：两个 candidate editing results (同 source image + instruction)
- $r_i, r_j$：从 Gaussian 采样的 reward scores
- $\mu_i, \mu_j$：reward head 输出的 mean reward
- $\sigma_i, \sigma_j$：learned reward uncertainty (标量，反映标注可信度)
- $c$：editing instruction (条件)

实际计算时这个积分可以用 probit approximation 闭式解 (类似 Bradley-Terry with Gaussian noise)。Ranking loss：

$$\mathcal{L}_{\text{rank}} = -\mathbb{E}_{(i,j)}[\log P(x_i \succ x_j \mid c)] \tag{5}$$

### 3.2 Language modeling loss (公式 6)

标准 next-token cross-entropy，target $y$ 是自动生成的 evaluation reasoning：

$$\mathcal{L}_{\text{LM}} = -\sum_{t=1}^{T} \log p(y_t \mid y_{<t}, x, c) \tag{6}$$

$y_t$ 是 reasoning text 的第 $t$ 个 token，$y_{<t}$ 是 prefix。这个 $y$ 是用 Qwen-VL 在 EditReward 数据上自动生成的 structured JSON，包含 `<|bbox k|>` 区域 tag 和 `<|global|>` 全局评估 tag (见 Appendix A)。

### 3.3 Joint objective (公式 7)

$$\mathcal{L}_{\text{total}} = (1-\alpha)\mathcal{L}_{\text{rank}} + \alpha\mathcal{L}_{\text{LM}} \tag{7}$$

$\alpha = 0.7$ (重 LM 轻 ranking)。这个权重选择很重要——他们消融显示 α 越大性能越好 (Figure 4)，意味着 language supervision 不是 regularization，而是 **主导 inductive bias**。

---

## 4. Latent CoT 的理论解释 (公式 8-9)

关键 insight 在 section 3.3。Joint training 把 reasoning "压缩" 进 $\mathbf{h}$：

$$r = f_\theta(\mathbf{h}_{\text{CoT}}) \tag{8}$$

这里 $\mathbf{h}_{\text{CoT}}$ 表示编码了 implicit reasoning structure 的 hidden state。

为什么这能 work？两个 loss 对 $\mathbf{h}$ 提供互补约束：
- Ranking loss 要求 $\mathbf{h}$ **support stable preference ordering** (线性 separability of preferred vs rejected)
- LM loss 要求 $\mathbf{h}$ **sufficient for generating structured reasoning text** (high information content)

单独 ranking loss 容易让 $\mathbf{h}$ collapse 到低维子空间 (只编码 "好/坏" 一维信号)。LM loss 强迫 $\mathbf{h}$ 编码足够 semantic factors (region identity, instruction satisfaction, background preservation, artifact presence...) 才能生成 reasoning。结果：

$$\text{rank}(\text{Cov}(\mathbf{h})) \uparrow \quad \text{under joint training} \tag{9}$$

即 hidden state covariance matrix 的有效 rank 上升。这直接对抗 representation collapse。

这个 idea 和几个方向有共鸣：
- **Implicit CoT via knowledge distillation** (Deng et al. 2023, https://arxiv.org/abs/2311.01460)：把 explicit CoT 蒸馏成 implicit internal steps
- **Pause tokens / Think before speaking** (Goyal et al. 2023, https://arxiv.org/abs/2310.02226)：用 dummy tokens 给模型更多 compute budget
- **DeepSeek-R1** (https://arxiv.org/abs/2501.12948)：通过 RL 让 reasoning emergent，但仍是 explicit token-level
- **Anisotropy in LLM representations** (Ethayarajh 2019, https://aclanthology.org/2019.emnlp-main.415/)：CLS representation collapse 是已知问题
- **Information bottleneck** 理论：LM loss 作为 auxiliary constraint 防止 reward head over-fitting 到 spurious features

---

## 5. Representation Space Analysis：SVD 证据 (Section 4.3)

这是 paper 最强的 empirical evidence。他们提取 backbone hidden states，做 SVD：

对 feature matrix $H \in \mathbb{R}^{N \times d}$ (N samples, d hidden dim)：

$$H = U \Sigma V^\top$$

Singular value spectrum 的 decay rate 反映信息集中度。Baseline (α=0) 谱衰减快 (信息集中在少数 principal directions)；JRM (α=0.7) 谱平坦 (信息分散在更多 components)。

量化指标 (Table in Figure 6)：

| Metric | Baseline (α=0) | JRM (α=0.7) |
|---|---|---|
| Effective rank | 46.86 | **91.77** |
| Spectral entropy | low | high |
| Isotropy | low | high |

Effective rank 定义 (常见形式)：

$$r_{\text{eff}} = \exp\left(-\sum_i p_i \log p_i\right), \quad p_i = \frac{\sigma_i^2}{\sum_j \sigma_j^2}$$

即 entropy of normalized singular value distribution。JRM 几乎翻倍，证明 representation 没塌缩。

3D PCA plot 也显示 baseline 塌缩成 concentrated cluster，JRM 分散在空间中。这印证 "Latent CoT reshape 了 representation topology"。

---

## 6. Training Data Construction

Preference data 来自 EditReward dataset (https://arxiv.org/abs/2509.26346)。Language supervision 用 Qwen-VL 自动生成 (Appendix A)：

**Template 关键设计**：
- 要求 model 先做 **grounding** (识别 edit regions，输出 bounding box `[ymin, xmin, ymax, xmax]` on 0-1000 scale)
- 再写 **reasoning**，用 special tokens `<|bbox k|>` 引用具体区域，`<|global|>` 讨论全局一致性
- 按 score (1-4) 分档，每档有 mandatory 缺陷描述

Example reasoning output：
```
<|bbox_0|> [Analysis of edited region 0: texture mismatch...]
<|bbox_1|> [Analysis of edited region 1: color correct...]
<|global|> [Background preserved, overall composition coherent...]
```

这个 grounding-tagged text 训练时让 backbone 学到 **spatial-semantic grounding** 的 implicit ability，即使推理时不生成 text，hidden state 也编码了 region-level 结构。Attention visualization (Figure 3) 直接显示 JRM 准确 attend 到 instruction-specified region，baseline 则散乱。

---

## 7. 实验：Benchmark 表现

### 7.1 EditReward-Bench (Table 1)

| Method | Prompt Following | Consistency | Overall |
|---|---|---|---|
| GPT-4.1 | 0.673 | 0.602 | 0.705 |
| GPT-5 | 0.777 | 0.669 | 0.755 |
| Gemini-2.5-Pro | 0.703 | 0.560 | 0.722 |
| EditScore-72B | 0.638 | 0.586 | 0.703 |
| PaCo-Reward-7B | 0.777 | 0.709 | 0.751 |
| Gemini-3.0-Flash | 0.717 | 0.662 | 0.769 |
| EditReward | 0.832 | — | 0.792 |
| **JRM** | **0.854** | — | **0.851** |

JRM 比 GPT-5 高 9.6%，比 EditReward (strong discriminative baseline) 高 5.9%。Inference cost 和 discriminative model 相同 (无 text generation)。

### 7.2 MMRB2 (Table 2, https://arxiv.org/abs/2512.16899)

| Method | Single | Multi | Overall |
|---|---|---|---|
| Qwen3-VL-32B | 0.467 | 0.461 | 0.466 |
| Gemini-2.5-Pro | 0.545 | 0.483 | 0.534 |
| EditScore-8B | 0.579 | 0.528 | 0.570 |
| GPT-5 | 0.627 | 0.584 | 0.619 |
| EditReward | 0.672 | 0.590 | 0.657 |
| **JRM** | **0.703** | **0.646** | **0.693** |

Multi-image 维度提升最大 (+5.6%)，说明 joint training 对 cross-image reasoning 帮助最大。

---

## 8. α 消融 (Section 4.2, Figure 4-5)

α 从 0 → 0.7 性能单调上升。两个 observation：

1. **Ranking loss 收敛更快更稳** 当 α > 0：说明 LM loss 提供 inductive bias 帮助 reward learning，两个 objective 互补不冲突。
2. **Cross-entropy loss 平滑收敛**：说明 backbone 能同时学习两个 task，没有 catastrophic interference。

训练 dynamic 见 Figure 5：α=0 时 ranking loss noisy，α=0.7 时 smooth。

---

## 9. Self-Correction 实验 (Section 4.4, Table 3)

这个实验很 clever。流程：
1. 选 VIEScore 低的 challenging samples
2. 启用 language head 生成 critique
3. 用 critique 作为额外 condition 引导 editing model 修正
4. 关闭 language head，只用 reward head 评估修正前后

结果 (Table 3)：

| VIEScore Threshold | Samples | VIEScore Δ | JRM Δ |
|---|---|---|---|
| < 7.0 | 254 | +0.44 | +0.28 |
| < 5.0 | 169 | +1.23 | +0.28 |
| < 3.0 | 91 | +2.39 | +0.43 |

最难的 samples (VIEScore<3) 修正后 reward head 给 +0.43 提升。这证明 **language head 和 reward head 共享 representation space**——language supervision 注入的 semantic 信息 reward head 能 "读到"。

---

## 10. Downstream Online RL (Section 4.5, Table 4)

用 Flow-GRPO (https://arxiv.org/abs/2505.05470) fine-tune OmniGen2 (https://arxiv.org/abs/2506.18871)。

| Config | GEdit Δ | ImgEdit Δ |
|---|---|---|
| Base | — | — |
| w/ GPT-4.1 | +0.45 | +0.26 |
| w/ EditScore-8B | +0.61 | +0.22 |
| w/ EditReward | +0.77 | +0.19 |
| w/ Baseline (α=0) | +0.82 | +0.23 |
| **w/ JRM** | **+1.00** | **+0.50** |

JRM 在两个 benchmark 都显著领先，特别是 ImgEdit +0.50 vs GPT-4.1 +0.26。Reward curve (Figure 8) 和 GEdit score 强相关，证明 reward signal accuracy 高。

### Flow-GRPO 关键超参 (Appendix B.2)：

- T = 20 discrete timesteps
- σ = 0.9 diffusion coefficient
- Group size G = 12 (每 prompt 采样 12 candidates)
- PPO clip: ε_low = 1e-4, ε_high = 5e-4 (asymmetric clipping 防止 over-optimization)
- KL penalty β = 0.04
- LoRA: r=32, α=64
- 32 GPUs

---

## 11. Training Setup 细节 (Appendix B.1)

- Base: Qwen3-VL-8B-Instruct (https://arxiv.org/abs/2511.21631)
- AdamW (β1=0.9, β2=0.95)
- LR = 2e-6, cosine decay, 5% warmup
- 10 epochs, global batch 64 (per-device 2, grad accum 4, 8 GPUs)
- Max seq len 8192
- Image resolution adaptive (~256×28×28 = 200,704 pixels)
- bf16, gradient checkpointing (non-reentrant), DeepSpeed ZeRO-2
- 40-50 GPU hours total

Special token design：appended learnable token，final hidden state 作为 $\mathbf{h}$。Reward head 输出 2 维。

---

## 12. 与相关工作的 positioning

| Work | Paradigm | CoT | Preference | Inference cost |
|---|---|---|---|---|
| EditReward | Discriminative | None | Direct | Low |
| EditScore | Generative | Explicit | Indirect | High |
| HPSv3 | Discriminative | None | Uncertainty-aware | Low |
| VIEScore | Generative | Explicit | None | High |
| R1-Reward | Generative + RL | Explicit | RL-aligned | High |
| **JRM** | **Joint** | **Latent** | **Direct** | **Low** |

R1-Reward (https://arxiv.org/abs/2505.02835) 是相关工作，用 RL 训 reward model，但仍是 explicit reasoning。JRM 把 reasoning 内化。

Critique-out-loud RM (Ankner et al., https://arxiv.org/abs/2408.09127) 让 RM 生成 critique 同时打分，但推理时仍要生成 text，没有 internalize。JRM 更激进——训练时生成，推理时丢弃。

---

## 13. Intuition 总结：为什么 Joint Training 工作

我从几个 angle 给你 build intuition：

**Angle 1: Information bottleneck 视角**

单 ranking loss 是 weak supervision——只告诉 model "A 好 B 坏"，没说为什么。Model 会找 shortcut (texture sharpness, color saturation)。LM loss 是 dense supervision——每个 token 都提供梯度，强制 $\mathbf{h}$ 编码 region identity, instruction grounding, artifact type 等结构化 factors。这些 factors 恰好也是 ranking 需要的，所以 ranking loss 收益。

**Angle 2: Representation collapse 视角**

Discriminative RM 训练容易让 backbone 退化成 "goodness detector"，hidden state 几乎一维 (https://arxiv.org/abs/1911.02572 是 RLHF reward model collapse 的经典分析)。LM loss 是 anti-collapse regularizer——生成 coherent text 需要 high-dim representation。SVD 分析 (effective rank 46.86 → 91.77) 直接量化这点。

**Angle 3: Distillation without explicit distillation**

传统 knowledge distillation 是 teacher 生成 reasoning，student 模仿 reasoning。JRM 把 reasoning generation 和 reward prediction 放同一 backbone，backbone 学 reasoning 时同时调整 reward pathway。这比 distillation 更紧耦合——shared parameters。

**Angle 4: Human expert analogy**

人学诊断：先显式 reason ( checklist, 症状对照)，熟练后 intuition 化 (一瞥即知)。JRM 是这个过程的 computational analog：training 时 explicit reasoning (LM loss)，inference 时 fast intuition (reward head only)。

**Angle 5: 浅层 vs 深层 feature 的 spectral 证据**

Baseline 的 singular value 谱衰减快说明 backbone 用少数 directions 编码 "goodness"——这是 shallow feature。JRM 谱平坦说明用更多 directions 编码 diverse semantic factors——这是 deep feature。Isotropy 提升 直接对应 Anisotropy 问题 (https://aclanthology.org/2019.emnlp-main.415/)。

---

## 14. 局限与 open questions

Paper 没深入讨论的：

1. **Language supervision 质量**：依赖 Qwen-VL 自动生成，noise 会被 internalize。如果 supervision 有 systematic bias (e.g. 对某些 artifact 类型 blind)，latent CoT 也会有 blind spot。
2. **α=0.7 的理论上界**：为什么 0.7 最优？更高 (e.g. 0.9) 会不会让 ranking signal 被淹没？Figure 4 没显示 α>0.7 的结果。
3. **Latent CoT 的可解释性**：reward head 从 $\mathbf{h}_{\text{CoT}}$ 读 reward，但无法反向 trace reasoning steps。Self-correction 实验靠 language head diagnostic mode，不是真正的 mechanistic interpretability。
4. **Generalization**：只在 image editing 上验证。是否 work 于 text-only RM (e.g. LLM helpfulness RM)？多模态 grounding 可能是关键 inductive bias，纯文本可能没同等效果。
5. **Reward hacking robustness**：online RL 中 reward model 容易被 policy exploit。Paper 没分析 JRM 在 over-optimization 下的 robustness，虽然 Flow-GRPO 用了 asymmetric clipping 和 KL penalty 缓解。
6. **Compute cost**：训练 40-50 GPU hours on 8 GPUs (Qwen3-VL-8B) 还算 reasonable，但 language supervision 数据生成本身有成本 (Qwen-VL inference)。

---

## 15. 对你的启发 (Karpathy 视角)

你之前在 "Deep Learning: System 2 thinking" 系列讨论过 implicit reasoning。JRM 是这个方向在 reward modeling 上的具体实例。几个值得思考的点：

- **Latent CoT vs Pause tokens**：pause tokens (https://arxiv.org/abs/2310.02226) 给 model 更多 compute 但不指定 compute 内容。JRM 通过 LM loss 指定 compute 内容 (reasoning structure)，更 structured。
- **RL on latent reasoning**：DeepSeek-R1 用 RL 让 explicit reasoning emerge。能否用 RL 让 latent reasoning emerge？JRM 是 supervised，可能 RL 版本更强。
- **Multi-token latent reasoning**：JRM 用单 special token 的 hidden state。能否用 multiple latent tokens 编码 longer reasoning chain？类似 latent transformer (https://arxiv.org/abs/2412.06869)。
- **Reward model as world model**：如果 $\mathbf{h}_{\text{CoT}}$ 真的编码 semantic structure，能否用它做更多事 (e.g. planning, counterfactual reasoning)？这是 reward model → world model 的演化路径。

---

## References

- Paper GitHub: https://github.com/Kwai-Keye/JRM-Joint-Reward-Modeling
- EditReward: https://arxiv.org/abs/2509.26346
- EditScore: https://arxiv.org/abs/2509.23909
- Flow-GRPO: https://arxiv.org/abs/2505.05470
- Qwen3-VL: https://arxiv.org/abs/2511.21631
- OmniGen2: https://arxiv.org/abs/2506.18871
- HPSv3: https://arxiv.org/abs/2509.08826 (Ma et al. 2025)
- MMRB2: https://arxiv.org/abs/2512.16899
- Implicit CoT distillation: https://arxiv.org/abs/2311.01460
- Pause tokens: https://arxiv.org/abs/2310.02226
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- R1-Reward: https://arxiv.org/abs/2505.02835
- Critique-out-loud RM: https://arxiv.org/abs/2408.09127
- Representation anisotropy: https://aclanthology.org/2019.emnlp-main.415/
- DPO: https://arxiv.org/abs/2305.18290
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- DanceGRPO: https://arxiv.org/abs/2505.07818
- Latent reasoning (concept): https://arxiv.org/abs/2412.06869

希望这些细节帮 build 你对 latent CoT + joint reward modeling 的 intuition。核心 take-away：**reasoning 是 representation 的 property，不是 output sequence 的 property**。JRM 用 LM loss 把 reasoning structure 压进 hidden state，再让 discriminative head 直接读取。这比显式生成 reasoning text 更高效，且 preference alignment 更准。如果你想 push 这个方向，latent multi-step reasoning + RL on latent space 是最有潜力的 next step。
