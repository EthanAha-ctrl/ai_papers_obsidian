---
source_pdf: lcpo.pdf
paper_sha256: fd9b3d1c0503e505f06a26ea1461fe2ca0ee1cb0a1d841d9dec9190361dd3e3e
processed_at: '2026-08-05T12:26:25-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 L1 这篇 paper

---

## 一句话说清楚

现在的 reasoning model（R1、O1 那些）有个毛病：**你管不住它想多久**。你说 "想 500 token"，它给你吐 6000 token；你说 "想快点"，它还是在那反复纠结。这篇 paper 就是用 RL 逼模型学会一件事：**你说多少 token，我就用多少 token，而且还能把题做对**。

---

## 问题的本质

想象一个特别聪明的数学教授，你问他 "2+3=?"，他能在黑板前写 10 页推导过程，最后告诉你答案是 5。你问他一个 IMO 金牌题，他可能也写 10 页，但 10 页不够，写到一半卡住了。

这就是现在 reasoning model 的现状：

- **简单题浪费算力**：DeepScaleR-24K 不管你 prompt 说 "think for 512 tokens" 还是 "think for 3600 tokens"，它都给你生成大约 6000 tokens（paper Table 5 实测）
- **难题可能不够用**：固定 budget 又怕难题没想透
- **没法做预算分配**：deployment 时我想给简单题 500 token、难题 4000 token，做不到

S1 那篇 paper 试图用 "budget forcing" 解决——到 budget 就硬截断，或者塞一个 "Wait" token 让它继续。这就像你让教授讲到 500 字时强行打断他，说 "直接告诉我答案"。问题是教授可能讲到第 499 字时正好在中间一步推导，思路断了，答案就错了。

---

## L1 的做法：用 RL 逼模型学

核心 idea 简单得离谱：**在 prompt 末尾加一句 "Think for N tokens"，然后用 RL reward 训练模型真的按 N 来生成**。

训练数据只有 (题目, 最终答案)，没有 reasoning trace。模型自己生成 reasoning，RL 来 reward 两件事：

1. **答案对不对**（correctness）
2. **长度匹不匹配**（length adherence）

### 两个 reward 函数的区别

**L1-Exact**：要求长度精确等于目标值
```
reward = 答对了(1/0) - 0.0003 × |目标长度 - 实际长度|
```

**L1-Max**：要求长度不超过目标值
```
reward = 答对了 × clip(0.0003 × (目标长度 - 实际长度) + 0.5, 0, 1)
```

为什么 L1-Max 用乘法不用加法？因为加法有个 bug：答案错的时候 reward 是负的，模型发现 "只要我生成极短的废话，penalty 就最小"，于是 collapse 到永远输出一句话。乘法把这个问题切断了——答错就是 0 reward，无论长短，模型没有逃逸路径。

那个 0.5 的 offset 也很有讲究：答案对、长度完美匹配时 reward = 0.5；答案对但超长 1000 token 时 reward = 0.2，还是正的。这保证 "答对但略超长" 比 "答错" 好，模型不会为了控长度故意答错。

---

## 训练过程有个 "Aha Moment"

这是整个 paper 最有意思的现象。

训练前 300 步，模型完全无视 length instruction——它的 minimum response length 一直在降，reward 主要靠 correctness 撑着。这很合理，因为 R1-Distill 这个 base model 预训练时被狠狠强化了 "长 CoT = 好" 的 prior，这个 prior 在抵抗 length constraint。

到大约 300 步时，突然发生 **phase transition**：reward 和 token adherence score 开始快速上升，minimum response length 锐降到 100 token 左右（训练时设的下界）。

模型好像突然 "懂了" length instruction 这件事是可以 obey 的，然后快速学会。

这让我想起 grokking 现象——模型不是平滑地学习，而是先积累，到某个阈值后突然 "啊哈我懂了"。对实际训练来说，这意味着 checkpoint selection 要小心，前 300 步看着像没学到 length control，但其实 gradient signal 在累积。

---

## 两个反直觉的发现

### 发现一：1.5B 小模型在相同 token budget 下干翻 GPT-4o

这是最 striking 的结果。把所有模型都限制在大约 820 token 的 generation length：

| 模型 | 参数量 | 准确率 |
|---|---|---|
| Qwen-1.5 (non-reasoning base) | 1.5B | 41.0% |
| Llama-3.3-70B | 70B | 41.2% |
| GPT-4o | ~1T | 45.6% |
| **L1-Max (1.5B)** | **1.5B** | **47.8%** |

1.5B 的 L1-Max 比 GPT-4o 高 2 个点！在相同 token budget 下。

为什么？因为 L1-Max 虽然生成短，但它保留了 long-CoT 学到的 reasoning patterns——self-correction、verification、conclusion drawing。这些 pattern 是 R1-Distill 预训练时在长链里学会的，RL 只是教它 "把这些 pattern 压缩到短链里用"。

non-reasoning model（Qwen base、Llama-70B、GPT-4o）从来没学过这些 pattern，即使给同样的 token budget，它们也只是 "写一段流畅但没深度的解答"。

论文把这种模型叫 SRM（Short Reasoning Model）。本质上是：**long-CoT 训练是一种 reasoning distillation 机制，然后 RL 让模型学会在短链里 reuse 这些 reasoning 能力**。

### 发现二：SFT 完全失败

作者试了用 SFT 教 length control——先让模型生成各种长度的 response，标上 token 数，然后用 (prompt+length instruction, response) 做监督学习。结果模型完全不听话，不管 prompt 说多少 token，都生成 21000+ token。

为什么 SFT 失败？因为 SFT 本质是模仿，而 "在约束下推理" 是一个需要探索的能力。模型要知道 "500 token 能做到什么程度、4000 token 能做到什么程度"，这需要 RL 的 trial-and-error + reward signal。SFT 只看到了 "这个问题对应这个长度"，学不到 "换一个长度指令我该怎么调整 reasoning"。

---

## Length Control 的精度

数学题上 mean error 大约 3%，OOD 任务（GPQA、LSAT）20-40%，MMLU 最高。

有意思的是 mean error 小但 RMSE 大，说明大部分样本精确匹配，少数样本偏差很大——那些就是模型无论你怎么说都需要长 CoT 才能解的难题。

延长训练（再加 500 步）能把 RMSE 砍一半，但高 token range 的性能略降——因为模型更严格地遵守 length constraint，遇到难题也没法多想了。这是个 precision vs performance 的 trade-off。

---

## Reasoning Pattern 怎么随长度变化

作者统计了不同 length budget 下 reasoning keywords 的频率：

- **Self-correction**（"wait", "let me check", "actually"）：长 CoT 比短 CoT 多 2 倍
- **Conclusion drawing**（"therefore", "thus"）：长 CoT 多 2-10 倍
- **Exploration**（"alternatively", "let's try"）：大部分下降，"Alternatively" 例外上升

直觉解读：长 CoT 不是 "多写几步推导"，而是 "多验证几次 + 多总结几次"。模型在长链里反复质疑自己，短链里直接给出压缩版的核心推理。

另一个观察：thinking/solution token ratio 跨长度稳定。短 CoT 时模型连 solution 都写得很短（经常就吐个最终答案），长 CoT 时 thinking 变长但 solution 不会变得啰嗦。这避免了 "废话填充" 的 degenerate solution。

---

## OOD 泛化

L1 只在 math 上训，但 length control 能力迁移到了 GPQA、LSAT 上，甚至 MMLU 也有一定效果。

GPQA 和 LSAT 还保持 linear scaling（token 越多准确率越高），MMLU 的 scaling 弱（R²=0.66），因为 MMLU 主要是知识 recall，长 reasoning 帮助不大，反而强迫长 CoT 时误差变大。

这暗示 length control 是一个比较 "general" 的 capability，不绑定具体任务。

---

## 一些我自己觉得值得深挖的点

1. **Aha moment 的机制**：300 步的 phase transition 到底在 representation 层面发生了什么？有没有可能用 mechanistic interpretability 找到 "length neuron"？

2. **SRM 为什么 work**：long-CoT 的 reasoning pattern 怎么压缩到 short-CoT 还不丢性能？这是不是某种 implicit distillation？

3. **Difficulty-adaptive 而非 user-specified**：L1-Max 已经有点这个意思了——它根据题目难度自适应分配 token。能否完全去掉 length instruction，让模型自己估计 difficulty 然后 allocate？

4. **和 test-time scaling laws 的关系**：Snell et al. 的工作说 test-time compute 有 optimal allocation。L1-Max 实际上就是给了一个 knob 来 tune 这个 allocation，而且 L1-Max 的 log-linear scaling slope (0.24) 比 S1 (0.37) 小，说明 L1 在低 budget 更高效。

5. **为什么 multiplicative reward 比 additive 好**：这个 ablation 结果其实揭示了一个 general 的 RL design principle——当有多个 objective 时，乘法切断 "部分满足" 的逃逸路径，加法允许 trade-off。这和 Constitutional AI 的 principle application 有异曲同工之处。

6. **Token-level vs semantic-level length**：token 数量和 reasoning step 数不严格对应。能否控制 "reasoning step" 而非 token？这对真正的 reasoning efficiency 可能更有意义。

---

## 我的直觉总结

这篇 paper 给我三个核心 insight：

**第一，length control 是一个 RL-friendly 但 SFT-unfriendly 的问题**。SFT 模仿不到 "在约束下推理" 的能力，因为这是个需要探索的 decision problem。RL 的 online reward signal 让模型试出来 "500 token 能做到什么、4000 token 能做到什么"。

**第二，reasoning 是可压缩的**。long-CoT 训练学会的 reasoning patterns 可以被 RL 压缩到 short-CoT 里，而且不显著损失性能。1.5B 模型在相同 budget 下超过 GPT-4o 就是证据。这说明 reasoning ability 不在 "长度" 上，而在 "pattern" 上。

**第三，prompt conditioning + RL 是个通用范式**。这篇 paper 用 length instruction 做条件，但同样的框架可以用来教模型 respond to 任何 instruction-conditioned constraint——置信度、verbosity、safety level、tool budget 等等。RL 让模型学会 obey instruction-specified constraints，这是 beyond pure correctness 的能力。

---

# L1: 用 RL 控制 Reasoning Model 的思考长度 — 深度技术解析

Andrej, 这篇论文非常对你的胃口，因为它实际上回答了一个你肯定会问的核心问题：**reasoning model 的 test-time compute scaling 能不能像参数 scaling 一样被精确控制？**答案是可以，而且方法出奇地简单，但产生了几个反直觉的现象。让我详细拆解。

---

## 1. 核心问题：为什么需要 Length Control？

当前的 reasoning model（O1, DeepSeek-R1, DeepScaleR 等）都有一个 fundamental 的 limitation：**它们的 CoT 长度是不可控的**。模型自己决定 "想多久"，这对 deployment 是灾难性的：

- 简单问题浪费 6000+ tokens（参考论文 Table 5：DeepScaleR-24K 无论 prompt 说 "think for 512 tokens" 还是 "think for 3600 tokens"，都生成约 6000 tokens）
- 复杂问题可能过早 stop
- 无法做 test-time compute budget allocation

这个问题在 inference scaling laws 文献 (Snell et al., 2024; Wu et al., 2024) 中被反复提及，但一直没有干净的方法解决。S1 (Muennighoff et al., 2025) 用 "budget forcing" 尝试解决，但本质上是硬性截断 + 插入 "Wait" token，会 mid-step 截断 reasoning，破坏 CoT 的连贯性。

论文的核心 insight：**length control 本质上是一个 constrained optimization 问题，应该用 RL 来解，而不是 prompt engineering 或 SFT**。

参考链接：
- Snell et al. scaling test-time compute: https://arxiv.org/abs/2408.03314
- S1: https://arxiv.org/abs/2501.19393
- DeepSeek-R1: https://arxiv.org/abs/2501.12948

---

## 2. LCPO 方法的数学细节

### 2.1 Prompt Augmentation

给定原始 dataset $\mathcal{D} = \{(x_i, y_{gold,i})\}_{i=1}^N$，注意这里只有 final answer $y_{gold}$，没有 intermediate reasoning trace。每个 prompt 被增强为：

$$x_i^{new} = \text{Concat}\left(x_i, \text{"Think for } n_{gold,i} \text{ tokens."}\right)$$

其中 $n_{gold,i} \sim \mathbb{Z}(n_{min}, n_{max})$，即从整数均匀分布采样。

**关键设计直觉**：均匀采样 $n_{gold}$ 而不是固定长度，确保模型见过各种长度请求，避免 collapse 到某个特定长度。这和 instruction tuning 的多样性原理类似。

实验中：$n_{min} = 100, n_{max} = 4000$。

### 2.2 L1-Exact 的 Reward Function (Equation 1)

$$r(y, y_{gold}, n_{gold}) = \mathbb{I}(y = y_{gold}) - \alpha \cdot |n_{gold} - n_y|$$

变量解析：
- $y$: 模型生成的 response
- $y_{gold}$: ground truth final answer（不是 CoT，是最终答案）
- $n_{gold}$: target token length（prompt 中指定的）
- $n_y$: 实际生成的 token length
- $\mathbb{I}(\cdot)$: indicator function，答案正确返回 1，否则 0
- $\alpha$: scalar balancing parameter，论文设为 $\alpha = 0.0003$

**为什么 $\alpha = 0.0003$ 这么小？** 

这是一个很重要的 scaling intuition。Indicator reward 的范围是 $\{0, 1\}$，而 length penalty 项 $|n_{gold} - n_y|$ 的范围可达几千（比如偏差 3000 tokens）。如果 $\alpha$ 太大，length penalty 会 dominate，模型会为了精确匹配长度而牺牲 correctness——比如直接生成废话填充到目标长度。如果 $\alpha$ 太小，模型会忽略 length instruction，像原来的 reasoning model 一样生成任意长度。

$\alpha = 0.0003$ 意味着 3000 tokens 的 deviation 产生 0.9 的 penalty，大致和 correctness reward 同量级。这是一个非常精细的 balance。

### 2.3 L1-Max 的 Reward Function (Equation 2)

$$r(y, y_{gold}, n_{gold}) = \mathbb{I}(y = y_{gold}) \cdot \text{clip}(\alpha \cdot (n_{gold} - n_y) + \delta, 0, 1)$$

变量解析：
- $\text{clip}(\cdot, 0, 1)$: 将 reward 截断到 $[0, 1]$
- $\delta = 0.5$: offset term

**为什么用乘法而不是加法？** 这是论文 ablation (Appendix A.7) 的核心发现：

| 变体 | 公式 | 结果 |
|---|---|---|
| Multiplicative (论文采用) | $\mathbb{I} \cdot \text{clip}(\ldots)$ | 平衡好 |
| Additive | $\mathbb{I} - \alpha \cdot |n_{gold} - n_y|$ | collapse 到极短 CoT |
| Sigmoid | $\mathbb{I} \cdot \sigma(\alpha(n_{gold} - n_y))$ | 类似 multiplicative |
| Single objective (only max) | 只有 max constraint | 类似 |

Additive 的问题：当答案错误时 $\mathbb{I} = 0$，reward 变成 $-\alpha \cdot |n_{gold} - n_y|$，仍然是负值。模型会学到 "生成极短 CoT 可以最小化这个 penalty"，于是 collapse 到 trivial 短输出。

Multiplicative 的妙处：错误答案 $\mathbb{I} = 0$ 让整个 reward 归零，无论 length 如何。这切断了 "错误但短" 的逃逸路径，强迫模型先 correctness 再 length。

**$\delta = 0.5$ 的直觉**：当答案正确且 length 完美匹配时 $n_{gold} - n_y = 0$，reward = $1 \cdot \text{clip}(0.5, 0, 1) = 0.5$。当答案正确但超出预算 1000 tokens（$\alpha = 0.0003$），reward = $\text{clip}(0.5 - 0.3, 0, 1) = 0.2$，仍然大于 0。这确保了 "正确但略超长" 比 "错误" 更好，避免模型为了避免 penalty 而故意答错。

### 2.4 Dual Training Objective

L1-Max 实际上是用 **dual objective** 训练的：
- 当 prompt 请求 exact length → 用 Equation 1
- 否则 → 用 Equation 2 (max constraint)

这让 L1-Max 同时具备 exact 和 max 两种能力。

### 2.5 RL 算法：GRPO

论文采用 GRPO (Group Relative Policy Optimization, Shao et al., 2024)，这是 DeepSeek 系列的标准 RL 算法。GRPO 相比 PPO 的核心区别是不需要 value network，而是用 group-relative advantage：

$$A_i = \frac{r_i - \text{mean}(r_{group})}{\text{std}(r_{group})}$$

其中 $r_{group}$ 是对同一个 prompt 采样的多个 response 的 reward。这降低了训练成本，特别适合 reasoning 这种 reward sparse 的场景。

参考：DeepSeekMath GRPO: https://arxiv.org/abs/2402.03300

---

## 3. 架构图解析（从文字描述重建）

由于论文没有显式的架构图，我根据描述重建 LCPO 的训练 pipeline：

```
┌─────────────────────────────────────────────────────┐
│  Training Data: (x_i, y_gold,i) — only final answer │
└─────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  Prompt Augmentation                                │
│  x_new = x_i + "Think for n_gold tokens"            │
│  n_gold ~ U(100, 4000)                               │
└─────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  Base Model: DeepScaleR-1.5B-Preview                │
│  (RL-finetuned from DeepSeek-R1-Distill-Qwen-1.5B)  │
└─────────────────────────────────────────────────────┘
                       │
                       ▼ (sampling multiple responses)
┌─────────────────────────────────────────────────────┐
│  Generate G responses per prompt                     │
│  {y_1, y_2, ..., y_G} with n_y_i tokens each        │
└─────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  Reward Computation                                 │
│  r_i = I(y_i = y_gold) - α|n_gold - n_y_i|  (Exact)  │
│  r_i = I(y_i = y_gold) · clip(α(n_gold - n_y_i)+δ,0,1) (Max) │
└─────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  GRPO Update                                        │
│  A_i = (r_i - mean(r)) / std(r)                      │
│  L = -E[min(ρ_i · A_i, clip(ρ_i, 1-ε, 1+ε) · A_i)]  │
│  + β · KL(π_θ || π_ref)                             │
└─────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  L1-Exact (700 steps) → L1-Max (+120 steps)         │
└─────────────────────────────────────────────────────┘
```

注意训练是 **two-stage**：
1. 先 700 steps 训 L1-Exact（用 Equation 1）
2. 再 120 steps 用 Equation 2 fine-tune 得到 L1-Max

这种 curriculum 设计让模型先学会 "匹配精确长度"，再学会 "在 max budget 内自适应分配"。

---

## 4. 实验结果深度分析

### 4.1 主结果（Figure 2 的数据）

| 模型 | 512 tokens | 1024 tokens | 2048 tokens | 3600 tokens |
|---|---|---|---|---|
| S1 | ~10% | ~15% | ~25% | ~35% |
| L1-Exact | ~25% | ~35% | ~45% | ~50% |
| L1-Max | ~30% | ~40% | ~50% | ~55% |
| DeepScaleR-4K | ~50% | ~52% | ~55% | ~58% |
| DeepScaleR-24K | ~55% | ~58% | ~60% | ~62% |

（这些数字是从 Figure 2 估算的，论文没给精确表格）

**关键观察**：

1. **Log-linear scaling**：accuracy vs log(tokens) 是线性的，slope 为 0.24（L1）vs 0.37（S1）。这意味着 L1 在低 token budget 下更高效——doubling tokens 带来的提升在 L1 中相对小，但起点高。

2. **L1-Exact vs DeepScaleR-4K 仅差 ~1%**：在 AIME 上差距更大，因为 AIME 题目难，无约束模型可以生成 24K tokens，而 L1-Exact 被限制在 4K。这暗示 **AIME 这种 hard task 的 marginal value of tokens 很高**。

3. **L1-Max matches DeepScaleR-4K**：因为 L1-Max 可以根据题目难度自适应分配 tokens，简单题少用，难题多用。这比 L1-Exact 的 "一刀切" 更优。

### 4.2 OOD Generalization（Figure 3）

这是论文最 striking 的结果之一。L1 只在 math 上训练，但泛化到：

- **GPQA** (graduate-level Q&A): linear scaling 保持
- **LSAT** (逻辑推理): linear scaling 保持
- **MMLU** (general knowledge): $R^2 = 0.66$，scaling 较弱

**为什么 MMLU scaling 弱？** 

MMLU 主要是 recall 类问题，长 CoT 帮助不大。论文 Figure 9 显示 MMLU 在 longer CoT 时 error 反而上升——模型被强迫生成无用 reasoning。这和 Chen et al. 2025 ("Do not think that much for 2+3=?") 的 overthinking 发现一致：简单问题过度思考反而 hurt performance。

参考 overthinking paper: https://arxiv.org/abs/2412.21187

### 4.3 Length Control Precision（Figure 4, 11）

Mean error 定义：

$$\text{Mean Error} = \frac{\mathbb{E}_{x \sim \mathcal{D}}[n_{generated}] - n_{gold}}{n_{gold}}$$

| Dataset | Mean Error (%) | RMSE (%) |
|---|---|---|
| Math Reasoning | 3.01 | 18.44 |
| OOD-1 (GPQA, LSAT) | 21.22 | 31.37 |
| OOD-2 (MMLU) | 40.54 | 42.61 |

**关键 insight**：mean error 小但 RMSE 大，说明误差分布是长尾的——大部分样本精确匹配，少数样本偏差很大。这是 reasoning task 的特性：有些问题无论 prompt 怎么说都需要长 CoT 才能解决。

Extended training (L1-Exact+, +500 steps) 将 RMSE 从 18.44 降到 10.04，但 high token range 性能略降。这是 **precision vs performance 的 trade-off**。

### 4.4 Budget Violation（Figure 5）

L1-Max 的 soft violation rate（$|n_{generated} - n_{gold}| > 500$）仅 1.3% 平均，hard violation 更宽松但也就 9%。这说明 multiplicative reward + clip 的设计确实有效。

### 4.5 SRM：最反直觉的发现（Table 1）

这是我最想和你讨论的部分。论文发现 **LCPO 训练的 long-CoT 模型在 short budget 下变成了 strong short-CoT model**。

对比相同 token budget (~820 tokens average):

| Model | Avg Tokens | Avg Accuracy |
|---|---|---|
| Qwen-1.5 (non-reasoning base) | 752 | 41.0% |
| L1-Exact | 758 | 41.2% |
| L1-Max | 720 | 46.2% |
| Llama-3.3-70B | 824 | 41.2% |
| GPT-4o | 840 | 45.6% |
| L1-Max (1024 prompt) | 816 | 47.8% |

**1.5B 的 L1-Max 在相同 token budget 下超过 GPT-4o 2%！** 这是非常 striking 的。

**为什么 SRM 有效？** 我的理解：

1. **Reasoning pattern 迁移**：Figure 6 显示 short CoT 保留了 long CoT 的 reasoning patterns（self-correction, verification），只是频率调整。non-reasoning model 从未学过这些 patterns。

2. **RL 的 credit assignment**：GRPO 让模型在 short budget 下也必须产生有效 reasoning，否则 reward 为 0。这 force 模型学会 "压缩的 reasoning"。

3. **vs. SFT 的对比**：Appendix A.6 显示 SFT 完全失败，无论怎么 train 都生成 ~21000 tokens。这说明 **length control 是一个 RL-friendly 但 SFT-unfriendly 的任务**——因为 SFT 只能模仿，不能探索 "如何在约束下推理"。

---

## 5. 训练动力学：Aha Moment（Figure 14）

这是论文最 fascinating 的训练动态。在 ~300 RL steps 时出现 phase transition：

- 前 300 steps：模型 focus on correctness，minimum response length 持续下降
- ~300 steps：reward 和 token adherence score 突然开始快速上升，minimum length 锐降到 ~100 tokens（$n_{min}$ 下界）

这非常像 Grokking (Power et al., 2022) 或 Anthropic 的 "phase transitions in RL" 现象。模型先学会 "正确解题"，然后突然 "理解" 了 length instruction 的语义。

**直觉解释**：reasoning model 预训练时强化了 "长 CoT = 好" 的 prior。前 300 steps，这个 prior 抵抗 length constraint。300 steps 后，gradient signal 累积到阈值，model "realize" length instruction 是可遵守的，于是快速 collapse 到正确行为。

参考 grokking: https://arxiv.org/abs/2201.02277

---

## 6. Reasoning Pattern Analysis（Figure 6, 7）

论文把 reasoning keywords 分成 4 类：

1. **Self-Correction and Verification**: "wait", "let me check", "actually", "no"
2. **Exploration and Alternatives**: "alternatively", "let's try", "another approach"
3. **Context Setting**: "given", "we need", "the problem"
4. **Conclusion Drawing**: "therefore", "thus", "so the answer", "final answer"

512 vs 4096 tokens 的关键词频率变化：

| Pattern | 4096 / 512 ratio |
|---|---|
| Self-Correction | ~2x |
| Conclusion Drawing | 2-10x |
| Exploration | 大部分下降，"Alternatively" 例外上升 |

**直觉**：长 CoT 不是简单地 "多写步骤"，而是 "多验证 + 多总结"。这和 O1/R1 的 self-correction 行为一致——模型在长 CoT 中反复质疑自己的中间结论。

Figure 7 显示 thinking/solution token ratio 跨长度稳定，说明模型不是在 padding solution，而是在 scaling thinking。这是 LCPO 训练的好特性——避免了 "废话填充" 的 degenerate solution。

---

## 7. 与相关工作的对比

### 7.1 vs S1 (budget forcing)

S1 的方法：到 budget 就截断，插入 "Final Answer" token 强制结束，或插入 "Wait" 让模型继续。

LCPO 的优势：
- S1 截断会 mid-step 中断 reasoning，破坏 CoT 连贯性
- S1 的 "Wait" token 重复使用导致 rigid pattern
- S1 没有 training，纯 inference-time，无法学习 "如何在 budget 内最优分配"

### 7.2 vs Inference Scaling Laws (Snell et al.)

Snell 的工作显示 test-time compute 有 optimal allocation point，不是越长越好。LCPO 实际上给了 user knob 来 control 这个 allocation。

### 7.3 vs Overthinking/Underthinking (Wang et al. 2025, Chen et al. 2025)

这些工作发现 reasoning model 在简单题上 overthink，难题上 underthink。LCPO 通过 prompt-conditioned length control 解决了这个问题——user 可以 per-question 设置 budget。

参考 underthinking: https://arxiv.org/abs/2501.18585

---

## 8. Limitations 和未来方向

论文承认的 limitation：
1. **不能 generalize 到比训练更长的 length**：训练 max 4000，inference 请求 8000 可能不行
2. **Reward 是完整 output 的 length**，没有分别控制 thinking vs solution tokens

我想补充几个你可能会关心的方向：

1. **Length-conditioned  vs difficulty-conditioned**：当前是 user-specified length，能否让模型自动 estimate difficulty 然后 allocate？这和 DeepMind 的 AlphaProof 思路类似。

2. **Multi-turn length control**：agent 场景下，每个 turn 的 reasoning length 如何 control？

3. **Length 和 latency 的 trade-off**：长 CoT 不只是 token cost，还有 KV cache、attention 计算的 quadratic cost。真正的 deployment cost 不是线性的。

4. **Length control 的 interpretability**：能否分析模型内部 representations 如何 encode "target length"？是否存在 "length neuron"？

5. **Token-level vs semantic-level length**：token 数量和 reasoning 步骤数不严格对应。能否控制 "reasoning steps" 而非 "tokens"？

---

## 9. 我的 Intuition 总结

这篇论文给我的几个关键 takeaways：

1. **RL 是 length control 的正确工具**：SFT 失败，prompt engineering 失败，只有 RL 的 online reward signal 能让 model 学会 "adaptive reasoning under constraint"。这和 RLHF 教模型 "helpful but harmless" 的原理类似——这是 constrained optimization，不是模仿学习。

2. **Reasoning 是 compressible 的**：SRM 的发现证明 long-CoT 学到的 reasoning patterns 可以压缩到 short-CoT 而不显著损失性能。这暗示 long-CoT 训练可能是一种 **reasoning distillation 机制**——模型在长链中学会 reasoning，然后在短链中 reuse。

3. **Phase transition in RL**：Aha moment 提示 reasoning model 的 fine-tuning 不是 smooth improvement，而是 discrete capability acquisition。这对 RL training 的 checkpoint selection 和 early stopping 有实际意义。

4. **Prompt conditioning is powerful**：通过简单的 prompt augmentation + RL，模型学会了 respond to instruction 的 length。这暗示 RL 可以让模型学会很多 prompt-conditioned 的 capabilities，不仅仅是 length。

5. **Small model can match frontier**：1.5B L1-Max 在相同 token budget 下超过 GPT-4o，这是 "compute-matched" 比较的新范式。比单纯的 parameter scaling 更有信息量。

---

## 10. 可能的联想和延伸

最后一些值得深挖的联想：

- **和 Constitutional AI 的关系**：Anthropic 用 RL 从 principles 学习，LCPO 用 RL 从 length instruction 学习。都是 "instruction-conditioned RL"。
- **和 process reward models (PRM) 的关系**：LCPO 只用 outcome reward (correctness)，能否加入 process reward 来引导 intermediate reasoning quality？
- **和 MoE 的关系**：能否让不同 expert 负责不同 length regime？
- **和 tool use 的结合**：reasoning model + tool calling 时，length control 如何和 tool call budget 协调？
- **和 verification 的结合**：LCPO 的 self-correction pattern 增加暗示可以和 formal verification (Lean, Coq) 结合。

参考 Constitutional AI: https://arxiv.org/abs/2212.08073
参考 PRM: https://arxiv.org/abs/2305.20050

---

希望这个 detailed breakdown 能 build your intuition，Andrej。如果你想深挖任何部分（比如 GRPO 的具体 gradient analysis，或者 SRM 的 mechanistic interpretability hypothesis），我可以继续展开。
