---
source_pdf: FAST-DLLM V2- Efficient Block-Diffusion LLM.pdf
paper_sha256: 9e3dcb349e0269fee8ac8c525131284a6408e506900c69940bb898f41274d49f
processed_at: '2026-08-04T07:08:16-07:00'
target_folder: LLM-Training
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 Fast-dLLM v2

## 一句话总结

你手头有个训练好的 AR 模型 (比如 Qwen2.5-7B)，它只会一个字一个字蹦。你想让它一次写一整块文字来加速，但又不想从头重新训练 500B tokens。Fast-dLLM v2 告诉你：把文本切成 32 个 token 一块，**块之间保持 AR（左到右，能用 KV cache），块内部让 tokens 互相看一眼（bidirectional，能并行 refine）**。因为大部分计算模式跟原来 AR 模型几乎一样，只微调 1B tokens 就够了，速度还能快 2.5 倍。

---

## 为什么要做这件事

AR 模型 (GPT, Qwen, LLaMA) 生成 256 个 token 要做 256 次 forward pass，一次出一个字，GPU 大部分时间在闲着等。

Diffusion LLM (LLaDA, Dream) 想解决这个问题：一口气把所有 token 都 mask 掉，然后 iteratively refine，每步可以并行 update 多个 token。听起来很美，但实际有几个坑：

**坑 1：KV cache 用不了**。AR 模型之所以快，是因为已经算过的 prefix token 的 KV 可以存起来下次直接用。但 diffusion 模型用 bidirectional attention，每个 token 能看到所有其他 token，当你 unmask 一个新 token 时，之前所有 token 的 representation 都变了，cache 就废了，得全部重算。

**坑 2：要重新训练**。Dream 想把 Qwen2.5-7B 变成 full bidirectional diffusion model，结果要从头训练 500B tokens。因为 bidirectional attention 跟 AR model 训练时的 causal attention 完全不一样，模型原来学到的东西基本得推翻重来。

**坑 3：序列长度不灵活**。很多 diffusion model 要预先固定长度，不好处理变长生成。

---

## Block Diffusion 的 intuition

核心 idea 很朴素：**别走极端**。

纯 AR = 每个 token 只看左边 → 太慢
纯 diffusion = 每个 token 看所有方向 → cache 废了

折中方案：**切成 block，block 间 AR，block 内 diffusion**

```
Block 1  →  Block 2  →  Block 3  →  Block 4    (AR, 左到右, 有 cache)
[token token token token] [token ...] [token ...]
         ↑ block 内部 bidirectional，可以并行 refine
```

这样你同时拥有：
- **Block 间是 AR** → KV cache 照用，block 1 算完存起来，block 2、3、4 直接读
- **Block 内是 diffusion** → 32 个 token 可以一起 refine，并行出结果
- **序列长度灵活** → 想生成多少个 block 就生成多少个，不用预固定

这个 idea 本身 BD3-LM (https://arxiv.org/abs/2503.09573) 已经提过了，但只在 small scale 验证过。Fast-dLLM v2 把它真正 scale 到 7B model 上，并且解决了一个关键问题：**怎么用很少的数据把 AR 模型变成 block diffusion 模型**。

---

## 为什么 1B tokens 就够了 (核心 insight)

这是整篇 paper 最 clever 的地方。

Dream 用 500B tokens 是因为它要把 AR 模型改造成 full bidirectional — 模型的 attention pattern 变了，原来学的"看到左边这些 token，预测下一个"这个能力得重新学。

Fast-dLLM v2 的 insight 是：**block-wise attention 其实跟 AR attention 非常接近**，大部分计算模式没变，所以 pretrained model 的知识大部分能保留。

具体靠两个 trick：

### Trick 1: Token Shift (保留 next-token prediction 的习惯)

AR 模型训练时学的是：在位置 $i$，用 hidden state $h_i$ 预测 $x_{i+1}$（next token）。

标准 masked LM 会让模型在位置 $i$ 用 $h_i$ 预测 $x_i$（current token）。这跟 AR 模型的习惯完全相反，模型要重新学"在当前位置预测当前 token"。

Token shift 的做法：如果位置 $i$ 被 mask 了，用位置 $i-1$ 的 hidden state 来预测 $x_i$。

$$
p(x_i | \text{context}) = \text{softmax}(W \cdot h_{i-1})
$$

这跟 AR 模型的 next-token prediction **完全一样**：位置 $i-1$ 的 hidden state 预测位置 $i$ 的 token。模型不用改预测习惯，只需要学会"怎么从 block 内其他 token 那里获得额外 context"。

### Trick 2: Block-Causal Attention Mask (大部分 attention 跟 AR 一模一样)

训练时把 noised sequence $x_t$ 和 clean sequence $x_0$ 拼起来 (长度 $2L$)，用这个 attention mask：

$$
\mathcal{M}_{\text{full}} = \begin{bmatrix} \mathcal{M}_{BD} & \mathcal{M}_{OBC} \\ 0 & \mathcal{M}_{BC} \end{bmatrix}
$$

**左下角的 0** 是关键：clean sequence 的 attention 完全不被 noised sequence 污染。而 $\mathcal{M}_{BC}$ (Block-Causal) 让 clean sequence 内部保持标准 AR attention pattern — 每个 clean token 看同 block 和更早 block 的所有 token。

这意味着 clean sequence 的 hidden states 跟原始 pretrained model 算出来的几乎一样。模型只需要学两件新事：

1. **$\mathcal{M}_{BD}$**: block 内 bidirectional — noised tokens 之间互相看（新能力）
2. **$\mathcal{M}_{OBC}$**: noised tokens 从 clean prefix 那里读 context（类似 cross-attention）

这两件事是"增量学习"，不需要推翻重来，所以 1B tokens 就够了。

---

## Complementary Mask — 保证每个 token 都被训练到

Diffusion 训练时，每个 block 随机 mask 一些 token。问题：如果 mask ratio 是 $t$，平均只有 $t \times D$ 个 token 会被 mask，loss 信号很稀疏。有些 token 可能好几轮训练都没被 mask 过。

Complementary mask 的做法：每个训练样本复制成两个 views：
- View A: mask pattern $m$，时间步 $t$
- View B: mask pattern $\bar{m} = 1 - m$，时间步 $1 - t$

两个 views 放同一个 batch。这样每个 token 在 View A 或 View B 中**一定有一个被 mask**，loss 信号全覆盖。

附带好处：标准 masked LM loss 里有个 $\frac{1}{t}$ 的归一化项（因为期望上 $tL$ 个 token 被 mask），当 $t \to 0$ 时会数值不稳定。用了 complementary mask 后，两个 views 加起来正好 mask 了 $L$ 个 token ($tL + (1-t)L = L$)，归一化项自然消掉，不需要除以 $t$。

Ablation 显示 complementary mask 贡献 +2.8 平均分。对 code generation 尤其有效（HumanEval+ +6.1），因为 code 里每个 token 都重要，都需要被训练到。

---

## Padding — 防止 sample 之间"串味"

训练时会把多个 sample 拼接 (packing) 填满 context length。问题：如果 sample A 的 EOS 后面紧跟 sample B 的 BOS，它们可能在同一个 block 内。Block 内是 bidirectional attention，模型会看到 A 的结尾 attend 到 B 的开头，造成 cross-sample leakage。

Solution：每个 sample 右 pad `[MASK]` tokens 让长度是 block size (32) 的整数倍。这样每个 sample 的 boundary 跟 block boundary 对齐，不会串。

Ablation 显示 padding 把 IFEval 从 39.9 拉到 45.8，提升 +5.9。这个看起来不起眼的 trick 效果巨大，因为 instruction following 对 context 干净程度很敏感。

---

## Inference 流程

```
1. 给 prompt，后面 pad [MASK] tokens 凑够 block 对齐
2. 对第一个 block:
   a. 全部 [MASK]，做 forward pass
   b. 算每个位置的 confidence = max(prob)
   c. confidence > 0.9 的位置 → unmask（finalize）
   d. 剩下的 [MASK] 继续 refine
   e. 重复直到全部 unmask
3. 第一个 block 完成后，KV cache 存起来
4. 对第二个 block: 同样流程，但能 attend 到第一个 block 的 cached KV
5. ...继续
```

### Confidence-based parallel decoding 的 intuition

threshold = 1.0：每次只 unmask confidence 最高的 1 个 token，最稳但最慢（相当于 full diffusion denoising）

threshold = 0.9：每次把 confidence > 0.9 的 token 都 unmask，通常一次能 finalize 好几个。快很多，accuracy 几乎不掉。

threshold 太低（比如 0.5）：一次 finalize 太多 token，有些可能猜错，后面没法纠正（因为已经 unmask 了），accuracy 掉。

GSM8K 实验显示 threshold = 0.9 是 sweet spot：throughput 从 39.1 → 101.7 tokens/s（**2.6x**），accuracy 几乎不变。

### Hierarchical Cache — 两层 cache

**第一层：Block-level cache (block 之间)**
已经 decode 完的 block 的 KV 直接存起来，后续 block 读就行。这部分跟 AR 模型的 KV cache 一模一样，因为这些 token 已经 finalize 了，不会再变。

**第二层：Sub-block cache / DualCache (block 内部)**
一个 block 内做 refinement 时，每次 forward pass 要算整个 block 的 KV。但大部分 masked 位置的 representation 变化不大（因为只有少数 token 被 unmask）。DualCache 把 prefix（已 unmask 部分）和 suffix（还 masked 部分）的 KV 都缓存，新 token unmask 时只做 incremental update，不全部重算。

Figure 6 显示：sub-block cache 对 accuracy 零影响（纯加速），在 batch size 大时（compute-bound）throughput 提升显著，batch size 小时（memory-bound）效果不大。

---

## 实验结果说人话

### 跟 AR 模型比

**1.5B model**: Fast-dLLM v2 平均 45.0 vs Qwen2.5-1.5B-Nemo-FT 44.3，基本持平

**7B model**: Fast-dLLM v2 平均 **60.3** vs Qwen2.5-7B-Nemo-FT 59.6，甚至还高一点

分 task 看：
- **Code (HumanEval, HumanEval+)**: 大幅领先。HumanEval 63.4 vs 52.4 (+11)。Code 有强 structural pattern，parallel decoding 很自然
- **GSM8K (简单 math)**: 83.7 vs 84.1，持平
- **MATH (难 math)**: 61.6 vs 72.0，**掉了 10.4 分**。Complex math reasoning 需要严格 step-by-step，parallel decoding 会干扰 reasoning chain
- **IFEval (instruction following)**: 61.4 vs 69.5，**掉了 8 分**。Instruction following 也要严格按顺序

**Intuition**: Block diffusion 适合"结构性生成"（code, 模板化文本, 简单 reasoning），不适合"严格 sequential reasoning"（complex math, 严格 instruction）。这给出了清晰的 deployment 指南。

### 跟其他 diffusion LLM 比

**vs Dream 7B**: Fast-dLLM v2 平均 60.3 vs Dream 57.6，且训练数据少 500x

**vs LLaDA 8B**: Fast-dLLM v2 大幅领先 (60.3 vs 43.3)

### 速度

Figure 5 显示在 A100 和 H100 上，各 batch size 下 Fast-dLLM v2 throughput 都超过 Qwen2.5-7B：
- A100 batch=64: **1.5x** speedup
- H100 batch=64: **1.8x** speedup（新硬件更能利用 parallelism）

加上 threshold=0.9 的 parallel decoding，GSM8K 上 **2.6x** speedup（39.1 → 101.7 tokens/s）。

---

## 理论 speedup 的直觉

生成 256 tokens:
- **AR**: 256 个 sequential forward pass
- **Block diffusion**: 8 个 block × 每 block ~5 步 refinement = ~40 步

理论极限 256/40 ≈ 6.4x，实际 2.5x。差距来自：
- 每 step 算 32 tokens 而非 1 个（更贵，但 GPU parallel 弥补）
- Refinement 步数比理论多
- Memory bandwidth 有开销

但 2.5x 已经是 diffusion LLM 第一次真正在 latency 上 beat AR，这很 significant。

---

## 这篇 paper 的 bigger picture

1. **AR 和 diffusion 不是对立的**。Block diffusion 是 design space 上的 smooth interpolation，你可以根据 task 在 AR 和 diffusion 之间滑动。

2. **Pretrained AR model 的知识可以 cheap 迁移**。Token shift + block-causal attention 让 ~99% 的计算模式保持不变，只需要学 intra-block bidirectional 这个增量能力。这个 insight 可能适用于其他 AR → non-AR 的迁移。

3. **KV cache 不兼容 diffusion 的 problem 可以用 architecture 解决**，而不是 approximate cache。Block 间保持 AR 就天然有 KV cache，不需要 DualCache 那种近似。

4. **未来 LLM 可能是 hybrid**。Reasoning task 用 AR mode (threshold=1.0)，generation task 用 diffusion mode (threshold=0.9)，dynamically switch。这篇 paper 给了这个方向的 infrastructure。

5. **Limitation**: Complex math (MATH) 和 strict instruction following (IFEval) 掉分说明 parallel decoding 跟严格 sequential reasoning 有 fundamental tension。需要更多 research 来 reconcile。

---

## 关键参考链接

- Fast-dLLM v1 (DualCache 来源): https://arxiv.org/abs/2505.22618
- BD3-LM (block diffusion 开创): https://arxiv.org/abs/2503.09573
- Dream (full diffusion, 500B tokens): https://arxiv.org/abs/2508.15487
- LLaDA (masked diffusion from scratch): https://arxiv.org/abs/2502.09992
- Qwen2.5 (base model): https://arxiv.org/abs/2412.15115
- LLaMA-Nemotron (training data): https://arxiv.org/abs/2505.00949
- SDAR (concurrent work): https://github.com/JetAstra/SDAR
- FlexAttention (PyTorch, 实现 structured mask): https://pytorch.org/blog/flexattention/
- SEDD (diffusion LM 理论基础): https://arxiv.org/abs/2310.16834
- D3PM (discrete diffusion 早期): Austin et al., NeurIPS 2021

---

# Fast-dLLM v2: 详解 Block-Diffusion LLM

## 1. 核心动机与 high-level intuition

Fast-dLLM v2 解决的核心矛盾是: **AR LLMs (e.g., Qwen2.5, LLaMA) 的 sequential decoding 与 diffusion LLMs 的 parallel decoding 之间的 trade-off**.

AR LLM 的痛点是 $O(L)$ sequential steps, GPU 利用率低. 纯 diffusion LLM (e.g., LLaDA, Dream) 看似能并行 decode 全部 tokens, 但是有几个致命问题:
1. Bidirectional attention 导致 **无法使用 KV cache**, 每次重新算整个 sequence
2. 通常需要 fixed sequence length, 灵活性差
3. Full retraining 需要海量数据 (Dream 用了 ~500B tokens)

Fast-dLLM v2 的关键 insight 是: **block-wise attention mask 实际上是 "AR-friendly" 的**, 这意味着可以保留 pretrained AR model 的绝大部分知识, 只需要少量 fine-tuning (~1B tokens, 比 Dream 少 500×) 就能把它变成一个 block-diffusion model, 同时获得 KV cache 兼容性、灵活序列长度和 intra-block 并行性.

这种设计的三层结构可以类比:
- **Inter-block**: 仍然 AR, 左到右, KV cache 友好
- **Intra-block**: bidirectional diffusion, 可以并行 refine
- **Sub-block**: confidence-based parallel decoding, 进一步加速

这种 hierarchy 同时获得了 AR 的效率和 diffusion 的并行性.

参考链接:
- Paper: https://arxiv.org/abs/2505.22618 (Fast-dLLM v1, DualCache 来源)
- BD3-LM: https://arxiv.org/abs/2503.09573
- Dream: https://arxiv.org/abs/2508.15487
- LLaDA: https://arxiv.org/abs/2502.09992
- Qwen2.5: https://arxiv.org/abs/2412.15115

---

## 2. Methodology 深入解析

### 2.1 Preliminary: Masked Diffusion 的数学框架

给定 token 序列 $x = \{x^1, x^2, \ldots, x^L\}$, 长度 $L$. 

**Forward noising process**: 在时间 $t \in (0,1)$, 每个 token 独立地以概率 $t$ 被 mask, 产生 corrupted sequence $x_t$.

**Reverse denoising model**: 学习 $p_\theta(x_0 | x_t)$, 给定 noised input 预测 original tokens.

标准 masked diffusion loss:

$$
\mathcal{L}(\theta) = -\mathbb{E}_{t, x_0, x_t} \left[ \frac{1}{t} \sum_{i=1}^{L} \mathbf{1}[x_t^i = \text{[MASK]}] \log p_\theta(x_0^i \mid x_t) \right]
$$

变量含义:
- $\theta$: model 参数 (neural network weights)
- $t$: 时间步, $t \sim \text{Uniform}(0,1)$, 控制噪声水平. $t=0$ 表示 clean, $t=1$ 表示 fully masked
- $x_0$: 原始未污染序列 (clean sequence)
- $x_t$: 时间 $t$ 的 noise 版本, 通过以概率 $t$ 独立 mask 每个 token 得到
- $x_t^i$: $x_t$ 第 $i$ 个位置的 token
- $L$: 序列长度
- $\mathbf{1}[\cdot]$: indicator function, 括号内条件成立时取 1, 否则 0
- $[MASK]$: 特殊 mask token
- $\frac{1}{t}$: 归一化系数. 因为期望上 $tL$ 个 tokens 被 mask, 除以 $t$ 等价于按 mask token 数量归一化

这里 $\frac{1}{t}$ 是因为: $\mathbb{E}[\sum_i \mathbf{1}[x_t^i = [MASK]]] = tL$, 所以 $\frac{1}{t}$ 把 loss 归一化到 per-mask-token scale.

### 2.2 Block-wise organization (训练数据组织)

给定一批 tokenized samples, 关键操作:

1. **Padding 对齐**: 每个 sequence padding 到 block size $D$ 的整数倍, 用 [MASK] tokens 填充. 这些 padding tokens 在 loss 和 gradient 中忽略.

2. **Packing**: 把 padded sequences 拼接成 long token stream, 然后切成 fixed context length $L$ 的训练序列.

3. **Natural block alignment**: 每个 packed sequence 自然地被分成 $B = L/D$ 个不重叠 blocks, 每个 block 大小 $D$.

**关键 insight**: 这种 block-aligned packing 确保 sample 边界与 block 边界对齐. 这是必要的, 因为如果 EOS of sample A 后面紧跟 BOS of sample B 在同一个 block 内, bidirectional attention 会导致 cross-sample leakage. Padding 创造了干净的 block 边界.

在 ablation 中验证: "+pad" 把 HumanEval+ 从 32.9 提升到 34.1, 把 IFEval 从 39.9 提升到 45.8, 这是相当显著的提升, 证明了 padding 策略的重要性.

### 2.3 Complementary Masking Strategy

对每个 block, 采样 binary mask $m \in \{0, 1\}^D$, 其中 $m_j = 1$ 表示位置 $j$ 被 [MASK] 替换.

**Complementary masking**: 每个训练样本 $x_0$ 被复制成两个 views:
- View 1: 使用 mask $m$, 在时间 $t$ 应用
- View 2: 使用 complement mask $\bar{m} = 1 - m$, 在时间 $1 - t$ 应用

两个 views 放在同一个 batch 中, 模型同时看到 masked 和 unmasked 的 contexts.

这个设计有几个重要作用:

**作用 1: 全 token 覆盖**. 没有 complementary mask 时, 每个 token 只以概率 $t$ 被 masked, 训练信号稀疏. 用了 complementary mask 后, 每个 token 一定在其中一个 view 中被 masked, 确保所有 tokens 都收到 supervised signal.

**作用 2: Loss normalization 简化**. 标准 masked LM loss 有 $\frac{1}{t}$ 归一化, 但用 complementary mask 时, 两个 views 总共贡献 $L$ 个 masked tokens (因为 $tL + (1-t)L = L$), 归一化常数自然消失. 这避免了 $t \to 0$ 时的数值不稳定.

具体 loss (complementary mask 版本):

$$
- \left[ \sum_{i=1}^{L} \mathbf{1}[x_t^i = \text{[MASK]}] \log p_\theta(x_0^i | \ldots) \right] + \left[ \sum_{i=1}^{L} \mathbf{1}[x_{1-t}^i = \text{[MASK]}] \log p_\theta(x_0^i | \ldots) \right]
$$

由于 complementary mask, 两个 views 加起来正好覆盖所有 $L$ 个 tokens, 不需要 $\frac{1}{t}$ normalization.

**作用 3: 训练-推理 consistency**. 推理时每个 block 总是 fully masked 开始 decode 的, 训练时见到 fully masked 状态很重要. Complementary mask 确保模型见过各种 mask 比例.

Ablation 结果: "+ pad + CM" 比 "+ pad" 平均提升 +2.8 (42.2 → 45.0), 比 naive 高 +3.7.

### 2.4 Token Shift for Prediction (关键的 AR-preserving trick)

这是 Fast-dLLM v2 最巧妙的设计之一. 

**问题**: 标准 masked LM 在位置 $i$ 预测 $x_i$, 使用 hidden state at position $i$. 但 pretrained AR model 训练时, 在位置 $i$ 预测 $x_{i+1}$ (next-token prediction). 这两个任务 mismatch, 导致 adapter 时 representation drift.

**Solution**: Token shift strategy. 如果 $x_i$ 被 mask, 用位置 $i-1$ 的 hidden state 来预测 $x_i$:

$$
p_\theta(x_0^i | x_t) = \text{softmax}(W \cdot h_{i-1})
$$

其中 $h_{i-1}$ 是 transformer 在位置 $i-1$ 的 hidden state, $W$ 是 output projection.

这与 causal LM 的 next-token prediction 完全一致: 位置 $i-1$ 的 hidden state 预测 $x_i$. 这意味着 pretrained AR model 的 representation 直接 reuse, 不需要"重新学"如何预测 next token, 只需要学习:
1. 处理 [MASK] tokens 的 bidirectional attention (intra-block)
2. 接收来自 future positions (intra-block) 的 bidirectional context

这是数据效率的核心原因: 模型只需学习"如何利用 intra-block bidirectional context 增强 next-token prediction", 而不是从头学习如何 generate text.

参考 Dream 的方法: Dream 用 bidirectional attention 处理整个序列, 预测每个位置的 token, 与 pretrained AR model 完全 mismatch, 需要 500B tokens 才能 learn from scratch.

### 2.5 Training Objective

最终的 block-wise loss:

$$
\mathcal{L}_{\mathrm{block}}(\theta) = -\mathbb{E}_{x, m} \left[ \sum_{i=1}^{L} \mathbf{1}[x_t^i = \text{[MASK]}] \log p_\theta(x_0^i \mid x_{<i}, x_{\mathrm{block}(i)}) \right]
$$

变量含义:
- $x_{<i}$: 来自更早 blocks 的 clean tokens (causal context, 类似 AR prefix)
- $x_{\mathrm{block}(i)}$: 包含位置 $i$ 的整个 block 的所有 tokens (包括 masked 和 unmasked 的, 提供 intra-block bidirectional context)

这个 formulation 美妙之处: 同时具有 AR conditioning ($x_{<i}$, 保留 pretrained model 的能力) 和 diffusion refinement ($x_{\mathrm{block}(i)}$, 提供 intra-block context).

注意: token shift 意味着虽然 loss 写成 $p_\theta(x_0^i | \ldots)$, 实际计算时是用 position $i-1$ 的 logit 来预测 $x_0^i$.

### 2.6 Attention Mask 设计 (核心技术细节)

这是论文最精妙的部分. 训练时, 把 noised sequence $x_t$ 和 clean sequence $x_0$ 拼接, 总长度 $2L$, 然后应用 hybrid attention mask $\mathcal{M}_{\mathrm{full}} \in \{0, 1\}^{2L \times 2L}$.

矩阵结构:

$$
\mathcal{M}_{\mathrm{full}} = \begin{bmatrix} \mathcal{M}_{BD} & \mathcal{M}_{OBC} \\ 0 & \mathcal{M}_{BC} \end{bmatrix}
$$

行表示 query, 列表示 key. 上半部分是 $x_t$ 的 queries, 下半部分是 $x_0$ 的 queries. 左半部分是 $x_t$ 的 keys, 右半部分是 $x_0$ 的 keys.

**四个 sub-masks 详细解释**:

#### (1) $\mathcal{M}_{BD}$ (Block-Diagonal) — block 内 bidirectional

$$
[\mathcal{M}_{BD}]_{ij} = \begin{cases} 1 & \text{if } i, j \text{ belong to the same block} \\ 0 & \text{otherwise} \end{cases}
$$

作用: noised tokens 在同一 block 内可以互相 attend (bidirectional). 这是 diffusion 的关键 — 每个 masked token 可以从 block 内其他 (可能未 mask 的) tokens 获得 context.

注意: 跨 block 的 noised tokens 不互相 attend, 保持 inter-block isolation.

#### (2) $\mathcal{M}_{OBC}$ (Offset Block-Causal) — noised attend to clean prefix blocks

$$
[\mathcal{M}_{OBC}]_{ij} = \begin{cases} 1 & \text{if } j \text{ is in a block before } i \\ 0 & \text{otherwise} \end{cases}
$$

作用: noised tokens 可以 attend 到 clean sequence 中 **更早的 blocks** (但不是同 block). 这让 noised block 可以从 clean prefix 中获取 causal context, 类似 AR 的 prefix conditioning.

"Offset" 含义: noised tokens 的 block $b$ 只能 attend clean tokens 的 blocks $< b$, 即存在一个 block offset.

#### (3) $\mathcal{M}_{BC}$ (Block-Causal) — clean tokens 的标准 AR attention

$$
[\mathcal{M}_{BC}]_{ij} = \begin{cases} 1 & \text{if } j \text{ is in the same or an earlier block as } i \\ 0 & \text{otherwise} \end{cases}
$$

作用: clean tokens 之间保持标准的 block-causal attention, 每个 clean token 可以 attend 到同 block 和更早 blocks 的所有 tokens. 这等价于 pretrained AR model 的训练分布, 保留其表征能力.

#### (4) Lower-left block 是 0 — clean tokens 不受 noised 干扰

这个 zero block 确保 clean sequence 的 hidden states 不被 noised sequence 污染. 这非常重要: clean sequence 提供给 noised tokens 一个 "anchor" — 它的 representation 与 pretrained model 完全一致, 给 noised tokens 提供高质量的 context.

**为什么这个设计 data-efficient**:

观察: $\mathcal{M}_{BC}$ 部分 (clean sequence 自注意力) 完全等价于 pretrained AR model 的 attention pattern. 这意味着 clean sequence 的 hidden states 与 pretrained model 完全一致, 没有 distribution shift. 模型只需学习:
- $\mathcal{M}_{BD}$ 部分: 如何处理 intra-block bidirectional (新能力)
- $\mathcal{M}_{OBC}$ 部分: 如何从 clean prefix 提取 context to noised tokens (类似 cross-attention, 但 reused pretrained representations)

由于大部分 transformer 计算仍是 standard AR, fine-tuning 数据需求大幅降低.

实现上用 **flex-attention** (PyTorch 2.5+ feature) 高效实现这种结构化 mask, 避免了 dense attention 的浪费.

### 2.7 Inference-time Attention Mask

推理时简化设计 (Figure 7b):

- 之前已 decode 的 blocks $x_0^{<b}$ 被 cache (KV cache), 作为 frozen prefix
- 当前 noised block $x_t^b$ 是唯一活跃计算的
- 当前 block 内部 bidirectional attention (类似 $\mathcal{M}_{BD}$)
- 当前 block 对所有之前 blocks 的 clean tokens causal attention (类似 $\mathcal{M}_{OBC}$)

这意味着每次 decode 一个 block 时, attention 计算只涉及:
- 当前 block 的 $D$ 个 tokens (self-attention)
- 之前所有 blocks 的 cached keys/values (cross-attention to prefix)

总计算复杂度: $O(D^2 + D \cdot L_{\text{prefix}})$ per block, 比 full bidirectional attention $O(L^2)$ 高效得多.

---

## 3. Inference Pipeline 详细流程

### 3.1 Block-wise AR Decoding with Caching

整体流程:

```
Initialize: x = [prompt] + [MASK] * (target_length)
For each block b in 1, 2, ..., B:
    Cache: KV for blocks 1, ..., b-1 (read-only prefix context)
    Iteratively refine block b:
        Forward pass with current block + cached prefix
        Compute confidence for each masked position
        Unmask tokens above threshold
        Repeat until all tokens unmasked or max iterations
    Finalize block b, add to cache
```

### 3.2 Parallel Refinement (Confidence-based)

来自 Fast-dLLM v1 (Wu et al., 2025) 的技术:

每个 denoising step:
1. 对当前 block 的所有 masked 位置, 模型输出预测分布
2. 对每个 masked 位置 $i$, 取 $\max_k p_\theta(x_i = k)$ 作为 confidence
3. 如果 confidence > threshold $\tau$, 这个 token 被 "finalize" (unmask)
4. 剩余 masked tokens 进入下一轮 refinement

**Threshold trade-off** (Figure 4, GSM8K):
- $\tau = 1.0$: 标准 diffusion (full denoising), 全部 tokens 在最后一步 unmask, 速度最慢但精度最高
- $\tau = 0.9$: 2.6x speedup, accuracy 仅 marginal drop (GSM8K 几乎不变)
- $\tau$ 越低 → 越多 tokens 提前 finalize → 越快但越 noisier

论文实验显示 $\tau = 0.9$ 是 sweet spot.

### 3.3 DualCache (Hierarchical KV Cache)

这是核心加速机制. 分两层:

**Layer 1: Block-level cache** (inter-block)
- 已 decode 完成的 blocks 的 KV 直接 cache
- 后续 blocks 处理时, 直接 reuse 这些 KV, 不重算
- 这部分 cache 是精确的 (clean tokens, 与 AR KV cache 完全一致)

**Layer 2: Sub-block cache** (intra-block, DualCache)
- 在一个 block 内做 parallel decoding 时, 多次 refinement 需要多次 forward pass
- DualCache 同时 cache prefix (已 unmask 部分) 和 suffix (still masked 部分) 的 KV
- Suffix cache 的关键 insight: 大部分 masked 位置的 representation 在 refinement 过程中变化不大, 可以 reuse. 当新 token 被 unmask 时, 只需 incremental update, 不需要 full recompute

DualCache 与 standard AR KV cache 的区别:
- AR KV cache: prefix tokens 是确定的, 永不变化, cache 直接 reuse
- Diffusion: 当前 block 中, masked tokens 的 representation 在每次 refinement 后变化 (因为它们的邻居被 unmask 了), 但变化是 incremental 的
- DualCache 利用这个 incremental 变化, 避免每次 refinement 都重算整个 block 的所有 layers 的 KV

具体实现细节见 Fast-dLLM v1 paper: https://arxiv.org/abs/2505.22618

### 3.4 Batch Decoding with Padding

支持 batch generation:
- 不同 sequence 长度可能不同, 但 block-aligned
- 右 pad [MASK] tokens 让长度是 $D$ 的倍数
- Batch 内所有 sequences 同时 decode 下一个 block, 不论各自还剩多少 real tokens
- 这保证了 GPU 的高效 batching

---

## 4. Experimental Results 深入分析

### 4.1 Training Setup

- Base model: Qwen2.5-Instruct 1.5B / 7B
- Training data: LLaMA-Nemotron post-training dataset (https://arxiv.org/abs/2505.00949)
- Hardware: 64 NVIDIA A100 GPUs, DeepSpeed Zero-3
- Context length: 2048, Batch size: 256
- 每步处理 tokens: $256 \times 2048 = 524,288$

**1.5B model**:
- LR: $2 \times 10^{-5}$, 6000 steps, AdamW, 500 steps warmup
- 总 tokens: $6000 \times 524,288 \approx 3.15B$
- 训练时间: ~8 hours

**7B model**:
- LR: $1 \times 10^{-5}$, 2500 steps
- 总 tokens: $2500 \times 524,288 \approx 1.31B$
- 训练时间: ~12 hours

对比 Dream: 500B tokens for fine-tuning. Fast-dLLM v2 只需 ~1B, 减少 500×.

### 4.2 Main Results (Table 1)

**1.5B Scale 对比**:

| Model | HumanEval+ | MBPP+ | GSM8K | MATH | IFEval | MMLU | GPQA | Avg |
|-------|-------------|--------|---------|-------|---------|-------|-------|-----|
| LLaMA-3.2 1.2B | 31.1 | 29.4 | 23.8 | 58.9 | 44.4 | 24.1 | 35.9 | (avg) |
| SmolLM2 1.7B | 28.7 | 46.0 | 21.1 | 55.1 | 49.1 | 29.2 | 40.7 | (avg) |
| Qwen2.5-1.5B | 37.2 | 41.3 | 57.0 | 46.8 | 41.2 | 54.6 | 30.6 | 44.3 |
| Qwen2.5-1.5B-Nemo-FT | 33.5 | 44.4 | 58.5 | 43.5 | 39.4 | 58.1 | 31.0 | 44.3 |
| LLaDA 1.5B | 40.2 | 41.3 | 62.0 | 38.1 | 47.0 | 55.1 | 27.7 | 45.0 |
| **Fast-dLLM v2 1.5B** | **40.2** | **41.3** | **62.0** | **38.1** | **47.0** | **55.1** | 27.7 | **45.0** |

Fast-dLLM v2 1.5B 平均 45.0, 比 Qwen2.5-1.5B-Nemo-FT (44.3) 略高, 与 LLaDA 1.5B 相当.

**7B Scale 对比**:

| Model | HumanEval | HumanEval+ | MBPP | MBPP+ | GSM8K | MATH | IFEval | MMLU | GPQA | Avg |
|-------|------------|------------|-------|--------|---------|-------|---------|-------|-------|-----|
| LLaDA-1.5 8B | 35.4 | 31.7 | 31.5 | 28.6 | 78.6 | 26.6 | 59.9 | 65.5 | 31.8 | 43.3 |
| LLaDA-MoE 8B | 52.4 | - | 42.8 | - | 83.3 | 42.6 | 58.2 | 66.0 | 36.9 | - |
| Dream 7B | 61.6 | - | 70.0 | - | 82.4 | 58.7 | 59.3 | 67.2 | - | 57.6 |
| Qwen2.5-7B | 51.2 | 47.6 | 57.7 | 49.5 | 71.4 | 73.3 | 70.8 | 68.7 | 33.5 | 58.2 |
| Qwen2.5-7B-Nemo-FT | 52.4 | 48.2 | 57.1 | 50.0 | 84.1 | 72.0 | 69.5 | 68.6 | 34.2 | 59.6 |
| **Fast-dLLM v2 7B** | **63.4** | **58.5** | **63.0** | **52.3** | 83.7 | 61.6 | 61.4 | 66.6 | 31.9 | **60.3** |

**关键观察**:

1. **Avg score**: Fast-dLLM v2 7B (60.3) > Qwen2.5-7B-Nemo-FT (59.6) > Dream (57.6). Fast-dLLM v2 实际上是最高平均分.

2. **Code generation 大幅领先**: HumanEval 63.4 vs 52.4 (Qwen Nemo-FT), HumanEval+ 58.5 vs 48.2. Code 具有强 structural regularity, parallel decoding 能很好利用这点.

3. **MATH 明显下降**: 61.6 vs 72.0 (Qwen Nemo-FT). 数学推理需要严格的 step-by-step chain, parallel/block decoding 可能干扰 reasoning chain. 这是一个 trade-off.

4. **GSM8K 几乎持平**: 83.7 vs 84.1. GSM8K 的 reasoning 比 MATH 简单, parallel decoding 仍能很好处理.

5. **IFEval 也下降**: 61.4 vs 69.5. Instruction following 需要严格遵守 user instruction, 可能也 benefit from strict AR.

**Intuition**: Block-diffusion 在 "结构性 generation" (code, 简单 reasoning) 上有优势, 在 "严格 sequential reasoning" (MATH, IFEval) 上有劣势. 这暗示 fast-dLLM v2 更适合 production deployment (latency-sensitive) 而非 competitive math solving.

### 4.3 Throughput (Figure 5)

**GSM8K, threshold 0.9, sub-block cache**:

| GPU | Batch Size | Qwen2.5-7B | Fast-dLLM v2 | Speedup |
|-----|------------|-------------|---------------|---------|
| A100 | 1 | (baseline) | higher | ~1.5x |
| A100 | 4 | (baseline) | higher | ~1.5x |
| A100 | 64 | (baseline) | higher | **1.5x** |
| H100 | 64 | (baseline) | higher | **1.8x** |

关键发现:
- A100 上 1.5x speedup (max)
- H100 上 1.8x speedup — 新硬件更好地利用 parallelism
- Speedup 在大 batch 时更显著, 说明 block-diffusion 在 compute-bound regime 更有优势

**With threshold 0.9 on GSM8K**: throughput 从 39.1 → 101.7 tokens/s, **2.6x speedup** with marginal accuracy drop. 这是 paper headline 的 2.5x speedup 来源.

### 4.4 Ablation Studies 深度分析

#### (1) Token Shift Strategies (Table 2)

| Method | HumanEval+ | MBPP+ | GSM8K | MATH | IFEval | MMLU | GPQA | Avg |
|--------|------------|--------|---------|-------|---------|-------|-------|-----|
| Naive token shift | 32.9 | 38.6 | 59.0 | 37.3 | 39.9 | 52.9 | 27.9 | 41.3 |
| + pad | 34.1 | 38.4 | 60.1 | 37.0 | 45.8 | 53.5 | 27.7 | 42.2 |
| + pad + CM | **40.2** | **41.3** | 62.0 | 38.1 | 47.0 | 55.1 | 27.7 | **45.0** |

**+ pad 的作用**: 
- IFEval 提升 +5.9 (39.9 → 45.8), 这是最显著的提升
- 原因: 没有 padding 时, sample 边界处 bidirectional attention 让模型 attend 到下一个 sample, 这种"泄露"破坏了 instruction-following 的 clean context
- Padding 创建 clean block boundaries, 让 attention 在 sample 内部进行

**+ CM 的作用**:
- HumanEval+ 提升 +6.1 (34.1 → 40.2), MBPP+ 提升 +2.9 (38.4 → 41.3)
- Code 任务的提升最显著, 因为 code 有强 structural dependencies, 每个 token 都需要 supervised
- CM 确保 all tokens 都见到 masked 和 unmasked 两种 context

#### (2) Sub-Block Size (Table 3)

| Sub-Block Size | 2 | 4 | 8 | 16 | 32 |
|----------------|-----|-----|-----|-----|-----|
| GSM8K | 62.8 | 61.8 | 62.0 | 61.3 | 60.2 |
| HumanEval | 42.7 | 43.3 | **43.9** | 39.6 | 38.4 |
| HumanEval+ | 39.6 | 40.2 | **40.2** | 36.0 | 34.8 |

Sub-block size 8 最优平均. 

**Intuition**: Sub-block 控制 intra-block parallel decoding 的 granularity.
- Size 2: 每个 sub-block 太小, parallel gain 有限, 但每个 token 充分 refine
- Size 8: sweet spot, 8 tokens 一起 decode, parallelism 与 accuracy 平衡
- Size 32 (整个 block): 全部 parallel, 但 token 间 conditioning 不充分

注意 GSM8K 在 size 2 时最高 (62.8), 因为 math reasoning 对每个 token 的 accuracy 非常敏感. HumanEval 在 size 8 时最高, 因为 code 的 structural patterns 让 parallel decoding 容易.

#### (3) Block Size at Inference (Table 4) — Mismatch 时性能下降

| Block Size | 2 | 4 | 8 | 16 | 32 |
|------------|-----|-----|-----|-----|-----|
| GSM8K | 53.2 | 56.8 | 58.5 | 59.7 | 60.2 |
| HumanEval | 37.8 | 43.3 | 43.3 | 38.4 | 38.4 |
| HumanEval+ | 34.1 | 39.0 | 39.6 | 34.1 | 34.8 |

训练时 block size = 32. 推理时 mismatch 会导致严重 degradation. GSM8K 从 60.2 (block=32) 降到 53.2 (block=2).

**原因**: Model 训练时学到了 block size 32 的 attention pattern, 推理时 block size 改变意味着 attention 结构变化, 引起 distribution shift. 这就是为什么 sub-block 设计重要 — 它不改变 training block size, 只是控制 intra-block parallel decoding 的 granularity.

#### (4) Sub-Block Cache 效果 (Figure 6)

(a) Accuracy: cache 不影响 accuracy (cache 是纯 efficiency 优化)
(b) Throughput: 
- Small batch (e.g., batch=1): cache 效果不显著 (memory bandwidth underutilized)
- Large batch (e.g., batch=32): cache 显著提升 throughput (compute-bound regime)
- Larger sub-block size → 更高 throughput (更多 parallelism)

---

## 5. 与 Related Work 的对比

### 5.1 vs. BD3-LM (Arriola et al., 2025)

BD3-LM 是 block diffusion 的开创性工作, 但仅在 small-scale 验证. Fast-dLLM v2 把它 scale 到 7B LLM 并适配到现代 LLM tasks. 关键区别:
- BD3-LM: trained from scratch, small scale
- Fast-dLLM v2: adapt pretrained AR LLM, 7B scale, data-efficient

### 5.2 vs. Dream (Ye et al., 2025)

Dream 也是从 Qwen2.5-7B 适配为 diffusion LLM, 但:
- Dream: full-attention diffusion, 需要 500B tokens fine-tuning
- Fast-dLLM v2: block-wise attention, 只需 1B tokens, 500× 数据减少

### 5.3 vs. SDAR (Cheng et al., 2025)

Concurrent work, 也 fine-tune AR model 为 block diffusion. Fast-dLLM v2 区别在于 data efficiency (1B tokens) 和 hierarchical caching.

### 5.4 vs. D2F (Wang et al., 2025)

D2F 是 distillation approach, 把 large diffusion LLM distill 成 block diffusion. Fast-dLLM v2 直接 adapt AR model, 不需要 teacher diffusion model.

### 5.5 vs. Set Block Decoding (Gat et al., 2025)

Set Block Decoding 在 single architecture 内结合 NTP 和 MATP. Fast-dLLM v2 用 pure block diffusion formulation.

### 5.6 Diffusion LLM 加速方法对比

| Method | Approach |
|--------|----------|
| Fast-dLLM (v1) DualCache | Prefix + suffix KV cache |
| dKV-Cache | Delayed caching |
| dLLM-Cache | Adaptive partial response cache |
| Sparse-dLLM | Attention-based token dropping |
| DPad | Fixed-size suffix window |
| EB-Sampler | Entropy bounded unmasking |
| Dimple | Confident decoding |
| WINO | Draft-and-verify |
| SlowFast Sampling | Adaptive stage switching |
| LaViDa | Timestep shift |
| Prophet | Dynamic refinement decision |

Fast-dLLM v2 集成了 v1 的 DualCache + 新的 block-level cache, 形成 hierarchical caching.

---

## 6. 关于 Inference Latency 的直觉分析

考虑一个 7B model 生成 256 tokens:

**AR decoding (Qwen2.5-7B)**:
- 256 sequential forward passes
- KV cache 让每步只算新 token 的 KV
- 每步 ~$O(L \cdot d)$ where $L$ 是 prefix length

**Block diffusion (Fast-dLLM v2)**:
- 8 blocks of size 32
- 每个 block 需要 ~5-10 refinement steps (with parallel decoding threshold 0.9, 通常更少)
- 但每个 refinement step 算整个 block (32 tokens) 的 KV
- Block cache 让 prefix 不重算
- 总 steps: $8 \times 5 = 40$ (vs 256 for AR)

**Speedup 来源**:
1. Fewer sequential steps (40 vs 256) — 6.4× 理论极限
2. Each step 算 32 tokens in parallel — GPU 充分利用
3. Block cache 避免 prefix 重算
4. DualCache 让 intra-block refinement 高效

实际 speedup 2.5x (vs 6.4x 理论极限), 因为:
- Refinement steps > 1 per block
- Overlap of parallel decoding 有时需要 more iterations
- Memory bandwidth 不是 free

---

## 7. Open Questions 与 Limitations

1. **MATH performance 下降**: 61.6 vs 72.0 (Nemo-FT). Parallel decoding 可能干扰 multi-step reasoning. 可能的解决方向: 在 reasoning tasks 上用 threshold=1.0 (full denoising), 在 generation tasks 上用 threshold=0.9.

2. **Block size 32 是否 optimal**: Long-form generation 可能需要更大 blocks (longer dependencies). 但更大 blocks 意味着每个 refinement step 更昂贵.

3. **Reasoning chains 与 parallel decoding 的兼容性**: Complex reasoning (MATH) 需要 strict sequential dependencies, 与 intra-block bidirectional 矛盾. 是否需要 task-adaptive decoding strategy?

4. **Cache memory 开销**: Block-level cache + DualCache + sub-block cache 的总 memory footprint. 对于 long context generation, 这可能成为 bottleneck.

5. **Training data diversity**: 只用 LLaMA-Nemotron SFT data. Domain shift (e.g., 长篇 reasoning, 多语言) 可能需要 more fine-tuning data.

6. **Speculative decoding integration**: Block diffusion 已经是某种 "speculative" 形式, 是否能进一步与 speculative decoding (e.g., Medusa, EAGLE) 结合?

7. **Long context 支持**: 2048 context length 训练. Long context (e.g., 32K+) 是否能 well-adapt? Block structure 应该天然支持, 但 attention pattern 需要验证.

---

## 8. 总结: 为什么这个 work 重要

Fast-dLLM v2 的关键贡献:

1. **Practical data efficiency**: 1B tokens (vs 500B for Dream) 让 AR → diffusion 适配变得可行. 这可能改变 LLM deployment landscape — 现有 AR LLM 可以 cheaply 转换成 fast diffusion LLM.

2. **Architecture preservation**: Token shift + block-causal attention 让 pretrained model 的 representation 大部分保留, 只需学习 intra-block bidirectional context integration.

3. **Hierarchical caching**: Block-level + sub-block DualCache 解决了 diffusion LLM 的 KV cache 不兼容问题, 让 diffusion 在 latency 上第一次真正 competitive with AR.

4. **Real-world scale validation**: 7B model, 现代 benchmarks, 实际 2.5x speedup — 这把 block diffusion 从 academic curiosity 推向 production-ready.

5. **Trade-off characterization**: 在 code, simple reasoning 上 on-par or better than AR; 在 complex math, strict instruction following 上有 trade-off. 这给出了清晰的应用场景指南.

这个 work 暗示: AR LLM 与 diffusion LLM 不是 mutually exclusive, 而是 design space 上的两个极端. Block diffusion 提供了一个 smooth interpolation, 让我们可以根据 task 选择 optimal point. 未来 LLM 可能是 hybrid — AR for reasoning, block diffusion for generation, dynamically switched.

参考链接汇总:
- Fast-dLLM v2 GitHub (代码会公开): 见 paper "Links: Github Code | Project Page"
- Fast-dLLM v1: https://arxiv.org/abs/2505.22618
- BD3-LM: https://arxiv.org/abs/2503.09573
- Dream: https://arxiv.org/abs/2508.15487
- LLaDA: https://arxiv.org/abs/2502.09992
- SDAR: https://github.com/JetAstra/SDAR
- Qwen2.5: https://arxiv.org/abs/2412.15115
- LLaMA-Nemotron: https://arxiv.org/abs/2505.00949
- FlexAttention (PyTorch): https://pytorch.org/blog/flexattention/
- D2F: https://arxiv.org/abs/2508.09192
- Set Block Decoding: https://arxiv.org/abs/2509.04185
- EB-Sampler: https://arxiv.org/abs/2505.24857
- dKV-Cache: https://arxiv.org/abs/2505.15781
- dLLM-Cache: https://arxiv.org/abs/2506.06295
- Sparse-dLLM: https://arxiv.org/abs/2508.02558
- DPad: https://arxiv.org/abs/2508.14148
- WINO: https://arxiv.org/abs/2507.18578
- LaViDa: https://arxiv.org/abs/2505.16839
- Prophet: https://arxiv.org/abs/2508.19982
- SlowFast Sampling: https://arxiv.org/abs/2506.10848
- Temporal Self-Consistency Voting: https://arxiv.org/abs/2508.09138
- LLaDA 1.5: https://arxiv.org/abs/2505.19223
- D3PM: Austin et al., NeurIPS 2021
- SEDD: https://arxiv.org/abs/2310.16834
- AR-Diffusion: Wu et al., NeurIPS 2023
- SSD-LM: https://arxiv.org/abs/2210.17432
- Diffusion Forcing: Chen et al., NeurIPS 2024
- CausVid: Yin et al., CVPR 2025
