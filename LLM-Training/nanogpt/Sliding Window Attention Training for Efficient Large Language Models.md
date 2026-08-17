---
source_pdf: Sliding Window Attention Training for Efficient Large Language Models.pdf
paper_sha256: 5e8ef4374a00359432c42099578b17b27d7b9c293518662a0aa4bc9715e13dbc
processed_at: '2026-08-12T07:49:11-07:00'
target_folder: LLM-Training/nanogpt
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用大白话讲讲 SWAT

## 一句话概括

这篇 paper 说：**Sliding Window Attention 不能只在推理的时候用，得从训练就开始用，而且得把 softmax 换成 sigmoid 才训得起来**。

---

## 问题是什么

Transformer 的 attention 是 $O(N^2)$ 的，序列一长就爆。大家想了很多办法，比如 sliding window（每个 token 只看附近 $\omega$ 个 token），把复杂度压成 $O(N\omega)$。

但问题是：**现有的 LLM（Llama、Qwen 这些）虽然结构上支持 sliding window inference，但一用效果就崩**。Paper 里 Figure 2 画得很清楚——Llama-2-7b 训练时 context 是 4096，你 inference 时把 window 设成 1024，perplexity 直接飞上天。

为什么崩？两个原因：

### 原因一：Attention Sink

LLM 有个怪毛病，特别喜欢盯着 sequence 开头的几个 token 看，把大量 attention weight 都堆在 "第一个 token" 上。这就是 attention sink。

Paper 有个很漂亮的观察（Figure 3）：第一个 token 的 embedding variance 特别大，跟 attention score 的分布高度相关。说明什么？说明 softmax 这个 normalization 操作，会在训练时隐式地把 positional information 塞进 token embedding 的 variance 里。模型其实偷偷学会了 "靠 variance 大小来判断位置"。

你一滑动窗口，开头的 token 被 evict 掉了，模型就懵了——它赖以判断位置的 "anchor" 没了。

### 原因二：Softmax 太 "稀疏"

Softmax 是个 winner-takes-all 的东西。举个例子：

```
[1.5, 5.0, 2.4, 0.5, 1.3] → softmax → [0.03, 0.88, 0.07, 0.01, 0.02]
```

5.0 那个直接吃了 88% 的概率，其他全被压扁。Appendix A 用 extreme value theory 证明：如果 attention scores 近似高斯分布，最大值会比平均值高 $\sigma\sqrt{2\ln L}$，导致其他 token 的 attention weight 指数级 vanish。

这在 full attention 下是优点（selective focus 嘛），但在 sliding window 下是灾难——window 内本来就没几个 token，你再稀疏化，历史信息全丢了。窗口一滑走，老 token 被 evict，它本来应该把信息 compress 进剩余 token 的，但 softmax 不让 compress，全 kill 掉了。

---

## SWAT 怎么解决

三个 component 组合：

### 1. Sigmoid 替代 Softmax

$$\text{Attention}(Q,K,V) = \sigma\left(\frac{QK^T}{\sqrt{d}}\right)V$$

Sigmoid 的好处：每个 attention weight 独立计算，没有 shared denominator，不会 winner-takes-all。window 内所有 token 的信息都能被 retain 进 output。

直觉：softmax 像 selector，逼你选一个赢家；sigmoid 像 accumulator，让每个 token 按 relevance 独立贡献。SWA 训练需要把老信息 compress 进当前 token，sigmoid 的 dense 特性正好 enable 这种 compression。

### 2. Balanced ALiBi

但 sigmoid 太 dense 了，所有 weight 都在 0.5 附近，token 之间没区分度，信息 overload。所以需要加 positional bias。

原始 ALiBi 只用 negative slope（越远的 token attention 越低）。SWAT 提出双向 balanced 版本——一半 head 用 negative slope（关注近期），一半 head 用 positive slope（保留远期历史）。

$$s_k = \begin{cases} -2^{-k} & \text{forward-looking heads} \\ 2^{-k} & \text{backward-looking heads} \end{cases}$$

直觉：单向 ALiBi 假设 "近的永远比远的重要"，这在 full attention 下合理。但 SWA 下，远 token 已经被 window evict 了，剩下的都算 "近的"。你需要一部分 head 专门去 "记住" 历史，一部分 head 去 "关注" 当前，分工合作。

### 3. RoPE

换了 sigmoid 之后，softmax normalization 带来的 implicit position 没了，训练不稳定。ALiBi 的 positional signal 又比较弱。所以加 RoPE 补 explicit position encoding。

最终公式：

$$\text{Attention}_m = \sum_{n=m-\omega+1}^{m} \sigma\left(\frac{(R_{\Theta,m} q_m)^T (R_{\Theta,n} k_n)}{\sqrt{d_k}} + s \cdot (m-n)\right) v_n$$

三个东西各司其职：sigmoid 保证信息 dense retention，balanced ALiBi 提供 directional differentiation，RoPE 提供 explicit position。

---

## 最关键的 Insight：SWA 训练逼模型学会 "压缩"

Section 3.1 这段我觉得是 paper 最深刻的地方。

Vanilla Transformer 训练时，序列长度 < window size，每个新 token 能看到所有 past token。模型学的是 **"extraction"**——从全 context 里提取有用信息，softmax 强化这种 selective extraction。

SWA 训练时，窗口不断滑动，老 token 被 discard。但 Transformer 的 upper layers 里，新 token 的 embedding 通过 residual + attention 仍然 retain 了一部分老 token 信息。所以模型被逼着学会 **"compression"**——把历史信息压进固定大小的 hidden state，让它 survive window eviction。

这本质上让 Transformer 训练成了一个 pseudo-RNN，但 architecture 还是标准 Transformer。信息传播范围理论上是 $1 + (\omega-1) \cdot L$（$\omega$ 是 window size，$L$ 是层数）。

---

## 实验结论

### Overall（Table 1）

760M 参数，8 个 commonsense reasoning benchmark：
- SWAT(-) 拿了 SOTA，avg 51.85%，超过 Titans 的 51.56%
- SWAT(-) 在 short-text 任务（PIQA、HellaSwag）最好
- SWAT(-+) 在 BoolQ（需要历史 context 的 QA）最好，证明 balanced 配置的价值
- SWAT(+) 单独 backward-looking 效果差，说明光记历史不 focus 当前不行

### SWA Training 本身就有效（Table 2）

这个实验证明 SWA training（不依赖 sigmoid/ALiBi）就有价值：
- Vanilla Transformer 只在 eval length = training length 时最优，长度一变就崩
- SWA-trained 模型在任意 eval length 上都稳定，甚至更长更好
- Sliding Window B（training window 1024, training length 4096）在 eval 16384 上全面碾压 Vanilla B

### Ablation（Table 3）

- No.1 vs No.2：直接换 sigmoid 在 vanilla Transformer 上 catastrophic failure（avg 10.57 vs 5.51），证明 sigmoid 不能单独用
- No.10 vs No.11：加 ALiBi 后 sigmoid 能 work（2.74 vs 4.33）
- No.8（Sigmoid + balanced ALiBi + RoPE）最优，avg 2.51

---

## 跟其他工作的关系

SWAT 本质上是把 Transformer 训练成 pseudo-RNN，但保留标准 architecture。跟 Mamba、RWKV、RetNet 这些 SSM/linear RNN 不同，SWAT：
- 能复用所有 Transformer training infra（FlashAttention 等）
- 不需要 custom CUDA kernel
- Inference 可解释性更好

跟 StreamingLLM（inference 时 fix attention sink）不同，SWAT 从 training 根除 sink，更 fundamental。

跟 Apple 的 sigmoid attention paper（Ramapuram et al. 2025）思路一致，但 SWAT 的 contribution 是把它跟 SWA training + balanced ALiBi + RoPE 组合起来，解决 long-context 问题。

---

## 局限性

1. **Maximum attention distance 有限**：$\omega \times L$，比如 window 1024、32 层，最大 range 约 32K，超长序列必然丢信息
2. **Hyperparameter sensitive**：window size、depth、slope 配置都影响大，没有系统 search guidance
3. **340M 时 perplexity 比 baseline 差**，760M 才反超，说明 sigmoid attention 可能需要更大 scale 才能克服 optimization difficulty

---

## Intuition 总结

这篇 paper 的核心 message 我觉得是：**你想要 efficient long-context，不能只在 inference 端做 trick，得从 training 就让模型学会 "在有限带宽下压缩信息"**。SWA training 提供了这个 learning pressure，sigmoid + balanced ALiBi + RoPE 提供了让这个 learning 能 succeed 的 architectural support。

简单说就是：**别让模型训练时养成 "全程 context 都给我看" 的坏习惯，得从小训练它 "边看边忘边压缩" 的能力**。

---

# SWAT: Sliding Window Attention Training 深度解析

## 1. Core Problem: Training-Inference Gap in SWA

这篇 paper 要解决的核心问题是 Sliding Window Attention (SWA) 在现有 LLM 中 training 和 inference 之间的 gap。现有开源 LLM (Llama-2-7b, Llama-3.1-8B, Qwen2-7B, Mistral-7B) 虽然结构上 support SWA inference，但实际效果很差。

Figure 2 的实验非常直观：在 PG-19 test set 上，固定 window size (例如 Llama-2-7b 设为 1,024)，当 evaluation length 超过 training length 时 perplexity 急剧上升。这说明 Transformer inherently 学习了 training length specific 的 contextual pattern，无法 extend 到 variable-length text。

这个 failure 的 root cause 有两个：

### 1.1 Attention Sink Phenomenon

Attention sink 是指 LLM 把 excessive attention 分配给 initial tokens。Xiao et al. 2023 在 StreamingLLM 中首次系统观察这个现象。Paper 的关键 insight 在 Figure 3：Qwen2-7B 的 attention score heatmap 和 token embedding variance 之间存在强 correlation——第一个 token 的 hidden state variance 显著高于 subsequent tokens。

这个 correlation 揭示了一个机制：causal attention 本身 non-permutation invariant，positional information 通过 softmax normalization 之后 token embedding 的 variance 隐式 emergent (Chi et al. 2023, ACL short paper: https://aclanthology.org/2023.findings-acl.85/)。即使模型用了 RoPE 这种 explicit relative position embedding，softmax normalization 还是会让模型 learn and rely on 这种 implicit absolute positional information via variance propagation。

### 1.2 Softmax 导致的 Information Loss

Paper 在 Section 2.2 给了一个非常 pedagogical 的例子，Appendix A 给了严格数学证明。核心公式是：

$$\alpha_i = \frac{\exp(E_i)}{\sum_{j=1}^{n} \exp(E_j)}$$

其中 $E_i = \frac{q \cdot k_i}{\sqrt{d}}$ 是 query $q$ 和第 $i$ 个 key $k_i$ 的 scaled dot-product energy，$d$ 是 head dimension。

定义 gap $\Delta_i = E_1 - E_i$，那么 ratio：

$$\frac{\alpha_i}{\alpha_1} = \exp(-\Delta_i)$$

如果 $\Delta_i$ 大且正，$\alpha_i$ 相对 $\alpha_1$ 指数级 vanish。

Appendix A 的关键论证基于 extreme value theory。假设 $E_i \sim \mathcal{N}(\mu, \sigma^2)$ i.i.d.，那么 $L$ 个样本的 maximum：

$$E_{(L)} \approx \mu + \sigma\sqrt{2\ln L}$$

所以 expected gap $\Delta \approx \sigma\sqrt{2\ln L}$，ratio：

$$\frac{\alpha_i}{\alpha_1} \approx \exp(-\sigma\sqrt{2\ln L})$$

当 $L$ 大时这个 ratio 指数级小。这就是 "winner-takes-most" — softmax 本质上是个 sparsifier。

这个 sparsification 在 full-context Transformer 里有 benefit (selective focus)，但在 SWA scenario 下，window 内的 tokens 本来就少，再 sparse 化就导致 historical information 严重丢失。window 滑动之后，被 evict 的 token 的信息本来应该被 compress 进 remaining tokens，但 softmax 的 aggressive filtering 阻碍了这种 information retention。

参考 softmax attention sparsity 的讨论：https://arxiv.org/abs/2409.04431 (Ramapuram et al. 2025, "Theory, analysis, and best practices for sigmoid self-attention")

## 2. SWAT Architecture: Sigmoid + Balanced ALiBi + RoPE

### 2.1 Sigmoid Replacement

核心 modification (Equation 2)：

$$\text{Attention}(Q, K, V) = \sigma\left(\frac{QK^T}{\sqrt{d}}\right)V$$

其中 $Q \in \mathbb{R}^{N \times d}$, $K \in \mathbb{R}^{N \times d}$, $V \in \mathbb{R}^{N \times d}$ 是 packed query/key/value matrices，$\sigma(\cdot)$ 是 sigmoid 函数 $\sigma(z) = \frac{1}{1+e^{-z}}$。

Sigmoid 的关键性质 (Appendix B)：每个 attention weight 独立计算，没有 shared denominator，没有 winner-takes-most。这使得 window 内所有 token 的信息都能被 retain 进 output embedding。

直觉上：softmax 像是个 selector，强迫模型 pick one winner；sigmoid 像是个 accumulator，让每个 token 根据 relevance 独立贡献。在 SWA 训练下，模型需要把 past token 信息 compress 进 current token，sigmoid 的 dense property 正好 enable 这种 compression。

### 2.2 Balanced ALiBi

但 sigmoid 有个问题：太 dense 了，没有 discriminative bias。如果所有 attention weight 都在 0.5 附近，token representations 会 overloaded，无法 differentiate。所以需要引入 position-dependent differentiation。

原始 ALiBi (Press et al. 2022, https://arxiv.org/abs/2108.12409) 只用 negative slope，enforce 一个 directional inductive bias (attention 衰减到更远的 past token)。SWAT 提出双向 balanced ALiBi (Equation 3, 4)：

$$\text{Attention}(Q, K, V) = \sigma\left(\frac{QK^T}{\sqrt{d}} + s \cdot (m - n)\right)V$$

其中 $m$ 是当前 token 的 index，$n$ 是被 attend 的 token 的 index，$s$ 是 slope。注意 $m > n$ 因为 causal mask (只 attend past 和 self)。

Slope 配置 (Equation 4)：

$$s_k = \begin{cases} -2^{-k} & \text{for forward-looking heads} \\ 2^{-k} & \text{for backward-looking heads} \end{cases}$$

其中 $k \in \{1, 2, \ldots, h/2\}$，$h$ 是总 head 数。

这里 "forward-looking" 指 negative slope (传统 ALiBi 方向，penalize 远的 past token，focus 近的)；"backward-looking" 指 positive slope (reward 远的 past token，preserve historical information)。

一半 head forward-looking，一半 backward-looking。这种 bidirectional specialization 让 attention heads 分工：一部分 focus recent context，一部分 preserve history。

直觉：单向 ALiBi 假设 recent > distant always，这在 full attention 下合理，但在 SWA 下，distant tokens 已经被 window evict 了，remaining 的都是 "近的" relative to window。需要 bidirectional 来 balance — 既要 forget irrelevant noise，又要 remember useful history。

### 2.3 RoPE Enhancement

问题：sigmoid 替换 softmax 之后，normalization 带来的 implicit position information 没了，导致 training instability。ALiBi 提供的 positional signal 也比较 weak。

Solution：加入 RoPE (Su et al. 2023, https://arxiv.org/abs/2104.09864) 提供 explicit positional encoding。最终 attention (Equation 5)：

$$\text{Attention}(Q, K, V)_m = \sum_{n=m-\omega+1}^{m} \sigma\left(\frac{(R_{\Theta,m}^d q_m)^T (R_{\Theta,n}^d k_n)}{\sqrt{d_k}} + s \cdot (m-n)\right) v_n$$

其中：
- $R_{\Theta,m}^d$ 是 position $m$ 对应的 rotation matrix (block-diagonal，每个 block 是 2D rotation by angle $\theta_i m$)
- $R_{\Theta,n}^d$ 同理
- $\omega$ 是 window size
- $d_k$ 是 key dimension
- 求和范围 $n \in [m-\omega+1, m]$ 体现 sliding window constraint $m - n < \omega$

RoPE 的作用是让 $q_m$ 和 $k_n$ 的 dot-product 自然 decay with relative distance $|m-n|$ (因为 rotation angle 差)，给 sigmoid 提供 additional position-aware differentiation。

### 2.4 Computational Complexity (Equation 6)

$$\text{Cost} = N\omega \times (1 + \delta_{\text{ALiBi}}), \quad 0 < \delta_{\text{ALiBi}} \ll 1$$

其中 $N$ 是 sequence length，$\omega$ 是 window size，$\delta_{\text{ALiBi}}$ 是 ALiBi bias 计算的额外开销 (非常小)。总体 linear in $N$，比 vanilla attention 的 $O(N^2)$ 快得多。

## 3. Information Transmission: SWA Training 的新范式

Section 3.1 是 paper 最 insightful 的部分之一。Figure 4 展示了 SWA training 引入的 new paradigm。

在 vanilla Transformer training 中，sequence length < window size，每个新 token 可以 attend 到所有 past tokens (包括 sequence 开头的 token)。模型 learn 的是 per-token information extraction，softmax 强化这种 selective extraction。

SWA training 引入 fundamentally 不同的 learning paradigm：

- 每次窗口滑动，old token embedding 被 discard
- 但在 Transformer 的 upper layers，new token 的 embedding 通过 residual connection 和 attention 仍然 retain 了一部分 old token 的信息 (Figure 1 的 yellow lines 表示这种 information transition)
- 因此模型被 incentivized to compress 所有 past information 进 upper-layer token embeddings，防止 sliding window 导致的信息丢失

Paper 给出 theoretical information range：第 $l$-th transformer layer，单个 token 的 information range 是 $1 + (\omega-1) \cdot l$，最大范围 $1 + (\omega-1) \cdot L$，其中 $L$ 是总层数。

例如 Figure 1 中 $\omega=3$, $L=2$，最大 range = $1 + 2 \cdot 2 = 5$。

这个 paradigm shift 很关键：vanilla Transformer learns to **extract** information from full context；SWA-trained Transformer learns to **compress** information into fixed-size hidden states that survive window eviction。这本质上让 Transformer 表现得像 RNN/SSM，但 architecture 仍然是标准 Transformer。

## 4. Experimental Results 解析

### 4.1 Overall Performance (Table 1)

340M 和 760M 参数，分别 15B 和 30B tokens pre-training，在 FineWeb-Edu 100BT subset 上。

8 个 commonsense reasoning benchmarks: Wikitext, Lambada, PIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, SIQA, BoolQ。

Baselines: Transformer++, RetNet, GLA, Mamba, Mamba2, DeltaNet, TTT, Gated DeltaNet, Titans。

SWAT 有三个配置：
- SWAT (-): 只用 negative slopes (forward-looking)
- SWAT (+): 只用 positive slopes (backward-looking)  
- SWAT (-+): 一半 negative 一半 positive (balanced)

760M 结果 (最 informative)：
- Titans: avg 51.56% (最强 baseline)
- SWAT (-): avg 51.85% (SOTA, statistically significant with p<0.05)
- SWAT (-+): avg 51.01%
- SWAT (+): avg 50.48%

观察：
1. SWAT (-) 在 short-text benchmarks (PIQA 69.80, HellaSwag 48.65) 表现最好，因为 focus recent tokens 适合 short context
2. SWAT (-+) 在 BoolQ (62.11%, 需要历史 context 的 QA) 上最好，证明 balanced 配置 preserve history 的价值
3. SWAT (+) 单独 backward-looking 效果差，说明 purely remember history 不 work，需要和 forward attention 结合

Perplexity 上 SWAT (-) 在 340M 时比 baselines 高，但到 760M 显著下降 (Wiki 23.41, LMB 21.05)，暗示 sigmoid attention 有 better scaling behavior。

### 4.2 SWA Training Effectiveness (Table 2)

这个 table 是 paper 的核心实验之一，证明 SWA training 本身 (independent of sigmoid/ALiBi) 的价值。

比较 vanilla Transformer (training length = training window) vs SWA-trained Transformer，在 OpenWebText, PG-19, OpenOrca 上 evaluation length 从 128 到 16,384。

关键发现：
1. **SWA training 显著提升 long-context 性能**：Sliding Window B (training window 1024, training length 4096) 在 eval length 16,384 上 OpenWebText 2.9128, PG-19 4.4383, OpenOrca 5.8802，全面超过 Vanilla B (1024/1024) 的 3.0786/5.2372/7.9706
2. **Vanilla Transformer 只在 eval length = training length 时最优**：Vanilla B 在 eval 1024 时 OpenWebText 2.9636 最优，但 eval 16,384 时退化到 3.0786
3. **SWA-trained 模型 cross-length 稳定**：Sliding Window B 在 eval 1024/4096/16384 上 OpenWebText 分别 3.0197/2.9638/2.9128，甚至随 length 增加而 improve

第二个观察特别重要：SWA training 让模型 learn 了 "compress and retrieve" 的能力，而 vanilla Transformer 学的是 "memorize all, retrieve by position"，后者在 length mismatch 时崩溃。

### 4.3 Ablation Study (Table 3, Figure 5)

11 个配置的 systematic ablation，是理解 SWAT 各组件 contribution 的关键。

| No. | Model | Activation | Position Emb | OWT | PG-19 | OpenOrca | Avg |
|-----|-------|-----------|--------------|-----|-------|----------|-----|
| 1 | Vanilla | Softmax | RoPE | 4.84 | 5.69 | 6.01 | 5.51 |
| 2 | Vanilla | Sigmoid | RoPE | 14.26 | 15.48 | 1.99 | 10.57 |
| 10 | Vanilla | Softmax | RoPE (1024) | 2.96 | 4.54 | 5.47 | 4.33 |
| 11 | Vanilla | Sigmoid | ALiBi (1024) | 2.97 | 5.07 | 0.17 | 2.74 |
| 8 | Sliding | Sigmoid | AliRope-6:6 | 3.05 | 4.31 | 0.17 | 2.51 |

关键 insights：

1. **No.1 vs No.2**: 直接把 softmax 换成 sigmoid 在 vanilla Transformer 上 catastrophic failure (avg 10.57 vs 5.51)。Sigmoid 没有 mutual suppression，token embeddings 信息 overloaded。证明 sigmoid 不能单独用。

2. **No.10 vs No.11**: 用 ALiBi 之后 sigmoid 可以 work (avg 2.74 vs 4.33)，ALiBi 提供 position-dependent differentiation 弥补 sigmoid 的 over-density。

3. **Slope 配置 matters**: No.4 (ALiBi-12:0, 全 negative) avg 2.62; No.5 (ALiBi-8:4) avg 2.65; No.6 (ALiBi-6:6, balanced) avg 2.73。差距不大但 balanced 在需要 history 的任务上更好。

4. **No.8 (AliRope-6:6)** 是最优配置 avg 2.51：Sigmoid + balanced ALiBi + RoPE。Figure 5 显示这个配置 training loss 最低且最 stable。

5. **No.6 vs No.7**: training length 从 1024 扩到 2048，fixed layers 和 window size，没有帮助 (2.73 vs 2.76)。说明 information retention 受限于 window × depth，单纯加 length 不 work。

## 5. 关键联想与 Intuition Building

### 5.1 SWAT 与 Linear RNN / SSM 的关系

SWAT 本质上把 Transformer 训练成了一个 "pseudo-RNN"。Sliding window 是 recurrent state 的 finite approximation：window 内的 tokens 相当于 RNN 的 hidden state，window eviction 相当于 RNN 的 state update。

但与 Mamba (https://arxiv.org/abs/2312.00752), RWKV (https://arxiv.org/abs/2305.13048), RetNet (https://arxiv.org/abs/2307.08621) 不同，SWAT 保留了标准 Transformer architecture，只是改了 attention pattern 和 activation。这意味着：
- 可以复用所有 Transformer training infra (FlashAttention 等)
- 不需要 custom CUDA kernel for parallel scan
- Inference 时仍然是 attention-based，可解释性更好

### 5.2 与 Attention Sink 文献的联系

Attention sink 的相关工作：
- StreamingLLM (Xiao et al. 2023): https://arxiv.org/abs/2309.17453 — 通过保留 sink tokens 实现 streaming
- When attention sink emerges (Gu et al. 2024): https://arxiv.org/abs/2410.10781 — empirical 分析 sink 的 emergence

SWAT 的 approach 是从 training 阶段就 eliminate sink，通过 sigmoid 替换 softmax 根除 variance propagation 机制。这比 StreamingLLM 的 inference-time fix 更 fundamental。

### 5.3 Sigmoid Attention 的更广 context

最近 sigmoid attention 重新受到关注：
- Apple 的 "Theory, analysis, and best practices for sigmoid self-attention" (Ramapuram et al. 2025): https://arxiv.org/abs/2409.04431
- 这是 SWAT 的直接 inspiration 之一

Sigmoid attention 的核心 trade-off：dense (information retention 好) vs discriminative (selection 能力弱)。SWAT 用 ALiBi + RoPE 补 discriminative 部分，是个 elegant 的组合。

### 5.4 Information Range 的局限性

Paper Section 7 limitations 指出：SWAT 的 maximum attention distance = $1 + (\omega-1) \cdot L$。例如 $\omega=1024$, $L=32$ (typical 7B model)，max range ≈ 32,768。超过这个 length 信息必然丢失。

这解释了为什么 SWAT 在 BoolQ 这种需要 long-range context 的任务上用 balanced (-+) 配置更好——它更高效地利用有限的 information bandwidth。

未来方向可能是 hybrid architecture：SWAT for medium context + explicit memory retrieval (像 Memorizing Transformers https://arxiv.org/abs/2203.08913 或 Focused Transformer https://arxiv.org/abs/2307.03170) for ultra-long context。

### 5.5 Balanced ALiBi 的 inductive bias

双向 slope 设计让我想到 bidirectional RNN (Schuster & Paliwal 1997)。但 SWAT 不是真正的 bidirectional (仍然 causal mask)，而是让不同 head specialize in 不同 temporal direction。这类似 multi-head attention 中不同 head 学习不同 pattern 的思路，但 here 是 explicitly encode direction bias。

Geometric slope sequence $2^{-k}$ 来自原始 ALiBi，是经验性的。为什么 geometric 而非 linear？可能因为 attention decay 通常是 exponential 的，geometric slopes 让不同 head 覆盖 different timescales (类似 multi-scale)。

## 6. 批判性思考

### 6.1 Perplexity 在 340M 上的退化

SWAT (-) 在 340M 时 Wiki ppl 33.32, LMB 36.75，显著差于 Titans (26.18/29.97)。但到 760M 时反超 (23.41/21.05 vs 20.04/21.96 — 实际上 Wiki 上 Titans 仍更好)。

这暗示 sigmoid attention 可能需要 larger scale 才能 overcome dense representation 的 optimization difficulty。Paper 没有深入讨论这个 scaling behavior 的 mechanism，是个 open question。

### 6.2 Hyperparameter Sensitivity

Paper 承认 SWAT 对 window size, model depth, ALiBi slope distribution 非常 sensitive。这是 practical deployment 的 concern。Table 3 显示不同配置性能差异大，没有给出系统的 hyperparameter search guidance。

### 6.3 与 FlashAttention 的兼容性

Sigmoid attention 可以用 FlashAttention 的 tiling 思想，但 ALiBi bias 需要额外加进 tile computation。Paper 提到用 flash-linear-attention repo (https://github.com/sustcsonglin/flash-linear-attention)，但没给 detailed kernel implementation。

## 7. Summary

SWAT 的核心 contribution 是一个 simple insight：**SWA 要从 training 阶段就做，不能只改 inference**。而要 enable effective SWA training，需要解决 attention sink 和 information loss，solution 是 sigmoid (dense, no sink) + balanced ALiBi (directional differentiation) + RoPE (explicit position)。

这个工作的重要性在于：它证明了不需要复杂的 SSM/linear attention architecture，standard Transformer 经过适当 modification 就能 achieve comparable 甚至 better long-context performance，同时保持 linear inference complexity 和 architectural simplicity。这对于 deployment-friendly long-context LLM 是个 attractive direction。

代码参考：
- flash-linear-attention: https://github.com/sustcsonglin/flash-linear-attention
- lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
- nanoGPT: https://github.com/karpathy/nanoGPT

相关 paper 链接汇总：
- ALiBi: https://arxiv.org/abs/2108.12409
- RoPE: https://arxiv.org/abs/2104.09864
- Longformer: https://arxiv.org/abs/2004.05150
- StreamingLLM: https://arxiv.org/abs/2309.17453
- Mamba: https://arxiv.org/abs/2312.00752
- RetNet: https://arxiv.org/abs/2307.08621
- Titans: https://arxiv.org/abs/2501.00663
- Sigmoid attention theory: https://arxiv.org/abs/2409.04431
- GLA: https://arxiv.org/abs/2312.06635
- DeltaNet: https://arxiv.org/abs/2406.06484
- TTT: https://arxiv.org/abs/2407.04620
- FineWeb-Edu: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
