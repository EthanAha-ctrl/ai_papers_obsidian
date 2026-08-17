---
source_pdf: Reasoning with Sampling.pdf
paper_sha256: 8f029968f20292b320dda95d253911a187e34a18d33aa6cdc65883b9cc1a47d4
processed_at: '2026-08-11T21:40:28-07:00'
target_folder: LLM-engine
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：这篇 paper 到底在说什么

## 一句话总结

**你不需要 RL，base model 已经够聪明，只是采样方式没对。**

---

## 故事从头讲

最近 RLVR (RL with verifiable rewards) 火得不行，DeepSeek-R1、o1 都靠这个把 reasoning 拉上来。大家觉得 "哇，RL 让模型学会推理了"。

但有几个人开始嘀咕：

- **Yue et al.** 发现：RL 之后模型生成的 reasoning trace，在 base model 下的 likelihood 很高 —— 说明 RL 没造出"新"行为，只是把 base model 本来就会的东西"拉"到前台
- **Song et al.** 发现：random reward 也能 sharpen distribution，说明 RL 的 mechanism 可能就是个 distribution sharpening
- **He et al.** 发现：GRPO 之后 pass@k (multi-shot) 反而下降 —— RL 用 diversity 换 single-shot 准确率

所以问题就来了：**既然 RL 只是 sharpening，那直接 sample sharpened distribution 不就行了？**

这篇 paper 说：**对，而且能跟 GRPO 打平，甚至超越**。

---

## 核心 insight：两个 sharpening 长得很像，但根本不一样

### 大家都以为 sharpening = 降温度

降温度就是把 next-token logits 除以 τ 再 softmax。τ < 1 让分布更尖。这看起来就是把 p "变尖"。

paper 说：**这是错觉**。降温度采样的不是 p^α，是 (marginal)^α。

### 为什么不一样？讲个 toy example

假设 vocab 只有 {a, b}，两 token 序列：

| seq | p |
|-----|---|
| aa  | 0.00 |
| ab  | 0.40 |
| ba  | 0.25 |
| bb  | 0.25 |

marginal: p(a)=0.40, p(b)=0.50

设 α=2，问第一个 token 选 a 还是 b？

**降温度法**：直接看 marginal，a^2=0.16, b^2=(0.25+0.25)^2=0.25，选 **b**。

**Power distribution**：把所有 future path 先平方再求和。a: 0^2+0.4^2=0.16; b: 0.25^2+0.25^2=0.125，选 **a**。

**结果完全相反**！

为什么？因为 a 后面只有一条路 (ab, likelihood 0.40)，是 high-likelihood 的 narrow path。b 后面有两条路 (ba, bb)，但每条都只有 0.25，是 low-likelihood 的 broad path。

降温度只看当前位置 marginal，看不到 future shape。Power distribution 因为是对 joint prob 取幂，**天然隐式 lookahead**。

---

## 为什么这对 reasoning 重要？

Reasoning 错误的本质经常是 **pivotal token** 现象 —— 某个关键 token 选错了，后面全错。

- **正确的 pivotal token**: 后面只有少数几条 high-likelihood future → Power distribution 喜欢这种
- **错误的 pivotal token**: 后面有一堆 low-likelihood future (看似有很多选择，但每条都不太对) → 降温度会喜欢这种

降温度在关键决策点容易被"看起来选项多"的 token 骗。Power distribution 因为 implicit lookahead，能躲开这个坑。

---

## 那怎么 sample from p^α？

p^α 是 unnormalized 的，没法直接 sample。要用 MCMC，具体是 **Metropolis-Hastings**。

**MH 流程**：
1. 当前 sequence x
2. 随机选个位置，从那里开始重新 sample 一段 (用 proposal LLM)
3. 算新 sequence 的 p^α 和旧 sequence 的 p^α 比值
4. 比值大于 1 直接接受；小于 1 按概率接受
5. 重复 N 次

最后 stationary distribution 就是 p^α。

---

## 但 naive MH 在 LLM 上会爆炸

Token sequence space 维度太高，mixing time 可能 exponential。直接对整条 sequence 跑 MH 会 stuck。

**paper 的 trick：block annealing**

把 sequence 切成 block (paper 用 192 token/block)，逐步从短到长 sample：

```
p(x_0:B)^α → p(x_0:2B)^α → ... → p(x_0:T)^α
```

每个 block 用上一个 block 的结果作为 init，再跑 N_MCMC 次 MH (paper 用 10 次)。这样 MH 起点不会太离谱，mixing 快。

---

## 结果有多惊艳？

Qwen2.5-Math-7B 上：

| 方法 | MATH500 | HumanEval | GPQA | AlpacaEval |
|---|---|---|---|---|
| Base | 0.496 | 0.329 | 0.278 | 1.61 |
| 降温度 | 0.690 | 0.512 | 0.353 | 2.09 |
| **Power Sampling** | **0.748** | **0.573** | **0.389** | **2.88** |
| GRPO (训过) | 0.785 | 0.537 | 0.399 | 2.38 |

- MATH500 (in-domain for GRPO): 几乎打平 (差 3.7%)
- HumanEval (out-of-domain): **超过 GRPO +3.6%**
- AlpacaEval (non-verifiable): **超过 GRPO +21%**

Phi-3.5 上 HumanEval 更夸张：Power Sampling 0.732, GRPO 0.134 —— **GRPO 训完 HumanEval 崩了** (过度 sharpening 到错误模式)。

---

## 更好的是：diversity 没丢

GRPO 的 pass@k 在 k 大时 saturate —— 说明它 collapse 到几条 trace，再 sample 也是同样几条。

Power Sampling 的 pass@k 跟 base model 接近 —— 说明它保留了 base model 的 diversity，同时把 single-shot accuracy 拉到 GRPO 水平。

**Single-shot 准确率 + Multi-shot diversity 全要**，RL 一直做不到的事，Power Sampling 做到了。

---

## Cost 怎么样？

代价是约 **8.84× 标准 inference cost**。因为每个 block 要跑 10 次 MH resampling。

但跟 GRPO 训练一个 epoch 的 cost 接近 (GRPO 要 16 rollouts/sample)。而且**训练只要训一次，推理每用一次都要花钱** —— 长期看 power sampling 更贵，但短期看免训练的优势大。

---

## 一个 qualitative 例子感受一下

HumanEval 任务：filter 列表中以 prefix 开头的字符串。

**Power Sampling**：`return [s for s in strings if s.startswith(prefix)]` ✓

**GRPO**：`return [string for string in strings if string.startswith(f'{prefix}'*2)]` ✗

GRPO 学了 `'{prefix}'*2` 这种荒谬 pattern —— 它 overfit 到训练数据某个 template，sharpening 把它 lock 死了。Power Sampling 因为从 base model 采样，不会被 training shortcut 污染。

---

## 这篇 paper 想告诉你的事

**你 (Karpathy) 一直说的 "model 已经很聪明，采样方式不对" 这句话被这篇 paper 算法化、实验化了。**

- "聪明" = base model 高 likelihood 区域就有好 reasoning trace
- "采样方式不对" = 降温度是 marginal sharpening，看不到 future
- "对的采样方式" = Power distribution p^α，implicit lookahead

Inference-time scaling 不止 best-of-N 一个轴。**Sharpening 是另一个 axis**，而且不需要 verifier —— 这意味着可以推广到 creative writing、alignment 这些 RLVR 摸不到的地方。

---

## 一句话 takeaway

**RL post-training ≈ 在 base model 上做 distribution sharpening。这个 sharpening 你完全可以在 inference time 用 MCMC 直接 sample，不用训练。而且因为没动 model weights，diversity 也保住了。**

这篇 paper 把 "RL 究竟在干什么" 这个问题用一个非常 elegant 的算法回答了 —— 然后顺便证明了答案。

---

# Reasoning with Sampling: Your Base Model is Smarter Than You Think — Paper 深度解读

这篇 paper 来自 Harvard 的 Aayush Karan 和 Yilun Du，发表时间 2025 年下半年。核心 thesis 极其 provocative：**base model 本身就已经具备 RL post-training 所"激发"的 reasoning capability，只需要在 inference time 通过正确的 sampling 算法就能把这部分 capability 释放出来**。这是一个非常 "Karpathy-flavored" 的 insight —— 让我想起你之前在 LSTM 和 nanoGPT 中反复强调的 "the model already knows, we just need to sample it right"。

下面我尽量把每个细节都讲透，把直觉 build 起来。

---

## 1. 大背景：RL post-training 究竟在做什么？

近期关于 RLVR (Reinforcement Learning with Verifiable Rewards) 的研究有一个反复出现的争议：**RL post-training 是真的"教会"模型新的 reasoning 行为，还仅仅是在 base model distribution 上做 sharpening**？

相关文献：
- **DeepSeek-R1** (Guo et al., 2025) — RL 让模型产生 long-form reasoning traces
- **He et al., 2025** "Rewarding the unlikely" — 指出 GRPO 在 large k 时 pass@k 反而下降
- **Yue et al., 2025** "Does RL really incentivize reasoning beyond base model?" — 发现 RL 产生的 reasoning trace 在 base model 下 likelihood 很高
- **Song et al., 2025** "Spurious rewards" — 即使 random reward 也能 sharpen distribution

Yue et al. 的核心观察是：RL 后的 reasoning traces **集中在 base model 的高 likelihood 区域**。这暗示 RL 没有产生"新行为"，仅是把 base model 本来就 high-likelihood 的好 reasoning trace "拉"到 single-shot 生成中。

paper Figure 4 左图非常 striking —— GRPO 的 log-likelihood histogram 是一个极高极窄的 peak，而 base model 是 spread-out 的 multi-modal。**这就是 distribution sharpening 的可视化**。

paper 的问题就是：**既然 RL 只是 sharpening，那我们能不能跳过 RL，直接从 sharpened distribution 采样？**

Reference: 
- [DeepSeek-R1 paper](https://arxiv.org/abs/2501.12948)
- [Yue et al. - Does RL incentivize reasoning](https://arxiv.org/abs/2504.13837)
- [Song et al. - Spurious rewards](https://arxiv.org/abs/2506.10947)

---

## 2. 关键 insight：Sharpening = Power Distribution p^α

最直接的 sharpening 方式是把分布提升到幂次：从 p(x) 变到 p(x)^α，其中 α ≥ 1。

**为什么 p^α 是 sharpening？** 看公式 (2)：

$$
p(\mathbf{x}) > p(\mathbf{x}') \implies \frac{p(\mathbf{x})^\alpha}{p(\mathbf{x}')^\alpha} > \frac{p(\mathbf{x})}{p(\mathbf{x}')} \quad (\alpha \in [1, \infty])
$$

变量说明：
- p: base model 给 token sequence 的 probability
- x, x': 两个不同 token sequence
- α: power exponent，控制 sharpening 强度
- 当 α=1 时 degenerate 回 base distribution
- 当 α→∞ 时退化成 argmax (greedy MAP)

**直觉**：如果 x 比 x' 在 base model 下更 probable，那 p^α 会把这个 gap 放大指数倍。低 likelihood 的 sequence 被压得更低，高 likelihood 的 sequence 被抬得更高 —— 这就是 sharpening。

paper Figure 2 给了一个 toy example：一个 Gaussian mixture，p 是双峰，p^4 变成 dominant 峰被放大、次峰被压扁。

---

## 3. ⚠️ 关键 trap：Low-temperature sampling ≠ Power distribution sampling

这是 paper 最深刻的 insight 之一，也是绝大多数人会踩的坑。

### 3.1 Low-temperature sampling 定义 (公式 3)

$$
p_{\text{temp}}(x_t | x_0 \dots x_{t-1}) = \frac{p(x_t | x_{t-1} \dots x_0)^\alpha}{\sum_{x_t' \in \mathcal{X}} p(x_t' | x_{t-1} \dots x_0)^\alpha}
$$

变量：
- p_temp: low-temperature sampling 的 conditional next-token distribution
- α: 与 1/temperature 等价 (τ = 1/α)
- x_t: 当前要采样的 token
- 分母：对所有 vocab token x_t' 求和，做 normalization

这是大家熟悉的 temperature sampling —— token logits 除以 τ 再 softmax。

### 3.2 Power distribution 的 conditional (公式 4)

真正从 p^α 采样的 conditional next-token distribution 是：

$$
p_{\text{pow}}(x_t | x_0 \ldots x_{t-1}) = \frac{\sum_{x_{>t}} p(x_0, \ldots, x_t, \ldots, x_T)^\alpha}{\sum_{x_{\geq t}} p(x_0, \ldots, x_t, \ldots, x_T)^\alpha}
$$

变量：
- 分子：固定 prefix x_{0:t}，对**所有 future completion** x_{>t} 求和，每个 full sequence 的 joint probability 取 α 次幂
- 分母：对当前 token x_t 也求和，再对 future 求和
- 这是一个 **marginalize over future** 的过程

### 3.3 两者 difference 的数学表达

经过 Bayes rule 展开 (公式 7 vs 公式 8)：

**Power distribution**: sum of exponents
$$
p_{\text{pow}}(x_t | \mathbf{x}_{<t}) \propto \sum_{x_{>t}} p(x_0, \ldots, x_t, \ldots, x_T)^\alpha
$$

**Low-temperature**: exponent of sums
$$
p_{\text{temp}}(x_t | \mathbf{x}_{<t}) \propto \left( \sum_{x_{>t}} p(x_0, \ldots, x_t, \ldots, x_T) \right)^\alpha
$$

**直觉差异**（这是 paper 最精华的部分）：

- Low-temp 在每个 token 决策点 **greedily 看所有 future paths 的 average likelihood**，然后 sharpen 这个 average
- Power distribution 先 **sharpen 所有 future paths**，再求和 —— 等价于隐式地 **look ahead**，prefer 那些 "未来有少数但极 high-likelihood paths" 的 token

**Observation 1**: Power distribution upweights tokens with **few but high likelihood future paths**; low-temperature upweights tokens with **several but low likelihood completions**.

### 3.4 Toy Example 1 — 直击心脏

vocab = {a, b}, two-token sequences, joint probs：

| sequence | p |
|----------|---|
| aa | 0.00 |
| ab | 0.40 |
| ba | 0.25 |
| bb | 0.25 |

marginal：p(x_0=a)=0.40, p(x_0=b)=0.50

设 α=2：

**Power distribution**:
- p_pow(x_0=a) ∝ 0.00² + 0.40² = 0.160
- p_pow(x_0=b) ∝ 0.25² + 0.25² = 0.125
- **选 a** (虽然 marginal 低)

**Low-temperature**:
- p_temp(x_0=a) ∝ (0.00 + 0.40)² = 0.160
- p_temp(x_0=b) ∝ (0.25 + 0.25)² = 0.250
- **选 b**

这就是 paper 想表达的核心：选 a 意味着只有一条 future path 但 high likelihood (0.40)；选 b 意味着有两条 future paths 但每条都只有 0.25。

Reasoning task 里，**正确答案往往是一条 narrow high-likelihood path**，错误答案是很多条 broad low-likelihood paths。Power sampling 自然 prefer 前者，low-temperature prefer 后者。

### 3.5 与 "pivotal tokens" / "critical windows" 的联系

这联系到 **Li et al., 2025** "Blink of an eye" 和 **Abdin et al., 2024** Phi-4 tech report 中的 "pivotal token" 现象：在 reasoning trace 中，少数几个 token 极大决定最终输出正确性。这些 token 通常对应 "narrow high-likelihood future"。

paper 在 Appendix A.1 给出 Proposition 3 的形式化：

- **Positive pivotal token**: marginal weight ε 全部集中在 1 个 future path (singular support)
- **Negative pivotal token**: marginal weight ε 均匀分到 N 个 future paths (likelihood 各为 ε/N)

条件 (公式 14)：
$$
\frac{\varepsilon'}{N^{1 - 1/\alpha}} < \varepsilon < \varepsilon'
$$

在该条件下：
- x_t (positive pivotal) 的 marginal weight ε **小于** x_t' (negative pivotal) 的 marginal weight ε'
- 所以 low-temperature 选 x_t'（marginal 高的那个）
- Power distribution 选 x_t（max future likelihood 高的那个）

变量：
- ε: positive pivotal token 的 marginal weight
- ε': negative pivotal token 的 marginal weight
- N: negative pivotal token 的 future support size
- α: power exponent

这个 proposition 把 reasoning 失败的本质讲清楚了 —— **low-temperature 在 pivotal 决策点容易选错 token**，因为它只看 marginal，不看 future shape。

References:
- [Li et al. - Blink of an eye](https://arxiv.org/abs/2502.00921)
- [Phi-4 technical report](https://arxiv.org/abs/2412.08905)
- [Wang et al. - Contextual temperature](https://arxiv.org/abs/2012.13575)

---

## 4. Algorithm: Metropolis-Hastings + Autoregressive Block Annealing

### 4.1 为什么要 MCMC？

p^α 是 unnormalized 的 —— 要 normalize 需要对所有 token sequence 求和，这在 vocab size V 和长度 T 下是 V^T 量级，intractable。

经典解决方案：**Metropolis-Hastings** (MH, Metropolis et al., 1953)。MH 只需要 unnormalized probability 的相对值，通过构造一个 Markov chain，其 stationary distribution 就是 target distribution p^α。

### 4.2 MH acceptance rule (公式 9)

$$
A(\mathbf{x}, \mathbf{x}^i) = \min\left\{1, \frac{p^\alpha(\mathbf{x}) \cdot q(\mathbf{x}^i | \mathbf{x})}{p^\alpha(\mathbf{x}^i) \cdot q(\mathbf{x} | \mathbf{x}^i)}\right\}
$$

变量：
- A: acceptance probability, 取值 [0,1]
- x: proposed candidate sequence
- x^i: current state at iteration i
- q(x^i | x): proposal distribution 的 reverse transition probability
- q(x | x^i): proposal distribution 的 forward transition probability
- p^α(x): unnormalized target probability
- p^α(x^i): current state 的 unnormalized target probability

接受规则：draw u ~ Uniform(0,1)，如果 u ≤ A 接受 (x^{i+1}=x)，否则保持 x^{i+1}=x^i。

**MH 的优美**：normalization constant 在分子分母中 cancel，所以只需 unnormalized 值。

### 4.3 Proposal distribution: Random Resampling

paper 选择一个特别简单的 proposal：

1. Uniformly 随机选一个位置 t ∈ [1, T]
2. 用 proposal LLM p_prop 重新 sample 从 t 开始的 suffix
3. 这个 resampling 的 likelihood 就是 q(x | x^i)
4. Reverse q(x^i | x) 由 symmetry 也是可计算的

这个 proposal 满足 **irreducible** (任何两个 sequence 之间都有 non-zero transition prob，因为可以从 t=1 全部 resample) 和 **aperiodic** (chain 不会陷入固定周期)。

### 4.4 ⚠️ 为什么 naive MH 在 LLM 上不可行？

paper Section 4.3 指出关键问题：**mixing time 在高维 space 中可能 exponential**。

- Token sequence space |X|^T 是极 high dimensional
- 长 sequence 下 naive MH 会 stuck 在局部 mode，无法 explore
- 这是 Bandeira et al., 2022 和 Schmidler & Woodard, 2013 在 high-dim MCMC 中证明的现象

### 4.5 ⭐ 核心 trick: Block-wise Annealing (公式 10)

paper 的核心算法创新：**用 autoregressive 结构 + block-wise intermediate distributions 来 initialize MH**。

定义一系列 intermediate distributions：

$$
\emptyset \longrightarrow p(x_{0:B})^\alpha \longrightarrow p(x_{0:2B})^\alpha \longrightarrow \cdots \longrightarrow p(x_{0:T})^\alpha
$$

变量：
- B: block size (paper 用 192)
- π_k(x_{0:kB}) ∝ p(x_{0:kB})^α: 第 k 个 intermediate distribution (公式 11)

**算法流程 (Algorithm 1)**：

```
for k = 0 to T/B - 1:
    # 第 k 个 block, 目标是 sample from π_{k+1}
    # Step 1: 用 p_prop autoregressive 生成下一个 block 作为 init
    x^0 = p_prop(x_{kB+1:(k+1)B} | x_{<kB+1})
    
    # Step 2: N_MCMC 次 MH iteration
    for n = 1 to N_MCMC:
        # 随机选位置 m ∈ [1, (k+1)B]
        m = uniform(1, (k+1)B)
        
        # 从位置 m 开始用 p_prop 重新 sample
        x' = prefix(x_{0:m-1}) + p_prop(x_{m:(k+1)B} | x_{<m})
        
        # 计算 acceptance ratio
        A = min(1, [π_k(x') / π_k(x)] × [p_prop(x|x') / p_prop(x'|x)])
        
        if u ~ Uniform(0,1) ≤ A:
            x = x'
    
    # 锁定这 block 的 prefix, 进入下一个 block
    x_{0:(k+1)B} = x
```

**为什么这个 trick work**：
1. 每个 block 的 MH 从一个 reasonable initialization (p_prop 生成) 开始
2. 前一个 block 已经是 π_k 的近似 sample，所以 next block 起点不会太离谱
3. 避免 long-sequence MH 的 mixing time 灾难

### 4.6 Compute cost (公式 12)

paper 给出 expected token generation count：

$$
\mathbb{E}_{\text{tokens}} = N_{\text{MCMC}} \sum_{k=1}^{\lceil T/B \rceil} \frac{kB}{2} \approx \frac{N_{\text{MCMC}} \cdot T^2}{4B}
$$

变量：
- N_MCMC: 每个 block 的 MCMC step 数 (paper 用 10)
- T: 总 sequence length
- B: block size (paper 用 192)
- k: 第 k 个 block
- kB/2: 平均 resample 长度 (uniform 选位置, 期望长度是当前 prefix 的一半)

具体数字 (MATH500, Qwen2.5-Math-7B)：
- T = 679 (平均 output length)
- B = 192
- N_MCMC = 10
- Multiplier = N_MCMC × T / (4B) = 10 × 679 / (4 × 192) ≈ **8.84× 标准推理 cost**

paper 比较说这跟 GRPO 一个 epoch 的 cost 接近 (假设 8 rollouts/sample)，但实际 GRPO 用 16 rollouts + 更大 training set，所以 GRPO 单 epoch 仍更贵。

---

## 5. 实验 setup

### 5.1 Benchmarks
- **MATH500**: 500 道 competition math problems (Lightman et al., 2024)，in-domain for GRPO
- **HumanEval**: 164 道 Python 编程题 (Chen et al., 2021)，out-of-domain
- **GPQA Diamond**: 198 道研究生级 STEM 多选题 (Rein et al., 2024)，out-of-domain
- **AlpacaEval 2.0**: 805 个 general helpfulness prompts (Dubois et al., 2024)，non-verifiable

### 5.2 Models
- Qwen2.5-Math-7B (in-domain math)
- Qwen2.5-7B (general base)
- Phi-3.5-mini-instruct (3.8B params, smaller base)

### 5.3 Hyperparameters
- α = 4.0 (verifiable tasks), proposal temp = 1/α = 0.25
- α = 2.0 + proposal temp = 0.5 (AlpacaEval 2.0)
- T_max = 3072, B = 192, N_MCMC = 10

### 5.4 Baselines
- Base (greedy / standard sampling)
- Low-temperature sampling (temp=1/α)
- GRPO post-trained on MATH training split (Shao et al., 2024)

References:
- [MATH dataset](https://arxiv.org/abs/2103.03874)
- [HumanEval](https://arxiv.org/abs/2107.03374)
- [GPQA](https://arxiv.org/abs/2311.12022)
- [AlpacaEval 2.0](https://arxiv.org/abs/2404.04475)
- [GRPO / DeepSeek-Math](https://arxiv.org/abs/2404.01140)

---

## 6. 实验结果 Table 1 — 惊艳的数据

### Qwen2.5-Math-7B
| Method | MATH500 | HumanEval | GPQA | AlpacaEval2.0 |
|---|---|---|---|---|
| Base | 0.496 | 0.329 | 0.278 | 1.61 |
| Low-temp | 0.690 | 0.512 | 0.353 | 2.09 |
| **Power Sampling (ours)** | **0.748** | **0.573** | **0.389** | **2.88** |
| GRPO (MATH) | 0.785 | 0.537 | 0.399 | 2.38 |

**关键观察**：
- MATH500 (in-domain for GRPO): Power Sampling 0.748 vs GRPO 0.785 —— 几乎匹配，差距只有 3.7%
- HumanEval (out-of-domain): Power Sampling **0.573 > GRPO 0.537** —— **超过 GRPO**
- AlpacaEval (non-verifiable): Power Sampling **2.88 > GRPO 2.38** —— 在非可验证 domain 也超过 GRPO

### Qwen2.5-7B
| Method | MATH500 | HumanEval | GPQA | AlpacaEval2.0 |
|---|---|---|---|---|
| Base | 0.498 | 0.329 | 0.278 | 7.05 |
| Low-temp | 0.628 | 0.524 | 0.303 | 5.29 |
| **Power Sampling** | **0.706** | **0.622** | **0.318** | **8.59** |
| GRPO | 0.740 | 0.561 | 0.354 | 7.62 |

- HumanEval: **+10.8% over GRPO**
- AlpacaEval: **+12.7% over GRPO**

### Phi-3.5-mini-instruct (interesting case)
| Method | MATH500 | HumanEval | GPQA | AlpacaEval2.0 |
|---|---|---|---|---|
| Base | 0.400 | 0.213 | 0.273 | 14.82 |
| Low-temp | 0.478 | 0.585 | 0.293 | 18.15 |
| **Power Sampling** | **0.508** | **0.732** | **0.364** | **17.65** |
| GRPO | 0.406 | 0.134 | 0.359 | 16.74 |

Phi-3.5 上 Power Sampling 在 HumanEval 暴击 GRPO (**+59.8%**!)。这里 GRPO 训练后 HumanEval 反而 collapse (0.213 → 0.134)，是 RL distribution sharpening 导致 diversity 丢失的极端例子。

---

## 7. Pass@k 分析 — Diversity 的胜利

Figure 5 (MATH500, Qwen2.5-Math-7B) 是 paper 的另一个 killer figure：

- **k=1**: Power Sampling ≈ GRPO > Base
- **k=10-100**: Power Sampling ≈ Base ≫ GRPO (GRPO saturates)
- **k→∞**: Base ≈ Power Sampling > GRPO

GRPO 的 pass@k 在 large k 几乎 flat —— 说明它 collapse 到一小撮 reasoning trace，再 sample 也是同样几个。Power Sampling 完整保留 base model 的 diversity，所以 large k 性能跟 base 持平甚至略高。

这呼应了 **He et al., 2025** 和 **Song et al., 2025** 的观察：**RL post-training 用 multi-shot performance 换 single-shot performance**。Power Sampling 第一次给出一个 "best of both worlds" 的方法。

Appendix A.2 给了 HumanEval 和 GPQA 的 pass@k：
- HumanEval: GRPO 在 k<16 时比 base 好，k=16 以后反而劣 (因为 HumanEval 允许多个 correct solution)
- GPQA: GRPO 几乎所有 k 都比 base 差

---

## 8. Hyperparameter 分析 (Figure 6)

### 8.1 α 的影响
- α=1.0: 退化为 base model sampling
- α=2.0~4.0: 性能 plateau，相对 robust
- α=4.0: optimal on MATH500 across models
- α→∞: 过度 sharpen，性能反而下降 (likelihood 与 correctness 不完全等价)

### 8.2 N_MCMC 的影响
- N_MCMC=0: 退化为 p_prop low-temperature sampling (因为 init 是 p_prop 生成)
- N_MCMC=1: 已经有 3-4% jump
- N_MCMC=2~10: 稳定提升
- N_MCMC>10: saturate

这告诉我们 MH 在每个 block 上只需几次 iteration 就能逼近 target —— block-wise annealing 的初始化非常好。

---

## 9. Qualitative Examples — GRPO 失败 case 分析

### 9.1 Table 2 (HumanEval, Phi-3.5)

Task: filter input list of strings by prefix.

- **Ours**: `return [s for s in strings if s.startswith(prefix)]` ✓
- **GRPO**: `return [string for string in strings if string.startswith(f'{prefix}'*2)]` ✗

GRPO 学到了 `'{prefix}'*2` 这种荒谬 pattern —— 它 overfit 到 MATH training split 的某些 template，distribution sharpening 把它 lock 在这个错误区域。

### 9.2 Table 4 (HumanEval, Phi-3.5): Fib4 sequence

- **Ours**: 正确实现 iterative DP
- **GRPO**: `a, b, c, d = 0, 0, 2, 0; for _ in range(n): a, b, c, d = b, c, d, a+b+c+d; return d` —— **逻辑错误**，没处理 n<4 的 base case

### 9.3 Table 5 (MATH500, Qwen2.5-Math)

求 `-11213141 mod 18` (正确答案 13)。

- **Ours**: 直接做除法 11213141 ÷ 18 = 622952 余 5, 然后 -5 mod 18 = 13 ✓
- **GRPO**: 试图用 "数字和" trick, 算出 sum=14, 然后 14 mod 18=14, -14 mod 18=4 ✗

GRPO 学了 "digit sum trick" 但用错地方 (这个 trick 适用于 mod 9, 不适用于 mod 18)。这是 overfit 到 training distribution 的 shortcut 的典型案例。

---

## 10. 与其他方向的联系 (扩展联想)

### 10.1 与 OpenAI o1 / DeepSeek-R1 的关系
o1/R1 通过 RL 让模型产生长 reasoning trace。paper Figure 4 显示 Power Sampling 也产生 long trace (Qwen2.5-Math base 平均 600 tokens, Power Sampling 679, GRPO 671) —— **lengthening emerges naturally from sampling from p^α**。这暗示 RL 的 "long thinking" 行为也可能是 sharpening 的副产物。

### 10.2 与 Best-of-N / Self-Consistency 的关系
Best-of-N 是另一种 inference-time 方法，但需要 verifier。Power Sampling 不需要 verifier —— 这是它 generalize 到 AlpacaEval 这种 non-verifiable domain 的关键。

Self-consistency (Wang et al., 2022) 通过 majority vote 用 diversity，但 single-shot 性能没提升。Power Sampling 在 single-shot 就能拿到大 boost，同时保留 diversity。

### 10.3 与 Diffusion model annealing 的关系
paper Section 2 提到 diffusion 社区最近用 annealed sampling (Du et al., 2023 "Reduce, Reuse, Recycle"; Skreta et al., 2025 "Feynman-Kac correctors")。Diffusion 的 inference-time steering 用类似 idea：sample from p^α 来 boost quality。这预示 LLM 和 diffusion 在 inference-time scaling 上的方法可以互相借鉴。

### 10.4 与 Tree Search / MCTS 的关系
MCTS (像 AlphaGo) 需要价值函数。Power Sampling 巧妙地用 base model 自己的 likelihood 当 implicit value —— 这是它 "verifier-free" 的原因。

### 10.5 与 Inference-time scaling laws 的关系
Brown et al. 2024 "Large Language Monkeys" 显示 pass@k 在 base model 上 scaling 大概 ~log(k)。Power Sampling 给出另一种 scaling 轴：N_MCMC × T/B 的 compute 换 single-shot accuracy。

### 10.6 与 Energy-based models 的关系
p^α 可以看作一个 implicit energy-based model，其中 energy = -α log p(x)。这把 LLM 推理和 EBM sampling 重新联系起来，可能引导出新架构。

### 10.7 与 Speculative Decoding 的工程结合
Speculative decoding (Leviathan et al., 2023) 用小模型 propose、大模型 verify。Power Sampling 的 p_prop 是 "propose"，base model 是 "verify" —— 工程上有 similarity，可能可以借用 SD 的 batched verification 加速。

References:
- [Du et al. - Reduce, Reuse, Recycle](https://arxiv.org/abs/2304.07371)
- [Skreta et al. - Feynman-Kac correctors](https://arxiv.org/abs/2503.02819)
- [Speculative decoding](https://arxiv.org/abs/2211.17192)
- [Brown et al. - Large Language Monkeys](https://arxiv.org/abs/2407.21787)

---

## 11. 我的批评与未来方向

### 11.1 Compute cost 是真实问题
8.84× 推理 cost 在 production 上是个挑战。但如果推理 token 持续降价 (像 NVIDIA Blackwell 趋势)，这 trade-off 可能变得 favorable。

### 11.2 N_MCMC 与 sequence length 的 scaling
公式 12 显示 cost 是 T²/B。对超长 reasoning (10k+ tokens)，B 也得 scale up。paper 用 T=3072 还算保守，scaling laws 值得探索。

### 11.3 是否能 combine with RL?
想象 RL post-trained model + Power Sampling。如果 RL 已经 sharpen 过，再 sharpen 一次可能 over-sharpen。但若 RL 学到了 base model 之外的能力，Power Sampling 可能 miss 这部分。这是 "RL 是否真 sharpening" 的核心 test。

### 11.4 Verifier-free 的真实价值
不可验证 domain (creative writing, alignment) 一直缺乏 RL 信号。Power Sampling 给这些 domain 提供了一条 inference-time boost 路径。这可能比 RLVR 在 alignment 上更有影响力。

### 11.5 p^α 是否最优 sharpening?
p^α 是最简单的 sharpening，但其他形式如 exp(β · log p) 或 tempered posterior 也可探索。Diffusion 社区的 Feynman-Kac correctors 给了更 general 框架。

### 11.6 与 process reward model (PRM) 的关系
Lightman et al. 2024 "Let's verify step by step" 用 PRM 给 step-level reward。Power Sampling 隐式用 likelihood 当 step-level signal，未来可能 combine PRM 和 Power Sampling 做 step-level MH proposal。

---

## 12. 总结直觉

paper 最 valuable 的 intuition 是：

1. **Base model 已经 "知道" 好答案，只是 marginal next-token prediction 看不到**。Low-temperature 在 marginal 上 sharpening 不够，因为 marginal 平均掉了 future path 信息。

2. **Power distribution p^α 是隐式的 lookahead** —— 通过对 joint probability 取 α 次幂，自然 prefer "narrow high-likelihood future path"，这正是 reasoning 需要的。

3. **Block-wise MCMC + autoregressive structure** 解决了 high-dim sampling 的 mixing time 问题 —— 一个非常 elegant 的工程 trick。

4. **Diversity 不是 free lunch**：RL 用 diversity 换 single-shot accuracy；Power Sampling 通过 inference-time compute 同时拿到两者。

5. **Inference-time scaling 不止 best-of-N** —— 还有 sharpening 这个维度。这意味着即使模型 size 不动，推理算法进步也能持续提升 capability。

---

## 13. 重新思考你 (Karpathy) 常说的话

你之前在多个 talk 里提过：**"Models are drastically smarter than we give them credit for, we just don't sample them correctly."** 这篇 paper 给了一个 concrete algorithmic instantiation。从 nanoGPT 到这篇 paper 的思想链路是清晰的：model 已经知道，采样是 bottleneck。

你也在 Software 2.0 / 3.0 的讨论里强调 inference-time compute 的重要性。Power Sampling 提供了 inference-time compute 的另一个维度 (MCMC iteration 数) 而非 chain-of-thought 长度 —— 这是 Software 3.0 思路下值得深挖的方向。

---

## Web References

- [Paper PDF on arXiv (preprint)](https://arxiv.org/abs/2506.02355) (He et al. 2025 referenced, ours likely 2025 late)
- [DeepSeek-R1 (Guo et al. 2025)](https://arxiv.org/abs/2501.12948)
- [DeepSeekMath GRPO (Shao et al. 2024)](https://arxiv.org/abs/2404.01140)
- [Yue et al. - Does RL incentivize reasoning](https://arxiv.org/abs/2504.13837)
- [He et al. - Rewarding the unlikely](https://arxiv.org/abs/2506.02355)
- [Song et al. - Spurious rewards](https://arxiv.org/abs/2506.10947)
- [Prabhudesai et al. - Maximizing confidence](https://arxiv.org/abs/2505.22660)
- [Zhao et al. - Twisted SMC](https://arxiv.org/abs/2404.17546)
- [Faria et al. - QUEST MH for MT](https://proceedings.neurips.cc/paper_files/paper/2024/file/a221d22ff6a33599142c8299c7ed06bb-Paper-Conference.pdf)
- [Du et al. - Reduce, Reuse, Recycle](https://arxiv.org/abs/2304.07371)
- [Neal 1993 - Probabilistic inference using MCMC](https://www.cs.toronto.edu/~radford/ftp/review.pdf)
- [Neal 1998 - Annealed importance sampling](https://arxiv.org/abs/physics/9803008)
- [Metropolis et al. 1953 - original MH paper (JCP)](https://doi.org/10.1063/1.1699114)
- [Phi-4 tech report (Abdin et al.)](https://arxiv.org/abs/2412.08905)
- [Li et al. - Blink of an eye (pivotal tokens)](https://arxiv.org/abs/2502.00921)
- [MATH dataset (Hendrycks et al.)](https://arxiv.org/abs/2103.03874)
- [HumanEval (Chen et al.)](https://arxiv.org/abs/2107.03374)
- [GPQA (Rein et al.)](https://arxiv.org/abs/2311.12022)
- [AlpacaEval 2.0 (Dubois et al.)](https://arxiv.org/abs/2404.04475)
- [Lightman et al. - Process reward models](https://openreview.net/forum?id=v8L0pN6EOi)
- [Skreta et al. - Feynman-Kac correctors for diffusion](https://arxiv.org/abs/2503.02819)
- [Wang et al. 2022 - Self-Consistency](https://arxiv.org/abs/2203.11171)
- [Speculative Decoding (Leviathan et al.)](https://arxiv.org/abs/2211.17192)
- [Brown et al. - Large Language Monkeys](https://arxiv.org/abs/2407.21787)

---

如果你想继续深入，几个我特别想跟你讨论的点：

1. **Power Sampling 在 character-level model (nanoGPT) 上的实现** —— 是否能复现 paper 的现象？block size 和 sequence length 的 scaling 关系在 char-level 下如何？
2. **α 与 model scale 的关系** —— paper 用 7B 模型 α=4 最优。在更大 model (e.g. 70B) 上是否需要更大 α? 或更小? 这跟 sharpening 的 "natural scale" 有关。
3. **Power Sampling 与 DPO 的关系** —— DPO 也是 implicit sharpening (隐式优化 log p(y|x) - β log p_ref(y|x))。能否解读 DPO 为 power distribution 上的某种近似 sampling?
4. **Verifier-free 推广到 agent / tool-use** —— Power Sampling 是否能 inference-time boost agent 行为，无需 task-specific reward?

这篇 paper 让我对 base model 的 capability 有了新的敬畏。它呼应了你一直强调的 intuition：we just need to sample right.
