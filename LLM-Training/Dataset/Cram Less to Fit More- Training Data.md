---
source_pdf: Cram Less to Fit More- Training Data.pdf
paper_sha256: 8c0b633cec72cae523846384bdb24e13b16df410ce70ba104d2f9a5b7f5c3582
processed_at: '2026-08-03T17:40:57-07:00'
target_folder: LLM-Training/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：Cram Less to Fit More

Andrej，我把这篇 paper 用大白话给你讲一遍，同时 build intuition。

---

## TL;DR

小模型装不下 Wikipedia 所有 facts。硬装的结果是：**每个 fact 都记住了一点点，但没有一个 fact 能完整 recall**。Fact accuracy 直接归零。

解法很反直觉：**删掉 80% 的 data**。只让模型学那 20% 它能真正记住的 facts。结果：110M 的小模型，fact accuracy 反而 match 了 1.3B 的大模型。

---

## 故事是这样开始的

你 train 一个 GPT2-Small（110M params）on Wikipedia 3B tokens。你期望它能记住 "Einstein born in 1879" 这种 facts。你 run 了 8 epochs，loss 蛮低的，感觉良好。

然后你 test 它：

> Q: "When was Einstein born?"
> Model: "1897" (错)

你换个问题：

> Q: "Who wrote the Declaration of Independence?"
> Model: "Thomas Edison" (错)

所有 fact 都答错。**All of them**。

但你查 training loss，明明是低的。怎么回事？

---

## Intuition: 杯子装水的类比

把 model 想象成一个杯子，capacity 是 **2 bits/param**。110M params × 2 bits = 220 Mbits 的容量。

Wikipedia 里有 60M 个 facts，每个 fact 平均 ~73 bits（比如 22 位数字的电话号码，$\log_2(10^{22}) \approx 73$ bits）。

总信息量：$60\text{M} \times 73 \approx 4.4 \text{Gbits}$。

杯子 220 Mbits，要装 4.4 Gbits 的水。**装不下**，会溢出来——准确说，会被均匀 distribute 到所有 facts 上，每个 fact 分到 $\frac{220\text{M}}{60\text{M}} \approx 3.7$ bits。

一个 fact 要 73 bits 才能完整 recall。你只给了它 3.7 bits，肯定答不对。Stochastic decoding 下，correctness probability $= e^{-\text{loss}}$，loss 仍然有 $\approx 73 - 3.7 \approx 69$ bits，所以 $e^{-69} \approx 0$。

**结果：每个 fact 都只记住一丁点碎片，没有一个 fact 能完整 recall，accuracy = 0。**

这就是 Figure 2a 蓝色曲线在 facts > 3M 时崩到 0 的原因。

---

## Power-law 让事情更糟

Real-world data 不是 uniform 的。Wikipedia 里 "Einstein" 出现 10000 次，"Pterostylis stricta"（某种兰花）只出现 2 次。这是 power law：$\Pr[\text{fact}_i] \propto 1/i^\beta$，$\beta$ 越大越 skew。

Skew 的后果：
- **Frequent facts 被 over-memorized**："Einstein" 出现 10000 次，模型花 10000 倍的 capacity 去记它，但其实记 1 次就够了，浪费 9999 份 capacity
- **Rare facts 完全没被 memorized**：那个兰花出现 2 次，模型根本没学到

所以 power law ($\beta=1.0$) 比 uniform ($\beta=0$) 的情况更糟。Figure 2 显示 $\beta=1.0$ 时即便只有 1.6M facts（远低于 3M capacity），accuracy 也只有 1.18M。

---

## Theory: Corollary 3.7 的简单版

Theorem 3.6 用 Fano's inequality 证明：

$$\mathbb{E}[\text{Accurate Fact Count}] \leq \frac{\ln|\mathcal{W}| + N\ln 2}{b}$$

变量解释：
- $|\mathcal{W}|$ = model 的 parameter space 大小，$\ln|\mathcal{W}|$ = total capacity in nats
- $N$ = 总 facts 数
- $b$ = 每个 fact 的 entropy（bits 量）
- $N\ln 2$ 是来自 binary accuracy indicator 的修正项

当 $N \cdot b \gg \ln|\mathcal{W}|$（i.e. data 信息量远超 model capacity），bound 简化为 $\ln|\mathcal{W}|/b$。

对 110M model + 22-digit phonebook：$\frac{2 \times 110\text{M}}{73} \approx 3.01\text{M facts}$。

这是理论上能准确记住的 facts 数量上限。

---

## 关键实验：Standard Training 远未达到 capacity

Figure 2a 蓝色曲线（full dataset training）：
- 1.6M facts → accuracy 1.53M（接近 capacity，因为 1.6M < 3M，能装下）
- 2.24M facts → accuracy 1.90M
- 5.12M facts → accuracy **0.01M**（崩了）
- 10.24M facts → accuracy **0**（完全归零）

**Model 明明有 3M 的 capacity，5M facts 时 accuracy 就崩到 0 了**。不是 capacity 不够，是 **training data distribution 导致 capacity allocation 低效**。

Model 把 capacity 平均 spread 到所有 5M facts 上，每个 fact 分到 $\frac{220\text{M}}{5\text{M}} = 44$ bits。73 bits 的 fact 只记住了 44 bits，recall 时 $e^{-(73-44)} = e^{-29} \approx 0$。

Ablation（Figure 6）：
- Train 8x longer → 几乎没用（shaded area 微小）
- 10x larger model (1.4B) → 能完美 memorize 全部 facts

这证明：**问题不是 optimization 不够，是 capacity allocation 低效**。

---

## Solution: Loss-based Data Selection

### 核心 insight

**Loss 是 fact frequency + entropy 的 proxy**。

- Low loss → frequent fact（见过很多次）or easy fact（low entropy）
- High loss → rare fact or high-entropy fact

所以 loss 从低到高 sort，等价于按 frequency 从高到低 sort。

### Algorithm 1: LossHF

```
每个 step:
1. Sample 一个 batch B
2. 算每个 record 的 sum-of-token-CE-loss ℓ(x; θ_t)
3. 算 τ = lower-percentile_α(所有 ℓ 值)  # 只保留最低 α 比例
4. 对每个 record x:
   - 如果 ℓ(x) ≤ τ: 以概率 ℓ(x)/τ 保留
   - 如果 ℓ(x) > τ: 丢掉
5. 累积到 batch size b，做一次 SGD update
```

两个操作合起来：
- **Head selection**（丢掉 high-loss 的 records）：限制 facts 数量到 model capacity 内
- **Flattening**（低 loss 的 records 保留概率更低 $\propto \ell/\tau$）：把 frequent facts 的 effective frequency 拉平到接近 uniform

为什么 flatten 用 $\ell/\tau$ 这个概率？因为低 loss 的 fact 已经被模型"记住"了，再看也不会提升 accuracy，纯粹浪费 capacity。Down-sample 它，把 capacity 留给 loss 稍高（但仍在 α 内）的 facts。

### 为什么必须 fact-level 而非 token-level

这很关键。如果你像 [Rho-1](https://arxiv.org/abs/2404.13936) 那样按 token-level 选，会 break fact boundaries。比如 fact "Einstein was born in 1879" 有 4 个 tokens，你选了 "Einstein was born in" 但丢掉了 "1879"，模型永远学不到这个 fact。

Paper 在 Section D.2 验证：token-level selection variant 下，fact accuracy 仍然为 0，不管 $\alpha$ 怎么调。

Fact-level selection 必须知道 fact boundaries。Wikipedia 实验中用 `<|start_of_fact|>...<|end_of_fact|>` annotation，来自 [Zhao et al. 2025b](https://arxiv.org/abs/2505.15962)。

### 为什么用 sum-of-loss 而非 average-of-loss

考虑两个 facts：
- Fact A: 10 tokens，每个 token loss 0.1，sum = 1.0，average = 0.1
- Fact B: 2 tokens，每个 token loss 0.5，sum = 1.0，average = 0.5

如果按 average 选，A 看起来 "更 easy"（0.1 < 0.5），但其实 A 有 10 tokens 要记，B 只有 2 tokens。Sum 才反映 total information content。

Figure 11 实验：在 heterogeneous prefix/suffix lengths 的 phonebook 上，sum-of-loss 与 fact weight 的 Spearman correlation 远高于 average-of-loss。

这与 [Allen-Zhu & Li 2024](https://arxiv.org/abs/2404.05405) Theorem 3.2 的 loss-based memorization lower bound 一致——memorization $\geq H(\text{facts}) - \sum \ell(\text{fact})$，用的是 sum。

### Online threshold 为什么必要

每个 batch 重新算 percentile $\tau$，而不是用 global loss 统计。Reason（Figure 10）：

训练早期，loss 主要反映 format 难度（"1879" vs "abcde" 哪个更难 predict）。训练后期，loss 才主要反映 fact frequency（frequent facts 的 loss 降得快）。

Spearman correlation between $-\ell$ 和 fact weight 随训练 step 上升，尤其当 data > capacity 时。所以 online threshold 能更好 adapt 到 training dynamics。

---

## Phonebook 实验结果

Table 2 关键数据（110M model, $\beta=1.0$ power law, 5.12M facts）：

| Method | Accurate Fact Count |
|---|---|
| Full Dataset (800k steps) | 0.86M |
| Full Dataset (8x longer = 6.4M steps) | 0.85M |
| Oracle Head | 1.19M |
| Oracle Head-Flattened | **1.87M** |
| LossH (ours) | 1.00M |
| LossHF (ours) | **1.72M** |

LossHF 几乎 match oracle Head-Flattened（1.72M vs 1.87M），证明 loss 是 excellent proxy for fact frequency。

注意 capacity limit 是 3.01M，LossHF 达到 1.72M，还没到 capacity。为什么？因为 $\beta=1.0$ 时 frequent facts 太频繁，flatten 没法完全 equalize（loss proxy 有 approximation error）。Uniform 分布下 LossHF 能 reach capacity 的 80%+。

---

## Wikipedia 实验：Main Result

110M model，Wikipedia 3B tokens，66k steps，batch 320，~8 epochs。

### Figure 4 + Table 1 核心数据

| Model | α | Test Fact Accuracy | NLU Accuracy |
|---|---|---|---|
| 110M | 1.0 (full) | 696 | 0.364 |
| 110M | 0.2 (LossH-Wiki) | **929** | 0.360 |
| 335M | 1.0 (full) | ~870 | ~0.37 |
| 1.3B | 1.0 (full) | ~930 | ~0.37 |

**110M + 20% data = 1.3B + 100% data on fact accuracy**。10x compute saving。

NLU（CommonsenseQA, HellaSwag, PIQA, SocialIQA, ARC-Easy）基本持平，证明 prune facts 不伤 general language understanding capability。

MMLU-Knowledge subset（11 个 knowledge tasks）: LossH-Wiki with α=0.5 matches 335M Full (3x larger)。

### Random Pruning Baseline（Table 1）

Random fact-level pruning（选 α fraction 的 facts，但随机选）：
- α=0.2: Random=647, LossH-Wiki=929
- α=0.5: Random=676, LossH-Wiki=852
- α=1.0: Random=696, LossH-Wiki=696（相同，因为没选）

Random pruning 完全没用，证明 **loss-based selection 是 essential**，不只是 subsetting。

---

## LoRA Finetuning on arXiv Papers

171k arXiv papers published in 2025（after Llama-3.2-1B cutoff）。Task: title-to-authors mapping。

这模拟 "teaching pretrained model new knowledge" 的现实场景。

Table 4：

| LoRA rank | Full Data Fact Acc | LossHF Fact Acc |
|---|---|---|
| r=4 | 0.2% | 7.9% |
| r=8 | 0.2% | 18.1% |
| r=16 | 0.6% | 34.6% |
| r=32 | 3.6% | **55.5%** |

Full data training 下 fact accuracy 接近 0（LoRA rank-32 的 capacity 远小于 171k facts）。LossHF 提升 **15-150x**。

General capabilities（HellaSwag, CommonsenseQA, PIQA, MMLU, ARC）基本不变，证明 LossHF **不加剧 catastrophic forgetting**。这与 [Sanyal et al. 2025](https://arxiv.org/abs/2502.02797) "upweight easy samples reduces forgetting" 的结论一致。

---

## 为什么与 Rho-1 相反

[Rho-1](https://arxiv.org/abs/2404.13936) 和 [Rho-Loss](https://arxiv.org/abs/2207.07018) 选的是 **high excess loss** 的 tokens（模型最不确定的），与 LossH/LossHF 选 low loss 的 facts **方向相反**。

Paper Section 2 论证：Rho-1 的 benefits **不来自 better fact memorization**。Rho-1 选 high-loss tokens 会把 capacity spread 到更多 facts 上，对 fact accuracy 反而有害。

Rho-1 的好处应该来自 generalization 和 reasoning——那些 tasks 不需要"完整记住 single fact"，需要的是 learning patterns across many examples。

Open question: 能否设计一个 selection method 同时 optimize fact accuracy 和 generalization？目前看这两个目标在 selection direction 上是 opposite 的。

---

## Practical Recipe

如果你要 pretrain 一个 <1B model 且关心 knowledge：

1. **准备 fact-annotated corpus**。用 lightweight LM 自动标注 fact boundaries，参考 [Maini et al. 2024](https://aclanthology.org/2024.acl-long.755/) 的 rephrasing 方法。

2. **Warmup training**：先在 full data 上训几个 epoch，让 loss 信号稳定（反映 frequency 而非 format）。

3. **启用 LossHF selection**：
   - 每个 batch 算 per-fact sum-of-loss
   - 保留 low-loss α fraction（用 online percentile）
   - Low-loss 的 facts 进一步 down-sample（prob $\propto \ell/\tau$）
   - Upscale selected fact tokens 的 loss weight，保持 fact/non-fact ratio

4. **Selection ratio $\alpha$ 的选择**：理论上 $\alpha \cdot N_{\text{facts}} \approx 2 \cdot N_{\text{params}} / b$。实际中 grid search $\alpha \in \{0.1, 0.2, ..., 1.0\}$。Paper 发现 Wikipedia 上 110M 最优 $\alpha=0.2$，arXiv LoRA r=32 最优 $\alpha$ 更小。

5. **Scaling law**（equation D.1）：若 facts power law 分布 exponent $\beta$，则 optimal $\alpha$ 随 model size 变化：
   - $\beta=0$ (uniform): $\alpha \propto |\mathcal{W}|$（线性增长）
   - $0 < \beta < 1$: $\alpha \propto |\mathcal{W}|^{1-\beta}$
   - $\beta=1$: $\alpha \propto \ln\ln|\mathcal{W}|$（极慢）

这可以用来从小 model 的 optimal $\alpha$ 推大 model 的 $\alpha$，省去大 model 上的 grid search。

---

## 这篇 paper 的 limitation

1. **Fact boundaries 需要 annotation**。Real-world corpus 没有 `<|start_of_fact|>...<|end_of_fact|>`。Paper 说用 LM 自动标注，但没实验验证 scalability。

2. **只考虑 dense transformer**。MoE 架构可能更适合 fact memorization，参考 [Jelassi et al. 2024 "Mixture of Parrots"](https://arxiv.org/abs/2410.19034)——experts improve memorization more than reasoning。

3. **Sufficient training regime only**。所有实验都 train to convergence。实际中 fixed FLOPs budget 下，selection 的 forward pass overhead 是否值得？Open question。

4. **Fact 定义太 narrow**。"Einstein was born in 1879" 是 fact，但 "Einstein was a physicist" 算不算？Paper 的 definition 要求 deterministic mapping，但 real knowledge 很多是 probabilistic 或 multi-hop 的。

5. **Power law exponent 估计**。Real-world data 的 $\beta$ 不知道，scaling law 难以直接 apply。

---

## Big Picture

这篇 paper 在我看来填补了一个重要 gap：

[Chinchilla](https://arxiv.org/abs/2203.15556) 告诉我们：**给定 compute，应该选多少 tokens**。答案是 ~20 tokens/param。

这篇 paper 告诉我们：**给定 model size，应该选多少 facts**。答案是 ~2 bits/param 的 total information content，而且要 **flatten frequency distribution**。

两个 axis 正交：
- Chinchilla optimize compute efficiency
- 这篇 optimize capacity utilization for fact memorization

Combine 起来可能是：先按 Chinchilla 确定 tokens 总量，再按 capacity-aware selection 确定其中 facts 的数量和分布。

对 frontier model 的含义：GPT-4 ~1T params，capacity ~250 GB factual info。但 SimpleQA 只有 ~50% accuracy，说明 **real-world training 的 capacity utilization 远低于 50%**。这篇 paper 的 method 可能是提升 frontier model factuality 的一条 path。

---

## 相关工作链接

- [Chinchilla scaling laws](https://arxiv.org/abs/2203.15556)
- [Allen-Zhu & Li "Physics of LM" Part 3.3: Knowledge Capacity](https://arxiv.org/abs/2404.05405)
- [Allen-Zhu & Li Part 3.1: Knowledge Storage](https://arxiv.org/abs/2309.14316)
- [Morris et al. 2025: How much do LMs memorize?](https://arxiv.org/abs/2505.24832)
- [Jelassi et al. 2024: Mixture of Parrots](https://arxiv.org/abs/2410.19034)
- [Gu et al. 2025: Data mixing phase transitions](https://arxiv.org/abs/2505.18091)
- [Zucchet et al. 2025: How do LMs learn facts?](https://arxiv.org/abs/2503.21676)
- [Rho-1](https://arxiv.org/abs/2404.13936)
- [Rho-Loss / Mindermann et al. 2022](https://arxiv.org/abs/2207.07018)
- [Feldman 2020: Does learning require memorization?](https://arxiv.org/abs/2003.02933)
- [Kandpal et al. 2023: Long-tail knowledge](https://arxiv.org/abs/2211.08411)
- [Mallen et al. 2023: When not to trust LMs](https://arxiv.org/abs/2305.18290)
- [Zhao et al. 2025b: Memory LM pretraining](https://arxiv.org/abs/2505.15962)
- [Sanyal et al. 2025: Upweight easy samples](https://arxiv.org/abs/2502.02797)
- [Maini et al. 2024: Rephrasing the web](https://aclanthology.org/2024.acl-long.755/)
- [Su et al. 2025: Nemotron-CC](https://aclanthology.org/2025.acl-long.235/)
- [SimpleQA](https://arxiv.org/abs/2411.04368)
- [Lee et al. 2022: Deduplication](https://aclanthology.org/2022.acl-long.113/)
- [Xie et al. 2023: Data selection via importance resampling](https://arxiv.org/abs/2306.16944)
- [Pythia suite](https://arxiv.org/abs/2304.01373)
- [OLMo](https://arxiv.org/abs/2402.00838)
- [Llama 3](https://arxiv.org/abs/2407.21783)
- [Carlini et al. 2022: Quantifying memorization](https://arxiv.org/abs/2202.07646)
- [Fano's inequality (Wikipedia)](https://en.wikipedia.org/wiki/Fano%27s_inequality)
- [Cheng et al. 2026: Conditional memory via lookup](https://arxiv.org/abs/2601.07372)
- [Pouransari et al. 2025: Hierarchical memories](https://arxiv.org/abs/2510.02375)

---

## Final intuition

一句话总结：**模型是个固定大小的杯子，factual knowledge 是固定大小的冰块。装太多冰块会全碎，装 capacity 数量的冰块且 size 均匀，每块都能完整保留**。

Loss 是 model 告诉你 "哪些冰块我见过很多次（low loss）/很少见（high loss）" 的信号。用这个信号 head-select（只装能装下的）+ flatten（把见过太多次的切小），就能 maximize 完整冰块的数量。

反直觉但 elegant。110M model 用 20% Wikipedia data match 1.3B 用 100% data——这就是 "Cram Less to Fit More"。

---

# Paper 解读：Cram Less to Fit More

## 1. 核心问题

Karpathy 你会喜欢这篇 paper 的，因为它直接挑战了一个常见的 misconception：**train on more data is always better**。在 fact memorization 这个特定场景下，train on full dataset 反而会导致 fact accuracy 接近 0，而 prune 掉大部分 data 之后，small model 能 match 10x larger model。

paper 的关键 insight 是 information-theoretic 的：当 training data 中 facts 的总信息量超过 model capacity 时，模型无法完整 memorize 任何 single fact，于是 "accuracy"（严格二值判断）collapse。这是 phenomenologically 有趣的——loss 仍然在下降，模型确实在 "记住" 一些 partial information，但任何 single fact 都不能被准确 recall。

参考 [Feldman 2020](https://arxiv.org/abs/2003.02933) 和 [Allen-Zhu & Li 2024](https://arxiv.org/abs/2404.05405) 关于 memorization capacity 的工作。

---

## 2. Fact Memorization 的信息论定义

### 2.1 Facts as Deterministic Mappings

Definition 3.1 把 fact 定义为 $(Q_i, A_i)$，其中 $Q_i$ 是 fixed question string（与 distribution 无关），$A_i: \Theta \to Y_i$ 是从 world state $\theta$ 到 answer 的确定性映射。

举例：$Q_i = $ "When was Einstein born?"，$A_i(\theta) = $ "1879" in world $\theta$。

这里 world $\theta$ 是从 meta-prior $\Psi$ 中采样的——这 capture 了 algorithm 对 "world state" 的 uncertainty。一个 learning algorithm $\mathcal{A}$ 接受 dataset $D$（从 $\mathcal{P}_\theta^n$ 采样）并输出 trained model $\mathcal{A}(D) \in \mathcal{W}$。

### 2.2 Fact Accuracy (Definition 3.4)

$$\text{Acc}_{(Q_i, A_i)_{i=1}^N}(\mathcal{A}; \theta, n) = \frac{\sum_{i=1}^N \Pr_{D \sim \mathcal{P}_\theta^n, \mathcal{A}}[f(\mathcal{A}(D); Q_i) = A_i(\theta)]}{N}$$

变量解释：
- $\mathcal{A}$：learning algorithm
- $\theta$：current world state
- $n$：dataset size
- $f(\mathcal{A}(D); Q_i)$：trained model 对 question $Q_i$ 的 prediction
- $I_i \triangleq \mathbf{1}[f(\mathcal{A}(D); Q_i) = A_i(\theta)]$：accuracy indicator（per-fact）
- Nominator 称为 **Accurate Fact Count** (Acc-Cnt)

注意 prediction function $f$ 对于 free-form QA 用的是 stochastic decoding，所以 $\Pr[f = a] = e^{-\ell(a; \mathcal{A}(D), Q_i)}$，即 correctness probability 随 sum of token cross-entropy loss 指数衰减。这个细节很重要——后面 Theorem 3.6 的 bound 依赖这个 stochastic decoding 假设。

### 2.3 Fact Memorization (Definition 3.5)

$$\text{Mem}_{(Q_i, A_i)_{i=1}^N}(\mathcal{A}; \Psi, n) = I\big((A_1(\theta), \dots, A_N(\theta)); \mathcal{A}(D)\big)$$

这是 **mutual information** between fact values 和 trained model。区别于 prior work 如 [Brown et al. 2021](https://arxiv.org/abs/2012.07705) 的 unintentional memorization $I(D; \mathcal{A}(D))$（dataset-specific 信息），这里关注的是 **关于 distribution 本身的信息**。

### 2.4 Capacity Limit (Proposition A.1)

$$\text{Mem} \leq \ln|\mathcal{W}|$$

证明很简洁：
$$I((A_1, \dots, A_N); \mathcal{A}(D)) = H(\mathcal{A}(D)) - H(\mathcal{A}(D) | A_1, \dots, A_N) \leq H(\mathcal{A}(D)) \leq \ln|\mathcal{W}|$$

最后一步用了 discrete space $\mathcal{W}$ 上 uniform distribution 有最大 entropy $\ln|\mathcal{W}|$。

对于 bfloat16 训练的 110M model：$\ln|\mathcal{W}| \approx 110\text{M} \times 16 \times \ln 2 = 1.22 \times 10^9$ nats。但 [Allen-Zhu & Li 2024](https://arxiv.org/abs/2404.05405) 和 [Morris et al. 2025](https://arxiv.org/abs/2505.24832) 实验上发现 effective capacity 是 **2 bits/param**，所以实际约 $220$ Mbits $\approx 152$ Mnats。

---

## 3. Theorem 3.6：从 Accuracy 推回 Memorization

这是 paper 的理论核心，把 accuracy 和 capacity 严格挂钩。

### 3.1 推导

从 mutual information 展开：
$$\text{Mem} = H(A_1, \dots, A_N) - H(A_1, \dots, A_N | \mathcal{A}(D))$$

用 entropy sub-additivity：
$$H(A_1, \dots, A_N | \mathcal{A}(D)) \leq \sum_{i=1}^N H(A_i | \mathcal{A}(D))$$

所以：
$$\text{Mem} \geq H(A_1, \dots, A_N) - \sum_{i=1}^N H(A_i | \mathcal{A}(D))$$

对每个 $H(A_i | \mathcal{A}(D))$ 应用 [Fano's inequality](https://en.wikipedia.org/wiki/Fano%27s_inequality)：
$$H(A_i | \mathcal{A}(D)) \leq H(I_i) + \Pr[I_i = 0] \cdot H(A_i | I_i = 0)$$

其中 $I_i = \mathbf{1}[f(\mathcal{A}(D); Q_i) = A_i(\theta)]$ 是 binary accuracy indicator。

合起来得到 Theorem 3.6：
$$\text{Mem} \geq H(A_1, \dots, A_N) - \sum_{i=1}^N \big(\Pr[I_i=0] \cdot H(A_i | I_i=0) + H(I_i)\big)$$

### 3.2 Corollary 3.7 的特化

假设每个 $A_i(\theta)$ uniform over $\mathcal{M}_i$ with $\ln|\mathcal{M}_i| = b$，且 $A_1, \dots, A_N$ 在 $\theta \sim \Psi$ 下独立。则：

- $H(A_1, \dots, A_N) = \sum_i H(A_i) = Nb$
- $H(A_i | I_i = 0) \leq b$（uniform 上 max entropy）
- $H(I_i) \leq \ln 2$（binary var 的 max entropy）

代入：
$$\text{Mem} \geq Nb - \sum_i \big(\Pr[I_i=0] \cdot b + \ln 2\big) = b \cdot \sum_i \Pr[I_i=1] - N\ln 2 = b \cdot \mathbb{E}[\text{Acc-Cnt}] - N\ln 2$$

结合 $\text{Mem} \leq \ln|\mathcal{W}|$：
$$\boxed{\mathbb{E}[\text{Acc-Cnt}] \leq \frac{\ln|\mathcal{W}| + N\ln 2}{b}}$$

### 3.3 Numerical Example (Phonebook)

每个 fact: 6 alphabet tokens + 22 digit tokens。Fact 是 22-digit phone number。
- 每个 digit: $\ln 10$ nats
- $b = 22 \ln 10 \approx 50.6$ nats $\approx 73$ bits

110M model with 2 bits/param effective capacity:
$$\text{Cap} = \frac{2 \times 110\text{M}}{22 \log_2 10} = \frac{220\text{M}}{73} \approx 3.01\text{M facts}$$

这正是 Figure 2 中画出的 dashed capacity limit line。Table 2 中 LossHF 在 5.12M total facts 时达到 1.94M accurate facts，已经非常接近 3.01M 的 capacity（注意 capacity 是 over-parameter 量的，accuracy 还要打折）。

### 3.4 Intuition: 为什么 "Cram Less Fit More"

当 $N \cdot b \gg \ln|\mathcal{W}|$（i.e. $N \gg \ln|\mathcal{W}|/b$），corollary 3.7 给出的 bound 是 $\ln|\mathcal{W}|/b$——模型最多能准确回答 capacity 个 facts。

但如果在 full dataset 上均匀训练，每个 fact 平均只分到 $\ln|\mathcal{W}|/N$ bits 的 capacity。当 $\ln|\mathcal{W}|/N \ll b$ 时，每个 fact 都只被 partially memorized，stochastic decoding 下 correctness probability $\approx e^{-b} \to 0$，于是 **所有** fact accuracy 都崩到 0。

这就是 Figure 2a 蓝色曲线（Full Dataset）在 facts 数量超过 ~3M 后迅速掉到 0 的根本原因。

---

## 4. Synthetic Phonebook 实验

### 4.1 Setup

Fact format: `<bos><6 alphabet tokens>|<22 digit tokens><eos>`，vocabulary 只有 39 个 tokens。

Power-law distributed: $\Pr[x_j = (Q_i, A_i)] \propto 1/i^\beta$，对 $\beta \in \{0, 0.5, 1.0\}$。$\beta=0$ 即 uniform。

Model: GPT2-style decoder，从 42m 到 1.4B params，bfloat16。训练 800k steps，extensive hyperparameter sweep（batch size, learning rate）。

### 4.2 关键观察 (Figure 2, Figure 5, Figure 6)

**Observation 1**: Uniform distributed facts，当 facts 数量超过 capacity（~3M for 110M model），fact accuracy 急剧下降到 0（Figure 2a 蓝线）。同时 fact memorization（in bits）确实达到 2 bits/param 的 capacity limit（Figure 5 左）——模型确实"记住"了信息，但是 spread out 在所有 facts 上，每个 fact 都不到 $b$ bits。

**Observation 2**: Power-law 分布加剧 suboptimality。$\beta=1.0$ 时即便 facts 数量较少（1.6M），accuracy 也只有 1.18M（capacity 是 3M）。原因是 frequent facts 浪费了 capacity（over-memorized 到远超 $b$ bits），rare facts 完全没被 memorized。

**Observation 3**: Train 8x longer 几乎没用（Figure 6 阴影区域）。10x larger model（1.4B）能完美 memorize 所有 facts。这证明 suboptimality 是 capacity 问题，i.e. optimization 问题，而非 training 不够。

---

## 5. Loss-based Data Selection (Algorithm 1)

### 5.1 核心 insight

Loss 是 fact frequency 和 entropy 的 proxy：
- Low loss → frequent 或 easy fact
- High loss → rare 或 high-entropy fact

为什么不直接用 frequency？因为 real-world 数据中 fact boundaries 和 frequencies 都不知道。Loss 是 model-aware 的，自动 capture 这些。

### 5.2 Algorithm 1 详解

```
Input: selection ratio α, model θ_t, target batch size b
1. Sample fresh batch B
2. Compute τ = lower-percentile_α({ℓ(x; θ_t) : x ∈ B})
   (ℓ 是 per-record sum of token cross-entropy)
3. For each x ∈ B:
   - LossH: keep with prob 1 if ℓ(x;θ_t) ≤ τ, else 0
   - LossHF: keep with prob ℓ(x;θ_t)/τ if ℓ(x;θ_t) ≤ τ, else 0
4. Accumulate until |B_t| = b, then do SGD step
```

**LossH (Head)**: 只保留低 loss 的 α fraction。这等于 "limit number of facts to model capacity"——因为低 loss 的 facts 通常是 frequent ones，少数 frequent facts 占了大部分 capacity budget。

**LossHF (Head-Flattened)**: 在 LossH 基础上，对很低 loss 的样本 down-sample（prob $\propto \ell/\tau$）。直觉：$\ell$ 越低（越 frequent），保留概率越低，于是 flatten 了 fact frequency distribution。

### 5.3 设计细节（Section D.2）

**Fact-level vs token-level**: 必须 fact-level selection。Token-level selection（如 [Rho-1](https://arxiv.org/abs/2404.13936)）会 break fact boundaries，导致 fact accuracy 仍然为 0。这是 paper 的 ablation 重要结论——token-level loss selection 适合 general LM training，不适合 fact memorization。

**Online threshold**: 每个 batch 重新计算 percentile。Reason: Spearman correlation between negative loss 和 fact weight 随训练进行显著提高（Figure 10），尤其当 data 超过 capacity 时。早期训练时 loss 主要反映 format 难度，后期才反映 fact frequency。

**Sum vs average of token loss**: 用 sum 而非 average。Sum 更好地对应 fact 的 total entropy。Figure 11 显示 sum-of-loss 和 weight-to-bits ratio 的 Spearman correlation 远高于 average-of-loss。这与 [Allen-Zhu & Li 2024](https://arxiv.org/abs/2404.05405) 的 loss-based memorization lower bound 一致。

### 5.4 为什么 LossHF 比 LossH 更好

Table 2 显示在 power-law 分布下，LossHF（最后一列）几乎总是最好的：
- $\beta=0.5$, 5.12M facts: Full=0.93M, LossH=1.46M, LossHF=2.16M
- $\beta=1.0$, 5.12M facts: Full=0.86M, LossH=1.00M, LossHF=1.72M

Head selection 限制了 facts 数量，但若不 flatten，frequent facts 仍会 over-replicated，浪费 capacity。Flattening 把每个 selected fact 的 expected frequency 拉到接近 uniform。

---

## 6. Wikipedia 实验

### 6.1 Annotated Wikipedia Corpus

来自 [Zhao et al. 2025b](https://arxiv.org/abs/2505.15962)，3B tokens，facts 用 `<|start_of_fact|>...<|end_of_fact|>` 标注。总共 59.7M facts / 6.25M records（平均每 record ~10 facts）。

Example:
```
Pterostylis stricta was first described in <|start_of_fact|>1972<|end_of_fact|> by <|start_of_fact|>Stephen Clemesha and Bruce Gray<|end_of_fact|>
```

### 6.2 Algorithm 2 (LossH-Wiki / LossHF-Wiki)

Wikipedia 中 facts 和 non-facts 混在同一 record 内，不能丢掉整个 record。Algorithm 2 用 mask：

1. 保留所有 non-fact tokens（mask=1）
2. 对每个 fact $(Q, A)$，计算 per-fact loss $\ell(A; \theta_t, Q)$
3. 计算 $\tau$ = percentile_α over all fact losses in batch
4. LossH-Wiki: 保留 fact A 的 tokens if $\ell \leq \tau$，prob 1
5. LossHF-Wiki: 保留 fact A 的 tokens with prob $\ell/\tau$ if $\ell \leq \tau$
6. Upscale selected fact tokens 的 mask weight，保持 fact/non-fact token weight ratio 不变

这第 6 步很关键——避免 distorting 学习信号。

### 6.3 实验结果 (Figure 4, Table 1)

**Fact Accuracy (Test split, 7135 facts)**:
- 110M Full (α=1.0): 696 facts
- 110M LossH-Wiki (α=0.2): **929 facts** (+33%)
- 335M Full: ~870 facts (从 Figure 4 dashed line 读出)
- 1.3B Full: ~930 facts

110M with α=0.2 **matches 1.3B Full** (10x larger model) on fact accuracy!

**MMLU-Knowledge**: LossH-Wiki (α=0.5) matches 335M Full (3x larger).

**NLU** (CommonsenseQA, HellaSwag, PIQA, SIQA, ARC-Easy): α=0.2 到 α=0.9 之间基本持平，只有 α=0.1 下降。说明 data pruning 不伤 general capability。

Table 1 显示 random fact-level pruning 完全没用——α=0.2 时 Random=647 facts vs LossH-Wiki=929 facts。这证明 loss-based selection 是 essential 的，不只是 subsetting。

---

## 7. LoRA Finetuning on arXiv Papers

### 7.1 Setup

171,104 arXiv papers from 2025（after Llama-3.2-1B 的 cutoff）。Fact format: "title: ___ | authors: ___"，teaching 模型 title-to-authors mapping。

这模拟"教 pretrained model 新知识"——Llama 3.2 cutoff 在 2024，所以 2025 paper 的 title-author mapping 是 unseen 的。

### 7.2 Results (Figure 8, Table 4)

| LoRA rank | Full Data Fact Acc | LossHF Fact Acc |
|-----------|---|---|
| r=4  | 0.2% | 7.9%  |
| r=8  | 0.2% | 18.1% |
| r=16 | 0.6% | 34.6% |
| r=32 | 3.6% | **55.5%** |

General capabilities（Hellaswag, MMLU 等）保持基本不变，证明 LossHF 不加剧 catastrophic forgetting。这与 [Sanyal et al. 2025](https://arxiv.org/abs/2502.02797) 的"upweight easy samples reduces forgetting"结论一致。

注意 Full Data fact accuracy 几乎为 0——说明 LoRA capacity 远小于 171k facts，符合 Corollary 3.7 的预测。

---

## 8. Ablation: Oracle-Aided Selection (Section D.1)

Paper 比较 LossHF 与三个 oracle-aided baselines（假设知道 ground-truth fact frequency）：
- **Head**: 选 top $\ln|\mathcal{W}|/b$ 频率最高 facts
- **Head-Flattened**: Head + down-sample frequent facts 到 uniform
- **Flattened**: 只 flatten 不 head select

Table 2 关键发现：
1. **Head-Flattened oracle 总是最好**（在 capacity 内：facts ≤ 2.56M 时）
2. **LossHF 与 Head-Flattened oracle 性能接近**（除了 $\beta=1.0$ 且 facts 极多时），证明 loss 是 excellent proxy
3. **Flattened vs Head**:
   - facts < capacity: Flattened 更好（capacity 够，只需 flatten）
   - facts > capacity: Head 更好（必须丢掉 tail facts）
4. 这证明 **head selection 和 flattening 都是必要的**

Figure 3 visualize 了 selected data 的 frequency histogram——LossHF 几乎完美匹配 Head-Flattened oracle 的分布。

---

## 9. 与 Rho-1 / Rho-Loss 的对比

Paper Section 2 明确指出：**LossH 选低 loss 样本，与 [Rho-1](https://arxiv.org/abs/2404.13936) 和 [Rho-Loss](https://arxiv.org/abs/2207.07018) 选高 excess loss 样本相反**。

这说明 Rho-1 的 benefits **不来自更好的 fact memorization**——Rho-1 选的是模型最不确定的样本，对于 fact memorization 反而会 spread capacity 到太多 facts。Rho-1 的实际好处应该来自其他机制（可能是 reasoning / generalization）。

paper 留下 open question：什么 data selection 能同时 optimize fact accuracy 和 generalization？这两个目标在 selection 上是 opposite direction 的。

---

## 10. Limitations & Open Questions

1. **Fact boundaries 需要标注**：Algorithm 2 依赖 `<|start_of_fact|>...<|end_of_fact|>` annotation。Paper 提到可以用 lightweight LM 自动标注，参考 [Maini et al. 2024](https://aclanthology.org/2024.acl-long.755/) 和 [Su et al. 2025](https://aclanthology.org/2025.acl-long.235/)。
2. **Catastrophic forgetting**：LoRA 实验中 forgetting 仍然严重，需要结合 data replay，参考 [Buzzega et al. 2020](https://arxiv.org/abs/2004.07111)。
3. **MoE / specialized memory**：paper 只考虑 dense transformer。结合 [Jelassi et al. 2024](https://arxiv.org/abs/2410.19034) 的 MoE 或 [Cheng et al. 2026](https://arxiv.org/abs/2601.07372) 的 lookup memory 可能进一步提升。
4. **Bounded FLOPs regime**：所有实验都是 sufficient training setting。在 fixed FLOPs 下，selection 的 overhead（forward pass 数增加）vs full data training 的 trade-off 不清楚。

---

## 11. Intuition Summary

整篇 paper 的核心 mental model：

**Model 是一个 fixed-size container（2 bits/param）。Facts 是固定大小的 stones（每个 ~73 bits for 22-digit phonebook）。**

- 如果 stones 数量 << container capacity：随便装，都能装下
- 如果 stones 数量 >> container capacity：均匀撒进去，每颗 stone 都被压碎到一些 dust，没有完整 stone 幸存（accuracy = 0）
- 如果 stones 数量 >> capacity 且 stones 大小不一：big stones（frequent facts）占据大量空间，挤掉 small stones（rare facts）
- **Solution**: 只装 capacity 数量的 stones，且把它们压平到一样大（uniform frequency），每颗 stone 都能完整保留

Loss 是 model-driven proxy for stone size + frequency。Low loss = big/frequent stone。Head selection = 丢掉多余的 stones。Flattening = 把 big stones 切小，让所有 stones 大小一致。

这个 model 简洁但 powerful——它直接解释了为什么 110M model 用 20% 的 Wikipedia data 能 match 1.3B model 用 100% data。

---

## 12. Practical Implications

如果你在 pretrain 一个 small model（<1B）且关心 knowledge-intensive tasks：
1. **不要 train on full web data**。先用 reference model（或当前 checkpoint）计算 per-fact loss
2. 选择低 loss 的 facts（LossH），或低 loss + flatten（LossHF）
3. Selection ratio $\alpha$ 大致满足 $\alpha \cdot N_{\text{facts}} \approx 2 \cdot N_{\text{params}} / b$，其中 $b$ 是平均 fact 信息量
4. 对 finetuning 新知识，LossHF 比 full data finetuning 能多记住 ~30x 的新 facts（Table 4, r=32: 3.6% vs 55.5%）

Scaling law（Section D.3, equation D.1）：
$$\alpha(\mathcal{W}) \propto \sum_{i=1}^{\ln|\mathcal{W}|/b} \frac{1}{i^\beta} \propto \begin{cases} \ln\ln|\mathcal{W}| & \beta = 1 \\ \text{const} \cdot |\mathcal{W}|^{1-\beta} & 0 < \beta < 1 \\ |\mathcal{W}| & \beta = 0 \end{cases}$$

这给出 optimal $\alpha$ 如何随 model size 变化。Uniform 分布（$\beta=0$）时 $\alpha$ 与 model size 成正比；power-law 时 $\alpha$ 增长更慢。

---

## 13. 我的 take

这篇 paper 的核心 insight 其实很朴素：**模型是个容量有限的 container，超过容量硬塞会全崩**。但它的贡献在于：

1. **形式化了 fact accuracy 的 capacity limit**（Corollary 3.7），把 2 bits/param 的 memorization capacity 翻译成 accurate fact count
2. **实验上证明 standard training 远未达到 capacity**（Figure 2 蓝线）
3. **简单 loss-based algorithm 几乎达到 oracle-aided 最优**（Table 2 LossHF vs Head-Flattened）
4. **在真实 Wikipedia 上验证**：110M + selection match 1.3B full data

与 [Chinchilla](https://arxiv.org/abs/2203.15556) 的 compute-optimal scaling 不同，这篇关注的是 **data-capacity matching**——给定 fixed model size，应该选多少 data。这补充了 scaling laws 的另一个 axis。

一个有意思的思考：如果 model capacity 是 2 bits/param，那 GPT-4 的 ~1T params 理论上能存 ~250 GB 纯 factual information。但 SimpleQA 上 GPT-4 也只有 ~50% accuracy，说明 **real-world training data 的 power-law skew 远超 synthetic experiments**，standard training 离 capacity limit 还很远。这篇 paper 给出了第一个 systematic 的 method 来 close 这个 gap。

相关链接：
- Paper PDF: 假设发布在 arXiv，标题搜索 "Cram Less to Fit More"
- [Allen-Zhu & Li "Physics of LM" Part 3.3](https://arxiv.org/abs/2404.05405)
- [Morris et al. 2025 on memorization capacity](https://arxiv.org/abs/2505.24832)
- [Jelassi et al. 2024 "Mixture of Parrots"](https://arxiv.org/abs/2410.19034)
- [Rho-1 paper](https://arxiv.org/abs/2404.13936)
- [Feldman 2020 "Does learning require memorization?"](https://arxiv.org/abs/2003.02933)
- [Zhao et al. 2025b on memory LM pretraining](https://arxiv.org/abs/2505.15962)
- [SimpleQA](https://arxiv.org/abs/2411.04368)
- [Kandpal et al. 2023 on long-tail knowledge](https://arxiv.org/abs/2211.08411)
- [Chinchilla scaling laws](https://arxiv.org/abs/2203.15556)
- [Sanyal et al. 2025 on upweighting easy samples](https://arxiv.org/abs/2502.02797)
- [Fano's inequality (Wikipedia)](https://en.wikipedia.org/wiki/Fano%27s_inequality)
- [Pythia suite](https://arxiv.org/abs/2304.01373)
- [OLMo](https://arxiv.org/abs/2402.00838)
- [Llama 3](https://arxiv.org/abs/2407.21783)
