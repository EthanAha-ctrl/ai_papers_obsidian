---
source_pdf: Reinforcement Pre-Training.pdf
paper_sha256: f76e63febe2bf17c3eff442ec8608a62484777b3376603daca7dd2560d3085a9
processed_at: '2026-08-11T22:23:33-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# RPT 用人话说

## 一句话版本

平时训练 LLM 就是让它看一堆文本猜下一个词，猜错就调参数——这是 pattern matching。RPT 的想法是：**让模型在猜之前先"想一想"**，想对了给奖励，想错了没奖励。这样就把整个 pre-training 变成了一个巨大的 RL game，而 reward 不需要人标注，因为文本本身就有 ground truth。

---

## 为什么这个 idea 有意思

### 先说现有 paradigm 的问题

训练 LLM 有两条 main pipeline：

1. **Pre-training (NTP)**：海量文本，self-supervised，每个 token 一次 forward pass 直接输出概率分布，用 cross-entropy 更新。Cheap、scalable，但本质是 pattern matching——模型学的是 surface-level correlation，不知道"为什么"这个 token 该跟在后面。

2. **Post-training RL (RLHF / RLVR)**：给模型一个 question，让它 generate response，用 reward model 或 verifiable rule 打分，用 PPO/GRPO 更新。Powerful，但 expensive——需要 human preference data 或 annotated QA pairs，数据稀缺，局限于特定 domain。

这两条路一直没打通：pre-training cheap 但"浅"，post-training powerful 但"窄"。

### RPT 干了什么

RPT 的 insight 极其简单：**pre-training 的 next-token prediction 本身就是一个 reasoning problem**。

给定 context $x_{<t}$，ground truth next token 是 $x_t$。标准 NTP 让模型直接输出 $P(x_t | x_{<t})$。RPT 让模型先 generate 一个 chain-of-thought $c_t$，然后给 prediction $y_t$，整体 response 是 $o_t = (c_t, y_t)$。

Reward 就是：$y_t$ 对不对？对就给 1，错就给 0。

这个 reward 是 **verifiable** 的——直接拿 ground truth token 比一下就行，不需要任何外部标注。于是整个 web text corpus 瞬间变成了一个巨大的 RL 数据集：每个 token 位置都是一个 independent RL episode，context 是 state，next token 是 target，correctness 是 reward。

---

## 关键技术细节，用人话翻译

### 1. Prefix Matching Reward（公式 3）

最 naive 的 reward 是 first-token matching：模型预测的第一个 token 对了就 reward 1。但这有两个问题：
- 模型可能预测多个 token（比如一个完整数学表达式），你想支持 multi-token prediction
- byte-level 和 token-level 对齐问题（OOV token）

RPT 用 **byte-level prefix matching**：

$$r_t^i = \begin{cases} 1 & \text{if } \overline{y}_t^i = \overline{x}_{\geq t}[1:l] \text{ and } l \in \mathcal{L}_{gt} \\ 0 & \text{otherwise} \end{cases}$$

变量解释：
- $\overline{y}_t^i$：第 $i$ 个 rollout 的 prediction $y_t^i$ 转成 byte sequence
- $\overline{x}_{\geq t}$：ground truth continuation 转成 byte sequence
- $l$：prediction 的 byte length
- $\mathcal{L}_{gt}$：ground truth 中所有 token boundary 对应的 cumulative byte offset 集合

两个条件：
1. prediction 的 byte sequence 必须是 ground truth 的精确 prefix（内容对）
2. prediction 的 byte length 必须恰好落在某个 token boundary 上（不能是半个 token）

第二个条件防止 model 学 byte-level hacking——比如 ground truth 是 "parameter"，模型预测 "parame"（5.5 个 token 的 byte length），内容 prefix 匹配但不是完整 token，不给 reward。

### 2. GRPO 训练（公式 4）

RPT 的 objective：

$$\mathcal{I}_{\text{RPT}}(\theta) = \mathbb{E}_{(x_{<t}, x_{\geq t}) \sim \mathcal{D}, \{o_t^i\} \sim \pi_\theta(\cdot | x_{<t})} \left[ r_t^i \right]$$

对每个 context $x_{<t}$，模型 on-policy 采样 $G=8$ 个 rollouts $\{o_t^i\}_{i=1}^G$，每个 rollout 独立计算 reward $r_t^i$，用 GRPO 更新。

GRPO 的 trick 是：不训练 value model（PPO 需要），而是用组内 reward 均值作为 baseline，计算 group-relative advantage。这样省了 value model 的开销，适合大规模 RL。

具体来说，对 group $\{o_t^1, \ldots, o_t^G\}$：
- 计算 group mean reward $\bar{r}_t = \frac{1}{G}\sum_{i=1}^G r_t^i$
- 每个 rollout 的 advantage $A_t^i = r_t^i - \bar{r}_t$
- 用 $A_t^i$ 作为 policy gradient 的权重更新 $\theta$

如果 8 个 rollout 全对（$\bar{r}_t = 1$）或全错（$\bar{r}_t = 0$），所有 $A_t^i = 0$，没有梯度信号。这就是为什么 paper 从 step 500 开始用 dynamic sampling 过滤这些 degenerate groups。

### 3. Entropy-based Data Filtering

这是个很 pragmatic 的工程细节。Pre-training corpus 里大部分 token 是高度可预测的——"the"、"a"、"is"——这些 token 让模型做 chain-of-thought 是浪费 compute。

RPT 用一个小 proxy model（Deepseek-R1-Distill-Qwen-1.5B）计算每个 token 位置的 entropy：

$$H(x_t | x_{<t}) = -\sum_{k=1}^{16} P(x_t^{(k)} | x_{<t}) \log P(x_t^{(k)} | x_{<t})$$

其中 $x_t^{(k)}$ 是 top-16 candidate next tokens。High entropy = model uncertain = 需要 reasoning。Low entropy = trivial token = 跳过。

验证集按 entropy threshold 分三档：
- Easy: entropy > 0.5
- Medium: entropy > 1.0  
- Hard: entropy > 1.5

这样 RPT 的 compute 集中在最有价值的 token 上，类似于 importance sampling。

---

## 实验结果，挑重点说

### Table 1: Next-token prediction accuracy

| Model | Easy | Medium | Hard |
|---|---|---|---|
| Qwen2.5-14B (NTP) | 41.90 | 30.03 | 20.65 |
| R1-Distill-14B (NTP) | 41.60 | 29.46 | 20.43 |
| R1-Distill-14B (reasoning) | 3.31 | 1.66 | 1.41 |
| **RPT-14B** | **45.11** | **33.56** | **23.75** |

几个 takeaway：

1. **R1-Distill 直接做 next-token reasoning 效果极差**（3.31% on easy）。这很 intuitive——R1-Distill 是为 problem-solving 训练的，让它对每个 token 都生成完整 chain-of-thought 是 misuse，它没学过这个。

2. **RPT-14B 全面超越**：每个 difficulty level 都有 +3 到 +4 个点提升。Figure 4 还显示 RPT-14B 匹配了 2 倍参数量的 R1-Distill-32B——这是很强的 scaling 信号。

3. **Hard token 相对提升更大**：20.43 → 23.75 是 +16.3% 相对提升，easy 41.60 → 45.11 是 +8.4%。说明 RPT 的收益集中在 model 最不确定、最需要 reasoning 的 token 上，这正是你期望的。

### Table 2: Reinforcement Fine-Tuning

| Model | Before RL | After RL |
|---|---|---|
| R1-Distill-14B | 51.2 | 52.7 |
| + Continual NTP training | 10.7 | 13.0 |
| **RPT-14B** | **56.3** | **58.3** |

这个表有个很 striking 的结果：**把 reasoning model 用 NTP objective 继续训练，reasoning 能力从 51.2 暴跌到 10.7**。这是 catastrophic forgetting——NTP objective 逼迫模型放弃 chain-of-thought，回到直接 token prediction，破坏了已学的 reasoning pattern。后续 RLVR 也救不回来（只到 13.0）。

这说明 **objective consistency 很重要**。RPT 本身就是 RL，fine-tuning 只是延续同一个 paradigm，gap 很小。而 R1-Distill 是 distillation 产物，与 RLVR 有 objective gap。

### Table 3: Zero-shot end tasks

| Model | SuperGPQA | MMLU-Pro |
|---|---|---|
| R1-Distill-14B (NTP) | 32.0 | 48.4 |
| R1-Distill-32B (NTP) | 37.2 | 56.5 |
| R1-Distill-14B (reasoning) | 36.1 | 68.9 |
| **RPT-14B (reasoning)** | **39.0** | **71.1** |

RPT-14B 超越了 2 倍大的 R1-Distill-32B。考虑到 RPT 只在 math corpus 上训练，能在 general knowledge benchmark (MMLU-Pro) 上有提升，说明 next-token reasoning 学到的是某种 general reasoning disposition，不局限于 math。

### Figure 5: Scaling curves

RPT 的 accuracy 随 compute 的 scaling 用 power-law 拟合：

$$P(C) = \frac{A}{C^\alpha} + P^*$$

变量：
- $P(C)$: compute 为 $C$ 时的 accuracy
- $P^*$: asymptotic accuracy（compute → ∞ 时的上限）
- $\alpha$: scaling exponent
- $A$: 常数

所有 difficulty level 上 $R^2$ 都很高，说明 RPT 的 scaling 是 predictable 的——你可以预测投入更多 compute 会带来多少 accuracy 提升。这对 resource planning 很有价值。

---

## Reasoning Pattern 的本质区别（Section 4.5）

这部分很有意思。Paper 统计了 reasoning 中的 keyword pattern：

| Pattern | Keywords |
|---|---|
| Transition | alternatively, think differently |
| Reflection | wait, looking back, thought process |
| Breakdown | break down |
| Hypothesis | probably, something like |
| Divergent | etc., or something, exploring |
| Deduction | summarize, conclusion, consequently |

结果：
- RPT-14B 的 **hypothesis pattern 比 R1-Distill 多 161.8%**
- **deduction pattern 多 26.2%**
- R1-Distill 更多用 **breakdown pattern**

这个差异的 intuition 是：problem-solving 面对结构化问题，有明确解题步骤，所以 breakdown（分解问题）是主要策略。而 next-token reasoning 面对的是 **不确定性下的推断**——模型不知道哪个 token 对，必须先提出多个 hypothesis（"可能是 para，可能是 =，可能是空格"），再通过 deduction 验证。这更像 Bayesian inference，而非 deterministic problem-solving。

Table 4 的 case study 很好地展示了这点：模型分析语义上下文（"calculating vector magnitude"），识别关键短语（"go over some..."），brainstorm 多个可能延续，考虑 markdown 格式、token-level 细节（空格），最终选最 probable 的。这是 high-level semantic understanding 和 low-level textual feature 的结合。

---

## 为什么 RPT 有效？三个层面的 intuition

### 1. Compute Allocation

标准 NTP 对每个 token 分配相同 compute（一次 forward）。但不同 token 预测难度差异巨大——"the" 几乎不需要思考，数学证明关键步骤可能需要大量推理。RPT 让 model 对每个 token 动态分配 compute，通过 chain-of-thought 长度隐式实现 **adaptive computation**。

这类似 MoE 在空间维度上的 sparse activation，RPT 是时间维度上的 sparse computation。

### 2. Exploration

NTP 是 teacher forcing，model 只见 ground truth continuation，无 exploration。RPT 让 model 采样 G=8 个不同 reasoning trajectory，探索多种推理路径。这种 exploration 让 model 学到的不只是"什么 token 跟在后面"，而是"为什么这个 token 应该跟在后面"——即推理过程本身的 robustness。

### 3. Credit Assignment

NTP 中如果 model 预测错了某 token，梯度信号只告诉它"这个 token 概率应该更高"，但不告诉它"为什么"。RPT 中通过 chain-of-thought，推理过程和最终预测的因果关系变得 explicit。GRPO 的 group-relative advantage 让 model 知道"这条 reasoning path 导致正确预测，那条导致错误"，从而 reinforce 正确 reasoning pattern。

---

## 几个关键 Limitation

1. **Training cost 巨大**：RPT 每个 token 需要 G=8 个 rollouts，每个 rollout 包含 chain-of-thought（可能几百 tokens），加上 entropy filtering 只用一小部分 tokens。有效 compute 远超 NTP。Paper 说 RPT-14B 匹配 32B，但如果算总 compute（包括所有 rollouts），可能不比训练 32B 便宜。这个 cost-effectiveness 比较 paper 没有详细讨论。

2. **Inference cost 也高**：用 RPT model 做 language modeling 也需要生成 chain-of-thought。Paper 说 RPT model 也可以用 NTP mode（直接预测），但 Table 1 没报告 RPT-14B 在 NTP mode 下的表现——这是个重要缺失。

3. **Math-specific bias**：OmniMATH 是高度结构化 domain，next-token 的 verifiable reward 语义清晰。General web text 中 next token 往往 creative、多义、甚至任意选择（同义词替换）。在这些 token 上 verifiable reward 可能 sparse 且 noisy。Paper 承认这点是 future work。

4. **从 reasoning model 开始**：RPT 从 R1-Distill 开始，继承了已有 reasoning 能力。如果从 base model (Qwen2.5-14B) 开始，model 需要先学会"如何思考"才能做 next-token reasoning，这个 cold-start 问题是 RPT 能否成为真正 pre-training 范式的关键。

5. **Reward sparsity**：Binary reward (0/1) 在 hard tokens 上可能极 sparse——如果 8 个 rollouts 全错，GRPO advantage 全为零，没有梯度。Appendix A 的 dense reward 是缓解方案，但 paper 说效果相当，没深入分析何时 dense reward 更优。

---

## 与相关工作的关系网络

### Quiet-STaR (Zelikman et al. 2024)
最直接前置工作。也让 model 在 token prediction 前生成 rationale。区别在 reward：Quiet-STaR 用 helpfulness-based reward（rationale 是否帮助预测），容易被 hack（model 重复 target token 在 rationale 中）。RPT 用 correctness-based rule reward 避免 hacking。[论文](https://arxiv.org/abs/2403.09629)

### DeepSeek-R1 (Guo et al. 2025)
RPT 用的 GRPO 算法和 RLVR framework 直接来自 R1。R1 证明 RLVR 在 math reasoning 上的威力，RPT 把 RLVR 从 fine-tuning 扩展到 pre-training。[论文](https://arxiv.org/abs/2501.12948)

### Scaling Laws (Kaplan 2020; Hoffmann 2022)
RPT 的 scaling law 形式继承 NTP 的 power-law tradition，但针对 accuracy 而非 loss。这转变反映了 RPT 关注 prediction accuracy 而非 likelihood。[Kaplan](https://arxiv.org/abs/2001.08361), [Chinchilla](https://arxiv.org/abs/2203.15556)

### LeCun's Cherry-on-the-Cake
Figure 1 引用 LeCun [NIPS 2016 keynote](https://www.youtube.com/watch?v=I09mVd5_pog) 的比喻：cake 主体是 unsupervised/predictive learning，icing 是 supervised，cherry 是 RL。LeCun 认为 RL 占 compute 量应该很小。RPT 在某种意义上 **颠覆了这个比喻**——把 RL 变成 cake 主体，让 predictive learning 通过 RL 实现。

---

## RPT 的深层意义

RPT 的贡献超越一个新 training technique。它重新定义了 pre-training 的本质：

**传统视角**：pre-training 是学习语言的 statistical regularity，通过 maximum likelihood 拟合数据分布。

**RPT 视角**：pre-training 是学习如何 reasoning about 世界的下一个状态，通过 RL 在 verifiable reward 上优化。

这个视角转变有几个深远含义：

1. **Pre-training 不再 "cheap"**：RPT 的 compute cost 远超 NTP，但可能值得——因为 model 学到的是 reasoning ability 而非 surface pattern。

2. **RL 与 self-supervised learning 的统一**：RPT 暗示 RL 可能是比 MLE 更 fundamental 的 learning paradigm。任何 prediction task 都可转化为 RL（prediction correctness 作为 reward）。

3. **Intelligence as prediction**：RPT 呼应 predictive coding 和 free energy principle——大脑核心功能是预测下一个感官输入，通过最小化 prediction error 学习。RPT 把这个思想在 LLM 上实现，用 RL 替代 MLE 优化 prediction。

4. **Towards AGI**：如果 RPT 能 scale 到 general web text 且从 base model 开始 work，它可能成为真正 general-purpose pre-training paradigm，让 model 在 pre-training 阶段就获得 strong reasoning ability。

---

## 开放问题与未来方向

1. **General text RPT**：如何定义 general text 上的 verifiable reward？可能方向：(a) 多个 candidate tokens 都算正确（top-k reward）；(b) 用更强 model 作为 verifier；(c) 结合 semantic similarity 而非 exact match。

2. **From base model**：从 Qwen2.5-14B（无 reasoning 能力）开始 RPT，需解决 cold-start。可能方案：(a) 先用少量 reasoning data SFT；(b) curriculum learning，从简单 reasoning 开始；(c) hybrid objective，初期 NTP + RPT 逐渐过渡纯 RPT。

3. **Hybrid thinking**：Paper 提到与 [Hybrid Reasoning Models](https://arxiv.org/abs/2506.09992) 结合，让 model adaptively 决定何时触发 next-token reasoning。简单 token 直接预测，困难 token 才 reasoning。这可大幅降低 inference cost。

4. **Multi-modal extension**：RPT 思想可扩展到 vision-language models——预测下一个 image patch 或 visual token，用 reconstruction error 或 perceptual similarity 作为 reward。

5. **RPT + World Model**：RPT 本质是训练 model 预测世界下一个状态（用 token 表示）。这与 LeCun 的 JEPA 思想有呼应。RPT 在 token space 预测，JEPA 在 latent space 预测，两者可能互补。[LeCun on JEPA](https://openreview.net/pdf?id=BZ5a1r-kVsf)

6. **Reward design for creativity**：在 creative writing 等任务上，next token 没有唯一正确答案。如何设计 reward 鼓励 model 探索多种合理延续？可能需要 distributional reward 或 multi-reference reward。

---

## 最终总结

RPT 的核心 insight 用一句话概括：**把 test-time compute scaling 的思想搬到 training-time，让每个 next-token prediction 都成为一个 RL episode，用 prediction correctness 作为 verifiable reward**。

这个 idea 的 elegance 在于它打通了 pre-training 和 post-training 两个世界：pre-training 不再是 cheap but shallow 的 pattern matching，post-training 的 RL 不再是 expensive but narrow 的 fine-tuning。RPT 让两者统一在同一个 RL framework 下，用 web text 的 intrinsic structure 作为 reward signal。

当然，还有很多 open question：general text 的 reward 怎么定义？从 base model 怎么 cold-start？inference cost 怎么降低？但这些是 engineering problem，不是 conceptual blocker。RPT 提供了一个新的 paradigm，剩下的只是 how to scale it 的问题。

**核心参考资源**：
- [RPT 项目页](https://aka.ms/GeneralAI)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [Quiet-STaR](https://arxiv.org/abs/2403.09629)
- [GRPO / DAPO](https://arxiv.org/abs/2503.14476)
- [Scaling Laws (Kaplan)](https://arxiv.org/abs/2001.08361)
- [Scaling Laws (Chinchilla)](https://arxiv.org/abs/2203.15556)
- [LeCun NIPS 2016 Keynote](https://www.youtube.com/watch?v=I09mVd5_pog)
- [verl RL framework](https://arxiv.org/abs/2409.19256)
- [OmniMATH](https://arxiv.org/abs/2410.07985)
- [SuperGPQA](https://arxiv.org/abs/2502.14739)
- [MMLU-Pro](https://arxiv.org/abs/2406.01574)
- [Hybrid Reasoning Models](https://arxiv.org/abs/2506.09992)
- [On-policy RL with Optimal Reward Baseline](https://arxiv.org/abs/2506.06999)
- [Let's Verify Step by Step (PRM)](https://arxiv.org/abs/2305.20050)

---

# Reinforcement Pre-Training (RPT): 把 Next-Token Prediction 重新定义为 Reasoning Task

## 1. 核心直觉：一个范式的重新定义

RPT 这篇工作的核心思想极其简洁，但其含义深远。传统 LLM pre-training 的 objective 是 maximum likelihood——给定 context $x_{<t}$，模型一次 forward pass 直接输出下一个 token $x_t$ 的概率分布，通过 cross-entropy 更新参数。这是 self-supervised，可规模化，但本质上是 **pattern matching**：模型学习的是 surface-level 的 token-level correlation，分配给每个 token 的 compute 是均匀的、固定的（一次 forward）。

RPT 的 insight 是：**每个 next-token prediction 本身都可以被视为一个 reasoning problem**。给定 context $x_{<t}$，模型不应该直接吐出 $x_t$，而应该先在 latent space / chain-of-thought 中"思考"一下——brainstorm 可能的延续、验证假设、自我批判——然后再给出预测。如果预测正确（匹配 ground-truth token），给 reward 1；否则 reward 0。这个 verifiable reward 直接来自 corpus 本身，不需要任何外部标注。

这是一个范式转变：把 **test-time compute scaling** 的思想直接移植到 **training-time**。每个 token 都变成一个独立的 RL episode，模型在生成 token 之前被允许"多想一会儿"。

### 1.1 与现有范式的对比

- **Standard NTP**: teacher forcing，模型只见 ground-truth continuation，无 exploration，每个 token 计算量固定
- **RLHF**: 需要 human preference data，reward model 是 learned 的，容易被 hack（模型学到 reward model 的 bug 而非真正对齐）
- **RLVR (RL with Verifiable Rewards)**: 需要 annotated QA pairs，数据稀缺，局限于特定 domain（如 math、code）
- **RPT**: 用 web text 本身的 next token 作为 verifiable reward，把整个 pre-training corpus 转化为一个大规模 RL 数据集

关键点在于 RPT 解决了 RLVR 的 **scalability 瓶颈**：RLVR 依赖 human-annotated QA pairs，而 RPT 把任何文本都变成 QA pair——context 是 question，next token 是 answer，verifier 是 byte-level prefix matching。

参考：
- [DeepSeek-R1 论文](https://arxiv.org/abs/2501.12948) - RLVR 的代表作
- [Quiet-STaR](https://arxiv.org/abs/2403.09629) - RPT 最相关的前置工作，鼓励 LM 在预测前生成 rationale

---

## 2. 方法详解：从 NTP 到 Next-Token Reasoning

### 2.1 任务定义

给定 pre-training corpus 的一个 sequence $x_0 \cdots x_T$，对每个位置 $t \in \{1, \ldots, T\}$：
- Context: $x_{<t}$ (prefix)
- Target: $x_t$ (ground-truth next token)

模型 $\pi_\theta$ 被要求生成一个 response $o_t = (c_t, y_t)$，其中：
- $c_t$: chain-of-thought reasoning sequence（中间的思考过程）
- $y_t$: 最终对 next token 的预测

整体采样过程：$o_t \sim \pi_\theta(\cdot | x_{<t})$

这与标准 NTP 的区别可以用 Figure 2 直观理解：NTP 是 $x_{<t} \rightarrow x_t$ 的直接映射；RPT 是 $x_{<t} \rightarrow c_t \rightarrow y_t$ 的两阶段过程，中间 $c_t$ 可以是任意长度的 reasoning trajectory。

### 2.2 Prefix Matching Reward

这是 RPT 的一个关键技术细节。简单的 first-token matching reward 有两个问题：
1. **Multi-token prediction**: 模型可能预测多个 token（例如预测一个完整的数学表达式），需要支持 multi-token verification
2. **OOV tokens**: byte-level 和 token-level 的对齐问题

RPT 引入 byte-level prefix matching。定义：
- $\overline{x}_{\geq t}$: ground-truth completion $x_{\geq t}$ 的 byte sequence
- $\overline{y}_t^i$: 第 $i$ 个 rollout 的 prediction $y_t^i$ 的 byte sequence
- $l$: $\overline{y}_t^i$ 的 byte length
- $\mathcal{L}_{gt}$: ground-truth 中所有 token 边界对应的 cumulative byte lengths 集合（即每个 token 结束时的 byte offset）

Reward 函数：

$$r_t^i = \begin{cases} 1 & \text{if } \overline{y}_t^i = \overline{x}_{\geq t}[1:l] \text{ and } l \in \mathcal{L}_{gt} \\ 0 & \text{otherwise} \end{cases}$$

这个公式有两个条件：
1. **内容匹配**: prediction 的 byte sequence 必须是 ground-truth byte sequence 的精确 prefix
2. **边界对齐**: prediction 的 byte length $l$ 必须恰好落在某个 token 边界上（即 $l \in \mathcal{L}_{gt}$）

为什么要第二个条件？因为如果模型预测了半个 token（byte-level 不对齐到 token boundary），即使内容匹配也不算正确——这样确保模型学到的是 token-level 的预测，而不是 byte-level 的 hacking。

### 2.3 RL 训练 Objective

RPT 的总体 objective：

$$\mathcal{I}_{\text{RPT}}(\theta) = \mathbb{E}_{(x_{<t}, x_{\geq t}) \sim \mathcal{D}, \{o_t^i\} \sim \pi_\theta(\cdot | x_{<t})} \left[ r_t^i \right]$$

其中：
- $\mathcal{D}$: 所有 $\{x_{<t}\}_{t=1}^T$ 构成的数据集
- 对每个 context $x_{<t}$，模型 on-policy 采样 $G$ 个 rollouts $\{o_t^i\}_{i=1}^G$
- 每个 rollout 独立计算 reward $r_t^i$
- 通过 GRPO 算法更新参数

这里用的是 GRPO（Group Relative Policy Optimization），是 DeepSeek-R1 用的算法。GRPO 的核心是：对同一个 prompt 的 $G$ 个 rollout，用组内 advantage（reward 减去组内均值）作为 baseline，避免训练一个 value model。这降低了计算开销，适合大规模 RL。

### 2.4 Entropy-based Data Filtering

这是一个很重要的工程细节。pre-training corpus 中大部分 token 是高度可预测的（如 "the", "a", "is"），对这些 token 做 reasoning 是浪费 compute。RPT 用一个 proxy model (Deepseek-R1-Distill-Qwen-1.5B) 计算每个 token 位置的 entropy：

$$H(x_t | x_{<t}) = -\sum_{k=1}^{16} P(x_t^{(k)} | x_{<t}) \log P(x_t^{(k)} | x_{<t})$$

其中 $x_t^{(k)}$ 是 top-16 next tokens。通过 entropy threshold 过滤掉 low-entropy 位置，只在 challenging tokens 上训练。这类似于 **importance sampling** 的思想——把 compute 集中在 model 最不确定、最需要 reasoning 的位置。

验证集也按 difficulty 分 split：
- Easy: entropy > 0.5
- Medium: entropy > 1.0
- Hard: entropy > 1.5

---

## 3. 实验设计详解

### 3.1 Setup

| Hyperparameter | Value |
|---|---|
| Base model | Deepseek-R1-Distill-Qwen-14B |
| Dataset | OmniMATH (4,428 competition-level math problems) |
| Algorithm | GRPO |
| Rollouts per context (G) | 8 |
| Sampling temperature | 0.8 |
| Learning rate | $1 \times 10^{-6}$ |
| Max prompt length | 4,096 |
| Max response length | 8,192 |
| Batch size | 256 |
| Total training steps | 1,000 |
| KL penalty | 0 (zero KL, exact on-policy) |
| Entropy loss coefficient | 0 |

几个关键选择值得讨论：

1. **从 R1-Distill 开始**: 这一点很关键。R1-Distill 已经具备基本 reasoning 能力，所以 RPT 不需要从零学习"如何思考"，而是在已有 reasoning 能力基础上强化 next-token reasoning pattern。从纯 base model (Qwen2.5-14B) 开始是否 work，是 paper 明确提到的 future work。

2. **Zero KL penalty**: 这是 exact on-policy RL 的设置（参考 [On-policy RL with Optimal Reward Baseline](https://arxiv.org/abs/2506.06999)），避免 KL regularization 把模型拉回 reference policy。

3. **OmniMATH 作为 pre-training corpus**: 这其实是一个 domain-specific 的选择。Paper 的 limitation 部分承认了这点——general web text 的效果还需验证。Math 文本的好处是 next-token 的 verifiable reward 语义清晰（数学符号的延续有明确对错），而 general text 的 next token 往往是 creative、多义的。

4. **Dynamic sampling**: 从 step 500 开始，用 [DAPO](https://arxiv.org/abs/2503.14476) 的 dynamic sampling 技术，过滤掉全错（reward 全为 0）或全对（reward 全为 1）的 groups，因为这些 groups 的 advantage 为零，不产生有效梯度。

---

## 4. 实验结果解析

### 4.1 Language Modeling Accuracy (Table 1)

| Model | Easy | Medium | Hard |
|---|---|---|---|
| Qwen2.5-14B (NTP) | 41.90 | 30.03 | 20.65 |
| R1-Distill-14B (NTP) | 41.60 | 29.46 | 20.43 |
| R1-Distill-14B (reasoning) | 3.31 | 1.66 | 1.41 |
| **RPT-14B** | **45.11** | **33.56** | **23.75** |

几个值得注意的点：

1. **R1-Distill 的 reasoning mode 在 NTP 上极差** (3.31% on easy): 这是预期内的。R1-Distill 是为 problem-solving 训练的，让它对每个 token 都生成完整 chain-of-thought 是 misuse。这说明 reasoning 能力不能直接迁移到 token-level prediction。

2. **RPT-14B 全面超越 baselines**: 在所有 difficulty level 上都有 +3 到 +4 个点的提升。更惊人的是，Figure 4 显示 RPT-14B 匹配了 R1-Distill-Qwen-32B（2 倍参数量）的表现。这是一个很强的 scaling 信号——RPT 的 compute 投入换来了参数量的等效提升。

3. **Hard tokens 的相对提升更大**: 从 20.43 到 23.75 是 +16.3% 的相对提升，而 easy 从 41.60 到 45.11 是 +8.4%。这说明 RPT 的收益在高 entropy（model 不确定）的 token 上更显著，这正是 reasoning 最有价值的地方。

### 4.2 Scaling Properties (Figure 5)

RPT 的 scaling law 形式：

$$P(C) = \frac{A}{C^\alpha} + P^*$$

变量含义：
- $P(C)$: training compute 为 $C$ 时的 next-token prediction accuracy
- $P^*$: asymptotic accuracy（compute 趋于无穷时的理论上限）
- $\alpha$: scaling exponent（控制曲线衰减速率）
- $A$: 常数（控制初始 accuracy 与 asymptotic 的差距）

这与 Kaplan et al. 2020 的 NTP scaling law $\mathcal{L}(C) = (C_c/C)^{\alpha_c} + L_\infty$ 形式一致，但针对的是 accuracy 而非 loss，且 power-law decay 转化为 power-law approach to asymptote。

关键发现：在所有 difficulty level 上，$R^2$ 都很高，说明 RPT 的 accuracy 随 compute 的 scaling 是 predictable 的。这对未来的 resource allocation 规划很重要——你可以预测投入更多 compute 会带来多少 accuracy 提升。

### 4.3 Reinforcement Fine-Tuning (Table 2)

| Model | Before RL | After RL |
|---|---|---|
| R1-Distill-14B | 51.2 | 52.7 |
| + Continual NTP training | 10.7 | 13.0 |
| **RPT-14B** | **56.3** | **58.3** |

这个表有几个 deep insights：

1. **Continual NTP training 是灾难性的**: 把 reasoning model (R1-Distill) 用 NTP objective 在 math corpus 上继续训练，reasoning 能力从 51.2 暴跌到 10.7！这是 catastrophic forgetting——NTP objective 逼迫模型放弃 chain-of-thought，回到直接 token prediction，破坏了已学的 reasoning pattern。后续 RLVR 也救不回来（只到 13.0）。

2. **RPT 提供更好的 RL 起点**: RPT-14B 起点是 56.3（比 R1-Distill 的 51.2 高 5 个点），经过 RL fine-tuning 后达到 58.3。更重要的是，RPT 与后续 RLVR 的 objective gap 更小——RPT 本身就是 RL，fine-tuning 只是延续同样的 paradigm，而 R1-Distill 是 distillation 产物，与 RLVR 有 gap。

这个结果对 post-training pipeline 设计有重要含义：**预训练和后训练的 objective 一致性**很关键。RPT 让 pre-training 和 RL fine-tuning 共享同一个 RL framework，减少了 phase shift。

### 4.4 Zero-Shot End Tasks (Table 3)

| Model | SuperGPQA | MMLU-Pro |
|---|---|---|
| R1-Distill-14B (NTP mode) | 32.0 | 48.4 |
| R1-Distill-32B (NTP mode) | 37.2 | 56.5 |
| R1-Distill-14B (reasoning) | 36.1 | 68.9 |
| **RPT-14B (reasoning)** | **39.0** | **71.1** |

RPT-14B 超越了 2 倍大的 R1-Distill-32B（在 NTP mode 下）。这说明 RPT 学到的 reasoning 能力是 generalizable 的，不仅仅局限于 next-token prediction，而是迁移到了 general problem-solving。

特别值得注意的是 MMLU-Pro 的提升（71.1 vs 68.9, +2.2）虽然不大，但考虑到 RPT 是在 math corpus 上训练的，能在 general knowledge benchmark 上有提升说明 next-token reasoning 学到的不仅是 math-specific 的推理，而是某种 general 的 reasoning disposition。

### 4.5 Reasoning Pattern Analysis (Figure 6, Table 4)

这部分非常有意思，揭示了 RPT 学到的 reasoning pattern 与传统 problem-solving 的本质区别。

六个 reasoning pattern 类别：
- **Transition**: "alternatively", "think differently" — 切换策略
- **Reflection**: "wait", "looking back", "thought process" — 自我检查
- **Breakdown**: "break down" — 分解问题
- **Hypothesis**: "probably", "something like" — 提出假设
- **Divergent thinking**: "etc.", "or something", "exploring" — 探索可能性
- **Deduction**: "summarize", "conclusion", "consequently" — 逻辑推断

统计结果：
- RPT-14B 的 **hypothesis pattern 比 R1-Distill 多 161.8%**
- **deduction pattern 多 26.2%**
- 而 R1-Distill 更多使用 **breakdown pattern**

这个差异的 intuition 是：problem-solving 面对的是结构化问题，有明确的解题步骤，所以 breakdown（分解问题）是主要策略。而 next-token reasoning 面对的是 **不确定性下的推断**——模型不知道哪个 token 是对的，必须先提出多个 hypothesis（"可能是 para，可能是 =，可能是空格"），然后通过 deduction 验证哪个最合理。这是一种更接近 **Bayesian inference** 的 reasoning mode。

Table 4 的案例很好地展示了这种 reasoning：模型分析语义上下文（"calculating vector magnitude"），识别关键短语（"go over some..."），然后 brainstorm 多个可能延续，考虑 markdown 格式、token-level 细节（空格），最终选择最 probable 的。这是 high-level semantic understanding 和 low-level textual feature 的结合。

---

## 5. Appendix 中的关键技术细节

### 5.1 Reward Design 的替代方案 (Appendix A)

Paper 探索了三种 alternative reward：

1. **First-token matching**: 只看 prediction 的第一个 token 是否匹配，忽略后续 tokens
2. **Dense reward**: 正确给 1，错误给 $P(y_t^i[0] | x_{<t}; \theta)$（LM 自己的概率）。这提供了比 binary reward 更密集的梯度信号
3. **Conditional dense reward**: 只在 group 中至少一个 rollout 正确时用 dense reward，否则给 zero/penalty

结论是这些 alternative 与 prefix matching reward 效果相当。这说明 RPT framework 对 reward 设计 **robust**，核心收益来自 RL paradigm 本身，而非 reward 的精细设计。这是一个好的信号——意味着 RPT 可能不需要大量 reward engineering。

### 5.2 Prompt Template 的影响 (Appendix D, Table 8, 10)

| Template | Random@1 | Pass@8 |
|---|---|---|
| v0 (used) | 3.0 | 8.5 |
| v1 | 5.7 | 11.0 |
| v2 | 5.7 | 16.0 |
| v3 | 5.3 | 11.0 |
| v4 | 4.0 | 9.0 |
| v5 | 4.4 | 12.5 |
| v6 | 6.0 | 19.0 |

v6 最好（19.0 vs v0 的 8.5），而实验用的是最差的 v0。这意味着 RPT 的实际效果可能还有显著提升空间——更好的 prompt engineering 可能让初始 performance 翻倍。这也暴露了一个问题：RPT 对 prompt 敏感，而 NTP 不需要 prompt engineering，这是 RPT 相比 NTP 的一个额外复杂度。

---

## 6. 深层 Intuition 与思考

### 6.1 为什么 RPT 有效？三个层面的解释

**Compute allocation 层面**：
标准 NTP 对每个 token 分配相同的 compute（一次 forward pass）。但不同 token 的预测难度差异巨大——"the" 几乎不需要思考，而数学证明中的关键步骤可能需要大量推理。RPT 允许 model 对每个 token 动态分配 compute，通过 chain-of-thought 长度隐式地实现 **adaptive computation**。这类似于 mixture of experts 在空间维度上的 sparse activation，RPT 是在时间维度上的 sparse computation。

**Exploration 层面**：
NTP 是 teacher forcing，model 只见到 ground-truth continuation，没有 exploration。RPT 让 model 采样 G=8 个不同的 reasoning trajectory，探索多种可能的推理路径。这种 exploration 让 model 学到的不只是"什么 token 跟在后面"，而是"为什么这个 token 应该跟在后面"——即推理过程本身的 robustness。

**Credit assignment 层面**：
在 NTP 中，如果 model 预测错了某个 token，梯度信号告诉它"这个 token 的概率应该更高"，但不告诉它"为什么"。在 RPT 中，通过 chain-of-thought，model 的推理过程和最终预测之间的因果关系变得 explicit。GRPO 的 group-relative advantage 让 model 知道"这条 reasoning path 导致了正确预测，那条导致了错误预测"，从而 reinforce 正确的推理 pattern。

### 6.2 与 LeCun 的 Cherry-on-the-Cake 的关联

Figure 1 引用了 LeCun 的 cherry-on-top cake 比喻。LeCun 在 [NIPS 2016 keynote](https://www.youtube.com/watch?v=I09mVd5_pog) 中提出：cake 的主体是 unsupervised/predictive learning（预测世界模型），icing 是 supervised learning，cherry 是 RL。LeCun 认为 RL 占的 compute 量应该很小。

RPT 在某种意义上是 **颠覆了这个比喻**——它把 RL 变成了 cake 的主体，让 predictive learning（next-token prediction）通过 RL 来实现。这是一个范式上的反转，暗示 RL 可能比 LeCun 设想的更重要、更 scalable。

### 6.3 与 Test-Time Compute Scaling 的关系

RPT 的核心 insight 可以表述为：**test-time compute scaling 应该在 training 时就被 incentivize，而不是只在 inference 时启用**。

OpenAI o1 和 DeepSeek-R1 的成功表明，让 model 在 inference 时多想能显著提升 performance。但这些 model 的 reasoning 能力主要是在 post-training 阶段通过 RL 获得的。RPT 把这个 RL training 推到 pre-training 阶段，让基础 model 本身就具备 next-token reasoning 能力。

这引出一个更深的思考：**pre-training 和 post-training 的边界是否应该模糊化**？传统 pipeline 是 pre-train (NTP) → SFT → RLHF/RLVR，三个阶段 objective 不同。RPT 暗示我们或许应该有一个统一的 RL objective 贯穿所有阶段，只是 data 和 reward 不同。

### 6.4 潜在问题与 Limitations

1. **Training cost**: RPT 每个 token 需要 G=8 个 rollouts，每个 rollout 包含 chain-of-thought（可能几百个 tokens），加上 entropy filtering 只用一小部分 tokens。这意味着 RPT 的有效 compute 远超 NTP。Table 1 显示 RPT-14B 匹配 32B，但如果算上 RPT 的总 compute（包括所有 rollouts），可能并不比训练 32B 便宜。这个 cost-effectiveness 的比较 paper 没有详细讨论。

2. **Inference cost**: 用 RPT model 做 language modeling 也需要生成 chain-of-thought，inference 成本远高于标准 NTP。虽然 paper 说 RPT model 也可以用 NTP mode（直接预测），但 Table 1 显示 RPT-14B 在 NTP mode 下的表现没有报告——这是一个重要缺失。

3. **Math-specific bias**: OmniMATH 是高度结构化的 domain，next-token 的 verifiable reward 语义清晰。General web text 中，next token 往往是 creative、多义的、甚至任意选择的（如同义词替换）。在这些 token 上，verifiable reward 可能过于 sparse 且 noisy。Paper 承认这点是 future work。

4. **From reasoning model**: 从 R1-Distill 开始意味着 RPT 继承了已有的 reasoning 能力。如果从 base model (Qwen2.5-14B) 开始，model 需要先学会"如何思考"才能做 next-token reasoning，这可能需要更长的训练和更精细的 curriculum。这个 cold-start 问题是 RPT 能否真正成为 pre-training 范式的关键。

5. **Reward sparsity**: Binary reward (0/1) 在 hard tokens 上可能极 sparse——如果 8 个 rollouts 全错，GRPO 的 advantage 全为零，没有梯度信号。Appendix A 的 dense reward 是缓解方案，但 paper 说效果相当，没有深入分析何时 dense reward 更优。

---

## 7. 与相关工作的联系网络

### 7.1 Quiet-STaR (Zelikman et al. 2024)
最直接的前置工作。Quiet-STaR 也让 model 在 token prediction 前生成 rationale。区别在于 reward：Quiet-STaR 用 helpfulness-based reward（rationale 是否帮助预测），容易被 hack（model 重复 target token 在 rationale 中）。RPT 用 correctness-based rule reward 避免 hacking。[论文链接](https://arxiv.org/abs/2403.09629)

### 7.2 DeepSeek-R1 (Guo et al. 2025)
RPT 用的 GRPO 算法和 RLVR framework 直接来自 R1。R1 证明了 RLVR 在 math reasoning 上的威力，RPT 把 RLVR 从 fine-tuning 扩展到 pre-training。[论文链接](https://arxiv.org/abs/2501.12948)

### 7.3 Scaling Laws (Kaplan et al. 2020; Hoffmann et al. 2022)
RPT 的 scaling law 形式继承自 NTP 的 power-law tradition。但 RPT 的 scaling 是 accuracy vs compute，而传统是 loss vs compute。这个转变反映了 RPT 关注的是 prediction accuracy 而非 likelihood。[Kaplan](https://arxiv.org/abs/2001.08361), [Chinchilla](https://arxiv.org/abs/2203.15556)

### 7.4 Process Reward Models & Step-level RL
RPT 的 reward 是 outcome-level（最终 prediction 对错），但 chain-of-thought 中的中间步骤没有直接 reward。这与 process reward models (PRM) 的思路不同——PRM 对每个推理步骤给 reward，RPT 只看最终结果。RPT 依赖 GRPO 的 group-relative advantage 来隐式地做 credit assignment。[Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)

### 7.5 Search-based Test-Time Scaling (e.g., beam search, MCTS)
RPT 在 training 时就让 model 探索 G 个 rollout，这与 inference 时的 beam search / MCTS 有相似之处。区别是 RPT 的 exploration 是 on-policy 且通过 reward 反向传播来优化，而 beam search 是 inference-time 的 heuristic。RPT 可以看作把 search 的思想内化到 model parameters 中。

---

## 8. 未来方向与开放问题

基于 paper 的 discussion 和我的思考：

1. **General text RPT**: 如何定义 general text 上的 verifiable reward？可能的方向包括：(a) 多个 candidate tokens 都算正确（top-k reward）；(b) 用更强的 model 作为 verifier；(c) 结合 semantic similarity 而非 exact match。

2. **From base model**: 从 Qwen2.5-14B（无 reasoning 能力）开始 RPT，需要解决 cold-start 问题。可能的方案：(a) 先用少量 reasoning data SFT；(b) curriculum learning，从简单 reasoning 开始；(c) hybrid objective，初期 NTP + RPT，逐渐过渡到纯 RPT。

3. **Hybrid thinking integration**: Paper 提到与 [Hybrid Reasoning Models](https://arxiv.org/abs/2506.09992) 结合，让 model adaptively 决定何时触发 next-token reasoning。简单 token 直接预测，困难 token 才 reasoning。这可以大幅降低 inference cost。

4. **Multi-modal extension**: RPT 的思想可以扩展到 vision-language models——预测下一个 image patch 或 visual token，用 reconstruction error 或 perceptual similarity 作为 reward。

5. **RPT + World Model**: RPT 本质上是训练 model 预测世界的下一个状态（用 token 表示）。这与 LeCun 的 JEPA (Joint-Embedding Predictive Architecture) 思想有呼应。RPT 在 token space 预测，JEPA 在 latent space 预测，两者可能互补。[LeCun on JEPA](https://openreview.net/pdf?id=BZ5a1r-kVsf)

6. **Reward design for creativity**: 在 creative writing 等任务上，next token 没有唯一正确答案。如何设计 reward 鼓励 model 探索多种合理延续？可能需要 distributional reward 或 multi-reference reward。

---

## 9. 总结：RPT 的深层意义

RPT 的贡献超越了一个新 training technique。它重新定义了 pre-training 的本质：

**传统视角**: pre-training 是学习语言的 statistical regularity，通过 maximum likelihood 拟合数据分布。
**RPT 视角**: pre-training 是学习如何 reasoning about 世界的下一个状态，通过 RL 在 verifiable reward 上优化。

这个视角的转变有几个深远含义：

1. **Pre-training 不再是 "cheap" 的 self-supervised learning**：RPT 的 compute cost 远超 NTP，但这可能是值得的——因为 model 学到的是 reasoning ability 而非 surface pattern。

2. **RL 与 self-supervised learning 的统一**：RPT 暗示 RL 可能是比 maximum likelihood 更 fundamental 的 learning paradigm。任何 prediction task 都可以转化为 RL（prediction correctness 作为 reward）。

3. **Intelligence as prediction**: RPT 呼应了 predictive coding 和 free energy principle 的理论——大脑的核心功能是预测下一个感官输入，通过最小化 prediction error 来学习。RPT 把这个思想在 LLM 上实现，用 RL 替代 MLE 来优化 prediction。

4. **Towards AGI**: 如果 RPT 能 scale 到 general web text，并且从 base model 开始 work，它可能成为一个真正的 general-purpose pre-training paradigm，让 model 在 pre-training 阶段就获得 strong reasoning ability，而不仅仅依赖 post-training 的 RL。

最后，RPT 也提出了一些深刻的科学问题：为什么 RL 比 MLE 更有效地学习 reasoning？是因为 exploration？credit assignment？还是 adaptive computation？这些问题值得理论上的深入分析。

**参考资源**：
- [RPT 项目页](https://aka.ms/GeneralAI)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [Quiet-STaR](https://arxiv.org/abs/2403.09629)
- [GRPO / DAPO](https://arxiv.org/abs/2503.14476)
- [Scaling Laws](https://arxiv.org/abs/2001.08361)
- [LeCun NIPS 2016](https://www.youtube.com/watch?v=I09mVd5_pog)
- [verl RL framework](https://arxiv.org/abs/2409.19256)
- [OmniMATH](https://arxiv.org/abs/2410.07985)
- [SuperGPQA](https://arxiv.org/abs/2502.14739)
- [MMLU-Pro](https://arxiv.org/abs/2406.01574)
- [Hybrid Reasoning Models](https://arxiv.org/abs/2506.09992)
- [On-policy RL with Optimal Reward Baseline](https://arxiv.org/abs/2506.06999)
