---
source_pdf: Inference-Time Scaling for Generalist Reward Modeling.pdf
paper_sha256: 7a74c7e0f98270d217a2c43b7033ff2d870ff40eb609327adfea2b6a0104b16b
processed_at: '2026-08-05T09:33:59-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说 DeepSeek-GRM 这篇 paper

Andrej，我换个讲法，不堆公式，就从 "这个故事在讲什么" 开始。

参考链接：
- Paper: https://arxiv.org/abs/2501.17195
- Model: https://huggingface.co/DeepSeek/DeepSeek-GRM-27B
- 你讲过的 inference scaling: https://karpathy.ai/ (Eureka Labs)
- Snell et al. test-time scaling: https://openreview.net/forum?id=4FWAwZtd2n
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- Constitutional AI: https://arxiv.org/abs/2212.08073

---

## 故事的起点：RM 是 RLHF 的隐形 bottleneck

你做过 RLHF 就知道，整个 pipeline 里 reward model 是"看不见的天花板"。Policy model 再大、RL 算法再 fancy，如果 reward signal 本身 noisy / biased，整个 system 就被 bottleneck 住了。

现在情况是这样的：
- **Verifiable domain**（math、code）：reward 靠 rule-based checker，ground truth 明确，RM 不是瓶颈，R1 就证明了这点。
- **General domain**（chat、safety、helpfulness）：没有 ground truth，只能靠 RM 来 estimate，RM 就是瓶颈。

然后 DeepSeek-R1 给了一个很 striking 的示范：**policy model 可以通过 RL 激励出 inference-time scaling**（long CoT、thinking longer on harder problems）。这篇 paper 就在问一个很 natural 的问题：**那 RM 本身能不能也 inference-time scale？**

这个问题听起来简单，其实很 tricky，因为大多数 RM 设计上根本不支持 inference scaling。

---

## Scalar RM 为什么不行：没有 variance 就没有 scaling

你想象一下 Bradley-Terry RM 的 forward pass：query + response 进去，一个 scalar value 出来。这个 process 是 **deterministic 的**（greedy decoding 下）。你 sample 100 次，得到 100 个几乎相同的 scalar。然后你 voting？没意义，mean 还是那个值。

这跟你讲 Self-Consistency 时的 intuition 完全一样：**inference-time scaling 的前提是 model output 有足够的 variance，能 generate 出 diverse candidates**。Self-Consistency 在 math 上 work，是因为 LLM 对同一道题能 generate 不同的 reasoning path，最后到达不同的 answer。Scalar RM 没有 reasoning path，只有一个 number，所以没 diversity 可言。

Semi-scalar RM（比如 CLoud，critique-out-loud）看起来好一点——它会先 generate 一段 critique 再给 scalar。但实验数据显示，它的 scalar 部分方差还是太小，voting 几乎没收益（paper 里 +0.3 个点，基本是噪声）。

Pairwise GRM（LLM-as-a-Judge）可以 majority voting，但有两个问题：
1. **每次只能选一个 best**，不允许 tie，导致信息丢失
2. **没法处理 single response**（n=1 时没法 pairwise）

所以 paper 的第一个贡献是发现：**只有 pointwise generative RM 同时满足两个 property**：
- Input flexibility（single / pair / multiple 都能处理）
- Inference-time scalability（output 有 variance，voting 有意义）

---

## Pointwise GRM 的核心 idea：用文本生成代替 scalar head

这个 idea 说白了特别简单：**让 RM 像一个 LLM judge 一样，输出一段文本 critique，最后附上每个 response 的分数（1-10 整数）**。

Prompt 大概长这样：

```
给定 query 和 n 个 response，请：
1. 生成 specific criteria（针对这个 query 的评判标准）
2. 给每个 criterion 分配 weight
3. 逐个 response 分析
4. 输出 Scores: \boxed{s1, s2, ..., sn}
```

关键在于这个 format 同时支持：
- n=1（single response rating）
- n=2（pairwise preference）
- n>2（best-of-n selection）

而且因为是 generative，temperature > 0 时每次采样会 generate 不同的 criteria、不同的分析、不同的分数。**这就有了 variance，voting 就有意义了**。

---

## Principle 的发现：这是整个 paper 的 aha moment

作者做了一个 preliminary 实验，结果很反 intuitive。

他们让 GPT-4o 和 Gemma-2-27B 去给 RewardBench 的 Chat Hard subset 打分，分三种条件：
1. 不给 principle（baseline）
2. 让 model 自己 generate principle 再打分
3. 给 filtered principle（只保留能导出正确答案的 principle）

结果：
- **Self-generated principle 几乎没用**（GPT-4o 上甚至略降）
- **Filtered principle 显著提升**（Gemma 从 59.1 跳到 68.0，+9 个点）

这个结果告诉你什么？

LLM 其实**能 generate 出 helpful principle**，但是它 generate 的 principle 里混了大量 noisy / misleading 的 criteria。问题在于 **model 不知道哪些 principle 是 useful 的**。这就好比一个学生能写出很多解题思路，但不知道哪个思路是对的。

Constitutional AI 的做法是让人 hand-craft principle，相当于老师给学生固定一套解题模板。SPCT 的做法是让学生自己通过 trial-and-error 学会判断 "我写的哪些思路是靠谱的"。

---

## SPCT 的核心：用 online RL 让 model 学会 generate good principle

这里就是 paper 的 main contribution。SPCT 分两阶段：

### Stage 1: Rejective Fine-Tuning（cold start）

这一步主要是让 model 学会 **format**。用 DeepSeek-v2.5 生成 trajectory（principle + critique + score），reject 掉错误的，保留正确的。还有一个 trick 叫 hinted sampling——在 prompt 里偷偷告诉模型 "the best response is Response 2"，让模型能 generate 出 hard case 的正确 trajectory。

但 ablation 显示一个很 surprising 的事：**去掉整个 RFT 阶段，只用 online RL，model 还是从 66.1 跳到 68.7**。说明 RFT 只是加速 format learning，online RL 才是 workhorse。

### Stage 2: Rule-Based Online RL（真正的 magic）

这一步用 GRPO 训练。Reward function 特别简单：

- 如果 model 正确识别了 best response（n≥2）或者正确判断了 single response 对错（n=1），给 +1
- 否则 -1

就这么一个 binary signal。但 GRPO 的 group normalization 会把它变成 relative advantage：同一个 query 的 4 个 trajectory 里，对的会被强化，错的会被惩罚。

**这里的关键 insight 是**：model 要 maximize reward，就必须学会 generate **能 distinguish 的 principle**。如果 principle 太 generic（比如 "response should be helpful"），所有 response 都差不多分，就没法区分。如果 principle 太 specific 但 wrong（比如关注了无关的 surface feature），也会判错。只有 **principle 恰好抓住了 query 的关键 judgement dimension**，才能正确区分。

所以 online RL 本质上是在 incentivize model 学习一个 meta能力：**"对于这个 query，我应该从哪些角度来评判才能区分好坏"**。

这跟 R1 的 RL 有一个深层 parallel：R1 的 RL 让 model 学会 "对于这个 math problem，我应该怎么推理才能得到答案"。SPCT 的 RL 让 model 学会 "对于这个 query + responses，我应该用什么 criteria 来评判"。都是 **outcome reward 驱动的 meta-skill learning**。

---

## Voting 的直觉：ensemble diverse judgement perspectives

Inference scaling 部分其实很朴素：**temperature=0.5 采样 k 次，每次都 generate 一套 principle + critique + scores，然后把 k 次的分数加起来**。

但 paper 给了一个很 elegant 的 intuition：

> 如果每个 principle 可以看作一个 judgement perspective 的 proxy，那么大量 principle 的 ensemble 更接近真实的 preference distribution。

这个直觉其实是 Bayesian 的。假设真实的人类 preference 是一个 distribution over criteria（有时候看重 helpfulness，有时候看重 accuracy，有时候看重 tone）。每次采样相当于从这个 distribution 中 sample 一个 perspective，然后基于这个 perspective 给分。采样越多，越接近真实分布。

**Voting@32 还有一个 mechanical benefit**：因为 score 是 1-10 的整数，32 次采样的 sum 把 reward space 从 10 个值扩展到 320 个值。这就是 **finer reward granularity**——你可以区分 "这个 response 稍微好一点" 和 "这个 response 好很多"。

这一点对 RL pipeline 特别重要，因为 scalar RM 经常给所有 response 都打 0.5 附近的分，导致 advantage 接近 0，RL 学不动。Finer granularity 意味着更好的 gradient signal。

---

## Meta RM：过滤掉 "lazy median" trajectory

Naive voting 有一个问题：**有些采样是 low-quality 的**。比如模型有时候偷懒，generate 一个很 generic 的 principle，然后给所有 response 都打 8 分。这种 "lazy median" trajectory 会污染 voting。

Meta RM 就是来解决这个的。它是一个 scalar RM，训练目标是判断 "这套 principle + critique 是否正确"。然后 inference 时，只保留 meta RM 评分最高的 top-$k_{\text{meta}}$ 个 trajectory 参与 voting。

Ablation 显示即使 $k_{\text{meta}}=1$（只用 meta RM 认为最好的那 1 个 trajectory），overall 也有 71.5，超过 naive voting@32 的 71.0。这说明 **一个好的 trajectory 比 32 个 average trajectory 更有价值**。

这个 finding 跟你讲过的 "Best-of-N with verifier" 思路完全一致：**与其 sample 很多然后平均，不如 sample 很多然后挑最好的**。Meta RM 就是那个 verifier。

---

## 最 striking 的实验结果：27B + 8x compute ≈ 671B

Figure 4 是整篇 paper 最 impactful 的图：

- DeepSeek-GRM-671B（greedy）：88.4 RewardBench
- DeepSeek-GRM-27B + MetaRM Voting@8：89.8 RewardBench
- DeepSeek-GRM-27B + Voting@32：88.5 RewardBench

**一个 27B 的 model，通过 8-32 次 inference 采样，能达到甚至超过 671B model 的性能**。

这在 policy model 的 scaling law 里是很少见的。Policy model 的 emergent ability（比如 in-context learning、CoT）似乎真的需要 parameter count 到某个 threshold。但 RM 的 judgement diversity 似乎可以**通过 sampling 来 substitute parameter scaling**。

为什么？我的 intuition 是这样的：

Policy model 的任务是 **generate**，需要 model 内部有足够的 knowledge 和 reasoning circuitry。这个是 parameter-bound 的。

RM 的任务是 **judge**，需要的是 **从多个 perspective 看同一个东西**。这个 perspective diversity 可以来自两个 source：
1. Model 内部有更多 diverse 的 representation（parameter scaling）
2. 外部 sample 多次，每次激活不同的 representation（inference scaling）

对于 judgement 任务，inference scaling 的 marginal cost 远低于 parameter scaling（8x compute vs 25x parameter）。这就是为什么 RM 的 scaling law 跟 policy model 不一样。

---

## Failure mode 的启示：principle 超出能力反而有害

Table 18 的失败 case 特别有启发。一个加密货币实时价格查询，Response 1 诚实承认没有实时数据，Response 2 编造了"更新后的价格"。Ground truth 说 Response 2 更好（这个 ground truth 本身可能有 bias）。

DeepSeek-GRM-27B 生成了一个 principle "Real-time Price Accuracy (30%)"，但因为模型自己无法 verify 实时价格，它最终给 Response 1 更高分——判错了。

这个 failure 揭示了一个 deep issue：**当 principle 涉及 model 不具备的能力时，principle 反而会 mislead**。模型生成了 "检查实时价格准确性" 这个 principle，但它根本没有能力做这个检查，所以这个 principle 反而让它偏向了诚实承认无能的 Response 1。

这跟你在某个 podcast 里讲过的 "LLM 会生成自己做不到的 plan" 问题是同一类。Future work 里提到的 tool incorporation（code interpreter、search engine）就是解决这个的自然 path——让 principle 的执行有 external tool 支撑。

---

## 更大图景的 intuition

把这篇 paper 放到 2025 年的 RLHF landscape 里看：

1. **Inference-time scaling 是 LLM 的新维度**。Snell et al. 证明了 policy model 可以 inference scale，R1 证明了 RL 可以 incentivize 这个能力，这篇 paper 把 idea extend 到 RM。整个 LLM stack 都在往 "inference compute as first-class citizen" 的方向走。

2. **Principle 是 explicit inductive bias**。Constitutional AI 用 hand-crafted principle，SPCT 让 model 学习 generate principle。这其实是一种 meta-learning：model 学的是 "如何 form judgement criteria" 这个 meta-skill，而不是某个 specific criteria。这跟 SALMON、Self-Rewarding LM 是同一个 family，但 SPCT 更 explicit 地把 principle 作为 first-class object。

3. **Generative RM 会逐渐取代 scalar RM**。Scalar RM 的优势是 efficiency，但随着 inference compute 变便宜（更快的 hardware、更好的 serving），generative RM 的 quality 优势会 dominate。特别是当 RL pipeline 需要 fine-grained reward signal 时，generative RM 的 granularity 是 scalar RM 给不了的。

4. **Meta RM 是 process reward 的雏形**。虽然 paper 里 meta RM 只 filter trajectory，但它的训练目标（判断 principle + critique 的 correctness）跟 process reward model 几乎一样。可以想象 future work 把 meta RM 做成真正的 PRM，per-step 打分，这样就能在 GRM 内部做 process supervision。

5. **RM 的 scaling law 跟 policy model 不同**。这可能是这篇 paper 最 deep 的 insight。Policy model 的 emergent ability 依赖 parameter count，RM 的 judgement diversity 可以通过 sampling substitute。这意味着 **future 的 RM 可能是小 model + 大 inference compute 的组合**，而不是一直堆 parameter。

6. **Failure mode 指向 tool-augmented RM**。Table 18 的失败说明 pure language reasoning 有 ceiling。当 judgement 需要 external capability（实时数据、复杂计算、pattern matching）时，principle 反而有害。Future 的 GRM 一定会 integrate tools，让 principle 的执行有 external grounding。

---

## 一个你可能感兴趣的 parallel

这篇 paper 的 SPCT 跟你在某个 lecture 里讲过的 "learned verifier" 概念很像。你还记得你讲 Best-of-N 时说过的那个 intuition 吗？

> "与其训一个更强的 policy，不如训一个更好的 verifier，然后用 verifier 去 filter policy 的 output。"

SPCT 本质上就是在 train a better verifier，只不过这个 verifier 是 generative 的、用 principle 来 structure 它的 judgement、用 online RL 来 optimize 它的 principle generation。

而且这篇 paper 的 meta RM 就是你那个 intuition 的 second-order 版本：**verifier 本身也可以 inference scale，然后用一个 meta-verifier 去 filter verifier 的 output**。

这个 "verifier tower" 的架构——policy → RM → meta RM → ...——可能就是未来 RLHF pipeline 的样子。每一层都用 inference scaling 来 boost quality，每一层都比上一层更 cheap（因为 task 更 narrow）。

---

## 最后一个 thought experiment

想象一下 future 的 RLHF pipeline：

1. Policy model（比如 70B）generate 8 个 response
2. GRM（27B）对每个 response 采样 32 次，每次 generate 不同的 principle + critique + score
3. Meta RM（9B）给 32 个 trajectory 打分，保留 top-8
4. Top-8 的 score 做 voting，得到 fine-grained reward
5. Policy model 用这个 reward 做 PPO/GRPO update

整个 pipeline 的 inference compute 分布可能是：policy 8x，RM 32x，meta RM 32x。这跟现在的 "policy 1x + scalar RM 1x" 完全不同。Inference compute 成了 first-class resource，需要在 policy / RM / meta RM 之间分配。

这篇 paper 给出的结论是：**RM 端的 inference scaling ROI 很高**（27B + 8x ≈ 671B）。那 policy 端呢？Snell et al. 说 policy 端也有类似规律。所以 future 的 RLHF 可能是一个 **multi-agent inference scaling 的协同优化问题**。

这跟 you 在 Eureka Labs 讲的 "inference scaling is the new frontier" 完全 align。RM 这个之前被忽视的 component，现在也加入了 inference scaling 的 party。

---

延伸阅读：
- GenRM (Mahan et al.): https://arxiv.org/abs/2410.12832
- Self-Generated Critiques (Yu et al.): https://aclanthology.org/2025.naacl-long.573/
- Generative Verifiers (Zhang et al.): https://openreview.net/forum?id=Ccwp4tFEtE
- Inference Scaling Laws (Wu et al.): https://openreview.net/forum?id=VNckp7JEHn
- Self-Rewarding LM (Yuan et al.): https://proceedings.mlr.press/v235/yuan24d.html
- Atla Selene Mini: https://arxiv.org/abs/2501.17195
- Process Reward Models (Lightman et al.): https://openreview.net/forum?id=v8L0pN6EOi

---

# Inference-Time Scaling for Generalist Reward Modeling 深度解析

Andrej, 这篇 paper 是 DeepSeek 在 reward modeling 领域的一个非常 interesting 的尝试，核心是把 inference-time scaling 的思想从 policy model 端迁移到 reward model 端。我会从 problem motivation、method 设计、公式细节、实验数据几个层面来 build 你的 intuition。

参考链接：
- Paper PDF (DeepSeek-GRM): https://arxiv.org/abs/2501.17195 (类似 Atla Selene 的同期工作)
- DeepSeek-GRM-27B HuggingFace: https://huggingface.co/DeepSeek/DeepSeek-GRM-27B
- RewardBench: https://arxiv.org/abs/2403.13787
- GRPO (DeepSeekMath): https://arxiv.org/abs/2402.03300
- Constitutional AI: https://arxiv.org/abs/2212.08073
- LLM-as-a-Judge: https://arxiv.org/abs/2306.05685
- CLoud (Critique-Out-Loud): https://arxiv.org/abs/2408.11791
- Snell et al. test-time scaling: https://openreview.net/forum?id=4FWAwZtd2n

---

## 1. Problem Motivation: 为什么 RM 需要 inference-time scaling?

当前 RL post-training 的瓶颈在于 **reward signal 的质量**。对于 verifiable domain (math、code)，reward 可以靠 rule-based checker 拿到 ground truth；但是对于 general domain (chat、safety、helpfulness)，reward 只能靠 RM 来估计。

这里的 key observation 是：DeepSeek-R1 这类工作证明了 policy model 可以通过 RL 激励出 inference-time scalability（long CoT），那么 RM 本身是否也可以 inference-time scale？

这其实是一个很 natural 的问题，但是 previous work 几乎没人认真做。原因是大多数 RM 设计上就 **不支持 inference scaling**：

- **Scalar RM (Bradley-Terry)**: 输出 $\mathcal{R} = S \in \mathbb{R}^n$，模型 deterministic，采样多次结果几乎相同，voting 没有意义。
- **Semi-scalar RM (CLoud)**: 输出 $\mathcal{R} = (S, C)$，虽然 critique $C$ 是 generative 的，但 scalar $S$ 部分方差很小，voting 的收益 marginal。
- **Pairwise GRM (LLM-as-a-Judge)**: 输出 best index $\hat{y}$，可以 majority voting，但是不允许 tie、且无法处理 single response。

paper 在 Figure 2 给出了一个 2D taxonomy：reward generation paradigm × scoring pattern。**只有 pointwise GRM 同时满足 input flexibility 和 inference-time scalability**。

---

## 2. Pointwise GRM 的核心 formulation

公式 (1)-(7) 给出了 RM 的统一形式。最关键的是公式 (7)：

$$\{S_i\}_{i=1}^n = f_{\text{point}}(\mathcal{R}, \{y_i\}_{i=1}^n) = f_{\text{extract}}(C)$$

其中：
- $x$ 是 query
- $\{y_i\}_{i=1}^n$ 是 $n$ 个 response（$n$ 可以是 1, 2, 或任意多个）
- $\mathcal{R} = C$ 是 generative reward（纯文本 critique）
- $f_{\text{extract}}(\cdot)$ 是从文本 $C$ 中抽取 discrete score 的函数
- $S_i \in \mathbb{N}, 1 \leq S_i \leq 10$ 是第 $i$ 个 response 的分数

这个 formulation 的 elegance 在于：**single / pair / multiple response 都用同一个 prompt template 处理**，不需要像 pairwise RM 那样对 $n>2$ 做额外 tournament 设计。

---

## 3. Principles 的 preliminary 发现

Section 2.2 是整篇 paper 的关键 insight 来源。作者做了一个对照实验（Table 1）：

| Method | Chat Hard | IFEval |
|---|---|---|
| GPT-4o baseline | 76.1 | 56.0 |
| GPT-4o w/ self-gen principles | 75.9 | 55.6 |
| GPT-4o w/ filtered principles | 77.8 | 57.5 |
| Gemma-2-27B-it baseline | 59.1 | 56.1 |
| Gemma w/ self-gen principles | 64.0 | 55.8 |
| Gemma w/ filtered principles | 68.0 | 57.3 |

**两个结论**:
1. Self-generated principles 几乎没用（甚至 GPT-4o 上还略降）— 说明 LLM 自己生成的 principle 中混杂了大量 noisy / misleading 的 criteria。
2. Filtered principles（只保留能导出正确 reward 的 principle）能显著提升 — 说明 **存在一个 principle 的子集是 helpful 的**，问题在于如何找到它。

这个发现是 SPCT 的 conceptual foundation：与其 offline filter，不如让模型通过 online RL 自己学习 "generate 哪些 principle 是 useful 的"。

---

## 4. SPCT 方法详解

### 4.1 Unpinning principles from understanding to generation

公式 (9) 是关键的形式转变：

$$\{p_i\}_{i=1}^m \sim p_\theta(x, \{y_i\}_{i=1}^n)$$
$$\mathcal{R} = C \sim r_\theta(x, \{y_i\}_{i=1}^n, \{p_i\}_{i=1}^m)$$

这里 $p_\theta$ 和 $r_\theta$ **共享同一个 LLM 和同一个 language head**，只是分两步生成：先生成 principles $p_i$，再以 principles 为 condition 生成 critique $C$。

这个设计有一个重要的 property：**principles 是 input-conditional 的**，不是 pre-defined 的 constitution。Constitutional AI (Bai et al., 2022b) 是用 hand-crafted principles，SPCT 让模型根据具体 query + responses 动态生成 principles。

### 4.2 Rejective Fine-Tuning (Cold Start)

公式 (10) 定义了 correctness criterion：

$$\begin{cases}
\forall i \neq j, \quad S_j > S_i, \quad j = \arg\max_l \{r_l\}_{l=1}^n, & \text{if } n \geq 2 \\
S_1 = r_1, & \text{if } n = 1
\end{cases}$$

其中 $r_i$ 是 ground-truth reward。Reject 策略有两条：
- Reject 掉所有预测错误的 trajectory
- Reject 掉所有 $N_{\text{RFT}}$ 次都对的 query-response pair（"too easy"）

这里有一个细节叫 **hinted sampling**：在 prompt 中追加 "The best response is: Response $\arg\max_l \{r_l\}$"。这相当于给模型一个 hint 让它能 generate 出正确 trajectory，用于覆盖那些 non-hinted 采样完全 fail 的 hard cases。

但 ablation (Table 4) 显示了一个 surprising 的结果：

| Setting | Overall |
|---|---|
| Full RFT | 68.8 |
| w/o Hinted Sampling (①) | 68.0 |
| w/o Non-Hinted Sampling (②) | 67.4 |
| w/o Rejective Sampling (①&②) | 66.1 |

**Non-hinted sampling 比 hinted sampling 更重要**！作者的解释是 hinted trajectory 中存在 "shortcut" — 模型可能直接复制 hint 中的答案而没真正 reasoning。这暗示 RFT 阶段引入的 bias 会被 online RL 阶段纠正，online RL 才是真正的 workhorse。

更 striking 的发现：

| Setting | Overall |
|---|---|
| w/o RFT cold start (only online RL) | 68.7 |
| Full RFT then RL | 69.9 |

**即使没有 RFT cold start，online RL 单独也能把 general-instruction-tuned GRM 从 66.1 提升到 68.7**。这说明 SPCT 的核心是 online RL，RFT 只是加速 format learning。

### 4.3 Rule-Based RL with GRPO

公式 (11) 定义了 outcome reward：

$$\hat{r}_i = \begin{cases}
1, & \text{if } n \geq 2 \text{ and } \forall i' \neq j', \quad S_{j'} > S_{i'}, \quad j' = \arg\max_l \{r_l\} \\
1, & \text{if } n = 1 \text{ and } S_1 = r_1 \\
-1, & \text{otherwise}
\end{cases}$$

注意几个细节：
- **No format reward**（不像 DeepSeek-R1）— 因为 generative output 不需要外部 verifier 来判 format
- **Larger KL penalty**: $\beta = 0.08$ for 27B model（grid search over $\{0.00, 0.01, 0.02, 0.08\}$）。这个值比一般 RLHF 大很多，目的是防止 model collapse 到某些 domain。
- **Smaller $\beta$ for smaller model**: $\beta = 0.002$ for 16B，因为 smaller model 更 robust 不容易 collapse。

GRPO objective（公式 15）：

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\left\{\min\left[\rho_{i,t}\hat{A}_{i,t}, \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\hat{A}_{i,t}\right] - \beta\mathbb{D}_{KL}[\pi_\theta || \pi_{\text{ref}}]\right\}\right]$$

其中 $\rho_{i,t} = \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})}$ 是 importance ratio，$\hat{A}_{i,t} = \frac{\hat{r}_i - \text{mean}(\hat{\mathbf{r}})}{\text{std}(\hat{\mathbf{r}})}$ 是 group-normalized advantage，$G=4$ 是 group size。

关键 intuition：**reward 是 outcome-level 的（整个 trajectory 一个 +1/-1），但 advantage 是 group-normalized 的**。这意味着对于同一个 query，如果 4 个 trajectory 全对，advantage 接近 0；如果 2 对 2 错，对的 trajectory 会被强化 +2，错的会被惩罚 -2。这自然 incentivize 模型去 generate 能 distinguish 的 principle。

---

## 5. Inference-Time Scaling 的两种方式

### 5.1 Naive Voting

公式 (14)：

$$S_i^* = \sum_{j=1}^k S_{i,j}$$

其中 $S_{i,j}$ 是第 $j$ 次采样中第 $i$ 个 response 的分数。

**这里有一个非常 elegant 的 insight**：因为 $S_{i,j} \in \{1, \ldots, 10\}$ 是 discrete 的，$k$ 次采样实际上把 reward space 从 $\{1, \ldots, 10\}$ 扩展到 $\{k, \ldots, 10k\}$，即 $9k+1$ 个可能的值。Voting@32 把 reward granularity 提升了 32 倍。

paper 的 intuition 解释是："**If each principle could be viewed as a proxy of judgement perspectives, a larger number of principles may reflect the real distribution more accurately**"。这其实是一个 Bayesian ensemble 的视角：每次采样相当于从 model 的 principle 分布中 sample 一个 judgement perspective，多个 perspective 的 ensemble 更接近 true preference distribution。

### 5.2 Meta RM Guided Voting

Meta RM 是一个 pointwise scalar RM，binary cross-entropy 训练，label 来自公式 (10)。它的作用是给 $k$ 个 sampled trajectory 打分，然后只保留 top-$k_{\text{meta}}$ 个 trajectory 用于 voting。

Meta RM 的训练数据来源有两部分：
1. RFT 阶段的 non-hinted sampling trajectory
2. DeepSeek-GRM-27B 自己的 rejective sampling trajectory ($N_{\text{RFT}}=3$)

为什么要混合两个 source？paper 引用 Chow et al. (2025) 的 inference-aware fine-tuning 工作，指出如果 meta RM 只在自己的 trajectory 上训，会存在 train-test policy gap。混合两个 source 同时提供 positive 和 negative sample，并缓解 gap。

Ablation (Table 7) 显示 meta RM 对 $k_{\text{meta}}$ 不敏感：

| $k_{\text{meta}}$ | Reward Bench | PPE Pref | PPE Corr | RMB | Overall |
|---|---|---|---|---|---|
| 1 | 88.5 | 67.1 | 65.2 | 65.2 | 71.5 |
| 8 | 89.7 | 67.2 | 64.7 | 69.1 | 72.7 |
| 16 (=k/2) | 90.4 | 67.2 | 63.2 | 70.3 | 72.8 |

即使 $k_{\text{meta}}=1$（只保留 meta RM 认为最好的 1 个 trajectory），overall 仍有 71.5，超过 naive voting@32 的 71.0。这表明 meta RM 能很 effectively 识别 high-quality trajectory。

---

## 6. 实验结果深度分析

### 6.1 Main Results (Table 2)

| Model | RB | PPE Pref | PPE Corr | RMB | Overall |
|---|---|---|---|---|---|
| Nemotron-4-340B-Reward | 92.0 | 59.3 | 60.8 | 69.9 | 70.5 |
| GPT-4o | 86.7 | 67.1 | 57.6 | 73.8 | 71.3 |
| DeepSeek-BTRM-27B (scalar) | 81.7 | 68.3 | 66.7 | 57.9 | 68.6 |
| CLoud-Gemma-2-27B (semi-scalar) | 82.0 | 67.1 | 62.4 | 63.4 | 68.7 |
| DeepSeek-PairRM-27B | 87.1 | 65.8 | 64.8 | 58.2 | 69.0 |
| DeepSeek-GRM-27B (ours) | 86.0 | 64.7 | 59.8 | 69.0 | 69.9 |
| DeepSeek-GRM-27B + Voting@32 | 88.5 | 65.3 | 60.4 | 69.7 | 71.0 |
| DeepSeek-GRM-27B + MetaRM Voting@32 | 90.4 | 67.2 | 63.2 | 70.3 | 72.8 |

**Key observations**:
1. Scalar RM (BTRM) 在 PPE Correctness 上很强 (66.7) — verifiable domain 优势明显，但在 RMB 上崩了 (57.9)。这印证了 paper 反复强调的 "**scalar RM 有 severe domain bias**"。
2. GRM-27B 单点 69.9 已经接近 GPT-4o (71.3)，加上 MetaRM voting 后 72.8 超过所有 public model。
3. **Domain bias 问题在 GRM 上被解决了** — GRM 在 4 个 benchmark 上都接近最佳，没有显著短板。

### 6.2 Inference-Time Scaling Gains (Table 3)

| Method | Voting@1 | Voting@8 | Δ@8 | Voting@32 | Δ@32 |
|---|---|---|---|---|---|
| LLM-as-a-Judge | 67.0 | 67.6 | +0.6 | - | - |
| LLM-as-a-Judge w/ TokenProb | - | 68.1 | +1.1 | - | - |
| CLoud-Gemma-2-27B | 68.5 | 68.8 | +0.3 | - | - |
| DeepSeek-GRM-27B-RFT | 67.8 | 69.3 | +1.5 | - | - |
| DeepSeek-GRM-27B | 67.9 | 70.6 | **+2.7** | 71.0 | +3.1 |
| DeepSeek-GRM-27B + MetaRM | - | 72.0 | **+4.1** | 72.8 | +4.9 |

这里非常 informative：
- **Semi-scalar (CLoud) 几乎不 scale** (+0.3) — 因为 scalar 部分方差太小，voting 时被 mean 主导。
- **Pairwise GRM (LLM-as-a-Judge) scaling 有限** (+0.6) — majority voting 没有 quantitative weight。
- **LLM-as-a-Judge + TokenProb** 把 token probability 当 scalar weight，scaling 提升到 +1.1，验证了 quantitative weight 的重要性。
- **Pointwise GRM + RFT** 已经有 +1.5，说明 format 学对了就有 scaling 潜力。
- **Pointwise GRM + RL** 跳到 +2.7 — 这是 SPCT 真正的贡献。
- **MetaRM 再加 +1.4** — 通过过滤 low-quality trajectory 进一步榨取 scaling potential。

### 6.3 Training-Time vs Inference-Time Scaling (Figure 4)

这是 paper 最 striking 的 figure：

- DeepSeek-GRM-16B greedy: ~82.9 RB
- DeepSeek-GRM-27B greedy: ~86.0 RB
- DeepSeek-GRM-230B greedy: ~85.3 RB
- DeepSeek-GRM-671B greedy: ~88.4 RB
- DeepSeek-GRM-27B + Voting@32: ~88.5 RB
- DeepSeek-GRM-27B + MetaRM Voting@8: ~89.8 RB

**27B model + 8x inference compute ≈ 671B model 的 performance**。这暗示在 RM 这个任务上，inference scaling 的 efficiency 远高于 parameter scaling。

更有意思的是 DeepSeek-R1-0120 在 downsampled 300 样本上只有 84.9 RB — R1 的 long CoT 在 general RM 上并没有显著优势（除了 Reasoning subset）。作者的解释是 long CoT 适合 verifiable reasoning，但 general RM 需要的是 diverse judgement perspectives，不是 deep single-perspective reasoning。

---

## 7. Ablation 与细节

### 7.1 Principle Generation 的影响 (Table 4)

| Setting | Overall (greedy) | Overall (Voting@8) |
|---|---|---|
| Full GRM-27B | 69.9 | 70.6 |
| w/o Principle Generation | 67.5 | 68.0 |

去掉 principle generation（直接 generate critique）掉了 2.4 个点，voting@8 仍然只到 68.0。这说明 **principle 是 inference scaling 的核心 enabler** — 没有 principle，多次采样只是在文本 surface form 上做 variation，没有引入真正的 judgement diversity。

### 7.2 General Instruction Data 的影响

| Setting | Overall |
|---|---|
| Full | 69.9 |
| w/o General Instruction Data | 63.3 |

掉了 6.6 个点，是所有 ablation 中 drop 最大的。这印证了 Cao et al. (2024) 的发现：GRM 需要 general instruction data 来维持 base capability，否则会被 RM-specific 数据 narrow 化。

### 7.3 Training Data Generalization (Table 15)

| Setting | Chat | Chat Hard | Safety | Reasoning | RB |
|---|---|---|---|---|---|
| Full | 94.1 | 78.3 | 88.0 | 83.8 | 86.0 |
| w/o MATH RM Data | 96.1 | 70.4 | 85.3 | 82.5 | 83.0 |

去掉 MATH 训练数据后，Reasoning 只降了 1.3 点（83.8→82.5），但 Chat Hard 降了 7.9 点（78.3→70.4）。这表明 **MATH preference data 学到的是 "rigorous comparison" 能力，能 transfer 到 chat domain 的 hard cases**。这是一个很有意思的 cross-domain transfer 现象。

### 7.4 Response Length Analysis (Figure 7)

RL 后 response length 变化：
- Chat subset: 几乎不变
- Safety subset: 略微下降
- **Reasoning subset: 显著增加**

这暗示 SPCT 学会了 **adaptive compute allocation** — 对 reasoning-heavy task 自动用更多 token（更多 principle），对 safety 这种 rule-based task 用更少 token。这是一个 emergent behavior，类似于 R1 的 "thinking longer on harder problems"。

---

## 8. Case Study 分析

### 8.1 Robustness Case (Table 16)

Query 要求写 behavioral neuroscience 复习材料，两个 response 都很长且都很好。DeepSeek-BTRM-27B 给出 Response 1: 0.4665 vs Response 2: 0.3209（差 0.15），但 ground truth 是 Response 2 更好。Scalar RM 在这里被 surface form features 误导了。

DeepSeek-GRM-27B 生成了 4 个 specific criteria（Depth of Analogies 30%, Comprehensiveness 30%, Clarity 20%, Engagement 20%），然后逐项分析，最终给出 $\boxed{8, 9}$，正确判断 Response 2 更好。**Principle 显式化了 judgement process，让 model 不容易被 surface features hack**。

### 8.2 Inference Scaling Case (Table 17)

一个 Tribonacci sequence 的代码题，两个 response 都有 bug，但 ground truth 是 Response 2 略好。

3 次采样结果：
- Result 1: $\boxed{8, 8}$（meta score: -15.78，最低）
- Result 2: $\boxed{9, 5}$（meta score: 未显示，但 Response 1 实际更好）
- Result 3: $\boxed{10, 7}$（meta score: 1.67，最高）

Naive voting: Response 1=20, Response 2=27 → 选 Response 2（正确）。

这个 case 揭示了 meta RM 的作用：Result 3 给了一个极端的 $\boxed{10, 7}$ 判断，meta RM 识别这是高质量 trajectory；Result 1 给了模棱两可的 $\boxed{8, 8}$，meta RM 给了最低分 -15.78。**Meta RM 在过滤掉 "lazy median" trajectory**。

### 8.3 Failure Case (Table 18)

加密货币实时价格查询，Response 1 诚实承认没有实时数据并给出方法论指导，Response 2 编造了"更新后"的实时价格。Ground truth 是 Response 2 更好（这个 ground truth 本身可能有问题）。

DeepSeek-GRM-27B 生成了 principle "Real-time Price Accuracy (30%)"，但因为模型本身无法 verify 实时价格准确性，最终给 $\boxed{7, 5}$，选了 Response 1。

这个 failure 揭示了 GRM 的局限：**当 principle 涉及 model 不具备的能力（如实时数据、复杂计算、pattern matching）时，principle 反而会 mislead**。作者在 Limitations 中也提到需要 tool incorporation 来解决这类问题。

---

## 9. Method 的 Limitations 与 Future Direction

paper 自己承认的 limitations（Appendix B）：

1. **Efficiency**: generative RM 比 scalar RM 慢一个数量级，online RL pipeline 部署困难。但 8x 并行采样的 latency 还在可接受范围。

2. **Verifiable domain 弱**: GRM 在 PPE Correctness (math/code) 上仍弱于 scalar RM。Appendix E.1.3 显示加 reference 后能到 91.6%，说明 GRM 有能力但需要 external knowledge。

3. **Process RM 潜力未开发**: pointwise GRM 理论上可以做 process reward，paper 没深入探索。

Future directions 我觉得最 interesting 的几个：

- **Tool-augmented GRM**: code interpreter 解决 pattern matching / counting / 实时数据问题
- **Two-stage decomposition**: principle 生成与 critique 生成分离，principle 可以 pre-generate 缓存
- **Inference-time co-scaling**: RM 和 policy model 同时 inference scaling，类似 Self-Consistency + verifier 的 best-of-n
- **Long-horizon reasoning + principle**: R1-style CoT 与 principle-guided judgement 结合

---

## 10. 我的 Intuition 总结

把这篇 paper 放到更大的图景中看：

1. **Inference-time scaling 是 LLM 的新维度**: Snell et al. 证明了 policy model 可以 inference scale，这篇 paper 把这个 idea extend 到 RM。Key insight 是 inference scaling 要求 model output 有足够的 variance + diversity，scalar RM 天生没这个 property，generative + principle-based 才有。

2. **SPCT 本质是 "learn to ensemble"**: 通过 online RL 让模型学会 generate diverse principle（对应不同 judgement perspective），voting 时 ensemble 这些 perspective。这和 Self-Consistency 在 math reasoning 中的成功是同一个 mechanism，只是从 answer space 移到 judgement space。

3. **Principle 是 explicit inductive bias**: Constitutional AI 用 hand-crafted principle，SPCT 让 model 学习 generate principle。这其实是一种 **meta-learning**：模型学的是 "如何 form judgement criteria" 这个 meta-level 能力，而不是某个 specific criteria。

4. **Meta RM 是 process reward 的雏形**: 虽然论文中 meta RM 只 filter trajectory，但它的训练目标（识别 principle + critique 的 correctness）和 process reward model (PRM) 几乎一样。可以想象 future 工作把 meta RM 做成真正的 PRM，per-step 打分。

5. **27B + 8x compute ≈ 671B** 这个结果意味着 RM 的 scaling law 可能与 policy model 不同。policy model 的 emergent ability 依赖 parameter count，RM 的 judgement diversity 似乎可以通过 sampling 来 substitute。这个 insight 对 RLHF pipeline 的 cost optimization 很重要。

6. **Failure mode 的启示**: Table 18 的失败 case 揭示了一个 fundamental 问题 — 当 principle 超出 model capability 时，principle 反而有害。这暗示 **principle generation 应该受到 model capability 的约束**，不能 generate 自己做不到的 judgement criteria。这可能是 future work 的一个方向：capability-aware principle generation。

最后，这篇 paper 和你在 Eureka Labs / Neural Modelling 讲过的 inference scaling 思路非常 resonance — RM 作为 RL pipeline 中"看不见的 bottleneck"，理应享有和 policy model 一样的 scaling treatment。SPCT 给出了一个具体的实现 path，虽然还不完美，但 direction 我觉得是对的。

参考相关工作可以延伸阅读：
- GenRM (Mahan et al.): https://arxiv.org/abs/2410.12832
- Self-Generated Critiques (Yu et al.): https://aclanthology.org/2025.naacl-long.573/
- Generative Verifiers (Zhang et al.): https://openreview.net/forum?id=Ccwp4tFEtE
- Inference Scaling Laws (Wu et al.): https://openreview.net/forum?id=VNckp7JEHn
- Self-Rewarding LM (Yuan et al.): https://proceedings.mlr.press/v235/yuan24d.html
- Atla Selene Mini: https://arxiv.org/abs/2501.17195
