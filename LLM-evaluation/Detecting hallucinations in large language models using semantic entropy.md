---
source_pdf: Detecting hallucinations in large language models using semantic entropy.pdf
paper_sha256: 52e7ad2740b25cc7697b69e12d69ca0dde844eca9f79dfa304a79be5b0e99c1a
processed_at: '2026-08-03T20:03:19-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版本

## 一句话

问模型同一个问题 10 遍，看它每次说的"意思"是不是一回事。意思都一样 = 它真懂，意思乱七八糟 = 它在瞎编。

## 为啥要这么搞

LLM 胡说八道是个大问题。你问它一个东西，它可能答对，也可能一脸自信地编一个假答案出来。你想用的时候得知道哪些答案能信、哪些不能信。

最直觉的想法：看模型对自己说的话有多"确信"。概率高 = 有把握，概率低 = 心虚。

但这里有个坑。

你问"法国首都是哪"，模型可能回答"Paris"、"是 Paris"、"法国的首都是 Paris"。这三句话**意思完全一样**，但 token 序列不同。你要是傻乎乎去看 token 概率，会觉得"模型每次说的都不一样哎，它肯定不确定"——其实模型非常确定，只是每次换了种说法而已。

这就叫 **naive entropy 的 bug**：它把"换一种说法"和"换一个意思"搞混了。

## Semantic Entropy 的招

很简单三步：

**第一步**：同一个问题，让模型回答 10 次（temperature=1，让它有点随机性）。

**第二步**：拿一个 NLI 模型（判断句子之间逻辑关系的模型），两两比对这 10 个答案——"A 和 B 意思一样吗？" 如果双向都 entail（A 推得出 B，B 也推得出 A），就把它们归到同一类。

比如 "Paris"、"It's Paris"、"France's capital Paris" 三个会被归到一类。

**第三步**：数一下你有几个类，每个类占多少比例，在这个分布上算 entropy。

- 10 个答案全归一类 → entropy 低 → 模型**心里有数**
- 10 个答案分了 5 类，各说各的 → entropy 高 → 模型在**瞎蒙**

## 为啥这招 work

核心 insight：**confabulation 的本质是 model 在 meaning 层面没有 peak**。

模型真懂一个东西的时候，它的 knowledge 在 meaning 空间是 sharp 的——不管怎么 phrasing，说的都是同一件事。模型不懂的时候，它的 sampling 在 meaning 空间是散的——每次随机落到一个不同的 factoid 上。

Token entropy 看不到这层，因为它在 surface form 空间算的。Semantic entropy 通过 clustering 做了一次"折叠"，把所有表面不同但意思一样的 trajectory 折到同一个点，然后看这个折叠后的分布 sharp 不 sharp。

数学上这相当于一次 **Rao-Blackwellization**——在 meaning 这个 sub-σ-algebra 上做 marginalize，把无关的 syntactic noise average 掉，只留下你真正关心的 semantic signal。

## 长段落怎么办

句子级别的 QA 好办，直接 resample 10 次就行。但如果你让 GPT-4 写一段传记，你 resample 每次顺序都不同——这次先讲家庭再讲事业，下次先讲事业再讲家庭。Token 层乱，meaning 层其实没乱。

解决方案：把长段落拆成一个个 factoid（"XX 出生于 1965 年" 这种），然后针对每个 factoid 反推"这可能是哪个问题的答案"，再针对那个问题 sample 几次新答案，算 SE。

绕了一圈，但必须绕——直接 resample sentence 会把"讲什么"和"怎么讲"的 noise 混在一起。

## 它 detect 不了啥

很重要：SE 只能 detect **confabulation**——就是 model 不知道答案、随机瞎编那种。

它 detect 不了：
- Model 训练数据里就学错了东西，一直 confidently wrong
- RLHF 把模型训得 sycophantic
- 推理路径 systematic 出错

这些是 different mechanism。Symptoms 一样（都答错），但根子不同。SE 的哲学是：**detection 方法必须 match underlying mechanism**，你不能指望一个方法搞定所有 wrong answer。

## 最 punch 的一句

论文里有句话特别到位：

> LLMs are even better at "knowing what they don't know" than was argued by Kadavath—they just don't know they know what they don't know.

翻译成人话：模型其实"心里有数"自己不懂啥，只是它自己说不出来这个"心虚"。Semantic entropy 是一个外部 probe，帮它把这个 latent uncertainty readout 出来。

你不用教模型说自己不确定（P(True) 那条路线），你直接看它采样行为的 meaning 分布就行——它的 uncertainty 已经写在 sampling distribution 里了，只是需要正确的 lens 去看。

## 一句话再总结

**同一个问题问 10 遍，把意思一样的归成一堆，堆越多越乱 = 越在瞎编。就这么简单。**

---

# Semantic Entropy: 在 meaning 层面 detect confabulations

这篇 Nature 2024 论文由 Oxford 的 Yarin Gal 组（Sebastian Farquhar 主笔，现 DeepMind；Jannik Kossen；Lorenz Kuhn，现 OpenAI）写就。核心 insight 一句话：**LLM 自己 sampling 出 M 个答案，把 meaning 相同的 cluster 起来，再在 cluster 分布上算 entropy；entropy 高 = confabulation**。

论文链接：
- Nature 版本: https://doi.org/10.1038/s41586-024-07421-0
- Code (sentence-length): https://github.com/jlko/semantic_uncertainty
- Code (paragraph-length): https://github.com/jlko/long_hallucinations
- Zenodo: https://doi.org/10.5281/zenodo.10964366
- Anthropic P(IK) baseline: https://arxiv.org/abs/2207.05221
- DeBERTa-NLI entailment: https://openreview.net/forum?id=XPZIaotutsD
- LLaMA 2: https://arxiv.org/abs/2307.09288
- Mistral 7B: https://arxiv.org/abs/2310.06825

---

## 1. 为什么 naive token entropy 会 fail（核心 motivation）

LLM native 的 uncertainty 表达在 token sequence 上的 distribution $P(s|x)$。最朴素的 predictive entropy 直接在 token 序列上算：

$$
\mathrm{PE}(x) = H(Y|x) = -\sum_y P(y|x)\,\ln P(y|x)
$$

变量含义：
- $x$: 输入 prompt（context sentence）
- $Y$: output random variable，其取值 $y$ 是一个 token sequence
- $P(y|x)$: LLM 对该 sequence 的 joint probability

对 autoregressive LLM，joint log-prob 分解为：

$$
\log P(s|x) = \sum_i \log P(s_i \mid s_{<i}, x)
$$

- $s_i$: 第 $i$ 个 output token
- $s_{<i}$: 前面所有 token
- $x$: conditioning context

**Length normalization**（关键 trick）：

$$
\frac{1}{N}\sum_{i}^{N} \log P(s_i \mid s_{<i}, x)
$$

- $N$: sequence length

为什么必须 normalize？因为 joint likelihood $\prod_i P(s_i)$ 随 $N$ 指数衰减（每多一个 token，期望上都乘上一个 < 1 的概率），negative log-prob 线性增长。如果不 normalize，长句子永远看起来比短句子 entropy 大，完全淹没真实语义信号。这个 trick 在 NMT 里早有先例（Murray & Chiang 2018），但理论 justification 一直薄弱——作者也承认这点。

**核心 problem**：哪怕 LLM 在 meaning 层非常 confident（"Paris"、"It's Paris"、"France's capital Paris"），token-level entropy 仍然高——因为 expression 上有 variation。Naive entropy 把 "表达上的不确定" 与 "意思上的不确定" 混淆了。Confabulation 检测只关心后者。

---

## 2. Semantic Entropy 的形式化（这是论文的 main contribution）

### 2.1 定义 semantic equivalence class

让 token space 是 $\mathcal{T}$，长度为 $N$ 的 sequence space 是 $\mathcal{S}_N \equiv \mathcal{T}^N$。

引入一个 **semantic equivalence relation** $E(\cdot,\cdot)$：两句话 $s, s'$ satisfy $E(s, s')$ 当且仅当它们 mean the same thing（在 context $x$ 下）。

$E$ 满足 reflexive, symmetric, transitive，因此诱导出一个 equivalence class partition $\mathcal{C}$。每个 $c \in \mathcal{C}$ 是一组 meaning-identical 的 sequences。

### 2.2 Bidirectional entailment 作为 E 的 operationalization

理论上 meaning 是哲学问题（参考 Speaks, Stanford Encyclopedia 2021），论文采取 operational 视角：用 **bidirectional entailment** 实现 $E$。

- $s \models s'$ 且 $s' \models s$ ⟺ $E(s, s')$

Entailment 由 NLI classifier $\mathcal{M}$ 预测：
- DeBERTa-Large fine-tuned on MNLI（1.5B 参数，轻量）
- 或 GPT-3.5 Turbo 1106 / GPT-4 with entailment prompt

关键 subtlety：**entailment 必须 conditioned on context**。"Paris" 单独不 entail "The capital of France is Paris"，但在 question "What is the capital of France?" 下就 entail。Prompt template：

```
We are evaluating answers to the question {question}
Possible Answer 1: {text1}
Possible Answer 2: {text2}
Does Possible Answer 1 semantically entail Possible Answer 2?
Respond with entailment, contradiction, or neutral.
```

### 2.3 Bidirectional Entailment Clustering（Algorithm 1）

输入：context $x$，sampled sequences $\{s^{(1)}, \dots, s^{(M)}\}$，NLI classifier $\mathcal{M}$

```
C ← {{s^(1)}}
for m = 2..M:
    for c in C:
        s^(c) ← first sequence in c
        left  ← M(s^(c), s^(m))
        right ← M(s^(m), s^(c))
        if left=entailment and right=entailment:
            c ← c ∪ {s^(m)}
            break
    else:
        C ← C ∪ {{s^(m)}}
return C
```

利用 transitivity：只需要 check against 任意一个 representative（取 cluster 第一个），不必 pairwise 全比。

### 2.4 Cluster probability：从 token-level 求和到 meaning-level

这是公式 (2)，是论文最核心的数学 step：

$$
P(c|x) = \sum_{s \in c} P(s|x) = \sum_{s \in c} \prod_i P(s_i \mid s_{<i}, x)
$$

变量：
- $c$: 某 semantic equivalence class
- $s \in c$: 所有 meaning-identical 的 sequences
- $P(s|x)$: LLM 对 sequence 的 joint probability

直觉：在 token 空间的 sub-$\sigma$-algebra（由 $\mathcal{C}$ 诱导）上 marginalize。把所有 syntactic 变体的概率 mass 累加到同一个 meaning bucket 里。这相当于在 meaning space 上做了一次 Rao-Blackwellization（reduce variance）。

### 2.5 Semantic Entropy（main formula, eq. 3-4）

$$
\mathrm{SE}(x) = -\sum_c P(c|x) \log P(c|x) = -\sum_c \Big[ \big(\sum_{s \in c} P(s|x)\big) \log\big(\sum_{s \in c} P(s|x)\big) \Big]
$$

形式上这就是把 PE 中的 $P(y|x)$ 换成 cluster-level $P(c|x)$。Event space 从 token sequence 空间 $S$ 替换为 meaning class 空间 $\mathcal{C}$，$\mathcal{C}$ 是 $S$ 上的 sub-$\sigma$-algebra。

### 2.6 Monte Carlo estimator（eq. 5，实际用）

理论上 $\mathcal{C}$ 是无穷的，无法遍历。实际做法：

$$
\mathrm{SE}(x) \approx -\sum_{i=1}^{|\mathcal{C}|} P(C_i|x) \log P(C_i|x),
\quad P(C_i|x) = \frac{P(c_i|x)}{\sum_c P(c|x)}
$$

变量：
- $|\mathcal{C}|$: 实际 sampled 出来的 cluster 数
- $P(c_i|x)$: 由 M 个 generations 的 length-normalized joint prob 求和
- 归一化 $\sum_c P(c|x)$ 是为了避免 length-normalization 导致 sum ≠ 1 的 degeneracy

### 2.7 Discrete Semantic Entropy（GPT-4 黑盒版）

当 LLM 不 expose token probabilities（GPT-4 在论文写作时）时，用 empirical distribution：

$$
P(C_i|x) \approx \frac{1}{M}\sum_{j=1}^{M} \mathbb{I}[s^{(j)} \in C_i]
$$

- $M$: 总 sample 数
- $\mathbb{I}[\cdot]$: indicator

就是 count cluster 里成员数 / 总数。Law of large numbers 保证 $M \to \infty$ 时收敛到真分布。论文实验显示 discrete 版本性能与 full SE 接近——这意味着在 meaning level 的 signal 主导，token prob 的细节其实并不 critical。

---

## 3. Sampling protocol 细节

- Temperature = 1
- Nucleus sampling $p = 0.9$
- Top-K = 50
- $M = 10$ generations（sentence-length）
- 额外 sample 一条 $T = 0.1$ 的 "best generation" 作为 accuracy 评估对象
- FactualBio (paragraph): $M = 3$ + 原 factoid = 4 generations per question

为什么 $M=10$？Supplementary Fig. 2 做 ablation：再增加 generations 收益 marginal。

---

## 4. 实验 setup（数据集 + models + baselines）

### Datasets
| Dataset | Domain | Type |
|---|---|---|
| TriviaQA | trivia | free-form QA |
| SQuAD 1.1 | Wikipedia comprehension | free-form QA |
| BioASQ 11B | life sciences | free-form QA |
| NQ-Open | Google Search queries | open-domain QA |
| SVAMP | elementary math | word problems |
| FactualBio | biographies | paragraph-length |

每个 dataset 400 train + 400 test。**关键设计**：不给 context passage——给 context 模型 accuracy 太高，induce 不到 confabulation。

### Models
- LLaMA 2 Chat: 7B / 13B / 70B
- Falcon Instruct: 7B / 40B
- Mistral Instruct: 7B
- FactualBio 专用于 GPT-4 (v.0613)

### Baselines
1. **Naive entropy**：直接在 length-normalized joint token log-prob 上算 entropy
2. **Embedding regression**：取 LLM final hidden state，训 logistic regression 预测 correctness（inspired by Kadavath P(IK)）
3. **P(True)**：把 M 个 generations + main answer 给模型，问 "Is the possible answer true?"，取 next-token 是 'a' 的概率；用 up to 20-shot examples in-context（supervised in-context）

### Metrics
1. **AUROC**：二分类预测 answer wrong 与否
2. **Rejection accuracy at X%**：扔掉 confidence 最低 X% 后剩余的 accuracy
3. **AURAC**：rejection accuracy curve 下面积

---

## 5. 主要实验结果

### 5.1 Sentence-length results（Fig. 2）

| Method | AUROC (avg over 30 combos) |
|---|---|
| **Semantic Entropy** | **0.790** |
| Discrete SE | ~0.78 |
| Naive entropy | 0.691 |
| P(True) | 0.698 |
| Embedding regression (in-dist) | 0.687 |
| Embedding regression (OOD) | lower |

SE 跨 model families 稳定在 0.78–0.81。P(True) 随 model size 改善（70B 时更接近 SE），暗示 P(True) 依赖 model 的 self-knowledge 能力。

**OOD critical**：embedding regression 训在 dataset A，测在 dataset B 时性能崩溃。SE 是 unsupervised，无此问题——这是 deployment 场景最关心的 property。

### 5.2 FactualBio / GPT-4（Fig. 3）

21 个 Wikipedia notable individuals，150 factual claims（其中 45 错误）。

Pipeline：
1. GPT-4 生成 paragraph biography
2. GPT-4 自动 decompose 成 factoids
3. 每个 factoid → 生成 6 个 questions（2 次 prompt × 3 questions）
4. 每个 question → GPT-4 生成 3 个新 answers
5. 加上原 factoid，共 4 个 sequences per question
6. 对这 4 个做 bidirectional entailment clustering
7. 算 discrete SE
8. 对 6 个 questions 的 SE 取平均 → factoid-level uncertainty

为什么不能直接 resample sentence？因为 resampled sentence 会 drift 到不同主题（这次讲家庭，下次讲事业），这种 variation 反映的是 ordering 不确定，而不是 fact 本身的不确定——和 SE 想解决的 original problem 同构。

Discrete SE 显著优于 P(True) 和 self-check baseline，直到 reject 20% 时 P(True) 才反超（narrow edge）。

---

## 6. 表 1 的四个 case 详解（build intuition）

**Case 1**："Refineries, process chemical, power generation, mills..." 是什么 sector？
- 3 个 generations: "under the industrial sector" / "under the industrial sector of construction" / "under the heavy industrial sector"
- Token 上不同 → naive entropy 高 → 误判 confabulation
- Meaning 上 bidirectional entail → cluster 1 个 → SE 低 → 正确判 not confabulation

**Case 2**：STARD10 protein function？
- Ref answer: "lipid transfer protein"
- 3 generations 各说不同（actin cytoskeleton / meiotic recombination / lipid metabolism）
- Token 不同 + meaning 不同 → 两个方法都判 confabulation

**Case 3**：fado 是哪国国乐？
- 3 generations 几乎完全一样
- 两个方法都判 not confabulation

**Case 4**（SE fail case）：BSkyB digital service 何时 launch？
- Ref: "1 October 1998"
- Generations: "in 1998" / "on 1 October 1998"
- SE 把 "year only" 与 "specific date" 分成两个 cluster → entropy 高 → 误判 confabulation
- 但在 QA context 下，"1998" 与 "1 October 1998" 是可接受的——这是 NLI 太 sensitive 的局限

**Insight**：SE 在 NLI 不够 context-aware 时会 over-segment，导致 false positive。这个 failure mode 暴露了 semantic clustering 本身的 ambiguity。

---

## 7. 与 Kadavath P(IK) / P(True) 的关系（最值得 build intuition 的对比）

Kadavath et al. (Anthropic, 2022, https://arxiv.org/abs/2207.05221) 提出 "Language models (mostly) know what they know"：

- P(IK)：fine-tune LLM 预测自己能否答对——supervised，需要 ground truth labels
- P(True)：sample M 个答案，prompt 模型问 "is the proposed answer true?"，取 token prob——in-context supervised

本文 SE 的优势：
1. **Unsupervised**：完全不需要 ground truth labels，也不需要 in-context examples
2. **OOD robust**：P(IK)/embedding regression 需要训练，distribution shift 时崩溃
3. **Meaning-level**：P(True) 仍然在 token 层判断，没有 cluster 掉 paraphrase variation

但 P(True) 的优势：当模型本身足够 honest 时（如 GPT-4 in-domain），P(True) 不需要 sampling clustering，计算更轻。Fig. 2 显示 P(True) 随 model scale 改善——可能 future frontier model 上 P(True) 重新胜出。

论文一句金句：**"LLMs are even better at 'knowing what they don't know' than was argued by ref. 24—they just don't know they know what they don't know."** 即模型有 latent uncertainty signal，但需要外部 probe（SE）才能 readout 出来，模型自己 verbalize 不出来。

---

## 8. 与其他 hallucination detection 工作的关系

- **SelfCheckGPT** (Manakul et al. 2023, https://arxiv.org/abs/2303.08896)：black-box，用 consistency between samples 检测 hallucination。SE 是其 principled probabilistic 形式化——SelfCheckGPT 用 heuristic consistency score，SE 在 meaning cluster 上算 entropy。
- **TRUE benchmark** (Honovich et al. 2022)：NLI 评估 factual consistency，SE 把 NLI 用作 unsupervised uncertainty 工具
- **SummaC** (Laban et al. 2022)：NLI-based summarization inconsistency detection——SE 借鉴其在合适 granularity 应用 NLI 的发现
- **Variational transformer OOD detection** (Xiao, Gomez, Gal 2019)：早期把 entropy 用于 translation OOD，但是 token-level。SE 是其 semantic-level 升级。

---

## 9. Limitations 与 failure modes

1. **只 detect confabulation**——即 "arbitrary, ungrounded generation"。Detect 不了：
   - 训练数据中的 systematic bias（model confidently wrong）
   - RLHF 诱导的 sycophancy
   - 推理 systematic error
   论文明确 disclaim 这是 different mechanism，需要 separately 处理。

2. **NLI classifier 是 bottleneck**：
   - DeBERTa 在长段落上 recall 不够
   - GPT-3.5 entailment 太 sensitive（case 4 over-segmentation）
   - Context-dependent entailment 仍 unsolved

3. **Length normalization 缺理论**：empirical good，但理论上 unfounded

4. **Sentence-level OK，paragraph 需 decompose**：long-form 需要 factoid 分解 + question reconstruction pipeline，pipeline 自己引入 noise

5. **计算成本**：每个 prompt 要 sample 10 次，每次还要跑 NLI pairwise。推理成本 ×10+，deployment 不轻

6. **Cluster 数随 M 增长**：当真 distribution heavy-tail 时，归一化的 $\sum_c P(c|x)$ 会偏差大

---

## 10. 跟你的 work 的潜在连接（hypothetical）

考虑到 Andrej 在 nanoGPT / educational content / 上 Stanford CS231n 的背景：

1. **从 train dynamics 角度看 SE**：SE 是 inference-time uncertainty。一个有趣问题：训练时是否可以直接监督 model 产生 low-entropy（在 meaning 层）的 output？这相当于一种 semantic-level consistency regularization，类似 contrastive learning。

2. **Token-level entropy 与 semantic entropy 的 gap**：gap 大的小说明 model 把 capacity 浪费在 paraphrase variation 上。可能可以用作 RLHF reward 的一个 component——penalize 表面 variation 但 meaning 一致的 entropy。

3. **Mixture of experts / MoE connection**：每个 expert 可能对同一个 prompt 产生不同 surface form 但 meaning 一致。SE 提供了一种 expert disagreement metric 不被 surface form 误导。

4. **MCTS / tree search for reasoning**：在 reasoning chain 上每个 step 做 SE，作为 search 时的 value function uncertainty estimate——比单纯 token entropy 更 informative。

5. **Micrograd-style minimal implementation**：SE 的核心代码 < 100 行，可以做成 nanoGPT 的一个 inference-time 插件 module，作为教学 demo。

---

## 11. 公式小结 cheatsheet

```
Naive PE:   H(Y|x) = -Σ_y P(y|x) log P(y|x)
Joint:      log P(s|x) = Σ_i log P(s_i|s_<i,x)
Length-norm: (1/N) Σ_i log P(s_i|s_<i,x)
Cluster prob: P(c|x) = Σ_{s∈c} Π_i P(s_i|s_<i,x)
SE:         -Σ_c P(c|x) log P(c|x)
MC SE:      -Σ_i P(C_i|x) log P(C_i|x), P(C_i|x)=P(c_i|x)/Σ_c P(c|x)
Discrete SE: P(C_i|x) ≈ (1/M) Σ_j I[s^(j)∈C_i]
```

---

## 12. 总结金句（intuition）

SE 把 LLM 当成一个分布采样器，问的不是 "token 分布 entropy 多大"，而是 "meaning 分布 entropy 多大"。Confabulation 在 meaning 层是 uniform distribution（model 随便 hallucinate 不同事实），SE 高；confident answer 在 meaning 层是 delta-like（model 总是说同一件事的不同表达），SE 低。Bidirectional entailment 是把 syntactic paraphrase collapse 到同一个 meaning point 的工具。整个 pipeline 可以看作一次 **Rao-Blackwellization on the semantic σ-algebra**。

未来 frontier model 上，如果模型本身能 reliable 地 verbalize 自己的不确定（Evan & Lin "Teaching models to express uncertainty in words" 2022 路线），SE 的相对优势会缩小；但在 OOD / unseen task / 模型不一定 honest 的场景下，SE 这种 unsupervised external probe 仍是 robust baseline。

最后提醒一个 conceptual caveat：论文只 detect "confabulation"（arbitrary wrong），不能 detect "confidently wrong"——这两者 symptoms 相同（错答）但 mechanism 不同（前者是 uncertainty，后者是 bias 或 RLHF misalignment）。Detection 方法必须 match underlying mechanism，这是 SE 整套工作的一个 philosophy。
