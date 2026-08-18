---
source_pdf: BERT-as-a-Judge A Robust Alternative to Lexical Methods.pdf
paper_sha256: c90dbf77c880a89f09d7a60d8f755454464f6e5148b3567da8d9bfdcdef09ee3
processed_at: '2026-08-18T02:39:48-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版

## 问题出在哪

现在大家怎么测LLM行不行？让模型做题，然后看答案对不对。但"看答案对不对"这一步特别麻烦。

模型输出的格式五花八门。你让它输出"Final answer: C"，它可能给你整一大段reasoning然后说答案在最后，或者格式完全不对。你得先想办法把答案从一堆文字里抠出来，这步叫**parsing**。

抠出来之后还要跟标准答案比。比也对不上——模型说"2.00"，标准答案是"2$"，你说算对还是错？lexically不一样但semantically一样。

传统做法用regex做parsing，用exact match/ROUGE做比对。问题：

- **Llama-3 70B做数学题，60%以上的输出parse不出来**。你以为是模型不行，其实是你的评估工具不行
- 即使parse成功，Gemma-3做context extraction，准确率也被低估了27-31个百分点，因为ROUGE-L搞不定verbose output

所以你看到的leaderboard排名，很多是formatting artifact，不是真实能力差异。

## LLM-as-a-Judge的问题

用大模型当裁判当然更准——它理解语义。但一个14B的Qwen-3当裁判，跑一遍评估的成本跟重新跑一遍inference差不多。模型越多、benchmark越多，这个成本就爆炸了。

## 这篇paper的idea

**训一个小encoder (210M参数)，专门干一件事：给定question + candidate答案 + reference答案，判断candidate对不对。**

就这么简单。本质上是个二分类任务。

训练数据怎么来？用Nemotron-Super-v1.5（一个大模型）当teacher，给1M个(question, candidate, reference)三元组打True/False标签。然后让小encoder学这些标签。人类annotator验证了3212个样本，跟synthetic label的一致率97.5%，说明teacher打标签的质量是可靠的。

## 为什么encoder比decoder合适

这是paper最核心的intuition：

**分类任务，encoder在结构上就比decoder优。**

Decoder有causal mask——每个token只能看到前面的token，这是为了生成下一个token设计的。但判断答案对不对不需要生成，需要的是同时看question、candidate、reference三段文字然后做判断。Encoder的双向注意力天然支持这个。

而且decoder当裁判要generate "True"/"False"这几个token，是sequential的。Encoder一次forward pass直接出概率，快得多。在Apple M1 CPU上一个sample 200ms。

## 结果有多好

跟70倍大的Qwen-3 14B和Gemma-3 12B比，这个小encoder在几乎所有task上要么打平要么更好。

跟同为encoder的BLEURT比，BLEURT完全不行。因为BLEURT是通用text quality metric，没有针对answer correctness fine-tune。

跟专门为evaluation训练的JudgeLM 33B比，也吊打。说明generative judge的scaling效率很低。

Figure 3画了accuracy vs compute的Pareto frontier：encoder在最左上角，LLM judge在10B之后saturate，加reasoning tokens也没用。

## 一些有意思的发现

1. **100K训练样本就够了**（2个GPU小时），不需要海量数据
2. **去掉question，MC和math几乎不受影响**，只有context extraction掉5个点——因为context extraction真的需要question提供disambiguating信息
3. **在free-form答案上训练，在formatted答案上测，比反过来好**——说明training format越diverse越robust
4. **OOD generalization很好**：训练时没见过的model family，评估准确率几乎不降
5. **不用调threshold**：0.5就行，从0.3到0.7都stable
6. **Calibration不错**：temperature scaling τ=1.75后基本完美

## 那个accuracy correction公式

Paper里有个很漂亮的理论结果。你的评估准确率是拿synthetic label比的（$A_S$），但你真正想知道的是跟人类判断比的准确率（$A_H$）。这两个怎么关联？

假设evaluator的prediction $\hat{Y}$ 和human label $Y_H$ 在给定synthetic label $Y_S$ 的条件下独立。这个假设的核心含义是：evaluator犯错的模式跟人类犯错的模式不相关。

然后推出：

$$A_H = (2\rho - 1) \cdot A_S + (1 - \rho)$$

其中 $\rho$ 是synthetic和human label的一致率。

这个公式告诉你：当 $\rho = 0.975$（paper的实际值），$A_H = 0.95 \cdot A_S + 0.025$。即使synthetic label有2.5%的noise，对最终accuracy估计的影响也很小。

极端情况 $\rho = 0.5$（完全随机一致），$A_H = 0.5$——评估完全没信息量。

## 对实际用的人意味着什么

1. **不要再只用regex了**，它系统性低估模型能力，尤其是math task
2. **不用上大LLM judge了**，一个小encoder又快又准
3. **如果实在没compute**，用regex先跑，parse失败的fallback到BERT judge，能省5倍计算
4. **想跨format robust**，训练时用free-form答案
5. **这个方法的scope是objectively verifiable的task**，主观任务（summary质量、翻译质量）不适用

## 我的理解

这篇paper本质上在说一件事：**evaluation是一个可以distill的skill，而且encoder是比decoder更efficient的carrier。**

大模型当裁判之所以work，是因为它有semantic understanding。但semantic understanding本身可以蒸馏到一个小模型里——只要你有1M个labeled example和一个好teacher。这跟知识蒸馏的思路一脉相承，但applied到evaluation这个具体场景。

Paper最大的贡献不是BERT-as-a-Judge本身，而是它systematically揭示了当前evaluation pipeline的结构性缺陷，并用很clean的方式证明了small specialized model是sustainable的方向。

那个accuracy correction公式更是锦上添花——给了distillation-based evaluation一个theoretical foundation，让你能rigorously bound true accuracy，而不仅仅trust synthetic label。

## References

- [BERT-as-a-Judge paper](https://arxiv.org/abs/2506.05275)
- [EuroBERT](https://arxiv.org/abs/2506.05275)
- [LLM-as-a-Judge (Zheng et al.)](https://arxiv.org/abs/2306.05685)
- [JudgeLM](https://arxiv.org/abs/2310.17631)
- [BLEURT](https://aclanthology.org/2020.acl-main.704/)
- [BERTScore](https://arxiv.org/abs/1904.09675)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [Math-Verify](https://github.com/huggingface/Math-Verify)

---

# BERT-as-a-Judge: 技术深度解析

## 1. 核心问题与动机

这篇paper要解决的问题是LLM evaluation中一个很根本的矛盾。当evaluation community评估instruction-tuned models在zero-shot generative settings下的表现时，需要做两件事:

1. **Answer extraction**: 从模型输出中提取predicted answer
2. **Answer comparison**: 将predicted answer与reference answer比较

传统方法用regex做extraction(强制格式如"Final answer: X"), 然后用exact match, ROUGE-L, Math-Verify等做matching。但这有两个failure mode:

- **Parsing failures**: 模型不遵守格式约束, 比如Llama-3 70B在open-form math上parsing failure rate超过60% (Figure 2)
- **Matching errors**: 即使parse成功, lexical matching无法捕获semantic equivalence, 比如"2.00" vs "2$"

而LLM-as-a-Judge虽然robust但计算开销大。核心question: **如何在不依赖output formatting的情况下, 用低计算成本衡量模型的核心problem-solving ability?**

## 2. 方法论细节

### 2.1 实验protocol

**Tasks**: 覆盖三类benchmark
- Multiple-choice: MMLU, MMLU-Pro, TruthfulQA, ARC-Easy/Challenge, GPQA
- Context extraction: SQuAD-v2, HotpotQA, DROP, CoQA
- Open-form math: GSM8K, MATH, AsDiv, AIME 24, AIME 25

**Models**: 36个instruction-tuned models, 从135M到70B参数
- Llama-3 (1B, 3B, 8B, 70B)
- Qwen-3 (600M, 4B, 8B, 14B, 32B)
- Gemma-3 (1B, 4B, 12B, 27B)
- 以及Falcon-3, Phi-4, SmolLM-2/3, OLMo-3, Ministral-3, LFM-2, EuroLLM, Apertus等

**Generation**: zero-shot, greedy decoding, max 2048 tokens, 统一用"Final answer: [answer]"格式suffix

### 2.2 Labeling strategy

**Synthetic labeling**: 用Nemotron-Super-v1.5作为自动evaluator。输入question + candidate + reference, 输出True/False判断candidate是否correct。Greedy decoding, non-reasoning mode。

**Human validation**: 3,212个annotation from 11个human evaluator, 与synthetic label的agreement rate达97.5% (Table 12)。这个数字很关键, 因为它validate了用synthetic label作为ground-truth的合理性。

Disagreement分析(Table 13)显示:
- Interpretation ambiguity: 33%
- Verbose answer: 22%
- Reading error (human): 19%
- Reading error (synthetic): 17%
- Semantic closeness: 9%

### 2.3 BERT-as-a-Judge架构

**Core idea**: 训练一个BERT-style encoder model在labeled (question, candidate, reference) triplets上, 利用bidirectional attention做binary classification。

**Architecture choice的关键intuition**: 为什么用encoder而不是decoder?

Bidirectional attention让encoder能同时attend到question, candidate, reference的每个token, 这对于structured text classification task是ideal的。而decoder-only model由于causal mask的限制, 只能单向attend, 在matching task上效率更低。

**Training specifics**:
- Initialization: EuroBERT 210M (一个multilingual encoder, Boizard et al. 2025a)
- Training data: ~1M synthetically labeled triplets, 来自有training split的tasks (MMLU, ARC-Easy, ARC-Challenge, SQuAD-v2, HotpotQA, GSM8K, MATH)
- 数据平衡: across task categories和models
- Loss: binary cross-entropy
- Learning rate: $2 \times 10^{-5}$
- Warmup: 5% ratio
- Decay: linear schedule
- Hardware: 8 MI250x GPUs
- Batch size: 32 (effective)
- Duration: ~20 GPU hours per run, 1 epoch

**Inference**: ~200 ms/sample on Apple M1 CPU — 这与LLM-as-a-Judge形成鲜明对比

### 2.4 Baselines

- **Regex**: 用"Final answer: (.+)" pattern提取, 然后用exact match (multiple-choice), ROUGE-L (context extraction), Math-Verify (open-form math)
- **LLM-as-a-Judge**: Qwen-3 (0.6B-14B), Gemma-3 (1B-27B), JudgeLM (7B/13B/33B), 以及一个fine-tuned Qwen-3 0.6B
- **BLEURT-base/large**: encoder-based metric baseline

## 3. 关键实验结果

### 3.1 Regex的parsing failure分析 (Section 3, Figure 2)

Table 15-17提供了detailed breakdown。一些striking numbers:

**Multiple-choice** (Table 15):
- SmolLM-2 0.135B: parsing failure rate接近100% across所有MC benchmarks
- EuroLLM 1.7B: ARC-Challenge 97.6%, MMLU 94.1%
- Larger models (8B+) generally <5% failure rate

**Context extraction** (Table 16):
- SmolLM-2 0.135B: CoQA 47.8%, DROP 46.1%
- Llama-3 1B: CoQA 31.2%, DROP 53.0%, SQuAD-v2 33.9%
- Qwen-3全系列: ~0% (formatting compliance极好)

**Open-form math** (Table 17) — 最catastrophic:
- Llama-3 1B: AIME24 100%, AIME25 100%, Math 84.9%
- Llama-3 70B: AIME24 100%, AIME25 100%, Math 92.1%
- Falcon-3 1B: AIME24 83.3%, AIME25 90.0%, ASDiv 69.0%

这说明formatting compliance不仅是小模型的问题, 大模型在math reasoning task上也表现糟糕。

### 3.2 Table 1: Regex对accuracy的distortion

Table 1是paper的核心findings之一。以Llama-3 family为例:

**Multiple-choice**:
- Llama-3 8B: ∆Accuracy = -23.3% (其中parsing failure贡献-0.9%, post-parsing matching贡献-22.3%)
- Llama-3 32B: ∆Accuracy = -23.8%, rank下降17.6位

关键insight: 对于Gemma-3在context extraction上, 即使parsing failure rate接近0 (formatting compliance极好), ∆Accuracy仍然有-27%到-31%! 这是因为**verbose outputs**虽然技术符合formatting rules, 但lexical matching (ROUGE-L)无法正确评估。

这直接说明问题不只是"formatting compliance", 而是lexical matching本身的结构性缺陷。

### 3.3 Table 2: BERT-as-a-Judge vs all baselines

| Task | Regex | Qwen-3 0.6B | Qwen-3 0.6B (FT) | Qwen-3 14B | Gemma-3 12B | JudgeLM 33B | BLEURT-Large | **BERT-Judge** |
|------|-------|-------------|------------------|------------|-------------|-------------|--------------|----------------|
| ARC-Easy | 88.2 | 54.0 | 99.1 | 98.9 | 99.7 | 59.5 | 15.1 | **99.7** |
| MMLU | 88.1 | 50.2 | 94.3 | 98.6 | 98.3 | 66.6 | 38.0 | **98.5** |
| GPQA | 68.0 | 65.9 | 82.0 | 95.3 | 92.5 | 73.5 | 68.1 | **93.5** |
| CoQA | 67.0 | 75.2 | 89.8 | 80.9 | 90.0 | 31.8 | 23.5 | **88.2** |
| GSM8K | 94.4 | 71.2 | 97.2 | 99.0 | 98.6 | 44.3 | 26.4 | **98.8** |
| MATH | 73.3 | 58.7 | 90.5 | 93.6 | 94.1 | 45.5 | 39.5 | **93.7** |

几个关键观察:

1. **BERT-Judge (210M params) vs Qwen-3 14B (14B params)**: BERT-Judge在多数task上match或exceed, 这是70x的参数效率ratio
2. **vs Qwen-3 0.6B (FT)**: BERT-Judge始终better, 说明encoder架构本身在classification task上有structural advantage
3. **vs JudgeLM 33B**: JudgeLM虽然专门为evaluation fine-tuned, 但performance远inferior, 说明generative judge的scaling并不efficient
4. **vs BLEURT**: BLEURT完全失败, 说明generic text generation metric不能直接用于answer correctness assessment

### 3.4 Table 3: Out-of-domain model generalization

Table 3测试了BERT-as-a-Judge在training mixture中excluded的model family上的表现:

| Family | Size | MC-ID | MC-OOD | CE-ID | CE-OOD | Math-ID | Math-OOD |
|--------|------|-------|--------|-------|--------|---------|----------|
| Ministral-3 | 14B | 98.2 | 98.3 | 89.1 | 88.6 | 83.5 | 84.8 |
| LFM-2 | 2.6B | 97.9 | 97.8 | 91.2 | 90.9 | 94.6 | 94.4 |
| EuroLLM | 22B | 98.6 | 98.5 | 90.7 | 90.6 | 94.5 | 94.1 |
| Apertus | 70B | 98.1 | 98.0 | 89.5 | 89.5 | 97.2 | 97.1 |

ID (in-distribution) vs OOD (out-of-distribution)的差距通常<1%, 说明encoder学到的是task-general的assessment ability, 而不是model-specific patterns。

### 3.5 Figure 3: Compute-aware comparison

Figure 3是Pareto frontier分析。X轴是inference FLOPs, Y轴是assessment accuracy。BERT-as-a-Judge位于左上角 — Pareto optimal。

Key finding: LLM-as-a-Judge在~10B parameter scale后saturate, beyond that additional compute yield limited gains。而intermediate reasoning tokens (CoT)对于assessment accuracy没有帮助 — 这是一个counter-intuitive但important的发现。

## 4. 详细实验分析 (Section 5)

### 4.1 Training efficiency (Figure 4)

| Training samples | MC | Context Extraction | Math |
|------------------|-----|---------------------|------|
| 100K | ~97% | ~85% | ~93% |
| 200K | ~98% | ~87% | ~93% |
| 500K | ~98% | ~88% | ~94% |
| 1M | ~98% | ~89% | ~94% |

100K samples (2 GPU hours)就足以在MC和math上达到near-optimal performance。Context extraction需要更多data因为task更复杂(需要understand question context)。

### 4.2 Hybrid Regex + BERT-Judge (Table 4)

| Task Category | Regex only | BERT-Judge only | Regex + BERT-Judge (fallback) |
|---------------|------------|-----------------|-------------------------------|
| Multiple-Choice | 88.8 | 97.7 | 90.5 |
| Context Extraction | 73.0 | 89.2 | 75.2 |
| Open-Form Math | 87.3 | 93.9 | 89.9 |

这个hybrid approach在regex parse失败时fallback到BERT-Judge。对于parsing failure rate 20%的model, 能减少5x total compute, 是production deployment的practical compromise。

### 4.3 Question ablation (Table 5)

| Task Category | w/ Question | w/o Question |
|---------------|-------------|--------------|
| Multiple-Choice | 97.7 | 97.3 |
| Context Extraction | 89.2 | 84.2 |
| Open-Form Math | 93.9 | 93.9 |

Context extraction受影响最大(-5%), 因为question提供了disambiguating context。MC和math几乎不受影响, 因为answer本身自包含足够信息。

### 4.4 Formatting robustness (Table 6)

| Task | Test (Form.) - BERT-J (Form.) | Test (Form.) - BERT-J (Free) | Test (Free) - BERT-J (Form.) | Test (Free) - BERT-J (Free) |
|------|-------------------------------|-------------------------------|-------------------------------|-------------------------------|
| MC | 97.7 | 97.4 | 94.0 | 97.6 |
| CE | 89.2 | 85.8 | 84.3 | 91.6 |
| Math | 93.9 | 93.7 | 93.1 | 93.5 |

Key insight: **Free-to-formatted** encoder (trained on free-form, tested on formatted) > **Formatted-to-free**。这表明training on diverse formats比restrictive formatting更generalizable。

### 4.5 Calibration (Figure 6)

Temperature scaling with $\tau = 1.75$ achieves near-perfect calibration。原始probability已经reasonably calibrated, 说明encoder的confidence estimation是reliable的。

### 4.6 Human accuracy correction公式

这是paper中最elegant的theoretical contribution。定义:

- $A_H$ = accuracy w.r.t. human labels (unknown, 我们想estimate)
- $A_S$ = accuracy w.r.t. synthetic labels (observable)
- $\rho$ = agreement rate between synthetic and human labels
- $\hat{Y}$ = evaluator prediction
- $Y_H$ = human label
- $Y_S$ = synthetic label

推导:

$$A_H = P(\hat{Y} = Y_H)$$

用total probability law on event $Y_H = Y_S$:

$$A_H = P(\hat{Y} = Y_H | Y_H = Y_S) \cdot P(Y_H = Y_S) + P(\hat{Y} = Y_H | Y_H \neq Y_S) \cdot P(Y_H \neq Y_S)$$

关键assumption: $\hat{Y}$ conditionally independent of $Y_H$ given $Y_S$。这意味着evaluator的prediction只通过synthetic label与human label相关。

在这个assumption下:
- $P(\hat{Y} = Y_H | Y_H = Y_S) = P(\hat{Y} = Y_S) = A_S$
- $P(\hat{Y} = Y_H | Y_H \neq Y_S) = P(\hat{Y} \neq Y_S) = 1 - A_S$

因此:

$$A_H = A_S \cdot \rho + (1 - A_S) \cdot (1 - \rho)$$
$$= (2\rho - 1) \cdot A_S + (1 - \rho)$$

**Interpretation**:
- 当 $\rho = 0.5$ (random agreement): $A_H = 0.5$, 即evaluation完全uninformative
- 当 $\rho = 1$ (perfect agreement): $A_H = A_S$, 直接使用synthetic accuracy
- 当 $\rho = 0.975$ (paper的实际值): $A_H = 0.95 \cdot A_S + 0.025$

这意味着即使synthetic label有2.5%的noise, 对estimated accuracy的影响也很小。

## 5. Multilingual extension (Appendix D)

Table 14显示在multilingual MMLU上, English-only fine-tuning已经competitive, 轻量multilingual adaptation (20K samples)进一步提升:

| Language | Regex | Gemma-3 12B | Qwen-3 14B | BERT-J (Eng) | BERT-J (Multi) |
|----------|-------|-------------|------------|---------------|-----------------|
| ar | 89.3 | 97.3 | 97.6 | 97.2 | **97.8** |
| zh | 89.1 | 96.4 | 96.2 | 96.9 | **97.3** |
| vi | 86.4 | 94.7 | 92.5 | 95.8 | **96.1** |

这说明EuroBERT的multilingual pretraining提供了strong cross-lingual transfer。

## 6. 核心intuition总结

### 6.1 为什么encoder比decoder更efficient for this task?

1. **Bidirectional attention**: 对classification task, 能同时看到所有input tokens的representation是optimal的。Decoder的causal mask是generative necessity, 但对classification是structural limitation。

2. **No autoregressive overhead**: Decoder需要sequential token generation, cost $O(n^2)$ per token, total $O(n^3)$ for sequence length $n$。Encoder只需要single forward pass, $O(n^2)$ total。

3. **Direct classification head**: Encoder通过[CLS] token + linear head直接输出binary probability, 而decoder需要generate "True"/"False" tokens再parse, 引入额外failure mode。

### 6.2 为什么synthetic label是reliable的?

关键在于**distillation from a strong teacher**。Nemotron-Super-v1.5是large, capable model, 在reasoning mode disabled的情况下做binary judgment。Human agreement 97.5%说明:
- Task本身是well-defined的(correctness有客观标准)
- Teacher model足够强能match human judgment
- Disagreement case主要是inherent ambiguity而非systematic bias

### 6.3 为什么不要CoT for assessment?

Figure 3显示reasoning tokens (L prompting)没有improve assessment accuracy。这个counter-intuitive的finding可能因为:
- Binary correctness judgment本身是pattern-matching task, 不需要multi-step reasoning
- CoT可能引入更多variance和error propagation
- 对于semantic equivalence判断, 直接comparison比decomposed reasoning更robust

## 7. Limitations & Future work

Paper承认的限制:
1. 只cover English benchmarks with objectively verifiable answers
2. 不适用于open-ended generation (summarization, translation等)
3. 没有multimodal extension

Future directions:
- Multilingual extension (Appendix D已初步验证)
- Multimodal evaluation (VQA, image captioning)
- Open-ended generation assessment

## 8. 与related work的关系

- **vs BERTScore**: BERTScore用encoder做representation similarity, 但不fine-tune for correctness judgment。BERT-as-a-Judge是task-specific fine-tuned。
- **vs BLEURT**: BLEURT fine-tuned for text generation quality, 但generic。BERT-as-a-Judge针对reference-based correctness, 更narrow但更accurate。
- **vs COMET/MetricX**: 这些是MT-specific metrics, BERT-as-a-Judge是general-purpose correctness evaluator。
- **vs JudgeLM/Prometheus**: 这些是generative judges, 用decoder做assessment。BERT-as-a-Judge证明encoder architecture更efficient。

## 9. 对practitioner的guidance

1. **默认用BERT-as-a-Judge standalone**: 200ms/sample, accuracy 97%+, 是best accuracy-efficiency tradeoff
2. **如果compute极度constrained**: 用Regex + BERT-Judge fallback hybrid (Table 4)
3. **如果要cross-format robustness**: Train on free-form answers (Table 6)
4. **Training data需求**: 100K samples足够MC和math, context extraction需要更多 (Figure 4)
5. **Threshold**: 0.5默认即可, across wide range都stable (Figure 5)
6. **Calibration**: 原始probability已经well-calibrated, 如需perfect calibration用 $\tau=1.75$ temperature scaling

## 10. 批判性思考

**Strengths**:
- 实验设计rigorous: 36 models × 15 tasks, comprehensive baselines
- Human validation of synthetic labels (97.5% agreement)
- OOD generalization tested (Table 3)
- Compute-aware Pareto analysis (Figure 3)
- Theoretical contribution: accuracy correction formula

**Potential concerns**:
1. **Synthetic label bias**: 虽然human agreement 97.5%, 但remaining 2.5%可能systematically biased toward certain model families或answer styles
2. **Conditional independence assumption**: accuracy correction公式假设 $\hat{Y} \perp Y_H | Y_S$, 如果evaluator和human在synthetic label出错的same cases上也错, 这个assumption就violate了
3. **Task coverage**: 只cover fact-based tasks with objective correctness, 不适用于subjective tasks
4. **Distribution shift**: 如果evaluated model严重不同于training mixture中的模型(比如完全不同的alignment策略), generalization可能degrade
5. **Adversarial robustness**: paper没有测试model是否能generate adversarial outputs that fool the encoder while being incorrect

**Missing experiments**:
- Per-model-family analysis of BERT-Judge accuracy (which model families are hardest to judge?)
- Error analysis: 在哪些case types上BERT-Judge仍然fail?
- Comparison with cross-encoder architectures (e.g., DeBERTa)
- Sensitivity to reference answer quality

## References

- [EuroBERT: Scaling multilingual encoders for european languages](https://arxiv.org/abs/2506.05275) - Boizard et al. 2025a
- [Llama-Nemotron: Efficient reasoning models](https://arxiv.org/abs/2506.10904) - Bercovich et al. 2025
- [When does reasoning matter? A controlled study](https://arxiv.org/abs/2509.22193) - Boizard et al. 2025b
- [JudgeLM: Fine-tuned large language models are scalable judges](https://arxiv.org/abs/2310.17631) - Wang et al. 2023
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) - Zheng et al. 2023
- [BLEURT: Learning robust metrics for text generation](https://aclanthology.org/2020.acl-main.704/) - Sellam et al. 2020
- [BERTScore: Evaluating text generation with BERT](https://arxiv.org/abs/1904.09675) - Zhang et al. 2019
- [InfoLM: A new metric to evaluate summarization & data2text generation](https://ojs.aaai.org/index.php/AAAI/article/view/21254) - Colombo et al. 2022
- [Holistic evaluation of language models](https://arxiv.org/abs/2211.09110) - Liang et al. 2022
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) - Gao et al. 2024
- [Math-Verify](https://github.com/huggingface/Math-Verify) - Hugging Face 2024
- [Do BERT-like bidirectional models still perform better on text classification in the era of LLMs](https://arxiv.org/abs/2505.18215) - Zhang et al. 2025
- [MMLU-Pro: A more robust and challenging multi-task language understanding benchmark](https://arxiv.org/abs/2406.01574) - Wang et al. 2024
- [Prometheus: Inducing fine-grained evaluation capability in language models](https://openreview.net/forum?id=hCZukR2q4mZ) - Kim et al. 2023
- [Prometheus 2: An open source language model specialized in evaluating other language models](https://aclanthology.org/2024.emnlp-main.271/) - Kim et al. 2024
- [Generalizing verifiable instruction following](https://arxiv.org/abs/2504.05035) - Pyatkin et al. 2025
- [xCOMET: Transparent machine translation evaluation through fine-grained error detection](https://aclanthology.org/2024.tacl-1/) - Guerreiro et al. 2024

---

**Final technical intuition**: 这篇paper的core insight是, 对于well-defined correctness assessment tasks, 一个small bidirectional encoder distilled from a large generative teacher, 在accuracy上match 70x larger generative judges的同时, 实现orders of magnitude的compute savings。这challenges了当前"bigger judge is better"的prevailing assumption, 并suggests evaluation-specific small models是more sustainable direction for scalable LLM evaluation。

The accuracy correction formula $A_H = (2\rho-1)A_S + (1-\rho)$ 更是elegant的theoretical tool, 让我们能在有imperfect synthetic labels的情况下rigorously bound true human-aligned accuracy, 这是distillation-based evaluation pipelines的foundational result。
