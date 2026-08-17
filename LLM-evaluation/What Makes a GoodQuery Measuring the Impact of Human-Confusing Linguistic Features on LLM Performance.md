---
source_pdf: What Makes a GoodQuery Measuring the Impact of Human-Confusing Linguistic
  Features on LLM Performance.pdf
paper_sha256: 82bce1626b44a94e50a0cc4259c19ef611acf42dfa1d2f44477078e05c050331
processed_at: '2026-08-13T04:07:21-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：这篇paper到底在说啥

## 一句话总结

LLM hallucinate 不全是model的锅，**你问问题的方式本身就在决定model会不会胡说**。

---

## 这事儿为啥重要

Karpathy 你想啊，我们平时聊hallucination，都是说"model不行"、"training data有问题"、"decoding strategy不好"。整个field的mitigation策略也是按这个思路走的——要么在generation之后检查（SelfCheckGPT、self-consistency），要么在generation之前加context（RAG）。

但这篇paper的人提了个很朴素的问题：**如果用户问的问题本身就是一坨糊糊，model怎么可能答好？**

这个insight其实跟我们在Tesla做autopilot时遇到的问题一模一样——sensor input的质量决定下游一切。你给model一个blurry的camera frame，再牛的detector也得翻车。LLM的input就是query，query质量差，model的output自然不可靠。

但"query质量差"到底意味着什么？这篇paper用linguistics的工具给了一个可操作的答案。

---

## 17个features：把"问得好不好"拆成可测量的东西

作者从classical linguistics里搬了17个已知会confuse人类的feature。我给你分五类说：

### 1. 结构类
- **Query length**: 多长
- **Anaphora**: 有没有"it/this/that"这种指代不明的代词。比如"Tell me about it"——it是啥？
- **Clause complexity**: 有没有深度嵌套的从句。比如"If the trial succeeds, which regulators will, according to the memo that leaked, approve it first?"——嵌了三层
- **Dependency depth**: 语法dependency tree有多深
- **Parse tree height**: parse tree有多高

### 2. 场景类
- **Query type**: 是extractive、multiple choice还是abstractive
- **Query-scenario mismatch**: 比如在abstractive场景下问"extract the exact span"——场景跟query对不上
- **Presupposition**: 隐藏假设。比如"When did the CEO admit the fraud?"——预设了有fraud且CEO承认了
- **Pragmatics**: 语用学，字面意思跟实际意图不一致。比如"Can you pass the salt?"不是问能力，是请求

### 3. 词汇类
- **Rare words**: 生僻词
- **Negation**: 有没有"not/never/no"
- **Superlatives**: 最高级
- **Polysemy**: 一词多义。比如"bank"是银行还是河岸

### 4. 风格类
- **Answerability**: 这问题能不能答。比如"Will stock X crash next month?"——没法答
- **Excessive details**: 不相关细节太多。比如"In my blue notebook from Italy trip, can you define mitosis?"——notebook跟mitosis有啥关系
- **Subjectivity**: 主观判断。比如"Which phone is best?"
- **Lack of specificity**: 太宽泛。比如"Tell me about Tesla"——公司？车？股票？

### 5. 语义grounding类
- **Intention grounding**: 意图明不明确。"Summarize the article in 3 bullets"意图明确；"Java?"意图不明
- **Contextual constraints**: 有没有time/place/entity约束
- **Named entities**: 有没有人名地名
- **Domain specificity**: 是不是专业领域

这17个feature用三个工具组合提取：LLM structured output (gpt-4o) + spaCy parser + tiktoken。

---

## 怎么定义"hallucination risk"——这是方法学最clever的地方

直接用benchmark的gold answer打分？不行——model会memorize benchmark (Carlini 2021, https://arxiv.org/abs/2012.07878)。你拿SQuAD的题问gpt-4o，它可能不是在reasoning，是在reciting training data里的pattern。

作者的trick：**给每个query造6个paraphrase，看model在这6个"意思一样但说法不同"的版本上stable不稳**。

### Paraphrase生成

对原始query $q_{orig}$，用T=1.0采样生成paraphrase，instruction是"PRODUCE A SEMANTICALLY INDIFFERENT BUT LEXICALLY PERTURBED VERSION OF THE QUERY"。

保留前6个，条件是hybrid similarity要够高：

$$s(q_{orig}, q_i) = 0.6 \cdot \cos(\mathbf{e}_{bi}(q_{orig}), \mathbf{e}_{bi}(q_i)) + 0.4 \cdot \frac{1}{2}[\Pr(q_{orig}, q_i) + \Pr(q_i, q_{orig})]$$

翻译成人话：
- 60%权重给bi-encoder embedding的cosine相似度——semantic层面的相似
- 40%权重给cross-encoder的双向score——更细粒度的语义匹配
- 阈值0.85，保证paraphrase语义不变

### Hallucination打分

对每个paraphrase $q_i$，生成answer后用三个metric组合打分：

$$\hat{h}(q_i) = 0.6 \cdot s_{llm} + 0.3 \cdot s_{fuzz} + 0.1 \cdot s_{bleu}$$

- $s_{llm} \in \{0,1\}$: LLM-judge判断答案对不对（semantic，占60%）
- $s_{fuzz} \in [0,1]$: fuzzy string similarity（surface，占30%）
- $s_{bleu} \in [0,1]$: BLEU-1（lexical，占10%）

为啥这个权重？作者在100条human-labeled set上sweep了整个simplex $(w_0, w_1, w_2)$ with $w_0+w_1+w_2=1$，发现：
- $w_0$ (LLM-judge)在±0.2内ROC-AUC变化<0.5%——非常robust
- $w_1$ (fuzzy)增大迅速退出Pareto——sensitive
- $w_2=1$ (BLEU-only)最差——surface overlap不代表semantic correctness

所以选了(0.6, 0.3, 0.1)，落在Pareto plateau中间。

阈值$\hat{h}(q_i) > 0.5$算这条paraphrase hallucinated。

### Query-level triage label

6个paraphrase里hallucinated的数量决定query的risk label：
- **Safe**: 0/6 hallucinated
- **Borderline**: 1-3/6
- **Risky**: 4-6/6

这本质上是measure **model在semantically equivalent variants上的stability**。如果model真的understand了query，6个paraphrase应该都答对或都答错；如果model在投机，6个paraphrase可能3个答对3个答错——这就是Borderline signal。

这个思路很妙——把"hallucination"从binary判断变成了**稳定性问题**。

---

## 统计模型：Ordinal Logistic Regression

有了每个query的 (17维feature vector, 3-class ordinal label)，作者fit了一个proportional-odds model：

$$\log \frac{\Pr(Y_i \leq k \mid x_i, c_i)}{\Pr(Y_i > k \mid x_i, c_i)} = \tau_k - \eta_i$$

人话翻译：
- $Y_i \in \{0, 1, 2\}$: Safe=0, Borderline=1, Risky=2
- $k$: cutpoint index, $k \in \{0, 1\}$（3个class有2个cutpoint）
- $\tau_0 < \tau_1$: 要估计的threshold
- $\eta_i = x_i^\top \beta + \alpha_{d(i)} + \gamma_{s(i)}$: linear predictor
  - $x_i$: 17维binary feature
  - $\beta$: 我们关心的coefficient
  - $\alpha_{d(i)}$: dataset fixed effect（控制不同dataset的baseline difficulty）
  - $\gamma_{s(i)}$: scenario fixed effect（控制E/M/A的baseline）

Class probability：
$$p_0 = \sigma(\tau_0 - \eta_i)$$
$$p_1 = \sigma(\tau_1 - \eta_i) - \sigma(\tau_0 - \eta_i)$$
$$p_2 = 1 - \sigma(\tau_1 - \eta_i)$$

为啥用ordinal而不是multinomial？因为ordinal假设"一个feature要么整体推risk升，要么整体降"，更parsimonious，统计效力更高。如果用multinomial，每个feature要估计2个coefficient（cutpoint 0-1一个，cutpoint 1-2一个），容易overfit。

Odds Ratio $OR = e^\beta$：feature present时risk odds是absent时的OR倍。$OR > 1$是risk-increasing，$OR < 1$是protective。

---

## 实验规模

- **Model**: gpt-4o-2024-08-06, T=1.0
- **Datasets**: 13个QA dataset, 3个scenario, 16个configuration
- **Total**: **369,837** query-response pairs

| Scenario | Safe | Borderline | Risky |
|---|---|---|---|
| Extractive | 69.0% | 23.0% | 8.0% |
| Multiple Choice | 47.0% | 29.9% | 23.1% |
| Abstractive | 33.4% | 22.0% | **44.5%** |

这个baseline分布本身就说明问题——**Abstractive场景下44.5%的query是Risky**。RAG的motivation就在这里：没context，model就乱猜。

---

## 核心结果：Risk Landscape

### Table 1的coefficients（按β排序）

**Risk-increasing features (β > 0)**:

| Feature | β | OR | 直觉解释 |
|---|---|---|---|
| **Lack of Specificity** | **+0.868** | **2.382** | "Tell me about Tesla"——model不知道你想知道啥，就瞎编 |
| Clause Complexity | +0.568 | 1.764 | 深度嵌套的从句让model parse不过来 |
| Negation Usage | +0.311 | 1.364 | "not"让model容易搞反 |
| Excessive Details | +0.247 | 1.281 | 不相关细节distract model |
| Anaphora Usage | +0.214 | 1.238 | "it"指代不清，model猜错antecedent |
| Polysemous Words | +0.096 | 1.101 | "bank"是啥？model猜错sense |
| Rare Word Usage | +0.095 | 1.100 | 生僻词embedding质量差 |
| Pragmatic Features | +0.072 | 1.074 | "Can you..."是请求不是能力问题 |
| Presupposition | +0.056 | 1.058 | 隐藏假设model可能不buy in |

**Protective features (β < 0)**:

| Feature | β | OR | 直觉解释 |
|---|---|---|---|
| **Answerability** | **-1.106** | **0.331** | 能答的问题，model答对的probability高 |
| Intention Grounding | -0.168 | 0.846 | "Summarize/Compare/Extract"明确verb让model知道干啥 |
| Subjectivity | -0.168 | 0.846 | 主观题反而没那么容易hallucinate |
| Query Token Length | -0.212 | 0.809 | 长query提供更多context cue |
| Number of Clauses | -0.262 | 0.769 | 多clauses提供更多structural cue |
| Dependency Depth | -0.128 | 0.879 | Deep dependency反而是protective |
| Superlative Usage | -0.103 | 0.902 | 最高级反而protective |
| Query-Scenario Mismatch | -0.064 | 0.938 | (见下面counterintuitive部分) |

**Null**:
- Named Entities Present: β=0.009, p=0.205 (n.s.)
- Domain Specificity: β=0.003, p=0.692 (n.s.)

---

## 几个counterintuitive的finding

### Finding 1: Clause Complexity (+0.568) vs Number of Clauses (-0.262) 方向相反

这两个feature相关$\rho=0.79$，但系数方向相反。咋回事？

- **Clause Complexity**是LLM-judge标的qualitative flag——"multiple subordinate/relative/conditional clauses"，捕捉**深度嵌套**
- **Number of Clauses**是spaCy数的clause数量——捕捉**raw clause数**

Insight：**深度嵌套**的clause confuse model（嵌套越深，long-range dependency越难学）；但**浅而多**的clauses反而是protective——提供更多explicit context cues，降低hypothesis space。

这跟人类working memory literature完全一致 (Lewis et al. 2006, https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(06)00214-5)：嵌套深度比clause数量更影响理解。

### Finding 2: Length和clause count都是protective

按raw直觉，越长越复杂越容易错。但pooled model里length的β=-0.212，clause count的β=-0.262，都是protective。

可能解释：长query提供更多disambiguating context。但注意Figure 3显示——在Abstractive scenario下，length确实是risk-increasing；在Extractive下flat。Pooled model被Extractive + MC的protective信号dominate了。

这可能是Simpson's paradox——不同scenario下length effect方向不同，pooled coefficient误导。

### Finding 3: Human-confusing features对LLM影响很小

- Word Rarity: β=0.095
- Superlatives: β=-0.103 (protective!)
- Negation: β=0.311 (modest)

这些在cognitive science里是经典的人类comprehension障碍 (Kassner & Schütze 2020, https://aclanthology.org/2020.acl-main.703/)。但对gpt-4o几乎没影响。

Insight：**人类和LLM的failure modes不重合**。LLM在training data里见过海量superlative和negation，学到了robust representation。但underspecification和deep nesting这些"hypothesis space expansion"类features，是LLM training objective本身没法消除的——ambiguous input必然导致speculative completion。

这对你Karpathy来说应该很有意思——这暗示LLM的failure mode跟人类的不是同一种"confusion"。人类的working memory瓶颈是capacity；LLM的"working memory"是context window，capacity不是瓶颈，但**representation的sharpness**是瓶颈。

---

## ECDF separations：不光看point estimate

Figure 2画了每个feature Present vs Absent的predicted $P(\text{Risky})$ ECDF。Report两个metric：

- **KS distance**: 两个ECDF的最大差距
- **$\Delta$median**: median的差

Top separations (Table 5):

| Feature | KS | $\Delta$median | n_absent | n_present |
|---|---|---|---|---|
| **Answerability** | 0.72 | -0.58 | 25,280 | 343,340 |
| **Intention Grounding** | 0.66 | -0.59 | 13,576 | 355,044 |
| **Lack of Specificity** | 0.56 | +0.42 | 302,781 | 65,839 |
| Excessive Details | 0.43 | +0.30 | 361,479 | 7,141 |
| Clause Complexity | 0.40 | +0.27 | 307,849 | 60,771 |

KS=0.72意味着Answerability present和absent的ECDF最大差距72%——非常强的separation。

但注意sample imbalance：Answerability **absent**只25,280 (7%)，**present** 343,340 (93%)。这两个protective feature几乎是default——**缺它们才是异常**。

---

## Propensity overlap & IPW：从association到quasi-causal

光看coefficient有confounding问题——Lack of Specificity的query可能同时polysemous、anaphora多。怎么隔离Lack of Specificity的pure effect？

用propensity score + IPW。对每个feature $f$，fit一个logistic预测"这个query会不会present feature $f$"，用其他所有features当covariate：

$$\pi_f(z) = \Pr(T_f = 1 \mid Z_f = z) = \sigma(\phi_{0f} + z^\top \phi_f)$$

然后IPW ATE:

$$\hat{\tau}_f^{IPW} = \frac{\sum_i \frac{T_{fi}}{\hat{\pi}_{fi}} O_i}{\sum_i \frac{T_{fi}}{\hat{\pi}_{fi}}} - \frac{\sum_i \frac{1-T_{fi}}{1-\hat{\pi}_{fi}} O_i}{\sum_i \frac{1-T_{fi}}{1-\hat{\pi}_{fi}}}$$

人话：把treated group里"propensity低"的样本upweight（rare case代表性强），把control group里"propensity高"的样本upweight，比较两组weighted mean。

但IPW要求overlap——treated和control的propensity分布要有重合。如果某个feature几乎总是present或总是absent，overlap太低，IPW估计fragile。

Table 6的关键结果：

| Feature | Overlap | ATE (IPW) | 可信度 |
|---|---|---|---|
| **Lack of Specificity** | 0.808 | **+21.2pp** | **高** |
| **Clause Complexity** | 0.969 | **+10.3pp** | **高** |
| Anaphora Usage | 0.918 | +5.9pp | 中 |
| Rare Word Usage | 0.937 | +1.5pp | 中 |
| Presupposition | 0.838 | +1.6pp | 中 |
| **Answerability** | **0.338** | — | **degenerate** |
| **Intention Grounding** | **0.225** | — | **degenerate** |
| **Excessive Details** | **0.126** | — | **degenerate** |

Insight：**最强的risk-increasing features (Lack of Specificity, Clause Complexity) 有good overlap，IPW可信，可以当quasi-causal toggle**。但**最强的protective features (Answerability, Intention Grounding) overlap太低，只能当associational，不能当causal toggle**——因为缺它们本身就是rare event。

这给practical guidance很清晰的信号：
- **能改的（quasi-causal）**：加specificity、简化clause结构——能直接降risk
- **不能当toggle的（associational）**：answerability和intention grounding是strong correlate，但没法简单"加上"

---

## LODO Robustness

Leave-One-Dataset-Out，每次hold out一个dataset，看β稳不稳。

结果（Figure 5）：Answerability始终protective (-1.1 ± 0.05)，Lack of Specificity始终risk-increasing (+0.87 ± 0.05)，Clause Complexity也是。说明risk landscape不是某个dataset的artifact。

---

## Correlation Clusters (Figure 6)

17个features不是独立的，它们co-occur成"linguistic syndromes"。Hierarchical clustering发现4个主要cluster：

### Cluster 1: Syntactic Complexity
Query Token Length, Dependency Depth, Parse Tree Height, Number of Clauses (ρ最高0.79)
→ Protective

### Cluster 2: Semantic Grounding
Intention Grounding, Answerability (ρ=0.60), Contextual Constraints
→ Protective

### Cluster 3: Ambiguity
Lack of Specificity, Query-Scenario Mismatch, Polysemous Words, Pragmatic Features (Lack of Specificity与Query-Scenario Mismatch的ρ=0.38)
→ Risk-increasing

### Cluster 4: Lexical/Stylistic
Negation, Excessive Details, Subjectivity, Superlative
→ Weak individual effects

### Cluster 5: Domain
Domain Specificity, Named Entities, Presupposition (loose, ρ=0.21)
→ Mixed

这个cluster结构是paper的latent finding——**features co-occur成syndrome**。一个underspecified query很可能同时polysemous、anaphora、scenario mismatch。这意味着intervention要target整个syndrome，不能只改一个feature。

---

## 实用takeaways：三条low-effort rewrite rules

基于最强leverage features，paper Section 5.10给出actionable rules：

### Rule 1: Add disambiguating constraints
"Tell me about Tesla" → "Summarize Tesla's 2024 Q4 earnings call in 5 bullets"
加了time + scope + format约束

### Rule 2: Always state intent explicitly
"Java?" → "Compare Java the programming language with Python on concurrency"
明确verb "compare"

### Rule 3: Resolve polysemy up front
"What is the weather in Java?" → "What is the weather in Java the island?"
明确sense

Figure 10显示这些edits在**短query上最重要**——短query高度依赖explicit specification，长query自带context cue。

---

## 我的批判

### Paper自述的
- Observational不是causal
- 只English，只gpt-4o
- Detector本身是LLM，可能mislabel
- 没model feature interactions

### 我额外的

**1. Detector hallucination没量化**：用gpt-4o提取features，但gpt-4o自己也会hallucinate。一个risk-increasing feature可能被underestimate。Paper提了"measurement error attenuates magnitude"，但没量化effect有多大。

**2. Convex proxy的judge bias**：$s_{llm}$占60%权重，但gpt-4o judge gpt-4o answer可能leniency bias——systematic给false positive correct。Appendix C说ROC-AUC plateau flat，但没给absolute level。

**3. Paraphrase引入额外confounding**：T=1.0采样可能引入新features。原query没anaphora，paraphrase引入了anaphora——把feature effect混进了risk estimate。

**4. Benchmark memorization没完全消除**：Paraphrase是缓解，但model可能still memorize answer pattern。真实deployment的hallucination rate可能被underestimate。

**5. "Number of Clauses protective"的解释弱**：更plausible的解释是Simpson's paradox——length在abstractive下risk-increasing (Figure 3)，但pooled后extractive + MC的protective信号dominate。

**6. Domain Specificity的null result可能collinear**：domain跟dataset强相关，$\alpha_{d(i)}$ fixed effect可能吸收了domain variance。这点paper没讨论。

**7. Length × scenario interaction没model**：Figure 3显示length在不同scenario下方向不同，但ordinal model是additive的，没model interaction term。

---

## 这paper在大图上的位置

Karpathy，从你的视角，我会说这paper给LLM reliability加了个**input-side diagnostic维度**。我们花了大量精力在：
- Model internals (attention probing, activation steering, mechanistic interp)
- Output verification (Constitutional AI, RLHF, judge scoring)

但query本身作为hypothesis space的约束器，相对underexplored。

这paper的core claim用Bayesian语言说：**$P(y|x)$在$x$信息量少时entropy高，model必须从prior sample，prior里有factually wrong content。Query rewriting本质是prior tightening**。

这跟你的"LLM as autoregressive next-token predictor with cached KV"视角完全自洽——query决定了KV cache的初始state，bad query = bad KV = bad trajectory。

---

## Future directions我觉得有意思的

1. **Cross-model robustness**: 同一query在Claude/Gemini/Llama上risk landscape是否一致？不一致说明features通过model-specific mechanism影响hallucination。

2. **Feature interactions**: Additive linear假设太强。Lack of Specificity × Query-Scenario Mismatch可能multiplicative。用GBDT替ordinal logit。

3. **Causal rewrite studies**: RCT——把Lack of Specificity的query rewrite成specific，measure risk是否真降。这是从associational到causal的必经之路。

4. **Multilingual**: 这17个features在中文/日文/阿拉伯文上哪些transfer？Anaphora在pro-drop language上更频繁。

5. **Latent factor model**: 17个features当indicators，提取4-5个latent "linguistic syndrome" dimensions，比raw features更parsimonious。

6. **Automated rewrite model**: 训练small model，input是query + 17 features，output是rewritten query，目标minimize $P(\text{Risky})$。本质是Section 5.10的自动化。

---

## Web Links

### Tools
- spaCy: https://spacy.io/
- statsmodels: https://www.statsmodels.org/
- tiktoken: https://github.com/openai/tiktoken
- rapidfuzz: https://github.com/maxbachmann/rapidfuzz
- Sentence-BERT MS-MARCO: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2
- OpenAI Embeddings: https://platform.openai.com/docs/guides/embeddings

### Datasets
- SQuADv2: https://arxiv.org/abs/1806.03822
- TruthfulQA: https://arxiv.org/abs/2109.07958
- MMLU: https://arxiv.org/abs/2009.03300
- PIQA: https://arxiv.org/abs/1911.11641
- BoolQ: https://arxiv.org/abs/1905.10044
- OpenBookQA: https://arxiv.org/abs/1804.01424
- MathQA: https://arxiv.org/abs/1905.13319
- ARC: https://arxiv.org/abs/1803.05457
- HotpotQA: https://arxiv.org/abs/1809.09600
- TriviaQA: https://arxiv.org/abs/1705.03551
- SciQ: https://arxiv.org/abs/1707.06209
- WikiQA: https://arxiv.org/abs/1502.06025

### Key papers
- RAG: https://arxiv.org/abs/2005.11461
- Toolformer: https://arxiv.org/abs/2302.04761
- SelfCheckGPT: https://arxiv.org/abs/2303.08896
- Self-consistency: https://arxiv.org/abs/2203.11171
- Carlini extraction: https://arxiv.org/abs/2012.07878
- Blevins prompting for linguistic structure: https://arxiv.org/abs/2211.07830
- Reimers & Gurevych Sentence-BERT: https://arxiv.org/abs/1908.10084
- Papineni BLEU: https://aclanthology.org/P02-1040/
- Kassner & Schütze negation: https://aclanthology.org/2020.acl-main.703/
- Lewis et al. 2006 working memory: https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(06)00214-5
- HalluciBot: https://ojs.aaai.org/index.php/AAAI/article/view/33136
- MultiQ&A: https://arxiv.org/abs/2502.03711
- CLAM: https://arxiv.org/abs/2212.07769
- CLAMBER: https://arxiv.org/abs/2405.12063

---

## TL;DR

这篇paper做的事：
1. 把"query质量"拆成17个可测量的linguistic features
2. 用369K真实query做大规模empirical mapping
3. 发现**underspecification和deep nesting是risk-increasing；answerability和intention grounding是protective**
4. 但**human-confusing features (rare words, superlatives, negation)对LLM影响很小**——human和LLM的failure mode不重合
5. 提出三条low-effort rewrite rules：加约束、明确意图、消歧

从你的视角，这paper的takeaway是：**LLM hallucination的相当部分是query给model留了多少speculative completion空间**。Grounding紧、answerability强——hypothesis space窄；underspecified、deeply nested——hypothesis space宽，model就从prior里sample，prior里有错误内容。Query rewriting是prior tightening。

---

# What Makes a Good Query? 深度技术解析

这篇paper来自 J.P. Morgan AI Research (Watson, Cho, Ganesh, Veloso)，核心问题非常巧妙：**hallucination 通常被归咎于 model 或 decoding strategy，但 query 本身的 linguistic form 是否也在塑造 model 的失败模式？** 作者把 classical linguistics 中已知会 confuse 人类的 17 个 linguistic features 搬到 LLM 上，用 369,837 个真实 query 做大规模 empirical mapping，构建了一个 "risk landscape"。

paper: arXiv 链接 (查不到精确的，但作者系列前作在 https://arxiv.org/abs/2502.03711 和 AAAI 2025 HalluciBot https://ojs.aaai.org/index.php/AAAI/article/view/33136)

---

## 1. 核心洞察与 problem framing

绝大部分 hallucination mitigation 工作分两类：
- **Reactive (post-generation)**: self-verification, logit-based detection, SelfCheckGPT (https://arxiv.org/abs/2303.08896), self-consistency (https://arxiv.org/abs/2203.11171)
- **Proactive (pre-generation, context-side)**: RAG (https://arxiv.org/abs/2005.11461), tool use, Toolformer (https://arxiv.org/abs/2302.04761)

但 **proactive, input-side** 的视角（直接审视 query 本身）相对稀缺，只有 ambiguity detection 类工作 (CLam, https://arxiv.org/abs/2405.12063; CLAM, https://arxiv.org/abs/2212.07769)。这篇 paper 的独特之处：把 query 看成一个 feature vector，问"哪些 feature 与 hallucination 共变"。

这里 Karpathy 你应该会感兴趣——这本质上是一个 **model interpretability 的 input-side 补集**。我们平时关心 attention/activation/weights 内部表征，但 input 侧的 linguistic prior 同样决定 model 的 hypothesis space 大小。

---

## 2. Methodology 详细拆解

### 2.1 17 维 linguistic feature taxonomy

按 linguistics 经典分类组织成 5 组 (Appendix B):

| Group | Features | 经典依据 |
|---|---|---|
| Structural | Query/Context Length, Anaphoric Reference, Clause Complexity, Dependency Tree Depth, Parse Tree Height | Marton et al. 2006 (working memory); Lewis et al. 2006 (dependency) |
| Scenario-Based | Query Type, Query-Scenario Mismatch, Presupposition, Pragmatics | Van der Sandt 1992 (presupposition); Levinson 1983 (pragmatics) |
| Lexical | Word Rarity, Negation Usage, Superlatives, Polysemy | Schick & Schütze 2019 (rare words); Kassner & Schütze 2020 (negation); Haber & Poesio 2024 (polysemy) |
| Stylistic | Answerability, Excessive Details, Subjectivity, Lack of Specificity | Oraby et al. 2017 (rhetorical); Brown et al. 2020 (specificity) |
| Semantic Grounding | Intention Grounding, Contextual Constraints, Named Entity Presence, Domain Specificity | Clarke et al. 2009 (intention); Khalidi 2023 (domain) |

Feature 提取用三套工具组合：
- **LLM structured output (gpt-4o-2024-08-06)** + Pydantic schema，每个 feature 配 5-shot ICL positive/negative examples (Appendix G 给了完整 prompt templates)
- **spaCy** dependency parser (https://spacy.io/) for syntactic features
- **tiktoken** (o200k_base encoding) for token length

每个 detector 同时输出 `label ∈ {0,1}` 和 `rationale`，并且 schema 中有一个 chain-of-thought slack variable。Detector noise 被当作 classical measurement error 处理——理论上只会 attenuate magnitude，不会 flip sign (Blevins et al. 2023, https://arxiv.org/abs/2211.07830)。

### 2.2 Observed risk via semantics-preserving perturbations

这是 paper 最有意思的方法学部分。Benchmark items 会被 memorize，raw hallucination rate 会偏差 (Carlini et al. 2021, https://arxiv.org/abs/2012.07878)。所以作者对每个原始 query $q_{orig}$ 构造一个 local semantic equivalence class:

$$\mathcal{N}(q_{orig}) = \{q_1, \dots, q_m\}$$

采样策略：T=1.0 paraphrase sampling，instruction 是 "PRODUCE A SEMANTICALLY INDIFFERENT BUT LEXICALLY PERTURBED VERSION OF THE QUERY"。保留前 6 个 paraphrases，要求 hybrid similarity:

$$s(q_{orig}, q_i) = \lambda_{bi} \cdot \cos\bigl(\mathbf{e}_{bi}(q_{orig}), \mathbf{e}_{bi}(q_i)\bigr) + \lambda_{cross} \cdot \frac{1}{2}\bigl[\underbrace{\Pr(q_{orig}, q_i)}_{\text{cross-encoder } q_{orig} \to q_i} + \underbrace{\Pr(q_i, q_{orig})}_{\text{cross-encoder } q_i \to q_{orig}}\bigr]$$

变量解释：
- $\mathbf{e}_{bi}(\cdot)$: bi-encoder embedding, 用 **TEXT-EMBEDDING-3-LARGE** (3,072-d, https://platform.openai.com/docs/guides/embeddings)
- $\Pr(\cdot, \cdot)$: cross-encoder score, 用 **MS-MARCO-MINILM-L-6-V2** (https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)，Reimers & Gurevych 2019 (https://arxiv.org/abs/1908.10084)
- $\lambda_{bi} = 0.6, \lambda_{cross} = 0.4$: 60% bi-encoder + 40% 平均的双向 cross-encoder
- 阈值 $s \geq 0.85$ 才保留 paraphrase

为什么要双向 cross-encoder 取平均？因为 cross-encoder 是 asymmetric 的，$q_{orig} \to q_i$ 和 $q_i \to q_{orig}$ 的 semantic entailment 强度可能不同。取平均是个 cheap 对称化 trick。

### 2.3 Convex hallucination proxy

对每个 $q_i \in \mathcal{N}(q_{orig})$，compute:

$$\hat{h}(q_i) = w_0 \cdot s_{llm} + w_1 \cdot s_{fuzz} + w_2 \cdot s_{bleu}$$

变量解释：
- $s_{llm} \in \{0, 1\}$: **LLM-judge binary decision** (semantic match), 用 gpt-4o 当 judge，prompt 在 Appendix G
- $s_{fuzz} \in [0, 1]$: **fuzzy string similarity** (surface overlap)，用 rapidfuzz (https://github.com/maxbachmann/rapidfuzz)
- $s_{bleu} \in [0, 1]$: **BLEU-1** (lexical n-gram overlap)，Papineni et al. 2002 (https://aclanthology.org/P02-1040/)
- $(w_0, w_1, w_2) = (0.6, 0.3, 0.1)$

权重选取：在 100 条 human-labeled validation set 上 sweep $(w_0', w_1', w_2')$ simplex（约束 $w_0' + w_1' + w_2' = 1$），选 ROC-AUC 最高点。作者报告：
- $w_0$ 在 ±0.2 范围内 ROC-AUC 变化 < 0.5% (LLM-judge 是 robust 主导项)
- $w_1$ 增大会快速退出 Pareto region (fuzzy match sensitive)
- $w_2 = 1$ (BLEU-only) 是最差的，AUC 最低

阈值 $\hat{h}(q_i) > 0.5$ → 该 paraphrase 算 hallucinated。

Aggregation 到 query-level triage label:
- **Safe**: 6 个 paraphrase 中 0 个 hallucinated (0/6)
- **Borderline**: 1-3/6
- **Risky**: 4-6/6

这是一个 ordinal outcome $y_i \in \{0, 1, 2\}$，对应 SAFE < BORDERLINE < RISKY。

### 2.4 Ordinal logistic regression (Proportional-Odds Model)

这是 paper 的核心统计模型 (https://www.statsmodels.org/ 的 `OrderedModel`, BFGS 优化):

$$\log \frac{\Pr(Y_i \leq k \mid x_i, c_i)}{\Pr(Y_i > k \mid x_i, c_i)} = \tau_k - \eta_i \tag{1}$$

变量和上下标解释：
- $Y_i$: 第 $i$ 个 query 的 ordinal risk label ∈ {0=Safe, 1=Borderline, 2=Risky}
- $k \in \{0, 1\}$: cutpoint index (因为有 3 个 ordered class，所以 2 个 cutpoints)
- $\tau_0 < \tau_1$: ordered cutpoints (要估计的 threshold)
- $\eta_i = x_i^\top \beta + \alpha_{d(i)} + \gamma_{s(i)}$: linear predictor
  - $x_i \in \{0, 1\}^{17}$: 17 维 binary feature vector
  - $\beta \in \mathbb{R}^{17}$: feature coefficients (要估计的)
  - $\alpha_{d(i)}$: dataset $d(i)$ 的 fixed effect
  - $\gamma_{s(i)}$: scenario $s(i)$ 的 fixed effect

Class probabilities 推导:

$$p_0 = \sigma(\tau_0 - \eta_i)$$
$$p_1 = \sigma(\tau_1 - \eta_i) - \sigma(\tau_0 - \eta_i)$$
$$p_2 = 1 - \sigma(\tau_1 - \eta_i)$$

逻辑：$Y \leq 0$ 的 cumulative logit 用 $\tau_0$；$Y \leq 1$ 用 $\tau_1$；相减得到中间 class 的 mass。

为什么 proportional-odds？因为作者假设 features 对 log-odds 的影响在所有 cutpoint 上方向一致（一个 feature 要么整体推 risk 升，要么整体降，不会 cutpoint 0-1 升、cutpoint 1-2 降）。这是 ordinal outcome 的标准建模，比 multinomial logit 更 parsimonious，统计效力更高。

Loss: NLL + $\ell_2$ penalty $\lambda_{reg} \|\beta\|_2^2$，no explicit intercept (因为 fixed effects 已经吸收了 baseline)。

两种 specification:
- $S_\beta$ (feature-only): 只 $\beta$
- $S_{\beta, \gamma, \alpha}$ (full): $\beta$ + scenario $\gamma$ + dataset $\alpha$

### 2.5 Propensity modeling & IPW uplift

为了从 association 往 quasi-causal 靠，作者做 propensity score adjustment。对每个 binary feature $f$ (treatment $T_f \in \{0,1\}$)，fit 一个 logistic:

$$\pi_f(z) = \Pr(T_f = 1 \mid Z_f = z) = \sigma(\phi_{0f} + z^\top \phi_f)$$

其中 $Z_f$ 是除 $f$ 之外的所有 feature indicator + dataset/scenario fixed effects。

Inverse Probability Weighting (IPW) ATE:

$$\hat{\tau}_f^{IPW} = \frac{\sum_i \frac{T_{fi}}{\hat{\pi}_{fi}} O_i}{\sum_i \frac{T_{fi}}{\hat{\pi}_{fi}}} - \frac{\sum_i \frac{1-T_{fi}}{1-\hat{\pi}_{fi}} O_i}{\sum_i \frac{1-T_{fi}}{1-\hat{\pi}_{fi}}}$$

变量解释：
- $T_{fi} \in \{0,1\}$: 第 $i$ 个 query 是否 present feature $f$
- $\hat{\pi}_{fi}$: 估计的 propensity score (bounded to $[10^{-3}, 1-10^{-3}]$ 防 extreme weight)
- $O_i \in [0,1]$: outcome (risky label indicator 或 model-implied $P_i(\text{RISKY})$)

第一项是 treatment group 的 weighted mean (用 $1/\hat{\pi}$ upweight under-represented treated)；第二项是 control group 的 weighted mean。差就是 ATE under unconfoundedness assumption。

Overlap 诊断 (positivity):

$$\text{overlap}_f = \frac{1}{n} \sum_{i=1}^n \mathbf{1}\{\hat{\pi}_{fi} \in [\alpha, 1-\alpha]\}, \quad \alpha = 0.05$$

如果 overlap 太低 (mass 集中在 0 或 1)，IPW 估计会很 fragile，作者只报 associational。

---

## 3. Experimental setup

### 3.1 Model & datasets

- Model under test: **gpt-4o-2024-08-06**, temperature $\tau = 1.0$ (用于 both answering 和 paraphrase sampling)
- 13 QA datasets, 3 scenarios, 16 configurations, total **N = 369,837** query-response pairs

| Scenario | Datasets |
|---|---|
| **Extractive (E)** | SQuADv2 |
| **Multiple Choice (M)** | TruthfulQA, SciQ, MMLU, PIQA, BoolQ, OpenBookQA, MathQA, ARC-Easy, ARC-Challenge |
| **Abstractive (A)** | SQuADv2, TruthfulQA, SciQ, WikiQA, HotpotQA, TriviaQA |

注意 SQuADv2/TruthfulQA/SciQ 在多个 scenario 下都有，所以是 16 configurations。

Dataset references:
- SQuADv2: https://arxiv.org/abs/1806.03822
- TruthfulQA: https://arxiv.org/abs/2109.07958
- MMLU: https://arxiv.org/abs/2009.03300
- PIQA: https://arxiv.org/abs/1911.11641
- BoolQ: https://arxiv.org/abs/1905.10044
- OpenBookQA: https://arxiv.org/abs/1804.01424
- MathQA: https://arxiv.org/abs/1905.13319
- ARC: https://arxiv.org/abs/1803.05457
- HotpotQA: https://arxiv.org/abs/1809.09600
- TriviaQA: https://arxiv.org/abs/1705.03551
- SciQ: https://arxiv.org/abs/1707.06209
- WikiQA: https://arxiv.org/abs/1502.06025

### 3.2 Risk distribution by scenario (Table 3)

| Scenario | Safe | Borderline | Risky |
|---|---|---|---|
| Extractive | 69.0% | 23.0% | 8.0% |
| Multiple Choice | 47.0% | 29.9% | 23.1% |
| Abstractive | 33.4% | 22.0% | 44.5% |

直觉：Abstractive 没 explicit supporting context，hallucination 比例 (44.5%) 显著高于 Extractive (8%)。这跟 RAG 的 motivation 完全一致——context 缺失是 abstractive 的核心 risk factor。

### 3.3 Length × scenario interaction (Figure 3)

- **Abstractive**: risk 随 length 增加而显著上升
- **Extractive**: risk 在所有 length 都 low/flat
- **Multiple-Choice**: 居中

Karpathy 你应该觉得这个 finding 很有意思——这暗示 length 在 abstractive scenario 下是 risk amplifier，但在 extractive 下不是。可能因为 extractive 有 context 提供 grounding，length 只增加 noise 而不增加 hypothesis space。

---

## 4. 核心结果：Risk Landscape

### 4.1 Ordinal regression coefficients (Table 1, 按系数排序)

| Feature | β | SE | z | OR | 方向 |
|---|---|---|---|---|---|
| **Lack of Specificity** | **0.868** | 0.010 | 85.9 | **2.382** | risk ↑↑ |
| Clause Complexity | 0.568 | 0.010 | 57.4 | 1.764 | risk ↑↑ |
| Negation Usage | 0.311 | 0.016 | 19.5 | 1.364 | risk ↑ |
| Excessive Details | 0.247 | 0.026 | 9.7 | 1.281 | risk ↑ |
| Anaphora Usage | 0.214 | 0.009 | 23.8 | 1.238 | risk ↑ |
| Polysemous Words | 0.096 | 0.007 | 13.8 | 1.101 | risk ↑ |
| Rare Word Usage | 0.095 | 0.011 | 9.0 | 1.100 | risk ↑ |
| Pragmatic Features | 0.072 | 0.008 | 8.5 | 1.074 | risk ↑ |
| Presupposition | 0.056 | 0.010 | 5.6 | 1.058 | risk ↑ |
| Contextual Constraints | 0.044 | 0.007 | 5.8 | 1.045 | risk ↑ (但 ECDF 反方向) |
| Parse Tree Height | 0.011 | 0.005 | 2.3 | 1.011 | risk ↑ (弱) |
| Named Entities Present | 0.009 | 0.007 | 1.3 | 1.009 | n.s. (p=0.205) |
| Domain Specificity | 0.003 | 0.009 | 0.4 | 1.003 | n.s. / mixed |
| Query-Scenario Mismatch | -0.064 | 0.014 | -4.7 | 0.938 | risk ↓ (但 ECDF 反方向) |
| Superlative Usage | -0.103 | 0.012 | -8.7 | 0.902 | risk ↓ |
| Dependency Depth | -0.128 | 0.005 | -24.4 | 0.879 | risk ↓ |
| Intention Grounding | -0.168 | 0.023 | -7.3 | 0.846 | risk ↓ |
| Subjectivity | -0.168 | 0.019 | -8.9 | 0.846 | risk ↓ |
| Query Token Length | -0.212 | 0.010 | -21.0 | 0.809 | risk ↓ |
| Number of Clauses | -0.262 | 0.009 | -28.7 | 0.769 | risk ↓ |
| **Answerability** | **-1.106** | 0.017 | -63.4 | **0.331** | risk ↓↓ (最强 protective) |

所有 p < $10^{-5}$，除了 "Named Entities Present" (p=0.115)。

OR (Odds Ratio) = $\exp(\beta)$，解释：feature present 时 risk odds 是 absent 时的 OR 倍。OR=2.382 表示 Lack of Specificity present 时 odds of being in higher risk category 是 absent 时的 2.38 倍。OR=0.331 表示 Answerability present 时 odds 是 absent 时的 1/3。

### 4.2 关键 counterintuitive findings

#### Finding 1: Clause Complexity (+0.568) vs Number of Clauses (-0.262) 方向相反

这两个 feature 高度相关 ($\rho = 0.79$，Figure 6)，但系数方向相反。怎么理解？

- **Clause Complexity** 是 LLM-judge 标注的 "multiple subordinate/relative/conditional clauses" qualitative flag——捕捉 deep nesting 和 obfuscation
- **Number of Clauses** 是 spaCy 计数的 clause 数量——捕捉 raw structural richness

Hypothesis: deeply nested clauses (深度嵌套，qualitative) 会 confuse model；但 raw clause count 高（more clauses, 但 shallow）反而提供更多 explicit context cues，是 protective。这跟人类的 working memory literature 一致：嵌套深度比 clause 数量更影响理解 (Lewis et al. 2006, https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(06)00214-5)。

#### Finding 2: Query Token Length (-0.212), Dependency Depth (-0.128), Parse Tree Height (+0.011) 

Token length 是 protective，跟 raw intuition（更长 = 更复杂 = 更易错）相反。可能因为更长的 query 提供 more context / more disambiguation。这跟 Figure 3 一致——length 在 abstractive 下确实是 risk ↑，但在 extractive 下 flat，而 ordinal model 是 pool 所有 scenario 的，所以 protective 信号 dominate。

#### Finding 3: Human-confusing features 在 LLM 上不显著

- Word Rarity (β=0.095)
- Superlatives (β=-0.103, protective!)
- Complex Negation (β=0.311, modest)

这些在 cognitive science 上是经典的 human comprehension 障碍 (Marton et al. 2006; Kassner & Schütze 2020, https://aclanthology.org/2020.acl-main.703/)。但在 LLM 上影响很小。这是 paper 最 striking 的 finding：**human 和 model 的 failure modes 并不重合**。

这可能解释：LLM 训练数据里 superlatives 和 negation 的 frequency 远高于儿童语言样本，所以 model 学到了 robust representation。但 underspecification 和 deep nesting 这些 "hypothesis space expansion" 类的 features 是 LLM 训练目标本身没法消除的——model 在 ambiguous input 上必然 speculative complete。

### 4.3 ECDF separations (Figure 2, Table 5)

为了不依赖 point estimate，作者比较 Present vs Absent 的 predicted $P(\text{Risky})$ ECDF。Report KS distance 和 $\Delta\text{median} = \text{median}(P(\text{Risky})|f=1) - \text{median}(P(\text{Risky})|f=0)$。

Top separations (Table 5):

| Feature | KS | $\Delta$median | $n_{abs}$ | $n_{pres}$ | Direction |
|---|---|---|---|---|---|
| Answerability | 0.72 | -0.58 | 25,280 | 343,340 | risk ↓ |
| Intention Grounding | 0.66 | -0.59 | 13,576 | 355,044 | risk ↓ |
| Lack of Specificity | 0.56 | +0.42 | 302,781 | 65,839 | risk ↑ |
| Excessive Details | 0.43 | +0.30 | 361,479 | 7,141 | risk ↑ |
| Clause Complexity | 0.40 | +0.27 | 307,849 | 60,771 | risk ↑ |
| Query-Scenario Mismatch | 0.40 | +0.34 | 333,939 | 34,681 | risk ↑ |
| Anaphora Usage | 0.28 | +0.14 | 287,833 | 80,787 | risk ↑ |
| ... | | | | | |
| Domain Specificity | 0.07 | 0.00 | 71,623 | 296,997 | n.s. |

KS = 0.72 (Answerability) 意味着 Present 和 Absent 的 ECDF 最大差距是 0.72——非常强的 separation。$\Delta\text{median} = -0.58$ 意味着 Answerability present 的 median 预测 risk 比 absent 低 58 个百分点。

注意 sample imbalance：Answerability absent 只有 25,280 (7%)，present 343,340 (93%)。同样 Intention Grounding absent 只 13,576 (3.7%)。这两个 protective features 几乎是 default 的，缺它们才是异常。

### 4.4 Propensity overlap & IPW uplift (Table 6)

只有 overlap ≥ 0.5 左右的 features 才能 trust IPW ATE:

| Feature | Overlap | ATE (IPW) | ATE (Strat.) | 可信度 |
|---|---|---|---|---|
| Lack of Specificity | 0.808 | **+0.212** | +0.199 | 高 |
| Clause Complexity | 0.969 | **+0.103** | +0.083 | 高 |
| Anaphora Usage | 0.918 | +0.059 | +0.071 | 中 |
| Rare Word Usage | 0.937 | +0.015 | +0.009 | 中 |
| Presupposition | 0.838 | +0.016 | +0.007 | 中 |
| Superlative Usage | 0.890 | -0.032 | -0.025 | 中 |
| Domain Specificity | 0.979 | +0.001 | +0.006 | 高（但效应近零） |
| **Answerability** | **0.338** | — | — | **degenerate** |
| **Intention Grounding** | **0.225** | — | — | **degenerate** |
| **Excessive Details** | **0.126** | — | — | **degenerate** |
| Negation Usage | 0.338 | — | — | degenerate |
| Subjectivity | 0.266 | — | — | degenerate |

关键 insight：最强 protective 的 Answerability 和 Intention Grounding 反而 overlap 太低，IPW 不能用——因为这两个 feature 几乎总是 present，"absent" 的样本太少而且其他 features 上系统差异大。

所以作者分两类结论：
- **Quasi-causal (overlap-supported)**: Lack of Specificity (+21pp risk), Clause Complexity (+10pp) — 这两个可以"toggle" 来降 risk
- **Associational only (degenerate overlap)**: Answerability, Intention Grounding — 是 robust protective correlate，但不能当 causal toggle，因为缺它们本身就 rare

### 4.5 LODO robustness (Figure 5)

Leave-One-Dataset-Out refit，每个 dataset hold out 一次，看 $\beta$ 是否稳定。

报告 mean ± 1 s.d. across 13 holds。Key findings:
- Answerability 始终 protective (β around -1.1, ±0.05)
- Lack of Specificity 始终 risk-increasing (β around +0.87, ±0.05)
- Clause Complexity 始终 risk-increasing
- Query-Scenario Mismatch 始终 risk-increasing

这表明 risk landscape 不是某单一 dataset 的 artifact。

### 4.6 Correlation clusters (Figure 6)

Spearman correlation matrix with complete linkage + correlation distance 的 hierarchical clustering。四大 cluster:

1. **Syntactic Complexity cluster**: Query Token Length, Dependency Depth, Parse Tree Height, Number of Clauses (ρ up to 0.79)。这些 negatively associate with risk。

2. **Semantic Grounding cluster**: Intention Grounding, Answerability (ρ=0.60), Contextual Constraints。Protective。

3. **Ambiguity cluster**: Lack of Specificity, Query-Scenario Mismatch, Polysemous Words, Pragmatic Features (ρ=0.38 between Lack of Specificity & Query-Scenario Mismatch)。Risk-increasing。

4. **Lexical/Stylistic**: Negation, Excessive Details, Subjectivity, Superlative。Weak correlations。

5. **Domain cluster**: Domain Specificity, Named Entities, Presupposition (loose, ρ=0.21)。Mixed。

这个 cluster 结构本身是 paper 的 latent finding——**features 不是独立的，它们 co-occur 成 "linguistic syndromes"**。如果一个 query 有 Lack of Specificity，它也很可能 Query-Scenario Mismatch 和 Polysemous Words。

---

## 5. Practical takeaways: 三条 low-effort rewrite rules

paper Section 5.10 给出 actionable rules (基于最强 leverage features):

1. **Add disambiguating constraints** (time, place, entity) — 提升 Specificity
2. **Always state intent explicitly** ("summarize / compare / extract / verify") — 提升 Intention Grounding  
3. **Always resolve polysemy up front** ("Java the language vs. the island") — 消除 Polysemy

作者特别指出：**这些 edits 在 short, open-ended prompts 上最重要**，因为 Figure 10 显示 Present vs Absent 的 risk gap 在 short queries 上最大。长 query 自带 context cue，自然消歧；短 query 高度依赖 explicit specification。

---

## 6. Triage system 提议

paper 提出一个 inference-time pipeline:

1. **Detect features** (LLM structured output + spaCy + tiktoken)
2. **Compute predicted $P(\text{Risky})$** under $S_{\beta, \gamma, \alpha}$ specification
3. **Route high-risk queries** to:
   - Clarifying step (ask user for disambiguation)
   - Retrieval/tool-grounded path (RAG)
   - Human-in-the-loop review

这跟 HalluciBot (Watson et al. 2025, https://ojs.aaai.org/index.php/AAAI/article/view/33136) 的 ratiocination-rewriting-ranking-routing 框架一脉相承，但加了 linguistically-grounded feature basis。

---

## 7. Limitations & 我的批判

### 7.1 Paper 自述 limitations

- **Observational, 不是 causal**: 即使有 IPW，也是 quasi-causal at best。Answerability 这种 semantic feature 没法 "manipulate without changing meaning"
- 仅 English, 仅一类 LLM (gpt-4o-2024-08-06)
- 没考虑 multimodal input
- Feature extraction 依赖 LLM 自身 (detector 和 model-under-test 都是 GPT 系)——可能 systematic bias
- 没 model feature interactions

### 7.2 我的额外批判

**1. Detector 自身 hallucination**: 用 gpt-4o 提取 features，但 gpt-4o 自己也会 hallucinate。一个 risk-increasing feature 可能因为是 LLM-detector 的 mislabel 而被 underestimate。Paper 提了 "measurement error attenuates magnitude"，但没量化这个 effect 有多大。

**2. Convex proxy $\hat{h}$ 的 judge bias**: $s_{llm}$ 占 60% 权重，但 LLM-judge 对自己的 output 有 leniency bias (Panickssery et al. 2024 等)。所以 gpt-4o judge gpt-4o answer 可能 systematically 给 false positive correct。Paper 在 Appendix C 说 ROC-AUC plateau 是 flat，但 plateau 的 absolute level 是多少没给。

**3. Paraphrase sampling 的 confounding**: T=1.0 采样的 paraphrase 可能引入额外的 linguistic feature 变化。比如原 query 没有 anaphora，但某个 paraphrase 引入了 anaphora——这就把 feature effect 混进了 risk estimate 里。Paper 假设 paraphrase "semantically indifferent but lexically perturbed"，但实际可能引入 syntactic 变化。

**4. Benchmark memorization bias 没完全消除**: Paraphrase 是缓解，但 model 可能 still memorize answer pattern。Hallucination rate 在 paraphrase 上可能 underestimate 真实 deployment scenario。

**5. Lack of Specificity 的 OR=2.382 是否 causal**: 这是 paper 最强 result，但 Lack of Specificity 高度 correlate with Query-Scenario Mismatch (ρ=0.38) 和 Polysemous Words。IPW 调整了其他 features，但 residual confounding 仍在。

**6. Domain Specificity 的 null result**: β≈0, p=0.69。但 Table 5 显示 $\Delta\text{median}=0$ 且 KS=0.07。这可能是 domain effect 真的 null，也可能是 dataset/scenario fixed effect 吸收了 domain variance。在 13 datasets 里，domain 跟 dataset 强相关，$\alpha_{d(i)}$ fixed effect 可能 collinear with Domain Specificity。这点 paper 没讨论。

**7. "Number of Clauses" protective 的解释弱**: 作者说 "richer syntactic structure can provide helpful context"，但更 plausible 的解释可能是：longer queries 在 abstractive 下 risk 上升 (Figure 3)，但 pooled 后 extractive + MC 的 protective 信号 dominate，导致 length 和 clause count 系数都是负。这可能是 Simpson's paradox——length 的 effect 在不同 scenario 下方向不同，pooled coefficient 误导。

---

## 8. 与 related work 的 positioning

### 8.1 与 ambiguity detection 的区别

- **CLAM** (Kuhn et al. 2023, https://arxiv.org/abs/2212.07769): selective clarification for ambiguous questions
- **CLAMBER** (Zhang et al. 2024, https://arxiv.org/abs/2405.12063): benchmark for identifying ambiguous info needs

这篇 paper 的区别：不是 binary "ambiguous or not"，而是 17 维 feature vector + ordinal risk model。提供 finer-grained, actionable signal。

### 8.2 与 HalluciBot (Watson et al. 2025) 的关系

HalluciBot 是同作者前作 (https://ojs.aaai.org/index.php/AAAI/article/view/33136)，提出 "perturb queries to estimate hallucination likelihood" 的 ratiocination-rewriting-ranking-routing 框架。这篇 paper 是其后续——把 perturbation-based risk estimation 跟 linguistically-grounded feature extraction 结合，回答 "why does perturbation indicate risk" 的问题。

### 8.3 与 RAG 的关系

RAG (Lewis et al. 2021, https://arxiv.org/abs/2005.11461) 是 context-side proactive mitigation。这篇 paper 的视角是 **query-side proactive mitigation**——不增加 context，而是优化 query form 本身。两者互补：RAG 解决 "context 缺失" (abstractive scenario 的 44.5% risky)；这篇 paper 解决 "query 本身 bad form"。

---

## 9. 我对 future directions 的想法

基于 paper 的 limitation 和 finding，几个有意思的 follow-up 方向：

1. **Cross-model robustness**: 这篇只测了 gpt-4o。同一个 query 在 Claude、Gemini、Llama 上的 risk landscape 是否一致？如果不一致，说明 features 通过 model-specific mechanisms 影响 hallucination。

2. **Feature interactions**: paper 假设 additive linear predictor。但 Lack of Specificity × Query-Scenario Mismatch 可能有 multiplicative effect。加 interaction terms 或用 gradient boosting 替代 ordinal logit。

3. **Causal rewrite studies**: 做 randomized controlled study——把 Lack of Specificity 的 query rewrite 成 specific，measure risk 是否真降。这是从 associational 到 causal 的必经之路。

4. **Multilingual extension**: 这 17 个 features 在中文、日文、阿拉伯文上哪些 transfer？比如 anaphora 在 pro-drop language (中文/日文) 上更频繁。

5. **Compositional features**: 把这 17 个 features 当 latent factor model 的 indicators，提取 4-5 个 latent "linguistic syndrome" dimensions，再做 risk modeling。可能比 17 个 raw features 更 parsimonious 且 stable。

6. **Query rewriting as inference-time intervention**: 训练一个 small model，input 是 query + 17 features，output 是 rewritten query，目标是 minimize $P(\text{Risky})$。这本质上是 paper Section 5.10 的自动化版本。

---

## 10. Web links 汇总

### Tools & libraries
- spaCy: https://spacy.io/
- statsmodels: https://www.statsmodels.org/
- tiktoken: https://github.com/openai/tiktoken
- rapidfuzz: https://github.com/maxbachmann/rapidfuzz
- Sentence-BERT (MS-MARCO-MINILM-L-6-V2): https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2
- OpenAI Embeddings: https://platform.openai.com/docs/guides/embeddings

### Datasets
- SQuADv2: https://arxiv.org/abs/1806.03822
- TruthfulQA: https://arxiv.org/abs/2109.07958
- MMLU: https://arxiv.org/abs/2009.03300
- PIQA: https://arxiv.org/abs/1911.11641
- BoolQ: https://arxiv.org/abs/1905.10044
- OpenBookQA: https://arxiv.org/abs/1804.01424
- MathQA: https://arxiv.org/abs/1905.13319
- ARC: https://arxiv.org/abs/1803.05457
- HotpotQA: https://arxiv.org/abs/1809.09600
- TriviaQA: https://arxiv.org/abs/1705.03551
- SciQ: https://arxiv.org/abs/1707.06209
- WikiQA: https://arxiv.org/abs/1502.06025

### Key related papers
- RAG (Lewis et al. 2021): https://arxiv.org/abs/2005.11461
- Toolformer: https://arxiv.org/abs/2302.04761
- SelfCheckGPT: https://arxiv.org/abs/2303.08896
- Self-consistency: https://arxiv.org/abs/2203.11171
- Carlini et al. (extraction): https://arxiv.org/abs/2012.07878
- Blevins et al. (prompting for linguistic structure): https://arxiv.org/abs/2211.07830
- Reimers & Gurevych (Sentence-BERT): https://arxiv.org/abs/1908.10084
- Papineni et al. (BLEU): https://aclanthology.org/P02-1040/
- Kassner & Schütze (negation): https://aclanthology.org/2020.acl-main.703/
- Lewis et al. 2006 (dependency & working memory): https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(06)00214-5
- HalluciBot (Watson et al. 2025): https://ojs.aaai.org/index.php/AAAI/article/view/33136
- MultiQ&A (Cho & Watson 2025): https://arxiv.org/abs/2502.03711
- CLAM (Kuhn et al. 2023): https://arxiv.org/abs/2212.07769
- CLAMBER (Zhang et al. 2024): https://arxiv.org/abs/2405.12063
- PUB (pragmatics benchmark, Sravanthi et al. 2024): https://aclanthology.org/2024.findings-acl.722/
- Flowmind (Zeng et al. 2024): https://arxiv.org/abs/2404.13050
- LAW (Watson et al. 2024): https://arxiv.org/abs/2412.11063

---

## 11. 总结：这篇 paper 在研究地图上的位置

Karpathy，从你之前在 Stanford / OpenAI / Tesla 的工作角度，我会说这篇 paper 的 contribution 是给 LLM reliability 加了一个 **input-side diagnostic 维度**。我们花了大量精力在 model internals (attention probing, activation steering, mechanistic interpretability) 和 output verification (Constitutional AI, RLHF, judge-based scoring)，但 query 本身作为 hypothesis space 的约束器，相对 underexplored。

paper 的核心 claim 可以总结成一句话：**hallucination risk 的相当部分是 query 给 model 留了多少 speculative completion 的空间**。Grounding 紧、answerability 强、intent 明确——hypothesis space 窄，hallucination 少；underspecified、deeply nested、anaphora unresolved——hypothesis space 宽，hallucination 多。

这跟 Bayesian 视角下的 LLM 生成完全自洽：$P(y|x)$ 在 $x$ 信息量少时 entropy 高，model 必须从 prior sample，prior 里有 factually wrong content。Query rewriting 本质上是 prior tightening。

paper 的局限也很清楚——observational 而非 experimental，且没跨 model 验证。但作为 first large-scale mapping from linguistic features 到 hallucination，它建立了一个可复现的 baseline，后续 RCT 类工作可以直接基于这 17 个 features 做 intervention study。
