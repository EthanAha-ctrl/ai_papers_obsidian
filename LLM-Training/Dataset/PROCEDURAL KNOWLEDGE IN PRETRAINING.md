---
source_pdf: PROCEDURAL KNOWLEDGE IN PRETRAINING.pdf
paper_sha256: b2dfdee58a5ca1c2b20bf0aa2393a816fe3256e0fb95571e755ff1d7e5305432
processed_at: '2026-08-06T06:32:55-07:00'
target_folder: LLM-Training/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用大白话讲清楚这篇 paper

## 1. 一句话概括

LLM 算数学题的时候，它**靠的是"学了方法"**，靠的不是"背了答案"。这篇 paper 用一个统计学工具（influence function）去 pretraining data 里翻了 2.5B tokens，找到证据。

> 写到这里我刻意避免用"不是 X 而是 Y"那种句式，但意思就是上面这个。Karpathy 你懂的，直接讲。

## 2. 问题出在哪

LLM benchmark 分高，但大家吵：
- 一边说 LLM 真能 reasoning（Webb et al. 2023 https://www.nature.com/articles/s41562-023-01659-w）
- 一边说 LLM 是 stochastic parrot，看见啥背啥（McCoy et al. 2023 https://arxiv.org/abs/2309.13628，Wu et al. 2024 https://aclanthology.org/2024.naacl-long.102）

中间还混着 contamination（Yang et al. 2023 https://arxiv.org/abs/2311.04850 报告 rephrased benchmark 也能 leak 进 pretraining）。所以光看 benchmark 分解决不了问题。

作者换了个角度：**别解读 weights，去解读 data**。问：当模型对一道数学题吐出 CoT 答案时，pretraining data 里哪几篇 document 在 likelihood 上"造成"了这个答案？

## 3. Influence function 是个啥东西

最朴素的想法：把某条 training data 拿掉，重训一次模型，看输出变没变。但 pretraining 是 trillions of tokens，重训一次几百万美金，玩不起。

Influence function 用 Taylor 展开绕过这个问题。公式：

$$\mathcal{T}_f(x) = -\nabla_\theta f(\theta^\star)^T \mathbf{H}^{-1} \nabla_\theta \mathcal{L}(x, \theta^\star)$$

逐字翻译：
- $x$：pretraining 里的某一条 document
- $\nabla_\theta \mathcal{L}(x, \theta^\star)$：这条 document 当年在训练时贡献的 gradient（"这条 data 想把参数往哪个方向推"）
- $\mathbf{H}^{-1}$：Hessian 的逆，可以理解成"loss landscape 在当前位置的曲率"，它把 gradient 调整成"真正能改变参数方向"的量
- $\nabla_\theta f(\theta^\star)^T$：query 的 gradient（"我这条 prompt-completion 对的 loss 想让参数往哪走"）
- 整个公式：在 H 度量下，document gradient 和 query gradient 的内积。内积大 = 它俩想推参数往同个方向 = document 对 query 有正影响

所以 influence score 就是"这条 document 对这个回答的 log-likelihood 贡献多少"。Score = 1 意思是稍微 upweight 一下这条 document，回答的 log-prob 增加 1 nat。

这东西是 Hampel 1974 (https://www.tandfonline.com/doi/abs/10.1080/01621459.1974.10482962) 老统计学方法，Koh & Liang 2017 (http://proceedings.mlr.press/v70/koh17a.html) 搬到 ML，Grosse et al. 2023 (https://arxiv.org/abs/2308.03296) 搬到 billion-scale transformer。本文用 EK-FAC 估 H，block-diagonal + approximate SVD 压内存，勉强能跑 7B 和 35B。

## 4. 实验怎么设计的

两套题，每套 40 道：

**Factual 题**（要靠 memorize）：
- "What is the tallest mountain in the world and how tall is it?" → "Mount Everest, 29,029 feet"
- 半数让模型答对、半数答错，方便看 retrieval failure

**Reasoning 题**（要靠算）：
- 7B：两步算术，如 `(7-4)*7`；斜率，如 (93,28) 和 (74,47) 两点斜率
- 35B：斜率（同 7B）；解线性方程，如 `5x + 21 = 91`
- 全部 zero-shot CoT，模型至少 80% 答对才选进来

还有两组 control 题（天才设计）：
- "The planet Zog's tallest mountain is Wirtu, 29,029 feet. What is the largest mountain on Zog?"——表面像 factual，实际只需 reading comprehension
- "The slope of the line is -22. What is the slope?"——表面像 reasoning，实际只需复读

Control 集的作用是：如果 reasoning 的 pattern 也出现在 reasoning control 上，说明 pattern 是"表面格式"导致的，跟 procedural knowledge 无关。

然后从 5M pretraining documents（2.5B tokens，每个 doc 512 tokens）里算每条 document 对每个 query 的 influence，排个序。

## 5. 核心发现

### 发现 1：同类 reasoning 题之间，top influential documents 几乎完全重合

对 5M documents 和 80 个 query 两两算 Pearson R。结果：
- 同类 reasoning 之间（比如 20 道斜率题互相）相关性高得离谱，最高 0.96，p < 4e-8
- 不同类 reasoning、factual 之间几乎不相关
- Control 集之间相关 0.05-0.38，远低于真 reasoning

**直觉**：模型在不同斜率题上激活的是同一批 documents。它学的是"算斜率的 procedure"，这 procedure 来自一批 code/数学文档。换个数字，激活的还是同一批。

### 发现 2：Reasoning 的 influence 总量比 factual 小，但分散

| | Factual | Reasoning |
|---|---|---|
| 总 influence per nat | 大 | 小 |
| Volatility（query 间方差） | 大 | 小 |

意思是：factual retrieval 时，模型重度依赖少数几篇 documents（比如 Wikipedia 那条）；reasoning 时，模型从一大批低 influence 的 documents 各借一点，每个贡献很小。35B 这个效应比 7B 更明显，说明大模型更"data efficient"——更擅长从抽象 procedural 数据上学习。

### 发现 3：答案出现在 top influential 的频率

| Model | Factual 答案在 top 500 | Reasoning 答案在 top 500 |
|---|---|---|
| 7B | 55% (22/40) | 7.4% (中间步骤答案分散在几篇) |
| 35B | 30% (12/40) | 0% (0/40) |

**直觉**：如果模型是 retrieval，factual 和 reasoning 都该在 top influential 里看到答案。Factual 完全符合；reasoning 几乎完全不符。即使把搜索范围扩大到 5M docs 的随机子集，发现 reasoning 中间步骤的答案确实存在于 pretraining data 里，但它们**不在 top influential 排名里**——模型不用它们。

### 发现 4：top influential documents 里是什么

手动找 + 用 Command R+ (https://cohere.com/blog/command-r-plus) 标注：
- 斜率题：在 16/20（7B）、20/20（35B）的 top 100 里找到计算斜率的 code 或数学公式
- 找到 7 个 JavaScript 实现斜率的 documents，13 个数学公式表达
- 这些 documents 在多个 query 的 top 100 里反复出现

让 Command R+ 标 top 500 的关系类型（Table 15 节选）：

| Keyword | Count |
|---|---|
| Other types of maths | 10787 |
| Similar arithmetic on similar numbers | 7312 |
| Code that contains arithmetic | 5035 |
| Text explaining how to calculate slope | 3911 |
| Math that calculates slope between two numbers | 2490 |
| Code that calculates slope between two numbers | 1633 |

**直觉**：模型靠的是"做类似事情的 examples" + 少量"显式展示 procedure 的 documents"。这非常像人学 skill：看一堆类似例题 + 少量规范讲解，抽象出 method。

### 发现 5：source dataset 分析

看 top-k ranking 里各 source 占比 vs training distribution 占比的 multiplier：

| Query 类型 | 7B top-50 overrepresented sources | 35B top-50 |
|---|---|---|
| Factual | Wikipedia ×5, Math & Trivia ×27 | Wikipedia ×6, Math & Trivia ×16 |
| Reasoning | StackExchange ×50, Math & Trivia ×24 | StackExchange ×62, Math & Trivia ×21 |

Code data 在 k=5000 到 k=50000 区间 multiplier ≈ 2（factual 同区间 ≈ 0.5）。ArXiv 也 overrepresented for reasoning。

**直觉**：Wikipedia/trivia 喂事实，StackExchange/code/ArXiv 喂方法。Code data 双重证据：source label 是 code（Figure 7）+ content 真的是 code（Figure 11，用 classifier 验证）。这和 Aryabumi et al. 2024 (https://arxiv.org/abs/2408.10914) "To code or not to code" 一致。

## 6. 一个隐藏彩蛋：Cross-lingual transfer

搜 factual 答案时，发现答案在非英语 documents 里出现：
- Portuguese 文档提到 Mount Everest "a montanha, de 8.848m, é a mais alta do mundo"
- French 文档提到 Brussels 是 Belgium 首都
- Spanish 也有

7B 找到 8 次 cross-lingual 命中，35B 4 次（但只是 keyword overlap 暴露的，实际可能更多）。暗示模型在 multilingual pretraining 中把 fact 编码成 language-agnostic representation。这连接到 Kotha et al. 2024 (https://openreview.net/forum?id=VrHiF2hsrm) 等 catastrophic forgetting/implicit inference 的工作。

## 7. 还有个小发现：7B vs 35B 几乎不相关

36 个共享 prompt 的 query，跨模型 influence score 相关性几乎 0（平均 R=0.02，最高 0.19，对"What is the capital of Belgium?"，两个模型 completion 一模一样但相关仍只有 0.19）。

**直觉**：35B 不是"7B + 更多 data"。它从 qualitatively 不同的 documents 抽知识。呼应 Grosse et al. 2023 的 finding：larger models rely on more abstractly related documents。

## 8. 这个方法的可信度怎么验证

Influence function 是估 likelihood 的，但作者关心的是 accuracy。Appendix A.1 做了 elegant 验证：

在 GPT-2 124M + Wikitext-2 上：用三种方法（random / TracIn / EK-FAC IF）选 k 个 documents 删掉，重训模型 5 次，看 perplexity 变化。

| k → | 50 | 100 | 200 | 300 |
|---|---|---|---|---|
| Random | 22.09 | 22.12 | 22.20 | 22.15 |
| TracIn | 22.16 | 22.22 | 22.35 | 22.45 |
| IF | 22.49 | 22.66 | 22.88 | 23.05 |

删 IF 选的 top-k influential，perplexity 上升远超 random 和 TracIn。删 bottom influential，perplexity 反而下降。

在 Command 7B + DROP（free generation）和 RACE（multiple choice）上重复：删 IF 选的 docs，accuracy 下降显著超过 random，多数情况超过 TracIn。

DROP:

| k → | 500 | 1000 | 1500 | 2000 |
|---|---|---|---|---|
| Random | 0.61 | 0.60 | 0.56 | 0.57 |
| TracIn | 0.55 | 0.49 | 0.44 | 0.43 |
| IF | 0.51 | 0.50 | 0.40 | 0.38 |

这证明：influence on log-likelihood 是 accuracy change 的有效 proxy。整篇 paper 后面的论断都建在这个 proxy 上。

## 9. 局限性

1. **只看 5M documents**：trillions of tokens 没全看。反方可以说"reasoning 靠的 documents 太稀少，5M sample 都没 surface 到真正 influential 的"。作者反驳：①qualitative 显示 influential docs 直觉相关；②同类 reasoning correlation 显著；③control set 没这 pattern。但这个反驳对复杂 reasoning 就弱了。

2. **只看 MLP**：attention 的 EK-FAC 没定义好，attention 内部 dense layer 仍考虑。作者明确说这是关键 future direction，因为 reasoning 一部分发生在 attention heads（Olsson et al. 2022 https://transformercircuits.pub/2022/in-context-learning-and-induction-heads/index.html）。本文可能低估了 attention 的角色。

3. **Accuracy 是离散的**：IF 估的是单 doc 删除效应，删多个的累积效应没建模。所以 counterfactual re-training 是"crude but necessary"。

4. **35B 有噪声**：个别 query 的 top-1 document 完全不相关（一个 lunar eclipse 文档排在 slope query 的 top 1）。对应 Barshan et al. 2020 (https://proceedings.mlr.press/v108/barshan20a.html)、Choe et al. 2024 (https://arxiv.org/abs/2405.13954) 报告的"high gradient norm unrelated document"现象。作者用 re-rank by gradient norm 缓解。

## 10. 一个能 build intuition 的比喻

把 LLM 想成一个学生：

- **Factual retrieval** = 学生背了百科全书。问"世界最高峰"，脑子里翻到 Mount Everest 那一页。Influence function 一查，发现"Mount Everest, 29,029 feet"这句话在 pretraining 里出现 N 次（Wikipedia、trivia、各种语言版本），这些 copies 都高度 influential。

- **Reasoning** = 学生学过算斜率的公式 $m = (y_2 - y_1) / (x_2 - x_1)$，做过一堆例题（code 里、StackExchange Q&A 里、数学课本里）。问"(93,28) 和 (74,47) 的斜率"，学生套公式算出来。Influence function 一查，发现**答案 "-1" 不在任何 top document 里**，但一大批"算其他两点斜率的 code/examples"高度 influential。学生不是在 retrieval 答案，是在 retrieval method。

这就解释了为什么同类 reasoning 题的 influence ranking 高度重合（R=0.96）：学生每次都激活同一批"算斜率的 method 文档"，只是套到不同数字上。

## 11. 对未来的含义

### Pretraining data selection
- Factual：要 dense coverage，模型得见一个 fact 多次才记住
- Reasoning：要 diverse procedural examples，10 个不同 slope 的 code 实现可能比 100 份同一实现更有效

### Code data 真有用
但什么 code？本文暗示"实现 procedure 的 code"。开放问题：是算法 code 还是业务 code？是 Python 还是 JS？（本文找到的 slope code 是 JavaScript）

### StackExchange 类 Q&A 数据 overrepresented for reasoning
multiplier 50+，暗示"step-by-step worked examples" format 有效。

### Cross-domain procedural transfer 仍未验证
本文只在 math reasoning 内看到 procedural transfer。Code 是否能 cross-domain transfer（数学 → 逻辑 → planning）是开放问题。

### Mechanistic + data-level interpretability 结合
SAE（Bricken et al. 2023 https://transformercircuits.pub/2023/monosemantic-features/index.html, Templeton et al. 2024 https://transformercircuits.pub/2024/scaling-monosemanticity/index.html）找"内部 features"，influence function 找"外部 documents 形成那些 features"。两端结合能完整讲 mechanistic + data-level 故事。

## 12. 联想到的 follow-up

- EK-FAC 扩展到 attention layers
- Influence function on RLHF/preference data
- 什么样的 code data 是 positively influential vs negatively influential？
- Cross-lingual procedural transfer：procedure template 是否 language-agnostic？
- Influence function 做 active learning loop 的 data curation
- In-context learning（few-shot）中 influence pattern 是否不同？
- SAE 拆出来的 monosemantic features 能否对齐到 influential documents？
- Procedural vs factual 的 catastrophic forgetting 特性是否不同？
- O1/RFT 类 RL on reasoning trace 系统：base model 必须先有 procedural substrate，RL 才能 amplify。本文是 base model 层的研究。

## 13. 最后

这篇 paper 最妙的地方在于它**绕开了"LLM 是否真理解"的形而上学辩论**，用一个非常具体的统计学问题替代：模型答这道题时，哪些 documents 在 likelihood 上 caused 它？

答：factual retrieval 靠少数 high-influence copies；reasoning 靠一大批 low-influence 但 procedural-similar 的 documents（code、StackExchange、ArXiv）。一个 document 对多个同类 reasoning 有相似 influence，因为它们激活同一个 procedure template。

这把 LLM reasoning 从"stochastic parrot" vs "genuine understanding"的两极拉到一个中间地带：**procedural synthesis over abstracted templates**。比 retrieval 强，比真 reasoning 弱，但机制清楚。和人类学 skill 的过程惊人地像——我们也是从一堆类似例子 + 少量规范讲解中抽象出 method，然后套到新数字上。

Key references:
- Grosse et al. 2023, IF for LLMs: https://arxiv.org/abs/2308.03296
- Koh & Liang 2017, original IF for ML: http://proceedings.mlr.press/v70/koh17a.html
- kronfluence: https://github.com/pomonam/kronfluence
- Wei et al. 2022, CoT: https://openreview.net/forum?id=_VjQlMeSB_J
- Olsson et al. 2022, Induction Heads: https://transformercircuits.pub/2022/in-context-learning-and-induction-heads/index.html
- Templeton et al. 2024, Scaling Monosemanticity: https://transformercircuits.pub/2024/scaling-monosemanticity/index.html
- Aryabumi et al. 2024, Code in pretraining: https://arxiv.org/abs/2408.10914
- Command R: https://cohere.com/blog/command-r
- Cohere 35B on HF: https://huggingface.co/CohereForAI/c4ai-command-r-plus

希望这版"人话版"能让你 grab 到核心 intuition：**LLM reasoning = procedural synthesis over templates learned from code/math/StackExchange documents**。Influence function 是显微镜，procedural knowledge 是观察对象，attention 层补完另一半故事是 future work。

---

# PROCEDURAL KNOWLEDGE IN PRETRAINING DRIVES REASONING IN LARGE LANGUAGE MODELS 深度讲解

## 1. 研究问题的根本张力

这篇 paper 抓住了一个非常具体、非常 deep 的张力：LLM benchmark saturate 了，但大家对它们是否真的 "reasoning" 存在根本分歧。两派证据打架：
- **Versatile reasoning 派**：Webb et al. 2023 (analogical reasoning, https://www.nature.com/articles/s41562-023-01659-w), McLeish et al. 2024 (https://arxiv.org/abs/2405.17399) 报告 transformer 在 arithmetic 上能做 robust generalization。
- **Brittle reasoning 派**：Razeghi et al. 2022 (https://aclanthology.org/2022.findings-emnlp.59) 报告 numerical reasoning 强依赖 pretraining term frequency；McCoy et al. 2023 (https://arxiv.org/abs/2309.13628) 的 "embers of autoregression"；Ullman 2023 (https://arxiv.org/abs/2302.08399) 的 ToM trivial alterations；Wu et al. 2024 (https://aclanthology.org/2024.naacl-long.102) 的 counterfactual tasks；Mahowald et al. 2024 (https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(24)00026-8) 的 "language vs thought" dissociation。

中间夹着 data contamination 的 dirty secret：Brown et al. 2020、Touvron et al. 2023 (https://arxiv.org/abs/2307.09288)、Yang et al. 2023 (https://arxiv.org/abs/2311.04850) 都报告 benchmark leakage 严重，并且 rephrased samples 也能 leak through n-gram 检测。

论文想 answer 的核心问题是：**"how do LLMs learn to reason from pretraining data?"** 它不解释 weights（像 mechanistic interpretability 那样），而是去解释 data：当 LLM 产出一个 reasoning trace 时，pretraining data 里哪些 document 在 likelihood 上"caused"它？

## 2. Influence Functions 的数学骨架

### 2.1 Counterfactual 的精确表述

定义 pretrained model 的参数 θ^u（u 表示不一定 converged），条件分布 p_{θ^u}(y_c | y_p)，其中 y_c = {y_1, ..., y_m} 是 completion，y_p = {y_1, ..., y_n} 是 prompt。Pretraining set D = {x_i}_{i=1}^N。

理想 counterfactual：从 D 删掉 x_j，retrain model，比较参数。Intractable。Influence function 通过 **response function** 做 Taylor expansion 近似。

定义 upweighted training objective:

$$\theta^\star(\epsilon) = \arg\min_{\theta \in \mathbb{R}^D} \mathcal{I}(\theta, \mathcal{D}, \epsilon) = \arg\min_{\theta \in \mathbb{R}^D} \frac{1}{N} \sum_{i \neq j} \mathcal{L}(x_i, \theta) + \epsilon \mathcal{L}(x_j, \theta)$$

变量含义：
- $\theta^\star(\epsilon)$：当 document $x_j$ 被以权重 $\epsilon$ upweight 后的最优参数。
- $\epsilon$：document 的 upweighting 系数，$\epsilon = 0$ 对应原始训练。
- $\mathcal{L}(x_i, \theta)$：典型情况是 per-token cross-entropy。
- 求和跳过 $i \neq j$：把 $x_j$ 从平均项里分离，单独用 $\epsilon$ 调节。

### 2.2 从 response function 到 influence

对 $\theta^\star(\epsilon)$ 在 $\epsilon = 0$ 处做 first-order Taylor，并使用 **implicit function theorem** 求导数（因为 $\theta^\star$ 是 argmin，stationary 条件 $\nabla_\theta \mathcal{I} = 0$）。结果：

$$\mathcal{T}_{\theta^\star}(x) = \left. \frac{d\theta^\star}{d\epsilon} \right|_{\epsilon=0} = -\mathbf{H}^{-1} \nabla_\theta \mathcal{L}(x, \theta^\star)$$

变量含义：
- $\mathcal{T}_{\theta^\star}(x)$：document $x$ 对参数空间的影响方向（一个 D 维向量，D 是参数数）。
- $\mathbf{H} = \nabla_\theta^2 \mathcal{I}(\theta^\star, \mathcal{D})$：objective 在最优点的 Hessian，对 billions 参数 intractable。
- $\nabla_\theta \mathcal{L}(x, \theta^\star)$：document $x$ 的训练梯度。
- 负号来自 stationarity 条件求导：$\nabla_\theta^2 \mathcal{I} \cdot \frac{d\theta^\star}{d\epsilon} + \nabla_\theta \mathcal{L}(x_j, \theta^\star) = 0$。

直觉：在 loss landscape 的"曲率" $\mathbf{H}$ 下，document $x$ 的训练梯度方向 $\nabla_\theta \mathcal{L}$ 被 $\mathbf{H}^{-1}$ "rescale"，得到它对参数的真正推动方向。$\mathbf{H}^{-1}$ 的作用是承认高曲率方向上的小梯度也会带来大参数变化。

### 2.3 Chain rule 到 completion 的 influence

我们其实关心 document $x$ 对某个 query 的 loss $f(\theta)$ 的影响（$f$ 是 prompt-completion 对的 cross-entropy）。由 chain rule：

$$\mathcal{T}_f(x) = -\nabla_\theta f(\theta^\star)^T \mathbf{H}^{-1} \nabla_\theta \mathcal{L}(x, \theta^\star) \quad \text{(Equation 1)}$$

变量含义：
- $\mathcal{T}_f(x)$：document $x$ 对 query loss $f$ 的 influence 标量。
- $\nabla_\theta f(\theta^\star)^T$：query 的梯度，转置后做内积。
- $\mathbf{H}^{-1}$：把 document 的"参数空间方向"投到 query 关心的方向。
- 结果是标量：document 对 query log-likelihood 的边际影响。

直觉构建：这是一个二次型 $\langle g_{query}, H^{-1} g_{doc} \rangle$。可以看成在 "natural geometry"（H 度量）下，query gradient 与 doc gradient 的内积。Grosse et al. 2023 (https://arxiv.org/abs/2308.03296) 把它推广到 transformer 的 MLP 层。

### 2.4 Normalization by information content

跨不同长度 completion 比较时，作者 normalize by：

$$\mathbb{I}(y_c) = -\log p_{\theta^u}(y_c | y_p)$$

即 completion 的自信息（nats）。Influence score 的单位变成 "log-prob 增量 / nat of query information"。Score = 1 意味着如果 upweight 这个 document 一点点，sequence log-prob 增加 1 nat。这样不同 completion 长度的 score 可比较。

## 3. EK-FAC 估计与工程化实现

### 3.1 为什么不能用真 Hessian

7B 模型 D ≈ 7×10^9，Hessian 是 D×D 矩阵 ≈ 5×10^18 entries，存不下也求不出逆。**EK-FAC** (Eigenbasis Kronecker-Factored Approximate Curvature, George et al. 2018, https://proceedings.neurips.cc/paper_files/paper/2018/file/48000647b315f6f00f913caa75a70b3-Paper.pdf) 用两个低维 expectation 估计替代：

$$\mathbb{E}_{p_\pm}[\Delta\theta \Delta\theta^T] \quad \text{和} \quad \mathbb{E}_{p_\theta}[\mathbf{A}\mathbf{A}^T]$$

变量含义：
- $\mathbb{E}_{p_\pm}[\Delta\theta \Delta\theta^T]$：参数梯度的外积期望，$p_\pm$ 是模型分布的 +/- 采样。
- $\mathbb{E}_{p_\theta}[\mathbf{A}\mathbf{A}^T]$：activations 的外积期望，$\mathbf{A}$ 是 model 的 activation。
- 两个 expectation 在 Kronecker eigenbasis 下分解，把 D×D 的问题降到几个小矩阵的特征分解。

### 3.2 关键 approximation 假设清单

paper 在 Appendix A.7 列出所有近似：

1. **First-order Taylor to PBRF**：response function 只展开到一阶。
2. **Layer independence**：MLP 各层独立，Gauss-Newton Hessian block-diagonal。
3. **Activations 独立于 pre-activation pseudo-gradients**：EK-FAC 的核心独立假设。
4. **MC estimation of Fisher**：用经验样本估 Fisher Information Matrix。
5. **Block-diagonal eigenvector matrices**：7B 用 2 blocks，35B 用 4 blocks（这是作者对 Grosse et al. 的额外近似）。
6. **Low-rank query gradient**：用 approximate SVD (Halko et al. 2011, https://doi.org/10.1137/090771806) 压缩 query gradient，因为内存里只能存几个 model-size 的 tensor。
7. **SFT stage EK-FAC 视为 identity** (Bae et al. 2024, https://arxiv.org/abs/2405.12186)：只考虑 pretraining 的影响，忽略 fine-tuning。
8. **只看 MLP 参数**：attention layer 的 EK-FAC 没有定义好，attention 内部 dense layer 仍考虑。

### 3.3 Approximation 误差验证（Table 7）

在 GPT-2 small 124M 上比较：

| Approximation | Pearson R vs full |
|---|---|
| Full SVD | 0.96 ± 0.01 |
| Approximate SVD | 0.96 ± 0.01 |
| Approx SVD + 2-block EK-FAC | 0.95 ± 0.00 |
| Approx SVD + 4-block EK-FAC | 0.93 ± 0.00 |

重要观察：approximate SVD 几乎无损，block-diagonal 略有损失但仍然 > 0.9。和 public 的 kronfluence 实现 (https://github.com/pomonam/kronfluence) 对比 Pearson R = 0.993 ± 0.003。

### 3.4 Pipeline（Figure 6）

完整流程：100k documents 估 EK-FAC → 对 80 queries 计算 query gradient（SVD 压缩存储）→ loop 5M documents 一次，计算每个的 doc gradient 与 query gradient 的 influence score → 得到 ranking。

计算复杂度：每个 query-document pair 2 次 forward + 2 次 backward。一个 query 在 5M docs 上 ranking 的复杂度 ≈ 一次 pretraining。这就是为什么只看 80 个 query。

## 4. Counterfactual Re-training 实验（Appendix A.1）的妙处

为了让读者相信 influence score 真的对应 causal effect on accuracy（不只是 log-likelihood），作者做了非常 elegant 的实验。

### 4.1 Wikitext-2 + GPT-2 124M 的 perplexity 验证（Figure 4, Table 3）

Fine-tune GPT-2 3 epochs on Wikitext-2。用三种方法选 k 个 documents 删掉：random / TracIn (Pruthi et al. 2020, https://proceedings.neurips.cc/paper_files/paper/2020/file/e6385d39ec9394f2f3a354d9d2b88eec-Paper.pdf) / EK-FAC IF。Re-train 5 次（不同 seed），比较 validation perplexity。

| k → | 50 | 100 | 200 | 300 |
|---|---|---|---|---|
| Random | 22.09 | 22.12 | 22.20 | 22.15 |
| TracIn | 22.16 | 22.22 | 22.35 | 22.45 |
| IF (ours) | 22.49 | 22.66 | 22.88 | 23.05 |

删除 IF 选的 top-k influential docs，perplexity 上升远超 random 和 TracIn。同样删 bottom influential（最负影响），perplexity 显著下降（Table 4：random 27.40, TracIn 26.73, IF 25.96 at k=50）。

### 4.2 DROP / RACE accuracy 验证（Figure 5, Tables 5-6）

更难也更 relevant：fine-tune Command 7B 在 DROP (https://aclanthology.org/N19-1246) 8k 例、RACE (https://aclanthology.org/D17-1082) 10k 例，1 epoch，看删 influential docs 后 accuracy 下降多少。

DROP (free generation):

| k → | 500 | 1000 | 1500 | 2000 |
|---|---|---|---|---|
| Random | 0.61 | 0.60 | 0.56 | 0.57 |
| TracIn | 0.55 | 0.49 | 0.44 | 0.43 |
| IF | 0.51 | 0.50 | 0.40 | 0.38 |

RACE (multiple choice):

| k → | 1000 | 1500 | 2000 | 2500 |
|---|---|---|---|---|
| Random | 0.85 | 0.83 | 0.82 | 0.81 |
| TracIn | 0.84 | 0.78 | 0.80 | 0.79 |
| IF | 0.80 | 0.76 | 0.74 | 0.74 |

IF 一致优于 random，多数情况优于 TracIn。证明：influence on log-likelihood 是 accuracy change 的有效 proxy。这一点对整篇 paper 至关重要——后面所有"reasoning 不靠 retrieval"的论断都建立在这个 proxy 之上。

## 5. Query Set 与 Document Set 的精心设计

### 5.1 Reasoning queries（40 个）

三类，每类选 model 在 zero-shot CoT 下 ≥80% accuracy 的：

- **Two-step arithmetic**（7B，如 Table 1: `Calculate the answer: (7-4)*7 Think step-by-step.`）
- **Slope between two points**（7B 和 35B 共用，20 题，如 Table 8: `(93,28)` 和 `(74,47)`）
- **Linear equation solving**（35B only，20 题，如 Table 9: `5x + 21 = 91`，限制 x > 0）

为什么这些任务？它们 procedural、多步、答案不直接出现在 prompt 里。CoT (Wei et al. 2022, https://openreview.net/forum?id=_VjQlMeSB_J) 强制展开中间步骤，便于追踪每个 reasoning step 是否对应 pretraining 的 procedural 文档。

### 5.2 Factual queries（40 个）

设计上保证 model 答对一半、答错一半（看 retrieval failure 模式）。7B 和 35B 有 16 题重叠。例子（Table 2）：`What is the tallest mountain in the world and how tall is it?` → "Mount Everest, 29,029 feet"。这部分是用来对比 reasoning 的"retrieval baseline"。

### 5.3 Control queries（共 40 个）

天才设计：每组 10 个 "factual control" 和 10 个 "reasoning control"。它们**表面**类似但不需要 retrieval/reasoning。

- Factual control（Table 10）："The planet Zog's tallest mountain is Wirtu, 29,029 feet. What is the largest mountain on Zog?"——只需 reading comprehension。
- Reasoning control（Table 12）："The slope of the line is -22. What is the slope?"——只需复读。

控制集的作用是 falsify 假设：如果 reasoning 的 correlation pattern 也出现在 reasoning control 上，说明 correlation 是由 superficial format 驱动的，而不是 procedural knowledge。

### 5.4 Document set

5M documents ≈ 2.5B tokens，每 document 512 tokens。**采样策略很讲究**：按 pretraining 时见到的 batch 顺序，从每个 batch 随机采 6 例。这样既保持 distribution 一致，又保持时序一致。EK-FAC estimation 用 100k documents，7B 和 35B 用同一组。

### 5.5 模型选择

Cohere Command R 7B 和 35B (https://cohere.com/blog/command-r) base + SFT 版本。Base 用来估 EK-FAC 和 doc gradient，SFT 用来生成 completion 和计算 query gradient。假设 SFT 的 EK-FAC 是 identity（Bae et al. 2024 的 finding：SFT 主要 enhance 已有能力，不引入新机制；Prakash et al. 2024 https://openreview.net/forum?id=8sKcAWOf2D 也支持）。

## 6. 核心发现深度剖析

### Finding 1: Reasoning 内部的 influence correlation（Figure 1, 12, 13）

**计算**：对 5M documents 和 80 queries 的每对计算 Pearson R，得 80² = 6400 个相关系数。Figure 12 是 100×100 完整 matrix（含 control）。

**结果**：
- 同类 reasoning queries 之间相关系数显著为正（p < 4e-8）。Slope（35B）最高，可达 0.96。
- 不同类 reasoning、factual-factual、factual-reasoning 之间几乎不相关。
- Control queries 之间相关 0.05-0.38，远低于真 reasoning。

**质化案例**（Tables 22-23, 25-27）：
- Table 22 (35B slopes, R=0.89)：两个 query 数字不同（94,62 vs 90,20）但 CoT 步骤格式完全一致，influence 几乎完全重合。
- Table 23 (R=0.55)：相同答案 -22，但 completion format 有差异（少几个 newline），correlation 下降。
- Table 25 (R=0.35)：format 完全不同，positive score correlation 只有 0.2，但 negative score correlation 0.44。

**直觉**：procedural knowledge（CoT 步骤、运算顺序、公式表达）由一组共享 documents 提供；具体数字的 influence 主要来自 attention layer（paper 没看 attention），所以"数字部分"的 influence 在 query 间几乎独立。Format similarity 额外推动 correlation。**这意味着 LLM 学到的是"procedure template"，被不同的数字实例化。**

### Finding 2: Magnitude & volatility（Figure 2, 14-19）

**Metric**：对 ranking 的前 k percentile，sum 这些 documents 的 influence score，画 cumulative curve。

**观察**：
1. Reasoning 的总 influence（per nat）整体低于 factual。
2. Factual 的 magnitude **波动大**：对一些 factual query，influence 集中在少数 docs；对另一些，几乎没集中。
3. 35B 比 7B 更明显。
4. 即使只看 7B 和 35B 共享的 16 个 factual + 20 个 slope query（Figure 14），35B 的 volatility 仍然更大，说明这是 model size 效应。

**直觉**：
- Factual retrieval 需要看到"那个 fact"足够多次才能 memorize，所以影响集中在特定 document（多份 Wikipedia / trivia 副本）。
- Reasoning 通过 procedural abstraction，从一大批"做类似运算"的 documents 各借一点点，每个都贡献很小。
- Volatility 低 = 更"一般化"的依赖；volatility 高 = 更"特异"的依赖。
- 35B 更"data efficient"——它更能在 procedural 文档上抽象。

**Power law fit（Figure 20, Table 28）**：

| Model/Query | α (slope in log-log) |
|---|---|
| 7B Reasoning (correct) | -0.36 ± 0.03 ⋆ |
| 7B Factual | -0.34 ± 0.03 |
| 35B Reasoning (correct) | -0.36 ± 0.04 ⋆⋆ |
| 35B Factual | -0.32 ± 0.05 |

35B reasoning 的 power law 稍陡，意味着 top ranking 包含更大比例的 total positive influence。但作者谨慎指出 noise 风险：35B 个别 query 的 top-1 document 完全不相关（一个 lunar eclipse 的 document 排在 slope query 的 top 1），这对应 Barshan et al. 2020 (https://proceedings.mlr.press/v108/barshan20a.html) 和 Choe et al. 2024 (https://arxiv.org/abs/2405.13954) 报告的 "high gradient norm unrelated document" 现象。

### Finding 3: Answer 在 top docs 中的出现率（Figure 3）

**Protocol**：对每个 query 在 top 500 (top 0.01%) documents 中搜答案。Factual 用手工 keywords（如 Mount Everest → "tallest", "highest", "Mount Everest", "29029", "8848"）。Reasoning 用程序生成的 keyword 列表（见 Appendix A.4 的 sample，包含各种运算符号变体、单词变体、format 变体）。还用 Command R+ (https://cohere.com/blog/command-r-plus) 作为 LLM judge 复核。

**结果**：

| Model | Factual 答案出现率 | Reasoning 答案出现率 |
|---|---|---|
| 7B | 55% (22/40) | 7.4% (2/27 reasoning queries, 分散在不同 docs) |
| 35B | 30% (12/40) | 0% (0/40) |

进一步：在 5M docs 的更广子集中找 reasoning step 答案，发现 13/20 arithmetic queries 的中间步骤答案确实存在于 pretraining，但**它们不在 top influential 排名里**。

**直觉**：如果 LLM 是 retrieval-based，factual 和 reasoning 都应该在 top influential 中看到答案。Factual 符合；reasoning 几乎完全不符。即使中间步骤的答案存在于 pretraining data，模型也不用它们——它用 procedural documents。

### Finding 4: Procedural knowledge documents（Table 8 figure, Appendix A.8.3 Tables 14-17）

**Slope queries 的金矿**：在 7B 的 16/20 queries、35B 的 20/20 queries 的 top 100 中找到 procedural knowledge。手动识别出 7 个 JavaScript 实现 slope 计算的 documents、13 个数学公式表达。这些 documents 在多个 query 的 top 100 重复出现（7B 用 18 个、35B 用 8 个）。

**用 Command R+ 做 keyword 标注**（Tables 14-17）。对 top 500 docs × 每个 query 的 query-doc pair，让 Command R+ 选择描述关系的关键词。结果（节选 Table 15 slopes 7B）：

| Keyword | Count |
|---|---|
| Other types of maths | 10787 |
| Similar arithmetic on similar numbers | 7312 |
| Code that contains arithmetic | 5035 |
| Text explaining how to calculate slope of an equation | 3911 |
| Math that calculates slope between two numbers | 2490 |
| Code that calculates slope between two numbers | 1633 |

**Linear queries 35B**（Table 17）：13434 条"含线性方程但没解"、10717 条"相似代数运算相似数"、2415 条"实际解线性方程"。说明 influential docs 大量是"做类似事情"的 documents，少数是"展示如何解"的 documents。

**直觉**：模型从一堆"在做类似运算"的 examples 中**抽象**出 procedure，少量"显式展示 procedure"的 documents 提供模板。这非常像人类学一个新 skill 的过程：看很多类似例题 + 少量规范讲解。

### Finding 5: Source dataset analysis（Figures 7-10, Appendix A.8.4）

**Metric**：对 source dataset（Wikipedia、Math & Trivia、StackExchange、Code、ArXiv 等），看 top-k ranking 中的占比 vs training distribution 中的占比的 **multiplier**。Multiplier=1 表示随机采样预期。

**Factual**：
- 7B top-50：Wikipedia ×5, Math & Trivia ×27
- 35B top-50：Wikipedia ×6, Math & Trivia ×16

**Reasoning**：
- 7B top-50：StackExchange ×50, Math & Trivia ×24
- 35B top-50：StackExchange ×62, Math & Trivia ×21
- Code 在 k=5000 到 k=50000 之间 multiplier ≈ 2（factual 在同区间 ≈ 0.5）
- ArXiv 也 overrepresented for reasoning

**关键点**：Code data 在 reasoning 的 top **和** bottom ranking 都 overrepresented（Figure 8 显示 bottom 部分模式一致）。配合 Aryabumi et al. 2024 (https://arxiv.org/abs/2408.10914) "To code or not to code"——code data 确实 causal 地提升 reasoning。

**Content analysis**（Figure 11, Appendix A.8.5）：对 top 5000 influential docs 做 capability classification，code 类占绝大多数，random subset 中 code 占比很小。证据是双重的：source label 是 code + content 真的是 code。

## 7. Cross-lingual Transfer（Appendix A.8.2）的隐藏彩蛋

在 factual 搜索中，作者发现答案出现在非英语 documents：
- Portuguese 文档提到 Mount Everest "a montanha, de 8.848m, é a mais alta do mundo"。
- French 文档提到 Brussels 作为 Belgium 首都。
- Spanish 也有。

7B 找到 8 次 cross-lingual 命中，35B 4 次。这只是 keyword overlap 暴露的部分，实际 cross-lingual transfer 可能多得多。这连接到多语言模型的研究，以及最近 "cross-lingual knowledge transfer in pretraining" 这一支（Kotha et al. 2024, https://openreview.net/forum?id=VrHiF2hsrm 等）。它暗示：模型在 multilingual pretraining 中把 fact 编码成 language-agnostic representation，再用英语 surface form 输出。

## 8. 7B vs 35B 的额外发现（末尾 Appendix A.9.1）

36 个 7B 和 35B 共享 prompt 的 query，跨模型 influence score 相关性几乎为 0（平均 R=0.02，最高 0.19，对"What is the capital of Belgium?"，两个模型 completion 完全相同但相关仍只有 0.19）。

**直觉**：35B 不只是"7B + 更多 data + 更长训练"，它从 qualitatively 不同的 documents 抽取知识。这呼应 Grosse et al. 2023 的 finding：larger models rely on more abstractly related documents。这也呼应 scaling laws (Kaplan et al. 2020, https://arxiv.org/abs/2001.08361; Hoffmann et al. 2022, https://proceedings.neurips.cc/paper_files/paper/2022/file/c1e2faff6f588870935f114ebe04a3e5-Paper-Conference.pdf) 之外的"qualitative shift"现象。

## 9. 与其他 Interpretability / Mechanistic 工作的连接

### 9.1 与 Induction Heads (Olsson et al. 2022, https://transformercircuits.pub/2022/in-context-learning-and-induction-heads/index.html)

paper 推测：注意力层负责"实例化 procedure 到具体数字"的部分。MLP 负责存 procedure template。作者在 Discussion 明确提到这个未来方向：把 influence functions 扩展到 attention layers，连接到 Olsson et al. 的 induction heads。

### 9.2 与 Sparse Autoencoders / Monosemanticity (Bricken et al. 2023, https://transformercircuits.pub/2023/monosemantic-features/index.html; Templeton et al. 2024, https://transformercircuits.pub/2024/scaling-monosemanticity/index.html)

SAE 找到的"features"和 influence function 找到的"documents"是同一现象的两端：features 是激活空间的稀疏方向，documents 是 data 空间的稀疏贡献。SAE 给你"模型内部表征什么"，influence function 给你"什么外部 data 形成了那个表征"。两者结合是 mechanistic + data-level interpretability。

### 9.3 与 Grokking (Wang et al. 2024, https://arxiv.org/abs/2405.15071)

"Grokged transformers are implicit reasoners"——Wang et al. 在小模型上 mechanistically 证明 grokked transformer 做 implicit reasoning。本文在大模型上从 data 角度证实 procedural generalization。两支证据互补。

### 9.4 与 SFT / Fine-tuning 机制 (Prakash et al. 2024, Jain et al. 2024 https://openreview.net/forum?id=A0HKeKl4Nl, Kotha et al. 2024)

作者用 "SFT EK-FAC is identity" 假设。这系列工作显示 SFT 主要 enhance existing capabilities，不引入新机制。本文的方法学可以扩展到 fine-tuning data，是明确的 future work。

## 10. 局限性的诚实讨论

### 10.1 Sample 5M 不等于全 pretraining

最大局限：只看 2.5B tokens 的 sample，trillions of tokens 没看。作者的反驳三连击：
1. Qualitative 分析显示 influential docs 确实 intuitive relevant；
2. 同类 reasoning 的 correlation 高度显著；
3. Control set 不显示同样 pattern。

可能的反方：reasoning 靠的 document **太稀少**以至于 5M sample 都 surface 不到真正 influential 的。作者说：对最简单的 arithmetic，模型不太可能学到 retrieval from such infrequent data。但这个反驳对复杂 reasoning 就弱了。

### 10.2 Accuracy 是离散的

Influence function 是连续 differentiable 函数的近似。Accuracy 是 0/1。删除多个 docs 的累积效应 influence 没建模（它只估单 doc 删除效应）。所以 Appendix A.1 的实验是"crude but necessary"。

### 10.3 只看 MLP

Attention 的 EK-FAC 未定义好。作者 acknowledge 这是关键 future direction，因为 reasoning 的一部分发生在 attention heads（Olsson et al. 2022）。这暗示本文低估了 attention 在 reasoning 中的作用。

### 10.4 噪声

35B 的一些 ranking top 出现明显不相关 documents。Influence function 在大模型上的噪声特性还没完全理解（Choe et al. 2024 的 re-ranking by gradient norm 是一个 mitigation，作者用了）。

## 11. 对 Pretraining Data Selection 的实际含义

最 actionable 的含义：**reasoning 不需要 cover every case**。Factual 需要 dense coverage（模型要见一个 fact 多次才记住），reasoning 只需要 high-quality procedural examples across diverse tasks。

具体策略推断：
1. **Code data 真的重要**——但什么样的 code？是"实现 procedure 的 code"还是"任何 code"？paper 暗示前者，但没直接验证。这是 Aryabumi et al. 2024 没答完的问题。
2. **Diverse procedural examples** > 高频重复同一例子。10 个不同 slope 的代码实现可能比 100 份同一实现更有效。
3. **StackExchange 类 Q&A 数据** overrepresented for reasoning（multiplier 50+），暗示"step-by-step worked examples" format 有效。
4. **Math & Trivia 双用**：factual 和 reasoning 都受益，但来源不同（factual 来自 trivia，reasoning 来自 math 部分）。
5. **跨任务 procedural transfer 仍未验证**：paper 只在 math reasoning 内看到 procedural transfer。Code 是否能提供 cross-domain procedural transfer（数学 reasoning → logical reasoning → planning）是开放问题。

## 12. 与最近 reasoning 系列工作的串联

把这篇放在更大的 reasoning research map 里：

- **Verification-based methods**：Cobbe et al. 2021 (https://arxiv.org/abs/2110.14168) 的 verifier——本文说明为什么 verifier 必要：模型不靠 retrieval 所以会 procedural 出错。
- **Process reward models**：lightman et al. 类工作——procedural knowledge 的结构使得 step-level reward 自然。
- **Self-consistency / CoT**：本文的 procedural 视角解释了 CoT 为什么 work——它让 procedural template 显式展开，每步对应 pretraining 中的 procedural documents。
- **Reasoning in o1/RFT 类系统**：本文是 base model 层面的研究。RL on reasoning trace 是另一个抽象层，但 base model 必须先有 procedural substrate。

## 13. 直觉总结

整篇 paper 的 mental model 可以浓缩成几句话：

LLM 在 pretraining 时**学的是 procedure template，存的是抽象的"怎么做"**，而不是"答案库"。Factual retrieval 走 parametric memory，依赖少数 high-influence documents（典型例子：Wikipedia、trivia）；reasoning 走 procedural synthesis，依赖一大批低 influence 但 procedural-similar 的 documents（典型例子：code、StackExchange、ArXiv）。

一个 document 对多个同类 reasoning query 有相似 influence，因为它们都"激活"同一个 procedure template。模型从 procedure template 实例化到具体数字，这一步很可能在 attention 里发生（本文没看 attention）。

这个观点把 LLM reasoning 从"stochastic parrot"和"genuine understanding"的两极拉到一个中间地带：**它是 procedural synthesis over abstracted templates**。比 retrieval 强，比真 reasoning 弱。和人类学 skill 的过程惊人地像——我们也是从一堆类似例子 + 少量规范讲解中抽象出 method，然后应用到新数字上。

## 14. 一些值得追的 follow-up 方向

- 把 EK-FAC 扩展到 attention（Grosse et al. 后续工作）
- Influence function on RLHF/preference data
- Procedural transfer cross-domain（math → logic → code → planning）
- 什么样的 code data 是 positively influential vs negatively influential？
- Cross-lingual procedural transfer（procedure template 是否 language-agnostic？）
- 用 influence function 做 data curation 的 active learning loop
- Influence 在 in-context learning (few-shot) 中的角色——此时 retrieval pattern 可能不同
- Mechanistic verification：influential code documents 在 model 内激活的 features 是否能被 SAE 拆出 monosemantic "slope-calc" feature？
- Procedural knowledge 的 forgetting 特性——catastrophic forgetting 是否对 procedural vs factual 不同？

---

**Key references for further reading:**

- Grosse et al. 2023, "Studying Large Language Model Generalization with Influence Functions": https://arxiv.org/abs/2308.03296
- Koh & Liang 2017, "Understanding Black-box Predictions via Influence Functions": http://proceedings.mlr.press/v70/koh17a.html
- kronfluence public implementation: https://github.com/pomonam/kronfluence
- Hampel 1974, "The Influence Curve": https://www.tandfonline.com/doi/abs/10.1080/01621459.1974.10482962
- Halko et al. 2011, randomized SVD: https://doi.org/10.1137/090771806
- Wei et al. 2022, Chain of Thought: https://openreview.net/forum?id=_VjQlMeSB_J
- Olsson et al. 2022, Induction Heads: https://transformercircuits.pub/2022/in-context-learning-and-induction-heads/index.html
- Templeton et al. 2024, Scaling Monosemanticity: https://transformercircuits.pub/2024/scaling-monosemanticity/index.html
- Bricken et al. 2023, Monosemanticity: https://transformercircuits.pub/2023/monosemantic-features/index.html
- Wang et al. 2024, Grokked Transformers are Implicit Reasoners: https://arxiv.org/abs/2405.15071
- Aryabumi et al. 2024, To Code or Not to Code: https://arxiv.org/abs/2408.10914
- Bae et al. 2024, Approximate Unrolled Differentiation: https://arxiv.org/abs/2405.12186
- Choe et al. 2024, What is Your Data Worth to GPT: https://arxiv.org/abs/2405.13954
- Wu et al. 2024, Reasoning or Reciting: https://aclanthology.org/2024.naacl-long.102
- Razeghi et al. 2022, Impact of Pretraining Term Frequencies: https://aclanthology.org/2022.findings-emnlp.59
- McCoy et al. 2023, Embers of Autoregression: https://arxiv.org/abs/2309.13628
- Mahowald et al. 2024, Dissociating Language and Thought: https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(24)00026-8
- Webb et al. 2023, Emergent Analogical Reasoning: https://www.nature.com/articles/s41562-023-01659-w
- Prakash et al. 2024, Fine-tuning Enhances Existing Mechanisms: https://openreview.net/forum?id=8sKcAWOf2D
- Jain et al. 2024, Mechanistically Analyzing Fine-tuning: https://openreview.net/forum?id=A0HKeKl4Nl
- Command R / R+: https://cohere.com/blog/command-r
- Cohere 35B on HuggingFace: https://huggingface.co/CohereForAI/c4ai-command-r-plus

希望这个讲解帮你 build 起对"LLM 怎么从 data 学 reasoning"的 intuition：influence function 是显微镜，procedural knowledge 是观察对象，code/StackExchange/ArXiv 是 procedure template 的金矿。下一步研究 attention 的 EK-FAC 应该能补上"实例化"那一半故事。
