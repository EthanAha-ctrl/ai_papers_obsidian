---
source_pdf: Natural Questions a Benchmark for Question Answering Research.pdf
paper_sha256: 1ca22c60950ea77b7dc0d89b1e7e5dc40695626fdb40ac050bf46b6b5028d9d4
processed_at: '2026-08-05T21:53:33-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

用最直白的大白话来 build 你的 intuition，这篇 paper 的核心可以拆成几个层次。

### 1. 这篇 Paper 到底在干嘛？
一句话：Google 把自家搜索引擎里真实的用户 query 抓出来，配上对应的 Wikipedia 页面，雇人标出答案，做了一个巨大的 dataset，用来测 machine learning 模型能不能像人一样做 end-to-end QA。

之前的 dataset（比如 SQuAD，https://arxiv.org/abs/1606.05250）怎么搞？让人看一段文章，再编问题。这会导致问题与文章高度耦合，模型随便学点 pattern 就能高分，甚至超过人类。NQ 彻底打破这点，直接用 raw 的 Google search log，真实用户怎么搜，数据就怎么长。

### 2. 数据怎么来的？真实世界长啥样？
Google 用了 5 条 heuristics（规则）去 query log 里过滤。比如要求 query 必须是 8 个词以上，必须以 who/when/where 开头，后面跟着 do/be/have 等动词。这些规则本质上是 keyword matching，确保留下来的都是 seeking factual information 的问题。

拿到 (question, Wikipedia page) pair 后，标注员走三步流水线：
1. Question Identification：光看问题，判断问题合不合规。
2. Long Answer Identification：在页面上选一个 HTML bounding box（通常是一个 paragraph），里面必须包含能推导出答案的所有信息。如果页面没答案，标 NULL。
3. Short Answer Identification：在 long answer 内部挑出具体的 entity 作为 short answer。

这就是 NQ 的 task formulation。一个非常关键的统计数据：**51% 的 examples 最终的 long answer 是 NULL**。这跟 SQuAD 100% 都有答案的分布完全不同，直接逼迫模型学会说“我不知道”。

### 3. 标注的本质：人就是一个带噪音的随机函数
这是整篇 paper 最深刻的 insight。我们通常认为标注数据是 ground truth，是一个确定的映射。但这篇 paper 用概率论告诉你：人本质上是一个带噪音的 random function。

paper section 4.1 定义了核心概率分布：
$$p(l, q, d) = p(q, d) \times p(l | q, d)$$

变量解释：
- $q$: question (用户问题)
- $d$: document (Wikipedia 页面)
- $l$: long answer (标注员给的长答案)
- $p(q, d)$: 从 Google query log 采样到 (问题, 文档) 对的概率
- $p(l | q, d)$: 给定问题和文档，随机抽一个标注员，他给出答案 $l$ 的条件概率

注意这里的上下标和逻辑：$p(l | q, d)$ 是非确定性的。哪怕同一个人，早上精神好和晚上犯困，标的答案可能不一样。所以标注只是从真实分布 $p(l | q, d)$ 里抽的一个 sample。你手里那 30 万条训练数据，全是带噪音的 single sample。

### 4. “灰色地带”有多坑？
为了研究标注的 noise，paper 做了极端的 25-way annotation（同一个问题让 25 个人分别标）。

他们定义 $\mu^{(i)} = \frac{1}{25} \sum_{j=1}^{25} [[a_j^{(i)} \neq \text{NULL}]]$。
变量解释：$a_j^{(i)}$ 是第 $j$ 个标注员给的第 $i$ 个例子的答案，$[[ \cdot ]]$ 是 indicator function（条件成立为1，不成立为0）。$\mu^{(i)}$ 就是 25 个人里有多少比例给了非 NULL 答案。

实验数据表（Figure 4 对应的 text 描述）发现了一个强关联：
- 当 $\mu^{(i)} > 0.8$（80%的人觉得有答案）：Expert 判断基本全是 Correct。
- 当 $\mu^{(i)} \leq 0.2$（80%的人觉得没答案）：Expert 判断基本全是 Wrong。
- 中间地带（众说纷纭）：大部分属于 $\mathcal{C}_d$ (Correct but debatable)，比如“血液从右心室泵到哪里”，有人标“lungs”，有人标“肺动脉”。都对，但 granularity 不同。

Table 2 显示，Long answer 有 31% 属于这种 debatable 状态。Disagreement 本身就是 task-inherent ambiguity 的 signal。

### 5. 评测怎么搞？上界到底在哪？
既然标注是个概率分布，怎么评模型？如果模型预测了一个答案，跟你的 single sample 不一样，能说明模型错了吗？不能。

于是 paper 设计了 5-way annotation 的评估 metric。找 5 个人标，用 $\beta=2$ 作为 threshold。
核心评测函数 $h_\beta(a, l)$：
- 如果 5 个人里至少 2 个给了非 NULL 答案，且模型预测的 $l$ 等于这 5 个人里的某一个，得 1 分。
- 如果 5 个人里少于 2 个给答案，且模型预测 NULL，得 1 分。
- 其他情况 0 分。

Accuracy 公式：
$$A_\beta(f_\theta) = \frac{1}{n} \sum_{i=1}^{n} h_\beta(a^{(i)}, f_\theta(q^{(i)}, d^{(i)}))$$
变量解释：$f_\theta$ 是你的模型，$n$ 是测试样本数，$a^{(i)}$ 是 5 个人的标注向量。

**最震撼的 Bayes Optimal Insight：**
paper 指出，一个模型如果学到了真实的 $p(l | q, d; \theta)$，它可以打败 single annotator！
直觉：单个标注员是从分布里抽一次 sample，必然有偏差。模型如果学到了整个分布，它直接输出概率最大的那个 mode（众数），相当于一个“超级标注员” aggregating 了全人类的智慧。
paper 用 25-way 数据构造了这个 super-annotator。Table 3 结果：单个人 F1 只有 73.4%，super-annotator F1 达到 87.2%。这就是理论上的 human upper bound。模型理论上可以比单个人聪明。

### 6. Baseline 架构与实验数据表解析
paper 给了两个主要 baseline。看 Table 3 的 Long Answer Test F1 数据：
- First paragraph: 28.3
- DocumentQA: 45.7
- DecAtt + DocReader: 55.0
- Super-annotator (天花板): 87.2

DocumentQA (https://arxiv.org/abs/1711.06155) 直接拿 SQuAD 的模型改，效果一般。paper 自己设计的 DecAtt + DocReader pipeline 最好。

架构图解析（Conceptual Pipeline）：
paper 把任务拆成两步，两步用不同架构。
1. **Long answer selection (选段落)**：这步更像 NLI（自然语言推理）。用 Decomposable Attention (DecAtt, https://arxiv.org/abs/1606.01933) 算 question 和 candidate passage 的 entailment score。
2. **Short answer extraction (抽词)**：这步像传统 RC。用 DocReader (DrQA 的 reader 部分, https://arxiv.org/abs/1704.00051) 在选出的段落里做 span extraction。

Long answer 的打分公式：
$$z_l = \mathbf{w}^\top [x_l, r_l, u_l, v_l] + b$$

变量与维度解释：
- $z_l$：candidate passage $l$ 的最终分数。标量。
- $x_l$：DecAtt 算出的 NLI feature 向量。
- $r_l$：段落位置的 10 维 embedding。表示这是页面上第几个候选段落。
- $u_l$：question 和 passage 共享的 word 数量。标量。
- $v_l$：TFIDF 加权的 word overlap 数量。标量。
- $\mathbf{w}^\top$：可训练的权重向量，维度与拼接后的特征一致。
- $b$：bias 标量。
最终接一个 softmax 选出 long answer。

这个 baseline 离 super-annotator 还差 30 个 F1 点，证明 NQ 非常难，留给后续模型巨大的空间。

### 7. 对今天 LLM / RLHF 时代的深远影响
这篇 2019 年的 paper，里面的思想在 2026 年的 LLM 时代完全应验了。

1. **RLHF 的 disagreement 问题**：现在训 LLM 做 RLHF，通常让标注员对两个回答做 pairwise preference。很多标注员会打架。NQ paper 提出的 $p(l|q,d)$ 概率框架直接启发了后来的工作（比如 Anthropic 的 Constitutional AI, https://arxiv.org/abs/2212.08073）。我们应该 modeling 标注分布，去拟合一个 Bayes optimal 的 reward model，让它超越单个人的判断。
2. **Self-Consistency 的前身**：NQ 的 super-annotator 思想，本质就是现在 LLM 推理时用的 Self-Consistency（https://arxiv.org/abs/2203.11171）。让模型 sample 多次，取 majority vote，比单次 greedy decoding 效果好得多。
3. **RAG 的标准 benchmark**：NQ 的 setup（给定整篇 Wikipedia 找答案）就是今天 RAG 应用的完美模拟。后来 Facebook (Lewis et al. 2020, https://arxiv.org/abs/2005.11401) 提出 RAG 架构，主战场就是 NQ。

总结一句：NQ 告诉我们，真实用户数据天生充满 ambiguity 和 noise。别把人当 oracle，把人当 random function，你才能 build 出真正 robust 的 AI 系统。

---

# Natural Questions (NQ) Paper 深度解析

## 一、Paper 的设计哲学与历史定位

这篇 paper 是 Google Research 在 2019 年 TACL 发表的，核心作者阵容豪华（Tom Kwiatkowski, Slav Petrov, Jacob Devlin, Kenton Lee 等 BERT 系列的同一拨人）。要 build intuition 的话，需要先理解 2018-2019 这个时间点的 QA landscape：

- SQuAD 1.1 (Rajpurkar et al., 2016, https://arxiv.org/abs/1606.05250)：annotator 先读段落再写问题，问题与段落高度耦合，Jia & Liang (2017) 用一句话 distractor 就能摧毁系统性能
- SQuAD 2.0 (https://arxiv.org/abs/1806.03822)：加入了 unanswerable questions，但这些"不可答"是被设计成不可答的，模型识别起来容易
- MS MARCO (https://arxiv.org/abs/1611.09268)：用 Bing search log，但依赖 BLEU 评估 free-form answer，BLEU 本身就是 noise
- NarrativeQA (https://arxiv.org/abs/1712.07040)：太难，baselines 没超过 random
- HotpotQA (https://arxiv.org/abs/1809.09600)：multi-hop，但其 human upper bound 是通过 partition maximization 算的，分布偏移了

NQ 的核心 design philosophy 是把 "end-to-end QA" 这个应用场景还原到 rawest form：
1. 用户输入一个真实的 search query（不是 annotator 编造的）
2. 系统给出 evidence passage（long answer）
3. 系统给出 entity-level extraction（short answer）

这个 setup 对应了 search engine 中 featured snippet 的产品需求，非常接地气。

---

## 二、Question Filtering Heuristics 的技术细节

paper section 3.1 给了 5 条 heuristic 规则从 Google query log 中筛 question：

1. start with 'who/when/where' + finite form of 'do' or modal verb
2. start with 'who/when/where' + finite form of 'be'/'have' + verb in later position
3. start with 'who' + verb (not 'be' form)
4. contain multiple entities + adjective/adverb/verb/determiner
5. contain categorical noun phrase preceded by preposition or relative clause
6. end with categorical noun phrase, no preposition/relative clause

同时要求 ≥8 words，且有 Wikipedia page 在 top-5 search result。

这个 heuristic 其实是 keyword-based pattern matching，相当于把 query understanding 的工作量转移到了搜索引擎自己的 ranking 信号上。值得注意的是，这种 heuristic 会引入 selection bias：偏向于长 question、偏向 Wikipedia-friendly topic（历史/政治/地理/名人），这跟 TriviaQA (https://arxiv.org/abs/1705.03551) 类似的 trivia 偏好一致，但 NQ 比 TriviaQA 更偏"普通用户想要的事实信息"。

paper 自己也承认了这一点，建议未来 dataset 引入 "how to" 问题（实际上后来 Google 自己出了 Natural Instructions / BIG-Bench 等扩展任务集，可以参考 https://github.com/google/BIG-bench）。

---

## 三、Annotation 流水线与 decision process

annotation 分 3 个 conceptual stage，单个 annotator 一气呵成（平均 80 秒一个 example）：

**Stage 1: Question Identification**
仅看 question 文本，判断是否是 "fact-seeking" question。这避免了 annotator 在看到 evidence 后强行构造一个"答案 fitting"的 cognitive bias。这一步在 SQuAD 这种"先看文章再写问题"的 setup 中是反过来的，所以 SQuAD 问题质量有 inherent 偏向。

**Stage 2: Long Answer Identification**
要求 annotator 选 earliest HTML bounding box，可以是 paragraph / table / table row / list / list item。统计上 paragraph 73%、table 19%、table row 1%、list 3%、list item 3%。

**Stage 3: Short Answer Identification**
在 long answer 内选 entity 或 entity set，也可以标 yes/no 或 NULL。

paper Figure 2 给出了 path proportion：
- 51% 标 NULL（没有 long answer）
- 49% 有 long answer
- 36% 有 short answer

这意味着 NQ 接近一半 examples 的 "正确答案" 就是 "no answer"，这个分布对 model 训练至关重要，paper section 6.3 的 DecAtt baseline 用 η 来 down-weight NULL label，就是因为 NULL label 信息量低（paper section 6.3 loss 公式）。

---

## 四、Annotation Quality 的概率论建模（核心创新之一）

paper section 4.1 把整个 annotation 过程形式化成一个 sampling distribution，这是这篇 paper 最被低估的部分。

定义随机变量 (Q, D, L, S) 表示 question, document, long answer, short answer。联合分布：

$$p(l, q, d) = p(q, d) \times p(l | q, d)$$

其中：
- $p(q, d)$：从 Google query log 采样 question，再从 top-5 search result 中选 Wikipedia page 的 PMF
- $p(l | q, d)$：随机抽一个 annotator，让 ta 标 long answer 的 PMF

注意 $p(l | q, d)$ 是非确定性的，因为：
- annotator 是 random 选择的
- 即使同一个 annotator，受时间/状态影响也会给出不同答案

这是非常本质的建模——把"标注过程"当作 random function 而不是 ground truth oracle。这跟 Bowman 在 SNLI 里、Pavlick & Kwiatkowski (2019, https://aclanthology.org/D19-1333/) 的 "Inconsistency in Human Inferences" 的思想一脉相承。后来 Pavlick & N Berger 在 https://aclanthology.org/2022.findings-emnlp.382/ 进一步把这个 idea 推广到了 LLM-disagreement 研究上。

定义 conditional distribution：

$$p(l, q, d | L \neq \text{NULL}) = \frac{p(l, q, d)}{P(L \neq \text{NULL})} \text{ if } l \neq \text{NULL}, \text{ else } 0$$

其中 $P(L \neq \text{NULL}) = \sum_{l,q,d: l \neq \text{NULL}} p(l, q, d)$。

然后定义 correctness function $\pi(l, q, d) \in \{0, 1\}$，得到 **annotation precision**：

$$\Psi = \sum_{l,q,d} p(l, q, d | L \neq \text{NULL}) \times \pi(l, q, d)$$

直觉上：从所有非 NULL 的标注中采样一个 triple，它在 expert 判断下是正确的概率。这就是 paper section 4.3 报告的 Long answer precision 90% / Short answer precision 84% 的形式化定义。

---

## 五、Expert Evaluation 三档分类

paper section 4.2 设计了一个非常有意思的三档分类（用 4 个 expert 做的 consensus judgment）：

- **$\mathcal{C}$ (Correct)**: clear beyond reasonable doubt
- **$\mathcal{C}_d$ (Correct but debatable)**: 合理人可以满意，但另一个合理人可以提出合理怀疑
- **$\mathcal{W}$ (Wrong)**: 没有令人信服的证据

Table 2 结果：

| Quantity | Long answer | Short answer |
|---|---|---|
| $\hat{\Psi}$ (precision) | 90% | 84% |
| $\hat{E}(\mathcal{C})$ | 59% | 51% |
| $\hat{E}(\mathcal{C}_d)$ | 31% | 33% |
| $\hat{E}(\mathcal{W})$ | 10% | 16% |

$\mathcal{C}_d$ 占 30% 这个数字很关键——说明 1/3 的标注处于"灰色地带"。paper 明确把 $\mathcal{C}_d$ 算入 precision，理由是 "用户也认为这是 high-quality answer"，但这个决策本身是 arguable 的。

Figure 3 的 Example 2 非常说明问题：
> Question: which type of rock forms on the earth's crust
> Long answer: Igneous and metamorphic rocks make up 90-95% of the top 16 km of the Earth's crust by volume. Igneous rocks form about 15% of the Earth's current land surface...
> Judgment: Correct (but debatable)

reason 是问题问"哪种 rock forms on crust"，答案答了 90-95% 的 crust composition，但严格说没有 "form" 这个动作的对应。这种 fine-grained semantic ambiguity 是 NQ 真正要测的能力。

---

## 六、25-way Annotation 与 Variability 研究

paper section 4.4 收集了 302 个 examples 的 25-way annotation，这是 paper 最有价值的资产之一。

定义 $\mu^{(i)} = \frac{1}{25} \sum_{j=1}^{25} [[a_j^{(i)} \neq \text{NULL}]]$：25 个标注中 non-null 的比例。

Figure 4 的核心 finding：
- $\mu^{(i)} \in (0.8, 1.0]$：几乎全部都是 $\mathcal{C}$（correct）
- $\mu^{(i)} \in (0, 0.2]$：几乎全部都是 $\mathcal{W}$（wrong）
- 中间区间：主要由 $\mathcal{C}_d$ 占据

这个发现非常 powerful，因为它建立了一个 **agreement-accuracy correlation**：annotator 之间的 disagreement 不简单是噪声，而是反映了 question-answer pair 的内在 ambiguity。

极端 variability 例子：
- "name the substance used to make the filament of bulb"：annotator 找到了 7 个不同的段落都讨论 tungsten wire filament
- "where is blood pumped after it leaves the right ventricle"：25 个标注里有 11 个 distinct 但都 correct 的答案，从 "lungs" 到 "through the semilunar pulmonary valve into the left and right main pulmonary arteries" 各种 granularity

这种 variability 在评估 metric 设计中需要被 accounting。

---

## 七、5-way Annotation Metric 设计（核心创新之二）

paper section 5.1 是这篇 paper 最 dense 的技术部分。定义两个核心函数：

**$g(a^{(i)})$**: 5 个标注中 non-null 的数量

**$h_\beta(a, l)$** （Definition 1）：
- if $g(a) \geq \beta$ and $l \neq$ NULL and $l = a_j$ for some $j$: $h_\beta = 1$
- elif $g(a) < \beta$ and $l =$ NULL: $h_\beta = 1$
- else: $h_\beta = 0$

threshold $\beta = 2$。直觉：至少 2 个 annotator 给出 non-null 答案时，要求 model 输出 non-null 答案；否则要求 model 输出 NULL。

**Model accuracy**:

$$A_\beta(f_\theta) = \frac{1}{n} \sum_{i=1}^{n} h_\beta(a^{(i)}, f_\theta(q^{(i)}, d^{(i)}))$$

**Precision / Recall**:
$$t(q, d, a, f_\theta) = h_\beta(a, f_\theta(q, d)) \cdot [[f_\theta(q, d) \neq \text{NULL}]]$$

$$R(f_\theta) = \frac{\sum_i t(q^{(i)}, d^{(i)}, a^{(i)}, f_\theta)}{\sum_i [[g(a^{(i)}) \geq \beta]]}$$

$$P(f_\theta) = \frac{\sum_i t(q^{(i)}, d^{(i)}, a^{(i)}, f_\theta)}{\sum_i [[f_\theta(q^{(i)}, d^{(i)}) \neq \text{NULL}]]}$$

直觉：
- Recall 的分母是"应该被回答的 examples 数量"
- Precision 的分母是"模型实际给出 non-null 答案的数量"
- $t$ 是 true positive indicator

---

## 八、Bayes Optimal Hypothesis 的 Insight（最深刻的发现）

paper 在 section 5.1 末尾脚注 11 给了一个非常重要的 theoretical insight：**训练好的 model 可以超越单个 annotator**。

直觉：单个 annotator $a_j$ 是从 $p(l | q, d)$ 中 sampling 一次的结果，含有 sampling noise；而一个学到了 $p(l | q, d; \theta)$ 的 model 可以 marginalize 掉这个 noise。

形式化定义：
- $l^* = \arg\max_{l \neq \text{NULL}} p(l | q, d)$：最可能的 non-null 答案
- $\gamma = p(l^* | q, d)$：最可能 non-null 答案的概率
- $\bar{\gamma} = p(\text{NULL} | q, d)$：NULL 的概率

特殊 case $\gamma + \bar{\gamma} = 1$ 下：
- $P(h_\beta(a, \text{NULL}) = 1 | \gamma, \bar{\gamma}) = \bar{\gamma}^5 + 5\bar{\gamma}^4(1 - \bar{\gamma})$
  - 直觉：5 个 annotator 全标 NULL（$\bar{\gamma}^5$）+ 恰好 1 个标 non-null（$5\bar{\gamma}^4(1-\bar{\gamma})$）
- Bayes optimal 策略：输出 $l^*$ if $\gamma \geq \alpha$，输出 NULL otherwise
- $\alpha \approx 0.31381$，是 $\bar{\alpha}^5 + 5\bar{\alpha}^4(1-\bar{\alpha}) = 0.5$ 的解

这个 $\alpha \approx 0.31381$ 的来源其实就是 5-way annotation 中 majority 投票（≥2 个 non-null）的期望值的 threshold。

**这个 insight 对今天 LLM RLHF 时代依然有效**：RLHF 通常用 pairwise preference，而 pairwise 数据本身也有 annotator disagreement。如果我们能 estimate $p(\text{pref}|x, y_1, y_2)$ 而不是取一个 majority vote，理论上可以训练出超越平均 annotator 的 reward model。这跟 Anthropic 的 Constitutional AI (https://arxiv.org/abs/2212.08073) 思想有部分重叠。

---

## 九、Super-Annotator Upper Bound 构造

paper section 5.2 用 25-way annotation 构造了一个 "super-annotator"：

- 输入：20 个标注 $b_1^{(i)} \ldots b_{20}^{(i)}$（与 5-way 的 $a^{(i)}$ 来自同一分布）
- 策略：
  - 如果 $g(b^{(i)}) < \alpha$：输出 NULL
  - 否则：输出 $l^* = \arg\max_{l \in d} \sum_j [[l = b_j^{(i)}]]$
- 用 $\alpha = 8$

Table 3 结果（Long answer, dev set）：
- Single annotator: P=80.4, R=67.6, F1=73.4
- Super-annotator: P=90.0, R=84.6, F1=87.2

super-annotator 比单 annotator 高了 14 个 F1 point，这正好验证了 Bayes optimal insight 的实践可行性。这一方法跟现在 LLM 时代的 self-consistency (Wang et al. 2022, https://arxiv.org/abs/2203.11171)、Universal Self-Consistency (https://arxiv.org/abs/2311.17311) 在精神上同源——都是用多次 sample 的 majority vote 来逼近 $p(l|q,d)$ 的 mode。

---

## 十、Baseline Architecture 深度解析

paper section 6 报告 5 个 baseline，分两类：

### 10.1 Untrained Baselines

| Method | Long P | Long R | Long F1 |
|---|---|---|---|
| First paragraph | 22.2 | 37.8 | 27.8 |
| Most frequent | 43.1 | 20.0 | 27.3 |
| Closest question (TFIDF) | 37.7 | 28.5 | 32.4 |

直觉：
- First paragraph 召回高（37.8%）因为 Wikipedia 第一段是 summary，常常含答案
- Most frequent 精度高（43.1%）但召回低，因为 79% dev page 在 train set 里出现过，可以"记住"
- Closest question 是 retrieval-by-question-similarity，是后来 dense-retriever 思路的雏形

### 10.2 DocumentQA Baseline

DocumentQA (Clark & Gardner 2018, https://arxiv.org/abs/1711.06155) 原本是为 SQuAD/TriviaQA 设计的，paper section 6.2 做了两个关键 adaptation：

1. 处理 long answer 不含 short answer 的情况：用特殊 [start]/[end] of passage tokens 标记，prediction time 可以单独预测 long answer
2. 处理 NULL answer：训练只在有答案 examples 上做，prediction 时用 score threshold ≥ 3.0 过滤

Hyperparameter：top-16 passages（TFIDF pruning）、max 17-word short answer、30 epoch、batch 15、threshold 3.0。结果 Long F1 = 45.7。

### 10.3 DecAtt + DocReader Pipeline

paper section 6.3 自己设计的 baseline，最有意思的部分。核心 insight：**long answer selection 更接近 NLI 任务，short answer extraction 更接近 SQuAD 任务**。

公式：
$$x_l = \text{DecAtt}(q, t(d, l))$$

$x_l$ 是 Decomposable Attention (Parikh et al. 2016, https://arxiv.org/abs/1606.01933) 在 question 与 candidate passage tokens 上的 NLI-style score。再加上 4 个 features：
- $r_l$: 10-dim trainable position embedding
- $u_l$: word overlap count (integer)
- $v_l$: TFIDF-weighted word overlap (scalar)

最终 linear combination：

$$z_l = \mathbf{w}^\top [x_l, r_l, u_l, v_l] + b$$

Training loss（重要细节）：

$$-\sum_{i=1}^{n} \log \frac{\exp(z_{l^{(i)}})}{\sum_l \exp(z_l)} \times (1 - \eta \cdot [[l^{(i)} = \text{NULL}]])$$

变量解释：
- $l^{(i)}$：ground-truth label for example $i$
- $z_l$：candidate $l$ 的 score
- $\eta$：down-weighting factor for NULL examples
- $[[l^{(i)} = \text{NULL}]]$：indicator function，1 当 ground-truth 是 NULL

直觉：NULL label 的 loss 被 $(1 - \eta)$ 缩放，paper 解释 "NULL label less informative than an answer location"。这个 trick 对今天 LLM fine-tuning 也有启发——imbalanced label 需要不同 weight，但 weight 的形式可以更精细。

DecAtt 部分 push 候选段落到 softmax 里，等价于 classification over passages。Short answer 用 DocReader (Chen et al. 2017, https://arxiv.org/abs/1704.00051)，是 DrQA 的 reader 部分。

---

## 十一、Baseline 实验数据表深度分析

Table 3 完整结果：

| Method | Long P | Long R | Long F1 | Short P | Short R | Short F1 |
|---|---|---|---|---|---|---|
| DocumentQA | 48.9 | 43.3 | 45.7 | 40.6 | 31.0 | 35.1 |
| DecAtt+DocReader | 54.3 | 55.7 | 55.0 | 31.9 | 31.1 | 31.5 |
| Single annotator† | 80.4 | 67.6 | 73.4 | 63.4 | 52.6 | 57.5 |
| Super-annotator† | 90.0 | 84.6 | 87.2 | 79.1 | 72.6 | 75.7 |

观察：

1. DecAtt+DocReader 在 long answer 上完胜 DocumentQA（55.0 vs 45.7），但在 short answer 上落后（31.5 vs 35.1）。这印证了 paper 的 hypothesis：long answer 任务是 NLI-style，short answer 任务是 RC-style。这是 task-specific architecture 的胜利。

2. 任何 baseline 距离 single annotator (73.4 long, 57.5 short) 都有 15-20 F1 point gap，距离 super-annotator (87.2, 75.7) 差得更远。paper section 6.4 用这个 gap 论证 NLU 还需要 significant advance。

3. paper section 6.4 还做了条件分析：把 NQ 限制在"有 long & short answer"的子集上（类似 SQuAD setup），DecAtt+DocReader Long F1 = 68.0, Short F1 = 40.4；给定 long answer 提取 short answer，DocReader F1 = 57.7。这个 57.7 与 SQuAD 上 DocReader 原文报告（69.5）的差距，是 NQ question 更难（multi-paragraph reasoning）的证据。

4. Long answer breakdown by type：
   - Paragraph：65.1 F1
   - Table / table row：66.4 F1
   - List / list item：32.0 F1

list 类型崩盘（32.0）是 NQ 的一个 specific failure mode。后来 BERT-based 模型（Alberti et al. 2019, https://arxiv.org/abs/1901.08603）也用了 BERT-large 解决了部分问题，但是 list 类型一直是个难点。

---

## 十二、与今天（2026 LLM 时代）的关联

NQ 在 LLM 时代仍然有独特价值：

1. **True user distribution**：LLM 时代的 SFT 数据很多时候还是 researcher-crafted prompt，NQ 这种从 search log 来的 distribution 仍然 rare。OpenAI 的 WebQuestions / ELI5 也尝试过类似路线。

2. **Open-domain / closed-book QA benchmark**：NQ 后来被用作 LLM 的 closed-book QA benchmark（Roberts et al. 2020, https://arxiv.org/abs/2002.08909 用 T5 在 NQ 上做 closed-book，F1 ≈ 36%，开了 closed-book QA 这个方向）。

3. **Realistic baseline for RAG**：RAG (Lewis et al. 2020, https://arxiv.org/abs/2005.11401) 用 NQ 做主 benchmark，因为 NQ 的 setup（question + 整篇 Wikipedia page）跟 RAG 应用场景非常 match。

4. **KILT benchmark**：KILT (Petroni et al. 2021, https://arxiv.org/abs/2109.00643) 把 NQ 等 11 个 task 统一到一个 framework，NQ 仍是其中核心的 factoid QA task。

5. **NQ-open**：Lee et al. 2021 (https://arxiv.org/abs/2108.00661) 把 NQ 转成 open-domain 版本，去掉给定 Wikipedia page，让 retriever 也参与。这是 PaLM, GPT-3, LLaMA 系列 LLM closed-book eval 的主战场。

6. **Natural Instructions / FLAN**：NQ 后来成为 instruction tuning 的标配 task 之一（https://arxiv.org/abs/2103.10360, https://arxiv.org/abs/2210.11416）。

7. **Annotation variability 的研究**：paper 25-way annotation 思想直接启发了 Pavlick & N Berger 2022 (https://aclanthology.org/2022.findings-emnlp.382/)，他们用 5 个 annotator 在 SNLI 上证明 annotator disagreement 是 systematic 的、可建模的。这又进一步影响 today's RLHF 的 disagreement modeling（比如 Anthropic 的 https://arxiv.org/abs/2204.05862）。

---

## 十三、Dataset Bias 与 Limitations 讨论

paper section 1 与 section 2 自我批评：
- Wikipedia-only evidence（"stylistically similar to other factual info"，但实际 web 有 forum, news, blog 等多样风格）
- ≥8 words 长 question 偏好（短 question 被过滤）
- 没有 "how to" question（filter 主动排除）
- 偏向 factoid，不含 opinion / reasoning question

这些 bias 在 LLM 时代更明显：GPT-4 在 NQ 上 closed-book F1 已经超过 70%（参考 https://arxiv.org/abs/2305.0101 等），但这不意味着 LLM 真正"理解"，可能是 memorize + retrieval by pattern matching。所以今天做 LLM eval，NQ 已经是 saturated benchmark，需要更新版本（比如 NQ v2、KILT+，或者完全替代品如 MMLU-Redux）。

paper section 3.1 也提到 "Future data-collection efforts may introduce shorter queries, 'how to' questions, or domains other than Wikipedia"——实际上 Google 自己后来出了 https://arxiv.org/abs/2203.10244 等扩展。

---

## 十四、为什么这篇 paper 值得细读

如果你只关心今天用 LLM 解决 QA 任务，这篇 paper 可能看起来 outdated。但它真正 lasting contribution 是：

1. **建立 annotation variability 可以建模的概率框架**：从 $p(l|q,d)$ 出发，把 annotator 当成 random function，这跟现在 Bayesian deep learning / ensemble 思路殊途同归。

2. **建立 super-annotator upper bound 方法论**：用多次标注 + majority vote 构造 upper bound，是后来 human ceiling 比较的范式。

3. **建立"model 可以超越单 annotator"的 insight**：这是 RLHF / DPO 时代 reward model 训练的理论基础——理论上模型可以是 Bayes optimal 而不局限于 single annotator 的 noise level。

4. **建立 evaluation metric 的形式化**：$h_\beta$ 函数 + precision/recall split 是后来 QA task 的标准 metric 框架。

5. **建立 end-to-end QA 的 task formulation**：NQ 之后，WebQuestions, TriviaQA, Open-NQ, KILT-QA 都沿用了"real user query + evidence retrieval + answer extraction"的范式。

paper 官网：https://storage.googleapis.com/naturalquestions/
GitHub baseline: https://github.com/google-research-datasets/natural-questions
数据 download: https://ai.google.com/research/NaturalQuestions/download
Long answer baseline repo (BERT): https://github.com/google-research/language/tree/master/language/question_answering

如果你想深入 build intuition 关于 annotation noise modeling，强烈推荐再看：
- Pavlick & Kwiatkowski 2019: https://aclanthology.org/D19-1333/
- Polyak et al. 2021 on NQ variability: https://aclanthology.org/2021.acl-long.351/
- Plank 2022 survey on disagreement: https://aclanthology.org/2022.cl-1.21/

---

## 十五、Intuition 总结

NQ 这篇 paper 最值得 internalize 的几个 intuition：

1. **真实分布 > 人工构造**：用 Google search log 的真实 query 比 SQuAD-style "annotator 编造问题"在 distribution shift 上更鲁棒。这对今天合成数据生成（synthetic data generation for LLM training）的反思：synthetic data 可能 over-fit 到 LLM 自己的 prior 上，真实用户数据有 unmatched 价值。

2. **Annotation 是 random function**：不要把 annotation 当 ground truth，要把它当成 $p(l|q,d)$ 的 sample。这与今天 RLHF 中 pairwise preference 的建模思路一致。

3. **Bayes optimal > Single annotator**：model 通过学习 $p(l|q,d; \theta)$ 而不是 single sample，可以超越单个 annotator。这是 self-consistency, ensemble, RLHF reward averaging 的共同理论基础。

4. **Disagreement 信号**：annotator disagreement 不是 noise，是 task-inherent ambiguity 的 signal。25-way annotation 中间区间主要由 $\mathcal{C}_d$ 占据证明了这一点。

5. **Task-specific architecture**：long answer 是 NLI 风格，short answer 是 RC 风格，分开训练比 end-to-end 更好（DecAtt+DocReader 证明）。当然 LLM 时代这个 distinction 消失了，但 insight 仍可启发 multi-task training。

6. **Evidence selection 是一等公民**：NQ 强调 long answer 任务的独立性，把它作为 short answer 的 prerequisite，这是 RAG 时代 retriever 优先设计的雏形。

7. **Realistic upper bound**：用 super-annotator 而不是 single-annotator 作为 human ceiling，避免了"human 已经 100% 完美"的错觉。

希望这个解析帮你 build 出对 QA dataset 设计、annotation variability 建模、以及 LLM RLHF 时代相关 insight 的 intuition。
