---
source_pdf: What is a Number That a Large Language Model May Know It.pdf
paper_sha256: 98725c45615c79ce2bf8fbdcb31661a4bc20fd71a2458a063a7ff1abdd3f621a
processed_at: '2026-08-13T04:04:52-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇paper

## 一句话总结

LLM看数字的时候，脑子里同时装着两套系统——一套把数字当**数值**算，一套把数字当**字符串**看。这两套系统搅在一起，分不开。

---

## 问题的根源

你给LLM看"911"这三个字符，它其实很懵：

- 如果上下文是"拨打911报警"，那这是字符串，就像一个ID编号
- 如果上下文是"911加1等于多少"，那这是数值九百一十一
- 如果上下文是"911事件"，那这是日期

人类一眼能分清楚，但LLM只能从上下文猜。问题是，在没有任何上下文提示的时候，LLM的"默认状态"是什么？

答案是：**它把两套表示混在一起用了**。

---

## 怎么证明这件事

作者用了心理学里一个很老的trick：**问相似度**。

你不用问"这个数字是多少"，因为那会逼model做数学。你只问"785和685有多像？给个0到1的分数"。这个问题本身很neutral，model得用自己的internal representation来回答。

然后作者拿了两个理论参照：

**Levenshtein距离**——就是"改几个字符能变成另一个"。785改成685，改1个字符。785改成791，改2个字符。这是字符串的视角。

**Log-Linear距离**——785和685在数值上差100，785和791差6。但人类对大数不敏感，所以加个log压缩。这是数值的视角。

然后用回归把LLM的相似度矩阵拆成这两部分的线性组合。

结果：**两个都有贡献**。平均能解释73%的variance，其中数值占大头（61%），字符串占小头（21%），但字符串那部分统计上绝对显著。

---

## Context有没有用

作者试了一招：在prompt里加Python type hint。

```
Number: int(785)    ← 明确告诉model这是整数
Number: str(785)    ← 明确告诉model这是字符串
```

结果很有意思：

- 加`int()`之后，字符串成分从21%降到15%——**降了但没消**
- 加`str()`之后，字符串成分从21%升到31%——**推一下就往那边倒**

所以context能**调节**方向，但**消除不了**字符串表示。字符串的那部分已经baked into weights里了，prompt改不动。

---

## 最精彩的实验：换进制

作者让model在base 4和base 8下做相似度判断。

这招太狠了。LLM训练数据里base 4的数字几乎不存在，所以model完全没"数值直觉"。那它怎么办？

**退回到字符串模式**。

Llama-3.1-70b在base 10下，Log-Linear占78%，Levenshtein只占11%。到了base 4，Levenshtein反超到37%，Log-Linear掉到34%。

这说明字符串表示是**fallback**——当数值表示不够用的时候，model默认回到字符层面。

---

## 内部probe：打开脑子看看

作者拿Llama-3.1-8b开了膛。方法是在第25层residual stream上训一个线性probe，专门decode两种距离。

结果：

- Integer probe和Log-Linear距离相关0.917——数值subspace确实存在
- String probe和Levenshtein距离相关0.650——字符串subspace也存在
- 两个probe之间相关0.667——**两个subspace有重叠**

所以不是"分了两个干净的房间"，而是"一个房间里有两套家具混着放"。

用MDS可视化更直观：在integer subspace，数字按log scale线性排开；在string subspace，共享digit的数字扎堆。两个map完全不同的结构，但probe能互相predict对方的一部分。

---

## 真实场景会出什么事

作者构造了一个化浓度选择task：

```
你需要785 ppm的化合物。
两个试管：一个685 ppm，一个791 ppm。
选哪个最接近？
```

人类一眼看：791差6，685差100，选791。

但如果model有string bias，它看785和685共享"85"两个字符，可能觉得更像。

结果：

| Model | 3位数字错误率 | 5位数字错误率 |
|-------|------------|------------|
| Llama-3.1-8b | 37% | **47%** |
| Llama-3.1-70b | 11% | 19% |
| GPT-4o | 0.02% | 0.41% |

Llama-3.1-8b在5位数字下接近random！而GPT-4o几乎全对。

**两个insight**：

1. 数字越长，string bias越严重——更多digit参与edit distance，"假相似"更多
2. Scale和RLHF能suppress但不是消除string bias——GPT-4o那么强也不是0%

---

## 为什么是Log-Linear而不是Linear

作者发现纯 $|a-b|$ 距离拟合不如Log-Linear。这说明LLM的数值表示本身就有log压缩。

这和人类一样。人类对"2和4差多少"很敏感，对"1002和1004差多少"就糊了。LLM在人类写的text上训练，吸收了这个prior。

所以LLM不只学到了数字的数学结构，还学到了**人类感知数字的方式**。

---

## 这意味着什么

几个deep point：

**Token type ambiguity是根本问题**。LLM从来没有"看见"过数字，它只看见过数字的text。所以representation必然带textual痕迹。这不是bug，是text-only training的必然结果。

**Scale helps但不解决**。GPT-4o在applied task上几乎不出错，但internal representation里string成分依然存在。大model只是学会了在output层面suppress它，没在representation层面eliminate它。

**这是autoregression的ember**。McCoy et al. 2024说LLM的训练objective本身留下autoregressive artifacts。String bias可能就是其中一种——model学会了"字符相似"这个统计regularity，因为它对next-token prediction有用。

**Number不是孤例**。代码、化学式、日期、电话号码、邮编——所有"符号序列但语义独立"的东西都可能受同样的entanglement影响。数字只是最容易量化的test case。

---

## 我觉得最cool的地方

作者用100年前McCulloch的问题开头："What is a number, that a man may know it"。然后用2024年的工具回答：**machine知道的number是两个东西的混合物——一个mathematical object，一个textual object**。

人类能清楚区分这两个，因为我们有sensorimotor grounding——我们能数手指、量长度、感受"多"和"少"。LLM没有这个grounding，只能从text co-occurrence里infer。Inference不完美，所以entanglement留下了。

这不是LLM的缺陷，是**纯text training的必然代价**。要解决，可能得给model别的grounding channel——或者干脆用neuro-symbolic方法把数值运算从text processing里剥离出去。

---

# What is a Number, That a Large Language Model May Know It? — 深度解析

## 1. The Core Question: Number as Token Ambiguity

这篇paper的核心observations非常优雅，可以追溯到McCulloch 1961年的著名论文 "What is a number, that a man may know it, and a man, that he may know a number?"。在人类认知中，数字有dual nature：既是magnitude（量），又是symbol（符号）。当LLM通过text prediction训练时，它实际上面临一个**tokenization-level ambiguity**——同一串digit tokens `911`，在不同context下可能是emergency phone number的字符串，也可能是数值911，也可能是日期September 11。

这种ambiguity和linguistic polysemy/homonymy有结构相似性，但paper指出它extends across different symbolic systems（numeric vs. string）。这就引出了paper的核心hypothesis：**LLM学到的number representation是string-like和number-like representations的entanglement**。

直觉上来说，因为LLM只通过next-token prediction训练，它从未"看见"过数字本身——它只看见过数字的textual realization。所以model必须从context中infer number type，这个infer过程必然留下痕迹在representation中。

Reference: McCulloch 1961 — https://groups.csail.mit.edu/medg/ftp/psz/McCulloch-Number.html

---

## 2. Methodology: Similarity Judgments from Cognitive Science

### 2.1 The Paradigm

Paper借用cognitive science中经典的**similarity judgment paradigm**（Shepard 1962, 1980; Tversky & Hutchinson 1986; Tenenbaum & Griffiths 2001）。给定domain $\mathcal{D}$ 和一个agent $A$，通过elicit pairwise similarity judgments $s_{ij}$ across all pairs $(x_i, x_j)$，构造一个similarity matrix。这个matrix定义了agent对domain施加的proximity structure。

关键点：similarity是"neutral"的——"how similar is $x_i$ to $x_j$?" 不会impose任何structure给agent。Agent必须用自己的internal representation来回答。这就和RSA（Representational Similarity Analysis, Kriegeskorte 2008）相通。

Paper中prompt是：
```
How similar are the two numbers on a scale of 0 (completely dissimilar) 
to 1 (completely similar)? Respond only with the rating.
Number: {NUM1}
Number: {NUM2}
Rating:
```

### 2.2 Theoretical Distance Measures

Paper用两个理论距离作为regressor来decompose LLM的similarity matrix：

#### Levenshtein Edit Distance (string-like)
递归定义：

$$
d_{Lev}(a, b) = \begin{cases} 
|a|, & \text{if } |b| = 0 \\
|b|, & \text{if } |a| = 0 \\
d_{Lev}(a_{1..n}, b_{1..n}), & \text{if } a_0 = b_0 \\
1 + \min\{d_{Lev}(a_{1..n}, b), d_{Lev}(a, b_{1..n}), d_{Lev}(a_{1..n}, b_{1..n})\}, & \text{else}
\end{cases}
$$

变量解释：
- $a = a_0 \ldots a_n$, $b = b_0 \ldots b_n$ 是两个字符string
- $a_{1..n}$ 表示去掉首字符后的substring
- $a_0 = b_0$ 检查首字符是否相等
- 三种operation：deletion ($d_{Lev}(a_{1..n}, b)$), insertion ($d_{Lev}(a, b_{1..n})$), substitution ($d_{Lev}(a_{1..n}, b_{1..n})$)
- 例：$d_{Lev}(\text{"200"}, \text{"100"}) = 1$，只需一次substitution (2→1)

#### Log-Linear Distance (number-like)
基于Piantadosi (2016)的psychological number representation：

$$
d_{Log}(x, y) = 1 - \exp\left(-|\log(x + \epsilon) - \log(y + \epsilon)|\right)
$$

变量解释：
- $x, y$ 是两个非负整数
- $\epsilon = 10^{-4}$ 是regularizer，处理domain包含0的情况（避免$\log(0)$发散）
- 这个距离encode了Weber-Fechner-like logarithmic sensitivity：人类对大数的representation fidelity随magnitude递减
- 直觉：$x \to \log(x)$ 使得 $100$ vs $200$ 的距离 = $1000$ vs $2000$ 的距离（在log space中两者都是 $\log 2$）

为什么Log-Linear而不是简单的 $|x - y|$？这源于Piantadosi的rational analysis of the approximate number system (ANS)——人类对number的mental representation是log-compressed的，这是evolutionary optimal的因为environment中number magnitude的分布近似scale-invariant。Paper的Appendix E验证了简单的$\ell_1$ distance $|a - b|$ 确实拟合更差（$R^2$从.726降到.567）。

Reference: Piantadosi 2016 — https://link.springer.com/article/10.3758/s13423-015-0996-6

---

## 3. Experimental Design & Results

### 3.1 Models Tested

6个SOTA models：
- GPT-4o (gpt-4o-2024-08-06, Azure API)
- Claude-3.5-Sonnet (claude-3-5-sonnet-20241022)
- DeepSeek-V3 (together.ai)
- Llama-3.1-8b-Instruct-Turbo-128K
- Llama-3.1-70b-Instruct-Turbo
- Mixtral-8x22B-Instruct-v0.1

关键细节：选0-999范围是因为在这个range内，所有integer都是**unique tokens** in the tokenizer，这控制了tokenization-specific artifacts（比如GPT-4o的BPE中"1000"可能被切成"1"+"000"或者"100"+"0"）。

Cost考虑：1000×1000 = 1,000,000 pairwise comparisons，对Claude-3.5-Sonnet要花~$160。所以全部用temperature=0进行greedy decoding，附录F补充了temperature=0.7的robustness check。

### 3.2 Regression Analysis

将distance转换为similarity：
$$
s_{ij}(d) = 1 - \frac{d_{ij}}{\max\{d_{ij}\}}
$$

然后z-score后做线性回归：
$$
s_{ij} = \alpha + \beta \cdot s_{ij}^{(Lev)} + \gamma \cdot s_{ij}^{(Log)}
$$

用scikit-learn的LinearRegression，bootstrap over number pairs（1000次）得到95% CIs。

### 3.3 Default Context Results

Figure 1的similarity matrices显示了一个非常striking的pattern：所有model都展现出**block-diagonal structure** + **sub-diagonal structures**。这两种structure分别对应：
- Block-diagonal：Log-Linear characteristic（数值接近的数聚集）
- Sub-diagonal：Levenshtein characteristic（共享digit的数聚集，比如100 vs 200，110 vs 210）

**Quantitative breakdown** (default context)：

| Predictor | $R^2$ (95% CI) |
|-----------|---------------|
| Combined | [.725, .726] |
| Log-Linear only | [.607, .609] |
| Levenshtein only | [.213, .215] |
| Linear $\ell_1$ only (control) | worse, combined $R^2$ = .567 |

关键insight：Levenshtein alone能explain ~21%的variance，这意味着model的number representation远非pure numerical。Log-Linear dominant但Levenshtein显著contribute——这就是**entanglement**的quantitative evidence。

### 3.4 Type Specification Context (int() vs. str())

实验设计：用Python-style type hint来disambiguate token type：
```
Integer Similarity Prompt:
Number: int({NUM1})
Number: int({NUM2})
```
```
String Similarity Prompt:
Number: str({NUM1})
Number: str({NUM2})
```

**int() context**：

| Predictor | $R^2$ (95% CI) |
|-----------|---------------|
| Combined | [.721, .722] |
| Log-Linear | [.645, .646] |
| Levenshtein | [.156, .158] |

**str() context**：

| Predictor | $R^2$ (95% CI) |
|-----------|---------------|
| Combined | [.620, .621] |
| Log-Linear | [.410, .412] |
| Levenshtein | [.309, .311] |

观察：
1. int() context把Levenshtein contribution从~21%降到~15%，Log-Linear从~61%升到~65%
2. str() context把Levenshtein contribution从~21%升到~31%，Log-Linear从~61%降到~41%
3. **但Levenshtein永远无法被完全eliminate**——即使在int() context，string component仍然显著

这是paper的核心claim：**context can reduce entanglement, but cannot eliminate it**。这暗示string representation深植于model的weights中，是训练数据的intrinsic property。

Figure 2A还显示qualitative变化：GPT-4o在int() context下lose了block-diagonal structure（更linear），而Claude-3.5-Sonnet在str() context下反而gain了block-diagonal（更string-like）。这表明不同model对context的response direction不同——可能反映了training data和RLHF的差异。

### 3.5 Different Number Bases (Base 4 and Base 8)

这是一个非常clever的实验设计：如果model真的是用edit distance来represent number，那么base的change不应该影响string contribution——因为edit distance不care语义。

**Llama-3.1-70b in base 10**：
- Log-Linear: [.780, .783]
- Levenshtein: [.108, .112]

**Llama-3.1-70b in base 4**：
- Log-Linear: [.336, .340]
- Levenshtein: [.366, .371]

**Llama-3.1-70b in base 4 vs base 10**：Levenshtein contribution从~11%**逆转**到~37%！

这个结果非常深刻：在uncommon base下，model失去了numerical intuition（因为training data中base 4 representations极少），只能fall back到string edit distance。这强烈支持了**string representation是default/fallback，numerical representation是需要数据learned的**这一观点。

### 3.6 Probing Internal Representations

这部分用Llama-3.1-8b做internal probing。技术细节：

#### Residual Stream Structure
Transformer每层更新token的latent：
$$
h_i^{(j)} = h_i^{(j-1)} + g(h_1^{(j-1)}, \ldots, h_i^{(j-1)})
$$

变量：
- $h_i^{(j)}$ 是token $i$ 在第 $j$ 层的residual stream representation
- $g(\cdot)$ 是attention + MLP + normalization的复合函数
- $h_i^{(0)}$ 是初始embedding (来自vocabulary embedding matrix E)

#### Linear Probe Design
对每个layer训练一个affine transformation：
$$
\hat{d} = w \cdot h_i^{(j)} + b
$$

变量：
- $h_i^{(j)} \in \mathbb{R}^{4096}$ 是Llama-3.1-8b的hidden dimension（latent dim = 4096）
- $w \in \mathbb{R}^{4096}$, $b \in \mathbb{R}$ 是可学习参数
- $\hat{d}$ 是预测的距离值（Levenshtein或Log-Linear）

训练细节：
- 9,500 random pairs from 0-999用于training
- Evaluation on 0-500 range（500×500 = 250,000 pairs）
- Layer ablation选了**layer 25**（共32层）
- Adam optimizer, 100 epochs
- Important detail: input是prompt最后一个token ("":" after "Rating:") 的residual，因为这通常是information aggregation的位置

#### Probing Results (Table 2)

|  | String Probe | Levenshtein | Int Probe | Log-Linear |
|--------------|--------------|-------------|-----------|------------|
| String Probe | 1 | 0.650*** | 0.667*** | 0.527*** |
| Levenshtein | 0.650*** | 1 | 0.393*** | 0.266*** |
| Int Probe | 0.667*** | 0.393*** | 1 | 0.917*** |
| Log-Linear | 0.527*** | 0.266*** | 0.917*** | 1 |

关键观察：
1. **Int probe与Log-Linear相关0.917**，与Levenshtein相关0.393——integer subspace确实encodes数值
2. **String probe与Levenshtein相关0.650**，与Log-Linear相关0.527——string subspace也encode一些数值信息
3. **Int probe与String probe相关0.667**——两个subspace之间有shared structure，这是entanglement在embedding level的直接证据

#### MDS Visualization
Figure 4用了Multidimensional Scaling（MDS, Shepard 1962）将decoded similarity matrices投影到2D。Int subspace中数字按log-linear scale线性排列；string subspace中数字按edit distance非线性排列（共享digits的cluster在一起）。

这是非常震撼的visualization——你能在latent space中直接看到两种representation同时存在。

---

## 4. Behavioral Implications: The Compound Concentration Task

### 4.1 Close Triplets Construction

这是paper中most applied的部分。Goal：构造diagnostic triplets $(q_0, q_1, q_2)$ 使得：
- (i) 数值上$q_1, q_2$都接近$q_0$，有unambiguous的numerically correct answer
- (ii) String edit distance上，"wrong"选项反而更接近$q_0$

构造算法：
1. 从 $\{2, \ldots, 9\}$ 中随机选3个digits组成$q_0$，e.g., $q_0 = 331$
2. **Levenshtein-aligned option** $q_1$：从$q_0$的最大digit位置减1，e.g., $q_1 = 231$（共享2个digits）
3. **Log-aligned option** $q_2$：保留$q_0$的最大digit，其他两个digits从 $\{1, \ldots, 9\}$ 中重新采样（排除已用digits），e.g., $q_2 = 357$

这样：
- 数值上：$|q_0 - q_1| = |331 - 231| = 100$, $|q_0 - q_2| = |331 - 357| = 26$ → $q_2$数值上更近
- String上：$d_{Lev}(q_0, q_1) = 1$, $d_{Lev}(q_0, q_2) = 2$ → $q_1$ string上更近

如果model选$q_1$，说明string bias胜过numerical judgment。

5-digit同样构造，e.g., $(25337, 15337, 26886)$。

数据：3-digit有6,474 unique triplets，5-digit有9,995 unique triplets。

### 4.2 Prompt

```
You require a compound with a concentration of approximately 785 ppm. 
Two test tubes are available: one containing 685 ppm and the other 791 ppm. 
Your task is to determine which test tube provides the most similar 
concentration to your required dosage. Which one will you choose? 
Respond only with the ppm value of the test tube you choose.
```

数值上791明显更接近785（差6 vs 100），但string上685与785共享两个digits。

### 4.3 Results (Figure 5)

String-bias error rate (averaged over presentation order):

| Model | 3-digit | 5-digit |
|-------|---------|---------|
| Llama-3.1-8b | **36.86%** | **47.03%** |
| Llama-3.1-70b | 11.38% | 19.04% |
| GPT-4o | 0.02% | 0.41% |

震撼的observations：
1. **Llama-3.1-8b在5-digit上错误率47%**——接近random！这说明小model的string bias极其严重
2. **GPT-4o几乎无错误**（<0.5%）——大model + 强RLHF能suppress string bias
3. **5-digit错误率比3-digit更高**——更长numbers有更多digits参与edit distance计算，string bias被exacerbated
4. **Order effect存在但不symmetric**——某些model对presentation order敏感

这个实验有现实意义：chemistry/biology/medicine中concentration calculation是high-stakes scenario。如果LLM用string similarity来判断concentration closeness，可能导致严重错误。

---

## 5. Connections to Broader Literature

### 5.1 Mechanistic Interpretability of Numbers

- **Zhu et al. 2025** "Language models encode the value of numbers linearly" — 用linear probe从latent embedding extract number value。这paper证实了这个finding，但展示了non-arithmetic context下representation是log-linear而非linear。
- **Hanna et al. 2023** "How does GPT-2 compute greater-than" — circuit analysis of GPT-2的">" operation。
- **Stolfo et al. 2023** — causal mediation analysis of arithmetic reasoning。
- **Nanda et al. 2023a** — grokking in modular arithmetic。

Reference: Zhu et al. 2025 — https://aclanthology.org/2025.coling-1.53/

### 5.2 Number Representation in Cognitive Science

- **Dehaene 2011** "The Number Sense" — 人类number cognition经典综述
- **Piantadosi et al. 2014** — Tsimané儿童number word learning
- **Cheyette & Piantadosi 2020** — unified account of numerosity perception
- **Miller & Gelman 1983** — 儿童number representation的MDS分析（这正是paper方法的灵感来源！）

Reference: Dehaene 2011 — https://academic.oup.com/book/35234

### 5.3 Probing & Representation Analysis

- **Alain & Bengio 2018** — linear classifier probes
- **Belinkov 2021** — probing的promises & shortcomings
- **Gurnee & Tegmark 2024** — "Language models represent space and time"（非常类似的方法论，发现LLM学到了linear spatial-temporal representations）
- **Gurnee et al. 2023** — sparse probing for neurons
- **Meng et al. 2022** — activation patching (ROME)
- **Cunningham et al. 2023** & **Gao et al. 2024** — sparse autoencoders
- **Geiger et al. 2024** — Distributed Alignment Search (DAS)

Reference: Gurnee & Tegmark 2024 — https://arxiv.org/abs/2310.02207

### 5.4 Behavioral Analysis of LLMs

- **Marjieh et al. 2024b** — LLMs predict human sensory judgments across 6 modalities
- **Binz & Schulz 2023** — cognitive psychology + GPT-3
- **Bai et al. 2024** — implicit bias measurement
- **Webb et al. 2023** — analogical reasoning in LLMs
- **McCoy et al. 2024a, 2024b** — "embers of autoregression"——LLM受training objective塑造，可能exhibitautoregressive artifacts。这个角度和paper的string bias相关：string edit可能就是autoregressive training的"ember"

Reference: McCoy et al. 2024 — https://www.pnas.org/doi/10.1073/pnas.2322420121

---

## 6. Theoretical Implications & Intuitions

### 6.1 Why Does This Entanglement Exist?

从first principles思考：LLM training是token sequence prediction。当model看到"9 + 11 = 20"，它需要知道9和11是numbers；当它看到"dial 911"，它需要知道911是string。Model必须从context infer type。在default context（no type hint），model面临type uncertainty。

最optimal的representation策略是什么？如果context真的能disambiguate，model应该学一个**type-conditional representation**。但paper显示default representation是**mixture**，not conditional——这暗示model没有完全solve这个disambiguation problem，而是用prior（基于training data中number vs. string的相对频率）做marginalization。

### 6.2 Log-Linear Representation Hypothesis

为什么LLM学到的numerical representation是log-linear而非linear？两个可能：

1. **Training data distribution**: 自然text中number frequencies服从power law——小数高频，大数低频。model学到的representation自然reflect这个distribution，导致log-compression。

2. **Implicit cognitive prior**: LLMs trained on human-generated text可能implicitly absorb了人类的number cognition patterns，包括ANS和Weber-Fechner law。这和Marjieh et al. 2024b的finding一致——LLM predict human sensory judgments。

### 6.3 Tokenization Effects

Paper刻意选0-999范围避免tokenization artifacts。但Appendix F.1扩展到0-2000发现pattern开始"broaden"——这可能是因为超出unique token范围后，multi-token numbers需要composition，破坏了single-token representation structure。

这是tokenization层面的open question：
- GPT-4o BPE: 0-1000是single tokens (with some exceptions like round numbers)
- Llama tokenizer: 不同tokenization scheme
- 不同tokenizer是否lead to不同entanglement level？这是future work

### 6.4 Reasoning Models (o1, R1)

Paper在discussion中提及OpenAI o1和DeepSeek R1等reasoning models。这非常interesting——chain-of-thought可能让model在CoT中explicitly manipulate numbers as integers，从而reduce string bias。但反过来，CoT中每次generated number token也可能引入string bias。这是值得实验的方向。

Reference: OpenAI o1 — https://arxiv.org/abs/2412.16720, DeepSeek R1 — https://arxiv.org/abs/2501.12948

### 6.5 Mathematical Operations & String Errors

Paper的discussion提到一个有趣的speculation：如果LLM把1和2看作strings，那么"sum 1 and 2"可能生成"12"而非"3"（字符串concatenation而非加法）。这种error pattern在LLM arithmetic evaluation中确实有reports，可能和string-number entanglement直接相关。

---

## 7. Critical Analysis & Open Questions

### 7.1 Methodological Considerations

**Strengths**:
1. Similarity judgment是model-internal-agnostic——可以apply to closed-source models
2. Two theoretical distances提供clean decomposition
3. Triple实验（behavioral + probing + applied）形成evidence triangle

**Limitations**:
1. 仅Llama-3.1-8b做了internal probing——其他model的latent structure未知
2. Temperature=0 only (尽管Appendix F验证了robustness)
3. 0-999 range——大number和多-token numbers的behavior未充分探索
4. 只测了integer——rational numbers, scientific notation, decimals的behavior未知

### 7.2 Open Questions for Future Research

1. **Causal intervention**: 能否在embedding level patch来causally切换model的type perception？
2. **Tokenization**: 不同tokenizer如何影响entanglement？BPE vs. Unigram vs. WordPiece
3. **Reasoning models**: CoT是否suppress string bias？或者introduce新的bias？
4. **Beyond numbers**: 这种symbol-string duality在code、化学式、特殊符号中是否也存在？
5. **Training data attribution**: training corpus中number作为string vs. number的ratio是否predict entanglement level？
6. **Multi-token numbers**: 12345（5 digits）vs. 12,345（with comma）vs. 12345.0（with decimal）的representation是否一致？
7. **Cross-lingual**: 中文"一二三"vs. "123"的representation是否相同？这能揭示model的abstract number concept。

### 7.3 Connection to Number Format Issues

这paper让我联想到LLM在实际应用中常见的number format issues：
- Phone numbers: model经常把phone number当number处理，丢失leading zeros
- Zip codes: 01234变成1234
- Dates: 09/11可能被当作arithmetic expression
- Scientific notation: 1.5e10的parse问题

这些应用层issue可能都根植于paper揭示的number-string entanglement。

---

## 8. Summary of Key Insights

1. **LLM的number representation是string和number的entanglement**，而非pure numerical
2. 这种entanglement在behavioral（similarity judgment）和internal（latent embedding）level都observable
3. **Context can reduce but not eliminate** string bias——int() hint降低Levenshtein contribution但无法消除
4. **Uncommon number bases放大string bias**——base 4下Levenshtein甚至dominate Log-Linear
5. **String bias有real-world behavioral consequences**——在concentration selection task中，Llama-3.1-8b错误率47%
6. **Model scale & RLHF显著reduce string bias**——GPT-4o几乎无错误
7. **Log-Linear representation暗示model absorb了人类number cognition的psychological prior**

最后这个finding特别深刻——LLM不仅学到数字的数学structure，还学到人类如何*感知*数字。这与McCulloch 1961年的原始question形成跨越64年的对话：machine通过artificial neural networks学到的number，仍然和人类number cognition有meaningful差异，但也展现了惊人的psychological plausibility。

Reference: Paper GitHub repo — https://github.com/vminvsky/numbers-in-llms

---

## 9. Personal Reflections

这篇paper让我想到几个deep questions关于LLM representation nature：

1. **Symbol grounding problem的新形式**: LLM如何grounding "数字"这个abstract concept？没有sensorimotor experience，它的grounding完全来自textual co-occurrence patterns。String bias可能就是这种imperfect grounding的痕迹。

2. **Format vs. Content**: 这paper揭示的是**format-content entanglement**——符号的表面format（digit sequence）和semantic content（数值）在representation中混合。这可能是text-trained model的fundamental limitation。

3. **Implicit Bayesian type inference**: Model可能在做implicit Bayesian inference来decide token type。Default context下的mixture representation可能就是marginalize over type posterior的结果。

4. **Implications for AGI**: 如果LLM连number这种最basic的concept都entangle，那更abstract的concepts（justice, democracy, meaning）的representation质量如何？这paper可能是一个microcosm反映broader representation challenge。

5. **Teaching LLMs "what a number is"**: 也许我们需要显式的number grounding module——像Neuro-Symbolic AI那样将numeric operations detach from text processing。或者更好的tokenizer设计（专门处理number tokens）。

这篇paper的方法论（cognitive science + probing + behavioral）也非常值得借鉴——这种multi-level evidence triangulation是研究LLM representation的gold standard。期待看到类似方法applied到其他abstract concepts。
