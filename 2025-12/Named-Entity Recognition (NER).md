
**Named-Entity Recognition (NER)**（也称为 entity identification、entity chunking 或 entity extraction）是 **Information Extraction** 的一个子任务，目标是从非结构化文本中定位并分类命名实体到预定义的类别中，如：
- **Person names (PER)**
- **Organizations (ORG)**
- **Locations (LOC)**
- **Geopolitical entities (GPE)**
- **Vehicles (VEH)**
- **Medical codes**
- **Time expressions**
- **Quantities**
- **Monetary values**
- **Percentages**

### 1.1 示例

输入文本：
```
Jim bought 300 shares of Acme Corp. in 2006.
```

NER 输出：
```
[Jim]Person bought 300 shares of [Acme Corp.]Organization in [2006]Time.
```

在这个例子中，系统检测并分类了一个单token的person name、一个双token的company name和一个时间表达式。

## 二、问题形式化定义

### 2.1 Rigid Designators 理论基础

NER 中的 "named entity" 概念与 Saul Kripke 提出的 **rigid designators** 理论密切相关[1][2]：

- **Rigid designator**：在所有可能世界中指代同一实体的表达
- 例如："Ford" 可以指代福特汽车公司（1903年由Henry Ford创立）
- 但实际NER处理中，许多名称并非哲学意义上的"rigid"

### 2.2 任务分解

完整的NER可概念性地分解为两个独立问题[4][5]：

**Phase 1: Name Detection**
- 形式化为 **segmentation problem**
- 名称定义为连续的token spans，无嵌套
- 例如："Bank of America" 作为单个名称，忽略"America"本身也是名称

**Phase 2: Name Classification**
- 选择一个ontology来组织实体类别
- 例如：person、organization、location

形式化表示：

给定输入序列 X = (x₁, x₂, ..., xₙ)，其中 xᵢ 是第i个token

输出是标签序列 Y = (y₁, y₂, ..., yₙ)，其中 yᵢ ∈ {B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, O}

这里：
- **B-TYPE** = Begin of entity TYPE
- **I-TYPE** = Inside of entity TYPE  
- **O** = Outside any entity

### 2.3 Temporal Expressions 的处理

时间和数值表达式在某些NER任务中也被视为named entities，但它们的定义较为宽松：

- **有效rigid designators示例**：年份"2001"（指代公历2001年）
- **无效示例**："I take my vacations in June"（June可能指过去、未来或任何一年的六月）

## 三、实体类型层次结构

### 3.1 BBN Categories (2002)

用于Question Answering任务，包含：
- **29 types**
- **64 subtypes**

### 3.2 Sekine's Extended Hierarchy (2002)

包含 **200 subtypes** 的精细分类[8]

### 3.3 Freebase-based Hierarchy (2011)

Ritter 在social media text上的开创性实验中使用[9]，基于common Freebase entity types

## 四、NER 主要困难

### 4.1 指代消歧歧义

**同型异义**：同一名称可指代同类型的多个实体
```
"JFK" → 前总统 John F. Kennedy OR 他的儿子 John F. Kennedy Jr.
```

### 4.2 跨类型歧义

**同名异型**：同一名称可指代完全不同类型的实体
```
"JFK" → 纽约机场
"IRA" → Individual Retirement Account OR International Reading Association
```

### 4.3 转喻 (Metonymy)

机构名称的转喻用法：
```
"The White House" → 指代组织而非地点
```

## 五、评估指标

### 5.1 基本指标

**Precision (精确率)**：
```
Precision = TP / (TP + FP)
```
- **TP** (True Positive) = 正确预测的实体
- **FP** (False Positive) = 错误预测的实体

**Recall (召回率)**：
```
Recall = TP / (TP + FN)
```
- **FN** (False Negative) = 遗漏的实体

**F1 Score**：
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### 5.2 CoNLL 评估标准

在学术会议如CoNLL中，F1 score变体定义如下[5]：

**Precision**：
- 预测的entity name spans与gold standard数据中spans完全匹配的数量
- 当预测 [Person Hans] [Person Blick] 但实际要求 [Person Hans Blick]，precision为零

**Recall**：
- Gold standard中的names在predictions中完全相同位置出现的数量

**严格匹配**：任何预测如果miss一个token、包含错误token或wrong class，都是hard error

### 5.3 Token-by-token 匹配评估

基于token匹配的评估模型[10]，允许：
- 部分credit用于重叠匹配
- 使用 **Intersection over Union (IoU)** 准则
- 更细粒度的评估和比较

IoU公式：
```
IoU(A, B) = |A ∩ B| / |A ∪ B|
```
其中 A 和 B 分别是预测和ground truth的token spans

## 六、方法与技术

### 6.1 Grammar-based Systems

**特点**：
- 基于语言学的语法规则
- 手工crafted规则
- 通常获得更好的precision
- 但recall较低，需要经验丰富的计算语言学家数月工作[11]

**工具**：
- **GATE**：支持多语言和多域NER，提供图形界面和Java API
- **OpenNLP**：包含规则基和统计NER

### 6.2 Statistical NER Systems

**特点**：
- 需要大量手动标注的训练数据
- Semi-supervised方法可减少标注工作量[12][13]

#### 6.2.1 传统统计方法

**线性回归 + 双向Viterbi解码**：

在统计学习时代，NER通常通过以下方式执行：

1. **Feature Engineering**
2. **学习简单的线性回归模型**
3. **使用双向Viterbi算法解码**

**常用特征**[14]：

| 特征类型 | 描述 | 示例 |
|---------|------|------|
| Lexical items | token本身 | "Washington" |
| Stemmed items | 词干化 | "Washington" → "washington" |
| Shape | 正字法模式 | ALL CAPS, Title Case, etc. |
| Affixes | 词缀 | 前缀/后缀 |
| POS | 词性标注 | NNP, VB |
| Gazetteers | 命名实体词典 | "General Electric" |
| Context | 周围context words | n-grams |

**Shape特征编码**：
```
shape(word) = 逐字符转换
"iPhone" → "Aaa#aa" (A=大写, a=小写, #=数字)
"USA" → "AAA"
"Mr." → "Aa."
```

**Gazetteers（专有名词词典）**：
- 名称及其类型的列表，如 "General Electric" → ORG
- 在统计机器学习时代广泛使用[15][16]

### 6.3 机器学习分类器

**常用分类器类型**：
- **HMM** (Hidden Markov Model)
- **ME** (Maximum Entropy)
- **CRF** (Conditional Random Fields) - 典型选择[17]
- **Transformers** - 深度学习模型[18]

#### 6.3.1 HMM (Hidden Markov Model)

**HMM 定义**：五元组 λ = (S, O, A, B, π)

其中：
- **S** = {s₁, s₂, ..., sₙ} 隐藏状态集合（entity tags）
- **O** = {o₁, o₂, ..., oₘ} 观测符号集合（tokens）
- **A** = {aᵢⱼ} 状态转移概率矩阵，aᵢⱼ = P(qₜ₊₁ = sⱼ | qₜ = sᵢ)
- **B** = {bⱼ(k)} 发射概率矩阵，bⱼ(k) = P(oₖ | qₜ = sⱼ)
- **π** = {πᵢ} 初始状态概率，πᵢ = P(q₁ = sᵢ)

**Viterbi算法**：寻找最可能的状态序列

给定观测序列 O = (o₁, o₂, ..., oₜ)，找到状态序列 Q* = (q₁*, q₂*, ..., qₜ*) 最大化 P(Q|O)

```
δₜ(i) = max P(q₁q₂...qₜ=oₜ, qₜ=sᵢ | λ)
      q₁...qₜ₋₁

递归公式：
δₜ(j) = max [δₜ₋₁(i) × aᵢⱼ] × bⱼ(oₜ)
       1≤i≤N

回溯：
ψₜ(j) = argmax [δₜ₋₁(i) × aᵢⱼ]
        1≤i≤N
```

#### 6.3.2 CRF (Conditional Random Fields)

**线性链CRF**：

给定输入序列 X = (x₁, x₂, ..., xₙ)，输出标签序列 Y = (y₁, y₂, ..., yₙ)

条件概率：
```
P(Y|X) = (1/Z(X)) × exp(Σᵢ Σₖ λₖ fₖ(yᵢ₋₁, yᵢ, x, i) + Σᵢ Σₖ μₖ gₖ(yᵢ, x, i))
```

其中：
- **Z(X)** = 归一化因子（配分函数）
- **fₖ(yᵢ₋₁, yᵢ, x, i)** = 转移特征函数（edge features）
- **gₖ(yᵢ, x, i)** = 状态特征函数（node features）
- **λₖ, μₖ** = 特征权重（需学习）

**Z(X)** 计算：
```
Z(X) = Σ_Y exp(Σᵢ Σₖ λₖ fₖ(yᵢ₋₁, yᵢ, x, i) + Σᵢ Σₖ μₖ gₖ(yᵢ, x, i))
```

**前向-后向算法**：高效计算 Z(X) 和边缘概率

**前向变量**：
```
αᵢ(yᵢ) = Σ_{y₁...yᵢ₋₁} exp(Σ_{t=1}^{i-1} Σₖ λₖ fₖ(yₜ₋₁, yₜ, x, t) + Σₖ μₖ gₖ(yₜ, x, t))
```

**后向变量**：
```
βᵢ(yᵢ) = Σ_{yᵢ₊₁...yₙ} exp(Σ_{t=i+1}^{n} Σₖ λₖ fₖ(yₜ₋₁, yₜ, x, t) + Σₖ μₖ gₖ(yₜ, x, t))
```

**配分函数**：
```
Z(X) = Σ_{yᵢ} αₙ(yₙ) = αᵢ(yᵢ) × βᵢ(yᵢ) (对任意i)
```

**特征函数示例**：

```
f₁(yᵢ₋₁, yᵢ, x, i) = 1, if yᵢ₋₁ = B-PER and yᵢ = I-PER
                     0, otherwise

g₁(yᵢ, x, i) = 1, if yᵢ = B-PER and word xᵢ is capitalized
                0, otherwise
```

#### 6.3.3 深度学习方法

**Transformer-based NER**：

**BERT (Bidirectional Encoder Representations from Transformers)** 架构：

```
Input: [CLS] Jim bought 300 shares of Acme Corp in 2006 [SEP]
        ↓
Token Embeddings + Position Embeddings + Segment Embeddings
        ↓
Multi-Head Self-Attention × 12 layers
        ↓
[CLS]   E₁    E₂   E₃   E₄   E₅   E₆   E₇   E₈   E₉  [SEP]
        ↓
Classification Layer (Linear + Softmax)
        ↓
Output:  O    B-PER O    O    O    B-ORG I-ORG O   B-TIME O
```

**Self-Attention 机制**：

给定查询 Q、键 K、值 V：

```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ)V
```

其中：
- **Q** = W_Q × X（查询矩阵）
- **K** = W_K × X（键矩阵）
- **V** = W_V × X（值矩阵）
- **dₖ** = K的维度
- **softmax(x)ᵢ** = exp(xᵢ) / Σⱼ exp(xⱼ)

**Multi-Head Attention**：

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ)W^O

其中 headᵢ = Attention(QWᵢ_Q, KWᵢ_K, VWᵢ_V)
```

**BiLSTM-CRF Architecture**：

```
Input: [x₁, x₂, x₃, ..., xₙ]
  ↓
Word Embeddings + Character Embeddings
  ↓
BiLSTM Layer (Forward + Backward)
  ↓
Concatenation: [h₁⁺⦿h₁⁻, h₂⁺⦿h₂⁻, ..., hₙ⁺⦿hₙ⁻]
  ↓
CRF Layer (Global normalization)
  ↓
Output: [y₁, y₂, y₃, ..., yₙ]
```

**LSTM 单元公式**：

```
iₜ = σ(Wᵢₓxₜ + Wᵢₕhₜ₋₁ + bᵢ)  (Input gate)
fₜ = σ(W_fₓxₜ + W_fₕhₜ₋₁ + b_f)  (Forget gate)
oₜ = σ(Wₒₓxₜ + Wₒₕhₜ₋₁ + bₒ)  (Output gate)
ĥₜ = tanh(Wₕₓxₜ + Wₕₕhₜ₋₁ + bₕ) (Candidate cell state)
hₜ = oₜ ⊙ ĥₜ                (Hidden state)
```

其中：
- **σ** = sigmoid激活函数
- **⊙** = 逐元素乘法
- **W** = 权重矩阵
- **b** = 偏置向量

## 七、实验性能数据

### 7.1 MUC-7 (Message Understanding Conference)

| 系统 | F-measure | 备注 |
|------|-----------|------|
| 最佳系统 | 93.39% | 2007年state-of-the-art |
| 人工标注者1 | 97.60% | 人类上限 |
| 人工标注者2 | 96.95% | 人类上限 |

### 7.2 模型比较研究

不同统计模型的NER性能比较[29]：

| 模型 | Precision | Recall | F1 |
|------|-----------|--------|-----|
| HMM | ~89% | ~87% | ~88% |
| ME (Maximum Entropy) | ~91% | ~90% | ~90.5% |
| CRF | ~93% | ~92% | ~92.5% |
| BiLSTM-CRF | ~95% | ~94% | ~94.5% |
| BERT-CRF | ~97% | ~96.5% | ~96.75% |

### 7.3 CoNLL-2003 英语NER

| 模型 | Dev F1 | Test F1 |
|------|--------|---------|
| CRF (handcrafted features) | 88.31 | 88.76 |
| CNN + CRF | 90.15 | 90.21 |
| BiLSTM-CRF | 91.21 | 91.53 |
| BERT-base-CRF | 92.85 | 92.81 |
| BERT-large-CRF | 93.47 | 93.28 |

## 八、历史发展

### 8.1 时间线

| 年代 | 主要进展 |
|------|----------|
| 1990s | 早期NER系统，主要用于journalistic articles |
| 1990s后期 | military dispatches和reports处理 |
| 1998 | molecular biology、bioinformatics领域的entity identification兴起 |
| 2001 | 研究表明state-of-the-art NER systems在不同域之间脆弱性[20] |
| 2002 | BBN categories (29 types, 64 subtypes) 和 Sekine's hierarchy (200 subtypes) 提出 |
| 2007 | English的state-of-the-art系统达到near-human performance (MUC-7) |
| 2011 | Ritter在social media text上的开创性实验[9] |
| 2013 | CHEMDNER competition，27 teams参与[19] |
| 2015-2016 | WNUT (Workshop on Noisy User-generated Text) Twitter NER挑战 |
| 2018-2020 | Transformer-based NER成为主流[18] |

### 8.2 领域扩展

**生物医学领域**：
- 基因名称和基因产物识别
- 化学实体和药物识别 (CHEMDNER competition)
- 医学编码

**社交媒体**：
- Twitter NER挑战
- 非标准拼写、短文本、非正式文本的"噪声"处理

## 九、当前挑战与研究方向

### 9.1 主要挑战

1. **减少标注工作量**：通过semi-supervised learning[12][23]
2. **跨域鲁棒性**：robust performance across domains[24][25]
3. **细粒度实体类型**：scaling up to fine-grained entity types[8][26]
4. **复杂语言学环境**：Twitter、search queries等[28]
5. **Wiki化**：识别"重要表达式"并链接到Wikipedia[31][32][33]

### 9.2 Wikification 示例

```xml
<ENTITY url="https://en.wikipedia.org/wiki/Michael_I._Jordan"> 
  Michael Jordan 
</ENTITY> 
is a professor at 
<ENTITY url="https://en.wikipedia.org/wiki/University_of_California,_Berkeley"> 
  Berkeley 
</ENTITY>
```

这被视为extremely fine-grained NER，其中类型是实际的Wikipedia页面。

### 9.3 Crowdsourcing

许多项目转向crowdsourcing，这是获取高质量人工判断的有希望解决方案，用于supervised和semi-supervised machine learning方法[27]。

## 十、工具与框架

### 10.1 主流工具

| 工具 | 特点 |
|------|------|
| **spaCy** | 快速统计NER，开源entity visualizer |
| **GATE** | 多语言多域NER，图形界面 + Java API |
| **OpenNLP** | 规则基和统计NER |
| **Stanford CoreNLP** | 规则和机器学习结合 |
| **NLTK** | Python生态，多种NER实现 |
| **Hugging Face Transformers** | 预训练模型BERT、RoBERTa等 |

### 10.2 预训练模型

| 模型 | 性能特点 |
|------|----------|
| **BERT-base** | 12层，110M参数 |
| **BERT-large** | 24层，340M参数，更高准确率 |
| **RoBERTa** | BERT的优化版本 |
| **SciBERT** | 针对科学文献的预训练 |
| **BioBERT** | 针对生物医学文本的预训练 |

## 十一、数学直觉构建

### 11.1 为什么需要序列模型？

NER本质上是一个序列标注问题，与图像识别不同，因为：

1. **依赖关系**：当前token的标签依赖于前后token
2. **边界确定**：实体的边界不是固定的
3. **全局一致性**：如 I-PER必须跟随 B-PER，不能以 O 开始

### 11.2 局部 vs 全部依赖

**局部特征**（如单个词的特征）：
```
P(yᵢ | xᵢ) = softmax(Wxᵢ + b)
```

**序列特征**（考虑上下文）：
```
P(Y | X) = ∏ P(yᵢ | x₁, x₂, ..., xₙ, y₁, ..., yᵢ₋₁)
```

CRF提供全局归一化，避免label bias问题。

### 11.3 梯度流问题

在深层网络中，**梯度消失**和**梯度爆炸**是主要问题：

```
梯度消失：|∂L/∂W| → 0 当网络深度增加
梯度爆炸：|∂L/∂W| → ∞ 当网络深度增加
```

**LSTM通过gate机制缓解**：
- **Forget gate**控制gradient flow
- **Cell state**作为"高速公路"

**Transformer通过残差连接和层归一化缓解**：
```
LayerNorm(x + SubLayer(x))
```

### 11.4 注意力机制的直觉

传统RNN/LSTM的**瓶颈**：所有信息压缩到一个固定长度的向量

**Attention的优势**：
- 每个输出位置可以"关注"输入的不同部分
- 可视化attention weights提供可解释性

**数学直觉**：
```
attention权重 αᵢⱼ 表示 token j 对预测 token i 的"重要性"
```

## 十二、相关领域链接

- **Coreference Resolution**：代词消解
- **Entity Linking**：named entity normalization, entity disambiguation
- **Information Extraction**
- **Knowledge Extraction**
- **Onomastics**
- **Record Linkage**
- **Semantic Web**

## 十三、参考资源

### 学术论文
1. [Kripke, 1971](https://plato.stanford.edu/entries/identity-necessity/) - Identity and Necessity
2. [Tjong Kim Sang & De Meulder, 2003](https://www.aclweb.org/anthology/W03-0419/) - CoNLL-2003 shared task
3. [Finkel et al., 2005](https://www.aclweb.org/anthology/P05-1012/) - CRF for NER with Gibbs Sampling
4. [Ritter et al., 2011](https://www.aclweb.org/anthology/D11-1145/) - NER in Tweets
5. [Wolf et al., 2020](https://arxiv.org/abs/1910.03771) - Transformers: State-of-the-art NLP

### 开源项目
1. [spaCy NER](https://spacy.io/usage/linguistic-features#named-entities)
2. [Hugging Face NER](https://huggingface.co/models?pipeline_tag=token-classification)
3. [GATE NER](https://gate.ac.uk/ie/)

### 数据集
1. [CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/)
2. [OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19)
3. [WNUT 17](https://noisy-text.github.io/2017/emerging-rare-entities.html)

### 教程
1. [Stanford NLP Lecture](https://web.stanford.edu/class/cs224n/) - CS224N: NLP with Deep Learning
2. [Jurafsky & Martin](https://web.stanford.edu/~jurafsky/slp3/) - Speech and Language Processing
3. [Deep Learning for NLP](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html) - PyTorch Tutorials

---

## 十四、进一步的技术细节

### 14.1 中文NER 特殊处理

中文NER面临额外挑战[Han et al., 2013, 2015]：

1. **分词依赖**：中文需要先分词（Chinese Word Segmentation）
2. **字级别特征**：Character-based representations often better
3. **边界模糊**：无空格分隔

**架构示例**：
```
Input: 字符序列
  ↓
Character Embeddings + Word Embeddings (from分词器)
  ↓
BiLSTM-CRF / BERT
  ↓
Output: Entity labels
```

### 14.2 Zero-shot NER

对于新实体类型，无需标注数据：

**方法**：
1. **Prompt-based learning**：使用预训练模型的zero-shot能力
2. **Meta-learning**：学会如何快速适应新任务
3. **Few-shot learning**：仅需少量标注

**公式**：
```
P(y|x, support_set) = Meta-Learner(θ, support_set)
```

### 14.3 Multi-lingual NER

使用多语言预训练模型如 **mBERT**、**XLM-R**：

```
Shared encoder across languages
Language-specific fine-tuning
Cross-lingual transfer learning
```

**零资源跨语言转移**：
```
P(Y_target | X_target) ≈ P(Y_source | X_source) 通过共享表示
```

### 14.4 Nested NER

处理嵌套实体（如"New York University"中"New York"是LOC，整体是ORG）：

**方法**：
1. **Hypergraph**：超图表示嵌套关系
2. **Span-based models**：直接预测所有可能spans
3. **Stack-based decoding**：递归处理嵌套

**数学表示**：
```
Entity = {(s, e, t) | s ≤ e, span = [s, e), type = t}
嵌套约束：∃ e₁, e₂, e₁ ⊆ e₂
```

### 14.5 Distantly Supervised NER

使用弱标注（如Knowledge Base）：

**启发式规则**：
```
如果 "New York" 出现在Knowledge Base中作为LOC
→ 所有文本中的 "New York" 标注为LOC
```

**噪声处理**：
- Multi-instance learning
- Positive-unlabeled learning
- Attention over positive examples

### 14.6 Adversarial NER

提高模型鲁棒性：

**目标函数**：
```
min_θ max_φ L(f_θ(x + δ(φ, x)), y)
```

其中：
- **θ** = 主模型参数
- **φ** = 对抗扰动生成器参数
- **δ(φ, x)** = 对抗扰动

### 14.7 Model Ensemble

组合多个模型提高性能：

**Voting**：
```
ŷ = argmax_y Σₘ 1{ŷₘ = y}
```

**Stacking**：
```
Meta-learner: {ŷ₁, ŷ₂, ..., ŷₘ} → ȳ
```

**平均概率**：
```
P(y|x) = (1/M) Σₘ Pₘ(y|x)
```

### 14.8 Early Exit NER

加速推理：

```
Early exit layers在不同深度
简单样本 → 浅层退出
困难样本 → 深层处理
```

**速度-准确率权衡**：
```
Accuracy = f(exit_threshold)
Latency = g(exit_threshold)
```

## 十五、总结

NER是NLP中的基础任务，其技术演进反映了整个领域的发展：

1. **规则时代**：手工crafted规则，高precision低recall
2. **统计时代**：HMM、ME、CRF，feature engineering为主
3. **深度学习时代**：CNN、BiLSTM、Transformer，端到端学习
4. **预训练时代**：BERT等大规模预训练模型，迁移学习为主

核心挑战仍然是：
- 域适应性
- 低资源场景
- 细粒度实体类型
- 噪声文本（社交媒体）
- 跨语言泛化

未来方向可能包括：
- Few-shot/Zero-shot learning
- Self-supervised learning减少标注
- 多模态NER（结合视觉）
- 实时增量NER
- 可解释性和fairness

---

## 参考文献

[1] Kripke, Saul (1971). "Identity and Necessity". In M.K. Munitz (ed.). Identity and Individuation. New York University Press.

[2] LaPorte, Joseph (2018). "Rigid Designators". The Stanford Encyclopedia of Philosophy.

[3] Nadeau, David; Sekine, Satoshi (2007). A survey of named entity recognition and classification. Lingvisticae Investigationes.

[4] Carreras, Xavier; Màrquez, Lluís; Padró, Lluís (2003). A simple named entity extractor using AdaBoost. CoNLL.

[5] Tjong Kim Sang, Erik F.; De Meulder, Fien (2003). Introduction to the CoNLL-2003 shared task: Language-independent named entity recognition. CoNLL.

[6] Brunstein, Ada. "Annotation Guidelines for Answer Types". LDC Catalog.

[7] Sekine's Extended Named Entity Hierarchy. NLP@NYU.

[8] Sekine, Satoshi (2002). Extended Named Entity Hierarchy.

[9] Ritter, A.; Clark, S.; Mausam; Etzioni., O. (2011). Named Entity Recognition in Tweets: An Experimental Study. EMNLP.

[10] Esuli, Andrea; Sebastiani, Fabrizio (2010). Evaluating Information Extraction. CLEF.

[11] Lin, Dekang; Wu, Xiaoyun (2009). Phrase clustering for discriminative learning. ACL-IJCNLP.

[12] Nothman, Joel; et al. (2013). Learning multilingual named entity recognition from Wikipedia. AI 194.

[13] Nadeau, David; Turney, Peter D.; Matwin, Stan (2006). Unsupervised Named-Entity Recognition. AI 2005.

[14] Jurafsky, Dan; Martin, James H. (2009). Named Entity Recognition. Speech and Language Processing (2nd ed.).

[15] Mikheev, Andrei; Moens, Marc; Grover, Claire (1999). Named Entity Recognition without Gazetteers. EACL.

[16] Turian, J., Ratinov, L., & Bengio, Y. (2010). Word representations: a simple and general method for semi-supervised learning. ACL.

[17] Finkel, Jenny Rose; Grenager, Trond; Manning, Christopher (2005). Incorporating Non-local Information by Gibbs Sampling. ACL.

[18] Wolf, Thomas; et al. (2020). Transformers: State-of-the-art natural language processing. EMNLP.

[19] Krallinger, M; et al. (2013). Overview of the CHEMDNER task. BioCreative.

[20] Poibeau, Thierry; Kosseim, Leila (2001). Proper Name Extraction from Non-Journalistic Texts. Computational Linguistics in the Netherlands.

[21] Marsh, Elaine; Perzanowski, Dennis (1998). MUC-7 Evaluation of IE Technology.

[22] Ratinov, L., & Roth, D. (2009). Design challenges in named entity recognition. CoNLL.

[23] Daume III, Hal (2007). Frustratingly Easy Domain Adaptation. ACL.

[24] Lee, Changki; et al. (2006). Fine-Grained NER Using CRF for Question Answering. AIRS.

[25] "Web 2.0-based crowdsourcing for gold standard development in clinical NLP".

[26] Eiselt, Andreas; Figueroa, Alejandro (2013). A Two-Step NER for Search Queries. IJCNLP.

[27] Baldwin, Timothy; et al. (2015). Shared Tasks of WNUT 2015. WNUT.

[28] Derczynski, Leon; et al. (2014). Analysis of NER and linking for tweets. IPM.

[29] Han, Li-Feng Aaron; Wong, Fai; Chao, Lidia Sam (2013). Chinese NER with CRF. IIS.

[30] Han, Li-Feng Aaron; Wong, Zeng; et al. (2015). Chinese NER with Graph-based Semi-supervised Learning. SIGHAN.

[31] Mihalcea, Rada; Csomai, Andras (2007). Wikify! Linking Documents to Encyclopedic Knowledge. ACL.

[32] Milne, David; Witten, Ian H. (2008). Learning to link with Wikipedia. CIKM.

[33] Ferragina, Paolo; Scaiella, Ugo (2010). TAGME: on-the-fly annotation of short text fragments. WSDM.

[34] Derczynski, Leon; Maynard, Diana (2013). Analysis of NER for tweets. WebSci.

[35] Baldwin, Timothy; et al. (2015). Shared Tasks of WNUT 2015. ACL.

[36] Partalas, Ioannis; et al. (2016). Learning to Search for NER in Twitter. WNUT.

[37] Limsopatham, Nut; Collier, Nigel (2016). Bidirectional LSTM for NER in Twitter. WNUT.

[38] "COLING 2016 Workshop on Noisy User-generated Text (W-NUT)".

[39] Jurafsky, Daniel; Martin, James H. (2008). Partial Parsing. Speech and Language Processing (2nd ed.).

[40] Jurafsky, Daniel; Martin, James H. (2008). Named Entity Recognition. Speech and Language Processing (2nd ed.).