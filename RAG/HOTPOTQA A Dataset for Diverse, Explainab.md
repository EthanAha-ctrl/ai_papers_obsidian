---
source_pdf: HOTPOTQA A Dataset for Diverse, Explainab.pdf
paper_sha256: 1ff47fefeb264db40b96dd9044835bf3c400d318d3272eb86cf6cc3d35697023
processed_at: '2026-08-04T23:59:57-07:00'
target_folder: RAG
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，用大白话再过一遍。

## 这篇 paper 到底在干嘛

你看 SQuAD 那会儿，QA 看起来已经被"解决"了，模型答得很好。但大家心里都清楚：那些题太水了，基本就是从一段话里找一句话抄答案，根本算不上"推理"。后来 TriviaQA、SearchQA 给你一大堆 document，听着吓人，但其实答案还是藏在某一段的一句话里，其他段就是噪音而已。

所以 2018 年那会儿，整个 field 有个尴尬：**没有哪个数据集真正逼模型去做"我得先查 A，用 A 的结果去查 B，再综合出答案"这种多步推理**。

HotpotQA 就是来填这个坑的。

## 怎么保证问题是 multi-hop 的

这是这篇 paper 最聪明的地方。你想想，如果你随便给 crowd worker 两段话，让他编一道需要两段的题，大部分人编出来的还是能用一段话答的题 —— 人偷懒嘛。

他们的招是：**先从 Wikipedia 的 hyperlink graph 里挖**。

比如说文章 A 是讲某个音乐节的，文章 A 的第一段里有个 hyperlink 指向文章 B（比如某个歌手）。那我让 worker 看着这两段，问他："这个音乐节的 MVP 现在在哪个 NBA 球队？" 他就得先从 A 段知道 MVP 是谁（bridge entity），再去 B 段查这人现在效力哪个队。你没法只看一段就答出来，两段都得用。

这个 hyperlink graph 不是随便挖的 —— 他们手动 curated 了 591 个 category，把"美国"这种万能 entity 排除了，因为指向"美国"的 link 太多太杂，编不出有意义的题。

另外还有一类 comparison 题，也是手动搞了 42 个 list（比如 NBA 球员 list、英国摇滚乐队 list），从同一个 list 里抽两个 entity，问 worker 拿这俩编比较题。比如"谁效力过的 NBA 球队更多，Jordan 还是 Kobe？"这种你得两边都数一遍再比。

## Supporting facts —— 这才是论文的 soul

光有难题不够，关键是你得知道模型是"真推理出来的"还是"蒙对的"。

所以他们让 worker **额外标出哪几句话是解题关键**。就是 Figure 1 里那个 "Supporting facts: 1, 2, 4, 6, 7" —— 第 1、2、4、6、7 句是推理必须的，其他句子是 noise。

这个 annotation 厉害在哪？你可以拿它当 strong supervision 训模型，逼模型不光要答对，还得标出"我是靠哪几句话推理的"。然后 evaluation 的时候有个 joint metric：answer 答对 + supporting facts 也标对，才算这一题通过。

这招直接戳破了一个老问题：以前 QA 模型答对了，你不知道它是不是瞎猫碰死耗子。现在它答对了但 supporting facts 标错了，你就知道它在胡来。**explainability 终于可以量化了**。

## 难度分级的小心机

数据收回来之后，他们没直接全用，而是用 baseline model 先跑了一遍，按 model 答得怎么样分三档：

- **train-easy**：其实是单跳题（worker 偷懒了），单拿出来不用
- **train-medium**：model 高置信答对的多跳题
- **train-hard**：model 答错或没把握的多跳题

dev 和 test 只用 hard 那部分。

为什么这么干？因为如果 dev/test 里掺了一堆 model 本来就能答的题，那 leaderboard 就刷不出真本事了。他们想让 benchmark 真正区分"谁解决了一部分 multi-hop"和"谁彻底没解决"。

后来发现 train-medium 里 Type II（交集推理，"哪个 Pirates 球员外号叫 The Cobra" 这种）特别多，占 32%，而 hard set 里只有 15%。说明当时的模型对交集类多跳已经能学到，但对 bridge 类（先查 A 找中间 entity 再去 B 查答案）还是束手无策。这是个很有意思的诊断结果。

## 两个 test setting

- **Distractor**：把 2 段 gold paragraph 混在 8 段 tf-idf 检索来的 distractor 里给你 10 段，让你答。测的是在有噪音的小池子里推理。
- **Full wiki**：把整个 Wikipedia 500 万段都给你，你自己去找哪 10 段相关，再答。测的是"野外"完整能力。

Full wiki 那个 setting 的 retrieval 是个大灾难 —— bigram tf-idf 找回来的 paragraph，两个 gold 平均排名在 300 名开外，Hits@10 才 56%。所以模型在 full wiki 下 F1 从 58 掉到 34，主要不是 reasoning 不行，是**压根没把对的 paragraph 检索进来**。

这其实给后来的 DPR（Dense Passage Retrieval）、ColBERT 这些工作提了个大醒：tf-idf 对 multi-hop 检索彻底不够用，因为 query 里没有 bridge entity 的词，你怎么 lexical match？

## 模型架构

就是 SQuAD 那套 BiDAF + self-attention + char-level，加了一个 3-way classifier 处理 yes/no 题，加了一个 sentence-level binary classifier 预测 supporting facts。都是 multi-task 联合训练。

ablation 里最扎心的一行：

- 拿掉 supporting facts supervision：F1 掉 2 分
- 但如果你 oracle 给模型只看 supporting facts 那几句话：F1 比 baseline 高 10 分

意思是：**supporting facts 信号蕴含的信息量巨大，但他们的利用方式（拿句子首尾 token 拼 a 起来过个 linear classifier）太糙了**，只榨出了 2 分的收益，理论上有 10 分以上的空间。这等于明牌告诉大家"来挖这个坑吧"，后来果然一堆 follow-up（DFGN、HGN、QFE）都是来挖这个的。

## 人 vs 机器

Human 在 distractor 上 answer F1 = 91，模型 69。Joint F1 人类 82，模型 52。差距是实打实的 30 分级别。

更有意思的是 human upper bound —— 多个 annotator 取最优 —— 几乎所有指标都 95+。说明这题对人来说不难，对模型来说难，**task 本身 well-defined**，不是数据有病。

## 我觉得这篇 paper 真正的 legacy

1. **Supporting facts annotation paradigm**：后来的 explainable QA 几乎全沿用这套"标出关键句"的做法。R4C 后来更进一步标成有向 reasoning route，但思想是一脉相承的。

2. **Multi-hop 作为一等公民**：把"多步推理"从 KB-only 的 toy setting 拉到 free-text realistic setting，直接催生了一个 subfield。

3. **Retriever-reader 范式暴露的 retrieval 瓶颈**：full wiki setting 是后来 dense retrieval 兴起的直接导火索之一。

4. **Joint metric 设计**：用乘法耦合 answer 和 explanation 的 F1，简单粗暴但有效，后续被大量沿用。

## 也有它的毛病

后来的 MuSiQue (Trivedi et al. 2022) 狠批了 HotpotQA 一顿：说很多题表面上 multi-hop，但 BERT 这种强 reader 一次 forward 就能答，根本不需要显式多跳 —— 因为 distractor 不够 deceptive，gold paragraph 的特征太明显。所以 MuSiQue 强制每个 supporting fact 必须 answerable，并且把 reasoning chain 显式分解。

Jiang & Bansal 2019 也说过类似的话：HotpotQA 里相当一部分 bridge 题，self-attention 加 BERT 就能糊弄过去，不需要真做 retrieve-then-reason。

不过这是后话了，2018 年那时候 BERT 还没出，HotpotQA 已经是当时 reasoning benchmark 设计的天花板。

## 一句话总结

**HotpotQA 的核心贡献是证明了一件事：你可以用 crowdsourcing + Wikipedia hyperlink graph 系统性地造出真正需要多步推理的自然语言问题，并且通过 sentence-level supporting facts annotation 让"可解释推理"第一次变成可量化的评估目标。** 它的设计哲学比它的 baseline 数字重要得多。

---

# HotpotQA 详解

## 1. Paper 核心动机与定位

HotpotQA 由 Zhilin Yang、Peng Qi 等人于 2018 年提出,定位为 **multi-hop reasoning + explainability** 的 benchmark。在它之前,QA 数据集主要有三个痛点:

1. **Single-hop bias**:SQuAD、TriviaQA、SearchQA 多数问题可在单个 paragraph 内通过 pattern matching 解决
2. **KB-schema 限制**:QAngaroo、ComplexWebQuestions 基于 Freebase/DBpedia,question 多样性被 relation schema 束缚
3. **Distant supervision**:只有 answer label,缺少 reasoning process supervision,模型学到的 reasoning 不可解释

HotpotQA 同时解决这三点:基于 free-text Wikipedia、强制 multi-hop、提供 sentence-level supporting facts。

paper 链接:https://arxiv.org/abs/1809.09600
项目主页:https://hotpotqa.github.io/
leaderboard:https://hotpotqa.github.io/wiki.html

## 2. Data Collection Pipeline 的关键设计

### 2.1 Wikipedia Hyperlink Graph

使用 2017-10-01 的 English Wikipedia dump,通过 WikiExtractor 抽取 text + hyperlinks。构建有向图 $G = (V, E)$,其中 edge $(a, b) \in E$ 表示 article $a$ 的 first paragraph 中存在指向 article $b$ 的 hyperlink。

**为什么用 first paragraph?** 因为 Wikipedia 的 lead section 通常 summary 了 entity 的核心属性(person 的 nationality、band 的 origin 等),这些属性对 multi-hop reasoning 最有用。

### 2.2 Bridge Entity Curation

并非所有 entity $b$ 都适合作为 bridge。例如 country 类 page (如 "United States") 被 inlink 太多且没有 specific 主题;IPv4 这种技术性 page 对 crowd worker 不友好。作者手动 curated 591 个 categories 来自 WikiProject 的 popular pages lists,构成 set $B$。

Candidate pair 生成:$\{(a, b) : (a, b) \in E \land b \in B\}$

### 2.3 Comparison Questions

手动 curated 42 个 lists (如 "List of NBA players"、"List of English rock bands"),记作 $L$。从同一 list 中 uniformly sample 两个 entity $(a, b)$ 让 worker 提问。

进一步引入 yes/no 子集(ratio $r_2 = 0.5$),因为 "Is Iron Maiden or AC/DC from the UK?" 这类问题其实 single-hop 就能答,改为 "Are Iron Maiden and AC/DC from the same country?" 才强制双 paragraph 都要看。

### 2.4 Algorithm 1 全流程

```
Input: bridge/comparison ratio r=0.75, yes/no ratio r2=0.5
loop:
  if random() < r:
    sample b ∈ B uniformly
    sample edge (a, b) with this b
    worker asks question on (a, b)
  else:
    sample list from L weighted by size
    sample (a, b) from list uniformly
    if random() < r2:
      worker asks yes/no question
    else:
      worker asks span-answer question
  worker provides supporting facts
```

注意 $r = 0.75$ 表示 75% 是 bridge type、25% 是 comparison type;comparison 内部一半 yes/no、一半 span。

## 3. 数据 Split 的巧妙设计

这是这篇 paper 容易被忽视但很重要的部分。作者用 baseline model 做了 **three-fold cross validation** 来分离 single-hop 和 multi-hop:

| Split | 性质 | 数量 |
|---|---|---|
| train-easy | mostly single-hop (top-contributing turkers 抽样判定) | 18,089 |
| train-medium | model high-confidence 答对的 multi-hop | 56,814 |
| train-hard | model 答错或低置信度的 multi-hop | 15,661 |
| dev / test-distractor / test-fullwiki | hard multi-hop | 各 7,405 |

**关键 idea**:用 model confidence 来 proxy "question 难度"。Train-medium 之所以 model 能答对,不是因为它 single-hop,而是 Type II (intersection reasoning,占 32%) 比例偏高 —— 这类问题现有 architecture 已经能处理。dev 中 Type II 只占 15%,Type I (bridge reasoning) 占 42%,后者更难。

训练时默认 combine 全部三个 train splits,因为 Appendix C 显示 train-medium 在 full-wiki retrieval 上和 hard examples 一样难。

## 4. Two Benchmark Settings

### Distractor Setting
- 2 gold paragraphs + bigram tf-idf retrieve 8 distractors (用 question 作 query)
- 10 paragraphs shuffled 后 input
- 测 reading comprehension in noisy context

### Full Wiki Setting
- 用 inverted-index filtering 先取 ≤5000 candidates (Algorithm 2)
- 再 bigram tf-idf 取 top-10
- 用 distractor-trained model 在这 10 paragraphs 上推理
- 真正测试 "in-the-wild" multi-hop retrieval + reasoning

**Algorithm 2 核心思想**:通过逐步提高 gram overlap threshold $C_{gram}$,缩小 candidate pool 直到 $|S_{cand}| \leq N = 5000$。这是 efficient first-pass filtering。

Retrieval 性能 (Table 5):
- MAP ~43%
- Mean Rank ~314 (两 gold paragraphs 平均排名)
- Hits@10 ~56%

这个 Mean Rank 数字非常糟糕 —— 说明 bigram tf-idf 对 multi-hop 检索力不从心,这也解释了 full wiki setting 性能 drop。

## 5. Model Architecture

### 5.1 Baseline 结构

基于 Clark & Gardner (2017) 的 SQuAD SOTA architecture,核心组件:

1. **Character-level embedding**:CNN over characters,捕捉 morphological 信息
2. **Word embedding**:GloVe (pre-trained) + char-level concat
3. **Contextual embedding**:BiLSTM over question 和 context
4. **Bi-attention flow** (BiDAF, Seo et al. 2017):question-to-context 和 context-to-question attention
5. **Self-attention** (Wang et al. 2017):context 内部 token 间的 attention,捕捉 long-range dependency
6. **Output layer**:
   - Span start / end position classifier (与 SQuAD 一致)
   - **3-way classifier** (yes / no / span) 处理 yes/no questions —— 这是新增

### 5.2 Supporting Fact 预测 Head

对每个 sentence,取 self-attention layer 在该 sentence 第一个 token 和最后一个 token 的输出,concat 后送入 binary linear classifier:

$$p(\text{sup}_i) = \sigma(W \cdot [h_{start_i}; h_{end_i}] + b)$$

其中:
- $h_{start_i}$:sentence $i$ 第一个 token 在 self-attention 层的 hidden state
- $h_{end_i}$:sentence $i$ 最后一个 token 的 hidden state
- $W \in \mathbb{R}^{1 \times 2d}$,$d$ 是 hidden dim
- $\sigma$:sigmoid

Loss = Binary cross-entropy,与 span loss 多任务联合优化:

$$\mathcal{L} = \mathcal{L}_{span} + \lambda \mathcal{L}_{sup}$$

### 5.3 Why Sentence-level?

作者选 sentence granularity 而非 token 或 paragraph level,理由:
- Token-level:太细,annotation 噪声大
- Paragraph-level:太粗,无法定位 reasoning
- Sentence-level:语义完整 + 易标注 + 与人类解释粒度一致

## 6. 评估指标体系

### 6.1 标准 EM / F1
对 answer span 计算,与 SQuAD 一致。

### 6.2 Supporting Fact EM / F1
把 predicted supporting sentences set 和 gold set 比较,F1 用 set-level precision/recall。

### 6.3 Joint EM / F1

这是这篇 paper 的核心创新 metric,公式:

$$P^{(\text{joint})} = P^{(\text{ans})} \cdot P^{(\text{sup})}$$
$$R^{(\text{joint})} = R^{(\text{ans})} \cdot R^{(\text{sup})}$$
$$\text{Joint } F_1 = \frac{2 P^{(\text{joint})} R^{(\text{joint})}}{P^{(\text{joint})} + R^{(\text{joint})}}$$

变量含义:
- $P^{(\text{ans})}, R^{(\text{ans})}$:answer span 的 precision / recall(对 token-level partial match)
- $P^{(\text{sup})}, R^{(\text{sup})}$:supporting fact sentence set 的 precision / recall
- Joint EM = 1 当且仅当 answer EM=1 且 sup EM=1

**直觉**:用乘法耦合,惩罚 "答对但没解释" 或 "解释对但答错" 的 system。这强制模型同时具备 accuracy 和 explainability。

## 7. 实验结果深度分析

### 7.1 Main Results (Table 4)

| Setting | Split | Ans EM | Ans F1 | Sup EM | Sup F1 | Joint EM | Joint F1 |
|---|---|---|---|---|---|---|---|
| distractor | dev | 44.44 | 58.28 | 21.95 | 66.66 | 11.56 | 40.86 |
| distractor | test | 45.46 | 58.99 | 22.24 | 66.62 | 12.04 | 41.37 |
| full wiki | dev | 24.68 | 34.36 | 5.28 | 40.98 | 2.54 | 17.73 |
| full wiki | test | 25.23 | 34.40 | 5.07 | 40.69 | 2.63 | 17.85 |

Distractor → full wiki 的 Ans F1 从 59 掉到 34,主要 drop 来自 retrieval。Table 6 显示 bridge questions 在 full wiki 下 Br F1 从 59 掉到 30,而 Comparison 仅从 55 掉到 51 —— 因为 comparison question 通常显式包含两个 entity name,IR 容易。

### 7.2 Ablation (Table 7)

| Setting | EM | F1 |
|---|---|---|
| full model | 44.44 | 58.28 |
| − sup fact | 42.79 | 56.19 |
| − sup, self-attn | 41.59 | 55.19 |
| − sup, char | 41.66 | 55.25 |
| − sup, train-easy | 41.61 | 55.12 |
| − sup, train-easy, train-medium | 31.07 | 43.61 |
| gold only | 48.38 | 63.58 |
| sup fact only | 51.95 | 66.98 |

**关键 insight**:
- Strong supervision 仅带来 ~2 F1 gain,作者承认这 "suboptimal"
- Oracle upper bound (sup fact only context) 比 no-sup 高 10+ F1,说明**supporting facts 信号蕴含巨大潜力未被利用**
- 砍掉 train-easy + train-medium 只用 train-hard,F1 掉到 43.61 —— easy data 也有迁移价值

### 7.3 Human Performance (Table 8)

| Setting | Ans EM | Ans F1 | Sup EM | Sup F1 | Joint EM | Joint F1 |
|---|---|---|---|---|---|---|
| Model (distractor) | 60.88 | 68.99 | 30.99 | 74.67 | 20.06 | 52.37 |
| Human (distractor) | 83.60 | 91.40 | 61.50 | 90.04 | 52.30 | 82.55 |
| Human UB | 96.80 | 98.77 | 87.40 | 97.56 | 84.60 | 96.37 |

**注意几点**:
- Human 在 distractor setting 上 Ans F1 = 91.40,模型 68.99,差距 ~22 F1
- Sup fact 的 human EM = 61.50 比 Ans EM 略低 —— 因为 annotator 间对 "哪些是 supporting fact" 分歧大,这本身是 subjectively-defined task
- Human UB 通过取多 annotator 的最大值,几乎所有指标都接近 100% —— 说明 task 是 well-defined 且 achievable,只是 hard

## 8. Multi-hop Reasoning Types (Table 3)

这是 paper 中最具 intuition-building 的部分。作者 hand-label 100 个 dev/test examples:

| Type | 占比 | 描述 |
|---|---|---|
| Type I (Bridge) | 42% | 先找 bridge entity,再回答关于它的问题 |
| Comparison | 27% | 比较两个 entity 的属性 |
| Type II (Intersection) | 15% | 多个 property 同时 filter |
| Type III (Bridge Property) | ~4% | 通过 bridge entity 推断原 entity 的属性 |
| Other (>2 hops) | ~4% | 需要更多 supporting facts |
| Single-hop (noise) | 6% | 标注错误 |
| Unanswerable | 2% | question 有问题 |

### Type I 经典案例

Paragraph A:2015 Diamond Head Classic MVP = Buddy Hield
Paragraph B:Buddy Hield 现在效力于 Sacramento Kings
Q:Which team does the player named 2015 Diamond Head Classic's MVP play for?

**Reasoning chain**:Q → identify bridge "Buddy Hield" via Paragraph A → query Paragraph B → "Sacramento Kings"

### Type II Intersection 案例

Q:Which former member of the Pittsburgh Pirates was nicknamed "The Cobra"?

需要同时满足两个 property:{former member of Pittsburgh Pirates} ∩ {nicknamed "The Cobra"} → Dave Parker

### Type III Bridge Property 案例

Q:What city is the Marine Air Control Group 28 located in?

Paragraph A:MACG-28 based at MCAS Cherry Point
Paragraph B:MCAS Cherry Point located in Havelock, NC

**关键**:Cherry Point 是 bridge,但 question 问的是 MACG-28 的 city —— 推理路径是 MACG-28 → based at → Cherry Point → located in → Havelock

### Comparison 案例

Q:Did LostAlone and Guster have the same number of members?

需要从两段 paragraph 各抽取 member count,然后做 numerical comparison,answer = yes/no。这要求模型具备跨 paragraph 的算术能力 —— 当时几乎所有 QA model 都做不到。

## 9. Paper 的弱点与后续影响

### 9.1 弱点

1. **Single-hop 噪声**:作者承认 ~6% 实际是 single-hop,说明 crowdsourcing 多跳难控
2. **Bridge-heavy bias**:Type I 占 42%,Type III 仅 4%,分布不均
3. **Strong supervision 利用率低**:ablation 显示只 +2 F1,说明 concatenation-of-endpoints 这种 crude 方法浪费了信号
4. **Yes/no questions 处理简单**:3-way classifier 是粗放设计

### 9.2 后续影响

这篇 paper 直接催生了大量 follow-up:

- **DFGN** (Dynamically Fused Graph Network, EMNLP 2019):https://arxiv.org/abs/1905.06978
- **QFE** (Question-Focused Evidence, EMNLP 2019):https://arxiv.org/abs/1906.00600
- **C2QA Reader**:针对 supporting fact 的 hierarchical reasoning
- **HGN** (Hierarchical Graph Network, EMNLP 2020):https://arxiv.org/abs/2010.02870
- **Tu et al. (Select Answer Select Explain)**:https://arxiv.org/abs/1911.05934

更新 SOTA 一直延续到 BERT 时代:
- BERT-based DFGN:Joint F1 ~58
- HGN + BERT-large:Joint F1 ~70+

数据集也衍生出变种:
- **2WikiMultiHopQA** (Kim et al. 2020):https://arxiv.org/abs/2011.01065
- **MuSiQue** (Trivedi et al. 2022):https://arxiv.org/abs/2108.00581

### 9.3 关于 multi-hop reasoning 的争议

后来 Trivedi et al. (2022) 在 MuSiQue 中指出,HotpotQA 中很多 "multi-hop" 实际上可被 strong reader 单 pass 解决,因为 distractor 不够 deceptive。Trivedi 提出 connectedness-decomposed questions 来强制 reasoning。Min et al. (2019) 的 "Compositional Questions" 也讨论了类似问题:https://arxiv.org/abs/1902.00698

Jiang & Bansal (2019) "Self-Reasoning" 显示 HotpotQA 中大量 question 可以通过 self-attention + BERT 直接 answer,无需显式 multi-hop:https://arxiv.org/abs/1901.08535

## 10. 实现细节中的 "暗坑"

### 10.1 Inverted Index Filtering (Algorithm 2)

这个 algorithm 在 appendix 里容易被忽略,但它对 full wiki baseline 极其关键。Variable 含义:

- $r_q$:question $q$ 抽取出的 unigram + bigram 集合
- $D[w]$:包含 n-gram $w$ 的 Wikipedia paragraphs 列表(inverted index)
- $C_{gram}$:overlap 阈值,从 1 开始递增
- $S_{overlap}$:dict 记录每个 doc $d$ 命中的 n-gram 数
- $N$:控制 threshold,默认 5000

**Algorithm 思路**:从最宽松条件(任一 n-gram 命中)开始,如 candidates 太多则提高要求(至少 2 个 n-gram,3 个,...),直到 candidate pool 收缩到 ≤ N。这是 lazy filtering,平衡 efficiency 和 recall。

### 10.2 Bonus Structure

参考 Yang et al. 2018 (Mastering the Dungeon),worker bonus 模式有两种:
1. **Top-K reward**:每 200 examples 给 top-contributor 发奖
2. **Productivity reward**:按 examples/hour 计费

Top-contributor 贡献了 70%+ data,所以 train-easy split 主要从这些 turker 抽样判定 single-hop —— 这本身有 selection bias。

### 10.3 Stanford CoreNLP 的作用

使用 CoreNLP 3.8.0 做 word + sentence tokenization。Sentence boundary 用于 supporting fact 索引(如 Figure 1 的 "1, 2, 4, 6, 7" 是 sentence index)。Token boundary 用于验证 worker 给的 answer span 是完整 token,避免 half-word 错误。

## 11. 一些更广的关联

### 11.1 与 TriviaQA / SearchQA 的本质区别

TriviaQA 的 multi-document 来自 web search,question 本身是 single-hop。HotpotQA 是 question 本身 multi-hop,distractor 是人工检索的。前者测 robustness to long context,后者测 multi-hop reasoning ability。

### 11.2 与 QAngaroo 的对比

QAngaroo (Welbl et al. 2018) 含 WikiHop / MedHop,基于 Wikidata triplets 构造 multi-hop path。每个 question 对应一条 KB path,question diversity 受 KB schema 限制。HotpotQA 直接从 text 提问,question 自然多样。但 KB-based 数据集的 reasoning chain 更 explicit,适合做 graph reasoning 的研究。

### 11.3 与 BERT 时代的对话

BERT (Devlin et al. 2018) 出现后,HotpotQA 的 baseline 数字很快被刷新:
- BERT-large single-paragraph baseline:distractor Ans F1 ~75
- 但 Joint F1 仍 ~50,说明 supporting fact prediction 是独立瓶颈

### 11.4 关于 reasoning 的可解释性

HotpotQA 引入了 "explainability via supporting facts" 范式。后续很多工作沿这条思路:
- Edge probing (Tenney et al. 2019)
- Evidence-aware QA (Perez et al. 2019)
- R4C (Reasoning Routes):https://arxiv.org/abs/1906.05664

R4C 提出 reasoning route 是 supporting facts 的有向链,而非无序 set —— 这是 HotpotQA supporting facts 形式的自然 extension。

## 12. 与 Andrej 个人可能感兴趣的关联点

- **Self-attention 在 multi-hop 中的作用**:本文 Table 7 显示 self-attention 贡献 ~1 F1,说明 long-range dependency modeling 重要但单层 self-attention 不够
- **Multi-task learning 的权重**:paper 没明确给出 $\lambda$ 的值,但后续 QFE work 显示 $\lambda$ 需要仔细 tune
- **Joint metric 设计**:乘法形式 $P_{joint} = P_a \cdot P_s$ 是 conservative 设计,后续 SAE (Select Answer Explain) paper 改成 averaging
- **Distractor 数量**:8 是经验值,Fisch et al. (2019) 探索了 distractor 数量对 difficulty 的影响
- **Retriever-Reader 解耦**:HotpotQA 是早期 retriever-reader 体系的明确 dataset,影响了后续 ORQA、REALM、DPR 等一系列 open-domain QA 工作

## 参考链接

- HotpotQA paper:https://arxiv.org/abs/1809.09600
- HotpotQA official website:https://hotpotqa.github.io/
- HotpotQA GitHub:https://github.com/hotpotqa/hotpot
- DFGN (EMNLP 2019):https://arxiv.org/abs/1905.06978
- HGN (EMNLP 2020):https://arxiv.org/abs/2010.02870
- QFE:https://arxiv.org/abs/1906.00600
- SAE:https://arxiv.org/abs/1911.05934
- R4C:https://arxiv.org/abs/1906.05664
- MuSiQue (analysis of HotpotQA limitations):https://arxiv.org/abs/2108.00581
- Compositional Questions:https://arxiv.org/abs/1902.00698
- Self-Reasoning critique:https://arxiv.org/abs/1901.08535
- TriviaQA:https://arxiv.org/abs/1705.03551
- SearchQA:https://arxiv.org/abs/1704.05179
- QAngaroo (WikiHop):https://arxiv.org/abs/1710.06481
- BiDAF:https://arxiv.org/abs/1611.01603
- Clark & Gardner 2017:https://aclanthology.org/P17-1179/
- ParlAI:https://arxiv.org/abs/1705.06476

这篇 paper 是 NLP reasoning benchmark 设计的一个范本 —— 它的 supporting fact annotation paradigm 至今仍是 explainable QA 的 gold standard。如果你想深入到 implementation 层面,推荐看 HGN 的 GitHub:https://github.com/yuwenzhou/HGN,他们的 code 复现了 HotpotQA 完整 pipeline 含 retrieval + reader。
