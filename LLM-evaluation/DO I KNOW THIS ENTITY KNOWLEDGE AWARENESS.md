---
source_pdf: DO I KNOW THIS ENTITY KNOWLEDGE AWARENESS.pdf
paper_sha256: e59843c25d2c1914053392e947240ab27c9e8c4c729ea2a2ee278c0a33d37f2b
processed_at: '2026-08-18T06:27:30-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇paper

## 这篇paper在搞什么?

一句话: **LLM在回答问题之前, 它自己心里"有数" — 它知道自己知不知道你问的那个东西**。这篇paper就是把这个"有数"的过程找出来了, 证明它真的存在, 而且能被你手动操控。

## 为什么这件事有意思?

你想啊, 你问ChatGPT "LeBron James哪年生的?", 它能答对。你问"Wilson Brown哪年生的?" (一个不存在的人), 它要么说"我不知道", 要么瞎编一个。

那问题来了: **model在生成答案之前, 内部到底是怎么决定走"我知道"这条路还是"我不知道"这条路的?**

之前大家研究factual recall (model怎么recall已知事实) 研究得挺透, 但对hallucination和refusal的机制理解很差。这篇paper填的就是这个gap。

## 怎么找的? — 用SAE挖feature

作者用了**Sparse Autoencoders (SAEs)** 这个工具。SAE是啥? 简单说, LLM内部的representation是一大坨数字 (几千维), 但这些数字混在一起 (superposition), 你看不懂。SAE把这坨数字拆成更多但更sparse的"feature", 每个feature是一个方向 (direction), 对应一个可解释的概念。

具体来说, 用的是Gemma Scope的JumpReLU SAE。公式不细讲了, 核心就是:

$$\mathbf{x} \approx \sum_j a_j(\mathbf{x}) \mathbf{W}_{\text{dec}}[j, :]$$

- $\mathbf{x}$: 模型内部某层的residual stream (hidden state)
- $a_j(\mathbf{x})$: 第$j$个feature的activation强度
- $\mathbf{W}_{\text{dec}}[j, :]$: 第$j$个feature对应的方向向量

既然representation是这些方向的线性组合, 那你人为调大某个$a_j$, 就等于在residual stream里加上那个方向 — 这叫**steering**:

$$\mathbf{x}^{\text{new}} \leftarrow \mathbf{x} + \alpha \mathbf{d}_j$$

$\alpha$正就是加强, 负就是抑制。

## 实验怎么设计的?

作者从Wikidata拉了4类entity: basketball player, movie, city, song。每类几千个。

每个entity都问model它的属性 (比如player的birthplace, movie的director)。根据model答得对不对, 分成两类:
- **Known**: model至少答对2个属性 — 说明model真认识这entity
- **Unknown**: model全答错 — 说明model不认识

然后在entity的**最后一个token**位置 (比如"LeBron James"的"James"那个token), 看SAE的哪些feature在known entity上fire多, 在unknown上fire少 (或反过来)。

关键指标是**separation score**:

$$s_{l,j}^{\text{known}} = f_{l,j}^{\text{known}} - f_{l,j}^{\text{unknown}}$$

- $f_{l,j}^{\text{known}}$: latent $j$ 在known entity上fire的频率
- $f_{l,j}^{\text{unknown}}$: 在unknown上fire的频率
- $s$大: 这个latent几乎只在known上fire
- $s$负得大: 几乎只在unknown上fire

为了找**跨entity type通用**的feature (不是只对player起作用), 用MaxMin策略:

$$\text{MaxMin}^{\text{known}, l} = \max_j \min_t s_{l,j}^{\text{known}, t}$$

$t$是entity type (player/movie/city/song), $\min_t$取4类里最差的那个分数, $\max_j$再在所有latent里挑最好的。意思就是: 找一个在4类entity上**都至少这么好**的latent。

## 发现了什么?

### 发现1: 真的有"我认识"和"我不认识"这两个方向

Figure 1的scatter plot很直观: x轴是latent在known entity上的activation频率, y轴是在unknown上的。有些latent挤在最左边 (只在unknown上fire), 有些挤在最右边 (只在known上fire)。这说明不是渐变, 是**bimodal** — model内部有明确的binary判断。

而且这俩latent跨entity type都work — 对player, movie, city, song都一致fire。Table 1举的例子很说明问题:

| Known Entity Latent激活 | Unknown Entity Latent激活 |
|---|---|
| Michael Jordan | Michael Joordan (拼错的) |
| LeBron James | Wilson Brown (虚构的) |
| San Francisco | Anthon (不存在的小城) |
| 12 Angry Men | 20 Angry Men (虚构电影) |
| Yellow Submarine | Turquoise Submarine (虚构歌) |

### 发现2: 在中间层最强

Figure 2显示, 这些latent的separation score在layer 9左右达到peak, 然后plateau。这跟之前factual recall circuit的发现吻合 — entity识别在中间层完成, 上层做attribute提取。

### 发现3: 能causally操控model行为

这是最酷的部分。作者做了几个实验:

**实验A — Steering**: 在chat model上, 对unknown entity的问题, 故意加上known entity方向 (steering coefficient $\alpha > 0$), model就开始hallucinate — 原本会说"我不知道"的, 现在瞎编。反过来, 对known entity加unknown方向, model就拒绝回答了, 哪怕是LeBron James这种常识。

**实验B — Orthogonalization**: 把unknown entity方向从model所有weight matrix里project掉:

$$\mathbf{W}_{\text{out}}^{\text{new}} \leftarrow \mathbf{W}_{\text{out}} - \mathbf{W}_{\text{out}} \mathbf{d}^\top \mathbf{d}$$

$\mathbf{d}$是unknown方向 (unit vector), 减去的是$\mathbf{W}_{\text{out}}$在$\mathbf{d}$方向上的component。做完这个手术, model几乎不refuse了 — 说明这个方向是refusal的必要条件, 不是偶然相关。

**最striking的点**: SAE是在base model上train的, 但steering对chat model有效。这说明chat finetuning没从零建refusal机制, 而是**repurpose了base model已有的entity识别能力**。Pretraining已经让model学会"我认不认识这个entity", finetuning只是把这个信号连到"I don't know"的output behavior上。

这跟[Kissane et al. 2024](https://www.alignmentforum.org/posts/YWo2cKJgL7Lg8xWjj/base-llms-refuse-too)发现base model也会refuse是consistent的 — 机制本来就存在, finetuning只是强化和formalize它。

### 发现4: 机制层面 — 通过attention gating

那这个entity recognition方向到底怎么影响后续的? 作者用activation patching找到了完整circuit:

1. **Early layers**: attention heads把entity name的几个token merge到last token (比如"LeBron", "James"的信息都聚到"James"位置)
2. **Middle layers (~layer 9)**: entity recognition latent判断known/unknown, 写入residual stream
3. **Upper layers (L18H5, L20H3等)**: attribute extraction heads从entity last token attend过去, 把attribute拉到final token

关键observation (Figure 4c): 在attribute extraction heads里, **last token对entity token的attention score, known entity显著高于unknown entity**。Model认识的时候, attention强; 不认识的时候, attention弱 — circuit被"关掉"了。

Steering实验causally证实了这个chain (Figure 4 d, e, f):
- 对known entity加unknown方向 → attention to entity **降低**
- 对unknown entity加known方向 → attention **升高**
- Random vector → 无效果

所以完整故事是:

$$\text{Entity recognition latent} \rightarrow \text{修改entity token的key} \rightarrow \text{影响下游attention} \rightarrow \text{attribute是否被extract} \rightarrow \text{recall vs hallucinate}$$

### 发现5: 还有"我不确定"方向能预测hallucination

除了entity token位置的recognition方向, 作者还找了**答案之前**的uncertainty方向。

在end-of-instruction token (即`<start_of_turn>model\n`, 聚合了整个question信息的位置), 对比correct answer和incorrect answer的SAE activations, 用t-statistic筛选:

$$\text{t-statistic}_{l,j} = \frac{\mu(a_{l,j}^{\text{correct}}) - \mu(a_{l,j}^{\text{error}})}{\sqrt{\frac{\sigma(a_{l,j}^{\text{correct}})^2}{n^{\text{correct}}} + \frac{\sigma(a_{l,j}^{\text{error}})^2}{n^{\text{error}}}}}$$

分子是correct和error两组activation均值之差, 分母是Welch's t-test的标准误。

找到的top latent作为classifier, **AUROC 73.2, F1 72** — 在model生成答案之前就能预测它会不会错。

而且这个latent在Neuropedia上查, 发现它在large corpus上fire的句子都是关于uncertainty的:
- "the cause of the fire remains under investigation"
- "His condition was not disclosed"
- "platforms TBA"

它promote的top tokens也是"unknown", "undetermined", "TBA"这种。所以这确实是uncertainty feature, 不是偶然。

**Practical意义**: 如果你在generation前监控这个latent, 就能detect即将到来的hallucination, 提前干预。

### 发现6: 不是token likelihood的confound

一个合理的怀疑: known entity的tokens本身更predictable (training data里见得多), 那这些latent会不会只是反映token predictability?

作者在FineWeb上测了latent activation和ground-truth next-token probability $p(t_i | t_{<i})$ 的correlation (Table 9):

| Model | Latent | Correlation |
|---|---|---|
| Gemma 2 2B | Known | 0.067 |
| Gemma 2 2B | Unknown | -0.000 |
| Gemma 2 9B | Known | 0.062 |
| Llama 3.1 8B | Known | 0.003 |

Correlation极低。所以这些latent不是在encode "这个token好不好预测", 而是真的在encode "model认不认识这个entity" — 是更高层的semantic判断, 不是低层statistical regularity。

## 这paper好在哪?

对我来说, 这paper的beauty在于它讲了一个**完整的多层级故事**:

1. **Representation层**: SAE latent是可解释的feature, 不是black box
2. **Circuit层**: 这些feature通过modulate attention来gate factual recall circuit
3. **Behavior层**: 直接对应refusal vs hallucination行为
4. **Developmental层**: Pretraining学机制, finetuning repurpose机制 — 不从零建

这种"从feature到circuit到behavior到训练动力学"的完整narrative, 正是mechanistic interpretability追求的holy grail。而且method是general的 — 你可以用同样方法找任何binary或continuous feature的neural substrate。

## 有什么caveat?

- Known/unknown分类用fuzzy string matching, 有labeling noise
- Binary简化, 但知识其实是continuous的 (partial knowledge)
- 跨model一致性不完美 (Gemma 2 9B效果比2B弱, Llama的known latent steering没显著降refusal)
- 只测了4类entity, 没覆盖abstract concept
- 作者自己也caution: entity recognition不等于model有general self-knowledge, 可能只对factual recall这个specific mechanism有效

## 一句话总结

**LLM在中间层有一个内部的"我认识不认识这个entity"的binary switch, 这个switch通过调节attention来决定是recall还是refuse, chat finetuning把这个switch连到了explicit的"I don't know"输出上, 而hallucination就是这个switch失效的时候**。

参考资料:
- [Paper本身 (ICLR 2025)](https://openreview.net/forum?id=DoIKnowThisEntity)
- [Gemma Scope SAEs](https://arxiv.org/abs/2408.05147)
- [JumpReLU SAE architecture](https://arxiv.org/abs/2407.14435)
- [Arditi et al. - Refusal mediated by single direction](https://openreview.net/forum?id=pH3XAQME6c)
- [Nanda et al. - Factual recall circuit](https://www.alignmentforum.org/posts/iGuwZTHWb6DFY3sKB/fact-finding-attempting-to-reverse-engineer-factual-recall)
- [Geva et al. - Dissecting factual recall](https://aclanthology.org/2023.emnlp-main.751/)
- [Neuronpedia - SAE exploration平台](https://www.alignmentforum.org/posts/BaEQoxHhWPrkinmxd/announcing-neuronpedia-platform-for-accelerating-research)
- [Anthropic - Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
- [Kissane et al. - Base LLMs refuse too](https://www.alignmentforum.org/posts/YWo2cKJgL7Lg8xWjj/base-llms-refuse-too)
- [Kossen et al. - Semantic entropy probes](https://arxiv.org/abs/2406.15927)

---

# "Do I Know This Entity?" — Knowledge Awareness and Hallucinations in Language Models 深度解析

## 1. 核心问题与动机

这篇paper触及了一个非常fundamental的问题: **当LLM被问到一个它不知道的entity时, 它内部到底发生了什么?** Hallucination一直是LLM部署的critical bottleneck, 尤其在healthcare、legal等high-stakes domain. 但mechanistic interpretability community之前主要focus在**已知事实如何被recall** ([Geva et al. 2023](https://aclanthology.org/2023.emnlp-main.751/); [Nanda et al. 2023](https://www.alignmentforum.org/posts/iGuwZTHWb6DFY3sKB/fact-finding-attempting-to-reverse-engineer-factual-recall); [Chughtai et al. 2024](https://arxiv.org/abs/2402.07321)), 而对**hallucination或refusal的发生机制**理解甚少.

作者的核心假设是: model可能存在一种**self-knowledge** — 关于自己是否知道某个entity的内部表征. 如果这种表征存在, 那么hallucination就不是random failure, 而是这种self-knowledge信号未能正确传导到output的failure mode.

**Intuition building point**: 想象你作为人类被问到一个不认识的运动员的生日. 你的大脑会先经历一个"我认识这个人吗?"的快速判断, 然后才决定是guess还是say "I don't know". 这篇paper就是在reverse engineer LLM里这个"判断步骤"的neural substrate.

## 2. 背景: Sparse Autoencoders 与 JumpReLU

### 2.1 为什么需要SAEs?

LLM的residual stream维度通常几千维, 但里面的features数量远超这个维度 — 这就是[superposition](https://transformer-circuits.pub/2023/monosemantic-features/index.html)现象. 一个neuron可能同时编码多个feature, 一个feature也可能分散在多个neuron上. SAEs通过projecting到更高维的sparse space, 试图disentangle这些features, 让每个latent变得monosemantic (单一含义).

### 2.2 JumpReLU SAE 公式详解

这篇paper用的是[Gemma Scope](https://arxiv.org/abs/2408.05147)的JumpReLU SAEs ([Rajamanoharan et al. 2024](https://arxiv.org/abs/2407.14435)).

**Encoder**:
$$a(\mathbf{x}) = \text{JumpReLU}_\theta(\mathbf{x} \mathbf{W}_{\text{enc}} + \mathbf{b}_{\text{enc}}) \tag{2}$$

其中:
- $\mathbf{x} \in \mathbb{R}^d$: input residual stream representation (维度d, 例如Gemma 2 2B是2304维)
- $\mathbf{W}_{\text{enc}}$: encoder weight matrix, shape $d \times d_{\text{SAE}}$ (project到更高维, 例如$d_{\text{SAE}} = 16384$或更大)
- $\mathbf{b}_{\text{enc}}$: encoder bias, shape $d_{\text{SAE}}$
- $\theta$: learnable threshold vector, shape $d_{\text{SAE}}$ — 每个latent有自己的threshold

**JumpReLU activation function**:
$$\text{JumpReLU}_\theta(\mathbf{x}) = \mathbf{x} \odot H(\mathbf{x} - \theta)$$

其中:
- $H$: Heaviside step function ($H(z) = 1$ if $z \geq 0$, else $0$)
- $\odot$: element-wise product
- Intuition: 低于threshold直接归零(不是ReLU那种渐变), 高于threshold保留原值并在threshold处有一个discontinuous jump. 这种hard thresholding强制sparsity, 让latent activations更"决策性" — 要么明显active要么完全silent.

**Decoder**:
$$\text{SAE}(\mathbf{x}) = a(\mathbf{x}) \mathbf{W}_{\text{dec}} + \mathbf{b}_{\text{dec}} \tag{1}$$

其中:
- $\mathbf{W}_{\text{dec}}$: decoder weight matrix, shape $d_{\text{SAE}} \times d$, 每一行 $\mathbf{W}_{\text{dec}}[j, :]$ 是一个latent direction
- $\mathbf{b}_{\text{dec}}$: decoder bias
- Reconstruction: $\mathbf{x} \approx \sum_j a_j(\mathbf{x}) \mathbf{W}_{\text{dec}}[j, :]$

**Training loss**:
$$\mathcal{L}(\mathbf{x}) = \underbrace{\|\mathbf{x} - \text{SAE}(\mathbf{x})\|_2^2}_{\mathcal{L}_{\text{reconstruction}}} + \underbrace{\lambda \|a(\mathbf{x})\|_0}_{\mathcal{L}_{\text{sparsity}}} \tag{3}$$

其中:
- $\|\cdot\|_0$: L0 norm (非零元素个数, 严格sparse penalty)
- $\lambda$: sparsity coefficient
- L0是non-differentiable的, 但JumpReLU的discontinuous jump使其approximate可train (具体技巧见原JumpReLU paper)

### 2.3 SAE Steering

关键观察: 既然 $\mathbf{x} \approx \sum_j a_j(\mathbf{x}) \mathbf{W}_{\text{dec}}[j, :]$, 那么人为调整某个latent的activation $a_j(\mathbf{x})$ 等价于在residual stream上加/减对应的decoder direction:

$$\mathbf{x}^{\text{new}} \leftarrow \mathbf{x} + \alpha \mathbf{d}_j \tag{4}$$

其中:
- $\mathbf{d}_j = \mathbf{W}_{\text{dec}}[j, :]$: 第j个latent direction
- $\alpha$: steering coefficient (正: 激活该feature; 负: 抑制该feature)

这就是[activation steering](https://arxiv.org/abs/2308.10248)的SAE版本 — 比传统的random direction steering更interpretable, 因为$\mathbf{d}_j$对应一个具体的、可解释的feature.

## 3. 方法论: 构建Known/Unknown Entity Dataset

### 3.1 数据收集

从[Wikidata](https://cacm.acm.org/research/wikidata/)提取4类entity及其attributes:

| Entity Type | 数量 | Attributes |
|------------|------|-----------|
| Player (basketball) | 7487 | Birthplace, birthdate, teams |
| Movie | 10895 | Director, screenwriter, release date, genre, duration, cast |
| City | 7904 | Country, population, elevation, coordinates |
| Song | 8448 | Artist, album, publication year, genre |

### 3.2 Known vs Unknown分类

对每个entity $e_i$, 模型被prompt询问其attributes (template如下):

```
The movie {entity_name} was directed by ___
```

用fuzzy string matching评估correctness. 设阈值 $\tau = 1$:
- **Known**: 至少2个attributes正确
- **Unknown**: 全部attributes错误
- In-between的entity被discard

**Caveat**: 作者承认这有labeling noise — model可能"猜对"unknown entity的某个attribute (例如entity name暗示了location), 或者model知道entity但failed to recall我们specific测试的attribute. 但目标是reasonable differentiation而非perfect classification.

### 3.3 Latent Separation Scores — 核心筛选机制

对每个layer $l$, 每个latent $j$, 在known和unknown entity prompts上计算activation频率:

$$f_{l,j}^{\text{known}} = \frac{\sum_i^{N^{\text{known}}} \mathbb{1}[a_{l,j}(\mathbf{x}_{l,i}^{\text{known}}) > 0]}{N^{\text{known}}} \tag{6}$$

$$f_{l,j}^{\text{unknown}} = \frac{\sum_i^{N^{\text{unknown}}} \mathbb{1}[a_{l,j}(\mathbf{x}_{l,i}^{\text{unknown}}) > 0]}{N^{\text{unknown}}}$$

其中:
- $N^{\text{known}}$, $N^{\text{unknown}}$: known/unknown prompts的总数
- $\mathbb{1}[\cdot]$: indicator function (条件为真返回1)
- $\mathbf{x}_{l,i}^{\text{known}}$: 第i个known prompt在layer $l$ 的residual stream, 位置是**entity的最后一个token** (这是关键 — 作者hypothesize entity recognition发生在entity token位置, 而非final token)

**Separation scores**:
$$s_{l,j}^{\text{known}} = f_{l,j}^{\text{known}} - f_{l,j}^{\text{unknown}}$$
$$s_{l,j}^{\text{unknown}} = f_{l,j}^{\text{unknown}} - f_{l,j}^{\text{known}}$$

High $s^{\text{known}}$ 意味着该latent几乎只在known entity上fire; high $s^{\text{unknown}}$ 则相反.

### 3.4 Cross-Entity-Type Generalization: MaxMin Score

为了找到**跨entity type泛化**的latent (而非某个specific type的), 作者定义:

$$\text{MaxMin}^{\text{known}, l} = \max_j \min_t s_{l,j}^{\text{known}, t} \tag{7}$$

其中:
- $t$: entity type index (players, movies, cities, songs)
- $\min_t$: 在所有entity type上取最小值 — 衡量该latent的"最弱表现"
- $\max_j$: 在所有latent中选最弱表现最强的那个

Intuition: 一个真正general的entity recognition latent应该对所有4种entity type都有高separation score. MaxMin找的就是这种"在所有type上都至少这么好"的latent.

最终选择:
$$\text{known entity latent} = \arg\max_{l,j} \min_t s_{l,j}^{\text{known}, t}$$

外加一个filtering step: 排除在[Pile](https://arxiv.org/abs/2101.00027) random tokens上activation频率>2%的latent (避免选到generic activations).

## 4. 主要发现: Entity Recognition Directions

### 4.1 Scatter Plot 发现

Figure 1的scatter plot显示了每个latent在known vs unknown entity上的activation频率. 关键观察: **存在一些latent几乎exclusively在known entity上fire, 另一些几乎exclusively在unknown entity上fire**. 这不是gradient, 而是bimodal — 强烈暗示model内部有binary的"我认识/不认识这个entity"信号.

### 4.2 Layer-wise Pattern

Figure 2展示了Top 5 latents的separation score随layer变化:
- Early layers: separation score低 (entity还没被recognize)
- **Layer ~9 (Gemma 2 2B)达到peak**, 然后plateau
- Middle layers最优

这与[factual recall circuit](https://aclanthology.org/2023.emnlp-main.751/)的发现一致 — entity processing在middle layers完成, upper layers做attribute extraction.

MaxMin (红色线)同样在middle layers最高, 说明general latents集中在intermediate layers. **Hierarchical organization hypothesis**: early layers有specialized但lower-quality的entity representations, middle layers才emerge generalized, entity-type-agnostic的recognition features.

### 4.3 Llama 3.1 8B Replication

作者用[LlamaScope](https://arxiv.org/abs/2410.20526)在Llama 3.1 8B上重复实验, 发现同样pattern (Appendix Q). 这种cross-architecture replication增强了发现的可信度.

### 4.4 Post-Cutoff Songs Validation

为了进一步验证, 作者测试了283首2024年8月之后发布的songs (model knowledge cutoff之后). 结果:

| Model | Known Latent Activation | Unknown Latent Activation |
|-------|------------------------|---------------------------|
| Gemma 2 2B | 6% | 53% |
| Gemma 2 9B | 22% | 55% |
| Llama 3.1 8B | 13.4% | 76% |

Unknown latent在post-cutoff songs上activation频率显著更高 — 正如预期. (当然, 部分songs可能在pretraining data里, 但pattern consistent across models增强了confidence.)

## 5. Causal Experiments: 这些Directions真的控制Refusal吗?

### 5.1 Knowledge Refusal定义

作者区分了**knowledge refusal** (因缺乏信息而拒绝回答) 与safety refusal. Knowledge refusal的detection用string matching on常见refusal completions (例如 "Unfortunately, I don't have access to real-time information...").

### 5.2 Steering实验

Chat model上对100个unknown entity questions进行steering (在entity last token + 后续end-of-instruction tokens上):

**结果** (Figure 3 left, Gemma 2 2B):
- **Original model**: ~30-60% refusal (因entity type而异)
- **Steer with unknown latent**: ~100% refusal across all types — 强制model拒绝
- **Steer with known latent**: 略降refusal rate — 强制model hallucinate
- **Orthogonalized model** (project掉unknown direction): 大幅降refusal — 该direction确实是refusal的必要component
- **Random latent steering**: 几乎无effect — 排除random direction的可能性

**Example** (Figure 3 right): 对well-known篮球运动员LeBron James, 用unknown latent steering后model拒绝回答, 给出 "I don't have access to real-time information..." 类型的response. 反之, 对虚构运动员Wilson Brown (原model会refuse), 用known latent steering后model开始hallucinate birthplace.

### 5.3 Orthogonalization: Weight Surgery

公式(9):
$$\mathbf{W}_{\text{out}}^{\text{new}} \leftarrow \mathbf{W}_{\text{out}} - \mathbf{W}_{\text{out}} \mathbf{d}^\top \mathbf{d} \tag{9}$$

其中:
- $\mathbf{W}_{\text{out}}$: 任意写向residual stream的weight matrix (例如attention的$W_O$, MLP的$W_{\text{down}}$)
- $\mathbf{d}$: 要orthogonalize的direction (unit vector)
- $\mathbf{d}^\top \mathbf{d}$: 注意这里如果$\mathbf{d}$是unit vector, $\mathbf{d}^\top \mathbf{d}$应该理解为外积 $\mathbf{d} \otimes \mathbf{d}$ (即projection matrix $\mathbf{d}\mathbf{d}^\top$, shape $d \times d$)
- 减去的项 $\mathbf{W}_{\text{out}} (\mathbf{d} \otimes \mathbf{d})$ 是$\mathbf{W}_{\text{out}}$在$\mathbf{d}$方向上的component

直观上: 这个operation让$\mathbf{W}_{\text{out}}$的每一行都变得perpendicular to $\mathbf{d}$, 从而model完全无法写入该direction. 这是[Arditi et al. 2024](https://openreview.net/forum?id=pH3XAQME6c)用于safety refusal的方法的adaptation.

**Key insight**: Orthogonalized model几乎不refuse — 证明unknown entity direction是refusal mechanism的critical component, 而非spurious correlation.

### 5.4 Base Model SAE → Chat Model Behavior

这是paper最striking的发现之一: **SAEs训练在base model上, 但找到的directions对chat model的refusal有causal effect**. 

这呼应了[finetuning repurposes existing mechanisms](https://openreview.net/forum?id=A0HKeK14N1)的hypothesis ([Jain et al. 2024](https://openreview.net/forum?id=A0HKeK14N1); [Prakash et al. 2024](https://arxiv.org/abs/2402.14811); [Kissane et al. 2024](https://www.alignmentforum.org/posts/YWo2cKJgL7Lg8xWjj/base-llms-refuse-too)). Base model已经学到了"识别known/unknown entity"的能力 (可能因为pretraining data里有很多"I don't know about X"的text), chat finetuning只是repurpose这个existing mechanism, 将其连接到显式的refusal behavior.

**Intuition**: 这就好比人类pretraining学到"识别不熟悉的概念"的metacognitive能力, education只是教会我们把这种metacognition表达为"I don't know"的言语行为. Brain circuit已经存在, education只是wiring up output.

## 6. Mechanistic Analysis: Attention Circuit层面

### 6.1 Activation Patching Setup

使用[denoising setup](https://arxiv.org/abs/2404.15255):
- **Clean run**: known entity prompt (例如 "The player LeBron James...")
- **Corrupted run**: unknown entity prompt (例如 "The player Wilson Brown...")
- **Patching**: 把clean run的某个intermediate activation (residual stream或attention head output) 替换到corrupted run的对应位置
- **Metric**: logit difference recovery

$$\text{recovery} = \frac{\text{logit}_{\text{Lakers-Warriors}}(\text{corr} | \text{do}(\mathbf{x}^{\text{unknown}} \leftarrow \mathbf{x}^{\text{known}}))}{\text{logit}_{\text{Lakers-Warriors}}(\text{clean})} \tag{13}$$

其中:
- $\text{logit}_{\text{Lakers-Warriors}}$: correct answer (Lakers)的logit减去corrupted answer (Warriors, 即unknown entity的hallucinated attribute)的logit
- $\text{do}(\cdot)$: [Pearl's do-operator](https://www.cambridge.org/core/books/causality/C5E5FE16D5576A4F1B6A4EC49B8A8A5D), 表示intervention
- Recovery越接近1, 说明patched activation越能restore correct behavior

### 6.2 Factual Recall Circuit (Replicated)

作者replicate [Nanda et al. 2023](https://www.alignmentforum.org/posts/iGuwZTHWb6DFY3sKB/fact-finding-attempting-to-reverse-engineer-factual-recall)的发现, 在Gemma 2 2B/9B上找到类似circuit (Figure 4 a, b):

1. **Early attention heads**: merge entity name tokens到entity last token (例如把"LeBron", "James"的信息aggregate到"James"位置)
2. **Downstream attribute extraction heads**: 从entity last token读取信息, 通过attention把相关attributes移到final token位置

例如Table 7中, L18H5和L20H3在Gemma 2 2B上是attribute extraction heads — 对"Kawhi Leonard"它们promote "Clippers", "Raptors", "NBA"等tokens.

### 6.3 Attention Disparity: Known vs Unknown

Figure 4c显示: 在attribute extraction heads (如L18H5, L20H3), **last token对entity last token的attention score在known entity上显著高于unknown entity**.

Intuition: 当model recognize一个entity时, attribute extraction heads会强烈attend to entity tokens以pull out attributes; 当model不recognize时, attention减弱 — circuit被"关闭"了.

### 6.4 Steering Causally Modulates Attention

Figure 4 d, e, f展示了steering对attention score的causal effect:
- **Steer with unknown latent** (on known entity prompt): attention to entity **降低** (d)
- **Steer with known latent** (on unknown entity prompt): attention to entity **升高** (e)
- **Random vector baseline**: 几乎无effect (f)

这给出了完整的causal chain:

$$\text{Entity recognition latent} \rightarrow \text{Attention score to entity} \rightarrow \text{Attribute extraction} \rightarrow \text{Hallucinate vs Recall}$$

**Mechanistic hypothesis**: entity recognition directions通过影响entity last token的keys (而非values或queries), 调节downstream attention heads能多大程度上"找到"entity信息. 当unknown latent active时, entity token的key representation被modify, 使得attribute extraction heads难以attend到它, circuit被disabled.

### 6.5 Statistical Significance

Appendix M显示, 与10个random SAE latents对比:
- Gemma 2 2B: known latent显著增加attention (10/10 cases), unknown latent显著降低 (9/10)
- Gemma 2 9B: known (10/10), unknown (1/10 for top latent, 但second latent 9/10)
- Llama 3.1 8B: known (7/10), unknown (10/10)

这种statistical robustness排除了"任何SAE direction都能影响attention"的可能性.

## 7. Self-Knowledge Reflection

### 7.1 Explicit Uncertainty Expression

除了implicit refusal behavior, 作者还测试了explicit self-knowledge:

```
Are you sure you know the {entity_type} {entity_name}? Answer yes or no.
```

Steering entity last token, 测量logit difference (Yes - No).

Figure 5结果:
- **Steer known entity with unknown latent**: logit difference降低 (model更倾向say "No")
- **Steer unknown entity with known latent**: logit difference升高 (model更倾向say "Yes")
- Effect size较小 (作者note model对unknown entity有inherent bias toward "Yes", 见[Yona et al. 2024](https://arxiv.org/abs/2405.16908))

**Intuition**: 这表明entity recognition latents不只在implicit behavior (refusal)上起作用, 也在explicit metacognitive judgment ("do I know this?")上有微妙effect. 但effect size小, 说明explicit self-knowledge expression还有其他components.

## 8. Uncertainty Directions: 预测Hallucination

### 8.1 Motivation

除了entity recognition (在entity token位置), 作者还search for**在answer之前**表示uncertainty的directions — 这些能predict即将到来的error.

### 8.2 Setup

Focus on **end-of-instruction token** (model token `<start_of_turn>model\n`), 因为它aggregate整个question的信息 ([Marks & Tegmark 2023](https://arxiv.org/abs/2310.06824)).

排除refusal cases, 只留correct vs incorrect answers.

### 8.3 t-statistic筛选

对每个latent $j$ 在layer $l$:

$$\text{t-statistic}_{l,j} = \frac{\mu(a_{l,j}(\mathbf{x}_l^{\text{correct}})) - \mu(a_{l,j}(\mathbf{x}_l^{\text{error}}))}{\sqrt{\frac{\sigma(a_{l,j}(\mathbf{x}_l^{\text{correct}}))^2}{n^{\text{correct}}} + \frac{\sigma(a_{l,j}(\mathbf{x}_l^{\text{error}}))^2}{n^{\text{error}}}}} \tag{12}$$

其中:
- $\mu(\cdot)$: sample mean
- $\sigma(\cdot)$: sample standard deviation
- $n^{\text{correct}}, n^{\text{error}}$: correct/incorrect样本数
- 分母是[Welch's t-test](https://en.wikipedia.org/wiki/Welch%27s_t-test)的标准误 (不假设equal variance)

High positive t-statistic: 该latent在correct answers上activation显著更高 (knowledge signal)
High negative t-statistic: 该latent在errors上activation显著更高 (uncertainty signal)

同样用MaxMin across entity types找general latents.

### 8.4 结果

在Gemma 2B IT layer 13的SAE上找到top "unknown" latent:
- **AUROC = 73.2** (作为correct vs incorrect classifier)
- **F1 = 72** (after threshold calibration)
- Figure 6 left: correct vs incorrect answers在该latent activation上clear separation

**Neuropedia validation** (Table 2): 该latent在large corpus上fire的maximally activating examples都涉及uncertainty/disclosed information:
- "the cause of the fire remains under investigation"
- "His condition was not disclosed"
- "platforms TBA"

Figure 6 right: 该latent的top promoted tokens包括"unknown", "undetermined", "TBA"等 — 强烈confirming其uncertainty语义.

**Intuition**: 这意味着在model生成answer之前, residual stream里已经存在一个"我不确定"的signal. 如果我们能detect这个signal (例如通过probe或SAE latent monitoring), 就能在生成前predict并prevent hallucination. 这与[semantic entropy probes](https://arxiv.org/abs/2406.15927)和[CH-Wang et al. 2024](https://aclanthology.org/2024.findings-acl.260/)的direction一致, 但这里用SAE提供了更interpretable的feature.

## 9. Token Likelihood Hypothesis反驳

### 9.1 Confounding Concern

一个重要alternative explanation: **entity recognition latents可能只是encoding token likelihood**, 而非真正的knowledge awareness. Known entities (如"LeBron James")的tokens在training data里更frequent、更predictable, 所以latent可能只是反映这种predictability.

### 9.2 测试

在[FineWeb](https://arxiv.org/abs/2406.17557)上计算每个token位置的:
- Entity recognition latent activations
- Ground-truth next-token probability $p(t_i | t_{<i})$

如果token likelihood hypothesis为真, 应该看到strong correlation.

### 9.3 结果 (Appendix S, Table 9)

| Model | Latent | Activation Frequency | Correlation with $p(t_i | t_{<i})$ |
|-------|--------|---------------------|-----------------------------------|
| Gemma 2 2B | Known | 0.006 | 0.067 |
| Gemma 2 2B | Unknown | 0.005 | -0.000 |
| Gemma 2 9B | Known | 0.009 | 0.062 |
| Gemma 2 9B | Unknown | 0.009 | 0.009 |
| Llama 3.1 8B | Known | 0.002 | 0.003 |
| Llama 3.1 8B | Unknown | 0.026 | 0.047 |

**Findings**:
- Latents非常sparse (0.2%-2.6% activation frequency)
- Correlations极低 (最大0.067)
- 即使unknown latent active的tokens有稍低的prediction probability, effect modest

**Conclusion**: Token predictability alone无法解释entity recognition latents的行为, 支持这些latents encode更sophisticated knowledge awareness的interpretation.

## 10. 综合Intuition与Implications

### 10.1 完整的Mechanistic Story

整合所有findings, 一个coherent的picture emerge:

1. **Pretraining**: Model在大量text上学习, 包括"I don't know about X", "X is a famous..."等pattern. 通过统计学习, middle layers emerge出general的entity recognition features (known latent和unknown latent).

2. **Entity Processing**: 当prompt包含一个entity时, early layers merge entity tokens, middle layers (layer ~9)的entity recognition latents判断"我认识这个entity吗?", 写入residual stream的direction.

3. **Circuit Gating**: 这个direction影响entity last token的key representation, 从而modulate downstream attribute extraction heads能多大程度上attend to entity. Known latent: attention高, attributes被extracted. Unknown latent: attention低, extraction失败.

4. **Finetuning Repurposing**: Chat finetuning (RLHF等) connect这个existing entity recognition signal到explicit refusal behavior. 当unknown latent active, chat model生成"I don't have access to..."类refusal.

5. **Hallucination Failure Mode**: 当unknown entity的recognition信号未能正确触发refusal (例如unknown latent activation不够强), attribute extraction heads可能仍然attend到entity并generate plausible-sounding but wrong attributes — 这就是hallucination.

6. **Pre-answer Uncertainty**: 即使model决定回答 (不refuse), end-of-instruction token位置仍encode一个uncertainty signal, 这个signal能predict即将到来的error.

### 10.2 与Related Work的Connection

- **[Gottesman & Geva 2024](https://aclanthology.org/2024.emnlp-main.232/)**: Probe trained on entity residual streams correlates with answer accuracy. 这篇paper提供了mechanistic basis — entity recognition directions就是probe detect的signal.
- **[Yuksekgonul et al. 2024](https://openreview.net/forum?id=gfFVATffPd)**: Link between attention to entity tokens and factual accuracy. 这篇paper causally demonstrate这个link的mechanism.
- **[Yu et al. 2024](https://arxiv.org/abs/2403.18167)**: Two mechanisms for hallucination — inadequate entity enrichment和failure to extract attributes. 这篇paper的entity recognition directions对应第一个mechanism的gating signal.
- **[Arditi et al. 2024](https://openreview.net/forum?id=pH3XAQME6c)**: Safety refusal mediated by single direction. 这篇paper extends到knowledge refusal, 发现同样pattern但不同direction.
- **[Kissane et al. 2024](https://www.alignmentforum.org/posts/YWo2cKJgL7Lg8xWjj/base-llms-refuse-too)**: Base LLMs也refuse. 这篇paper提供explanation — base model有entity recognition mechanism, finetuning repurpose它.
- **[Kossen et al. 2024](https://arxiv.org/abs/2406.15927)**: Semantic entropy probes for hallucination detection. Uncertainty directions是更interpretable的alternative.

### 10.3 Practical Implications

1. **Hallucination Detection**: 监控end-of-instruction token的uncertainty latents能在generation前detect即将hallucinate的cases.
2. **Refusal Steering**: 用entity recognition directions能improve model的calibration — 让它更reliably refuse unknown entities.
3. **SAE-based Interpretability**: 证明了SAE能uncover meaningful, causal directions — 不仅是descriptive工具.
4. **Finetuning Understanding**: 支持"finetuning repurposes rather than creates"的hypothesis, 对alignment research有implications.

### 10.4 Limitations与Open Questions

- **Labeling Noise**: Known/unknown classification基于fuzzy matching, 有noise.
- **Binary简化**: 真实知识是continuous的 (partial knowledge), 但paper用binary分类.
- **Effect Size Variation**: Gemma 2 9B的effects比2B弱, Llama 3.1 8B的known latent steering没显著降低refusal — cross-model一致性不完美.
- **Limited Entity Types**: 只测4种entity type, 未覆盖更abstract concepts.
- **Self-knowledge Scope**: Authors cautious — entity recognition不necessarily imply其他forms of self-knowledge.
- **Attention Mechanism细节**: Paper显示attention受影响但没fully decompose是keys/values/queries哪个被modify.

### 10.5 Methodological Generalizability

作者强调methodology可generalize到任何binary (Section 3) 或continuous (Section 7) features — 不限于entity recognition. 这为future mechanistic interpretability研究提供了template:
1. 定义binary/continuous feature of interest
2. 构建labeled dataset
3. 计算SAE latent separation scores / t-statistics
4. MaxMin across categories找general latents
5. Causal validation (steering + orthogonalization)
6. Mechanistic decomposition (activation patching + attention analysis)

## 11. 总结

这篇paper是mechanistic interpretability在hallucination问题上的重要进展. 它不只是描述"model会hallucinate", 而是**causally identify了hallucination prevention mechanism的neural substrate** — entity recognition directions in middle layers, 它们gate attribute extraction circuit via attention modulation, 并被chat finetuning repurposed为explicit refusal behavior.

对Karpathy这样的researcher来说, 这paper的beauty在于它connects多个levels of analysis:
- **Representation level**: SAE latents as interpretable features
- **Circuit level**: Attention heads as attribute extractors
- **Behavioral level**: Refusal vs hallucination
- **Developmental level**: Pretraining learns mechanism, finetuning repurposes it

这种multi-level mechanistic story正是mechanistic interpretability community追求的 — 不只是"what does the model do"而是"why, in terms of internal mechanisms, does it do it".

**Key references for deeper dive**:
- [Gemma Scope SAEs](https://arxiv.org/abs/2408.05147)
- [JumpReLU SAEs](https://arxiv.org/abs/2407.14435)
- [Arditi et al. on refusal direction](https://openreview.net/forum?id=pH3XAQME6c)
- [Nanda et al. on factual recall](https://www.alignmentforum.org/posts/iGuwZTHWb6DFY3sKB/fact-finding-attempting-to-reverse-engineer-factual-recall)
- [Geva et al. on factual associations](https://aclanthology.org/2023.emnlp-main.751/)
- [Neuronpedia for SAE exploration](https://www.alignmentforum.org/posts/BaEQoxHhWPrkinmxd/announcing-neuronpedia-platform-for-accelerating-research)
- [Scaling Monosemanticity (Anthropic)](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
