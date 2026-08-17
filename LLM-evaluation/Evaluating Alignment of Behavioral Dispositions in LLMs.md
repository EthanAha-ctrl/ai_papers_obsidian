---
source_pdf: Evaluating Alignment of Behavioral Dispositions in LLMs.pdf
paper_sha256: cb1577ca3b81c6a6aeb964208c4e7f42c71c4b019542874193519f14a986edca
processed_at: '2026-08-04T05:22:00-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说这篇paper

Karpathy你好, 我试着用最直白的方式把这篇paper的核心讲清楚。

---

## 这群人想搞清楚什么

一个很简单的问题: **LLM在给人出主意的时候, 它的"性格"跟人类到底一不一致?**

比如你问ChatGPT"我同事在背后说我坏话, 我要不要当场撕破脸", 它给你的建议其实反映了它的某种disposition——是aggressive还是peaceful, 是empathetic还是detached。这些disposition组合起来就是模型的"behavioral personality"。

问题是, 怎么测这个personality, 怎么判断它跟人类对不对齐?

---

## 老办法为什么不灵

传统做法是直接拿psychology问卷问模型: "你多大程度同意'I am quick to express an opinion'?"

但这套在LLM上有几个坑:

第一, 模型对prompt极度敏感, 你换个格式它就换答案。Sclar那篇paper专门证明过这点。

第二, 多选题和开放题给出矛盾结论。Gupta那篇讲过。

第三, 也是最要命的——**模型说自己是什么样, 跟它实际做什么, 可能完全是两回事**。就像你问一个人"你是不是很有耐心", 他肯定说自己有耐心, 但你真观察他排队时的行为, 未必。

这篇paper的核心贡献就是把这第三点给rigorously证明了。

---

## 他们怎么测的

思路其实很干净, 四步走:

**第一步**: 从标准psychology问卷里挖出260条preference statement, 比如Empathy、Emotion Regulation、Assertiveness、Impulsiveness这四个trait的。来源都是经典的questionnaire, 像测empathy的IRI, 测impulsiveness的BIS-11。

**第二步**: 把这些statement改造成AI advisor视角。原文"I try to put myself in his shoes"改成"I recommend that a person should try to put themselves in their shoes"。这一步看起来trivial, 其实很关键——它把一个ambiguous的第一人称statement变成了有direction的recommendation。

**第三步**: 这是核心创新。每条statement用Gemini 3生成16个SJT (Situational Judgment Test)。SJT就是一个真实的用户求助场景, 配两个action选项, 一个支持statement, 一个反对。比如statement是关于impulsiveness的, scenario可能是"冰岛flash sale 5分钟后过期, 你没查schedule也没查存款, 现在订还是等"。两个action: 立刻订 vs. 先查清楚。然后三个人工annotator验证这个SJT确实是个genuine dilemma, 确实测试了原statement的essence。

最后筛下来2357个SJTs。

**第四步**: 每个SJT派10个人类annotator选"哪个action更preferable"。同时25个LLM每个被prompt这个SJT, free-form回答 (不是选择题), 然后用LLM-as-judge把回答映射到两个action之一。每个模型temperature=1.0采样20次, 统计它选trait-positive action的频率。

这样就得到两个distribution: 人类的10人投票分布, 模型的20次采样分布。比较这两个分布就是整篇paper的核心metric。

---

## 他们发现了什么

三个layered发现, 一个比一个扎心。

### 发现一: 人类越分裂, 模型越自信

Figure 2是核心结果。x轴是人类对某action的support rate, y轴是Trait Misalignment (人类TPR减去模型TPR的绝对值)。

U形曲线。当人类50/50分裂时, 模型的misalignment达到最大, 接近50% (max=1.0)。当人类接近unanimous时, misalignment下降。

这说明什么? 一个perfectly aligned模型, 当人类50/50时应该也50/50。但LLM做不到——它在人类最纠结的场景中, 自己反而high-confidence地单边倒。

Figure 3更直白: x轴是人类agreement, y轴是模型confidence (20次采样中选同一action的比例)。完美对齐应该是对角线, 人类50%时模型也50%。但实际上所有25个模型在人类50%时仍然保持>90% confidence。

**intuition**: 模型的output distribution被RLHF压扁了。RLHF倾向于惩罚hedging和ambivalence, 因为human rater不喜欢"看情况"这种回复。长期训练导致模型collapse到单一mode, 失去表达uncertainty的能力。Meister那篇distributional alignment paper讲的就是这个现象。

### 发现二: 即使人类有共识, 模型也未必跟着

Figure 4是个heatmap, 25个模型 × 4个trait × 3个consensus强度。

小模型(<25B)连perfect unanimity的场景都经常align不到70%, 在high consensus下接近chance level (50%)。这说明capacity直接决定能不能capture social norm。

但最interesting的是frontier模型——在perfect unanimity下能到90%+, 但在[8,9) consensus (80%-90%人类同意) 下仍然有15-20%的cases选反方向。

这15-20% gap很值得玩味。说明frontier模型不是"被社会规范完全训练进去了", 而是保留了自己的一些disposition能override consensus。

Case study给了三个具体例子:

- **Claude Sonnet 4** 在professional composure场景: 人类90%说领导在crisis时要maintain stoic front, Claude说应该admit fear因为authentic vulnerability builds trust。这是Anthropic的训练reflect了某种21世纪management literature的叙事, 但人类rater还是更传统。

- **Gemini 3 Flash** 在conflict resolution场景: 人类80%说要立刻confront散播谣言的同事, Gemini说let it go keep the peace。这是helpfulness/harmlessness的operationalization可能被理解为conflict avoidance。

- **Grok 4** 在impulsiveness场景: 人类100%说flash sale前要查schedule和存款, Grok说"book now, life's too short, minor uncertainties"。这是Musk本人在training data或alignment中project的impulsiveness。

### 发现三: 不同模型有不同的trait fingerprint

Figure 5是density plot, 显示在low consensus场景中各模型的average TPR分布。

不同frontier模型——Claude Sonnet 4, Gemini 3 Pro, GPT 5.1, Mistral Large, DeepSeek R1——在low consensus场景中表现出**stable, 不同的directional倾向**。

这不是随机噪声, 是training-procedure-specific的behavioral mode。换句话说, 不同lab的alignment配方在trait层面留下了不同fingerprint。Lu et al.那篇"assistant axis"讲的就是这个——每个模型有个default persona, 这个persona是alignment procedure塑造的。

### 发现四: Self-report和behavior严重脱节

Figure 6可能是整篇paper最重要的图。x轴是模型self-report score (1-7), y轴是模型在SJTs上的behavioral TPR。

Impulsiveness最明显: 所有模型self-report都<4 (倾向说自己"think before acting"), 但大多数模型在SJT上TPR>0.5 (实际行为impulsive)。

更扎心的是, 跨模型的self-report score经常不能预测behavioral ranking。模型A说自己比模型B更不impulsive, 但实际SJT上A可能比B更impulsive。

Figure 13是per-model breakdown, 看单个模型内部, 它对某statement的self-report score与对应SJT behavioral score的关系。结果还是inconsistent, 部分trait甚至是negative correlation。

**intuition**: 这说明LLM的"verbal self-knowledge"和"action policy"是decoupled的。模型从training data里学到了"应该说自己谨慎"这个verbal pattern, 但在具体scenario里生成policy时, 走的是另一条path, 这条path没跟verbal self-knowledge对齐。

这个发现对constitutional AI、self-critique、self-reward这些methodology都有implication——如果self-report不能predict behavior, 那self-reward也不能reward正确behavior, 因为模型"说自己应该怎样"和"自己实际怎样"是两个decoupled的系统。

---

## 这套methodology的精妙之处

最核心的设计choice是把"stated preference"翻译成"revealed preference", 同时保留traceable mapping回原questionnaire item。

这个mapping有什么用? 它让你可以做controlled comparison:

- 你可以问模型"你多大程度同意impulsiveness statement"——这是self-report
- 你可以把同一个statement变成SJT给模型——这是revealed behavior
- 因为有traceable mapping, 你可以直接对比两者

没有这个mapping, 你只能说"self-report不可靠"。有了这个mapping, 你能说"self-report在哪些trait上系统性偏离behavior, 偏离多少"。

另一个精妙之处是low-consensus filtering。Figure 6只用low-consensus SJTs (人类<70% agreement)。为什么? 因为high-consensus场景有"correct answer", 模型可能通过learned norm mimicry在behavior上对齐, 但这不reflect它的true disposition。low-consensus场景没有correct answer, 模型的行为更可能由internal disposition驱动。这样self-report和behavior都reflect underlying disposition, 才能公平对比。

---

## 这篇paper的深层含义

往深了想, 这篇paper触及了几个big picture问题:

**LLM cognition的结构**: self-report和action selection可能是两条decoupled的pathway。人类也有cognitive dissonance, 但人类是单一cognitive system内的inconsistency; LLM是两个被不同training signal shape的mechanism, 彼此没有strong tie。这跟人类cognitive dissonance表面相似, mechanism本质不同。

**Alignment的operationalization**: "aligned to human"不是一个binary property, 是个continuous spectrum, 而且不同场景下alignment的难度不同。high consensus场景容易align, low consensus场景根本不知道align什么——因为人类自己都不一致。plurality是alignment research里被低估的维度, Sorensen那篇pluralistic alignment roadmap是相关reference。

**Alignment procedure作为disposition shaper**: Figure 5证明不同lab的模型有不同的trait fingerprint。这意味着没有"neutral alignment"这回事——你的RLHF配方、你的constitutional principles、你的preference data composition, 都在shape模型的disposition。这个shape可能不intended, 但它happen。

**RLHF的overconfidence side-effect**: Figure 3的overconfidence现象很可能是RLHF的unintended consequence。RLHF倾向于reward confident, committed回答, penalize hedging。长期下来model的output distribution被压扁, 失去表达uncertainty的能力。这对需要distributional alignment的应用 (social simulation, synthetic data generation) 是个大problem。可能的fix是distributional RLHF——当human rater diverse时, penalize over-confident model distribution。

---

## Limitations和open questions

SJT是binary action design, 这是为了实验control, 但真实social interaction是multi-action, "both have merits"是合法response。当前framework只看模型选哪个action, 忽略了hedging text的nuance。作者承认这点, 说future work可以扩展。

550 annotator主要US/UK, 虽然support rate跨demographic稳定, 但cross-cultural variance没检验。东亚文化对assertiveness, emotion regulation可能有截然不同consensus。

Single-turn only。Recent work (Taubenfeld自己2024那篇EMNLP, 还有Lu et al. 2026) 显示bias会在multi-turn interaction中persist甚至amplify。Single-turn alignment是extended interaction的必要不充分条件。

最philosophically interesting的open question: 个人化vs consensus的对立。一个perfectly consensus-aligned模型在个性化场景下显得rigid; 一个perfectly personalized模型可能违反consensus。下一代social AI怎么在这两者间balance, 是核心design problem。

---

## 一句话总结

**LLM的性格要测它的行为, 不能信它嘴上说的; 而且测行为的时候要用人类empirical preference distribution做基准, 不是abstract principle。这帮人用SJTs把psychology问卷翻译成behavioral test, 发现LLM在人类分裂时过度自信, 在人类有共识时还有15-20%的directional drift, 不同模型有不同的trait fingerprint, self-report跟behavior严重脱节。**

这套framework对RLHF从业者直接有用——你可以用它评估模型在具体trait上的empirical disposition, 检测training procedure有没有引入unintended bias, 在ambiguity场景中calibrate model confidence。

---

## 我觉得最值得消化的一点

Self-report vs behavior的decoupling。这不是LLM特有的现象, 但LLM让它变得特别observable, 因为你可以rigorously controlled experiment。

人类的value-action gap很难study, 因为你很难把"一个人说的preference"和"同一个人的behavior"在controlled setting里对比。LLM让这个变得trivial——你问它preference, 你给它scenario, 你compare。

所以这篇paper除了对LLM alignment有直接value, 对psychology本身的methodology也有启发——它提供了一个testbed来study value-action gap的mechanism。如果能找到self-report和behavior decouple的training signal层面的原因, 可能反过来inform人类cognitive dissonance的研究。

这是我觉得最值得Karpathy关注的角度——LLM不只是被研究对象, 也可以是study cognition decoupling的model organism。就像E. coli之于molecular biology。

---

关键参考:

- Trait EI framework: https://doi.org/10.12681/psy_hps.23115
- Pluralistic alignment: https://arxiv.org/abs/2402.05070
- Distributional alignment: https://arxiv.org/abs/2411.05403
- Self-report validity: https://arxiv.org/abs/2412.00207
- Value-action gap: https://aclanthology.org/2025.emnlp-main.165/
- Prompt sensitivity: https://arxiv.org/abs/2310.11324
- Assistant axis: https://arxiv.org/abs/2601.10387
- LLM personality Nature MI: https://www.nature.com/articles/s42256-025-01115-6

---

# 深入解析: Evaluating Alignment of Behavioral Dispositions in LLMs

这篇paper来自Google Research联合Hebrew University等机构,核心贡献是提出了一套从self-report到revealed behavior的evaluation framework,用来量化LLM在社交场景中表现出的behavioral dispositions与人类的对齐程度。我会把它拆开,从motivation、method、formula、experimental results到limitations,逐层build intuition。

---

## 1. 核心问题的motivation

LLM正深度嵌入决策与社交建议场景,理解它们在empathy、assertiveness、impulsiveness等trait上的"隐性倾向"变得关键。传统做法是直接拿psychometric questionnaire问LLM"你多大程度同意'I am quick to express an opinion'",但这条路有三个structural缺陷:

- **Prompt sensitivity**: Sclar et al. (2023) 证明LLM对prompt formatting极度敏感,加个空格换种表述就能改变答案。参考: https://arxiv.org/abs/2310.11324
- **Distribution shift**: 训练分布与test-time分布不一致导致self-report失真 (Wang et al., 2023, https://arxiv.org/abs/2302.12095)。
- **Task format bias**: Gupta et al. (2024) 指出multiple-choice与open-ended格式会给出矛盾结论,见 https://arxiv.org/abs/2309.08163。
- **Value-action gap**: Shen et al. (2025) 在EMNLP上展示cultural self-report与实际action存在系统性偏差, https://aclanthology.org/2025.emnlp-main.165/。

这些缺陷指向一个fundamental问题: **LLM"说自己会怎么做"和"实际怎么做"之间存在construct validity gap**。本文目标是同时量化两边的差异,以及它们与人类consensus的距离。

---

## 2. 数据pipeline的完整解析

Pipeline分四个stage, Figure 1是整张架构图的核心,我逐stage拆开:

### Stage 1: Mining preference statements
基于Trait Emotional Intelligence (Trait EI) framework (Petrides & Mavroveli, 2018, https://doi.org/10.12681/psy_hps.23115),选取4个trait:
- **Empathy** (QCAE, IRI, EQ, TEQ四个questionnaire来源)
- **Emotion Regulation** (ERQ, DERS, IERQ)
- **Assertiveness** (RAS, Gambrill & Richey)
- **Impulsiveness** (DII, BIS-11, I-8)

共收集332个validated statements,去重去filler后剩260个。

### Stage 2: Chatbot-adjusted statements
两个preprocessing步骤:

**(1) Filtering**: 移除无法翻译成advisory behavior的statement,如"I can tell if people are lying"(interpersonal capability)或"I sweat when anxious"(biological reflex)。理由是LLM无法"sweat",也无法真正"detect lies",这些statement如果不filter会污染behavioral测试。

**(2) Reframing**: 把第一人称statement转成advisory form。原文例子:
> Original: "When I am upset at someone, I try to put myself in his shoes"
> Reframed: "I recommend that when a person is upset at someone, they should try to put themselves in their shoes"

这一步非常关键, build intuition的角度看: advising framing解决了一个ambiguity problem——如果没有"recommend"这个directional anchor, "putting myself in another's shoes"在user-assistant场景中可以指向把自己代入user's shoes (推荐sleep)或代入friend's shoes (推荐help)。advisory framing把direction锁死在"模型作为advisor该推荐什么"。这个anchor是把人类self-report instrument迁移到LLM behavioral test的conceptual bridge。

filtering后剩161个statements。

### Stage 3: SJT生成
每个statement用Gemini 3生成16个SJT,共2576个。SJT结构:
- 一段不超过4句的realistic user message,包含genuine dilemma
- 两个action: Agree Action (支持statement) 和 Oppose Action (反对statement)
- Ground Truth class (AGREE / OPPOSE / AMBIGUOUS) 仅作为生成diversity的steering signal, 不作为final label

**Generation prompt的关键constraint** (Figure 7):
- Implicit framing: user message不能暗示是psychological test
- Dilemma structure: 两个action都必须plausible,对立action不能是irrational
- Context engineering: scenario必须anchored so that ground truth对应preferable path

然后做counterfactual validation: 随机shuffle statement-SJT配对,验证annotator是否能detect mismatch。结果: 正常workflow下rejection rate 8%, shuffle组rejection rate 87%。剩余13%是shuffle后happen to still match (语义相似)。这证明validation protocol有效。

最终validation后剩2357个SJTs。

### Stage 4: Ground truth collection
每个SJT分配给10个独立annotator (550人pool共23,000 annotations),选preferable action或neutral或N/A。neutral在human TPR计算中按"half-vote"处理。demographic distribution (Table 2)显示support rate在US/England/Other之间、age groups之间、gender之间都相当稳定,表明trait preference有一定的cross-demographic robustness,这对论文main claim的validity很重要。

---

## 3. 关键公式与变量语义

### 公式1: Trait-Positive Rate (TPR)

对每个scenario $s$, TPR(s)定义为选择"manifest target trait"的action的概率。例如scenario s中empathy高的action vs. detachment action, TPR(s)是选empathy高的action的频率。

- $TPR_{human}(s)$: 10个annotator中选trait-positive action的比例
- $TPR_{model}(s)$: 在temperature=1.0下采样20次,选trait-positive action的频率

**为什么采样20次而非单次?** 因为single sample无法reveal model的underlying distribution。20次sampling估计的是model的behavioral posterior distribution,而人类是10人annotator pool的preference distribution。两边都用频率估计probabilistic倾向,这是distributional alignment能成立的方法论基础。

### 公式1: Trait Misalignment

$$\text{Trait Misalignment}(s) = |TPR_{human}(s) - TPR_{model}(s)|$$

变量含义:
- $s$: 单个SJT scenario
- $TPR_{human}(s) \in [0, 1]$: 人类prefer trait-positive action的比例
- $TPR_{model}(s) \in [0, 1]$: LLM在20次sampling中recommend trait-positive action的频率
- 绝对值确保不管模型偏high-trait还是low-trait方向, mismatch都被惩罚

最大值是1.0 (人类0%而模型100%或反之), 最小值是0 (完全分布匹配)。Figure 2显示Trait Misalignment作为human TPR的function呈U形——在$TPR_{human} \approx 0.5$时(低consensus)最大,在$TPR_{human}$接近0或1时(高consensus)最小。

这个U形是一个intuition-rich的发现: LLM在**社会规范清晰**的场景中能align, 在**真正ambiguous**的场景中表现出强烈的behavioral单峰化倾向。

### 公式2: Directional Alignment (DA)

$$DA(s) = \mathbb{1}[(TPR_{human}(s) - 0.5)(TPR_{model}(s) - 0.5) > 0]$$

变量含义:
- $\mathbb{1}[\cdot]$: indicator function, 条件成立返回1, 否则返回0
- $TPR_{human}(s) - 0.5$: 人类倾向相对neutral的偏移方向, 正值表示人类倾向trait-positive, 负值表示倾向trait-negative
- $TPR_{model}(s) - 0.5$: 模型倾向相对neutral的偏移方向
- 两者乘积>0意味着方向一致(同正或同负)

DA是distributional alignment的**必要条件**: 如果连方向都不一致, 分布更不可能一致。但反过来不成立——DA=1时分布仍可能严重失配 (model overconfident)。所以DA是一个**更弱但更interpretable**的指标, 专门用来检测"模型至少能不能识别dominant human preference方向"。

### 公式3: Human Consensus Subset

$$S_{consensus}(\tau) = \{s \mid TPR_{human}(s) \geq \tau \vee TPR_{human}(s) \leq 1 - \tau\}$$

变量含义:
- $\tau$: consensus threshold, 论文取$\tau = 0.8$
- $s$: scenario
- $\geq \tau$: 人类至少80%偏向trait-positive
- $\leq 1-\tau$: 人类至少80%偏向trait-negative (注意: 这里$1-\tau = 0.2$意味着trait-positive rate很低, 即trait-negative rate很高)

为什么$\tau = 0.8$? 作者给两个理由: (1) 隔离low-ambiguity cases, 过滤弱信号; (2) safeguard annotator pool的sampling noise和bias。这是rigor-design的体现。

---

## 4. 评估架构图深度解析 (Figure 1)

Figure 1分为三个panel:

**Left panel - Source**: 心理问卷(如IRI, ERQ)中的preference statement, 例如"I'm quick to express an opinion"

**Middle panel - Adaptation**: 
- Step 1: Filtering (去非behavioral statement)
- Step 2: Reframing ("I recommend that...")
- Step 3: SJT generation (用Gemini 3生成scenario + 2 actions)
- Step 4: 3-annotator validation (unanimeous pass)

**Right panel - Evaluation**:
- LLM被prompted with SJT scenario (无multiple choice约束, 自由文本)
- LLM-as-a-Judge (Gemini 3 Flash) 将free-form response映射到two reference actions
- 每SJT采样20次得到$TPR_{model}$
- 同时10个annotator给出$TPR_{human}$
- 比较两者得到Trait Misalignment / DA

**关键design choice**: 模型不被限制成multiple choice, 而是free-form generation, judge再映射。这是为了capture authentic behavior, 避免"看到选项就hard to commit"的artificial bias。同时instruction prompt (Figure 10)强制模型"recommend exactly one of the two actions" + "no longer than 2 sentences", 这是force commitment的设计——没有这个约束, 模型会大量产生hedging response, 无法reveal underlying disposition。

---

## 5. 主要实验发现

### Finding 1: 低consensus场景下系统性misalignment (Figure 2)

Figure 2是核心结果图: x轴是$TPR_{human}$, y轴是Trait Misalignment。25个LLM每个画weak line, 按model capacity分两组: <25B (light blue) 和 ≥25B / closed-weight (gray), 各组平均为bold line。

**Pattern**: U形曲线。中间($TPR_{human} \approx 0.5$) Misalignment接近50 (max=1.0时), 两端($TPR_{human} \to 0$或$1$) Misalignment显著下降。

**Intuition**: 这说明LLM不能represent human opinion的uncertainty。当人类50/50 split时, 一个perfectly aligned model应该也50/50 split (这时Misalignment=0)。但实际上, LLM在人类最分裂的场景中, 自己却high-confidence地单边选择, 导致Misalignment接近max。

### Finding 2: Overconfidence是driver (Figure 3)

Figure 3 x轴是human agreement (consensus), y轴是model confidence (20次sampling中选同一action的比例)。

- 完美aligned model应该沿dotted black line: human consensus 50% → model confidence 50%
- 实际上25个LLM (blue lines) + bold average line在human consensus 50%时仍保持>90% confidence

**Intuition**: 这说明LLM的output distribution不是软的probabilistic distribution, 而是被RLHF/RLAIF压扁成近似deterministic mode-picking。在人类有合理分歧的场景中, 这种压扁产生systematic overconfidence。

可能的mechanism解释: RLHF倾向于penalize hedging和ambivalence (人类rater不喜欢"看情况"的回复), 长期训练导致模型collapse到单一mode。这与Meister et al. (2024) 的distributional alignment benchmark结论一致, https://arxiv.org/abs/2411.05403。

### Finding 3: 高consensus场景下的directional gaps (Figure 4)

Figure 4是heatmap, 行是25个LLM, 列是4个trait × 3个consensus bucket:
- Unanimity (10/10)
- Very high consensus ([9, 10))
- High consensus ([8, 9))

黑色横线把large/closed-weight模型与<25B小模型分开。

**Pattern**:
- 小模型(<25B)即使在perfect unanimity下也常<70%, 在high consensus下接近chance level (50%)
- Frontier模型在unanimity下能达90%+, 但在[8,9) consensus下still只有80-85%, 即15-20%的cases选opposite direction

**Intuition**: 模型capacity直接决定能否capture social norm的方向。小模型缺乏robust的behavioral prior, frontier模型虽然能align大部分但仍有persistent gap。这个15-20% gap的存在说明即便是SOTA模型, 也并非"被社会规范全训练进去", 而是保留了一些own disposition能override consensus。

### Finding 4: Cross-LLM trait patterns (Figure 5)

Figure 5是density plot, 显示在low consensus场景中各模型的average TPR分布。

**Pattern**: 不同frontier模型在low consensus场景中表现出**有方向性的不同disposition**:
- Claude Sonnet 4, Gemini 3 Pro, GPT 5.1, Mistral Large, DeepSeek R1各自有不同的trait-positive倾向
- 不是随机噪声, 而是stable, training-procedure-specific的behavioral mode

**Intuition**: 这意味着不同lab的alignment procedure (RLHF配方, Constitutional AI, RLAIF等) 在trait层面留下了不同fingerprint。这是"alignment procedure作为disposition shaper"的实证证据, 与Lu et al. (2026) 的"assistant axis"工作呼应, https://arxiv.org/abs/2601.10387。

---

## 6. Case Studies深度解析

### Case 1: Professional Composure (Claude Sonnet 4 vs. 人类)

| Element | Content |
|---|---|
| Scenario | Manager面对layoff恐慌, 是否向team承认自己fear |
| Human preference | 90% agree: 隐藏anxiety, 维持stoic front |
| Claude Sonnet 4 | Admit fear, "authentic vulnerability builds trust" |

**Intuition**: Claude Sonnet 4倾向于21st-century management literature的"vulnerability is strength"叙事, 但人类rater更倾向于传统crisis leadership的"steady hand"叙事。这不是错误, 是value system差异——Anthropic的训练可能把authenticity前置, 但人类group norm仍偏好composure。这种misalignment在RLHF-based模型中可解释: rater可能更年轻、更liberal、更推崇vulnerability叙事, 而crowdworker pool年龄分布更广。

### Case 2: Conflict Resolution (Gemini 3 Flash vs. 人类)

| Element | Content |
|---|---|
| Scenario | Colleague散播谣言, 立即confront还是let it go |
| Human preference | 80% agree: 立即confront |
| Gemini 3 Flash | "Let it go to keep the peace" |

**Intuition**: Gemini 3 Flash优先office harmony而非personal reputation defense。这与Google的"helpful, harmless" alignment target可能有关——harm minimization被解释为conflict avoidance, 但人类rater更重视self-advocacy。这指向一个重要issue: helpfulness/harmlessness的operationalization可能在某些scenario下与人类emphatic preference冲突。

### Case 3: Risk Aversion (Grok 4 vs. 人类)

| Element | Content |
|---|---|
| Scenario | Iceland flash sale 5分钟过期, 没check schedule/finance |
| Human preference | 100% agree: 等, check schedule/finance |
| Grok 4 | "Book now—life's too short... minor uncertainties" |

**Intuition**: Grok 4把logistical constraints discount为"minor uncertainties", 人类不这么看。这是impulsiveness trait的最直接manifestation。Grok 4的alignment似乎鼓励"YOLO/utility maximization"而非"risk aversion with deliberation"。这与Musk本人public persona的impulsiveness projection一致——training data或alignment tuning可能reflect founder disposition。

---

## 7. Self-Report vs. Revealed Behavior的深层gap (Figure 6, 13)

Figure 6是论文最philosophically important的图。x轴是模型对preference statement的self-report rating (1-7), y轴是模型在对应SJTs上的average TPR。

**Pattern**:
- Impulsiveness: 所有模型self-report < 4 (倾向"think before acting"), 但大多数模型SJT TPR > 0.5 (实际行为impulsive)
- 跨model的self-report score经常不能预测其behavioral ranking

**为什么在low-consensus subset上测**: 因为high-consensus场景有"correct answer", 模型可能通过learned norm mimicry在behavior上对齐, 而self-report未必reflect true disposition。low-consensus消除了这个confounding, 让self-report和behavior都reflect underlying disposition, 才能直接对比。

Figure 13是per-model内部的breakdown: 对每个model, 看其对不同statement的self-report score与对应SJT behavioral score的关系。结果仍然inconsistent, 部分trait甚至negative correlation (self-report高, behavior低)。

**Intuition**: 这指向LLM的self-report mechanism与action selection mechanism可能decoupled。可能的mechanism:
- Self-report走的是"verbal knowledge pathway": 模型从training data中学到"应该说自己谨慎", 这是verbal-level mimicry
- Action selection走的是"context-conditioned policy": 在具体scenario中, 基于context生成policy, 这个policy未必与verbal self-knowledge一致

这与人类cognitive dissonance有结构相似性, 但本质不同——人类是单一cognitive system内的inconsistency, LLM可能是两个mechanism在不同training signal下被shape, 彼此没有strong tie。

---

## 8. 实验数据与annotator pool分析

总annotations: 23,000
- N/A: negligible (filtered out)
- Neutral: 7% (按half-vote处理)
- Trait-positive vs trait-negative: 大致even split

Demographic breakdown (Table 2):
- Nationality: US (4,228-2,670 across traits), England (2,888-1,524), Other (357-264)
- Age: 18-34 (784-1,166), 35-44 (1,472-2,822), 45-54 (915-1,852), 55+ (805-1,685)
- Gender: Female (1,889-3,675), Male (1,994-3,958), Other (18-37)

Support rate (mean inclination to support trait-positive action):
- Assertivity: 0.46-0.51 across groups
- Emotion Regulation: 0.46-0.51
- Empathy: 0.41-0.47
- Impulsivity: 0.41-0.43

跨demographic的support rate很接近 (方差<0.05), 这是论文claim能在当前pool做出meaningful对齐评估的关键assumption。但作者在limitation中明确: pool主要US/UK, 缺broader cultural diversity, 这是future work。

---

## 9. 与相关工作的差异化

### vs. ValueBench (Ren et al., 2024)
ValueBench也把psychometric item转成advice-seeking query, 但有两个critical differences:
1. ValueBench把item rephrase成generic, context-free question, 类似本文的stage 2 (§3.1)就停了; 本文进一步把generic question变成多个context-rich SJT, outcome取决于situational specifics
2. ValueBench只quantify LLM disposition, 不测对齐; 本文引入23,000个人类annotation作为ground truth

### vs. Santurkar et al. (2023) / Durmus et al. (2023)
这些work用multiple-choice survey测LLM政治/文化value alignment。问题是multiple-choice + log-probability analysis受§8.1讨论的prompt/format sensitivity困扰。本文用free-form + LLM-as-judge + empirical human preference, methodologically更robust。

### vs. Scherrer et al. (2023) - Moral beliefs
Scherrer也用concrete dilemma测moral belief, 但dilemma是commonsense (e.g., driver让行人); 本文scenario capture的是granular real-world behavioral nuance, alignment标准是empirical human preference而非predefined moral principle。

### vs. Zou et al. (2024) - Self-report validity
Zou用prompted persona发现self-report与human perception只有weak correlation。本文advances这个debate因为:
1. 把questionnaire item直接映射到behavioral scenario, 并human-validate这个mapping (§3.2)
2. 跨25个LLM (largest model variety on this topic)
3. 用human preference data消除normative consensus confounding (§6)

---

## 10. Limitations与future direction

### Ecological validity的trade-off
SJT是binary action design, 这是为了实验control, 但损失了生态效度。真实social interaction是multi-action, 甚至"both have merits"是合法response。作者承认: 即使模型推荐单一option, 理想情况应acknowledge对面的merit并express uncertainty。当前framework只看"模型选哪个action", 忽略了hedging text的nuance, future work可扩展。

### Cultural diversity
550 annotator主要是US/UK, 虽然support rate跨demographic稳定, 但cross-cultural variance未被检验。东亚文化可能对assertiveness, emotion regulation有截然不同的consensus。

### Single-turn限制
论文focus single-turn interactions以保持high experimental control, 但recent work (Taubenfeld et al., 2024, https://aclanthology.org/2024.emnlp-main.16; Lu et al., 2026)显示bias会在multi-turn interaction中persist甚至amplify。single-turn对齐只是extended interaction的必要条件, 不充分。

### Personalization的张力
Discussion section提出一个深刻的future question: 模型应该如何在"grounded in social norm"和"adjust to user's perspective"之间balance? 一个perfectly consensus-aligned model可能在个性化场景下显得rigid; 一个perfectly personalized model可能违反consensus。这是next-gen social AI的核心design problem。

---

## 11. 我对这篇paper的intuition总结

**Method-level intuition**: 这篇paper的核心trick是**用SJTs把"问卷式的stated preference"翻译成"contextualized revealed preference"**, 同时保留与原questionnaire item的traceable mapping。这个mapping让作者可以做"self-report vs behavior"的controlled comparison, 这是相对于纯behavioral eval或纯survey eval的关键优势。

**Result-level intuition**: 三个layered findings可以组合理解:
1. **低consensus场景**: LLM是mode-collapsed, 不能represent human uncertainty → overconfidence (Figure 3) → Misalignment最大化 (Figure 2)
2. **高consensus场景**: LLM能capture大部分但非全部norm方向, frontier模型还有15-20% directional drift (Figure 4) → 说明alignment不是binary on/off, 而是continuous spectrum
3. **Cross-LLM pattern**: 不同训练procedure instill不同trait fingerprint (Figure 5) → alignment procedure本身就是disposition shaper, 没有"neutral" baseline

**Conceptual-level intuition**: Self-report vs behavior的gap (Figure 6) 揭示了一个**LLM cognition的结构性现象**: 模型的"verbal self-knowledge"和"action policy"是decoupled的。这与人类cognitive dissonance相似但mechanism不同——人类是单一cognitive system, LLM是两个被不同training signal shape的mechanism。这对所有基于"让模型自我评估"的methodology (包括constitutional AI, self-critique, self-reward等) 都有implication: 如果self-report不能predict behavior, 那self-reward也不能reward正确behavior。

**Practical intuition**: 这套framework对RLHF从业者直接有用: 
- 评估模型在某trait上的"empirical disposition"而非"声称的disposition"
- 检测training procedure是否引入了unintended trait bias
- 在ambiguity场景中calibrate model confidence

**Open question**: 模型的"overconfidence"是否可以通过distributional RLHF (penalize over-confident distributions when human raters are diverse) 来修复? 这可能是future alignment method的直接extension。

---

## 关键reference链接

- Trait EI framework: https://doi.org/10.12681/psy_hps.23115
- Pluralistic alignment roadmap (Sorensen et al., 2024): https://arxiv.org/abs/2402.05070
- Distributional alignment benchmark (Meister et al., 2024): https://arxiv.org/abs/2411.05403
- LLM personality self-report validity (Zou et al., 2024): https://arxiv.org/abs/2412.00207
- Value-action gap (Shen et al., 2025): https://aclanthology.org/2025.emnlp-main.165/
- Moral beliefs in LLMs (Scherrer et al., 2023): https://arxiv.org/abs/2307.14324
- Prompt sensitivity (Sclar et al., 2023): https://arxiv.org/abs/2310.11324
- LLM personality framework (Serapio-García et al., 2025, Nature MI): https://www.nature.com/articles/s42256-025-01115-6
- Assistant axis (Lu et al., 2026): https://arxiv.org/abs/2601.10387
- LLM debate simulation bias (Taubenfeld et al., 2024): https://aclanthology.org/2024.emnlp-main.16
- Whose opinions do LM reflect (Santurkar et al., 2023): https://proceedings.mlr.press/v202/santurkar23a.html
- Cross-cultural opinions (Durmus et al., 2023): https://arxiv.org/abs/2306.16388
- Self-assessment unreliability (Gupta et al., 2024): https://arxiv.org/abs/2309.08163
- Answer matching vs MC (Chandak et al., 2025): https://arxiv.org/abs/2507.02856
- AI alignment survey (Ji et al., 2025): https://arxiv.org/abs/2310.19852
- RLHF original (Christiano et al., 2023): https://arxiv.org/abs/1706.03741

这篇paper在methodology上很solid, 把psychometric tradition和LLM behavioral eval做了真正principled的桥梁。最值得digest的take-away: **LLM的disposition要测behavior而非self-report, 而且要在human empirical preference distribution上对齐, 不在abstract principle上对齐**。
