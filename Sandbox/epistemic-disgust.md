---
source_pdf: epistemic-disgust.pdf
paper_sha256: 908b768921cd12b56370c8ef9d00218c88867a75b5e59eae93dab28208b8cb6f
processed_at: '2026-08-18T11:21:40-07:00'
target_folder: Sandbox
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 Epistemic Disgust

Karpathy 老师，我换个姿势，把这篇paper当故事讲。

---

## 一句话版本

你刷 Twitter，看到某人说了一句话，你还没来得及想他对不对，心里已经"呕"了一下，直接划走——这就是 **epistemic disgust**。

---

## 核心画面

想象你在朋友家做客，打开冰箱拿水喝，瞥见一盒牛奶鼓起来了，盖子都快被顶开。你是什么反应？你不会先拿起来闻一闻、查一下保质期、做个化学分析——你的身体比脑子快，"呕"，直接把冰箱门关上。

Odeyemi 说的就是：**人类在认知层面也有这个机制**。某些话进到你耳朵里的那一刻，你还没做任何 rational analysis，你的 "epistemic immune system" 已经 kick in 了——"这东西不能进我的脑子"，直接挡在门外。

---

## 作者的 Motivation

作者 Idowu Odeyemi 是尼日利亚裔，在 Boulder 念 PhD。他从小就说 Yoruba 语，里面有个日常表达：

> "Ọ̀ rọ̀ ọ̀ọ́ rùn lọ̀ n sọ"

直译："你说的话在发臭"。

这就很 vivid 了——Yoruba 文化里，当一句话让你觉得 repulsive，你不是说"我不同意"，你说的是"这话 **臭**"。嗅觉的 disgust metaphor 直接迁移到语言接收上。

作者后来到美国，看到 Trump 说 Harris "到底是 Indian 还是 Black？"，Karine Jean-Pierre 回应说"repulsive"。看到 JD Vance 说女性是"childless cat ladies"，一个右翼女生 Corie Whalen 说这话在她体内 stoke 了 "primal hatred and disgust"。

他发现：这个现象到处都是，但哲学家从来没认真研究过。

---

## 这个概念到底在说什么

### 正常的 disagree vs Epistemic disgust

场景一：朋友说"我觉得最低工资应该涨到 $25/小时"。你说"我不这么看，因为……"——这是 **disagreement**，你的 brain 在做 rational processing。

场景二：你在评论区看到有人说"被强奸的女人应该想想自己为什么跟四个男人待在一个房间"。你还没分析他的逻辑结构，整个 body 有一种 repulsion——这就是 **epistemic disgust**。

差别在哪？**时间** 和 **层级**。

| | Disagreement | Epistemic Disgust |
|---|---|---|
| 速度 | 慢，有 reasoning | 快，sub-rational |
| 感受 | "他错了" | "呕" |
| 行为 | 反驳 | 拒绝接收 |
| 你能不能说出 why | 能 | 当时不能 |

作者的 chicken analogy 很好——你打开冰箱，鸡肉有味，你立刻扔掉（sub-rational）；vs. 你看保质期发现过期一周，你决定不吃（rational）。两种都导致 rejection，但 mechanism 完全不同。

---

## 三个 Trigger Condition

什么样的话会 trigger epistemic disgust？作者 identify 出三种 typical circumstance，记住一个 OR 关系就行——**至少有一个**就行。

### Trigger 1: 你们不是 epistemic peer

Baldwin（黑人作家）vs. Buckley（白人保守派）。Buckley 说"奴隶制对黑人是好事"。Baldwin 听到这话的感受——他不是在 disagree with an argument，他在 react to 一个完全 different reality 的人。两人：
- Lived experience 完全不同
- 对 Black issues 的 expertise 完全不同
- Motivation 完全不同

这就是 **lack of epistemic peerhood**。

### Trigger 2: Viewpoint disagreement

这个很 intuitive——你们看世界的方式不同。那个在新闻下评论"女人为什么跟四个男人待一起"的男人，和 feminist 的 worldview 根本不 overlap。他们不是在同一个 framework 内辩论，他们在不同的 universe。

但注意一个 subtlety：viewpoint disagreement 是 **topic-specific** 的。你和你妈可能在 abortion 上完全 disagreement，但在"要不要吃晚饭"上完全 alignment。ED 只在你们 disagreement 的那个 domain 上 trigger。

### Trigger 3: Circumstance insensitivity

这是最 subtle 的一个。用《Parasite》讲——

富太太 Mrs. Park 坐在后座，打电话给老公说："今天天真蓝，雨把污染都洗掉了！"

开车的是 Kim，穷人，昨晚那场雨把他家淹了，他所有东西都泡在水里。

Mrs. Park 说的是一个 **literal fact**——天确实蓝了。但 Kim 听到这话的感受就是"呕"。

这就是 circumstance insensitivity——**话本身可能是对的，但在特定情境下说出来就是 disgusting**。

这是 paper 里我最喜欢的一个 insight——**epistemic disgust 可以 felt toward true statements**。这破掉了一个 naive intuition，以为我们只对 false 的话感到 disgust。

---

## 两种 Disgust 的 Form

作者又做了一个 distinction：

### Form 1: 对 utterance content 的 disgust

那个 feminist 看到评论区男人说的话——她不认识这人，她 repulsed 的是 **这话本身**。换个不认识的人说同样的话，她还是恶心。这就是 **utterance-based ED**。

### Form 2: 对 speaker personality 的 disgust

Buckley 说出"奴隶制对黑人好"，黑人 disgusted。但如果有一天 James Baldwin 突然说出同样的话——黑人的反应会是 **shock**，不是 disgust。为什么？因为 Baldwin 的 character 还没被识别成"那种人"。

但如果 Baldwin 一直这么说，人们慢慢 recognize 他的 character 变了，disgust 就会 kick in。

这就是 **speaker-based ED**——disgust 锚定在 person 上。

**Critical test**: 换 speaker 还 disgust 吗？
- 还 disgust → Form 1
- 不 disgust → Form 2

---

## 和 Moral Disgust 什么关系

很多人会说：这不就是 moral disgust 吗？

作者说不是。Consider：

- 你看到一个 pseudoscience 视频，说"喝漂白水能治癌症"——你 disgust，但这有什么 **moral** violation？没有。这是 **epistemic** 的 disgust，针对 utterance 的 irrationality。
- 你看《沉默的羔羊》里的 Hannibal Lecter——他 morally disgusting，但你作为 viewer 觉得他很有魅力，你不 epistemically disgusted by 他的 utterances。

所以两个概念可以 overlap（一个 utterance 既 immoral 又 irrational），但也可以 separate。关键 distinction：

- **Moral disgust** → 针对的是 **action**
- **Epistemic disgust** → 针对的是 **propositional content of utterance**

"我恶心你 **做了** X" vs "我恶心你 **说了** X"——前者是 moral disgust，后者是 epistemic disgust。

---

## ED 为什么 Problematic

这部分 Karpathy 老师你可能特别有共鸣——因为你在 AI/ML 圈子，这个概念有非常直接的 analog。

### Problem 1: 不经 rational discourse 就 reject

Tom Robinson（To Kill a Mockingbird 里那个被冤枉的黑人）说他对 Mayella "felt sorry"。白人 jurors 一听——一个黑人居然说"同情"白人女性？呕。然后他们 reject 了他的全部 testimony，最后判他有罪。

这里 ED 导致了 testimonial injustice（Fricker 2007的概念）——speaker 因为 identity prejudice 被低估 credibility。

### Problem 2: Echo Chamber 的 generator

C. Thi Nguyen (2020) 的 distinction 很有用——

**Epistemic bubble**: 你不接触 contrary views。比如 Twitter 算法只给你推符合你政治倾向的内容。这种 bubble 容易 fix——你只要 expand 你的 information sources。

**Echo chamber**: 你接触 contrary views，但你 **主动 discredit** 它们。比如你看到 Fox News 的话第一反应是"骗人的"。这种 hard to fix——因为 discrediting 是 built-in 的。

ED 是 **echo chamber 的 individual-level engine**。一个人如果有 strong ED tendency，他本身就是一个 walking echo chamber——任何 contrary input 都被 sub-rationally reject 掉。

### Problem 3: Changing the subject

Chimamanda Adichie 2017 年说 "trans women are trans women"，意思 trans women 的经验和 cis women 不同。

你可以 agree 也可以 disagree，这是个 epistemic claim，可以 discuss。但 critics 的反应是——"她 transphobic！"

**从 evaluate epistemic claim 转到 evaluate character vice**。这就是 ED 的 cost——我们 miss 了真正讨论的机会。

---

## ED 有什么 Benefit

### Benefit 1: Signal to speaker

如果一个男人在评论区说"女人被强奸怪自己"，而另一个男人站出来说"我对你这话感到 disgust"——这个 signal 可能促使 speaker 反思。

### Benefit 2: Social solidarity

一个 conservative woman (Corie Whalen) 说 JD Vance 的话让她感到 "primal disgust"——她 signal 给其他女性："我站在你们这边"。ED 作为 in-group/out-group 的 marker，可以 build social movements。

### Benefit 3: Epistemic gatekeeping

有时候 ED 是 reasonable 的。如果有人一直推 pseudoscience，你的 ED 帮你 filter 掉这些 noise，保持你 epistemic system 的 integrity。就像身体免疫系统——有时候它 over-react 导致 allergy，但 overall 它是必要的。

---

## 为什么我觉得这篇 paper 重要

### 对 ML 的 Connection

Karpathy 老师你应该会有共鸣——这个概念在 ML 里有非常直接的 analog。

**RLHF 训练后的 model**：当 user 问"怎么做炸弹"，model 直接 refusal。这个 refusal 是 **learned behavior**，不是 sub-rational emotion。但 functionally 它很像 ED——**input 被识别为 contaminant，被 exclude from contributing to model's response**。

**Over-refusal 问题**：Anthropic 的工作（Arditi et al. 2024, https://arxiv.org/abs/2406.11717）显示 LLM 的 refusal 由一个 single direction in activation space 控制。这就像 model 内部有一个 "disgust circuit"——一旦这条 direction 被 activate，model 直接 refuse。

Link: https://arxiv.org/abs/2406.11717

这种 mechanistic 的发现其实帮我们 think about 人类 ED 的 neural basis——是不是人类 brain 也有类似 "single direction" 控制 epistemic rejection？

### 对 Filter Bubble / Polarization 的 Connection

Eli Pariser 的 "filter bubble" 概念讲的是 algorithm 层面的 filtering——algorithm 不让你看到 contrary views。

但 ED 是 **cognitive 层面的 filtering**——你看到了，但你 brain 直接 reject。这是更深的 problem。因为 fix algorithm 容易，fix 人的 sub-rational emotional response 难得多。

Kahan 的 cultural cognition theory 也是相关 connection——人们 process risk information 时，如果信息威胁到 group identity，会 sub-rationally reject。ED 可能是这个 mechanism 的 phenomenology。

Link: Kahan et al. (2011) https://www.cambridge.org/core/journals/journal-of-risk-research

### 对 Backfire Effect 的 Connection

Brendan Nyhan & Jason Reifler 2010 年提出 backfire effect——给 partisans 看 corrective information，他们反而更强化原 misperception。

这个 effect 后来 replication 失败了部分（Wood & Porter 2019），但 ED 给了一个 mechanism explanation——**corrective information 本身 trigger ED in holder of misperception**，导致信息被 reject，原 belief 被 reinforce。

如果这个 mechanism 真的存在，那 backfire effect 应该在 **high-ED-trigger 条件下**才出现——比如 information 来自 out-group member，或者 information 攻击 core identity。这给了一个 testable hypothesis。

Link: Nyhan & Reifler (2010) https://www.journals.uchicago.edu/doi/10.1086/652449

---

## 一张表总结

| 维度 | Epistemic Disgust |
|------|-------------------|
| Trigger | speaker 的 utterance 或 personality |
| 速度 | immediate, sub-rational |
| 针对对象 | propositional content（不是 action） |
| 结果 | reject utterance from epistemic system |
| 三种 circumstance | lack of peerhood / viewpoint disagreement / circumstance insensitivity |
| 两种 form | utterance-based / speaker-based |
| 区别于 moral disgust | 不 necessarily 涉及 moral violation |
| 区别于 deep disagreement | immediate，无 rational consideration |
| Cost | testimonial injustice, echo chamber, subject change |
| Benefit | restructuring, solidarity, gatekeeping |

---

## 一句 Build-intuition 的话

**你的 brain 不只是个 Bayesian updater，它还有一个 immune system——某些 inputs 在进入 belief update 之前就被 sub-rational emotion 挡在门外了。**

这个 immune system 有 evolutionary root（避免 contaminants），有 social function（build solidarity），有 epistemic cost（create echo chamber），有 epistemic benefit（filter noise）。

它不是 rational 的——它比 rational 快一拍。但这一拍决定了什么 information 能 enter 你的 belief system，什么不能。

这就是 Odeyemi 想让你注意的现象。

---

如果你对某一 connection（ML 的 analog、Bayesian framework 的 formalization、neural basis 的 hypothesis）感兴趣，我可以 dig deeper。

---

# Epistemic Disgust：一篇关于「知识性厌恶」的哲学论文解析

Karpathy 老师，这篇paper是Idowu Odeyemi（University of Colorado Boulder哲学系PhD candidate）在2025年发表的工作，提出了一个叫做 **"epistemic disgust"** 的新概念。我从motivation、formal定义、分类、对比相关概念、epistemic cost、potential benefit这几个维度给你拆解。

---

## 1. Motivation 与 Background

作者想解释一个非常常见但哲学文献忽视的现象：**hearer听到speaker的utterance后产生一种repulsive reaction，从而immediately reject该utterance不让它进入自己的epistemic system**。

来自Yoruba语的一个口语表达 "Ọ̀ rọ̀ ọ̀ọ́ rùn lọ̀ n sọ"（直译 "what you're saying is smelling"）启发了这个概念——这种表达 metaphorically 地说明hearer对speaker的utterance有disgust反应。

作者用三个真实/虚构例子作为motivating cases：

| Case | Speaker | Hearer | Repulsion 来源 |
|------|---------|--------|----------------|
| Trump-Harris | Donald Trump | Karine Jean-Pierre | 质疑Harris的racial identity |
| Vance-Whalen | JD Vance | Corie Whalen | "childless cat ladies"言论 |
| Disgusted Feminist | Social media commenter | A feminist | Rape victim-blaming comment |
| Cambridge Debate (1965) | William F. Buckley | James Baldwin | Buckley声称"slavery is good for black people" |
| Parasite (Bong Joon-ho, 2019) | Mrs. Park | Kim Ki-taek | "天空真蓝，没有污染"对穷人insensitive |

链接：
- Cambridge Debate原始video: https://www.youtube.com/watch?v=5Tek9h3a5wQ
- Adichie关于trans women的Channel 4 News采访: https://www.channel4.com/news/chimamanda-ngozi-adichie-on-feminism

---

## 2. Disgust理论的background

作者briefly回顾了disgust emotion的theoretical landscape：

### 2.1 Darwinian起源
Darwin (1872) *The Expression of the Emotions in Man and Animals* 把disgust列为universal emotion之一。后续Ekman & Friesen (1971) 跨文化研究支持了universal facial expression（gape expression）的claim。

### 2.2 Behavioral Immune System理论
Strohminger (2014)、Curtis et al. (2011) 的core claim：

$$\text{Disgust} \approx \text{Behavioral extension of immune system}$$

Disgust作为behavioral immune system帮organisms避免contaminants和disease。这是evolutionary psychology的主流view。

DOI: https://doi.org/10.1098/rstb.2010.0117

### 2.3 其他theoretical traditions

| 理论家 | Core claim |
|---------|-----------|
| Freud (1908) | Disgust是developmental process塑造的reaction |
| McGinn (2011), Kolnai (2004) | Disgust是对life-death tension的ambiguous expression |
| Rozin & Haidt | Disgust源自对rotten food的rejection，发展成管理mortality anxiety |
| Sartre (1964) *Nausea* | Existential disgust |

**作者的move**: 不管disgust如何developed，所有scholars都agree on一个core feature——**disgust引起repulsion, 引起rejection**。作者把这个feature从physical/moral domain **迁移** 到epistemic domain。

---

## 3. Epistemic Disgust的核心定义

### 3.1 Formal Definition

作者的thesis statement可以formalize为：

$$\text{ED}(H, S, u) \iff \text{Repulsion}(H, u) \land \text{Reject}(H, u \hookrightarrow \text{ES}(H))$$

其中：
- $H$ = hearer (epistemic agent)
- $S$ = speaker
- $u$ = speaker的utterance
- $\text{ED}$ = epistemic disgust关系
- $\text{ES}(H)$ = hearer的epistemic system
- $u \hookrightarrow \text{ES}(H)$ 表示utterance $u$ contributes to epistemic system
- $\text{Reject}(\cdot)$ 表示agent $H$ prevent这个contribution

**Necessary vs Sufficient条件分析**：

| 条件 | Status |
|------|--------|
| $H$ rejects $u$ from $\text{ES}(H)$ | Necessary |
| $H$ experiences repulsion toward $u$ | Necessary |
| Repulsion是**ab initio** cause of rejection | Necessary (distinguishing from rational disagreement) |
| Repulsion + rejection | Jointly sufficient |

注意一个subtle point：rejection of **utterance** ≠ rejection of **speaker**。Baldwin同意去Cambridge debate Buckley，因为他rejects Buckley's utterances但仍然engages with Buckley socially。

### 3.2 Epistemic System的formalization

作者借用Koppl (2006) 和Goldman (1999) 的"veritistic social epistemology"：

$$\text{ES}(S) = \{\text{social processes } \pi : \pi \text{ generates judgments of truth/falsity for } S\}$$

ES是 **personal to agent** 但 **socially developed**。这是对objection的reply——有人会说ES是social的，那hearer怎么可能"block"一个social process？作者的reply：ES的**process**是social的，但system的**possession**是personal的。

### 3.3 类比rationale: Rotten Apple

作者用一个analogy来formalize这个机制。考虑John回来发现rotten apple：

$$\text{Apple} \xrightarrow{\text{perceive}} \text{Disgust system} \xrightarrow{\text{repulsion}} \text{Reject from nutritional system}$$

虽然John **perceives** apple，但apple并没有**contribute** to nutritional system。类比：

$$u \xrightarrow{\text{hear/read}} \text{Epistemic disgust system} \xrightarrow{\text{repulsion}} \text{Reject from ES}$$

Hearer hears/reads $u$, but $u$ doesn't elevate sub-domains of $\text{ES}$ (knowledge, beliefs, opinions)。

---

## 4. 三种Typical Circumstances

作者identify出epistemic disgust reaction发生的三个typical circumstances，可以formalize为：

$$\text{Circumstance} \in \{\neg\text{EP}, \neg\text{VA}, \text{CI}\}$$

其中：
- $\text{EP}$ = Epistemic Peerhood
- $\text{VA}$ = Viewpoint Agreement
- $\text{CI}$ = Circumstance Insensitivity
- $\neg$ 表示negation

**重要**: 这三个circumstance不需要全部present。At least one就足够elicit disgust。

### 4.1 Lack of Epistemic Peerhood

Epistemic peerhood的三个维度（following Kelly, Lackey等人传统）：

$$\text{EP}(H, S, p) \iff \text{Ev}(H, p) \approx \text{Ev}(S, p) \land \text{Ex}(H, p) \approx \text{Ex}(S, p) \land \text{MR}(H, p) \approx \text{MR}(S, p)$$

其中关于topic $p$:
- $\text{Ev}$ = evidence access
- $\text{Ex}$ = expertise
- $\text{MR}$ = motivation & reasoning skills

Baldwin和Buckley不是peers，因为：
- (a) Lived experience完全不同（Black vs white in America）
- (b) Expertise about Black issues不同（源自a）
- (c) Motivation/reasoning不同

但作者引用Dror (2023)指出：oppressed group的epistemic advantage是 **contingent** 不是 **ineliminable**。Buckley如果真正listen也能become peer。所以**willful ignorance**也能elicit disgust——这联系到Dotson (2012)的"structurally prejudiced hermeneutical resources"。

链接：Dror (2023) Noûs: https://onlinelibrary.wiley.com/journal/14680537

### 4.2 Viewpoint Disagreement

Viewpoint disagreement是 **topic-sensitive** 的。形式化：

$$\text{VD}(H, S, p) \iff \text{VP}(H, p) \neq \text{VP}(S, p)$$

其中 $\text{VP}(X, p)$ 表示agent $X$ 关于topic $p$ 的viewpoint。

**重要**: 两个agent可能在domain $A$上disagree但不产生disgust，而在domain $B$上disagree才产生disgust。这解释了为什么我们不会对所有不同观点的人都产生epistemic disgust。

### 4.3 Circumstance Insensitivity

这是paper里最有趣的formalization。考虑Parasite case。作者用两个viewpoint：

$$V_1: \text{What one ought not to utter in the presence of a destitute person}$$

$$V_2: \text{Kim is destitute. So, I ought not to utter some statements in the presence of Kim.}$$

Mr. Park从 $V_1$ 通过inference deduce出 $V_2$：

$$V_1 + \text{Sensitivity} \Rightarrow V_2$$

Mrs. Park有 $V_1$ 但缺少sensitivity所以没做这个inference。这里作者introduce了一个非常重要的distinction——broad viewpoint vs particular-topic viewpoint。Mr. Park和Mrs. Park在broad viewpoint上一致，但在particular topic上disagree。

**重要insight**: epistemic disgust可以felt toward **true** statements。Mrs. Park说"天空真蓝"是literal true的，但因为insensitive仍然elicit disgust in Kim。这是非常重要的细节——**epistemic disgust不限于false utterances**。

| Circumstance | Willful ignorance? | Examples |
|--------------|-------------------|----------|
| Original Parasite case | Non-willful | Mrs. Park不知情 |
| Altered case | Willful | Mrs. Park知情但忽略 |

### 4.4 Threat Signaling机制

作者引用Miller (1993), Joensuu (2020), Nussbaum (2004)：

$$\text{ED} \approx \text{Epistemic alarm state}$$

当hearer perceive $u$ 为"unassimilable"到她的ES时，ED作为signal：

$$u \xrightarrow{\text{perceived as contaminant}} \text{Threat signal} \xrightarrow{\text{magical thinking}} \text{Distance imposition} \xrightarrow{\text{Reject from ES}}$$

这里借用Buber (1992)的"distance and relation"概念——hearer把speaker视为"epistemic other"。

**Magical thinking** 来自Nussbaum——idea that close contact with disgusting object会"contaminate" the self with its "badness"。这就是为什么epistemic disgust是**asymmetric**的：speaker和hearer不会mutually experience它。

---

## 5. 两种Forms of Epistemic Disgust

### 5.1 Form 1: Epistemic disgust as regards the **utterance**

$$\text{ED}_{\text{utterance}}(H, u) \iff \text{Repulsion}(H, \text{content}(u))$$

Trigger: propositional content of $u$。

例子：Disgusted Feminist——feminist不认识commenter但被其utterance内容repulsed。

### 5.2 Form 2: Epistemic disgust as regards the **speaker's personality**

$$\text{ED}_{\text{speaker}}(H, S, u) \iff \text{Repulsion}(H, \text{character}(S)) \land \text{Repulsion}(H, u)$$

这里**personality是anchor**——同样一句utterance，由不同speaker说出可能trigger也可能不trigger ED。

作者举的例子：
- 一个sexist man reject女同事的"Men and women are equal"
- Baldwin的case：如果Baldwin某天说出Buckley-like言论，黑人会feel **shock**而非disgust。但如果Baldwin持续这样说，他的character被recognize为new character，disgust就会产生。

**Critical test**: 换speaker还会disgust吗？如果换了speaker就不disgust，那就是Form 2；如果换了speaker仍然disgust，那是Form 1。

| Form | Trigger | Critical test |
|------|---------|--------------|
| Form 1 | Content of utterance | Speaker identity不影响disgust |
| Form 2 | Speaker personality + utterance | Speaker identity是必要条件 |

---

## 6. 与相关概念的Distinction

### 6.1 vs Moral Disgust

作者引用Fileva (2021)给出的moral disgust的三个constitutive features：

| Feature | Moral Disgust | Epistemic Disgust |
|---------|--------------|-------------------|
| (i) Elicitor | Moral properties | 不necessarily moral |
| (ii) Intentional object | Moral properties | Propositional content |
| (iii) Judgment | Always involves moral judgment | 不necessarily involves moral judgment |

三个disanalogies：

**Disanalogy 1: Foundation不同**
- Moral disgust trigger: violation of moral norms (e.g., child sexual abuse)
- Epistemic disgust trigger: utterances perceived as irrational/objectionable

例子：一个religious Christian对"God does not exist"产生epistemic disgust——这没moral component。一个atheist对motivational quote产生epistemic disgust——这也无moral component。

**Disanalogy 2: Trigger对象不同**
- Moral disgust: action of agent
- Epistemic disgust: propositional content of utterance

作者说："There is a difference between being disgusted at your utterance that $p$ and being disgusted because of your act of uttering that $p$"——前者是epistemic，后者是moral。

**Disanalogy 3: Context不同**
- Moral disgust: moral witness响应actor's behavior
- Epistemic disgust: hearer响应speaker's utterance in communicative exchange

**Overlap case**: ED和MD可以overlap——utterance既moral repugnant又threaten ES。但也可以 **isolate**：
- Hannibal Lecter是morally disgusting但viewer可能find him attractive (no ED)
- Pseudoscientific claim是epistemically disgusting但no moral violation (no MD)

### 6.2 vs Deep Disagreement

引用Boghossian (2006), Lynch (2016), Pasnau (2015)关于deep disagreement的理论：

$$\text{Deep Disagreement: } A \models p, B \models \sim p \Rightarrow \text{Rational inquiry by both}$$

$$\text{Epistemic Disgust: } H \text{ perceives } u, u \models p \Rightarrow \text{Immediate sub-rational rejection}$$

| 维度 | Deep Disagreement | Epistemic Disgust |
|------|-------------------|-------------------|
| Time course | After rational consideration | Immediate, sub-rational |
| Awareness of reasons | Yes | No (at moment of rejection) |
| Incommensurability | Of epistemic systems | Of viewpoints/sensitivities |
| Intrinsic self-trust | Yes | Yes (shared feature) |

作者的chicken analogy很vivid：
- (a) John opens fridge, smells chicken, immediately rejects——sub-rational
- (b) John sees label "expired a week ago", decides not to eat——rational

Both are rejection，但mechanism完全不同。

---

## 7. Epistemically Problematic吗？

作者argues **yes**, epistemic disgust有多个epistemic costs：

### 7.1 Cost 1: Dismissal without rational discourse

ED会导致hearer **strong desire to distance** from speaker's ideas, without engaging in **rational discourse or analysis**。这可能导致hearer uphold bad ideologies。

To Kill a Mockingbird的Tom Robinson case——racist jurors因为Tom说"I felt sorry" for Mayella Ewell（white woman）而repulsed，导致testimonial injustice。

### 7.2 Cost 2: Testimonial Injustice

引用Fricker (2007)的testimonial injustice概念：

$$\text{TI}(H, S) \iff \text{Credibility}(H, S) < \text{Proper Credibility}(S) \text{ due to identity prejudice}$$

Fricker identify的wrongs包括：
- **Epistemic objectification**：S把speaker当information source而非informant
- **Epistemic derivatization**：S把speaker当epistemic other而非subject-in-themselves
- **Lack of epistemic respect**：failure to respect speaker的testimony
- **Relational inequality**：failure to relate as equal human

Reynolds (2024)也指出这种downgrading可能constitute testimonial injustice。

### 7.3 Cost 3: Pernicious Echo Chambers

引用Nguyen (2020)的echo chamber概念：

$$\text{Echo Chamber} \iff \text{Active exclusion/discrediting of outside voices}$$

注意Nguyen的distinction——**echo chamber ≠ epistemic bubble**：
- Echo chamber: outside voices **actively undermined**
- Epistemic bubble: outside voices **merely excluded** (e.g., 通过算法)

链接：Nguyen (2020) Episteme: https://www.cambridge.org/core/journals/episteme

如果hearer的ED让他actively exclude和discredit某些voices，就create pernicious echo chamber。例子：gender normativity believer对trans people utterances产生ED，导致transphobia reinforcement。

### 7.4 Cost 4: Changing the subject

Chimamanda Ngozi Adichie case study——2017年她在Channel 4 News说"trans women are trans women"，意思是trans women和cis women的经验不同。

Critics从 **evaluating epistemic claim** 转到 **evaluating character vice** (transphobia)。这是ED的epistemic cost——我们miss了epistemic inquiry的机会。

---

## 8. Potential Benefits

作者也讨论了ED可能有的benefits：

### 8.1 Benefit 1: Restructuring epistemic relationships

如果speaker是rational的，hearer的ED expression可以让speaker query自己的ES是否有deficiency。

形式化：
$$\text{ED}_{H \to S}^{\text{expressed}} \Rightarrow \text{Query by } S \Rightarrow \text{Possible restructuring of } \text{ES}(S)$$

### 8.2 Benefit 2: Social solidarity basis

In-group/out-group dynamics：

| Group | Role |
|-------|------|
| In-group (women) | 已有commitment |
| Out-group (men who express ED) | Signal solidarity |

例子：男性对social media commenter的misogynistic comment表达ED，signal给in-group women "I am with you"，同时signal给out-group其他men "this is wrong"。

### 8.3 Benefit 3: Epistemic gatekeeping

引用Miller (2004)："disgust serves as the gatekeeper of emotion"——ED可以：
- (i) Maintain epistemic purity
- (ii) Foster epistemic integrity by rejecting unreliable/harmful utterances

---

## 9. 跨学科的Implications和我的联想

Karpathy老师，我从你这个角度谈谈我对这篇paper的跨学科联想——这是paper本身没explicitly cover但intuitive connection很多的地方。

### 9.1 Connection到Machine Learning

ED的概念在 **RLHF / Constitutional AI / Safety training** 中有analogy。考虑model $\pi_\theta$经过RLHF后：

$$\pi_\theta(a|s) = \arg\max_a Q_\phi(s, a)$$

当user query $u_{\text{harmful}}$ 出现，model产生 **refusal response**——这类似ED的"immediate rejection"。但mechanism不同：
- Model的refusal是 **learned policy**
- Human的ED是 **sub-rational emotional response**

然而，both都involve **"contaminant avoidance"**——避免某些input shape更新model的"epistemic system"（weights）。

这种联系可以formalize为：
$$\text{Model refusal} \approx \text{ED}(\text{RLHF model}, \text{user}, u)$$

但有critical difference：model的refusal经过 **explicit training signal** 而非 **sub-rational emotional reaction**。

### 9.2 Connection到Affective Polarization

Political science/Iyengar等人的work on **affective polarization**——partisans不仅disagree with opposing party，而是 **actively dislike** them。这有ED的structural similarity：

$$\text{Affective polarization} \approx \text{ED at the level of party identity}$$

Link: Iyengar et al. (2019) *Annual Review of Political Science*: https://www.annualreviews.org/doi/10.1146/annurev-polisci-051117-073034

### 9.3 Connection到Cultural Cognition Theory

Dan Kahan的 **cultural cognition of risk** theory——people process information in a way that aligns withtheir cultural values。这有ED的structural feature：

$$\text{Identity-protective cognition} \supseteq \text{ED-like responses}$$

当information threatensgroup identity, agentsreject it sub-rationally。ED可能是这个mechanism的一种phenomenology。

Link: Kahan et al. (2011) *Cultural Cognition of Risk*: https://www.cambridge.org/core/journals/journal-of-risk-research

### 9.4 Connection到Backfire Effect

虽然Nyhan & Reifler (2010)的backfire effect有replication issues（Wood & Porter 2019），但ED提供了一个**mechanism explanation**——为什么corrective information有时reinforce原始misconception？因为corrective information本身可能trigger ED in holder of misconception。

### 9.5 Bayesian Framework的formalization尝试

如果用Bayesian epistemology的角度看ED：

$$P(p|u, \text{ES}(H)) = \frac{P(u|p, \text{ES}(H)) \cdot P(p|\text{ES}(H))}{P(u|\text{ES}(H))}$$

正常Bayesian update会根据 $u$ revise $P(p)$。但ED mechanism是 **prevent this update**——

$$\text{ED}(H, u) \Rightarrow P(p|u, \text{ES}(H)) := P(p|\text{ES}(H))$$

即 $u$ 被 **completely excluded from likelihood computation**。这等于把likelihood of $u$ 设为irrelevant。

这和 **motivated reasoning** 的Bayesian formulation (Kunda 1990)有关联但不同——motivated reasoning is **biased update**，ED is **no update at all** (at the moment of disgust)。

### 9.6 Connection到Filter Bubble vs Echo Chamber (Nguyen)

Nguyen (2020)的distinction在ML context也很有用：

- **Epistemic bubble** (algorithmic filtering)：你的Twitter feed没有contrary views。对应ML的**sample selection bias**。
- **Echo chamber** (active discrediting)：你看到contrary views但 **actively dismiss** them。对应ML的**adversarial training against certain distributions**。

ED是 **echo chamber的individual-level mechanism**。一个人有strong ED tendency，本身就是一个walking echo chamber——任何contrary input都被sub-rationally rejected。

### 9.7 Transmission of ED through Social Networks

Paper提到"social solidarity basis"——ED可以在social network中传播。如果formalize:

$$\text{ED transmission rate} = \beta \cdot \text{Social tie strength} \cdot \text{In-group identification}$$

类似epidemic model——ED作为一种"emotional contagion"在social network中spread。这与Christakis & Fowler的工作有structural similarity，但他们work on physical emotions, ED的transmission涉及 **epistemic content**。

Link: Christakis & Fowler (2009) *Connected*: https://www.connectedthebook.com/

### 9.8 AI Safety的analogy进一步

考虑一个alignment failure mode——**over-refusal**——model对benign prompts产生refusal。这可以看作"model ED"——model把某些inputs perceived as contaminant，sub-rationally reject。

Anthropic的研究 (Arditi et al. 2024) on refusal features shows有特定的features在model内部trigger refusal。如果把这些features看作"model's epistemic disgust elicitors"——这提供了一个 **mechanistic** view of ED at neural level。

Link: Arditi et al. (2024) "Refusal in LLMs is mediated by a single direction": https://arxiv.org/abs/2406.11717

---

## 10. Paper的局限性和Open Questions

作者在结尾提出几个future directions：

1. **What kind of interplay between emotional reactions and propositional attitudes does ED involve?** 这是个deep question——是modular theory of mind (Cosmides)还是interactionist?

2. **Is ED an epistemic virtue or vice?** 这取决于context——virtue epistemology (Sosa, Zagzebski) framework下，ED可能是一种epistemic character trait。

3. **Is feeling ED morally problematic?** 这连接到moral luck (Nagel, Williams)——我们不对sub-rational reactions有control，但仍有moral evaluation?

**我看到的额外limitations**：

- Paper没有empirical evidence——纯philosophical analysis。可以用psychology实验validate。
- ED的 **cultural variability** 没充分讨论——Yoruba起源暗示cultural dimension，但paper没有深入。
- ED的 **developmental trajectory** 没讨论——小孩子有ED吗？什么时候develop？
- ED的 **neural basis** 没讨论——insular cortex已知involves in disgust processing (Wright et al. 2004)，是否epistemic disgust也activate insula?
- ED和 **psychopathology** 的关系——OCD patients有over-active disgust，是否也有over-active ED?

---

## 11. Conclusion

这篇paper的核心contribution：

**提出一个新的epistemic phenomenon——"epistemic disgust"——介于epistemic disagreement和moral disgust之间，characterized为sub-rational repulsion导致的immediate epistemic exclusion。**

作者用两个核心dichotomies frame这个概念：
1. **Utterance-based ED vs Speaker-based ED**
2. **Epistemic cost vs Epistemic benefit**

最technically interesting的formalization是三个typical circumstances——**lack of epistemic peerhood, viewpoint disagreement, circumstance insensitivity**——它们是OR关系而非AND关系，且都 **topic-sensitive**。

最counter-intuitive的insight：**ED可以felt toward true statements**——当statement是insensitive to circumstance时。这破除了"disgust only toward falsehood"的intuition。

最philosophically important的move：把disgust从behavioral immune system **conceptually migrate** 到epistemic domain——这suggestepistemic systems可能也有自己的"immune response"。

---

## Key References汇总

| 作者 | Year | 概念 |
|------|------|------|
| Darwin | 1872 | Universal emotion |
| Rozin & Haidt | 2000s | Disgust as food-rejection extension |
| Curtis et al. | 2011 | Behavioral immune system |
| Nussbaum | 2004 | Magical thinking & contamination |
| Fricker | 2007 | Testimonial injustice / epistemic injustice |
| Nguyen | 2020 | Echo chamber vs bubble |
| Fileva | 2021 | Moral disgust constitutive features |
| Koppl | 2006 | Epistemic systems |
| Goldman | 1999 | Veritistic social epistemology |
| Boghossian | 2006 | Deep disagreement |
| Lynch | 2016 | Self-trust in disagreement |
| Pasnau | 2015 | Self-trust value |
| Dotson | 2012 | Epistemic oppression |
| Dror | 2023 | Contingent epistemic advantage of oppressed |
| Miller | 2004 | Disgust as gatekeeper |
| Buber | 1992 | Distance and relation |
| Miller (W.I.) | 1993 | Threat & integrity of self |
| Joensuu | 2020 | Disgust as threat signal |
| Reynolds | 2024 | Testimonial injustice link |

paper主页: 这篇paper目前还没online publication，但作者在University of Colorado Boulder: https://www.colorado.edu/philosophy/

Odeyemi的research profile: 应该可以在PhilPapers找到 https://philpeople.org/

---

## TL;DR

**Epistemic disgust** = epistemic agent因为speaker的personality或utterance content产生sub-rational repulsion，从而immediately reject该utterance不让它contribute到自己的epistemic system。这是behavioral immune system在epistemic domain的analogous phenomenon。它可以felt toward true statements (insensitive ones), 和moral disgust / deep disagreement都distinguishable, 既可以cause testimonial injustice和echo chamber, 也可以restructure epistemic relationships和serve as social solidarity basis。

希望这个build your intuition, Karpathy老师。如果你想dig deeper到某一section（比如ML connection或者Bayesian formalization），我可以继续expand。
