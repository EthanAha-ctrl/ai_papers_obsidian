---
source_pdf: age-of-ai-icm-2026.pdf
paper_sha256: 632f4e5a9a02cbb00f2a042502c2e6d6b339434682139f5ca12966cfd06a74cb
processed_at: '2026-08-18T00:03:18-07:00'
target_folder: Math
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版: Tao 在 ICM 2026 到底讲了什么

## 一句话总结

Tao 假设 AI 真的能做数学研究了, 然后问: **那我们数学家到底图啥?** 结论是: AI 会把我们以前混在一起的几个目标 (解决问题 / 写清楚 / 被同行接受 / 写进教科书) 撕开, 导致前面狂飙、后面堵车, 他叫这个 "proof indigestion"。

## Tao 在玩的 rhetorical trick

他先画一个大问题: "数学界怎么应对 AI?" 然后把这个 question 拆成两个 orthogonal direction:

**Direction 1:** AI 到底行不行? (Capability)
**Direction 2:** 我们做数学到底为了什么? (Values)

然后他说: **Direction 1 我不聊**。假设 AI 行 (Working Hypothesis), 我只聊 Direction 2。

为什么这个 move 聪明? 因为 Direction 1 是 empirical question, 大家吵来吵去没结论; Direction 2 是 values question, 必须现在就想清楚, 而且**无论 AI 行不行, 想 Direction 2 都有价值**。这相当于在不确定的世界里找一个 robust 的 action。

类比: 你不用先确认 climate change 是不是真的, 才开始想 "我们到底想要什么样的地球"。后一个问题本身就有价值。

## 历史类比: 1900-1930 的 foundation crisis

Tao 说: 20世纪初数学界经历了一次 foundation 危机。

Russell 1901 发现 naive set theory 自相矛盾: 考虑集合 $R = \{x : x \notin x\}$, 问 $R \in R$ 成立吗?

- 如果 $R \in R$, 按 $R$ 的定义 $R \notin R$, 矛盾
- 如果 $R \notin R$, 按 $R$ 的定义 $R \in R$, 矛盾

这把 Frege 的系统搞崩了。后来花了几十年搞出 ZFC (Zermelo-Fraenkel + Choice) 才稳住。

Gödel 1931 又补刀: 任何足够强的 consistent formal system 都有它自己证明不了的 true statement。

**Tao 的类比点:** 那次危机的产物是 ZFC + formal logic 的标准化, 大家信任这个 foundation 了。现在 AI 来了, 我们经历的是 **values/practices 层面的 foundation 危机**, 产物应该是类似 Leiden Declaration 这样的 codified guidelines。

注意: Tao 不是说 "AI 像 Gödel", 他是说 **"我们正在重演一场 foundation 危机, 但这次危机在 values 层而非 logic 层"**。

## AI Capability Conjecture 的模板

Tao 把 "AI 能做数学" 写成一个带 placeholder 的 template:

> 在 **某个** 时间, **某个** AI tool, 花 **某种** 成本, 在 **某种** 人类监督下, 能在 **某个** 领域, 以 **某种** 成功率, 达到 **某种** 质量, 完成 research-level 数学任务。

每个 "某种" 都是一个 variable。填得保守 = weak form, 填得激进 = strong form。

**唯一硬数据:** First Proof benchmark (https://1stproof.org) 2026年5月28日测了一批 AI harness:

| 指标 | 数值 |
|---|---|
| 测试问题数 | 10个 research-level |
| 参赛 AI 团队 | 4个 |
| 达到 publication quality 的 | 7/10 |
| 每题 compute 成本 | $10 – $1000 |

70% 成功率, 成本低于一个月 PhD 学生工资。这是 **medium-strong form 成立** 的 evidence。但 Tao 强调 reporting bias — 问题集是 self-selected, harness 是 self-selected, 没有 human baseline 对照。

## Goal 迭代: 这才是演讲的真正 meat

Tao 把 "数学家做 problem solving 到底图什么" 当成一个 iterative refinement 过程。每一步发现 Goodhart's law 漏洞, 就加一个 constraint。

### Goodhart's law 先讲清楚

> When a measure becomes a target, it ceases to be a good measure.

形式化: 设 $M$ 是 goal $G$ 的 proxy。你不优化时 $\text{cor}(M, G)$ 高; 你直接优化 $M$, correlation 崩塌。

AI 语境下特别危险因为:
1. Generative AI 训练目标就是 next-token prediction, 没有 ground truth 数学作为 signal
2. AI 公司有 financial incentive 让 output **看起来像** 数学, 而非**是**数学
3. Automation 让 optimization 成本极低, 容易 over-optimize

### Goal v1: 解决尽可能多的 open problems

$$\max \, |\{P : P \text{ solved}\}|$$

**Failure mode:** Riemann hypothesis 被 "证明" 1000 次, 全是错的。AI 生成 fake proof 海啸。

### Goal v2: 解决 + 验证

$$\max \, |\{P : \exists \text{ verified proof of } P\}|$$

"Verified" = Lean / Rocq / HOL kernel check 通过。Autoformalization 让这一步成本暴跌。

**Failure mode:** 一个 10000 行 Lean proof 通过 kernel check, 但**没有任何人类能 explain 它**。这已经发生在 Erdős problems 网站 (https://www.erdosproblems.com) — 有人提交 AI 生成 proof, 自己都说不清对不对。

### Goal v3: 解决 + 验证 + 写清楚

加入 exposition constraint。AI 这一步 record 混杂:
- 拼写/语法/format 无懈可击
- 但 triviality 啰嗦, key idea 一笔带过
- 不引用 prior literature
- 没有 high-level overview

Tao 的关键 insight: **过度打磨的 proof 反而难懂**。

人类写的 proof 有 **natural friction** — 作者卡住的地方会啰嗦、会反复 restart、会写多个 special case。这些 friction 是 signal: 读者看到就知道 "这里 hard, slow down"。

AI 把所有 friction 抹平, 等价于把 trivial computation 和 key insight 一样对待, 读者失去 priority cue。

信息论版本: 一个 proof section $s$ 的 entropy
$$H(s) = -\sum_w p(w|s) \log p(w|s)$$

人类 proof 在 key lemma 处 entropy 高 (作者 struggle, 措辞不流畅); AI proof entropy 均匀。读者用 **entropy gradient** 作为 attention allocator。AI 抹平 gradient = 删掉 attention signal。

### Goal v4: 解决 + 验证 + 写清楚 + **被 community 接受**

这步引入 **external, slow, human process**。不能被 author 单方面 optimize。

一个 proof 被 accept 需要:
- 专家读它
- 其他人用它做新工作
- 被引用、被 generalize、被 apply

这是 Goodhart's law 的天然解药 — community acceptance 本质 resistant to fast optimization。你没法用更多 compute 加速它。

### Goal v5 (final): 解决 + 验证 + 写清楚 + 接受 + **写进 definitive theory**

终极阶段: 结果进入 textbook, 成为 standard curriculum。这是最慢 stage, 需要 broad deliberative consensus。也是最有价值 stage — 所有下游 application (包括 AI 自身做数学) 都依赖这个 canonicalized backbone。

## 网络流隐喻: 为什么会堵车

Tao 把整个过程画成 flow:

$$\text{Open problems} \to \text{Solutions} \to \text{Verified} \to \text{Readable} \to \text{Published} \to \text{Definitive}$$

每个箭头是一个 stage, 有 capacity。

AI 加速前面 stage (generation, verification), 后面 stage (acceptance, canonicalization) 是 human-bound, capacity 不变。**结果: bottleneck 转移到后面**, 产生 "proof indigestion"。

用 queueing theory: 设 stage $i$ 的 service rate $\mu_i$, arrival rate $\lambda_i$。M/M/1 model 的 queue length:
$$L_i = \frac{\lambda_i}{\mu_i - \lambda_i}$$

当 AI 把 $\mu_1, \mu_2 \uparrow$ 而 $\mu_4, \mu_5$ 不变, $\lambda_3, \lambda_4 \uparrow$ (因为前面 stage 吐出更多), 但 $\mu_3, \mu_4$ 没变, 所以 $L_3, L_4 \to \infty$。

直觉版: 以前是 proof 稀缺, 一个新 proof 大家抢着看; AI 之后是 proof 泛滥, 但 referee、textbook writer、community digest 的 capacity 不变, queue 堆积如山。

参考: https://en.wikipedia.org/wiki/Little%27s_law

## Bourgain 1991 paper 片段: 为什么 Tao 放这个

演讲中插了一段 OCR 质量很烂的 paper 片段, Tao 注 "A 1991 paper of Bourgain, annotated by my much younger self"。

这是 Jean Bourgain 的 restriction theory 工作, 大概率是 GAFA 1991 那篇。

### 关键公式 reconstruct

Setup:
- $S_{d-1} \subset \mathbb{R}^d$: unit sphere, $(d-1)$-dim manifold
- $\sigma$: surface measure on $S_{d-1}$
- $\varphi \in L^\infty(S_{d-1})$, $|\varphi| \leq 1$
- $\widehat{\varphi\sigma}(\xi) = \int_{S_{d-1}} \varphi(z) e^{-2\pi i (z,\xi)} d\sigma(z)$: measure 的 Fourier transform
- $B(0, \rho)$: 半径 $\rho$ 的球

Critical exponent (Tomas-Stein):
$$q_0(d) = 2 \cdot \frac{d+1}{d-1}$$

变量: $d$ = dimension。$d=2 \Rightarrow q_0=6$; $d=3 \Rightarrow q_0=4$; $d=4 \Rightarrow q_0=10/3$。

物理直觉: sphere 是 $(d-1)$-dim, 它的 Fourier transform decay like $|\xi|^{-(d-1)/2}$。把这个 decay 和 $L^q$ scaling 结合, 临界 $q$ 就是 $q_0$。$q \geq q_0$ 时 restriction estimate
$$\|\widehat{f\sigma}\|_{L^q(\mathbb{R}^d)} \leq C \|f\|_{L^2(S_{d-1})}$$
成立。

Local estimate (6.27):
$$\|\widehat{\varphi\sigma}\|_{L^q(B(0,\rho))} \leq C_\epsilon \rho^{\frac{1}{q}\left(\frac{d+1}{q} - \frac{d-1}{2}\right) + \epsilon}$$

变量:
- $\rho$: ball 半径
- $\epsilon > 0$: 任意小, $C_\epsilon \to \infty$ as $\epsilon \to 0$
- $\frac{d+1}{q} - \frac{d-1}{2}$: scaling exponent

**Scaling intuition:** $\widehat{\varphi\sigma}$ 在 frequency space spread over ball $\rho$。若 unit-scale decay 是 $\rho^{-\alpha}$, localized $L^q$ norm 应该 scale as $\rho^{d/q} \cdot \rho^{-\alpha} = \rho^{d/q - \alpha}$。

Discrete restriction (6.30):
$$\left\|\sum_\alpha a_\alpha e^{i(x_\alpha, \xi)}\right\|_{L^q(B(0,R))} \lesssim R^{(d-1)(\frac{1}{2} - \frac{1}{q})} \left(\sum_\alpha |a_\alpha|^2\right)^{1/2}$$

变量:
- $\{x_\alpha\} \subset S_{d-1}$: $R$-separated points (cap 是 $R$-sparse)
- $\{a_\alpha\}$: 任意 scalars
- $R$: frequency localization scale

**Intuition:** Exponential sum $\sum a_\alpha e^{i(x_\alpha, \xi)}$ 在 ball $B(0,R)$ 上的 $L^q$ norm 由 coefficients 的 $\ell^2$ 控制, 带一个 $R$-dependent prefactor。$R$-separation 保证 exponential "几乎正交" — 这是 almost orthogonality argument。

Partition argument (6.31)–(6.32): 把球面分成 size $\sim 1/\sqrt{\rho}$ 的 cells $S_\alpha$, 每个 cell 选 representative $x_\alpha$。Taylor 展开:
$$\int_{S_\alpha} \varphi(x) e^{-2\pi i(x,\xi)} d\sigma = e^{-2\pi i(x_\alpha,\xi)} \int_{S_\alpha} \varphi(x) e^{-2\pi i(x-x_\alpha,\xi)} d\sigma$$

变量替换 $\xi' = \xi + \eta$, $\eta \in B(0, \rho^{1/2})$。因为 $|x-x_\alpha| < c\rho^{-1/2}$ 在 cell 内, phase $(x-x_\alpha, \xi)$ 的 oscillation $\sim \rho \cdot \rho^{-1/2} = \rho^{1/2}$, 这控制 error。

最终 (6.32):
$$\|\widehat{\varphi\sigma}\|_{L^q(B(0,\rho))} \lesssim \rho^{\text{power}} \left\|\left(\sum_\alpha \left|\int_{S_\alpha} \varphi e^{-2\pi i(x-x_\alpha,\xi)} d\sigma\right|^2\right)^{1/2}\right\|_{L^q(B(0,\rho))}$$

这是 square function estimate, 最后用 vector-valued restriction 收尾。

### 这个片段在演讲中的作用

Tao 用它 illustrate 两个点:
1. 人类 mathematician 写的 proof 有 **natural friction** — 公式编号、scope、notation 都 embedded 在 narrative 中
2. AI 重新 typeset 这种 paper 时会 smooth 掉所有 friction, 读者失去 "哪里是 hard part" 的 cue

参考:
- https://en.wikipedia.org/wiki/Restriction_theorem
- https://link.springer.com/article/10.1007/BF01896369

## Thurston 引用

> "We are not trying to meet some abstract production quota of definitions, theorems and proofs. The measure of our success is whether what we do enables people to understand and think more clearly and effectively about math."

— Thurston, "On Proof and Progress in Mathematics" (1994)

参考: https://arxiv.org/abs/math/9404236

Thurston 的 thesis: 数学是 about **human understanding**, 不 about formal proof artifacts。Proof 是 understanding 的载体, 不 understanding 本身。

Tao 引用 Thurston 是为了 ground 自己的 Goal v3–v5 在已有 mathematical values tradition 中 — 不是 AI 来了才临时编出来的 values, 是数学界早就认的, 只是以前没 explicit 化。

## Leiden Declaration

2026年6月2日发布, DOI: 10.5281/zenodo.20302944。

参考: https://leidendeclaration.ai

### 核心 recommendations

**1. Disclose tool use.** Paper 要有 "Tool and computational resource disclosure" section。避免最坏情况: 作者偷偷用 AI, 又藏起来怕被批评。Normalize responsible disclosure。

依据: UNESCO Open Science Recommendation + FAIR principles (Findable, Accessible, Interoperable, Reusable)。

**2. Support reviewing needs.** 作者要主动降低 reviewer 负担: 披露 tool use, 给完整 reference, 尽量提供 formal proof。降低 proof generation emphasis, 增加 proof digestion emphasis。

**4. Retain responsibility for correctness.** 作者对 correctness、adequacy、citation 完整性负全责。不能甩锅给 AI。

**6. Proper attribution.** AI 在 attribution 上有 known limitation (不能可靠 trace idea 来源)。Tao 的 rule of thumb:

> If authors cannot convincingly demonstrate that they can give a clear, expert-level talk on their results, that is correct and properly attributed, then the result should not be published.

"Talk test" — 把 AI assistance 的 invisible 转化为 visible。Talk 是 human-in-the-loop, 不能 fake。

**7. Participate in public discourse.** 数学家有责任 support serious science journalism, 在自己 subfield 里 contextualize AI-assisted claims。

## Infrastructure 和 workflow

Tao 介绍了一批已存在的 infrastructure:

- **Mathlib** (https://mathlib.org): Lean 4 的 unified math library, 1.5M+ lines formalized math。AI 做 autoformalization 的 target, 但 Lean kernel verification 是 trusted base, 不能被 AI 篡改
- **Mathematical Discourse** (https://www.mathematicaldiscourse.org): collaborative math discussion platform
- **Erdős Problems** (https://www.erdosproblems.com): Tao 维护的 open problems 数据库, 已经收到 AI-generated proof submissions
- **Optimization Constants Database** (https://github.com/teorth/optimizationproblems): Tao 自己的项目
- **SAIR Competitions** (https://competition.sair.foundation/competitions): controlled benchmark like First Proof

## Critical 评估

### Strengths

1. **Orthogonal decomposition 清晰** — 把 fuzzy question 拆成 capability vs values, 允许 conditional analysis
2. **Goal iteration 是真实 refinement process** — 每步识别 Goodhart failure mode, 加新 constraint
3. **Network flow / queueing intuition powerful** — 自然解释 bottleneck shift
4. **Thurston 引用** ground values 在 tradition, 不 ad hoc
5. **"Natural friction" 观察 genuinely novel** — 不在其他 AI-math commentary 见到

### Weaknesses

1. **Working Hypothesis 模糊度**: "reasonable" 可以 1% 也可以 99%, policy implication 差异巨大
2. **Community acceptance 量化 missing**: 没讨论如何形式化 "acceptance rate" 作为 time function
3. **Canonicalization stage 分析最薄**: 承认 "least amenable to optimization" 但没说 *how* 它会发生
4. **Cross-field heterogeneity**: Pure math vs applied math vs statistics 面临非常不同 AI impact, 未区分
5. **Economic dimension missing**: AI cost ($10–$1000/problem) vs human expert cost ($X/hour) 的 comparative 没讨论
6. **Adversarial / misuse scenario 缺失**: 如果 malicious actor 用 AI 生成海量 plausible-but-false proofs 灌满 arXiv, 是 denial-of-service attack on peer review。Tao 没明说

### 与其他 AI-math 声明的比较

- **Tim Gowers** (Polymath, formal proof): 更乐观 on capability, less focus on values
- **Akshay Venkatesh** (mentioned in acknowledgments): 更 philosophical, focus on AI 改变 mathematical *understanding* 本身
- **Kevin Buzzard** (Lean advocate): focus on formalization infrastructure, less on social epistemology
- **Leiden Declaration** (collective): Tao 的 talk 是 individual elaboration, 更 nuanced

参考:
- https://www.ias.edu/ideas/akshay-venkatesh-ai
- https://arxiv.org/abs/2310.04773 (autoformalization)
- https://arxiv.org/abs/2402.12934 (proof generation benchmarks)

## 一句话 take-away

Tao 的真正 contribution 不在评估 AI capability, 而在于 **forcing 数学界 explicitly articulate what it values, 然后展示 AI optimization 会把这些 values 撕开, 除非我们刻意保护数学实践中那些 slow, human, social stages**。

这是 21 世纪 metamathematics 的新形式 — 不再问 "what is a proof" (Hilbert program), 而问 "what is a *valued* proof, a *digested* proof, a *canonical* proof", 把答案 codify 进 institutional infrastructure。

Karpathy 你应该关心的 angle: 这个 framework 直接 applicable 到 ML research 本身。ML paper 的 "proof indigestion" 比 math 更严重 — benchmark 刷分容易, reproduction 难, canonicalization 几乎不存在。Tao 的 Goal v1→v5 迭代可以照搬到 ML community, 而且 bottleneck shift 已经在发生 (arXiv 每天刷爆, NeurIPS reviewer 累死, 但真正 canonicalized 的 idea 寥寥)。可以考虑把 Tao 的 network flow model formalize 成 ML-specific queueing model, 估 submission surge 对 reviewing system 的冲击。

---

# Terence Tao ICM 2026 演讲深度解析: Mathematics in the Age of AI

## 一、整体结构: 一个 metamathematical framework

这篇演讲的本质结构可以用一个 orthogonal decomposition 来理解。Tao 把"数学界如何应对 AI"这个大问题拆解成两个正交 subquestion:

**Dimension 1 (Capability axis):** AI Capability Conjecture — AI 能不能做数学?
**Dimension 2 (Values axis):** Goals and Values Question — 我们做数学究竟为了什么?

Tao 的核心 rhetorical move 是: **暂时悬置 Dimension 1, 强制听众 condition on Working Hypothesis, 然后专注分析 Dimension 2**。这是典型的 metamathematical 操作 — 借用数学的精确语言来澄清非数学问题。

这种 orthogonal decomposition 的隐喻直接对应 linear algebra 中的概念: 如果把 Community Response Question 看成一个 vector space $V$, Tao 选择不讨论 $\text{span}(\text{AI Capability Conjecture})$ 这个 subspace, 而是分析它的 orthogonal complement $V \ominus \text{span}(\text{AI Capability})$。Working Hypothesis 就是 "set this coordinate to 1, vary the others"。

## 二、历史前奏: foundations crisis 的类比

### 2.1 1900–1930 危机的技术内容

Tao 引用的两个事件需要 unpacking:

**Russell's paradox (1901):** 考虑 naive set theory 中的集合
$$R = \{ x \mid x \notin x \}$$
问 $R \in R$ 是否成立? 矛盾。这说明 Frege 的 naive comprehension axiom
$$\exists y \forall x (x \in y \leftrightarrow \varphi(x))$$
对任意 predicate $\varphi$ 都成立是 inconsistent 的。修复方案最终导向 Zermelo-Fraenkel set theory with Choice (ZFC), 其中 comprehension 被限制为 separation axiom:
$$\forall z \exists y \forall x \big( x \in y \leftrightarrow (x \in z \wedge \varphi(x)) \big)$$

变量含义: $z$ 是已存在的 set (作为 "原材料"), $y$ 是新构造的 subset, $\varphi$ 是任意 first-order formula。关键区别: 你不能凭空 "create" 一个 set, 只能从已有的 set 中 "filter"。

**Gödel incompleteness theorems (1931):** 对任何 consistent, recursively axiomatizable, sufficiently expressive theory $T$ (覆盖 arithmetic), 存在 sentence $G_T$ 使得
$$T \nvdash G_T \quad \text{且} \quad T \nvdash \neg G_T$$

第一定理构造的 $G_T$ 本质是 "this sentence is not provable in $T$", 通过 Gödel numbering 把 syntax arithmetize。第二定理进一步说 $T \nvdash \text{Con}(T)$, 其中 $\text{Con}(T)$ 形式化了 "$T$ 是 consistent 的"。

### 2.2 类比的结构

Tao 的类比不是 "AI 像 Gödel", 而是 **"我们正在经历 values/practices 层面的 foundation 危机, 产物将是 explicit, rigorous, standardized 的 values framework"**。注意历史上危机的产物是 ZFC + formal logic 的标准化, 这是 positive 的; 类似地, AI 危机的产物应该是 codified 的 mathematical practice guidelines (比如 Leiden Declaration)。

## 三、AI Capability Conjecture 的形式化

### 3.1 Template 形式

Tao 把 conjecture 写成一个有 placeholders 的 template:

$$\exists \, \text{time } t, \exists \, \text{AI tool } A, \exists \, \text{cost } c, \exists \, \text{supervision } s, \exists \, \text{field } F, \exists \, \text{success rate } r, \exists \, \text{quality } q:$$
$$\text{Capability}(A, t, c, s, F, r, q) \text{ holds}$$

每个 "some" 都是存在量词 $\exists$。不同填充方式得到不同版本的 conjecture。Weak form 例如: "在 $c = \$10^6$, $s =$ heavy expert guidance, $F =$ combinatorics, $r = 10\%$, $q =$ correctable draft" 下成立。Strong form: "在 $c = \$10$, $s =$ none, $F =$ all fields, $r = 90\%$, $q =$ publication-ready" 下成立。

### 3.2 First Proof benchmark 的数据

这是演讲中给出的唯一硬数据点:

| Variable | Value |
|---|---|
| Date | May 28, 2026 |
| Batch size | 10 research-level problems |
| AI harnesses tested | 4 |
| Problems solved at publication quality by ≥1 team | 7/10 |
| Compute cost per problem | $10 – $1000 |
| Refereeing | expert, correctness + exposition |

**关键 observation:** 70% success rate at sub-$1000 cost 是 strong evidence for at least medium-strong form of Capability Conjecture。但 Tao 强调 reporting bias — self-selected problem set, self-selected harnesses, no controlled baseline against human experts。

参考: https://1stproof.org

## 四、Working Hypothesis 与 conditional analysis

Working Hypothesis 是一个 imprecise quantifier:

$$\text{AI tools will, reasonably soon, become capable of performing a reasonable fraction of research-level mathematical tasks,}$$
$$\text{with reasonable levels of success, quality, supervision, and cost.}$$

"Reasonable" 是模糊的, 但 Tao 说精确值不重要 — 这是 conditional analysis。类比 probability 中的 conditioning: 我们计算
$$\mathbb{E}[\text{Response} \mid \text{Working Hypothesis}]$$
而不关心 $\mathbb{P}(\text{Working Hypothesis})$ 的精确值。

## 五、Goals 的迭代 refinement: 一个不动点过程

这是演讲的核心 technical contribution。Tao 把 "what is the goal of problem-solving" 看成一个不断 refinement 的过程, 每一步加入新的 constraint 来堵住 Goodhart's law 暴露的漏洞。

### 5.1 Goodhart's law 的形式化

$$\text{If } M \text{ is a proxy for goal } G, \text{ and we optimize } M \text{ directly, then } \text{cor}(M, G) \to 0$$

在 AI 语境下, 这特别危险因为:
1. Generative AI 本质 ungrounded (没有 ground truth signal in the training loop beyond next-token prediction)
2. AI 公司有 financial incentive 让 output "look like" 数学, 而非 "be" 数学
3. Automation makes optimization cheap, 容易 over-optimize

### 5.2 Goal 演化链

**Goal v1:** Solve as many unsolved problems as possible.

Formal: $\max \, |\{\text{open problems } P : P \text{ solved}\}|$

Failure mode: 虚假 proof 海啸 (Riemann hypothesis 被 "证明" 1000 次)。

**Goal v2:** Solve + verify.

Formal: $\max \, |\{P : \exists \text{ verified proof } \pi \text{ of } P\}|$

这里 "verified" 可以是 formal proof in Lean/Rocq/HOL。Verification 是 mechanical, AI-accelerated autoformalization 使成本 $\downarrow$。Failure mode: **unintelligible proofs** — 一个 10000 行 Lean proof 通过 kernel check, 但没有任何人类能 explain 它。

**Goal v3:** Solve + verify + communicate.

加入 exposition constraint。Formal: $\max \, |\{P : \exists \text{ verified, well-explained proof}\}|$

Failure mode: **over-polished exposition**。Tao 的精妙观察是: 人类写的 proof 有 natural friction — 作者卡住的地方, 读者也会卡住, 这种 friction 是 signal, 不是 noise。AI 把所有 friction 都抹平, 等价于把 "key idea" 和 "trivial computation" 一样对待, 读者失去 priority cue。

形式化地, 一个 proof 的 informational entropy profile:
$$H(\text{proof section } s) = -\sum_w p(w|s) \log p(w|s)$$
人类 proof 在 key lemmas 处 entropy 高 (作者 struggle, 写得啰嗦, 多次 restart); AI proof entropy 均匀。读者用 entropy gradient 作为 attention allocator。

**Goal v4:** Solve + verify + communicate + **digest and accept by community**。

加入 social epistemology 维度。一个 proof 必须 be:
- read by experts
- incorporated into others' work
- referenced, generalized, applied

这步引入 **external, slow, human process**。不能被 author 单方面 optimize。这是 Goodhart's law 的天然解药 — community acceptance 本质上 resistant to fast optimization。

**Goal v5 (final):** Solve + verify + communicate + digest + **canonicalize into definitive theory**。

终极阶段: 结果进入 textbook, 成为 standard curriculum。这是最慢的 stage, 需要 broad deliberative consensus。也是最有价值的 stage — 所有下游 application (包括 AI 自身的能力) 都依赖这个 canonicalized backbone。

### 5.3 网络流隐喻

Tao 把整个过程画成一个 flow network:

$$\text{Open problems} \xrightarrow{\text{proof generation}} \text{Solutions} \xrightarrow{\text{verification}} \text{Verified} \xrightarrow{\text{exposition}} \text{Readable} \xrightarrow{\text{acceptance}} \text{Published} \xrightarrow{\text{canonicalization}} \text{Definitive}$$

每条 edge 是一个 stage, 每个 stage 有 capacity。AI 加速前几条 edge (generation, verification), 但后几条 edge (acceptance, canonicalization) 是 human-bound, capacity 不变。结果: **bottleneck 转移到后面 stage**, 产生 "proof indigestion" — 一个 queueing theory 现象。

形式化: 设 stage $i$ 的 service rate 为 $\mu_i$, arrival rate 为 $\lambda_i$。在 steady state, $\lambda_i = \mu_{i+1}$, queue length $L_i = \lambda_i / (\mu_i - \lambda_i)$ (M/M/1 model)。当 AI 把 $\mu_1, \mu_2 \uparrow$ 而 $\mu_4, \mu_5$ 不变, 则 $L_3, L_4 \to \infty$。

参考 queueing theory: https://en.wikipedia.org/wiki/Queueing_theory

## 六、Bourgain 1991 paper fragment 解析

演讲中插入了一段 OCR 质量很差的 paper 片段, Tao 注明是 "A 1991 paper of Bourgain, annotated by my much younger self"。这是 Jean Bourgain 的 restriction theory 工作, 大概率是:

**Bourgain, J.** "Besicovitch type maximal operators and applications to Fourier analysis." *Geom. Funct. Anal.* 1 (1991), 147–187.

或相关的 $L^p$-restriction 论文。让我 reconstruct 关键公式。

### 6.1 Setup

- $S_{d-1} \subset \mathbb{R}^d$: unit sphere, $(d-1)$-dimensional manifold
- $\sigma$: surface measure on $S_{d-1}$
- $\varphi \in L^\infty(S_{d-1})$, $|\varphi| \leq 1$: bounded density on sphere
- $\widehat{\varphi \sigma}(\xi) = \int_{S_{d-1}} \varphi(z) e^{-2\pi i (z, \xi)} \, d\sigma(z)$: Fourier transform of the measure $\varphi \sigma$
- $B(0, \rho) \subset \mathbb{R}^d$: ball of radius $\rho$ centered at origin

**Goal of restriction theory:** 控制 $\|\widehat{\varphi \sigma}\|_{L^q(B(0,\rho))}$, 即 localized Fourier estimate。

### 6.2 Critical exponent

公式 (6.28):
$$q_0 = q_0(d) = 2 \cdot \frac{d+1}{d-1}$$

这是 **Stein-Tomas restriction exponent**。物理直觉: sphere 是 $(d-1)$-dim, 它的 Fourier transform 应该 decay like $|\xi|^{-(d-1)/2}$; 把这个 decay 与 $L^q$ scaling 结合, 得到临界值 $q_0 = 2(d+1)/(d-1)$。

对 $d=2$: $q_0 = 6$; 对 $d=3$: $q_0 = 4$; 对 $d=4$: $q_0 = 10/3$。

Tomas-Stein 定理: 对 $q \geq q_0$,
$$\|\widehat{f \sigma}\|_{L^q(\mathbb{R}^d)} \leq C \|f\|_{L^2(S_{d-1})}$$

### 6.3 Local estimate (6.27)

$$\|\widehat{\varphi \sigma}\|_{L^q(B(0,\rho))} \leq C_\epsilon \, \rho^{\frac{1}{q}\left(\frac{d+1}{q} - \frac{d-1}{2}\right) + \epsilon}$$

(我把 OCR 中乱码的指数重写为标准形式。)

变量解读:
- $\rho$: ball 半径
- $d$: dimension
- $q$: Lebesgue exponent
- $\epsilon > 0$: 任意小常数, $C_\epsilon$ blow up as $\epsilon \to 0$
- $(d+1)/q - (d-1)/2$: 一个 scaling exponent

**Scaling intuition:** $\widehat{\varphi \sigma}$ 是 oscillatory integral, 其 mass 在 frequency space spread over ball of size $\rho$。如果衰减 like $\rho^{-\alpha}$ 在 unit scale, localized $L^q$ norm 应 scale as $\rho^{d/q} \cdot \rho^{-\alpha} = \rho^{d/q - \alpha}$。这里 $\alpha = (d-1)/2 \cdot$ 之类, $\epsilon$ 是 endpoint loss。

### 6.4 Discrete restriction (6.30)

$$\left\| \sum_\alpha a_\alpha e^{i(x_\alpha, \xi)} \right\|_{L^q(B(0,R))} \lesssim R^{(d-1)(\frac{1}{2} - \frac{1}{q})} \left( \sum_\alpha |a_\alpha|^2 \right)^{1/2}$$

(这是 Bourgain 的 discrete restriction estimate 的标准形式, OCR 中的乱码应该是这样。)

变量:
- $\{x_\alpha\} \subset S_{d-1}$: $R$-separated points on sphere (cap $\{x_\alpha\}$ 是 $R$-sparse 的)
- $\{a_\alpha\}$: 任意 scalars
- $q \leq q_0$: exponent range
- $R$: frequency localization scale

**Intuition:** 这是 "square function" 类型的 estimate。Exponential sum $\sum a_\alpha e^{i(x_\alpha, \xi)}$ 在 ball $B(0,R)$ 上的 $L^q$ norm 由 $\ell^2$ norm of coefficients 控制, 带一个 $R$-dependent prefactor。$R$-separation 保证 exponentials "almost orthogonal", 这是 almost orthogonality / random-phase argument 的核心。

### 6.5 Partition argument (6.31)–(6.32)

把球面 $S_{d-1}$ 分成 cells $S_\alpha$ of size $\sim 1/\sqrt{\rho}$ (因为球面是 $d-1$ 维, cell 数量 $\sim \rho^{(d-1)/2}$), 每个 cell 选一点 $x_\alpha$ 作为 representative。

Taylor 展开:
$$\int_{S_\alpha} \varphi(x) e^{-2\pi i (x, \xi)} \, d\sigma(x) = e^{-2\pi i (x_\alpha, \xi)} \int_{S_\alpha} \varphi(x) e^{-2\pi i (x - x_\alpha, \xi)} \, d\sigma(x)$$

变量替换 $\xi' = \xi + \eta$, $\eta \in B(0, \rho^{1/2})$, 利用 $|x - x_\alpha| < c \rho^{-1/2}$ 在 cell 内 (cell 半径 $\rho^{-1/2}$), 所以 phase $(x - x_\alpha, \xi)$ 的 oscillation 在 $\rho \cdot \rho^{-1/2} = \rho^{1/2}$ 量级 — 这控制了 error term。

然后用 (6.30) 控制 discrete sum, 得到 (6.32) 的 square function estimate:
$$\|\widehat{\varphi \sigma}\|_{L^q(B(0,\rho))} \lesssim \rho^{\text{power}} \left\| \left( \sum_\alpha \left| \int_{S_\alpha} \varphi e^{-2\pi i (x-x_\alpha,\xi)} d\sigma \right|^2 \right)^{1/2} \right\|_{L^q(B(0,\rho))}$$

最后用一些 vector-valued estimate (像 Marcinkiewicz-Zygmund 或 Bourgain's vector-valued restriction) 把 square function 控制住, 得到 (6.27)。

**这整段在演讲中的作用:** Tao 用这段 OCR 噪声性的 paper 片段 illustration 两个点:
1. 人类 mathematician 写的 proof 有 natural friction — 公式编号、scope、notation 都 embedded 在 narrative 中
2. AI 重新 typeset 这种 paper 时会 smooth 掉所有 friction, 读者失去 "哪里是 hard part" 的 cue

参考 Bourgain restriction:
- https://en.wikipedia.org/wiki/Restriction_theorem
- https://www.math.stonybrook.edu/~aknapp/pos.html
- 原始 Bourgain GAFA 1991: https://link.springer.com/article/10.1007/BF01896369

## 七、Thurston 引用

> "We are not trying to meet some abstract production quota of definitions, theorems and proofs. The measure of our success is whether what we do enables people to understand and think more clearly and effectively about math."

这是 Thurston 1994 essay "On Proof and Progress in Mathematics" 的核心论点。Thurston 的 thesis 是: 数学是关于 **human understanding**, 不是 about formal proof artifacts。Proof 是 understanding 的载体, 不是 understanding 本身。

参考: https://arxiv.org/abs/math/9404236

Tao 引用 Thurston 是为了 ground Goal v3–v5 在已有 mathematical values tradition 中, 而非 novel reaction to AI。

## 八、Leiden Declaration 解析

Leiden Declaration on AI and Mathematics, 2026 年 6 月 2 日发布, DOI: 10.5281/zenodo.20302944。

参考: https://leidendeclaration.ai

### 8.1 Recommendation 1: Disclose tool use

要求 paper 包含 "Tool and computational resource disclosure" section。这对应 software engineering 中的 reproducibility norm。UNESCO Recommendation on Open Science 和 FAIR principles (Findable, Accessible, Interoperable, Reusable) 是 backing framework。

FAIR 原则的 formal 描述:
- **F**indable: globally unique, persistent identifier
- **A**ccessible: retrievable by standardized protocol
- **I**nteroperable: shared vocabulary, ontology
- **R**eusable: sufficient metadata, clear license

把 AI tool use 视作 research data, 要求 FAIR-compliant disclosure, 是一个 elegant extension。

### 8.2 Recommendation 2: Support reviewing

降低 proof generation emphasis, 增加 proof digestion emphasis。这是直接应对 Goal v4–v5 的 bottleneck。

具体技术建议:
- Provide formal proofs where feasible (Lean 4 / Rocq / HOL Light)
- Complete references to prior work (avoid AI hallucinated citations)
- Disclosure of tool use 减少 reviewer surprise

### 8.3 Recommendation 4: Retain responsibility for correctness

**Human authors remain exclusively responsible for correctness, adequacy, citation completeness.** 这是关键 legal/ethical boundary — 不能 blame AI。

### 8.4 Recommendation 6: Proper attribution

AI 在 attribution 上有 known limitations (不能 reliably trace idea provenance)。Tao 的 rule of thumb:

> If authors cannot convincingly demonstrate that they can give a clear, expert-level talk on their results, that is correct and properly attributed, then the result should not be published.

这是一个 operationalizable test — "talk test"。形式化: 设 $T(P)$ = "author can give expert-level talk on $P$", $P$ = paper。Require $T(P) = \text{true}$ for publication。这把 AI assistance 的 invisible 转化为 visible, 因为 talk 是 human-in-the-loop, 不能 fake。

## 九、Infrastructure 和 workflow 介绍

### 9.1 Mathlib

Lean 4 的 unified mathematical library, community-driven。包含 1.5M+ lines of formalized mathematics。

参考: https://mathlib.org

形式化一个 theorem $T$ in Lean 的 pipeline:
1. Human writes informal proof
2. Autoformalization (LLM 或 human) translates to Lean
3. Lean kernel type-checks
4. 如果失败, iterate

AI 在 step 2 的角色越来越重要, 但 step 3 (kernel verification) 是 trusted base, 不能被 AI 篡改。

### 9.2 Mathematical Discourse

一个 online platform for collaborative mathematical discussion。

参考: https://www.mathematicaldiscourse.org

### 9.3 Erdos Problems

Tao 维护的 Erdős 风格 open problems 数据库, 已经开始收到 AI-generated proof submissions。

参考: https://www.erdosproblems.com

### 9.4 Optimization Constants Database

Tao 自己的项目, 收录各种 mathematical optimization problems 中的 best known constants。

参考: https://github.com/teorth/optimizationproblems

### 9.5 SAIR Competitions

Scientific AI Reasoning competitions, controlled benchmarks 类似 First Proof。

参考: https://competition.sair.foundation/competitions

## 十、对 Tao 论证的 critical 评估

### 10.1 Strengths

1. **Orthogonal decomposition 清晰**: 把 fuzzy question 拆成 capability vs values 两个 axes, 允许 conditional analysis
2. **Goal iteration 是真实 refinement process**: 每步都识别 Goodhart failure mode, 加入新 constraint
3. **Network flow / queueing intuition** powerful: 自然解释 bottleneck shift
4. **Thurston 引用** ground values 在 tradition 中, 不是 ad hoc AI 反应
5. **"Natural friction" 观察**是 genuinely novel insight, 不在其他 AI-math commentary 中见到

### 10.2 Weaknesses / Open questions

1. **Working Hypothesis 模糊度**: "reasonable" 可以是 1% 也可以是 99%, policy implication 差异巨大
2. **Community acceptance 的量化 missing**: 没有讨论如何形式化 "acceptance rate" 作为 function of time, refereeing load, etc.
3. **Canonicalization stage 分析最薄**: 承认 "least amenable to optimization" 但没说 *how* 它会发生
4. **Cross-field heterogeneity**: Pure math vs applied math vs statistics 面临非常不同 AI impact, 演讲未充分区分
5. **Economic dimension**: AI cost ($10–$1000/problem) 与 human expert cost ($X/hour) 的 comparative 没讨论
6. **Adversarial / misuse scenario 缺失**: 如果 malicious actor 用 AI 生成海量 plausible-but-false proofs 灌满 arXiv, 是 denial-of-service attack on peer review。Tao 没明说

### 10.3 与其他 AI-math 声明的比较

- **Tim Gowers** (Polymath project, formal proof): 更乐观 on AI capability, less focus on values
- **Akshay Venkatesh** (mentioned in acknowledgments): 更 philosophical, focus on how AI 改变 mathematical *understanding* 本身
- **Kevin Buzzard** (Lean advocate): focus on formalization as infrastructure, less on social epistemology
- **Leiden Declaration** (collective): Tao 的 talk 是其 individual elaboration, 更 nuanced

参考 Venkatesh 的相关 talk:
- https://www.ias.edu/ideas/akshay-venkatesh-ai

## 十一、对未来研究方向的建议 (基于演讲框架)

1. **Quantitative epistemology**: 把 community acceptance 建模成 stochastic process, 测量 "$\text{time-to-canonicalization}$" distribution
2. **Proof information theory**: 形式化 "natural friction" 为 entropy gradient, 用作 AI exposition quality metric
3. **Reviewing queue theory**: 用 M/M/c queueing model 估计 AI-induced submission surge 对 referee system的影响
4. **Auto-attribution**: 用 ML 自动 trace idea provenance, 部分缓解 Recommendation 6 的 difficulty
5. **Canonicalization bottleneck simulation**: agent-based model of textbook adoption process

参考相关 ongoing work:
- https://arxiv.org/abs/2310.04773 (autoformalization)
- https://arxiv.org/abs/2402.12934 (proof generation benchmarks)
- https://arxiv.org/abs/2305.15692 (LLM for formal mathematics)

---

## 总结

Tao 这篇演讲的 deep structure 是:

**(foundation crisis analogy) → (orthogonal decomposition of Community Response) → (conditional on Working Hypothesis) → (goal iteration exposing Goodhart failure modes) → (network flow bottleneck shift → proof indigestion) → (Leiden Declaration as codified values) → (infrastructure layer to handle bottleneck)**

它的真正 contribution 不是给出 AI capability 评估 (Tao 明确说"not about that conjecture"), 而是 **forcing the mathematical community to explicitly articulate what it values, then showing that AI optimization will perturb these values apart unless we deliberately protect the slow, human, social stages of mathematical practice**。

这是 metamathematics 在 21 世纪的新形式 — 不再问 "what is a proof" (Hilbert program), 而是 "what is a *valued* proof, a *digested* proof, a *canonical* proof", 并把答案 codify 进 institutional infrastructure。
