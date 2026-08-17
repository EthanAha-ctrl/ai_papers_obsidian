---
source_pdf: Meta-Harness End-to-End Optimization of Model Harnesses.pdf
paper_sha256: 7d9b90f53a9f4801a090f1a4acb843340cde81e4f865f97d53fff2a6a570eb73
processed_at: '2026-08-05T17:54:17-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 Meta-Harness

## 一句话版本

LLM 系统里，包裹 model 的那层代码（harness）超级重要，能造成 6× 性能差距。现在大家手工调它，很慢很贵。这篇 paper 的做法是：**让一个 coding agent 自己搜 harness 代码，搜的时候能翻看所有历史尝试的源码和运行日志**，比手工调的好，比以前的自动搜索方法也好一大截。

---

## 为什么 harness 这件事重要

你用 Claude 写代码的时候，其实不只是 "Claude model" 在工作。背后有一堆代码决定：
- 给 Claude 看什么 context
- 存什么 memory
- 什么时候 retrieve
- 怎么 parse 它的 output
- 什么时候让它再试一次

这堆代码叫 harness。paper 引用了一个数字：同一个 model，只换 harness，性能差 6×。

这意味着什么？意味着你平时看 leaderboard 说"Claude 比 GPT 强 5 个点"，这个数字里其实混了 harness 的功劳。model 和 harness 是绑在一起比的，没法单独拎出来。当 harness 能造成 6× gap 的时候，leaderboard 排名的含义就变得很模糊了。

---

## 为什么以前不自动搜

其实有人试过自动搜 prompt、自动搜 code。比如 OPRO、TextGrad、AlphaEvolve、GEPA 这些方法。它们都能 iteratively 改进 text artifact。

问题在哪？**它们把 feedback 压缩得太狠了。**

打个比方。假设你在调一个复杂系统，每次改完跑一遍实验。有两种 feedback：

**Feedback A**（老方法）：跑完后有人告诉你"这次 35 分，上次 40 分，加油"。

**Feedback B**（Meta-Harness）：跑完后，整个 lab notebook 摊在你面前——你改了哪行 code、model 在 step 3 说了什么、step 7 为什么走偏、step 12 的 prompt 长什么样、tool call 返回了什么。你可以翻、可以 grep、可以对比上一次的 notebook。

调过复杂系统的人都知道，Feedback A 基本没用。你只知道"变差了"，但不知道为什么。Feedback B 才能让你 diagnose。

paper 给了一个 quantitative 对比，很震撼：

| 方法 | 每次能看到的 feedback 信息量 |
|------|---------------------------|
| OPRO | 0.002 million tokens |
| TextGrad | 0.015 |
| AlphaEvolve | 0.022 |
| GEPA | 0.008 |
| Meta-Harness | **10.0** |

差了 3 个数量级。这就是为什么老方法在 harness engineering 上不行——harness 是 stateful program，一个 early decision 会 cascade 影响后面很多步。你要 diagnose 一个 step 5 的错误，可能要 trace 回 step 1 的 prompt 构造逻辑。scalar score 和 summary 会把诊断信号丢掉。

---

## Meta-Harness 怎么做的

非常 simple 的 outer loop，用伪 code 说就是：

```
准备一个空文件夹 D
放几个 baseline harness 进去，跑一遍，把 code + 分数 + 运行日志存进 D
循环 N 次：
    coding agent 翻文件夹 D（用 grep、cat 这些标准工具）
    coding agent 写几个新 harness
    跑新 harness，把 code + 分数 + 日志存进 D
返回 Pareto frontier 上最好的那些
```

没了。就这么简单。

几个关键设计选择，每一个都是 deliberate 的"少做一点"：

### 1. 用 filesystem 当 feedback channel，不用 prompt packing

每次跑完一个 harness，产生一个文件夹，里面有：
- source code
- evaluation score
- execution traces（prompt、tool call、model output、state update）

一个 run 能产生 10 million tokens 的 diagnostic 信息。这塞不进任何 context window。所以 proposer 用 `grep` 和 `cat` 去 selectively 查询，而不是 bulk ingest。

### 2. Proposer 是 coding agent，不是 raw LLM

proposer 用 Claude Code（Opus-4.6）。它不是"给一个固定 prompt 让 LLM 生成 text"，它是一个 agent，能自己决定看什么文件、改什么 code、怎么验证。diagnosis 和 edit 决策全部 delegate 给 agent，不 hard-code 在 outer loop 里。

好处：**coding agent 变强，Meta-Harness 自动受益**。这是 future-proofing。

### 3. 不硬编码 parent selection

AlphaEvolve 这类方法有 tournament selection、固定 mutation operator。Meta-Harness 不做这些。proposer 想看哪个 prior harness 就看哪个，想做局部 edit 还是 full rewrite 都行。

### 4. 在 code space 搜，不在 prompt space 搜

prompt space 的 search 是 short-horizon 的——一个 prompt 一个 score。code space 是 long-horizon 的——一个 retrieval 逻辑的改变会影响后续整个 evaluation。code space 的好处还有 natural regularization：coding model 倾向于 propose coherent algorithm，不会给你写一堆 brittle if-else。

---

## 最精彩的部分：proposer 真的在"推理"

Appendix A.2 讲了一个 10-iteration 的 TerminalBench-2 search story，我看了觉得很震撼。让我复述一下：

**Iteration 1-2**：proposer 想修两个 bug，但同时改了 prompt template。两个 candidate 都从 64.4% baseline 大幅 regression（掉到 58.9% 和 57.8%）。

**Iteration 3**：proposer 做了一个 causal inference。它说：
> "前两次 regression 的 root cause 是 prompt template changes，不是 structural bugfixes。两次失败的共同因素是那个 cleanup-heavy prompt rewrite。我现在 revert 到 original prompt，只 test structural fix。"

这是教科书级别的 **confound identification**。它注意到两个失败 candidate 的共同点不是各自的 bugfix，而是共享的 prompt 改动。于是隔离变量，重新 test。

结果：63.3%，比之前好很多。支持了 confound diagnosis。

**Iteration 4-6**：直接修诊断出的 failure mode，还是 regress。proposer 学到一个 empirical lesson：**改 prompt 和 completion flow 是 high risk**。

**Iteration 7**：连续 6 次 regression 后，proposer 战略转向。它说：
> "前 6 次都失败了，因为我动了 completion flow、prompt template 或 observation processing。这次我换个思路——purely additive。在 agent loop 开始前，跑一个 shell 命令收集 environment snapshot，注入 initial prompt。不动任何现有逻辑。"

这个 candidate 成为 best result。

**Iteration 8**：试图把 winning idea 和 earlier structural fix 组合。

**Iteration 10**：甚至 reference 了另一个 separate search run 的 lesson："evolution history 显示 'don't cleanup service artifacts' 值 +18pp。"

这个故事让我觉得，proposer 不是 random mutation。它在做：
1. confound identification
2. hypothesis testing
3. strategic pivot（从 modify 转向 additive）
4. cross-run transfer

这些是 compressed-feedback 方法根本做不到的。你给它 scalar score，它只能知道"变差了"；你给它 raw traces，它能 infer "为什么变差了，根因在哪，下一步该怎么改"。

---

## 实验结果有多好

三个 domain，都很 strong：

### Text Classification

base model: GPT-OSS-120B。三个 dataset：LawBench（刑事罪名预测，215 类）、Symptom2Disease（22 类）、USPTO-50k（化学反应前驱体预测，180 类）。

对比 ACE（当前 SOTA hand-designed harness）：

| | 平均 accuracy | context token |
|---|---|---|
| ACE | 40.9% | 50.8K |
| Meta-Harness | **48.6%** | **11.4K** |

高 7.7 个点，context 只用 1/4。这是非常 strong 的 Pareto improvement。

对比其他 text optimizer（GEPA、OpenEvolve、TTT-Discover）：Meta-Harness 在 0.1× 的 evaluation 次数内 match 它们的性能，最终 accuracy 高 10+ 个点。

### Math Reasoning

setup：给 model 加 retrieval from 535K solved math problems。搜了 40 iterations，只看 GPT-OSS-20B 的性能选 harness。然后在 200 个 IMO-level problem 上 test，跨 5 个 held-out models。

| Method | 平均 accuracy gain |
|--------|-------------------|
| No retriever | baseline |
| Dense retrieval (k=5) | +4.0 |
| BM25 | +3.4 |
| **Meta-Harness** | **+4.7** |

同一个 harness，在 5 个没见过的 model 上都 improve。这说明 harness 捕获的是 model-agnostic 的 reasoning support pattern，不是某个 model 的 quirk。

### TerminalBench-2

89 个 hard agentic coding task。

Opus 4.6 上：Meta-Harness 76.4%，rank #2（第一名 ForgeCode 81.8% 但 paper 说无法 reproduce from public code）。

Haiku 4.5 上：Meta-Harness 37.6%，**rank #1**。

有意思的 observation：在 weaker Haiku 上的 improvement 比 Opus 上更大。intuition：weaker model 自己不能 compensate harness 缺陷，所以 harness 的 marginal value 更高。

---

## 搜出来的 harness 长什么样

这是我最喜欢的部分，因为 paper 把 search 的 output 变成 inspectable 了。

### Text Classification 的 8 个 Pareto-optimal variant

最便宜的那个叫 **Draft Verification**：
1. 第一次 call：retrieve 5 个最相似的 labeled example，做一个 draft prediction
2. 第二次 call：condition on draft label，retrieve 5 个同 label 的 + 5 个不同 label 的，让 model 决定 maintain 还是 revise

这就是 abductive reasoning + verification——先形成 hypothesis，再 actively test 它。第二次 retrieval 同时 condition on query 和 draft prediction，所以能 surface 针对当前 guess 的 counterexample。

最准的那个叫 **Label-Primed Query**，single call 三个 part：
1. **Label primer**：先列出所有 valid label，让 model 看到完整 answer space
2. **Coverage block**：每个 label retrieve 1 个 query-relevant example
3. **Contrastive block**：构建"很像但 label 不同"的 pair，sharpen local decision boundary

这种 structured prompt engineering 人很难 manually discover，因为它需要同时理解 retrieval、prompt structure、model behavior 三者。

### Math Retrieval 的 4-route program

一个 lexical router 把每个 problem 分到 4 个 route：

- **Combinatorics**：fetch 20 BM25 candidates → dedup 到 8 → rerank by score + difficulty → return top 3
- **Geometry**：return 1 个 hard NuminaMath reference + 2 个 raw BM25 neighbor（search 发现 raw match 比 difficulty rerank 好，这反直觉）
- **Number Theory**：fetch 12 candidates → bonus 给那些 "early state technique" 的 solution（偏好 proof strategy explicit 的）
- **Default**：fetch 10 candidates → adaptive 数量 based on score concentration

router 是 lightweight lexical predicate（keyword set + regex），简单到可以 verify。最终 harness 是两个 search lineage 的 merge，由 proposer autonomously 组合——这说明 Meta-Harness 能做 cross-lineage composition，类似 evolutionary crossover，但完全 agent 自主决定。

### TerminalBench-2 的 environment bootstrap

基于 Terminus-KIRA，加了一个 **environment bootstrapping**：在 agent loop 开始前，跑一个 shell 命令收集 sandbox snapshot（working dir、/app listing、available languages、package managers、memory），注入 initial prompt。

proposer 的 hypothesis：
> "在第一个 LLM turn 之前注入 environment snapshot，能在 dependency-heavy task 上节省 3-5 个 wasted exploration turn。"

这个改动只在 7/89 task 上 gain，但 largest gain 在 protein-assembly 和 path-tracing——这些 task 需要非 obviously available 的 domain-specific tooling（bioinformatics library、rendering pipeline）。没有 bootstrap 时 agent 花前 2-4 个 turn 探索 environment；在 turn budget 紧的 task 上，这些 wasted turn 决定 pass/fail。

这个 insight 可能 transfer 到其他 agentic setting：**environment non-obvious 时，bootstrap value 最大**。

---

## 一个关键 ablation：raw traces 是灵魂

这个 ablation 我觉得是 paper 最 important 的 single experiment：

| 给 proposer 看什么 | Median accuracy | Best accuracy |
|--------------------|-----------------|---------------|
| 只给 score | 34.6 | 41.3 |
| score + LLM 写的 summary | 34.9 | 38.7 |
| **全 access（raw traces）** | **50.0** | **56.7** |

两个发现：

1. **Raw traces 是关键**。从 34.6 跳到 50.0，15 个点的 gap。
2. **Summary 反而有害**。38.7 < 41.3。lossy compression 不只是丢信息，是 actively harmful——它会引入 spurious pattern 让 proposer 误判。

这个结果对整个 text optimization 领域都是一个 cautionary tale。大家为了 scalability 用 summary、用 template、用 scalar score，觉得"差不多够了"。但这个 ablation 显示，在 harness engineering 这种 long-horizon setting 下，compression 的代价可能远超预期。

---

## 我的几个 take

### Bitter Lesson 又来了

Sutton 2019 年那篇 Bitter Lesson 说：general-purpose method + more compute 最终会赢过 hand-engineered domain-specific solution。这个 pattern 在 game playing、architecture search、weight optimization 都出现过。Meta-Harness 又是一个 instance：自动 search 超过手工 harness engineering。

但 paper 更激进——它论证说 **search structure 本身也应该 delegate 给 general-purpose agent**。AlphaEvolve hard-code mutation operator、parent selection、tournament；Meta-Harness 全部 delegate。这让我想到一个 deeper question：AI 系统设计中，哪里该 hand-engineered，哪里该 learned？answer 似乎在不断向 "all learned" 移动。

### Filesystem 作为 universal interface

用 filesystem 当 feedback channel 是非常 clever 的设计。Filesystem 是 universal interface——所有 Unix 工具围绕它设计，coding agent 训练数据里充满 filesystem interaction。通过 filesystem，paper 把 "access to history" 问题转化为 "coding agent 的 standard workflow" 问题。

这也呼应 MemGPT [37] 的思路，但 MemGPT 把 OS abstraction 实现在 LLM 内部；Meta-Harness 用真实 OS filesystem，agent 通过 tools 访问，更 scalable——context 不需要塞进 LLM 的 context window。

### Meta-learning in code space

在我看来，paper 的核心贡献是把 meta-learning 从 weight space 移到 code space。传统 meta-learning（MAML、Andrychowicz 的 "learning to learn by gradient descent by gradient descent"）学一个 weight initialization，能在 few example 上 fast adapt。Meta-Harness 学一个 harness program，能在 fixed model 上 fast adapt to task distribution。

这个转移有几个 implications：
- code space 比 weight space 更 interpretable（discovered harness 是 readable Python）
- code space 比 weight space 更 inspectable for overfitting（brittle if-chain 可见，weight overfit 不可见）
- code space search 不需要 differentiability，可以用 arbitrary black-box evaluation
- code space 的 prior 来自 pre-trained LLM，不是 random init

一个 speculation：未来 LLM 系统 may 主要通过 harness search adapt to new task，而不是 fine-tuning。fine-tuning 需要数据、compute、小心 regularization；harness search 只需要 evaluation signal 和强 coding agent。在 many practical setting，后者更 accessible。

### Limitation

paper 自己承认的：只 test 了 Claude Code 作为 proposer，其他 coding agent 的 effect 未知。我补充几个没明说的：

1. **Cost**：每次 evaluation 10M tokens，60 个 harness 就是 600M tokens per run。这是为什么 paper 说 evaluation 是 main bottleneck。
2. **Search set contamination**：如果 search set 和 test set 太像，overfit 风险。paper 用 manual inspection + regex audit 缓解，但 labor-intensive。
3. **Proposer bias**：coding agent 可能有 systematic design pattern preference，限制 search space探索。
4. **Domain specificity**：虽然 cross-domain 和 cross-model generalization，但 harness 仍然 domain-specific。text classification harness 不能用于 math。
5. **Reproducibility**：ForgeCode 81.8% 无法 reproduce——agentic benchmark 的 reproducibility 问题是 industry-wide 的。
6. **Longer horizon**：实验里最长的 horizon 是 TerminalBench-2 的 task。真正 long-horizon agent（运行数小时、数天）的 harness search 还没 test。

---

## 总结

paper 做了三件事：

1. **诊断**：现有 text optimizer 的 feedback compression 在 harness engineering 上 fundamentally 不够，差 3 个数量级。
2. **方法**：filesystem + coding agent 的 minimal outer loop，把 diagnosis 和 edit 全 delegate 给 agent。
3. **验证**：text classification、math reasoning、agentic coding 三个 domain，自动发现的 harness 超过 hand-engineered SOTA。

最让我 excited 的不是数字，而是 Appendix A.2 的 narrative——proposer 在做 confound identification、strategic pivot、cross-run transfer。这种 causal reasoning over prior failures 是 compressed-feedback 方法根本无法支持的。

这篇 paper 可能是 harness engineering 自动化的 early-days 工作，就像 AlphaGo 之于 game playing。limitation 很多——cost 巨大、proposer capability 是 ceiling、generalization 仍然 domain-bounded。但方向 promising。当 coding agent 继续变强，Meta-Harness 自动 benefit，因为它故意把 outer loop 做得很 minimal，所有 intelligence 都在 agent 那一层。

---

# Meta-Harness: 让 coding agent 自己优化 LLM harness 的 end-to-end search

paper 链接：https://yoonholee.com/meta-harness/
代码：https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact
作者：Yoonho Lee, Roshen Nair, Qizheng Zhang, Kangwook Lee, Omar Khattab, Chelsea Finn（Stanford + MIT + KRAFTON）

---

## 1. 核心问题：harness 比 model 本身更重要

paper 一开篇就甩出一个关键事实：在同一个 benchmark 上，固定 model weights、只改 harness（包裹 model 的代码——决定 store/retrieve/present 什么信息给 model）能产生 **6×** 的性能差距，引用 SWE-bench Mobile [47]（https://api.semanticscholar.org/CorpusID:285462974）。

这个数字的含义非常深远。我们日常说"Claude 比 GPT 强多少"的时候，其实隐含了一个巨大的 confound：我们衡量的从来是"model + harness"的联合能力，model-only 的对比几乎不存在。当 harness 能造成 6× gap 时，leaderboard 上的排名到底是 model 的功劳还是 harness engineering 团队的功劳？这个问题没法简单回答。

paper 的核心 question：**harness engineering 能不能自动化？** 目前这个过程是纯手工的——practitioner 看 failure、调 heuristics、迭代少数 designs。这非常像 pre-RL 时代的 game playing：人类写 heuristic，机器执行。

---

## 2. 现有 text optimizers 为什么不够用？——3 个数量级的 feedback gap

这是 paper 最锋利的诊断。OPRO [51]、TextGrad [53]、AlphaEvolve [35]、GEPA [1]、Feedback Descent [26]、TTT-Discover [54] 这些 text optimization 方法，在 harness engineering 场景下统统不 work，根本原因是 **feedback 压缩太激进**。

具体三种表现：
- **memoryless**：只 condition on current candidate（Self-Refine [31]、OPRO）
- **only scalar scores**：AlphaEvolve、AdaEvolve [12] 只看分数
- **template/summary 限制 feedback**：GEPA、Feedback Descent 把 feedback 压成短模板

Table 1 给出 quantitative 证据（一次 evaluation 产生的 context tokens，MTok = million tokens）：

| Method | History | Log content | MTok/iter |
|--------|---------|-------------|-----------|
| OPRO | Window | (solution, score) pairs | 0.002 |
| TextGrad | Last | textual feedback on current | 0.015 |
| AlphaEvolve | Window | program database + scores | 0.022 |
| GEPA | Summary | reflective feedback | 0.008 |
| Feedback Descent | Summary | comparison + feedback | 0.012 |
| TTT-Discover | Window | prev. solution fragment | 0.026 |
| **Meta-Harness** | **Full** | **all logs and scores** | **10.0** |

**3 个数量级的差距**。这个数字单独看就很震撼。

为什么 harness engineering 需要这么多 feedback？因为 harness 是 **stateful program**——一个设计选择（比如 memory 存什么）会 cascade 穿过整个 evaluation sequence。一个在 reasoning step 5 出错的根因，可能在 step 1 的 prompt 构造逻辑里。要诊断这种 long-range causal chain，必须能 inspect raw execution traces；scalar scores 和 summaries 会 lossy 地丢掉诊断信号。

这给我一个重要 intuition：传统 text optimizer 在 prompt space 工作，prompt 是 short-horizon artifact——一个 prompt 一个 score，feedback loop 短。但 harness 是 long-horizon artifact，feedback loop 跨越整个 evaluation，需要 long-range credit assignment。这就是为什么 paper 在 Section 2 大量引用 meta-learning 文献 [40, 46, 3, 17, 44, 2]——Andrychowicz 的 "learning to learn by gradient descent by gradient descent" [3]、MAML [17]、prototypical networks [44]、Akyurek 的 in-context learning 理论 [2]。Meta-Harness 本质上是把 meta-learning 的 credit assignment 思想，从 weight space 转移到 code space。

---

## 3. 形式化目标——一个 outer-loop optimization

paper Section 3 给出干净的形式化：

$$
H^* = \underset{H}{\arg\max}\; \mathbb{E}_{x \sim \mathcal{X},\, \tau \sim p_M(H, x)}\; r(\tau, x)
$$

变量解释：
- $H$ = harness，一个 stateful program，wraps LLM $M$
- $M$ = frozen language model（base model，永远不变）
- $\mathcal{X}$ = task distribution
- $x$ = 一个 task instance，从 $\mathcal{X}$ 采样
- $\tau$ = rollout trajectory（model 在 harness 包裹下产生的完整轨迹）
- $p_M(H, x)$ = 给定 harness $H$ 和 task $x$，model $M$ 产生的 trajectory 分布（stochasticity 来自 model sampling）
- $r(\tau, x)$ = task-specific reward function，对 trajectory 打分
- $H^*$ = 最优 harness

这是一个 **bi-level optimization**：
- 内层（inference）：给定 $H$，model $M$ 在 task $x$ 上 rollout，产生 $\tau$
- 外层（search）：找最优 $H$，使 expected reward 最大

当多个 objective 相关时（accuracy 和 context cost），用 Pareto dominance 评估，report Pareto frontier。

---

## 4. Algorithm 1 的核心设计——把 outer loop 委托给 coding agent

```
1: Input: tasks X, LLM M, proposer P, iterations N
2: Initialize: population H   (initial set of valid harnesses)
3: Initialize: filesystem D = ∅   (stores code, scores, traces)
4: for H in H do
5:   E_H ← Evaluate(H, M, X)
6:   D ← D ∪ {(H, E_H)}
7: for t = 1...N do
8:   Proposer P queries filesystem D   (inspects prior harnesses and scores)
9:   Proposer P proposes k new harnesses {H_1, ..., H_k}
10:  for H in {H_1, ..., H_k} do
11:    if H passes interface validation then
12:      D ← D ∪ {(H, Evaluate(H, M, X))}
13: return Pareto frontier of harnesses stored in D
```

关键设计点（每一处都是 deliberate 的 simplicity）：

### 4.1 Filesystem 作为 feedback channel

每个 evaluated harness 在 filesystem 里有一个 directory，包含：
- source code
- evaluation scores
- execution traces（prompts、tool calls、model outputs、state updates）

proposer 通过标准 terminal 工具查询：
- `grep`：在所有 prior harnesses 里搜索 pattern
- `cat`：读 specific file
- 其他 Unix 工具

为什么用 filesystem 不用 prompt packing？因为 filesystem 比 context window 大几个数量级——一个 run 在 paper 最大设置下产生 10M tokens 的 diagnostic 信息。proposer 必须能 **selectively inspect**，不能 bulk ingest。

### 4.2 Proposer 是 coding agent，不是 raw LLM

这是 paper 最关键的设计选择。proposer 用 Claude Code [4]（Opus-4.6）。它不是 raw next-token model 操作在一个 outer-loop 组装的固定 prompt 上；它是一个 agent，能 retrieve information、navigate prior artifacts、edit code——这些都是 search 本身的一部分。

为什么这个选择 critical？因为 diagnosis 和 edit 决策被 delegate 给 coding agent，而不是 hard-code 进 outer loop。这意味着 Meta-Harness 会随着 coding agent 进步而自动 improve——这是 deliberate 的 future-proofing。

### 4.3 No parent-selection rule

和 AlphaEvolve/OpenEvolve 不同，Meta-Harness **不硬编码** parent 选择规则。proposer 可以 inspect 任何 prior harness 和它的 trace。这是 deliberate：通过把 diagnosis 和 edit 决策留给 proposer，Meta-Harness 能自动 benefit 于 coding agent 的进步。

### 4.4 Code-space search 的 regularization bias

虽然 search space 很大，但把 harness 表示为 programs 提供了 natural regularization bias：coding models 倾向于 propose coherent algorithms，而不是 brittle hard-coded solutions。这种 bias 把 search 引向 reusable context-management procedures。

这个 action space 和 frontier coding assistants 训练的 read-write-execute workflows 紧密 aligned——这是 Meta-Harness 能 work 的一个重要 enabler。

---

## 5. Appendix A：proposer 真的在做 causal reasoning（这是 paper 最精彩的部分）

paper 在 TerminalBench-2 的 10-iteration run 上记录了所有 file reads，quantitative 结果（Table 8）：
- **Median 82 files per iteration**（range 69–99）
- 41% 是 prior harness source code
- 40% 是 execution traces
- 6% 是 score summaries
- 13% 是其他文件

这个数字证明 proposer 是 **non-Markovian** 的——它 routinely inspect 大部分可用 history，远超最近 parent。这是 scalar-score optimizer 不可能支持的访问模式。

### 5.1 定性 narrative——一个 7-iteration 的 confound identification 故事

**Iteration 1-2**：proposer 把 plausible structural fixes 和 prompt-template edits 混在一起，两个 candidate 都从 64.4% 的 Terminus-KIRA baseline 大幅 regression（58.9% 和 57.8%）。

Iteration 1 hypothesis：
> "CMDEND marker fragments leak into LLM observations on long-running tasks, causing the model to get confused and enter infinite no-tool-call loops."

Iteration 2 hypothesis：
> "Double-confirmation completion mechanism causes verification spirals. Observed in trajectories where the agent solves the task early but burns 15–40+ additional steps re-verifying."

两个 candidate 都同时携带了 cleanup-heavy prompt rewrite。

**Iteration 3**：关键的 causal inference（这是 paper 的核心证据）：
> "Root cause of regressions: Prompt template changes (cleanup directives) caused the agent to delete necessary state before task completion. The structural bugfixes were confounded with harmful prompt changes. evo strip only isolates the two proven structural fixes."

proposer 注意到前两个失败 candidate 的 **共同因素** 不是各自的 bugfix，而是共享的 cleanup-heavy prompt rewrite。这是经典的 **confound identification**——它 revert 到 original prompt，只 test marker-stripping 和 loop-breaker。结果 63.3%，比之前好很多（虽然还是 -1.1pp），支持 confound 诊断。

**Iteration 4-6**：直接修复诊断出的 failure mode 仍然 regress。Iteration 4 提出具体 state-machine bug 假设：
> "Remove the two self.pending_completion = False lines that reset the completion flag when intermediate commands run. This fixes a state machine bug where: (1) Agent calls task complete → sees QA checklist, pending_completion = True (2) Agent runs verification commands → pending_completion = False (bug!) (3) Agent calls task complete again → sees checklist AGAIN → infinite loop."

proposer 甚至 cite 具体 trajectory 证据："configure-git-webserver produced baseline failures with agents stuck in 30–60 step verification spirals."

Iteration 6 尝试 systems-level optimization（smart-waiting）也 regress。到这时，proposer 学到了一个具体 empirical lesson：**修改 prompts 和 completion flow 是 high risk**。

**Iteration 7**：连续 6 次 regression 后，proposer **战略转向**——从 modifying control loop 转向在 loop 开始前添加信息：
> "All 6 prior iterations regressed from the 64.4% baseline because they modified the completion flow, prompt template, or observation processing. evo env bootstrap takes a different approach — purely additive. It gathers an environment snapshot via a single shell command before the first LLM call and appends it to the initial prompt."

这个 candidate 成为 run 中的 best result。关键不是它赢了，而是 proposer 能 **articulate 为什么它 safer**：避开 fragile 的 completion machinery，加 additive 信息只在 hard tasks 上有用。

**Iteration 8**：composition——试图把 winning idea 和 earlier structural fix 组合：
> "Combining two orthogonal fixes — env snapshot (saves early exploration turns) + marker stripping with no-tool-call loop breaker — will yield +1–3pp because they address independent failure modes without touching prompts or confirmation flows."

**Iteration 10**：**cross-run transfer**——reference 之前 separate search run 的结果：
> "The evolution history showed 'don't cleanup service artifacts' was worth +18pp."

这个 narrative 直接证明 paper 的核心论点：**full-history filesystem access 让 proposer 能 form 和 test causal hypothesis，并相应 revise harness，这是 compressed-feedback optimizer 无法支持的。**

---

## 6. 实验 1：Online Text Classification

数据集：LawBench [16]（215 classes，刑事罪名预测）、Symptom2Disease [19]（22 classes）、USPTO-50k [41]（180 classes，化学反应 precursor 预测）。base model: GPT-OSS-120B。

### 6.1 主结果（Table 2）

| Harness | USPTO | S2D | Law | Avg Acc | Ctx↓ |
|---------|-------|-----|-----|---------|------|
| Zero-shot | 12.0 | 63.2 | 7.0 | 27.4 | 0 |
| Few-shot (8) | 14.0 | 67.9 | 21.0 | 34.3 | 2.0 |
| Few-shot (32) | 13.0 | 72.2 | 21.0 | 35.4 | 7.9 |
| Few-shot (all) | 15.0 | 78.3 | 29.0 | 40.8 | 12.3 |
| MCE [52] | 14.0 | 83.0 | 23.0 | 40.0 | 28.5 |
| ACE [59] | 16.0 | 77.8 | 29.0 | 40.9 | 50.8 |
| **Meta-Harness** | **14.0** | **86.8** | **45.0** | **48.6** | **11.4** |

**Meta-Harness 比 ACE 高 7.7 points，同时 context 只用 1/4（11.4K vs 50.8K）**。这是非常 strong 的 Pareto improvement。

为什么这个对比特别有意思？ACE（Agentic Context Engineering，https://arxiv.org/abs/2510.04618）和 MCE（Meta Context Engineering，https://arxiv.org/abs/2601.21557）是当前 SOTA 的 hand-designed harness。ACE 用 reflective memory curation 在线构建 context；MCE 维护 evolving 的 natural-language skill library。Meta-Harness 自动发现的 harness 不仅 accuracy 更高，context 还更小——这是 Sutton's Bitter Lesson [45]（http://www.incompleteideas.net/IncIdeas/Bitter-Lesson.html）的又一个 instance。

### 6.2 Ablation——raw traces 是关键（Table 3）

| Method | Scores | Code | Summ. | Traces | Median↑ | Best↑ |
|--------|--------|------|-------|--------|---------|-------|
| Scores Only | ✓ | ✓ | ✗ | ✗ | 34.6 | 41.3 |
| Scores + Summary | ✓ | ✓ | ✓ | ✗ | 34.9 | 38.7 |
| Meta-Harness (full) | ✓ | ✓ | — | ✓ | 50.0 | 56.7 |

这是 paper 最 important 的 ablation。**关键发现**：summary 不但没补回 missing signal，反而 hurt performance（38.7 < 41.3）。raw execution traces 是关键 ingredient。

这个结果的含义：lossy compression 不只是丢信息，是 actively harmful——它会引入 spurious patterns 让 proposer 误判。这解释了为什么 GEPA 这类 reflection-based 方法在 short-horizon setting work，在 harness engineering 不 work。

### 6.3 vs 其他 text optimizers（Table 4, Figure 4）

| Method | Median | Best |
|--------|--------|------|
| GEPA [1] | 32.6 | 40.2 |
| Best-of-N | 34.0 | 44.2 |
| OpenEvolve [43] | 39.1 | 43.3 |
| TTT-Discover [54] | 34.1 | 45.6 |
| **Meta-Harness** | **50.0** | **56.7** |

Meta-Harness 在 **0.1× evaluations** 内 match best prior optimizer 的性能，最终 accuracy 比 best 高 **10+ points**。

注意：online text classification 是 paper 研究的最 small-context setting（Table 1）。如果在 small-context setting 上 structure-heavy optimizer 已经落后，在 harder regimes 上 gap 只会更大。

### 6.4 OOD generalization（Table 5）

9 个 unseen datasets：SciCite、FiNER、Amazon Reviews、Financial PhraseBank、GoEmotions、Banking77、AG News、SciTail、TweetEval。

- Zero-shot: 67.0%
- Few-shot (32): 69.6%
- ACE: 70.2%
- **Meta-Harness: 73.1%**

在 6/9 datasets 上 Meta-Harness 最高。重要观察：naively adding few-shot examples 超过 32 反而在 7/9 tasks 上 hurt——这暗示 hand-designed 的"多 example 总是更好" intuition 是错的，optimal 数量是 task-dependent 的，Meta-Harness 能 discover 这种 dependence。

---

## 7. 实验 2：Retrieval-Augmented Math Reasoning

这个 setup 有点 non-standard：给 model 加 retrieval from a corpus of solved math problems。

### 7.1 为什么这个 setup 有意思

理论上 retrieval 应该 help math——solutions 共享 reusable proof patterns。但实践上 retrieval 在 reasoning-intensive benchmarks 上一直不如 fact-grounded domains 成功 [42, 49, 6]。原因是 **naive retrieval 很少 surface right traces in right form**。所以成功的关键不是 retrieval per se，而是 **right retrieval policy**。

paper 给 Meta-Harness 一组 hard olympiad problems，让 retrieval behavior 自己 emerge。

### 7.2 Corpus（Table 10）

535,356 个 solved problems：
- OpenMathReasoning: 281,743（34% proof）
- DeepMath-103K: 103,021
- NuminaMath-1.5: 129,520（13% proof）
- PolyMath: 11,083
- Omni-MATH: 4,289
- FineProofs-SFT: 4,275
- AIME 1983–2024: 933
- Putnam-AXIOM: 492（100% proof）

Decontamination：exact prefix matching + fuzzy Jaccard similarity（threshold 0.8），任何 matching eval problem 的 corpus entry 都 discard。

### 7.3 主结果（Table 6）

5 个 held-out models，200 IMO-level problems（IMO-AnswerBench 100 + IMO-ProofBench 60 + ArXivMath Dec 2025 17 + ArXivMath Jan 2026 23），3 samples per problem，pass@1：

| Method | GPT-5.4n | GPT-5.4m | Gem-3.1FL | Gem-3F | GPT-20B | Avg |
|--------|----------|----------|-----------|--------|---------|-----|
| No Retriever | 23.0 | 28.8 | 28.6 | 42.6 | 47.6 | 34.1 |
| Dense (k=1) | 27.1 (+4.1) | 24.5 (-4.3) | 31.3 (+2.7) | 42.3 (-0.3) | 46.9 (-0.7) | 34.4 (+0.3) |
| Dense (k=5) | 31.1 (+8.1) | 28.3 (-0.5) | 37.1 (+8.5) | 47.2 (+4.6) | 46.7 (-0.9) | 38.1 (+4.0) |
| Random Few-shot | 23.1 (+0.1) | 24.5 (-4.3) | 31.0 (+2.4) | 40.4 (-2.2) | 41.8 (-5.8) | 32.2 (-1.9) |
| BM25 | 30.2 (+7.2) | 29.2 (+0.4) | 32.8 (+4.2) | 46.6 (+4.0) | 48.9 (+1.3) | 37.5 (+3.4) |
| **Meta-Harness** | **31.7 (+8.7)** | **30.4 (+1.6)** | **34.9 (+6.3)** | **46.3 (+3.7)** | **50.6 (+3.0)** | **38.8 (+4.7)** |

**关键观察**：
1. Meta-Harness 在 **所有 5 个 held-out models** 上都超过 no-retrieval baseline
2. 它用 **同一个 BM25 stack**（没有 dense encoder），却超过 dense retrieval (k=5) 的 average
3. 避免 dense retrieval 和 random few-shot 在某些 models 上的 regression

cross-model generalization 非常重要：search 只用 GPT-OSS-20B，但发现的 harness 在 GPT-5.4-nano、GPT-5.4-mini、Gemini-3.1-Flash-Lite、Gemini-3-Flash 上都 work。这暗示 harness 捕获的是 **model-agnostic 的 reasoning support patterns**，不是某个 model 的 quirks。

---

## 8. 实验 3：TerminalBench-2——agentic coding 的 frontier

TerminalBench-2 [33]（https://arxiv.org/abs/2601.11868）：89 个 challenging tasks，要求 long-horizon autonomous execution。

paper 把这个 benchmark 当作 "discovery problem" [54]——目标是 discover harness configuration 改善 benchmark 性能。这是 standard practice（[18, 34, 25] 都描述了 TerminalBench-specific 的 harness iteration），但 paper 也 manual inspect 和 regex audit 检查 task-specific string leakage。

### 8.1 主结果（Table 7）

**Claude Opus 4.6 leaderboard**：
- Claude Code: 58.0%
- Terminus 2 [33]: 62.9%
- Mux: 66.5%
- Droid: 69.9%
- TongAgents: 71.9%
- MAYA-V2: 72.1%
- Terminus-KIRA [25]: 74.7%
- Capy: 75.3%
- **Meta-Harness: 76.4%（rank #2）**
- ForgeCode: 81.8%（paper 注明无法 reproduce from public code）

**Claude Haiku 4.5 leaderboard**：
- OpenHands: 13.9%
- Claude Code: 27.5%
- Terminus 2: 28.3%
- Mini-SWE-Agent: 29.8%
- Terminus-KIRA: 33.7%
- Goose: 35.5%
- **Meta-Harness: 37.6%（rank #1）**

**反直觉的 observation**：在 weaker Haiku 4.5 上的 improvement（+2.1 over next-best Goose）比 Opus 4.6 上（+1.1 over Terminus-KIRA）更大。interpretation：**在 weaker model 上，harness 的 marginal value 更高**，因为 model 自己不能 compensate harness 缺陷。这呼应 [42]（https://arxiv.org/abs/2602.07213）的 observation：adaptive retrieval 在 weaker model 上更有效。

---

## 9. Discovered Harnesses——paper 揭示 search 到底找到了什么（Appendix B）

这部分让我非常 excited，因为它把 search 的 output 变成 inspectable 的 design patterns。

### 9.1 Text Classification——8 个 Pareto-optimal variants（Table 9）

| Variant | USPTO | S2D | Law | Avg | Ctx |
|---------|-------|-----|-----|-----|-----|
| Draft Verification | 18.0 | 85.4 | 17.0 | 40.1 | 5.4K |
| Error-Annotated | 9.0 | 87.7 | 24.0 | 40.2 | 22.3K |
| CoT Replay | 13.0 | 88.2 | 25.0 | 42.1 | 23.3K |
| Cluster Coverage | 12.0 | 86.8 | 33.0 | 43.9 | 31.2K |
| Cascade Retrieval | 12.0 | 86.8 | 36.0 | 44.9 | 39.2K |
| RRF + Contrastive | 18.0 | 89.6 | 35.0 | 47.5 | 41.4K |
| Relevance + Contrastive | 18.0 | 90.6 | 36.0 | 48.2 | 43.9K |
| **Label-Primed Query** | 14.0 | 86.8 | 45.0 | **48.6** | 45.5K |

#### Draft Verification（lowest context frontier point）

两-stage 程序：
- **Stage 1 (Draft)**：retrieve 5 nearest labeled examples，做 initial prediction
- **Stage 2 (Verification)**：condition retrieval on draft label，retrieve 5 confirmers（同 label）+ 5 challengers（不同 label），让 model decide maintain 或 revise
- **Cold start**：<5 labeled examples 时 fallback 到 single-call few-shot

**关键 insight**：第二次 retrieval 同时 condition on query 和 draft prediction，所以能 surface 针对当前 guess 的 counterexamples，而不是 generic near neighbors。这非常像 **abductive reasoning + verification**——先形成 hypothesis，再 actively test 它。

#### Label-Primed Query（highest accuracy frontier point，main text 用的）

single larger call，三个 part：
- **Label primer**：先列出所有 valid output labels，让 model 看到完整 answer space（before any examples）
- **Coverage block**：对每个 label，retrieve 1 个 query-relevant example（guarantee every label 有代表）
- **Contrastive block**：构建 highly similar examples with different labels 的 pairs，sharpen local decision boundaries
- **Retrieval rule**：TF-IDF similarity + query-anchored partner selection（不是 label-agnostic nearest neighbors）

这个 design 非常 elegant。它把"理解 label space"、"exemplify each label"、"sharpen local boundary"三个功能 **stratified 组合**。这种 structured prompt engineering 是 human 很难 manually discover 的——它需要同时理解 retrieval、prompt structure、model behavior 三者。

### 9.2 Math Retrieval Harness——一个 4-route BM25 program

Lexical router 把每个 problem 分到 4 个 route 之一：

- **Combinatorics**：fetch 20 BM25 candidates → dedup 到 8 → 按 lexical score + difficulty rerank → return top 3。这是 explicit 的 **diversity vs hard-problem matching tradeoff**。
- **Geometry**：return 1 hard NuminaMath reference + 2 raw BM25 neighbors。Search consistently 偏好 raw structural matches over difficulty reranking——这反直觉，但 search 发现了。
- **Number Theory**：fetch 12 BM25 candidates → 按 lexical score + difficulty + small bonus（solutions that state technique early）rerank。这偏好 **proof strategy explicit** 的 examples。
- **Default**：fetch 10 BM25 candidates → 按 lexical score + difficulty rerank → 根据 top retrieval scores 的 concentration 选 adaptive 数量。

**关键 design 细节**：
1. Routes 是 lightweight lexical predicates（keyword sets + regex for geometry notation）——简单到可以 verify，没有 learned classifier
2. 不 cross-route aggregate：选定 route 后只用那个 route retrieve
3. BM25 index 用 **math-aware tokenizer**，保留 LaTeX tokens（如 `\frac`, `\{2\}`）作为 atomic units——这是 critical 的实现细节
4. 最终 harness 是 **两个 search lineage 的 merge**，由 proposer autonomously 组合：一个贡献了更强的 geometry route，另一个贡献了更强的 combinatorics route

最后一点非常 important——它证明 Meta-Harness 能 **do cross-lineage composition**，这是 evolutionary search 里的 crossover 操作，但完全由 agent 自主决定，不需要 hard-coded crossover operator。

### 9.3 TerminalBench-2 Harness——environment bootstrapping

基于 Terminus-KIRA [25]，inherit 它的：
- Native tool calling（替代 Terminus 2 的 ICL-based JSON parsing）
- 30KB output cap
- Multiperspective completion checklist

Meta-Harness 发现的关键 modification 是 **environment bootstrapping**：在 agent loop 开始前，运行一个 compound shell command 收集 sandbox snapshot，注入 initial prompt。

Snapshot 包含：
- Working directory
- `/app` listing（大目录 truncate 到 20 entries）
- Available languages + versions（Python, GCC, G++, Node, Java, Rust, Go）
- Installed package managers（pip, apt-get）
- Available memory

proposer 的 hypothesis（verbatim）：
> "Injecting an environment snapshot (OS, installed languages, package managers, /app contents) before the first LLM turn will reduce wasted exploration episodes by 3–5 turns on dependency-heavy tasks"

实现细节：80 行 code on top of Terminus-KIRA，15-second timeout，fail silently。

**Per-task analysis**：在 7/89 tasks 上 gain，largest 在 protein-assembly 和 path-tracing。这些 tasks 共享 property：需要 domain-specific tooling（bioinformatics libraries, rendering pipelines, chess engines, cryptographic utilities, CoreWars simulators），availability 不能 assumed。没有 bootstrap 时，agent 花前 2-4 个 turn 探索 environment；在 tight turn budget 或 early wrong assumptions cascade 的 tasks 上，这些 wasted turns 决定 pass/fail。

**这个发现揭示的 intuition**：bootstrap 的 value 在 environment non-obvious 且 task 需要 agent match strategy to installed tools 时最大。这是一个非常 general 的 insight，可能 transfer 到其他 agentic settings。

---

## 10. Practical Implementation Tips（Appendix D）

这部分是 engineering lessons，对想 reproduce 的人非常有用：

1. **Write a good skill**：skill text 是 steering search 的 primary interface。constrain outputs 和 safety-relevant behavior，**不** constrain diagnosis procedure。从 log 观察：accumulated traces 在足够 iterations 后比 skill 本身更能 shape proposer behavior。Iterate on skill text 比 change iteration count 影响更大。

2. **Start with a baseline harness and a hard search set**：写 simple baseline，construct search set by filtering baseline gets wrong。50–100 examples 足够。fast, discriminative eval 比 large eval 更 valuable。

3. **Log everything in navigable format**：JSON, hierarchical organization, consistent file names, 让 grep/regex work well。

4. **Make logs queryable through a small CLI**：list Pareto frontier, show top-k harnesses, diff code/results。Converting offline experience to same directory structure 可以 warm-start exploration。

5. **Lightweight validation before expensive benchmarks**：写小 test import module, instantiate class, call methods on tiny example set。catch malformed candidates in seconds，keep cost of failures near zero。

6. **Automate evaluation outside the proposer**：running evals 不值得让 proposer 做。separate harness score candidates 并 write results to filesystem。

---

## 11. 我对这篇 paper 的整体 intuition

### 11.1 Bitter Lesson 的 next instance

Sutton's Bitter Lesson [45] 的 pattern 在 Meta-Harness 上重复：**一旦 search space 变得 accessible，strong general-purpose agents 能 outperform hand-engineered solutions。**

这个 pattern 在 game playing（AlphaGo）、architecture search（NAS）、weight optimization（SGD vs hand-crafted features）都出现过。Meta-Harness 把它带到 harness engineering。

但 paper 更激进：它论证说 **search structure 本身也应该 delegate 给 general-purpose agent**。AlphaEvolve/OpenEvolve hard-code mutation strategies、parent selection、tournament；Meta-Harness 全部 delegate 给 coding agent。这让我想到一个 deeper question：在 AI 系统设计中，哪里应该是 hand-engineered 的，哪里应该是 learned 的？目前 answer 似乎在不断向 "all learned" 移动。

### 11.2 Filesystem 作为 universal context engineering interface

用 filesystem 作为 feedback channel 是非常巧妙的设计。Filesystem 是 universal interface——所有 Unix 工具围绕它设计，coding agents 训练数据里充满 filesystem interactions。通过 filesystem，paper 把 "access to history" 问题转化为 "coding agent 的 standard workflow" 问题。

这和 MemGPT [37]（https://arxiv.org/abs/2310.08560）思路相似，但 MemGPT 是把 OS abstractions 实现在 LLM 内部；Meta-Harness 用真实 OS filesystem 作为 external memory，agent 通过 tools 访问。后者更 scalable——context 不需要塞进 LLM 自己的 context window。这也呼应 Recursive Language Models [56]（https://arxiv.org/abs/2512.24601）的思路。

### 11.3 Proposer capability 是 ceiling

paper 自己承认 limitation："our experiments demonstrate that harness search can work with one particularly strong coding-agent proposer (Claude Code); a broader study of how the effect varies across proposer agents remains for future work."

这意味 Meta-Harness 的 capability ceiling 是 coding agent 的 capability。当 coding agent 更强时，Meta-Harness 自动 benefit——这是 deliberate simplicity 的回报。但 weak coding agent 可能无法 do 这种 causal reasoning over traces。

### 11.4 Harness Engineering vs RLHF——complementary not competing

harness engineering 和 RLHF 都是"塑造 model behavior"的方法。RLHF 通过 weight update，harness engineering 通过 context manipulation。两者 complementary——paper 在 Discussion 提到 future work：**co-evolve harness 和 model weights**。让 strategy shape what model learns，vice versa。

这个方向非常有前途。如果 model 在训练时就适应某种 harness 结构，两者能 joint optimize 到更好的 local optimum。这让我想到 RLHF 里 reward shaping 的难题——如果 harness 能提供更好的 training-time context，RLHF 可能 converge 到更好的 policy。

### 11.5 和 Software 3.0 的联系

Karpathy 一直讲 Software 1.0/2.0/3.0：1.0 是 human-written code，2.0 是 learned weights，3.0 是 prompts/in-context programs。Meta-Harness 在 Software 3.0 里做 search——把"什么 prompt/context"作为优化目标。

这和 nanoGPT、llm.c 的精神一致：用最少 abstraction，直接暴露问题本质。Meta-Harness 的 outer loop 极 minimal——没有 fixed scaffold，没有 archive，没有 persistent memory，只有 filesystem + coding agent。这种 simplicity 是它的 power。

### 11.6 Limitations paper 没明说

1. **Cost**：每次 evaluation 10M tokens 巨大。60 harnesses × 10M = 600M tokens per run。这就是为什么 paper 说 evaluation 是 main computational bottleneck。
2. **Search set contamination**：如果 search set 和 test set 太相似，overfit 风险。paper 用 manual inspection + regex audit 缓解，但 labor-intensive。
3. **Proposer bias**：coding agent 可能有 systematic biases（code style、design pattern preference），限制 search space 探索。
4. **Domain specificity**：虽然 cross-domain 和 cross-model generalization，但 harness 仍然 domain-specific。text classification harness 不能用于 math。
5. **Reproducibility**：ForgeCode 81.8% 无法 reproduce——这暴露了 agentic benchmark 的 reproducibility 问题。Meta-Harness 76.4% 也需要独立确认。
6. **Generalization to longer horizons**：实验里最长的 horizons 是 TerminalBench-2 的 long-horizon tasks。真正 long-horizon agents（运行数小时、数天）的 harness search 还没 test。

### 11.7 一个更深层的思考——meta-learning 在 code space

paper 的核心贡献，在我看来是把 meta-learning 从 weight space 移到 code space。传统 meta-learning（MAML [17]、Andrychowicz [3]）学一个 weight initialization，能在 few examples 上 fast adapt。Meta-Harness 学一个 harness program，能在 fixed model 上 fast adapt to task distribution。

这个转移有几个 implications：
- code space 比 weight space 更 interpretable（discovered harnesses 是 readable Python）
- code space 比 weight space 更 inspectable for overfitting（brittle if-chains 可见，weight overfit 不可见）
- code space 的 search 不需要 differentiability，可以用 arbitrary black-box evaluation
- code space 的 prior 来自 pre-trained LLM，而不是 random initialization

这让我想到一个 speculation：未来 LLM 系统 may 主要通过 harness search 而不是 fine-tuning 来 adapt to new tasks。fine-tuning 需要数据、compute、小心 regularization；harness search 只需要 evaluation signal 和强 coding agent。在 many practical settings，后者更 accessible。

---

## 12. 总结

Meta-Harness 是一篇很有思想深度的 paper。它做三件事：
1. **诊断**：现有 text optimizer 的 feedback compression 在 harness engineering 上 fundamentally 不够（3 个数量级 gap）
2. **方法**：用 filesystem + coding agent 的 minimal outer loop，delegate diagnosis 和 edit 给 agent
3. **验证**：在 text classification、math reasoning、agentic coding 三个 domain 上，自动发现的 harness 超过 hand-engineered SOTA

最让我 excited 的不是数字，而是 Appendix A.2 的 narrative：proposer 真的在做 confound identification、strategic pivot、cross-run transfer。这种 causal reasoning over prior failures 是 compressed-feedback 方法根本无法支持的。

paper 的 limitation 也很明显：cost 巨大、proposer capability 是 ceiling、generalization 仍然 domain-bounded。但方向非常 promising——这可能是 harness engineering 自动化的 early-days 工作，就像 AlphaGo 之于 game playing。

---

参考链接：
- paper 主页：https://yoonholee.com/meta-harness/
- 代码：https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact
- ACE [59]：https://arxiv.org/abs/2510.04618
- MCE [52]：https://arxiv.org/abs/2601.21557
- AlphaEvolve [35]：https://arxiv.org/abs/2506.13131
- OpenEvolve [43]：https://github.com/algorithmicsuperintelligence/openevolve
- GEPA [1]：https://arxiv.org/abs/2507.19457
- TTT-Discover [54]：https://arxiv.org/abs/2601.16175
- Terminus-KIRA [25]：https://github.com/krafton-ai/kira
- TerminalBench-2 [33]：https://arxiv.org/abs/2601.11868
- MemGPT [37]：https://arxiv.org/abs/2310.08560
- Recursive Language Models [56]：https://arxiv.org/abs/2512.24601
- Feedback Descent [26]：https://arxiv.org/abs/2511.07919
- Sutton's Bitter Lesson [45]：http://www.incompleteideas.net/IncIdeas/Bitter-Lesson.html
- ADAS [20]：https://openreview.net/forum?id=t9U3LW7JVX
- MathArena [6]：https://matharena.ai/
- DSPy [23]：https://arxiv.org/abs/2310.03723
- LangChain [13]：https://github.com/langchain-ai/langchain
