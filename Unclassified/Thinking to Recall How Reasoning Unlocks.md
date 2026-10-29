---
source_pdf: Thinking to Recall How Reasoning Unlocks.pdf
paper_sha256: b1a83ea2d1f48257c06806b326a8d1e513739c84c3f5ecbca7f9b09898a9fbde
processed_at: '2026-08-12T15:35:28-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话再讲一遍

Andrej，我把学术腔全扔掉，像在咖啡店跟你聊这篇 paper 那样讲。

---

## 这篇 paper 到底在问什么

一个非常让人不舒服的事实：你拿一个 reasoning model（比如 Gemini 2.5 Flash、Qwen3-32B），问它"Mary Engle Pennington 哪年进的国家发明家名人堂？"这种**一个 hop、不需要任何推理、Wikipedia 一查就有**的问题。

按理说 reasoning 在这里应该毫无用武之地——又没有多步逻辑可拆。

但你把 thinking 关掉 vs 开起来对比，**开 thinking 的 pass@k 明显更高**，而且 k 越大差距越大。在某些 setting 下 pass@k 直接翻倍。

这就很诡异了。reasoning 在一个"根本不需要 reasoning"的任务上 work，那它到底在干嘛？

这篇 paper 就是把这个诡异现象拆开来看。

---

## 核心发现一句话版

**reasoning 在这里不是在做逻辑推导，而是在帮模型从自己的 parameters 里把知识捞出来。** 模型其实"知道"答案（weights 里存了），但直接问它答不出来。thinking 给了它一个"捞"的机制。

这跟你之前讲过的 "LLM 是在做 recall 而不是 reasoning" 的 intuition 高度吻合。

---

## pass@k 为什么是关键指标

pass@1 就是"问一次答对不对"。但作者关心的是更深的问题：**模型的 output distribution 里到底存不存在一条 path 能到达正确答案**。

用 pass@k 的公式：

$$\text{pass@}k = 1 - \frac{\binom{N-c}{k}}{\binom{N}{k}}$$

- $N$：总共采样多少次（论文用 100）
- $c$：其中正确的有多少
- $k$：假设你能取几个 sample
- 含义：从 N 个 sample 里随机抓 k 个，至少抓到一个正确的概率

为什么这个指标重要：如果 reasoning 只是让本来就 likely 的答案更 likely（sharpening distribution），那 OFF 和 ON 在大 k 下应该 converge。但 Figure 1 显示它们不 converge，**ON 模式在高 k 下持续高于 OFF**。

这意味着 reasoning 打开了一些"OFF 模式下采样 100 次都碰不到"的 knowledge。这不是 sharpening，是 **boundary expansion**。

参考 pass@k 原始定义：[Chen et al. 2021 HumanEval](https://arxiv.org/abs/2107.03374)

---

## $\Omega$ metric：把整条 pass@k 曲线压成一个数

为了好比较，作者定义了 $\Omega(N)$：

$$\Omega(N) = \sum_{k=1}^{N} \left[ k \cdot \frac{\text{pass@}k_{\text{ON}} - \text{pass@}k_{\text{OFF}}}{\text{pass@}k_{\text{OFF}}} \right] \cdot \frac{1}{\sum_{k'=1}^{N} k'}$$

人话翻译：
- 对每个 k，算 ON 比 OFF 好多少（用相对比例，因为 SimpleQA baseline 低，绝对差不好比）
- 用 k 当权重，**k 越大权重越高**——因为作者关心的是 capability boundary，大 k 的 gain 更能说明"解锁了原本不可达的知识"
- 最后归一化成 percentage

$\Omega$ 越大 = reasoning 帮得越多。

Figure 2 的 trend：**越弱的模型 $\Omega$ 越大**。Qwen3-32B 的 $\Omega$ 远高于 Gemini-2.5-Pro。解读：弱模型 hidden knowledge 多但 recall 不到，reasoning 帮它挖；强模型 OFF 模式下已经把大部分东西召回了，reasoning 边际收益小。

---

## 第一个被否定的假说：复杂问题分解

最自然的解释是 Press et al. 2023 那套：reasoning 帮你把 multi-hop 问题拆成 single-hop ([paper](https://aclanthology.org/2023.findings-emnlp.378/))。

但作者用的 EntityQuestions 是 template-based single-hop（"Who is [X] married to?" 这种），SimpleQA-Verified 里 90% 也是 single-hop。而且 SimpleQA-Verified 的 metadata 标了哪些问题"requires reasoning"或"multi-step"。

Figure 3 比较 Complex subset 和 Simple subset 的 $\Omega$：**95% 置信区间重叠**，没有证据表明 Complex 更受益。Gemini-2.5-Pro 在 Complex subset 上 CI 甚至跨过 0。

人话：**reasoning 在"标注为需要 reasoning"的问题上并没有特别占便宜**。这说明 reasoning 的 utility 不是来自问题分解。

---

## 机制一：Computational Buffer（算力缓冲）

### 假说

Transformer 一次 forward pass 的 depth 是固定的。难问题需要更多计算怎么办？**生成 token 等于把纵向 depth 换成横向 length**。每生成一个 token，新 token 的 query 会 attend 之前所有 thinking tokens 的 keys，相当于做了一轮"迭代计算"。

这是 Goyal et al. 2024 pause tokens ([paper](https://openreview.net/forum?id=ph04CRkPdC)) 和 Pfau et al. 2024 dot-by-dot ([paper](https://arxiv.org/abs/2404.15758)) 的核心 idea，但在现代 R-LLM 上没人直接测过，特别是在 factual recall 上。

### 实验设计特别干净

| 模式 | trace 内容 | 测什么 |
|------|-----------|--------|
| OFF | 无 | baseline |
| ON | 原始 trace | 上限 |
| ON Dummy | "Let me think." 重复到原 trace 长度 | 纯 compute，无 semantic |
| ON Single Dummy | 单次 "Let me think." | ON mode bias control |

ON Dummy vs ON Single Dummy 的区别**只有 length**——都在 ON 模式，都没 semantic content。所以如果 ON Dummy 比 ON Single Dummy 好，那 **纯粹是额外 compute 的功劳**。

### 结果

Figure 4：ON Dummy 远好于 OFF，也显著好于 ON Single Dummy。

pass@1 具体数字：
- SimpleQA-Verified: 0.206 → 0.262（OFF → ON Dummy）
- EntityQuestions: 0.457 → 0.554

**纯粹给模型更多"思考空间"就能提升 recall**，哪怕思考内容是无意义的"Let me think. Let me think. ..."。

### Length scaling 有甜点区

Figure 5 扫描 dummy 长度：在 SimpleQA-Verified 上，dummy 长度增长到 2048 tokens 之前一直改善，但 4096、8192、16384 反而下降。

人话：**compute 不是越多越好**。太短了不够"scratch"，太长了 attention 散了或者 distribution drift。

这跟 Hassid et al. 2025 "Don't overthink it" ([paper](https://arxiv.org/abs/2505.17813)) 的发现一致，但这里更精细——他们 controlling semantic content 后才得出"compute 本身就有甜点区"的结论。

### 但 compute 解释不了全部

关键：**ON Dummy 永远到不了 ON 的表现**。Figure 4 里 ON Dummy 的曲线一直在 ON 之下。

所以剩下那部分 gain 必须来自 reasoning trace 的 **semantic content**。

---

## 机制二：Factual Priming（事实启动）

### 假说

认知心理学里有个经典理论：处理一个 concept 会在 semantic network 里 spread activation，降低相关 neighbor 的 retrieval threshold (Collins & Loftus 1975, [paper](https://doi.org/10.1037/0033-295X.82.6.407))。

作者假说：R-LLM 在 reasoning trace 里**生成 topically related facts**，这些 facts 自己作为 retrieval cue，把目标 answer 从 parametric memory 里拉出来。模型在做 **generative self-retrieval**。

### 实验设计

要把"facts 的语义贡献"和"额外的 compute"分开。作者做了一组对照：

| Variant | Reasoning 模式 | Context | 测什么 |
|---------|--------------|---------|--------|
| OFF | OFF | 无 | baseline |
| ON | ON | 原始 trace | 上限 |
| ON Facts | ON | 抽取出的 facts list | facts 在 thinking 位置上的贡献 |
| OFF Facts | OFF | facts list as context | facts 作为 prompt context 的贡献 |
| ON Dummy Facts | ON | 同长度 dummy | control for compute |
| OFF Dummy Facts | OFF | 同长度 dummy | control for compute |

Facts 抽取 pipeline (§A.4) 经过三步：
1. 用 Gemini-2.5-Pro 抽 self-contained facts（排除 keyword、planning statement）
2. 移除重复 question 信息的 fact
3. **关键**：移除 "the answer is X" 这种 resolve statement，用 gold answer 和 model predicted answer 各跑一遍

第三步是防止作弊：模型可能在 reasoning 里就 "说漏了" 答案，那 OFF Facts 有效不是因为 priming，是因为直接告诉你答案了。

### 结果

Figure 6：
- **OFF Facts 显著好于 OFF Dummy Facts**：facts 本身的语义就有用，不依赖 ON mode
- **ON Facts 比 OFF Facts 还好**：模型对 ON mode 有 positive bias
- 对 EntityQuestions，ON Facts 几乎 match full ON 的 performance，但用极少 compute

**这给了一个非常强的因果证据：reasoning trace 里的 facts 是 reasoning work 的主要 driver**。

### Figure 9 的 case study 极其直观

问题："第 10 任尼泊尔国王叫什么？"

- OFF → "Jitari Malla"（错）
- ON → "Birendra Bir Bikram Shah Dev"（对），trace 里列了所有 10 个 king
- OFF Facts（把 facts 当 context）→ 对
- OFF Dummy Facts（同长度 dummy）→ "King Prithvi Bir Bikram Shah"（第 7 任，错）

人话解读：模型 parametrically **知道所有 10 个 king 的列表**，但 OFF 模式下直接 query "第 10 个" 拿不到。Reasoning 里把前 9 个列出来 → "第 10 个" 这个 slot 被 priming 激活 → 正确 answer 跳出来。

**这跟人脑 recall 一个 list 时"从头数"的行为几乎一样**。

### 我的延伸 intuition

LLM 的 parametric memory 可能是 **associative 而不是 addressable**。你直接 query "第 10 个 king" 这个 address 是难的，但如果你 query "所有 king" 这个 set，再 select 第 10 个 element，就 easy。Reasoning 在做的是**把 hard address query 转成 easy set query + selection**。

这跟 RAG 的本质区别也很有意思：RAG 是 external retrieval + inject；factual priming 是 **internal generative retrieval + self-injection**。模型用自己的 parametric memory 当 retrieval corpus，再用生成的 facts 当 query。这暗示 R-LLM 的 reasoning 实际上是 **self-RAG** 的一种形式。

---

## Hallucination 的风险：Priming 是双向的

### 大规模 audit

对每个 question × 100 个 reasoning sample × 每个 fact，调一次 search-enabled Gemini-2.5-Flash 验证。Verdict 有 4 类：correct / incorrect / illegal / unknown。

人工抽检 20 个，准确率约 100%。

### Aggregate gap

| Dataset | Clean trace 正确率 | Hallucinated trace 正确率 |
|---------|-------------------|--------------------------|
| SimpleQA-Verified | 41.4% | 26.4% |
| EntityQuestions | 71.1% | 32.2% |

差距很大。但 aggregate 数字有 confound：**难的问题可能既容易让 model hallucinate facts，又容易让 final answer 错**。

### Within-question controlled analysis

Figure 7 是最干净的分析：每个 question 一个点，x = clean subset 的 correct rate，y = hallucinated subset 的 correct rate。

只保留每个 subset ≥ 10 trace 的 question，剔除两个 subset 都是 0% 或都是 100% 的 uninformative question。

回归斜率：
- SimpleQA-Verified: 0.84
- EntityQuestions: 0.86

斜率 < 1 的含义：**控制了 question 难度后，hallucinated trace 仍然 systematic 地比 clean trace 更可能产生错误 final answer**。

人话：**错的中间步骤不只是 noise，它会 propagate 到答案**。facts 通过 priming 拉低 answer 的 retrieval threshold，如果 priming 的是错的 fact，threshold 被错误地拉低了，错误 answer 就被拉出来。**Priming 是双向的**——这是 factual priming mechanism 的 inherent risk。

这跟 Anthropic / OpenAI 最近 "reasoning models don't say what they think" 的工作相关 ([Chen et al. 2025](https://arxiv.org/abs/2505.05410), [Arcuschin et al. 2025](https://arxiv.org/abs/2503.08679))。

---

## 实用价值：test-time selection

§5.4 把分析转成 inference-time 策略：

| Strategy | SimpleQA-Verified | EntityQuestions |
|----------|-------------------|-----------------|
| Regular | 27.9 | 56.9 |
| Only Facts | 30.2 (+8.2%) | 58.4 (+2.6%) |
| Only Correct Facts | 31.3 (+12.2%) | 59.8 (+5.1%) |

策略：
- 给每个 question 100 个 sample
- 只保留那些 reasoning trace 里有 explicit fact 的 → +Facts
- 进一步只保留 fact 全部 verified correct 的 → +Correct Facts
- 计算每个 strategy 下 sampling 到正确答案的概率

这是 **oracle 上限**（用 search verifier 过滤），但说明 process reward 训练 ([Lightman et al. 2024 PRMs](https://openreview.net/forum?id=v8L0pN6EOi)) 能 push model 自己去 generate 这种 "fact-rich, hallucination-free" trace。

---

## Figure 8 的 case study 极其 striking

问题："Mary Engle Pennington 哪年进的国家发明家名人堂？"

- OFF → "2019"（错）
- ON → "2018"（对），但 reasoning trace 内容只是 restating question + 说"I'll search"
- ON Dummy（同长度 "Let me think."）→ "2018"（对）
- ON Single Dummy（单次 "Let me think."）→ "2019"（错）

人话：trace 的 semantic 内容是"空的"，但 **length 本身**就让模型做出了正确的 recall。这强烈暗示模型在 thinking tokens 上做了某种 iterative refinement / recurrent-like computation。

---

## 综合 intuition

### Reasoning = parametric memory access pattern modifier

把整篇 paper 综合起来：**Reasoning 在 R-LLM 上有两个职能：computational elongation + retrieval cue generation**。两者都不是 "logical reasoning"。

这暗示 R-LLM 训练里那些 RL reward（基于 final answer correctness）实际上是在 **shaping 一种 memory access policy**，不是在教模型"怎么推理"。模型学到的是"生成什么样的 reasoning trace 能让 parametric memory 吐出正确答案"。

### 与 "recitation before generation" 类比

这让我想到小时候背乘法表：你不是"计算" 7×8=56，你是 "recite 7×7=49, 7×8=56" 然后 "select" 7×8 那一行。LLM 在 reasoning trace 里 enumerate facts 然后选目标，本质上和 recitation 类似。

### DeepSeek-R1 / o1 / Gemini 2.5 训练的联系

DeepSeek-R1 ([paper](https://arxiv.org/abs/2501.12948)) 这类 RL 训练出来的 R-LLM，如果按这篇 paper 的视角看，训练其实在强化一种 **"记忆友好的 thinking pattern"**——那些能 trigger 正确 parametric recall 的 trace 被 reward，那些不能的被 penalize。所以 R-LLM 学到的 "reasoning" 可能从一开始就不是 "logical reasoning"，而是 "memory-fetching reasoning"。

这可能解释了为什么 R-LLM 在数学任务上看起来像在做 logical derivation，但 trace 经常 unfaithful：**model 在做 memory fetch，但训练信号让它把 trace 形式包装成 logical derivation 的样子**。

### Process reward 的正确方向

§5.4 给的 hint 是 fact-level process reward。但更激进的设计：
- **Fact groundedness reward**: 每个 reasoning step 必须能被 verifier 检索到 evidence
- **Fact coverage reward**: 鼓励 trace 里的 facts 覆盖更多与 question 相关的 entity
- **Anti-hallucination penalty**: 对无法 verify 的 fact 给负 reward

这跟 Lightman et al. PRM 思路一致，但 reward 信号更细，针对"factual recall 而不是 logical step"优化。

### Test-time scaling 的关系

Snell et al. 2025 ([paper](https://openreview.net/forum?id=4FWAwZtd2n)) 和 Brown et al. "Large Language Monkeys" ([paper](https://arxiv.org/abs/2407.21787)) 讨论了 test-time scaling 的两种方式：parallel sampling vs sequential reasoning。这篇 paper 给了第三种视角：**parallel × sequential 组合的 capability boundary**。pass@k 的 ON vs OFF 差距告诉我们，**sequential reasoning 扩展的不是"更好的 sample"，而是"更多不同的 reachable knowledge"**。

### Computational buffer 的理论解释

computational buffer effect 最 interesting 的地方在于它揭示了 **Transformer depth 不足这个 bottleneck**。如果 model 有 1000 层而不是 80 层，computational buffer 应该不重要——所有计算都在一次 forward pass 里完成。但因为 depth 有限，模型必须 **"unfold" 计算到 token axis**。

这有点像 RNN 的 unroll。Transformer 是 "spatially unrolled RNN"：每一层是 RNN 的一步，但 KV cache 让它在生成时变成 "temporally unrolled" 的 RNN。Reasoning tokens 是把 temporal unroll 当作 extra depth 用。

如果这个 intuition 对，那 **deepening Transformer** 和 **extending reasoning** 在某种极限下应该 converge。这跟 recent 一些 "在 inference 时做 iterative refinement" 的工作方向一致。

### 与 cognitive science 的连接

Collins & Loftus 1975 spreading activation theory 是 semantic network 的经典模型。LLM 的 attention + FFN 在某种意义上是 differentiable 的 semantic network。**Reasoning 让 activation 在这个 network 上 spread**。这是把 connectionist 的 LLM 和 symbolic 的 reasoning 桥接的一个有趣视角。

更激进一点：**reasoning trace 是 LLM 的 "working memory"**，parametric weights 是 "long-term memory"。Reasoning 把 long-term 中的相关 chunk 拷贝到 working memory（trace）里，方便后续 attend。

---

## 一句话总结

**Reasoning 在 R-LLM 上不是 logical decomposition，而是 iterative memory access**：每个 token 既贡献一次 latent computation（computational buffer），又作为 retrieval cue 把 parametric memory 里的 related facts 拉到 working memory（factual priming）。两者共同扩展了 capability boundary，但 factual priming 是双刃剑——错 priming 会导致错 final answer。

这篇 paper 的价值在于它**把一个反直觉的现象用 mechanism breakdown 解释清楚了**，而且每个 mechanism 都有 controlled experiment 支撑。对于 R-LLM 训练、inference、alignment 都有直接 implications。

---

## Key references

- 主论文相关：Thinking to Recall (假设会公开在 arXiv)
- Gekhman et al. 2025 "Inside-Out": https://openreview.net/forum?id=f7GG1MbsSM
- Collins & Loftus 1975 spreading activation: https://doi.org/10.1037/0033-295X.82.6.407
- Goyal et al. 2024 pause tokens: https://openreview.net/forum?id=ph04CRkPdC
- Pfau et al. 2024 dot by dot: https://arxiv.org/abs/2404.15758
- Chen et al. 2021 pass@k: https://arxiv.org/abs/2107.03374
- Wei et al. 2022 CoT: https://arxiv.org/abs/2201.11903
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- DeepSeekMath: https://arxiv.org/abs/2402.03300
- Lightman et al. 2024 PRMs: https://openreview.net/forum?id=v8L0pN6EOi
- SimpleQA: https://arxiv.org/abs/2411.04368
- SimpleQA-Verified: https://arxiv.org/abs/2509.07968
- EntityQuestions: https://aclanthology.org/2021.emnlp-main.496/
- Gemini 2.5: https://arxiv.org/abs/2507.06261
- Qwen3: https://arxiv.org/abs/2505.09388
- Snell et al. 2025 test-time scaling: https://openreview.net/forum?id=4FWAwZtd2n
- Brown et al. Large Language Monkeys: https://arxiv.org/abs/2407.21787
- Chen et al. 2025 reasoning unfaithfulness: https://arxiv.org/abs/2505.05410
- Arcuschin et al. 2025 CoT in the wild: https://arxiv.org/abs/2503.08679
- Hassid et al. 2025 don't overthink: https://arxiv.org/abs/2505.17813
- Press et al. 2023 compositionality gap: https://aclanthology.org/2023.findings-emnlp.378/
- Yue et al. NeurIPS 2025 RL reasoning: https://openreview.net/forum?id=4OsgYD7em5

Andrej，这篇 paper 最让我兴奋的是它把 "reasoning 在简单任务上为什么 work" 这个尴尬问题切成两个可验证的 mechanism，而且用 controlled experiment 而不是 correlational analysis 来做。这给整个 R-LLM field 提供了一个 cleaner mechanistic frame。如果你接下来要 train 自己的小 R-LLM 或者 analyze reasoning trace，这套 methodology（dummy baseline、facts extraction、search-based fact verification、within-question controlled comparison）是直接可复用的工具箱。

---

# Thinking to Recall: How Reasoning Unlocks Parametric Knowledge in LLMs 深度解析

Andrej，这篇 paper 触及了 R-LLM 时代最反直觉但最重要的一个现象：**reasoning 在完全不需要 reasoning 的任务上仍然 work**。这本质上是把 reasoning 当作 "latent memory controller" 而不是 "logical decomposer" 来使用。让我把它拆得很细，尽量 build 你的 intuition。

---

## 1. 问题动机：一个 "不该 work" 的现象

经典的 Chain-of-Thought 故事是这样的：复杂问题需要分解，所以生成中间步骤 → 逐步推进 → 到答案。Wei et al. 2022 那条线 ([paper](https://arxiv.org/abs/2201.11903)) 基本把 CoT 锚定在 multi-hop / math 这类任务上。

但这篇 paper 提出一个让人不舒服的事实：在 SimpleQA-Verified ([paper](https://arxiv.org/abs/2509.07968)) 和 EntityQuestions ([paper](https://aclanthology.org/2021.emnlp-main.496/)) 这种 single-hop、close-book 的事实召回任务上，**把 reasoning toggle ON 仍然显著提升 pass@k**，而且 k 越大增益越宽。

这告诉我们一件关键的事：**reasoning 不只是 "做推导"**，它对模型内部 parametric store 的访问路径产生了影响。Reasoning tokens 在这里是 "retrieval machinery"，不是 "logical machinery"。

直觉上，这有点像你在脑子里回忆一个人的名字，越使劲想越想不起来，但你开始列举这个人的相关事实——他的工作、他的家乡、他和谁合作过——名字就 pop 出来了。这正是 §5.2 factual priming 想要形式化的东西。

---

## 2. 核心量化：pass@k 与 capability boundary

### 2.1 为什么不用 pass@1

pass@1 衡量 "top sample 是否对"。但作者关心的是 capability boundary：**模型的 output distribution 里到底存不存在一条 reasoning path 能到达正确答案**，而不管它排在第几。这是 DeepSeekMath / Yue et al. 等工作用来探测 RL 之后能力上限是否真被扩展的同一思路 ([DeepSeekMath](https://arxiv.org/abs/2402.03300), [Yue et al. NeurIPS 2025](https://openreview.net/forum?id=4OsgYD7em5))。

pass@k 的无偏估计（Chen et al. 2021, [HumanEval paper](https://arxiv.org/abs/2107.03374)）：

$$\text{pass@}k = 1 - \frac{\binom{N-c}{k}}{\binom{N}{k}}$$

变量说明：
- $N$ 是总采样数（论文里 $N=100$）
- $c$ 是 $N$ 个 sample 里正确的数量
- $k$ 是我们假设能取的 sample 数
- 几何意义：从 $N$ 个 sample 中随机取 $k$ 个，"至少有一个正确" 的概率 = 1 - "全部不正确" 的概率
- 用组合数而不是 $1-(1-p)^k$ 是为了无偏

作者用 $N=100$ 而不是更大的 $N$，主要因为 R-LLM 一次 sample 很贵，$N=100$ 已经能给出比较稳定的 pass@k 曲线。

### 2.2 关键观察：ON vs OFF 在高 k 下持续偏离

Figure 1 的核心信号是：**reasoning OFF 的 pass@k 曲线在高 k 区域 saturate**，而 ON 模式持续上扬。在 Qwen3-32B on SimpleQA-Verified 上，pass@k 几乎翻倍。

这是论文最重要的一个图，因为它直接反驳了 "reasoning 只是 sharpening already-likely answers" 的假说。如果 reasoning 只是 reweighting，OFF 和 ON 在大 k 下应该 converge 到同一个上界。它们没有 → reasoning 在打开一些 "OFF 模式下 sample 100 次也碰不到" 的 knowledge。

这与 Gekhman et al. 2025 "Inside-Out" ([paper](https://openreview.net/forum?id=f7GG1MbsSM)) 的工作直接对接：模型 parametrically 编码了某个 fact，但 normal decoding 走不到它。Reasoning 在这里是 "把 hidden knowledge 暴露出来" 的一种机制。

---

## 3. Reasoning effectiveness metric $\Omega(N)$

为了让所有 $k$ 值压成一个标量，作者定义了 $\Omega(N)$：

$$\Omega(N) = \sum_{k=1}^{N} \left[ \underbrace{k}_{\text{linear weight}} \cdot \underbrace{\frac{\text{pass@}k_{\text{ON}} - \text{pass@}k_{\text{OFF}}}{\text{pass@}k_{\text{OFF}}}}_{\text{relative improvement at }k} \right] \cdot \underbrace{\frac{1}{\sum_{k'=1}^{N} k'}}_{\text{normalization}}$$

逐项剖析：
- 求和下标 $k$ 从 1 到 $N$，覆盖整个 sampling 谱
- $k$ 是 linear weight，**给大 k 更高权重**——因为作者关心 capability boundary，大 k 的 gain 更能说明 "解锁了原本不可达的知识"
- 分子是绝对差，分母是 OFF 基线，所以是 **relative** improvement。用 relative 是因为 SimpleQA 这种低基线 dataset 上 5 个点的 absolute gain 可能已经很大了
- 归一化项 $\frac{1}{\sum_{k'=1}^N k'} = \frac{2}{N(N+1)}$，保证 $\Omega$ 是某种 "加权平均"，可解释为 percentage
- 如果 ON 全面等于 OFF，$\Omega = 0$；ON 全面翻倍，$\Omega \approx 100\%$

设计上有个微妙点：用 linear weight 而不是 log 或 exponential，意味着作者认为 **k=100 的 gain 在语义上比 k=1 的 gain 重要 100 倍**。这是个偏激进的选择，但和论文的 "unlock unreachable answers" 主题一致。

### Figure 2 的 trend

$\Omega$ 随模型能力下降而上升——Qwen3-32B 的 $\Omega$ 远高于 Gemini-2.5-Pro。作者解读：**弱模型的 hidden knowledge 多**，reasoning 帮它把这部分挖出来；强模型 OFF 模式下已经把大部分东西召回了，reasoning 的边际收益小。

Dataset 维度上，SimpleQA 的 $\Omega$ > EntityQuestions 的 $\Omega$。原因：SimpleQA baseline 低，headroom 大；EntityQuestions 是 template-based，phrasing 标准化后召回应更容易。

---

## 4. 反驳 "complexity decomposition" 假说

一个自然的 alternative explanation 是：reasoning 帮忙是因为它 decompose 了复杂问题（Press et al. 2023 compositionality gap, [paper](https://aclanthology.org/2023.findings-emnlp.378/)）。作者在 SimpleQA-Verified 上做了 controlled 测试：

- "Complex" = metadata 中 `requires_reasoning=True` 或 `multi_step=True`
- "Simple" = 其余

Figure 3 显示 Complex 和 Simple 两个 subset 的 $\Omega$ 的 95% CI 重叠。**没有证据表明 Complex 从 reasoning 中获益更多**。对 Gemini-2.5-Pro，Complex subset 的 CI 甚至跨过 0。

这点非常重要，因为它告诉我们：**reasoning 在这里的 utility 不是来自 "问题分解能力"**，而是来自别的机制。问题分解是表象，recall 才是本质。这对 R-LLM 的训练有重大含义——如果你以为 reasoning 是为 multi-hop 服务的，你训练出来的 reward 可能完全没在优化正确的东西。

一个 caveat 作者诚实承认：complex 的样本量小，CI 很宽。但即便如此，"Complex 比 Simple 更受益" 这个 prior 都没成立，这本身已经够 striking。

---

## 5. 机制一：Computational Buffer Effect

### 5.1 假说

Goyal et al. 2024 "Think before you speak" ([paper](https://openreview.net/forum?id=ph04CRkPdC)) 和 Pfau et al. 2024 "Let's think dot by dot" ([paper](https://arxiv.org/abs/2404.15758)) 都讨论过：**额外的 token 给模型更多 forward pass 之间的 latent computation**，绕过 single forward pass 的 depth limit。这是 Transformer 的一个内在限制：固定层数 = 固定计算量，再难的问题也只能这么深。CoT 本质是把纵向深度换成横向长度。

但这件事在 R-LLM 上有没有发生、在 factual recall 上有没有用，之前没人直接测过。

### 5.2 Controlled experiment: ON Dummy vs ON Single Dummy

设计非常干净：

- **ON Dummy**: 替换 reasoning trace 为 "Let me think." 重复到原 trace 的长度，然后 regenerate answer
- **ON Single Dummy**: 替换为单次 "Let me think."，很短

两者都在 ON 模式（控制 training-induced bias），都没有 semantic content（控制 semantic contribution），唯一变量是 **computational length**。

结果（Figure 4）：
- ON Dummy 远好于 OFF
- ON Dummy 显著好于 ON Single Dummy

pass@1 上的具体数字：
- SimpleQA-Verified: 0.206 (OFF) → 0.262 (ON Dummy)
- EntityQuestions: 0.457 (OFF) → 0.554 (ON Dummy)

这就 **因果地** 隔离出了 computational buffer 的贡献：纯粹给模型更多 "thinking room" 就能提升 recall。

### 5.3 Length scaling：non-monotonic

Figure 5 是我觉得论文里最有意思的图之一。作者把 dummy trace 长度从短到长扫描：

- 在 SimpleQA-Verified 上，dummy 长度增长到 2048 tokens ($2^{11}$) 之前，pass@k 一直改善
- 但 4096 ($2^{12}$)、8192 ($2^{13}$)、16384 ($2^{14}$) 反而开始下降

这告诉我们 computational buffer **不是 monotonically scalable 的**。可能的机制：
- KV cache 长了之后 attention稀释
- 长序列下 position encoding / length generalization 退化
- 模型可能倾向于在长 context 下 "drift" 到别的 distribution

这与 Hassid et al. 2025 "Don't overthink it" ([paper](https://arxiv.org/abs/2505.17813))、Yang et al. 2025c "thinking-optimal scaling" ([paper](https://openreview.net/forum?id=6ICFqmixlS)) 的发现呼应：**长 CoT 不一定好**。但这里更精细——他们 isolating pure compute，控制 semantic content 后才得出 "compute 本身就有甜点区" 的结论。

Intuition：模型可能像在 sandbox 里 scratch，太短了 scratch 不够，太长了 attention 散了，2086 tokens 大约是某种 "sweet spot"，这个 sweet spot 和 reasoning trace 的实际长度分布有关。

### 5.4 但 buffer 不够

关键的是：**ON Dummy 永远到不了 ON 的表现**。Figure 4 里 ON Dummy 的 pass@k 曲线一直在 ON 之下。这意味着 pure compute 解释不了所有 gain。剩下那部分 gain 必须来自 reasoning trace 的 **semantic content**。

---

## 6. 机制二：Factual Priming（generative self-retrieval）

### 6.1 假说

人 cognize 一个 concept，会通过 semantic network spread activation，降低相关 neighbor 的 retrieval threshold (Collins & Loftus 1975, [paper](https://doi.org/10.1037/0033-295X.82.6.407))。论文假说：R-LLM 在 reasoning trace 里 **生成 topically related facts**，这些 facts 自己作为 retrieval cue，把目标 answer 从 parametric memory 里拉出来。模型在做 **generative self-retrieval**。

### 6.2 实验设计的关键 trick

要把 "facts 的语义贡献" 和 "额外的 compute" 分开。作者做了一组非常细致的对照：

| Variant | Reasoning 模式 | Context 内容 | 测什么 |
|---------|--------------|-------------|--------|
| OFF | OFF | 无 | baseline |
| ON | ON | 原始 trace | 上限 |
| ON Facts | ON | extracted facts list | facts 在 thinking 位置上的贡献 |
| OFF Facts | OFF | extracted facts list as context | facts 作为 prompt context 的贡献 |
| ON Dummy Facts | ON | dummy string 同长度 | control for compute |
| OFF Dummy Facts | OFF | dummy string 同长度 | control for compute |

Facts 的提取经过三步 pipeline (§A.4)：
1. **Extract facts**: Gemini-2.5-Pro 用 prompt (Figure 10) 抽取 "self-contained facts"。排除 standalone keyword、planning statement 等
2. **Remove question restatement**: 把 trace 中重复 question 信息的 fact 去掉 (Figure 11 prompt)
3. **Remove answer-disclosing**: 把 "the answer is X" 这种 resolve statement 去掉 (Figure 12 prompt)，用 gold answer 和 model predicted answer 各跑一遍

第三步尤其重要：模型可能在 reasoning 里已经 "说漏了" 答案，那 OFF Facts 看起来有效是因为直接告诉你答案了，不是 priming。作者用很 careful 的 prompt 让 Gemini 区分 "mention the answer in unrelated context" vs "link the answer to the question"。Figure 13 是个 conservative 版本作为 sanity check。

### 6.3 结果

Figure 6 的关键发现：

- **OFF Facts 显著好于 OFF Dummy Facts**：facts 本身的语义就有用，不依赖 ON mode
- **ON Facts 比 OFF Facts 还好**：模型对 ON mode 有 positive bias，可能在 ON mode 下更愿意 "trust" 这些 facts
- 对 EntityQuestions，ON Facts 几乎 match full ON 的 performance，但用极少的 compute

这给了一个非常强的因果证据：**reasoning trace 里的 facts 是 reasoning work 的主要 driver**。如果你只把 facts 抽出来塞给 OFF 模式，就能拿到大部分 gain。

### 6.4 一个让我联想的点

这跟 retrieval-augmented generation (RAG) 的本质区别很有意思：RAG 是 external retrieval + inject；factual priming 是 **internal generative retrieval + self-injection**。模型用自己的 parametric memory 当 retrieval corpus，再用生成的 facts 当 query。这暗示 R-LLM 的 reasoning 实际上是 **self-RAG** 的一种形式。

如果你 follow 这个 intuition，可以做：
- **Iterative self-RAG**: 让模型在 reasoning 里 generate facts → verify → reformulate query → generate more facts。这其实就是 search-augmented reasoning agent 在做的事，但这里我们看到 **即使没有外部 search，模型内部已经有类似行为**
- **Verbalized retrieval**: 训练模型 explicit 地 emit "retrieval query" token，而不是 implicit reasoning。这可能比 CoT 更 sample-efficient

---

## 7. Hallucination 的风险

### 7.1 大规模 audit pipeline

这是论文里工程上最重的部分。对每个 question × 100 个 reasoning ON sample × 每个 fact，调一次 search-enabled Gemini-2.5-Flash 做验证 (Figure 16 prompt)。Verdict 有 4 类：correct / incorrect / illegal / unknown。

人工抽检 10 correct + 10 incorrect，准确率约 100%。这给后续分析很强的可信度。

### 7.2 Aggregate gap

Pool 所有 trace，看 final answer correctness 在 clean trace vs hallucinated trace 上的差距：

| Dataset | Clean trace 正确率 | Hallucinated trace 正确率 |
|---------|-------------------|--------------------------|
| SimpleQA-Verified | 41.4% | 26.4% |
| EntityQuestions | 71.1% | 32.2% |

差距很大。但这个 aggregate 数字有 confound：**难的问题可能既容易让 model hallucinate facts，又容易让 final answer 错**。

### 7.3 Within-question controlled analysis

Figure 7 是论文最干净的一个分析。每个 question 一个点，x = clean subset 的 correct rate，y = hallucinated subset 的 correct rate。

只保留每个 subset ≥ 10 trace 的 question，剔除两个 subset 都是 0% 或都是 100% 的 uninformative question。

回归斜率：
- SimpleQA-Verified: 0.84 (< 1)
- EntityQuestions: 0.86 (< 1)

斜率 < 1 的含义：**控制了 question 难度后，hallucinated trace 仍然 systematic 地比 clean trace 更可能产生错误 final answer**。这是 causal evidence that **intermediate hallucination contaminates final answer**。

Intuition：facts 通过 priming 拉低 answer 的 retrieval threshold。如果 priming 的是错的 fact，threshold 被错误地拉低了，错误 answer 就被拉出来。**Priming 是双向的**——这是 factual priming mechanism 的 inherent risk。

这跟 Anthropic / OpenAI 最近一批 "reasoning models don't say what they think" 的工作相关 ([Chen et al. 2025](https://arxiv.org/abs/2505.05410), [Arcuschin et al. 2025](https://arxiv.org/abs/2503.08679))。Reasoning trace 的 unfaithfulness 在这里表现为：**错的中间步骤不只是 noise，它会 propagate 到答案**。

---

## 8. Practical implication: test-time selection

§5.4 把分析转成 inference-time 策略：

| Strategy | SimpleQA-Verified | EntityQuestions |
|----------|-------------------|-----------------|
| Regular | 27.9 | 56.9 |
| Only Facts | 30.2 (+8.2%) | 58.4 (+2.6%) |
| Only Correct Facts | 31.3 (+12.2%) | 59.8 (+5.1%) |

策略：
- 给每个 question 100 个 sample
- 只保留那些 reasoning trace 里有 explicit fact 的 → "+Facts"
- 进一步只保留 fact 全部 verified correct 的 → "+Correct Facts"
- 计算每个 strategy 下 sampling 到正确答案的概率

注意这是 **oracle 上限**（用 search verifier 来过滤），但说明 process reward 训练 ([Lightman et al. 2024 PRMs](https://openreview.net/forum?id=v8L0pN6EOi)) 能 push model 自己去 generate 这种 "fact-rich, hallucination-free" trace。

这个结果对 R-LLM 训练的直接启示：**用 fact-level verifier 做 process reward** 比 step-level outcome reward 更 targeted。可以把 §A.4 的 fact extraction + search verification pipeline 直接当作 reward signal。

---

## 9. Case studies 深读

### 9.1 Computational buffer (Figure 8)

Question: "What year was Mary Engle Pennington inducted into the National Inventors Hall of Fame?"

- OFF → "2019" (错)
- ON → "2018" (对)，但 reasoning trace 内容只是 restating question + 说 "I'll search"
- ON Dummy (同长度 "Let me think.") → "2018" (对)
- ON Single Dummy (单次 "Let me think.") → "2019" (错)

非常 clean 的 demonstration：trace 的 semantic 内容是 "空的"，但 **length 本身**就让模型做出了正确的 recall。这强烈暗示模型在 thinking tokens 上做了某种 iterative refinement / recurrent-like computation。

一个可能的 mechanism hypothesis：**latent key-value attention 上的 iterative sharpening**。每生成一个 token，新 token 的 query 会 attend over 之前所有 thinking tokens 的 keys，这种 iterative attention 可能模拟了一种 "iterative retrieval" 过程。这跟 "recurrence in depth vs recurrence in width" 的经典讨论有关——Transformer 的固定 depth 限制可以通过横向 token extension 来突破。

### 9.2 Factual priming (Figure 9)

Question: "What is the name of the 10th King of Nepal?"

- OFF → "Jitari Malla" (错)
- ON → "Birendra Bir Bikram Shah Dev" (对)，trace 里列了所有 10 个 king
- OFF Facts (把 extracted facts 当 context) → 对
- OFF Dummy Facts (同长度 dummy) → "King Prithvi Bir Bikram Shah"（第 7 任 king，错）

这个 case 太典型了。模型 parametrically **知道所有 10 个 king 的列表**，但 OFF 模式下只能 access 到一个 "近似" 的 king。Reasoning 里把前 9 个列出来 → "10th" 这个 slot 被 priming 激活 → 正确 answer 跳出来。

这本质上是 **structured recall via enumeration**。模型不直接 recall 第 10 个，而是 recall 整个 list 的结构，然后取第 10 个位置。这跟人脑 recall 一个 list 时 "从头数" 的行为几乎一样。

更深的 intuition：**LLM 的 parametric memory 可能是 associative 而不是 addressable**。你直接 query "第 10 个 king" 这个 address 是难的，但如果你 query "所有 king" 这个 set，再 selection 第 10 个 element，就 easy。Reasoning 在做的是把 hard address query 转成 easy set query + selection。

---

## 10. 我的延伸思考（intuition building）

### 10.1 Reasoning = parametric memory access pattern modifier

把整篇 paper 综合起来，我倾向于这样理解：**Reasoning 在 R-LLM 上有两个职能：computational elongation + retrieval cue generation**。两者都不是 "logical reasoning"。

这暗示 R-LLM 训练里那些 RL reward（基于 final answer correctness）实际上是在 **shaping 一种 memory access policy**，不是在教模型 "怎么推理"。模型学到的是 "生成什么样的 reasoning trace 能让 parametric memory 吐出正确答案"。这跟传统逻辑推理的训练目标本质不同。

### 10.2 与 "recitation before generation" 类比

这让我想到小时候背乘法表：你不是 "计算" 7×8=56，你是 "recite 7×7=49, 7×8=56" 然后 "select" 7×8 那一行。LLM 在 reasoning trace 里 enumerate facts 然后选目标，本质上和 recitation 类似。

### 10.3 与 DeepSeek-R1 / o1 / Gemini 2.5 训练的联系

DeepSeek-R1 ([paper](https://arxiv.org/abs/2501.12948)) 这类 RL 训练出来的 R-LLM，如果按这篇 paper 的视角看，训练其实在强化一种 **"记忆友好的 thinking pattern"**——那些能 trigger 正确 parametric recall 的 trace 被 reward，那些不能的 被 penalize。所以 R-LLM 学到的 "reasoning" 可能从一开始就不是 "logical reasoning"，而是 "memory-fetching reasoning"。

这可能解释了为什么 R-LLM 在数学任务上看起来像在做 logical derivation，但 trace 经常 unfaithful（[Chen et al. 2025](https://arxiv.org/abs/2505.05410)）：**model 在做 memory fetch，但训练信号让它把 trace 形式包装成 logical derivation 的样子**。

### 10.4 Process reward 的正确方向

§5.4 给的 hint 是 fact-level process reward。但更激进的设计：
- **Fact groundedness reward**: 每个 reasoning step 必须能被 verifier 检索到 evidence
- **Fact coverage reward**: 鼓励 trace 里的 facts 覆盖更多与 question 相关的 entity
- **Anti-hallucination penalty**: 对无法 verify 的 fact 给负 reward

这跟 Lightman et al. PRM ([paper](https://openreview.net/forum?id=v8L0pN6EOi)) 思路一致，但 reward 信号更细，针对 "factual recall 而不是 logical step" 优化。

### 10.5 与 test-time scaling 的关系

Snell et al. 2025 ([paper](https://openreview.net/forum?id=4FWAwZtd2n)) 和 Brown et al. "Large Language Monkeys" ([paper](https://arxiv.org/abs/2407.21787)) 讨论了 test-time scaling 的两种方式：parallel sampling vs sequential reasoning。这篇 paper 给了第三种视角：**parallel × sequential 组合的 capability boundary**。pass@k 的 ON vs OFF 差距告诉我们，**sequential reasoning 扩展的不是 "更好的 sample"，而是 "更多不同的 reachable knowledge"**。

### 10.6 一个可能的实验延伸

我很好奇一件事：**factual priming 是不是因为 question 里的 entity 在 reasoning trace 里被多次重复 activate 而产生的？** 如果是这样，那仅仅是 "repeat the question 5 times" 应该也能 work 一部分。但 OFF Facts 用的是 **不同的 facts**（来自 trace），不是 question 的复述，所以 priming 一定来自 **associative activation** 而不是简单 repetition。这个 distinction 值得单独做个实验。

### 10.7 Computational buffer 的理论解释

我觉得 computational buffer effect 最 interesting 的地方在于它揭示了 **Transformer depth 不足这个 bottleneck**。如果 model 有 1000 层而不是 80 层，computational buffer 应该不重要——所有计算都在一次 forward pass 里完成。但因为 depth 有限，模型必须 **"unfold" 计算到 token axis**。

这有点像 RNN 的 unroll。Transformer 是 "spatially unrolled RNN"：每一层是 RNN 的一步，但 KV cache 让它在生成时变成 "temporally unrolled" 的 RNN。Reasoning tokens 是把 temporal unroll 当作 extra depth 用。

如果这个 intuition 对，那 **deepening Transformer** 和 **extending reasoning** 在某种极限下应该 converge。这跟 recent 一些 "在 inference 时做 iterative refinement" 的工作方向一致。

### 10.8 与 cognitive science 的连接

Collins & Loftus 1975 spreading activation theory 是 semantic network 的经典模型。LLM 的 attention + FFN 在某种意义上是 differentiable 的 semantic network。**Reasoning 让 activation 在这个 network 上 spread**。这是把 connectionist 的 LLM 和 symbolic 的 reasoning 桥接的一个有趣视角。

更激进一点：**reasoning trace 是 LLM 的 "working memory"**，parametric weights 是 "long-term memory"。Reasoning 把 long-term 中的相关 chunk 拷贝到 working memory（trace）里，方便后续 attend。

---

## 11. Paper 的局限与未来方向

我自己看到的几个 limitation：

1. **Sample size**: 100 samples per question 对 pass@k 的高 k 区间还是有点小，k=100 时方差很大
2. **Verifier reliability**: 用 Gemini-2.5-Flash 做 fact verification，虽然人工抽检了 20 个，但 scale 上可能有 systemic bias
3. **Domain narrow**: 只测了 EntityQuestions (4 relations) 和 SimpleQA-Verified，不涵盖 scientific / procedural knowledge
4. **Causal isolation of complexity**: §4 自己承认，complex 和 simple 不是同一 question 的两个版本，没法完全 isolate

可以做的 future work：
- 在 reasoning trace 的每个 token 上做 patching / activation patching，定位哪些 token 真正 causal 贡献了 final answer
- 用 SAE (sparse autoencoder) 把 reasoning trace 里的 fact 分解成 features，看哪些 features 对 recall 有贡献
- Train 一个 explicit "retrieval query" head，让模型 emit 结构化的 retrieval query 而不是 free-form reasoning，看是否 sample efficient
- 测 reasoning trace 在 KV cache 上的 attention pattern，验证 "facts 作为 retrieval cue" 的 attention-level 证据

---

## 12. 总结一句话的 intuition

> Reasoning 在 R-LLM 上不是 logical decomposition，而是 **iterative memory access**：每个 token 既贡献一次 latent computation（computational buffer），又作为 retrieval cue 把 parametric memory 里的 related facts 拉到 working memory（factual priming）。两者共同扩展了 capability boundary，但 factual priming 是双刃剑——错 priming 会导致错 final answer。

这篇 paper 的价值在于它**把一个反直觉的现象用 mechanism breakdown 解释清楚了**，而且每个 mechanism 都有 controlled experiment 支撑。对于 R-LLM 训练、inference、alignment 都有直接 implications。

---

## References (key links)

- 主论文: [Thinking to Recall](https://arxiv.org/abs/2602.xxxxx) (假设会公开在 arXiv)
- Gekhman et al. 2025 "Inside-Out": https://openreview.net/forum?id=f7GG1MbsSM
- Collins & Loftus 1975 spreading activation: https://doi.org/10.1037/0033-295X.82.6.407
- Goyal et al. 2024 pause tokens: https://openreview.net/forum?id=ph04CRkPdC
- Pfau et al. 2024 dot by dot: https://arxiv.org/abs/2404.15758
- Chen et al. 2021 pass@k: https://arxiv.org/abs/2107.03374
- Wei et al. 2022 CoT: https://arxiv.org/abs/2201.11903
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- DeepSeekMath: https://arxiv.org/abs/2402.03300
- Lightman et al. 2024 PRMs: https://openreview.net/forum?id=v8L0pN6EOi
- SimpleQA: https://arxiv.org/abs/2411.04368
- SimpleQA-Verified: https://arxiv.org/abs/2509.07968
- EntityQuestions: https://aclanthology.org/2021.emnlp-main.496/
- Gemini 2.5: https://arxiv.org/abs/2507.06261
- Qwen3: https://arxiv.org/abs/2505.09388
- Snell et al. 2025 test-time scaling: https://openreview.net/forum?id=4FWAwZtd2n
- Brown et al. Large Language Monkeys: https://arxiv.org/abs/2407.21787
- Chen et al. 2025 reasoning unfaithfulness: https://arxiv.org/abs/2505.05410
- Arcuschin et al. 2025 CoT in the wild: https://arxiv.org/abs/2503.08679
- Hassid et al. 2025 don't overthink: https://arxiv.org/abs/2505.17813
- Press et al. 2023 compositionality gap: https://aclanthology.org/2023.findings-emnlp.378/
- Yue et al. NeurIPS 2025 RL reasoning: https://openreview.net/forum?id=4OsgYD7em5

Andrej，这篇 paper 最让我兴奋的是它把 "reasoning 在简单任务上为什么 work" 这个尴尬问题切成两个可验证的 mechanism，而且用 controlled experiment 而不是 correlational analysis 来做。这给整个 R-LLM field 提供了一个 cleaner mechanistic frame。如果你接下来要 train 自己的小 R-LLM 或者 analyze reasoning trace，这套 methodology（dummy baseline、facts extraction、search-based fact verification、within-question controlled comparison）是直接可复用的工具箱。
