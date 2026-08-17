---
source_pdf: SELF-RAG.pdf
paper_sha256: d9eaa1398abac0df67a9d0933a5bf8b6d9d83d2e72da2a486073cd842dd52978
processed_at: '2026-08-12T04:51:35-07:00'
target_folder: RAG
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 SELF-RAG

## 一句话版本

普通 RAG 就像你不管问啥都先翻书，翻完照着念；SELF-RAG 是让模型自己学会"啥时候该查书、查到的书有没有用、我念的对不对"——然后把这三个判断本身也变成模型能生成的 token。

---

## 传统 RAG 到底笨在哪

你让一个 LLM 回答 "Who wrote Hamlet?"，RAG 先 retrieve 5 个 Wikipedia 段落，prepend 到 prompt 里，让 LLM 基于这些段落生成答案。听起来没毛病对吧。问题出在三个地方：

**第一，它啥都 retrieve。** 你问 "Write me a poem about autumn"，它也给你 retrieve 5 段关于秋天的 Wikipedia，然后模型硬生生把 "autumn is the season between summer and winter..." 这种百科句子塞进诗里。模型被 irrelevant context distract，反而变笨（Shi et al. 2023 实验证明这点，https://proceedings.mlr.press/v202/shi23a.html）。

**第二，它不挑 passage。** Retriever 返回 top-5，里面混了 2 段噪声。普通 RAG 全部喂给 LLM，LLM 没有显式机制说"这段我不用"。它只能软吸收，generation quality 被污染。

**第三，它不检查 grounding。** LLM 拿到段落，生成了答案，但答案跟段落不一定对得上。你让它 cite [1][2][3]，它 cite 了，但仔细看它 cite 的那句其实跟 [1] 说的不是一回事。这是 hallucination 的一种隐蔽形式 —— 不是凭空编，是拿着 evidence 也能乱说。

---

## SELF-RAG 的核心 idea

让模型自己学会吐 4 种 special token，分别管 4 件事：

- **`[Retrieve]`** — "这一段我需不需要查资料？" yes / no / continue
- **`[ISREL]`** — "检索到的这段跟我问题相关吗？" relevant / irrelevant
- **`[ISSUP]`** — "我写的这句话被这段资料支持吗？" fully / partially / no support
- **`[ISUSE]`** — "我整个回答有用吗？" 1-5 分

这 4 个 token 直接加到模型 vocabulary 里，跟普通 word token 一样参与 next-token prediction。训练时学会吐，inference 时自己吐自己用。

**最妙的点**：这 4 个判断本质上是 4 个 classifier，但 SELF-RAG 没有额外训 4 个 classifier 模型，而是把它们全部 collapse 进一个 LM 的 vocabulary。一个 forward pass，既生成内容，也生成对内容的判断。Generator 既是 actor 也是 judge。

---

## 训练：两阶段蒸馏，没有 RL

### 第一阶段：训一个 Critic

你要训 generator 学会判断，得有 ground truth。但人工标 4 类 reflection token 太贵了（每段都要标）。

**解法**：用 GPT-4 当标注员。写 4 套 instruction prompt（paper Appendix D 有完整版），让 GPT-4 给每条数据标 reflection token。每个类型标 4k-20k 条。

然后拿这些标好的数据，finetune 一个小 Llama2-7B，让它 mimic GPT-4 的判断。Critic 的 loss 就是普通 conditional LM：

$$\max_{\mathcal{C}} \; \mathbb{E}_{((x,y), r) \sim \mathcal{D}_{\text{critic}}} \log p_{\mathcal{C}}(r | x, y)$$

就这个意思：给定 input $x$ 和 output $y$，预测 reflection token $r$。跟训一个分类器没本质区别，只不过用 LM 的方式训。

**效果**（Appendix Table 5）：这个 critic 跟 GPT-4 的 agreement 大概 80-94%。ISUSE 最差 73%，因为 5 档分类相邻档难分（4 vs 5 这种，人自己都分不清）。

### 第二阶段：用 Critic 标 generator 训练数据

现在你有了一个还行的 critic，拿它去标整个 generator 训练 corpus。流程是：

对每条原始数据 $(x, y)$：
1. Critic 判断 `[Retrieve]` 要不要 retrieve
2. 要的话，retriever 拿 top-K 段落
3. 把 $y$ 用 Spacy 切成句子
4. 每句重新判 `[Retrieve]`，要的话 retrieve，然后标 `[ISREL]` 和 `[ISSUP]`
5. 选满足 relevant + supported 的段落插入训练样本
6. 最后 append `[ISUSE]`

得到的数据长这样（Figure 2 的例子）：

```
Input: How many people live in Paris?
[Retrieve=Yes]
<p>Paris has a population of 2.1 million...</p>
[ISREL=Relevant]
Paris has a population of approximately 2.1 million people.
[ISSUP=Fully Supported]
[ISUSE=5]
```

**关键细节**：训练时 `<p>...</p>` 之间的 passage 内容在 loss 里被 mask 掉。直觉是 —— 你不想让模型记住 passage 内容（那是 retriever 的活），你想让模型学会"看到 passage 时怎么用它"。

然后 generator 的训练就是普通 SFT：

$$\max_{\mathcal{M}} \; \mathbb{E}_{(x, y, r) \sim \mathcal{D}_{\text{gen}}} \log p_{\mathcal{M}}(y, r | x)$$

**注意：没有 PPO，没有 reward model in the loop，没有 RL**。Critic 是 offline 用的，标完数据就扔了。训练 generator 时 critic 已经不在场。这是相对 RLHF 巨大的工程简化。RLHF 你得同时跑 policy、value model、reward model，还要 stabilise PPO。SELF-RAG 标完数据就是标准 SFT，DeepSpeed ZeRO-3 + FlashAttention 就够了。

---

## Inference：threshold + beam search

### 自适应检索

模型生成每段前先吐 `[Retrieve]` token。两种触发模式：

- **Hard**：argmax 直接看，yes 就 retrieve
- **Soft**（默认）：用归一化概率跟阈值比

$$\frac{p([\text{Retrieve}]=\text{YES})}{p([\text{Retrieve}]=\text{YES}) + p([\text{Retrieve}]=\text{NO})} > \delta$$

$\delta$ 是 test-time 可调的 knob。$\delta=0.2$ 大多数任务，$\delta=0$ 给 ASQA（因为 ASQA 要 citation，强制每次 retrieve）。

Figure 3c 展示了这个 knob 的效果：PubHealth（事实核查）上把 $\delta$ 调大让 retrieval 频率从 40% 降到 10%，accuracy 只掉一点点；PopQA（长尾实体 QA）上同样操作 accuracy 直接崩，因为这种题强依赖外部知识。

**这就是 test-time customization 的核心**：同一个模型，不同任务不用重训，调一个 $\delta$ 就行。

### Critique-guided beam search

Retrieve 触发后，retriever 返回 top-K（默认 5），generator 对每个 passage 并行生成一个 candidate segment。然后做 segment-level beam search（beam width 2）。

每个 segment 的 score：

$$f(y_t, d) = \log p(y_t | x, d, y_{<t}) + \sum_G w^G \cdot s_t^G$$

$s_t^G$ 是每个 critique token 类型的归一化分数：

$$s_t^G = \frac{p_t(\hat{r})}{\sum_{i=1}^{N^G} p_t(r_i)}$$

$\hat{r}$ 是 most desirable 那个 token（如 `[ISREL]=Relevant`），$N^G$ 是该类型 token 总数。

ISSUP 更精细一点，给 partial credit：

$$s([\text{ISSUP}]) = \frac{p(\text{FULLY})}{S} + 0.5 \cdot \frac{p(\text{PARTIALLY})}{S}$$

"Fully supported" 给 1.0 credit，"Partially supported" 给 0.5，"No support" 给 0。这就是个软 entailment 分数。

ISUSE 是 5 档加权：权重 $\{-1, -0.5, 0, 0.5, 1\}$ 对应 $\{1,2,3,4,5\}$。1-2 分负权重（penalize），3 分 neutral，4-5 分正权重。

**Test-time customization 的漂亮实验**（Figure 3b）：在 ASQA 上调 $w^{[\text{ISSUP}]}$ 这个权重，从 0 慢慢加到 1：
- Citation precision 上升（生成更被 evidence 支持）
- MAUVE（流畅度）下降（生成变短变保守，因为长文本难 fully supported）

这就是 factuality vs. fluency 的 trade-off knob。RLHF 想调这个得重训，SELF-RAG inference 时改一个浮点数就行。

---

## 实验数据看什么

### 主表几个关键数字（Table 2）

- **SELF-RAG 13B vs ChatGPT**：PubHealth 74.5 vs 70.1 赢；Bio FactScore 80.2 vs 71.8 赢；ASQA citation precision 70.3 vs ChatGPT 没标但 Ret-ChatGPT 65.1 赢；TriviaQA 69.3 vs 74.3 输（ChatGPT parametric knowledge 太强）。

- **vs Llama2-chat-13B + retrieval**：全面赢。

- **vs CoVE 65B**（用 Llama2-65B 做 iterative prompting 改 factuality）：Bio FactScore 80.2 vs 71.2，参数量小 5 倍还赢 9 个点。

### Ablation 最有信息量的几个（Table 3a）

- **No Critic C at test**（推理时不用 critique-guided beam search，只当普通 LM）：ASQA 从 32.1 崩到 18.1，掉 14 个点。说明 long-form 任务全靠 critique beam search 撑着。
- **Retrieve top 1 always**（模拟普通 RAG）：PopQA 从 45.5 掉到 41.8，ASQA 从 32.1 掉到 28.6。证明 relevance filtering + multi-passage beam search 比盲目 top-1 强。
- **No retrieval**：PopQA 从 45.5 崩到 24.7。PopQA 是长尾实体 QA，没检索就废。

### 人工评估（Figure 4d）

50 个 PopQA + 50 个 Bio 样本人工评估：
- 模型自己吐的 `[ISREL]` 跟人判断一致率 95%（PopQA）/ 90%（Bio）
- `[ISSUP]` 一致率 90% / 85%

证明这些 reflection token 不是 trained artifact，是真的有 semantic meaning 的。

### Parametric vs non-parametric 分析（Appendix C.1）

模型答对时，答案是不是真在 retrieved passage 里？
- Alpaca 13B：20% 答对但答案不在 passage（用 parametric knowledge 硬答）
- Llama2-chat 13B：18%
- **SELF-RAG：仅 2%**

SELF-RAG 几乎完全 grounded。这是 factuality 最硬的证据。

---

## 这玩意儿本质上是啥

把传统 RAG 的"系统决策"（啥时候 retrieve、用哪个 passage、生成了检查一下）从 engineering 代码挪进了 model weights。怎么挪的？把决策变成 special tokens，让模型自己学着生成。

更抽象地讲：**任何需要模型 self-monitoring 的行为，都可以通过 vocabulary 扩展变成 next-token prediction 的一部分**。这个 insight 是 generalizable 的：

- **Tool use**：把 "call tool" → "tool output useful" → "response grounded in tool output" 做成 reflection token chain
- **Multi-step reasoning**：每步生成一个 `[STEP_VALID]` token 做 CoT self-check
- **Safety**：生成 `[SAFE]` / `[UNSAFE]` token 做 self-censorship

这跟 Anthropic 的 Constitutional AI (https://arxiv.org/abs/2212.08073) 和 OpenAI 的 PRM (https://arxiv.org/abs/2305.20050) 思路相通 —— 都是把 evaluation 信号结构化，区别是 SELF-RAG 把它 collapse 进 vocabulary 而不是单独搞个 reward model。

---

## 局限

1. **Retriever 是 frozen 的**，用 Contriever-MS MARCO，没跟 generator joint train。RA-DIT (https://arxiv.org/abs/2310.01352) 探索了 joint training。
2. **Segment-level beam search 慢**，对每个 passage 并行生成 candidate，inference 成本是普通 greedy 的 $K \times B$ 倍。
3. **Reflection token 离散**，ISSUP 只有 3 档颗粒度有限。Continuous reward 可能更 expressive 但牺牲可解释性。
4. **依赖 GPT-4 蒸馏**，reproducibility 受限。
5. **Critic 是 Llama2-7B finetune 的**，如果 GPT-4 在某些 domain 判断就不好（比如小众领域），critic 也会继承这个 bias。

---

## 工程实现注意点

如果你要复现，几个容易踩坑的地方：

1. **Critic 和 generator 用同 base LM**（Llama2-7B），但 critic 其实不用太大，FLAN-3B 也能 80%+ accuracy。
2. **训练数据混合很关键**：150k 实例里 instruction-following（ShareGPT, Alpaca 各种）+ knowledge-intensive（NQ, FEVER, ASQA 等）。纯 knowledge-intensive 会让模型 over-retrieve，每次都 `[Retrieve=Yes]`。
3. **Passage masking in loss** 容易被忽视。不 mask 的话模型会去记忆 passage 内容，generalization 变差。
4. **Segment 切分用 Spacy sentence-level**。更细（clause-level）或更粗（paragraph-level）的影响论文没探索，是个 open question。
5. **Inference 用 vLLM** (https://arxiv.org/abs/2309.06180) 加速，不然 beam search 慢得受不了。
6. **Tokenizer 要扩展**，把 reflection tokens 加进 vocab，注意 embedding 要重新 init 或者用 mean init。

---

## 最值得带走的一句话

SELF-RAG 教会我们的不是某个具体技术，是一个 design pattern：

> **把 model 的 self-monitoring signals 显式化为 vocabulary tokens，然后用 SFT 训进模型 weights。**

这样 inference 时一个 forward pass 就完成了 generate + critique，不需要外部 critic model，不需要 RL，还能 test-time 调权重定制行为。这是 LLM 时代 system building 的一个范式 —— 用 vocabulary 扩展代替 system orchestration。

Reference links:
- Paper: https://arxiv.org/abs/2310.11511
- Code: https://github.com/AkariAsai/self-rag
- Project: https://selfrag.github.io/
- Author thread: https://twitter.com/AkariAsai/status/1719371206742364160
- Original RAG: https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf
- RA-DIT concurrent: https://arxiv.org/abs/2310.01352
- Toolformer: https://arxiv.org/abs/2302.04761
- ALCE benchmark: https://arxiv.org/abs/2305.14627
- FLARE: https://arxiv.org/abs/2305.06983
- Self-Refine: https://arxiv.org/abs/2303.17651
- CoVE: https://arxiv.org/abs/2309.11495
- Constitutional AI: https://arxiv.org/abs/2212.08073
- PRM (Let's verify step by step): https://arxiv.org/abs/2305.20050
- Contriever: https://openreview.net/forum?id=jKN1pXi7b0
- vLLM: https://arxiv.org/abs/2309.06180

---

# SELF-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection

这篇 paper 由 Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi 写的，来自 University of Washington + AI2 + IBM Research。论文链接：https://arxiv.org/abs/2310.11511 ，项目主页：https://selfrag.github.io/ ，代码：https://github.com/AkariAsai/self-rag 。

---

## 1. 核心痛点与设计 intuition

### 1.1 传统 RAG 的三个根本问题

传统 RAG (Lewis et al., 2020, https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf) 把检索当做一个"无条件 prepend"的过程。这个设计在三个层面有缺陷：

**(a) Indiscriminate retrieval** — 不管 query 是 "Who wrote Hamlet?" 还是 "Write a poem about autumn"，都会 prepend 同样数量的 retrieved passages。前者需要 factual grounding，后者需要 creativity，强行检索反而引入 irrelevant context，会让模型 distracted (Shi et al., 2023, https://proceedings.mlr.press/v202/shi23a.html)。

**(b) Lack of relevance filtering** — 检索器返回 top-K，但里面可能掺杂 off-topic passages。模型没有显式机制去 reject 它们，只能"软吸收"，导致 generation quality 被污染。

**(c) No attribution enforcement** — 即使 retrieved passages 是相关的，生成出来的文本也未必 grounded in passages。Citation 是事后贴标签，没有 training-time supervision 强制 alignment。

### 1.2 SELF-RAG 的核心 insight

SELF-RAG 的设计哲学是 **把 retrieval, generation, critique 三件事 unify 进一个 LM 的 next-token prediction 框架里**，通过引入一组特殊的 "reflection tokens" 作为 control plane。关键直觉是：

> 如果一个 LM 能在生成过程中 **自己判断**(1) "这一段需不需要外部知识？"(2) "这个 passage 相关吗？"(3) "我生成的这句话被 passage 支持吗？"(4) "整体回答有用吗？" —— 那它就能在 inference 时 **自主地做 retrieval decision 和 self-verification**，而不需要外部 critic model。

这其实是一个 "self-play" 思想在 RAG 上的迁移：generator 既是 actor 也是 judge。Reflection tokens 是 judge 的输出，被嵌入到 generator 的 vocabulary 里，让两者 share 同一个 token stream。

---

## 2. Reflection Tokens 的设计

这是整篇 paper 最核心的"语言扩展"。SELF-RAG 把原始 vocabulary $V$ 扩展为 $V \cup \{\text{Retrieve}, \text{ISREL}, \text{ISSUP}, \text{ISUSE}\}$，每一类有若干 discrete 取值（见 paper Table 1）：

| Token Type | Input | Output Values | Semantics |
|---|---|---|---|
| **Retrieve** | $x$ 或 $(x, y)$ | {yes, no, continue} | 是否触发 retriever $R$ |
| **ISREL** | $(x, d)$ | {relevant, irrelevant} | passage $d$ 是否对 $x$ 有用 |
| **ISSUP** | $(x, d, y)$ | {fully supported, partially supported, no support} | $y$ 是否被 $d$ entail |
| **ISUSE** | $(x, y)$ | {1, 2, 3, 4, 5} | $y$ 对 $x$ 的整体 utility（5 最好）|

**设计直觉**：
- `Retrieve` 控制 **when** — 决定 retrieval 的 timing 和 frequency，避免 indiscriminate retrieval。
- `ISREL` 控制 **what** — 过滤掉检索器 noise，让 generator 显式表达"我看这个 passage 不行"。
- `ISSUP` 控制 **whether grounded** — 这是 attribution 的 training-time 监督信号，让模型学会"truthfulness w.r.t. evidence"。
- `ISUSE` 控制 **whether helpful** — 即使 grounded，回答可能跑题或没用，5-scale 的 utility 提供整体 quality 信号。

注意 `continue` 这个取值允许模型对同一个 passage 跨多个 segment 复用，避免每段都重新检索的 overhead。这是一个很精细的设计 —— 一个 rich passage 可以支撑多段 generation。

---

## 3. 训练：Critic-then-Generator 两阶段蒸馏

### 3.1 阶段一：训练 Critic Model $C$

**问题**：手动标注 reflection tokens 极其昂贵，因为每个 segment 都要标 4 类。
**解法**：用 GPT-4 当 oracle 标注员，然后蒸馏到一个 in-house LM。

具体流程：
1. 从训练集 $\{X, Y\}$ 里随机 sample $\{X^{\text{sample}}, Y^{\text{sample}}\}$。
2. 对每个 reflection token 类别，设计一个 type-specific instruction prompt（见 paper Appendix D，Table 8-12 列出了完整 instruction 和 few-shot demonstrations）。
3. 用 GPT-4 生成 reflection token 作为 pseudo-label：$p(r | I, x, y)$，其中 $I$ 是 instruction。
4. 收集约 4k-20k 实例 per type，组成 $\mathcal{D}_{\text{critic}}$。

**Critic 的 training objective**（Eq. 1）：

$$
\max_{\mathcal{C}} \; \mathbb{E}_{((x,y), r) \sim \mathcal{D}_{\text{critic}}} \log p_{\mathcal{C}}(r | x, y)
$$

变量解释：
- $\mathcal{C}$：Critic model，初始化为 Llama2-7B（与 generator 同 base）。
- $(x, y)$：input-output pair。
- $r$：reflection token（如 `ISREL=Relevant`）。
- 这就是一个标准的 conditional LM objective，把 reflection token 当做 target label。

**蒸馏效果**（paper Appendix Table 5）：
- `Retrieve`：93.8% agreement with GPT-4
- `ISSUP`：93.5%
- `ISREL`：80.2%
- `ISUSE`：73.5%（5-scale 的相邻档位如 4 vs 5 难分，human 也难）

这种 "GPT-4 as data labeler, distill to small LM" 的 pattern 在 RLHF 的 reward model 训练里很常见 (Stiennon et al., 2020, https://arxiv.org/abs/2009.01325)，但 SELF-RAG 把它用在了 **fine-grained, multi-aspect** 的 reflection 上，比单标量 reward 更结构化。

### 3.2 阶段二：构造 Generator Training Data

这一步把整个训练 corpus 用 $C$ 标注成"reflection token 增强版"，让 generator 训练时 mimic inference 的行为。Algorithm 3 给了流程：

对每个 $(x, y) \in (X, Y)$：
1. 用 $C$ 预测 `Retrieve`：
   - 若 `Retrieve=No`：只在最后 append `ISUSE`，得到 $(x, y, \text{ISUSE})$。
   - 若 `Retrieve=Yes`：用 $R$ retrieve top-$K$ passages $D$。
2. 用 Spacy 把 $y$ 切成 sentences $\{y_1, \dots, y_T\}$。
3. 对每个 $y_t$，用 $C$ 再次预测 `Retrieve`（基于 $x, y_{<t}$, 和 initial retrieved passage）：
   - 若 `Yes`：用 $(x, y_t)$ 作为 query 重新 retrieve，对每个 passage 预测 `ISREL` 和 `ISSUP`。
   - 选取满足 `ISREL=Relevant` 且 `ISSUP=Fully/Partially Supported` 的 passage（若多个，取 retrieval score 最高的；若都不满足，随机选一个）。
4. 在 segment 末尾 append `ISUSE`。

最终得到 $\mathcal{D}_{\text{gen}}$，里面每条样本都长这样（见 paper Figure 2 的例子）：

```
[Retrieve=Yes]
<p>passage d</p>
[ISREL=Relevant]
y_t (next segment text)
[ISSUP=Fully Supported]
...
[ISUSE=5]
```

**关键细节**：训练时 retrieved passage $\langle p \rangle \dots \langle /p \rangle$ 之间的 token **在 loss 里被 mask 掉**。直觉是：模型不应该去"记忆"passage 内容，而应该学会"在看到 passage 时如何利用它"。这与 RAG 的 training philosophy 一致 —— passage 是 context，不是 target。

### 3.3 Generator 的 Training Objective（Eq. 2）

$$
\max_{\mathcal{M}} \; \mathbb{E}_{(x, y, r) \sim \mathcal{D}_{\text{gen}}} \log p_{\mathcal{M}}(y, r | x)
$$

变量解释：
- $\mathcal{M}$：Generator LM。
- $y$：output tokens（含 reflection tokens，但不含被 mask 的 passage tokens）。
- $r$：等价于 $y$ 里 reflection token 部分（这里写法上 redundant，强调 reflection tokens 是 $y$ 的一部分）。
- $(x, y, r)$：从 $\mathcal{D}_{\text{gen}}$ 采样的 augmented 样本。

这跟标准 LM 没有本质区别 —— 把 reflection tokens 加入 vocabulary 后，就是普通的 next-token prediction。**没有 RL，没有 PPO，没有 reward model in the loop**，只是 offline 用 $C$ 标好的数据做 SFT。这一点相对 RLHF 的成本优势是巨大的（paper Section 3.2.2 末段强调）。

---

## 4. Inference：自适应检索 + Critique-Guided Beam Search

这是 SELF-RAG 最有意思的部分。Algorithm 1 给出整体流程。

### 4.1 Adaptive Retrieval via Threshold

模型在生成每个 segment 前先预测 `Retrieve`。两种模式：

**Hard mode**：直接看 argmax，`Retrieve=Yes` 就触发检索。

**Soft mode（默认）**：用归一化概率比较阈值（Appendix A.3）：

$$
\frac{p([\text{Retrieve}]=\text{YES})}{p([\text{Retrieve}]=\text{YES}) + p([\text{Retrieve}]=\text{NO})} > \delta
$$

变量解释：
- $p([\text{Retrieve}]=\text{YES})$：模型给 `Retrieve=YES` token 的 next-token probability。
- $\delta$：阈值 hyperparameter。$\delta$ 越大 → 检索越保守；$\delta$ 越小 → 检索越激进。
- 默认 $\delta=0.2$（大多数任务），ASQA 设为 $0$（强制总是检索，因为需要 citation）。

这个 threshold 是一个 **test-time control knob**，可以让同一个模型在不同任务上调节 retrieval frequency。Paper Figure 3c 展示了在 PubHealth 和 PopQA 上调节 $\delta$ 时 retrieval frequency 和 accuracy 的 trade-off：PubHealth 上 retrieval 频率从 ~40% 降到 ~10% 时 accuracy 下降很缓，说明很多 query 不需要 retrieval 也能答对；PopQA 上下降陡峭，因为它是 long-tail entity QA，强依赖 external knowledge。

### 4.2 Segment-Level Beam Search with Critique Scores

当 retrieval 被触发后，$R$ 取回 top-$K$ passages（默认 $K=5$，论文里 ASQA 用 GTR-XXL 的官方 top-5；Bio/PopQA 还会加 Google Programmable Search 的 5 个）。对每个 passage $d$，generator 并行生成一个 candidate segment $y_t^{(d)}$。然后做 segment-level beam search（beam width $B=2$）。

每个 segment 的 score（Eq. 3-4）：

$$
f(y_t, d, \text{critique}) = \log p(y_t | x, d, y_{<t}) + \mathcal{S}([\text{Critique}])
$$

$$
\mathcal{S}([\text{Critique}]) = \sum_{G \in \mathcal{G}} w^G \cdot s_t^G, \quad \mathcal{G} = \{[\text{ISREL}], [\text{ISSUP}], [\text{ISUSE}]\}
$$

变量解释：
- $\log p(y_t | x, d, y_{<t})$：generator 对该 segment 的 log-likelihood（standard LM score）。
- $\mathcal{S}([\text{Critique}])$：critique token scores 的加权和。
- $w^G$：每种 critique token 的 weight，是 inference-time 可调的 hyperparameter。默认 $w^{[\text{ISREL}]}=1.0, w^{[\text{ISSUP}]}=1.0, w^{[\text{ISUSE}]}=0.5$。
- $s_t^G$：第 $G$ 类 critique token 的 normalized score。

每个 $s_t^G$ 的归一化（Eq. 4 上面）：

$$
s_t^G = \frac{p_t(\hat{r})}{\sum_{i=1}^{N^G} p_t(r_i)}
$$

- $\hat{r}$：most desirable token，比如 `ISREL=Relevant`，`ISSUP=Fully Supported`，`ISUSE=5`。
- $N^G$：第 $G$ 类下的 token 数量（ISREL=2, ISSUP=3, ISUSE=5）。
- 分母是所有候选值的概率之和，做 normalization 让 score 在 $[0, 1]$ 区间。

附录 A.3 给了更精细的 **multi-scale scoring**（不是只取 most desirable，而是部分 credit）：

**ISREL score**：
$$
s([\text{ISREL}]) = \frac{p([\text{ISREL}]=\text{RELEVANT})}{p([\text{ISREL}]=\text{RELEVANT}) + p([\text{ISREL}]=\text{IRRELEVANT})}
$$
就是个二分类概率。

**ISSUP score**（关键：partial credit）：
$$
s([\text{ISSUP}]) = \frac{p([\text{ISSUP}]=\text{FULLY})}{S} + 0.5 \cdot \frac{p([\text{ISSUP}]=\text{PARTIALLY})}{S}
$$
$$
S = \sum_{t \in \{\text{FULLY, PARTIALLY, No}\}} p([\text{ISSUP}]=t)
$$
"Fully supported" 给 1.0 credit，"Partially supported" 给 0.5 credit，"No support" 给 0。这模拟了一个软 entailment 分数。

**ISUSE score**（5-scale weighted sum）：
$$
s([\text{ISUSE}]) = \sum_{i}^{5} w_i \cdot \frac{p([\text{ISUSE}]=i)}{S}
$$
$$
w = \{-1, -0.5, 0, 0.5, 1\} \quad \text{对应 } [\text{ISUSE}] = \{1, 2, 3, 4, 5\}
$$
1-2 分给负权重（penalize），4-5 分给正权重（reward），3 分 neutral。

### 4.3 Hard Constraints 可选

除了 soft score，可以加 hard filter：如果一个 candidate segment 的 `ISSUP=No support`，直接从 beam 里删掉。这是更激进的 factuality enforcement。Paper Section 3.3 末段提到这个选项。

### 4.4 Test-Time Customization

这是 SELF-RAG 区别于 RLHF 的一个核心卖点：**同一个训练好的模型，在不同任务上可以不重训就调整行为**。

Paper Figure 3b 做了一个非常漂亮的实验：在 ASQA 上调 $w^{[\text{ISSUP}]}$ 的权重：
- 增大 $w^{[\text{ISSUP}]}$ → citation precision 上升（generation 更被 evidence 支持）。
- 增大 $w^{[\text{ISSUP}]}$ → MAUVE（fluency）下降（generation 更短更保守，因为长 generation 更难 fully supported）。

这是一个直观的 **factuality vs. fluency trade-off** 的 control knob，类似 RLHF 里的 KL-penalty 或 temperature 调节，但 SELF-RAG 用的是结构化的 critique weight，更细粒度。

---

## 5. 实验结果深度解析

### 5.1 主表（Table 2）的关键 takeaways

实验覆盖 6 个任务：PopQA, TriviaQA（short-form QA）, PubHealth, ARC-Challenge（closed-set）, Biography generation, ASQA（long-form with citation）。

**vs. ChatGPT**：
- PubHealth（fact verification）：SELF-RAG-13B 74.5 vs ChatGPT 70.1 → **胜**
- PopQA：55.8 vs 29.3（无检索），但 Ret-ChatGPT 50.8 → SELF-RAG 凭 on-demand retrieval + relevance filtering 超过
- ASQA citation precision：70.3 vs ChatGPT 没标（Ret-ChatGPT 65.1） → SELF-RAG 13B 70.3 **胜**
- Bio FactScore：80.2 vs 71.8 → **胜**
- TriviaQA：69.3 vs 74.3 → 略输（ChatGPT parametric knowledge 强）

**vs. Llama2-chat-13B (Ret-Llama2-C13B)**：在所有任务上 SELF-RAG-13B 都赢。

**vs. SAIL, Toolformer**：这些是 retrieval-augmented instruction-tuning 的 baseline，SELF-RAG 全面胜出。

### 5.2 Ablation Studies（Table 3a）

这是理解每个 component 价值的关键：

| Ablation | PopQA | PubHealth | ASQA (em) |
|---|---|---|---|
| SELF-RAG (50k) | 45.5 | 73.5 | 32.1 |
| No Retriever R | 43.6 (-1.9) | 67.8 (-5.7) | 31.0 |
| No Critic C Test | 42.6 (-2.9) | 72.0 | 18.1 (**-14**!) |
| No retrieval | 24.7 (**-20.8**) | 73.0 | - |
| Hard constraints | 28.3 | 72.6 | - |
| Retrieve top 1 | 41.8 | 73.1 | 28.6 |
| Remove ISSUP | 44.1 | 73.2 | 30.6 |

**关键观察**：
1. **No Critic C Test** 在 ASQA 上从 32.1 崩到 18.1 —— 说明 critique-guided beam search 是 long-form 任务的核心。
2. **No retrieval** 在 PopQA 上从 45.5 崩到 24.7 —— PopQA 是 long-tail entity QA，强依赖 retrieval。
3. **Retrieve top 1**（传统 RAG 的做法）在 PopQA 上 41.8 vs SELF-RAG 45.5，在 ASQA 上 28.6 vs 32.1 —— 证明 relevance filtering 和 multi-passage beam search 比盲目用 top-1 强。
4. **Remove ISSUP** 在 ASQA 上从 32.1 掉到 30.6 —— ISSUP 提供 grounding 信号，去掉就少了 attribution 监督。

### 5.3 Training Data Scaling（Figure 4a-c）

用 5k, 10k, 20k, 50k, 150k 训练数据做 scaling 实验：
- PopQA 和 ASQA 上 scaling 效果显著（从 50k 到 150k 仍有提升）。
- PubHealth 上 scaling 效果弱（数据量级可能已经够用）。
- 对比 baseline Llama2-FT：50k → 150k 提升不明显，说明 **SELF-RAG 的 reflection token 框架能更好利用更多数据**，而 plain SFT 在 instruction-following 数据上饱和更快。

### 5.4 Human Evaluation（Figure 4d）

50 个 PopQA + 50 个 Bio 样本，人工评估：
- **S&P（supported & plausible）**：PopQA 92.5%, Bio 70.0% —— PopQA 短回答很容易 fully supported，Bio 长回答很难。
- **ISREL 准确率**：PopQA 95.0%, Bio 90.0% —— 模型自己判断的 relevance 跟人类高度一致。
- **ISSUP 准确率**：PopQA 90.0%, Bio 85.0% —— entailment 判断也很可靠。

这证明 reflection tokens 不只是 trained signal，在 test-time 也是可靠的 self-verification 信号。

### 5.5 Parametric vs. Non-parametric Memory（Appendix C.1）

在 TriviaQA 和 PopQA 上分析：模型答对时，answer 是否在 retrieved passage 里？
- Alpaca 13B：20% 答对但 answer 不在 passage 里 → 用 parametric knowledge 硬答
- Llama2-chat 13B：18%
- Alpaca 7B：15%
- **SELF-RAG：仅 2%** → 几乎完全 grounded in retrieved passages

这是 factuality 的强证据：SELF-RAG 真的在"follow evidence"，而不是 hallucinate then 碰巧答对。

---

## 6. 与相关工作的关系

### 6.1 vs. RLHF (Ouyang et al., 2022, https://openreview.net/forum?id=TG8KACxEON)

| 维度 | RLHF | SELF-RAG |
|---|---|---|
| Reward 信号 | 单标量 scalar reward | 结构化 4 类 reflection tokens |
| 训练成本 | 高（PPO + reward model in loop） | 低（offline SFT） |
| Inference 控制 | KL-penalty 调节 | $w^G$ 权重 + $\delta$ 阈值 |
| Attribution | 无显式监督 | ISSUP 训练时显式监督 |

SELF-RAG 本质上是把 RLHF 的 reward signal 拆解成 **离散的、可解释的、可生成的 tokens**，从而把 RL 问题 reduce 成 supervised learning。

### 6.2 vs. Toolformer (Schick et al., 2023, https://arxiv.org/abs/2302.04761)

Toolformer 也训模型生成 API call token，但只对 named entity 触发，且 retrieval 后不 critique。SELF-RAG 的 reflection tokens 覆盖 retrieval decision + passage quality + generation grounding + overall utility 四个维度，更全面。

### 6.3 vs. SAIL (Luo et al., 2023, https://arxiv.org/abs/2305.15225)

SAIL 在 instruction tuning 时固定 prepend top retrieved passages。无 relevance filtering，无 grounding check。SELF-RAG 在消融里 "No Critic C Test" 类似 SAIL 的设置，结果在 ASQA 上掉 14 个点。

### 6.4 vs. Self-Refine (Madaan et al., 2023, https://arxiv.org/abs/2303.17651) / CoVE (Dhuliawala et al., 2023, https://arxiv.org/abs/2309.11495)

Self-Refine 是 prompt-time 的 iterative refinement，没有训练时 grounding。SELF-RAG 把 critique baked 进 model weights，inference 时一次 forward pass 就完成（虽然 beam search 有成本）。CoVE 用 Llama2-65B iterative prompting，参数量大 10 倍，在 Bio 上才达到 71.2 FactScore，而 SELF-RAG 13B 是 80.2。

### 6.5 vs. Active Retrieval (Jiang et al., 2023, https://arxiv.org/abs/2305.06983)

FLARE 主动检索基于 generation confidence，但用 proprietary LM，没有 training-time grounding，没有 attribution 监督。SELF-RAG 用 reflection tokens 实现更结构化的 on-demand retrieval。

---

## 7. 局限性与延伸思考

### 7.1 明显局限

1. **Retriever 是 frozen 的** — 用 Contriever-MS MARCO (Izacard et al., 2022, https://openreview.net/forum?id=jKN1pXi7b0)，没有跟 generator joint train。Concurrent work RA-DIT (Lin et al., 2023, https://arxiv.org/abs/2310.01352) 探索了 joint training。
2. **Segment-level beam search 成本高** — 每个 retrieved passage 要并行生成 candidate，比 standard greedy decoding 慢 $K \times B$ 倍。Paper 用 vLLM (Kwon et al., 2023, https://arxiv.org/abs/2309.06180) 加速，但 inference latency 仍是 bottleneck。
3. **Reflection token 是离散的** — `ISSUP` 只有 3 档，颗粒度有限。Continuous reward (如 RLHF 里的 scalar) 可能更 expressive，但牺牲可解释性。
4. **依赖 GPT-4 蒸馏** — Critic training 数据来自 GPT-4 标注，reproducibility 受限。未来 work 可以用 human-annotated subset 或 self-bootstrapping。

### 7.2 设计哲学的延伸

SELF-RAG 的核心 insight 可以推广到更广的 setting：**任何需要模型 self-monitoring 的任务，都可以通过把 monitoring signal 转化为 special tokens 来做 training-time supervision**。比如：

- **Tool use**: 把 "call tool X" → "tool result useful?" → "response grounded in tool result?" 串成 reflection tokens，统一进 LM。
- **Multi-step reasoning**: 每步生成 `STEP_VALID` token，做 chain-of-thought 的 self-verification。
- **Safety**: 生成 `SAFE`/`UNSAFE` token 做 self-censorship。

这跟 Anthropic 的 Constitutional AI (Bai et al., 2022, https://arxiv.org/abs/2212.08073) 和 OpenAI 的 process reward models (PRM, Lightman et al., 2023, https://arxiv.org/abs/2305.20050) 思路相通，都是把 evaluation 信号结构化。

### 7.3 跟后续工作的联系

- **InstructRAG / RA-DIT** (2023): 把 retriever 也纳入 instruction tuning。
- **CRAG / Adaptive RAG** (2024, https://arxiv.org/abs/2401.15884): 在 SELF-RAG 基础上加 retrieval correctness 分类器。
- **Self-RAG+** (2024): 扩展到更大规模 base model。
- **Search-augmented LLM with citation** (Gao et al., 2023, https://arxiv.org/abs/2305.14627): ALCE benchmark，SELF-RAG 是 SOTA baseline。

### 7.4 Implementation 注意点

如果你要复现 SELF-RAG，几个关键 engineering 细节：
- **Critic 和 Generator 用同 base LM**（Llama2-7B），但 critic 不需要太大，因为只是分类任务。FLAN-3B 也能跑出 80%+ accuracy。
- **训练数据混合**：150k 实例，instruction-following (ShareGPT, GPT-4 Alpaca, Stanford Alpaca, FLAN-V2, OpenAssistant) + knowledge-intensive (NQ, WoW, FEVER, OpenBookQA, ARC-Easy, ASQA)。混合比很关键，纯 knowledge-intensive 会让模型 over-retrieve。
- **Passage masking in loss**：这是容易被忽视的细节，但不 mask 会让模型把 passages 当成 memorization target，generalization 变差。
- **Segment 切分用 Spacy**，sentence-level granularity。论文里没探索更细（如 clause-level）或更粗（如 paragraph-level）的影响，是个 open question。

---

## 8. 总结：这篇 paper 真正的 contribution

抛开实验数字，SELF-RAG 真正教会我们的是：

**"Self-reflection is a learnable skill, and can be unified with generation through vocabulary expansion."**

通过把 retrieval decision、relevance judgment、entailment check、utility scoring 都转成 next-token prediction 的一部分，SELF-RAG 把一个看似复杂的 multi-component system 简化成 **a single LM with special tokens + a frozen retriever**。这种"用语言扩展代替系统设计"的思路，是 LLM 时代 system building 的一个范式。

Reference links:
- Paper: https://arxiv.org/abs/2310.11511
- Code: https://github.com/AkariAsai/self-rag
- Project page: https://selfrag.github.io/
- Author Twitter thread: https://twitter.com/AkariAsai/status/1719371206742364160
- Original RAG paper: https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf
- RA-DIT (concurrent): https://arxiv.org/abs/2310.01352
- Toolformer: https://arxiv.org/abs/2302.04761
- ALCE benchmark: https://arxiv.org/abs/2305.14627
- FLARE (Active retrieval): https://arxiv.org/abs/2305.06983
- Self-Refine: https://arxiv.org/abs/2303.17651
- CoVE: https://arxiv.org/abs/2309.11495
- Contriever: https://openreview.net/forum?id=jKN1pXi7b0
- vLLM: https://arxiv.org/abs/2309.06180
