---
source_pdf: Recursive Language Models Meet Uncertainty The Surprising.pdf
paper_sha256: c5bf64a4a8f944e7a78fe7be5fb0d8f020ba91c14027e9072e1aedf43855f136
processed_at: '2026-08-11T21:57:41-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

我们用人话拆解一下这篇paper的底层逻辑，看看它到底在解决什么痛点，以及背后的intuition。

### 1. 痛点在哪？
现在的 LLM 即便 context window 有上百万 token，处理 long context 时依然拉胯，经常会 lost in the middle。之前有个叫 RLM 的 SOTA 方法试图解决这个问题。RLM 的思路是：别让模型硬吞上百万 token，把文本丢进一个 sandbox 里当 external variable，让模型写 Python 代码去 query、slice 这些文本。模型还能自己调用自己去处理 sub-problems，这叫 recursion。

### 2. 这篇paper发现了什么问题？
RLM 有个致命问题一直没人深究：模型写出来的 context-interaction 代码轨迹（trajectory）有好有坏，但 RLM 每次只跑一条，全凭运气。而且，RLM 成功的头号功臣真的是 recursion 吗？

### 3. SRLM 怎么破局？
这篇 paper 提出 SRLM：既然是写代码处理文本，干脆把多条路径都跑一遍，挑个最好的。它并行跑 8 条 trajectory，然后靠模型自己的“内省”信号来 selection，完全不需要外部的 reward model 或者 labeled data。

具体用了三个互补的 uncertainty signal：
1. **Self-Consistency**：8 条路径里，如果有 6 条算出答案 A，那大概率 A 是对的。先保留算出 A 的集合 $S$。
2. **Verbalized Confidence**：在每一步生成时，要求模型输出一个 JSON 报自信分 $\nu_t^{(k)}$。聚合公式是 $\text{VC}(p^{(k)}) = \sum_{t=1}^{T^{(k)}} \log(\nu_t^{(k)} / 100)$。因为 $\nu_t^{(k)} \in (0, 100]$，除以 100 后在 $(0,1]$ 之间，取 log 必为负数。求和后 $\text{VC}(p^{(k)}) \leq 0$，值越接近 0 说明整条轨迹 globally 越自信。
3. **Reasoning Trace Length**：模型越没把握，废话越多。$\text{Len}(p^{(k)}) = \sum_{t=1}^{T^{(k)}} \ell_t^{(k)}$，也就是生成的总 token 数。

选的时候，把后两个信号相乘：$s(p) = \text{VC}(p) \cdot \text{Len}(p)$。因为 $\text{VC}(p) \leq 0$ 且 $\text{Len}(p) > 0$，乘积必定 $\leq 0$。我们要选 $s(p)$ 最大的那个（最接近 0），这就意味着这条轨迹既高度自信，又精简干练，没有废话。

### 4. 实验结果有多颠覆？
1. **Recursion 纯属多余**：把 RLM 的“自己调用自己”功能关掉，只靠 SRLM 的反思机制，性能居然比带 recursion 的 RLM 还好，而且因为 8 条路径是 parallel 跑的，wall-clock time 反而更短。Recursion 在 GPT-5 这种强 base model 上甚至会让性能下降，属于帮倒忙。
2. **短文本上 RLM 是毒药**：在模型原生窗口能装下的短文本（<131K token）上，硬用 RLM 搞 recursion 拆解，性能还不如啥都不做的 Base LLM。因为本来就不用拆，硬拆引入了不必要的 overhead 和误差。SRLM 在长短文本上都很 robust。
3. **语义任务 RLM 废了**：Code QA 这种结构化任务很适合 RLM 拆解。但在 Document QA 或 Dialogue History 这种需要整体语义理解的任务上，硬拆就抓瞎了。SRLM 靠语义层面的反思，全面碾压。

### 5. Ablation 细节与 Intuition
为什么这三个信号缺一不可？Ablation study 的 heatmap 显示，单一信号都不靠谱：
- Self-Consistency 只看最终答案，太 coarse，不管过程多烂。
- Verbalized Confidence 容易 overconfident，模型可能自己骗自己。
- Trace Length 可能被模型本身的 verbosity 偏好带偏（有的模型天生话多）。

只有当 Verbalized Confidence 高且 Trace Length 短的情况 align 时，Accuracy 才是最高的。这就 build 了一个 intuition：长文本推理的真正瓶颈在于“面对海量信息，如何有效地在思路上搜索并验证”。Recursion 只是一种强加的 structural inductive bias，假设问题能完美 hierarchical 拆解，这在现实中往往不成立。SRLM 用 Self-Reflection 做 Inference-time scaling，通过采样多个候选并利用 intrinsic uncertainty 信号筛选，提供了一种更 flexible 的解法。这本质上是一种无需训练的 Best-of-K 搜索，verifier 就是模型自己的 internal state。

### References
- [SRLM/RLM paper](https://arxiv.org/abs/2512.24601)
- [Self-Consistency](https://arxiv.org/abs/2203.11171)
- [Trace Length as Uncertainty](https://arxiv.org/abs/2510.10409)
- [Verbalized Confidence](https://arxiv.org/abs/2306.13063)
- [LongBench-v2](https://arxiv.org/abs/2412.15204)
- [OOLONG](https://arxiv.org/abs/2511.02817)
- [BrowseComp-Plus](https://arxiv.org/abs/2508.06600)

---

# SRLM: Self-Reflective Program Search for Long Context 深度解析

## 1. Paper 的核心 Question

这篇 paper 来自 Apple, 核心问题是: **在 RLM (Recursive Language Models) 中, recursion 到底是不是性能提升的真正 driver?**

RLM 的做法是把长 context externalize 成一个编程环境里的 variable, 让 model 生成 program 来 query、slice、recursively interact context。但 paper 指出一个被忽略的维度: **如何选择 context-interaction program 的 trajectory** 这个问题一直没被研究过。RLM 依赖固定的 recursion scheme, 缺少 principled 的机制来 evaluate 和 select candidate reasoning trajectories。

这篇 paper 提出 SRLM, 用 uncertainty-aware self-reflection 来 augment programming-based context interaction。

---

## 2. Problem Formulation 的技术细节

给定 query $q$ 和长 context $\mathcal{C} = (c_1, c_2, \ldots, c_N)$, 其中 $N \gg L$ ($L$ 是 model 的 effective context window)。

关键 design choice: 不直接把 $\mathcal{C}$ feed 给 model, 而是把它当作 sandboxed execution environment 里的 external variable。

一个 context-interaction program 定义为:
$$p = (p_1, p_2, \ldots, p_T)$$

其中 $T$ 是 executable operations 的数量 (比如 slicing、querying、aggregating over $\mathcal{C}$)。每个 operation autoregressively 生成并在 REPL 里执行, 产生 intermediate execution state:

$$e_t = \text{EXEC}(p_t, e_{t-1}, \mathcal{C})$$

- $e_t$: 第 $t$ 步的 execution state
- $e_{t-1}$: 上一步的 state (初始 $e_0 = \emptyset$)
- $p_t$: 第 $t$ 个 operation
- $\mathcal{C}$: 长 context (作为外部可访问的 variable)

Terminal step 产生 program output $\text{out}(p) \in \mathcal{A}$, $\mathcal{A}$ 是 answer space。

**关键区别**: SRLM 不要求 program instantiate explicit self-query sub-calls 或 recursive model invocations。这把 context interaction 的质量和 recursion 的结构 decouple 开来, 把 long-context reasoning 的改进焦点从 "怎么 decompose" 转移到 "怎么 select candidate trajectories"。

---

## 3. SRLM 的三个 Uncertainty Signals

SRLM 从 model policy $\pi_\theta$ 独立采样 $K$ 个 candidate programs:
$$p^{(k)} \sim \pi_\theta(\cdot \mid q, \mathcal{C}), \quad k = 1, \ldots, K$$

每个 $p^{(k)}$ 是一个 distinct reasoning trajectory, 在 inspect 哪些 context segments、怎么 decompose sub-problems、对 intermediate conclusions 的 confidence 都不同。

paper 提出 3 个 complementary 的 uncertainty signals, 全部 derived from model 自己的 generation process, 不需要 verifier、reward model 或 external labeled data:

### 3.1 Sampling-based Uncertainty (Self-Consistency)

基于 [Wang et al., 2022](https://arxiv.org/abs/2203.11171) 的 self-consistency 思想。给定 $K$ 个 independent draws, 任何 candidate answer $a \in \mathcal{A}$ 的 empirical frequency 作为 model marginal confidence 的估计:

$$\text{prob}(a) = \frac{1}{K} \sum_{k=1}^{K} \mathbf{1}[\text{out}(p^{(k)}) = a] \approx \mathbb{P}_{\pi_\theta}[\text{out}(p) = a \mid q, \mathcal{C}]$$

- $\mathbf{1}[\cdot]$: indicator function
- $K$: 采样的 candidate program 数量 (paper 里用 $K=8$)
- $\text{out}(p^{(k)})$: 第 $k$ 个 program 的 final answer

Plurality answer:
$$\hat{a} = \arg\max_{a \in \mathcal{A}} \text{prob}(a)$$

Consistent candidate set (和 plurality answer 一致的 programs):
$$S = \{p^{(k)} \in \mathcal{P} : \text{out}(p^{(k)}) = \hat{a}\} \subseteq \mathcal{P}$$

**Intuition**: 这一步做 implicit verification。但 paper 明确指出 self-consistency 是一个 **coarse** 的 uncertainty signal — 它只 operate 在 final output 层面, 对 trajectory 的 quality 不 sensitive。$S$ 里的 programs 可能 share 同一个 $\hat{a}$, 但在 "inspect 了哪些 context segments"、"intermediate sub-problems 解决得有多 confident"、"deliberation 有多深" 上差异很大。

### 3.2 Semantic Uncertainty (Verbalized Confidence)

受 [Xiong et al., 2023](https://arxiv.org/abs/2306.13063) 启发, 在每个 intermediate generation step $t$ 都 elicit model 对自己 confidence 的 self-assessment。

具体做法: append 一个 structured instruction 到 prompt, 要求 model 在 standardized format 里 report confidence score:

```json
{"confidence": $\nu_t^{(k)}$}
```

其中 $\nu_t^{(k)} \in (0, 100]$, 反映 model 在 step $t$ 对 intermediate conclusion 的 self-assessed certainty。

Verbalized confidence score of program $p^{(k)}$:

$$\text{VC}(p^{(k)}) = \sum_{t=1}^{T^{(k)}} \log\left(\nu_t^{(k)} / 100\right) \leq 0$$

- $T^{(k)}$: program $p^{(k)}$ 的总步数
- $\nu_t^{(k)}$: step $t$ 的 verbalized confidence (0 到 100)
- $\nu_t^{(k)} / 100 \in (0, 1]$, 所以 $\log(\cdot) \leq 0$
- 整个 trace 在 log-space 聚合, values 越接近 0 表示 globally higher confidence

**为什么用 log-space aggregation**: 这是概率论里的标准做法 — log probability 是 additive 的, 对应 joint probability 的乘法。每个 step 的 confidence 独立相乘 → 取 log 变成求和。这给 trajectory 一个全局的 confidence 度量。

**关键性质**: 和 self-consistency 不同, $\text{VC}(p^{(k)})$ 是 **semantic** uncertainty measure, 捕获 model 如何 endorse 每个 intermediate reasoning step, 而 self-consistency 只看 final answer。

### 3.3 Behavioral Uncertainty (Reasoning Trace Length)

这是个 implicit behavioral signal。$\ell_t^{(k)}$ 是 step $t$ 的 reasoning + output token 数量:

$$\text{Len}(p^{(k)}) = \sum_{t=1}^{T^{(k)}} \ell_t^{(k)}$$

**Intuition**: 当 model 不确定时, 倾向于生成 longer、more deliberative 的 traces; 而 confident、well-grounded 的 reasoning 往往 associated with concise outputs。这和近期 reasoning model 的观察一致 — incorrect reasoning trajectories 倾向于比 correct 的更长 ([Devic et al., 2025](https://arxiv.org/abs/2510.10409); [Marjanović et al., 2025](https://arxiv.org/abs/2504.07128))。

**为什么 complementary**: verbalized confidence 依赖 model 的 explicit self-report, 容易受 miscalibration 影响 (model 可能 overconfident)。Trace length 是 observable generation statistics, 不直接受 stated confidence 的 miscalibration 影响。

---

## 4. Joint Uncertainty-guided Selection

在 consistent candidate set $S$ (self-consistency 已经 enforced) 内, 把剩下两个信号 unify 成 joint uncertainty score:

$$s(p) = \text{VC}(p) \cdot \text{Len}(p)$$

- lower values of $s(p)$ → better candidates
- $s(p) \leq 0$ 因为 $\text{VC}(p) \leq 0$ 且 $\text{Len}(p) > 0$

**Intuition**: 这个 score penalize 两类 programs:
1. Express low confidence (VC 很负)
2. Require excessively long reasoning traces (Len 很大)

两者都是 uncertainty 的 indicators。

Optimal program:
$$p^* = \arg\max_{p \in S} s(p)$$

注意: $s(p) \leq 0$, $\arg\max$ 选的是 least negative 的, 即 confidence 高且 trace 短的。

Final prediction: $\hat{y} = \text{out}(p^*)$

---

## 5. 实验结果的核心 Takeaways

### 5.1 Main Results (Table 1)

| Backbone | Dataset | RLM | SRLM | 改进 |
|----------|---------|-----|------|------|
| Qwen3-Coder-480B | LongBench-v2 CodeQA | 59.8 | 64.9 | +5.1 |
| Qwen3-Coder-480B | BrowseComp+ (1K) | 37.1 | 59.7 | **+22.6** |
| Qwen3-Coder-480B | OOLONG (131K) | 45.7 | 51.8 | +6.1 |
| GPT-5 | LongBench-v2 CodeQA | 59.5 | 68.9 | +9.4 |
| GPT-5 | BrowseComp+ (1K) | 86.0 | 92.4 | +6.4 |
| GPT-5 | OOLONG (131K) | 53.0 | 65.5 | +12.5 |

SRLM 在所有 dataset 和 backbone 上都最好, 最高 22% improvement over RLM (Qwen3-Coder-480B on BrowseComp+ 1K)。

### 5.2 Recursion 不是 Performance 的 Primary Driver

这是 paper 最 striking 的发现。看 Figure 3 的 Pareto comparison:

- **Self-reflection 可以 outperform recursion** in both performance 和 cost (wall-clock time)
- SRLM 的 $K$ 个 trajectories 是 parallel 执行的, 所以 wall-clock time 相比只跑 1 个 trajectory 的 RLM 没有显著增加
- 关键证据: 在 LongBench CodeQA with Qwen3-Coder-480B 上, base model 20 → RLM without sub-calls 53.8 → RLM with recursive sub-calls 59.8。但 SRLM without sub-calls 也能达到 59.0, 说明 recursion 只贡献 marginal gains

**Intuition**: recursion 本质上是 inference-time scaling through "model as tool use" (model decompose problem into sub-queries, recursively call itself)。SRLM without sub-calls 是 inference-time scaling through "model internals" (implicit uncertainty-guided self-reflection)。后者更高效。

### 5.3 Context Length Robustness (Figure 2)

- **SRLM 的优势随 context length 增加而更显著**
- **RLM 对 context length 非常 sensitive**: 在 shorter contexts (<131K, 在 model native context window 内), RLM 往往 **underperform base model** — recursive decomposition 在 context 已经 manageable 时引入 unnecessary overhead
- **SRLM 在 short 和 long context 上都 robust**, 都比 base model 有 consistent gains

**为什么这重要**: 这说明 RLM 的 recursive decomposition 不是 universally beneficial 的。当 context 本来就在 window 内, 硬要 decompose 反而 hurt performance。SRLM 的 self-reflection 没有这个问题。

### 5.4 Task Semantics (Figure 4)

paper 扩展到 LongBench-v2 的所有 domain (不只是 CodeQA):

- **Recursion 在 structured、search-oriented tasks** (Code QA, Structured Data QA) 上更 effective
- **Recursion 在 semantically demanding tasks** (Dialogue History QA, Document QA) 上 less beneficial
- **SRLM 的 self-reflection 在所有 task category 上都 consistent gains**

**Intuition**: search-oriented 任务的 context 是 modular、well-organized 的, 适合 recursive、programmatic traversal。而 semantically intensive 任务需要 understand 和 integrate evidence distributed throughout entire context, heuristic program search 不够。SRLM 的 uncertainty-aware self-reflection 提供 higher-level semantic signal, 更有效 steer reasoning。

### 5.5 Ablation Study (Figure 5)

Top row: 每个 uncertainty signal 的 contribution
- Full SRLM (三个信号组合) consistently outperforms 任何 single signal
- 这证明三个信号是 **complementary** 的

Bottom row: verbalized confidence 和 reasoning trace length 的 interaction
- Performance jointly depend on both signals
- Relation with accuracy **不是 strictly linear** — high confidence 或 short trace alone 不 reliably indicate correctness
- Highest accuracy 出现在 **两个信号 align** 时

---

## 6. 架构图解析 (Figure 1)

SRLM 的 pipeline:

1. **Input**: query $q$ + 长 context $\mathcal{C}$
2. **REPL Environment**: context externalized as variable, model 生成 program 来 query/interact context
3. **K Candidate Programs**: 从 $\pi_\theta$ 独立采样 $K=8$ 个 programs
4. **三个 Uncertainty Signals 并行计算**:
   - Self-consistency: 比较 $K$ 个 programs 的 final answers
   - Verbalized confidence: 每个 step 的 $\nu_t^{(k)}$
   - Reasoning trace length: 每个 step 的 $\ell_t^{(k)}$
5. **Self-Reflective Selection**: 在 consistent set $S$ 里用 $s(p) = \text{VC}(p) \cdot \text{Len}(p)$ 选 $p^*$
6. **Output**: $\hat{y} = \text{out}(p^*)$

关键: **no external supervision**, 全部信号来自 model 自己的 generation process。

---

## 7. 我的 Intuition 构建

### 7.1 为什么这个工作有意思

从 first principles 看, RLM 的 recursion 其实是把 long-context 问题 transform 成一个 **search problem over program trajectories**。但 RLM 只 run 一个 trajectory (greedy 的 fixed recursion), 浪费了这个 search space。

SRLM 的 insight 是: **如果你已经有 search space, 就应该 explore 它**。通过 $K=8$ 个 samples + uncertainty-guided selection, SRLM 实际上做的是 best-of-K with learned (self-reflective) verifier。

### 7.2 三个信号的 complementary 性质

可以这样理解:
- **Self-consistency**: "agreement 意味着 reliability" — 但只看 final answer, coarse
- **Verbalized confidence**: "model 自己说有多确定" — semantic 但受 miscalibration 影响
- **Trace length**: "model 想了多久" — behavioral, 不受 verbalized miscalibration 影响, 但也可能被 model 的 verbosity 偏好 confound

三个信号从不同角度 (sampling、semantic、behavioral) approximate model 的 internal uncertainty, 互相 cover 各自的 blind spots。

### 7.3 为什么 recursion 不是关键

这个发现其实不意外。RLM 的 recursion 是一种 **structural inductive bias** — 它假设 long-context 问题可以 decompose 成 hierarchical sub-problems。但:
1. 不是所有 long-context 任务都是 hierarchical decomposable 的 (semantically intensive tasks)
2. 当 context 已经在 window 内, decomposition 是 unnecessary overhead
3. Decomposition 本身有 error accumulation 的风险

SRLM 用 self-reflection 替代 structural recursion, 更 flexible, 更 adaptive, 不强加 task structure 的假设。

### 7.4 和 test-time compute scaling 的关系

这篇 paper 可以看作 test-time scaling 的一个 instantiation。和 [Snell et al.](https://arxiv.org/abs/2408.03314) 的 "Scaling test-time compute" 工作一脉相承, 但 focus 在 long-context interaction 的 program search 上, 而不是 pure reasoning。

$K=8$ 的 parallel sampling + selection 是一种 best-of-K, 但 selection 不是用 external reward model, 而是用 model 自己的 intrinsic signals。这让它在 long-context (没有 easy reward signal) 的场景下特别 applicable。

---

## 8. Limitations 和 Future Directions

paper 自己提到:
- 用的是 relatively simple 的 self-reflection (三个 intrinsic signals 的乘积)
- 未来可以探索 richer 的 intrinsic self-reflection
- 可以 integrate decision-making with self-reflective signals 来 enable earlier termination 和 improved token usage control

我认为还有几个有意思的方向:
1. **Learned selection**: 现在的 $s(p) = \text{VC}(p) \cdot \text{Len}(p)$ 是 hand-designed 的, 可以 train 一个 small model 来 learn 更好的 combination
2. **Adaptive $K$**: 根据 query 难度动态调整 $K$, 简单 query 少 sample, 难 query 多 sample
3. **Process reward**: 把 verbalized confidence 和 trace length 作为 process reward signal 来 train model

---

## References

- [Paper: SRLM - arXiv (推测的 link)](https://arxiv.org/abs/2512.24601) (注: paper 里 reference [75] 指向 RLM, SRLM 本身的 link 未明确给出)
- [RLM: Recursive Language Models](https://arxiv.org/abs/2512.24601)
- [Self-Consistency: Wang et al., 2022](https://arxiv.org/abs/2203.11171)
- [Verbalized Confidence: Xiong et al., 2023](https://arxiv.org/abs/2306.13063)
- [Trace Length as Uncertainty: Devic et al., 2025](https://arxiv.org/abs/2510.10409)
- [Reasoning Model Behavior: Marjanović et al., 2025](https://arxiv.org/abs/2504.07128)
- [LongBench-v2](https://arxiv.org/abs/2412.15204)
- [OOLONG](https://arxiv.org/abs/2511.02817)
- [BrowseComp-Plus](https://arxiv.org/abs/2508.06600)
- [Semantic Entropy: Farquhar et al., Nature 2024](https://www.nature.com/articles/s41586-024-07421-0)
- [CodeAct](https://arxiv.org/abs/2402.01030)
- [ReAct](https://arxiv.org/abs/2210.03629)
- [GPT-5 System Card](https://arxiv.org/abs/2505.21306)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)

---

## 9. 总结

这篇 paper 的核心 contribution 不只是 SRLM 这个 method, 而是对 RLM 这类 programming-based context-interaction framework 的 **mechanistic understanding**。它 reposition recursion 为 long-context reasoning 的一个 component, 而不是 defining feature。Uncertainty-aware self-reflection 提供了一个 simple yet effective 的 alternative。

对 Andrej 来说, 这篇 paper 的 intuition 是: **long-context reasoning 的 bottleneck 不在 "能不能 process 长 context", 而在 "能不能在 candidate interaction programs 里做 principled selection"**。这是一个 search problem, 而 self-reflection 是一个 cheap、intrinsic、no-external-supervision 的 search signal。

值得注意的是, 论文作者名字和单位是 Apple, 一作是 Keivan Alizadeh 和 Parshin Shojaee。后者 (Parshin Shojaee) 也是 "The Illusion of Thinking" paper (reference [54]) 的作者, 那篇 paper 也是关于 reasoning model 的 limitation, 思路一脉相承。
