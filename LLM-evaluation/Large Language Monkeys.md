---
source_pdf: Large Language Monkeys.pdf
paper_sha256: 33889c6cb265ae11ed67e50070d3a31cd894036d8c9b404212a38df1783e8884
processed_at: '2026-08-05T11:56:24-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Large Language Monkeys 的人话版

## 一句话讲清楚在干嘛

大模型 inference 的时候别只 sample 一次。Sample 10000 次, 配一个能自动判对错的 verifier, 弱模型也能把强模型单 sample 打穿。这件事 AlphaCode 早就干过 (sample 100 万次), 这篇 paper 是把它 systematize 成一个 inference scaling 的 study, 顺便发现 coverage 跟 sample 数之间有类似 training scaling law 的规律。

## Mental model: 一个 "采样 + 验证" 的二选一框架

你面对一个问题, 有两个独立的 axis 决定 final success rate:

**Coverage**: 我 sample N 次, 至少有一次是对的, 这个概率是多少?
**Precision**: 在这 N 个 sample 里, 我能不能把那个对的挑出来?

两个缺一不可。Coverage 是上限 (oracle verifier 能做到的), precision 是实际能达到的。这篇 paper 的核心发现: **coverage 这个 axis 在 sample 数上 log-linearly 增长, 非常便宜; precision 这个 axis 早就 plateau 了**。

所以能不能 scale, 完全取决于你有没有一个好的 verifier。代码有 unit test, Lean4 有 proof checker — perfect verifier, coverage 增长直接等价 success rate 增长。数学 word problem 没有 verifier, 只能靠 majority voting 或者 reward model, 这俩东西在 100 sample 之后就饱和了。

## 为什么 coverage 会 log-linear 增长

直觉是这样的: 模型对每个问题有一个隐含的 "pass 概率" $p$。Sample 一次, pass 概率是 $p$。Sample $k$ 次, 至少 pass 一次的概率是 $1 - (1-p)^k$。当 $p$ 小的时候, 这个近似成 $1 - e^{-pk}$, 对 $k$ 来说是 exponential 增长。

但问题在于不同 problem 的 $p$ 是不同的。Benchmark 上有 easy problem ($p = 0.9$), 也有 hard problem ($p = 0.001$)。当你 sample 到 100 次时, easy 早就 solved 了, 你在解决的是 $p$ 比较小的那一批。所以总 coverage 是一个积分:

$$\text{coverage}(k) = \int 1 - (1 - p)^k \, dP(p)$$

这玩意的形状取决于 $p$ 的分布。如果是 power-law 分布 (大部分 problem 是 hard, 少部分 easy), 那 coverage 随 $k$ 的增长就是 sublinear 的 power law。

Paper 里用的 fit 形式是:

$$c \approx \exp(a \cdot k^b)$$

$a$ 基本对应 pass@1 的 log, $b$ 是 curvature。$b=1$ 是 "每多 sample 一次, log coverage 线性涨", 这是过于乐观的情形; 实际 $b < 1$, 表示 coverage 增长在减速。这个 form 是从 GPT-4 tech report 借来的 (他们用来 fit training compute vs pass rate), 这里把 training compute 换成了 inference compute (sample 数)。

**这个 form 没有理论 derivation, 就是 empirical fit**。但是 fit 得还挺好, 暗示着某种 inference-time scaling law 的存在。

## 最戏剧化的数字

- **Gemma-2B on CodeContests**: pass@1 = 0.02%, pass@10000 = 7.1%。**300 倍提升**。一个 2B 的小模型在 competitive programming 上能 hit 7%, 这是 sample 10000 次的暴力结果。
- **Pythia-160M on MATH**: pass@1 = 0.27%, pass@10000 = 57%。一个 160M 的 model (想想 GPT-2 mini 时代) 在 MATH 上 oracle coverage 57%。当然 oracle 意味着 perfect verifier, 实际上做不到, 但这个上限的存在本身就 surprising。
- **SWE-bench Lite**: DeepSeek-Coder-V2-Instruct, pass@1 = 15.9%, pass@250 = 56%。**单 sample SOTA 是 43%** (CodeStory Aide, 用 GPT-4o + Claude 3.5 Sonnet 的混合)。一个 open-source 模型 + 250 个 samples, 把 closed-source 单 sample 的 SOTA 超了 13 个百分点。

## 弱模型 + 多 sample vs 强模型 + 少 sample 的 cost trade-off

固定总 inference FLOPs, 哪个更划算?

Paper 的发现: **取决于任务**。
- GSM8K / MATH / MiniF2F (这些 task 的 "correct solution distribution" 比较 narrow): 8B + many samples > 70B + fewer samples
- CodeContests (solution space巨大, 弱模型 sample 10000 次基本就饱和了): 70B + fewer samples > 8B + many samples

这个分界线的 intuition: 如果 task 的 correct solution 在 model 的 distribution 里 mass 比较集中, 那多 sample 很快就把 correct solution "采到了", 弱模型就够了; 如果 solution space 巨大, 弱模型即使 sample 10000 次也只 cover 了其中一小部分, 这时候强模型每个 sample 质量更高的优势就显现出来。

SWE-bench 上的 dollar cost 对比更直接:

| Model | 单次成本 | sample 数 | 解决率 | 总成本 |
|---|---|---|---|---|
| DeepSeek-Coder-V2-Instruct | $0.0072 | 5 | 29.62% | $10.8 |
| GPT-4o | $0.13 | 1 | 24.00% | $39 |
| Claude 3.5 Sonnet | $0.17 | 1 | 26.70% | $51 |

便宜 18 倍的模型, 多 sample 几次, 性能反超 3-5 倍贵的 SOTA。这是 product-level 直接可用的 insight。

## Precision 问题: 为什么 math word problem 卡住

Figure 7 是全文最重要的一张图。Llama-3-8B-Instruct 在 MATH 上:

- Coverage: 100 samples → 82.9%, 10000 samples → 98.44% (持续上升)
- Majority voting: 100 samples → ~40%, 10000 samples → ~41% (基本平的)
- Reward model + Best-of-N: 类似 plateau
- Reward model + Majority voting: 类似 plateau

Gap 接近 57 个百分点。

为什么 majority voting 在 100 sample 后饱和? 直觉: 假设一个 hard problem 的 correct 概率 $p = 0.01$。100 个 sample 里, 期望 1 个对。但 wrong answer 是分散的, 100 个 sample 可能有 50 种不同的 wrong answer。Plurality (出现最多的) 仍然是某个 wrong answer (出现 2-3 次), correct 只出现 1 次。所以 majority vote 选错。只有当 $p \cdot k \gg 1$ 且 wrong answers 分散时, correct 才会成为 plurality, 但那时候 coverage 早就 saturate 了。

Reward model (ArmoRM-Llama3-8B-v0.1, 当时 RewardBench leaderboard 上表现很好的) 为什么也没好多少? 作者手动 review 了 105 个 correct sample 的 CoT, 发现 90%+ 是 faithful reasoning (即使 pass@1 < 10% 的 hard problem 也是这样)。**所以 correct sample 里是有 signal 的**, reward model 理论上应该能识别, 但实际不能。

这指向一个明确的 research direction: **现在的 reward model 训练时见到的 "correct reasoning" 大多是 high-confidence 的 case, 对 low-pass-rate region 的 hard correct reasoning 严重 OOD**。Process reward model (PRM, Lightman et al.) 的思路是对的方向, 但训练数据需要更 aggressive 地 mine hard examples。

Figure 8 那张 bar chart 把这件事可视化得很好: 每个 bar 是一个 problem, bar 高度是 10000 sample 里 correct 的 fraction。绿色的 bar 是 self-consistency 选对了, 红色是选错了。你能看到大量高度 < 1% 的 bar 是红色的 — correct sample 确实存在, 但 vote 选不出来。

## 同 family 模型的 coverage curve 有 "平行偏移"

这是 paper 里被低估的一个观察。Llama-3 8B 和 70B 在同一个 task 上的 coverage 曲线, 如果你把 x 轴 log scale, 然后横向平移让两条曲线都穿过同一个 anchor 点 $(1, c)$ (c 是 max pass@1), 它们的 shape 几乎重合。

意思是: 在同一个 model family 内, **scale model size 约等于在 log-sample-axis 上做一个 horizontal shift**。"用 2 倍大的模型" 和 "用 10 倍多的 samples" 在某些 coverage 区间能达到一样的效果。

这个 observation 如果 robust, 实际用途很大: 给定固定 FLOPs 预算, 你可以推算最优的 "model size × sample 数" 分配。这跟 Kaplan 当年 training scaling law 里 "scale width vs depth vs data" 的等价置换是同一种思路。Hoffmann 的 Chinchilla paper 修正了 Kaplan 的最优分配比例; 我觉得 follow-up 工作会修正这里的 sample-vs-size 分配。

## 两个 "verifier 也不完美" 的坑

这部分是 paper 的诚实之处, 也是同行继续做 benchmark 时应该注意的。

**SWE-bench Lite 的 flaky tests**: 11.3% 的 problem 跑 unit test 不稳定。同一份 candidate solution 跑多次, 有时 pass 有时 fail。其中 30/34 即使是 gold solution 也会 flaky。原因是 Python set 的 unordered、未 pin 的 dependency 等。Paper 的处理: 对 flaky problem 跑 11 次 test 取 majority vote。

**CodeContests 的 false negatives**: 122 个 Python3 problem 里有 35 个, **官方的 reference solution 跑官方的 test case 会 fail**。原因是 problem 允许多个 correct output 但 test 只接受一个, 或者程序生成的 mutated test case 违反 input spec (比如该是正整数却 mutated 成 0)。

这意味着所有在 CodeContests / SWE-bench 上 report 的 X% 数字, 都有 verifier noise, 都应该带 confidence interval。

## 几个延伸联想

**和 o1 的关系**: o1 的思路是单 sample + 长 CoT (RL 训出来的 thinking), 这篇是 多 sample + 短 CoT。两者是正交的, 可以叠加 (multi-sample × long-CoT)。o1 在 MATH 上 single sample 接近 90%, 等价于 Llama-3-8B 在 1000+ samples 下的 oracle coverage。所以 o1 某种意义上是把 "sampling scaling" 压缩进了 "single-sample long reasoning" 里, token efficiency 应该更高 (不重复 prefill, 不重复早期 reasoning)。

**和 random forest 的类比**: "弱模型 + 10000 samples + verifier" 可以看成 "10000 个 weak learner 的 ensemble + perfect selector"。这跟传统 ML 里 random forest vs single deep network 的 trade-off 是一个 mental model。Random forest 用很多弱树 + averaging 取胜, deep network 用一个强网络 + end-to-end training 取胜。这里也是: 你可以训一个超大模型, 也可以用中等模型 sample 一万次。后者在 verifier 可用的时候出奇地 work。

**为什么 i.i.d. sampling 浪费**: i.i.d. + temperature sampling 倾向于在 high-probability mode 周围重复采样。AlphaCode 当年用 metadata conditioning (用 problem tag 作为额外 condition) 强制让 sample 落到不同 mode 上, 这本质上是在做 "intentional diversity"。I.i.d. 是个 baseline, 不是最优。可以想象用 Gumbel-softmax、contrastive prompting、或者 explicit mode-seeking sampling 来提升 sample efficiency。

**System 层的 low-hanging fruit**: 当你 sample 10000 次同一个 prompt, prompt 的 prefill 只算一次, 只 decode 不同。Hydragen (作者 Juravsky 上一篇 paper)、SGLang、Bifurcated Attention 这些 shared-prefix attention 优化能大幅降低实际 wall-clock cost。Paper 用 FLOPs 估算 cost 可能高估了 5-10 倍。所以 "weak model + many samples" 的 cost advantage 在实际部署时可能更显著。

**Autoformalization 的连接**: Math word problem 上 verifier 难, 是因为没法自动判对错。如果把 informal math statement 自动翻译成 Lean, 然后用 Lean proof checker, math 就享受到了 code 的 perfect verifier 红利。这是 Wu et al. ICLR 2022 的方向, 也是 paper Section 5 末尾暗示的 "converters" 思路。这条路如果 work, math benchmark 的格局会变。

**一个未解问题**: Power law fit 里那个 $b$ 参数, paper 没给数值表 (只在 Appendix 给图)。我推测 MATH 的 $b \approx 0.5-0.7$, CodeContests 的 $b \approx 0.4$。如果 $b$ 跟 task difficulty 有单调关系, 那就能用 $b$ 来做 task hardness 的一个 intrinsic measure。这是 follow-up 可以挖的。

## 最核心的 takeaway

1. **Inference scaling 是真的, 而且有规律**。Coverage 大致遵循 $c \approx \exp(a k^b)$, 跟 training scaling law 是同一种 "predictable regularity"。
2. **Verifier 是 bottleneck, 不是 sampler**。Sample 多很简单, 难的是把 rare correct sample 从一堆 wrong 里挑出来。Math word problem 上的 majority voting / reward model 在 100 sample 后就 plateau 了。
3. **弱模型 + 多 sample 可以打赢强模型 + 单 sample**, 在有 verifier 的 domain 上, 而且 cost-effective。SWE-bench Lite 上 open-source 模型 + 250 samples 打败 GPT-4o + Claude 3.5 Sonnet 单 sample 的 SOTA, 这是个 product-level actionable 的结论。
4. **Benchmark 的 verifier 也不完美**, SWE-bench 11.3% flaky, CodeContests 35/122 false-negative。报道 X% on benchmark 时应该有 noise 估计。
5. **同 family 内, scale model size ≈ log-axis 上 shift sample 数**, 这给出一个 budget allocation 的 scaling law-like 框架。

---

# Large Language Monkeys: 一篇关于 inference-time scaling 的 systematization paper

## 1. 核心动机与定位

Andrej, 这篇 paper 的位置在 inference-time scaling 这个 emerging subfield 里, 和 OpenAI 的 o1 / "test-time compute" 系列、DeepMind 的 AlphaCode 早期观察、以及 Snell et al. 的 "Scaling test-time compute" 形成 sibling works。它的特殊之处在于 **systematic characterization**: 把 AlphaCode 当年那篇 "we sample 1M times" 的孤证, 变成 across 5 tasks、across 3 model families、across 4 orders of magnitude sample budget 的实证研究, 并且首次给出 inference-time 的 scaling law 形式化。

核心论点可以拆成两条 axis:
- **Coverage axis**: 当 sample budget k 增长, "至少有一个 sample 是对的" 的概率 (即 pass@k / oracle-verifier success rate) 如何变化?
- **Precision axis**: 当我们有 10000 个 sample, 能否把那个对的 sample 挑出来?

paper 的暗线是: **coverage 的 scaling 是 "cheap" 的, precision 的 scaling 是 "expensive" 的**。这正好解释了为什么 AlphaCode 在 coding 上能 work (unit test = perfect verifier), 而 math word problem 上 self-consistency 早早就 plateau。

Project page: https://scalingintelligence.stanford.edu/pubs/large_language_monkeys/

arXiv (后期版本): https://arxiv.org/abs/2412.19437

---

## 2. Setup: 把 inference scaling 定义清楚

### 2.1 Repeated sampling 的 pipeline

```
prompt → LLM (temperature T > 0) → sample_1, sample_2, ..., sample_N (i.i.d.)
       → domain-specific verifier → 选出 final answer
```

注意一个细节: paper 里的所有 experiments 都是 **single-turn, i.i.d. sampling**, 没有任何 cross-sample communication、没有 feedback loop、没有 iterative refinement。这是刻意 "naive baseline" 的选择 — 想看 pure sampling 的 scaling behavior。Section 5 discussion 里列了三个明确改进方向: solution diversity (像 AlphaCode 的 metadata conditioning)、multi-turn interaction、learning from previous attempts。

### 2.2 Coverage 的定义与 pass@k 公式

Coverage 在有 verifier 时就是 pass@k。Chen et al. (Codex paper, https://arxiv.org/abs/2107.03374) 给出了无偏估计:

$$
\text{pass@k} = \frac{1}{\#\text{problems}} \sum_{i=1}^{\#\text{problems}} \left( 1 - \frac{\binom{N - C_i}{k}}{\binom{N}{k}} \right)
$$

变量解读:
- $N$: 每个问题实际生成的总 sample 数 (paper 里最多 10000)
- $C_i$: 第 $i$ 个问题上, 这 $N$ 个 sample 里 correct 的个数
- $k$: 我们想评估的 sample budget (例如 pass@1, pass@100, pass@10000)
- $\binom{N-C_i}{k} / \binom{N}{k}$: 是从 $N$ 个 sample 中随机抽 $k$ 个, **全部抽到 wrong** 的概率 (hypergeometric distribution 的尾部)
- $1 - (\cdot)$: 至少抽到一个 correct 的概率
- 对所有问题取平均 → coverage

**Intuition**: 这个公式的本质是把 "我从 $N$ 里 sample $k$ 次, 至少 hit 一次 correct" 当成 hypergeometric trial。直接用 $1 - (1 - C_i/N)^k$ 是有偏的 (高估), 因为它假设 with replacement 且 $C_i/N$ 估计有 noise。Codex 的无偏估计在 small $C_i$ 时尤其重要 — 例如 $C_i = 1, N = 10000$ 时, 简单公式会严重高估 pass@k。

对于 GSM8K 和 MATH, "correct" 的定义是用 minerva_math 的 `is_equiv` 函数做 exact answer match (LMEval 框架, https://zenodo.org/records/10256836)。

### 2.3 5 个 task 与 verifier 的可用性矩阵

| Task | Verifier | Verifier 性质 |
|---|---|---|
| GSM8K | 无 (oracle only) | 需要 majority voting / reward model |
| MATH | 无 (oracle only) | 需要 majority voting / reward model |
| MiniF2F-MATH (Lean4) | Lean4 proof checker | 严格, deterministic |
| CodeContests | hidden test cases | black-box, 有 false negative 问题 |
| SWE-bench Lite | repo unit test suite | black-box, 有 flaky test 问题 |

这个矩阵其实埋了一条伏线: **verifier 的 quality 决定了 coverage gain 能否转化为 success rate gain**。Lean4 是 gold standard; CodeContests 和 SWE-bench 是 "noisy verifier"; GSM8K/MATH 是 "no verifier"。paper 后半部分的两个 "cautionary tales" 都落在 noisy verifier 一列。

---

## 3. Coverage 实验: 主要结果

### 3.1 Figure 2 的 headline numbers

| Task | Model | pass@1 | pass@N (N=?) | 单样本 GPT-4o | 单样本 SOTA |
|---|---|---|---|---|---|
| GSM8K | Llama-3-8B-Instruct | ~85% | 接近 100% @ 10k | ~92% | - |
| MATH | Llama-3-8B-Instruct | ~30% | ~98% @ 10k | ~76% | - |
| MiniF2F-MATH | Llama-3-8B-Instruct | ~25% | ~80% @ 10k | ~50% | - |
| CodeContests | Llama-3-8B-Instruct | ~2% | ~15% @ 10k | ~11% | - |
| SWE-bench Lite | DeepSeek-Coder-V2-Instruct | 15.9% | 56% @ 250 | 24% | 43% (CodeStory Aide, mix of GPT-4o + Claude 3.5) |

SWE-bench Lite 那条 56% > 43% 是 paper 的 headline claim。值得注意: 这是用同一个 agent framework (Moatless Tools, https://github.com/aorwall/moatless-tools) 下, **open-source 弱模型 + 250 samples 打败 closed-source 强模型 + 1 sample 的 mix**。这个对比稍微有点 cherry-picking, 因为 CodeStory Aide 那个 SOTA 也可以多 sample, 但作为 inference scaling 上限的 demonstration 是 valid 的。

### 3.2 Figure 3: across model sizes / families

Paper 跑了 Llama-3 (8B, 70B), Gemma (2B, 7B), Pythia (70M 到 12B 全套)。两个最 dramatic 的 data points:
- **Gemma-2B on CodeContests**: pass@1 = 0.02% → pass@10k = 7.1%, **300× 提升**
- **Pythia-160M on MATH**: pass@1 = 0.27% → pass@10k = 57%

Pythia-160M 是个 2023 年的小模型, 在 MATH 上能跑到 57% coverage 听起来很 crazy, 但这是 oracle coverage — 假设有 perfect verifier。真实 success rate 会差很多 (precision 问题后面讲)。

一个反例: **Pythia 全系列在 CodeContests 上 pass@10k = 0%**, 即使是 12B。论文推测是 Pythia 训练数据里 code-specific 内容太少。这其实暗示了一个 inverse 的事实: **repeated sampling 不能创造 capability, 只能放大已有的 capability**。如果一个 model 在某 domain 的 pass@1 真的是 0 (不是因为温度高被 truncate, 而是 model 的 mass 根本没分配到正确 token sequence 上), sample 多少次都没用。这呼应他们后面 Section 3 的 power law fit — 公式 $c \approx \exp(a k^b)$ 中 $a$ 就是 essentially pass@1 的近似 (代入 $k=1$, $c \approx e^a$)。

---

## 4. Cost-performance tradeoff (Section 2.3)

### 4.1 FLOPs-based comparison

论文用了一个简化公式估算 inference FLOPs:

$$
\text{TotalInferenceFLOPs} \approx \left( \sum_{t=1}^{\text{NumPromptTokens}} \text{FLOPsPerToken}(t) \right) + \left( \sum_{t=1}^{\text{NumDecodeTokens}} \text{FLOPsPerToken}(t + \text{NumPromptTokens}) \cdot \text{NumCompletions} \right)
$$

变量含义:
- 第一项是 **prompt prefill** 的 FLOPs, 只算一次 (因为 prompt 跨 sample 是 shared 的)
- 第二项是 **decode** 的 FLOPs, 每个 sample 都要单独 decode, 所以乘 $\text{NumCompletions}$
- $\text{FLOPsPerToken}(\cdot)$ 是关于 position 的, 因为 KV cache 增长, attention 的 FLOPs 随 token position 增长 (O(t) per attention layer)
- 这个公式假设 dense transformer, MoE 模型需要修正

**Intuition**: 这个公式其实埋了一个非常重要的 system-level insight — 因为 prompt prefill 只算一次, 当 NumCompletions 很大时, **prompt prefill 的成本被摊薄**。这一点在 vLLM、SGLang (https://arxiv.org/abs/2312.07166)、Hydragen (https://arxiv.org/abs/2402.05099, 作者 Juravsky 自己上一篇 paper) 这些 shared-prefix attention 优化下, 实际 wall-clock cost 远低于 naive 估算。论文 Section 5 末尾专门提了这点, 但没量化。

### 4.2 Figure 4 的核心发现

当 total inference FLOPs budget 固定时:
- **MiniF2F, GSM8K, MATH**: 8B + many samples > 70B + fewer samples。换句话说, 弱模型 + 多 sample 是 Pareto-optimal。
- **CodeContests**: 70B + fewer samples > 8B + many samples。

为什么 CodeContests 反过来? 我推测是: competitive programming 的 solution space 太大, 弱模型 (8B) 在 sample 10000 次后已经基本 saturation (它的 "reachable" solution pool 已经被反复采样完), 而 70B 模型每个 sample 的 marginal quality 更高。这暗示 **task 的 solution-space entropy 决定了 optimal sample 数**。

### 4.3 Table 1: SWE-bench Lite 的 API dollar cost

| Model | Cost/attempt | #attempts | Solved | Total cost |
|---|---|---|---|---|
| DeepSeek-Coder-V2-Instruct | $0.0072 | 5 | 29.62% | $10.8 |
| GPT-4o | $0.13 | 1 | 24.00% | $39 |
| Claude 3.5 Sonnet | $0.17 | 1 | 26.70% | $51 |

这里 DeepSeek-V2 (MoE, https://arxiv.org/abs/2405.04434) 单价是 GPT-4o 的 ~1/18, 而在 Moatless Tools framework 下 5 个 samples 解决的 issue 比 GPT-4o 单 sample 多 5.62 个百分点, 比 Claude 多 2.92 个百分点, 总成本是 1/3.6 ~ 1/4.7。

**Punchline**: 这张表本质上是在说 API pricing 时代的 "intelligence per dollar" 优化目标。DeepSeek 的 MoE 架构让 active parameters 少, 所以 inference 便宜; 配合 repeated sampling, 弱但便宜的 model 可以 dominate 强但贵的 model。这点对 product 团队非常有操作意义。

---

## 5. Scaling law for coverage (Section 3)

### 5.1 Exponentiated power law

GPT-4 tech report (https://arxiv.org/abs/2303.08774) 在 training compute vs mean-log-pass-rate 上发现 power law。本文借用这个 functional form, 但换成 inference:

$$
\log(c) \approx a \cdot k^b
$$

$$
c \approx \exp(a \cdot k^b)
$$

变量解读:
- $c$: coverage, $\in (0, 1)$
- $k$: sample budget (positive integer)
- $a \in \mathbb{R}$: 控制 base-level coverage, 当 $k=1$ 时 $c \approx e^a$, 所以 $a \approx \log(\text{pass@1})$
- $b \in \mathbb{R}$: 控制 scaling 的 curvature
  - $b = 1$ 时, $\log c$ 关于 $k$ 线性, $c$ 关于 $k$ 指数增长 — 这是 "free lunch" 但很快 saturate
  - $b < 1$ 时, $\log c$ 关于 $k$ concave (sublinear), 这是 paper 里大部分 fit 的 regime
  - $b \to 0$ 时退化成 $c \approx e^a$, 即 scaling 完全 stall

**Intuition**: 为什么这个 form 而不是 $c = 1 - (1-p)^k$? 后者是 i.i.d. Bernoulli 假设下的精确解。但实际 sample 不是 i.i.d. — 同一 prompt + temperature 下, sample 之间有相关性 (modes、clusters); 且每个 problem 的难度是 heterogeneous 的。Exponentiated power law 是经验 fit, 不源自 first principle, 但能 capture "long-tail" 的 hard problem: easy problem 在 $k$ 小时就 solved, hard problem 需要 $k$ 大时才偶尔被 hit。

### 5.2 Fit quality

paper 在 Figure 5 / Appendix C.2 给了大量 fit。**MiniF2F-MATH (Llama-3-8B) fit 较差**, 其他 dataset 都不错。Llama-3-8B 在 MiniF2F 上 coverage 曲线有明显的 "elbow" — 早期增长慢, 中期加速, 后期再 saturation。这种 S-curve 用单 exponentiated power law 是 capture 不到的。

### 5.3 Figure 6: 同 family 模型的 coverage curve 有平行偏移

这是 paper 里一个我觉得被低估的观察。对于同 family 不同 size 的模型 (e.g. Llama-3 8B vs 70B), 把 coverage 曲线沿 log-x 轴平移到都穿过 $(1, c)$ 这个 anchor point (c = max pass@1), 它们的 shape **几乎重合**。

含义: 在同 family 内, **scale model size 等价于 horizontal shift on log-sample-axis**。也就是说, "用 2× 大的模型" 和 "用 ~10× 多的 sample" 在某些区间可以达到同样的 coverage 增益。

这立刻给出一个 scaling law 式的 budget allocation 公式: 给定固定 FLOPs, 当 model size 增长 $\Delta$ FLOPs/sample 时, coverage 应该 shift 多少? 如果 model size 和 sample 数在 log-x 上是 "等价维度", 那么 optimal allocation 取决于两边 cost 的 elasticity。

**Intuition 联想**: 这跟 Kaplan et al. (https://arxiv.org/abs/2001.08361) 的 training scaling law 里 "scale width vs depth vs data" 的等价置换非常像。这里相当于 inference 版本: "scale params vs scale samples" 的等价置换。Hoffmann et al. Chinchilla paper (https://arxiv.org/abs/2203.15556) 修正了 Kaplan 的 optimal compute allocation; 我觉得这篇 paper 之后会有 follow-up 修正这里的 sample-vs-size allocation。Hassid et al. (https://arxiv.org/abs/2404.00725) 实际上已经做了类似 analysis for code, 并发工作。

---

## 6. Precision 问题 (Section 4) — 这才是 paper 的真正深度

### 6.1 Verification 方法的 plateau

Figure 7 是 paper 最 informative 的一张图。在 MATH + Llama-3-8B-Instruct 上:

- Coverage: 100 samples → 82.9%, 10000 samples → 98.44% (持续增长)
- Majority Vote: 100 samples → ~40%, 10000 samples → ~41% (几乎 plateau)
- Reward Model + Best-of-N: 类似 plateau
- Reward Model + Majority Vote: 类似 plateau

**Gap = 57 percentage points** 在 10000 samples 时。这是 "needle in the haystack" 问题的具体 manifestation。

为什么 majority voting 必然 plateau? 直观论证: 假设 model 对某 hard problem 的 correct probability 是 $p = 0.01$。100 个 sample 里, 期望有 1 个 correct。但 wrong answer 的 distribution 是 dispersed 的 — 100 个 sample 可能有 50 种不同的 wrong answer。所以 plurality winner 仍是某个 wrong answer (它出现 2-3 次, 而 correct 只出现 1 次)。只有当 $p \cdot k \gg 1$ 且 wrong answers 分散时 majority vote 才会胜出。但 $p \cdot k$ 增长意味着 correct 已经不是 "rare" 了, 此时 coverage 已经 saturate。

Reward model (ArmoRM-Llama3-8B-v0.1, https://arxiv.org/abs/2406.12845) 没好多少, 说明 **current reward models 的 calibration 在 rare-correct-sample regime 下严重不足**。Reward model 训练时见到的 "correct reasoning" 大多是 high-confidence case, OOD 到 rare correct reasoning 上 signal 弱。

### 6.2 Table 2: CoT 是 faithful 的

这是 paper 一个 supporting investigation: 也许 majority vote 之所以 fail 是因为 correct sample 的 CoT 其实是 non-sensical (model 在乱猜), 那 verifier 就根本没 signal 可用。

作者手动标注了 105 个 Llama-3-8B-Instruct 在 GSM8K 上的 CoT (3 个 per problem), 结果:
- pass@1 ∈ [0%, 10%] 的 hard problem: 11/15 = 73% 的 correct CoT 是 logically faithful
- pass@1 ∈ [75%, 100%] 的 easy problem: 30/30 = 100% faithful

所以 **correct sample 的 CoT 大多是真的有推理的**, verifier (reward model) 理论上应该能 exploit, 但实际上做不到。这指向一个明确的 research direction: **verifier 需要专门在 low-pass-rate region 上训练**, 类似 process reward model (PRM, Lightman et al. https://arxiv.org/abs/2305.20050) 但需要更 aggressive 的 hard-example mining。

### 6.3 Figure 8 的 bar chart

这张图我特别想 highlight。每个 bar 代表一个 problem, bar 的高度 = 10000 samples 里 correct 的 fraction。bar 颜色: 绿 = self-consistency 选对, 红 = 选错。

关键观察: **大量 correct fraction < 1% 的 problem 是红色的**。这些 problem 的 correct sample 确实存在 (所以 coverage 计进去), 但 majority vote / RM 选不出来。Figure 7 里 coverage 和 verification method 的 gap, 在这张图上可视化得很清楚。

**Intuition 联想**: 这跟 "long-tail" learning 的 connection 很强。Easy problem (高 p) 已经 solved, 剩下的都是 $p \ll 1$ 的 hard problem。要让 coverage 在 hard problem 上转化为 success rate, verifier 必须能在 100 个 wrong 里识别 1 个 right。这本质上是个 **recall@k 问题**, 不是 ranking 问题。

---

## 7. 两个 cautionary tales (Section 4.2)

### 7.1 SWE-bench Lite 的 flaky tests (Appendix B)

11.3% 的 SWE-bench Lite 问题有 flaky tests — 同一 candidate solution 跑多次, 有时 pass 有时 fail。其中 30/34 即使是 gold solution 也会 flaky。原因是:
- Python `set` 的 unordered 性: 如果 candidate solution 不强制排序, test 可能 fail 但其实语义正确
- 未 pin 的 dependency 版本 (astropy 的 case)

paper 处理方式: 对 flaky problem 跑 11 次 test, majority vote 决定 pass/fail。Appendix B Table 3 列出了所有 34 个 problem instance ID。

**Implication**: SWE-bench Lite 的真实 noise level 比想象的更高。后续做 SWE-bench 工作的同行 (e.g. SWE-agent follow-ups) 应该把这个 caveat 纳入 measurement noise 估计。这点之前 Princeton 的 SWE-bench 原 paper (https://arxiv.org/abs/2310.06770) 没明确强调。

### 7.2 CodeContests 的 false negatives

35/122 的 Python3 problem, **官方给出的 reference solution 跑官方给出的 test cases 会 fail**。原因:
- Problem 允许多个 correct output, 但 test 只接受一个
- 程序生成的 mutated test cases 违反 input spec (e.g. 应该正整数, 但 mutated 成 0)

**Implication**: CodeContests 上的 pass@k 数字有 ceiling < 100%, 因为即使是 perfect model 也无法 100% pass。这对 AlphaCode 当年的数字也适用 — AlphaCode 论文 (https://www.science.org/doi/10.1126/science.abq1158) 报告的 14% 是在这个 noisy verifier 下的, 实际 capability 可能更高。

---

## 8. 想到的一些延伸联想 (build your intuition)

### 8.1 这篇和 AlphaCode 的关系

AlphaCode 是 inference scaling 的 " existence proof" — 1M samples + filtering 让 model 在 CodeContests 上达到人类水平。但这篇 paper 把 AlphaCode 的方法论 abstract 成了 **"coverage × precision" framework**, 并且把这个 framework 推广到 non-code domain。AlphaCode 用了一个 "cluster-then-vote" 的 trick (把 sample 按输出聚类, 然后在最大 cluster 里挑), 这个 trick 在 math word problem 上对应的就是 majority voting, 但因为 math answer space 太大 (不像 code 输出可聚类), voting 直接退化。

AlphaCode paper: https://www.science.org/doi/10.1126/science.abq1158

### 8.2 和 o1 / test-time compute scaling 的关系

o1 的思路是 **单 sample + 长 CoT**, 通过 RL-trained "thinking" 来 scale per-sample compute。这篇 paper 的思路是 **多 sample + short CoT**。两者是正交的: 理论上可以 multi-sample × long-CoT。o1 在 MATH 上的 success rate 接近 90%+ (single sample), 这等价于这篇 paper 里 Llama-3-8B 在 1000+ samples 下的 coverage。所以 o1 大致是把 "sampling scaling" 压缩到了 "single-sample long reasoning" 里。从 inference FLOPs 角度, o1 的 token efficiency 应该比 repeated sampling 高, 因为它不重复 prefill。

### 8.3 Power law 拟合的脆弱性

$c \approx \exp(a k^b)$ 这个 form 在 $b$ 的拟合上很 sensitive。我估算了一下: 如果 $b = 0.5$, 则 $\log c \propto \sqrt{k}$, 这对应 "each 4× samples gives 2× log-coverage gain"。如果 $b = 1$, 则 linear in $k$, 即每多 1 sample, $\log c$ 涨 $a$。Paper 里没给 fit 出来的 $b$ 数值表 (Appendix C.2 有图但没数), 这是个遗憾, 不然可以比较不同 task 的 $b$ 来判断 "sample 效率"。

我推测 MATH 的 $b$ 接近 0.5-0.7, CodeContests 的 $b$ 更接近 0.4 (sample 效率较低), MiniF2F 因为是 S-curve 所以 fit 不好。

### 8.4 Connection to "scaling inference compute" (Snell et al., 2024)

Berkeley/DeepMind 的 Snell et al. 在同期 arXiv (https://arxiv.org/abs/2408.03314) 做了非常 complementary 的工作: 他们 focus 在 "best-of-N with process reward model" 和 "rejection sampling" 上, 在 MATH 上做了类似 sweep。两边结论 converge 在 "100-1000 samples 是 sweet spot"。但 Snell 那篇没做 SWE-bench, 也没做 power law fit。两篇一起读会得到比较完整的 inference scaling 图景。

### 8.5 Multi-sample 范式下的 systems 工作

Section 5 提到的一个 underexplored 方向: 当 sample 数到 10k+, 系统层优化空间巨大。相关 systems paper:
- Hydragen (https://arxiv.org/abs/2402.05099): shared-prefix attention, 同 prompt 多 sample 的 attention 计算可以 batch
- SGLang (https://arxiv.org/abs/2312.07166): structured generation, 减少 sample 浪费在 syntactically invalid 的输出上
- Bifurcated Attention (Athiwaratkun et al., https://arxiv.org/abs/2403.08845): AWS 的 parallel decoding 方案

如果这些优化叠加, repeated sampling 的实际 wall-clock cost 可能比 paper 用 FLOPs 估算的低 10×, 这进一步强化了 "weak model + many samples" 的 cost advantage。

### 8.6 反思: 为什么 math 上 verifier 比 code 上难这么多

paper 用 Lean4 做 MiniF2F 上的 verifier 是 deterministic 的。但 Lean4 formalization 本身 hard — 把 informal MATH problem 翻译成 Lean statement 是个 bottleneck。所以 "verifier 变强" 跟 "任务 formalizable" 是 coupled 的。

一个 research 方向是 **autoformalization** (Wu et al., ICLR 2022, https://arxiv.org/abs/2205.12615): 用 LLM 把 informal math 翻译成 Lean, 然后用 Lean verifier。这等价于 paper Section 5 末尾提到的 "converters"。这条路如果 work, math 也能享受到 code 那种 perfect verifier 的红利。

### 8.7 一个我没见人讨论的角度: sample diversity

paper 里所有 sample 都是 temperature sampling, i.i.d.。但 i.i.d. sampling 倾向于在 high-probability mode 周围重复采样。AlphaCode 用了 metadata conditioning (problem tags) 来强制 diversity, 这等价于把 sampling 过程从一个 mode 拆成多 mode, 提高 coverage。

理论分析: 如果 model 的真分布是 $p(x)$, i.i.d. sample 的 coverage 是 $1 - (1 - p_{\text{correct}})^k$。如果用 conditional sampling $p(x | z)$ 其中 $z$ 是 diversity cue, 那不同 $z$ 下的 $p_{\text{correct}}$ 可能互补 — 比如 $z=1$ 让某个 problem 的 correct prob 从 0.001 变成 0.1, 那 coverage 增长会快得多。

这种 "intentional diversity" 没在这篇 paper 探索, 是个明显的 follow-up 方向。

### 8.8 一个 brainstorming 方向: 把 repeated sampling 当成 "implicit ensembling"

10000 个 i.i.d. sample 加上 oracle verifier 等价于一个 huge ensemble, 每个 ensemble member 是 model 在某 temperature 下的一个 instantiation。这个 view 下, "weak model + 10k samples" 的 coverage 56% on SWE-bench Lite 等价于说 "10000 个 weak model 的 ensemble > 1 个 GPT-4o"。这跟 traditional ML 里的 random forest vs deep network 的 trade-off 类似。Paper 没明说这个 connection, 但这个 framing 可以 borrow random forest 的 intuition 来理解为什么 power law fit 工作。

---

## 9. Limitations (我从 paper 里挖出来的)

1. **只有 single-turn i.i.d.**: 真实推理系统里 multi-turn refinement (Self-Refine, https://arxiv.org/abs/2303.17651) 可能比纯 repeated sampling 更 sample-efficient, 但没做对比。

2. **Coverage 是 oracle 上限**: paper 反复强调 coverage 是 "if you have perfect verifier" 的上限。对于 math word problem, 真实 deployable 性能远低于 coverage。但论文标题/abstract 里 headline 数字是 coverage, 容易被误读。

3. **No formal scaling law theory**: $c \approx \exp(a k^b)$ 是纯 empirical fit, 没有 theoretical derivation 从 model distribution 推出来。Snell et al. 同期工作也没做这个。

4. **Temperature 是 hyperparameter**: SWE-bench 上他们 sweep 了 T ∈ {1.0, 1.4, 1.6, 1.8}, 选了 1.6。但 temperature 对 coverage 曲线 shape 的影响没系统分析。直觉上, 太低 temperature 让 sample 都聚在 mode 上, coverage 增长慢; 太高 temperature 让 sample 都 noisy, 单 sample quality 下降。可能有 task-specific optimal T。

5. **同 family 假设**: Figure 6 的 "curve 平行偏移" 只在 Llama-3 family 内验证。跨 family (Llama vs Gemma) 是不是也成立不知道。如果跨 family 也成立, 就能 cross-model budget transfer; 如果不成立, 说明 family-specific。

6. **Reward model 是 frozen 的**: 用 ArmoRM-Llama3-8B-v0.1 没在 task-specific 上 finetune。可能 task-specific RM 在 hard problem 上能 break plateau, 但 paper 没试。

---

## 10. 最值得 takeaway 的几点

1. **Inference-time scaling law 是真的存在**, 形式类似 training scaling law, 但 functional form 不同 (exponentiated power law vs pure power law)。

2. **"Amplify 弱模型" 在有 verifier 的 domain 上是 free lunch**: SWE-bench Lite 上 open-source 8B-class 模型 + 250 samples 打败 GPT-4o + Claude 3.5 Sonnet single sample 的 SOTA。这是 product-level actionable 的结论。

3. **Verification 是 inference scaling 的真正 bottleneck**, 不是 sampling 本身。Math word problem 上 self-consistency / RM 在 100 samples 后 plateau 是 paper 最 actionable 的 finding — 指向 process reward model、autoformalization、verifier training 等 direction。

4. **Cost-performance optimal 点位 task-dependent**: easy task (GSM8K, MATH) 上 "small model × many samples" Pareto-dominate "big model × few samples"; hard task (CodeContests) 反过来。这不是 universal 结论, 是 task-specific 测量结果。

5. **Verifier 不完美这件事被低估了**: SWE-bench Lite 11.3% flaky, CodeContests 35/122 false-negative。所有"X% on benchmark"的数字都有 verifier noise, 报道时应该有 confidence interval。

References for further reading:
- Snell et al., "Scaling test-time compute": https://arxiv.org/abs/2408.03314
- AlphaCode: https://www.science.org/doi/10.1126/science.abq1158
- Self-Consistency: https://arxiv.org/abs/2203.11171
- Process Reward Models (Lightman et al.): https://arxiv.org/abs/2305.20050
- Codex (pass@k definition): https://arxiv.org/abs/2107.03374
- SWE-bench: https://arxiv.org/abs/2310.06770
- GPT-4 tech report (training scaling law for pass-rate): https://arxiv.org/abs/2303.08774
- Hassid et al. (concurrent work, budget reallocation): https://arxiv.org/abs/2404.00725
- DeepSeek-Coder-V2: https://arxiv.org/abs/2405.04434
- Hydragen (systems for shared prefix): https://arxiv.org/abs/2402.05099
- SGLang: https://arxiv.org/abs/2312.07166
- Autoformalization (Wu et al.): https://arxiv.org/abs/2205.12615
- Kaplan et al. training scaling laws: https://arxiv.org/abs/2001.08361
- Chinchilla (Hoffmann et al.): https://arxiv.org/abs/2203.15556
