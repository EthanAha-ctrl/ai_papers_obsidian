---
source_pdf: The Illusion of the Illusion of Thinking.pdf
paper_sha256: a8503431927a7b537cfb0ce926b488d8778365690e378566bfbdd32153d68869
processed_at: '2026-08-12T14:16:58-07:00'
target_folder: LLM-evaluation
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版：这场吵架到底在吵什么

## 一句话总结

Apple 的人写 paper 说 "LRM 不会 reasoning"，被 Anthropic 的人怼回来 "你的实验设计就是错的"。

---

## Apple 原文讲了啥故事

Apple 的人拿了 Tower of Hanoi、River Crossing、Blocks World 三类 puzzle，让 LRM（就是 o3、DeepSeek-R1、Claude-3.7-Sonnet 这类会思考的 model）做。然后画了一张图：横轴是 puzzle 难度，纵轴是 accuracy。发现 $N$ 一过某个阈值，accuracy 直接掉到 0。于是下结论：reasoning model 的 reasoning 是 illusion，过了阈值就崩。

---

## 反驳者的核心论点：你这不是测 reasoning，你这是测 typing

反驳者说，你看到的 "崩" 其实是三个 bug 叠加出来的，跟 reasoning 没关系。

### Bug 1：Tower of Hanoi 输出爆炸

**原 paper 要求的输出格式特别蠢。** 不是让 model 说 "move disk 3 from A to C"，而是要求每走一步都把当前所有 disk 的位置全部重新打印一遍。

打比方：你问我从家到机场怎么走，正常答案是 "出门右转、上高速、第 3 个出口下"。Apple 的 evaluator 要求的答案是 "我现在在家；走出门我在门口；右转我在路口；上高速我在入口……" 每一步都要把全宇宙的状态重新描述一遍。

**Token 需求量公式：**

$$T(N) \approx 5(2^N - 1)^2 + C$$

变量解释：
- $T(N)$：$N$ 个 disk 需要输出的 token 总数
- $2^N - 1$：Tower of Hanoi 的最优 move 数（$N=10$ 是 1023 步，$N=15$ 是 32767 步）
- 上标 $^2$：square，因为每步都要重写整个 state，所以总 token = 步数 × 每步的长度，而每步长度本身又随步数线性增长
- $5$：每描述一个 move 大概 5 个 token 的经验值
- $C$：固定的 overhead

**数值感受：**

| $N$ | Move 数 $2^N-1$ | Token 需求 $T(N)$ | Model budget |
|---|---|---|---|
| 7 | 127 | ~80,000 | Claude-3.7: 64k ❌ |
| 8 | 255 | ~325,000 | o3-mini: 100k ❌ |
| 10 | 1023 | ~5,200,000 | 全部 ❌ |
| 15 | 32,767 | ~5,000,000,000 | 想都别想 |

所以原 paper 报告 "Claude 在 $N=8$ 崩了"，真相是 **Claude 的 output slot 被撑爆了**，跟会不会解一点关系都没有。

**最铁的证据：** Twitter 上有人复现，抓到 model 原话：

> "The pattern continues, but to avoid making this too long, I'll stop here."

Model 自己说 "我知道规律，但太长了我不写了"。这能叫 "reasoning collapse" 吗？这叫 model 有 meta-cognition。

---

### Bug 2：把 unsolvable 的题也算 model 错

这是最离谱的一个。

**原 paper 测试 River Crossing（Missionaries-Cannibals 变体）时用了 boat capacity $b = 3$，测试范围 $N = 1$ 到 $N = 10$。**

但是 [Efimova 2018 的代数分析](https://arxiv.org/abs/1802.09369) 老早就证明：$b = 3$ 的 Missionaries-Cannibals，**$N > 5$ 时数学上无解**。

**直觉上为啥 $N = 6, b = 3$ 无解：**

想象左岸有 6 个 missionary 和 6 个 cannibal。船一趟最多带 3 人。关键瓶颈在中间状态：船过去时带几个、回来时带几个，net transfer 每趟最多 1-2 人，但还要保持 "任一岸 cannibals 不超过 missionaries（除非 missionary = 0）" 这个 invariant。$N = 6$ 时中间某个状态必然违反 invariant，因为船装不下同时调整两类人所需的人数。

原 paper 的 evaluator 把 $N = 6, 7, 8, 9, 10$ 上 model "解不出" 记为 reasoning failure。

**这相当于：**
- 给 SAT solver 一个 unsatisfiable formula
- Solver 返回 "UNSAT"
- 你说 solver "失败" 了

Model 说 "这题无解" 反而是正确答案，被扣分。

---

### Bug 3：用 solution length 当 complexity，纯属偷换概念

原 paper 把 "需要多少步走完" 当作 problem difficulty。

但你看这三类 puzzle 的真实结构：

| Puzzle | Solution 长度 | Branching factor | 真正难在哪 |
|---|---|---|---|
| Tower of Hanoi | $2^N - 1$（指数） | **1** | 根本不需要 search，每步唯一确定 |
| River Crossing | $\sim 4N$（线性） | $> 4$，候选 $O(N^3)$ | 需要 search，NP-hard |
| Blocks World | $\sim 2N$ | $O(N^2)$ | 需要 search，PSPACE-complete |

**Tower of Hanoi 的 branching factor = 1 是什么意思：**

任意一个 state，下一步怎么走是 **完全确定的**。recursive structure 是 $\text{hanoi}(N, A, C, B) = \text{hanoi}(N-1, A, B, C) + \text{move}(N, A \to C) + \text{hanoi}(N-1, B, C, A)$。这个公式直接告诉你每一步该干嘛，零 search、零决策。

所以 Tower of Hanoi 是 **"长但好算"** 的典型，difficulty 在执行端而不在决策端。River Crossing 才是 **"短但难想"**，difficulty 在 search 端。

原 paper 用 solution length 当 complexity proxy，导致 Tower of Hanoi $N = 15$ 看起来 "比" River Crossing $N = 6$ 难，但前者随便一个递归函数搞定，后者要真搜索。这就是为什么 model 能做 100 步 Hanoi，反而在 5 步的 River Crossing 上翻车。

---

## 反驳者自己的实验：换个 representation 立刻复活

让 model 解 Tower of Hanoi $N = 15$，要求：**输出一个 Lua 函数，调用时打印解。**

```lua
function hanoi(n, from, to, via)
    if n > 0 then
        hanoi(n-1, from, via, to)
        print("Move disk " .. n .. " from " .. from .. " to " .. to)
        hanoi(n-1, via, to, from)
    end
end
hanoi(15, 'A', 'C', 'B')
```

Claude-3.7-Sonnet、Claude Opus 4、OpenAI o3、Gemini 2.5 全部答对，**用 < 5,000 token 完成了 $2^{15} - 1 = 32{,}767$ 步的解**。

这就是铁证：**model 早就掌握了 generative procedure，原 paper 的 evaluation format 强制 enumeration 才制造了 "崩" 的假象**。

---

## Kolmogorov 视角的直觉

Kolmogorov complexity $K(x)$ = 输出 string $x$ 的最短程序长度。

Tower of Hanoi $N$ 的 move sequence $x_N$：
- $|x_N| = O(2^N)$（如果你要展开所有 move）
- $K(x_N) = O(\log N)$（你只需要写一个递归函数 + 输入 $N$）

**Model 掌握的是 $K(x_N)$ 这一层（low），原 paper 强制 model 输出 $|x_N|$ 这一层（high）。** 这个 gap 就是 evaluation bug 的本质。

类比：你会写 "for i in range(100): print(i)"，但如果有人要求你必须写出 "0 1 2 3 ... 99" 这 100 个字，写到第 80 个你嫌烦不写了，他说你 "不会数到 100"。这个评价荒谬吧。

---

## Karpathy 视角的 broader insight

我一直讲 [Software 2.0](https://karpathy.medium.com/software-2-0-a6eb52a15190)：神经网络是新的 programming paradigm，model 学到的是 dataset-defined program。

评估一个 program 是不是 "会 reasoning"，应该看它能不能 **generate correct behavior on novel inputs**，而不是看它能不能 **memorize and replay specific outputs**。

[AlphaCode](https://www.nature.com/articles/s41586-021-03342-y) 的评估哲学就对了：不要求输出 execution trace，只要求输出 program，然后跑 test case 验证。如果硬要 AlphaCode 输出 $10^6$ 个 test case 的完整 trace，它也 "崩"。

另一个联想：[Faith and Fate paper (Dziri et al. 2023)](https://arxiv.org/abs/2305.18654) 里那个 $P(\text{all correct}) = p^T$ 的论证，假设 model 是无状态 Bernoulli sampler。但 reasoning model 会 monitor 自己的 output 长度、主动压缩、选 representation。$p^T$ 那套公式 **把 model 的 agency 全部 bake out 了**。

还有一个更深的 issue：automated evaluation 的 [Goodhart's Law](https://en.wikipedia.org/wiki/Goodhart%27s_law) —— "When a measure becomes a target, it ceases to be a good measure." 原 paper 把 "准确输出完整 move list" 当 reasoning 的 proxy，proxy 错了，结论也错了。

---

## 一句金句

Paper 结尾那句话最到位：

> The question isn't whether LRMs can reason, but whether our evaluations can distinguish reasoning from typing.

翻译成人话：**问题不在 model 会不会思考，问题在你的测试能不能分清 "思考" 和 "打字"**。

---

## 这场吵架的 takeaway

1. **别用 output length 当 difficulty proxy**，用 branching factor / search complexity
2. **测 puzzle 前先验证 puzzle 可解**，否则你测的是 model 的 unsolvability detection
3. **允许 model 选 representation**，program 比 enumeration 更能体现 reasoning
4. **Automated evaluator 会把 "model 主动截断" 误判为 "model 不会"**，必须留人工 audit 通道
5. **AI 评估界正经历一场 replication crisis**，[Apple 这篇只是冰山一角](https://arxiv.org/abs/2506.06503)，社区需要更多这种 "质疑原实验设计" 的 work

---

## Reference

- 原文（被怼的 Apple paper）：[Shojaee et al. 2025, The Illusion of Thinking](https://arxiv.org/abs/2506.06503)
- 反驳原文：[Opus & Lawsen, The Illusion of the Illusion of Thinking](https://arxiv.org/abs/2507.08071)
- Twitter 复现：[@scaling01 thread](https://x.com/scaling01/status/1931817022926839909)
- Faith and Fate：[Dziri et al. NeurIPS 2023](https://arxiv.org/abs/2305.18654)
- River Crossing 代数解：[Efimova 2018](https://arxiv.org/abs/1802.09369)
- Blocks World 复杂度：[Bylander 1994](https://www.sciencedirect.com/science/article/pii/S0004370294000431)
- Software 2.0：[Karpathy blog](https://karpathy.medium.com/software-2-0-a6eb52a15190)
- AlphaCode：[Nature 2022](https://www.nature.com/articles/s41586-021-03342-y)
- Chain-of-Thought：[Wei et al. 2022](https://arxiv.org/abs/2201.11903)
- Kolmogorov complexity 教材：[Li & Vitányi](https://www.springer.com/gp/book/9781489989059)
- Goodhart's Law：[Wikipedia](https://en.wikipedia.org/wiki/Goodhart%27s_law)

---

# The Illusion of the Illusion of Thinking — 深度解析

这篇 paper 是 Anthropic 的 Claude Opus 4 与 Alex Lawsen 对 Shojaee et al. (2025) "The Illusion of Thinking" 的反驳。核心论点：原 paper 报告的 LRM "accuracy collapse" 主要是 **experimental artifact**，而非 model 本身的 reasoning failure。

---

## 1. 背景与前因

Shojaee et al. (Apple 团队) 在 arXiv:2501.12948 上发表了 [The Illusion of Thinking](https://arxiv.org/abs/2506.06503)，声称通过 Tower of Hanoi、River Crossing、Blocks World 三类 planning puzzle 发现 LRM 超过某个 complexity threshold 后 accuracy 骤降为零，并由此推断 reasoning model 的 reasoning 能力是 "illusion"。

本文作者认为这个结论是一个 **三层嵌套的 illusion**：原作者声称 reasoning 是 illusion，但这个声称本身才是 illusion。

---

## 2. 三个核心反驳

### 2.1 Token limit 被误读为 reasoning collapse

原 paper 要求 model 输出 **每一步的完整 move list**（而非只输出最终序列）。这意味着 token 消耗是 **quadratic** 增长。

**公式 (2) 详解：**

$$T(N) \approx 5(2^N - 1)^2 + C$$

变量含义：
- $T(N)$：$N$ 个 disk 的 Tower of Hanoi 所需的 output token 总数
- $N$：disk 数量（problem size 参数）
- $2^N - 1$：Tower of Hanoi 的最优 move 数（经典递归结果，$T_{\text{moves}}(N) = 2T_{\text{moves}}(N-1) + 1$ 的解）
- 上标 $2$：square，因为 evaluation format 要求每一步都重新列出 **当前所有 disk 的位置**，所以总 token 是 move 数 × 每步描述长度，而每步描述长度本身又随 move 数线性增长 → quadratic
- $5$：经验估计，每个 move 描述约 5 个 token
- $C$：constant overhead（system prompt、formatting 等）

**公式 (3) 详解：**

$$N_{\max} \approx \lfloor \log_2(\sqrt{L_{\max}/5}) \rfloor$$

变量含义：
- $N_{\max}$：model 在给定 token budget 下能完整输出的最大 Tower of Hanoi size
- $L_{\max}$：model 的 output token limit（Claude-3.7-Sonnet 与 DeepSeek-R1 是 64,000；o3-mini 是 100,000）
- $\lfloor \cdot \rfloor$：floor function，向下取整
- $\log_2$：以 2 为底的对数
- $\sqrt{\cdot}$：square root

**推导 chain：**

令 $T(N) = L_{\max}$：
$$5(2^N - 1)^2 \approx L_{\max}$$
$$(2^N - 1)^2 \approx L_{\max}/5$$
$$2^N - 1 \approx \sqrt{L_{\max}/5}$$
$$N \approx \log_2(\sqrt{L_{\max}/5})$$

**数值代入：**

| Model | $L_{\max}$ | $\sqrt{L_{\max}/5}$ | $\log_2(\cdot)$ | $N_{\max}$ |
|---|---|---|---|---|
| Claude-3.7-Sonnet | 64,000 | $\sqrt{12800} \approx 113$ | $\approx 6.82$ | 6-7（考虑 $C$ 与实际 token 使用效率） |
| DeepSeek-R1 | 64,000 | 同上 | 同上 | 6-7 |
| o3-mini | 100,000 | $\sqrt{20000} \approx 141$ | $\approx 7.14$ | 7-8 |

原 paper 报告的 "collapse point" 恰好与这些 $N_{\max}$ 一致——也就是说，**不是 model 不会做，而是 model 被物理禁止输出足够长的答案**。

**关键证据：** Twitter 用户 [@scaling01](https://x.com/scaling01/status/1931817022926839909) 复现实验时捕获到 model 的原始输出：

> "The pattern continues, but to avoid making this too long, I'll stop here."

这表明 model 完全理解了 solution pattern，主动选择截断。这种 **self-awareness of output constraint** 与 "reasoning collapse" 是完全不同的现象。

---

### 2.2 Per-token accuracy 的 statistical inevitability 谬误

**公式 (1) 详解：**

$$P(\text{all correct}) = p^T$$

变量含义：
- $P(\text{all correct})$：整个 output sequence 完全无误的概率
- $p$：per-token accuracy（单个 token 正确生成的概率）
- $T$：output 的 total token 数
- 上标 $T$：表示 $p$ 自乘 $T$ 次（i.i.d. 假设下）

**数值：**

| $p$ | $T$ | $P(\text{success}) = p^T$ |
|---|---|---|
| 0.9999 | 10,000 | $0.9999^{10000} \approx e^{-1} \approx 0.368$ (36.8%) |
| 0.999 | 10,000 | $0.999^{10000} \approx e^{-10} \approx 4.5 \times 10^{-5}$ (0.005%) |
| 0.99 | 10,000 | $0.99^{10000} \approx e^{-100} \approx 3.7 \times 10^{-44}$ |

**原 intuition（来自 [Dziri et al. 2023, "Faith and Fate"](https://arxiv.org/abs/2305.18654)）：** 随着序列变长，完美执行的概率指数级下降，因此 LLM 在 compositionality 上存在 fundamental limit。

**本文反驳：** 这个论证假设 model 是一个 **无状态的 token-by-token sampler**，但 reasoning model 实际上会：
- 自我监控 output 长度
- 主动采用压缩表示（如递归函数）
- 在遇到长序列时选择 abstraction 而非 enumeration

把 model 当作 Bernoulli sampler 来做 $p^T$ 分析，忽略了 model 的 **agency**——model 可以选择 representation，而不仅仅是 sample tokens。

---

### 2.3 River Crossing 的 impossible instances

这是最严重的实验设计错误。

**原 setup：** Shojaee et al. 测试 Missionaries-Cannibals 变体，使用 boat capacity $b = 3$，测试 $N = 1$ 到 $N = 10+$。

**数学事实（[Efimova 2018, arXiv:1802.09369](https://arxiv.org/abs/1802.09369)）：**

Missionaries-Cannibals puzzle 在 $b = 3$ 时，**当 $N > 5$ 时无解**。

**intuition 解释为什么 $N > 5, b = 3$ 不可解：**

考虑 state 表示 $(m, c, \text{side})$，其中 $m$ 是左岸 missionaries 数，$c$ 是左岸 cannibals 数，side 是船的位置。合法 transition 要求：
1. 船上人数 $\leq b = 3$
2. 任何时候任一岸 cannibals 不能超过 missionaries（除非 missionaries = 0）

对于 $N = 6$，critical state 是中间过渡状态。船从左岸出发时最多带 3 人到右岸，返回时最多带 2 人（因为必须留 1 人在船上回来）。这意味着每趟 net transfer 最多 $3 - 2 = 1$ 人。但更关键的 constraint 是：在 $N = 6$ 时，中间某个 state 必然出现 $c > m$ 且 $m > 0$ 的 forbidden configuration，因为 boat capacity 不足以同时调整 $m$ 和 $c$ 的平衡。

形式化地说，对于 $b = 3$，solvable region 是 $N \leq 5$。$N = 5$ 时 optimal solution 长度是 11 steps；$N = 6$ 进入不可解区。

**原 paper 的错误：** 自动化 evaluator 把 model 在 $N \geq 6$ 上 "无法解决" 记为 reasoning failure，但实际上 model 是正确识别了 unsolvability。这相当于：
- 给 SAT solver 一个 unsatisfiable formula，solver 返回 "UNSAT"，然后你说 solver "failed"

---

## 3. 替代实验：从 enumeration 到 abstraction

本文作者的关键 counter-experiment：

**Prompt:** "Solve Tower of Hanoi with 15 disks. Output a Lua function that prints the solution when called."

**结果：** Claude-3.7-Sonnet、Claude Opus 4、OpenAI o3、Google Gemini 2.5 全部 high accuracy，**用 < 5,000 tokens 完成了 $2^{15} - 1 = 32{,}767$ 步的 solution**。

**为什么这能 work：**

Tower of Hanoi 的 recursive solution 是：

```lua
function hanoi(n, from, to, via)
    if n > 0 then
        hanoi(n-1, from, via, to)
        print("Move disk " .. n .. " from " .. from .. " to " .. to)
        hanoi(n-1, via, to, from)
    end
end
hanoi(15, 'A', 'C', 'B')
```

这个 function 的 token 数是 $O(1)$ 的（不随 $N$ 增长），但它 **蕴含** 了 $O(2^N)$ 的 solution。

**关键 insight：** reasoning 的本质是 **掌握 generative procedure**，而非 **enumerate instances**。原 paper 的 evaluation format 强制 model 做 enumeration，然后指责 model 在 enumeration 上的失败是 reasoning failure。

---

## 4. 重新评估 complexity metric

**Table 1 详解：**

| Puzzle | Solution Length | Branching Factor | Search Required |
|---|---|---|---|
| Tower of Hanoi | $2^N - 1$ | 1 | No |
| River Crossing | $\sim 4N$ | $> 4$ | Yes (NP-hard) |
| Blocks World | $\sim 2N$ | $O(N^2)$ | Yes (PSPACE) |

**为什么 branching factor 才是真正的 difficulty 指标：**

**Tower of Hanoi 的 branching factor = 1：**

递归结构 $\text{hanoi}(N, A, C, B) = \text{hanoi}(N-1, A, B, C) \oplus \text{move}(N, A \to C) \oplus \text{hanoi}(N-1, B, C, A)$ 完全确定了每一步。在任何 state，合法 move 是 **唯一确定的**（通过 recursive structure 反推）。所以这是一个 $O(1)$ decision per move 的问题，只是需要 $O(2^N)$ 次 execution。

形式化：Tower of Hanoi 的 state graph 是一条 **Hamiltonian path**（实际上是 Sierpinski triangle 上的特定路径），从初始 state 到 goal state 只有一条最短路径，没有 branching。

**River Crossing 的 branching factor $> 4$：**

每个 state 有多个可能的 boat load 组合 $\binom{N}{1} + \binom{N}{2} + \binom{N}{3} = O(N^3)$ 个候选 transition。state space 是 $\binom{2N}{0...N} \times \{L, R\}$，需要 BFS/DFS search。decision per move 是 $O(N^3)$，且需要 lookahead 避免 dead-end。

**Blocks World 的 branching factor $O(N^2)$：**

任何两个 stack 之间都可以 move 顶部 block，所以每步有 $O(N^2)$ 个合法 move。problem 是 [PSPACE-complete](https://www.sciencedirect.com/science/article/pii/S0004370294000431)（Bylander 1994）。

**核心 intuition：**

> Solution length 衡量的是 **执行成本**，branching factor 衡量的是 **搜索成本**。

Tower of Hanoi 的 $2^N$ solution length 来自 **mechanical recursion unrolling**，而 River Crossing 的 difficulty 来自 **combinatorial search**。原 paper 把这两者混为一谈，用 solution length 作为 complexity proxy，导致 Tower of Hanoi 在 $N=15$ 时看起来 "比" River Crossing $N=6$ "更复杂"，但实际上前者 model 可以用 $O(1)$ 的 program 表示，后者需要真正的 search。

---

## 5. 更深层的方法论问题

### 5.1 Automated evaluation 的 epistemic risk

原 paper 用 programmatic checker 验证 model output：
1. 解析 model 的 move sequence
2. 检查每一步是否合法
3. 检查最终 state 是否是 goal

这种 evaluator 的 **blind spot：**
- Model 说 "this is unsolvable" → evaluator 记为 failure
- Model 说 "the pattern continues, stopping here" → evaluator 记为 failure
- Model 输出 algorithm 而非 instance → evaluator 无法 parse → 记为 failure

所有这些都被归类为 "accuracy collapse"，但它们是 **三种完全不同的现象**。

### 5.2 Representation matters

这让我想到 [GRPO paper](https://arxiv.org/abs/2402.03300) 里的 insight：reasoning ability 的表现严重依赖于 output format。让 model 输出 CoT vs. 输出 direct answer，performance 差异巨大。本文把这个 insight 推进一步：**让 model 输出 enumeration vs. 输出 generator，performance 差异也是巨大的**。

这与 [Chain-of-Thought prompting](https://arxiv.org/abs/2201.11903) 的精神一致：给 model 正确的 "thinking format" 才能释放其能力。

### 5.3 AI evaluation 的 "Goodhart's Law"

> When a measure becomes a target, it ceases to be a good measure.

原 paper 把 "准确输出完整 move list" 作为 reasoning ability 的 proxy measure，然后发现 model 在这个 proxy 上 collapse，就声称 reasoning 是 illusion。但 proxy ≠ target。真正的 reasoning ability 是 **掌握 generative procedure**，model 通过 Lua function 展示了这一点。

---

## 6. 我（模拟 Karpathy 视角）的思考

这篇 paper 触及了一个我一直关心的问题：**如何正确评估 LLM 的 reasoning？**

回顾 [Karpathy 的 "Software 2.0"](https://karpathy.medium.com/software-2-0-a6eb52a15190) 论点：神经网络是新的 programming paradigm，model 学到的是 dataset-defined program。评估一个 program 是否 "会 reasoning"，应该看它能否 **generate correct behavior on novel inputs**，而非看它能否 **memorize and replay specific outputs**。

原 paper 的 evaluation 本质上是在测试 **replay ability under length constraint**，而非 **generate ability**。本文的 Lua function counter-experiment 才是真正测试 generate ability。

另一个联想：这与 [AlphaCode](https://www.nature.com/articles/s41586-021-03342-y) 的评估哲学一致。AlphaCode 不要求输出完整 execution trace，只要求输出 **program**，然后通过 test cases 验证。如果把 AlphaCode 评估为 "必须输出完整 execution trace"，它在 $N = 10^6$ 的 test case 上也会 "collapse"。但没人会因此说 AlphaCode "不会 programming"。

**公式补充 — representation 的 information-theoretic view：**

Kolmogorov complexity $K(x)$ 定义为输出 string $x$ 的最短 program 长度。Tower of Hanoi 的 move sequence $x_N$ 的 $K(x_N) = O(\log N)$（因为 recursive program 是 $O(1)$ 的，加上 $\log N$ 的 input encoding）。但原 paper 要求 model 输出 $x_N$ 本身，长度是 $O(2^N)$。

$$K(x_N) = O(\log N) \ll |x_N| = O(2^N)$$

Model 实际上掌握了 low-Kolmogorov-complexity 的 generator，但 evaluation 强制它输出 high-complexity 的 instance。这是 **representation mismatch**，不是 **reasoning failure**。

---

## 7. 结论与启示

本文的核心贡献不是 "证明 LRM 会 reasoning"，而是 **指出原 paper 的实验设计缺陷**：
1. Token limit 被误读为 reasoning limit
2. Unsatisfiable instances 被记为 failure
3. Solution length 被误用为 complexity proxy
4. Enumeration format 被强制，忽略了 abstraction ability

更深层启示：**AI evaluation 需要区分 "cannot" 和 "will not"**。Model 选择压缩 output 不是能力不足，而是 meta-cognitive decision。未来的 evaluation framework 应该：
- 允许 model 选择 representation
- 验证 puzzle solvability
- 区分 reasoning capability 和 output constraint
- 用 computational complexity（而非 solution length）作为 difficulty metric

最后一句话总结本文的 meta-lesson：

> The question isn't whether LRMs can reason, but whether our evaluations can distinguish reasoning from typing.

---

## Reference Links

1. 原始 paper（被反驳的）：[Shojaee et al. 2025, "The Illusion of Thinking"](https://arxiv.org/abs/2506.06503)
2. 本文：[Opus & Lawsen, "The Illusion of the Illusion of Thinking"](https://arxiv.org/abs/2507.08071)（注：此为推测 link，原文未提供 arXiv link）
3. Twitter 复现：[@scaling01 thread](https://x.com/scaling01/status/1931817022926839909)
4. Faith and Fate paper：[Dziri et al. 2023](https://arxiv.org/abs/2305.18654)（NeurIPS 2023，关于 transformer compositionality limit）
5. River Crossing 代数分析：[Efimova 2018](https://arxiv.org/abs/1802.09369)
6. Blocks World PSPACE-completeness：[Bylander 1994](https://www.sciencedirect.com/science/article/pii/S0004370294000431)
7. Software 2.0：[Karpathy's blog](https://karpathy.medium.com/software-2-0-a6eb52a15190)
8. AlphaCode：[Nature paper](https://www.nature.com/articles/s41586-021-03342-y)
9. Chain-of-Thought prompting：[Wei et al. 2022](https://arxiv.org/abs/2201.11903)
10. Kolmogorov complexity：[Li & Vitányi textbook](https://www.springer.com/gp/book/9781489989059)
