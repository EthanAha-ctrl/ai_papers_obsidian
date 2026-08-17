---
source_pdf: Do Language Models Need Sleep Offline Recurrence for Improved Online Inference.pdf
paper_sha256: 5fc9cccf89da5d47d12b487702f27470f9b9fc752a9da479fc38d3f757aa2bd1
processed_at: '2026-08-03T22:53:24-07:00'
target_folder: LLM-engine
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej, 我们抛开那些数学公式，直接 build up the intuition for 这篇 paper 的核心 idea。

## 一、用“读书做笔记”来理解 Memory 机制

现在的 LLM 处理长文本，主要靠两种 memory 机制。

第一种是 Transformer 的 self-attention。这就像“过目不忘”的照相记忆，模型把所有看过的 token 都存在 KV cache 里。只要没忘，随时可以精确回看。问题在于，KV cache 会随着 sequence length 线性增长，compute 更是二次方膨胀。读一本十万字的书，脑子直接塞爆。

第二种是 SSM (State Space Model，比如 Mamba) 的 fast weight。这就像“边读边做笔记”。模型有一个固定大小的矩阵 $\mathbf{S}$ (fast weight)，每读一个 token，就根据一个 local rule 把信息写进 $\mathbf{S}$ 里。读完之后，原书就被扔掉，只留这本固定大小的笔记。这种做法极其省内存，但显然是有损压缩。

现在的 hybrid 模型 (比如 Samba, Jet-Nemotron) 把两者结合：近期的高保真内容用 attention 存，太久远的 context 压缩进 SSM 的 fast weight 里。

## 二、痛点：光“记下来”没用，还得“算明白”

Hybrid 模型听起来很完美，但作者发现它在 deep reasoning 任务上直接拉胯。

想象一个任务：你读了一段 Rule 110 (一种细胞自动机) 的初始状态，然后要求你预测它演化 $t=32$ 步之后的第一位是什么。

如果 $t=0$，这就是个简单的 retrieval 任务，笔记里抄下来就行。如果 $t=32$，你必须把初始状态在脑子里推演 32 步，把最终结果记在笔记里，原状态才能被 evict (清除出 KV cache)。

Vanilla SSM 的问题在于，它的写入过程是 online 且 single-pass 的。每来一个 token，它只有一次机会用公式 $\mathbf{S}_t = \alpha_t \mathbf{S}_{t-1} + \beta_t \mathbf{v}_t \mathbf{k}_t^\top$ 更新一次笔记。这相当于要求模型在看到初始状态的那一瞬间，就立刻在脑子里完成 32 步推演并写好笔记。对于固定深度的网络来说，这根本做不到。

作者指出，这背后的 bottleneck 在于 consolidation 阶段的 compute 不足。单纯增加 fast weight matrix 的容量毫无意义，因为模型没有足够的计算步数去把 raw context transform 成可以用于后续 reasoning 的 structured representation。

## 三、Sleep 机制：离线反复咀嚼，醒时秒答

生物大脑是怎么处理这个问题的？人类在白天学习时，海马体短暂存储短期记忆；到了晚上睡觉时，大脑会进行 hippocampal replay，把白天的记忆反复回放，慢慢巩固到大脑皮层的神经元连接里。这个过程需要时间，所以动物必须睡觉，暂停响应外部刺激。

这篇 paper 直接把这个机制搬到了 LLM 上。

当模型的 context window 满了，准备要 evict KV cache 之前，模型进入“睡眠状态”。在这个 sleep phase，模型不接收任何新 token，而是对当前 context window 里的内容，反复执行 $N$ 次 recurrent forward passes。

在每次 loop 中，模型都在不断 refine SSM blocks 里的 fast weight $\mathbf{S}$。你可以理解为：模型在离线状态下，把刚才看过的那段书翻出来重新读了 $N$ 遍，每读一遍就重新整理一次笔记，把那些需要深度推理才能得出的结论，慢慢算清楚并压缩进 fast weight 里。

经过 $N$ 次 sleep passes 之后，笔记整理完毕。模型果断清空 KV cache，带着精炼后的 fast weight $\mathbf{S}$ 继续读下一段 context。等到真正提问时 (prediction phase)，模型只需要做 single forward pass，直接从笔记里找答案。完全没有任何额外的 prediction latency。

## 四、技术细节：梯度穿透 Fast Weights

这个机制在实现上有一个非常巧妙的细节。

传统的 depth-recurrent models (比如 Universal Transformers) 是在 prediction time 做循环，不断 refine feature vectors。而本文的 sleep 机制，在 sleep 结束后直接把 hidden states $h$ 丢弃，只保留更新后的 fast weight $\mathbf{S}$。这是一个非常强的 information bottleneck，强制模型把所有有用的 reasoning 结果都压缩进 weights 里。

训练时，模型通过 backpropagation through time (BPTT) 把 gradient 传过这 $N$ 次 recurrent passes。由于 gradient flows through refined fast weights (而非 refined features)，模型本质上是在 meta-learn 一个“如何高效整理记忆”的 update rule。这与 test-time training 或者 context distillation 有本质区别，后两者通常是执行预定义的 gradient descent，而本文的 update rule 是完全 end-to-end learned 的。

参考: [Universal Transformers (Dehghani et al., 2018)](https://arxiv.org/abs/1807.03819) | [Teaching LMs to Think Deeper (McLeish et al., 2025)](https://arxiv.org/abs/2511.07384)

## 五、实验数据：越难的问题，越需要睡觉

作者在三个任务上验证了这个 idea，结果 trend 极其一致。

**1. Cellular Automaton (Rule 110)**
在需要推演 $t=32$ 步的设定下，没有 sleep 的 vanilla hybrid model 接近 random guessing (10% accuracy)。加上 2 次 sleep passes 提升到 20%，4 次 sleep passes 直接干到 30% 以上。Context length 和 eviction rule 完全没变，纯粹是多睡几觉带来的 compute红利。

**2. Depo (Multi-hop Graph Retrieval)**
这个任务要求模型在图里做 $k$-hop 查找。$k$ 越大，需要的 traversal 越深。实验表明，1-loop model 在 4-hop 就卡住了，2-loop model 在 8-hop 卡住。只有 4-loop model 才能在 16-hop 任务上开始 improve。这完美印证了：deeper reasoning 需要更多 sleep duration。

**3. GSM-Infinite (真实数学推理)**
作者拿预训练好的 Jet-Nemotron 2B 和 Ouro 1.4B 进行 fine-tuning。对于只需要 2-4 步运算的简单题，睡不睡都差不多。但面对需要 6-8 步运算的难题时，增加 sleep passes $N$ 带来了巨大的 accuracy 提升 (Ouro 在 6-op 上从 0.419 飙升到 0.615)。

参考: [GSM-Infinite (Zhou et al., 2025)](https://arxiv.org/abs/2502.05292) | [Jet-Nemotron (Gu et al., 2025)](https://arxiv.org/abs/2508.15884)

## 六、最终 Intuition: Amortized Reasoning

总结一下，这篇 paper 的核心洞察在于 **reasoning compute 的重新分配**。

传统的 test-time scaling (比如 Chain-of-Thought) 是在看到问题后，花大量时间思考。这增加了 prediction latency，并且每来一个新问题都要重新想一遍。

Sleep mechanism 提供了另一种范式：**Amortized reasoning**。在阅读 context 的时候，多花 $N$ 倍的 compute 把复杂逻辑推演清楚，把结果固化成 weight connections。等查询到来时，直接做单次前向传播提取答案。

这就像学生复习备考：与其在考场上对着题目冥思苦想 (CoT)，不如在考前一天晚上好好睡一觉，把所有知识点都梳理清楚记在脑子里 (Sleep consolidation)，第二天考试直接默写，又快又准。当同一个 context 会被多次 query 时，这种 amortization 的性价比极高。

---

# Do Language Models Need Sleep? 深度解析

Andrej, 这篇 paper 的核心 insight 其实非常优雅, 让我先 build up the intuition 然后再展开技术细节。

## 一、核心直觉: 为什么需要 "Sleep"

考虑一个 hybrid attention-SSM model 处理 long context。当 token stream 超过 context window $L$ 时, KV cache 必须被 evict。SSM blocks 中的 fast weight $\mathbf{S}_t$ 充当 compressed memory, 但是问题在于:

**Memory capacity ≠ Reasoning capacity**

vanilla SSM 写入 fast weight 的过程是 online、single-pass 的: 每来一个 token, 用一个 local rule $\mathbf{S}_t = \alpha_t \mathbf{S}_{t-1} + \beta_t \mathbf{v}_t \mathbf{k}_t^\top$ 更新一次。这个 update 本身就是一种极其受限的 "computation" —— 只有一个 step 来 transform incoming token 成 fast weight 的一部分。

当 task 需要 deep reasoning (例如 simulate Rule 110 for $t=32$ steps, 或者 multi-hop graph traversal $k=16$), single-pass consolidation 根本不够。这不是 storage bottleneck, 而是 **compute bottleneck during consolidation**。

生物学启发非常 natural: 人类 hippocampus 在 sleep 期间通过 replay 把 short-term memory consolidate 到 cortical weights, 这个 process 是 iterative 的, 不是 one-shot 的。Paper 把这个 idea 直接搬过来: context window 满了之后, 让 model 在 offline 阶段对当前 context 做 $N$ 次 recurrent forward passes, 每次 refine fast weights, 然后才 evict KV cache。

参考: [Sleep's role in memory (Rasch & Born, 2013)](https://doi.org/10.1152/physrev.00032.2012) | [Complementary Learning Systems (McClelland et al., 1995)](https://doi.org/10.1037/0033-295X.102.3.419)

---

## 二、Preliminaries: 两种 Memory 机制对比

### 2.1 Softmax Attention

$$
\mathbf{q}_t = \mathbf{W}_Q \mathbf{x}_t, \quad \mathbf{k}_t = \mathbf{W}_K \mathbf{x}_t, \quad \mathbf{v}_t = \mathbf{W}_V \mathbf{x}_t
$$

- $\mathbf{x}_t \in \mathbb{R}^d$: timestep $t$ 的 token representation (column vector)
- $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d \times d}$: learned projection matrices
- $\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t \in \mathbb{R}^d$: query, key, value vectors

Attention output:
$$
\mathbf{o}_t = \mathbf{V}_t^\top \operatorname{softmax}\left(\frac{\mathbf{K}_t \mathbf{q}_t}{\sqrt{d}}\right)
$$

- $\mathbf{K}_t = [\mathbf{k}_1, \dots, \mathbf{k}_t]^\top \in \mathbb{R}^{t \times d}$: stacked keys up to time $t$
- $\mathbf{V}_t = [\mathbf{v}_1, \dots, \mathbf{v}_t]^\top \in \mathbb{R}^{t \times d}$: stacked values
- $\sqrt{d}$: scaling factor 防止 dot product 过大导致 softmax saturation

KV cache 大小 $O(t \cdot d)$, linear growth, retrieval 是 exact 的。

### 2.2 Linear Recurrent Layer (Mamba2-style)

$$
\mathbf{S}_t = \alpha_t \mathbf{S}_{t-1} + \beta_t \mathbf{v}_t \mathbf{k}_t^\top, \quad \mathbf{o}_t = \mathbf{S}_t \mathbf{q}_t
$$

- $\mathbf{S}_t \in \mathbb{R}^{d \times d}$: fast weight matrix at time $t$, **fixed size**, 不随 $t$ 增长
- $\alpha_t \in (0, 1)$: data-dependent forget gate, 控制 prior memory 衰减
- $\beta_t \in (0, 1)$: data-dependent input gate, 控制新信息写入强度
- $\mathbf{v}_t \mathbf{k}_t^\top \in \mathbb{R}^{d \times d}$: rank-1 outer product, Hebbian-like association
- $\mathbf{q}_t$: query 用于读取 memory

关键: $\mathbf{S}_t$ 是 **matrix-valued state**, 容量 $d^2$, 不管看过多长 context 都不增长。这是 SSM 的 memory efficiency 来源, 同时也是 lossy compression 的根源。

Gated Delta Networks (GDN) 在此基础上加 delta-rule correction:
$$
\mathbf{S}_t = \alpha_t \mathbf{S}_{t-1} + \beta_t (\mathbf{v}_t - \mathbf{S}_{t-1} \mathbf{k}_t) \mathbf{k}_t^\top
$$
这里 $(\mathbf{v}_t - \mathbf{S}_{t-1} \mathbf{k}_t)$ 是 prediction error, delta rule 让 memory 可以 overwrite 而不只 append。具体 update rule 对本文 method 不重要, 重要的是 fast weight 是被 recursively refined 的对象。

参考: [Transformers are SSMs (Dao & Gu, 2024)](https://arxiv.org/abs/2405.21060) | [Gated Delta Networks (Yang et al., 2024)](https://arxiv.org/abs/2412.06464) | [Linear Transformers as Fast Weight Programmers (Schlag et al., 2021)](https://proceedings.mlr.press/v139/schlag21a.html)

---

## 三、Motivating Example: Rule 110 Stress Test

### 3.1 Task Setup

Rule 110 是 1D elementary cellular automaton, 每个 cell 根据自身和左右邻居依据固定 local rule 更新。Predicting $t$-th state 是 P-complete ([Neary & Woods, 2006](https://doi.org/10.1007/11786986_12)), 意味着没有 known parallel shortcut, 必须 sequentially simulate $t$ 步。

Task 构造:
- 4 个独立 length-24 binary strings, 作为 4 个 initial states
- Character-level tokenizer ('0' 和 '1' 是 tokens)
- Total context: $T = 24 \times 4 + 4 = 100$ tokens (4 个 label tokens)
- 每个 label 是对应 state unroll $t$ 步后取 first bit

$$
\underbrace{0101\ldots1101}_{\text{state0}} | \underbrace{1101\ldots1000}_{\text{state1}} | \underbrace{101\ldots1000}_{\text{state2}} | \underbrace{1101\ldots0110}_{\text{state3}} | \underbrace{1}_{l_0} \underbrace{0}_{l_1} \underbrace{1}_{l_2} \underbrace{1}_{l_3}
$$

### 3.2 Hard Eviction Constraint

设 window size $L = 24$, 每 24 tokens 完全清空 KV cache。这意味着:
- 处理 state0 时, state1/2/3 还没看到
- 处理 state1 时, state0 已经从 KV cache evict, 只能存在 fast weights $\mathbf{S}_t$ 中
- Prediction phase (最后 4 tokens) 时, **所有 state context 都被 evict**, 必须从 fast weights 读取

### 3.3 Prediction-Phase Latency Constraint

关键约束: prediction 阶段每个 label token 只能用 single forward pass, **不允许 chain-of-thought 或者 loops**。这是 paper 的核心 trade-off —— 把 extra compute 推到 sleep 阶段, 严格保留 wake-time latency。

### 3.4 Vanilla Hybrid Model Fails

Figure 2a 显示: 4-layer GDN-attention hybrid (attention → GDN → attention → GDN) 在 $t$ 增大时性能急剧下降。当 $t=0$ (pure retrieval), SSM 完全够用。当 $t$ 增大, 需要 simulate automaton $t$ 步, single-pass fast weight update 根本无法 embed 这种 deep computation。

注意: 这不是 memory capacity 问题 (paper 反复强调), 因为 sequence length $T$ 固定, 只有 $t$ 变。是 **consolidation-time compute 不够**。

---

## 四、LLM Sleep: 离线 Recursive Memory Consolidation

### 4.1 Architecture

核心 modification: 在 eviction boundary, 对当前 chunk 做 $N$ 次 recurrent forward passes, 每次 update fast weights, 然后才 evict KV cache。

$$
\text{Embed} \to \left[\mathcal{B}_0^{\text{attn}} \to \mathcal{B}_1^{\text{ssm}} \to \cdots \to \mathcal{B}_{D-1}^{\text{attn}}\right]^{\times N} \to \text{OutProj}
$$

- $\mathcal{B}_\ell^{\text{attn}}$: 第 $\ell$ 个 attention block (包含 attention + MLP + residual + norm)
- $\mathcal{B}_\ell^{\text{ssm}}$: 第 $\ell$ 个 SSM block (fast weight memory + MLP + residual + norm)
- $\times N$: 整个 block stack loop $N$ 次

### 4.2 Algorithm 1 详解

```
Require: tokens x, loss mask m, window size L, sleep passes N
1: Zero-initialize SSM fast weights S
2: Split x, m into non-overlapping chunks of length at most L
3: for each token chunk c and its loss mask m_c do
4:     h ← Embed(c)
5:     if m is all-zero then  ▷ consolidation phase
6:         for n = 1, ..., N do
7:             h, S ← Blocks(h, S)
8:         end for
9:     else  ▷ prediction phase
10:        h, S ← Blocks(h, S)
11:        L ← MaskedCE(OutProj(h), c, m_c)
12:    end if
13: end for
14: Backpropagate L and take an optimizer step
```

关键点:
- **Consolidation phase**: chunk 没有 label (loss mask 全 0), 跑 $N$ 次 forward, 每次都用同一个 chunk 的 hidden states $h$ 作为输入, 但 S 内部 state 被 carry over。也就是说每次 loop 都基于同一个 context 再做一轮 computation, refine $\mathbf{S}$。
- **Prediction phase**: chunk 有 label, 只跑一次 forward, 计算 masked cross-entropy loss。
- **Backprop through entire graph**: gradient flows through 整个 $N$-pass recurrent computation, 包括 refined fast weights。这点跟 prior depth-recurrent models 不同 —— 那些是 gradient through refined feature vectors, 这里 gradient through refined **weights** (因为 refined features 被 discard)。

### 4.3 与 Depth-Recurrent Models 的区别

Universal Transformers [Dehghani et al., 2018] 和后续 depth-adaptive 工作 [Graves, 2016; Geiping et al., 2025; McLeish et al., 2025] 是在 **prediction time** 反复 apply 同一组 weights 处理同一个 sequence, refine features。

本文的 twist: sleep 阶段 refine 的对象是 **fast weights** $\mathbf{S}$, 而 features $h$ 被 discard。这样 prediction time 可以 single-pass, 把所有 iterative compute 提前到 consolidation 阶段 amortize 完。

Intuition: 把 "thinking" 从 query time 移到 memory encoding time。Biological sleep 干的就是这个 —— 你白天不能边处理 sensory input 边深度 consolidate, 必须 offline。

参考: [Universal Transformers (Dehghani et al., 2018)](https://arxiv.org/abs/1807.03819) | [Adaptive Computation Time (Graves, 2016)](https://arxiv.org/abs/1603.08983) | [Scaling Test-Time Compute with Latent Reasoning (Geiping et al., 2025)](https://arxiv.org/abs/2408.03314) | [Retrofitted Recurrence (McLeish et al., 2025)](https://arxiv.org/abs/2511.07384)

---

## 五、Experiments 详解

### 5.1 Cellular Automaton (Rule 110)

| Model | $t=32$ Accuracy | 备注 |
|-------|----------------|------|
| No loop (vanilla hybrid) | ~10% | 接近 random (4 states, 第一位 0/1 二分类, 4 个 label 全对才算 exact accuracy, random 是 $1/16 \approx 6.25\%$) |
| 2 loops | ~20% | |
| 3 loops | ~30%+ | |
| 4 loops | ~30%+ | |

架构: 4-layer GDN-attention hybrid, hidden dim $d=256$, layout attention → GDN → attention → GDN。Training 5B tokens。Muon optimizer, lr 2e-3 (在 $N=1$ 上 tune 后用于所有 looped models, 给 baseline 优势)。

观察: 增加 $N$ 同时加速 learning 和提升 final accuracy。Context length、eviction rule、prediction-phase computation 全部固定, 改进完全来自 consolidation-time compute。

### 5.2 Depo: Multi-hop Graph Retrieval

[Allen-Zhu & Li, 2025](https://openreview.net/forum?id=kxv0M6I7Ud) 提出, 评估 reasoning depth。Sequence 是 shuffled directed cycle (e.g., `b->a, f->l, ..., e->b`), 后跟 queries (`1 hop after a: c`, `4 hops after e: d`)。

Setup:
- 75 nodes per cycle, 300 tokens, 左 padding 到 300
- 10 query-answer pairs, 60 tokens
- Total $T = 360$, window $L = 75$ (cycle 跨 4 个 windows!)
- $k \in [1, 16]$ training, test on $k \in \{1, 2, 4, 8, 16\}$
- 10-layer GDN-attention hybrid, $d=512$

为什么 Depo 比 Rule 110 难:
1. **Cycle fragmented across 4 windows** —— 不像 automaton 一个 state 在一个 window 内
2. **Query-agnostic representation** —— $k$ 和 start node 都是 random, 不像 automaton 的 $t$ 是 fixed。Model 必须 store graph 结构本身, 不能预先 simulate 出 answer。

| Hop $k$ | $N=1$ | $N=2$ | $N=4$ |
|---------|-------|-------|-------|
| 1 | converges | converges | converges |
| 2 | converges | converges | converges |
| 4 | stalls | improves | improves |
| 8 | stalls | stalls | improves |
| 16 | stalls | stalls | begins to improve |

非常 clean 的 trend: **deeper queries 需要更多 sleep passes**。$k=16$ 在 budget 内只有 $N=4$ 开始 improve。

Intuition: model 在 sleep 期间本质上是在做 multi-hop traversal 的 precomputation。$N$ 越大, traversal 越深, encode 到 fast weights 的 "经过 $k$ 跳后的节点信息" 越准确。这是一个 amortized reasoning process。

### 5.3 GSM-Infinite: 真实数学推理

[GSM-Infinite (Zhou et al., 2025)](https://arxiv.org/abs/2502.05292) 是 procedural generated 的 GSM8K 变体, 通过加 distractor tokens 控制长度, 通过控制 arithmetic operations 数量控制难度。

Setup:
- 2000-3300 tokens per problem
- 1-8 operations
- Question 放在 context **之前** (让 model 知道要 query 什么, 选择性 consolidate)
- **No CoT traces**, 强制 single forward pass 预测 final answer
- $L = 2000$, problem 不能全放进 active window

两个 base models:

**Jet-Nemotron 2B** ([Gu et al., 2025](https://arxiv.org/abs/2508.15884)): SSM-attention hybrid, fine-tuned from Qwen 2.5 1.5B by replacing some attention with Jet layers (dynamic convolution)。Loop middle 14/28 blocks。$N \in \{1, 2, 4, 6\}$。

**Ouro 1.4B** ([Zhu et al., 2025](https://arxiv.org/abs/2510.25741)): depth-recurrent attention-only model。Insert 6 Jet layers (no MLP) 增 fast weight memory, params 增 <10%。Loop 全部 blocks。$N \in \{1, 2, 4\}$。

| Model | Ops | $N=1$ Acc | $N=\max$ Acc | Gain |
|-------|-----|-----------|--------------|------|
| Jet | 6 | 0.742 | 0.812 (N=6) | +0.070 |
| Jet | 8 | 0.351 | 0.388 (N=6) | +0.037 |
| Ouro | 6 | 0.419 | 0.615 (N=4) | +0.196 |
| Ouro | 8 | 0.210 | 0.272 (N=4) | +0.062 |

Pattern:
- Easy problems (2-4 ops): accuracy 接近 saturation, loops 帮助小
- Hard problems (6-8 ops): loops 帮助显著, Ouro 上 gap 更大
- Ouro 的更大 gap 可能 reflect 其 depth-recurrent pretraining 让它更善于利用 extra recurrence

Jet 上 gap 小一些可能因为 Jet 本身有更多 fast weight capacity (28 blocks 里 14 个是 hybrid), baseline 已经更强。

### 5.4 Sliding-Window Eviction

之前是 hard eviction (全清), 这里改成 sliding window: 保留最近 $L-1$ tokens, evict 旧的。Peak inference memory 不变 (还是 $L$ tokens), $N=1$ 时退化成标准 SWA-SSM hybrid [Samba, Ren et al., 2024](https://arxiv.org/abs/2406.07522)。

Setup: $L=512$, sequence 是 4-6× window size。Ouro 1.4B, $N \in \{1, 2, 4\}$。先 warm up Jet layers only 一个 epoch (用 hard eviction!), 再 full model 训两个 epoch。

| Ops | $N=1$ | $N=4$ |
|-----|-------|-------|
| 2-op | 0.596 | 0.905 |
| All ops | improves | improves |

注意 $N=1$ 在 2-op 上只有 0.596, 远低于 hard eviction setup 的水平。原因: 当 active window 远小于 sequence length 时, 即使是简单 retrieval 也需要 compression, 单次 consolidation 不够。Loops 在 retrieval (不仅是 reasoning) 上也大幅提升 —— 这拓展了 paper 中心 claim: sleep-time compute 帮助 compression + retrieval, 不只是 multi-step reasoning。

Warm-up 用 hard eviction 很关键 —— 让 model 先学会 refine fast weights, 再切到 sliding window。这是 [Cabannes et al., 2025](https://arxiv.org/abs/2509.24552) 也观察到的现象: sliding window 容易让 SSM layers underutilized。

参考: [Samba (Ren et al., 2024)](https://arxiv.org/abs/2406.07522) | [Mamba in the Llama (Wang et al., 2024)](https://arxiv.org/abs/2412.08672) | [Short Window Attention enables Long-term Memorization (Cabannes et al., 2025)](https://arxiv.org/abs/2509.24552)

### 5.5 Training Throughput 分析

**Recurrence across context windows**: 训练时 windows 之间有 sequential dependency (window $j+1$ 依赖 window $j$ refined 后的 $\mathbf{S}$), 无法全 parallel。但当 $L$ 足够大让 GPU saturated, 这种 serialness 实际不影响 wall-clock time (Figure 6a)。

**Recurrent-depth cost**: throughput 大致 $\propto 1/N$ (Figure 6b)。Activation checkpointing 沿 context chunk axis 防 OOM。FlashAttention 2 ([Dao, 2024](https://arxiv.org/abs/2307.08691)) 用于 attention layers。

直觉: 这跟 test-time scaling 的 cost profile 类似 —— 你 pay $N$ 倍 compute 换更好 reasoning。区别是这里 compute amortized over all future queries to that consolidated memory, 而不是 per-query。

---

## 六、与 Related Work 的精细对比

### 6.1 Context Compression

[ICAEReal (Ge et al., 2023)](https://arxiv.org/abs/2307.06945): LM 把 long context compress 成短 hidden state sequence, 再 feed 给 LM。Still in attention context。
[Cartridges (Eyuboglu et al., 2025)](https://arxiv.org/abs/2506.06266): offline self-study 学一个 small KV cache 替代 full context。Still in KV cache form。

本文: transfer **evicted** context 到 **weight-based** memory, 彻底离开 KV cache。本质上是把 "短 context" 换成 "weights"。

### 6.2 Context Distillation

[Distilling Context (Snell et al., 2022)](https://arxiv.org/abs/2209.15189), [Askell et al., 2021](https://arxiv.org/abs/2112.00861): train model 不带 context 模仿带 context teacher。
[Tack et al., 2024](https://arxiv.org/abs/2403.04317): amortize context 到 memory。

这些都是 gradient descent on predefined loss, 本文是 **learned recurrent forward pass** 作为 update rule, 更 flexible, 不对应 fixed scalar objective。

### 6.3 Test-Time Training

[Tandon et al., 2025](https://arxiv.org/abs/2512.23675): sliding window attention + test-time gradient updates on MLP subset, 一个 gradient step per chunk。Cross-entropy loss on observed context。

差异:
- 本文: learned recurrent forward pass (不是 gradient step)
- 本文: 可以 $N$ 步 (不是 one-step)
- 本文: synthetic tasks 独立 control reasoning depth vs. problem length, Tandon 主要 perplexity on web text (retrieval + reasoning entangled)

[Zhang et al., 2026](https://arxiv.org/abs/2602.16839): LoRA adapter per chunk, RL setting, one update per chunk。

### 6.4 Sleep-Inspired Methods

[Sleep-time Compute (Lin et al., 2025)](https://arxiv.org/abs/2504.13171): LLM 在 sleep 时 generate expected questions 并 precompute answers。本质是 explicit planning, 不是 weight consolidation。
[Behrouz et al.](https://arxiv.org/abs/2502.15692): RL + parameter expansion + teacher-student distillation + synthetic data。Complex pipeline, 不是 end-to-end learned consolidation。

本文 closer to 生物学 hippocampal replay: same content replayed, weights refined, content discarded。End-to-end learned update rule, 不是 hand-designed pipeline。

### 6.5 Offline Planning

[Momennejad et al., 2018](https://elifesciences.org/articles/32548): 人类 offline replay during rest predicts improved planning。
[Chalvidal et al., 2022](https://arxiv.org/abs/2204.10338): single-layer net + recursive Hebbian updates for fast adaptation in RL。和精神接近, 但 scale 和 domain 不同。

---

## 七、Build Intuition: 这篇 Paper 在做什么

我想用一个类比来 crystallize intuition:

**普通 SSM 是 "scan"**: 一个 token 一个 token 顺序处理, 每步用 fixed update rule 把 information 压进 $\mathbf{S}$。这像你边读边做笔记, 但每个字只能看一眼, 不能回头重读。

**Sleep 是 "reread and reorganize"**: 把刚刚读过的 chunk 在脑子里反复 $N$ 遍, 每次重新组织笔记。最终笔记 (fast weights) 比 single-pass 版本结构化得多, 后续 query 时可以 single-pass 直接抽答案。

**Depth-recurrent at prediction time 是 "think step by step"**: 看到问题后反复思考。这是 chain-of-thought 的 latent 版本。

本文的 twist: 把 "reread and reorganize" 放在 sleep 阶段 (memory consolidation), prediction 时只做 single-pass。这对应生物 sleep 的功能 —— 白天高效响应, 夜里 consolidate。

为什么这 work: 很多 reasoning task 的 difficulty 不在 "query 复杂", 而在 "把 raw context transform 成可 query 的 representation 需要多步 computation"。Sleep amortize 这个 transformation。Prediction-time 的 query 可以简单 lookup, 因为 heavy lifting 已经做完。

为什么不直接做 chain-of-thought at prediction:
1. Prediction latency constraint (real-time serving 不允许 long CoT)
2. Amortization: 一次 consolidation 服务多个 future queries
3. Reasoning over **evicted** context: CoT 也 access 不到 evicted tokens, 但 fast weights 保留着

为什么 prior SSM-attention hybrid 在 deep reasoning 上 fail: 它们的 fast weight update 是 online single-pass, 等于 "边读边记一次"。Sleep 允许 "读一遍记 $N$ 次", 每次 refine。

---

## 八、Limitations 和 Open Questions

Paper 自己指出:
1. **训练慢且 unstable**: $N$ 次 forward + backward, gradient through $N$-step recurrence 容易 vanish/explode。
2. **Sequential across windows**: 失去 sequence-axis parallelism, 但 $L$ 大时影响小。
3. **Implicit gradients / truncated BPTT** ([Deep Equilibrium Models, Bai et al., 2019](https://arxiv.org/abs/1909.01377); [Parcae, Prairie et al., 2026](https://arxiv.org/abs/2604.12946)) 可能缓解。

我自己的 observations / open questions:

1. **为什么 discard features $h$ 而只保留 $\mathbf{S}$?** Paper 说 "gradient flows through refined fast weights because we discard the refined features after sleep"。这是 design choice —— 如果保留 features 就退化成 depth-recurrent at prediction time。Discard 强制 information bottleneck through weights, 类似 autoencoder 的 bottleneck principle。但这也限制了 expressivity, 因为 $d^2$ 的 matrix state 可能不够 encode 所有有用 features。

2. **$N$ 如何 scale with task difficulty?** Rule 110 $t=32$ 需要 $N=4$, Depo $k=16$ 需要 $N=4$。是否有个 scaling law $N \propto t$ 或者 $N \propto \log(\text{difficulty})$? Paper 没明说, 但 Figure 2/3 的 trend 暗示 linear-ish。这个对 practical deployment 重要 —— 如果 $N$ 必须 $\gg$ problem depth, 那 amortization 不划算。

3. **Generalization across consolidation chunks**: 同一个 $\mathbf{S}$ 被 multiple chunks 累积 update, 后续 chunks 的 consolidation 是否会 overwrite 早期 chunks 的 refined state? $\alpha_t$ gate 的设计在这里关键。Paper 没深入讨论 catastrophic forgetting in $\mathbf{S}$ across windows。

4. **Query-agnostic vs. query-aware consolidation**: Depo 必须 query-agnostic (因为 $k$ 和 start node random), 但 GSM-Infinite 把 question 放前面, 允许 query-aware consolidation。这两种 regime 的 optimal $N$ 可能差很多。Query-aware 时 model 可以做 "selective consolidation" 只 encode relevant info, query-agnostic 必须 encode 全图。

5. **与 Test-Time Training 的更深统一**: 本文的 learned update rule 是 "meta-learned gradient descent" 的 generalization吗? 如果把 $N$ 次 recurrent forward 看作 $N$ 步 inner-loop gradient descent on some implicit objective, 那 sleep 就是 MAML-style meta-learning 的 amortized 版本。Paper 没做这个 connection, 但我觉得这是 deeper 的视角。[Chalvidal et al., 2022](https://arxiv.org/abs/2204.10338) 在 RL 上做了类似事情。

6. **Memory $\mathbf{S}$ 的 capacity bound**: $d \times d$ matrix, 对于 $d=256$ 是 65K params per layer, 对于 $d=512$ 是 262K。这些 capacity 够 encode 多少信息? [Arora et al., 2024](https://arxiv.org/abs/2402.18668) 的 recall-throughput tradeoff 给了理论上界, 但 sleep 是否突破了这个 bound? 我认为 sleep 不增 capacity, 但提高 capacity 的 **utilization** —— 同样大小的 matrix, 经过 $N$ 次 refine 可以 encode 更 structured 的信息。

7. **Biological plausibility**: Hippocampal replay 是真的 replay 序列, 本文是 replay chunk 内 hidden states。更 biological 的版本可能是 replay **compressed** representation 而不是 raw chunk。但 paper 的设计更 practical —— 直接 reuse forward pass。

---

## 九、对 Long-Context LLM 设计的 Implications

这篇 paper 让我重新思考几个 long-context 的设计 trade-offs:

1. **Inference latency vs. consolidation compute 的分离**: 传统上 long context 处理要么用 sparse attention (latency 好但 lossy), 要么 full attention (lossless 但 latency 差)。Sleep 提供 third option: 长 context 分 chunk, chunk 内 full attention, chunk 间通过 sleep consolidate。Inference latency 取决于最后一个 chunk, 与总长度无关。

2. **Amortization ratio**: 如果一个 consolidated memory 服务 $Q$ 个 queries, sleep cost $\propto N \cdot L$ 被 $Q$ 个 query 摊销。当 $Q \gg 1$ (例如 long document QA, RAG over fixed corpus), sleep 极划算。当 $Q=1$ (one-shot query), 不如直接 full attention。这给了一个清晰的 cost-benefit 分析框架。

3. **Hybrid architecture 的 role**: Pure SSM 太 lossy, pure attention 太 expensive。Hybrid + sleep 是 Pareto improvement —— attention 处理 recent high-fidelity, SSM + sleep 处理 long-range consolidated。这与 [Samba](https://arxiv.org/abs/2406.07522), [Griffin](https://arxiv.org/abs/2402.19427), [Hymba](https://arxiv.org/abs/2411.13676), [Nemotron Nano 2](https://arxiv.org/abs/2508.14444) 的 design philosophy 一致, 但加了 sleep 这个新维度。

4. **Pre-training vs. fine-tuning**: Paper 在 GSM-Infinite 上 fine-tune 预训练 model。Pre-training 阶段直接用 sleep 训练可能更有潜力 —— 让 model 从一开始就学会利用 sleep compute。但 cost $N$ 倍, 实际可行性取决于 budget。

---

## 十、Final Takeaways

1. **Core claim**: SSM-attention hybrid 的 fast weight memory bottleneck 是 consolidation-time compute, 不是 capacity。Sleep (offline recurrence) 解决这个。

2. **Mechanism**: 在 eviction boundary 对当前 chunk 做 $N$ 次 recurrent forward passes, 每次 refine fast weights $\mathbf{S}$ via learned local rule。Gradient backprop through refined weights (不是 refined features)。

3. **Empirical**: Rule 110 ($t=32$) 上 no-loop 10%, 4-loop 30%+。Depo 上 16-hop 只有 4-loop 开始 improve。GSM-Infinite 上 6-8 ops 提升 5-20%。

4. **Trade-off**: 训练 cost $\propto N$, inference latency 不变 (single-pass prediction)。当 consolidated memory 服务 multiple queries, amortization 划算。

5. **Biological plausibility**: Hippocampal replay during sleep 的直接 analog, but operationalized in a learnable, end-to-end fashion。

6. **Open questions**: scaling laws for $N$, capacity utilization of $\mathbf{S}$ after sleep, query-agnostic vs. query-aware regimes, relationship to meta-learning / test-time training。

References:
- Paper (推测 arxiv): 应该是 2025 年的工作
- [Rule 110 (Wikipedia)](https://en.wikipedia.org/wiki/Rule_110)
- [Mamba2 (Dao & Gu, 2024)](https://arxiv.org/abs/2405.21060)
- [Gated Delta Networks (Yang et al., 2024)](https://arxiv.org/abs/2412.06464)
- [Universal Transformers (Dehghani et al., 2018)](https://arxiv.org/abs/1807.03819)
- [Adaptive Computation Time (Graves, 2016)](https://arxiv.org/abs/1603.08983)
- [Scaling Latent Reasoning (Geiping et al., 2025)](https://arxiv.org/abs/2408.03314)
- [Retrofitted Recurrence (McLeish et al., 2025)](https://arxiv.org/abs/2511.07384)
- [Jet-Nemotron (Gu et al., 2025)](https://arxiv.org/abs/2508.15884)
- [Ouro (Zhu et al., 2025)](https://arxiv.org/abs/2510.25741)
- [GSM-Infinite (Zhou et al., 2025)](https://arxiv.org/abs/2502.05292)
- [Sleep-time Compute (Lin et al., 2025)](https://arxiv.org/abs/2504.13171)
- [Cartridges (Eyuboglu et al., 2025)](https://arxiv.org/abs/2506.06266)
- [Test-Time Training for Long Context (Tandon et al., 2025)](https://arxiv.org/abs/2512.23675)
- [Physics of LMs 4.1 (Allen-Zhu & Li, 2025)](https://openreview.net/forum?id=kxv0M6I7Ud)
- [Samba (Ren et al., 2024)](https://arxiv.org/abs/2406.07522)
- [FlashAttention-2 (Dao, 2024)](https://arxiv.org/abs/2307.08691)
- [Linear Transformers as Fast Weight Programmers (Schlag et al., 2021)](https://proceedings.mlr.press/v139/schlag21a.html)
- [Distilling Context (Snell et al., 2022)](https://arxiv.org/abs/2209.15189)
- [Deep Equilibrium Models (Bai et al., 2019)](https://arxiv.org/abs/1909.01377)
- [Sleep's role in memory (Rasch & Born, 2013)](https://doi.org/10.1152/physrev.00032.2012)
- [Offline replay supports planning (Momennejad et al., 2018)](https://elifesciences.org/articles/32548)
- [Repeat after me: Transformers vs SSMs at copying (Jelassi et al., 2024)](https://arxiv.org/abs/2402.01032)
- [Simple Linear Attention (Arora et al., 2024)](https://arxiv.org/abs/2402.18668)
- [Transformers to SSMs Distillation (Bick et al., 2024)](https://arxiv.org/abs/2412.08672)
- [Short Window Attention (Cabannes et al., 2025)](https://arxiv.org/abs/2509.24552)
- [NVIDIA Nemotron Nano 2](https://arxiv.org/abs/2508.14444)
- [Griffin (De et al., 2024)](https://arxiv.org/abs/2402.19427)
- [Hymba (Dong et al., 2024)](https://arxiv.org/abs/2411.13676)
- [Can you learn an algorithm (Schwarzschild et al., 2021)](https://arxiv.org/abs/2106.09093)
- [Meta-RL with Self-Modifying Networks (Chalvidal et al., 2022)](https://arxiv.org/abs/2204.10338)
- [Serial Scaling Hypothesis (Liu et al., 2025)](https://arxiv.org/abs/2507.12549)
- [Transformers learn shortcuts to automata (Liu et al., 2022)](https://arxiv.org/abs/2210.10749)
- [End-to-end Algorithm Synthesis (Bansal et al., 2022)](https://arxiv.org/abs/2206.00826)

---

如果想继续 deep dive, 我特别好奇的几个方向:
1. **Sleep 与 inference-time scaling laws 的关系**: $N$ 应该与 problem difficulty 怎么 scale?
2. **Sleep + reinforcement learning**: 把 sleep 作为 policy improvement 的 offline phase, 类似 AlphaZero 的 self-play 但 consolidate 到 weights
3. **Sleep 在 vision/multimodal 的 analog**: 视频 stream 也是 long-context, frame chunk 之间 consolidate 是否 work
4. **Multi-layer $\mathbf{S}$ 的 hierarchical consolidation**: 不同层 fast weights 应该 encode 不同 abstraction level, $N$ 是否应该 per-layer adaptive
5. **Sleep 与 continual learning**: 防止 catastrophic forgetting 的新 mechanism —— sleep 期间可以 replay old chunks (从 fast weights 解码?) 重新 consolidate

Paper 写得相当 clean, synthetic tasks 设计得很好 (Rule 110 和 Depo 都能 independently vary reasoning depth vs. memory load)。唯一让我觉得 thin 的是 GSM-Infinite 部分 —— 只 fine-tune 不 pre-train, 限制了能看到的 gain。如果在 pre-training 阶段就 incorporate sleep, scaling behavior 可能完全不同。
