---
source_pdf: Screening Is Enough.pdf
paper_sha256: 301fa713ddd2c5f2ed59b55376b38f9a630ecf405d06965594a556c13452b31e
processed_at: '2026-08-12T04:21:53-07:00'
target_folder: LLM-engine
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好,Andrej,我换个说法,用大白话从头讲一遍这个故事。

---

## 故事的起点:softmax attention 有个"原罪"

Standard attention 你可以想象成一个班级投票。query 是个学生,问:"我想知道关于 X 的信息,谁能帮我?"然后所有 keys(candidate)都举手,attention score 是每个 key 的 "相关度评分",softmax 就是把这些评分变成投票比例。

问题在哪?**不管这些 key 多么不相关,投票总和必须等于 1**。哪怕全班没有一个真正懂 X 的人,softmax 也会硬把票分给那些"最不差"的人。你没办法说"这题没人会"。

更要命的是,context 变长就像班级变大。原来 10 个人里有 1 个真懂,他能拿到大约 1/10 的票(还算可以)。现在 10000 个人里还是只有 1 个真懂,但 softmax 会被那 9999 个 noise 分散,真懂的那个人可能只拿到 1/10000 的 attention。Signal 被稀释了。

这就是为什么 Transformer 做 "needle in haystack" 检索特别难 — 不是参数不够,是 mechanism 本身不允许 "我只要这个 needle,其他全部 ignore"。

参考: [Lost in the middle (Liu et al.)](https://arxiv.org/abs/2307.03172)

---

## Screening 的核心 idea:别投票,改筛选

作者说:为什么非要投票?我们改成 **招聘面试** 模式。

每个 key 独立过面试,达标的留下,不达标直接滚蛋。留下的 key 各自按相关度贡献 value,不需要互相竞争,不需要 normalize。

这听起来简单,但要做对有几个关键技术点。

---

### 关键点 1:让 similarity 有"绝对刻度"

Softmax 的 score $q \cdot k$ 是 unbounded 的,可以是 -100 也可以是 1000。这种情况下你说"score > 5 算合格"没意义,因为 scale 会随数据变化。

所以 screening 先做 **unit length normalization**:

$$
\bar{q}_i = \frac{q_i}{\|q_i\|}, \quad \bar{k}_j = \frac{k_j}{\|k_j\|}
$$

这样 $s_{ij} = \bar{q}_i \cdot \bar{k}_j \in [-1, 1]$,永远 bounded。

- $\bar{q}_i$:第 $i$ 个 query 的 unit vector,只保留方向信息
- $\bar{k}_j$:第 $j$ 个 key 的 unit vector
- $s_{ij}$:cosine similarity,范围 $[-1, 1]$
- $s_{ij} = 1$:完全同方向(perfect match)
- $s_{ij} = 0$:正交(无关)
- $s_{ij} = -1$:反方向(完全相反)

这个 bounded scale 是后面 threshold 操作的基础。Value 也被 normalize,防止某个 value 因为 norm 大就 dominate aggregation。

---

### 关键点 2:Trim-and-Square —— 真正的 "拒绝" 操作

这是 screening 的灵魂:

$$
\alpha_{ij} = \left[\max\left(1 - r(1 - s_{ij}), 0\right)\right]^2
$$

变量解释:
- $s_{ij} \in [-1, 1]$:query-key similarity
- $r = e^{s_r} + 1$:learned parameter,定义 "acceptance width" $1/r$
- $1/r$:threshold 的严格程度,$r$ 越大 threshold 越严
- $\alpha_{ij} \in [0, 1]$:relevance,不归一化

具体行为:
- 当 $s_{ij} \leq 1 - 1/r$ 时,里头 $\leq 0$ 被 max clamp 到 0,所以 $\alpha_{ij} = 0$ —— **真的归零了,不是接近零**
- 当 $s_{ij} = 1$ 时,$\alpha_{ij} = 1$,max relevance
- 中间是 quadratic 过渡

为什么 square?我的猜测:让在 threshold 边界 gradient 平滑归零,避免硬边界导致 optimization 卡壳,同时强化接近 perfect match 的信号。

对比 [sigmoid attention (Ramapuram et al., 2025)](https://arxiv.org/abs/2411.06691):sigmoid 永远 $>0$,只是衰减,做不到 exact rejection。而且 sigmoid attention 需要 length-dependent bias 控制总和,这就破坏了 "context length 变了 relevance 不变" 这个性质。Screening 的 $\alpha_{ij}$ 只看 $q_i, k_j$ 本身,context length 怎么变它都不变 —— 这就是 paper 强调的 "absolute relevance"。

---

### 关键点 3:Softmask —— 可学习的 attention window

还有一个距离 mask:

$$
m_{ij}(w) = \begin{cases} \frac{1}{2}\left(\cos\frac{\pi(j-i)}{w} + 1\right), & -w < j - i \leq 0 \\ 0, & \text{otherwise} \end{cases}
$$

变量:
- $w = e^{s_w} + 1$:每个 tile 自己学的 window width
- $j - i$:key 相对 query 的位置(causal,只看过去)
- $m_{ij} \in [0, 1]$:距离 weight,越远越小

然后 final relevance:

$$
\alpha_{ij}^d = \alpha_{ij} \cdot m_{ij}(w)
$$

意思是:**一个 key 必须 content 相关 AND 距离在 window 内,才能 contribute**。这是 AND 关系,非常 strict 的筛选。

Inference 的时候有个 trick:如果 learned $w$ 比训练时见过的最大 seq len 还大,直接 set $w = \infty$,等于全 causal attention。这避免 window 边界的奇怪边界效应。

不同 tile 初始化时 $w$ 在 heads 间 linearly spaced,所以从一开始就被鼓励去探索不同的时间 scale —— 有的 tile 学 local pattern,有的 tile 学 long-range retrieval。

---

### 关键点 4:MiPE —— 只在需要时才用 position

这个设计我觉得最聪明。先看公式:

$$
M_i(w) = \begin{pmatrix} R(\phi(i, w)) & 0 \\ 0 & I_{d_K - 2} \end{pmatrix}
$$

$$
\phi(i, w) = \frac{\pi i \gamma(w)}{w}
$$

$$
\gamma(w) = \begin{cases} \frac{1}{2}\left(\cos\frac{\pi w}{w_{\text{th}}} + 1\right), & w < w_{\text{th}} \\ 0, & w \geq w_{\text{th}} \end{cases}
$$

变量:
- $R(\phi)$:2D rotation matrix(只用前两个 coordinates)
- $w_{\text{th}} = 256$:threshold,实验中固定
- $\gamma(w)$:smooth switch,小 window 时 $\to 1$,大 window 时 $\to 0$

直觉:
- 当 tile 的 window $w$ 小(局部操作):你需要 locality 信息,$\gamma \to 1$,MiPE 启用,RoPE-like rotation 注入位置信息
- 当 $w$ 大(全局 retrieval):你想做纯 content addressing,位置信息反而是 noise,$\gamma \to 0$,MiPE 自动关闭

为什么只动前两个 coordinates?我推测是为了 "最小扰动" —— 大部分 key 维度保持 content signal,只在必要的一小部分维度上注入 position。

对比 [RoPE](https://arxiv.org/abs/2104.09864):RoPE 是 unconditional 的,所有 distance 都被 encode,所以 long context 时需要 RoPE scaling,而 RoPE scaling 是个 hack(PI, NTK, LongRoPE 等),都不是根本解。MiPE 直接 bypass 这个问题:long-range 不用 position,就不存在 extrapolation 问题。

参考: [NoPE work (Kazemnejad et al.)](https://arxiv.org/abs/2305.19466) 也发现远距离依赖根本不需要 explicit position。

---

### 关键点 5:TanhNorm —— 不归一化的 bounded aggregation

最后聚合:

$$
h_i = \sum_j \alpha_{ij}^d \bar{v}_j, \quad u_i = \text{TanhNorm}(h_i) = \frac{\tanh(\|h_i\|)}{\|h_i\|} h_i
$$

变量:
- $h_i$:sum of survivors 的 value(没归一化,所以可以很小,也可以很大)
- $\|h_i\|$:$h_i$ 的 norm
- $\tanh(\|h_i\|)$:bounded in $[0, 1)$
- $u_i$:$h_i$ 保持方向,norm 被 capped 到 1

行为:
- $\|h_i\|$ 小(没找到什么 relevant):$\tanh(x) \approx x$,所以 $u_i \approx h_i$,几乎 identity
- $\|h_i\|$ 大(找到很多 relevant):$\tanh \to 1$,$u_i$ 的 norm 被压到 1,方向保留

为什么这是天才设计:
- 不像 softmax 强制 $\sum = 1$,所以 $\|h_i\| \to 0$ 时 model 可以 encode "什么都没找到" 这个信号
- 又不像 raw sum 那样可能爆炸,TanhNorm 提供了 soft bound
- "有多少 relevant context" 这个信息被编码进 $\|h_i\|$ 然后 transferred 到 output 的 scale

对比 softmax:softmax output 的 norm 由 $v_j$ 决定,和 "找到多少 relevant stuff" 无关。TanhNorm 让 norm 本身变成一个 meaningful signal。

---

## 把 attention 和 FFN 合并:Gated Screening Tile

传统 Transformer:

```
x → Attention → +x → FFN → +x → next layer
```

两步 sequential。

Multiscreen 直接合并:

$$
q_i = x_i W_Q, \; k_i = x_i W_K, \; v_i = x_i W_V, \; g_i = x_i W_G
$$

$$
u_i = \text{Screening}(\{q, k, v\})_i \quad \text{(retrieve)}
$$

$$
\hat{g}_i = \tanh(\text{SiLU}(g_i)) \quad \text{(gate)}
$$

$$
h_i = u_i \odot \hat{g}_i \quad \text{(GLU-style mix)}
$$

$$
\Delta x_i = h_i \cdot (e^{s_O} W_O)
$$

然后整个 layer 的 output 是 $N_H$ 个 parallel tile 的 sum:

$$
x_i^{(\ell)} = x_i^{(\ell-1)} + \sum_h \Delta x_i^{(\ell, h)}
$$

变量:
- $W_Q, W_K \in \mathbb{R}^{d_E \times d_K}$:query/key projection
- $W_V, W_G \in \mathbb{R}^{d_E \times d_V}$:value/gate projection
- $W_O \in \mathbb{R}^{d_V \times d_E}$:output projection
- $s_O$:learned output scale

直觉:一个 tile 同时干 attention 和 FFN 的活 —— screening unit 做 context retrieval,GLU gate 做 feature selection。这和 [Shazeer 的 GLU variants](https://arxiv.org/abs/2002.05202) 是同源思想,只不过把 linear transform 换成 screening-based aggregation。

最近 [Gemini 报告](https://arxiv.org/abs/2305.13245) 也提到 parallel attention+FFN 是有效的设计,但 Multiscreen 更彻底 —— 直接融合成一个 op,不是简单 parallel。

---

## 实验结果:四个维度同时提升

### 1. Parameter efficiency:省 40%

[Figure 3](https://arxiv.org/abs/2501.19399):同 token budget 下,Multiscreen 的 scaling curve 大致平移到 Transformer 的 0.6x params 位置。

8M/18M/28M/45M/124M/286M/353M/4B 多个 scale 一致。

我猜测原因:
- attention+FFN 合并减少 sequential overhead
- absolute relevance 让 model 不用学 "normalize 后怎么保留 strong signal" 这种 tricky 的事
- TanhNorm 提供 stable output scale,residual stream 信息流更 efficient

### 2. Learning rate stability:能在 LR=1 训练

[Figure 4](https://arxiv.org/abs/2501.19399):Transformer 在 LR $\geq 2^{-4} = 0.0625$ 时 diverge。Multiscreen 在 LR = $2^0 = 1$ 时仍稳定。

这个非常 dramatic。28M 模型用 LR 0.0625 都不算小,但 Multiscreen 用 1.0 都能稳定收敛。

[Appendix D](https://arxiv.org/abs/2501.19399) 的 gradient norm:Multiscreen 快速 decay 到接近 0,Transformer 维持 non-zero gradient floor with spikes。

直觉解释:softmax 的 "winner-take-all" dynamics 对 perturbation 敏感 —— 某个 key 异常高分就 steal 所有 attention,这种 non-smooth dynamics 在大 LR 下容易 blow up。Screening 每个 key 独立,perturbation 不 propagate 到其他 keys,landscape 更 smooth。

参考:这个发现和 [scaled dot product attention 的稳定性的分析](https://arxiv.org/abs/2106.09660) 有点联系。

### 3. Long context perplexity:稳定

[Figure 5](https://arxiv.org/abs/2501.19399) 用 PG-19 测到 $2^{17} \approx 131K$ context:

- Training context:base $2^{12}$,continual $2^{15}$
- Transformer:超过 training context perplexity 急升,RoPE scaling 只是延缓
- Multiscreen:平稳

原因:MiPE 在 long-range 时关闭 position,所以没有 "position extrapolation" 问题。retrieval 是纯 content-based,length-invariant。

### 4. ABCDigits retrieval:碾压级表现

这是 paper 最 striking 的实验。

任务设计:
- 26 个 uppercase letters 各 assign 一个 n-digit number(A=967892, L=169428 等)
- Shuffled list,target letter 只出现一次,在 specified depth
- Query "L=",model 要 complete

关键设计点:
1. **Fixed 26 keys** across context length —— 控制 key 数量 confounder
2. **无语义** —— 不能靠 language prior 猜
3. **Skewed frequency distribution** —— 模拟真实 language 的 token frequency,避免低频 key 反常
4. **One-to-one unique answer** —— 不依赖 instruction following

[Figure 6](https://arxiv.org/abs/2501.19399) 结果:

| Model | 训练 context | 最长测试 context | 表现 |
|-------|------|------|------|
| 353M Transformer | $2^{12}$ | $2^{17}$ | 超过 training context 就崩 |
| 286M Multiscreen | $2^{12}$ | $2^{17}$ | 几乎 100% accuracy,all lengths & depths |
| **28M Multiscreen** | $2^{12}$ | $2^{17}$ | $2^{17}$ 时仍有 80% accuracy |

最夸张:**28M Multiscreen(只有 353M Transformer 8% 的 params)在 training context length 上都 beat 353M Transformer**。

这说明 retrieval ability 主要由 mechanism 决定,和 parameter count 关系不大。Softmax normalization 本身在 retrieval 上是个 fundamental bottleneck。

为什么 screening 这么强:
- Exact rejection 让 "找不到就是找不到",不被 noise 分散
- $\alpha_{ij}$ 不 normalize,context length 增加 target contribution 不变 —— length invariant
- Long-range 时 MiPE 关闭,纯 content addressing
- TanhNorm 让 "找到" vs "没找到" 信号保持 discriminative

### 5. Inference latency:快 3 倍

[Table 4](https://arxiv.org/abs/2501.19399) 在 100K context 下 next-token prediction:

- 353M Transformer: 4.04s
- 286M Multiscreen: 1.72s (base), 1.26s (continual)

快 2.3-3.2 倍。

Continual pretraining 后进一步加速,是因为更多 tile 的 learned window 仍然 finite(可以在 window 外 skip computation)。Base 时 9.4% tile 是 $w = \infty$,continual 后降到 4.7%。

---

## 更深的联想

### 联想 1:这可能是 induction head 的直接 architectural 实现

[Anthropic 的 induction heads 论文 (Olsson et al., 2022)](https://arxiv.org/abs/2209.11895) 说 in-context learning 主要靠特定 head 做 "前缀 matching"。

Screening 的 Trim-and-Square 本质就是 explicit prefix matching —— $s_{ij} \to 1$ 表示 query 和 key 几乎 same direction,正是 induction 需要的。Multiscreen 在 architectural level 就 enable 这个 behavior,不需要靠 training 涌现。

这可能解释为什么 28M Multiscreen 能 beat 353M Transformer —— 大 Transformer 要花很多 capacity 学习 "怎么从 softmax noise 中 extract signal",而 Multiscreen architectural 就 avoid 这个问题。

### 联想 2:linear attention 的另一条路

[Linear attention (Performer 等)](https://arxiv.org/abs/2009.14794) 用 kernel trick 把 attention 变 $O(N)$,但还是 normalize。

Screening 不 normalize,思路完全不同。但 Trim-and-Square 如果能写成 kernel form,也许能 combine linear time + exact rejection,这是个有趣的 future direction。

### 联想 3:不同iable retrieval / memory-augmented networks 的复活

[Neural Turing Machines (Graves et al., 2014)](https://arxiv.org/abs/1410.5401) 早就想做 differentiable memory access,但当时 optimization 难,没 scale up。

Screening 其实是一种 working 的 differentiable retrieval —— bounded similarity + threshold + non-normalized aggregation。也许这个 paradigm 可以 extend 到 explicit memory-augmented LLM。

### 联想 4:和 MoE 的关系

每个 tile 有自己的 learned window,不同 tile 处理不同时间 scale —— 这有点像 implicit MoE。但 routing 是 static(每个 tile 都 activate),不是 data-dependent。

如果加一个 data-dependent tile selection,可能进一步 reduce computation,同时保留 retrieval 能力。这是个有趣的 future direction。

### 联想 5:value normalization 的意义

把 value 也 normalize 是个 deep 选择。作者引用 [Kobayashi et al., 2020](https://aclanthology.org/2020.emnlp-main.346/) 和 [Guo et al., 2024](https://arxiv.org/abs/2406.04246) 说 attention 的实际效果经常被 value norm dominate,而不是 attention weight。

传统 attention 分析盯着 attention weight,但其实 output 是 $\sum w_j v_j$,$v_j$ 的 norm 影响 output norm,经常被忽视。Screening 把这个 confound 消除,所有 value 都是 unit vector,attention 的 magnitude 完全由 $\alpha$ 控制。这让 attention pattern 的可解释性大大提升。

---

## 潜在的问题和 limitation

1. **Trim 的 spiky gradient**:虽然实验稳定,但在更大规模 / 更难任务上 threshold 附近的 gradient 可能 problematic。Square 是个 mitigation,但不一定够。

2. **$s_{ij} \in [-1, 1]$ 的表达力损失**:unit norm 让 attention 失去 "magnitude-based" 表达。如果某些 key 应该因 magnitude 大而更重要,这个信息被 lost。Softmax 的 unbounded score 给 model 更大 expression space。

3. **ABCDigits 是 screening-friendly benchmark**:fixed-key retrieval 正好是 screening 设计的目标 scenario。真实 language modeling 上 Multiscreen 只省 40% params,优势小很多。需要更难的 reasoning benchmark 验证。

4. **Training cost 仍高**:虽然 inference 快,training 仍然是 $O(N_L \cdot N_H \cdot T \cdot w)$,如果大部分 $w$ 是 $O(T)$ 还是 quadratic。

5. **Weight tying 的潜在限制**:output head 和 input embedding 共享,可能限制 value representation 的表达空间。但实验没看到问题,说明实际 OK。

6. **4B 模型 undertrained**:作者承认 4B 偏离 scaling curve,因为 token budget 不够。40% fewer params 这个 claim 在更大规模上是否仍成立,需要更多实验。

---

## 一个 meta-level 的 insight

我觉得这篇 paper 最 deep 的 contribution 是 challenge 了 attention 的 "weight competition" 这个 default 假设。

从 [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) 以来,几乎所有 attention 变体 —— [sparsemax](https://arxiv.org/abs/1602.02068), [entmax](https://arxiv.org/abs/1905.05702), [linear attention](https://arxiv.org/abs/2009.14794), [Mamba](https://arxiv.org/abs/2312.00752), [RetNet](https://arxiv.org/abs/2307.08621) —— 都默认 normalize across keys。

Screening 提出一种 alternative paradigm:用 explicit threshold 控制"是否纳入",用 non-normalized aggregation 控制"总和"。这把 attention 从 "weighted average" 重构为 "filter + sum"。

这个 paradigm shift 的 implications 可能比具体 architecture 更深远。可能的 future:
- Learned threshold dynamics(让 threshold 在 layer 间自动调整)
- Hierarchical screening(粗筛 + 精筛)
- Cross-modal screening(VLM, video-text 等)
- Differentiable retrieval database(把 screening 拓展到 explicit memory)

---

## 一句话总结

**Softmax attention 是 "必须平均" 的 attention,screening 是 "可以拒绝" 的 attention。**

这个看似简单的区别,让 Multiscreen 在 parameter efficiency、training stability、long-context retrieval、inference speed 四个维度同时提升,而且 28M 模型在 retrieval 上 beat 353M Transformer —— 说明 mechanism 比 parameter count 更重要。

参考链接汇总:
- [Paper: Screening Is Enough](https://arxiv.org/abs/2501.19399)
- [Scalable-Softmax (前作)](https://arxiv.org/abs/2501.19399)
- [Sigmoid attention](https://arxiv.org/abs/2411.06691)
- [Sparsemax](https://arxiv.org/abs/1602.02068)
- [Entmax](https://arxiv.org/abs/1905.05702)
- [RoPE](https://arxiv.org/abs/2104.09864)
- [NoPE](https://arxiv.org/abs/2305.19466)
- [Mamba](https://arxiv.org/abs/2312.00752)
- [RetNet](https://arxiv.org/abs/2307.08621)
- [Induction heads](https://arxiv.org/abs/2209.11895)
- [Lost in the middle](https://arxiv.org/abs/2307.03172)
- [NormFormer](https://arxiv.org/abs/2110.09456)
- [GLU variants](https://arxiv.org/abs/2002.05202)
- [Value norm matters](https://arxiv.org/abs/2406.04246)
- [Original Transformer](https://arxiv.org/abs/1706.03762)

希望这版人话版能帮 build intuition,Andrej。如果哪个点想再 dive deep —— 比如 TanhNorm 的 $\tanh$ 为什么不用其他 saturating function,或者 Trim-and-Square 的 threshold $1/r$ 为什么用 exponential parameterization —— 我们可以继续展开。

---

# Multiscreen: Screening Is Enough 深度解析

好,Andrej,这篇 paper 我看了几遍,挺有意思的,核心 idea 其实很简洁,但我觉得它的深层 implications 可能比作者自己强调的还要更广。我把理解铺开来跟你聊一下,顺便把相关的联想也丢进来。

---

## 1. 核心 motivation:为什么 softmax attention 有问题

作者的出发点是 attention 的 normalization 本身就有结构性的缺陷。Standard softmax attention 的公式:

$$
\text{Attn}(Q,K,V)_i = \sum_j \text{softmax}_j\left(\frac{q_i k_j^\top}{\sqrt{d_K}}\right) v_j = \sum_j \frac{e^{s_{ij}}}{\sum_{j'} e^{s_{ij'}}} v_j
$$

这里 $s_{ij} = q_i k_j^\top / \sqrt{d_K}$ 是 unbounded 的 attention score。Softmax 之后 attention weights $\in [0,1]$ 且 $\sum_j w_{ij} = 1$。

关键问题:这个 $\sum_j w_{ij} = 1$ 的约束意味着 **relevance 永远是 relative 的**。一个 key 收到大 weight,是因为它相对其他 keys 的 score 高,而不是因为它通过了一个 absolute threshold。

这导致几个具体的 pathology:

**(a) 无法 reject irrelevant keys**:即使所有 keys 都和 query 无关,softmax 还是会分配总和为 1 的 weight。模型不能 "说没有相关 context"。

**(b) Context dilution**:context 变长时,相关 keys 的 weight 会被越来越多无关 keys 稀释。具体地说,假设有一个 strong signal key $s^* = 10$,而其他 $N$ 个 keys 是 random noise $\sim \mathcal{N}(0, \sigma)$,那么 relevant key 的 attention weight 大致是:

$$
w^* \approx \frac{e^{s^*}}{e^{s^*} + N \cdot \mathbb{E}[e^s]} \to 0 \quad \text{as } N \to \infty
$$

这其实和 [Scalable-Softmax (Nakanishi, 2025)](https://arxiv.org/abs/2501.19399) 想解决的 "attention fading" 是同一个问题的不同视角。

**(c) 无法 represent absence of context**:context retrieval 的输出应该是一个 length-invariant 的 representation,但 softmax 的输出 norm 完全由 $v_j$ 决定,没有内置的 "我什么都没找到" 信号。

你的 own intuition 在 [Let's build GPT from scratch] 里面讲 attention 的时候也提过 softmax 是 "soft dictionary lookup",但其实它更准确说是 "soft weighted average" — 它强制你要 average 一些东西。这个强制性是这篇 paper 要打破的。

---

## 2. Screening unit 的设计:从 normalization 跳到 thresholding

Screening 的核心思想:对每个 key 独立评估,通过 absolute threshold 决定要不要它,然后对 survivors 加权求和,**不 normalize**。

### 2.1 让 similarity bounded:unit length normalization

第一步把 $q_i, k_j, v_j$ 都 normalize 到 unit length:

$$
\bar{q}_i = \frac{q_i}{\|q_i\|}, \quad \bar{k}_j = \frac{k_j}{\|k_j\|}, \quad \bar{v}_j = \frac{v_j}{\|v_j\|}
$$

这一步让 $s_{ij} = \bar{q}_i \bar{k}_j^\top \in [-1, 1]$ 严格 bounded。这非常关键,因为后面的 threshold 操作需要一个 well-defined 的 scale。如果 $s_{ij}$ 还是 unbounded,threshold 就变成 data-dependent 的东西,失去了 "absolute" 的意义。

这个 unit normalization 在 [NormFormer (Wu et al., 2021)](https://arxiv.org/abs/2110.09456) 里也讨论过,但那里主要是为了 scale。这里是为了定义 absolute relevance。

**Value 也归一化**很有意思。作者引用 [Kobayashi et al., 2020](https://aclanthology.org/2020.emnlp-main.346/) 和 [Guo et al., 2024](https://arxiv.org/abs/2406.04246) — value norm 经常 dominate attention 的实际效果,这一直是 attention 可解释性研究里被忽视的点。

### 2.2 MiPE:Minimal Positional Encoding

MiPE 是个非常聪明的设计。它把 RoPE 只应用到 query/key 的前两个 coordinates:

$$
M_i(w) = \begin{pmatrix} R(\phi(i,w)) & 0 \\ 0 & I_{d_K - 2} \end{pmatrix}
$$

其中 $R(\phi)$ 是 2D rotation matrix,$\phi(i, w) = \frac{\pi i \gamma(w)}{w}$。

关键点:$\gamma(w)$ 是个 smooth switch,当 $w < w_{\text{th}}$ 时启用,当 $w \geq w_{\text{th}}$ 时为 0:

$$
\gamma(w) = \begin{cases} \frac{1}{2}(\cos\frac{\pi w}{w_{\text{th}}} + 1), & w < w_{\text{th}} \\ 0, & w \geq w_{\text{th}} \end{cases}
$$

这个设计的 intuition:
- 当 screening window $w$ 小(局部交互)时,你需要 locality 信息,positional encoding 启用
- 当 window $w$ 大(全局 retrieval)时,你想做 content-based addressing,位置信息会干扰 retrieval,所以 MiPE 自动关闭

这其实呼应了 [NoPE (Kazemnejad et al., 2023)](https://arxiv.org/abs/2305.19466) 和 [NoPE length generalization (Wang et al., 2024)](https://aclanthology.org/2024.findings-acl.271/) 的思路 — 远距离依赖根本不需要 explicit position。但这里更精细:不是完全不用 position,是 **条件性** 用。

注意作者用 $w_{\text{th}} = 256$ 在实验里,这是个 hyperparameter。直觉上应该和 base pretraining 的 context length $2^{12} = 4096$ 有关系,可能是一个小的局部范围。

### 2.3 Trim-and-Square:absolute threshold

这是 screening 的核心操作:

$$
\alpha_{ij} = \left[\max\left(1 - r(1 - s_{ij}), 0\right)\right]^2
$$

让我分解一下:
- $s_{ij} \in [-1, 1]$
- 当 $s_{ij} \leq 1 - 1/r$ 时,$1 - r(1 - s_{ij}) \leq 0$,被 clamp 到 0,所以 $\alpha_{ij} = 0$ — **exact rejection**
- 当 $s_{ij} > 1 - 1/r$ 时,$\alpha_{ij}$ 是 $\geq 0$ 的 quadratic
- 在 $s_{ij} = 1$ 时 $\alpha_{ij} = 1$ 达到 max

这里 $r = e^{s_r} + 1$ 是 learned parameter,定义 "acceptance width" $1/r$。$r$ 越大,threshold 越严格,只接受 $s_{ij}$ 非常接近 1 的 keys。

为什么是 **square** 而不是线性?我推测是 smoothness + gradient considerations。Square 让在 threshold 边缘的 gradient 平滑过渡到 0,避免 boundary discontinuity。同时 quadratic 增长让接近 perfect match 的 keys 被额外 emphasized。这和 [entmax](https://arxiv.org/abs/1905.05702) 的 sparsity-inducing 思路有点像,但完全 different — 这里不 normalize。

**关键差异 vs sigmoid attention**: [Sigmoid attention (Ramapuram et al., 2025)](https://arxiv.org/abs/2411.06691) 也去掉了 normalization,但 sigmoid 严格 $>0$,无法 exact reject,而且需要 sequence-length-dependent bias 来控制 magnitude,这就破坏了 length invariance。Screening 的 trim 让 weight 可以严格 = 0,而且因为每个 key 独立,context length 不改变 $\alpha_{ij}$ 本身。

### 2.4 Softmask:learned screening window

距离相关的 soft mask:

$$
m_{ij}(w) = \begin{cases} \frac{1}{2}\left(\cos\frac{\pi(j-i)}{w} + 1\right), & -w < j-i \leq 0 \\ 0, & \text{otherwise} \end{cases}
$$

注意 causal:只有 $j \leq i$ 贡献,window 范围 $[-w, 0]$。

然后:
$$
\alpha_{ij}^d = \alpha_{ij} \cdot m_{ij}(w)
$$

要 survive 必须同时通过 content-based screening AND distance-based screening。这非常像一种 AND operation。

$w = e^{s_w} + 1$ 是 learned per-tile,初始化时在 heads 之间 linearly spaced 从 0 到 $\log w_{\text{th}}$。这意味着不同 heads 从一开始就被鼓励去 explore 不同的时间 scale。

**Inference trick**:如果 learned $w$ 超过 training 时的 max sequence length,直接 set $w = \infty$,即全 causal。这避免了在 window 边界做无意义计算,同时让 model 可以 **自动决定** 哪些 tile 需要全局 attention,哪些可以局部。

### 2.5 TanhNorm:bounded aggregation without normalization

最后的 aggregation:

$$
h_i = \sum_j \alpha_{ij}^d \bar{v}_j, \quad u_i = \text{TanhNorm}(h_i) = \frac{\tanh(\|h_i\|)}{\|h_i\|} h_i
$$

TanhNorm 保持 $h_i$ 的方向,但把 norm 平滑 bounded 到 1。$\tanh$ 在小 norm 时近似 identity,在大 norm 时饱和。

这是非常聪明的设计:
- 它 **不 normalize** $\alpha$ 的和,所以可以 represent "nothing found"(norm 接近 0)
- 但又防止 norm 在大 context 下爆炸(尤其当 survivors 很多时)
- 它的 "bounded but not normalized" 性质让 representation 有 absolute scale 信息

如果对比 softmax 输出(永远是 $\sum w_j v_j$,norm 取决于 $v_j$ 的 norm),TanhNorm 让 representation 的 norm 实际上 encode 了 "有多少 relevant context 被找到" 这个信息。

---

## 3. Gated Screening Tile:把 attention 和 FFN 合并

这部分我觉得是这篇 paper 里最被低估的设计之一。

传统的 Transformer block 是:
```
x → Attention(x) → x + Attn_out → FFN(x + Attn_out) → x' + FFN_out → ...
```

Multiscreen 把 attention 和 FFN 合并成一个 gated unit:

$$
q_i = x_i W_Q, \; k_i = x_i W_K, \; v_i = x_i W_V, \; g_i = x_i W_G
$$
$$
\hat{g}_i = \tanh(\text{SiLU}(g_i)) \quad \text{(gate)}
$$
$$
h_i = u_i \odot \hat{g}_i \quad \text{(GLU-style)}
$$
$$
\Delta x_i = h_i (e^{s_O} W_O)
$$

这其实是把 [GLU variants (Shazeer, 2020)](https://arxiv.org/abs/2002.05202) 的 gating 推广了 — linear transform 被替换成 screening-based aggregation。所以一个 tile 同时承担:
- **Context retrieval**(screening unit)
- **Feature selection**(gate)
- **Output projection**

整个 layer 是 $N_H$ 个 parallel tiles 求和:

$$
x_i^{(\ell)} = x_i^{(\ell-1)} + \sum_{h=1}^{N_H} \Delta x_i^{(\ell, h)}
$$

没有 sequential attention → FFN 的两步,直接 parallel merge。这和最近 [Gemini-like parallel attention/FFN](https://arxiv.org/abs/2305.13245) 的设计是同一个方向,但更彻底。

### 3.1 Scaling rule:supraparameter $\Psi$

作者用一个 single scaling parameter $\Psi$ 控制 $N_L = N_H = \Psi$,$d_E = \Psi^2$。

这是个很 aggressive 的 scaling — 模型 size 大致 $\propto \Psi^3$。

对比 Transformer 的常见 scaling(比如 GPT-3 里 layer 数增长比 dim 快),Multiscreen 的 scaling 让 layer, head, dim 比较均衡。我推测这是因为每个 tile 都同时做 retrieval 和 gating,功能比单纯 attention 更 "dense",所以需要适度减小每层的 depth 而增加 layer 数。

---

## 4. 实验结果解读

### 4.1 Scaling efficiency:40% fewer params

[Figure 3](https://arxiv.org/abs/2501.19399) 显示在 8M, 18M, 28M, 45M, 124M, 286M, 353M, 4B 这个 range 内,Multiscreen 的 scaling curve 大致平移到 Transformer 的 0.6× parameter 位置。即 ~40% fewer params for same loss。

这是非常显著的提升。对比:
- [Mamba (Gu & Dao, 2024)](https://arxiv.org/abs/2312.00752):在某些 scaling 区间接近 Transformer
- [Hyena (Poli et al., 2023)](https://arxiv.org/abs/2302.10866):一般比 Transformer 略差
- [RetNet (Sun et al., 2023)](https://arxiv.org/abs/2307.08621):声称 inference 优势

40% fewer params 是一个相当 striking 的 gap。但要小心 — token budget fixed,所以大模型是 undertrained 的。作者承认 4B 模型偏离 scaling curve,就是因为 undertraining。

**Intuition**:为什么 screening 会更 parameter efficient?我推测有几个原因:
1. 每个 tile 同时做 attention + FFN 的工作,等于减少 sequential computation
2. Absolute relevance 让 model 不需要学习 "如何在 normalize 之后还能保留 strong signals" 这种比较 tricky 的 thing
3. TanhNorm 提供了一个 stable 的 output scale,可能让 residual stream 的信息流动更 efficient

### 4.2 Learning rate stability

[Figure 4](https://arxiv.org/abs/2501.19399) 显示 Transformer 在 LR $\geq 2^{-4} = 0.0625$ 时 diverge,而 Multiscreen 在 LR $2^{-4}$ 甚至 $2^0 = 1$ 时仍稳定。

这是个非常 dramatic 的结果。LR $2^{-4} = 0.0625$ 对 28M model 来说是非常大的 LR。而 Transformer 在 45M 用 $10^{-3}$ 都还是接近 optimal。

作者的解释是 "no competition across keys"。我自己思考一下 — softmax 的竞争性让它对 perturbation 非常敏感:如果某个 key 的 score 异常大,它会 "steal" 所有的 attention mass,导致其他 keys 完全被 ignore,这种 "winner-take-all" 的 dynamics 是 non-smooth 的,大 LR 下容易 blow up。

Screening 每个 key 独立,perturbation 一个 key 不影响其他 keys,所以 optimization landscape 更 smooth。

[Appendix D](https://arxiv.org/abs/2501.19399) 的 gradient norm 数据支持这个观察:Multiscreen 的 gradient norm 快速 decay 到接近 0,而 Transformer 维持一个 non-zero gradient floor with spikes。这其实暗示 Multiscreen 训练得 "更彻底"(gradient 接近 0 表示已经 reach 一个 flat region)。

不过这里也有一个隐忧:gradient norm 接近 0 也可能意味着 capacity underutilization。需要看最终 loss 是否真的 low。

### 4.3 Long-context perplexity (PG-19)

[Figure 5](https://arxiv.org/abs/2501.19399) 是 long-context perplexity 的对比。

- Training context:base = $2^{12} = 4096$, continual = $2^{15} = 32K$
- Evaluation context:up to $2^{17} = 131K$

Transformer 在超过 training context 后 perplexity 急剧上升,即使 RoPE scaling factor 增加到 ×64 也只是延缓。Multiscreen 平稳。

直觉:RoPE 是个 inductive bias,它假设相对位置 $i-j$ 的 rotation pattern 可以 extrapolate,但实际上在 training length 之外的 rotation pattern 没见过,model 没法 generalize。MiPE 通过 **让 long-range behavior 不依赖 position** 来 bypass 这个问题 — 当 $w$ 大时 position 自动 disable,所以没有 "extrapolate position pattern" 这个问题。

### 4.4 ABCDigits:无语义 retrieval benchmark

这个 benchmark 设计得很妙。任务:
- 给 26 个 uppercase letters 各 assign 一个 n-digit number(如 A=967892)
- 构造一个 shuffled list,其中 target letter 只出现一次,在 specified depth
- Query 是 "L=",model 要 complete 这个 number

关键设计:
1. **Fixed 26 keys** across all context lengths → 控制 "key 数量" 这个 confounder
2. **无语义** → 不能用 language modeling priors 模糊地 "猜"
3. **One-to-one unique answer** → 不依赖 instruction following
4. **Skewed distribution** → 模拟自然 language 中 token frequency 的分布,让低频 key 不显得 anomalous

[Figure 6](https://arxiv.org/abs/2501.19399) 的结果非常 striking:

- **286M Multiscreen**:在所有 context lengths(从 $2^{12}$ 到 $2^{17}$),所有 depths(0.1 to 0.9),accuracy 都接近 100% — 即使是 base model(没 continual pretraining)
- **28M Multiscreen**:在 $2^{17} \approx 131K$ context 下还能保持 ~80% accuracy
- **353M Transformer**:即使选 best RoPE scaling factor,在 context length 超过 training length 后迅速崩溃

最 striking 的对比:**28M Multiscreen(92% fewer params)在 training context length 上都 outperform 353M Transformer**。这是一个 capacity vs architecture 的故事 — retrieval ability 不是单纯 parameter count 决定的,而是 mechanism 决定的。

为什么 Multiscreen 这么强?我的理解:
1. Screening 的 exact rejection 让它 "找不到就找不到",而 softmax attention 永远会被 noise keys 分散 attention
2. 因为 $\alpha_{ij}$ 不 normalize,context length 增加,只要 target key 还在,它的 contribution 不会变 — 完全 length-invariant
3. MiPE 在大 window 时 disable,所以纯 content-based addressing 不被 position 干扰
4. TanhNorm 的 bounded output 让 "找到" vs "没找到" 的信号保持 discriminative

对比 Transformer 的 "needle in haystack" 问题:即使 attention 权重能找到 needle,softmax normalize 之后 needle 的 weight 还是会被大量 haystack 稀释,导致 value contribution 被淹没。这是 [Lost in the middle (Liu et al., 2024)](https://arxiv.org/abs/2307.03172) 现象的根源。

### 4.5 Inference latency

[Table 4](https://arxiv.org/abs/2501.19399) 在 100K context 下的 next-token prediction latency:

| Model | Base | After continual pretraining |
|-------|------|------|
| 353M Transformer | 4.04s | 4.05s |
| 286M Multiscreen | 1.72s | 1.26s |

Multiscreen 快 2.3-3.2×。

Continual pretraining 后 Multiscreen 进一步加速,是因为更多 tile 的 learned window $w$ 仍然 < 训练时 max seq len,所以保持 finite window,而 finite window 可以跳过 window 外的 computation。Base 模型只有 9.4% tile 是 $w = \infty$,continual 后 4.7%。

这是 screening 的另一个 side benefit — learned window 给了一种 **adaptive sparsity**,不需要 explicit sparse attention 的复杂 dispatching logic。

---

## 5. 更深的联想和潜在问题

### 5.1 联想:这和 efficient attention / linear attention 的关系

Linear attention(比如 [Performer (Choromanski et al., 2020)](https://arxiv.org/abs/2009.14794))用 kernel trick 把 attention 变成 $O(N)$。但 linear attention 仍然 normalize across keys(softmax 被替换成 kernel mean)。

Screening 不 normalize,所以和 linear attention 思路不同。但有可能 fuse — 如果 Trim-and-Square 可以写成 kernel 形式,screening 也可能 linear-time。

### 5.2 联想:这和 MoE 的关系

每个 tile 有自己的 learned window,所以不同 tile 处理不同时间 scale。这有点像 implicit MoE — 不同 tile "expert" 处理不同 context range。但 routing 不是 data-dependent(每个 tile 都被 activate),所以不是严格 MoE。

如果加一个 data-dependent tile selection,可能可以进一步 reduce computation。

### 5.3 联想:这和 induction heads 的关系

[Anthropic 的 induction heads 工作 (Olsson et al., 2022)](https://arxiv.org/abs/2209.11895) 表明 in-context learning 主要由特定 head 做 "前缀 matching" 实现。Screening 的 Trim-and-Square 本质上就是 explicit prefix matching — $s_{ij}$ 接近 1 表示 query 和 key 几乎 same direction,正是 induction 需要的。所以 Multiscreen 可能在 architectural level 直接 enable induction-like behavior,这可能是它 retrieval 能力强的 deep reason。

### 5.4 潜在问题

**(a) Trim 的 spiky gradient**:$[\max(\cdot, 0)]^2$ 在 threshold 附近 gradient 是连续的但导数变化快。虽然 paper 报告训练稳定,但在大模型/更复杂任务上会不会有 issue 还需要验证。

**(b) $s_{ij}$ bounded 到 $[-1,1]$ 的代价**:unit normalization 让 attention 失去了 "magnitude-based" 的表达力。如果某些 key 应该因为 magnitude 大而更重要,这个信息被 lost。Softmax 虽然有 normalization issue,但它的 unbounded score 让模型有更大表达空间。

**(c) Long context training cost**:虽然 inference 快,但 training 仍然是 $O(N_L \cdot N_H \cdot T \cdot w)$,如果 $w$ 是 $O(T)$ 还是 quadratic。Continual pretraining 用 $2^{15}$ context,实际训练 cost 仍然高。

**(d) Output 绑定到 input embedding**:weight tying 在 paper 里强调了,但其实对 long-context retrieval 可能是双刃剑。如果 value representation 需要 diverge from token embedding,这种 tying 会限制表达。但 paper 报告结果好,说明实际问题不大。

**(e) ABCDigits 设计是否太 aligned with screening**:ABCDigits 是 fixed-key retrieval,正好是 screening 设计的目标 scenario。在真实 language 任务里,retrieval 不一定有明确的 key-value structure。Validation loss 只差 40% params 这个结果显示在 general language modeling 上,Multiscreen 优势小很多。可能需要一个更难的 reasoning benchmark 来看 screening 的真实威力。

### 5.5 推广:Meta-level insight

我觉得这篇 paper 最 deep 的 contribution 是把 attention 的 "weight competition" 这个 default 设计 challenge 了。从 softmax attention 出现以来,几乎所有 attention 变体都假设需要 normalize across keys。Screening 提出一种 **alternative paradigm**:用 explicit threshold 而不是 normalization 来 control 求和。

这个 paradigm shift 可能比具体 architecture 更有影响力。可能的 future direction:
- **Learned threshold dynamics**:让 threshold 在不同 layer 自动调整
- **Hierarchical screening**:多层 screening,先粗筛后精筛
- **Cross-modal screening**:把 screening 用到 vision-language model 等
- **Differentiable retrieval**:screening 本质上是 differentiable database lookup,可能拓展到 explicit memory-augmented networks(如 [Neural Turing Machines (Graves et al., 2014)](https://arxiv.org/abs/1410.5401))

---

## 6. 总结:核心 takeaways

1. **Softmax attention 的核心 limitation** 不是 quadratic cost,而是 normalization 强制了 relative relevance,导致长 context 下 dilution 和 inability to reject。

2. **Screening 的 core fix** 是通过 (a) bounded similarity (unit norm) + (b) explicit threshold (Trim-and-Square) + (c) non-normalized aggregation 实现 absolute relevance。

3. **MiPE 是关键 trick**:position 只在 short-range 时启用,让 long-range behavior 不依赖 position extrapolation。

4. **Gated screening tile** 把 attention 和 FFN 合并成 single module,既 reduce sequential compute 又把 GLU 的 inductive bias 嵌入到 attention 里。

5. **Empirically**:40% fewer params,LR stability 极大改善,retrieval 在 4× training context 仍 perfect,inference 快 3×。

6. **Deep intuition**:retrieval ability 不是 parameter count 决定的,而是 mechanism 决定的。一个 28M 的 screening model 在 retrieval 上能 outperform 353M Transformer,说明 softmax normalization 本身在 retrieval 任务上是个 fundamental bottleneck。

7. **Open question**:ABCDigits 是 screening-friendly 的 benchmark,真实 LM 任务上 screening 是否仍 dominate,需要看 reasoning / in-context learning benchmark 上的结果。

参考链接:
- [Paper: Screening Is Enough](https://arxiv.org/abs/2501.19399)
- [Scalable-Softmax (同一作者前作)](https://arxiv.org/abs/2501.19399)
- [Sigmoid attention (Ramapuram et al.)](https://arxiv.org/abs/2411.06691)
- [Sparsemax / entmax](https://arxiv.org/abs/1602.02068)
- [GLU variants (Shazeer)](https://arxiv.org/abs/2002.05202)
- [RoFormer / RoPE (Su et al.)](https://arxiv.org/abs/2104.09864)
- [NoPE (Kazemnejad et al.)](https://arxiv.org/abs/2305.19466)
- [Mamba (Gu & Dao)](https://arxiv.org/abs/2312.00752)
- [Induction heads (Olsson et al., Anthropic)](https://arxiv.org/abs/2209.11895)
- [Lost in the middle (Liu et al.)](https://arxiv.org/abs/2307.03172)
- [NormFormer](https://arxiv.org/abs/2110.09456)
- [Value norm matters (Guo et al.)](https://arxiv.org/abs/2406.04246)

希望这个解读对你 build intuition 有帮助,Andrej。如果你对某个细节想再 dive deeper — 比如 TanhNorm 为什么这样设计,或者 MiPE 的 $\gamma$ 函数为什么用 cosine 而不是其他 — 我们可以继续展开。
