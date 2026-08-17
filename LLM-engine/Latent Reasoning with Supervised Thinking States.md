---
source_pdf: Latent Reasoning with Supervised Thinking States.pdf
paper_sha256: 10cc4f211e4f45ea0baaa009076c2a2fbf83d7a70aaa687b71cc5ba8d4cee5cf
processed_at: '2026-08-05T12:16:04-07:00'
target_folder: LLM-engine
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好,我用人话给你捋一遍,Andrej。

## 一句话版本

CoT 太贵,大家想把 reasoning 压成 latent representation。之前的方案要么 supervision 困难 (Coconut 要 BPTT,train 不动),要么 expressivity 不够 (iCoT 把 reasoning 蒸进 query embeddings,parallel 处理 → 没法 sequential reasoning)。这篇 paper 的 insight 很 simple：让 thoughts **依然是 natural language tokens** (这样能用 teacher-forcing 监督、能 parallel train),但 **不 append 到 context** (用 fixed-size state 注入回 future chunks),通过 deep-to-shallow recurrence 实现 thought 对 future tokens 的 conditioning。

reference: https://arxiv.org/abs/2412.06769 (Coconut baseline)

---

## 为什么 previous work 卡住了

你想想 latent reasoning 这个赛道的两个极端：

**Coconut** (Hao et al., 2024) 用 continuous embeddings 当 thought tokens。听起来 elegant,信息密度高。但问题是,这些 latents 没有监督信号 — 你不知道 target 该是什么,只能从 final answer loss 反向传,gradient 要穿过所有 recurrent steps,这就是 BPTT。后果很 brutal: Figure 6 显示 10 个 recurrent steps 时训练成本已经 10× 膨胀,实际最多只能用 6-21 个 latent tokens (Table 3,加更多 latent 准确率不涨反平)。BPTT 本质上把你的 recurrent reasoning 深度 lock 死在很浅的水平。

**iCoT** (Deng et al., 2024) 走另一条路：通过 curriculum learning,逐步把 CoT reasoning 蒸馏进 query 自身的 representation。好处是 parallel training,问题是 query tokens 在 Transformer 里是 **同时** 处理的,sequential reasoning 只能靠 model 的固定 depth。depth 是死的,reasoning steps 多了就 express 不出来。Table 1 的 Vars task N=10,这种隐式方法基本 2% 准确率,完全废掉。

所以设计空间里有个 tradeoff：你要 recurrence (sequential conditioning) 就得 BPTT,你要 parallel training 就丢掉 recurrence 的表达力。

---

## Thinking States 怎么破局

核心 insight 就一句话：**thoughts 用 natural language 表示,但不进 context window**。

具体怎么实现,你 follow 数据流走一遍就懂了：

输入 $L$ 个 tokens,切成 $K$ 个 chunks,每个 chunk $c = L/K$ 个 tokens (实验里默认 $c=8$)。

对第 $i$ 个 chunk,先做一件很简单的事:

$$\tilde{\mathbf{X}}_i = \mathbf{X}_i + \mathbf{S}_i$$

这里 $\mathbf{X}_i \in \mathbb{R}^{c \times d}$ 是 chunk 在 shallow layer (第 1 层之后) 的 embeddings,$\mathbf{S}_i \in \mathbb{R}^{c \times d}$ 是上一个 chunk 产出的 thinking state,$\mathbf{S}_1 = \mathbf{0}$。就是 element-wise addition,residual 调制,类似 adapter 但不加参数。

然后正常 forward 过 backbone:

$$\mathbf{H}_i^{out} = M_\theta(\tilde{\mathbf{X}}_i \mid \tilde{\mathbf{X}}_{<i})$$

$\mathbf{H}_i^{out}$ 是 chunk 在 deep layer (倒数第二层) 的 hidden states,历史 chunks 通过 KV-cache 访问,不重算。注意这步 **同时** 服务两个目的：next-token prediction 和 thought generation 共享同一次 forward,这就是 paper 说的 "compute sharing"。

接下来,从 $\mathbf{H}_i^{out}$ 生成 natural language thoughts:

$$\mathbf{Z}_{i+1} = T(\mathbf{H}_i^{out})$$

$T$ 是个 1-layer Transformer decoder,很 lightweight。但它吃的不是 raw embeddings,是 deep layer 的 rich representations,所以即使 1 层也能产出有意义的 reasoning tokens。$\mathbf{Z}_{i+1} = (z_1, \ldots, z_n)$ 是变长 token sequence,以 `<EOS>` 结尾。如果这个位置不需要 reasoning,$\mathbf{Z}_{i+1}$ 就只有 `<EOS>` — 这点对后面 inference 加速至关重要。

最后,把变长 thoughts 压回 fixed-size state:

$$\mathbf{S}_{i+1} = C(\mathbf{Z}_{i+1}) \in \mathbb{R}^{c \times d}$$

$C$ 是 1-layer Transformer encoder + pooling,取最后 $c$ 个 contextualized representations。循环闭合。

---

## Deep-to-Shallow 是这个架构的灵魂

你可能会问：为什么从 deep layer 提取、在 shallow layer 注入？看起来很 arbitrary。

Figure 3a 的 ablation 直接回答了这个：

| 参与 recurrence 的 layers 数 | GSM Accuracy |
|---|---|
| 浅层 (少 layers 在 loop 里) | ~25% |
| 深层 (多 layers 在 loop 里) | ~42% |

差了 20 个点。

intuition 是这样的：thought 从 deep layer 提取,在 shallow layer 注入,意味着 thought 信息要 **穿越 backbone 几乎所有 layers** 才能影响下一个 thought 的生成。这就相当于让 thought 经历了完整的 computation,类似于 CoT 里每个 thought token 都经过全部 layers 一样。

对比一下 Universal Transformers (Dehghani et al., 2019, https://openreview.net/forum?id=HyzdRiR9Y7) 的思想：用同一个 block 反复应用实现 adaptive compute。Thinking States 没有反复应用同一个 block,但它让 thought 信号穿越整个 model depth,在效果上近似 recurrent depth。

如果你在 shallow layer 注入、在 shallow layer 提取,thought 只经历了几个 layers 就被读出来,reasoning 能力很弱,这跟 iCoT 的困境一样。

---

## Training 的 trick 是这篇 paper 最聪明的地方

前面说的是 architecture,training 才是真正的 magic。

你想,如果 inference 时 $\mathbf{S}_i$ 由 model 生成,那训练时是不是得等 model 生成 $\mathbf{S}_1$ 才能算 $\mathbf{X}_2$,等生成 $\mathbf{S}_2$ 才能算 $\mathbf{X}_3$?这就是 BPTT,Coconut 的噩梦。

Thinking States 的解法是 **teacher-forcing the states**:

$$\tilde{\mathbf{X}}_i = \mathbf{X}_i + \mathbf{S}_i^*, \quad \mathbf{S}_i^* = C(\mathbf{Z}_i^*)$$

$\mathbf{Z}_i^*$ 是 ground-truth reasoning sequence (从 CoT annotation 来),$\mathbf{S}_i^*$ 是用 gold thoughts 压出的 state。训练时 **不用 model-generated state,用 gold state**。

后果：所有 chunks 的输入 $\tilde{\mathbf{X}}_i$ 在训练开始时就完全确定,不依赖 model 的任何输出。所以所有 $\mathbf{H}_i^{out}$ 可以在 **单次 parallel forward pass** 里全部算出。然后 $T$ 在所有 chunks 上并行训练 next-token prediction。

Loss 就是：

$$\mathcal{L} = \mathcal{L}_{LM} + \sum_{i=1}^K \mathcal{L}_T(\mathbf{Z}_i, \mathbf{Z}_i^*)$$

$\mathcal{L}_{LM}$ 是标准 LM loss,$\mathcal{L}_T$ 是 thought sequence 上的 cross-entropy,$K$ 是 chunk 数。每个 $\mathbf{H}_i^{out}$ 都在 gold state history $\mathbf{S}_1^*, \ldots, \mathbf{S}_i^*$ 下计算,所以预测 $\mathbf{Z}_{i+1}$ 隐式条件于所有 prior gold reasoning steps,这跟 CoT 的 sequential conditioning 是等价的。

Figure 6 的对比很直观：BPTT (Coconut) 随 recurrent steps 线性涨成本,10 steps 时 10× penalty;Thinking States 几乎常数,因为 teacher-forcing 把时间链从计算图里去掉了,gradient 只在单次 forward 内流动。

---

## 但 teacher-forcing 需要监督信号,监督信号从哪来

这是另一个被低估的贡献。你手头有 CoT data,长这样:

```
Query: "A coin's state is heads. Alice flips, then Bob flips. What's the state?"
CoT: "heads → tails → heads"
```

Thinking States 需要的是 **chunk-level** supervision — 每个 chunk 对应什么 thought。所以要做 step-to-chunk alignment:

**Step 1**: 找到每个 reasoning step **最早能被推断** 的 query 位置,插入 `<T>` marker:

```
"A coin's state is heads.<T> Alice flips,<T> then Bob flips.<T> What's the state?"
```

每个 `<T>` 标记 "到这个位置为止,信息已经足够推断对应的 reasoning step"。这个 alignment 用 Gemini 2.5-Flash 做 (Figure 7 的 prompt),或者用程序规则 (state tracking 这种结构化任务)。

**Step 2**: tokenize 后按 chunk size $c$ 切分,每个 chunk 的 target = 该 chunk 内所有 reasoning steps 的拼接。空 chunk → `<EOS>`。

这个步骤把 **全局 CoT trajectory** 转化成了 **局部 thought supervision**,使 teacher-forcing 成为可能。没有这步,你拿不到 gold $\mathbf{Z}_i^*$,就得回退到 BPTT。

---

## Inference 怎么办 — Speculative Thinking

训练 parallel 了,inference 还是 sequential,因为 $\tilde{\mathbf{X}}_i$ 依赖 $\mathbf{S}_i$,$\mathbf{S}_i$ 由 model 生成。Naive prefill 要 $O(K)$ 轮串行 forward。

但有个观察：**大多数 chunks 的 state 是 trivial** (只有 `<EOS>`)。GSM 数据里,大部分 query 位置不触发计算 — 只有 "10×12=120" 这种位置才需要 thought。

Algorithm 1 的逻辑:

1. 推测所有 chunks 的 state 都是 trivial ($\mathbf{S}_i = \mathbf{0}$),并行 forward 全部
2. 对每个 chunk 用 $T$ 和 $C$ 生成真正的 state
3. 找到最早产生 non-trivial state 的 chunk $i_1$ — 之前的 chunks 真的是 trivial,计算正确,cache 它们
4. 从 $i_1$ 开始 conditioning on 正确 $\mathbf{S}_{i_1}$,对剩余 chunks 重复 step 1
5. 终止条件：没有新的 non-trivial state

复杂度：恰好 $|R| + 1$ 轮,$|R|$ 是 non-trivial states 数量。当 $|R| \ll K$ (典型情况),prefill 接近单次 parallel pass。

这是 **exact** 算法,不是近似 — trivial state 的 speculation 恰好等于真实值,被 cache 的计算完全正确。本质上就是 speculative decoding 的变种,但 speculation 对象是 "future states 都是 trivial" 而非 "future tokens 来自 draft model"。

reference: speculative decoding 原理可参考 https://arxiv.org/abs/2302.01318

---

## 实验讲了什么故事

### State Tracking — length generalization (Table 1)

训练时只见 N=10 operations,测试到 100 operations：

| Method | Parity N=10 | Vars N=10 | Vars N=40 |
|---|---|---|---|
| No CoT | 54.67 | 02.15 | 02.19 |
| CoT | 12.35 | 06.78 | 87.75 |
| Thinking States | **98.37** | **33.76** | **97.71** |

CoT 在 N=10 训练时只有 12.35%,因为它靠生成中间 token 推理,error 在长序列上累积,长度泛化差。Thinking States 几乎完美,因为 recurrent state update 是位置无关的 — model 学到的是一个 state update function,不是 memorize 特定长度的 trajectory。这呼应 Deletang et al. (2022, https://arxiv.org/abs/2207.02098) 关于 recurrent models 在 Chomsky hierarchy 上表现更好的发现。

### General Reasoning (Table 2)

| Method | GSM Acc | GSM Speedup | 2-Hop FK | 2-Hop PK |
|---|---|---|---|---|
| CoT | 60.50 | ×1 | 54.79 | 43.07 |
| Thinking States | 42.22 | ×2.66 | **54.91** | 43.05 |
| Coconut | 32.65 | ×3.14 | 33.71 | 32.60 |
| iCoT | 34.00 | ×5.71 | 28.84 | 36.31 |

两个故事：

**2-Hop QA 上 match 甚至略超 CoT**。这说明对 "retrieve + compose" 类型任务,latently reasoning 已经够用。Biran et al. (2024, https://arxiv.org/abs/2406.12775) 和 Yang et al. (2024, https://arxiv.org/abs/2411.16679) 显示标准 LLM 已有部分 latent multi-hop 能力,Thinking States 把这个能力放大了。

**GSM 上有 18 点 gap** (42.22 vs 60.50)。数学需要精确多步计算,latently state 压缩会丢中间数值精度。但仍然 **远超** 其他 latent 方法 (Coconut 32.65, iCoT 34.00) — 在 latent reasoning 赛道里这是目前最好的 GSM 结果。而且 Coconut 加更多 latent tokens 也没用 (Table 3: 6→21 tokens, 32.65→32.65,BPTT 优化瓶颈)。

---

## Chunk Size ablation 的 intuition (Figure 3b)

- 小 chunk ($c=1-2$)：每个 state 容量太小,encode 不了复杂 thought;且 iteration 多,latency 差
- 中 chunk ($c=8$)：sweet spot
- 大 chunk ($c>8$)：多个 sequential reasoning steps 被塞进同一 chunk,它们 **无法访问 full recurrent loop**,只能靠 LLM depth + 轻量 $T$ 处理,破坏 deep-to-shallow 设计

chunk size 是 "per-state capacity" 与 "reasoning frequency" 的权衡。太小 → state 信息量不足;太大 → 失去 recurrence 的 sequential composition 优势。

---

## Error Analysis 讲了什么

### Thinking States 成功、CoT 失败 (Figure 4)

两个典型 case：

**Case 1**: CoT hallucinate 额外 step。"Derrick 每天 10 dozen doughnuts, $2 每个, 6 月卖完" → CoT 多算了一步 ×6 (误以为 6 月 = 6 个月)。Thinking States 因为每个 thought anchored 到 query 特定位置,不会自由发挥。

**Case 2**: CoT 过度复杂计算。"17 green, 2× red, total 60, blue=?" → CoT 试图一步算 60-17-34 出错;Thinking States 的 supervision 强制 step-by-step 单操作,正确分步。

insight 是：Thinking States 的 supervision alignment 规定了 "何时算什么",约束了 reasoning trajectory 形状,减少 hallucination。

### Thinking States 失败 — State Ambiguity (Figure 5)

"Richard 住 15 层楼,每层 8 单元,3/4 occupied,每层 unoccupied 数?"

Thinking States 在前面 chunks 已经 commit 到 "总 unoccupied" (120-90=30),但 query 末尾才问 "每层" (正确答案是 8-6=2)。

这是 causal autoregressive processing 的固有限制 — 模型只能 left-to-right,无法回头看最终问题。Zero-shot 修复：把问题 prepend 到 query 开头,准确率 42.22 → 48.65 (+6.43)。Bidirectional query processing 可能根本解决,但会破坏 KV-cache streaming。

---

## 我的 overall intuition

这篇 paper 最 clever 的地方在于它 **解耦了三个看似绑定的东西**：

1. "thoughts 是 natural language" 与 "thoughts 进 context window" — 解耦后,前者给你 supervision signal 和 interpretability,后者省 context length
2. "recurrent conditioning" 与 "BPTT training" — 解耦后,用 teacher-forcing 拿到 parallel training,inference 时保留 recurrence
3. "deep representations" 与 "shallow injection" — 这个 asymmetry 让 thought 信号穿越整个 model depth,实现类似 recurrent depth 的效果

最根本的 insight 可能是：**natural language 是一种天然的 information bottleneck**。你可能会觉得 continuous latents 信息密度更高、应该更好,但实验证明 natural language thoughts 显著优于 continuous latents (GSM 42.22 vs 32.65)。原因在于 natural language 携带人类 reasoning 的 inductive bias,compositionality 天然存在,而且 discrete bottleneck 起到信息过滤作用 — 只保留可语言化的 reasoning step,过滤掉 noise。

这跟你之前在 lectures 里讲过的 "language as a latent variable" 的直觉是一致的 — 语言的 discrete 结构提供了 regularization,防止 representation collapse。

**主要的 open question**：teacher-forcing 的经典 exposure bias 问题。训练用 gold states,inference 用 model-generated states,如果某步生成错误 thought,后续 chunks condition on 错误 state,error cascades。Paper Section 5 提到可以用 Thinking States 作为 warm start 再 RL fine-tune,让 model-generated states 接近 gold 行为 — 这是个很自然的下一步,类似 RLHF 但对 thinking trajectory 优化。

如果你感兴趣某个具体方面我可以继续展开,比如 alignment prompt engineering 的细节、speculative thinking 的正确性证明、或者跟 Mamba state update 的对比。

---

# Latent Reasoning with Supervised Thinking States - 深度技术解析

这篇 paper 由 Ido Amos, Mor Geva, Amir Globerson 等 (Tel Aviv University + Google Research) 提出，解决一个核心问题：**如何在保留 CoT reasoning 能力的同时消除其 inference cost**。

参考链接：
- CoT 原始 paper: https://arxiv.org/abs/2201.11903
- Coconut (对比 baseline): https://arxiv.org/abs/2412.06769
- iCoT: https://arxiv.org/abs/2405.14838
- Qwen2.5 技术报告: https://arxiv.org/abs/2412.15115
- Universal Transformers (recurrent 思想源头): https://openreview.net/forum?id=HyzdRiR9Y7
- Length generalization (Anil et al.): https://arxiv.org/abs/2207.04901
- Pause tokens: https://arxiv.org/abs/2310.02226
- Quiet-STaR: https://arxiv.org/abs/2403.09629
- Recurrent depth (Geiping): https://arxiv.org/abs/2502.05171

---

## 1. 核心 Motivation - 设计空间分析

当前 latent reasoning 方案在两个极端之间摆动：

**(A) Continuous latent thoughts (Coconut)** — 用 embedding 替代 natural language tokens，可短，但 supervision 困难，必须用 **BPTT (Backpropagation Through Time)** 优化。BPTT 把整个 recurrent trajectory 展开成计算图，gradient 需要从最终 loss 反向传回每一个 latent step，导致：
- 显存随 step 数线性增长
- 优化不稳定 (vanishing/exploding gradients)
- 实际只能用 6-21 个 latent tokens (见 Table 3)

**(B) Distill CoT into query representations (iCoT)** — 通过 curriculum learning 把 reasoning 压进 query 自身的 representation。问题在于 query tokens 是 **并行** 处理的，sequential reasoning 只能依靠 LLM 的固定 depth 实现。depth 不可变 → 表达力受限 → 在需要多步 composition 的任务上崩盘 (Vars task N=10 时 iCoT 等隐式方法 ≈ 2% 准确率)。

**Thinking States 的洞察**：可以同时拿到 (A) 的 recurrent conditioning 和 (B) 的 parallel training — 关键是让 thoughts 以 **natural language token** 形式存在 (从而可用 teacher-forcing 监督)，但 **不 append 到 context** (从而不增加 length)，并通过 recurrent state 反馈到 future tokens。

---

## 2. 架构详解

### 2.1 三个组件

| 组件 | 角色 | 实现 |
|---|---|---|
| $M_\theta$ | Backbone LLM (Qwen2.5-Base) | 处理 input chunks，提供 rich representations |
| $T$ | Thinking Block | 1-layer causal Transformer decoder，从 deep layer 的 hidden states 自回归生成 natural language thoughts |
| $C$ | Compression Block | 1-layer Transformer encoder + pooling，把变长 thought sequence 压成 fixed-size state |

**初始化 trick** (Appendix A.1)：
- $T$ 的 Transformer block 拷贝自 $M_\theta$ 的最后一层 → 已具备 "hidden → token" 的转换能力
- $C$ 拷贝自 $M_\theta$ 的第一层 → 已具备 contextualize token embeddings 的能力，且输出与 $L^{in}$ (injection layer) 共享 latent space
- $T$ 的 unembedding 拷贝自 $M_\theta$，embedding layer 与 $M_\theta$ **共享** (不是 copy)

这种初始化避免了从头学习一个全新模块，让 $T$ 和 $C$ 从一开始就在合理的 feature space 上工作。

### 2.2 数据流 - 公式逐项解析

输入：长度 $L$ 的 token sequence，切分为 $K$ 个 non-overlapping chunks $\mathbf{X}_1, \ldots, \mathbf{X}_K$。

每个 chunk $\mathbf{X}_i \in \mathbb{R}^{c \times d}$ 其中：
- $c = L/K$ = chunk size (实验中默认 $c=8$)
- $d$ = hidden dimension

这些 embeddings 是从 shallow layer $L^{in}$ (默认第 1 层，跳过 token embedding layer) 之后提取的，**不是** raw token embeddings。

**Step 1: State injection** (公式 1)
$$\tilde{\mathbf{X}}_i = \mathbf{X}_i + \mathbf{S}_i$$

- $\tilde{\mathbf{X}}_i \in \mathbb{R}^{c \times d}$: 注入 state 后的 chunk representation
- $\mathbf{S}_i \in \mathbb{R}^{c \times d}$: 当前 thinking state，与 chunk 同形
- $\mathbf{S}_1 = \mathbf{0}$ (第一个 chunk 没有 prior thought)

**关键直觉**：这是 element-wise addition，意味着 state 是 chunk 表示的 **residual 调制**，类似于 LoRA 或 adapter 的思路，但不引入新参数，只引入新信息。state 在 shallow layer 注入，意味着它要经过 LLM 几乎所有 layers 的 processing，最大化 influence。

**Step 2: Backbone forward** (公式 2)
$$\mathbf{H}_i^{out} = M_\theta(\tilde{\mathbf{X}}_i \mid \tilde{\mathbf{X}}_{<i})$$

- $\mathbf{H}_i^{out}$: chunk 在 deep layer $L^{out}$ (默认 second-to-last) 的 hidden states
- $\tilde{\mathbf{X}}_{<i}$: 历史 chunk representations，通过 **KV-cache** 访问 (不重算)

这步实现了 **compute sharing**：next-token prediction 和 thought generation 共享同一次 forward pass。LLM 处理 input 的同时，已经在为 thought 生成准备 rich features。

**Step 3: Thought generation** (公式 3)
$$\mathbf{Z}_{i+1} = T(\mathbf{H}_i^{out})$$

- $\mathbf{Z}_{i+1} = (z_1, \ldots, z_n)$: variable-length natural language token sequence
- 以 `<EOS>` 结尾
- 若无需 reasoning，$\mathbf{Z}_{i+1}$ 仅包含 `<EOS>` (这点对 inference 加速至关重要)

$T$ 是 lightweight (1 layer) 但条件很 rich — 它消费 $\mathbf{H}_i^{out}$ (来自 deep layer) 而非 raw embeddings，所以即使 1 层也能产出有意义的 thoughts。

**Step 4: Compression** (公式 4)
$$\mathbf{S}_{i+1} = C(\mathbf{Z}_{i+1}) \in \mathbb{R}^{c \times d}$$

- 变长 thought → fixed-size state
- $C$ 取最后 $c$ 个 contextualized representations (不足则 padding)

至此完成一次 recurrence 循环。

### 2.3 Deep-to-Shallow Recurrence - 为什么这么设计

这是 paper 最核心的 architectural insight。看 Figure 3a 的 ablation：

| Extraction layer $L^{out}$ | Accuracy (GSM) | Speedup |
|---|---|---|
| 浅层 (少 layers 参与 recurrence) | ~25% | 高 |
| 深层 (多 layers 参与 recurrence) | ~42% | 中 |

**直觉**：把 thought 从 deep layer 提取、在 shallow layer 注入，意味着 thought 信息要 **穿越 LLM 大部分 layers** 才能影响下一个 thought。这与 Universal Transformers 的 recurrent depth 思想相通 — 用 **同一个 weight 反复应用** 来获得 adaptive compute。

对比 CoT：CoT 把 thought 作为 context tokens 喂回 input，每个 thought token 经过 LLM 全部 layers。Thinking States 通过 deep-to-shallow 实现等价的 "thought 经过大部分计算"，但 **不增加 context length** — state 是 residual 调制，不是新 token。

---

## 3. 训练 - Teacher-Forcing 的精妙之处

### 3.1 关键 trick

公式 (5):
$$\tilde{\mathbf{X}}_i = \mathbf{X}_i + \mathbf{S}_i^*, \quad \mathbf{S}_i^* = C(\mathbf{Z}_i^*)$$

- $\mathbf{Z}_i^*$: ground-truth reasoning sequence (来自 CoT annotation，经 step-to-chunk alignment)
- $\mathbf{S}_i^*$: 用 ground-truth thoughts 压出的 state

**训练时不用 model-generated $\mathbf{S}_i$，而用 gold $\mathbf{S}_i^*$**。这意味着所有 chunk 的输入 $\tilde{\mathbf{X}}_i$ 在训练开始时就完全确定 — 不需要等 model 先生成 $\mathbf{S}_1$ 才能算 $\mathbf{X}_2$。

**后果**：所有 $\mathbf{H}_i^{out}$ 可以在 **单次 parallel forward pass** 中算出。然后 $T$ 在所有 chunks 上并行训练 next-token prediction：

公式 (6) - 联合损失：
$$\mathcal{L} = \mathcal{L}_{LM} + \sum_{i=1}^K \mathcal{L}_T(\mathbf{Z}_i, \mathbf{Z}_i^*)$$

- $\mathcal{L}_{LM}$: 标准 language modeling loss (在最终 answer 上)
- $\mathcal{L}_T$: thought sequence 上的 cross-entropy
- $K$: chunk 数

每个 $\mathbf{H}_i^{out}$ 在 gold state history $\mathbf{S}_1^*, \ldots, \mathbf{S}_i^*$ 下计算，所以预测 $\mathbf{Z}_{i+1}$ 隐式条件于所有 prior gold reasoning steps。

### 3.2 与 BPTT 的成本对比

Figure 6 显示了 forward+backward wall-clock time vs recurrent steps：

- **BPTT (Coconut)**: 线性增长，10 steps 时约 10× penalty
- **Thinking States**: 几乎常数 — 因为 teacher-forcing 把 sequential dependency 从计算图里去掉了

直觉上：BPTT 必须保存每个 step 的所有 activations 以供 backward；teacher-forcing 因为 states 是 gold 的 (作为常数输入)，计算图不展开成时间链，gradient 只在单次 forward 内流动。

### 3.3 Supervision 构造 - Step-to-Chunk Alignment

这是另一个被低估的关键贡献。给定 query + CoT trajectory：

```
Query: "A coin's state is heads. Alice flips, then Bob flips. What's the state?"
CoT: "heads → tails → heads"
```

**Step 1: Step-to-Token Alignment** — 找到每个 reasoning step **最早可以被推断** 的 query 位置：

```
"A coin's state is heads.<T> Alice flips,<T> then Bob flips.<T> What's the state?"
```

每个 `<T>` marker 标记 "此位置之前的信息足以推断对应 reasoning step"。Alignment 通过 Gemini 2.5-Flash 或程序规则获得 (见 Appendix A.3, Figure 7 的 prompt)。

**Step 2: Token-to-Chunk Alignment** — 把 aligned 序列按 chunk size $c$ 切分，每个 chunk 的 target = 该 chunk 内所有 reasoning steps 的拼接。空 chunk → `<EOS>`。

**为什么这很重要**：这个 alignment 把 "全局 CoT" 转化为 "局部 thought supervision"，使 teacher-forcing 成为可能。没有这个步骤，就没有 gold $\mathbf{Z}_i^*$，就要回退到 BPTT。

---

## 4. Inference - Speculative Thinking 算法

### 4.1 问题

训练时 teacher-forcing 让所有 chunks 并行；但 inference 时 $\tilde{\mathbf{X}}_i$ 依赖 $\mathbf{S}_i$，而 $\mathbf{S}_i$ 由 model 生成，**必须 sequential**。Naive prefill = $O(K)$ 串行 forward。

### 4.2 关键观察

**绝大多数 chunks 产生的 state 是 trivial** (只有 `<EOS>`)。在 GSM 数据中，大部分 query positions 不触发计算 — 只有 "10×12=120" 这种位置才需要 thought。

### 4.3 算法 (Algorithm 1)

```
1. 推测所有 chunks 的 state 都是 trivial (S_i = 0)
2. 并行 forward 所有 chunks
3. 对每个 chunk 用 T 和 C 生成真正的 state
4. 找到最早产生 non-trivial state 的 chunk i_1
   - 之前的 chunks 真的是 trivial → 计算正确，cache 它们
5. 从 i_1 开始，conditioning on 正确 S_{i_1}，对剩余 chunks 重复 step 1
6. 终止条件：没有新的 non-trivial state 产生
```

**复杂度**：恰好 $|R| + 1$ 轮，$|R|$ = non-trivial states 数量。当 $|R| \ll K$ (典型情况)，prefill 接近单次 parallel pass。

**正确性**：这是 **exact** 算法，不是近似 — 因为 trivial state 的 speculation 恰好等于真实值，所以被 cache 的部分计算完全正确。

**类比**：这本质上是 **speculative decoding** 的变种，但 speculation 的对象是 "future states 都是 trivial" 而非 "future tokens 来自 draft model"。

---

## 5. 实验结果深度解读

### 5.1 State Tracking - Length Generalization (Table 1)

| Method | Parity N=10 | Parity N=40 | Vars N=10 | Vars N=40 |
|---|---|---|---|---|
| No CoT | 54.67 | 59.60 | 02.15 | 02.19 |
| CoT | 12.35 | 64.38 | 06.78 | 87.75 |
| **Thinking States** | **98.37** | **100.00** | **33.76** | **97.71** |

**惊人结果**：训练时只见 N=10，测试到 100，Thinking States 几乎完美；CoT 在 N=10 训练时只有 12.35%！

**为什么 CoT 在短训练时崩盘**：CoT 通过生成中间 token 实现推理，但 length generalization 要求模型在 **未见过的长度** 上保持 reasoning chain 不漂移。CoT 的每个 token 都是 autoregressive 生成，error 累积。

**为什么 Thinking States 泛化好**：recurrent state 机制天然支持任意长度 — state 更新规则是位置无关的。这呼应 Deletang et al. (2022) 关于 recurrent models 在 Chomsky hierarchy 上表现更好的发现。LLM 通过 Thinking States 实际学到了一个 **state update function**，而非 memorize 特定长度的 trajectory。

**Vars N=10 的 33.76 vs CoT 6.78**：多变量追踪比 parity 难得多，但 Thinking States 仍然显著领先，说明 recurrent mechanism 对复杂 state dynamics 也有效。

### 5.2 General Reasoning (Table 2)

| Method | GSM Acc | GSM Speedup | 2-Hop FK | 2-Hop PK |
|---|---|---|---|---|
| CoT | 60.50 | ×1 | 54.79 | 43.07 |
| No CoT | 34.11 | ×5.59 | 33.47 | 31.92 |
| **Thinking States** | **42.22** | **×2.66** | **54.91** | **43.05** |
| Coconut | 32.65 | ×3.14 | 33.71 | 32.60 |
| iCoT | 34.00 | ×5.71 | 28.84 | 36.31 |

**关键观察**：

1. **2-Hop QA 上 match CoT** — FK 上甚至略超 (54.91 vs 54.79)。这说明对 "retrieve + compose" 类型任务，latent reasoning 已经足够。Biran et al. (2024) 和 Yang et al. (2024) 显示标准 LLM 已有部分 latent multi-hop 能力，Thinking States 把这种能力放大。

2. **GSM 上有 gap** (42.22 vs 60.50) — 数学需要精确的多步计算，latent state 压缩会丢失中间数值精度。但仍然 **远超** 其他 latent 方法 (Coconut 32.65, iCoT 34.00, No CoT 34.11)。

3. **Speedup vs Coconut**：Coconut 在 GSM 上 ×3.14 但准确率低 10 个点；Thinking States ×2.66 但准确率高 10 个点。**Pareto dominance**。

4. **Coconut scaling 失败** (Table 3)：把 latent tokens 从 6 增到 21，准确率没涨 (32.65 → 32.65)，speedup 还掉了。这是 BPTT 训练 fundamental 限制 — 更多 latents 优化更难。

### 5.3 Ablation - Chunk Size (Figure 3b)

- **小 chunk (c=1-2)**: 性能低，latency 差。每个 chunk 容量太小，无法 encode 复杂 thought；且 iteration 多。
- **中 chunk (c=8)**: 性能峰值。
- **大 chunk (c>8)**: 性能下降。因为多个 sequential reasoning steps 被塞进同一 chunk，它们 **无法访问 full recurrent loop** — 只能靠 LLM depth + 轻量 $T$ 处理，破坏 deep-to-shallow 设计。

**直觉**：chunk size 是 "per-state capacity" 与 "reasoning frequency" 的权衡。太小 → state 信息量不足；太大 → 失去 recurrence 的 sequential composition 优势。

---

## 6. Error Analysis - 失败模式与 Intuition

### 6.1 Thinking States 成功、CoT 失败的 case (Figure 4)

**Example 1 - CoT hallucinate 额外 step**：
```
Query: Derrick 每天 10 dozen doughnuts, $2 每个, 6 月卖完所有...
CoT: 10×12=120 → 120×2=240 → 240×30=7200 → 7200×6=43200 (✗)
Thinking States: 10×12=120 → 120×2=240 → 240×30=7200 (✓)
```
CoT 多算了一步 "×6" (可能误以为 6 月 = 6 个月)。Thinking States 因为每个 thought 都 anchored 到 query 的特定位置，不会 "自由发挥" 生成多余 step。

**Example 2 - CoT 过度复杂计算**：
```
Query: 17 green, 2× red, total 60, blue = ?
CoT: 17×2=34 → 60-17-34=8 (✗, 算错)
Thinking States: 17×2=34 → 17+34=51 → 60-51=9 (✓)
```
CoT 试图在一步内做 "60-17-34" 多操作，出错；Thinking States 的 supervision 强制 step-by-step 单操作。

**Insight**：Thinking States 的 supervision alignment 规定了 "何时算什么"，约束了 reasoning trajectory 的形状，减少 hallucination。

### 6.2 Thinking States 失败、CoT 成功的 case - State Ambiguity (Figure 5)

```
Original query:
"Richard 住 15 层楼, 每层 8 单元. 3/4 单元 occupied. 
 每层 unoccupied 单元数?"
TS trajectory: 15×8=120 → (3/4)×120=90 → 120-90=30 (✗, 这是总 unoccupied)
正确答案: 2 (每层 unoccupied)
```

**问题**：query 末尾才问 "每层"，但 TS 在前面 chunks 已经 commit 到 "总 unoccupied" 这个 quantity。

**Zero-shot 修复**：把最后的问题 prepend 到 query 开头：
```
"每层 unoccupied 单元数? Richard 住 15 层楼..."
TS: 0.75×8=6 → 8-6=2 (✓)
```
准确率从 42.22 → 48.65 (+6.43)。

**Root cause**：这与 Thinking States 配合 **causal autoregressive** backbone 有关 — 模型只能 left-to-right 处理，无法 "回头看" 最终问题。**Bidirectional query processing** (如用 encoder 处理 query) 可能解决，但会破坏 KV-cache 的 streaming 性质。这是个开放的架构问题。

---

## 7. 更深层的 Intuition 与 Open Questions

### 7.1 与 Universal Transformers 的精神联系

Universal Transformers (Dehghani et al., 2019) 通过 **同一个 Transformer block 反复应用** 实现 adaptive compute，理论上 Turing complete。但训练困难 (BPTT over steps)。

Thinking States 实质上是用 **teacher-forcing** 把 recurrent depth 训练问题解决了一半 — 不需要 gradient 穿过 recurrent steps，但 inference 时仍保留 recurrence 的表达力。这是一种 **"训练时简化、推理时完整"** 的不对称设计，很有意思。

### 7.2 State 作为 "Working Memory"

Thinking State $\mathbf{S}_i$ 起到 working memory 的作用 — 它不是 query 的简单摘要，而是 **当前 reasoning 进展的压缩表示**。这与 Mamba / SSM 的 hidden state 概念类似，但有两个关键差异：

1. **Discrete bottleneck**: state 来源于 natural language tokens，经过 $C$ 压缩。discrete bottleneck 起到 **信息过滤** 作用 — 只保留可语言化的 reasoning step。
2. **Bidirectional influence**: state 既影响 future token processing，又被 future tokens 触发更新 (通过 $M_\theta$ 和 $T$)。这是 SSM 的 additive state update 所没有的。

### 7.3 为什么 Natural Language Thoughts 比 Continuous Latents 好？

Coconut 用 continuous embeddings，理论上信息密度更高。但实验显示 Thinking States 显著更好 (GSM 42.22 vs 32.65)。原因可能是：

1. **Supervision signal richness**: natural language thoughts 来自人类标注的 CoT，携带强 inductive bias；continuous latents 必须从 final answer loss 反推，signal 弱。
2. **Compositionality**: natural language 天然支持 compositional reasoning ("A=B, B=C → A=C")；continuous embeddings 容易塌缩成全局 feature。
3. **Interpretability 带来的调试性**: Figure 4 的 error analysis 之所以可能，正是因为 thoughts 可读。Coconut 的 latents 不可读，难以诊断失败模式。

### 7.4 未解决的张力

**Train-test mismatch**：训练用 gold states ($\mathbf{S}_i^*$)，inference 用 model-generated states ($\mathbf{S}_i$)。如果 model 在某步生成错误 thought，后续 chunks 会 condition on 错误 state，error cascades。这是 teacher-forcing 的经典 **exposure bias** 问题。

Paper 没有报告 inference 时 thought 生成准确率。如果 thought 生成质量低，parallel training 的优势会被 inference error 抵消。可能的缓解：RL fine-tuning (Section 5 提到的 future work) — 用 Thinking States 作为 warm start，再 RL 优化让 model-generated states 接近 gold 行为。

**Fixed chunk size $c$**：当前 $c$ 是 hyperparameter。理想情况下应该 adaptive — 简单位置用大 chunk，复杂位置用小 chunk。这接近 Geiping et al. (2025) 的 dynamic depth 思想。

### 7.5 推测的扩展方向

1. **Decoder-side Thinking States**：当前只在 query prefill 阶段 reasoning。如果把同样的 mechanism 用在 token generation 阶段 (每生成 $c$ 个 tokens 就 inject 一个 thought state)，可以让 model 在生成 answer 时也 "think"，类似 OpenAI o1 的内部 reasoning。
2. **Hierarchical thoughts**：当前 thoughts 是 single-level。可以设计 multi-scale — coarse thoughts 每 $K_1$ chunks 一次，fine thoughts 每 $K_2 < K_1$ chunks 一次，类似 hierarchical RNN。
3. **Cross-attention injection 替代 addition**：公式 (1) 用 element-wise addition。如果用 cross-attention 让 chunk tokens attend to state，可能表达力更强，但参数更多。

---

## 8. 个人评价

这篇 paper 的核心 trick **很简洁**：deep-to-shallow recurrence + teacher-forcing via natural language supervision。但简洁背后是对 latent reasoning 设计空间的清晰梳理：

- 把 "thoughts 是 natural language" 与 "thoughts 不进 context" 解耦
- 用 alignment annotation 把 CoT supervision 转化为 chunk-local supervision
- 用 speculative thinking 把 sequential inference 的 average case 降到接近 parallel

实验设计有诚意：state tracking 测 length generalization (隔离 optimization 效应)，2-Hop QA 测 latent reasoning 的天然 testbed，GSM 测精确多步计算，ablation 清晰指向 deep-to-shallow 的关键性。

**主要局限**：GSM 上仍比 CoT 差 18 个点，说明 latent compression 对精确数值推理有损；state ambiguity 暴露了 causal processing 与 "晚到的关键信息" 之间的张力。这两个问题都指向同一个未来方向 — **更灵活的 query processing** (bidirectional 或 query-aware chunk scheduling)。

如果能确认 paper 的 arxiv ID 我会附上，根据作者阵容和内容判断这是 2025 年中期的 paper，应该可以在 arxiv 上通过标题搜索到。

---

希望这个讲解帮你 build 起对 latent reasoning 设计空间的 intuition，Andrej。如果你想深挖某个具体方面 (比如 alignment 数据生成的 prompt 工程、speculative thinking 的正确性证明、或与 Mamba state 的对比)，我可以继续展开。
