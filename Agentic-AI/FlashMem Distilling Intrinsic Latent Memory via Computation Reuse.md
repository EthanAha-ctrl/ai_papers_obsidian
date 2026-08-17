---
source_pdf: FlashMem Distilling Intrinsic Latent Memory via Computation Reuse.pdf
paper_sha256: 0fc4ac6d2774f7f2d25ccbecbbd0fd6e8c4d4d1b1ea8bb343bd8a4d8bdc15c6a
processed_at: '2026-08-04T08:52:02-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# FlashMem 人话版

## 故事从哪说起

想象 LLM 是个超级聪明但**每次进门都失忆**的博士生。你跟他聊了三小时数学，他全部听懂了，然后他出门喝杯水回来，你跟他说 "继续刚才那个证明"，他一脸懵 — 全忘了。你只能把前三小时的草稿纸重新摊开给他看一遍。

这就是 LLM agent 的现实。每次推理都是一次 forward pass,历史信息要么塞进 context window,要么重新编码。Agent 要做长任务,这个 "重新摊草稿纸" 的动作反复发生,又贵又慢。

---

## 现有方法在干什么蠢事

大家意识到这个 memory 问题后,想了个方案:给 agent 配一个**额外的 memory encoder**,专门负责把历史压缩成几个 latent vectors,然后塞回 context。

这个思路听起来合理,实际上有个**很蠢的浪费**:

```
[Backbone forward]  →  处理完历史,产生 KV cache (信息已经在这了)
[Memory Encoder]    →  把同样的历史再 forward 一遍,产生 memory tokens
```

历史被 encode 了**两次**。Backbone 的 KV cache 里已经有所有信息了,memory encoder 还在那儿吭哧吭哧重新算一遍。这就是 MemGen [\[1\]](https://arxiv.org/abs/2509.24704)、SoftCoT [\[2\]](https://arxiv.org/abs/2502.12134) 这些方法的通病。

打个比方:你写了个 Python 函数,内部已经算好了一个大 dict。然后你的同事又写了另一个函数,把同样的 input 重新算一遍,就为了拿到同一个 dict。你看着会觉得 "这人是不是傻?"

---

## FlashMem 的核心 insight

**一句话: backbone 算过的东西,不要重新算。**

FlashMem 说:backbone 在处理完历史之后,它最后那一层的 hidden state $h_t$ 已经是历史的"充分统计量"了 — 也就是 $h_t$ 里**包含了历史所有信息**(至少理论上,基于 LLM representation 的 injectivity [\[3\]](https://arxiv.org/abs/2510.15511))。

那我们就**直接拿 $h_t$ 当 memory 的种子**,不再重新 forward 历史。

更激进的是,FlashMem 说:既然 backbone 已经把历史的 K, V cache 都算好了,consolidator 连自己的 $W_K$, $W_V$ 都不要了 — **直接 query backbone 的 KV cache**。

$$\text{Attn}(x, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{(x W_Q) \mathbf{K}^\top}{\sqrt{d}}\right) \mathbf{V}$$

变量:
- $x$: consolidator 当前层的 input
- $W_Q$: **唯一**要学的参数
- $\mathbf{K}, \mathbf{V}$: backbone 已经算好的 KV cache,frozen,直接用
- $d$: hidden dimension

这是极致的 parameter-efficient design。LoRA [\[4\]](https://arxiv.org/abs/2106.09685) 还保留 base matrix 学个 delta,FlashMem 干脆**连 K, V 的 projection 都不要**,只学一个 Q。

---

## 什么时候生成 memory — model 自己说了算

这是 FlashMem 另一个亮点。现有方法要么**每个 query 都注入 memory** (SoftCoT,浪费),要么训练个**单独的 classifier** 决定何时触发 (MemGen,引入额外参数和 distribution shift)。

FlashMem 说:model 自己**困惑的时候**才需要 memory。怎么知道 model 困惑了?

**看 attention entropy。**

当 LLM 对下一个 token 很自信时,它的 attention 会**聚焦**在几个关键 token 上 — entropy 低。当它困惑、不知道看哪里时,attention 会**散开** — entropy 高。这个 correlation 来自 Kuhn et al. [\[5\]](https://arxiv.org/abs/2302.09671) 和 Farquhar et al. Nature paper [\[6\]](https://www.nature.com/articles/s41586-024-07421-0) 的工作。

### 但是有个坑 — Attention Sink

LLM 有个怪癖:它会把大量 attention 倾倒在开头那几个 tokens 上(比如 [BOS]),无论这些 tokens 是否语义相关。这叫 **attention sink** [\[7\]](https://arxiv.org/abs/2309.17453)。

如果不处理,attention sink 会**人为压低 entropy** — 因为大量 mass 集中在 sink tokens 上,看起来 entropy 不高。但这不是 model confident,而是 model "卡"在 sink 上。

FlashMem 的解法:先把 sink tokens **mask 掉**,再把剩下的 attention weights **重新归一化**,然后算 entropy:

$$\tilde{A}_{t,h}[j] = \frac{A_{t,h}[j] \cdot \mathbb{I}(j \notin S_{\text{sink}})}{\sum_k A_{t,h}[k] \cdot \mathbb{I}(k \notin S_{\text{sink}})}$$

变量:
- $A_{t,h}[j]$: head $h$ 在 step $t$ 对 token $j$ 的 attention weight
- $S_{\text{sink}}$: sink tokens 的 index 集合
- $\mathbb{I}(\cdot)$: indicator function,sink tokens 取 0,其他取 1
- $\tilde{A}_{t,h}[j]$: mask 并归一化后的 attention

然后算 Shannon entropy 并对所有 head 平均:

$$\mathcal{H}_t = \frac{1}{H} \sum_{h=1}^{H} \left( -\sum_{j \notin S_{\text{sink}}} \tilde{A}_{t,h}[j] \log \tilde{A}_{t,h}[j] \right)$$

变量:
- $H$: attention head 数
- $h$: head index
- $\mathcal{H}_t$: step $t$ 的 system uncertainty

当 $\mathcal{H}_t > \tau$,触发 memory consolidation。

### 阈值 $\tau$ 怎么定

不是拍脑袋给的固定值。FlashMem 用 **percentile-based calibration** — 在 validation set 上统计 entropy 分布,取 **85th percentile** 作为阈值:

$$\tau = \text{Percentile}(\{\mathcal{H}_i\}_{i=1}^N, 85)$$

变量:
- $\{\mathcal{H}_i\}_{i=1}^N$: validation set 上的 N 个 entropy 值
- $85$: 第 85 百分位

这意味着**只对 entropy 最高的 15% 的 step 触发** memory 生成。大部分时候 model 跑得好好的,不浪费 compute。

这个 percentile strategy 自带**自适应** — 大 model 的 baseline entropy 低,小 model 的高,但 percentile 都能适应。不需要为大 model 和小 model 分别调阈值。

---

## Memory Consolidator 怎么工作

触发之后,consolidator 要生成 memory。流程:

1. **拿 $h_t$ 当种子**: 把 backbone 最后一个 hidden state 通过两层 MLP,project 成 consolidator 的初始 query $m_0$
2. **生成 K 个 memory tokens**: consolidator autoregressive 地生成 8 个 latent vectors (greedy decoding)
3. **Soft-inject 回 backbone**: 这 8 个 memory tokens 经过 backbone forward,产生自己的 K, V,**直接 append 到 backbone 的 KV cache 里**
4. 后续所有 tokens 都能 attend 到这些 memory tokens,就像它们是 context 的一部分

关键点:memory tokens 是 **continuous embeddings**,不是 text tokens。它们在 context window 里占据 attention position,但不占 text token budget。

### Weight Inheritance Trick

Consolidator 的初始化不是 random,是**直接 copy backbone 的最后一层**。

为什么?如果 random init,consolidator 的 representation space 和 backbone 的 KV cache **完全不在同一个 manifold**,cross-attention 会 fail — Q 和 K 不对齐。

Copy backbone 的权重,consolidator 一开始就在 backbone 的 semantic manifold 内,只需要 fine-tune 就能 align。这跟 Anthropic 的 superposition research [\[8\]](https://transformer-circuits.pub/2022/toy_model/index.html) 说的 "features must be in the same basis to interact" 一个道理。

---

## 训练: Knowledge Distillation from the Future

Training sequence 是:

$$S = [x, \mathcal{M}_{\text{gen}}, y]$$

变量:
- $x$: instruction prompt (mask掉,不计算 loss)
- $\mathcal{M}_{\text{gen}}$: 8 个 memory tokens (mask掉,不计算 loss)  
- $y$: ground-truth expert trajectory (计算 loss)

Loss 是标准 cross-entropy,但**只在 $y$ 上算**:

$$\mathcal{L} = -\sum_{t=1}^{|y|} \log P_\theta(y_t \mid x, \mathcal{M}_{\text{gen}}, y_{<t})$$

变量:
- $y_t$: 第 $t$ 个 ground-truth token
- $\theta$: backbone parameters (**frozen, $\nabla_\theta := 0$**)
- $\mathcal{M}_{\text{gen}}$: consolidator 生成的 memory

**关键**: backbone 是 frozen 的,但 gradient 可以**穿过 frozen layers** 反向传播到 memory tokens,再传到 consolidator 的参数 $\psi$。

这迫使 consolidator 学到一个隐含目标:**把 trajectory $y$ 的所有信息压缩到 8 个 memory tokens 里**,让 frozen backbone 能 reproduce $y$。

这就是 "knowledge distillation from the future" — 把未来的答案提前 distill 到 memory 里。

---

## 效果:为什么 FlashMem 真的快

Table 3 是最 convincing 的数据。64k context:

| | Vanilla | MemGen | FlashMem |
|---|---|---|---|
| VRAM | 31.21 GB | 40.78 GB | 31.44 GB |
| Throughput | 25.67 tok/s | 4.13 tok/s | 20.86 tok/s |
| Latency | 9.97 ms | 61.99 ms | 12.28 ms |

MemGen 在 64k 下**基本不可用** — throughput 掉到 4.13 tok/s,因为 re-encoding 的 cost 是 $O(L^2)$ (attention on 64k tokens)。

FlashMem 的 cost 几乎和 Vanilla 一样 — consolidator 只跑 8 步,cost 是 $O(K \cdot L)$ where $K=8$,远小于 $O(L^2)$。

**5 倍加速,VRAM 几乎没增加**。这是 architectural efficiency 的胜利,不是 trick-level 的优化。

---

## 为什么单层 Consolidator 够用

Table 4 的 ablation:

| L | GSM8K | KodCode |
|---|---|---|
| 1 | 69.55 | 50.10 |
| 2 | 69.47 | 50.19 |
| 6 | 69.55 | 50.29 |

$L=1$ 到 $L=6$ 几乎没差。

**Intuition**: Memory consolidation 是**selection + compression** 任务,不是 sequential transformation。选 "哪些历史信息重要" 这个操作,单次 attention 就能做。深层网络只是冗余的 stack,不会提升 selection 能力。

这跟 SnapKV [\[9\]](https://arxiv.org/abs/2404.14469)、H2O [\[10\]](https://arxiv.org/abs/2306.14048) 的 KV compression 类似 — 它们也都是单次 importance scoring。

---

## 几个我觉得可疑的点

### Sufficient Statistic Claim

"$h_t$ 是 $\tau_{<t}$ 的充分统计量" 这个 claim 在 long-context 下有问题。

64k context,最后 hidden state 对**最早期的 tokens** 的 attention 已经严重衰减(RoPE [\[11\]](https://arxiv.org/abs/2104.09864) 只能 partial fix)。所以 $h_t$ 对**最近 context** 是 sufficient 的,对**远期 context** 是 information-lossy 的。

这解释了为什么 FlashMem 在 BookSum 这种需要远期依赖的任务上表现不稳定。

### Memory Capacity 瓶颈

8 个 memory tokens,$d \approx 4096$,total capacity = 32,768 floats。

对于 GSM8K 这种 short reasoning 够用,但要 losslessly 压缩 64k tokens 的 BookSum,显然不够。FlashMem 的 memory 是**有损压缩**,不是无损。但 paper 没分析 consolidator 到底学到了什么 abstraction,这是 interpretability 的空白。

### Attention Entropy 作为 Uncertainty Proxy

Attention entropy 高不一定等于 confusion。Model 在 retrieval 任务中可能**故意 diffuse attention** 去 scan 多个 candidates — 这时候 entropy 高是 feature,不是 bug。FlashMem 可能在这种情况下 false positive 触发 memory 生成。

Table 5 显示 76.5% 的触发是有效的,但 23.5% 是无效的。这个 false positive rate 不低。

---

## 真正的 Takeaway

FlashMem 的核心 insight 是:

**不要做额外的事,复用已经做的事。**

Backbone 算完历史,KV cache 已经在那了 — 直接 query 它,不要重新 encode。Backbone 的 hidden state 已经包含了历史信息 — 直接 project 它,不要重新 forward。

这个 principle 简单得像废话,但放到 latent memory 的 context 里,它推翻了整个 segregated architecture paradigm 的 efficiency 假设。MemGen 这些方法看起来 "sophisticated"(三个独立模型协同),实际上是在做 backbone 已经做过的事。FlashMem 看起来 "simple"(就一个单层 consolidator),但它做的是**别人没做的 selection**。

第二个 takeaway 是 **Cognitive Monitor 的 parameter-free 设计** — 用 attention entropy 做 real-time confusion detection,不需要训练额外 classifier。这个 idea 独立于 FlashMem,可以推广到很多场景:adaptive test-time compute、early exit、dynamic depth switching、hallucination detection。

最后一个 takeaway:这个工作建立了一个新 paradigm — **intrinsic memory**。Memory 不再是 agent 的外部 attachment,而是 backbone 内部 computational flow 的 byproduct。这个 paradigm shift 可能比 FlashMem 这个具体方法更重要。

---

## References

- [\[1\] MemGen](https://arxiv.org/abs/2509.24704)
- [\[2\] SoftCoT](https://arxiv.org/abs/2502.12134)
- [\[3\] Nikolaou et al. - LLM Injectivity](https://arxiv.org/abs/2510.15511)
- [\[4\] LoRA](https://arxiv.org/abs/2106.09685)
- [\[5\] Kuhn et al. - Semantic Uncertainty](https://arxiv.org/abs/2302.09671)
- [\[6\] Farquhar et al. - Nature Hallucination Detection](https://www.nature.com/articles/s41586-024-07421-0)
- [\[7\] Xiao et al. - Attention Sinks (StreamingLLM)](https://arxiv.org/abs/2309.17453)
- [\[8\] Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)
- [\[9\] SnapKV](https://arxiv.org/abs/2404.14469)
- [\[10\] H2O](https://arxiv.org/abs/2306.14048)
- [\[11\] RoPE](https://arxiv.org/abs/2104.09864)

---

# FlashMem 深度解读：从 Computation Reuse 角度重新思考 Latent Memory

Andrej, 这篇 paper 触到了 LLM agent memory 设计中一个我一直觉得被低估的痛点 — **segregated architecture 的 re-encoding 诅咒**。让我把 intuition 拆开讲。

---

## 1. Problem Setup: 为什么现有 latent memory 走歪了

LLM 本质是 stateless function `f_θ: x → y`,参数 θ frozen 之后,任何 "memory" 都得通过 input stream 重新进入。这件事本身没有问题,问题出在 **memory 怎么生成**。

现有的 generative latent memory 范式 (MemGen [\[1\]](https://arxiv.org/abs/2509.24704), SoftCoT [\[2\]](https://arxiv.org/abs/2502.12134), VisMem [\[3\]](https://arxiv.org/abs/2511.11007)) 都采用公式 (2) 的形式:

$$\mathcal{M} = \mathcal{G}_{\phi}(\tau_{<t})$$

其中 $\mathcal{G}_{\phi}$ 是一个 **独立训练的 auxiliary encoder**, $\tau_{<t} = (o_1, a_1, \ldots, o_{t-1}, a_{t-1})$ 是 interaction history。

这个公式看起来无害,但隐含一个昂贵的 implication: $\mathcal{G}_{\phi}$ 必须 **重新 forward pass 处理 $\tau_{<t}$**,即使 backbone $\pi_\theta$ 在生成 $a_t$ 时已经处理过 $\tau_{<t}$ 了。也就是说,历史被 encoded 了 **两次**:一次在 backbone 的 KV cache 中,一次在 $\mathcal{G}_{\phi}$ 的内部 state 中。

更糟糕的是, $\mathcal{G}_{\phi}$ 通常还要 autoregressive 地 decode 出 memory tokens (像 MemGen 要跑 Trigger + Weaver + Reasoner 三个独立模型 [\[1\]](https://arxiv.org/abs/2509.24704)),这又是一次额外的 forward pass。在 long-context (e.g., 64k tokens) 场景下,这种 re-encoding 的 latency 会爆炸 — Table 3 显示 MemGen 在 64k 时 latency 飙到 61.99 ms/step,而 vanilla 只要 9.97 ms。

**Intuition**: 这就好比你写代码,有一个庞大的中间状态在 CPU cache 里已经计算好了,但内存系统非要重新跑一遍同样的计算再 cache 一次。这不是效率问题,是 **architectural mistake**。

---

## 2. FlashMem 的核心 Insight: Computation Reuse

FlashMem 的核心赌注是这样一个 claim:

> **The last hidden state $h_t$ of the backbone is a sufficient statistic for the interaction history $\tau_{<t}$.**

这个 claim 引用了 Nikolaou et al. 2025 [\[4\]](https://arxiv.org/abs/2510.15511) 关于 LLM injectivity 的工作。其逻辑链条是:

1. Transformer 的 hidden state $h_t = f_\theta(\tau_{<t}, o_t)$ 在理论上 (under mild conditions) 是 injective 的 — 不同的 input trajectories 会产生不同的 hidden states (至少在 over-parameterized regime 下)。
2. 如果 $h_t$ 是 injective 的,那么它就 **reversible**,即可以从 $h_t$ 反推回 $\tau_{<t}$ 的信息。
3. 既然 $h_t$ 已经包含了 $\tau_{<t}$ 的所有信息,就没有必要再 forward 一遍 $\tau_{<t}$ 去生成 memory — 直接 **reuse** $h_t$ 就行了。

这个推理 chain 的关键是 "sufficiency" — 在统计学意义上,$h_t$ 是 $\tau_{<t}$ 的充分统计量。这个 claim 有争议空间 (后面我会展开),但作为 design principle 它 elegant 得惊人。

**Intuition**: 与其把 memory 当成 backbone 之外的独立 module,不如把它看成 backbone 内部某种 latent state 的 **projection**。Backbone 是 encoder,Consolidator 是 projector,projection 不需要重新 encode。

---

## 3. Cognitive Monitor: Attention Entropy 作为 epistemic probe

这部分是 paper 中我最喜欢的 — 因为它 **parameter-free**。

### 3.1 为什么 attention entropy 能 detect confusion

理论依据来自 Kuhn et al. 2023 [\[5\]](https://arxiv.org/abs/2302.09671) 和 Farquhar et al. 2024 Nature paper [\[6\]](https://www.nature.com/articles/s41586-024-07421-0) 的工作,核心 idea 是:

- 当 LLM 对 next-token 不确定时,它的 attention distribution 会变得 **diffuse** (高 entropy),因为没有明确的 salient token 可以聚焦。
- 相反,当 model confident 时,attention 会 **sharpen** (低 entropy),集中在几个 key tokens 上。
- Hallucination 通常伴随 high semantic entropy,而 attention entropy 是 semantic entropy 的 proxy。

### 3.2 Attention Sink 问题

直接计算 attention entropy 会踩坑 — **attention sink phenomenon** [\[7\]](https://arxiv.org/abs/2309.17453)。LLM 会把大量 attention probability mass 倾倒在初始 tokens (比如 [BOS]) 上,无论这些 tokens 语义上是否相关。这会人为制造一个 **low-entropy anchor**,掩盖真实的 attention 分散程度。

Darcet et al. 2024 [\[8\]](https://arxiv.org/abs/2304.09408) 在 ViT 中也发现类似现象,称为 "registers"。

### 3.3 公式 (3) 和 (4) 详解

公式 (3) 处理 attention sink:

$$\tilde{A}_{t,h}[j] = \frac{A_{t,h}[j] \cdot \mathbb{I}(j \notin S_{\text{sink}})}{\sum_k A_{t,h}[k] \cdot \mathbb{I}(k \notin S_{\text{sink}})}$$

变量含义:
- $A_{t,h}[j] \in \mathbb{R}^t$: head $h$ 在 step $t$ 对 context 中第 $j$ 个 token 的 attention weight
- $S_{\text{sink}}$: attention sink tokens 的 index 集合 (通常是 [BOS], 以及前几个 system prompt tokens)
- $\mathbb{I}(\cdot)$: indicator function, sink tokens 取 0,非 sink tokens 取 1
- $\tilde{A}_{t,h}[j]$: re-normalized 之后的有效 attention distribution

这个操作的本质是 **soft masking + re-normalization** — 把 sink tokens 的 attention 设为 0,然后重新归一化剩下的 weights 使其和为 1。

公式 (4) 计算 Shannon entropy:

$$\mathcal{H}_t = \frac{1}{H} \sum_{h=1}^{H} \left( -\sum_{j \notin S_{\text{sink}}} \tilde{A}_{t,h}[j] \log \tilde{A}_{t,h}[j] \right)$$

变量含义:
- $H$: attention head 总数
- $h$: head index,从 1 到 $H$
- $j$: token index,遍历所有 non-sink tokens
- $\mathcal{H}_t$: step $t$ 的 system uncertainty score,通过对所有 head 的 entropy 取平均

**Intuition**: 每个 attention head 看到的"世界"不一样,有的关注 syntax,有的关注 semantics。把它们各自的不确定性平均起来,得到一个 robust 的整体 confusion score。这是 multi-view uncertainty estimation。

### 3.4 Threshold $\tau$ 的自适应选择

公式 (9) 给出的 percentile-based calibration:

$$\tau = \text{Percentile}(\{\mathcal{H}_i\}_{i=1}^N, P_{\text{target}})$$

其中 $P_{\text{target}} = 85$th percentile,$N$ 是 validation set 样本数。

这个设计很 clever — 它 **不固定绝对 threshold**,而是根据当前 model + task 的 entropy 分布动态调整。Section C.1 给出的经验观察:
- **Larger model → lower baseline entropy** (attention 更 sharp)
- **Open-ended task (code generation, creative writing) → higher baseline entropy**
- **Convergent reasoning (math, logic) → lower baseline entropy**

所以 percentile-based calibration 自动适应这些变化 — 在 GPT-4 级别的 model 上,数学任务可能 0.3 entropy 就算 "high",但在小 model creative writing 任务上 0.6 才算 high。Percentile 把这种 relative deviation 抽象掉了。

**Intuition**: 这就像给 model 装了一个"心率监测器" — 我们不在乎绝对心率,而在关心心率是否偏离了 baseline。Memory consolidation 只在"心率异常"时触发。

---

## 4. Memory Consolidator: Shared-KV Cross-Attention

这是 paper 的 architectural 核心。

### 4.1 Projection-Free Cross-Attention 公式 (5)

标准 cross-attention 是:
$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d}}\right) V$$
其中 $Q = x W_Q$, $K = x' W_K$, $V = x' W_V$,$W_Q, W_K, W_V$ 都是 learnable projections。

FlashMem 的 modification:

$$\text{Attn}(x, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{(x W_Q) \mathbf{K}^\top}{\sqrt{d}}\right) \mathbf{V}$$

变量含义:
- $x$: consolidator 当前 layer 的 input state,初始为 $m_0 = \text{MLP}(h_t)$
- $W_Q \in \mathbb{R}^{d \times d}$: **唯一** learnable projection matrix
- $\mathbf{K}, \mathbf{V} \in \mathbb{R}^{t \times d}$: backbone 已经计算好的 KV cache,**frozen,直接复用**
- $d$: hidden dimension

**关键差异**: $W_K$ 和 $W_V$ 被丢弃。Consolidator 不再重新 project backbone 的 hidden states,直接 query 它们。

**Intuition**: 这是 parameter-efficient cross-attention 的极致。如果说 LoRA [\[9\]](https://arxiv.org/abs/2106.09685) 是 "保留 base matrix,只学 low-rank delta",那么 Shared-KV 是 "保留所有 K/V matrices,只学 Q"。极端到不能再极端了。

### 4.2 Weight Inheritance Strategy

Paper 提到 consolidator 初始化用 backbone 的 **最后 $L$ 层** ($L \ll N_{\text{backbone}}$)。这是一个关键的训练 trick — 如果 random init,consolidator 的 representation space 和 backbone 的 KV cache 完全不在同一个 manifold 上,cross-attention 会 fail。

这让我想到 Anthropic 的 Toy Models of Superposition [\[10\]](https://transformer-circuits.pub/2022/toy_model/index.html) 中的概念 — representations 必须在同一个 "feature basis" 下才能相互 attend。Weight inheritance 确保 consolidator 一开始就在 backbone 的 semantic manifold 内,只需要 fine-tune 一下就能 align。

### 4.3 单层 Consolidator 足够

Table 4 的 ablation 显示,$L=1$ 已经足够,$L=6$ 几乎没有提升。这印证了一个 deeper insight:

**Memory consolidation 不是 deep reasoning,是 selective projection**。

如果 consolidator 要做的是 "从 backbone 的 KV cache 中提取关键信息并投影到 latent memory space",这是一个 **selection + compression** 任务,而不是 sequential transformation 任务。单层 attention 就能完成 selection,深层结构只是冗余。

这跟 SnapKV [\[11\]](https://arxiv.org/abs/2404.14469), H2O [\[12\]](https://arxiv.org/abs/2306.14048) 的 KV compression 类似 — 这些方法也都是单次的 importance scoring,不需要 deep network。

---

## 5. 训练细节: Knowledge Distillation from the Future

公式 (6) 和 (7) 描述 training objective:

$$S = [x, \mathcal{M}_{\text{gen}}, y]$$

$$\mathcal{L} = -\sum_{t=1}^{|y|} \log P_\theta(y_t \mid x, \mathcal{M}_{\text{gen}}, y_{<t})$$

变量含义:
- $x$: instruction prompt
- $\mathcal{M}_{\text{gen}}$: K 个 latent memory tokens (paper 中 $K=8$),由 consolidator greedy 生成
- $y$: ground-truth expert trajectory
- $\theta$: backbone parameters,**frozen** ($\nabla_\theta := 0$)
- Loss 只在 $y$ 上计算,$x$ 和 $\mathcal{M}_{\text{gen}}$ 的 label 都 mask 为 -100 (PyTorch ignore_index)

**关键 trick**: gradient flow 路径是

```
Loss → y logits → backbone (frozen) → M_gen (trainable) → Consolidator ψ (trainable)
```

Backbone frozen,但 gradient 可以 **backprop through frozen layers** 进入 $\mathcal{M}_{\text{gen}}$,再进入 consolidator $\psi$。这迫使 consolidator 学到一个 "最佳 prompt" — 把未来 trajectory $y$ 的所有信息压缩到 K 个 latent vectors 中,使得 frozen backbone 能 reproduce $y$。

这本质是 **prompt tuning 的 reversed 版本**:不是学 prompt 让 backbone 表现好,而是学 latent memory 让 backbone **在推理时** 表现好。更精确地说,这是 **knowledge distillation from future** — 把 trajectory $y$ 的 "essence" 提前 distill 到 memory tokens 中。

参考 Gist Tokens [\[13\]](https://arxiv.org/abs/2304.08497) 的思想类似,但 gist tokens 是 offline compress prompt,FlashMem 是 online compress reasoning context。

---

## 6. 实验数据解读

### 6.1 Table 1 (Qwen) 和 Table 2 (Llama) — 性能 parity

几个关键观察:

1. **FlashMem 在小 model 上 (Llama 3.2 3B) MATH 反超 MemGen** (48.05% vs 45.17%)。这暗示 Shared-KV design 在小 parametric space 中更稳定 — 因为小 model 的 representation 已经 compact,再 extra encoder 反而引入 noise。

2. **长文本任务 (GovReport, BookSum) FlashMem 普遍领先** — 这印证了 Shared-KV 直接复用 backbone KV cache 比 re-encoding 更能 capture global semantics。

3. **SnapKV (KV compression baseline) 显著落后** — 这说明 generative latent memory (压缩成 dense vector) 比 extractive pruning (保留 top-k tokens) 更能 preserve long-range dependencies。这是 generative vs extractive 的经典对决,paper 站在 generative 一边。

4. **GPQA 上 FlashMem 略低于 MemGen** — GPQA 是 graduate-level QA,需要 deep factual knowledge。可能 Shared-KV 的 "compression loss" 在 knowledge-intensive 任务上更明显 — 因为 memory tokens 容量有限,无法 encode 所有 factual details。

### 6.2 Table 3 — Efficiency Benchmark

这是 paper 的杀手锏。64k context:

| Metric | Vanilla | MemGen | FlashMem |
|--------|---------|--------|----------|
| Peak VRAM | 31.21 GB | 40.78 GB | 31.44 GB |
| Throughput | 25.67 tok/s | 4.13 tok/s | 20.86 tok/s |
| Latency | 9.97 ms | 61.99 ms | 12.28 ms |

**5× speedup over MemGen**,VRAM 几乎和 Vanilla 持平。这意味着 FlashMem 在 long-context 场景下 **practically usable**,而 MemGen 已经 **practically unusable**。

**Intuition**: MemGen 的 cost 随 context length **super-linearly** 增长 (因为 re-encoding 的 cost = O(L²) for attention),FlashMem 的 cost 几乎是 **constant overhead** (因为 consolidator 只跑 K=8 步,cost = O(K·L))。

### 6.3 Table 5 — Entropy Reduction Statistics

- Mean Entropy Reduction: 0.215 ± 0.125
- Probability of Reduction: 76.5%
- Probability of Significant Reduction (>0.5): 7.2%

76.5% 这个数字很重要 — 它说明 **大部分时候 memory injection 是有用的**,但有 23.5% 的情况没效果甚至反向。这可能是 false positive (entropy 高但不是真 confusion) 或者 memory 内容不 relevant。

7.2% 的 significant reduction 对应 "aha moments" — 这些时刻 memory 真的把 model 从 hallucination 边缘拉回来。

---

## 7. Algorithm 1 的 Inference Loop

让我把这个 loop 展开,因为它揭示了 FlashMem 的 runtime 动力学:

```
while generating:
    1. Forward backbone → get logits, attentions, h_last
    2. Compute attention entropy H from last layer attentions
    3. If H > τ:  # Confusion detected
       a. m_0 = MLP(h_last)  # Project hidden state
       b. M = GreedyGen(Consolidator, m_0, K, V, steps=8)  # Generate 8 memory tokens
       c. Soft-append M to backbone input
       d. Update K, V with M's K, V  # Memory tokens become part of context
    4. Sample next token x_next
    5. Append x_next to context
```

**关键细节**: Memory tokens 不是 text tokens,是 **continuous embeddings**。它们经过 backbone forward 产生自己的 K, V 对,**直接 append 到 backbone 的 KV cache** 中。后续所有 tokens 都能 attend 到这些 memory tokens。

这就像在 context window 中插入 "虚拟 tokens" — 不是真实 text,但占据 attention position,可以被 attend 到。SoftCoT [\[2\]](https://arxiv.org/abs/2502.12134) 也是类似思路,但 SoftCoT 是 **static prepend**,FlashMem 是 **dynamic inject at confusion points**。

### 7.1 Greedy Decoding for Memory

Paper 在 Section A.3 强调 memory generation 用 greedy search 而非 sampling。理由:
1. **Stability**: 避免 high-variance noise 注入 backbone context
2. **Reproducibility**: 相同 context → 相同 memory

这个设计选择很有意思 — 它假设 memory 应该是 **deterministic function of context**,而不是 stochastic。这与 LLM 的 text generation (通常 sampling with temperature) 形成对比。

**Intuition**: Memory 是 "summary",不是 "creative output"。Summary 应该 deterministic,因为它要 faithfully represent context,而不是 introduce randomness。

---

## 8. Critical Analysis: FlashMem 的潜在问题

### 8.1 Sufficient Statistic Claim 的可疑性

FlashMem 的核心 claim 是 "$h_t$ is a sufficient statistic for $\tau_{<t}$"。但这个 claim 有几个 caveat:

1. **Nikolaou et al. [\[4\]](https://arxiv.org/abs/2510.15511) 的 injectivity result 是关于整个 model 的 representation**,不是 specifically about **last hidden state**。Last hidden state 已经经过所有 layer 的 transformation,可能丢失了 early-layer 的 syntactic 信息。

2. **Information loss in deep layers**: Geva et al. 2021 [\[14\]](https://arxiv.org/abs/2012.14913) showed FFN layers 是 key-value memories,这暗示 representation 在 deep layers 是 "task-specialized",可能 **过滤掉** 了 general context 信息。如果是这样, $h_t$ 反而 **不是** sufficient statistic。

3. **Long-context degradation**: 在 64k context 下,即使 transformer 用 RoPE [\[15\]](https://arxiv.org/abs/2104.09864) 等位置编码,last hidden state 对早期 tokens 的 attention 会衰减 (LongRoPE [\[16\]](https://arxiv.org/abs/2402.13753) 也只是 partial fix)。所以 $h_t$ 对 $\tau_{<t}$ 的 "早期部分" 可能已经 information-lossy。

**Implication**: FlashMem 在 **recent context** rich 的任务上应该表现好,在 **distant context critical** 的任务上可能不如 re-encoding 方法。Table 1 中 BookSum 上 FlashMem 表现不一致 (Qwen 2.5 上 13.77 vs MemGen 12.86,Qwen 3 上 10.99 vs MemGen 10.18),这可能与这个 limitation 相关。

### 8.2 Attention Entropy 作为 Uncertainty Proxy 的局限

虽然 Kuhn et al. [\[5\]](https://arxiv.org/abs/2302.09671) 和 Farquhar et al. [\[6\]](https://www.nature.com/articles/s41586-024-07421-0) 建立了 entropy-uncertainty correlation,但这个 correlation 有几个 nuance:

1. **Attention entropy ≠ token-level entropy**: Paper 用的是 attention distribution 的 entropy,而不是 output token probability distribution 的 entropy。这两者相关但不同 — attention entropy 高可能只是因为 model 在 "broadly surveying" context,不一定 confusion。比如在 retrieval 任务中,model 可能故意 diffuse attention 来 scan 多个 candidates。

2. **Calibration drift**: percentile-based threshold 假设 validation set 的 entropy 分布和 test set 一致。如果 distribution shift (e.g., OOD inputs),85th percentile 可能不再 meaningful。

3. **Multi-modal uncertainty**: Kadavath et al. 2022 [\[17\]](https://arxiv.org/abs/2207.05221) (Anthropic 的 "Language models (mostly) know what they know") 指出 LLM 的 self-uncertainty 是 multi-modal 的 — model 可能在 syntax 上 confident 但在 semantics 上 confused。Average 所有 head 的 entropy 会 smooth 掉这种 structured uncertainty。

### 8.3 Memory Token Capacity 瓶颈

$K=8$ 个 memory tokens。每个 token 是 $d$-dimensional vector, $d \approx 4096$ (Qwen 2.5)。Total memory capacity = $8 \times 4096 = 32,768$ floats。

这个容量对于 GSM8K 这种 short reasoning 也许够,但对于 64k context 的 BookSum summary,显然不足以 "losslessly compress" 所有信息。

**Intuition**: FlashMem 的 memory 是 **lossy compression**,而非 lossless。它必须 prioritize "what matters most"。问题是 — 如何确保 consolidator 学到的是真正重要的信息?Paper 没有给出 memory content 的 interpretability 分析 (Section D 只展示了 attention patterns,没有展示 memory embedding 的语义内容)。

### 8.4 Weight Inheritance 的局限

Consolidator 用 backbone 最后 $L=1$ 层 init。但如果 backbone 是 Instruct-tuned (e.g., Qwen 2.5 Instruct),最后几层可能已经经过 RLHF [\[18\]](https://arxiv.org/abs/2203.02155) / DPO [\[19\]](https://arxiv.org/abs/2305.18290) 调整,representation 可能偏 "alignment direction" 而非 "raw knowledge direction"。

这意味着 inherited weights 可能让 consolidator 偏向 "helpful/harmless" 的 memory,而非 "maximally informative" 的 memory。Paper 没有讨论这个潜在 bias。

---

## 9. 与相关工作的 Connection

### 9.1 Memory 类别谱系

Paper 把 agent memory 分为三类:
- **Parametric** (ROME [\[20\]](https://arxiv.org/abs/2202.05262)): 改 weights,plasticity 差
- **Token-level** (RAG [\[21\]](https://arxiv.org/abs/2005.11401), MemGPT [\[22\]](https://arxiv.org/abs/2310.08560), Mem0 [\[23\]](https://arxiv.org/abs/2504.19413)): density 低
- **Latent** (MemGen, SoftCoT, FlashMem): density 高,但 computational cost 不同

FlashMem 在 latent 谱系内定位为 **intrinsic + parameter-light**,而 MemGen 是 **extrinsic + parameter-heavy**。

### 9.2 与 Transformer-XL 的联系

Dai et al. 2019 [\[24\]](https://arxiv.org/abs/1901.02860) 的 Transformer-XL 也是 reuse 之前 segment 的 hidden states (作为 memory),但它的 reuse 是 **direct concatenation** — 把之前 segment 的 hidden states 直接拼到当前 segment 的 input 上。

FlashMem 更进一层 — 不 reuse hidden states 作为 **input**,而是 reuse **KV cache** 作为 **attention target**。这避免了 representation 重新 encode 的 cost,只 query 现成的 K, V。

这是很关键的差别 — Transformer-XL 还是 "input concatenation" 思路,FlashMem 是 "attention target reuse" 思路,后者更轻量。

### 9.3 与 Memorizing Transformers 的联系

Wu et al. 2022 [\[25\]](https://arxiv.org/abs/2203.08913) 的 Memorizing Transformers 用 kNN 检索过去的 (key, value) pairs 加到当前 attention 中。这和 FlashMem 的 Shared-KV 思路有相似 — 都是 reuse 过去的 K, V。

但 Memorizing Transformers 是 **all history retrieval** (用 kNN 找 top-k relevant past K,V),FlashMem 是 **synthesized memory retrieval** (consolidator 生成 K 个 query vectors 去 attend backbone KV cache)。前者 retrieve raw past states,后者 synthesize new queries。

**Intuition**: Memorizing Transformers = "翻历史相册",FlashMem = "写一个 query 让系统找相关历史"。

### 9.4 与 Continual Learning 的联系

FlashMem 的 "Cognitive Monitor" 概念让我想到 continual learning 中的 **plasticity-stability dilemma** [\[26\]](https://arxiv.org/abs/2304.07911):
- Plasticity: model 能学新东西
- Stability: model 不忘旧东西

FlashMem 通过 **conditional memory injection** (只在 high entropy 时触发) 实现 "stability by default, plasticity on demand"。这比 always-on memory (如 SoftCoT) 更 stability-preserving。

### 9.5 与 Test-Time Compute Scaling 的联系

FlashMem 的 Cognitive Monitor 本质上是 **adaptive test-time compute** — 在难问题上多花 compute (生成 memory),在简单问题上少花 compute。这与 OpenAI o1 [\[27\]](https://openai.com/o1/), DeepSeek R1 [\[28\]](https://arxiv.org/abs/2501.12948) 的 test-time compute scaling 思路呼应,但 FlashMem 把 compute 花在 **memory consolidation** 上,而非 chain-of-thought generation 上。

这是两种不同的 test-time scaling 路径:
- **CoT scaling**: 生成更多 reasoning tokens (serial depth)
- **Memory scaling**: 生成更高密度的 memory tokens (parallel density)

FlashMem 是后者,理论上更适合 **long-horizon agent** (需要 persistent memory) 而非 **single-turn reasoning** (需要更深的 thought chain)。

---

## 10. Memory Interpretability 缺失

Section D 的 visualization 只展示了 **attention patterns** — consolidator attend 到哪里,backbone attend 到 memory tokens 的程度。但缺少关键分析:

**Memory tokens 到底 encode 了什么?**

如果能做 probing 实验:
1. 训练一个 linear probe 从 memory token $m_i$ 预测原始 context 中的某些属性 (e.g., 数学问题的 variables, 代码的 function names)
2. 检查 memory tokens 在 representation space 中的 geometry — 它们是 orthogonal 还是 redundant?
3. 对比不同 context 下生成的 memory tokens 的 similarity — 是否能 cluster 出 "memory types"?

这种分析能告诉我们 memory consolidation 学到了什么 **abstraction**。Paper 现在只展示了 "memory works",没有展示 "memory learns what"。

参考 Anthropic 的 Mechanistic Interpretability 工作 [\[29\]](https://transformer-circuits.pub/) — 如果能 circuits-level 分析 consolidator,会非常 illuminating。

---

## 11. 一个未触及的方向: Memory Lifecycle

Paper 假设 memory 一旦生成就 **permanent**。但在 long-horizon agent 中,memory 应该有 **lifecycle**:
- **Consolidation**: 把 short-term context 压缩成 memory (FlashMem 做了)
- **Maintenance**: 定期 refresh / update memory (paper 没做)
- **Forgetting**: 主动 discard 过时 memory (paper 没做)
- **Retrieval**: 按需 retrieve 相关 memory (paper 用 always-on attention,没有 selective retrieval)

人类 memory 的 forgetting curve [\[30\]](https://en.wikipedia.org/wiki/Forgetting_curve) 表明,forgetting 是 feature 不是 bug。一个理想的 agent memory 系统应该有这些 lifecycle mechanisms。

FlashMem 现在的 memory 是 **append-only** — 每个 high-entropy step 都 generate 8 个 tokens append 进去。在 super-long horizon (e.g., 1000+ steps) 下,memory tokens 会累积太多,eventually 占满 context window。

这是 FlashMem scale 到 truly long-horizon agent 的一个 open problem。

---

## 12. Implementation 细节中的几个亮点

### 12.1 Deterministic Memory Generation (Section A.3)

Memory tokens 用 greedy search 生成,而 text tokens 用 sampling。这个 asymmetry 很有意思 — 它假设 memory 应该 deterministic,但 text 可以 stochastic。

**Potential improvement**: 可以 explore stochastic memory generation + ensembling — 生成多个 memory candidates,选 entropy 最高的一个。这会增加 compute 但可能提升 memory diversity。

### 12.2 Gradient Clipping at 0.53

这个数字看起来 magic。0.53 不是 0.5,也不是 1.0。可能这个值是 hyperparameter search 找出来的,但 paper 没解释。Gradient clipping 对于 backprop through frozen backbone 很关键 — 因为 frozen layers 不会 update,gradients 累积到 memory tokens 时可能 explode。

### 12.3 K=8 的选择

为什么是 8 个 memory tokens?不是 4,不是 16?Paper 在 Table 6 给了 value 但没给 ablation。直觉上 K 应该和 task complexity 相关 — math reasoning 可能 K=4 够,code generation 可能需要 K=16。

参考 Gist Tokens [\[13\]](https://arxiv.org/abs/2304.08497) 用 1 gist token 压缩 prompt,效果就不错。FlashMem 用 8 个,可能保守了一些。

---

## 13. Potential Extensions 我会想看的

### 13.1 Hierarchical Memory

FlashMem 现在所有 memory tokens 是 flat 的。如果引入 hierarchy:
- **Episodic memory**: 每个 high-entropy event 一个 memory chunk
- **Semantic memory**: 跨 episodes 的 abstractions
- **Procedural memory**: 学到的 reasoning patterns

这是 cognitive science [\[31\]](https://en.wikipedia.org/wiki/Memory) 中经典分类。FlashMem 可以扩展成 episodic memory,但 semantic / procedural memory 需要不同 mechanism。

### 13.2 Memory Verification

现在 memory 是 consolidator 单向生成的。如果加入 verification step:
1. Consolidator 生成 memory candidates
2. 另一个 verifier (or backbone 自己) 评估 memory 是否 faithful to context
3. Reject 不 faithful 的 memory

这是 self-consistency [\[32\]](https://arxiv.org/abs/2203.11171) 思路应用到 memory generation 上。

### 13.3 Multimodal Extension

Paper 在 Limitations 提到 multimodal 是 future work。VisMem [\[3\]](https://arxiv.org/abs/2511.11007) 和 CoMEM [\[33\]](https://arxiv.org/abs/2505.17670) 已经做了 VLM 版本,但都是 segregated architecture。如果 FlashMem 扩展到 VLM:
- Visual tokens 的 KV cache 也可以 reuse
- Cognitive Monitor 可能需要考虑 vision-specific attention patterns
- Visual memory consolidation 可能需要 different architecture

### 13.4 Memory-Based Curriculum Learning

如果 agent 在不同任务上 use FlashMem,生成的 memory tokens 可以 **跨任务 transfer**。比如 math reasoning 中学到的 memory pattern 可以 transfer 到 code reasoning。这是 meta-learning 的思路 — memory 作为 inductive bias 跨 episode transfer。

---

## 14. 总结: FlashMem 的真正贡献

回到 Karpathy 视角,我会这样评价 FlashMem:

**真正的贡献**: 不是某个 specific technique,而是 **重新定义了 latent memory 的 efficiency frontier**。Paper 用一个简单 observation (backbone 的 KV cache 已经是 sufficient statistic) 推翻了整个 segregated architecture paradigm 的 efficiency 假设。

**真正的 insight**: Memory generation 不应该是 "additional forward pass",应该是 "computation reuse"。这个 principle 可以推广到很多地方 — 比如 retrieval-augmented generation (RAG) 也可以 reuse retriever 的内部 states,而不是只拿 final embeddings。

**真正的 limitation**: Sufficient statistic claim 在 long-context 下存疑,memory capacity 瓶颈明显,缺少 lifecycle management。这些是 paper 没解决但 future work 必须面对的问题。

**真正的 inspiration**: Cognitive Monitor 这个 parameter-free gating mechanism 太优雅了。Attention entropy 作为 "model confusion" 的 real-time proxy,这个 idea 可以独立于 FlashMem 用在很多地方 — 比如 adaptive test-time compute, early exit, dynamic depth switching。

如果让我预测 follow-up 方向:
1. **Hierarchical FlashMem**: 多层 memory,不同 time scale
2. **Multimodal FlashMem**: 扩展到 VLM
3. **Memory Lifecycle**: 加 forgetting 和 maintenance
4. **Circuit-level Interpretability**: 理解 consolidator 学到的 abstractions
5. **Theoretical Analysis**: 严格化 sufficient statistic claim,给出 information-theoretic bound

FlashMem 是一个 elegant 的 system paper,它提醒我们:有时候最优雅的 solution 是 **不要做额外的事**,而是 **复用已经做的事**。

---

## References

- [\[1\] MemGen](https://arxiv.org/abs/2509.24704)
- [\[2\] SoftCoT](https://arxiv.org/abs/2502.12134)
- [\[3\] VisMem](https://arxiv.org/abs/2511.11007)
- [\[4\] Nikolaou et al. - LLM Injectivity](https://arxiv.org/abs/2510.15511)
- [\[5\] Kuhn et al. - Semantic Uncertainty](https://arxiv.org/abs/2302.09671)
- [\[6\] Farquhar et al. - Nature Hallucination Detection](https://www.nature.com/articles/s41586-024-07421-0)
- [\[7\] Xiao et al. - Attention Sinks (StreamingLLM)](https://arxiv.org/abs/2309.17453)
- [\[8\] Darcet et al. - ViT Registers](https://arxiv.org/abs/2304.09408)
- [\[9\] LoRA](https://arxiv.org/abs/2106.09685)
- [\[10\] Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)
- [\[11\] SnapKV](https://arxiv.org/abs/2404.14469)
- [\[12\] H2O](https://arxiv.org/abs/2306.14048)
- [\[13\] Gist Tokens](https://arxiv.org/abs/2304.08497)
- [\[14\] Geva et al. - FFN as Key-Value Memories](https://arxiv.org/abs/2012.14913)
- [\[15\] RoPE](https://arxiv.org/abs/2104.09864)
- [\[16\] LongRoPE](https://arxiv.org/abs/2402.13753)
- [\[17\] Kadavath et al. - Self-Knowledge](https://arxiv.org/abs/2207.05221)
- [\[18\] RLHF](https://arxiv.org/abs/2203.02155)
- [\[19\] DPO](https://arxiv.org/abs/2305.18290)
- [\[20\] ROME](https://arxiv.org/abs/2202.05262)
- [\[21\] RAG](https://arxiv.org/abs/2005.11401)
- [\[22\] MemGPT](https://arxiv.org/abs/2310.08560)
- [\[23\] Mem0](https://arxiv.org/abs/2504.19413)
- [\[24\] Transformer-XL](https://arxiv.org/abs/1901.02860)
- [\[25\] Memorizing Transformers](https://arxiv.org/abs/2203.08913)
- [\[26\] Continual Learning Survey](https://arxiv.org/abs/2304.07911)
- [\[27\] OpenAI o1](https://openai.com/o1/)
- [\[28\] DeepSeek R1](https://arxiv.org/abs/2501.12948)
- [\[29\] Anthropic Circuits](https://transformer-circuits.pub/)
- [\[30\] Forgetting Curve](https://en.wikipedia.org/wiki/Forgetting_curve)
- [\[31\] Memory Types](https://en.wikipedia.org/wiki/Memory)
- [\[32\] Self-Consistency](https://arxiv.org/abs/2203.11171)
- [\[33\] CoMEM](https://arxiv.org/abs/2505.17670)
- [PyramidKV](https://arxiv.org/abs/2406.02069)
- [Memento - Fine-tuning Agents without Fine-tuning LLMs](https://arxiv.org/abs/2508.16153)
- [Memory Survey](https://arxiv.org/abs/2512.13564)
