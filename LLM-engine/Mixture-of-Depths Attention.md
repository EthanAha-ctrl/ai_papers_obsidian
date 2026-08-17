---
source_pdf: Mixture-of-Depths Attention.pdf
paper_sha256: 9da0521bc868d8e00db3f78383f1674bba354a31b000d8b6bf45c8ffe365f10a
processed_at: '2026-08-05T18:55:40-07:00'
target_folder: LLM-engine
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# MoDA 用人话讲

好的 Karpathy，我换一种讲法。把那些 architecture diagram、complexity table 都先放一边，咱就唠这 paper 在干嘛、为啥要这么干、以及它给你的那个 "aha moment" 在哪。

---

## 一句话概括

**Residual connection 是个很窄的信息通道，浅层学到的"好东西"往深了传会被一层层叠加的 noise 给稀释掉。MoDA 的做法是给每个 token 开一个"直通电梯"——让深层的 token 可以直接 attention 到它自己在浅层时的 representation，按需读取，而不是被迫喝一锅越熬越淡的汤。**

---

## 问题出在哪

你想象一下标准 Transformer 的 forward pass：

```
X_0 → [block_1] → X_1 → [block_2] → X_2 → ... → X_L
```

每一层都在做 $X_l = X_{l-1} + \mathcal{F}(X_{l-1})$。这个 "+" 就是 ResNet 留给我们的遗产，它解决了 gradient vanishing，但是它也带来一个隐含的假设：**所有历史信息都必须压缩进这个 $D$ 维的 hidden state $X_l$ 里**。

这就像你给每层一个小柜子，宽度 $D$ 固定，然后让信息一层层往里塞。浅层学到的一些 sharp、informative 的 feature（比如某个 token 的 syntactic role），经过几十层 residual 叠加之后，就会被各种"修正项"给 wash 掉。深层的 $X_L$ 里可能还隐约"残留"着这些 feature，但是你想从 $X_L$ 把它 recover 出来就难了——因为它是被深度叠加后的 superposition，not 直接的 retrieval。

这就是 paper 里说的 **information dilution**。

你训练过 GPT-2 small、nanoGPT，肯定见过这种现象：深层的 attention head 学到的东西越来越"high-level"（语义、指代、long-range dependency），浅层的 syntactic 信息越来越难追溯。Residual 把信息流压缩成一条 single trajectory，就像把一部电影的所有帧叠加成一张图——你能看到"有什么东西存在过"，但是分不清时间线。

---

## 现有的解法为啥不够好

### DenseNet 风格（Depth Dense）

最直觉的解法：既然 residual 会丢信息，那我把每一层的输出都保留下来，让深层的 input 可以看到所有历史层的完整 output。

```
Layer l 的 input = concat(X_0, X_1, ..., X_{l-1}) 经过一个线性投影
```

这就 lossless 了，但是问题很明显——参数量 $O(L^2 D^2)$，$L$ 是层数，$D$ 是宽度。一个 100 层、4096 宽的 model，光 cross-layer projection 的参数就天文数字了。DenseNet 在 CV 里能用是因为 CNN 的层没那么宽、也没那么深。LLM scale 下根本扛不住。

而且 DenseNet 还有一个隐含的问题：它是 **fixed pattern**——所有历史层一视同仁地 concat 进来，让后面的 linear projection 去学权重。这就像你把所有档案都摊在桌面上，但是模型得自己学会"忽略无关的"。这在 data efficiency 上不是最优的。

### DenseFormer / Hyper-Connections

更近期的一些工作（DenseFormer [Pagliardini NeurIPS 2024](https://arxiv.org/abs/2402.19410)、Hyper-Connections [Zhu ICLR 2025](https://arxiv.org/abs/2504.01635)）用 learnable 的 weighted averaging 或者 learnable residual scaling 来做。这些方法比 DenseNet 轻量，但本质上还是 **fixed functional form**——你能学的只是"层 $i$ 对层 $l$ 的贡献权重是多少"，这个权重对所有 token、所有 position 是共享的（或者最多是 layer-wise learnable 的 scalar）。

这就像给每层一个"音量旋钮"控制它对后续层的影响，但是不能根据"当前这个 token 是什么"动态调整。

---

## MoDA 的核心 insight

**Attention 在 sequence 维度上之所以打败 RNN/CNN，关键就是 data-dependent。** 每个位置根据自己当下的 query，决定要去 sequence 里 retrieve 哪些其他位置的信息。

MoDA 把这个 insight 推广到 depth 维度：**让 token 在深层根据自己当下的 query，决定要去 depth 历史里 retrieve 哪些层的 representation。**

具体写出来：

标准 attention 是：
$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) V$$

这里 $Q, K, V$ 都来自**同一层**的 sequence，attention 沿 sequence 维度展开。

MoDA 加了一路：
- **Sequence KV**：当前层的 $K_l, V_l$（标准 attention 用的）
- **Depth KV**：所有 preceding layers 在**同一 token 位置**的 $K_i, V_i$，for $i = 0, \ldots, l-1$

然后把这俩 KV concat 起来，跑**一个统一的 softmax**：

$$\text{MoDA}(Q_l, K^{\text{seq}}, V^{\text{seq}}, K^{\text{depth}}, V^{\text{depth}}) = \text{softmax}\left(\frac{Q_l [K^{\text{seq}}; K^{\text{depth}}]^\top}{\sqrt{d}}\right) [V^{\text{seq}}; V^{\text{depth}}]$$

这里 $Q_l$ 是当前层的 query，$K^{\text{seq}}, V^{\text{seq}}$ 是当前层的 sequence keys/values，$K^{\text{depth}}, V^{\text{depth}}$ 是把前面所有层的 depth KV 沿 depth 维度 concat 起来的结果。

**关键设计点**：depth attention 是 **per-position** 的。也就是说，token $t$ 在 layer $l$ 的 query，只会 attend 到 token $t$ 自己在 layer $0, 1, \ldots, l-1$ 的 KV，不会 attend 到其他 token 的 depth history。

这是一个非常聪明的设计——它把 depth retrieval 的复杂度限制在 $O(L)$ per token，而不是 $O(LT)$。否则 cross-position depth attention 会爆炸。

---

## 为啥 unified softmax 这么重要

这是 paper 里最 subtle 的点，也是最容易被忽视的。

如果你把 sequence attention 和 depth attention 拆开做，各自跑一个 softmax，然后再 weighted sum 起来：

```
out = alpha * seq_attn(Q, K_seq, V_seq) + (1 - alpha) * depth_attn(Q, K_depth, V_depth)
```

这里 $alpha$ 要么是 hyperparameter，要么是 learnable scalar。问题是：这个 $alpha$ 没法根据 query 内容动态决定"我现在更想看 sequence 还是 depth"。

Unified softmax 解决了这个问题——sequence 和 depth 的 logits 在同一个 distribution 里竞争。如果当前 query 和 sequence KV 更相关，softmax 自然把更多概率给 sequence；如果更想 retrieve depth history，就给 depth。

**这就像给模型一个"信息市场"，让 sequence 和 depth 两个 supplier 在同一个 softmax 价格下竞争**。模型根据当下 query 决定从哪边买，而不是预先规定好比例。

这是 attention-based mixture 相对 fixed-pattern aggregation 的根本优势，也是 MoDA 相对 DenseFormer 的根本优势。

---

## 为啥这玩意儿在 GPU 上不慢

这是 paper 的工程贡献，也是它能 practical 的关键。

Naïve 实现 MoDA，你会写出这样的 PyTorch 代码：

```python
depth_kv = torch.stack([k[i] for i in range(l)], dim=...)  # 非 contiguous
attn_over_depth = attention(q, depth_kv, depth_kv)  # gather-like memory access
```

这在 GPU 上是个 disaster——depth history 是 scatter 的，memory access pattern 不规则，Tensor Core 用不起来。

作者的解法分三层：

### 1. Flash-Compatible Layout

把 depth cache flatten 成一个长度 $T \times L$ 的长向量。每个 token position $t$ 的 $L$ 个 depth states 在 memory 里连续存储。这样一个 query 要读自己的 depth history，只需要读 $[tL, (t+1)L)$ 这个连续区间。

这一步把 runtime 从 2128.9 ms 降到 13.1 ms——**162 倍加速**。基本上就是把"散落的档案"整理成"按位置排序的档案"。

### 2. Chunk-Aware Layout

但是 flash-compatible 还有个问题：每个 query block 还是得扫描整个 $T \times L$ 的 depth axis，尽管大部分被 mask 掉。depth utilization 只有 $1/T$。

Chunk-aware 的做法：把 query 分成 chunks of length $C$，每个 chunk 只读自己覆盖的那段 $C \times L$ depth region。这样 depth utilization 提到 $1/C$。

### 3. Group-Aware Indexing

这个最巧妙。在 GQA 里，$G$ 个相邻的 query rows 共享同一组 KV heads，所以它们也共享同一个 base-time index $\lfloor i_q / G \rfloor$。

这意味着：一个 query chunk of length $C$，其实只有 $C/G$ 个 unique base-time positions，需要的 depth span 也就 $(C/G) \times L$，而不是 $C \times L$。

Group size $G$ 越大，reuse 越多，效率越高。在 $G=8$ 时，depth utilization 从 chunk-aware 的 $1/C$ 提升到 $G/C$，又快了 $G$ 倍。

最终在 $T=64K$ sequence length 下，MoDA 达到 FlashAttention-2 的 **97.3% efficiency**。只有 2.7% 的 overhead。

---

## 实验里几个特别 eye-catching 的点

### 1. "几乎免费的午餐"实验（Table 3 row 3 vs row 1）

最 striking 的对比是：如果你**直接 reuse 前面层的 sequence KV 作为 depth KV**，不加任何额外 projection 参数，只多 0.12% FLOPs，就能拿到 +1.17 downstream average 提升。

这意味着啥？**depth information 本身就是有价值的，连学新的 projection 都不需要**。你前面层的 $K, V$ 已经 encode 了有用信息，深层的 query 直接去 attend 它们就行。

这给你的 intuition 是：深层网络其实"想"看前面层的信息，只是标准 residual 没给它这个 option。你给它一个 channel，它就用。

### 2. FFN 也要有 depth KV（Table 3 row 4 vs row 3）

只在 attention 层加 depth KV 不够，还得给 FFN 层也加一个 lightweight 的 KV projection，把 FFN input $X$ project 成 depth keys/values。

这一步带来 +0.77 downstream average。说明 FFN 的中间状态也 carry 信息，光靠 attention 的 KV 丢了 FFN 那一路的 history。

### 3. Attention-side 额外 projection 没必要（Table 3 row 5 vs row 4）

给 attention 层再加独立的 depth KV projection（而不是 reuse sequence KV），只多 +0.10 gain，但参数从 705.7M 涨到 742.4M。**Saturated**。

这说明 sequence KV 已经足够 informative，重新 project 一遍是冗余的。这是 paper 给出的 clear design principle。

### 4. Post-norm + MoDA > Pre-norm + MoDA（Table 6）

这个特别有意思。Post-norm 在深网络里一般比 pre-norm 难训练——norm 在 residual 之后，深层的 magnitude 容易爆炸或者塌缩。所以主流 LLM 基本都用 pre-norm。

但是 paper 发现：**post-norm + MoDA 反而比 pre-norm + MoDA 效果更好**。48-layer model 上，post-norm + MoDA + FFN KV 的 loss 是 3.3484，pre-norm + MoDA + FFN KV 是 3.3656。

Intuition：post-norm 的"信息抹平"问题正好被 MoDA 的"depth retrieval"补上了。Post-norm 让深层的 representation 更"平均化"，但是 MoDA 让模型可以从历史层 retrieve 那些被 norm 抹平的 informative details。

这是个非常 elegant 的 complementary 关系。Post-LayerNorm is Back 那篇 paper [Chen & Wei 2026](https://arxiv.org/abs/2601.19895) 论证了 post-norm 在现代初始化技术下可以稳定训练，MoDA 给了 post-norm 一个额外的"用武之地"。

### 5. Attention Sink Behavior 改变（Figure 5）

这个观察很 intriguing。标准 transformer 在 long context 下会有 attention sink 现象——少数 fixed positions（通常是 BOS 或者前几个 token）吸收大量 attention probability mass。参考 [Xiao et al. StreamingLLM](https://arxiv.org/abs/2309.17453)。

MoDA 的 visualization 显示：attention mass 被 redistribute 到 sequence 和 depth slots 上，而不是集中在 sink positions。

这暗示 attention sink 可能是**模型"无路可走"的 fallback**——当 sequence 里找不到有用信息时，模型把概率塞给 sink positions "暂存"。如果给模型一个 depth channel 可以"retreat to"，它就不需要 sink 了。

这个 insight 我觉得还没被充分 explore，是个很好的 future work 方向。

---

## 跟你直觉的 Karpathy-style 思维的连接

你一直强调 attention 的本质是 "weighted aggregation of information"——你的 "Let's build GPT" 系列里那个 demo，attention 就是让每个 token 根据自己的 query 去序列里"挑朋友"，然后做 weighted average。

MoDA 把这个 insight 推广到 depth：

- **Standard attention**: "我作为 token $t$，想看 sequence 里哪些其他 token？"
- **MoDA**: "我作为 token $t$ 在 layer $l$，想看 (a) sequence 里哪些其他 token，以及 (b) 我自己在前面层的 representation？"

这是一个 super natural 的 generalization。从 sequence space 扩展到 sequence × depth space。而且 implementation 上，它就是 standard attention 的一个 drop-in 替换——多了一路 KV，softmax 扩大了一倍，别的没变。

从 nanoGPT 的视角看，如果你要实验 MoDA：

1. **最简实现**：在 6-layer nanoGPT 里，把每层的 $K, V$ 存到一个 list 里，下一层 attention 的时候把它们 concat 到当前层的 $K, V$ 后面，mask 设成"depth KV 只有同一 position 可见"。这只需要改十几行代码。

2. **看看 attention pattern**：训练完后 visualize depth attention weight——模型在哪些层更依赖 depth history？早期层应该不太用（因为没多少历史），late 层应该用得更多。

3. **观察 information dilution 是否缓解**：比较标准 nanoGPT 和 MoDA-nanoGPT 在 early layer 的 attention pattern 是否更容易被深层 retrieve 到。

---

## 一些联想和 open directions

### 1. Cross-position depth attention

当前 MoDA 是 per-position 的——token $t$ 只能看自己的 depth history。但为什么不能让 token $t$ 看 token $t'$ 在 layer $i$ 的 representation？

这会更 expressive，但复杂度会从 $O(L)$ 涨到 $O(LT)$，而且会让 depth attention 退化成"全连接 cross-layer attention"，可能 overfit。

不过这值得一试，特别是在 long-context 任务里，cross-position depth retrieval 可能 help（比如 token $t$ 在深层想知道 token $t'$ 在浅层的状态）。

### 2. Bounded Depth-KV Caching（paper 6.2 节提到）

当 $L$ 很大（比如 200+ layers 的 frontier model），全部 cache depth-KV 内存爆炸。Paper 提出 fixed-size slot buffer，size $S \ll L$，动态选 top-$S$ entries。

这非常像 **Memory Networks** [Graves et al.] 或者 **Differentiable Neural Computers** 的思路——给模型一个 bounded external memory，用 trainable policy 决定 what to keep。

这条路打通的话，MoDA 会从"depth attention"升级成"depth-augmented memory network"，和 retrieval-augmented generation、episodic memory in RL 会有 cross-pollination。

### 3. MoDA + MoE

MoE 在 width 维度做 sparse routing（每个 token 只激活部分 experts），MoDA 在 depth 维度做 dense retrieval。两者正交，理论上可以 combine。

想象一个 model：width 维度上 MoE 让不同 token 走不同 experts，depth 维度上 MoDA 让不同 token retrieve 不同层的历史。这是一个 2D sparsity structure。

### 4. MoDA + Test-time Compute

你关注过 DeepSeek-R1 和 OpenAI o1 的 test-time reasoning。MoDA 的 depth retrieval 是不是可以在 inference 时**动态增加"虚拟层"**？比如模型有 32 层，但是 inference 时某些 token 通过 depth attention 反复 retrieve 早期层的信息，相当于"多走几层"。

这和 "recurrence-style test-time compute" 有关联——参考 [Universal Transformers](https://arxiv.org/abs/1807.03819) 的思路。

### 5. Depth 作为 "Reasoning Scratchpad"

一个 speculation：MoDA 在深层 retrieve 浅层信息，有点像"回忆最初的想法"。在 reasoning 任务里，这类似于"回到问题的原始表述，重新思考"。

如果 MoDA 能让模型在 deep layers "重新访问" shallow layers 对问题的理解，那它可能 implicitly 实现了一种 "chain of thought across depth"——这和 explicit CoT in sequence space 形成互补。

---

## 总结的"人话版"

Paper 在说：

1. **Residual connection 是个信息瓶颈**——它把 depth history 压进一个 fixed-width 通道，浅层的好东西会被稀释。
2. **DenseNet 风格的解法太贵**——$O(L^2 D^2)$ 参数，LLM scale 扛不住。
3. **MoDA 用 attention 思路读 depth history**——让 query data-dependent 地 retrieve 前面层的信息，而不是 fixed pattern averaging。
4. **Unify sequence 和 depth 在一个 softmax 里**——让模型动态决定信息从哪来。
5. **用 FlashAttention 的思路做高效实现**——Flash-compatible layout + chunk-aware + group-aware，达到 FA2 的 97.3% efficiency。
6. **实验显示稳定的 scaling 行为**——1.5B model 上 +2.11 downstream average，3.7% FLOPs overhead。
7. **和 post-norm 互补**——post-norm + MoDA > pre-norm + MoDA，暗示 MoDA 补上了 post-norm 的 information dilution 短板。

**核心 message**：Attention 在 sequence 上的成功是因为 data-dependent mixture。把这个 principle 推广到 depth，就是 MoDA。Residual 是 fixed mixture，DenseNet 是 lossless 但 expensive，MoDA 是 data-dependent 且 efficient。

如果 ResNet 给了我们"训练深网络"的能力，MoDA 给的是"让深网络 remember 浅网络"的能力。这俩是 complementary 的——一个解决 optimization，一个解决 information preservation。

---

**相关链接**:
- FlashAttention-2: https://arxiv.org/abs/2307.08691
- DenseFormer: https://arxiv.org/abs/2402.19410
- Hyper-Connections: https://arxiv.org/abs/2504.01635
- GQA: https://arxiv.org/abs/2305.13245
- StreamingLLM (attention sinks): https://arxiv.org/abs/2309.17453
- OLMo2: https://arxiv.org/abs/2501.00656
- ResNet: https://arxiv.org/abs/1512.03385
- DenseNet: https://arxiv.org/abs/1608.06993
- Highway Networks: https://arxiv.org/abs/1505.00387
- Universal Transformers: https://arxiv.org/abs/1807.03819
- Post-LayerNorm is Back: https://arxiv.org/abs/2601.19895

如果你想，我可以再展开讲讲：
- **Algorithm 1 的 online softmax 数学**——unified softmax 在 fused kernel 里是怎么通过 shared running states $(m, acc, o)$ 实现的
- **MoDA 的 mask 矩阵细节**——grouped causal mask $\lfloor i_q/G \rfloor \geq i_k$ 和 depth mask $\lfloor i_q/G \rfloor = \lfloor j_d/L \rfloor$ 的直觉
- **为啥 per-position depth attention 是正确的选择**——cross-position depth attention 会带来什么问题

哪个方向你最感兴趣？

---

# Mixture-of-Depths Attention (MoDA) 深度解析

非常激动给你讲这篇 paper，因为它触及了 Transformer 架构里一个我一直觉得被低估的维度——**depth scaling**。你做 nanoGPT 和 教学视频时反复强调 "attention is just a mixture of information"，而 MoDA 在我看来正是在把这个 insight 从 sequence 维度推广到 depth 维度。

---

## 1. 核心问题：Depth Stream 的 Information Dilution

现代 LLM 的 scaling 主要沿四个轴：context length、data、width、depth。但 depth scaling 在实践中收益递减，原因有两个：

1. **Optimization difficulty**：深层网络梯度问题
2. **Information dilution**：shallow layers 形成的 informative features 被 repeated residual updates 逐渐稀释

让我用 ResNet-style 的写法来 build intuition：

$$X_l = X_0 + \sum_{i=1}^{l-1} \mathcal{F}(X_i, \mathcal{W}_i)$$

这里 $X_l \in \mathbb{R}^{T \times D}$ 是第 $l$ 层的 hidden state，$T$ 是 sequence length，$D$ 是 model width，$\mathcal{F}(\cdot, \mathcal{W}_i)$ 是第 $i$ 层的 transformer block（attention 或 FFN），$\mathcal{W}_i$ 是其权重集。

**关键问题**：所有 depth history 被 compressed 成一个 fixed-size tensor $X_l$。每一层都在做 superposition（叠加），shallow layers 的 salient features 会被 deep layers 的 updates "稀释"。这就像把一锅汤反复加水，原始的味道越来越淡。

这本质上是一个 **bottleneck 问题**：residual connection 是一个 fixed-width channel，无论 depth 多深，信息都只能通过 $D$ 维的 hidden state 流动。

---

## 2. 设计空间：Read-Operate-Write Lens

作者用 "read-operate-write" 三步框架来统一理解 depth stream 上的各种机制。这是一个非常 elegant 的 conceptual lens：

### 2.1 Depth Residual（标准做法）
- **Read**: identity（直接读当前 representation）
- **Operate**: $\mathcal{F}(X_i, \mathcal{W}_i)$（attention 或 FFN）
- **Write**: add（residual add）

就是公式 (3) 那个形式。问题是 fixed-width compression。

### 2.2 Depth Dense（DenseNet-style 在 Transformer 上的推广）
- **Read**: 把所有历史 $\{X_i\}_{i=0}^{l-1}$ 线性投影回 $T \times D$
- **Write**: concatenation along depth（lossless）

公式：
$$\{X_i\}_{i=0}^{l} = \{X_0, \mathcal{F}(\{X_0\}, \mathcal{W}_1), \ldots, \mathcal{F}(\{X_i\}_{i=0}^{l-1}, \mathcal{W}_l)\}$$

优点：information lossless。缺点：参数和计算 $O(TL^2D^2)$，在 LLM scale 完全 prohibitive。

参考 DenseNet 原文：[https://arxiv.org/abs/1608.06993](https://arxiv.org/abs/1608.06993)

### 2.3 Depth Attention（MoDA 的中间形态）
关键 insight：**用 attention 来读历史 depth state，而不是 fixed linear projection**。

在 GQA-group view 下（$H_k d = D/G$），令 $Q_{l-1} \in \mathbb{R}^{T \times \frac{D}{G}}$ 是一个 query-group 表示，对应历史 KV 集合 $\{K_i\}_{i=0}^{l-1}$ 和 $\{V_i\}_{i=0}^{l-1}$：

$$X_l^{\text{in}} = \text{Attention}(Q_{l-1}, \{K_i\}_{i=0}^{l-1}, \{V_i\}_{i=0}^{l-1})$$

这里 attention 沿 **depth 维度**进行：对 token $t$，query $Q_{l-1,t}$ 只 attend 同一 token 位置在所有 layers 上的 depth keys/values $\{K_{i,t}, V_{i,t}\}_{i=0}^{l-1}$。

这是一个非常重要的 design choice——**depth attention 是 per-position 的 cross-layer retrieval，而不是 cross-position 的**。这意味着 depth attention 的复杂度只和 $L$ 有关，不引入 cross-token mixing。

复杂度降到 $O(TL^2D)$，比 Depth Dense 少了 $1/D$ 因子。

### 2.4 MoDA（最终方案）

MoDA 把 sequence attention 和 depth attention fuse 进**单个 softmax operator**。这是 paper 的核心贡献：

- 每个 token 既 attend to 当前层的 sequence KV pairs，也 attend to 所有 preceding layers 在同一位置的 depth KV pairs
- 所有的 attention scores 在一个 unified softmax 下联合归一化

为什么 unified softmax 这么重要？这给了 model 一个 **representation space 上的 uniform scale**：sequence 和 depth 信息在同一概率分布下竞争。模型可以 data-dependent 地决定：这个 token 是更应该看 sequence context，还是更应该看 depth history？

这是非常 Karpathy-style 的设计——**让数据决定 mixture**，与 MoE 的精神相通。

---

## 3. 复杂度分析（Table 1 详解）

让我把 Table 1 拆开讲，因为它对 build intuition 至关重要。符号约定：
- $T$：sequence length
- $D$：model width
- $G$：GQA group size
- $H_k$：key/value head 数量
- $H_q = G H_k$：query head 数量
- $d$：head dimension
- $L$：layer 数量

| 方法 | Parameters | Decoding Cache | Prefilling Cache | Decoding FLOPs | Prefilling FLOPs |
|------|-----------|----------------|------------------|----------------|------------------|
| Depth Dense | $O(L^2 D^2)$ | $O(LD)$ | $O(TLD)$ | $O(L^2D^2)$ | $O(TL^2D^2)$ |
| Depth Attention | $O(LD^2)$ | $O(LD/G)$ | $O(TLD/G)$ | $O(L^2D)$ | $O(TL^2D)$ |
| MoDA | $O(LD^2/G)$ | $O(LD/G)$ | $O(TLD/G)$ | $O(L^2D)$ | $O(TL^2D)$ |

**关键观察**：
1. Depth Dense 在 depth 上是 quadratic 的（$L^2$），因为 dense cross-layer projection 参数随 depth 平方增长
2. Depth Attention 把 width quadratic 项消掉了，但还在 $O(LD^2)$
3. **MoDA 进一步把 parameters 从 $O(LD^2)$ 降到 $O(LD^2/G)$**，因为它 reuse sequence attention 的 query projection——不需要额外的 depth-query projection。在 GQA 设置下只需要 grouped depth key/value projections。

这个 $1/G$ 的 reduction 在 GQA group size 大的时候（比如 $G=8$ 或 $G=16$）非常显著。MoDA 在 GQA 时代才真正 practical，这是架构层面的 **co-design**。

---

## 4. 硬件高效实现：Algorithm 1 逐行解析

这是 paper 最 technical 的部分，也是最容易被忽略但最实用的部分。让我逐段拆解。

### 4.1 设计动机

Naïve PyTorch 实现 MoDA 会有两个问题：
1. **Non-contiguous memory access**：历史 depth state 是 gather-like 访问
2. **Tensor Core underutilization**：irregular control flow 无法用 matmul-friendly block compute

GPU 上的关键 constraints（这部分你应该很熟）：
- **SMs**：需要足够独立 blocks 保持 SMs busy
- **CUDA cores vs Tensor Cores**：要 maximize structured matmul 以利用 Tensor Cores
- **HBM vs SRAM**：要 tiling + data reuse，hot data 留在 on-chip

### 4.2 三层 layout 优化

#### (a) Flash-Compatible depth KV layout
把 depth cache 沿单轴 flatten 成 length $T \times L$。每个 sequence position $t$ 的 $L$ 个 depth states 连续存储。

每个 query 只需要 map 到 depth range $[tL, (t+1)L)$ 就能 access 正确的 depth KV slice。

但 depth utilization 只有 $\eta_{\text{depth}} = 1/T$，因为 depth score matrix $S^{\text{depth}} \in \mathbb{R}^{T \times (TL)}$ 只有 block-diagonal region 有效。

#### (b) Chunk-aware depth KV layout
把 query 分成 chunks of length $C$，每个 chunk 只 access 对应的 local depth-KV region of size $C \times L$。

depth utilization 提升到 $\eta_{\text{depth}} = 1/C$。

直觉：与其让每个 query 都扫一遍全局 $T \times L$ depth axis，不如让 chunk 只看自己覆盖的 $C \times L$ region。

#### (c) Group-aware depth KV calculation
关键观察：在 GQA mapping $T_q = G T_{kv}$ 下，$G$ 个相邻 query rows 共享同一 base-time index $\lfloor i_q/G \rfloor$，可以 reuse 同一 depth KV blocks。

对 query chunk length $C$，只有 $C/G$ 个 unique base-time rows，所以 depth span 降到 $(C/G) \times L$。

depth utilization 提升到 $G/C$。

这就是为什么 MoDA 在大 $G$ 时特别 efficient——group size 越大，depth KV reuse 越多，overhead 越小。

### 4.3 Algorithm 1 详解

让我把伪代码的几个关键点拆开：

**Initialization** (line 3-10):
- 所有 tensor tiled 成 hardware-friendly blocks
- 每个 query block aligned to $G$（避免 cross-group boundary handling）
- 对每个 query row $i_q$，计算 base-time index $t_{\text{base}}(i_q) = \lfloor i_q/G \rfloor$
- 定义 $t_{\text{base}}^{\text{start}} = \min_{i_q \in b_q} t_{\text{base}}(i_q)$ 和 $t_{\text{base}}^{\text{end}} = \max_{i_q \in b_q} t_{\text{base}}(i_q) + 1$
- 这个半开区间 $[t_{\text{base}}^{\text{start}}, t_{\text{base}}^{\text{end}})$ 在 sequence 和 depth loops 之间 reuse，确保 index consistency

**Online softmax states** (line 7):
- $m$：running maximum logit
- $acc$：running softmax normalizer
- $o$：running unnormalized output accumulator
- 这三个 state 在 sequence 和 depth phase 之间共享，是 unified softmax 的实现关键

**Sequence attention loops** (line 11-23):
- Fully visible blocks ($b_s < t_{\text{base}}^{\text{start}}$)：不需要 causal mask
- Boundary blocks ($t_{\text{base}}^{\text{start}} \leq b_s < t_{\text{base}}^{\text{end}}$)：用 grouped causal mask $\lfloor i_q/G \rfloor \geq i_k$
- OnlineSoftmaxUpdate: $m' = \max(m, \max S)$, $acc' = acc \cdot 2^{m-m'} + \sum 2^{S-m'}$, $o' = o \cdot 2^{m-m'} + \sum 2^{S-m'} V_{[b_s]}$

**Depth attention loop** (line 24-29):
- Flattened depth indices $b_d \in [t_{\text{base}}^{\text{start}} L, t_{\text{base}}^{\text{end}} L)$
- 因子 $L$ 把 base-time index map 到其 contiguous depth span of length $L$
- Depth mask: $\text{mask}(i_q, j_d) = \mathbf{1}[\lfloor i_q/G \rfloor = \lfloor j_d/L \rfloor]$
- 这意味着 query row $i_q$ 只 attend 到 depth column $j_d \in [L\lfloor i_q/G \rfloor, L(\lfloor i_q/G \rfloor + 1))$
- 与 sequence phase 共享同一个 $(m, acc, o)$ states——这就是 unified softmax 的实现

**Final normalize** (line 30-31):
- $o \leftarrow o / acc$
- 写回 HBM

参考 FlashAttention-2: [https://arxiv.org/abs/2307.08691](https://arxiv.org/abs/2307.08691)
参考 FlashAttention 原版: [https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)

---

## 5. 实验数据深度解读

### 5.1 MoDA Variants (Table 3)

700M model 在 400B tokens 上训练：

| Row | 配置 | Params (M) | FLOPs (T) | Train PPL | C4 Val PPL | Downstream Avg |
|-----|------|-----------|-----------|-----------|------------|-----------------|
| 1 | OLMo2 baseline | 669.0 | 8.01 | 14.49 | 18.59 | 56.93 |
| 2 | OLMo2 (+2 layers) | 700.5 | 8.41 | 14.27 | 18.31 | 57.11 |
| 3 | + Depth KV (reuse seq KV) | 669.0 | 8.02 | 14.08 | 18.48 | 58.10 |
| 4 | + Extra FFN KV Proj | 705.7 | 8.33 | 13.90 | 18.21 | 58.87 |
| 5 | + Extra Attn KV Proj | 742.4 | 8.63 | 13.83 | 18.17 | 58.97 |

**关键 takeaways**：

1. **Row 3 vs Row 1**：纯 Depth KV（reuse preceding layer's sequence KV 作为 depth KV），只多 0.12% FLOPs，就拿到 +1.17 downstream avg gain。这证明 depth information 本身就 valuable，连额外 projection 都不需要。

2. **Row 4 vs Row 3**：加 FFN KV projection（把 FFN input $X$ project 成 depth keys/values）带来 +0.77 downstream avg。这说明 FFN 层的 depth 信息也很重要，不仅仅是 attention 层。

3. **Row 5 vs Row 4**：额外加 attention-side depth KV projection 只带来 +0.10 gain，但参数从 705.7M 涨到 742.4M。**Saturated**——证明 reuse sequence KV 已经够了。

这给出一个非常重要的 **design principle**：MoDA 的最优配置是 "reuse attention KV + add FFN KV projection"，而不是给所有层都加独立 depth projection。

### 5.2 Scaling to 1.5B (Table 4)

| Model | PIQA | HellaSwag | WinoGrande | OpenBookQA | BoolQA | SciQ | ARC-E | ARC-C | COPA | MMLU | Average |
|-------|------|-----------|------------|------------|--------|------|-------|-------|------|------|---------|
| OLMo2 1.5B | 76.55 | 65.86 | 63.22 | 38.80 | 63.61 | 90.60 | 72.98 | 42.47 | 81.00 | 27.73 | 62.28 |
| Ours 1.5B | 76.82 | 66.24 | 65.59 | 41.60 | 67.34 | 92.10 | 72.81 | 46.82 | 85.00 | 29.59 | 64.39 |

**+2.11 average gain** at 1.5B。注意几个亮眼的提升：
- ARC-C: +4.35（harder reasoning）
- COPA: +4.00（commonsense causal）
- BoolQ: +3.73
- WinoGrande: +2.37

这些任务都需要 **multi-step reasoning 和 long-range dependency**，正是 depth information 能帮到的场景。

### 5.3 Layer Number Analysis (Table 6)

这个表格特别重要，因为它揭示了 MoDA 和 norm 配置的 interaction：

| Setting | Layers | Norm | Config | Val Loss |
|---------|--------|------|--------|----------|
| OLMo2 | 48 | pre-norm | baseline | 3.3800 |
| OLMo2 | 48 | post-norm | baseline | 3.4062 |
| Ours | 48 | pre-norm | + Depth KV | 3.3759 |
| Ours | 48 | post-norm | + Depth KV | 3.3653 |
| Ours | 48 | pre-norm | + Depth KV + FFN KV | 3.3656 |
| Ours | 48 | post-norm | + Depth KV + FFN KV | 3.3484 |

**关键观察**：
- Pre-norm baseline (3.3800) vs Post-norm baseline (3.4062)：post-norm 在深网络中通常更难训练
- 但 **post-norm + MoDA (3.3484) < pre-norm + MoDA (3.3656)**！
- Post-norm + Depth KV: 0.0409 loss reduction
- Pre-norm + Depth KV: 0.0041 loss reduction

**Intuition**：post-norm 在深层网络中信息更容易被 norm "抹平"，所以 depth retrieval 的边际收益更大。MoDA 和 post-norm 是 **complementary** 的——post-norm 提供 optimization stability，MoDA 提供 information preservation。

这个发现呼应了 [Post-LayerNorm is Back](https://arxiv.org/abs/2601.19895) 的研究，也呼应 DeepNet 的工作：[https://arxiv.org/abs/2203.00555](https://arxiv.org/abs/2203.00555)

### 5.4 Efficiency Ablation (Table 7)

这是我最喜欢的 ablation 之一：

| No. | Naive Torch | Flash-Compatible | Chunk-Aware | Group-Aware | Time (ms) |
|-----|-------------|------------------|-------------|-------------|-----------|
| 1 | ✓ | | | | 2128.900 |
| 2 | | ✓ | | | 13.102 |
| 3 | | ✓ | ✓ | | 6.286 |
| 4 | | ✓ | ✓ | ✓ | **1.460** |

**Total speedup: 1458×** over naive PyTorch baseline。

分解：
- Flash-compatible layout: 162.5× speedup（最大头！）
- Chunk-aware: 2.08× additional speedup
- Group-aware: 4.31× additional speedup

**Insight**：Flash-compatible layout 是 game-changer。Depth KV 在 HBM 中 contiguous 存储是所有优化的基础。Chunk-aware 和 group-aware 是在这个基础上的精细化 memory access 优化。

---

## 6. Attention Visualization (Figure 5) 的 Insight

Figure 5 展示了 MoDA 在 layers [0, 11, 23, 35] 的 attention heatmaps。Red dashed line 标记 Sequence KV | Depth KV 的边界。

**关键观察**：
1. **Substantial attention mass on depth-KV block**，尤其在 middle 和 late layers。模型 actively retrieves cross-layer depth information。
2. **Complementary pattern**：sharp diagonal heads（local sequence attention）仍然 allocate probability 到 depth slots；broader heads 更依赖 depth-KV。
3. **Reduced attention sink behavior**：typical transformer 有 attention sinks（少数 fixed positions 吸收大量 probability mass，参见 [StreamingLLM](https://arxiv.org/abs/2309.17453)）。MoDA 把 probability 重新 distribute 到 informative sequence 和 depth slots。

第三点非常 intriguing——attention sink 一直被认为是 long-context 的 "necessary evil"，MoDA 提供了一个 alternative：当模型有 depth channel 可以 "retreat to"，就不需要把 probability 浪费在 sink positions 上。

这暗示 attention sink 可能不只是 numerical artifact，而是一种 "information compression" 的副作用——当模型无法从 sequence 找到 useful information，就 fallback 到 sinks。MoDA 给了模型另一个选项。

---

## 7. 与相关工作的联系与联想

### 7.1 DenseFormer
DenseFormer (Pagliardini et al., NeurIPS 2024) 是非常相关的工作，用 depth-weighted averaging 来增强 information flow。
- 参考: [https://arxiv.org/abs/2402.19410](https://arxiv.org/abs/2402.19410)
- 区别：DenseFormer 用 fixed-weight averaging，MoDA 用 data-dependent attention。

### 7.2 Hyper-Connections
Hyper-Connections (Zhu et al., ICLR 2025) 是 DeepSeek 的工作，提供了更灵活的 connection 机制，learnable 的 residual weighting。
- 参考: [https://arxiv.org/abs/2504.01635](https://arxiv.org/abs/2504.01635)

### 7.3 MHC (Manifold-constrained hyper-connections)
DeepSeek 的后续工作，进一步约束 hyper-connections 在 manifold 上。
- 参考: [https://arxiv.org/abs/2512.24880](https://arxiv.org/abs/2512.24880)

### 7.4 Highway Networks
最早尝试解决深度网络信息流的 work，gated skip connection。
- 参考: [https://arxiv.org/abs/1505.00387](https://arxiv.org/abs/1505.00387)

### 7.5 Virtual Width Networks
Recent work on width scaling alternatives。

### 7.6 GQA
Group Query Attention，MoDA 的高效实现依赖 GQA structure。
- 参考: [https://arxiv.org/abs/2305.13245](https://arxiv.org/abs/2305.13245)

### 7.7 Attention Sinks
StreamingLLM 的工作，揭示 attention sink 现象。
- 参考: [https://arxiv.org/abs/2309.17453](https://arxiv.org/abs/2309.17453)

### 7.8 OLMo2
MoDA 的 strong baseline。
- 参考: [https://arxiv.org/abs/2501.00656](https://arxiv.org/abs/2501.00656)

---

## 8. Discussion 部分的前瞻

### 8.1 Industrial-scale CUDA Engineering
作者承认 current kernel 虽然已经 97.3% of FA2 efficiency，但 industrial-scale training 还需要：
- Improved memory scheduling
- Deeper computation pipelining
- Tighter overlap between fused attention kernels and distributed communication

### 8.2 Bounded Depth-KV Slot Caching
这是 paper 提出的最重要的 future direction：

当 depth 非常深时，cache 所有 depth-KV states 引入 substantial memory/bandwidth overhead。Cost 线性增长，会成为 long-context training 和 serving 的 dominant bottleneck。

**Proposal**：fixed-size Depth KV slot buffer，size $S \ll L$。两种 policy：
1. **Dynamic selection**：用 utility score 选 top-S entries
2. **Sliding window**：keep most recent depth-KV entries，evict older ones
3. **Hybrid**：一部分 recency，一部分 high-score global memories

这非常像 **Memory Networks** 和 **Differentiable Neural Computers** 的思想！让模型有 bounded "external memory"，用 retrieval policy 决定 what to keep。

这个方向如果做深，会和 **MoE routing**、**Retrieval-augmented generation**、** episodic memory in RL** 等领域产生 cross-pollination。

---

## 9. Building Intuition：MoDA 在概念上的位置

让我尝试把 MoDA 放在一个更大的 conceptual map 上：

```
Information Flow Mechanisms in Deep Networks
├── Fixed-pattern (non-data-dependent)
│   ├── ResNet (identity + add)
│   ├── DenseNet (concat all)
│   └── Highway (gated)
├── Linear-learnable (data-independent weights)
│   ├── DenseFormer (weighted avg)
│   └── Hyper-connections (learnable residual)
└── Attention-based (data-dependent)
    ├── Depth Attention (this paper, intermediate)
    └── MoDA (unified sequence + depth softmax)
```

MoDA 的核心 conceptual contribution：**把 attention 从 sequence 维度推广到 sequence × depth 二维空间，并用 unified softmax 统一归一化**。

这背后的 principle 是：**所有 information aggregation 都应该是 data-dependent 的**。ResNet 假设所有 layers 等价 contribute，DenseNet 假设所有 layers 都需要 preserve，MoDA 让 model 自己决定 what to retrieve from depth history。

这和 attention 本身在 sequence 上的成功是一脉相承的——attention 之所以超越 RNN/CNN，正是因为它 data-dependent；同理，depth attention 应该超越 residual/dense connection。

---

## 10. 与你的教学和 nanoGPT 工作的潜在联系

你之前在 "Let's build GPT" 系列里讲 attention 的时候，强调 attention 的本质是 "weighted aggregation of information"。如果把这个 insight 推广到 depth：

- **Standard attention**: "我作为 token t，应该 attend to sequence 中哪些其他 tokens？"
- **MoDA**: "我作为 token t 在 layer l，应该 attend to (a) sequence 中哪些 tokens，以及 (b) 我自己在 preceding layers 的 representations？"

这是一个非常自然的 generalization，且和你强调的 "attention is a mixture" 的思想完全 consistent。

如果要在 nanoGPT 里实验 MoDA：
1. Start simple：在 6-layer nanoGPT 上加 depth KV（reuse sequence KV，不加额外 projection）
2. Visualize depth attention：看模型在 early layers 怎么用 depth channel
3. Try post-norm：可能 small scale 看不出差异，但是 conceptual 验证

---

## 11. 可能的 Limitations 和 Open Questions

1. **Depth KV memory cost**：虽然有 chunk-aware 和 group-aware 优化，但 depth-KV cache 仍然是 $O(TLD/G)$。在 1M context + 100 layers 的 frontier 场景下，这会是 bottleneck。Bounded slot caching 是必经之路。

2. **Cross-position depth interaction**：当前 MoDA 的 depth attention 是 per-position 的（token t 只看自己的 depth history）。是否应该允许 cross-position depth retrieval？比如 token t 看 token t' 在 layer i 的 representation？这会更 expressive 但复杂度会涨。

3. **MoDA + KV cache eviction**：在 inference 时如何与 PagedAttention、vAttention 等 KV cache 管理机制集成？

4. **MoDA + MoE**：MoE 在 width 维度做 sparse routing，MoDA 在 depth 维度做 dense retrieval。两者是否可以 combine？

5. **MoDA + Test-time compute**：depth retrieval 是否可以在 inference 时增加 layers（如 DeepSeek-R1 的 test-time reasoning）？

6. **MoDA + Long-context extrapolation**：depth KV 是否 help models extrapolate to longer context？因为 depth 提供了 position-invariant 的 information channel。

---

## 12. 总结

MoDA 是一个 **conceptually clean + technically well-engineered** 的工作。它：

1. **Conceptual contribution**：把 attention 从 sequence 推广到 sequence × depth，用 unified softmax 统一
2. **Engineering contribution**：Flash-compatible + chunk-aware + group-aware layout，实现 97.3% of FA2 efficiency
3. **Empirical contribution**：1.5B scale 上 +2.11 downstream avg gain，只有 3.7% FLOPs overhead
4. **Future-facing**：和 post-norm 互补，和 bounded slot caching 结合可扩展到 industrial scale

这工作给我最大的启发是：**Transformer architecture 的创新空间还远未枯竭**。Residual connection 是 2015 年的 ResNet 带来的，到 2026 年我们还在用。但 residual 只是最朴素的 information flow mechanism。Attention-based depth retrieval 是一个更 expressive 的 alternative，且在 GPU 上可以高效实现。

如果让我预测 MoDA-style mechanism 何时进入 production LLM，我会说 1-2 年内。这种架构创新通常先在 research 出现，然后被 big lab 内部验证，最后被 open-source 推广。MoDA 的高效实现 + 与 GQA 的 co-design + 与 post-norm 的 compatibility，使得它 adoption 的技术门槛很低。

**References**:
- Paper PDF (推测 arXiv): search "Mixture-of-Depths Attention Zhu ByteDance" on arXiv
- FlashAttention-2: https://arxiv.org/abs/2307.08691
- DenseNet: https://arxiv.org/abs/1608.06993
- DenseFormer: https://arxiv.org/abs/2402.19410
- Hyper-Connections: https://arxiv.org/abs/2504.01635
- GQA: https://arxiv.org/abs/2305.13245
- Attention Sinks (StreamingLLM): https://arxiv.org/abs/2309.17453
- OLMo2: https://arxiv.org/abs/2501.00656
- ResNet: https://arxiv.org/abs/1512.03385
- Highway Networks: https://arxiv.org/abs/1505.00387
- DeepNet: https://arxiv.org/abs/2203.00555
- Original Transformer: https://arxiv.org/abs/1706.03762

如果你有兴趣，我可以进一步深挖某一个部分——比如 depth mask 的实现细节、online softmax 在 fused kernel 中的数学推导、或者 attention sink reduction 的机理分析。
