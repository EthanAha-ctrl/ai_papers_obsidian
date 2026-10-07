---
source_pdf: Mixture of Contexts for Long Video Generation.pdf
paper_sha256: 38e4f5577dfb86f6923046bde7ee89e0abd13387e7afd45fa460727a1da002a6
processed_at: '2026-08-05T18:52:10-07:00'
target_folder: DiffusionModel
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说 Mixture of Contexts

## 这篇 paper 到底在干嘛

想象你在写一部长篇小说，写到第 200 页的时候，作者突然要回忆第 3 页里某个配角的名字和长相。人怎么处理？你不会把前 200 页每一页都重新读一遍——那太蠢了。你会翻回第 3 页，只看那一段，然后回来继续写。

视频生成模型现在就面临同样的问题，但更糟：它要生成 1 分钟的视频（对应 ~180k 个 token），用的是 transformer 的 self-attention。Self-attention 的机制是 "每一个 token 都要和所有其他 token 打个招呼"。180k × 180k 的交互，计算量是平方级爆炸的。

之前人们怎么解决？基本两条路：
- **把前面所有内容压成一个小向量**：像 FramePack。优点是省事，缺点是 lossy——细节丢了就找不回来，长期一致性会崩。
- **固定让每个 token 只看邻居**：像滑动窗口。优点是省 compute，缺点是死板——模型不知道什么时候需要跳很远去找信息。

这篇 paper 的核心 insight：**把生成长视频当成一次 "internal retrieval" 过程**。每个 token 在 generate 自己之前，先问一句 "我现在最需要回忆哪几段历史"，然后只去 attend 那几段。

听起来简单，但实现起来有几个 tricky 的地方，下面拆开说。

---

## 核心机制：怎么决定 "去看哪几段历史"

### Chunk the stream

先把整个长视频按 natural boundaries 切成 chunks——按 frame、按 shot、按 caption 切。为什么按 content 而不是按固定长度切？因为视频是 3D 的（空间 × 时间 × modality），均匀切会把 "一段静态背景 + 一段快速运动" 塞进同一个 chunk，mean pooling 之后那个 chunk 的 representative 就糊了——chunk 内部自己都打架，没法做 retrieval。

### Mean pool 每个 chunk

对每个 chunk，把它里面所有 key tokens 做个简单平均，得到一个代表向量。这个向量就当 chunk 的 "thumbnail"。

你会问：平均不会丢信息吗？作者解释说——mean pooling 只用于 **decide 要不要 attend 这个 chunk**，不是用于 attention 本身。一旦决定 attend，**完整的 K, V 都进 attention 计算**。所以 mean pool 只是 "coarse filter"，不是 "final summary"。这跟检索系统里用 embedding 先粗筛、再用 fine model rerank 是一回事。

还有个更深的 reason：diffusion autoencoder 学出来的 representation 本来就是 semantic-meaningful 的，一个 chunk 内的 tokens 第一主成分解释 90%+ variance，mean 就是 first principal component 的 estimator，已经抓住了 chunk 的主要语义。

### Top-k routing

每个 query token 拿自己的向量去和所有 chunk 的代表向量做点积，选 top-k 个最相关的 chunks。k 很小，比如 5。然后只对这 5 个 chunks 内的 tokens 做 fine-grained attention。

这步从 O(L²) 降到 O(k × L)，**线性于序列长度**。

### Per-head routing

这个很容易被忽略但很重要：**每个 attention head 每层独立做 routing**。

为什么？因为不同 head 关注的东西不一样。有的 head 关注 identity，它需要去 recall 几十秒前人物的脸；有的 head 关注 texture，它只需要看最近的几帧。如果让所有 head 共享一个 routing pattern，等于强迫所有 head 想同一件事——这就回到 fixed sparsity 的 bottleneck 了。

每个 head 独立 route，所有 head 的 union 覆盖了相当大比例的 context。这是一种 ensemble 思路。

---

## 几个 "why this works" 的细节

### 为什么 mean pooling 虽然简单但能 work

这里有个我觉得很 elegant 的论证：

top-k 这个操作不可微，没有直接的 gradient 给 router。但 mean pool 出来的 descriptor 是从 K/V projection 里来的——一旦某个 chunk 被选中但其实不相关，attention 的 loss 会通过 attention output 反向流回 K/V projection，间接惩罚这个 chunk 的 representation。下次这个 chunk 的 key projection 会被调整，使得它不那么容易被选中。

这就是 "indirect adaptation"——router 自己没参数，但整个 query/key projection 网络都在适应 routing 决策。从优化角度看，这比 hard routing MoE 还干净，因为没有 router 参数需要 special 处理。

### 为什么要强制 attend 所有 text tokens

text tokens 占总量 <1%，但编码了 character identity、scene description、action 这些最 semantic 的东西。作者直接 hardcode：**每个 visual query 都强制 attend 所有 text tokens**，不进 top-k 竞争。

这借鉴了 StreamingLLM 的 "attention sink" 思想——保留一个 small、persistent 的 token set 让 attention 有个 well-conditioned anchor，避免在长 generation 中 prompt drift。

### 为什么要强制 attend 同一个 shot 内的所有 tokens

类似地，每个 token 强制 attend 它所属 shot 内的兄弟们。这保证了 short-range consistency（object trajectory、lighting continuity），让 sparse routing budget 全部留给 genuinely long-range 的 retrieval。

如果连 short-range 都要靠 routing 来选，模型很容易 collapse——因为 short-range 信息太基础，必须保底。

### 为什么要 causal routing（这个是 hidden gem）

sparse routing 引入一个隐患：**loop closure**。

假设 chunk 5 觉得 chunk 6 很相关，route 到它；同时 chunk 6 觉得 chunk 5 也很相关，route 回去。这就形成了一个 two-node cycle——两个 chunk 只在彼此之间转圈，和外部断了联系。

作者在 ablation 里看到这个现象非常严重：生成视频时会出现 "stalled motion" 或帧重复。Information 在 cycle 里出不来。

解决方案是在 routing 阶段加 causal mask：**每个 chunk 只能 attend 比它早的 chunks**。这一刀切下去，routing graph 变成 directed acyclic graph（DAG），结构性消除 cycle。

这个观察其实挺深刻的——它揭示了 sparse attention 不只是 "少算点东西"，而是 "在 token 之间引入了 directionality"。引入 directionality 之后就一定要管 ordering，否则会退化成 pathological structure。

### Context Drop-off / Drop-in

借鉴 MoE 的 dead expert 问题。Random drop 一些 selected chunks，random add 一些 unselected chunks。这样训练更 robust，gradient 流得更广，不会出现某些 chunk 永远被选中或永远被忽略。

---

## 效果到底怎么样

### Quantitative

在 LCT baseline 上替换 attention，fine-tune：
- Sparsity 85%（砍掉 85% 的 token pair 交互）
- FLOPs 降低 >7×
- End-to-end 2.2× speedup
- 所有 VBench metric 基本持平或微升

**最反直觉的结果**：Dynamic Degree 从 0.46 升到 0.56。也就是说，attention 更稀疏，动作反而更丰富。

为什么？因为 dense attention 让模型 over-rely on redundant history——旁边一堆类似帧，attention 一平均就陷入 "复制粘贴" mode。Sparse routing 强迫模型从 salient history 中 recall，然后 genuinely generate 新内容。这跟人类记忆有点像——你回忆得越精准，越能 produce 新东西；你被一堆无关信息淹没，反而只会重复。

### Zero-shot 实验（最 striking）

作者把 MoC kernel 直接插进 pretrained dense model，**冻结所有权重**，75% sparsity，居然还能保持合理的 subject identity、background、coarse motion。

这说明 mean-pool routing 本身就提供了 usable retrieval signal，甚至不需要训练——因为 mean 本来就是 first principal component estimator，自然 align with dominant visual pattern。

但 fine-tune 能进一步 refine，让 query/key projection 学会更好地利用这个 routing。

---

## 我觉得最 interesting 的几个点

1. **Reframe**：从 "compress history" 变成 "retrieve history"。Compression 是 lossy 的，retrieval 保留全量信息。这是 paradigm shift。

2. **Parameter-free router**：mean pool + top-k，没一个额外参数。整个 query/key projection 网络通过 indirect adaptation 学会 routing。这比 hard MoE 优雅太多。

3. **Causal routing + DAG**：sparse attention 引入 directionality，必须管 ordering，否则 pathological cycle。这个 insight 对所有 sparse attention method 都适用。

4. **Per-head distributed routing**：每个 head 独立做 retrieval，union 覆盖广。这等价于 "attention head 是 ensemble" 的思想。

5. **Outer loop hierarchical routing**：在 attention 之前先做 coarse filter，把 context 压到 trained length 内，绕过 RoPE extrapolation 问题。这是为 million-token scale 准备的。

---

## 一句话总结

**Long video generation 的瓶颈不在算力，在于 memory。Memory 问题的解法是 retrieval，不是 compression。Retrieval 可以 end-to-end 学，靠 mean-pool + top-k 这种 parameter-free router 就够用，因为 diffusion model 学出来的 representation 本来就 linearly separable。Sparse routing 引入方向性后必须加 causal constraint 防 cycle。结果：85% sparsity 下 FLOPs 降 7×、速度 2.2×、质量不降反升。**

---

# Mixture of Contexts (MoC): 把 long video generation 重新定义成 internal information retrieval

## 1. Big Picture: 这篇 paper 在解决什么问题

Long video generation 的根本 bottleneck 不是 modeling capacity，而是 **memory problem**。当 sequence length 推到 180k tokens（一个 1 分钟 480p 12fps 的视频在 VAE 16× 空间 / 4× 时间下采样后），dense self-attention 的 O(L²) 成本不仅 FLOPs 爆炸，更关键的是 optimization landscape 变得很难——所有 token 都要互相 attend，模型分不清什么是 "salient" 什么是 "redundant"，结果就是 identity drift、scene collapse。

之前的两条 main 路线：
- **Lossy compression**：把 history 压成 fixed-size latent（FramePack [60]、TTTVideo [7]、LaCT [63]）——但丢细节，hard ceiling 在保持 coherence 上。
- **Static sparsity**：固定 pattern（Radial Attention [26]、STA [62]、SpargeAttn [57-59]）——但不能 content-adaptive，无法在 "什么时候 recall 哪段 history" 这件事上 learning。

MoC 的核心 reframe 是：**long video generation 是 internal information retrieval**。每个 query token 应该 dynamically 路由到最相关的几个 context chunks，而不是无差别 dense attend。这等价于把 attention 变成 retrieval——只不过 retrieval 的 query/key embedding 是 end-to-end 训出来的，不是 hand-crafted。

Project page: https://primecai.github.io/moc/

---

## 2. Method: Mixture of Contexts 的数学构造

### 2.1 起点 - Vanilla Attention

DiT 中的 attention 模块：

$$
\operatorname{Attn}(Q, K, V) = \operatorname{Softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d}}\right) \cdot V \tag{1}
$$

其中 Q, K, V 是 query/key/value features，d 是 head dimension。关键观察是 attention 可以 query-wise 写成：

$$
\operatorname{Attn}(q_i, K, V) = \operatorname{Softmax}(q_i K^{\top}/\sqrt{d}) \cdot V
$$

每个 query $q_i$ 独立地从所有 keys 加权取 value。这告诉我们：sparse 一个 query 是完全独立的操作，可以 per-query 决定要 attend 哪些 keys。

### 2.2 Dynamic Routing via Top-k

把所有 tokens 分成 chunks 集合 Φ，每个 query $q_i$ 只 attend 它 top-k 最相关的 chunks：

$$
\operatorname{Attn}(q_i, K, V) = \operatorname{Softmax}\!\left(\frac{q_i K_{\Omega(q_i)}^{\top}}{\sqrt{d}}\right) \cdot V_{\Omega(q_i)} \tag{2}
$$

$\Omega(q_i)$ 是这个 query 要 attend 的所有 context 位置的 indices 集合。选 top-k chunks 的方式：

$$
\Omega(q_i) = \left[\operatorname*{argmax}_{\Omega^*} \sum_{\omega \in \Omega^*} \big(q_i^{\top} \phi(K_{\omega})\big)\right] \quad \text{where } \Omega^* \subseteq \Phi \text{ and } |\Omega^*| = k \tag{3}
$$

变量含义：
- $\Omega^*$：候选 chunk index 子集，size 必须等于 k
- $\phi(K_{\omega})$：chunk $\omega$ 的 descriptor transformation——paper 用 **mean pooling**：$\phi(K_{\omega}) = \text{mean\_pool}(K_{\omega})$
- $[\cdot]$：把所有 top-k chunks 的 indices concatenate 起来

**为什么 mean pooling 这么简单的东西 work？** 这里有几个深层的 intuition：

1. **DDAE 的发现 [43]**：denoising diffusion autoencoders 自然学到 semantically meaningful 且 linearly separable 的内部 representation。一个 chunk 的全局均值近似它的 first principal component——对 centered data 来说，arithmetic mean 就是 first-component estimator。作者说实际上 chunk 内 tokens 的 first principal component 解释 >90% 局部 variance。所以均值保留了 dominant semantics，丢掉 high-frequency noise。

2. **Video token 的冗余性**：相邻 patch、相邻帧往往表示同一物体或背景区域，patch embedding 后 token 在一个很窄的 subspace 里。

3. **Parameter-free 但 trainable**：top-k 不可微，但梯度通过 attention 上的 loss 反向流到 K/V projections（mean-pooled descriptor 的源头）。如果某个 chunk 被选中但实际上 irrelevant，attention 的 gradient 会惩罚它的 K/V，间接逼 query/key projection 学出更具 discriminative 的 similarity。这是 hard routing MoE 系统里 "indirect adaptation" 的典型机制。

### 2.3 Context Drop-off / Drop-in

借鉴 MoE 的 dead expert 问题：

**Drop-off**：对每个 query $q_i$，算完 $\Omega(q_i)$ 后采样 $p_{\text{drop}} \sim \text{Uniform}(0, p_{\max})$，mask 掉 $\lfloor p_{\text{drop}} \cdot k \rfloor$ 个随机 chunks。这强迫模型在 context 偶尔不可用时也能 coherent output，让 dependencies 学得有 redundancy，避免 routing error 导致 catastrophic failure。

**Drop-in**：随机 sample $m \sim \text{Poisson}(\lambda)$ 个 extraneous chunks 加进 selected pool。这激活 underutilized chunks，让 gradient 流到更广的 context segments，平衡 routing distribution。

这两个 trick 都不干扰 routing 本身的学习（因为 router 是 parameter-free 的），而是通过 attention 的 backprop 间接塑造 Q/K projection。

### 2.4 Per-Head Distributed Routing

这是个非常关键的设计。Routing 在 **每个 attention head 每层** 独立做，相当于 $L_{\text{layers}} \times H_{\text{heads}}$ 个独立 routers。

为什么？两个理由：
1. **Head specialization**：DiT 里不同 head 关注不同 feature subspace——low-level texture vs high-level semantic identity。它们需要访问不同的历史 segments。
2. **Bottleneck avoidance**：每个 head 都 sparse（只 k 个 chunks），但所有 head 跨 layer 的 union 覆盖了相当大比例的 context。这避免了单一 global router 的 information bottleneck。

直觉上：每个 head 是一个 sparse "viewpoint"，整个网络通过 diverse sparse viewpoints 重建 full context manifold。这和 ensemble 的精神类似。

### 2.5 Content-aligned Chunking（核心 insight）

普通 LLM long context 里 chunking 简单：1D sub-word tokens + 单一 RoPE，按 fixed-length 切窗（MoBA [28]）即可。

Video DiT 是 **heterogeneous 3D+modality lattice**：spatial patches × temporal frames × text tokens，每个有独立的 3D RoPE factor。两个邻居 index 可能在 space-time 上离得很远，或跨过 shot cut。一个 static background patch 可以连续几百帧，紧挨一个高 entropy 的 motion token。

如果用 uniform windows：
- Mean-pooled key 被 "internally inconsistent" 的 tokens 污染
- top-k selector 浪费 slot 在 inconsistent keys 上

MoC 的方案：沿 content-aware boundaries 切——**frames、shots、modality stripes**。每个 chunk 语义 homogeneous、在 3D positional manifold 上 geometrically local。这保留了 Eq. 3 中 mean-pooled key 的 discriminative power。

这个设计也兼容 LCT [14] 的 interleaved text-video 长序列处理方式。

### 2.6 Fixed Cross-Modal Selection as Attention Sink

每个 visual query token **强制 attend 到所有 text tokens**。借鉴 StreamingLLM 的 attention sink [44] 思想：
- Text tokens 占 <1% 的总量，但编码了最 semantic 的信号（global style、character identity、key action）
- 提供 low-entropy、semantically meaningful 的 attention anchor
- 保证每个 attention matrix 至少有一个 well-conditioned dense block
- 创造一个 global gradient highway
- 防 prompt-drift 和 rare attribute word fading
- 紧化 text/visual 共享 latent space，提升下游 text-guided video editing

### 2.7 Fixed Intra-Shot Selection as Local Window

每个 token **强制 attend 它所属 shot 内所有 tokens**。视频天然是 hierarchical 的：frames ⊂ shots ⊂ scenes。
- 捕获 object trajectories、lighting continuity
- 让 MoC 把 sparse attention budget 用在 genuinely long-range dependencies
- 防止 scene cut 处的 semantic discontinuity
- 保证每个 attention matrix 有一个 well-conditioned block
- 提供 contiguous、memory-efficient 的 fallback path

特别在 fine-tune pretrained video model 时有效：保留每个 shot 的 fidelity，模型逐渐学 broader context alignment。

### 2.8 Causal Routing - DAG 约束

sparse routing 引入方向性到 token interaction graph。**但没有 ordering 约束时，会退化成 pathologically closed loops**。

作者观察到一个 ablation：每个 chunk 只能选 1 个 peer 时，经常出现 chunk 5 → chunk 6 同时 chunk 6 → chunk 5，形成 isolated two-node cycle（见 Fig. 2）。这种 self-loop：
- 局部化信息
- 阻碍 gradient propagation
- bidirectional generation 时表现为 stalled motion 或重复帧

解决方案：在 routing 阶段加 causal mask，限制每个 chunk 只能 attend 比它早的位置的 keys：任何 edge $(i \to j)$ with $j \geq i$ 在 top-k selection 之前被 mask 掉。这把 routing graph 变成 **directed acyclic graph (DAG)**。

经验上：causal routing 不仅消除 isolated feedback pair，还促进更丰富的 long-range dependencies，让 training 更稳定、temporal dynamics 更平滑。

这个观察让我联想到 RNN / state-space model 中的 "vanishing gradient via recurrent loop" 问题，以及 diffusion forcing [3] 中通过 inject noise 处理 compounding error 的思路。MoC 用 DAG 结构性消除 cycle，是一个 elegant 的 graph-level 修复。

---

## 3. Implementation: Flash-Attention 整合

### 3.1 处理 variable-length chunks

content-aligned chunks 长度极度不均匀（text 几十 tokens，video shot 几千 tokens），比 MoBA [28]/NSA [55] 的均匀切分复杂得多。

实现细节：
1. **Tag boundaries**：用 `torch.bucketize` 和 prefix-sum tables (cu_seqlen, cu_shot, ...) 给 flattened token stream 标 frame/shot/caption 边界
2. **Pre-routing mask**：先插入 forced links（caption-visual、intra-shot self edges），保证 router 不会在 mandatory chunk 上浪费 budget
3. **On-the-fly pooling**：用 `torch.segment_reduce` 算每个 chunk 的 mean-pooled key，**不 materialize 整个 chunk**，保持 memory flat
4. **Head-major order**：用 `rearrange(..., 's x h d -> h s x d')` 让后续 gather 是 coalesced 的
5. **Single Flash-Attention var-len call**：把 heterogeneous (Q, K) pair 打包进一个 Flash-Attention [8,9] var-len 调用，保留 kernel fusion

整个 routing + attention 是 head-independent 的，可以 fully tensor parallel/shard across devices。

### 3.2 FLOPs 分析

每个 attention head，给定：
- $L$：sequence length（query tokens 数量）
- $C$：chunk 数量
- $k$：每个 query 保留的 top-k chunks
- $\bar{m}$：选中 chunks 的平均长度
- $d$：head dimension

三部分 FLOPs：
1. **Mean-pooling**：$Ld$ adds（可忽略）
2. **Routing**：每个 query-chunk pair 一个 inner product，cost $2LCd$ FLOPs（×2 因为 inner product 是 multiplication + addition per dimension）
3. **Fine-grain attention**：在 pruned set 上做 QK 和 PV product，每个 query 最多 $k\bar{m}$ keys，约 $4Lk\bar{m}d$ FLOPs

总和：

$$
\text{FLOPs}_{\text{MoC}} \approx Ld + 2LCd + 4Lk\bar{m}d \tag{4}
$$

对应 dense attention：

$$
\text{FLOPs}_{\text{dense}} = 4L^2 d \tag{5}
$$

比值：

$$
\frac{\text{FLOPs}_{\text{dense}}}{\text{FLOPs}_{\text{MoC}}} \approx \frac{2L}{Cd + 2k\bar{m}} \tag{6}
$$

**线性于 sequence length L 增长**。

具体数字：480p、12fps、1 minute video → 180k tokens。$\bar{m} \approx 1024, k = 5, C = 36, d = 128$：
- $\text{FLOPs}_{\text{MoC}} \approx 2.32 \times 10^{12}$
- $\text{FLOPs}_{\text{dense}} \approx 1.66 \times 10^{13}$
- **>7× reduction**

---

## 4. 实验

### 4.1 Setup

- Base model: LCT [14]（3B MMDiT，scene-level context window，最多 8 shots ≈ 8s each ≈ 22k tokens per shot）
- 替换 dense attention 为 MoC，fine-tune
- 测试：8-shot sequence, 480p, 12fps, ~180k tokens per 64s scene
- VBench metrics: Subject Consistency, Background Consistency, Motion Smoothness, Dynamic Degree, Aesthetic Quality, Image Quality

### 4.2 Quantitative（Table 1）

| Method | Subject Cons ↑ | BG Cons ↑ | Motion Smooth ↑ | Dynamic Degree ↑ | Aesthetic ↑ | Image Quality ↑ | Sparsity | FLOPs ↓ |
|--------|---------------|-----------|------------------|-------------------|--------------|-------------------|----------|---------|
| LCT [14] | 0.9378 | 0.9526 | 0.9859 | 0.4583 | 0.5436 | 0.5140 | 0% | 1.7×10¹³ |
| MoC (ours) | 0.9421 | 0.9535 | 0.9920 | 0.5625 | 0.5454 | 0.5003 | 85% | **2.3×10¹²** |

关键观察：
- **FLOPs >7× 降低，end-to-end 2.2× speedup**
- **Dynamic Degree 从 0.46 → 0.56**：这是反直觉但很重要的——更稀疏的 attention 反而产生更多 motion。直觉是 dense attention 让模型 over-rely on redundant history，陷入 "复制粘贴" mode；sparse routing 强迫它从 salient history recall，然后 generate 新内容。
- 其他 quality metrics 基本持平或微升

### 4.3 Single-shot 实验（Table 2）

即使短序列（6k tokens），83% sparsity 下 MoC 在所有 VBench 指标上 match 或超过 base model。但短序列下，index gathering + pooling 的 overhead 超过 computational savings——所以 MoC 真正发挥优势在长序列。

### 4.4 Wan-2.1-1.3B 泛化（Table 5）

在 Wan-2.1-1.3B 上只 fine-tune 1 天（2000 iter），MoC 也超过 dense attention baseline，证明算法的 backbone generalization。

### 4.5 Zero-shot 实验（Appendix D）

这是最 striking 的：直接把 MoC kernel 插进 pretrained dense model，**冻结所有权重**，>75% sparsity，依然能保持合理的 subject identity、background layout、coarse motion。

这证明 mean-pooled descriptor 即使在从未见过 sparse mask 的 weights 上也提供 usable retrieval signal——因为 mean 就是 first principal component estimator，自然 align with dominant foreground/background pattern。

### 4.6 Ablation

**Chunk size + k（Table 3，single-shot）**：
- 太小 chunk (64, 128)：prune 太激进，motion 受损
- 太大 chunk (1024)：sparsity 不够（35%）
- k=3 + chunk=256 是甜点（83% sparsity, decent motion）
- **Progressive sparsification**：从大 chunk + 大 k 渐进到小 chunk + 小 k 是最优策略

**Forced links（Table 4，multi-shot）**：
- 没有 intra-shot forced link → 训练极度不稳定（与 learning rate、batch size 强相关）
- Cross-modal links + Context Drop In & Out 都改善性能
- 这与 MoBA [28] 中保留某些 dense layer for gradient flow 一致

---

## 5. 关键 Design Choices 的 Intuition

### 5.1 为什么 learned routing 比 fixed sparse pattern 强

固定 sparse pattern（Radial Attention [26]、STA [62]）基于 "spatiotemporal energy decay" 之类的先验，但 video 内容差异极大：
- Dialogue scene：人物 identity 跨 shot 必须保持，但 camera angle 变化大
- Action scene：motion 连续性重要，但 identity 可能 obscure
- Montage：cuts 跨越不同 location，相邻 shot 完全无关

Learned routing 让模型自己学："此刻我应该 recall 哪段 history"。这是 content-adaptive，而非 content-agnostic。

### 5.2 为什么 mean pooling 够用

最容易质疑的点：mean pooling 把整个 chunk 压成一个 vector，岂不是丢了所有细节？

但注意——mean pooled key 只用于 **routing decision**（选 top-k chunks），不是用于 final attention。一旦 chunks 被选中，**完整的 K, V 都进入 attention 计算**。所以 mean pool 只是个 "thumbnail"，用于 coarse retrieval，fine-grained 信息由后续 attention 保留。

这跟 retrieval system 里用 coarse embedding 做 first-stage filtering，再用 fine-grained model 做 second-stage re-ranking 是同一个 idea。

### 5.3 为什么 causal routing 必要

sparse attention 一般默认 bidirectional。但 sparse routing 引入方向性后，没有 ordering 约束会产生 cycle。Cycle 在 graph 上是 "信息 trap"——gradient 无法 escape，roll-out 时信息无法 escape，导致局部 stuck state。

Causal mask 把 routing graph 变成 DAG。这等价于 RNN 中的 "feedforward in time" 约束，但保留 sparse routing 的灵活性。

### 5.4 为什么 per-head routing 而非 global routing

如果 global router 在所有 head 共享，相当于整个网络共享一个 sparse pattern。但 head 之间是 specialized 的——texture head 需要的 history 和 identity head 需要的 history 不同。

Per-head routing 让每个 head 独立做 retrieval decision，union 覆盖整个 context。这是把 ensemble 思想用到 attention routing 上。

---

## 6. 与相关工作的对比

### 6.1 LCT [14]

最直接的 baseline。LCT 把 dense attention 从 per-shot 扩到 scene-level（8 shots, 22k tokens each），用 interleaved 3D RoPE。FLOPs $\sim (8 L_{\text{shot}})^2$。MoC 在 LCT 上 fine-tune，把 dense attention 换成 sparse routing，保留 LCT 的 context 扩展能力但消除 quadratic cost。

### 6.2 FramePack [60]

把任意多 frames 压成 fixed vector，做 next-frame prediction。lossy compression，hard ceiling on coherence。MoC 保留全部 history tokens，只是 dynamically route，无信息损失。

### 6.3 MoBA [28]

LLM long context 的 mixture of block attention，fixed-length chunks。MoC 借鉴这个 idea 但适配 video 的 heterogeneous 3D+modality lattice，用 content-aligned chunking。

### 6.4 NSA [55]

Native sparse attention，hardware-aligned and trainable。MoC 与 NSA 精神类似——trainable sparse attention——但 NSA 是 LLM，MoC 是 video DiT。

### 6.5 VSA [61]

Trainable sparse attention for video diffusion。VSA 用 coarse-to-fine sparse kernel 替换 full attention。MoC 在 VSA 之上加了 content-aligned chunking、causal routing、forced anchors。

### 6.6 Radial Attention [26]

Static O(n log n) mask 基于 spatiotemporal energy decay。固定先验，content-agnostic。MoC 是 learned + content-adaptive。

### 6.7 SpargeAttn / SageAttention [57-59]

Training-free 的 sparse attention，通过 outlier smoothing + int4 quantization 加速 inference。MoC 是 trainable，专注于 long-context memory。

### 6.8 Diffusion Forcing [3]

通过 inject noise 到 historical context 训练模型 denoise，增加 robustness to compounding error。MoC 不用 noise injection，而是用 sparse routing + causal DAG。

### 6.9 Attention Sinks [44]

StreamingLLM 发现保留 initial tokens 作为 "sink" 能稳定 long-context generation。MoC 借鉴这个，把 text tokens 作为 attention sink——每个 query 都强制 attend 全部 text tokens。

---

## 7. Outer Loop Context Routing（Appendix I）

为了 scale 到 extreme long sequence（百万 tokens），paper 提出一个 outer-inner 层级 routing：

**Outer loop**：在 attention 计算前，先做 coarse selection。把 token stream 分成 high-level chunks $\Psi = \{\Psi_1, ..., \Psi_P\}$，每个 $\Psi_j$ 包含多个 lower-level chunks。对当前 generation block，算一个 aggregated representation $x_g = \text{mean\_pool}(X_g)$，对每个 $\Psi_j$ 算 descriptor $\phi(\Psi_j) = \text{mean\_pool}(X[\Psi_j])$，similarity score $s_j = \langle x_g, \phi(\Psi_j) \rangle$，取 top-M 个 high-level chunks：

$$
\Omega_g = \arg\max_{\Omega^* \subseteq \Psi, |\Omega^*| = M} \sum_{j \in \Omega^*} s_j
$$

这把 context 压缩到模型 trained capacity 内，**绕过 RoPE extrapolation 问题**——因为 attention 操作在 curated 短子序列上，positional encoding 不必 extend 到 trained length 之外。

经验上 outer loop 能把 shot 数量再扩 2-3 倍，under autoregressive sampling。

---

## 8. Limitations & Future Directions

作者承认：
1. 只测了 LCT setup，更长序列的 saving 还没探索
2. Runtime 依赖 general-purpose var-len attention + framework-level gathers，理论上 7× FLOPs 节省还有 headroom
3. 未来：block-sparse、chunk-aware CUDA/Triton kernel、fused routing+attention、persistent execution、KV quantization

我的额外联想：
- **Hierarchical routing** 可以更深——3 层或更多
- **Routing sparsification schedule** 目前是 manual，可以 learnable
- **Routing interpretability**：能否可视化 "模型在 generate 哪帧时 recall 哪段历史"？这是研究 long-context memory 机制的 probe
- **Multi-modal extension**：audio + video chunking，cross-modal routing
- **World model connection**：MoC 的 retrieval 机制和 WorldMem [46] 的 external memory bank 思路可以结合——learned routing 替代 FoV overlap 选择

---

## 9. 我的 takeaway

这篇 paper 的 elegance 在于：**它把 long video generation 从 "how to compress history" 重新 frame 成 "how to retrieve history"**。

之前的路线都在 compress——keyframes、latent states、fixed vector。Compression 本质是 lossy，所以有 hard ceiling。

MoC 的洞察是：**video 的大部分内容是冗余的，模型不需要 always-on dense attention，需要的是 "right context at right time"**。这就是 retrieval problem。

而 retrieval 的 query/key embedding 可以 end-to-end 学。Mean pooling 看起来 naive，但因为 diffusion autoencoder 学出的 representation 已经 semantic-meaningful 且 linearly separable，mean 就是 first principal component estimator，已经足够做 coarse retrieval。

Causal routing + DAG 是另一处 elegant 设计——它把 sparse routing 引入的 "directionality" 用 graph constraint 结构性 stabilize，避免 cycle。

实验上 85% sparsity 下性能不降反升（特别是 Dynamic Degree）是反直觉但合理解释的——dense attention 让模型 over-rely on redundant history，sparse routing 强迫它做 genuine recall + generate。

潜在 impact：这套 idea 可能 transfer 到 LLM long context、multimodal agent memory、world model simulation——任何需要 "selective recall over long horizon" 的场景。

---

## References

- Paper project page: https://primecai.github.io/moc/
- LCT (Long Context Tuning): https://arxiv.org/abs/2412.07853 (ICCV 2025)
- FramePack: https://arxiv.org/abs/2504.12626
- FlashAttention-2: https://arxiv.org/abs/2307.08691
- StreamingLLM (Attention Sinks): https://arxiv.org/abs/2309.17453
- MoBA: https://arxiv.org/abs/2502.13189
- NSA (Native Sparse Attention): https://arxiv.org/abs/2502.11089
- VSA (Trainable Sparse Attention): https://arxiv.org/abs/2502.07645
- Radial Attention: https://arxiv.org/abs/2504.19287
- SpargeAttn: https://arxiv.org/abs/2502.16037
- SageAttention: https://arxiv.org/abs/2502.14859
- STA (Sliding Tile Attention): https://arxiv.org/abs/2502.07548
- SparseVideoGen: https://arxiv.org/abs/2502.14617
- VMem: https://arxiv.org/abs/2507.13993
- WorldMem: https://arxiv.org/abs/2507.13993
- Diffusion Forcing: https://arxiv.org/abs/2404.01155
- Diffusion Autoencoders (DDAE): https://arxiv.org/abs/2311.11347
- DiT (Peebles & Xie): https://arxiv.org/abs/2212.09748
- RoPE (RoFormer): https://arxiv.org/abs/2104.09864
- Wan 2.1: https://github.com/Wan-Video/Wan2.1
- CausVid: https://arxiv.org/abs/2506.13338
- MAGI-1: https://arxiv.org/abs/2506.21898
- SkyReels-V2: https://arxiv.org/abs/2503.11058
- VBench: https://arxiv.org/abs/2311.17918
