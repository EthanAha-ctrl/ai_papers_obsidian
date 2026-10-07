---
source_pdf: DeltaKV Residual-Based KV Cache Compression via Long-Range Similarity.pdf
paper_sha256: 4c2c444b7fac162e3257a79f69654aba9143b8aa5f30b071d3b41912dda3451c
processed_at: '2026-08-03T19:40:27-07:00'
target_folder: LLM-engine/vllm
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej，咱们用大白话来拆解一下这篇 paper，重点 build 一下 intuition。

## 1. 解决什么痛点？

跑 long-context LLM (比如 128k tokens)，最大的瓶颈是 KV cache 吃显存太猛。Llama-3.1-8B 在 128k context、batch size 为 8 的情况下，光 KV cache 就要 130GB，单卡根本塞不下。

以前大家怎么解决呢？主要有两派：
第一派是“暴力删 token”（比如 SnapKV）。算一算 attention score，把看着不重要的 token 直接扔掉。问题在于，多轮对话或者复杂推理时，现在看着没用的 token，后面可能突然变关键，删了就彻底崩盘。
第二派是“假装删了其实没删”（比如 OmniKV、Quest）。保留全量 KV cache，每次计算时只挑 top-k 个 token 算 attention。精度保住了，但是显存一点没省，如果想省显存只能 offload 到 CPU，这就引入了 PCIe 延迟。

DeltaKV 想做的，就是**既不删 token，又要大幅省显存，还得在 GPU 上跑得飞快**。

## 2. 核心直觉：KV Cache 里的“废话”太多

作者先做了两个非常 insightful 的实验观察：

**观察一：Token 的相似性是“全局”的，不是“局部”的**
以前的方法（比如 CacheGen）总觉得相邻的 token 差不多，可以放一起压。但作者发现，一个 token 的“双胞胎”往往在好几万 token 之前。超过 60% 的相似 token 距离都在 16 个位置以上。所以靠“局部相似性”做压缩是错位的，必须要做全局检索。

**观察二：KV Cache 有很强的“共享骨架”**
对 KV cache 做 SVD 分解，发现能量极度集中在少数几个高范数的方向上。这说明大部分 token 其实都在共享一套基础的语义和结构“模板”。如果你把那些相似的 token 找出来减掉，剩下的 residual（残差）能量非常小，分布全挤在 0 附近。

**Intuition**: 这就像 JPEG 压缩图片，或者 Git 存代码。没必要存完整的像素或完整的文件，存一个“基准底版”加上一个“微小差异的 patch”就够了。KV cache 也是这个逻辑。

## 3. DeltaKV 怎么做压缩？

这个方法可以概括为“**找历史相似 token，减去共享成分，只压残差**”。

具体步骤：
1. **建参考池**: 每 $s=10$ 个 token 抽一个，放进一个 reference pool。这样池子大小只有原来的 1/10。注意，这里是在 pre-RoPE（应用位置编码之前）做的，保证找相似性时不受位置干扰，纯粹看内容语义。
2. **找最近邻**: 当前 token 来了，去池子里用 L2 距离找 top-$k=4$ 个最像的历史 token。
3. **求残差并压缩**: 把这 4 个历史 token 求个平均 $\overline{KV}_R$。然后把当前 token 和这个平均值都送进一个 compressor $f_c$，在 latent space 里相减：
   $$z_\Delta = f_c(KV) - f_c(\overline{KV}_R)$$
   这里 $z_\Delta$ 就是压缩后的残差向量。维度从 $2d_k$ 压到了 $d_c$（通常只占原来的 25%）。
4. **恢复**: 读取的时候，用一个 decompressor $f_d$ 把 $z_\Delta$ 解压回高维，再加上之前的平均值 $\overline{KV}_R$ 就完事了：
   $$\widehat{KV}_i = f_d(z_\Delta) + \overline{KV}_R$$

**一个非常 clever 的工程细节**：Compressor 很重（用 SwiGLU 神经网络），Decompressor 很轻（直接一个无 bias 的线性矩阵 $W_d$）。为什么？因为压缩一个 token 只做一次，但解压在 decode 阶段每生成一个 token 都要反复做，解压必须极快。

训练这套参数也很快，只拿 160M tokens 的 Fineweb-Edu 数据，在单张 RTX PRO 6000 上跑 8 个小时就搞定了。Loss 是 MSE（保证数值还原）加上 NTP（保证语言模型预测能力不掉）。

## 4. 系统落地：Sparse-vLLM 怎么榨干 GPU？

算法 paper 经常写得好，但一部署就废，因为 vLLM 那套 PagedAttention 假设 KV cache 是整整齐齐的 page，你搞这种“一半全量、一半压缩残差”的非规则内存，vLLM 根本玩不转。

所以作者写了 Sparse-vLLM，核心是**解耦内存管理和计算**。

1. **双内存池**: 开辟一个 Full Pool（存那些 sink token 或最近的高精度 token）和一个 Latent Pool（存压缩后的 $z_\Delta$）。
2. **Slot 虚拟化**: 算 attention 之前，Sparse Controller 会先算好哪些 token 要被选中，然后把 Latent Pool 里的残差拉出来，解压重构到一块临时显存里，最后给 attention kernel 一个“看似连续”的 virtual slot mapping。
3. **Kernel 融合**: 用 Triton 改写了 Flash-Decoding kernel，让它支持非连续内存的直接读取（indirect addressing），不用先把数据 copy 到一块连续空间。找相似 token 的 L2 距离计算、求平均、加残差这些操作全融合在一个 kernel launch 里，减少显存读写。

最后跑出来的效果：在 512k context 下，vLLM 跑 33 tokens/s，Sparse-vLLM 跑 DeltaKV 能到 67.7 tokens/s，直接 2x 提速。

## 5. 这套东西的深层启发

聊点更底层的技术联想。

**与 Activation Steering / ReFT 的同构性**
DeltaKV 这种“减去一个均值方向，只保留残差”的做法，跟 Representation Engineering (ReFT) 里的 activation steering 极其相似。ReFT 是在 hidden state 里找一些 steering directions，然后 project out 去控制模型行为。DeltaKV 实际上在 KV space 做了类似的事情：把 shared 的 anisotropy 方向剥离开来，只处理那个 low-magnitude 的 residual。这也解释了为什么残差那么好压缩——它已经被剥离了主要的流形方向，变成了近似高斯噪声。

**与 Attention Sink 现象的呼应**
StreamingLLM 发现开头的几个 token（attention sink）绝对不能删，删了 perplexity 就炸。DeltaKV 的发现间接给了一个解释：这些 sink token 很可能就是那些 dominant shared latent components 的物理载体。因为它们被所有 token 频繁 attend 到，所以它们身上积累了全局的共享信息。在 DeltaKV 里，这些 sink token 自然也会被频繁选作 reference，它们的 $\overline{KV}_R$ 也就捕获了全局的骨架。

**对 Linear Attention 的启示**
Mamba、GLA、RetNet 试图用固定大小的 state 压缩历史。DeltaKV 证明了在标准 Transformer 里，通过找 reference + 压缩 delta 的方式，也能达到类似“可压缩历史”的效果，而且可以无损恢复。这其实是在标准 attention 架构内打了一个“近似线性 attention”的补丁。

**VQ-VAE 与 Product Quantization 的对比**
PQCache 那种用 codebook 做 Product Quantization 的方法，在 GPU 上跑很别扭，因为查表操作对显存极不友好。DeltaKV 抛弃了 codebook，改用参数式的 MLP/Linear 做映射，虽然压缩率没有极极端的 PQ 高，但换来了极高的 GPU 友好度，这是很典型的 hardware-aware algorithm design。

## 6. 现存缺陷与未来想象

DeltaKV 目前还有一些没吃完的红利：
1. **Python 开销大**: 论文里自己承认，BS=16 时 step latency 91ms 里有 62ms 花在了 KV reconstruction 和 Python level 的 view/slot bookkeeping 上。如果以后把所有 bookkeeping 逻辑下沉到 Triton/CUDA kernel 里，latency 还能再降 1.5x-1.7x。
2. **未全链路量化**: 目前只对 residual $z_\Delta$ 做了 4-bit 量化，参考 token 和 filter layers 还是 BF16。如果全链路 4-bit 量化，显存能压到 7.2%。配合 PCIe offload，单卡跑 1M context 完全有希望。
3. **Universal Compressor**: 目前每换个 LLM 底座都要重训 8 小时 compressor。如果能做成即插即用的 universal compressor（比如用某种 hypernetwork 生成投影矩阵），工程价值会翻倍。

总结一句，DeltaKV 的核心贡献在于**把“语义相似性”显式地引入了 KV cache 的数学表达中**，不再盲目做低秩分解或粗暴量化，而是先剥离出共享语义骨架，再对真正 token-specific 的 delta 下手。这种 content-aware 的压缩，比 hardware-blind 的数值压缩走得更远。

参考链接：
- Sparse-vLLM repo: https://github.com/CURRENTF/Sparse-vLLM
- vLLM (PagedAttention): https://arxiv.org/abs/2309.06180
- StreamingLLM (Attention Sink): https://arxiv.org/abs/2309.17453
- ReFT (Representation Finetuning): https://arxiv.org/abs/2402.04092
- Mamba (Linear State Space): https://arxiv.org/abs/2312.00752
- PQCache (Product Quantization): https://arxiv.org/abs/2409.02636
- SnapKV: https://arxiv.org/abs/2404.14469
- OmniKV: https://openreview.net/forum?id=OmniKV

---

# DeltaKV: Residual-Based KV Cache Compression 深度讲解

非常有趣的一篇 paper, 让我从多个 angle 来帮你 build intuition。

## 1. 核心问题：Long-Context LLM 的 KV Cache Bottleneck

首先回顾 background. 对 Llama-3.1-8B-Instruct 而言, 128k context length + batch size 8 需要 **130+ GB** 的 KV cache storage, 远超单张 GPU 的容量。KV cache memory 随 sequence length **linearly** 增长, 而 attention computation 是 O(n²) 的 quadratic cost。

现有方法的三类问题, 也是作者想 break through 的：

| 问题 | 描述 | 代表方法 |
|------|------|---------|
| Local Similarity Bias | 假设 token 相似性局限于邻域 | CacheGen, Chelsea |
| GPU-Unfriendly Pipeline | 多 stage 压缩 pipeline 导致 irregular memory access | PQCache, Lexico |
| Poor Framework Integration | per-layer budget 异构难以部署 | SnapKV, PyramidKV, AdaKV |

## 2. 两个 Empirical Observations（这是 paper 的灵魂）

### Observation 1: Long-Range Inter-Token Similarity

作者在 Qwen2.5-7B-Instruct-1M 上分析发现: token 间的相似性 **not local**. 超过 **60%** 的最相似 token 距离 > 16 positions. 这意味着 CacheGen 等基于 local 的假设 fundamentally flawed。

Figure 2a 展示了 cosine similarity 分布: 很多 token pair 的 cosine similarity > 0.9, 且这些 pair 分布在 distant positions. Figure 2b 的 log-distance histogram 清楚地显示相似性是 **global** distributed 的。

### Observation 2: Highly Shared Latent Components

SVD 分析 (Figure 2c) 显示原始 KV cache 的 spectrum **steeply decays**, 意味着少数几个 high-norm 的 latent directions capture 了大部分能量。减去 reference 之后, residual 的 spectrum **flattens**, L2 norm 大幅缩小 (Figure 2d), 分布集中在零附近 (Figure 2f)。

**Intuition**: KV cache 本质上是由 (1) 共享的 high-norm structural components + (2) token-specific 的 low-magnitude residual 组成。这类似于 PCA decomposition 的思想——大部分 energy 在 principal components 里, residual 是 noise-like 但 attention-critical。

## 3. DeltaKV 方法详解

### 3.1 Notation

Standard Transformer 中:
- $W_q \in \mathbb{R}^{d_q \times d}$: query projection matrix, $d_q$ 是 total query dimension, $d$ 是 hidden size
- $W_k \in \mathbb{R}^{d_k \times d}$, $W_v \in \mathbb{R}^{d_v \times d}$: key/value projections
- 一般 $d_k = d_v$, 且 $d_k = N \cdot D$ (N 个 head, 每 head 维度 D)
- KV cache 形状: $[B, L, N, D]$ 或等价 $[B, L, d_k]$

### 3.2 Strided Reference Selection

不 search 全部 history (太贵), 维护一个 **stride set**:

$$\mathcal{T} = \{ kv_t \mid t \mod s = 0, t < i \}$$

变量解释:
- $kv_t \in \mathbb{R}^{2d_k}$: 第 $t$ 个 token 的 KV concat (key 和 value 拼起来, 维度是 $2 \cdot d_k = 2ND$)
- $s$: stride, paper 中默认 $s = 10$, 即每 10 个 token 选一个 reference
- $i$: 当前 token 的 position index
- $\mathcal{T}$: reference pool, 大小约 $L/s$

**重要**: 所有操作在 **pre-RoPE** 的 KV 上做, 这样保证 position-invariant. RoPE (Rotary Position Embedding) 会在 attention 计算时 re-apply。这是关键设计——避免了 position encoding 干扰 similarity computation。

对于当前 token $i$, 从 $\mathcal{T}$ 中 retrieve top-$k$ nearest neighbors via L2 distance:

$$\mathscr{R}_i = \arg\top_k_{kv_j \in \mathcal{T}} \left( -\|kv_i - kv_j\|_2^2 \right)$$

变量:
- $\mathscr{R}_i$: 当前 token $i$ 的 top-$k$ reference index 集合
- $\arg\top_k$: 选 top-$k$ 最大 (因为取负, 等价于 L2 距离最小)
- $k$: paper 默认 $k=4$, ablation 显示 $k=4$ 或 $k=8$ 都是 sweet spot

Reference representation 取 mean:

$$\overline{KV}_R = \frac{1}{k} \sum_{j \in \mathscr{R}_i} kv_j \in \mathbb{R}^{2d_k}$$

### 3.3 Compressor & Decompressor

**Compressor** $f_c: \mathbb{R}^{2d_k} \to \mathbb{R}^{d_c}$ 是 MLP:

$$f_c(x) = \text{GeLU}(xW_{c1} + b_{c1})W_{c2} + b_{c2}$$

变量:
- $W_{c1} \in \mathbb{R}^{2d_k \times d_h}$: 第一层 weight, $d_h$ 是 hidden width (默认 4096, Light 版本 3072)
- $W_{c2} \in \mathbb{R}^{d_h \times d_c}$: 第二层 weight, $d_c$ 是 compressed dimension (默认 $0.25 \times 2d_k$)
- $b_{c1}, b_{c2}$: bias

Residual latent code:

$$z_\Delta = f_c(KV) - f_c(\overline{KV}_R)$$

注意这里一个 subtle 的设计: **compressor 应用于 current KV 和 reference mean 之后**再做减法。这相当于在 latent space 中做 residual, 让 $z_\Delta$ 更容易 compress。

**Decompressor** $f_d: \mathbb{R}^{d_c} \to \mathbb{R}^{2d_k}$:

$$\widehat{KV}_\Delta = f_d(z_\Delta)$$

标准版本是 MLP: $f_d(z_\Delta) = \text{GeLU}(z_\Delta W_{d1} + b_{d1})W_{d2} + b_{d2}$

Light 版本 (用 † 标记) 是 **bias-free linear**:

$$f_d(x) = xW_d$$

这是 asymmetric design 的核心——compressor 重 (用 SwiGLU), decompressor 轻 (用 linear). 因为 compression 只做一次, reconstruction 在 decoding 时反复做。

SwiGLU compressor 公式:

$$f_c(x) = (\text{Swish}(xW_1) \otimes (xW_2))W_3$$

- $\text{Swish}(x) = x \cdot \sigma(x)$, $\sigma$ 是 sigmoid
- $\otimes$: element-wise product
- $W_1, W_2 \in \mathbb{R}^{2d_k \times d_h}$, $W_3 \in \mathbb{R}^{d_h \times d_c}$
- SwiGLU 是 GLU (Gated Linear Unit) 的一种变体, 在 LLaMA 中广泛使用

Final reconstruction:

$$\widehat{KV}_i = \widehat{KV}_\Delta + \overline{KV}_R$$

### 3.4 Hybrid Training Objective

$$\mathcal{L} = \sum \|KV - \widehat{KV}\|^2 + \mathcal{L}_{\text{ntp}}(\theta, \phi)$$

- $\theta$: frozen LLM parameters (不训练)
- $\phi$: learnable DeltaKV modules (compressor + decompressor)
- 第一项: MSE reconstruction loss
- 第二项: Next Token Prediction loss (CrossEntropy)

**Ablation insight (Figure 6)**: MSE-only training 会 reduce NTP loss, **但** NTP-only training 不 reduce MSE. 说明数值 reconstruction 对 LM 有帮助, exact KV reconstruction 不是 strictly required。这跟 model 编辑 / activation steering 的某些发现类似——模型对 KV 的 numerical noise 有 robustness。

训练 setup: 160M tokens from Fineweb-Edu, sequence length 8192, **8 GPU hours** on single RTX PRO 6000. 训练短但 generalize 到 100k+ context, 因为 compression 在 pre-RoPE 上做, 学到的 mapping 是 position-invariant 的。

## 4. Sparse-vLLM 系统设计

这是 paper 的另一半, 把 compression gains 真正 translate 到 throughput. 现有 vLLM 和 SGLang 紧密 coupling memory management 和 model execution, 难以集成 sparsity。

### 4.1 CacheManager 三种 backend

| Backend | 用途 | 设计 |
|---------|------|------|
| Per-Layer Independent | Physical eviction (SnapKV, PyramidKV) | L 个独立 page table |
| Global Shared | Logical masking (OmniKV, Full Attention) | 单一共享 req_to_token_slots |
| Heterogeneous DeltaKV | Hybrid compression | Dual pool: Full Pool + Latent Pool |

DeltaKV 的 heterogeneous storage 是 highlight:
- **Full Pool**: 高精度 tokens (sink/recent)
- **Latent Pool**: 压缩的 residual vectors
- **Copy-on-Write slot sharing**: observation layer 选出 top-k 后, 后续 sparse layers 复用 reconstruction 出来的临时 slots, 避免重复 decompression

### 4.2 Sparse Controller Workflow

**Pre-Forward (View Construction)**:
1. Index Resolution: 确定哪些 token 需要 decompress
2. Batch Reconstruction: 从 Latent Pool fetch 压缩 vectors 和 reference
3. Slot Virtualization: 把 reconstructed KV 写入临时 buffer, 构造 virtual slot_mapping 给 attention kernel 一个 contiguous 视图

**Post-Forward (Lifecycle Management)**:
当 Recent Buffer 溢出, 触发 fused kernel:
1. 计算 overflow token 对 reference 的 residual
2. 通过 encoder 压缩
3. 写入 Latent Pool, 立即释放 Full Pool slot

### 4.3 Kernel Optimizations

- **Indirect addressing via slot mapping**: 修改 Flash-Decoding 接受 `req_to_token_slots` 数组, 直接从 non-contiguous memory 读, 避免 copy 或 block-table lookup
- **Batch L2 distance kernel**: 快速 reference search
- **Fused Reconstruction kernel**: gather reference + mean + residual add 一个 kernel launch
- 用 LightLLM 的 token-level Triton attention operator 处理 non-contiguous memory

### 4.4 Latency Profiling (Figure 7)

BS=16, 128k context: 总 latency 91.0 ms, 其中:
- KV reconstruction: 37.3 ms
- View/slot bookkeeping: 24.7 ms
- Model compute: 剩余

Bottleneck 在 Python-level control overhead 和 fragmented memory traffic. 作者估计 deeper operator fusion 可以降到 ~55-60 ms (~1.5-1.7x speedup)。这是 future work。

## 5. 实验数据深度解读

### 5.1 LongBench 主表 (Table 1)

Llama-3.1-8B 上 30% budget:
- Full Attention baseline: 50.0 avg
- SnapKV: 49.8 (KR=30%, CR=30%)
- OmniKV: 50.2 (KR=100%, CR=30%)  ← 不省 memory
- OmniKV + DeltaKV: 50.2 (KR=45%, CR=30%)  ← 同等精度但省 55% memory
- OmniKV + DeltaKV + 4-bit: 50.3 (KR=29%, CR=30%)  ← 进一步压到 29%

**关键 insight**: DeltaKV 把 dynamic sparsity (OmniKV) 的 100% KR 降到 45%, 精度几乎不丢。再叠加 quantization 到 29%, 接近 SnapKV 的 memory footprint 但保持 dynamic 的精度优势。

### 5.2 SCBench (Table 2) — Multi-turn 关键测试

Retrieval KV (R.KV) 是 multi-turn 的 killer test:
- Llama-3.1-8B Full: 79.0
- SnapKV: 0.4 ← **catastrophic**, 静态 eviction 在 multi-turn 上彻底崩盘
- OmniKV: 72.2
- OmniKV + DeltaKV: 58.0  ← 有些 drop, 因为 R.KV 含 SSID-like 字符串, distribution mismatch
- +4-bit: 60.4  ← quantization 反而稍好, 可能是 noise regularization

SnapKV 在 R.KV 上几乎归零 (0.4 vs 79.0 full), 这正印证了 paper 的论点: static eviction 在 multi-turn 中丢弃的 token 后期变 critical. DeltaKV 用 compression 而非 eviction, 保留了所有信息 (虽然 compressed)。

### 5.3 AIME (Table 3) — Reasoning

DeepSeek-R1-Distill-Qwen-7B:
- Full: 50.0
- SnapKV: 33.3 (掉 16.7 分, 巨大)
- OmniKV: 46.7
- DeltaKV: 43.3 (掉 6.7 分, 可接受)

Reasoning tasks 对 KV 信息更敏感, 任何 compression 都会有 loss, 但 DeltaKV 的 loss 明显小于 static eviction。

### 5.4 Throughput (Table 4)

512k context:
- vLLM Full Attention: 33.1 tokens/s (BS=2)
- Sparse-vLLM Full Attention: 32.1 (BS=2, ~5% overhead)
- Sparse-vLLM DeltaKV†: 67.7 (BS=4, **2x improvement**)

900k context (单卡跑不起 Full Attention):
- vLLM: 18.6 (BS=1)
- Sparse-vLLM DeltaKV†: 38.9 (BS=2, **2.1x**)

**有意思**: SnapKV 的 raw throughput 更高 (128k 时 338.8 vs DeltaKV 187.0), 因为 SnapKV 简单粗暴地 evict token, compute ratio 极低。但 SnapKV 精度崩盘。DeltaKV 是 accuracy-efficiency trade-off 的更优 Pareto frontier。

### 5.5 Ablation (Table 5)

Llama-3.1-8B 上:
- DeltaKV (full): 50.2 avg
- w/o $f_c$ and $f_d$ (无 compressor/解压器): 46.7 (-3.5)  ← residual 不经 compression 直接用
- w/o Reference Tokens $\mathcal{T}$: 45.9 (-4.3)  ← 不做 reference subtraction, 直接 compress raw KV

两个 component 都 critical. 去掉 reference 的版本相当于低秩 compression (类似 Palu), 表现差很多。这证明 **long-range similarity 的 reference subtraction 是核心**, 单纯的低秩 projection 不够。

### 5.6 超参 sensitivity (Figure 5)

- $d_c$ (compressed dim): 越大 loss 越低, 但有 OOM 风险. 默认 $0.25 \times 2d_k$
- $k$ (reference 数): 4 或 8 是 sweet spot, 太少 noise 多, 太多 similarity 平均掉
- $d_h$ (compressor hidden): 越大 reconstruction 越好, 但 inference 慢
- $s$ (stride): 越小精度越好, 但 retrieval 开销线性增长. 默认 $s=10$ (reference ratio ~10%)

## 6. 相关联想与延伸

### 6.1 与 LoRA / Low-Rank Adapter 的类比

DeltaKV 的 residual 在 latent space 表示, 类似 LoRA 的 idea: 大部分 information 在 base, 只学一个 low-rank delta。Palu (Chang et al., 2025) 直接做低秩分解 KV cache, 但忽略 inter-token similarity. DeltaKV 用 reference subtraction 显式提取 shared component, 比纯 SVD-style 低秩更高效。

### 6.2 与 Retrieval-Augmented Generation (RAG) 的联系

DeltaKV 本质上在 KV cache 内部做了一个 **internal retrieval**: 给当前 token 找 history 中最相似的几个 reference, 用它们的 mean 作为 "prior", 压缩 delta。这跟 RAG / memory network 的思想高度一致, 但是在 KV cache 层面 native 实现, 不需要外部 retrieval 数据库。

### 6.3 与 Model Editing / Activation Steering 的关联

Activation steering (如 Representation Engineering, ReFT) 通过修改 hidden state 的特定 direction 来控制 model 行为。DeltaKV 的 reference subtraction 类似先做 "成分剥离"——剥离 shared structure, 只保留 token-specific 的 residual, 这跟 model editing 中 "find steering direction, project out" 的操作形式上一致。

参考: ReFT paper https://arxiv.org/abs/2402.04092

### 6.4 Linear Attention / Gated Linear Attention 的关联

Mamba / GLA / RetNet 等线性 attention 模型本质是用一个 compressible state 替代 KV cache. DeltaKV 可以看作在 standard Transformer attention 内部, 通过工程手段达到类似的 compression 效果——把历史信息压缩进一个小的 latent code。

参考:
- Mamba: https://arxiv.org/abs/2312.00752
- Gated Linear Attention: https://arxiv.org/abs/2312.06635
- RetNet: https://arxiv.org/abs/2307.08621

### 6.5 VQ-VAE / Product Quantization 的对比

PQCache (Zhang et al., 2025) 用 product quantization, 把 KV 压成 codebook indices. DeltaKV 不用 codebook, 而是用 reference + residual 的 parametric compression。两者 trade-off:
- VQ: 极高 compression ratio, 但 codebook lookup 有 memory access pattern 问题
- DeltaKV: moderate compression, 但 GPU-friendly, 易融合 kernel

### 6.6 关于 Anisotropy

KV cache 的 anisotropy (各向异性) — 少数方向 capture 大部分能量 — 这跟 token embedding 的 anisotropy 现象 (Ethayarajh 2019) 类似. Geva et al. 的 "Transformer Feed-Forward Layers Are Key-Value Memories" (https://arxiv.org/abs/2012.14913) 也指出 FFN 类似于 key-value memory. KV cache 的 high-norm shared components 很可能是这些 FFN memories 的 propagation。

### 6.7 关于 "Train Short, Test Long" 的 generalization

DeltaKV 在 8192 length 训练但 generalize 到 100k+, 原因是 pre-RoPE 操作。这跟 length generalization 的研究 (如 YaRN, https://arxiv.org/abs/2309.00071, NTK-aware scaling) 不同角度——后者是修改 RoPE scaling, 前者是绕开 RoPE 直接在内容空间操作。

### 6.8 推论: Attention Sink 现象的新解释?

StreamingLLM (Xiao et al., 2023b) 发现 attention sink tokens (通常在 sequence 开头) 极其重要, 删掉它们 perplexity 爆炸。DeltaKV 的 reference set 是否包含 sink tokens? 如果 sink tokens 高 norm 且 globally shared, 它们会被频繁选作 reference。这可能是 attention sink 现象的另一个视角: sink tokens 是 dominant shared latent components 的载体。

### 6.9 未来方向: Full-Pipeline Quantization

Paper Appendix B.4 提到: 如果对 full attention layers 和 reference tokens 也 4-bit quantize, 理论上可降到 **7.2%** of original size. 这是 consumer GPU 上跑 million-token context 的 path. 但要保持精度, 需要 complex kernel fusion。

### 6.10 关于 vLLM PagedAttention 的对比

vLLM 的 PagedAttention (Kwon et al., 2023) 用 OS-style virtual memory 管理 KV cache (page-level), 解决 fragmentation 问题。但 PagedAttention 假设 page 内 contiguous, 适合 batch serving. Sparse-vLLM 把 memory 抽象为 CacheManager, **token-level** 而非 page-level 管理, 这跟 LRU/LFU cache eviction 天然 align。

参考 vLLM: https://arxiv.org/abs/2309.06180

## 7. 局限与思考

1. **Reference retrieval 开销**: 即使 stride=10, 1M context 有 100k reference, 每 token 要做 L2 search. 作者说 GPU 矩阵乘很快, 但 BS 大时仍可能瓶颈
2. **Training cost**: 8 GPU hours 听起来低, 但每个 model 要重训 (compressor 是 per-model 的, 不 universal)
3. **Distribution mismatch on R.KV**: SSID-like 字符串压缩效果差, 说明对某些 distribution (高 entropy, 难压缩模式) DeltaKV 不够鲁棒
4. **Layer-wise full attention 选择**: 手动基于 importance profiling 选 layer indices (Table 7), 不够自动化. 可考虑 learned gating
5. **Python overhead**: 当前 Sparse-vLLM 仍有 Python 控制流开销, 完全 fused kernel 是 future work

## 8. 总结性 Intuition

DeltaKV 的核心 insight 是 **"KV cache = shared structural backbone + token-specific delta"**. 通过在 pre-RoPE 空间 retrieve 历史 reference 并 subtract, 显式剥离 shared components, 只 compress 剩下的 low-magnitude residual. 这个操作在 attention 之前做, 保留 position encoding 的功能。

更深层的意义: 这是把 "内容相似性" 作为 inductive bias 引入 KV cache 压缩, 而非纯数学上的低秩或量化。Token 之间的语义相似性 (在 embedding 空间度量) 对应到 KV space 也相似, 这就是 long-range similarity 的来源. Compression 在 content space 做更 intelligent, 比 pure numerical compression (quantization, low-rank) 更 accurate。

sparse attention + compression 的组合也很有意思: OmniKV 提供了 "哪些 token 重要" 的 dynamic 选择, DeltaKV 提供 "如何更便宜存储所有 token" 的手段, 两者互补。

## 参考链接

- vLLM: https://github.com/vllm-project/vllm
- SGLang: https://github.com/sgl-project/sglang
- LightLLM: https://github.com/ModelTC/lightllm
- SnapKV: https://arxiv.org/abs/2404.14469
- H2O: https://arxiv.org/abs/2306.14048
- PyramidKV: https://arxiv.org/abs/2406.02069
- AdaKV: https://arxiv.org/abs/2407.11550
- Quest: https://arxiv.org/abs/2406.10774
- OmniKV: https://openreview.net/forum?id=OmniKV
- Palu: https://arxiv.org/abs/2407.21118
- PQCache: https://arxiv.org/abs/2409.02636
- KIVI: https://arxiv.org/abs/2402.02750
- KVQuant: https://arxiv.org/abs/2401.18079
- CacheGen: https://arxiv.org/abs/2310.07240
- Chelsea: https://arxiv.org/abs/2506.11418
- Lexico: https://arxiv.org/abs/2412.08890
- ClusterKV: https://arxiv.org/abs/2412.02130
- LongBench: https://github.com/THUDM/LongBench
- SCBench: https://arxiv.org/abs/2412.10319
- RoPE / RoFormer: https://arxiv.org/abs/2104.09864
- StreamingLLM: https://arxiv.org/abs/2309.17453
- Mamba: https://arxiv.org/abs/2312.00752
- ReFT: https://arxiv.org/abs/2402.04092
- FFN as Key-Value Memories: https://arxiv.org/abs/2012.14913
- YaRN: https://arxiv.org/abs/2309.00071
- Llama 3: https://arxiv.org/abs/2407.21783
- Qwen2.5-1M: https://arxiv.org/abs/2501.15383
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- PagedAttention/vLLM paper: https://arxiv.org/abs/2309.06180
- Sparse-vLLM (paper code): https://github.com/CURRENTF/Sparse-vLLM
