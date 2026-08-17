---
source_pdf: MemoryLayersatScale.pdf
paper_sha256: 8928f823d2d6db370c10ee7644102f408e94cbeba91d7ca8de9b50b88c520b8f
processed_at: '2026-08-05T17:36:45-07:00'
target_folder: LLM-Training/nanogpt
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用"人话"讲透 Memory Layers at Scale

Andrej, 咱们用最直白的直觉来拆解这篇 paper。如果一个 LLM 是个参加 trivia 问答比赛的学生, Dense transformer 试图把 Wikipedia 上所有的 facts 死记硬背在脑神经突触里。每次学一个新 fact, 比如 "巴黎是法国首都", 都要调整数百万个 synaptic weights。这非常消耗能量和算力。

Memory layer 的逻辑非常符合人类直觉: 给这个学生发一本巨大的笔记本。当被问到 "法国首都是哪里", 他不调用脑神经去回忆, 直接翻开笔记本查找 "France" 这一页, 读出 "Paris"。

这本笔记本就是 **sparse key-value lookup table**。它让 LLM 的 **parameter count** 暴涨, 但 **FLOPs** 几乎不变, 因为查字典不需要做矩阵乘法, 只需要算 cosine similarity 和 gather operation。

Paper 核心贡献在于, 他们把这本笔记本做到了 **128 Billion 页** (128B memory parameters), 并且证明了这在 1 Trillion tokens 的 scale 下依然 smooth scaling, 甚至在 factual QA 任务上用 1.3B base model + 128B memory 打平了用 10 倍算力训练的 Llama 7B。

参考链接: 
- Paper Code: https://github.com/facebookresearch/memory
- Meta FAIR Blog: https://ai.meta.com/blog/meta-fair-updates-agents-robustness-safety-architecture

---

## 1. 从 Dense 到 Sparse Memory: 核心数学公式

Dense FFN 的计算是 $y = W_2 \sigma(W_1 x)$, 存储信息完全依赖 weight matrix 的 dense multiplication。

Memory layer 借用了 attention 的 QKV 形式, 但 K 和 V 是 **trainable parameters**, 并且做 top-k sparse retrieval:

$$
I = \mathrm{SelectTopkIndices}(K q) \quad \text{(1a)}
$$
$$
s = \mathrm{Softmax}(K_I q) \quad \text{(1b)}
$$
$$
y = s V_I \quad \text{(1c)}
$$

变量与上标下标解析:
- $q \in \mathbb{R}^n$: Query vector, 来自前一层 attention sublayer 的 output, $n$ 代表 base model 的 hidden dimension (e.g., Llama 1.3B 是 2048)。你可以把它理解为 "学生脑子里生成的查找线索"。
- $K \in \mathbb{R}^{N \times n}$: Trainable key matrix, $N$ 代表 notebook 里的总页数 (论文最大到 64 million), $n$ 是 key 的维度。这就是 "字典的索引"。
- $V \in \mathbb{R}^{N \times n}$: Trainable value matrix, 同样 $N \times n$。这是 "字典页面上写的实际内容"。
- $I$: Top-k indices 集合, 选出与 $q$ 最相似的 $k$ 个 pages。
- $K_I, V_I \in \mathbb{R}^{k \times n}$: 被 top-k 选中的 key 和 value submatrix。
- $s \in \mathbb{R}^k$: Softmax 得分, 代表这 $k$ 个 pages 的相对相关度。
- $y \in \mathbb{R}^n$: 最终 output, 注入回 transformer residual stream。

直觉: 模型学会了 **生成什么样的 query** 能触发 **正确的 key**, 从而取出 **正确的 value**。这比 dense weight 去编码 factual association 自然得多。

---

## 2. Product-Key Lookup: 怎么在 6400 万页里瞬间查字典?

一个巨大的工程挑战是, 如果 $N = 64,000,000$, naive top-k 需要算 $64M \times 2048$ 次 dot product, 这在 GPU 上慢得不可接受。

Paper 采用 Product-Key Memory (Lample 2019) 的 trick。直觉上: 你不需要真的存 6400 万个完整索引。你只需存 8000 个 "姓" 和 8000 个 "名"。它们的组合就构成了 6400 万个完整名字。

数学公式上, 把 key 矩阵拆分为两半:

$$
K_1, K_2 \in \mathbb{R}^{\sqrt{N} \times \frac{n}{2}}
$$

变量解释:
- $\sqrt{N}$: Half-key 的数量。如果总 key 数 $N=64M$, 则 $\sqrt{N} = 8000$。
- $\frac{n}{2}$: 每半个 key 的维度。总 key 维度 $n$ 被对半切分。

完整的 key 集合是 $K_1$ 和 $K_2$ 的 Cartesian product (笛卡尔积), 从未在内存中显式实例化。

Query $q$ 也拆分为 $q_1, q_2 \in \mathbb{R}^{n/2}$。算法步骤:
1. 在 $K_1$ 中找 top-k, 得 $I_1, s_1$。
2. 在 $K_2$ 中找 top-k, 得 $I_2, s_2$。
3. 候选集为 $I_1 \times I_2$ (共 $k^2$ 个组合), Score $= s_1[i_1] + s_2[i_2]$。
4. 在 $k^2$ 个候选里取 final top-k。

复杂度从 $O(N \cdot n)$ 降到 $O(\sqrt{N} \cdot n + k^2 \log k)$。当 $N=64M$ 时, 加速比达 8000 倍。这好比用两步哈希代替一次全表扫描。

参考: https://arxiv.org/abs/1907.05242

---

## 3. Memory+ Block: 加上 Silu Gating 防止查错字典乱说话

光给字典内容还不够。如果模型查出 "Paris" 的 value embedding, 但当前 context 根本不需要这个 fact, 直接把 $y$ 加到 residual stream 会导致训练不稳定, 甚至破坏已有 representation。

Memory+ 加了 input-dependent gating:

$$
\mathrm{output} = (y \odot \mathrm{silu}(x^T W_1))^T W_2 \quad \text{(2)}
$$

变量解析:
- $x \in \mathbb{R}^n$: 输入 memory layer 的 token embedding (前一层 attention output)。
- $W_1 \in \mathbb{R}^{n \times n}$: Up-projection weight, 生成 gate signal。
- $W_2 \in \mathbb{R}^{n \times n}$: Down-projection weight, 把 gated output 映射回 residual stream。
- $\mathrm{silu}(x) = x \cdot \mathrm{sigmoid}(x)$: SiLU 激活函数。当 $x < 0$ 时 output 接近 0 (关闭 gate), 当 $x > 0$ 时近似线性 (打开 gate)。
- $\odot$: Element-wise Hadamard product。
- $y \in \mathbb{R}^n$: 公式 (1c) 算出的 memory retrieval result。

直觉: $\mathrm{silu}(x^T W_1)$ 是一个开关。模型基于当前 input context 学会 "我现在需不需要查字典", 以及 "查出来的东西要多少比例加进我的思维流里"。这种 non-linear gating 避免了 irrelevant facts 污染 dense reasoning pathway。

稳定性上, 大规模 memory 训练容易爆 loss, paper 引入了 qk-normalization (Chameleon Team 2024) 来稳定 key/query 的分布。

参考: 
- SiLU / GELU: https://arxiv.org/abs/1606.08415
- Chameleon qk-norm: https://arxiv.org/abs/2405.09818

---

## 4. 为什么能干翻 MOE? 核心差异在 Granularity

这或许是 paper 最深刻的 insight。MOE (Mixture of Experts) 同样是 sparse parameter augmentation, 但 memory layer 在 factual tasks 上以 1.3B base model 为例, NQ 准确率达 13.68, 碾压 parameter 匹配的 MOE (8.14)。

差异在于 **storage granularity**:
- MOE 拥有几十个 experts, 每个 expert 是一个完整 FFN。这就像脑子里有 8 个分管不同领域的 "顾问", 每个 consultant 脑子里都塞满了上万条 facts。Routing 决定找哪个顾问, 但顾问之间会 interference, 且 expert collapse 需要复杂的 auxiliary loss 来平衡。
- Memory layer 拥有 1 million 个独立的 key-value pairs。每个 fact 储存在自己独立的 "插槽" 里。没有任何 competition 或 superposition。Query 直接精确命中对应 fact。

LLM 学的 factual knowledge 本质上就是高度离散的 lookup table。Memory layer 的归纳偏置完美匹配 factual recall 的数据结构。

---

## 5. 实验数据表深度解析

### 5.1 1.3B Base Model 对比 (Table 1)

| Model | Total Params | NQ | TQA | PIQA | OBQA | HotPot |
|---|---|---|---|---|---|---|
| Dense 1.3B | 1.3b | 7.76 | 32.64 | 72.74 | 23.40 | 13.92 |
| MOE | 3.545b | 8.14 | 31.46 | 73.72 | 25.20 | 15.15 |
| PEER | 3.646b | 12.33 | 42.46 | 73.34 | 26.60 | 15.39 |
| Memory+ | 3.377b | **13.68** | **42.89** | 75.35 | 26.80 | **16.72** |
| Memory+ 64m | 138.748b | 20.78 | 62.14 | 77.31 | 30.00 | 20.47 |
| Llama2 7B (2T) | 7b | 25.10 | 64.00 | 78.40 | 33.20 | 25.00 |

直觉解读:
- 看 NQ (NaturalQuestions) 列, Dense 1.3B 只有 7.76, 加了 1 million memory values 的 Memory+ 直接飙到 13.68, 几乎翻倍。
- Memory+ 64m 版本有 138B params, 但 FLOPs 和 1.3B dense 一样。它在 NQ 上拿到 20.78, 离 Llama2 7B (25.10) 差距很小, 但 Llama 7B 用了 10 倍 FLOPs 和 2 倍 training tokens。
- 这证明了 FLOP-bound 的 dense scaling 并非唯一路径, Memory-bound 的 sparse scaling 在 factual tasks 上效率高出几个数量级。

### 5.2 8B Base Model 接近 Llama3 (Table 2)

| Model (tokens) | HellaS. | MMLU | NQ | TQA |
|---|---|---|---|---|
| Llama3.1 8B (15T) | 60.05 | 66.00 | 29.45 | 70.36 |
| Dense 8B (1T) | 58.90 | 59.68 | 25.24 | 63.62 |
| **Memory+ 8B (1T)** | 60.29 | 63.04 | 27.06 | 68.15 |

直觉解读: Memory+ 8B 只训练了 1 Trillion tokens, 就在 MMLU 上达到 63.04, 接近训练了 15 Trillion tokens 的 Llama3.1 8B (66.00)。这非常惊人。意味着 Memory layer 让模型 **学 facts 的速度极快**, 不需要见 15 次才记住, 见 1 次就能存进 key-value slot。

---

## 6. 工程挑战: 为什么大家以前不这么做?

Paper Section 6 坦言, 这东西实现起来非常痛苦。Dense matrix multiplication 是 GPU 设计的最初目标, 几十年来 cuBLAS 把它优化到了极致。但 Sparse embedding lookup 在 GPU 上其实是非典型 workload。

Memory layer 瓶颈在于 **Memory Bandwidth**, 不在 FLOPs。

PyTorch 自带的 EmbeddingBag operation 慢得要命, 带宽利用率不到 400 GB/s。Meta 团队自己写了 Custom CUDA Kernels, 把 H100 的带宽榨到了 3 TB/s (H100 极限是 3.35 TB/s, 利用率达 90%)。端到端速度比原生 PyTorch 快 6 倍。

Backward pass 更难, 多个 token 的 gradient 要 accumulate 到同一个 key 上。Paper 对比了三种 CUDA 策略:
1. Atomics: 简单粗暴的 atomicAdd, 竞争激烈。
2. Lock: Row-level atomic lock, 在 embedding dimension 上 amortize lock 开销。
3. Reverse_indices: 预先 invert mapping, 让每个 embedding 知道哪些 token 会贡献给它, 变 scatter 为 gather。这是经典 GPU sparse 优化思路。

并且为了把 128B 参数的 value matrix 分布到多 GPU, 他们设计了 **Parallel EmbeddingBag**。按 embedding dimension 把 $V$ 切分到不同 GPU, 每个 GPU 只算自己那一维度的 partial result, 再 all-to-all 通信。

---

## 7. 认知科学直觉: 为什么 Memory 适合 Continual Learning

Paper 在结尾提到了对 Continual Learning 和 Hallucination 的展望。这触及了 AI 架构的深层问题。

Dense neural network 存储 fact 是 **superposition** (叠加) 形态。一个 fact 分布在数百万个 weight 中。学习新 fact 必然修改这些 weights, 导致旧 fact 被覆盖, 也就是 Catastrophic Forgetting。

Memory layer 类似人类大脑的 **Hippocampus** (海马体)。海马体负责快速记忆单次事件, 每个记忆有独立 slot, 互不干扰。这符合 Complementary Learning Systems 理论。

Memory layer 天然支持 **Continual Learning**: 训练完基础模型后, 如果要教模型一个新 fact "Elon Musk bought Twitter", 只需要更新对应的 key-value pair, 完全不干扰其他 facts。这在 Dense model 里几乎不可能做到。

对于 **Hallucination**: 当 query $q$ 找不到强匹配的 key 时, softmax 分布会平坦。配合 silu gating, 模型能学会 "关闭" 输出, 避免胡说八道。这是架构层面的 hallucination mitigator, 比 RLHF 或数据清洗更底层。

---

## 8. Ablation: 几个关键 Hyper-parameter 的权衡

Paper 做了大量 ablation, 构建你的直觉需要看这几个点:

1. **Memory Layer Placement**: 放在 transformer 哪一层?
   - 放中间偏均匀最好 (e.g., Layer 4, 12, 20 in a 24-layer model)。
   - 太浅 (Layer 1-3) 学不到 semantic query, 太深 (Layer 20+) 离 output 太近缺乏 reasoning 传导。
   - 3 个 memory layers 是 sweet spot, 6 个反而退化, 因为 dense FFN 被替换太多, 模型丧失了 transformation 能力。

2. **Shared Memory**: 所有 memory layers 共享同一个 $K, V$ pool。
   - 不同层的 query 不同, 同一个 $V$ 可以被不同 depth 读取。
   - 保持参数量不随 layer 数量线性增长。

3. **Key Dimension vs Value Dimension**:
   - 固定总参数量, Value dim = 1024 (与 base model hidden dim 对齐) 最好。每个 fact 需要足够 bandwidth 的 representation。
   - Key dim 越大越好 (区分度强), 但会增加 dense projection 参数, paper 取 key_dim = n/2。

---

## 9. 与 PEER 和 RAG 的本质区别

- **PEER (He 2024)**: 同样用 product-key, 但 retrieve 的是 rank-1 matrices, 再组合成 dynamic FFN。相当于查字典时, 查出来的是一个个 "算法小模块", 组合起来算一道题。Memory layer 查出来的是直接的内容, 更适合 fact storage, 参数效率更高。
- **RAG (Lewis 2021)**: Non-trainable external datastore。Gradient 不 flow 进 retrieval index。模型学不会如何组织 dictionary, 只能被动检索。Memory layer 是 fully differentiable, Key 会跟着 training objective 走, 模型学会了如何最优地组织 facts。

---

## 10. 总结: Intuition 核心三连击

1. **分工明确**: Dense layer 做 transformation (reasoning), Memory layer 做 storage (fact recall)。把 lookup table 从 FLOPs domain 转移到 Bandwidth domain。

2. **Product-key 是神器**: 把 $O(N)$ 查找降到 $O(\sqrt{N})$, 让 128B params 的 memory 能在单机多卡上 run 起来。

3. **Sparse Update 带来 Continual Learning**: 脱离 dense superposition, 每个 fact 一个 slot, 天然避免 catastrophic forgetting, 为下一代 Agent architecture 提供 long-term memory 基础设施。

Paper 呼吁 "Memory layers should be integrated into all next generation AI architectures"。从 Scaling Laws 和 Compute/Memory Pareto Frontier 看, 这个呼吁具有充分的 empirical 和 theoretical 基础。算力换 parameter 的时代快到头了, 稀疏激活的 trainable memory 是非常明确的下一代方向。

---

## Web References

- Memory Layers at Scale Code: https://github.com/facebookresearch/memory
- Meta FAIR Blog: https://ai.meta.com/blog/meta-fair-updates-agents-robustness-safety-architecture
- Product-Key Memory Paper: https://arxiv.org/abs/1907.05242
- PEER Paper: https://arxiv.org/abs/2407.04153
- End-to-end Memory Networks: https://proceedings.neurips.cc/paper_files/paper/2015/file/8fb21ee7a2207526da55a679f0332de2-Paper.pdf
- Mixture of Experts (Shazeer 2017): https://arxiv.org/abs/1701.06538
- Llama 2: https://arxiv.org/abs/2307.09288
- Llama 3: https://arxiv.org/abs/2407.21783
- RAG: https://arxiv.org/abs/2005.11401
- Scaling Laws for Neural LMs: https://arxiv.org/abs/2001.08361
- SiLU/GELU Paper: https://arxiv.org/abs/1606.08415
- Chameleon qk-norm: https://arxiv.org/abs/2405.09818

---

# Memory Layers at Scale 详细技术讲解

## 1. Paper 核心动机与背景

这篇 paper 来自 Meta FAIR，作者 Vincent-Pierre Berges 和 Barlas Oğuz 等人。核心 question 非常 simple 但深刻: dense transformer 的参数 scaling 必然带 compute scaling, 但 LLM 中大量需要存储的信息其实只是 **associative lookup** —— 例如 celebrity 的 birthday、country 的 capital、concept 之间的 association。这类信息天然适合 key-value memory, 而非 dense matrix multiplication。Dense FFN 理论上是 universal approximator (Hornik 1989), 但用 dense layer 去模拟一个 lookup table 是 **FLOP-inefficient** 的。

Memory layer 的设计哲学: 保持 dense layer 做 "computation" (reasoning, transformation), 让 sparse memory layer 做 "storage" (fact retrieval)。这种解耦让我们可以 scale parameters 而 FLOPs 几乎不变。

与已有方向的关系:
- **MOE** (Shazeer 2017, Lepikhin 2020): 增加 parameters 但每个 token 只激活几个 experts, 每个专家是完整 FFN, 路由是 token-to-expert
- **PEER** (He 2024): 用 product-key 检索 rank-1 matrices, 类似 memory 但每个 "key" 对应两个 embedding, parameters 翻倍
- **RAG** (Lewis 2021, Karpukhin 2020): external database retrieval, non-trainable
- **Memory Networks** (Weston 2015, Sukhbaatar 2015) & **NTM** (Graves 2014): 早期 trainable memory, 未 scale 到 modern size
- **Product-Key Memory** (Lample 2019): 已经提出 product-key trick 但只到 100M-1B 参数级别

这篇 paper 把 memory layer 从 "proof-of-concept" 推到 128B memory params + 1T tokens 的 contemporary scale, 是 **两个数量级**的飞跃。

参考链接:
- Paper: https://arxiv.org/abs/2412.21098 (此处以 file 给出)
- Code: https://github.com/facebookresearch/memory
- Blog: https://ai.meta.com/blog/meta-fair-updates-agents-robustness-safety-architecture
- Product-key 原始 paper: https://arxiv.org/abs/1907.05242
- PEER: https://arxiv.org/abs/2407.04153
- Memory Networks: https://arxiv.org/abs/1410.3916
- End-to-end Memory Net: https://proceeddings.neurips.cc/paper_files/paper/2015/file/8fb21ee7a2207526da55a679f0332de2-Paper.pdf
- MOE: https://arxiv.org/abs/1701.06538
- RAG: https://arxiv.org/abs/2005.11401

---

## 2. Memory Layer 数学形式

给定 query $q \in \mathbb{R}^n$ (来自前一层 attention 的 output), key 矩阵 $K \in \mathbb{R}^{N \times n}$, value 矩阵 $V \in \mathbb{R}^{N \times n}$。Memory layer 的 forward pass:

$$
I = \mathrm{SelectTopkIndices}(Kq) \quad \text{(1a)}
$$

$$
s = \mathrm{Softmax}(K_I q) \quad \text{(1b)}
$$

$$
y = s V_I \quad \text{(1c)}
$$

变量解释:
- $q$: query embedding, 来自 attention sublayer 的 output, 维度 $n$ (base model hidden dim, 例如 Llama-1.3B 是 2048)
- $K$: trainable key matrix, 共 $N$ 行 (key 数量, 论文里到 64M), 每行维度 $n$
- $V$: trainable value matrix, 同样 $N \times n$, 是真正承载 "fact" 的 storage
- $I \in \mathbb{Z}^k$: top-k indices 集合, $k$ 是稀疏激活数 (论文默认 k=256 量级)
- $K_I, V_I \in \mathbb{R}^{k \times n}$: 选中的 key/value submatrix
- $s \in \mathbb{R}^k$: softmax 归一化的相似度权重
- $y \in \mathbb{R}^n$: 输出 embedding, 接回 transformer residual stream

与 attention 的关键差异:
1. $K, V$ 是 **trainable parameters**, 而非 activations (attention 中 $K = X W_K$, $V = X W_V$ 是 input 的函数)
2. $N$ 是 million 级别, 必须 **sparse top-k**; attention 中 $N$ 是 sequence length, 通常 dense

**直觉**: 整个 memory layer 就是一个 internal, differentiable, trainable 的 hash table。Token embedding 作为 query 触发 retrieval, 把相关 fact 的 value embedding 加权汇总回流。

---

## 3. Product-Key Lookup: 巧妙的 scaling trick

Naive 的 top-k 需要 $O(N \cdot n)$ 的 dot product, 当 $N = 64M$ 时不可行。

**Product-key** 思路 (Lample 2019): 把 key 拆成两半, 利用 Cartesian product 的结构。

设两个 half-key 集合:
$$
K_1, K_2 \in \mathbb{R}^{\sqrt{N} \times \frac{n}{2}}
$$

完整的 $N$ 个 keys 是 $K_1 \times K_2$ 的 Cartesian product (从未显式实例化):
$$
K[i_1, i_2] = [K_1[i_1] \,\|\, K_2[i_2]] \in \mathbb{R}^n
$$

其中 $[\cdot \,\|\, \cdot]$ 表示 concatenation。

Query 也对应拆分: $q = [q_1 \,\|\, q_2]$, $q_1, q_2 \in \mathbb{R}^{n/2}$。

Lookup 算法:
1. 在 $K_1$ 上找 top-k: 得 $I_1, s_1$ (各 $k$ 个), 复杂度 $O(\sqrt{N} \cdot n/2)$
2. 在 $K_2$ 上找 top-k: 得 $I_2, s_2$ (各 $k$ 个), 同样复杂度
3. Cartesian product 候选: $k^2$ 个组合 $(i_1, i_2)$, score $= s_1[i_1] + s_2[i_2]$
4. 在 $k^2$ 候选里取 top-k, 复杂度 $O(k^2 \log k)$

总复杂度: $O(\sqrt{N} \cdot n + k^2 \log k)$, 相比 naive $O(N \cdot n)$ 是 $\sqrt{N}$ 倍加速。当 $N = 64M$, $\sqrt{N} = 8K$, 加速比 ~8000x。

**直觉**: 利用 key 的 "factored" 结构, 把一个 $N$-class classification 问题降成两个 $\sqrt{N}$-class 的组合。这种 trick 在 hash table、bloom filter、count-min sketch 里都有类似思想 (tensor product decomposition)。

---

## 4. Memory+ Block: 在 vanilla memory 上加 silu gating

单纯的 memory output $y$ 直接接回 residual stream 在 large scale 训练不稳定, 且 capacity 利用率低。Memory+ 在 $y$ 上加 input-dependent gating:

$$
\mathrm{output} = (y \odot \mathrm{silu}(x^T W_1))^T W_2 \quad \text{(2)}
$$

变量解释:
- $x$: 进入 memory layer 的 input (前一层 attention 的 output)
- $W_1 \in \mathbb{R}^{n \times n}$, $W_2 \in \mathbb{R}^{n \times n}$: 可训练 projection
- $\mathrm{silu}(x) = x \cdot \mathrm{sigmoid}(x)$: SiLU 激活函数 (Hendrycks & Gimpel 2017, GELU 同类)
- $\odot$: element-wise (Hadamard) product
- $y \in \mathbb{R}^n$: 来自公式 (1c) 的 memory retrieval 结果

直觉: memory 输出本身是一个 "fact retrieval", 但模型需要决定 **何时使用这个 fact、用多少**。$x^T W_1$ 是 input-dependent gate, silu 让 gating 是 smooth + non-monotonic, 让模型既能 "trust" memory (gate 接近 1) 也能 "ignore" memory (gate 接近 0)。$W_2$ 再做一次 linear projection, 类似 FFN 的 "down-projection" 之后的输出投影, 让 memory output 融入 residual stream 时不破坏 hidden state 的统计性质。

为什么 silu 比 sigmoid gating 好: silu 在 $x<0$ 时 output 是负值且接近 0, 比 ReLU 更 smooth; 在 $x>0$ 时近似 linear, 比 sigmoid 不容易 saturate。这种 "soft but not bounded" 性质让 gradient flow 在 large memory (large parameter variance) 下更稳定。

参考 Figure 3: 左边是 vanilla Memory (直接 $y$ 进 residual), 右边是 Memory+ (额外加了 $W_1$ projection + silu + $W_2$ projection)。

稳定性技巧: 还会加 **qk-normalization** (Chameleon Team 2024) 来缓解 large memory 训练的 instability, 尤其 small base model 上。

---

## 5. 工程实现: Parallel Memory + Custom CUDA Kernels

### 5.1 Parallel EmbeddingBag

Memory layer 的 value 矩阵 $V$ 是 $N \times n$, 当 $N = 64M$, $n = 1024$ 时, 仅 weights 就是 64G params = 128GB (fp16) = 256GB (fp32 with momentum), 单 GPU 装不下。论文用 **embedding-dim sharding**:

- 把 $V$ 在 $n$ 维度上切分到多个 GPU (Memory Group)
- 每个 GPU 持有 $V_{\text{local}} \in \mathbb{R}^{N \times n/g}$, 其中 $g$ 是 group size
- Forward: 每个 GPU 收到所有 indices $I$, 只 lookup 自己那一 dimension 的子嵌入
- All-to-all gather: 每个 GPU 拿到自己负责的 index 对应的 partial embedding
- 最后每个 GPU 在自己 shard 上做 aggregation

关键: **不实例化完整 $y \in \mathbb{R}^{B \times L \times n}$**, 每个 GPU 只持有 $n/g$ 那一段, 这样 activation memory 也 sharded。

这个并行方案 **独立于** tensor parallel / context parallel / pipeline parallel, 在自己的 process group 内运行, 与 FSDP/DDP/ZeRO 等都可叠加。

### 5.2 Custom CUDA Kernel 性能对比

EmbeddingBag 的 forward 是 memory-bandwidth bound 操作 (FLOP 极少, 主要是 gather + weighted sum)。PyTorch 默认实现效率低下:

| 实现 | 带宽利用率 |
|---|---|
| PyTorch EmbeddingBag | < 400 GB/s |
| Custom CUDA forward | **3 TB/s** (H100 spec 是 3.35 TB/s, 利用率 90%) |
| 端到端加速 | **6x** vs PyTorch |

Backward pass 更 tricky, 因为多个 token 的 gradient 要 accumulate 到同一个 embedding row。论文对比三种策略:

1. **Atomics**: 每个梯度写入用 atomicAdd。简单但 contention 严重, 但已比 PyTorch 快 5x。
2. **Lock**: row-level atomic lock, 把 lock cost 在 embedding 维度上 amortize。当 embedding dim 较大时优于 atomics。
3. **Reverse indices**: 预处理 inverse mapping (embedding_id → list of token_ids), 每个 embedding gradient row 知道哪些 token 贡献给它, 完全 atomic-free。在 embedding dim > 128 且 load balance 时最快。

**直觉**: EmbeddingBag backward 本质是 scatter-add 操作。在 sparse lookup 场景下, 不同 token 命中同一 embedding 的概率 (collision) 取决于 softmax distribution 的 entropy。Reverse_indices 通过预先 invert 索引把 scatter 变成 gather, 是经典 GPU sparse kernel 优化思路 (类似 CSR format 的 SpMV)。

---

## 6. Shared Memory Across Layers

与 Lample 2019 不同, 论文让 **所有 memory layer 共享同一个 key-value pool**。这意味着:
- 不管用 1 个还是 3 个 memory layer, memory 参数总量不变
- 不同层用同一组 keys 但 query 不同 (因为 query 来自不同 depth 的 attention output), 可以 retrieve 不同 facts
- 这是一种 **parameter sharing** 形式, 类似 ALBERT 的 cross-layer parameter sharing

Ablation (Table 3 左):

| Memory layer 位置 | nll | NQ nll | TQA nll |
|---|---|---|---|
| 仅 layer 12 | 2.11 | 12.13 | 8.34 |
| 12, 16, 20 | 2.08 | 11.60 | 7.54 |
| 8, 12, 16 | 2.07 | 11.79 | 7.64 |
| **4, 12, 20** (centered, stride 8) | **2.06** | **11.32** | **7.20** |
| 5, 8, 11, 14, 17, 21 (6 层) | 2.11 | 11.79 | 7.73 |

结论:
- 3 层 sweet spot, 6 层开始退化 (因为 dense FFN 被替换太多, dense capacity 不足)
- Centered + larger stride 最好 (e.g., layer 4, 12, 20 在 24 层 transformer 中均匀且对称分布)

**直觉**: Transformer 不同深度承担不同抽象级别。Early layer 偏 syntactic/lexical, middle 偏 semantic, late 偏 task-specific。Memory layer 在 middle 层放置收益最大, 因为这是 fact-intensive 的层级。Stride 8 让相邻 memory layer 之间有足够 dense layer 做 transformation, 避免 query 重复触发相同 keys。

---

## 7. 实验设置

**Base model**: Llama 系列架构 (Llama2 用于 134m/373m/720m/1.3b, Llama3 用于 8B)。Base model 的 dense FFN 被替换为 memory layer。

**Training data**: 134m-1.3b 用 Llama2-style mix (32k tokenizer), 1T tokens; 8B 用 Llama3-style mix (128k tokenizer)。

**Baselines**:
- Dense: 不加 memory
- **MOE** (Shazeer 2017): expert choice routing (Zhou 2022) 训练, top-1 推理
- **PEER** (He 2024): 用 product-key 检索 rank-1 矩阵对, 参数翻倍但概念类似

**Benchmarks**:
- Factual QA: **NaturalQuestions** (Kwiatkowski 2019), **TriviaQA** (Joshi 2017) - exact match / F1
- Multi-hop: **HotpotQA** (Yang 2018)
- Knowledge: **MMLU** (Hendrycks 2021), **HellaSwag** (Zellers 2019), **OBQA** (Mihaylov 2018), **PIQA** (Bisk 2019)
- Coding: **HumanEval** (Chen 2021), **MBPP** (Austin 2021) - pass@1

参考:
- NQ: https://aclanthology.org/Q19-1026
- TriviaQA: https://arxiv.org/abs/1705.03551
- HotpotQA: https://arxiv.org/abs/1809.09600
- MMLU: https://arxiv.org/abs/2009.03300
- HellaSwag: https://arxiv.org/abs/1905.07830
- HumanEval: https://arxiv.org/abs/2107.03374
- MBPP: https://arxiv.org/abs/2108.07732
- Expert Choice routing: https://arxiv.org/abs/2202.09368

---

## 8. 核心实验结果

### 8.1 Compute-controlled 对比 (Table 1)

以 1.3B base model 为例:

| Model | Total Params | NQ | TQA | PIQA | OBQA | HotPot |
|---|---|---|---|---|---|---|
| Dense 1.3B | 1.3b | 7.76 | 32.64 | 72.74 | 23.40 | 13.92 |
| MOE | 3.545b | 8.14 | 31.46 | 73.72 | 25.20 | 15.15 |
| PEER | 3.646b | 12.33 | 42.46 | 73.34 | 26.60 | 15.39 |
| Memory | 3.377b | 9.83 | 39.47 | 72.29 | 25.80 | 15.46 |
| **Memory+** | 3.377b | **13.68** | **42.89** | 75.35 | 26.80 | **16.72** |
| Memory+ 4m | 9.823b | 14.43 | 51.18 | 75.03 | 27.80 | 18.59 |
| Memory+ 16m | 35.618b | 20.14 | 58.67 | 76.39 | 26.80 | 20.65 |
| Memory+ 64m | 138.748b | 20.78 | 62.14 | 77.31 | 30.00 | 20.47 |
| **Llama2 7B (2T tokens)** | 7b | 25.10 | 64.00 | 78.40 | 33.20 | 25.00 |

关键观察:
- Memory+ 1.3B + 64m keys (总 138B params, 但 FLOPs 与 1.3B 相当) 在 NQ 上达到 20.78, 接近 Llama2 7B (25.10) 的 83%, 在 TQA 上达到 62.14, 接近 7B (64.00) 的 97%
- Llama2 7B 用了 2T tokens, 10x FLOPs
- **Memory+ 4m (9.8B params) 已经超越 PEER (3.6B params) 1.6x**, 在相同 base model 下

### 8.2 Scaling memory size (Figure 1)

1.3B base model, 训练 1T tokens, varying memory size:

- Factual QA accuracy 随 memory size **log-linearly 增长**, 一直涨到 64M keys (128B params)
- NLL 也 smooth 改善, 没看到 saturation
- 这是非常 encouraging 的 scaling 信号

### 8.3 8B Scale (Table 2)

8B base + 16m memory values (64B extra params):

| Model (tokens) | HellaS. | Hotpot | HumanE. | MBPP | MMLU | NQ | OBQA | PIQA | TQA |
|---|---|---|---|---|---|---|---|---|---|
| Llama3.1 8B (15T) | 60.05 | 27.85 | 37.81 | 48.20 | 66.00 | 29.45 | 34.60 | 79.16 | 70.36 |
| dense 8B (200B) | 53.99 | 20.41 | 21.34 | 30.80 | 41.35 | 18.61 | 31.40 | 78.02 | 51.74 |
| Memory+ 8B (200B) | 54.33 | 21.75 | 23.17 | 29.40 | **50.14** | 19.36 | 30.80 | 79.11 | 57.64 |
| dense 8B (1T) | 58.90 | 25.26 | 29.88 | 44.20 | 59.68 | 25.24 | 34.20 | 80.52 | 63.62 |
| Memory+ 8B (1T) | 60.29 | 26.06 | 31.71 | 42.20 | **63.04** | 27.06 | 34.40 | 79.82 | 68.15 |

亮点:
- Memory+ 8B 训练 **1T tokens** 接近 Llama3.1 8B 训练 **15T tokens** (15x 数据) 在 MMLU (63 vs 66)、HellaSwag (60 vs 60)、PIQA、OBQA、Hotpot 上
- Memory+ 在 200B tokens 时 MMLU 50.14 vs dense 41.35, 差距 +8.79, 说明 memory 让模型 **学 facts 速度快很多**
- Coding (HumanEval/MBPP) 也有提升但相对小, 因为 coding 更偏 reasoning 而非 fact recall

**直觉**: Memory layer 让 LLM 不必把每个 fact 压缩进 dense weight 的 nonlinear function, 而是直接存储到 value embedding。新 fact 通过 sparse update 就能学到, 而 dense layer 需要调整大量 weights 才能 encode 同样信息。这也暗示 **continual learning** 和 **减少 hallucination** 上的潜力 (paper section 6 提到)。

---

## 9. Ablations 深入

### 9.1 Memory layer 变体 (Table 3 右)

基于 1.3B model 的 NLL ablation:

| 变体 | nll | NQ nll | TQA nll |
|---|---|---|---|
| PK base | 2.11 | 12.13 | 8.34 |
| +gated (线性 gating) | 2.11 | 12.24 | 8.17 |
| **+swilu** | 2.11 | 12.05 | **8.09** |
| +random values (随机 KV 加入 top-k) | 2.11 | 12.36 | 8.09 |
| +softmax sink (固定 anchor key) | 2.11 | 12.19 | 8.04 |

观察:
- **swilu** 是最 consistent 的提升 (NQ 和 TQA nll 都降)
- 单纯 linear gating 不稳定, swilu 已经包含 gating 效果
- Random values 和 softmax sink 在 small model 上 minor 改善, 但 large scale 时不 consistent 且 training speed 降低, 最终 paper 没用

**Softmax sink** 思想类似 Attention sink (Xiao 2023 StreamingLLM): 给 softmax 一个 "无意义但恒定" 的 anchor, 让其他 key 的 softmax distribution 更稳定。这种 trick 在长序列 attention 中很有效, 但 memory layer 这里收益 marginal。

### 9.2 Value vs Key dimension (Table 4)

固定总 memory 参数量, 改变 value dim 和 #values 的 trade-off:

| v_dim | #values | nll | NQ nll | TQA nll |
|---|---|---|---|---|
| 64 | 16m | 2.15 | 12.86 | 8.75 |
| 256 | 4m | 2.14 | 12.63 | 8.49 |
| **1024** | **1m** | **2.11** | **12.13** | **8.34** |
| 2048 | 512k | 2.14 | 12.49 | 8.53 |

373m base model (latent dim 1024) 上, value dim = base model dim (1024) 是最优。说明每个 fact 需要 "足够维度" 的 representation, 不能光增加 #facts 而压缩每个 fact 的 capacity。

Key dim 的影响 (固定 value dim):

| key_dim | nll | NQ nll | TQA nll |
|---|---|---|---|
| 256 | 2.11 | 12.13 | 8.34 |
| 512 | 2.12 | 12.32 | 8.15 |
| 1024 | 2.11 | 12.37 | 8.25 |
| 2048 | 2.09 | 11.98 | 7.83 |

Key dim 越大 NLL 越好 (2048 时 NQ nll 11.98 vs 256 时 12.13)。但 key dim 增加 dense parameters, 不能无限大。Paper 选 key_dim = base_dim / 2。

**直觉**: Key 承担 "addressing" 任务, 维度越大越能区分细微 query 差异; Value 承担 "content" 任务, 维度需要匹配 base model 的 representation bandwidth。两者需要 balance, 而非单纯扩大一个。

---

## 10. Why Memory Beats MOE on Factual Tasks

这是一个关键问题。MOE 和 Memory 都是 sparse-activated parameter augmentation, 为什么 Memory 在 factual QA 上明显胜出?

Table 1 1.3B 行: MOE 3.5B params NQ 8.14, Memory+ 3.4B params NQ 13.68, 差距 ~70%。

我的分析 (paper 里隐含但没明说):

1. **Granularity 差异**: MOE 每个 expert 是一个完整 FFN (e.g., 4hidden dim), 数量最多几十个。Memory 有百万级 keys, 每个 key-value 是一个独立 "fact slot"。Factual knowledge 是高度 **granular** 的 (一个 fact = 一对 key-value), Memory 的 granularity 完美匹配。

2. **Routing 责任**: MOE 的 router 决定 token → expert, 一个 expert 处理多种 fact 类型, 容易 interference。Memory 的 key 自身就是 "fact identifier", query 直接 retrieve 对应 fact, 无 routing bottleneck。

3. **Capacity utilization**: MOE 在 expert choice routing 下, 不同的 experts 处理不同的 batch 分布; 但每个 expert 只在被路由到时才更新。Memory 的每个 key-value pair 几乎在每个 batch 都被部分 update (因为 softmax 给 top-k 都分配 weight), **dense update on sparse subset**。

4. **No expert collapse**: MOE 有 expert collapse / load balancing 问题, 需要 auxiliary loss。Memory 通过 product-key 的对称结构, key 之间自然分散, 不需要 load balancing loss。

5. **Inference latency**: MOE top-1 时每 token 调用 1 个 expert (一个 FFN 的 FLOPs); Memory top-k 时每 token 调用 k 个 value (k 个 embedding 加权和, FLOP 极少, 但 memory bandwidth 重)。两者 FLOP 都低, 但 memory layer 真正的瓶颈是 bandwidth, 这是 paper section 3 重点优化的。

---

## 11. 与其它 memory-augmented 工作的对比

| Method | Keys trainable? | Scale (params) | Sparse? | Update mechanism |
|---|---|---|---|---|
| Memory Networks (Weston 2015) | Yes | small | weak | end-to-end |
| End-to-end Memory Net (Sukhbaatar 2015) | Yes | small | weak | end-to-end |
| NTM (Graves 2014) | Yes | small | no | controller + read/write heads |
| Product-Key Memory (Lample 2019) | Yes | ~1B | top-k | product-key + top-k |
| kNN-LM (Khandelwal 2020) | No (external datastore) | arbitrary | kNN | non-trainable retrieval |
| RAG (Lewis 2021) | No | arbitrary | retrieval | non-trainable |
| PEER (He 2024) | Yes | large | top-k | product-key + rank-1 |
| **Memory Layers (this paper)** | **Yes** | **128B** | **top-k + product-key** | **end-to-end + shared + silu** |

参考:
- kNN-LM: https://arxiv.org/abs/1911.00172
- REALM: https://arxiv.org/abs/2002.08909
- RETRO (未在 paper 引用但相关): https://arxiv.org/abs/2112.04426

**关键区分**: 这篇 paper 的 memory 是 **in-network trainable**, 与 RAG 的 external datastore 本质不同。RAG 需要 external retrieval system + non-differentiable index, gradient 不 flow 到 retrieval; Memory layer 是 fully differentiable, key 和 value 都通过 backprop 更新。这使得 model 可以 **学习如何组织 memory**, 而 RAG 是 fixed index。

---

## 12. Engineering & Practical Implications

### 12.1 硬件友好性

Paper section 6 坦言: dense FFN 与 GPU co-evolved 数十年, memory layer 还远没到 dense 的 efficiency。但 paper 已展示 custom CUDA 能达到 H100 峰值带宽的 90% (3/3.35 TB/s)。

未来方向:
- **HBM-aware kernel**: 把 key query、top-k、value gather 融合到 single kernel, 减少 global memory round-trip
- **Sparse update optimization**: backward pass 的 reverse_indices 思路可以扩展到 multi-GPU
- **Quantized memory**: 64M keys fp16 是 128GB, 如果 int8 量化可减半, int4 可减到 32GB, 让 8B base + memory 在 single H100 (80GB) 上可行

### 12.2 Continual Learning 暗示

Paper section 6 提到 "fewer hallucinations, and continual learning" 的 potential。直觉是:
- Dense weight 的 fact 存在 superposition 中, 新 fact 会 interference 旧 fact (catastrophic forgetting)
- Memory layer 的 fact 是 sparse addressable, update 一个 key-value 不影响其他, **天然支持 continual learning**
- 这与 LoRA、Tennie 的 complementary learning systems 理论呼应: hippocampus (sparse episodic) + neocortex (dense semantic)

### 12.3 Hallucination Reduction

如果一个 fact 在 memory 中有 dedicated slot, 模型 retrieve 时 confidence 高; 如果 query 没匹配上任何 strong key, softmax 分布平坦, gating 可以 suppress output。这给 hallucination 提供了 architectural handle, 而 dense model 只能通过 training data 或 RLHF 来 mitigate。

### 12.4 与 Linear Attention / RNN 的关系

Memory layer 的 key-value lookup 与 linear attention (Katharopoulos 2020)、RWKV、Mamba 的 state 有形式上的相似:
- Linear attention: $y = \sum_i \phi(K_i) \phi(q) V_i$, 是 "in-context memory" (KV 来自当前 sequence)
- Memory layer: $y = \sum_{i \in \text{topk}} s_i V_i$, 是 "parameter memory" (KV 是 trainable)
- Mamba: state-space, memory 是 compressed recurrent state

Memory layer 可以理解为 **static global memory**, 而 RNN/linear attention 是 **dynamic per-sequence memory**。两者可以共存: Mamba 处理 long context, memory layer 处理 long-term facts。

参考:
- Linear Transformers: https://arxiv.org/abs/2006.16236
- Mamba: https://arxiv.org/abs/2312.00752
- RWKV: https://arxiv.org/abs/2305.13048

---

## 13. 局限与 Open Questions

Paper 自承 (section 6):
1. **Engineering 还不成熟**: 不能 plug-and-play 替换 dense FFN, 需要 custom kernel
2. **只测 short-form QA**: long-form generation、agentic tasks、math reasoning 未充分测试
3. **Memory lookup 还是 token-independent**: 每个 token 独立 lookup, 没利用 cross-token context (vs RNN/attention 的 stateful memory)

我的额外思考:
1. **Key collision**: 64M keys 时, 不同 fact 可能 compete 相同 key region, 是否产生 interference? Paper 没分析 key 的 utilization 分布
2. **Routing interpretability**: 哪些 query 触发哪些 keys? 可解释性研究 open
3. **Compositionality**: Memory 是 "flat" lookup table, 复合 fact (e.g., "X 的妻子的生日") 需要多次 lookup, 是否需要 multi-hop memory addressing (类似 Neural Turing Machine 的 multi-head read)?
4. **Memory editing**: 能否在 inference 时直接 edit 一个 key-value 来更新 fact (model editing)? 这是 dense model 难做的, memory layer 极适合
5. **Cross-modal memory**: Vision-audio-text unified memory 是否可行? Chameleon (paper 引用) 的多模态 setting 可以扩展
6. **Hierarchical memory**: 当前所有 keys 同一层, 是否需要 coarse-to-fine key hierarchy (类似 RAG 中的 hierarchical clustering)?

---

## 14. 与 LLM 训练范式的 broader implication

如果 memory layer 真的能 scale 到 1T+ memory params (paper 只到 128B), 且 inference bandwidth 解决, 那么:
- **Pretraining data 利用率** 大幅提升: 当前 dense model 学一个 fact 需要见多次, memory layer 一次 update 就能 store
- **Compute / memory 的 Pareto frontier 重新画**: paper Figure 1 显示 1.3B + 128B memory 接近 7B dense, 是 ~10x compute 节省, 但 memory storage 大增, 改变 TCO 模型
- **Personalization & on-device memory**: base model 小 + large memory 可以做 per-user memory, 隐私 + 持续学习, 适合 agentic use case

结合 Meta 当前 agent 战略 (paper blog post 链接) , memory layer 是 agent architecture 的自然组件: agent 需要 long-term memory, fact memory, tool memory, 这些都适合 sparse key-value store。

---

## 15. 总结: Intuition 的核心

1. **Dense = compute-bound storage, Memory = bandwidth-bound storage**。LLM 的 fact 不需要 dense computation, 需要 addressable storage。Memory layer 把 storage 从 FLOP domain 移到 bandwidth domain, 解耦 parameter scaling 与 compute scaling。

2. **Product-key + top-k = scalable retrieval**。$O(\sqrt{N})$ lookup 让 64M keys 可行, 这是 paper 能 scale 到 128B 的核心技术。

3. **Shared memory + silu gating = stable training**。共享降低参数总量, silu gating 提供 input-dependent trust, qk-norm 保证 large-scale 稳定性。

4. **Sparse update = natural continual learning substrate**。每个 fact 在独立 slot, 互不干扰, 与 dense superposition 形成对比。

5. **Factual tasks benefit most**。因为 facts 是 "lookup-like", memory layer 的归纳偏置完美匹配; reasoning tasks 受益少因为它们需要 dense transformation。

6. **Engineering 是 bottleneck**。Paper 6x kernel speedup 还远没到 dense FFN 的优化深度, 这是 next 1-2 年的工程方向。

整体看, 这是 **after-MoE 的下一代 sparse parameter augmentation**。MoE 的粒度是 expert (coarse), memory 的粒度是 key-value pair (fine), 与 fact 的天然粒度对齐。如果未来 2-3 年硬件对 sparse embedding lookup 优化到位, memory layer 很可能成为 standard architecture component, paper 的呼吁 "should be integrated into all next generation AI architectures" 是有充分 empirical + theoretical 依据的。

---

## Web Reference 汇总

- Memory Layers at Scale (this paper, GitHub): https://github.com/facebookresearch/memory
- Meta FAIR blog: https://ai.meta.com/blog/meta-fair-updates-agents-robustness-safety-architecture
- Product-Key Memory (Lample 2019): https://arxiv.org/abs/1907.05242
- PEER (He 2024): https://arxiv.org/abs/2407.04153
- Memory Networks (Weston 2015): https://arxiv.org/abs/1410.3916
- End-to-end Memory Networks (Sukhbaatar 2015): https://proceedings.neurips.cc/paper_files/paper/2015/file/8fb21ee7a2207526da55a679f0332de2-Paper.pdf
- Neural Turing Machines (Graves 2014): https://arxiv.org/abs/1410.5401
- MOE (Shazeer 2017): https://arxiv.org/abs/1701.06538
- GShard (Lepikhin 2020): https://arxiv.org/abs/2006.16668
- Expert Choice Routing (Zhou 2022): https://arxiv.org/abs/2202.09368
- Mixtral: https://arxiv.org/abs/2401.04088
- RAG (Lewis 2021): https://arxiv.org/abs/2005.11401
- DPR (Karpukhin 2020): https://arxiv.org/abs/2004.04906
- REALM (Guu 2020): https://arxiv.org/abs/2002.08909
- kNN-LM (Khandelwal 2020): https://arxiv.org/abs/1911.00172
- Scaling Laws (Kaplan 2020): https://arxiv.org/abs/2001.08361
- GPT-3 (Brown 2020): https://arxiv.org/abs/2005.14165
- Emergent Abilities (Wei 2022): https://arxiv.org/abs/2206.07682
- Llama 2 (Touvron 2023): https://arxiv.org/abs/2307.09288
- Llama 3 (Dubey 2024): https://arxiv.org/abs/2407.21783
- Chameleon (Team 2024): https://arxiv.org/abs/2405.09818
- GELU/SiLU (Hendrycks 2017): https://arxiv.org/abs/1606.08415
- Hallucination Survey (Ji 2023): https://arxiv.org/abs/2202.03629 (ACM DOI: 10.1145/3571730)
- SiLU / SwiGLU 相关 SwiGLU paper (Shazeer 2020): https://arxiv.org/abs/2002.05202
- Billion-scale similarity search (FAISS, Johnson 2019): https://arxiv.org/abs/1702.08734
- Universal Approximator (Hornik 1989): https://www.sciencedirect.com/science/article/pii/0893608089900208
- Attention is All You Need (Vaswani 2023): https://arxiv.org/abs/1706.03762
- NQ (Kwiatkowski 2019): https://aclanthology.org/Q19-1026
- TriviaQA: https://arxiv.org/abs/1705.03551
- HotpotQA: https://arxiv.org/abs/1809.09600
- MMLU: https://arxiv.org/abs/2009.03300
- HellaSwag: https://arxiv.org/abs/1905.07830
- OBQA: https://arxiv.org/abs/1809.02789
- PIQA: https://arxiv.org/abs/1911.11641
- HumanEval: https://arxiv.org/abs/2107.03374
- MBPP: https://arxiv.org/abs/2108.07732
- KILT benchmark: https://arxiv.org/abs/2009.02253
- How much knowledge (Roberts 2020): https://arxiv.org/abs/2002.08910
- StreamingLLM / Attention Sink (Xiao 2023, paper 没引用但相关): https://arxiv.org/abs/2309.17453
- Linear Transformers (Katharopoulos 2020): https://arxiv.org/abs/2006.16236
- Mamba (Gu & Dao 2023): https://arxiv.org/abs/2312.00752
- RWKV: https://arxiv.org/abs/2305.13048
- RETRO (Borgeaud 2022, 相关): https://arxiv.org/abs/2112.04426

如需进一步深入某个 ablation 或某个 engineering 细节 (例如 reverse_indices CUDA kernel 的具体实现, 或 product-key backward 的 gradient flow 推导), 我可以继续展开。
