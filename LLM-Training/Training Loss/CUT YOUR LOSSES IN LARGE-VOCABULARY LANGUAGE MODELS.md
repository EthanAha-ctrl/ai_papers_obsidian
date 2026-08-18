---
source_pdf: CUT YOUR LOSSES IN LARGE-VOCABULARY LANGUAGE MODELS.pdf
paper_sha256: 19026f0d3f943c7b47e50f27a0c3319aec809ed2f22ad11df0419b67b53947c7
processed_at: '2026-08-18T04:24:30-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 人话版

好, 我把公式都收起来, 咱们就聊这件事。

---

## 问题到底出在哪

训练 LLM 的时候, 你往前看, 显存一步一步被攻克了: optimizer states 有 ZeRO 分片, activation 有 checkpointing, attention matrix 有 FlashAttention. 等这些全搞定之后, 你回头看 —— 哦, cross-entropy loss 这一层居然占了整个训练显存的 90%.

为什么? 因为 loss 层要算一个 logits 矩阵, shape 是 `[batch 里的 token 数, vocabulary 大小]`. 现在的模型 vocab 动不动 128K、256K (Gemma 2 2B 是 256K), 一个 batch 假设 8192 tokens, 这个矩阵就是 8192 × 256K = 21 亿个数, 存成 fp32 就是 8 GB, PyTorch 还要留一份 bf16 copy, 再算 gradient 又一份, 加起来 28 GB. Gemma 2 2B 这一个 sequence 的 loss 计算就把 80 GB H100 吃满了。

而且 vocab 还在涨。有人做了 scaling law 说大模型应该配更大 vocab (Tao et al. 2024), 所以这事只会越来越严重。

---

## 核心发现: 你其实不需要那个大矩阵

把 cross-entropy 的公式拆开看, 其实就两部分:

1. **ground-truth token 对应的那一个 logit** —— 你只需要 N 个 scalar, 不需要整个矩阵
2. **一个 log-sum-exp** —— 这是个 normalization 项, 跟 ground-truth 是谁无关, 就是对整个 vocab 求个和

这两个操作**都不需要把完整的 logits 矩阵写到显存里**. 第一个直接 gather 一下 classifier 对应的列就行, 第二个可以一边算 logit 一边累加, 流式地把 LSE reduce 出来。

这跟 FlashAttention 是一个套路 —— 把那个 O(N²) 的大矩阵藏在 SRAM 里, 不落到 HBM。但是 cross-entropy 这里可以更狠, 下面说。

---

## 怎么做的: 三个 trick 叠加

**Trick 1: indexed matmul fused**

第一项 (ground-truth logit) 不用先把整个 logits 算出来再 gather, 而是直接进 Triton kernel, 在 SRAM 里 gather classifier 对应的列, 直接算 dot product, 写出去只是一个 N 维向量。中间的 logits 矩阵根本不 materialize。

**Trick 2: online log-sum-exp**

第二项 (LSE) 用跟 FlashAttention 一样的 online softmax 技巧: 把 vocab 切成 block, 每个 CUDA block 算自己那块的 partial LSE, 然后用 atomic 操作 merge 到 global LSE buffer。整个过程中, logits 只在 SRAM 里短暂存在, 用完就丢。

这里跟 FlashAttention 有个区别: FlashAttention 在一个 kernel 里 single-pass 就完事, CCE 因为跨 CUDA block, 需要用 atomic spin-lock 同步, 多了一点点工程复杂度。

**Trick 3: gradient filtering (这是 paper 的 secret weapon)**

backward pass 要用 softmax 矩阵 S, 这个矩阵又是 `|V| × N` 的大块头. 但是仔细想想, softmax over 256K vocab, 每列加起来等于 1, 大部分概率都集中在 top 几十个 token 上。剩下的 25 万个 token, 每个的概率都 < 2^{-12}。

为什么是 2^{-12}? bfloat16 的 mantissa 是 7 bits. 当一个数比另一个数小 2^7 = 128 倍以上时, 加上去就直接被 truncate 成 0. 我们关心的"大数"大约是 2^{-5} 这个量级 (logit softcap 之后), 所以小于 2^{-12} 的小数加进去就消失。

于是: **这些元素的梯度, 不算白不算, 算了也白算**. 直接整块 skip 掉。

实测数据很惊人: frontier models 里只有不到 0.02% 的 softmax 元素是非 trivial 的。也就是说, 99.98% 的计算可以直接跳过。

**Trick 3.5: vocabulary sorting**

gradient filtering 是按 block 跳的, 如果 non-trivial 元素散落在不同 block 里, 每个 block 还都得算。所以用 average logit 给 vocab 排个序, 让"热门" token 聚在一起, "冷门" token 聚在一起, block 级别的 sparsity 就更干净, skip 效率更高。

排序不需要改模型, 因为 softmax 是 permutation invariant 的, 换个 token id 顺序而已。

---

## 效果

Gemma 2 2B 这个 setup:

- Baseline (PyTorch 默认): 28 GB, 208 ms
- torch.compile: 16 GB, 143 ms
- Liger Kernels: 1.47 GB, 304 ms (省内存但慢一倍, chunking overhead)
- **CCE: 1.16 GB, 145 ms**

1.16 GB 是 lower bound, 因为 ∇C (classifier 的梯度) 本身就是 `D × |V| × 2 bytes = 2304 × 256K × 2 ≈ 1.18 GB`, 这是必须存的, 算不掉。

速度跟 torch.compile 几乎一样, memory 差了 14000 倍。

实际训练场景, 16 卡 H100 上, batch size 提升幅度: Gemma 2 2B 是 9.5×, Llama 3 8B 是 3×, GPT 2 是 11.9×。vocab 越大收益越大。

训练曲线跟 torch.compile 完全重合, fine-tuning 和 pretraining 都验证过了。

---

## 一个 pretraining 的小坑

fine-tuning 直接用 CCE 就行, 但是 pretraining 有两个数值问题:

1. **gradient filtering on ∇C 的副作用**: rare token 的 average logit 一直低, 每次都被 filter 掉, classifier 那一列永远不更新, 越来越差。Pretraining 要覆盖整个 vocab, 这个不能忍。

2. **bf16 累加误差**: CCE 在 HBM 里累加梯度, 用 bf16 是最快但精度差。Pretraining 几十亿 token 累下来, 误差会累积。

解决方案叫 **CCE-Kahan-FullC**:
- Kahan summation: 经典浮点误差补偿算法, 每次加法记住被 truncate 的低位, 下次补回来, 误差从 O(nε) 降到 O(ε)
- FullC: 对 ∇C 关掉 gradient filtering, 保证 rare token 有梯度; 对 ∇E 保留 filtering, 因为 E 方向 rare token 的影响小, 可以省

memory 从 1.16 GB 涨到 2.33 GB, 速度慢一些, 但仍然远好于 baseline。pretraining 实测曲线跟 torch.compile 完全一致。

---

## 一句话总结

Cross-entropy loss 层是当代大 vocab LLM 训练的显存头号杀手, 但其实它的数学结构告诉我们根本不需要 materialize 那个大 logits 矩阵. 借鉴 FlashAttention 的 SRAM tiling, 再加上 softmax 在大 vocab 上天然 99.98% sparse 这个数值事实, 可以把 24 GB 的显存压到 1 MB, 速度还不掉。这是 systems-for-ML 工作的好范本 —— 没发明新算法, 把已有的算术结构、kernel 技巧、数值事实三者拼起来, 解决一个具体且重要的瓶颈。

---

# Cut Cross-Entropy (CCE): 深入解析

Andrej, 这篇 paper 我觉得非常 elegant, 核心思想跟 FlashAttention 在 spirit 上是孪生兄弟——都是把 N² 量级的中间矩阵从 HBM "藏" 到 SRAM 里。但是 cross-entropy 的特殊性使得这里还可以再做一件事 FlashAttention 做不到的: **大量元素本就数值上 negligible, 直接 skip 掉**。让我把数学、kernel 设计、和实测数据都摊开讲。

---

## 1. 问题本质: 为什么 cross-entropy 是当代 LLM 训练的 memory hog

LLM 训练过去几年 sequential 地解决了一系列 memory bottleneck:

1. **Optimizer states** → ZeRO (Rajbhandari et al., 2020): sharding weights/grads/Adam states
   https://arxiv.org/abs/1910.02054
2. **Activations** → Activation checkpointing (Chen et al., 2016): 用 recompute 换 memory
   https://arxiv.org/abs/1604.06174
3. **Attention matrix** → FlashAttention (Dao et al., 2022): O(N²) → O(N) HBM
   https://arxiv.org/abs/2205.14135
4. **Cross-entropy logits** → 本 paper

memory 占比实测 (Table A4, Fig. 1a):
| Model | |V| | logits 占总训练 memory |
|---|---|---|
| Phi 3.5 Mini | 32K | 40% |
| Llama 3 8B | 128K | 65% |
| Gemma 2 2B | 256K | **89%** |

为什么 cross-entropy 占这么多? 因为 logits tensor 的 shape 是 `[batch_tokens, vocab_size]`, 在 teacher forcing 下整个 sequence 同时算 loss, memory = `O(N × |V|)`. 例如 Gemma 2 (2B), `N=8192, |V|=256128, D=2304`:
- logits in fp32: `8192 × 256128 × 4 bytes = 8.4 GB`
- logits in bf16 (PyTorch 会留一份): `4.2 GB`
- 加上 softcap 后的 copy (Gemma 2 用了 logit softcap): 又一份
- gradients 同尺寸: `12 GB`+
- 加起来 baseline 实测 **28 GB** (Table 1, row 5)

Gemma 2 (2B) 在 H100 (80GB) 上, 单条 8192 token 的 sequence 光 loss layer 就把 GPU 吃满。这显然不 scale。

vocabulary 还在变更大: Tao et al., 2024 的 scaling law 工作说大模型应该配大 vocab:
https://arxiv.org/abs/2407.13623

所以 cross-entropy 一定会越来越占大头, 必须从 root 解掉。

---

## 2. 关键 Insight: Cross-entropy 只依赖 ground-truth 的 log probability

这是整个 paper 的 foundational observation。我们看 loss 的精确表达式。

给定:
- Backbone $f(\cdot): \mathbb{R}^{?} \to \mathbb{R}^D$, 输出 hidden embedding
- Classifier $\mathbf{C} \in \mathbb{R}^{D \times |V|}$, 把 D-dim hidden 投到 vocab dim
- Vocabulary $V$, 第 $i$ 个 token 的 ground-truth 标签 $x_i \in V$
- Logits $= \mathbf{C}^\top E_i \in \mathbb{R}^{|V|}$, 其中 $E_i = f(x_1...x_{i-1})$

Token $i$ 的 cross-entropy loss:
$$
\ell_i(\mathbf{x}) = \log \text{softmax}_{x_i}(\mathbf{C}^\top E_i) = C_{x_i}^\top E_i - \log \sum_{j \in V} \exp(C_j^\top E_i)
$$

变量含义:
- $C_{x_i} \in \mathbb{R}^D$: classifier 的第 $x_i$ 列, 对应 ground-truth token 的 "embedding vector"
- $C_j \in \mathbb{R}^D$: classifier 的第 $j$ 列
- $\log \sum_j \exp(\cdot)$ 是 numerically stable 的 log-sum-exp (LSE)

注意看上式: **整个 cross-entropy 只需要一个 scalar $C_{x_i}^\top E_i$ (ground-truth 那个 logit) 和一个对所有 vocab 的 log-sum-exp**. 

第二个观察: LSE 项 $\log \sum_j \exp(C_j^\top E_i)$ 完全独立于 ground-truth label $x_i$. 它只是 normalization 项 (partition function 的 log)。

把 batch 维度 (N 个 token 同时算) 写出来:
$$
\ell = \underbrace{(\mathbf{C}^\top \mathbb{E})_\mathbf{x}}_{\text{indexed matmul, 输出 }\mathbb{R}^N} - \underbrace{\log \sum_j \exp(C_j^\top \mathbb{E})}_{\text{linear-LSE, 输出 }\mathbb{R}^N}
$$

其中:
- $\mathbb{E} = [E_1, ..., E_N] \in \mathbb{R}^{D \times N}$
- $(\mathbf{C}^\top \mathbb{E})_\mathbf{x} \in \mathbb{R}^N$, 第 $i$ 个 entry 是 $C_{x_i}^\top E_i$ —— 只取 ground-truth 那一列

关键: **这两项都不需要把完整的 $\mathbf{C}^\top \mathbb{E} \in \mathbb{R}^{|V| \times N}$ 矩阵 materialize 出来**. 第一项只需要 gather $N$ 列 classifier (而不是 $|V| \times N$), 第二项可以流式累加 LSE, 一边算 logit 一边 online reduce。

直觉上, 这跟 FlashAttention 的 online softmax 是同一招:
https://arxiv.org/abs/2205.14135

但是 cross-entropy 比 attention 还要好处理, 因为 reduction 是 1D 的 (over vocab), 不像 attention 还要按 row 累加 softmax 同时还要存 row max。

---

## 3. Algorithm 1: Memory-efficient Indexed MatMul

第一项 $(\mathbf{C}^\top \mathbb{E})_\mathbf{x}$ 算 ground-truth logits。

**Naive 方案**: 算全 logits $\mathbf{C}^\top \mathbb{E}$, 然后 gather 第 $x_i$ 行 → `O(N × |V|)` memory, 等于没省。
**稍微聪明点**: 先 gather $\mathbf{C}_\mathbf{x} = [C_{x_1}, ..., C_{x_N}] \in \mathbb{R}^{D \times N}$, 再 batch dot product → `O(N × D)` memory, 还行但还要一次 gather + 一次 matmul。
**CCE**: 把 gather 和 dot-product fuse 进一个 Triton kernel, 全程只在 SRAM。

伪代码 (Algorithm 1):
```
Inputs: E ∈ R^{D×N}, C ∈ R^{D×|V|}, x ∈ R^N
Block sizes N_B, D_B
Output: o = (C^T E)_x ∈ R^N

for each E_n (size D × N_B), x_n (size N_B):
    o_n = 0 in SRAM                    # 累加器
    for each E_{n,d} (size D_B × N_B):
        c = C[x_n, d]                   # indexed load, size D_B × N_B
        o_n += E_{n,d} · c              # 列方向 dot product
    write o_n to global memory
```

访问 pattern:
- $E_{n,d}$ 顺序读, L2 cache-friendly
- $\mathbf{C}_{\mathbf{x}_n, d}$ 是 gather (indexed load), 因为 $x_i$ 顺序由数据决定, 是 random access pattern, 但是量小 ($D_B \times N_B$), 能放进 SRAM
- 输出 $\mathbf{o}_n$ 大小 $N_B$ 个 scalar, 写一次 HBM

总 memory: 输出 buffer 是 $O(N)$, 中间全在 SRAM。**完全 bypass logits matrix materialization**.

---

## 4. Algorithm 2: Memory-efficient Linear-LogSumExp (Forward)

第二项 $\text{LSE}_n = \log \sum_j \exp(C_j^\top E_n)$ 是难点。需要 reduce over 整个 vocab。

直接 serial 实现: 三层 for 循环
- 最内层: dot product $C_v^\top E_n$
- 中层: 沿 vocab 累加 LSE
- 外层: 遍历 batch

serial 完全利用不到 GPU. 要 GPU-friendly 必须 tile。

### Tiling 策略

把输出 $\mathbf{O} = \mathbf{C}^\top \mathbb{E} \in \mathbb{R}^{|V| \times N}$ 切成 $V_B \times N_B$ 块。每个 CUDA block 负责:
1. 从 HBM 加载 $\mathbb{E}_n$ (size $D \times N_B$) 和 $\mathbf{C}_v$ (size $D \times V_B$) 
2. 在 SRAM 里算 $A_{nv} = \mathbf{C}_v^\top \mathbb{E}_n$ (大小 $V_B \times N_B$), 内部还分 $D_B$ 子块累加
3. 在 SRAM 里算 block-local LSE: $\text{LSE}_{nv} = \log\sum \exp(A_{nv}^\top)$ (沿 $V_B$ 维)
4. 把 block-local LSE **原子地** merge 到 global LSE buffer

```
LSE = -∞ in HBM (size N)
for each (E_n, C_v) block pair:
    A_nv = 0 in SRAM                   # V_B × N_B
    for each (E_{n,d}, C_{v,d}) sub-block:
        A_nv += C_{v,d}^T · E_{n,d}
    LSE_nv = logsumexp(A_nv^T)          # block-local, stable
    # Spin-lock on atomic for thread-safe log-add-exp:
    LSE_n = log(exp(LSE_n) + exp(LSE_nv))
```

**关键 trick: thread-safe log-add-exp**. 不同 CUDA block 都要更新同一个 $\text{LSE}_n$, paper 用了 spin-lock on atomic operation (在 Triton 里好写), 或者可以 atomic compare-and-swap loop (CUDA 里更优)。

numerically stable log-add-exp 公式:
$$
\text{logaddexp}(a, b) = \max(a,b) + \log(1 + \exp(-|a-b|))
$$
当 $|a-b|$ 大时, $\exp(-|a-b|) \to 0$, $\text{logaddexp} \approx \max(a,b)$ —— 正确行为。

这个跟 FlashAttention 的 online softmax 在 structure 上完全一致, 区别只在 reduction 维度 (这里是 vocab, attention 是 seq) 和 atomic 的同步方式 (FlashAttention 在一个 kernel 内 single-pass, CCE 跨 CUDA block 需要 global atomic)。

> 关于 online softmax 的原始思路: Milakov & Gimelshein, 2018: https://arxiv.org/abs/1805.02867

---

## 5. Algorithm 3: Memory-efficient Linear-LSE Backward (重点中的重点)

forward 容易, backward 是真正的工程难点。需要算:
$$
\nabla \mathbb{E}^\top = (\mathbf{S} \cdot \nabla \text{LSE}) \mathbf{C}, \quad \nabla \mathbf{C}^\top = (\mathbf{S} \cdot \nabla \text{LSE})^\top \mathbb{E}
$$

变量:
- $\mathbf{S} = \text{softmax}(\mathbf{C}^\top \mathbb{E}) = \exp(\mathbf{C}^\top \mathbb{E} - \text{LSE}) \in \mathbb{R}^{|V| \times N}$ —— 完整 softmax 矩阵, 不可能 materialize!
- $\nabla \text{LSE} \in \mathbb{R}^N$ —— 从 downstream 回传的梯度 (标量对每个 token)
- $\hat{\mathbf{S}} = \mathbf{S} \cdot \nabla \text{LSE}$ (elementwise, 沿 N 方向 broadcast)

注意 $\mathbf{S}$ 不需要 normalize 再算, 因为 $\text{LSE}$ 已经把 normalizer 烧进去了: $\mathbf{S} = \exp(\text{logits} - \text{LSE})$ 直接是 softmax probability. forward 已经算过 $\text{LSE}$, 这里 reuse。

### 5.1 朴素 backward 的瓶颈

backward 是两次 matmul:
1. $\nabla \mathbb{E}^\top = \hat{\mathbf{S}} \mathbf{C}$, shape $N \times D$
2. $\nabla \mathbf{C}^\top = \hat{\mathbf{S}}^\top \mathbb{E}$, shape $|V| \times D$

中间要存 $\mathbf{S} \in \mathbb{R}^{|V| \times N}$ —— 又是 `O(N × |V|)` 内存, 跟 forward 的 logits 一样大。要么 materialize, 要么 recompute.

CCE 选择 recompute in SRAM, 跟 forward 一样的 tiling, 但是边算边累加梯度到 global memory。

### 5.2 Gradient Filtering (这篇 paper 的 secret weapon)

这是 CCE 比 FlashAttention "更强" 的地方。FlashAttention 也跳过 padded 部分, 但 CCE 可以跳过**几乎所有 vocab entries**.

**Intuition**:
- softmax over vocab, 每列 sum 到 1
- 在 bfloat16, fraction 是 7 bits
- 任何小于 $\varepsilon = 2^{-12}$ 的值, 在跟一个 ≥ $2^{-5}$ 的值相加时, 会被 mantissa 重对齐时**直接 truncate 掉** (见 Appendix E 详细推导)

为什么 $\varepsilon = 2^{-12}$? 因为 bfloat16 mantissa 7 bits, 当 $|a| < |b| / 2^7$ 时, $a$ 在加到 $b$ 上前要把 exponent align 到 $b$, mantissa shift 后 7 bits 全部 fall off, $a$ 实际上变成 0。我们关心 $b \approx 2^{-5}$ 的 regime (soft cap), 所以 $a < 2^{-12}$ 一定被吃掉。

**Sparsity 数据**:
- 理论上限: 每列最多 $1/\varepsilon = 4096$ 个 non-trivial entries (因为 sum=1, max 单元素 ≤ 1, 所以最多 $1/\varepsilon$ 个元素贡献 ≥ $\varepsilon$)
- 实测: frontier models 里 **<0.02%** 的 entries non-zero
- Fig. 3 显示在 top ~50 个 most-likely token 后概率就掉到 $2^{-12}$ 以下
- 这是因为 token frequency 是 Zipfian: https://en.wikipedia.org/wiki/Zipf%27s_law
- top tokens 占了 99% 概率质量

**Implementation**: 在 backward 的 inner loop, 每算完一个 block 的 $\mathbf{S}_{nv}$, 检查 `all(S_nv < ε)`, 若是则跳过该 block 的两次梯度更新:

```
S_nv = exp(A_nv - LSE_n)              # compute softmax in SRAM
if all(|S_nv| < ε):
    skip                              # 整个 block 跳过
else:
    ∇E_{n,d} += (S_nv · ∇LSE_n) C_{v,d}
    ∇C_{v,d} += (S_nv · ∇LSE_n)^T E_{n,d}
```

### 5.3 Vocabulary Sorting (让 block-level sparsity 更稠密)

gradient filtering 的效率取决于 **block 级** sparsity: 如果 non-trivial 元素散落在不同 block 里, 每个 block 都得算; 如果挤在一起, 就能整块跳过。

CCE 的 heuristic:
- Forward pass 时, 用 atomic add 累加每个 token $v$ 的 average logit
- Backward 时, 把 vocab 按 average logit 排序
- 这样"热门" tokens (高 average logit) 会聚在一起, 对应的 block 更可能整块"全活"
- "冷门" tokens 聚在另一头, 整块全 0, 直接 skip

临时 buffer 大小 `O(|V|)`: 256K × 4 bytes = 1 MB, 完全 negligible.

为什么这个排序 safe? 因为 softmax 是 permutation invariant 的 —— 把 vocab 重新 label 不改变 loss/gradient 数值, 只是 reorder classifier $\mathbf{C}$ 的列顺序, $\mathbf{C}$ 仍然是同一个线性映射 (按列重新排列)。

实测 (Table 1 row 6 vs row 1): 没 vocab sorting 慢 15% (23 ms).
没 gradient filtering 慢 3.4× (356 ms)。

---

## 6. Algorithm 4: 合并 indexed-matmul 和 linear-LSE 的 backward

paper 提到实际实现里, 把第一项 (indexed matmul 的 backward) 和第二项 (linear-LSE 的 backward) 合并了, 因为它们 share 访问 pattern.

完整 cross-entropy 对 logits 的梯度 (含 indicator):
$$
\mathbf{G}_{nv} = [[\mathbf{v}_v = \mathbf{x}_n^\top]] - \mathbf{S}_{nv}
$$

变量:
- $\mathbf{v}_v$: vocab index, 当前 block 处理的 token range
- $\mathbf{x}_n$: 当前 block 的 ground-truth labels
- $[[\cdot]]$: indicator matrix, 第 $(i, j)$ 项 = 1 if $a_j = b_i$ else 0

这是 cross-entropy 的标准梯度: $\nabla_{\text{logits}} \ell = \mathbf{y}_{onehot} - \mathbf{p}_{softmax}$。

合并后的 Algorithm 4 在 SRAM 里就把这两部分加好, 不用分别 forward/backward 两次。

---

## 7. 实验数据深度剖析

### Table 1 (Gemma 2 2B 设置, N=8192, |V|=256K, D=2304, A100 80GB)

| Method | Loss mem | Loss time | Grad mem | Grad time | L+G mem | L+G time |
|---|---|---|---|---|---|---|
| **Lower bound** | 0.004 MB | – | 1161 MB | – | 1161 MB | – |
| **CCE (ours)** | **1 MB** | 46 ms | 1163 MB | 100 ms | **1164 MB** | 145 ms |
| Liger Kernels | 1474 MB | 304 ms | – | – | 1474 MB | 304 ms |
| Torch Tune (8 chunks) | 8000 MB | 55 ms | 1630 MB | 115 ms | 9631 MB | 169 ms |
| torch.compile | 4000 MB | 49 ms | 12000 MB | 92 ms | 16000 MB | 143 ms |
| Baseline (PyTorch) | 24000 MB | 82 ms | 16000 MB | 122 ms | 28000 MB | 208 ms |
| CCE (no vocab sort) | 0.09 MB | 45 ms | 1162 MB | 115 ms | 1162 MB | 159 ms |
| CCE (no grad filter) | 0.09 MB | 45 ms | 1163 MB | 314 ms | 1162 MB | 357 ms |
| CCE-Kahan | 1 MB | 47 ms | 2325 MB | 114 ms | 2326 MB | 160 ms |
| CCE-Kahan-FullC | 1 MB | 47 ms | 2326 MB | 268 ms | 2326 MB | 313 ms |
| CCE-Kahan-FullE | 1 MB | 47 ms | 2326 MB | 247 ms | 2326 MB | 292 ms |

**关键观察**:

1. **Memory**: CCE 从 28 GB → 1.16 MB, ~24000× 减少. Lower bound 是 ∇E + ∇C 不可消除 (1.16 GB), CCE 只多了 3 MB.

2. **Speed**: CCE 比 torch.compile 略快 (145 ms vs 143 ms, 6% 差距在边界). 比 Baseline 快 30%.

3. **Liger Kernels**: memory 也省 (1.47 GB), 但 latency 翻倍 (304 ms). 因为 chunked 实现 kernel launch overhead 大。

4. **Gradient filtering 是关键**: 没它 (row 7) 速度 357 ms vs 145 ms, 慢 2.5×. Vocab sorting 也重要但只 10% 影响.

5. **Kahan summation**: 用 Kahan 算法补偿 bfloat16 累加误差, memory 翻倍 (1.16 MB → 2.33 MB), 速度慢 10%. 这是 pretraining 必需的.

### Kahan summation 是什么

经典 floating point 误差补偿算法 (Kahan 1965):
https://en.wikipedia.org/wiki/Kahan_summation_algorithm

```python
def kahan_sum(items):
    s = 0.0
    c = 0.0  # compensation
    for x in items:
        y = x - c
        t = s + y
        c = (t - s) - y   # recovers lost low-order bits
        s = t
    return s
```

每次加法, $c$ 记住被 truncate 掉的"低位", 下次加进来. 误差从 $O(n \epsilon)$ 降到 $O(\epsilon)$ (independent of $n$)。

### 为什么 pretraining 需要 Kahan + FullC?

paper Section 5.3 给了两个理由:

1. **Gradient filtering on ∇C 的副作用**: 如果一个 token $v$ 在整个 training set 里很少出现 (long-tail), 它的 average logit 一直很低, 每次都被 filter 掉, 于是 $C_v$ 永远不更新, 越来越差。Fine-tune 时数据集小可以容忍, pretraining 时 (要覆盖整个 vocab 的) 必须保证 $C$ 上所有列有梯度。

2. **Global memory 中的 bf16 summation**: CCE 在 HBM 里累加梯度, 用目标 dtype (bf16) 是最快但精度差。Pretraining 几十亿 token 累加, 误差会显著累积。

**CCE-Kahan-FullC**: 只对 $\nabla \mathbf{C}$ 取消 gradient filtering (full precision over C), 对 $\nabla \mathbb{E}$ 仍 filter. 这样 rare token 的 classifier embedding 也有梯度, 但 E 方向仍可省 compute。

### Table A4: 各模型的 max batch size 提升

16 × H100 80GB, FSDP + activation checkpointing + bf16 AdamW:

| Model | |V| | Max batch before | Max batch after | 提升 |
|---|---|---|---|---|
| GPT 2 | 50K | 5.87M | 69.8M | **11.9×** |
| Gemma 2B | 256K | 1.16M | 17.2M | **14.9×** |
| Gemma 2 2B | 256K | 1.11M | 10.6M | **9.5×** |
| Llama 2 7B | 32K | 3.16M | 4.71M | 1.5× |
| Llama 3 8B | 128K | 1.58M | 4.67M | 3.0× |
| Llama 3 70B | 128K | 0.40M | 0.55M | 1.4× |
| Qwen 1.5 7B | 152K | 1.41M | 4.68M | 3.3× |

vocab 越大, 收益越大 (因为 logits memory 占比更高). Llama 2 13B 只有 1.3×, 因为它 vocab 只有 32K, loss layer 占比小.

### Fig. 4 / Fig. 5 训练曲线

- Fine-tuning (Alpaca): CCE vs torch.compile 曲线 indistinguishable
- Pretraining (5% OpenWebText): CCE-Kahan-FullC vs torch.compile validation perplexity 曲线完全重合

说明: gradient filtering 在数值上"丢弃"的部分, 在统计上也确实不影响学习。

---

## 8. Appendix B 的 bonus: 跳过 ignored tokens

训练时很多 token 不参与 loss (padding, system prompt, user input). 大多数实现是先算 logits 再 mask zero out. CCE 在 forward 前直接 filter 掉这些 token, 节省 compute.

Table A1 给出数据: Gemma 2 2B, Liger Kernels 不变 (300ms+, 因为 chunking bound on launch overhead), 其他方法普遍快 2-3×. Baseline 208 ms → 75 ms. CCE 145 ms → 54 ms.

这是 simple but easy to miss 的优化, 配合 CCE 特别有效 (因为 CCE 已经 minimal memory, filter 掉 token 直接成比例省 compute).

---

## 9. 跟其他方法的 positioning

### vs FlashAttention

| 维度 | FlashAttention | CCE |
|---|---|---|
| 处理的矩阵 | Attention N×N | Logits N×|V| |
| Reduction | row-wise softmax + matmul | column-wise LSE + matmul |
| 同步 | 单 kernel 内 single-pass | cross-block atomic spin-lock |
| Filter 跳过 | padded tokens | ~99.98% of vocab entries |
| Memory saving | O(N²) → O(N) | O(N×|V|) → O(N + |V|) |

两者都是 IO-aware tiling + recompute in SRAM 范式, 但 CCE 多利用了 softmax 在大 vocab 上的 extreme sparsity, 这是 attention 没有的特性。

FlashAttention 原文: https://arxiv.org/abs/2205.14135

### vs Liger Kernels

Liger (LinkedIn, Hsu et al., 2024): https://github.com/linkedin/Liger-Kernel

也做 memory-efficient cross-entropy, 用 chunking + fused loss+gradient. 优势是简单纯 Triton, 劣势是 chunked 实现 kernel launch 多, latency 翻倍。且 chunking 仍然 O(N×D) memory, 不如 CCE 的 O(N + |V|).

### vs Torch Tune chunked CE

PyTorch 官方 torchtune: https://github.com/pytorch/torchtune

8 chunks 把 vocab 切成 8 段, 每段算一次. Memory 9.6 GB, 还是不小. 而且要写 user-defined masking 必须改 kernel (因为 fuse 一起了).

CCE 把 forward/backward 分开, user-defined transform on loss (label smoothing, weighting, etc.) 可以在外面正常写 PyTorch.

### vs Hierarchical Softmax (Grave et al., 2017)

经典做法: 把 vocab 组织成树, 每次只算一条 path. Memory O(log |V|).

https://arxiv.org/abs/1609.04309

代价: 改变了 classifier 结构, 不是 dense linear head, 训练精度会变. CCE 保持 dense classifier 数学完全等价, 不损失任何精度。

---

## 10. 工程层面的细节讨论

### 10.1 Triton vs CUDA

paper 用 Triton 实现 (Tillet et al., 2019): https://arxiv.org/abs/1909.07929

Triton 限制: control flow 只能在 block 级别, 所以 gradient filter 和 log-add-exp 也只能 block-level. 如果用 raw CUDA, 可以做 thread-level control flow, 更细粒度的 filter, 性能可能进一步提升。

### 10.2 Pipeline parallelism 的间接收益

大模型训练常用 pipeline parallelism (GPipe, Megatron): https://arxiv.org/abs/1811.06965

Pipeline stage 之间要平衡 compute/memory ratio. 之前 classifier head 的 memory-to-compute ratio 极高 (28 GB memory, 仅 ~200 ms compute), 是 imbalance 的 outlier, pipeline stage 切分很别扭. CCE 把它降到 1 GB, 让 classifier head 的 ratio 跟 transformer block 接近, 利于更均衡的 pipeline 切分, 可能减少 stage 数量。

### 10.3 推广到其他 classification

paper Section 6 提到可以推广到:
- Image classification with huge #classes (e.g., JFT-300M, iNaturalist)
- Contrastive learning (CLIP-like): 每个 batch 的 contrastive matrix 也是 [batch, batch]
- Recommendation systems (item vocab millions)

数学结构都是 `matmul + softmax/LSE + matmul`, 都能套 CCE.

---

## 11. 公式与代码直觉梳理

把整套数学串起来, 一图概括:

```
Forward:
  E = f(x)                              # backbone, shape (D, N)
  
  Part 1: ground-truth logits
    o_i = C_{x_i}^T E_i                 # Algorithm 1, only N scalar outputs
  
  Part 2: log-sum-exp
    LSE_n = log Σ_v exp(C_v^T E_n)      # Algorithm 2, online over vocab
    (uses atomic log-add-exp across CUDA blocks)
  
  Loss: ℓ = o - LSE                     # elementwise, shape N

Backward:
  Given ∇ℓ (upstream), compute:
  
  Part A: gradient wrt logits
    G_vn = [[v == x_n]] - exp(C_v^T E_n - LSE_n)    # in-SRAM
    (gradient filter: skip blocks where all |G| < ε = 2^{-12})
  
  Part B: gradients wrt E, C
    ∇E_n = Σ_v (G_vn · ∇ℓ_n) C_v        # matmul, accumulated in HBM
    ∇C_v = Σ_n (G_vn · ∇ℓ_n)^T E_n      # matmul, accumulated in HBM
  
  (Algorithm 4 fuses Part A + Part B in single backward kernel)
```

内存分析 (Gemma 2 2B, N=8192, |V|=256K, D=2304, bf16):
- Baseline logits: `8192 × 256128 × 2 = 4 GB` (bf16) + fp32 copy + grad copy ≈ 28 GB
- CCE: 几个 MB 的 LSE buffer + 1 MB 的 vocab-sort buffer + ∇E (37 MB) + ∇C (1.1 GB) ≈ 1.16 GB

∇C 的 1.1 GB 不可消除 (lower bound), 因为它是最终梯度, 必须存。`|V| × D × 2 bytes = 256128 × 2304 × 2 ≈ 1.18 GB`, 跟实验数据 1.16 GB 吻合.

---

## 12. 我的几点延伸思考

### (a) 数值精度问题更深的角度

gradient filtering 用 $\varepsilon = 2^{-12}$ 是 "在 bfloat16 一定被 round 掉" 的下界。如果模型用 fp16 (10 bit mantissa), 阈值应该更高 (paper 没讨论)。如果用 fp32 训练, gradient filtering 完全不能开, 因为 fp32 mantissa 23 bits, 有效精度 $\sim 2^{-23}$, 这时只有最 top 的几个 token 能跳过, filtering 收益微乎其微。

这其实暗示了一件事: **CCE 的 efficiency 跟训练用的数值精度强耦合**. 未来如果出现 fp8 training (NVIDIA Hopper/Blackwell 支持), filtering 会更激进 (fp8 mantissa 只有 3-4 bits, 阈值要提到 $\sim 2^{-4}$, 能跳过更多). 这是个有意思的 scaling dimension。

fp8 训练相关工作: https://arxiv.org/abs/2305.14314

### (b) 跟 mixture-of-experts 的联系

MoE (Mixtral, DeepSeek-MoE) 在 router 上做 sparse top-k. CCE 在 classifier 上做 implicit sparsity via filtering. 都是利用 softmax 的 sparsity。

未来 MoE router 的训练 loss 也可能用类似 CCE 的 trick: https://arxiv.org/abs/2101.03961

### (c) 是否能 batch dimension 也做 filtering?

paper 只 filter vocab 维度. 但在 LM 训练, 很多 token 是 "easy" (高 confidence, 大量梯度也 negligible). 理论上也能 filter. 不过 backward 的 batch 维 reduction 通常已经 GPU-saturating, 收益小。

### (d) Vocabulary sorting 的更好方案

paper 用 average logit 排序, 但这是 token-frequency 的代理。更准的代理:
- 训练数据直接统计 token frequency (cheap)
- 按 frequency 排序
- 但要注意, 这要求 vocab 顺序固定, 跟 BPE 的 token id 解耦. Gemma 2 等大 vocab 模型通常 BPE id 跟 frequency 已经有 weak correlation (高频 token 优先 merge, id 较小), 所以 paper 用 average logit 的 reordering 收益 15%, 而用 frequency 可能 25%+.

### (e) 对 inference 的影响

paper 完全没提 inference, 因为 inference 时只算 1 个 token 的 logits, memory `O(|V|)`, 已经不是 bottleneck. 但是 CCE 的 indexed matmul kernel 可能对 inference 也有用 (减少 KV cache 之后的 logits 计算 memory). 这是 open question.

vLLM 等推理框架: https://arxiv.org/abs/2309.06180

### (f) 跟 logit softcap 的交互

Gemma 2 用了 logit softcap (logits = tanh(logits/softcap) × softcap):
https://arxiv.org/abs/2408.00118

softcap 让 max logit 限制在 ±softcap (Gemma 2 用 30). 这意味着 LSE 的 max 项 bounded, 数值更稳。但同时也意味着 softmax 分布更"平", gradient filtering 的阈值可能要重算。paper 把 softcap integrated 在 CCE 里 (Appendix C.1 显示 softcap 在 CCE 里只占 4.4% backward time).

---

## 13. 实用建议 (for practitioners)

- **Fine-tuning**: 直接用 CCE (默认配置), memory 省一大笔, 速度等价 torch.compile, 训练曲线无差。
- **Pretraining**: 用 CCE-Kahan-FullC, 保证 rare token 的 classifier 有梯度, Kahan 补 bf16 累加误差。
- **Vocab 大 (>128K)**: 必须用 CCE, 否则 logits 一层就吃满 GPU。
- **Vocab 小 (<32K)**: CCE 收益小 (~1.3×), Liger Kernels 或 torch.compile 可能更省事。
- **Pipeline parallelism**: 用 CCE 后 classifier head 可以单独成一个 stage 而不 imbalance, 简化切分。
- **代码**: https://github.com/apple/ml-cross-entropy

---

## 14. 总结: intuition 链条

1. Cross-entropy = (ground-truth logit) - (log-sum-exp over vocab)
2. 两项都不需要 materialize 整个 logits 矩阵, 都可以 streaming 算
3. FlashAttention 的 online softmax 技术直接搬过来, 但要在 cross-CUDA-block 加 atomic 同步
4. Backward 时 softmax 矩阵 S 在 bf16 下 99.98% 元素低于精度阈值, 整块 skip 不影响梯度
5. Vocab 按 average logit 排序, 让稀疏结构 block-aligned, skip 更高效
6. Pretraining 需要 Kahan summation + 对 ∇C 关闭 filter, 保证 rare token embedding 不死

整个 paper 的 elegance 在于: **不是发明新算法, 而是把已有的算术结构 (softmax decomposition)、已有的 kernel 技巧 (FlashAttention tiling)、已有的数值事实 (bf16 truncation) 三者组合起来, 解决一个具体且重要的工程瓶颈**。这是 systems-for-ML 类工作的范本。

参考阅读:
- FlashAttention: https://arxiv.org/abs/2205.14135
- ZeRO: https://arxiv.org/abs/1910.02054
- Activation checkpointing: https://arxiv.org/abs/1604.06174
- Liger Kernel: https://github.com/linkedin/Liger-Kernel
- Gemma 2: https://arxiv.org/abs/2408.00118
- Vocab scaling laws: https://arxiv.org/abs/2407.13623
- Online softmax: https://arxiv.org/abs/1805.02867
- Triton: https://arxiv.org/abs/1909.07929
- CCE code: https://github.com/apple/ml-cross-entropy
