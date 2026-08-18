---
source_pdf: CUT YOUR LOSSES.pdf
paper_sha256: 19026f0d3f943c7b47e50f27a0c3319aec809ed2f22ad11df0419b67b53947c7
processed_at: '2026-08-18T04:26:42-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说 CCE

---

## 问题一句话

你训 LLM 的时候，**最后一个 layer（classifier head）算 cross-entropy loss 这一步，会把整张 H100 的内存吃光**。

具体多夸张？Gemma 2 2B 那个模型，vocab 256K，单条 80K token 的 sequence，光算个 loss 就要 81GB——**一张 H100 装不下一条 sequence**。

你买了一张 8 万块的 GPU，89% 的内存被一个 `log_softmax` 函数占了，backbone 才用 11%。这荒谬。

---

## 为什么会这样

因为 loss 的计算长这样：

```
logits = E @ C.T    # shape [N_tokens, V]  ← 这玩意儿巨大
loss = cross_entropy(logits, targets)
```

那个 `logits` matrix 是 `[sequence_length × vocab_size]`。sequence 8 万，vocab 25 万，fp32 存，就是 81GB。

但你仔细想想——**你其实只需要 ground-truth 那一个 token 的 logit，加上所有 logit 的 sum 来做 normalization**。最终 loss 就是 $N$ 个标量。中间那个巨大的 matrix 是纯纯的"中间垃圾"，算完就扔。

类比一下：你要知道班里某个学生的排名，不需要把全班成绩单打印出来贴墙上，只需要扫一遍记个 max 和该生的分数就行。

---

## CCE 怎么做的

三步：

### 第一步：数学重构

把 cross-entropy 拆成两部分：

```
loss = (ground-truth 的 logit) - log(sum of exp(所有 logits))
        ↑ 只要 1 个数              ↑ 只要 1 个数（per token）
```

两部分**输出都是 O(N)**——只有 N 个标量。中间那个 O(N×V) 的 logit matrix 是虚的，不需要存下来。

### 第二步：不存中间结果，在 SRAM 里算完

写个 CUDA kernel，把 embedding 和 classifier matrix 分 block 加载到 on-chip SRAM（几十 KB，超快），在 SRAM 里算 dot product、算 log-sum-exp，**只把最终那 N 个标量写回 global memory**。

这就是 FlashAttention 的同款套路——FlashAttention 解决了 attention 的 $N^2$ 中间 matrix，CCE 解决了 loss 的 $N \times V$ 中间 matrix。俩难兄难弟。

### 第三步：聪明的偷懒（gradient filtering）

backward pass 要算 softmax matrix 的 gradient，这玩意儿也是 $N \times V$，也巨。

但 softmax 有个特点：**特别 peaky**。训练好的模型，99.98% 的概率质量集中在 top-50 个 token 上，剩下 25 万个 vocab entry 的概率基本是 0。

更妙的是 bfloat16 只有 7 bit mantissa，太小 的数加到大数上直接被舍 入当 0 处理。所以**那些小到会被舍入的 entry，你算了也白算**。

CCE 直接 skip 掉这些 block 的 gradient 计算——**3.5× 加速，精度零损失**。

再配一个 vocab sorting（把高频 token 排到一起，让 skip 更高效），又省 15%。

---

## 效果

Gemma 2 2B 上：

| | 之前 | 之后 |
|---|---|---|
| Loss 内存 | 24 GB | 1 MB |
| 整个 head 内存 | 28 GB | 1 GB |
| 速度 | 143 ms | 145 ms |
| 收敛 | 正常 | 正常 |

内存降 24000 倍，速度基本不变。白捡的 batch size 提升 10 倍。

---

## 一句话总结

**cross-entropy 那个巨大的 logit matrix 是纯中间垃圾，算完就扔，根本不需要存到显存里。** 在 SRAM 里算完只写结果回去就行。顺便利用 softmax 天然稀疏，把 backward 也偷个懒。

这就是 CCE。本质就是把 FlashAttention 的思想搬到 loss layer 上，再叠加一个"bfloat16 精度有限所以小数不用管"的免费午餐。

---

paper 链接：https://github.com/apple/ml-cross-entropy

---

# Cut Cross-Entropy (CCE) 深度解析

Andrej，这篇 paper 触及了一个我在 nanoGPT 训练中也经常纠结的痛点：**classifier head 的内存爆炸**。让我从直觉到实现细节，逐一拆解。

---

## 1. The Problem: 为什么 cross-entropy 成了 memory hog？

### 1.1 历史背景：所有其他层都被"打掉"了

过去几年 LLM training 的内存优化路线非常清晰：

| 技术 | 解决的问题 | 内存降低 |
|------|-----------|---------|
| Data-parallel + Large batch (Goyal 2017) | 利用更多 GPU | - |
| ZeRO (Rajbhandari 2020) | 参数/梯度/optimizer state 跨 GPU 分片 | O(P) → O(P/N_gpu) |
| Activation checkpointing (Chen 2016) | 深层网络 activation 内存 | O(N) → O(sqrt(N)) |
| FlashAttention (Dao 2022) | Attention 的 O(N²) 中间矩阵 | O(N²) → O(N) |

所有这些优化做完之后，剩下的"最后一个胖子"就是 **cross-entropy loss 的 logit matrix**。这个 matrix 的 shape 是 `[N_tokens, |V|]`，没有任何现有优化能直接打它，因为它就是 loss 计算的输入。

### 1.2 量化感受

paper 给的数据非常直观：

- **Gemma 2 (2B)**: `|V| = 256,128`, 单个 sequence `N = 80,000`
- Logit matrix 内存 = `80,000 × 256,128 × 4 bytes (fp32) ≈ 81 GB`
- **一个 sequence 就把 80GB H100 吃光了**

Figure 1a 显示的 memory breakdown：
- Phi 3.5 Mini: log-probs 占 **40%**
- Llama 3 8B: 占 **65%**
- Gemma 2 2B: 占 **89%**

这就是说：你买了一张 H100，89% 的内存被一个简单的 `log_softmax` 占了。这听起来荒谬，但确实是事实。

---

## 2. The Key Insight: Loss 只需要 ground-truth 的 log-prob

### 2.1 数学 reformulation

标准 cross-entropy loss 对单个 token：

$$\ell_i(\mathbf{x}) = \log \text{softmax}_{x_i}(\mathbf{C}^\top E_i)$$

变量含义：
- $\ell_i$：第 $i$ 个 token 的 loss（标量）
- $\mathbf{x}$：input token sequence
- $x_i$：第 $i$ 个 ground-truth token（是 vocab 中的一个 index，标量）
- $\mathbf{C} \in \mathbb{R}^{D \times |V|}$：classifier matrix（unembedding matrix），$D$ 是 hidden dim，$|V|$ 是 vocab size
- $E_i = f(x_1 \dots x_{i-1}) \in \mathbb{R}^D$：backbone 输出的第 $i$ 个 embedding
- $\mathbf{C}^\top E_i \in \mathbb{R}^{|V|}$：logits vector（所有 vocab entries 的 raw score）
- $\text{softmax}_{x_i}$：取 logits vector 在 $x_i$ 位置的 softmax 值

展开 softmax：

$$\ell_i(\mathbf{x}) = \underbrace{C_{x_i}^\top E_i}_{\text{Term 1: ground-truth logit}} - \underbrace{\log \sum_j \exp(C_j^\top E_i)}_{\text{Term 2: log-sum-exp over vocab}}$$

变量含义：
- $C_{x_i}$：classifier matrix 的第 $x_i$ 列，shape $\mathbb{R}^D$（即 ground-truth token 对应的那一列）
- $C_{x_i}^\top E_i$：ground-truth logit，标量（term 1）
- $C_j$：classifier matrix 的第 $j$ 列
- $\sum_j$：遍历整个 vocab
- Term 2 就是 $\log Z$，即 normalization constant

### 2.2 关键观察

**Term 1** 只需要 $C_{x_i}$ 一列，是 indexed matmul，输出 $N$ 个标量。
**Term 2** 需要遍历整个 vocab，但输出也只有 $N$ 个标量（每个 token 一个 LSE）。

两个 term 的**输出都是 $O(N)$**，但**中间的 logit matrix $\mathbf{C}^\top \mathbf{E}$ 是 $O(N \cdot |V|)$**。

这个 logit matrix 是"中间产物"，不需要 materialize 到 global memory！

### 2.3 Batch 化（teacher forcing）

$$\ell = (\mathbf{C}^\top \mathbf{E})_{\mathbf{x}} - \log \sum_j \exp(C_j^\top \mathbf{E})$$

- $\mathbf{E} = [E_1 \dots E_N] \in \mathbb{R}^{D \times N}$
- $(\mathbf{C}^\top \mathbf{E})_{\mathbf{x}} = [C_{x_1}^\top E_1 \dots C_{x_N}^\top E_N] \in \mathbb{R}^N$
- 第二项是 row-wise log-sum-exp，输出 $\mathbb{R}^N$

---

## 3. 实现：三个 CUDA kernel 的"乐高"

CCE 的核心是把上述 reformulation 落地为三个 Triton kernel，全部在 SRAM 中操作。

### 3.1 Algorithm 1: Indexed MatMul（Term 1, forward）

**输入**：$\mathbb{E} \in \mathbb{R}^{D \times N}$, $\mathbf{C} \in \mathbb{R}^{D \times |V|}$, $\mathbf{x} \in \mathbb{R}^N$（ground-truth indices）
**输出**：$\mathbf{o} = (\mathbf{C}^\top \mathbb{E})_{\mathbf{x}} \in \mathbb{R}^N$（ground-truth logits）

```
for blocks E_n, x_n (block size D × N_B):
    o_n = 0 in SRAM
    for sub-blocks E_{n,d} (block size D_B × N_B):
        c = C_{x_n, d}   # indexed load: 只取 ground-truth 列
        o_n += E_{n,d} · c   # dot product in SRAM
    write o_n to global memory
```

**直觉**：naive 做法是先算 $\mathbf{C}^\top \mathbb{E}$（$O(N|V|)$ 内存）再 indexing；或者先 index $\mathbf{C}_{\mathbf{x}}$（$O(ND)$ 内存）再 matmul。CCE 把 indexing 和 dot-product **fuse** 到一个 kernel 里，只在 SRAM 里持有 $D_B \times N_B$ 的小 block。

### 3.2 Algorithm 2: Linear-LSE Forward（Term 2, forward）

**输入**：$\mathbf{E} \in \mathbb{R}^{D \times N}$, $\mathbf{C} \in \mathbb{R}^{D \times |V|}$
**输出**：$\text{LSE} = \log \sum_j \exp(C_j^\top \mathbf{E}) \in \mathbb{R}^N$

```
LSE = -∞ vector in global memory (size N)
for all block pairs (E_n, C_v) of sizes (D × N_B, D × V_B):
    A_nv = 0 in SRAM (size V_B × N_B)
    for sub-blocks E_{n,d}, C_{v,d}:
        A_nv += C_{v,d}^T · E_{n,d}   # blockwise matmul, 结果在 SRAM
    LSE_nv = logsumexp(A_nv^T)  # numerically stable
    LSE_n = log(exp(LSE_n) + exp(LSE_nv))  # atomic + spin-lock 同步
```

**关键技术点**：

1. **Online log-sum-exp**（类似 FlashAttention 的 online softmax，Milakov & Gimelshein 2018）：每个 CUDA block 算出自己的局部 LSE，再用 atomic operation 合并。数值稳定公式：
   $$\text{LSE}_{new} = \log(\exp(\text{LSE}_{old}) + \exp(\text{LSE}_{local})) = \max(a, b) + \log(1 + \exp(-|a-b|))$$

2. **Spin-lock on atomic**：多个 CUDA blocks 写同一个 LSE 位置（同一个 $n$，不同 $v$ block），用 atomic compare-and-swap 或 spin-lock 同步。paper 说在 Triton 里 spin-lock 简单实现，CUDA 原生可能用 CAS loop 更快。

3. **完全在 SRAM 里 materialize $A_{nv}$**：$V_B \times N_B$ 的 block，不写到 global memory。

### 3.3 Algorithm 3/4: Backward Pass

反向需要计算两个梯度：

$$\nabla \mathbf{E}^\top = (\mathbf{S} \cdot \nabla \text{LSE}) \mathbf{C}, \quad \nabla \mathbf{C}^\top = (\mathbf{S} \cdot \nabla \text{LSE})^\top \mathbf{E}$$

- $\mathbf{S} = \text{softmax}(\mathbf{C}^\top \mathbf{E}) \in \mathbb{R}^{|V| \times N}$：softmax matrix
- $\nabla \text{LSE} \in \mathbb{R}^N$：从上游传回来的 gradient
- $\mathbf{S} \cdot \nabla \text{LSE}$：element-wise 乘（broadcasting），记为 $\hat{\mathbf{S}}$
- 这是两个 matmul：$\hat{\mathbf{S}} \mathbf{C}$ 和 $\hat{\mathbf{S}}^\top \mathbf{E}$

**问题**：$\mathbf{S}$ 也是 $O(N|V|)$，不能 materialize。

**CCE 的做法**：在 SRAM 里重算 $\mathbf{C}^\top \mathbf{E}$，然后 $\mathbf{S} = \exp(\mathbf{C}^\top \mathbf{E} - \text{LSE})$（不需要重算 normalization，因为 LSE 已经有了）。

---

## 4. Gradient Filtering: 利用 softmax 的 sparsity

### 4.1 数值精度的 truncation

这是 paper 最 elegant 的 insight 之一。

bfloat16 的结构：
- 1 bit sign
- 8 bit exponent
- **7 bit mantissa**（fraction）

当两个数 $a, b$ 相加，$|a| < |b|$ 时：
1. 把 $a$ 的 mantissa 对齐到 $b$ 的 exponent
2. 如果 $a$ 的 exponent 比 $b$ 小超过 $2^7 = 128$，则 $a$ 的 mantissa 在对齐时**完全移出精度范围**，$a$ 被当作 0

对于 softmax 输出 $s \in [0, 1]$：
- 如果 $s < \varepsilon = 2^{-12}$，则 $s$ 加到任何 $\geq 2^{-5}$ 的值上都会被 truncation
- 因此每列最多有 $\frac{1}{\varepsilon} = 4096$ 个非零 entry

### 4.2 实际 sparsity

Figure 3 的 log-log plot 显示：
- Top tokens 概率衰减很快
- 到第 ~50 个 token，概率已经掉到 $\varepsilon$ 以下
- Frontier models 实测：**< 0.02% 元素非零**

直觉：softmax 是高度 peaky 的，特别是训练好的模型——大部分概率质量集中在 top-k tokens。这就给了我们"白吃午餐"：**跳过那些无论如何都会被 truncation 的 entry 的 gradient 计算**。

### 4.3 Threshold 选择

$$\varepsilon = 2^{-12}$$

- 选这个值是因为它是 bfloat16 下"不会被 truncation 的最小值"
- 实测：在 backward pass 中跳过 $|G_{nv}| < \varepsilon$ 的 block，带来 **3.5x speedup**，精度无损

### 4.4 Vocabulary Sorting: 让 sparsity 变得"block-friendly"

光有 sparsity 不够——如果非零 entry 散落在 vocab 各处，每个 block 都得算。

**Vocabulary sorting 的思想**：按 token 的 average logit 排序 vocab，让 high-logit tokens 聚集在一起。

- Forward pass 时顺便用 atomic add 累积每个 token 的 average logit
- Backward pass 时按 average logit 分 block
- 高 logit 区：dense block，全部计算
- 低 logit 区：sparse block，整个 skip

代价：一个 $O(|V|)$ 的临时 buffer（~1MB for 256K vocab）。

Table 1 row 1 vs. 6：**没有 vocab sorting 慢 15%**（23ms）。

---

## 5. 实验：内存与速度

### 5.1 主结果表（Table 1, Gemma 2 2B, batch 8192, vocab 256K, D=2304）

| Method | Loss Mem | Loss Time | Grad Mem | Grad Time | Loss+Grad Mem | Loss+Grad Time |
|--------|----------|-----------|----------|-----------|---------------|----------------|
| Lower bound | 0.004 MB | - | 1,161 MB | - | 1,161 MB | - |
| **CCE (Ours)** | **1 MB** | 46 ms | **1,163 MB** | 100 ms | **1,164 MB** | 145 ms |
| Liger Kernels | 1,474 MB | 304 ms | - | - | 1,474 MB | 304 ms |
| Torch Tune (8 chunks) | 8,000 MB | 55 ms | 1,630 MB | 115 ms | 9,631 MB | 169 ms |
| torch.compile | 4,000 MB | 49 ms | 12,000 MB | 92 ms | 16,000 MB | 143 ms |
| Baseline (PyTorch) | 24,000 MB | 82 ms | 16,000 MB | 122 ms | 28,000 MB | 208 ms |

**关键观察**：

1. **内存**：CCE 的 1,164 MB 几乎就是 lower bound（1,161 MB，即 ∇E + ∇C 的输出 buffer）。**24,000× reduction** for loss，**24×** for loss+grad。

2. **速度**：CCE 和 torch.compile 几乎一样快（145 ms vs 143 ms）。看似违反直觉（CCE 要重算 $\mathbf{C}^\top \mathbf{E}$），但 CCE 不写 logits 到 global memory，省下的 memory bandwidth 抵消了重算成本。这正是 FlashAttention 的同款 insight：**compute is cheap, memory traffic is expensive**。

3. **Liger Kernels 的局限**：内存 1,474 MB 不错，但速度 304 ms 慢一倍。因为 chunking 带来 kernel launch overhead，不能像 CCE 那样把整个 vocab dimension 放进 SRAM 流水。

### 5.2 Ablation（Table 1 row 6, 7）

- **No Vocab Sorting**: 159 ms（+10%）
- **No Grad Filter**: 357 ms（+146%）

Gradient filtering 是性能的关键。

### 5.3 Batch size 解锁（Table A4）

| Model | Max Batch Before | Max Batch After | Increase |
|-------|------------------|-----------------|----------|
| GPT 2 | 5.87M | 69.8M | 11.9× |
| Gemma 2 (2B) | 1.11M | 10.6M | 9.5× |
| Gemma 2 (27B) | 739K | 2.53M | 3.4× |
| Llama 3 (70B) | 397K | 552K | 1.4× |
| Llama 2 (13B) | 2.20M | 2.89M | 1.3× |

注意：**模型越大，相对增益越小**。因为大模型的 weights+optimizer+activation 占比更高。但即使 Llama 3 70B，1.4× batch size 也意味着训练时间缩短 ~30%。

### 5.4 训练稳定性（Figure 4, 5）

- **Fine-tuning**：CCE 和 torch.compile 的 loss 曲线几乎重合（Alpaca, 5 seeds）
- **Pretraining**：需要 **CCE-Kahan-FullC**（Kahan summation + 不对 ∇C 做 gradient filtering），才能匹配 torch.compile 的 validation perplexity

为什么 pretraining 需要 Kahan？
- Pretraining 时 bf16 global memory summation 的精度损失会累积
- Kahan summation 用 compensation variable 追踪舍入误差：
  $$y = x + c; \quad c = (x - y) + c$$
  其中 $c$ 是 running compensation

为什么不对 ∇C 做 gradient filtering？
- Pretraining 时，rare tokens 在训练集里出现很少，本来 gradient 就小
- 如果 filter 掉，这些 token 永远学不到（"no gradient propagated to tokens with little support"）
- Fine-tuning 没这个问题，因为模型已经 pretrained 过

---

## 6. 与 FlashAttention 的平行

CCE 的思想几乎是 FlashAttention 的"loss layer 版本"：

| 维度 | FlashAttention | CCE |
|------|----------------|-----|
| 中间产物 | $N \times N$ attention matrix | $N \times |V|$ logit matrix |
| 留在哪里 | SRAM | SRAM |
| Reduction | Online softmax | Online log-sum-exp |
| 数据结构 | Blockwise tiling | Blockwise tiling |
| 同步 | 单 block 内 | Cross-block atomic + spin-lock |
| 加速 trick | - | Gradient filtering + vocab sorting |

可以理解为：FlashAttention 解决了"前向过程中 attention 的 $O(N^2)$"，CCE 解决了"末端 loss 的 $O(N|V|)$"。两者合起来，把 transformer 训练的所有大中间产物都"flash 化"了。

---

## 7. 为什么 Cross-Entropy 之前没人这么做？

这个问题值得深思。我的几个猜测：

1. **历史包袱**：PyTorch 的 `F.cross_entropy` 是 `log_softmax + NLLLoss` 的组合，先 materialize logits 再算 loss。这个 API 设计天然鼓励 materialization。

2. **Attention 的痛苦更早到来**：context length 增长比 vocab size 增长更激进（GPT-3 时代 2K context, 50K vocab；现在 128K context, 256K vocab）。FlashAttention 先被逼出来。

3. **Vocab 的增长是渐进的**：50K → 128K → 256K，每一步都没"突然爆炸"，所以没人专门优化。直到 Gemma 2 把 vocab 推到 256K，问题才尖锐化。

4. **Gradient filtering 这个 insight 很微妙**：需要同时理解 softmax sparsity、bfloat16 浮点结构、blockwise kernel 设计。这种跨 domain 的 insight 罕见。

---

## 8. 实现细节：Triton 的限制

paper 在 Discussion 里提到 Triton 的局限：

- Triton 的 control flow 只能在 **block level** 操作
- Gradient filtering 只能 filter 整个 block，不能 filter 单个元素
- 原生 CUDA 实现可能带来进一步性能提升

这暗示了一个研究方向：**用 CUTLASS 或者直接 CUDA 写 CCE 的精细版本**。Apple 的开源实现是 Triton 的，可能还有 30-50% 的性能空间。

---

## 9. 联想：对 nanoGPT / minGPT 训练的启示

对于你常提到的 nanoGPT 这类小模型训练：

1. **小 vocab（10K-50K）时 CCE 帮助有限**：从 Table A3 看，Phi 3.5 Mini（|V|=32K, D=3072）CCE 比 torch.compile 慢 50%（12ms）。但内存仍然省一个数量级。

2. **真正受益的是 mid-size model + large vocab**：Gemma 2 2B 这种"小模型大 vocab"组合，CCE 让 batch size 提升 10×。这对研究社区特别有价值——单卡 80GB 就能跑起来。

3. **Pretraining 用 CCE-Kahan-FullC**：这是我之前没意识到的细节。如果你用 nanoGPT pretrain 一个 GPT-2 scale 的模型 with 50K+ vocab，bf16 的 gradient summation 精度损失可能确实影响 perplexity。Kahan summation 是个 cheap fix。

4. **Pipeline parallelism 的潜在好处**：paper 提到 classifier head 的 memory-to-compute ratio 异常高，是 pipeline balancing 的痛点。CCE 把这个 ratio 拉平，可能让 PipelineParallel 的 stage 划分更自然。这对 Megatron-LM 风格的训练有直接意义。

---

## 10. 未来方向

paper 自己提的：
- CUDA 原生实现，finer-grained control flow
- 扩展到 image classification / contrastive learning（large class count 场景）

我额外联想的：
- **Mixture-of-Experts 的 routing loss**：MoE 的 expert routing 也是 softmax over experts，expert 数量增长时类似问题
- **Multimodal vocab**：图像 token + 文本 token 混合，vocab 爆炸到 1M+，CCE 直接 applicable
- **Speculative decoding 的 draft model training**：draft model 通常小但 vocab 大，CCE 让单卡训练可行
- **On-device fine-tuning**：iPhone 上的 RAM 比 GPU 更紧张，CCE 可能让 256K vocab 模型在 device 上 fine-tune
- **Connection to sampled softmax**（Jean et al. 2014, https://arxiv.org/abs/1412.2007）：sampled softmax 是另一种解决大 vocab 的方法，但它牺牲精度。CCE 是"精确版"的 sampled softmax——本质上利用了 softmax 的天然 sparsity，等价于 adaptive sampling。

---

## References

- Paper GitHub: https://github.com/apple/ml-cross-entropy
- FlashAttention (Dao et al. 2022): https://arxiv.org/abs/2205.14135
- Online softmax (Milakov & Gimelshein 2018): https://arxiv.org/abs/1805.02867
- ZeRO (Rajbhandari et al. 2020): https://arxiv.org/abs/1910.02054
- Activation checkpointing (Chen et al. 2016): https://arxiv.org/abs/1604.06174
- Liger Kernels: https://github.com/linkedin/Liger-Kernel
- Gemma 2 (Riviere et al. 2024): https://arxiv.org/abs/2408.00118
- Llama 3 (Dubey et al. 2024): https://arxiv.org/abs/2407.21783
- Phi-3 (Abdin et al. 2024): https://arxiv.org/abs/2404.14219
- Triton (Tillet et al. 2019): https://scholar.google.com/scholar?q=Triton+intermediate+language+compiler
- CUTLASS: https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/
- Scaling laws with vocabulary (Tao et al. 2024): https://arxiv.org/abs/2407.13623
- Sampled softmax (Jean et al. 2014): https://arxiv.org/abs/1412.2007
- Kahan summation: https://en.wikipedia.org/wiki/Kahan_summation_algorithm

---

## TL;DR for your intuition

CCE 的核心 insight 三句话：
1. Cross-entropy loss 只需要 ground-truth 的 log-prob 和 log-sum-exp normalization constant，两者都是 $O(N)$ 输出，中间的 $O(N|V|)$ logit matrix 是"phantom memory"。
2. 用 FlashAttention 风格的 blockwise SRAM kernel + online log-sum-exp，把这个 phantom memory 消除。
3. Softmax 天然 sparse（bfloat16 精度下 < 0.02% 非零），用 gradient filtering + vocab sorting 把 backward 的计算量也压下去。

结果是：**Gemma 2 2B 的 loss 内存从 24GB 降到 1MB，训练速度不变，convergence 不变**。这是 LLM training infrastructure 的一个"free lunch"级别的优化。
