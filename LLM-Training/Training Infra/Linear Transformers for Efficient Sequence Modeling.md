---
source_pdf: Linear Transformers for Efficient Sequence Modeling.pdf
paper_sha256: 23cf459dee061797166b23a2aa65623599bf46c6c34423c9e7209c86d062d60f
processed_at: '2026-08-05T15:01:11-07:00'
target_folder: LLM-Training/Training Infra
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej，要把这篇 slides 用“人话”讲清楚，核心就是一句话：**如何把 Attention 压缩成一个固定大小的矩阵，并且让它在 GPU 上跑得飞快，还能学会精准的“擦除旧记忆/写入新记忆”。**

这背后的逻辑链非常漂亮，我把它拆成五个直觉阶段，并附上对应的底层细节。

---

### 1. 核心魔法：去掉 Softmax，Attention 就变成了一个 Matrix RNN

标准的 Softmax Attention 在生成推理时，需要把历史所有的 Key 和 Value 都存下来（KV cache），内存随 sequence length $L$ 线性增长。这在长文本生成时极不友好。

如果我们把 Softmax 拿掉，只留内积：

$$ \mathbf{o}_t = \sum_{j=1}^{t} (\mathbf{q}_t^{\top}\mathbf{k}_j) \mathbf{v}_j $$

利用矩阵乘法的结合律，我们可以把 $\mathbf{q}_t$ 提出来：

$$ \mathbf{o}_t = \mathbf{q}_t^{\top} \underbrace{\left( \sum_{j=1}^{t} \mathbf{k}_j \mathbf{v}_j^{\top} \right)}_{\mathbf{S}_t \in \mathbb{R}^{d \times d}} $$

**直觉**：这里的 $\mathbf{S}_t$ 就是一个 $d \times d$ 的 Matrix-valued hidden state。每个 step 的新记忆只是加上一个 rank-1 矩阵 $\mathbf{k}_t \mathbf{v}_t^{\top}$：

$$ \mathbf{S}_t = \mathbf{S}_{t-1} + \mathbf{k}_t \mathbf{v}_t^{\top} $$

**结果**：推理时的内存开销从 $O(Ld)$ 直接降到了 $O(d^2)$，跟序列长度无关。这就是一个 Linear RNN。

---

### 2. 训练痛点：纯递归无法打满 Tensor Cores

虽然 Recurrent form 推理很省内存，但训练时如果按时间步一步步算，极慢。原因有两点：
1. 严格的时间序列依赖，无法在 sequence 维度做并行。
2. 每一步的操作都是 element-wise add 或 reduction，**全都是非矩阵乘法操作**。现代 GPU (如 H100) 的算力绝大部分在 Tensor Cores 上，只做 element-wise 等于把 GPU 当 CPU 用。

如果把序列切成大小为 $C$ 的 chunk，就能在“全并行”和“全递归”之间找到平衡点：
- **Chunk 内部**：用 $O(C^2 d)$ 的矩阵乘法算 local attention。
- **Chunk 之间**：用 $O(L/C)$ 步递归传递状态 $\mathbf{S}$。

这就是 [FlashLinearAttention](https://github.com/sustcsonglin/flash-linear-attention) 的底层逻辑。它把 local state computation 和 state passing 融合进一个 Triton kernel 里，hidden state $\mathbf{S}$ 尽量留在 SRAM 里，避免 HBM I/O 开销。

---

### 3. Gated Linear Attention (GLA)：让记忆学会“遗忘”

Simple Linear Attention 的致命伤是记忆只增不减。长序列里早期的噪声会永远留在 $\mathbf{S}$ 里，导致 retrieval 任务表现极差。

Mamba / RetNet 加了 scalar decay，GLA 则走得更远：使用 data-dependent 的 per-channel gate：

$$ \mathbf{S}_t = \mathbf{G}_t \odot \mathbf{S}_{t-1} + \mathbf{k}_t \mathbf{v}_t^{\top} $$

其中 $\mathbf{G}_t = \boldsymbol{\alpha}_t^{\top}\mathbf{1}$，而 $\boldsymbol{\alpha}_t = \sigma(\mathbf{x}_t W_{\alpha_1} W_{\alpha_2})^{1/\tau}$。

**公式变量解析**：
- $\boldsymbol{\alpha}_t \in \mathbb{R}^d$：每个 channel 独立的衰减率。
- $\mathbf{1} \in \mathbb{R}^d$：全 1 向量。$\boldsymbol{\alpha}_t^{\top}\mathbf{1}$ 构成了一个 rank-1 的 $d \times d$ 矩阵。
- $\tau$：温度系数，控制 gate 的锐度。
- $W_{\alpha_1}, W_{\alpha_2}$：两层 MLP，让 gate 拥有足够的非线性来学习复杂的遗忘模式。

**直觉**：不同的 feature channel 可能有不同的遗忘需求。有些 channel 记长程依赖（gate 接近 1），有些只看近期（gate 接近 0）。slides 里点出一个深刻洞察：**Mamba, Mamba-2, RWKV-6, GLA 本质上全是同一个公式**，区别仅仅在于 $\mathbf{G}_t$ 怎么参数化。参考 [GLA Paper](https://arxiv.org/abs/2312.06635)。

---

### 4. DeltaNet：解决“覆盖”问题，从加法变成减法

GLA 虽然能遗忘，但它的遗忘是“模糊衰减”。如果输入 `A=1, B=2, A=3`，当我们 query `A` 的时候，我们希望能精准输出 `3`，而不是 `1` 和 `3` 的叠加。

[DeltaNet](https://arxiv.org/abs/2102.11174) 借鉴了 Fast Weight Programmers 的思路：在写入新 value 之前，先用 key 把旧 value 读出来，然后删掉旧的，写上新的。

1. 读旧记忆：$\mathbf{v}_t^{\text{old}} = \mathbf{S}_{t-1} \mathbf{k}_t$
2. 算新值：$\mathbf{v}_t^{\text{new}} = \beta_t \mathbf{v}_t + (1 - \beta_t) \mathbf{v}_t^{\text{old}}$
3. 擦旧写新：$\mathbf{S}_t = \mathbf{S}_{t-1} - \mathbf{v}_t^{\text{old}} \mathbf{k}_t^{\top} + \mathbf{v}_t^{\text{new}} \mathbf{k}_t^{\top}$

把上面的式子合并化简，可以得到一个非常优雅的递推：

$$ \mathbf{S}_t = \mathbf{S}_{t-1} (\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^{\top}) + \beta_t \mathbf{v}_t \mathbf{k}_t^{\top} $$

**公式变量解析**：
- $\mathbf{I} \in \mathbb{R}^{d \times d}$：单位矩阵。
- $\beta_t \in \mathbb{R}$：学习率的标量，控制更新的步幅。
- $(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^{\top})$：这就是经典的 **Householder reflection** 矩阵。

**直觉**：每次更新状态时，先在 $\mathbf{k}_t$ 的方向上做一次反射，把旧的投影抹掉，再投影上新的 $\mathbf{v}_t$。这赋予了模型精准的 in-context recall 能力。

---

### 5. WY Representation：让 Householder 乘法链可并行

DeltaNet 递推公式里的 $\prod (\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^{\top})$ 是一串 Householder 矩阵连乘。如果要一步步乘，训练速度惨不忍睹。

paper 里用了一个 1987 年数值线性代数的老古董：**WY Representation** ([Bischof & Van Loan '87](https://www.cs.cornell.edu/cv/PDFPDF/WYREP.pdf))。这个定理证明：一串 Householder 矩阵的乘积，可以重写为一个单位矩阵减去两个矩阵的乘积：

$$ \mathbf{P}_n = \prod_{t=1}^{n} (\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^{\top}) = \mathbf{I} - \sum_{t=1}^{n} \mathbf{w}_t \mathbf{k}_t^{\top} $$

其中 $\mathbf{w}_t$ 可以用纯线性递推算出来：

$$ \mathbf{w}_n = \beta_n \mathbf{k}_n - \beta_n \sum_{t=1}^{n-1} \mathbf{w}_t (\mathbf{k}_t^{\top} \mathbf{k}_n) $$

**直觉**：原本是沉重的 $d \times d$ 矩阵连乘，现在变成了向量的递推。在这个形式下，$\sum \mathbf{w}_t (\mathbf{k}_t^{\top} \mathbf{k}_n)$ 完全可以用 Tensor Cores 跑矩阵乘法。于是，DeltaNet 顺理成章地套用了 GLA 的 chunkwise parallel form，在单张 H100 上获得了最高 13.7 倍的加速比。

---

### 6. 实验结果与 Hybrid 架构

从实验数据看（1.3B models, 100B tokens）：

| Model | PPL↓ | LM Eval↑ | Retrieval↑ |
|---|---|---|---|
| Transformer++ | 16.9 | 50.9 | 41.8 |
| Mamba | 17.1 | 50.0 | 27.6 |
| GLA | 17.2 | 51.1 | 37.7 |
| DeltaNet | 16.9 | 51.6 | 34.7 |
| **Hybrid (DeltaNet + 2 layers Global Attn)** | 16.6 | 51.8 | **47.9** |

DeltaNet 的 PPL 追平甚至超过了 Transformer++，在 mechanistic recall tasks 上完胜 Mamba 和 GLA。

但纯 Linear Attention 在 global retrieval 上依然略逊于 Softmax Attention。paper 给出的终极答案是 **Hybrid 架构**：大部分层用 DeltaNet 保证线性复杂度和高吞吐，只插 2 层 Global Softmax Attention。这样既保住了极低的推理内存，又把 Retrieval 分数拉到了 47.9，超越了纯 Transformer。

参考链接：
- [Gated Linear Attention (ICML '24)](https://arxiv.org/abs/2312.06635)
- [Parallelizing DeltaNet (arXiv '24)](https://arxiv.org/abs/2406.06484)
- [FlashLinearAttention GitHub Repo](https://github.com/sustcsonglin/flash-linear-attention)

---

# Linear Transformers for Efficient Sequence Modeling 深度解析

Andrej，这份 slides 实际上整合了 Songlin Yang 的两个工作：ICML'24 的 **Gated Linear Attention Transformers with Hardware-Efficient Training** 和 arXiv'24 的 **Parallelizing Linear Transformers with the Delta Rule over Sequence Length**。核心 narrative 是：把 attention 重写成一个 linear RNN with matrix-valued hidden state，然后在 hardware 上让它跑得快、在 expressivity 上让它够强。下面我尽量把每一层拆开讲透。

---

## 1. Attention 的 dual form：parallel vs recurrent

Softmax attention 的标准 parallel form：

$$\mathbf{O} = \mathrm{softmax}\big((\mathbf{Q}\mathbf{K}^{\top}) \odot \mathbf{M}\big)\mathbf{V}$$

变量含义：
- $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{L \times d}$：query / key / value，由 $\mathbf{X} \in \mathbb{R}^{L \times d}$ 经 $W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$ 投影得到，$L$ 是 sequence length，$d$ 是 hidden dim。
- $\mathbf{M} \in \mathbb{R}^{L \times L}$：causal mask，下三角为 1，上三角为 $-\infty$（softmax 后变 0）。
- $\mathbf{A} = \mathrm{softmax}(\mathbf{Q}\mathbf{K}^{\top} \odot \mathbf{M}) \in \mathbb{R}^{L \times L}$：attention matrix。

Cost 分解：
- $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ 投影：$O(Ld^2)$
- $\mathbf{Q}\mathbf{K}^{\top}$：$O(L^2 d)$
- $\mathbf{A}\mathbf{V}$：$O(L^2 d)$

Total $O(L^2 d + Ld^2)$。Sequential steps $O(1)$（完全 parallel），这是 attention 的杀手锏。

Generative inference 的 recurrent form（autoregressive decode）：

$$\mathbf{o}_t = \sum_{j=1}^{t} \frac{\exp(\mathbf{q}_t^{\top}\mathbf{k}_j)}{\sum_{l=1}^{t}\exp(\mathbf{q}_t^{\top}\mathbf{k}_l)} \mathbf{v}_j$$

需要保留 $\{\mathbf{k}_j, \mathbf{v}_j\}_{j=1}^{t}$，即 **KV-cache**，memory $O(Ld)$，每步 compute $O(Ld)$。这是 inference 时 attention 的痛点：序列越长 KV-cache 越大。

| | Training (Parallel) | Inference (Recurrent) |
|---|---|---|
| Compute | $O(L^2)$ | $O(L^2)$ |
| Memory | $O(L)$ | $O(L)$ |
| Steps | $O(1)$ | $O(L)$ |

Reference: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

---

## 2. 从 Softmax 到 Linear Attention：核技巧的极简版

[Katharopoulos et al. '20](https://arxiv.org/abs/2006.16236) 的关键 observation：把 softmax 拿掉，让 similarity 直接是 inner product：

$$\mathbf{O} = \big((\mathbf{Q}\mathbf{K}^{\top}) \odot \mathbf{M}\big)\mathbf{V}$$

Recurrent form：

$$\mathbf{o}_t = \sum_{j=1}^{t} (\mathbf{q}_t^{\top}\mathbf{k}_j)\mathbf{v}_j$$

**这是推导的 magic moment**：因为 $\mathbf{q}_t^{\top}\mathbf{k}_j$ 是 scalar，可以重新组织：

$$\mathbf{o}_t = \mathbf{q}_t^{\top}\underbrace{\left(\sum_{j=1}^{t}\mathbf{k}_j\mathbf{v}_j^{\top}\right)}_{\mathbf{S}_t \in \mathbb{R}^{d \times d}}$$

这里 $\mathbf{S}_t$ 是一个 $d \times d$ 的 matrix-valued hidden state！更新规则：

$$\boxed{\mathbf{S}_t = \mathbf{S}_{t-1} + \mathbf{k}_t\mathbf{v}_t^{\top}, \quad \mathbf{o}_t = \mathbf{q}_t^{\top}\mathbf{S}_t}$$

这就是一个 **linear RNN with matrix hidden state**。Intuition：每个 (key, value) pair 作为 rank-1 update 写入 memory，query 用 inner product 读取。Inference memory 从 $O(Ld)$ 降到 $O(d^2)$（constant w.r.t. $L$），这是革命性的。

但 training 仍卡在 parallel form 的 $O(L^2 d)$。Recurrent form 训练的三个问题：
1. Sequential，无 sequence-level parallelism
2. 全是 elementwise add/mul 或 reduction，**没有 matmul，无法用 tensor cores**
3. 每步要 materialize hidden state，I/O 开销大

Mamba1 用 [parallel scan + SRAM fusion](https://arxiv.org/abs/2312.00752) 缓解了第 3 点，但 SRAM 限制了 state size。

---

## 3. Chunkwise Parallel Form：interpolate 两种极端

[Hua et al. '22](https://arxiv.org/abs/2209.14755) (Flowformer) 和 [Sun et al. '23](https://arxiv.org/abs/2305.13248) 的洞察：把 sequence 切成大小 $C$ 的 chunk，分三步算。

记 $\mathbf{Q}_{[i]}, \mathbf{K}_{[i]}, \mathbf{V}_{[i]} \in \mathbb{R}^{C \times d}$ 为第 $i$ 个 chunk，$\mathbf{S}_{[i]}$ 为 chunk $i$ 末尾的 state。

**Step 1 - Local state computation**（intra-chunk，parallel）：
$$\mathbf{S}_{[i+1]}^{\text{local}} = \mathbf{K}_{[i+1]}^{\top}\mathbf{V}_{[i+1]}$$
这是每个 chunk 内部的 $\sum_{j}\mathbf{k}_j\mathbf{v}_j^{\top}$，纯 matmul，跑 tensor cores。

**Step 2 - State passing**（inter-chunk，sequential）：
$$\mathbf{S}_{[i+1]} = \mathbf{S}_{[i]} + \mathbf{S}_{[i+1]}^{\text{local}}$$
只有 $L/C$ 步 sequential。

**Step 3 - Output computation**（parallel）：
$$\mathbf{O}_{[i+1]} = \mathbf{Q}_{[i+1]}\mathbf{S}_{[i]} + \big(\mathbf{Q}_{[i+1]}\mathbf{K}_{[i+1]}^{\top}\odot\mathbf{M}_{\text{causal}}\big)\mathbf{V}_{[i+1]}$$
第一项是 previous chunk 的 contribution，第二项是 intra-chunk causal attention。

Trade-off：
- $C=L$：fully parallel，$O(L^2)$ compute
- $C=1$：fully recurrent，$O(L)$ compute 但无并行
- 实践 $C \in \{64, 128, 256\}$，必须 16 的倍数以利用 tensor cores

---

## 4. FlashLinearAttention：把 chunkwise 搬到 Triton

slides 里提出两个版本，对应该 repo [sustcsonglin/flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention)：

### Non-materialization 版（适合短序列）

- Hidden state 始终留在 **SRAM**，整个 recurrence 期间没有 HBM↔SRAM I/O
- Q/K/V 只从 HBM load 一次
- Cons：缺 sequence-level parallelism，需要大 batch 把 SM 喂满

这本质上就是 [FlashAttention](https://arxiv.org/abs/2205.14135) 的思路用在 linear attention 上，但因为 chunkwise 的 sequential step 更轻（一个 matmul + add），SRAM 容得下整个 $d \times d$ state。

### Materialization 版（适合长序列 / 大规模训练）

两步 kernel：
1. **Sequential step**：fuse local state computation + state passing 进单 kernel，一次性 load K/V，store S 到 HBM
2. **Parallel step**：每个 chunk 独立算 output，基于 previous chunk 的 state 和当前 chunk 的 Q/K/V

Trade-off：K/V load 两次，S 要 store + load，I/O 上升，但换来了 chunkwise parallelism 和高 SM occupancy。Backward 用 recomputation 降 memory。

---

## 5. Gated Linear Attention (GLA)：data-dependent decay

Simple linear attention 的问题：memory 只增不减，长序列上信息无限累积，retrieval 能力差。RetNet 加了固定 scalar decay $\gamma$：

$$\mathbf{S}_t = \gamma\mathbf{S}_{t-1} + \mathbf{k}_t\mathbf{v}_t^{\top}$$

但 $\gamma$ 是 constant，不能 data-dependent。GLA 的核心：

$$\boxed{\mathbf{S}_t = \mathbf{G}_t \odot \mathbf{S}_{t-1} + \mathbf{k}_t\mathbf{v}_t^{\top}}$$

其中 $\mathbf{G}_t$ 是 data-dependent 的 gating matrix。具体 parameterization（GLA 的选择）：

$$\mathbf{G}_t = \boldsymbol{\alpha}_t^{\top}\mathbf{1}, \quad \boldsymbol{\alpha}_t = \sigma(\mathbf{x}_t W_{\alpha_1} W_{\alpha_2})^{1/\tau}$$

变量：
- $\boldsymbol{\alpha}_t \in \mathbb{R}^d$：per-channel decay rate（在 $(0,1)$ 之间）
- $\mathbf{1} \in \mathbb{R}^d$：全 1 向量，$\boldsymbol{\alpha}_t^{\top}\mathbf{1}$ 是 rank-1 outer product $\in \mathbb{R}^{d \times d}$，保证 $\mathbf{G}_t$ 是 rank-1 → 仍然能用 chunkwise form
- $\sigma$：sigmoid，$\tau$：temperature（控制 decay sharpness）
- $W_{\alpha_1}, W_{\alpha_2}$：两层 MLP 投影，给 gate 足够 expressivity

### Decay-aware Chunkwise Parallel Form

定义 cumulative decay $b_t := \prod_{j=1}^{t}\alpha_j$（标量版，对每个 channel 独立）。在 chunk $i$ 内位置 $j$：

$$\Lambda_{iC+j} = \frac{b_{iC+j}}{b_{iC}}, \quad \Gamma_{iC+j} = \frac{b_{(i+1)C}}{b_{iC+j}}, \quad \gamma_{i+1} = \frac{b_{(i+1)C}}{b_{iC}}$$

- $\Lambda$：intra-chunk relative decay（从 chunk 起点到 $j$）
- $\Gamma$：从 $j$ 到 chunk 末尾的 decay
- $\gamma_{i+1}$：整个 chunk $i$ 的 decay，用来 pass state

State update：
$$\mathbf{S}_{[i+1]} = (\gamma_{i+1}^{\top}\mathbf{1})\odot\mathbf{S}_{[i]} + (\mathbf{K}_{[i+1]}\odot\Gamma_{[i+1]})^{\top}\mathbf{V}_{[i+1]}$$

Chunk-internal attention 要保持数值稳定，slides 用 log-space：

$$\mathbf{P}_{ij} = \sum_{k=1}^{d}\mathbf{Q}_{ik}\mathbf{K}_{jk}\exp(\log\mathbf{B}_{ik} - \log\mathbf{B}_{jk})$$

把 decay 写成 $\log B$ 的差，避免 $b_t$ 当 $t$ 大时 underflow。这是 [RetNet](https://arxiv.org/abs/2307.08621) 风格的 numerically stable trick。

### GLA 性能（1.3B, 100B tokens）

| Model | PPL↓ | LM Eval↑ | Retrieval↑ |
|---|---|---|---|
| Transformer++ | 16.9 | 50.9 | 41.8 |
| RetNet | 18.6 | 48.9 | 30.6 |
| Mamba | 17.1 | 50.0 | 27.6 |
| GLA | 17.2 | 51.1 | 37.7 |

GLA 在 LM Eval 上甚至超过 Transformer++，retrieval 也比 Mamba/RetNet 强很多。但 retrieval 还是不如 Transformer++，因为 linear attention 的 additive write 不能 overwrite。

Reference: [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635)

---

## 6. GLA == Structured SSM：统一视角

slides 把 Mamba / Mamba-2 / mLSTM / Gated RetNet / HGRN-2 / RWKV-6 / GLA 全部塞进一个 framework：

$$\mathbf{S}_t = \mathbf{G}_t \odot \mathbf{S}_{t-1} + \mathbf{k}_t\mathbf{v}_t^{\top}$$

区别只在 $\mathbf{G}_t$ 的 parameterization：

| Model | $\mathbf{G}_t$ form | Parameters |
|---|---|---|
| Mamba | $\exp(-(\mathbf{1}^{\top}\boldsymbol{\alpha}_t)\odot\exp(\mathbf{A}))$ | $A, W_{\alpha_1}, W_{\alpha_2}$ |
| Mamba-2 | $\gamma_t\mathbf{1}^{\top}\mathbf{1}$, $\gamma_t=\exp(-\text{softplus}(\mathbf{x}_t W_\gamma)\exp(a))$ | $W_\gamma, a$ |
| mLSTM | $\gamma_t\mathbf{1}^{\top}\mathbf{1}$, $\gamma_t=\sigma(\mathbf{x}_t W_\gamma)$ | $W_\gamma$ |
| Gated RetNet | $\gamma_t\mathbf{1}^{\top}\mathbf{1}$, $\gamma_t=\sigma(\mathbf{x}_t W_\gamma)^{1/\tau}$ | $W_\gamma$ |
| HGRN-2 | $\boldsymbol{\alpha}_t^{\top}\mathbf{1}$, $\boldsymbol{\alpha}_t = \gamma + (1-\gamma)\sigma(\mathbf{x}_t W_\alpha)$ | $W_\alpha, \gamma$ |
| RWKV-6 | $\boldsymbol{\alpha}_t^{\top}\mathbf{1}$, $\boldsymbol{\alpha}_t=\exp(-\exp(\mathbf{x}_t W_\alpha))$ | $W_\alpha$ |
| GLA | $\boldsymbol{\alpha}_t^{\top}\mathbf{1}$, $\boldsymbol{\alpha}_t=\sigma(\mathbf{x}_t W_{\alpha_1} W_{\alpha_2})^{1/\tau}$ | $W_{\alpha_1}, W_{\alpha_2}$ |

Intuition：所有这些都是 **scalar-gated linear RNN with rank-1 gating**，区别只在 gate 的来源（fixed A vs data-dependent, 1-layer vs 2-layer, sigmoid vs softplus vs double-exp）。Mamba 的 $\mathbf{A}$ 是 input-independent parameter（structured matrix），其他都是 input-dependent。GLA 用 2-layer MLP 是为了 gate 有足够 nonlinearity 来学复杂的 forgetting pattern。

参考 [Mamba](https://arxiv.org/abs/2312.00752), [Mamba-2](https://arxiv.org/abs/2405.21060), [RWKV-6](https://arxiv.org/abs/2404.16950)。

---

## 7. DeltaNet：从 additive 到 delta rule

Multi-Query Associative Recall 任务：输入 `A4 B3 C6 F1 E2`，问 `A? C? F? E? B?`，要输出 `4 6 1 2 3`。Linear attention 写 memory 是 $\mathbf{S}_t = \mathbf{S}_{t-1} + \mathbf{k}_t\mathbf{v}_t^{\top}$，如果同一个 key 出现两次，旧 value 还在 memory 里没被 overwrite，recall 会出错。

[DeltaNet / Schlag et al. '21](https://arxiv.org/abs/2102.11174) 借鉴 **Fast Weight Programmers**：用 key 先 retrieve 旧 memory，再决定怎么 update。

```
1. Retrieve old memory:        v_old = S_{t-1} k_t
2. Combine with new value:     v_new = β_t v_t + (1 - β_t) v_old
3. Remove old, write new:      S_t = S_{t-1} - v_old k_t^⊤ + v_new k_t^⊤
4. Output:                     o_t = S_t q_t
```

第 3 步的物理意义：把 $\mathbf{k}_t$ 方向上的旧 projection 减掉，再写新的。如果 $\beta_t = 1$，就是完全 overwrite；$\beta_t = 0$ 就是 no-op。这等价于 **Householder-style reflection update**，让 memory 学会"忘记旧的、记住新的"。

### 简化推导

把 $\mathbf{v}_t^{\text{new}} - \mathbf{v}_t^{\text{old}}$ 记作 $\mathbf{u}_t$，则：

$$\mathbf{S}_t = \mathbf{S}_{t-1} + \mathbf{u}_t\mathbf{k}_t^{\top}$$

但 $\mathbf{u}_t$ 依赖 $\mathbf{S}_{t-1}$（因为 $\mathbf{v}_t^{\text{old}} = \mathbf{S}_{t-1}\mathbf{k}_t$），所以这不是简单 linear attention。Naive 的 parallel form 是 $O(L^2)$ 还要 unroll，没法 scale。

### Reparameterization 的关键

把 update 重写成：

$$\mathbf{S}_t = \mathbf{S}_{t-1}(\mathbf{I} - \beta_t\mathbf{k}_t\mathbf{k}_t^{\top}) + \beta_t\mathbf{v}_t\mathbf{k}_t^{\top}$$

展开成 unrolled form：

$$\mathbf{S}_t = \sum_{i=1}^{t}\beta_i(\mathbf{v}_i\mathbf{k}_i^{\top})\left(\prod_{j=i+1}^{t}(\mathbf{I} - \beta_j\mathbf{k}_j\mathbf{k}_j^{\top})\right)$$

这就是 product of **Householder matrices** $(\mathbf{I} - \beta\mathbf{k}\mathbf{k}^{\top})$，每个是一个 rank-1 reflection。问题：直接存 product 是 $O(d^2)$ per step，且 product 本身要 sequential 算。

---

## 8. WY Representation：把 product of Householder 压成 rank-1 sum

这是 slides 的核心 trick，来自 [Bischof & Van Loan '87](https://www.cs.cornell.edu/cv/PDFPDF/WYREP.pdf) 的经典数值线性代数结果。

**Theorem (WY representation)**：任何 product of Householder matrices 可以写成：

$$\mathbf{P}_n = \prod_{t=1}^{n}(\mathbf{I} - \beta_t\mathbf{k}_t\mathbf{k}_t^{\top}) = \mathbf{I} - \sum_{t=1}^{n}\mathbf{w}_t\mathbf{k}_t^{\top}$$

其中 $\mathbf{w}_t$ 可以 **recursively** 构造，不需要 unroll 整个 product。

### 推导 $\mathbf{w}_t$ 的 recursion

假设 $\mathbf{P}_{n-1} = \mathbf{I} - \sum_{t=1}^{n-1}\mathbf{w}_t\mathbf{k}_t^{\top}$，则：

$$\mathbf{P}_n = \mathbf{P}_{n-1}(\mathbf{I} - \beta_n\mathbf{k}_n\mathbf{k}_n^{\top})$$

展开：

$$\mathbf{P}_n = \left(\mathbf{I} - \sum_{t=1}^{n-1}\mathbf{w}_t\mathbf{k}_t^{\top}\right)(\mathbf{I} - \beta_n\mathbf{k}_n\mathbf{k}_n^{\top})$$

$$= \mathbf{I} - \sum_{t=1}^{n-1}\mathbf{w}_t\mathbf{k}_t^{\top} - \beta_n\mathbf{k}_n\mathbf{k}_n^{\top} + \left(\sum_{t=1}^{n-1}\mathbf{w}_t\mathbf{k}_t^{\top}\right)\beta_n\mathbf{k}_n\mathbf{k}_n^{\top}$$

把后两项合并成关于 $\mathbf{k}_n^{\top}$ 的项：

$$= \mathbf{I} - \sum_{t=1}^{n-1}\mathbf{w}_t\mathbf{k}_t^{\top} - \underbrace{\left(\beta_n\mathbf{k}_n - \beta_n\sum_{t=1}^{n-1}\mathbf{w}_t(\mathbf{k}_t^{\top}\mathbf{k}_n)\right)}_{\mathbf{w}_n}\mathbf{k}_n^{\top}$$

所以：

$$\boxed{\mathbf{w}_n = \beta_n\mathbf{k}_n - \beta_n\sum_{t=1}^{n-1}\mathbf{w}_t(\mathbf{k}_t^{\top}\mathbf{k}_n)}$$

这是一个 **linear recursion in $\mathbf{w}$**，可以用 matmul 加速（不像原 Householder product 需要 sequential matrix multiply）。

### $\mathbf{u}_t$ 的对应 recursion

把 $\mathbf{S}_n$ 也写成 $\sum_t \mathbf{u}_t\mathbf{k}_t^{\top}$：

$$\mathbf{S}_n = \mathbf{S}_{n-1}(\mathbf{I} - \beta_n\mathbf{k}_n\mathbf{k}_n^{\top}) + \beta_n\mathbf{v}_n\mathbf{k}_n^{\top}$$

$$= \sum_{t=1}^{n-1}\mathbf{u}_t\mathbf{k}_t^{\top} - \left(\sum_{t=1}^{n-1}\mathbf{u}_t\mathbf{k}_t^{\top}\right)\beta_n\mathbf{k}_n\mathbf{k}_n^{\top} + \beta_n\mathbf{v}_n\mathbf{k}_n^{\top}$$

$$= \sum_{t=1}^{n-1}\mathbf{u}_t\mathbf{k}_t^{\top} + \underbrace{\left(\beta_n\mathbf{v}_n - \beta_n\sum_{t=1}^{n-1}\mathbf{u}_t(\mathbf{k}_t^{\top}\mathbf{k}_n)\right)}_{\mathbf{u}_n}\mathbf{k}_n^{\top}$$

$$\boxed{\mathbf{u}_n = \beta_n\mathbf{v}_n - \beta_n\sum_{t=1}^{n-1}\mathbf{u}_t(\mathbf{k}_t^{\top}\mathbf{k}_n)}$$

---

## 9. DeltaNet 的 Chunkwise Parallel Form

现在有了 $\mathbf{u}_t, \mathbf{w}_t$ 都是 linear recursion，可以直接套 GLA 的 chunkwise 模板：

**Step 1 - Local computation（intra-chunk）**：
- Recurrent 地构造 $\mathbf{W}_{[i]}, \mathbf{U}_{[i]}$（每个 chunk 内部，长度 $C$）
- 这一步 sequential 但 $C$ 小，且每步是 matmul（vector-matrix），可用 tensor cores

**Step 2 - State passing**：
$$\mathbf{S}_{[i+1]} = \mathbf{S}_{[i]}\mathbf{P}_{[i+1]} + \mathbf{U}_{[i+1]}\mathbf{K}_{[i+1]}^{\top}$$
其中 $\mathbf{P}_{[i+1]} = \prod_{j}(\mathbf{I} - \beta_j\mathbf{k}_j\mathbf{k}_j^{\top})$ within chunk，用 WY 表示成 $\mathbf{I} - \mathbf{W}_{[i+1]}\mathbf{K}_{[i+1]}^{\top}$。$L/C$ 步 sequential。

**Step 3 - Output computation（parallel）**：
$$\mathbf{V}_{[i+1]}^{\text{new}} = \mathbf{U}_{[i+1]} - \mathbf{S}_{[i]}\mathbf{W}_{[i+1]}^{\top}$$

然后 output 就是普通 linear attention with new values：
$$\mathbf{O} = (\mathbf{Q}\mathbf{K}^{\top}\odot\mathbf{M})\mathbf{V}^{\text{new}}$$

每个 chunk 内是 $O(C^2 d)$ matmul，所有 chunk parallel。

### 加速比（vs recurrent，single H100）

| Dim | Length | Speedup |
|---|---|---|
| 64 | 2048 | 5.5× |
| 64 | 4096 | 7.6× |
| 64 | 8192 | 11.5× |
| 128 | 2048 | 8.9× |
| 128 | 4096 | 13.2× |
| 128 | 8192 | 13.7× |

序列越长加速比越大，因为 parallel chunk 越多。

---

## 10. DeltaNet 性能：补上 retrieval gap

| Model | PPL↓ | LM Eval↑ | Retrieval↑ |
|---|---|---|---|
| Transformer++ | 16.9 | 50.9 | 41.8 |
| RetNet | 18.6 | 48.9 | 30.6 |
| Mamba | 17.1 | 50.0 | 27.6 |
| GLA | 17.2 | 51.1 | 37.7 |
| **DeltaNet** | **16.9** | **51.6** | 34.7 |
| Hybrid 1: DeltaNet + Sliding window | 16.6 | 52.1 | 40.0 |
| Hybrid 2: DeltaNet + Global attn on 2 layers | 16.6 | 51.8 | **47.9** |

观察：
- DeltaNet 的 PPL 追平 Transformer++，LM Eval 超过
- Retrieval (34.7) 仍不如 Transformer++ (41.8)，因为 DeltaNet 用 Householder update 虽然能 overwrite，但 rank-1 reflection 不如 softmax 的 exact retrieval
- **Hybrid 2**（DeltaNet + 2 层 global attention）拿到 47.9 retrieval，超过纯 Transformer++！这暗示 linear attention + 少量 exact attention 是 sweet spot

### Mechanistic Interpretability 数据

slides 引用了一个 mechanistic design benchmark：

| Model | Compress | Fuzzy Recall | In-Context Recall | Memorize | Noisy Recall | Selective Copy | Avg |
|---|---|---|---|---|---|---|---|
| Transformer | 51.6 | 29.8 | 94.1 | 85.2 | 86.8 | 99.6 | 74.5 |
| Mamba | 52.7 | 6.7 | 90.4 | 89.5 | 90.1 | 86.3 | 69.3 |
| GLA | 38.8 | 6.9 | 80.8 | 63.3 | 81.6 | 88.6 | 60.0 |
| **DeltaNet** | 42.2 | **35.7** | **100** | 52.8 | **100** | **100** | 71.8 |

DeltaNet 在 fuzzy recall / in-context recall / noisy recall / selective copy 都接近或满分，唯独 memorize（52.8）和 compress 偏低。Intuition：Householder update 擅长精确 overwrite（recall 类任务），但 ranking/softmax-based memorization 弱。

---

## 11. Generalization framework：从 elementwise gate 到 structured matmul

slides 最后给出一个漂亮的统一视角：

**Level 1 - Elementwise gate（GLA/SSM）**：
$$\mathbf{S}_t = \mathbf{S}_{t-1}\odot\mathbf{G}_t + \mathbf{v}_t\mathbf{k}_t^{\top}$$
$O(d^2)$ update，但 channel 之间无 interaction。

**Level 2 - Full matmul gate**：
$$\mathbf{S}_t = \mathbf{S}_{t-1}\mathbf{G}_t + \mathbf{v}_t\mathbf{k}_t^{\top}$$
$O(d^3)$，太贵。

**Level 3 - Identity + low-rank gate（DeltaNet）**：
$$\mathbf{S}_t = \mathbf{S}_{t-1}(\mathbf{I} - \mathbf{a}_t\mathbf{b}_t^{\top}) + \mathbf{v}_t\mathbf{k}_t^{\top}$$
$O(kd^2)$（$k$ 是 low-rank），DeltaNet 用 $\mathbf{a}_t = \beta_t\mathbf{k}_t, \mathbf{b}_t = \mathbf{k}_t$ 是 rank-1 special case。

**Open question**：更一般的 associative operator $\bullet$ 怎么并行化？这是 [Mamba-2 的 structured mask attention 视角](https://arxiv.org/abs/2405.21060) 和 [Lightning Attention / RWKV-7](https://arxiv.org/abs/2503.05746) 等后续工作的方向。

---

## 12. Build your intuition 的几个 takeaways

1. **Linear attention = linear RNN with $d \times d$ matrix state**：这个 reparameterization 是一切的起点。Memory 从 $O(L)$ 变 $O(1)$ w.r.t. sequence length。

2. **Chunkwise form 是 interpolate knob**：$C$ 在 1 和 $L$ 之间调，控制 parallel/recurrent 的 trade-off。这是 hardware-aware ML 的范式：算法层面留个旋钮，让 kernel 根据序列长度和 SRAM size 调。

3. **Gate 的 expressivity 决定 retrieval 能力**：scalar decay（RetNet）< per-channel data-dependent gate（GLA）< Householder delta rule（DeltaNet）。每一层 expressivity 提升都对应一类 recall 任务的飞跃。

4. **WY representation 是把 product of Householder 变成 sum of rank-1 的关键**：这是 1987 年数值线性代数的结果，30 多年后在 deep learning 里复活，说明老 toolbox 在新场景往往有未挖掘的价值。

5. **Linear attention 与 SSM 是同构的**：所有 Mamba-family 模型都可以写成 $\mathbf{S}_t = \mathbf{G}_t \odot \mathbf{S}_{t-1} + \mathbf{k}_t\mathbf{v}_t^{\top}$，区别只在 $\mathbf{G}_t$ 的 parameterization。这让 transfer knowledge between frameworks 变得容易。

6. **Hybrid 是当前的 practical sweet spot**：纯 linear attention 在 retrieval 上仍有 gap，加 2 层 global attention 就能把 retrieval 从 34.7 拉到 47.9，超过纯 Transformer。这呼应 [Jamba](https://arxiv.org/abs/2403.19887) 和 [Zamba](https://arxiv.org/abs/2405.16712) 的 hybrid 设计哲学。

---

## References

- [Attention Is All You Need (Vaswani et al. '17)](https://arxiv.org/abs/1706.03762)
- [Linear Transformers (Katharopoulos et al. '20)](https://arxiv.org/abs/2006.16236)
- [Flowformer (Hua et al. '22)](https://arxiv.org/abs/2209.14755)
- [RWKV / Linear Attention chunkwise (Sun et al. '23)](https://arxiv.org/abs/2305.13248)
- [FlashAttention (Dao et al. '22)](https://arxiv.org/abs/2205.14135)
- [RetNet (Sun et al. '23)](https://arxiv.org/abs/2307.08621)
- [Gated Linear Attention (Yang et al. ICML'24)](https://arxiv.org/abs/2312.06635)
- [Parallelizing DeltaNet (Yang et al. '24)](https://arxiv.org/abs/2406.06484)
- [DeltaNet / Fast Weight Programmers (Schlag et al. '21)](https://arxiv.org/abs/2102.11174)
- [WY Representation (Bischof & Van Loan '87)](https://www.cs.cornell.edu/cv/PDFPDF/WYREP.pdf)
- [Mamba (Gu & Dao '23)](https://arxiv.org/abs/2312.00752)
- [Mamba-2 (Dao & Gu '24)](https://arxiv.org/abs/2405.21060)
- [RWKV-6](https://arxiv.org/abs/2404.16950)
- [Jamba hybrid](https://arxiv.org/abs/2403.19887)
- [FlashLinearAttention repo](https://github.com/sustcsonglin/flash-linear-attention)
- [Songlin Yang's site](https://songlinyang.github.io/)
