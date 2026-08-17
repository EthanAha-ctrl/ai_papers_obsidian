---
source_pdf: Gated Attention for Large Language Models.pdf
paper_sha256: 949c7ed67dea3ddc6b88854cf72631ed2e1f466ad4c824c8aa1286739bcf2359
processed_at: '2026-08-04T12:17:54-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好的，我用大白话给你讲讲这篇paper到底在说啥。

## 一句话总结

这篇paper发现：**在Transformer的attention计算完之后，加一个由当前词控制的"开关"（gate），就能让模型变聪明、训练变稳、长文本不崩。**

---

## 背景：标准Attention有啥毛病？

Transformer的attention机制，你可以理解成"每个词去看其他所有词，然后综合信息"。但这个机制有几个让人头疼的问题：

### 毛病1：Low-Rank Bottleneck（表达力受限）

Attention里有两步矩阵乘法：先把输入X变成Value（用$W_V$），算完attention后再变回来（用$W_O$）。数学上，这两步可以合并成一步：

$$o_i^k = \sum_{j=0}^i S_{ij}^k \cdot X_j (W_V^k W_O^k)$$

变量意思：
- $o_i^k$：第$i$个词在第$k$个head的输出
- $S_{ij}^k$：第$i$个词对第$j$个词的attention分数
- $X_j$：第$j$个词的输入
- $W_V^k, W_O^k$：Value和Output的参数矩阵

问题是，$W_V^k W_O^k$合并后是个low-rank矩阵（因为中间维度$d_k$比model维度小），相当于信息被压缩了。连续两个线性层没有非线性，表达力打折扣。

### 毛病2：Attention Sink（注意力垃圾桶）

模型训练完会发现，不管输入啥，第一个token总会分到超高比例的attention（paper里说46.7%）。模型把第一个token当成"垃圾桶"，把不需要的注意力都倒进去。这浪费了attention的容量，还导致长文本时性能崩盘。

### 毛病3：Massive Activations（数值爆炸）

模型中间某些层的hidden state会出现超大的数值（paper里M-Act达到1053），这在BF16训练时容易导致数值不稳定，训练会spike甚至发散。

---

## 核心方法：加个Gate就完事了

### Gate长啥样？

公式特别简单（Equation 5）：

$$Y' = Y \odot \sigma(X W_\theta)$$

变量意思：
- $Y$：attention的输出，需要被调制的东西
- $X \in \mathbb{R}^{n \times d_{model}}$：当前层的输入hidden state（n是序列长度，$d_{model}$是模型维度）
- $W_\theta$：gate自己学的参数
- $\sigma$：sigmoid函数，$\sigma(x) = \frac{1}{1+e^{-x}}$，输出在[0,1]之间
- $\odot$：逐元素相乘
- $Y'$：gate之后的最终输出

**人话**：就是拿当前词的信息，算一个0到1之间的系数，去乘attention的输出。系数接近0的地方，输出就被"关掉"了；系数接近1的地方，输出就保留。

### Gate放哪儿？

Paper试了5个位置（Figure 1）：
- $G_4$：Query projection之后
- $G_3$：Key projection之后  
- $G_2$：Value projection之后
- $G_1$：**SDPA（attention计算）之后** ← 最优
- $G_5$：Final output projection之后

实验结论：**$G_1$位置最好**，就是在attention算完、concat多个head之前，对每个head的输出单独加gate。

### 具体配置

经过30个变体的实验，最优配置是：
- 位置：$G_1$（SDPA输出后）
- 粒度：Elementwise（每个维度一个gate分数）
- Head关系：Head-Specific（每个head有自己的gate）
- 方式：Multiplicative（乘法）
- 激活函数：Sigmoid

---

## 为什么有效？三个原因

### 原因1：Non-Linearity（打破线性瓶颈）

在$G_1$加gate后，公式变成（Equation 8）：

$$o_i^k = \text{Non-Linearity-Map}\left(\sum_{j=0}^i S_{ij}^k \cdot X_j W_V^k\right) W_O^k$$

这里Non-Linearity-Map就是sigmoid gate，它依赖当前词$X_i$（query-dependent）。

**人话**：原来$W_V$和$W_O$之间是纯线性的，可以合并成一层，表达力受限。现在中间插了个sigmoid这种非线性操作，没法合并了，模型的表达力直接上一个台阶。这就是为什么$G_5$位置（$W_O$之后）加gate没用——它没打破$W_V$和$W_O$之间的线性关系。

Paper里还做了个实验验证：只加个SiLU非线性（不加参数，Table 3 row 6），PPL也能降一点。加RMSNorm（也是非线性，Table 3 row 5）也有效。说明非线性的引入确实是性能提升的原因之一。

### 原因2：Sparsity（稀疏过滤）

Paper观察gate分数的统计特性（Table 4）：
- $G_1$ elementwise gate的平均分数只有**0.116**
- 大部分gate分数集中在0附近
- 这意味着模型学会了一个"稀疏过滤器"

**关键发现：Query-Dependency Matters**

$G_1$的gate依赖当前query token $X_i$，$G_2$的gate依赖历史key/value token $X_j$。实验显示$G_1$比$G_2$更好，因为：
- $G_1$是"根据我现在想知道啥，去过滤attention输出"
- $G_2$是"根据历史token自己，去过滤value"

前者更符合直觉——你应该根据当前需求去筛选信息，而不是根据信息源去筛选。

Paper还做了个对照实验（Table 4 row 6）：把gate变成input-independent（直接学一个固定参数），结果gate平均分数升到0.335，sparsity大减，效果也差了。说明**input-dependent的sparsity**才是关键。

### 原因3：Attention-Sink-Free（消除注意力垃圾桶）

这是最惊艳的发现。加了$G_1$ gate后：
- 第一个token的attention比例从**46.7%降到4.8%**（Figure 2）
- Massive activations从**1053降到94**（Table 4 M-Act列）

**人话解释**：原来模型需要attention sink是因为softmax强制要求所有attention分数加起来等于1，即使模型对某些token不感兴趣，也得把分数分配出去，最后全堆到第一个token上。现在有了gate，模型可以直接把不需要的attention输出乘以0，不用再靠attention sink来"消化"多余的注意力了。

Paper里一个精妙的解耦实验（Table 4和Figure 6）：
- 只在$G_2$（value后）加gate：消除了massive activations，但attention sink还在
- 在$G_1$（SDPA后）加gate：两者都消除

这说明**massive activations不是attention sink的必要条件**，真正的解药是query-dependent的sparsity。

---

## 实验数据有多炸裂？

### 性能提升（Table 1，15B MoE模型）

| 配置 | 增加参数 | Avg PPL | MMLU | GSM8k |
|------|---------|---------|------|-------|
| Baseline | 0 | 6.026 | 58.79 | 52.92 |
| 加4个Expert | 400M | 5.964 | 58.84 | 52.54 |
| **$G_1$ Elementwise Gate** | **201M** | **5.761** | **60.82** | **55.27** |
| $G_1$ Headwise Gate | 1.6M | 5.792 | 60.05 | 54.44 |

**人话**：加201M参数的gate，比加400M参数的expert还猛。Headwise版本只加1.6M参数，PPL就降了0.2，性价比爆表。

### 训练稳定性（Table 2，1.7B Dense模型）

| 配置 | Max LR | Avg PPL | 状态 |
|------|--------|---------|------|
| Baseline | 4e-3 | 7.421 | 正常 |
| Baseline | 8e-3 | 9.195 | **发散** |
| Baseline + Sandwich Norm | 8e-3 | 7.407 | 稳定但没提升 |
| **$G_1$ Gate** | 4e-3 | 7.288 | 正常 |
| **$G_1$ Gate** | **8e-3** | **7.325** | **稳定且更好** |

**人话**：学习率翻倍后，原始模型直接炸了。加了gate的模型不仅不炸，性能还更好了。因为gate消除了massive activations，BF16训练时数值更稳定，能容忍更大的学习率。

### 长文本外推（Table 5，RULER benchmark）

| 配置 | 4k | 32k | 64k | 128k |
|------|-----|-----|-----|------|
| Baseline（YaRN扩展） | 82.90 | 37.94 | 37.51 | 31.65 |
| **$G_1$ Gate（YaRN扩展）** | **88.13** | **72.88** | **66.60** | **58.82** |

**人话**：用YaRN把context从32k扩展到128k后，原始模型在128k上性能从79.5暴跌到31.65。加了gate的模型还能保持58.82，差了将近30分。

原因：原始模型依赖attention sink来调节attention分布，YaRN改了RoPE base后，attention sink模式适应不了新的位置编码。gate模型不依赖attention sink，靠input-dependent gate控制信息流，对RoPE变化天生鲁棒。

---

## 联想到的其他工作

### 1. 和Mamba等SSM的关系

[Mamba](https://arxiv.org/abs/2312.00752)的核心创新是input-dependent selectivity，就是让模型根据输入决定记住啥、忘记啥。这篇paper本质上把同样的思想搬到了softmax attention里——gate就是一种soft的、query-dependent的token masking。这说明selectivity可能是序列建模的通用原则，不管是SSM还是attention都受益。

### 2. 和LSTM/GRU的传承

LSTM的forget gate控制细胞状态的保留/遗忘，这篇paper的gate控制attention输出的保留/遗忘。思路完全一致，只是应用场景从RNN的时间步变成了attention的空间维度。[Highway Networks](https://arxiv.org/abs/1505.00387)也是类似思想在深度方向的应用。gating作为一种信息控制机制，在不同架构里反复被证明有效。

### 3. 和SwiGLU的呼应

[SwiGLU](https://arxiv.org/abs/2002.05202)把gating引入了FFN层，现在已经是标准配置。这篇paper把gating引入attention层，效果同样显著。也许未来"gated attention"会像SwiGLU一样成为LLM的标准组件。

### 4. 和[StreamingLLM](https://arxiv.org/abs/2309.17453)的对比

StreamingLLM发现attention sink后，选择保留前几个token来维持长文本性能。这篇paper直接消除了attention sink，从根源上解决问题。两种思路：一种是利用问题，一种是解决问题。

### 5. 和[ViT Registers](https://arxiv.org/abs/2309.16588)的统一

ViT发现vision transformer也会把信息堆到register tokens上。这篇paper在NLP里展示了类似现象，并用gate解决。这暗示了一个统一原则：**给模型显式的"丢弃通道"，比让它隐式地把信息堆到特定token上要健康得多。**

### 6. 和[Differential Transformer](https://arxiv.org/abs/2410.05258)的对比

Differential Transformer用两个attention head的softmax相减来去噪。这篇paper用gate乘法来去噪。目的相同（让attention输出更sparse），手段不同（减法vs乘法）。两者都在摆脱标准softmax attention"必须关注所有token"的束缚。

### 7. 和[Quantizable Transformers](https://arxiv.org/abs/2310.10837)的联系

这篇2023年的工作在BERT和ViT上发现gating能消除outliers，主要用于量化。本篇paper把同样的思想scale up到15B MoE和3.5T tokens，还发现了attention sink消除、长文本外推等新现象。

---

## 对未来的启示

这篇paper给我们的intuition：

1. **简单改动有大效果**：一个sigmoid gate，增加不到2M参数，就能解决多个架构层面的问题。好的架构设计不一定复杂，关键是要打到痛点上。

2. **Sparsity是feature不是bug**：gate的稀疏性让模型学会了"啥都不看"的能力，这比强迫模型关注所有token要健康。也许未来的attention设计应该更鼓励input-dependent的sparsity。

3. **Attention Sink不是必须的**：长期以来大家觉得attention sink是softmax attention的固有特性，只能利用不能消除。这篇paper证明用gate可以彻底消除，还带来一堆好处。

4. **Non-linearity的位置很重要**：在$W_V$和$W_O$之间加非线性有效，在$W_O$之后加无效。这提示我们设计架构时要关注计算的线性可合并性，避免"看似加了层实际没用"的情况。

5. **Query-Dependency是关键**：$G_1$比$G_2$好，说明根据当前需求筛选信息比根据信息源筛选更合理。未来的attention变体可能都应该考虑query-dependent的信息过滤机制。

总之，这篇paper用一个极简的gating机制，把non-linearity、sparsity、attention sink、training stability、long-context extrapolation这几个看似不相关的问题统一解决了。这种"一石多鸟"的发现，往往是深刻洞察的标志。

---

你好 Andrej。这篇由 Qwen Team 发表的 paper《Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free》非常对你的口味，它用非常扎实的 empirical study 揭示了一个极其简单却影响深远的架构修改：在 Scaled Dot-Product Attention (SDPA) 之后加上一个 head-specific 的 sigmoid gate。这个微小的改动不仅提升了性能，还极大地增强了 training stability，甚至彻底消除了困扰 LLM 许久的 attention sink 现象，并在 long-context extrapolation 中展现了惊人的鲁棒性。

下面我为你详细拆解这篇 paper 的技术细节、公式推导、实验数据，并尝试 build your intuition。

---

### 1. Architecture & Formulation: 在哪里加 Gate？

Transformer 的 standard multi-head softmax attention 包含四个阶段：QKV projections, SDPA, Multi-Head Concatenation, 以及 Final Output Projection。这篇 paper 系统性地探索了 5 个插入 gating 的位置（见 Figure 1）：
- $G_4$: After Query projection
- $G_3$: After Key projection
- $G_2$: After Value projection
- $G_1$: After SDPA output (Concatenation 之前，per-head 进行)
- $G_5$: After Final Dense Output layer ($W_O$)

Gating mechanism 的通用公式（Equation 5）定义为：
$$Y' = g(Y, X, W_\theta, \sigma) = Y \odot \sigma(X W_\theta)$$

变量解析：
- $Y$: The input to be modulated (需要被调制的 tensor，例如 SDPA 的输出)。
- $X \in \mathbb{R}^{n \times d_{model}}$: The hidden states input (n 是 sequence length, $d_{model}$ 是 hidden dim)。
- $W_\theta$: Learnable parameters of the gate (映射维度取决于具体的 head-specific 或 head-shared 设置)。
- $\sigma$: Activation function (默认 sigmoid，$\sigma(x) = \frac{1}{1+e^{-x}}$)。
- $\odot$: Element-wise multiplication (Hadamard product)。

通过大量的消融实验，paper 发现 **$G_1$ (SDPA output gating) + Head-specific + Multiplicative + Sigmoid** 是最优组合。这意味着对每一个 attention head 计算出的结果，乘以一个由当前 query token 算出的、范围在 $[0, 1]$ 之间的 scalar 或 vector。

---

### 2. Intuition Building: 为什么 $G_1$ Gate 如此有效？

作者将 $G_1$ 位置 gating 的成功归结为三个核心机制：Non-linearity, Sparsity, 以及 Attention-Sink-Free。

#### 2.1 Non-linearity: 打破 Low-Rank Bottleneck
在标准 attention 中，第 $i$ 个 token 在第 $k$ 个 head 的输出可以写成（Equation 6）：
$$o_i^k = \left( \sum_{j=0}^i S_{ij}^k \cdot X_j W_V^k \right) W_O^k = \sum_{j=0}^i S_{ij}^k \cdot X_j (W_V^k W_O^k)$$
变量解析：
- $o_i^k$: 第 $i$ 个 token 在第 $k$ 个 head 的 output。
- $S_{ij}^k$: 第 $i$ 个 token 对第 $j$ 个 token 在第 $k$ 个 head 上的 attention score。
- $X_j$: 第 $j$ 个 token 的 input hidden state。
- $W_V^k, W_O^k$: 分别是 Value projection 和 Output projection 在第 $k$ 个 head 对应的参数。

**Intuition**: 因为 $d_k < d_{model}$，所以 $W_V^k W_O^k$ 实际上构成了一个 low-rank 的 linear mapping。连续的两个 linear projection 可以合并为一个，这极大限制了 attention 层的表达能力。如果在 $G_1$ 位置加入 gating，公式变成（Equation 8）：
$$o_i^k = \text{Non-Linearity-Map}\left( \sum_{j=0}^i S_{ij}^k \cdot X_j W_V^k \right) W_O^k$$
这里的 Non-Linearity-Map 依赖于当前 token $X_i$ (即 query-dependent)。由于 sigmoid 是非线性的，这就把原本的 low-rank linear transformation 变成了非线性的 mapping，显著增强了 model 的 expressiveness。这也是为什么在 $G_5$ 位置（$W_O$ 之后）加 gate 没用，因为它没有打破 $W_V$ 和 $W_O$ 之间的线性关系。

#### 2.2 Sparsity: Query-Dependent 信息过滤
Paper 中一个非常深刻的发现是：**Effective gating scores are sparse**。
在 $G_1$ 位置使用 Sigmoid gate，其平均 gating score 只有 0.116，大部分值极度接近 0。这意味着 gate 实际上在执行一种 input-dependent 的 sparsification，把无用的 context 信息过滤掉。

更关键的是，**Query-Dependency Matters**。
如果我们在 $G_2$ (Value projection 之后) 加 gate，gate 的计算依赖于历史 token $X_j$ (key/value token)。而在 $G_1$ 位置加 gate，它依赖于当前的 query token $X_i$。实验表明（Table 4），$G_1$ 的 sparsity 比 $G_2$ 更强，性能也更好。这说明，根据当前 query 的需求去动态抹除 SDPA 输出中的冗余信息，比在 key/value 端抹除更有效。为了证明这一点，作者做了一个 Input-Independent Gate 实验（把 $X W_\theta$ 换成可学习的 parameter），结果发现虽然引入了 non-linearity，但由于失去了 query-dependent sparsity，效果大打折扣。

#### 2.3 Attention-Sink-Free: 消除 Massive Activations
"Attention sink" 是指模型将极高比例的 attention score 分配给序列的第一个 token 或某些无意义的 token。这篇 paper 证明了，带有 sparse 特性的 $G_1$ gating 彻底消除了这个问题（见 Figure 2 和 Table 4 中的 F-Attn 列）。
- Baseline 模型将 46.7% 的注意力分配给了第一个 token。
- 加上 $G_1$ gate 后，这个比例降到了 4.8%。

**Intuition**: 以前的研究认为 attention sink 是由于 softmax 归一化迫使模型必须把多余的 attention score 倾倒到某个 "sink" 上。而 $G_1$ 位置的 sparse gating 直接把不需要的 SDPA 输出乘以接近 0 的值，这使得模型不需要再依赖 attention sink 来丢弃冗余信息。同时，这极大地减少了 hidden states 中的 massive activations (M-Act 从 1053 降到 94)，从而从根本上提升了 training stability。

---

### 3. Experimental Data Deep-Dive

Paper 在 15B MoE (15A2B) 和 1.7B Dense 模型上进行了详尽的实验，训练数据高达 3.5T tokens。

#### 3.1 Gating 变体对比 (Table 1 摘要分析)
在 15A2B MoE models (400B tokens) 上：

| Method | Position | Act Func | Added Param | Avg PPL | MMLU |
|---|---|---|---|---|---|
| Baseline | - | - | 0 | 6.026 | 58.79 |
| Add 4 Experts | - | - | 400 | 5.964 | 58.84 |
| SDPA Elementwise | $G_1$ | Sigmoid | 201 | **5.761** | **60.82** |
| V Elementwise | $G_2$ | Sigmoid | 25 | 5.820 | 59.17 |
| SDPA Head-Shared | $G_1$ | Sigmoid | 201 | 5.801 | 60.06 |
| SDPA Additive | $G_1$ | SiLU | 201 | 5.821 | 60.06 |

数据解读：
1. **Position matters**: $G_1$ 和 $G_2$ 都能引入 non-linearity，但 $G_1$ 效果最好。
2. **Head-Specific matters**: 共享 gate scores 跨 heads 会增加平均 gating score (从 0.116 升到 0.271)，降低 sparsity，导致 MMLU 下降。
3. **Multiplicative > Additive**: Additive gate 无法像 multiplicative 那样强制把输出压到 0，sparsity 较弱。
4. **参数效率极高**: 仅增加 201M (甚至 1.6M for headwise) 参数，就超过了增加 400M 参数的 MoE expert 扩容效果。

#### 3.2 Stability & Scaling Up (Table 2 摘要分析)
在 1.7B Dense 模型 (48 Layers) 上，作者尝试了 aggressive 的 learning rate 和 batch size：

| Method | Max LR | Avg PPL | MMLU | GSM8k |
|---|---|---|---|---|
| Baseline | $4.0 \times 10^{-3}$ | 7.421 | 52.04 | 32.98 |
| Baseline | $8.0 \times 10^{-3}$ | 9.195 (Diverge) | 44.28 | 15.24 |
| Baseline + Sandwich Norm | $8.0 \times 10^{-3}$ | 7.407 | 52.07 | 32.90 |
| SDPA Elementwise | $4.0 \times 10^{-3}$ | 7.288 | 52.44 | 32.37 |
| SDPA Elementwise | $8.0 \times 10^{-3}$ | **7.325** | **54.47** | **36.62** |

数据解读：
当 LR 翻倍到 $8.0 \times 10^{-3}$ 时，Baseline 模型直接发散 (PPL 飙到 9.195)。加入 Sandwich Norm 能挽救发散，但性能毫无提升。相比之下，带有 $G_1$ Gate 的模型不仅不发散，反而吃下了更大的 LR，PPL 进一步降低到 7.325，各项 benchmark 全面上升。这证明 gating 带来的 sparsity 消除 massive activations 后，模型的 numerical stability 得到了本质提升，这对 BF16 训练尤其重要。

#### 3.3 Long-Context Extrapolation (Table 5 摘要分析)
作者使用 YaRN 将 32k context 扩展到 128k，并在 RULER 上测试：

| Method | 4k | 32k | 64k | 128k |
|---|---|---|---|---|
| Baseline | 88.89 | 79.50 | - | - |
| SDPA-Gate | 90.56 | 79.77 | - | - |
| **YaRN Extended** | **4k** | **32k** | **64k** | **128k** |
| Baseline | 82.90(-6.0) | 37.94(-41.56) | 37.51 | 31.65 |
| SDPA-Gate | 88.13(-2.4) | 72.88(-6.89) | **66.60** | **58.82** |

数据解读：
在训练长度内 (32k)，两者差不多。但使用 YaRN 扩展到 128k 后，Baseline 在 128k 上的性能从原本的 79.50 暴跌到 31.65。而 SDPA-Gate 模型依然保持在 58.82。**Intuition**: Baseline 模型依赖 attention sink 来调节 attention 分布，当 YaRN 改变了 RoPE 的 base，原有的 attention sink 模式无法适应新的 position encoding，导致性能崩溃。而 Gated Attention 不依赖 attention sink，而是靠 input-dependent gate 来控制信息流，因此对 RoPE 的变化具有极强的鲁棒性。

---

### 4. 联想与延伸

这个工作让我联想到很多前序的研究，脉络非常清晰：

1. **从 RNN 到 SSM 再到 Attention 的 Gating 演化**
   Gating 一直是 RNN 家族（LSTM, GRU）的核心。近年来在 State Space Models (如 [Mamba](https://arxiv.org/abs/2312.00752)) 中，gating 也被证明是引入 input-dependent selectivity 的关键。而在 Attention 机制中，虽然 FFN 层早就普及了 GLU 变体（如 [SwiGLU](https://arxiv.org/abs/2002.05202)），但 attention 内部的 gating 一直未被系统性地作为核心架构来研究。这篇 paper 实际上是把 SSM 中的 "selectivity" 思想移值到了 softmax attention 中，通过 gate 实现了一种 soft 的、query-dependent 的 token masking。

2. **Attention Sink 与 Massive Activations 的因果解耦**
   之前 [StreamingLLM](https://arxiv.org/abs/2309.17453) 发现了 attention sink，而 [Massive Activations](https://arxiv.org/abs/2402.17762) 指出 attention sink 与 hidden state 中的 massive activations 高度相关。这篇 paper 精妙地解耦了这两者：如果在 $G_2$ (Value projection 后) 加 gate，能消除 massive activations，但 attention sink 依然存在。只有在 $G_1$ (SDPA 后) 加 query-dependent gate，才能同时消除两者。这证明 massive activations 并非 attention sink 的必要条件，真正的解药是 query-dependent 的 sparsity。

3. **与 Vision Transformers Registers 的对比**
   [ViT Registers](https://arxiv.org/abs/2309.16588) 发现 ViT 会把全局信息倾倒到几个 register tokens 上。这篇 paper 在 NLP 领域展示了类似的现象：模型用第一个 token 当垃圾桶。而 $G_1$ Gate 的出现，让模型有了“直接把不需要的输出清零”的能力，从而不再需要寄存器 token 或 attention sink。这暗示了一种统一的架构设计原则：**给模型显式的 dropout/forget 通道，比让它通过 softmax 把信息倾倒到特定 token 上要健康得多。**

4. **与 Differential Transformer 的关联**
   最近还有一篇 [Differential Transformer](https://arxiv.org/abs/2410.05258)，通过两个 attention head 的 softmax 相减来消除噪声。虽然机制不同，但目的与本篇 paper 高度一致：都是为了让 attention 输出变得更 sparse，过滤掉无关 context。Differential Transformer 用减法，而 Gated Attention 用乘法，两者都在摆脱标准 softmax attention 那种“必须关注所有 token”的束缚。

### 5. 总结

Qwen Team 这篇 paper 的核心贡献在于，用极低成本的 $G_1$ Sigmoid Gate 解决了 Transformer 架构中多个深层次问题。其核心驱动力在于引入了 query-dependent 的 sparsity，这同时打破了 $W_V W_O$ 的 low-rank bottleneck，消除了 massive activations 导致的 numerical instability，并让模型摆脱了对 attention sink 的依赖，从而在 long-context extrapolation 中大放异彩。

对于下一代 foundation models 的设计，这个 simple gate 极有可能成为像 SwiGLU 一样的 standard component。
