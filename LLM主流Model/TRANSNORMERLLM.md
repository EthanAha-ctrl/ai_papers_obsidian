---
source_pdf: TRANSNORMERLLM.pdf
paper_sha256: 8177e225f5a3b2e08a82f5e8ddecc66e3680fb7e25d434d92f1e62cffe5dec88
processed_at: '2026-08-12T18:13:22-07:00'
target_folder: LLM-engine/Torch
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说 TransNormerLLM

## 一句话说清楚

Transformer 的 attention 太贵了（$O(n^2)$），所以这帮人搞了个 **linear attention** 版本，又快又省内存，而且精度居然没掉，甚至比 Transformer 还好一点点。

---

## 问题出在哪

标准 Transformer 的 attention 要算一个 $n \times n$ 的大矩阵：

```
sequence length = 8192 → 67 million 个 attention score
sequence length = 100K → 10 billion 个 attention score
```

GPU 内存扛不住，速度也慢。所以大家一直在想：**能不能把 attention 从 $O(n^2)$ 降到 $O(n)$？**

linear attention 的想法很简单：把 softmax 去掉，改成一个随便的 activation function $\phi$，然后矩阵乘法换个顺序：

```
原来:  (Q K^T) V        → 先算 n×n，再乘 V    → O(n²d)
现在:  Q (K^T V)        → 先算 d×d，再乘 Q    → O(nd²)
```

数学上很漂亮。但实际有两个大坑：

**坑 1：精度差。** 没有 softmax 的 "winner-take-all" 效果，attention 变得很钝，所有 token 权重差不多大，模型学不好。

**坑 2：因果模式下根本不快。** 训练时需要 causal mask（每个 token 只能看前面的），这时候 $K^T V$ 不能直接算，要逐行 cumsum，并行性全毁了，实际跑起来比 Transformer 还慢。

所以之前 linear attention 一直没人拿来 scale 到 LLM 级别。

---

## TransNormerLLM 怎么解决的

### 解决坑 1：精度问题

他们用了三个 trick 叠在一起：

**Trick A：Exponential decay 位置编码**

给 attention score 加一个距离衰减：

```
a(s,t) = q_s · k_t · λ^(s-t)     其中 λ < 1
```

意思是：**离我越远的 token，权重自动越小**。这其实模拟了 softmax attention 天然带有的 "近的 token 更重要" 的倾向。

而且不同 head、不同 layer 用不同的 decay rate：
- 浅层 head 衰减快 → 只看 local（类似卷积）
- 深层 head 衰减慢 → 看 global

**Trick B：Gating**

attention 输出后面乘一个 gate $U$：

```
output = (Q K^T V) ⊙ U
```

$U$ 是学出来的，相当于每个 token 可以决定 "我要不要用 attention 的结果"。这个 gate 让训练更稳定，loss 直接降了。

**Trick C：去掉多余的 activation**

GLU 里面原来的 sigmoid 他们发现是多余的，直接去掉：

```
原来:  (V ⊙ sigmoid(U)) W_o
现在:  (V ⊙ U) W_o          ← SGLU
```

element-wise 乘法本身就是 non-linear 的，sigmoid 反而多余。省了计算量，精度还没掉。

### 解决坑 2：速度问题

**Lightning Attention —— 借 FlashAttention 的思路给 linear attention 加速。**

核心 idea：GPU 有两层 memory：
- **HBM**：大但慢（80GB）
- **SRAM**：小但快（每 SM ~192KB）

FlashAttention 的 trick 是把 Q, K, V 切成小块，加载到 SRAM 里算，避免在 HBM 里 materialize 整个 attention matrix。

他们把这个 trick 搬到 linear attention 上，效果惊人：

```
sequence = 8192 时:
  PyTorch baseline:  ~800ms,  显存爆炸
  Lightning Attention: ~100ms,  显存 1/4
```

### 还有一个坑：Inference 时数值爆炸

linear attention 可以像 RNN 一样 inference，维护一个 hidden state $kv_t$：

```
naive 版本:
  kv_t = kv_{t-1} + k_t · λ^(-t) · v_t^T     ← λ^(-t) 会爆炸到 ∞
  o_t   = q_t · λ^t · kv_t                    ← λ^t 会衰减到 0
```

生成到几千个 token 后，float32 就 overflow 了。

**Robust Inference 的 fix**：把 $\lambda^{-t}$ 移到递推里面：

```
robust 版本:
  kv̄_t = λ · kv̄_{t-1} + k_t · v_t^T          ← 只有 λ，没有 λ^(-t)
  o_t   = q_t · kv̄_t
```

数学上完全等价（paper Appendix C 用归纳法证明了），但数值上稳定得多。这个 trick 特别 clean，是整个 paper 最 elegant 的地方。

---

## 效果到底怎么样

### 精度

7B 模型对比：

| Model | HellaSwag | MMLU | C-Eval |
|-------|-----------|------|--------|
| LLaMA2-7B | 76.0 | 45.3 | 33.2 |
| Baichuan2-7B | 72.2 | 54.2 | 54.0 |
| TransNormerLLM-7B | 75.2 | 43.1 | 43.2 |

基本和 LLaMA2 打平，有些 task 略好略差。考虑到这是 linear attention，这个结果已经很 impressive 了。

而且 scale up 的时候优势更明显：
- 385M: 比 Transformer 好 5% loss
- 1B: 比 Transformer 好 9% loss

**越大越占便宜**，这点很重要，说明 scaling 没问题。

### 速度

```
7B model, training:
  Transformer:     3363 tokens/s/GPU
  TransNormerLLM:  4081 tokens/s/GPU    ← 快 21%

175B model, max context:
  Transformer:     10K context
  TransNormerLLM:   12K context, 而且快 35%
```

175B 的 context length 优势更明显，因为 linear attention 的 $O(nd^2)$ 在长 context 下优势放大。

---

## 为什么我觉得这个工作重要

之前大家觉得 linear attention 是个 "理论上很美但实际用不了" 的东西。这篇 paper 基本证明了：

1. **linear attention 可以 scale 到 175B**，不是玩具
2. **精度可以 match softmax attention**，只要 design 对
3. **实际速度真的快**，不只是纸面上的复杂度

它把 "linear attention LLM" 从 "可能行" 变成了 "确实行"。

和 RWKV、RetNet 一样，都是 2023 年这波 "post-Transformer" 探索的重要工作。区别是 TransNormerLLM 更接近 attention 的原始形式，保留了 attention 的 global interaction 语义。

---

## 我的吐槽

1. **没有报告 175B 的 accuracy benchmark**，只报了 speed。这让 "可以 scale 到 175B" 这个 claim 有点弱。

2. **没有 long context benchmark**。既然支持 48K context，为什么不跑个 LongBench / Needle-in-Haystack？光报 context length 能跑多长，不报实际 retrieval 效果，说服力不够。

3. **In-context learning 没评估**。linear attention 在 ICL 上是否 work 是个开放问题，paper 回避了。

4. **$\lambda$ 不能 learnable 这件事**没有深入分析。Mamba 通过 S4 的 special initialization 让类似的东西可学了，这里直接放弃 learnable 有点遗憾。

5. **和 Mamba 的对比缺失**。2023 年 Mamba 很火，paper 完全没提，可能是写的时候 Mamba 还没出，但 timeline 上有点紧。

---

## 参考

- TransNormerLLM GitHub: https://github.com/OpenNLPLab/TransnormerLLM
- 前身 TransNormer: https://aclanthology.org/2022.emnlp-main.473
- FlashAttention (Lightning Attention 灵感来源): https://arxiv.org/abs/2205.14135
- RWKV (类似思路的 linear RNN LLM): https://arxiv.org/abs/2305.13048
- Mamba (state space model, 没对比但值得看): https://arxiv.org/abs/2312.00752
- RetNet (微软的 retention, 思路类似): https://arxiv.org/abs/2307.08621
- Linear Transformer 原始 paper: https://arxiv.org/abs/2006.16236

---

核心 takeaway：**linear attention + multi-scale decay + gating + IO-aware kernel + numerical stability trick = 一个能用的 LLM backbone**。每一个 ingredient 都不复杂，但组合在一起 work 了，这就是 engineering 的价值。

---

# TransNormerLLM 详细技术讲解

让我从 Andrej 的视角来 build intuition about 这篇 paper，这是一个把 linear attention 真正 scale 到 LLM 级别的工作。

## 一、Core Motivation: 为什么需要 Linear Attention LLM?

传统的 softmax attention:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

它的 time complexity 是 $O(n^2 d)$，因为要计算 $n \times n$ 的 attention matrix。当 sequence length $n$ 很大时，这就是 bottleneck。

Linear Attention 的核心想法是去掉 softmax，改用 kernel function $\phi$:
$$\text{Attention}(Q, K, V) = \phi(Q)\phi(K)^\top V = \phi(Q)(\phi(K)^\top V)$$

通过"右乘"重排，复杂度从 $O(n^2 d)$ 降到 $O(n d^2)$，因为 $\phi(K)^\top V$ 是 $d \times d$ 矩阵。

**但这里有两个 critical problems**:
1. **Performance gap**: 大多数 linear attention 变体在语言建模上比 Transformer 差
2. **Causal 的实际速度**: 在 causal attention 场景下，右乘需要 cumsum 操作（Hua et al. 2022, Flash paper），这会破坏并行性，实际速度反而不快

TransNormerLLM 同时解决了这两个问题，做到了"既准又快"。

---

## 二、Positional Encoding: LRPE-d (Linearized Relative Positional Encoding with exponential decay)

### 2.1 公式解析

公式 (1):
$$a_{st} = \mathbf{q}_s^\top \mathbf{k}_t \lambda^{s-t} \exp^{i\theta(s-t)}$$

变量含义:
- $a_{st}$: 位置 $s$ 的 query 和位置 $t$ 的 key 之间的 attention score
- $\mathbf{q}_s \in \mathbb{R}^d$: 位置 $s$ 的 query 向量
- $\mathbf{k}_t \in \mathbb{R}^d$: 位置 $t$ 的 key 向量
- $\lambda \in (0, 1)$: exponential decay 因子，$|s-t|$ 越大权重越小
- $\theta$: learnable 的 frequency 参数
- $\exp^{i\theta(s-t)}$: 类似 RoPE 的旋转位置编码

**关键 insight**: 这个 position encoding 可以分解为 $s$ 和 $t$ 分开的形式（公式 16）:
$$a_{st} = (\mathbf{q}_s \lambda^s \exp^{i\theta s})^\top (\mathbf{k}_t \lambda^{-t} \exp^{i\theta t})$$

这使得 linear attention 的 RNN-style inference 成为可能，因为 $s$ 和 $t$ 是 decoupled 的。

### 2.2 Multi-scale Decay 设计

公式 (2):
$$\lambda = \exp\left(-\frac{8h}{H} \times \left(1 - \frac{l}{L}\right)\right)$$

变量含义:
- $h \in \{0, 1, ..., H-1\}$: head index
- $H$: 总 head 数（e.g., 32 for 7B）
- $l \in \{1, 2, ..., L\}$: layer index
- $L$: 总 layer 数（e.g., 30 for 7B）
- $\frac{8h}{H}$: head-specific decay rate, h 越大 decay 越快
- $(1 - \frac{l}{L})$: layer-specific temperature, $l$ 越小 decay 越快

**Intuition building**:
- **Lower layers**: $l$ 小, $(1 - \frac{l}{L})$ 接近 1, decay 强 → Theoretical Receptive Field (TRF) 小, 关注 local pattern
- **Higher layers**: $l$ 大, $(1 - \frac{l}{L})$ 接近 0, decay 弱 → TRF 大, 关注 global context
- **最后一层** ($l=L$): decay rate = 1, 每个 token 可以 attend to 全局信息

这和 CNN 的 hierarchical receptive field 思想类似, 但通过 decay rate 实现平滑过渡。

**为什么 $\lambda$ 不 learnable?** Paper 中说 empirically 发现 $\lambda$ learnable 会导致 gradient unstable 和 NaN。这也是 state space models 如 S4 早期遇到的 training stability 问题。

### 2.3 Mix 策略: 训练速度 vs 性能的 trade-off

Table 3 的 ablation:
| PE Methods | Loss | PPL |
|-----------|------|-----|
| Mix (first layer LRPE-d + others Exp-Decay) | 2.248 | 4.770 |
| LRPE-d (all layers) | 2.236 | 4.728 |
| Exp-Decay (all layers) | 2.267 | 4.834 |
| LRPE (no decay) | 2.287 | 4.899 |
| APE | 2.387 | 5.253 |

**关键观察**:
- LRPE-d 性能最好 (PPL 4.728)
- Mix 仅差 0.04 PPL, 但训练快 15-20%
- APE (absolute position encoding) 最差, 说明 relative position 的重要性

Mix 策略的 intuition: 第一层用完整 LRPE-d 提供 global relative position signal, 后续层只需要 exponential decay 就够了, 因为 high-level representation 已经隐含了 position 信息。

---

## 三、Gating Mechanism: GLA + SGLU

### 3.1 Token Mixer: Gated Linear Attention (GLA)

公式 (3) & (4):
$$\mathbf{O} = \text{Norm}(\mathbf{Q}\mathbf{K}^\top \mathbf{V}) \odot \mathbf{U}$$

其中:
$$\mathbf{Q} = \phi(\mathbf{X}\mathbf{W}_q), \quad \mathbf{K} = \phi(\mathbf{X}\mathbf{W}_k), \quad \mathbf{V} = \mathbf{X}\mathbf{W}_v, \quad \mathbf{U} = \mathbf{X}\mathbf{W}_u$$

变量:
- $\mathbf{X} \in \mathbb{R}^{n \times d}$: input sequence
- $\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v, \mathbf{W}_u \in \mathbb{R}^{d \times d}$: weight matrices
- $\phi$: swish activation (即 $\text{swish}(x) = x \cdot \text{sigmoid}(x)$)
- $\odot$: element-wise multiplication (Hadamard product)
- $\mathbf{U}$: gate vector, 控制 token mixing 的输出

**Intuition**: 
- $\mathbf{Q}\mathbf{K}^\top \mathbf{V}$ 是 linear attention 的输出
- $\mathbf{U}$ 作为 gate, 让 model 学会 "何时传递" vs "何时抑制" attention 信息
- 类似 LSTM 的 forget gate 或 GRU 的 update gate, 但这里是 multiplicative gating

Table 5 ablation 显示 gate 让 loss 从 2.263 降到 2.248 (w/o gate 用 379M params, w/ gate 用 385M, 因为多了 $\mathbf{W}_u$)。

### 3.2 Activation Function 选择 (GLA)

Table 6:
| GLA Act | Loss | PPL |
|---------|------|-----|
| Swish | 2.248 | 4.770 |
| 1+elu | 2.252 | 4.767 |
| No Act | 2.283 | 4.882 |

Swish 和 1+elu 性能接近, 但 1+elu 在 7B model 上遇到 NaN 问题, 所以最终用 Swish。1+elu 是 Katharopoulos et al. 2020 原始 linear transformer 用的, 但它在数值上不够 stable, 这也是为什么 linear attention 训练容易崩。

### 3.3 Channel Mixer: Simple Gated Linear Unit (SGLU)

公式 (5):
$$\mathbf{O} = [\mathbf{V} \odot \mathbf{U}]\mathbf{W}_o, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_v, \quad \mathbf{U} = \mathbf{X}\mathbf{W}_u$$

对比标准 GLU:
$$\text{GLU}(x) = (\mathbf{X}\mathbf{W}_v \otimes \sigma(\mathbf{X}\mathbf{W}_u))\mathbf{W}_o$$

SGLU 去掉了 $\sigma$ activation function, 直接用 $\mathbf{U}$ 作为 gate。

**Intuition**: 
- $\mathbf{V} \odot \mathbf{U}$ 本身就是 non-linear operation (element-wise product)
- 去掉 $\sigma$ 后, computation 更简单, 速度更快
- Paper 实验 (Table 7) 显示 performance 没有下降, 反而略好 (PPL 4.770 vs 4.788 with Swish)

这是个很 clean 的发现: GLU 中的 sigmoid 非线性是 redundant 的, gating 本身就够 non-linear 了。

---

## 四、Tensor Normalization: SimpleRMSNorm (SRMSNorm)

公式 (8):
$$\text{SRMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\|\mathbf{x}\|_2 / \sqrt{d}}$$

变量:
- $\mathbf{x} \in \mathbb{R}^d$: input vector
- $\|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2}$: L2 norm
- $d$: dimension
- $\sqrt{d}$: scaling factor (类似 attention 中的 $\sqrt{d_k}$)

对比 RMSNorm:
$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\|\mathbf{x}\|_2 / \sqrt{d}} \cdot \gamma$$

SRMSNorm 去掉了 learnable scale parameter $\gamma$。

**为什么去掉 $\gamma$?** Paper 没有明确说, 但我的猜测:
1. 减少 parameter count
2. 在 Triton implementation 中, 没有 $\gamma$ 让 kernel 更简单
3. 实验显示性能无差异 (Table 8)

**NormAttention 的形式** (公式 6 & 7):
$$\mathbf{O} = \text{Norm}((\mathbf{Q}\mathbf{K}^\top)\mathbf{V}) = \text{Norm}(\mathbf{Q}(\mathbf{K}^\top \mathbf{V}))$$

这里 $\text{Norm}$ 是 SRMSNorm, 用在 attention output 上, 代替 softmax 的 normalization 作用。这是 TransNormer 原始工作的核心 idea, 让 attention 可以 linear 化。

---

## 五、Lightning Attention: IO-aware Linear Attention

这是 paper 的工程亮点, 借鉴 FlashAttention 的 tiling 思想到 linear attention。

### 5.1 核心公式

公式 (10):
$$\mathbf{O} = (\mathbf{Q}\mathbf{K}^\top \odot \mathbf{M})\mathbf{V}$$

- $\mathbf{M} \in \mathbb{R}^{n \times n}$: attention mask (包含 causal masking + positional encoding)
- $\odot$: element-wise product

### 5.2 Algorithm 3 (Forward Pass) 详解

```
Input: Q, K, V ∈ R^(n×d), mask M ∈ R^(n×n), block sizes B_c, B_r
Divide Q into T_r = n/B_r blocks, each B_r × d
Divide K, V into T_c = n/B_c blocks, each B_c × d
Divide M into T_r × T_c blocks, each B_r × B_c

for i = 1 to T_r:
    Load Q_i from HBM to SRAM  # on-chip
    Initialize O_i = 0 on SRAM
    for j = 1 to T_c:
        Load K_j, V_j from HBM to SRAM
        Load M_ij from HBM to SRAM
        # On-chip computation
        A_ij = [Q_i @ K_j^T] ⊙ M_ij  # B_r × B_c
        O_i = O_i + A_ij @ V_j  # B_r × d
    Write O_i to HBM
```

**Intuition**:
- HBM (high bandwidth memory, GPU 的 main memory) 读写慢
- SRAM (on-chip memory) 读写快但小 (~100KB per SM)
- 通过 tiling, 把数据分块加载到 SRAM, 在 SRAM 中完成所有计算, 只在最后写回 HBM
- 这避免了 materializing 整个 $n \times n$ attention matrix 在 HBM 中

### 5.3 性能数据 (Figure 3)

| Seq Length | Baseline (PyTorch) | Lightning Attention | Speedup |
|-----------|-------------------|---------------------|---------|
| 2048 | ~50ms | ~25ms | 2x |
| 4096 | ~200ms | ~50ms | 4x |
| 8192 | ~800ms | ~100ms | 8x |

Memory: Lightning Attention 是 linear scaling, baseline 是 quadratic scaling, 在 seq=8192 时节省 4x memory。

---

## 六、Robust Inference Algorithm

这是 paper 中最 elegant 的数学 trick, 解决 linear attention inference 的数值稳定性问题。

### 6.1 问题分析

公式 (16) 的分解形式:
$$a_{st} = (\mathbf{q}_s \lambda^s \exp^{i\theta s})^\top (\mathbf{k}_t \lambda^{-t} \exp^{i\theta t})$$

当 $\lambda < 1$ 且 $t$ 很大时 (公式 17):
- $\|\mathbf{q}_s \lambda^s\| \to 0$ (query 衰减到 0)
- $\|\mathbf{k}_t \lambda^{-t}\| \to \infty$ (key 爆炸到无穷)

这导致 float32/float16 的 numerical overflow。

### 6.2 原始 Inference (Algorithm 1)

```
Initialize: [kv]_0 = 0
for t = 1 to n:
    [kv]_t = [kv]_{t-1} + k_t * λ^(-t) * v_t^T  # 注意 λ^(-t)
    o_t = q_t * λ^t * [kv]_t
```

问题: $\lambda^{-t}$ 在 long sequence 下爆炸。

### 6.3 Robust Inference (Algorithm 2)

```
Initialize: [kv̄]_0 = 0
for t = 1 to n:
    [kv̄]_t = λ * [kv̄]_{t-1} + k_t * v_t^T  # 注意 λ (不是 λ^(-t))
    o_t = q_t * [kv̄]_t
```

### 6.4 等价性证明 (Appendix C)

证明 $[\mathbf{kv}]_t = \lambda^{-t}[\overline{\mathbf{kv}}]_t$:

**Base case** ($n=1$):
$$[\mathbf{kv}]_1 = [\mathbf{kv}]_0 + \mathbf{k}_1 \lambda^{-1} \mathbf{v}_1^\top = \lambda^{-1}(\mathbf{k}_1 \mathbf{v}_1^\top) = \lambda^{-1}[\overline{\mathbf{kv}}]_1$$

**Inductive step** (假设 $n=m-1$ 成立, 证 $n=m$):
$$[\mathbf{kv}]_m = [\mathbf{kv}]_{m-1} + \mathbf{k}_m \lambda^{-m} \mathbf{v}_m^\top$$
$$= \lambda^{-(m-1)}[\overline{\mathbf{kv}}]_{m-1} + \mathbf{k}_m \lambda^{-m} \mathbf{v}_m^\top$$
$$= \lambda^{-m}(\lambda[\overline{\mathbf{kv}}]_{m-1} + \mathbf{k}_m \mathbf{v}_m^\top)$$
$$= \lambda^{-m}[\overline{\mathbf{kv}}]_m$$

所以 $o_t = q_t \lambda^t [\mathbf{kv}]_t = q_t \lambda^t \cdot \lambda^{-t}[\overline{\mathbf{kv}}]_t = q_t [\overline{\mathbf{kv}}]_t$, 两者等价。

**关键 trick**: 把 $\lambda^{-t}$ 的累积从 $\mathbf{k}_t$ 移到 $[\mathbf{kv}]$ 的递推中, 通过 $\lambda \cdot [\overline{\mathbf{kv}}]_{t-1}$ 实现 decay, 避免任何 $\lambda^{-t}$ 的 explicit computation。

这个 trick 类似于 RWKV 中的 time-mixing, 也有 state space models 的影子 (Mamba 的 $A$ matrix)。

---

## 七、整体架构 (Figure 1)

每个 TransNormerLLM block:
```
Input X
    │
    ▼
SRMSNorm ──► GLA (Token Mixer) ──┐
    │                              │ (residual)
    ▼                              ▼
    └──────────────────────────── X + GLA_output
                                  │
                                  ▼
                          SRMSNorm ──► SGLU (Channel Mixer) ──┐
                                  │                            │ (residual)
                                  ▼                            ▼
                                  └──────────────────────────── X + SGLU_output
```

伪代码 (公式 9):
```python
X = X + GLA(SRMSNorm(X))
X = X + SGLU(SRMSNorm(X))
```

PreNorm 结构, 和 GPT-2 / LLaMA 一致。

**与 Transformer 的对比**:
| Component | Transformer | TransNormerLLM |
|-----------|-------------|----------------|
| Token Mixer | Softmax Self-Attention | GLA (Linear Attention + Gate) |
| Channel Mixer | MLP (或 SwiGLU) | SGLU |
| Normalization | LayerNorm / RMSNorm | SRMSNorm |
| Positional Encoding | RoPE / ALiBi | LRPE-d (LRPE + Exp Decay) |
| Attention Complexity | $O(n^2 d)$ | $O(n d^2)$ |
| Inference | KV Cache | RNN-style Recurrent |

---

## 八、Model Parallelism

### 8.1 SGLU 的 Model Parallelism

公式 (12) & (13):
$$[\mathbf{O}_1', \mathbf{O}_2'] = \mathbf{X}[\mathbf{W}_v^1, \mathbf{W}_v^2] \odot \mathbf{X}[\mathbf{W}_u^1, \mathbf{W}_u^2]$$
$$\mathbf{O} = [\mathbf{O}_1', \mathbf{O}_2'][\mathbf{W}_o^1, \mathbf{W}_o^2]^\top = \mathbf{O}_1'\mathbf{W}_o^1 + \mathbf{O}_2'\mathbf{W}_o^2$$

- $\mathbf{W}_v, \mathbf{W}_u$ 沿 column 分割到不同 GPU
- 输出 $\mathbf{O}'$ 沿 column 分割
- $\mathbf{W}_o$ 沿 row 分割
- 最后 all-reduce 求和 (类似 Megatron-LM)

### 8.2 GLA 的 Model Parallelism

公式 (14) & (15):
$$[\mathbf{O}_1, \mathbf{O}_2] = \text{SRMSNorm}(\mathbf{Q}\mathbf{K}^\top \mathbf{V}) \odot \mathbf{U}$$

其中 Q, K, V, U 都沿 column 分割:
$$\mathbf{Q} = [\phi(\mathbf{X}\mathbf{W}_q^1), \phi(\mathbf{X}\mathbf{W}_q^2)]$$
$$\mathbf{K} = [\phi(\mathbf{X}\mathbf{W}_k^1), \phi(\mathbf{X}\mathbf{W}_k^2)]$$

**关键点**: 线性 attention 中 $\mathbf{Q}\mathbf{K}^\top \mathbf{V}$ 在 head 维度上是独立的, 所以可以按 head 分割 (类似 multi-head attention 的 parallelism)。

### 8.3 实验数据 (Table 12)

| Model | MP Size | Tokens/s | Memory/GPU |
|-------|---------|----------|------------|
| Transformer-7B | 1 | 26896 | 66.3 GB |
| Transformer-7B | 8 | 19973 | 28.7 GB |
| TransNormerLLM-7B | 1 | 32048 | 64.0 GB |
| TransNormerLLM-7B | 8 | 24280 | 24.1 GB |

**观察**:
- TransNormerLLM 比 Transformer 快 ~20% (same MP size)
- MP=8 时 memory 节省 62.3% vs MP=1
- 速度下降较少 (因为 NVLink interconnect)

---

## 九、Benchmark 性能 (Table 9)

### 9.1 7B 规模比较

| Model | BoolQ | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA | MMLU | CMMLU | C-Eval |
|-------|-------|------|-----------|-----------|-------|-------|------|------|-------|--------|
| LLaMA1-7B | 76.5 | 79.8 | 76.1 | 70.1 | 72.8 | 47.6 | 57.2 | 35.1 | 25.6 | 25.7 |
| LLaMA2-7B | 77.7 | 78.1 | 76.0 | 69.0 | 76.3 | 46.3 | 44.2 | 45.3 | 33.0 | 33.2 |
| Baichuan2-7B | 72.7 | 76.5 | 72.2 | 68.4 | 75.2 | 42.3 | 39.6 | 54.2 | 57.1 | 54.0 |
| ChatGLM2-6B | 77.7 | 69.4 | 50.5 | 57.6 | 59.1 | 34.3 | 37.0 | 45.5 | 48.8 | 52.6 |
| **TransNormerLLM-7B** | 75.9 | 80.1 | 75.2 | 66.1 | 75.4 | 44.4 | 63.4 | 43.1 | 48.0 | 43.2 |

**观察**:
- TransNormerLLM 在大多数任务上和 LLaMA2、Baichuan2 competitive
- PIQA, OBQA 上甚至更好
- HellaSwag 和 WinoGrande 略差, 可能是因为 commonsense reasoning 需要 strong global attention
- 中文 benchmark (CMMLU, C-Eval) 表现不错, 说明多语言能力

### 9.2 规模性能 (Table 1)

| Model Size | Transformer Loss | TransNormerLLM Loss | Improvement |
|-----------|-----------------|---------------------|-------------|
| 385M | 2.362 | 2.248 | 5% |
| 1B | 2.061 | 1.896 | 9% |

**重要观察**: 规模越大, TransNormerLLM 的优势越明显 (1B 时差 9% loss)。这可能是因为 linear attention 在 long context 和 large model 上的 scaling 更平滑。

---

## 十、Scaling to 175B (Table 13 & 14)

### 10.1 Training Speed

| Model Size | Transformer (tokens/s/GPU) | TransNormerLLM (tokens/s/GPU) |
|-----------|---------------------------|-------------------------------|
| 7B | 3362 | 4081 |
| 13B | 1736 | 2104 |
| 65B | 318 | 407 |
| 175B | 106 | 137 |

TransNormerLLM 在所有规模上比 Transformer 快 ~20-30%。

### 10.2 Max Context Length

| Model Size | Transformer Context | TransNormerLLM Context | Speed Ratio |
|-----------|--------------------|-----------------------|-------------|
| 7B | 37K | 48K | 1.21x |
| 13B | 24K | 35K | 1.23x |
| 65B | 19K | 23K | 1.29x |
| 175B | 10K | 12K | 1.35x |

**关键发现**: 模型越大, TransNormerLLM 的 context length 优势越大 (175B 时 1.35x)。这验证了 linear attention 的 scaling 优势。

---

## 十一、Intuitive Summary: 为什么 TransNormerLLM 工作?

### 11.1 三层直觉

**Level 1: 单个 head 的 attention**
$$a_{st} = q_s^\top k_t \lambda^{s-t}$$
- 类似 LSTM 的 hidden state $h_t = \lambda h_{t-1} + k_t v_t^\top$
- $\lambda$ 控制 "forget rate", 模拟 exponential decay memory

**Level 2: multi-head / multi-layer**
$$\lambda_{h,l} = \exp\left(-\frac{8h}{H} \times \left(1 - \frac{l}{L}\right)\right)$$
- 不同 head 学习不同 time-scale (类似 Inception 的 multi-scale conv)
- 不同 layer 处理不同 abstraction level
- 形成 hierarchical receptive field, 类似 CNN 的 inductive bias

**Level 3: 整个 architecture**
- GLA (gated) → RNN-like recurrent state, 但 training 时可以 parallel
- SGLU → 通道维度的 non-linear mixing
- Lightning Attention → IO-aware tiling, 充分利用 GPU memory hierarchy
- Robust Inference → 数值稳定的 RNN-style decoding

### 11.2 和其他工作的关系

1. **vs RWKV**: RWKV 是 linear RNN, TransNormerLLM 是 linear attention + gating, 两者都是 RNN-style inference, 但 TransNormerLLM 保留了 attention 的 global interaction
2. **vs Mamba (S4)**: Mamba 用 SSM (state space model), TransNormerLLM 用 gated linear attention, 都是 linear complexity, 但 TransNormerLLM 的 LRPE-d 提供 explicit positional encoding
3. **vs FlashAttention**: Lightning Attention 借鉴 FlashAttention 的 tiling 思想, 但应用到 linear attention, 利用 linear attention 的数学性质实现更好的 tiling
4. **vs RetNet (Microsoft)**: RetNet 用 retention mechanism, 和 TransNormerLLM 的 exponential decay 很类似, 都是 multi-scale decay + linear attention

### 11.3 局限性 (paper 没有讨论)

1. **Long-range dependency**: exponential decay 本质上是 local 的, 虽然最后一层 no decay, 但中间层的 information flow 可能受限
2. **In-context learning**: linear attention 在 in-context learning 上可能不如 softmax (因为 softmax 的 sharp attention 更适合 retrieval-like 任务)
3. **LRPE-d 的 theoretical analysis**: paper 没有提供为什么 LRPE-d 比 RoPE/ALiBi 更好的理论分析
4. **175B 的 benchmark**: paper 只报告了 385M, 1B, 7B 的 benchmark, 175B 的 accuracy 没有报告

### 11.4 工程上的启示

1. **Linear attention 是 viable 的 LLM backbone**: 之前大家觉得 linear attention 性能不够, TransNormerLLM 证明通过正确的 design (decay + gating + normalization) 可以 match softmax attention
2. **IO-aware implementation 是 key**: linear attention 的理论复杂度优势只有在 IO-aware 实现下才能转化为实际速度
3. **Numerical stability 是 critical**: linear attention 的 RNN-style inference 容易遇到数值问题, robust inference 是必须的

---

## 十二、Web Links for Reference

- **Paper**: https://arxiv.org/abs/2310.01222 (TransNormerLLM, though the actual ID may differ)
- **GitHub**: https://github.com/OpenNLPLab/TransnormerLLM
- **Original TransNormer**: https://aclanthology.org/2022.emnlp-main.473
- **LRPE**: https://openreview.net/forum?id=Bl8CQrx2Up4 (cosFormer, related)
- **FlashAttention**: https://arxiv.org/abs/2205.14135 (FlashAttention by Tri Dao)
- **FlashAttention-2**: https://arxiv.org/abs/2307.08691
- **RWKV**: https://arxiv.org/abs/2305.13048
- **Mamba (S4)**: https://arxiv.org/abs/2111.00396
- **RetNet**: https://arxiv.org/abs/2307.08621
- **Linear Transformer**: https://arxiv.org/abs/2006.16236 (Katharopoulos et al.)
- **Megatron-LM**: https://arxiv.org/abs/1909.08053
- **PyTorch FSDP**: https://arxiv.org/abs/2304.11277
- **Triton**: https://www.eecs.harvard.edu/~htk/publications/2019-tillet-kung-cox.pdf
- **Swish activation**: https://arxiv.org/abs/1710.05941
- **RoPE (Rotary Positional Embedding)**: https://arxiv.org/abs/2104.09864
- **ALiBi**: https://arxiv.org/abs/2108.12409
- **LM-Eval-Harness**: https://github.com/EleutherAI/lm-evaluation-harness
- **MMLU**: https://arxiv.org/abs/2009.03300
- **C-Eval**: https://arxiv.org/abs/2305.08322
- **CMMLU**: https://arxiv.org/abs/2306.09212

---

## 十三、详细架构图解析

Figure 1 的结构 (textual reconstruction):

```
┌─────────────────────────────────────────────────────────┐
│                TransNormerLLM Block                     │
│                                                         │
│   Input X                                               │
│      │                                                  │
│      ▼                                                  │
│   ┌──────────┐    Q = φ(XW_q)                           │
│   │SRMSNorm │    K = φ(XW_k)    ┌──────────────────┐    │
│   └──────────┘    V = XW_v      │                  │    │
│      │           U = XW_u      │   GLA Module     │    │
│      └──────────────►          │   O = Norm(QK^TV) │    │
│                                │     ⊙ U          │    │
│                                └────────┬─────────┘    │
│                                         │              │
│   ─────────────────────────────────────┼──────────    │
│                                         ▼              │
│                                X + GLA_out            │
│                                         │              │
│                                         ▼              │
│                                ┌──────────┐           │
│                                │SRMSNorm │           │
│                                └──────────┘           │
│                                         │              │
│      V = XW_v  U = XW_u                 ▼              │
│      ┌──────────────────┐   ┌──────────────────┐      │
│      │   SGLU Module    │◄──│  Channel Mixer   │      │
│      │ O = [V⊙U]W_o     │   │                  │      │
│      └────────┬─────────┘   └──────────────────┘      │
│               │                                        │
│   ───────────┼────────────────────────────────        │
│               ▼                                        │
│      X + SGLU_out ──── Output                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 十四、可能的研究方向

基于这篇 paper, 我想到几个可能延伸方向:

1. **Learnable decay via different parameterization**: Paper 说 $\lambda$ learnable 会 NaN, 但 Mamba 通过 S4 的 special initialization 解决了类似问题, 或许可以借鉴

2. **Multi-scale attention without explicit decay**: 用 learnable kernel function 代替固定的 exponential decay, 让 model 自动学习 multi-scale pattern

3. **Hybrid attention**: 在 lower layer 用 linear attention (local), higher layer 用 softmax attention (global), 类似 Cole et al. 2019 的 Sparse Transformer 思路

4. **LRPE-d 的理论分析**: 为什么 LRPE + exponential decay 比 ALiBi/RoPE 更好? 需要更深入的理论分析

5. **In-context learning 评估**: TransNormerLLM 在 few-shot ICL 上的表现如何? 这是 LLM 的重要能力, 但 paper 没有详细评估

6. **Long context benchmark**: 虽然支持 48K context, 但没有 long context benchmark (如 LongBench, NIAH) 的评估

7. **Quantization compatibility**: linear attention 的 RNN-style inference 对 quantization 友好吗? 这对部署很重要

---

希望这个详细讲解能 build your intuition about TransNormerLLM。核心 takeaway 是: linear attention 通过正确的 positional encoding (LRPE-d)、gating (GLA)、normalization (SRMSNorm)、IO-aware implementation (Lightning Attention) 和 numerical stability (Robust Inference), 可以成为 LLM 的 viable backbone, 同时获得 linear complexity 的 efficiency 优势。这是 linear attention 系列 LLM 的一个 important milestone。
