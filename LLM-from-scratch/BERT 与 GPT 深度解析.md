# BERT 与 GPT 深度解析

## 一、第一性原理：从 Language Modeling 出发

### 1.1 Language Modeling 的本质

Language Model 的核心目标是估计一个 sequence 的概率分布：

$$P(w_1, w_2, ..., w_n) = \prod_{i=1}^{n} P(w_i | w_1, ..., w_{i-1})$$

其中：
- $w_i$ 表示第 $i$ 个 token
- $P(w_i | w_1, ..., w_{i-1})$ 表示给定前面所有 token，当前 token 的条件概率
- $n$ 是 sequence 的总长度

### 1.2 两种建模范式

从第一性原理出发，有两种根本不同的建模思路：

| Aspect | Auto-regressive (AR) | Auto-encoding (AE) |
|--------|---------------------|-------------------|
| Direction | Unidirectional | Bidirectional |
| Objective | Next token prediction | Masked token reconstruction |
| Example | GPT series | BERT |
| Strength | Natural generation | Deep context understanding |
| Formula | $\prod_{i=1}^{n} P(w_i\|w_{<i})$ | $\sum_{i \in M} \log P(w_i\|\tilde{w})$ |

---

## 二、Transformer Architecture：共同基石

### 2.1 Transformer Block 核心组件

BERT 和 GPT 都基于 Transformer architecture，但使用不同的部分：

```
┌─────────────────────────────────────────────────────────┐
│                  Transformer Architecture               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   ┌─────────────────────────────────────────────────┐  │
│   │              Encoder Stack (BERT)               │  │
│   │  ┌───────────────────────────────────────────┐  │  │
│   │  │        Multi-Head Self-Attention          │  │  │
│   │  │   Q = XW_Q, K = XW_K, V = XW_V           │  │  │
│   │  │   Attention(Q,K,V) = softmax(QK^T/√d_k)V │  │  │
│   │  └───────────────────────────────────────────┘  │  │
│   │                    ↓ Add & Norm                  │  │
│   │  ┌───────────────────────────────────────────┐  │  │
│   │  │         Feed-Forward Network              │  │  │
│   │  │   FFN(x) = max(0, xW_1 + b_1)W_2 + b_2   │  │  │
│   │  └───────────────────────────────────────────┘  │  │
│   └─────────────────────────────────────────────────┘  │
│                                                         │
│   ┌─────────────────────────────────────────────────┐  │
│   │              Decoder Stack (GPT)                │  │
│   │  ┌───────────────────────────────────────────┐  │  │
│   │  │    Masked Multi-Head Self-Attention       │  │  │
│   │  │   (Causal Mask: 只看过去,不看未来)         │  │  │
│   │  └───────────────────────────────────────────┘  │  │
│   │                    ↓ Add & Norm                  │  │
│   │  ┌───────────────────────────────────────────┐  │  │
│   │  │         Feed-Forward Network              │  │  │
│   │  └───────────────────────────────────────────┘  │  │
│   └─────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Self-Attention 机制详解

Self-Attention 的核心公式：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

变量解释：
- $Q \in \mathbb{R}^{n \times d_k}$：Query matrix，$n$ 是 sequence length，$d_k$ 是 dimension
- $K \in \mathbb{R}^{n \times d_k}$：Key matrix
- $V \in \mathbb{R}^{n \times d_v}$：Value matrix，$d_v$ 通常等于 $d_k$
- $\sqrt{d_k}$：缩放因子，防止 dot product 过大导致 softmax gradient 消失
- $QK^T \in \mathbb{R}^{n \times n}$：Attention score matrix

Multi-Head Attention 公式：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

其中：
- $h$：head 的数量
- $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$：第 $i$ 个 head 的 Query projection matrix
- $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$：第 $i$ 个 head 的 Key projection matrix
- $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$：第 $i$ 个 head 的 Value projection matrix
- $W^O \in \mathbb{R}^{hd_v \times d_{model}}$：Output projection matrix

---

## 三、BERT：Bidirectional Encoder Representations from Transformers

### 3.1 BERT 的核心思想

BERT 的核心 innovation 是 **bidirectional context understanding**。与传统的 left-to-right 或 right-to-left 模型不同，BERT 同时利用左右两侧的 context 来理解每个 token。

### 3.2 BERT Architecture 详细解析

```
┌────────────────────────────────────────────────────────────────┐
│                      BERT Model Architecture                   │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Input: [CLS] Token₁ Token₂ ... [SEP] Token₁' ... [SEP]       │
│           ↓     ↓      ↓         ↓      ↓           ↓         │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              Token Embeddings + Positional             │  │
│  │                 Embeddings + Segment Embeddings        │  │
│  └─────────────────────────────────────────────────────────┘  │
│                           ↓                                    │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                   Encoder Layer 1                       │  │
│  │  ┌─────────────────────────────────────────────────┐    │  │
│  │  │         Multi-Head Self-Attention               │    │  │
│  │  │     (Bidirectional - 可以看到所有 positions)     │    │  │
│  │  └─────────────────────────────────────────────────┘    │  │
│  │                       ↓ Add & Norm                      │  │
│  │  ┌─────────────────────────────────────────────────┐    │  │
│  │  │              Feed-Forward Network               │    │  │
│  │  └─────────────────────────────────────────────────┘    │  │
│  │                       ↓ Add & Norm                      │  │
│  └─────────────────────────────────────────────────────────┘  │
│                           ↓                                    │
│                        ... L layers ...                        │
│                           ↓                                    │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                   Encoder Layer L                       │  │
│  └─────────────────────────────────────────────────────────┘  │
│                           ↓                                    │
│         Output: Hidden States for all input tokens            │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 3.3 BERT Input Representation

BERT 的 input embedding 由三部分组成：

$$E_{input} = E_{token} + E_{position} + E_{segment}$$

其中：
- $E_{token} \in \mathbb{R}^{n \times d_{model}}$：Token embedding from vocabulary
- $E_{position} \in \mathbb{R}^{n \times d_{model}}$：Positional encoding (learned, not sinusoidal)
- $E_{segment} \in \mathbb{R}^{n \times d_{model}}$：Sentence embedding (区分两个 sentence)

### 3.4 BERT Pre-training Objectives

BERT 使用两个 pre-training tasks：

#### 3.4.1 Masked Language Modeling (MLM)

随机 mask 掉 15% 的 input tokens，然后预测这些 masked tokens。

Loss function：

$$\mathcal{L}_{MLM} = -\sum_{i \in M} \log P(w_i | \tilde{w})$$

其中：
- $M$：被 mask 的 token positions 的集合
- $\tilde{w}$：被 mask 后的 sequence
- $w_i$：原始 token

**Masking Strategy：**
- 80% 的情况：替换为 `[MASK]`
- 10% 的情况：替换为随机 token
- 10% 的情况：保持不变

这个设计是为了 alleviate **pre-training-fine-tuning discrepancy** 问题。

#### 3.4.2 Next Sentence Prediction (NSP)

判断两个 sentence 是否是连续的。

$$\mathcal{L}_{NSP} = -\log P(y | s_1, s_2)$$

其中：
- $y \in \{IsNext, NotNext\}$
- $s_1, s_2$ 是两个 input sentences

### 3.5 BERT Model Configurations

| Model | Layers | Hidden Size | Attention Heads | Parameters |
|-------|--------|-------------|-----------------|------------|
| BERT-Base | 12 | 768 | 12 | 110M |
| BERT-Large | 24 | 1024 | 16 | 340M |

### 3.6 BERT Fine-tuning Paradigm

```
┌─────────────────────────────────────────────────────────────┐
│                BERT Fine-tuning Paradigm                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Task 1: Single Sentence Classification                     │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ Input: [CLS] I love this movie [SEP]                  │ │
│  │          ↓                                             │ │
│  │        [CLS] hidden state → Classifier → Label        │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  Task 2: Sentence Pair Classification                       │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ Input: [CLS] Premise [SEP] Hypothesis [SEP]           │ │
│  │          ↓                                             │ │
│  │        [CLS] hidden state → Classifier → Label        │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  Task 3: Question Answering (SQuAD)                         │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ Input: [CLS] Question [SEP] Context [SEP]             │ │
│  │          ↓                                             │ │
│  │        All tokens → Start/End classifiers             │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  Task 4: Named Entity Recognition (Token Classification)    │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ Input: [CLS] Apple is based in Cupertino [SEP]        │ │
│  │          ↓         ↓              ↓                    │ │
│  │        [CLS]    [Apple]       [Cupertino]             │ │
│  │          ↓         ↓              ↓                    │ │
│  │         O        B-ORG          B-LOC                 │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 四、GPT：Generative Pre-trained Transformer

### 4.1 GPT 的核心思想

GPT 采用 **autoregressive language modeling** 范式，通过预测下一个 token 来学习 language representation。

### 4.2 GPT Architecture 详细解析

```
┌────────────────────────────────────────────────────────────────┐
│                      GPT Model Architecture                    │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Input: The cat sat on the                                    │
│          ↓   ↓    ↓    ↓   ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │        Token Embeddings + Positional Embeddings         │  │
│  └─────────────────────────────────────────────────────────┘  │
│                           ↓                                    │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                   Decoder Layer 1                       │  │
│  │  ┌─────────────────────────────────────────────────┐    │  │
│  │  │      Masked Multi-Head Self-Attention           │    │  │
│  │  │                                                 │    │  │
│  │  │   Causal Mask Matrix:                          │    │  │
│  │  │   ┌─────────────────────────┐                  │    │  │
│  │  │   │ 1  0  0  0  0 │        │                  │    │  │
│  │  │   │ 1  1  0  0  0 │        │                  │    │  │
│  │  │   │ 1  1  1  0  0 │        │                  │    │  │
│  │  │   │ 1  1  1  1  0 │        │                  │    │  │
│  │  │   │ 1  1  1  1  1 │        │                  │    │  │
│  │  │   └─────────────────────────┘                  │    │  │
│  │  └─────────────────────────────────────────────────┘    │  │
│  │                       ↓ Add & Norm                      │  │
│  │  ┌─────────────────────────────────────────────────┐    │  │
│  │  │              Feed-Forward Network               │    │  │
│  │  └─────────────────────────────────────────────────┘    │  │
│  │                       ↓ Add & Norm                      │  │
│  └─────────────────────────────────────────────────────────┘  │
│                           ↓                                    │
│                        ... L layers ...                        │
│                           ↓                                    │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              Final Linear + Softmax                     │  │
│  │         Output: Probability distribution over V          │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 4.3 GPT Training Objective

GPT 的 training objective 是 **causal language modeling**：

$$\mathcal{L}_{CLM} = -\sum_{i=1}^{n} \log P(w_i | w_1, w_2, ..., w_{i-1}; \theta)$$

其中：
- $\theta$：model parameters
- $n$：sequence length
- $P(w_i | w_{<i})$：给定前面所有 tokens，预测当前 token 的概率

### 4.4 GPT Model Configurations Evolution

| Model | Layers | Hidden Size | Attention Heads | Parameters | Context Length |
|-------|--------|-------------|-----------------|------------|----------------|
| GPT-1 | 12 | 768 | 12 | 117M | 512 |
| GPT-2 Small | 12 | 768 | 12 | 124M | 1024 |
| GPT-2 Medium | 24 | 1024 | 16 | 355M | 1024 |
| GPT-2 Large | 36 | 1280 | 20 | 774M | 1024 |
| GPT-2 XL | 48 | 1600 | 25 | 1.5B | 1024 |
| GPT-3 | 96 | 12288 | 96 | 175B | 2048 |
| GPT-4 | ~120 | ~16000+ | ~120 | ~1.7T (MoE) | 8192+ |

### 4.5 GPT Generation Process

```
┌─────────────────────────────────────────────────────────────┐
│                 GPT Autoregressive Generation               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 1:                                                    │
│  Input: [The cat sat]                                       │
│  Output: [mat] (highest probability)                        │
│                                                             │
│  Step 2:                                                    │
│  Input: [The cat sat mat]                                   │
│  Output: [.]                                                │
│                                                             │
│  Step 3:                                                    │
│  Input: [The cat sat mat .]                                 │
│  Output: [</s>] (end of sequence)                           │
│                                                             │
│  Final Output: "The cat sat mat."                           │
│                                                             │
│  Sampling Strategies:                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. Greedy: argmax P(w_t | w_{<t})                   │   │
│  │ 2. Top-k: Sample from top k probable tokens         │   │
│  │ 3. Top-p (Nucleus): Sample from minimal set with    │   │
│  │    cumulative probability ≥ p                       │   │
│  │ 4. Temperature: P'(w) = softmax(logits/T)          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.6 Temperature Sampling 详解

Temperature scaling 公式：

$$P(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

其中：
- $z_i$：logit for token $w_i$
- $T$：temperature parameter
  - $T \rightarrow 0$：Greedy decoding
  - $T = 1$：Original distribution
  - $T \rightarrow \infty$：Uniform distribution

---

## 五、BERT vs GPT 深度对比

### 5.1 Architecture Level Comparison

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BERT vs GPT Architecture                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│         BERT (Encoder-only)           GPT (Decoder-only)           │
│         ──────────────────           ──────────────────            │
│                                                                     │
│     Input Layer                      Input Layer                   │
│     ────────────                     ────────────                  │
│     [CLS] A [SEP] B [SEP]            Token₁ Token₂ ... Tokenₙ      │
│       ↓                               ↓                            │
│     Token + Position                 Token + Position              │
│     + Segment Embedding              Embedding                     │
│       ↓                               ↓                            │
│                                                                     │
│     Encoder Layers                   Decoder Layers                │
│     ──────────────                   ──────────────                │
│     ┌─────────────────┐              ┌─────────────────┐          │
│     │ Self-Attention  │              │ Masked Self-    │          │
│     │ (Bidirectional) │              │ Attention       │          │
│     │                 │              │ (Unidirectional)│          │
│     │  Can see ALL    │              │  Can only see   │          │
│     │  positions      │              │  past positions │          │
│     └─────────────────┘              └─────────────────┘          │
│       ↓                               ↓                            │
│     Add & Norm                       Add & Norm                    │
│       ↓                               ↓                            │
│     FFN                              FFN                          │
│       ↓                               ↓                            │
│     Add & Norm                       Add & Norm                    │
│       ↓                               ↓                            │
│                                                                     │
│     Output Layer                     Output Layer                  │
│     ───────────────                  ───────────────               │
│     All hidden states                Next token probs              │
│     for downstream tasks             P(w_{t+1}|w_{≤t})            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Attention Mask 对比

**BERT 的 Attention Mask (Bidirectional):**

$$M_{BERT} = \begin{bmatrix} 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 \end{bmatrix}$$

每个 position 都可以 attend to 所有其他 positions（除了 padding positions）。

**GPT 的 Attention Mask (Causal):**

$$M_{GPT} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 1 & 1 & 0 & 0 \\ 1 & 1 & 1 & 0 \\ 1 & 1 & 1 & 1 \end{bmatrix}$$

Lower triangular matrix，position $i$ 只能 attend to positions $j \leq i$。

### 5.3 Pre-training Objectives 对比

```
┌─────────────────────────────────────────────────────────────────────┐
│              Pre-training Objectives Comparison                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  BERT: Masked Language Modeling (MLM)                               │
│  ─────────────────────────────────                                  │
│                                                                     │
│  Input:    The [MASK] sat on the [MASK]                            │
│  Target:            cat            mat                             │
│                                                                     │
│  Loss:     L = -log P(cat | context) - log P(mat | context)        │
│                                                                     │
│  Advantage: Bidirectional context understanding                     │
│  Disadvantage: [MASK] token mismatch at fine-tuning                │
│                                                                     │
│  ───────────────────────────────────────────────────────────────    │
│                                                                     │
│  GPT: Causal Language Modeling (CLM)                                │
│  ────────────────────────────────                                   │
│                                                                     │
│  Input:    The cat  sat  on   the                                   │
│  Target:   cat sat  on   the  mat                                   │
│                                                                     │
│  Loss:     L = -Σ log P(w_i | w_{<i})                              │
│                                                                     │
│  Advantage: Natural for generation, no mismatch                     │
│  Disadvantage: Only left context                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.4 Downstream Tasks Performance Comparison

| Task Type | BERT Strength | GPT Strength |
|-----------|---------------|--------------|
| Text Classification | Excellent (bidirectional context) | Good (can be fine-tuned) |
| Named Entity Recognition | Excellent | Moderate |
| Question Answering | Excellent (SQuAD) | Moderate |
| Sentiment Analysis | Excellent | Good |
| Text Generation | Poor | Excellent |
| Translation | Moderate | Good (especially with encoder) |
| Summarization | Moderate | Excellent |
| Code Generation | Poor | Excellent |

---

## 六、技术细节深度解析

### 6.1 Position Encoding 详解

**Sinusoidal Position Encoding (Original Transformer):**

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

其中：
- $pos$：position index in sequence
- $i$：dimension index
- $d_{model}$：model dimension

**Learned Position Embedding (BERT & GPT):**

$$PE = \text{Embedding}(pos, d_{model})$$

Position embedding 是一个 learnable parameter matrix $E \in \mathbb{R}^{L_{max} \times d_{model}}$，其中 $L_{max}$ 是 maximum sequence length。

### 6.2 Layer Normalization

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

其中：
- $\mu = \frac{1}{d}\sum_{i=1}^{d} x_i$：mean across hidden dimension
- $\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2$：variance
- $\gamma, \beta$：learnable scale and shift parameters
- $\epsilon$：small constant for numerical stability

**Pre-LN vs Post-LN:**

```
Post-LN (Original Transformer, BERT):
    x → Attention → Add → LN → FFN → Add → LN → output

Pre-LN (GPT-2 and later):
    x → LN → Attention → Add → LN → FFN → Add → output
```

Pre-LN 有更稳定的 training dynamics，避免了 gradient vanishing/exploding 问题。

### 6.3 Feed-Forward Network Details

$$\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$$

其中：
- $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$：first linear layer
- $W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$：second linear layer
- $d_{ff}$：通常为 $4 \times d_{model}$
- GELU：Gaussian Error Linear Unit

**GELU Activation Function:**

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}[1 + \text{erf}(x/\sqrt{2})]$$

其中 $\Phi(x)$ 是 standard Gaussian CDF。

Approximation：

$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{2/\pi}(x + 0.044715x^3)\right]\right)$$

### 6.4 Parameter Counting

**BERT-Base 参数计算：**

| Component | Formula | Count |
|-----------|---------|-------|
| Token Embedding | $V \times d_{model}$ | $30,000 \times 768 = 23.04M$ |
| Position Embedding | $L_{max} \times d_{model}$ | $512 \times 768 = 0.39M$ |
| Segment Embedding | $2 \times d_{model}$ | $2 \times 768 = 1.5K$ |
| Per Layer Attention | $4 \times d_{model}^2$ | $4 \times 768^2 = 2.36M$ |
| Per Layer FFN | $2 \times d_{model} \times d_{ff}$ | $2 \times 768 \times 3072 = 4.72M$ |
| Layer Norm per Layer | $2 \times 2 \times d_{model}$ | $4 \times 768 = 3K$ |

Total per layer ≈ 7.1M, 12 layers ≈ 85.2M
加上 embeddings ≈ 110M

### 6.5 Computational Complexity

**Self-Attention Complexity:**

$$\mathcal{O}(n^2 \cdot d_{model})$$

其中 $n$ 是 sequence length。这是 attention mechanism 的主要 bottleneck。

**FFN Complexity:**

$$\mathcal{O}(n \cdot d_{model} \cdot d_{ff}) = \mathcal{O}(n \cdot d_{model}^2)$$

---

## 七、Experimental Results

### 7.1 BERT GLUE Benchmark Results

| Model | MNLI | QQP | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE | Average |
|-------|------|-----|------|-------|------|-------|------|-----|---------|
| BERT-Base | 84.6 | 71.2 | 90.5 | 93.5 | 52.1 | 85.8 | 88.9 | 66.4 | 79.1 |
| BERT-Large | 86.7 | 72.1 | 92.7 | 94.9 | 60.5 | 86.5 | 89.3 | 70.1 | 81.6 |

### 7.2 GPT-3 Few-shot Learning Results

| Task | GPT-3 (175B) Few-shot | SOTA (Fine-tuned) |
|------|----------------------|-------------------|
| LAMBADA | 76.2% | 68.0% |
| TriviaQA | 64.3% | 68.0% |
| Natural Questions | 29.9% | 36.6% |
| WebQ | 41.5% | 45.0% |
| SuperGLUE | 71.8 | 83.7 |

### 7.3 Scaling Laws

**Kaplan et al. (2020) 发现的 scaling law:**

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}$$

其中：
- $L$：loss
- $N$：number of parameters
- $N_c, \alpha_N$：拟合得到的常数

这表明 model performance 可预测地随 scale 提升。

---

## 八、Key Innovations and Variants

### 8.1 RoBERTa (BERT 改进)

Key improvements over BERT:
1. Dynamic masking (每次 epoch 重新 mask)
2. More training data (16GB → 160GB)
3. Larger batch size (256 → 8000)
4. Longer training (100K → 500K steps)
5. Remove NSP objective

### 8.2 ALBERT (Parameter Efficient)

$$\text{Factorized Embedding} = E \times V \times H \rightarrow E \times V + E \times H$$

Cross-layer parameter sharing: 所有 layers share same parameters.

### 8.3 ELECTRA (Efficient Pre-training)

使用 generator-discriminator framework:

$$\mathcal{L}_{ELECTRA} = \mathcal{L}_{MLM}(G) + \lambda \mathcal{L}_{Disc}(D)$$

其中 discriminator 判断每个 token 是 original 还是 replaced。

### 8.4 GPT Variants

| Model | Key Innovation |
|-------|---------------|
| GPT-2 | Scale up, zero-shot learning |
| GPT-3 | Few-shot in-context learning |
| GPT-4 | Multimodal, RLHF, MoE |
| InstructGPT | RLHF for alignment |

---

## 九、First Principles Intuition Building

### 9.1 为什么 BERT 适合理解任务？

从信息论角度，BERT 的 bidirectional attention 实际上是在最大化：

$$I(X; Y) = H(X) - H(X|Y)$$

其中 $X$ 是 masked token，$Y$ 是 context。通过看到完整的 context，BERT 可以最大化 mutual information。

### 9.2 为什么 GPT 适合生成任务？

GPT 的 autoregressive objective 等价于最小化：

$$D_{KL}(P_{data} || P_{model})$$

这直接优化了 model distribution 与 data distribution 的匹配度，使得 sampling 可以产生 realistic sequences。

### 9.3 Attention 的本质

Attention 本质上是一个 **differentiable retrieval mechanism**：

$$\text{Attention}(q, K, V) = \sum_i \alpha_i v_i$$

其中 $\alpha_i = \text{softmax}(q \cdot k_i)$ 是 retrieval weight。这允许 model 学会 "query" relevant information from context。

---

## 十、Practical Considerations

### 10.1 Choosing Between BERT and GPT

```
┌─────────────────────────────────────────────────────────────┐
│            When to Use BERT vs GPT                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Choose BERT when:                                          │
│  ─────────────────                                          │
│  • Need deep understanding of input text                    │
│  • Classification, NER, QA tasks                            │
│  • Have labeled data for fine-tuning                        │
│  • Computation budget is limited                            │
│  • Sentence-level or token-level tasks                      │
│                                                             │
│  Choose GPT when:                                           │
│  ──────────────────                                         │
│  • Need text generation capability                          │
│  • Few-shot or zero-shot scenarios                          │
│  • Need creative/diverse outputs                            │
│  • Have access to large model APIs                          │
│  • Open-ended text completion tasks                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 10.2 Memory Optimization Techniques

| Technique | Description | Memory Reduction |
|-----------|-------------|------------------|
| Gradient Checkpointing | Store subset of activations | 50-70% |
| Mixed Precision | FP16 for forward, FP32 for gradients | ~50% |
| ZeRO | Shard optimizer states | Up to 8x |
| LoRA | Low-rank adaptation | ~90% for fine-tuning |
| Quantization | INT8/INT4 weights | 2-4x |

---

## 十一、Future Directions

### 11.1 Encoder-Decoder Models (T5, BART)

结合 BERT 和 GPT 的优势：

$$P(y_1, ..., y_m | x_1, ..., x_n) = \prod_{i=1}^{m} P(y_i | x_1, ..., x_n, y_1, ..., y_{i-1})$$

Encoder 处理 input (bidirectional)，Decoder 生成 output (autoregressive)。

### 11.2 Unified Models (UL2)

Unified Language Learning 结合了多种 pre-training objectives:

$$\mathcal{L}_{UL2} = \mathcal{L}_{AR} + \mathcal{L}_{MLM} + \mathcal{L}_{Prefix-LM}$$

### 11.3 Retrieval-Augmented Models

RAG 结合了 neural retrieval 和 language modeling:

$$P(y|x) = \sum_{z \in \text{Top-k}(p(z|x))} P(z|x) P(y|x, z)$$

---

## References

1. **BERT Original Paper**: Devlin, J., et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2018). https://arxiv.org/abs/1810.04805

2. **GPT-1 Paper**: Radford, A., et al. "Improving Language Understanding by Generative Pre-Training" (2018). https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

3. **GPT-2 Paper**: Radford, A., et al. "Language Models are Unsupervised Multitask Learners" (2019). https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

4. **GPT-3 Paper**: Brown, T., et al. "Language Models are Few-Shot Learners" (2020). https://arxiv.org/abs/2005.14165

5. **Transformer Paper**: Vaswani, A., et al. "Attention Is All You Need" (2017). https://arxiv.org/abs/1706.03762

6. **RoBERTa**: Liu, Y., et al. "RoBERTa: A Robustly Optimized BERT Pretraining Approach" (2019). https://arxiv.org/abs/1907.11692

7. **ALBERT**: Lan, Z., et al. "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations" (2019). https://arxiv.org/abs/1909.11942

8. **ELECTRA**: Clark, K., et al. "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators" (2020). https://arxiv.org/abs/2003.10555

9. **GPT-4 Technical Report**: OpenAI (2023). https://arxiv.org/abs/2303.08774

10. **InstructGPT**: Ouyang, L., et al. "Training language models to follow instructions with human feedback" (2022). https://arxiv.org/abs/2203.02155

11. **Scaling Laws**: Kaplan, J., et al. "Scaling Laws for Neural Language Models" (2020). https://arxiv.org/abs/2001.08361

12. **T5**: Raffel, C., et al. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (2019). https://arxiv.org/abs/1910.10683

13. **UL2**: Tay, Y., et al. "Unifying Language Learning Paradigms" (2022). https://arxiv.org/abs/2205.05131

14. **RAG**: Lewis, P., et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020). https://arxiv.org/abs/2005.11401