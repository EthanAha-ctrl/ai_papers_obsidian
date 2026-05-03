# Temporal Transformer 深度解析

## 一、核心概念与直觉理解

### 1.1 从第一性原理出发

**Temporal Transformer** 的本质是：**将时间序列建模为序列到序列的映射问题**，利用 Self-Attention 机制捕捉时间维度上的 Long-range dependencies。

传统方法的核心困境：
- **RNN/LSTM**：Sequential processing 导致无法并行化，且存在 vanishing/exploding gradient 问题
- **CNN**：Local receptive field 限制了 long-range dependencies 的建模能力

Temporal Transformer 的突破性在于：
$$\text{Temporal Dependency} = \text{Self-Attention}(\mathbf{X}_t, \mathbf{X}_{t-\Delta t}, \ldots, \mathbf{X}_{t-n\Delta t})$$

### 1.2 关键直觉

将时间序列视为 **ordered token sequence**，每个时间步作为一个 token，通过 Positional Encoding 注入时序信息：

$$\mathbf{Z}^{(0)} = \mathbf{X} + \mathbf{P}$$

其中：
- $\mathbf{X} \in \mathbb{R}^{T \times d}$：输入时间序列的 embedding
- $\mathbf{P} \in \mathbb{R}^{T \times d}$：Positional encoding matrix
- $T$：Sequence length (时间步数)
- $d$：Embedding dimension

---

## 二、架构详解

### 2.1 整体架构

```
Input Time Series
       ↓
   [Embedding Layer]
       ↓
+ [Positional Encoding]
       ↓
   [Temporal Transformer Encoder]
       ↓
   [Task-specific Head]
       ↓
    Output
```

### 2.2 Temporal Self-Attention 机制

**核心公式推导**：

给定输入序列 $\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T]^{\top}$，首先通过线性变换得到 Query, Key, Value：

$$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V$$

其中：
- $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d \times d_k}$：Learnable projection matrices
- $d_k$：Key/Query 的维度（通常 $d_k = d / h$，$h$ 为 attention heads 数量）

**Scaled Dot-Product Attention**：

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_k}}\right)\mathbf{V}$$

变量解析：
- $\mathbf{Q}\mathbf{K}^{\top} \in \mathbb{R}^{T \times T}$：Attention score matrix，表示每对时间步之间的相关性
- $\sqrt{d_k}$：Scaling factor，防止 dot product 数值过大导致 softmax gradient 消失
- Softmax 的输出 $\mathbf{A} \in \mathbb{R}^{T \times T}$：即 **Temporal Attention Map**

**时间维度的物理意义**：
$$A_{ij} = \frac{\exp(q_i^{\top}k_j / \sqrt{d_k})}{\sum_{j'=1}^{T} \exp(q_i^{\top}k_{j'} / \sqrt{d_k})}$$

$A_{ij}$ 表示时间步 $i$ 对时间步 $j$ 的关注程度，实现了 **adaptive temporal aggregation**。

### 2.3 Multi-Head Attention for Temporal Modeling

$$\text{MultiHead}(\mathbf{X}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\mathbf{W}^O$$

其中每个 head：
$$\text{head}_i = \text{Attention}(\mathbf{X}\mathbf{W}_i^Q, \mathbf{X}\mathbf{W}_i^K, \mathbf{X}\mathbf{W}_i^V)$$

**直觉理解**：
- 不同的 head 可以学习不同类型的 temporal patterns
- 例如：head₁ 关注 short-term dependencies，head₂ 关注 long-term trends，head₃ 关注 seasonal patterns

### 2.4 Positional Encoding 的时序特性

**Sinusoidal Positional Encoding**（原始 Transformer）：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

变量：
- $pos$：时间步位置（0 到 $T-1$）
- $i$：embedding 维度的索引
- $d$：embedding dimension

**Temporal Transformer 的改进方案**：

1. **Learnable Positional Embedding**：
$$\mathbf{P} = \text{Embedding}(pos)$$
直接学习位置向量，更灵活但需要更多数据

2. **Time Embedding**（针对连续时间）：
$$\mathbf{P}_t = \sum_{i=0}^{d/2-1} \left[\sin(\omega_i t), \cos(\omega_i t)\right]$$
其中 $\omega_i$ 是可学习的或预设的频率

3. **Temporal Convolutional Encoding**：
使用 1D CNN 提取局部时间特征作为 positional encoding

---

## 三、Temporal Transformer 的变体架构

### 3.1 Informer (AAAI 2021)

**核心创新**：ProbSparse Self-Attention

**动机**：Standard self-attention 的 $O(T^2)$ 复杂度在长序列上不可行

**ProbSparse Attention 公式**：

$$\mathcal{A}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{SoftMax}\left(\frac{\bar{\mathbf{Q}}\mathbf{K}^{\top}}{\sqrt{d_k}}\right)\mathbf{V}$$

其中 $\bar{\mathbf{Q}}$ 是从 $\mathbf{Q}$ 中采样的 **top-u queries**，采样依据是 **Query Sparsity Measurement**：

$$M(q_i, \mathbf{K}) = \max_j \left(\frac{q_i k_j^{\top}}{\sqrt{d_k}}\right) - \frac{1}{L_K} \sum_{j=1}^{L_K} \frac{q_i k_j^{\top}}{\sqrt{d_k}}$$

**直觉**：
- 高 sparsity score 的 query 更"active"，需要更多关注
- 只计算这些 query 的 attention，复杂度降为 $O(T \log T)$

### 3.2 Autoformer (NeurIPS 2021)

**核心创新**：Series Decomposition + Auto-Correlation

**Decomposition Block**：
$$\mathbf{X} = \mathcal{T}(\mathbf{X}) + \mathcal{S}(\mathbf{X})$$
- $\mathcal{T}(\cdot)$：Trend component（通过 moving average）
- $\mathcal{S}(\cdot)$：Seasonal component（$\mathbf{X} - \mathcal{T}(\mathbf{X})$）

**Auto-Correlation 机制**：

替代 self-attention，利用时间序列的 **periodicity**：

$$\text{AutoCorrelation}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \sum_{i=1}^{k} \text{Roll}(\mathbf{V}, \tau_i) \cdot \text{SoftMax}(R_{\mathbf{Q}, \mathbf{K}}(\tau_i))$$

其中：
- $R_{\mathbf{Q}, \mathbf{K}}(\tau)$：Autocorrelation at lag $\tau$
- $\text{Roll}(\cdot, \tau)$：Time delay operation
- $\tau_i$：Top-$k$ delays with highest autocorrelation

### 3.3 PatchTST (ICLR 2023)

**核心创新**：Patching + Channel Independence

**Patching**：

将时间序列分割为 non-overlapping 或 overlapping patches：
$$\mathbf{X}_{patch} = [\mathbf{x}_1^p, \mathbf{x}_2^p, \ldots, \mathbf{x}_N^p]$$
其中每个 patch $\mathbf{x}_i^p \in \mathbb{R}^{P}$ 包含 $P$ 个时间步

**公式**：
$$\mathbf{Z}_0 = \mathbf{W}_p \mathbf{X}_{patch} + \mathbf{P}$$

- $\mathbf{W}_p \in \mathbb{R}^{d \times P}$：Patch embedding matrix
- $N = \lfloor (T - P) / S \rfloor + 1$：Patch 数量
- $S$：Stride（步长）

**优势**：
- 降低序列长度，复杂度从 $O(T^2)$ 降为 $O(N^2)$
- 每个 patch 作为局部语义单元，类似 Vision Transformer

### 3.4 iTransformer (ICLR 2024)

**核心创新**：Inverted Transformer

**核心思想**：将 **变量维度视为序列长度**，时间步作为 embedding 维度

传统做法：
$$\text{Input}: \mathbf{X} \in \mathbb{R}^{T \times C} \rightarrow \text{Embedding}: \mathbb{R}^{T \times d}$$

iTransformer：
$$\text{Input}: \mathbf{X}^{\top} \in \mathbb{R}^{C \times T} \rightarrow \text{Embedding}: \mathbb{R}^{C \times d}$$

**Attention 计算**：
$$\mathbf{Q}, \mathbf{K}, \mathbf{V} = \mathbf{X}^{\top}\mathbf{W}_Q, \mathbf{X}^{\top}\mathbf{W}_K, \mathbf{X}^{\top}\mathbf{W}_V$$

**物理意义**：Attention map $\mathbf{A} \in \mathbb{R}^{C \times C}$ 表示不同变量之间的相关性

---

## 四、时间序列特有的设计考量

### 4.1 Temporal Embedding 的丰富设计

**Time Features Encoding**（多粒度时间特征）：

$$\mathbf{P}_{time} = \mathbf{W}_h \mathbf{E}_h + \mathbf{W}_d \mathbf{E}_d + \mathbf{W}_w \mathbf{E}_w + \mathbf{W}_m \mathbf{E}_m$$

其中：
- $\mathbf{E}_h$：Hour-of-day embedding (24 classes)
- $\mathbf{E}_d$：Day-of-week embedding (7 classes)
- $\mathbf{E}_w$：Week-of-year embedding (52 classes)
- $\mathbf{E}_m$：Month-of-year embedding (12 classes)

### 4.2 Decoder 设计：Autoregressive vs. One-shot

**Autoregressive Decoding**（标准 Transformer）：
$$P(\mathbf{Y}|\mathbf{X}) = \prod_{t=1}^{T_y} P(y_t|\mathbf{X}, \mathbf{y}_{<t})$$

**One-shot Decoding**（Informer, Autoformer）：
$$\hat{\mathbf{Y}} = f_{decoder}(\mathbf{Z}_{enc}, \mathbf{Y}_{start})$$

其中 $\mathbf{Y}_{start}$ 是 start token + placeholder sequences

### 4.3 长序列预测的处理策略

**Generation Length 的问题**：

假设预测 horizon 为 $L$，传统方法需要 $L$ 次解码步骤

**解决方案**：

1. **Generative Style**（Informer）：
$$\mathbf{Y}_{prob} = [\mathbf{y}_0, \mathbf{y}_1, \ldots, \mathbf{y}_L]$$
其中 $\mathbf{y}_0$ 是 start token，其他是 placeholder

2. **Direct Multi-step Forecasting**：
$$\hat{\mathbf{Y}} = \mathbf{W}_{out} \mathbf{Z}_T$$
直接预测整个输出序列

---

## 五、实验数据与性能分析

### 5.1 Benchmark 数据集

| Dataset | Domain | Series Length | Variables | Prediction Horizons |
|---------|--------|---------------|-----------|---------------------|
| ETTh1/ETTh2 | Electricity | 17,420 | 7 | 24, 48, 168, 336, 720 |
| ETTm1/ETTm2 | Electricity | 69,680 | 7 | 24, 48, 96, 288, 672 |
| Electricity | Power | 26,304 | 321 | 24, 48, 96, 192, 336, 720 |
| Traffic | Transport | 17,544 | 862 | 24, 48, 96, 192, 336, 720 |
| Weather | Meteorology | 52,696 | 21 | 24, 48, 96, 192, 336, 720 |
| Exchange | Finance | 7,581 | 8 | 24, 48, 96, 192, 336, 720 |
| ILI | Disease | 966 | 7 | 24, 36, 48, 60 |

### 5.2 性能对比表（MSE / MAE）

**ETTh1 数据集（低维多变量）**：

| Model | 24 | 96 | 192 | 336 | 720 |
|-------|-----|------|------|------|------|
| LSTM | 0.296/0.372 | 0.450/0.458 | 0.540/0.508 | 0.591/0.540 | 0.650/0.582 |
| Transformer | 0.283/0.361 | 0.432/0.449 | 0.512/0.492 | 0.562/0.526 | 0.621/0.567 |
| Informer | 0.246/0.332 | 0.374/0.402 | 0.453/0.452 | 0.498/0.471 | 0.559/0.516 |
| Autoformer | 0.226/0.314 | 0.338/0.375 | 0.406/0.420 | 0.453/0.437 | 0.504/0.482 |
| PatchTST | 0.215/0.298 | 0.312/0.350 | 0.372/0.389 | 0.421/0.408 | 0.472/0.454 |
| iTransformer | **0.198/0.281** | **0.298/0.338** | **0.358/0.375** | **0.402/0.392** | **0.458/0.438** |

**Electricity 数据集（高维多变量，321 变量）**：

| Model | 24 | 96 | 192 | 336 | 720 |
|-------|-----|------|------|------|------|
| Transformer | 0.245/0.268 | 0.342/0.312 | 0.392/0.336 | 0.442/0.362 | 0.512/0.408 |
| Informer | 0.218/0.242 | 0.305/0.285 | 0.352/0.312 | 0.408/0.348 | 0.478/0.392 |
| Autoformer | 0.198/0.225 | 0.278/0.262 | 0.325/0.295 | 0.378/0.325 | 0.448/0.372 |
| PatchTST | 0.178/0.205 | 0.252/0.238 | 0.298/0.272 | 0.348/0.302 | 0.418/0.348 |
| iTransformer | **0.162/0.192** | **0.232/0.218** | **0.278/0.255** | **0.328/0.285** | **0.398/0.332** |

### 5.3 复杂度分析

| Model | Time Complexity | Space Complexity | 特点 |
|-------|-----------------|------------------|------|
| Vanilla Transformer | $O(T^2 d)$ | $O(T^2 + Td)$ | 精确但昂贵 |
| Informer | $O(T \log T \cdot d)$ | $O(T \log T + Td)$ | ProbSparse sampling |
| Autoformer | $O(T \log T \cdot d)$ | $O(T \log T + Td)$ | Auto-correlation |
| PatchTST | $O((T/P)^2 d)$ | $O((T/P)^2 + Td)$ | Patching reduces length |
| iTransformer | $O(C^2 T)$ | $O(C^2 + CT)$ | Variable-wise attention |

---

## 六、关键变体与扩展

### 6.1 Temporal Fusion Transformer (TFT)

**核心特点**：Interpretable Multi-horizon Forecasting

**架构组件**：
1. **Variable Selection Network**：学习输入变量的重要性
2. **Static Covariate Encoders**：处理静态特征
3. **Gated Residual Network (GRN)**：非线性特征提取
4. **Multi-head Attention**：Temporal dynamics

**Variable Selection 公式**：
$$\zeta_t = \text{SoftMax}(\text{GRN}(\mathbf{e}_t, \mathbf{c}_s))$$
$$\mathbf{x}_t^{selected} = \sum_{j=1}^{n_x} \zeta_{t,j} \cdot \tilde{\mathbf{x}}_{t,j}$$

其中 $\mathbf{c}_s$ 是 static context，$\zeta_t$ 是变量选择权重

### 6.2 FEDformer (ICML 2022)

**核心创新**：Frequency Enhanced Decomposition

**思想**：在频域进行 attention

$$\mathbf{X}^{freq} = \mathcal{F}(\mathbf{X})$$
$$\mathbf{Z}^{freq} = \text{Attention}_{freq}(\mathbf{X}^{freq})$$
$$\mathbf{Z} = \mathcal{F}^{-1}(\mathbf{Z}^{freq})$$

**优势**：
- 频域操作的复杂度可降低至 $O(T \log T)$
- 自然捕捉 periodic patterns

### 6.3 Pyraformer (ICLR 2022)

**核心创新**：Pyramidal Attention

**思想**：构建多分辨率 temporal pyramid

**复杂度**：$O(T)$ linear complexity

**结构**：
$$\mathbf{Z}^{(l)} = \text{C-Attention}(\mathbf{Z}^{(l-1)})$$
- $l$：pyramid level
- 越高层，分辨率越粗，覆盖 longer range

---

## 七、代码实现要点

### 7.1 标准 Temporal Transformer Block

```python
class TemporalTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        # x: [seq_len, batch, d_model]
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        # FFN with residual
        x = self.norm2(x + self.ffn(x))
        return x
```

### 7.2 Temporal Positional Encoding

```python
class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [seq_len, batch, d_model]
        x = x + self.pe[:x.size(0), :].unsqueeze(1)
        return self.dropout(x)
```

---

## 八、应用场景与实际考量

### 8.1 典型应用领域

| 领域 | 任务 | 输入 | 输出 | 挑战 |
|------|------|------|------|------|
| Energy | Power forecasting | Historical load, weather | Future demand | Seasonality, exogenous factors |
| Finance | Stock prediction | Price, volume, news | Price trend | Volatility, regime changes |
| Traffic | Flow prediction | Sensor readings | Traffic flow | Spatial-temporal coupling |
| Healthcare | Patient monitoring | Vital signs | Risk score | Irregular sampling |
| Industry | Predictive maintenance | Sensor data | Failure prediction | Anomaly rarity |
| Climate | Weather forecasting | Meteorological data | Future conditions | Long horizon |

### 8.2 实际部署考量

**Memory Efficiency**：
- Gradient checkpointing
- Mixed precision training (FP16/BF16)
- Model parallelism for long sequences

**Data Processing**：
- **Normalization**：通常使用 instance normalization
  $$\tilde{x}_t = \frac{x_t - \mu}{\sigma}$$
- **Missing value handling**：Masked attention

**Hyperparameter Guidelines**：
- Embedding dimension: 64-512 (取决于数据复杂度)
- Number of heads: 4-16
- Number of layers: 2-6 (过深易 overfitting)
- Dropout: 0.1-0.3

---

## 九、研究前沿与开放问题

### 9.1 当前研究方向

1. **Efficient Attention Mechanisms**
   - Linear attention: Performer, Linear Transformer
   - Sparse attention: Longformer, BigBird
   - Low-rank approximation: Nystromformer

2. **Multi-scale Temporal Modeling**
   - Wavelet-based decomposition
   - Multi-resolution Transformers

3. **Continuous-time Modeling**
   - Neural ODE + Transformer
   - Irregular time series handling

4. **Interpretability**
   - Attention visualization
   - Variable importance analysis
   - Temporal contribution decomposition

### 9.2 开放挑战

| Challenge | Description | Current Approaches |
|-----------|-------------|-------------------|
| Long-term dependencies | $O(T^2)$ complexity | Sparse attention, patching |
| Distribution shift | Train-test mismatch | Adaptive normalization |
| Multivariate coupling | Cross-variable dependencies | Cross-attention, graph modeling |
| Uncertainty quantification | Probabilistic forecasting | Variational approaches, ensemble |
| Domain adaptation | Transfer across domains | Pre-training, fine-tuning |

---

## 十、总结：构建 Intuition 的关键要点

### 10.1 核心直觉

1. **Temporal Transformer = Learnable Temporal Correlation Matrix**
   - Self-attention 学习的是 $T \times T$ 的相关性矩阵
   - 每个 $A_{ij}$ 表示时间步 $i$ 和 $j$ 之间的依赖强度
   - 这是 **data-dependent** 的，不同于固定的 correlation

2. **为什么比 RNN 更好？**
   - **并行化**：所有时间步同时处理
   - **Long-range**：没有 gradient vanishing 问题
   - **灵活性**：可以关注任意历史时间步

3. **为什么比 CNN 更好？**
   - **Global receptive field**：每个时间步可以"看到"整个序列
   - **Adaptive**：相关性权重是动态学习的

### 10.2 设计选择指南

| 场景 | 推荐模型 | 原因 |
|------|----------|------|
| Short sequences (< 200) | Vanilla Transformer | 精确、实现简单 |
| Long sequences (200-2000) | Informer / Autoformer | 复杂度优化 |
| Very long sequences (> 2000) | PatchTST | Patching 大幅降低序列长度 |
| High-dimensional multivariate | iTransformer | Variable-wise attention |
| Need interpretability | TFT | Built-in variable selection |

### 10.3 与其他时间序列方法的对比

| Method | Long-range | Parallelizable | Interpretable | Multi-var |
|--------|-----------|---------------|---------------|-----------|
| ARIMA | ❌ | ❌ | ✅ | ❌ |
| RNN/LSTM | ⚠️ | ❌ | ❌ | ⚠️ |
| TCN | ⚠️ | ✅ | ⚠️ | ✅ |
| Temporal Transformer | ✅ | ✅ | ⚠️ | ✅ |
| N-BEATS | ✅ | ✅ | ✅ | ❌ |

---

## 参考文献与资源

### 核心论文

1. **Vaswani et al. (2017)** - "Attention Is All You Need" - [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

2. **Zhou et al. (2021)** - "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting" (AAAI 2021) - [arXiv:2012.07436](https://arxiv.org/abs/2012.07436)

3. **Wu et al. (2021)** - "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting" (NeurIPS 2021) - [arXiv:2106.13008](https://arxiv.org/abs/2106.13008)

4. **Nie et al. (2023)** - "A Time Series is Worth 64 Words: Long-Term Forecasting with Transformers" (ICLR 2023) - [arXiv:2211.14730](https://arxiv.org/abs/2211.14730)

5. **Liu et al. (2024)** - "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting" (ICLR 2024) - [arXiv:2310.06625](https://arxiv.org/abs/2310.06625)

6. **Lim et al. (2021)** - "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" (Int. J. Forecast.) - [arXiv:1912.09363](https://arxiv.org/abs/1912.09363)

7. **Zhou et al. (2022)** - "FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting" (ICML 2022) - [arXiv:2201.12740](https://arxiv.org/abs/2201.12740)

8. **Liu et al. (2022)** - "Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling and Forecasting" (ICLR 2022) - [arXiv:2201.08661](https://arxiv.org/abs/2201.08661)

### 开源代码库

- **Time-Series Library**: https://github.com/thuml/Time-Series-Library
- **Informer**: https://github.com/zhouhaides/Informer2020
- **Autoformer**: https://github.com/thuml/Autoformer
- **PatchTST**: https://github.com/yuqinie98/PatchTST
- **iTransformer**: https://github.com/thuml/iTransformer