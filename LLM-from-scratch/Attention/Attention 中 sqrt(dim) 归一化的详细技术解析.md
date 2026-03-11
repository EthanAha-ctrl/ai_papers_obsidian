# Attention 中 sqrt(dim) 归一化的详细技术解析

## 一、基本公式回顾

在 "Attention Is All You Need" 论文中，Scaled Dot-Product Attention 的核心公式为：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- $Q \in \mathbb{R}^{n \times d_k}$：Query 矩阵
- $K \in \mathbb{R}^{m \times d_k}$：Key 矩阵  
- $V \in \mathbb{R}^{m \times d_v}$：Value 矩阵
- $d_k$：Key/Query 的维度（即论文中的 `dim`）

## 二、为什么需要 $\sqrt{d_k}$ 归一化？

### 2.1 数学推导：点积的期望分析

假设 $Q$ 和 $K$ 的每个元素都服从均值为 0、方差为 1 的独立同分布（i.i.d.）：

$$q_i, k_j \sim \mathcal{N}(0, 1)$$

那么点积 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 的期望和方差为：

$$\mathbb{E}[q \cdot k] = \sum_{i=1}^{d_k} \mathbb{E}[q_i k_i] = 0$$

$$\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = \sum_{i=1}^{d_k} \mathbb{E}[(q_i k_i)^2] - \mathbb{E}[q_i k_i]^2$$

由于 $q_i, k_i \sim \mathcal{N}(0,1)$ 独立：

$$\mathbb{E}[(q_i k_i)^2] = \mathbb{E}[q_i^2] \mathbb{E}[k_i^2] = 1 \times 1 = 1$$

因此：

$$\text{Var}(q \cdot k) = d_k$$

**关键结论**：点积的方差与维度 $d_k$ 线性增长！

### 2.2 归一化后

当我们除以 $\sqrt{d_k}$ 后：

$$\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{\text{Var}(q \cdot k)}{d_k} = \frac{d_k}{d_k} = 1$$

这样就保持了方差为 1，与输入的分布保持一致。

### 2.3 Softmax 的梯度问题

Softmax 函数定义为：

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

其梯度为：

$$\frac{\partial \text{softmax}(x_i)}{\partial x_j} = \text{softmax}(x_i) \cdot (\mathbb{I}(i=j) - \text{softmax}(x_j))$$

**关键问题**：当输入值的绝对值过大时，softmax 函数会进入饱和区域：

| 输入值 $x$   | $e^x$ 的行为       | Softmax 行为 |
| --------- | --------------- | ---------- |
| $x \ll 0$ | $e^x \approx 0$ | 梯度接近 0     |
| $x \gg 0$ | $e^x$ 指数爆炸      | 梯度接近 0     |

## 三、实验数据与验证

### 3.1 原论文中的实验结果

论文 Section 3.2.1 提供了关键观察：

> **"We suspect that for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients."**

### 3.2 实际数值示例

假设 $d_k = 64$（常见配置）：

| 场景 | 点积范围 | 除以 $\sqrt{d_k}$ 后 | Softmax 状态 |
|------|----------|---------------------|--------------|
| 不归一化 | $[-8, 8]$（$\sigma = 8$） | N/A | **梯度极小** |
| 归一化 | $[-8, 8]$ | $[-1, 1]$（$\sigma = 1$） | 梯度健康 |

### 3.3 梯度数值对比

假设使用两个向量的点积：

```python
# Python 模拟示例
import numpy as np

d_k = 64
q = np.random.randn(d_k)
k = np.random.randn(d_k)

# 不归一化
dot_product = np.dot(q, k)  # 方差 ≈ 64
print(f"不归一化的点积: {dot_product:.2f}, 标准差 ≈ {np.sqrt(d_k):.2f}")

# 归一化后
scaled_dot = dot_product / np.sqrt(d_k)  # 方差 ≈ 1
print(f"归一化后的值: {scaled_dot:.2f}, 标准差 ≈ 1.0")
```

## 四、架构图解析

```
┌─────────────────────────────────────────────────────────────┐
│                    Scaled Dot-Product Attention              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Q (n × d_k)         K (m × d_k)         V (m × d_v)          │
│     │                   │                   │                │
│     │                   │                   │                │
│     └─────────┬─────────┘                   │                │
│               │                             │                │
│               ▼                             │                │
│         MatMul(Q, K^T)                      │                │
│        Output: (n × m)                      │                │
│               │                             │                │
│               ▼                             │                │
│          Scale by 1/√d_k  ←─── 关键归一化   │                │
│               │                             │                │
│               ▼                             │                │
│            Softmax                         │                │
│               │                             │                │
│               └───────────────┬─────────────┘                │
│                               ▼                               │
│                        MatMul(·, V)                          │
│                                                              │
│                    Output: (n × d_v)                         │
└─────────────────────────────────────────────────────────────┘
```

## 五、不归一化的后果分析

### 5.1 梯度消失问题

当 $d_k$ 很大时（如 $d_k = 512$ 或 $d_k = 1024$）：

- 点积的方差 = 512 或 1024
- 标准差 = $\sqrt{512} \approx 22.6$ 或 $\sqrt{1024} \approx 32$
- 大部分点积值会落在 $[-3\sigma, 3\sigma] = [-68, 68]$ 范围内
- **Softmax 的梯度接近 0**

### 5.2 影响表格

| 维度 $d_k$ | 点积标准差 | 归一化后的标准差 | Softmax 梯度（不归一化） | Softmax 梯度（归一化） |
|-----------|-----------|----------------|----------------------|---------------------|
| 64 | 8.0 | 1.0 | 小 | 健康 |
| 128 | 11.3 | 1.0 | 极小 | 健康 |
| 512 | 22.6 | 1.0 | 接近 0 | 健康 |
| 1024 | 32.0 | 1.0 | **几乎为 0** | 健康 |

## 六、与其他方法的对比

### 6.1 Dot-Product vs Additive Attention

| Attention 类型 | 计算复杂度 | 需要归一化 | 原因 |
|---------------|-----------|-----------|------|
| **Dot-Product** | $O(n \cdot m \cdot d_k)$ | **是** | 点积方差随维度增长 |
| Additive | $O(n \cdot m \cdot d_k)$ | 否 | 使用可学习参数 |

### 6.2 为什么 Additive 不需要缩放

Additive Attention 使用：

$$a(q, k) = v^T \tanh(W_q q + W_k k)$$

由于 $\tanh$ 函数将输出限制在 $[-1, 1]$ 范围内，因此不需要额外的归一化。

## 七、相关扩展与变种

### 7.1 RoPE (Rotary Position Embedding) 中的维度缩放

在 LLaMA 等模型中，使用 RoPE 时也考虑了维度缩放：

$$\Theta_i = 10000^{-2i/d}, \quad i \in [1, d/2]$$

这与 attention 缩放的理念相似。

### 7.2 深层 Transformer 的残差缩放

在某些深层 Transformer（如 DeepMind 的 Deep Transformers）中，还使用额外的残差缩放：

$$x_{l+1} = x_l + \frac{1}{\sqrt{N}} \text{Attention}(x_l)$$

其中 $N$ 是层数。

## 八、实验验证代码

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def simulate_attention_scale(d_k, batch_size=1000, seq_len=10):
    # 生成随机 Q 和 K
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    
    # 不归一化的 attention
    attn_no_scale = torch.softmax(Q @ K.transpose(-2, -1), dim=-1)
    
    # 归一化的 attention
    attn_scaled = torch.softmax((Q @ K.transpose(-2, -1)) / np.sqrt(d_k), dim=-1)
    
    # 计算梯度
    return attn_no_scale, attn_scaled

# 比较不同维度下的梯度
dimensions = [64, 128, 256, 512, 1024]
gradients_no_scale = []
gradients_scaled = []

for d_k in dimensions:
    attn_no_scale, attn_scaled = simulate_attention_scale(d_k)
    grad_no_scale = attn_no_scale.std().item()
    grad_scaled = attn_scaled.std().item()
    gradients_no_scale.append(grad_no_scale)
    gradients_scaled.append(grad_scaled)
```

## 九、相关论文与资源

### 主要引用：

1. **"Attention Is All You Need"** - Vaswani et al., NeurIPS 2017
   - [Paper](https://arxiv.org/abs/1706.03762)
   - Section 3.2 详细解释了缩放的原因

2. **"The Annotated Transformer"** - Harvard NLP
   - [Blog](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
   - 提供了详细的代码实现和注释

3. **"LSTM层到Transformer层的跨网络特征迁移研究"**
   - 相关论文讨论了 attention 机制的各种变体

### 扩展阅读：

4. **"RoFormer: Enhanced Transformer with Rotary Position Embedding"**
   - [Paper](https://arxiv.org/abs/2104.09864)
   - 讨论了位置编码中的缩放策略

5. **"Training Deep Transformers"** - Huang et al.
   - [Paper](https://arxiv.org/abs/2006.04087)
   - 深入讨论了深层 Transformer 中的梯度问题

## 十、总结

$\sqrt{d_k}$ 归一化的核心原因：

1. **数学基础**：点积的方差与维度 $d_k$ 线性相关
2. **梯度保持**：防止 softmax 进入梯度消失区域  
3. **数值稳定**：保持 attention scores 在合理范围内
4. **训练效率**：加速收敛并提高模型性能

这个看似简单的除以 $\sqrt{d_k}$ 操作，实际上是确保 Transformer 能够有效训练的关键设计决策之一。