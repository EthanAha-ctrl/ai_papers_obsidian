## Transformer中的LayerNorm详解

### 1. LayerNorm的基本原理

#### 1.1 公式定义

LayerNorm对单个样本的所有特征进行归一化，公式如下：

$$\mu = \frac{1}{H}\sum_{i=1}^{H}x_i$$
$$\sigma^2 = \frac{1}{H}\sum_{i=1}^{H}(x_i - \mu)^2$$
$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
$$y_i = \gamma \hat{x}_i + \beta$$

其中：
- $H$ 是hidden dimension
- $x_i$ 是第i个特征维度的值
- $\gamma$ 和 $\beta$ 是可学习参数
- $\epsilon$ 是防止除零的小常数

#### 1.2 在Transformer中的两种应用方式

**Pre-LayerNorm (Post-Norm) vs Post-LayerNorm:**

```
Pre-LayerNorm架构:
Input → LN → Multi-Head Attention → Add → LN → FeedForward → Add → Output

Post-LayerNorm架构 (原始Transformer):
Input → Multi-Head Attention → Add → LN → FeedForward → Add → LN → Output
```

#### 1.3 代码实现（PyTorch）

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        mean = x.mean(dim=-1, keepdim=True)  # [batch_size, seq_len, 1]
        var = x.var(dim=-1, keepdim=True)   # [batch_size, seq_len, 1]
        
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
```

### 2. BatchNorm详解

#### 2.1 基本公式

BatchNorm对batch中所有样本的同一特征维度进行归一化：

$$\mu_B = \frac{1}{m}\sum_{i=1}^{m}x_i$$
$$\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_B)^2$$
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y_i = \gamma \hat{x}_i + \beta$$

其中 $m$ 是batch size。

#### 2.2 训练与推理阶段的区别

**训练阶段：**
- 使用当前batch的统计量
- 计算running mean和running variance

**推理阶段：**
- 使用训练期间积累的running statistics
- 不使用当前batch的统计量

```python
class BatchNorm1d(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.momentum = momentum
        self.eps = eps
    
    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0)
            
            # 更新running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
```

### 3. LayerNorm vs BatchNorm 核心区别

#### 3.1 归一化维度对比

| 维度 | BatchNorm | LayerNorm |
|------|-----------|-----------|
| 归一化方向 | 跨batch样本 | 跨特征维度 |
| 统计量依赖 | 依赖batch size | 不依赖batch size |
| 适用场景 | CV (CNN) | NLP (Transformer) |

#### 3.2 在不同batch size下的表现

**BatchNorm的问题：**
- 小batch size导致统计量不稳定
- 训练和推理分布不一致

**LayerNorm的优势：**
- 对每个样本独立归一化
- 不受batch size影响
- 特别适合variable length sequences

### 4. Transformer中使用LayerNorm的原因

#### 4.1 序列特性

NLP任务中：
- 输入是variable-length sequences
- 不同样本长度差异大
- batch size通常较小

#### 4.2 注意力机制特性

Self-attention输出：
```python
# attention计算后的输出分布不稳定
attention_output = softmax(Q @ K^T / sqrt(d_k)) @ V
```

LayerNorm可以：
1. 稳定attention输出分布
2. 加速梯度流动
3. 防止梯度消失/爆炸

#### 4.3 残差连接配合

```
output = LayerNorm(x + SubLayer(x))
```

残差连接与LayerNorm结合：
- 保持信息流动
- 平衡不同层级的输出scale

### 5. 实验数据和性能对比

#### 5.1 Pre-LN vs Post-LN性能

根据"On Layer Normalization in the Transformer Architecture"研究：

| 模型配置 | Post-LN | Pre-LN |
|----------|---------|--------|
| 6层Transformer | 需要warmup | 可以无warmup训练 |
| 深度模型(12+层) | 训练不稳定 | 训练稳定 |
| 收敛速度 | 较慢 | 较快 |

#### 5.2 梯度分析

Post-LN的梯度分布：
$$\frac{\partial \mathcal{L}}{\partial x^{(l)}} = \frac{\partial \mathcal{L}}{\partial \hat{x}^{(l)}} \cdot \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} \cdot \left(1 - \frac{1}{H} - \frac{(x - \mu)^2}{H(\sigma^2 + \epsilon)}\right)$$

深度网络中梯度需要通过多个LayerNorm层，导致梯度缩放问题。

### 6. 更多变体和改进

#### 6.1 RMSNorm (Root Mean Square Layer Normalization)

简化版LayerNorm，不需要计算均值：

$$\hat{x}_i = \frac{x_i}{\sqrt{\frac{1}{H}\sum_{j=1}^{H}x_j^2 + \epsilon}}$$

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    
    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x / rms)
```

#### 6.2 DeepNorm

针对深度网络的改进：

```
output = x / alpha + LayerNorm(x + alpha * SubLayer(x))
```

其中 alpha 是超参数，通常设置为 2^N (N是层数)

#### 6.3 GroupNorm

介于BatchNorm和LayerNorm之间：

```python
class GroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels):
        super().__init__()
        self.num_groups = num_groups
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))
    
    def forward(self, x):
        N, C, * = x.shape
        x = x.view(N, self.num_groups, -1)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + 1e-5)
        x = x.view(N, C, *).transpose(0, 1)
        return (self.gamma * x + self.beta).transpose(0, 1)
```

### 7. 详细的技术细节分析

#### 7.1 LayerNorm的梯度计算

给定 $y = \frac{x - \mu}{\sigma} \gamma + \beta$

对 $x$ 的梯度：

$$\frac{\partial y}{\partial x_i} = \frac{\gamma}{\sigma}\left(1 - \frac{1}{H} - \frac{(x_i - \mu)^2}{H\sigma^2}\right)$$

这个公式的关键洞察：
- 梯度依赖于当前样本的所有特征
- 项 $(1 - \frac{1}{H})$ 表示直接贡献
- 项 $-\frac{(x_i - \mu)^2}{H\sigma^2}$ 表示通过方差项的间接影响

#### 7.2 BatchNorm的梯度计算

$$\frac{\partial y}{\partial x_i} = \frac{\gamma}{\sigma_B}\left(1 - \frac{1}{m} - \frac{(x_i - \mu_B)^2}{m\sigma_B^2}\right)$$

与LayerNorm的区别在于 $H$ (特征维度) vs $m$ (batch size)

### 8. 参考资源

**论文：**
- [Layer Normalization (Ba et al., 2016)](https://arxiv.org/abs/1607.06450)
- [Batch Normalization (Ioffe & Szegedy, 2015)](https://arxiv.org/abs/1502.03167)
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
- [Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)](https://arxiv.org/abs/1910.07467)
- [DeepNet: Scaling Transformers to 1,000 Layers](https://arxiv.org/abs/2203.00555)

**代码实现：**
- [PyTorch LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)
- [PyTorch BatchNorm](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)
- [HuggingFace Transformer实现](https://github.com/huggingface/transformers)

**博客教程：**
- [The Transformer Family](https://lilianweng.github.io/posts/2020-04-07-the-transformer-family/)
- [Normalization Methods in Deep Learning](https://mlexplained.com/2018/11/30/an-overview-of-normalization-methods-in-deep-learning/)
- [Understanding Layer Normalization](https://stats.stackexchange.com/questions/474440/what-is-layer-normalization)

**可视化工具：**
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
- [Attention Visualization](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)