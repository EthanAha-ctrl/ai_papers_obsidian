我来详细读取这篇论文的内容，为你提供深入的技术讲解。












# Anytime Pretraining: Horizon-Free Learning-Rate Schedules with Weight Averaging 论文详解

## 一、问题背景与动机

### 1.1 当前的LLM预训练范式问题

大语言模型（LLM）正在越来越多地在连续或开放式设置中训练，**其中总训练horizon（训练时长/步数）不是预先知道的**。然而：

- 现有的预训练recipes大多数**不是anytime的**：它们依赖于**horizon-dependent的学习率调度**和在固定计算预算下的广泛调整
- **Cosine decay**自Loshchilov和Hutter（2016）提出以来，已成为LLM预训练的de facto学习率调度器
- **Cosine decay的致命缺陷**：它是一个horizon-dependent调度器，因为需要提前知道训练运行的长度，这使得它不适用于持续学习设置

### 1.2 Cosine Decay的问题

从Figure 1可以看到：
- **Cosine envelope（红色曲线）**：为每个终端计算预算（1×-32× Chinchilla）独立调整horizon-aware cosine schedule，取该horizon下的最佳验证loss形成
- **灰色曲线**：为单一固定终端预算调整的同一cosine schedule在中间checkpoint评估时的表现
- **差距说明**：为什么cosine decay不是anytime的：为长horizon调整在较短预算下可能远非最优

**关键洞察**：标准训练recipes在不知道停止时间时不会产生anytime程序，因为horizon的选择**隐式地决定了整个loss轨迹**。

### 1.3 现有替代方案

#### Warmup-Stable-Decay (WSD)
- 由Hu等（2024）和Wen等（2024）引入
- 包含3个阶段：
  1. 线性学习率warm-up阶段达到最大学习率
  2. 保持学习率不变直到训练长度的90%
  3. 在剩余10%内衰减学习率
- **不是严格anytime的**：需要以稀疏间隔存储训练运行的checkpoints
- 已被前沿实验室（如Team等，2025）采用

#### Weight Averaging（Model Merging）
- 也称为model merging（Li等，2025）
- 涉及维护最近比例迭代的平均值：
  - 显式地：对最后N个迭代进行平均
  - 通过指数移动平均（EMA）：由参数β控制，β ≈ 1 - 1/N
- 使用平均模型进行评估，训练时仍仅使用最后迭代
- 已被用于schedule-free算法（Defazio等，2024；Morwani等，2025）和作为学习率衰减的替代方案

## 二、核心贡献与理论分析

### 2.1 主要理论结果（Theorem 1非正式版本）

**定理**：对于在N个样本上运行的SGD过程，形式为ηₜ = 1/t^γ（0 < γ < 1）的多项式衰减学习率配合尾部平均，能够匹配良好调整的SGD（具有平均）的收敛率，其中指数γ取决于数据的谱特性。

**数学表达**：
```
ER( ̄w) - σ² ≲
1/N ‖w*‖²_Λ_{1:k*} + ‖w*‖²_Λ_{k*:∞} +
k*σ²/(ηN) + σ² Σ_{k>k*}(ηλ²ₖN^(1-2γ) + λₖN^(-γ))
```

其中：
- **R( ̄w)**：平均参数 ̄w的风险
- **σ²**：噪声方差
- **w***：最优权重（最小化器）
- **‖w*‖²_Λ_{1:k*}**：在头部特征空间（前k*个特征）中的目标向量范数
- **‖w*‖²_Λ_{k*:∞}**：在尾部特征空间中的目标向量范数
- **k***：分离特征的特征索引
- **Λ = diag(λᵢ)**：数据协方差H的特征值对角矩阵
- **λₖ**：第k个特征值
- **ηₜ = η/t^γ**：学习率调度
- **N**：总样本数

**特征阈值k*的定义**：
```
k*: = max{k: λₖ ≥ logN/(ηs^(1-γ))}
```

### 2.2 Bias-Variance分解

论文使用bias-variance分解来分析SGD的收敛性：

**SGD更新规则**：
```
w_{t+1} = w_t - ηₜ x_t (xᵀ_t w_t - y_t)
y_t = xᵀ_t w* + ε_t
```

其中：
- **x_t**：第t步的输入样本，x ~ N(0, H)
- **y_t**：第t步的标签，带有噪声ε_t ~ N(0, σ²)
- **ε_t**：独立噪声

**减去最优权重w***：
```
w_{t+1} - w* = (I - ηₜ x_t xᵀ_t)(w_t - w*) + ηₜ ε_t x_t
```

**协方差递归**：
```
Σ_{t+1} = E[(w_{t+1} - w*)(w_{t+1} - w*)ᵀ]
       = Σ_t - ηₜ Σ_t H - ηₜ H Σ_t + 2ηₜ² H Σ_t H + ηₜ² Tr(HΣ_t)H + ηₜ² σ² H
```

**旋转到Q特征基后的对角递归**：
```
m_t = (I - 2ηₜ Λ + Λ² + λλᵀ)m_{t-1} + σ² ηₜ² λ
     ≤ (I - ηₜ Λ)² + cσ² ηₜ² λ
```

其中**Assumption 1**：存在常数c > 1，使得对于任何时间t > 0，有R(w_t) ≤ cσ²

**偏差和方差迭代器的定义**：
```
 ̃m_{t+1}: = exp[-2ηΛ t^(1-γ)] m_0
m_{t+1}:   = ησ² Σ_{p=0}^t (1/p^(2γ)) exp[-2ηΛ(t^(1-γ) - p^(1-γ))] λ
```

#### 2.2.1 偏差界

**Bias bound（公式7）**：
```
bias_{t+1} ≲ (1/N)‖w*‖²_Λ_{1:k*} + ‖w*‖²_Λ_{k*:∞}
```

**推导思路**：
1. 头部特征（k ≤ k*）：使用exp[-ηλₖ j^(1-γ)]的衰减
2. 尾部特征（k > k*）：直接用1上界

#### 2.2.2 方差异

**Variance bound（公式9）**：
```
variance_{t+1} ≲ σ² η(s+N)^γ k* / (Ns^γ) + σ² Σ_{k>k*}(ηλ²ₖ s^(1-2γ) + λₖ s^(-γ))
```

**头部方差**（k ≤ k*）：
```
variance_{1:k*} ≲ σ² Σ_{k≤k*} (s+N)^γ/(ηN²) Σ_{i=s}^{s+N-1} i^(-γ)
               ≲ σ² η(s+N)^γ k* / (Ns^γ)
```

**尾部方差**（k > k*）：
```
variance_{k*:∞} ≲ σ² Σ_{k>k*}(ηλ²ₖ s^(1-2γ) + λₖ s^(-γ))
```

### 2.3 功率律谱设置（Corollary 1）

**容量和源指数的定义**：
```
λᵢ ≂ i^(-a)    （capacity exponent, a > 1）
E[λᵢ(w*_i)²] ≂ i^(-b)   （source exponent, b > 1）
```

**最优γ的选择**：
```
γ* = max{1 - a/b, 0}
```

**最优率**：
```
R(w) - σ² ≲ (σ²/N)^{1 - 1/b}
```

**解读**：
- **b量化了目标向量w*中的信号在数据协方差H的特征向量中的分布**
- **更高的b值**：信号主要位于H的前几个方向
- **b ≥ a**：建立的无穷维minimax最优率
- **b ≫ a > 1**：有效维度较低，接近强凸情况，最优调度器是1/t配合平均
- **b ≤ a**：信号主要在尾部方向，需要较大的学习率来衰减偏差，常数学习率配合平均是最优调度器

### 2.4 WSD的理论分析

**Theorem 2**：
考虑t₀ = ρN，其中ρ ∈ (0, 1)。假设H上的功率律谱，capacity exponent a ∈ (1, 2)，source exponent b > 1。考虑两阶段学习率调度：

```
ηₜ = {
    η              1 ≤ t ≤ t₀
    η(1 - (t-t₀)/(N-t₀))  t₀ < t ≤ N
}
```

**超出风险界**：
```
R(w_N) - σ² ≲ (1/N)^{b/a - 1/a} + σ²(1/N)^{1 - 1/a}
```

**分析**：
- **b > a**：方差项主导，信号主要在前特征方向，可以在有限步内拟合偏差
- **b < a**：偏差项主导，信号主要在后特征方向，需要较大的学习率来拟合它们
- **b = a**：两项以指数1 - 1/b衰减（至对数因子），匹配Corollary 1中的率

### 2.5 常数学习率

**有限维情况**：
- 常数学习率配合尾部平均可以实现方差项的minimax率：σ² d/N

**高维情况**：
- b ≤ a时，常数学习率配合平均是真正的anytime调度器
- b > a时，学习率必须缩放为1/√N才能达到minimax率（不是anytime方案）

## 三、实验设置与结果

### 3.1 模型和数据集

**模型架构**（基于OLMo代码库）：
- **150M模型**：depth=12, heads=16, width=1024
- **300M模型**：depth=24, heads=16, width=1024

**优化器设置**：
```
AdamW, 无weight decay
ε = 10^(-8)
β₁ = 0.9
β₂ ∈ {0.95, 0.98, 0.99} (150M)
β₂ ∈ {0.95, 0.99} (300M)
```

**学习率搜索空间**：
- 150M: η ∈ {0.0001, 0.0003, 0.001, 0.003, 0.01}
- 300M: η ∈ {0.0003, 0.001, 0.003}

**数据集**：C4，T5 tokenizer，序列长度L = 1024，无数据重复

### 3.2 训练设置

**Chinchilla计算预算**：
- 150M模型：3.3B tokens（20 tokens-per-parameter）
- 300M模型：6.6B tokens

**Critical Batch Size（CBS）**：
- 150M模型：256
- 300M模型：512

**Warmup持续时间**：各模型1× Chinchilla持续时间的40%

### 3.3 Weight Averaging实现

**EMA更新**：
```
 ̄w_{t+1} = (1 - τₜ) ̄w_t + τₜ θ_t
τₜ = 1/2^{f/t}
```

**半生命期设计**：
- 在时间t，EMA平均约最后1/f个迭代
- 维护多个EMA，f ∈ {0.0, 6.25, 12.5, 25.0, 50.0, 100.0}
- f = 0.0：仅使用最后迭代，无平均

### 3.4 实验结果（Figure 2）

#### 150M模型（1× - 32× Chinchilla）：
- **Const. + Avg.**：常数学习率配合平均，在大部分训练范围内与cosine相当
- **1/√t**：在长训练 regime 中表现良好
- **WSD**：与cosine竞争力强，但不是严格horizon-free

#### 300M模型（1× - 16× Chinchilla）：
- **Const. + Avg.**：在8×及之后略优于cosine
- **1/√t**：与cosine接近
- **WSD**：与cosine竞争力强

**关键观察**：
- 在所有中间点（包括非常长的训练regime），anytime方法紧密匹配cosine annealing
- 仅在训练开始和结束时付出微小性能代价
- Anytime方法可以来自单一运行（训练至32×），而cosine需要在每个计算预算下单独训练

### 3.5 大Batch设置（Figure 3）

**Batch size = 4096**（远超CBS）：

**关键洞察**：
- 在大batch regime，常数学习率配合平均在1× Chinchilla之外的所有horizon上**显著优于cosine**
- 在长训练中接近最优的学习率在整个训练过程中保持接近最优（不像CBS设置需要权衡短运行和长运行性能）
- 1/√t schedule保持竞争力并改进cosine，但表现不如常数（大batch中不必要的衰减）

**理论解释**：
在quadratic regime，SGD可以写成GD加上batch噪声项：
```
θ_{t+1} - θ* ≈ (I - ηH)(θ_t - θ*) + ηξ_t
E[ξ_t] = 0, Cov(ξ_t) ∝ 1/B
```

移动到非常大的batch抑制梯度噪声，使得学习率衰减对于控制方差项不再必要。

### 3.6 线性回归理论验证（Figure 4）

**设置**：
- 问题维度：d = 500,000
- 最大样本数：N = 50,000
- Batch size = 1
- 标签噪声：σ² = 0.01

**源指数和容量指数组合**：
- a = 1.1, b = 1.1（b = a情况）
- a = 1.5, b = 1.5（b = a情况）
- a = 1.9, b = 1.9（b = a情况）
- a = 1.1, b = 2.2（b = 2a情况）
- a = 1.5, b = 3.0（b = 2a情况）
- a = 1.9, b = 3.8（b = 2a情况）

**学习率搜索**：
η ∈ {0.0001, 0.0002, 0.0005, 0.0007, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0}

**观察**：
- b = a情况：Const. + Avg. 和1/√t性能相似
- b = 2a情况：1/√t明显优于Const. + Avg.
- WSD在两种情况下表现竞争力强

### 3.7 γ的选择（γ = 1/2的动机）

**Mlodozeniec等（2025）和Bjorck等（2024）的发现**：
实践中最优学习率近似地按1/√N缩放。

**Quadratic regime中的偏差收缩**：
偏差沿特征方向i的收缩率由累积步长控制，近似为：
```
exp(-λᵢ Σ_{s≤t} ηₛ)
```

**持续做实质性进展的条件**：
```
λᵢ N / Σ_{s=1}^N ηₛ ≳ 1 ⇒ λᵢ ≳ 1/Σ_{s=1}^N ηₛ
```

对于ηₛ ≂ 1/√N：
```
Σ_{s=1}^N ηₛ ≂ √N
```

为确保anytime方案在任何时间t有相同的缩放，合理的选择是：
```
ηₛ = 1/√s
```

因为：
```
Σ_{s=1}^t s^(-1/2) ≂ √t
```

**实践参数化**：
```
ηₜ = √α/(t + α)
α ∈ {400, 800, 1600, 3200, 6400, 12800, 25600, 51200}
```

虽然α引入了对总步数的依赖（因此隐式依赖于时间horizon），这种依赖较弱。

## 四、1D示例：数学等价性

考虑学习N个i.i.d.样本的均值，x₁...x_N，E[x] = μ，Var(x) = σ²。

**目标函数**：
```
f(w) = 1/2 E_x[(w - x)²]
```

**在w* = μ处最小化**

**SGD更新**：
```
w_{t+1} = (1 - ηₜ)w_t + ηₜ x_t
```

**从w₀ = 0展开递归**：
```
w_N = Σ_{k=1}^{N-1} a_k x_k
a_k = η_k Π_{s=k+1}^{N-1} (1 - ηₛ)
```

**常数步长η的等价性**：
选择：
```
 ̃a_k = (1 - (1-η)^{N-k})/N
```

得到一个EMA，等价于平均：
```
 ̄w_N = (1/N) Σ_{t=1}^N w_t
```

**数学等价性**：
在quadratic设置中，**常数学习率配合迭代平均和衰减学习率无平均在数学上是等价的**，仅在于它们如何实现相同的样本隐式加权。

## 五、讨论与结论

### 5.1 主要发现

1. **Anytime schedules的可行性**：Constant LR + Weight Averaging和1/√t + Weight Averaging可以在不需要知道总训练horizon的情况下匹配cosine性能

2. **理论保证**：对于具有功率律谱的线性回归，形式为1/t^γ的调度可以实现最优率（至对数和常数因子）

3. **WSD的实用性**：虽然WSD不是严格horizon-independent（因为它依赖于运行90%附近的额外checkpoints），但实现了与cosine和提议的anytime schedules相当的性能

4. **Weight Averaging的重要性**：强调了weight averaging在预训练中的实际重要性

### 5.2 实践建议

| 场景 | 推荐方案 | 原因 |
|------|---------|------|
| 持续学习/开放式训练 | Constant LR + Weight Averaging | 真正anytime，无需预知horizon |
| 知道大概训练长度 | WSD | 与cosine竞争，简单实用 |
| 大batch regime | Constant LR + Weight Averaging | 显著优于cosine |
| 不确定数据谱特性 | 1/√t + Weight Averaging | 鲁棒性好 |

### 5.3 未来方向

1. **扩展理论分析到深度神经网络**：当前分析集中在线性回归
2. **自动化EMA参数选择**：如何自动选择最佳的f值
3. **动态EMA窗口**：根据训练进度自适应调整EMA窗口
4. **结合其他优化技巧**：如momentum、schedule-free optimizer等

## 六、参考文献链接

1. **SGDR: Stochastic Gradient Descent with Warm Restarts**: https://arxiv.org/abs/1608.03983
2. **Llama 3**: https://arxiv.org/abs/2407.08325
3. **Olmo 3**: https://arxiv.org/abs/2512.13961
4. **DeepSeek-V3**: https://arxiv.org/abs/2412.19437
5. **Understanding Warmup-Stable-Decay Learning Rates**: https://arxiv.org/abs/2410.05192
6. **Acceleration of Stochastic Approximation by Averaging**: https://doi.org/10.1137/0310049
7. **The Road Less Scheduled**: https://arxiv.org/abs/2310.07831
8. **Critical Batch Size Scaling**: https://arxiv.org/abs/2410.21676
9. **Schedule-Free Optimizers**: https://arxiv.org/abs/2401.04015

这篇论文为持续学习和开放式训练提供了一个强有力的理论基础和实践方法，通过weight averaging和简单的horizon-free步长，提供了cosine学习率调度的一个实用有效的anytime替代方案。