# TensorLLM: 张量化多头注意力以增强LLM推理和压缩能力

## 1. 研究背景与核心问题

### 1.1 问题动机

大语言模型（LLM）基于Transformer架构，如GPT系列和LLaMA系列，通过大规模参数和数据训练展现出强大的能力。然而，研究表明这些模型**严重过参数化**，实际上并不需要所有学习到的参数来维持性能。这引发了**后训练压缩技术**的探索，旨在提高LLM在资源受限环境中的实用效率。

### 1.2 现有方法的局限性

**LASER模型** [12]：通过SVD分解单独的权重矩阵并移除对应最小奇异值的因子来提升推理性能，特别是在压缩前馈网络（FFN）权重矩阵时效果显著。但LASER对单个权重矩阵应用SVD，无法利用权重矩阵间的共享信息。

**TRAWL模型** [16]：采用基于张量的方法，将MHA或FFN块的权重矩阵简单地堆叠成高维张量后应用张量分解。虽然能利用权重间的固有张量结构并优于LASER，但仅在FFN块上去噪有效，对MHA块无效。

### 1.3 核心洞察

作者基于**MHA领域知识**和实证证据提出两个关键假设：

1. **同一层内的注意力头捕获相同级别的模式**
2. **同一层内不同的注意力头学习不同的专门知识**

这些假设得到文献支持：
- [17] 通过分析对应注意力头输出的Jensen-Shannon散度，发现Transformer层中存在明显的注意力头聚类
- [18, 19] 在多个尺度上可视化注意力，展示不同头的专门功能（如捕获位置和词汇模式、检测命名实体、识别句法和语义关系）

### 1.4 核心问题

> **能否通过在同一Transformer层的多个注意力头权重之间强制执行共享的高维子空间，来提升LLM的推理能力？**

## 2. 方法论详解

### 2.1 多头张量化

#### 2.1.1 张量化流程

**Step 1: 分离权重矩阵**
将全局权重矩阵分离到每个注意力头：

$$
\mathbf{W}^{Q} = [\mathbf{W}^{Q}_{1}, \mathbf{W}^{Q}_{2}, \ldots, \mathbf{W}^{Q}_{h}] \in \mathbb{R}^{d_{model} \times h \cdot d_{v}}
$$

$$
\mathbf{W}^{K} = [\mathbf{W}^{K}_{1}, \mathbf{W}^{K}_{2}, \ldots, \mathbf{W}^{K}_{h}] \in \mathbb{R}^{d_{model} \times h \cdot d_{v}}
$$

$$
\mathbf{W}^{V} = [\mathbf{W}^{V}_{1}, \mathbf{W}^{V}_{2}, \ldots, \mathbf{W}^{V}_{h}] \in \mathbb{R}^{d_{model} \times h \cdot d_{v}}
$$

$$
\mathbf{W}^{O} = [\mathbf{W}^{O}_{1}, \mathbf{W}^{O}_{2}, \ldots, \mathbf{W}^{O}_{h}] \in \mathbb{R}^{h \cdot d_{v} \times d_{model}}
$$

其中：
- $d_{model}$：嵌入维度（embedding dimension）
- $h$：注意力头数量（number of heads）
- $d_{v} = d_{model}/h$：每个头的维度（head dimension）
- $i$：注意力头索引，$1 \leq i \leq h$

**Step 2: 构建3D张量**
对每个注意力头 $i$，将四个2D子矩阵堆叠成3D张量：

$$
\mathcal{W}_{i}[:,:,j] = 
\begin{cases} 
\mathbf{W}^{Q}_{i} & \text{if } j=1, \\
\mathbf{W}^{K}_{i} & \text{if } j=2, \\
\mathbf{W}^{V}_{i} & \text{if } j=3, \\
(\mathbf{W}^{O}_{i})^{T} & \text{if } j=4.
\end{cases}
$$

其中 $\mathcal{W}_{i} \in \mathbb{R}^{d_{model} \times d_{v} \times 4}$

**Step 3: 构建4D张量**
将所有 $h$ 个头的3D张量堆叠成4D张量：

$$
\mathcal{W}_{all}[:,:,:,i] = \mathcal{W}_{i}, \quad \text{for } 1 \leq i \leq h
$$

其中 $\mathcal{W}_{all} \in \mathbb{R}^{d_{model} \times d_{v} \times 4 \times h}$

### 2.2 Tucker分解与共享因子矩阵

#### 2.2.1 标准Tucker分解回顾

Tucker分解是SVD到高维的推广，将原始张量分解为因子矩阵和较小核心张量的乘积：

$$
\mathcal{T} = \mathcal{G} \times_{1} \mathbf{U}^{(1)} \times_{2} \mathbf{U}^{(2)} \times_{3} \ldots \times_{N} \mathbf{U}^{(N)}
$$

其中：
- $\mathcal{T} \in \mathbb{R}^{I_{1} \times I_{2} \times \ldots \times I_{N}}$：N维张量
- $\mathcal{G} \in \mathbb{R}^{R_{1} \times R_{2} \times \ldots \times R_{N}}$：核心张量，表示缩放系数
- $\mathbf{U}^{(n)} \in \mathbb{R}^{I_{n} \times R_{n}}$：第n维的因子矩阵
- $(R_{1}, R_{2}, \ldots, R_{N})$：多重线性秩（multilinear ranks）
- $\times_{n}$：Mode-n张量积

**Mode-n积定义**：
$$
\mathcal{C} = \mathcal{A} \times_{n} \mathbf{B}
$$

$$
\mathcal{C}[i_{1},\ldots,i_{n-1},j_{n},i_{n+1},i_{N}] = \sum_{i_{n}=1}^{I_{n}} \mathcal{A}[i_{1},\ldots,i_{n-1},i_{n},i_{n+1},i_{N}] \mathbf{B}[j_{n},i_{n}]
$$

#### 2.2.2 共享因子矩阵的Tucker分解

**关键创新**：对每个注意力头的3D张量 $\{\mathcal{W}_{i}\}_{i=1}^{h}$ 应用Tucker分解，同时共享**同一组因子矩阵**：

$$
\mathcal{W}_{i} = \mathcal{G}_{i} \times_{1} \mathbf{U}^{(1)} \times_{2} \mathbf{U}^{(2)} \times_{3} \mathbf{U}^{(3)}, \quad \text{for } 1 \leq i \leq h
$$

其中：
- $\mathbf{U}^{(1)} \in \mathbb{R}^{d_{model} \times R_{1}}$：共享的嵌入维度因子矩阵
- $\mathbf{U}^{(2)} \in \mathbb{R}^{d_{v} \times R_{2}}$：共享的头维度因子矩阵
- $\mathbf{U}^{(3)} \in \mathbb{R}^{4 \times R_{3}}$：共享的投影类型因子矩阵（Q,K,V,O）
- $\mathcal{G}_{i} \in \mathbb{R}^{R_{1} \times R_{2} \times R_{3}}$：第i个头的核心张量（每个头独立）

这可以等价地写成4D张量的特殊Tucker分解形式：

$$
\mathcal{W}_{all} = \mathcal{G}_{all} \times_{1} \mathbf{U}^{(1)} \times_{2} \mathbf{U}^{(2)} \times_{3} \mathbf{U}^{(3)} \times_{4} \mathbf{I}
$$

其中：
- $\mathbf{I} \in \mathbb{R}^{h \times h}$：单位矩阵（可省略）
- $\mathcal{G}_{all} \in \mathbb{R}^{R_{1} \times R_{2} \times R_{3} \times h}$：4D核心张量

#### 2.2.3 优化目标

去噪过程通过最小化重构误差：

$$
\frac{1}{2} \left\| \sum_{i=1}^{h} \mathcal{W}_{i} - \sum_{i=1}^{h} \mathcal{G}_{i} \times_{1} \mathbf{U}^{(1)} \times_{2} \mathbf{U}^{(2)} \times_{3} \mathbf{U}^{(3)} \right\|^{2}_{F}
$$

或等价形式：

$$
\frac{1}{2} \left\| \mathcal{W}_{all} - \mathcal{G}_{all} \times_{1} \mathbf{U}^{(1)} \times_{2} \mathbf{U}^{(2)} \times_{3} \mathbf{U}^{(3)} \right\|^{2}_{F}
$$

其中 $\|\cdot\|_{F}$ 表示Frobenius范数。

**实现细节**：使用TensorLy库[30]和HOOI（Higher Order Orthogonal Iterations）算法[31]实现这种特殊的Tucker分解。

### 2.3 方法论架构图解析

```
┌─────────────────────────────────────────────────────────────┐
│                     Step 1: 分离权重矩阵                      │
├─────────────────────────────────────────────────────────────┤
│  W^Q → [W^Q_1, W^Q_2, ..., W^Q_h]                           │
│  W^K → [W^K_1, W^K_2, ..., W^K_h]                           │
│  W^V → [W^V_1, W^V_2, ..., W^V_h]                           │
│  W^O → [W^O_1, W^O_2, ..., W^O_h]                           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                Step 2: 构建3D张量（每个头）                   │
├─────────────────────────────────────────────────────────────┤
│  W_i[:,:,1] = W^Q_i  (Query投影)                            │
│  W_i[:,:,2] = W^K_i  (Key投影)                              │
│  W_i[:,:,3] = W^V_i  (Value投影)                            │
│  W_i[:,:,4] = (W^O_i)^T  (输出投影)                         │
│  其中 W_i ∈ ℝ^(d_model × d_v × 4)                           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              Step 3: Tucker分解（共享因子矩阵）                │
├─────────────────────────────────────────────────────────────┤
│  W_i = G_i ×₁ U^(1) ×₂ U^(2) ×₃ U^(3)  (i=1,2,...,h)        │
│                                                             │
│  共享因子矩阵：                                              │
│  U^(1) ∈ ℝ^(d_model × R₁)  - 嵌入维度子空间                  │
│  U^(2) ∈ ℝ^(d_v × R₂)      - 头维度子空间                    │
│  U^(3) ∈ ℝ^(4 × R₃)       - 投影类型子空间                   │
│                                                             │
│  独立核心张量：                                              │
│  G_i ∈ ℝ^(R₁ × R₂ × R₃)  - 每个头的变异信息                  │
└─────────────────────────────────────────────────────────────┘
```

### 2.4 与其他方法的对比

**LASER的分解**：
$$
\mathbf{A} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^{T}
$$

- 对单个矩阵应用SVD
- 无法利用矩阵间的相关性

**TRAWL的分解**：
$$
\mathcal{T} = \mathcal{G} \times_{1} \mathbf{A} \times_{2} \mathbf{B} \times_{3} \mathbf{C}
$$

- 将矩阵堆叠成3D张量后分解
- 对MHA块效果不佳

**TensorLLM的分解**：
$$
\mathcal{W}_{i} = \mathcal{G}_{i} \times_{1} \mathbf{U}^{(1)} \times_{2} \mathbf{U}^{(2)} \times_{3} \mathbf{U}^{(3)} \quad (\forall i)
$$

- 多头共享因子矩阵
- 强制不同头在同一子空间中操作
- 每个头有独立的核心张量编码专门信息

## 3. 实验设置

### 3.1 模型架构

| 模型 | 参数量 | 架构类型 | 特点 |
|------|--------|----------|------|
| RoBERTa | 125M | Encoder-only | 预测上下文中缺失的token，使用`<mask>` token |
| GPT-J | 6B | Decoder-only | 自回归生成token |
| LLaMA 2 | 7B | Decoder-only | 自回归生成token |

### 3.2 数据集

**HotPotQA**：
- 大规模问答数据集
- 任务格式："<<<question>>> The answer is"
- 评估：答案出现在生成文本中即认为正确

**FEVER**：
- 事实验证数据集
- 二元分类：0（虚假），1（真实）
- 任务格式："Consider the following claim: <<<claim>>>. Is this claim true or false? The claim is"
- 评估：真实概率超过虚假概率则分类为真

**Bios Profession**：
- 性别偏见分析基准
- 任务：根据传记预测职业（10个职业之一）
- 任务格式："Consider the following text: <<<bio>>>. What is the profession of the person in this text? The profession of this person is"
- 评估：输出概率最高的职业

**BigBench-WikidataQA**：
- 使用Wikidata结构化知识的问答
- 任务：回答事实性问题
- 评估：答案出现在生成文本中即认为正确

### 3.3 评估指标

1. **准确率**：测试集中正确预测实例的百分比
2. **损失**：真实token的负对数似然，衡量模型不确定性
3. **压缩比**：
$$
CR = \frac{N_{original}}{N_{compressed}}
$$
其中 $N_{original}$ 和 $N_{compressed}$ 分别表示原始和压缩后的MHA参数总数。

## 4. 实验结果详解

### 4.1 主要结果

#### 4.1.1 RoBERTa 125M结果

| 数据集 | 原始准确率 | 方法准确率 | 提升 | 压缩比 |
|--------|------------|------------|------|--------|
| HotPotQA | 6.1% | 7.33% | +1.23% | 1.12× |
| FEVER | 50.0% | 50.45% | +0.45% | 3.74× |
| Bios Profession | 64.5% | 72.57% | +8.07% | 8.78× |
| WikidataQA | 28.0% | 32.72% | +4.72% | 2.52× |

#### 4.1.2 GPT-J 6B结果

| 数据集 | 原始准确率 | 方法准确率 | 提升 | 压缩比 |
|--------|------------|------------|------|--------|
| HotPotQA | 19.6% | 20.15% | +0.55% | 247.30× |
| FEVER | 50.2% | 58.94% | +8.74% | 14.69× |
| Bios Profession | 75.6% | 81.18% | +5.58% | 74.68× |
| WikidataQA | 51.8% | 68.81% | +17.01% | 46.77× |

#### 4.1.3 LLaMA 2 7B结果

| 数据集 | 原始准确率 | 方法准确率 | 提升 | 压缩比 |
|--------|------------|------------|------|--------|
| HotPotQA | 16.5% | 18.44% | +1.94% | 3.54× |
| FEVER | 59.3% | 66.75% | +7.45% | 3.54× |
| Bios Profession | 85.0% | 86.61% | +1.61% | 3.54× |
| WikidataQA | 59.5% | 60.37% | +0.87% | 5.81× |

### 4.2 与LASER的混合方法对比

| 案例 | 配置 |
|------|------|
| Case 1 | LASER仅应用于FFN块的一个权重矩阵 |
| Case 2 | LASER应用于FFN和MHA块的所有矩阵 |
| Case 3 | 本文方法应用于MHA块 + LASER应用于FFN块 |

#### 4.2.1 RoBERTa混合结果

| 数据集 | Case 1准确率 | Case 2准确率 | Case 3准确率 |
|--------|--------------|--------------|--------------|
| HotPotQA | 6.7% | 5.24% | **7.05%** |
| FEVER | 52.3% | 53.6% | **55.23%** |
| Bios Profession | 72.5% | 71.14% | **72.51%** |
| WikidataQA | 30.7% | 34.49% | **37.40%** |

#### 4.2.2 GPT-J混合结果

| 数据集 | Case 1准确率 | Case 2准确率 | Case 3准确率 |
|--------|--------------|--------------|--------------|
| HotPotQA | 19.5% | 19.62% | **19.91%** |
| FEVER | 56.2% | 55.59% | **58.98%** |
| Bios Profession | 82.1% | 81.28% | **82.52%** |
| WikidataQA | 65.9% | 65.68% | **68.20%** |

#### 4.2.3 LLaMA 2混合结果

| 数据集 | Case 1准确率 | Case 2准确率 | Case 3准确率 |
|--------|--------------|--------------|--------------|
| HotPotQA | 17.2% | 18.88% | **19.22%** |
| FEVER | 64.5% | 65.13% | **66.39%** |
| Bios Profession | 86.7% | 86.07% | **87.07%** |
| WikidataQA | 62.0% | 61.21% | 61.78% |

### 4.3 消融实验

分别压缩Q、K、V、O权重矩阵的对比（GPT-J 6B）：

| 数据集 | 原始 | W^Q | W^K | W^V | W^O | **全部（本文方法）** |
|--------|------|-----|-----|-----|-----|---------------------|
| HotPotQA | 19.6% | 19.19% | 19.25% | 19.70% | 19.62% | **20.15%** |
| FEVER | 50.2% | 54.41% | 53.40% | 55.86% | 56.07% | **58.94%** |
| Bios Profession | 75.6% | 76.06% | 74.97% | 79.39% | 79.71% | **81.18%** |
| WikidataQA | 51.8% | 49.72% | 51.01% | 48.82% | 48.87% | **68.81%** |

## 5. 技术创新与意义分析

### 5.1 核心创新点

1. **共享高维子空间强制机制**：
   - 通过共享因子矩阵 $\mathbf{U}^{(1)}, \mathbf{U}^{(2)}, \mathbf{U}^{(3)}$ 强制不同注意力头在同一子空间操作
   - 每个头通过独立核心张量 $\mathcal{G}_{i}$ 编码专门化信息
   - 符合"同一层注意力头捕获相似抽象级别但具有不同专门化"的直觉

2. **多头张量化技术**：
   - 将Q、K、V、O四个投影矩阵组合成3D张量
   - 保留了不同投影类型间的关联信息
   - 消融实验证明同时处理四种矩阵优于单独处理

3. **零训练后处理框架**：
   - 不需要额外数据、训练或微调
   - 直接应用于预训练模型
   - 实用性强，可快速部署

### 5.2 理论意义

**Remark 1分析**：
从公式（13）可以看出，每个注意力头的权重共享同一组因子矩阵 $\mathbf{U}^{(1)}, \mathbf{U}^{(2)}, \mathbf{U}^{(3)}$，同时每个注意力头被分配自己的Tucker核心张量。这种设计与"单个Transformer层内的注意力头在相似抽象级别上捕获不同专门化信息的模式"的直觉一致。

**Remark 2分析**：
公式（10）-（14）中的张量分解过程使我们能够根据共享的高维低秩结构对注意力权重矩阵进行结构化去噪。此外，这也允许通过较小的因子表示原始权重张量，从而实现参数压缩。

### 5.3 实际应用价值

1. **推理能力提升**：
   - GPT-J在WikidataQA上提升17.01%
   - LLaMA 2在FEVER上提升7.45%
   - 说明结构化去噪确实能提升推理性能

2. **显著压缩效果**：
   - GPT-J在HotPotQA上达到247.3×压缩比
   - 同时保持或提升性能
   - 适用于资源受限场景

3. **模块化设计**：
   - 可与FFN压缩方法（如LASER）结合
   - Case 3在大多数场景下达到最佳性能
   - 为LLM压缩提供新的模块化组件

## 6. 局限性与未来方向

### 6.1 已知局限

1. **超参数敏感性**：
   - 不同数据集需要不同的超参数设置
   - 缺乏统一的、泛化的超参数配置

2. **层选择性应用**：
   - 当前研究仅应用于单层
   - 多层同时应用的效果有待探索

### 6.2 未来研究方向

1. **自适应秩选择**：
   - [26]提出LLM引导的多线性张量网络秩选择
   - [27]提出SVDinsTN：基于正则化建模视角的高效结构搜索张量网络范式

2. **与训练结合**：
   - 在训练过程中集成张量化约束
   - 探索Tucker分解的正则化效果

3. **扩展到其他模块**：
   - 除MHA和FFN外的其他Transformer组件
   - 嵌入层的张量化[15]

## 7. 相关技术对比

### 7.1 张量分解方法对比

| 方法 | 分解形式 | 适用场景 | 优势 | 劣势 |
|------|----------|----------|------|------|
| SVD | $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$ | 矩阵去噪 | 理论成熟 | 无法利用高维结构 |
| Tucker | $\mathcal{T} = \mathcal{G} \times_{1}\mathbf{U}^{(1)}\ldots$ | 高维张量 | 灵活的秩控制 | 计算复杂度高 |
| CP | $\mathcal{T} = \sum_{r=1}^{R} \lambda_{r} \mathbf{u}^{(1)}_{r} \circ \ldots$ | 稀疏张量 | 高度压缩 | 对复杂结构建模不足 |
| Tensor Train | $\mathcal{T} = \sum_{i_{1},\ldots,i_{N}} G^{(1)}_{i_{1}}\ldots G^{(N)}_{i_{N}}$ | 大规模张量 | 可扩展性强 | 交互建模有限 |

### 7.2 LLM压缩方法对比

| 方法 | 目标模块 | 技术手段 | 需要训练 | 推理提升 |
|------|----------|----------|----------|----------|
| LASER | FFN | SVD | 否 | 显著 |
| TRAWL | FFN/MHA | Tucker | 否 | 仅FFN有效 |
| TensorLLM | MHA | 共享Tucker | 否 | 显著 |
| CPT [15] | Embedding | Tensor Train | 否 | 中等 |

## 8. 数学直觉构建

### 8.1 为什么共享子空间有效？

**类比**：想象一个交响乐团的指挥家（共享子空间）和不同的乐器手（专门化信息）：
- 指挥家定义了整体节奏和调性（$\mathbf{U}^{(1)}, \mathbf{U}^{(2)}, \mathbf{U}^{(3)}$）
- 每个乐器手在同一框架下演奏不同的旋律（$\mathcal{G}_{i}$）
- 这种结构既保证了协调性（共享子空间），又保留了多样性（专门化）

**数学解释**：
共享因子矩阵实际上是在执行一种**联合维度约减**：

$$
\mathbf{W}^{Q}_{i} \approx \mathbf{U}^{(1)} \mathcal{G}_{i}[:,:,1]^{T} \mathbf{U}^{(2)T}
$$

这种约束使得：
1. 不同头的投影矩阵在同一个低维子空间中
2. 去除了训练过程中引入的随机噪声
3. 保留了每个头的专门化信息（通过$\mathcal{G}_{i}$）

### 8.2 压缩比的来源

**原始参数量**：
$$
N_{original} = 4 \times h \times d_{model} \times d_{v} = 4 \times h \times d_{model} \times \frac{d_{model}}{h} = 4 d_{model}^2
$$

**压缩后参数量**：
$$
N_{compressed} = h \times (R_{1} R_{2} R_{3}) + d_{model} R_{1} + d_{v} R_{2} + 4 R_{3}
$$

**压缩比**：
$$
CR = \frac{4 d_{model}^2}{h R_{1} R_{2} R_{3} + d_{model} R_{1} + d_{v} R_{2} + 4 R_{3}}
$$

当 $R_{n} \ll I_{n}$ 时，压缩比可以达到数十甚至数百倍。

## 9. 实现细节

### 9.1 算法流程

```
Algorithm: TensorLLM MHA Compression
Input: Pre-trained MHA weights W^Q, W^K, W^V, W^O
Input: Target ranks (R₁, R₂, R₃)
Output: Compressed MHA weights

1: // Step 1: Split weights by heads
2: for i = 1 to h do
3:     W^Q_i = W^Q[:, (i-1)*d_v : i*d_v]
4:     W^K_i = W^K[:, (i-1)*d_v : i*d_v]
5:     W^V_i = W^V[:, (i-1)*d_v : i*d_v]
6:     W^O_i = W^O[(i-1)*d_v : i*d_v, :].T
7: end for

8: // Step 2: Tensorise each head
9: for i = 1 to h do
10:    W_i = stack(W^Q_i, W^K_i, W^V_i, W^O_i, dim=2)
11:    // W_i has shape (d_model, d_v, 4)
12: end for

13: // Step 3: Tucker decomposition with shared factors
14: // Using HOOI algorithm
15: Initialize U^(1), U^(2), U^(3) using SVD
16: repeat
17:     // Update factor matrices (shared across heads)
18:     U^(1) = left_singular_vectors( unfold(W_all, mode=1) )
19:     U^(2) = left_singular_vectors( unfold(W_all, mode=2) )
20:     U^(3) = left_singular_vectors( unfold(W_all, mode=3) )
21:     
22:     // Update core tensors (one per head)
23:     for i = 1 to h do
24:         G_i = W_i ×₁ U^(1)ᵀ ×₂ U^(2)ᵀ ×₃ U^(3)ᵀ
25:     end for
26: until convergence

27: // Step 4: Reconstruct compressed weights
28: for i = 1 to h do
29:     W_compressed_i = G_i ×₁ U^(1) ×₂ U^(2) ×₃ U^(3)
30:     // Extract back to matrices
31:     W^Q_compressed_i = W_compressed_i[:, :, 0]
32:     W^K_compressed_i = W_compressed_i[:, :, 1]
33:     W^V_compressed_i = W_compressed_i[:, :, 2]
34:     W^O_compressed_i = W_compressed_i[:, :, 3].T
35: end for

36: return W^Q_compressed, W^K_compressed, W^V_compressed, W^O_compressed
```

### 9.2 计算复杂度分析

**HOOI算法复杂度**：
- 每次迭代：$O(N \times \sum_{n=1}^{N} I_{n} R_{n})$
- 对于4D张量：$O(4 \times (d_{model} R_{1} + d_{v} R_{2} + 4 R_{3} + h R_{4}))$

**存储优势**：
- 原始：$O(4 h d_{model} d_{v})$
- 压缩：$O(h R_{1} R_{2} R_{3} + d_{model} R_{1} + d_{v} R_{2} + 4 R_{3})$

## 10. 参考文献

这篇论文引用了以下重要工作：

1. **Transformer基础**：
   - [28] Vaswani et al., "Attention is All you Need", NeurIPS 2017
   - [1] Brown et al., "Language models are few-shot learners", NeurIPS 2020

2. **LLM压缩与去噪**：
   - [12] Sharma et al., "The Truth is in There: Improving Reasoning in Language Models with Layer-Selective Rank Reduction", ICLR 2024
   - [16] Luo et al., "TRAWL: Tensor Reduced and Approximated Weights for Large Language Models", 2024
   - [15] Calvi et al., "Compression and Interpretability of Deep Neural Networks Via Tucker Tensor Layer", 2019

3. **注意力机制分析**：
   - [17] Clark et al., "What does BERT look at? an analysis of BERT's attention", 2019
   - [18] Vig, "A Multiscale Visualization of Attention in the Transformer Model", ACL 2019
   - [19] Vig & Belinkov, "Analyzing the Structure of Attention in a Transformer Language Model", 2019

4. **张量分解理论**：
   - [23] Tucker, "Some mathematical notes on three-mode factor analysis", 1966
   - [24] De Lathauwer et al., "A multilinear singular value decomposition", SIAM 2000
   - [21] Cichocki et al., "Tensor decompositions for signal processing applications", IEEE SPM 2015

5. **张量网络**：
   - [20] Orús, "A practical introduction to tensor networks", Annals of Physics 2014
   - [30] Kossaifi et al., "TensorLy: Tensor Learning in Python", JMLR 2019

## 11. 总结

**TensorLLM的核心贡献**：

1. ✅ **首个有效的MHA块去噪方法**：解决了现有方法仅对FFN有效的问题
2. ✅ **共享高维子空间机制**：基于MHA领域知识设计，强制多头协同工作
3. ✅ **显著的推理提升**：在多个基准数据集上持续提升推理能力
4. ✅ **强大的压缩能力**：最高达到247.3×压缩比
5. ✅ **零训练框架**：无需额外数据或训练，实用性强
6. ✅ **模块化设计**：可与现有FFN压缩方法无缝结合

**对LLM领域的启示**：

1. **结构化去噪的重要性**：不仅仅是参数减少，更重要的是去除训练噪声
2. **领域知识的价值**：基于MHA本质的设计比通用方法更有效
3. **多头协同机制**：注意力头不是独立的，而是共享基础子空间
4. **张量方法的潜力**：高维结构建模在LLM中的应用前景广阔

**代码资源**：
- GitHub: https://github.com/guyuxuan9/TensorLLM

这篇论文为LLM的推理能力提升和模型压缩提供了一个新颖、有效且实用的框架，通过巧妙的张量分解设计和共享子空间机制，成功解决了长期存在的MHA块去噪难题。