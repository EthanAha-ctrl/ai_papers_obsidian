
![[Pasted image 20260120220849.png]]

![[Pasted image 20260120220919.png]]
## 一、Data Parallelism (DP) - 数据并行

### 核心原理

Data Parallelism 是最基础的并行策略，将**整个模型**复制到每个GPU上，每个GPU处理**不同的数据batch**。

### 技术架构详解

```
┌─────────────────────────────────────────────────────┐
│                  Training Process                    │
├─────────────────────────────────────────────────────┤
│  GPU 0: Model Copy + Batch[0,1,2] → Forward + Backward
│  GPU 1: Model Copy + Batch[3,4,5] → Forward + Backward
│  GPU 2: Model Copy + Batch[6,7,8] → Forward + Backward
│  GPU 3: Model Copy + Batch[9,10,11] → Forward + Backward
├─────────────────────────────────────────────────────┤
│         All-Reduce Operation (Gradient Sync)         │
│         → Averaged gradients synchronized            │
└─────────────────────────────────────────────────────┘
```

### 数学公式

在标准的Data Parallelism中，梯度聚合遵循：

```
g_avg = (1/N) × Σ(g_i)  for i = 0 to N-1

其中：
- N = 数据并行worker数量
- g_i = 第i个GPU的梯度
- g_avg = 全局平均梯度
```

### ZeRO优化 (Zero Redundancy Optimizer)

DeepSpeed的ZeRO将优化器状态、梯度和参数切分到不同GPU：

```
┌─────────────────────────────────────────────────────┐
│                   ZeRO Stages                       │
├─────────────────────────────────────────────────────┤
│ ZeRO-1: 切分Optimizer States                         │
│   每个GPU存储 1/N 的optimizer states                 │
│   Communication: All-Gather during optimizer step   │
├─────────────────────────────────────────────────────┤
│ ZeRO-2: 切分Gradients + Optimizer States            │
│   每个GPU存储 1/N 的gradients                       │
│   Communication: All-Gather + All-Reduce            │
├─────────────────────────────────────────────────────┤
│ ZeRO-3: 切分Parameters + Gradients + Opt States     │
│   每个GPU存储 1/N 的模型参数                        │
│   Communication: All-Gather during forward pass     │
└─────────────────────────────────────────────────────┘
```

### 内存开销分析

| 配置 | 每个GPU内存占用 | 通信开销 |
|------|----------------|----------|
| Standard DP | Full Model | 2× model size per step |
| ZeRO-1 | Model + (1/N)×Opt | +10-15% |
| ZeRO-2 | Model + (1/N)×(Opt+Grad) | +20-25% |
| ZeRO-3 | (1/N)×Model | +50-100% |

### 优势与劣势

**优势**：
- 实现简单，易于理解
- 对于小模型到中等模型效率最高
- 通信模式固定，易于优化

**劣势**：
- 模型必须能放入单个GPU内存
- 对超大规模模型（如175B+）不可行
- 通信随模型大小线性增长

**适用场景**：模型参数 < 100B，且序列长度适中

## 二、Tensor Parallelism (TP) - 张量并行

### 核心原理

Tensor Parallelism 将**模型层的张量**切分到多个GPU上，每个GPU只持有完整模型的一部分权重。这是Megatron-LM提出的核心技术。

### 技术架构详解

#### MLP Layer的TP切分

以GPT的MLP层为例（假设hidden_size=4096，ffn_size=16384，TP=4）：

```
原始计算流程：
X [B, S, H] → W1 [H, 4H] → GELU → W2 [4H, H] → Output [B, S, H]

TP切分后（每个GPU获得 1/4 的维度）：

┌─────────────────────────────────────────────────────┐
│                   Forward Pass                      │
├─────────────────────────────────────────────────────┤
│ GPU 0: X → W1[:, 0:1024] → GELU → W2[0:1024, :]
│ GPU 1: X → W1[:, 1024:2048] → GELU → W2[1024:2048, :]
│ GPU 2: X → W1[:, 2048:3072] → GELU → W2[2048:3072, :]
│ GPU 3: X → W1[:, 3072:4096] → GELU → W2[3072:4096, :]
├─────────────────────────────────────────────────────┤
│         All-Reduce: Σ(output_i) → Final Output       │
└─────────────────────────────────────────────────────┘

注意：X在所有GPU上是相同的（通过broadcast）
```

#### Multi-Head Attention的TP切分

```
原始Attention：
Q = X @ W_q, K = X @ W_k, V = X @ W_v
Output = Attention(Q, K, V) @ W_o

TP切分（假设TP=4）：

┌─────────────────────────────────────────────────────┐
│              Head-wise Partitioning                 │
├─────────────────────────────────────────────────────┤
│ GPU 0: W_q[:, 0:H/4], W_k[:, 0:H/4], W_v[:, 0:H/4]   │
│        W_o[0:H/4, :]                                 │
│        Handles heads[0:head_num/4]                  │
├─────────────────────────────────────────────────────┤
│ GPU 1: W_q[:, H/4:2H/4], W_k[:, H/4:2H/4], ...      │
│        W_o[H/4:2H/4, :]                              │
│        Handles heads[head_num/4:2*head_num/4]       │
├─────────────────────────────────────────────────────┤
│ Forward:                                            │
│   1. 每个GPU独立计算Q, K, V（无需通信）              │
│   2. 每个GPU独立计算local attention                  │
│   3. All-Reduce聚合W_o的输出                        │
└─────────────────────────────────────────────────────┘
```

### 数学公式

对于Tensor Parallelism，核心是矩阵乘法的分布式计算：

**Column-wise Partitioning (W1)**：
```
Y = X @ W1

其中 W1 = [W1_0, W1_1, ..., W1_{n-1}]

每个GPU i计算：
Y_i = X @ W1_i

最终通过All-Reduce合并：
Y = Σ(Y_i) = X @ Σ(W1_i)
```

**Row-wise Partitioning (W2)**：
```
Output = Y @ W2

其中 W2 = [W2_0; W2_1; ...; W2_{n-1}] (垂直拼接)

每个GPU i计算：
Output_i = Y @ W2_i

最终通过All-Reduce合并：
Output = Σ(Output_i)
```

### 通信模式

```
┌─────────────────────────────────────────────────────┐
│            TP Communication Pattern                 │
├─────────────────────────────────────────────────────┤
│ Layer Norm + Linear (Column-wise):                  │
│   Forward: All-Reduce after Linear                   │
│   Backward: All-Reduce before Linear gradient       │
├─────────────────────────────────────────────────────┤
│ Layer Norm + Linear (Row-wise):                     │
│   Forward: All-Reduce after Linear                   │
│   Backward: All-Reduce after Linear gradient        │
├─────────────────────────────────────────────────────┤
│ Total per layer: ~4 All-Reduce operations          │
│ Communication Volume: O(2 × hidden_size × TP)       │
└─────────────────────────────────────────────────────┘
```

### 内存节省

```
对于TP = N：

Parameter Memory per GPU = Original_Memory / N
Activation Memory per GPU = Original_Memory / N (部分)

实际内存占用：
┌─────────────────────────────────────────────────────┐
│  Memory = (Model_Params / TP) + (Acts / TP) +       │
│           (Opt_States / TP) + Communication_Buf     │
└─────────────────────────────────────────────────────┘
```

### 优势与劣势

**优势**：
- 可以训练无法放入单个GPU的模型
- 通信仅限于TP组内，延迟可控
- 计算效率高，适合GPU集群

**劣势**：
- 每个TP组内需要同步所有层
- TP degree不能太大（通常≤8）
- 跨TP组通信频繁

**适用场景**：
- 模型参数大（50B-500B）
- 单节点内GPU数量有限
- 需要低延迟通信

## 三、Pipeline Parallelism (PP) - 流水线并行

### 核心原理

Pipeline Parallelism 将**模型的不同层**分配到不同的GPU上，形成计算流水线。

### 技术架构详解

#### 基本Pipeline结构

```
┌─────────────────────────────────────────────────────┐
│              Pipeline Parallelism (PP=4)            │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Stage 0 (GPU 0):  Layer 0-7   →                    │
│                                  ↓                   │
│  Stage 1 (GPU 1):  Layer 8-15   →                    │
│                                  ↓                   │
│  Stage 2 (GPU 2):  Layer 16-23  →                    │
│                                  ↓                   │
│  Stage 3 (GPU 3):  Layer 24-31  → Output            │
│                                                      │
└─────────────────────────────────────────────────────┘

每个GPU只存储和计算模型的一部分层
```

#### Micro-batch Scheduling (1F1B策略)

```
时间步    Micro-batch 流动
─────────────────────────────────────────────────────────────
t=0:      MB0 → GPU0
t=1:      MB1 → GPU0, MB0 → GPU1
t=2:      MB2 → GPU0, MB1 → GPU1, MB0 → GPU2
t=3:      MB3 → GPU0, MB2 → GPU1, MB1 → GPU2, MB0 → GPU3
t=4:           MB3 → GPU1, MB2 → GPU2, MB1 → GPU3 (MB0 done)
t=5:                MB3 → GPU2, MB2 → GPU3
t=6:                     MB3 → GPU3
t=7:                          (MB3 done)

 steady-state: 每个时间步完成1个micro-batch
```

### 内存分析

```
┌─────────────────────────────────────────────────────┐
│           PP Memory per Stage                        │
├─────────────────────────────────────────────────────┤
│ Parameter Memory: Model_Params / PP                 │
│                                                      │
│ Activation Memory:                                  │
│   - 1F1B方案: O(2 × micro_batch_size × seq_len)     │
│   - Interleaved: 可进一步优化                       │
├─────────────────────────────────────────────────────┤
│ 优势: Activation memory随stage减少                  │
│ 劣势: Pipeline bubble导致GPU利用率下降              │
└─────────────────────────────────────────────────────┘
```

### Pipeline Bubble

```
Efficiency = (Number_of_Stages × Microbatches) / 
             (Number_of_Stages + Microbatches - 1)

示例：
- 4 stages, 8 microbatches:
  Efficiency = (4 × 8) / (4 + 8 - 1) = 32 / 11 ≈ 72.7%

- 4 stages, 16 microbatches:
  Efficiency = (4 × 16) / (4 + 16 - 1) = 64 / 19 ≈ 84.2%

结论: 更多的microbatches提高利用率，但增加延迟
```

### Interleaved Pipeline Parallelism

DeepSpeed引入的优化技术，将每个stage包含多个连续的流水线子stage：

```
Traditional PP (4 stages, 4 GPUs):
GPU0: Stage0 → GPU1: Stage1 → GPU2: Stage2 → GPU3: Stage3

Interleaved PP (4 stages, 8 GPUs, 2 models):
GPU0: Model0-Stage0, Model1-Stage0
GPU1: Model0-Stage1, Model1-Stage1
GPU2: Model0-Stage2, Model1-Stage2
GPU3: Model0-Stage3, Model1-Stage3
GPU4: Model0-Stage0, Model1-Stage0
GPU5: Model0-Stage1, Model1-Stage1
GPU6: Model0-Stage2, Model1-Stage2
GPU7: Model0-Stage3, Model1-Stage3

优势: 减少pipeline bubble, 提高GPU利用率
```

### 通信分析

```
┌─────────────────────────────────────────────────────┐
│              PP Communication                       │
├─────────────────────────────────────────────────────┤
│ Forward: Send activations to next stage            │
│          Volume: O(micro_batch_size × hidden_size)  │
├─────────────────────────────────────────────────────┤
│ Backward: Send gradients to previous stage         │
│          Volume: O(micro_batch_size × hidden_size)  │
├─────────────────────────────────────────────────────┤
│ Frequency: 每个micro-batch 1次forward + 1次backward │
│          Total: 2 × num_microbatches × PP          │
└─────────────────────────────────────────────────────┘
```

### 优势与劣势

**优势**：
- 参数内存线性减少（随PP数量）
- 通信只在相邻stage间，通信量小
- 适合超深模型（如100+层）

**劣势**：
- Pipeline bubble导致GPU空闲
- 小batch size时效率低
- 调试复杂，容易出现deadlock

**适用场景**：
- 模型层数深（>50层）
- 需要大规模模型并行
- 可容忍一定延迟

## 四、Context Parallelism (CP) - 上下文并行

### 核心原理

Context Parallelism（也称Sequence Parallelism）将**序列长度维度**切分到多个GPU上，每个GPU处理序列的一部分。这是处理超长序列的关键技术。

### 技术架构详解

#### Ring Attention机制

```
┌─────────────────────────────────────────────────────┐
│              Ring Attention (CP=4)                  │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Sequence: [T0, T1, T2, T3, T4, T5, T6, T7]         │
│                                                      │
│  GPU 0: 处理 [T0, T1]  +  KV Cache [T2,T3,T6,T7]    │
│  GPU 1: 处理 [T2, T3]  +  KV Cache [T4,T5,T0,T1]    │
│  GPU 2: 处理 [T4, T5]  +  KV Cache [T6,T7,T2,T3]    │
│  GPU 3: 处理 [T6, T7]  +  KV Cache [T0,T1,T4,T5]    │
│                                                      │
│  通信模式: 环形 (Ring) All-to-All                    │
│                                                      │
└─────────────────────────────────────────────────────┘
```

#### Megatron-Core CP实现

```
┌─────────────────────────────────────────────────────┐
│         Megatron-Core Context Parallelism           │
├─────────────────────────────────────────────────────┤
│                                                      │
│  核心组件:                                           │
│  1. CP All-to-All: 重分配activations                │
│  2. CP All-Gather: 聚合KV cache用于attention        │
│  3. FlashAttention kernel支持                        │
│                                                      │
│  Forward Pass流程:                                  │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐         │
│  │  Query  │ →  │ All-to- │ →  │ Local   │         │
│  │ Split   │    │  All    │    │ Attn    │         │
│  └─────────┘    └─────────┘    └─────────┘         │
│                      ↓                                 │
│              ┌──────────────┐                        │
│              │  KV Cache    │                        │
│              │  All-Gather  │                        │
│              └──────────────┘                        │
│                      ↓                                 │
│              ┌──────────────┐                        │
│              │  Output      │                        │
│              │  All-Gather  │                        │
│              └──────────────┘                        │
│                                                      │
└─────────────────────────────────────────────────────┘
```

#### Ulysses Attention (序列内All-Reduce)

```
┌─────────────────────────────────────────────────────┐
│            Ulysses Attention CP                     │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Step 1: All-Reduce KV heads                        │
│    - 每个GPU获得完整的KV缓存                         │
│    - 通信量: O(seq_len × head_dim)                  │
│                                                      │
│  Step 2: Compute local attention                    │
│    - 每个GPU独立计算其负责的query                    │
│    - 无需额外通信                                    │
│                                                      │
│  Step 3: All-Gather outputs                         │
│    - 合并所有GPU的输出                               │
│                                                      │
│  优势: 对CP扩展性好                                  │
│  劣势: 初始All-Reduce通信量大                       │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 数学分析

对于Ring Attention，时间复杂度为：

```
Original Attention Complexity:
O(S² × H)

Ring Attention Complexity (CP = N):
Computation per GPU: O((S/N)² × H) + O(S² × H / N)
Communication per step: O(S × H / N)

其中：
- S = 序列长度
- H = 隐藏层维度
- N = Context并行度

Total Time ≈ Computation + Communication × (N-1)
```

### FlashAttention优化

```
┌─────────────────────────────────────────────────────┐
│           FlashAttention + Context Parallelism      │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Key Optimizations:                                 │
│  1. IO-aware精确计算                                 │
│  2. 在HBM和SRAM之间tiling                           │
│  3. 减少memory access次数                            │
│                                                      │
│  Memory Complexity:                                 │
│  Original: O(S²) for attention matrix               │
│  FlashAttention: O(S × H)                           │
│  With CP: O(S × H / CP) per GPU                    │
│                                                      │
│  启用的序列长度:                                     │
│  - 无CP: ~8K - 16K tokens                           │
│  - With CP: ~32K - 128K+ tokens                     │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### Hybrid 4D Parallelism

现代框架结合所有四种并行策略：

```
┌─────────────────────────────────────────────────────┐
│              4D Parallelism Example                 │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Model: 175B GPT, Batch=1M tokens, Seq=128K        │
│                                                      │
│  配置:                                               │
│  - DP = 8   (Data Parallelism)                      │
│  - TP = 4   (Tensor Parallelism)                    │
│  - PP = 2   (Pipeline Parallelism)                  │
│  - CP = 8   (Context Parallelism)                   │
│                                                      │
│  Total GPUs = 8 × 4 × 2 × 8 = 512 GPUs             │
│                                                      │
│  内存分配:                                           │
│  Per GPU:                                           │
│    - Params: 175B / (PP × TP) = 175B / 8 ≈ 22B     │
│    - Seq: 128K / CP = 16K tokens per GPU           │
│    - Batch: 1M / DP = 125K tokens                   │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 优势与劣势

**优势**：
- 支持超长序列（100K+ tokens）
- 激活内存线性减少
- 与TP、PP、DP兼容

**劣势**：
- 实现复杂度高
- 通信模式复杂（Ring All-to-All）
- 对某些attention变体支持有限

**适用场景**：
- 长文档理解
- 代码分析
- 视频理解（长序列）
- 多轮对话历史

## 五、四种并行技术详细对比

### 综合对比表

| 维度 | Data Parallelism | Tensor Parallelism | Pipeline Parallelism | Context Parallelism |
|------|------------------|-------------------|---------------------|---------------------|
| **切分维度** | Batch维度 | 模型参数/隐藏层维度 | 模型层维度 | 序列长度维度 |
| **模型复制** | 完整复制 | 部分复制 | 部分复制 | 完整复制（在CP组内） |
| **内存节省** | 低（需ZeRO） | 高（1/TP） | 中高（1/PP） | 高（1/CP for activations） |
| **通信模式** | All-Reduce | All-Reduce（每层） | Point-to-Point | Ring All-to-All + All-Gather |
| **通信频率** | 每步1次 | 每层2-4次 | 每micro-batch 2次 | 每层3-4次 |
| **通信量** | O(2×ModelSize) | O(2×HiddenSize) | O(MicroBatch×HiddenSize) | O(SeqLen×HiddenSize) |
| **计算效率** | 高（小模型） | 高 | 中（有bubble） | 高（长序列） |
| **实现难度** | 低 | 中 | 高 | 很高 |
| **适用规模** | <100B参数 | 50B-500B+ | 任意深度 | 长序列（>8K） |
| **延迟影响** | 低 | 低 | 中 | 中高 |
| **扩展性** | 差（通信瓶颈） | 有限（TP≤8） | 好 | 好 |

### 通信开销对比

```
┌─────────────────────────────────────────────────────┐
│         Communication Volume per Step               │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Data Parallelism:                                  │
│    Volume = 2 × Model_Size (forward + backward)    │
│    Pattern = All-Reduce                             │
│                                                      │
│  Tensor Parallelism (per layer):                    │
│    Volume = 4 × Hidden_Size × (B × S)              │
│    Pattern = All-Reduce                             │
│                                                      │
│  Pipeline Parallelism (per micro-batch):            │
│    Volume = 2 × MicroBatch_Size × Hidden_Size       │
│    Pattern = Point-to-Point                        │
│                                                      │
│  Context Parallelism (per layer):                   │
│    Volume = 3 × SeqLen × Hidden_Size / CP          │
│    Pattern = Ring All-to-All + All-Gather           │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### GPU利用率对比

```
┌─────────────────────────────────────────────────────┐
│            GPU Utilization Analysis                 │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Ideal (100%): 所有GPU同时进行计算                   │
│                                                      │
│  Data Parallelism: 95-98%                           │
│    - 通信期间短暂空闲                                │
│    - 计算与通信overlap良好                           │
│                                                      │
│  Tensor Parallelism: 85-95%                         │
│    - 每层需要同步                                    │
│    - TP组内紧密耦合                                  │
│                                                      │
│  Pipeline Parallelism: 70-90%                       │
│    - 取决于micro-batch数量                          │
│    - 有pipeline bubble                               │
│                                                      │
│  Context Parallelism: 80-92%                        │
│    - Ring通信有overlap潜力                           │
│    - 取决于序列长度和CP大小                          │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 可扩展性对比

```
┌─────────────────────────────────────────────────────┐
│         Scalability (Maximum Effective GPUs)        │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Data Parallelism:                                  │
│    Max GPUs: ~100-200 (取决于网络带宽)              │
│    Bottleneck: All-Reduce bandwidth                 │
│                                                      │
│  Tensor Parallelism:                                │
│    Max TP Degree: 4-8 (通常)                        │
│    Bottleneck: TP组内通信延迟                        │
│                                                      │
│  Pipeline Parallelism:                              │
│    Max PP Degree: 8-16+                             │
│    Bottleneck: Pipeline bubble                      │
│                                                      │
│  Context Parallelism:                               │
│    Max CP Degree: 8-32+                             │
│    Bottleneck: Ring communication                   │
│                                                      │
│  Combined (3D/4D):                                  │
│    Max GPUs: DP × TP × PP × CP                      │
│    Example: 8 × 8 × 4 × 16 = 4096 GPUs             │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 实际配置建议

```
┌─────────────────────────────────────────────────────┐
│          Recommended Configuration                  │
├─────────────────────────────────────────────────────┤
│                                                      │
│  小模型 (<10B参数):                                  │
│    - DP为主 + (optional) TP=2                       │
│    - 不需要PP和CP                                   │
│                                                      │
│  中等模型 (10B-100B参数):                           │
│    - TP=4-8 + DP                                    │
│    - PP=2 (如果模型很深)                             │
│    - CP=4 (如果序列长>16K)                           │
│                                                      │
│  大模型 (100B-1T参数):                              │
│    - TP=4-8 + PP=4-8                                │
│    - DP根据cluster规模                              │
│    - CP=4-8 (如果需要长序列)                         │
│                                                      │
│  超长序列 (Seq>64K):                                │
│    - CP必须（CP=8-32）                              │
│    - 结合FlashAttention                              │
│    - 考虑Ring Attention或Ulysses                     │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 最佳实践

1. **优先级顺序**：DP → TP → PP → CP
   - 先用DP（最简单）
   - 模型太大用TP
   - 模型太深用PP
   - 序列太长用CP

2. **通信优化**：
   - 尽量使用InfiniBand或NVLink
   - Overlap communication with computation
   - Use mixed precision (FP16/BF16)

3. **内存优化**：
   - ZeRO for DP
   - Gradient checkpointing
   - Activation recomputation

4. **性能调优**：
   - Tuning micro-batch size
   - Balancing pipeline stages
   - Optimizing ring communication for CP

## 总结

四种并行技术各有专长：
- **Data Parallelism**: 基础，适合小模型
- **Tensor Parallelism**: 核心技术，处理大模型
- **Pipeline Parallelism**: 处理深模型
- **Context Parallelism**: 处理长序列

现代LLM训练通常采用**混合并行策略（3D/4D Parallelism）**，根据模型大小、序列长度、集群规模等因素灵活组合这四种技术，以达到最佳的训练效率和资源利用率。

## 参考链接

1. [A rapid guide about LLM's training in parallelism](https://medium.com/@yananchen1116/a-rapid-guide-about-llms-training-in-parallelism-d6edf0dba876)
2. [Megatron-LM Documentation](https://github.com/NVIDIA/Megatron-LM)
3. [DeepSpeed Pipeline Parallelism](https://www.deepspeed.ai/tutorials/pipeline/)
4. [DeepSpeed ZeRO Tutorial](https://www.deepspeed.ai/tutorials/large-models-w-deepspeed/)
5. [Sequence Sharding: How to train long-context LLMs](https://changlan.org/posts/sequence-sharding/)
6. [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691)
7. [Megatron-Core Context Parallelism](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/context_parallel.html)
8. [UvA Deep Learning - 3D Parallelism](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/3d_parallelism.html)
9. [Tensor Parallelism and Sequence Parallelism Analysis](https://insujang.github.io/2024-01-11/tensor-parallelism-and-sequence-parallelism-detailed-analysis/)
10. [NVIDIA Developer - Parallelism for Trillion Parameter Models](https://developer.nvidia.com/blog/demystifying-ai-inference-deployments-for-trillion-parameter-large-language-models/)