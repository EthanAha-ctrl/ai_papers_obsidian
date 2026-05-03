# 《How to Scale Your Model: A Systems View of LLMs on TPUs》深度解析

这是一篇来自 **Google DeepMind** 团队的系统性技术文章，发表于 2025 年 2 月 4 日，旨在揭开 LLM 扩展的"黑魔法"面纱。让我从第一性原理出发，为你详细拆解这篇文章的核心内容。

---

## 一、文章定位与核心价值

### 1.1 为什么这篇文章重要？

**三四年前的现状**：大多数 ML 研究者不需要关心硬件细节
**今天的现实**：即使"小"模型也运行在硬件极限边缘

> "A 20% win on benchmarks is irrelevant if it comes at a 20% cost to roofline efficiency."

这句话揭示了一个残酷的现实：**模型架构创新如果不能高效扩展，等于零**。

### 1.2 核心问题定义

文章的核心问题是：

> **"给定一个模型大小和若干芯片，如何并行化模型以保持 strong scaling 状态？"**

---

## 二、Roofline Analysis（屋顶线分析）——性能瓶颈的第一性原理

### 2.1 三大瓶颈

任何算法的性能受限于三个因素：

| 瓶颈类型 | 描述 | 公式化表达 |
|---------|------|-----------|
| **Compute-bound** | 计算能力饱和 | $T_{\text{compute}} > T_{\text{memory}}, T_{\text{communication}}$ |
| **Memory-bound** | 内存带宽瓶颈 | $T_{\text{memory}} > T_{\text{compute}}, T_{\text{communication}}$ |
| **Communication-bound** | 通信带宽瓶颈 | $T_{\text{communication}} > T_{\text{compute}}, T_{\text{memory}}$ |

### 2.2 Roofline 模型详解

**Roofline Model** 是由加州伯克利实验室提出的性能分析模型：

$$
\text{Attainable Performance} = \min\left(\text{Peak FLOPs}, \text{Memory Bandwidth} \times \text{Arithmetic Intensity}\right)
$$

**关键变量**：
- **Arithmetic Intensity (算术强度)**：$I = \frac{\text{FLOPs}}{\text{Bytes transferred}}$，单位：FLOPs/Byte
- **Peak FLOPs**：硬件峰值计算能力
- **Memory Bandwidth**：内存带宽，单位：Bytes/s

**Roofline 图形解释**：
```
Performance (FLOPs/s)
    |
    |      /
    |     /  ← Compute-bound regime (斜率 = Peak FLOPs)
    |    /
    |___/____  ← Memory-bound regime (斜率 = Memory Bandwidth)
    |  /
    | /
    |/________ AI (Arithmetic Intensity)
```

**转折点 (Ridge Point)**：
$$
\text{AI}_{\text{ridge}} = \frac{\text{Peak FLOPs}}{\text{Memory Bandwidth}}
$$

当 $\text{AI} > \text{AI}_{\text{ridge}}$ 时，进入 compute-bound 区域。

### 2.3 实际案例：矩阵乘法的 Roofline 分析

假设：
- 矩阵乘法 $C = A \times B$，其中 $A \in \mathbb{R}^{m \times k}$，$B \in \mathbb{R}^{k \times n}$
- **FLOPs**：$2mnk$ (每个元素需要 $k$ 次乘法和 $k-1$ 次加法)
- **Memory Access**：读取 $A$ ($mk$ elements)、$B$ ($kn$ elements)，写入 $C$ ($mn$ elements)

**Arithmetic Intensity**：
$$
\text{AI} = \frac{2mnk}{(mk + kn + mn) \times \text{element\_size}}
$$

对于 FP32 (4 bytes per element)：
$$
\text{AI} \approx \frac{2mnk}{4(mk + kn + mn)} = \frac{mnk}{2(mk + kn + mn)}
$$

当 $m = n = k$ 时：
$$
\text{AI} = \frac{k^3}{6k^2} = \frac{k}{6}
$$

**结论**：矩阵越大，算术强度越高，越容易进入 compute-bound 区域。

---

## 三、TPU 硬件架构深度解析

### 3.1 TPU vs GPU 的核心差异

| 特性 | TPU | GPU |
|-----|-----|-----|
| **计算单元** | Matrix Multiply Unit (MXU) | CUDA Cores / Tensor Cores |
| **计算模式** | Systolic Array | SIMT |
| **精度** | bfloat16 优化 | FP16/FP32/FP64 |
| **互连** | ICI (Inter-Chip Interconnect) | NVLink / NVSwitch |
| **编程模型** | JAX / TensorFlow | CUDA |

### 3.2 Systolic Array 原理

**TPU 的核心计算单元**是 **Systolic Array**，这是一个二维的脉动阵列：

```
        权重流 →
    ┌─┬─┬─┬─┬─┐
  ↓ │●│●│●│●│●│
激 ├─┼─┼─┼─┼─┤
励 │●│●│●│●│●│
流 ├─┼─┼─┼─┼─┤
  ↓ │●│●│●│●│●│
    └─┴─┴─┴─┴─┘
        ↓
      部分和流
```

**工作原理**：
1. 数据从左侧和顶部流入
2. 每个处理单元 (PE) 执行 $a_{ij} \times w_{ij} + s_{ij}$ (multiply-accumulate, MAC)
3. 结果向下传递，继续累积
4. 最终从底部输出

**优势**：
- **数据复用**：每个数据在阵列中流动时被多次使用
- **低功耗**：减少内存访问次数
- **高吞吐**：适合规则的大矩阵运算

### 3.3 TPU 内存层次结构

```
┌─────────────────────────────────────┐
│         HBM (High Bandwidth Memory) │  ← 主内存，~1-2TB/s
├─────────────────────────────────────┤
│         Vector Memory               │  ← 向量单元专用内存
├─────────────────────────────────────┤
│         Accumulator                 │  ← 累加器，MXU 输出缓冲
└─────────────────────────────────────┘
```

### 3.4 ICI (Inter-Chip Interconnect) 互连拓扑

TPU Pod 通过 **ICI** 连接，形成 **3D Torus 拓扑**：

```
     ┌───┐     ┌───┐     ┌───┐
    /│   │\   /│   │\   /│   │\
   / │   │ \ / │   │ \ / │   │ \
  │  └───┘  X  └───┘  X  └───┘  │
  │    │      │      │      │    │
  │  ┌───┐  X  ┌───┐  X  ┌───┐  │
   \ │   │ / \ │   │ / \ │   │ /
    \│   │/   \│   │/   \│   │/
     └───┘     └───┘     └───┘
```

**关键参数**：
- **带宽**：每条链路 ~200-500 GB/s (取决于代次)
- **延迟**：~1-10 μs
- **拓扑**：3D Torus（x, y, z 三个维度）

---

## 四、Transformer 数学详解

### 4.1 参数量计算

**标准 Transformer Layer** 包含：

| 组件 | 公式 | 参数量 |
|-----|------|--------|
| **QKV 投影** | $W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{head}} \times n_{\text{heads}}}$ | $3 \times d_{\text{model}} \times d_{\text{model}}$ |
| **输出投影** | $W_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ | $d_{\text{model}}^2$ |
| **FFN Up** | $W_1 \in \mathbb{R}^{d_{\text{model}} \times 4d_{\text{model}}}$ | $4d_{\text{model}}^2$ |
| **FFN Down** | $W_2 \in \mathbb{R}^{4d_{\text{model}} \times d_{\text{model}}}$ | $4d_{\text{model}}^2$ |
| **LayerNorm** | $\gamma, \beta \in \mathbb{R}^{d_{\text{model}}}$ | $4d_{\text{model}}$ (通常忽略) |

**每层总参数量**：
$$
P_{\text{layer}} \approx 12d_{\text{model}}^2
$$

**总参数量**（假设 $n_{\text{layers}}$ 层）：
$$
P_{\text{total}} \approx 12n_{\text{layers}}d_{\text{model}}^2
$$

### 4.2 FLOPs 计算

#### 4.2.1 矩阵乘法 FLOPs

对于 $C = A \times B$，其中 $A \in \mathbb{R}^{m \times k}$，$B \in \mathbb{R}^{k \times n}$：

$$
\text{FLOPs} = 2mnk
$$

**解释**：
- 每个输出元素 $c_{ij}$ 需要 $k$ 次乘法和 $k-1$ 次加法，约 $2k$ FLOPs
- 共有 $mn$ 个输出元素
- 总 FLOPs = $2k \times mn = 2mnk$

#### 4.2.2 Attention FLOPs

**Forward Pass**：
$$
\text{FLOPs}_{\text{attention}} = 2L^2 d_{\text{model}} + 2L d_{\text{model}}^2
$$

其中 $L$ 是序列长度。

**分解**：
1. **QKV 投影**：$3 \times 2L d_{\text{model}}^2$ (三个矩阵乘法)
2. **Attention Scores**：$2L^2 d_{\text{head}} \times n_{\text{heads}} = 2L^2 d_{\text{model}}$
3. **Softmax**：$O(L^2 n_{\text{heads}})$ (通常忽略)
4. **Attention Output**：$2L^2 d_{\text{model}}$
5. **Output Projection**：$2L d_{\text{model}}^2$

### 4.3 KV Cache 内存计算

**KV Cache** 存储所有之前 token 的 Key 和 Value：

$$
\text{KV Cache Size} = 2 \times n_{\text{layers}} \times L \times d_{\text{model}} \times \text{bytes\_per\_element}
$$

**实际案例**：LLaMA 3 70B
- $n_{\text{layers}} = 80$
- $d_{\text{model}} = 8192$
- 序列长度 $L = 8192$
- FP16 (2 bytes)

$$
\text{KV Cache} = 2 \times 80 \times 8192 \times 8192 \times 2 = 21.5 \text{ GB}
$$

---

## 五、并行化策略详解

### 5.1 四种主要并行策略

#### 5.1.1 Data Parallelism (DP)

**原理**：每个设备持有完整模型副本，处理不同数据批次

```
Device 0: Model Copy → Batch 0 → Grads 0 ─┐
Device 1: Model Copy → Batch 1 → Grads 1 ─┼─→ AllReduce → Update
Device 2: Model Copy → Batch 2 → Grads 2 ─┤
Device 3: Model Copy → Batch 3 → Grads 3 ─┘
```

**通信量**：
$$
\text{Comm}_{\text{DP}} = 2 \times P_{\text{model}} \times \text{num\_devices}
$$

**局限性**：每个设备需要存储完整模型，内存压力大

#### 5.1.2 Tensor Parallelism (TP)

**原理**：将模型权重切分到多个设备

**Attention 层的 TP**：
```
          ┌─────────┐
          │   Q     │
Input ──→ │   K  ×  │──→ Attention Output
          │   V     │
          └─────────┘
              ↓
        按注意力头切分
    ┌──────┬──────┬──────┐
    │Head 0│Head 1│Head 2│
    │Device│Device│Device│
    │  0   │  1   │  2   │
    └──────┴──────┴──────┘
```

**数学表达**：
- 将 $W_Q, W_K, W_V$ 按列切分
- 每个设备计算部分 attention heads
- 需要 AllGather 或 AllReduce 合并结果

**通信量**：
$$
\text{Comm}_{\text{TP}} = O\left(\frac{L \times d_{\text{model}} \times n_{\text{layers}}}{\text{TP\_degree}}\right)
$$

#### 5.1.3 Pipeline Parallelism (PP)

**原理**：将模型层切分到不同设备，形成流水线

```
Time →
       t0    t1    t2    t3    t4    t5    t6
Dev0: [B0]─────────────────[B1]────────────
Dev1: ────[B0]─────────────────[B1]────────
Dev2: ────────[B0]─────────────────[B1]────
Dev3: ────────────[B0]─────────────────[B1]
```

**关键问题：Pipeline Bubbles**

$$
\text{Bubble Ratio} = \frac{PP_{\text{degree}} - 1}{\text{num\_microbatches}}
$$

**解决方案**：
- **GPipe**：将 batch 切分为 micro-batches
- **1F1B** (One Forward One Backward)：交替执行前向和反向

#### 5.1.4 Expert Parallelism (EP)

**原理**：Mixture of Experts (MoE) 架构中，将不同 experts 放置在不同设备

```
         Router
           │
    ┌──────┼──────┐
    ↓      ↓      ↓
Expert 0 Expert 1 Expert 2
(Device 0)(Device 1)(Device 2)
```

**通信模式**：
- **All-to-All**：将 token 路由到对应 expert 所在设备
- **通信量**：$O(L \times d_{\text{model}})$

### 5.2 内存优化技术

#### 5.2.1 Rematerialization (Activation Checkpointing)

**原理**：在前向传播时不存储所有中间激活，而是在反向传播时重新计算

**内存节省**：
$$
\text{Memory}_{\text{checkpointed}} = O(\sqrt{P_{\text{model}}})
$$

**代价**：增加约 33% FLOPs（需要重新计算）

#### 5.2.2 ZeRO (Zero Redundancy Optimizer)

**三个阶段**：

| Stage | 分片内容 | 内存节省 |
|-------|---------|---------|
| **ZeRO-1** | Optimizer States | $4\times$ |
| **ZeRO-2** | + Gradients | $8\times$ |
| **ZeRO-3** | + Model Parameters | $N\t$ (N = 设备数) |

**数学分析**：
- 原始内存：$M_{\text{original}} = 2P + 2P + 12P = 16P$ (FP16 参数、梯度、Adam 状态)
- ZeRO-3 后：$M_{\text{ZeRO-3}} = \frac{16P}{N}$

---

## 六、训练与推理成本估算

### 6.1 训练成本公式

**训练时间估算**：
$$
T_{\text{train}} = \frac{N_{\text{tokens}} \times 6P_{\text{model}}}{N_{\text{devices}} \times \text{Peak FLOPs} \times \text{MFU}}
$$

其中：
- $N_{\text{tokens}}$：训练 token 数量
- $P_{\text{model}}$：模型参数量
- $6P_{\text{model}}$：每个 token 的 forward + backward FLOPs
- **MFU (Model FLOPs Utilization)**：实际达到的效率比例（通常 30-60%）

**成本估算**：
$$
\text{Cost} = T_{\text{train}} \times \text{price\_per\_hour} \times N_{\text{devices}}
$$

### 6.2 推理延迟分析

**推理的两个阶段**：

1. **Prefill Phase**：处理输入 prompt
   - Compute-bound
   - 延迟：$O\left(\frac{L_{\text{input}}^2 \times d_{\text{model}}}{\text{FLOPs}}\right)$

2. **Decode Phase**：逐 token 生成
   - Memory-bound（需要加载全部权重）
   - 延迟：$O\left(\frac{P_{\text{model}}}{\text{Memory Bandwidth}}\right)$ per token

**关键指标**：
- **Time to First Token (TTFT)**：首 token 延迟
- **Time Per Output Token (TPOT)**：每 token 生成时间
- **Throughput**：tokens/second

---

## 七、JAX 编程模型与 SPMD

### 7.1 GSPMD (General SPMD)

JAX 的核心抽象是 **`jax.pjit`** 和 **`jax.pmap`**，允许用户指定张量的分片方式：

```python
from jax.sharding import PartitionSpec as P

# 定义分片策略
mesh = jax.sharding.Mesh(
    jax.devices(), 
    ['data', 'model']
)

# 参数分片
weight_sharding = jax.sharding.NamedSharding(
    mesh, 
    P('model', None)  # 按模型维度分片
)

# 数据分片
data_sharding = jax.sharding.NamedSharding(
    mesh, 
    P('data', None)  # 按数据维度分片
)
```

### 7.2 常见通信原语

| 操作 | 功能 | 通信复杂度 |
|-----|------|-----------|
| **AllReduce** | 所有设备求和并广播结果 | $O(N)$ |
| **AllGather** | 收集所有设备的数据 | $O(N)$ |
| **ReduceScatter** | 求和后分片结果 | $O(N)$ |
| **AllToAll** | 全连接数据交换 | $O(N)$ |

---

## 八、文章的实践价值

### 8.1 你能从这篇文章学到什么？

1. **估算模型训练成本**
   - 给定模型大小、数据量、硬件规格
   - 计算理论最短时间和实际预期时间

2. **选择并行策略**
   - 小模型（<1B）：Data Parallelism
   - 中等模型（1B-100B）：TP + DP
   - 大模型（>100B）：TP + PP + DP

3. **优化推理服务**
   - KV Cache 管理
   - Batch size 选择
   - Latency-Throughput trade-off

### 8.2 实际案例：LLaMA 3 70B

文章在 Section 6 和 Section 8 以 LLaMA 3 为例，展示了如何应用这些原理：

**训练估算**：
- 参数量：70B
- 训练数据：~15T tokens
- 假设 MFU = 50%
- TPU v5 peak FLOPs ≈ 277 TFLOPS (bf16)

$$
T_{\text{train}} \approx \frac{15 \times 10^{12} \times 6 \times 70 \times 10^9}{N \times 277 \times 10^{12} \times 0.5}
$$

---

## 九、延伸阅读与参考

### 9.1 核心参考文献

1. **Transformer 原论文**：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. **Roofline Model**：[Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf)
3. **ZeRO 论文**：[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
4. **Megatron-LM**：[Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)

### 9.2 硬件架构参考

- **TPU 论文**：[In-Datacenter Performance Analysis of a Tensor Processing Unit](https://arxiv.org/abs/1704.04760)
- **GPU 架构**：NVIDIA Whitepapers on Ampere/Hopper Architecture

### 9.3 在线资源

- **JAX 文档**：https://jax.readthedocs.io/
- **TPU 性能分析工具**：TensorBoard Profiler

---

## 十、总结

这篇文章的价值在于它提供了一个**从硬件到算法的统一视角**：

```
┌─────────────────────────────────────────────────────┐
│                    Application Layer                │
│              (LLM Training / Inference)             │
├─────────────────────────────────────────────────────┤
│                 Parallelism Layer                   │
│        (DP / TP / PP / EP + Memory Optimization)    │
├─────────────────────────────────────────────────────┤
│                   System Layer                      │
│           (Roofline Analysis, Communication)        │
├─────────────────────────────────────────────────────┤
│                  Hardware Layer                     │
│        (TPU Architecture, Memory Hierarchy)         │
└─────────────────────────────────────────────────────┘
```

**核心洞见**：
1. **性能优化不是黑魔法**，而是可以量化的工程问题
2. **理解硬件是高效扩展的前提**
3. **不同规模的模型需要不同的并行策略组合**

这篇文章是每一位想要深入理解 LLM 系统层面的研究者和工程师的必读材料。