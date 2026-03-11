让我深入搜索论文中关于streaming memory的详细内容。






让我详细读取论文中关于streaming memory的更多技术细节：







# SAM 2 的 Streaming Memory 详解

## 一、Streaming Memory 概念与动机

Streaming Memory 是 SAM 2 的核心创新，它使模型能够：
1. **实时处理视频**：逐帧处理视频数据，无需缓存整个视频
2. **跨帧传播信息**：将历史帧的信息传递到当前帧
3. **支持交互式修正**：用户可以在任意帧提供提示来修正分割

与传统的 RNN/LSTM 或 Transformer 的全局记忆不同，Streaming Memory 采用**滑动窗口**的方式维护有限的历史信息，这使得模型在保持长时记忆的同时，计算复杂度与序列长度无关。

---

## 二、Streaming Memory 架构详解

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SAM 2 Streaming Memory                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Frame t-2    Frame t-1         Frame t         Frame t+1              │
│     │            │               │               │                     │
│     ▼            ▼               ▼               ▼                     │
│  ┌─────────┐  ┌─────────┐   ┌─────────┐    ┌─────────┐                │
│  │  Image  │  │  Image  │   │  Image  │    │  Image  │                │
│  │ Encoder │  │ Encoder │   │ Encoder │    │ Encoder │                │
│  └────┬────┘  └────┬────┘   └────┬────┘    └────┬────┘                │
│       │            │               │               │                   │
│       │            │               ▼               │                   │
│       │            │         ┌──────────┐           │                   │
│       │            │         │  Memory  │           │                   │
│       │            │         │ Attention│           │                   │
│       │            │         └────┬─────┘           │                   │
│       │            │              │                 │                   │
│       │            │              ▼                 │                   │
│       │            │         ┌──────────┐           │                   │
│       │            │         │   Mask   │           │                   │
│       │            │         │  Decoder │           │                   │
│       │            │         └────┬─────┘           │                   │
│       │            │              │                 │                   │
│       ▼            ▼              ▼                 ▼                   │
│  ┌─────────┐  ┌─────────┐   ┌─────────┐      ┌─────────┐              │
│  │Memory   │  │Memory   │   │Memory   │      │Memory   │              │
│  │Encoder  │  │Encoder  │   │Encoder  │      │Encoder  │              │
│  └────┬────┘  └────┬────┘   └────┬────┘      └────┬────┘              │
│       │            │               │               │                   │
│       └────────────┼───────────────┼───────────────┘                   │
│                    │               │                                   │
│                    ▼               ▼                                   │
│              ┌─────────────────────────┐                             │
│              │      Memory Bank        │                             │
│              │   (Spatial Memory +     │                             │
│              │    Object Pointers)     │                             │
│              └─────────────────────────┘                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 2.2 Memory Bank (记忆库)

Memory Bank 是 Streaming Memory 的核心组件，维护两类记忆：

#### (1) 空间记忆队列

这是一个 **FIFO (First-In-First-Out)** 队列，存储：
- **N 个最近帧**的记忆特征
- **M 个被提示帧**的记忆特征

默认参数：
```
N = 6  (最近帧的数量)
M = 无上限 (或与视频长度相关)
```

FIFO 队列更新策略：
```
IF frame t 是被提示的 THEN
    存储到 Prompt Memory Queue
ELSE
    存储到 Recent Memory Queue (如果队列未满)
    如果队列已满，移除最旧的帧
END
```

**空间记忆维度**：
```
Spatial Memory: (N_frames × H_mem × W_mem × C_mem)
其中：
- N_frames: 帧数 (默认 N=6)
- H_mem, W_mem: 空间维度 (通常是 64×64)
- C_mem: 通道数 (默认 64)
```

#### (2) Object Pointers (对象指针)

Object Pointers 是**轻量级的高层语义向量**，从 mask decoder 的输出 token 提取：

```
Object Pointer 提取：
Mask Decoder Output Token (256-dim) 
    ↓
Split into 4 tokens of 64-dim
    ↓
Stored as Object Pointer
```

每个帧的 Object Pointer 携带：
- 对象的**高层语义信息**
- 当前帧的分割质量（通过 IoU score）
- 遮挡状态（通过 occlusion score）

**Object Pointer 存储结构**：
```
Object Pointers List = {
    op_1: [64, 64, 64, 64]  (来自 frame 1 的 4 个 64-dim token)
    op_2: [64, 64, 64, 64]  (来自 frame 2 的 4 个 64-dim token)
    ...
    op_k: [64, 64, 64, 64]  (来自 frame k 的 4 个 64-dim token)
}

其中每个 op_i = [op_i^1, op_i^2, op_i^3, op_i^4]，每个 op_i^j ∈ ℝ^64
```

---

### 2.3 Memory Attention (记忆注意力)

Memory Attention 是 Streaming Memory 的核心模块，负责将当前帧特征与 Memory Bank 中的记忆进行条件化。

#### (1) 架构设计

Memory Attention 由 **L = 4 个 Transformer 块**堆叠而成：

```
对于第 ℓ 个块 (ℓ = 1, 2, ..., L)：

输入：X_ℓ  (当前帧特征)

1. Self-Attention (处理当前帧内部关系):
   SA(X_ℓ) = softmax(Q_self(X_ℓ) K_self(X_ℓ)^T / √d) V_self(X_ℓ)
   
   Q_self(X_ℓ) = X_ℓ W_Q^ℓ
   K_self(X_ℓ) = X_ℓ W_K^ℓ
   V_self(X_ℓ) = X_ℓ W_V^ℓ

2. Cross-Attention to Memories (与记忆交互):
   CA(X_ℓ) = softmax(Q_cross(X_ℓ) K_cross(M)^T / √d) V_cross(M)
   
   Q_cross(X_ℓ) = X_ℓ W_Qc^ℓ
   K_cross(M) = concat([M_spatial, M_pointers]) W_Kc^ℓ
   V_cross(M) = concat([M_spatial, M_pointers]) W_Vc^ℓ
   
   其中 M_spatial 是空间记忆，M_pointers 是 object pointers

3. MLP (前馈层):
   MLP(X) = σ(X W_1 + b_1) W_2 + b_2

输出：X_{ℓ+1} = LayerNorm(X_ℓ + SA(X_ℓ) + CA(X_ℓ) + MLP(...))
```

#### (2) 位置编码：2D-RoPE

SAM 2 在 Memory Attention 中使用 **2D Rotary Positional Embedding (2D-RoPE)**。

**RoPE 原理**：
对于 2D 空间位置 $(x, y)$，定义旋转角度：

```
θ_x^(i) = x / θ_base^(2i/d)
θ_y^(i) = y / θ_base^(2i/d)

其中：
- i: 维度索引 (i = 0, 1, ..., d/2)
- d: 注意力头维度
- θ_base: 基础角度 (通常是 10000 或类似值)
```

**2D-RoPE 应用**：
对于查询向量 $q$ 和键向量 $k$ 的第 $i$ 对维度 $(q_{2i}, q_{2i+1})$ 和 $(k_{2i}, k_{2i+1})$：

```
对 x 维度：
q'_2i = q_{2i} cos(θ_x^(i)) - q_{2i+1} sin(θ_x^(i))
q'_{2i+1} = q_{2i} sin(θ_x^(i)) + q_{2i+1} cos(θ_x^(i))

对 y 维度：
q''_2i = q'_2i cos(θ_y^(i)) - q'_{2i+1} sin(θ_y^(i))
q''_{2i+1} = q'_2i sin(θ_y^(i)) + q'_{2i+1} cos(θ_y^(i))

对 k 向量应用相同的旋转得到 k''

最终注意力分数：
Attention(q, k) = (q'' · k'') / √d
```

**关键点**：
- Object Pointer tokens **不使用 RoPE**，因为它们没有空间对应关系
- RoPE 提供**相对位置**感知能力，使模型理解记忆中不同空间位置的相对关系

#### (3) 时间位置编码

Memory Bank 中存储的记忆会嵌入**时间位置信息**：

```
对于第 τ 个记忆（相对当前帧的时间偏移）：

Temporal Encoding = TE(τ)

其中：
- TE 是可学习的 sinusoidal embedding
- τ ∈ {1, 2, ..., N} 用于最近 N 帧
- Prompted frames 不编码时间位置（因为它们在推理时可能与训练时的位置差异很大）
```

时间位置编码的公式：

```
TE(τ)_2i = sin(τ / freq_base^(2i/d_temp))
TE(τ)_{2i+1} = cos(τ / freq_base^(2i/d_temp))

其中：
- i: 维度索引
- d_temp: 时间编码维度
- freq_base: 频率基础参数
```

---

### 2.4 Memory Encoder (记忆编码器)

Memory Encoder 将预测的 mask 转换为记忆特征。

**Memory Encoder 架构**：

```
输入：
1. Predicted Mask M_t (H × W)
2. Unconditioned Frame Embedding F_t (H_mem × W_mem × C_frame)

处理流程：

Step 1: Mask 下采样
M_t_down = ConvDownsample(M_t)
其中 ConvDownsample 是步长为 4 或 8 的卷积层

Step 2: 特征融合
M_fused = M_t_down + F_t

Step 3: 轻量级卷积融合
Memory = LightConvLayers(M_fused)
LightConvLayers 包含多个 3×3 卷积层，逐层处理

输出：
Memory ∈ ℝ^(H_mem × W_mem × C_mem)
默认：H_mem × W_mem = 64 × 64, C_mem = 64
```

**关键设计**：
- Memory Encoder **不使用额外的图像编码器**，直接复用 Hiera Encoder 的输出
- 这使得记忆特征能够受益于强大的图像表示

---

### 2.5 Mask Decoder 与 Occlusion Prediction

#### (1) Mask Decoder

Mask Decoder 在 SAM 基础上进行了增强：

```
输入：
- Memory Attention 条件化的帧嵌入 F'_t
- 提示 (点击、框、mask)

架构：
堆叠 "two-way" Transformer 块，其中：
1. Self-Attention (帧到帧)
2. Token-to-Image Attention (提示token到图像特征)
3. Image-to-Token Attention (图像特征到提示token)
4. MLP

输出：
1. 分割 mask(s)
2. IoU 分数(s)
3. Occlusion 分数
4. Object Pointer token
```

#### (2) Occlusion Prediction

SAM 2 引入了遮挡预测头：

```
Occlusion Prediction:
输入：OcclusionToken (256-dim，与 mask token 并行输出)
输出：OcclusionScore ∈ [0, 1]

公式：
OcclusionScore = σ(MLP(OcclusionToken))

其中：
- σ 是 sigmoid 函数
- MLP 是多层感知机
```

**Occlusion Embedding 处理**：

```
如果 OcclusionScore > threshold (默认 0.5):
    在 Memory Bank 中添加 learned occlusion embedding 到该帧的记忆特征
    
Memory_with_occlusion = Memory + OcclusionEmbedding

其中 OcclusionEmbedding ∈ ℝ^(H_mem × W_mem × C_mem) 是可学习参数
```

---

## 三、Streaming Memory 工作流程

### 3.1 完整推理流程

```
初始化：
Memory Bank = 空队列

对于视频的每一帧 t = 1, 2, ..., T：

1. Image Encoding:
   Frame Embedding F_t = ImageEncoder(VideoFrame_t)

2. Memory Attention:
   IF Memory Bank 非空 THEN:
       F'_t = MemoryAttention(F_t, Memory Bank)
   ELSE:
       F'_t = F_t
   END

3. Mask Decoding:
   IF t 有提示 THEN:
       Mask_t = MaskDecoder(F'_t, Prompts_t)
   ELSE:
       Mask_t = MaskDecoder(F'_t)
   END

4. Memory Encoding:
   Memory_t = MemoryEncoder(Mask_t, F_t)

5. Memory Bank 更新:
   IF t 是被提示的帧 THEN:
       添加 Memory_t 到 Prompt Memory Queue
       生成 Object Pointer op_t
   ELSE:
       添加 Memory_t 到 Recent Memory Queue (如果空间允许)
       生成 Object Pointer op_t
       如果 Queue 已满，移除最旧的帧
   END

6. Occlusion 处理:
   IF OcclusionScore_t > threshold THEN:
       向 Memory_t 添加 occlusion embedding
   END

输出：
对于所有 t = 1, 2, ..., T:
    Mask_t (分割结果)
    IoU_t (质量分数)
    Occlusion_t (遮挡状态)
```

### 3.2 交互式修正流程

```
用户在 frame t 提供 prompt P_t:

1. 立即响应:
   Mask_t = MaskDecoder(F'_t, P_t)
   
2. 更新 Memory Bank:
   Memory_t = MemoryEncoder(Mask_t, F_t)
   添加到 Prompt Memory Queue
   
3. 传播:
   对于后续帧 t+1, t+2, ...:
       F'_{t+k} = MemoryAttention(F_{t+k}, Memory Bank)
       Mask_{t+k} = MaskDecoder(F'_{t+k})
       
4. 迭代修正 (可选):
   如果用户在 frame t' 提供修正:
       重复步骤 1-3
```

---

## 四、技术细节与公式

### 4.1 Memory Bank 的数学表示

```
Memory Bank = (M_spatial, M_pointers)

其中：
M_spatial = [m_1, m_2, ..., m_{n}]
每个 m_i ∈ ℝ^(H × W × C) 是第 i 帧的空间记忆

M_pointers = [op_1, op_2, ..., op_{n}]
每个 op_i ∈ ℝ^(4 × 64) 是第 i 帧 object pointer 的 4 个 token

n = min(t, N) (不超过 N 个最近帧)
```

### 4.2 Cross-Attention 公式详解

```
Cross-Attention 公式：

CA(F_t, M) = softmax(Q K^T / √d_k) V

其中：
Q = F_t W_Q  ∈ ℝ^(N_tokens × d_k)
K = [M_spatial; M_pointers] W_K  ∈ ℝ^(N_memory × d_k)
V = [M_spatial; M_pointers] W_V  ∈ ℝ^(N_memory × d_v)

注意力权重矩阵 A ∈ ℝ^(N_tokens × N_memory)：
A_{i,j} = exp(Q_i · K_j / √d_k) / Σ_k exp(Q_i · K_k / √d_k)

输出：
CA(F_t, M) = A V  ∈ ℝ^(N_tokens × d_v)

最终应用 RoPE：
Q_i'' = RoPE2D(Q_i, x, y)
K_j'' = RoPE2D(K_j, x_j, y_j)
```

### 4.3 Object Pointer 生成公式

```
从 Mask Decoder 的输出 token t_mask (256-dim):

op_t = SplitAndProject(t_mask)

Splitting:
t_mask = [t_1, t_2, t_3, t_4], 每个 t_i ∈ ℝ^64

Projection (可选):
op_t = [t_1 W_p, t_2 W_p, t_3 W_p, t_4 W_p]
每个 op_t^i ∈ ℝ^64

存储到 Memory Bank:
M_pointers = M_pointers ∪ {op_t}
```

---

## 五、消融实验分析

### 5.1 Memory Size (记忆数量)

| N (最近帧数) | J&F | mIoU | MOSE dev | SA-V val | 9 zero-shot | Speed |
|-------------|-----|------|----------|----------|-------------|-------|
| 4 | 71.1 | - | - | - | - | - |
| 6 (默认) | 70.7 | - | - | - | - | 1.00× |
| 8 | 71.1 | - | - | - | - | - |

**结论**：N=6 是性能和速度的良好平衡点。

### 5.2 Positional Encoding 消融

| 配置 | J&F | MOSE dev | SA-V val | LVOSv2 val | 9 zero-shot | Speed |
|------|-----|----------|----------|------------|-------------|-------|
| RPB only | 73.0 | 68.3 | 71.6 | 70.7 | 1.00× | 59.7 |
| RPB + 2D-RoPE | 73.6 | 67.9 | 71.0 | 71.5 | 0.93× | 60.0 |
| 无 RoPE | 72.8 | 67.1 | 70.3 | 70.3 | 1.04× | 59.9 |

**结论**：移除 RPB 并使用 2D-RoPE 带来更好的性能，同时允许使用 FlashAttention-2 提升速度。

### 5.3 Object Pointers 消融

| Object Pointers | GRU | MOSE dev | SA-V val | LVOSv2 val | 9 zero-shot | Speed |
|-----------------|-----|----------|----------|------------|-------------|-------|
| ✓ | - | 64.5 | 67.0 | 70.9 | 1.00× | 59.9 |
| - | ✓ | 65.3 | 68.9 | 70.5 | 0.97× | 60.0 |
| - | - | 68.3 | 71.6 | 70.7 | 1.00× | 59.7 |

**结论**：Object Pointers 在 SA-V val 和 LVOSv2 等挑战性数据集上带来显著提升，虽然对平均性能影响不大。

### 5.4 GRU Recurrent Memory 消融

```
对比直接存储 vs. GRU 状态：

直接存储 (默认):
M_bank[t] = MemoryEncoder(Mask_t)

GRU:
h_t = GRU(MemoryEncoder(Mask_t), h_{t-1})
M_bank[t] = h_t
```

**结果**：GRU 并未带来改进（除 LVOSv2 略有提升），直接存储记忆更简单高效。

---

## 六、Streaming Memory 的优势与局限

### 6.1 优势

1. **实时处理**：逐帧处理，支持实时视频分割
2. **有限内存**：FIFO 队列限制内存使用，O(1) 空间复杂度
3. **交互性**：支持在任意帧进行修正
4. **鲁棒性**：Object Pointers 提供高层语义，增强遮挡处理能力

### 6.2 局限性

1. **Shot Boundary**：在镜头切换时可能丢失对象
2. **长时遮挡**：长时间遮挡后难以恢复
3. **相似对象**：多个外观相似对象可能混淆
4. **独立处理**：每个对象独立处理，缺乏对象间通信

---

## 七、参考链接

- **SAM 2 Demo**: https://sam2.metademolab.com
- **SAM 2 Code**: https://github.com/facebookresearch/sam2
- **SAM 2 Website**: https://ai.meta.com/sam2
- **Paper**: https://arxiv.org/abs/2408.00714
- **RoPE Paper**: https://arxiv.org/abs/2104.09864
- **Hiera**: https://arxiv.org/abs/2305.12994

Streaming Memory 是 SAM 2 的核心创新，它通过精心设计的 Memory Bank、Memory Attention 和 Object Pointer 机制，实现了高效的实时视频分割，同时保持了强大的跨帧传播能力和交互性。