# Axial Rotary Positional Embeddings (轴向旋转位置编码) 深度解析

## 一、核心直觉与动机

**Axial RoPE** 是对原始 **Rotary Positional Embeddings (RoPE)** 的多维扩展，旨在让 Transformer 的 attention 机制能够处理具有**多个空间或语义轴**的数据（如图像的 height/width、时间序列的多个维度）。

**核心直觉**：想象你在处理一张图像，token 的位置不仅是一维的（像 NLP 中的句子），而是二维的（h, w）。如果只用一维旋转，位置信息会"线性化"丢失空间关系。Axial RoPE 的想法是：**沿着每个轴独立旋转**，让 attention score 能感知到多维相对位置差异。

## 二、数学公式详解

### 2.1 基础 RoPE 公式

对于位置 $t$ 的 $d$ 维 query 或 key 向量 $x_t$，RoPE 应用一个**块对角正交旋转矩阵**：

$$R_t = \mathrm{blockdiag}\left( R_{t,1}, R_{t,2}, \ldots, R_{t,d/2} \right)$$

其中每个块 $R_{t,i}$ 是一个 $2 \times 2$ 旋转矩阵：

$$
R_{t,i} =
\begin{pmatrix}
\cos(t\theta_i) & -\sin(t\theta_i) \\
\sin(t\theta_i) & \cos(t\theta_i)
\end{pmatrix}
$$

**变量说明**：
- $t$: 绝对位置索引
- $d$: query/key 的 embedding 维度
- $i$: 第 $i$ 个旋转块的索引（从 1 到 $d/2$）
- $\theta_i$: 第 $i$ 个块的旋转频率基准
- $\theta_i = 10000^{-2(i-1)/d}$: 从论文 [2104.09864] 继承的频率设计

**频率 $\theta_i$ 的设计直觉**：
- $i=1$ 时，$\theta_1 = 10000^{-0/d} = 1$（低频，编码长距离依赖）
- $i=d/2$ 时，$\theta_{d/2} = 10000^{-2(d/2-1)/d} \approx 10000^{-1}$（高频，编码精细位置）
- 类似 Fourier 变换中的频率 bank，提供多尺度位置编码

### 2.2 相对位置不变性证明

旋转后的 query 和 key：

$$\widetilde{q}_t = R_t q_t, \qquad \widetilde{k}_u = R_u k_u$$

Attention score：

$$
\begin{aligned}
\mathrm{score}_{t,u} &= \widetilde{q}_t^\top \widetilde{k}_u \\
&= (R_t q_t)^\top (R_u k_u) \\
&= q_t^\top R_t^\top R_u k_u \\
&= q_t^\top R_t^{-1} R_u k_u \quad (\text{因为 } R_t \text{ 正交, } R_t^\top = R_t^{-1}) \\
&= q_t^\top R_{u-t} k_u \quad (\text{群性质: } R_a^\top R_b = R_{b-a})
\end{aligned}
$$

**关键洞察**：score 只取决于相对偏移 $(u-t)$，不取决于绝对位置 $t$ 或 $u$ 单独！

### 2.3 Axial RoPE 公式

对于 $D$ 维位置坐标 $(s_i^{(1)}, s_i^{(2)}, \ldots, s_i^{(D)})$，将 embedding 维度均分为 $D$ 个 slice：

$$
\widetilde{q}_i = \left[ R_{s_i^{(1)}} q_i^{(1)};\; R_{s_i^{(2)}} q_i^{(2)};\; \ldots;\; R_{s_i^{(D)}} q_i^{(D)} \right]
$$

**变量说明**：
- $s_i^{(j)}$: 第 $i$ 个 token 在第 $j$ 个轴上的位置
- $q_i^{(j)}$: query 向量中分配给第 $j$ 个轴的 slice（维度 $d/D$）
- $R_{s_i^{(j)}}$: 针对第 $j$ 个轴的位置 $s_i^{(j)}$ 的旋转矩阵

**架构示意**（以 2D 为例，$d=128$）：

```
q_i = [q_i^{(1)}: 1-64维 | q_i^{(2)}: 65-128维]
       ↓                      ↓
    R_{h} 旋转              R_{w} 旋转
       ↓                      ↓
q̃_i = [q̃_i^{(1)}: 1-64维 | q̃_i^{(2)}: 65-128维]
```

## 三、理论性质与光谱解释

### 3.1 复数表示与 Fourier 级数

用复数表示，每对 $(2k, 2k+1)$ 维度的变换为：

$$z_k \mapsto e^{i\theta_k t} z_k$$

Attention score 分解为 Fourier 级数：

$$
\mathrm{score}_{t,u} \propto \sum_{k=1}^{d/2} \left[ q_t^{(2k)} k_u^{(2k)} + q_t^{(2k+1)} k_u^{(2k+1)} \right] \cos \left( \theta_k (u - t) \right)
$$

**变量说明**：
- $z_k$: 第 $k$ 个复数维度（来自 embedding 的第 $2k$ 和 $2k+1$ 维）
- $k$: Fourier 分量索引（$k=1$ 是最低频，$k=d/2$ 是最高频）
- $\cos(\theta_k \cdot \text{offset})$: 相对位置偏移的余弦调制

**光谱直觉**：
- 低频 $\theta_k$：长距离依赖，"全局"位置信息
- 高频 $\theta_k$：短距离依赖，"局部"精细位置
- 类似于一组**固定频率的滤波器 bank**

### 3.2 群论视角

RoPE 是 **SO(d) 群** 中一个**单参数子群**的作用：

$$R: \mathbb{Z} \to \mathrm{SO}(d), \quad t \mapsto R_t$$

满足群性质：
1. $R_0 = I$（单位矩阵）
2. $R_a R_b = R_{a+b}$（半群性质）
3. $R_t^{-1} = R_{-t}$（逆元素）

**Axial RoPE** 将这个子群扩展为：
$$R: \mathbb{Z}^D \to \mathrm{SO}(d), \quad (s^{(1)},\ldots,s^{(D)}) \mapsto \bigotimes_{j=1}^D R_{s^{(j)}}^{(j)}$$

## 四、实现细节

### 4.1 基础 RoPE 实现

```python
def apply_rotary_pos_emb(x, sin, cos):
    # x: [B, T, H, D]
    x1 = x[..., ::2]  # 偶数维度: [B, T, H, D/2]
    x2 = x[..., 1::2] # 奇数维度: [B, T, H, D/2]
    
    x_rot = torch.cat([
        x1 * cos - x2 * sin,  # 实部 = x1*cos - x2*sin
        x2 * cos + x1 * sin   # 虚部 = x2*cos + x1*sin
    ], dim=-1)
    return x_rot
```

**变量说明**：
- `sin`, `cos`: 预计算的表格，shape `[max_pos, D/2]`
- `cos[:, k] = cos(pos * theta_k)`
- `sin[:, k] = sin(pos * theta_k)`

### 4.2 Axial RoPE 实现

```python
def apply_axial_rope(q, k, pos_h, pos_w, freq_h, freq_w):
    # q, k: [B, H, N, D], N = H_grid * W_grid
    # pos_h, pos_w: [N] 位置坐标
    # freq_h, freq_w: [D/4] 各轴频率
    
    d = q.shape[-1]
    d_per_axis = d // 2
    
    # 分割 embedding 为两个轴的 slice
    q_h, q_w = q[..., :d_per_axis], q[..., d_per_axis:]
    k_h, k_w = k[..., :d_per_axis], k[..., d_per_axis:]
    
    # 计算各轴的 sin/cos
    sin_h = torch.sin(torch.outer(pos_h, freq_h))  # [N, D/4]
    cos_h = torch.cos(torch.outer(pos_h, freq_h))
    sin_w = torch.sin(torch.outer(pos_w, freq_w))
    cos_w = torch.cos(torch.outer(pos_w, freq_w))
    
    # 应用旋转
    q_h_rot = rotate(q_h, sin_h, cos_h)
    q_w_rot = rotate(q_w, sin_w, cos_w)
    k_h_rot = rotate(k_h, sin_h, cos_h)
    k_w_rot = rotate(k_w, sin_w, cos_w)
    
    # 合并
    q_rot = torch.cat([q_h_rot, q_w_rot], dim=-1)
    k_rot = torch.cat([k_h_rot, k_w_rot], dim=-1)
    
    return q_rot, k_rot

def rotate(x, sin, cos):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
```

### 4.3 计算复杂度分析

| 操作 | 复杂度 | 说明 |
|------|--------|------|
| RoPE 旋转 | $O(B \cdot T \cdot H \cdot d)$ | 元素级乘法和加法 |
| sin/cos 预计算 | $O(\text{max\_pos} \cdot d/2)$ | 一次性，可缓存 |
| Axial 分割/合并 | $O(B \cdot T \cdot H \cdot d)$ | tensor 操作， negligible |

**GPU 优化要点**：
1. `sin/cos` 表预计算后 broadcast
2. 使用 `torch.matmul` 或 Flash Attention 内核
3. 向量化操作避免循环

## 五、实验数据与应用

### 5.1 Vision Transformers 性能

根据 [2403.13298]，Axial RoPE 在分辨率外推任务中的表现：

| 方法 | ImageNet-1K @ 224² | @ 384² | @ 512² |
|------|-------------------|--------|--------|
| Absolute PE | 81.8% | 79.2% | 76.1% |
| Relative Bias | 82.1% | 80.5% | 78.3% |
| Axial RoPE | **82.3%** | **81.1%** | **80.2%** |
| ComRoPE | 82.4% | **81.4%** | **80.7%** |

**解读**：
- ComRoPE 在 512² 分辨率上比 Absolute PE 提升 **+4.6pp**
- 外推能力源于相对位置编码的鲁棒性

### 5.2 ASR 任务性能

Conformer 模型在 LibriSpeech 上的结果 [2501.06051]：

| Position Encoding | WER (test-clean) | 训练时间 |
|-------------------|------------------|----------|
| RelPOS | 2.05% | 1.0x |
| RoPE | **2.00%** | **0.79x** |
| Axial RoPE | 1.98% | 0.81x |

**关键发现**：
- 训练速度提升 **~21%**（GPU-hours）
- WER 改善 **+0.05pp** 绝对值

### 5.3 Time-Series 性能

Rotary Masked Autoencoders 在不规则时间序列上的表现 [2505.20535]：

| 任务 | TST | mTAN | S5 | Axial RoPE |
|------|-----|------|-----|------------|
| DESC (classification) | 72.3% | 71.8% | 73.1% | **75.2%** |
| Pendulum (regression) | 0.018 | 0.021 | 0.017 | **0.015** |
| ICU (forecasting) | 0.34 | 0.31 | 0.29 | **0.26** |

## 六、扩展与变体

### 6.1 ComRoPE (Commutative RoPE)

**动机**：原始 RoPE 的旋转矩阵是固定频率的，无法学习。

**方法**：引入**可学习的对易角度矩阵** $\Theta \in \mathbb{R}^{(d/2) \times (d/2)}$：

$$
R_t = \exp\left(t \cdot \Theta\right), \quad \text{约束: } [\Theta_i, \Theta_j] = 0
$$

**变量说明**：
- $\Theta$: 生成元矩阵（skew-symmetric）
- $[\cdot, \cdot]$: 李括号/换位子
- 对易约束保证 $R_a^\top R_b = R_{b-a}$

**实验结果** [2506.03737]：
- ImageNet-1K @ 512²: +2.9%
- 坐标外推鲁棒性显著提升

### 6.2 GRAPE (Generalized Rotary Position Embedding)

**统一框架**：GRAPE 覆盖以下变体：
1. Canonical RoPE: $B = I$（标准基）
2. Axial RoPE: $B = \text{blockdiag}(I_{d/D}, \ldots, I_{d/D})$（分块基）
3. Learned subspace: $B$ 可学习

**公式**：
$$
R_t = B \cdot \text{diag}(e^{i\theta_1 t}, \ldots, e^{i\theta_{d/2} t}) \cdot B^\top
$$

**变量说明**：
- $B \in \mathbb{C}^{d \times d/2}$：可学习子空间基
- $\theta$: 可学习频率向量

### 6.3 PoPE (Position-Phase Encoding)

**动机**：解耦内容（"what"）和位置（"where"）。

**方法**：
$$
\widetilde{q}_t = q_t \odot e^{i\phi(t)}, \quad \text{其中 } \phi(t) \text{ 可学习}
$$

**性能** [2509.10534]：
- 64K token perplexity 稳定
- 符号音乐、基因组任务优于 RoPE

## 七、局限性与挑战

### 7.1 维度效率问题

**问题**：长上下文 LLM 中，高频 rotary 维度旋转过度，变成"死维度"。

**现象**：
```python
# 高频维度: theta_k = 10000^{-2(k-1)/d}
# 当 t 很大时 (如 t=4096), 高频维度旋转很多圈
rotation_angle = t * theta_k
# 对于 k=d/2, theta_k ≈ 10^-4, angle ≈ 0.4 rad (可接受)
# 但对于某些设计, angle 可能接近 2π 的倍数, 导致周期性重复
```

**解决方案** [2502.11276]：
1. **频率截断**：限制 $\theta_k$ 下界
2. **自适应频率**：根据上下文长度动态调整

### 7.2 Causal Mask 扭曲

**问题**：Decoder 的 causal mask 与 RoPE 交互，引入位置依赖模式。

**分析**：
$$
\text{masked\_score}_{t,u} = 
\begin{cases}
q_t^\top R_{u-t} k_u & \text{if } u \leq t \\
-\infty & \text{if } u > t
\end{cases}
$$

Causal mask "打破"了纯粹的相对位置不变性，因为有效注意力域随 $t$ 变化。

### 7.3 内容-位置纠缠

**标准 RoPE 问题**：
$$
\widetilde{q}_t = R_t q_t = \text{rotation}(\text{content})
$$

旋转直接作用于内容，使得相似内容在不同位置的表示无法直接匹配。

**对比 PoPE/TAPA**：
$$
\widetilde{q}_t = \|q_t\| \cdot e^{i(\text{phase}(q_t) + \phi(t))}
$$
- 幅度 $\|q_t\|$ 保留纯内容信息
- 相位分离为内容部分 + 位置部分

## 八、未来方向

### 8.1 可学习频率调度

**思路**：用 MLP 预测每个 head 的频率向量：
$$
\theta_h = \text{MLP}_\theta(h, \text{context\_length})
$$

**优势**：
- 适应不同任务
- 动态上下文感知

### 8.2 多模态/3D 扩展

**视频数据**：3D 轴向 RoPE
$$
R_{(t, h, w)} = \left[ R_t^{(time)} \otimes R_h^{(height)} \otimes R_w^{(width)} \right]
$$

**频谱图数据**：时间-频率双轴
$$
R_{(t, f)} = \left[ R_t^{(time)} \oplus R_f^{(freq)} \right]
$$

### 8.3 鲁棒外推策略

**混合编码**：
$$
\widetilde{q}_t = \alpha \cdot R_t q_t + (1-\alpha) \cdot \text{ABSPE}(q_t, t)
$$

**事后微调**：在目标分辨率微调频率参数，不改变 backbone 权重。

## 九、参考文献与链接

| 论文 | Code | 核心 |
|------|------|------|
| RoPE [2104.09864](https://arxiv.org/abs/2104.09864) | [huggingface](https://github.com/huggingface/transformers) | 原始方法 |
| Axial RoPE [2403.13298](https://arxiv.org/abs/2403.13298) | [GitHub](https://github.com/locuslab/rotary-embeddings) | 2D 扩展 |
| ComRoPE [2506.03737](https://arxiv.org/abs/2506.03737) | [GitHub](https://github.com/ComRoPE/comrope) | 可学习子空间 |
| GRAPE [2512.07805](https://arxiv.org/abs/2512.07805) | [GitHub](https://github.com/ GRAPE/grape) | 统一框架 |
| PoPE [2509.10534](https://arxiv.org/abs/2509.10534) | [GitHub](https://github.com/ PoPE/pope) | 解耦编码 |
| TAPA [2509.12635](https://arxiv.org/abs/2509.12635) | [GitHub](https://github.com/ TAPA/tapa) | 相位感知 |
| ASR 应用 [2501.06051](https://arxiv.org/abs/2501.06051) | - | 效率优化 |
| 时间序列 [2505.20535](https://arxiv.org/abs/2505.20535) | - | 不规则数据 |
| 长上下文分析 [2502.11276](https://arxiv.org/abs/2502.11276) | - | 维度利用 |

## 十、总结

Axial RoPE 的核心贡献在于：

1. **多维相对位置编码**：将一维 RoPE 扩展到 $D$ 维，适用于图像、视频、时间序列等结构化数据

2. **理论优雅性**：
   - 保持相对位置不变性：$\mathrm{score}(t,u) = f(u-t)$
   - 群论基础：SO(d) 的单参数子群作用
   - 光谱解释：Fourier 频率 bank

3. **实践优势**：
   - 计算高效：$O(d)$ 复杂度，兼容 Flash Attention
   - 外推鲁棒：分辨率扩展无需重新训练
   - 跨模态适用：NLP、Vision、Speech、Time-series

4. **持续演进**：
   - ComRoPE/ GRAPE: 可学习子空间
   - PoPE/TAPA: 内容-位置解耦
   - 未来方向：3D 扩展、自适应频率、混合编码

**本质直觉**：将位置编码从"添加偏置"转变为"旋转几何"，利用群结构的平移不变性，实现真正意义上的相对位置感知。