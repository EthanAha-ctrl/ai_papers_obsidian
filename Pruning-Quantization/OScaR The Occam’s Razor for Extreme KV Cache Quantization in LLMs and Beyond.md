---
source_pdf: OScaR The Occam’s Razor for Extreme KV Cache Quantization in LLMs and
  Beyond.pdf
paper_sha256: 5d8c54cc72fc670d09975da78017917e8f4506f4bb79b1eefdacd4129e0690ed
processed_at: '2026-08-06T01:35:02-07:00'
target_folder: Pruning-Quantization
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# OScaR 人话版

## 1. 大背景：KV cache 是个 memory hog

LLM 推理时，每生成一个 token 都要重新算 attention。但历史 token 的 Key 和 Value 可以 cache 住复用，这就是 KV cache。问题是它随 context length 线性增长，128K context 下 Qwen3-8B 的 KV cache 能吃掉 32GB HBM，batch size 直接被卡死。

最直接的解法：把 KV cache 从 BF16 压到 INT2，8× 压缩。但 INT2 只有 4 个 level（0,1,2,3），dynamic range 极度受限，任何 outlier 都会瞬间毁掉精度。

## 2. 现有方法 per-channel 为什么是 default 选择

Key 和 Value 的分布特性完全不同：
- **Key** 有 channel-wise outliers——某些 channel 维度上几乎所有 token 都异常大。per-channel quantization（沿 channel 维度共享 scale）天生适合 Key。
- **Value** 分布均匀，per-token quantization 更合适。

KIVI 这个方法就是 Key 用 per-channel、Value 用 per-token 的 hybrid scheme。看起来挺合理，但 paper 发现它在 INT2 下精度还是掉得厉害。

## 3. TNI：per-channel 的阿喀琉斯之踵

paper 作者跑了一堆 profiling，发现一个普遍现象：每个 attention state 里都有一小撮 token 的 L2 norm 异常小。这些 token 不是随机出现的，而是稳定的 sparse subset，对应 **attention sink tokens**（StreamingLLM 那篇 work 里讲的，softmax 的 sum-to-one 约束逼着模型把注意力往无信息 token 上"排洪"）。

这带来什么问题？用一个类比：

> per-channel quantization 像给一个班统一发校服。校服尺寸 range 要覆盖整个班级的身高 span。如果班里既有 2.2m 的姚明又有 0.5m 的小朋友，校服的 size step 就得拉很大，正常身高的人穿起来都不合身。

TNI（Token Norm Imbalance）就是这个班级身高差距过大的问题。block 内最大 norm token 和最小 norm token 的 gap 直接决定量化误差下界——这是 paper 给出的理论结论：

$$
\overline{\text{MSE}}_g \gtrsim \frac{(\|\mathbf{k}_m\|_2 - \|\mathbf{k}_n\|_2)^2}{12(2^b-1)^2}
$$

norm gap 越大，误差越大。这不是工程 bug，是 per-channel 范式的结构 weakness。

multi-modal LLM 里 TNI 更严重：text token 和 vision patch token 的 norm 天生不在一个 scale，跨 modality 混在一个 block 里量化，误差暴涨 140%。

## 4. 最直觉的解法为什么不行：Scaling-Induced Outlier Artifact

第一反应：既然 norm 差异大，直接把每个 token 的 norm 拉平不就行了？对每个 token 做 L2 normalization。

但 paper 证明这会引入 **Scaling-Induced Outlier Artifact**，用人话讲：

> 正常 token 像一个高个子，但他的"高度"主要集中在某一两个维度（outlier channel）。小个子 token 各维度都小但均匀。如果你硬把他们 normalize 到同样高度，小个子的所有维度都被放大，结果小个子在那些"正常维度"上反而显得比高个子还高——人工制造了新的 outlier。

具体例子：
- Normal token $\mathbf{a} = [1, 1, 1, 100]$，主要 magnitude 在 channel 4
- Low-norm token $\mathbf{b} = [0.1, 0.1, 0.1, 0.1]$，均匀小
- Normalize 到 norm=1：$\mathbf{a}' = [0.01, 0.01, 0.01, 1]$，$\mathbf{b}' = [0.5, 0.5, 0.5, 0.5]$
- channel 1-3 上 $\mathbf{b}'$ 反而成了 outlier（0.5 vs 0.01），per-channel dynamic range 被拉大 50×

per-channel 的 scale 被这些人工 outlier 撑大，正常 token 的精度又被毁掉。

## 5. OScaR 的两步走：先打散再拉平

paper 的核心 insight：**必须先把 channel-wise outlier 能量打散到所有维度，再做 token-wise normalization**。

### Step 1: Canalized Rotation

对 Key 做 Hadamard transform：

$$
\mathbf{K}_h = \frac{\mathbf{H}(\mathbf{K})}{\sqrt{D}}
$$

Hadamard matrix 是正交的，不改变 attention 内积。它的作用是把 channel-wise sparse outlier 的能量均匀 spread 到所有 channel——每个 channel 的 dynamic range 都变窄了，分布更均匀。

Query 也要做同样的 transform，这样 $Q \cdot K$ 的内积语义不变。

### Step 2: Omni-Token Scaling

Rotation 之后，每个 token 在所有 channel 上都有相似的 magnitude 分布。这时再做 token-wise L2 normalization 就安全了：

$$
n_k = \|\mathbf{K}_h\|_2, \quad \mathbf{K}_u = \frac{\mathbf{K}_h}{n_k}
$$

$\mathbf{K}_u$ 是单位方向向量，存进 INT2 cache；norm $n_k$ 作为 metadata 单独存。

inference 时 dequant 出来乘上 $n_k$ 恢复原 magnitude，attention 计算完全正确。

### 为什么两步缺一不可

- **只 Omni-Token Scaling**（不 rotate）→ Scaling-Induced Outlier Artifact，精度反而掉（paper 的 ablation 实测掉 4.4 pp）
- **只 Canalized Rotation**（不做 scaling）→ channel outlier 解决了，但 token norm 还是差异巨大，TNI 没解决
- **两步合起来**→ rotation 先消除 channel outlier 主导地位，scaling 再安全统一 token norm

这就是 paper 标题 "Occam's Razor" 的含义：不需要 calibration、不需要 mixed-precision、不需要离线 training、不需要单独 outlier path，两个简单 closed-form 操作搞定。

## 6. Value 侧：更简单

Value 没有 channel-wise outlier，per-token quantization 就够了。paper 用一个 trick：把 Hadamard transform **离线融进权重** $W_V$ 和 $W_O$：

$$
W_V \leftarrow W_V \mathbf{H}, \quad W_O \leftarrow \mathbf{H} W_O
$$

inference 时完全无额外开销，但 Value 的量化友好度提升了。

## 7. 系统实现：fused kernel + Tensor Core

光有算法不够，还要跑得快。paper 的 CUDA 实现几个关键点：

1. **Fused Hadamard-Norm kernel**：FHT 和 token scaling 融合进一个 kernel launch，避免中间结果写回 HBM
2. **HadaCore Kronecker 分解**：$H_{128} = H_8 \otimes H_{16}$，大块 $H_{16}$ 跑在 Tensor Core 的 WMMA tiles 上，小块 $H_8$ 用 scalar butterfly
3. **rsqrt 硬件指令**：$1/\sqrt{\sum x_i^2}$ 一步算出来，比 sum + sqrt + division 快很多
4. **Residual 机制**：新 token 先进 FP16 buffer，攒够 128 个再 block-wise quantize（类似 KIVI），避免单 token 量化找不到 scale

## 8. 实验结果有多强

**LongBench-E（Llama-3.1-8B）**：
- 16-bit baseline: 41.70
- KIVI: 39.84
- **OScaR (INT2): 41.75**（甚至略超 16-bit）

**Needle-in-a-Haystack（Qwen3-8B, 42K context）**：
- 16-bit: 96.0%
- KIVI: 88.8%
- **OScaR: 96.5%**（超过 16-bit）

**Multi-modal (OCRBench, Qwen3-VL-8B)**：
- 16-bit: 858
- **OScaR: 856**（几乎无损）

**Efficiency (128K context, Qwen3-8B)**：
- BF16 baseline: 92.9 ms/tok
- **OScaR: 30.9 ms/tok**（3.0× speedup）
- Memory 5.3× 压缩，throughput 4.1× 提升

OScaR 的 latency 随 context length 几乎不涨（16K 时 24.1，128K 时 30.9），因为 KV cache 压到 INT2 后完全 memory-bound 变 compute-bound。

## 9. 一句话总结 Intuition

> **per-channel quantization 的精度由 block 内 token 的 norm gap 决定。attention sink + multi-modal heterogeneity 让这个 gap 极大化。OScaR 用 Hadamard rotation 先把 channel outlier 平摊到所有维度，再用 token-wise normalization 把所有 token norm 拉平。两步组合才能避开单独使用任一步引入的 artifact。**

这个设计 philosophy 就是 Occam's Razor：能简单就别复杂，两个 closed-form 操作解决所有问题，比 TurboQuant 那种 mixed-precision + Lloyd-Max + QJL + residual correction 的复杂 pipeline 在精度和效率上都占优。

参考链接：
- Paper: https://github.com/ZunhaiSu/OScaR-KV-Quant
- StreamingLLM (attention sink 原始 paper): https://arxiv.org/abs/2309.17453
- KIVI (per-channel baseline): https://arxiv.org/abs/2402.02750
- QuaRot (只 rotate 不 scale): https://arxiv.org/abs/2410.16244
- TurboQuant (复杂 pipeline 对比): https://arxiv.org/abs/2504.19874

---

# OScaR: Occam's Razor for Extreme KV Cache Quantization 深度解析

## 1. 问题背景：为什么 KV cache 是 deployment 的核心瓶颈

在 LLM 推理时，autoregressive decoding 每生成一个 token 都要重新计算 attention，但 Key 和 Value tensor 可以 cache 复用，这就是 KV cache。Memory footprint 随 sequence length 线性增长：

$$
\text{Memory}(S) = \mathcal{L} \cdot 2 \cdot S \cdot D \cdot \text{sizeof}(\text{dtype})
$$

其中 $\mathcal{L}$ 是 transformer 层数，$S$ 是 sequence length，$D$ 是 hidden dimension。对于 128K context 的 Qwen3-8B，BF16 下 KV cache 占用 ~32GB HBM，这直接把 batch size 卡死，throughput 上不去。

**Extreme low-bit quantization（INT2）的必要性**：从 BF16 到 INT2 是 8× 压缩，能把 KV cache 从 memory-bound 的瓶颈中解放出来。但 INT2 只有 4 个 levels（0,1,2,3），dynamic range 极度受限，任何 outlier 都会瞬间毁掉精度。

## 2. Per-Channel Quantization 范式的根本问题：TNI

### 2.1 Per-channel vs Per-token 的选择逻辑

Key 和 Value tensor 有非常不同的分布特性（见 paper Figure 2）：
- **Key states**：存在显著的 **channel-wise outliers**——某些 channel 维度上几乎所有 tokens 都有异常大的 magnitude。Per-channel quantization（共享 scale/zero-point 沿 channel 维度）天生适合。
- **Value states**：分布相对均匀，per-token quantization 是常规选择。

KIVI 等方法用 hybrid scheme：Key per-channel，Value per-token。

### 2.2 Block-wise Per-Channel Quantization 的数学形式

给定 Key cache $K \in \mathbb{R}^{S \times d}$，将每个 channel 沿 sequence 维度切成 size 为 $G$ 的 blocks。对第 $j$ 个 channel 在 block $g$ 内：

$$
\Delta_{j,g} = \frac{\max_{i \in \mathcal{G}} K_{i,j} - \min_{i \in \mathcal{G}} K_{i,j}}{2^b - 1}, \quad z_{j,g} = \left\lfloor -\frac{\min_{i \in \mathcal{G}} K_{i,j}}{\Delta_{j,g}} \right\rceil
$$

变量解释：
- $K_{i,j}$：block $\mathcal{G}$ 中第 $i$ 个 token 在第 $j$ 个 channel 的值
- $\Delta_{j,g}$：quantization step size（动态范围除以 level 数 $2^b - 1$）
- $z_{j,g}$：zero-point，让 0 浮点值映射到整数范围中点附近
- $b$：bit-width（INT2 时 $b=2$，$2^b - 1 = 3$）

量化与重建：

$$
Q(K_{i,j}) = \text{clamp}\left(\left\lfloor \frac{K_{i,j}}{\Delta_{j,g}} \right\rceil + z_{j,g}, 0, 2^b - 1\right), \quad \hat{K}_{i,j} = \Delta_{j,g} \cdot (Q(K_{i,j}) - z_{j,g})
$$

### 2.3 核心洞察：Token Norm Imbalance (TNI)

Paper 通过对 Llama-2-7B、Llama-3.1-8B、Qwen3-8B 的 token-wise L2 norm 分析，发现：

**每个 attention state（Q、K、V）中都存在一个 sparse 但 consistent 的 token 子集，其 L2 norm 异常小**。这些 low-norm outlier tokens 对应 **Attention Sink tokens**（参见 StreamingLLM, Xiao et al. 2023 [https://arxiv.org/abs/2309.17453]）。

token norm 计算公式：

$$
\mathcal{N}_t^{(M)} = \left\{ \|\mathbf{t}_{t,h}^{(M)}\|_2 \mid h = 1, \ldots, H \right\}, \quad \|\mathbf{t}_{t,h}^{(M)}\|_2 = \sqrt{\sum_{j=1}^{d_h} (s_{t,h,j}^{(M)})^2}
$$

变量解释：
- $M \in \{\text{Query, Key, Value}\}$：attention 状态类型
- $t$：token position
- $h$：attention head index，$H$ 是 head 数量
- $d_h$：head dimension（通常 128）
- $s_{t,h,j}^{(M)}$：token $t$ 在 head $h$ 第 $j$ 个分量上的值

**TNI 的本质**：Per-channel quantization 假设同一 channel 内的 tokens magnitude 相近。但当 block 内既有 norm=100 的 normal token 又有 norm=0.2 的 attention sink token 时，shared scale 必须覆盖整个 range，导致 step size 被拉大，quantization error 系统性放大。

### 2.4 TNI 在 Multi-modal LLMs 中的三种 pattern

在 multi-modal LLMs（Qwen3-VL-8B 等）中 TNI 表现得更复杂：

1. **Broader token norm variation**（Figure 19）：相比 text-only LLMs，norm 分布更分散
2. **Inter-modality norm disparities**（Figure 20）：每个 modality 内部 norm 平滑，但跨 modality 差异巨大（text tokens vs vision patch tokens）
3. **Exceptionally large-norm outlier tokens**（Figure 21）：与 low-norm attention sink 相反，还有 high-norm outliers

## 3. 理论分析：TNI 为什么是 per-channel 量化的根本弱点

### 3.1 量化误差的下界推导

**Step 1**: 均匀量化器的 MSE。对于均匀分布量化误差 $\epsilon \in [-\Delta_{j,g}/2, \Delta_{j,g}/2]$：

$$
\text{MSE}_{j,g} = \mathbb{E}[\epsilon^2] = \frac{1}{\Delta_{j,g}} \int_{-\Delta_{j,g}/2}^{\Delta_{j,g}/2} \epsilon^2 d\epsilon = \frac{\Delta_{j,g}^2}{12}
$$

这是经典的 Bennett 公式。

**Step 2**: Pairwise lower bound。$\Delta_{j,g}$ 由 block 内 sample range 决定，$\mathcal{R}_{j,g} \geq |K_{u,j} - K_{v,j}|$ 对任意 $u, v \in g$ 成立。代入：

$$
\text{MSE}_{j,g} \geq \frac{(K_{u,j} - K_{v,j})^2}{12(2^b - 1)^2}
$$

**Step 3**: 整 block 的下界。设 $m = \arg\max_{t \in g} \|\mathbf{k}_t\|_2$，$n = \arg\min_{t \in g} \|\mathbf{k}_t\|_2$。对所有 channel 求和：

$$
\overline{\text{MSE}}_g \gtrsim \frac{1}{12(2^b-1)^2} \sum_{j=1}^d (K_{m,j} - K_{n,j})^2 = \frac{\|\mathbf{k}_m - \mathbf{k}_n\|_2^2}{12(2^b-1)^2} \geq \frac{(\|\mathbf{k}_m\|_2 - \|\mathbf{k}_n\|_2)^2}{12(2^b-1)^2}
$$

最后一步用 reverse triangle inequality。

**直觉**：这个不等式说明——**block 内最大 norm 与最小 norm token 的差距，直接决定了 per-channel quantization 的误差下界**。TNI 越严重，量化误差越大。这是 per-channel 范式的结构性弱点，不是工程实现问题。

### 3.2 实验定量验证（Table 2）

在 LLaVA-v1.5-7B 上：
- **Outlier tokens 的影响**：INT2 下，包含 outlier tokens 的 group MSE=5.92（×100），去除后 MSE=3.52，**增加 35%**
- **Mixed-modality 的影响**：INT2 下视觉部分 MSE 从 5.87（mixed）涨到 6.17（单模态），mixed 比 single 增加 **140%**
- Per-token Value quantization 几乎不受 TNI 影响（误差 0.40 vs 0.52），因为 norm 变化被限制在单个 token 内部

## 4. OScaR 框架：Canalized Rotation + Omni-Token Scaling

### 4.1 为什么不能直接 token-wise scaling？Scaling-Induced Outlier Artifact

直觉上最简单的 TNI 解决方案：直接对每个 token 做 L2 normalization，把所有 token 的 norm 拉到同一水平。但 paper 证明这会引入 **Scaling-Induced Outlier Artifact**。

数学分析（Appendix I）：
- 设 normal token $\mathbf{a} = [1, 1, 1, 100]$，主要 magnitude 集中在第 4 个 channel（outlier channel），$\|\mathbf{a}\|_2 \approx 100.015$
- 设 low-norm token $\mathbf{b} = [0.1, 0.1, 0.1, 0.1]$，所有分量均匀小，$\|\mathbf{b}\|_2 = 0.2$
- 都 normalize 到 $N=1$：$\alpha = N/\|\mathbf{b}\|_2 = 5$，$\beta = N/\|\mathbf{a}\|_2 \approx 0.01$
- 结果：$\mathbf{a}' = [0.01, 0.01, 0.01, 1.00]$，$\mathbf{b}' = [0.5, 0.5, 0.5, 0.5]$

**问题**：在 channel 1-3 中，原本 $\mathbf{a}$ 的值是 0.01，$\mathbf{b}$ 缩放后变成 0.5——$\mathbf{b}'$ 在这些 channel 中**变成了人工 outlier**！per-channel 的 dynamic range 从 0.01 被拉到 0.5，step size 放大 50×，normal token 的精度被毁。

### 4.2 OScaR 的两步流程

**核心思想（Occam's Razor）**：不引入复杂 pipeline（如 TurboQuant 的 mixed-precision + Lloyd-Max + QJL + residual correction），而是用两个互相依赖的简单操作。

#### Step 1: Canalized Rotation

对 Key 应用 Hadamard transform：

$$
\mathbf{K}_h = \frac{\mathbf{H}(\mathbf{K})}{\sqrt{D}}, \quad D = d_h
$$

其中 $\mathbf{H}$ 是 $D \times D$ 的 Hadamard matrix，$\sqrt{D}$ 是归一化系数保证正交性。

**作用**：Hadamard transform 把 channel-wise outlier 的能量**重新分布到所有 channel**，smooth out per-channel 分布。这样后续 token-wise scaling 不会在某些 channel 上制造人工 outliers。

Query 也要做同样的 transform，保证 attention 内积不变（因为 Hadamard 是正交的）：

$$
\mathbf{Q}_h = \frac{\mathbf{H}(\mathbf{Q})}{\sqrt{D}}, \quad \text{logits} = (\mathbf{Q}_h \cdot \text{dequant}(\mathbf{K}_u)) \cdot n_k
$$

#### Step 2: Omni-Token Scaling

计算每个 token 的 L2 norm 并归一化：

$$
n_k = \|\mathbf{K}_h\|_2, \quad \mathbf{K}_u = \frac{\mathbf{K}_h}{n_k}
$$

$\mathbf{K}_u$ 是单位方向向量，存进 INT2 cache；$n_k$ 作为 metadata 单独存。

**为什么这两步缺一不可**：
- 只做 Omni-Token Scaling（不先 rotate）→ Scaling-Induced Outlier Artifact
- 只做 Canalized Rotation（不做 scaling）→ token norm 仍然差异巨大，TNI 没解决
- 两步合起来：rotation 先消除 channel-wise outlier 的主导地位，scaling 再安全地统一 token norm

### 4.3 Value 侧的处理

Value 用 **offline Hadamard transform** + per-token quantization：

$$
W_V \leftarrow W_V \mathbf{H}, \quad W_O \leftarrow \mathbf{H} W_O
$$

这相当于把 Hadamard transform 融入模型权重，**inference 时无额外开销**，同时改善 Value 的量化友好度。Per-token 量化不需要 norm scaling，因为 Value 本身没有 channel-wise outlier。

### 4.4 完整算法流程（Algorithm 1）

**Preprocess（offline）**：
1. 对 $W_V$ 和 $W_O$ 做 Hadamard transform

**Inference（per layer）**：
1. Project：$X_Q = X W_Q$，$X_K = X W_K$，$X_V = X W_V$
2. Canalized Rotation：$X_Q \leftarrow \text{FHT}(X_Q)$，$X_K \leftarrow \text{FHT}(X_K)$
3. Omni-Token Scaling：$s_K \leftarrow \|X_K\|_2$，$X_K \leftarrow X_K / s_K$
4. 重建 Key：$K_{all} = \text{Concat}([\text{Dequant}(Q(X_K^{hist})) \cdot s_{K_g}^{hist}, X_{K_r}^{hist} \cdot s_{K_r}^{hist}, X_K \cdot s_K])$
5. Attention + output projection
6. BufferQuant：新 token 进 residual buffer，buffer 满 128 时 flush 成 quantized block

**Residual 机制**：类似 KIVI，新 token 先以 FP16 存在 residual cache，攒够 R=128 个再 block-wise quantize，避免单 token 量化时找不到合适的 scale。

## 5. 系统设计与 CUDA 实现

### 5.1 三个 CUDA kernels

OScaR 实现基于 HadaCore [https://arxiv.org/abs/2412.08832] 和 BitDecoding 两个框架：

1. **Online FHT + Scaling kernel**：fused FHT 和 token scaling（Key 侧），Query 只做 FHT
2. **Quantization kernel**：对 Key 做 per-channel quantization，对 Value 做 per-token quantization
3. **Dequantization + De-Scaling + Attention kernel**：三步融合，避免中间结果写回 HBM

### 5.2 Tensor Core 加速 FHT

对于 $d_h = 128$，naive Fast Hadamard Transform 用 scalar butterfly 在 shared memory 中跑，但有大量 scalar instruction 压力。OScaR 采用 HadaCore 的 **Kronecker decomposition**：

$$
\mathbf{H}_{128} = \mathbf{H}_8 \otimes \mathbf{H}_{16}
$$

- $\mathbf{H}_{16}$ 用 **WMMA (Warp Matrix Multiply-Accumulate) tiles** 跑在 Tensor Cores 上
- $\mathbf{H}_8$ 用 scalar butterfly（小规模，开销小）

这种分解把大部分算力转移到 Tensor Core，scalar instruction 压力显著降低。

### 5.3 rsqrt 硬件指令

Omni-Token Scaling 的核心是 $1/\sqrt{\sum x_i^2}$。直接算需要 sum of squares + sqrt + division 三步。GPU 提供 hardware-accelerated `rsqrt` 指令，一步算出 $1/\sqrt{x}$，被 OScaR 采用。Ablation study（Table 10）显示 rsqrt 与 $\ell_2$ norm 精度等价但 latency 更低。

### 5.4 Cache 组织

**Packed cache**（量化后的）：
- $K_u$ 2-bit payload（8 个值 pack 进一个 uint16）
- $K$ scale 和 zero-point（per-channel group）
- $V$ 2-bit payload
- $V$ scale 和 zero-point（per-token group）
- $K$ token-wise norms $n_k$（FP16 metadata）

**Residual cache**（未量化的 buffer）：
- $K_u$ residual（FP16）
- $V$ residual（FP16）
- $K$ norm residual（FP16）

Residual 长度 $R = 128$ 时 flush，正好对应 4 个 quantization group（$G = 32$），硬件对齐。

## 6. 实验结果

### 6.1 LongBench-E（Table 1）

Llama-3.1-8B 上：
| Method | Avg. |
|---|---|
| 16-bit | 41.70 |
| QuaRot | 37.94 |
| RotateKV | 37.98 |
| KIVI | 39.84 |
| OTT | 40.74 |
| TurboQuant+ (2.5-bit) | 40.03 |
| **OScaR (INT2)** | **41.75** |

Qwen3-8B 上：
- 16-bit: 49.56
- KIVI: 47.95
- OTT: 48.21
- **OScaR: 48.74**（相对 16-bit 仅掉 1.7%）

**关键观察**：OScaR 在 INT2 下达到 41.75，**略高于 16-bit baseline 的 41.70**——这是 near-lossless 量化的标杆。

### 6.2 Needle-in-a-Haystack（Figure 29）

42K context，15 个 depth positions：
- 16-bit baseline: 96.0%
- KIVI: 88.8%
- OTT: 90.1%
- TurboQuant+ (2.5-bit): 92.7%
- **OScaR: 96.5%**（**甚至超过 16-bit baseline**）

这个结果说明 OScaR 在长 context retrieval 场景下，量化不仅不掉点反而有微弱增益（可能是 INT2 量化引入了某种 implicit regularization，过滤掉了不重要的 high-frequency 信息）。

### 6.3 Multi-modal 和 Omni-modal 评测

**OCRBench（Table 6）**：
- LLaVA-v1.6-vicuna-7B：OScaR 519 vs 16-bit 536
- Qwen3-VL-8B：OScaR 856 vs 16-bit 858（**几乎无损**）
- Qwen3-VL-4B：OScaR 838 vs OTT 831（**比第二名高 2.5 pp**）

**DocVQA（Table 7）**：
- Qwen3-VL-8B：OScaR 95.01 **超过 16-bit 94.93**
- Qwen3-VL-4B：OScaR 93.85 vs 16-bit 94.23（仅低 0.38）

**MMAU-Pro omni-modal（Table 8）**：
- Open-ended QA：OScaR 67.4 vs 16-bit 66.2（**超过 baseline**）
- Good Rate：OScaR 29.8 vs 16-bit 27.8
- Audio Instruction Following：OScaR 88.5 vs 16-bit 87.4

这些结果显示 OScaR 在 X-LLMs（text/multi-modal/omni-modal）上都有强 generalization。

### 6.4 效率（Figure 6, Table 11）

H20 GPU + Qwen3-8B + BF16 FlashDecoding-v2 baseline：

| Context | Baseline (ms/tok) | OScaR (ms/tok) | TurboQuant+ (ms/tok) |
|---|---|---|---|
| 1K | 19.5 | 25.1 | 7.8 |
| 16K | 28.3 | 24.1 | 15.7 |
| 64K | 56.3 | 25.3 | 40.2 |
| 128K | 92.9 | 30.9 | 72.9 |

**关键观察**：
- 128K context 下 OScaR **3.0× speedup**（92.9 vs 30.9 ms/tok）
- OScaR latency 几乎随 context length 平稳（24.1 → 30.9 ms/tok，16K → 128K）
- TurboQuant+ 在短 context 下很快（7.8 ms/tok @ 1K），但随 context 增长急剧恶化（72.9 @ 128K）
- Batch size 48 下：memory 减少 **5.3×**，throughput 提升 **4.1×**

## 7. 计算复杂度分析（Table 3, 4）

每 token prefill 和每 step decode 的 operation counts（$d = 4096$, $h = 128$, $L = 10000$）：

| Method | Prefill (M units) | Decode/step (M units) |
|---|---|---|
| KIVI | 204.8 | 81.9 |
| QuaRot | 778.2 | 82.0 |
| OScaR | 901.1 | 123.0 |
| TurboQuant (orig) | 32,051 | 249.0 |
| TurboQuant+ | 21,187 | 247.9 |

**为什么 OScaR 的 decode cost 只比 KIVI 高 1.5×**：
- Canalized Rotation：$2d/\log_2 h \approx 2 \times 4096 / 7 \approx 1170$ ops（butterfly 结构）
- Omni-Token Scaling：$3d \approx 12288$ ops（sum of squares + rsqrt + scale）
- 主要开销来自 dequant 时 $3Ld$（dequant + de-scale + multiply by norm），但这些都是 fused kernel 高效执行

**TurboQuant+ 为什么这么贵**：
- Dense Haar QR rotation：$4dh = 4 \times 4096 \times 128 = 2.1$M ops（不是 butterfly，是 dense matrix multiply）
- 二分搜索 Lloyd-Max quantization：$2.25d$ comparisons
- $Ld$ table lookups（按 1:5 加权变成 $5Ld$，对 $L=10000$ 是 $2 \times 10^8$ units）

**Pareto Front（Figure 9）**：OScaR 在 accuracy（48.74）和 cost（123.0M）上都占优——KIVI 最便宜但精度差（47.95），TurboQuant+ 贵 3× 但精度只比 KIVI 高一点。

## 8. 消融研究（Table 9）

在 Qwen2.5-Omni-7B WorldSense benchmark：

| 配置 | Accuracy |
|---|---|
| INT2 KCVT (baseline) | 39.62 |
| + Omni-Token Scaling only | 35.22 (**掉点**，验证 Scaling Artifact) |
| + Canalized Rotation only | 41.51 |
| + Canalized Rotation + Omni-Token Scaling | **42.77** |

**直接证明**：单独用 Omni-Token Scaling 反而**掉 4.4 pp**——这就是 Scaling-Induced Outlier Artifact 的危害。必须先 Canalized Rotation。

## 9. 与相关工作的关系

### 9.1 vs QuaRot [https://arxiv.org/abs/2412.08832 (HadaCore)]
QuaRot 也用 Hadamard rotation，但**只 rotate，不做 token scaling**——所以它解决了 channel-wise outlier 但没解决 TNI。在 LongBench-E 上 QuaRot 只有 37.94（Llama-3.1-8B），OScaR 41.75。

### 9.2 vs RotateKV [https://arxiv.org/abs/2501.16383]
RotateKV 用 outlier-aware adaptive rotation，需要 pre-calibration 找 outliers。OScaR 是 training-free 且 calibration-free。

### 9.3 vs KIVI [https://arxiv.org/abs/2402.02750]
KIVI 是 per-channel baseline，没有 rotation 也没有 scaling。在 LongBench-E 上 Llama-3.1-8B 掉 1.86 pp（39.84 vs 41.70），OScaR 仅掉 0.05 pp。

### 9.4 vs TurboQuant [https://arxiv.org/abs/2504.19874]
TurboQuant 用 mixed-precision（2.5-bit）+ dense Haar QR rotation + QJL projection + residual error correction，pipeline 极复杂。QJL 被发现会**降低性能**（Table 5），TurboQuant+ 把 QJL 去掉。OScaR 的简单设计在 INT2 下达到 TurboQuant+ 2.5-bit 的精度。

### 9.5 vs OTT [https://aclanthology.org/2025.acl-long.547/]
OTT 把 outlier tokens 单独高精度保存，引入 mixed-precision 硬件碎片化。OScaR 用 Omni-Token Scaling 把 outlier token 的 norm 统一到正常水平，不需要特殊路径。

## 10. 局限性与未来方向（Appendix A）

1. **RoPE 阻碍 offline fusion**：因为 RoPE 是 position-dependent 的，Hadamard transform 无法离线融合进 $W_K$，必须 online 计算。QuaRot 也有同样问题。这是相比纯 per-channel（KIVI）的额外开销来源。

2. **更多模型架构未验证**：当前实验集中在 LLM backbone 模型，未来可扩展到：
   - StreamVGGT [https://arxiv.org/abs/2507.11539]
   - Visual autoregressive models
   - Diffusion LLM with KV cache [https://arxiv.org/abs/2505.22618]

## 11. 关键 Intuition 总结

**整个 paper 的核心 insight 可以浓缩成一句话**：per-channel quantization 的精度由 block 内 token 的 **norm gap** 决定，attention sink + multi-modal heterogeneity 让这个 gap 极大化，OScaR 用 Hadamard rotation 把 channel outlier 平摊到所有维度，再用 token-wise normalization 把所有 token norm 拉平，两步组合才能避开单独使用任一步骤引入的 artifact。

**为什么 Hadamard transform 在这里 work**：
- Hadamard matrix $H_D$ 满足 $H_D^T H_D = D \cdot I$，是 orthogonal transform，不改变 attention 内积
- 把 channel-wise sparse outlier energy 均匀 spread 到所有 channel，每个 channel 的 dynamic range 收窄
- FHT 的 $O(d \log d)$ 复杂度远低于 dense rotation 的 $O(d^2)$

**为什么 Omni-Token Scaling 在 rotation 之后 work**：
- Rotation 后每个 channel 都有相似 magnitude 分布
- 把 low-norm token 放大时，它原本均匀小的 entries 不会在某 channel 上变成 outlier（因为 normal token 在该 channel 也不小）
- Norm metadata $n_k$ 存为 FP16，inference 时与 dequantized $K_u$ 相乘恢复原 magnitude，不影响 attention 计算正确性

**为什么这套设计是 Occam's Razor**：
- 不需要 calibration（如 RotateKV）
- 不需要 mixed-precision（如 TurboQuant）
- 不需要离线 training（如 AWQ）
- 不需要单独 outlier token path（如 OTT）
- 两个操作都是 closed-form，训练 free，硬件 friendly

参考资源：
- Paper GitHub: https://github.com/ZunhaiSu/OScaR-KV-Quant
- StreamingLLM (Attention Sink): https://arxiv.org/abs/2309.17453
- KIVI: https://arxiv.org/abs/2402.02750
- QuaRot: https://arxiv.org/abs/2410.16244
- HadaCore: https://arxiv.org/abs/2412.08832
- TurboQuant: https://arxiv.org/abs/2504.19874
- RotateKV: https://arxiv.org/abs/2501.16383
- OTT: https://aclanthology.org/2025.acl-long.547/
- LongBench: https://arxiv.org/abs/2408.03326
- MMAU-Pro: https://arxiv.org/abs/2410.18930
- DocVQA: https://arxiv.org/abs/2007.00398
- OCRBench: https://arxiv.org/abs/2305.07895
