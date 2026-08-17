---
source_pdf: Spiking Transformer with Spatial-Temporal Attention.pdf
paper_sha256: 31df7deca14a51495ae00b336a35f143ec9be02cb7ff339787896ad7dfbd6704
processed_at: '2026-08-12T10:00:00-07:00'
target_folder: Pruning-Quantization
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好的，让我们用最直白的方式，把这篇论文的 "Aha moment" 拆解出来。

这篇 paper 的核心故事其实非常优美，它挖掘出了 Spiking Neural Networks (SNNs) 在处理 Transformer attention 时一个被所有人忽略的特性，并且利用 binary computation 的 algebraic freedom 算了一个非常漂亮的 "free lunch"。

---

## 1. The Problem: 现有 SNN Transformers 都在 "装瞎"

目前所有的 SNN Transformers (比如 Spikformer, Spike-driven Transformer) 基本上都遵循一个套路：把一张图片在 $T$ 个 timesteps 内编码成 spikes，输入 shape 是 $\mathbf{X} \in \mathbb{R}^{T \times N \times D}$。
- $T$: Timesteps (时间步数)
- $N$: Number of tokens / patches (空间上的 token 数量)
- $D$: Feature dimension (特征维度)

然后在计算 Self-attention 的时候，它们的做法是：**把 $T$ 当成一个独立的 batch dimension**。也就是说，在 $t=1$ 时算 $t=1$ 的 attention，在 $t=2$ 时算 $t=2$ 的 attention。

这在逻辑上存在巨大的缺陷。SNN 的核心在于 Leaky Integrate-and-Fire (LIF) 神经元的动态演化，我们看 LIF 的公式：

$$\mathbf{u}[t+1]^l = \tau \mathbf{u}[t]^l + \mathbf{W}^l f(\mathbf{u}[t]^{l-1})$$

- $\mathbf{u}[t]^l$: 第 $l$ 层在 timestep $t$ 的 membrane potential (膜电位)
- $\tau \in (0, 1]$: Leaky factor (衰减系数)
- $\mathbf{W}^l$: 权重
- $f(\cdot)$: LIF 发放函数，超过阈值 $V_{th}$ 发放 spike (设为 1)，否则为 0

因为 $\tau$ 的衰减作用和 spike 发放后的 reset 机制，同一个 token 在 $t=1$ 和 $t=4$ 时的 spike pattern 是完全不同的，信息是随时间演化的。如果只做 spatial-only attention，等于完全丢弃了 SNN 最宝贵的 temporal dynamics。模型只是在看 $T$ 张孤立的、毫无关联的二值图片。

## 2. The Naive Solution & The "Dead Neuron" Trap

最直观的解决办法是像 Video Transformer (ViViT) 那样，把时间和空间结合起来，直接在 $(T \times N)$ 这个维度上做 attention。

但是 SNN 的 binary 特性在这里挖了一个大坑。在 SNN 里，Q, K, V 都是 0 或 1 的矩阵。如果我们要计算跨时间步的 attention，比如 $\mathbf{Q}_t$ 和 $\mathbf{K}_{t'}$ 的乘积：

$$(\mathbf{Q}_t \mathbf{K}_{t'}^\top)_{i,j} = \sum_{d=1}^{D} q_{t,i,d} \cdot k_{t',j,d}$$

- $i, j \in \{1, ..., N\}$: Token positions (空间 token 的索引)
- $d \in \{1, ..., D\}$: Feature dimension (特征维度索引)
- $q_{t,i,d}, k_{t',j,d} \in \{0, 1\}$: 具体某个位置的 spike 值

如果 $t$ 和 $t'$ 离得非常远，spike pattern 之间的相似度就会极低。0 乘 0 等于 0，0 乘 1 也等于 0。这个点乘结果会变成一大堆的 0。因为后面紧跟着一个 LIF neuron (需要累加到阈值 $V_{th}$ 才能发放 spike)，这种全是 0 的输入根本无法 trigger 神经元。这就是所谓的 **Dead Neuron Problem**。远距离的 temporal correlation 在 SNN 里会直接导致信息流断裂。

## 3. The Genius Move: Block-wise + Matrix Reordering

STAtten 的解决方案极其巧妙，它分为两步。

### Step A: Block-wise Temporal Processing
既然跨越所有时间步会导致 dead neurons，那就只看相邻的时间步。把 $T$ 个 timesteps 切分成大小为 $B$ 的 blocks。

$$\text{STAtten}(\mathbf{X}[b]) = \text{LIF}(\mathbf{Q}[b] \mathbf{K}^\top[b] \mathbf{V}[b] \cdot \alpha)$$

- $[b] = [iB : (i+1)B, :, d]$, $i \in \{0, 1, ..., T/B - 1\}$
- $B$: Block size (块大小)
- $\alpha$: Scaling factor (替代 Softmax 的缩放因子)

比如 $T=4, B=2$，就把 $[t_0, t_1]$ 作为一个 block 处理，$[t_2, t_3]$ 作为一个 block 处理。相邻的 spikes 相关性高，点乘不会全变成 0，神经元活下来了。

### Step B: The Algebraic Trick (This is where the magic happens)
到现在为止，思路依然很朴素。但这篇 paper 最厉害的地方在于，它发现 **因为去掉了 Softmax，矩阵乘法的顺序可以随意调换！**

在传统的 ANN Transformer 里，由于 Softmax 的存在：
$$\text{Attn} = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{D}}\right)\mathbf{V}$$
Softmax 是按行进行的 non-linear operation，迫使你必须先计算 $\mathbf{Q}\mathbf{K}^\top$。如果要结合时间和空间，$\mathbf{Q}\mathbf{K}^\top$ 的维度是 $(T \cdot N) \times (T \cdot N)$，这个中间结果的 memory 占用是灾难性的 ($O(T^2 N^2 D)$)。

而在 SNN 里，全是 binary 运算，没有 Softmax！根据矩阵乘法的结合律：
$$\mathbf{Q} (\mathbf{K}^\top \mathbf{V}) = (\mathbf{Q}\mathbf{K}^\top) \mathbf{V}$$

STAtten 先计算 $\mathbf{K}^\top \mathbf{V}$。我们看维度变化：

$$\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{\frac{T}{B}N \times D} \xrightarrow{\text{reshape}} \mathbb{R}^{(\frac{T}{B}N) \times D}$$

$$\mathbf{K}^\top \mathbf{V} \in \mathbb{R}^{D \times D} \xrightarrow{\text{attention}} \mathbf{Q}(\mathbf{K}^\top \mathbf{V}) \in \mathbb{R}^{\frac{T}{B}N \times D}$$

- $\mathbf{K}^\top$: Shape 为 $D \times (\frac{T}{B}N)$
- $\mathbf{V}$: Shape 为 $(\frac{T}{B}N) \times D$
- $\mathbf{K}^\top \mathbf{V}$: Shape 为 $D \times D$

由于 $\mathbf{K}^\top \mathbf{V}$ 的结果变成了 $D \times D$（这个尺寸非常小，因为 $D$ 通常是 384 或 512），中间 memory 爆炸的问题瞬间消失了。然后再乘以 $\mathbf{Q}$ 恢复到原始维度。

这个 trick 把 spatial-temporal attention 的复杂度从 $O(T^2 N^2 D)$ 直接降到了 $O(T N D^2)$，和原来纯 spatial-only 的 SNN Transformer 一模一样！

## 4. Entropy 视角的 Intuition Building

Paper 里用 Information Theory 的角度论证了为什么 STAtten 有用。他们计算了 Attention map 的 Shannon Entropy：

$$H(\text{Attn}) = -\sum_{t=1}^{T}\sum_{n=1}^{N}\sum_{d=1}^{D} \hat{\mathbf{p}}[t,n,d] \log \hat{\mathbf{p}}[t,n,d]$$

- $\hat{\mathbf{p}}[t,n,d] = \text{Softmax}(\text{Attn}[t,n,d])$

Spatial-only attention 的 Entropy 是 5.81，而 Spatial-temporal (STAtten) 的 Entropy 降到了 4.85。

在 SNN 中，Lower entropy 意味着 spike patterns 更加 focused，更加结构化。把时间和空间混在一起做 attention，模型不再散漫地看每一个 timestep，而是能够锁定那些跨时间步依然存在的 robust features，表征变得更集中、更锐利。

## 5. Experimental Results: 一致且显著的 Free Lunch

因为这是一个 plug-and-play 的 module，作者把它插进了所有主流的 SNN Transformer 架构里。

### ImageNet Classification Performance
| Method | Param (M) | Timestep | Accuracy (%) |
|--------|----------:|----------:|----------:|
| Spike-driven Transformer (SDT) | 66.34 | 4 | 76.32 / 77.07* |
| SpikingReformer | 66.38 | 4 | 78.77 / 79.40* |
| SDT-V2 (Baseline) | 55.4 | 4 | 79.49 / 79.98* |
| **STAtten + SDT-V2** | 55.4 | 4 | **79.85 / 80.67*** |

*\*表示 288×288 的高分辨率测试*

STAtten 在不增加任何参数（因为只改变计算顺序，不加网络层）的情况下，在 ImageNet 上把 SDT-V2 从 79.98% 推到了 80.67%。这甚至打败了参数量更大的 SpikingReformer。

### Object Detection (PASCAL VOC)
| Method | mAP@0.5 (%) | mAP@0.5:0.9 (%) |
|--------|----------:|----------:|
| SDT (Baseline) | 51.63 | 25.31 |
| **STAtten + SDT** | **52.98** | **27.53** |

在 Object detection 这种极其依赖 spatial-temporal context 融合的任务上，STAtten 带来了 +1.35% 和 +2.22% 的显著提升，证明这个机制不仅仅对 classification 有效，而是真正学习到了更好的 feature representation。

### Energy Consumption Analysis
| Method | Precision | Memory (MB) | Energy (mJ) |
|--------|----------|----------:|----------:|
| SDT (Baseline) | 32(8)/1 | 283.97 | 12.42 |
| **STAtten + SDT** | 32(8)/1 | 283.97 | 12.36 |
| SDT-V2 (Baseline) | 32(8)/1 | 250.22 | 52.40 |
| **STAtten + SDT-V2** | 32(8)/1 | 250.22 | 52.38 |

这个 table 非常震撼。STAtten 的 Energy consumption 和 Baseline 几乎完全一样（甚至微微低了一点点，因为在 memory access 上的 pattern 优化了）。算法变强了，但是功耗没有增加。这对于 SNN 领域来说是梦寐以求的结果。

## 6. The Big Picture: 我的 Takeaway

读完这篇 paper，我的 intuition 是这样的：

1. **Binary 是一种 Constraint，也是一种 Freedom。** 传统 ANN 由于浮点数运算的复杂性，必须依赖 Softmax 来 normalize attention weights。Softmax 强行锁死了矩阵乘法的顺序，导致 video transformer 的计算复杂度是 $O(T^2 N^2 D)$。SNN 因为用了 Binary spikes，抛弃了 Softmax，反而获得了矩阵乘法可结合的 algebraic freedom。这种 constraint 变成 freedom 的范式转移，是非常深刻的系统级 insight。

2. **Local correlation is all you need.** Paper 里关于 dead neuron 的分析非常直观。在 long sequence 或 long temporal duration 的 modeling 中，做 full attention 的信息信噪比极低。STAtten 证明了在 SNN 里，local block-wise correlation 既省内存，又避开了 binary 死区，效果还最好。这和 Transformer 领域的 local window attention (Swin Transformer) 在哲学上殊途同归。

3. **SNNs 需要拥抱自己的 Temporal Nature。** 把 SNN 当成 ANN 用，按 timestep 独立处理，是买椟还珠。SNN 的精髓在于膜电位的动态变化。这篇 paper 通过极其轻量的工程巧思，把 temporal 维度重新拉回了 SNN Transformer 的表征空间里，并且证明了这种表征能力的提升是可以无缝迁移到 detection, transfer learning 等各种下游任务的。

这种“看似只是改了一行矩阵乘法顺序”的工作，背后是对 SNN computational property 的深刻理解，这种 insight-driven 的 research 设计得非常优雅。

**References & Further Reading:**
*   STAtten GitHub Repository: [https://github.com/Intelligent-Computing-Lab-Yale/STAtten](https://github.com/Intelligent-Computing-Lab-Yale/STAtten)
*   Spikformer (The first SNN Transformer): [https://arxiv.org/abs/2209.15425](https://arxiv.org/abs/2209.15425)
*   Spike-driven Transformer (SDT, the main baseline): [https://arxiv.org/abs/2310.07457](https://arxiv.org/abs/2310.07457)
*   Spike-driven Transformer V2: [https://arxiv.org/abs/2404.03663](https://arxiv.org/abs/2404.03663)
*   ViViT (ANN Video Transformer for complexity comparison): [https://arxiv.org/abs/2103.15691](https://arxiv.org/abs/2103.15691)

---

# STAtten: Spiking Transformer with Spatial-Temporal Attention 深度解析

## 1. 背景与核心问题

### 1.1 SNN Transformer的现状

Spiking Neural Networks (SNNs) 通过 binary spike computation 提供 energy-efficient 的计算范式, 可以部署在 TrueNorth, Loihi, Tianjic 等 neuromorphic chips 上. 但是传统 convolution-based SNNs (如 VGGNet, ResNet 的 SNN 版本) 在 binary spike activations 下存在 information loss 问题. 为此, 研究者将 Transformer 架构引入 SNN domain, 产生了 Spikformer, Spike-driven Transformer (SDT), SpikingReformer, QKFormer 等一系列工作.

这些工作有一个共同的**盲点**: 它们只关注 **spatial-only attention**, 即在每个 timestep 内独立计算 self-attention, 完全忽略了 spike 处理中 inherent 的 **temporal dependencies**.

### 1.2 为什么 temporal 信息如此重要?

SNN 的一个本质特征是: **不同 timestep 的 feature map 携带不同的信息**, 这是由两个机制造成的:
- **Leaky factor** τ ∈ (0, 1]: membrane potential 随时间衰减
- **Reset mechanism**: spike firing 后 membrane potential 重置为 0

这意味着同一张 image 在 T=4 时, 第 1 个 timestep 和第 4 个 timestep 的 spike pattern 是不同的, 它们之间存在 temporal evolution. Spatial-only attention 完全无法捕获这种 evolution.

### 1.3 Entropy 分析的 motivation

论文用 Shannon entropy 来量化不同 attention 机制下的信息分布 (Eq. 7):

$$H(\text{Attn}) = -\sum_{t=1}^{T}\sum_{n=1}^{N}\sum_{d=1}^{D} \hat{\mathbf{p}}[t,n,d] \log \hat{\mathbf{p}}[t,n,d]$$

其中 $\hat{\mathbf{p}}[t,n,d] = \text{Softmax}(\text{Attn}[t,n,d])$.

变量含义:
- $t$: timestep index (时间步索引)
- $n$: token position (token 位置)
- $d$: feature dimension (特征维度)
- $T$: total timesteps
- $N$: number of tokens
- $D$: feature dimension

Figure 2 的结果非常有意思:
- **Spatial-only**: entropy = 5.81, accuracy = 77.7% (CIFAR100, pretrained SDT)
- **Spatial-temporal**: entropy = 4.85, accuracy = 79.9%

**Lower entropy → more focused spike patterns → higher accuracy**. 这个反向关系告诉我们: spatial-temporal attention 能产生更 structured 的 feature representation, 这是 STAtten 设计的核心 motivation.

---

## 2. 设计挑战与 insight

### 2.1 为什么不能直接用 full spatial-temporal attention?

如果直接把所有 timestep 的信息全部 correlate (像 ViViT 那样), 会遇到两个严重问题:

#### Challenge 1: Memory 爆炸

Full spatial-temporal attention 的 attention matrix 维度是 $(TN) \times (TN)$, 这导致 matrix multiplication 的中间结果巨大. Figure 3(a) 显示, 在 24GB VRAM 的 A5000 GPU 上, full temporal attention 的 max batch size 比 block-wise 方法小 **1.6 倍**.

#### Challenge 2: Dead neuron 问题

这是 SNN 特有的问题. Binary spike matrix multiplication 在 temporal distance 增大时会产生大量零值. 论文 Appendix C.1 给了一个非常直观的例子.

考虑两个 binary matrices $\mathbf{Q}_t$ 和 $\mathbf{K}_{t'}$ 的乘积:

$$(\mathbf{Q}_t \mathbf{K}_{t'}^\top)_{i,j} = \sum_{d=1}^{D} q_{t,i,d} \cdot k_{t',j,d}$$

变量含义:
- $i, j \in \{1, ..., N\}$: token positions
- $d \in \{1, ..., D\}$: feature dimension
- $q_{t,i,d}$: Q matrix 在 timestep $t$, token $i$, feature $d$ 的 binary value

**关键 insight**: 当 $|t - t'|$ 增大时, spike patterns 变得 less correlated, $q_{t,i,d} \cdot k_{t',j,d} = 0$ 的 probability 增加. 这个 multiplicative effect 在 dimension $D$ 上 accumulate, 导致大量零输出, 进而产生 silent neurons.

具体例子 (Eq. 16-19):

**Nearby timesteps** (t 和 t+1, spike patterns 相似):
$$\mathbf{Q}_t \mathbf{K}_{t+1}^\top = \begin{bmatrix} 3 & 2 & 2 & 3 \\ 2 & 3 & 2 & 1 \\ 2 & 2 & 2 & 2 \\ 2 & 2 & 2 & 2 \end{bmatrix}$$

**Distant timesteps** (t 和 t+Δ, spike patterns 差异大):
$$\mathbf{Q}_t \mathbf{K}_{t+\Delta}^\top = \begin{bmatrix} 1 & 1 & 1 & 1 \\ 1 & 1 & 0 & 1 \\ 1 & 1 & 1 & 1 \\ 0 & 1 & 1 & 1 \end{bmatrix}$$

虽然两个 matrix 的 spike density 相同, 但 distant timesteps 的乘积值显著降低. 由于后面会接 LIF neuron (只有超过 threshold $V_{th}$ 才 fire), 低值无法 trigger spike, 这就是 dead neuron 的来源.

Figure 3(b) 在 CIFAR100 上验证了这一点: same-timestep 的 QKV computation 有更多 active neurons, 而 temporal distance 越大, activity 越低.

### 2.2 Block-wise 设计的 rationale

基于以上两个 challenge, STAtten 提出了 **local temporal correlation** 的设计哲学:
- 只在 **local temporal block** 内 correlate spike patterns, 避免远距离的 silent neuron 问题
- 通过 block partitioning 降低 memory footprint
- 保留 spatial-temporal 信息融合的核心 benefit

---

## 3. STAtten 的核心机制

### 3.1 Block-wise Temporal Partitioning

STAtten 把 temporal sequence 分成 size 为 $B$ 的 blocks (Eq. 8):

$$\text{STAtten}(\mathbf{X}[b]) = \text{LIF}(\mathbf{Q}[b] \mathbf{K}^\top[b] \mathbf{V}[b] \cdot \alpha)$$

其中 $[b] = [iB:(i+1)B, :, d], \quad i \in \{0, 1, ..., T/B - 1\}$

变量含义:
- $[iB:(i+1)B]$: 从 timestep $iB$ 到 $(i+1)B$ 的 aggregated features
- $B$: block size
- $\alpha$: scaling factor (替代了 Softmax 的 normalization 作用)

具体例子:
- Static datasets (CIFAR, ImageNet): $T=4, B=2$ → blocks 为 [0,1], [2,3]
- Neuromorphic datasets (DVS): $T=16, B=4$ → blocks 为 [0,1,2,3], [4,5,6,7], ..., [12,13,14,15]

### 3.2 关键 trick: Q,K,V 的 computation reordering

这是 STAtten 最精妙的设计, 也是它能保持 $O(TND^2)$ 复杂度的关键 (Eq. 9):

$$\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{\frac{T}{B} \times N \times D} \xrightarrow{\text{reshape}} \mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{(\frac{T}{B}N) \times D}$$

$$\mathbf{K}^\top \mathbf{V} \in \mathbb{R}^{D \times D} \xrightarrow{\text{attention}} \mathbf{Q}(\mathbf{K}^\top \mathbf{V}) \in \mathbb{R}^{\frac{T}{B} \times N \times D}$$

**为什么这个 reordering 可行?** 因为 STAtten 去掉了 Softmax! 在 vanilla attention 中:
$$\text{Attn} = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{D}}\right)\mathbf{V}$$

Softmax 是 row-wise normalization, 必须先计算 $\mathbf{Q}\mathbf{K}^\top$ 才能归一化, 然后 multiply $\mathbf{V}$. 这就强制了 computation order: $\mathbf{Q}\mathbf{K}^\top \to \text{Softmax} \to \times \mathbf{V}$. 中间结果 $\mathbf{Q}\mathbf{K}^\top$ 是 $N \times N$ (或 $TN \times TN$ for spatial-temporal).

而 binary spike 不需要 Softmax (因为 binary Q, K, V 不需要 normalization), 所以可以利用 matrix multiplication 的 **associativity**:
$$\mathbf{Q}(\mathbf{K}^\top \mathbf{V}) = (\mathbf{Q}\mathbf{K}^\top)\mathbf{V}$$

**先计算 $\mathbf{K}^\top \mathbf{V}$** (结果是 $D \times D$), 然后 multiply $\mathbf{Q}$. 中间结果从 $(TN)^2$ 降到了 $D^2$, 这是巨大的 memory 节省.

### 3.3 复杂度对比 (Table 1)

| Method | ST | Complexity | Energy |
|--------|----|-----------:|-------:|
| ViT | ✗ | $O(N^2 D)$ | $E_{MAC} \cdot N^2 D$ |
| ViViT | ✓ | $O(T^2 N^2 D)$ | $E_{MAC} \cdot T^2 N^2 D$ |
| Spikformer | ✗ | $O(TND^2)$ | $E_{AC} \cdot TND^2 \cdot (S_Q + S_K + S_V)$ |
| SDT | ✗ | $O(TND)$ | $E_{AC} \cdot TND \cdot (S_Q + S_K)$ |
| SDT-V2 | ✗ | $O(TND^2)$ | $E_{AC} \cdot TND^2 \cdot (S_Q + S_K + S_V)$ |
| QKFormer | ✗ | $O(TND^2)$ | $E_{AC} \cdot TND^2 \cdot (S_Q + S_K + S_V)$ |
| **STAtten** | ✓ | $O(TND^2)$ | $E_{AC} \cdot TND^2 \cdot (S_Q + S_K + S_V)$ |

变量含义:
- $E_{MAC}$: 32-/8-bit ANN multiply-accumulate 操作的 energy (≈ 4.6 pJ in 45nm CMOS)
- $E_{AC}$: binary SNN accumulate 操作的 energy (≈ 0.9 pJ)
- $S_Q, S_K, S_V$: Q, K, V 的 firing rates (spike 稀疏性)
- $T$: timesteps, $N$: patches/tokens, $D$: dimension

**关键观察**: STAtten 虽然引入了 spatial-temporal attention, 但 complexity 和 energy 都与 spatial-only spiking transformers 相同, 而 ViViT 这样的 ANN spatial-temporal 方法需要 $O(T^2 N^2 D)$ 的复杂度. 这完全得益于 non-softmax 设计.

---

## 4. Plug-and-Play 集成

STAtten 的设计哲学是 **不破坏现有架构的核心特性**, 只替换 attention module. 论文详细分析了 4 种主流 spiking transformer 的集成方式.

### 4.1 Residual Connection 的两种范式

#### SEW (Spike Element-Wise) Shortcut
- 使用者: Spikformer, QKFormer
- 机制: spike activations 直接加到 attention output
- $\mathbf{F}_{out} = \text{STAtten}(\mathbf{X}) + \text{spikes}$

#### MS (Membrane-Shortcut)
- 使用者: SDT, SDT-V2, SpikingReformer
- 机制: 传播 membrane potential 而非 spikes
- 这避免了 spike 的 quantization error accumulate

STAtten 与两种 shortcut 都兼容.

### 4.2 各架构的具体集成

**Spikformer (Eq. 10)**:
$$\text{SSA}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathbf{LIF}(\mathbf{Q} \odot \mathbf{K}^\top \odot \mathbf{V} \cdot \alpha)$$

Spikformer 确立了两个原则: (1) non-softmax, (2) flexible Q,K,V ordering. 但 SSA 在每个 timestep 独立处理. STAtten 直接替换 SSA.

**Spike-driven Transformer (Eq. 11)**:
$$\text{SDSA}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathbf{Q} \otimes \text{LIF}\left(\sum_c (\mathbf{K} \otimes \mathbf{V})\right)$$

SDSA 用 element-wise multiplication $\otimes$ 和 column-wise summation $\sum_c$ 实现 extreme efficiency, 但代价是 even spatial domain 内的 feature correlation 都受限. STAtten 恢复了完整的 spatial-temporal correlation.

**SpikingReformer (Eq. 12)**:
$$\text{DSSA}(\mathbf{X}) = \mathbf{LIF}(\mathbf{X} \odot \mathbf{W}^\top \odot \mathbf{X}^\top \cdot \alpha)$$

DSSA 甚至省略了 Q, K, V 的区分, 直接用 input $\mathbf{X}$ 和 projection $\mathbf{W}$. STAtten 替换 DSSA 后证明: 传统的 Q, K, V matrix multiplication 在 spike-based processing 中仍然是 viable 的.

**QKFormer (Eq. 13)**:
$$\text{QKTA} = \text{LIF}\left(\sum_N (\mathbf{Q})\right) \otimes \mathbf{K}$$

QKFormer 有 dual attention: QKTA (token-wise) + SSA (spatial). STAtten 只替换 SSA, 保留 QKTA.

### 4.3 整体信息流

替换后, 各 block 的输出 concat 起来:
$$\mathbf{F}_{out} = \text{Concat}\left(\text{STAtten}(\mathbf{X}[iB:(i+1)B])_{i=0}^{T/B-1}\right)$$

然后通过 conv + BN:
$$\mathbf{Z} = \mathbf{BN}(\text{Conv}(\mathbf{F}_{out}))$$

---

## 5. 实验结果详解

### 5.1 Sequential CIFAR10/100 (Table 2)

这个实验最直接地验证了 temporal modeling 能力. 每张 image 被列向切分成 32 个 columns, 每个 column 作为一个 timestep 的 input. 这样就把 spatial image classification 转化成了 temporal sequence processing.

| Method | s-CIFAR10 (%) | s-CIFAR100 (%) |
|--------|--------------:|---------------:|
| Spikformer [64] | 79.29 | 57.17 |
| STAtten + [64] | 82.45 (+3.16) | 58.75 (+1.58) |
| SDT [53] | 80.32 | 61.08 |
| **STAtten + [53]** | **83.41 (+3.09)** | **64.30 (+3.22)** |
| SpikingReformer [42] | 79.25 | 57.48 |
| STAtten + [42] | 81.84 (+2.59) | 58.30 (+0.82) |

**Insight**: 在 sequential task 上, STAtten 对 SDT 的提升最大 (3.09%, 3.22%), 这是因为 SDT 的 SDSA 用 element-wise multiplication 严重限制了 feature correlation, STAtten 的 full matrix multiplication 在这里 benefit 更明显.

### 5.2 Standard CIFAR10/100 (Table 3)

| Method | Architecture | T | CIFAR10 | CIFAR100 |
|--------|-------------|---|--------:|---------:|
| Spikformer | 4-384 | 4 | 93.99 | 75.06 |
| STAtten + [64] | 4-384 | 4 | 94.36 | 75.85 |
| SDT | 2-512 | 4 | 95.60 | 78.40 |
| STAtten + [53] | 2-512 | 4 | 96.03 | 79.85 |
| SpikingReformer | 6-384 | 4 | 95.03 | 77.16 |
| STAtten + [42] | 6-384 | 4 | 95.26 | 77.90 |
| QKFormer | 4-384 | 4 | 95.12 | 79.79 |
| STAtten + [62] | 4-384 | 4 | 95.35 | **80.20** |

即使在 standard image classification (temporal 信息相对弱) 上, STAtten 仍能稳定提升 0.2-1.5%, 说明 spatial-temporal fusion 对静态数据也有帮助 (可能是因为 SNN 的 T=4 编码本身就引入了 temporal dynamics).

### 5.3 ImageNet (Table 4)

这是最关键的 scaling 实验:

| Method | Param (M) | T | Acc (%) |
|--------|----------:|---|--------:|
| Spikformer | 66.34 | 4 | 74.81 |
| SpikingReformer | 66.38 | 4 | 78.77 / 79.40* |
| QKFormer | 64.96 | 4 | 84.22 |
| SDT | 29.68 | 4 | 74.57 |
| SDT (8-768) | 66.34 | 4 | 76.32 / 77.07* |
| STAtten + [53] | 29.68 | 4 | 76.18 / 76.56* |
| STAtten + [53] (8-768) | 66.34 | 4 | 78.11 / 78.39* |
| SDT-V2 | 55.4 | 4 | 79.49† / 79.98*† |
| **STAtten + [52]** | 55.4 | 4 | **79.85 / 80.67*** |

*表示 288×288 resolution, †表示作者自己复现.

**关键 insight**: STAtten + SDT-V2 用 55.4M 参数就达到 80.67% accuracy, 比 SpikingReformer (66.38M, 79.40%) 和 Spikformer (66.34M, 74.81%) 都要好, 同时参数量少 20%. 这说明 spatial-temporal attention 带来的 representation capacity 提升可以转化为参数效率.

### 5.4 Neuromorphic Datasets (Table 5)

| Method | T | CIFAR10-DVS | N-Caltech101 |
|--------|---|------------:|-------------:|
| Spikformer | 16 | 80.9 | - |
| SDT | 16 | 80.0 | 81.80 |
| STAtten + [53] | 16 | 81.1 | 83.15 |
| SpikingReformer | 16 | 78.80 | 81.29 |
| STAtten + [42] | 16 | 80.60 | 81.95 |
| QKFormer | 16 | 82.90 | 83.58 |
| **STAtten + [62]** | 16 | **83.90** | **84.25** |

Neuromorphic datasets (DVS) 本身就是 event-based temporal data, 所以 STAtten 的提升在这里特别有意义. QKFormer + STAtten 在 CIFAR10-DVS 上达到 83.90%, 在 N-Caltech101 上达到 84.25%, 都是 SOTA.

### 5.5 Memory & Energy (Table 6)

| Method | Precision | T | Memory (MB) | Energy (mJ) |
|--------|----------|---|------------:|------------:|
| ViT-B/16 | 32/32 | 1 | 351.8 | 254.84 |
| Liu (ViT-B) | 8/8 | 1 | 87.96 | 110.8 |
| Spikformer | 32(8)/1 | 4 | 285.99 (86.48) | 21.48 |
| SpikingReformer | 32(8)/1 | 4 | 273.57 (85.4) | 8.76 |
| QKFormer | 32(8)/1 | 4 | 280.28 (84.99) | 38.91 |
| SDT | 32(8)/1 | 4 | 283.97 (87.42) | 12.42 |
| STAtten + [53] | 32(8)/1 | 4 | 283.97 (87.42) | 12.36 |
| SDT-V2 | 32(8)/1 | 4 | 250.22 (84.02) | 52.40 |
| STAtten + [52] | 32(8)/1 | 4 | 250.22 (84.02) | 52.38 |

**关键观察**: STAtten 几乎不增加任何 energy overhead (12.36 vs 12.42 mJ, 52.38 vs 52.40 mJ). 这是因为 STAtten 的 complexity 和 baseline 相同, 只是 computation order 改变了. 这对 neuromorphic hardware deployment 极其友好.

### 5.6 Model Capacity (Figure 5)

在 CIFAR100 上, STAtten 在 2.56M 到 22.97M 参数范围内都保持 0.5-1.0% 的稳定提升. 在 Sequential CIFAR100 上, 提升达到 3-5%, 即使在小模型上也很明显. 这说明 STAtten 的 benefit 不是依赖大模型的 capacity, 而是 attention 机制本身的 design 改进.

---

## 6. Ablation Study 的关键 insights

### 6.1 Timestep Combination (Table 9)

这个实验验证了 "nearby timesteps 应该一起处理" 的 hypothesis. 在 CIFAR100 (T=4, B=2) 上:

- **Same-range combination**: Q,K,V 都用 [1,2] 或 [3,4] → accuracy 79.85%
- **Cross-range combination**: Q,V 用 [1,2], K 用 [3,4] → accuracy 79.28-79.09%

Cross-range combination 性能下降 0.5-0.8%, 证实了 distant timesteps 之间的 spike correlation 弱, 会导致 information loss.

### 6.2 Block Size Analysis (Table 10)

| Dataset | T | B | Accuracy |
|---------|---|---|---------:|
| CIFAR100 | 4 | 2 | 79.85 |
| CIFAR100 | 4 | 4 | 79.90 |
| ImageNet | 4 | 1 | 77.65 |
| ImageNet | 4 | 2 | 78.00 |
| ImageNet | 4 | 4 | 78.06 |
| s-CIFAR100 | 32 | 8 | 60.89 |
| s-CIFAR100 | 32 | 16 | 62.95 |
| s-CIFAR100 | 32 | 32 | 64.30 |
| N-Caltech101 | 16 | 4 | 83.15 |
| N-Caltech101 | 16 | 8 | 82.49 |
| N-Caltech101 | 16 | 16 | 82.40 |

**重要 insight**: Optimal block size 取决于 **temporal-to-spatial information ratio**:
- **Vision tasks** (CIFAR, ImageNet): temporal 信息相对弱, smaller blocks 更好 (保留 spike correlation)
- **Sequential tasks** (s-CIFAR100): temporal 信息 dominate, larger blocks 更好 (更长的 temporal modeling)
- **Neuromorphic** (N-Caltech101): B=4 最优, 介于两者之间

这个结论对未来 SNN transformer 的设计很有指导意义.

---

## 7. Limitations 与 Future Work

论文诚实地讨论了 hardware deployment 的挑战:

1. **Full spatial-temporal attention** (无 block partitioning) 在 neuromorphic deployment 上 impractical, 因为需要 complete temporal information.
2. **Block-wise approach** 只是部分解决, 在传统 neuromorphic chips (TrueNorth, Loihi) 上仍然有挑战, 因为这些 chips 是 step-by-step processing 的.
3. **Layer-by-layer neuromorphic chips** ([22, 54, 56, 63]) 可能是更好的 deployment 方案, 因为它们支持 layer-wise processing across timesteps.
4. **Parallel LIF neurons** ([15]) 可以进一步加速 layer-by-layer 架构中的 computation.

---

## 8. Vision Tasks 的扩展 (Appendix D)

### 8.1 Object Detection (PASCAL VOC, Table 11)

| Method | mAP@0.5 | mAP@0.5:0.9 |
|--------|--------:|------------:|
| Spiking-YOLO | 51.83 | - |
| SDT | 51.63 | 25.31 |
| STAtten + [53] | 52.98 | 27.53 |

STAtten 作为 EMS-YOLO 的 backbone, mAP@0.5 提升 1.35%, mAP@0.5:0.9 提升 2.22%, 证明其不仅限于 classification.

### 8.2 Transfer Learning (Table 12)

| Method | CIFAR-10 | CIFAR-100 |
|--------|---------:|---------:|
| Spikformer | 97.03 | 83.83 |
| SpikingReformer | 97.40 | 85.98 |
| STAtten + [53] | 97.76 | 86.67 |

用 ImageNet pre-trained weights, STAtten 在 transfer learning 上也表现最好, 说明 spatial-temporal features 更具 generalization 能力.

---

## 9. Energy Calculation Details (Appendix A, Table 7)

详细的 energy 计算公式:

| Block | Layer | Energy Consumption |
|-------|-------|-------------------:|
| Embedding | 1st Conv | $E_{MAC} \cdot F_{Conv} \cdot T$ |
| Embedding | Other Convs | $E_{AC} \cdot F_{Conv} \cdot T \cdot S_{Conv}$ |
| Attention | Q, K, V | $3 \cdot E_{AC} \cdot F_{Conv} \cdot T \cdot S_{Conv}$ |
| Attention | Self-attention | $E_{AC} \cdot TND^2 \cdot (S_K + S_V + S_Q)$ |
| Attention | MLP | $E_{AC} \cdot F_{Conv} \cdot T \cdot S_{Conv}$ |
| MLP | MLP1 | $E_{AC} \cdot F_{Conv} \cdot T \cdot S_{Conv}$ |
| MLP | MLP2 | $E_{AC} \cdot F_{Conv} \cdot T \cdot S_{Conv}$ |

Conv layer 的 FLOPs (Eq. 14):
$$F_{Conv} = K \cdot K \cdot H_{out} \cdot W_{out} \cdot C_{in} \cdot C_{out}$$

变量含义:
- $K$: kernel size
- $H_{out}, W_{out}$: output feature map 的高和宽
- $C_{in}, C_{out}$: input/output channel dimension

第一层 Conv 用 $E_{MAC}$ 因为 input 是 float pixel (direct coding 还没转成 spike), 后续层都是 binary spike 用 $E_{AC}$.

Appendix E 给出了 SDT 8-768 在 ImageNet 上每层的 firing rate 和 energy, 可以看到不同 timestep 的 firing rate 有显著差异 (如 Encoder-1 的 Q,K,V: T=1 时 0.2159, T=4 时 0.2728), 这印证了 temporal information 确实不同.

---

## 10. 整体 Intuition 总结

STAtten 的核心 insight 可以归纳为三层:

1. **Temporal information 在 SNN 中是真实的、有价值的**: 不同 timestep 的 spike pattern 不同 (由 leaky + reset 造成), 应该被 attention 机制利用而非忽略.

2. **Full temporal correlation 在 SNN 中是 harmful 的**: Binary multiplication 对 distant spikes 极度敏感, 会产生大量 silent neurons. Local temporal block 是正确的 balance point.

3. **Non-softmax design 是 enabler**: 去掉 Softmax 后, Q, K, V 的 computation order 变得 flexible, 可以先算 $\mathbf{K}^\top \mathbf{V}$ (D×D) 再算 $\mathbf{Q}(\mathbf{K}^\top \mathbf{V})$, 完美避开 $(TN)^2$ 的中间结果, 让 spatial-temporal attention 的 complexity 保持在 $O(TND^2)$.

这种 design pattern (利用 binary + non-softmax 的 algebraic freedom 来 reorder computation) 可能对未来 SNN transformer 架构设计有更广泛的启发, 比如可以扩展到更长 sequences, 更复杂的 temporal patterns, 甚至 multi-modal spike fusion.

**References**:
- Paper GitHub: https://github.com/Intelligent-Computing-Lab-Yale/STAtten
- Spikformer: https://arxiv.org/abs/2209.15425
- Spike-driven Transformer: https://arxiv.org/abs/2310.07457 (NeurIPS 2024)
- Spike-driven Transformer V2: https://arxiv.org/abs/2404.03663
- SpikingReformer: https://arxiv.org/abs/2304.11954 (CVPR 2024)
- QKFormer: https://arxiv.org/abs/2403.16552
- ViViT (ANN spatial-temporal baseline): https://arxiv.org/abs/2103.15691
- MS-ResNet (Membrane Shortcut): https://arxiv.org/abs/2302.01808
