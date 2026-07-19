---
source_pdf: BHViT.pdf
paper_sha256: 690b53bc80783db9aae441272e29d2121664a04675d03c8d4575b3ebed490ceb
processed_at: '2026-07-18T18:14:01-07:00'
target_folder: Pruning-Quantization
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# BHViT: Binarized Hybrid Vision Transformer 深度解析

## 1. Paper 的核心 Motivation 与 Big Picture

这篇 paper 解决的核心痛点是：把 binary CNN 的成功技巧（比如 ReActNet 的 RSign、RPReLU）直接搬到 ViT 上会崩盘。从 Figure 1 可以看到 ReActNet 在 MobileNet 上能拿到 70%+ 的 ImageNet accuracy，但套到 ViT 上直接掉到 49% 左右。作者定位到两个根本原因：

1. **Attention 模块的反向传播被多个 clip 函数和 sign operator 的 non-differentiability 破坏**，导致 activation 中大部分元素 gradient vanishing；
2. **Binary attention matrix 无法准确表达 token 之间相似度的差异**，因为二值化后只有 {-1, +1} 或 {0, 1}，引入大量 noise，降低 SNR。

这篇文章的精髓在于它构建了一套 intuition-driven 的设计哲学：不强行把 ViT 二值化，而是改造 ViT 的架构使其"binarization-friendly"，这是一个非常聪明的研究思路。代码开源在 https://github.com/IMRL/BHViT 。

## 2. 三个关键 Observation 的数学剖析

这部分是这篇 paper 的灵魂。作者没有停留在经验性观察，而是给出了相当严肃的数学推导。

### Observation 1: Avoiding excessive numbers of tokens is beneficial for Binary ViT

**直觉理解**：在 full-precision 的 attention 里，token 数量增加只是让 softmax 分布更尖锐；但在 binary attention 里，token 数量增加会让 attention matrix 趋向于 uniform distribution，这就摧毁了 attention 的选择性。

**数学推导**：

考虑 attention matrix 在 softmax 之前的一行 $x = [x_1, x_2, \ldots, x_k]$，其中 $k$ 是 token 数量。每个 $x_i$ 是 binary Q 和 binary K 的点积：

$$x_i = \sum_{l=1}^{d} B_a(\mathbf{Q}, a_1, b_1)^{m,l} \times B_a(\mathbf{K}^T, a_2, b_2)^{l,i}$$

- $d$: Q 和 K 的 channel 数量
- $l$: channel 索引
- $m, i$: token 索引
- $B_a(\cdot)$: 二值化函数
- $a_1, a_2, b_1, b_2$: scale 和 bias

每个乘积项 $\gamma = B_a(Q)^{m,l} \times B_a(K^T)^{l,i}$ 服从 Bernoulli 分布（$p \approx 0.5$）。所以 $x_i$ 实际服从 binomial distribution，取值为 $2t - d$（$t$ 是 +1 的个数）。

根据 **De Moivre-Laplace theorem**，当 $d$ 足够大（ViT 中 $d \geq 256$），$x_i$ 可以用 Gaussian $\mathcal{N}(\mu, \sigma^2)$ 近似。

Gaussian distribution 的信息熵：
$$H_G(\mathbf{x}, k) = \frac{k}{2} \ln(2\pi e \sigma^2)$$

注意这里的 $k$ 是 token 数量，$H_G$ 与 $k$ 成正比。

然后经过 softmax 后，entropy 变为：
$$H_s(\mathbf{x}, k) = \ln(k) + \mu + \frac{\sigma^2}{2} - \mu_s$$

其中 $\mu_s = \frac{\sum_{i=1}^k e^{x_i} \cdot x_i}{\sum_{j=1}^k e^{x_j}}$ 是 weighted sum，且 $\mu_s < d$（有上界）。

关键结论：**当 $k \to \infty$ 时，$p_{sof}^i \approx \frac{1}{k}$**，即 attention 趋向 uniform distribution。这对 attention mechanism 是致命的，因为 uniform distribution 意味着所有 token 被同等对待。

**另一个角度**：当 token 数量大时，$A_{tt}$ 的值都很小，导致 binarization 的 scale factor $a$ 必须非常小才能把 $\frac{A_{tt} - b}{a}$ 映射到 $[0,1]$。小 $a$ 导致前向传播时 binarization 结果很小，反向传播时 gradient 直接消失。

**Build intuition**: 这个观察告诉我，binary ViT 里 attention 的"分辨率"被严重压缩了。Full-precision attention 用浮点数表达 token 之间微妙的相似度差异，binary 后只剩两个状态，所以 token 越多，binary 越无法区分谁更相似。这就像用黑白二色画渐变图，颜色越多越画不清楚。

### Observation 2: Adding a residual connection in each binary layer is beneficial for binary ViT

**直觉理解**：ViT 原生只在 MLP 和 attention 外面套一层 residual，但 binary layer 内部没有任何 shortcut。这意味着反向传播时 gradient 要穿过多个 binarization function，每个都会 truncate gradient，导致 vanishing gradient。

**数学推导**：

考虑 attention 模块里 $Y$ 对 $W_q$ 的 Jacobian：

$$\frac{\partial Y^{n_l, c_i}}{\partial W_q^{c_i, c_j}} = \frac{\partial Y}{\partial B(A)} \cdot \frac{\partial B(A)}{\partial A} \cdot \frac{\partial A}{\partial M} \cdot \frac{\partial M}{\partial B(Q)} \cdot \frac{\partial B(Q)}{\partial Q} \cdot \frac{\partial Q}{\partial B(W_q)} \cdot \frac{\partial B(W_q)}{\partial W_q}$$

变量解释：
- $n_l$: token 索引
- $c_i, c_j$: channel 索引
- $t$: token 数量
- $d$: channel 数量
- $B()$: binarization 函数
- $M$: softmax 之前的 attention matrix
- $A$: softmax 之后的 attention matrix

代入具体值后得到：
$$\frac{\partial Y^{n_l, c_i}}{\partial W_q^{c_i, c_j}} = G \cdot \frac{\partial B(Q^{n_l, c_i})}{\partial Q^{n_l, c_i}} \cdot B(X^{n_l, c_j}) \cdot \frac{\partial B(W_q^{c_i, c_j})}{\partial W_q^{c_i, c_j}}$$

其中：
$$G = \sum_{k=1}^{t} \left( B(V^{n_k, c_i})^T \cdot \mathbf{1}_{0.5 \leq A^{n_l, n_k} \leq 1} \cdot H_k \right)$$
$$H_k = \frac{A^{n_l, n_k} \otimes (1 - A^{n_l, n_k})}{\sqrt{d}} \cdot B(K^{c_i, n_k})$$

这里 $\mathbf{1}_{0.5 \leq A^{n_l, n_k} \leq 1}$ 是 indicator function（binary attention 取 {0, 1}），$\otimes$ 是 Hadamard product。

加入 residual connection（从 Q 直接连到 Y）后：
$$\frac{\partial Y^{n_l, c_i}}{\partial W_q^{c_i, c_j}} = \left(1 + G \cdot \frac{\partial B(Q^{n_l, c_i})}{\partial Q^{n_l, c_i}}\right) \cdot B(X^{n_l, c_j}) \cdot \frac{\partial B(W_q^{c_i, c_j})}{\partial W_q^{c_i, c_j}}$$

那个 $+1$ 就是 residual 的贡献，它确保 gradient 至少有 1，避免完全消失。

**Build intuition**: 这就像给信号加 DC bias。原本 gradient 路径上每个 binarization 都像一个 high-pass filter 把信号衰减，加上 residual 后信号永远有一个 baseline 不会被衰减到 0。这个思路在 Bi-RealNet 中已经用过，但作者在 binary attention 内部使用是新的应用。

### Observation 3: Adam optimizer enlarges weight oscillation in later stages of training

**直觉理解**：Binary network 的 latent weight 在 0 附近会震荡，而 Adam 的 second-order momentum 会放大这种震荡，让大量参数最终 deactivate。

**数学推导**：

Adam 的更新规则：
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$s_t = \beta_2 s_{t-1} + (1-\beta_2) g_t^2$$

- $m_t$: first-order momentum
- $s_t$: second-order momentum
- $g_t$: gradient at step $t$
- $\beta_1 = 0.9, \beta_2 = 0.999$

Bias correction 后：
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{s}_t = \frac{s_t}{1 - \beta_2^t}$$

最终更新梯度：
$$g_t' = \frac{\eta \hat{m}_t}{\sqrt{\hat{s}_t} + \varepsilon}$$

- $\eta$: learning rate
- $\varepsilon = 10^{-8}$

代入化简后（用 geometric series 性质 $\sum_{i=1}^t \beta^i = \frac{\beta(1-\beta^t)}{1-\beta}$）：

$$g_t' \approx 3.51\eta \cdot \frac{\sum_{i=1}^t \beta_1^{t-i+1} g_i}{\sqrt{\sum_{i=1}^t \beta_2^{t-i+1} g_i^2} + \sqrt{\sum_{i=1}^t \beta_2^i} \cdot \varepsilon}$$

注意 $\frac{\sqrt{\sum_{i=1}^t \beta_2^i}}{\sum_{i=1}^t \beta_1^i} \approx 3.51$ 当 $i > 5000$。

**关键 insight**: 当 weight oscillation 发生时，$g_i$ 的 sign 频繁变化，导致分子 $\sum \beta_1^{t-i+1} g_i$ 在不同 iteration 之间相互抵消；但分母中的 $\sum \beta_2^{t-i+1} g_i^2$（注意 $g_i^2$ 永远为正）却持续增长，而且 $\beta_2^t$ 衰减比 $\beta_1^t$ 慢得多。结果是 $g_t'$ 越来越小，参数被"困住"在 0 附近。

**Build intuition**: Adam 在 full-precision network 中表现好是因为它对 sparse gradient 有自适应能力。但在 binary network 中，weight 必须从 latent value 收敛到 ±1，这个收敛过程需要"果断"的更新。Adam 的 second-order momentum 反而让 update 变得"优柔寡断"，把 weight 锁在 0 附近震荡。

参考链接：
- Adam 原始 paper: https://arxiv.org/abs/1412.6980
- Oscillation in QAT: https://arxiv.org/abs/2203.11086

## 3. BHViT 架构细节解析

### 3.1 整体架构

参考 Figure 3，BHViT 是一个 4-stage pyramid 结构：
- **Stage 1, 2**: 用 MSGDC（Multi-Scale Grouped Dilated Convolution）作为 token mixer
- **Stage 3, 4**: 用 MSMHA（Multi-Scale Multi-Head Attention）作为 token mixer
- 每个 stage 之间用 stride=2, kernel=2×2 的 conv 下采样
- 每个维度 channel 数：[64, 128, 256, 512]（BHViT-Tiny 为 [n, 2n, 4n, 8n]，n=64）

**为什么要 hybrid？** Table 7 给出答案：
- Hybrid（MSGDC + MSMHA）：70.1%
- 纯 MSMHA：68.8%（差 1.3%）
- 纯 MSGDC：67.2%（差 2.9%）

CNN 在早期 stage 处理 high-resolution feature 更高效，binary 算子也更适合卷积；后期 stage 的 token 数量已经降下来，可以用 attention 做 global reasoning。

### 3.2 MSGDC 模块

公式：
$$H_{l-1}^n = \text{RPReLU}(B.Cov_{3\times 3, g}^{dil=2n-1}(B_a(X_{l-1})) + X_{l-1})$$
$$H_{l-1} = bn(H_{l-1}^1 + H_{l-1}^2 + H_{l-1}^3)$$

- $n \in \{1, 2, 3\}$: 三个不同的 dilation rate
- $2n-1$: dilation rate，即 {1, 3, 5}
- $g$: grouped conv 的 group 数
- $B.Cov$: binary convolution
- $B_a()$: activation binarization function

**Intuition**: 三个不同的 dilation 让 receptive field 多尺度融合，弥补 binary conv 表达能力不足的问题。Grouped conv 减少 parameter 和 computation。

### 3.3 MSMHA 模块

这是 paper 的核心创新之一。结构是 window attention 的变体，结合 global context。

**输入处理**：
1. 对 $X_{l-1} \in \mathbb{R}^{H \times W \times C}$ 做 7×7 average pooling，得到 high-scale feature $X_{l-1}^{high} \in \mathbb{R}^{\frac{H}{7} \times \frac{W}{7} \times C}$
2. 把 spatial resolution 分成 7×7 的 window，得到 $X_{l-1}^{win} \in \mathbb{R}^{\frac{HW}{49} \times 7 \times 7 \times C}$
3. 把 $X_{l-1}^{win}$ flatten 成 1D，与 $X_{l-1}^{high}$（repeat 后 flatten）concat，得到 hidden state $H \in \mathbb{R}^{\frac{HW}{49} \times (49 + \frac{HW}{49}) \times C}$

**Q, K, V 计算**：
$$Q_{l-1} = B_q(H), \quad K_{l-1} = B_k(H), \quad V_{l-1} = B_v(H)$$
$$B_q(X) = \text{RPReLU}(bn(B_{Lin}(X, W_q)) + X)$$

注意 $B_q, B_k, B_v$ 都有 internal residual connection（这就是 Observation 2 的应用）。

**Attention 计算**：
$$A_{tt} = \text{softmax}\left(\frac{B_a(Q_{l-1}) * B_a(K_{l-1})}{\sqrt{d}}\right)$$

### 3.4 Quantization Decomposition (QD)

这是 attention matrix binarization 的核心 trick。

**Motivation**: Softmax 输出在 [0, 1] 之间，binary 后只能选 {0, 1}，丢失了大量信息。

**方法**: 引入全局 scaling constant $s = 2^n - 1$（paper 中 $n=2$，所以 $s=3$）。把 attention matrix 分解成 $s$ 个 binary matrix：

$$\hat{A}_{tt}^\sigma = \varphi\left(\text{round}(s A_{tt}) \geq \sigma - 0.5\right), \quad \sigma = (1, 2, \ldots, s)$$

- $\varphi()$: boolean function
- $\sigma$: 阈值索引
- $s$: scaling constant

这样，原本是 0.7 的 attention 值，会被分解为：$\hat{A}^1 = 1, \hat{A}^2 = 1, \hat{A}^3 = 0$（因为 $3 \times 0.7 = 2.1$，大于 0.5, 1.5 但小于 2.5）。

然后 V 的计算：
$$V_l = \left(\sum_{\sigma=1}^s \hat{A}_{tt}^\sigma \circledast B_a(V_{l-1})\right) + Q_{l-1} + K_{l-1} + V_{l-1}$$

- $\circledast$: binary matrix multiplication（Xnor + popcount）
- 后面 $+Q_{l-1} + K_{l-1} + V_{l-1}$ 是三个 residual shortcut

**Intuition**: QD 本质上是把 attention 的量化精度从 1-bit 提升到 log2(s)-bit，但实现上仍然用 binary operation。这是 binary network 中很经典的"用多组 binary basis 近似 multi-bit"思路，类似于 XNOR-Net 的多基近似。论文中提到这个 trick 用 logical operation 实现开销很小。

参考链接：
- XNOR-Net: https://arxiv.org/abs/1603.05279
- Attention quantization 相关: https://arxiv.org/abs/2211.07091

### 3.5 Binary MLP with Shift Operations

参考 Figure 5，作者在标准 binary MLP 基础上加入 shift operation。

**MLP 计算**：
$$H_{l-1}^{10} = bn(B_{L1}(B_a(H_{l-1})))$$
$$H_{l-1}^1 = \text{RPReLU}(H_{l-1}^{10} + \text{repeat}(H_{l-1}))$$
$$H_{l-1}^{20} = bn(B_{L2}(B_a(H_{l-1}^1)))$$
$$H_{l-1}^2 = \text{RPReLU}(H_{l-1}^{20} + \text{poal}(H_{l-1}^1))$$
$$S_k = L_s(S_k^{hor}(H_{l-1})) + L_s(S_k^{ver}(H_{l-1})) + L_s(S_k^{mix}(H_{l-1})), \quad k \in \{1, 2\}$$
$$X_l = H_{l-1}^2 + S_1 + S_2$$

变量：
- $B_{L1}$: expand ratio 4 的 binary linear layer
- $B_{L2}$: shrink ratio 0.25 的 binary linear layer
- $L_s$: learnable channel-wise scaling
- $\text{repeat}$: 把 input 重复 4 次 concat
- $\text{poal}$: 1D average pooling with stride 4
- $S_1, S_2$: 两组不同 stride 的 shift operations

**Shift operation 三种类型**：
1. **Horizontal shift**: 整张 feature map 向右移 1 像素（第一列移到最后一列）
2. **Vertical shift**: 类似 horizontal，但沿垂直方向
3. **Mixed shift**: 找当前 token 的 4 个邻居 token，每个取 1/4 的 channel 特征，concat 后替换当前 token

**Intuition**: Shift operation 在 Binary MLP 中的作用是引入 spatial 信息交互。Binary linear layer 只做 channel-wise mixing，spatial 信息必须靠 attention 或 conv，但 MLP 没有。Shift 操作零计算量（只是 memory copy），却能引入 local spatial context，这对 binary 网络很关键，因为 binary 表达能力受限，需要更多 inductive bias 补偿。

这种思路让我想起 TSM (Temporal Shift Module) 和 MobileViT 的设计哲学：用几乎零成本的操作引入 spatial/temporal 交互。Shift operation 在 efficient network design 中是一个被反复验证有效的 trick。

参考链接：
- TSM: https://arxiv.org/abs/1811.08383
- MobileViT: https://arxiv.org/abs/2110.02178

## 4. Training Settings

### 4.1 Knowledge Distillation

用 DeiT-Small 作为 teacher model。Loss function：
$$\mathcal{L} = (1 - \lambda - \beta) \mathcal{L}_{cls} + \lambda \mathcal{L}_{dis} + \beta \mathcal{L}_{re}$$

- $\mathcal{L}_{cls}$: cross-entropy with ground truth label
- $\mathcal{L}_{dis}$: cross-entropy with teacher's output
- $\mathcal{L}_{re}$: L1 regularization on latent weights
- $\lambda = 0.8$（最佳值，见 Table 6）
- $\beta$: 动态调度，训练后期（$T_{now} \geq 0.9 \times T_{max}$）才启用，值为 0.1

**L1 Regularization**：
$$\mathcal{L}_{re} = \frac{1}{n} \sum_{i=1}^n \left| |w_i| - 1 \right|$$

这个 loss 直接鼓励 latent weight 远离 0，靠近 ±1。这与 Observation 3 的分析完全一致——Adam 会把 weight 锁在 0 附近，L1 regularization 反向 push weight 向 ±1 移动。

**Intuition**: 这个 trick 的精髓在于"时机"。如果在训练初期就加 L1，会破坏 weight 探索；只在最后 10% 的 epoch 加，让 weight 先正常学习，最后再"定型"。这是一种 schedule-aware regularization，类似于 cosine annealing 的思想。

参考链接：
- DeiT: https://arxiv.org/abs/2012.12877
- Weight oscillation in BNN: https://arxiv.org/abs/2203.11086

## 5. 实验结果深度分析

### 5.1 CIFAR-10 Results (Table 1)

| Architecture | Method | W-A | NP | Top-1 (%) |
|---|---|---|---|---|
| ResNet-18 | ReActNet | 1-1 | 11.2M | 92.31 |
| ResNet-18 | ReCU | 1-1 | 11.2M | 92.80 |
| DeiT-Small | GSB | 1-1 | 21.6M | 91.20 |
| **BHViT** | **Ours-Tiny** | **1-1** | **13.2M** | **93.30** |
| **BHViT** | **Ours-Small** | **1-1** | **22.1M** | **95.00** |

**关键发现**: BHViT-Tiny 用 13.2M 参数超过 ReCU 的 92.80%，这是 binary ViT 第一次在 small dataset 上超过 binary CNN。这是非常重要的 milestone，因为之前 binary ViT 在小数据集上表现差被认为是 ViT 缺乏 inductive bias 的固有问题。

### 5.2 ImageNet-1K Results (Table 2)

最值得关注的是 BHViT-Small† 达到 70.1%，相比：
- ReActNet (MobileNet-based): 69.4%（基本持平但 model 更小）
- Si-BiViT (Swin-based): 63.8%（领先 6.3%）
- BiViT (Swin-based): 58.6%（领先 11.5%）
- BinaryViT: 67.7%（领先 2.4%）
- Bi-ViT (DeiT-based): 40.9%（领先 29.2%！）

**关键 insight**: Figure 1 中展示的 binary ViT 性能崩盘问题，BHViT 完全解决了。BHViT-Small† 不仅超过了所有 binary ViT 方法，还超过了基于 CNN 的 ReActNet-A。这证明了 hybrid 架构对 binarization 的 friendliness。

注意 † 标记表示 downsampling layer 保留 full precision。这是 binary network 中常见的妥协，因为 downsampling layer 对精度敏感且计算量占比小。

### 5.3 Ablation Study (Table 5)

| Shift | MSGDC | MSMHA | QD | RL | FDL | Top-1 (%) |
|---|---|---|---|---|---|---|
| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 95.0 |
| ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | 92.1 |
| ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | 90.7 |
| ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | 88.9 |
| ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | 86.7 |
| ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | 85.6 |
| ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | 83.2 |

**每个模块的贡献**：
- FDL (Full-precision Downsampling Layer): +2.4%（83.2 → 85.6）
- RL (Regularization Loss): +1.1%（85.6 → 86.7）
- QD (Quantization Decomposition): +2.2%（86.7 → 88.9）
- MSMHA: +1.8%（88.9 → 90.7）
- MSGDC: +1.4%（90.7 → 92.1）
- Shift: +2.9%（92.1 → 95.0）

**Shift operation 贡献最大（+2.9%）**，这有点反直觉——一个 zero-FLOP 的操作竟然有这么大影响。这说明 binary MLP 的 bottleneck 不是计算能力，而是信息流。Shift operation 引入的 spatial context 几乎"免费"地补偿了 binary linear layer 的表达力损失。

### 5.4 Weight Distribution Visualization (Figure 7, 8)

Figure 7 显示加 RL 后，latent weight 的分布明显从集中在 0 附近变成双峰（±1 附近）。Figure 8 显示加 RL 后，weight flip 的次数显著减少，证明 weight oscillation 被有效抑制。

这与 Observation 3 的分析完全吻合：Adam 在训练后期会把 weight 锁在 0 附近震荡，L1 regularization 强制 weight 收敛到 ±1。

### 5.5 Latency on Edge Device (Supplementary Table 3)

| Network | W/A | Latency (ms) |
|---|---|---|
| Si-BiViT | 32/32 | 1029 |
| BHViT | 32/32 | 612 |
| Si-BiViT | 1/1 | 863 |
| BHViT | 1/1 | 157 |

**关键数据**: BHViT 的 binary 版本在 ARM Cortex-A76 CPU 上 latency 仅 157ms，比 full-precision 快 4 倍，比 Si-BiViT binary 快 5.5 倍。

作者坦诚提到目前部署还不够理想，因为 ViT-specific 的算子没有专门优化。这暗示了 binary ViT 部署还处于早期阶段，需要工程层面的进一步优化。

## 6. 我的 Intuition 与联想

### 6.1 关于 Hybrid Architecture 的思考

这篇 paper 让我想到 MetaFormer (https://arxiv.org/abs/2111.11418) 的论点：ViT 的成功主要归功于架构范式，而不是 self-attention 本身。BHViT 的实验进一步验证了这一点——在 binary 设定下，CNN-style token mixer 在早期 stage 反而更优。

更深层的思考：**Binary network 改变了"什么算子好"的判断标准**。Full-precision network 中 attention 优于 conv 是因为 attention 的 dynamic weighting 能力。但 binary 后，attention 的 dynamic weighting 能力被严重压缩（只剩 ±1 或 0/1），conv 的 inductive bias 优势反而凸显。这是一个非常深刻的 insight：算子的优劣不是绝对的，取决于 quantization precision。

### 6.2 关于 Quantization Decomposition 的延伸

QD 的思路让我联想到:
1. **Residual quantization** (https://arxiv.org/abs/2202.04770): 用多个 quantizer 串联逼近 full-precision
2. **Multi-basis binarization** (XNOR-Net 后续): 用多个 binary basis 的线性组合近似 full-precision
3. **Ternary weight networks** (https://arxiv.org/abs/1605.04711): 用 {-1, 0, +1} 三值化，介于 binary 和 full-precision 之间

QD 的巧妙之处在于它针对 attention matrix 的特殊性（softmax 后值在 [0,1] 之间）做了定制化设计。普通的 binary basis 是直接对 weight 做 sign，而 QD 是对 attention 做 threshold decomposition，这相当于把 attention 的 quantization 转化为多个 binary mask 的 OR 操作。

### 6.3 关于 Weight Oscillation 的根本性思考

Observation 3 的分析让我想到一个更深层的问题：**Adam 是为 full-precision network 设计的，它的 second-order momentum 假设 gradient 是连续的、有意义的。但 binary network 中，latent weight 在 0 附近的行为完全不同——它是一个 decision boundary，weight 在 0 附近震荡意味着 sign 频繁 flip，这本质上是离散优化问题。**

Adam 在离散优化场景下的失效，类似于用 gradient descent 优化组合优化问题。可能的方向：
1. 用 Straight-Through Estimator 之外的 gradient approximation
2. 用 Gumbel-Softmax 类的技巧
3. 用专门的 discrete optimizer（比如 Bayesian optimization 的某些变种）

作者用 L1 regularization 是一种简单但有效的工程解法，但理论上还有探索空间。

### 6.4 与 LLM Binary 时代的关联

Paper 在 Introduction 中提到 LLM 和 VLM 都用 transformer 架构，binary ViT 的研究有 practical significance。这是一个很有前瞻性的观点。

实际上，binary LLM 已经成为一个研究方向：
- BitNet (https://arxiv.org/abs/2310.11453): 1-bit LLM
- BitNet b1.58 (https://arxiv.org/abs/2402.17764): ternary LLM

BHViT 中的很多 insight（residual connection 的重要性、Adam 与 weight oscillation 的矛盾、attention binary 的特殊处理）很可能直接 transferable 到 LLM binary 场景。特别是 Observation 1 关于 token 数量的分析，对 LLM 的 long-context 场景很有启发——如果未来要做 binary LLM 处理 long context，token 数量爆炸会让 binary attention 完全失效。

### 6.5 Architecture 与 Optimization 的耦合

这篇 paper 最让我欣赏的地方是它把 architecture design 和 optimization 紧密耦合。三个 observation 分别对应：
- Observation 1 → 架构选择
- Observation 2 → 架构选择
- Observation 3 → optimization 策略

这种"问题驱动的设计"比"盲目堆模块"高明得多。每个设计决策都有数学或经验性的 justification，这让 paper 的说服力很强。

### 6.6 对 Binary Network 未来方向的预测

基于这篇 paper 的 insight，我预测 binary network 未来几个方向会很重要：

1. **Hardware-aware binary design**: 论文承认部署还不够理想。未来需要 co-design 算子和硬件。
2. **Dynamic precision allocation**: 不同 layer、不同 token 对 quantization 的 sensitivity 不同，dynamic allocation 能进一步压缩。
3. **Training-aware binary**: 当前方法多为 train-then-deploy，未来可能需要 fully differentiable 的 binary training pipeline。
4. **Binary + Sparse**: Binary 和 sparsity 是两种正交的压缩维度，结合可能产生新的 SOTA。

## 7. 总结性 Intuition

如果用一句话总结这篇 paper 的精髓：**Binary network 不是简单地把 full-precision network 的 weight 截断，而是需要重新思考架构和优化。** BHViT 通过三个 observation 系统性地揭示了 binary ViT 失败的根本原因，并给出了对应的解决方案：

- Token 数量问题 → hybrid CNN/Attention 架构
- Gradient vanishing 问题 → layer-by-layer residual connection
- Weight oscillation 问题 → scheduled L1 regularization

这种问题驱动的研究方法论值得学习。论文不是先有 trick 再找问题，而是先理解问题再设计 trick。这种 research taste 在 binary network 这个已经高度竞争的领域中尤其珍贵。

最后想提一个 paper 没有深入讨论但我觉得重要的问题：**Binary network 的 generalization 能力**。Paper 在 CIFAR-10、ImageNet、ADE20K、RS-LVF 上都做了实验，但都是 classification/segmentation 任务。Binary network 在 detection、generation、reinforcement learning 等任务上的表现如何？这是未来需要验证的方向。如果 binary network 只能在 classification 上 work，它的 practical impact 会受限；如果能在更广泛任务上 work，那 binary AI 的时代可能真的不远了。

参考链接汇总：
- BHViT 代码: https://github.com/IMRL/BHViT
- ReActNet: https://arxiv.org/abs/2003.03488
- Bi-Real Net: https://arxiv.org/abs/1808.00278
- DeiT: https://arxiv.org/abs/2012.12877
- Adam: https://arxiv.org/abs/1412.6980
- XNOR-Net: https://arxiv.org/abs/1603.05279
- BiViT: https://arxiv.org/abs/2211.07091
- MetaFormer: https://arxiv.org/abs/2111.11418
- BitNet: https://arxiv.org/abs/2310.11453
- BitNet b1.58: https://arxiv.org/abs/2402.17764
- Oscillation in QAT: https://arxiv.org/abs/2203.11086
- TSM: https://arxiv.org/abs/1811.08383
- MobileViT: https://arxiv.org/abs/2110.02178
- Swin Transformer: https://arxiv.org/abs/2103.14030
- Ternary Weight Networks: https://arxiv.org/abs/1605.04711
- Residual Quantization: https://arxiv.org/abs/2202.04770
