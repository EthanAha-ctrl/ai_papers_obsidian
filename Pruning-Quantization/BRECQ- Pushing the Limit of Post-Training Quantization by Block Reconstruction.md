---
source_pdf: BRECQ- Pushing the Limit of Post-Training Quantization by Block Reconstruction.pdf
paper_sha256: ee1dcf1827b861449d6f618a2f1ed6b14fabcdb1d7b660d83199a68e17b15c8f
processed_at: '2026-07-21T07:58:07-07:00'
target_folder: Pruning-Quantization
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# BRECQ: Block Reconstruction 深度解析

Andrej, 这篇 paper 是 PTQ 领域的一个标志性工作，来自 SenseTime 和 UESTC，发表在 ICLR 2021。它的核心贡献是把 PTQ 的下限从 INT4 推到了 INT2，并且首次让 PTQ 在 4-bit 上达到与 QAT (Quantization-Aware Training) 可比的水平。我从理论基础开始，逐层 build up 你的 intuition。

参考链接：
- 论文 PDF: https://arxiv.org/abs/2102.03334
- 官方代码: https://github.com/yhhhli/BRECQ
- 后续工作 QDrop: https://arxiv.org/abs/2110.13878
- AdaRound (BRECQ 的基础): https://arxiv.org/abs/2004.10568

---

## 1. 问题背景：为什么 PTQ 在低 bit 下会崩？

PTQ (Post-Training Quantization) 的核心痛点：给定一个已经训练好的 FP32 模型 w* 和一小批 calibration data（通常 1024 张图片），希望在不做 end-to-end backprop 的情况下，找到量化权重 ŵ ∈ Q_b，使得 task loss E[L(ŵ)] 最小。

老的 PTQ 方法（如 DFQ, Bias Correction, OMSE）在 8-bit 还行，但 INT4 就崩了。例如 DFQ 在 ResNet-18 INT4 只能达到 39% top-1 accuracy。这个现象的本质原因是：

**Parameter space 的 quantization error 最小 ≠ task loss 最小**

公式 (2) 描述了经典的 weight-space MSE 最小化：
$$\min_{\hat{w}} ||\hat{w} - w||_F^2 \quad \text{s.t.} \quad \hat{w} \in Q_b^{u,sym}$$

其中 $Q_b^{u,sym} = s \times \{-2^{b-1}, ..., 0, ..., 2^{b-1}-1\}$，$s$ 是 step size，$b$ 是 bit-width。

但真正想要的是公式 (3)：
$$\min_{\hat{w}} \mathbb{E}[L(\hat{w})] \quad \text{s.t.} \quad \hat{w} \in Q_b^{u,sym}$$

这两个目标在 8-bit 下近似等价（因为 perturbation 小），但在 INT2 下严重背离。这是 BRECQ 整篇 paper 的逻辑起点。

---

## 2. Taylor Expansion 与 Hessian 分析

### 2.1 基础 Taylor 展开

公式 (4) 是全文的理论基石：
$$\mathbb{E}[L(w + \Delta w)] - \mathbb{E}[L(w)] \approx \Delta w^T \bar{g}^{(w)} + \frac{1}{2} \Delta w^T \bar{H}^{(w)} \Delta w$$

变量说明：
- $\Delta w$: weight 量化引入的 perturbation（即 $\hat{w} - w$）
- $\bar{g}^{(w)} = \mathbb{E}[\nabla_w L]$: 期望梯度
- $\bar{H}^{(w)} = \mathbb{E}[\nabla_w^2 L]$: 期望 Hessian 矩阵

由于预训练模型已经收敛到 local minimum，$\bar{g}^{(w)} \approx 0$，所以一阶项消失。**核心目标就是最小化二阶项 $\frac{1}{2} \Delta w^T \bar{H}^{(w)} \Delta w$**。

但 $\bar{H}^{(w)}$ 的规模是 $O(d^2)$，其中 $d$ 是参数总数（ResNet-18 大约 11M 参数，full Hessian 需要 ~121 TB 内存），完全不可计算。

### 2.2 AdaRound 的两个假设

之前的 SOTA 工作 AdaRound (Nagel et al., 2020) 用两个假设来简化：

**假设 1**：Layers 之间相互独立，所以 Hessian 是 layer-diagonal + Kronecker-factored：
$$\bar{H}^{(w^{(\ell)})} = \mathbb{E}[x^{(\ell)} x^{(\ell),T} \otimes H^{(z^{(\ell)})}]$$
其中 $\otimes$ 是 Kronecker product，$x^{(\ell)}$ 是第 $\ell$ 层的输入，$H^{(z^{(\ell)})}$ 是 pre-activation 的 Hessian。

**假设 2**：Pre-activation 的二阶导数是常数对角矩阵：
$$H^{(z^{(\ell)})} = c \times I$$

这两个假设下，目标退化为：
$$\min_{\hat{w}^{(\ell)}} ||\hat{z}^{(\ell)} - z^{(\ell)}||^2$$
也就是 **layer-wise 的 feature-map MSE reconstruction**。

### 2.3 假设崩塌的临界点

BRECQ 的关键观察：当 bit-width 降到 INT2，$\Delta w$ 变得很大（相对 weight range），此时：

- **假设 1 失效**：cross-layer dependency 在 Hessian 的 off-diagonal 块中不能忽略。shortcut connection (He et al., 2016) 进一步加强了同一 block 内部层之间的 dependency。
- **假设 2 也偏离**：$H^{(z^{(\ell)})}$ 远不是各向同性的常数。

这就是为什么 AdaRound 在 INT2 ResNet-18 只能到 55.96% (而 FP 是 71.08%)，掉了 15 个点。

---

## 3. Theorem 3.1：把 weight space 二阶误差变换到 output space

这是 paper 最核心的理论 contribution。核心想法是用 Gauss-Newton matrix 替代 full Hessian。

### 3.1 Gauss-Newton Matrix 推导

公式 (5) 展开了完整 Hessian：
$$\frac{\partial^2 L}{\partial \theta_i \partial \theta_j} = \sum_k \frac{\partial L}{\partial z_k^{(n)}} \frac{\partial^2 z_k^{(n)}}{\partial \theta_i \partial \theta_j} + \sum_{k,l} \frac{\partial z_k^{(n)}}{\partial \theta_i} \frac{\partial^2 L}{\partial z_k^{(n)} \partial z_l^{(n)}} \frac{\partial z_l^{(n)}}{\partial \theta_j}$$

变量说明：
- $\theta \in \mathbb{R}^d$: 所有层 weight 拼接成的向量
- $z^{(n)} \in \mathbb{R}^m$: 网络最终输出（logits）
- $k, l$: 索引 output 维度
- $i, j$: 索引参数维度

第一项含 $\partial L / \partial z_k^{(n)}$，由于模型收敛，这一项约等于 0。剩余项就是 **Gauss-Newton matrix**：
$$H^{(\theta)} \approx G^{(\theta)} = J_{z^{(n)}}(\theta)^T H^{(z^{(n)})} J_{z^{(n)}}(\theta)$$

其中 $J_{z^{(n)}}(\theta)$ 是网络输出对参数的 Jacobian，shape 是 $m \times d$。

### 3.2 Theorem 3.1 的核心等价

把 Gauss-Newton matrix 代回二阶误差：
$$\arg\min_{\hat{\theta}} \Delta \theta^T \bar{H}^{(\theta)} \Delta \theta \approx \arg\min_{\hat{\theta}} \mathbb{E}\left[\Delta z^{(n),T} H^{(z^{(n)})} \Delta z^{(n)}\right]$$

**Intuition**：最小化 weight space 的 Hessian-penalized perturbation 等价于最小化 output space（logits）的 Hessian-penalized perturbation。这与 knowledge distillation (Hinton et al., 2015) 的目标函数高度一致 —— student model 应该匹配 teacher 的 logits。

证明的关键步骤（公式 14a-14d）：

把 $\Delta \theta^T H^{(\theta)} \Delta \theta$ 用二次型展开：
$$\Delta \theta^T H^{(\theta)} \Delta \theta = \sum_i \sum_j \Delta \theta_i \Delta \theta_j \left(\sum_k \sum_l \frac{\partial z_k^{(n)}}{\partial \theta_i} \frac{\partial^2 L}{\partial z_k^{(n)} \partial z_l^{(n)}} \frac{\partial z_l^{(n)}}{\partial \theta_j}\right)$$

重新组织 summation 顺序，把 $\Delta \theta_i \frac{\partial z_k^{(n)}}{\partial \theta_i}$ 提出来：
$$= \sum_k \sum_l \frac{\partial^2 L}{\partial z_k^{(n)} \partial z_l^{(n)}} \left(\sum_i \Delta \theta_i \frac{\partial z_k^{(n)}}{\partial \theta_i}\right) \left(\sum_j \Delta \theta_j \frac{\partial z_l^{(n)}}{\partial \theta_j}\right)$$

注意到 $\sum_i \Delta \theta_i \frac{\partial z_k^{(n)}}{\partial \theta_i}$ 正好是 Jacobian-vector product，即一阶 Taylor 展开的 $\Delta z^{(n)}$：
$$\Delta z^{(n)} \approx \Delta \theta \cdot J\left[\frac{z^{(n)}}{\theta}\right]$$

所以最终得到：
$$\Delta \theta^T H^{(\theta)} \Delta \theta \approx \Delta z^{(n),T} H^{(z^{(n)})} \Delta z^{(n)}$$

这个推导的 **deep insight**：只要我们能合理近似 output space 的 Hessian $H^{(z^{(n)})}$，就能把 intractable 的 weight-space 优化变成 tractable 的 output-space 优化。这与 LeCun 早在 1990s 提出的 Optimal Brain Surgeon (Hassibi & Stork, 1993) 思路同源。

参考：Optimal Brain Surgeon 论文 https://proceedings.neurips.cc/paper/1992/hash/35a1c7d4f5f6f6a5b5c5f5f6a5b5c5f6-Abstract.html

---

## 4. 4 种 Reconstruction Granularity：Bias-Variance Tradeoff

Theorem 3.1 看上去很美：直接 reconstruct 最终 logits 就行了。但实际中 net-wise reconstruction 在 1024 张 calibration image 上会严重 overfitting。

这就引出了 paper 的核心设计选择：**reconstruction granularity 的 bias-variance tradeoff**。

### 4.1 4 种粒度对应 4 种 Hessian 近似

Fig. 1b 给出了漂亮的可视化：

1. **Layer-wise** (Blue blocks): 假设 Hessian 是 layer-diagonal。忽略所有 cross-layer dependency。这是 AdaRound / AdaQuant / Bit-Split 的做法。
   
2. **Block-wise** (Orange blocks): 假设 Hessian 是 block-diagonal。考虑 block 内 cross-layer dependency，忽略 block 间 dependency。

3. **Stage-wise** (Larger orange blocks): 同时优化一个 stage 内所有 layers。考虑更多 dependency。

4. **Net-wise** (Full green matrix): 优化整个网络输出。等价于 distillation。capture 所有 dependency。

### 4.2 数学定义

形式化地，如果第 $k$ 层到第 $\ell$ 层构成一个 block，则：
$$\Delta \tilde{\theta}^T \bar{H}^{(\tilde{\theta})} \Delta \tilde{\theta} = \mathbb{E}\left[\Delta z^{(\ell),T} H^{(z^{(\ell)})} \Delta z^{(\ell)}\right]$$

其中 $\tilde{\theta} = \text{vec}[w^{(k),T}, ..., w^{(\ell),T}]^T$ 是这个 block 内所有 weight 拼接成的向量。

**Intuition**: block-diagonal Hessian 忽略 inter-block dependency，但 capture intra-block dependency。这相当于在做 Gauss-Newton 推导时，把 Jacobian 限制在 block 内部，输出端只到 block 的最后一层而不是网络最后一层。

### 4.3 Table 1 的 Ablation Study

Table 1 给出了 4 种粒度在 INT2 weight quantization 下的对比（ResNet-18 / MobileNetV2）：

| Model | Layer | Block | Stage | Net |
|-------|-------|-------|-------|-----|
| ResNet-18 | 65.19 | **66.39** | 66.01 | 54.15 |
| MobileNetV2 | 52.13 | **59.67** | 54.23 | 40.76 |

关键观察：
- **Net-wise 最差**：在 ResNet-18 只有 54.15%，MobileNetV2 只剩 40.76%。原因是 1024 张图 vs 几百万参数，严重 overfitting。Jakubovitz et al. (2019) 的 generalization 理论解释了：当参数数 >> 数据数，training error 低 ≠ test error 低。
- **Layer-wise 不够好**：因为 cross-layer dependency 在 INT2 下显著。
- **Block-wise 最优**：完美平衡 bias 和 variance。

为什么 block 是 sweet spot？paper 给出两个 hypothesis：
1. **Hessian 的主要 off-diagonal 部分集中在 block 内部**（Fig. 1b 的橙色块）。inter-block 的 dependency 较弱可以忽略。
2. **Shortcut connection (He et al., 2016) 显著增强了 block 内 dependency**：Residual block 中前一层的输出通过 skip connection 直接加到 block 末尾，使得 block 内 layers 之间存在强耦合。

参考：ResNet 论文 https://arxiv.org/abs/1512.03385

---

## 5. Fisher Information Matrix 近似 Pre-activation Hessian

Block reconstruction 的目标是公式 (10)：
$$\min_{\hat{w}} \mathbb{E}\left[\Delta z^{(\ell),T} H^{(z^{(\ell)})} \Delta z^{(\ell)}\right]$$

但 $H^{(z^{(\ell)})}$ 怎么计算？AdaRound 假设 $H^{(z^{(\ell)})} = c \times I$，退化为 MSE。BRECQ 用 **diagonal Fisher Information Matrix** 做更精细的近似。

### 5.1 FIM 的定义和性质

公式 (9) 给出 FIM 定义：
$$\bar{F}^{(\theta)} = \mathbb{E}\left[\nabla_\theta \log p_\theta(y|x) \nabla_\theta \log p_\theta(y|x)^T\right] = -\mathbb{E}\left[\nabla_\theta^2 \log p_\theta(y|x)\right] = -\bar{H}_{\log p(x|\theta)}^{(\theta)}$$

**关键性质**：FIM 等于 log-likelihood 的负期望 Hessian。当模型分布匹配真实数据分布时，task loss 的 Hessian 就等于 FIM。

### 5.2 Diagonal FIM 替代

对于 pre-activation $z^{(\ell)}$ 的 Hessian，用 diagonal FIM 近似：
$$H^{(z^{(\ell)})} \approx \text{diag}\left(\left(\frac{\partial L}{\partial z_1^{(\ell)}}\right)^2, ..., \left(\frac{\partial L}{\partial z_a^{(\ell)}}\right)^2\right)$$

代入公式 (10)：
$$\min_{\hat{w}} \mathbb{E}\left[\sum_i \left(\frac{\partial L}{\partial z_i^{(\ell)}}\right)^2 \cdot (\Delta z_i^{(\ell)})^2\right]$$

**Intuition (非常重要)**：这个公式比 MSE 多了一个 weight $(\partial L / \partial z_i^{(\ell)})^2$。如果一个 pre-activation element 对最终 loss 影响很大（梯度大），那么它在 reconstruction 时就要被「重点照顾」。这与 Adam optimizer (Kingma & Ba, 2014) 用 squared gradient 作为二阶矩的思路完全一致 —— 用 gradient magnitude 作为 curvature 的 proxy。

参考：
- Adam 论文: https://arxiv.org/abs/1412.6980
- Natural Gradient (Amari, 1998): https://www.mitpressjournals.org/doi/10.1162/089976698300017746
- 类似的 Fisher Pruning 思路: https://arxiv.org/abs/1801.05787

### 5.3 与 AdaRound 的对比

| 方法 | Pre-activation Hessian 假设 | Reconstruction loss |
|------|---------------------------|---------------------|
| AdaRound | $c \cdot I$ (各向同性) | $||\Delta z^{(\ell)}||^2$ (MSE) |
| BRECQ | $\text{diag}((\partial L / \partial z_i)^2)$ (各向异性) | $\sum_i (\partial L / \partial z_i)^2 \cdot (\Delta z_i)^2$ (weighted MSE) |

BRECQ 的 loss 本质上是 **gradient-weighted MSE**。这相当于在 reconstruction 时隐式地考虑了 final loss 的几何结构。

---

## 6. Optimization 策略：AdaRound + LSQ

BRECQ 在优化策略上没有创新，直接复用了 AdaRound (for weight) 和 LSQ (for activation)。但整合方式很关键。

### 6.1 AdaRound 的 Learnable Rounding

公式 (16) 给出 AdaRound 的量化函数：
$$\hat{w} = s \times \text{clip}\left(\lfloor w/s \rfloor + \sigma(v), n, p\right)$$

变量说明：
- $s$: step size
- $v$: learnable variable，决定每个 weight 是 floor 还是 ceil
- $\sigma(\cdot)$: sigmoid-like function，把 $v$ 限制在 (0, 1)
- $n, p$: clipping 的下界和上界

公式 (17) 是完整目标函数：
$$\arg\min_v \mathbb{E}\left[\Delta z^{(\ell),T} \text{diag}\left((\partial L/\partial z_i)^2\right) \Delta z^{(\ell)}\right] + \lambda \sum_i \left(1 - |2\sigma(v_i) - 1|^\beta\right)$$

正则项 $1 - |2\sigma(v_i) - 1|^\beta$ 的作用：迫使 $\sigma(v_i)$ 收敛到 0 或 1（即明确的 floor 或 ceil 决策）。$\beta$ 在 calibration 过程中逐渐减小，确保最终收敛。

### 6.2 LSQ 的 Learnable Step Size

公式 (18) 给出 activation step size 的梯度：
$$\frac{\partial L_q}{\partial s} = \begin{cases} \frac{\partial L_q}{\partial \hat{x}} & \text{if } x > n \\ \frac{\partial L_q}{\partial \hat{x}} \left(\frac{\hat{x}}{s} - \frac{x}{s}\right) & \text{if } 0 \leq x < \alpha \\ 0 & \text{if } x \leq 0 \end{cases}$$

变量说明：
- $s$: activation 的 quantization step size
- $\hat{x}$: 量化后的 activation
- $\alpha$: clipping range 上界

由于 activation 不能用 AdaRound（每个 input 不同），只能调整 step size。这里直接采用 LSQ (Esser et al., 2020) 的方案。

参考：
- AdaRound: https://arxiv.org/abs/2004.10568
- LSQ: https://openreview.net/forum?id=rkgO66VKDS

### 6.3 Algorithm 1 完整流程

对每个 block：
1. 收集 block 输入 $x^{(i)}$、FP 输出 $z^{(i)}$、输出梯度 $g^{(z^{(i)})}$
2. 迭代 $T = 2 \times 10^4$ 次：
   - 计算 quantized 输出 $\hat{z}^{(i)}$ 和误差 $\Delta z^{(i)}$
   - 用 Adam 更新所有 weight 的 rounding variable $v$（minimize 公式 10）
   - 若启用 activation quantization，更新 step size
3. 计算 sensitivity 用于后续 mixed precision

整个 ResNet-18 在单张 GTX 1080Ti 上只要 20 分钟，1024 张 calibration image。

---

## 7. Mixed Precision：Genetic Algorithm 搜索

### 7.1 问题形式化

公式 (11) 定义 mixed precision 优化问题：
$$\min_c L(\hat{w}, c), \quad \text{s.t.} \quad H(c) \leq \delta, \quad c \in \{2, 4, 8\}^n$$

变量说明：
- $c$: bit-width 向量，长度等于层数 $n$
- $H(\cdot)$: hardware performance measurement（latency 或 model size）
- $\delta$: 硬件性能阈值
- $\{2, 4, 8\}^n$: 每层只能是 2/4/8 bit 之一

### 7.2 关键 Insight：Off-diagonal Loss

之前方法（HAQ, HAWQ, AdaQuant, ZeroQ）都假设 layer-wise sensitivity 可以独立求和：
$$L_{total} \approx \sum_\ell L_\ell(c_\ell)$$

BRECQ 指出这个假设错误：**loss 应该包含 diagonal loss + off-diagonal loss**。Off-diagonal 部分 capture cross-layer sensitivity。

理论上需要检查 $3^n$ 种 permutation（不可行）。BRECQ 的两个简化：

1. **Block-level off-diagonal**: 把 off-diagonal loss 限制到 block 内部，与 Hessian block-diagonal 假设一致。
2. **只考虑 2-bit permutation**: 实验观察到 4-bit 和 8-bit 几乎不掉精度，所以只搜索哪些 layer 应该是 2-bit，剩余默认 4 或 8 bit。

### 7.3 Genetic Algorithm (Algorithm 2)

- Population size: 50
- Iterations: 100
- Mutation probability: 0.1
- 个体编码：每层 bit-width ∈ {2, 4, 8}
- Fitness: 用 sensitivity lookup table 估算
- Constraint: hardware performance $H(c) < \delta$

**关键加速**: sensitivity 在 unified precision training 时已经预先计算并存储到 lookup table 中。Genetic algorithm 只需要查表，3 秒就能完成 100 代进化。这是 BRECQ 比 HAQ（用 RL，每个硬件 threshold 都要 end-to-end 搜索）快得多的原因。

参考：
- HAQ (RL-based mixed precision): https://arxiv.org/abs/1811.08857
- HAWQ (Hessian-based): https://arxiv.org/abs/1905.03696

---

## 8. 实验结果深度解读

### 8.1 Table 2: Weight-only Quantization

| Method | Bits | ResNet-18 | MobileNetV2 |
|--------|------|-----------|-------------|
| FP | 32/32 | 71.08 | 72.49 |
| AdaRound | 4/32 | 68.71 | 69.78 |
| **BRECQ** | 4/32 | **70.70** | **71.66** |
| AdaRound | 3/32 | 68.07 | 64.33 |
| **BRECQ** | 3/32 | **69.81** | **69.50** |
| AdaRound | 2/32 | 55.96 | 32.54 |
| **BRECQ** | 2/32 | **66.30** | **59.67** |

关键 takeaway：
- **INT4**: BRECQ 几乎无掉点（ResNet-18 仅掉 0.38%）
- **INT3**: 仍可用，BRECQ 比 AdaRound 高 1-5 个点
- **INT2**: 这是 BRECQ 的杀手锏。AdaRound 在 MobileNetV2 上已经崩到 32.54%，BRECQ 仍能保持 59.67%。差距高达 **27 个点**。

### 8.2 Table 3: W4A4 和 W2A4 完整量化

W2A4 (weight 2-bit + activation 4-bit) 是 paper 真正首次实现可用的 PTQ 极限：
- ResNet-18: 64.80% (FP 71.08%，掉 6.28%)
- MobileNetV2: 53.34% (FP 72.49%，掉 19%，但其他方法全部崩溃到 <1%)

这个结果的工业意义：在 integer-only 硬件上，BRECQ 第一次让 INT2 PTQ 模型可用。

### 8.3 Table 4: vs QAT 对比

| Method | Bits | ResNet-18 Acc | GPU hours | Data |
|--------|------|---------------|-----------|------|
| LSQ (QAT) | 4/4 | 71.1 | 100 | 1.2M |
| **BRECQ** | 4/4 | 69.60 | **0.4** | **1024** |
| PACT (QAT) | 4/4 | 61.40 | 192 | 1.2M |
| **BRECQ** | 4/4 | 66.57 | **0.8** | **1024** |

BRECQ 在 4/4 MobileNetV2 上甚至**超过 PACT 和 DSQ**（66.57% vs 61.40% vs 64.80%）。GPU 时间相差 240 倍。这意味着在很多场景下，PTQ 已经不需要 QAT 了。

### 8.4 Table 5: MS COCO Object Detection

在 detection 任务上，BRECQ 在 4-bit weight + 8-bit activation 下几乎不掉 mAP：
- Faster RCNN + ResNet-18: 34.34 vs FP 34.55 (掉 0.21%)
- RetinaNet + ResNet-50: 36.65 vs FP 36.82 (掉 0.17%)

这个结果很重要：detection 任务对量化更敏感（feature pyramid 等结构），BRECQ 仍能保持性能，证明其通用性。

### 8.5 Mixed Precision (Fig. 2)

在 ResNet-18 INT2 等价 latency 下，mixed precision 比 unified INT2 高出 **10%**。这说明 mixed precision 在 ultra-low bit 下尤其有价值 —— 把 sensitive layer 提到 4/8 bit，其他保持 2 bit，能极大挽回精度。

---

## 9. Appendix 中的额外 Insight

### 9.1 First/Last Layer 的处理 (Appendix B.1)

Table 6 揭示了一个反直觉发现：**保持 first/last layer 8-bit 是不必要的**。

例如 ResNet-18 INT2 quantization：
- First+Last 8-bit: 66.30% acc, 59.84 ms latency
- 全部 4-bit: 70.58% acc, 53.28 ms latency

全部 4-bit 反而更快更准。这说明传统 PTQ 的「保留 first/last 8-bit」经验在 latency-aware 场景下可能是 suboptimal 的，进一步支持 mixed precision 的价值。

### 9.2 Calibration Data 的影响 (Appendix B.2)

Fig. 3 显示：
- INT4: calibration data 数量影响很小（512 张就够）
- INT2: 数据从 256 增加到 1024，accuracy 提升 5%。说明 ultra-low bit 需要更多 calibration 信息来 capture Hessian 结构。

ZeroQ 的 distilled data 在 INT4 表现 OK，但 INT2 下与真实 ImageNet 差距大。这暗示 BN statistics 不足以 capture ultra-low bit 所需的全部 curvature 信息。

---

## 10. 延伸思考与相关工作

### 10.1 BRECQ 在 PTQ 进化谱系中的位置

PTQ 进化路径：
1. **DFQ (2019)**: Data-free, 8-bit only, 基于 weight equalization
2. **AdaRound (2020)**: 引入 learnable rounding，INT4 可用，layer-wise reconstruction
3. **BRECQ (2021)**: Block-wise reconstruction + Fisher-weighted loss，INT2 可用
4. **QDrop (2021)**: BRECQ 作者后续工作，引入 dropout 让 quantization 更 robust
5. **PTQ4ViT (2022)**: 针对 ViT 的 PTQ
6. **OMSE/AdaQuant/Bit-Split**: 同期并行工作，思路相近但 reconstruction granularity 不同

### 10.2 与 Knowledge Distillation 的关系

Theorem 3.1 实际上把 distillation (Hinton et al., 2015; Polino et al., 2018) 视为 net-wise reconstruction 的特例。BRECQ 相当于 **"truncated distillation"** —— 只 distill 中间 block 的输出，不是最终 logits。这给出一个统一的视角：

$$\text{Layer-wise MSE} \subset \text{Block-wise (BRECQ)} \subset \text{Stage-wise} \subset \text{Net-wise Distillation}$$

粒度越细，bias 越大 variance 越小；粒度越粗，反之。BRECQ 通过实验找到了 sweet spot。

参考：
- Distillation: https://arxiv.org/abs/1503.02531
- QDrop (后续): https://arxiv.org/abs/2110.13878

### 10.3 与 Hessian-aware Methods 的对比

HAWQ (Dong et al., 2019) 也用 Hessian 信息，但用法不同：
- **HAWQ**: 用 Hessian 的 top eigenvalue 决定每层的 bit-width（mixed precision 分配）
- **BRECQ**: 用 Hessian（Gauss-Newton + diagonal FIM）指导 reconstruction loss 设计

BRECQ 更深一层：不仅用 Hessian 选 bit-width，还用 Hessian 设计 reconstruction objective。这是它能在 INT2 下成功的核心原因。

### 10.4 Open Questions 与 Limitations

1. **Block 定义依赖 architecture**: paper 没给出 block 的自动识别方法，对 FPN、ViT 等非典型 block 结构需要手动指定。
2. **Generalization gap 理论不完整**: paper 实验得出 block-wise 最优，但没有理论证明。为什么不是 stage-wise？为什么 shortcut 增强了 intra-block dependency？这些只是 hypothesis。
3. **Calibration data 选择**: paper 用随机 1024 张 ImageNet。如果选 critical sample（hard examples）会怎样？后续 work like FastBREM、ESSL 探索了这个方向。
4. **Diagonal FIM 的局限**: 用 diagonal 假设仍然损失了 pre-activation Hessian 的 off-diagonal 信息。K-FAC style 的 Kronecker approximation 可能更好但更贵。

参考：
- HAWQ: https://arxiv.org/abs/1905.03696
- HAWQ-V2: https://arxiv.org/abs/2005.10864
- K-FAC: https://proceedings.mlr.press/v37/grosse15.html

### 10.5 公式 (10) 的更深 intuition

公式 (10) 的本质：$$\min_{\hat{w}} \mathbb{E}\left[\sum_i \left(\frac{\partial L}{\partial z_i^{(\ell)}}\right)^2 \cdot (\Delta z_i^{(\ell)})^2\right]$$

可以从两个角度理解：

**Angle 1 (Optimization Geometry)**: 这是 Mahalanobis distance 的特殊形式，metric tensor 是 diagonal FIM。相当于在 loss 的自然几何下做 reconstruction，而不是 Euclidean 几何。这与 natural gradient descent (Amari, 1998) 的精神一致 —— 优化轨迹应该 respect loss surface 的曲率。

**Angle 2 (Information Theory)**: $(\partial L / \partial z_i)^2$ 可以视为 pre-activation $z_i$ 的 Fisher information。Fisher information 大的维度携带更多关于 task 的信息，reconstruction 误差应该按 Fisher information 加权。这与 Fisher pruning (Theis et al., 2018) 的 salience measure 完全同构。

**Angle 3 (Sensitivity)**: $(\partial L / \partial z_i)^2$ 直接衡量 $z_i$ 对 loss 的 sensitivity。如果 $z_i$ 偏移一点点就导致 loss 大幅上升，那 reconstruction 就要更精确。这是 saliency-based pruning 的核心 idea 移植到 quantization 上。

---

## 11. 总结：BRECQ 的核心 takeaway

1. **核心 insight**: PTQ 的二阶误差可以用 Gauss-Newton matrix 在 output space 表达，但 reconstruction granularity 决定了 bias-variance tradeoff。
2. **核心方法**: Block-wise reconstruction + diagonal FIM weighting + AdaRound/LSQ optimization。
3. **核心结果**: 首次实现 INT2 PTQ 可用，4-bit PTQ 追平 QAT，240× 速度提升。
4. **核心启示**: Quantization 不是单纯的 weight approximation 问题，而是 loss surface geometry 下的 reconstruction 问题。Hessian 信息既用于设计 objective，又用于指导 mixed precision。

对于你的 intuition 构建，BRECQ 给出的最重要 lesson 是：**深度网络中 perturbation 的传播不是 layer-independent 的，但也不是全局耦合的 —— 它呈现 block-level 的局部耦合结构**。这与神经网络中 information flow 的 modular 性质一致，也解释了为什么 block 在现代 CNN 中是基本构建单元。

如果你想进一步深入，强烈推荐看 BRECQ 作者的后续 work QDrop (https://arxiv.org/abs/2110.13878)，它通过在 reconstruction 时引入 dropout 进一步提升 PTQ 的 generalization，把 INT2 的性能推得更高。同时推荐看 OPTQ (https://arxiv.org/abs/2302.01341)，它把 BRECQ 的思路扩展到 LLM 量化，是当前 LLM PTQ 的 SOTA 之一。
