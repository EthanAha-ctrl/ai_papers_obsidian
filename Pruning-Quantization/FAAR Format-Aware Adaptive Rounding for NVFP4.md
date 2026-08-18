---
source_pdf: FAAR Format-Aware Adaptive Rounding for NVFP4.pdf
paper_sha256: 3b6ee24fa8ec3a2617b34e1a90c858237232add8208fe91233a44843e266ed40
processed_at: '2026-08-18T12:05:33-07:00'
target_folder: Pruning-Quantization
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 FAAR

## 1. 这篇 paper 在干一件什么事

NVIDIA 出了新硬件 Blackwell，原生支持一种叫 **NVFP4** 的 4-bit 浮点格式。理论上比 BF16 快 4-6 倍、省 4 倍显存，听起来美滋滋。但问题来了——把 LLM 权重量化到 NVFP4 之后，模型性能掉得厉害，传统方法（RTN、GPTQ、AdaRound）都救不回来。

这篇 paper 提出了 **FAAR**，一个专门为 NVFP4 设计的 "learnable rounding" 方法。简单说：**让模型自己学每个 weight 应该 round up 还是 round down，而不是死板地按距离 round 到最近的 node**。配合一个两阶段 fine-tuning pipeline，在 Llama3-1B 上把 PPL 从 14.28 拉到 12.60（BF16 是 11.98），只花 4 个 GPU hour。

NVIDIA NVFP4 官方介绍: https://developer.nvidia.com/blog/nvfp4-tensor-core
NVFP4 pretraining paper: https://arxiv.org/abs/2509.25149

---

## 2. NVFP4 为什么 tricky——直觉解释

先讲讲 NVFP4 这个 format 长什么样。

它是个浮点 4-bit（2 bit exponent + 1 bit mantissa），归一化后能表示的数值集合是：

$$\{0, \pm 0.5, \pm 1.0, \pm 1.5, \pm 2.0, \pm 3.0, \pm 4.0, \pm 6.0\}$$

你眯着眼看一下这个集合，应该立刻能感觉到一个事：**靠近 0 的地方密密麻麻，远离 0 的地方稀疏得要命**。

| 区间 | gap |
|---|---|
| 0 → 0.5 | 0.5 |
| 0.5 → 1.0 | 0.5 |
| 1.5 → 2.0 | 0.5 |
| 2.0 → 3.0 | 1.0 |
| 4.0 → 6.0 | **2.0** |

这就是浮点数的本质特性——指数位少意味着大数附近分辨率急剧变差。换来的是动态范围广，对 LLM 那种 heavy-tailed 的 weight 分布友好。

**问题就出在这个非均匀性上。**

传统量化（INT4）的 grid 是均匀的，所有相邻 node 之间 gap 一样大。这时候 "round 到最近 node"（也就是 RTN）是天然 MSE-optimal 的，因为距离最近 = error 最小。

但 NVFP4 的 grid 不均匀。一个 weight 如果恰好落在 4.0 到 6.0 之间，gap 是 2.0——你不管 round 到 4 还是 6，**单点误差最多能到 1.0**，是落在 0→0.5 区间 weight 的 4 倍。

更糟的是：这些大 magnitude weight 在 LLM 里恰恰是 "outlier"，对输出贡献特别大（参考 LLM.int8()、SmoothQuant 都在讲这个）。一个 outlier 的 rounding 决策做错，误差会通过 matmul 传递到下游 layer，越传越大。

LLM.int8(): https://arxiv.org/abs/2208.07339
SmoothQuant: https://arxiv.org/abs/2211.10438

---

## 3. RTN 为什么会崩——做个实验就懂了

paper 做了个非常 elegant 的小实验（Table 1）。

在 Llama3-1B 上，他们把所有 weight 的 rounding 决策随机扰动——每个 weight 有 50% 概率 round up、50% 概率 round down，跑 100 次随机采样。

结果：
- **RTN baseline**: PPL = 13.04
- **100 次随机采样里，有 13 次比 RTN 更好**
- 最好的那次，PPL 比 RTN 低 0.02

这件事意义重大。它告诉你：**RTN 不是最优的，存在比 RTN 更好的 rounding 配置，只是你不知道怎么找到它**。

为什么 RTN 不最优？因为最优 rounding 不是 local 决策——单看一个 weight，距离最近的 node 误差最小没错，但 LLM 的 loss 是经过 matmul、layer stack、softmax 一长串非线性变换后的 task loss，你优化单点 L2 误差不等于优化最终 PPL。这是个 combinatorial optimization 问题，组合数是 $2^N$（N 是 weight 数，Llama3-1B 大概 10 亿），brute force 不可能。

直觉上类比一下：这就像你在调音台，每个推子（weight）单独调到 "音量最准" 不等于整体合奏最好听。你需要的是全局优化，而不是局部贪心。

---

## 4. FAAR 的核心 trick——把离散变连续

既然 brute force 搜 $2^N$ 不现实，那就**松弛**。

对每个 weight $w$，它落在两个 NVFP4 node $w_{lower}$ 和 $w_{upper}$ 之间。原本 RTN 是根据距离选一个，FAAR 改成：**引入一个连续变量 $v \in [0, 1]$，用 $v$ 决定是 round up 还是 round down**。

具体公式：

$$w_q = \text{sign}(w) \cdot \Big[ w_{lower} + h_\beta(v) \cdot \underbrace{(w_{upper} - w_{lower})}_{\text{interval span } \Delta} \Big] \cdot (s_g \cdot s_{global})$$

其中 $h_\beta(v) = \sigma(\beta(v - 0.5))$ 是一个 sigmoid 函数（带温度 $\beta$）。

直觉解释：
- $v = 0$：$h = 0$，$w_q = w_{lower}$（round down）
- $v = 1$：$h = 1$，$w_q = w_{upper}$（round up）
- $v = 0.5$：$h = 0.5$，$w_q$ 在两个 node 中间（训练初期）

训练时 $v$ 是连续的，可以用梯度下降优化；训练结束做 hardening，$v \geq 0.5$ 就变 1，否则变 0，回到离散决策。

**但这里有个关键细节，是 FAAR 和 AdaRound 的真正差别**。

AdaRound 是为 uniform INT4 设计的，它的公式里 $\Delta$ 是常数（uniform grid），所以 $\partial w_q / \partial v$ 对所有 weight 是等比缩放的。

FAAR 显式把 $\Delta = w_{upper} - w_{lower}$ 放进公式里。这意味着：
- 落在 0→0.5 区间的 weight，$\Delta = 0.5$，梯度小
- 落在 4→6 区间的 weight，$\Delta = 2.0$，**梯度自动大 4 倍**

这是 paper 标题里 "Format-Aware" 的真正含义。它让 optimizer 感知到 grid 的非均匀性——大 magnitude weight 的 rounding 决策影响大，所以梯度自动给它更强的 corrective signal。**不需要手动设计 loss weighting，formula 本身就把非均匀性 baked in 了**。

AdaRound 原文: https://arxiv.org/abs/2004.10568

---

## 5. 训练怎么 setup——三个关键设计

### (a) Initialization

$v$ 怎么初始化？paper 给了个非常聪明的 init：

$$v_{init} = \frac{\text{weight 在 interval 里的相对位置}}{\text{interval 总长}}$$

也就是说，如果 weight 在 $w_{lower}$ 和 $w_{upper}$ 中间偏 $w_{upper}$ 70% 的位置，$v_{init} = 0.7$。

**好处**：训练起点 ≈ RTN 的决策。FAAR 不是从 random 开始学 rounding，是从 RTN 出发做微调。RTN 是 inductive bias，FAAR 在这基础上做 gradient-based refinement。这种 "warm start" 思路在 BRECQ 等 PTQ 工作里也很常见。

### (b) Sigmoid temperature annealing

$h_\beta(v) = \sigma(\beta(v - 0.5))$ 里的 $\beta$ 是个温度参数：
- 训练初期 $\beta$ 小：sigmoid 平缓，$h$ 在 0.5 附近徘徊，梯度流畅
- 训练后期 $\beta$ 大：sigmoid 陡峭，逼近 step function，$h$ 趋向 0 或 1

这和 Gumbel-Softmax 的 temperature annealing 一模一样的思路。好处是让训练有个 smooth landing 到 discrete decision，避免直接做 hard rounding 导致的 gradient 死亡。

Gumbel-Softmax: https://arxiv.org/abs/1611.01144

### (c) Rounding regularization

loss 里有个正则项：

$$\mathcal{L}_{round} = \frac{1}{N}\sum_i \big(1 - (2v_i - 1)^2\big)$$

这个公式看起来唬人，但行为简单：
- $v_i = 0$ 或 $1$：loss = 0（鼓励状态）
- $v_i = 0.5$：loss = 1（惩罚状态）

就是个 "把 $v$ 推向 binary" 的正则。配合最后的 hardening 步骤（$v \geq 0.5$ 变 1），保证训练结束时 $v$ 已经很接近 0 或 1，hardening 几乎无损。

---

## 6. 2FA 两阶段——为什么需要两阶段

FAAR 解决了 "单层 rounding 怎么做最优"，但还有个问题没解决：**layer 之间误差累积**。

你 layer 1 的 rounding 决策是基于 layer 1 的 input activation 优化的。但 layer 1 量化后，输出 activation 已经偏了，layer 2 看到的 input 不是你以为的 input 了。一层层下去，误差雪球越滚越大。

paper 用 2FA (Two-Stage Format Alignment) 解决这个问题：

**Stage 1: Layer-wise optimization**
- 每层独立优化自己的 rounding variable $\mathbf{V}$
- Loss 是 reconstruction MSE：$\|\mathbf{X}\mathbf{W} - \mathbf{X}_q\mathbf{W}_q(\mathbf{V})\|_F^2$
- 其他层 frozen，只动当前层
- 这是局部贪心，类似 BRECQ 的 block reconstruction

**Stage 2: Full-model alignment**
- 把所有层组装成完整 NVFP4 model
- 跑端到端 forward，对齐 BF16 teacher model
- Loss 三项：
  - KL divergence on logits（让量化 model 输出分布跟 BF16 一致）
  - MSE on last hidden state（让 hidden representation 几何对齐）
  - Rounding regularization（防止 stage 2 把 $v$ 推回 0.5）

Stage 2 的 KL+MSE 组合是知识蒸馏的标配。hidden state MSE 这条尤其关键——Table 4 显示量化 model 和 BF16 的 last hidden state cosine similarity 达到 99.02%（Qwen3-1.7B），这说明量化模型在 representation space 上几乎是 BF16 的等距变换，下游 task 性能自然好。

LLM-QAT 用的也是类似 KL on logits 思路: https://arxiv.org/abs/2305.12285

---

## 7. 实验上发生了什么

### 主结果（Table 3，WikiText-2 PPL）

| Method | Llama3-1B | Llama3-8B | Qwen3-1.7B |
|---|---|---|---|
| BF16 | 11.98 | 7.54 | 21.04 |
| RTN | 14.28 | 8.44 | 23.06 |
| GPTQ | 13.74 | 8.32 | 21.48 |
| GPTQ+4/6 | 13.66 | 8.29 | 22.68 |
| **FAAR+2FA** | **12.60** | **8.13** | **21.27** |

几个观察：

1. **1B model 上增益最大**（14.28 → 12.60，减 1.68）。8B 上只减 0.31。这是 scaling law 的直觉——小模型容量小，quantization error 占比大，optimization 收益高。大模型天然 robust，fix 的空间小。

2. **GPTQ 在 NVFP4 上表现平庸**。GPTQ 是为 uniform INT4 设计的 Hessian-based 方法，在 non-uniform floating grid 上它的 weight update 机制失灵——你 update 出来的 optimal weight 不一定落在 NVFP4 node 上。这印证了 paper 的核心 motivation：format-aware 才是正道。

3. **逼近 BF16**：Llama3-1B 上 FAAR 把 PPL 从 RTN 的 14.28 拉到 12.60，离 BF16 的 11.98 只差 0.62。这个 gap 在 1B 这种小模型上已经非常小了。

### 消融实验（Table 6）

| 阶段 | Llama3-1B |
|---|---|
| RTN | 14.28 |
| + FAAR | 13.01 |
| + FAAR + 2FA | 12.60 |

FAAR 单独贡献 -1.27 PPL，2FA 再贡献 -0.41。**FAAR 是主菜，2FA 是 dessert**。这符合直觉——FAAR 直接 attack rounding suboptimality 这个根本问题，2FA 只是 global 微调。

### 训练成本

Llama3-1B: 4 GPU hours on H200。这是 PTQ 级别的成本，不是 QAT 级别。对 edge deployment 友好。

---

## 8. 我对这篇 paper 的看法

### 优点

1. **Motivation 干净**：NVFP4 grid 非均匀 → RTN 失效 → 需要 learnable rounding，逻辑链条非常清晰。Table 1 的小实验尤其 elegant，三言两语就把 RTN suboptimality 这件事证死了。

2. **方法设计 elegant**：FAAR 的 formula 里直接 bake 进 $\Delta$，让梯度自动感知非均匀性。这个设计 minimal 且 principled，没有引入额外 hyperparameter 或者 complex loss term。

3. **工程价值高**：4 GPU hour 拿到接近 BF16 的 PPL，对实际部署非常有吸引力。

### 疑问和不足

1. **Activation 量化怎么处理没讲清楚**。W4A4 setting 需要量化 activation，但 paper 的 formula 主要描述 weight。Activation 也用 FAAR 吗？还是 RTN？这点模糊。

2. **Block size 没有 ablation**。NVFP4 的 block size 通常 16，但这个选择对 per-block scale $s_g$ 的 granularity 影响巨大。Outlier 集中的 block 会把 $s_g$ 拉大，导致 block 内 normal weight 被压扁。这个 tradeoff 没讨论。

3. **Per-block scale $s_g$ 怎么算的没说**。应该是 absmax-based 或者类似的。Four over Six（ref [23]）专门优化这个，FAAR 没动。可以联合优化 $s_g$ 和 $\mathbf{V}$。

4. **Stage 2 的 KL temperature $\tau$ 没给值**。distillation 里这个超参敏感。

5. **GPTQ baseline 偏弱**。GPTQ 本来就不是为 NVFP4 设计的，跑得差意料之中。更 fair 的对比应该是 MR-GPTQ（ref [22]，专门适配 microscaling format 的 GPTQ variant），paper 里有这个 baseline 但效果也一般，说明 PTQ 在 ultra-low precision 下确实需要 trainable approach。

---

## 9. 一些联想和扩展方向

### (a) FAAR 思路可以迁移到所有 microscaling format

NVFP4 是 NVIDIA 私有 variant，对应 OCP 标准是 MXFP4。其他 microscaling formats 包括 MXFP6、MXFP8、MXINT8 等。只要 grid 非均匀，FAAR 的 $\Delta$ injection 就有效。MXINT8 是 uniform 的，FAAR 退化为 AdaRound。

OCP MX Spec: https://www.opencompute.org/docs/2023/30545824-OCP-Microscaling-Formats-MX-v1.0.pdf

### (b) FAAR + Hessian 加权

Stage 1 的 MSE loss 可以改成 Hessian-weighted：

$$\mathcal{L} = \|(\mathbf{X}\mathbf{W} - \mathbf{X}_q\mathbf{W}_q) \mathbf{H}^{-1/2}\|_F^2$$

这样 sensitive row 的 error 被惩罚更重。GPTQ 的 Hessian insight 可以嫁接到 FAAR。

### (c) Mixed-precision FAAR

不是所有 weight 都需要 FAAR。可以先用 saliency score（Hessian trace 之类）筛 top-k sensitive weight，只对这些跑 FAAR，其他用 RTN。这样训练参数量大幅降低。

### (d) FAAR 用于 pretraining

NVIDIA 自己的 NVFP4 pretraining paper（ref [1]）用 NVFP4 做 pretraining。如果把 FAAR 的 differentiable rounding operator 用作 pretraining 的 quantization function，可能比 STE 收敛更好。STE 是经典 QAT 的 backward trick，但 gradient 估计 biased，FAAR 的 continuous relaxation 是 unbiased 的。

### (e) Activation 也用 FAAR

Activation 的 distribution 比 weight 更 dynamic、更 heavy-tailed，FAAR 在 activation 上的增益可能更大。但 activation 是 runtime 数据，learnable rounding 在 inference 时怎么 deploy？可能需要 input-conditional rounding predictor——一个小网络预测 $v$，而不是直接学 $v$ 本身。这是个开放问题。

### (f) 和 LoRA 的结合

把 FAAR 学到的 rounding pattern 当作 frozen base，再用 LoRA 在 NVFP4 backbone 上 fine-tune downstream task。这可能是 edge deployment 的最佳 stack——NVFP4 当 backbone 省显存，LoRA 做 task adaptation。

---

## 10. 一句话总结直觉

NVFP4 这种非均匀浮点 grid 把传统量化方法（RTN、AdaRound、GPTQ）的 uniform grid 假设打破了，最优 rounding 不再是 local 距离能决定的。FAAR 把每个 weight 的 "round up or down" 决策松弛成连续变量，通过公式里显式的 interval span 让梯度自动按 grid 非均匀性 scale，配合 sigmoid 退火和 binary 正则收敛到 hard decision，再通过两阶段 fine-tuning（layer-wise 局部优化 + full-model KL/MSE 对齐）缓解 inter-layer 误差累积。最终 Llama3-1B 在 4 GPU hour 内把 PPL 从 RTN 的 14.28 拉到 12.60，逼近 BF16 的 11.98。

核心 insight 就一句话：**当 grid 不均匀时，optimizer 看到的 loss landscape 必须显式 encode 这种不均匀性，否则它优化的方向和真实量化误差 geometry 是错位的**。

如果你想继续聊某个方向（比如 activation FAAR 怎么 deploy、Hessian 加权版 FAAR 的具体 form、或者 FAAR 和 QAT 的 trade-off 分析），随时说。

---

# FAAR: Format-Aware Adaptive Rounding for NVFP4 深度解析

Karpathy 你好，这篇 paper 切中了一个非常有意思的 corner case——NVIDIA 的 NVFP4 format 在 Blackwell 上是硬件原生支持的 ultra-low precision path，但 algorithmic 端的 optimization 一直比较粗糙。我从 motivation、format 分析、FAAR 公式、2FA pipeline、实验数据几个层面把这篇工作拆开讲，并尽量做 intuition building + 相关联想。

---

## 1. Paper 一句话定位

FAAR 把 NVFP4 的 rounding decision 从 "training-free heuristic" 提升为 "learnable continuous relaxation problem"，通过显式注入 non-uniform interval span $\Delta = w_{upper} - w_{lower}$ 让梯度对大-magnitude weight 给出更强 corrective signal；外加 2FA 两阶段 alignment 来缓解 inter-layer error accumulation。Llama3-1B 在 WikiText-2 上 PPL 从 RTN 的 14.28 压到 12.60，逼近 BF16 的 11.98，仅 4 GPU hours。

Paper link (arXiv 待发，作者单位是 Li Auto Inc.，可能是理想汽车 AI Lab):
- https://arxiv.org/ — 暂时未上线，只能从作者列表 {lihanglin, tianshuchang, linchen, zhaozhiyong1, zhankun}@lixiang.com 推断
- NVIDIA NVFP4 官方: https://developer.nvidia.com/blackwell-architecture
- NVIDIA NVFP4 pretraining paper (ref [1] arXiv 2509.25149): https://arxiv.org/abs/2509.25149

---

## 2. NVFP4 Format 的非均匀性：一切 motivation 的源头

### 2.1 NVFP4 numerical grid

NVFP4 基于 FP4 的 E2M1 表示 (2-bit exponent, 1-bit mantissa)，结合 block-wise scaling。它可表示的归一化 node 集合是：

$$\mathcal{N} = \{0.0,\ \pm 0.5,\ \pm 1.0,\ \pm 1.5,\ \pm 2.0,\ \pm 3.0,\ \pm 4.0,\ \pm 6.0\}$$

我把相邻 gap 列出来感受一下 spacing 的 non-uniformity：

| Interval (相邻 node) | Gap |
|---|---|
| 0.0 → 0.5 | 0.5 |
| 0.5 → 1.0 | 0.5 |
| 1.0 → 1.5 | 0.5 |
| 1.5 → 2.0 | 0.5 |
| 2.0 → 3.0 | **1.0** |
| 3.0 → 4.0 | **1.0** |
| 4.0 → 6.0 | **2.0** |

直觉上：靠近 0 的区域 dense，4.0→6.0 这段超 sparse。这是浮点 format 的固有特性——mantissa 位少 → 大数附近 resolution 急剧变差，但换来的是 wide dynamic range。如果用 uniform INT4，interval 是恒定的，但表示范围太窄，对 LLM 的 heavy-tailed weight distribution 不友好。

### 2.2 Two-level scaling mechanism

每个 weight $w$ 归一化时分两级：
1. **Per-block scale** $s_g$（block size 通常是 16 elements）：FP8 (E4M3) 精度，覆盖 local variations。
2. **Global scale** $s_{global}$：FP32 精度，是 "scale of scales"，确保所有 FP8 block scales 落在 E4M3 的可表示范围里。

归一化值：

$$\tilde{w} = \frac{w}{s_g \cdot s_{global}}$$

这样设计的好处：FP8 的动态范围有限 (E4M3 最大 ~448)，加上 global FP32 scale-of-scales 可以让 per-block scale 始终保持 FP8 可表示，又不会因 global scale 失精度。这种 hierarchical scaling 是 microscaling formats (MX) 的核心 idea，OCP MX spec 也采用类似设计：
- OCP MX Format Spec: https://www.opencompute.org/docs/2023/30545824-OCP-Microscaling-Formats-MX-v1.0.pdf

### 2.3 Non-uniformity 为什么 RTN 会失效

经典 RTN (Round-to-Nearest) 的 implicit assumption 是：**距离最近的 node 就是 MSE-optimal**。这个 assumption在 uniform grid 下成立——因为所有 interval 等距，所以 "closest node" 与 "min L2 distance" 等价。

但在 NVFP4 上：
- $w_{lower} \leq |\tilde{w}| \leq w_{upper}$
- 如果 $|\tilde{w} - w_{lower}| < |w_{upper} - |\tilde{w}||$，RTN 选 $w_{lower}$

但 paper Table 1 做了个非常 elegant 的实验，直接 hammer 这个 assumption：
- 在 Llama3-1B 上采样 100 个 stochastic rounding 候选
- 100 个里有 **13 个 PPL 比 RTN 更好**
- 最好的 stochastic config 比 RTN 好 0.02 PPL

这表明：**最优 rounding assignment 是一个 global combinatorial optimization 问题**，不是 local 距离就能决定的。原因在于 quantization error 通过 $Y = XW$ 矩阵乘法后，会通过 layers 累积放大，下游 loss 对 rounding 的依赖远比 local L2 distance 复杂。

这点让我想到 Lloyd-Max quantizer 在 scalar 量化理论中的最优性——它对分布做迭代优化，找到使 expected distortion 最小的 reconstruction level。但 LLM 场景下我们要 minimize 的不是 input-space distortion，而是 task loss $\mathcal{L}$，所以必须把 rounding 决策放进端到端优化里。

- AdaRound (Nagel et al. ICML 2020): https://arxiv.org/abs/2004.10568
- BRECQ (Li et al. ICLR 2021): https://arxiv.org/abs/2102.05426

---

## 3. FAAR Method 详解

### 3.1 Continuous relaxation 的核心公式

paper 的核心是 Eq. (2)，我把变量意义逐一展开：

$$w_q = \text{sign}(w) \cdot \Big[ w_{lower} + h_\beta(v) \cdot \underbrace{(w_{upper} - w_{lower})}_{\text{interval span } \Delta} \Big] \cdot (s_g \cdot s_{global})$$

变量与含义：
- $w \in \mathbb{R}$: 原始 BF16 weight element
- $\text{sign}(w) \in \{-1, +1\}$: 保留 sign，量化只对 magnitude 做（NVFP4 grid 对称）
- $w_{lower}, w_{upper} \in \mathcal{N}$: $|\tilde{w}|$ 所在 interval 的两个相邻 NVFP4 node
- $h_\beta(v) \in [0, 1]$: 可微 rounding function，输出插值系数（0 表示 round-down 到 $w_{lower}$，1 表示 round-up 到 $w_{upper}$）
- $v \in [0, 1]$: learnable rounding variable，对每个 weight element 独立
- $s_g$: per-block scale (FP8)
- $s_{global}$: tensor-level scale-of-scales (FP32)

**关键 intuition**：paper 在 ablation 里隐含一个非常重要的对比点——AdaRound 的 formula 是 $w_q = w_{lower} + h(v) \cdot \Delta$，其中 uniform quantization 下 $\Delta$ 是常数。AdaRound 的梯度 $\partial \mathcal{L} / \partial v$ 在 uniform grid 下被 $\Delta$ 等比缩放，所有 weight element 收到同等 magnitude 的 corrective signal。

FAAR 把 $\Delta = w_{upper} - w_{lower}$ **显式作为可变项放进 formula**：
- 处于 0→0.5 interval 的 weight，$\Delta = 0.5$
- 处于 4→6 interval 的 weight，$\Delta = 2.0$

大-magnitude weight 自动获得 4× 大的梯度 magnitude，因为 $\partial w_q / \partial v \propto \Delta$。这是 paper "Format-Aware" 的真正意义——让 optimization **感知 grid 的非均匀性**，而不是把 NVFP4 当 INT4 来处理。

### 3.2 Differentiable soft rounding function

$$h_\beta(v) = \sigma\big(\beta (v - 0.5)\big) = \frac{1}{1 + e^{-\beta(v - 0.5)}}$$

- $\sigma$: sigmoid 函数
- $\beta > 0$: temperature 参数
- 偏移 0.5: 让 $v = 0.5$ 时 $h = 0.5$，对应 "中立" 决策点
- $\beta \to 0$: $h_\beta(v) \to 0.5$ (constant)，相当于在 $w_{lower}$ 和 $w_{upper}$ 之间均匀混合——这阶段梯度平滑
- $\beta \to \infty$: $h_\beta(v) \to \mathbb{I}(v \geq 0.5)$，hard step function

annealing schedule：训练初期 $\beta$ 小，做 smooth interpolation，便于 gradient flow；后期 $\beta$ 大，逼近 binary decision。这和 Gumbel-Softmax 的 temperature annealing 是一个思路：
- Gumbel-Softmax (Jang et al. ICLR 2017): https://arxiv.org/abs/1611.01144
- Concrete distribution (Maddison et al. ICLR 2017): https://arxiv.org/abs/1611.00712

### 3.3 Initialization：一个非常聪明的细节

paper 在 Eq. (4) 里给了一个关键 init：

$$v_{init} = \frac{\frac{|w|}{s_g \cdot s_{global}} - w_{lower}}{w_{upper} - w_{lower}}$$

也就是 $v_{init}$ = weight 在 interval 里的相对位置 (fractional position)。这个 init 的意义：
1. 保留 RTN 的 implicit prior（如果 $v_{init} > 0.5$ 就是 round up，反之 round down）
2. 让训练从一个 near-RTN 的起点出发，避免 random init 导致的 poor local minima
3. 和 annealed $\beta$ 配合，训练初期 $h(v_{init}) \approx$ RTN decision，后期慢慢被 loss 拉到最优 binary decision

**这点非常关键**——它把 RTN 作为 inductive bias，让 learnable rounding 变成 "RTN + 微调"，而不是从 scratch 学。这种 "warm start" 在 quantization literature 里很常见，比如 BRECQ 也从 pre-trained model 出发。

### 3.4 Rounding regularization loss

Eq. (5) 中的 regularization term：

$$\mathcal{L}_{round} = \frac{1}{N}\sum_{i=1}^{N} \Big(1 - (2v_i - 1)^2\Big)$$

变量：
- $N$: 量化 weight element 总数
- $v_i \in [0, 1]$: 第 i 个 weight 的 learnable rounding variable

行为分析：
- $v_i = 0$: $(2 \cdot 0 - 1)^2 = 1$，loss = 0
- $v_i = 1$: $(2 \cdot 1 - 1)^2 = 1$，loss = 0
- $v_i = 0.5$: $(2 \cdot 0.5 - 1)^2 = 0$，loss = 1 (max)

所以这是一个 "push-to-binary" 的正则——把 $v$ 推向 0 或 1，惩罚中间值。这和 hardening 阶段 ($v \to \hat{v} \in \{0,1\}$) 配合：如果训练结束时 $v$ 仍然接近 0.5，hardening 会引入大的 quantization jump；反之如果 $v$ 已经接近 0/1，hardening 几乎无损。

这让我想到 BinaryConnect / BNN 里的 saturation regularizer，也用了类似的 $(2v-1)^2$ 形式：
- BinaryConnect (Courbariaux et al.): https://arxiv.org/abs/1511.00363

---

## 4. 2FA (Two-Stage Format Alignment) Pipeline

paper 把训练分成两 stage，每个 stage 解决不同 scale 的 error。

### 4.1 Stage 1: Layer-wise adaptive rounding

对每层 $l$ 独立优化，loss 如 Eq. (5)：

$$\mathcal{L}_{stage1} = \underbrace{\| \mathbf{X}\mathbf{W} - \mathbf{X}_q \mathbf{W}_q(\mathbf{V}) \|_F^2}_{\text{reconstruction MSE}} + \lambda_{round} \underbrace{\frac{1}{N}\sum_i (1 - (2v_i - 1)^2)}_{\text{rounding reg}}$$

变量：
- $\mathbf{X}$: 从 BF16 frozen model 采样的 input activation
- $\mathbf{X}_q$: 量化后的 activation (W4A4 setting)
- $\mathbf{W}$: 原始 BF16 weight
- $\mathbf{W}_q(\mathbf{V})$: NVFP4 量化 weight，依赖 learnable rounding variable tensor $\mathbf{V}$
- $\|\cdot\|_F$: Frobenius norm
- $\lambda_{round}$: rounding reg 的权重 hyperparameter

**关键设计**：每层独立优化时，其他层 frozen，只更新当前层的 $\mathbf{V}$。这一步等同于 layer-wise block reconstruction——BRECQ 的核心 idea 是 block-wise 而非 layer-wise，paper 这里用 layer-wise 可能是因为 NVFP4 已经够 low-bit，block 内的 correlation 不是 dominant 因素。

类比联想：SmoothQuant 也是 layer-wise 处理 outlier，AWQ 是 layer-wise scale search。这一脉 PTQ 工作都遵循 "layer-wise local reconstruction" paradigm：
- SmoothQuant: https://arxiv.org/abs/2211.10438
- AWQ: https://arxiv.org/abs/2306.00978
- GPTQ: https://arxiv.org/abs/2210.17323

### 4.2 Stage 2: Full-model alignment

Stage 1 解决 local error，但 inter-layer error accumulation 没管。Stage 2 把所有层组装成完整 NVFP4 model，做端到端 alignment：

$$\mathcal{L}_{stage2} = \lambda_{KL} \mathcal{L}_{KL} + \mathcal{L}_{MSE} + \lambda_{round} \sum_{l=1}^{L} \mathcal{L}_{round}^{(l)}$$

三个 loss 项：

**1. KL divergence** (logits level):

$$\mathcal{L}_{KL} = \text{KL}\big(\mathbf{P}_{fp} \| \mathbf{P}_q\big)$$

其中 $\mathbf{P}_{fp} = \text{softmax}(\mathbf{Z}_{fp}/\tau)$, $\mathbf{P}_q = \text{softmax}(\mathbf{Z}_q/\tau)$
- $\mathbf{Z}_{fp}, \mathbf{Z}_q$: BF16 与量化 model 的 logits
- $\tau$: temperature，控制分布的 sharpness
- $\mathbf{P}_{fp}, \mathbf{P}_q$: softened 后的 next-token probability distribution

这是经典的 knowledge distillation loss——让 quantized student 跟 BF16 teacher 的 output distribution 对齐。LLM-QAT、Compact Neural Representations 等工作都用了类似 KL on logits：
- LLM-QAT: https://arxiv.org/abs/2305.12285

**2. MSE on last hidden state**:

$$\mathcal{L}_{MSE} = \|\mathbf{H}_{fp} - \mathbf{H}_q\|_F^2$$

- $\mathbf{H}_{fp}, \mathbf{H}_q$: BF16 与量化 model 在最后一个 transformer layer 后的 hidden representation

**Intuition**: KL 只对齐 output behavior，但 hidden state 直接对齐 representation space 能保留更多 internal feature information，下游 task 的 transfer 更好。Table 4 的 cosine similarity 提升就是这条 loss 的功劳。

**3. Rounding reg (per layer)**: 把 stage 1 学到的 $v$ 锁在 binary 附近，防止 stage 2 全局优化把 $v$ 推回 0.5。

### 4.3 Hardening & inference

训练结束，按 Eq. (7) 做 deterministic hardening：

$$\hat{v} = \mathbb{I}(v \geq 0.5) \in \{0, 1\}$$

然后代回 Eq. (2):

$$w_{final} = \text{sign}(w) \cdot [w_{lower} + \hat{v} \cdot (w_{upper} - w_{lower})] \cdot (s_g \cdot s_{global})$$

得到的 $w_{final}$ 是 NVFP4 grid 上的 node × scale，可以直接 pack 到 NVFP4 hardware format 推理。整个 pipeline 在 inference 时**零额外开销**——所有学习量都已经 bake 进 weight。

---

## 5. 实验数据深度分析

### 5.1 Main PPL results (Table 3)

| Method | Llama3-1B WikiText-2 | Llama3-8B | Qwen3-1.7B | Qwen3-8B |
|---|---|---|---|---|
| BF16 | 11.98 | 7.54 | 21.04 | 12.21 |
| RTN | 14.28 | 8.44 | 23.06 | 12.67 |
| GPTQ | 13.74 | 8.32 | 21.48 | 12.49 |
| MR-GPTQ | 13.73 | 8.32 | 21.42 | 12.44 |
| 4/6 | 13.89 | 8.30 | 23.57 | 12.55 |
| GPTQ+4/6 | 13.66 | 8.29 | 22.68 | 12.62 |
| **Ours (FAAR+2FA)** | **12.60** | **8.13** | **21.27** | **12.32** |

几个观察：
1. **1B model 增益最大**：14.28 → 12.60，减少 1.68。8B 上 8.44 → 8.13，只减少 0.31。这是因为 small model 容量小，quantization error 占比大，所以 optimization 收益高。Scaling laws 视角看，model size 越大，quantization robustness 越强。
2. **Ours (strong baseline) 14.03 比 RTN 14.28 略好**：paper 设了个 "strong baseline" intermediate variant，从架构图 Figure 1 看可能是 "RTN + 某些工程改进"。从 Table 3 看，FAAR+2FA 比这个 strong baseline 还要好 1.43 PPL，说明 FAAR/2FA 是真正 drive 性能的核心。
3. **C4 上趋势一致**：Llama3-1B 从 36.19 → 34.41，Qwen3-1.7B 从 65.54 → 62.24。

GPTQ 这里效果不算特别亮眼，我推测原因：GPTQ 是 Hessian-based，对 uniform INT4 grid 假设较强，对 NVFP4 这种 non-uniform grid，Hessian-based compensator 未必能直接对应到 grid node 上的 optimal rounding。MR-GPTQ (ref [22]) 是把 GPTQ 适配到 microscaling format 的工作，但效果也一般，说明 PTQ 在 ultra-low precision 下确实需要 trainable approach。

### 5.2 Cosine similarity (Table 4)

| Method | Llama3-1B WikiText-2 | Llama3-1B C4 |
|---|---|---|
| RTN | 96.08 | 94.27 |
| GPTQ | 97.78 | 96.01 |
| GPTQ+4/6 | 97.93 | 97.14 |
| **Ours** | **98.06** | **97.50** |

cosine similarity 是 BF16 vs quantized model 最后 hidden state 的余弦相似度。**高 cosine similarity 解释了 PPL 提升的来源**——FAAR 的核心增益不是 logits 层面的语义对齐，而是 internal representation 的几何对齐。Stage 2 的 $\mathcal{L}_{MSE}$ on last hidden state 是关键。

让我做更细的 intuition：hidden state 是 next-token prediction 的 "原料"，cosine similarity 高意味着 quantized model 在 feature space 上几乎是 BF16 的旋转/缩放版，下一层 LM head 看到的 input 分布几乎不变。这种 alignment 比单纯 logits KL 更鲁棒，因为 hidden state 是更早的 representation，保留了更多信息。

### 5.3 Downstream task performance (Table 5)

| Method | BoolQ 1B | Arc-E 1B | Arc-C 1B | HellaSwag 1B | Avg 1B |
|---|---|---|---|---|---|
| BF16 | 63.61 | 62.08 | 36.86 | 64.24 | 56.70 |
| RTN | 58.55 | 57.31 | 34.28 | 59.97 | 52.53 |
| GPTQ | 61.06 | 57.41 | 33.07 | 60.58 | 53.03 |
| **Ours** | **63.27** | **61.70** | **36.09** | **62.80** | **55.97** |

注意：**Ours 在 BoolQ 上甚至超过 BF16** (63.27 vs 63.61，差 0.34，但 RTN 是 58.55，提升 +4.72)。这种 "quantization 反而 task 性能更好" 现象在 LLM quantization literature 里偶尔出现，可能 explanation 是 quantization 起到了 regularization 作用，类似于 stochastic depth / dropout 的效果。但更可能是 noise——0.34 在 BoolQ 这种 binary classification task 的统计 noise range 内。

8B 上 Avg: BF16 75.02 → Ours 73.28，gap 1.74；1B 上 gap 0.73。1B 反而 robust，跟 PPL 趋势一致。

### 5.4 Ablation (Table 6-8)

**Component ablation (Table 6)**:
| Method | Llama3-1B | Qwen3-1.7B |
|---|---|---|
| RTN | 14.28 | 23.06 |
| FAAR (only) | 13.01 | 21.86 |
| FAAR + 2FA | **12.60** | **21.27** |

FAAR 单独贡献 -1.27 PPL，2FA 再贡献 -0.41。**FAAR 是主菜，2FA 是 dessert**。这点合理：FAAR 直接 attack rounding suboptimality，2FA 只是把 local 优化组装起来做 global 微调，增量收益有限但稳定。

**Steps sensitivity (Table 7)**:
| Steps | Llama3-1B |
|---|---|
| 0 (=FAAR only) | 13.01 |
| 500 | 12.84 |
| 2500 | 12.60 |
| 10000 | 12.58 |

2500 steps 后基本收敛，10000 steps 只多 0.02。这是非常理想的 scaling 特性——4× compute 换 0.02 PPL，ROI 极低。对 edge deployment 很友好。

**LR sensitivity (Table 8)**:
| LR | Llama3-1B | Qwen3-1.7B |
|---|---|---|
| 5e-5 | 12.78 | 21.33 |
| 1e-4 | 12.69 | **21.27** |
| 5e-4 | **12.60** | 21.45 |
| 1e-3 | 12.82 | 21.91 |

不同 model optimal LR 差 5×，Llama3-1B 喜欢大 LR (5e-4)，Qwen3-1.7B 喜欢小 LR (1e-4)。paper 解释是 "weight variance 和 loss landscape 差异"。从架构角度看，Llama3 和 Qwen3 的 attention/MLP 实现细节、RoPE 配置、normalization 位置都不同，确实可能影响 loss landscape。但 5× 差异比较大，可能还有 batch size、init scheme 等因素。

---

## 6. Intuition 总结 & 相关联想

### 6.1 为什么 FAAR 工作——我的直觉解释

把 FAAR 和 AdaRound 放一起对比，能 build 很强的 intuition：

**AdaRound 的隐含假设**：uniform grid 下，每个 weight 的 rounding 决策独立同分布地贡献 $\Delta \cdot v$ 到 quantization error，loss landscape 在 $v$-space 上是 isotropic 的。

**FAAR 的核心修正**：non-uniform grid 下，不同 interval 的 $\Delta$ 差 4×（0.5 vs 2.0）。如果不显式注入 $\Delta$，optimizer 看到的梯度是经过 scaling 的，会 over-prioritize dense-region weight（小 $\Delta$ → 小梯度 magnitude → 学得慢），而 sparse-region weight（大 $\Delta$ → 大 error）本应优先优化。

但 FAAR 的 formula $\partial w_q / \partial v \propto \Delta$ 自动 compensate：大 $\Delta$ 的 weight 不仅有更大的 error sensitivity，还自动获得更大的 gradient。这就是 "format-aware" 的本质——**让 optimizer 看到的几何和真实 quantization error geometry 对齐**。

### 6.2 与 GPTQ 的对比

GPTQ 用 Hessian-based weight update，公式上是：

$$w_q^* = \arg\min_w \| W X - W_q X \|_F^2 \quad \text{s.t.} \quad W_q \in \text{INT4 grid}$$

通过 Cholesky 重建 Hessian 来 sequentially update weight。但 GPTQ 的核心 operation 是 **weight update**——它直接改 weight 数值来 minimize output reconstruction error，而不限于 grid node。

NVFP4 是 floating-point grid，可表示 node 是离散的 ±{0.5, 1.0, ..., 6.0} × scale，**weight 必须落在某个 node 上**，GPTQ 的 continuous update 失效。所以 GPTQ 对 NVFP4 不友好——paper Table 3 也印证：GPTQ 13.74 vs Ours 12.60。

FAAR 把 "search over discrete node assignment" 转化为 "optimize continuous rounding variable + hardening"，绕开了 GPTQ 在 non-uniform floating grid 上的限制。

### 6.3 与 QAT (Quantization-Aware Training) 的对比

QAT 的 idea 是把 quantization 模拟进 forward pass，通过 STE (Straight-Through Estimator) backward gradient，让整个网络 end-to-end 学会适应量化噪声。FAAR 的设计哲学类似——把 discrete 决策 relaxation 成 continuous，再用 annealing 推回 discrete。但 FAAR 是 **PTQ (post-training)**，仅用 4 GPU hours；QAT 通常需要 full pretraining-scale compute。

差异点：
1. QAT 优化所有 weight，FAAR 只优化 rounding variable $\mathbf{V}$（参数量等于 weight 数）
2. QAT 需要 training data + label，FAAR 只需 calibration data (无 label)
3. QAT 训练 full model，FAAR 只做 rounding decision，weight 数值本身不动

但这里有个微妙处：FAAR 的 $\mathbf{V}$ 参数量等于 weight 数，所以"训练参数量"上 FAAR ≈ QAT。差异在 compute：FAAR 用 layer-wise stage 1 + model-wise stage 2 + small calibration set，而 QAT 用 full forward-backward on full training data。

延伸联想——是否可以做 **NVFP4 QAT**？paper ref [24] TetraJet-v2、ref [26] Quartet II 都是 NVFP4 training 工作，说明这个方向也在 active research。FAAR 的 rounding relaxation 思路完全可以嵌入 QAT pipeline 作为 differentiable rounding operator，替代 STE。

### 6.4 与 MXFP4 / MXINT 的关系

NVFP4 是 NVIDIA 私有 variant，对应 OCP 标准 MXFP4 (Microscaling FP4)。MXFP4 spec 见 ref [21]，format 上和 NVFP4 几乎一致（E2M1 + per-block FP8 scale）。其他 microscaling formats:
- MXINT8 (8-bit integer + scale)
- MXFP6 (E3M2 或 E2M3)
- MXFP8 (E5M2 或 E4M3)

FAAR 的 format-aware 思路可以直接迁移到所有 microscaling formats——只要 grid 是 non-uniform 的，FAAR 的 $\Delta$ injection 就有效。MXINT8 是 uniform 的，FAAR 退化为 AdaRound。

paper ref [25] M2XFP 提出 metadata-augmented microscaling format，ref [22] MR-GPTQ 是 microscaling 适配版 GPTQ——这块生态在快速 build out。

### 6.5 我对 paper 的几点批评/疑问

1. **Activation quantization 怎么处理**？paper 在 W4A4 setting 下需要量化 activation，但 formula 主要描述 weight quantization。activation 用什么 rounding？是否也跑 FAAR？这点没讲清楚。

2. **Block size 影响**？paper 提到 "16 elements per block"，但没 ablate block size。block size 影响 $s_g$ 的 granularity，对小 magnitude variation 的捕捉很关键。这个 hyperparameter 应该有 sensitivity 分析。

3. **Outlier weight 怎么办**？NVFP4 grid 的最大 node 是 ±6.0，加上 $s_g \cdot s_{global}$ scaling。如果某个 weight 是 super-outlier（比如 >10×median），归一化后还是会落在 4→6 这个稀疏 interval，$\Delta=2$，FAAR 的 $\Delta$ injection 帮助大。但如果 outliers 集中在一个 block，$s_g$ 会被拉大，导致该 block 内其他 normal weight 被压缩到 0→0.5 这个 dense interval，反而损失信息。这是 microscaling formats 的经典 tradeoff。

4. **Per-block scale 的优化**？paper 假设 $s_g$ 给定（应该是 absmax-based 或类似），但 $s_g$ 的 choice 对 NVFP4 误差影响巨大。Four over Six (ref [23]) 就是 attack 这个问题——它优化 block scaling 算法。FAAR 没动 $s_g$，可能可以 joint optimize。

5. **Stage 2 的 KL temperature $\tau$ 没给具体值**。从 formula 看应该有，但 paper 没列。这个 hyperparameter 在 distillation 里通常很 sensitive。

### 6.6 推测的扩展方向

基于 FAAR 的 design philosophy，可以 speculate 几个 extension：

**a. FAAR + Hessian weighting**: 把 stage 1 的 MSE loss 改成 Hessian-weighted MSE，对 outlier row/col 给更高 weight：
$$\mathcal{L} = \| (\mathbf{X}\mathbf{W} - \mathbf{X}_q \mathbf{W}_q) \cdot \mathbf{H}^{-1/2} \|_F^2$$
这样 sensitive row 的 error 被惩罚更重。GPTQ 的 Hessian insight 可以嫁接到 FAAR。

**b. FAAR for activation**: activation quantization 也用 NVFP4，可以学一个 activation-level rounding variable。Activation 的 distribution 比 weight 更 dynamic，FAAR 在 activation 上的增益可能更大。但 activation 是 runtime 数据，learnable rounding 在 inference 时怎么 deploy？可能需要 input-conditional rounding predictor。

**c. Mixed-precision FAAR**: 不是所有 weight 都需要 FAAR。可以先用 saliency score (e.g., Hessian trace) 筛 top-k sensitive weight，只对这些跑 FAAR。Insensitive weight 用 RTN。这样参数量可以大幅降低。

**d. NVFP4 + LoRA**: 把 FAAR 学到的 rounding pattern 作为 "frozen base"，再用 LoRA 在 NVFP4 backbone 上 fine-tune downstream task。这可能是 edge deployment 的最佳 deployment stack。

**e. FAAR for pretraining**: NVIDIA 自己的 NVFP4 pretraining paper (ref [1], arXiv 2509.25149) 用 NVFP4 做 pretraining。如果把 FAAR 的 differentiable rounding operator 用作 pretraining 的 quantization function，可能比 STE 收敛更好。

---

## 7. Reference 与相关链接

### Paper 内引用
- NVFP4 pretraining (NVIDIA): https://arxiv.org/abs/2509.25149
- AdaRound: https://arxiv.org/abs/2004.10568
- BRECQ: https://arxiv.org/abs/2102.05426
- GPTQ: https://arxiv.org/abs/2210.17323
- AWQ: https://arxiv.org/abs/2306.00978
- SmoothQuant: https://arxiv.org/abs/2211.10438
- MR-GPTQ (ref [22]): https://arxiv.org/abs/2509.23202
- Four over Six (ref [23]): https://arxiv.org/abs/2512.02010
- TetraJet-v2 (ref [24]): https://arxiv.org/abs/2510.27527
- M2XFP (ref [25]): https://arxiv.org/abs/2601.19213
- MS-EDEN / Quartet II (ref [26]): https://arxiv.org/abs/2601.22813
- OCP Microscaling Formats (MX) Spec v1.0: https://www.opencompute.org/docs/2023/30545824-OCP-Microscaling-Formats-MX-v1.0.pdf
- Llama 3: https://arxiv.org/abs/2407.21783
- Qwen3: https://arxiv.org/abs/2505.09388

### Intuition building 相关经典工作
- BinaryConnect (Bengio group): https://arxiv.org/abs/1511.00363
- Gumbel-Softmax (Jang et al.): https://arxiv.org/abs/1611.01144
- Concrete Distribution (Maddison et al.): https://arxiv.org/abs/1611.00712
- Straight-Through Estimator (Bengio 2013): https://arxiv.org/abs/1308.3432
- Lloyd-Max quantizer (经典 scalar quantization): https://en.wikipedia.org/wiki/Lloyd%27s_algorithm
- LLM-QAT: https://arxiv.org/abs/2305.12285

### NVIDIA Blackwell / NVFP4 资源
- Blackwell architecture whitepaper: https://resources.nvidia.com/en-us-blackwell-architecture
- NVIDIA Blackwell B200 announcement: https://www.nvidia.com/en-us/data-center/blackwell-ai-processors/
- NVFP4 blog / developer docs: https://developer.nvidia.com/blog/

### 相关综述 / 教程
- A Survey of Quantization Methods for Efficient Neural Network Inference: https://arxiv.org/abs/2103.13630
- A White Paper on Neural Network Quantization: https://arxiv.org/abs/2106.06166
- LLM.int8() / SpQR (Dettmers): https://arxiv.org/abs/2208.07339

---

## 8. 一段话总结直觉

NVFP4 把 FP4 浮点格式 + hierarchical scaling 做到了 hardware-native ultra-low precision path，但其 non-uniform grid 让传统的 RTN、AdaRound 这类 uniform-grid 优化方法都失效。FAAR 的核心 move 是把每个 weight 的离散 rounding decision 松弛成连续变量 $v \in [0,1]$，并通过 formula 中显式的 interval span $\Delta = w_{upper} - w_{lower}$ 让 gradient 自然按 grid 非均匀性 scale，从而把 optimizer 的搜索几何对齐到真实 quantization error geometry。配合 sigmoid temperature annealing 和 binary regularization 收敛到 hard decision，再通过两阶段 2FA alignment（layer-wise local reconstruction → full-model KL + hidden state MSE）缓解 inter-layer error accumulation。最终 Llama3-1B 在 4 GPU hours 内把 PPL 从 RTN 的 14.28 拉到 12.60，逼近 BF16 11.98——这是 NVFP4 ultra-low precision deployment 的工程级里程碑。

如果你想继续深入某个方向（比如 activation quantization 的 FAAR extension、Hessian 加权版 FAAR、或者 NVFP4 vs MXFP4 的细节差异），我可以继续展开。
