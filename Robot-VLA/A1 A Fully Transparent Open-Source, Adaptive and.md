---
source_pdf: A1 A Fully Transparent Open-Source, Adaptive and.pdf
paper_sha256: 4505535c95afc23c858005d5570319e63f335333826d2d13f599d3d2979f8ef0
processed_at: '2026-07-17T22:21:44-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# A1 Paper 深度讲解

## 1. 核心问题与 Motivation

A1 paper tackle 的核心问题是 VLA (Vision-Language-Action) model 的 deployment cost。当前 SOTA VLA 系统（如 $\pi_0$, $\pi_{0.5}$）依赖 billion-scale VLM backbone + iterative diffusion/flow-matching action head（10-20 步 denoising），导致 real-time control 在 commodity hardware 上几乎不可行。

**三个 empirical observations** 构成了 A1 的直觉基础：

| Observation | 含义 | Implication |
|------------|------|-------------|
| Trajectory convergence | flow-matching 轨迹在 <3 步内锁定 correct mode | 后续 iterations 只是 refinement，diminishing returns |
| Action redundancy | 连续 control step 的 action 变化平滑 | 简单动作只需要 coarse update |
| Layer-wise coupling | intermediate VLM hidden states 已编码 sufficient spatial/visual features | full-depth backbone evaluation 经常 redundant |

**核心原则**："Spend compute only when it changes the action" —— 只有当计算会改变 action 时才花 compute。这个原则同时驱动了 backbone early-exit 和 action head 的 truncation。

---

## 2. Architecture 详解

### 2.1 整体结构

A1 由两个核心 component 构成：

**VLM Backbone**: Molmo-7B
- Vision encoder: CLIP/SigLIP（2,013.36 GFLOPs, sequence length 352）
- LLM: Qwen2-7B（28 层，每层 323.61 GFLOPs，total 9,061.01 GFLOPs）
- Total VLM: 11,074.39 GFLOPs

Molmo 提供 implicit affordance priors，这是关键 —— VLM 在大规模 multimodal pretraining 中已经学到了 object affordance 的 representation，可以直接被 action head 利用。

**Action Head** 两种实现：
- $A_1$-FM (Flow Matching): Qwen3-400M，0.493 GFLOPs/timestep
- $A_1$-MLP: 轻量 MLP head，1.850 GFLOPs（action dim=7, chunk size=8）

### 2.2 VLM 与 Action Head 的 Bridging

对于 FM action expert，采用 **KV-conditioned self-attention**：
- Main LLM 产生的 prefix context 作为 past keys/values 注入 decoder-only stack
- Suffix tokens (action + state) 可以 attend 到 cached prefix + 自己的 block
- 这样实现了 VLM 语义信息到 action 生成的桥接

对于 MLP head，添加 special query action tokens $S$ 到 input，通过 parallel decoding 输出 action。

---

## 3. Flow Matching 数学原理

### 3.1 训练目标

公式 (1)：
$$\mathcal{L}^{\tau}(\boldsymbol{\theta}) = \mathbb{E}_{p(\mathbf{A}_t|\mathbf{o}_t), q(\mathbf{A}_t^{\tau}|\mathbf{A}_t)} \left\| \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{A}_t^{\tau}, \mathbf{o}_t) - \mathbf{u}(\mathbf{A}_t^{\tau}|\mathbf{A}_t) \right\|^2$$

**变量解析**：
- $\tau \in [0,1]$：flow matching timestep。$\tau=0$ 表示 pure noise，$\tau=1$ 表示 clean action
- $\mathbf{A}_t = [\mathbf{a}_t, \mathbf{a}_{t+1}, \ldots, \mathbf{a}_{t+H}] \in \mathbb{R}^{H \times D}$：ground truth action chunk，$H$ 是 chunk length，$D$ 是 action dimension
- $\mathbf{o}_t = [\mathbf{I}_t^1, \ldots, \mathbf{I}_t^n, \mathbf{q}_t]$：observation，包含 $n$ 个 camera images + proprioceptive state $\mathbf{q}_t$ (gripper pose, joint angles)
- $\mathbf{A}_t^{\tau} = \tau \mathbf{A}_t + (1-\tau)\boldsymbol{\epsilon}$：noisy action，其中 $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$
- $\mathbf{v}_\theta(\mathbf{A}_t^{\tau}, \mathbf{o}_t)$：network 预测的 vector field（要学习的）
- $\mathbf{u}(\mathbf{A}_t^{\tau}|\mathbf{A}_t) = \boldsymbol{\epsilon} - \mathbf{A}_t$：target denoising vector field（supervision signal）

**直觉**：flow matching 学习一个 vector field $\mathbf{v}_\theta$，它能把任意 noise sample $\mathbf{A}_t^0$ 通过 ODE 积分到 clean action $\mathbf{A}_t^1$。$\tau$ 从 beta distribution 采样，偏向 lower (noisier) timesteps，因为早期 denoising 步骤更难学。

### 3.2 Inference: Forward Euler Integration

公式 (2)：
$$\mathbf{A}_t^{\tau+\delta} = \mathbf{A}_t^{\tau} + \delta \mathbf{v}_\theta(\mathbf{A}_t^{\tau}, \mathbf{o}_t)$$

**变量**：
- $\delta$：integration step size。从 $\tau=0$ 积分到 $\tau=1$，总步数为 $1/\delta$（通常 $\delta=0.1$ 即 10 步，或 $\delta=0.05$ 即 20 步）
- 起始条件 $\mathbf{A}_t^0 \sim \mathcal{N}(0, \mathbf{I})$，即 random Gaussian noise

这是最简单的 first-order ODE solver。每步只需一次 network forward pass。

### 3.3 MLP Head 监督

公式 (3)-(5)：
$$\mathbf{H}_t = f_\phi([\mathbf{o}_t, \ell, S])_S$$
$$\hat{\mathbf{A}}_t = g_\psi(\mathbf{H}_t)$$
$$\mathcal{L}_{\mathrm{MLP}}(\phi, \psi) = \mathbb{E}_{p(\mathbf{A}_t|\mathbf{o}_t, \ell)} \|\hat{\mathbf{A}}_t - \mathbf{A}_t\|$$

MLP head 用 L1 loss（不是 L2），因为 L1 对 outlier 更 robust，能 suppress noise。

---

## 4. Adaptive Inference 机制（核心创新）

### 4.1 Multi-Exit Training

训练时，不是只用最后一层，而是随机采样 layer index $i \sim \mathcal{U}(0, L)$：
- 对 MLP head：从第 $i$ 层 hidden state 预测 action，监督 $\mathcal{L}^{(i)}$
- 对 FM head：LLM 执行到第 $i$ 层，FM action head 也执行相应层数

这迫使每一层都能独立产生合理的 action prediction，为 inference 时的 early exit 铺路。

### 4.2 Early-Termination via Action-Consistency

公式 (6)：
$$\Delta_t^i = d(\mathbf{A}_t^{(i)}, \mathbf{A}_t^{(i-1)}) < \eta_i$$

**变量**：
- $\mathbf{A}_t^{(i)}$：从第 $i$ 层产生的 action chunk
- $d(\cdot, \cdot)$：discrepancy metric（cosine similarity / L2 distance / mean absolute deviation）
- $\eta_i$：layer-specific threshold，offline calibrated
- 上标 $(i)$ 表示从第 $i$ 层 generated 的 action

**直觉**：如果连续两层产生的 action 足够接近，说明模型已经"想清楚了"，不需要继续往后算。

### 4.3 Threshold Calibration via Filtered Quantile

公式 (7)：
$$V \in \mathbb{R}^{K \times N}, \quad V[k, n] \equiv \Delta_{t_n}^{i_k}$$

**变量**：
- $K$：exit layer 的数量（$K \leq L$，例如每隔 2 层设一个 exit，则 $K=14$）
- $N$：training set 总样本数
- $V[k, n]$：第 $n$ 个样本在第 $k$ 个 exit layer 的 action discrepancy

公式 (8)：exit probability distribution
$$p_k \propto \rho^k, \quad \rho = c > 0$$

**变量**：
- $p_k$：第 $k$ 个 exit 的 target probability mass
- $c$：exit criterion，越小越偏向 early exit
- 这是 exponential distribution，鼓励 early exit

公式 (9)：filtered-quantile procedure
$$\eta_{i_k} = Q_{p_k}\left(\{V[k, n]\}_{n \in \mathbb{Z}}\right), \quad \mathbb{Z} \gets \{n \in \mathbb{Z} | V[k, n] > \eta_{i_k}\}$$

**变量**：
- $Q_{p_k}$：$p_k$-quantile operator
- $\mathbb{Z}$：尚未被分配到 earlier exit 的 sample index set

**直觉**：这个 "filtered" 的关键在于 disjoint assignment —— 一旦某个样本被 early exit 捕获，它就不会出现在后续 exit 的 calibration 中。这样确保 exit probability 之和为 1，且每个 exit 真正负责"剩余"的样本。

也支持 Gaussian 分布（公式 11）和 Gamma 分布（公式 12）：
$$p_k \propto \exp\left(-\frac{(k-c)^2}{2\sigma^2}\right), \quad c = \text{exit\_criterion}$$
$$p_k \propto \text{GammaPDF}(k; \alpha, \text{scale}), \quad \alpha = \text{exit\_criterion}$$

### 4.4 Inter-Layer Truncated Flow Matching（核心创新）

**Problem**：early exit 把 LLM 的计算 reduce 了，但 FM action head 在每个 exit layer 都要跑 $\delta=10$ 步 denoising，反而把 bottleneck 转移到了 action head。

**Solution**：公式 (10)
$$\mathbf{A}_t^{0(i+1)} = \mathbf{A}_t^{1(i)}$$

**变量**：
- $\mathbf{A}_t^{0(i+1)}$：第 $i+1$ 层 denoising 的初始条件（$\tau=0$ 时的 sample）
- $\mathbf{A}_t^{1(i)}$：第 $i$ 层 denoising 的最终输出（$\tau=1$ 时的 sample）

**直觉**：传统做法每层都从 random noise $\mathcal{N}(0, \mathbf{I})$ 开始 denoise，浪费了前一层的成果。A1 把前一层的 denoised output 作为下一层的 warm-start initialization，并且每层只跑 $\delta=2$ 步（而不是 10 步）。

这相当于：
- Denoising 进度跨层累计（cumulative denoising progress across layers）
- 每层做 fine-grained refinement，而不是从头来
- Warm-start 还鼓励 early exit，因为后续层的初始 action 已经"接近收敛"

---

## 5. Training Recipe 细节

### 5.1 数据组成

**Open-source datasets**: DROID, AgiBot, RoboCOIN, RoboMind, GM-100, RoboChallenge
**Self-collected**: 15,951 real-world trajectories across ARX, Franka, UR5, Agibot

统一 episodic format: $(o_t, s_t, \ell, a_t)$
- $o_t$：visual observations (RGB, multi-view)
- $s_t$：robot state/proprioception
- $\ell$：language goal
- $a_t$：continuous action

**关键设计**：intentionally avoid state normalization across datasets，preserve intrinsic action space characteristics of identical embodiments。

### 5.2 训练超参数（Table 8）

| Configuration | Value |
|--------------|-------|
| Optimizer | AdamW |
| Batch size | 1024 |
| Total steps | 200K |
| ViT backbone LR | 0 (frozen) |
| VLM components LR | $5 \times 10^{-6}$ |
| Action head LR | $5 \times 10^{-5}$ |
| Warmup steps | 2,000 |
| Freeze steps (VLM) | 1,000 |
| LR decay | Cosine annealing |
| State mask probability | 0.5 |
| Visual augmentation | Random erasing, Sharpening |

**关键设计**：
1. ViT 全程 frozen —— 保留 pretrained visual representation
2. VLM 前 1000 步 frozen，防止 catastrophic forgetting
3. Action head LR 比 VLM 高 10x，因为它需要快速适应 motor control
4. State zero-out augmentation —— 0.5 概率 mask 掉 state 维度，提升 partial observability robustness

### 5.3 Fine-tuning（Table 9）

| Task | Batch Size | Training Steps | State Mask Prob |
|------|-----------|----------------|-----------------|
| LIBERO | 128 | 50K | 0.0 |
| VLABench | 64 | 50K | 0.0 |
| RoboChallenge (Aloha) | 64 | 100K | 0.3 |
| RoboChallenge (ARX5) | 32 | 50K | 0.3 |
| RoboChallenge (UR5) | 64 | 50K | 0.3 |
| RoboChallenge (Franka) | 64 | 50K | 0.3 |

---

## 6. 实验结果深度分析

### 6.1 Simulation Benchmark（Table 1）

| Model | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | LIBERO-Long | LIBERO-Avg | VLABench-Avg |
|-------|---------------|---------------|-------------|-------------|------------|--------------|
| Octo | 78.9 | 85.7 | 84.6 | 51.1 | 75.1 | 1.5 |
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 | 14.5 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 | - |
| MolmoAct | 87.0 | 95.4 | 87.6 | 77.2 | 86.6 | - |
| SmolVLA | 93.0 | 94.0 | 91.0 | 77.0 | 88.8 | - |
| $\pi_0$ | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 | 42.0 |
| $\pi_{0.5}$ | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 | 49.5 |
| **A1** | **97.4** | **99.8** | **97.6** | **91.4** | **96.6** | **53.5** |

A1 在 LIBERO 上达到 96.6%（接近 $\pi_{0.5}$ 的 96.9%），在 VLABench 上 53.5%，比 $\pi_{0.5}$ 高 4%。VLABench 强调 world knowledge + common sense + multi-step reasoning，A1 优势明显。

### 6.2 Real-World Multi-Robot（Table 2）

| Model | UR5-stack | UR5-arrange | Franka-cup | Franka-arrange | Franka-small | AgiBot-move | AgiBot-glue | OpenArm-table | OpenArm-tidy | Dobot-yellow | Dobot-cook | Dobot-pour | Mean |
|-------|-----------|-------------|------------|----------------|--------------|-------------|-------------|---------------|--------------|--------------|-------------|-------------|------|
| $\pi_0$ | 100 | 80 | 70 | 30 | 10 | 10 | 30 | 0 | 30 | 80 | 10 | 40 | 40.8 |
| $\pi_{0.5}$ | 80 | 100 | 50 | 20 | 30 | 40 | 60 | 10 | 80 | 60 | 20 | 20 | 47.5 |
| **A1** | **100** | 60 | 50 | 40 | **50** | **80** | **80** | 20 | **80** | 70 | 20 | 30 | **56.7** |

A1 比 $\pi_{0.5}$ 高 9.2%，比 $\pi_0$ 高 15.9%。值得注意的是 **few-shot learning**：在 "fruits (small)" 任务上只用 50 samples，A1 达到 50%，比 baseline 高 20-40%。

### 6.3 RoboChallenge（Table 4）

| Model | Type | Mean Success Rate |
|-------|------|-------------------|
| DM0 | closed-source | 62.00 |
| Spirit-v1.5 | closed-source | 51.00 |
| GigaBrain | closed-source | 51.67 |
| $\pi_{0.5}$ | closed-source | 42.67 |
| wall-oss | open | 35.33 |
| **A1** | **fully open** | **29.00** |
| $\pi_0$ | closed-source | 28.33 |
| X-VLA | open | 21.33 |
| RDT-1B | open | 15.00 |

A1 在 fully open-source 类别中排名第一，整体第六。特别在 "Open Drawer" (100%)、"Put Cup on Coaster" (90%)、"Stack Bowls" (80%) 等 precise manipulation 任务上表现出色。

### 6.4 Ablation: Early-Exit 效果（Table 5, A1-MLP）

| Config | Spatial | Object | Goal | Long | Avg | TFLOPs | Inf. time |
|--------|---------|--------|------|------|-----|--------|-----------|
| no exit (full-layer train) | 98.3 | 99.3 | 97.0 | 88.3 | 95.8 | 243.0 | 17.5s |
| no exit (multi-exit train) | 97.4 | 100.0 | 97.4 | 91.0 | 96.5 | 243.0 | 17.5s |
| c=1.0 | 97.4 | 99.8 | 97.6 | 91.4 | 96.6 | 205.0 (15.6%↓) | 16.5s↓ |
| c=0.7 | **98.4** | 99.8 | 97.4 | 89.6 | 96.3 | 148.1 (39.1%↓) | 6.8s↓ |
| c=0.4 | 95.6 | 98.4 | 95.0 | 87.0 | 94.0 | 100.8 (58.5%↓) | 5.6s↓ |
| c=0.1 | 96.2 | 98.2 | 94.4 | 80.4 | 92.3 | 57.0 (76.6%↓) | - |

**惊人发现**：
1. c=1.0 时，multi-exit training + early exit 比 full-layer training + no exit 还好（96.6% vs 95.8%）
2. 减少 76.6% compute，success rate 只掉 1.7%（96.6→92.3）
3. **Less computation sometimes leads to better performance**：c=0.7 时 Spatial 任务 98.4% > c=1.0 时的 97.4%。这表明 deep layers 可能引入 noise/redundancy，early exit 反而更鲁棒

### 6.5 Ablation: Inter-Layer Truncated FM（Table 6, A1-FM）

| c, δ | Spatial | Object | Goal | Long | Avg | TFLOPs | Inf. time |
|------|---------|--------|------|------|-----|--------|-----------|
| no exit†, 10 | 97.2 | 99.2 | 94.6 | 79.2 | 92.6 | 231.3 | 37.9s |
| no exit‡, 10 | 97.4 | 99.2 | 96.2 | 91.2 | 96.0 | 229.8 | 37.8s |
| no exit‡, 2 | 97.4 | 98.6 | 96.8 | 89.6 | 95.6 | 226.9 | 32.2s |
| 1.0, 10 | 97.2 | 99.6 | 97.0 | 91.8 | 96.4 | 150.6↓ | 40.9s (7.9%↑) |
| 1.0, 2 | 94.6 | 99.0 | 98.0 | 90.0 | 95.4 | 167.9↓ | 27.5s (27.4%↓) |
| **1.0, 2*** | 95.4 | 99.0 | 97.8 | 93.2 | 96.4 | 156.8↓ | **10.5s (72.3%↓)** |
| 0.8, 2* | 96.6 | 98.6 | 94.8 | 88.2 | 94.6 | 116.8↓ | 9.0s (76.3%↓) |

**关键观察**：
1. **没有 warm-start 时** (1.0, 2)：early exit 反而让 inference time 增加（40.9s vs 37.8s），因为每层都要跑 FM head
2. **加 warm-start 后** (1.0, 2*)：inference time 从 40.9s 暴降到 10.5s（72.3% reduction），success rate 还从 95.4% 升到 96.4%
3. Warm-start 鼓励 early exit，因为前一层 denoised output 已经接近收敛，consistency test 更容易触发

### 6.6 Computational Cost Analysis（Table 3）

| Component | Time (s) | GFLOPs |
|-----------|----------|--------|
| CLIP | 0.167 | 2013.36 |
| LLM (L=28) | 0.612 | 9061.01 |
| FM (δ=10) | 0.366 | 4.93 |
| A1-FM (δ=10, normal) | 1.151 | 11130.30 |
| A1-FM^e (δ=10, early-exit to final) | 4.44 | 31503.30 |
| A1-FM^e (δ=2, early-exit) | 0.728 | 11160.20 |

**洞察**：FM head 单步计算量小（0.493 GFLOPs），但 iterative denoising 步骤的时间开销 substantial。δ=10 时 early-exit 到 final layer 要 4.44s，而 δ=2 只要 0.73s。这正是 Inter-Layer Truncated FM 必要性的 quantitative evidence。

### 6.7 Zero-Shot Generalization（Table 7, LIBERO-Plus）

| Method | Spatial | Object | Goal | Long | Avg |
|--------|---------|--------|------|------|-----|
| OpenVLA | 19.4 | 14.0 | 15.1 | 14.3 | 15.6 |
| OpenVLA-OFT | 84.0 | 66.5 | 63.0 | 66.4 | 69.6 |
| $\pi_0$ | 60.7 | 61.4 | 44.9 | 48.4 | 53.6 |
| $\pi_0$-FAST | 74.4 | 72.7 | 57.5 | 43.4 | 61.6 |
| **A1-FM** | **86.6** | **80.0** | **66.8** | **58.0** | **75.3** |

A1 在 distribution shift（object layouts, language instructions, textures, lighting）下达到 75.3%，显著优于所有 baseline。

### 6.8 Real-World Early-Exit（Table 10）

| Config | Metric | Pick glue |
|--------|--------|-----------|
| no exit† | accuracy | 80 |
| c=1.0 | accuracy / compute↓ | 70 / 49.3%↓ |
| c=0.8 | accuracy / compute↓ | 70 / 64.7%↓ |
| c=0.4 | accuracy / compute↓ | 80 / 84.6%↓ |

**惊艳结果**：c=0.4 时 reduce 84.6% compute，accuracy 与 full inference 完全相同（80%）。Figure 6 显示 model 通常在 layer 3 或 5 就 exit。

---

## 7. 可视化分析：Adaptive Inference 行为

Figure 4 展示了 LIBERO-Long 任务（"turn on the stove and put the moka pot on it"）的 layer exit pattern：
- **简单动作**（movement）：layer 3 或 5 exit
- **复杂动作**（turn on stove, pick up pot）：layer 17 或 25 exit

这验证了 "spend compute only when it changes the action" 原则 —— 模型自适应地在简单 step 上省 compute，在关键 step 上花 compute。

---

## 8. 与 Related Work 的区别

### 8.1 vs DeeR-VLA [40, 41]
- DeeR-VLA: dynamic early-exit inference，但只加速 VLM backbone
- A1: 同时加速 backbone + action head（Inter-Layer Truncated FM），且 training 时用 single shared action head

### 8.2 vs $\pi_0$ [2] / $\pi_{0.5}$ [12]
- $\pi_0$: 引入 flow-matching atop VLM，但 inference cost 高
- A1: 同样用 flow-matching，但加了 truncated + warm-start，大幅 reduce latency

### 8.3 vs EfficientVLA [38] / VLA-Cache [35]
- 这些方法用 training-free acceleration, layer pruning, token caching
- A1: 用 consistency-based adaptive exit + cross-layer denoising warm-start，更系统化

---

## 9. Limitations

1. **Affordance pretraining 依赖 labeled data**：未来可探索 unsupervised affordance mining from robot data + human behavior videos
2. **Imitation learning 的 cumulative error**：可考虑加入 reinforcement learning 做 environment interaction feedback
3. **Threshold calibration 需要额外 training pass**：虽然 minor，但仍是 overhead
4. **Cloud-server 与 local arm 之间的 network latency**：需要 asynchronous execution methods 提升 smoothness

---

## 10. 我的 Intuition 构建

### 10.1 为什么 Early Exit 有效？

VLM 的深层主要在做 abstract reasoning / 语义整合，但对 motor control 来说，intermediate layers 的 spatial + visual features 已经 sufficient。这类似于 Mixture-of-Experts 的思想：不同 layer 负责不同抽象层级，action generation 不需要最深层的高阶抽象。

### 10.2 为什么 Warm-Start 是关键？

如果不 warm-start，early exit 把 bottleneck 从 VLM 转移到 FM head。每层都从 $\mathcal{N}(0, \mathbf{I})$ 开始 denoise 10 步，相当于每层独立 solve 一个 denoising problem，浪费严重。

Warm-start 把 denoising 看作 **跨层累积的 refinement process**：
- Layer 3: 从 random noise 跑 2 步 → 粗 action
- Layer 5: 从 layer 3 的输出跑 2 步 → 更精 action  
- Layer 17: 从 layer 15 的输出跑 2 步 → 精细 action

这相当于把 28 层 LLM × 2 步 denoising = 56 步 effective denoising，比传统 10 步还多，但分布在 LLM forward pass 中，几乎不增加开销。

### 10.3 为什么 Multi-Exit Training 比 Full-Layer Training 好？

Multi-exit training 强迫每一层都独立产生 action，相当于一种 **deep supervision**。这：
1. 防止深层 overfitting 到 training distribution
2. 让每一层学习更 general 的 representation
3. Implicit regularization，类似 DropPath / stochastic depth

### 10.4 Budget-Aware 的优雅之处

通过 exit criterion $c$ 可以 smooth 调节 compute-accuracy trade-off：
- $c \to \infty$：退化为 full inference
- $c \to 0$：尽可能 early exit
- Exponential distribution $p_k \propto c^k$ 是 natural choice，因为 early exit 的"风险"应该随层数指数衰减

---

## 11. Reference Links

- **A1 GitHub**: https://github.com/ATeam-Research/A1
- **A1 Project Page**: http://www.ateam.xin/#/research/A1
- **Molmo (VLM backbone)**: https://arxiv.org/abs/2409.17146
- **$\pi_0$ (flow-matching VLA)**: https://arxiv.org/abs/2410.24164
- **$\pi_{0.5}$**: https://arxiv.org/abs/2504.16054
- **DeeR-VLA (early-exit baseline)**: https://arxiv.org/abs/2411.02359
- **Flow Matching (original)**: https://arxiv.org/abs/2210.02747
- **LIBERO benchmark**: https://arxiv.org/abs/2311.09129 (NeurIPS 2023)
- **VLABench**: https://arxiv.org/abs/2501.13557
- **RoboChallenge**: https://arxiv.org/abs/2510.17950
- **DROID dataset**: https://arxiv.org/abs/2403.12945
- **AgiBot World**: https://arxiv.org/abs/2503.06669
- **OpenVLA-OFT**: https://arxiv.org/abs/2502.19645
- **RDT-1B**: https://arxiv.org/abs/2410.07864
- **X-VLA**: https://arxiv.org/abs/2510.10274
- **Qwen3**: https://arxiv.org/abs/2505.09388
- **SmolVLA**: https://arxiv.org/abs/2506.01844
- **EfficientVLA**: https://arxiv.org/abs/2506.10100
- **VLA-Cache**: https://arxiv.org/abs/2502.02175
- **Pivot-R (related work from same group)**: https://arxiv.org/abs/2410.09633 (NeurIPS 2024)

---

## 12. 总结

A1 的核心贡献在于 **jointly accelerating VLM backbone + action head**，通过：
1. **Action-consistency early exit**：把 VLM 的 redundant computation 砍掉
2. **Inter-Layer Truncated Flow Matching with warm-start**：把 iterative denoising 的 overhead 砍掉

两者协同工作，达到 72% latency reduction（FM inference）+ 76.6% backbone computation reduction，几乎不损失 performance。配合 fully open-source stack（code, data pipeline, weights, evaluation），A1 是 VLA efficiency 研究的重要 milestone，也证明了 transparent research 在 real-world robotic manipulation 上可以 competitive with closed-source systems。
