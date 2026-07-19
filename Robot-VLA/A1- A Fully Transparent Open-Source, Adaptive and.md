---
source_pdf: A1- A Fully Transparent Open-Source, Adaptive and.pdf
paper_sha256: 4505535c95afc23c858005d5570319e63f335333826d2d13f599d3d2979f8ef0
processed_at: '2026-07-17T22:26:50-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# A1: Adaptive Truncated VLA 深度解析

## 1. 核心问题与 Motivation

这篇paper要解决的核心矛盾：**VLA models在inference时的computational cost过高**。具体来说有两个bottleneck：

1. **VLM backbone cost**: billion-scale VLM (Molmo-7B = CLIP + 28-layer Qwen2-7B)的forward pass约11 TFLOPs
2. **Action head cost**: Flow matching需要10-20步iterative denoising，每步约0.5 GFLOPs但latency显著

关键intuition来自三个empirical observations：

**Observation 1 - Trajectory Convergence**: Flow matching的denoising trajectory在<3步内就lock onto correct mode，后续步骤主要是refinement with diminishing returns。这说明大部分denoising steps是redundant的。

**Observation 2 - Action Redundancy**: 连续control steps的actions变化平滑（参考Pivot-R [45]），只需要coarse updates。

**Observation 3 - Layer-wise Coupling**: VLM的intermediate hidden states已经encode足够的spatial/visual features来seed action prediction。

这三个observations指向一个原则：**只在action会改变时才spend compute**。

## 2. 架构详解

### 2.1 整体架构

```
Input: o_t = [I_t^1, ..., I_t^n, q_t]  (multi-view images + proprioception)
       + language instruction ℓ
            ↓
    [CLIP Vision Encoder]  (frozen, 2013 GFLOPs)
            ↓
    [Qwen2-7B, 28 layers]  (每层 323 GFLOPs)
            ↓ KV-conditioned self-attention
    [Action Head]
        ├── Flow Matching (Qwen3-400M, ~0.5 GFLOPs/step × 10 steps)
        └── MLP (1.85 GFLOPs)
            ↓
    Action chunk A_t = [a_t, a_{t+1}, ..., a_{t+H}] ∈ R^{H×D}
```

### 2.2 Action Head的两种实现

**Flow Matching Head**使用conditional flow matching loss：

$$\mathcal{L}^\tau(\theta) = \mathbb{E}_{p(\mathbf{A}_t|\mathbf{o}_t), q(\mathbf{A}_t^\tau|\mathbf{A}_t)} \|\mathbf{v}_\theta(\mathbf{A}_t^\tau, \mathbf{o}_t) - \mathbf{u}(\mathbf{A}_t^\tau|\mathbf{A}_t)\|^2$$

变量解释：
- $\tau \in [0,1]$: flow matching timestep，控制噪声水平（0=纯噪声，1=clean action）
- $\mathbf{A}_t \in \mathbb{R}^{H \times D}$: target action chunk，H=chunk size, D=action dimension
- $\mathbf{A}_t^\tau = \tau \mathbf{A}_t + (1-\tau)\epsilon$: noisy action，通过线性插值
- $\epsilon \sim \mathcal{N}(0, \mathbf{I})$: 标准高斯噪声
- $\mathbf{v}_\theta(\mathbf{A}_t^\tau, \mathbf{o}_t)$: 网络预测的vector field（速度场）
- $\mathbf{u}(\mathbf{A}_t^\tau|\mathbf{A}_t) = \epsilon - \mathbf{A}_t$: target vector field（指向clean action的方向）

Inference使用forward Euler integration从τ=0积分到τ=1：

$$\mathbf{A}_t^{\tau+\delta} = \mathbf{A}_t^\tau + \delta \mathbf{v}_\theta(\mathbf{A}_t^\tau, \mathbf{o}_t)$$

- $\delta$: integration step size，论文中δ=10或2
- 从 $\mathbf{A}_t^0 \sim \mathcal{N}(0, \mathbf{I})$ 开始

**MLP Head**使用parallel decoding（类似OpenVLA-OFT [15]）：

$$\mathbf{H}_t = f_\phi([\mathbf{o}_t, \ell, S])_S$$
$$\hat{\mathbf{A}}_t = g_\psi(\mathbf{H}_t)$$

- $S$: extra special query action tokens
- $f_\phi$: VLM backbone
- $g_\psi$: MLP head
- Loss: $\mathcal{L}_{\text{MLP}} = \mathbb{E}\|\hat{\mathbf{A}}_t - \mathbf{A}_t\|_1$ (L1 loss)

### 2.3 KV-Conditioned Self-Attention

这是连接VLM和action head的关键机制。VLM的prefix context被inject为past keys/values进入decoder-only action head，使得suffix tokens (action + state)可以attend to cached prefix和自己的block。这避免了重新计算VLM representations。

## 3. 核心创新：Budget-Aware Adaptive Inference

### 3.1 Multi-Exit Training

训练时随机采样layer index $i \sim \mathcal{U}(0, L)$，在第i层的hidden state上监督action loss。这训练了所有exit points的action predictor。

对于FM head：LLM执行到第i层，FM head也执行对应的i层，计算loss。
对于MLP head：直接从第i层的hidden state预测action。

### 3.2 Early-Termination via Action-Consistency

公式6定义了exit criterion：

$$\Delta_t^i = d(\mathbf{A}_t^{(i)}, \mathbf{A}_t^{(i-1)}) < \eta_i$$

- $\mathbf{A}_t^{(i)}$: timestep t时第i层预测的action chunk
- $d(\cdot, \cdot)$: discrepancy metric，可选cosine similarity / L2 / MAD
- $\eta_i$: 第i层的exit threshold
- 当连续两层action差异足够小时，early exit

### 3.3 Threshold Calibration

这是非常elegant的工程化设计。给定训练集 $\mathcal{D}$，做一次forward pass收集所有layer-wise discrepancies：

$$V \in \mathbb{R}^{K \times N}, \quad V[k, n] \equiv \Delta_{t_n}^{i_k}$$

- $K$: exit points数量（如K=14，每2层一个exit）
- $N$: 训练样本总数
- $V[k, n]$: 第n个样本在第k个exit point的action discrepancy

然后定义target exit distribution $\mathbf{p} = (p_1, ..., p_K)$。使用exponential distribution：

$$p_k \propto \rho^k, \quad \rho = c > 0$$

- $c = \text{exit\_criterion}$: 控制参数
- $c$越小，越倾向于early exit（更激进的加速）
- $c$越大，越倾向于late exit（更保守）

Threshold通过filtered-quantile procedure设置：

$$\eta_{i_k} = Q_{p_k}(\{V[k,n]\}_{n \in \mathcal{T}}), \quad \mathcal{T} \gets \{n \in \mathcal{T} | V[k,n] > \eta_{i_k}\}$$

- $Q_{p_k}$: $p_k$-quantile operator
- $\mathcal{T}$: 未被分配到earlier exit的样本集合
- 这确保了disjoint assignment across exits

**Intuition**: 这个calibration本质上是在说："我希望x%的样本在第k层exit，那么我把threshold设为使得x%的剩余样本的discrepancy低于它"。

### 3.4 Inter-Layer Truncated Flow Matching

这是处理FM head的关键创新。问题：early exit时每个exit layer都要跑FM denoising（10步），这shifts cost from backbone to action head。

**解决方案**: 
1. 减少每层denoising steps到δ=2
2. Warm-start: 下一层denoising从上一层输出开始

公式10:

$$\mathbf{A}_t^{0(i+1)} = \mathbf{A}_t^{1(i)}$$

- $\mathbf{A}_t^{0(i+1)}$: 第i+1层denoising的初始条件（τ=0）
- $\mathbf{A}_t^{1(i)}$: 第i层denoising的最终输出（τ=1）

**为什么不从random noise开始？** Intuition来自flow matching的几何理解：
- Standard FM: 从 $\mathcal{N}(0, \mathbf{I})$ 到 $p(\mathbf{A}_t|\mathbf{o}_t)$，需要完整trajectory
- Warm-start: 第i层已经approximate了target distribution，第i+1层只需要refine
- 这相当于把单次长trajectory分解成多次短trajectory，每次都warm-start

**数学motivation**: Flow matching的vector field $\mathbf{v}_\theta$学习的是从 $\mathcal{N}(0, \mathbf{I})$ 到 $p(\mathbf{A}_t|\mathbf{o}_t)$ 的transport map。如果 $\mathbf{A}_t^{1(i)}$ 已经接近 $p(\mathbf{A}_t|\mathbf{o}_t)$，那么从 $\mathbf{A}_t^{1(i)}$ 到target的transport map更简单，fewer steps足够。

**深层intuition**: 这其实利用了layer-wise coupling。第i层的FM output已经是基于partial VLM features的approximate action。第i+1层有更丰富的features，只需要在已有approximation基础上refine，而不是从头来。这是一种"coarse-to-fine"的hierarchical denoising。

## 4. 训练细节

### 4.1 Two-Stage Pipeline

**Stage 1 - Pretraining**:
- 数据: DROID, AgiBot, RoboCOIN, RoboMind, GM-100, RoboChallenge + 15,951 in-house trajectories
- 硬件配置: 多机器人平台（ARX, Franka, UR5, AgiBot）
- 关键: 不做state normalization，保留每个embodiment的intrinsic action space

**Stage 2 - Fine-tuning**:
- 任务特定的小数据集
- LIBERO: batch 128, 50K steps
- RoboChallenge: batch 64, 100K steps (Aloha)

### 4.2 Optimization

```
Optimizer: AdamW
Batch size: 1024 (pretrain)
Total steps: 200K

Learning Rates:
  - ViT backbone: 0 (frozen)
  - VLM components: 5e-6
  - Action head: 5e-5  (higher LR for rapid adaptation)

Schedule:
  - Warmup: 2000 steps (linear)
  - First 1000 steps: VLM frozen (zero LR)
  - Then: cosine annealing
```

**为什么action head LR比VLM高10倍？** Action head是from scratch训练（除了FM用Qwen3初始化），需要rapid adaptation到motor control；VLM已经有good representations，只需要轻量adaptation。

### 4.3 Data Augmentation

- **Visual**: Random erasing, Sharpening（防overfitting到background）
- **State**: Random masking (prob=0.5)（防partial observability）
- **Filtering**: 去除static frames和low-velocity segments

### 4.4 Hierarchical Sampling

```
Dataset level: 等概率采样每个dataset
Embodiment level: 每个dataset内等概率采样不同robot morphology
```

这防止单一data source或robot dominate训练。

## 5. 实验结果深度分析

### 5.1 Simulation Results (Table 1)

| Model | LIBERO Avg | VLABench Avg |
|-------|-----------|--------------|
| π0.5 | 96.9 | 49.5 |
| **A1** | **96.6** | **53.5** |
| π0 | 94.2 | 42.0 |
| SmolVLA | 88.8 | - |

A1在LIBERO上接近π0.5，在VLABench上超越π0.5 4%，特别在reasoning-heavy的Painting任务上达到70%（vs π0.5的44%）。

### 5.2 RoboChallenge Results (Table 4)

| Model | Mean Success |
|-------|-------------|
| DM0 (closed) | 62.00 |
| Spirit-v1.5 | 51.00 |
| GigaBrain | 51.67 |
| π0.5 | 42.67 |
| **A1 (open)** | **29.00** |
| π0 | 28.33 |
| X-VLA | 21.33 |
| RDT-1B | 15.00 |

A1在open-source models中SOTA，显著超越π0、X-VLA、RDT-1B。关键任务表现：
- Open Drawer: 100%
- Put Cup on Coaster: 90%
- Stack Blocks: 60%

### 5.3 Efficiency Analysis (Tables 5, 6)

**A1-MLP on LIBERO**:

| Config | Avg Success | TFLOPs | Inf Time |
|--------|------------|--------|----------|
| no exit (full) | 95.8 | 243.0 | 17.5s |
| c=1.0 | 96.6 | 205.0 (15.6%↓) | 16.5s |
| c=0.7 | 96.3 | 148.1 (39.1%↓) | 6.8s |
| c=0.4 | 94.0 | 100.8 (58.5%↓) | 5.6s |
| c=0.1 | 92.3 | 57.0 (76.6%↓) | - |

**关键发现**: 减76.6% computation只降1.7% success rate！甚至c=0.7时Spatial任务达到最高98.4%，说明less computation sometimes better。

**A1-FM on LIBERO**:

| Config | Avg Success | TFLOPs | Inf Time |
|--------|------------|--------|----------|
| no exit (δ=10) | 96.0 | 229.8 | 37.8s |
| 1.0, 10 | 96.4 | 150.6↓ | 40.9s (↑7.9%) |
| 1.0, 2 | 95.4 | 167.9↓ | 27.5s (↓27.4%) |
| **1.0, 2*** (warm-start) | **96.4** | 156.8↓ | **10.5s (↓72.3%)** |
| 0.8, 2* | 94.6 | 116.8↓ | 9.0s (↓76.3%) |

**Critical insight**: 
- δ=10 with early exit反而slower (40.9s vs 37.8s)，因为cost shift到FM head
- **Inter-Layer Truncated FM (warm-start)是enabler**: 37.8s → 10.5s，maintain 96.4% success
- Warm-start还提升accuracy: 95.4% → 96.4%

### 5.4 Real-World Results (Table 2)

跨4个机器人平台，12个任务：

| Model | UR5 | Franka | AgiBot | OpenArm | Dobot | Mean |
|-------|-----|--------|--------|---------|-------|------|
| π0 | 90 | 36.7 | 20 | 15 | 40 | 40.8 |
| π0.5 | 90 | 33.3 | 50 | 45 | 20 | 47.5 |
| **A1** | **80** | **46.7** | **80** | **50** | **25** | **56.7** |

A1在fine manipulation (AgiBot pick glue 80% vs 60%/30%)和long-horizon tasks表现强。Few-shot能力：50 samples on "fruits (small)"达到50% success。

### 5.5 Generalization (Table 7)

LIBERO-Plus zero-shot transfer (distribution shift):

| Method | Avg |
|--------|-----|
| OpenVLA-OFT | 69.6 |
| **A1-FM** | **75.3** |
| π0-FAST | 61.6 |
| π0 | 53.6 |
| OpenVLA | 15.6 |

A1在significant distribution shift下仍达75.3%，超越所有baselines。

## 6. Adaptive Inference的可视化理解 (Figure 4)

LIBERO-Long任务"turn on the stove and put the moka pot on it"：

- **简单动作**（如movement）: 模型在layer 3或5 early exit
- **复杂动作**（如turn on stove, pick up pot）: 模型进入deeper layers (17或25)

这说明模型learned to adaptively select effective features based on action complexity。

## 7. Intuition Building: 为什么这些方法work？

### 7.1 为什么Early Exit Work？

**Layer-wise feature hierarchy**: VLM的浅层编码low-level visual features（边缘、纹理、spatial layout），深层编码high-level semantics。对于很多actions（如简单movement），浅层features已经足够。只有需要precise semantic reasoning的actions才需要深层。

**类比**: 就像人类执行简单动作不需要full conscious processing，复杂动作才需要deliberate reasoning。

### 7.2 为什么Warm-Start Work？

**Flow matching的几何视角**: 
- Standard FM学习从 $\mathcal{N}(0, \mathbf{I})$ 到 $p(\mathbf{A}_t|\mathbf{o}_t)$ 的continuous transport map
- 这个map在action space中是smooth的
- 第i层的 $\mathbf{A}_t^{1(i)}$ 是基于partial features的approximation，已经接近target distribution
- 第i+1层只需要学一个"correction transport map"：从 $p_i(\mathbf{A}_t|\mathbf{o}_t^{(i)})$ 到 $p_{i+1}(\mathbf{A}_t|\mathbf{o}_t^{(i+1)})$
- 这个correction map更简单，fewer steps足够

**数学上**: 如果 $p_i$ 和 $p_{i+1}$ 的Wasserstein距离小（因为features只refined），transport map的Lipschitz常数小，Euler integration的step size可以更大。

### 7.3 为什么FM Trajectory Convergence快？

π0的FM使用beta distribution采样τ，emphasize lower (noisier) timesteps。这意味着网络在noisy regime训练更充分。推理时，trajectory在前期就lock onto mode，后期只是refinement。

### 7.4 VLM的Affordance Prior

Molmo预训练提供了implicit affordance priors。这意味着VLM的intermediate features已经encode了"哪里可以抓"、"如何approach"等信息，action head只需要decode这些priors，不需要从头learn。

## 8. 关键技术贡献总结

1. **Joint acceleration**: 同时加速backbone和action head，避免cost shifting
2. **Inter-Layer Truncated FM**: Warm-start denoising across layers，enabling aggressive step reduction
3. **Budget-aware calibration**: Exponential distribution + filtered-quantile，将compute budget转化为exit probability
4. **Multi-exit training**: 训练时随机exit layer，确保所有exit points都有action prediction能力
5. **Fully transparent**: Open-source full stack（code, data, checkpoints, eval）

## 9. Limitations & Future Work

1. **依赖labeled affordance data**: 未来探索unsupervised affordance mining
2. **Imitation learning only**: 累积误差，未来可加RL
3. **Calibration overhead**: 需要一次forward pass收集discrepancies（但相比inference加速是小代价）
4. **Cloud-server latency**: 异步执行可改善smoothness
5. **Dual-arm platform**: 已部署到8-DoF per arm的mobile platform

## References

- [GitHub](https://github.com/ATeam-Research/A1)
- [Project Page](http://www.ateam.xin/#/research/A1)
- [π0 paper](https://arxiv.org/abs/2410.24164)
- [π0.5 paper](https://arxiv.org/abs/2504.16054)
- [Molmo](https://arxiv.org/abs/2409.17146)
- [DeeR-VLA](https://arxiv.org/abs/2411.02359)
- [Flow Matching](https://arxiv.org/abs/2210.02747)
- [OpenVLA-OFT](https://arxiv.org/abs/2502.19645)
- [LIBERO](https://arxiv.org/abs/2306.03310)
- [RoboChallenge](https://arxiv.org/abs/2510.17950)
- [Pivot-R](https://arxiv.org/abs/2410.16194)

---

**最后intuition总结**: A1的精髓在于"spend compute only when it changes the action"。通过multi-exit training让每个layer都能predict action，通过action-consistency test决定何时stop，通过Inter-Layer Truncated FM让denoising也能hierarchical warm-start。这本质上是一种adaptive computation在robotic control中的应用，类似Universal Transformers的思想但specifically designed for VLA的bottleneck structure。
