我来帮你解析这篇文章。



我来详细讲解这篇 **Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning** 的文章。

## 论文概览

这篇论文来自 **NVIDIA** 和 **Stanford University**，提出了 **Cosmos Policy** —— 一种将大型预训练 **video generation model**（Cosmos-Predict2-2B）通过单阶段fine-tuning转化为有效 **robot policy** 的方法。核心思想是利用pretrained video models的 **spatiotemporal priors**，直接在 **latent diffusion process** 中生成 **robot actions**、**future state images** 和 **values**，无需任何架构修改。

## 核心创新：Latent Frame Injection

### 技术原理

**Cosmos-Predict2** 原本是 **latent video diffusion model**，采用 **Wan2.1 spatiotemporal VAE tokenizer** 进行编码。训练使用 **EDM denoising score matching formulation**，核心训练目标函数为：

$$\mathcal{L}(D_{\theta},\sigma) = \mathbb{E}_{\mathbf{x}_{0},\mathbf{c},\mathbf{n}}\left[\left\|D_{\theta}(\mathbf{x}_{0}+\mathbf{n};\sigma,\mathbf{c})-\mathbf{x}_{0}\right\|_{2}^{2}\right]$$

**公式变量详解**：
- $\mathbf{x}_{0}$：clean VAE-encoded image sequence（清洁的VAE编码图像序列）
- $\mathbf{c}$：textual description encoded as T5-XXL embeddings（文本描述的T5-XXL嵌入）
- $\mathbf{n} \sim \mathcal{N}(\mathbf{0},\sigma^{2}\mathbf{I})$：i.i.d. Gaussian noise（独立同分布的高斯噪声，用于corrupt $\mathbf{x}_{0}$）
- $D_{\theta}$：diffusion transformer denoiser network（扩散Transformer去噪网络）
- $\sigma$：noise level（噪声级别）
- $\|\cdot\|_{2}^{2}$：L2范数的平方（欧几里得距离的平方）

**Wan2.1 tokenizer** 压缩率：视频序列 $(1+T) \times H \times W \times 3$ 压缩到 latent 序列 $(1+T') \times H' \times W' \times 16$，其中：
- $T' = T/4$（时间压缩）
- $H' = H/8$（高度压缩）
- $W' = W/8$（宽度压缩）

### Latent Frame Injection实现

对于具有两个静态第三人称相机和腕部相机的机器人平台，latent sequence 包含 **11个latent frames**：

1. Blank placeholder（空白占位符）
2. Robot proprioception（机器人本体感知，如末端执行器位姿或关节角）
3. Wrist camera image（腕部相机图像）
4. First third-person camera image（第一个第三人称相机图像）
5. Second third-person camera image（第二个第三人称相机图像）
6. Action chunk（动作序列块）
7. Future robot proprioception（未来机器人本体感知）
8. Future wrist camera image（未来腕部相机图像）
9. Future first third-person camera image（未来第一个第三人称相机图像）
10. Future second third-person camera image（未来第二个第三人称相机图像）
11. Future state value（未来状态值）

其中，(2)、(6)、(7)、(11) 代表 **new modalities**，(3)、(5)、(8)、(10) 代表 **additional camera views**。

将新的modalities编码为latent frames的方法：将每个 $H' \times W' \times C'$ 的latent volume填充为归一化并重复复制的robot proprioception、action chunk或value（归一化将值rescale到 $[-1, +1]$）。

这个排序代表 $(s, a, s', V(s'))$，允许从左到右自回归解码actions、future state和future state value。

## MDP Formulation and Imitation Learning

### MDP定义

机器人操控任务被定义为 **finite-horizon Markov decision processes (MDPs)**，由元组 $\langle S, A, T, R, H \rangle$ 定义：

**符号详解**：
- $S$：set of states（状态集合）
- $A$：set of actions（动作集合）
- $T: S \times A \rightarrow \Pi(S)$：state transition function（状态转移函数）
- $R: S \times A \rightarrow \mathbb{R}$：reward function（奖励函数）
- $H \in \mathbb{N}$：time horizon（时间范围）
- $t \in \{1, 2, \ldots, H\}$：time steps（时间步）

### 奖励函数设计

使用 **sparse rewards**：
- $R(s_{t}, a_{t}) = 0$ for $t < H$（在终止时间步之前奖励为0）
- $R(s_{H}, a_{H}) \in [0, 1]$（终止时间步的奖励在0到1之间）

### Value Function定义

Policy $\pi$ 在状态 $s$ 的value function定义为：

$$V^{\pi}(s) = \mathbb{E}_{\tau \sim \pi}\left[\sum_{k=t}^{H}\gamma^{k-t}R(s_{k},a_{k}) \mid s_{t}=s\right] = \mathbb{E}_{\tau \sim \pi}\left[\gamma^{H-t}R(s_{H},a_{H}) \mid s_{t}=s\right]$$

**符号详解**：
- $V^{\pi}(s)$：policy $\pi$ 在状态 $s$ 的value（期望折现回报）
- $\tau$：trajectory（轨迹）
- $\mathbb{E}_{\tau \sim \pi}$：under policy $\pi$ 的期望（在策略$\pi$下的数学期望）
- $\gamma$：discount factor（折扣因子，通常在0到1之间）
- $\gamma^{H-t}$：从时间步 $t$ 到终止时间步 $H$ 的折扣系数

在稀疏奖励设置中，直接使用 **Monte Carlo returns**，每个transition标记为observed return $\gamma^{H-t}R(s_{H},a_{H})$。

## Joint Training of Policy, World Model & Value Function

### 训练数据分配

每个训练step采样一批 $(s, a, s', V(s'))$ tuples。**50%的batch** 来自 **demonstrations dataset**，用于训练policy $p(a, s', V(s')|s)$；**另外50%** 来自 **rollouts dataset**，平分为两半：一半训练world model $p(s', V(s')|s, a)$，另一半训练value function $p(V(s')|s, a, s')$。

### Conditioning Scheme

**Conditioning scheme**（条件方案）决定了latent diffusion sequence的哪部分用作conditioning，哪部分用作target来生成，从而决定正在训练三种函数中的哪一种。

### Auxiliary Supervision

Policy和world model训练涉及 **auxiliary targets**：
- Policy训练 $p(a, s', V(s')|s)$ 而非仅 $p(a|s)$
- World model学习 $p(s', V(s')|s, a)$ 而非仅 $p(s'|s, a)$

**Auxiliary supervision improves policy performance**（辅助监督提高了策略性能）。

### Parallel vs Autoregressive Decoding

由于Cosmos Policy学习 jointly 和 conditionally 预测targets $(a, s', V(s'))$，它可以 **parallel（并行）** 或 **autoregressive（自回归）** 地生成actions、future states和values：

- **Parallel decoding**：速度更快，适合直接policy评估（无需planning）
- **Autoregressive decoding**：提供更高质量的预测，适合planning场景

## Model-Based Planning with Cosmos Policy

### Dual Deployment

Cosmos Policy通过 **dual deployment** 实现model-based planning：

1. **Policy model**：原始Cosmos Policy checkpoint，用于生成action proposals
2. **Planning model**：fine-tuned checkpoint，用于world model和value function预测

### Best-of-N Sampling流程

1. 从policy model采样 **multiple action proposals**（多个动作提案）
2. 使用planning model为每个proposal预测 **future state** 和 **value**
3. 选择并部署导致 **highest predicted value**（最高预测值）的action

### Ensemble策略

为了更高的准确性和更好的建模：
- **World model queries**：每个action查询3次
- **Value function queries**：每个future state查询5次
- **Total value predictions**：每个action proposal共15个value predictions

**Aggregation method**："majority mean"（多数均值）：
- 确定majority预测成功或失败（通过固定阈值）
- 在majority group内平均values

这种方法比在value predictions呈现bimodal或高方差时的简单averaging更robust。

## 实验结果

### LIBERO Simulation Benchmark

**LIBERO benchmark** 包含四个主要任务套件：
- LIBERO-Spatial（空间布局）
- LIBERO-Object（物体）
- LIBERO-Goal（语言指定目标）
- LIBERO-Long（长视距任务）

| Method | Spatial | Object | Goal | Long | Average |
|--------|---------|--------|------|------|---------|
| Diffusion Policy | 78.3% | 92.5% | 68.3% | 50.5% | 72.4% |
| π₀ | 96.8% | 98.8% | 95.8% | 85.2% | 94.2% |
| π₀.₅ | 98.8% | 98.2% | 98.0% | 92.4% | 96.9% |
| CogVLA | 98.6% | 98.8% | 96.6% | 95.4% | 97.4% |
| **Cosmos Policy (ours)** | 98.1% | **100.0%** | 98.2% | 97.6% | **98.5%** |

**Cosmos Policy achieves state-of-the-art 98.5% average success rate**（Cosmos Policy取得了最先进的98.5%平均成功率）。

### RoboCasa Simulation Benchmark

**RoboCasa benchmark** 包含24个静态厨房操控任务，使用 **50 human-teleoperated demonstrations**（相较于其他方法使用300+ demos，数据效率更高）。

| Method | Demos | Average SR |
|--------|-------|------------|
| GR00T-N1 | 300 | 49.6% |
| Video Policy | 300 | 66.0% |
| FLARE | 300 | 66.4% |
| **Cosmos Policy (ours)** | 50 | **67.1%** |

**Cosmos Policy achieves state-of-the-art 67.1% with only 50 demos**（Cosmos Policy仅用50个演示就达到了67.1%的最先进平均成功率）。

### Real-World ALOHA Robot Tasks

**ALOHA platform** 特点：
- 两个ViperX 300S机器人臂
- 三个相机：一个top-down和两个wrist-mounted
- 控制频率：50 Hz → 25 Hz（为了计算效率）
- Action chunk：50 timesteps（2秒）

**四个挑战性bimanual manipulation tasks**：
1. "Put X on plate"（80 demos）：基于语言指令将物品放在盘子上
2. "Fold shirt"（15 demos）：折叠T恤，多步骤，long-horizon contact-rich manipulation
3. "Put candies in bowl"（45 demos）：收集分散的糖果，multimodal grasp sequences
4. "Put candy in ziploc bag"（45 demos）：打开并将物品放入自封袋，high-precision manipulation with millimeter tolerance

**Cosmos Policy achieves highest overall score 93.6%**，在三个任务中outperform所有其他方法。

### Model-Based Planning Results

在最后两个更具挑战性的ALOHA任务中：

| Variant | Score Improvement |
|---------|-------------------|
| Base Cosmos Policy (no planning) | baseline |
| Model-based (V(s')) | **+12.5 points average** |
| Model-free (Q(s,a)) | lower than model-based |

**Model-based planning using V(s') consistently improves success rates**（使用V(s')的model-based planning持续提高成功率）。

## 关键优势与局限

### 优势
1. **Single-stage fine-tuning**：无需多阶段训练
2. **No architectural modifications**：保持原有架构
3. **Leverages pretrained spatiotemporal priors**：利用预训练的时空先验
4. **Multimodal support**：支持多种modalities和camera views
5. **Enables model-based planning**：支持基于模型的规划
6. **Data efficiency**：仅需50个demos就在RoboCasa上达到SOTA

### 局限
1. **Inference speed**：Model-based planning大约需要5秒生成一个action chunk
2. **Rollout data requirement**：Effective planning需要substantial rollout data
3. **Limited search depth**：Focus on best-of-N with one layer in search tree

## 相关工作对比

### Video-based Robot Policies
- **UVA, Video Policy**：先fine-tune video models再训练separate action modules
- **UWM**：Train unified video-action models但don't leverage pretrained video models
- **Cosmos Policy**：Single-stage fine-tuning directly adapts pretrained video models

### Vision-Language-Action Models
- **RT-2, OpenVLA, π₀.₅, UniVLA, CogVLA**：Fine-tune vision-language models on large-scale robotic imitation data
- 这些方法主要在 **static image-text pairs** 上训练
- **Cosmos Policy**：利用学习过spatiotemporal dynamics和implicit physics的pretrained video model

### World Models and Value Functions
- **Dyna, MBPO, TD-MPC, Dreamer family**：Learned dynamics models for planning
- **FLARE**：Adds learnable future tokens to diffusion transformer sequences
- **SAILOR**：Uses separate world and reward models with MPPI planning
- **Latent Policy Steering**：Pretrains world models using optical flow
- **Cosmos Policy**：Single unified architecture serving as policy, world model, and value function simultaneously

## 技术细节补充

### Ablation Studies
1. **Removing auxiliary losses**：平均成功率下降1.5%（absolute）
2. **Training from scratch**：平均成功率下降3.9%
3. **ALOHA "fold shirt" task**：From-scratch variant score 80.8 vs full Cosmos Policy 99.5（18.7 points lower）

### Future State Prediction Improvement

**Base Cosmos Policy**（仅训练在demonstrations上）：
- World model可能fail预测errors如losing grasp of ziploc bag slider

**Fine-tuned checkpoint**（训练在policy rollout数据上）：
- World model更准确地预测resulting state
- Enables more effective planning和eventual episode success

### Common Failure Modes of VLAs
1. **π₀.₅**：
   - Struggles to handle ziploc bag
   - Often misses initial grasp of slider with right arm
   - Doesn't grasp left side of bag securely enough with left arm

2. **OpenVLA-OFT+**：
   - Often reaches in between two candies rather than directly for one
   - L1 regression of actions leads to inaccurate modeling of action distribution in high-multimodality tasks

## Intuition Building

### 为什么Video Models适合Robotics？

**Pretrained video generation models** 从数百万videos中学习：
- **Temporal causality**（时间因果关系）
- **Implicit physics**（隐式物理规律）
- **Motion patterns**（运动模式）

这些 **spatiotemporal priors** 对robotics applications非常valuable，因为：
1. Robot control需要理解 **temporal dynamics**
2. Physical interactions需要 **physics understanding**
3. Motion planning需要 **coherent motion patterns**

### 为什么Latent Frame Injection有效？

**Video models** 擅长建模 **complex, high-dimensional, multimodal distributions**：
- 能生成 **temporally coherent videos**（数百frames）
- Learning algorithms适合representing actions alongside other modalities

**Latent Frame Injection** 的核心intuition：
1. 将robot actions, future states, values都视为 **"frames"** 在latent space
2. Model原本就擅长处理temporal sequences of latent frames
3. 无需修改architecture就能处理新的modalities

### 为什么Model-Based Planning有效？

**Demonstrations alone** 的局限性：
- Data仅covers successful outcomes
- World model和value function看到 **narrow state-action distribution**
- May struggle to generalize beyond demonstration distribution

**Rollout data的价值**：
- Provides **diverse experiences**（包括failures）
- Enables learning **accurate world model** beyond demonstration distribution
- Improves **value function accuracy** for better planning

**Dual deployment的优势**：
- Policy model保持trained on demonstrations（高质量baseline）
- Planning model learns from on-policy rollouts（refined world model）
- Avoids catastrophic forgetting of successful behaviors

### 为什么V(s')优于Q(s,a)？

**Model-based (V(s'))**：
- Leverages learned **environment dynamics** via world model
- More sample-efficient planning
- Can predict multiple possible futures（handles multimodality）

**Model-free (Q(s,a))**：
- Requires accurate Q-function learning
- Higher input dimensionality leads to overfitting with limited data
- Cannot leverage learned dynamics

## Reference Links

相关论文和项目链接：

1. **Cosmos Policy Official Website**: https://research.nvidia.com/labs/dir/cosmos-policy/
2. **Cosmos-Predict2**: NVIDIA video foundation model (NVIDIA et al., 2025)
3. **Wan2.1 VAE**: Wan et al., 2025 - https://arxiv.org/abs/...
4. **EDM Denoising**: Karras et al., 2022 - https://arxiv.org/abs/2206.00364
5. **Diffusion Policy**: Chi et al., 2023 - https://arxiv.org/abs/2303.04137
6. **RT-2**: Brohan et al., 2023 - https://arxiv.org/abs/2307.15818
7. **OpenVLA**: Kim et al., 2024 - https://arxiv.org/abs/...
8. **π₀**: Black et al., 2024 - https://arxiv.org/abs/2410.24164
9. **LIBERO Benchmark**: Liu et al., 2024 - https://arxiv.org/abs/...
10. **RoboCasa Benchmark**: Nasiriany et al., 2024 - https://arxiv.org/abs/...
11. **ALOHA Platform**: Zhao et al., 2023 - https://arxiv.org/abs/2304.13705
12. **MBPO**: Janner et al., 2019 - https://arxiv.org/abs/1906.00269
13. **TD-MPC**: Hansen et al., 2022 - https://arxiv.org/abs/2203.04955
14. **Dreamer**: Hafner et al., 2019 - https://arxiv.org/abs/1912.01603

这篇论文展示了如何将 **pretrained video generation models** 的强大 **spatiotemporal priors** 转移到 **robot control** 任务中，通过创新的 **Latent Frame Injection** 技术和 **joint training** 策略，在多个benchmarks上取得了state-of-the-art性能。



我来详细为你讲解这篇关于WAN视频生成模型的论文。让我先读取更多关键内容来全面分析。












# WAN: Open and Advanced Large-Scale Video Generative Models 详细技术讲解

## 📊 一、论文核心概况

**WAN**是由阿里巴巴团队推出的**开源大规模视频生成模型系列**，包含1.3B和14B参数两个版本。这篇论文的核心贡献可以概括为以下**四个关键特征**：

| 特征 | 详细说明 |
|------|----------|
| **Leading Performance** | 14B模型在数十亿图像和视频数据上训练，展现出视频生成的scaling laws，在多个内部和外部基准测试中超越现有开源模型和商业模型 |
| **Comprehensiveness** | 提供1.3B（效率）和14B（效果）两个模型，涵盖8个下游任务：图像到视频、指令引导视频编辑、个性化视频生成等 |
| **Consumer-Grade Efficiency** | 1.3B模型仅需8.19GB VRAM，可在消费级GPU运行，且性能超越更大的开源模型 |
| **Openness** | 完整开源代码和模型，推动视频生成社区发展 |

### 官方资源
- GitHub: https://github.com/Wan-Video/Wan2.1
- 论文arXiv: arXiv:2503.20314v2

---

## 🏗️ 二、模型架构详解

### 2.1 整体架构流程图

WAN采用主流的**扩散Transformer (DiT)** 范式，结合**Flow Matching**框架，整体架构包含三大核心组件：

```
┌─────────────────────────────────────────────────────────────┐
│                     输入视频 V ∈ R^(1+T)×H×W×3                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                      Wan-VAE Encoder                         │
│  压缩: [1+T, H, W, 3] → [1+T/4, H/8, W/8, 16]                │
│  压缩比: 4×8×8 = 256倍                                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    Latent x ∈ R^(1+T/4)×H/8×W/8×16             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Video Diffusion Transformer (DiT)               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Patchify: 3D Conv kernel=(1,2,2)                        │  │
│  │ 输入形状: [B, L, D]                                      │  │
│  │ L = (1+T/4)×H/16×W/16 (序列长度)                         │  │
│  │ D = latent dimension (隐层维度)                          │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ N×Transformer Blocks:                                   │  │
│  │  • Self-Attention (时空注意力)                          │  │
│  │  • Cross-Attention (文本条件嵌入)                        │  │
│  │  • FFN (前馈网络)                                        │  │
│  │  • AdaLN (自适应归一化, 共享策略)                         │  │
│  └───────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                      Wan-VAE Decoder                         │
│  解码: [1+T/4, H/8, W/8, 16] → [1+T, H, W, 3]                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   输出视频 V' ∈ R^(1+T)×H×W×3                  │
└─────────────────────────────────────────────────────────────┘

并行输入:
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   umT5     │─────►│   时间步 t  │─────►│  用户提示  │
│  Text Encoder│     │  Timestep   │     │   Prompt   │
└─────────────┘     └─────────────┘     └─────────────┘
        │                  │                  │
        └──────────────────┴──────────────────┘
                           │
                           ▼
                  Cross-Attention嵌入到DiT中
```

### 2.2 Wan-VAE：时空变分自编码器

Wan-VAE是一个**3D因果VAE**架构，专门为视频生成设计，具有以下关键特性：

#### 核心设计公式

**输入-输出映射：**
```
输入视频: V ∈ R^(1+T)×H×W×3
压缩后latent: [1+T/4, H/8, W/8, C]  (C=16)
```

**压缩比分解：**
- **第一帧**: 仅空间压缩 (8×8=64倍)
- **后续帧**: 时空压缩 (4×8×8=256倍)

#### 特征缓存机制（Feature Cache）

为支持任意长度视频的高效编码/解码，Wan-VAE实现了特征缓存机制：

```
视频分块策略:
• 输入格式: 1+T (1个条件帧 + T个生成帧)
• 分块数: 1 + T/4 (与latent特征数一致)
• 每块处理帧数: ≤4 (防止内存溢出)

因果卷积缓存:
┌─────────────────────────────────────────────────┐
│ Chunk 0: [f0, f1, f2, f3]                        │
│ Cache: [padding, padding] → [f0, f1]            │
└─────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────┐
│ Chunk 1: [f4, f5, f6, f7]                        │
│ Cache: [f2, f3] → [f4, f5]                       │
└─────────────────────────────────────────────────┘
         ↓
    ...
```

#### Wan-VAE与其他VAE对比

| 模型 | 参数量 | 压缩比 | PSNR (720×720) | 效率 (帧/秒) |
|------|--------|--------|----------------|-------------|
| Wan-VAE | 127M | 4×8×8 | **38** | **40** |
| HunYuan Video | 246M | 4×8×8 | 30 | 32 |
| CogVideoX | 182M | 4×8×8 | 34 | 36 |
| SVD | 97M | 1×8×8 | 24 | 46 |
| Mochi | 239M | 6×8×8 | 32 | 34 |
| Open Sora Plan | 460M | 4×8×8 | 24 | 28 |
| Step Video | 499M | 8×16×16 | 30 | 24 |

**关键洞察：** Wan-VAE在保持高质量重建的同时，实现2.5倍于现有SOTA方法的重建速度，这得益于小模型设计和特征缓存机制。

### 2.3 Video Diffusion Transformer

#### Transformer Block结构

```
┌───────────────────────────────────────────────────────┐
│              V-Tokens (视频tokens)                       │
└───────────────────┬───────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────┐
│                  Layer Norm                             │
└───────────────────┬───────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│ Self-Attention│       │Cross-Attention│
│ (时空全注意力) │       │ (文本条件嵌入) │
└───────────────┘       └───────────────┘
        │                       │
        └───────────┬───────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────┐
│                  Layer Norm                             │
└───────────────────┬───────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────┐
│                      FFN                                 │
│              (前馈神经网络)                               │
└───────────────────────────────────────────────────────┘
```

#### Patchify模块详解

**卷积操作：**
```
输入: x ∈ R^(1+T/4)×H/8×W/8×16

3D Convolution:
  - kernel size: (1, 2, 2)
  - 第一个维度1: 保持时间维度不变
  - 后两个维度2: 空间下采样2×2
  
输出: [B, L, D]
  - B: batch size (批量大小)
  - L = (1+T/4)×H/16×W/16 (序列长度)
  - D: latent dimension (隐层维度)
```

#### 自适应归一化（AdaLN）创新

WAN采用**完全共享的AdaLN**设计，这是与PixArt的重要区别：

```
标准AdaLN (非共享):
Block 1: MLP(t) → [scale_1, shift_1]
Block 2: MLP(t) → [scale_2, shift_2]
  ...
Block N: MLP(t) → [scale_N, shift_N]
参数量: O(N × h)

共享AdaLN (WAN采用):
Block 1: MLP(t) → [scale_global, shift_global]
Block 2: 使用 [scale_global, shift_global]
  ...
Block N: 使用 [scale_global, shift_global]
参数量: O(h)  (减少约25%)
```

**消融实验结果：**

| 配置 | 模型深度 | 参数量 | 训练Loss |
|------|---------|--------|----------|
| Full-shared-adaLN-1.3B | 30层 | 1.3B | 0.42 |
| Half-shared-adaLN-1.5B | 30层 | 1.5B | 0.40 |
| Full-shared-adaLN-1.5B | 35层 | 1.5B | **0.38** ✓ |
| Non-shared-adaLN-1.7B | 30层 | 1.7B | 0.39 |

**结论：** 增加模型深度比增加AdaLN参数更有效，因此采用完全共享AdaLN设计。

### 2.4 文本编码器选择

WAN经过大量实验选择了**umT5 (5.3B)**作为文本编码器，原因如下：

#### 文本编码器对比

| 编码器 | 类型 | 参数量 | FID Score | 优势 |
|--------|------|--------|-----------|------|
| umT5 | 双向注意力 | 5.3B | **43.01** ✓ | 多语言能力、收敛快、组合能力强 |
| Qwen2.5-7B-Instruct | 因果注意力 | 7B | 43.72 | 语言理解强 |
| GLM-4-9B | 因果注意力 | 9B | 43.74 | 大模型能力 |
| Qwen-VL-7B (倒数第二层) | 多模态LLM | 7B | 42.91 | 视觉理解强但模型大 |

**关键洞察：** umT5采用双向注意力，更适合扩散模型（需要同时考虑所有token），而因果注意力LLM更适合自回归生成。

---

## 📚 三、数据处理流水线

### 3.1 预训练数据处理

WAN遵循三大核心原则：**高质量、高多样性、大规模**

#### 四步数据清洗流程

```
原始数据 → [步骤1: 基础维度过滤] → [步骤2: 视觉质量筛选] 
  → [步骤3: 运动质量评估] → [步骤4: 视觉文本数据处理] → 训练数据
```

#### 步骤1：基础维度过滤（Fundamental Dimensions）

| 过滤项 | 方法 | 目的 |
|--------|------|------|
| 文本检测 | 轻量级OCR检测器 | 排除文本过多的视频/图像 |
| 美学评估 | LAION-5B美学分类器 | 初步质量筛选 |
| NSFW评分 | 内部安全评估模型 | 过滤不当内容 |
| 水印/Logo检测 | 检测并裁剪 | 去除水印元素 |
| 黑边检测 | 启发式检测 | 自动裁剪黑边 |
| 过曝检测 | 专家分类器 | 过滤色调异常数据 |
| 合成图像检测 | 专家分类器 | 过滤AI生成图像（<10%污染即可显著降低性能） |
| 模糊检测 | 内部模糊评分模型 | 移除模糊内容 |
| 时长和分辨率 | 时长>4秒，分辨率阈值 | 确保基本质量 |

**效果：** 消除了约50%的初始数据集。

#### 步骤2：视觉质量筛选（Visual Quality）

**聚类策略：**
```
将数据分成100个集群
  ↓
从每个集群选择一定量数据
  ↓
避免长尾分布导致的小但重要数据段丢失
```

**评分策略：**
```
从每个集群采样 → 人工评分 (1-5分)
  ↓
训练专家评估模型 → 评分整个数据集
```

#### 步骤3：运动质量评估（Motion Quality）

将视频运动质量分为六个等级：

| 等级 | 特征 | 采样优先级 |
|------|------|-----------|
| **Optimal Motion** | 大运动布局、透视和幅度，运动流畅干净 | 高 |
| **Medium-quality Motion** | 明显运动但有小问题（多主体、部分遮挡） | 中 |
| **Static Videos** | 聊天、访谈类视频，少运动 | 低（单独处理） |
| **Camera-driven Motion** | 无人机镜头等相机主导运动 | 极低 |
| **Low-quality Motion** | 过多主体、严重遮挡、主次不清 | 排除 |
| **Shaky Camera Footage** | 业余录制，相机抖动严重 | 排除 |

#### 步骤4：视觉文本数据处理（Visual Text Data）

这是WAN的独特创新，使其成为首个能生成中英文本的模型：

```
┌─────────────────────────────────────────────────────────┐
│              视觉文本生成增强                             │
├─────────────────────────────────────────────────────────┤
│ 分支1: 合成数据                                          │
│  • 在纯白背景上渲染数百万中文字符图像                     │
│                                                         │
│ 分支2: 真实世界数据                                      │
│  • 收集大量含文本的图像/视频                              │
│  • 多个OCR模型识别中英文文本                              │
│  • 输入Qwen2-VL生成自然描述                              │
│  • 确保描述包含精确的文本内容                             │
└─────────────────────────────────────────────────────────┘
```

### 3.2 后训练数据处理

#### 图像处理

```
高质量图像池 → [专家模型收集 + 手动收集] → 精选数据集
  • 前20%基于专家模型评分
  • 考虑风格和类别平衡
  • 手动补充缺失概念
  → 数百万精选图像
```

#### 视频处理

```
候选视频 → [视觉质量分类器 + 运动质量分类器] → 精选视频
  • 数百万简单运动视频
  • 数百万复杂运动视频
  • 12大类平衡（科技、动物、艺术、人类、车辆等）
```

### 3.3 密集视频字幕生成（Dense Video Caption）

#### 模型架构

```
图像/视频输入
  ↓
ViT编码器 → 视觉嵌入
  ↓ (通过两层MLP投影)
Qwen LLM
  ↓
密集字幕输出
```

**动态高分辨率处理：**
```
图像: 最多分成7个patch，每个patch自适应池化到12×12网格表示
视频: 3帧/秒采样，上限129帧
  • 每4帧保持原分辨率
  • 其余帧全局平均池化
  • Slow-fast编码策略
```

**训练阶段：**
```
阶段1: 冻结ViT和LLM，训练MLP对齐 (LR=1e-3)
阶段2: 所有参数可训练 (LLM&MLP: LR=1e-5, ViT: LR=1e-6)
阶段3: 小规模高质量数据端到端训练
```

#### 评估结果（与Gemini 1.5 Pro对比）

| 维度 | Gemini 1.5 Pro | Ours | 优势方 |
|------|----------------|------|--------|
| Event (事件) | 52.6 | 97.6 | ✓ Ours |
| Action (动作) | 88.2 | 52.6 | ✓ Gemini |
| Camera Angle (相机角度) | 52.6 | 97.0 | ✓ Ours |
| Camera Motion (相机运动) | 41.4 | 79.6 | ✓ Ours |
| OCR (文本识别) | 88.0 | 84.3 | ✓ Gemini |
| Style (风格) | 59.5 | 87.5 | ✓ Ours |
| Scene (场景) | 89.1 | 72.1 | ✓ Gemini |
| Color (颜色) | 52.6 | 87.8 | ✓ Ours |
| Category (类别) | 97.6 | 55.4 | ✓ Gemini |
| Counting (计数) | 84.5 | 79.1 | ✓ Gemini |

---

## 🎯 四、模型训练策略

### 4.1 Flow Matching框架

WAN采用Flow Matching理论框架，而非传统的DDPM/DDIM。

#### 核心公式

**1. 噪声插值：**
```
给定:
  - 图像/视频latent: x₁
  - 随机噪声: x₀ ~ N(0, I)
  - 时间步: t ∈ [0,1] (从logit-normal分布采样)

中间latent定义为线性插值:
  x_t = t·x₁ + (1-t)·x₀      (公式1)
```

**2. 真实速度：**
```
v_t = dx_t/dt = x₁ - x₀       (公式2)
```

**3. 损失函数：**
```
L = E[x₀, x₁, c_txt, t] ||u(x_t, c_txt, t; θ) - v_t||²   (公式3)

其中:
  - c_txt: umT5文本嵌入序列 (512 tokens)
  - θ: 模型权重
  - u(x_t, c_txt, t; θ): 模型预测的速度
```

**Flow Matching优势：**
- 避免迭代速度预测
- 通过ODEs实现稳定训练
- 等价于最大似然目标

### 4.2 分阶段训练策略

#### 阶段1：低分辨率图像预训练

```
配置:
  • 分辨率: 256px
  • 任务: Text-to-Image
  • 目的: 建立跨模态语义对齐和几何结构保真度

原因:
  直接联合训练高分辨率图像和长视频序列面临两大挑战:
  1. 扩展序列长度(1280×720视频通常81帧)显著降低训练吞吐量
  2. GPU内存消耗过大导致批次大小不足，训练不稳定
```

#### 阶段2：图像-视频联合训练（渐进式课程）

```
子阶段2.1:
  • 图像: 256px
  • 视频: 192px, 5秒, 16fps
  • 目的: 引入视频模态

子阶段2.2:
  • 图像: 480px
  • 视频: 480px, 5秒
  • 目的: 空间分辨率扩展

子阶段2.3:
  • 图像: 720px
  • 视频: 720px, 5秒
  • 目的: 最终高质量训练
```

#### 训练配置

```
精度: BF16混合精度
优化器: AdamW
  • weight decay: 1e⁻³
  • 初始学习率: 1e⁻⁴
学习率调度: 基于FID和CLIP Score平台期动态降低
```

### 4.3 后训练（Post-training）

```
架构: 与预训练阶段相同
初始化: 预训练checkpoint
数据: 后训练视频数据集（Sec 3.2）
分辨率: 480px和720px联合训练
```

---

## 🚀 五、模型扩展与训练效率

### 5.1 工作负载分析

#### 计算成本分解

**DiT模型主导计算：**
```
主要计算来自DiT (>85%)，文本编码器和VAE编码器计算较少

DiT计算成本公式:
  Cost = L(α·b·s·h² + β·b·s²·h)

其中:
  - L: DiT层数
  - b: micro batch size (微批次大小)
  - s: 序列长度
  - h: hidden dimension (隐层维度)
  - α: 线性层成本系数
  - β: 注意力层成本系数
  - 非因果注意力下: β_forward = 4, β_backward = 8
```

**关键洞察：**
```
当序列长度达到1M token时:
  • 注意力计算占比: 高达95%
  • 线性层计算: 仅5%
  • 原因: 注意力成本O(s²)，线性层成本O(s)
```

#### GPU内存使用

```
DiT GPU内存使用公式:
  Memory = γ·L·b·s·h

其中:
  - γ: 取决于DiT层实现
  - 普通LLM: γ ≈ 34
  - DiT模型: γ > 60 (输入包括视频tokens、提示词、时间步)

示例 (1M tokens, batch=1, 14B模型):
  激活值内存 > 8TB
```

### 5.2 并行策略

#### DiT并行策略设计

```
┌─────────────────────────────────────────────────────────────┐
│                    128 GPU 示例配置                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  CP (Context Parallelism) = 16                              │
│    ├── Ulysses = 8 (内层)                                    │
│    └── Ring Attention = 2 (外层)                            │
│                                                             │
│  FSDP (Fully Sharded Data Parallel) = 32                    │
│  DP (Data Parallel) = 4                                      │
│                                                             │
│  全局批次大小 = 8 × micro-batch大小                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 2D Context Parallelism (CP)

**为什么选择CP而非TP+SP？**
```
CP通信开销 < (TP+SP)通信开销

2D CP设计 (类似USP):
  外层: Ring Attention
  内层: Ulysses
  
优势:
  • 减轻Ulysses的跨机器慢通信
  • 解决Ring Attention分片后的大块大小需求
  • 最大化外层通信与内层计算的重叠
```

**性能提升：**
```
配置: 256K序列长度, 16 GPU, 2台机器

仅Ulysses: 通信开销 > 10%
2D CP: 通信开销 < 1% ✓
```

#### 分布式策略切换

```
问题: 不同模块使用相同资源但策略不同
  • VAE和Text Encoder: DP
  • DiT: DP + CP

解决方案:
  1. CP组内设备最初读取不同数据
  2. CP前执行循环遍历(大小=CP size)
  3. 广播不同设备读取的数据确保CP输入一致
  4. VAE和Text Encoder时间比例降至1/CP
```

### 5.3 内存优化

**激活卸载 vs 梯度检查点：**
```
长序列场景下，计算时间 > PCIe传输时间

激活卸载优势:
  • 可重叠计算
  • 卸载1层DiT激活可与计算1-3层DiT重叠

组合策略:
  • 优先激活卸载
  • 对高内存/计算比层使用梯度检查点(GC)
```

### 5.4 集群可靠性

```
阿里云智能调度:
  • 启动阶段检测慢机器
  • 仅分配健康节点
  • 运行时故障节点自动隔离和修复
  • 任务自动重启无缝恢复
```

---

## ⚡ 六、推理优化

### 6.1 并行策略

#### 多GPU推理扩展

```
并行策略:
  • Context Parallelism: 减少单个视频生成延迟
  • Model Sharding (FSDP): 缓解大模型GPU内存限制

性能:
  Wan 14B模型在2D Context Parallel + FSDP下
  → 接近线性加速比
```

### 6.2 扩散缓存（Diffusion Cache）

**观察到的WAN推理特性：**
```
1. Attention相似性: 同一DiT块中，不同采样步的注意力输出高度相似
2. CFG相似性: 采样后期，条件和无条件DiT输出显著相似
```

**缓存策略：**

| 缓存类型 | 策略 | 细节 |
|---------|------|------|
| **Attention Cache** | 每隔几步执行注意力前向传播并缓存 | 重用缓存结果给其他步 |
| **CFG Cache** | 每隔几步执行无条件DiT前向传播 | 重用条件结果，应用残差补偿 |

**性能提升：**
```
Wan 14B Text-to-Video模型:
  推理性能提升: 1.62×
```

### 6.3 量化优化

#### FP8 GEMM

```
配置:
  • GEMM操作: FP8精度
  • 权重量化: per-tensor量化
  • 激活量化: per-token量化 (采样步中)
  
性能:
  FP8 GEMM速度 = 2× BF16 GEMM速度
  DiT模块整体加速: 1.13×
```

#### 8-bit FlashAttention优化

**问题：** FlashAttention3原生FP8实现在视频生成中质量下降严重

**解决方案：混合8-bit优化**

```python
# 混合精度策略
S = QKᵀT → Int8 (使用整数精度)
O = PV → FP8 (使用FP8精度)

# 跨块FP32累积
# FP8 WGMMA用于块内PV归约
# CUDA核FP32寄存器用于跨块累积
```

**优化技术：**

| 技术 | 目的 |
|------|------|
| FP32累积与intra-warpgroup流水线融合 | 减轻Float32 PV累积的性能损失 |
| Block size调优 | 减少跨块Float32累积的寄存器压力 |

**性能：**
```
NVIDIA H20 GPU上:
  • MFU: 95%
  • 推理效率提升: >1.27×
```

### 6.4 Prompt对齐

#### 策略1：多样性增强

```
每张图像/视频配对多个描述:
  • 不同长度: 长、中、短
  • 不同风格: 正式、非正式、诗意
  
目的: 覆盖用户可能提供的各种提示词范围
```

#### 策略2：LLM提示词重写

**重写原则：**
```
1. 添加细节但不改变原意
2. 融入自然运动属性
3. 结构: [视频风格] → [内容摘要] → [详细描述]
```

**示例对比：**

| 用户原始提示词 | LLM重写后 |
|--------------|----------|
| "A Viking warrior wields a great axe with both hands, battling a mammoth at dusk, amidst a snowy landscape with snowflakes swirling in the air." | "An epic battle scene, unfolds as a tall and muscular Viking warrior wields a heavy great axe with both hands, facing off against a massive mammoth. The warrior is clad in leather armor and a thorned helmet, with prominent muscles and a fierce, determined expression. The mammoth is covered in long hair, with sharp tusks, and roars angrily. It is dusk, and the snowy landscape is filled with swirling snowflakes, creating an intense and dramatic atmosphere. The backdrop features a barren ice field with the faint outlines of distant mountains. The use of cool-toned lighting emphasizes strength and bravery. The scene is captured in a dynamic close-up shot from a high-angle perspective." |

---

## 📊 七、评估基准与结果

### 7.1 Wan-Bench基准测试

WAN提出Wan-Bench，一个自动化、全面、人类对齐的视频生成评估基准。

#### 评估维度

```
Wan-Bench
├── Dynamic Quality (动态质量)
│   ├── Large Motion Generation (大运动生成)
│   ├── Human Artifacts (人工伪影检测)
│   ├── Physical Plausibility (物理合理性)
│   ├── Smoothness (平滑度)
│   ├── Pixel-level Stability (像素级稳定性)
│   └── ID Consistency (身份一致性)
├── Image Quality (图像质量)
│   ├── Comprehensive Image Quality (综合图像质量)
│   ├── Scene Generation Quality (场景生成质量)
│   └── Stylization Ability (风格化能力)
└── Instruction Following (指令遵循)
    ├── Single Object (单物体)
    ├── Multiple Objects (多物体)
    ├── Spatial Positions (空间位置)
    ├── Camera Control (相机控制)
    └── Action Instruction Following (动作指令遵循)
```

#### 关键评估方法

**1. 大运动生成：**
```
使用RAFT计算生成视频的光流
通过归一化光流幅度评估运动分数
```

**2. 人工伪影检测：**
```
训练YOLOv3模型在20,000个人工标注的AI生成图像上
识别伪影位置
伪影分数 = (预测概率 × 边界框 × 持续时间)的综合考量
```

**3. 物理合理性与平滑度：**
```
物理合理性: Qwen2-VL通过视频问答检测违反物理定律的情况
平滑度: Qwen2-VL识别运动中的伪影评估流畅性
```

**4. 身份一致性：**
```
三类子维度:
  • 人体一致性
  • 动物一致性
  • 物体一致性
  
方法: 提取帧级DINO特征，计算帧间相似度
```

**5. 综合图像质量：**
```
Fidelity (保真度): MANIQA
Aesthetics (美学): 
  • LAION-based aesthetic predictor
  • MUSIQ
最终分数 = 三个评估器的平均值
```

**6. 场景生成质量：**
```
场景一致性: 帧间CLIP相似度
场景文本对齐: 帧与对应文本的CLIP相似度
最终分数 = 加权平均
```

#### 人类反馈引导加权策略

```
收集: >5,000个不同模型生成视频的成对比较
方法: 人类根据文本评估配对，表达偏好并分配近似分数
权重: 模型生成分数与人类评分的Pearson相关系数作为权重因子
```

### 7.2 定量结果对比

#### Wan-Bench对比（分数越高越好）

| 模型 | 大运动 | 物理合理 | 平滑度 | 综合图像 | 相机控制 | 指令遵循 | **加权总分** |
|------|--------|---------|--------|---------|---------|---------|-------------|
| Sora | 0.482 | 0.933 | 0.930 | 0.665 | 0.380 | 0.721 | **0.700** |
| **Wan 14B** | **0.415** | **0.939** | **0.910** | 0.640 | 0.527 | 0.860 | **0.724** ✓ |
| Wan 1.3B | 0.468 | 0.912 | 0.790 | 0.596 | 0.483 | 0.844 | 0.689 |
| Mochi | 0.420 | 0.728 | 0.530 | 0.530 | 0.605 | 0.907 | 0.639 |
| Hunyuan | 0.413 | 0.898 | 0.890 | 0.605 | 0.406 | 0.735 | 0.673 |
| CN-Top A | 0.405 | 0.836 | 0.765 | 0.621 | 0.465 | 0.917 | 0.690 |
| CN-Top B | 0.284 | 0.759 | 0.880 | 0.668 | 0.529 | 0.783 | 0.693 |

#### Vbench Leaderboard对比

| 模型 | 质量分数 | 语义分数 | **总分** |
|------|---------|---------|---------|
| **Wan 14B** | **86.67%** | **84.44%** | **86.22%** ✓ |
| Sora | 85.51% | 79.35% | 84.28% |
| Wan 1.3B | 84.92% | 80.10% | 83.96% |
| Hunyuan (开源版) | 85.09% | 75.82% | 83.24% |
| MiniMax-Video-01 | 84.85% | 77.65% | 83.41% |
| Gen-3 | 84.11% | 75.17% | 82.32% |
| CogVideoX1.5-5B | 82.78% | 79.76% | 82.17% |
| Kling (高性能模式) | 83.39% | 75.68% | 81.85% |

#### 人类评估结果（胜率）

| 评估维度 | CN-Top A | CN-Top B | CN-Top C | Runway | Wan 14B |
|---------|---------|---------|---------|--------|---------|
| Visual Quality (视觉质量) | 30.6% | 15.9% | 27.8% | 48.1% | **57.1%** ✓ |
| Motion Quality (运动质量) | 16.1% | 9.7% | 14.9% | 40.3% | **57.9%** ✓ |
| Matching (匹配度) | 46.0% | 57.9% | 56.7% | 69.1% | **55.8%** ✓ |
| Overall Ranking (整体排名) | 44.0% | 44.0% | 48.9% | 67.6% | **67.6%** ✓ |

### 7.3 消融实验结果

#### VAE vs VAE-D对比

| 模型 | 10k steps | 15k steps |
|------|-----------|-----------|
| VAE (WAN采用) | **42.60** ✓ | **40.55** ✓ |
| VAE-D (扩散损失) | 44.21 | 41.16 |

**结论：** VAE模型在FID上始终优于VAE-D。

---

## 🎬 八、扩展应用

### 8.1 图像到视频生成（Image-to-Video）

#### 模型设计

```
输入:
  • 条件图像: I ∈ R^C×1×H×W
  • 零填充帧: 沿时间轴拼接
  • 指导帧: I_c ∈ R^C×T×H×W

编码:
  Wan-VAE压缩 → z_c ∈ R^c×t×h×w
  • c = 16 (latent通道)
  • t = 1+(T-1)/4
  • h = H/8
  • w = W/8

掩码:
  二进制掩码: M ∈ {0,1}^(1×T×h×w)
  • 1: 保留帧
  • 0: 生成帧

融合:
  噪声latent z_t + 条件latent z_c + 掩码 m
  → 沿通道轴拼接 → DiT模型

额外条件:
  CLIP图像编码器提取特征 → 三层MLP投影 → 全局上下文
  → 通过解耦交叉注意力注入DiT
```

#### 数据集策略

```
I2V数据集:
  计算第一帧与剩余帧的SigLIP特征余弦相似度
  保留相似度 > 阈值的视频
  
视频延续数据集:
  计算前1.5秒与后3.5秒的SigLIP特征余弦相似度
  选择时间一致性高的视频

首末帧转换数据集:
  增加首末帧有显著过渡的数据比例
```

#### I2V人类评估结果

| 评估维度 | CN-Top A | CN-Top B | CN-Top C | CN-Top D | Wan I2V |
|---------|---------|---------|---------|---------|---------|
| Visual Quality (视觉质量) | 29.2% | 60.8% | 24.6% | 55.6% | **89.6%** ✓ |
| Motion Quality (运动质量) | 21.7% | 21.7% | 32.5% | 67.0% | **89.0%** ✓ |
| Matching (匹配度) | -4.2% | 35.0% | 51.7% | 72.2% | **89.6%** ✓ |
| Overall Ranking (整体排名) | 10.8% | 47.5% | 50.8% | 81.6% | **89.0%** ✓ |

### 8.2 统一视频编辑（VACE框架）

#### Video Condition Unit (VCU)

```
V = [T; F; M]

其中:
  • T: 文本提示
  • F: 上下文视频帧序列 {u₁, u₂, ..., uₙ}
  • M: 掩码序列 {m₁, m₂, ..., mₙ}
  • u: RGB空间, 归一化到[-1,1]
  • m: 二进制, 1=编辑位置, 0=不编辑
```

#### 概念解耦策略

```
F_c = F × M    (reactive frames, 要修改的像素)
F_k = F × (1-M) (inactive frames, 要保留的像素)

目的:
  • 明确分离不同模态和分布的数据
  • 确保清晰的任务定义
  • 保证模型在不同任务上的收敛
```

#### 支持的任务

| 任务类型 | 说明 |
|---------|------|
| Outpainting (外补) | 扩展视频边界 |
| Extension (扩展) | 延长视频时长 |
| Depth (深度) | 深度图引导 |
| Pose (姿态) | 姿态引导 |
| Inpainting (内补) | 视频修复 |
| Gray (灰度) | 灰度化 |
| Scribble (涂鸦) | 涂鸦引导 |
| Layout (布局) | 布局引导 |
| Object (物体) | 物体替换 |
| Face (人脸) | 人脸编辑 |

### 8.3 文本到图像生成

WAN同时在图像和视频数据集上训练，图像数据集比视频数据集大约10倍。

**生成能力：**
- 艺术文本视觉效果
- 照片级肖像
- 想象力创意设计
- 专业级产品摄影

### 8.4 视频个性化

#### 模型设计

```
在Wan-VAE latent空间中:

1. 扩展K帧前置帧使用分割的人脸图像
   • 人脸图像来自配对视频的人脸检测和分割
   • 人脸关键点对齐到与视频帧相同大小的黑色画布

2. 沿通道维度拼接:
   • 前K帧: 人脸图像 + 全1掩码
   • 后续帧: 空白图像 + 全0掩码
   
3. 条件信号:
   扩展视频的通道条件信号
   
4. 扩散过程:
   在时间扩展的视频上执行扩散
   以inpainting方式条件化通道条件信号
```

**训练技巧：**
```
随机丢弃K扩展帧中的一部分人脸图像
→ 支持0到K参考人脸视频生成
```

#### 个性化数据集构建

```
步骤:
1. 从T2V基础模型训练数据中筛选 O(100)M 视频
2. 内部人类分类器过滤
3. 1 FPS人脸检测
   • 任一帧检测到多个人脸 → 丢弃
   • >10%帧无人脸 → 丢弃
4. 连续帧ArcFace相似度计算
   • 低相似度 → 丢弃
5. 人脸分割去除背景
6. 人脸关键点检测辅助画布对齐

结果: 约 O(10)M 个性化视频，每视频平均5个分割人脸
```

**自动数据合成增强：**
```
1. 从中随机选择 O(1)M 个性化视频
2. 使用Instant-ID合成多样人脸
3. 文本模板: 100+提示词 (动漫、线稿、电影、Minecraft等)
4. 随机提示词 + 随机姿态估计 → Instant-ID输入
5. ArcFace相似度过滤

结果: 约 O(1)M 合成人脸视频
→ 大幅提升个性化数据集的风格、姿态、光照和遮挡多样性
```

#### 评估结果

| 模型 | Arcface相似度 |
|------|--------------|
| Wan | **0.5526** ✓ |
| CN-Top A | 0.5655 |
| CN-Top B | 0.5197 |
| CN-Top C | 0.4998 |

### 8.5 相机运动控制

#### Camera Pose Encoder

```
输入:
  • 外参: R, t ∈ R^(3×4)
  • 内参: K_f ∈ R^(3×3)

处理:
  • Plücker坐标变换 → 细粒度位置序列 P ∈ R^(6×F×H×W)
  • PixelUnshuffle操作 → 降低空间分辨率，增加通道数
  • 卷积模块编码 → 多级相机运动特征
```

#### Camera Pose Adapter

```
公式:
  f_i = (γ_i + 1) × f_(i-1) + β_i

其中:
  • γ_i, β_i: 从相机运动特征序列转换得到的缩放因子和偏移参数
  • f_i: 第i层的视频latent特征
  • f_(i-1): 上一层的视频latent特征

方法:
  两个零初始化卷积层转换输入相机运动特征序列 → γ_i, β_i
```

#### 训练数据

```
使用VGG-SfM算法从训练视频中提取相机轨迹
→ 约 O(1)千个视频片段

训练器: Adam优化器
```

### 8.6 实时视频生成

#### 方法

```
基于预训练WAN模型构建实时生成管道

优势:
  1. 加速收敛和训练稳定性
  2. 模型已捕获有价值知识
```

#### Streaming Video Generation & Consistency Model Distillation

**Streaming Video Generation:**
```
流式生成框架:
  • 增量生成视频帧
  • 实时响应用户输入
  • 适用于交互式娱乐、VR等场景
```

**Consistency Model Distillation:**
```
一致性模型蒸馏:
  • 将多步扩散蒸馏为一步生成
  • 大幅提升生成速度
  • 保持生成质量
```

### 8.7 音频生成

#### 模型设计

```
音频生成模块:
  • 基于预训练音频模型
  • 与视频生成管道集成
  • 同步生成音视频
```

#### 评估

使用音频质量指标评估生成音频的质量和与视频的同步性。

---

## 📝 九、总结与展望

### 9.1 核心贡献总结

| 贡献维度 | 具体内容 |
|---------|---------|
| **模型架构** | Wan-VAE (3D因果VAE), Video DiT (共享AdaLN), umT5文本编码器 |
| **训练策略** | Flow Matching, 渐进式课程学习, 图像-视频联合训练 |
| **数据处理** | 四步数据清洗, 运动质量分级, 视觉文本数据处理, 密集字幕生成 |
| **优化技术** | 2D Context Parallelism, 特征缓存, 扩散缓存, FP8量化 |
| **评估体系** | Wan-Bench基准, 人类反馈加权, 多维度自动化评估 |
| **扩展应用** | I2V, VACE, 个性化, 相机控制, 实时生成, 音频生成 |

### 9.2 技术洞察

1. **Scaling Laws验证：** WAN 14B模型展现了视频生成在数据和模型规模上的scaling laws
2. **高效设计原则：** 增加模型深度 > 增加AdaLN参数；CP通信开销 < TP+SP
3. **缓存机制重要性：** 特征缓存和扩散缓存显著提升推理效率
4. **文本对齐策略：** LLM重写提示词 + 多样性增强改善生成质量
5. **视觉文本生成：** 合成+真实数据组合首次实现中英文本视频生成

### 9.3 开源影响

WAN的开源将：
- 推动视频生成社区发展
- 为学术界提供高质量视频基础模型
- 促进工业界创意视频生产可能性扩展

### 9.4 未来方向

1. **更长视频生成：** 当前5秒，向更长视频扩展
2. **更高分辨率：** 从720p向1080p、4K提升
3. **多模态扩展：** 音频、3D、更多感官模态
4. **实时交互：** 更低延迟的实时生成
5. **个性化增强：** 更精细的身份和风格控制

---

## 🔗 参考链接

### 官方资源
- **Wan GitHub:** https://github.com/Wan-Video/Wan2.1
- **论文arXiv:** https://arxiv.org/abs/2503.20314

### 相关工作链接
- **DiT (Diffusion Transformer):** https://arxiv.org/abs/2212.09748
- **Flow Matching:** https://arxiv.org/abs/2210.02703
- **Sora:** https://openai.com/sora
- **VBench:** https://arxiv.org/abs/2310.14789
- **RAFT (Optical Flow):** https://arxiv.org/abs/2003.12039
- **FlashAttention3:** https://arxiv.org/abs/2307.08691
- **SageAttention:** https://github.com/FasterDecoding/SageAttention
- **Instant-ID:** https://github.com/InstantID/InstantID
- **VGG-SfM:** https://github.com/Anttwo/SfM

### 评估工具
- **VBench Leaderboard:** https://vbench.readthedocs.io/en-US/
- **VideoMME:** https://arxiv.org/abs/2406.04401
- **MANIQA:** https://arxiv.org/abs/2209.03518
- **MUSIQ:** https://arxiv.org/abs/2107.06442

---

这篇WAN论文代表了当前开源视频生成模型的最高水平，通过精心设计的架构、大规模数据训练和全面的优化技术，在性能上超越了包括Sora在内的商业模型。希望这个详细的技术讲解能够帮助你深入理解WAN模型的核心技术和创新点！