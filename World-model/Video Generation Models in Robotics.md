我来为你详细讲解这篇关于视频生成模型在机器人学中应用的综述文章。让我先深入阅读文章的各个部分。



让我再读取一些中间部分的内容来确保完整性。

# 视频生成模型在机器人学中的应用全面解析

## 文章核心概览

这篇由Princeton University和Temple University团队发表的综述文章**"Video Generation Models in Robotics: Applications, Research Challenges, Future Directions"**系统性地回顾了视频生成模型作为**embodied world models（具身世界模型）**在机器人学中的应用。文章的核心理念是：视频生成模型通过学习互联网规模数据，能够生成高保真的物理世界预测视频，捕捉agent与环境之间精细的交互，这解决了传统基于物理的模拟器面临的根本性挑战。

### 文章结构概览

```
Figure 2所示的组织结构:
├── Section 2: 背景知识
│   ├── 2.1 Learned World Models（学习到的世界模型）
│   │   ├── Markovian State-Based World Models
│   │   └── Video World Models
│   ├── 2.2 Diffusion/Flow-Matching Video Models（扩散/流匹配视频模型）
│   │   ├── 2.2.1 Mathematical Formulation（数学公式）
│   │   ├── 2.2.2 Classifier-Based and Classifier-Free Guidance
│   │   ├── 2.2.3 Model Architecture（模型架构）
│   │   └── 2.2.4 Conditioning Modalities（条件模态）
│   └── 2.3 Video Joint-Embedding Predictive Architecture Models
├── Section 3: 机器人学中的四大应用
│   ├── 3.1 模仿学习中的数据生成和动作预测
│   ├── 3.2 强化学习中的动力学和奖励建模
│   ├── 3.3 可扩展策略评估
│   └── 3.4 视觉规划
├── Section 4: 评估指标和基准
├── Section 5: 开放挑战和未来方向
└── Section 6: 结论
```

---

## 第一部分：世界模型的演进

### 1.1 核心问题：为什么需要世界模型？

**传统模拟器的局限：**

1. **简化假设**：物理引擎需要做简化假设，限制视觉和物理保真度
2. **资源昂贵**：需要昂贵的资产curation（curation）程序
3. **sim-to-real gap**：模拟到现实的差距难以消除
4. **可变形物体困难**：难以准确模拟复杂形态和动力学的可变形物体

**语言模型的局限：**

1. **表达能力不足**：语言抽象缺乏捕捉复杂物理交互所需的表达力
2. **时空依赖缺失**：无法准确建模真实世界现象之间的空间和时间依赖关系

**视频模型的优势：**

> "Video generation models provide photorealistic, physically consistent spatiotemporal models of the world."

### 1.2 Markovian State-Based World Models

这类世界模型假设未来环境的演化只依赖于当前状态和动作：

**核心公式：**

```
s_{t+1} ~ p_η(s_{t+1}|s_t, a_t)  (1)
```

其中：
- `s_t` = 当前时间步t的状态
- `a_t` = 动作（物理动作或潜在动作）
- `p_η` = 参数为η的动力学预测器
- `s_{t+1}` = 预测的下一状态

**架构组件：**

```
(2) Encoder: s_t ~ E_γ(s_t|o_t)
(3) Dynamics Predictor: ŝ_{t+1} ~ p_η(ŝ_{t+1}|s_t, a_t)
(4) Rewards Predictor: r̂_{t+1} ~ p_ζ(r̂_{t+1}|ŝ_{t+1})
```

变量说明：
- `o_t` = 观测（RGB图像、本体感受等）
- `E_γ` = 编码器（参数γ），将观测嵌入到潜在空间
- `ŝ_{t+1}` = 预测的下一状态
- `r̂_{t+1}` = 预测的奖励
- `p_ζ` = 奖励预测器（参数ζ）

**历史发展：**
- 早期：RNN（循环神经网络）或RSSMs（循环状态空间模型）
- 近期：Transformers、在像素空间或潜在空间的扩散

**训练目标：** 最小化预测状态与真实状态之间的误差，使用MSE损失或KL散度损失。

### 1.3 Video World Models

视频世界模型学习时空映射，捕获环境在空间和时间上的演化，**不显式建模Markovian状态**。

**关键技术演进：**

1. **空间变换方法**（早期）：仅捕获帧的局部扰动，表达能力有限
2. **GAN扩展**：更高保真度，但易受mode collapse影响
3. **VAE变分推断**：学习像素上的潜在分布，显式编码随机性
4. **Video Transformer**：自回归预测，但面临latent collapse问题
5. **VQ-VAEs**：量化潜在表示到离散码本，解决collapse问题

**VQ-VAE的关键改进：**
```
- 空间和时间压缩以获得跨帧的紧凑潜在嵌入
- 用RMSNorm替换GroupNorm操作以实现时间特征缓存
```

### 1.4 为什么视频模型是更好的世界模型？

| 维度 | State-Based World Models | Video World Models |
|------|--------------------------|-------------------|
| 表达能力 | 局限于潜在状态 | 高保真像素级细节 |
| 物理一致性 | 依赖简化假设 | 从真实世界视频学习 |
| 可变形物体模拟 | 困难 | 天然支持 |
| sim-to-real gap | 显著 | 较小 |

---

## 第二部分：扩散/流匹配视频模型的数学基础

### 2.1 扩散模型核心原理

扩散模型通过学习**逆转渐进加噪过程**来生成数据。

#### 2.1.1 前向过程（Forward Process）

**数学公式：**

```
q(x_t|x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)  (5)
```

变量解释：
- `x_0` = 原始数据（干净数据）~ q(x_0)
- `x_t` = 时间步t的噪声数据
- `β_t` = 噪声调度，t=1,2,...,T
- `N` = 高斯分布
- `√(1-β_t)` = 保留原始数据的系数
- `β_t I` = 添加的噪声

**边际分布的闭式解：**

```
q(x_t|x_0) = N(x_t; √(ᾱ_t) x_0, (1-ᾱ_t) I)  (6)
```

其中：
- `α_t = 1 - β_t`
- `ᾱ_t = Π_{i=0}^{α_t}`（累乘）

**直观理解：**
```
x_0 (干净) → x_1 (加噪) → x_2 (加更多噪) → ... → x_T (纯噪声)
```

#### 2.1.2 反向过程（Reverse Process）

反向过程迭代移除前向过程添加的噪声：

```
p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))  (7)
```

参数说明：
- `μ_θ(x_t, t)` = 学习到的均值
- `Σ_θ(x_t, t)` = 学习到的协方差
- `θ` = 可学习权重

#### 2.1.3 训练损失函数

**噪声预测损失：**

```
L_ε = E_{x_0,t,ε}[‖ε - ε_θ(x_t, t)‖²]  (8)
```

变量说明：
- `ε` = 真实噪声
- `ε_θ(x_t, t)` = 预测的噪声
- `E` = 期望算子

**速度参数化（Velocity-Based Parameterization）：**

```
v = √(ᾱ_t) ε - √(1-ᾱ_t) x_0

L_v = E_{x_0,t,v_t}[‖v_t - v_θ(x_t, t)‖²]  (9)
```

**为什么用速度参数化？**
- 更好的数值稳定性
- 更快的收敛

### 2.2 Classifier-Free Guidance（无分类器引导）

**问题：** 如何让扩散模型生成符合特定属性的视频？

#### 2.2.1 Classifier-Based Guidance（有分类器引导）

训练一个外部分类器 `p_φ(y|x_t)` 来从噪声样本预测属性。

**引导公式：**

```
ε̂_θ(x_t, t, y) = ε_θ(x_t, t) - s · σ_t ∇_{x_t} log p_φ(y|x_t)  (10)
```

变量说明：
- `ε̂_θ` = 偏置噪声样本
- `s` = 引导强度
- `σ_t` = 时间步t的噪声标准差
- `y` = 目标属性

**问题：**
- 计算成本高
- 训练不稳定

#### 2.2.2 Classifier-Free Guidance（无分类器引导）

**核心思想：** 训练一个联合条件和无条件的模型。

**引导公式：**

```
ε̃_θ(x_t, t, y) = (1 + ω) ε_θ(x_t, t, y) - ω ε_θ(x_t, t)  (11)
```

参数说明：
- `ω` = 引导尺度（guidance scale）
- `ε_θ(x_t, t, y)` = 条件噪声预测
- `ε_θ(x_t, t)` = 无条件噪声预测

**为什么成为主流方法？**
- 简单性、通用性、鲁棒性

### 2.3 模型架构

#### 2.3.1 U-Net架构

```
传统U-Net结构:
├── Encoder（编码器）
│   ├── 下采样路径
│   └── 层次化卷积层（捕获高层语义特征）
├── Decoder（解码器）
│   ├── 上采样路径
│   └── 恢复原始分辨率
└── Skip connections（跳过连接）
    └── 保存细粒度空间信息，稳定训练
```

**视频U-Net的扩展：**
- 将2D卷积提升到3D卷积
- 或引入时间注意力模块

**代表工作：** Stable Video Diffusion (SVD)

#### 2.3.2 Diffusion Transformers (DiT)

**DiT的核心设计：**

```
DiT处理流程:
1. 视觉输入分割为patches
2. Patches嵌入为tokens + 位置编码
3. Uniform transformer架构处理所有tokens
4. Self-attention机制
```

**与U-Net的对比：**

| 特性 | U-Net | DiT |
|------|-------|-----|
| 归纳偏置 | 局部性、平移等变性 | 灵活，建模长程依赖 |
| 特征融合 | Skip connections | 自注意力 |
| 时间连贯性 | 较弱 | 更强 |

**为什么SOTA视频模型使用DiT？**
- 更好地捕获时间连贯性
- 整体场景一致性
- 大规模视频扩散模型的优势

**视频编码器的作用：**
- 执行时间压缩
- 显著减少token数量
- 提高生成效率

### 2.4 Conditioning Modalities（条件模态）

#### 2.4.1 条件信号注入机制

| 机制 | 适用场景 | 特点 |
|------|---------|------|
| Channel Concatenation（通道拼接） | 空间对齐条件（I2V、深度引导、姿态） | 像素级对应，强结构约束 |
| Cross-Attention（交叉注意力） | 语义条件（文本提示） | 捕获交互，非空间对齐 |
| Adaptive Normalization（自适应归一化） | 全局属性控制（帧率、运动强度） | 调整归一化层的缩放和偏移 |

#### 2.4.2 主要条件模态

**1. Text-to-Video (T2V) Generation**

```
核心机制: Cross-attention
├── 模型选择性地关注单词或短语
├── 合成特定空间区域
└── 保留跨帧的时间一致性
```

**2. Image-to-Video (I2V) Generation**

```
输入: 参考图像（第一帧）+ 可选文本提示
├── Low-level: 帧级输入拼接（沿时间或通道维度）
└── High-level: 图像特征通过cross-attention（使用CLIP编码器）
```

**3. Motion/Trajectory-Guided Generation**

```
运动编码方式:
├── Coordinate maps（坐标图）
├── Optical flow fields（光流场）
├── Keypoint heatmaps（关键点热图）
└── Applied forces（施加力）

注入方式:
├── Channel concatenation（通道拼接）
├── Dense cross-attention（密集交叉注意力）
└── Specialized adapters（如ControlNet）
```

**机器人特定条件：**

```
直接条件输入:
├── Low-level robot states（低级机器人状态）
└── Actions（动作，如关节位置、力矩）

优势: 细粒度可控性和物理基础
```

---

## 第三部分：视频世界模型的两大类别

### 3.1 Implicit Video World Models（隐式视频世界模型）

**定义：** 3D场景表示完全编码在视频模型内部，没有外部表示。

**代表方法：**

| 方法 | 条件输入 | 关键技术 |
|------|---------|---------|
| Pandora | 文本条件 + 语言指令 | 细粒度控制，长视频 |
| FreeAction | 动作幅度 | 调制classifier-free guidance |
| Vid2World | 动作条件 | Casual attention + diffusion forcing |
| WristWorld | 仅场景相机输入 | 腕部相机视频生成 |
| Enerverse-AC | 动作条件 | Action-conditioned video generation |

**局限性：** 场景演化只能通过生成视频来实现可视化。

### 3.2 Explicit Video World Models（显式视频世界模型）

**定义：** 使用视频模型创建具体的3D场景表示。

**代表方法：**

| 方法 | 场景表示 | 技术路径 |
|------|---------|---------|
| Enerverse | Multi-view视频 | 4D Gaussian Splatting |
| Aether | 深度 + Camera raymap视频 | Back-projection提取3D表示 |
| Genie Envisioner | 潜在视频状态 + Action decoder | 联合视频生成和动作预测 |

**Figure 4: 视频模型的具身世界建模架构**

```
视频世界模型分类:

Implicit（隐式）:
┌─────────────────────────────────────────┐
│  Video Model Only                        │
│  ┌──────────┐     ┌──────────┐          │
│  │ Video    │ ─→  │ Generated│          │
│  │ Frames   │     │ Video    │          │
│  └──────────┘     └──────────┘          │
│    ↑                                     │
│    │ (3D场景编码在模型内部)              │
│  ┌──────────┐                           │
│  │ Text/    │                           │
│  │ Action   │                           │
│  └──────────┘                           │
└─────────────────────────────────────────┘

Explicit（显式）:
┌─────────────────────────────────────────┐
│  Video Model + 3D Representation         │
│  ┌──────────┐     ┌──────────┐          │
│  │ Video    │ ─→  │ 3D Scene │          │
│  │ Model    │     │ Rep      │          │
│  └──────────┘     └──────────┘          │
│       │                │                 │
│       └────────────────┼─────────────┐   │
│                        │             │   │
│                        ↓             │   │
│                ┌───────────────┐     │   │
│                │ Gaussian      │     │   │
│                │ Splatting     │     │   │
│                │ / Voxel Grid  │     │   │
│                └───────────────┘     │   │
│                                      │   │
│  ┌──────────┐                       │   │
│  │ Multi-   │ ──────────────────────┘   │
│  │ view     │                            │
│  │ + Depth  │                            │
│  └──────────┘                            │
└─────────────────────────────────────────┘
```

---

## 第四部分：机器人学中的四大应用

### 4.1 Cost-Effective Data Generation and Action Prediction in Imitation Learning

#### 4.1.1 问题背景

**模仿学习的数据挑战：**
- SOTA VLA模型（如RT-2、π0.5）的成功依赖于大规模专家演示
- 收集专家演示数据的成本高昂（时间、人力成本）
- 数据scaling已被证明是提升机器人策略性能的关键

**解决方案：** 使用视频生成模型作为低成本的数据生成器

#### 4.1.2 视频模型作为数据生成器

**核心流程（Figure 5）：**

```
视频数据生成流程:

┌─────────────────────────────────────────────────────────┐
│  1. Pre-training（预训练）                               │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Large-scale Text/Image Video Models              │  │
│  │ (Cosmos Predict, Wan, Sora, etc.)                │  │
│  └──────────────────────────────────────────────────┘  │
│                         │                               │
│                         ↓                               │
│  2. Fine-tuning（微调）                                 │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Robot Datasets (Bridge, DROID, etc.)             │  │
│  │ → Adapt to robot embodiments & environments      │  │
│  └──────────────────────────────────────────────────┘  │
│                         │                               │
│                         ↓                               │
│  3. Action Conditioning（动作条件）                    │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Controllable Video Generation                    │  │
│  │ - Text → Video                                   │  │
│  │ - Image → Video                                  │  │
│  │ - Keypoints/Trajectory → Video                  │  │
│  └──────────────────────────────────────────────────┘  │
│                         │                               │
│                         ↓                               │
│  4. Action Estimation（动作估计）                      │
│  ┌──────────────────────────────────────────────────┐  │
│  │  ┌──────────────────────────────────────┐        │  │
│  │  │ End-to-End Methods                  │        │  │
│  │  │ - Latent Action Models              │        │  │
│  │  │ - Inverse-Dynamics Models (IDMs)    │        │  │
│  │  └──────────────────────────────────────┘        │  │
│  │                    │                              │  │
│  │                    ↓                              │  │
│  │  ┌──────────────────────────────────────┐        │  │
│  │  │ Modular Methods                     │        │  │
│  │  │ - Pose tracking (FoundationPose,    │        │  │
│  │  │   MegaPose)                         │        │  │
│  │  │ - Optical flow extraction           │        │  │
│  │  │ - CAD model matching                │        │  │
│  │  └──────────────────────────────────────┘        │  │
│  └──────────────────────────────────────────────────┘  │
│                         │                               │
│                         ↓                               │
│  5. Policy Learning（策略学习）                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │ - VLA models (RT-2, π0.5)                       │  │
│  │ - Direct robot application                      │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**方法分类：**

##### 方法1：End-to-End（端到端）

**a) Latent Action Models（潜在动作模型）**

```
核心思想:
在潜在空间推断动作，解释视频帧之间的转换，无需真实动作标签。

架构流程:
┌──────────────────────────────────────────┐
│  Input: Frame pair (x_t, x_{t+1})       │
│           │                              │
│           ↓                              │
│  ┌──────────────────┐                   │
│  │ Encoder          │                   │
│  │ (VQ-VAE based)  │                   │
│  └──────────────────┘                   │
│           │                              │
│           ↓                              │
│  ┌──────────────────┐                   │
│  │ Latent Action    │ ─→ z_t (latent) │
│  │ Encoder          │                   │
│  └──────────────────┘                   │
│           │                              │
│           ↓                              │
│  ┌──────────────────┐                   │
│  │ Decoder          │                   │
│  │ Reconstruct x_{t+1}│                 │
│  └──────────────────┘                   │
└──────────────────────────────────────────┘
```

**为什么用VQ-VAEs？**
- 表达性和训练效率的最佳权衡

**局限性：**
- 需要fine-tuning阶段对齐真实动作空间
- 潜在动作空间与真实动作空间不等价

**代表工作：** Gen2Act, LuciBot

**b) Inverse-Dynamics Models（IDMs，逆动力学模型）**

```
监督学习流程:
┌──────────────────────────────────────────┐
│  Training Data:                          │
│  (Video frames, Ground-truth actions)    │
│           │                              │
│           ↓                              │
│  ┌──────────────────┐                   │
│  │ Encoder-Decoder  │                   │
│  │ Model            │                   │
│  │ (Diffusion-based)│                   │
│  └──────────────────┘                   │
│           │                              │
│           ↓                              │
│  Output: Action prediction               │
└──────────────────────────────────────────┘
```

**优点：**
- 零样本部署（无需fine-tuning）
- 直接预测真实动作

**缺点：**
- 需要大量动作标注数据
- 训练分布外泛化能力有限

**代表工作：** DreamGen, VPP, ARDuP, Vidar

##### 方法2：Modular（模块化）

```
模块化动作估计流程:

Step 1: Object Pose Estimation（物体姿态估计）
┌─────────────────────────────────────────┐
│  Video Frames →                        │
│  ┌─────────────────────────────────┐   │
│  │ Pose Tracker (FoundationPose,  │   │
│  │   MegaPose)                    │   │
│  │ Monocular Depth Estimator      │   │
│  └─────────────────────────────────┘   │
│           │                               │
│           ↓                               │
│  3D Object Pose (per frame)              │
└─────────────────────────────────────────┘

Step 2: Motion Extraction（运动提取）
┌─────────────────────────────────────────┐
│  Option A: Optical Flow Predictor       │
│  Option B: CAD Model Matching           │
│           │                               │
│           ↓                               │
│  Object Trajectory                       │
└─────────────────────────────────────────┘

Step 3: Retargeting（重定向）
┌─────────────────────────────────────────┐
│  Object Trajectory →                    │
│  ┌─────────────────────────────────┐   │
│  │ Rigid Transformation to Robot   │   │
│  │ End-effector Pose               │   │
│  └─────────────────────────────────┘   │
│           │                               │
│           ↓                               │
│  Robot Action Sequence                    │
└─────────────────────────────────────────┘
```

**假设：** 物体参考点与机器人末端执行器之间存在固定的刚体变换。

**优点：**
- 零样本部署
- 利用低级控制例程

**代表工作：** AVDC, VideoAgent, Lucibot

#### 4.1.3 视频模型作为策略骨干

**统一视频-动作方法：** 联合预测视频和动作，条件是语言指令和初始观测。

**架构分类：**

| 方法 | 核心架构 | 特点 |
|------|---------|------|
| GR1 | Autoregressive Transformer | 联合预测未来图像和动作 |
| RPT | Masking-based | Mask inputs before action prediction |
| UVA | VLA + Joint latent | 联合视频帧和动作的潜在表示 |
| PAD | Stable Diffusion + DiT/U-Net | 减少训练开销 |
| UWM | Independent diffusion processes | 动作和视频生成的独立扩散过程 |
| DreamVLA | VLA + depth/dynamic regions | 更强的监督信号 |
| UniVLA | VLA + language token prediction | 预测语言token+视频+动作 |

### 4.2 Dynamics and Rewards Modeling in Reinforcement Learning

#### 4.2.1 问题背景

**模仿学习 vs 强化学习：**

| 维度 | 模仿学习 | 强化学习 |
|------|---------|---------|
| 训练效率 | 较高 | 较低 |
| 泛化能力 | 有限（训练分布内） | 更好（分布外） |
| 需求 | 动力学+奖励模型 | 自动探索 |

**RL的挑战：**
1. 需要指定动力学和奖励模型（通常非平凡）
2. 样本效率低

**解决方案：** 使用视频模型作为表达性动力学和奖励模型

#### 4.2.2 应用方法

**Figure 6: 动力学和奖励建模架构**

```
视频模型在RL中的应用:

┌────────────────────────────────────────────────────────┐
│  Method 1: Dynamics Modeling（动力学建模）              │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Action-Conditioned Video Model                  │  │
│  │  s_t, a_t → Predict s_{t+1} (video frames)      │  │
│  │                                                  │  │
│  │  Training:                                      │  │
│  │  - Dreamer 4: Text → Video → Action fine-tune  │  │
│  │  - World-Env: Pre-trained model + VLM rewards │  │
│  └──────────────────────────────────────────────────┘  │
│                         │                               │
│                         ↓                               │
│  Used for: Policy refinement with RL                 │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│  Method 2: Reward Modeling（奖励建模）                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │ VIPER: Video prediction likelihood as reward    │  │
│  │                                                  │  │
│  │  L_reward = log p(video | expert_data)         │  │
│  │                                                  │  │
│  │  Diffusion Reward: Conditional entropy          │  │
│  │                                                  │  │
│  │  L_reward = -H(video | condition)               │  │
│  │                                                  │  │
│  │  Idea: Lower entropy → Closer to expert        │  │
│  └──────────────────────────────────────────────────┘  │
│                         │                               │
│                         ↓                               │
│  Used for: Learning from expert trajectories         │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│  Method 3: Exploration Guidance（探索引导）             │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Video Model → Generate exploration videos     │  │
│  │  → Save to buffer                              │  │
│  │  → Fine-tune policy                            │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

**代表工作详解：**

1. **Dreamer 4：**
```
训练流程:
1. 训练文本条件视频模型
2. 在动作标注数据上fine-tune为动作条件视频模型
3. 使用动作条件视频模型作为动力学预测器
4. 使用RL fine-tune模仿学习的策略和奖励头
应用: Minecraft环境
```

2. **World-Env：**
```
训练流程:
1. 使用预训练动作条件视频模型进行视频生成
2. 使用VLM（instance reflector）提供密集奖励信号
3. 使用RL fine-tune VLA
```

3. **VIPER：**
```
奖励信号设计:
r_t = log p(video_{t:t+H} | s_t, a_t, ...)

其中:
- H: 视频预测horizon
- p(): 视频预测概率
```

4. **Diffusion Reward：**
```
奖励信号设计:
r_t = -H(video | condition)

其中:
- H(): 条件熵
- 思想: 接近专家轨迹的视频具有更低熵
```

### 4.3 Scalable Policy Evaluation（可扩展策略评估）

#### 4.3.1 问题背景

**现实世界策略评估的挑战：**
1. **硬件成本高**：每次试验需要人工监控、重置环境、记录成功分数
2. **劳动成本高**：通用机器人策略需要在多种操作环境中评估（组合数量）
3. **物理模拟器的局限**：
   - 需要大量设置时间（手动重建环境）
   - 需要精心调参（光照、材料属性）
   - 仍然存在sim-to-real gap
   - 可变形物体模拟困难

**视频模型的优势：**
- 更高保真度、更可扩展
- 建模复杂的机器人-环境交互
- 可快速构造OOB（out-of-distribution）场景

#### 4.3.2 评估流程

```
视频模型策略评估流程:

Step 1: Policy Rollout in Video World Model
┌─────────────────────────────────────────┐
│  Sample Initial Observations (images)   │
│           │                              │
│           ↓                              │
│  ┌─────────────────────────────────┐   │
│  │ Closed-loop with Video Model    │   │
│  │                                 │   │
│  │  o_t, a_t ─→ Video Model ─→   │   │
│  │                 o_{t+1}         │   │
│  │                                 │   │
│  │  Repeat for T timesteps         │   │
│  └─────────────────────────────────┘   │
│           │                              │
│           ↓                              │
│  Generated Video Trajectory              │
└─────────────────────────────────────────┘

Step 2: Scoring（打分）
┌─────────────────────────────────────────┐
│  Evaluate based on rubric:              │
│  - Task completion                       │
│  - Instruction following                 │
│  ┌─────────────────────────────────┐   │
│  │ Bernoulli score (0/1)          │   │
│  │ Partial credit                  │   │
│  └─────────────────────────────────┘   │
│           │                              │
│           ↓                              │
│  Individual rollout scores               │
└─────────────────────────────────────────┘

Step 3: Aggregation（聚合）
┌─────────────────────────────────────────┐
│  Aggregate scores across N rollouts      │
│  → Empirical success rate                │
└─────────────────────────────────────────┘

Step 4: Accuracy Assessment（准确度评估）
┌─────────────────────────────────────────┐
│  Metrics:                                │
│  ┌─────────────────────────────────┐   │
│  │ Pearson Correlation Coefficient │   │
│  │ - Measures linear relationship  │   │
│  │   between predicted & real     │   │
│  │   success rates                 │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │ Mean Maximum Rank Violation     │   │
│  │ (MMRV)                          │   │
│  │ - Captures ranking inconsistencies│ │
│  │ - Weights by real-world diff   │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

**Pearson Correlation Coefficient公式：**

```
ρ = Cov(X, Y) / (σ_X · σ_Y)

其中:
- X: 视频模型预测的成功率
- Y: 真实世界成功率
- Cov(): 协方差
- σ: 标准差

理想情况: ρ ≈ 1（高相关性）
```

**MMRV计算：**

```
MMRV = (1/N) Σ_{i=1}^{N} max_{j ≠ i} [rank_violation(i, j) · |Y_i - Y_j|]

其中:
- N: 策略数量
- rank_violation(i, j): 策略i和j的排序违规
- Y_i: 策略i的真实成功率

MMRV越低，排序一致性越好
```

#### 4.3.3 提高生成质量的技术

**1. 多视角生成**

```
优势: 
- 减少幻觉
- 提供冗余信息
- 提高空间一致性

设置: 
机器人操作通常涉及多个相机视角
- 场景相机（全局视角）
- 腕部相机（近场视角）

方法: 
适配一致的multi-view生成
```

**2. 历史信息整合**

```
问题: 长期预测中误差累积

解决方案:
- Condition future frames on past observations
- History options:
  a) All frames within fixed window length
  b) Sparsely sampled subset of past frames

效果: 减少预测漂移，提高长期一致性
```

**3. 姿态增强**

```
方法: Augment observations with robot joint poses

效果: 改进帧级动作可控性
```

**4. 潜在动作条件化**

```
创新: Condition video model on latent action representation 
      instead of physical robot actions

优势: 
- 更好的动作可控性
- 更稳定的预测
```

#### 4.3.4 安全性和鲁棒性评估

**Veo World Simulator（DeepMind）应用：**

```
OOB场景快速构造:
┌─────────────────────────────────────────┐
│  Image Editing →                        │
│  ┌─────────────────────────────────┐   │
│  │ - Alter background              │   │
│  │ - Add novel objects            │   │
│  │ - Add distractors              │   │
│  └─────────────────────────────────┘   │
│           │                              │
│           ↓                              │
│  Video Model → Performance prediction   │
└─────────────────────────────────────────┘

安全测试:
┌─────────────────────────────────────────┐
│  VLM → Generate safety-critical scenes  │
│  ┌─────────────────────────────────┐   │
│  │ - Test semantic safety         │   │
│  │ - Test physical safety         │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

**Failure Data的重要性：**

```
问题: 
仅用成功数据训练会导致乐观偏差（总是预测成功）

解决方案:
- Incorporate failure data during training
- 准确预测成功和失败场景
- 避免bias towards success

代表工作: WorldGym, Polaris, Veo World Simulator
```

### 4.4 Visual Planning（视觉规划）

#### 4.4.1 问题背景

**视觉规划定义：** 生成一系列图像或视频帧，展示完成语言指令和初始观测指定任务所需的步骤。

**传统规划 vs 视频规划：**

| 维度 | 传统规划 | 视觉规划 |
|------|---------|---------|
| 动力学模型 | 需要显式建模 | 从视频模型学习 |
| 演示数据 | 需要大规模专家演示 | 利用编码的多样性 |
| 任务范围 | 有限 | 广泛的机器人任务 |

#### 4.4.2 优化方法

**采样方法：**
- Cross-Entropy Method（交叉熵方法）
- 梯度无关的轨迹优化

**梯度方法：**
- 梯度下降
- Levenberg-Marquardt优化器

**模型预测控制（MPC）：**
- 嵌入优化例程
- 通过传感器反馈融入新观测

#### 4.4.3 Action-Guided Visual Planning（动作引导的视觉规划）

**三步流程：**

```
Step 1: Generate Action Proposals（生成动作提案）
┌─────────────────────────────────────────┐
│  Option A: Sampling-based              │
│  ┌─────────────────────────────────┐   │
│  │ Cross-Entropy Method           │   │
│  │ - Sample actions from Gaussian │   │
│  │ - Fit Gaussian to best actions │   │
│  │ - Iterate                      │   │
│  └─────────────────────────────────┘   │
│                                         │
│  Option B: Learned Approaches          │
│  ┌─────────────────────────────────┐   │
│  │ Conditional VAE                 │   │
│  │ - Learn action distribution    │   │
│  └─────────────────────────────────┘   │
│                                         │
│  Option C: VLM-based                   │
│  ┌─────────────────────────────────┐   │
│  │ - Propose camera trajectories  │   │
│  │ - Task decomposition           │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
│                │                        │
│                ↓                        │
│  Action Proposals {a_1, ..., a_N}       │
└─────────────────────────────────────────┘

Step 2: Synthesize Video Trajectories（合成视频轨迹）
┌─────────────────────────────────────────┐
│  For each action proposal a_i:          │
│  ┌─────────────────────────────────┐   │
│  │ Video Model + a_i               │   │
│  │    │                            │   │
│  │    ↓                            │   │
│  │  Generated Video Trajectory    │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
│                │                        │
│                ↓                        │
│  Video Trajectories {v_1, ..., v_N}    │
└─────────────────────────────────────────┘

Step 3: Evaluate Trajectories（评估轨迹）
┌─────────────────────────────────────────┐
│  Objective Function:                    │
│  ┌─────────────────────────────────┐   │
│  │ Photometric error (goal image)  │   │
│  │ Euclidean error (keypoints)     │   │
│  │ Value function (learned)        │   │
│  └─────────────────────────────────┘   │
│                                         │
│  Select best action: a* = argmax L      │
└─────────────────────────────────────────┘
```

**代表方法详解：**

1. **FLIP（Flow-centric Generative Planning）：**
```
流程:
1. Conditional VAE生成动作提案
2. Fine-tune language-image value network
3. 学习生成视频轨迹上的价值函数
4. 选择最大化折扣返回的动作

价值网络训练:
┌─────────────────────────────────────────┐
│  Video Trajectory → Value Network → Q   │
│           │                              │
│           ↓                              │
│  Discounted Return: R + γQ'              │
└─────────────────────────────────────────┘
```

2. **MindJourney：**
```
VLM的作用:
┌─────────────────────────────────────────┐
│  VLM → Propose camera trajectories      │
│  VLM → Evaluate generated videos        │
│  Test-time scaling for spatial reasoning│
└─────────────────────────────────────────┘
```

3. **Du等人（2023）- Video Language Planning：**
```
任务分解流程:
┌─────────────────────────────────────────┐
│  Language Task → VLM Decomposition     │
│      ↓                                  │
│  Natural-language subtasks              │
│      ↓                                  │
│  Text inputs for video model            │
│      ↓                                  │
│  Generated videos                       │
│      ↓                                  │
│  VLM analysis → Tree search refinement  │
│      ↓                                  │
│  Best video plan → Goal-conditioned     │
│  policy → Robot actions                 │
└─────────────────────────────────────────┘
```

#### 4.4.4 Action-Free Visual Planning（无动作视觉规划）

**核心思想：** 直接生成视频计划，使用视频帧作为图像子目标进行规划。

```
Action-Free Planning Flow:

Step 1: Generate Video Plans
┌─────────────────────────────────────────┐
│  Text-Conditioned Video Model           │
│           │                              │
│           ↓                              │
│  Video Plan = {frame_1, ..., frame_N}   │
└─────────────────────────────────────────┘
                │
                ↓
Step 2: Use Frames as Image Subgoals
┌─────────────────────────────────────────┐
│  Extract actions from image subgoals    │
│                                         │
│  Option A: Goal-Conditioned BC          │
│  ┌─────────────────────────────────┐   │
│  │ Current obs + Subgoal → Action  │   │
│  └─────────────────────────────────┘   │
│                                         │
│  Option B: IDM (Inverse Dynamics Model) │
│  ┌─────────────────────────────────┐   │
│  │ Current obs + Subgoal → Action  │
│  │ + Error between embeddings     │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

**代表方法：**

1. **CLOVER：**
```
IDM设计:
┌─────────────────────────────────────────┐
│  Input:                                  │
│  - Current observation o_t               │
│  - Image subgoal o_goal                  │
│  - Embedding error:                      │
│    ‖E(o_t) - E(o_goal)‖                  │
│           │                              │
│           ↓                              │
│  Output: Action a_t                      │
└─────────────────────────────────────────┘
```

2. **UniPi：**
```
两阶段生成:
┌─────────────────────────────────────────┐
│  Phase 1: Coarse Video Trajectory       │
│           │                              │
│           ↓                              │
│  Phase 2: Super-resolution Refinement   │
│           │                              │
│           ↓                              │
│  Fine video plan → IDM → Actions        │
└─────────────────────────────────────────┘
```

3. **NovaFlow：**
```
可变形物体处理:
┌─────────────────────────────────────────┐
│  Generated Video →                      │
│  Extract object pose (particles)         │
│           │                              │
│           ↓                              │
│  Coarse robot actions →                  │
│  Non-linear least-squares refinement     │
│  with pre-trained particle dynamics     │
│           │                              │
│           ↓                              │
│  Refined action trajectory               │
└─────────────────────────────────────────┘
```

---

## 第五部分：评估视频模型的指标和基准

### 5.1 Frame-Level Metrics（帧级指标）

#### 5.1.1 PSNR (Peak Signal-to-Noise Ratio)

```
PSNR = 10 · log₁₀(MAX_I² / MSE)

其中:
- MAX_I: 最大像素值（对于8-bit图像，MAX_I = 255）
- MSE: Mean Squared Error
  MSE = (1/N) Σ_{i=1}^{N} (I_i - R_i)²
  - I_i: 生成图像的第i个像素
  - R_i: 真实图像的第i个像素

特点:
- 与像素MSE成反比
- 与人类感知相关性较差
```

#### 5.1.2 SSIM (Structural Similarity Index)

```
SSIM(x, y) = (2μ_xμ_y + C₁)(2σ_xy + C₂) / 
             (μ_x² + μ_y² + C₁)(σ_x² + σ_y² + C₂)

其中:
- μ_x, μ_y: 两个图像的均值
- σ_x², σ_y²: 两个图像的方差
- σ_xy: 两个图像的协方差
- C₁, C₂: 稳定常数

特点:
- 评估亮度、对比度、结构
- 比PSNR更符合感知相似性
- 但仍无法捕捉高层图像结构
```

#### 5.1.3 CLIP Similarity Score

```
CLIP Sim(I₁, I₂) = cos(E_θ(I₁), E_θ(I₂))

其中:
- E_θ(): CLIP编码器
- cos(): 余弦相似度

特点:
- 在学习到的特征空间中评估语义对齐
- 更符合人类感知
```

#### 5.1.4 Inception Score

```
IS = exp(E_x[KL(p(y|x) || p(y))])

其中:
- p(y|x): 生成图像x的预测类别分布
- p(y): 所有生成图像的边缘分布
- KL(): KL散度

特点:
- 评估模型生成多样且有语义意义的图像
- 不使用真实样本
- 依赖于固定标签空间
```

#### 5.1.5 FID (Fréchet Inception Distance)

```
FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_rΣ_g)^(1/2))

其中:
- μ_r, μ_g: 真实和生成特征的均值
- Σ_r, Σ_g: 真实和生成特征的协方差矩阵
- Tr(): 矩阵迹

特点:
- 计算多维高斯分布之间的Wasserstein-2距离
- 比IS更稳定
```

#### 5.1.6 LPIPS (Learned Perceptual Image Patch Similarity)

```
LPIPS(x, x₀) = Σ_l w_l · ‖E_l(x) - E_l(x₀)‖₂

其中:
- E_l(): 深度图像识别网络第l层的特征
- w_l: 学习到的权重
- ‖·‖₂: L2范数

特点:
- 基于深度网络多层特征
- 与人类判断具有可比性
```

### 5.2 Spatiotemporal Metrics（时空指标）

#### 5.2.1 FVD (Fréchet Video Distance)

```
FVD扩展FID到视频:
FVD = ||μ_r^v - μ_g^v||² + Tr(Σ_r^v + Σ_g^v - 2(Σ_r^vΣ_g^v)^(1/2))

其中:
- μ_r^v, μ_g^v: 真实和生成视频特征的均值
- 特征同时捕获运动特征和图像质量

特点:
- 在时间维度上扩展FID
- 评估跨帧的运动特征
```

#### 5.2.2 KVD (Kernel Video Distance)

```
KVD使用核方法:
KVD = MMD^2(P_r, P_g)

其中:
- MMD: Maximum Mean Discrepancy
- P_r, P_g: 真实和生成的特征分布

特点:
- 不假设特征为高斯分布
- 捕获更高阶的变化
```

#### 5.2.3 FVMD (Fréchet Video Motion Distance)

```
FVMD流程:
1. Extract keypoints trackable across frames
2. Compute velocities and accelerations
3. Compute Fréchet distance on motion features

特点:
- 专注于运动特征
- 评估动态一致性
```

### 5.3 Benchmarks（基准）

#### 5.3.1 Broad Benchmarks（广泛基准）

| 基准 | 评估维度 | 关键指标 |
|------|---------|---------|
| WorldModelBench | 指令遵循+物理 adherence | 物体大小一致性（违反质量守恒） |
| EvalCrafter | 美学+运动+时间一致性+文本对齐 | DOVER, Kinetics-400, Optical Flow, CLIP |
| EWMBench | 场景+运动+语义质量 | DINOv2嵌入余弦相似度 |
| VBench | 16个细粒度标准 | 背景一致性、主体一致性、时间闪烁、美学质量 |
| PAI-Bench | 时间一致性+运动平滑度+美学质量 | - |
| T2V-CompBench | 对象属性+动作一致性 | 颜色、形状、纹理、动作一致性 |
| WorldSimBench | 物理一致性 | 深度感知、不同介质中的速度变化 |

#### 5.3.2 Physical Commonsense Benchmarks（物理常识基准）

| 基准 | 评估内容 | 覆盖范围 |
|------|---------|---------|
| Physics-IQ | 物理定律理解 | 光学、热力学、磁学、流体动力学 |
| PhyGenBench | 常识知识 | 27条物理定律（重力、升华、溶解、摩擦等） |
| VideoPhy | 对象交互 | 物体间的物理交互 |
| VP² | 视觉规划中的物理定律对齐 | 证明感知指标强≠物理一致性好 |

**关键发现：**
```
VP²的主要结论:
1. 强性能在感知指标上不指示物理一致性
2. 扩展模型大小和训练数据集提高性能，但增益快速饱和
3. 当前视频模型缺乏对物理定律的坚实理解
```

---

## 第六部分：开放挑战和未来方向

### 6.1 Hallucinations and Violations of Physics（幻觉和物理违规）

#### 6.1.1 问题表现

**幻觉类型（T2V生成）：**
- Vanishing subject（主体消失）
- Omission error（遗漏错误）
- Numeric variability（数值变化）
- Visual incongruity（视觉不一致）
- Subject dysmorphia（主体变形）

**物理违规：**
```
违反的基本原理:
1. Newton's law of motion（牛顿运动定律）
2. Conservation of energy and mass（能量和质量守恒）
3. Gravitational effects（重力效应）

具体表现:
- 固体-固体交互不真实
- 不理解材料属性
- 动量守恒违规
- 物体不可穿透性违规
- 流体力学理解缺失
- 质量守恒违规
  → 例：倒入杯子里的饮料没有相应增加杯子中的液体体积
```

**模仿训练示例的倾向：**
```
优先级顺序（从高到低）:
1. Color（颜色）
2. Size（大小）
3. Velocity（速度）
4. Shape（形状）

问题: 限制了对新（未见）任务的泛化能力
```

#### 6.1.2 减轻幻觉的技术

**1. 多视角输入：**
```
效果:
- 使用multi-view frame inputs减少幻觉
- 特别是包括wrist camera view

原因:
- 提供冗余空间信息
- 增加几何一致性约束
```

**2. 物理先验集成：**

**Hamiltonian Neural Networks（HNN）：**
```
核心思想: 学习系统的Hamiltonian函数

Hamiltonian形式:
H(q, p) = T(p) + V(q)

其中:
- q: 广义坐标
- p: 广义动量
- T(p): 动能
- V(q): 势能

动力学方程:
dq/dt = ∂H/∂p
dp/dt = -∂H/∂q

优势: 自动满足能量守恒
```

**Lagrangian Neural Networks（LNN）：**
```
Lagrangian形式:
L(q, q̇) = T(q, q̇) - V(q)

Euler-Lagrange方程:
d/dt(∂L/∂q̇) - ∂L/∂q = 0

优势: 自动满足能量守恒，更通用
```

**3. 物理模拟器集成：**

**PhysGen流程：**
```
┌─────────────────────────────────────────┐
│  Input: Image + Physical Properties     │
│           │                              │
│           ↓                              │
│  ┌─────────────────────────────────┐   │
│  │ VLM → Object Segmentation      │   │
│  │      + Physical Properties     │   │
│  └─────────────────────────────────┘   │
│           │                              │
│           ↓                              │
│  ┌─────────────────────────────────┐   │
│  │ Rigid-body Dynamics Equations  │   │
│  │ → Feasible Trajectories        │   │
│  └─────────────────────────────────┘   │
│           │                              │
│           ↓                              │
│  ┌─────────────────────────────────┐   │
│  │ Apply trajectories to pixels   │   │
│  │ → Initial video (with artifacts)│   │
│  └─────────────────────────────────┘   │
│           │                              │
│           ↓                              │
│  ┌─────────────────────────────────┐   │
│  │ Video Diffusion Model          │   │
│  │ → Artifact editing              │   │
│  │ → High-quality video            │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

**WonderPlay流程：**
```
┌─────────────────────────────────────────┐
│  Input: Conditioning inputs             │
│           │                              │
│           ↓                              │
│  ┌─────────────────────────────────┐   │
│  │ Construct 3D Scene Representation │   │
│  └─────────────────────────────────┘   │
│           │                              │
│           ↓                              │
│  ┌─────────────────────────────────┐   │
│  │ VLM → Estimate physical       │   │
│  │     properties of objects      │   │
│  └─────────────────────────────────┘   │
│           │                              │
│           ↓                              │
│  ┌─────────────────────────────────┐   │
│  │ Physics-based Simulator       │   │
│  │ → Coarse trajectory conditioned │   │
│  │   on applied forces             │   │
│  └─────────────────────────────────┘   │
│           │                              │
│           ↓                              │
│  ┌─────────────────────────────────┐   │
│  │ Video Model (conditioned on    │   │
│  │ coarse trajectory)             │   │
│  │ → Generate future frames       │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

**4. LLM提示细化：**
```
方法:
┌─────────────────────────────────────────┐
│  Raw Prompt → LLM → Refined Prompt      │
│                                         │
│  Refined Prompt包含:                    │
│  - Comprehensive physical attributes   │
│  - Interaction descriptions            │
└─────────────────────────────────────────┘
```

**5. Affordance-based Video Understanding（基于affordance的视频理解）：**

```
Affordance Map生成流程:
┌─────────────────────────────────────────┐
│  Video (human-object interaction)       │
│           │                              │
│           ↓                              │
│  ┌─────────────────────────────────┐   │
│  │ Localize interaction regions    │   │
│  │ → Contact hotspots              │   │
│  └─────────────────────────────────┘   │
│           │                              │
│           ↓                              │
│  ┌─────────────────────────────────┐   │
│  │ Embed hotspots into latent space│   │
│  │ → Predict hotspot evolution     │   │
│  └─────────────────────────────────┘   │
│           │                              │
│           ↓                              │
│  ┌─────────────────────────────────   │
│  │ Affordance maps as conditioning │   │
│  │ signals in video synthesis     │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘

优势:
- 改进生成视频的物理一致性
- 捕获可行的交互模式
```

### 6.2 Uncertainty Quantification（不确定性量化）

#### 6.2.1 挑战

**现有UQ方法的局限性：**

| 挑战 | 说明 |
|------|------|
| 非IID假设 | 视频帧之间相关，违反标准贝叶斯UQ的IID假设 |
| 计算成本高 | 视频生成昂贵，阻碍集成UQ方法 |
| 无法表达置信度 | 现有视频模型无法表达或表达其置信度 |

#### 6.2.2 现有方法

**S-QUBED：**
```
特点:
- 在语义空间中量化T2V生成的不确定性
- 仅估计任务级置信度

局限:
- 不提供细粒度的不确定性估计
```

**C³（Conditional Confidence Calibration）：**
```
创新点:
- 同时进行视频生成和不确定性量化
- 在潜在空间中预测每个subpatch的不确定性
- 提供空间和时间上的密集置信度估计

输出:
┌─────────────────────────────────────────┐
│  Generated Video                        │
│  ┌─────────────────────────────────┐   │
│  │ Each subpatch:                  │   │
│  │ {pixel_values, uncertainty_map} │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘

局限:
- 仅在训练数据分布内保证校准的不确定性
- 分布外效果有限
```

#### 6.2.3 未来方向

```
研究方向:
1. 更成本效益的UQ方法
2. 具有可证明保证的UQ（分布内和分布外）
3. 推理时不确定性表达机制
4. 稀疏不确定性估计（降低计算成本）
```

### 6.3 Instruction Following（指令遵循）

#### 6.3.1 问题表现

```
指令遵循失败的表现:
1. 提取和转移预期动作失败
   - 生成视频中包含指定的agent
   - 但只部分遵循指定动作
   - 有时完全未能包含指定动作

2. 文本生成质量差
   - 视频中无法生成高质量文本（注释）
   - 特别是当prompt明确请求文本注释时

3. 摄像机运动控制困难
   - 即使要求静态摄像机视角
   - 模型倾向于模仿训练视频（通常包含摄像机运动）
   - 导致与输入prompt不遵循
```

#### 6.3.2 对机器人的影响

```
影响1: 机器人数据生成
- 许多方法假设静态摄像机位置
- 用于准确的3D末端执行器姿态估计
- 违反此假设导致不准确的目标姿态
- 降低机器人任务性能

影响2: 策略学习
- 未能生成正确遵循动作的视频
- 导致训练数据污染
- 限制模仿学习的有效性
- 依赖高质量（专家）演示
```

#### 6.3.3 改进方法

**ViMi（Vision-Multimodal Instruction）：**
```
流程:
┌─────────────────────────────────────────┐
│  Interleave language and images into    │
│  single instruction prompt              │
│           │                              │
│           ↓                              │
│  ┌─────────────────────────────────┐   │
│  │ VLM → Extract conditional       │   │
│  │     embedding for video         │   │
│  │     generation                   │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

**Aid（Adaptive Image Diffusion）：**
```
流程:
┌─────────────────────────────────────────┐
│  VLM → Predict states of future frames  │
│           │                              │
│           ↓                              │
│  ┌─────────────────────────────────┐   │
│  │ Fuse with text instruction →    │   │
│  │ Conditional embedding for       │   │
│  │ video synthesis                 │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

**InteractiveVideo：**
```
特点:
- 通过图像、文本和轨迹提示控制内容
- 描述视频中不同元素期望的运动

多模态输入:
┌─────────────────────────────────────────┐
│  User Inputs:                          │
│  ┌─────────────────────────────────┐   │
│  │ Image prompts                  │   │
│  │ Text prompts                   │   │
│  │ Trajectory prompts             │   │
│  └─────────────────────────────────┘   │
│           │                              │
│           ↓                              │
│  Fine-grained control over video content│
└─────────────────────────────────────────┘
```

**ATI（Any Trajectory Instruction）：**
```
特点:
- 使用关键点和运动路径的局部控制
- 控制变形
```

**Instruction Fine-Tuning:**
```
方法:
┌─────────────────────────────────────────┐
│  Train with preference-based reward     │
│  models                                 │
│           │                              │
│           ↓                              │
│  Better instruction following          │
└─────────────────────────────────────────┘

局限:
- 依赖VLM或其他可幻觉的学习模型
- 限制实际有效性
```

#### 6.3.4 未来方向

```
研究方向:
1. 内在方法改进任务理解和指令遵循
2. 推理时"reasoning" over生成的视频patch和帧
3. 类似于推理语言模型的方法
```

### 6.4 Evaluating Video Models（评估视频模型）

#### 6.4.1 问题

```
缺乏统一评估框架:
┌─────────────────────────────────────────┐
│  现有指标关注:                          │
│  ┌─────────────────────────────────┐   │
│  │ 感知质量 (PSNR, SSIM, FID, LPIPS)│   │
│  │ 语义一致性 (CLIP, DINOv2)       │   │
│  └─────────────────────────────────┘   │
│                                         │
│  机器人需要的:                          │
│  ┌─────────────────────────────────┐   │
│  │ 物理一致性                       │   │
│  │ 预测准确性                       │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘

代理度量:
- 相关性（真实和预测策略成功率）
- 人类判断（定性、昂贵、有偏见）
```

#### 6.4.2 现有基准

```
下游任务对齐基准:
- WorldModelBench（VLM-based judge）
- 评估：指令遵循、物理adherence、常识

局限性:
- 环境复杂度和视觉质量受模拟器限制
- 缺乏细粒度机器人操作任务的评估能力
```

#### 6.4.3 未来方向

```
研究方向:
1. 机器人中心基准
2. 多维定量、高效、任务相关的视频模型评估指标
3. 评估管道：比较生成视频和对应3D场景重建的物理一致性
4. 自动化评估管道（减少人工判断）
```

### 6.5 Safe Content Generation（安全内容生成）

#### 6.5.1 问题

```
不安全内容类型:
┌─────────────────────────────────────────┐
│  Crime（犯罪）                           │
│  Offensive activities（冒犯性活动）     │
│  Violence（暴力）                       │
│  Misinformation（虚假信息）             │
└─────────────────────────────────────────┘

现有工作有限:
- 仅少数论文探索改进视频模型安全性的方法
- 缺乏类似LLM的广泛安全guardrails文献
```

#### 6.5.2 现有方法

**SAFEWatch：**
```
机制:
┌─────────────────────────────────────────┐
│  Enforce user-specified safety policies │
│  during video generation                │
└─────────────────────────────────────────┘

局限:
- 任务特定
- 限制对更广泛应用的适用性
```

#### 6.5.3 未来方向

```
研究方向:
1. 更通用的安全guardrails
2. 更全面的安全基准
   - 现有：犯罪、仇恨内容、隐私违规、虐待内容
   - 需要：更广泛的评估
3. 安全内容的生成式建模
4. 实时安全过滤机制
```

### 6.6 Safe Robot Interaction（安全机器人交互）

#### 6.6.1 机器人安全的分类

```
物理安全:
┌─────────────────────────────────────────┐
│  避免所有形式的碰撞                     │
└─────────────────────────────────────────┘

语义安全:
┌─────────────────────────────────────────┐
│  避免常识认为有害的情况                 │
│  例：向其他agent投掷尖锐物体            │
└─────────────────────────────────────────┘
```

#### 6.6.2 现有方法

**基于视频模型的安全评估：**
```
┌─────────────────────────────────────────┐
│  Robot action proposals → Video Model   │
│                              │          │
│                              ↓          │
│  Video predictions → Safety assessment  │
└─────────────────────────────────────────┘
```

**潜在空间安全过滤：**
```
┌─────────────────────────────────────────┐
│  Direct safety filtering in latent space│
│  of world models                        │
│                              │          │
│                              ↓          │
│  Predict failures → Preempt occurrence  │
└─────────────────────────────────────────┘
```

**不确定性量化：**
```
┌─────────────────────────────────────────┐
│  Quantify uncertainty in latent space   │
│                              │          │
│                              ↓          │
│  Guard against OOD hazards               │
│  and user-specified constraint violations│
└─────────────────────────────────────────┘
```

**局限性：**
- 主要局限于Markovian state-based world models
- 扩展到视频世界模型的spatiotemporal潜在空间是挑战
- 训练数据分布外的泛化仍是核心挑战

#### 6.6.3 未来方向

```
研究方向:
1. 扩展到视频世界模型的时空潜在空间
2. 长尾或安全关键场景的泛化
3. 分布外的安全保证
4. 实时安全过滤算法
5. 语义安全的形式化建模
```

### 6.7 Action Estimation（动作估计）

#### 6.7.1 问题

```
数据需求:
SOTA模仿学习机器人策略需要高质量数据
- 现实世界收集成本高

视频生成模型:
- 生成的视频不包含动作标注帧
- 动作标注对学习机器人策略至关重要

现有方法的局限性:
- 未能达到细粒度任务所需的高精度
- 阻碍视频模型在策略学习框架中的有效性
```

#### 6.7.2 两大方法及其局限

**1. Latent Action Models（潜在动作模型）**

```
工作原理:
┌─────────────────────────────────────────┐
│  Estimate actions between video frame   │
│  pair in (discrete) latent space       │
│                                         │
│  Defined by fixed set of action codes  │
│  (primitives)                          │
└─────────────────────────────────────────┘

局限:
1. 表达能力受codebook大小影响
2. 扩展codebook大小时:
   - 训练不稳定性
   - 更高计算成本
3. 潜在动作难以解释
   - 使数据curation和分析更具挑战性
4. 需要额外真实世界数据fine-tune
   - 对齐潜在动作和真实动作空间
```

**2. Inverse-Dynamics Models（IDMs，逆动力学模型）**

```
工作原理:
┌─────────────────────────────────────────┐
│  Learn to predict actions from videos   │
│  in supervised fashion                  │
│                                         │
│  Training: Action-labeled video data    │
└─────────────────────────────────────────┘

局限:
1. 需要大量训练数据
   - 充分覆盖机器人动作空间
2. 训练分布外泛化能力有限
   - 阻碍现实世界应用
3. 与其他模仿学习模型类似
```

#### 6.7.3 未来方向

```
研究方向:
1. 可解释的潜在动作
   - 小codebooks → 映射到可解释的条件输入
   - 但无法扩展到复杂任务

2. 新架构
   - 在不损害潜在动作空间表达能力的前提下
   - 诱导可解释的潜在动作

3. 健壮的训练程序
   - 对潜在动作模型的泛化至关重要

4. 半监督训练技术
   - 仅需少量人类标注
   - 高效训练可泛化的IDMs
```

### 6.8 Long Video Generation（长视频生成）

#### 6.8.1 问题

```
机器人任务时长: 通常为几分钟
SOTA视频模型限制:
┌─────────────────────────────────────────┐
│  Veo 3.1: 8 seconds                     │
│  Wan 2.5: 10 seconds                    │
└─────────────────────────────────────────┘

扩展方法的问题:
- 引入artifacts
- 降低时间一致性和物理一致性
```

#### 6.8.2 现有架构

**MALT（Memory-Augmented Latent Transformers）：**
```
方法:
┌─────────────────────────────────────────┐
│  Encode past segments into compact      │
│  latent memory vector                   │
│           │                              │
│           ↓                              │
│  Facilitate autoregressive generation   │
└─────────────────────────────────────────┘

局限:
- 仍然易受误差累积影响
```

**FramePack：**
```
方法:
┌─────────────────────────────────────────┐
│  Compress frame contexts based on       │
│  importance                            │
│           │                              │
│           ↓                              │
│  Establish early endpoints to anchor   │
│  generation process                    │
└─────────────────────────────────────────┘

目标: 减少漂移
```

**TTTVideo & LaCT（Test-Time Training）：**
```
方法:
┌─────────────────────────────────────────┐
│  Test-Time Training (TTT)               │
│  动态编码历史到模型权重或神经隐藏状态    │
│  during inference                      │
└─────────────────────────────────────────┘

问题:
- 视频长度增加，生成质量下降
```

**LCT（Long-Context Tuning）：**
```
方法:
┌─────────────────────────────────────────┐
│  Expand context window                  │
│  Maintain dense attention across       │
│  multi-shot scenes                      │
└─────────────────────────────────────────┘

局限:
- 自注意的二次复杂度
- 对最大生成长度施加计算上限
```

**MoC（Mixture of Contexts）：**
```
方法:
┌─────────────────────────────────────────┐
│  稀疏注意力路由机制                     │
│  将生成重构为信息检索任务                │
│  避免完全压缩历史                       │
└─────────────────────────────────────────┘
```

**Diffusion Forcing：**
```
方法:
┌─────────────────────────────────────────┐
│  训练模型去噪具有独立噪声水平的token   │
│           │                              │
│           ↓                              │
│  Enable variable-horizon generation     │
│  在策略rollouts中经验地提高稳定性       │
└─────────────────────────────────────────┘
```

**NUWA-XL：**
```
方法:
┌─────────────────────────────────────────┐
│  Hierarchical "diffusion-over-diffusion"│
│  结构                                  │
│  ┌─────────────────────────────────┐   │
│  │ Global model:                   │   │
│  │ Generate sparse keyframes       │   │
│  └─────────────────────────────────┘   │
│              │                          │
│              ↓                          │
│  ┌─────────────────────────────────┐   │
│  │ Local models:                   │   │
│  │ Recursively fill gaps           │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

#### 6.8.3 未来方向

```
研究方向:
1. 高效扩展记忆（上下文窗口）
2. 层次化视频生成框架
3. 自适应上下文管理
4. 误差累积缓解技术
5. 分钟级视频生成（现实世界应用所需）
```

### 6.9 Data Curation Costs（数据curation成本）

#### 6.9.1 问题

```
高质量数据的重要性:
- 训练高保真、物理一致视频模型
- 文本条件或动作条件视频模型需要广泛数据覆盖

互联网视频数据的问题:
┌─────────────────────────────────────────┐
│  WebVideo-10M, Panda-70M                │
│  ┌─────────────────────────────────┐   │
│  │ 聚焦规模而非质量                │   │
│  │                                │   │
│  │ 问题:                          │   │
│  │ - 不准确、非描述性视频caption   │   │
│  │ - 模糊视频                      │   │
│  │ - 快速镜头变化（时间不一致）   │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

#### 6.9.2 数据预处理流水线

```
三个阶段:

Stage 1: Video Splitting（视频分割）
┌─────────────────────────────────────────┐
│  Candidate video data                   │
│           │                              │
│           ↓                              │
│  ┌─────────────────────────────────┐   │
│  │ Classical shot detection tools  │   │
│  │ → Temporal segmentation         │   │
│  │ → Short continuous clips        │   │
│  └─────────────────────────────────┘   │
│           │                              │
│           ↓                              │
│  Remove too short clips                 │
└─────────────────────────────────────────┘

Stage 2: Video Filtering（视频过滤）
┌─────────────────────────────────────────┐
│  Clips → Video Filters                  │
│           │                              │
│           ↓                              │
│  ┌─────────────────────────────────┐   │
│  │ Learned models评估:            │   │
│  │ - Visual quality                │   │
│  │ - Text quality                  │   │
│  │ - Motion smoothness             │   │
│  │ - Jitter                        │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘

Stage 3: Video Annotation（视频标注）
┌─────────────────────────────────────────┐
│  Processed clips → VLMs                │
│                          │              │
│                          ↓              │
│  ┌─────────────────────────────────┐   │
│  │ Fine-tuned for higher-quality   │   │
│  │ video captioning                │   │
│  │                                │   │
│  │ Output:                        │   │
│  │ Text descriptions paired with  │   │
│  │ video data                     │   │
│  └─────────────────────────────────┘   │
│                          │              │
│                          ↓              │
│  Human supervision（增加成本）         │
└─────────────────────────────────────────┘
```

#### 6.9.3 新数据集

```
VidGen-1M, OpenVid-1M:
- 预处理技术
- 过滤视频（使用明确定义的质量分数）

质量分数评估:
┌─────────────────────────────────────────┐
│  - Aesthetics（美学）                   │
│  - Temporal consistency（时间一致性）   │
│  - Motion fidelity（运动保真度）        │
│  - Caption descriptiveness（caption描述性）│
│                                        │
│  目标: 识别更可能自然、无现实运动       │
│        的视频片段                       │
└─────────────────────────────────────────┘

局限:
- 相对较小 → 零样本泛化有限

SOTA模型:
- 使用专有的curated datasets
- 密集标注（VLMs和人类）
- 确保高质量控制

问题:
- VLMs易产生幻觉
- 人类标注数据收集昂贵
```

#### 6.9.4 机器人特定数据需求

```
失败演示的重要性:
┌─────────────────────────────────────────┐
│  高保真未来预测需要:                    │
│  ┌─────────────────────────────────┐   │
│  │ Successful task rollouts        │   │
│  │ Unsuccessful task rollouts      │   │
│  └─────────────────────────────────┘   │
│                                        │
│  失败演示的关键作用:                   │
│  - 训练视频模型faithfully执行策略动作   │
│  - 避免乐观偏差（总是预测成功）        │
│  - 没有此类数据:                       │
│    → 幻觉重新定位对象到成功位置        │
└─────────────────────────────────────────┘
```

#### 6.9.5 未来方向

```
研究方向:
1. Grounding VLMs
   - 最小化幻觉风险

2. Novel-view synthesis techniques
   - 从小量高质量视频高效扩展数据收集
   - 到新场景

3. 更准确的分割和过滤方法
   - Curating高度多样化数据集
   - 良好的时间和空间一致性

4. 自动化质量评估
   - 减少人类监督需求

5. 失败演示数据收集策略
   - 自主策略rollouts
   - 主动学习
```

### 6.10 Training and Inference Costs（训练和推理成本）

#### 6.10.1 问题

```
训练成本:
- SOTA视频模型需要大量计算资源
- 真实训练成本通常保密（闭源模型）
- 最具成本效益的SOTA开源视频模型:
  ┌─────────────────────────────────┐
  │ Open-Sora 2.0: $200k            │
  │ 需要数十万美元训练              │
  └─────────────────────────────────┘

因素:
┌─────────────────────────────────────────┐
│  - 数十亿参数                           │
│  - 主要成本贡献者                       │
└─────────────────────────────────────────┘

推理成本:
高保真视频模型（Veo, Wan, Gen-3）:
┌─────────────────────────────────────────┐
│  Veo 3: ~12 frames/second on NVIDIA A100 │
└─────────────────────────────────────────┘

机器人应用的影响:
┌─────────────────────────────────────────┐
│  视觉规划需要:                          │
│  ┌─────────────────────────────────┐   │
│  │ 闭环执行                         │   │
│  │ 实时反馈                         │   │
│  │ 提高鲁棒性                       │   │
│  └─────────────────────────────────┘   │
│                                        │
│  现状:                                  │
│  ┌─────────────────────────────────┐   │
│  │ 视频模型规划器:                 │   │
│  │ - 几秒钟生成可行动作轨迹        │   │
│  │ - 单个episode                   │   │
│  │ - 不够快，无法实时操作          │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

#### 6.10.2 现有优化技术

**空间和时间压缩：**

**Dreamer 4：**
```
方法:
┌─────────────────────────────────────────┐
│  Temporal attention applied sparsely   │
│  → Every fourth video frame             │
│  Decouple spatial and temporal attention│
└─────────────────────────────────────────┘
```

**OpenSora：**
```
方法:
┌─────────────────────────────────────────┐
│  Deep compression autoencoders          │
│  → Downsample input tokens at greater   │
│     ratios                             │
│  → Speed up inference by order of       │
│     magnitude                           │
└─────────────────────────────────────────┘
```

**Wan：**
```
方法:
┌─────────────────────────────────────────┐
│  Feature cache mechanism                │
│  → Enable chunk-based video synthesis   │
│  → Preserve temporal continuity        │
└─────────────────────────────────────────┘
```

**Shortcut Models（快捷模型）：**
```
方法:
┌─────────────────────────────────────────┐
│  Finer control over sampling steps     │
│  → Fewer steps                         │
│  → No significant degradation in       │
│    video quality                       │
└─────────────────────────────────────────┘
```

**Consistency Models（一致性模型）：**
```
方法:
┌─────────────────────────────────────────┐
│  Efficient video diffusion              │
│  → Single-step denoising process       │
└─────────────────────────────────────────┘
```

**经典优化技术：**
- Quantization（量化）
- Model distillation（模型蒸馏）

#### 6.10.3 未来方向

```
研究方向:
1. 更快训练和推理技术
2. 更高效的视频生成架构
3. 推理时优化
4. 硬件加速（专用芯片）
5. 分布式训练和推理框架
6. 实时视觉规划能力
```

---

## 第七部分：核心参考文献与链接

### 关键论文链接

#### Diffusion/Flow-Matching Models:
1. **Denoising Diffusion Probabilistic Models** - Ho et al. 2020
   - https://arxiv.org/abs/2006.11239
   
2. **Flow Matching for Generative Modeling** - Lipman et al. 2023
   - https://arxiv.org/abs/2210.02747

#### Video Diffusion Models:
3. **Stable Video Diffusion** - Blattmann et al. 2023
   - https://arxiv.org/abs/2311.15127
   
4. **Video Diffusion Models** - Ho et al. 2022
   - https://arxiv.org/abs/2204.03458

#### DiT (Diffusion Transformers):
5. **Scalable Diffusion Models with Transformers** - Peebles & Xie 2023
   - https://arxiv.org/abs/2212.09748

#### Video Models in Robotics:
6. **DreamGen: Unlocking Generalization in Robot Learning through Video World Models** - Jang et al. 2025
   - https://arxiv.org/abs/2505.12705
   
7. **Ctrl-World: A Controllable Generative World Model for Robot Manipulation** - Guo et al. 2025
   - https://arxiv.org/abs/2510.10125

#### World Models:
8. **Genie: Generative Interactive Environments** - Bruce et al. 2024
   - https://arxiv.org/abs/2401.13667
   
9. **Dreamer 4: Training Agents Inside of Scalable World Models** - Hafner et al. 2025
   - https://arxiv.org/abs/2509.24527

#### Visual Planning:
10. **Video Language Planning** - Du et al. 2023
    - https://arxiv.org/abs/2310.10625

11. **FLIP: Flow-centric Generative Planning as General-Purpose Manipulation World Model** - Gao et al. 2024
    - https://arxiv.org/abs/2412.08261

#### Policy Evaluation:
12. **WorldGym: World Model as an Environment for Policy Evaluation** - Quevedo et al. 2025
    - https://arxiv.org/abs/2506.00613

13. **Scalable Policy Evaluation with Video World Models** - Tseng et al. 2025
    - https://arxiv.org/abs/2511.11520

#### Evaluation Benchmarks:
14. **VBench: Comprehensive Benchmark Suite for Video Generative Models** - Huang et al. 2024
    - https://arxiv.org/abs/2311.17923

15. **WorldModelBench: Judging Video Generation Models as World Models** - Li et al. 2025
    - https://arxiv.org/abs/2502.20694

16. **Physics-IQ: Assessing Physical Commonsense for Video Generation** - Motamed et al. 2025
    - https://arxiv.org/abs/2501.09038

#### Physical Commonsense:
17. **VideoPhy: Evaluating Physical Commonsense for Video Generation** - Bansal et al. 2024
    - https://arxiv.org/abs/2406.03520

18. **PhyGenBench: Towards World Simulator** - Meng et al. 2024
    - https://arxiv.org/abs/2410.05363

#### Long Video Generation:
19. **MALT Diffusion: Memory-Augmented Latent Transformers for Any-Length Video Generation** - Yu et al. 2025
    - https://arxiv.org/abs/2502.12632

20. **Mixture of Contexts for Long Video Generation** - Cai et al. 2025
    - https://arxiv.org/abs/2508.21058

#### Data Curation:
21. **Koala-36M: A Large-Scale Video Dataset Improving Consistency** - Wang et al. 2025
    - https://arxiv.org/abs/2501.00000

22. **VidGen-1M: A Large-Scale Dataset for Text-to-Video Generation** - Tan et al. 2024
    - https://arxiv.org/abs/2408.02629

---

## 第八部分：总结与展望

### 8.1 核心贡献

这篇综述文章系统性地回顾了视频生成模型作为**embodied world models（具身世界模型）**在机器人学中的应用，主要贡献包括：

1. **全面的模型分类：**
   - 非扩散视频模型 vs 扩散/流匹配视频模型
   - 隐式视频世界模型 vs 显式视频世界模型

2. **四大机器人应用：**
   - 模仿学习中的数据生成和动作预测
   - 强化学习中的动力学和奖励建模
   - 可扩展策略评估
   - 视觉规划

3. **详细的数学公式解析：**
   - 扩散模型的数学基础
   - Classifier-free guidance
   - 评估指标的数学定义

4. **十大开放挑战：**
   - 幻觉和物理违规
   - 不确定性量化
   - 指令遵循
   - 评估框架
   - 安全内容生成
   - 安全机器人交互
   - 动作估计
   - 长视频生成
   - 数据curation成本
   - 训练和推理成本

### 8.2 未来研究方向总结

```
短期（1-2年）:
┌─────────────────────────────────────────┐
│  1. 改进物理一致性的架构               │
│  2. 细粒度机器人任务评估基准           │
│  3. 高效的动作估计方法                 │
│  4. 安全内容生成的guardrails            │
└─────────────────────────────────────────┘

中期（2-5年）:
┌─────────────────────────────────────────┐
│  1. 分钟级视频生成                     │
│  2. 推理时的不确定性表达               │
│  3. 机器人中心的世界模型评估框架       │
│  4. 实时视觉规划能力                   │
└─────────────────────────────────────────┘

长期（5年以上）:
┌─────────────────────────────────────────┐
│  1. 具有可证明安全保证的机器人控制     │
│  2. 通用的机器人世界模型平台            │
│  3. 自主数据收集和curation系统          │
│  4. 物理常识的内在理解                 │
└─────────────────────────────────────────┘
```

### 8.3 对领域的影响

这项综述对以下领域有重要影响：

1. **机器人学：**
   - 减少对昂贵专家演示的依赖
   - 加速策略开发和评估
   - 改进安全性和可靠性

2. **计算机视觉：**
   - 推动物理一致的视频生成
   - 改进视频理解和预测

3. **强化学习：**
   - 提供高保真世界模型
   - 改进样本效率

4. **人工智能安全：**
   - 视频模型的安全保证
   - 不确定性量化

---

这篇文章为视频生成模型在机器人学中的应用提供了全面、深入的视角，既涵盖了技术细节，又指出了开放挑战和未来方向，是研究者的重要参考资源。