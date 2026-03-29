# Skild AI 公司深度解析

让我为您搜索这家公司的详细信息。# Skild AI 公司深度解析

## 公司概况

**Skild AI** 是一家专注于 **embodied AI（具身智能）** 和 **robotics foundation models（机器人基础模型）** 的初创公司，成立于2023年，总部位于美国。

根据公开信息，该公司由卡内基梅隆大学的研究人员创立，致力于开发能够使机器人具备通用智能能力的AI系统。

---

## 核心技术方向

### 1. 机器人基础模型

Skild AI 的核心目标是构建一个 **通用的机器人大脑**，类似于大语言模型（LLM）之于文本领域，但他们专注于物理世界的机器人控制。

其核心思想可以表示为：

$$\pi_\theta(a_t | o_t, g) = \text{Neural Network Policy}$$

其中：
- $\pi_\theta$ = 可学习的策略函数，参数为 $\theta$
- $a_t$ = 时刻 $t$ 的动作
- $o_t$ = 时刻 $t$ 的观测
- $g$ = 目标指令

### 2. 具身智能

Embodied AI 强调智能体通过与物理环境的交互来学习和执行任务，区别于纯软件的AI系统。

---

## 技术架构详解

### 基于Transformer的机器人策略

Skild AI 采用类似 **Vision-Language-Action (VLA)** 的架构：

```
┌─────────────────────────────────────────────────────────┐
│                    Skild AI Architecture                 │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   ┌──────────┐    ┌──────────┐    ┌──────────────┐      │
│   │  Vision  │    │  Language │    │   Proprio    │      │
│   │ Encoder  │    │  Encoder  │    │   Encoder    │      │
│   │ (ViT)    │    │ (BERT/LLM)│    │  (MLP)       │      │
│   └────┬─────┘    └────┬─────┘    └──────┬───────┘      │
│        │               │                  │              │
│        └───────────────┼──────────────────┘              │
│                        │                                 │
│                        ▼                                 │
│              ┌─────────────────┐                         │
│              │   Transformer   │                         │
│              │   Backbone      │                         │
│              │   (Cross-Attn)  │                         │
│              └────────┬────────┘                         │
│                       │                                  │
│                       ▼                                  │
│              ┌─────────────────┐                         │
│              │   Action Head   │                         │
│              │   (MLP Decoder) │                         │
│              └─────────────────┘                         │
│                       │                                  │
│                       ▼                                  │
│              Robot Action Commands                       │
│         (joint positions, velocities, etc.)              │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 训练数据规模

Skild AI 声称使用了大规模的机器人操作数据集进行训练，数据来源包括：

| 数据类型 | 规模估计 | 来源 |
|---------|---------|------|
| Simulation Data | ~1M+ trajectories | Isaac Gym, MuJoCo |
| Real Robot Data | ~100K+ trajectories | Multiple robot platforms |
| Human Demonstrations | ~50K+ videos | Teleoperation, YouTube |

---

## 核心创新点

### 1. Scalable Robot Learning

遵循 **Scaling Laws** 在机器人领域的应用：

$$L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + E$$

其中：
- $N$ = 模型参数量
- $D$ = 训练数据量
- $L$ = 损失函数值
- $A, B, E, \alpha, \beta$ = 经验常数

### 2. Cross-Embodiment Learning

核心创新在于 **跨机器人平台泛化**，即同一个模型可以控制不同类型的机器人：

$$\mathcal{L}_{\text{total}} = \sum_{e \in \mathcal{E}} \lambda_e \cdot \mathcal{L}_e(\theta)$$

其中：
- $\mathcal{E}$ = 所有机器人embodiment的集合
- $\lambda_e$ = 每种embodiment的权重
- $\mathcal{L}_e$ = 特定embodiment的损失

---

## 商业模式与竞争格局

### 融资情况

据报道，Skild AI 已获得显著融资：

| 轮次 | 金额 | 投资方 |
|------|------|--------|
| Seed | ~$10M | Various VCs |
| Series A | ~$300M (rumored) | Lightspeed, Coatue, etc. |

估值据说达到 **$1B+**，成为 **unicorn** 级别公司。

### 竞争对手

| 公司 | 核心技术 | 差异化 |
|------|---------|--------|
| **Google DeepMind (RT-X)** | Robotics Transformer | 数据规模大 |
| **OpenAI (with Figure)** | VLA models | 商业合作紧密 |
| **Tesla Optimus** | End-to-end neural nets | 垂直整合 |
| **Physical Intelligence** | Foundation models | 类似定位 |
| **Skild AI** | Scalable foundation model | CMU背景，跨平台 |

---

## 第一性原理分析

### 从根本问题出发

机器人智能的本质问题是：

$$\max_\pi \mathbb{E}_{\tau \sim p_\pi(\tau)} \left[ \sum_{t=0}^{T} \gamma^t r(s_t, a_t) \right]$$

其中：
- $\pi$ = 策略
- $\tau$ = 轨迹 $(s_0, a_0, s_1, a_1, \ldots)$
- $\gamma$ = 折扣因子
- $r(s_t, a_t)$ = 奖励函数

### Skild AI 的解决思路

**传统方法的问题：**
1. 每个任务需要单独训练 → 不scalable
2. 需要大量task-specific engineering → 成本高
3. 泛化能力差 → 无法适应新环境

**Foundation Model 方法：**
1. 大规模预训练 → 学到通用representation
2. Multi-task learning → 单一模型多任务
3. Zero-shot/Few-shot transfer → 快速适应新任务

---

## 技术细节深入

### 视觉编码器

通常采用 **Vision Transformer (ViT)** 架构：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- $Q, K, V \in \mathbb{R}^{N \times d_k}$ = Query, Key, Value 矩阵
- $N$ = patch数量
- $d_k$ = 特征维度

输入图像被分成 patches：
$$\mathbf{x} = [\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_N], \quad \mathbf{x}_i \in \mathbb{R}^{P^2 \times C}$$

- $P$ = patch size（如 16×16）
- $C$ = 通道数（RGB = 3）

### 动作空间设计

机器人动作空间的表示至关重要：

| 动作空间类型 | 表示 | 优缺点 |
|------------|------|--------|
| Joint Position | $\mathbf{a} \in \mathbb{R}^{n_{joints}}$ | 精确但需逆运动学 |
| End-effector Pose | $\mathbf{a} \in SE(3)$ | 直观但需运动规划 |
| Velocity | $\mathbf{a} \in \mathbb{R}^6$ | 平滑但难以精确定位 |
| Delta Position | $\mathbf{a} = \Delta \mathbf{q}$ | 平衡精度和平滑性 |

$SE(3)$ 表示特殊欧几里得群：
$$SE(3) = \left\{ \begin{bmatrix} R & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix} : R \in SO(3), \mathbf{t} \in \mathbb{R}^3 \right\}$$

---

## 训练方法论

### 模仿学习

Skild AI 大量使用模仿学习方法：

**Behavior Cloning 目标函数：**

$$\mathcal{L}_{BC}(\theta) = \mathbb{E}_{(\mathbf{o}, \mathbf{a}) \sim \mathcal{D}} \left[ \| \pi_\theta(\mathbf{o}) - \mathbf{a} \|^2 \right]$$

其中：
- $\mathcal{D}$ = expert demonstration 数据集
- $\mathbf{o}$ = 观测
- $\mathbf{a}$ = expert动作

### Diffusion Policy

最新研究采用 **Diffusion-based Policy**：

前向扩散过程：
$$q(\mathbf{a}_t | \mathbf{a}_{t-1}) = \mathcal{N}(\mathbf{a}_t; \sqrt{1-\beta_t}\mathbf{a}_{t-1}, \beta_t \mathbf{I})$$

反向去噪过程：
$$p_\theta(\mathbf{a}_{t-1} | \mathbf{a}_t) = \mathcal{N}(\mathbf{a}_{t-1}; \mu_\theta(\mathbf{a}_t, t), \sigma_t^2 \mathbf{I})$$

其中：
- $\beta_t$ = noise schedule
- $\mu_\theta$ = 学习的均值网络
- $\sigma_t$ = 方差

---

## 潜在应用场景

| 领域 | 具体应用 | 技术要求 |
|------|---------|---------|
| **Manufacturing** | 装配、质检、包装 | 高精度、重复性 |
| **Logistics** | 分拣、搬运、码垛 | 速度、鲁棒性 |
| **Healthcare** | 手术辅助、护理 | 安全性、精确性 |
| **Home** | 清洁、烹饪、陪伴 | 适应性、安全性 |
| **Agriculture** | 采摘、播种 | 户外环境适应 |

---

## 参考资源

### 官方与新闻
- Skild AI 官网：https://www.skild.ai
- TechCrunch 报道：https://techcrunch.com/2024/07/09/skild-ai-raises-300m-to-build-a-brain-for-robots/
- Reuters 报道：https://www.reuters.com/technology/skild-ai-raises-300-mln-valuation-over-1-bln-source-says-2024-07-09/

### 相关论文与技术
- RT-X: https://robotics-transformer-x.github.io
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu
- ACT (Action Chunking Transformer): https://tonyzhaozh.github.io/aloha/

### 竞争对手
- Physical Intelligence: https://www.physicalintelligence.company
- Google DeepMind Robotics: https://deepmind.google/discover/blog/rt-x
- Tesla Bot: https://www.tesla.com/AI

---

## 总结

**Skild AI** 是一家致力于构建 **机器人基础模型** 的前沿AI公司，其核心愿景是：

> 创建一个统一的、可扩展的AI系统，使任何机器人都能执行广泛的物理任务，无需为每个任务单独编程。

这代表了从 **specialized robotics** 向 **general-purpose robotics** 的范式转变，类似于 GPT 在NLP领域带来的变革。