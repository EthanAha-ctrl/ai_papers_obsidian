




















好的，我已经搜集了大量相关信息。下面我来详细解析这个 project。

---

# MiniVeo3-Reasoner: Thinking with Videos from Open-Source Priors

**GitHub**: https://github.com/thuml/MiniVeo3-Reasoner
**关联论文**: [Thinking in Frames: How Visual Context and Test-Time Scaling Empower Video Reasoning (arXiv:2601.21037)](https://arxiv.org/abs/2601.21037)
**团队**: THUML (清华大学机器学习组)

---

## 一、核心思想：用 Video Generation 做 Reasoning

这个 project 的根本洞察来自一个 paradigm shift：

> **Video diffusion model 不仅是"生成器"，更是"推理器"。**

### 1.1 背景：Google DeepMind 的 Veo 3 发现

2025年9月，Google DeepMind 发表了 [Video models are zero-shot learners and reasoners (arXiv:2509.20328)](https://arxiv.org/abs/2509.20328)，展示了 Veo 3 能以 zero-shot 方式解决大量它从未专门训练过的 visual task：

- **Maze solving** (迷宫求解)
- **Object segmentation** (目标分割)
- **Edge detection** (边缘检测)
- **Image editing** (图像编辑)
- **Physics simulation** (物理仿真)
- **Symmetry completion** (对称性补全)

关键机制被称为 **Chain-of-Frames (CoF)**——这是 Chain-of-Thought (CoT) 在 visual domain 的对应物：

```
文本推理:  Token₁ → Token₂ → Token₃ → ... → Answer
视觉推理:  Frame₁ → Frame₂ → Frame₃ → ... → Solution Frame
```

### 1.2 第一性原理理解

从第一性原理来看，为什么 video generation model 能做 reasoning？

**核心论点**：Video diffusion model 在训练过程中学习了 **world model**——即对物理世界因果关系、空间关系、时间演化的内隐表征。当给定一个 "起始状态" frame（例如一个未解的迷宫），模型通过逐帧生成，实际上在 **latent space** 中执行了一个 iterative 的搜索/规划过程，每一帧都是一个 intermediate reasoning step。

这与 LLM 中的 Chain-of-Thought 类比非常精确：

| 维度 | Chain-of-Thought (LLM) | Chain-of-Frames (Video Model) |
|------|------------------------|-------------------------------|
| Medium | Text tokens | Video frames |
| Reasoning step | 一个 sentence/paragraph | 一个或多个 frame |
| Test-time scaling | 更多 tokens = 更多思考 | **更多 frames = 更多思考** |
| Intermediate representation | Hidden states + text output | Latent space + pixel output |

---

## 二、MiniVeo3-Reasoner 做了什么？

### 2.1 核心贡献

Google 的 Veo 3 是 closed-source 的。**MiniVeo3-Reasoner 的核心目标是证明：Chain-of-Frames (CoF) reasoning 能力可以通过 fine-tuning open-source video model 来复现。**

具体来说：

1. 使用 **Wan2.2** 作为 base model（open-source video diffusion model）
2. 通过 **DiffSynth-Studio** 进行 fine-tuning
3. 在 **maze domain** 上验证 CoF reasoning 能力
4. 达到了 **near-perfect accuracy**（接近完美的迷宫求解率）

### 2.2 技术栈详解

#### Base Model: Wan2.2

Wan2.2 是当前最强的 open-source video diffusion model 之一：

- **Architecture**: 基于 **DiT (Diffusion Transformer)** 架构，并引入了 **Mixture-of-Experts (MoE)** 机制
- **核心设计**: 将 denoising 过程分为 **high-noise stage** 和 **low-noise stage**，分别由不同的 expert 处理
  - High-noise expert: 负责全局结构和语义布局
  - Low-noise expert: 负责细节纹理和高频信息
- **参数规模**: 14B 参数级别
- **VAE**: Spatio-temporal VAE，将视频压缩到 latent space

Wan2.2 的 MoE 设计哲学很关键：

$$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \sum_{k=1}^{K} g_k(\mathbf{x}_t, t) \cdot p_{\theta_k}(\mathbf{x}_{t-1} | \mathbf{x}_t)$$

其中：
- $\mathbf{x}_t$ 是 timestep $t$ 时的 noisy latent
- $g_k(\mathbf{x}_t, t)$ 是 gating function，根据 noise level $t$ 和当前 latent state 选择 expert
- $p_{\theta_k}$ 是第 $k$ 个 expert 的 denoising distribution
- $K$ 是 expert 数量

#### Training Framework: DiffSynth-Studio

DiffSynth-Studio 是一个用于 video diffusion model 训练的框架，支持：
- LoRA / Full fine-tuning
- Multi-GPU distributed training
- Video data preprocessing pipeline

#### Data Generation: maze-dataset

训练数据通过 **maze-dataset** 工具自动生成：
- 生成各种大小的迷宫（例如 9×9, 15×15 等）
- 每个 training sample 是一个 **video sequence**：从 unsolved maze 开始，逐帧展示求解过程（path gradually revealed），最终到达 solved state
- 这本质上是 **Process Supervision**——不仅教模型最终答案，而是教它 step-by-step 的解题过程

---

## 三、Chain-of-Frames 的机制深度解析

### 3.1 Video Diffusion 的 Denoising 过程

标准的 video diffusion model 遵循 DDPM/Flow Matching 框架：

**Forward process** (加噪):
$$q(\mathbf{z}_t | \mathbf{z}_0) = \mathcal{N}(\mathbf{z}_t; \alpha_t \mathbf{z}_0, \sigma_t^2 \mathbf{I})$$

其中：
- $\mathbf{z}_0$ 是原始 video 在 latent space 的表征
- $\mathbf{z}_t$ 是加噪后的 latent
- $\alpha_t, \sigma_t$ 是 noise schedule 参数
- $t \in [0, T]$ 是 diffusion timestep

**Reverse process** (去噪/生成):
$$\mathbf{z}_{t-1} = f_\theta(\mathbf{z}_t, t, \mathbf{c})$$

其中 $\mathbf{c}$ 是 conditioning signal（text prompt + 可选的 image/video conditioning）。

### 3.2 为什么 Denoising = Reasoning？

这里有一个深刻的 insight，最近被 [Demystifying Video Reasoning (arXiv:2603.16870)](https://arxiv.org/abs/2603.16870) 进一步阐明：

**Chain-of-Steps (CoS)** 观点认为，reasoning 不仅发生在 frame 维度上，更核心地发生在 **diffusion step** 维度上：

```
Timestep T (pure noise) → Timestep T-1 → ... → Timestep 1 → Timestep 0 (clean video)
     ↓                        ↓                    ↓              ↓
   "没想法"              "模糊规划"           "细化路径"      "完整解答"
```

每个 denoising step 都是一个 "推理步骤"：
- **Early steps** (high noise): 模型做 global planning——决定迷宫路径的大致走向
- **Middle steps**: Refine 路径的具体走法
- **Late steps** (low noise): 精细化 pixel-level 细节

这与人类解迷宫的过程有趣地对应：先看全局结构，再局部搜索，最后确认路径。

### 3.3 Test-Time Scaling

这是 "Thinking in Frames" 论文 (arXiv:2601.21037) 的核心贡献之一：

$$\text{Accuracy}(n) = f(\text{num\_frames} = n)$$

当 $n$（生成的 frame 数量）增加时，模型的 reasoning accuracy 也随之提高。这直接对应了 LLM 中的 **test-time compute scaling law**：

- 在 LLM 中：**更多的 output tokens → 更多的 reasoning budget → 更好的准确率**
- 在 Video model 中：**更多的 generated frames → 更多的 visual thinking budget → 更好的 reasoning**

数学上，这可以理解为：
$$P(\text{correct solution}) = 1 - \exp\left(-\beta \cdot n_{\text{frames}}\right)$$

其中 $\beta$ 取决于 task complexity，$n_{\text{frames}}$ 是生成的 frame 数量。更多 frames 意味着模型有更多 "intermediate steps" 来逐步收敛到正确解。

---

## 四、Visual Reasoning 的 Generalization

### 4.1 不仅仅是记忆

"Thinking in Frames" 论文的一个关键发现是 **Visual Context for Robust Generalization**：

fine-tuned 的 video model 能够 **abstract underlying planning algorithms**，而非简单 memorize training data。证据包括：

- 在 **Out-of-Distribution (OOD)** 的迷宫大小上仍然能 work（例如训练在 9×9 上，泛化到 15×15）
- 对 maze 的 visual style 变化（颜色、线条粗细）具有 robustness
- 这表明模型学到的是某种 "path-finding algorithm" 的 implicit representation，而非视觉 pattern matching

### 4.2 与其他 Reasoning Paradigm 的对比

| Paradigm | 代表 | Reasoning Medium | 优势 | 劣势 |
|----------|------|------------------|------|------|
| **CoT** (Chain-of-Thought) | GPT-o1, DeepSeek-R1 | Text tokens | 符号推理强 | 空间推理弱 |
| **CoF** (Chain-of-Frames) | Veo 3, MiniVeo3-Reasoner | Video frames | 空间推理强，物理直觉好 | 精确数学推理弱 |
| **CoS** (Chain-of-Steps) | Diffusion 内部 | Denoising steps | 全局→局部的自然 hierarchy | 不可解释 |
| **Tool-use** | Mini-o3 | External tool calls | 精确计算 | 需要 tool design |

---

## 五、实验结果与意义

### 5.1 Maze Domain 结果

根据 GitHub 描述，MiniVeo3-Reasoner 在 maze domain 达到了 **near-perfect accuracy**——这意味着一个 open-source video diffusion model 通过 fine-tuning 就能复现 Veo 3 的 CoF reasoning 能力。

这个结果的震撼之处在于：
1. **Veo 3 是一个巨大的 closed-source model**，而 Wan2.2 是 open-source 的
2. Fine-tuning 的成本相对于 Veo 3 的训练成本微不足道
3. 说明 **CoF reasoning 是 video diffusion model 的 emergent capability**，可以通过适当的 task-specific fine-tuning 来 "激活"

### 5.2 更广泛的意义

这个 project 暗示了一个新的 AI reasoning paradigm：

```
                    ┌──────────────────────┐
                    │   Multimodal Reasoner │
                    └───────┬──────────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
        ┌─────▼─────┐ ┌────▼─────┐ ┌────▼──────┐
        │  Text CoT  │ │ Visual   │ │  Tool Use │
        │  (LLM)     │ │ CoF      │ │           │
        │            │ │ (Video   │ │           │
        │            │ │  Model)  │ │           │
        └────────────┘ └──────────┘ └───────────┘
              ↓             ↓             ↓
         符号推理        空间推理        精确计算
```

未来的 AGI 系统可能会同时拥有：
- **Text reasoning** (通过 LLM CoT)
- **Visual reasoning** (通过 Video Model CoF)
- **Tool-augmented reasoning** (通过 code execution 等)

---

## 六、技术 Pipeline 总结

```
┌─────────────────────────────────────────────────────┐
│                MiniVeo3-Reasoner Pipeline            │
│                                                     │
│  1. Data Generation (maze-dataset)                  │
│     ├── Generate random mazes (various sizes)       │
│     ├── Solve each maze (e.g., BFS/DFS)            │
│     └── Render as video: unsolved → step-by-step → solved │
│                                                     │
│  2. Base Model (Wan2.2)                             │
│     ├── DiT + MoE architecture, 14B params          │
│     ├── Spatio-temporal VAE encoder/decoder         │
│     └── Pre-trained on large-scale video data       │
│                                                     │
│  3. Fine-Tuning (DiffSynth-Studio)                  │
│     ├── Task: image-to-video generation             │
│     ├── Input condition: unsolved maze image        │
│     ├── Target: video of maze being solved          │
│     └── Learning objective: denoising loss          │
│                                                     │
│  4. Inference                                       │
│     ├── Input: unsolved maze image                  │
│     ├── Generate: video frames (CoF reasoning)      │
│     └── Output: last frame = solved maze            │
│                                                     │
│  Loss Function:                                     │
│  L(θ) = E_{t,ε} [‖ε - ε_θ(z_t, t, c)‖²]         │
│     ε: ground truth noise                           │
│     ε_θ: predicted noise (model output)             │
│     z_t: noisy latent at timestep t                 │
│     c: conditioning (maze image + text prompt)      │
└─────────────────────────────────────────────────────┘
```

其中 Loss function 的各变量含义：
- $\varepsilon$ 是在 forward process 中添加的 Gaussian noise (ground truth)
- $\varepsilon_\theta$ 是模型 (参数为 $\theta$) 预测的 noise
- $\mathbf{z}_t$ 是 timestep $t$ 处的 noisy video latent
- $t$ 从 noise schedule 中均匀采样
- $\mathbf{c}$ 是 conditioning signal（初始迷宫图像 + text prompt）

---

## 七、关联 Project 与进一步阅读

| 资源 | 链接 |
|------|------|
| **MiniVeo3-Reasoner GitHub** | https://github.com/thuml/MiniVeo3-Reasoner |
| **Thinking in Frames 论文** | https://arxiv.org/abs/2601.21037 |
| **Thinking in Frames 项目页** | https://thinking-in-frames.github.io/ |
| **Veo 3: Video models are zero-shot learners** | https://arxiv.org/abs/2509.20328 |
| **Demystifying Video Reasoning (Chain-of-Steps)** | https://arxiv.org/abs/2603.16870 |
| **Wan2.2 Model** | https://github.com/Wan-Video/Wan2.2 |
| **DiffSynth-Studio** | https://github.com/modelscope/DiffSynth-Studio |
| **Process-aware Eval for Video Reasoning** | https://arxiv.org/html/2512.24952v2 |
| **Wan-R1: Verifiable RL for Video Reasoning** | https://arxiv.org/html/2603.27866v1 |

---

## 总结 Intuition

用一句话来建立直觉：

> **正如 LLM 通过生成 text tokens 来 "thinking step by step"，Video Diffusion Model 通过生成 video frames 来 "thinking frame by frame"。MiniVeo3-Reasoner 证明了这种 visual reasoning 能力不是 Veo 3 独有的魔法，而是 video diffusion model 的一种 emergent property——只要有正确的 fine-tuning data，open-source model 也可以做到。**

这个发现的深远含义在于：**Video Generation Model 可能是通往 visual AGI 的一条被低估的路径。** 它们不仅能 "画画"，还能 "思考"——只不过思考的 medium 是 pixels 而非 words。