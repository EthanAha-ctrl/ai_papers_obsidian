让我详细讲解这篇关于Dream-VL和Dream-VLA的论文。这是一个非常有意思的工作，它展示了diffusion-based LLMs在vision-language和robotics领域的强大潜力。

## 一、核心动机与问题背景

### 1.1 Autoregressive VLM/VLA的局限性

论文指出了当前主流Autoregressive (AR) VLMs存在的三个关键问题：

**数学化的问题表述**：

- **Sequential generation bottleneck**: AR模型基于next-token prediction，其目标函数为：
  
  $$p(x_t|x_{<t}) = \prod_{i=1}^{T} p(x_i|x_{<i})$$
  
  其中$x_t$是第$t$个token，$x_{<t}$是之前的所有token。这种sequential方式在需要长horizon planning的任务中表现不佳。

- **Error accumulation**: 在推理过程中，每个token的生成依赖于前一个token，因此误差会累积：
  
  $$\epsilon_{cumulative} = \sum_{i=1}^{T} \epsilon_i$$
  
  当$T$（规划长度）增大时，$\epsilon_{cumulative}$会迅速增大，严重影响最终性能。

- **Parallel decoding limitation**: AR模型无法同时生成多个action，这对于需要快速响应的robotic control来说是致命的。

### 1.2 Diffusion Language Models的优势

**Diffusion过程的基本原理**：

Denoising Diffusion Probabilistic Models (DDPM) 通过前向添加噪声和反向去噪过程来学习数据分布：

- **Forward process (加噪)**：
  $$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$
  
  其中$\beta_t$是噪声调度参数，$x_t$是时刻$t$的加噪数据。

- **Reverse process (去噪)**：
  $$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t,t), \Sigma_\theta(x_t,t))$$
  
  通过神经网络$\mu_\theta$预测去噪方向。

对于discrete diffusion（本文使用的），使用**masked diffusion**策略：

$$p_\theta(x_0|x_T) = \prod_{t=T}^{1} p_\theta(x_{t-1}|x_t)$$

其中$x_0$是clean sequence，$x_T$是completely masked sequence，$x_t$是部分masked的中间状态。

**Diffusion在planning中的优势**：

1. **Bidirectional attention**: 模型可以看到全局上下文，不像AR模型只能看到前序tokens
  
2. **Global coherence**: Iterative refinement过程鼓励生成全局一致的输出

3. **Parallel generation**: 可以同时预测多个actions或tokens

## 二、Dream-VL架构详解

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    Dream-VL Architecture                 │
├─────────────────────────────────────────────────────────┤
│  Vision Input → Qwen2ViT → Vision Features              │
│                          ↓                               │
│                          │ Concatenate                   │
│                          ↓                               │
│  Text Input  → Embedding → Text Features →              │
│                          ↓                               │
│                  Combined Features                       │
│                          ↓                               │
│           Dream 7B (Diffusion Transformer)               │
│                          ↓                               │
│              Masked Diffusion Denoising                  │
│                          ↓                               │
│                    Output Generation                      │
└─────────────────────────────────────────────────────────┘
```

### 2.2 关键组件

**1. Vision Encoder (Qwen2ViT)**:

Qwen2ViT使用标准的ViT架构：
- **Patch embedding**: 将图像$H \times W \times 3$分割成$N = \frac{H \times W}{P^2}$个patches
- **Positional encoding**: 可学习的positional embeddings
- **Transformer layers**: 使用self-attention和feed-forward networks

Vision特征提取：
$$z_v = \text{Qwen2ViT}(I) \in \mathbb{R}^{L_v \times d}$$

其中$I$是输入图像，$L_v$是视觉token的数量，$d$是hidden dimension。

**2. Text Embedding**:

$$z_t = \text{Embedding}(x_{\text{text}}) \in \mathbb{R}^{L_t \times d}$$

其中$L_t$是文本长度。

**3. Feature Fusion**:

$$z = [z_v; z_t] \in \mathbb{R}^{(L_v + L_t) \times d}$$

使用简单的concatenation，与LLaVA类似。

**4. Diffusion Backbone (Dream 7B)**:

Dream 7B是diffusion-based LLM，使用**masked discrete diffusion**：

- **Forward process**: 随机mask掉输入序列的tokens
- **Reverse process**: 根据上下文预测被mask的tokens

训练目标（**Discrete Diffusion Loss**）：

$$\mathcal{L}_{\text{diffusion}} = \mathbb{E}_{q(x_t|x_0)} \left[ -\log p_\theta(x_{t-1}|x_t) \right]$$

其中：
- $x_0$是clean sequence（包含vision + text tokens）
- $x_t$是noisy sequence（部分tokens被mask）
- $p_\theta(x_{t-1}|x_t)$是模型预测的去噪分布

### 2.3 三阶段训练策略

论文采用了三阶段训练范式，每个阶段的参数和目标如下：

**Stage 1: LCS (Language-Image Contrastive Self-training)**

| Parameter | Value |
|-----------|-------|
| Dataset | 558K samples |
| Trainable Parameters | Projector: 25.7M |
| Batch Size | 512 |
| Learning Rate | 1e-3 |
| Epochs | 1 |

**目标**：训练vision-language projector，对齐vision和text embeddings。

**损失函数**（类似CLIP的对比学习）：

$$\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\text{sim}(z_v, z_t)/\tau)}{\sum_{j} \exp(\text{sim}(z_v, z_t^{(j)})/\tau)}$$

其中$\text{sim}(\cdot,\cdot)$是cosine similarity，$\tau$是temperature。

**Stage 2: Single Image (SI) Fine-tuning**

| Parameter | Value |
|-----------|-------|
| Dataset | 10M samples |
| Trainable Parameters | Full Model: 8.3B |
| Batch Size | 256 |
| Learning Rate | 1e-5 (1ep) + 5e-6 (2ep) |
| Epochs | 3 |

**目标**：在单图像任务上fine-tune整个模型。

**Stage 3: Multi-Image & Video Training**

| Parameter | Value |
|-----------|-------|
| Dataset | 2M samples |
| Trainable Parameters | Full Model: 8.3B |
| Batch Size | 256 |
| Learning Rate | 5e-6 |
| Epochs | 1 |

**目标**：扩展到多图像和视频理解任务。

## 三、Dream-VL的实验结果分析

### 3.1 多学科知识和数学推理基准

**Table 2的核心数据解读**：

| Model | MMMU | MMMU Pro | MMStar | MathVista | MathVerse |
|-------|------|----------|--------|-----------|-----------|
| GPT-4o | 69.1 | 49.7 | 64.7 | 63.8 | 50.2 |
| Qwen2.5-VL | 58.6 | - | 63.9 | 68.2 | 49.2 |
| MAmmoTH-VL | 50.8 | 25.3 | 63.0 | 67.6 | 34.2 |
| **Dream-VL** | **52.2** | **26.0** | **59.9** | **63.1** | **31.5** |

**关键观察**：

1. Dream-VL在**相同训练数据量**下，相比其他open-data模型有显著优势：
   - 相比MAmmoTH-VL (同样使用open data)，MMMU Pro从25.3%提升到26.0%
   - 相比LLaDA-V (同为diffusion VLM)，MMMU从48.6%提升到52.2%

2. 与top-tier AR models (如Qwen2.5-VL)相比仍有差距，但考虑到：
   - Qwen2.5-VL使用的是**proprietary data**
   - Dream-VL使用的是**fully open data**

### 3.2 视觉规划任务的核心优势

论文特别强调了Dream-VL在**visual planning**任务上的优势，这是最有意思的部分。

**ViPlan Benchmark**：

ViPlan评估两种模式：
1. **Grounding mode**: 判断状态描述是否满足
2. **Planning mode**: 生成符号动作序列

**Figure 4的结果解读**：

论文中的Figure 4显示了在ViPlan benchmark上的性能比较：

- **Diffusion vs AR**: Dream-VL在planning tasks上相比AR模型有明显优势
- **Household domain**: 特别是在复杂的家庭环境场景中，Dream-VL的优势更明显
- **Long-horizon planning**: 随着任务长度增加，AR模型的error accumulation更严重

**为什么Diffusion在Planning上表现更好？**

1. **Bidirectional attention**:
   - AR模型: $p(x_t|x_{<t})$，只能看到历史
   - Diffusion模型: $p(x_0|x_t)$，可以看到全局上下文

2. **Iterative refinement**:
   - AR模型: 一次生成，无法回溯
   - Diffusion模型: 可以iteratively refine，类似于人类planning时的思考过程

3. **Parallel action generation**:
   - 对于需要生成多个actions的任务，diffusion可以同时生成所有actions

### 3.3 Low-level Action Planning (LIBERO Benchmark)

**实验设置**：

- **LIBERO-Goal**: 测试procedural learning，fixed objects但varying goals
- **LIBERO-Long**: 测试long-horizon planning，varying objects、layouts和goals

**Action space**:
每个action是7维的robot control：
$$a_t = [\Delta_{pos_x}, \Delta_{pos_y}, \Delta_{pos_z}, \Delta_{rot_x}, \Delta_{rot_y}, \Delta_{rot_z}, \text{gripper}]$$

**Discretization**:
每个维度被discretize成256个bins：
$$\text{action}_t = \text{Encode}(a_t) \in \{0,1,...,255\}^7$$

**Table 5的核心发现**：

| Model | VLM Type | LIBERO-Goal | LIBERO-Long |
|-------|----------|-------------|-------------|
| Qwen2.5-VL | AR | 68.0 | 34.0 |
| OpenVLA | AR | 79.2 | 53.7 |
| **Dream-VL** | **Diffusion** | **83.2** | **59.0** |

**惊人的结果**：

1. Dream-VL在**没有robotic pretraining**的情况下，直接在LIBERO数据上fine-tune，就超过了经过大规模pretraining的OpenVLA

2. 在LIBERO-Long上，Dream-VL (59.0%) vs Qwen2.5-VL (34.0%)，差距高达25个百分点

**Action Chunking Analysis (Figure 6)**：

论文还研究了action chunk size对性能的影响：

| Model | Optimal Chunk Size (LIBERO-Goal) | Optimal Chunk Size (LIBERO-Long) |
|-------|---------------------------------|----------------------------------|
| Qwen2.5-VL | 3 | 5 |
| Dream-VL | 9 | 10 |

**关键洞察**：

- **AR模型的error accumulation**: 当chunk size过大时，error accumulation导致性能下降
  - 对于Qwen2.5-VL，当chunk size > 5时，性能反而下降
  
- **Diffusion模型的parallel generation**: Dream-VL可以支持更大的chunk size而不降低性能
  - Dream-VL可以同时生成10个actions而不积累error

- **Speed advantage**: 
  - Dream-VL: 只需要1个diffusion step就能生成12个actions，实现**27× speedup**
  - AR模型: 需要 sequentially生成12个actions

## 四、Dream-VLA：基于大规模Robotics Pretraining的VLA模型

### 4.1 Robotcis Pretraining

**Open-X Embodiment Dataset**:

- **Size**: 970k trajectories
- **Diversity**: Spans multiple robot embodiments、tasks和scenes

**Training configuration**:

| Parameter | Value |
|-----------|-------|
| Base Model | Dream-VL (pretrained) |
| Dataset | Open-X Embodiment (970k) |
| Batch Size | 1024 |
| Learning Rate | 1e-5 |
| Action Chunk Size | 8 |
| Training Steps | 610k |

**Loss function**:

使用与Dream-7B相同的**discrete diffusion loss**：
$$\mathcal{L}_{\text{VLA}} = \mathbb{E}_{q(a_t|a_0)} \left[ -\log p_\theta(a_{t-1}|a_t, I, \text{instruction}) \right]$$

其中：
- $a_0$是clean action sequence
- $a_t$是masked action sequence
- $I$是vision observation
- $\text{instruction}$是自然语言指令

### 4.2 Fine-tuning策略

**Downstream tasks**:

1. **LIBERO**: 4个task suites (Spatial, Object, Goal, Long)
2. **SimplerEnv**: 
   - WidowX robot tasks (BridgeData V2)
   - Google robot tasks (Visual Matching, Variant Aggregation)

**Fine-tuning objectives**:

论文比较了多种fine-tuning objectives（Table 9）：

1. **L1 Regression**: 
   $$\mathcal{L}_{L1} = \|a_{\text{pred}} - a_{\text{gt}}\|_1$$

2. **Continuous Diffusion**:
   使用continuous diffusion for actions
   
3. **Discrete Diffusion**:
   使用discrete diffusion，与pretraining一致
   
4. **Flow Matching**:
   基于Flow Matching (lipman2023flow)：
   $$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t,x_0} \left\| v_\theta(x_t,t) - (x_1 - x_0) \right\|^2$$
   
   其中$x_t = (1-t)x_0 + tx_1$是linear interpolation，$v_\theta$预测velocity field。

**Table 9的关键发现**：

Dream-VLA在各种fine-tuning objectives下都**显著优于**OpenVLA-OFT：

| Fine-tuning Objective | OpenVLA-OFT (Overall) | Dream-VLA (Overall) |
|----------------------|----------------------|---------------------|
| L1 | 36.5 | 56.3 |
| Discrete | 31.3 | 44.8 |
| Cont. Diffusion | 4.2 | 57.3 |
| Disc. Diffusion | 34.4 | 51.0 |
| Flow Matching | 10.4 | 60.4 |

**Flow Matching的优越性**：

Dream-VLA使用Flow Matching达到60.4%的最佳性能，显著超过其他方法。

### 4.3 性能结果详解

**LIBERO Results (Table 6)**:

| Model | Spatial | Object | Goal | Long | Average |
|-------|---------|--------|------|------|---------|
| π₀ | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| OpenVLA-OFT | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| **Dream-VLA** | **97.6** | **98.8** | **97.2** | **95.0** | **97.2** |

**关键洞察**：

1. Dream-VLA达到**97.2% average success rate**，超过所有其他模型

2. 特别在LIBERO-Long上（最challenging的长horizon task），Dream-VLA (95.0%)超过π₀ (85.2%)
   - 这证明了diffusion在long-horizon planning上的优势

**WidowX Robot Results (Table 7)**:

| Model | Spoon on Towel | Carrot on Plate | Stack Green Block | Eggplant in Basket | Overall |
|-------|----------------|-----------------|-------------------|--------------------|---------|
| OpenVLA-OFT | 50.0 | 12.5 | 41.7 | 4.2 | 18.8 |
| π₀ | 45.8 | 29.1 | 25.0 | 0.0 | 27.1 |
| DiscreteDiffusionVLA | 70.8 | 29.2 | 58.3 | 29.2 | 37.5 |
| **Dream-VLA** | **91.7** | **79.2** | **58.3** | **41.7** | **71.4** |

**惊人的提升**：

- 相比OpenVLA-OFT: 18.8% → 71.4% (提升52.6个百分点)
- 相比π₀: 27.1% → 71.4% (提升44.3个百分点)

**Per-task highlights**:

- **Put Eggplant in Basket**: Dream-VLA达到100% success
- **Put Spoon on Towel**: Dream-VLA达到79.2% success

**Google Robot Results (Table 8)**:

| Model | Visual Matching | Variant Aggregation | Overall |
|-------|-----------------|---------------------|---------|
| OpenVLA-OFT | 63.0 | 45.5 | 54.3 |
| π₀ | 58.8 | 54.8 | 56.8 |
| **Dream-VLA** | **66.5** | **54.6** | **60.5** |

Dream-VLA在更challenging的**Variant Aggregation**任务上也表现优异。

### 4.4 为什么Dream-VLA表现如此出色？

论文在Section 4.4中总结了三个关键优势：

**1. Architectural Consistency**:

- **AR-based VLA (OpenVLA)**: 下游fine-tuning需要修改attention mask来支持action chunking
  - 修改attention mask可能导致性能损失
  
- **Diffusion-based VLA (Dream-VLA)**: 天然支持parallel generation，无需架构修改
  - 从LLM到VLA，架构完全一致
  - 避免了因架构变化导致的性能损失

**2. Accelerated Convergence (Figure 8)**:

论文观察到Dream-VLA在fine-tuning时**收敛更快**：

- **Loss curve analysis**:
  - Dream-VLA在较少的steps下达到lower loss
  - 特别在discrete diffusion fine-tuning时优势最明显（与pretraining objective一致）

**3. Objective Consistency**:

从pretraining到fine-tuning使用一致的diffusion objective：
- LLM pretraining: Discrete diffusion
- VLM pretraining: Discrete diffusion  
- VLA pretraining: Discrete diffusion
- VLA fine-tuning: 可选多种objectives (Flow Matching表现最好)

这种一致性使得模型能够**更好地transfer knowledge**。

## 五、技术深度剖析

### 5.1 Discrete Diffusion for Actions

**为什么要使用Discrete Diffusion而不是Continuous？**

论文的讨论（Section 5）提供了关键insight：

1. **与文本的一致性**: Text是discrete的，使用discrete diffusion可以与text planning保持一致

2. **Token-level control**: Discrete representation提供了更精细的control

3. **可扩展性**: Discrete representation更容易扩展到更复杂的action space

**但是，实验发现Continuous actions在下游fine-tuning时往往表现更好**：

- **Table 9**: Dream-VLA with Continuous Diffusion达到57.3%，显著优于Discrete (44.8%)
- **Flow Matching (continuous)**: 达到60.4%，表现最佳

**这个矛盾如何解释？**

论文的解释是：
- **Discrete representation**: 适合pretraining和general planning
- **Continuous representation**: 适合fine-grained control
- **Future direction**: 可以在continuous space上进行robotic pretraining，或者设计更好的discrete action representation (如FAST)

### 5.2 Diffusion Timestep Analysis

**在Low-level planning时**:

论文发现只需要**1个diffusion step**就能达到good performance：

```
For low-level actions:
  Input: [I, instruction, masked_actions]
  Output: [unmasked_actions] (in 1 diffusion step)
```

**为什么1个step就够？**

1. **Action space的结构性**: Robot actions具有内在的结构和约束
2. **Strong priors**: 大规模pretraining提供了很强的priors
3. **Deterministic decoding**: 对于control tasks，不需要像text generation那样的creativity

**Text generation vs Action generation**:

- **Text generation**: 通常需要多个diffusion steps来确保质量和coherence
- **Action generation**: 1个step足够，因为actions更加deterministic

### 5.3 Bidirectional Attention vs Autoregressive

**Mathematical comparison**:

**Autoregressive**:
$$p(x_{1:T}) = \prod_{t=1}^{T} p(x_t|x_{<t})$$

每个token $x_t$的生成只依赖于之前的tokens $x_{<t}$。

**Diffusion**:
$$p(x_{1:T}) = \int p(x_T) \prod_{t=T}^{1} p(x_{t-1}|x_t) dx_T$$

在去噪过程中，模型可以看到全局上下文。

**在Planning中的影响**:

**Example**: 规划一个sequence of actions [a₁, a₂, a₃, a₄] to achieve a goal

- **AR model**:
  - Generate a₁ based on [observation, goal]
  - Generate a₂ based on [observation, goal, a₁]
  - Generate a₃ based on [observation, goal, a₁, a₂]
  - Generate a₄ based on [observation, goal, a₁, a₂, a₃]
  
  如果a₁或a₂是错误的，后续的a₃、a₄也会受到影响

- **Diffusion model**:
  - Start with masked actions [MASK, MASK, MASK, MASK]
  - In each denoising step, predict all masked positions simultaneously
  - Use global context (observation, goal, and other actions) to refine
  
  可以同时refine所有actions，确保global coherence

## 六、与相关工作的比较

### 6.1 与其他Diffusion VLMs的比较

| Model | Base LLM | Training Data | MMMU | LIBERO-Long |
|-------|----------|---------------|------|-------------|
| LLaDA-V | LLaDA 8B | ~13M | 48.6 | - |
| Dimple | Dream 7B | ~2M | 45.2 | - |
| LaViDa-D | Dream 7B | ~2M | 42.6 | - |
| **Dream-VL** | **Dream 7B** | **~12M** | **52.2** | **59.0** |

**关键差异**：

1. **Training data scale**: Dream-VL使用12M数据，显著大于其他diffusion VLMs
2. **Base model strength**: Dream 7B是strong的diffusion LLM
3. **Training strategy**: 三阶段训练更系统

### 6.2 与其他VLA模型的比较

**Architecture comparison (Figure 6)**:

| Model | Backbone | Action Expert | Action Chunking Support |
|-------|----------|---------------|-------------------------|
| OpenVLA | AR LLM | No | Requires architecture modification |
| π₀ | AR LLM | Yes (separate) | Requires architecture modification |
| **Dream-VLA** | **Diffusion LLM** | **No** | **Native support** |

**关键差异**：

1. **No action expert needed**: Dream-VLA不需要separate action expert模块
2. **Native action chunking**: Diffusion天然支持parallel action generation
3. **Unified architecture**: 从LLM到VLA，架构保持一致

## 七、未来方向与Limitations

### 7.1 当前Limitations

论文在Section 5中诚实地讨论了几个关键limitations：

**1. Data scale**:

- **VL training data**: 12M samples，相比 proprietary models (使用百亿级)还有很大差距
- **VLA pretraining data**: 970k trajectories，相比其他方法还不够大

**2. General capabilities gap**:

- 在standard VL benchmarks上，Dream-VL仍落后于top closed-data AR models (如Qwen2.5-VL)
- 说明diffusion在general multimodal understanding上还有improvement space

**3. Real-world experiments**:

- Current real-robot experiments仍然是preliminary的
- 需要更大、更多样的real-world datasets来系统评估

### 7.2 有趣的未来方向

**1. Joint high-level and low-level training**:

论文建议探索类似RT-2的联合训练：
- **High-level planning**: Symbolic action generation
- **Low-level control**: Continuous action prediction

由于dLLMs在high-level planning上已经显示出优势，这种联合训练可能会产生synergy。

**2. Continuous-space robotic pretraining**:

- **Option 1**: 在continuous dVLMs上进行robotic pretraining
- **Option 2**: 设计更好的discrete action representation (如FAST)

**3. Real-world generalization**:

- 收集更大的real-world datasets
- 系统评估Dream-VLA在realistic environments中的performance
- 与AR-based VLAs进行比较，评估generalization能力

## 八、总结与Intuition Building

### 8.1 核心贡献总结

这篇论文的核心贡献可以总结为：

1. **证明了diffusion-based VL/VLA的可行性**: Dream-VL和Dream-VLA在多个benchmarks上达到state-of-the-art performance

2. **揭示了diffusion在planning上的优势**: 在visual planning和robotic control任务上，diffusion模型表现出明显优势

3. **提供了统一的架构**: 从LLM到VLA，使用统一的diffusion架构，避免了architectural modifications

4. **开源了strong models**: Dream-VL和Dream-VLA都开源，促进了community的研究

### 8.2 为什么Diffusion在Robotics上表现更好？

**Intuitive explanation**:

**Human-like planning process**:

当我们人类做复杂的任务规划时（比如整理厨房）：
1. 我们会**同时考虑多个steps**，而不是一个接一个地决定每个action
2. 我们会**iteratively refine**我们的计划，不断调整
3. 我们会**考虑全局目标**，而不仅仅是局部最优

**Autoregressive models**:
- 像是"one-step-at-a-time"的决策者
- 容易陷入局部最优，难以进行global reasoning

**Diffusion models**:
- 更像是human的planning过程
- 可以同时考虑multiple steps
- Iteratively refine，确保global coherence

**Mathematical intuition**:

对于planning问题，我们需要找到：
$$a^* = \arg\max_{a_{1:T}} \sum_{t=1}^{T} r(s_t, a_t)$$

其中$r(s_t, a_t)$是reward function。

- **AR model**: 使用sequential决策$p(a_t|s_t, a_{<t})$，容易greedy地选择每个$a_t$
- **Diffusion model**: 可以直接优化整个sequence$p(a_{1:T}|s_{1:T}, \text{goal})$，更容易找到global optimum

### 8.3 关键Takeaways

**For researchers**:

1. **Diffusion LLMs是promising的方向**: 特别是在需要planning和reasoning的任务上
2. **Architectural consistency很重要**: 从pretraining到fine-tuning，保持架构一致可以提升performance
3. **Training data scale仍然关键**: 即使是strong的diffusion models，也需要sufficient training data

**For practitioners**:

1. **考虑使用diffusion VLA for robotics tasks**: 特别是需要long-horizon planning的任务
2. **Action chunking更efficient**: 使用diffusion可以支持更大的action chunk size
3. **Fine-tuning objective的选择很重要**: Flow Matching在下游fine-tuning时表现最好

**For intuition**:

这篇论文的核心intuition是：**diffusion models更适合planning tasks，因为它们可以同时考虑multiple actions，并且可以iteratively refine，这更接近人类做planning的方式**。

## References

论文中提到的关键相关工作：

1. **Diffusion Language Models**:
   - Dream: https://arxiv.org/abs/2310.04478
   - LLaDA: https://arxiv.org/abs/2309.08572
   - Gemini Diffusion: https://arxiv.org/abs/2406.07904

2. **Vision-Language Models**:
   - LLaVA: https://arxiv.org/abs/2304.08485
   - Qwen2-VL: https://arxiv.org/abs/2409.12191
   - MAmmoTH-VL: https://arxiv.org/abs/2406.17095

3. **Vision-Language-Action Models**:
   - RT-2: https://arxiv.org/abs/2307.15818
   - OpenVLA: https://arxiv.org/abs/2406.09175
   - π₀: https://arxiv.org/abs/2409.01491

4. **Benchmarks**:
   - LIBERO: https://arxiv.org/abs/2306.09778
   - ViPlan: https://arxiv.org/abs/2409.05701
   - SimplerEnv: https://arxiv.org/abs/2404.12974

这篇论文为diffusion-based multimodal models的future研究提供了strong baseline和valuable insights。