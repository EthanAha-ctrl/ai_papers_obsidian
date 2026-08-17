---
source_pdf: MEMORYVLA.pdf
paper_sha256: 07161ae7543b8ebabdac6dfbc2af58e8e3a4669723882d7ff924490892cc1117
processed_at: '2026-08-05T17:37:32-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# MemoryVLA 用人话说

## 这篇paper到底在解决什么问题？

想象你让机器人"按顺序按下红、绿、蓝三个按钮"。机器人按完红色按钮后，看了一眼桌面——**红色按钮看起来和没按之前一模一样**。这时候机器人就懵了：我到底按过没有？下一个该按什么？

这就是当前所有主流VLA模型（OpenVLA、π₀、CogACT）的通病。它们就像**只有1秒记忆的金鱼**，每次决策只看当前这一帧画面。但robotic manipulation本质上是**non-Markovian**的——你之前做了什么直接影响接下来该做什么。

作者举了几个特别直观的例子：
- **Change Food**：桌上有两个食物，当前帧根本看不出哪个是刚放上去的、哪个是要换掉的
- **ShellGameTouch**：cube被cup盖住后，当前帧完全丢失了cube位置信息
- **Clean Table & Count**：你得记得自己已经清理了几个、按了几次计数按钮

**naive的做法行不通**：
- 把连续帧拼起来喂给VLM？self-attention的 $O(n^2)$ 复杂度爆炸
- 而且VLM预训练时都是单帧的，你喂多帧它根本不适应

所以作者想：**人类是怎么解决这个问题的？**

参考：https://shihao1895.github.io/MemoryVLA

## 从认知科学偷思路

人脑有两套记忆系统（参考Baddeley & Hitch 1974, Tulving 1972）：

**Working Memory（工作记忆）**：在visual cortex和prefrontal cortex里靠神经元的transient activity短时缓存当前感知和认知，寿命很短，用于即时决策。你可以理解为"脑子里的RAM"。

**Episodic Memory（情景记忆）**：在hippocampus里长期存储过去经验，有两种形式：
- **verbatim**：保留精确细节（低层感知）
- **gist**：捕捉抽象语义（高层认知）

执行任务时，working memory从episodic memory里**检索**相关历史，和当前表征**融合**，再通过cerebellum控制动作，同时把新经验**固化**进episodic memory。

作者就把这套机制原封不动搬到了VLA里：Cognition → Memory → Action三段式。

## 架构拆解：从像素到动作

整个pipeline大概长这样：

```
RGB image + Language instruction
    ↓
[VLM Cognition Module] —— 7B Prismatic VLM
    ├── DINOv2 + SigLIP → raw visual tokens
    ├── SE-bottleneck压缩 → 256个perceptual tokens p
    └── LLaMA-7B的EOS hidden state → 1个cognitive token c
    ↓
{p, c} = Working Memory
    ↓
[PCMB Memory Module]
    ├── Retrieval: cross-attention + timestep PE → H^p, H^c
    ├── Gate Fusion: gate控制 → p̃, c̃
    └── Consolidation: 相邻entry合并 → 更新PCMB
    ↓
{p̃, c̃} = Memory-augmented tokens
    ↓
[Diffusion Action Expert] —— DiT + DDIM
    ├── Cognition-attention with c̃
    ├── Perception-attention with p̃
    └── 10步denoising → 16步7-DoF action chunk
```

### 4.1 Cognition Module：怎么看图？

用Prismatic VLM（7B），在Open-X Embodiment上预训练过。

视觉编码用**双backbone并行**：
- **DINOv2**（self-supervised）：擅长几何/位姿/空间结构
- **SigLIP**（language-aligned）：擅长语义/物体识别

两个特征concat后形成raw visual tokens。

然后分两路：

**Perceptual path**：通过SE-bottleneck压缩。SE-bottleneck就是Squeeze-and-Excitation（Hu et al. 2018），先global average pooling压缩spatial维度，再通过FC+sigmoid学习channel-wise weight重新加权。最终压缩到 $N_p = 256$ 个perceptual tokens：
$$p \in \mathbb{R}^{256 \times d_p}, \quad d_p = 256$$

**Cognitive path**：raw visual tokens通过linear projection映射到language embedding space，和tokenized instruction拼一起喂给LLaMA-7B。取**EOS position**的hidden state作为cognitive token：
$$c \in \mathbb{R}^{1 \times 4096}$$

注意这里**只用1个token**。Tab.11做了ablation：1个vs 4个cognitive token几乎没差别（71.9% vs 69.8%）。一个4096维的EOS token已经够编码"我在执行什么任务、处于哪一步"这种高层语义了。

最终working memory就是 $\{p, c\}$，对应大脑里的visual cortex和prefrontal cortex活动。

### 4.2 PCMB Memory Module：怎么记？怎么取？怎么合并？

这是这篇paper的核心创新。PCMB维护两条独立stream：
$$M_{pcmb} = \{m^{per}, m^{cog}\}$$

每条stream最多存 $L$ 个entry（默认 $L=16$，real-world temporal tasks用 $L=256$）。

**为什么要分两条stream？** 因为perceptual token是256维 × 256个，cognitive token是4096维 × 1个，维度差太多。如果混在一起做attention，高维的会淹没低维的，或者反之。而且它们检索语义不同——cognitive回答"我在哪一步"，perceptual回答"物体上次在哪"。

#### Retrieval：怎么取历史？

每次决策时，working memory $\{p, c\}$ 作为query去检索PCMB。用cross-attention，关键是要加**timestep positional encoding**：

$$K^x = [m_1^x + \text{TE}(t_1); \ldots; m_L^x + \text{TE}(t_L)]$$
$$V^x = [m_1^x; \ldots; m_L^x]$$
$$\hat{H}^x = \text{softmax}\left(\frac{q^x (K^x)^\top}{\sqrt{d_x}}\right) V^x$$

这里 $\text{TE}(\cdot)$ 是sinusoidal embedding， $t_i$ 是第 $i$ 个entry在episode中的时间步。 $q^x$ 是query（ $p$ 或 $c$ ）， $K^x, V^x$ 是历史entry构建的key/value。 $\sqrt{d_x}$ 是scaling factor防止点积过大。

经过2层Transformer（attention + FFN）得到最终的 $H^p$ 和 $H^c$。

**Timestep PE为什么关键？** Tab.7 ablation显示加PE从69.8% → 71.9%。没PE时模型只能靠content similarity检索——但两帧内容相似不等于语义相同。比如ShellGameTouch任务中，cube刚揭示那帧和很久后某个相似视角的帧，content可能像但含义完全不同。Timestep PE让模型知道"这是5步前发生的"vs"这是50步前发生的"，赋予不同权重。

#### Gate Fusion：怎么融合历史和当前？

这个设计很关键。不是简单相加，而是用learned gate：

$$g^x = \sigma\big(\text{MLP}(\text{concat}[x, H^x])\big)$$
$$\tilde{x} = g^x \odot H^x + (1 - g^x) \odot x$$

这里 $g^x \in (0,1)^{d_x}$ 是per-channel gate vector， $\sigma$ 是sigmoid， $\odot$ 是element-wise乘。 $x$ 是当前working memory token， $H^x$ 是retrieved historical embedding。

**Intuition**：考虑Change Food任务，当前帧已经能看清两个食物的精确位置，但不知道哪个是"先放的要换掉的"。
- 对perceptual stream：当前帧已够精细，gate趋近0，用当前 $p$
- 对cognitive stream：需要历史语义"我在换食物步骤"，gate趋近1，用retrieved $H^c$

Gate允许**per-stream、per-channel**自适应决定"信任历史还是信任当前"。Add是固定权重混合，做不到这种灵活性。

Tab.7 ablation：Gate 71.9% vs Add 67.7%。

#### Consolidation：容量满了怎么办？

当entry数超过 $L$ 时，对相邻entry算cosine similarity，合并最相似的一对：

$$i_x^* = \arg\max_{i=1,\ldots,L-1} \cos(\tilde{x}_i, \tilde{x}_{i+1})$$
$$m_{i_x^*}^x \gets \frac{1}{2}(\tilde{x}_{i_x^*} + \tilde{x}_{i_x^*+1})$$

这里 $i_x^*$ 是最相似相邻pair的index，合并方式是simple averaging。

**为什么Token Merge而不是FIFO？** Tab.7：Token Merge 71.9% vs FIFO 66.7%。

FIFO假设"越老越没用"，但non-Markovian任务中恰恰相反——ShellGameTouch中最初揭示cube位置的那帧，跨越整个episode都关键，FIFO会把它丢掉。Token Merge假设"相邻相似则冗余"，这和机械臂缓慢移动导致相邻帧视觉变化平缓的物理先验一致。合并冗余同时自然保留独特事件——类似hippocampus在sleep期间做的memory consolidation。

### 4.3 Diffusion Action Expert：怎么生成动作？

用DiT（Diffusion Transformer）+ DDIM（10步sampling）。

**为什么用diffusion而不是autoregressive？** robotic action是continuous multimodal space——同一观察下可能有多种合理动作。Autoregressive要tokenize，量化损失精度。Diffusion通过iterative denoising自然支持multimodal distribution。

每个denoising step：
1. Noisy action tokens注入denoising timestep的sinusoidal encoding
2. 和cognitive representation $\tilde{c}$ 拼接
3. **Cognition-attention**： $\tilde{c}$ 作condition提供高层语义guidance
4. **Perception-attention**： $\tilde{p}$ 作condition补充精细视觉细节
5. FFN refinement → 该step的denoised action

训练用MSE loss。最终denoised vectors通过MLP输出7-DoF actions。Inference时用classifier-free guidance（CFG scale=1.5）平衡condition影响力和生成多样性。

预测 **T=16步action chunk**，一次预测多步可以减少累积误差、提供foresight。

## 实验结果：到底多有效？

### 5.1 SimplerEnv-Bridge（Tab.1）

| Method | Spoon on Towel | Carrot on Plate | Stack Cube | Eggplant in Basket | Avg |
|---|---|---|---|---|---|
| OpenVLA | 4.2 | 0.0 | 0.0 | 12.5 | 4.2 |
| TraceVLA | 12.5 | 16.6 | 16.6 | 65.0 | 27.7 |
| CogACT-Large | 58.3 | 45.8 | 29.2 | 95.8 | 57.3 |
| π₀-Beta* | 84.6 | 55.8 | 47.9 | 85.4 | 68.4 |
| **MemoryVLA** | **75.0** | **75.0** | **37.5** | **100.0** | **71.9** |

比CogACT-Large高**+14.6个点**。Carrot on Plate从45.8跃升到75.0——这个任务需要把carrot放到plate上，位置感知很重要，perceptual memory帮了大忙。

### 5.2 SimplerEnv-Fractal（Tab.2）

分Visual Matching (VM) 和 Visual Aggregation (VA) 两套设置。VA故意扰动background/lighting/distractors/textures来stress-test鲁棒性。

| Method | VM Avg | VA Avg | Overall |
|---|---|---|---|
| CogACT | 74.8 | 61.3 | 68.1 |
| **MemoryVLA** | **77.7** | **67.7** | **72.7** |

VA下提升更显著（+6.4），说明memory不仅解决时序问题，还增强**robustness**——历史context帮模型抵抗单帧扰动。

**Open/Close Drawer在VA下**：MemoryVLA 53.2% vs CogACT 28.3%，**+24.9**。Drawer是hinge-like object，开/关状态单帧难判断，memory提供了关键state-tracking能力。

### 5.3 LIBERO（Tab.3）

| Method | Spatial | Object | Goal | Long-10 | Long-90 | Avg |
|---|---|---|---|---|---|---|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 73.5 | 75.9 |
| CogACT | 97.2 | 98.0 | 90.2 | 88.8 | 92.1 | 93.2 |
| **MemoryVLA** | **98.4** | **98.4** | **96.4** | **93.4** | **95.6** | **96.5** |

**Long-10从88.8 → 93.4**，长horizon任务受益最大。注意π₀-FAST和π₀用了wrist-camera + proprioceptive state，MemoryVLA仅用third-person RGB依然超越。

### 5.4 Mikasa-Robo（Tab.4）

这是专门设计的**memory-dependent benchmark**。

| Method | ShellGame | Intercept | Remb.Color3 | Remb.Color5 | Remb.Color9 | Avg |
|---|---|---|---|---|---|---|
| OpenVLA-OFT | 47 | 14 | 59 | 16 | 6 | 28.4 |
| PI-0 | 33 | 42 | 35 | 22 | 15 | 29.4 |
| **MemoryVLA** | **88** | 24 | 44 | 30 | 20 | **41.2** |

**ShellGameTouch上88% vs 47%，+41个点**。这个任务cube被cup盖住后必须记住最初位置——正是timestep PE + retrieval机制发挥作用的场景。

### 5.5 Real-World（Tab.5）

**General Tasks**：MemoryVLA 85% vs CogACT 76% vs π₀ 72%

**Long-horizon Temporal Tasks**：MemoryVLA 83% vs CogACT 57% vs OpenVLA 9%

Temporal tasks具体对比：
- Seq. Push Buttons：58% vs 15%（**+43**）
- Change Food：85% vs 47%（**+38**）
- Guess Where：72% vs 40%（**+32**）
- Clean Table & Count：84% vs 67%（**+17**）

这些任务都是教科书级non-Markovian例子，提升幅度直接印证memory mechanism价值。

## Ablation：哪个设计最重要？

### 6.1 Memory Type（Tab.6）

| Variant | Avg |
|---|---|
| Cognitive only | 63.5% |
| Perceptual only | 64.6% |
| Both | **71.9%** |

两者结合产生**+7.3 ~ +8.4协同效应**，超出单独贡献之和，说明它们编码的信息是正交的——cognitive管"我在哪一步"，perceptual管"物体上次在哪"。

### 6.2 Memory Length（Tab.6, Tab.10）

- Bridge：4 → 67.7%，16 → 71.9%，64 → 67.7%
- Clean Table & Count：64 → 78%，256 → 84%，512 → 81%
- LIBERO-90：8 → 94.2%，16 → 95.6%，32 → 95.6%

**最优length和episode length正相关**。过短截断关键历史，过长稀释attention signal并增加噪声。Tab.9显示real-world temporal任务平均1672步，所以用256才够。

### 6.3 Robustness（Appendix B）

Pick Place Order在unseen background 100%、unseen distractors 92%、unseen lighting 96%、occlusion 96%下保持高成功率。但**unseen camera view大幅下降**（Pick Coke Can 42.0%）——视角变化改变了perceptual feature distribution，memory里存的是旧视角特征，和新帧不匹配。

## 效率：是不是很贵？

| Model | Latency (4090) | Throughput | Memory |
|---|---|---|---|
| Baseline | 0.187s | 85.6 Hz | 15.8 GB |
| MemoryVLA | 0.194s | 82.5 Hz | 16.6 GB |

**仅+3.6% latency，+0.8 GB memory**。几乎是免费午餐。原因：
- Cognitive memory每条仅1个token
- Perceptual memory压缩到256 channel
- Token Merge持续合并冗余，限制retrieval的key/value数量

## 一句话总结

**MemoryVLA本质上就是robotic版的RAG**——从同一episode的历史中retrieve决策相关信息，用gate融合，用token merge压缩。但它借鉴认知科学把memory分成perceptual/cognitive双流，对应人脑的verbatim/gist双形式episodic memory，配合timestep PE让模型有时序感知能力。整个module轻量到可以即插即用到现有VLA上，在non-Markovian temporal tasks上获得巨大提升（real-world +26个点）。

参考链接：
- Project Page: https://shihao1895.github.io/MemoryVLA  
- Code: https://github.com/shihao1895/MemoryVLA
- Models: https://huggingface.co/collections/shihao1895/memoryvla

---

如果你想深入某部分，比如diffusion action expert的具体denoising过程、PCMB在training-time vs inference-time的行为差异、或者Fig.10里memory retrieval的可视化案例，告诉我我可以继续展开。

---

# MemoryVLA详解：基于认知科学的 VLA 时序记忆框架

## 1. 核心动机：为什么 VLA 需要记忆？

当前主流的 VLA 模型（OpenVLA、π₀、CogACT）都依赖 **single-frame observation** 进行决策，这在本质上违反了 robotic manipulation 的 **non-Markovian** 性质。

考虑 Fig.1(a) 中的 Push Buttons 任务：按下按钮前后的视觉状态几乎完全相同（像素层面几乎无差异），这意味着 **仅凭当前帧无法判断这个按钮是否已经按过**。类似地：
- **Change Food**：桌上同时存在两个食物，单帧无法判断该拿哪一个
- **Shell Game Touch**：cube 被 cup 盖住后，当前帧无法知道 cube 在哪个 cup 下面
- **Clean Table & Count**：必须记住已经清理了几个物体、按了几次计数按钮

这些都是 **temporal ambiguity** 问题——当前观察 $I_t$ 不足以推断出正确的 action $a_t$，必须借助历史信息 $I_{1:t-1}$。

**naive 方案的困境**：
- 直接拼接连续帧作为 VLM 输入 → self-attention 的 $O(n^2)$ 复杂度限制了 context length
- 与 VLM 的 single-frame pretraining distribution 不匹配，导致分布漂移

参考链接：
- 论文：https://shihao1895.github.io/MemoryVLA
- Code：https://github.com/shihao1895/MemoryVLA
- Models：https://huggingface.co/collections/shihao1895/memoryvla

## 2. 认知科学启发：Dual-Memory System

Paper 借鉴了 Baddeley & Hitch (1974) 的 working memory 理论 和 Tulving (1972) 的 episodic memory 理论：

**人类大脑的两套记忆系统**：
1. **Working Memory**（工作记忆）：由 visual cortex 和 prefrontal cortex 的 transient neural activity 实现，短时缓存当前感知与认知表征，用于即时决策
2. **Episodic Memory**（情景记忆）：由 hippocampus 支持，长期存储过去经验，包含两种形式：
   - **verbatim representations**：保留精确细节（对应低层感知）
   - **gist representations**：捕捉抽象语义（对应高层认知）

执行任务时，working memory 从 episodic memory 中 **retrieve** decision-relevant contexts，与当前表征 **fuse**，再通过 cerebellar control 产生动作，同时将新经验 **consolidate** 进 episodic memory。

MemoryVLA 将这套机制映射到 VLA 架构：
- **Cognition**：VLM 编码 observation → perceptual + cognitive tokens（working memory）
- **Memory**：Perceptual-Cognitive Memory Bank（PCMB）模拟 hippocampus
- **Action**：Memory-conditioned diffusion action expert 模拟 cerebellar control

## 3. 整体架构：Cognition-Memory-Action

### 3.1 Problem Formulation

给定当前 RGB 图像 $I \in \mathbb{R}^{H \times W \times 3}$ 和语言指令 $L$，policy $\pi$ 输出未来动作序列：

$$\mathcal{A} = (a_1, \ldots, a_T) = \pi(I, L)$$

其中每个动作 $a_t = [\Delta x, \Delta y, \Delta z, \Delta\theta_x, \Delta\theta_y, \Delta\theta_z, g]^\top$ 包含 6-DoF 的相对位姿增量 + 1 维 binary gripper state $g \in \{0, 1\}$。**T=16** 表示预测未来 16 步的 action chunk。

### 3.2 Vision-Language Cognition Module

基于 **Prismatic VLM (7B)**，在 Open-X Embodiment 上预训练。

**双分支视觉编码**：
- **DINOv2**：捕捉 self-supervised spatial features（适合精细几何/位姿）
- **SigLIP**：捕捉 language-aligned semantic features
- 两者特征 concat 形成 raw visual tokens

**Perceptual Compression**：
通过 SE-bottleneck（Squeeze-and-Excitation，参考 Hu et al. 2018 的 SE-Net）压缩到 $N_p = 256$ 个 perceptual tokens：
$$p \in \mathbb{R}^{N_p \times d_p}, \quad N_p = 256$$

这里 $d_p$ 是 perceptual token 的 channel dimension（从 Tab.8 看为 256）。

**Cognitive Token 提取**：
raw visual tokens 通过 linear projection 映射到 language embedding space，与 tokenized instruction concat 后输入 LLaMA-7B。取 **EOS position** 的 hidden state 作为 cognitive token：
$$c \in \mathbb{R}^{1 \times d_c}$$

这里 $d_c = 4096$（LLaMA-7B 的 hidden size）。**只用一个 token** 足够——Tab.11 ablation 显示 1 个 vs 4 个 cognitive token 几乎无差别（71.9% vs 69.8%），说明单个 4096-dim EOS token 已编码充足语义。

最终形成 working memory：
$$M_{wk} = \{p \in \mathbb{R}^{N_p \times d_p}, c \in \mathbb{R}^{1 \times d_c}\}$$

### 3.3 Perceptual-Cognitive Memory Bank (PCMB)

PCMB 维护两条独立流：
$$M_{pcmb} = \{m^x \mid x \in \{\text{per}, \text{cog}\}\}$$
$$m^x = \{m_i^x \in \mathbb{R}^{N_x \times d_x}\}_{i=1}^{L}$$

其中 $L$ 是 memory capacity（默认 16，real-world temporal tasks 用 256）。$m_i^{per}$ 存储精细视觉细节，$m_i^{cog}$ 存储高层语义摘要。**分离双流的设计**避免了高维 perceptual 信息淹没 low-dim cognitive 语义，反之亦然。

#### 3.3.1 Memory Retrieval

如 Fig.3(a)，working memory 作为 dual query 检索 PCMB。

**Timestep Positional Encoding**：每个 memory entry 关联其 episode timestep，通过 sinusoidal embedding $\text{TE}(\cdot)$ 加入位置编码：

$$K^x = [m_1^x + \text{TE}(t_1); \ldots; m_L^x + \text{TE}(t_L)]$$
$$V^x = [m_1^x; \ldots; m_L^x]$$

这里 $K^x, V^x$ 是 cross-attention 的 key/value，$t_i$ 是第 $i$ 个 entry 的 episode timestep。**Timestep PE 至关重要**——Tab.7 ablation 显示加入后从 69.8% → 71.9%。它让模型能区分"刚发生"与"很久前发生"的事件，对 ShellGameTouch 这种"最初揭示位置"的任务关键。

Scaled dot-product attention：
$$\hat{H}^x = \text{softmax}\left(\frac{q^x (K^x)^\top}{\sqrt{d_x}}\right) V^x, \quad q^x \in \{p, c\}$$

这里 $q^x$ 是 query（perceptual tokens $p$ 或 cognitive token $c$），$\sqrt{d_x}$ 是 scaling factor 防止点积过大。perceptual stream 产生 $\hat{H}^p \in \mathbb{R}^{N_p \times d_p}$，cognitive stream 产生 $\hat{H}^c \in \mathbb{R}^{1 \times d_c}$。

经过 1 个 FFN + 2 层 Transformer，得到最终 retrieved embeddings $H^p$ 和 $H^c$。

**Intuition**：perceptual stream 让模型找回"上次看到 cube 在哪"的精细视觉，cognitive stream 找回"我在执行哪一步骤"的高层语义。两者独立检索避免互相干扰。

#### 3.3.2 Memory Gate Fusion

如 Fig.3(b)，用 gated mechanism 自适应融合 retrieved memory 与 current tokens：

$$g^x = \sigma\big(\text{MLP}(\text{concat}[x, H^x])\big)$$
$$\tilde{x} = g^x \odot H^x + (1 - g^x) \odot x$$

这里：
- $g^x \in (0, 1)^{d_x}$ 是 learned gate vector
- $\sigma$ 是 sigmoid
- $\odot$ 是 element-wise multiplication
- $x$ 是当前 working memory token（$p$ 或 $c$）
- $H^x$ 是 retrieved historical embedding

**Gate vs Add 的差别**（Tab.7）：Gate 71.9% vs Add 67.7%。Gate 允许模型 **per-channel** 决定"信任历史还是信任当前"——当当前观察信息充分时 gate 趋近 0，当历史信息关键时 gate 趋近 1。Add 是固定权重融合，缺乏适应性。

#### 3.3.3 Memory Consolidation

如 Fig.3(c)，PCMB 容量满时，对相邻 entry 计算 cosine similarity，合并最相似的一对：

$$i_x^* = \arg\max_{i=1,\ldots,L-1} \cos(\tilde{x}_i, \tilde{x}_{i+1})$$
$$m_{i_x^*}^x \gets \frac{1}{2}(\tilde{x}_{i_x^*} + \tilde{x}_{i_x^*+1})$$

这里 $i_x^*$ 是最相似相邻对的 index，合并方式是 simple averaging。

**为什么是 Token Merge 而非 FIFO**（Tab.7）：Token Merge 71.9% vs FIFO 66.7%。FIFO 简单丢弃最老 entry，但最老的 entry 可能正是关键信息（如 ShellGameTouch 中最初揭示 cube 位置的帧）。Token Merge 合并冗余的相邻 entry，既保持容量稳定，又保留独特信息——类似 hippocampus 在 sleep 期间的 memory consolidation。

**Intuition**：相邻帧往往视觉相似（机械臂缓慢移动），合并它们损失最小；跨越关键事件边界的帧差异大，不会被合并，自然被保留下来。

### 3.4 Memory-Conditioned Diffusion Action Expert

采用 **DiT (Diffusion Transformer)** + **DDIM**（10 sampling steps）。

**为何用 diffusion 而非 autoregressive**：
- Robotic action 是 continuous multimodal space（同一观察下可能有多种合理动作）
- Autoregressive tokenization 量化损失精度
- Diffusion 通过 iterative denoising 自然支持 multimodal distribution

**Conditioning 机制**：
每个 denoising step：
1. Noisy action tokens 注入 denoising timestep 的 sinusoidal encoding
2. 与 cognitive representation $\tilde{c}$ **concat**
3. **Cognition-attention layer**：用 $\tilde{c}$ 作 condition，提供高层语义 guidance
4. **Perception-attention layer**：用 $\tilde{p}$ 作 condition，补充精细视觉细节
5. FFN refinement → 该 step 的 denoised action

**训练目标**：MSE loss between predicted and target actions。最终 denoised vectors 通过 MLP 输出 7-DoF actions。

**Classifier-Free Guidance**（Ho & Salimans 2022）：guidance scale = 1.5，平衡 condition 影响力与生成多样性。

## 4. 实验结果

### 4.1 SimplerEnv-Bridge（Tab.1）

| Method | Avg Success |
|---|---|
| OpenVLA | 4.2% |
| TraceVLA | 27.7% |
| CogACT-Large | 57.3% |
| π₀-Beta* | 68.4% |
| **MemoryVLA** | **71.9% (+14.6 vs CogACT)** |

亮点：**Eggplant in Basket** 达到 100%，**Carrot on Plate** 从 CogACT 的 45.8% 跃升到 75.0%。

### 4.2 SimplerEnv-Fractal（Tab.2）

**Visual Matching (VM)** vs **Visual Aggregation (VA)** 两套设置：
- VM：模拟贴近 real-world 的 setup
- VA：stress-test 鲁棒性（扰动 background/lighting/distractors/textures）

MemoryVLA 整体 **72.7%**，比 CogACT 提升 +4.6。在 VA 设置下提升更显著（+6.4），说明 memory 机制不仅解决时序问题，还增强了 **robustness**——历史 context 帮助模型抵抗单帧扰动。

Open/Close Drawer 任务：VA 设置下 MemoryVLA 53.2% vs CogACT 28.3%（**+24.9**）。Drawer 是 hinge-like object，开/关状态在单帧下难以判断，memory 提供了关键的 state-tracking 能力。

### 4.3 LIBERO（Tab.3）

五个 suite：Spatial / Object / Goal / Long-10 / Long-90

| Method | Spatial | Object | Goal | Long-10 | Long-90 | Avg |
|---|---|---|---|---|---|---|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 73.5 | 75.9 |
| CogACT | 97.2 | 98.0 | 90.2 | 88.8 | 92.1 | 93.2 |
| **MemoryVLA** | **98.4** | **98.4** | **96.4** | **93.4** | **95.6** | **96.5** |

注意 Long-90 从 92.1 → 95.6，**Long-10 从 88.8 → 93.4**，长 horizon 任务受益最大。

**关键对比**：π₀-FAST 和 π₀ 使用了 wrist-camera + proprioceptive state，MemoryVLA 仅用 third-person RGB，依然超越。

### 4.4 Mikasa-Robo（Tab.4）

这是专门设计的 **memory-dependent benchmark**，包含 ShellGameTouch、Intercept Medium、Remember Color3/5/9 等。

| Method | Avg |
|---|---|
| CronusVLA | 18.0% |
| OpenVLA-OFT | 28.4% |
| PI-0 | 29.4% |
| **MemoryVLA** | **41.2% (+11.8)** |

**ShellGameTouch 上 +41.0%**（88% vs 47%）。这个任务中 cube 被 cup 盖住后，模型必须记住最初位置——正是 PCMB 的 timestep PE + retrieval 机制发挥作用。

### 4.5 Real-World（Tab.5）

**General Tasks**：MemoryVLA 85% vs CogACT 76% vs π₀ 72%（+9 over CogACT）

**Long-horizon Temporal Tasks**：MemoryVLA 83% vs CogACT 57% vs OpenVLA 9%（**+26 over CogACT**）

逐任务对比（temporal）：
- Seq. Push Buttons：58% vs 15%（**+43**）
- Change Food：85% vs 47%（**+38**）
- Guess Where：72% vs 40%（**+32**）
- Clean Table & Count：84% vs 67%（**+17**）

这些任务都是教科书级的 non-Markovian 例子，MemoryVLA 的提升幅度直接印证了 memory mechanism 的价值。

## 5. Ablation Studies 深度解析

### 5.1 Memory Type（Tab.6）

| Variant | Avg Success |
|---|---|
| Cognitive only | 63.5% |
| Perceptual only | 64.6% |
| Both | **71.9%** |

**Intuition**：Cognitive memory 帮助"我在哪一步"，perceptual memory 帮助"物体上次在哪"。两者互补——单用任一都丢失关键信息维度。两者结合产生 **+7.3 ~ +8.4 的协同效应**，超出单独贡献之和，说明它们编码的信息是正交的。

### 5.2 Memory Length（Tab.6, Tab.10）

- Bridge：4 → 67.7%，16 → 71.9%，64 → 67.7%
- Clean Table & Count：64 → 78%，256 → 84%，512 → 81%
- LIBERO-90：8 → 94.2%，16 → 95.6%，32 → 95.6%

**Intuition**：最优 length 与 episode length 正相关。Bridge 平均 119 步，16 已够覆盖关键决策窗口；real-world temporal 任务可达 1672 步，需要 256 的 capacity。过短截断关键历史，过长稀释 attention signal 并增加噪声。

### 5.3 Robustness（Appendix B）

Pick Place Order 在 unseen background 100%、unseen distractors 92%、unseen lighting 96%、occlusion 96% 下保持高成功率。但 **unseen camera view 大幅下降**（Pick Coke Can 42.0%），因为视角变化改变了 perceptual feature distribution，memory 中存的是旧视角特征，与新帧不匹配。

## 6. 关键设计 Intuition 总结

### 6.1 为什么分离 Perceptual 和 Cognitive 双流？

- **维度差异**：$d_p = 256$ vs $d_c = 4096$，如果 concat 后做单一 attention，高维 cognitive 会被低维 perceptual 主导或反之
- **检索语义不同**：cognitive retrieval 回答"我在执行哪个子任务"，perceptual retrieval 回答"关键物体上次在哪个位置"
- **更新频率不同**：cognitive 在子任务切换时变化，perceptual 每帧变化，独立 consolidation 更合理

### 6.2 为什么 Gate Fusion 优于 Add？

考虑 Change Food 任务：当前帧已能看见两个食物的精确位置，但不知道哪个是"先拿走的"。此时：
- 对 **perceptual stream**，gate 应趋近 0（当前帧已够精细），用当前 $p$
- 对 **cognitive stream**，gate 应趋近 1（需要历史语义"我在换食物步骤"），用 retrieved $H^c$

Gate 允许 per-stream、per-channel 自适应，Add 强制等权混合，无法区分两类信息的不确定性。

### 6.3 为什么 Token Merge 优于 FIFO？

FIFO 假设"越老越无用"，但 non-Markovian 任务中恰恰相反——最初揭示的信息（如 ShellGameTouch 中 cube 位置）可能跨越整个 episode 都关键。Token Merge 假设"相邻相似则冗余"，这与机器人操作中"相邻帧视觉变化平缓"的物理先验一致，合并冗余同时自然保留独特事件。

### 6.4 Timestep PE 的作用

没 timestep PE 时，attention 只能基于 content similarity 检索。但 content 相似的两个历史帧可能含义完全不同（一个"刚揭示 cube"，一个"已被盖住很久后再次类似视角"）。Timestep PE 让模型能区分**时间远近**，对"最初揭示"vs"近期执行"两类信息赋予不同权重。

## 7. 效率分析（Tab.15）

| Model | Latency (4090) | Throughput (4090) | Memory |
|---|---|---|---|
| Baseline | 0.187s | 85.6 Hz | 15.8 GB |
| MemoryVLA | 0.194s | 82.5 Hz | 16.6 GB |

仅 **+3.6% latency** 和 **+0.8 GB memory**。原因：
- Cognitive memory 每条仅 1 个 token（不是多 token 序列）
- Perceptual memory 压缩到 256 channel
- Token Merge 持续合并冗余，限制 retrieval 的 key/value 数量

这意味着 memory module 几乎是"免费午餐"，可即插即用到现有 VLA 架构。

## 8. 局限与未来方向

Paper 自己提到：
1. **Memory reflection**：当前 memory 是隐式 embedding，未来可 align 到 LLM input space，实现 embedding-space chain-of-thought reasoning
2. **Lifelong memory**：当前 consolidation 是 averaging，未来可借鉴生物学机制将高频经验蒸馏为永久 representation，支持跨 scene/task/embodiment 的 scalable generalization

**我自己补充的思考**：
- **Memory reflection** 听起来很有意思，类似将 episodic memory 转化为 semantic memory 的过程，让模型能"反思过去"而不仅仅是"检索过去"
- 当前 cognitive token 只用 1 个 EOS token，是否可以引入 hierarchical cognitive tokens（不同抽象层次）值得探索
- Memory capacity 固定，是否可以做 dynamic capacity allocation，简单任务少存、复杂任务多存
- 与 retrieval-augmented generation (RAG) 的关系值得深究——MemoryVLA 本质上是 **robotic RAG**，但 retrieval 是从同一 episode 的历史而非外部知识库

## 9. 与相关工作的位置

在 VLA 时序建模谱系中：
- **Interleave-VLA** / **RoboFlamingo**：直接拼接帧，计算昂贵 / LSTM 压缩损失细节
- **TraceVLA**：在当前帧上画历史轨迹，丢弃语义
- **UniVLA**：将历史 action 加入 prompt，仅 CoT 形式
- **CronusVLA**：sliding window 聚合多帧 VLM features，缺乏 consolidation
- **MemoryVLA**：首个引入 hippocampus-like **dual-stream + retrieval + gating + consolidation** 完整 memory 框架

参考 GitHub：https://github.com/shihao1895/MemoryVLA

---

如果你想深入某部分（例如 diffusion action expert 的具体 denoising 步骤、PCMB 的 training-time vs inference-time 行为差异、或某具体任务的 memory retrieval 可视化分析），我可以进一步展开。
