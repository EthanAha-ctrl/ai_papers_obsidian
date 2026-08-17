---
source_pdf: MultiWorld Scalable Multi-Agent Multi-View.pdf
paper_sha256: 0f8d14d90d5f1401b5f75569f885ea31d9b63ddbf3d1fab3ca96e68fcfb27336
processed_at: '2026-08-05T21:42:06-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好的 Karpathy，我们抛开那些学术黑话，直接从最底层的 intuition 来拆解这篇 paper。这篇文章的核心其实就是试图回答一个非常硬核的问题：**当 environment 里有多个 agent 同时搞事情，且每个 agent 都有自己的视角时，video world model 怎么才能不崩盘？**

我们之前熟悉的 video world model（比如 Oasis, Genie, The Matrix）默认都是 single-agent 的，模型只需要处理一个 observation 和一个 action stream。但真实世界是 multi-agent 的，比如两个人一起打游戏、两个机器人一起搬砖。一旦变成 multi-agent，就会出现两个非常致命的底层逻辑问题，MultiWorld 就是来解这两个问题的。

---

### 痛点 1：Agent 身份识别危机与 Action 聚合

**问题出在哪？**
假设有两个 agent，A 和 B。如果 A 向左走，B 向右走，此时 environment 的变化是确定的。但如果你把 A 和 B 的 action 简单地 stack 在一起扔给 Transformer，由于 self-attention 天生的 permutation invariance（排列不变性），模型根本分不清到底是"A左B右"还是"A右B左"。这就导致 action-conditioning 完全失效，模型只能看到"一个左一个右"，但不知道谁对应谁，最后生成的视频里两个 agent 的动作可能全是乱的。

另外还有一个问题：某一帧里 A 在疯狂操作，B 挂机不动。如果模型把两者的 action 等权相加，B 的 zero-action 会把 A 的 action signal 给稀释掉。

**MultiWorld 的解法：MACM (Multi-Agent Condition Module)**
MACM 用了两个非常巧妙的 trick 来解决身份和稀释问题。

**Trick 1: Agent Identity Embedding (AIE) —— 借用 RoPE 给 agent 贴标签**
作者直接把 LLM 里的 Rotary Position Embedding (RoPE) 拿过来，原本 RoPE 是用来编码 token 在 sequence 中的位置的，这里直接拿来编码 agent 的身份。

公式很简单：
$$ \text{AIE}(a_i, i) = \mathbf{R}_{\Theta, i} a_i $$
这里的 $a_i$ 是第 $i$ 个 agent 的 action embedding（维度是 $D$）。$\mathbf{R}_{\Theta, i}$ 是一个 rotation matrix（旋转矩阵）。这就相当于在 high-dimensional space 里，把 agent 1 的 action embedding 旋转一个角度，把 agent 2 的 action embedding 旋转另一个角度。

为什么旋转能解决身份问题？我们看 attention score 的计算：
$$ (\mathbf{R}_m \mathbf{a}_m)^\top (\mathbf{R}_n \mathbf{a}_n) = \mathbf{a}_m^\top \mathbf{R}_{n-m} \mathbf{a}_n $$
变量解释：$m$ 和 $n$ 是两个不同 agent 的 index。$\mathbf{R}_{n-m}$ 是它们之间的相对旋转矩阵。
Intuition：两个 agent 之间的 attention 只取决于它们的 **相对身份差** ($n-m$)。这就打破了 permutation symmetry！如果模型在训练时见过 agent 1 和 agent 2 的互动，当推理时来了 agent 3 和 agent 4，只要它们的相对差也是 1，模型就能直接泛化过去。这种 relative encoding 的 extrapolation 能力，让 framework 可以支持任意数量的 agent。

**Trick 2: Base Frequency 调整 —— 把 LLM 的刻度尺换成 Multi-Agent 的刻度尺**
这个细节非常体现工程直觉。RoPE 原本在 LLM 里 base frequency $b$ 一般设为 10000。但在 multi-agent 场景里，agent 数量通常只有 2 到 4 个。
公式里频率的计算是 $\theta_j = b^{-2j/D}$。如果 $b=10000$，相邻 agent 之间的旋转角度差微乎其微，在 embedding space 里几乎重合，模型还是分不清谁是谁。
作者在 ablation (Table 3) 里把 $b$ 降到了 20。这就相当于放大了刻度尺，相邻 agent 之间的角度差变大了，模型一眼就能区分开。
参考：[RoFormer (RoPE) paper](https://arxiv.org/abs/2104.09864)

**Trick 3: Adaptive Action Weighting (AAW) —— 动态聚焦**
用一个 MLP 给每个 agent 的 action token 算一个权重 $w_k$，然后加权求和：
$$ \mathbf{a}_f^{\text{unified}} = \sum_{k=1}^K w_k \cdot \mathbf{a}_f^k $$
Intuition：谁在动，谁的权重大。挂机的人 action 被 zero out，活跃的人主导当前的 environment dynamics。

---

### 痛点 2：Multi-View 的 3D 一致性幻觉

**问题出在哪？**
两个 agent 玩《双人成行》，屏幕是 split-screen，左边是 A 的视角，右边是 B 的视角。由于他们在同一个 3D 空间里，如果 A 的视角里把一个箱子推倒了，B 的视角里这个箱子也必须是倒下的，且在 3D 几何上必须严丝合缝。
如果我们用两个独立的 video generation model 分别生成 A 和 B 的视角，几何肯定崩掉。如果像 concurrent work Solaris 那样，把两个 view 的 token 在 sequence 维度上 interleave 起一起做 self-attention，计算量会随 view 数量呈平方爆炸，根本没法 scale 到多个 view。

**MultiWorld 的解法：Global State Encoder (GSE) 借用 3D Foundation Model**
作者的 insight 是：**所有的 view 都只是 underlying 3D environment 的一个 partial observation。我们要维护的应该是那个 underlying 3D state，而不是去强行对齐 2D pixel。**

但重新建一个 3D point cloud 或者 NeRF 太慢且不好 differentiate。作者直接白嫖了一个最近爆火的 3D foundation model **VGGT** (Visual Geometry Grounded Transformer)。
参考：[VGGT project page](https://vggt.github.io/)

公式：
$$ \mathbf{H}_{\text{vggt}} = \text{VGGT}(\mathbf{O}) $$
$$ \mathbf{H} = \text{MLP}(\mathbf{H}_{\text{vggt}}) $$
变量解释：$\mathbf{O} = \{\mathbf{O}_1, \dots, \mathbf{O}_C\}$ 是 $C$ 个相机视角的 initial frame。VGGT 把这些图片吃进去，吐出 latent $\mathbf{H}_{\text{vggt}} \in \mathbb{R}^{C \times n \times d}$（$n$ 是每张图的 token 数，$d$ 是 latent dimension）。这个 latent 里已经隐式包含了 multi-view 的 3D 几何关系。然后用个 MLP 对齐一下维度，作为 condition 喂给 video DiT。

为什么这个设计极好？
1. **3D 一致性白嫖**：VGGT 在海量 3D 数据上 pretrain 过，它的 latent space 天生就是 3D-aligned 的。
2. **View 数量解耦**：不管你喂进去 2 个 view 还是 10 个 view，VGGT 都能 encode 成一个 global state，计算复杂度和 view 数量 $C$ 没有强绑定。
3. **Parallel Generation**：既然有了 global state $\mathbf{H}$，各个 view 的视频就可以 **完全并行生成**，这就好比几个画家看着同一个 3D 模型在不同角度画画，互不干扰，这比让几个画家互相看对方画板来对齐要快得多。

---

### 把数学也翻译成人话：Flow Matching 在干嘛？

这篇 paper 用的 video generation backbone 是基于 Flow Matching (FM) 的 DiT（用的阿里的 Wan2.2-5B）。FM 现在是 diffusion model 的升级版，公式看起来这样：
$$ \mathbf{x}_c^t = (1-t)\mathbf{x}_c + t\boldsymbol{\epsilon} $$
$$ \mathbf{u} = \boldsymbol{\epsilon} - \mathbf{x}_c $$
变量解释：$\mathbf{x}_c$ 是 camera $c$ 的真实视频帧。$\boldsymbol{\epsilon}$ 是随机高斯噪声。$t$ 是时间进度（从 0 到 1）。
Intuition：把生成过程看作一条从噪声 ($t=1$) 到真实数据 ($t=0$) 的直线。网络 $v_\theta$ 要学的就是这条直线上的速度场 $\mathbf{u}$。训练时就是让网络去猜这个速度。生成时，从纯噪声出发，顺着网络猜出来的速度一步步倒推回真实视频。
这里的一个关键细节是 **Causal Mask**：预测第 $i$ 帧的时候，模型只能看 $\{0, \dots, i\}$ 帧的 action，绝对不能偷看未来的 action。这对于长程的 autoregressive rollout 至关重要，不然模型在推理时会因为无法作弊而产生严重的 train-test gap。
参考：[Flow Matching paper](https://arxiv.org/abs/2210.02747), [Wan2.2](https://github.com/Wan-Video/Wan2.2)

---

### 实验里的门道：我们到底信不信这些 Metric？

这篇 paper 用的评估指标非常刁钻，专门针对 multi-agent 和 multi-view 的痛点。

**1. RPE (Reprojection Error) —— 拿 SLAM 的尺子量 Video Generation**
怎么证明生成的 multi-view 视频在 3D 几何上是一致的？作者直接调用了 DROID-SLAM。
公式：
$$ \text{RPE} = \frac{1}{|\mathcal{V}|} \sum_{(i,j) \in \mathcal{V}} \|\mathbf{p}_{ij}^* - \Pi(\mathbf{P}_{ij})\|_2 $$
变量解释：$\mathcal{V}$ 是特征匹配点对的集合。$\mathbf{P}_{ij}$ 是由 SLAM 估算出来的 3D 空间点。$\Pi$ 是相机投影函数。$\mathbf{p}_{ij}^*$ 是生成视频里的 2D 像素点。
Intuition：如果你生成的视频里，camera A 看到一个点，把它在 3D 里还原，然后投影到 camera B 的画面上，如果和 camera B 实际生成的像素位置对不上，RPE 就会很大。这就直接打脸了那些"每个 view 看起来都很漂亮，但拼在一起 3D 空间是扭曲的"模型。
参考：[DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM)

**2. Action Following —— 让 IDM 当裁判**
怎么证明视频真的听从了 multi-agent action？用 Inverse Dynamics Model (IDM) 反推。
Intuition：训练一个网络，输入生成的视频，让它猜"这俩人当时按了什么键？"。如果猜出来的 action 和输入给 world model 的 ground truth action 一样，说明 world model 真的把 action 落实到了画面里。这是从 VPT 借来的思路。
参考：[VPT (Video PreTraining)](https://arxiv.org/abs/2206.11695)

**3. 核心对比实验 Table 1 解读**
在《双人成行》游戏上，Standard baseline (无 MACM 无 GSE) 的 FVD 是 245。加上 MACM 和 GSE 后，MultiWorld 降到了 179。这个下降幅度非常夸张。RPE 从 0.75 降到了 0.67。
更有意思的是 Table 5 的 ablation。作者试了用 Wan VAE 和 DINOv2 做 Global State Encoder，结果 FVD 反而变差了（256 和 232，比不加还差）。只有用 VGGT 才有正收益。这证明随便塞 2D 特征是没用的，必须要有强 3D prior 的 latent 才能帮到 video generation。

---

### 更深层的 Intuition 联想与未来推演

Karpathy，聊到这里，我们可以再往深了挖一下这篇 paper 背后的哲学。

**1. 离散的 Agent interaction 与连续的 Video Latent 之间的桥梁**
在这个框架里，agent 的 action 是离散的或者结构化的（键盘按键、机械臂关节），而 environment 的变化是 high-dimensional continuous video。MACM 本质上是一个 **信息瓶颈**，它把多组离散 action 先通过 AIE 和 self-attention 在 agent 之间进行协商，然后通过 AAW 压缩成一个 unified continuous token，再去驱动 video generation。这种设计避免了让 DiT 直接去处理复杂的多 agent 控制逻辑，分工明确。

**2. 3D Foundation Model 赋能 Video World Model**
之前大家做 world model，基本都是纯 2D 卷积或者纯 Transformer 硬啃，试图让模型自己从 video 数据里 "悟出" 3D 规律（比如 Genie）。MultiWorld 走了一条更实用主义的路：**既然我已经有了 VGGT 这样强大的 3D foundation model，为什么不直接把它的 3D 感知能力 "注入" 到 video model 里？**
这种不同 modality foundation model 之间的嫁接，可能会成为未来 world model 的标配。以后可能会有 Physics Engine foundation model 专门注入物理规律，Audio foundation model 注入声音规律。

**3. Long-horizon Autoregressive 的 Memory 机制**
论文里提到支持 2x 甚至 4x training context length 的 long-horizon generation。它的做法很朴素：生成完一个 chunk 后，把最后一个 frame 抽出来，重新喂回 VGGT 更新 global state $\mathbf{H}$，然后继续生成下一个 chunk。
Intuition：这就相当于 agent 在 environment 里走了一段路，回头拍张照，把这张照片作为新的"起点记忆"继续往前走。目前的 memory 就是这最后一张图，非常 short-term。
未来如果要 scale 到几千个 agent、几天的交互，这种 single-frame memory 肯定不够。可能需要引入 LLM 里的 KV-cache 机制，或者在 latent space 里维护一个持续更新的 episodic memory module，就像 WorldMem 或者 InfiniteWorld 那样。
参考：[WorldMem](https://arxiv.org/abs/2504.12369), [InfiniteWorld](https://arxiv.org/abs/2602.02393)

**4. 为什么这不只是 Video Generation，而是真正的 "World Simulator"？**
看 Figure 4 里的 Multi-Robot Failure Trajectory Simulation。机器人训练 RL 的时候，最缺的就是 failure case，因为收集 failure case 要撞坏机器人。MultiWorld 能根据 action 生成机器人相撞的视频。这意味着这个 model 真的学到了 environment dynamics 里的因果关系，它知道"这样操作会撞"。
这已经触及了 world model 的核心价值：它是一个可交互的、可以生成反事实情况的 simulator。以后 multi-agent RL 训练，可能直接在 MultiWorld 这样的 latent space world model 里跑，速度比物理引擎快几个数量级。

总结一下，MultiWorld 这篇 paper 的核心 intuition 就是：**用 RoPE 解决 agent 间的相对关系，用 VGGT 解决 view 间的 3D 几何关系**。虽然公式看着多，但底层逻辑非常清晰，是我们在构建通用 world model 路上的一个非常扎实的 step。

希望这个大白的讲解能帮你建立起对这篇 paper 最直接的 intuition。如果你对其中某个 module（比如 VGGT 内部是怎么处理任意数量图片的，或者 AAW 的 MLP 具体架构）还有疑问，我们可以继续往下钻。

---

# MultiWorld: Scalable Multi-Agent Multi-View Video World Models 深度技术讲解

## 1. 核心问题与 Motivation 的 Intuition

这篇 paper 要解决的核心问题可以抽象为一个非常深刻的建模难题：**当一个 environment 被多个 agent 同时影响，且每个 agent 从不同 viewpoint 观察 environment 时，video world model 如何同时满足 (1) 每个 agent action 的精确控制，(2) 跨 viewpoint 的几何一致性，(3) agent 数量和 view 数量的可扩展性**。

Karpathy 你在 world model 上有大量思考，你知道 single-agent video world model（如 Genie 2, Oasis, GameFactory, The Matrix）已经能在单一 observer 假设下 work 得不错。但是真实 environment 几乎总是 multi-agent 的：collaborative robotics、multi-player games、autonomous driving 中的多车交互。当扩展到 multi-agent 时，会出现三个 single-agent 框架无法自然处理的问题：

**(1) Multi-Agent Controllability 的 identity ambiguity**：简单地把多个 agent 的 action stack 起来，模型无法区分 "Agent 1 向左、Agent 2 向右" 与 "Agent 1 向右、Agent 2 向左"——这是一个 permutation symmetry 问题。在 attention 操作下，这两组输入会得到相同的输出，导致 action-conditioning 失效。

**(2) Multi-View Consistency 的几何约束**：每个 agent 看到的是 shared 3D environment 的一个 partial observation。不同 view 必须在 underlying 3D scene 上保持一致。如果只是独立生成每个 view，物体可能在一个 view 中移动而在另一个 view 中消失，违反 3D 几何。

**(3) Framework Scalability 的架构问题**：之前的工作（如 concurrent work Solaris [34]）interleave 两个 view 在 sequence dimension 上做 shared self-attention，这种方式 memory 随 view 数 $C$ 二次增长，无法 scale 到任意 view 数。COMBO [67] 把多个 single-agent 模型组合起来，但完全忽略 inter-agent interaction。

MultiWorld 的核心 insight 是：**把 multi-view generation 解耦为一组 single-view generation，所有 view 共享一个从多视角 observation 中提取的 compact 3D-aware global environment state；同时把 multi-agent action 通过 rotary position embedding 打破 permutation symmetry，并通过 adaptive weighting 聚合为 unified action token**。这种设计让 computation 与 $C$ 解耦，与 $K$ 也解耦。

参考链接：
- MultiWorld (假设): https://arxiv.org/abs/2507.XXXXX
- VGGT: https://vggt.github.io/
- Wan2.2: https://github.com/Wan-Video/Wan2.2
- Genie 3: https://deepmind.google/discover/blog/genie-3
- Oasis: https://oasis-model.github.io/
- The Matrix: https://arxiv.org/abs/2412.03568

---

## 2. 方法细节：从公式到 Intuition

### 2.1 Backbone 与 Notation

Backbone 是基于 Flow Matching (FM) [Lipman et al. 2023] 的 DiT (Diffusion Transformer) [Peebles & Xie 2023]，具体使用了 Wan2.2-5B 作为 base model。这里需要把 FM 的数学讲清楚，因为后面 MACM 和 GSE 都是在 FM 的 velocity prediction 框架上做 conditioning。

**符号定义**：
- $K$：agent 数量（player 或 robot）
- $C$：camera view 数量，与 $K$ 独立
- $a_i = (a_i^1, \dots, a_i^K)$：在第 $i$ 帧所有 $K$ 个 agent 的联合 action，其中上标 $k$ 表示第 $k$ 个 agent
- $\mathbf{a} = \{a_0, \dots, a_I\}$：完整 action 序列，$I$ 是总帧数
- $\mathbf{x}_c$：camera $c \in \{1,\dots,C\}$ 录制的视频
- $\mathbf{o} = \{\mathbf{o}_c\}_{c=1}^C$：environment observation 集合，$\mathbf{o}_c$ 是 $\mathbf{x}_c$ 的第一帧（initial frame），作为 global scene context

**Flow Matching 公式**：

$$\mathbf{x}_c^t = (1-t)\mathbf{x}_c + t\boldsymbol{\epsilon}, \quad \mathbf{u} = \boldsymbol{\epsilon} - \mathbf{x}_c$$

这里：
- $t \sim \mathcal{U}(0,1)$ 是 flow time，$t=0$ 对应 clean data $\mathbf{x}_c$，$t=1$ 对应 pure noise $\boldsymbol{\epsilon}$
- $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 是 Gaussian noise
- $\mathbf{u}$ 是 target velocity field，把 $\mathbf{x}_c^t$ 从 $t$ 推到 $t+dt$ 的方向
- 注意这里的 parameterization 是 rectified flow 风格（Liu et al. 2023），velocity 直接指向 noise-to-data 的反方向

**Velocity network**：

$$v_\theta(\mathbf{x}_c^t, t, \mathbf{a}, \mathbf{o})$$

这个 network 接收四个输入：noisy observation $\mathbf{x}_c^t$、flow time $t$、action sequence $\mathbf{a}$、environment observation $\mathbf{o}$。这里有一个非常重要的设计点：**action 和 observation 对所有 view 共享，但 $\mathbf{x}_c^t$ 是 per-view 的**。这是后面 parallel generation 的基础。

**Causal mask for temporal causality**：在 action cross-attention 上应用 frame-wise causal mask，确保第 $i$ 帧的 video token 只能 attend 到 $\{a_0, \dots, a_i\}$，防止未来 action 信息泄漏。这对 long-horizon autoregressive generation 至关重要——如果没有 causal mask，autoregressive rollout 时模型会"作弊"利用未来 action。

**FM 训练目标**：

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t,\epsilon}\left[\|v_\theta(\mathbf{x}^t, t, \mathbf{a}) - \mathbf{u}\|_2^2\right]$$

**Sampling**：通过 Euler solver 迭代求解 ODE：

$$d\mathbf{x} = v_\theta(\mathbf{x}^t, t, \mathbf{a}) \cdot dt$$

从 $t=1$ (noise) 积分到 $t=0$ (clean video)。

### 2.2 Multi-Agent Condition Module (MACM)

这是这篇 paper 最有 idea 的部分。MACM 要解决两个 sub-problem：(1) identity ambiguity，(2) action strength variation。

#### 2.2.1 Agent Identity Embedding (AIE) —— 打破 permutation symmetry

**核心 insight**：如果不给每个 agent 一个独特的 "身份标记"，attention 操作在 multi-agent action 上是 permutation-invariant 的。比如有 $K=2$ 个 agent，action $(a^1, a^2)$ 和 $(a^2, a^1)$ 经过 self-attention 会得到相同输出。这就导致 "Agent 1 左移 + Agent 2 右移" 与 "Agent 1 右移 + Agent 2 左移" 在模型看来无法区分。

**为什么用 RoPE？** RoPE (Rotary Position Embedding, Su et al. 2024) 原本用于 sequence position，这里被创造性地用于 agent identity dimension。RoPE 的好处是 **relative encoding**：两个 agent 之间的相对位置（identity 差）会被自然编码进 attention score。这支持 extrapolation 到训练时未见过的 agent 数量。

**公式 (1)**：

$$\text{AIE}(a_i, i) = \mathbf{R}_{\Theta, i} a_i$$

- $a_i \in \mathbb{R}^D$：第 $i$ 个 agent 的 action embedding，$D$ 是 latent dimension
- $\mathbf{R}_{\Theta, i}$：第 $i$ 个 agent 的 rotation matrix，由 frequencies $\theta_j = b^{-2j/D}$ 定义，$b$ 是 base frequency
- 下标 $i$ 是 agent index（注意这里 $i$ 不是 frame index，paper 中省略了 frame 下标）
- 上标维度 $(2j, 2j+1)$ 是 embedding 中相邻的一对维度

**公式 (2)** —— 对每对维度 $(2j, 2j+1)$ 的 2D rotation：

$$\begin{pmatrix} a^{(2j)} \\ a^{(2j+1)} \end{pmatrix}_{out} = \begin{pmatrix} \cos(i\theta_j) & -\sin(i\theta_j) \\ \sin(i\theta_j) & \cos(i\theta_j) \end{pmatrix} \begin{pmatrix} a^{(2j)} \\ a^{(2j+1)} \end{pmatrix}_{in}$$

这是一个标准的 2D rotation matrix 作用在 embedding 的每一对维度上，rotation angle 是 $i \cdot \theta_j$，其中 $i$ 是 agent index，$\theta_j = b^{-2j/D}$ 是第 $j$ 对维度的频率。

**公式 (3)** —— 证明 RoPE 的 relative property 在 attention score 中起作用：

$$(\mathbf{R}_m \mathbf{a}_m)^\top (\mathbf{R}_n \mathbf{a}_n) = \mathbf{a}_m^\top \mathbf{R}_m^\top \mathbf{R}_n \mathbf{a}_n = \mathbf{a}_m^\top \mathbf{R}_{n-m} \mathbf{a}_n$$

这里：
- $\mathbf{R}_m, \mathbf{R}_n$ 是 agent $m$ 和 agent $n$ 的 rotation matrix
- 由于 rotation matrix 的性质 $\mathbf{R}_m^\top \mathbf{R}_n = \mathbf{R}_{n-m}$（两个 rotation 的复合等于一个 rotation，angle 是差值）
- 所以 attention score $\mathbf{a}_m^\top \mathbf{R}_{n-m} \mathbf{a}_n$ 只依赖于 agent $m$ 和 agent $n$ 的 **relative identity** $n-m$

这就是为什么 AIE 可以 extrapolate：训练时见过 agent 1, 2, 3，推理时来了 agent 4，它和 agent 1 的 relative identity 是 3，如果训练时见过 relative identity 3 (agent 1 vs agent 4)，那么 extrapolation 就是 well-defined 的。

**Base frequency 的 ablation（Table 3）**：

| Config | FVD↓ | PSNR↑ | Action↑ |
|--------|------|-------|---------|
| base=10k | 234 | 17.53 | 89.2 |
| base=20 | 228 | 17.56 | 89.7 |

这个 ablation 非常关键。原始 RoPE 用 $b=10000$ 是因为 LLM 的 sequence length 通常是几千 token，需要频率覆盖大范围。但 multi-agent 场景下 $K$ 通常只有 2-4，相邻 agent 的 angle 差 $\theta_0 = b^{-0} = 1$ rad（base=10000 时）会让所有 agent 的 rotation 几乎相同，失去 distinguishability。

把 base 降到 20 后，$\theta_0 = 1, \theta_1 = 20^{-2/D}$。对于 $D=128$，相邻维度的 angle 差从约 $0$ 增大到约 $0.6$ rad，让相邻 agent 在 embedding space 中分开。这是一个非常 fine-grained 但 crucial 的 design choice，体现作者对 RoPE 在不同 scale 下的行为的深入理解。

#### 2.2.2 Adaptive Action Weighting (AAW)

**Motivation**：不同 agent 在不同时刻的 "action strength" 不同。比如游戏里一个 player 在快速奔跑，另一个 player 站着不动。如果对所有 agent action 等权求和，stationary agent 的 zero action 会稀释 active agent 的信号。

**方法**：用 MLP 预测每个 agent action token 的 weight，然后 weighted sum 聚合为 per-frame 的 unified action token：

$$\mathbf{a}_f^{\text{unified}} = \sum_{k=1}^K w_k \cdot \mathbf{a}_f^k$$

其中 $w_k = \text{MLP}(\mathbf{a}_f^k) \in \mathbb{R}$ 是第 $k$ 个 agent 在 frame $f$ 的 adaptive weight。

**Ablation（Table 4）**：

| Config | FVD↓ | PSNR↑ | Action↑ |
|--------|------|-------|---------|
| w/o AAW | 245 | 17.48 | 88.4 |
| w/ AAW | 236 | 17.52 | 88.6 |

提升幅度不大但方向一致。我直觉这个 module 在 multi-robot scenario（4 agents）比 2-agent game 更重要，因为 agent 数多时 stationary 的概率更高。

#### 2.2.3 MACM 的整体 pipeline

1. Encode 每个 agent 的 action → action tokens $\mathbf{a}_f \in \mathbb{R}^{K \times D}$
2. Apply AIE (RoPE rotation along agent dimension) → break permutation symmetry
3. Self-attention across agent dimension → model inter-agent interaction (公式 3)
4. AAW: MLP 预测 weights + weighted sum → unified action token $\mathbf{a}_f^{\text{unified}} \in \mathbb{R}^D$
5. Causal cross-attention: DiT 的 video tokens attend to unified action tokens，frame $i$ 只看 $\{a_0, \dots, a_i\}$

### 2.3 Global State Encoder (GSE) —— 多视角 3D 一致性

这是另一个核心 idea，解决了 multi-view consistency 问题。

**Motivation**：multi-agent environment 是一个 shared 3D world，每个 agent 看到 partial observation。要保证 view consistency，最直接的想法是 reconstruct 一个 3D representation (point cloud, NeRF, 3DGS) 然后从不同视角 render。但这 expensive 且 differentiability 差。

**Insight**：用 pretrained 3D foundation model VGGT (Visual Geometry Grounded Transformer, Wang et al. 2025) 作为 frozen backbone，提取 3D-aware latent representation，作为 cross-attention 的 condition。**不显式 reconstruct 3D，直接用 latent**。

**VGGT 简介**：VGGT 是一个 feed-forward transformer，输入任意数量图像，输出 camera pose、depth map、point map 等 3D 量。它通过大规模 multi-view 3D 数据训练，内部隐式建模了 multi-view geometry。关键 property：它的 latent space 是 3D-aware 的，即 latent 之间满足某种 3D 一致性约束。

**GSE 公式**：

输入：multi-view observation $\mathbf{O} = \{\mathbf{O}_c\}_{c=1}^C$，其中 $\mathbf{O}_c \in \mathbb{R}^{3 \times H \times W}$

VGGT 编码：
$$\mathbf{H}_{\text{vggt}} = \text{VGGT}(\mathbf{O}) \in \mathbb{R}^{C \times n \times d}$$

- $C$：view 数量
- $n$：每张 image 的 token 数（如 256）
- $d$：latent dimension

MLP 维度对齐：
$$\mathbf{H} = \text{MLP}(\mathbf{H}_{\text{vggt}})$$

注入 DiT 通过 cross-attention，与 action cross-attention 并列。

**为什么这个设计 work？**

1. **3D-aware consistency**：VGGT 的 latent 已经 encode 了 multi-view 几何关系，所以从这个 latent condition 出来的不同 view 自然满足 3D 一致性
2. **View count scalability**：VGGT 处理任意 $C$，输出 $C \times n \times d$。GSE 不显式依赖 $C$，只要 $C \geq 1$
3. **Parallel generation**：每个 view 的 video generation 独立做 FM sampling，但都 condition on 同一个 $\mathbf{H}$，所以可以并行

**Ablation（Table 5）—— GSE backbone 选择**：

| Global State Encoder | FVD↓ | LPIPS↓ | SSIM↑ | PSNR↑ | RPE↓ |
|---------------------|------|--------|-------|-------|------|
| w/o Global State | 228 | 0.36 | 0.51 | 17.56 | 0.75 |
| Wan VAE | 256 | 0.36 | 0.50 | 17.38 | 0.71 |
| DINOv2 | 232 | 0.36 | 0.50 | 17.48 | 0.72 |
| VGGT (Ours) | **179** | 0.35 | 0.51 | 17.72 | **0.67** |

观察：
- **w/o Global State**：FVD 228, RPE 0.75。这是 baseline，没有跨 view 信息
- **Wan VAE**：FVD 反而变差到 256！这是因为 VAE 只是 per-image latent encoder，不建模 cross-view 关系。引入这些 latent 给 model 增加了 noise 而非 signal
- **DINOv2**：FVD 232，比 baseline 略差。DINOv2 是 2D semantic feature，不建模 3D geometry，对 multi-view consistency 帮助有限
- **VGGT**：FVD 大幅提升到 179，RPE 从 0.75 降到 0.67。这是 dramatic improvement

**Intuition**：这验证了一个核心 hypothesis——multi-view consistency 需要 3D-aware representation，而不仅仅是 2D semantic 或 per-view latent。VGGT 在 3D reconstruction 任务上 pretrain 的 latent 天然携带 3D 几何 prior，可以直接 transfer 到 video world model 的 multi-view conditioning。这是一个非常好的 transfer learning 例子。

### 2.4 Scalable Framework 与 Autoregressive Generation

**Scalability 的两层**：
1. Agent scalability：AIE 用 RoPE 的 relative encoding，可以 extrapolate 到任意 $K$
2. View scalability：GSE 把任意 $C$ 个 view 压缩为 unified global state，computation 与 $C$ 解耦

**Parallel generation**：每个 view 的 FM sampling 独立运行，shared $\mathbf{H}$ 作为 condition。在 double-view 实验中实现约 1.5× speedup over sequential generation。

**Autoregressive long-horizon generation**：
1. 生成第一个 chunk（所有 $C$ 个 view）
2. 取每个 view 的最后一帧作为 new observation
3. 把 new observation 喂回 GSE，更新 $\mathbf{H}$
4. 用 updated $\mathbf{H}$ 作为下一个 chunk 的 condition
5. 重复

这个 design 让 model 可以 generate 超过 training context length 2 倍长度的 video 而无明显 quality drop，4 倍长度 minimal degradation。这是 long-horizon world simulation 的关键能力。

---

## 3. 架构图（Figure 2）解析

Figure 2 展示了完整 pipeline：

**Input side**：
- Multi-view initial frames $\{\mathbf{o}_1, \mathbf{o}_2, \dots, \mathbf{o}_C\}$ → VGGT (frozen) → MLP → Global State $\mathbf{H}$
- Multi-agent actions per frame $\{a_0, a_1, \dots, a_I\}$，每个 $a_i = (a_i^1, \dots, a_i^K)$ → action encoder → AIE (RoPE) → self-attention → AAW → unified action tokens

**Backbone**：
- DiT 接收 noisy video $\mathbf{x}_c^t$、flow time $t$
- 通过 cross-attention 接收 global state $\mathbf{H}$ 和 action tokens（causal mask）
- 输出 velocity prediction $v_\theta$

**Output side**：
- Per-view FM sampling 并行进行
- 生成 $\{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_C\}$

**Autoregressive loop**：
- 生成完成后，最后帧喂回 VGGT 更新 global state，循环

注意一个 important design choice：**VGGT 是 frozen 的**，只有 MLP 和 DiT 是 trainable。这避免了 expensive 3D pretraining，同时利用了 VGGT 的 3D prior。这种 partial freezing 与 FLUX、Stable Diffusion 3 用 frozen VAE 的思路一致。

---

## 4. 实验数据深度分析

### 4.1 主实验（Table 1）

**Multi-Player Video Game (It Takes Two)**：

| Method | FVD↓ | LPIPS↓ | SSIM↑ | PSNR↑ | Action↑ | RPE↓ |
|--------|------|--------|-------|-------|---------|------|
| Standard | 245 | 0.36 | 0.50 | 17.48 | 88.4 | 0.75 |
| Concat-View | 215 | 0.36 | 0.49 | 17.54 | 89.1 | 0.74 |
| COMBO | 207 | 0.34 | 0.51 | 17.82 | 89.3 | 0.72 |
| Ours | **179** | 0.35 | 0.51 | 17.72 | **89.8** | **0.67** |

**观察**：
- MultiWorld 在 FVD 上从 245 (Standard) 降到 179，relative improvement 27%。FVD 是 video quality 的综合指标，这个提升很大
- RPE 从 0.75 降到 0.67，relative improvement 10.7%。这是 multi-view consistency 的直接度量
- Action following 从 88.4 提升到 89.8。这个提升看起来不大，但考虑到 inverse dynamics model (IDM) 本身有上限，1.4% absolute 已经显著
- PSNR 上 COMBO 略高（17.82 vs 17.72），可能因为 COMBO 是 single-agent model 组合，每个 view 独立生成可能更"清晰"但牺牲 consistency

**Multi-Robot Manipulation (RoboFactory)**：

| Method | FVD↓ | LPIPS↓ | SSIM↑ | PSNR↑ | Action↑ | RPE↓ |
|--------|------|--------|-------|-------|---------|------|
| Standard | 100 | 0.07 | 0.90 | 26.39 | 88.2 | 1.60 |
| Concat-View* | 106 | 0.06 | 0.90 | 27.44 | 92.0 | 0.82 |
| COMBO | 99 | 0.08 | 0.90 | 26.49 | 88.5 | 1.54 |
| Ours | **96** | 0.07 | 0.90 | 26.60 | 88.7 | **1.52** |

**观察**：
- 在 robotics 场景 FVD baseline 已经很低（100），MultiWorld 提升到 96，幅度较小
- Concat-View 标注了 *，因为只在 dual-view 训练，不可直接比较。它在 RPE 上看起来很好（0.82），但这是因为 concat 在 2-view 时强行 enforce 了 alignment，无法 scale
- MultiWorld 的 RPE 1.52 比 Standard 的 1.60 略好，但远不如 game scenario 的改进。我直觉这是 robotics 场景视角差异小（都从机器人上方/侧方看），3D consistency 难度较低

### 4.2 主架构 ablation（Table 2）

| Config | FVD↓ | LPIPS↓ | SSIM↑ | PSNR↑ | Action↑ | RPE↓ |
|--------|------|--------|-------|-------|---------|------|
| Standard | 245 | 0.36 | 0.50 | 17.48 | 88.4 | 0.75 |
| + MACM | 228 | 0.36 | 0.51 | 17.56 | 89.7 | 0.76 |
| Both | 179 | 0.35 | 0.51 | 17.72 | 89.8 | 0.67 |

**关键观察**：
- **MACM 单独**：FVD 从 245 → 228（提升 17），Action 从 88.4 → 89.7（提升 1.3），RPE 几乎不变（0.75 → 0.76）。这证实 MACM 主要改善 action controllability，对 multi-view consistency 无直接帮助
- **MACM + GSE**：FVD 从 228 → 179（再提升 49），Action 微提升（89.7 → 89.8），RPE 从 0.76 → 0.67（提升 0.09）。这证实 GSE 主要改善 multi-view consistency，同时通过更 coherent 的 global context 间接提升 visual quality
- **Synergy**：FVD 从 245 → 179 的总提升 66 中，MACM 贡献 17，GSE 贡献 49，两者有强 synergy——GSE 提供的 global state 让 MACM 的 action 更好地 align 到 environment

### 4.3 长程生成（Figure 5）

Figure 5 展示 3 个 robot 协作堆叠 cube 的 long-horizon generation。论文声称：
- 2× training context window: 无显著 quality degradation
- 4× training context window: minimal quality loss

这个能力来自 autoregressive + global state update 的设计。每次 chunk 完成后，最后帧喂回 VGGT 重新计算 $\mathbf{H}$，相当于给 model 一个"短期记忆"的 anchor。这与 WorldMem [56]、InfiniteWorld [53] 的 memory mechanism 思路相通，但用 VGGT latent 作为 memory 比 token retrieval 更紧凑。

### 4.4 失败轨迹模拟（Figure 4）

这是一个有意思的应用：multi-robot 任务中，failure trajectory 难以收集（会损坏 robot），但 world model 可以 simulate。论文展示了 inter-robot collision 的 realistic simulation。这对 RL training data augmentation 很有价值——可以生成 "near-miss" trajectory 让 policy 学会避免 collision。

### 4.5 Action controllability（Figure 6）

论文展示 zero-action input 时 MultiWorld 生成 static video，避免 action bias（即模型在无 action 时仍生成 motion 的常见 failure mode）。这是 action-conditioning 严格的体现。我直觉这是因为 causal mask + AAW 的组合——当 action 全为零时，AAW 给 zero action 极低 weight，让 model 默认依赖 visual context 而非"想象" motion。

### 4.6 Physical consistency（Figure 8）

展示两个 case：
1. Shadow 在 opposite view 中保持一致
2. Snow footprint 在两个 camera 中持久存在

这表明 MultiWorld 不只是 pixel-level consistency，而是 capture 了 underlying physics (shadow、deformation)。这是 VGGT 3D prior 的 indirect benefit——3D 一致的 geometry 自然导致 shadow 和 footprint 一致。

---

## 5. Dataset 构建的细节

### 5.1 It Takes Two dataset
- 500 小时 raw 录制 @ 60 FPS，1440×2560 分辨率
- 筛选后保留 100 小时，21M 帧
- 降至 320×640 训练
- 筛选标准：
  1. 去除 cutscene（非交互）
  2. 只保留 side-by-side split-screen
  3. 去除大 camera motion 段（避免 motion blur）
  4. 去除 stationary 段（player 长时间 idle）

### 5.2 Action space unification
两个 player 用不同 controller（keyboard/mouse vs gamepad），action space 异构。解决方案：concatenate 两种 action 为单一 vector，对每个 player mask 另一 player 的 action 为 zero。这是一个 elegant 的 trick，让 single action encoder 处理两种 controller。

### 5.3 RoboFactory dataset
- 4 个 task：striking、2-robot stacking、3-robot stacking、4-robot passing
- 每个 task：1000 成功 episode + 2000 失败 episode
- Failure case 构造：基于成功 plan，在每步引入 controlled randomness，模拟 execution error。**纯 random action 产生 meaningless episode，反而 hurt training**。这是一个重要的 empirical insight。

### 5.4 数据 release
- Game dataset: 只 release 一个 chapter 的 subset（版权限制）
- Robotics dataset: 完全公开

---

## 6. Metrics 实现细节

### 6.1 Reprojection Error (RPE) —— Appendix B.1

RPE 用 DROID-SLAM [Teed & Deng 2021] 做 scene reconstruction：
1. 提取 frame-to-frame feature
2. 通过 differentiable Dense Bundle Adjustment (DBA) 优化 camera pose $G_t$ 和 per-pixel depth $d_t$
3. RPE 公式：

$$\text{RPE} = \frac{1}{|\mathcal{V}|} \sum_{(i,j) \in \mathcal{V}} \|\mathbf{p}_{ij}^* - \Pi(\mathbf{P}_{ij})\|_2$$

- $\mathcal{V}$：valid feature correspondence 集合
- $\mathbf{p}_{ij}^*$：generated frame 中观测到的 pixel position
- $\mathbf{P}_{ij}$：从 optimized depth 和 pose 重建的 3D point
- $\Pi$：camera projection function
- $\|\cdot\|_2$：Euclidean distance

RPE 衡量"如果我把一个 view 中的 3D point 投影到另一个 view，与实际观测的 pixel 差距"。RPE 越小说明 multi-view 几何一致性越好。这是 SLAM 中的标准 metric，借用到 video generation 评估很合适。

### 6.2 Action Following —— Appendix B.2

用 Inverse Dynamics Model (IDM) 反推 action：
- Architecture: ResNet-50 backbone + temporal processing layer
- 训练 20 epoch, lr=1e-4
- Game scenario：报告 discrete action accuracy
- Robotics scenario：报告 $100 \times (1 - \text{MSE}(a_{\text{pred}}, a_{\text{gt}}))$，normalized 让 higher 更好

这是一个间接评估 action following 的方法：如果 generated video 真的 execute 了指定 action，那么从 video 反推 action 应该和 ground truth 一致。

---

## 7. Implementation Details

- Base model: Wan2.2-5B (https://github.com/Wan-Video/Wan2.2)
- Training: 81 frames per video, 320×320 (game) 或 320×256 (robotics)
- Iterations: 40,000
- Learning rate: 5e-5, cosine scheduler
- Batch size: 64 (global)
- Hardware: 8× NVIDIA A800 GPU
- Training time: ~4 days

值得注意：Wan2.2-5B 是 text-to-video 模型，MultiWorld 在此基础上 fine-tune 为 action-conditioned multi-view model。这种 transfer learning 思路与 GameFactory、Oasis 一致。

---

## 8. 与 Related Work 的对比 Intuition

### 8.1 Single-agent world model 系列
- Genie 2/3 [DeepMind]：foundation world model，single agent
- Oasis [Decart]：real-time Minecraft，single agent
- The Matrix [Feng et al.]：infinite-horizon，single agent
- GameFactory [Yu et al.]：可创建新 game，single agent
- Hunyuan-GameCraft-2 [Tencent]：instruction-following，single agent

这些都在 single agent 假设下。MultiWorld 是 first systematic extension to multi-agent。

### 8.2 Multi-agent planning/simulation
- RoboFactory [Qin et al.]：physical simulator，需要手工设计 environment
- TeamCraft [Long et al.]：Minecraft multi-agent benchmark
- CausalMACE [Chai et al.]：causality-based multi-agent
- SeqWM-MARL [Zhao et al.]：sequential world model for MARL

这些都依赖 physical simulator 或 game engine。MultiWorld 用 video generation 作为 generalizable simulator。

### 8.3 Multi-agent video world model (concurrent)
- Solaris [Savva et al.]：two-player Minecraft，fixed agent/view count，interleave view in sequence dimension。无法 scale。
- MultiVerse [Enigma team]：two-player racing game
- COMBO [Zhang et al.]：compositional，combine single-agent models，ignore inter-agent interaction

MultiWorld 相对这些的核心优势：
1. Variable $K$ 和 $C$（Solaris 限制 $K=2, C=2$）
2. Explicit inter-agent interaction modeling (AIE + self-attention)
3. 3D-aware global state (VGGT) 保证 view consistency
4. Parallel generation 优于 sequential interleave

### 8.4 3D foundation model 在 video generation 中的应用
- Geometry Forcing [Wu et al.]：marry video diffusion 和 3D
- ViewCrafter [Yu et al.]：novel view synthesis via video diffusion
- GAIA-2 [Russell et al.]：multi-view autonomous driving
- Tesseract [Zhen et al.]：4D embodied world model
- VGGT [Wang et al.]：3D foundation model，MultiWorld 用其 latent

MultiWorld 的 insight 是：**不需要显式 reconstruct 3D，用 VGGT 的 latent 作为 condition 就够**。这与 NeRF-based world model 思路不同，更轻量。

---

## 9. 我的 Intuition 与 Critical Thoughts

### 9.1 为什么 AIE 用 RoPE 而不是 learnable embedding？

Karpathy 你应该会问这个。Learnable agent embedding 是最直接的做法：给每个 agent 一个 $D$ 维 learnable vector，加到 action embedding 上。问题：
1. **Cannot extrapolate**：训练时 $K=2$，推理时 $K=4$，agent 3, 4 没有 learnable embedding
2. **Data inefficient**：每个 agent 位置需要独立学习，无 sharing

RoPE 的 relative encoding 解决了这两个问题：
1. 任意 agent index 都有 well-defined rotation
2. 相对关系 $n-m$ 在训练中可能见过（agent 1 vs 2 的 relative identity 1 = agent 2 vs 3 的 relative identity 1），可以 transfer

这是一个非常 elegant 的 transfer learning from LLM to multi-agent modeling 的例子。

### 9.2 为什么 GSE 用 VGGT 而不是显式 3D reconstruction？

显式 reconstruct point cloud 然后 render 是最 direct 的 multi-view consistency 方案。问题：
1. **Differentiability**：reconstruction→rendering pipeline 不一定 differentiable
2. **Computation**：3D reconstruction expensive
3. **Lossy**：reconstruction 会丢 detail，render 回来不 fidelity

VGGT latent 是 differentiable、compact、3D-aware 的。它 encode 了 multi-view geometry prior 但不强制 explicit reconstruction。这给 video model 足够 flexibility 同时保证 consistency。

### 9.3 Concerns 与 limitations

1. **Scale**：论文 limitation section 提到"current scale is still limited, large-scale training unexplored"。在 Wan2.2-5B 上 40k iteration 可能不足以 fully exploit backbone capacity
2. **VGGT frozen**：可能 limit adaptability 到 domain shift 大的场景。但 fine-tune VGGT expensive
3. **Memory for long-horizon**：论文 future work 提到 memory mechanism 需求。当前 autoregressive 只用最后一帧 update global state，long horizon 可能有 drift
4. **Failure case (Figure 9)**：distant/small object 模糊。这是 resolution 限制 + 3D prior 在 small region 上 insufficient 的 combined effect
5. **K and C independence**：论文声称 $K$ 和 $C$ 独立。但实际上 multi-agent 场景每个 agent 通常对应一个 view（如 It Takes Two 是 split-screen，每个 player 一个 view）。当 $K \neq C$ 时（如 multi-robot 多 camera）是否仍有 well-defined semantics？论文处理了这种 decoupling 但 discussion 不深

### 9.4 Connection 到你 (Karpathy) 的 world model 思路

你最近经常讨论 world model 的 property：physical consistency、long-horizon coherence、action controllability。MultiWorld 在这些基础上加了 **multi-agent interaction** 和 **multi-view consistency** 两个新维度。

你提过 world model 应该是 "model that can predict future given action"。MultiWorld 把这扩展为 "model that can predict multi-view future given multi-agent joint action"。这更接近真实 environment 的 complexity。

你提过 LLM 是 implicit world model (text-based)。Video world model 是另一条路 (vision-based)。MultiWorld 走 vision-based，但用 VGGT 引入 3D prior，相当于在 implicit visual world model 中嵌入了 explicit 3D structure。这是一个 hybrid approach，可能 future direction。

### 9.5 推测的扩展方向

1. **Multi-agent RL with world model**：用 MultiWorld 作为 model-based RL 的 environment，train multi-agent policy。比 physical simulator 快
2. **VLA training data augmentation**：用 MultiWorld 生成 failure trajectory，augment VLA 训练数据
3. **Real-time multi-agent**：论文 future work 提到。需要 one-step distillation (Yin et al. 2023) 或 causal forcing (Zhu et al. 2026)
4. **Scaling up K and C**：当前 $K \leq 4, C \leq 2$。能否 scale 到 $K=10, C=10$？RoPE extrapolation 和 VGGT view count 都是 unknown factor
5. **Multi-modal multi-agent**：加入 language communication between agents。当前 action 只有 motor command
6. **VGGT fine-tuning**：joint train VGGT + DiT，可能 better domain adaptation 但 expensive
7. **Memory mechanism**：long-horizon 多 agent simulation 需要 episodic memory。WorldMem、InfiniteWorld 的思路可借鉴

---

## 10. 总结

MultiWorld 是一个 elegant 的 multi-agent multi-view video world model 框架。核心 contributions：

1. **MACM**：用 RoPE 在 agent dimension 打破 permutation symmetry + AAW 自适应聚合 action。简单但 effective
2. **GSE**：用 frozen VGGT latent 作为 3D-aware global state，跨 view conditioning。Transfer learning 的精彩应用
3. **Scalability**：$K$ 和 $C$ 都 decouple，支持 parallel generation 和 autoregressive long-horizon

实验在 multi-player game 和 multi-robot manipulation 两个 domain 都验证了有效性。Ablation 充分，design choice 有 justification（如 base=20 vs 10k）。

我的整体感受：这是一个 solid 的方法 paper，insight 在于把 LLM 的 RoPE 和 3D vision 的 VGGT 用到 video world model 的两个核心 challenge 上。Technical contribution 不算 revolutionary 但组合得很巧妙。真正的 impact 取决于 scale 能否推到更大——如果在 10B+ parameter、10000+ hour data 上验证，可能成为 multi-agent world model 的标准 framework。

参考链接汇总：
- VGGT: https://vggt.github.io/
- Wan2.2: https://github.com/Wan-Video/Wan2.2
- RoPE (RoFormer): https://arxiv.org/abs/2104.09864
- Flow Matching: https://arxiv.org/abs/2210.02747
- DiT: https://arxiv.org/abs/2212.09748
- DROID-SLAM: https://github.com/princeton-vl/DROID-SLAM
- RoboFactory: https://arxiv.org/abs/2503.16408
- COMBO: https://arxiv.org/abs/2404.10775
- Genie 3: https://deepmind.google/discover/blog/genie-3
- Oasis: https://oasis-model.github.io/
- GameFactory: https://arxiv.org/abs/2503.07699
- The Matrix: https://arxiv.org/abs/2412.03568
- WorldMem: https://arxiv.org/abs/2504.12369
- InfiniteWorld: https://arxiv.org/abs/2602.02393
- Geometry Forcing: https://arxiv.org/abs/2507.07982
- ViewCrafter: https://arxiv.org/abs/2409.02048
- VPT (Video PreTraining): https://arxiv.org/abs/2206.11695

Karpathy，希望这个 deep dive 帮你 build intuition on this paper。如果哪里需要更深 dive（比如 VGGT 内部 architecture、RoPE 的 continuous extrapolation property、DROID-SLAM 的 DBA 算法），可以继续问。
