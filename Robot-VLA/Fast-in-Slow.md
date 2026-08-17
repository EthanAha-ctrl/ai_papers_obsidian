---
source_pdf: Fast-in-Slow.pdf
paper_sha256: d7c25db92db94c68e55d43a86f3bfdd897bb3a7e667287341e499bbe9522e879
processed_at: '2026-08-04T07:11:07-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# FiS-VLA 人话版

Andrej，我换一种讲法，像在 whiteboard 前面聊天的感觉。

---

## 这篇 paper 在解决什么实际问题

你做一个 robot policy，用 VLM 当 backbone（比如 7B 的 LLaMA2 + SigLIP + DINOv2），你会发现一个很尴尬的事情：**模型很聪明，但手很慢**。

VLM 每次要 forward 整个 7B transformer，autoregressive 一个 token 一个 token 吐 action，跑下来可能 6-10 Hz。Robot 闭环 control 要 30-100 Hz 才能做 smooth manipulation，这个 gap 很致命。

社区之前想了个办法：**拆成两个 system**。System 2 是 VLM，慢但聪明，负责理解 scene 和 reasoning；System 1 是一个小 policy head，快但 dumb，负责实时出 action。典型的像 CogACT、π₀、HiRT。

这个方案的问题是：**System 1 是一个新加的 module，从零开始训练**。它只从 System 2 拿一个 latent vector 当 condition，自己没有 internet-scale pretraining。相当于一个博士大脑指挥一个婴儿手脚，手脚只能听懂一句简短指令。

FiS 的核心 claim 很简单：**不要新加 module，把 VLM 自己的最后几个 transformer block 拿出来当 System 1**。这样 System 1 直接 inherit 了 VLM 的所有 pretrained weight，它和 System 2 共享参数，本质上就是大脑自己的运动皮层，不是外接的假肢。

---

## 为什么这个想法 work

### Intuition 1：VLM 后几层本来就有 action-relevant representation

你 fine-tune 过 LLM 就知道，最后几层 transformer block 已经编码了大量 task-relevant feature。FiS 做 ablation 发现：**只用最后 2 个 block 当 System 1，性能最好（69%）；用 1 个 block 不够（49%），用 4 个反而下降（66%）**。

这个 2-block sweet spot 很有意思。说明什么？VLM 的 representation 在中间层已经成熟，最后 2 层做的就是 "formatting"——把 high-level representation 转成 task-specific output。你拿这 2 层做 action decoding，它已经有足够 context，只需要 minimal adaptation 就能从 "language token prediction" 切到 "diffusion action generation"。

用 4 个 block 反而变差，这个现象也值得想。可能是过多 transformer 容量 overfit 到 action generation task，反而破坏了对 System 2 latent 的 interpretation 能力。也可能是参数太多 diffusion 优化 landscape 变复杂。

### Intuition 2：Bottleneck 问题被消除了

Prior dual-system 方法的信息流：
```
VLM (32 blocks) → 1 个 latent vector → Policy Head (新 module)
```

那个 latent vector 是 bandwidth bottleneck。System 2 看到 scene 理解了一堆东西，最终只能通过一个 vector 传给 System 1。

FiS 的信息流：
```
VLM (32 blocks，其中前 30 是 System 2，后 2 是 System 1，共享参数)
    ↓
System 1 直接 access System 2 的所有中间 representation
```

System 1 不是外接的，它就是 System 2 的延伸。System 2 的 layer 30 输出直接喂给 System 1 的 layer 31，中间没有 bottleneck。

---

## Asynchronous frequency 这件事

这是 paper 的第二个核心 idea，我之前讲得有点抽象，换个说法。

你在开车。大部分时候你在做 reflexive 操作：方向盘微调、油门控制，这些是 System 1，high frequency。但偶尔你需要 reasoning：前面路口要不要转弯？这个是 System 2，low frequency。

你不能每 100ms 都做一次完整 reasoning，太慢；也不能永远不 reasoning，会撞车。所以自然的做法是：**reasoning 一次，然后 trust 这个 reasoning 一段时间，期间用 reflex 快速执行**。

FiS 做的就是这件事。System 2 每 4 个 control step 跑一次，输出一个 latent；System 1 每个step跑一次，用那个 stale 的 latent + 当前 observation 出 action。

Ablation 发现 **1:4 是 sweet spot**：
- 1:1 太频繁，System 2 成了 bottleneck，没有 reactive 自由度
- 1:8 太稀疏，latent stale，System 1 不能 compensate scene 变化
- 1:4 刚好——4 个 step 内 scene 变化在容忍范围内，System 1 用 fresh observation 补上

这个 4 步窗口其实就是 "VLM 的理解能管多远" 的经验度量。它告诉你：对于 tabletop manipulation 这种 task，4 个 control step（大概 100-200ms）内 VLM 的 scene understanding 还 valid，超过这个就 stale 了。

---

## Heterogeneous modality 为什么重要

System 2 和 System 1 给不同的 input，这件事看似工程 trick，其实有 deep reason。

System 2 是 VLM，它在 internet image-text pair 上 pretrain。给它 image + language，它能 leverage 所有 commonsense reasoning：这个物体是杯子，杯子用来装水，task 是倒水，所以应该拿起杯子倾斜。这是 semantic-level 理解。

System 1 要出 action，它需要什么？需要知道杯子的精确 3D 位置（point cloud）、当前 gripper 在哪（robot state）、杯子相对 gripper 的几何关系（point cloud + image）。这是 geometric-level 信息。

如果你把 3D point cloud 塞给 System 2，它没用——VLM 没在 3D 上 pretrain 过，反而会 dilute 它的 semantic reasoning 能力。如果你只给 System 1 semantic latent 不给 3D，它抓不到东西——semantic 知道是杯子，但不知道杯子精确在哪。

所以分工是：
- **System 2**：semantic understanding（image + text）→ 输出 high-level latent
- **System 1**：geometric + state info（3D + image + robot state）+ System 2 latent → 输出 action

Ablation 印证：去掉 point cloud 掉 8%，再去掉 image 掉 17%，再去掉 state 掉 22%。每个 modality 都 essential，组合起来效果最好。

---

## Training 怎么做

这里有个 tricky 的点：你同一个 model 同时要学两件事。

**System 1（最后 2 个 block）要学 diffusion action generation**。给一个 clean action $\tilde{a}$，加噪 $\eta$，让 model predict noise。Loss 是 MSE：
$$\mathcal{L}_{\text{fast}} = \mathbb{E}[\| \eta - \pi_{\theta_f}(\text{noised action}, c, \tau) \|^2]$$

$c$ 是 condition，包括 System 2 latent + System 1 high-freq input。$\tau$ 是 diffusion timestep。

**System 2（整个 32 block）要保留 autoregressive reasoning**。给 image + text，预测下一个 token（discrete action token 或 language plan）。Loss 是 cross-entropy：
$$\mathcal{L}_{\text{slow}} = -\sum \log P(\hat{a}_i \mid \text{context})$$

总 loss = $\mathcal{L}_{\text{fast}} + \mathcal{L}_{\text{slow}}$，同时 backprop。

为什么要 co-training？如果你只训 $\mathcal{L}_{\text{fast}}$，最后 2 个 block 会 catastrophically forget 它原来做 language token prediction 的能力，System 2 整体 reasoning 退化，给 System 1 的 latent 也变 garbage。Ablation 验证：去掉 $\mathcal{L}_{\text{slow}}$，性能从 69% 掉到 62%。

这个 co-training 本质上是个 multi-task regularization。Diffusion 和 next-token prediction 两个 loss 在共享参数上互相约束，防止任何一个 task 把 representation 拉偏。

---

## 实验结果怎么样

### Simulation（RLBench，10 个 task）

FiS-VLA 69% mean success rate，比 CogACT（61%）高 8%，比 π₀（55%）高 14%。

Inference speed：21.9 Hz（action chunk = 1），比 CogACT（9.8 Hz）快 2 倍多。注意 FiS-VLA 是 7B，π₀ 是 2.6B，FiS 既大又快。

### Real-world dual-arm

Agilex 机器人：68% vs π₀ 59%
AlphaBot 机器人：74% vs π₀ 61%

最大提升在 Fold towel（deformable object），这个 task 需要精细 force control 和 geometric reasoning，FiS 的 3D input + VLM pretrained knowledge 优势明显。

### Generalization

换了 unseen object、加复杂背景、变光照，FiS-VLA 的 performance drop 都比 π₀ 小很多。比如 AlphaBot 上换 unseen object，FiS 掉 19%，π₀ 掉 38%。

这说明什么？**System 1 inherit 了 VLM 的 pretrained knowledge，对 perception perturbation 更 robust**。VLM 在 internet-scale data 上见过各种 object、background、lighting，这些 commonsense 通过共享参数 transfer 到了 System 1。

### Control frequency

Action chunk = 8 时达到 **117.7 Hz**。这个数字是理论值，实际 robot hardware 可能有 latency，但说明 model 本身不是 bottleneck。

---

## 我觉得这篇 paper 的 limitation

1. **System 1 的"快"主要来自 action chunking**。System 1 每次还是要 forward 2 个 transformer block + diffusion（T=100 step，实际可能 10-50 step）。真正让它快的是 H=8 一次吐 8 个 action，amortize 了 inference cost。如果 H=1 只有 21.9 Hz，说明 single-step inference 本身还是不便宜。

2. **Frequency ratio 是 static 的**。作者自己承认这是 limitation。理想情况 System 2 应该 adaptive：scene 变化大时多 reasoning，stable 时少 reasoning。这个 dynamic scheduling 是 obvious next step。

3. **Point cloud tokenizer 用 shared vision encoder 处理 3D 信息**，这个设计 elegant 但可疑。SigLIP/DINOv2 是 2D-pretrained，对 tokenize 后的 3D token 能不能真正 encode geometric structure？还是只是当作 "奇怪的 image patch" 处理？这个需要更多 ablation。

4. **Real-world 数据规模小**，每个 task 100 demos。Generalization 实验里 disturbed 场景掉到 50-65%，离 robust deployment 还有距离。

5. **没有和 HybridVLA 直接比较**。HybridVLA [32] 也是 unified diffusion + autoregression，思路接近，缺这个 comparison 让人有点不踏实。

---

## 我从这篇 paper 学到的 intuition

1. **Pretrained model 的后几层已经接近 task-relevant representation**。你不需要新加 module，只需要 repurpose 现有 layer。这个 insight 在 LLM fine-tuning 里也适用。

2. **Dual-system 的核心是 amortized computation**。一次昂贵推理 amortize 到 H 个 cheap action step，H 的选择取决于 latent 的 "validity horizon"。

3. **Modality 分配应该 match task symmetry**。Reasoning system 和 execution system 需要的信息类型不同，强行统一 input 会浪费 capacity。

4. **Co-training 防止 forgetting 比冻结参数更优雅**。冻结 System 2 会限制 adaptation，co-training 让两个 loss 互相 regularize。

---

## 一句话总结

FiS-VLA 告诉你：**别给 VLM 外接一个 dumb policy head 当手脚，把 VLM 自己的后几层 transform 成手脚，让它自己指挥自己**。这个 sharing 参数的简单 trick 消除了 representation bottleneck，让 robot policy 同时享受 VLM 的 commonsense 和 fast execution。

项目主页：https://fast-in-slow.github.io

如果你想 follow up，我觉得最有意思的方向是 **dynamic frequency scheduling** 和 **让 System 2 输出 structured plan 而不是 latent**。前者让 system 自适应 scene complexity，后者让 reasoning 和 execution 的接口从 black-box latent 变成 interpretable plan，可能会 unlock 更好的 long-horizon task 性能。

---

# Fast-in-Slow (FiS-VLA) 深度技术讲解

Andrej，这篇 paper 我从 motivation → architecture → training → experiments 一路拆开讲，重点放在它和 prior dual-system VLA 的本质区别，以及为什么 "embedding System 1 inside System 2" 这件事不只是一个工程 trick，而是触及了 VLA 里面 representation bottleneck 的核心问题。

---

## 1. Background：为什么需要 Fast-in-Slow

### 1.1 Kahneman dual-system theory 的工程映射

Kahneman 在《Thinking, Fast and Slow》[21] 里提出 human cognition 有两套 system：
- **System 1**：fast、intuitive、unconscious，对应 reflexive control
- **System 2**：slow、logical、deliberate reasoning，对应 high-level planning

VLA 社区把这个 mapping 直接搬到 robot 上：System 2 用 VLM 做 high-level scene understanding 和 task reasoning，System 1 用一个 lightweight action head 做 low-level continuous control。典型代表如 **CogACT** [22]、**π₀** [23]、**HybridVLA** [32]、**Helix** [25]、**HiRT** [1]、**DexVLA** [62]。

### 1.2 Prior dual-system VLA 的痛点

现有 dual-system 方法（HiRT、Helix、DexVLA 等）的结构是：

```
VLM (System 2) ──latent feature──►  Policy Head (System 1, 新加的 module)
```

这里有一个根本问题：System 1 是一个 **freshly initialized** 的 policy head（diffusion U-Net 或者 small transformer），它只从 System 2 拿一个 frozen latent vector 作为 condition。System 1 本身 **没有任何 internet-scale pretraining**，它的 visual-textual commonsense 全部来自 System 2 传过来的那个 bottleneck feature。

这相当于：你有一个博士大脑（System 2），但手脚是一个婴儿（System 1），手脚只能听大脑一句简短的指令，无法复用大脑的 rich representation。

### 1.3 FiS 的核心 insight

FiS 提出的问题是："If a VLM model serves as the 'brain' of the robot, can it integrate System 1 and System 2 processes to enable coordinated reasoning and execution?"

具体做法：**把 LLaMA2-7B 的最后 2 个 transformer blocks repurpose 成 System 1**，让它 **复用 VLM 的所有 pretrained weight**，而不是新加一个 head。System 1 仍然走 diffusion action generation，但 diffusion denoiser 用的就是 VLM 的最后几层 transformer。

这个设计在 neuroscience 上对应 [26, 27] 的 dual-process cognition 研究——human brain 里面 System 1 和 System 2 共享大量 neural substrate，没有严格的解剖隔离。

参考链接：
- Kahneman book: https://www.goodreads.com/book/show/11468377-thinking-fast-and-slow
- CogACT: https://arxiv.org/abs/2411.19650
- π₀: https://arxiv.org/abs/2410.24164
- Helix: https://www.figure.ai/news/helix
- HiRT: https://arxiv.org/abs/2410.05273
- HybridVLA: https://arxiv.org/abs/2503.10631

---

## 2. Architecture 详细拆解

### 2.1 整体结构（对应 Figure 2）

FiS-VLA 基于 **Prismatic VLM** [16] 作为 base：

| Component | 具体选择 | 维度 |
|---|---|---|
| Vision encoder (2D) | SigLIP [64] + DINOv2 [65] | $f^{\text{SigLIP}} \in \mathbb{R}^{N_v \times 1024}$, $f^{\text{DINO}} \in \mathbb{R}^{N_v \times 1152}$ |
| Point cloud tokenizer | 3-block FPS + kNN + linear [66] | 输出 token 进入 shared vision encoder |
| LLM backbone | LLaMA2-7B [68] | 32 transformer blocks |
| System 2 | 完整 32-block LLM | 处理 language + 2D image |
| System 1 | 最后 2 个 transformer blocks | 处理 3D point cloud + 2D image + robot state，输出 diffusion action |
| Projector / state encoder / timestep MLP / noise MLP | MLP | 轻量辅助 |

### 2.2 Vision encoder 双分支

每张 image resize 到 224×224，过两个 encoder：

- **SigLIP**：提供 high-level semantic feature（language-aligned）
- **DINOv2**：提供 local spatial detail（self-supervised geometry-aware）

两个 feature 在 channel 维度 concat，得到 unified visual embedding。这是 Prismatic 的标准做法，FiS 直接继承。

### 2.3 Point cloud tokenizer（关键设计）

FiS 没有像 3D Diffusion Policy [28] 或 PointVLA [56] 那样新加一个 PointNet++ 之类 3D encoder，而是用 **3D tokenizer** [66] 把 point cloud 转成 high-dimensional tokens，然后 **送进 shared vision encoder**（也就是和 2D image 共享 SigLIP+DINOv2）。

3D tokenizer 的结构：
- 3 个 block 串联
- 每个 block 包含：
  - **FPS (Farthest Point Sampling)** [67]：downsample point cloud
  - **kNN (k-Nearest Neighbors)**：local aggregation
  - **learnable linear layer**：feature encoding

得到 token 之后送进 vision encoder 提取 spatial feature，这样 point cloud 的 representation 直接落在 LLM 的 embedding space 里面。

这个设计的 motivation：
1. 复用 VLM vision encoder 的 vision-language alignment 能力，把 3D 信息 project 进 language-aligned space
2. 避免引入大量新参数，保持 computation efficiency

公式描述 point cloud 输入：
$$\mathcal{P} = \{\mathbf{p}_i \in \mathbb{R}^3\}_{i=1}^{N_p}$$

其中 $\mathbf{p}_i$ 是第 $i$ 个点的 3D 坐标 $(x, y, z)$，$N_p$ 是点数（论文中 downsample 到 1024 points）。

### 2.4 System 2 (Slow) 信息流

System 2 的输入：
- **Language instruction** $l$
- **2D image observation** $o^{2D}_{t-1}$

输出：
- LLM 中间某个 block 的 latent feature $h^{\text{slow}}_t$，作为 System 1 的 condition
- 同时也做 autoregressive next-token prediction（discrete action token 或 sub-task plan）

### 2.5 System 1 (Fast) 信息流

System 1 的输入（注意这是 heterogeneous modality 设计）：
- **2D image**（high-frequency，每 step 更新）
- **3D point cloud**（从 depth map 通过 camera intrinsics/extrinsics 反投影得到）
- **Robot proprioceptive state**
- **System 2 的 latent feature** $h^{\text{slow}}_t$（低频更新，跨 H step 复用）
- **Diffusion timestep** $\tau$（diffusion process 内部使用）
- **Noised action** $\tilde{a}_\tau$（diffusion process 内部使用）

输出：clean action $\hat{a} \in \mathbb{R}^d$，其中 $d$ 是 action 维度。

注意 System 1 用的是 **LLaMA2 最后 2 个 transformer blocks**，参数和 System 2 共享。System 2 走完整 32 blocks，System 1 走 last 2 blocks，但 last 2 blocks 的输入同时受 System 2 中间层 latent 和 System 1 自己的 high-frequency input 影响。

### 2.6 控制信号维度

不同 robot 的 action space：
- **Franka Panda (simulation)**: 7-DoF end-effector pose = $\Delta x, \Delta y, \Delta z$（3-DoF position offset）+ Euler angles（3-DoF rotation）+ gripper open/close（1-DoF）
- **Agilex (real-world)**: 14-DoF dual-arm end-effector pose
- **AlphaBot (real-world)**: 16-DoF dual-arm joint position

---

## 3. Dual-System Coordination

### 3.1 Asynchronous frequency design

System 2 由于是 billion-scale VLM，跑得慢；System 1 只有 2 个 transformer block + diffusion，跑得快。

论文核心问题："How many future action steps can be effectively guided by the intermediate comprehension output from System 2?"

设 System 2 : System 1 频率比为 $1:n$。在 $t$ 时刻：
1. System 2 跑一次，输出 $h^{\text{slow}}_t$
2. 接下来 $H = n$ 个 step，System 1 跑 $n$ 次，每次用 $h^{\text{slow}}_t$ 作为 condition，但接收最新的 2D image / 3D point cloud / robot state

Ablation 测了 $1:1, 1:2, 1:4, 1:8$，最佳是 $1:4$（见 Table 8）。

为什么 $1:8$ 反而变差？因为 System 2 的 latent 在 8 步之后 stale，scene 已经变化太多，System 1 无法 compensate。为什么 $1:1$ 也不行？因为 System 2 跑太频繁反而成了 bottleneck，且没有给 System 1 足够的 reactive 自由度。

### 3.2 Heterogeneous modality input

这是 paper 的一个 subtle 但 important 的设计 choice：

**System 2 收到的 modality**：
- Language instruction
- 2D image

理由：VLM 在 internet-scale image-text pair 上 pretrain，给它 image+text 最能 leverage 它的 high-level semantic reasoning。

**System 1 收到的 modality**：
- 2D image（实时）
- 3D point cloud（实时）
- Robot proprioceptive state（实时）
- System 2 的 latent feature（低频）

理由：System 1 要做 precise manipulation，需要 spatial geometry（3D）+ 当前 robot state（闭环 control 需要）+ 实时 visual feedback。

Ablation（Table 7）：
- 完整 input：69% mean S.R.
- 去掉 point cloud (No PC)：61%
- 去掉 PC 和 2D image：44%
- 去掉 PC, image, state：22%

可以看出 3D point cloud 贡献了 8% 的提升，2D image 又贡献 17%，robot state 又贡献 22%。每个 modality 都 essential。

### 3.3 Asynchronous training sampling

为了让 System 1 学会容忍 System 2 latent 的 staleness，训练时对 System 2 做 **asynchronous sampling**——System 2 不每个 step 都更新，而是每隔 $H$ step 才 sample 一次。这 force System 1 学会基于 stale condition + 当前 observation 生成 action。

---

## 4. Training Objective 细节

### 4.1 总目标

$$\mathcal{L}_{\text{FiS-VLA}} = \mathcal{L}_{\text{fast}} + \mathcal{L}_{\text{slow}}$$

两个 loss 同时优化整个 model（System 2 的 32 blocks 和 System 1 的 last 2 blocks 共享参数）。

### 4.2 System 1 diffusion loss

前向加噪过程：
$$\tilde{a}_\tau = \sqrt{\beta_\tau} \tilde{a} + \sqrt{1 - \beta_\tau} \eta$$

变量解释：
- $\tilde{a}$：clean action sequence
- $\eta \sim \mathcal{N}(0, I)$：标准 Gaussian noise
- $\tau \sim \mathcal{U}(1, T)$：uniform 采样的 diffusion timestep，$\tau \in \mathbb{Z}$，$T = 100$
- $\beta_\tau$：noise scaling factor，按 predefined schedule [70]（cosine 或 linear）
- $\tilde{a}_\tau$：加噪后的 noisy action

Training loss（predict noise 而非 predict clean action）：
$$\mathcal{L}_{\text{fast}} = \mathbb{E}_{\tau, c, \tilde{a}, \eta} \left[ \| \eta - \pi_{\theta_f}(\sqrt{\beta_\tau} \tilde{a} + \sqrt{1 - \beta_\tau} \eta, c, \tau) \|^2 \right]$$

变量解释：
- $\pi_{\theta_f}$：System 1 的 denoiser network，参数 $\theta_f$ 是 LLaMA2 最后 2 个 transformer blocks + 投影 MLP 的参数
- $c$：conditioning source，包含两部分：
  - 低频 System 2 latent feature $h^{\text{slow}}_t$
  - 高频 System 1 input（2D image + 3D point cloud + robot state）
- $\theta_f \subseteq \theta$：System 1 参数是整个 model 参数的子集

### 4.3 System 2 autoregressive loss

$$\mathcal{L}_{\text{slow}} = -\sum_{i=1}^{D_t} \log P(\hat{a}_i \mid \text{context}, \theta)$$

变量解释：
- $D_t$：discrete action token sequence 总长度
- $\hat{a}_i$：第 $i$ 个 ground-truth action token（discrete，比如 bin 后的 action token）
- $P(\hat{a}_i \mid \text{context}, \theta)$：LLM 在给定 context 和参数 $\theta$ 下预测第 $i$ 个 token 的概率
- context：包括 image embedding、language instruction、前面已生成的 token

**为什么必须有 $\mathcal{L}_{\text{slow}}$**：如果只训练 $\mathcal{L}_{\text{fast}}$，System 1 的 diffusion objective 会 catastrophically forget System 2 的 autoregressive reasoning capability。Ablation 显示去掉 $\mathcal{L}_{\text{slow}}$ 后 mean S.R. 从 69% 掉到 62%（降 7%）。

### 4.4 Pretraining → Fine-tuning recipe

**Pretraining**：
- Dataset：Open X-Embodiment [19] + DROID [20] + ROBOMIND [34] + 30+ 其他 dataset，共 860K trajectories，36M frames（详见 Table 4）
- 5 epochs
- System 2 用 discrete action sequence 监督（因为 pretraining data 没有 sub-goal language）
- System 1 用 diffusion action generation
- 单 image 作为 observation

**Fine-tuning**：
- 加入 manually annotated sub-task language plan 给 System 2 做 language supervision
- 加入 self-collected RLBench + real-world dual-arm data
- 多 view image + 3D point cloud

---

## 5. Experiments 详解

### 5.1 RLBench 仿真实验（Table 1）

10 个 task：Close box, Close laptop lid, Toilet seat down, Sweep to dustpan, Close fridge, Phone on base, Umbrella out, Frame off hanger, Wine at rack, Water plants

Baseline：
- ManipLLM [50]
- OpenVLA [7]
- π₀ [23]（2.6B LLM，dual-system synchronous）
- CogACT [22]（7B LLM，dual-system synchronous）

FiS-VLA 结果：**mean S.R. 69%**，比 CogACT（61%）高 8%，比 π₀（55%）高 14%。

Inference speed（NVIDIA 4090, action chunk = 1）：
- ManipLLM: 2.2 Hz
- OpenVLA: 6.3 Hz
- π₀: 13.8 Hz
- CogACT: 9.8 Hz
- **FiS-VLA: 21.9 Hz**

注意 FiS-VLA 是 7B 模型，π₀ 是 2.6B 模型，FiS-VLA 既大又快，验证了 asynchronous frequency design 的有效性。

### 5.2 真实世界双臂实验（Table 2）

**Agilex Robot**（14-DoF end-effector pose control，3 个 camera view）：
- Pick objects and place in basket
- Lift ball and place in basket
- Place bottles at rack
- Wipe blackboard

FiS-VLA mean S.R. 68% vs π₀ 59%（高 9%）

**AlphaBot**（16-DoF joint position control，3 个 camera view）：
- Pick bowl and place object
- Handover object and place
- Pour water and move cup
- Fold towel and place in bucket

FiS-VLA mean S.R. 74% vs π₀ 61%（高 13%）

最大提升在 **Fold towel**（deformable object manipulation），FiS-VLA 显著优于 π₀。

### 5.3 Generalization 实验（Table 3）

三个干扰场景：
- **Unseen object**：替换 manipulated object（e.g., banana → hot dog bun）
- **Complex background**：引入 mug, hamburger, bottle 等无关物体
- **Varying lighting**：光照变化

结果（以 AlphaBot Pick bowl 为例）：
- Original：FiS-VLA 80%，π₀ 65%
- Object：FiS-VLA 65%（-19%），π₀ 40%（-38%）
- Background：FiS-VLA 60%（-25%），π₀ 40%（-38%）
- Lighting：FiS-VLA 55%（-31%），π₀ 35%（-46%）

FiS-VLA 在所有 generalization 场景下的 performance drop 都比 π₀ 小。这印证了 paper 的核心论点：**System 1 复用 VLM pretrained knowledge 后，对 perception perturbation 更 robust**。

---

## 6. Ablation 深度解读

### 6.1 System 1 共享 blocks 数量（Table 6）

固定 frequency ratio 1:4，all modality input，变化 System 1 用的 transformer block 数：

| Blocks | Mean S.R. |
|---|---|
| 1 | 49 ± 0.05 |
| **2 (FiS-VLA)** | **69 ± 0.03** |
| 4 | 66 ± 0.02 |
| 8 | 64 ± 0.03 |

最佳是 2 blocks。1 block 太少（容量不够 inherit pretrained knowledge），4 blocks 以上反而下降——可能是过多 transformer 容量 overfit 到 action generation，破坏了对 System 2 latent 的 interpretation。

这个 2-block sweet spot 给我们一个 insight：**VLM 的最后几层本身已经具备 action-relevant representation，只需要少量 adaptation 就能变成 efficient action decoder**。

### 6.2 Modality input ablation（Table 7）

| Config | Mean S.R. |
|---|---|
| FiS-VLA (full) | 69 ± 0.03 |
| No PC | 61 ± 0.02 |
| No PC, No 2D Img | 44 ± 0.03 |
| No PC, No Img, No State | 22 ± 0.05 |

去掉所有 high-frequency input 只用 System 2 latent 时，掉到 22%——这验证了 System 1 不能纯靠 System 2 latent 当 condition，必须有实时 observation。

### 6.3 Frequency ratio（Table 8）

| Ratio | Mean S.R. |
|---|---|
| 1:1 | 60 ± 0.02 |
| 1:2 | 63 ± 0.03 |
| **1:4** | **69 ± 0.03** |
| 1:8 | 61 ± 0.04 |

1:4 是 sweet spot。

### 6.4 Action chunk size（Table 9, Figure 5 left）

| Chunk H | Mean S.R. | Control freq |
|---|---|---|
| 1 | 69 ± 0.03 | 21.9 Hz |
| 2 | 68 ± 0.03 | ~44 Hz |
| 4 | 66 ± 0.04 | ~88 Hz |
| **8** | **69 ± 0.02** | **117.7 Hz** |

Performance 在 H=1~8 之间稳定，frequency 随 H 近似线性提升。H=8 时达到 **117.7 Hz** control frequency。

### 6.5 Input variant（Table 10）

Variant 1: System 2 = {lang, 2D, 3D}, System 1 = {2D, state} → 63%
Variant 2: System 2 = {lang, 2D, 3D, state}, System 1 = {2D} → 61%
Variant 3: System 1 = {2D, 3D, state}, System 2 = {lang, 2D, 3D} → 68%（接近 FiS-VLA）

结论：point cloud 给 System 1 用比给 System 2 用更重要，因为 System 1 需要精细 spatial 信息做 action；robot state 必须给 System 1 才能做闭环 control。

### 6.6 Training strategy

去掉 $\mathcal{L}_{\text{slow}}$：69% → 62%（降 7%）。这验证了 dual-aware co-training 的必要性——必须同时保持 System 2 的 reasoning capability，否则它给 System 1 的 latent guidance 会退化。

---

## 7. Failure Cases（Appendix D）

4 类典型失败（AlphaBot 平台）：
1. **Bimanual collision**（Handover task）：双臂 motion coordination 不足，wrist camera 位置 suboptimal
2. **Height error**（Fold towel）：thin deformable object 的 height prediction 困难
3. **Position error**（Pick bowl）：banana 位置 mispredict
4. **Handover rotation error**（Handover task）：rotation 预测错误导致 grasp 失败

作者承认当前 FiS-VLA 的 shared parameters 和 frequency ratio 是 **statically configured**，未来希望做 dynamic adaptation based on task complexity。

---

## 8. 我对这篇 paper 的 critical thoughts

### 8.1 强的地方

1. **Insight 准确**：System 1 应该 inherit System 2 的 pretrained knowledge，这个观点切中了 prior dual-system VLA 的痛点。
2. **Asynchronous frequency + heterogeneous modality** 的组合设计 elegant，让两个 system 各司其职。
3. **2-block sweet spot** 的 ablation 结果很有信息量，说明 VLM 后几层已经具备 action-relevant representation。
4. **Dual-aware co-training** 防止 catastrophic forgetting 的做法简洁有效。

### 8.2 可以质疑的地方

1. **System 1 的"快"本质上是共享 LLM 的 2 blocks**，但因为 System 2 的前 30 blocks 仍然要 forward（System 1 需要 System 2 的 latent），所以 inference 速度提升主要来自：
   - 不再 forward 整个 32 blocks（只走 2 blocks 做 diffusion）
   - Action chunking（H=8 时 117.7 Hz 是理论值，实际还要考虑 robot hardware latency）
   
   真正的 fast path 仍然是：System 2 偶尔跑一次 → System 1 每 step 跑 2 blocks + diffusion。System 1 没有跳过 System 2 前 30 blocks 的 forward，因为它需要 System 2 的 latent 作为 condition。**这意味着 System 1 的 inference 实际上仍是 conditioned on System 2 的最新 output**，速度优势主要来自 action chunk。

2. **Diffusion 用 2 个 LLaMA block 当 denoiser** 是否真的 efficient？Diffusion 需要 T=100 step 去噪（虽然实际可能用 10-50 step），每 step 都要 forward 2 个 transformer block。相比专门的 lightweight diffusion U-Net，参数量仍然偏大。

3. **Point cloud tokenizer 通过 shared vision encoder** 这个设计虽然 elegant，但 SigLIP/DINOv2 是 2D-pretrained encoder，对 3D 信息是否真能 encode 好？Tokenize 之后丢失了几何结构？这个需要更多 ablation 验证。

4. **Frequency ratio static** 是 limitation，作者自己也承认。理想情况下 System 2 应该在 task ambiguous 时主动 increase frequency（多 reasoning），在 stable execution 时 decrease frequency（让 System 1 free run）。这个 adaptive scheduling 是 obvious next step。

5. **Real-world 数据规模偏小**：每个 task 100 demonstrations。作者在 generalization 实验里展示了 robustness，但 absolute success rate 在 disturbed 场景下仍掉到 50-65%，距离 robust deployment 还有距离。

6. **缺少 explicit comparison with HybridVLA** [32]：HybridVLA 同样是 diffusion + autoregression 在一个 unified model 里，作者引用了但没有 baseline 比较。

### 8.3 联系到 broader VLA research

这篇 paper 让我想到几个相关工作：

- **HybridVLA** [32]：同样 unified diffusion + autoregression，但没有 dual-system 异步 frequency 设计
- **π₀.5** [61]：Physical Intelligence 的 open-world generalization VLA，也用 dual-system 思路
- **KIM (Knowledge Insulating VLA)** [63]：探讨 VLA train fast / run fast / generalize better 的 trade-off
- **Deer-VLA** [69]：dynamic inference of MLLM for efficient robot execution，思路接近 FiS 的 dynamic adaptation future work
- **RDT-1B** [31]：1B diffusion foundation model for bimanual manipulation，pure diffusion 没有 VLM reasoning
- **SpatialVLA** [29]：3D spatial representation for VLA，强调 3D 信息重要性，FiS 借鉴了这个 insight

参考链接：
- HybridVLA: https://arxiv.org/abs/2503.10631
- π₀.5: https://arxiv.org/abs/2505.07253（实际链接需查证）
- KIM: https://arxiv.org/abs/2505.23705
- Deer-VLA: https://arxiv.org/abs/2411.02359
- RDT-1B: https://arxiv.org/abs/2410.07864
- SpatialVLA: https://arxiv.org/abs/2501.15830
- OpenVLA: https://arxiv.org/abs/2406.09246
- Prismatic VLMs: https://arxiv.org/abs/2402.07865
- DROID: https://arxiv.org/abs/2403.12945
- Open X-Embodiment: https://arxiv.org/abs/2310.08864
- ROBOMIND: https://arxiv.org/abs/2412.13877
- Diffusion Policy: https://arxiv.org/abs/2307.01849 (实际 ID 需查证)
- 3D Diffusion Policy: https://arxiv.org/abs/2403.03954
- PointNet: https://arxiv.org/abs/1612.00593
- LLaMA2: https://arxiv.org/abs/2307.09288
- SigLIP: https://arxiv.org/abs/2303.01987
- DINOv2: https://arxiv.org/abs/2304.07193

---

## 9. 关键 intuition 总结

1. **Dual-system 的本质问题** 是 representation bottleneck：System 1 如果是新加的 module，它和 System 2 之间只有一个 latent vector 通道，信息带宽太窄。FiS 通过共享参数把这个 bottleneck 消除。

2. **VLM 后 2 层已经接近 action-relevant representation**，这个发现非常 Karpathy-style——它说明 large pretrained model 的 capacity 已经包含 task-relevant feature，我们只需要 minimal adaptation（2 个 block）就能 repurpose。

3. **Asynchronous frequency 的本质** 是 amortized computation：System 2 的 reasoning 在 H step 内有效，相当于把一次昂贵推理 amortize 到 H 个 cheap action step 上。这个 trade-off 在 1:4 处最优，说明 VLM 的 latent 在 4 个 control step 后开始 stale。

4. **Heterogeneous modality 的本质** 是任务对称性破缺：System 2 需要 semantic-level input（image+text），System 1 需要 geometric-level input（3D+state+2D）。强行让两个 system 接收同样 input 会造成 information overload 或者 underload。

5. **Dual-aware co-training 的本质** 是 multi-task learning 防止 forgetting：next-token prediction 和 diffusion denoising 在共享参数下互为正则化，loss 之间形成了 representation regularization。

---

希望这个 deep dive 帮你 build 起对 FiS-VLA 的 intuition。如果你想做 follow-up 工作，我觉得最 promising 的方向是 **dynamic frequency scheduling**（让 System 2 根据 scene complexity 自适应决定何时再 reasoning）和 **explicit hierarchical subgoal generation**（让 System 2 输出更结构化的 plan 而不仅仅是 latent feature 给 System 1）。这两个方向都能 push VLA 从 reactive policy 走向真正 agentic control。
