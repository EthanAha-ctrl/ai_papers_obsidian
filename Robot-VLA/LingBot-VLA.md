---
source_pdf: LingBot-VLA.pdf
paper_sha256: 1b0a361d6084d74afc0bc9fcdee5051375b701a8e41013460107a46902bd0426
processed_at: '2026-08-05T15:03:01-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话重讲一遍 LingBot-VLA

## 一句话版

有一帮人在 9 种双臂机器人上采了 2 万小时真机数据，用 Qwen2.5-VL 当 backbone 跑了一个 flow matching 的 action expert，共享 attention 分开 FFN，顺便把一个 depth model 的知识蒸馏进去，最后在 3 个平台 100 个任务上做了 22500 次真机评测，证明了一件事：real robot data 也能 scale，而且到 2 万小时还没饱和。

## 这事为什么有意思

Karpathy 你之前在特斯拉那阵子一直在问一个问题——机器人数据到底能不能像 LLM 的 token 一样越喂越多越涨。这件事 community 喊了两年，π0 真的训了 1 万小时，GR00T 训了 5 千小时，但没人把 scaling curve 真的画到 2 万小时这个量级。

LingBot 这篇 paper 的核心贡献就一句话：**他们把这条曲线画出来了，到 2 万小时还是直线，没看到拐点**。

这件事听起来简单，实际上极难。难的不是写模型，是后面那堆工程：你要采 2 万小时真机数据意味着几百个 teleop 员工干几个月，你要训完意味着几千张 H200 跑几周，你要评测意味着 25 台机器人 100 个任务每个跑 15 次还要 5 个 baseline 一起跑——总评测量 22500 次 trial。这件事在任何学术 lab 都做不动，只有像 Robbyant 这种公司级资源才能搞。

所以这篇 paper 的真正卖点根本不是模型架构创新，而是**有人真的把 foundation model playbook 在 real robot 上完整跑了一遍**。

## 架构怎么搭的

直觉上想象一个 VLA 模型，传统做法（RT-2、OpenVLA）是把 image token、text token、action token 全塞进一个 transformer 一起 next-token predict。问题是 image 是稀疏语义信号，action 是 50Hz 的连续控制信号，两个统计性质完全不一样，硬塞一起 FFN 会打架。

π0 的解法是 VLM 跑完之后单独接一个小 action transformer，相当于两个大脑分两个房间。

LingBot 选了一个中间路线，叫 **Mixture-of-Transformers**，来自 BAGEL [arXiv:2505.14683](https://arxiv.org/abs/2505.14683)。做法是 VLM 和 action expert 各自有自己的 FFN，但每一层共享同一个 self-attention。想象两个大脑共用一块白板，谁有想法写到白板上对方随时能看，但各自记笔记用自己的笔记本。这样 high-level 语义信息每一层都能 leak 进 action pathway，又不会因为 FFN 共享而 gradient 打架。

Action 用 **Flow Matching** 做生成，跟 π0 一样。Flow Matching 是 diffusion 的近亲，训练目标是预测从噪声到真值的 vector field，公式长这样：

$$\mathcal{L}_{\mathrm{FM}} = \mathbb{E}_{s, \mathbf{A}_t, \epsilon} \| v_\theta(\mathbf{A}_{t,s}, \mathbf{O}_t, s) - (\mathbf{A}_t - \epsilon) \|^2$$

$s$ 是 generative timestep 跟真实时间无关，$\mathbf{A}_{t,s} = s \cdot \mathbf{A}_t + (1-s) \cdot \epsilon$ 是噪声和真值的线性插值，$v_\theta$ 学的是从当前噪声点指向真值的"速度向量"。推理时用 10 步 Euler ODE 解出来就行，比 DDPM 50-1000 步快得多。

一个细节：action chunk length 设 50。如果控制频率 50Hz，50 个 action 就是 1 秒轨迹，刚好一个 atomic manipulation primitive 的时长。

另外一个有意思的设计是 **depth distillation**。VLM 视觉语义强但几何弱，LingBot 就把自己实验室的 LingBot-Depth 模型当 teacher，用 learnable query 加 cross-attention 把 depth 信息 soft-inject 到 VLM 表示里。关键 trick 是**只在训练时用 depth，推理时不需要 depth camera**。模型已经把 depth prior 内化进权重了。这对部署极友好——客户不用额外装 depth sensor。

## 数据这件事

9 种双臂机器人，包括 AgiBot G1、AgileX、Galaxea R1Lite/R1Pro、Realman Rs-02、Leju KUAVO、Qinglong humanoid、ARX Lift2、Bimanual Franka。DoF 从 12 到 16 不等，camera 配置从 3 个 RGB-D 到 1 stereo + 2 wrist 各种都有。

总数据量 2 万小时。作为对比，π0.5 大概 1 万小时级别，GR00T N1 大概 5 千小时。LingBot 的数据规模是目前公开报道里最大的。

annotation 用 Qwen3-VL-235B-A22B 自动写 instruction。用 235B MoE 而不是 GPT-4V，估计是中文场景和国产物体识别更靠谱，成本也低。

## 工程 infra 才是真正的杀手锏

2 万小时数据如果用 naive PyTorch DDP 训，估计要好几个月。他们重写了一个 codebase，在 8×H200 上跑到 **261 samples/s/GPU**。跟 OpenPI、StarVLA、Dexbotic 比，速度提升 1.5× 到 2.8×。

具体 trick 三招：

1. **FSDP2 + HSDP hybrid**：只对 action expert 模块做 shard group，VLM backbone 保持 replicate。action expert 参数少但 gradient 通信频繁，shard 后省通信；VLM 参数大但 freeze 或 low LR 微调，replicate 反而省事。
2. **Mixed precision**：reduction 用 FP32 保数值稳定，storage 和 communication 用 bfloat16 省显存带宽。bfloat16 的 8-bit exponent 跟 FP32 一样，action gradient 高频连续信号不会 overflow。
3. **FlexAttention + torch.compile**：blockwise causal attention 这种自定义 mask pattern 用 FlexAttention 编译成 fused kernel，避免 naive implementation 三步开销。

Fig.4 显示从 8 卡 scale 到 256 卡，throughput 几乎线性。通信 overhead 没爆炸。这件事对任何想 scale 的 lab 都是决定性的。

## 实验结果

GM-100 benchmark，3 个平台 100 个任务，每个任务 130 个 demo post-training，每个模型 15 trial。

平均 SR 数字：

| 模型 | Average SR |
|------|------------|
| WALL-OSS | 4.05% |
| GR00T N1.6 | 7.59% |
| π0.5 | 13.02% |
| LingBot w/o depth | 15.74% |
| **LingBot w/ depth** | **17.30%** |

17% 听起来低，但 100 个 detail-oriented task 平均下来，里面有大批 task 是所有方法都接近 0% 的 hard task（看 appendix Table S1-S6 里 #002、#006、#025 这种）。Foundation model scaling 的典型 pattern：不是把最难的 task 拉到能用，是把 medium difficulty task pool 扩大。

一个有意思的观察：**GR00T N1.6 在 Galaxea R1Pro 上反常**，SR 14.29% 跟 π0.5 持平，明显比它在另外两个平台好。原因是 GR00T 预训练数据里有大量 Galaxea R1Pro 数据。这条 observation 很重要——**cross-embodiment generalization 不是 free lunch，预训练数据的 embodiment overlap 是 strong prior**。所谓 universal robot foundation model 现阶段还是有 embodiment bias 的。

Simulation 上 RoboTwin 2.0 跑 50 个 task，LingBot w/ depth 在 randomized scene 下 86.68%，π0.5 是 76.76%，差 10 个点。Randomized 场景比 clean 场景提升更大，说明 depth prior 主要帮助 visual generalization，对 clutter 和光照变化鲁棒性贡献大。

## Scaling 和 data efficiency

Fig.5 是关键：pretraining 数据从 3000 小时到 20000 小时，SR 和 PS 单调上升，no saturation。三个平台的 individual curve 都跟 aggregate 一致。这是 community 第一次在 real robot 数据上看到 power-law-like scaling 持续到这个量级。

Fig.6 是 data efficiency：8 个 representative task 在 Agibot G1 上，LingBot 用 80 个 demo 比 π0.5 用 130 个 demo 还好。更重要的 trend——post-training data 越多，LingBot 跟 π0.5 的 gap 越大。预训练给了一个好 init，这个 init 在小数据下表现为先验帮忙，在大数据下表现为 exploration efficiency——模型知道在 data manifold 的哪一部分去 search，每一份新数据都更 informative。这就是 foundation model paradigm 的核心 selling point。

## 我的 take

这篇 paper 真正的卖点不是模型，是把整个 LLM playbook 搬到 real robot 上跑了一遍：

1. 大规模预训练数据（2 万小时）
2. 高效训练 infra（261 samples/s/GPU）
3. 严格 benchmark（22500 trial 真机评测）
4. 显示 scaling law 在 real robot 上成立

每一步单看都不新，组合起来 community 之前没人完整做过。LingBot 的工作相当于给后续所有 robot foundation model 工作 set 了一个 baseline——你如果说你的 model 更好，请在这个 benchmark、这个数据规模上跑一遍再说。

Limitations 也明显：全部 dual-arm、全部 tabletop、3 分钟 timeout、LingBot-Depth 没开源、VR teleop 数据可能有人类 motion prior bias。这些是后续工作的空间。

链接汇总：
- https://technology.robbyant.com/lingbot-vla
- https://github.com/robbyant/lingbot-vla
- https://huggingface.co/robbyant/lingbot-vla
- https://technology.robbyant.com/lingbot-depth
- https://arxiv.org/abs/2505.14683
- https://arxiv.org/abs/2210.02747
- https://arxiv.org/abs/2502.13923
- https://arxiv.org/abs/2503.14734
- https://arxiv.org/abs/2506.18088
- https://arxiv.org/abs/2508.02317

如果你想 build 更深的 intuition，我建议两件事：第一，仔细看 Fig.5 的 scaling curve，三条 platform 线是不是真的一致上升，这决定 scaling law 是不是 robust；第二，看 appendix Table S1-S6 里每个 task 的 per-model SR 分布，会看到 VLA 性能提升是 medium task pool 扩大，hard task 依然 0%——这告诉你当前 VLA 的能力 ceiling 在哪。剩下的所有架构选择都是为了让这个 scaling 能 efficient 跑起来的 supporting engineering。

---

# LingBot-VLA 深度解析：一个实用主义 VLA Foundation Model

## 一、论文核心叙事：为什么这篇paper重要

Karpathy 老师您最关心的 scaling law 问题，这篇 paper 给了第一个真实世界（real-world）的实证答案。之前的 VLA 工作（π0、π0.5、GR00T N1、RT-2 等）大多在几千小时级别的数据上做实验，scaling 曲线只画到几千小时就停了，到底 real-robot data 是否像 LLM 的 token 一样具有 power-law scaling，community 一直没有定论。LingBot-VLA 把这个曲线推到了 **20,000 小时**，并且明确报告：**no saturation observed**——这是 robotic foundation model 领域第一次在真实机器人数据上看到类似 chinchilla-style 的 favorable scaling 行为。

更难得的是，这篇 paper 的工程贡献跟科学贡献几乎同等重要：他们重新写了一个训练 codebase，在 8×H200 上做到了 **261 samples/s/GPU**，相对 OpenPI/StarVLA/Dexbotic 加速 1.5×~2.8×。这等价于把 20,000 小时数据的预训练周期从几个月压到几周——这是任何想做 real-world VLA scaling 的实验室绕不开的工程门槛。

官方资源：
- Project page: https://technology.robbyant.com/lingbot-vla
- Code: https://github.com/robbyant/lingbot-vla
- Checkpoints: https://huggingface.co/robbyant/lingbot-vla
- LingBot-Depth（配套 depth model）: https://technology.robbyant.com/lingbot-depth

---

## 二、Architecture 深度解析：Mixture-of-Transformers + Flow Matching

### 2.1 整体设计哲学：Modality Decoupling via Shared Attention

LingBot-VLA 的核心架构选择是 **Mixture-of-Transformers (MoT)**，灵感来自 BAGEL [arXiv:2505.14683](https://arxiv.org/abs/2505.14683)。直觉上可以这么理解：

- 传统 VLA（如 RT-2、OpenVLA）把 vision token、language token、action token 全部塞进同一个 transformer 里跑 next-token prediction 或者 diffusion。问题是 action 是高频连续信号（50Hz 控制），而 vision/language 是稀疏语义信号，两者统计特性差异极大，强行共享 FFN 会产生 **cross-modal interference**。
- π0 系列用了一个 trick：VLM backbone 跑完之后，action expert 单独接一个小 transformer 做 flow matching，相当于 "hard decoupling"。
- LingBot-VLA 选择 **soft decoupling**：VLM pathway 和 action pathway 走各自的 FFN/MLP，但每一层共享同一个 self-attention。这样 high-level semantic prior 可以在每一层都 leak 进 action pathway（而不是只在最后一层），同时 modality-specific FFN 避免了梯度互相干扰。

直觉 build：可以把它想象成两个人在同一个房间里——一个人是"语义大脑"（Qwen2.5-VL），一个人是"运动控制大脑"（action expert）。他们共享一个"注意力黑板"（shared self-attention），谁需要谁的输出随时看黑板，但写自己的笔记（FFN activation）用各自的笔记本。这比 π0 那种"语义大脑在隔壁房间"更近，比 RT-2 那种"两个人共用一个大脑"更分工明确。

### 2.2 序列构造公式详解

**(1) Observation context**

$$\mathbf{O}_t = [\mathbf{I}_t^1, \mathbf{I}_t^2, \mathbf{I}_t^3, \mathbf{T}_t, \mathbf{s}_t]$$

变量逐个拆解：
- $\mathbf{O}_t$：时刻 $t$ 的观测上下文（observation context），是一个 token 序列
- $\mathbf{I}_t^{1,2,3}$：三个视角的操作图像 tokens。上标 $1, 2, 3$ 分别对应 head camera、left wrist camera、right wrist camera。每张图经过 ViT encoder（Qwen2.5-VL 内置）后 patchify 成 token 序列
- $\mathbf{T}_t$：任务指令 tokens，由 Qwen2.5-VL 的 text tokenizer 编码。注意这里 $t$ 下标其实只是表示该 episode 的指令，指令本身在 episode 内不随时间变
- $\mathbf{s}_t$：机器人本体感知（proprioceptive state），通常是关节角 + 夹爪状态，例如 14 维 dual-arm 关节 + 2 维 gripper = 16 维，通过 MLP 投影到 token embedding 空间

**(2) Action chunk**

$$\mathbf{A}_t = [\mathbf{a}_t, \mathbf{a}_{t+1}, \ldots, \mathbf{a}_{t+T-1}]$$

- $\mathbf{A}_t$：从时刻 $t$ 开始的 action chunk
- $\mathbf{a}_t$：时刻 $t$ 的 action 向量（dual-arm 通常 16 维：14 关节 + 2 gripper）
- 下标 $t, t+1, \ldots, t+T-1$：时间索引，$T$ 是 **action chunk length**
- **$T = 50$** 在预训练阶段。这个数字的选择背后有控制频率的考量——如果控制频率是 50Hz，那 50 个 action 对应 1 秒的轨迹，刚好是一个 atomic manipulation primitive 的典型时长

### 2.3 Flow Matching 训练目标：公式逐项推导

Flow Matching 来自 Lipman et al. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)，是 diffusion 的近亲，但有几条 nice properties：训练更稳定、ODE 而非 SDE、不需要 score function 估计。

**(3) 概率路径构造**

$$\mathbf{A}_{t,s} = s \cdot \mathbf{A}_t + (1-s) \cdot \boldsymbol{\epsilon}$$

- $s \in [0,1]$：flow timestep。**注意这跟真实时间 $t$ 完全无关**，$s$ 是 generative modeling 的"扩散步"。$s=0$ 时 $\mathbf{A}_{t,0} = \boldsymbol{\epsilon}$（纯噪声），$s=1$ 时 $\mathbf{A}_{t,1} = \mathbf{A}_t$（真值）
- $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$：标准高斯噪声，维度与 $\mathbf{A}_t$ 相同
- 这是 linear interpolation path，比 diffusion 的 Markov chain 简单得多

**(4) 条件分布**

$$p(\mathbf{A}_{t,s} | \mathbf{A}_t) = \mathcal{N}(s \cdot \mathbf{A}_t, (1-s) \cdot \mathbf{I})$$

- 均值 $s \cdot \mathbf{A}_t$：随 $s$ 增大，分布中心从 0 滑向真值
- 方差 $(1-s) \cdot \mathbf{I}$：随 $s$ 增大，分布越集中（确定性越高）
- 这个写法等价于 Optimal Transport Flow Matching（OT-FM），是 Lipman paper 里最简单也最有效的 variant

**(5) Flow Matching loss**

$$\mathcal{L}_{\mathrm{FM}} = \mathbb{E}_{s \sim \mathcal{U}[0,1], \mathbf{A}_t, \boldsymbol{\epsilon}} \left\| v_\theta(\mathbf{A}_{t,s}, \mathbf{O}_t, s) - (\mathbf{A}_t - \boldsymbol{\epsilon}) \right\|^2$$

逐项解释：
- $v_\theta$：action expert 神经网络，参数为 $\theta$。它接收三个输入：noisy action $\mathbf{A}_{t,s}$、观测 $\mathbf{O}_t$、flow timestep $s$
- 训练目标 $v^*(\mathbf{A}_{t,s}) = \mathbf{A}_t - \boldsymbol{\epsilon}$：这是从 $\mathbf{A}_{t,s}$ 指向真值 $\mathbf{A}_t$ 的"理想速度场"（ideal vector field）
- $\mathbb{E}_{s \sim \mathcal{U}[0,1]}$：$s$ 从均匀分布 $[0,1]$ 采样，等价于所有扩散步权重相同
- $\|\cdot\|^2$：L2 范数平方，回归损失

**Intuition**：跟 DDPM 的 noise prediction 不同（DDPM 学 $\epsilon_\theta$ 去预测噪声），Flow Matching 学一个 vector field $v_\theta$，使得沿着这个 vector field 走 ODE 就能从噪声流到数据。这个 vector field 是线性的（因为是 linear path），所以训练目标很简单——预测"差值" $\mathbf{A}_t - \boldsymbol{\epsilon}$。推理时用 Euler 或者 midpoint method 解 ODE，通常 10 步就够（vs. DDPM 通常要 50-1000 步）。

### 2.4 Blockwise Causal Attention：防止 information leakage

这个设计直接来自 π0 [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)。把序列 $[\mathbf{O}_t, \mathbf{A}_t]$ 分成三个 block：

| Block | 内容 | 注意力模式 |
|-------|------|-----------|
| Block 1 | $[\mathbf{I}_t^1, \mathbf{I}_t^2, \mathbf{I}_t^3, \mathbf{T}_t]$ | Bidirectional（块内全连接）|
| Block 2 | $[\mathbf{s}_t]$ | 仅能看 Block 1 + 自己 |
| Block 3 | $[\mathbf{a}_t, \mathbf{a}_{t+1}, \dots, \mathbf{a}_{t+T-1}]$ | Bidirectional（块内全连接）+ 能看 Block 1, 2 |

跨 block 的 causal mask：Block 3 → Block 2 → Block 1。这种设计的关键意图：
- **Block 3 内部 bidirectional**：因为 action chunk 是一组相关联的未来轨迹，预测 $\mathbf{a}_{t+10}$ 时能看到 $\mathbf{a}_{t+20}$ 的 ground truth，这让训练信号更强（类似 masked language modeling）。推理时因为 chunk 已经生成，不需要 causal 约束。
- **阻止 Block 3 → Block 1/2 的反向信息流**：防止 future action token "作弊" leak 到观测表示里。如果不阻止，模型会学一个 shortcut：从未来 action 反推当前 observation，破坏训练-推理一致性。

### 2.5 Vision Distillation from LingBot-Depth

这是本文相对 π0 的一个 architecture-level 创新。直觉是：VLM（Qwen2.5-VL）虽然语义强，但几何/深度感知弱。Spatial VLA 系列（[arXiv:2501.15830](https://arxiv.org/abs/2501.15830), [arXiv:2508.09071](https://arxiv.org/abs/2508.09071), [arXiv:2510.12276](https://arxiv.org/abs/2510.12276)）证明 depth prior 能显著提升 manipulation 性能。LingBot-VLA 用了一个 distillation 方式：

**(5) Distillation loss**

$$\mathcal{L}_{distill} = \mathbb{E}_{\mathbf{Q}_t} \left| \mathrm{Proj}(\mathbf{Q}_t) - \mathbf{D}_t \right|$$

变量：
- $\mathbf{Q}_t = [\mathbf{Q}_t^1, \mathbf{Q}_t^2, \mathbf{Q}_t^3]$：三视角图像对应的 **learnable queries**。这些是模型自动学的"询问 depth 的 question token"
- $\mathrm{Proj}(\cdot)$：一个 cross-attention 投影层，把 query 维度对齐到 depth token 维度
- $\mathbf{D}_t = [\mathbf{D}_t^1, \mathbf{D}_t^2, \mathbf{D}_t^3]$：从 **LingBot-Depth** 模型蒸馏出的 depth tokens。LingBot-Depth 是同实验室的配套工作，做 masked depth modeling [paper ref 24]

直觉：让 VLM 的视觉 pathway 在每张图旁边挂几个"问 depth 的小抄位"，强迫这些位置的 hidden state 经过投影后等于 depth model 的输出。这样 depth 几何信息被 soft-injected 到 VLM 内部表示里，而不是简单 concatenate 一个 depth channel。这种设计好处是推理时 **不需要 depth camera**——distillation 只在训练时用，推理时模型已经"内化"了 depth prior。

---

## 三、Pre-training Dataset：9 个 embodiment 的双臂数据

### 3.1 9 种机器人配置清单

| Embodiment | Arm DoF | Camera | Teleop 方式 |
|-----------|---------|--------|-------------|
| AgiBot G1 | 2×7 | 3× RGB-D | VR |
| AgileX | 2×6 | 3 cameras | Isomorphic arms |
| Galaxea R1Lite | 2×6 | 1 stereo + 2 wrist | - |
| Galaxea R1Pro | 2×7 | 1 stereo + 2 wrist | - |
| Realman Rs-02 | 2×7 + 2 gripper (16-dim) | 3 cameras | - |
| Leju KUAVO 4 Pro | 2×7 + 2 gripper | 1 head + 2 wrist | - |
| Qinglong (humanoid) | 2×7 | 1 head + 2 wrist | - |
| ARX Lift2 | 2×6 | 3 cameras | - |
| Bimanual Franka | 2×7 + 2 gripper (16-dim) | 3 cameras | - |

总数据规模约 20,000 小时，这是迄今为止公开报道的最大规模真实世界双臂 manipulation 数据集。作为对照：
- Open X-Embodiment（OXE）：~1M episodes，但 heterogeneous quality，很多是仿真
- AgiBot World：~1M episodes
- π0.5 dataset：~10,000 hours 量级
- GR00T N1 pretrain：~5000 小时左右

LingBot 的数据多样性体现在两个维度：embodiment 多样（9 种）+ 行为多样（atomic action word cloud 显示训练集 top-100 atomic action 只覆盖测试集 50% 的 atomic action，意味着测试集有大量 out-of-distribution 行为类别，确保了真正的 generalization 评估）。

### 3.2 Data Labeling Pipeline

两步：
1. **Video Segment**：人工标注员把多视角视频按 predefined atomic actions 切分。同时去除首尾静态帧——这一步看似琐碎实则关键，因为静态帧如果带 noise 进入训练会让 action expert 学到 "do nothing" mode。
2. **Instruction Annotation**：用 Qwen3-VL-235B-A22B [arXiv:2502.13923 family](https://arxiv.org/abs/2502.13923) 自动生成 task instruction 和 sub-task instruction。

Intuition：用 235B 量级的 VLM 做 annotation 而不是 GPT-4V，是因为 Qwen3-VL 在中文场景和细粒度物体识别上对国内机器人公司更友好。同时 235B 的 A22B (active 22B) MoE 推理成本可控。

---

## 四、Training Efficiency：让 20K hours 数据训练可行

### 4.1 Distributed Strategy

三层优化：

**(a) FSDP2 + HSDP hybrid**

- **FSDP (Fully Sharded Data Parallel)**：PyTorch 原生的 ZeRO-3 实现，把 optimizer states、gradients、parameters 全部 shard 到所有 GPU 上。优点：内存占用最低。缺点：每步 all-gather 通信量大。
- **HSDP (Hybrid Sharded Data Parallel)** from VeOmni [arXiv:2508.02317](https://arxiv.org/abs/2508.02317)：在 FSDP 之上引入 "shard groups"——只在 group 内部 shard，group 之间用 DDP。这样通信范围被限制在小组内。
- **LingBot 的关键 trick**：**只对 action expert 模块构建 shard group**，VLM backbone 保持完整 replicate。直觉是 action expert 参数少但梯度通信频繁，shard 后能减少通信；VLM backbone 参数多但被 freeze 或者 low LR finetune，replicate 反而省事。

**(b) Mixed precision policy**

- Reduction（梯度同步）：`torch.float32` 保证数值稳定
- Storage 和 communication：`torch.bfloat16` 节省显存和带宽

bfloat16 在 VLA 训练里特别重要，因为 action 信号是高频连续的，FP16 的 limited dynamic range 容易在 action gradient 上 overflow。bfloat16 有 8-bit exponent 跟 FP32 相同，更适合这种场景。

### 4.2 Operator-Level Optimization

- **FlexAttention**：PyTorch 2.5+ 提供的 flexible attention API，能编译 arbitrary attention pattern。VLA 的 blockwise causal attention 就是 FlexAttention 的标准用例。FlexAttention 把 attention mask 编译成 fused kernel，避免了 naive implementation 里 "create mask → apply mask → softmax" 三步的开销。
- **torch.compile operator fusion**：把多个 elementwise op 融合成一个 CUDA kernel，减少 kernel launch overhead。

### 4.3 Throughput Numbers

报告的核心数字：**261 samples/s/GPU** 在 8×H200 配置下。

跟 baselines 对比（在 LIBERO 数据集上、用 π-like 模型架构、batch size=32 控制 variable）：

| Codebase | Distributed | Throughput 相对值 |
|----------|-------------|------------------|
| StarVLA | ZeRO | 1.0×（基线）|
| Dexbotic | ZeRO | 1.0× |
| OpenPI | DDP | 1.0× |
| **Ours** | FSDP2 + HSDP + FlexAttention | **1.5×~2.8×** |

论文 Fig.4 显示在 8/16/32/128/256 GPU 配置下，scaling efficiency 接近 theoretical linear scaling。这个 scaling efficiency 数字对 large-scale 实验室是决定性的——通信 overhead 不爆炸意味着可以无痛 scale 到几百卡。

---

## 五、Evaluation：22,500 trials 的真实世界 benchmark

### 5.1 GM-100 Benchmark

GM-100 是 paper reference 29 提到的 100 个 detail-oriented manipulation tasks。LingBot 团队在这个 benchmark 上做了：

- **25 physical robots** 横跨 3 个平台（Agibot G1、AgileX、Galaxea R1Pro）
- **100 tasks**
- **130 expert demos per task per platform**（从 150 raw 筛选）
- **15 trials per task per robot** 评估
- 总评估量：3 platforms × 100 tasks × 15 trials = **4,500 trials per model**，5 个 model 对比 = **22,500 trials**

数据收集严格 protocol：
1. **Standardized Objects**：所有 task 用 GM-100 规定的统一物体，跨 site 可复现
2. **Environmental Diversity**：物体 pose 在 workspace 内 randomize，防止 overfitting 到固定位置
3. **Teleoperation Guidelines**：
   - 末端执行器与桌面保持 clearance 防止碰撞
   - 接触阶段降速保证 smooth manipulation
   - episode 起止帧视觉差异显著
4. **Automated Filtering + Manual Review**：双保险，自动算法筛掉技术异常 episode，人工 multi-view 视频复核

### 5.2 Evaluation Metrics

**(a) Success Rate (SR)**：3 分钟内完成 task 所有步骤的 trial 比例。Primary metric。

**(b) Progress Score (PS)**：partial completion 比例。例：6 步的 "Stack Bowls" task，模型完成到第 4 步失败，PS = 4/6 ≈ 0.67。Diagnostic metric，反映 failure mode。

**(c) Termination Criteria**：
- 连续 3 次 subtask 失败
- Safety-critical event（如 collision）

### 5.3 主实验结果（Table 1 详解）

| Platform | WALL-OSS SR/PS | GR00T N1.6 SR/PS | π0.5 SR/PS | Ours w/o depth SR/PS | **Ours w/ depth SR/PS** |
|----------|---------------|------------------|-----------|---------------------|------------------------|
| Agibot G1 | 2.99 / 8.75 | 5.23 / 12.63 | 7.77 / 21.98 | 12.82 / 30.04 | **11.98 / 30.47** |
| AgileX | 2.26 / 8.16 | 3.26 / 10.52 | 17.20 / 34.82 | 15.50 / 36.31 | **18.93 / 40.36** |
| Galaxea R1Pro | 6.89 / 14.13 | 14.29 / 24.83 | 14.10 / 26.14 | 18.89 / 34.71 | **20.98 / 35.40** |
| **Average** | **4.05 / 10.35** | **7.59 / 15.99** | **13.02 / 27.65** | **15.74 / 33.69** | **17.30 / 35.41** |

几个观察：

1. **LingBot-VLA w/ depth vs π0.5**：平均 SR +4.28%，平均 PS +7.76%。相对提升分别 ~33% 和 ~28%。
2. **GR00T N1.6 在 Galaxea R1Pro 上反常**：SR 14.29% 跟 π0.5 持平，明显比它在另外两个 platform 上的表现好。原因是 GR00T N1.6 pretraining 数据里大量包含 Galaxea R1Pro 数据——**pretraining data 的 embodiment overlap 直接 translate 到下游性能**。这条观察很重要：cross-embodiment generalization 不是 free lunch，预训练数据结构相似性是 strong prior。
3. **Depth 的贡献 platform-dependent**：在 Agibot G1 上 depth 反而让 SR 略降（12.82% → 11.98%），在 AgileX 和 Galaxea 上提升明显。可能因为 Agibot G1 的 RGB-D camera 配置跟 distillation 的 depth prior 不完全 match，或者 G1 的 task 集合对几何信息需求低。
4. **整体绝对 SR 不高**：17% 这个数字看起来低，但要在 100 个 detail-oriented tasks 上平均，并且很多 task 是 paper appendix 表格里的 #002、#006 这种 hard task（所有方法都接近 0% SR）。这是 community 第一次看到 large-scale real-world benchmark 上的真实 baseline 数字。

### 5.4 Simulation Benchmark（Table 2）

RoboTwin 2.0 [arXiv:2506.18088](https://arxiv.org/abs/2506.18088)，50 个 representative tasks。

| Setting | π0.5 | Ours w/o depth | **Ours w/ depth** |
|---------|------|----------------|-------------------|
| Clean scenes | 82.74% | 86.50% | **88.56%** |
| Randomized scenes | 76.76% | 85.34% | **86.68%** |

观察：
- **Randomized 场景下提升更大**（+9.92% vs +5.82%）：说明 depth prior 主要帮助的是 visual generalization，对 clutter/光照变化鲁棒性贡献大。
- Simulation 绝对数字远高于 real-world（~88% vs 17%），这跟所有 VLA paper 一致——sim 还是 simpler。LingBot-VLA 在 sim 上跟 π0.5 拉开 +5~10% 差距，real-world 上拉开 +4%，说明 **sim-to-real gap 在 LingBot-VLA 上没比 baseline 大**——尽管预训练数据是 real-world，模型没有 overfit 到 real 数据分布。

附录 Table S7 给出 50 个 task 的 per-task breakdown，其中 Hanging Mug、Click Bell、Blocks Ranking Size 这种精细 task 上 depth 提升最显著（Hanging Mug 在 randomized 场景下 π0.5 17% → Ours w/ depth 53%）。这些 task 共同点是需要精确的 3D pose 估计来挂/对齐物体，跟 depth prior 的 inductive bias 完美 match。

---

## 六、Ablation Studies：Scaling Law 与 Data Efficiency

### 6.1 Scaling Experiment（Fig. 5）

实验设置：25 个 representative tasks，pretraining 数据量从 3,000 → 20,000 小时。

观察：
- **SR 和 PS 都随数据量单调上升**，无 saturation
- **三个 embodiment 的 individual 曲线**（Agibot G1、AgileX、Galaxea R1Pro）**都跟 aggregate 趋势一致**，说明 scaling law 不是 platform-specific

这条结论的 community significance：
- LLM scaling law（Kaplan 2020、Chinchilla 2022）建立在 token-based 文本数据上，大家一直怀疑 real-robot data 是否也有类似性质——因为 robot data 是 trajectory-based、强 embodiment-correlated、cost 高。
- LingBot 的结果显示：**至少在 3K→20K 区间，real-robot data 也有 power-law-like scaling，且 no saturation**。这给出一个非常 actionable 的信号：继续 scale data 是值得的。
- 但 20K hours 之后会怎样？paper 没说。可能 50K hours 才能看到 saturation，也可能 100K 才看到。这留给了 community 后续工作。

### 6.2 Data-efficient Analysis（Fig. 6）

实验：8 个 representative tasks，Agibot G1 平台。

结论：**只用 80 demonstrations 的 LingBot-VLA > 用 130 demonstrations 的 π0.5**。换句话说，LingBot 的 pretraining 把 sample efficiency 提升了至少 1.6×。

更重要的趋势：**随着 post-training data 增加，LingBot 和 π0.5 的 gap 拉大**（不是缩小）。直觉解释：pretraining 给了 model 一个好的 initialization，这个 initialization 在小数据下表现为"先验帮忙"，在大数据下表现为"exploration efficiency"——model 知道在 data manifold 的哪一部分去 search，从而每一份新数据都更 informative。这正是 foundation model paradigm 的核心 selling point。

---

## 七、和 Related Work 的对照思考

### 7.1 跟 π0 / π0.5 的对比

π0 [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) 和 π0.5 [Ref 5] 是目前最 close 的 baseline。架构上：
- π0: PaliGemma + 独立 action expert transformer（hard decoupling）
- π0.5: 在 π0 之上加 web data co-training + hierarchical planning
- LingBot-VLA: Qwen2.5-VL + MoT-shared-attention（soft decoupling）+ depth distillation

LingBot 选择 Qwen2.5-VL 而不是 PaliGemma 是一个有信息量的决策。PaliGemma 3B 是 2024 年初的 SOTA 通用 VLM，但 Qwen2.5-VL 在中文 + 细粒度 OCR + 物体 grounding 上明显更强。对于国内 robot 公司的场景（大量中文 instruction + 国产物体），Qwen2.5-VL 的 prior 更 align。

### 7.2 跟 GR00T N1/N1.6 的对比

GR00T N1 [arXiv:2503.14734](https://arxiv.org/abs/2503.14734) 是 NVIDIA 的工作，预训练数据包含大量 humanoid 双臂数据，但 embodiment 多样性不如 LingBot（LingBot 9 种 vs GR00T 4-5 种）。GR00T 在 Galaxea R1Pro 上反常表现正是 embodiment overlap 的体现。

### 7.3 跟 Spatial VLA 系列的对比

SpatialVLA [arXiv:2501.15830](https://arxiv.org/abs/2501.15830)、GeoVLA [arXiv:2508.09071](https://arxiv.org/abs/2508.09071)、Spatial Forcing [arXiv:2510.12276](https://arxiv.org/abs/2510.12276)、InternVLA-M1 [arXiv:2510.13778](https://arxiv.org/abs/2510.13778) 这些工作都试图给 VLA 加 spatial inductive bias。LingBot 的方式（learnable query + cross-attention distillation from LingBot-Depth）相对 elegant，因为它：
- 训练时利用 depth prior
- 推理时不需要 depth camera（部署友好）
- 不改 VLM backbone（与 MoT 架构兼容）

### 7.4 跟 BAGEL 的关系

BAGEL [arXiv:2505.14683](https://arxiv.org/abs/2505.14683) 是 MoT 架构的提出者，原本是 multimodal understanding & generation 的 unified model。LingBot 把 MoT 拓展到 action modality，证明 MoT 在 VLA 场景也 work。这个 transfer 很自然——action 可以看作一种 "continuous generation modality" 跟 image generation 类似。

---

## 八、Appendix 数据细节：per-task breakdown 的 insight

附录 Table S1-S6 给了 100 个 task 的 per-platform per-model SR/PS。挑几个有信息量的：

**#015（应该是简单 task，可能是单臂 pick）**：所有方法都接近 80-100% SR。说明 task 难度分布跨度很大。

**#002、#006、#025、#026**：所有方法都接近 0% SR。这些是 "impossible tasks"——要么 task 定义本身有歧义，要么需要超出当前 VLA 能力的 long-horizon reasoning。这部分任务暴露了 current VLA 的真实能力 ceiling。

**#030、#077、#085、#086**：π0.5 和 LingBot 都能做 60-100% SR，但 WALL-OSS 和 GR00T N1.6 失败。这类 task 应该是 π0/LingBot pretraining 数据覆盖的 atomic action 类别。

**#035 在 Galaxea R1Pro 上**：LingBot w/o depth 0% SR，w/ depth 0% SR——这种 task 即使 depth 也救不了，可能是 task 本身 definition 问题。

整体看 per-task 数据分布，可以推断：**VLA 性能提升不是平均提升所有 task，而是 unlock 一批 medium-difficulty task**。Hard task 依然 0%，easy task 早就饱和。这是 foundation model scaling 的典型 pattern——不是把最难的 task 拉到能用，而是把 medium task pool 扩大。

---

## 九、Limitations 没明说但能 infer 的几点

1. **全部 dual-arm**：paper 自己提到未来工作会 integrate single-arm + mobile robot。dual-arm bias 是 limitation，对工业单臂场景迁移性未验证。
2. **Tabletop only**：所有 task 都是 tabletop manipulation，没有 locomotion + manipulation 联合任务。mobile manipulation 是 missing piece。
3. **3 分钟 timeout**：长 horizon task（>3 min）未被覆盖。
4. **VR teleoperation 数据分布 bias**：teleop 数据有人类 demonstration 的 motion prior，可能跟自主执行的最优 trajectory 不同（人类 tend to over-cautious）。
5. **Depth 模块只在 distillation 时用**：如果部署时 depth camera available，能否用上？paper 没讨论 inference-time depth integration。
6. **LingBot-Depth 不开源**：作为 distillation teacher，depth model 不开源会限制 reproduction。但 paper 提到 open access code/model/benchmark data，depth model 可能 proprietary。

---

## 十、对 community 的 takeaway

1. **Scaling law for real-robot data 是真实存在的**，至少到 20K hours 还没 saturation。这给了所有 robot lab 一个明确信号：继续 collect data。
2. **训练 infrastructure 是 bottleneck**：LingBot 的 codebase 把 8×H200 训练 throughput 推到 261 samples/s，对应每秒 ~13K action tokens（按 T=50、batch 256 估算）。这种 infrastructure 开源对中小实验室价值巨大。
3. **Foundation model paradigm 在 robotic manipulation 上 work**：80 demos > 130 demos 的 data efficiency 实验是最直接证据。
4. **Cross-embodiment generalization 不是 free**：GR00T 在 Galaxea 上反常表现说明 embodiment overlap 仍是 strong prior，"universal robot foundation model" 还有很长的路。
5. **Depth prior 通过 distillation 注入是 viable path**：避免 inference-time depth camera 依赖，同时享受 depth 的几何 inductive bias。

---

## Reference links

- Project page: https://technology.robbyant.com/lingbot-vla
- GitHub: https://github.com/robbyant/lingbot-vla
- HuggingFace: https://huggingface.co/robbyant/lingbot-vla
- LingBot-Depth: https://technology.robbyant.com/lingbot-depth
- π0 paper: https://arxiv.org/abs/2410.24164
- Flow Matching: https://arxiv.org/abs/2210.02747
- Qwen2.5-VL: https://arxiv.org/abs/2502.13923
- BAGEL: https://arxiv.org/abs/2505.14683
- GR00T N1: https://arxiv.org/abs/2503.14734
- RoboTwin 2.0: https://arxiv.org/abs/2506.18088
- VeOmni: https://arxiv.org/abs/2508.02317
- SpatialVLA: https://arxiv.org/abs/2501.15830
- GeoVLA: https://arxiv.org/abs/2508.09071
- Spatial Forcing: https://arxiv.org/abs/2510.12276
- InternVLA-M1: https://arxiv.org/abs/2510.13778
- PaliGemma: https://arxiv.org/abs/2407.07726
- Gemini Robotics: https://arxiv.org/abs/2503.20020
- Gemini Robotics 1.5: https://arxiv.org/abs/2510.03342

---

Karpathy 老师如果要 build deeper intuition，我建议重点看 paper 的 Fig.5 scaling 曲线和 Fig.6 data efficiency 曲线——这两张图是 foundation model paradigm 在 robotic 上是否成立的核心证据。剩下所有的 architecture choice（MoT、Flow Matching、depth distillation）都是为了让这个 scaling 能 efficient 地跑起来的 supporting engineering。LingBot 的工作本质上是把 LLM playbook（large-scale pretrain + efficient infra + careful benchmark）完整地搬到了 real-robot 上，并第一次展示了 playbook 在这个 domain 也 work。
