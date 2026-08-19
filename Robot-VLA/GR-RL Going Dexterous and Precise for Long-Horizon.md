---
source_pdf: GR-RL Going Dexterous and Precise for Long-Horizon.pdf
paper_sha256: 8b94f6cc63f6f7d2a36448015583b27573dd90a8313e6ba8aaacbfd0e7b79f07
processed_at: '2026-08-19T09:47:53-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Andrej，咱们用最直白的“人话”把这篇 paper 拆解一下。这篇 paper 核心想解决一个极其反直觉的问题：**人类给的 demonstration 数据，在精细操作任务里，其实是“有毒”的。**

为了 build your intuition，我们先看一个场景：你教机器人穿鞋带。你拿着遥操作手柄，对准鞋眼，手抖了一下，没穿进去。你退回来一点，换个角度，再试一次，穿进去了。这条轨迹喂给传统的 Behavior Cloning (BC) 模型，模型会把“抖一下、退回来”这种由于人类操作失误产生的 multimodal 噪声全部学进去。结果就是，机器人在真实部署时，也会学着你的样子“抖一下、退回来”，但在毫米级的精度要求下，这一抖，任务直接失败。另外，训练时模型预测的是 raw action chunks，但实际跑机器人时，为了平滑，系统会做 temporal ensembling 或者 receding horizon control，这导致模型训练时看到的东西和推理时执行的东西完全对不上号。

GR-RL 的核心 intuition 就是：**与其让模型死记硬背人类充满瑕疵的演示，不如先用强化学习的价值观去其糟粕、取其精华，然后让它自己在真实世界里闭着眼试错对齐。** 整个 process 分为三步。

### 第一步：用 RL 的 Q-value 当“进度条”，过滤掉人类的烂操作

既然人类数据有噪声，怎么把“退回来重试”这种烂操作摘出来？人工去标定太主观。GR-RL 的做法是直接跑 Offline RL (TD3+BC) 训练一个 Critic 模型 $Q_\phi$。

**Reward 公式 (公式 1) 解析：**
$$r(\mathbf{o}_t, l, \mathbf{s}_t, \mathbf{a}_t) = \begin{cases} \gamma^{T-t} \mathbb{I}(\tau), & t > T-k \\ 0, & t \le T-k \end{cases}$$
*   $\mathbf{o}_t, l, \mathbf{s}_t, \mathbf{a}_t$ 分别是 $t$ 时刻的观测、语言指令、机器人状态和动作。
*   $T$ 是整条轨迹的长度，$k$ 是 action chunk 的长度。
*   $\gamma$ 是 discount factor (折扣因子)。
*   $\mathbb{I}(\tau)$ 是一个 indicator function，如果整条轨迹 $\tau$ 最终成功了就是 1，失败就是 0。
*   **Intuition：** 这个 reward 极度 sparse。只有在轨迹快结束的最后 $k$ 步，并且整条轨迹最终成功了，才给一个非零的 reward。前面的所有步 reward 全是 0。通过 $\gamma^{T-t}$ 的衰减，越接近成功的 state，其 value 越高。

为了让 Critic 学会分辨什么是“失败”，他们用了一个很巧的 hindsight trick：在人类成功操作的轨迹里，把人类“退回来重试”的那个 keyframe 找出来，直接把前面的轨迹截断，假装这是一条失败的轨迹喂给模型。

训练完之后，这个 Critic $Q_\phi$ 就成了一个完美的“任务进度条”。它用了 **Distributional RL**，输出是一个 $[0, 1]$ 的 bounded distribution，而不是一个 unbounded 的 scalar。这是因为在极度 sparse 的 reward 下，传统 scalar critic 会疯狂 over-estimation（早期 state 没有监督信号，Q值会爆炸），而 distributional critic 强制把值域压在 $[0, 1]$，自然反映进度。

**过滤逻辑：** 计算每一步的 progress $\rho_t = \text{mean}(Q_\phi(\dots))$。如果在某一段里 $\rho$ 突然掉下来了，说明人类在这里犯蠢了，直接把这些 transition 从 dataset 里踢掉。用清洗后的数据做 BC，成功率从 45.7% 直接飙到 61.6%。

### 第二步：左右手互搏之术

这步极简极暴力。既然是 bimanual (双臂) 机器人，物理上是对称的。那就把所有图像左右翻转，左右手腕的相机画面互换，动作 $\mathbf{s}_t$ 和 $\mathbf{a}_t$ 按照世界坐标系的 mirror symmetry 转换回局部关节坐标。同时把语言指令里的 "left" 改成 "right"。数据量白嫖一倍，模型泛化性大增，成功率干到 72.7%。

### 第三步：在 Latent Space 里做 Online RL，解决 Train-Inference Mismatch

现在进入最硬核的部分。模型部署时，由于有平滑控制，实际执行的 action 和训练时预测的 raw action 不一致。怎么补这个 gap？必须做 Online RL，让机器人在真实环境里试错。但穿鞋带这种任务要毫米级精度，你如果在 raw action space（关节角度或者末端位姿）里加高斯噪声去探索，机器人一顿乱抖，鞋带早飞了，永远拿不到 reward。

GR-RL 的解法是 **Latent Space Steering**。Policy 采用的是 Action Diffusion Transformer (DiT)，通过 Flow Matching 训练。熟悉生成模型的话都知道，Diffusion/Flow 模型生成 action，是从一个初始噪声 $\epsilon_t$ 起步的。既然不能在 action space 加噪声，那就在初始噪声 $\epsilon_t$ 空间加噪声！

他们加了一个极小的 Noise Predictor $\pi_{\theta'}$ (51.5M 参数) 挂在 VLM backbone 后面，专门预测这个 $\epsilon_t$。

**Policy Loss (公式 3) 解析：**
$$\mathcal{L}(\pi_{\theta'}) = \mathbb{E} \left[ -Q_{\phi'}(\mathbf{o}_t, l, \mathbf{s}_t, \epsilon_t) + c \max\left(\frac{1}{2}\|\epsilon_t\|^2 - \beta, 0\right) \right]$$
*   $Q_{\phi'}(\dots, \epsilon_t)$ 是 noise space 的 Q-value。第一项就是标准 RL，最大化这个 Q 值，即寻找能带来高回报的 latent noise。
*   第二项是 penalty term。$c$ 是系数，$\|\epsilon_t\|^2$ 是噪声的 L2 范数，$\beta$ 是发散阈值。
*   **Intuition：** 如果 $\pi_{\theta'}$ 为了追求高 Q 值，输出的 $\epsilon_t$ 偏离标准正态分布 $\mathcal{N}(0,1)$ 太远（即 $\frac{1}{2}\|\epsilon_t\|^2 > \beta$），就会被狠狠惩罚。这就强行把探索的 noise 约束在 offline 训练时模型见过的分布内，防止机器人生成 OOD 的乱码动作导致物理崩溃。这种在 latent space 做约束的思想，和 Classifier-Free Guidance 里面控制 unconditional noise 和 conditional generation 的平衡非常神似。

因为 Flow Matching 模型反传梯度太贵，他们蒸馏了一个 noise space 的 Critic $Q_{\phi'}$ 来算 loss。

**Critic Loss (公式 4) 解析：**
$$\mathcal{L}(Q_{\phi'}) = \text{cross\_entropy}(Q_{\phi'}(\dots, \epsilon_t), Q_{\phi}(\dots, \pi_{\theta}(\dots | \epsilon_t)))$$
*   这里用 cross-entropy 做蒸馏。Target 是原 action space 的 Critic $Q_\phi$ 评估给定 $\epsilon_t$ 生成的 action $\pi_{\theta}(\dots | \epsilon_t)$ 的 value。
*   采样时有个 50/50 trick：输入的 $\epsilon_t$ 有 50% 概率从标准正态分布 $\mathcal{N}(\mathbf{0}, \mathbf{1})$ 采，50% 概率从 $\pi_{\theta'}$ 采。确保 $Q_{\phi'}$ 对整个 noise space 都有 coverage，防止 mode collapse。

经过这一步 Online RL，模型在真实环境里试错，把训练和推理的 gap 彻底抹平，成功率最终干到了 83.3%。

### 关键实验数据直觉

| Training Stage | Method | Success Rate |
| :--- | :--- | :--- |
| Baseline | GR-3 (BC on all data) | 45.7% |
| Stage 1 | + Filtered BC | 61.6% |
| Stage 2 | + Morphological Symmetry Aug. | 72.7% |
| Stage 3 | + Online Steering RL | 83.3% |

看 Figure 6 的失败模式分析，Filtered BC 主要降低了 "Threading" (穿鞋眼) 阶段的失败率，因为剔除了人类的瞎操作。Online RL 主要降低了 "Handover" (双手交接鞋带) 和 "Pull tight" (拉紧) 的失败率，因为这两个阶段受平滑控制的影响最大，必须靠 Online 试错来对齐。

### 顺带聊聊硬件 ByteMini-v2

既然是做 high-precision 任务，硬件如果拉胯，算法再牛也白搭。ByteMini-v2 把肘关节电机的峰值扭矩从 17 Nm 干到了 35 Nm，负载翻倍。更重要的是底盘投影面积缩小 (450x650 mm)，加上专门设计的球状腕关节，在狭窄空间里的操作冗余度大增。

### 相关联想与扩展

1.  **与 Physical Intelligence 的 $\pi_{0.6}^*$ 对比：** $\pi_{0.6}^*$ 也是用 distributional critic 来学 progress，但它直接用 advantage-conditioned 去引导 denoising。GR-RL 觉得先把 offline base policy 用 filtered BC 训练扎实，能极大缩小 online RL 的探索空间，这在样本效率极其低下的 real-world RL 里是致命的优势。
2.  **World Model 的替代路径：** 像 TD-MPC2 那套，学个 world model 然后在 imagination 里做 RL，在低维控制里很爽，但在 deformable object (鞋带这种软体) 的像素级预测上极易崩盘。这也是 GR-RL 坚持做 real-world off-policy RL 的原因。
3.  **Diffusion Policy 的通病：** Action chunking 加上 temporal ensembling 虽然让动作平滑了，但破坏了 VLA 模型的 autoregressive causality。模型其实不知道自己前一秒真正执行了什么动作，这导致了 compounding error。Online RL 的核心价值就是把这个 execution gap 的因果链重新接上。

总而言之，这篇 paper 的 takeaway 很简单：**人类数据不可全信，RL 的价值函数是清洗数据最好的滤波器，而在生成式 policy 里做 RL，潜变量空间是唯一可行的探索维度。**

**Reference Web Links:**
*   GR-RL Project Page: https://seed.bytedance.com/gr_rl
*   GR-3 Technical Report: https://arxiv.org/abs/2507.15493
*   TD3+BC (Offline RL): https://arxiv.org/abs/2106.01345
*   Steering Diffusion Policy (Latent RL): https://arxiv.org/abs/2506.15799
*   $\pi_{0.6}^*$ (VLA learns from experience): https://arxiv.org/abs/2511.14759
*   Distributional RL (C51): https://arxiv.org/abs/1707.06887
*   Flow Matching for Generative Modeling: https://arxiv.org/abs/2210.02747

---

这篇 paper 介绍了一个名为 **GR-RL** 的 robotic learning framework，它的核心目标是把 generalist vision-language-action (VLA) policy 转化为能够执行 long-horizon、high-precision、dexterous manipulation 任务的 specialist。最亮眼的成就是实现了 83.3% 成功率的 autonomous shoe lacing (穿鞋带) 任务。这项工作由 ByteDance Seed 团队完成。

为了 build your intuition，我会从 core motivation、architecture、multi-stage training recipe 的数学细节、hardware 以及实验数据这几个维度进行深度拆解。

### 1. Core Motivation: 为什么需要 GR-RL?

现有的 VLA models (如 GR-3, $\pi_0$) 在 generalization 上表现很好，但在面对需要 millimeter-level precision 和 long-horizon reasoning 的任务时面临两个核心 bottleneck：
1.  **Suboptimal Human Demonstrations:** 在极度精细的操作中，人类遥操作员会犹豫、犯错、重试。直接使用 Behavior Cloning (BC) 会把这些 noisy multimodal actions 强行学进去，导致 policy 性能下降。
2.  **Train-Inference Mismatch:** 训练时，VLA 预测固定长度的 action chunks。但在实际部署时，为了保证平滑控制，系统会使用 temporal ensembling 或 asynchronous receding horizon control。这意味着模型在训练时看到的 raw actions 和推理时实际执行的 optimized actions 之间存在偏差。

GR-RL 的解决思路是 multi-stage reinforcement-augmented training：Filter (过滤) -> Augment (增强) -> Reinforce (强化)。

### 2. Architecture: Mixture-of-Transformer (MoT)

GR-RL 采用了 5B 参数的 Mixture-of-Transformer (MoT) 架构，包含两个核心组件：

*   **Policy $\pi_\theta$:** 基于 GR-3 架构，使用 **Qwen2.5-VL-3B-Instruct** 作为 Vision-Language-Model (VLM) backbone。它接收 language instruction $l$，observation $\mathbf{o}_t$ 和 robot state $\mathbf{s}_t$，输出 $k$-length 的 action chunk $\mathbf{a}_t = a_{t:t+k}$。Action 生成采用 Action Diffusion Transformer (DiT)，通过 **Flow Matching** 目标进行训练。为了快速推理，只使用 VLM 后半部分的 KV cache。
*   **Critic $Q_\phi$:** 评估 action chunk 的好坏。采用 **Q-chunking** 机制预测一个 chunk 的 Q-values。关键在于，它使用了 **Distributional Reinforcement Learning**。Critic 把 value 当作一个有上下界的 discrete distribution (上限 1，下限 0)，而非无界的 scalar regression。

**Intuition behind Distributional Critic:** 在 long-horizon 和 sparse reward 场景下，如果用传统的 scalar critic，因为 reward 稀疏，早期的 state 几乎没有监督信号，很容易导致 value over-estimation (Q-value 爆炸)。Distributional critic 限制输出在 $[0, 1]$，强制模型学习一个 bounded distribution，这不仅捕捉了真实世界的不确定性，还能在 sparse reward 下 robustly converge，从而完美反映 task 的 progress。

### 3. Training Recipe 深度解析

#### Stage 1: Data Filtering with Learned Task Progress Evaluator

这是 Offline RL 阶段，目标是训练一个 critic $Q_\phi$ 来评估每一步的 progress，剔除 suboptimal transitions。使用 **TD3+BC** 算法。

**Reward Function (公式 1):**
$$r(\mathbf{o}_t, l, \mathbf{s}_t, \mathbf{a}_t) = \begin{cases} \gamma^{T-t} \mathbb{I}(\tau), & t > T-k \\ 0, & t \le T-k \end{cases}$$
*   $\mathbf{o}_t, l, \mathbf{s}_t, \mathbf{a}_t$: 分别代表时刻 $t$ 的 observation, language instruction, state 和 action。
*   $T$: Trajectory 长度。$k$: Action chunk 长度。
*   $\gamma$: Discount factor (折扣因子)。
*   $\mathbb{I}(\tau)$: Indicator function，如果 trajectory $\tau$ 成功则为 1，否则为 0。
*   **Intuition:** 只有在 trajectory 的最后 $k$ 步，且整个 trajectory 成功时，才有非零 reward。前面的所有步骤 reward 为 0。这是一个极度 sparse 的设置。$\gamma^{T-t}$ 确保越接近成功的状态，reward 衰减越少，value 越高。

**Hindsight Trajectory Augmentation:** 因为大部分收集的数据都是成功的，为了让 critic 学会区分好坏，他们在成功的 trajectory 中标注 **retry keyframes** $m_i$。假设一个成功轨迹在 $m_i$ 处发生了重试，就把 $0$ 到 $m_i$ 截断，当作一条失败的 trajectory 喂给模型。

**Progress Calculation (公式 2):**
$$\rho_t := \text{mean}(Q_\phi(\mathbf{o}_t, l, \mathbf{s}_t, \mathbf{a}_t))$$
*   $\rho_t$: 时刻 $t$ 的 task progress。
*   $Q_\phi$ 输出一个 categorical distribution，取其 mean 作为 progress 的标量评估值。
*   **Filtering Logic:** 如果在 sequence $\rho_{t:t+k}$ 中出现大于阈值 $\delta$ 的 value drop，就认为 teleoperator 在此犯了错 (如犹豫、掉落物体)，这些 transitions 被标记为 suboptimal 并从训练集中剔除。

#### Stage 2: Morphological Symmetry Augmentation

这是一个极其简单但有效的 trick，专门针对 bimanual manipulation。
*   **Images:** 水平翻转所有 image observations，并交换左右手腕相机的图像。
*   **States & Actions:** $\mathbf{s}_t$ 和 $\mathbf{a}_t$ 通过 world frame 的 mirror symmetry 进行转换，然后再转回 local wrist frames。
*   **Language:** 翻转空间描述，例如 "the hole on the left" 变成 "the hole on the right"。
*   **Intuition:** 利用机器人物理形态的对称性，免费且合理地将数据量翻倍，强迫模型学习对称的 spatial representation，极大提升了 generalization。

#### Stage 3: Online Steering for Policy Deployment Alignment

这个阶段解决 Train-Inference Mismatch。采用 Offline-to-Online RL。由于任务需要毫米级精度，在 raw action space 加噪声进行探索几乎不可能成功。因此，GR-RL 在 **latent space** (即 Flow Matching DiT 的初始噪声 $\epsilon_t$ 空间) 进行 structured exploration。

引入一个轻量级 **Noise Predictor $\pi_{\theta'}$** (51.5M params)，接在 VLM backbone 之后，预测初始噪声 $\epsilon_t$。

**Policy Loss (公式 3):**
$$\mathcal{L}(\pi_{\theta'}) = \mathbb{E}_{(\dots) \sim \mathcal{D}} \left[ -Q_{\phi'}(\mathbf{o}_t, l, \mathbf{s}_t, \epsilon_t) + c \max\left(\frac{1}{2}\|\epsilon_t\|^2 - \beta, 0\right) \right], \quad \epsilon_t \sim \pi_{\theta'}(\mathbf{o}_t, l, \mathbf{s}_t)$$
*   $Q_{\phi'}(\dots, \epsilon_t)$: Noise space 的 Q-function。第一项鼓励 predictor 寻找高 return 的 noise。
*   $c$: Penalty coefficient。
*   $\|\epsilon_t\|^2$: Noise 的 L2 范数。
*   $\beta$: Divergence threshold。
*   **Intuition:** 第二项是一个 penalty term。如果 $\pi_{\theta'}$ 输出的 noise 偏离标准正态分布太远 (即 $\frac{1}{2}\|\epsilon_t\|^2 > \beta$)，就会被惩罚。这防止了 predictor 为了盲目追求高 Q-value 而生成 out-of-distribution 的乱码 noise，从而避免了生成危险且无意义的 actions。

**Critic Loss in Noise Space (公式 4):**
$$\mathcal{L}(Q_{\phi'}) = \text{cross\_entropy}(Q_{\phi'}(\mathbf{o}_t, l, \mathbf{s}_t, \epsilon_t), Q_{\phi}(\mathbf{o}_t, l, \mathbf{s}_t, \pi_{\theta}(\mathbf{o}_t, l, \mathbf{s}_t | \epsilon_t))), \quad \epsilon_t \sim \begin{cases} \mathcal{N}(\mathbf{0}, \mathbf{1}) & \text{w.p. } 0.5 \\ \pi_{\theta'} & \text{otherwise} \end{cases}$$
*   $Q_{\phi}$: 原始 action space 的 critic (通过标准 TD3 训练)。
*   $\pi_{\theta}(\dots | \epsilon_t)$: 给定特定 noise $\epsilon_t$ 时，Flow Policy 生成的 action。
*   **Intuition:** $Q_{\phi'}$ 是通过 distillation 学习的。它试图模仿 $Q_{\phi}$ 的输出。为什么不直接用 $Q_{\phi}$ 训练 $\pi_{\theta'}$？因为通过 Flow Model (DiT) 反向传播梯度计算成本极高且不稳定。通过 distillation 一个 noise space 的 $Q_{\phi'}$，我们绕过了 backprop through the flow model 的问题。
*   **Sampling Trick:** $\epsilon_t$ 有 50% 概率来自标准正态分布 $\mathcal{N}(\mathbf{0}, \mathbf{1})$，50% 来自 $\pi_{\theta'}$。这确保了 $Q_{\phi'}$ 在整个 noise space 上有良好的 coverage，避免其只拟合了 predictor 当前输出的窄分布。

### 4. Hardware: ByteMini-v2

论文同时推出了新一代硬件 **ByteMini-v2**，一个带轮式移动底盘的双臂机器人 (7-DoF arms)。
*   **Higher Load:** Elbow actuator 峰值扭矩从 17 Nm 提升到 35 Nm，峰值负载从 1.4 kg 翻倍至 3.15 kg。
*   **Enhanced Mobility:** Chassis 投影面积缩小 (450x650 mm)，适应狭窄空间。Servo steering wheels 优化了 yaw 和 pitch 的同步调节。
*   **Wrist Design:** 采用了独特的 spherical joint 设计 (ByteWrist)，提升了 confined spaces 中的灵活性。

### 5. Experiments & Data Insights

任务：Shoe Lacing (包括 pick up, threading through eyelets, handover, pull tight)。

**Multi-stage Training Success Rate (Figure 5 解析):**

| Stage | Method | Success Rate |
| :--- | :--- | :--- |
| Baseline | GR-3 (BC on all data) | 45.7% |
| Stage 1 | + Filtered BC (Data Filtering) | 61.6% (+15.9%) |
| Stage 2 | + Morphological Symmetry Aug. | 72.7% (+11.1%) |
| Stage 3 | + Online Steering RL | 83.3% (+10.6%) |

**Failure Mode Analysis (Figure 6 解析):**
论文将任务分为四个关键阶段：Pick up shoelace -> Thread eyelet -> Handover -> Pull tight。
*   **Data Filtering** 的最大贡献在于极大减少了 **Threading** 阶段的失败率。因为人类在 threading 时最容易犹豫和犯错，Filtering 剔除了这些 bad habits。
*   **Data Augmentation** 在所有阶段都有提升，虽然幅度不大，但说明 symmetry 提升了整体 spatial robustness。
*   **Online RL** 显著提升了 **Handover** 和 **Pull tight** 的成功率。这两个阶段受 train-inference mismatch 影响最大，Online RL 闭合了这个 gap。

**Distributional vs. Non-distributional Critic (Figure 7 解析):**
对比实验中，Non-distributional critic (unbounded scalar) 在 trajectory 前期出现严重的 value over-estimation，无法正确反映 task progress。而 GR-RL 的 Distributional critic (bounded in [0,1]) 能够平滑且单调地递增，完美对齐 temporal order，验证了其 robustness。

### 6. Broader Intuition & Related Context

*   **与 $\pi_{0.6}^*$ 的对比:** Physical Intelligence 的 $\pi_{0.6}^*$ 同样使用了 distributional critic 来学习 task progress。区别在于 $\pi_{0.6}^*$ 采用 advantage-conditioned denoising，而 GR-RL 采用 filtered BC 加上 latent space steering。GR-RL 认为先把 base policy 训练好 (Filtered BC)，可以缩小 online RL 的探索空间。
*   **Latent Space Steering:** 这项技术借鉴自 Wagenmaker et al. (2025)。在 high-precision 任务中，action space 的 dimensionality 很高且对扰动极度敏感。在 $\epsilon$ space (latent space) 探索，等于在一个更平滑、更抽象的 manifold 上寻找更好的 solution，避开了 raw action space 的 combinatorial explosion。
*   **Hindsight Experience Replay (HER):** 在这里被巧妙用来生成 failed trajectories。因为收集纯粹的失败数据成本高且危险，把成功轨迹中的 "retry" 片段截断作为失败数据，是一种 zero-cost 的 negative mining。

**Reference Links:**
*   GR-RL Project Page: [https://seed.bytedance.com/gr_rl](https://seed.bytedance.com/gr_rl)
*   GR-3 Technical Report: [https://arxiv.org/abs/2507.15493](https://arxiv.org/abs/2507.15493)
*   Distributional RL (C51): [https://arxiv.org/abs/1707.06887](https://arxiv.org/abs/1707.06887)
*   TD3+BC: [https://arxiv.org/abs/2106.01345](https://arxiv.org/abs/2106.01345)
*   $\pi_0$ Flow Matching: [https://arxiv.org/abs/2410.24164](https://arxiv.org/abs/2410.24164)
*   $\pi_{0.6}^*$ (VLA learns from experience): [https://arxiv.org/abs/2511.14759](https://arxiv.org/abs/2511.14759)
*   Steering Diffusion Policy: [https://arxiv.org/abs/2506.15799](https://arxiv.org/abs/2506.15799)
