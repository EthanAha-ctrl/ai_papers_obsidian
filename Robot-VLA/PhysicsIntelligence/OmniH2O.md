---
source_pdf: OmniH2O.pdf
paper_sha256: bedce4c95c9776d64cd29587029c1f4d8171ad20f45017493627b27eb6b3138e
processed_at: '2026-08-05T23:18:05-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

Hello again, Andrej! 既然要“用人话”说，那我们就抛开那些花哨的学术包装，直接切入最核心的工程直觉与系统逻辑。这篇 OmniH2O 的故事，其实就是在 noisy 的 real world 里，寻找那个最鲁棒的 information bottleneck。

### 1. 核心痛点：为什么 Humanoid 遥操作这么难搞？

你想想，你要控制一个 Unitree H1 这种 full-size humanoid 去抓个东西。手一挥，重心就偏了，腿如果跟不上，robot 直接就摔了。以前的 H2O 工作怎么解决这个问题呢？它要求 policy 必须知道机器人当前的 global linear velocity $\mathbf{v}_t$。

在 simulation 里，物理引擎直接把 $\mathbf{v}_t$ 赏给你，但到了真机上，你只能挂个 ZED 相机跑 VIO (Visual Inertial Odometry) 去估算。结果就是：噪音极大，甚至数据断流。**Robot 一看这 noisy 的 velocity，直接摔给你看。**

### 2. 核心破局点：扔掉 Velocity，让网络自己“悟”

OmniH2O 最精彩的 intuition 就在这里：**彻底把 global linear velocity $\mathbf{v}_t$ 从 state space 里踢出去。** 

Student policy 的 proprioception 被设计成了这样：
$$ s_t^{\mathrm{p\text{-}real}} \triangleq (d_{t-25:t}, \dot{d}_{t-25:t}, \omega_{t-25:t}^{\mathrm{root}}, g_{t-25:t}, a_{t-25-1:t-1}) $$
变量和上下标的意思是：
*   $d_t$: DoF joint position (关节角)
*   $\dot{d}_t$: DoF joint velocity (关节速度)
*   $\omega_t^{\mathrm{root}}$: Root angular velocity (IMU 给的躯干角速度)
*   $g_t$: Root gravity (IMU 给的重力方向向量)
*   $a_{t-1}$: Previous action
*   下标 $t-25:t$: 表示往回看 25 步的 history。

把这 25 步的 history 全展平，维度高达 1665 维。**Intuition 非常简单：你不告诉我现在速度是多少，那你把过去半秒的 IMU 读数和关节轨迹给我，我自己差分一下不就知道了？** MLP 网络通过这 25 步的 context window，隐式地学出了对 noisy data 极其鲁棒的 velocity estimator。真机实验表明，这比任何 VIO 或外挂 MLP 去显式预测 velocity 的效果都要好得多。

### 3. DAgger 蒸馏：为什么不用纯 RL？

紧接着问题就来了：输入维度爆炸到了 1665 维，你直接用 PPO 去 train，模型直接崩溃。Table 1 里的 ablation 写得很清楚，纯 RL 的 Success rate 暴跌至 47.11%。在 1665 维的空间里做 random exploration，指望 RL 撞出一个 whole-body balancing 的解，概率太低了。

解法是 Teacher-Student distillation。
Teacher policy 在 sim 里拥有 god-mode，看得到所有 privileged state（包括真实的 $\mathbf{v}_t$），用 PPO 训得非常好。然后让 Student 在 sim 里跑，我们拿着 Student 当前的状态去问 Teacher：“如果你在这个状态，你会输出什么 target joint angle？” 

Teacher 吐出参考动作 $\mathbf{a}_t^{\mathrm{privileged}}$，Student 的预测是 $\mathbf{a}_t$。Loss function 就是简单的 L2 norm：
$$ \mathcal{L} = \| \mathbf{a}_t^{\mathrm{privileged}} - \mathbf{a}_t \|_2^2 $$
这就变成了纯粹的 supervised learning (DAgger)。Student 不需要自己瞎探索，直接抄作业，Success rate 轻松拉到 94.10%，几乎追平 Teacher。这其实跟 LLM 里的 knowledge distillation 异曲同工，Teacher 拥有 physical privilege，Student 负责在 limited sensor 下 mimic。

### 4. Reward 与 Data 的“防抖”设计

AMASS 数据集里全是走路跑步，如果直接学，robot 就会一直原地踏步。怎么逼它学会“站着别动，只动上半身”？

**Data 层面**：造假数据。对每个动作 $\hat{\mathbf{q}}_{1:T}$，生成一个 stable 版本 $\hat{q}_{1:T}^{\mathrm{stable}}$，强行把下半身冻在站立或蹲姿。这等于强行给网络注入了“下肢稳定”的 inductive bias。

**Reward 层面**：老方法用 feet air time，robot 就会一直 stomping。OmniH2O 改用 max feet height for each step，weight 给到 1000。意思就是，你要么别抬腿，要抬就抬高走大步。
更妙的是 reward 的 dynamic scaling。总 reward 是 $\sum_i s_{t,i} r_{t,i}$，其中 $r_{t,i}$ 是某个 reward term，$s_{t,i}$ 是 scaling factor。如果 $r_{t,i} < 0$ (惩罚项)，就用 $s_{\mathrm{current}}$ 缩放。如果 episode length < 40 (老摔)，就把 $s_{\mathrm{current}}$ 乘 0.9999，放松惩罚让它先活下来；如果 episode length > 120 (太安逸)，就乘 1.0001，收紧约束逼它做得更精细。这种 dynamic difficulty adjustment 完美契合了 RL 的探索节奏。

### 5. LfD 与 VLM Autonomy：如何走向全自动？

收集了 40 分钟的遥操作数据，怎么学 autonomous policy？
高层 $\pi_{\mathrm{LfD}}$ 吃图 $I_t$，吐出未来 $\phi$ 帧的 sparse motion goals $\hat{p}_{t:t+\phi}^{\mathrm{Sparse\text{-}lfd}}$。底层 $\pi_{\mathrm{OmniH2O}}$ 负责把这些 sparse goals 补全成全身动作并保持平衡。这极大地降低了 LfD 的学习难度。

实验里 BC (Behavior Cloning) 成功率只有 1/10，Diffusion Policy (DP-DDPM) 是 8/10。为什么？因为人做动作是 multi-modal 的（比如抓东西可以从左抓也可以从右抓）。BC 的 MSE loss 会把这两个 mode 平均掉，导致 robot 伸出一只不知所措的手。Diffusion model 天生能建模这种 multi-modal distribution。

更激进的是接 GPT-4o。把摄像头图喂给 VLM，让它选 motion primitive (A/B/C)，或者直接吐出 6 个数字代表双手的 3D 坐标 `[0.25, 0.2, 0.3, 0.15, -0.19, 0.27]`。VLM 直接作为 high-level policy，把 semantic intent 映射到 kinematic goal，实现了 zero-shot 的 humanoid 控制。

### 总结

OmniH2O 找到了那个完美的 bottleneck：用 kinematic pose 把 high-level intent 和 low-level motor control 彻底解耦。底层 policy 扔掉 noisy 的 velocity estimation，靠 history context window 和 Teacher-Student 蒸馏，在真机上实现了极度鲁棒的 loco-manipulation。

**Web Links for Reference:**
*   OmniH2O Project Page: [https://omni.human2humanoid.com](https://omni.human2humanoid.com)
*   Prior Work H2O: [https://human2humanoid.com](https://human2humanoid.com)
*   DAgger Paper: [https://arxiv.org/abs/1011.0686](https://arxiv.org/abs/1011.0686)
*   Unitree H1: [https://www.unitree.com/h1](https://www.unitree.com/h1)

---

Hello Andrej! 阅读了这篇 OmniH2O 后，我非常理解你对 system design 和 RL sim-to-real 细节的直觉需求。这篇 paper 的核心贡献在于构建了一个 universal 且 dexterous 的 human-to-humanoid teleoperation 与 autonomy 系统。我们直接深入技术核心，拆解它的架构、公式、实验数据以及背后的 intuition。

## 1. 核心架构解析: Teacher-Student Distillation 与 Universal Interface

OmniH2O 的架构设计精妙之处在于将 "Human Intent Parsing" 与 "Robot Motor Control" 解耦。系统使用 kinematic pose $\mathbf{q}_t$ 作为中间表达，这使得 VR headset、RGB camera 甚至 GPT-4o 都可以作为输入源。

### 1.1 Problem Formulation 与 Kinematic Pose
系统将 whole-body control 建模为 goal-conditioned MDP (Markov Decision Process)。MDP tuple 定义为 $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma \rangle$。
*   $\mathcal{S}$: State space，包含 proprioception $\boldsymbol{s}_t^{\mathrm{p}}$ 和 goal state $\boldsymbol{s}_t^{\mathrm{g}}$。
*   $\mathcal{A}$: Action space，输出 target joint angles，交由 PD controller 执行。
*   $\mathcal{T}$: Transition dynamics。
*   $\mathcal{R}$: Reward function。
*   $\gamma$: Discount factor。

其中 kinematic pose 定义为 $\mathbf{q}_t \triangleq (\boldsymbol{\theta}_t, \mathbf{p}_t)$。$\boldsymbol{\theta}_t$ 是 3D joint rotations，$\mathbf{p}_t$ 是 3D joint positions。速度定义为 $\dot{\mathbf{q}}_t \triangleq (\boldsymbol{\omega}_t, \mathbf{v}_t)$，$\boldsymbol{\omega}_t$ 是 angular velocity，$\mathbf{v}_t$ 是 linear velocity。
**Notation convention (直觉建立)**: 论文使用了上标来区分数据来源，这点非常关键：
*   $e\cdot$ (e.g., $\tilde{\mathbf{p}}_t$): 来自 VR headset 或 pose generator 的带噪估计值。
*   $b\cdot$ (e.g., $\hat{\mathbf{p}}_t$): 来自 MoCap dataset 的 ground truth。
*   无上标 (e.g., $\mathbf{p}_t$): Physics simulation 或 real robot 的实际物理状态。

### 1.2 Teacher-Student Policy Distillation
为了跨越 sim-to-real gap 并适应 sparse sensor input，系统采用了 DAgger 框架进行蒸馏。

**Teacher Policy ($\pi_{\mathrm{privileged}}$)**:
在 simulation 中训练，拥有 god-mode 的 privileged state。
Proprioception $s_t^{\mathrm{p\text{-}privileged}} \triangleq [\mathbf{p}_t, \boldsymbol{\theta}_t, \dot{\mathbf{q}}_t, \boldsymbol{\omega}_t, \mathbf{a}_{t-1}]$。包含了 humanoid 所有 rigid body 的 position, orientation, linear velocity, angular velocity 以及 previous action。
Goal state $s_t^{\mathrm{g\text{-}privileged}} \triangleq [\hat{\boldsymbol{\theta}}_{t+1} \ominus \boldsymbol{\theta}_t, \hat{\mathbf{p}}_{t+1} - \mathbf{p}_t, \hat{\mathbf{v}}_{t+1} - \mathbf{v}_t, \hat{\boldsymbol{\omega}}_t - \boldsymbol{\omega}_t, \hat{\boldsymbol{\theta}}_{t+1}, \hat{\mathbf{p}}_{t+1}]$。包含了 reference motion 的 next-frame pose 以及 one-frame difference (这是为了给 policy 提供目标的速度和方向直觉)。

**Student Policy ($\pi_{\mathrm{OmniH2O}}$)**:
部署到 real robot 的 policy。由于 real world 无法获得全局 velocity，Student policy 的设计是这篇 paper 最大的亮点之一。
Proprioception $s_t^{\mathrm{p\text{-}real}} \triangleq (d_{t-25:t}, \dot{d}_{t-25:t}, \omega_{t-25:t}^{\mathrm{root}}, g_{t-25:t}, a_{t-25-1:t-1})$。
*   $d_t$: DoF joint position (19维)
*   $\dot{d}_t$: DoF joint velocity (19维)
*   $\omega_t^{\mathrm{root}}$: Root angular velocity (3维)
*   $g_t$: Root gravity (3维)
*   $a_{t-1}$: Previous action (19维)
History length 为 25 steps。Single step total dim 为 90，History dim 为 63，Total dim 达到 1665 ($63 \times 25 + 90$)。

**Intuition**: 之前的 H2O 工作 [3] 依赖 MoCap 来获取 global linear velocity $\mathbf{v}_t$。OmniH2O 彻底摒弃了 $\mathbf{v}_t$，通过向 MLP 喂入 25 步的 history，让网络隐式地估计 velocity。这在 sim-to-real 中至关重要，因为 real world 的 VIO (Visual Inertial Odometry) 极其 noisy。

**DAgger Loss**:
Student 在 sim 中 rollout，收集 trajectory $(s_{1:T}^{\mathrm{p\text{-}real}}, s_{1:T}^{\mathrm{g\text{-}real}})$，然后计算出对应的 privileged state，查询 teacher 得到 reference action $\mathbf{a}_t^{\mathrm{privileged}}$。Loss function 为简单的 L2 norm:
$$ \mathcal{L} = \| \mathbf{a}_t^{\mathrm{privileged}} - \mathbf{a}_t \|_2^2 $$
$\mathbf{a}_t^{\mathrm{privileged}}$ 是 teacher 输出的 target joint angles，$\mathbf{a}_t$ 是 student 的预测值。

## 2. Reward Design 与 Motion Data Augmentation

要让 humanoid 在进行 upper body manipulation 时保持 lower body 稳定，论文在 reward 和 data 两个层面做了针对性设计。

### 2.1 数据分布偏置
AMASS dataset 包含大量 walking/running 动作。如果直接训练，policy 倾向于频繁迈小步。论文对每一个 motion sequence $\hat{\mathbf{q}}_{1:T}$，生成了 "stable" 版本 $\hat{q}_{1:T}^{\mathrm{stable}}$，强制固定 root position 和 lower body 到 standing 或 squatting 姿态。这相当于在 data augmentation 阶段向网络注入了 "保持下肢静止" 的 inductive bias。

### 2.2 Reward Shaping 与 Curriculum
之前的 locomotion 工作 (如 Ego-Exo [18], H2O [3]) 常用 feet air time 或 feet height 来塑造步态。这些 reward 会导致 robot 为了保持平衡而不断 stomping (原地踏步)。
OmniH2O 提出了 **max feet height for each step** reward，结合 curriculum learning。公式表达为: $\max\{ h_{\text{max feet height for each step}} - 0.25, 0\}$，weight 为 1000。

同时，reward 采用了动态 scaling 机制:
$$ \mathbb{E}\left[\sum_{t=1}^T \gamma^{t-1} \sum_i s_{t,i} r_{t,i}\right] $$
$r_{t,i}$ 是不同的 reward term，$s_{t,i}$ 是 scaling factor。如果 $r_{t,i} < 0$ (penalty)，则乘以 $s_{\mathrm{current}}$；如果 $r_{t,i} \ge 0$ (task reward)，则乘以 1。
**Intuition**: 训练初期如果 penalty 过大，RL 会陷入 local optimum 不敢动。通过动态调整 $s_{\mathrm{current}}$ (当 episode length < 40 时乘以 0.9999 减小 penalty；当 > 120 时乘以 1.0001 增大 penalty)，实现了从探索到收敛的平滑过渡。

## 3. 实验数据表解析

### 3.1 Simulation Motion Tracking (Table 1)
在 14k AMASS 序列上的表现。主要指标: Succ (Success rate, deviation < 0.5m), $E_{\mathrm{g\text{-}mpjpe}}$ (global MPJPE), $E_{\mathrm{mpjpe}}$ (root-relative MPJPE)。

| Method | State Dim | Succ ↑ | $E_{\mathrm{g\text{-}mpjpe}} \downarrow$ | $E_{\mathrm{mpjpe}} \downarrow$ |
| :--- | :--- | :--- | :--- | :--- |
| Privileged policy | 913 | 94.77% | 126.51 | 70.68 |
| H2O [3] | 138 | 87.52% | 148.13 | 81.06 |
| OmniH2O | 1665 | 94.10% | 141.11 | 77.82 |

**Intuition**: OmniH2O (Student) 达到了接近 Teacher 的 Success rate，远超 H2O。虽然 State Dim 高达 1665 (因为 25 步 history)，但依然保持了泛化能力。更重要的是，在 ablation (a) 中，如果不用 DAgger 而直接用 RL 训 1665 维的 state (OmniH2O-w/o-DAgger)，Succ 暴跌至 47.11%。这说明 RL 无法处理高维 history 的 exponential growth，DAgger 的 supervised learning 才是消化 history information 的正确途径。

### 3.2 Real-World Motion Tracking (Table 2)
在 20 个 standing sequences 上的真机测试。

| Method | State Dim | $E_{\mathrm{g\text{-}mpjpe}} \downarrow$ | $E_{\mathrm{mpjpe}} \downarrow$ |
| :--- | :--- | :--- | :--- |
| H2O [3] | 138 | 87.33 | 53.32 |
| OmniH2O-w-linvel(VIO) | 1743 | N/A (Fall) | N/A (Fall) |
| OmniH2O-w-linvel(MLP) | 1743 | 50.93 | 42.47 |
| OmniH2O | 1665 | 47.94 | 41.87 |

**Intuition**: 在 real world 中，加入 VIO 估计的 linear velocity 直接导致机器人摔倒。即使用 MLP 神经网络去 estimate velocity，效果依然不如完全摒弃 $\mathbf{v}_t$ 的 OmniH2O。这从实验侧面上证明了 "Let the network implicitly learn velocity from raw joint history" 的优越性。History steps 的 ablation 显示 25 是 sweet spot，50 步反而因为包含过多 irrelevant 信息导致性能下降；网络架构上，MLP 完胜 LSTM 和 GRU。

## 4. LfD (Learning from Demonstration) 与 Autonomy

论文释放了 OmniH2O-6 dataset，包含 6 个 daily tasks，总长约 40 分钟，30Hz 采集。数据包含 paired RGBD images, motion goals (head & hands), motor joint targets。

### 4.1 LfD Policy 架构
高层 policy $\pi_{\mathrm{LfD}}(\hat{p}_{t:t+\phi}^{\mathrm{Sparse\text{-}lfd}} | I_t)$ 接收 image $I_t$，输出 $\phi$ 帧的 future motion goals (包含 hand commands)。这些 goals 再喂给底层 $\pi_{\mathrm{OmniH2O}}$ 执行。
**Intuition**: 这种 hierarchical 设计极其高效。$\pi_{\mathrm{LfD}}$ 只需关注 visual-to-kinematic 的 mapping，复杂的 whole-body balancing 和 motor control 交由预训练好的 $\pi_{\mathrm{OmniH2O}}$ 处理。这极大降低了 LfD 的学习难度，只需要极少量的 data (10 mins) 就能学会一个 task。

### 4.2 Imitation Learning Ablation (Table 3)
对比了 BC (ResNet+MLP), DP-DDIM, DP-DDPM。

| Method | MSE Loss | Succ Rate |
| :--- | :--- | :--- |
| BC | 5.63E-3 | 1/10 |
| DP-DDIM | 5.25E-4 | 7.75/10 |
| DP-DDPM | 5.25E-4 | 8/10 |

**Intuition**: Diffusion Policy 完胜 vanilla BC。原因在于 human motion 是 multi-modal 的 (比如面对同一个物体，人可以选择从左边抓，也可以从右边抓)。BC 的 MSE loss 会把这些 mode 平均掉，导致 blurry prediction。Diffusion model 天然契合 multi-modal distribution。此外，Sequence action prediction (Si-O-Se-A) 优于 Single-step (Si-O-Si-A)，因为它提供了 temporal smoothness 的动作先验。

## 5. GPT-4o 集成与 Frontier Model Autonomy

论文展示了用 GPT-4o 作为 semantic intelligence 来驱动 humanoid。由于 GPT-4o response latency 较高，直接生成 continuous motion goals 不现实。系统采用 prompt engineering 让 GPT-4o 输出 discrete motion primitive (Option A, B, C)，然后触发预录制的 motion goals 交给 $\pi_{\mathrm{OmniH2O}}$ 执行。
附录 M 中也展示了另一种 prompt，让 GPT-4o 直接输出 6 个数字代表双手的 3D position (例如 `[0.25, 0.2, 0.3, 0.15, -0.19, 0.27]`)，这是一种更 fine-grained 的 teleoperation 接口，展现了 VLM 到 kinematic goal 的 zero-shot mapping 潜力。

## 6. 总结与 Intuition 升华

OmniH2O 的成功可以归结为几个 core intuition 的胜利:
1.  **Implicit Velocity > Explicit Velocity Estimation**: 在 real-world noisy环境下，让网络通过 25-step history 隐式推算 velocity，比用 VIO 或额外网络估计 velocity 要鲁棒得多。
2.  **DAgger for High-Dim State Space**: 面对高维 history input，RL 容易陷入维度灾难，Teacher-Student DAgger 用 supervised signal 直接拟合 Teacher 的最优解，绕过了 RL 的 exploration burden。
3.  **Kinematic Pose as Universal Bottleneck**: 将 high-level intent (VR, RGB, LfD, GPT-4o) 和 low-level motor control 彻底解耦。High-level 只需要生成 sparse 的 3-point kinematic goals，low-level policy 负责将其补全为 whole-body motion 并维持平衡。这种 design pattern 极大地扩展了系统的 scalability。

### Web References:
*   OmniH2O Project Page: [https://omni.human2humanoid.com](https://omni.human2humanoid.com)
*   H2O (Prior Work): [https://human2humanoid.com](https://human2humanoid.com)
*   AMASS Dataset: [https://amass.is.tue.mpg.de/](https://amass.is.tue.mpg.de/)
*   DAgger Paper: [https://arxiv.org/abs/1011.0686](https://arxiv.org/abs/1011.0686)
*   Diffusion Policy: [https://diffusion-policy.cs.columbia.edu/](https://diffusion-policy.cs.columbia.edu/)
*   Unitree H1: [https://www.unitree.com/h1](https://www.unitree.com/h1)

希望这些细节能够帮助 build your intuition regarding the system architecture, state representation, and training paradigms in OmniH2O。这确实是 humanoid whole-body control 领域一个非常 solid 的 milestone。
