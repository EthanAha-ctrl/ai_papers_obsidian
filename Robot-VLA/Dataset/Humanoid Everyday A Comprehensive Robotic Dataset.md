---
source_pdf: Humanoid Everyday A Comprehensive Robotic Dataset.pdf
paper_sha256: 831ca6eada80353438665e9c697b65ad91f2ddf457ae794196a8d1a668d17b6a
processed_at: '2026-08-19T11:55:06-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

用大白话来拆解这篇 paper，核心就是我们终于有了一个让 humanoid robot 干日常杂活的大规模 dataset，并且作者把目前最顶尖的 AI 模型拿过来跑了一遍，结果发现大家竟然都干得一塌糊涂。

这篇 paper 做的事情其实很直观：USC 和 Toyota Research Institute 的 team 给 Unitree G1 和 H1 humanoid robot 造了一个叫 Humanoid Everyday 的 dataset (Reference: https://humanoideveryday.github.io)。这个 dataset 里有机器人走路、叠毛巾、给人递饺子玩具、甚至插花等 260 个日常动作。为了让大家不用买几十万美元的机器人也能做实验，他们还搭了个 cloud-based evaluation platform，你可以把你的 AI model 传上去，远程控制他们实验室的实体机器人做测试。

我们深入看看背后的技术门道和为什么模型会翻车。

### 1. 遥操作收集数据：算力瓶颈与异步流水线
要教机器人干活，首先得有演示数据。人类戴 Apple Vision Pro 做动作，机器人跟着模仿。但这里有个巨大的工程瓶颈：humanoid 太复杂了。Unitree G1 加上灵巧手总共有 28 个自由度 (DoF)。当人挥手时，系统要把人手的笛卡尔空间坐标，转换成机器人 14 个关节的转动角度，这叫 Inverse Kinematics (IK)。

我们可以看 Damped Least Squares (DLS) 的核心公式：
$$ q_{t+1} = q_t + J^T (J J^T + \lambda^2 I)^{-1} (x_{target} - f(q_t)) $$
- $q_t$: 当前时刻的 joint angle vector (关节角度向量)。
- $J$: Jacobian matrix (雅可比矩阵)，描述关节微小转动对末端位姿的影响。
- $x_{target}$: Vision Pro 捕获并映射后的目标末端位姿。
- $f(q_t)$: Forward Kinematics 算出的当前末端位姿。
- $\lambda$: damping factor (阻尼系数)。

直觉上，当机器人手臂伸直接近奇异点时，$J$ 的某些奇异值会趋近于 0，导致逆矩阵爆炸，机器人的关节会瞬间产生极大的速度狂抖。公式里加上 $\lambda^2 I$ 就是为了压制这种狂抖，让矩阵始终可逆。但在 28-DoF 的双臂加躯干协调上，实时解这个数学优化极其吃 CPU。Unitree 官方的代码是同步阻塞的，一边算数学一边等硬件响应，导致延迟高达 500ms，动作一卡一卡。作者把整个 pipeline 重构成了异步 multiprocessing 架构，把 IK 计算、硬件通信、数据存储扔到不同的 process 里用 shared memory 通信，硬生生把延迟降到了 2ms，这才能收集出 30Hz 的高频平滑数据。

### 2. AI 模型大翻车：维度灾难与高频信号丢失
作者把现在最火的几个模型拿过来跑这个 dataset，包括 Diffusion Policy (DP) (Reference: https://diffusion-policy.cs.columbia.edu/), 3D Diffusion Policy (DP3) (Reference: https://3d-diffusion-policy.github.io/), OpenVLA (Reference: https://openvla.github.io/), $\pi_0$-FAST (Reference: https://fast-data.github.io/), 以及 NVIDIA 的 GR00T N1.5 (Reference: https://research.nvidia.com/labs/gear/gr00t-n1_5/)。

结果非常惨烈。原因可以拆解为以下几点：

**维度爆炸让 Diffusion 失效**
Diffusion Policy 的本质是在高维连续空间里逐步去噪。它的 loss function 是：
$$ \mathcal{L}(\theta) = \mathbb{E}_{t, \epsilon, a_K^0} \left[ || \epsilon - \epsilon_\theta(a_K^t, t, o_t) ||^2 \right] $$
- $a_K^t$: 加了噪声的 action chunk (动作序列块)。
- $\epsilon$: 随机采样的高斯噪声。
- $\epsilon_\theta$: 神经网络预测的噪声。
- $o_t$: 当前观测 (如 RGB 图像)。

在 7-DoF 的简单机械臂上，神经网络能很好地预测 $\epsilon$。但在 28-DoF 的高频数据下，动作的方差极大，去噪过程极不稳定，导致 DP 平均成功率只有 29%。

**Loco-Manipulation 摧毁了 3D Policy**
DP3 用 3D 点云作为输入，在静止状态下比 DP 强。但在 "Walk to grab door handle" 这种走路加抓取的任务上，成功率直接掉到 0%。直觉上，机器人一走，LiDAR 扫描到的点云在每一帧的世界坐标系下都在剧烈晃动。如果没有极好的 ego-motion compensation (自运动补偿)，神经网络看到的 3D 输入每一帧都在错位，根本无法对齐空间特征，直接导致模型崩溃。

**VLA Tokenizer 的水土不服**
对于 VLA model 来说，处理高频动作通常需要把连续动作离散化成 token。$\pi_0$-FAST 用的是 Discrete Cosine Transform (DCT) 进行动作压缩：
$$ X_k = \sum_{n=0}^{N-1} x_n \cos\left[\frac{\pi}{N} \left(n + \frac{1}{2}\right) k\right] $$
- $x_n$: 原始的 action sequence (动作序列)。
- $X_k$: 频域系数。
DCT 的逻辑是：机械臂动作通常很平滑，保留低频的 $X_k$ 就能还原动作。但是 humanoid 在站立或走路时，为了保持平衡，躯干和腿部会产生大量高频的微调信号。DCT 把这些高频信号当作冗余信息扔掉了，结果就是机器人解码动作时出错，甚至在运行中突然卡死。

**0% 成功率的插花任务：缺乏 Closed-Loop 误差纠正**
在 "Insert rose into vase" 这个高精度任务上，所有模型全军覆没。这暴露了 open-loop imitation learning 的致命缺陷。28-DoF 的运动学误差累积会导致末端执行器产生几厘米的漂移。要把很细的玫瑰花茎插进花瓶里，需要毫米级的精度。这些模型仅仅根据图像预测动作序列，完全缺乏 closed-loop 的 visual servoing (视觉伺服) 去动态纠正微小的误差。当前的 VLA 架构本质上是在做前馈开环控制，这在精细任务上是死路一条。

### 3. Foundation Model 的 Pretraining Prior 救了场
最终，NVIDIA 的 GR00T N1.5 拿到了 51% 的最高分。这主要归功于它在海量 humanoid 数据上做过 pretraining。这就像一个学过走路的婴儿再去学跑步，总比直接从零开始学要好。Foundation model 学到了关于 humanoid 物理动力学和常见 motion pattern 的强大 prior。Ablation study 也证明了，先在 Humanoid Everyday 这个大数据集上预训练，再针对具体任务 finetune，效果会显著提升。

总结来说，这篇 paper 像是一盆冷水，泼醒了那些觉得把 LLM 架构随便套在机器人身上就能通用的幻想。把 VLA 模型从桌面机械臂迁移到 full-body humanoid，面临着高维空间灾难、高频信号处理困难、移动观测错位等多重物理层面的挑战。未来的突破可能需要 hierarchical control architecture，把底层高频平衡控制与顶层 VLA planner 解耦，或者开发出真正具备 closed-loop 纠错能力的新一代 diffusion policies。

---

Andrej, 很高兴和你探讨这篇 paper。这篇工作来自 USC 和 Toyota Research Institute，核心贡献是提出了 Humanoid Everyday dataset，一个大规模、多模态、涵盖全身运动及人机交互的 humanoid 数据集，同时提供了一个基于云端的 evaluation platform。

### Hardware 与 Teleoperation 架构
这篇 paper 使用的硬件平台是 Unitree G1 (29-DoF，配合 7-DoF Dex3-1 灵巧手) 以及 Unitree H1 (27-DoF，配合 6-DoF INSPIRE 手)。传感器配置非常丰富，包含 Intel RealSense RGB-D camera，Livox LiDAR，且 G1 的指尖带有 tactile sensors。

数据收集的核心在于其基于 Apple Vision Pro 的遥操作 pipeline。操作者佩戴 Vision Pro，底部的摄像头捕获 wrist 和 finger 的 keypoints。手部动作通过 dex-retargeting 系统 (Reference: https://github.com/dexsuite/dex-retargeting) 映射到灵巧手。手臂动作则通过基于 Pinocchio 库 (Reference: https://stack-of-tasks.github.io/pinocchio/) 的 Inverse Kinematics (IK) 算法转化为 joint commands。

这里我们可以深入看一下 Damped Least Squares (DLS) IK 的数学形式，这有助于 build intuition 关于为什么 humanoid teleoperation 很容易产生抖动或奇异点问题：
$$ q_{t+1} = q_t + J^T (J J^T + \lambda^2 I)^{-1} (x_{target} - f(q_t)) $$
其中：
- $q_t \in \mathbb{R}^{n}$ 是当前的 joint angle vector，$n$ 代表自由度 (这里 $n=14$ 对于双臂)。
- $J \in \mathbb{R}^{6 \times n}$ 是末端笛卡尔空间到关节空间的 Jacobian matrix。
- $x_{target}$ 是由 Vision Pro 捕获并映射后的目标末端位姿。
- $f(q_t)$ 是 Forward Kinematics 计算出的当前末端位姿。
- $\lambda$ 是 damping factor。

直觉上，当机械臂接近奇异点时，$J$ 的某些奇异值趋于 0，普通的 pseudo-inverse $J^{\dagger}$ 会产生极大的关节速度导致失控。引入 $\lambda^2 I$ 保证了矩阵始终可逆，但也牺牲了部分末端追踪精度。在 14-DoF 双臂且带躯干补偿的 humanoid 上，这种 IK 求解非常吃算力，这也是他们重构 pipeline 的核心动机。

### Efficient Data Collection Pipeline
官方 Unitree teleoperation 脚本采用了 blocking synchronous IO，导致控制延迟高达 500ms。作者重新设计了 multiprocessing 架构，将 IK computation, robot joint control, 以及 IO data streaming/writing 解耦到不同的 process 和 thread 中，利用 shared memory 进行 inter-process communication (IPC)。

这种异步设计将 control delay 大幅降低到了 2ms。直觉上，30Hz 的控制频率要求每帧在 33ms 内完成处理。如果 IK 求解耗时 20ms，硬件通信阻塞 10ms，同步执行会导致实际运行频率掉到 30Hz 以下甚至更低。异步流水线使得 IK 求解器可以全速运行并预测下一步，而 IO 操作在后台并行写入 buffer，从而实现了高频的平滑遥操作。

### Dataset 结构与特性
Humanoid Everyday 包含 10.3k trajectories，超过 3 million frames，涵盖 260 个 tasks，分为 7 个 categories:
1. Basic Manipulation (抓取放置)
2. Deformable Manipulation (布料等柔形物体)
3. Articulated Manipulation (铰链结构)
4. Tool Use (工具使用)
5. High-Precision Manipulation (高精度操作)
6. Human-Robot Interaction (人机协作)
7. Loco-Manipulation (行走与操作结合)

这种设定填补了现有 dataset 如 Open X-Embodiment (Reference: https://robotics-transformer-x.github.io/) 或 DROID (Reference: https://droid-dataset.github.io/) 主要局限于固定基座机械臂的空白。Loco-Manipulation 的引入非常关键，它打破了 upper-body manipulation 和 lower-body locomotion 的孤立状态，更接近真实世界的 humanoid 应用场景。

### 实验 Results 深度解析
作者测试了多种代表性的 Imitation Learning 和 VLA models，包括 Diffusion Policy (DP) (Reference: https://diffusion-policy.cs.columbia.edu/), 3D Diffusion Policy (DP3) (Reference: https://3d-diffusion-policy.github.io/), ACT (Reference: https://tonyzhaozh.github.io/aloha/), OpenVLA (Reference: https://openvla.github.io/), $\pi_0$-FAST, $\pi_{0.5}$ (Reference: https://www.physicalintelligence.company/blog/pi0-5), 以及 NVIDIA 的 GR00T N1.5 (Reference: https://research.nvidia.com/labs/gear/gr00t-n1_5/)。

观察 Table II，几个关键现象值得探讨：

1. **维度灾难**：Humanoid G1 的总 action space 达到了 28-DoF。Diffusion Policy 的核心是在高维连续空间中去噪：
   $$ \mathcal{L}(\theta) = \mathbb{E}_{t, \epsilon, a_K^0} \left[ || \epsilon - \epsilon_\theta(a_K^t, t, o_t) ||^2 \right] $$
   其中 $a_K^t$ 是加噪的 action chunk，$o_t$ 是多模态观测。在 7-DoF 机械臂上，DP 表现优异，但在 28-DoF 空间里，去噪过程的方差极大，导致 DP 整体成功率仅为 29%。

2. **3D 点云的局限性**：DP3 在大多数静态任务上优于 DP，得益于 3D spatial representation 的鲁棒性。但是在 Loco-Manipulation ("Walk to grab door handle") 任务上，DP3 成功率跌至 0%。直觉上，当 humanoid 行走时，LiDAR 点云在 world frame 下发生剧烈的 frame-to-frame 变化，如果缺乏精确的 ego-motion compensation，点云特征的不一致性会直接破坏 policy 的 spatial grounding。

3. **VLA Tokenization 的困境**：OpenVLA 直接回归连续 action，在 30Hz 高频数据下产生大量无意义抖动。降采样到 2Hz 虽然能工作，但动作僵硬。$\pi_0$-FAST 采用 Discrete Cosine Transform (DCT) 进行 action compression：
   $$ X_k = \sum_{n=0}^{N-1} x_n \cos\left[\frac{\pi}{N} \left(n + \frac{1}{2}\right) k\right] $$
   其中 $x_n$ 是原始 action sequence，$X_k$ 是频域系数。由于大多数机械臂动作平滑，保留低频 $X_k$ 即可恢复动作。但 humanoid 的 30Hz 步态及躯干平衡控制引入了大量高频信号，DCT 压缩导致高频信息丢失，引发 decoding errors。

4. **插花任务的 0% 成功率**：在 "Insert rose into vase" 这种 High-Precision 任务上，几乎所有方法全军覆没。这暴露了当前 open-loop imitation learning 的致命缺陷：缺乏 closed-loop 的 visual servoing。28-DoF 的运动学误差累积会导致末端产生厘米级的漂移，对于毫米级精度的插入任务，仅靠前馈的神经网络预测完全无法应对。

5. **Pretrain 的力量**：GR00T N1.5 取得了 51% 的最高平均成绩。这主要归功于其在海量 humanoid 数据上的预训练。Foundation model 学到了关于 humanoid 物理动力学和常见 motion pattern 的强大 prior，从而在下游任务中展现出更好的 generalization。同时，Ablation study 也证明了在 Humanoid Everyday 上进行预训练，再进行 task-specific finetuning，能显著提升 VLA models 的性能。

### Cloud Evaluation Platform
为了让社区能够复现实验，作者搭建了一个云端 evaluation platform。研究者可以远程将 policy 部署在 USC 实验室的实体 G1/H1 上。系统持续运行了 100 分钟，仅需 3 次人工干预 (由于电机过热)。这类似于 robotics 领域的 "Kaggle"，极大降低了研究门槛，同时提供了一个 standardized benchmarking 环境 (Reference: AutoEval https://auto-eval.github.io/ 是类似的机械臂云端评估先驱)。

### Intuition Building 总结
这篇 paper 揭示了将 VLA 模型从桌面机械臂迁移到 full-body humanoid 的核心痛点：**High-Dimensional Multimodal Control**。低自由度下有效的 action representation (如简单的 binned discretization 或 DCT) 在面对 28-DoF 且包含 bipedal balancing 的高频信号时迅速崩溃。未来的研究可能需要探索 hierarchical control architectures，例如将全身平衡的 low-level controller 与 high-level VLA planner 解耦，或者开发更适合高维高频 action 的新型 tokenizers，甚至引入 closed-loop diffusion policies 来解决 High-Precision 任务中的误差累积问题。

(Reference: Humanoid Everyday Project Page https://humanoideveryday.github.io)
