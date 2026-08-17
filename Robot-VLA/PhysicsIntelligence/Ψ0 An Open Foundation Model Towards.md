---
source_pdf: Ψ0 An Open Foundation Model Towards.pdf
paper_sha256: 3d95b1c90ff458c7c62ca329b9af04dc92f02a7826d5860c16bb1b23c95109aa
processed_at: '2026-08-13T07:06:26-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Ψ0 用人话讲

## 先说清楚要解决什么问题

做 humanoid robot 控制，最难的是**数据**。

想想看，要让一个双臂 humanoid 学会"拿起杯子 → 走两步 → 放到水槽里"这种任务，你得有示教数据。现在主流做法是 teleoperation：人穿戴设备远程操控机器人，记录下每个动作。但 humanoid 一个 episode 动辄 2000+ 步，双臂 + 灵巧手 + locomotion 联合控制，一个 teleoperator 一天能采多少 episode？成本高到让人绝望。

π0、π0.5、GR00T N1 这些方法都是这条路 — 砸钱采大量 teleop data，train 一个大 VLA 模型。结果就是只有 Physical Intelligence、NVIDIA 这种大厂玩得起。

**那有没有便宜的数据？** Human egocentric video。YouTube、EPIC-Kitchens、Ego4D、EgoDex 里有成千上万小时的人类第一人称操作视频，人手怎么抓东西、怎么倒水、怎么推门 — 全都有，免费的。

**问题来了：human video 不能直接拿来训 robot policy**。原因很本质：

1. 人的手有 21 个关节，Dex3-1 只有 3 个指头
2. 人 frame rate ~60Hz 自然运动，robot 控制 30Hz 离散
3. 人有 100+ DoF 全身自由度，Unitree G1 只有 29 DoF
4. 人的 motion dynamics 和 motor actuator 完全不同

之前的工作（EgoVLA、In-n-On）试图用一个"统一 representation" — 把 human 和 robot 都映射到同一个 task space (wrist pose + fingertip positions)，然后 co-train 一个大模型。

**作者说这个做法是 sub-optimal**。直觉上的理由：你让一个模型同时学两个根本不同的 distribution，它的 capacity 会被浪费在"bridging 两个 distribution"上，而不是学有用的东西。就像让一个人同时学体操和举重，两个都学不好。

Ψ0 的核心 insight：**解耦**。

---

## Ψ0 的核心想法：分工

把"看懂任务"和"精确执行"分开。

- **VLM（大脑）**：看图像 + 语言指令，理解"现在该做什么动作"。这部分用 human video 训，因为 human video 里有大量的 task semantics 和 visual cue。
- **Action Expert（小脑）**：给定 VLM 的理解，输出精确的关节控制信号。这部分只用 real robot data 训，因为只有 robot data 里有真实的 motor dynamics。
- **AMO RL Policy（脊髓）**：负责 lower body 平衡和 locomotion，已经训好了，不用动。

这个分层让每个 component 只学自己擅长的事，互不干扰。

类比一下：你学开车，教练告诉你"看到红灯要刹车"（VLM 学的 task understanding），但具体脚踩刹车踏板用多大力、什么角度（action expert 的 motor control），是另一回事。教练不用关心你的脚 anatomy，你也不用关心教练说的是英文还是中文。

---

## 三阶段训练，每阶段学不同的东西

### Stage 1: Pre-train VLM on human video

数据：EgoDex ~829 小时人类第一人称操作视频 + Humanoid Everyday 31 小时 robot data。

用 task-space action representation：48 DoF = 两个手腕的 9 DoF pose (3D position + 6D rotation) + 10 个 fingertip 的 3D position。这个 representation 对人和 robot 都通用，因为它描述的是"手在 3D 空间的目标"，不涉及具体 anatomy。

**关键 trade-off**: pre-train 时只预测 **single next-step action**，不预测 chunk。为什么？因为 autoregressive 生成 chunk 的计算量随 chunk size 线性增长，而 pre-train 的目的只是让 VLM 学 visual representation 和 task prior，不是学精确的 multi-step execution。预测一步够了，效率大幅提升。

用 FAST tokenizer（Pertsch et al., 2025）把 48-DoF continuous action 压成 ~20 个 discrete tokens，然后 VLM autoregressive 预测：
$$p_\theta(\mathbf{a}) = \prod_{t=1}^{N} p_\theta(\mathbf{a}_t \mid \mathbf{a}_{<t}, \ell, \mathbf{o}_t)$$
- $\mathbf{a}_t$: 第 $t$ 个 action token
- $\mathbf{a}_{<t}$: 前面已经生成的 tokens（causal）
- $\ell$: 语言指令
- $\mathbf{o}_t$: 当前图像 + 状态

训练 64 个 A100 跑 10 天，230k steps。最后 VLM 学到了"看到咖啡机 → 伸手到合适位置"这种 task prior。

### Stage 2: Post-train Action Expert on robot data

冻住 VLM，只训一个 ~500M 参数的 action expert。

Action representation 切换到 **joint-space 36 DoF**：14 hand joints + 14 arm joints + 3 torso rpy + 1 base height + 2 linear velocity + 1 yaw velocity + 1 target yaw。前 28 维是真实关节角，后 8 维是给 AMO 的高级 locomotion 命令。

**为什么不继续用 task space？** 因为 task space 推理时需要 inverse kinematics (IK) 解出关节角，而 IK 在高 DoF dexterous manipulation 里经常失败（多解、singularity、near-boundary）。直接在 joint space 输出，绕开 IK。

Action expert 是 MM-DiT（Multi-Modal Diffusion Transformer），灵感来自 Stable Diffusion 3。训练用 flow matching：
$$\mathcal{L}_{fm} = \mathbb{E}\left[\left\| v_\rho^{flow}(\mathbf{z}_t, \mathbf{a}_t^\tau, \tau) - (\boldsymbol{\epsilon} - \mathbf{a}_t) \right\|\right]$$

讲讲这个公式里每个变量：
- $v_\rho^{flow}$: MM-DiT 网络，参数 $\rho$，预测 flow velocity
- $\mathbf{z}_t = f_\theta^{vlm}(\mathbf{o}_t, \ell)$: VLM 输出的 conditioning feature（frozen）
- $\mathbf{a}_t$: ground truth clean action
- $\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$: 高斯噪声
- $\tau \in [0,1]$: flow timestep，uniformly sampled
- $\mathbf{a}_t^\tau = \tau \mathbf{a}_t + (1-\tau)\boldsymbol{\epsilon}$: noised action，在 clean 和 noise 之间的线性插值
- $(\boldsymbol{\epsilon} - \mathbf{a}_t)$: target velocity，从 noise 指向 clean action 的方向

直觉：flow matching 就是让网络学一个 vector field，在任何点都能告诉你"往哪个方向走能到 clean action"。推理时从纯噪声开始，按这个 vector field 走几步就到 clean action。

### Stage 3: Fine-tune on specific task

每个任务 80 个 teleop episode，只 fine-tune action expert（VLM 继续 frozen），40k steps，cosine lr。这一步让模型快速适配新任务。

---

## MM-DiT 比 naive DiT 强在哪

naive DiT 做 action prediction 时，VL feature 只通过 cross-attention 条件化 action token — 单向的，action 看 VL，但 VL 不看 action。

MM-DiT 改成 joint global self-attention：action token 和 VL token 一起做 self-attention，双向交互。再用 flow timestep $\tau$ 分别 modulate 两个模态。

Fig. 3 画得很清楚：
```
Naive DiT:        MM-DiT:
VL → K/V          VL ↔ A (joint attention)
A → Q             + τ modulates both
(cross-attn)      
```

实验上 MM-DiT 一致性更好（Table I ablation）。直觉上，VL 和 action 互相都能看到对方，信息融合更深。

---

## RTC: 让 2.5B 模型能跑在 30Hz 控制环上

这是个工程上很关键的问题。VLA 模型 2.5B 参数，一次 forward pass 要 160ms。如果 naive 地"stop-think-execute"，每个 action chunk 之间机器人要 pause，whole-body control 任务里这种 pause 会直接导致不稳定甚至摔倒。

Naive 解法是提前 start next inference，切换到新 chunk。但因为 diffusion 的随机性，新 chunk 和老 chunk 之间会有 discontinuity，transition 反而更糟。

Ψ0 用 **training-time RTC**（Black et al., 2025）：

训练时随机 mask 掉 chunk 的前 $d$ 个 tokens，$d \sim \text{uniform}(0, 6)$。这些 masked tokens 不计入 loss。模型被迫学：给定前面已经执行的 clean action prefix，生成与之平滑衔接的剩余 action。

推理时：
1. 当前 chunk 在执行
2. 执行到 $t \geq s_{\min}$ 时，启动新 chunk inference
3. 新 chunk 的前几个 token = 当前 chunk 还没执行的部分（作为 clean prefix）
4. Flow denoising 生成剩余 token，保证连续

系统设计：两个异步 thread
- Control loop 30Hz，读 action buffer 发给机器人
- Inference loop 异步，160ms 算一个新 chunk，写入 buffer

类比：开车时不能完全停下来再重新规划路线，要在行驶中规划下一段路。RTC 就是让模型在"行驶中"生成下一 chunk，并且保证和当前执行的 chunk 平滑过渡。

---

## Teleoperation 系统：怎么采 30 小时高质量数据

这部分很 engineering，但很关键。数据质量决定 policy 上限。

设计目标：单人操作，whole-body 控制，locomotion 稳定。

硬件：
- PICO 4 Ultra headset：头位姿
- 2 个 wrist tracker：手腕位姿
- MANUS gloves：手指关节
- 腰部 tracker：平移速度
- 脚部 tracker：yaw 命令

控制流：
1. 头 + 2 个手腕作为 3 个 end-effector，multi-target IK 解出 arm + torso 配置
2. MANUS glove 数据 retarget 到 Dex3-1 三指手
3. 腰/脚 tracker 提供 high-level locomotion command 给 AMO RL policy，AMO 输出 15 DoF lower body joint

**为什么不用纯 end-to-end whole-body retargeting（像 TWIST2、SONIC）？** 作者发现这样会导致 foot drifting、lower body 不稳定、过多小 corrective step，污染 downstream policy learning 数据。

**为什么 MANUS glove + wrist tracker 而不是 vision-based VR hand tracking？** Vision-based tracking 有 occlusion 和 out-of-view 问题，手套+wrist tracker 更可靠。

这个 teleop 系统让单人就能操作 humanoid 完成 long-horizon dexterous manipulation，采集效率高。

---

## 实验数据说话

8 个 long-horizon real-world 任务，每个 3-5 个 sub-tasks，大多 2000+ steps @ 30Hz。每个任务 10 trials。

基线对比（Fig. 7, Table III）：

| 方法 | Pre-training 数据 | 整体成功率 |
|------|-------------------|------------|
| Diffusion Policy | 无 | ~0% |
| ACT | 无 | ~10% |
| InternVLA-M1 | RT-1 Bridge | 低 |
| EgoVLA | EgoDex + 其他 | 低 |
| H-RDT | Human manipulation enhanced | 低 |
| π0.5 | DROID + 大规模 mobile manipulation | 中等 |
| GR00T N1.6 | 3B humanoid foundation | 第二名 |
| **Ψ0** | **800h human video + 30h robot** | **第一名，超 GR00T 40%+** |

Ψ0 用 10x 更少的数据，超过 baselines 40%。这是很 striking 的结果。

Ablation（Table I）证明每个组件都重要：
- 不 pre-train，只 fine-tune action head：0/10 整体成功率
- 只 EgoDex pre-train，不 post-train：6/10
- EgoDex + HE pre-train，不 post-train：8/10
- 加 post-training：9/10
- 加 RTC：9/10（smoothness 更好）

特别值得注意：
- 只用 10% EgoDex pre-train，性能大幅下降（Table V）— pre-training 数据规模重要
- 只用 HE（不用 EgoDex）pre-train，精细 manipulation 显著退化（Table VI）— human video 提供 critical prior
- Multi-task fine-tune 反而比 single-task 差（Fig. 11）— data 有限时分散学习目标导致 underfitting

---

## 我的 takeaway

1. **"Scale the right data in the right way" 比 "scale data volume" 重要**。800h human video + 30h robot data 打败 10x+ 数据的 baseline，说明数据效率的关键是架构和训练 paradigm，不是单纯堆数据。

2. **解耦是关键 inductive bias**。VLM 学 task understanding，action expert 学 motor control，各司其职。co-training 在 heterogeneous distribution 上是浪费 capacity。

3. **Human video 是 humanoid VLA 的 scalable data source**。比 teleop data 便宜几个数量级，只要设计合适的学习框架让两种数据各司其职。

4. **Engineering matters**。RTC、teleop 系统设计、action representation 选择，这些 engineering 决定能不能 deploy 到 real robot。academic paper 经常忽略这些，但实际部署时它们是成败关键。

5. **Triple-system hierarchy 是合理设计**。VLM (high-level) → action expert (mid-level) → AMO RL policy (low-level)，每层只关心自己的抽象层次，不互相干扰。

这个工作给我的启发是：humanoid VLA 不应该盲目追求 scaling teleop data。Human video 是宝藏，关键是怎么用。Ψ0 给出了一个 concrete、open-source、可复现的方案。

参考链接：
- 项目主页: https://psi-lab.ai/Psi0
- EgoDex: https://arxiv.org/abs/2505.11709
- Humanoid Everyday: https://arxiv.org/abs/2510.08807
- FAST: https://arxiv.org/abs/2501.09747
- Qwen3-VL: https://arxiv.org/abs/2511.21631
- Stable Diffusion 3 (MM-DiT): https://arxiv.org/abs/2403.03206
- AMO: https://humanoid-amo.github.io/
- Training-time RTC: https://arxiv.org/abs/2512.05964
- Test-time RTC: https://arxiv.org/abs/2506.07339
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- GR00T N1: https://arxiv.org/abs/2503.14734
- EgoVLA: https://arxiv.org/abs/2507.12440
- In-n-On: https://arxiv.org/abs/2511.15704
- H-RDT: https://arxiv.org/abs/2507.23523
- Being-H0: https://arxiv.org/abs/2507.15597
- InternVLA-M1: https://arxiv.org/abs/2510.13778
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- ACT: https://arxiv.org/abs/2304.13705
- TWIST2: https://arxiv.org/abs/2511.02832
- SONIC: https://arxiv.org/abs/2511.07820
- 6D rotation representation: https://arxiv.org/abs/1812.07035
- Ego-Exo4D: https://arxiv.org/abs/2311.18283

---

# Ψ0 详解：面向 Universal Humanoid Loco-Manipulation 的 Open Foundation Model

## 1. 核心问题与核心 Motivation

这篇paper来自 USC PSI Lab (Yue Wang组), NVIDIA 的 Marco Pavone, 以及 WorldEngine 的 Di Huang。要理解 Ψ0 的设计动机, 需要先看清 humanoid VLA 当前面临的数据困境。

**问题的本质**：humanoid manipulation 数据稀缺到令人绝望。RT-1/2、OpenVLA、π0/π0.5、GR00T N1 这些方法依赖大规模 teleoperation data, 但 humanoid teleoperation 既昂贵又难以扩展, 特别是涉及 locomotion + manipulation 联合的 long-horizon 任务。一个真实的双臂 humanoid 完成 2000+ 步任务, 一个完整 episode 的 teleoperation 成本极高。

**naive 的解决方案**: 共训练 (co-training) 在 human egocentric video + humanoid robot data 上, 用统一的 human-centric state-action representation (例如 EgoVLA、In-n-On 的工作)。但作者强调, 这种做法是次优的, 原因是 **human 和 humanoid 之间存在 fundamental kinematic 和 motion disparities**:
- action frequency 不同
- motion dynamics 不同  
- degrees of freedom 不同
- 一个 monolithic policy 同时建模两个根本不同的 action distribution, 本质上很困难

Ψ0 的核心 insight 是:**解耦学习目标, 最大化 heterogeneous data sources 的效用**。这个 insight 落地成三阶段 training paradigm。

参考链接:
- 项目主页: https://psi-lab.ai/Psi0
- π0 paper: https://arxiv.org/abs/2410.24164
- π0.5 paper: https://arxiv.org/abs/2504.16054
- GR00T N1: https://arxiv.org/abs/2503.14734
- EgoVLA: https://arxiv.org/abs/2507.12440
- In-n-On: https://arxiv.org/abs/2511.15704

---

## 2. 整体架构: Triple-System Design

Ψ0 采用 triple-system 架构, 沿用了 π0.5 和 GR00T N1 的设计思路, 但具体实现有重要区别。

```
┌─────────────────────────────────────────────────────────────────┐
│  System-2 (Vision-Language Backbone)                            │
│  Qwen3-VL-2B-Instruct                                           │
│  Input: image I_t, language ℓ, proprioceptive state q_t         │
│  Output: hidden feature z_t = f_vlm(o_t, ℓ)                    │
└──────────────────────┬──────────────────────────────────────────┘
                       │ z_t (conditioning)
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  System-1 (Flow-based Action Expert)                            │
│  MM-DiT (~500M params), inspired by Stable Diffusion 3         │
│  Input: z_t, noised action a_t^τ, flow timestep τ              │
│  Output: velocity v_flow → action chunk a_{t:t+H} (36-DoF)     │
└──────────────────────┬──────────────────────────────────────────┘
                       │ 36-DoF action chunk
                       │ split: 28 upper + 8 lower commands
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  System-0 (RL Tracking Policy)                                 │
│  AMO [Li et al., RSS 2025]                                       │
│  Input: 8-DoF commands {torso_rpy, h_b, v_x, v_y, v_yaw, p_yaw} │
│  Output: 15-DoF lower-body joint angles q_lower ∈ R^15         │
│  (3 DoF waist + 12 DoF leg joints)                              │
└─────────────────────────────────────────────────────────────────┘
                       │
                       ▼
        Whole-body 43-DoF control: q_arm(14) + q_hand(14) + q_lower(15)
```

### Action representation 细节

整篇 paper 用了两套不同的 action representation, 这是关键的设计点:

**Task-space 48-DoF** (用于 pre-training, 对齐 human 和 robot):
$$\mathbf{a} \triangleq \{\mathbf{a}_l, \mathbf{a}_r\}, \quad \mathbf{a}_l, \mathbf{a}_r \in \mathbb{R}^{24}$$
每个 $\mathbf{a}$ 包含:
- $\mathbf{T}_{wrist} \in \mathbb{R}^9$: 9-DoF wrist pose = 3D position (3) + 6D rotation (6)
- $\mathbf{P}_{thumb}, \mathbf{P}_{index}, \mathbf{P}_{middle}, \mathbf{P}_{ring}, \mathbf{P}_{pinky} \in \mathbb{R}^3$ 各一个: 5 个 fingertip 的 3D 位置

这里 6D rotation 用的是 Zhou et al. 2019 的 continuous representation, 避免 quaternion 的不连续性和 Euler 的 Gimbal lock 问题。task-space representation 让 human hand 和 robot end-effector 在同一个空间里, 通用性最好。

**Joint-space 36-DoF** (用于 post-training 和 deployment):
$$\mathbf{a} = \{\mathbf{q}_{hand}, \mathbf{q}_{arm}, \text{torso}_{rpy}, h_b, v_x, v_y, v_{yaw}, p_{yaw}\} \in \mathbb{R}^{36}$$
- $\mathbf{q}_{hand} \in \mathbb{R}^{14}$: 两个 7-DoF dexterous hand (Dex3-1)
- $\mathbf{q}_{arm} \in \mathbb{R}^{14}$: 两个 7-DoF arm
- $\text{torso}_{rpy} \in \mathbb{R}^3$: torso 的 roll/pitch/yaw
- $h_b \in \mathbb{R}$: humanoid base height
- $v_x, v_y \in \mathbb{R}$: 水平线速度 (locomotion command)
- $v_{yaw} \in \mathbb{R}$: 绕垂直轴角速度
- $p_{yaw} \in \mathbb{R}$: target yaw rotation

**关键观察**: post-training 的 36-DoF 里只有 28 个是真实关节角度 (upper body), 剩下 8 个是给 AMO 的**高级 locomotion command** (不是直接关节角), 由 AMO RL policy 翻译成 15-DoF lower body joint angles。这个 hierarchy 让上层 policy 不必关心 lower body 的具体 motor control, 大大简化了学习问题。

---

## 3. 三阶段 Training Recipe

这是 Ψ0 最核心的贡献, 也是它跟其他 humanoid VLA 的根本区别。

### Stage 1: Pre-Training on Egocentric Human Video

**数据**:
- EgoDex (Hoque et al., 2025): ~829 hours egocentric video, human hand dexterous manipulation, 900M frames, 提供 per-frame global transformation matrices (7 spine joints + 2 arms + 21 joints per hand)
- Humanoid Everyday (Zhao et al., 2025): 31 hours, 260 diverse tasks, G1 + H1 两种 embodiment

**数据处理**:
- 动作统一到 current head-camera coordinate frame
- frame rate 上采样 3x (从 10Hz 到 30Hz)
- action 值用 1st 和 99th quantiles 归一化 (避免 extreme outliers)
- 状态输入在 pre-training 阶段省略

**FAST Tokenization** (Pertsch et al., 2025): 
这是把 continuous action 转成 discrete tokens 的关键技术。原始 FAST tokenizer 在 EgoDex 上的 reconstruction L1 loss 是 $5.83 \times 10^{-4}$, 平均 token length 2.08。作者从头训了新 tokenizer:
- 500,000 random sampled actions
- vocabulary size 2048, scale 100, action horizon 1
- L1 reconstruction loss: $1.95 \times 10^{-4}$ (改进约 3x)
- 平均 token length 13.04 (从 2.08 增加到 13.04, 用更长但更准确的 token 序列换取 reconstruction 精度)

**核心 insight**: pre-train VLM 时只需要预测 single next-step action $\mathbf{a}_t$, 不需要 chunk $\mathbf{a}_{t:t+H}$。理由是 pre-training 的目标只是学 **task-level motion priors** 和 **visual representations** aligned with downstream robotic tasks。预测 chunk 在 autoregressive 框架下计算量爆炸 (序列长度乘以 chunk size), 收益却不大。

**训练目标 (autoregressive next-action prediction)**:
$$p_\theta(\mathbf{a}) = \prod_{t=1}^{N} p_\theta(\mathbf{a}_t \mid \mathbf{a}_{<t}, \ell, \mathbf{o}_t)$$

变量含义:
- $p_\theta$: 参数为 $\theta$ 的 VLM 模型 (Qwen3-VL-2B-Instruct)
- $\mathbf{a} = (\mathbf{a}_1, \mathbf{a}_2, \ldots, \mathbf{a}_N)$: FAST 编码后的 action token sequence, $N \approx 20$
- $\mathbf{a}_{<t}$: 已生成的前 $t-1$ 个 action tokens (causal/autoregressive)
- $\ell$: language instruction
- $\mathbf{o}_t$: 当前 observation (image + proprioception)

**训练细节**:
- 64 A100 GPUs, 10 days, 230k steps
- 前 200k steps 只用 EgoDex, 后 30k steps 用 Humanoid Everyday
- global batch size 1024
- learning rate: language backbone $1 \times 10^{-4}$, MM projector $1 \times 10^{-5}$, vision tower $1 \times 10^{-5}$ (constant)
- image resolution: $360 \times 240$ (从 $1920 \times 1080$ resize)
- DeepSpeed, 跟 Qwen3-VL 原始训练 setup

### Stage 2: Post-Training on Cross-Task Real Humanoid Data

**数据**: Humanoid Everyday (HE) dataset, ~3 million frames, G1 with Dex3-1 + H1 with Inspire Hand (两种 embodiment)
- 因为两种 hand 的 joint morphology 不同, 通过 reordering joint indices 对齐成统一 28-DoF representation (14 hand + 14 arm)
- 状态: 28-DoF joint positions (current frame, 不归一化)
- 为支持后续 fine-tune, 把 action 和 state padding 到 36-DoF 和 32-DoF, padding 部分对应 lower-body control signals (在 HE 里缺失)

**架构**: Multi-Modal Diffusion Transformer (MM-DiT), 灵感来自 Stable Diffusion 3 (Esser et al., ICML 2024), ~500M params。

**Flow matching 训练目标**:
$$\mathcal{L}_{fm} = \mathbb{E}\left[\left\| v_\rho^{flow}(\mathbf{z}_t, \mathbf{a}_t^\tau, \tau) - (\boldsymbol{\epsilon} - \mathbf{a}_t) \right\|\right]$$

这里需要详细拆解:
- $v_\rho^{flow}$: 参数为 $\rho$ 的 flow velocity prediction network (即 MM-DiT action expert)
- $\mathbf{z}_t = f_\theta^{vlm}(\mathbf{o}_t, \ell)$: VLM 提取的 conditioning feature (frozen, 不更新)
- $\tau \in [0, 1]$: flow timestep, uniformly sampled
- $\boldsymbol{\epsilon}$: Gaussian noise $\sim \mathcal{N}(0, I)$
- $\mathbf{a}_t^\tau = \tau \mathbf{a}_t + (1-\tau)\boldsymbol{\epsilon}$: noised action, 在 clean action $\mathbf{a}_t$ 和 noise $\boldsymbol{\epsilon}$ 之间线性插值
- target $(\boldsymbol{\epsilon} - \mathbf{a}_t)$: flow matching 的 velocity target, 表示从 noise 走向 clean action 的"直线"方向

注意这里的 target 形式 $\boldsymbol{\epsilon} - \mathbf{a}_t$ 等价于 $-(\mathbf{a}_t - \boldsymbol{\epsilon})$, 是从 $\boldsymbol{\epsilon}$ 到 $\mathbf{a}_t$ 的方向。这是 rectified flow 的标准形式, 跟 SD3、π0 一致。

**MM-DiT vs naive DiT 的区别** (Fig. 3):

```
Naive DiT:
  VL tokens ──┐
              ├──► cross-attention: A attends to VL
  A tokens  ──┘
  (VL 只是作为 cross-attention 的 K/V, 单向条件)

MM-DiT (Ψ0):
  VL tokens ──┐
              ├──► joint global self-attention (双向)
  A tokens  ──┘        + 双模态 modulation (由 τ 分别调制 VL 和 A)
  (action 和 vision-language 双向融合, 更强 conditioning)
```

具体来说, 在每个 transformer block 内, action tokens 和 VL tokens 一起做 global self-attention, 同时用 flow timestep $\tau$ 分别 modulate 两个模态。这个设计让 action expert 能更好地利用 visual cue。

**训练细节**:
- VLM backbone frozen, 只优化 action expert
- learning rate constant $1 \times 10^{-4}$
- global batch size 2048
- 30k steps, ~30 hours, 32 A100 GPUs
- image $320 \times 240$
- $\tau \in [0, 1]$ uniform sampling (验证发现跟其他 sampling 策略无显著差异)

### Stage 3: Fine-Tuning on In-Domain Teleoperation Data

- 每个 task 80 episodes teleoperation data
- global batch size 128, 40k steps per task
- cosine learning rate, initial $1 \times 10^{-4}$
- state 和 action 用 min/max normalization
- 只 fine-tune action expert, VLM backbone frozen

**关于 multi-task fine-tuning 的实验**: 作者尝试过多任务联合 fine-tune, 发现单任务性能反而下降 (Fig. 11)。hypothesis 是 multi-task training 分散学习目标, 导致 underfitting。这个发现值得注意 — 在 VLA 里 multi-task fine-tune 未必比 single-task 更好, 当 in-domain data 有限时。

---

## 4. Real-Time Action Chunking (RTC): 解决 Inference Latency

这是一个工程上极其重要的设计, 直接决定能不能在 real robot 上 smooth deploy。

**问题**: VLA 模型 size ~2.5B params, 单次 forward pass ≈ 160 ms。naive 的 "stop-think-execute" 策略会让 robot 在每个 chunk 之间 pause, 产生 visible jitter, 对 whole-body control 任务特别有害。

**Naive solution 的缺陷**: 提前 start next inference, 切换到新 chunk, 但因为 diffusion 的随机性和 chunk 间的不连续性, transition 会有 jitter, 比 pause 还糟糕。

**Ψ0 的选择: Training-Time RTC** (Black et al., 2025):

训练时随机 mask 掉 chunk 的前 $d$ 个 tokens, $d \sim \text{uniform}(0, d_{\max})$, $d_{\max} = 6$。这些 masked tokens 不计入 loss。模型被迫学习**给定 preceding clean action tokens 的条件下, 生成与 clean prefix 平滑衔接的剩余 tokens**。

数学上, 修改 Eq. 2 的 loss:
$$\mathcal{L}_{fm}^{RTC} = \mathbb{E}\left[\left\| v_\rho^{flow}(\mathbf{z}_t, \mathbf{a}_t^\tau, \tau) - (\boldsymbol{\epsilon} - \mathbf{a}_t) \right\| \cdot \mathbb{1}_{\text{not masked}}\right]$$

Inference 时:
1. 当前 chunk 还在执行
2. 当 $t \geq s_{\min}$ (执行了一部分), trigger inference
3. 新 chunk 的前几个 tokens = 当前 chunk 还没执行的部分 (作为 clean condition)
4. Flow denoising 生成剩余 tokens, 保证连续

**System 实现** (Fig. 9):
- Control Loop: 30Hz, 更新 observation, 查询 action, 发送给 client
- Inference Loop: 异步运行, 共享 action chunk、observation、timestep counter
- 当 $t \geq s_{\min}$, inference loop 启动新 chunk 计算
- 系统在 previous chunk 完成前切换到 new chunk, 避免 interruption

部署时两个异步 thread:
- Policy inference thread: 低频更新 shared action buffer
- Low-level control thread: 60Hz, 持续从 buffer 读 action, 喂给 AMO RL controller

---

## 5. Teleoperation 系统: 单人操作 Whole-Body Control

这部分解决"如何高效采集高质量 humanoid loco-manipulation 数据"的问题, 是 Ψ0 数据策略的关键支撑。

**设计哲学**: 解耦 upper-body pose tracking、dexterous manipulation、locomotion commands, 单人操作。

```
Operator Hardware:
  ├── PICO 4 Ultra headset → head pose (3 end-effector #1)
  ├── 2× wrist trackers   → wrist poses (3 end-effector #2, #3)
  ├── MANUS gloves       → 5 fingers × 2 hands joint tracking
  ├── waist tracker     → translational velocity (v_x, v_y)
  └── foot trackers     → yaw commands (v_yaw, p_yaw)

           ↓
  Multi-target IK (head + 2 wrists as end-effectors)
           ↓
  ┌────────────────────────────────┐
  │ q_arm (14 DoF) + torso_rpy (3) │
  │ + h_b (base height)             │
  └────────────────────────────────┘
           ↓
       AMO RL Policy
           ↓
  q_lower (15 DoF lower body)

Finger retargeting:
  thumb, index, middle → Dex3-1 三指 dexterous hand
```

**为什么不用纯 end-to-end whole-body retargeting (像 TWIST2、SONIC)**:
作者发现 end-to-end SMPL retargeting 容易导致:
- foot drifting
- lower body 不稳定
- 过多小的 corrective steps
这些都会污染 downstream policy learning 的数据质量。

**MANUS gloves + wrist trackers 的组合**避免 vision-based VR hand tracking 的 occlusion 和 out-of-view 问题, 提供更可靠的上肢和手部 pose 估计。

**Locomotion 命令处理**:
- 腰部 tracker 估操作者平移速度, 直接映射成 robot base translation
- 脚部 tracker 提供 yaw 信号
- 应用 clipping 和 filtering 抑制人体自然 sway 引入的噪声

---

## 6. 实验: 8 个 Long-Horizon Real-World Tasks

### 任务设计 (Fig. 6)
8 个任务, 大多 2000+ steps @ 30Hz, 真正 long-horizon, 每个任务 3-5 个 sub-tasks:

1. **Task 1**: Remove lid → turn on faucet → fill with water (4 sub-tasks: Grasp, Remove, Turn, Put)
2. **Task 2**: Spray bowl with water → wipe clean → fold up (4 sub-tasks: Grasp, Pull, Spray, Wipe, Stack)
3. **Task 3**: Pick bottle → turn around → pour into cup (4 sub-tasks: Grasp, Move, Pour, Place)
4. **Task 4**: Grab can → turn → pour onto plate → push cart forward (5 sub-tasks: Grasp, Rotate, Pour, Grab, Push)
5. **Task 5**: Push cart → grab grapes → place on plate (4 sub-tasks: Handle, Push, Grasp, Place)
6. **Task 6**: Put toy into basket → turn → hand it over (4 sub-tasks: Grasp, Hook, Walk, Hand)
7. **Task 7**: Hold lunch bag → squat down → place on table (3 sub-tasks: 需要 whole-body motion + locomotion)
8. **Task 8**: Pull out tray → turn → throw chip can into trash (4 sub-tasks: Grasp, Pull, Walk, ...)

### Evaluation Protocol
- 每个任务 10 trials
- 每个 task 80 episodes teleoperation data
- 所有 baseline 在相同数据、相同 image observation、相同 action/state representation 下 fine-tune
- Evaluator 可以在 sub-task 失败时 intervene, 让 rollout 继续, 以充分评估各 sub-task 能力
- 只有所有 sub-tasks 完成, rollout 才算 success
- 报告 per-sub-task success rate + overall success rate

### Baselines (重头戏)
作者在 reproduction 上花了大力气:

| Baseline | Pre-training | 适配方式 |
|---------|--------------|---------|
| **π0.5** | DROID + 大规模 mobile manipulation | action dim 30→36, chunk size 16, lr 1e-5→1e-4, batch 32→128 |
| **GR00T N1.6** | 3B foundation model for humanoid | 20k steps, batch 24, lr 1e-4, 3 A100 |
| **InternVLA-M1** | RT-1 Bridge + spatial reasoning | freeze VLM, fine-tune action head 30k steps |
| **H-RDT** | Large DiT 2B params | 10k steps, batch 32 |
| **EgoVLA** | EgoDex + others | 115 epochs, effective batch 16×8×4 |
| **Diffusion Policy** | ResNet-18 visual encoder | 40k steps, batch 32, 100 denoising steps |
| **ACT** | Transformer | chunk size 100, 4 enc + 1 dec layer |

### 主要结果 (Fig. 7 + Table III)

**Overall success rate**:
- Ψ0: 整体最好, 平均超过第二名 GR00T N1.6 **40% 以上**
- 用了 ~800 hours human video + 30 hours real robot data
- 对手用了 10x 以上数据

具体 per-task 数据 (Table III 节选):

**Task 1 (Remove lid, turn on faucet, fill with water)**:
| Method | Grasp | Remove | Turn | Put | Overall |
|--------|-------|--------|------|-----|---------|
| π0.5 | 4/10 | 4/10 | 8/10 | 2/10 | **2/10** |
| GR00T N1.6 | 10/10 | 3/10 | 2/10 | 3/10 | 2/10 |
| Ψ0 (Ours) | **10/10** | **10/10** | 6/10 | **10/10** | **6/10** |

**Task 6 (Put toy, walk, hand it over)**:
| Method | Grasp | Hook | Walk | Hand | Overall |
|--------|-------|------|------|------|---------|
| π0.5 | 8/10 | 9/10 | 3/10 | 3/10 | 3/10 |
| GR00T N1.6 | 7/10 | 9/10 | 8/10 | 7/10 | 4/10 |
| Ψ0 (Ours) | **9/10** | 9/10 | **10/10** | **10/10** | **9/10** |

观察: 在需要精细 manipulation 的 sub-task (Remove lid, Turn faucet) 上 Ψ0 优势最明显。π0.5 在 long-horizon 任务上整体成功率不高, GR00T N1.6 比 π0.5 好, 但仍不如 Ψ0。

### Ablation Studies (Table I, IV, V, VI)

**Ablation 1: 三阶段贡献分解** (Table I):

| Pre-training (EgoDex) | Pre-training (HE) | Post-Training (HE) | RTC | MM-DiT | DiT | Right Pick | Left Pick | Dual Carry | Overall |
|---|---|---|---|---|---|---|---|---|---|
| × | × | × | × | × | √ | 1/10 | 1/10 | 1/10 | **0/10** |
| × | × | × | × | √ | × | 9/10 | 2/10 | 3/10 | 2/10 |
| √ | × | × | × | √ | × | 8/10 | 6/10 | 6/10 | 6/10 |
| √ | √ | × | × | √ | × | 8/10 | 8/10 | 9/10 | 8/10 |
| √ | √ | √ | × | √ | × | 9/10 | 9/10 | 10/10 | 9/10 |
| √ | √ | √ | √ | √ | × | 9/10 | 9/10 | 9/10 | 9/10 |

关键发现:
1. **不 pre-train, 只 fine-tune action head**: 整体 0/10 — 原始 Qwen3-VL 完全不会 generate action tokens
2. **只 EgoDex pre-train, 不 post-train**: 6/10 — VLM 学到的 visual representation 有效
3. **EgoDex + HE pre-train, 不 post-train**: 8/10 — HE pre-train 进一步提升
4. **加 post-training**: 9/10 — joint-space 精确控制关键
5. **MM-DiT vs naive DiT**: 一致性 MM-DiT 更好

**Ablation 2: RTC 效果** (Table IV, 在 GR00T N1.6 上验证):
| GR00T-N1.6 | Pick dumpling | Pick hippo | Carry box | Overall |
|---|---|---|---|---|
| w/o RTC | 10/10 | 7/10 | 9/10 | 7/10 |
| w/ RTC | 6/10 | 7/10 | 10/10 | 6/10 |

有趣的发现: 在 GR00T 上 RTC 性能**没有显著提升, 反而 slightly 差**。作者的解释是 RTC 主要改善 smoothness, 不直接提升 success rate。在 Ψ0 上 RTC 是必要的因为 inference latency 大。

**Ablation 3: Pre-training 数据规模** (Table V):
- 用 10% EgoDex pre-train: Task 1 overall 1/10 (vs baseline 8/10), Task 2 overall 6/10 (vs baseline 7/10)
- 显示 pre-training 数据规模**显著影响** task 1 (精细 manipulation), 但对 task 2 (粗放 manipulation) 影响小

**Ablation 4: HE-only pre-training** (Table VI):
- 完全不用 EgoDex, 只用 HE pre-train: 
  - Task 1 overall 4/10 (vs 8/10) — 精细 manipulation 显著退化
  - Task 2 overall 4/10 (vs 7/10) — 也退化
- 证实了 EgoDex (human video) 的 critical role
- HE variant 在不需要 fine-grained manipulation 的 sub-task 上表现好, 但精细任务上落后

**Ablation 5: Multi-task fine-tuning** (Fig. 11):
- 联合 fine-tune 多任务: 每个单任务性能下降
- Hypothesis: data 有限时 multi-task 导致 underfitting

---

## 7. 我对 Ψ0 设计哲学的解读

### 7.1 为什么解耦是关键
传统 humanoid VLA 思路是"统一 representation + 联合训练":把 human 和 robot 数据映射到同一个 action space (通常 task space), 然后 co-train。EgoVLA、In-n-On、H-RDT 都是这条路。

Ψ0 的论点是:**co-training 在两个根本不同的 distribution 上是 sub-optimal**。具体表现:
- Human data 高频 (60+ Hz)、rich finger motion、free locomotion
- Robot data 低频 (10-30 Hz)、actuator dynamics、discrete joint control
- 一个 monolithic policy 同时拟合两者, capacity 被浪费在 distribution bridging 上

Ψ0 的解耦:
- **Pre-training (VLM only)**: 学 task semantics + visual representation, 用 task-space (human + robot 共享) action
- **Post-training (action expert only)**: 学 robot-specific dynamics, 用 joint-space action
- **Fine-tuning (action expert only)**: 学 task-specific skill

这本质上是 **modular inductive bias**: VLM 学"看到 X 应该做什么动作"的 high-level planning, action expert 学"如何精确执行"的 low-level control。

### 7.2 Pre-training 只预测 single next-step 的深意
这个细节特别有意思。autoregressive VLA 通常会预测 action chunk (像 OpenVLA 的 action chunking, 或者 π0 的 chunk)。Ψ0 在 pre-training 阶段**只用 single-step prediction**。

理由: pre-training 的目的**是学 visual representation 和 task prior**, 不是学精确的 multi-step execution。Multi-step autoregressive generation 在 VLM 框架下计算量随 chunk size 线性增长, 但收益有限 — 因为 post-training 阶段会从 VLM 拿 hidden feature, action expert 自己会学 chunk generation。

这个设计让 pre-training 计算量大幅下降, 是效率上的关键 trade-off。

### 7.3 Task-space → Joint-space 的 representation 转换
这是另一个关键设计。pre-training 用 task-space (48-DoF wrist pose + fingertip positions), post-training 用 joint-space (36-DoF joint angles + locomotion commands)。

为什么能这样切换? 因为 VLM 学的是 **task-level motion prior** — "看到咖啡机就要伸手到合适位置" — 这种 prior 在 task-space 和 joint-space 之间是 transferable 的。Action expert 在 joint-space 学习, 直接控制 motor, 避免 inverse kinematics 的不稳定。

EgoVLA 等方法用 task-space + IK inference, 但 IK 在 dexterous manipulation 中经常失败 (高 DoF, 多解, 接近 singularity)。Ψ0 完全绕开 IK, 让 action expert 直接输出 joint command。

### 7.4 Triple-System 的分工
- **System-2 (VLM)**: 学抽象、跨 embodiment 的 representation
- **System-1 (action expert)**: 学 embodiment-specific 的精细 control
- **System-0 (AMO RL policy)**: 学 lower body 的 motor control ( locomotion + balance)

这个 hierarchy 让每个 component 各司其职, 互相不干扰。System-0 是预训练好的, 不参与 VLA 训练, 大幅简化了问题。

---

## 8. 与同期工作的对比

### vs π0/π0.5
π0 是单 monolithic VLA + flow matching action head, 大量 teleoperation data 训练。π0.5 加入了 open-world generalization。两者都依赖大规模 robot data。Ψ0 的区别:
- 用 human video 代替大部分 robot data
- 解耦 pre-train 和 post-train
- 用 MM-DiT (SD3 风格) 而非 π0 的 flow transformer

### vs GR00T N1
GR00T N1 也是 triple-system 设计 (VLM + action expert + low-level controller), 但 pre-training 用大规模 real + synthetic humanoid data, 不用 human egocentric video。Ψ0 论证了 human video 更 efficient。

### vs EgoVLA
EgoVLA 直接在 EgoDex + 其他数据上 pre-train, co-train on human + robot, 用 task-space action + IK inference。Ψ0 在实验中大幅超过 EgoVLA, 论证了解耦的优越性。

### vs H-RDT
H-RDT 用 2B DiT action expert, 从 human manipulation enhanced, 但还是单一 monolithic policy。Ψ0 显示 MM-DiT 优于 naive DiT (Table I)。

### vs Being-H0
Being-H0 从 human video pre-train VLM, 但限定 single-arm tabletop manipulation。Ψ0 扩展到 whole-body loco-manipulation。

---

## 9. Limitations 和未来方向

作者承认的 limitation:
1. **Compute constraint**: 无法进一步 scale to 更大的 human video + robot data
2. **Hardware constraint**: Unitree G1 payload 限制, 制约更复杂 manipulation

我认为还有一些值得思考的方向:
1. **Long-horizon planning**: 当前每个 task 3-5 个 sub-tasks, 是否能扩展到 20+ steps 的 truly long-horizon?
2. **Failure recovery**: 评估 protocol 允许 evaluator intervene, 但实际部署需要 autonomous recovery
3. **Cross-embodiment generalization**: 当前固定 G1 + Dex3-1, 能否 zero-shot 迁移到其他 humanoid?
4. **Pre-training 的可扩展性**: 800 hours 看似不少, 但相比 internet-scale video 还是太小。能否用 Ego-Exo4D、Ego4D 等更大规模数据集?

---

## 10. 核心公式和变量索引

为方便查阅, 这里汇总所有关键公式:

**1. Autoregressive next-action prediction (pre-training)**:
$$p_\theta(\mathbf{a}) = \prod_{t=1}^{N} p_\theta(\mathbf{a}_t \mid \mathbf{a}_{<t}, \ell, \mathbf{o}_t)$$
- $\mathbf{a} \in \mathbb{R}^{N}$: FAST 编码后的 action token sequence, $N \approx 20$
- $\mathbf{a}_t$: 第 $t$ 个 action token
- $\mathbf{a}_{<t}$: 前 $t-1$ 个 tokens (causal context)
- $\ell$: language instruction
- $\mathbf{o}_t$: 当前 observation

**2. Flow matching loss (post-training, action expert)**:
$$\mathcal{L}_{fm} = \mathbb{E}\left[\left\| v_\rho^{flow}(\mathbf{z}_t, \mathbf{a}_t^\tau, \tau) - (\boldsymbol{\epsilon} - \mathbf{a}_t) \right\|\right]$$
- $v_\rho^{flow}$: flow velocity prediction network (MM-DiT)
- $\mathbf{z}_t = f_\theta^{vlm}(\mathbf{o}_t, \ell)$: VLM 提取的 frozen feature
- $\mathbf{a}_t^\tau = \tau \mathbf{a}_t + (1-\tau)\boldsymbol{\epsilon}$: noised action
- $\tau \in [0, 1]$: flow timestep
- $\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$: Gaussian noise
- $(\boldsymbol{\epsilon} - \mathbf{a}_t)$: flow matching velocity target

**3. RTC masking**:
$d \sim \text{uniform}(0, d_{\max})$, $d_{\max} \in [0, H-s)$
- $H$: action chunk prediction horizon
- $s$: execution horizon (每次实际执行的步数)
- masked tokens 在 loss 中被排除

---

## 11. 总结

Ψ0 的核心贡献可以概括为三句话:

1. **解耦的数据策略**: human video (pre-train VLM) + small real robot data (post-train + fine-tune action expert), 利用 heterogeneous data 各自的优势。
2. **MM-DiT action expert**: 双模态 modulation + joint global attention, 比 naive DiT 更有效地融合 VL feature 和 action generation。
3. **Training-time RTC**: 工程上让 2.5B 参数 VLA 能在 30Hz control loop 下 smooth deploy。

实验结果令人信服: **800h human video + 30h robot data, 超过 baselines (10x+ data) 40% overall success rate**。这个 data efficiency 是 VLA 领域罕见的, 说明 **"scaling the right data in the right way"** 比 "scale data volume" 更重要。

这个工作给我的启示是: humanoid VLA 不应该盲目追求 scaling robot teleoperation data, 而应该 leverage 大量已存在的 human video — 前提是设计合适的学习框架让两种数据各司其职。Ψ0 给出了一个 concrete、open-source、可复现的方案。

参考资源:
- 项目主页: https://psi-lab.ai/Psi0
- EgoDex: https://arxiv.org/abs/2505.11709
- Humanoid Everyday: https://arxiv.org/abs/2510.08807
- FAST tokenizer: https://arxiv.org/abs/2501.09747
- Qwen3-VL: https://arxiv.org/abs/2511.21631
- Stable Diffusion 3 / MM-DiT: https://arxiv.org/abs/2403.03206
- AMO: RSS 2025 (Li et al.)
- Training-time RTC: https://arxiv.org/abs/2512.05964
- Test-time RTC: https://arxiv.org/abs/2506.07339
- π0: https://arxiv.org/abs/2410.24164
- π0.5: https://arxiv.org/abs/2504.16054
- GR00T N1: https://arxiv.org/abs/2503.14734
- EgoVLA: https://arxiv.org/abs/2507.12440
- In-n-On: https://arxiv.org/abs/2511.15704
- H-RDT: https://arxiv.org/abs/2507.23523
- Being-H0: https://arxiv.org/abs/2507.15597
- InternVLA-M1: https://arxiv.org/abs/2510.13778
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- ACT: https://arxiv.org/abs/2304.13705
- 6D rotation representation (Zhou et al.): https://arxiv.org/abs/1812.07035
