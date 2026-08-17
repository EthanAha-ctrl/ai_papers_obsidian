---
source_pdf: HiWET Hierarchical World-Frame End-Effector Tracking for Long-Horizon
  Humanoid Loco-Manipulation.pdf
paper_sha256: 0482c6f4e546eb8d78c004df3f46cedb228ff030ccdb8b10548262409370c645
processed_at: '2026-08-04T23:52:49-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# HiWET 用人话说

## 一、这 paper 一句话讲清楚

让 humanoid 机器人在 world frame 下精确地用手去追一个 3D 空间里的目标点，比如让它边走边画一个圆，这个圆是定义在房间坐标系里的，不是定义在机器人肚子上的。

## 二、为啥这事难——一个生活类比

想象你戴着 VR 头显写字。头显里显示的字永远是"正的"——因为画面是相对于你的头渲染的。但旁边的人看你写出来的字歪歪扭扭，因为你头一直在动。

现在的 humanoid 机器人就处在类似困境。大多数方法告诉机器人"手相对你的胸口去这里"，这就是 **body-centric frame**。听起来很合理，但有两个麻烦：

1. **机器人走路时脚会打滑**，每一步都会让真实位置和它以为的位置差一点点。走十步累积下来，手在世界里的位置全跑偏了。这就是 **drift accumulation**。

2. **任务超出手的可达范围**。你让机器人去够远处一个点，它手伸到底也够不着。它应该主动挪步过去，但 body-centric formulation 根本不告诉它"你需要挪步"——它只知道"手相对胸口到这里"，够不到就是够不到。

HiWET 的核心 insight 就一句话：**别让机器人想"手相对胸口在哪"，让它想"手在房间里哪"**。这样机器人自然就要操心两件事——手要准、脚要配合挪到位。

## 三、他们的核心招数：分两层管

人脑就是这么干的。你伸手拿杯子的时候：
- **大脑（高层）** 想的是"杯子在房间那个角落，我得走过去，然后伸手"
- **小脑（低层）** 想的是"现在左脚支撑，右脚抬起来，右手三角肌收缩多少"

HiWET 把这个分层做成了两个 policy：

**High-level policy $\pi^H$（大脑）**：每隔 K 步思考一次"我现在该往哪个方向走、身体多高、两只手相对胸口摆到哪"。它的 action 是：

$$a_t^H = [v_b^{des}, h^{des}, {^bT_L^{des}}, {^bT_R^{des}}, \alpha_t]$$

翻译成人话：[前后左右走多快, 身体蹲多低, 左手目标位姿, 右手目标位姿, 腰部调节参数]

**Low-level policy $\pi^L$（小脑）**：每一步都跑，把上面的命令翻译成 29 个关节的目标角度，同时保证机器人不摔倒。

这种分层有个数学名字叫 **Semi-MDP**，意思是高层决策影响未来多个 step，低层决策影响当下。这种分解让两个 policy 各管各的：高层专心想"去哪"，低层专心想"怎么动"。

## 四、KMP——这个 paper 最聪明的设计

### 4.1 痛点

低层 policy 要输出 29 个关节角度，其中上身 17 个 joint（两条胳膊 14 + 腰 3）。如果让 RL 直接学这 17 维 absolute joint targets，它探索起来非常痛苦——绝大多数随机组合都是 kinematically 无效的，比如手肘往外拧个 180°，或者手跑到背后去。

### 4.2 类比

新手画画的时候，不会一上来就画最终稿。先铅笔打个草稿，再在草稿上微调。HiWET 借的就是这个思路。

他们预先训练了一个网络叫 **KMP（Kinematic Manifold Prior）**，这个网络干一件事：给它一个"手要去哪"，它返回一个**关节层面合理但可能不完美**的草稿配置 $\hat{q}_{up}$。

然后 RL policy 不学 absolute joint，只学**残差** $\Delta q_{up}$——也就是"在草稿基础上做多少修正"。最终：

$$q_{up}^{des} = \hat{q}_{up} + \Delta q_{up}$$

这个 trick 把探索空间从 17 维 absolute 压缩到了一个低维的 correction 流形上，sample efficiency 飙升。Table I 显示去掉 KMP 后手部误差从 12.4 mm 直接翻倍到 25.2 mm，variance 还放大了 5 倍——这就是这招的威力。

### 4.3 KMP 怎么训练

用 [PyRoki](https://arxiv.org/abs/2505.03728) 这个 IK 求解器离线解了一千万次 IK，把"手目标位姿 → 关节配置"这个 mapping 蒸馏进一个 ResNet。这就像把一个慢吞吞的数学家脑子里那套 IK 解法塞进一个快速神经网络。

好处：KMP 推理只要几毫秒，而 PyRoki 在 batch=4000 时延迟飞涨。RL 训练时每个 step 都要调一次 prior，慢 solver 会卡死整个训练 pipeline。

### 4.4 数据筛选的巧思

不是所有"手位姿"都该训练。比如手肘拧到背后的姿态虽然 IK 能解出来，但 manipulability 极差（手稍微动一下关节要狂转）。他们用两个指标筛选：

- **IK error**：求解器不收敛的直接扔掉
- **Manipulability index**：$w(q) = \sqrt{\det(J(q)J(q)^T)}$，这个公式里 $J(q)$ 是 Jacobian 矩阵，$J(q)J(q)^T$ 是它乘自身转置，行列式开根号衡量"末端能多灵活地往各方向动"

实验发现用 importance sampling 训的 KMP 比用 uniform cube 训的显著准——这告诉我们：**prior 的训练分布要对齐真实任务分布**，否则 prior 会浪费容量去学一些永远用不到的怪配置。

## 五、$\alpha_t$——这个 paper 第二聪明的设计

KMP 的输入里有个 scalar $\alpha_t \in [0.1, 10]$，是腰部 regularization weight。这个数被高层 policy 当 action 主动控制。

直觉：
- **$\alpha_t$ 小**（比如 0.1）：KMP 不太在乎腰稳不稳，会疯狂弯腰去够远的地方。reachability 强但 CoM 可能跑偏
- **$\alpha_t$ 大**（比如 10）：KMP 死守腰在 nominal 位置，CoM 稳但够不远

高层 policy 可以根据当下情况动态调这个数——远距离够东西时调小让腰弯起来，平衡吃紧时调大让腰锁死。

Table II 显示把 $\alpha$ 固定为 1.0 后，real-world 圆形轨迹误差从 12 mm 涨到 18 mm。这证明**让 $\alpha$ 变成 learnable action** 比 fixed hyperparameter 显著好。

这个 idea 推广一下其实就是 **"learnable hyperparameter as action dimension"**——把超参变成 policy 可以控制的维度。这种思路在 meta-RL 里也有点影子，但 HiWET 把它落地得很干净。

## 六、其他几个工程细节

### 6.1 State Estimator

机器人本体传感器测不到 base 的线速度，EEF 在 world frame 里的位姿也需要从关节编码器 + 正运动学算。他们训了一个 CNN auxiliary head 来预测这些 privileged 信息，actor 用预测值，critic 用 ground truth。

这是 robotics RL 里的标准做法，叫 **asymmetric actor-critic**。去掉这个 State Estimator 后误差从 12.4 涨到 23 mm，说明 EEF 反馈精度对补偿 locomotion 引入的振荡至关重要。

### 6.2 Importance Sampling for Command Space

训练时给 policy 的 command 不是 uniform 采样的，而是：

$$c \sim \beta \, p_{uniform}(c) + (1-\beta) \, p_{prior}(c)$$

- $\beta$：mixing weight
- $p_{uniform}$：在整个 Cartesian 体积里均匀采
- $p_{prior}$：从筛选过的数据集采

混合的目的是既覆盖 boundary case 又重点训练 functional 区域。去掉这个 IS 后误差从 12.4 涨到 16.1 mm，variance 涨到 5.3——bimanual coordination 一致性下降。

### 6.3 Knee-Aware Height Tracking

base height error 的 reward 是非对称的：

$$e_{h,t} = (h_t - h_t^{des}) \cdot \bar{w}_{knee}^{?}$$

当机器人比目标高时，用 knee flexion margin 加权（膝盖还能弯，余地大）；当比目标低时，用 knee extension margin 加权（膝盖快伸直了，余地小，要更狠惩罚防 hyper-extension 损硬件）。

这个 reward 只在零速度时激活，避免干扰走路。一个典型的 hardware-aware reward shaping。

### 6.4 Spatial Curriculum

高层 policy 训练时，world-frame 目标一开始只在机器人附近采样。等 EEF 误差低于阈值 $\epsilon_{th}$ 后才扩展采样半径：

$$R_{k+1} = R_k + \Delta R \cdot \mathbb{1}[e_{ee} < \epsilon_{th}]$$

$\mathbb{1}[\cdot]$ 是 indicator function，条件满足为 1 否则为 0。这个 curriculum 解决了 long-horizon locomotion 的 credit assignment 问题——先学近距离协调，再学远距离搬运。

## 七、实验结果讲人话

### 7.1 仿真里的核心数字

- **EEF 误差 12.4 mm**：大概一个指甲盖厚度
- **去掉 KMP 翻倍到 25.2 mm**：证明 kinematic prior 是核心
- **去掉 State Est. 退到 23 mm**：证明 EEF 反馈精度关键
- **vs HOMIE**：base 速度误差降 20%（但 HOMIE 接 joint command 不接 EEF command，没法直接比手部精度）

### 7.2 长程轨迹跟踪

让机器人画 star、heart、circle、spiral、rectangle，在 ±5 m 范围随机初始化。成功率判定：全程 EEF 平均误差 < 20 mm。HiWET 在所有 shape 上保持 < 5 mm，ablated 版本各种扭曲。

### 7.3 Base 主动挪位

8 个方向各放一个 5 m 远的目标。HiWET final positioning error **0.101 m**（10 cm），variance 最小。其他版本走起来歪歪扭扭像醉汉，因为高层在 base 和 arm 的冲突 objective 之间纠结。

### 7.4 真机部署

在 Unitree G1 上 zero-shot sim-to-real，policy 50 Hz 跑。World-frame pose 用 head-mounted [Livox Mid-360 LiDAR](https://www.livoxtech.com/mid-360) + IMU 跑 [Fast-LIO2](https://arxiv.org/abs/2203.02740) 估计，10 Hz 更新。

真机误差：圆形 12 mm，方形 15 mm。和仿真几乎一致，证明 transfer 成功。

## 八、我的几个直觉

### 8.1 这 paper 真正的 contribution

不是 12.4 mm 这个数字本身，而是两个 design pattern：

1. **Kinematic prior + residual action**：把 analytic knowledge（IK）distill 进 frozen network，然后只学残差。这种 pattern 一定会反复出现在未来 robotics RL 里
2. **Learnable scalar 作为 hierarchical interface**：$\alpha_t$ 让高层主动控制低层的 inductive bias 强度。这种"超参即 action"思想很 powerful

### 8.2 架构上的 wider pattern

HiWET 的"spatial reasoning + dynamic execution"分层和很多领域同构：
- LLM 里 "planning + token generation"
- 自动驾驶里 "route planning + motion control"  
- 传统 robotics 里 "TAMP + trajectory optimization"

Semi-MDP 给这种 two-timescale problem 提供了 principled framework。看到这种 pattern 反复出现，说明这是个 universal scaling law of hierarchical control。

### 8.3 局限也很明显

1. **没有 visual perception**——所有目标都假设是已知 world-frame pose。要进真实 unstructured environment 必须集成 visual servoing（参考 [VisualMimic](https://arxiv.org/abs/2509.20322)）
2. **KMP frozen**——robot morphology 改了要重新生成 10M 样本重训。online adaptation 或 meta-learned KMP 是显然的下一步
3. **没有 contact-rich manipulation**——没有力控，没抓取。这块可以结合 [FALCON](https://arxiv.org/abs/2505.06776) 的 force adaptation
4. **$\alpha_t$ 的 interpretation 模糊**——它 work 但没清晰物理意义，存在 reward hacking 风险
5. **Semi-MDP 的 $K$ 没讨论**——两个 timescale 的协调超参 sensitivity 没分析
6. **bimanual 长程只跑了 single-arm**——双臂 world-frame 长程还没验证

### 8.4 这 paper 让我想到的更大的话题

Prior injection 的 art：KMP 本质是把 analytic knowledge 蒸馏进 learnable module，再用 residual 保 adaptation capacity。这种 pattern 在 RL 里反复出现：
- Humanoid locomotion 用 reference motion 做 prior
- Manipulation 用 grasp pose prior
- RLHF 用 reference model 做 prior

10M IK samples 听起来多，但比 end-to-end RL 在 environment 里 interaction 几亿次便宜多了。这是 sim-to-real 的常见 trade-off：**用 offline compute 换 online sample efficiency**。

## 九、总结一句

HiWET 把 humanoid loco-manipulation 从 "body-centric" 升级到 "world-frame"，用 hierarchical decomposition + kinematic prior + residual action 给了一个工程上很干净的解法。真正 lasting 的 contribution 大概率是 KMP residual 这套 prior injection pattern，和 $\alpha_t$ 这种 learnable interface scalar 思路。下一步等别人接 visual perception 和 force control 把它推到 contact-rich 阶段。

参考链接：
- Paper PDF: 看附件
- [PyRoki](https://arxiv.org/abs/2505.03728)
- [HOMIE](https://arxiv.org/abs/2502.13013)
- [HOVER](https://arxiv.org/abs/2402.16796)
- [OmniH2O](https://arxiv.org/abs/2406.08858)
- [FALCON](https://arxiv.org/abs/2505.06776)
- [Fast-LIO2](https://arxiv.org/abs/2203.02740)
- [AMASS](https://amass.is.tue.mpg.de/)
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)
- [Unitree G1](https://www.unitree.com/g1)
- [DeepMimic](https://arxiv.org/abs/1804.02717)
- [PPO](https://arxiv.org/abs/1707.06347)
- [BeyondMimic](https://arxiv.org/abs/2508.08241)
- [VisualMimic](https://arxiv.org/abs/2509.20322)
- [DexMan](https://arxiv.org/abs/2510.08475)
- [ResNet](https://arxiv.org/abs/1512.03385)
- [Learning to Walk in Minutes](https://arxiv.org/abs/2109.11978)
- [RMA](https://arxiv.org/abs/2107.04034)
- [ExBody2](https://arxiv.org/abs/2412.13196)
- [SONIC](https://arxiv.org/abs/2511.07820)

---

# HiWET 深度解读：World-Frame End-Effector Tracking 的 Hierarchical 解法

## 一、Problem Motivation：为什么 World-Frame Tracking 才是真正的难点

这篇 paper 抓住的核心痛点非常清晰。现有 humanoid loco-manipulation 方法大多数在 **body-centric frame** 下表述任务，例如 OmniH2O [10](https://arxiv.org/abs/2406.08858)、HOMIE [2](https://arxiv.org/abs/2502.13013)、HOVER [13](https://arxiv.org/abs/2402.16796) 都将 EEF command 视为相对于 base 的目标。这种做法有两个深层问题：

1. **Drift accumulation**：legged locomotion 每步都引入小幅 slip 和 stance phase 误差，world-frame 中 base 的真实位置和 IMU/odometry 估计位置会持续偏离。如果 EEF 目标是 base-relative，那么所有这些 drift 都会原封不动地污染 world-frame 中的 EEF 轨迹。
2. **Workspace coupling**：当 task 轨迹延伸出 static reachable workspace 时，需要 base 主动移动来 reshape 可达范围。但 body-centric formulation 根本不暴露这个几何耦合——它把"base 是 passive platform"当作隐含假设。

HiWET 的核心论点就是把任务直接表述为 world-frame end-effector tracking，从而迫使 controller 显式地协调 locomotion 与 manipulation，让 base 成为 active degree of freedom。

## 二、Hierarchical Decomposition 的架构直觉

整体结构是一个 **two-timescale HRL**，对应公式 (1)：

$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}^H, \mathcal{A}^L, \mathcal{P}, \mathcal{R}, \gamma \rangle$$

其中 $\mathcal{A}^H$ 是 high-level action space（world-frame command policy $\pi^H$），$\mathcal{A}^L$ 是 low-level action space（whole-body tracking policy $\pi^L$），$\mathcal{P}$ 是 transition dynamics，$\gamma \in (0,1)$ 是 discount factor。

两层 policy 的分工是 Semi-MDP 结构：

- **High-level $\pi^H$**：每 $K$ steps 更新一次 subgoal command $u_t$，包含 $[v_b^{des}, h^{des}, {^bT_L^{des}}, {^bT_R^{des}}, \alpha_t]$，负责 global spatial reasoning
- **Low-level $\pi^L$**：每个 control step 执行一次，输出 joint position targets，负责 dynamic stability

这个分层的核心 motivation 是 **"where to go" vs "how to move"** 的解耦。把 spatial geometric reasoning 放在慢时间尺度，dynamic stabilization 放在快时间尺度，可以避免 joint-level policy 同时学习 long-horizon navigation 和 high-frequency balance，credit assignment 上更干净。

## 三、Low-Level Tracking Policy：细节逐项拆解

### 3.1 State & Action Representation

Observation（公式 2）：

$$s_t = [\omega_t, g_t, q_t, \dot{q}_t, a_{t-1}^L]$$

变量含义：
- $\omega_t \in \mathbb{R}^3$：base angular velocity 在 base frame 中表达
- $g_t \in \mathbb{R}^3$：gravity vector 在 base frame 中的投影（用于估计 base tilt）
- $q_t, \dot{q}_t$：joint positions 和 velocities
- $a_{t-1}^L$：上一步 action，promote smoothness

Command $\mathbf{u}_t$（公式 3）：

$$\mathbf{u}_t = [\mathbf{v}_b^{des}, h^{des}, {^b\mathbf{T}_L^{des}}, {^b\mathbf{T}_R^{des}}, \alpha_t]$$

- $\mathbf{v}_b^{des} = (v_x, v_y, \omega_z)$：desired base velocity，$v_x, v_y$ 是 planar linear components，$\omega_z$ 是 yaw rate
- $h^{des}$：target body height
- ${^b\mathbf{T}_{\{\cdot\}}^{des}}$：base-relative end-effector poses（包含 position + orientation，在 $SE(3)$ 中）
- $\alpha_t \in [0.1, 10]$：**waist regularization weight**，是一个 hierarchical interface scalar

### 3.2 Hybrid Action Space 的精妙之处

公式 (4)(5) 联合定义了 action space：

$$a_t^L = q_t^{des} = [q_{t,up}^{des}, q_{t,low}^{des}]$$

其中上身采用 **residual formulation**：

$$q_{t,up}^{des} = \hat{q}_{t,up} + \Delta q_{t,up}$$

- $\hat{q}_{t,up}$：KMP 输出的 kinematically consistent reference
- $\Delta q_{t,up}$：policy 输出的 residual correction

下身则直接输出 absolute joint targets $q_{t,low}^{des}$。

**为什么上身 residual、下身 absolute？** 直觉是：

- **下身**：gait generation 需要直接控制支撑相、摆动相、足部落点，没有明显可用的 kinematic prior，且 dynamic authority 必须在 joint level 完整保留
- **上身**：manipulation 的 kinematic structure 非常强（fixed base 下的 IK 在 workspace 内几乎是 well-defined mapping），用 residual learning 可以将探索空间从 17-DoF 的 absolute joint targets 压缩到一个低维 correction 流形

这种 hybrid 设计与dex / loco-manipulation 文献中的 residual policy idea 类似，例如 [FALCON](https://arxiv.org/abs/2505.06776) 用 force adaptation 但仍在 body frame，HiWET 通过 KMP 把 kinematic prior 显式植入 action space，而非仅在 reward 中软约束。

### 3.3 Kinematic Manifold Prior (KMP) 深度分析

KMP 是这篇 paper 最有 engineering 价值的设计。输入（公式 6）：

$$z_t = [{^b\mathbf{T}_L^{des}}, {^b\mathbf{T}_R^{des}}, \alpha_t]$$

输出（公式 7）：

$$\hat{q}_{t,up} = f_{KMP}(z_t)$$

$f_{KMP}$ 是 ResNet backbone（KMP-S: 4-layer, KMP-L: 5-layer）的 neural approximation，pretrained offline 后 frozen。

**关键设计点：$\alpha_t$ 的作用**

$\alpha_t \in [0.1, 10]$ 是 waist regularization weight：
- 低 $\alpha_t$：鼓励 KMP 利用 waist redundancy 扩展 effective workspace，例如弯腰去够远的点
- 高 $\alpha_t$：强制 CoM 保持在 nominal 周围，preferred for dynamic balance

这个 scalar 实际上是 high-level policy 控制 CoM 与 reachability trade-off 的接口。把 $\alpha_t$ 同时放进 low-level command $u_t$ 和 high-level action $a_t^H$，使整个 hierarchical framework 的上身 posture 由 high-level 主动调节。

**Dataset Generation**

训练数据用 [PyRoki](https://arxiv.org/abs/2505.03728)（一个 modular robot kinematic optimization toolkit）解 IK，约束包含 joint limits 和 waist regularization weighted by $\alpha$。大约 **10M samples**。

筛选策略：
1. **IK reconstruction error**：prune 掉 IK solver 不收敛的 unreachable poses
2. **Manipulability index**：$w(q) = \sqrt{\det(J(q)J(q)^T)}$，favor 高 manipulability configurations

最终 dataset 覆盖了从 "rigid-torso reaching" 到 "whole-body lunge" 的连续 spectrum，这种 spectrum 对应 humanoid manipulation 的 real operating regime。

**Efficiency vs Accuracy**

从 Fig. 6 的 benchmarking 看：
- vs PyRoki (5 iter)：KMP-L 在 single sample 推理时约 **5× speedup**
- 在 batch=4000 时，KMP-L 保持 millisecond-level latency，而 PyRoki 的 optimization cost 急剧上升
- 在 [AMASS](https://amass.is.tue.mpg.de/) retargeted test set 上：KMP-L (IS) median position error **< 15 mm**，orientation error **< 5°**
- 关键发现：**importance-sampled dataset 训练的 KMP 显著优于 uniform cube 训练的版本**——这印证了 workspace distribution matters for kinematic prior learning

这个效率优势使 KMP 可以无缝嵌入 RL training loop（每个 step 都要 forward 一次提供 reference），而 IK solver 在大规模 parallel sim 中会成 bottleneck。

### 3.4 Network Architecture

Actor 输入（公式 8）：

$$o_t^{actor} = [s_t, \mathbf{u}_t, \hat{p}_t, e_t, \hat{q}_{t,up}]$$

- $\hat{p}_t$：State Estimator 推断的 privileged info（base linear velocity + EEF poses）
- $e_t$：History Encoder 输出的 temporal latent
- $\hat{q}_{t,up}$：KMP reference

Critic 输入（公式 9）：

$$o_t^{critic} = [s_t, \mathbf{u}_t, \mathbf{p}_t, \mathbf{h}_t, \hat{q}_{t,up}]$$

注意 critic 用 ground truth privileged $\mathbf{p}_t$ 和 height map $\mathbf{h}_t$，这是 asymmetric actor-critic 的标准做法，降低 value estimation variance。

**History Encoder**：CNN over $H=5$ history tuples，输出 $e_t$，编码 temporal dependencies
**State Estimator**：CNN，jointly trained via MSE loss，predicts base linear velocity 和 EEF poses——这些在 deployment 时 onboard sensors 无法直接测得（Unitree G1 没有 base velocity sensor，EEF pose 需要从 joint encoders + FK 重建）

### 3.5 Importance Sampling for Command Space

公式 (10)：

$$c \sim \beta \, p_{uniform}(c) + (1-\beta) \, p_{prior}(c)$$

- $\beta \in [0,1]$：mixing weight
- $p_{uniform}$：在 full Cartesian volume 均匀采样
- $p_{prior}$：从 curated dataset 采样并加小扰动

这是一个 mixture distribution，避免 policy over-fit 到 curated region，同时确保训练信号集中在 functionally relevant commands 上。这种 mixture 思想与 RL 中的 **domain randomization + curriculum** 哲学一致，但应用在 command space 而非 dynamics space。

### 3.6 Reward Design 的关键 Terms

**KMP Reference Tracking**（公式 11）：

$$r_{kmp,t} = \exp\left(-\frac{1}{N_{up}\sigma_{kmp}^2}\lVert q_{t,up} - \hat{q}_{t,up}\rVert_2^2\right)$$

- $N_{up}$：upper body joint 数量
- $\sigma_{kmp}$：sensitivity，控制 reward 随 error 衰减的速度
- $\lVert \cdot \rVert_2^2$：joint space L2 error

这是一个 exponential kernel reward（高斯形式），在 RL 中常用于 tracking tasks，因为它 bounded 在 $[0,1]$ 且 gradient smooth。time-dependent scaling 在 command resampling 后启用，避免过早约束 transient response。

**Base Height Tracking with Knee Awareness**（公式 12）：

$$e_{h,t} = \begin{cases} (h_t - h_t^{des}) \cdot \bar{w}_{knee}^{flex} & \text{if } h_t > h_t^{des} \\ (h_t - h_t^{des}) \cdot \bar{w}_{knee}^{ext} & \text{if } h_t < h_t^{des} \end{cases}$$

- $\bar{w}_{knee}^{flex}$：knee flexion margin 的平均值
- $\bar{w}_{knee}^{ext}$：knee extension margin 的平均值

这个 asymmetric weighting 的直觉是：当 base 偏高时，knee 还能 flex 来收回（margin 大）；当 base 偏低时，knee 可能已经接近 fully extended（margin 小），需要更强 penalty 防止 hyper-extension 损坏硬件。这个 reward 只在 zero velocity condition 下激活，避免干扰 dynamic locomotion。

## 四、High-Level Command Policy 的设计

### 4.1 Observation & Action

Observation（公式 13）：

$$s_t^H = [s_t, {^w\mathbf{T}_b^t}, c_t]$$

- $^w\mathbf{T}_b^t \in SE(3)$：base 在 world frame 中的位姿
- $c_t$：task command（公式 14）

Task command：

$$c_t = [{^w\mathbf{T}_L^*}, {^w\mathbf{T}_R^*}, m_t]$$

- $^w\mathbf{T}_{\{\cdot\}}^* \in SE(3)$：world-frame EEF targets
- $m_t \in \{[1,0], [0,1], [1,1]\}$：binary mask 标识 active EEF，支持 unilateral 和 bilateral tasks

Critic 用 asymmetric privileged info（公式 15）：base linear velocity $\mathbf{v}_b^{w}$ + 当前 world-frame EEF poses。

Action（公式 16）：

$$a_t^H = \mathbf{u}_t = [\mathbf{v}_b^{des}, h^{des}, {^b\mathbf{T}_L^{des}}, {^b\mathbf{T}_R^{des}}, \alpha_t] \sim \pi^H(\mathbf{u}_t | s_t^H)$$

这恰好是 low-level policy 的 command input，形成一个 **structured interface**。$\alpha_t$ 被放在 action 中是关键设计：high-level policy 可以根据 global task context 和 locomotion state 主动调节 torso engagement。

### 4.2 Spatial Curriculum Strategy

公式 (17)：

$$R_{k+1} = R_k + \Delta R \cdot \mathbb{1}[e_{ee} < \epsilon_{th}]$$

- $R_k$：当前 planar command range
- $\Delta R$：range increment
- $\epsilon_{th}$：error threshold
- $\mathbb{1}[\cdot]$：indicator function

当 EEF tracking error $e_{ee}$ 低于 $\epsilon_{th}$ 时才扩展 range。这种 curriculum 解决了 long-horizon locomotion + manipulation 的 credit assignment 问题：先掌握 local coordination，再 tackle long-range transport。

### 4.3 Reward 组成

- **Workspace Optimization**：penalize base 与 active EEF targets 的 planar distance，reward base heading 与 velocity vector 对齐到 target center 方向——这鼓励 policy 主动 "steer" base 去 reshape workspace
- **Precise EEF Tracking**：直接 minimize $|^w\mathbf{T}_{\{\cdot\}} - {^w\mathbf{T}_{\{\cdot\}}^*}|$
- **Stability & Regularization**：penalize action rate 和 high-frequency EEF jitter，inactive arm 被 penalize 偏离 neutral pose

## 五、实验数据深度分析

### 5.1 Low-Level Tracking Performance (Table I)

| Method | Lin. Vel. Error (m/s) | Ang. Vel. Error (rad/s) | Height Error (m) | EE Pos. Error (mm) |
|--------|----------------------|--------------------------|------------------|---------------------|
| HiWET | 0.157±0.003 | 0.461±0.006 | 0.018±0.012 | **12.4±2.4** |
| HiWET w/o IS | 0.165±0.005 | 0.472±0.006 | 0.018±0.014 | 16.1±5.3 |
| HiWET w/o State Est. | 0.169±0.003 | 0.459±0.004 | 0.018±0.016 | 23.0±7.2 |
| HiWET w/o KMP | 0.149±0.004 | 0.423±0.005 | 0.015±0.010 | 25.2±12.8 |
| HOMIE | 0.194±0.003 | 0.451±0.006 | 0.022±0.019 | — |

关键 takeaway：

1. **KMP removal 影响最大**：EE error 翻倍到 25.2 mm，variance 增大 5×，说明 Cartesian task 在没有 kinematic guidance 时 exploration 极困难
2. **State Estimator removal 退化 ~10 mm**：accurate EEF feedback 对补偿 locomotion-induced oscillation 至关重要
3. **IS removal 退化 ~4 mm**：focus 在 functional workspace 区域提升 bimanual coordination consistency
4. vs HOMIE：base linear velocity error 降低 ~20%，但 HOMIE 接收 joint position 而非 EEF pose command，无法直接比较 EE error

### 5.2 Long-Horizon Trajectory Tracking (Fig. 3, 4)

测试 5 种 geometric trajectory：star, heart, circle, spiral, rectangle，在 ±5m 范围内随机初始化。success criterion: average EEF tracking error < 20 mm throughout。

HiWET 在所有 shape 上保持 < 5 mm error。Ablation 分析：
- **w/o KMP**：severe trajectory distortion，因为 high-level commands 把 arms 推入 kinematically ill-conditioned regions
- **w/o State Est.**：oscillation 增大，long-horizon spatial consistency 受损
- **w/ Fixed α=1.0**：large reaching task 退化，无法用 waist redundancy 调节 CoM

### 5.3 Base Mobility Assessment (Fig. 5)

8-directional base repositioning task，5m 距离。HiWET final positioning error **0.101 m**，最低 variance。其他 variant 出现 weaving 行为，反映 high-level policy 在 base 与 arm 冲突 objective 之间挣扎。

### 5.4 KMP Benchmarking (Fig. 6)

- **Latency**：vs PyRoki (5 iter)，KMP-L single-sample 5× speedup；batch=4000 时仍 millisecond-level
- **Accuracy**：KMP-L (IS) 在 AMASS retargeted test set 上 median position error < 15 mm，orientation < 5°
- **IS vs uniform**：IS 训练版本显著优于 uniform，证明 workspace distribution 影响 kinematic prior 学习

### 5.5 Real-World Deployment (Table II)

部署在 Unitree G1 上，low-level policy 在 50 Hz 运行。World-frame pose 通过 head-mounted Livox Mid-360 LiDAR + IMU 用 [Fast-LIO2](https://arxiv.org/abs/2203.02740) 估计，base position 10 Hz 更新。

| Method | Circle RMSE (m) | Square RMSE (m) |
|--------|-----------------|------------------|
| HiWET | **0.012±0.005** | **0.015±0.007** |
| HiWET w/ Fixed α | 0.018±0.008 | 0.019±0.009 |
| HiWET w/o State Est | 0.024±0.011 | 0.028±0.011 |
| HiWET w/o KMP | 0.032±0.013 | 0.039±0.015 |

Real-world 误差（~12-15 mm）与 simulation（12.4 mm）几乎一致，证明 zero-shot sim-to-real transfer 成功。KMP removal 在 real-world 上误差翻倍以上，进一步确认其重要性。

## 六、Intuition 与 Critique

### 6.1 为什么这个 framework work？

**Geometric decoupling 的力量**：将 world-frame reasoning 与 dynamic execution 分开，让 high-level policy 在低维 $SE(3) \times \mathbb{R}^3$ 中操作而非 29-DoF joint space，policy search 维度从 $O(29)$ 降到 $O(10)$ 左右，sample efficiency 大幅提升。

**KMP 的双角色**：
1. **Exploration prior**：把搜索空间从 absolute joints 压缩到 residuals，本质上是在 manipulation manifold 上做 optimization
2. **Inference efficiency**：避免每 step 都调用 IK solver，使 hierarchical control 在 50 Hz 实时运行

**$\alpha_t$ 作为 hierarchical interface** 是设计上的精彩一笔。它不是单纯的 hyperparameter，而是被 high-level policy 主动控制的 action dimension，让 framework 可以根据 task context 动态 trade-off reachability vs stability。这种"learnable hyperparameter"思想与 Meta-RL 的某些 idea 异曲同工。

### 6.2 局限与可能的扩展

作者明确指出 4 个 limitations：
1. World-frame tracking 精度受 LiDAR localization 限制
2. Evaluated trajectories 规模较小
3. Bimanual global tracking 实验 only single-arm
4. Contact-rich manipulation 未探索

我额外的几个观察：

- **没有 visual perception**：所有 task 都假设 EEF target 是已知 world-frame pose，没有 perception pipeline。要扩展到真实 unstructured environment，需要集成 visual servoing 或 6-DoF pose estimation（参考 [VisualMimic](https://arxiv.org/abs/2509.20322)）
- **KMP 是 offline frozen**：如果 robot morphology 或 end-effector tool 改变，需要重新生成 10M samples 并 retrain。一个可能的改进是 online adaptation 或 meta-learned KMP
- **$\alpha_t$ 的 interpretation 比较模糊**：虽然实验证明它有效，但 $\alpha \in [0.1, 10]$ 的具体 semantics 不清晰，可能引入 reward hacking
- **Semi-MDP 的 $K$ 选择**：paper 没有讨论 $K$ 的 sensitivity，这是一个超参，影响两个 timescale 的 coordination
- **没有与 [FALCON](https://arxiv.org/abs/2505.06776) 或 [RAMBO](https://arxiv.org/abs/2505.06776) 直接比较 world-frame tracking**：baselines 主要在 body-centric methods 上比较，没有 world-frame formulation 的直接对比

### 6.3 与相关工作的联系

- **DeepMimic** [28](https://arxiv.org/abs/1804.02717)：joint-space tracking 的奠基工作，HiWET 把 task 从 joint tracking 提升到 Cartesian tracking
- **HOVER** [13](https://arxiv.org/abs/2402.16796)：unified whole-body controller，HiWET 与之的区别是 explicit world-frame formulation + hierarchical decomposition
- **OmniH2O / H2O** [10,11](https://arxiv.org/abs/2406.08858)：teleoperation 方向，HiWET 的 KMP 可以视为 teleoperation 中的 retargeting 的高效替代
- **BeyondMimic** [23](https://arxiv.org/abs/2508.08241)：diffusion-based motion generation，HiWET 的 high-level policy 可以视为 trajectory-level subgoal generator
- **FALCON** [44](https://arxiv.org/abs/2505.06776)：force-adaptive loco-manipulation，HiWET 没有显式 force control，是未来扩展方向
- **DexMan** [14](https://arxiv.org/abs/2510.08475)：floating-base manipulation，HiWET 显式建模 leg-arm coupling
- **PyRoki** [18](https://arxiv.org/abs/2505.03728)：KMP 的数据生成 backend，HiWET 展示了 IK solver → neural prior 的 distillation 路径

### 6.4 从 Karpathy 视角的延伸

这个工作让我联想到几个更广的话题：

1. **Hierarchical control 的 universal pattern**：HiWET 的 "spatial reasoning + dynamic execution" 分解与 LLM 中的 "planning + token generation"、自动驾驶中的 "route planning + motion control" 是同构的。Semi-MDP 提供了一个 principled framework 来处理这种 two-timescale problem。

2. **Prior injection 的 art**：KMP 本质上是把 analytic knowledge（IK）distill 进一个 learnable module，然后用 residual learning 保留 adaptation capacity。这种 pattern 在 RL 中反复出现：humanoid locomotion 中的 reference motion、manipulation 中的 grasp pose prior、甚至 RLHF 中的 reference model。

3. **Sample efficiency 的工程化**：10M IK samples 看似很多，但与 end-to-end RL 所需的 environment interactions 相比是 small cost。这是 sim-to-real 的一个常见 trade-off：用 offline compute 换取 online sample efficiency。

4. **Asymmetric actor-critic 在 robotics 中的普遍性**：critic 用 privileged info（ground truth velocity, height map, EEF poses）而 actor 用 estimated info，这种 design choice 在 [RMA](https://arxiv.org/abs/2107.04034)、[Learning to Walk in Minutes](https://arxiv.org/abs/2109.11978) 等工作中都有体现。HiWET 的 State Estimator 是这个 pattern 的具体实现。

## 七、可能的相关 Web References

- **Isaac Lab**: https://isaac-sim.github.io/IsaacLab/
- **Unitree G1**: https://www.unitree.com/g1
- **PyRoki**: https://arxiv.org/abs/2505.03728
- **Fast-LIO2**: https://arxiv.org/abs/2203.02740
- **AMASS Dataset**: https://amass.is.tue.mpg.de/
- **HOMIE**: https://arxiv.org/abs/2502.13013
- **OmniH2O**: https://arxiv.org/abs/2406.08858
- **HOVER**: https://arxiv.org/abs/2402.16796
- **FALCON**: https://arxiv.org/abs/2505.06776
- **DeepMimic**: https://arxiv.org/abs/1804.02717
- **PPO**: https://arxiv.org/abs/1707.06347
- **ExBody2**: https://arxiv.org/abs/2412.13196
- **BeyondMimic**: https://arxiv.org/abs/2508.08241
- **VisualMimic**: https://arxiv.org/abs/2509.20322
- **DexMan**: https://arxiv.org/abs/2510.08475
- **ResNet**: https://arxiv.org/abs/1512.03385
- **RMA (Rapid Motor Adaptation)**: https://arxiv.org/abs/2107.04034
- **Learning to Walk in Minutes**: https://arxiv.org/abs/2109.11978

## 八、总结

HiWET 在 humanoid loco-manipulation 领域做了一个相当 solid 的 contribution：把 world-frame end-effector tracking 这个 problem formulation 提到 forefront，并用 hierarchical decomposition + KMP residual learning 给出 engineering 上 clean 的解决方案。12.4 mm simulation tracking + 12-15 mm real-world tracking + zero-shot sim-to-real 是有说服力的 numbers。

真正 lasting 的贡献很可能是两个 design pattern：
1. **$\alpha_t$ 作为 hierarchical interface scalar** 控制 CoM-reachability trade-off
2. **KMP 作为 IK distillation 进 frozen prior + residual action** 的范式

这两个 pattern 在未来的 humanoid controller 设计中应该会反复出现。下一步扩展的 obvious direction 是 integrating visual perception 和 force control，让 framework 真正进入 contact-rich manipulation 阶段。
