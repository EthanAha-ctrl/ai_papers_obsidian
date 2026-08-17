---
source_pdf: HumanPlus Humanoid Shadowing and Imitation.pdf
paper_sha256: f0ddc02f27e791222ec5f41e3a047c9bfbc8dd70dcd58d846cad08bd18527fb3
processed_at: '2026-08-05T08:16:02-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# HumanPlus 人话版

## 一句话说清楚

人形机器人理论上能 leverage 人类数据, 但实际上没人做出来一个完整 pipeline。这篇 paper 做出来了: 用一个 camera 让人 puppet 机器人收集数据, 再用这些数据训 imitation policy 让机器人自主完成任务。**$50 camera + 40 demos → 穿鞋走路 60% 成功率**。

---

## 为什么这件事 hard?

Karpathy 你做 neural net 训练时知道, 模仿学习本质是 supervised learning: $\arg\max_\theta \sum \log p_\theta(a_t | o_t)$。 那为什么 humanoid 上没大规模跑通?

三个 gap:

### Gap 1: Control Gap

Humanoid 有 33 DoF, 状态空间高维, dynamics 复杂 (浮点 base + 接触 rich)。传统 MPC (ZMP-based walking) 需要建模, 改一个 task 就要重调, 不 scale。

**解法**: 仿真里 RL 训一个 **task-agnostic low-level policy**。给它任意 target pose, 它输出 joint torque 让机器人跟踪这个 pose。 这个 policy 学的是 "how to actuate the body", 不学 "what to do"。学一次, 用一辈子。

### Gap 2: Morphology Gap

Human 用 SMPL-X 表示: 22 个 body joints + 30 个 hand joints, 都是 3-DoF spherical。 Humanoid 只有 19 个 revolute body joints + 12 个 hand DoF (6×2)。DoF 数量不匹配, 直接模仿不行。

**解法**: Retargeting。Body: copy 对应 Euler angle。 Hand: 取每根手指 middle joint 的旋转作为 1-DoF command。 Wrist: forearm 和 hand 全局 orientation 的相对旋转投影到 1-DoF。

这 mapping 是 lossy 的 (作者承认)。但够用, 因为大部分人类 motion 的信息集中在主要 joints 上, 复杂的 finger articulation 在 6-DoF hand 上本来就是 underactuated 的简化。

### Gap 3: Data Pipeline Gap

想在 real world 用 imitation learning 学 vision-based skill, 需要 (image, action) pairs。怎么采集?

传统方案: 
- **Mocap suit**: $50k+, 限制在 lab
- **VR headset + controllers**: $250-2000, 但只控制 end-effector, 不 whole-body
- **Exoskeleton**: 贵 + 重 + 限制自由度
- **Kinesthetic teaching**: 多 operator, 物理接触让机器人 stumble

**HumanPlus 方案**: 一个 RGB camera (Razer Kiyo Pro, $50) + 两个 pretrained pose estimator (WHAM 25Hz body + HaMeR 10Hz hand) + 仿真训好的 low-level policy。 Operator 站机器人旁边 line-of-sight puppet。Cost $50, 1 个 operator, whole-body。

---

## 系统 Stack 详解

```
                  ┌─────────────────────────────────────┐
                  │     LAYER 2: Imitation (HIT)        │
                  │  binocular RGB + proprio → pose chunk│
                  │  trained on ~40 teleop demos        │
                  └─────────────────────────────────────┘
                               ↓ target poses
                  ┌─────────────────────────────────────┐
                  │  LAYER 1: Low-Level (HST)           │
                  │  proprio + target pose → joint cmd  │
                  │  trained in sim via PPO on AMASS    │
                  └─────────────────────────────────────┘
                               ↓ torques
                  ┌─────────────────────────────────────┐
                  │     HARDWARE: 33-DoF humanoid       │
                  │  Unitree H1 + Inspire hands + cams  │
                  └─────────────────────────────────────┘
                               ↑ RGB
                  ┌─────────────────────────────────────┐
                  │   TELEOP: WHAM + HaMeR + retarget   │
                  │   human operator shadows motion     │
                  └─────────────────────────────────────┘
```

---

## Low-Level Policy 细节

### 架构

Decoder-only transformer (GPT-style), context length 8, 50Hz。

每 step input:
- Proprioception $\mathbf{o}_t^{\text{prop}} = [\phi, \theta, \omega_x, \omega_y, \omega_z, q_1, ..., q_{19}, \dot{q}_1, ..., \dot{q}_{19}, a_{t-1}]$
- Target pose $\mathbf{g}_t = [v_x^{\text{tg}}, v_y^{\text{tg}}, r^{\text{tg}}, p^{\text{tg}}, \omega_{\text{yaw}}^{\text{tg}}, q_1^{\text{tg}}, ..., q_{19}^{\text{tg}}]$

Output: 19-D joint position setpoints $\hat{a}_t$ → 1000Hz PD controller → torques。

变量:
- $\phi, \theta$: base 的 roll (绕 x) 和 pitch (绕 y), 单位 radians
- $\omega_x, \omega_y, \omega_z$: base angular velocity (rad/s)
- $q_i$: 第 i 个 joint 的 angle
- $\dot{q}_i$: joint angular velocity
- $a_{t-1}$: 上一步 action (smoothness)
- $v_x^{\text{tg}}, v_y^{\text{tg}}$: target base linear velocity in xy plane
- $r^{\text{tg}}, p^{\text{tg}}$: target roll/pitch (通常为 0, upright)
- $\omega_{\text{yaw}}^{\text{tg}}$: target yaw rate (转向)
- $q_i^{\text{tg}}$: retargeted target joint angle

### PPO 训练目标

$$J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1} \gamma^t r_t\right]$$

- $\pi_\theta$: policy parameterized by $\theta$
- $\tau$: trajectory
- $\gamma \approx 0.99$: discount factor
- $r_t$: per-step reward
- $T$: episode length

### Reward 设计 (关键!)

| Term | Formula | Intuition |
|---|---|---|
| xy velocity tracking | $\exp(-\|[v_x, v_y] - [v_x^{\text{tg}}, v_y^{\text{tg}}]\|)$ | exponential kernel, bounded, smooth |
| yaw velocity tracking | $\exp(-|\omega_{\text{yaw}} - \omega_{\text{yaw}}^{\text{tg}}|)$ | 转向跟踪 |
| joint position | $-\|q - q^{\text{tg}}\|_2^2$ | quadratic penalty |
| roll/pitch + energy | $-\|[r, p] - [r^{\text{tg}}, p^{\text{tg}}]\|_2^2 - \|\tau_i\|_2$ | 保持 upright + 节能 |
| foot contact | $\mathbb{1}[c == c^{\text{tg}}]$ | 二值, 鼓励正确 contact pattern |
| foot slipping | $-\|v_{\text{feet}} \cdot \mathbb{1}[F_{\text{feet}} > 1]\|_2$ | 接触时不让脚滑 |
| alive | $1$ | 没倒就 +1 |

**为什么用 exponential 而不是 L2?** L2 reward $\exp(-\|x\|)$ 其实是 L2 penalty 的 exponential transform。好处: bounded 在 $(0, 1]$, agent 不会为了无限制减小 error 而失速。Velocity tracking 这种 "good enough 就行" 的目标适合 exponential kernel。Joint position 用 L2 因为需要精确 tracking, 线性 penalty 鼓励持续优化。

### Domain Randomization

| Param | Range | 为什么 |
|---|---|---|
| base payload | $\pm 3$ kg | 电池/传感器/工具质量变化 |
| end-effector payload | $[0, 0.5]$ kg | 抓物体后惯性变化 |
| CoM offset | $\pm 0.1$ m 三轴 | 装配误差 |
| motor strength | $[0.8, 1.1]$ | actuator model + 电压波动 |
| friction | $[0.3, 0.9]$ | 地面材质 |
| control delay | $[20, 40]$ ms | 通信 + 计算延迟 |

Motor strength $[0.8, 1.1]$ 的 intuition: 训练时随机缩放 torque command 0.8-1.1 倍, policy 必须能在 20% torque deficit 下也保持平衡。这是 sim-to-real 最关键的 trick 之一, 因为 actuator model 在仿真里永远不准。

---

## Imitation Policy 细节

### 问题: Behavior Cloning 在 high-DoF 上容易失败

为什么? 经典 BC objective:
$$\mathcal{L}_{\text{BC}} = \mathbb{E}_{(o, a^*) \sim \mathcal{D}}\left[\|f_\theta(o) - a^*\|_2^2\right]$$

问题在于 $o = (\text{image}, \text{proprio})$, action $a$ 和 proprio 高度相关 (下一步 action ≈ 上一步 action + small delta)。网络很容易学到:
$$f_\theta(o) \approx \text{proprio} + \text{small constant}$$

完全忽略 image! 因为 image features 是高维稀疏, gradient signal 弱, 而 proprio 是低维 dense。这就是 ACT 在 Wear Shoe 和 Type "AI" 任务上失败的原因 (paper Table 5 显示 ACT 在 pick up shoe 后 stuck 重复, 完全不 vision feedback)。

### HIT 的解法

**改动 1**: Decoder-only, 不用 ACT 的 CVAE encoder-decoder。

ACT 原版用 CVAE 是为了处理 multimodality (同一 observation 多种合理 action)。但 humanoid task 通常 deterministic (穿鞋就一种方式), 不需要 VAE prior。去掉 encoder 简化架构, 推理更快。

**改动 2**: Forward dynamics prediction 作为 auxiliary loss。

除了预测 action chunk $\hat{a}_{t:t+50}$, 同时预测 future image features $\hat{\phi}_{t+1:t+51}$。

$$\mathcal{L} = \underbrace{\|\hat{a}_{t:t+50} - a^*_{t:t+50}\|_2^2}_{\text{action loss}} + \lambda \underbrace{\|\hat{\phi}_{t+1:t+51} - \phi^*_{t+1:t+51}\|_2^2}_{\text{forward dynamics loss}}$$

变量:
- $\hat{a}$: predicted action (50 个 target pose)
- $a^*$: ground truth action from demonstration
- $\hat{\phi}$: predicted image features (binocular, 每个 camera 一组)
- $\phi^* = \text{ResNet}(\text{image}_{t+k})$: ground truth image features from future frames
- $\lambda$: loss weight

### 为什么 forward dynamics loss 有效?

想象网络想偷懒: $f_\theta(o_t) \approx \text{proprio}_t + \Delta$。 这样 action loss 还能很小 (因为 action 确实接近 proprio)。 但 forward dynamics loss 要求网络预测 $\phi_{t+1}$。 Proprio 不包含环境信息 (物体在哪、手在哪), 只能从 image 提取。 网络被迫学习 visual representation。

这本质是把 world model 思想塞进 behavior cloning, 类似 DreamerV3 的 latent dynamics model, 但不用于 imagination planning, 仅作 representation regularizer。

Deployment 时 $\hat{\phi}$ 直接 discard, 只用 $\hat{a}$。

### HIT 运行

- 频率: 25Hz (onboard Jetson)
- Input: binocular RGB (left/right camera, 50° downward, 160mm baseline) + proprio
- Output: 50 个 target pose chunk → async 发给 HST (50Hz)
- Image encoder: pretrained ResNet

Binocular 很关键。Table 5 显示 monocular 在 Fold Clothes (40% vs 100%) 和 Wear Shoe (0% vs 60%) 上崩盘, 因为缺乏深度信息无法做精细接触。

---

## Teleop 对比 (Table 3) 的 insight

| Method | Cost | Operators | Whole-Body | Pick→Place (s) | Stand % |
|---|---|---|---|---|---|
| Kinesthetic | $50 | 3 | ✗ | 6.60 | 90.5 |
| ALOHA | $7050 | 2-3 | ✗ | 7.15 | 100 |
| Meta Quest | $250 | 2 | ✗ | 8.87 | 95.3 |
| **Ours** | **$50** | **1** | **✓** | **5.20** | **100** |

关键观察:

**ALOHA 慢的原因**: hardware 是 fixed bimanual puppet arm, 仿照 operator 身高设计。 但 operator 身高不同时 ergonomics 差, 而且它只能控制 arm, 不能 squat/walk。 Rearrange Lower Objects (需要蹲) ALOHA 直接做不了。

**Meta Quest 慢的原因**: 5-DoF arm + 1-DoF wrist 在 Cartesian IK 经常碰 singularity (arm 完全伸直时 Jacobian rank-deficient)。 Operator 看到 robot 卡住, 调整动作, 但 robot 跟不上, 产生 destabilizing motion。 95.3% stable standing 因为偶尔摔。

**Kinesthetic**: 物理接触 + 多人协作, external force 让 robot stumble。90.5%。

**HumanPlus**: 单 camera 被动感知, operator 自由移动, whole-body 一体化控制。 唯一能完成 Rearrange Lower Objects (15.34s) 的方法。

---

## Robustness 数据 (Table 4) 的 insight

| Direction | Ours | H1 Default | 比值 |
|---|---|---|---|
| Forward | 32N | 24N | 1.33× |
| Backward | 44N | 36N | 1.22× |
| Left | 70N | 40N | 1.75× |
| Right | 100N | 40N | 2.5× |
| Recovery time | 1.2s | 15s | 12.5× |

**为什么 lateral (左右) 比 sagittal (前后) robust?**

Sagittal perturbation 让 CoM 投影到 BoS (base of support) 前后边缘, 容易触发 stepping 策略, 步态切换风险大。

Lateral perturbation 只需要 hip abduction/adduction 调整, CoM 在 BoS 内 lateral 移动空间大 (双脚分开时), 不需要 step。 所以 100N rightward 都能扛住。

**为什么 H1 Default 恢复 15s?**

H1 Default 是 model-based controller (ZMP + footstep planning)。 失衡后要: (1) 检测 fall, (2) 重新 plan footstep, (3) 调 ZMP reference, (4) 执行 stabilizing gait。 整个 pipeline 慢。

RL policy 直接从 history 推断: 输入 proprio (含历史 8 step), 输出 torque。 没有 explicit planning, end-to-end learned, 1-2 步内恢复。

**0.35m high jump + 0.44m squat**: H1 Default 完全做不到。 这是 RL policy 学到的 whole-body coordination skill, 不在 model-based controller 设计 space 内。

---

## Imitation 实验 (Table 5) 的 insight

### Wear a Shoe and Walk (最难任务, 40 demos)

| Method | Whole Task |
|---|---|
| HIT (Ours) | **60%** |
| Monocular | 0% |
| ACT | 0% |
| Open-loop | 0% |

10 个 sub-step: flip → pick → put on → press → tangle → grasp R → grasp L → tie → stand → walk。

只有 HIT 能跑完。 60% 不是高, 但考虑到 task 复杂度 (精细 bimanual + locomotion transition) 和只有 40 demos, 已经 impressive。

为什么 ACT 失败? Paper 里明确说: "ACT overfits to proprioception, robot repeatedly attempts and stuck at Pick up Shoe after successful completing them, avoiding uses visual feedback." — 即网络学到 "pick up 动作 = 当前 proprio + 小扰动", 抓到鞋后 proprio 变化小, 网络输出重复 pick 动作, 卡死。 Forward dynamics loss 强制用 vision, 解决这个 bug。

为什么 Monocular 失败? Depth 信息缺失, 抓鞋时 hand 和 shoe 的相对深度估计不准, 抓不到或抓偏。 Binocular (160mm baseline, 50° downward) 提供立体深度。

### Fold Clothes (40 demos)

| Method | Fold Left | Fold Right | Fold Bottom | Whole |
|---|---|---|---|---|
| HIT | 100 | 100 | 100 | **100** |
| Monocular | 80 | 50 | 100 | 40 |
| ACT | 100 | 100 | 100 | 100 |
| Open-loop | 20 | 50 | 0 | 0 |

HIT 和 ACT 都 100%, 因为 fold 动作相对 deterministic, proprio 信息够用。 Monocular 失败因为 depth 缺失, hand 和 table 接触粗糙。

### Type "AI" (30 demos, 8s)

| Method | Type A | Leave A | Type I | Leave I | Whole |
|---|---|---|---|---|---|
| HIT | 90 | 100 | 89 | 100 | **80** |
| Monocular | 100 | 44 | 100 | 40 | 40 |
| ACT | 30 | 20 | 0 | 60 | 0 |
| Open-loop | 82 | 100 | 79 | 100 | 60 |

有意思: Open-loop 60%, 因为 Type "AI" 无随机化, 重播就行。 但 Leave A/I (释放按键) 需要精确 force control, open-loop 100%, 说明这个 sub-action 也 deterministic。

ACT 0%, 因为坐姿 typing 完全靠 proprio 就行, 网络 overfit, 输入 vision 被忽略, 但 keyboard 位置微小扰动就崩。

HIT 80%, forward dynamics 让网络 use vision 检测 finger-keyboard 相对位置。

---

## 作者承认的 Limitations

1. **1-DoF ankle**: 限制单腿 agile motion (人类 ankle 是 2-DoF: subtalar + talocrural)
2. **5-DoF arm + 1-DoF wrist**: 6-DoF operational space control 不可达, workspace 有 hole
3. **Fixed cameras**: 头部 camera 不能主动 gaze, hand 容易出 FoV
4. **Fixed retargeting**: 大量人类 joints 信息丢弃
5. **Pose estimation 鲁棒性**: 大面积 occlusion 时 WHAM/HaMeR 失败
6. **无 long-horizon navigation**: 只做 in-place manipulation + 短距离 walking

---

## 我的几个 opinion / follow-up 方向

### 1. Forward dynamics loss 应该更强

Paper 里 $\lambda$ 没明确数值, 暗示是辅助作用。 但这其实是让 BC 不 collapse 到 proprio shortcut 的关键。 如果把 forward dynamics 当 main objective (类似 world model), action prediction 当 auxiliary, 可能 sample efficiency 更好。 参考 DreamerV3: https://arxiv.org/abs/2304.10573

### 2. Diffusion policy 可能更适合 humanoid

HIT 用 L2 regression 输出 action chunk, 假设 action distribution unimodal。 但 humanoid task (尤其 bimanual) 可能有 multimodality (同一 observation 多种合理 grasp 方式)。 Diffusion policy (https://diffusion-policy.cs.columbia.edu/) 用 diffusion objective 天然支持 multimodality。 没在 paper 里试, 但应该是 next step。

### 3. Vision encoder 可以更强

HIT 用 pretrained ResNet, 这是 2015 年的 tech。 换 CLIP/SigLIP/DINOv2 embedding, 或者直接上 VLM (RT-2 style: https://robotics-transformer2.github.io/), 注入 semantic priors。 比如让 humanoid 听 "fold the red shirt" 完成任务, 而不是只能模仿固定 trajectory。

### 4. Active camera 是 big missing piece

固定头部 camera 是大 limitation。 如果 head 能主动 gaze (turn left/right/up/down), hand 就不会出 FoV, 且能主动 gather 信息。 这需要把 head control 加入 policy, 训练时 randomize head initial pose, 让 policy 自己决定 gaze 点。 参考 active vision literature (https://arxiv.org/abs/2306.11909)。

### 5. Hardware co-design

1-DoF ankle 真的限制太大。 Unitree H1-2 已经加了 ankle pitch (变 2-DoF), Figure 02 和 Atlas electric 都 more DoF。 下一版 HumanPlus 应该用 H1-2 或类似 platform。

### 6. VLA + Humanoid 融合

现在 HIT 是 task-specific, 40 demos 学一个 task。 如果用 VLA pretrain (OpenVLA: https://openvla.github.io/, RT-2: https://robotics-transformer2.github.io/), finetune 到 humanoid, 可能 fewer-shot 学新 task。 Humanoid 的优势是 form factor 匹配人类, 所以 internet video 数据可以直接用 (不像 single-arm robot 要 retarget)。

---

## 结论

**这篇 paper 的真正贡献**: 不是 algorithm 上的突破 (PPO 2017, ACT 2023, WHAM/HaMeR 2024 都不是新的), 而是 **系统级 integration**。第一次有一个 end-to-end 可复现的 stack 让 humanoid:

1. 用 $50 camera teleop (vs $7050 ALOHA 或 mocap)
2. whole-body (vs 仅 arm)
3. 单 operator (vs 多人)
4. 40 demos 学 complex skill (vs 1000+ demos)
5. 60-100% 成功率在 6 个真实 task 上

**关键 trick**: 
- Low-level policy 在 sim 学 (便宜)
- High-level policy 在 real 学 (避免 vision sim-to-real gap)
- Forward dynamics loss 防 proprio shortcut (BC 经典坑)
- Binocular vision 提供 depth (单 camera 不够)
- Domain randomization 保 sim-to-real (motor strength 0.8-1.1 是关键)

**未来方向**: active camera, diffusion policy, VLA integration, 更多 DoF hardware, long-horizon navigation。

这是 humanoid robotics 从 "demo-driven 工程项目" 走向 "data-driven learning system" 的关键一步。 Karpathy 你做 foundation model 时, robotics 是最后几个没被 conquer 的 domain, humanoid 是 robotics 里最 ambitious 的形态, 这篇 paper 给了一条 credible path。

相关链接汇总:
- Project: https://humanoid-ai.github.io
- AMASS: https://amass.is.tue.mpg.de/
- Unitree H1: https://www.unitree.com/h1/
- WHAM: https://wham.is.tue.mpg.de/
- HaMeR: https://geopavlakos.github.io/HaMeR/
- ACT/Aloha: https://tonyzhaozh.github.io/aloha/
- Mobile ALOHA: https://mobile-aloha.github.io/
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- DreamerV3: https://arxiv.org/abs/2304.10573
- RT-2: https://robotics-transformer2.github.io/
- OpenVLA: https://openvla.github.io/
- UMI: https://universal-manipulation-interface.github.io/
- OmniH2O: https://omni-humanoid.github.io/
- ExBody: https://expressive-humanoid.github.io/
- Humanoid Next Token: https://humanoidnext.github.io/

---

# HumanPlus 深度讲解

## 1. Paper核心 Thesis

这篇paper想解决一个根本问题: humanoid robot理论上应该能 leverage 海量人类数据(因为form factor相似),但实践中做不到,因为 (a) humanoid perception/control维度高且复杂, (b) morphology/actuation存在physical gap, (c) 缺一个从 egocentric vision 学autonomous skills的data pipeline。

HumanPlus 提供了一个full-stack方案, 两层stack:

**Layer 1 (Shadowing)**: 通过40小时offline human motion data (AMASS) 在仿真里用PPO训一个task-agnostic的low-level policy, 叫 Humanoid Shadowing Transformer (HST)。Deploy后用单个RGB camera + 实时pose estimation (WHAM + HaMeR) → retargeting → 实时teleoperation。

**Layer 2 (Imitation)**: 用shadowing收集real-world egocentric vision数据, 然后用supervised behavior cloning训一个 Humanoid Imitation Transformer (HIT), 输入binocular RGB + proprioception, 输出target pose chunk。

项目主页: https://humanoid-ai.github.io

---

## 2. Hardware Stack (33-DoF 180cm)

构建在 Unitree H1 之上 (19-DoF body):
- 每个 arm: 4-DoF shoulder/elbow组合 + 1-DoF wrist (Dynamixel servo + 两个thrust bearings)
- 每个 leg: 5-DoF
- waist: 1-DoF
- 每个 hand: Inspire-Robots RH56DFX, 6-DoF (4 fingers × 1-DoF + thumb 2-DoF)
- 头部: 两个 Razer Kiyo Pro RGB webcam, 向下50°, pupillary distance 160mm (binocular stereo)
- 手指最大10N, arm可持重7.5kg, 腿部电机瞬时torque 360Nm

设计intuition: anthropomorphic但DoF少于人类。Ankle只有1-DoF (人类是2-DoF的subtalar + ankle), wrist只有1-DoF (人类是2-DoF)。这就限制了agile motion (如单腿平衡晃动) 和6-DoF operational space control。

---

## 3. Retargeting 映射规则

人类motion用SMPL-X parameterize: 22个body 3-DoF spherical joints + 30个hand 3-DoF joints + 6D global transform。

**Body retargeting**: 直接copy对应Euler angle (hips, knees, ankles, torso, shoulders, elbows)。Hip和shoulder各3个正交revolute joint等价于一个spherical joint。

**Hand retargeting**: 取每个finger middle joint的rotation作为该finger的1-DoF command (thumb有2-DoF)。

**Wrist**: forearm和hand global orientation之间的相对rotation的1-DoF投影。

这个mapping是naive的。Intuition: 它丢弃了大量人类冗余DoF的信息,所以只能学到人类motion的一个subset。后续工作(如ExBody、Expressive Whole-Body Control)会做更复杂的retargeting (e.g., inverse kinematics in operational space)。

参考:
- AMASS: https://amass.is.tue.mpg.de/
- SMPL-X: https://smpl-x.is.tue.mpg.de/

---

## 4. Low-Level Policy: Humanoid Shadowing Transformer (HST)

### 4.1 架构

Decoder-only Transformer (类似GPT-style), context length = 8, 控制频率50Hz。

**Input** (每time step):
- Proprioception: root state (roll ϕ, pitch θ, base angular velocities ω_x, ω_y, ω_z), joint positions q (19维), joint velocities q̇ (19维), last action a_{t-1} (19维)
- Target pose: target forward/lateral velocities (v_x^tg, v_y^tg), target roll/pitch (r^tg, p^tg), target yaw velocity ω_yaw^tg, target joint angles q^tg (19维)

**Output**: 19维body joint position setpoints → 1000Hz PD controller转换成torques。

Hand target angles直接bypass policy送PD (因为hand不需要whole-body dynamics reasoning)。

### 4.2 训练目标 (PPO)

最大化discounted expected return:

$$\mathbb{E}\left[\sum_{t=0}^{T-1} \gamma^t r_t\right]$$

变量解释:
- t: time step index
- T: maximum episode length
- γ: discount factor (typically 0.99)
- r_t: reward at step t

### 4.3 Reward Function 设计

| Reward Term | Expression | Intuition |
|---|---|---|
| target xy velocities | $\exp(-\|[v_x, v_y] - [v_x^{\text{tg}}, v_y^{\text{tg}}]\|)$ | 鼓励跟踪水平速度, exponential kernel使reward光滑无界 |
| target yaw velocities | $\exp(-\|v_{\text{yaw}} - v_{\text{yaw}}^{\text{tg}}\|)$ | 跟踪转向角速度 |
| target joint positions | $-\|q - q^{\text{tg}}\|_2^2$ | quadratic penalty, 跟踪target pose |
| target roll & pitch energy | $-\|[r, p] - [r^{\text{tg}}, p^{\text{tg}}]\|_2^2 - \|\tau_i\|_2$ | 同时penalize躯干倾斜和motor torque (energy saving) |
| feet contact | $c == c^{\text{tg}}$ | 二值reward, 鼓励正确contact pattern (e.g., 双脚站立vs单脚swing) |
| feet slipping | $-\|v_{\text{feet}} \cdot \mathbb{1}[F_{\text{feet}} > 1]\|_2$ | 当foot接触地面 (F_feet > 1N) 时penalize脚的速度, 防止滑步 |
| alive | 1 | 每step +1只要没倒 |

变量解释:
- $v_x, v_y$: base的线速度水平分量
- $v_{\text{yaw}}$: base绕垂直轴的角速度
- $q$: 当前joint angles (19维)
- $q^{\text{tg}}$: retargeted target joint angles
- $r, p$: base的roll/pitch
- $\tau_i$: motor torque
- $v_{\text{feet}}$: 双脚线速度
- $c$: feet contact indicator (boolean)
- $F_{\text{feet}}$: feet受到的ground reaction force
- $\mathbb{1}[\cdot]$: indicator function
- 上标$\text{tg}$: target

Intuition: reward分成三类 — velocity tracking (鼓励跟踪human reference motion), posture matching (joint/roll/pitch), 和"safety/style" (no slip, energy efficient, alive)。Exponential kernel $\exp(-\|x\|)$ 在velocity tracking上的好处是bounded, 不像L2会随误差线性增长让agent失速冲。

### 4.4 Domain Randomization

| Env Param | Range |
|---|---|
| base payload | [-3.0, 3.0] kg |
| end-effector payload | [0, 0.5] kg |
| center of base mass | [-0.1, 0.1]^3 m |
| motor strength | [0.8, 1.1] |
| friction | [0.3, 0.9] |
| control delay | [0.02, 0.04] s |

Intuition: 这是sim-to-real的关键。Motor strength [0.8, 1.1]意味着policy必须能handle 20%的torque deficit到10%的torque excess, 模拟actuator model误差和电池电压波动。Control delay 20-40ms覆盖通信和控制loop的latency。Friction 0.3-0.9覆盖从光滑水泥到地毯的表面。

---

## 5. Real-Time Shadowing Pipeline

### 5.1 Body Pose Estimation: WHAM

WHAM (World-Grounded Humans with Accurate Motion) - CVPR 2024, 25 fps on RTX4090
- 输入: 单RGB视频流
- 输出: SMPL-X参数 + global translation/rotation
- 关键: 估计的是world-grounded (camera frame world coordinate), 不是relative-to-camera, 这样retargeting到humanoid时能保持spatial consistency

Paper: https://wham.is.tue.mpg.de/

### 5.2 Hand Pose Estimation: HaMeR

HaMeR (Reconstructing Hands in 3D with Transformers) - CVPR 2024, 10 fps on RTX4090
- 输入: 单RGB image crop
- 输出: MANO hand model参数 + camera + shape
- Transformer-based

Project: https://geopavlakos.github.io/HaMeR/

注意10 fps对hand偏慢, 是整个shadowing pipeline的bottleneck。Hand motion快时会有可见latency。

### 5.3 Pipeline 整体

```
RGB Camera (single)
  ├── Body stream → WHAM (25Hz) → SMPL-X body → retarget → body target pose
  └── Hand stream → HaMeR (10Hz) → MANO hand → retarget → hand target pose
                                                           ↓
                              HST low-level policy (50Hz, ctx=8) → body torque commands
                              PD controller (1000Hz) ← hand target poses
```

Intuition: operator站在humanoid旁边通过line-of-sight观察, 这是经典bilateral teleoperation思想但无需haptic feedback。Camera是被动感知, 不需要mocap/exoskeleton/VR headset。Cost $50 vs ALOHA $7050, Meta Quest $250。

---

## 6. Imitation: Humanoid Imitation Transformer (HIT)

### 6.1 与ACT的关系

HIT基于Action Chunking Transformer (ACT, RSS 2023) 改造:

**ACT (原始)**: encoder-decoder (CVAE-style), 用VAE的prior-posterior inference处理multimodality。输入当前proprio + image, 输出action chunk。

**HIT改造**:
1. **去掉encoder**, 变成decoder-only (类似decision transformer或trajectory model)
2. **加入forward dynamics prediction**: 在预测target pose chunk的同时, 预测对应的future image feature tokens
3. 用L2 loss on image features作为regularizer

### 6.2 Architecture

输入序列 (concatenated tokens):
- Image features from left camera (pretrained ResNet encoder)
- Image features from right camera (binocular)
- Proprioception
- Fixed positional embeddings (learnable)

输出:
- 50个target pose chunk (auto-regressive或parallel)
- Predicted future image feature tokens (用于auxiliary loss, deployment时discard)

运行频率: 25Hz onboard, 异步发送target到HST (50Hz)。

### 6.3 Forward Dynamics Loss

形式化: 设当前observation为$o_t$, predicted action chunk为$\hat{a}_{t:t+50}$, ground truth future observations为$o_{t+1:t+51}$。

预测target pose chunk:
$$\hat{a}_{t:t+50} = f_{\text{HIT}}(o_t)$$

同时预测future image features:
$$\hat{\phi}_{t+1:t+51} = g_{\text{HIT}}(o_t)$$

Forward dynamics loss:
$$\mathcal{L}_{\text{dyn}} = \|\hat{\phi}_{t+1:t+51} - \phi_{t+1:t+51}\|_2^2$$

其中$\phi_{t} = \text{ResNet}(o_t)$是ground truth image feature。

总loss:
$$\mathcal{L} = \mathcal{L}_{\text{action}}(\hat{a}, a^*) + \lambda \mathcal{L}_{\text{dyn}}(\hat{\phi}, \phi^*)$$

变量解释:
- $o_t$: observation at step t (left/right image + proprio)
- $a_{t:t+50}$: action chunk (target poses)
- $\phi_t$: image feature embedding
- $\lambda$: loss weight (paper没明确数值)
- $f_{\text{HIT}}, g_{\text{HIT}}$: HIT的两条预测head

### 6.4 为什么forward dynamics loss有用

**核心问题**: 在behavior cloning中, vision-based policy容易"忽略"visual input, 只依靠proprioception overfit。原因是action和proprioception高度相关 (下一步action基本就是上一步action的平滑变化), 而image features高维稀疏, 网络很容易学到shortcut。

**Forward dynamics强制**: 网络必须用image features预测下一时刻的image features。这意味着网络必须从当前image提取"环境状态"信息(比如物体位置、手与物体相对位置), 因为这些信息无法仅从proprio获得。

**Intuition**: 这相当于一个implicit world model regularizer。 类似DreamerV2/V3里actor-critic和world model共享backbone的思想, 也类似Diffusion Policy里的future prediction (但用L2而不是diffusion objective)。在ALOHA的follow-up工作UMI (Universal Manipulation Interface) 中也用了类似思想。

参考:
- ACT (ALOHA): https://tonyzhaozh.github.io/aloha/
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- UMI: https://universal-manipulation-interface.github.io/

---

## 7. Tasks 设计

### 7.1 Imitation Tasks (6个, 用40 demos以内)

1. **Wear a Shoe and Walk** (40 demos, 50s each): 10个sub-steps (flip → pick → put on → press → tangle → grasp R → grasp L → tie → stand → walk)。测试bimanual dexterity + locomotion transition。Shoe放置在2cm line上随机化。

2. **Warehouse** (25 demos, 20s each): 从warehouse rack上pick paint sprayer, squat, 放到quadruped背负的cart里, 站起。测试locomanipulation。

3. **Fold Clothes** (40 demos, 20s each): 折sweatshirt三步 (left sleeve, right sleeve, bottom)。Randomized yaw ±10°, 衣物位置10×10cm, 旋转±30°。测试柔性物体manipulation + 平衡。

4. **Rearrange Objects** (30 demos, 10s each): pick 4种soft object (stuffed toy, ice bag等), 放入basket。位置左右随机化10cm。

5. **Type "AI"** (30 demos, 8s each): 敲A, 释放, 敲I, 释放。测试坐姿高精度。

6. **Two-Robot Greeting** (30 demos, 5s each): 与另一bimanual robot握手。另一robot随机选哪只手, 停在5×5×5cm end-effector region内。需要fast visual recognition。

### 7.2 Shadowing Tasks (5个, 无定量指标)

Boxing, opening two-door cabinet存pot, tossing, playing piano, playing table tennis, typing "Hello World"。展示fast, diverse motion以及重物manipulation。

---

## 8. 关键实验结果分析

### 8.1 Teleop Comparisons (Table 3)

| Method | Cost | Operators | Whole-Body | Pick (s) | Place (s) | Whole (s) | Stand (%) |
|---|---|---|---|---|---|---|---|
| Kinesthetic | $50 | 3 | ✗ | 2.10 | 3.12 | 6.60 | 90.5 |
| ALOHA | $7050 | 2-3 | ✗ | 2.70 | 3.15 | 7.15 | 100 |
| Meta Quest | $250 | 2 | ✗ | 3.57 | 3.67 | 8.87 | 95.3 |
| **Ours** | **$50** | **1** | **✓** | **1.76** | **2.59** | **5.20** | **100** |

关键insight:
- ALOHA虽然精确但hardware fixed, 难适应不同height/operator body shape, 且default不支持whole-body
- Meta Quest IK常遇singularity (5-DoF arm + 1-DoF wrist), Cartesian tracking不稳
- Kinesthetic多operator且external force会让robot stumble
- Ours唯一能完成"Rearrange Lower Objects" (15.34s) 因为需要squat

### 8.2 Robustness (Table 4)

| | Forward | Backward | Left | Right | Recovery | Squat | Jump | Stand Up |
|---|---|---|---|---|---|---|---|---|
| Ours | 32N | 44N | 70N | 100N | 1.2s | 0.44m | 0.35m | ✓ |
| H1 Default | 24N | 36N | 40N | 40N | 15s | 0.85m | 0m | ✗ |

Lateral (left/right) 比 sagittal (forward/backward) 更robust, 因为lateral perturbation通常只触发hip abduction策略, 而sagittal涉及CoM投影到BoS外面的fall risk, 需要step。

H1 default controller的15s恢复时间是因为用ZMP-based walking, 失衡后需要重新规划footstep并稳定gait cycle。RL policy直接从history推断, 1-2步内recover。

### 8.3 Imitation成功率 (Table 5)

最有信息量的对比:

**Wear a Shoe and Walk** (最难, 40 demos):
- HIT (Ours): 60% whole task
- Monocular: 0%
- ACT: 0% (overfits to proprio, 在Pick Up Shoe后stuck重复)
- Open-loop: 0%

**Fold Clothes**:
- HIT: 100%
- Monocular: 40% (depth缺失导致与table rough interaction)
- ACT: 100%
- Open-loop: 0%

**Type "AI"**:
- HIT: 80%
- Monocular: 40% (深度缺失但narrow FoV有时反而好)
- ACT: 60% (overfits proprio)
- Open-loop: 60% (因为no randomization)

**Rearrange Objects**:
- HIT: 90%
- Monocular: 70%
- ACT: 50%
- Open-loop: 0%

Insight: 
- Binocular perception在需要depth interaction (抓取, 折衣, 穿鞋)上关键
- Forward dynamics loss防止overfitting到proprio, 这是ACT失败的核心原因
- Open-loop只在no-variation task (Type "AI")可用

---

## 9. Limitations (作者承认的)

1. **DoF不足**: 1-DoF ankle限制agile单腿motion, 5-DoF arm导致某些workspace不可达
2. **Fixed cameras**: 头部cameras不主动, hand容易out of view
3. **Fixed retargeting mapping**: 大量人类joints直接被丢弃
4. **Pose estimation鲁棒性**: 大面积occlusion时失败, 限制operator操作区域
5. **无long-horizon navigation**: 只做in-place manipulation + 短距离walking

---

## 10. 与同期工作的关系

### 10.1 OmniH2O (He et al., 2024)
https://omni-humanoid.github.io/

非常concurrent的工作, 也用RL训low-level policy, 也用RGB camera teleoperation。区别:
- OmniH2O强调sim-to-real的difficulty scoring
- HumanPlus多了imitation pipeline (HIT)
- HumanPlus hardware更custom (33-DoF vs H1 default)

### 10.2 ExBody / Expressive Whole-Body Control (Cheng et al., 2024)
https://expressive-humanoid.github.io/

专注upper-body expressive motion, 用perceptual reward让仿真motion看起来像人类。HumanPlus更偏task-oriented。

### 10.3 Mobile ALOHA (Zhao et al., 2024)
https://mobile-aloha.github.io/

同作者, 但在mobile base + bimanual arm, 不做人形bipedal locomotion。HIT的ACT改造思想与Mobile ALOHA一脉相承。

### 10.4 Humanoid Locomotion as Next Token Prediction (Radosavovic et al., 2024)
https://humanoidnext.github.io/

用supervised learning (而非RL)直接从motion data学locomotion policy, 把它framing成next token prediction。Casual transformer framework。

### 10.5 UMI (Universal Manipulation Interface)
https://universal-manipulation-interface.github.io/

手持gripper采集demonstration, 然后diffusion policy。解决"在没robot时如何采集in-the-wild demonstration"问题。HIT的forward dynamics regularizer思路类似——确保vision feature被充分利用。

---

## 11. 整体 Intuition Summary

HumanPlus的根本贡献是把"humanoid能从人类数据学"这个口号变成可复现的工程pipeline:

1. **Decoupling**: 把整个problem解耦成两层 — task-agnostic low-level controller (从40小时人类offline data学) + task-specific high-level vision policy (从40 demos学)。Lower level解决"dynamics", higher level解决"task semantics"。

2. **Asymmetric data use**: Offline data只用来学low-level (因为在sim中retargeting便宜), online teleop数据用来学high-level (因为vision sim-to-real难)。

3. **Forward dynamics as regularizer**: 让vision policy不能"偷懒"忽略vision。这是把world model思想当作auxiliary loss的经典trick。

4. **Single RGB camera teleop**: 把teleoperation成本从$7050 (ALOHA)或 mocap系统降到$50, 同时支持whole-body。这极大地democratize了humanoid data collection。

5. **Limitation honesty**: 作者承认fixed camera、limited DoF、fixed retargeting、no long-horizon navigation都是未来工作。

相关工作汇总链接:
- AMASS: https://amass.is.tue.mpg.de/
- SMPL-X: https://smpl-x.is.tue.mpg.de/
- Unitree H1: https://www.unitree.com/h1/
- Inspire-Robots RH56DFX: https://www.inspire-robots.com/
- WHAM: https://wham.is.tue.mpg.de/
- HaMeR: https://geopavlakos.github.io/HaMeR/
- PPO: https://arxiv.org/abs/1707.06347
- ACT/Aloha: https://tonyzhaozh.github.io/aloha/
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- OmniH2O: https://omni-humanoid.github.io/
- ExBody: https://expressive-humanoid.github.io/
- Mobile ALOHA: https://mobile-aloha.github.io/
- UMI: https://universal-manipulation-interface.github.io/
- Humanoid Next Token: https://humanoidnext.github.io/

如果你 (Karpathy) 想从这个paper引申更深的思考方向, 我会提到几个interesting follow-up:

- **Camera active control**: 把head DoF加入policy, 让vision主动gaze。这就接近"humanoid embodied perception"问题。可参考NeRF/active vision literature。
- **Diffusion policy替换HIT**: action chunk + diffusion可能better handle multimodality。 HumanPlus只用L2 regression可能限制task复杂度。
- **VLA integration**: HIT的vision encoder本质是pretrained ResNet, 可换CLIP/SigLIP/DINO甚至VLM, 注入semantic priors。RT-2/OpenVLA路线在humanoid上的延伸。
- **Self-supervised forward dynamics**: 现在forward dynamics只是auxiliary loss。如果让它成为main objective (类似DreamerV3 / World Models), 在imagination里做planning, 可能sample efficiency更好。
- **Hardware co-design**: ankle加1-DoF (passive toe joint) 可极大改善agile locomotion, 论文里limitation也提到这点。Boston Dynamics Atlas electric版、Unitree H1-2、Figure 02都在这条路上。

这paper本质是一个"integration paper" — 没有任何一个component技术上fundamentally new (PPO是17年, ACT是23年, WHAM/HaMeR是24年concurrent), 但系统性的end-to-end stack让humanoid imitate complex skill with 40 demos这个目标第一次有可复现的recipe。在Stanford以外的lab复现这个stack需要H1 ($90k左右) + 手 + cameras + 计算resource, 大概总投入$150k以内, 比起mocap+humanoid+exoskeleton的传统stack ($500k+)便宜一个量级。这就是为什么这个paper在humanoid community里被讨论得多。
