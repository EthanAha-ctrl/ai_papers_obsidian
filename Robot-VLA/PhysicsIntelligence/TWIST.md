---
source_pdf: TWIST.pdf
paper_sha256: cfcaed7ebf12a119f48b1b79ecb13ff23c1838d7dd058ca536bfed21ed273264
processed_at: '2026-08-12T18:41:10-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# TWIST 用人话说

## 1. 这帮人在干嘛？

Stanford 的一帮人想让 humanoid robot 完整模仿 human 的全身动作。你跳舞它跳舞，你踢球它踢球，你蹲下搬箱子它也蹲下搬箱子。听起来简单，做起来巨难。

难在哪？**robot 和 human 长得不一样**。human 有 100 多个 joint，Unitree G1 只有 29 个；limb 长度不同，mass distribution 不同，motor 力量也不同。你直接把 human 动作"copy paste"到 robot 上，robot 大概率摔给你看。

而且 teleoperation 这个场景有个特殊难点：**human operator 自己也不知道下一秒要做什么**。这和 offline replay 完全不同——offline 的时候你有完整 motion clip，可以提前规划；teleoperation 的时候你只有一个 current frame，future 是 unknown 的。

---

## 2. 核心思路：三个 stage 串起来

### Stage 1: 把 human 动作"翻译"成 robot 能理解的

human 的 MoCap data 是 120Hz 的 3D keypoint 位置，robot 需要的是 29 个 joint angle。中间需要一个 **retargeting** 步骤。

这里有个坑：offline retargeting 可以慢慢 optimize，质量很高；但 teleoperation 要 real-time，只能用 fast online retargeter，质量差一些，会有 jitter。

他们的 solution 是：**online retargeter 同时优化 position 和 orientation**，而不只是 orientation。这样 jitter 少很多。

### Stage 2: 在 simulation 里训练一个 controller

这是 paper 的核心。后面详细讲。

### Stage 3: Deploy 到 real robot

OptiTrack 120Hz 抓 human motion → online retargeter 50Hz 转成 robot target → policy 50Hz 推理输出 joint command → robot PD controller 1000Hz 执行。

总 delay 大概 0.9 秒，其中 0.7 秒花在 retargeting 上，policy inference 只要 0.2 秒。所以 bottleneck 不在 neural network，在 retargeting pipeline。

---

## 3. Controller 训练的四个关键 trick

这是 paper 的 meat。我一个个讲 intuition。

### Trick 1: Teacher-Student Framework

**问题**：如果 controller 只看 current frame，它不知道 human 接下来要做什么，会变得 **conservative**——动作犹豫，不敢 commit。Human operator 看到 robot 犹豫，会下意识放慢自己的动作来 compensate，结果 robot 更犹豫，形成恶性循环。

**Solution**：先训练一个 teacher policy，给它 **未来 2 秒的 motion frames** 作为 privileged information。Teacher 可以 anticipate，动作很 smooth。然后再 distill 到 student policy，student 只看 current frame，但通过 imitation learning 学到了 teacher 的 smooth behavior。

**Intuition**：这就像驾校教练坐在你旁边，教练知道前方路况（future frames），开得很顺；你只看眼前，开得磕磕绊绊。但如果你跟车开（imitation），你的轨迹也会变 smooth，即使你不知道未来路况。

### Trick 2: RL + BC Hybrid Loss

**问题**：怎么把 teacher 的 knowledge 传给 student？

- **Pure BC (DAgger)**：student 只模仿 teacher，遇到没见过的 motion 就崩了，因为 teacher 的 action distribution 覆盖不了所有情况
- **Pure RL**：student 自己探索，但 single-frame observation 让它 conservative，还会有 foot sliding artifact

**Solution**：两个一起用。

$$L(\pi_{\text{stu}}) = L_{\text{RL}}(\pi_{\text{stu}}) + \lambda D_{\text{KL}}(\pi_{\text{stu}} \parallel \pi_{\text{tea}})$$

- $\pi_{\text{stu}}$: student policy
- $L_{\text{RL}}$: PPO loss，用和 teacher 一样的 reward
- $D_{\text{KL}}$: KL divergence，让 student 的 action distribution 靠近 teacher
- $\lambda$: weight，训练过程中 **逐渐减小**

**Intuition**：初期 $\lambda$ 大，student 主要靠模仿 teacher 学到 smooth behavior；后期 $\lambda$ 小，student 自己探索，通过 RL reward 学到 generalization。这就像小孩学骑车——一开始爸爸扶着（BC），后来放手让你自己骑（RL）。

这个发现和 LLM post-training 里的 [SFT memorizes, RL generalizes](https://arxiv.org/abs/2501.17161) 是同一个道理——pure SFT/BC 会 memorize，pure RL 会 generalize 但不稳定，hybrid 最好。

### Trick 3: 加一点点 in-house MoCap data

**问题**：public dataset (AMASS, OMOMO) 很 clean 很 smooth，但 real teleoperation 里的 motion 有 noise、有 jitter、有 calibration drift。Distribution shift 导致 controller 在 real world 表现差。

**Solution**：自己用 MoCap 系统录了 **150 clips (~0.5 hours)**，随机录的，不针对任何 task。和 15K clips (~42 hours) 的 public data 一起训练。

**结果**：这 0.5 hours 的 data 产生了 **disproportionate large** 的效果——unseen motion 的 tracking error 显著下降。

**Intuition**：这 150 clips 虽然 content 上没什么特别的，但它包含了 **real teleoperation 的 noise distribution**。Controller 见过这些 noise，就学会了 robustness。这就像你训练一个 image classifier，全部用 clean ImageNet，遇到 real world 的 blurry image 就崩了；加一点 noisy data，立刻 robust 很多。

这是 **small data, big impact** 的经典案例。关键是 data 的 **distribution** 而不是 **quantity**。

### Trick 4: End-effector perturbation training

**问题**：controller 只学 motion tracking（reach target position），但 real task 需要 **apply force**（比如搬箱子）。Box 接触 end-effector 时产生 external force，controller 没见过这种 disturbance，会 jitter。

**Solution**：训练时在 end-effector 上加 **0-20N 的 random perturbation force**。

**Intuition**：这就像练拳击——你光打沙袋（position tracking）不够，还得让人推你（perturbation），这样实战（contact task）才不会一碰就倒。

Figure 7 (left) 显示得很清楚：没加 perturbation 的 controller，让 robot 拿着 box 保持 stationary pose，会慢慢 drift；加了 perturbation 的，稳如泰山。

---

## 4. Reward 设计的细节

Table 1 里的 reward 很有讲究：

**Tracking reward**（鼓励 mimic）：
- KeyBody Position: weight 2.0（最高，因为 key body 位置最直观）
- Root Velocity: weight 1.0（locomotion 的核心）
- Joint Position: 0.6
- Root Pose: 0.6
- Joint Velocity: 0.2

**Penalty**（避免 artifact）：
- Feet Slipping: **-0.1**（重罚！foot sliding 是 single-frame tracking 的常见 artifact）
- Action Rate: -0.01（避免 action jitter）
- Feet Contact: -5e-4（轻罚，避免乱踩）
- Joint Velocity: -1e-4（轻罚，避免动作过猛）
- Feet Air Time: **+5.0**（正 reward！鼓励适当的 air time，让 gait 自然）

注意 **Feet Slipping Penalty (-0.1)** 比 Feet Contact Penalty (-5e-4) 大了 200 倍。这说明 foot sliding 是 single-frame tracking 的头号问题，必须重罚。HumanPlus 用的 pure RL 就有 foot sliding artifact，TWIST 通过这个重罚 + teacher 的 future frame anticipation 解决了。

还有一个设计 choice：**在 local frame 里 tracking**，而不是 world frame。原因是 world frame 会让 small error 累积成 large drift，local frame 更 consistent with teleoperation setup。

---

## 5. Domain Randomization

Table 2 的 range 也有讲究：

- **Base Mass [-3, 3] kg**：robot 真实 mass 有不确定性
- **Friction [0.1, 2.0]**：地面摩擦力变化大
- **Motor Strength [0.8, 1.2]**：motor 力量 ±20% 波动
- **Push End-Effector [0, 20] N**：这个前面讲过，让 controller 学会 apply force

这些 randomization 让 controller 在 sim 里见过各种 condition，deploy 到 real world 就 robust 了。

---

## 6. 实验结果的几个 takeaway

### 6.1 RL+BC >> RL >> BC

Figure 6 很直观：
- **RL+BC**：tracking error 最低，motion smooth
- **Pure RL**：error 中等，但有 foot sliding
- **Pure BC (DAgger)**：error 最高，偶尔直接 fail

这说明：RL 负责 generalization，BC 负责 stability，缺一不可。

### 6.2 Tracking error 分布

Figure 8 (right) 显示：
- **Feet** error 最大
- **Hands** 第二大
- **Knees** 也比较大
- **Elbows** 较小

原因：
1. End-effector 在 kinematic tree 末端，error 累积
2. Lower body 涉及 contact dynamics，比 upper body 难 tracking

这也解释了为什么 reward 里 KeyBody Position (2.0) 和 Root Velocity (1.0) 权重最高——这些是 locomotion 的核心。

### 6.3 Reachability

Figure 9 (a) 显示 robot 可以 **nearly reach its toes with hands**。这说明 whole-body coordination 达到了前所未有的水平——之前的 HumanPlus、OmniH2O 都做不到这种 extreme reachability。

### 6.4 Failure case

Figure 9 (b) 是 motor overheating，5-10 分钟就过热，尤其 crouching task。这是 **hardware limitation**，不是 controller 的问题。但即使过热，controller 仍能 maintain balance——这说明 controller 的 robustness。

---

## 7. 和前人工作的对比

| System | Input | Controller | Whole-body? |
|--------|-------|------------|-------------|
| **HumanPlus** | RGB camera pose estimation | Single-stage RL | Partial，root position 不准 |
| **OmniH2O** | VR keypoints (upper body) | DAgger | Partial，缺下半身 |
| **Mobile-TV** | VR + joystick | Decoupled upper/lower | No，decoupled |
| **HOMIE** | Exoskeleton + foot pedal | Decoupled | No，decoupled |
| **TWIST** | MoCap (full body) | RL+BC teacher-student | **Yes，unified** |

TWIST 是第一个用 **single unified controller** 实现 **versatile whole-body teleoperation** 的系统。

---

## 8. Limitations

Paper 自己承认的：

1. **No egocentric vision feedback**：operator 看不到 robot 眼睛看到什么，occlusion 时很难 teleoperate
2. **No tactile feedback**：operator 不知道 grasp 是否成功
3. **Hardware reliability**：motor 过热，5-10 分钟就要休息
4. **MoCap system 依赖**：不 portable，难普及

Future work 方向：
- 用 RGB pose estimation 替代 MoCap
- 加 egocentric vision feedback
- 结合 teleoperation data 训练 visuomotor policy

---

## 9. 我的几个 intuition

### 9.1 为什么 future frame 这么重要？

Teleoperation 本质是一个 **closed-loop control** 问题，human 和 robot 形成 feedback loop。如果 robot 响应慢，human 会 compensate，导致整个 system 振荡。Teacher policy 通过 future frame anticipation 打破了这个振荡——robot 提前准备，human 不需要 compensate，system 稳定。

这和 LLM 里的 **prompt engineering** 有异曲同工之妙——给 model 更多 context，output 质量显著提升，即使最终 deploy 时 context 少了，distill 过的 model 也能保持质量。

### 9.2 为什么 small targeted data 这么 effective？

因为 **distribution** 比 **quantity** 重要。Public dataset 是 clean 的，real world 是 noisy 的。150 clips 的 noisy data 让 controller 见过 real world 的 noise distribution，就学会了 robustness。

这和 LLM 里 **high-quality preference data** 的作用类似——几百条精心标注的数据，效果可能比几万条 low-quality data 还好。

### 9.3 为什么 RL+BC 这个组合 work？

因为 RL 和 BC 解决的是 **不同的问题**：
- RL 解决 **generalization**——通过 reward signal，controller 学会处理 unseen situation
- BC 解决 **stability**——通过 imitation，controller 保持 teacher 的 smooth behavior

Pure RL 会 generalize 但不稳定（foot sliding）；pure BC 会稳定 但不 generalize（unseen motion 崩）。Hybrid 兼得两者优点。

$\lambda$ gradually reduced 是关键——初期靠 BC 打基础，后期靠 RL 精调。这和 LLM 里 **SFT → RLHF** 的 pipeline 思路一致。

### 9.4 为什么 perturbation training 对 force task 重要？

因为 **position tracking 和 force control 是不同的 problem**。Tracking 只关心 reach target，force control 关心 **interact with environment**。如果 controller 从没见过 external force，遇到 contact 就会 jitter。

Perturbation training 本质是把 controller 从 **kinematic problem** 扩展到 **dynamic problem**。这和 robot learning 里的 **interaction learning** 一脉相承——不仅要 move，还要 **interact**。

---

## 10. 总结一句话

TWIST 的核心 insight：**real-time teleoperation 的瓶颈在 reference motion 质量和 controller 对 imperfect reference 的 robustness，而不在 controller 本身的 tracking ability**。通过 teacher-student + RL+BC + in-house data + perturbation，他们系统性地解决了这些问题，第一次实现了 versatile whole-body teleoperation。

更多细节看 [project page](https://humanoid-teleop.github.io/) 和 [paper](https://arxiv.org/abs/2502.09048)。

---

# TWIST: Teleoperated Whole-Body Imitation System 深度解析

## 1. 研题背景与 Motivation

### 1.1 问题本质

Humanoid robot 的 teleoperation 长期存在一个核心矛盾：human body 与 humanoid robot 之间的 **embodiment gap**。当 human operator 做出一个 motion，robot 无法直接 reproduce，因为：

- **DoF 不匹配**：human 大约 100+ DoF，而 Unitree G1 只有 29 DoF
- **Kinematic structure 差异**：limb length、joint limit、mass distribution 都不同
- **Sensorimotor delay**：human 的 feedback loop 与 robot 的 feedback loop 不同步

传统方法（如 [Penco et al. 2019](https://ieeexplore.ieee.org/document/8490902)）采用 modular model-based controller，将 locomotion 与 manipulation 分开处理，导致无法执行 coordinated whole-body skills，例如：
- 用脚踢球同时手拿物体
- 蹿下身去捡地上的箱子
- 跳 Waltz 舞

### 1.2 与前人工作的区别

TWIST 在 related work 中明确区分了三类前人工作：

1. **HumanPlus** ([Fu et al., CoRL 2024](https://humanoid-ai.github.io/humanplus/))：使用 camera-based pose estimation，root position 精度不足，影响 locomotion fidelity
2. **OmniH2O** ([He et al., 2024](https://arxiv.org/abs/2406.08858))：使用 VR keypoints，只捕获上半身动作，缺乏 full whole-body control
3. **Mobile-TV** / **HOMIE**：decouple upper/lower body control，使用 joystick 或 foot pedal 作为外部 command，无法执行 kicking 或 obstacle traversal 等 whole-body tasks

TWIST 的核心差异在于：使用 **motion capture (MoCap) device** 作为 high-quality input，配合 **single unified neural network controller**，实现真正的 whole-body coordinated control。

---

## 2. 系统架构总览

TWIST 采用三阶段 pipeline（Figure 3）：

### Stage 1: Curating Humanoid Motion Dataset
- **Data source**: AMASS ([Mahmood et al., ICCV 2019](https://amass.is.tuebingen.mpg.de/)) + OMOMO ([Li et al., SIGGRAPH 2023](https://omomo.is.tuebingen.mpg.de/)) 共 15,000+ clips (~42 hours)，加上 150 clips in-house MoCap data (~0.5 hours)
- **Filtering**: 去除不可行动作（如 climbing stairs）
- **Retargeting**: Human motion → humanoid robot motion

### Stage 2: Training Whole-Body Controller in Simulation
- **Environment**: IsaacGym ([Makoviychuk et al., 2021](https://arxiv.org/abs/2108.10470)) for training, MuJoCo ([Todorov et al., 2012](https://arxiv.org/abs/2010.00348)) for evaluation
- **Method**: Two-stage teacher-student framework with RL + BC

### Stage 3: Real-World Teleoperation
- **Capture**: OptiTrack at 120Hz
- **Retarget**: 50Hz online retargeting
- **Inference**: 50Hz policy inference on Nvidia RTX 4090 GPU
- **Control**: PD controller at 1000Hz

---

## 3. 关键技术细节

### 3.1 Motion Retargeting: Offline vs Online

#### Offline Retargeting
用于处理 AMASS + OMOMO 等大规模 public dataset，采用类似 **PHC** ([Luo et al., ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Luo_Perpetual_Humanoid_Control_for_Real-Time_Simulated_Avatars_ICCV_2023_paper.pdf)) 的方法：

- **优化目标**: key body positions
- **附加**: temporal smoothness optimization
- **特点**: iterative optimization，质量高但无法实时

#### Online Retargeting
用于 real-time teleoperation，基于 IK 方法 ([mink library](https://github.com/kevinzakka/mink))：

- **速度**: fast inference
- **问题**: less smooth motion
- **改进**: 同时优化 3D joint positions **和** orientations，而不是只优化 orientations

这个改进的原因是：online retargeter 在 fast inference 下容易产生 jitter，只优化 orientation 会导致 reference motion 与 robot kinematic structure 不匹配。通过 joint optimization of positions + orientations，可以减少 offline-to-online gap。

Figure 8 (left) 显示，改进的 online retargeter 对 **with MoCap** 和 **without MoCap** 两种 setting 都能降低 tracking error。

### 3.2 Teacher Policy: Privileged Future Frames

#### 输入
- Proprioception (robot state)
- **2 seconds future motion frames** (privileged information)

#### 为什么 future frames 重要？
这是本文最关键的设计之一。考虑一个 teleoperation 场景：
- Human operator 想要 robot 蹿下
- 如果 controller 只看当前 frame，它不知道未来要做什么，会 **conservative and hesitant**
- Human operator 看到 robot 犹豫，会 **compensate** 自己的动作
- 这形成 **negative feedback loop**，导致 teleoperation 效率低下

有了 future frames，teacher policy 可以：
- **Anticipate**: 提前规划 gait transition
- **Plan**: 预测未来的 balance 需求
- **Smooth**: 产生更连续的 locomotion gaits

#### Reward 设计

$$r_{\text{tea}} = r_{\text{track}} + r_{\text{penalty}}$$

**Tracking rewards** (Table 1 left):

| Term | Weight | 说明 |
|------|--------|------|
| KeyBody Position Tracking | 2.0 | 关键身体点位置 |
| Joint Position Tracking | 0.6 | joint angle |
| Joint Velocity Tracking | 0.2 | joint angular velocity |
| Root Pose Tracking | 0.6 | base position + orientation |
| Root Velocity Tracking | 1.0 | base linear + angular velocity |

**Penalty terms** (Table 1 middle):

| Term | Weight | 说明 |
|------|--------|------|
| Feet Contact Penalty | -5e-4 | 惩罚异常 foot contact |
| Feet Slipping Penalty | -0.1 | 惩罚 foot sliding |
| Joint Velocities Penalty | -1e-4 | 惩罚过高 joint velocity |
| Action Rate Penalty | -0.01 | 惩罚 action jitter |
| Feet Air Time | 5.0 | 鼓励适当 air time |

注意 **Feet Slipping Penalty** 的权重 (-0.1) 远高于 Feet Contact Penalty (-5e-4)，说明 foot sliding 是一个非常严重的问题，需要重罚。这是因为 single-frame tracking 很容易导致 foot sliding artifact，正如 HumanPlus 中观察到的那样。

#### Local Frame Tracking
本文选择在 **robot's local frame** 而非 world frame 中 tracking joint positions 和 root velocities。原因有二：
1. **缓解累积误差**：world frame tracking 会让 small error 累积成 large drift
2. **Consistency with real-world teleoperation**：human operator 的 motion 也是在 local frame 中表达的

### 3.3 Student Policy: RL + BC Hybrid

#### 问题
Teacher policy 使用了 privileged future frames，但在 real-time teleoperation 中，human operator 的 future motion 是 **未知** 的（operator 自己也不知道下一秒会做什么）。

#### Solution: Knowledge Distillation with RL+BC

Student policy 的优化目标 (Equation 1)：

$$L(\pi_{\text{stu}}) = L_{\text{RL}}(\pi_{\text{stu}}) + \lambda D_{\text{KL}}(\pi_{\text{stu}} \parallel \pi_{\text{tea}})$$

**变量解释**：
- $\pi_{\text{stu}}$: student policy
- $L_{\text{RL}}$: PPO loss，使用与 teacher 相同的 reward $r_{\text{tea}}$
- $D_{\text{KL}}$: KL divergence，鼓励 student imitate teacher
- $\lambda$: weight，**gradually reduced during training**

#### 为什么不直接用 DAgger？
DAgger ([Ross et al., 2011](https://arxiv.org/abs/1011.0686)) 是经典的 imitation learning 方法，但本文发现它 **不能 stably and robustly track unseen motions**。原因是 DAgger 缺乏 task reward guidance，纯粹模仿 teacher，遇到 distribution shift 时容易失败。

RL+BC 的 hybrid 方法兼顾了两点：
1. **RL 的 generalization**: 通过 reward signal，student 可以 generalize 到 unseen motions
2. **BC 的 stability**: 通过 KL divergence，student 保持 teacher 的 smooth behavior

这个发现与最近 Chu et al. ([2025](https://arxiv.org/abs/2501.17161)) 在 LLM post-training 中的发现一致：**"RL generalizes, SFT memorizes"**。

#### 训练 Schedule
$\lambda$ gradually reduced during training 是关键。初期 $\lambda$ 大，student 主要靠 BC 学习 teacher 的 behavior；后期 $\lambda$ 小，student 更多地探索自己的策略，通过 RL 优化 reward。

### 3.4 In-House MoCap Data Matters

本文一个 surprising 的发现是：**仅添加 150 clips in-house MoCap data (~0.5 hours) 就能显著提升 generalization**。

#### 为什么？
虽然 in-house data 很少，但它包含了 real-world teleoperation 的 critical imperfections：
1. **Noise**: calibration drift, occlusions
2. **Less smooth motions**: online retargeter 产生的 jitter

通过将 clean public dataset 与 noisy in-house data 混合训练，controller 学到了对 real-world imperfections 的 robustness。

这类似于 **domain randomization** 的思想，但更加 targeted——直接从真实 teleoperation 中采样 noise distribution。

### 3.5 Learning to Apply Force

#### 问题
Tracking-based controller 只学习 reach target positions，但 real-world tasks 需要 **apply force**（如 lifting a box）。这导致 controller 在 contact-rich tasks 中表现 **jittery**。

#### Solution
在 training 中加入 **large end-effector perturbations** (Table 2: Push End-Effector [0, 20] N)。

这让 controller 学到：
- 即使 end-effector 受到外部 force，仍能 maintain balance
- 主动 apply force 来 manipulate objects

Figure 7 (left) 显示，没有 perturbation training 的 controller 在 stationary poses 中会出现 **drift and instability**，而有 perturbation training的则 stable。

### 3.6 Domain Randomization

Table 2 列出了 domain randomization 参数：

| Parameter | Range | 说明 |
|-----------|-------|------|
| Base Mass | [-3, 3] kg | robot mass uncertainty |
| Friction | [0.1, 2.0] | ground friction |
| Motor Strength | [0.8, 1.2] | motor torque scale |
| Gravity Change | [-0.1, 0.1] m/s² | gravity perturbation |
| Push Robot Base | [-0.1, 0.1] m/s | external push velocity |
| Push End-Effector | [0, 20] N | end-effector force |

值得注意的是 **Push End-Effector [0, 20] N** 这个 range 相当大——20 Newton 的 force 足以让 robot 的 arm 偏离 target position。这正是为了让 controller 学会 apply force。

---

## 4. 实验结果

### 4.1 Main Results

TWIST 在 Unitree G1 (29 DoF, 1.3m humanoid) 上展示了 diverse whole-body skills：

- **Whole-body manipulation**: uprighting trash can, crouching to lift box, carrying toy
- **Legged manipulation**: kicking door, kicking soccer ball, transporting box with feet
- **Locomotion**: sidesteps, backward walking, crouching under obstacles
- **Expressive motion**: boxing, Waltz dance

还展示了对 Booster T1 robot 的 **transferability** (Figure 4)。

### 4.2 Ablation Studies

#### Key Finding 1: RL+BC >> RL >> BC

Figure 6 (left) 显示 tracking error：
- **RL+BC**: 最低
- **RL (HumanPlus)**: 中等，但有 feet sliding artifacts
- **BC (DAgger, OmniH2O)**: 最高，偶尔 cannot robustly track unseen motions

Figure 6 (right) 显示 motion smoothness：RL+BC 产生 **smooth and robust behaviors**。

#### Key Finding 2: In-House MoCap Data Matters

Figure 7 (right) 展示不同 controller 在 MuJoCo 中 tracking MoCap data 的 rollout curves。加入 in-house data 后，tracking error 显著降低。

#### Key Finding 3: Learning to Apply Force

Figure 7 (left) 在 real world 中让 robot hold a box：
- **Without perturbation**: drift and instability
- **With perturbation**: stable

#### Key Finding 4: Better Online Retargeter

Figure 8 (left) 显示改进的 online retargeter (position + orientation) 比 only orientation 的 tracking error 更低。

### 4.3 System Analysis

#### Tracking Error Distribution (Figure 8 right)

- **End-effectors (hands, feet)**: largest errors
- **Lower-body (feet, knees)**: higher errors than upper-body (elbows, hands)
- **原因**: 
  1. End-effectors 处于 kinematic tree 的末端，error 会累积
  2. Lower-body 涉及更复杂的 contact dynamics

#### Teleoperation Delay

- **Total delay**: ~0.9 seconds
- **Retargeting**: 0.7 seconds (主要 overhead)
- **Policy inference**: 0.2 seconds

这个 delay 主要来自 retargeting，policy inference 已经很 efficient。Future work 应该聚焦于 faster retargeting。

#### Reachability (Figure 9a)

TWIST 的 robot 可以 nearly reach its toes with its hands，这说明 **exceptional whole-body reachability**。

#### Failure Cases (Figure 9b)

主要 failure 来自 hardware：
- Motor overheating after 5-10 minutes continuous operation
- 特别是在 crouching tasks 中

但即使 motor 过热，policy 仍能 maintain robust balance control——这说明 controller 的 robustness。

---

## 5. 技术贡献总结

1. **Two-stage teacher-student framework**: 通过 privileged future frames 训练 teacher，再 distill 到 student
2. **RL+BC hybrid loss**: 结合 RL 的 generalization 和 BC 的 stability
3. **In-house MoCap data**: 小规模但 targeted 的 data 提升 generalization
4. **Improved online retargeter**: joint optimization of positions and orientations
5. **End-effector perturbations**: 学习 apply force for contact-rich tasks
6. **Single unified controller**: 一个 neural network controller 处理 diverse skills

---

## 6. Limitations & Future Work

1. **Lack of Robotic Feedback**: 没有 egocentric vision streamed back to operator
2. **Hardware Reliability**: motor overheating 限制 continuous operation
3. **Dependence on MoCap Systems**: 不 portable，难普及
4. **Future**: 探索 RGB-based pose estimation，结合 TWIST-collected teleoperation data 和 egocentric human data 学习 visuomotor policies

---

## 7. 我的 Intuition Building

### 7.1 为什么 Teacher-Student 如此重要？

想象你在学开车：
- **Teacher** 知道前方 2 秒要发生什么（如红灯），可以 **anticipate**
- **Student** 只看当前，反应迟钝

在 teleoperation 中，human operator 倾向于 **compensate** robot 的迟钝动作，这会导致 **negative feedback**：
1. Robot 迟钝 → human 放慢动作
2. Human 放慢动作 → robot 依然迟钝
3. Human 进一步 compensate → teleoperation 效率低下

有了 teacher policy 的 future frame anticipation，robot 可以 pre-plan，减少迟钝，break 这个 negative feedback loop。

### 7.2 为什么 RL+BC > RL > BC？

- **BC alone**: 只模仿 teacher，遇到 distribution shift 容易失败，因为 teacher 的 action distribution 不能覆盖所有情况
- **RL alone**: 通过 reward 探索，但 single-frame observation 导致 conservative behavior，无法 anticipate
- **RL+BC**: 
  - RL 提供 generalization (通过 reward signal)
  - BC 提供 stability (通过 imitating teacher's smooth behavior)
  - Gradually reducing $\lambda$ 让 student 从 imitation 平滑过渡到 exploration

### 7.3 为什么 In-House Data 如此有效？

这类似于 **domain randomization**，但更加 targeted。Public dataset (AMASS, OMOMO) 是 clean、smooth 的，但 real-world teleoperation 中：
- MoCap device 有 calibration drift
- Online retargeter 产生 jitter
- Human operator 本身动作也不完美

通过将 clean public data 与 noisy in-house data 混合训练，controller 学到了对 real-world imperfections 的 robustness。这是 **small data, big impact** 的一个典型例子。

### 7.4 为什么 Perturbation Training 对 Force Tasks 重要？

Tracking-based controller 只学习 reach target positions，但 lifting a box 需要 **apply force**，这属于 **out-of-distribution** task。如果 controller 从未见过外部 force，当 box 接触 end-effector 时，controller 会产生 jittery behavior。

通过在 training 中加入 large end-effector perturbations (0-20N)，controller 学到了：
- 即使 end-effector 受到外部 force，仍能 maintain balance
- 主动 apply force 来 manipulate objects

这让 controller 从 **position tracking** 扩展到 **force control**，极大地扩展了 task 范围。

---

## 8. References

- [TWIST Project Page](https://humanoid-teleop.github.io/)
- [AMASS Dataset](https://amass.is.tuebingen.mpg.de/)
- [OMOMO Dataset](https://omomo.is.tuebingen.mpg.de/)
- [IsaacGym](https://developer.nvidia.com/isaac-gym)
- [MuJoCo](https://mujoco.org/)
- [mink IK Library](https://github.com/kevinzakka/mink)
- [OptiTrack](https://optitrack.com/)
- [Unitree G1](https://www.unitree.com/g1/)
- [Booster T1](https://www.boosterobotics.com/)
- [HumanPlus (CoRL 2024)](https://humanoid-ai.github.io/humanplus/)
- [OmniH2O](https://arxiv.org/abs/2406.08858)
- [PHC (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/papers/Luo_Perpetual_Humanoid_Control_for_Real-Time_Simulated_Avatars_ICCV_2023_paper.pdf)
- [DeepMimic](https://arxiv.org/abs/1804.02717)
- [PPO](https://arxiv.org/abs/1707.06347)
- [DAgger](https://arxiv.org/abs/1011.0686)
- [SFT memorizes, RL generalizes](https://arxiv.org/abs/2501.17161)

---

## 9. 对 Andrej 的额外思考

这篇 paper 体现了几个值得关注的 deep learning 哲学：

1. **Teacher-Student distillation with privileged information**: 这与 LLM 中的 chain-of-thought distillation 类似——teacher 用更多 context (future frames) 推理，student 用更少 context 模仿

2. **RL+BC > pure RL/BC**: 这与 LLM 的 RLHF + SFT hybrid 相呼应——单一方法都有局限，hybrid 方法可以兼得 generalization 和 stability

3. **Small targeted data matters**: 150 clips (~0.5 hours) vs 15K clips (~42 hours)，但 small targeted data 的影响 disproportionate large。这与 LLM 中 high-quality human preference data 的作用类似

4. **Sim-to-real is still hard**: 尽管有 domain randomization 和 in-house data，motor overheating 等 hardware issue 依然是 bottleneck。这说明 humanoid robot 的 hardware reliability 是当前 limit factor 之一

5. **Future frame anticipation**: 这与 LLM 中的 prompt engineering 类似——给 controller 更多 context (future frames) 可以极大提升 performance，即使 student 最终只用 single frame

6. **Local frame tracking**: 这是个 interesting 的设计 choice，类似于在 LLM 中做 relative position encoding 而非 absolute position encoding，可以避免累积误差

这篇 paper 的核心 insight 是：**real-time teleoperation 的瓶颈不仅仅在 controller 本身，更在于 reference motion 的质量和 controller 对 imperfect reference 的 robustness**。通过 teacher-student framework + RL+BC + in-house data + perturbation training，TWIST 系统性地解决了这些问题，最终实现了 versatile whole-body teleoperation。
