---
source_pdf: Fauna Sprout A lightweight, approachable, developer-ready humanoid robot.pdf
paper_sha256: f23d4e81627417cbb9d2e0a60ccd65e82d482144af9e752b6b9e64ee845c6f85
processed_at: '2026-08-04T08:00:42-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Sprout 用人话说

Andrej, 我把之前的技术解读再 "translate" 一下, 用更直白的方式讲清楚这个 robot 到底在干什么, 为什么这么设计, 以及它在 humanoid robotics 大图景里占据什么位置。

---

## 一、这个 robot 到底是干嘛的

**一句话**: Sprout 是一个 1.07 米高、22.7 公斤重的 mini humanoid, 做出来给 developer 玩的, 玩的地方是人旁边, 不是 lab 里。

就这么简单。它不追求 SOTA locomotion, 不追求 dexterous manipulation, 不追求 generalist AI。它追求的是 **能放到咖啡馆、教室、家里, 一个 designer 或者 educator 拿到就能用, 不需要 robotics PhD**。

这个 positioning 很重要。你想想 computing 历史里的 Apple II——它没做任何 technical breakthrough, 但它让非工程师能买到、能用、能写 software。Sprout 想做 humanoid 领域的 Apple II。Fauna Robotics 的 thesis 是: humanoid robotics 缺的不是一个更好的 RL algorithm, 缺的是一个 **能 deploy、能迭代、能被非专家使用** 的 platform。

参考: https://faunarobotics.com

---

## 二、为什么现在需要这么个东西

看一下当前 humanoid 的 landscape, 你会发现一个 gap:

**Industrial humanoid** (Atlas、Optimus、Figure 02、Unitree H1): 大、强、closed system。Tesla 不卖给你, Figure 不卖给你, Unitree G1 卖给你但设计目标是 warehouse 不是 preschool。这些 robot 没有 "safe around grandma" 这个 concept。

**Academic humanoid** (Berkeley Humanoid、ToddlerBot、OP3): open, 便宜, 但 manufacturability 差——你拿到 design 文件, 自己 3D 打印, 花 3 个月组装, 然后发现某个 motor 不兼容。这些是 research prototype, 不是 product。

**玩具 humanoid**: Robosen、UBTECH 那些, 能动但不可编程, 没有 SDK, 没有感知。

中间有个 gap: **能买到、能 deploy 到人旁边、有完整 SDK 的 capable humanoid**。Sprout 填的就是这个 gap。

这个 gap 之所以现在才被填, 是因为三个 enabling technology 刚好 mature:
1. **IsaacLab / MuJoCo Warp / Genesis** 让 sim training 变得 cheap 到小公司也能 train whole-body policy
2. **Jetson AGX Orin** 让 onboard compute 从 500W 降到 60W, 还能跑 perception + control
3. **VR headset** (Meta Quest 3, $499) 让 teleop hardware 从 $50K 降到 $500

这三件事在 2024 年之前都不成立。所以 Sprout 是 timing 上的必然, 不是偶然。

参考:
- Berkeley Humanoid: https://arxiv.org/abs/2504.05658
- ToddlerBot: https://arxiv.org/abs/2502.00893
- IsaacLab: https://isaac-sim.github.io/IsaacLab

---

## 三、Hardware 设计的几个关键 decision

### 3.1 为什么 1.07 米?

这是整个 hardware design 最 deliberate 的 choice。Paper 里算了一笔账: 动能 $E_k = \frac{1}{2} m v^2$。Sprout 22.7 kg 走 0.5 m/s, $E_k \approx 2.8$ J。Tesla Optimus 73 kg 走 1.5 m/s, $E_k \approx 82$ J。差 30 倍。

这个 30 倍意味着什么? 意味着 Sprout 撞到你, 顶多是瘀伤; Optimus 撞到你, 是骨折。这个 difference 决定了 Sprout 能放进 classroom, Optimus 不能。Safety 是 hardware 问题, software 解决不了——你 RL policy 再 fancy, 70kg 的 mass 倒下来该骨折还是骨折。

Rodney Brooks 2025 年那篇博客说得对: 今天 humanoid 的瓶颈不是 dexterity, 是 safe deployment。Sprout 完全接受这个 thesis。

参考: https://rodneybrooks.com/why-todays-humanoids-wont-learn-dexterity/

### 3.2 为什么不做 multi-finger hand?

Paper 里明确说: 1-DoF gripper, 12 N fingertip force limit。这个 12 N 是有意思的数字——人类 finger pinch force 大概 50-70 N, 12 N 是 "能 hold 物体但不会伤到人手指" 的 sweet spot。你手伸进去被 gripper 抓住, 12 N 顶多有点 pressure, 不会受伤。

为什么不做 dexterous hand? 三个 reason:
1. **Maintenance nightmare**: multi-finger hand 有 16+ DoF, 复杂度爆炸, 坏了一个 joint 整个 hand 报废
2. **Mass**: dexterous hand 通常 1-2 kg 每个, 加到 arm end 上 payload capacity 砍掉一半
3. **任务不匹配**: Sprout 的 target task 是 fetch、hand-off、physical interaction, 这些 task 不需要 in-hand rotation

Paper 说 "dependable whole-body behavior, compliance, ease of deployment are more immediate constraints than dexterous manipulation"。这句话很 Karpathy 风格——别在还没解决 deployment 的时候解决 dexterity。

### 3.3 Compute 架构: 为什么要分两层?

这是 hardware 里最容易被忽略但最关键的 design choice:

```
Jetson AGX Orin 64GB  ──── Ethernet/CBOR ────  MCM (Motor Control Module)
   perception                                           control loop
   planning                                             125 Hz deterministic
   LLM agent                                            independent of AGX load
   bursty workload                                      hard real-time
```

为什么这么分? 因为 AGX 的 workload 是 bursty 的——你在跑 LLM inference 时, AGX 可能 spike 100% 几百 ms。如果 control loop 也跑在 AGX 上, 这几百 ms robot 就摔了。

MCM 是独立 MCU, 跑 hard real-time control loop, AGX 卡不卡它都按 125 Hz 跑。这是 aerospace 和 automotive 的 standard design pattern——safety-critical 路径必须独立于 application-critical 路径。

CBOR (Concise Binary Object Representation, RFC 8949) 比 JSON 紧凑 30-50%, 比 Protobuf 更灵活。选这个 protocol 说明他们认真想过 latency 和 bandwidth trade-off。

参考: https://cbor.io/

### 3.4 Expressive head: 为什么用 LED + eyebrows 而不是 screen?

这是设计哲学问题。Paper 里说: "we use nonscreen show elements to emphasize the physicality of robotic embodiment"。

翻译成人话: **如果你给 robot 一个 screen face, 人会把它当 device**。如果你给它 physical eyebrows 和 LED 灯, 人会把它当 entity。

这有认知科学依据。人类对 face 的 processing 是 specialized 的 (fusiform face area), 但对 screen face 的 processing 是 categorical 的 (这是 iPad)。Sprout 想要前者, 不想要后者。Reachy Mini 也是同样思路。

这个 choice 对 HRI research 非常重要——如果你想 study "人对 embodied agent 的 trust", 用 screen face 的 robot 拿到的 data 是 contaminated 的。

参考: https://www.pollen-robotics.com/reachy-mini/

---

## 四、Software 架构: 为什么 hybrid 而不是 end-to-end VLA

这是 paper 里我觉得最值得讨论的 design decision。

### 4.1 当下 humanoid control 的两大派

**Monolithic VLA 派**: Google RT-1、Gemini Robotics、π0.5、Generalist AI。一个 neural network 从 image + language 直接输出 joint command。优点 simple, 缺点 safety 无法 verify。

**Modular 派**: NVIDIA GR00T N1、Figure Helix、Boston Dynamics + TRI Atlas。LLM emit high-level command, low-level control policy 执行。优点 safety 可以分层 verify, 缺点 architecture 复杂。

Sprout 选了 modular, 但保留了 monolithic 的可能性——SDK 暴露 APIs, 你可以从底层 swap 整个 stack。这是 pragmatic 的姿态: 我们现在选 modular, 因为 safety 必须 guarantee; 等未来 VLA safety 解决了, 你可以 swap 进来。

参考:
- GR00T N1: https://arxiv.org/abs/2503.14734
- Figure Helix: https://www.figure.ai/news/helix
- π0.5: https://arxiv.org/abs/2504.16054

### 4.2 Hybrid control 的三层结构

Sprout 的 control stack 是这样的:

**第一层: FSM (Finite State Machine) orchestrating modes**

把 control 分成 discrete mode: walking、kneeling、crawling、sitting。每个 mode 有自己的 validity condition、safety check、transition logic。外部 caller (teleop / autonomy / agent) request mode transition, FSM check 通过才 allow。

Intuition: 这相当于 robot 有 "operating mode" 的 concept。你不能在 walking mode 中突然让 robot 跪下, 必须先 transition。

**第二层: 每个 mode 里是 RL policy**

在 IsaacLab 里 train 的 PPO policy。Command 是 task-level 的:

$$\mathbf{c}_t = [v_x, v_y, \omega_z, \phi_{\text{roll}}, \phi_{\text{pitch}}, h_{\text{root}}, \mathbf{q}_{\text{upper}}]$$

- $v_x, v_y$: 前后/左右 base velocity (m/s)
- $\omega_z$: 转身 yaw rate (rad/s)
- $\phi_{\text{roll}}, \phi_{\text{pitch}}$: 想要的 body 倾斜角
- $h_{\text{root}}: 想要的 torso 离地高度
- $\mathbf{q}_{\text{upper}}$: 上半身 joint targets (arm、torso、neck、eyebrow)

Policy 输出 intermediate target $\mathbf{a}_t$, 被 PD + current limit 约束后送 motor:

$$\boldsymbol{\tau}_t = K_p(\mathbf{a}_t - \mathbf{q}_t) + K_d(\dot{\mathbf{a}}_t - \dot{\mathbf{q}}_t)$$

- $\boldsymbol{\tau}_t$: motor torque
- $K_p, K_d$: per-joint PD gains
- $\mathbf{q}_t, \dot{\mathbf{q}}_t$: 当前 joint position 和 velocity
- $\mathbf{a}_t$: policy 输出的 target position (中间量)

这层 PD + current limit 是 **hardware-level safety net**, RL policy 输出再离谱, 也被 bound 住。

Reward 包含 velocity tracking、energy penalty、smoothness, 以及最重要的 **compliance**: 在 random external force injection 下 minimize corrective force。这让 policy 学会 "被推就让", 而不是 "被推就硬抗"。

**第三层: Mode 之间的 transition 是独立的 IL policy**

这是最聪明的 design。问题是这样的: 你 train walking policy 时, state distribution 是 standing upright 周围。你 train kneeling policy 时, state distribution 是 kneeling 姿态周围。这两个 distribution 不 overlap。如果直接 switch, command discontinuity 会导致 robot 摔倒。

Solution: 每对 mode 之间 train 一个短 motion sequence 的 imitation learning policy。Training data 是 human mocap + animated trajectory。Transition policy 把 robot 从 source mode 的 state distribution 平滑带到 destination mode 的 state distribution。

Intuition: 你开车时, highway driving 和 parking 是两个完全不同的 skill。你不能在 60 mph 时突然 switch 到 parking mode。你需要一个 transition: 减速、换道、进入 parking lot、减速到 5 mph, 然后才进入 parking。Sprout 的 transition policy 就是这个 "减速换道" 的 bridge。

Transition 也有自己的 safety condition, 可以不同于 source 或 destination mode。比如 "transition to kneeling" 时允许的 pitch angle 范围可能比 "stable kneeling" 时更窄, 因为 transition 中 balance 更脆弱。

### 4.3 Projected gravity: 一个被低估的 trick

Teleop 模块里提到一个细节: 用 projected gravity 作为 control signal, 而不是 absolute orientation。

数学上, $\mathbf{g}_{\text{body}} = R^T \mathbf{g}_{\text{world}}$, 其中 $R$ 是 body-to-world rotation matrix, $\mathbf{g}_{\text{world}} = [0, 0, -9.81]^T$。$\mathbf{g}_{\text{body}}$ 是 3 维向量, 表示 gravity 在 robot body frame 中的方向。

**为什么这重要**: $\mathbf{g}_{\text{body}}$ 是 yaw-invariant。Robot 绕 vertical axis 转身, $\mathbf{g}_{\text{body}}$ 不变。这意味着 policy tracking $\mathbf{g}_{\text{body}}$ 可以在地形变化 (slope、stairs) 时自动保持 stable posture relative to gravity, 而不需要 explicit terrain estimation。

ANYmal、Cassie 这些 quadruped 早就用这个 trick, 但 humanoid 领域很少明确讨论。Sprout 把这个 trick 带进 humanoid context, 是从 legged robot community 借鉴的 standard practice。

参考 Softmimic: https://arxiv.org/abs/2510.17792

### 4.4 Compliance: 不是 mode 而是 fundamental property

Paper 反复强调 compliance 是所有 mode 的 fundamental property。实现方式是 training 时所有 environment 都 inject random external force, reward 包含 "minimize counteraction to external force"。

对比 traditional impedance control: $F = M\ddot{x} + B\dot{x} + K(x - x_0)$, compliance 通过手动调 $K$ (stiffness) 和 $B$ (damping)。Sprout 让 RL policy 隐式学习 variable impedance, 在不同 contact scenario 下自动调整。

这个 trade-off 是: 你 lose 了 explicit interpretability, gain 了 generality。对于 "deploy 到 unknown environment" 这个目标, generality 更重要。但如果你要做 medical robot 需要严格 force bound, traditional impedance control 仍然更合适。

---

## 五、VR Teleop + DAgger: 这才是 paper 的战略核心

### 5.1 VR Teleop 怎么工作

Meta Quest 3 → Unity app (Embody) → Meta Movement SDK → body pose keypoints → retargeting → robot。

**Calibration 是关键**: 启动时 estimate 用户的 arm length、torso height、squatting range, 用于 isomorphic retargeting。人 1.7 米, robot 1.07 米, 不能直接 mapping, 必须 scale。

Retargeting 数学: VR wrist keypoint (pelvis-relative frame) → robot frame, 用 scaling factor $s$ 缩放, 然后用 PINK library 解 IK:

$$\min_{\mathbf{q}} \quad w_p \|\text{FK}(\mathbf{q}) - \mathbf{p}_{\text{target}}\|^2 + w_o \|\text{FK}_{\text{orient}}(\mathbf{q}) - R_{\text{target}}\|^2 + w_r \|\mathbf{q} - \mathbf{q}_{\text{rest}}\|^2$$

- $\mathbf{q}$: joint configuration vector
- $\text{FK}$: forward kinematics (joint angle → Cartesian position)
- $w_p, w_o, w_r$: position、orientation、regularization weight
- $\mathbf{q}_{\text{rest}}$: rest pose (避免 singularity)

参考 PINK: https://github.com/stephane-caron/pink

### 5.2 DAgger: 解决 behavioral cloning 的致命问题

Behavioral cloning (BC) 有个 classic 问题叫 **covariate shift**。Demonstration data 集中在 expert trajectory 附近, policy deployment 时小 error accumulate, drift 到 unseen state, 然后 compounding error 让 policy 完全 fail。

DAgger (Dataset Aggregation, Ross 2011) 的 solution: deployment 时让 expert 介入, 在 failure state collect new data, 加入 training set 重训。

Sprout 的实现特别 elegant:

1. Policy 在 autonomous execution
2. Operator 戴 VR 监控
3. Policy 准备 fail 时, operator 按 button pause
4. **VR 中显示 "ghost controller" 在 robot 当前 pose**
5. Operator 把自己的 controller align 到 ghost
6. **从那个 exact pose 开始 demo correction**
7. Intervention segment 和 autonomous rollout 一起 log
8. Aggregate 重训 policy

为什么 "ghost alignment" 这么聪明? 因为如果你从 random pose 开始 correction, collected data 的 distribution 和 policy 实际遇到的 failure state distribution 不匹配。Ghost alignment 让 data 真正从 failure mode 开始, 这是 on-policy data。

这个设计的 strategic 意义超出 paper 本身。当前 humanoid VLA 的最大 bottleneck 是 demonstration data 不足。Open X-Embodiment 才 1M episodes, LLM pretraining 是 T tokens 级别。VLA 要 scale, 需要 100M+ episodes。

Sprout 的设计让每个 user 都是 potential data contributor。如果 Fauna 卖出 10000 台, 每台每天 1 小时 demo, 一年 36M hours demonstration data。这是 humanoid VLA 真正需要的 scale。这是 Sprout 最 long-term 的 play。

参考:
- DAgger: https://arxiv.org/abs/1011.0686
- π*0.6 (PI 最新 DAgger 应用): https://arxiv.org/abs/2511.14759
- TWIST2 data collection: https://arxiv.org/abs/2511.02832

---

## 六、Mapping & Navigation: 标准 pipeline 但有巧思

### 6.1 Fused Odometry EKF

50 Hz 输出, 融合四个 source:
1. ZED2i stereo camera (visual odometry)
2. IMU (high-rate angular velocity + linear acceleration)
3. **Learned proprioceptive velocity estimator** (RL policy 副产品)
4. Loop closure global position (when available)

EKF 标准 predict-update:

**Predict**:
$$\hat{\mathbf{x}}_{k|k-1} = f(\hat{\mathbf{x}}_{k-1|k-1}, \mathbf{u}_k)$$
$$P_{k|k-1} = F_k P_{k-1|k-1} F_k^T + Q_k$$

**Update**:
$$K_k = P_{k|k-1} H_k^T (H_k P_{k|k-1} H_k^T + R_k)^{-1}$$
$$\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + K_k (\mathbf{z}_k - h(\hat{\mathbf{x}}_{k|k-1}))$$
$$P_{k|k} = (I - K_k H_k) P_{k|k-1}$$

- $\hat{\mathbf{x}}$: state estimate (position + velocity + orientation)
- $P$: covariance (uncertainty)
- $F_k, H_k$: Jacobian of transition / observation
- $Q_k, R_k$: process / measurement noise covariance
- $K_k$: Kalman gain (权衡 prediction vs measurement)
- $\mathbf{z}_k$: measurement

**巧思在于用 learned proprioceptive velocity estimator 而非传统 leg odometry**。Bipedal locomotion 比 quadruped noisy得多——intermittent、asymmetric foot contact 让传统 leg kinematics 假设不成立。他们用 RL policy 的副产品作为 EKF 的一个 input, 这是 lateral thinking。

参考 Pronto: https://frontiersin.org/articles/10.3389/frobt.2020.00068

### 6.2 TSDF Volumetric Mapping + Maplet 架构

**TSDF (Truncated Signed Distance Field)**: 每个 voxel 存储 signed distance to nearest surface, truncated 到 $[-d_{\max}, +d_{\max}]$。比 point cloud 更 robust, 因为 weighted running average 自然 denoise。

更新公式:
$$\text{TSDF}_{\text{new}}(\mathbf{v}) = \frac{W_{\text{old}} \cdot \text{TSDF}_{\text{old}} + W_{\text{new}} \cdot \text{TSDF}_{\text{new meas}}}{W_{\text{old}} + W_{\text{new}}}$$

- $\mathbf{v}$: voxel 3D position
- $W$: voxel weight (confidence)
- TSDF: signed distance, 0 = surface, 负 = inside, 正 = outside

**Maplet 架构是关键创新**: 把世界分成 locally consistent submaps。每个 maplet 内部独立 TSDF fusion。新 maplet 触发条件:
1. Robot 移动超过当前 maplet 空间 extent (mitigate long-term drift)
2. Odometry 质量退化 (sudden pose jump)

这避免了 reconstruction error 累积污染 global map。Global alignment 用 GTSAM 做 pose graph optimization + async ML-based loop closure。

Paper 声称比 RTAB-Map 用 30% compute 达到更好效果, CPU 占用 "typically a third of a single core"。这是非常 impressive 的 efficiency。

参考:
- GTSAM: https://gtsam.org/
- GTSAM tutorial: https://arxiv.org/abs/1206.0526
- RTAB-Map: https://github.com/introlab/rtabmap

### 6.3 Navigation: Hybrid A* + Pure Pursuit

标准 pipeline, 但实现精细:

**Occupancy grid**: static layer (from mapping) + dynamic layer (OctoMap online update), 融合成 final grid。

**Hybrid A***: 比 A* 多了 continuous state space (x, y, θ), 用 Reeds-Shepp 或 Dubins curve 作为 motion primitive。Cost function:

$$f(n) = g(n) + h(n)$$

- $g(n)$: accumulated cost from start (distance + steering change + direction switch)
- $h(n)$: non-holonomic-without-obstacles heuristic

**Regulated pure pursuit**: 给定 lookahead distance $L_d$, 找 path 上 lookahead point, 计算 steering:

$$\delta = \arctan\left(\frac{2L \sin \alpha}{L_d}\right)$$

- $L$: wheelbase (相当于 step length)
- $\alpha$: angle from robot heading to lookahead point
- $L_d$: lookahead distance (调节这个改变 aggressive vs conservative)

10 Hz local planning cycle, 适合 bipedal locomotion 的动态特性。

参考:
- Hybrid A*: https://ai.stanford.edu/~ddolgov/papers/dolgov_gppp08.pdf
- Regulated pure pursuit: https://arxiv.org/abs/2306.13824
- OctoMap: https://octomap.github.io/

---

## 七、HRI: Slot-based Behavior Architecture

这是 paper 里我觉得 design 最 elegant 的部分之一。

### 7.1 问题: 怎么让 robot 同时做多件事

传统做法是 state machine: "alert state" preempt "idle state", 切换是 hard 的。问题: 你想让 robot 在 alert 时还能有 expression (比如 eyebrow 微微皱起表示 concern), state machine 做不到——要么 alert 要么 expression。

Sprout 用 **slot-based hierarchical behavior tree**。每个 control tick, 多个 computation node 同时 emit partial command 到不同 slot:
- LED slot
- Audio slot
- Body target slot
- Head target slot
- Eyebrow slot

Internal node 用 explicit merge policy combine children:

1. **Additive blending**: LED 可以叠加 (50% idle pulsing + 50% attention color)
2. **Select-one arbitration**: alert preempt expression (when necessary)
3. **Per-slot last-writer-wins**: 后来的 behavior override 前面的

Root level: alerts preempt expression + idle, 但 expression 和 idle 可以在 safe 时 contribution。这比 winner-take-all state machine 更 general, 允许 nuanced behavior composition。

### 7.2 这个 architecture 的 intuition

想象一个 stage play: 主角在说话, 同时 background 音乐响, 同时灯光 dim。三个 channel 独立但 synchronized。Sprout 的 HRI architecture 就是这个 concept——robot 的 LED、audio、motion、eyebrow 是独立 channel, 可以独立 control, 也可以 synchronize。

这是 robotics 里少见的 **explicit 多 channel composition** design。大多数 humanoid 的 HRI 是 hard-coded script, Sprout 是 composable system。这对 creative technologist 很友好——可以快速 prototype 不同 expression 组合。

---

## 八、Conversation: MCP 让 LLM 直接控制 robot

### 8.1 MCP (Model Context Protocol)

Anthropic 2024 年推出的 standard protocol for LLM tools。Sprout 把所有 robot capability 暴露为 MCP tools:
- Switch motor mode (walking, crawling, dancing, sitting)
- Call behavior (nod, shake, high-five, handshake)
- Navigate (query position, set goal)

LLM agent (Claude Desktop、Cursor、LangChain MCP adapter) 通过 standard MCP client 调用。这意味着你可以用任何 LLM framework 控制 Sprout, 没有 vendor lock-in。

这个 design 非常 aligned with 当前 LLM agent ecosystem 的发展方向。Sprout 不做自己的 agent framework, 而是暴露 standard interface 让 ecosystem 自然 grow。

参考: https://modelcontextprotocol.io/

### 8.2 ASR/TTS 选择

- **Wake-word**: openWakeWord (开源, "hey robot")
- **ASR**: Deepgram Flux 或 NVIDIA Riva (本地)
- **TTS**: ElevenLabs v2 或 NVIDIA Riva (本地)

Deepgram Flux 的特色是 **interruption handling**——conversational ASR 最大问题是 user 可以随时 interrupt, Flux 通过 streaming + endpointing 优化。ElevenLabs v2 是 voice clone quality 最高的 TTS 之一。

Cloud 还是 local 让 developer 选, 这符合 platform 哲学。

参考:
- openWakeWord: https://github.com/dscripka/openWakeWord
- Deepgram Flux: https://deepgram.com/learn/introducing-flux-conversational-speech-recognition
- ElevenLabs: https://elevenlabs.io/
- NVIDIA Riva: https://developer.nvidia.com/riva

---

## 九、Paper 在大图景里的位置

### 9.1 Humanoid paradigm 演进

**Era 1: Model-based control** (2000-2018): Atlas、HRP、Cassie 用 trajectory optimization + MPC + contact planning。可解释、可验证, 但 brittle to model error, hand-engineered。

**Era 2: Sim-to-real RL** (2018-now): ANYmal、Sprout、Unitree G1 用 IsaacLab + PPO。自动 discover gait, robust to perturbation, 但需要 accurate simulator 和 actuator model。

**Era 3: VLA** (2024-now): π0.5、Gemini Robotics、GR00T N1、Helix。Language-grounded, 泛化, 但 data hungry, unsafe to deploy。

Sprout 跨 era 2 和 3: low-level 是 era 2 (RL policies + IL transitions), high-level 接口预留 era 3 (MCP server for agents, DAgger data collection for VLA training)。

这是非常 pragmatic 的 design——用 era 2 解决今天能部署的问题, 同时为 era 3 铺路。

### 9.2 Platform layer 缺失是 humanoid bottleneck

你做 deep learning 这么多年, 应该对 platform layer 的价值有深刻 intuition。ImageNet 之前 vision 没有 standardized benchmark, 每个实验室自己 dataset, 进展缓慢。ImageNet 出来后, vision 突然指数级进步。

Humanoid robotics 现在处于 pre-ImageNet 阶段。每个 lab 自己 hardware、自己 dataset、自己 sim setup, 不可 reproduce。Sprout 想做的是 humanoid 领域的 ImageNet platform——standardized hardware + standardized software stack + standardized data collection protocol, 让任何 researcher 可以在上面迭代。

如果这个 vision 成功, humanoid progress 会从 linear 变成 exponential。这是 Sprout 的真正 thesis。

### 9.3 几个 critical thoughts

Paper 也有值得商榷的点:

**1. 没有实验结果**: 这是 platform paper 不是 research paper, 但缺少 quantitative sim-to-real gap measurement、policy training time、policy success rate。读者无法 evaluate "RL policies work well" 这个 claim。这比 Berkeley Humanoid paper 那种给具体 number 弱。

**2. 没说 cost**: Sprout 卖多少钱? Unitree G1 是 $16K, 如果 Sprout 是 $50K, "democratize access" 这个 thesis 弱化。Paper 完全没提 cost, 这是 suspicious 的 omission。

**3. 没说 open-source 程度**: SDK 暴露 APIs 但 hardware design 是 closed 还是 open? 如果 closed, "broad participation" 的 thesis 弱化很多。Open hardware 像 Berkeley Humanoid、ToddlerBot 让 community 可以 fork 和 improve, closed hardware 只能被动消费。

**4. Compliance 量化**: paper 说 "compliance 是 fundamental property" 但没给 force-disturbance response curve 或 quantitative safety metric (如 Head Injury Criterion)。这是 safety claim 没有定量 evidence。

**5. 29 DoF 具体分布**: paper 提到 "29 DoF including 2× eyebrows" 但没给详细 joint breakdown。这让 developer 难以评估 capability。

### 9.4 联想到的更大问题

**Robotics 会不会重演 AI 的 scaling law?**

LLM 成功来自 scale: more data + more compute + bigger model → emergent capability。VLA 是否同样? 如果是, Sprout 这种 "democratize data collection" 的 platform 战略就是正确的——谁能 crowdsource most demonstration data 谁赢。如果 robotics 需要 structured priors 和 physical understanding, 单靠 scaling 不够, 还需要 algorithmic breakthrough。

π0.5 显示 VLA 在 generalization 上有 promising 信号, 但还远没到 LLM 的 scaling regime。Sprout 的押注是: scaling 会在 humanoid 上 work, 只是 data bottleneck 还没解决。

**Sim-to-real gap 是 irreducible 吗?**

Sprout 高度依赖 IsaacLab sim training, 但 actuator model + calibration + domain randomization 是 "patching reality gap"。长期看, sim fidelity 会不会最终赶上 real, 让这些 patch 不必要? 还是 sim-to-real 是 irreducible problem?

Bjelonic 2025 那篇综述讨论这个问题, 没有定论。我个人倾向 sim-to-real gap 会持续存在, 因为 real world 有不可 reduce 的 noise (friction micro-variation、temperature-dependent motor behavior、human behavior)。但 sim-to-real patch 会越来越成熟, 让 gap 实用上可接受。

参考: https://arxiv.org/abs/2509.06342

**Expressive humanoid 是 feature 还是 uncanny valley 陷阱?**

Mori 的 uncanny valley 假说: robot 越 human-like, 越容易触发 revulsion, 直到完全 human-like 才恢复 acceptance。Sprout 用 LED + eyebrows 而不是 realistic face, 实际上是在 uncanny valley 之前 stop——它是 stylized 而不是 realistic。

HRI research 显示 stylized robot expressivity 确实提升 trust, 但前提是 expression 真实反映 internal state。Sprout 的 expression 是 state-machine-driven, 与 robot 的 "understanding" 解耦。这是 honest (robot 不会假装有 emotion) 还是 deceptive (robot 表达 emotion-like signal 但实际没有)? 这是 open philosophical question。

参考: https://humanrobotinteraction.org/

---

## 十、最终直觉总结

让我把所有这些 compress 成几个 takeaway:

**1. Sprout 是 humanoid 领域的 Apple II, 不是 humanoid 领域的 GPT-4。** 它做的是 platform, 不是 algorithm。价值在 system integration 和 developer accessibility, 不在单个 technical breakthrough。

**2. Hybrid control (FSM + RL + IL transitions) 是当前 safety-critical deployment 的 pragmatic 最优解。** 纯 end-to-end VLA 还不能 guarantee safety, 纯 model-based control 又不够 robust。Sprout 的 hybrid 让你既有 RL 的 expressive power, 又有 FSM 的 explicit safety envelope。

**3. VR Teleop + DAgger 是 humanoid VLA 的 data flywheel。** 这是 paper 最 long-term 的 play。如果 Fauna 卖出 10000 台 Sprout, 每台每天产生 1 小时 demonstration data, 一年就是 36M hours。这是 humanoid VLA scale up 需要的 data volume。

**4. Platform layer 缺失是 humanoid 当前 bottleneck。** 类似 pre-ImageNet 时代的 vision。Sprout 想做 humanoid 的 ImageNet platform, 让任何 researcher 可以在上面迭代。如果成功, humanoid progress 会从 linear 变成 exponential。

**5. Size choice (1.07m / 22.7kg) 是最 deliberate 的 safety design。** 动能 $E_k = \frac{1}{2}mv^2$ 让 mass 减半 energy 减半。这是 hardware-level safety, software 解决不了。Sprout 把 safety 放在 capability 之前, 这是 platform 哲学的核心。

**6. Slot-based HRI architecture 是被低估的 contribution。** 大多数 humanoid HRI 是 hard-coded script, Sprout 是 composable multi-channel system。这对 creative technologist 友好, 让 HRI prototype 从 days 变成 hours。

**7. MCP-based agent integration 是 future-proof design。** Sprout 不做自己的 agent framework, 而是暴露 standard MCP interface 让 ecosystem 自然 grow。这是 platform 思维, 不是 product 思维。

---

如果让我给 Fauna Robotics 一个 advice: **把 hardware design 开源**。Closed hardware 让 "broad participation" thesis 弱化。Open hardware 像 Berkeley Humanoid、ToddlerBot 让 community 可以 fork 和 improve, 加速 ecosystem growth。Open hardware 不影响 robot 销售——Pi 卖得很好尽管 Raspberry Pi design 是 open 的。Sprout 的 value 在 integration 和 software, 不在 hardware design secret。

如果让我给 developer 一个 advice: **关注 DAgger data collection 这部分**。这是 humanoid VLA scaling 的 infrastructure。如果你能 early build on Sprout, 积累 demonstration data, 你会在 humanoid VLA era 有 data moat。

Andrej, 你做 neural net 多年, 应该对 "platform 让 progress 从 linear 变成 exponential" 这个 phenomenon 有深刻 intuition。ImageNet 之于 vision, CUDA 之于 deep learning, Hugging Face 之于 NLP, 都是 platform layer 出现后 progress 突然加速。Sprout 想做的是 humanoid 领域的同一件事。能不能成功, 取决于 Fauna 能不能让足够多 developer 上车。这是 network effect 问题, 不是 technical 问题。

---

**核心参考 link 整理**:

Platform:
- Sprout: https://faunarobotics.com
- Berkeley Humanoid: https://arxiv.org/abs/2504.05658
- ToddlerBot: https://arxiv.org/abs/2502.00893
- Rodney Brooks: https://rodneybrooks.com/why-todays-humanoids-wont-learn-dexterity/
- Reachy Mini: https://www.pollen-robotics.com/reachy-mini/

Sim & Control:
- IsaacLab: https://isaac-sim.github.io/IsaacLab
- MuJoCo: https://mujoco.org/
- Genesis: https://github.com/Genesis-Embodied-AI/Genesis
- Legged Gym (Rudin 2022): https://arxiv.org/abs/2109.11978
- Softmimic (Margolis 2025): https://arxiv.org/abs/2510.17792
- Force control (Portela 2024): https://arxiv.org/abs/2405.01402
- Sim-to-real gap (Bjelonic 2025): https://arxiv.org/abs/2509.06342
- Dynamics randomization (Peng 2018): https://arxiv.org/abs/1710.06537

VLA & Data:
- π0.5: https://arxiv.org/abs/2504.16054
- π*0.6 (DAgger): https://arxiv.org/abs/2511.14759
- GR00T N1: https://arxiv.org/abs/2503.14734
- Figure Helix: https://www.figure.ai/news/helix
- Gemini Robotics 1.5: https://arxiv.org/abs/2510.03342
- DAgger: https://arxiv.org/abs/1011.0686
- TWIST2: https://arxiv.org/abs/2511.02832

Software Stack:
- ROS 2: https://docs.ros.org/en/rolling/
- RMW Zenoh: https://github.com/ros2/rmw_zenoh
- CBOR: https://cbor.io/
- Foxglove: https://foxglove.dev/
- PINK IK: https://github.com/stephane-caron/pink
- MCP: https://modelcontextprotocol.io/

Mapping & Nav:
- GTSAM: https://gtsam.org/
- GTSAM tutorial: https://arxiv.org/abs/1206.0526
- RTAB-Map: https://github.com/introlab/rtabmap
- OctoMap: https://octomap.github.io/
- Pronto: https://frontiersin.org/articles/10.3389/frobt.2020.00068
- Hybrid A*: https://ai.stanford.edu/~ddolgov/papers/dolgov_gppp08.pdf
- Regulated pure pursuit: https://arxiv.org/abs/2306.13824

Conversation:
- openWakeWord: https://github.com/dscripka/openWakeWord
- Deepgram Flux: https://deepgram.com/learn/introducing-flux-conversational-speech-recognition
- ElevenLabs: https://elevenlabs.io/
- NVIDIA Riva: https://developer.nvidia.com/riva

希望这个 "人话版" 对你 build intuition 有帮助。如果你对某个具体 subsystem 想再 deep dive (比如 EKF numerical stability、PPO reward shaping for compliance、TSDF vs NeRF mapping、或 hybrid FSM-RL 的 formal verification), 我可以继续展开。

---

# Fauna Sprout: 深度技术解读

Andrej, 这篇 paper 的核心 contribution 我觉得非常清晰——它不试图在算法层面推进 SOTA, 而是试图填补 humanoid robotics 缺失的 "platform layer"。这让我想到 computing 历史中 personal computer 出现之前的状态: 大型机昂贵、稀有、需要 PhD 操作员, 类似今天 Tesla Optimus、Figure 02 的处境。Sprout 试图做的是 humanoid 领域的 Apple II——让 educator、designer、creative technologist 也能参与。下面我会深入拆解每一层, 帮你建立 intuition。

---

## 1. Platform 定位与设计哲学

### 1.1 核心 thesis

Paper 反复强调三个 pillar: **safety**、**expressivity**、**developer accessibility**。三个 pillar 背后是一组很 explicit 的 trade-off:

| Trade-off | Sprout 的选择 | 代价 |
|-----------|--------------|------|
| Size vs capability | 1.07 m / 22.7 kg | 限制 reach、payload、kinetic envelope |
| Dexterity vs robustness | 1-DoF gripper (12 N tip force) | 放弃 in-hand manipulation |
| Sensing richness vs simplicity | 头部 RGB-D only, 无 wrist camera | 限制视觉伺服精度 |
| End-to-end vs modular | State machine + RL policies + IL transitions | 失去 monolithic VLA 的 simplicity |

这些 trade-off 都指向同一个目标: 让 robot 在 human-shared space 中可以 **fall、be pushed、be squeezed** 而不造成伤害。Rodney Brooks 2025 年那篇 "Why today's humanoids won't learn dexterity" 的论点被 paper 直接引用 [1]——dexterity 不是当下瓶颈, **safe deployment** 才是。

参考: https://rodneybrooks.com/why-todays-humanoids-wont-learn-dexterity/

### 1.2 与同类 platform 的对比

Paper 把现有 humanoid 分成两类: **open academic** (ToddlerBot、OP3、Berkeley Humanoid、ARTEMIS) 和 **proprietary industrial** (Atlas、Optimus、Figure、Agility、Unitree G1/R1、Booster T1/K1)。Sprout 想占据中间地带——既 open 又 manufacturable 又 safe。

一个有意思的对比维度是 **head design**:
- 大多数 industrial humanoid 用 featureless glossy surface (Atlas、Figure 02、Unitree G1)
- Anki、Enchanted Tools、Moxie 用 digital screen
- Sprout 用 **physical LEDs + actuated eyebrows**, 类似 Reachy Mini

这个选择背后是有认知科学依据的: digital screen 让人联想到 device 而不是 entity, 物理的 eyebrows + lights 保留了 embodied agent 的 affordance。Reachy Mini (Pollen Robotics) 的设计思路在这里被继承。

参考: https://www.pollen-robotics.com/reachy-mini/

---

## 2. Hardware 深度解析

### 2.1 三层 safety 架构

Paper 给出的 safety 设计是 **defense-in-depth**, 这是航空航天和自动驾驶领域的标准做法:

1. **Mechanical/Electrical layer**: 低 mass、低 kinetic energy、soft exterior、minimized pinch points、backdrivable motors、12 N fingertip force limit
2. **Embedded software layer**: dedicated safety subsystem on independent embedded processors, ToF obstacle sensing (VL53L8CX), system-level constraints enforcement 即使 application-level fault
3. **Application/policy layer**: compliant motor control policies, vision-based navigation safety

这里的 intuition 是: 不能依赖任何单层。即使 Jetson 上跑的 RL policy 完全 crash, embedded MCU 仍然可以通过 ToF 检测到 obstacle 并 power-cut。

### 2.2 Kinetic energy 的数学

Sprout 的 mass m = 22.7 kg, 假设行走速度 v ≈ 0.5 m/s, 则 translational kinetic energy:

$$E_k = \frac{1}{2} m v^2 = \frac{1}{2} \times 22.7 \times 0.25 = 2.84 \text{ J}$$

对比: Tesla Optimus (≈73 kg) 以 1.5 m/s 行走时 E_k ≈ 82 J, 差 30 倍。这就是 paper 强调 "kinetic energy reduction" 的核心论据——impact force 在 collision 中正比于 √(E_k / stopping_distance), 低 E_k 直接降低 injury severity。

### 2.3 Payload 分析

Supplementary Table II 给出了 per-arm payload:

| Configuration | Max (<10s) | Nominal (<90s) | Rated (indefinitely) |
|---------------|------------|----------------|----------------------|
| Forward Raise | 3.7 kg | 1.75 kg | 0.7 kg |
| Lateral Raise | 5.6 kg | 2.5 kg | 0.0 kg* |
| Bent Elbow | 11.2 kg | 5.0 kg | 1.1 kg |

这个递减反映 **thermal limit**——长时间负载下 motor winding 温升 dominate。Bent Elbow 时 leverage 最小所以 payload 最大。Lateral Raise 不能无限期保持是因为 shoulder abduction 时 abductor motor 持续 high current, 没有自然 cooling posture。

### 2.4 Compute architecture

NVIDIA Jetson AGX Orin 64GB 是主 compute, 通过 Ethernet 与 custom Motor Control Modules (MCMs) 通信。这个设计的关键是 **deterministic latency separation**:
- AGX 跑 perception、planning、high-level decision (workload bursty, soft real-time)
- MCM 跑 time-critical control loops (deterministic, hard real-time, 不受 AGX 负载影响)

通信协议选择:
- **CBOR over Ethernet** for MCM: CBOR (Concise Binary Object Representation, RFC 8949) 比 JSON 紧凑 30-50%, 比 Protobuf 更灵活, 适合 control messaging
- **ROS 2 + RMW Zenoh** for inter-process on AGX: Zenoh 比 default DDS 在 WAN/distributed 场景下性能更优
- **WebRTC** for VR headset: 低 latency video/audio
- **Foxglove WebSocket** for visualization

参考: 
- CBOR: https://cbor.io/
- RMW Zenoh: https://github.com/ros2/rmw_zenoh
- Zenoh: https://zenoh.io/
- Foxglove: https://foxglove.dev/

### 2.5 Power 系统

- 46.8 V nominal DC bus
- Molicel P50B cells (高 energy density, 5000 mAh nominal / 10000 mAh extended)
- Runtime 3-3.5 hours
- Commercial BMS

P50B 是 Molicel 2024 推出的高功率密度 21700 cell, ~3.6V nominal, 5000 mAh, 30A continuous discharge。46.8V 意味着 13S configuration (13 × 3.6 = 46.8V)。13S2P 给 5000 mAh, 13S4P 给 10000 mAh。

---

## 3. Software 架构: Component-based AI Platform

### 3.1 模块化 vs End-to-End 的辩证

Paper 在 Section II.B 明确把 control paradigm 分成两派:
1. **Monolithic VLA**: Gemini Robotics、π0.5、Generalist AI、RT-1, 单一 policy 从 perception 到 actuation
2. **Modular/Hierarchical**: NVIDIA GR00T N1、Figure Helix、Boston Dynamics+TRI Atlas, 高层 vision-language reasoning emit commands 给 low-level control policy

Sprout 的选择是 **modular**, 但 paper 也保留了 monolithic 的可能性——SDK 暴露 stable APIs 让开发者可以 swap 整个 stack。这是一个非常 pragmatic 的姿态。

参考:
- GR00T N1: https://arxiv.org/abs/2503.14734
- Figure Helix: https://www.figure.ai/news/helix
- π0.5: https://arxiv.org/abs/2504.16054
- Gemini Robotics 1.5: https://arxiv.org/abs/2510.03342

### 3.2 Container 化与 resource isolation

每个 service 跑独立 Docker container, core safety/control services 使用 CPU affinity + scheduling policy + cgroup limits 实现 **hard resource isolation**。这保证即使 perception service 因为大模型 inference 卡住, control loop 仍然按时跑。

Fig. 4 显示 CPU 占用 snapshot: baseline stack 大概用 50% 一颗 AGX Orin 核 (8 核 Cortex-A78AE @ 2.0 GHz), 剩余留给 user application。GPU profiling 没给, 但可以推断 perception + mapping + neural loop closure 共占用相当大比例。

### 3.3 Inter-process communication 优化

关键优化:
- **Shared memory / zero-copy** for 高带宽 images、point clouds: 避免序列化开销, latency 从 ms 级降到 μs 级
- **Hot paths 用 C++, Python bindings for ergonomics**: 这是非常常见的 robotics pattern, 比如 NVIDIA Isaac Gym、PyTorch C++ backend

---

## 4. Motor Control: State Machine + RL + IL Transitions

这是 paper 最技术 dense 的部分, 我重点展开。

### 4.1 整体架构: FSM orchestrating control modes

```
┌─────────────────────────────────────────────────┐
│  External caller (teleop / autonomy / agent)    │
└────────────────┬────────────────────────────────┘
                 │ request mode transition
                 ▼
┌─────────────────────────────────────────────────┐
│  FSM (validity checks, safety conditions)       │
└────────────────┬────────────────────────────────┘
                 │ current_mode
        ┌────────┴────────┐
        ▼                 ▼
  ┌─────────┐      ┌─────────────┐
  │ Walking │      │ Transition  │
  │ Policy  │      │ Policy (IL) │
  └─────────┘      └─────────────┘
        │                 │
        ▼                 ▼
  ┌─────────────────────────────────┐
  │ PD + Current Limit + Power Cap  │
  └─────────────────────────────────┘
                 │
                 ▼
            Actuators
```

设计哲学: **不依赖单一 monolithic controller**, 每个 mode 独立训练、独立验证、独立替换。这避免了 end-to-end policy 中一个 corner case crash 整个系统的风险。

### 4.2 Control Mode 内部结构

每个 mode 背后是 **one or more RL policies trained in IsaacLab**。Command interface 是 task-level:

$$\mathbf{c}_t = [v_x, v_y, \omega_z, \phi_{\text{roll}}, \phi_{\text{pitch}}, h_{\text{root}}, \mathbf{q}_{\text{upper}}^{\text{target}}]$$

变量含义:
- $v_x, v_y$: base linear velocity in body frame (m/s)
- $\omega_z$: yaw rate around vertical axis (rad/s)
- $\phi_{\text{roll}}, \phi_{\text{pitch}}$: desired root orientation
- $h_{\text{root}}$: desired torso-to-ground height (m)
- $\mathbf{q}_{\text{upper}}^{\text{target}}$: upper-body joint targets (arms, torso, neck, eyebrows)

Policy 输入 observation:
$$\mathbf{o}_t = [\mathbf{q}_t, \dot{\mathbf{q}}_t, \mathbf{a}_{t-1}, \text{IMU}, \mathbf{c}_t, \text{prev commands}]$$

Policy 输出 intermediate target $\mathbf{a}_t$, 被 PD + saturation 约束后送 actuator:

$$\boldsymbol{\tau}_t = K_p (\mathbf{a}_t - \mathbf{q}_t) + K_d (\dot{\mathbf{a}}_t - \dot{\mathbf{q}}_t)$$

变量:
- $\boldsymbol{\tau}_t$: motor torque vector
- $K_p, K_d$: per-joint PD gains (position, velocity)
- $\mathbf{a}_t$: policy 输出的 target position (中间量, 不是 raw torque)

这层 PD + current limit 是 **safety net**, 即使 RL policy 输出异常, 也被 bound 在 hardware-safe 范围内。

参考 IsaacLab: https://isaac-sim.github.io/IsaacLab

### 4.3 RL Training 细节 (推断 + paper 暗示)

Paper 没给完整 reward, 但暗示了几个 components:
- **Velocity tracking**: $\exp(-\|v_{\text{measured}} - v_{\text{command}}\|^2 / \sigma^2)$
- **Orientation tracking** via projected gravity (见 4.5)
- **Height tracking**: $\exp(-(h_{\text{measured}} - h_{\text{command}})^2 / \sigma_h^2)$
- **Energy penalty**: $\sum_i |\tau_i \dot{q}_i| \Delta t$ (mechanical work)
- **Smoothness penalty**: $\|\mathbf{a}_t - \mathbf{a}_{t-1}\|^2$
- **Compliance reward**: 在 random push disturbance 下 minimize corrective force

Training 是 standard PPO in IsaacLab, with **domain randomization** (mass, friction, motor torque constant, latency, external force), 对应参考文献 [32] Jakobi 1995 reality gap 和 [33] Peng 2018 dynamics randomization。

参考:
- PPO 原始: https://arxiv.org/abs/1707.06347
- Dynamics randomization: https://arxiv.org/abs/1710.06537
- Legged gym (Rudin 2022): https://arxiv.org/abs/2109.11978

### 4.4 Transitions as first-class citizens

这是 paper 最聪明的工程决策。问题描述: 每个 mode 独立训练, 各自的 state distribution 不同。直接 switch 会导致:
- Commanded motion discontinuity
- Safety constraint violation
- Excitation of unmodeled dynamics (oscillation, fall)

Solution: **每个 transition 是独立的 imitation-learned tracking policy**, training data 来自 human motion capture + animated trajectories。每个 transition 是一个短 motion sequence, 把 robot 从 source mode 的 state distribution 平滑带到 destination mode 的 state distribution。

实现细节:
- Transition 也有自己的 validity + safety conditions, 可能不同于 source 或 destination mode
- Training 时从 nominal trajectory 周围 broad distribution random initialization, 保证 generalization
- 这避免了 hard-coding transition logic, 同时比 monolithic policy 更 robust

Intuition: 这相当于在 mode 之间建立 "bridging policies"。每个 bridge 是独立 trained, 独立 testable, 独立 replaceable。Fig. 5 是这个 transition graph 的可视化。

### 4.5 Projected gravity 作为 control signal

Teleoperation 模块中提到一个关键技术点: 用 projected gravity 而非 absolute orientation 作为 control signal。

数学上, 设 $R \in SO(3)$ 是从 world frame 到 body frame 的旋转矩阵, $\mathbf{g}_{\text{world}} = [0, 0, -g]^T$ 是 world frame 中的 gravity 向量, 则:

$$\mathbf{g}_{\text{body}} = R^T \mathbf{g}_{\text{world}}$$

变量:
- $R$: body-to-world rotation matrix (3×3)
- $\mathbf{g}_{\text{body}}$: gravity 在 body frame 中的表示, 是 3 维向量
- $g \approx 9.81$ m/s²

**为什么这重要**: $\mathbf{g}_{\text{body}}$ 是 **yaw-invariant**——robot 绕 vertical axis 旋转, $\mathbf{g}_{\text{body}}$ 不变。这意味着 policy tracking $\mathbf{g}_{\text{body}}$ 可以在地形变化 (slope、stairs) 时自动保持 stable posture relative to gravity, 而不需要 explicit terrain estimation。这是 ANYmal、Cassie 等 legged robot 的 standard trick, 但在 humanoid context 很少被明确讨论。

参考 Margolis & Agrawal 的 Softmimic (ref [16]): https://arxiv.org/abs/2510.17792

### 4.6 Compliance as fundamental property

Paper 在 4.4 节提出 compliance 不是 mode 而是所有 mode 的 fundamental property。实现方式:
- Training 时所有 environment 都 inject random external forces
- Reward 包含 "minimize counteraction to external force"
- 这让 policy 学会 "yield appropriately"

对比 traditional impedance control: $F = M \ddot{x} + B \dot{x} + K(x - x_0)$, compliance 通过调节 $K$ (stiffness) 和 $B$ (damping) 实现。Sprout 的做法是让 RL policy 隐式学习一个 variable impedance, 在不同 contact scenario 下自动调整——这是更 general 的方案, 但牺牲了 explicit interpretability。

参考 Portela 2024 force control: https://arxiv.org/abs/2405.01402

### 4.7 Sim-to-real calibration

Paper 强调 calibration 不是可选 polish, 而是 sim-to-real 的必要前提。三个层级:

1. **Motor zero-referencing**: 用 precision calibration table 确保 encoder-to-geometry mapping 一致
2. **Camera extrinsic**: fiducial marker + self-calibration
3. **Actuator models**: DC motor dynamics + delay + saturation + power constraints, refined from dynamometer data

Actuator model 推断形式:
$$\tau_{\text{actual}} = \min(\tau_{\text{max}}, K_t \cdot i_{\text{motor}}, \tau_{\text{command}} \cdot \eta_{\text{gear}})$$
with delay $\delta t$:
$$\tau_{\text{actual}}(t) = f(\tau_{\text{command}}(t - \delta t))$$

变量:
- $\tau_{\text{actual}}$: 输出 torque
- $K_t$: motor torque constant
- $i_{\text{motor}}$: motor current (受 current limit 约束)
- $\eta_{\text{gear}}$: gearbox efficiency
- $\delta t$: actuator delay (通常 5-20 ms)

参考 Bjelonic 2025 sim-to-real: https://arxiv.org/abs/2509.06342

### 4.8 Description formats

SDK 同时提供 USD (Isaac Sim)、URDF (ROS)、MJCF (MuJoCo), 这是很务实的选择——不同社区有不同生态, 不强求统一。

参考:
- MuJoCo: https://mujoco.org/
- Genesis: https://github.com/Genesis-Embodied-AI/Genesis

---

## 5. Whole-Body Teleoperation + Data Collection

### 5.1 Embody App 架构

Meta Quest headset → Unity app → Meta Movement SDK → body pose keypoints → robot retargeting。

**Calibration 流程**: 启动时 estimate 用户的 arm length、torso height、comfortable vertical range, 用于 isomorphic retargeting。这是关键——人 vs robot 形态不同, 必须做 retargeting 而非直接 mapping。

**Modes**: Full-body (default)、upper-body only、seated。可以切换到 kneeling/sitting posture 同时保留上半身 teleop。

### 5.2 Retargeting 数学

从 VR wrist keypoints (pelvis-relative frame) 到 robot frame:

$$\mathbf{p}_{\text{robot}} = s \cdot \mathbf{p}_{\text{VR, pelvis-relative}}$$

其中 $s$ 是 calibration 计算的 scaling factor。然后通过 PINK (Python Inverse Kinematics based on Pinocchio) 解 IK:

$$\min_{\mathbf{q}} \quad w_p \|\text{FK}(\mathbf{q})_{\text{wrist}} - \mathbf{p}_{\text{target}}\|^2 + w_o \|\text{FK}(\mathbf{q})_{\text{orient}} - R_{\text{target}}\|^2 + w_r \|\mathbf{q} - \mathbf{q}_{\text{rest}}\|^2$$

变量:
- $\mathbf{q}$: joint configuration vector
- $\text{FK}$: forward kinematics
- $w_p, w_o, w_r$: position, orientation, regularization weights (避免 singularity)
- $\mathbf{q}_{\text{rest}}$: rest pose for regularization

Height 映射: VR standing/crouching heights linearly mapped 到 policy 的 min/max height range。

参考 PINK: https://github.com/stephane-caron/pink

### 5.3 DAgger-style interventions

这是 paper 最值得关注的 ML contribution。**Covariate shift problem** 在 behavioral cloning 中是经典 issue——demonstration 数据集中在 expert trajectory 附近, 但 deployment 时 policy 误差积累后 drift 到 unseen states, 导致 compounding errors。

DAgger (Dataset Aggregation, Ross et al. 2011) 的核心思想: 在 policy deployment 时让 expert 介入标注新 states, 把这些新 states 加入 training set 重训。Sprout 的实现非常 elegant:

1. Policy 在 autonomous execution
2. Operator 通过 VR 监控, 发现 policy 进入 OOD state 准备 fail
3. **Pause policy playback**, 在 VR 中 project "ghost controller" 显示 robot 当前 pose
4. Operator align 真实 controller 到 ghost, **从 exact failure pose 开始** demo correction
5. 这些 intervention segments 与 autonomous rollout 一起 log, 形成 on-policy data
6. Aggregate 重训 policy

这个 "ghost alignment" 设计很聪明——避免了 "operator 从 random pose 开始 correction" 造成的 distribution mismatch, 让 collected data 真正 from failure mode distribution。

参考:
- DAgger 原始 paper: https://arxiv.org/abs/1011.0686
- π*0.6 (PI 最新 DAgger 应用): https://arxiv.org/abs/2511.14759

### 5.4 数据采集 spec

- Stereo RGB @ 30 Hz
- Whole-body pose commands @ 50 Hz (end-effector Cartesian, gripper, pelvis pitch/roll/height, joint position, velocity)
- Proprioceptive state @ 125 Hz
- Annotation button for segmenting usable trajectories

125 Hz proprioceptive 是关键——这是 control loop 频率, 保证 collected state 足够 dense 用于 training high-frequency policies。

---

## 6. Mapping & Navigation

### 6.1 Fused Odometry EKF

50 Hz 输出, 融合四个 source:
1. ZED2i stereo camera (visual odometry)
2. Onboard IMU (high-rate angular velocity + linear acceleration)
3. **Learned proprioceptive velocity estimator** (从 motor control policy 输出)
4. Loop closure global position (when available)

EKF 标准方程:

**Predict**:
$$\hat{\mathbf{x}}_{k|k-1} = f(\hat{\mathbf{x}}_{k-1|k-1}, \mathbf{u}_k)$$
$$P_{k|k-1} = F_k P_{k-1|k-1} F_k^T + Q_k$$

**Update**:
$$\mathbf{y}_k = \mathbf{z}_k - h(\hat{\mathbf{x}}_{k|k-1})$$
$$S_k = H_k P_{k|k-1} H_k^T + R_k$$
$$K_k = P_{k|k-1} H_k^T S_k^{-1}$$
$$\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + K_k \mathbf{y}_k$$
$$P_{k|k} = (I - K_k H_k) P_{k|k-1}$$

变量:
- $\hat{\mathbf{x}}_{k|k-1}$: predicted state at time $k$ given measurements up to $k-1$
- $P_{k|k-1}$: predicted covariance
- $F_k$: Jacobian of state transition $f$
- $H_k$: Jacobian of observation function $h$
- $Q_k$: process noise covariance
- $R_k$: measurement noise covariance
- $K_k$: Kalman gain
- $\mathbf{y}_k$: innovation (prediction residual)
- $S_k$: innovation covariance

**关键 insight**: 用 learned proprioceptive velocity estimator 而非传统 leg kinematics。这是因为 bipedal locomotion 比 quadruped 更 noisy——intermittent、asymmetric foot contact 让传统 leg odometry 假设不成立。学到的 estimator 隐式 modeling 了 contact dynamics。

参考 Pronto (Camurri 2020): https://frontiersin.org/articles/10.3389/frobt.2020.00068
Cerberus (Yang 2023): https://arxiv.org/abs/2306.17275

### 6.2 TSDF Volumetric Mapping

**Truncated Signed Distance Field**: 每个 voxel 存储 signed distance to nearest surface, truncated 到 $[-d_{\text{max}}, +d_{\text{max}}]$ 提升计算效率。

数学: 对 voxel center $\mathbf{v}$ 和 depth measurement $\mathcal{D}$:
$$\text{TSDF}(\mathbf{v}) = \text{trunc}\left(\text{dist}(\mathbf{v}, \mathcal{D})\right)$$
$$W(\mathbf{v}) = \text{weight (confidence)}$$

更新采用 weighted running average:
$$\text{TSDF}_{\text{new}}(\mathbf{v}) = \frac{W_{\text{old}} \cdot \text{TSDF}_{\text{old}} + W_{\text{new}} \cdot \text{TSDF}_{\text{new meas}}}{W_{\text{old}} + W_{\text{new}}}$$

变量:
- $\mathbf{v}$: voxel 3D position
- $\mathcal{D}$: depth image + camera pose
- $d_{\text{max}}$: truncation distance (typically 5-10 cm)
- $W$: voxel weight

### 6.3 Maplet 架构

Paper 的关键架构创新: 把世界分成 locally consistent submaps (maplets), 每个 maplet 是 rigid non-deforming coordinate frame, 内部独立 TSDF fusion。

新 maplet 触发条件:
1. Robot 移动超过当前 maplet 预定义空间 extent (mitigate long-term drift)
2. Odometry 质量退化 (sudden pose jump)

这避免了 reconstruction errors 累积污染 global map, 同时保持 local fusion 高效。

### 6.4 Hierarchical pose graph + GTSAM

Global alignment 通过 hierarchical pose graph:
- Keyframe poses (节点)
- Maplet poses (节点)
- Odometry constraints (边)
- Visual loop closure constraints (边)

GTSAM (Georgia Tech Smoothing and Mapping) 后端用 factor graph 优化:
$$\min_{\Theta} \sum_i \|\mathbf{e}_i(\Theta)\|_{\Sigma_i}^2$$

变量:
- $\Theta$: 所有 pose variables
- $\mathbf{e}_i$: residual of factor $i$
- $\Sigma_i$: noise covariance of factor $i$

参考 GTSAM: https://gtsam.org/
参考 GTSAM tutorial: https://arxiv.org/abs/1206.0526

### 6.5 Performance 对比 RTAB-Map

Paper 声称 Volumetric Mapping 用 ~30% RTAB-Map 的 compute 达到更好 reconstruction + loop closure consistency。RTAB-Map 是 ROS 生态最常见的 dense mapping lib, 这个对比意味着 Sprout 通过 maplet 架构 + async loop closure 显著提升了 efficiency。CPU 占用 "typically a third of a single core"——非常 impressive。

参考 RTAB-Map: https://github.com/introlab/rtabmap

### 6.6 ML-based loop closure

Cascaded neural networks:
1. Visual place recognition (类似 NetVLAD)
2. Feature detection (类似 SuperPoint)
3. Feature matching (类似 SuperGlue)

异步跑, 不阻塞 real-time mapping。

### 6.7 Navigation stack

三层:
1. **Occupancy grid**: static layer (from mapping) + dynamic layer (OctoMap online update), 融合成 final grid
2. **Path planner**: Hybrid A* (global + local), 10 Hz local planning cycle
3. **Path tracker**: Regulated pure pursuit

**Hybrid A* cost function**:
$$f(n) = g(n) + h(n)$$
- $g(n)$: accumulated cost from start to $n$ (distance + steering change + direction switch)
- $h(n)$: heuristic, 通常 non-holonomic-without-obstacles distance (Reeds-Shepp 或 Dubins)

**Pure pursuit**:
给定 lookahead distance $L_d$, path 上找 lookahead point $\mathbf{p}_{\text{lookahead}}$, 计算 steering:
$$\delta = \arctan\left(\frac{2L \sin \alpha}{L_d}\right)$$

变量:
- $L$: wheelbase (这里相当于 step length)
- $\alpha$: angle from robot heading to lookahead point
- $L_d$: lookahead distance

参考 Hybrid A* (Dolgov 2008): https://ai.stanford.edu/~ddolgov/papers/dolgov_gppp08.pdf
参考 Regulated pure pursuit (Macenski 2023): https://arxiv.org/abs/2306.13824
参考 OctoMap: https://octomap.github.io/

---

## 7. Human-Robot Interaction: Hierarchical Behavior Tree

### 7.1 Slot-based command architecture

Fig. 9 展示了一个 hierarchical graph 的 computation nodes。每个 leaf node 输出 **slot-specific partial command**, 例如:
- LED patterns slot
- Audio cue slot
- Body target slot
- Head target slot
- Show-element pose slot

每个 control tick, eligible nodes run + emit slot outputs + propagate upward。Internal nodes 用 explicit merge policy combine children。

### 7.2 Merge operators

支持三种 merge:
1. **Additive blending**: layered LED effects (e.g., 50% idle pulsing + 50% attention color)
2. **Select-one arbitration**: exclusivity required (e.g., "alert" overrides "idle")
3. **Per-slot last-writer-wins**: later behavior override earlier

这是 **比 state machine 更 general** 的架构——允许同时多 behavior 作用不同 slot, 同时 high-priority safety behavior 能可靠 preempt。

### 7.3 优先级机制

Root level: **alerts preempt expression + idle layers**, 但 expression 和 idle layers 可以在 safe 时 contribution。这避免了 "all-or-nothing" 的硬切换。

---

## 8. Conversation & Reasoning: MCP Server

### 8.1 三个核心组件

- **Wake-word detector**: openWakeWord (开源, "hey robot")
- **ASR**: Deepgram Flux 或 NVIDIA Riva
- **TTS**: ElevenLabs v2 或 NVIDIA Riva

Deepgram Flux 的特色是 **interruption handling**——conversational ASR 中最大问题是用户随时可以 interrupt, Flux 通过 streaming + endpointing 优化这一点。

参考:
- openWakeWord: https://github.com/dscripka/openWakeWord
- Deepgram Flux: https://deepgram.com/learn/introducing-flux-conversational-speech-recognition
- ElevenLabs v2: https://elevenlabs.io/blog/eleven-multilingual-v2
- NVIDIA Riva: https://developer.nvidia.com/riva

### 8.2 MCP (Model Context Protocol)

这是 Anthropic 2024 推出的 standard protocol for LLM tools。Sprout 把所有 robot control capability 暴露为 MCP tools:
- Switch motor mode (walking, crawling, dancing, sitting)
- Higher-level behaviors (nod, shake, high-five, handshake)
- Navigation (query position, set goal)

LLM agent 通过 standard MCP client 调用这些 tools, 实现 agentic control。这让 Sprout 可以接入任何支持 MCP 的 agent framework (Claude Desktop、Cursor、LangChain MCP adapters)。

参考 MCP: https://modelcontextprotocol.io/

### 8.3 Think-Act-Observe loop

Fig. 10A 显示标准 reasoning loop, Fig. 10B 显示 state machine for both reasoning 和 conversational turn-taking, 用于 trigger 自动 HRI responses (listening、speaking、thinking)。

---

## 9. 直觉构建: 这篇 paper 在 robotics 大图景中的位置

让我后退一步, 帮你建立更宏观的 intuition。

### 9.1 三种 humanoid paradigm 的演进

1. **Model-based control era** (2000-2018): Atlas、HRP、Cassie 用 trajectory optimization + MPC + contact planning。优点: 可解释、可验证; 缺点: hand-engineered、brittle to model error
2. **Sim-to-real RL era** (2018-now): ANYmal、Sprout、Unitree G1 with IsaacLab。优点: 自动 discover gait、robust to perturbation; 缺点: 需要 accurate simulator
3. **VLA era** (2024-now): π0.5、Gemini Robotics、GR00T N1、Helix。优点: language-grounded、泛化; 缺点: data hungry、unsafe to deploy

Sprout 跨越 era 2 和 3: low-level 是 era 2 (RL policies + transitions), high-level 接口预留 era 3 (MCP server for agents, DAgger data collection for VLA training)。

### 9.2 与 Berkeley Humanoid、ToddlerBot 的关系

Berkeley Humanoid (Liao et al. 2025) 也是 mid-size learning-based platform, 但更偏 research-only, 没 human-interactive focus。ToddlerBot (Shi et al. 2025) 类似, 强调 open-source ML-compatible 但形态更小更 toy-like。Sprout 的差异化是:
- 强调 safe human deployment (soft exterior, compliant control)
- Expressive head (eyebrows, LED, audio) for HRI
- 完整 software stack (teleop, mapping, nav, HRI, conversation)
- Manufacturability 设计

参考:
- Berkeley Humanoid: https://arxiv.org/abs/2504.05658
- ToddlerBot: https://arxiv.org/abs/2502.00893

### 9.3 VR Teleop + DAgger = 数据飞轮

这是 paper 最有战略意义的部分。当前 VLA 的最大瓶颈是 **demonstration data 不足**。Open X-Embodiment 才 ~1M episodes, 而 LLM pretraining 数据是 T tokens 级别。

Sprout 的设计: VR teleop 让 non-expert 可以 demo + DAgger 让 expert 介入 correction + 数据格式 ready for VLA training。每个 Sprout 用户都是潜在 data contributor。如果 Sprout 卖出 10,000 台, 每台每天 1 小时 demo, 一年就是 36 M hours demonstration data——这是 humanoid VLA 真正需要的 scale。

类似思路见 TWIST2 (Ze et al. 2025): https://arxiv.org/abs/2511.02832

### 9.4 State Machine + RL + IL 的混合架构启示

我认为 paper 这部分对未来 humanoid 部署有重要启示。**纯 end-to-end VLA 的 safety 难题**还没解决——你无法 guarantee 一个 monolithic neural network 在 OOD 时不发 unsafe command。Sprout 的 hybrid 架构:
- Mode-level FSM 给出 explicit safety envelopes
- Mode 内 RL policy 实现 high performance
- Transition IL policy 保证 mode 间平滑切换
- Bottom layer PD + current limit 是 hardware-level safety net

这是 ** layered safety architecture**, 类似自动驾驶的 ODD (Operational Design Domain) + fallback + emergency brake 三层。

### 9.5 一些 critical thoughts

Paper 也有值得商榷的点:

1. **No experimental results section**: 这是 platform paper 不是 research paper, 但缺少 quantitative sim-to-real gap measurement、policy training time、policy success rate 等数据。读者无法 evaluate "RL policies work well" 这个 claim。

2. **Comparison to Unitree G1**: G1 是 1.32m / 35kg, 比 Sprout 大但也更便宜 (~$16K) 且同样有 RL community 支持。Paper 没明确说明 Sprout 的 cost positioning, 这对 developer adoption 至关重要。

3. **Compliance 量化**: paper 说 "compliance 是 fundamental property" 但没给 force-disturbance response curve 或定量 safety metric (如 Head Injury Criterion)。

4. **Open-source 程度**: SDK 暴露 APIs 但 paper 没说 hardware design 是否 open。如果是 closed hardware, "platform for broad participation" 的 thesis 弱化很多。

5. **29 DoF 的具体分布**: paper 提到 "29 DoF including 2× eyebrows" 但没给详细 joint breakdown, 这对开发者评估 capability 很关键。

---

## 10. 联想到的更大问题

这篇 paper 让我思考几个 deep question:

### 10.1 Robotics 会不会重演 AI 的 "scaling law" 故事?

LLM 的成功来自 scale: more data + more compute + bigger model → emergent capability。VLA 是否同样? 如果是, 那么 Sprout 这种 "democratize data collection" 的 platform 战略就是正确的——谁能 crowdsource most demonstration data 谁就赢。如果不是 (e.g., robotics 需要 structured priors、physical understanding), 那么 platform 战略就不够, 还需要 algorithmic breakthrough。

π0.5 paper 显示 VLA 在 generalization 上有 promising 信号, 但还远没到 LLM 的 scaling regime: https://arxiv.org/abs/2504.16054

### 10.2 Sim-to-real 还会持续多久?

Paper 高度依赖 IsaacLab sim training, 但 actuator model + calibration + domain randomization 是 "patching reality gap"。长期看, sim fidelity 会不会最终赶上 real, 让这些 patch 不再必要? 还是 sim-to-real 是 irreducible problem?

参考 Bjelonic 2025 综述讨论: https://arxiv.org/abs/2509.06342

### 10.3 "Expressive humanoid" 是 feature 还是 distraction?

Paper 强调 eyebrows + LED + audio for HRI。但这是 anthropomorphism陷阱? 当 robot 表达 "emotion" 但实际是 scripted behavior, 是否会造成 trust misalignment? Mori 的 uncanny valley 假说在这里仍然适用。

HRI 研究 (e.g., Bethel 实验室 work) 显示 robot expressivity 确实提升 trust, 但前提是 expression 真实反映 internal state。Sprout 的 "expression" 是 state-machine-driven, 与 robot 的 "understanding" 解耦——这是 honest 还是 deceptive?

参考: https://humanrobotinteraction.org/

### 10.4 Modular vs End-to-End 终局

我倾向于认为 modular 在 safety-critical domain (医疗、家居、公共空间) 会持续 dominant, end-to-end VLA 会在 industrial / warehouse 等 controlled environment 主导。Sprout 选择 modular 符合它 "deploy in shared human spaces" 的目标。

但 longer term, 如果 VLA 实现 verifiable safety guarantees (e.g., through conformal prediction or formal verification of NN outputs), modular 的优势会弱化。这是 open research question。

参考 Figure Helix 的 hierarchical VLA: https://www.figure.ai/news/helix
参考 NVIDIA GR00T N1: https://arxiv.org/abs/2503.14734

---

## 总结

Sprout 这篇 paper 价值不在于 algorithmic novelty, 而在于 **system-level integration 的清晰呈现**。它把 sim-to-real RL、VR teleop、DAgger、EKF SLAM、TSDF mapping、Hybrid A* navigation、HRI behavior tree、MCP-based agent control 这些 components 整合成一个 coherent、可部署、可迭代 platform。

从你 (Karpathy) 的视角, 我觉得最有意思的三点是:
1. **DAgger-style teleop data collection** 是 humanoid VLA scaling 的基础设施
2. **State machine + RL + IL hybrid** 是当前 safety-critical deployment 的 pragmatic 最优解
3. **Platform layer 缺失** 是 humanoid robotics 当前 bottleneck, 类似 pre-ImageNet 时代 vision 缺乏 standardized benchmark

如果你打算深入这篇 paper, 我建议从 Section IV.B (Motor Control) 和 Section V.B (Teleoperation + DAgger) 开始读, 这两部分 ML content 最 dense。

---

**References (full):**

- Sprout 主页: https://faunarobotics.com
- IsaacLab: https://isaac-sim.github.io/IsaacLab
- MuJoCo: https://mujoco.org/
- Genesis: https://github.com/Genesis-Embodied-AI/Genesis
- ROS 2: https://docs.ros.org/en/rolling/
- RMW Zenoh: https://github.com/ros2/rmw_zenoh
- CBOR: https://cbor.io/
- Foxglove: https://foxglove.dev/
- PINK IK: https://github.com/stephane-caron/pink
- GTSAM: https://gtsam.org/
- GTSAM tutorial: https://arxiv.org/abs/1206.0526
- RTAB-Map: https://github.com/introlab/rtabmap
- OctoMap: https://octomap.github.io/
- MCP: https://modelcontextprotocol.io/
- openWakeWord: https://github.com/dscripka/openWakeWord
- Deepgram Flux: https://deepgram.com/learn/introducing-flux-conversational-speech-recognition
- ElevenLabs: https://elevenlabs.io/
- NVIDIA Riva: https://developer.nvidia.com/riva
- DAgger: https://arxiv.org/abs/1011.0686
- Berkeley Humanoid: https://arxiv.org/abs/2504.05658
- ToddlerBot: https://arxiv.org/abs/2502.00893
- GR00T N1: https://arxiv.org/abs/2503.14734
- Figure Helix: https://www.figure.ai/news/helix
- π0.5: https://arxiv.org/abs/2504.16054
- π*0.6 (DAgger): https://arxiv.org/abs/2511.14759
- Gemini Robotics 1.5: https://arxiv.org/abs/2510.03342
- Dynamics randomization (Peng 2018): https://arxiv.org/abs/1710.06537
- Legged gym (Rudin 2022): https://arxiv.org/abs/2109.11978
- Softmimic (Margolis 2025): https://arxiv.org/abs/2510.17792
- Force control (Portela 2024): https://arxiv.org/abs/2405.01402
- Sim-to-real gap (Bjelonic 2025): https://arxiv.org/abs/2509.06342
- TWIST2 data collection: https://arxiv.org/abs/2511.02832
- Pronto (Camurri 2020): https://frontiersin.org/articles/10.3389/frobt.2020.00068
- Hybrid A* (Dolgov 2008): https://ai.stanford.edu/~ddolgov/papers/dolgov_gppp08.pdf
- Regulated pure pursuit (Macenski 2023): https://arxiv.org/abs/2306.13824
- Rodney Brooks blog: https://rodneybrooks.com/why-todays-humanoids-wont-learn-dexterity/
- Reachy Mini: https://www.pollen-robotics.com/reachy-mini/

希望这个深度解读对你有用, Andrej。如果某个具体 subsection 你想 deeper dive (e.g., EKF numerical stability, PPO reward shaping for compliance, TSDF vs Neural Radiance Field mapping), 我可以继续展开。
