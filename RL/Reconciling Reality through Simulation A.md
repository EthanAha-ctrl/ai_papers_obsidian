---
source_pdf: Reconciling Reality through Simulation A.pdf
paper_sha256: db7e57f6b298b535e912df1098c33669ffd4366935ea60c422cd161b1e0797ce
processed_at: '2026-08-11T21:46:54-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好，我用大白话重新讲一遍。

---

## 这篇论文到底在干嘛

想象你买了个机器人放家里，让它帮你把盘子放碗架、把书放书架。你教会它一次，它能做。但稍微有点变化就崩了——盘子挪了个位置，书架旁边多了个花瓶，或者抓盘子的时候手抖了一下盘子滑了，机器人就傻眼了。

这就是 imitation learning 的老毛病：**你 demo 里没出现过的状况，它不会处理**。

那怎么办？两条老路：

**第一条路：收集海量 demo**。RT-1、Open X-Embodiment 走的路。但你自己在家搞不出几万条 demo，而且就算搞出来了，policy 可能为了 cover 各种场景变得过于保守，在你自己这个特定场景反而不够 sharp。

**第二条路：让机器人在真实世界里自己试 RL**。问题更大——慢得离谱（机器人 trial and error 要几万次），危险（可能砸东西伤人），reward 你还得手动设计，reset 还得人帮它把东西摆回去。完全不 practical。

RialTo 的回答是：**把你家那个场景扫一遍，在 simulation 里建个 digital twin，然后让机器人在 simulation 里疯狂试，学会 robust 的 behavior，再 transfer 回真实世界**。

---

## 为什么这个想法听着简单，做起来难

难点有三个：

### 难点 1：怎么把真实场景变成 simulation 里能交互的 scene

你不能只搞个 NeRF 看着像就行。你需要在 simulation 里 **物理交互**——drawer 要能拉开，toaster 要能按下去，plate 要能抓起来放到 rack 上。这就要求：

- 几何准确（rack 的细金属杆不能扫成一根棍子，plate 要放得进去）
- object 是分离的（不能整个厨房是一个 mesh，你没法单独动 plate）
- articulation 要对（drawer 的 joint 在哪，转轴是什么方向，range 多大）
- physics 参数（mass、friction）至少差不多

RialTo 的做法很 pragmatic：用 iPhone 扫一遍（Polycam、AR Code、NeRFStudio 看场景选），然后一个简单 GUI 让人拖一拖、切一切、加个 joint。**用户研究显示 6 个完全没经验的人平均 15 分钟搞定**。这比收集 50 条 demo（1 小时 45 分钟）还快得多。

### 难点 2：你在真实世界里收集的 demo 没有 state 信息

RL 在 simulation 里跑，最 efficient 的做法是用 **privileged state**（所有 object 的 pose、joint angle 之类 low-dim 向量）做 input。如果你直接用 point cloud 做 RL input，**35 小时只学到 1% success rate**（论文 Appendix Fig 14），因为 point cloud encoder 太重，batch size 小 100 倍，rendering 慢 10 倍。

但你真实世界收集的 15 条 demo 只有 point cloud observation + action，**没有 object pose 这种 privileged state**。你怎么 bootstrap simulation 里的 RL？

RialTo 的 trick 叫 **Inverse Distillation**，三步：

1. 先用你那 15 条 real demo 训一个 perception policy $\pi_{real}(a|o)$（point cloud → action）
2. 把这个 policy 扔到 simulation 里 rollout，输入是 simulation 渲染的 point cloud
3. Rollout 的时候，simulation 同时记录每个 timestep 的 privileged state $s$（object pose、joint angle 等，simulation 里 a priori 就有）

这样你就拿到了 15 条 **带 privileged state 的 simulation demo** $\mathcal{D}_{sim} = \{(o, a, s)\}$。

**关键 insight**：虽然 real demo 里没 state，但你只要能 rollout perception policy 到 simulation 里，simulation 自动给你 state 标注。这就是为什么叫 "inverse" distillation——传统 distillation 是 teacher（有 state）→ student（只有 perception），这里是 student（real perception policy）→ teacher（sim privileged demos）→ 再训新 student。

有个 subtle point：如果 real policy 太烂 transfer 不进 simulation（比如 book on shelf 任务少于 15 条 demo 时），你 rollout 不出 successful trajectory，整个 pipeline 就断了。论文 Table X 显示这是个 **step function**——某个 demo 数量以下完全 0%，过了一个 threshold 突然 jump 到 90%。

### 难点 3：RL in simulation 学到的 behavior 可能 transfer 不了

如果直接 PPO from scratch（无 demo），policy 会 **exploit simulator 的不完美**。论文里有个很生动的例子（Appendix Fig 15）：toaster 的 joint 在重建 mesh 时位置稍微有点偏，RL from scratch 的 policy 学会了 **从底部推 toaster 让那个错位的 joint 工作** 来完成 "开门" 任务。这种 behavior 在真实世界完全没用——你 push 底部真 toaster 是不会开的。

RialTo 的解法：在 PPO loss 里加一个 BC loss against $\mathcal{D}_{sim}$：

$$
\mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{PPO} + \beta \cdot \mathcal{L}_{value} + \gamma \cdot \mathcal{L}_{BC}(\mathcal{D}_{sim})
$$

其中 $\gamma$ 是 BC weight（实验设 0.1），$\mathcal{L}_{BC}$ 就是 standard cross entropy：让 policy 在 $\mathcal{D}_{sim}$ 的 state 上输出和 demo action 接近的分布。

**这个 BC loss 的作用不是帮探索（虽然也有点），主要是 anchor policy 到 physical plausible behavior**。Demo 来自真实世界，已经 implicitly 只包含 physical 可行的动作。BC loss 把 policy 拴在这个 manifold 附近，不让它跑到 simulator bug exploit 的方向去。

实验结果（Table III）很 striking：RL from scratch 在 5 个 task 里 3 个 0%，剩下 2 个虽然能 succeed 但 behavior 不可 transfer。加 demo 的 RL fine-tuning 全部 80%+。

---

## 然后 transfer 回真实世界

Simulation 里学到的是 state-based policy $\pi_{sim}^*(a|s)$，但真实世界没 state，只有 camera point cloud。所以再做一次 **teacher-student distillation**：

- Teacher：simulation 里的 state-based policy
- Student：point cloud encoder + MLP → action
- 训练数据混合很关键：
  - 15000 条 **full point cloud** trajectory（从 mesh 直接采样，所有面都看得见，简单场景起步）
  - 5000 条 **camera viewpoint** trajectory（模拟真实相机位姿）
  - 2000 条 camera viewpoint + **distractor object**（增加 visual robustness）
  - 15 条真实 demo
  - 四类等概率 1/4 采样
- 然后再做 DAgger iteration，relabel action with teacher

**为什么还要 co-training real demo**？两个原因：
1. **Visual gap**：simulation point cloud 和 real point cloud 的 distribution 不一样（深度传感器噪声、missing point 等）。Real demo 帮 student 适应 real visual。
2. **Dynamics gap**：simulation physics 和 real physics 不完全一样。Real demo 让 student 学到的 action 更 conservative，更接近 real world 可行的。

实验（Figure 6）显示 co-training real data 在 sim-to-real gap 大的 task 上能提 3.5 倍 success rate。Qualitatively：没 co-training 的 policy 抓 plate 时贴太近导致 plate 掉下来，co-training 的 policy 会留更多 space。

---

## 最核心的实验结果

**Figure 7 是我最喜欢的实验**：在 Objaverse 里找 4 个不同的 drawer，训练 multi-task policy 想 generalize 到 target drawer，结果只有 10% success rate。而用 target drawer 自己的 digital twin 训练，90%。

**这个结果对 robotics 社区是个提醒**：在 object-level manipulation（比如 OpenAI 的 in-hand reorientation）上，大量 random objects + domain randomization 能 work。但在 **scene-level articulated manipulation** 上，精准的 in-domain digital twin 比盲目 diversity 强得多。因为 drawer 的 joint position、handle 形状、articulation range 都不同，4 个 drawer 的信号互相稀释，学不出 target drawer 的精确 behavior。

这对家用机器人 deployment 是个好消息：**你不需要造个万能 robot，你只需要 15 分钟扫一下你家，就能得到一个在你家 super robust 的机器人**。

---

## 几个我觉得 Karpathy 会觉得有意思的点

**1. 这个 pipeline 和 LLM 的 SFT → RLHF 几乎同构**

- 15 条 real demo = SFT data
- IL policy = SFT model
- Simulation RL + BC loss = RLHF with KL penalty to SFT model
- Teacher-student distillation = 把 privileged reasoning 蒸馏成 perception-based inference

**2. Inverse Distillation 就是 self-play / synthetic data generation**

用一个 weak model（real IL policy）在新 environment（simulation）里 rollout 生成 data，再用这个 data 训练 strong model（state-based RL policy）。和 LLM 里用 GPT-3.5 生成 SFT data 给 GPT-4 用一个套路。

**3. Co-training on real data 等价于 RLHF 里的 human preference anchor**

防止 RL drift 到 simulator exploit，就像 RLHF 里的 KL penalty 防止 policy 漂离 reference model。

**4. 2 天 wall clock time 还是不够 fast**

家用机器人要能 continual learning，遇到 failure 当场 fine-tune。现在 2 天太长。未来需要：
- 更快的 point cloud encoder（比如 PointNet 变体或 3D Gaussian Splatting 直接做 differentiable rendering）
- 更好的 parallel simulation（Isaac Sim 已经 GPU 加速了，但 distillation 的 rendering 还是 bottleneck）
- 可能用 world model（DreamerV3 之类）减少 rollout 需求

**5. Multi-task RialTo 的负结果有意思**

Table XII 显示 multi-task RialTo 在 mug on shelf 从 100% 掉到 80%，说明 **task 之间有干扰**。这和 LLM 里 multi-task instruction tuning 的 negative transfer 现象类似。可能需要更大的 network 或更聪明的 task balancing。

---

## 一句话总结

**用 15 分钟扫你家 + 15 条 demo，在 simulation 里 robustify 2 天，得到一个在你家 90% robust 的 manipulation policy**——这就是 RialTo。核心 trick 是 inverse distillation（用 perception policy 在 sim 里 rollout 反向获取 privileged state demos）+ BC anchored RL（防止 simulator exploit）+ real data co-training（弥补 sim-to-real gap）。比通用 foundation model 路线更适合家用 deployment，比纯 real-world RL 实用 100 倍。

---

# RialTo: Real-to-Sim-to-Real 用于 Robust Manipulation 的深度解析

这篇 paper 来自 MIT Improbable AI Lab 和 UW 的团队 (Marcel Torne, Anthony Simeonov, Zechu Li, April Chan, Tao Chen, Abhishek Gupta, Pulkit Agrawal)，提出了一个完整的 real-to-sim-to-real pipeline，用于 robustify 真实世界的 imitation learning policies。我会从 motivation、系统架构、核心算法、实验设计、相关联想几个维度展开。

## 一、Motivation 与核心 Insight

这篇工作要解决一个非常实际的 deployment 问题：一个在特定家庭场景部署的机器人，需要在 **该场景** 下 robust 地完成任务，能处理 object pose 变化、visual distractors、physical disturbances (比如 plate 在 gripper 里滑了、robot base 被推动)。

核心 insight 是: 与其训练一个 super general 的 policy 跨越多场景 (这条路 content creation + data collection 成本极高)，不如为 **特定 deployment scene** 快速构建一个 digital twin，在其中通过 RL 把已有的 imitation policy robustify 起来。

这里有两条路被巧妙结合起来：
- **Imitation Learning (IL)** 的优势：少量 demos (15 条左右) 就能 bootstrap 一个可工作的 policy，但 robustness 差，遇到 distractor 或 disturbance 就崩。
- **Reinforcement Learning (RL)** 的优势：能在 simulation 中自主探索，学到 recovery behaviors (re-grasping、re-aligning)，但直接在 real world 跑 RL 既慢又危险，且 reward 难设计。

RialTo 的核心是：把 IL 学到的 policy 作为 prior，把它 transfer 到 simulation 里作为 "privileged demos" 引导 RL fine-tuning，最后再 distill 回 perception-based policy。

## 二、系统架构 (Four-Stage Pipeline)

```
Real Scene  →  3D Reconstruction + GUI  →  Simulation Scene (USD/URDF)
                                              ↓
Real Demos (15)  →  IL Policy π_real(a|o)  →  Rollout in Sim  →  Privileged Demos D_sim (with Lagrangian state s)
                                              ↓
                                      RL Fine-tuning (PPO + BC loss)  →  π_sim*(a|s)
                                              ↓
                          Teacher-Student Distillation + Real Data Co-training  →  π_real*(a|o)
                                              ↓
                                          Real World Deployment
```

### Stage 1: Real-to-Sim Scene Construction

用 off-the-shelf 3D reconstruction 工具：
- **Polycam** [https://poly.cam](https://poly.cam) — 适合大场景，利用 iPhone LiDAR，但 fine details 差
- **AR Code** [https://ar-code.com](https://ar-code.com) — 适合 singulated object 360° 扫描
- **NeRFStudio** [https://nerf.studio](https://nerf.studio) — 适合需要精细几何的场景 (如 dish rack 的细金属杆)，用 nerfacto model + Poisson Surface Reconstruction

输出一个 globally-unified mesh G，需要进一步处理成 separated bodies $\{\mathcal{G}_i\}_{i=1}^M$ + kinematic relations $\mathcal{K}$ + physical parameters $\mathcal{P}$，最终得到 scene $\mathcal{S} = \{\{\mathcal{G}_i\}, \mathcal{K}, \mathcal{P}\}$。

GUI 关键功能: cut mesh、add joints (fixed/revolute)、设置 mass/friction defaults。User study 显示 6 个 non-expert 用户平均 14 分 40 秒 active time 就能建好一个 articulated scene。

### Stage 2: Inverse Distillation (核心创新点之一)

这个 step 是整篇 paper 最 trick 的地方。问题定义:

我们有 real-world demos $\mathcal{D}_{real} = \{(o_1^i, a_1^i), ..., (o_H^i, a_H^i)\}_{i=1}^N$，其中 $o$ 是 3D point cloud observation，$a$ 是 delta end-effector pose。我们想在 simulation 里跑 RL fine-tuning，但 RL 在 vision space 里跑太慢 (实验显示 RL from vision 35 小时只有 1% success，RL from compact state 12 小时 96% success)。

**Inverse distillation 流程**:
1. 在 $\mathcal{D}_{real}$ 上训练 perception-based policy $\pi_{real}(a|o)$ (就是 standard IL)
2. 把 $\pi_{real}$ 放到 simulation 里 rollout (输入是 simulation 渲染的 point cloud)，收集成功轨迹 $\mathcal{D}_{sim} = \{(o_1^i, a_1^i, s_1^i), ..., (o_H^i, a_H^i, s_H^i)\}_{i=1}^M$
3. 关键：simulation 中 $o$ 和 $s$ (Lagrangian state，含 object poses、joint angles) 是 a priori paired 的，所以 rollout 时能同时记录 privileged state

这样就把 "没标 state 的 real demos" 转化成 "有 privileged state 的 sim demos"。这个过程叫 "inverse" 是因为它从 perception policy 反向生成 privileged demos，而不是从 privileged teacher 蒸馏出 perception student (standard distillation 方向)。

### Stage 3: RL Fine-tuning in Simulation

用 $\mathcal{D}_{sim}$ bootstrap 一个 PPO 训练，policy 是个 2-layer MLP (256, 256)，input 是 privileged state，output 是 14-way categorical distribution (6 delta position ±0.03m, 6 delta rotation ±0.2rad, 2 gripper open/close)。

优化目标 (论文公式 1):

$$
\begin{aligned}
\max_{\theta, \phi} \quad & \alpha \sum_{(s_t, a_t, r_t) \in \tau_{\pi_\theta}} \min\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}\hat{A}_t, \mathrm{clip}\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}, 1-\epsilon, 1+\epsilon\right)\hat{A}_t\right) \\
& + \beta \sum_{(s_t, V_t^{targ}) \in \tau_{\pi_\theta}} (V_\phi(s_t) - V_t^{targ})^2 \\
& + \gamma \sum_{(s_t, a_t) \in \mathcal{D}_{sim}} \frac{\pi_\theta(a_t|s_t)}{\sum_{a_t}\pi_\theta(a_t|s_t)}
\end{aligned}
$$

变量解释:
- $\theta, \phi$: policy 和 value function 参数
- $\pi_\theta(a_t|s_t)$: 当前 policy 在 state $s_t$ 下选 action $a_t$ 的概率
- $\pi_{\theta_{old}}$: 上一轮 PPO iter 的 policy (importance sampling 的 reference)
- $\hat{A}_t$: advantage estimator at timestep $t$，由 GAE 计算
- $\epsilon$: PPO clip ratio (默认 0.2)
- $V_\phi(s_t)$: value function 预测的 expected return
- $V_t^{targ}$: value target (GAE-λ 的 target)
- $\alpha, \beta, \gamma$: 三项的权重 (PPO loss, value loss, BC loss)
- $\tau_{\pi_\theta}$: 当前 policy rollouts 的 trajectory batch
- $\mathcal{D}_{sim}$: inverse distillation 收集的 privileged demos

第一项是标准 PPO clipped surrogate objective；第二项是 value function MSE；第三项是关键创新——BC loss against $\mathcal{D}_{sim}$，作用是 bias policy toward physically plausible behaviors，避免 RL from scratch 时 policy 学会 exploit simulator 不准确 (比如 push toaster 底部 joint 错位来开门)。

实验 (Table III) 显示 RL from scratch 在 5 个 task 中 3 个 0% success，另外两个虽然能 succeed 但学到的行为不可 transfer。

### Stage 4: Teacher-Student Distillation with Real Data Co-Training

RL fine-tuning 得到的是 state-based policy $\pi_{sim}^*(a|s)$，但 real world 没有 privileged state。需要 distill 成 perception-based policy $\pi_{real}^*(a|o)$。

Point cloud encoder 用 Convolutional Occupancy Networks [https://github.com/convolutional-occupancy-networks](https://github.com/convolutional-occupancy-networks) 的架构: local point net → 3D U-Net → dense voxel grid → max pool + avg pool → concat → 128-dim embedding。然后 concat robot state (9-dim: EE pose + gripper state) → MLP (256, 256) → 14-way categorical。

Co-training objective (公式 2):

$$
\max_\theta \quad \alpha \sum_{(s_i, o_i, a_i) \sim \pi_\theta} \frac{\pi_\theta(\pi_{teacher}(s_i)|o_i)}{\sum_{a_c}\pi_\theta(a_c|o_i)} + \beta \sum_{(o_i, a_i) \in \mathcal{D}_{real}} \frac{\pi_\theta(a_i|o_i)}{\sum_{a_c}\pi_\theta(a_c|o_i)}
$$

变量解释:
- 第一项: DAgger-style training，用 $\pi_{teacher} = \pi_{sim}^*$ 在 simulation 里 rollout 并 relabel actions
- 第二项: supervised loss on real-world demos $\mathcal{D}_{real}$
- $\alpha, \beta$: 两项权重
- $\pi_{teacher}(s_i)$: teacher policy 在 state $s_i$ 下选的 action (relabel 用)

数据混合策略非常关键:
- 15000 trajectories with **full point clouds** (直接从 mesh 采样点，所有面可见，curriculum 起步)
- 5000 trajectories from **camera viewpoint** (与 real camera 位姿一致)
- 2000 trajectories from camera viewpoint + **distractor objects**
- 15 real-world trajectories

四类等概率 (1/4 each) 采样。然后做一轮 DAgger，混合 DAgger trajectories + sim distractor trajectories + real trajectories (1/3 each)。

## 三、实验结果的关键 insight

### 1. Robustness 三层难度
- **Pose randomization**: object/robot initial pose 随机
- **Distractors**: 加入 visual clutter
- **Disturbances**: episode 中途扰动 (move object, close drawer, move robot base)

RialTo 在 8 个 task 上平均 91% (randomization) / 77% (distractors) / 75% (disturbances)，BC (15 demos) 只有 25% / 11% / 5%。

### 2. Co-training 的影响 (Figure 6)
对 sim-to-real gap 大的 task (book on shelf, plate on rack)，real data co-training 比 sim data co-training 高 3.5x 和 2x。Qualitatively: no-co-training policy 会贴 plate 太近导致掉落，co-training policy 留更多 space，更保守更安全。

### 3. Real-to-Sim 的必要性 (Figure 7)
对比 "训练在 4 个 Objaverse drawers 多任务" vs "训练在 target drawer digital twin"：前者在 target drawer 上 10%，后者 90%。说明在 scene-level manipulation，targeted real-to-sim 比盲目 data diversity 更有效。这与 OpenAI 的 in-hand reorientation (用大量 random objects + domain randomization) 经验相反，因为 in-hand 是 object-level，而这里需要 scene-level 的 articulation 和 geometry precision。

### 4. Distractor Training (Table II)
Mug on shelf 任务: 不训练 distractor → 60% (pose rand) / 30% (distractors)，训练 distractor → 100% / 70%。说明 distractor training 不仅帮 distractor robustness，还帮 sim-to-real transfer 本身。

### 5. Real Demos vs Sim Demos (Figure 6, Table III)
RL fine-tuning 从 15 real demos (经过 inverse distillation) vs 从 15 sim demos 直接起步，结果几乎一样 (open toaster 91 vs 96, book 90 vs 89, plate 81 vs 82, mug 81 vs 82, drawer 96 vs 95)。说明 inverse distillation 成功把 real demos 转化成了等价的 sim demos，且 pipeline 灵活——可以从任意 source 起步。

### 6. Demos 数量的 Step Function (Table X)
Book on shelf: 0/5/10 demos 全 0%，15 demos 突然 90%。Drawer: 0% → 5 demos 89% → 10 demos 96%。原因: real policy 必须先能 transfer 到 sim 才能收集 sim demos；过不了这关，RL 就退化成 from scratch。Task 难度决定 threshold。

## 四、关键 Implementation Details

### Physics 设置
- Isaac Sim [https://developer.nvidia.com/isaac-sim](https://developer.nvidia.com/isaac-sim) + Orbit [https://isaac-orbit.github.io](https://isaac-orbit.github.io)
- Convex decomposition: 64 hull vertices, 32 convex hulls (默认)；dish rack 用 SDF 256 resolution
- Friction: 0.5 (dynamic + static), joint friction 0.1, mass 0.41kg (统一默认，通过 policy 约束 + demos 弥补 sim-to-real dynamics gap)

### Point Cloud Processing (Table VII)
- Total 6000 points: 3000 from arm mesh (利用 real robot joints 采样) + 1000 from object meshes + 其余 from camera
- Dropout ratio [0.1, 0.3], jitter ratio 0.3, jitter noise $\mathcal{N}(0, 0.01)$
- Grid size 32³ for voxel encoder
- Normalization: toaster 中心 [0,0,0] scale 0.625，其他中心 [0.35,0,0.4] scale 1

### 硬件
- Franka Panda (固定桌 + mobile table 两个)
- Polymetis [https://facebookresearch.github.io/fairo/polymetis](https://facebookresearch.github.io/fairo/polymetis) 做高低层通信
- Intel RealSense D455 / D435

### Compute
- RTX 2080 或 3090
- 总训练时间约 2 天 3 小时 (IL 7h + RL 20h + distillation 24h)

## 五、相关工作与延伸联想

### 与 Distillation 方向的关系
- **Teacher-Student Distillation** [Chen et al., Learning by Cheating, CoRL 2020] [https://arxiv.org/abs/1912.12294](https://arxiv.org/abs/1912.12294): 学习 by cheating 思想，teacher 用 privileged info，student 用 sensor。
- **Learning Cheating**: 这里和 RialTo 的 stage 4 完全一致，但 RialTo 多了 real data co-training。
- **TGRL** [Shenfeld et al., ICML 2023] [https://arxiv.org/abs/2306.12872](https://arxiv.org/abs/2306.12872): teacher guided RL，RialTo 的 RL+BC 是简化版。

### 与 Real-to-Sim Scene Reconstruction 的关系
- **Ditto** [Jiang et al., CVPR 2022] [https://arxiv.org/abs/2204.06232](https://arxiv.org/abs/2204.06232): 从 interaction 中构建 articulated digital twins。RialTo 用 human-in-the-loop GUI 替代自动化，更简单可靠。
- **URDformer** [Chen et al., CoRL 2023 Workshop] [https://urdformer.github.io](https://urdformer.github.io): 从真实图片生成 interactive URDF，自动化方向。
- **NeRF in the Palm** [Zhou et al., CVPR 2023] [https://arxiv.org/abs/2304.04308](https://arxiv.org/abs/2304.04308): corrective augmentation，用 NeRF 做数据增强。
- **Phone2Proc** [Deitke et al., CVPR 2023] [https://phone2proc.github.io](https://phone2proc.github.io): iPhone 扫描 → ProcTHOR 场景，主要服务 navigation。

### 与 Sim-to-Real Manipulation 的关系
- **Visual Dexterity** [Chen et al., Science Robotics 2023] [https://arxiv.org/abs/2210.13077](https://arxiv.org/abs/2210.13077): in-hand reorientation 用大量 random objects。RialTo 论文明确指出这条路径在 scene-level 上不 work (Figure 7)。
- **Dextreme** [Handa et al., ICRA 2023] [https://arxiv.org/abs/2211.01247](https://arxiv.org/abs/2211.01247): domain randomization + adapted curriculum for in-hand manipulation。
- **POCO** [Wang et al., 2024] [https://arxiv.org/abs/2402.02511](https://arxiv.org/abs/2402.02511): policy composition from heterogeneous learning，与 RialTo 的 co-training 思路有共鸣。

### 与 RL+Demos 的关系
- **AWAC** [Nair et al., NeurIPS 2020] [https://arxiv.org/abs/2006.09359](https://arxiv.org/abs/2006.09359): advantage-weighted regression from offline data。
- **Overcoming Exploration with Demos** [Nair et al., ICRA 2018] [https://arxiv.org/abs/1709.10089](https://arxiv.org/abs/1709.10089): 类似思路，RialTo 的 BC loss 是简化版本。
- **Rajeswaran et al. 2017** [https://arxiv.org/abs/1709.10087](https://arxiv.org/abs/1709.10087): dexterous manipulation with demos + RL，RialTo 引用作为 early bootstrap 工作。

### 与 Imitation Learning 的关系
- **Diffusion Policy** [Chi et al., RSS 2023] [https://arxiv.org/abs/2303.04137](https://arxiv.org/abs/2303.04137): RialTo baseline 之一，但没用 diffusion policy 而用更简单的 point cloud + MLP。
- **ACT / ALOHA** [Zhao et al., 2023] [https://arxiv.org/abs/2304.13705](https://arxiv.org/abs/2304.13705): bimanual low-cost，bigger data 不一定 robust。
- **RT-1** [Brohan et al., 2022] [https://arxiv.org/abs/2212.06817](https://arxiv.org/abs/2212.06817): large-scale real-world IL，但 RialTo 反对这种 scaling，主张 in-domain specialization。

### 与 RL in Real World 的关系
- **SERL** [Yang et al., 2023] [https://arxiv.org/abs/2310.15145](https://arxiv.org/abs/2310.15145): pre-training rewards + policies for autonomous real-world RL。
- **ASID** [Memmel et al., 2024] [https://arxiv.org/abs/2404.12308](https://arxiv.org/abs/2404.12308): active exploration for system identification，可作为 RialTo physics 参数标定的补充 (RialTo 现在用 uniform defaults)。
- **Autonomous RL with Human Feedback** [Balsells et al., 2023] [https://arxiv.org/abs/2310.20608](https://arxiv.org/abs/2310.20608): 同作者组的早期工作。

### Karpathy 视角: 这个工作和 LLM 范式的类比

如果你从 LLM 训练范式看这个工作，会发现一个很有意思的 parallel:

- **Pre-training (大量 unsupervised/supervised data) → SFT (small amount of demos) → RL fine-tuning (with sparse reward)** 这套 recipe 在 LLM 上是 ChatGPT 的标准 pipeline，在 RialTo 上是 IL → PPO+BC 的同构 pipeline。
- **InstructGPT** [Ouyang et al., NeurIPS 2022] [https://arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155) 的 SFT + RLHF 范式本质上和 RialTo 一样: 先用少量 demos 学个 baseline，再用 RL robustify。
- **Inverse Distillation** 这个概念其实和 LLM 里的 "self-distillation" 或 "synthetic data generation via base model rollout" 很像: 用一个 weak model (real IL policy) 在新 environment (sim) 中 rollout 生成 data，再用这个 data 训练 strong model (state-based RL policy)。
- **Co-training on real data** 类比于 RLHF 中的 preference data 保留人类监督，避免 RL 过度 optimize simulator exploit。
- **Domain Randomization vs Targeted Real-to-Sim** 类比于 general pretraining vs instruction tuning: 前者 broad，后者 specific。RialTo 明确指出在 scene-level manipulation 上，specific 比 broad 更 efficient (Figure 7)。

### 与 Foundation Model Robotics 的对比
- **RT-2** [https://arxiv.org/abs/2307.15818](https://arxiv.org/abs/2307.15818) / **Open X-Embodiment** [https://robotics-transformer-x.github.io](https://robotics-transformer-x.github.io) 主张 massive pretraining + cross-embodiment generalization。
- **RialTo 走相反路线**: 用 minimum human effort 在 specific deployment scene 做 specialization。两种范式都有道理，但 RialTo 对个人家庭机器人 deployment 更现实。
- **GenSim** [Wang et al., ICLR 2024] [https://arxiv.org/abs/2310.01361](https://arxiv.org/abs/2310.01361): 用 LLM 生成 sim tasks，是另一种 scaling 的方向。

## 六、Limitations 与 Future Directions

论文坦诚承认:
1. Depth sensor 对 thin/transparent/reflective 物体不行 → 需要更好 sensor 或 RGB 策略
2. 只支持 articulated rigid bodies → deformable 是未来 (参考 DiffCloud [https://arxiv.org/abs/2210.01805](https://arxiv.org/abs/2210.01805))
3. Quasi-static assumption → 高速 dynamic task 需要 system identification (ASID 可补)
4. 2 天 wall clock → continual learning 不友好，需要更高效 point cloud encoder + 更好并行
5. Physics 参数 fixed → 未来需要 active identification

延伸的几个可能方向 (我从 Karpathy 视角联想):
- **VLM as perception backbone**: 替换 point cloud encoder，用 CLIP/DINOv2 feature 做 representation，可能减少 sim-to-real visual gap
- **Diffusion Policy as student**: 当前 student 是 categorical MLP，换成 diffusion policy [https://diffusion-policy.cs.columbia.edu](https://diffusion-policy.cs.columbia.edu) 可能学到 multi-modal action distribution，对 recovery behavior 更友好
- **3D Gaussian Splatting as digital twin**: 比 NeRF mesh extraction 更快 + 更精细 [https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- **LLM-guided GUI**: 用户 cut mesh、add joint 的步骤可以由 LLM + vision model 自动化 (URDformer 方向)
- **World Model + RL**: 用 DreamerV3 之类 world model 替代纯 PPO，可能 sample efficiency 更高 [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)
- **Continual RialTo**: 把 RialTo 做成 online learning，deployment 中遇到 failure 自动扫描场景 + fine-tune
- **Multi-task RialTo scaling**: 论文 Table XII 显示 multi-task 略降，但 scaling 到更多 task 可能出现 foundation model 行为

## 七、最关键直觉总结

1. **Real-to-Sim 不只是 visual reconstruction，重点是 articulated + physical 可交互**: 这是为什么 NeRF-only 工作 [73] 不够，因为不能 physics interaction。
2. **Inverse Distillation 把"无 state 的 real demos"变成"有 state 的 sim demos"**: 这是 real-to-sim policy transfer 的核心 trick，类似 self-training 在 weak labeler 上生成 strong labeler data。
3. **BC loss in RL fine-tuning 不是为了 help 探索，是为了 prevent exploit simulator bug**: Table III 的 RL from scratch 学会推 toaster 错位 joint 开门，这种 behavior 不可 transfer。BC 是 physical plausibility prior。
4. **Co-training on real data 解决两个 gap**: visual distribution shift (sim point cloud vs real point cloud) + dynamics gap (sim physics vs real physics)。Figure 6 显示在某些 task 上 3.5x 提升。
5. **Curriculum in distillation dataset**: full pcd → camera viewpoint pcd → camera viewpoint + distractor → real demos，这是从 perfect perception 到 noisy real perception 的渐进式训练，是 student 训练成功的关键。
6. **Digital Twin 优于 random asset diversity (在 scene-level)**: Figure 7 是论文最有说服力的实验之一。object-level randomization work (in-hand reorientation)，但 scene-level articulated objects 需要精准 geometry + articulation，diversity 反而稀释学习信号。

## 八、Project Page & Code

- Project: [https://real-to-sim-to-real.github.io/RialTo/](https://real-to-sim-to-real.github.io/RialTo/)
- 论文应该会在 RSS 2024 或 CoRL 2024 出现 (从作者 affiliation 和时间推断)
- 相关 lab: MIT Improbable AI Lab [https://improbableai.lab](https://improbableai.lab) 和 UW WEIRD Lab [https://weirdlab.cs.washington.edu](https://weirdlab.cs.washington.edu)

这篇工作在 manipulation 领域代表了一种 "pragmatic specialization" 路线，与 RT-2 / Open X-Embodiment 的 "generalist foundation model" 路线形成有趣对比。RialTo 的 user study 数据 (14 分钟 active time 建场景) 让这个 pipeline 在实际家庭部署上有真实可行性，而不只是 lab demo。
