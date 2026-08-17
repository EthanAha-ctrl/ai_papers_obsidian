---
source_pdf: GenH2R Learning Generalizable Human-to-Robot Handover.pdf
paper_sha256: 6febebe8a2ee37368c38997b8acec07a5f8977c730c2bdefabe641d9bcb833d1
processed_at: '2026-08-04T14:31:12-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GenH2R 用人话讲

## 一句话说清楚这 paper 在干嘛

教 robot 接东西——人把东西递过来，robot 得接住。听起来简单，做起来巨难。这帮人用大规模 synthetic data 把这事儿干成了，sim-to-real 都 work，比 prior work 好一大截。

---

## 为什么这事儿难

你想想人跟人递东西的场景。我把一个杯子递给你，你会怎么做？

- 你用眼睛盯着我手和杯子的运动
- 你脑子在预测我下一步往哪走
- 你手提前伸出去，在合适时机合拢
- 我如果突然犹豫一下、改个方向，你也跟着调整

这整套是 closed-loop perception + prediction + action，而且 human motion 是 unpredictable 的。

Robot 要做同样的事，challenge 在哪？

**第一，data 不够。** HandoverSim 这个 prior work 只有 1000 个 scenes、20 个 objects，全靠 mocap 采集。Mocap 又贵又慢，你不可能 scale 到百万级。没有 diverse data，policy 就 generalize 不了。

**第二，demonstration 质量不只是"成功"。** 这点很 subtle，也是这 paper 最有意思的 insight。你用 planner 生成一堆成功轨迹，丢给 BC 去学，结果学出来一塌糊涂。为什么？因为 planner 生成的 action 可能跟视觉输入完全无关——它用了 privileged state（物体真实位姿），但 policy 只能看 point cloud。如果 demonstration 里"看不看都这么走"，policy 就学不到 vision-action 的对应关系。作者管这个叫 **distillability**。

**第三，4D 感知。** Handover 是动态过程，单帧 3D point cloud 不够，你得知道物体在过去几帧怎么动的，才能预测未来。但直接 frame stacking 又 lossy。

---

## 他们的解法：三件事一起 scale

### 1. GenH2R-Sim：用程序生成百万级 scenes

不搞 mocap 了，直接 procedural synthesis。

**Object 来源**：ShapeNet 里 3266 个 objects，覆盖从电脑到手机各种 size。

**Grasp pose 来源**：DexGraspNet 这个 prior work 用 optimization 生成了一百万个 human hand grasp pose。

**Motion trajectory**：用 Bezier curves 拼起来。Starting point 随机采，endpoints 随机采，中间用 Bezier curve 连，curve 的 key point 从高斯分布采。Translation speed 在 $[0.1, 0.2] \text{ m/s}$，angular speed 在 $[0.5, 1] \text{ rad/s}$。

这做出来的 trajectory 比 DexYCB 的 3 秒 mocap 复杂得多——8 秒、有转折、有犹豫、有方向变化，更像 real human。

**Reactive 机制**：当 robot gripper 离 object 小于 0.1m 时，human hand 停下来等。Prior work 没这个，就是无脑 replay 每一帧。这个 small detail 让 sim 更接近真实协作场景。

结果：1,000,000 scenes，比 HandoverSim 多三个数量级。

### 2. Demonstration 生成：Landmark Planning

这是 paper 最 core 的 technical contribution。

他们先分析了两种 naive 做法为啥不行：

**Destination Planning（foresighted）**：一开始就 plan 一条到终点的 smooth path。
- 问题：path 跟视觉完全脱节。Policy 看着物体往左走，但 demonstration 里 robot 直接往终点冲，学不出 vision-action correlation。
- 实验数据：在 complex t0 数据上 train，success rate 只有 0.93%，基本全挂。

**Dense Planning（shortsighted）**：每个 time step 都用 OMG planner 基于 privileged state replan。
- 问题：planner 是 multi-modal 的，同一个视觉状态可能对应完全不同的 robot trajectory。而且 robot morphology 约束导致 smooth vision 对应 zigzag action。
- 结果：success 能保住，但慢、不自然、distillation 效果差。

**Landmark Planning（他们的方法）**：在两个极端之间找 sweet spot。

核心 idea 分三步：

**Step A**：改进 dense planner，让它更 smooth。具体做法——按 grasp pose 到 end effector 的距离排序选最近的 grasp（保持 object 在视野里），IK 用上一帧的 arm pose 初始化（保证连续性），只有 IK 成功才调 OMG planner。

**Step B**：Trajectory resampling。OMG planner 输出固定 N 步 trajectory，但 step size 跟距离相关。Resample 成固定 step length $L$：

$$C'_i = \frac{s_{j+1} - iL}{s_{j+1} - s_j} C_j + \frac{iL - s_j}{s_{j+1} - s_j} C_{j+1}$$

这里 $C_j, C_{j+1}$ 是原始 trajectory 相邻两点（7D joint config），$s_j, s_{j+1}$ 是对应 accumulated length，$iL$ 是 resample 后第 $i$ 步的目标 length，$C'_i$ 是线性插值结果。这样不管原始距离多远，输出 trajectory 都有 consistent step size，vision-based model 不会被搞混。

**Step C**：稀疏 replanning + 看未来。训练一个 object pose forecasting network，找出 trajectory 上"过去预测不了未来"的转折点（endpoints）。用这些 endpoints 把轨迹分段。然后在每段内，每隔 $P$ 步 replan 一次，plan 的 target 是 $\hat{t} = \min(t+P, l_{i+1})$——未来的某个 landmark，但不超过 segment 边界。

Intuition：既不像 foresighted 那样看太远（到终点，破坏 vision-action correlation），也不像 shortsighted 那样只看当前（zigzag）。Landmark 是"能预测到的未来"，在这个范围内 plan。

### 3. Forecast-Aided 4D Imitation Learning

Policy 输入是 egocentric point cloud sequence。怎么把 4D 信息喂给网络？

**不用 frame stacking**，用 ICP（Iterative Closest Point）算 flow。对当前帧的每个点，用 ICP 估计它跟过去 $L_1$ 帧（paper 用 3）之间的 transformation matrix $\hat{M}_t^{t-j}$，然后把这个 transformation 应用到当前点，得到它过去几帧的大致位置。每个点的 feature 变成 $3 + 2 + 3 \cdot n$：当前坐标 3 维 + one-hot label 2 维 + 过去 $n$ 帧坐标 $3n$ 维。

为什么用 ICP 而不是 fancy 4D backbone？Paper 明说：P4Transformer、NSM4D 这些太慢，robot task 要 fast inference。ICP + PointNet++ 是 effectiveness 和 speed 的 sweet spot。

**网络结构**：PointNet++ encode → global feature → 两个 head
- Policy head：输出 6D egocentric action
- Prediction head：输出未来 $L_2$ 帧（也用 3）的 object pose transformation

**Loss**：
$$\mathcal{L} = \mathcal{L}_{\text{action}} + \lambda \mathcal{L}_{\text{pred}}$$

$\mathcal{L}_{\text{action}}$ 是 gripper 表面 3D points 的 L1 loss（来自 DeepIM）。$\mathcal{L}_{\text{pred}} = \sum_{i=t+1}^{t+L_2} \|\hat{M}_t^i - M_t^i\|$，预测未来 object pose transformation 跟 ground truth 的差。$\lambda = 0.1$。

为什么 prediction 有用？因为 demonstration 是基于 future landmark 生成的，policy 如果能预测未来 object motion，就等于直接 exploit 了 vision-action correlation。这个 auxiliary task 不是 multi-task learning 那种"顺便学点别的"，是直接 strengthen 主任务的 representation。

---

## 实验结果讲讲

### Main table 亮点

在 t0（大规模 synthetic）上 train 的 landmark planning：
- s0 Sequential：86.57% success
- s0 Simultaneous：85.65%
- t0：41.43%
- t1（real mocap HOI4D）：68.33%

对比 Handover-Sim2real* 在 s0 上 train：64.35%, 25.69%, 28.56%, 30.60%。

提升幅度 +11% 到 +16% 不等。在 t1（real data）上提升最猛，说明 synthetic data 真的能 generalize 到 real。

**Destination planning 在 t0 上彻底崩盘**（0.93% success），这很直观验证了 distillability 的重要性——complex trajectory 下，直接 plan 到终点完全破坏 vision-action correlation，policy 学不出来。

### Ablation 亮点

- 去掉 flow 信息：-9.77% success。4D 比 3D 强很多。
- 去掉 prediction loss：-2.25%。Forecasting objective 确实 help。
- Frame stacking 替代 ICP flow：-5.26%。ICP flow 比 naive stacking 好不少。
- 10% data：-5.93% on t1。Scale matters。

### Real-world

6 个 user、5 个 object、simple + complex 两个 setting：

| Setting | GA-DDPG | Handover-Sim2real | Ours |
|---------|---------|-------------------|------|
| Simple | 60% | 57% | 90% |
| Complex | 43% | 33% | 70% |

第二个 user study 用 15 个 novel object（含透明物体、奇怪形状）：Ours 72.2% vs baseline 28.9%~43.3%。

Complex setting 下 Ours 还能 70%，baseline 只剩 33%，这说明他们的 method 学到的 representation 真的更 generalizable，不是 overfit 到 simple pattern。

---

## 我的 Intuition 和联想

### 1. Distillability 这个概念很重要

通常我们说 demonstration quality 就看 success rate。但这 paper 指出还有个隐藏维度：vision-action correlation。一条成功但"看不见也这么走"的 demonstration 对 closed-loop policy 是有害的，因为它教 policy 忽略 vision。

这个 insight 其实跟 LLM 里一个现象同构：如果你用 privileged label 训练，模型可能学到 shortcut，不真正理解 input。这里 planner 用 privileged state 生成 action，如果 action 不跟 vision aligned，policy 就学不到真正的 vision-conditioned control。

延伸联想：RL 里的 offline RL 也有类似问题——如果 demonstration 来自不同 policy distribution，naive BC 会 fail。Distillability 是 demonstration 的一个被忽视的 quality dimension。

### 2. Landmark Planning 的 idea 可以泛化

"在能预测的范围内 plan" 这个 idea 很优雅。它本质是说：planning horizon 应该跟 predictability horizon 对齐。太短→zigzag，太长→跟 vision 脱节。

这个 idea 可以 apply 到很多 robot learning 场景。比如 autonomous driving——planning horizon 应该跟 perception 的可预测范围匹配。或者任何需要 privileged planner 生成 demonstration 再 distill 的 task。

联想：这跟 MPC（Model Predictive Control）的 receding horizon 有点像，但区别是 landmark planning 的 horizon 是 data-driven 决定的（用 forecasting error 找 endpoints），不是固定时间窗。

### 3. ICP Flow 是个 cheap but smart 的 trick

用 ICP 算 transformation matrix 当 flow feature，比 frame stacking 好，比 fancy 4D backbone 快。这是个典型的 engineering trade-off。

联想：在 robot learning 里，perception representation 的选择经常是 effectiveness vs speed 的 trade-off。PointNet++ + ICP flow 这个组合让我想到 FlowBot3D（https://flowbot3d.github.io/）也是用 flow 做 manipulation，效果不错。Flow 作为 intermediate representation 在 manipulation 里似乎被 underestimate 了。

### 4. Synthetic Data Scale > Real Data Variety

这 paper 最强的 claim 之一：100 万 synthetic scenes 比 1000 real mocap scenes 好。这跟 LLM 的 scaling law 呼应——在 robot learning 里，synthetic data scaling 可能比 real data curation 更 cost-effective。

联想：这跟 NVIDIA 的 Isaac Gym（https://developer.nvidia.com/isaac-gym）、Google 的 RT-2（https://robotics-transformer2.github.io/）方向一致。整个 embodied AI 社区都在往 synthetic scaling 走。GenH2R 是 handover 这个具体 task 上的一个 case study。

### 5. Prediction 作为 auxiliary task 的作用

Forecasting objective 在这 paper 里贡献 +2.25% success。这不算巨大，但方向一致。更深层的意义是：它让 encoder 学到 future-aware representation，这对 closed-loop control 是本质重要的。

联想：这跟 world model 的 idea（Dreamer 系列 https://danijar.com/project/dreamer/）有点像——学一个能预测未来的 representation 有助于 control。但 GenH2R 没做 full world model，只是 auxiliary prediction loss，更 lightweight。如果 future prediction 做得更强（比如用 diffusion 预测未来 point cloud sequence），可能进一步提升。

### 6. Reactive Sim 的重要性

GenH2R-Sim 加了"robot 靠近时 human 停下"这个 reactive 机制。Prior work（HandoverSim）没这个。这个 small detail 其实影响很大——它把 task 从 "chase and grab" 变成 "cooperative handover"。

联想：Real human-robot interaction 里，human 会根据 robot 行为调整自己。如果 sim 里 human 是 open-loop replay，policy 学到的就是 chase dynamic target，而不是协作。这个 reactive 机制是 sim2real 的关键 detail。未来可以做得更 complex——比如 human 感知到 robot 太快会缩手，这就涉及 intention modeling 了。

---

## 几个我想吐槽/追问的点

**1. Forecasting network 的细节没讲清。** Paper 说"train an object pose forecasting network"但没给架构。这个 network 怎么 train？用什么 architecture？Input/output 是什么？这是 reproducibility 的 gap。

**2. Landmark selection 的 threshold 怎么定？** Forecasting error threshold 决定 endpoints，但 paper 没说具体值。这个 hyperparameter 应该对 performance 敏感。

**3. 透明物体还是 fail。** Real-world experiment 里 transparent bottle/beaker 成功率低。这是 RGB-D 相机的固有限制。要解这个问题可能得换 sensing modality（比如 tactile sensing）或者用 transparent object 专门的 reconstruction method。

**4. 只测了 7-DoF arm。** Mobile manipulator（比如 Tiago、Stretch）没测。Handover 在更大 workspace 下 challenge 不同。

**5. 跟 Diffusion Policy 的对比缺。** Diffusion Policy（https://diffusion-policy.cs.columbia.edu/）现在 manipulation 里很强，paper 没跟它比。如果用 diffusion policy 替换 BC，可能能更好 handle multi-modal action distribution（landmark planning 本身可能也是 multi-modal 的）。

**6. Human intention modeling 缺失。** Paper 在 limitation 里承认了——sim 里 human 只在 robot 靠近时停下，没有更 complex 的 intention。Real handover 里 human 会根据 robot 速度、方向调整自己的递送方式。这块是 future work 的大方向。

---

## 相关工作链接

如果你想深挖：

- **HandoverSim** (prior simulator): https://github.com/NVlabs/handover_sim
- **Handover-Sim2real** (main baseline): https://github.com/seroyal19/handover-sim2real
- **DexYCB** (real mocap dataset): https://dex-ycb.github.io/
- **HOI4D** (4D HOI dataset, 用于 t1 test): https://hoi4d.github.io/
- **DexGraspNet** (grasp generation): https://github.com/PKU-EPIC/DexGraspNet
- **ShapeNet** (3D models): https://shapenet.org/
- **OMG-Planner** (grasp+motion planning): https://sites.google.com/view/omg-planner
- **PointNet++** (encoder backbone): https://arxiv.org/abs/1706.02413
- **DeepIM** (action loss 来源): https://arxiv.org/abs/1808.01285
- **ICP** (flow estimation): https://www.cs.princeton.edu/~smr/papers/icp.pdf
- **GenH2R 项目主页**: https://GenH2R.github.io
- **Diffusion Policy** (可能的 next step): https://diffusion-policy.cs.columbia.edu/
- **FlowBot3D** (flow-based manipulation): https://flowbot3d.github.io/
- **Dreamer** (world model 对比): https://danijar.com/project/dreamer/
- **RT-2** (large-scale robot learning): https://robotics-transformer2.github.io/

---

## Final Takeaway

这 paper 的核心 message 我觉得就一句：**在 robot learning from demonstration 里，scale 要 scale 对地方——不光 data 量要大，demonstration 还得 distillable，policy 还得能 exploit temporal structure。** 三件事一起做对，sim-to-real 才能 work。

Landmark planning 那个"在可预测范围内 plan"的 idea 我觉得是这 paper 最漂亮的贡献，它把 planning horizon 和 predictability horizon 对齐，这个 principle 可以泛化到很多 privileged-to-sensory distillation 的场景。

剩下就是工程执行到位——Bezier curve 生成 diverse trajectory、ICP flow 做 cheap 4D、prediction loss 做 auxiliary supervision、reactive sim 做 sim2real bridge。每一块单独看都不算惊天动地，但组合起来效果很 solid。

---

# GenH2R: Learning Generalizable Human-to-Robot Handover 深度讲解

## 1. Problem Motivation 与 Intuition Building

Human-to-Robot (H2R) handover 是 embodied AI 中一个核心能力：robot 需要根据 dynamic visual observations，可靠地从 human 手中接过以各种复杂 trajectory 移动的、geometry 未知的物体。这件事的 difficulty 在于 robot perception、motion planning、human intention 理解三者必须 closed-loop 耦合，而 human motion 是 unpredictable 的。

这篇 paper 的核心 thesis 是：**scaling** ——把 simulation scenes、demonstrations、policy learning 三件事都 scale up，就能在 sim-to-real 上获得显著的 generalization gain。这是一个和 LLM scaling law 同构的 insight，但 apply 到 robot learning 上时要解决独特 challenge。

为什么 prior work 难以 scale？以 HandoverSim [9] 为代表，它依赖 DexYCB mocap dataset，只覆盖 1000 个 scenes、20 个 objects。Mocap 数据 acquisition 成本极高，且 motion 多样性受限。GenH2R 选择 procedural synthesis 路线，scale 到 1,000,000 scenes + 3,626 objects，比 HandoverSim 多三个数量级的 scenes、两个数量级的 objects。

参考链接：
- HandoverSim: https://github.com/NVlabs/handover_sim
- DexYCB: https://dex-ycb.github.io/
- 项目主页: https://GenH2R.github.io

---

## 2. 框架总览（架构图解析）

整个 GenH2R pipeline 分三阶段（对应 Figure 1、Figure 2）：

**Stage 1 — GenH2R-Sim (Scalable Synthetic Handover Simulator)**
- 输入：ShapeNet [7] 3D models + DexGraspNet [45] grasp poses + Bezier curve trajectories
- 输出：1,000,000 个 handover scenes（包含 object geometry、grasp pose、human hand motion trajectory）

**Stage 2 — Distillation-friendly Expert Demonstration Generation**
- 输入：GenH2R-Sim scene（包含 privileged object 6D pose、human hand pose、candidate grasps）
- 输出：1,000,000 个 paired vision-action demonstrations，每个都是 distillable 的（vision-action correlation 强）

**Stage 3 — Forecast-Aided 4D Imitation Learning**
- 输入：4D point cloud sequence（egocentric camera）+ flow features
- 输出：6D egocentric action（gripper 的 closed-loop control）
- Aux output：future object pose prediction（forecasting objective）

三阶段串起来形成 sim-to-real 的完整 chain：synthetic scene → demonstration → policy → real robot deployment。

---

## 3. GenH2R-Sim 的技术细节

### 3.1 Grasp Pose 生成
利用 DexGraspNet [45]（arXiv:2210.02697），通过 optimization 生成 ~1,000,000 个 grasp poses，覆盖 3,266 个 ShapeNet objects。Object 类别从 large（computer）到 small（mobile phone），保证 size 和 shape 多样性。

### 3.2 Hand-Object Moving Trajectory 生成（核心创新）

**Translation 部分**：
- Starting point 从 region $[0.3, 0.9] \times [0, 0.2] \times [H+0.1, H+0.3]$ 均匀采样，其中 $H = 0.92$ 是 table surface 高度
- Endpoints 从 activity region $[0.1, 1.1] \times [-0.3, 0.1] \times [H+0.1, H+0.7]$ 采样
- 用 Bezier curves 连接 start 与 endpoints
- 每个 Bezier curve 的 key point 从以 midpoint 为中心、标准差 0.2 的高斯分布采样
- Translation speed 均匀采样自 $[0.1, 0.2] \text{ m·s}^{-1}$

**Rotation 部分**：
- Starting orientation $R \in \text{SO}(3)$ 均匀采样
- 物体沿 Bezier curve 运动时绕 random rotation axis 旋转，angular speed 均匀采样自 $[0.5, 1] \text{ rad·s}^{-1}$

**Hand pose 生成**：
$$\varsigma = \xi \circ T_{\text{object}}^{\text{hand}}$$

其中 $\xi = (\mathcal{T}_0, \mathcal{T}_1, \ldots, \mathcal{T}_{T-1})$ 是 object trajectory，$\mathcal{T}_t \in \text{SE}(3)$ 是 $t$-th frame 的 object pose in world frame；$T_{\text{object}}^{\text{hand}}$ 是 hand pose in object reference frame（固定 relative transform，因为 hand 抓住 object）。

**为什么 Bezier curves？** Bezier curves 由 control points 决定的 smooth curves，多个 Bezier curves 链接可生成 seamless 的 complex trajectory。这模仿了 real-world 中 human 犹豫、改变方向等行为，超出 HandoverSim 中 simple "give then receive" 的设定。

### 3.3 Reactive Handover（GenH2R-Sim 独有改进）

HandoverSim 只是 replay 每一帧，不响应 robot action。GenH2R-Sim 引入 reactive 机制：
- 设 $p \in \mathbb{R}^3$ 为 current gripper tip position
- $Q \subset \mathbb{R}^3$ 为 current object point cloud
- 当 $\min_{q \in Q} \|p - q\| \leq 0.1$ 时，human hand 停止移动等待 robot 抓取

这把 task 从 "chase-and-grasp game" 变成更 authentic 的 cooperative interaction。这是 sim2real 关键的 detail，prior work 缺这个就难以表达 human 的协作意图。

---

## 4. Demonstration Generation（关键技术贡献）

### 4.1 核心问题：Distillability

这是 paper 中最 subtle 也最重要的 insight。生成 demonstration 用于 closed-loop visuo-motor policy 的 distillation 时，**仅有 success 是不够的**，必须有 vision-action correlation。

为什么？考虑两个 failure mode：

**Failure Mode 1 — Foresighted Planner (Destination Planning)**
直接基于 human handover destination end state 规划一条 smooth 短路径。
- 问题：planned path 与 dynamic visual observations 不 align
- Distillation 时需要从 vision 准确预测 human trajectory end state，这在 complex handover 中极其困难
- 实验数据：在 t0 上 train 时 success rate 只有 0.93%（基本失败）

**Failure Mode 2 — Shortsighted Planner (Dense Planning)**
每个 time step 独立用 OMG planner [41] 基于 privileged state 重新规划 grasp 与 motion。
- 问题：robot morphology constraints + multi-solution nature of planners 导致 smooth visual observations 对应 unsmooth、multi-modal 的 robot trajectories
- 实验数据：success rate 可保持但 zigzag、慢、不自然

**Insight**：distillability 要同时考虑 robot morphology 和 dynamic vision，使 sequential smooth visual observations 对应 smooth grasp/motion plans。

### 4.2 Landmark Planning（Proposed Method）

**Step 1: Improved Shortsighted Planner**

基于 OMG planner [41]（参考 https://sites.google.com/view/omg-planner），输入 object 6D pose + candidate grasps + human hand poses（过滤 invalid grasps）。改进三点：
1. **Grasp sorting**：按 grasp pose 到 robot end effector 的距离排序，从最近的 grasp 开始尝试 IK，直到成功。这让 object 保持 in wrist camera 视野，减少 visually irrelevant actions
2. **IK initialization**：用上一 time step 的 robot arm pose 初始化 IK，提升 trajectory smoothness
3. **Conditional OMG invocation**：只在 IK 成功时调用 OMG planner

**Step 2: Trajectory Resampling**

OMG planner 生成固定 N 步 trajectory $C_0, C_1, \ldots, C_N$。问题：step size 依赖初始 end effector 与 target grasping pose 距离，会让 vision-based model 困惑（它不知道初始 end effector pose，无法 infer expert speed）。

Resampling 公式：
$$s_i = \sum_{j=1}^{i} \|C_j - C_{j-1}\|$$

其中 $s_i$ 是 accumulated step length。设 $L$ 为 desired step length，resampled trajectory $C'_0, C'_1, \ldots, C'_M$ 满足 $(M-1)L < s_N \leq ML$。对每个 $1 \leq i \leq M-1$，若 $s_j \leq iL \leq s_{j+1}$，则：

$$C'_i = \frac{s_{j+1} - iL}{s_{j+1} - s_j} C_j + \frac{iL - s_j}{s_{j+1} - s_j} C_{j+1} \quad (3)$$

变量含义：
- $C_j, C_{j+1}$：original trajectory 中相邻两个 joint configurations（7D，对应 7-DoF arm）
- $s_j, s_{j+1}$：对应 accumulated step length
- $iL$：resampled trajectory 上第 $i$ 步对应的 desired accumulated length
- $C'_i$：resampled 后的第 $i$ 个 joint configuration（线性插值）

这样无论初始距离多远，trajectory 都保持 consistent step length，对 vision-based model distillation 友好。

**Step 3: Landmark-based Replanning**

完整定义：
1. 设 object trajectory $\xi = (\mathcal{T}_0, \mathcal{T}_1, \ldots, \mathcal{T}_{T-1})$
2. 训练 object pose forecasting network：输入过去 + 当前 poses $(\mathcal{T}_0, \ldots, \mathcal{T}_t)$，预测未来 N 步 $(\mathcal{T}_{t+1}, \ldots, \mathcal{T}_{t+N})$
3. 对每个 time step 计算 forecasting error，用 threshold 找到 endpoints（past observations 无法很好预测 future 的点）
4. 用 endpoints 把轨迹分为 segments：$0 = l_0 < l_1 < \cdots < l_k = T$
   - $l_i$：第 $i$ 个 segment 的 endpoint index
   - $k$：segments 总数
5. 设 $P \in \mathbb{N}$ 为 replanning period hyperparameter
6. 对每个 planning frame $t = 0, P, 2P, \ldots$，假设 next endpoint 为 $l_{i+1}$（即 $l_i \leq t < l_{i+1}$），plan based on object pose at frame:
   $$\hat{t} = \min(t + P, l_{i+1})$$

变量含义：
- $t$：current planning frame index
- $P$：replanning period（控制稀疏度）
- $l_{i+1}$：current segment 的 endpoint（trajectory 转折点）
- $\hat{t}$：landmark frame index，从 $t$ 看 $\hat{t}$ 步之后的状态来 plan

**关键 intuition**：$\hat{t}$ 不超过 $l_{i+1}$，避免绕过 trajectory 转折点（那里 human motion 变 unpredictable）。Planning 基于未来 state 但只考虑 visually foreseeable futures。$P \to 1$ 时退化为 dense planning；$P \to \infty$ 时退化为 destination planning。Landmark planning 是这两极端的 sweet spot。

### 4.3 Ablation 直觉

paper 附录 Figure 6 显示 ablation：
- Dense Planning with Foreseeing：OMG-planner 每步 replan，但基于未来 object state
- Sparse Planning：OMG-planner 用 landmark planning 的稀疏度，但只基于当前 state

两条都 improve，说明**降低 replanning 频率**和**foreseeing future states** 都贡献正向效果，landmark planning 同时拿到两个好处。

---

## 5. Forecast-Aided 4D Imitation Learning

### 5.1 4D 表示与 Flow Feature

输入是 egocentric point cloud。第 $t$-th frame 中，$M_t^i \in \text{SE}(3)$ 是 current frame 与 $i$-th frame 之间的 relative object pose（egocentric view）。

直接 frame stacking 难以同时 capture motion 和 geometry。改用 ICP [37]（https://www.cs.princeton.edu/~smr/papers/icp.pdf）算法计算 transformation matrices：

$$\{\hat{M}_t^{t-1}, \hat{M}_t^{t-2}, \ldots, \hat{M}_t^{t-L_1}\}$$

变量含义：
- $\hat{M}_t^{t-j}$：estimated transformation from current frame $t$ 到 past frame $t-j$（$j = 1, \ldots, L_1$）
- $L_1$：past frames 的数量（paper 用 3）
- Hat 表示 estimated（ICP 估计可能 slightly imprecise，因为 partial point cloud）

把这些 transformation 应用到 current frame 的每个 point，得到它在 past frames 的 rough coordinates。这样每个 point 的 feature vector 长度 = $3 + 2 + 3 \cdot n$：
- 3：current 3D coordinates
- 2：one-hot hand/object label
- $3 \cdot n$：past $n$ frames 的 3D coordinates（来自 flow）

### 5.2 Network Architecture

Figure 5 给出 pipeline：
1. **Input**：egocentric point cloud（augmented with flow features）
2. **Encoder**：PointNet++ [35]（参考 https://arxiv.org/abs/1706.02413）encode 为 low-dim global feature
3. **Policy head**（MLP）→ 6D egocentric action
4. **Prediction head**（MLP）→ future pose transformations

为什么不用更复杂的 4D backbones 如 P4Transformer [48] 或 NSM4D [14]？Paper 明确说：这些 backbone "often not suitable for robotic tasks that require a fast reference speed"。GenH2R 在 effectiveness 和 simplicity 间取平衡。

### 5.3 Loss Function

**Action loss** $\mathcal{L}_{\text{action}}$：L1 loss on 3D points on robot gripper（来自 DeepIM [27]，https://arxiv.org/abs/1808.01285）。即在 gripper 表面 sample 3D points，比较 predicted action 应用后的 points 和 ground truth points。

**Prediction loss**：
$$\mathcal{L}_{\text{pred}} = \sum_{i=t+1}^{t+L_2} \|\hat{M}_t^i - M_t^i\| \quad (1)$$

变量含义：
- $i$：future frame index，从 $t+1$ 到 $t+L_2$
- $L_2$：预测 future frames 数量（paper 用 3）
- $\hat{M}_t^i$：predicted transformation from current frame $t$ 到 future frame $i$
- $M_t^i$：ground truth transformation
- $\|\cdot\|$：transformation matrix 的某种 norm（通常 Frobenius 或分解为 translation + rotation 的 L2）

**Total loss**：
$$\mathcal{L} = \mathcal{L}_{\text{action}} + \lambda \mathcal{L}_{\text{pred}}$$

$\lambda$ 是 weighting hyperparameter（paper 用 0.1）。

**Intuition**：因为 demonstration 是基于 future landmarks 生成的，预测 future object pose 能直接 exploit vision-action correlation。Policy 学到的 representation 因此同时编码 "object 在哪" 和 "object 要去哪"，使得 action 既反映当前 scene state 又反映 future motion。

### 5.4 Grasping Heuristic

Policy 只输出 6D egocentric action（closed-loop）。是否抓取和放置 target location 用 heuristic（类似 GA-DDPG [43]）：
- 若 gripper 附近点数 > threshold → 抓取，然后 open-loop retract 到 target location
- 不执行 policy network 的 egocentric action

这是个 hybrid design：closed-loop approach + open-loop grasp-retract。

---

## 6. 实验 Setup 详细解析

### 6.1 Datasets

| Benchmark | Source | Train scenes | Test scenes | Objects | Duration |
|-----------|--------|--------------|-------------|---------|----------|
| s0 (Sequential) | HandoverSim [9] / DexYCB [8] | 720 | 144 | 20 | 3s |
| s0 (Simultaneous) | 同上 | 720 | 144 | 20 | 3s |
| t0 | GenH2R-Sim | 1,000,000 | 3,260 | 3,266 | 8s |
| t1 | HOI4D [28] 提取 | 0 | 1,000 | - | - |

**Sequential vs Simultaneous**：
- Sequential：robot 等 human 到 handover location 才开始动
- Simultaneous：robot 从 episode 开始就动

**关键 detail**（Section B.1）：原 Handover-Sim2real 在 simultaneous 时设 `TIME_WAIT=1.5s`（即让 robot 等 1.5s）。作者认为真正 simultaneous 应该 `TIME_WAIT=0s`，因此 reproduce 了 baseline 在 true simultaneous 下的结果（带 * 的行）。

### 6.2 Metrics

**Success Rate (S)**：成功 grasping + 移到指定位置；failure 为 hand contact、object drop、timeout ($T_{\max} = 13s$)

**Time (T)**：execution time

**Average Success (AS)**（paper 创新 metric）：
$$\text{AS} = \int_0^1 \text{Success}(t) \, dt \quad (2)$$

变量含义：
- $t$：归一化时间比例，$t \in [0, 1]$
- $\text{Success}(t)$：只考虑 $t \cdot T_{\max}$ 内完成的成功 cases 的 success rate
- AS 是 success-time 曲线下面积

**Intuition**：类似 AP（Average Precision），同时衡量 success rate 和 completion efficiency。有的 policy 高 success 但慢（浪费 human 时间），有的快但低 success，AS 平衡两者。

---

## 7. 实验结果深度解读

### 7.1 Main Results（Table 1 / Table 4）

| Train | Method | s0 Seq S | s0 Sim S | t0 S | t1 S |
|-------|--------|----------|----------|------|------|
| s0 | Handover-Sim2real* [11] | 64.35 | 25.69 | 28.56 | 30.60 |
| s0 | Destination Planning | 74.31 | 76.16 | 25.68 | 48.40 |
| s0 | Dense Planning | 74.77 | 75.45 | 27.30 | 52.30 |
| s0 | Landmark Planning | 77.78 | 79.17 | 29.63 | 54.20 |
| t0 | Handover-Sim2real [11] | 65.97 | 62.50 | 33.71 | 47.10 |
| t0 | Handover-Sim2real* [11] | 63.55 | 38.89 | 33.31 | 33.35 |
| t0 | Destination Planning | 0.93 | 6.48 | 5.96 | 1.60 |
| t0 | Dense Planning | 81.48 | 84.95 | 38.04 | 57.90 |
| t0 | Landmark Planning | **86.57** | **85.65** | **41.43** | **68.33** |

**关键观察**：
1. **Scale 效应**：t0 training > s0 training，所有 method 都受益于大规模 synthetic data。Landmark planning 从 s0 → t0 提升 +8.79%, +6.48%, +11.80%, +14.13%
2. **vs SOTA**：Landmark planning on t0 vs Handover-Sim2real* on s0：+11.34%, +16.90%, +12.26%, +15.93% improvement
3. **Destination planning 在 t0 上崩溃**：success rate 从 s0 的 ~75% 跌到 t0 的 ~1%。原因：t0 trajectories 复杂，直接 plan 到 destination 完全破坏 vision-action correlation
4. **Landmark > Dense**：landmark planning 在所有 t0 setting 上 +5.09% (s0 Seq), +0.70% (s0 Sim), +3.39% (t0), +10.43% (t1)，且 time 更短

### 7.2 Ablation Study（Table 2 / Table 5）

| Method | S | T | AS |
|--------|---|---|----|
| w/o Flow | 31.66 | 5.67 | 17.9 |
| w/o Prediction | 39.18 | 6.11 | 20.7 |
| w/o Flow & Prediction | 37.04 | 5.93 | 20.1 |
| w/o Endpoints | 39.73 | 5.90 | 21.7 |
| Frame Stacking | 35.17 | 5.82 | 19.4 |
| Ours | 41.43 | 6.01 | 22.3 |

**关键观察**：
- Flow 信息贡献 +9.77%（Ours vs w/o Flow），证明 4D 表示比 3D 表示有效
- Prediction loss 贡献 +2.25%（Ours vs w/o Prediction），证明 forecasting objective 有助 exploit vision-action correlation
- Frame stacking 比 flow 信息差 5.26%，验证 ICP-based flow 的优越性
- Endpoints 在 landmark selection 中重要（-1.70% without）

### 7.3 Dataset Scale Ablation

10% data utilization → t1 上 -5.93% success rate。验证 dataset scale 对 generalization 的关键作用。

### 7.4 Real-World Experiments

**Setup**：ROKAE xMate3 ER robot（类似 Franka Panda）+ 2 个 Intel RealSense D435（一高视野近、一低视野远，merge 得到 comprehensive view）

**User Study 1**（Table 6）：
| Setting | GA-DDPG | Handover-Sim2real | Ours |
|---------|---------|-------------------|------|
| Simple | 18/30 (60%) | 17/30 (57%) | 27/30 (90%) |
| Complex | 13/30 (43%) | 10/30 (33%) | 21/30 (70%) |

**User Study 2**（Table 7，15 个 novel objects）：
- GA-DDPG: 43.3%
- Handover-Sim2real: 28.9%
- Ours: 72.2%

**关键 observation**：sim-to-real gap 在 simple setting 上较小，complex setting 上 Ours 仍保持 70%，远超 baseline 的 33%。这说明 landmark planning + 4D imitation learning 的组合让 policy 学到了更 generalizable 的 representation。

**Limitation**：透明物体（transparent bottle, transparent beaker）因 corrupted depth 仍失败较多。

---

## 8. 与 Related Work 的深度对比

### 8.1 vs HandoverSim [9] / Handover-Sim2real [11]
- HandoverSim：mocap-based，1000 scenes，20 objects
- Handover-Sim2real：two-stage teacher-student training，3s clipped motion
- GenH2R：synthetic-based，1M scenes，3266 objects，8s reactive motion，ICP flow + prediction

### 8.2 vs SynH2R [10]
SynH2R 也用 synthetic data，但 progress 有限。GenH2R 在 scenes 数量和 complexity 上更进一步。

### 8.3 vs TAMP methods [13, 22, 31]
TAMP 通常 focus on fairly static scenes，without active motion or object/task variety。GenH2R extend 到 dynamic H2R handover，并考虑 vision-action correlation（distillability）。

### 8.4 vs BC methods [3, 20, 29, 53]
BC 是 supervised learning 直接 imitate expert。GenH2R adopt BC 但 augment with forecasting objective 和 4D 表示。

### 8.5 vs RL methods [11, 43]
RL 需要大量训练且在不同 scenarios 上 unstable。GenH2R imitation learning 只需 8 hours 训练（1×RTX 3090，80,000 iterations，batch size 256，Adam lr=0.001，weight decay=0.0001）。

---

## 9. 我（Karpathy 视角）的 Intuition 总结

让我总结几个我认为最关键的 intuition：

**Intuition 1: Distillability 是 demonstration quality 的隐藏维度**
通常我们衡量 demonstration quality 只看 success rate。但 closed-loop visuo-motor policy distillation 时，vision-action correlation 是 hidden dimension。一条成功但 vision-irrelevant 的 demonstration 反而有害，因为它让 policy 学到 "不看也能做"，破坏 generalization。Landmark planning 本质是构造 vision-action aligned 的 demonstration。

**Intuition 2: Scaling 的 compound effect**
- Scale scenes → diverse geometry + motion
- Scale demonstrations → diverse vision-action pairs
- 4D + forecast → exploit temporal structure
三者 compound，每一步都依赖前一步。这和 LLM scaling 的 "data → model → capability" chain 同构。

**Intuition 3: ICP-based flow 是 cheap but effective 的 4D 表示**
对 robot 学习，reference speed 重要。复杂 4D backbone（P4Transformer, NSM4D）效果好但慢。ICP + PointNet++ 是 sweet spot：ICP 提供 motion 信息（cheap），PointNet++ 提供 geometry encoding（fast）。

**Intuition 4: Forecasting objective 是 auxiliary supervision，不是 multi-task**
$\mathcal{L}_{\text{pred}}$ 不直接产生 action，但通过 shared encoder 间接 improve policy。这和 self-supervised learning 在 vision 中的作用类似——通过预测 future 学习 useful representation。

**Intuition 5: Sim-to-real 不只靠 domain randomization，更靠 behavior diversity**
GenH2R 的 sim-to-real 不靠 heavy domain randomization，而是靠 behavior diversity（Bezier curves 模拟 human hesitation、change of direction）。这种 "behavior coverage" 比 "appearance coverage" 对 handover task 更重要。

---

## 10. Limitations 与 Future Directions

Paper 在 Section D 列出三方面 limitation：

1. **Robot morphology**：当前 7-DoF arm 活动区域受限。未来可扩展到 movable base robot（如 mobile manipulator），扩大 spatial 范围
2. **Human modeling**：只考虑 hand + object pose，未考虑整个 human body。Real-world robot 需要考虑 body motion 做 dynamic interaction
3. **Human intention**：GenH2R-Sim 仅在 robot 靠近时 human 停下，未实现更复杂 human behavior（如感知 danger 收回 hand）

我额外想到的 future direction：
- **LLM-conditioned handover**：用 LLM infer human intent（"递给我那个杯子"），让 policy condition on language
- **Force-feedback closed-loop**：当前只用 vision，加 force/torque sensing 可改善 grasp 稳定性
- **Diffusion policy** [https://diffusion-policy.cs.columbia.edu/]：替换 BC with diffusion，可能更好处理 multi-modal action distribution
- **3D Gaussian Splatting sim**：用 3DGS 替代 mesh-based rendering，可能缩小 sim-to-real gap
- **VLM-based demonstration filtering**：用 VLM 判断 demonstration 的 naturalness，filter 掉不 human-like 的

---

## 11. 关键 References

1. GenH2R 项目主页：https://GenH2R.github.io
2. HandoverSim: https://github.com/NVlabs/handover_sim
3. Handover-Sim2real: https://github.com/seroyal19/handover-sim2real
4. DexYCB: https://dex-ycb.github.io/
5. HOI4D: https://hoi4d.github.io/
6. DexGraspNet: https://github.com/PKU-EPIC/DexGraspNet
7. ShapeNet: https://shapenet.org/
8. PointNet++: https://arxiv.org/abs/1706.02413
9. OMG-Planner: https://sites.google.com/view/omg-planner
10. DeepIM: https://arxiv.org/abs/1808.01285
11. ICP: https://www.cs.princeton.edu/~smr/papers/icp.pdf
12. GA-DDPG: https://arxiv.org/abs/2106.02447
13. SynH2R: https://arxiv.org/abs/2311.05599
14. Diffusion Policy: https://diffusion-policy.cs.columbia.edu/

---

## 12. 一句话总结

GenH2R 通过 procedural synthesis 把 H2R handover 的 simulation、demonstration、policy learning 三件事各 scale up 三个数量级，并用 landmark planning + 4D imitation learning with forecasting 的设计确保 scaled data 真正 distillable，最终在 sim 和 real 都拿到 +10% 以上 success rate 的提升。这个工作的核心 contribution 不仅是 "scale works"，而是揭示 "scale 时如何保证 distillability" 这一隐藏维度——这个 insight 对所有 robot learning from demonstration 的工作都有启发意义。
