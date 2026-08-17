---
source_pdf: Planning-Guided Diffusion Policy Learning for.pdf
paper_sha256: 98648054505b093446d6aa9876a1f480e273981f581d1f8a0491a8f5c3431e1f
processed_at: '2026-08-06T04:49:38-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# GLIDE 的"人话版"

## 一句话总结

**机器人学双手推大物体太贵了，用 planner 在仿真里批量造数据，让 diffusion policy 去模仿，再加几个小 trick 就能真机部署，还能泛化到没见过的物体。**

---

## 问题到底难在哪

想象你让两个机械臂去转一个大纸箱，让它从某个姿态转到另一个姿态。

你不能用 gripper 夹起来（太重太大），只能用机械臂的"胳膊肘"去顶、去挤、去推。这就叫 **contact-rich manipulation**。

难点有三个：

1. **接触点会变**。夹住的时候在 A 点，推到一半可能要换到 B 点，中间有个松手再抓的过程，动力学是断续的（hybrid），不是 smooth 的连续运动。

2. **大角度旋转要分多次**。一只胳膊能转的范围受 joint limit 限制，转 150 度可能要"顶一次、松开、换个姿势、再顶一次"，这是 multi-phase 的长 horizon 任务。

3. **真实数据太贵**。Teleop 采数据，一个 episode 要精细协调两只 7-DoF 手臂，几小时可能才几十条 demo，根本不够训 policy。

---

## GLIDE 的思路

既然真实数据贵，那就**让 planner 在仿真里自己生成**。

但传统 contact-rich planner 有两个问题：慢（分钟级解一条 trajectory），且要完整物体信息（real 部署时不现实）。

GLIDE 的 insight 是：**planner 慢没关系，我离线用它造 12000 条 demo，然后训一个 learned policy 替代它做实时推理。** 这是典型的 "用昂贵 teacher 蒸馏出廉价 student"。

### Planner 怎么变快

传统 contact-implicit trajectory optimization 要把整条 trajectory 的所有 contact mode 一起优化，变量数随 horizon 指数爆炸。

GLIDE 偷了个懒：**不做 long-horizon 优化，只做 single-step greedy 优化**。

每一步只问一个问题："我现在这个 configuration，下一步怎么动能让物体最靠近目标？" 这是一个 QP，毫秒级解出。

然后不断 greedy 推，推到 joint limit 就重新 sample 一个 grasp 再来。Greedy 牺牲了 global optimality，但实验发现 success rate 几乎不掉，速度提升一个量级。

这样 96 核 CPU 跑两天，12000 条 demo 就出来了。

### 但 planner 产出不全是好数据

Planner 用了 linearized contact dynamics（近似有误差），RRT 有随机性，有些 trajectory 在真实物理下根本走不到目标。

所以加了一步 **filtered behavior cloning**：用 Drake 高保真物理引擎 rollout 每条 trajectory，走不到目标的、走太慢的，全扔掉。剩下 12000 条干净的 success demo 才用来训 policy。

这个 filter 步骤很重要——**你给 policy 什么数据，它就学什么分布。喂垃圾进去，出来的也是垃圾。**

---

## Policy 怎么设计

基于 DP3（3D Diffusion Policy），用 point cloud 作为输入，diffusion model 在 action space 上采样。

但直接拿 DP3 用不行，有几个坑。GLIDE 做了四个关键改动：

### 1. Task-conditioning 而不是 per-task 训练

DP3 原版每个 task 一组参数，学"转 45 度"就只会转 45 度。

GLIDE 让 policy 接收 task specification $c_t$（当前 pose 到 target pose 的 delta transformation）作为额外输入，训一个 universal policy。这样能泛化到任意目标 pose，不用重新训练。

### 2. 用 keypoint tracking 隐式表示 object pose

不假设知道物体形状，没法定义 canonical frame。

做法：第一帧用 Grounding DINO 分割物体，Farthest Point Sampling 选几个 keypoint，CoTracker 实时跟踪。用 keypoint 的位移反推 delta transformation。

**核心 insight：OOD 物体上，你不需要知道它是什么，只要能 track 它的几个点就行。**

### 3. Residual action（sim-to-real 的关键 trick）

原版预测 absolute joint angle $q_{t+1}$。问题是不同 episode 起点不同，"推 45 度"在不同起点对应完全不同的 absolute target，分布太散。

GLIDE 改成 residual：$a_t = q_{t+1} - q_t$，预测"从当前 joint position 往哪个方向推多少"。

这个量在 object-centric 视角下大致 invariant，scale 和 shift 跨 episode 一致，diffusion model 学起来轻松。

**Ablation 显示：sim 里几乎无影响，real 里从 0.52 提升到 0.80，+28 个点。这是整个 sim-to-real gap 的解药。**

### 4. Flying point augmentation

RealSense D455 的 depth 会有 outlier noise——边缘飞点、玻璃反射错误、hole-filling artifact。Sim 里的 point cloud 是干净的，policy 没见过这种噪声，部署就崩。

解决：训练时每个 point 以 0.5% 概率被加一个大 Gaussian 噪声"飞走"。

**Ablation：real 里从 0.80 掉到 0.32，掉 48 个点。这个 aug 看起来不起眼，实际是 real deployment 的救命稻草。**

---

## 实验结果说人话

### In-distribution（训练分布内的箱子）

固定转 45 度：**sim 0.74，real 0.80**。

注意 real 比 sim 还高一点，而且 policy 比 planner 自己（0.337）高得多。这是因为：

1. Filter 把 planner 失败的 trajectory 全扔了，policy 学的全是成功轨迹
2. Policy 有 visual feedback 做 closed-loop correction，planner 是 open-loop
3. Policy 学到了 multimodal average，比 greedy planner 选的单一 mode 更 robust

**Student 超过 Teacher，这是 imitation learning 的经典现象：filtering + smoothing + closed-loop。**

### OOD（没见过的物体）

软的橡胶盒、布袋、上窄下宽的容器、装满杂物的 container：

- 固定转 45 度：**0.66**
- 随机转：**0.28**

装满东西的容器和空容器表现差不多，说明 policy 主要从 geometry 推断 action，重量变化被机械臂的 torque margin 吸收了。

充气玩具（legged base、高重心、容易翻）：固定 0.52，随机 0.28。

### Ablation 干净到震撼

| 改动 | Sim | Real |
|---|---|---|
| 全开 | 0.74 | **0.80** |
| 去掉 residual action | 0.75 | 0.52 |
| 去掉 flying point | 0.78 | 0.32 |
| 都去掉 | 0.75 | **0.00** |

两个 trick 在 sim 里几乎看不出来（sim 数据干净），但在 real 里是生与死的差别。这是 sim-to-real gap 最干净的证据：**你以为没用的 trick，往往恰恰是 deployment 的关键。**

---

## Failure mode 分析（诚实的部分）

Random-Hard（转 90-150 度）失败里：

- **52%：卡在 bad joint configuration**，推不动了，需要重新 approach 但 policy 不知道怎么脱困
- 20%：物体滑了（没触觉反馈，contact 不稳定不知道）
- 16%：挤太用力 torque 超限

这给未来方向指了路：在 near-failure state 也生成 demo，让 policy 学 recovery。

---

## 我的 take

这篇 paper 最值得记的 insight：

1. **Data bottleneck 是 contact-rich manipulation 的真正瓶颈，不是 model 结构。** 用 planner 造数据是 scalable path。

2. **Greedy single-step 替代 long-horizon TO**，牺牲 optimality 换 scalability，再用 physics rollout filter 弥补质量。这个 trade-off 在 data gen 场景下 almost always 值得。

3. **Residual / relative quantity 是 sim-to-real 的通用 invariance。** 凡是能从 absolute 改成 relative 的地方，都该改。分布 shift 在 relative 量上天然小。

4. **Long-tail noise injection 比 jittering 更对路。** Real sensor 的 noise 不是高斯抖动，是 outlier 飞点，data aug 要 mimic 真实的 noise 分布。

5. **Student 超过 Teacher 不奇怪。** Filtered + smoothed + closed-loop 的 learned policy 在 stochastic 环境下比 greedy planner 更 robust，这是 imitation learning 的威力。

6. **OOD 泛化来自 geometry-only observation。** 不依赖 object shape prior、不依赖 canonical frame、只用 raw point cloud + tracked keypoints，让 policy 学到 generic 的 "geometry → contact action" mapping。

7. **诚实的 failure analysis 比吹牛的 success rate 有价值得多。** 52% 卡 joint limit 说明什么 demo 缺失，20% slippage 说明什么 sensing 缺失，16% torque exceed 说明什么 awareness 缺失。这是 roadmap，不是 disclaimer。

---

## 与你 (Karpathy) 的连接点

你在 nanoGPT / "State of GPT" 系列里反复强调的 "next-token prediction captures world model" 思路，和 GLIDE 有结构上的相似：

- **Planner 是 expensive teacher**，像 GPT-4 生成 high-quality 数据
- **Diffusion policy 是 cheap student**，像 SLM 蒸馏后做实时推理
- **Filter step 是 quality control**，像 RLHF 把 low-quality 产出砍掉
- **Residual action 是 invariance engineering**，像你强调的 "data normalization 决定模型能不能学"
- **Flying point aug 是 distribution matching**，像训练数据要匹配 deployment distribution

本质都是同一件事：**把 expensive, privileged, open-loop 的 capability，distill 成 cheap, sensor-based, closed-loop 的 skill。** GLIDE 用 planner + diffusion 做了 robotics 版本。

---

# GLIDE: Planning-Guided Diffusion Policy for Contact-Rich Bimanual Manipulation 详解

## Big Picture (一句话直觉)

GLIDE 的核心 insight 在于: **contact-rich bimanual manipulation 的瓶颈不在 policy 结构, 而在 data**。 真实世界 teleop 采集成本极高, 于是用 efficient contact-implicit trajectory optimization 在 Drake simulator 里 mass generate demonstration, 再用一个 task-conditioned point-cloud diffusion policy 做 behavior cloning, 配合若干 sim-to-real 关键 design choices (flying point aug, residual action, large prediction horizon) 实现泛化到 OOD 物体 (软的、有曲线的、装满东西的容器等)。

Project page: https://glide-manip.github.io/

---

## 1. 为什么这个 Task 难 (build intuition)

Contact-rich bimanual manipulation 与普通 manipulation 的本质区别:

**Single-arm grasp-and-place**: 接触点固定 (gripper 夹住), 动力学 smooth, MPC / RTMP 几乎可解。

**Contact-rich bimanual**: 两只 7-DoF arm, 没有 end-effector, 用 distal link 顶 / 挤 / 推 / pivot 物体。 接触点会随 configuration 切换 (mode switching), 形成 non-smooth hybrid dynamics。 一次 reorientation 可能需要多个 "approach → grasp via contact → manipulate → release → re-approach" 的 phase。 物体 bulky/heavy (无法用 gripper 直接 pick up), 一次 contact phase 内能转的角度受 joint limit 限制, 大角度旋转 (Hard, |Δθ| > 90°) 必须分多轮, 这就要求 policy 在长 horizon 下保持 stable contact 同时规划下一组 contact mode。

传统 MIP formulation 把每个 contact pair / time step 的 mode 作为 integer variable, contact pair 数 × horizon 指数爆炸。 最近 smoothing-based 方法 (Posa, Pang, Suh, Manchester 等) 用 smoothed contact 把 hybrid dynamics 变成可微的, 才让 planner 在 ~分钟级解出一条 trajectory, 但 still 需要完整 object geometry + state, 不能 online 跑。 这就是为什么需要一个 learned policy 替代 planner 做实时推理。

---

## 2. Method Walkthrough

### 2.1 Problem Formulation

State space $\mathcal{S}$, observation space $\mathcal{O}$ (point cloud + proprioception), action space $\mathcal{A}$ (joint commands), task space $\mathcal{C}$ (target SE(2) transformation)。

Policy: $\pi_\theta(a \mid o, c)$, condition 在 observation $o_t$ 与 task spec $c_t$ 上。

关键 design 决策: **不假设 object shape 已知**。 物体 pose 通过 keypoint tracking 从 visual observation 隐式得到, $c_t$ 是 current pose 到 target pose 的 delta transformation。 这就避免了在 OOD 物体上需要 "object frame" 的尴尬—— 你没法定义一个 soda bottle 的 canonical frame, 但可以 track 它的几个 keypoints 算 delta。

Task 是 SE(2) reorientation (tabletop 平面内旋转 + 平移), 不做 SE(3) — 这是一个有意的简化, 让 planning 可行, 同时涵盖绝大多数 tabletop 场景。

### 2.2 Demonstration Synthesis (Algorithm 1)

Planner pipeline 三层嵌套:

```
while object not at goal:
    q_grasp = SAMPLECONTACT(q^u)           # IK-based, distal link 夹住
    while robot not at grasp config:
        PLANCOLLISIONFREE (BiRRT + shortcut)
    while robot not at joint limit AND object not at goal:
        PLANCONTACT (single-step trajectory opt)
```

核心是 **greedy single-step contact optimization**, 不是 long-horizon trajectory opt。 长 horizon 的 contact-implicit TO (Tassa iLQG, Posa direct collocation, Le Cleac'h Fast CIMP) 计算成本太高, 不适合 mass data gen (12k trajectories × 多 random reset)。

#### Contact Planner 数学细节

Linear approximation of local contact dynamics:

$$q_+^u = f_{\text{local}}(q^u, q^a, a)$$

其中:
- $q^u \in \text{SE}(2)$: object 当前 pose
- $q^a \in \mathbb{R}^{14}$: 双臂 joint angles (7 per arm)
- $a \in \mathbb{R}^{14}$: commanded joint angles (action)
- $q_+^u$: 执行 action 后的 approximate next object pose
- $f_{\text{local}}$: 把 contact force 在当前 configuration 做 linearization 得到的 forward model (本质是把 quasi-dynamic contact 写成 linear complementarity 的 smoothed 版本, 见 Suh et al. "Bundled Gradients through Contact via Randomized Smoothing" RA-L 2022 和 Suh/Pang/Zhao/Tedrake 2025 preprint on contact trust region)

Single-step objective:

$$\min_{q_+^u, a} \; (q_+^u - q_{\text{goal}}^u)^T \mathbf{Q} (q_+^u - q_{\text{goal}}^u) + (a - q^a)^T \mathbf{R} (a - q^a)$$

变量解释:
- $\mathbf{Q}$: object pose 误差 cost matrix, 衡量 "推到目标" 的优先级, 通常 $Q \succ 0$ 对 $x, y, \theta$ 加权
- $\mathbf{R}$: action regularization, 鼓励 action 接近 current configuration $q^a$, 让 motion smooth + 避免极端 joint motion
- 第一项: 把物体推向 goal
- 第二项: action smoothness / minimum-motion penalty

这是 LQR-style 的 quadratic cost, 配合 linearized dynamics, 整体是 QP, 可以 ms 级解。 Greedy 体现在 "尽可能多推一把", 直到 joint limit 触发外层 loop 重新 sample grasp。

**关键 trick**: 把 long-horizon trajectory opt 拆成多次 single-step + re-grasp。 这牺牲了 global optimality, 换来大幅 speed-up, 而 trajectory quality 几乎不损失 (论文 experiment 说 greedy 几乎不影响 success rate)。 这是整个 data gen pipeline 能 scale 到 12k trajectories / 2 days / 96 CPU 的根本原因。

#### Filtered Behavior Cloning

Planner 出来的 trajectory 不全是好 demo: contact dynamics linearization 有误差, RRT 有 stochasticity, 部分 trajectory 在高保真 Drake physics 下 rollout 失败或 suboptimal (走太久)。 Filter step:
1. 用 Drake physics simulator rollout 每条 planned trajectory
2. 丢弃最终 pose 未达 threshold (10 cm, 0.2 rad) 的
3. 丢弃 suboptimal (耗时过长) 的
4. Rebalance 让 object 分布 uniform, render 不带 color 的 point cloud (只保留 geometry, 因为 real-world 用的是 depth-only D455)

最终 12,000 条 trajectories 用于 training。

### 2.3 Diffusion Policy Architecture

基于 DP3 (Ze et al. RSS 2024, https://3d-diffusion-policy.github.io/) 的 point cloud diffusion policy, 但加了几个关键改动。

DP3 本身: 用 pointnet++ style encoder 提 point cloud feature, 把它作为 condition 喂给 DDPM, 在 action space 上做 denoising diffusion 采样 action chunk。

GLIDE 的改动:

**(1) Task-conditioning (而非 multi-head)**:
原 DP3 / Diffusion Policy 每个 task 一组 parameters。 GLIDE 训 single policy, 额外把 task spec $c_t$ 喂进去, 实现任意 target pose。 这就避免了 "学 1000 个固定动作" 的死板, 让 policy 学到 "delta transformation → action" 的 mapping, 从而 zero-shot 泛化到新目标。

**(2) Task representation via keypoint tracking**:
不假设 object shape / canonical frame 已知。 程序:
1. 初始 frame $o_0$ 上用 Grounded DINO + EfficientViT (open-vocab segmentation) 把 target object 抠出
2. Farthest Point Sampling 选 keypoints (覆盖 object 几何)
3. CoTracker / TAPIR 在 3D space 实时 track 这些 keypoints
4. $c_t$ = 当前 keypoints 到 initial keypoints 的 transformation, 反推到 target pose

这样 OOD 物体也能得到合理 task representation, 因为不依赖 object model。

**(3) Residual action prediction (核心 sim-to-real trick)**:

原 DP3 / Diffusion Policy 预测 absolute end-effector pose 或 absolute joint angle $q_{t+1:t+T_a}$。

GLIDE 改成 residual:
$$a_{t+1:t+T_a} = \{ q_i - q_t \}_{i=t+1}^{t+T_a}$$

$q_t$ 是 current joint position, $q_i$ 是 predicted future joint position, $a$ 是 delta。

为什么这 work (build intuition):
- Absolute joint angle: 不同 episode 起点 $q_0^a$ 不同, 同样的 "push object 45°" 在不同 $q_0^a$ 下对应不同 absolute target $q$, 误差分布 broad
- Residual: "把 joint 往这个方向推 N 度" 是一个 object-centric 的 invariant 量, scale / shift 在训练集内 roughly 一致, diffusion model 学这个分布更容易

Ablation Tab. III 里, residual action 在 sim 几乎无影响 (0.78 vs 0.74), 在 real 从 0.52 → 0.80, 提升 +28 pt, 非常显著。 这是典型 "sim-to-real gap 在 absolute quantity 上放大, 在 relative quantity 上消解" 的 pattern。

**(4) Flying Point Augmentation**:

每个 point 以 0.5% 概率被加 large Gaussian noise "飞走"。 模拟 RealSense D455 的 outlier noise (depth hole-filling artifact, 边缘飞点, 玻璃 / 镜面反射错误)。

为什么 work: sim 的 point cloud 是 clean 的, real 永远有 outlier, 标准 data aug (jittering) 不够 cover 这种 long-tail。 用 small probability + large magnitude 让 model 见过 outlier, 等价于 robust loss + outlier rejection 训练。 Ablation Tab. III: real-world 0.80 → 0.32 (-48 pt), 必不可少。

**(5) Large prediction horizon $T_a$**:
训练用 $T_a = 64$, 推理用 $T_a = 20$ (DP3 默认 8)。

Hypothesis: contact-rich 任务需要 plan ahead 看到整个 contact phase, $T_a=8$ 太短 (一两次接触), policy 不知道未来要往哪儿推。 $T_a=20$ 能 cover 一个完整 contact phase, trajectory smoothness 大幅提升。

但 $T_a=40, 64$ 反而下降, 因为 open-loop horizon 长, 失去 closed-loop feedback 的 reactive 能力, 物体 slip 或 plan 不准时无法及时纠正。 $T_a=20$ 是 sweet spot。

Ablation Tab. IV:
| $T_a$ | Fixed 45° | Random Rot |
|---|---|---|
| 8 | 0.44 | 0.27 |
| 20 | 0.74 | 0.40 |
| 40 | 0.76 | 0.34 |
| 64 | 0.77 | 0.20 |

可以看到 Fixed rotation 一直涨 (open-loop OK 因为 target 固定), Random rotation 在 $T_a=20$ 后单调下降 (closed-loop 重要)。

**(6) Point cloud preprocessing**:
- Clip 到 robot workspace (去除背景桌椅)
- Uncolored (只 geometry, 因为 sim 渲染 color 不真实)

---

## 3. Experiment Analysis

### 3.1 In-Distribution (Tab. I)

| Task | Planner (Sim) | Policy (Sim) | Policy (Real) |
|---|---|---|---|
| Fixed 45° | 0.337 | 0.740 | **0.800** |
| Random Easy | 0.227 | 0.610 | 0.600 |
| Random Med | 0.141 | 0.410 | 0.360 |
| Random Hard | 0.099 | 0.180 | 0.200 |
| Random Overall | 0.156 | 0.400 | 0.387 |

惊人观察: **Policy success rate > Planner success rate** (Fixed 0.80 vs 0.34)。 这听起来反直觉, 因为 policy 是从 planner 数据训出来的。

可能的解释:
1. Filtered BC 把 planner 失败的 trajectory 都丢了, policy 学的全是 success trajectory
2. Policy 用 visual feedback, 可以做 closed-loop correction, 而 planner 是 open-loop (在 random reset 上可能选了 bad grasp seed)
3. Policy 在 trajectory 分布上做了 multimodal average, 学到了 "robust action mode", 而 planner 每次 greedy 选特定 mode 可能恰好遇到 local minimum

这是 planning-guided learning 相比 pure planning 的最大 selling point: **filtering + smoothing + closed-loop**。

### 3.2 OOD (Tab. II)

OOD objects: 软的 (rubber, fabric), 曲面, 上窄下宽, overfilled container (装满杂物, weight 更大且 boundary irregular)。

| Task | Empty | Overfilled | Overall |
|---|---|---|---|
| Fixed 45° | 0.688 | 0.625 | 0.657 |
| Random | 0.250 | 0.313 | 0.282 |

Performance drop 相对 in-distribution, 但仍然可用 (Fixed ~66%, Random ~28%)。 Empty vs Overfilled 几乎无差别, 说明 policy 主要是从 geometry 推断 action, weight 的影响在 contact-rich setting 下被 arm 力量吸收了 (没 end-effector, 直接用 link 推, arm 有 enough torque margin)。

Inflatable toy (Fig. 6): legged base + 高 center of mass, Fixed 52%, Random 28%。 容易 tipping, 需要 precise bimanual coordination, 是更具挑战的 OOD。

### 3.3 Ablation Summary

**Residual action + Flying point aug** (Tab. III):
- Sim 几乎无影响 (filter / segment / noise 在 sim clean data 上不发挥作用)
- Real: 两者齐备 0.80, 任一缺失 ~0.32-0.52, 都缺 0.00

这是非常干净的 sim-to-real gap 证据: 两个 trick 都只在 real 有用, 因为它们处理的是 real-world 特有的 distribution shift。

**Data scale** (Tab. V):
| Demos | Fixed | Random |
|---|---|---|
| 500 | 0.33 | 0.03 |
| 2500 | 0.69 | 0.17 |
| 7500 | 0.74 | 0.33 |
| 12000 | 0.80 | 0.40 |

性能仍在上升, 没有 plateau, 说明继续 scale data 还能涨。 Random rotation 对 data 更 sensitive (Easy task 500 demo 就 0.33, Hard task 12000 demo 才 0.40)。

### 3.4 Failure Mode Analysis (Sec IV-F)

52% Random-Hard failure: robot stuck in poor joint configuration (joint limit + bad grasp angle, 无法继续 rotate, 需要重新 approach 但 policy 不知道)
20%: object slippage (contact unstable)
16%: torque exceed (squeeze too hard)

这给未来方向: 在 near-failure state 也生成 demo (recovery data), 让 policy 学 "脱困"。

---

## 4. 与相关工作的联系 (build your intuition)

- **Diffusion Policy (Chi et al.)**: https://diffusion-policy.cs.columbia.edu/ — backbone 思路来源, 但 per-task params, 不 condition task。
- **DP3 (Ze et al.)**: https://3d-diffusion-policy.github.io/ — point cloud 版 diffusion policy, GLIDE 直接 build on 它。
- **Pang et al. "Global Planning for Contact-Rich Manipulation via Local Smoothing of Quasi-Dynamic Contact Models"**: https://arxiv.org/abs/2206.11135 — planner 框架来源。
- **Suh et al. "Bundled Gradients through Contact"**: https://arxiv.org/abs/2202.00786 — randomized smoothing for contact, 让 TO 可微。
- **Drake**: https://drake.mit.edu/ — MIT/Toyota 高保真 simulator, 提供 physics rollout verification。
- **ALOHA / Mobile ALOHA (Zhao, Fu, Finn et al.)**: https://mobile-aloha.github.io/ — bimanual teleop 替代路线, 用低 cost hardware 采 real demo, 但对 contact-rich 长 horizon task 采集成本仍高。
- **UMI (Chi et al.)**: https://universal-manipulation-interface.github.io/ — in-the-wild 教学工具, 思路上同类, 但 GLIDE 走 simulation 这条路。
- **Diffusion Policy as Visuomotor Policy**: 之所以用 diffusion 而不是 VAE / GAN / Flow Matching, 因为 diffusion 能 capture multimodal action distribution (一条 task 可以有多种解法, e.g. 左手抓 / 右手抓, 不同 grasp angle), diffusion 的 score-based 形式对 multimodal 更友好。

---

## 5. 关键 Take-away (给你 build intuition)

1. **Data bottleneck > Model bottleneck**: contact-rich bimanual 一直卡住, 不是因为 policy 架构不行, 是因为没人能 scale 出高质量 demo。 用 planner 自动 gen data 是 path forward。
2. **Greedy single-step TO 替代 long-horizon TO**: 牺牲 global optimality 换 scalability, 用 physics simulator rollout 做 verification 弥补质量。 这个 trade-off 在 data generation 场景下 almost always 值得。
3. **Filtered BC > Raw BC**: planner 产出有 noise, physics-based filter 把 noise 砍掉, 让 policy 学到的是 "成功且 efficient" 的 sub-distribution。
4. **Residual / relative quantity 是 sim-to-real 的关键 invariance**: absolute joint angle 在 sim/real 之间 distribution shift 大, relative delta 几乎 invariant。 这是 robotics sim-to-real 的通用原则之一。
5. **Flying point aug / outlier noise injection**: 处理 real sensor noise 的 long-tail, 是比 jittering 更对路的 data aug, 因为 long-tail 才是 deployment-time 真正的 killer。
6. **Closed-loop policy > Open-loop planner**: policy 在 success rate 上反超 planner (0.80 vs 0.34) 说明 visual feedback + multimodal smoothing + learned prior 在 stochastic 环境下比 greedy planner 更 robust。
7. **OOD generalization 来自 geometry-only observation**: 不依赖 object shape prior, 不依赖 object frame, 只用 raw point cloud + tracked keypoints, 让 policy 学到的是 "geometry → contact action" 的 generic mapping。
8. **Failure mode 是 future work 的 roadmap**: joint-limit stuck 52%, 说明需要 recovery / near-failure demo; slippage 20% 说明 contact stability 还没完全学到 (可能需要 tactile sensing, 但 paper 没用); torque exceed 16% 说明 force awareness 不足。

---

## 6. 局限与未来方向 (作者自己 admit)

- 只用 primitive shape (box) 训练, OOD container 是 "近似 box" 的物体, 真正 arbitrary geometry (e.g. 长杆, 球, 不规则石头) 还没验证
- Tabletop SE(2) only, 不做 6D reorientation
- 没用 tactile / force feedback, contact stability 完全靠 visual + proprioception, 这是 slippage 20% 的根源
- Demo gen 仍要 2 天 / 96 CPU, 还能 scale 但成本不低
- Future: large object dataset (e.g. ABO, GSO, Objaverse) diversify, 更 dexterous task (in-hand pivot, throw/catch)

---

## 参考 Web Links

- GLIDE 项目主页: https://glide-manip.github.io/
- Diffusion Policy (Columbia/Stanford/Mit): https://diffusion-policy.cs.columbia.edu/
- DP3 (3D Diffusion Policy, Tsinghua): https://3d-diffusion-policy.github.io/
- Pang et al. Global Planning for Contact-Rich Manipulation (T-RO 2023): https://arxiv.org/abs/2206.11135
- Suh, Pang, Zhao, Tedrake 2025 preprint (Contact Trust Region): https://groups.csail.mit.edu/locomotion/papers/2025-suh-contact-trust-region.pdf
- Suh et al. Bundled Gradients through Contact (RA-L 2022): https://arxiv.org/abs/2202.00786
- Le Cleac'h et al. Fast CIMP (T-RO 2024): https://arxiv.org/abs/2312.10701
- Drake simulator: https://drake.mit.edu/
- ALOHA (Zhao et al. RSS 2023): https://tonyzhaozh.github.io/aloha/
- Mobile ALOHA (Fu, Zhao, Finn 2024): https://mobile-aloha.github.io/
- UMI (Chi et al. 2024): https://universal-manipulation-interface.github.io/
- Grounding DINO: https://github.com/IDEA-Research/GroundingDINO
- CoTracker: https://github.com/facebookresearch/co-tracker
- TAPIR: https://deepmind-tapir.github.io/
- Kuka LBR iiwa 7-DoF: https://www.kuka.com/en-us/products/robot-systems/industrial-robots/lbr-iiwa
- Intel RealSense D455: https://www.intelrealsense.com/depth-camera-d455/

如果你 (Karpathy) 想进一步 explore, 我会建议看 Suh/Pang/Tedrake 的 contact smoothing 那条线 — 那是 GLIDE planner 背后的真正 engine, 理解了它你就能看清 "为什么 single-step greedy QP 能解 contact-rich 问题"。 再结合 Diffusion Policy 的 score-based formulation, 你会发现这套 planning + diffusion 的 pipeline 在概念上与你在 nanoGPT 里强调的 "next-token prediction captures world model" 有异曲同工之处 — diffusion 在 action space 上学的是 "action distribution conditioned on geometry", planner 在 demo 里把 high-value modes 标记出来, policy 用 score function 把它们 smooth interpolate 到 OOD 上。
