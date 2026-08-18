---
source_pdf: Agile But Safe Learning Collision-Free.pdf
paper_sha256: c7991f37edb30cde6433eef0a767755f1cf577e282471c2d1010955c0253b840
processed_at: '2026-08-18T00:18:43-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# ABS 用人话讲

## 这篇paper在搞啥

一句话：**让四足机器人跑得快（>3 m/s）还不撞东西**。

听起来简单，其实特别难。之前的两条路都走不通：

- **保守派**：机器人慢慢走（<1 m/s），保证不撞。能用，但太慢，浪费了机器人的运动能力。
- **激进派**：让机器人狂奔（RL训练），但不管撞不撞。跑得快，但在杂乱环境里会撞墙撞人。

ABS说：**我全都要**。跑得快 + 不撞。

---

## 为什么不能"训练时加个撞墙惩罚"就好了？

这是最直觉的想法：用RL训练机器人跑步，reward里加一个 "-100 if 撞墙"，机器人不就自己学会避障了吗？

作者试了（PPO-Lagrangian baseline），结果：

- 撞墙率确实从21%降到9%
- 但平均速度从2.4 m/s暴跌到1.4 m/s
- 而且经常站着不动（13%超时）

**原因**：RL很聪明但很懒。一旦发现"撞墙扣100分"，最优策略就是"别跑了，走着多安全"。Lagrangian那个safety constraint在训练早期就压上来，把探索空间砍掉一大半，机器人学到的不是"巧妙的避障跑步"，而是"怂着走"。

这就是**agility-safety trade-off边界**：你想更安全，就得更慢；想更快，就得冒撞的风险。end-to-end safe RL怎么调都在这条线上。

---

## ABS的核心招数：两个policy + 一个裁判

**类比**：想想人怎么在拥挤的街上跑步。

你脑子里有两个"模式"：
1. **正常跑步模式**：眼睛看路，腿使劲跑，绕开行人。大部分时间用这个。
2. **紧急避险模式**：突然有人窜出来，你可能急刹车、侧身跳、甚至膝盖着地摔倒。只有危险时才用。

关键是：你**平时跑步时脑子里不会一直想着"别撞别撞"**，那样跑不快。你正常跑，遇到危险才切换到避险反射。

ABS就是这个思路：

- **Agile policy（跑快policy）**：只管跑得快+绕障碍，不管安不安全
- **Recovery policy（救命policy）**：紧急刹车/侧移的技能包
- **RA value（裁判）**：每时每刻判断"现在安不安全"

裁判说安全 → agile policy跑  
裁判说危险 → recovery policy接管制

这样agile policy不被安全约束拖累，可以尽情狂奔；recovery policy只在关键时刻救场。

---

## 三个组件各自在干嘛

### 1. Agile policy：让它跑快

**关键设计选择**：用"到达目标点"而不是"跟踪速度指令"来训练。

为什么？之前的agile locomotion工作都是训练机器人"跟踪一个速度指令(v_x, v_y, ω)"，然后上面接一个导航规划器输出速度指令。问题是：

- 导航规划器不知道机器人能不能跟踪这个速度
- 为了安全，规划器只敢输出低速指令
- 机器人的运动能力被规划器层cap住了

ABS直接让policy接收"目标点在哪"，自己决定怎么跑过去。locomotion和navigation耦合在一起，机器人自己探索出最快的方式。

**有趣的发现**：这样训练出来的机器人自己学会了**gallop（飞奔）**，而速度跟踪的方法学到的是trot（小跑）。飞奔时一直有一只脚着地（1个自由度不可控），小跑时对角两脚同时离地（3个自由度不可控），所以飞奔在现实中更稳。这个不是设计出来的，是RL自己explore出来的。

**Reward设计**：
- 跑到目标点：+60分
- 快速跑：+10分（核心agility reward）
- 撞墙：-100分（但只是penalty，不是hard constraint）
- 还有一堆regularization：别飞太高、别扭太狠、扭矩别太大、别四脚同时离地...

这些regularization不是为了让它跑得快，是为了sim-to-real时不翻车（仿真器在大扭矩/大速度/极限姿态时不准）。

### 2. RA value：裁判怎么判

这是最技术的部分，但直觉很简单。

**RA value = "从这个状态出发，agile policy还能不能安全到达目标"**

- RA value ≤ 0：安全，继续跑
- RA value > 0：危险，准备切换recovery

怎么训练这个裁判？用control theory的reach-avoid理论。

**直觉**：RA value像一个"危险度地图"。在障碍物附近，危险度高（红色）；在空旷处，危险度低（绿色）。而且这个地图会随机器人速度变化——跑得越快，红色区域越大（因为刹不住）。

**两个关键trick**：

**Trick 1: Soft label**  
撞墙不是一个binary事件。撞墙前10帧的状态，其实已经"越来越危险"了。把撞墙前的状态relabel成 -0.8, -0.6, ..., 0.8, 1.0（渐变危险），而不是 0, 0, ..., 0, 1（突然撞）。这样RA value network能学会"预测危险"而不是"事后诸葛亮"。

**Trick 2: Policy-conditioned**  
之前的reach-avoid理论算的是"不管用什么策略，这个状态安不安全"——需要在action space上做minimization，很难。ABS改成"在agile policy下，这个状态安不安全"——只需要rollout agile policy，简单很多。代价是换agile policy就要重训RA value，但production系统policy是frozen的，所以OK。

**怎么用RA value**：  
裁判说危险时，不是直接切recovery就完了。还要**用RA value的梯度**去找一个"安全的twist指令"：

> 在所有可能的twist里，找一个(1)离目标更近(2)但RA value还在安全线以下的

这个twist交给recovery policy去执行。所以RA value不只是binary开关，还是recovery policy的**方向盘**。

### 3. Recovery policy：紧急刹车技能包

这个比较简单。训练一个"快速跟踪速度指令"的policy：

- 输入：当前状态 + 目标twist (v_x, v_y, ω)
- 输出：12个关节角度
- 训练：让它尽量快地跟踪上给定的twist

**不需要视觉**，因为安全的twist已经由RA value优化好了，recovery只管忠实执行。

**允许膝盖着地**：紧急刹车时膝盖着地是OK的（就像人急停时可能半蹲），这样刹车更有效。agile policy不允许这样，但recovery可以。

训练很快，10分钟就收敛了，因为任务简单（就是跟踪速度）。

### 4. Ray prediction：怎么"看"障碍物

**问题**：agile policy和RA value都需要知道"周围障碍物在哪"。直接用depth image太noisy、太高维、训练难。

**方案**：用11条射线，像稀疏激光雷达一样，从机器人往前扇形发射，记录每条射线打到障碍物的距离。

训练一个ResNet-18，输入depth image，输出11个ray distance。

**好处**：
- Depth image的noise/光照问题全由这个网络处理
- 上游的agile policy和RA value只看干净的11维ray，训练容易
- 低维输入更容易generalize到不同障碍物形状

训练时sim里的depth image很干净，真实世界很noisy，所以做了4种data augmentation（翻转、随机擦除、高斯模糊、高斯噪声）来缩小gap。

---

## 为什么这套设计能work

几个关键点：

### 1. 分而治之

不试图让一个policy同时做"跑快"和"不撞"。分开：
- Agile policy专攻速度
- Recovery policy专攻安全
- RA value负责判断和协调

每个组件的目标单一，容易训练，容易做好。

### 2. Recovery只管执行，不管决策

Recovery policy不需要看障碍物、不需要判断安不安全。安全的twist已经由RA value的优化算好了，recovery只管忠实执行。这让recovery policy极其简单、快速、robust。

### 3. RA value用梯度引导，不只是开关

如果RA value只是"安全/危险"的binary flag，recovery policy接到控制权后不知道往哪跑。ABS用RA value的梯度去做constrained optimization，找到"最安全且最接近目标"的twist。这是相比Recovery RL的关键改进——safety critic和backup policy有真正的interplay。

### 4. Sim-to-real的几个trick

- **Illusion randomization**：训练时只见过cylinder，但把"远距离射线"随机化，让policy只care近处障碍物。这样到现实见到墙、家具也不会慌。
- **ERFI-50**：random torque perturbation模拟motor sim-to-real gap。没有这个，机器人跑起来头会撞地（motor延迟在sim里没有）。
- **Regularization不碰hardware limit**：扭矩上限0.85倍、速度上限0.9倍、位置上限0.95倍。让policy不要探索到sim不准的区域。
- **Goal-reaching学到gallop**：飞奔比小跑sim-to-real更稳（一直有脚着地）。

---

## 实验结果怎么说

### 仿真

ABS打破了agility-safety trade-off边界：
- 比纯agile policy：撞墙率从21.7%降到5.7%，速度只降0.3 m/s
- 比PPO-Lagrangian：撞墙率更低，速度快1.5倍

### 现实

Unitree Go1机器人，3个场景：
- 昏暗走廊：9/10成功
- 家具大厅：10/10成功
- 户外操场：10/10成功，峰值3.1 m/s

还能：
- 雪地跑
- 背12kg（等于自重）
- 被球砸
- 被踢

### 有意思的bonus

Goal-reaching formulation让你可以在运行中**瞬间改目标点**实现急转弯。比如"前进→急右转→前进"，机器人能3 m/s前进+6 rad/s急转。这在velocity-tracking formulation里是out-of-distribution（急转意味着velocity command突变），但goal-reaching是in-distribution（目标点本来就在变）。

---

## 有啥局限

1. **太密的障碍物会懵**：形成local minimum，机器人在原地转圈（timeout率高）。需要记忆或全局规划。
2. **只能应对慢的动态障碍物**：RA value用static obstacle训练，只能generalize到quasi-static。快速移动的物体可能撞上。
3. **只有2D**：不能跳、不能爬楼梯。3D地形下locomotion和避障耦合更复杂。
4. **视觉在暗处会失败**：走廊里那次撞墙就是光线太暗ray prediction挂了。
5. **没有implicit system identification**：其他工作用temporal latent embedding建模现实动力学，但ABS的policy switch会让embedding out-of-distribution。

---

## 一句话总结

**ABS让机器人像人一样在拥挤环境跑步：平时放开跑，危险时本能避险，用"危险度地图"决定何时切换、往哪躲。关键是用两个policy分离agility和safety，用control-theoretic的reach-avoid value做裁判和方向盘，打破end-to-end safe RL的trade-off边界。**

直觉上这就是hierarchical control的胜利：不要指望一个end-to-end network同时做好速度和安全，而是用modular设计让每个component各司其职。

---

# Agile But Safe (ABS): 高速 collision-free legged locomotion

这篇 paper 来自 CMU LeCAR Lab (Guanya Shi, Changliu Liu) 和 ETH Zurich Hutter group (Chong Zhang)，2024 年的 work。核心要解决的问题是：四足机器人在 cluttered environment 中如何同时实现 high agility (>3 m/s) 和 collision-free safety。之前的工作要么 conservative (<1 m/s) 保 safety，要么只追求 agility 不 care collision。ABS 用一个 dual-policy setup + control-theoretic reach-avoid value 打破了 agility-safety trade-off boundary。

项目主页: https://agile-but-safe.github.io  
代码: https://github.com/LeCAR-Lab/ABS

---

## I. 核心思想：为什么不能 end-to-end safe RL 解决？

Karpathy 你应该会很熟悉一个直觉：end-to-end RL 学 agile locomotion 已经很强了 (Rapid locomotion, ANYmal parkour, extreme parkour)，为什么不能直接加一个 collision penalty 让它自己学 safe？

作者用 PPO-Lagrangian (LAG baseline) 实验证明这是 futile 的。Table III 里看得很清楚：

| Setting | Success | Collision | Timeout | v_peak | v_avg |
|---|---|---|---|---|---|
| ABS-n | 79.1±4.4 | 5.7±2.9 | 15.2±2.1 | 3.48±0.06 | 2.08±0.01 |
| π^Agile-n | 77.3±4.2 | 21.7±3.9 | 1.0±0.4 | 3.55±0.04 | 2.39±0.04 |
| LAG-n | 77.4±11.5 | 9.1±1.8 | 13.5±13.0 | 2.45±0.07 | 1.41±0.03 |

LAG 把 collision rate 从 21.7% 压到 9.1%，但代价是 average speed 从 2.39 m/s 跌到 1.41 m/s，而且 timeout 飙到 13.5%。这就是经典的 safe RL 困境：Lagrangian multiplier 在 convergence 前就 enforce constraint，hinder exploration，最终 policy 学到的是 "走慢点别撞"，agility 被严重 cap。

ABS 的策略：**agile policy 想怎么 fast 就怎么 fast，遇到 risky 状态时切换到 recovery policy，由 control-theoretic 的 reach-avoid value 来决定何时切、怎么 recover**。这样 agile policy 不需要被 safety constraint 拖累，可以尽情 explore high-speed gait；recovery policy 只在 critical 时刻介入，作为 safety shield。

直觉类比：这不是 self-driving 里经典的 "nominal controller + safety filter" (比如 CBF shield) 思路，只不过这里 nominal controller 是 learned RL policy，safety filter 是 learned reach-avoid value + learned recovery policy，全部 model-free。

---

## II. 四个 module 的整体架构

Figure 2 画得很清楚，ABS 在 simulation 里训练 4 个 module：

1. **Agile policy** π^Agile: goal-conditioned，输入 proprioception + exteroception (ray distances) + goal command，输出 12-d joint targets
2. **Reach-Avoid value network** V̂: 输入 reduced observation (twist + goal xy + rays)，输出 scalar RA value，conditioned on π^Agile
3. **Recovery policy** π^Recovery: twist-tracking，输入 proprioception + twist command，输出 12-d joint targets
4. **Ray-prediction network**: ResNet-18，输入 depth image，输出 11-d ray distances

部署时 (Figure 2b)：
- 计算 V̂(o^RA)
- 如果 V̂ < V_threshold = -0.05：用 agile policy
- 如果 V̂ ≥ V_threshold：solve 一个 constrained optimization 找 safe twist command，让 recovery policy track 这个 twist

---

## III. Agile Policy: goal-reaching formulation 的关键 insight

### A. 为什么 goal-reaching 而不是 velocity-tracking

这部分对 build intuition 很关键。Velocity-tracking (Rapid locomotion [48], ANYmal [50]) 是 locomotion RL 的主流 formulation：policy 接收 (v_x^c, v_y^c, ω_z^c) command，输出 joint action 跟踪这个 velocity。

但 velocity-tracking 在 navigation 场景下有一个 fundamental problem：**locomotion 和 navigation 被 decouple**。你需要一个 high-level planner 输出 velocity command，但 planner 不知道 low-level tracking error，也不知道 motor limit。为了 safety，planner 只能保守地输出低 velocity command，locomotion policy 的 agility 被 cap 在 planner 层。

Goal-reaching formulation 直接让 policy 接收 goal position (相对位置 + heading)，自己决定怎么 locomote 到达 goal。这样 locomotion 和 collision avoidance 是 coupled 的，policy 可以 fully unleash motor agility。Table IV 的对比很 informative：

| Term | Our π^Agile | Rapid [48] |
|---|---|---|
| Gait pattern | gallop | near trot |
| Max # uncontrollable DoFs | 1 | 3 |
| Peak vel sim | 4.0 m/s | 4.1 m/s |
| Peak torque sim | 23.5 Nm | 35.5 Nm |
| Peak joint vel sim | 22.0 rad/s | 30.0 rad/s |
| Peak vel real | 3.1 m/s | 2.5 m/s |
| Changing vel for steering | in distribution | out of distribution |

注意一个有意思的发现：goal-reaching 学到了 **gallop gait**（飞奔），而 velocity-tracking 学到的是 near trot。Gallop 在 sim-to-real 时更稳定，因为只有一个 DoF uncontrollable（飞奔时有一只脚在地面支撑），而 trot 有三个 DoF uncontrollable（对角两脚同时离地），real-world motor delay 下容易 destabilize。这是论文一个 empirical insight，作者说 "empirically find goal-reaching benefits sim-to-real"。

直觉：goal-reaching 让 policy 自己 explore 整个 motor skill 空间，找到最适合 reach goal 的 gait；velocity-tracking 则被 "tracking velocity" 这个 myopic objective constrain 住了。

### B. Observation space

o^Agile = [c_f (4-d foot contacts); ω (3-d base angular vel); g (3-d projected gravity); G^c (goal command, relative pos + heading in base frame); T-t (time left); q (12-d joint pos); q̇ (12-d joint vel); a_{t-1} (prev action); R (11-d log ray distances)]

注意：**只用 g 和 G^c 需要 state estimator**，其他都是 raw sensor。g 是 IMU-based orientation（roll/pitch 通常很准），G^c 是 odometry（可以 drift，policy robust 到可以运行时改变 goal）。这是为什么 sim-to-real 时不依赖高精度 state estimator 的关键。

### C. Reward 设计

r = r_penalty + r_task + r_regularization

**Penalty**: r_penalty = -100 · 1(undesired collision)，undesired collision 指 base/thigh/calf 碰撞 + 水平方向的 foot collision（垂直方向 foot contact 是正常的，允许）。

**Task** (Eq 9):
r_task = 60·r_possoft + 60·r_postight + 30·r_heading - 10·r_stand + 10·r_agile - 20·r_stall

- r_possoft: σ_soft=2m, T_r=2s，soft 鼓励 explore
- r_postight: σ_tight=0.5m, T_r=1s，tight 强制停在 goal
- r_heading: σ_heading=1 rad, T_r=2s，距离 > σ_soft 时 disable（不影响 collision avoidance）
- r_stand: 在 goal 附近鼓励站立姿态
- r_agile: **核心 agility reward**

r_agile = max{ReLU(v_x / v_max) · 1(correct direction), ReLU(-v_y/v_max) · 1(correct direction), ...}

其中 v_max = 4.5 m/s（hardware datasheet 上限，不可达），"correct direction" 指 robot heading 和 robot-goal line 夹角 < 105°。这个 reward 让 robot 要么 fast 跑要么 stay at goal。注意用了 ReLU 而不是 abs，避免 reward hacking（倒着跑）。

**Regularization** (Eq 13): 一大堆 term，Table XI 解释了每个的目的：
- -2·v_z²: 减少 vertical oscillation
- -0.05·(ω_x² + ω_y²): 减少 rotational oscillation
- -20·(g_x² + g_y²): 减少 tilting
- -0.0005·||τ||²: 减少 mechanical stress / power
- -20·Σ ReLU(|τ_i| - 0.85·τ_{i,lim}): 避免大 torque 造成 sim-to-real gap
- -0.0005·||q̇||²: 减少激进 motion
- -20·Σ ReLU(|q̇_i| - 0.9·q̇_{i,lim}): 避免 high joint vel 导致 sim 不准
- -20·Σ ReLU(|q_i| - 0.95·q_{i,lim}): 避免 joint pos 接近 limit
- -2×10⁻⁷·||q̈||²: 减少 jerk
- -4×10⁻⁶·||ȧ||²: smooth action
- -20·1(fly): 惩罚四脚同时离地（base 不可控，威胁 safety）

这些 regularization term 看起来很多但其实都是 standard practice in legged RL ([55, 48, 82, 29])。关键 insight 是 **0.85 / 0.9 / 0.95 这些阈值** —— 不让 policy explore 到 hardware limit 附近，因为 sim 在那里不准。

### D. Domain randomization 的两个 key trick

Table II 列了所有 randomization，但作者特别强调两个：

1. **Illusion**: 当 ray distance > d_goal + 0.3 时，用 U(d_goal+0.3, ray_distance) 覆盖观测值。这是为了让 policy 对**没见过的 geometry (e.g. walls)** robust。训练时只有 cylinder obstacle，real world 有 walls、furniture 等。如果 ray distance 显示 "很远" 的 obstacle，policy 可能会 tremble（因为没见过这种 long ray pattern）。Illusion 把这种 long ray 随机化成 short ray，让 policy 学到 "只 care 近处 obstacle"。

2. **ERFI-50** (Campanaro et al. [8], https://arxiv.org/abs/2209.12878): random torque perturbation，0.78 Nm × difficulty level。这是 implicit modeling motor sim-to-real gap。Figure 11 显示，没有 ERFI-50 的话 robot 跑起来 head 会撞地（motor response delay 在 sim 里没有，real world 有）。Curriculum 加上避免 early stage 学习受阻。

---

## IV. Reach-Avoid Value Network: control-theoretic safety shield

这是 paper 最 technical 的部分。我要详细讲，因为这是 ABS 的灵魂。

### A. Hamilton-Jacobi Reachability 背景

经典 HJ reachability (Bansal et al. 2017, https://arxiv.org/abs/1709.07523) 分析一个 dynamical system：给定一个 failure set F 和 target set Θ，计算 "reach-avoid set" RA(Θ; F) —— 所有能保证 "在到达 Θ 之前不进入 F" 的初始状态。

形式化 (Eq 3)：RA^π(Θ; F) := {s_t ∈ S | ξ_{s_t}^π(T-t) ∈ Θ ∧ ∀t' ∈ [0, T-t], ξ_{s_t}^π(t') ∉ F}

其中 ξ_{s_t}^π 是从 s_t 出发用 policy π rollout 的 trajectory。

对应的 value function (Eq 4)：
V_{RA*}^π(s) = max{ζ(s), min{l(s), V_{RA*}^π(f(s, π(s)))}}

变量解释：
- s ∈ S: state
- ζ(s): Lipschitz continuous failure indicator，ζ(s) > 0 ⟺ s ∈ F (e.g. collision)
- l(s): Lipschitz continuous target indicator，l(s) ≤ 0 ⟺ s ∈ Θ (e.g. 到达 goal)
- f(s, a): deterministic dynamics
- π(s): policy

这个 Bellman equation 的直觉：
- 如果当前 s 在 F 里，value = ζ(s) > 0 (unsafe)
- 如果当前 s 在 Θ 里，value = min{l(s), next value} = l(s) ≤ 0 (safe, reached)
- 否则 value = next value (递归)

V_{RA*}^π(s) ≤ 0 ⟺ s ∈ RA^π(Θ; F)，即 s 是 safe starting state。

### B. 为什么经典 RA Bellman 不适合 data-driven

问题：Eq (4) **不是 contraction**。在 value function 空间里 iteratively apply 这个 Bellman operator 不保证收敛。经典 HJ reachability 用 PDE solver 算，但 high-dimensional state space 下计算爆炸 (curse of dimensionality)。

### C. Time-discounted RA Bellman (Hsu et al. 2021)

Hsu et al. [30] (https://robotsconference.org/2021/program/) 提出加 time discount 让它变成 contraction。Eq (5)：

V_RA^π(s) = γ_RA · max{ζ(s), min{l(s), V_RA^π(f(s, π(s)))}} + (1-γ_RA) · max{l(s), ζ(s)}

变量：
- γ_RA ∈ [0, 1): discount factor
- 第一项：γ_RA 加权下一时刻的 RA value
- 第二项：(1-γ_RA) 加权当前时刻的 "instantaneous" 信号 max{l(s), ζ(s)}

性质：
- V_RA^π 是 V_{RA*}^π 的 **under-approximation**（保守估计）
- γ_RA → 1 时 V_RA^π → V_{RA*}^π
- V_RA^π(s) ≤ 0 ⟹ s ∈ RA^π(Θ; F)（保守保证，不会 false positive 说 safe 但其实 unsafe）

直觉：discount 把 "无限远未来" 的 risk 衰减掉，让 Bellman operator 变成 contraction，可以用 value iteration / data-driven approximation 收敛。代价是 under-approximation（一些 actually safe 的状态被标为 unsafe），但这正好是 safety-critical 想要的（宁可保守）。

### D. Policy-conditioned RA value (ABS 的创新)

Hsu et al. 学的是 **policy-agnostic** global RA value，需要在 action space 上 minimize：
V_RA(s) = min over a of V_RA^π(s, a)

这是为了回答 "在 state s，是否存在某个 action 让系统 safe"。但这有两个问题：
1. High-dim action space 上 minimization 难
2. Identifiability issue: global RA set 可能很难学

ABS 改成 **policy-conditioned**：只学 π^Agile 下的 RA value。这极大简化了 learning：
- 不需要 action space minimization
- 可以 **two-stage offline learning**：先 train agile policy，再 collect rollout 数据 train RA value network，两个阶段解耦，稳定性好

### E. RA value network 训练

Reduced observation (Eq 14)：
o^RA = [v (3-d base linear vel); ω (3-d base angular vel); G^c_{x,y} (2-d goal xy in robot frame); R (11-d log ray distances)]

注意去掉了 joint-level observation (q, q̇)，因为 high-dim 且和 goal-reaching 关系小。这个设计选择很重要：reduced input 让 RA value network 容易学，且 generalize 好。

Loss (Eq 16-17)：
L = (1/T) Σ_t (V̂(o_t^RA) - V̂^target)²

V̂^target = γ_RA · max{ζ(s_t), min{l(s_t), V̂^old(o_{t+1}^RA)}} + (1-γ_RA) · max{l(s_t), ζ(s_t)}

V̂^old 是上一轮 iter 的网络（类似 target network）。Terminal: V̂^old(o_{T+1}^RA) = +∞（没到达 goal 的话视为 unsafe）。

γ_RA = 0.9999999（几乎 1，尽量接近 ground truth）

数据：用 trained agile policy rollout 200k episodes，obstacle 分布用最高难度。

### F. Lipschitz continuity 的 soft trick

理论上 l(s) 和 ζ(s) 需要 Lipschitz continuous ([20], https://arxiv.org/abs/1509.07693) 来保证 value function 解的存在唯一性。

l(s) = tanh(log(d_goal / σ_tight))，自然 Lipschitz，bounded 在 (-1, 1)，d_goal ≤ σ_tight 时 l(s) ≤ 0（reached goal）。

ζ(s) 原始定义 (Eq 19)：ζ(s) = 2·1(collision) - 1，**不 Lipschitz**（discrete jump）。

ABS 的 soften trick：collision 发生时，把**前 10 个 timestep** 的 ζ relabel 为 -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0。即 collision 前的状态被 retroactively 标为 "越来越 unsafe"。

Figure 4 的可视化很 informative：用 raw ζ 的话，RA value 在 obstacle 侧面无法指示 collision（学不出来），并且 obstacle 前方有 local minima。用 softened ζ 的话，RA value landscape 平滑且正确。

Table VI 的 ablation 也佐证：softened ζ 把 collision rate 从 14.7% 降到 5.7%（success rate 略降 81.7→79.1，因为更保守）。

直觉：soft label 让 network 学到 "collision 不是 binary event，而是 trajectory 接近 collision 的过程"，这样 value function 能 anticipate collision，提前预警。

### G. RA value 的使用：recovery policy guidance

当 V̂ ≥ V_threshold = -0.05 时，触发 recovery policy。但 recovery policy 需要 track 一个 twist command，这个 twist 怎么选？这就是 Eq (21)：

tw^c = argmin d_goal^future s.t. V̂([tw^c; G^c_{x,y}; R]) < V_threshold

即：找一个 twist command，使得 (1) 跟踪这个 twist 后离 goal 尽量近，(2) 但同时 V̂ 在 threshold 之下（safe）。

d_goal^future 用 linearized integral 近似 (Eq 22)：
δx = v_x^c · δt - 0.5 · v_y^c · ω_z^c · δt²
δy = v_y^c · δt + 0.5 · v_x^c · ω_z^c · δt²

δt = 0.05 s。这考虑了 twist 在 base frame 下的旋转效应（科里奥利项）。

实际求解：gradient descent + Lagrangian multiplier，从当前 twist 初始化，5 步内收敛，能 real-time 部署。

Figure 8c 可视化了 recovery 触发时 V̂ 在 (v_x, ω_z) 和 (v_x, v_y) 平面上的 landscape，可以看出 searched twist 确实在 safe region (V̂ < V_threshold) 内。

直觉：V̂ 不仅作为 binary switch，还通过 **gradient information** 引导 recovery policy。这是 ABS 区别于 Recovery RL [66] 的关键 —— Recovery RL 的 safety critic 和 backup policy 没有 interplay，假设 backup policy 能 restore safety 但没 explicit optimization 满足 critic。ABS 用 V̂ 的 gradient 直接 shape recovery action。

---

## V. Recovery Policy: twist-tracking

### A. 设计哲学

Recovery policy 是一个**纯 twist-tracking** policy：给定 (v_x^c, v_y^c, ω_z^c)，输出 joint action 跟踪这个 twist。不需要 exteroception（因为 safe twist 已经被 RA value 优化好了）。

这和 agile policy 的 goal-reaching formulation 形成对比：recovery 是 "low-level tracking controller"，agile 是 "high-level navigation + locomotion"。

为什么这样分？因为 recovery 需要极 fast 响应（high-speed 运动中突然触发），twist-tracking 是 well-posed 问题（有 ground truth velocity），容易 train 出 robust tracking policy。而 collision avoidance 的逻辑被外置到 RA value optimization 里，recovery policy 只管 "execute twist faithfully"。

### B. Training

Observation: o^Rec = [c_f; ω; g; tw^c (3-d); q; q̇; a_{t-1}]

Reward (Eq 23):
r_task = 10·r_linvel - 0.5·r_angvel + 5·r_alive - 0.1·r_posture

- r_linvel = exp(-((v_x - v_x^c)² + (v_y - v_y^c)²) / σ_linvel²), σ_linvel = 0.5 m/s（Gaussian-like，near command 时 gradient 大）
- r_angvel = ||ω_z - ω_z^c||² (quadratic，比 Gaussian 更 soft)
- r_alive = 1·1(alive)
- r_posture = ||q - q̄_rec||₁，q̄_rec 是低姿态 nominal pose，方便切回 agile policy

Domain randomization 的关键变化：
- Episode length: 2s（短，因为 recovery 是 short burst）
- Initial roll/pitch: U(-π/6, π/6)（recovery 可能从 tilted 状态触发）
- Initial v_x: U(-0.5, 5.5) m/s（high-speed 触发，所以 initial velocity 大）
- Initial ω: U(-1.0, 1.0) rad/s
- Command ranges: v_x^c ~ U(-1.5, 1.5), v_y^c ~ U(-0.3, 0.3), ω_z^c ~ U(-3.0, 3.0)
- **允许 knee contact**（紧急刹车时膝盖着地，Figure 1a）

Curriculum: tracking error < 0.7·σ_linvel 就 promote，摔倒就 demote。

Training 效率：~500 iterations 收敛，10 分钟训完。比 agile policy (4 小时) 快很多，因为 task 简单。

---

## VI. Ray-Prediction Network: low-dimensional exteroception

### A. 为什么用 ray 而不是 raw depth

11 个 ray，角度均匀分布在 [-π/4, π/4]（front 90° FOV）。这个设计选择有几个 reason：

1. **Decoupling**: depth image 的 noise/光照/distribution shift 全部由 ray-prediction network 处理，agile policy 和 RA value network 只看 clean 11-d ray，training 容易
2. **Interpretability**: 人类可以 supervise（看 ray 长度对不对）
3. **Compute**: ray 在 sim 里 cheap（解析 ray tracing for cylinder），不用 render depth image
4. **Generalization**: low-dim input 让 policy 更容易 generalize 到不同 obstacle shape

注意用 log(ray distance) 作为 observation，因为近距离 obstacle 信息更重要（log 把近距离分辨率放大）。

### B. Sim-to-real 的 augmentation

Real-world depth image 远比 sim 里 rendered 的 noisy (Hoeller et al. [28], https://arxiv.org/abs/2104.08456)。ABS 用 4 种 augmentation (Figure 6)：
1. Horizontal flip
2. Random erase
3. Gaussian blur
4. Gaussian noise

部署时还用 hole filling (Stereolabs ZED SDK, https://www.stereolabs.com/docs/depth-sensing/depth-settings) 进一步减小 sim-real gap。

### C. Network choice

Table VIII 的 ablation：

| Architecture | Test MSE | Inference (ms) |
|---|---|---|
| EfficientNet-B0* | 3.627e-2 | 19 |
| MobileNet-V2* | 3.387e-2 | 15 |
| ResNet-34 | 3.081e-2 | 14 |
| ResNet-18 | 3.238e-2 | 9 |
| ResNet-18 (no pretrain) | 3.526e-2 | 9 |
| ResNet-18 (no augment) | 3.393e-2 | 9 |

选 ResNet-18 + pretrained + augmentation，平衡 accuracy 和 inference time (9ms on Jetson Orin NX)。Pretrain 和 augmentation 都显著降 MSE。

Input/output: log(depth) → log(ray distance)，MSE loss。Resolution 160×90。

---

## VII. 实验：打破 agility-safety boundary

### A. Simulation benchmark (Table III, Figure 7)

关键 insight 来自 Figure 7 的 agility-safety trade-off 图：
- π^Agile 系列（3 个 variant）和 LAG 系列（3 个 variant）形成一条 **trade-off boundary**
- ABS 系列**显著 break 这条 boundary**：同样 safety 下 speed 更高，同样 speed 下 safety 更高

具体数字：
- ABS-n: 79.1% success, 5.7% collision, v_avg=2.08 m/s
- π^Agile-n: 77.3% success, 21.7% collision, v_avg=2.39 m/s
- LAG-n: 77.4% success, 9.1% collision, v_avg=1.41 m/s

ABS 相比 π^Agile：collision rate 砍掉 4x，只损失 0.3 m/s avg speed。  
ABS 相比 LAG：collision rate 砍掉 1.6x，speed 提升 1.5x。

### B. Real-world experiments (Figure 9)

3 个 testbed：
- Indoor (a): dim narrow corridor — ABS 9/10 success, 1 collision (low light 导致 ray prediction 失败)
- Indoor (b): hall with furniture — ABS 10/10 success
- Outdoor: open space — ABS 10/10 success, peak 3.1 m/s

LAG baseline peak speed 只有 2.1 m/s，比 ABS 的 3.1 m/s 差很多。

### C. Robustness (Figure 10)

- Snowy terrain: OK
- 12-kg payload (等于 robot 自重): OK
- 球击打 running robot: OK
- 踢 standing robot: OK

这些 robustness 主要来自 domain randomization（add base mass, friction, ERFI-50）+ recovery policy 的 shielding。

### D. Instant steering (Figure 12, Table IX)

Goal-reaching formulation 的一个 bonus：可以 runtime 改 goal command 实现 instant steering。Table IX 列了几个 goal command preset：
- Forward: (5, 0, 0)
- Rapid Right Turn: (-2, 0, π) (倒过来跑！)
- Left/Right Turn: (2, ±1.5, ±2π/3)

Figure 12 展示了 "Forward → Rapid Right Turn → Forward" 的 sequence，robot 能 >3 m/s 前进，>6 rad/s 急转。这在 velocity-tracking formulation 里是 out-of-distribution（rapid 跑时改 velocity command），但 goal-reaching 是 in-distribution（goal 一直在变）。

---

## VIII. Limitations 和 Future Work

作者自己列了 5 个 limitation，build intuition 很重要：

1. **Local minimum in dense obstacles**: 太密的 obstacle 形成 local minimum，policy 容易 timeout。需要 memory ([74], https://openreview.net/forum?id=lTt4KjHSsyl) 或 global hint ([70], https://arxiv.org/abs/2305.01098)。

2. **Dynamic obstacle generalization**: RA value 用 static obstacle 学的，只能 generalize 到 quasi-static。如果 dynamic obstacle 比 recovery policy velocity limit 还快，会撞。需要 motion prediction ([40], [59])。

3. **2D only**: 没有 flying phase，不能处理 stairs/gaps。3D terrain 下 locomotion 和 collision avoidance 耦合更复杂。

4. **No implicit system identification**: [39, 48, 50] 用 temporal latent embedding model real-world dynamics，但 ABS 的 policy switch 会让 embedding out-of-distribution，难集成。

5. **Vision system**: Indoor (a) 的 collision 就是 ray-prediction 在暗光下失败。需要更多 camera（around body）或 event camera ([19], https://arxiv.org/abs/2104.02236)。

---

## IX. 几个值得思考的 deep points

### 1. RA value 的 under-approximation 是 feature 不是 bug

V_RA^π 是 V_{RA*}^π 的 under-approximation，意味着 RA value 偏 conservative（标更多 state 为 unsafe）。在 safety-critical 场景这是 desirable：宁可多 trigger recovery，也不要 false negative。Table V 的 V_threshold sweep（-0.001 到 -0.1）显示 performance 对 threshold 不敏感，说明 RA value landscape 在 threshold 附近 well-behaved。

### 2. Policy-conditioned vs policy-agnostic 的 trade-off

ABS 选 policy-conditioned 简化学习，但代价是 RA value 只对 π^Agile valid。如果 agile policy 改了，RA value 要重训。这对 production 系统是 OK 的（agile policy 训完就 frozen），但对 continual learning 不友好。

### 3. Two-stage offline learning 的稳定性

Hsu et al. [30] 是 online train RA value during RL，有 stability issue。ABS 的 two-stage（先 train agile，再 collect rollout train RA value）解耦，更稳定。这类似 DAgger vs behavior cloning 的 trade-off。

### 4. Recovery policy 不需要 exteroception 的设计

Recovery policy 只 track twist，不看 ray。这是因为 safe twist 已经被 RA value optimization 选好了（V̂ < V_threshold）。这是分层抽象的体现：collision avoidance 逻辑在 RA value 层，recovery policy 只管 motor execution。这让 recovery policy 简单、fast、robust。

### 5. Gait emergence: gallop vs trot

Goal-reaching 学到 gallop，velocity-tracking 学到 trot。这个 emergence 现象很有意思。直觉：goal-reaching 让 policy 自由 explore gait space，gallop 是 high-speed 下更稳定的 gait（1 DoF uncontrollable vs 3 DoF）。Velocity-tracking 的 "tracking velocity" objective 间接 bias 向 trot（trot 是 velocity tracking error 最小的 gait？需要更多实验验证）。

类似的现象在 RL locomotion 里不少见：[82] 的 risky terrain 学到不同 gait，[11] 的 extreme parkour 学到 jumping。

---

## X. 与相关工作的定位

- **vs Rapid locomotion [48]** (https://arxiv.org/abs/2205.02824): ABS 用 goal-reaching，rapid 用 velocity-tracking。ABS 加了 collision avoidance 和 safety shield。
- **vs ANYmal Parkour [29]** (https://arxiv.org/abs/2306.14874): Parkour 是 terrain traversal，ABS 是 cluttered environment navigation。Parkour 没有 dynamic obstacle。
- **vs Recovery RL [66]** (https://arxiv.org/abs/2010.04213): Recovery RL 的 safety critic 和 backup policy 没 interplay，ABS 用 RA value gradient 引导 recovery。
- **vs HJ reachability [3]** (https://arxiv.org/abs/1709.07523): 经典 HJ 是 model-based，ABS 是 model-free。
- **vs Hsu et al. [30]** (https://robotsconference.org/2021/program/): Hsu 学 policy-agnostic global RA value，ABS 学 policy-conditioned。
- **vs CBF shield [10, 84]** (https://arxiv.org/abs/1903.06720): CBF 需要 analytical dynamics model，ABS 全 learned。
- **vs PPO-Lagrangian [53]** (https://openai.com/research/benchmarking-safe-exploration-in-deep-reinforcement-learning): LAG 是 end-to-end safe RL，受 agility-safety trade-off 限制。ABS 用 hierarchical shielding 打破 trade-off。

---

## XI. 总结：ABS 的 design philosophy

ABS 的核心 design philosophy 是 **"分离 conquer"**：
- **Agility** 由 agile policy 全权负责，不受 safety constraint 拖累
- **Safety** 由 RA value + recovery policy 负责，作为 external shield
- **Perception** 由 ray-prediction network 负责，decouple high-dim image noise from policy learning
- **Switching** 由 RA value threshold 决定，control-theoretic 保证

这种 modular 设计让每个 component 都能 independently optimize，最终 system 性能超过 end-to-end safe RL。

直觉上，这类似人类跑步：fast running 时大脑主要关注 navigation 和 motor control（agile policy），遇到危险时 reflex system 接管做紧急避险（recovery policy），perception system 持续提供 low-dim obstacle 信息（ray prediction）。这种 biological 分层对应让 ABS 在 sim-to-real 和 robustness 上表现好。

---

## XII. 你可能感兴趣的延伸方向

1. **3D ABS**: 加 vertical locomotion (jump, climb stairs)。需要 3D reach-avoid，可能用 DeepReach [2] (https://arxiv.org/abs/2011.12758) 思路。
2. **Multi-robot ABS**: swarm 场景下，每个 robot 是 dynamic obstacle for others。需要 game-theoretic reachability。
3. **ABS + LLM**: 用 LLM 做 high-level goal generation (e.g. "去厨房拿水")，ABS 做 low-level execution。
4. **Continuous recovery**: 现在 recovery 是 discrete switch，可以做成 soft blend (e.g. differentiable policy mixture)。
5. **Event camera ABS**: 替换 ZED depth 用 event camera，high-speed dynamic obstacle 场景更好 (Falanga et al. [19], https://arxiv.org/abs/2104.02236)。
6. **Imitation learning ABS**: 从动物 (e.g. 猎豹) 跑步数据 distill agile policy，可能学到更好的 gait。

希望这个详细讲解 build 了你关于 ABS 的 intuition。这 paper 的核心 contribution 是把 control-theoretic reach-avoid 和 model-free RL 巧妙结合，用 policy-conditioned RA value + recovery policy guidance 打破 agility-safety trade-off。如果对某个 module 想深入讨论，可以继续。
