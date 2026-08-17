---
source_pdf: HumanoidBenchSimulatedHumanoidBenchmark forWhole-BodyLocomotionandManipulation.pdf
paper_sha256: 000dd3c95bcc8697d2e6e2425b60478dfb0c42ff3ccc82fabe63dd6e1cd55b03
processed_at: '2026-08-05T08:12:57-07:00'
target_folder: Robot-VLA/Dataset
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

好的，Andrej，咱们用最直白的话来聊聊 HumanoidBench 这篇 paper 到底在说什么，顺便把里面的技术细节掰碎了揉烂了讲清楚。

这篇 paper 核心讲了一件非常扎心的事情：**当我们给一个 humanoid 机器人装上两只灵巧手，并让它用 end-to-end RL 去做任务时，当前所有的 SOTA 算法全都崩溃了。**

表面上看，这是机器人 benchmark 的扩充。本质上，这篇 paper 像一面镜子，照出了目前 RL 算法在 high-dimensional action space 下的极端无力感。

---

### 1. 为什么加两只手，机器人就变傻了？

想象一下你让一个人去学走路。如果只让他用双腿走，他很快就能学会。但如果在学走路的同时，要求他的十个手指头必须同时做极其复杂的弹钢琴动作，这个人大概率连路都不会走了。

HumanoidBench 的核心实验就是这个意思。

Unitree H1 这个机器人，身体本身只有 19 维 action space。RL 算法在 19 维空间里学走路（`walk` task），虽然慢，但能学会。
当作者把两只 Shadow Hand 装上去后，action space 从 19 维直接飙升到了 **61 维**。每只手 21 维，身体 19 维。结果在 `walk` 任务上，所有 SOTA RL 算法的 learning curve 直接崩塌。

为了搞清楚到底是因为手的物理重量改变了 dynamics，还是因为 action 维度太高导致了 exploration 瓶颈，作者做了一个极其精彩的 ablation：

他们保留了两只手的物理模型（质量、惯量、observation 都在），但**把手的 action 强制设为 0**，也就是手固定不动，只让 RL 去学 19 维的身体控制。

实验数据（Table V 和 Figure 7）证明，一旦把手的 action dim 冻结，RL 算法立刻就学会了走路，性能和完全没有手的时候一模一样。

**Intuition 构建**：
这里面的数学根源在于 RL 的 exploration 是基于随机采样的。在 continuous action space 里，假设 action 的范围是 $[-1, 1]^D$。
探索体积的公式是：
$$ V = \int_{-1}^{1} \dots \int_{-1}^{1} dx_1 \dots dx_D = 2^D $$
这里的 $V$ 代表 state-action space 的总体积，$D$ 代表 action space 的维度（比如 19 或 61）。
当 $D=19$ 时，$V \approx 5.24 \times 10^5$。
当 $D=61$ 时，$V \approx 2.3 \times 10^{18}$。

RL 算法（如 SAC 或 TD-MPC2）靠 Gaussian 分布去 random shoot 动作。当维度从 19 变成 61，有效探索到“正确动作组合”的概率呈指数级下降。RL agent 无法学会“在走路的时候忽略手指头的 action”，它会试图去同时 optimize 所有的 dimension，结果就是全都学不好。这直接揭示了 RL 在高维空间的 **Curse of Dimensionality**。

参考链接：[TD-MPC2 Paper](https://arxiv.org/abs/2310.16828) | [SAC Paper](https://arxiv.org/abs/1801.01290)

---

### 2. HRL（Hierarchical RL）是怎么破局的？

既然 end-to-end flat RL 栽在了 61 维 action space 上，作者就引入了 HRL 的思路来降维打击。

HRL 的核心思想是：**把大脑和小脑分开**。
小脑负责低级的运动控制，大脑负责高级的任务规划。

在 HumanoidBench 的 HRL 实现中：
- **Low-level policy（小脑）**：只负责“把手伸到空间中的某个 3D 坐标点”。它的 action space 仍然是 61 维，但它的任务极其简单、明确。
- **High-level policy（大脑）**：任务变成了“决定把手伸向哪个 3D 坐标点”。它的 action space 仅仅只有 **3 维**（x, y, z 坐标）。

这就在数学上把原来的 MDP（Markov Decision Process）转换成了 Semi-MDP：
$$ M' = (\mathcal{S}, \mathcal{O}, P', R', \gamma') $$
公式里的 $\mathcal{S}$ 代表 state space（保持不变），$\mathcal{O}$ 代表 option space（高层策略的动作空间，维度极低），$P'$ 是 option 执行后的 state 转移概率，$R'$ 是高层策略收到的奖励，$\gamma'$ 是高层的 discount factor。

高层策略向 low-level policy 发出一个 3D 目标 $o_t = (x, y, z)$，low-level policy 就接管这接下来的 $k$ 步，去努力达到这个点。高层策略只在 $k$ 步之后才做下一次决策。这种时间上的抽象极大地简化了 long-horizon planning。

**Low-level 的训练细节**：
作者为了训练一个足够鲁棒的 low-level reaching policy，动用了 [MuJoCo MJX](https://mujoco.readthedocs.io/en/latest/mjx.html) 和 [PureJaxRL](https://arxiv.org/abs/2210.05639)。他们在 GPU 上并行跑了 **32,768** 个环境，用 PPO 跑了 2 billion steps（单手）和 4 billion steps（双手）。
而且他们在训练时，给机器人的每个 link 都施加了随机的 force perturbation（外力扰动），以此保证这个 reaching policy 在面对各种意外时都不会崩溃。这就是典型的 domain randomization 思路，为 sim-to-real 打基础。

**High-level 的训练结果**：
在 `push` 这个推箱子的任务上，flat DreamerV3 跑出了 -1252 的惨烈成绩。而加上 HRL 后，高层 DreamerV3 的成绩直接飙升到 **1000**（满分），成功率几乎 100%。

Intuition 就在于：高层大脑只需要思考“手往哪里伸”，箱子就会因为手伸过去而被推开。61 维的繁琐细节被 low-level 黑盒封装了。探索 3 维空间比探索 61 维空间容易了无数个数量级。

参考链接：[Options Framework Paper](https://www.sciencedirect.com/science/article/pii/S000437029900052X) | [PureJaxRL](https://arxiv.org/abs/2210.05639)

---

### 3. 奖励函数设计里的魔鬼细节

Andrej，你肯定对 reward hacking 极其敏感。HumanoidBench 里的任务设计暴露了大量 RL 在 reward 引导下的 local optimum 问题。

拿 `highbar`（单杠）任务举例。
目标是让机器人挂在单杠上，然后荡上去，直到身体完全倒立。
奖励函数这样设计：
$$ R(s, a) = \text{upright}_{\text{highbar}} \times \text{feet} \times e $$
其中 $\text{upright}_{\text{highbar}} = \text{tol}(z_{\text{proj}}, (-\infty, -0.9), 1.9)$，代表机器人躯干要朝下（倒立时的姿态），$z_{\text{proj}}$ 是机器人 z 轴在世界坐标系投影。$\text{feet} = \text{tol}((z_{\text{foot,left}} + z_{\text{foot,right}})/2, (4.8, +\infty), 2)$，要求脚要荡得极高。$e$ 是能量惩罚项。

按理说，要拿到高 reward，必须荡上去。但实验结果发现，RL agent 学到了一个极其保守的策略：死死抓住单杠不松手，身体微微晃动。因为一旦尝试大幅度摆荡，失败的概率极高，会导致 episode 提前终止（跌落），连基础的稳定 reward 都拿不到。Agent 发现“什么都不做，死死挂着”是短期 reward 最安全的选择。

这就是 short-horizon planning 的致命伤。RL 算法极度贪婪，在没有 curriculum 或者 intrinsic motivation 引导的情况下，它会被困在这个 local optimum 里永远出不来。

再看 `door` 任务。
目标是拉开门把手并穿过门。
奖励包含了打开门把手（$open_{\text{door}}$）、靠近门（$proximity_{\text{door}}$）和穿过门（$passage$）。
Agent 轻松学会了开门把手（拿到了 dense reward），但接下来它卡住了。因为要把门拉开并穿过去，机器人需要一边拉门把手，一边让整个 body 往后退。这种 locomotion（腿部后退）和 manipulation（手拉门）的极度紧密的 coordination，传统 RL 根本拼凑不出来。结果就是机器人站在原地死死拽着门把手，身体一动不动。

参考链接：[DeepMimic](https://dl.acm.org/doi/10.1145/3197517.3201925) | [Relay Policy Learning](https://arxiv.org/abs/1910.13728)

---

### 4. 仿真引擎的工程瓶颈

在跟你聊这个 paper 时，我还特别关注了它背后的工程实现。

MuJoCo 在处理 humanoid 加上 Shadow Hand 这种 75 DoF、全身布满 collision mesh 的复杂系统时，性能下降非常严重。
默认配置只能跑 **1050 FPS**（Table IV）。为了训练 low-level reaching policy 时能跑 32,768 个并行环境，作者不得不对 MJX 环境做极大简化：只保留脚和地面的 collision，把手的 collision 全去掉了，跑到了 **5100 FPS**。

这就引出了一个非常有意思的悖论：
如果要训练泛化能力强的 policy，就需要高保真的 collision detection（比如全身触觉），但 FPS 就会暴跌。
如果要利用 JAX 的大规模并行，就必须把 collision mesh 简化，但这又失去了全身 contact-rich 的物理真实性。

作者为了实现全身触觉，用了 [CoACD (Approximate Convex Decomposition)](https://arxiv.org/abs/2105.02955) 把原本非凸的 mesh 切碎成成百上千个小凸包。这导致 MuJoCo 在每一步都要计算海量小凸包之间的碰撞，计算量呈指数级上升。最终带触觉的版本只能跑 550 FPS。

这种工程上的瓶颈直接限制了算法的探索速度。DreamerV3 跑 10M steps 需要 48 小时。如果 FPS 只有 500，你根本没法在可接受的时间内做完实验。

参考链接：[CoACD Paper](https://arxiv.org/abs/2105.02955) | [MuJoCo Documentation](https://mujoco.readthedocs.io/)

---

### 5. 从 Foundation Model 角度的联想

把 HumanoidBench 放在当前大模型时代的背景下，它的失败结果非常发人深省。

为什么 LLM 能 scale 起来？因为 next-token prediction 提供了一个极其稠密且全宇宙无处不在的监督信号。互联网上的每一句废话都可以用来训练。
为什么 Humanoid 上的 RL scale 不起来？因为机器人跟物理世界交互的每一个 step 都要付出巨大的计算代价和时间代价，而且 reward 信号极度稀疏。

这篇 paper 等于是在向全社区喊话：单纯指望靠 SAC、PPO 这种 random exploration 的 RL 算法，在 61 维 action space 下硬刚 humanoid 任务，死路一条。

未来的解法大概率在于 **Behavior Priors** 或者 **Pretrained World Models**。
如果只靠 agent 在环境里随机瞎试，它永远发现不了“把箱子推到目标点”这个动作序列。但如果我们在仿真里预先用大量的 human motion capture 数据（比如 [DeepMimic](https://dl.acm.org/doi/10.1145/3197517.3201925) 或者 [AMP](https://dl.acm.org/doi/10.1145/3450626.3459975) 的思路）训练一个极其强大的 low-level motor prior，然后再让 high-level policy 去调用它，这可能是唯一的出路。

更进一步，如果我们能把 LLM 的 commonsense reasoning 接入进来，比如让 LLM 告诉 humanoid“推箱子之前你得先走到箱子旁边”，类似于 [RT-2](https://robotics-transformer2.github.io/) 或者 [Eureka](https://arxiv.org/abs/2310.12931) 的思路，那 high-level 的 planning 也就解决了。

HumanoidBench 搭好了台子，提出了极难的 61 维 action space 的 challenge，接下来就是看谁家的 pretraining model 或者 hierarchical structure 能在这个台子上拿满分了。

希望这些细节能帮你 build 起对这个 benchmark 的 intuition！如果我们要在上面跑新算法，第一件事就是把 action space 的 curse of dimensionality 给 break 掉。

---

# HumanoidBench: 深度技术解析

Andrej，这篇paper是UC Berkeley团队（Carmelo Sferrazza, Pieter Abbeel组）2024年发布的**首个综合性humanoid benchmark**，覆盖locomotion + whole-body manipulation。它揭示了一个非常重要的事实：**当前的SOTA RL算法在61维action space的humanoid上基本都失败了**，即使像walk这种DMC里早被解决的问题，加上灵巧手后都变得极难学。

让我深入讲讲，尽量build你的intuition。

---

## 1. 核心动机与定位

### 1.1 为什么需要这个benchmark

现有humanoid research的两个瓶颈：
- **硬件瓶颈**：Boston Dynamics Atlas, Tesla Optimus, Unitree H1等真机昂贵、易损、不安全
- **算法瓶颈**：现有controller多是hand-designed for specific task（参考 [Kuindersma et al. 2016](https://link.springer.com/article/10.1007/s10514-015-9476-3) Atlas的trajectory optimization）

而现有simulated benchmark的局限：
- Manipulation类（[robosuite](https://arxiv.org/abs/2009.12293), [MetaWorld](https://arxiv.org/abs/1910.10897), [RLBench](https://arxiv.org/abs/1909.12271)）多用单臂+parallel gripper，缺乏whole-body coordination
- Locomotion类（[DMControl](https://arxiv.org/abs/1801.00690), [LocoMujoco](https://arxiv.org/abs/2311.12472)）只关注移动
- Dexterous hand类（[Adroit](https://arxiv.org/abs/1802.09464), [MyoSuite](https://arxiv.org/abs/2205.13600), [Bi-DexHands](https://arxiv.org/abs/2307.04105)）是floating hand，脱离arm base

**HumanoidBench的独特之处**：第一次把high-dimensional humanoid body + dexterous hand + diverse long-horizon tasks合在一起，参见Table I。

### 1.2 Table I 关键对比

| Benchmark | Dexterous hands | Action dim | DoF | Task horizon | # Tasks |
|-----------|-----------------|------------|-----|--------------|---------|
| MyoHand | ✓ | 39 | 23D | 50-2000 | 9 |
| Adroit | ✓ | 24 | 24D | 200 | 4 |
| LocoMujoco (H1) | ✗ | 19 | 6D | 100-500 | 27 |
| DMControl (Humanoid) | ✗ | 24-56 | 22D | 1000 | 6 |
| robosuite | ✗ | 6-24 | 6-7D | 500 | 9 |
| MetaWorld | ✗ | 6 | 7D | 500 | 50 |
| **HumanoidBench** | ✓ | **61** | **75D** | 500-1000 | 27 |

注意：action dim 61 = 19(body) + 21×2(Shadow Hand)，DoF 75 = 25(body) + 25×2(hand)。这比之前任何benchmark的action space都大3-10倍，是核心challenge。

---

## 2. 模拟环境设计

### 2.1 机器人配置

主配置：**Unitree H1 + 双Shadow Hand**

为什么选H1而不是Digit或G1：
- H1是full-size humanoid，比G1大
- Digit有passive joints（通过four-bar linkage驱动），学习速度慢，作者归因于mechanical complexity
- H1的MJCF模型由Unitree官方提供，仿真准确

为什么Shadow Hand：
- 自由模型文件（来自[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)）
- 在[OpenAI Rubik's Cube](https://arxiv.org/abs/1910.07113)和[RoboPianist](https://arxiv.org/abs/2305.05933)中已被验证
- 21维action per hand，24 DoF per hand（underactuated）

**关键设计选择**：移除了Shadow Hand原本cumbersome的forearm，让hand更human-like。作者明确指出这与Tesla Optimus、Figure 01的下一代slim hand趋势一致。这是个**前瞻性design choice**：当前硬件做不到，但下一代会做到。

### 2.2 Observation Space

| 配置 | Observation | Action | DoF (body) | DoF (hands) |
|------|-------------|--------|------------|-------------|
| H1 w/o hands | 51 | 19 | 25 | 0 |
| H1 w/ ShadowHand | 151 | 61 | 25 | 50 |
| H1 w/ Robotiq | 55 | 23 | 25 | 4 |
| H1 w/ Unitree hand | 103 | 45 | 25 | 26 |
| Digit w/ ShadowHand | 221 | 65 | 57 | 50 |
| Unitree G1 | 87 | 37 | 29 | 14 |

注意：observation dimension比DoF + DoF（velocity）的总和稍多，因为**floating base用quaternion**（4维）表示orientation，而angular velocity只3维；ball joint同理。这是仿真中常见的细节。

**Observation类型**：
- Proprioceptive state（joint pos/vel）+ task-relevant env state（object pose/vel）
- Egocentric vision（两个head camera）
- Whole-body tactile（MuJoCo touch grid，448 taxels，每个提供3D force）

### 2.3 Tactile Sensing实现

这是个非trivial的工程：
- 用[CoACD (Approximate Convex Decomposition)](https://arxiv.org/abs/2105.02955)对原始mesh进行凸分解
- 把每个body part的mesh细分成多个小的convex mesh
- 每个convex mesh单独参与collision detection，增加contact point candidates
- MuJoCo touch grid把contact forces聚合到discretized bins

最终：448个taxel分布在全身，hand上高分辨率，其他部位低分辨率（类似人类）。性能：refined mesh + tactile运行在550 FPS。

### 2.4 Action Space

- **Position control**（默认）：61维target joint position，normalized to [-1, 1]^61
- 也支持torque control，但position control更稳定，允许更低control frequency
- Control frequency: **50 Hz**

---

## 3. 任务设计

### 3.1 Locomotion Tasks (12个)

让我讲几个有意思的：

**walk**: 保持forward velocity v_x ≈ 1 m/s不摔倒
$$R(s,a) = \text{stable} \times \text{tol}(v_x, (1, +\infty), 1)$$

其中stable = stand × e, stand = height × upright, e = control effort penalty, tol是DMC的tolerance function：
- $\text{tol}(x, (x_{lower}, x_{upper}), m)$: x在范围内返回1，否则在margin m内线性衰减到0

**balance**: 站在不稳定board上
$$R(s,a) = (e \times still) \times (height_{robot} \times upright)$$
有simple/hard两个版本，hard版本sphere pivot会移动。

**pole**: 在密集thin poles间穿行
$$R(s,a) = \gamma_{collision} \times (0.5 \cdot stable + 0.5 \cdot \text{tol}(v_x, (1, +\infty), 1))$$
$\gamma_{collision}$: 碰撞时0.1，不碰撞时1。

### 3.2 Manipulation Tasks (15个)

**push**: 把box推到3D target point
$$R(s,a) = \alpha_s \cdot success - \alpha_t \cdot d_{goal} - \alpha_h \cdot d_{hand}$$
$\alpha_s = 1000$, $\alpha_t = 1$, $\alpha_h = 0.1$。dense + sparse的混合。

**cabinet**: 4个subtask（sliding door, drawer, hinge cabinet transfer, pull-up cabinet），完成subtask i给 i×100 sparse reward，全完成给1000。这是典型的long-horizon task。

**basketball**: 两阶段（catch + throw）
$$R_{catch} = 0.5 \cdot proximity_{hand} + 0.5 \cdot stable$$
$$R_{throw} = 0.05 \cdot proximity_{hand} + 0.15 \cdot stable + 0.8 \cdot aim$$
aim = tol(d(ball, basket), (0,0), 7)。成功给1000 sparse reward。

**cube**: 双手各持一cube，让两个cube都达到random target orientation
$$R = 0.2 \cdot (stable \times still) + 0.5 \cdot orientation + 0.3 \cdot proximity_{cube}$$
orientation = 0.5 × [（quat_{cube,left} - quat_{target}）² + （quat_{cube,right} - quat_{target}）²]

**spoon**: 用spoon在pot里画圆轨迹
- destination随时间变化：x = x_pot + 0.06cos(tπ/20), y = y_pot + 0.06sin(tπ/20)
- 这是一个**trajectory tracking**任务，需要hand-eye coordination。

**kitchen**: 唯一**纯sparse reward**任务（max 4），最大return = 4。来自[Relay Policy Learning](https://arxiv.org/abs/1910.13728)环境，4个subtask：open microwave, move kettle, turn burner/light switch。结果是：所有算法return = 0（完全失败）。

---

## 4. Benchmarking Results

### 4.1 Baselines

| Algorithm | Type | Steps (48h) |
|-----------|------|------------|
| DreamerV3 | Model-based, world model | 10M |
| TD-MPC2 | Model-based, online planning | 2M |
| SAC | Off-policy, model-free | 10M |
| PPO | On-policy, model-free | 部分task |

PPO只在walk/kitchen/door/package上跑，因为没用大规模并行化时性能太差。

### 4.2 关键发现

**Finding 1: 所有SOTA RL在大多数任务上都fail**

看Table V（average return@10M）和Table VI（max return@10M）vs Target：

| Task | DreamerV3 avg | Target | 是否成功 |
|------|---------------|--------|----------|
| walk | 800 | 700 | ✓ |
| stand | 623 | 800 | 接近 |
| run | 634 | 700 | 接近 |
| reach | 7581 | 12000 | 失败 |
| hurdle | 126 | 700 | 失败 |
| stair | 131 | 700 | 失败 |
| push | **-1252** | 700 | 大失败 |
| cabinet | 57 | 2500 | 失败 |
| highbar | 9 | 750 | 完全失败 |
| basketball | 19 | 1200 | 完全失败 |
| kitchen | 0 | 4 | 完全失败 |
| package | **-18015** | 1500 | 大失败 |

注意package的DreamerV3竟然是-18015，意味着robot一直在主动做错事（很可能因为reward有-3·d(package, destination)项，robot学不会移动package，每步都在penalty）。

**Finding 2: With hands vs Without hands**

这是paper最有意思的ablation。在walk任务上：
- With hands (61D action): 性能大幅下降
- Without hands (19D action): 性能大幅提升

**Reduced action space ablation**：保留hands（保留mass和observation 151D），但**固定hand actuation为0**（action从61D降到19D）。结果显示性能几乎和without hands一样好。

**Intuition**：问题不是hands的物理存在（mass, dynamics），而是**action dimensionality itself**。RL算法无法ignore unused action dimensions，多余的自由度让exploration变得exponentially harder。这印证了[Curse of Dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)在RL exploration中的核心地位。

**Finding 3: Common Failures**

**highbar**: robot学会conservatively保持contact（避免episode termination），但**不learn whole-body rotation**。这是经典的**local optimum + short horizon planning**问题，dense reward反而助长了conservative behavior。

**door**: robot学会unlock door hatch（easy dense reward部分），但**不会pull door + 后退whole body**。这需要locomotion + manipulation的coordination，是当前RL算法的盲点。

**hurdle**: robot学会run forward（拿到velocity reward），但**不会jump**。它会conservative地撞hurdle以保持stability。对比[OpenAI Gym Walker2d](https://gym.openai.com/envs/Walker2D-v2/)，那里forward reward就足够学到jumping——humanoid因为更复杂dynamics，奖励hacking更容易发生。

---

## 5. Hierarchical RL Solution

### 5.1 Motivation

Flat end-to-end RL失败的根因：
1. High-dimensional action space让exploration几乎不可行
2. Long-horizon tasks需要multi-modal skills
3. 不同subtask需要完全不同的behavior

HRL的核心idea：**decouple low-level motor control from high-level planning**。

### 5.2 Architecture

**Low-level: Reaching Policy**
- 用[PureJaxRL](https://arxiv.org/abs/2210.05639)的PPO在[MuJoCo MJX](https://mujoco.readthedocs.io/en/latest/mjx.html)上训练
- 32,768 parallel environments（GPU大规模并行）
- 简化模型：只保留feet-ground collision（其他collision mesh去掉），移除hands
- Force perturbations on each link during training（domain randomization for robustness）
- One-hand reaching: 2B steps, 36h
- Two-hand reaching: 4B steps, 60h
- Target一旦reached就reset

**High-level: Task Policy**
- Low-level frozen，作为fixed motor primitive
- High-level outputs reaching targets（3D or 6D），作为它的"action space"
- 用DreamerV3或TD-MPC2训练
- 限制reaching target range到robot workspace内以facilitate exploration

### 5.3 Results

**push任务**：hierarchical DreamerV3达到1000/1000（success rate ~100%），而flat DreamerV3是-1252。**质的飞跃**。

**package任务**：hierarchical有改进但远未解决。Policy可以接近package但**无法lift它**——因为high-level从未在training中experience过lifting状态。

**Intuition**：HRL把61D action problem降到3-6D high-level action problem，exploration difficulty从$O(c^{61})$降到$O(c^6)$，加上low-level robust motor skill，效果立竿见影。但low-level skill的scope有限：reaching policy不知道关于object的manipulation，所以package任务需要更多primitive（grasping, lifting）。

---

## 6. 与相关工作的连接

### 6.1 Humanoid Locomotion
- [Expressive Whole-Body Control (Cheng et al. 2024)](https://arxiv.org/abs/2402.16796): 上一代humanoid upper body motion + locomotion，用phase-conditioned policy
- [Learning Humanoid Locomotion with Transformers (Radosavovic et al. 2023)](https://arxiv.org/abs/2303.03381): H1 locomotion with Transformer policy
- [Robot Parkour Learning (Zhuang et al. 2023)](https://arxiv.org/abs/2309.05665): quadruped parkour with RL + teacher-student

### 6.2 Dexterous Manipulation
- [OpenAI Rubik's Cube](https://arxiv.org/abs/1910.07113): Shadow Hand解魔方，大规模domain randomization是关键
- [RoboPianist (Zakka et al. 2023)](https://arxiv.org/abs/2305.05933): Shadow Hand弹钢琴
- [Bi-DexHands (Chen et al. 2023)](https://arxiv.org/abs/2307.04105): bimanual dexterous manipulation benchmark
- [In-Hand Manipulation (Andrychowicz et al. 2020)](https://arxiv.org/abs/1910.07113): Pixel-RL的hand manipulation

### 6.3 HRL与Skill Priors
- [Options Framework (Sutton, Precup, Singh 1999)](https://www.sciencedirect.com/science/article/pii/S000437029900052X): temporal abstraction的基础
- [HIRO (Nachum et al. 2018)](https://arxiv.org/abs/1805.08296): data-efficient HRL with off-policy correction
- [Relay Policy Learning (Gupta et al. 2019)](https://arxiv.org/abs/1910.13728): long-horizon via imitation + RL，kitchen任务来源
- [Skill Priors (Pertsch et al. 2020)](https://arxiv.org/abs/2010.11444): 用learned skill priors加速RL
- [Composing Complex Skills (Lee et al. 2019)](https://arxiv.org/abs/1906.11166): transition policies between skills

### 6.4 Sim-to-Real
- [ANYmal (Hwangbo et al. 2019)](https://www.science.org/doi/10.1126/scirobotics.aau5872): legged robot sim-to-real with domain randomization
- [RMA (Kumar et al. 2021)](https://arxiv.org/abs/2107.04034): rapid motor adaptation for legged robots
- [Sim-to-real tactile (Sferrazza & D'Andrea 2022)](https://www.liebertpub.com/doi/10.1089/soro.2021.0093): tactile sensing sim-to-real，本文作者的前作

### 6.5 RL Algorithms
- [DreamerV3 (Hafner et al. 2023)](https://arxiv.org/abs/2301.04104): world model + latent imagination rollout，跨多个domain第一个SOTA
- [TD-MPC2 (Hansen et al. 2024)](https://arxiv.org/abs/2310.16828): model-based + MPC planning
- [SAC (Haarnoja et al. 2018)](https://arxiv.org/abs/1801.01290): maximum entropy RL，off-policy SOTA
- [PPO (Schulman et al. 2017)](https://arxiv.org/abs/1707.06347): on-policy SOTA，parallelization-friendly

### 6.6 GPU加速与JAX生态
- [Isaac Gym (Makoviychuk et al. 2021)](https://arxiv.org/abs/2108.10470): GPU physics for RL
- [PureJAXRL (Lu et al. 2022)](https://arxiv.org/abs/2210.05639): end-to-end JAX RL
- [MuJoCo MJX](https://mujoco.readthedocs.io/en/latest/mjx.html): JAX backend for MuJoCo

### 6.7 其他benchmark
- [MyoSuite (Caggiano et al. 2022)](https://arxiv.org/abs/2205.13600): musculoskeletal simulation
- [FurnitureBench (Heo et al. 2023)](https://arxiv.org/abs/2305.01860): real-world long-horizon manipulation
- [BEHAVIOR-1K (Li et al. 2022)](https://behavior.stanford.edu/): 1000 everyday activities
- [Habitat 2.0 (Szot et al. 2021)](https://arxiv.org/abs/2106.13805): home assistant

### 6.8 与Diffusion Policy和Foundation Models的联系
- [Diffusion Policy (Chi et al. 2023)](https://arxiv.org/abs/2303.04137): visuomotor policy via diffusion
- [Mobile ALOHA (Fu et al. 2024)](https://mobile-aloha.github.io/): bimanual mobile manipulation with teleoperation
- [RT-2 (Brohan et al. 2023)](https://robotics-transformer2.github.io/): VLM for robot control

---

## 7. 我的Intuition与思考

### 7.1 这个paper暴露的核心问题

**Action space dimensionality是RL exploration的真正瓶颈**。这不是新发现，但HumanoidBench把它量化得非常清楚：把action从19D扩到61D（增加3.2倍），即使是walk这种"easy"任务都从可学到几乎不可学。这意味着：
- 我们对exploration的理解还远不够
- Maximum entropy（SAC）和stochastic policy在这些维度上无法scale
- Behavior priors可能不再是"nice to have"而是"must have"

### 7.2 HRL的局限性

作者的HRL结果显示了经典HRL的固有矛盾：
- Low-level必须**足够general**以被多task重用 → 但这样训练困难
- Low-level必须**足够task-specific**以解决task → 但这样不能重用
- 他们的reaching policy是compromise，所以package task失败

**未来的方向**：可能不是固定low-level，而是**jointly train** with **gradient flow** through low-level (类似[Date et al. 2024](https://arxiv.org/abs/2403.06229)的work)。或者用**large-scale behavior cloning**作为warm start（来自human video，如[Mobile ALOHA](https://arxiv.org/abs/2401.02117)的思路）。

### 7.3 仿真速度的瓶颈

Table IV的数据很有意思：
- 默认配置：1050 FPS
- Simplified body collisions: 3600 FPS
- Collisions only on feet: 5100 FPS

这告诉我们：**collision detection是humanoid sim的性能瓶颈**。也是为什么MJX对复杂humanoid几何加速有限——MJX的优势在于JAX的jit，但复杂collision mesh让JAX难以efficiently vectorize。这限制了用大规模并行化解决exploration瓶颈的能力。

### 7.4 与我（Karpathy）的观察的呼应

在我[Neural Net训练essay](https://karpathy.ai/zero123/)和[Eureka Labs](https://eureka-labs.ai/)的视角下，HumanoidBench展示了**embodied AI与internet AI的鸿沟**：
- Internet data: 几乎无限的diverse demonstrations
- Humanoid: 没有任何human demonstrations可用，每个task的reward都需要manually design
- LLM在internet text上能zero-shot generalize，但humanoid在walk上都不能跨walk→run transfer

这暗示**foundation model for robotics**的核心不是policy网络结构，而是**如何获取diverse, scalable supervision signal**。可能是：
- Video pretraining（[VPT](https://arxiv.org/abs/2206.10467)思路）
- Self-supervised world model（DreamerV3方向）
- Cross-embodiment transfer（[RT-X](https://robotics-transformer-x.github.io/)）

### 7.5 关于Tactile Sensing

作者实现了448 taxel的whole-body tactile，但只用在observation里。这让我想到[Robot Synesthesia (Yuan et al. 2023)](https://arxiv.org/abs/2312.01853)和[The Power of the Senses (Sferrazza et al. 2023)](https://arxiv.org/abs/2311.00924)，后者是本文作者的前作。Tactile在**closed-loop in-hand manipulation**和**contact-rich assembly**中至关重要，但当前benchmark只用state-based observation，tactile被搁置。我预测**未来版本的HumanoidBench会以tactile为核心modality**。

### 7.6 对future work的猜想

paper的future work提到：
1. Multimodal perception（vision + tactile）
2. Realistic objects with real-world diversity
3. Furniture assembly和screwing tasks（bimanual友好）
4. Human video demonstrations
5. Sim-to-real with domain randomization

我会加：
6. **Language-conditioned tasks**：把task用natural language指定，让LLM做high-level planner
7. **Cross-embodiment transfer**：在H1训练的策略能否transfer到G1或Digit？这是humanoid foundation model的关键
8. **Multi-agent scenarios**：多个humanoid协作，参考[Habitat 3.0](https://arxiv.org/abs/2310.13724)
9. **Hierarchical RL的alternative**：distillation from LLM-generated code (like [Eureka](https://arxiv.org/abs/2310.12931))作为low-level skill source

---

## 8. 关键Limitations（paper没明说但很重要）

1. **Reward engineering overhead**：每个task都需要精心设计dense + sparse reward组合，这与"减少domain knowledge"的初衷相悖。可能需要[reward learning from preferences](https://arxiv.org/abs/1811.06521)或[LLM reward design](https://arxiv.org/abs/2310.12931)。

2. **Sim-to-real gap未量化**：所有结果都在sim里。作者提到MJX + domain randomization promising for sim-to-real，但没有真机实验。

3. **State-based only**：当前benchmark只用proprioception，不测vision/tactile。但真机部署必须依赖vision。这是个大gap。

4. **Task horizon不够long**：最长1000 steps，相当于20秒（50Hz）。真humanoid任务（如做饭、打扫）可能需要分钟级。可能需要结合[PDDL planning](https://arxiv.org/abs/2305.15766)或LLM planner。

5. **没有failure recovery**：episode一旦robot摔倒就terminate，没有teaching robot如何爬起来。真机部署中fall recovery至关重要，参考[ANYmal的fall recovery](https://www.science.org/doi/10.1126/scirobotics.aau5872)。

6. **Single agent**：没有human-robot interaction或multi-robot collaboration，但real deployment中这很关键。

---

## 9. 总结

HumanoidBench是**humanoid robot learning研究的inflection point**。它第一次系统地揭示了：
- SOTA RL算法在61维action space上**整体失败**
- Action dimensionality是核心瓶颈（不是observation，不是reward design）
- Hierarchical RL with pretrained motor skills是**当前最可行的方向**，但仍有局限

它也为community设置了一个**high bar**：27个task覆盖wide skill spectrum，未来算法的进展会以这个benchmark为衡量。我预期接下来12-18个月会看到大量paper报告在HumanoidBench上的新SOTA，类似[MuJoCo Gym](https://arxiv.org/abs/1606.01540)对continuous control RL的催化作用。

**核心open problem**：如何让RL agent在61维action space中**scale**？我的猜想是答案不在单算法，而在：
1. **Foundation model for robot control**（大规模pretraining + fine-tuning）
2. **LLM as high-level planner**（language grounding + code generation）
3. **Cross-embodiment transfer**（让H1的policy也能用到G1）

这些都会让HumanoidBench成为关键的test bed。

---

## References

- Paper: [HumanoidBench Project Page](https://humanoid-bench.github.io)
- [MuJoCo](https://mujoco.readthedocs.io/)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- [Unitree H1](https://www.unitree.com/h1/)
- [Shadow Hand](https://www.shadowrobot.com/dexterous-adapters/)
- [DreamerV3](https://arxiv.org/abs/2301.04104)
- [TD-MPC2](https://arxiv.org/abs/2310.16828)
- [SAC](https://arxiv.org/abs/1801.01290)
- [PPO](https://arxiv.org/abs/1707.06347)
- [PureJAXRL](https://arxiv.org/abs/2210.05639)
- [CoACD](https://arxiv.org/abs/2105.02955)
- [DMControl Suite](https://arxiv.org/abs/1801.00690)
- [LocoMujoco](https://arxiv.org/abs/2311.12472)
- [Adroit](https://arxiv.org/abs/1802.09464)
- [MyoSuite](https://arxiv.org/abs/2205.13600)
- [Bi-DexHands](https://arxiv.org/abs/2307.04105)
- [RoboPianist](https://arxiv.org/abs/2305.05933)
- [robosuite](https://arxiv.org/abs/2009.12293)
- [MetaWorld](https://arxiv.org/abs/1910.10897)
- [RLBench](https://arxiv.org/abs/1909.12271)
- [FurnitureBench](https://arxiv.org/abs/2305.01860)
- [BEHAVIOR-1K](https://behavior.stanford.edu/)
- [Habitat 2.0](https://arxiv.org/abs/2106.13805)
- [OpenAI Rubik's Cube](https://arxiv.org/abs/1910.07113)
- [Expressive Whole-Body Control](https://arxiv.org/abs/2402.16796)
- [Humanoid Locomotion with Transformers](https://arxiv.org/abs/2303.03381)
- [Robot Parkour Learning](https://arxiv.org/abs/2309.05665)
- [ANYmal](https://www.science.org/doi/10.1126/scirobotics.aau5872)
- [RMA](https://arxiv.org/abs/2107.04034)
- [Mobile ALOHA](https://arxiv.org/abs/2401.02117)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [RT-2](https://robotics-transformer2.github.io/)
- [Relay Policy Learning](https://arxiv.org/abs/1910.13728)
- [Options Framework](https://www.sciencedirect.com/science/article/pii/S000437029900052X)
- [HIRO](https://arxiv.org/abs/1805.08296)
- [Skill Priors](https://arxiv.org/abs/2010.11444)
- [Eureka (LLM Reward Design)](https://arxiv.org/abs/2310.12931)
- [VPT (Video PreTraining)](https://arxiv.org/abs/2206.10467)
- [RT-X](https://robotics-transformer-x.github.io/)
- [Robot Synesthesia](https://arxiv.org/abs/2312.01853)
- [The Power of the Senses](https://arxiv.org/abs/2311.00924)
- [Atlas Control (Kuindersma et al.)](https://link.springer.com/article/10.1007/s10514-015-9476-3)
- [Karpathy - Eureka Labs](https://eureka-labs.ai/)
- [Isaac Gym](https://arxiv.org/abs/2108.10470)
- [DeepMimic](https://dl.acm.org/doi/10.1145/3197517.3201925)
- [AMP](https://dl.acm.org/doi/10.1145/3450626.3459975)
- [Composing Complex Skills (Lee et al.)](https://arxiv.org/abs/1906.11166)
- [Option-Critic](https://arxiv.org/abs/1609.05140)
- [Preference Learning RLHF](https://arxiv.org/abs/1811.06521)
- [Habitat 3.0](https://arxiv.org/abs/2310.13724)

希望这个analysis对你build intuition有帮助，Andrej！HumanoidBench的真正意义不只是benchmark本身，是它**quantify了一个open problem**：high-dimensional action space下的exploration。这与language model中token prediction的"easy supervision"形成鲜明对比，提示了robot learning独特的scale laws。
