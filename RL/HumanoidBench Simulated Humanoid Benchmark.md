---
source_pdf: HumanoidBench Simulated Humanoid Benchmark.pdf
paper_sha256: 000dd3c95bcc8697d2e6e2425b60478dfb0c42ff3ccc82fabe63dd6e1cd55b03
processed_at: '2026-08-05T08:07:49-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# HumanoidBench 人话版

OK 我重新用大白话给你讲一遍这个故事,Andrej。

---

## 这个 paper 在干嘛:一句话版本

**人形机器人硬件崛起了,但 software 还停在手工调参时代。作者搞了个 simulation benchmark 让大家在虚拟世界里测算法,结果发现 SOTA RL 在大多数任务上都 fail,只有 hierarchical RL 能救一部分场。**

---

## 背景:为什么需要这个 benchmark

你看现在 humanoid 这个领域,Boston Dynamics Atlas、Tesla Optimus、Unitree H1 这些硬件一年比一年猛。Unitree H1 一台 9 万美金左右,放在 5 年前这是科幻。但 controller 呢?全是 hand-engineered 的,每个 task 都得工程师重新 tune 一遍。

问题来了:你想做 algorithm research,但 real-world humanoid 实验又贵又危险。H1 摔一次维修费就够你哭半年。于是大家都被硬件卡死了。

那 simulation 呢?现有的 benchmark 都不完整:

- **MyoHand / Adroit**:只测手,不测全身。Action dim 最多 39。
- **LocoMujoco**:只测走路,不测手。Action dim 19。
- **robosuite / MetaWorld / RLBench**:机械臂 manipulation,6-7 DoF 的 toy problem。
- **BEHAVIOR / Habitat**:long-horizon,但把 low-level control 抽象掉了,你点哪它走哪。
- **FurnitureBench**:long-horizon assembly,但是单臂 8D action。

中间有个巨大的空白:**没有 benchmark 同时测高维 action + long-horizon + 全身协调**。HumanoidBench 就是填这个空。

参考: [HumanoidBench 项目主页](https://humanoid-bench.github.io) | [arXiv](https://arxiv.org/abs/2410.05005)

---

## 机器人配置:为什么是 H1 + Shadow Hand

作者选了 **Unitree H1** 当主体,挂两个 **Shadow Hand** 当手。这个组合有点意思:

**为什么选 H1 不选 Digit?**
Digit 用 four-bar linkage 驱动 passive joint,看起来 mechanical design 很 fancy,但 RL 学起来更慢。H1 mechanical design 简单,RL 反而 sample-efficient。这是一个很反直觉的 finding:**mechanical simplicity 是 RL-friendliness 的重要 factor**。你硬件搞得越 fancy,RL 越难学。

**为什么截掉 Shadow Hand 的 forearm?**
原版 Shadow Hand forearm 很 bulky,不像人手。作者干脆截掉,虽然这不符合现在的硬件,但 align with Tesla Optimus / Figure 01 这种下一代 slim hand 趋势。

**Underactuation 现象**:
- H1 body: 19D action, 25 DoF (少了 6D,是 floating base)
- 一个 Shadow Hand: 21D action, 25 DoF (少了 4D)
- 全系统: **61D action, 75 DoF** (少了 14D)

Observation 是 151D (51D body + 50D × 2 hands)。注意 floating base 用 quaternion 表示 orientation,所以 position 比 velocity 多一维。

参考: [Unitree H1](https://www.unitree.com/h1) | [Shadow Hand](https://www.shadowrobot.com/products/dexterous-hand/) | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)

---

## 全身触觉:工程亮点

作者还实现了 448 taxels 分布全身的触觉 sensing,每个 taxel 读 3D contact force (normal + 2 shear)。

这个工程不 trivial。MuJoCo 的 touch grid sensor 需要把 mesh 切成很多小块才能有好的 spatial resolution。作者用 [CoACD](https://github.com/SpatialComputing/CoACD) 把原始 mesh 切成多个 convex sub-meshes,每个 sub-mesh 独立做 contact detection。

代价:default model 跑 1050 FPS,refined mesh model 只跑 550 FPS。RL 训练时这是个 bottleneck。

参考: [Mittendorfer & Cheng, humanoid tactile](https://ieeexplore.ieee.org/document/5731340) | [Sferrazza sim-to-real optical tactile](https://www.liebertpub.com/doi/10.1089/soro.2021.0164)

---

## Action / Observation 设计哲学

**Action**: normalized 到 `[-1, 1]^61`,position control @ 50 Hz。作者说 position control 比 torque control 更稳定,可以低频控制。这背后的 implicit 假设是 MuJoCo 的 position actuator 自带 PD,已经帮你 smooth 了 control signal。

**Observation**: 作者刻意保持 observation 在所有 task 上一致,**不 task-specific tailor observation**。这避免了 observation engineering 的 reward hacking。代价是 algorithm 必须自己 figure out 哪些 dim relevant。

这是一个 strong design choice——很多 benchmark 会给 agent "提示"(e.g., 直接把目标距离塞进 observation),作者拒绝这么做,这样 algorithm 比较才公平。

---

## Reward 设计:统一的 building blocks

所有 task 的 reward 都 built on 一套共享的 auxiliary functions。这个 design pattern 很 elegant,讲讲核心几个:

### `tol(x, (x_lower, x_upper), m)`

这是从 [DM Control Suite](https://github.com/google-deepmind/dm_control) 拿来的 tolerance function。直觉是:如果 `x` 在 `(x_lower, x_upper)` 区间内,返回 1;偏离区间但在 margin `m` 内,平滑衰减到 0;超过 margin 返回 0。

这是 dense reward shaping 的标准 trick。比 raw distance penalty 好,因为它 saturates at 1,agent 不会偏离主任务去 minimize 一个 unbounded metric。

### `height = tol(z_head, (1.65, +∞), 0.4125)`

头部高度 reward。default 要求头至少 1.65m 高,margin 0.41m。

### `upright = tol(z_proj, (0.9, +∞), 1.9)`

躯干垂直度。`z_proj` 是 robot body frame 的 z-axis 在 world frame z-axis 上的投影(就是 rotation matrix 的 `R[2,2]`)。完全直立时 = 1,完全倒立时 = -1。Default 要求 ≥ 0.9。

### `stand = height × upright`

站立 reward。这里用 **multiplicative** 而不是 additive 很关键。如果 additive,robot 可能学会 "stand tall but lean forward" 这种 degenerate solution。multiplicative 强制两个条件同时满足。

### `e = 0.2 × |4 + (1/|u|) × Σ_i tol(u_i, (0,0), 10)|`

Effort reward。`u` 是 actuator input vector,`|u|` = 61。每个 actuator 都有一个 "接近零 actuation" 的 indicator,平均一下。整体设计成 positive shaping 而不是 penalty——避免 RL agent 因为 penalty 过大而 stalling。

### `stable = stand × e`

稳定站立 = 站直 + 低 effort。

### 举个 task 例子

**walk**: `R = stable × tol(v_x, (1, +∞), 1)`

要求:站直 + 低 effort + 向前速度 1 m/s。简洁。

**push**: `R = α_s × success - α_t × d_goal - α_h × d_hand`,其中 `α_s=1000, α_t=1, α_h=0.1`

sparse success reward + dense shaping。`d_hand` 是 hand 到 box 的距离,引导 agent 去接近 box。

**kitchen**: 纯 sparse,4 个 subtask 各完成给 1 分,最大 4 分。**所有 RL 算法都 return 0**。这是 sparse reward + 高维 action 的死亡组合。

参考: [DM Control Suite rewards.py](https://github.com/google-deepmind/dm_control/blob/main/dm_control/utils/rewards.py)

---

## 27 个 task 速览

**12 个 locomotion tasks**:
walk, stand, run, reach, hurdle, crawl, maze, sit (simple/hard), balance (simple/hard), stair, slide, pole

**15 个 manipulation tasks**:
push, cabinet (4 subtasks), highbar, door, truck, cube, bookshelf (simple/hard), basketball, window, spoon, kitchen, package, powerlift, room, insert (small/normal)

难度跨度很大。`walk` 是 toy,`cabinet` 和 `kitchen` 是 long-horizon 实用任务,`highbar` 需要 athletic 全身 swing。

---

## Baseline 结果:大部分 fail

跑了 4 个 SOTA RL:

| Algo | Type | Steps | 大致表现 |
|---|---|---|---|
| DreamerV3 | model-based world model | 10M | 整体最好 |
| TD-MPC2 | model-based + MPC | 2M | 第二 |
| SAC | off-policy model-free | 10M | 中等 |
| PPO | on-policy model-free | subset | 没并行根本学不动 |

参考: [DreamerV3](https://arxiv.org/abs/2301.04104) | [TD-MPC2](https://arxiv.org/abs/2310.16828) | [SAC](https://arxiv.org/abs/1801.01290) | [PPO](https://arxiv.org/abs/1707.06347)

看 Table V 的 average returns:
- `walk`: target 700,DreamerV3 拿 800——基本解决
- `stand`: target 800,基本解决
- `kitchen`: target 4,所有算法 0.0——彻底失败
- `push`: target 700,DreamerV3 avg -1251.9,max 1000——偶尔成功但不稳定
- `package`: target 1500,DreamerV3 avg -18015——大失败
- `bookshelf_hard`: target 2000,最好 ~870——远未解决
- `basketball`: target 1200,avg 19.3——基本失败

只有 ~5 个 task 真正 reach target。**这个 benchmark 远未 saturated**,这是好 benchmark 的标志。

### avg vs max 的 gap 揭示 training instability

`push` 的 DreamerV3 avg -1251.9 但 max 1000。说明 agent 偶尔 reach high return 但 collapse 回 poor behavior。这是 high-dim RL 的 typical symptom——policy 在 high-dim space 里 unstable。

---

## 关键 finding:为什么 flat RL fail

### Ablation:with hands vs without hands

在 `walk` 这个不需要手的 task 上做实验:
- **Without hands**: 19D action, 51D obs ——学得飞快
- **With hands, action fixed to 0**: 19D effective action, 151D obs——学得也快
- **With hands, full action**: 61D action, 151D obs——慢且差

**关键洞察**:observation 维度从 51 → 151 不是 bottleneck,action 维度从 19 → 61 才是。

**为什么 RL 不能自动忽略 irrelevant action dims?**

直觉讲:SAC 的 maximum entropy exploration 是 `α · H(π(·|s))`。对 61D Gaussian policy,entropy budget 被分到 42 个 irrelevant dims 上,真正 task-relevant 的 19D 的 effective exploration 被稀释。这本质上是 **exploration capacity allocation** 问题。

人类也不是每 step 控制 75 DoF 的——我们有 motor synergies,一组肌肉协同激活。RL 在 raw joint space 上学就吃了这个亏。

参考: [SAR: Synergistic Action Representation](https://arxiv.org/abs/2306.04786) 这个方向的工作

---

## Common Failures:算法的真正 limitation

### `highbar` 失败:local optimum trap

Reward 设计没问题——dense reward 鼓励 swing 到 upside-down (`z_proj < -0.9`) 且脚 ≥ 4.8m。

但 robot 学到的是:挂着不动。因为挂着 contact 不 terminate,episode 拿小但稳定的 reward。Swing-up 需要 transient reward dip(从 hanging 摆到 upside-down 中间过程 reward 低),RL 不愿意去。

这是 dense reward 也救不了的 local optimum。

### `door` 失败:multi-body coordination

Robot 学会 unlock hatch(转动门把手),但不会拉门。

为什么?拉门需要 *simultaneous* 手臂拉 + 整个身体后退。这种 locomotion 和 manipulation 的时序耦合,flat RL 学不出来。RL 学会了"手臂动作"和"走路",但没法把它们 time-align 到一个 dynamic 的 door 上。

### `hurdle` 失败:reward shaping 的 unintended consequence

Robot 学到向前跑 5 m/s,但撞 hurdle 不 jump。

为什么?因为 termination condition 有漏洞——撞 hurdle 不 terminate episode。Robot 找到保守策略:撞一下稳住,继续小步前进。这比 jump 的 reward 高(jump 失败概率高,一摔就 terminate)。

Walker2d 这种简化 humanoid 在 forward reward 下能学会 jump,但 H1 的 termination condition 不同,行为就不同。这揭示了:**reward + termination 的细节决定行为,微小的设计选择都会让 RL 走向完全不同的 local optimum**。

参考: [Lee et al. 2019, Composing Complex Skills](https://openreview.net/forum?id=rygrBhC5tQ)

---

## Hierarchical RL:作者的解法

### Architecture

```
High-level policy π_H(s_t) → 3D target setpoint
        ↓
Frozen low-level reaching policy π_L(target) → 19D joint command
        ↓
MuJoCo environment (hands fixed to 0)
```

High-level 输出 *reaching target*(3D 或 6D for two hands),low-level 把 target 翻译成 19D joint position command。

### Low-level pretraining:为什么 PPO 在这里能 work?

Standard PPO 在 HumanoidBench 上 fail,但作者用 PPO 在 low-level 上成功了。两个关键 differences:

1. **Massive parallelization**:用 [MuJoCo MJX](https://github.com/google-deepmind/mujoco_mjx) 在 JAX 上跑 32,768 个并行环境。PureJaxRL implementation。
2. **Simplified model**:只保留 feet-ground collisions,移除其他 body collision meshes,移除 hands。

FPS 对比:
- Default full model: 1050 FPS
- Simplified (only feet): 5100 FPS

5x 加速 + 32K 并行 = 几百倍 wallclock speedup。

训练 budget:
- 1-hand reaching: 2B steps / 36 小时
- 2-hand reaching: 4B steps / 60 小时

**Robustness trick**:训练时对每个 link 施加 random force perturbation。Why?因为 high-level policy 在 exploration 时会输出 drifting targets,low-level 必须能从 disturbance 中 recover。

参考: [PureJaxRL](https://github.com/luchris424/purejaxrl) | [Isaac Gym](https://developer.nvidia.com/isaac-gym)

### High-level training

用 DreamerV3 或 TD-MPC2,只训 high-level,low-level frozen。

**Action space restriction**:把 reaching target 限制在 robot reachable workspace 内。这是 non-trivial 的——避免 high-level 探索到 unreachable target 时 low-level 出现 undefined behavior。

数学上 composite policy:
$$
a_t = \pi_L(\pi_H(s_t))
$$

这是 [Sutton-Precup-Singh options framework](https://arxiv.org/abs/cs/9905014) 的 special case,option 是 deterministic 且 instantaneous。

参考: [Option-Critic](https://arxiv.org/abs/1609.05140) | [DIAYN](https://arxiv.org/abs/1802.06070) | [Data-efficient HRL](https://arxiv.org/abs/1805.08296)

### Hierarchical 结果

**push**: flat DreamerV3 拿 -1251,hierarchical DreamerV3 拿 1000。**完美解决**。

**package**: 改善但没解决。Why?因为 package 需要 *lift*——抓起来向上抬。Reaching primitive 只覆盖 "把手伸到某点",不覆盖 "grasp + lift"。Low-level skill repertoire 不够。

**Key insight**:Hierarchical RL 只在 low-level skill 覆盖 task 的 motor primitive 时 work。如果 task 需要 skill 之外的 motion,hierarchical 也救不了。

直觉:这就像让只会走路的人去做举重——locomotion primitive 不包含 lifting,得学新 skill。

---

## 这篇 paper 的 implicit assumptions 和 weaknesses

### Reward engineering 仍然 art

虽然 modular design 优雅,但 magic numbers 处处都是:
- `push` 的 `α_s = 1000`——为什么不是 100 或 10000?
- `tol` margin `0.4125` for height——为什么这么 specific?
- `kitchen` sparse reward 每个 subtask 1 分——为什么不是 100?

这些是 tuning 的 art,不利于 systematic algorithm comparison。更 rigorous 的做法是 automatic reward shaping 或 learn reward from demos。

### Hierarchical 只在 2 个 task 上试

27 个 task,只在 push 和 package 上试 hierarchical。那 highbar、basketball、window 这种 dynamic 全身任务呢?Reaching primitive 显然不够——你没法用 "把手伸到某点" 解决 swing-up 或投篮。

要 cover 所有 task,可能需要一组 primitives:walk, reach, grasp, lift, push, throw, swing。每个 primitive 2-4B steps pretrain,total pretraining cost 几十亿 steps。这是 significant infrastructure requirement。

### Compute 门槛

每个 baseline 48 小时 × 3 seeds。Hierarchical 还要加上 low-level pretraining 36-60 小时 + 32K parallel envs。这对 academic lab 是非 trivial 门槛。

作者 mention 用了 Google TPU Research Cloud。这对大多数 labs 不可得。

参考: [Mobile ALOHA](https://arxiv.org/abs/2401.02117) 显示 teleoperation + imitation 可能是 hierarchical RL 的 alternative

---

## 一些有意思的 future direction

### 1. Skill discovery

作者 hand-design 了 "reaching" 当 low-level skill。但能不能自动 discover motor primitives?

[DIAYN](https://arxiv.org/abs/1802.06070)、[Option-Critic](https://arxiv.org/abs/1609.05140) 这种方法可以 unsupervised 学出 diverse skills。然后 high-level policy 在 learned skill space 上 planning,可能比 hand-design 的 reaching 更 general。

### 2. Diffusion Policy on humanoid

[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) 在 bimanual manipulation 上 shown 很好,但 75 DoF humanoid 上没试过。Diffusion 的 multi-modal action distribution 可能正好解决 "high-dim action space 探索不过来" 的问题。

### 3. VLM + hierarchical

[RT-2](https://arxiv.org/abs/2307.15818) 这种 VLM 做 high-level planner + HumanoidBench 训的 low-level skills。LLM 把 "open cabinet and put cube in" decompose 成 subtask sequence,hierarchical RL 只需要 solve 每个 subskill。

### 4. Cross-embodiment transfer

H1 上训的 skill 能否 transfer 到 G1 或 Digit?这 test skill 的 embodiment-invariance。如果可以,foundation model for control 就有可能。

### 5. World model scaling

DreamerV3 在 61D action 上 partial work,但 world model 的 latent dynamics 是否能 capture 全身 coordination 是 open question。75 DoF 的 transition function 比 Atari 复杂几个数量级。

### 6. Sim-to-real 是大坑

Paper 提到 MJX + domain randomization 有希望,但 humanoid sim-to-real 远比 quadruped 难:
- 75 DoF 的 actuator dynamics modeling error 累积
- Dexterous hand 的 friction 在 sim 和 real 差异大
- Humanoid falling 是 catastrophic,不能像 quadruped 那样 recover

参考: [Hwangbo et al. 2019, legged robot sim-to-real](https://www.science.org/doi/10.1126/scirobotics.aau5872) | [Expressive Whole-Body Control](https://arxiv.org/abs/2402.16796)

---

## 整篇文章的 takeaway

1. **Hardware 起来了,software 还没**。HumanoidBench 给了 community 一个 standard testbed。

2. **高维 action 是 RL 的死敌**。即使 task 不需要某些 DoF,他们的存在就 ruin exploration。Action abstraction / motor synergies 是 must-have direction。

3. **Hierarchical RL 部分有效,但 skill library 必须够广**。Reaching 只能解决一类 task,真正 general 需要 diverse primitives。

4. **Massive parallelization 是必要条件**,不再是 nice-to-have。PPO 在 standard setup fail,在 MJX 32K envs 上能 pretrain robust skill。Compute 不再 optional。

5. **这个 benchmark 远未 saturated**。27 个 task 只有 ~5 个真正 solved。有大量 headroom,适合做 foundation model for control 的 testbed。

6. **Reward engineering 仍然是 art**。Magic numbers 处处都是,需要更 systematic 的方法。

7. **Mechanical simplicity 是 RL-friendliness 的 factor**。H1 比 Digit 更 RL-friendly,因为 simpler kinematics。这对未来 humanoid hardware design 有 implication。

---

## 一句话总结

**HumanoidBench 是 humanoid robotics 的 "ImageNet moment" 候选——它把一个 fragmented 的领域标准化了,然后大家发现 SOTA 算法在大多数任务上都 fail,这就是 research opportunity。**

如果你想动手,直接去 https://humanoid-bench.github.io 拉 code,跑个 baseline 看看。这个 benchmark 适合做:
- Foundation model for control 的 testbed
- LLM/VLM planner 的下游 task suite
- Hierarchical RL 的 skill library testbed
- Diffusion policy 在高维 action 上的 stress test

Happy hacking, Andrej。

---

# HumanoidBench 深度技术解析

很高兴和你聊这篇 paper,Andrej。这个 benchmark 的设计选择背后藏了很多非 trivial 的洞察,值得逐层拆解。下面我尽量把技术细节铺开,帮你 build intuition。

---

## 1. Paper 核心动机 (Why this benchmark, why now)

Humanoid robotics 处于一个尴尬的 phase:硬件 (Unitree H1, Tesla Optimus, Boston Dynamics Atlas) 已经 commercial-ready,但 controller 依然 hand-engineered,并且每个 task 都需要 extensive engineering effort。问题是算法 research 被硬件 bottleneck 住了——real-world humanoid 实验既贵又危险。

**Existing benchmark 的盲区** (Table I):

| Benchmark | Dexterous hands | Action dim. | DoF | # Tasks | Skills covered |
|---|---|---|---|---|---|
| MyoHand | ✓ | 39 | 23D | 9 | manipulation only |
| LocoMujoco (H1) | ✗ | 19 | 6D | 27 | locomotion only |
| robosuite | ✗ | 6-24 | 6-7D | 9 | quasi-static manipulation |
| FurnitureBench | ✗ | 8 | 6D | 8 | long-horizon assembly |
| **HumanoidBench** | **✓** | **61** | **75D** | **27** | **full stack** |

关键 gap 在于:previous benchmark 要么 focus 单一 skill (manipulation OR locomotion),要么抽象掉 low-level control (BEHAVIOR, Habitat)。HumanoidBench 同时 test 高维 control + long-horizon + whole-body coordination,这是一个真正 combined 的 challenge。

参考链接:
- 项目主页: https://humanoid-bench.github.io
- 论文 arXiv: https://arxiv.org/abs/2410.05005 (CoRL 2024)
- MuJoCo Menagerie: https://github.com/google-deepmind/mujoco_menagerie

---

## 2. Robot Configuration 的工程考量

### 2.1 为什么选 Unitree H1?

| Robot | DoF (body) | Notes |
|---|---|---|
| **Unitree H1** (chosen) | 25 | Full-size humanoid,simpler mechanical design |
| Unitree G1 | 29 | Smaller, less full-size |
| Agility Robotics Digit | 57 | Passive joints via four-bar linkage, complex dynamics |

这里有个隐含的 insight:Digit 的 four-bar linkage 让 inverse dynamics 变得 tricky,因为 passive joints 实际上是 underactuated 的 indirect control,RL 学起来更慢。H1 的 simpler mechanical design 反而 sample-efficient。这是 RL community 经常忽视的 design principle:**mechanical simplicity 是 RL-friendliness 的重要 factor**。

### 2.2 为什么 Shadow Hand (且截掉 forearm)?

Shadow Hand 模型来自 MuJoCo Menagerie,本来 forearm 很 bulky。作者把 forearm 移除,虽然不符合 real-world hardware,但 align with next-generation humanoid hand design trend (Tesla Optimus, Figure 01 都是 slim hand)。

**Underactuation 的关键 fact** (Table II):

| Component | Action dim | DoF | Underactuated by |
|---|---|---|---|
| H1 body | 19 | 25 | 6 (floating base 6D + 一些耦合) |
| One Shadow Hand | 21 | 25 | 4 |
| **Full system** | **61** | **75** | **14** |

Observation space breakdown:
- 51D (body) + 50D × 2 (hands) = **151D total**
- 这个 observation 维度 来自于 generalized positions 和 velocities,且 floating base 用 quaternion 表示 (4D position + 3D velocity),所以 position 比 velocity 多一维。

### 2.3 Tactile Sensing 的工程细节

Paper 实现了 448 taxels 分布全身,每个 taxel 提供 3D contact force reading (normal + 2 shear components)。这是 non-trivial 的工程:

1. 用 [CoACD (Convex Approximate Decomposition)](https://github.com/SpatialComputing/CoACD) 把原始 mesh subdiv 成多个 convex sub-meshes
2. MuJoCo touch grid 在每个 sub-mesh 上独立做 spatial binning
3. Sub-mesh 越多 → contact points 越多 → tactile 空间分辨率越高

参考 tactile 相关工作:
- Real humanoid body tactile: [Mittendorfer & Cheng, 2011](https://ieeexplore.ieee.org/document/5731340)
- Sim-to-real for optical tactile: [Sferrazza & D'Andrea, 2022](https://www.liebertpub.com/doi/10.1089/soro.2021.0164)

Performance 表 (Table IV):Default model 跑 1050 FPS,refined mesh model 跑 550 FPS,这对 RL sample efficiency 是个 concern (SAC/DreamerV3 都是 off-policy,每 step 都要 env step)。

---

## 3. Action/Observation Space 的设计哲学

### 3.1 Action Space

- Normalized to `[-1, 1]^61`
- Position control at 50 Hz (vs torque control,后者需要更高 frequency,typically 200-1000 Hz)
- 为什么 position control 更 stable:它 abstract away 了 PD gains 的 tuning,而且 MuJoCo 的 actuator dynamics (`position` type with built-in PD) 已经 smooth 了 control signal

### 3.2 Observation Space 的 key design choice

> "we maintain the robot observations the same across tasks to minimize domain knowledge"

这是一个 critical 的设计原则——不 task-specific tailoring observation。这避免了 reward hacking 风格的 observation engineering。代价是 algorithm 必须自己 figure out 哪些 observation dims relevant,这又增加了 learning 难度。

---

## 4. Reward Function 的统一框架

这是这篇 paper 最值得 build intuition 的部分之一。所有 task reward 都 built on 一组 auxiliary functions,这种 modular design 很 elegant。

### 4.1 核心辅助函数

**Tolerance function** (来自 DM Control Suite):

```
tol(x, (x_lower, x_upper), m) = 
  1                                    if x ∈ (x_lower, x_upper)
  smoothly decays to 0 within margin  otherwise
```

变量含义:
- `x`: 被评估的 scalar value (e.g., 速度、高度、距离)
- `(x_lower, x_upper)`: desired bounds,task-specific
- `m`: margin,控制 sigmoid 衰减的 slope——m 越大,衰减越平缓 (more forgiving)

直觉:tol 是一个 bounded reward shaping function,把 "在 acceptable range" 转换为 [0, 1] 的 dense signal。这比 raw distance penalty 更 sample-efficient,因为它 saturates at 1,不会让 agent 偏离 main task 去 minimize 一个 unbounded metric。

参考: [DM Control Suite source](https://github.com/google-deepmind/dm_control/blob/main/dm_control/utils/rewards.py)

**Height reward**:

```
height((x_lower, x_upper), m) := tol(z_head, (x_lower, x_upper), m)
```

- `z_head`: 机器人头部在世界 frame 下的 z 坐标
- Default: `(1.65, +∞, 0.4125)` — 头部高度至少 1.65m,margin 0.41m

**Upright reward**:

```
upright((x_lower, x_upper), m) := tol(z_proj, (x_lower, x_upper), m)
```

- `z_proj`: 机器人 body frame 的 z-axis 在 world frame z-axis 上的 unit projection
  - 数学上:`z_proj = R_{world←body}[2,2]` (rotation matrix 第三行第三列)
  - 完全 upright 时 `z_proj = 1`,完全 upside-down 时 `z_proj = -1`
- Default: `(0.9, +∞, 1.9)`

**Standing composite**:

```
stand := height × upright
```

这是 multiplicative 的关键设计——必须同时保持 head height 和 torso verticality。如果用 additive,robot 可能学会 "stand tall but lean forward" 这种 degenerate solution。

**Effort penalty**:

$$
e := 0.2 \cdot \left| 4 + \frac{1}{|u|} \sum_i \text{tol}(u_i, (0, 0), 10) \right|
$$

变量含义:
- `u`: actuator input vector (61D for full system)
- `|u|`: actuator 数量 (61)
- `tol(u_i, (0, 0), 10)`: 每个 actuator 的"接近零 actuation"程度,margin 10 (very forgiving)
- Sum over all actuators
- 这个 expression 让 `e` 越大越 reward (i.e., 接近 zero control 时 e 趋近 0.2 × |4 + 1| = 1)

直觉:e 是 "low effort bonus",鼓励 efficient control。但 design 上是 *positive shaping*,不是 penalty——这避免了 RL agent 因为 penalty 过大而 stalling。

**Stable composite**:

```
stable := stand × e
```

- 站直 × 低 effort = 稳定站立
- 注意这是 dense,全 episode 持续给

### 4.2 具体 task reward 例子

**walk**:
$$
R(s, a) = \text{stable} \times \text{tol}(v_x, (1, +\infty), 1)
$$
- `v_x`: body frame 下 x 方向速度 (positive forward)
- Goal: 1 m/s forward,稳定站立,low effort
- Termination: `z_pelvis < 0.2` (falling) 或 1000 steps

**push** (manipulation):
$$
R(s, a) = \alpha_s \cdot \text{success} - \alpha_t \cdot d_{\text{goal}} - \alpha_h \cdot d_{\text{hand}}
$$
- `success := 𝟙_{d_goal < 0.05}` (sparse binary)
- `d_goal = d(box, destination)`: box 到目标的距离
- `d_hand = d(box, hand_left)`: hand 接近 box 的程度 (shaping)
- Default: `α_s = 1000, α_t = 1, α_h = 0.1`
- Hierarchical 的 dense + sparse 组合

**highbar** (failure case):
$$
R(s, a) = \text{upright}_{\text{highbar}} \times \text{feet} \times e
$$
其中:
- `upright_highbar = upright((-∞, -0.9), 1.9)`: torso 必须 upside-down (`z_proj < -0.9`)
- `feet = tol((z_foot_left + z_foot_right)/2, (4.8, +∞), 2)`: 双脚平均高度 ≥ 4.8m

这里 reward 设计本身没问题,但 RL agent 学到的是 "保持 contact 但不 swing"——conservative local optimum。

**cabinet** (4 subtasks):
这个是 multi-stage 的典型 long-horizon 任务:
1. Open sliding door (top): `R_1 = 0.2·stable + 0.8·|l_cabinet / 0.4|`
2. Open drawer (bottom): `R_2 = 0.2·stable + 0.8·|l_drawer / 0.45|`
3. Move cube from drawer → hinge cabinet: 复合 reward with door opening + cube destination
4. Move cube from hinge cabinet → pull-up cabinet: 同上结构

每完成 subtask i 给 sparse reward `i × 100`,全部完成给 `1000`。这种 staircase-shaped reward 在 HRL 文献里很常见 ([Relay Policy Learning](https://arxiv.org/abs/1910.11905))。

**kitchen** (extreme case):
> "the only environment with a purely discrete, sparse reward, with a maximum of 4"

纯 sparse reward,4 个 subtask (open microwave, move kettle, turn burner, turn light)。所有 baseline 算法都 return 0.0 (Table V)——纯 sparse reward + 高维 action 直接死掉。

---

## 5. Baseline Algorithms 的对比

| Algorithm | Type | Key property | Steps trained |
|---|---|---|---|
| DreamerV3 | Model-based (world model) | Learns from imagined rollouts | 10M |
| TD-MPC2 | Model-based + MPC planning | Online planning via trajectory optimization | 2M |
| SAC | Off-policy model-free | Maximum entropy, soft critic | 10M |
| PPO | On-policy model-free | Stability via clipped objective | subset only |

参考链接:
- [DreamerV3](https://arxiv.org/abs/2301.04104)
- [TD-MPC2](https://arxiv.org/abs/2310.16828)
- [SAC](https://arxiv.org/abs/1801.01290)
- [PPO](https://arxiv.org/abs/1707.06347)

为什么 PPO 没全跑?因为它 sample efficiency 太差,without massive parallelization (像 Isaac Gym 那种 4096+ parallel envs) 根本学不动。Paper 里 PPO 只在 walk, kitchen, door, package 上跑——这本身就是一个 finding:standard PPO implementation (Stable-Baselines3, 4 parallel envs) 在 humanoid 上 fail。

---

## 6. 关键实验发现:Why Flat RL Fails

### 6.1 With Hands vs Without Hands (Figure 7)

这是最重要的 ablation。在 **walk** task 上 (不需要 hands):

| Setup | Action dim | Observation dim | Performance |
|---|---|---|---|
| Without hands | 19 | 51 | Fast learning, high return |
| With hands (fixed to 0) | 19 | 151 | Fast learning (similar to without) |
| With hands (full action) | 61 | 151 | Slow learning, low return |

**Insight #1**:问题不是 observation dimension (151D obs 加 hands 信息依然 works),而是 **action dimension**。

**Insight #2**:即使 hands 在 task 中根本不被使用 (walk 不需要 grasp),只要 action space 包含 hands 的 42D,RL 就大幅 fail。这意味着 RL algorithm 无法自动"忽略"irrelevant action dimensions。

直觉 build:think of it as exploration problem。SAC 的 maximum entropy exploration 是 `α · H(π(·|s))`,对 61D Gaussian policy 来说,entropy budget 被分到 42 个 irrelevant dims 上,真正 task-relevant 的 19D 的 effective exploration 被稀释。这本质上是 **exploration capacity allocation** 的问题。

### 6.2 Average vs Maximum Returns (Table V vs VI)

| Task | Target | DreamerV3 avg | DreamerV3 max | Gap |
|---|---|---|---|---|
| walk | 700 | 800.2 | 932.4 | solved |
| stand | 800 | 622.7 | 932.9 | solved (max) |
| kitchen | 4 | 0.0 | 0.0 | complete fail |
| push | 700 | -1251.9 | 1000.0 | solved (max) but avg fail |
| package | 1500 | -18015.2 | 1009.2 | not solved |
| bookshelf_hard | 2000 | 530.2 | 867.8 | far from solved |
| basketball | 1200 | 19.3 | 808.8 | not solved |

**Insight**:avg 远低于 max 说明 training instability——agent 偶尔 reach high return,但 collapse 回 poor behavior。这是 high-dim RL 的 typical symptom。`package` 的 avg 是 -18015 因为 reward 包含 `-3 · d(package, destination)`,只要 agent 不动,每 step 都 accumulate 负 reward。

### 6.3 Common Failures 揭示的 algorithmic limitation

**highbar 失败**:
- Reward 已经 dense (每 step 都给 `upright × feet × e`)
- Robot 学到:保持 contact with bar 来 avoid termination (`z_head < 2`)
- 但无法 perform swing-up rotation
- 这是 **local optimum trap**——保守 hanging 给小但稳定的 reward,而 swing-up 需要 transient reward dip

**door 失败**:
- Robot 学会 unlock hatch (`open_hatch`)
- 无法 learn pulling motion
- Root cause:pulling door 需要 *simultaneous* arm pull + whole-body backward locomotion
- 这是 **multi-body coordination** 问题,需要 locomotion 和 manipulation 的时序耦合

**hurdle 失败**:
- Robot 学到 forward velocity (5 m/s)
- 但撞 hurdle 而不 jump
- 因为 conservative collision 让 episode 不 terminate
- 这是 **reward shaping 的 unintended consequence**——termination condition 设计不当会让 agent 找 "卡 bug" 行为

参考类似 finding:[Lee et al., Composing Complex Skills, ICLR 2019](https://openreview.net/forum?id=rygrBhC5tQ)——Walker2d 的 forward reward 足以 learn jumping,但 humanoid 因为 termination condition 不同而 fail。

---

## 7. Hierarchical RL Approach (核心 methodological contribution)

### 7.1 Architecture Diagram

```
┌─────────────────────────────────────────────────┐
│ High-level policy π_H(s_t → goal_t)              │
│ Output: 3D target setpoint (for 1-hand) or       │
│         6D (for 2-hand, two 3D targets)          │
│ Trained: DreamerV3 / TD-MPC2                     │
│ Frozen: No (this is what we train)               │
└─────────────────┬───────────────────────────────┘
                  │ goal_t (3D or 6D target setpoint)
                  ↓
┌─────────────────────────────────────────────────┐
│ Low-level reaching policy π_L(goal_t → a_t)      │
│ Output: 19D joint position commands (body only)  │
│ Pretrained: PPO on MuJoCo MJX                    │
│ Frozen: YES (weights 固定 during high-level train)│
└─────────────────┬───────────────────────────────┘
                  │ a_t (19D joint position)
                  ↓
┌─────────────────────────────────────────────────┐
│ MuJoCo environment (full humanoid + hands)       │
│ Hands: fixed to 0 during hierarchical train      │
└─────────────────────────────────────────────────┘
```

### 7.2 Low-level Pretraining 细节

**为什么 PPO 在这里 work,而 flat PPO 在 benchmark 上 fail?**

两个关键 differences:
1. **Massive parallelization**:32,768 parallel environments on MuJoCo MJX
   - MJX 是 MuJoCo 的 JAX implementation,可以 JIT-compile 整个 physics step
   - PureJaxRL implementation (参考: [luchris424/purejaxrl](https://github.com/luchris424/purejaxrl))
2. **Simplified model**:只保留 feet-ground collisions,移除 hands,移除其他 body collision meshes
   - Full model 跑 1050 FPS,simplified (only feet) 跑 5100 FPS (Table IV)
   - Parallelization advantage 在 full model 上被 bottleneck 掉

**Training budget**:
- 1-hand reaching: 2 billion steps × 36 hours
- 2-hand reaching: 4 billion steps × 60 hours

**Robustness trick**:对 robot 的每个 link 施加 random force perturbations during training。Why?因为 high-level policy 在 exploration 时会输出 *drifting* targets,low-level policy 必须能 recover from external disturbances (包括来自环境的 contact force)。

参考相关工作:
- [Sutton, Precup, Singh - The options framework](https://arxiv.org/abs/cs/9905014)
- [Nachum et al. - Data-efficient HRL](https://arxiv.org/abs/1805.08296)
- [Option-Critic](https://arxiv.org/abs/1609.05140)

### 7.3 High-level Training 细节

Action space **restriction**:
> "To facilitate exploration, we restrict the range of reaching targets to the robot workspace."

这是 non-trivial 的——reaching policy 在 pretraining 时 target 是 unbounded 3D space,但 high-level 只在 robot reachable workspace 内 sample。这避免了 high-level 探索到 unreachable targets 时 low-level policy 的 undefined behavior。

**Mathematical formulation**:

Let `π_L: ℝ³ → ℝ¹⁹` (1-hand) 或 `π_L: ℝ⁶ → ℝ¹⁹` (2-hand) be the frozen low-level policy.

The high-level policy is `π_H: 𝒮 → 𝒜_H` where `𝒜_H ⊂ ℝ³` (or `ℝ⁶`).

The composite policy:
$$
a_t = \pi_L(\pi_H(s_t))
$$

High-level MDP 的 transition function:
$$
s_{t+1} = f_{\text{env}}(s_t, \pi_L(\pi_H(s_t)))
$$

这本质上是 [Sutton-Precup-Singh 的 options framework](https://arxiv.org/abs/cs/9905014) 的 special case,其中 option 是 deterministic 且 instantaneous (没有 option duration)。High-level decision 在 *every* environment step 都改变 low-level target,而不是 commit 到一个 option 执行 N steps。

### 7.4 Hierarchical Results 分析

**push task** (Figure 9a):
- Flat DreamerV3: -1251.9 avg (fails)
- Hierarchical DreamerV3: 1000.0 (solves)
- 这是 dramatic 的 improvement

**package task** (Figure 9b):
- 改善但 not solved
- Why? package 需要 *lift*——这不在 low-level reaching skill 覆盖范围内
- Reaching policy 输出 hand target position,但 lifting 需要 *grasp* + *upward force*——non-prehensile

**Key insight**:Hierarchical RL 依赖 low-level skill 覆盖 task 的 motor primitive set。如果 task 需要 skill outside low-level repertoire,hierarchical 反而 fail。

直觉:这就像让一个只会 walk 的人去做 powerlifting——locomotion primitive 不包含 lifting,必须学新 skill。

---

## 8. 与 Related Work 的深度连接

### 8.1 HRL 谱系

| Work | HRL type | Skill reuse |
|---|---|---|
| Options framework (Sutton 1999) | General framework | Theoretical |
| FuN (Vezhnevets 2017) | Learned options | - |
| Option-Critic (Bacon 2017) | End-to-end option learning | - |
| Relay Policy Learning (Gupta 2019) | Sequential skill composition | Kitchen tasks |
| Skill priors (Pertsch 2020) | Pretrained skill embeddings | - |
| **HumanoidBench HRL** | **Frozen pretrained reaching + high-level RL** | Cross-task |

### 8.2 Humanoid Learning 谱系

| Work | Focus | Approach |
|---|---|---|
| [Expressive Whole-Body Control (Cheng 2024)](https://arxiv.org/abs/2402.16796) | Locomotion + upper body pose tracking | RL + motion imitation |
| [Robot Parkour Learning (Zhuang 2023)](https://arxiv.org/abs/2309.05665) | Quadruped extreme locomotion | RL + distillation |
| [Bi-DexHands (Chen 2023)](https://arxiv.org/abs/2207.09252) | Bimanual dexterous | Multi-agent RL |
| [MyoSuite (Caggiano 2022)](https://arxiv.org/abs/2205.12900) | Musculoskeletal simulation | Muscle-actuated |
| [RoboPianist (Zakka 2023)](https://arxiv.org/abs/2304.04750) | Bimanual piano playing | RL on floating hands |
| **HumanoidBench** | **Whole-body manipulation + locomotion** | **Benchmark for flat + HRL** |

### 8.3 RL Algorithm Evolution Context

| Year | Algorithm | Key idea | Why it matters for HumanoidBench |
|---|---|---|---|
| 2018 | PPO | On-policy,clipped objective | Scales with parallelization (Isaac Gym) |
| 2018 | SAC | Maximum entropy off-policy | Standard for continuous control |
| 2023 | DreamerV3 | World model + latent rollout | Sample-efficient via imagination |
| 2024 | TD-MPC2 | World model + MPC planning | Combines learning + planning |

DreamerV3 在 HumanoidBench 上整体表现 best (Table V/VI),intuition 是:world model 可以"想象" long-horizon consequences,这对 humanoid 的 long-horizon tasks 有 natural advantage。但 world model 的 accuracy 在 75 DoF system 上 still 是 bottleneck。

参考: [TD-MPC2 paper](https://arxiv.org/abs/2310.16828), [DreamerV3 paper](https://danijar.com/project/dreamerv3/)

---

## 9. Engineering 细节与陷阱

### 9.1 MuJoCo MJX 的 trade-off

- **Pro**: GPU/TPU 上的 32,768× parallel envs → 100x training speedup
- **Con**: Complex collision geometry 会 bottleneck parallelization
- **Solution**: 用 simplified model (only feet collisions) pretrain,然后 transfer 到 full MuJoCo

这种 "sim-to-sim transfer" 是个有趣的 sub-problem——很多读者可能没注意到这其实是 zero-shot transfer between two simulators。如果 simplified policy 在 full sim 上 fail,就说明 simplified dynamics 漏掉了 critical physics。

### 9.2 Position Control 的 hidden assumption

Paper 用 50 Hz position control,但没讨论 MuJoCo 的 `<actuator>` 配置。default 上,position actuator 是:
```xml
<position joint="..." kp="..." kv="..."/>
```
其中 `kp` (P gain) 和 `kv` (D gain) 决定 actuator 的 stiffness 和 damping。这些 values 会显著影响 RL learning dynamics。如果 kp 太低,robot 会 floppy;太高会 oscillate。

这是一个 paper 没明说但 critical 的 implementation detail,需要去看 MuJoCo Menagerie 的 H1 model XML。

### 9.3 Reward Design 的 subtle issues

**Cabinet task 的 reward 复杂度**:4 个 subtask,每个有自己的 reward。Subtask 之间是 *sequential*——必须先开 sliding door,再开 drawer,再 move cube 等等。但 reward function 是 *state-dependent* 的:当前 subtask 决定 reward function。

这等价于一个 *compound MDP*,需要 algorithm 自动 discover subtask structure。Flat RL 在这里 fail 的 root cause 是 subtask transition 的 sparse signal。

参考: [Relay Policy Learning](https://arxiv.org/abs/1910.11905) 用 imitation learning 来 bootstrap subtask policies,HumanoidBench 的 baseline 没用这种 prior knowledge。

---

## 10. Open Problems & Future Directions (paper 没明说但 implied)

### 10.1 Multimodal Perception

Paper 实现了 vision 和 tactile 但没 benchmark。Vision + tactile fusion 在 [Sferrazza et al. 2023](https://arxiv.org/abs/2311.00924) 已经 shown promise,但 humanoid-scale 上未 test。直觉:whole-body tactile 对 whole-body manipulation (e.g., hugging, carrying) 可能 crucial——仅靠 vision 无法 solve occluded contact。

### 10.2 Sim-to-Real

Paper 提到:
> "the impressive results obtained via domain randomization in the newly developed MuJoCo MJX show promise to study sim-to-real transfer in more depth"

但 humanoid sim-to-real 远比 quadruped 难 ([Hwangbo et al. 2019](https://www.science.org/doi/10.1126/scirobotics.aau5872))。原因:
1. 75 DoF 的 actuator dynamics modeling error 累积
2. Dexterous hand 的 friction/contact 在 sim 和 real 差异大
3. Humanoid falling 是 catastrophic,无法像 quadruped 那样 recover

### 10.3 Learning from Demonstrations

Paper 明确:
> "we focus on reinforcement learning algorithms because collecting physical demonstrations with humanoid robots is particularly challenging"

但 [Mobile ALOHA (Fu et al. 2024)](https://arxiv.org/abs/2401.02117) 显示 teleoperation + imitation learning 在 bimanual mobile manipulation 上可行。Humanoid teleop 数据 + diffusion policy 可能是 hierarchical RL 的 alternative 或 complement。

### 10.4 Foundation Models for Control

Paper 没提 but 很 natural 的方向:
- VLM 作为 high-level planner (e.g., SayCan, RT-2 style)
- LLM 作为 skill selector
- Video diffusion 作为 action prior

直觉:HumanoidBench 的 long-horizon tasks (kitchen, cabinet, bookshelf) 是 natural testbed for LLM-as-planner approaches。如果 LLM 能 decompose "open cabinet and put cube in" 成 subtask sequence,hierarchical RL 只需 solve 每个 subskill。

---

## 11. 一些 Carpathy-Style 的 Critical Observations

### 11.1 Benchmark saturation 程度

看 Table V,大多数 task 的 best algorithm 都 *远低于* target。这意味着 HumanoidBench 是个 *unsolved benchmark*——还有大量 algorithmic progress 空间。Compare:
- Atari: human-level 已 saturated
- MetaWorld: 大多数 task solved
- HumanoidBench: 27 个 task,只有 ~5 个 reach target

这是好 benchmark 的标志——有 *headroom* for research。

### 11.2 Action Space 是 Bottleneck 吗?

Ablation 显示 reduced action space (61→19) 大幅提速。这暗示:
- Future humanoid RL 可能需要 *action masking* 或 *action abstraction*
- 人类也不是每 step 控制 75 DoF——我们有 motor synergies (参考 [SAR: Synergistic Action Representation](https://arxiv.org/abs/2306.04786))
- Synergy-based action space 可能是 humanoid RL 的 key direction

### 11.3 Hierarchical RL 的 generalization 疑问

Paper 只在 push 和 package 上 test hierarchical。但 27 个 task 中很多 (basketball, window, highbar) 涉及 dynamic 全身 motion,reaching primitive 不够。Question:
- Hierarchical RL 在哪些 task 上 *根本无法* work?
- 需要什么样的 low-level skill library 才能 cover 所有 27 个 task?

直觉 answer:可能需要一组 primitives 包括 walk, reach, grasp, lift, push, throw, swing。如果每个 primitive 需要 2-4B steps pretrain,total pretraining cost 是数十亿 steps——这是 significant infrastructure requirement。

### 11.4 Reward Engineering 的 hidden labor

Paper 给的 reward functions 看起来 "clean",但实际上每个 task 都需要 careful tuning:
- `α_s = 1000` for push 的 success reward (为什么不是 100 或 10000?)
- `tol` 的 margin `m` 选择 (e.g., 0.4125 for height——为什么不是 0.5?)
- Sparse reward 的 magnitude (e.g., 1000 for subtask completion)

这些 magic numbers 是 reward engineering 的 art,不利于 systematic algorithm comparison。**A more rigorous benchmark 会用 automatic reward shaping (e.g., potential-based shaping) 或 learn reward from demos。**

### 11.5 Reproducibility concern

每个 baseline 跑 48 小时,3 个 seeds。如果加上 hierarchical 的 low-level pretraining (36-60 小时 + 32K parallel envs),total compute 是 significant。这对 academic lab 是 non-trivial 门槛。

参考: 论文 mention 用了 Google TPU Research Cloud。这对 most labs 不可得。

---

## 12. 公式速查总结

### 12.1 Tolerance Function

$$
\text{tol}(x, (x_{\text{lower}}, x_{\text{upper}}), m) =
\begin{cases}
1 & x \in (x_{\text{lower}}, x_{\text{upper}}) \\
\text{sigmoid-like decay} & \text{within margin } m \\
0 & \text{beyond margin}
\end{cases}
$$

变量:
- `x`: evaluated scalar value
- `x_lower, x_upper`: target bounds
- `m`: margin, slope controller

### 12.2 Standing Composite

$$
\text{stand} = \text{height} \times \text{upright}
$$

- `height`: 头部高度 reward (default z_head ≥ 1.65)
- `upright`: 躯干垂直度 (default z_proj ≥ 0.9)

### 12.3 Effort Penalty

$$
e = 0.2 \cdot \left| 4 + \frac{1}{|u|} \sum_i \text{tol}(u_i, (0, 0), 10) \right|
$$

- `u`: actuator input vector (|u| = 61 for full system)
- `tol(u_i, ...)`: per-actuator low-effort indicator
- Sum average over actuators

### 12.4 Stable Composite

$$
\text{stable} = \text{stand} \times e
$$

### 12.5 Walk Reward

$$
R(s, a) = \text{stable} \times \text{tol}(v_x, (1, +\infty), 1)
$$

### 12.6 Push Reward

$$
R(s, a) = \alpha_s \cdot \mathbb{1}_{d_{\text{goal}} < 0.05} - \alpha_t \cdot d_{\text{goal}} - \alpha_h \cdot d_{\text{hand}}
$$

- `α_s = 1000, α_t = 1, α_h = 0.1`
- `d_goal`: box to destination distance
- `d_hand`: box to left hand distance

### 12.7 Hierarchical Policy Composition

$$
a_t = \pi_L(\pi_H(s_t))
$$

- `π_H: 𝒮 → 𝒜_H` (high-level,outputs 3D/6D target)
- `π_L: 𝒜_H → 𝒜` (low-level frozen,outputs 19D joint command)
- `a_t`: actual MuJoCo action

---

## 13. Takeaways for Building Intuition

1. **高维 action space 是 RL 的 enemy**:即使 task 不需要某些 DoF (e.g., hands for walk),他们的存在就 ruin exploration。Action abstraction / synergy 是 must-have direction。

2. **Reward engineering 仍然 art**:尽管 modular design 优雅,但 magic numbers 处处都是。Automated reward shaping 或 RLHF-style preference learning 是 future。

3. **Hierarchical RL works but 局限性大**:只能在 low-level skill repertoire 覆盖的 task 上 work。覆盖 broader task set 需要 diverse skill library。

4. **Massive parallelization 是必要条件**:PPO 在 standard setup fail,但在 MJX 上 32K envs 可以 pretrain robust reaching。Compute 不再 optional。

5. **Benchmark 是 unsolved,有 headroom**:27 task 中大多数没 reach target,意味着 humanoid robotics algorithm research 还在 early phase。

6. **Tactile/vision 的 role 还没 explore**:paper 实现了 multimodal sensing 但没 benchmark。Whole-body tactile 可能在 contact-rich manipulation 上 crucial,等待 future work。

7. **Mechanical simplicity 是 RL-friendliness 的 factor**:H1 比 Digit 更 RL-friendly,因为 simpler kinematics。这对未来 humanoid hardware design 有 implication。

---

## 14. 一些值得 Follow-up 的 Directions

1. **Skill discovery on HumanoidBench**:用 [DIAYN](https://arxiv.org/abs/1802.06070) 或 [Option-Critic](https://arxiv.org/abs/1609.05140) 自动 discover motor primitives,而不是 hand-design "reaching"。

2. **Diffusion Policy on HumanoidBench**:[Diffusion Policy (Chi et al. 2023)](https://diffusion-policy.cs.columbia.edu/) 在 bimanual 上 shown good,但 high-DoF humanoid 上未 test。

3. **VLM + hierarchical**:[RT-2](https://arxiv.org/abs/2307.15818) style VLM 作为 high-level planner + HumanoidBench trained low-level skills。

4. **World model scaling**:DreamerV3 在 61D action 上 work partially,但 world model 的 latent dynamics 是否 capture 全身 coordination 是 open question。

5. **Cross-embodiment transfer**:能否把 H1 上训的 skill transfer 到 G1 或 Digit?这 test skill 的 embodiment-invariance。

6. **Real-time tactile feedback for manipulation**:cube, basketball 这种 contact-rich task 是否真正需要 tactile 才能 solve?

---

希望这个 walkthrough 给你足够 technical depth 来 build intuition,Andrej。如果你想 deep dive 任何 specific task reward 或 algorithm implementation detail,我可以继续展开。

参考资源汇总:
- 项目主页: https://humanoid-bench.github.io
- MuJoCo: https://mujoco.org
- MuJoCo MJX: https://github.com/google-deepmind/mujoco_mjx
- MuJoCo Menagerie: https://github.com/google-deepmind/mujoco_menagerie
- Unitree H1: https://www.unitree.com/h1
- Shadow Hand: https://www.shadowrobot.com/products/dexterous-hand/
- DreamerV3: https://arxiv.org/abs/2301.04104
- TD-MPC2: https://arxiv.org/abs/2310.16828
- SAC: https://arxiv.org/abs/1801.01290
- PPO: https://arxiv.org/abs/1707.06347
- PureJaxRL: https://github.com/luchris424/purejaxrl
- Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3
- CoACD: https://github.com/SpatialComputing/CoACD
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu
- Mobile ALOHA: https://arxiv.org/abs/2401.02117
- Expressive Whole-Body Control: https://arxiv.org/abs/2402.16796
- Robot Parkour Learning: https://arxiv.org/abs/2309.05665
- Relay Policy Learning: https://arxiv.org/abs/1910.11905
- Bi-DexHands: https://arxiv.org/abs/2207.09252
- MyoSuite: https://arxiv.org/abs/2205.12900
- RoboPianist: https://arxiv.org/abs/2304.04750
- Options framework: https://arxiv.org/abs/cs/9905014
- Option-Critic: https://arxiv.org/abs/1609.05140
- DIAYN: https://arxiv.org/abs/1802.06070
- DM Control Suite: https://arxiv.org/abs/1801.00690
- FurnitureBench: https://arxiv.org/abs/2305.06036
- Robosuite: https://robosuite.ai
- RLBench: https://github.com/stepjam/RLBench
- MetaWorld: https://arxiv.org/abs/1910.10897
- BEHAVIOR-1K: https://arxiv.org/abs/2203.03044
- Isaac Gym: https://developer.nvidia.com/isaac-gym
