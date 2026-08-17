---
source_pdf: Learning Multi-Agent Collaborative Manipulation for Long-Horizon.pdf
paper_sha256: 281a60614a97e54d866553ab28f9b52d813153ebe7421350366a256bb43c1674
processed_at: '2026-08-05T13:26:06-07:00'
target_folder: Motor-control
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇 paper

## 这篇到底在干嘛

让两到四只 Unitree Go1 机器狗**合作推一个大物体穿过有障碍的房间**。物体比单只狗还重还大 (4-10kg, Go1 payload 才 5kg), 所以必须多狗协作。任务距离 10m+ 算 long-horizon, 中间有障碍物要绕。

就这么个事。但这件事之前没人做成过。Table I 里四个 box 全勾的只有这篇。

## 为什么这件事难

推东西这个动作本身就糟心:

1. **接触点不固定** — 你推 cuboid 一个面和推它一个角, 物体运动方向完全不一样。RL 从头学这个 mapping 很慢, 因为 action 和 outcome 之间是高度 nonlinear 的 contact dynamics。
2. **多狗协作有合成谬误** — 每只狗各自做"对的事", 合起来可能完全不动。比如两只狗站在 cuboid 两边对着推, 谁也没错, 但物体原地不动。
3. **Long-horizon reward 稀疏** — 推 10m 远, 中间 99% 的 action 都拿不到"到达目标"的 reward, RL 学不到东西。
4. **障碍物让几何可行路径≠物理可行路径** — RRT 能算出一条不碰障碍的路, 但推物体时物体本身有 size, 狗也要占空间, 物体 yaw 一偏 footprint 还会变, RRT 管不了这些。
5. **Sim-to-real** — 真机推重物时 low-level velocity tracking 会失准, learned low-level policy 在 sim 里 ok 上真机就拉胯。

这五个问题叠一起, 之前的工作每个都只解了其中一两个。

## 他们的解法: 三层 + 一个 trick

### 三层 hierarchy

**最底层**: 一只狗怎么走路。输入是"我要你以 v_x, v_y, v_yaw 速度走", 输出是 12 个关节的 motor command。这层**直接用 Unitree 出厂的 controller**, 不自己训。理由是出厂的 robust, 自己训的在推重物时 sim-to-real 崩。

**中间层**: 一只狗怎么推物体到某个 subgoal。输入是"物体在哪, 障碍在哪, 其他狗在哪, subgoal 在哪"(都在自己 local frame), 输出是 "我该以什么速度走"。**所有狗 share 同一个 policy 网络**。用 MAPPO 训练, CTDE。

**最高层**: 整个团队下一步该把物体推到哪。输入是地图 + 所有狗状态 + 物体状态 + RRT 给的参考路径, 输出是物体的下一个 2D subgoal (所有狗共享这个 subgoal)。用 PPO 训练, centralized。

数据流:

```
RRT (episode 开始跑一次) → 参考路径
       ↓
High-level (50Hz) → 物体 subgoal (所有狗共享)
       ↓
Mid-level (50Hz, 每只狗独立) → 每只狗的 (v_x, v_y, v_yaw)
       ↓
Low-level (50Hz, 真机用出厂, sim 用 WTW imitate) → motor command
       ↓
PD (2000Hz) → 关节 torque
```

### 一个关键 trick: OCB reward

这是我觉得整篇最 smart 的地方。

问题: 中间层要学"站在物体哪一侧推"。这是个 combinatorial 问题, RL 从头学很慢。

Insight: 如果你**看不到 subgoal** (被物体挡住), 说明你站在物体的**远端**那一侧, 这时你只要往物体方向推, 物体自然往 subgoal 方向走。这是几何常识, 可以用 closed-form 算出来。

Formulation: 找物体 convex hull 上离你最近的点, 算那个点的内法向量 $\vec{v}_i$ (指向物体内部)。再算从物体到 subgoal 的方向 $\vec{v}_{\mathrm{target}}$。两个向量点积就是 OCB reward。

- 同向 (你站在远端) → 正 reward → 鼓励
- 反向 (你站在 subgoal 同侧) → 负 reward → 惩罚

这把"选 contact point"这个 combinatorial 难题, 降维成一个 dot product 的 reward shaping。灵感来自 2015 年一篇 swarm robotics 的工作 (<https://ieeexplore.ieee.org/document/7052352>), 那篇是 hard-coded controller, 这篇 distilled 成 reward。

## 三个反直觉的设计

### 1. High-level 和 mid-level 同频 50Hz

经典 HRL 里 high-level 通常稀疏触发 (每 k 步一次), 这里 dense 触发。后果是 high-level 学的不再是 "macro planning", 而是 "在小邻域内 deviate from RRT path"。

这其实很合理: RRT 已经给了 macro path, high-level 只需要做 local correction, 没必要稀疏。这和 LLM 里 "planner 给骨架, executor 每步 refine" 的模式类似。

### 2. Subgoal reaching threshold = 1.0m

故意设这么大, 不让 policy 精确追中间 subgoal。如果 threshold = 0.1m, policy 会陷入对每个 intermediate subgoal 的微操, 整个 long-horizon 任务 timeout。

直觉: subgoal 是**方向 guidance**, 不是**精确 target**。就像开车导航, 你不需要精确到达每个 waypoint, 大致方向对就行, momentum 重要。

### 3. 真机用出厂 controller, sim 用 learned policy

真机部署: Unitree built-in controller (robust 但 CPU-only, 不能并行训)
Sim 训练: WTW-based learned policy (imitate built-in 的 behavior, GPU 4096 envs)

这是把 **training throughput** 和 **deployment robustness** 解耦。训练时需要 GPU 并行加速, 但 learned low-level 在真机推重物时 sim-to-real 崩; 真机需要 robust controller 但它没法在 GPU 上并行跑。Solution: 让 sim policy 模仿真机 controller 的 behavior, 训练用 sim 版本, 部署用真机版本。

## 实验结果人话版

主结果 (Table II):

- 他们的方法在三种物体上 SR 都 60%+
- Single agent 几乎全失败 (4.5%, 21.2%, 0%) — 证明多狗必要
- H+L (给每只狗各自 subgoal) 比 M+L 还差 — robot-centric subgoal 让狗各自走各自的, 合力抵消, 物体不动
- H+L FT (fine-tuned reward) 在 Cylinder 上 56% 很强, 但 T-Shape 7.3% 崩 — Cylinder 轴对称没方向歧义, T-Shape 有方向必须靠 mid-level
- 他们的方法 std 小, baseline std 巨大 — hierarchy + RRT prior 带稳定性

OCB ablation (Table III):

- w/o OCB 在 timeout 20s → 40s 时, SR 13% → 19% (+6%)
- w/ OCB 在 timeout 20s → 40s 时, SR 47% → 74% (+27%)

OCB 不是"给更多时间能补的", 它改变了 policy 的 adaptability — object 偏离预想方向后, OCB-trained policy 能调整 push 角度, w/o OCB 只会僵直推。

Real-world: 在 7.5×7.7m 房间用 2 只 Go1 + 24 个 OptiTrack camera, 成功完成 Push-T 和 Push-Cuboid (有障碍)。Push-T 时两狗始终从 T 两端推 (增大 contact surface + 方向控制), Push-Cuboid 时动态切换 forward push 和 side push。

## 这篇为什么 work

**不是因为某一个 algorithmic breakthrough, 而是 systems engineering**:

1. RRT 给 geometric skeleton (解决 long-horizon reward 稀疏)
2. RL 学 physical deviation (解决 RRT 不考虑 dynamics)
3. OCB reward 把 contact point selection 降维 (解决 combinatorial exploration)
4. Shared mid-level policy 借 homogeneous agent 对称性加速训练 (解决 sample efficiency)
5. 大 subgoal threshold 让 long-horizon 不被中间 goal 卡死 (解决 local optima)
6. Sim learned policy + Real built-in controller 解耦 (解决 sim-to-real)

每个 trick 单独都不算新, 但组合在 quadrupedal multi-robot pushing 这个 niche 上达到了 SOTA。

## 这篇没解决的

1. **Object yaw 不能控** — subgoal 只有 2D position, object orientation 是被动 result of contact
2. **No onboard perception** — 用 24 个 motion capture camera, 真要 deploy 需要 RGB-D / LiDAR
3. **4+ dogs 退化** — collision risk 增加, coordination 反而变差
4. **Object shape 不泛化** — cuboid / T / cylinder 各训一个 policy
5. **Static obstacles only** — 没 dynamic obstacle
6. **RRT prior 质量假设** — 如果 RRT path 完全不合理 (narrow passage), adaptive policy 在分布外可能 fail

## 我的 take

这篇 paper 的 contribution 主要是 **engineering integration**, 不是 algorithmic novelty。它把 RRT + hierarchical MARL + OCB reward + sim-to-real decoupling 这几个已有 idea 捏在一起, 解了一个 real-world 重要但之前没人解全的 task。

最有 idea-level 价值的部分是 **OCB reward** — 把 swarm robotics 的 geometric prior distilled 成 reward shaping, 这是一个可以泛化的 pattern。在任何 "contact point selection is combinatorial hard" 的 manipulation 任务里, 都可以问: **有没有 closed-form geometric prior 可以 inject 成 reward?**

最有 engineering 价值的部分是 **sim learned policy + real built-in controller 的解耦** — 这是一个可以泛化的 sim-to-real pattern。在任何 "learned low-level sim-to-real 不稳" 的场景, 都可以问: **能不能用 robust built-in controller 做 teacher, 让 sim student 模仿, 部署时用 teacher?**

Project page 有视频, 值得看看 real demo: <https://collaborative-mapush.github.io>

---

想深入哪个部分可以再展开, 比如 MAPPO 的 CTDE 细节, RRT 在 parallel sim 里为什么慢到必须用 random trajectory 替代, 或者 OCB reward 在 non-convex object 上怎么扩展。

---

# 这篇 Paper 的深度讲解

## 1. 30 秒电梯演讲

CMU + Google DeepMind 的工作，让多个 Unitree Go1 quadruped 协作推大型物体穿越有障碍的房间。核心是一个三层 hierarchical MARL 框架：**RRT planner 给 reference trajectory** → **centralized adaptive policy 输出 object-centric subgoal** → **decentralized MAPPO policy 把 subgoal 转成每个 robot 的 velocity command** → **WTW-based low-level locomotion 执行**。Project page: <https://collaborative-mapush.github.io>

这套设计本质上回答一个问题：long-horizon + multi-agent + contact-rich manipulation 三个难题同时出现时，怎么避免 reward 稀疏、credit assignment 爆炸、sim-to-real 崩溃？答案是**用 classical planner 做 prior、用 RL 学 deviation、用 CTDE 分解 multi-agent coordination、用 OCB reward 注入 geometric prior**。

---

## 2. 整体 Architecture 的 Intuition

### 2.1 三层 + 一个反直觉设计

```
┌──────────────────────────────────────────────────────────┐
│  High-Level (50 Hz)                                       │
│  RRT planner (once/episode) ──► τ_r                       │
│  + Adaptive Policy π_θ^h ──► a^h (object subgoal)         │
└────────────────────────┬─────────────────────────────────┘
                         │ a^h (共享给所有 robot)
┌────────────────────────▼─────────────────────────────────┐
│  Mid-Level (50 Hz, decentralized)                        │
│  π_φ^{m,i}: (o^{m,i}, a^h) ──► a^{m,i}=(v_x,v_y,v_yaw)   │
└────────────────────────┬─────────────────────────────────┘
                         │ velocity command
┌────────────────────────▼─────────────────────────────────┐
│  Low-Level (50 Hz policy, 200/2000 Hz PD)                │
│  π_φ^{l,i} ──► motor torques                              │
└──────────────────────────────────────────────────────────┘
```

**反直觉点**: high-level 和 mid-level 同频 50 Hz。在经典 HRL (HIRO、FuN、option-critic) 里 high-level 通常是稀疏 trigger (每 k 步)，这里改成 dense update，"adaptive" 才名副其实 —— 每帧都重新 select subgoal。代价是 high-level 学的其实是"在小邻域内 deviate from RRT path"，而不是 macro planning。

**另一个关键 trick**: RRT 在训练时被替换成"randomly generated long curved trajectory"。原因见 §3.3。

### 2.2 与 HRL literature 的位置

| 维度 | 经典 HRL (HIRO/FuN) | 这篇 |
|---|---|---|
| High-level 输出 | goal in latent space | object 2D pose |
| High-level 频率 | sparse (k-step) | dense (50 Hz) |
| Prior | 无 | RRT trajectory |
| Mid/Low 是否预训练 | 通常 end-to-end | stage-wise frozen |
| Multi-agent | 多为 single | CTDE (MAPPO) |

更接近 Nachum et al. 2019 "Multi-agent manipulation via locomotion using hierarchical sim2real" (<https://arxiv.org/abs/1908.05224>)，但加了 RRT prior 和 OCB reward。

---

## 3. 逐层公式与变量含义

### 3.1 Low-Level Controller

$$\pi_{\varphi}^{l,i}: \mathcal{O}^{l,i} \to \mathcal{A}^{l,i}$$

- 上标 $l$ = low-level
- 上标 $i$ = agent index
- $\varphi$ = policy 参数
- $\mathcal{O}^{l,i}$ = 第 $i$ 个 robot 的 low-level observation (joint pos, vel, IMU, last command…)
- $\mathcal{A}^{l,i}$ = motor commands (PD targets / torques)

Tracking target: $a^{m,i} = (v_x^i, v_y^i, v_{\mathrm{yaw}}^i)$ — torso 线速度 + yaw 角速度。

**关键工程选择**: 放弃 learning-based low-level (e.g. 原版 Walk-These-Ways), 改用 Unitree built-in controller。理由是 push 重物时 sim-to-real gap 太大, velocity tracking 失准。然后训练一个 WTW-based policy 去 **imitate** built-in controller 的行为, 这样 GPU 上能跑 4096 个并行 env 加速上层训练, 但真机部署时用 built-in。这是把 **sim training throughput** 和 **real deployment robustness** 解耦的实用 trick。

参考 WTW: <https://arxiv.org/abs/2212.03282>

### 3.2 Mid-Level Controller (decentralized MAPPO)

$$\pi_{\phi}^{m,i}: \mathcal{O}^{m,i} \to \mathcal{A}^{m,i}$$

输入 $o^{m,i}$ 包含 (全部在 robot $i$ 的 **local torso frame**):

- $s_{\mathrm{object}}^i = (x_{\mathrm{object}}^i, y_{\mathrm{object}}^i, \psi_{\mathrm{object}}^i)$ — object 2D 位置 + yaw
- $s_{\mathrm{obstacle}}^i$ — 障碍物 local 信息
- $\{s_j^i\}_{j=1, j \neq i}^N$ — 其他 robot 在本 robot frame 下的状态

输出 $a^{m,i} \sim \pi_\phi^{m,i}(a^{m,i} \mid o^{m,i}, a^h)$ — velocity command 给 low-level。

**关键设计**: 所有 robot **share 同一个** $\pi_\phi^m$ (parameter sharing)。理由: homogeneous agent + 类似 contact pattern → 共享参数加速训练、提升 sample efficiency。这和 MAPPO 在 cooperative setting 的 "surprising effectiveness" (Yu et al. 2022, <https://arxiv.org/abs/2103.01955>) 一致。

**优化目标**:

$$\mathcal{J}^m(\phi) = \mathbb{E}_{\tau \sim \rho_\pi}\left[\sum_{t=0}^{T} \gamma^t r^m\bigl(s_t, a_t^h, \{a_t^{m,i}\}_{i=1}^N\bigr)\right]$$

- $\phi$ = mid-level 参数 (注意 paper 里写成 $\theta$, 但应该是 $\phi$, 应为 typo)
- $\tau$ = trajectory
- $\rho_\pi$ = 由 $\pi_\phi^m$、初始分布 $\rho_0^m$ 和 transition $p^m$ 诱导的分布
- $\gamma$ = discount factor
- $r^m(\cdot)$ = mid-level reward
- $s_t$ = joint state (所有 robot + object)
- $a_t^h$ = high-level action (object subgoal, 共享给所有 agent)
- $\{a_t^{m,i}\}$ = 所有 agent 的 mid-level action

**Training trick**: $a^h$ 是 **randomly sampled subgoal**, 不是从 high-level policy 采的。这把 mid-level 训成 **goal-conditioned policy**, 与 high-level 解耦, 避免 joint training 的 non-stationarity。同时 low-level 是 frozen。这种 stage-wise 训练让 credit assignment 简化。

### 3.3 High-Level Controller (centralized PPO + RRT)

两个 component:

**(1) RRT planner**: $\mathcal{P}: \mathcal{M} \times \mathcal{G} \to \mathcal{T}$

- $\mathcal{M}$ = map 信息 (obstacle 位置, object 初始位置)
- $\mathcal{G}$ = object 目标空间
- $\mathcal{T}$ = trajectory 空间

输入: $g_{\mathrm{object}} \in \mathcal{G}$, $p_{\mathrm{map}} \in \mathcal{M}$
输出: $\tau_r \in \mathcal{T}$ — reference trajectory (几何可行, 不考虑 pushing dynamics)

**(2) Adaptive policy**:

$$\pi_\theta^h: \mathcal{M} \times \mathcal{T} \times S_{\mathrm{object}} \times S_1 \times \cdots \times S_N \to \mathcal{A}^h$$

输入: $g_{\mathrm{object}}, p_{\mathrm{map}}, \{s_i\}_{i=1}^N, s_{\mathrm{object}}, \tau_r$
输出: $a^h \sim \pi_\theta^h(a^h \mid \cdots)$ — object subgoal position

**优化目标**:

$$\mathcal{J}^h(\theta) = \mathbb{E}_{\tau \sim \rho_\pi}\left[\sum_{t=0}^{T} \gamma^t r^h\bigl(g_{\mathrm{object},t}, p_{\mathrm{map},t}, s_{\mathrm{object},t}, \{s_{i,t}\}_{i=1}^N, a_t^h, \tau_r\bigr)\right]$$

变量含义同 mid-level, 仅 reward 函数换成 $r^h$。训练时 **freeze mid-level + low-level**。

**关键 training trick**: RRT 在 massively parallel IsaacGym 中开销太大, 所以训练时用一个 **randomly generated long curved trajectory** 替代 RRT path, 起止点和 task 一致。obstacles 随机放置在 trajectory 周围 4m 宽 strip 内。这意味着:

> Adaptive policy 学的是 "在 reference 轨迹附近 deviate 以避障" 的能力, 而非从 scratch 学 planning。

这让训练稳定, 但要求 RRT 在 deployment 时提供合理 prior。换句话说, **RRT 给 baseline, RL 给 correction**。这和 LLM 里 "model + RLHF refinement" 类似: 先有粗略 ground truth, 再学小幅修正。

参考 RRT: <http://msl.cs.uiuc.edu/~lavalle/papers/Lav98c.pdf>

---

## 4. Reward Design 详解 (核心 contribution 之一)

### 4.1 Mid-Level Reward

$$r^m = r_{\mathrm{task}}^m + r_{\mathrm{penalty}}^m + r_{\mathrm{heuristic}}^m$$

#### (a) $r_{\mathrm{task}}^m$

| 项 | 表达式 | Weight |
|---|---|---|
| Subgoal Reaching | $\mathbb{1}(\text{reach subgoal})$ | 10 |
| Subgoal Approaching | $\alpha(d_{\mathrm{subgoal},t-1} - d_{\mathrm{subgoal},t}) - d_{\mathrm{subgoal},t}$ | 3.25e-3 |

- $d_{\mathrm{subgoal},t}$ = object 当前位置到 subgoal 的 2D 距离, 时刻 $t$
- $\alpha = 200$ = delta distance 系数
- 第一项鼓励 progress, 第二项惩罚绝对距离 (防止 "approach 但不动")
- Reaching threshold = 1.0m, 故意设大, **防止 policy 陷入对中间 subgoal 的 fine-grained 微操**。这是一个非常重要的 long-horizon trick: subgoal 是 guidance 不是 target

#### (b) $r_{\mathrm{penalty}}^m$

| 项 | 表达式 | Weight |
|---|---|---|
| Exception Avoidance | $\mathbb{1}(\text{exception})$ | -5 |
| Collision Avoidance | $\sum_{j \neq i}^N \frac{1}{0.02 + d_{i,j}/3}$ | -2.5e-3 |

- $d_{i,j}$ = agent $i$ 和 $j$ 的当前距离
- Exception = robot fall-over / timeout
- Collision 项形式有意思: $1/(0.02 + d/3)$, 当 $d \to 0$ 时趋 $50$, $d$ 大时趋 0。这是个 soft penalty, 距离越近惩罚越大但不会爆炸

#### (c) $r_{\mathrm{heuristic}}^m = r_{\mathrm{approach}}^m + r_{\mathrm{vel}}^m + r_{\mathrm{OCB}}^m$

**Object Approaching**: $-(d_{\mathrm{object},i} + 0.5)^2$, weight 7.5e-4

- $d_{\mathrm{object},i}$ = object 到 agent $i$ 距离
- +0.5 防止贴着时 reward 退化成 0 附近, 给一个"接近就好"的 baseline

**Object Velocity**: $\mathbb{1}(v_{\mathrm{object}} > 0.1)$, weight 1.5e-3

- 鼓励 object 在动, 防止 agent 在 object 附近 oscillation
- threshold 0.1 m/s

**OCB Reward** (核心创新):

$$r_{\mathrm{OCB}}^{m,i} = \vec{v}_i \cdot \vec{v}_{\mathrm{target}}$$

- $\vec{v}_i$ = **object convex hull 上离 robot $i$ 最近的点处的单位内法向量** (指向 object 内部)
- $\vec{v}_{\mathrm{target}}$ = **从 object 中心指向 subgoal 的单位向量**
- 点积范围 $[-1, 1]$, weight 4e-3

#### OCB 的 Intuition (这是我最想强调的)

想象一个 cuboid 和 subgoal 在它右边:

```
                subgoal →
                ↓ v_target
   robot i ←──[cuboid]──→
        ↓ v_i (向右, 指向 object 内部, 与 v_target 同向)
```

如果 robot 站在 **object 的左侧**, convex hull closest point 处的内法向量指向 **右**, 与 $\vec{v}_{\mathrm{target}}$ 同向 → 点积正 → 鼓励。这恰好是"看不到 subgoal 也知道往哪推"的几何 prior: **站在你这一侧, 推你身边的那块, object 自然往 subgoal 方向走**。

如果 robot 站在 object 右侧 (subgoal 同侧), 内法向量指向左, 与 $\vec{v}_{\mathrm{target}}$ 反向 → 点积负 → 惩罚。这告诉 robot "别从这一侧推, 你会推反方向"。

灵感来源: Chen et al. 2015, "Occlusion-based cooperative transport with a swarm of miniature mobile robots", <https://ieeexplore.ieee.org/document/7052352>。这篇是 swarm biology-inspired 的工作, 这篇把它从 hard-coded controller 转成 reward shaping。

### 4.2 High-Level Reward

$$r^h = r_{\mathrm{task}}^h + r_{\mathrm{penalty}}^h$$

| 项 | 表达式 | Weight |
|---|---|---|
| Target Reaching | $\mathbb{1}(\text{reach final target})$ | 2 |
| Target Approaching | $\frac{1}{1 + d_{\mathrm{target}}}$ | 0.3 |
| Path Following | $\frac{1}{1 + d_{\mathrm{subgoal,path}}^h}$ | 0.5 |
| Exception Avoidance | $\mathbb{1}(\text{exception})$ | -0.5 |
| Obstacle Avoidance | $\frac{1}{1 + d_{\mathrm{obstacle}}}$ | -0.1 |

- $d_{\mathrm{target}}$ = object 到 final target 的 Euclidean distance
- $d_{\mathrm{subgoal,path}}^h$ = 当前 subgoal 到 RRT path 最近采样点的距离
- $d_{\mathrm{obstacle}}$ = object 到最近障碍物距离

**设计直觉**: Target Approaching (0.3) 让 object 总体往 final target 走。Path Following (0.5) 比 Target Approaching 略大, 保证 subgoal 不偏离 RRT 太远, 但允许 minor deviation 处理 push 复杂性。这是 "trust the planner, but verify" 的 reward 形式。

注意 Obstacle Avoidance 的 weight 是负的 (-0.1), 用 $\frac{1}{1+d}$ 的形式, $d \to 0$ 时趋 1, 即 penalty 强度变大; $d$ 大时趋 0。

---

## 5. 实验数据深入分析

### 5.1 主结果 (Table II)

| Task | SA | H+L | M+L | H+L FT | M+L FT | **Ours** |
|---|---|---|---|---|---|---|
| Cuboid SR | 4.5% | 0.23% | 10.5% | 41.0% | 24.3% | **77.5%** |
| T-Shape SR | 21.2% | 9.0% | 1.8% | 7.3% | 25.8% | **63.5%** |
| Cylinder SR | 0.0% | 3.0% | 3.0% | 56.0% | 26.9% | **71.2%** |

| Task | Best baseline CT | Ours CT | 相对减少 |
|---|---|---|---|
| Cuboid | 0.76 (H+L FT) | 0.66 | -13% |
| T-Shape | 0.80 (M+L FT) | 0.68 | -15% |
| Cylinder | 0.70 (H+L FT) | 0.48 | -31% |

整体 improvement: **+36% SR, -24.5% CT** 相对 best baseline。

**值得关注的 pattern**:

1. **SA (single agent) 几乎完全失败** — Cuboid 4.5%, Cylinder 0%。验证 multi-robot 是必须的, 单 Go1 payload 5kg 推不动 4-10kg object + 大尺寸 contact surface 不够。
2. **H+L 比 M+L 还差** — H+L 给每个 robot 各自 subgoal, robot 能到 subgoal, 但 object 不动 (因为 robot 之间互相抵消)。这暴露了 "robot-centric subgoal" 在 push 任务里的根本缺陷: subgoal 一旦分到每个 robot, 各自朝着 robot 自己的 subgoal 走, 不保证合力沿 object 目标方向。这是把 [23] (Nachum et al.) 直接搬到 pushing 任务上的痛点。
3. **M+L 在 T-Shape 上崩溃 (1.8%)** — 因为 T-Shape 接触面小, 没有 high-level guidance 的 long-horizon 几乎不可能稳定。FT 后 M+L 恢复到 25.8%, 说明 reward shaping 重要, 但仍远低于 full method 63.5%。
4. **H+L FT 在 Cylinder 上很强 (56%)** — 因为 Cylinder 没有方向歧义 (轴对称), 即便没有 mid-level 也能凑合。但 T-Shape 上崩到 7.3%, 说明方向性强的 object 必须有 mid-level pushing policy。
5. **Ours 在所有 task 上稳定** — std 都比较小 (Cuboid 3.0%, T-Shape 7.7%, Cylinder 5.1%), 而 baseline std 巨大 (M+L FT T-Shape 28.9%)。这是 hierarchical + RRT prior 带来的稳定性收益。

### 5.2 OCB Ablation (Table III)

| Timeout | Ours SR | Ours w/o OCB SR |
|---|---|---|
| 20s | 47.0% | 13.0% |
| 40s | 74.0% | 19.0% |

| Timeout | Ours CT | Ours w/o OCB CT |
|---|---|---|
| 20s | 14.9s | 18.3s |
| 40s | 22.5s | 34.9s |

**关键 insight**: w/o OCB 在 timeout 从 20s 增加到 40s 时, SR 只从 13% 涨到 19% (+6%)。Ours 从 47% 涨到 74% (+27%)。这说明 OCB 不只是"给更多时间"能补的, 它实际改变了 policy 的 **adaptability** — object 偏离预想方向后, OCB-trained policy 能调整 push 角度, 而 w/o OCB 只会僵直推。这是 reward shaping 改变 policy **结构** 而非只是把 reward landscape 拉平的经典例子。

### 5.3 High-Level Adaptive Policy Ablation (Fig 4)

只用 RRT (w/o adaptive policy): RRT path 经常贴着 obstacle, 不考虑 object shape 和 pushing dynamics, 容易碰撞。

加 adaptive policy: 提前 deviate, 绕开 obstacle, 然后回到 path。这是 "geometric feasibility ≠ physical feasibility" 的明证。RRT 只考虑物体中心点不碰 obstacle, 不考虑物体本身的 extent + 推动时 robot 也占空间 + 物体 yaw 偏转后 footprint 变化。

### 5.4 Scalability (Fig 5, Cylinder)

Robot 数量从 1 → 2 → 3, SR 和 CT 都显著提升。但 **4 个 robot 时 SR 反而下降**, 原因分析: collision risk 增加, robot 维持更大间距, coordination 退化。这是 multi-robot 经典的 "diminishing returns + negative returns" pattern。在 swarm literature 里通常通过 explicit formation control 或 communication 解决, 这里没有处理 4+ 的情况, 留作 future work。

### 5.5 Real-World 实验

Setup:
- 7.5m × 7.5m room
- 24 OptiTrack PrimeX 22 cameras (motion capture)
- 2 Unitree Go1 robots, payload ~5kg each
- Deploy high + mid policy (training in sim), low-level 用 Unitree built-in

Tasks:
- **Push-T**: 3.3kg T-block, target 在 x∈[3.5, 4.5]m, y∈[-4, 4]m
- **Push-Cuboid**: 6.8kg cuboid (1.5×1.0×0.5m), target 在 x∈[5.5, 6.5]m, y∈[-3.5, 3.5]m, obstacles 在 start-target 连线 ±2m band 内

**Observations (build your intuition)**:

**Push-T**:
- 两个 robot 始终从 T-block **两端** push, 这样增大 contact surface + 持续 forward force + 保持 directional control
- $v_x$ 几乎一直 hit 0.5 m/s 上限 (训练时被激励快速完成)
- Yaw command 先正 (转向), 转完后归零或负 (回正)
- $v_y$ 做小幅调整保持 contact 位置

**Push-Cuboid**:
- 多种 pushing strategy 动态切换:
  - 有时 both heads push forward (max speed)
  - 有时 one or both 转 side push (facilitate turning)
- $v_x$ 频繁 hit 0.5 m/s 上限
- $v_y$ 和 yaw rate 适应 strategy 切换

**Action Homogeneity Analysis**: 两个 robot 的 xy command 高度相似, 原因:
1. Shared mid-level policy → 参数同质
2. 都 contact 同一 surface → observation 相似
3. Yaw rate 处理 strategic adjustment, 让 xy 保持 homogeneous

这其实是 multi-agent parameter sharing 在 cooperative manipulation 里的副作用 — 当 agents 真的 symmetric, 共享 policy 让它们倾向于"对称"行为, 有时是好事 (协调 push), 有时限制 (无法 emergent 出 role differentiation)。

---

## 6. 更多联想与 Context

### 6.1 与 HRL Literature 的连接

- **HIRO** (Nachum et al. 2018, <https://arxiv.org/abs/1805.08296>): off-policy HRL, 学 goal-conditioned low-level + high-level via reward relabeling。这篇不用 off-policy, 用 on-policy PPO + stage-wise frozen, 简单但 effective。
- **FuN** (Vezhnevets et al. 2017): feudal networks, hierarchy 由 manager-worker 组成。这篇的 high/mid 类似 manager/worker, 但 high-level 直接输出 object subgoal 而非 latent goal。
- **Options framework** (Sutton, Precup, Singh): 这里 mid-level 可以看作一个 "option" — goal-conditioned policy with implicit termination (reaching threshold 1.0m)。

### 6.2 与 Multi-Agent Manipulation Literature

- **Nachum et al. 2019** (<https://arxiv.org/abs/1908.05224>): 多 agent 协作推物体, 高层给每个 robot subgoal。这篇把高层 subgoal 从 robot-centric 改成 **object-centric** 共享 — 这是关键改进。robot-centric subgoal 在 push 时容易让 robot 各自往自己 subgoal 走导致合力错乱。
- **Xiong et al. 2024** (MQE, <https://arxiv.org/abs/2403.16015>): benchmark MARL 在 quadruped, box pushing 上挣扎。这篇用 hierarchical + RRT 解决了 long-horizon 问题。
- **An et al. 2024** (<https://arxiv.org/abs/2402.18345>): permutation-invariant network 做 short-horizon multi-object pushing, 但没处理 long-horizon + obstacle。
- **Sombolestan & Nguyen 2023** (<https://arxiv.org/abs/2307.06000>): 用 adaptive control 但只能 head-push, 限制 contact pattern。这篇 whole-body + RL 解除限制。

### 6.3 OCB Reward 的更深层意义

OCB 是 **geometric inductive bias** 注入 reward 的范例。在 manipulation 里, contact point 选择本质上是个 combinatorial 问题, RL 从头学很慢。Chen et al. 2015 的 swarm 工作提供了一个 closed-form geometric prior, 这篇把它 distilled 成 reward shaping, 让 RL 在学 fine-grained contact dynamics 时有方向性。

类似 spirit 的工作:
- **Curiosity-driven exploration** (ICM, RND): 用 surprise 做 intrinsic reward
- **Hindsight relabeling** (HER): 用 achieved goal 重 label
- 这里是 **geometric prior as reward**: 用 closed-form heuristic prior 做 reward shaping

### 6.4 Sim-to-Real Trick 详解

这是工程上很 smart 的选择:

```
真机 deployment:
  robot → Unitree built-in low-level controller → motors
         (robust, but CPU-only, slow for parallel training)

Sim training:
  robot → WTW-based learned policy (imitating built-in) → motors
         (GPU parallel, 4096 envs)
```

通过 **behavior cloning** 让 sim low-level 模仿 real built-in 的 velocity tracking behavior, 训练时用 fast sim policy, 部署时用 real built-in。这是 decoupling **training throughput** 和 **deployment robustness** 的范式。和 "privileged learning" (Tan et al. 2018) 思路类似: 训练时用 teacher (privileged info), 部署时用 student (proprioception only)。

### 6.5 Long-Horizon 的 Subtlety

Subgoal reaching threshold = **1.0m** 看似很粗糙, 但非常重要。如果 threshold = 0.1m, policy 会陷入对每个 intermediate subgoal 的 fine-grained manipulation, 整个 long-horizon 任务 timeout。这是个反直觉的设计: **故意让 subgoal tracking 不精确, 反而提升 long-horizon 性能**。

类比: 在 LLM 里用 chain-of-thought, 每个 reasoning step 不需要完美, 大致方向对就好, 关键是 momentum。

### 6.6 Frequency Hierarchy 的设计哲学

| 层 | 频率 | 决策粒度 |
|---|---|---|
| RRT | 1×/episode | 全局几何 path |
| High-level | 50 Hz | Subgoal (object 2D position) |
| Mid-level | 50 Hz | Velocity command (v_x, v_y, v_yaw) |
| Low-level policy | 50 Hz | Motor targets |
| PD | 200 Hz (sim) / 2000 Hz (real) | Motor torques |

High 和 mid 同频是关键。如果 high 频率更低 (e.g. 5 Hz), subgoal 切换慢, 无法 react 到 object state 快速变化 (e.g. T-block yaw 偏转)。50 Hz 让 high-level 持续 fine-tune subgoal, 相当于 "micro-management" 而非 "macro-management"。

### 6.7 与 Loco-Manipulation 大图景

四足 loco-manipulation 的方法谱系:

```
Prehensile (有 gripper/arm):
├── ALMA (Bellicoso et al. 2019)
├── LoCoMan (Lin et al. 2024, https://arxiv.org/abs/2403.18197)
├── Deep Whole-Body Control (Fu et al. 2023)
└── UMI on Legs (Ha et al. 2024)

Non-prehensile (无 gripper):
├── Head-push only
│   ├── Sombolestan & Nguyen 2023 (hierarchical adaptive control)
│   └── Rigo et al. 2023 (hierarchical MPC)
├── Leg manipulation
│   ├── Circus Anymal (Shi et al. 2021)
│   ├── DribbleBot (Ji et al. 2023)
│   └── Pedipulate (Arm et al. 2024)
├── Whole-body
│   └── Jeon et al. 2023 (single robot)
└── Multi-robot
    ├── Nachum et al. 2019 (sim2real, head-only)
    ├── Sombolestan et al. 2023 (head-only, adaptive control)
    ├── An et al. 2024 (permutation-invariant, short-horizon)
    └── THIS WORK (whole-body + long-horizon + obstacle)
```

这篇的 niche: **同时勾选 collaborative + long-horizon + whole-body + obstacle-aware 四个 box**, 是 Table I 里第一个 √√√√ 的工作。

### 6.8 Limitations 推断

Paper 没明说但可推断的 limitations:

1. **No generalization to unseen object shapes**: 三种 object (cuboid, T, cylinder) 都有专门训练的 policy。换 L-shape 或 irregular 需要重训。看 Table IV 三个 object 训练 setup 都不一样。
2. **Motion capture dependency**: real-world 实验用 24 个 OptiTrack camera, 不是 onboard sensing。Sim2real 真正 deploy 需要替换为 onboard perception (RGB-D, LiDAR)。
3. **4+ robots 退化**: scalability 到 4 已经不行, swarm-level (10+) 没探索。
4. **No object yaw control**: subgoal 只是 2D position, 不控制 object orientation。Push-T 里 yaw 是被动 result of contact, 不是 active target。
5. **Static obstacles only**: 没处理 dynamic obstacles / other agents。
6. **RRT path prior quality 假设**: 如果 RRT path 完全不合理 (e.g. 在 narrow passage), adaptive policy 在训练分布外可能 fail。

### 6.9 可能的扩展方向

- **Object yaw-aware subgoal**: $a^h = (x, y, \psi)$ 而非 $(x, y)$, 可控 orientation。但 mid-level action space 也要扩, contact point selection 更复杂。
- **Onboard perception**: 用 depth image 替代 motion capture, 加 visual encoder。可以借鉴 Visual-Locomotion (Yu et al. 2021, <https://arxiv.org/abs/2107.04034>) 思路。
- **Heterogeneous robots**: Go1 + Go2 + ANYmal 混合, parameter sharing 失效, 需要 agent-specific policy + communication。
- **Dynamic obstacle avoidance**: 把 obstacles 当作动态 agent, 在 high-level state 里加 obstacle velocity。
- **Contact-aware subgoal**: 让 high-level 不仅输出 position subgoal, 还输出 "推荐 contact region" 给 mid-level, 降低 mid-level 探索难度。

### 6.10 与 LLM/Transformer 的潜在 cross-pollination

- 这个三层 hierarchy 类似 LLM 里的 **planner + tool-use + execution**: high-level = planner (生成 subgoal = "API call"), mid-level = API interface, low-level = tool execution。
- OCB reward 作为 geometric prior, 类似 LLM 里的 **process reward model (PRM)** — 在中间步骤注入 prior 信号而非只在 final outcome。
- Stage-wise frozen training 类似 **pre-training + instruction tuning + RLHF** 三阶段, 每阶段 freeze 之前的部分。
- Shared mid-level policy 类似 **MoE 里的 expert sharing** — 不同 agent 共享同一个 expert 因为 task 是同质的。

---

## 7. 关键 References (web links)

| 文献 | 链接 | 用途 |
|---|---|---|
| 本文 project page | <https://collaborative-mapush.github.io> | 视频 + code |
| MAPPO (Yu et al. 2022) | <https://arxiv.org/abs/2103.01955> | mid-level CTDE 训练 |
| PPO (Schulman et al. 2017) | <https://arxiv.org/abs/1707.06347> | high-level 训练 |
| Walk-These-Ways (Margolis & Agrawal 2023) | <https://arxiv.org/abs/2212.03282> | low-level locomotion framework |
| Nachum et al. 2019 (multi-agent sim2real) | <https://arxiv.org/abs/1908.05224> | H+L baseline 出处 |
| HIRO (Nachum et al. 2018) | <https://arxiv.org/abs/1805.08296> | HRL 经典 |
| Chen et al. 2015 (OCB swarm) | <https://ieeexplore.ieee.org/document/7052352> | OCB reward 灵感 |
| RRT (LaValle 1998) | <http://msl.cs.uiuc.edu/~lavalle/papers/Lav98c.pdf> | high-level planner |
| IsaacGym | <https://arxiv.org/abs/2108.10470> | 仿真环境 |
| Xiong et al. 2024 (MQE) | <https://arxiv.org/abs/2403.16015> | M+L baseline 出处 |
| An et al. 2024 (permutation-invariant) | <https://arxiv.org/abs/2402.18345> | M+L baseline 出处 |
| Jeon et al. 2023 (whole-body manipulation) | <https://arxiv.org/abs/2306.00510> | single-robot whole-body 推 |
| Unitree Go1 | <https://www.unitree.com/products/go1/> | 硬件平台 |

---

## 8. 一句话 Intuition 总结

这篇的成功 = **RRT 提供 geometric skeleton + adaptive RL 学 physical deviation + OCB reward 把 contact point selection 的 combinatorial 难题简化为 dot product + shared mid-level policy 借 homogeneous agent 对称性加速训练 + 大 subgoal threshold 让 long-horizon 不被中间 goal 卡死**。每一个 trick 单独都不算新颖, 但组合在 quadrupedal multi-robot pushing 这个 niche 上达到了 SOTA, 是 **systems engineering over algorithmic novelty** 的范例。
