---
source_pdf: AMOR Adaptive Character Control through Multi-Objective Reinforcement
  Learning.pdf
paper_sha256: af03c1ae17f59153ecf8eee6728c92d1d930f6bab466ab5dda3d23f99b0f8ea6
processed_at: '2026-08-18T00:53:32-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 AMOR

## 一句话版本

**别在训练前调 reward 权重了,训练完之后随便调,实时生效。**

---

## 为什么需要这东西?

假设你教一个机器人跳舞。你给它设计了一堆"打分项":
- 上半身跟得准不准 (upper body tracking)
- 下半身跟得准不准 (lower body tracking)
- 脚的动作对不对 (feet tracking)
- 整体姿态对不对 (root pose)
- 速度快慢对不对 (velocity)
- 动作顺不顺滑 (smoothness)

这 7 个打分项很多时候是**打架的**。比如你要跳一个很炸的舞,动作幅度大,tracking 就很难精准;你要动作丝滑,tracking 就得放松一点。

传统做法是:你给每个打分项设个权重 $w_i$,加起来算总分,然后用 RL 去优化这个总分。问题是:
- 权重设多少? **不知道,得试**
- 试一次 = 重新训练 = **5 天**
- 改一个数重来,再改一个数再重来
- 训练里表现好的权重,搬到真机上一跑,**抖成狗** (sim-to-real gap)
- 跳舞和走路需要的权重还不一样,一个固定权重照顾不过来

这就是折磨做 character control 的人的问题。AMOR 说:**别调了,我帮你一次搞定。**

---

## AMOR 怎么做的?

核心 idea 非常简单,甚至有点"作弊"的感觉:

**把权重 $\mathbf{w}$ 也塞进 policy 的输入里。**

传统 policy:
$$\pi(\text{action} \mid \text{state})$$

AMOR policy:
$$\pi(\text{action} \mid \text{state}, \text{reference\_motion}, \textbf{weights})$$

训练的时候,每个 episode 随机扔一组权重进去 (从 simplex 上均匀采样),让 policy 学会"**给什么权重,就表现出对应的 trade-off 行为**"。

训练完了之后,你想让机器人更顺滑一点? 把 smoothness 权重调高,实时看到效果。想更精准一点? 调高 tracking 权重。**不用重新训练,零等待。**

这就好比你训练一个厨师,不是训练他只会做一道菜,而是训练他"你说要咸一点他就多放盐,你说要辣一点他就多放辣椒"。一次训练,终身定制。

---

## 为什么这能 work?

关键在于:不同权重下的最优行为,本质上是**同一个动作的不同版本**。

跳同一个舞,权重 A 下是"精准但有点抖"的版本,权重 B 下是"顺滑但稍微偏一点"的版本。这俩版本之间是**连续过渡**的,不是完全不同的技能。所以一个 network 可以学会 mapping: 权重 → 对应风格的行为。

数学上这靠的是 Pareto front 的 convexity —— 所有最优 trade-off 点构成一个凸面,你用线性组合 (就是加权求和) 就能 sweep 整个面。AMOR 学的就是这个面的 parameterization。

---

## 7 个打分项的设计直觉

这 7 个 reward terms 不是随便选的,它们对应不同"粒度"的关注点:

| 层次 | 打分项 | 关心什么 |
|------|--------|----------|
| 细 | 上半身关节 / 下半身关节 / 脚 | 局部姿态准不准 |
| 中 | 末端执行器 (手和脚) 的全局位姿 | 手脚在空间中位置对不对 |
| 粗 | 根 (躯干) 的朝向和速度 | 整体动态对不对 |
| 品质 | smoothness (力矩 / 加速度 / action 变化率) | 动作自不自然 |

这种分层设计让冲突自然冒出来: 你要让上半身精准,下半身可能就得妥协,因为 reference motion 本身可能是物理上不可行的 (mocap 数据 retarget 到机器人上有 artifact)。

---

## 真机实验最说明问题

他们在一个 20-DoF 的双足机器人上做了两个实验:

### 实验 1: 站着跳舞

用均匀权重 (所有 $w_i = 1/7$) 跑真机,机器人**抖得很厉害**。把 smoothness 权重调高,抖动立刻减少,动作变顺滑了。这个过程**不用重新训练**,当场调当场看效果。

### 实验 2: 双圈旋转 (pirouette)

这个动作超出了之前 state-of-the-art 的 VMP 控制器的能力。AMOR 怎么搞定的? 用**时变权重**:
- 动作开始阶段: velocity 权重拉高,先攒够旋转速度
- 动作结束阶段: smoothness 权重拉高,让旋转停下来时顺滑过渡

这个调参过程花了大约 1 天。对比训练的 5 天,如果用传统方法每次调参都要重新训练,根本不可能在有限时间内搞定。

---

## HLP: 让 AI 自己调权重

手动调权重还是要人盯着,不优雅。于是有了 HLP (High-Level Policy):

- 低层是 AMOR (已经训练好,frozen)
- 高层是一个小网络,根据当前状态和动作上下文,**实时输出一组权重**给 AMOR 用

高层用什么训练? 用一个 discriminator (判别器) 当裁判:判别器看过真实 mocap 数据,判断 AMOR 生成的动作"像不像真的"。HLP 的目标就是选出让判别器最满意的权重组合。

这跟 AMP (Adversarial Motion Priors) 那套东西很像,区别是: AMP 直接用判别器 reward 训练低层 policy,容易 mode collapse (策略学会骗判别器而不是真的做好动作)。AMOR 分两层:
- 低层用 explicit reward (看得见摸得着的 7 个 tracking terms)
- 高层用 implicit reward (判别器)
- 判别器通过"挑权重"间接影响行为,不直接操纵 action

好处是:**权重是可解释的**。HLP 说"这个状态下 velocity 权重要高",你能看懂为什么 —— 因为现在在旋转起步阶段,需要速度。判别器的"偏好"被翻译成了人类能看懂的权重语言。

---

## 实验 4 (Fig. 8) 很有意思

HLP 在一段包含走路 → 旋转 → 打拳 → 站定的混合动作里,自动选的权重会跟着变化:
- 走路时:脚和下半身权重高 (因为脚要踩准)
- 旋转起步时:velocity 权重突然飙升 (要攒旋转速度)
- 打拳时:tracking 权重两个尖峰 (对应两次出拳)
- 站定时:velocity 权重主导 (站住不动时速度应该是零,这个 signal 最重要)

这不是人写的规则,是 HLP 自己学出来的。而且它学到的东西**跟人类的直觉是吻合的**。

---

## 跟其他工作的关系

| 方法 | 怎么处理 reward | 问题 |
|------|-----------------|------|
| DeepMimic | 固定权重,一个动作训一个 policy | 不通用,要调就重训 |
| AMP | 用判别器替代 explicit reward | 不可解释,容易 mode collapse |
| UniCon | 多目标但固定权重 + cost threshold | 不灵活 |
| **AMOR** | **权重作为 input,训一个 policy 覆盖所有 trade-off** | **训练更难,但 inference 灵活** |

AMOR 跟 preference-conditioned RL (Yang et al. 2019) 是一脉相承的,创新点在于:
1. 加了 motion context conditioning (不是单纯 state-conditioned)
2. 用在 character control 这种高维连续控制上
3. 配套 HLP 让 implicit reward 和 explicit reward 各司其职

---

## 代价是什么?

天下没有免费的午餐:

1. **训练更难**: MOPPO 比 PPO 收敛慢,paper 里训练曲线显示 gap,作者假设充分收敛后会缩小但没完全验证
2. **7 个 reward terms 还是 hand-crafted 的**: 权重可调,但"打分项本身"的设计仍然需要专家知识
3. **只能 cover 凸 Pareto front**: 如果最优 trade-off 面有凹陷,线性 scalarization 摸不到,需要更复杂的参数化
4. **计算成本高**: 8192 个并行环境,RTX 4090 跑 5 天,普通实验室复现困难
5. **只在 20-DoF 机器人上验证了 sim-to-real**: 更高维度 (比如 40+ DoF 的人形机器人) 还没测

---

## 我(Andrej)的直觉总结

AMOR 的 philosophy 其实跟现在 ML 的一个大趋势一致: **一次训练,推理时适配**。

- GPT 训一次,inference 时靠 prompt 适配不同任务
- AMOR 训一次,inference 时靠权重向量适配不同 trade-off
- Decision Transformer 训一次,inference 时靠 return-to-go 适配不同目标

本质都是把"任务定义"从训练超参变成推理输入。这个 trend 在 robotics 里还相对新,AMOR 是 character control 领域一个挺漂亮的具体实现。

代码没开源 (Disney Research 习惯),但 idea 可以复现。如果我要玩这个,会先在 simpler setup 上试 (比如 3 个 reward terms 的小任务),验证 MOPPO 的 Pareto front coverage 质量,再 scaling 到 full character control。

---

# AMOR: 通过 Multi-Objective RL 实现 Adaptive Character Control 的深度解析

你好 Andrej! 这篇来自 Disney Research 的 paper 提出了一个相当 elegant 的 idea —— 用 multi-objective RL 的 Pareto front 来解耦 reward weight tuning 和 policy training。下面我从 intuition、技术细节、实验数据三个层面展开。

---

## 1. 核心问题与动机 (Motivation)

在 physics-based character control 中,标准 RL pipeline 是:

$$r_t = \sum_{i=1}^{m} w_i \cdot r_t^{(i)}$$

其中 $w_i$ 是第 $i$ 个 reward term 的权重,通常在 training 前 fixed。问题在于:

1. **Reward term 之间互相冲突**: 比如 maximize tracking accuracy vs. minimize energy (smoothness),需要在两者之间找 trade-off
2. **Tuning cost 巨大**: 每次调 weight 就要重新 train,而 RL training 通常需要 5+ 天 (这篇 paper 的 AMOR 在 RTX 4090 上 train 5 天,300k iterations,8192 个 parallel envs)
3. **Sim-to-real gap**: 在 simulation 中表现好的 weight 在 real robot 上可能不好 (尤其 smoothness term 在 real hardware 上通常需要 weight 更大)
4. **Different motions need different weights**: 一个 fixed weight set 很难 capture 多种 motion style 的需求

AMOR 的核心 insight: **train 一个 conditioned on weights 的 policy**,把 weights 当作 additional input。这样 post-training 可以 zero-shot 调整 weights 而无需 retrain。

---

## 2. MORL 理论基础

### 2.1 从 scalar RL 到 vector RL

传统 RL 的 objective:

$$J(\pi) = \mathbb{E}_{\mathbf{d}_0}\left[V^\pi(\mathbf{s}_0)\right] = \mathbb{E}_\pi\left[\sum_{t \ge 0} \gamma^t r_t \mid \mathbf{s}_0 \sim \mathbf{d}_0\right]$$

变量解释:
- $\pi$: policy
- $\mathbf{d}_0$: initial state distribution
- $V^\pi(\mathbf{s})$: value function, 从 state $\mathbf{s}$ 出发的 expected discounted return
- $\gamma \in [0,1)$: discount factor, 越大越重视 future reward
- $r_t$: scalar reward at time $t$

MORL 把 $r_t$ 变成 vector $\mathbf{r}_t \in \mathbb{R}^m$,对应 return 也变成 vector:

$$\mathbf{J}(\pi) = \mathbb{E}_\pi\left[\sum_{t \geq 0} \gamma^t \mathbf{r}_t \mid \mathbf{s}_0 \sim \mathbf{d}_0\right]$$

这里 $\mathbf{J}(\pi) \in \mathbb{R}^m$,$m$ 是 objective 个数 (这篇 paper 中 $m=7$)。

### 2.2 Pareto Front 的定义

Pareto non-dominance: 一个点 $\mathbf{J}(\pi)$ 是 Pareto non-dominated 的当且仅当不存在另一个 $\mathbf{J}(\pi')$ 使得 $J_i(\pi') \geq J_i(\pi), \forall i$ 且至少有一个 strict inequality。

对于 continuous robotic control,Pareto front 通常是 convex 的,可以用 linear scalarization 表达:

$$\mathcal{F} = \left\{\mathbf{J}(\pi) \mid \exists \mathbf{w} \text{ s.t. } \mathbf{J}(\pi) \cdot \mathbf{w} \geq \mathbf{J}(\pi') \cdot \mathbf{w}, \forall \pi'\right\}$$

其中 $\mathbf{w} \in \Delta^m$ (即 simplex,$\sum_{i=1}^m w_i = 1, w_i \geq 0$)。

这里有个关键的 linearity property:

$$J(\pi) = \mathbb{E}_\pi\left[\sum_{t \geq 0} \gamma^t \mathbf{r}_t \cdot \mathbf{w}\right] = \mathbb{E}_\pi\left[\sum_{t \geq 0} \gamma^t \mathbf{r}_t\right] \cdot \mathbf{w} = \mathbf{J}(\pi) \cdot \mathbf{w}$$

这就是为什么可以 train 一个 conditioned policy 来 cover 整个 Pareto front —— 任何 weight vector $\mathbf{w}$ 都对应一个 Pareto-optimal policy。

---

## 3. AMOR 算法细节

### 3.1 Context-Conditioned Policy

AMOR 的核心扩展是 condition policy 在 **context vector** $\mathbf{c}_t$ 上:

$$\pi(\mathbf{a}_t \mid \mathbf{s}_t, \mathbf{c}_t, \mathbf{w})$$

其中:
- $\mathbf{s}_t$: character state (joint angles, velocities, root pose, etc.)
- $\mathbf{c}_t = (\mathbf{m}_t, \mathbf{z}_t)$: motion context
  - $\mathbf{m}_t$: 当前 motion frame,包含 $(h_t, \theta_t, \mathbf{v}_t, \mathbf{q}_t, \dot{\mathbf{q}}_t, \mathbf{p}_t, \dot{\mathbf{p}}_t)$
    - $h_t$: root height
    - $\theta_t$: root orientation in 6D representation (避免 quaternion 的 discontinuity 问题,参考 [Zhou et al. 2019](https://arxiv.org/abs/1907.07111))
    - $\mathbf{v}_t$: root linear+angular velocities (6D)
    - $\mathbf{q}_t, \dot{\mathbf{q}}_t$: joint positions and velocities
    - $\mathbf{p}_t, \dot{\mathbf{p}}_t$: end-effector (hands+feet) poses
  - $\mathbf{z}_t$: VAE-encoded latent of motion window $\mathbf{M}_t = \{\mathbf{m}_{t-W}, ..., \mathbf{m}_{t+W}\}$,window size $2W+1$。这个 latent 捕捉 local motion pattern,继承自 VMP [Serifi et al. 2024](https://doi.org/10.1111/cgf.15175)
- $\mathbf{w}$: reward weight vector (在整个 episode 中 fixed)

### 3.2 MOPPO (Multi-Objective PPO)

标准 PPO 学 scalar value function $V^\pi(\mathbf{s}, \mathbf{c})$,MOPPO 学 vector-valued value function:

$$\mathbf{V}^\pi(\mathbf{s}, \mathbf{c}, \mathbf{w}) = \mathbb{E}_\pi\left[\sum_{t \geq 0} \gamma^t \mathbf{r}_t \mid \mathbf{s}_0 = \mathbf{s}, \mathbf{c}_0 = \mathbf{c}\right] \in \mathbb{R}^m$$

注意这里 $\mathbf{V}^\pi$ 是 vector (每个 objective 对应一个分量),且 conditioned on $\mathbf{w}$ (虽然 $\mathbf{w}$ 不直接进入 expectation,但它通过 policy 影响 trajectory distribution)。

Policy gradient 用 scalarized advantage:

$$\mathbb{E}_{d^\pi}\left[\sum_{t \geq 0} (\mathbf{A}^\pi(\mathbf{s}_t, \mathbf{c}_t, \mathbf{a}_t) \cdot \mathbf{w}) \nabla_\pi \log \pi(\mathbf{a}_t \mid \mathbf{s}_t, \mathbf{c}_t, \mathbf{w})\right]$$

其中:
- $d^\pi$: discounted stationary state distribution induced by $\pi$
- $\mathbf{A}^\pi$: vector-valued advantage function (用 GAE 计算,每个 objective 一个)
- $\mathbf{A}^\pi \cdot \mathbf{w}$: scalar advantage,做 batch-level normalization

**Weight sampling 策略**: 每个 episode 从 $\Delta^m$ 用 Dirichlet($\alpha=1$)采样一个 $\mathbf{w}$,在整个 episode 中保持 fixed (但 context 随时间变化)。Dirichlet(1) 等价于 uniform distribution over simplex,保证 Pareto front 均匀覆盖。

Replay buffer 存的 transition 是 7-tuple: $(\mathbf{s}_t, \mathbf{c}_t, \mathbf{a}_t, \mathbf{r}_t, \mathbf{s}_{t+1}, \mathbf{c}_{t+1}, \mathbf{w})$。

### 3.3 七个 Reward Terms

这是 AMOR 最关键的设计 —— 7 个 conflicting objectives:

| Objective | Term(s) | Humanoid Scale | Robot Scale |
|-----------|---------|----------------|-------------|
| $r^{\text{up}}$ | $\|\mathbf{q}^{\text{up}} - \hat{\mathbf{q}}^{\text{up}}\|_2^2$ | 1.0 | 7.0 |
| $r^{\text{lo}}$ | $\|\mathbf{q}^{\text{lo}} - \hat{\mathbf{q}}^{\text{lo}}\|_2^2$ | 1.0 | 7.0 |
| $r^{\text{feet}}$ | $\|\mathbf{q}^{\text{feet}} - \hat{\mathbf{q}}^{\text{feet}}\|_2^2$ | 1.0 | 7.0 |
| $r^{\text{rbs}}$ (rigid body) | pos + rot + vel | 1.0 | 1.0 |
| $r^{\text{root}}$ | $\|\mathcal{R}(\theta) - \mathcal{R}(\hat{\theta})\|_2^2$ | 1.0 | 1.0 |
| $r^{\text{vel}}$ | linear + angular | 1.0 | 2.0 |
| $r^{\text{smooth}}$ | $-\|\boldsymbol{\tau}\|^2 - \|\Delta\mathbf{a}\|^2 - \|\Delta^2\mathbf{a}\|^2 - \|\ddot{\mathbf{q}}\|^2$ | $10^{-5}, 10^{-5}, 10^{-5}, 10^{-6}$ | $10^{-4}, 1.5, 0.45, 2.5\times10^{-6}$ |

观察:
- Robot 的 smoothness scale 比 humanoid 大很多 (1.5 vs $10^{-5}$ for action rate penalty),反映了 real hardware 对 jitter 更敏感
- Robot 的 upper/lower body scale 都是 7.0 (vs humanoid 1.0),可能是因为 robot 的 DoFs 少,需要相对放大 tracking 信号
- $\mathcal{R}(\cdot)$ 是把 quaternion/6D 转 rotation matrix $SO(3) \subset \mathbb{R}^{3\times3}$ 的映射,避免直接用 quaternion 的不连续性
- 所有 reward 都加 constant survival bonus $c^{\text{alive}}$ 防止 policy 通过 early termination 来逃避 negative reward

### 3.4 Hierarchical Policy (HLP) 用于自动 weight tuning

AMOR 的 policy 是 zero-shot adaptive 的,但用户手动调 weight 仍然需要 intuition。HLP 把这个 automate 了:

$$\bar{\pi}(\mathbf{w}_t \mid \mathbf{s}_t, \mathbf{c}_t)$$

HLP 输出 weight vector (用 softmax 保证在 simplex 中),reward 用 discriminator-based implicit reward:

$$r^D(\mathbf{O}_t, \mathbf{z}_t) = -\log(1 - D(\mathbf{O}_t \mid \mathbf{z}_t))$$

其中:
- $\mathbf{O}_t = \{\mathbf{o}_{t-V}, ..., \mathbf{o}_t\}$: observation window of size $V$
- $\mathbf{o}_t = (\theta_t, \mathbf{v}_t, \mathbf{q}_t)$: 简化观察 (root pose, velocities, joint angles)
- $D(\mathbf{O}_t \mid \mathbf{z}_t)$: discriminator conditioned on motion latent $\mathbf{z}_t$ (防止 mode collapse,继承自 [CALM](https://arxiv.org/abs/2305.02165) 的设计)

Discriminator loss (Wasserstein GAN-style with gradient penalty):

$$\mathcal{L}^D = -\mathbb{E}_{\mathbf{M}_t \in \mathcal{D}}\left[\mathcal{L}^{\mathbf{M}} + \mathcal{L}^\pi + c^{\text{gp}} \mathcal{L}^{\text{gp}} \mid \mathbf{z}_t = e(\mathbf{M}_t)\right]$$

- $\mathcal{L}^{\mathbf{M}} = \mathbb{E}_{d^{\mathbf{M}}} \log D(\hat{\mathbf{O}}_t \mid \mathbf{z}_t)$: real data loss
- $\mathcal{L}^\pi = \mathbb{E}_{d^\pi} \log(1 - D(\mathbf{O}_t \mid \mathbf{z}_t))$: fake data loss
- $\mathcal{L}^{\text{gp}} = \mathbb{E}_{d^{\mathbf{M}}} \|\nabla_\phi D(\phi)\|^2\big|_{\phi=(\hat{\mathbf{O}}_t, \mathbf{z}_t)}$: gradient penalty on real samples

这个 loss 最小化 Jensen-Shannon divergence (见 [Nowozin et al. 2016](https://arxiv.org/abs/1606.00709))。Gradient penalty 来自 [Mescheder et al. 2018](https://arxiv.org/abs/1801.04406),提高 training stability。

**Key insight**: 直接用 discriminator reward train low-level policy 会导致 mode collapse (paper 中明确提到这个 negative result),所以采用 two-stage approach: 先 train AMOR with explicit rewards,再 freeze AMOR,train HLP 用 implicit reward。这跟 [AMP](https://arxiv.org/abs/2104.02180) 的 end-to-end 设计不同。

---

## 4. 实验数据深度分析

### 4.1 Setup

- **Characters**: Humanoid (36 DoFs, virtual actuators) + Bipedal robot (20 DoFs, realistic actuators following [Grandia et al. 2024](https://roboticsproceedings.org/rss20/p063/))
- **Dataset**: CMU mocap (1870 clips, 8.5h) + Reallusion (214 clips, 0.5h)
- **Simulation**: Isaac Gym, 8192 parallel envs, 250 Hz physics, 50 Hz policy
- **Network**: MLP 4 layers × 1024 units, ELU activations
- **Hardware**: RTX 4090, 5 days for 300k iterations

### 4.2 Pareto Front 可视化 (Fig. 4)

Paper 在 humanoid 上测试了三种 motion: Idle, Walking, Dancing。采样 8192 个 weight vectors,每个 evaluate 15 episodes 取平均。Pareto front 是 7-dimensional vector,用 pairwise 2D projection 可视化。

关键观察:
- 所有 8192 个点都是 Pareto non-dominated w.r.t. 全部 7 个 objectives
- 加黑边框的点是 additional Pareto non-dominated w.r.t. 当前 2D panel 的两个 objective
- 不同 motion 产生不同形状的 Pareto front,说明 fixed weights 无法 universal
- 对于 dancing motion,smoothness 和 lower-body tracking 有明显冲突 (dynamic motion 难以精确 track,引入 jitter)
- 即使看似独立的 upper/lower body tracking 也会冲突 (因为 physical coordination 不是 perfect)
- X-marker (equal weights $w_i = 1/m$) 通常不在 Pareto front 上,说明 uniform weights 不是最优

### 4.3 MOPPO vs PPO Training 对比

Paper 比较 MOPPO (random weights) 和 PPO (fixed equal weights) 的 training curve:
- PPO 收敛更快 (expected,因为 single-objective 是 simpler problem)
- MOPPO 需要在整个 simplex 上 generalize,task 难度大
- MOPPO 用 equal weights evaluate 的 reward 接近 random weights 的 average reward
- Paper 假设 gap 在 fully converge 后会缩小 (但未完全验证)

### 4.4 Robot Sim-to-Real (Fig. 7)

测试在 20-DoF bipedal robot 上的 stationary dancing motion:
- Simulated robot (purple curve) vs. real robot with uniform weights (blue curve)
- Real robot 在 joint positions 和 velocities 上有显著 jitter
- 通过增加 smoothness weight (green curve),jitter 显著减少,sim-to-real gap 缩小

Double pirouette 实验:
- 超出 state-of-the-art fixed-weight VMP controller 的能力
- 需要时变 weights: 初期高 velocity weight (建 rotational speed),末期高 smoothness weight (smooth transition out)
- 实验调 weight 用了 ~1 天,vs training 5 天 —— 直接验证了 AMOR 的 practical value

### 4.5 HLP 行为分析 (Fig. 8)

HLP 输出的 weights 在不同 motion phase 的变化:
- **Walking (red)**: feet + lower body tracking + smoothness balance,平滑步态
- **Spinning (yellow)**: velocity weight spike at start
- **Punches (green)**: tracking weight spike 对应两次 punch 动作
- **Standing still (purple)**: velocity weight 主导 (可能因为 standing 时 velocity tracking 是主要信号)

Fig. 9 比较 discriminator window size $V$ 的影响:
- $V=30$ frames: higher weight on lower body + feet tracking
- $V=2$ frames: higher weight on velocity tracking
- Longer window 注重 long-term motion consistency,shorter window 注重 instantaneous state match

### 4.6 Quantitative Performance (Table 2)

| Weights | MAE↓ (degrees) | Logits↑ |
|---------|----------------|---------|
| Uniform Weights | 10.02 | -18.02 |
| HLP Prediction | 9.55 | -15.48 |

- MAE: 80,000 个 30-second episodes 上的 joint position MAE
- Logits: 训练单独 discriminator (32M 30-frame windows 区分 simulated vs. dataset),在 8M windows 上 evaluate
- HLP 在两个 metric 上都更好 (MAE -0.47°, Logits +2.54)
- Logits 改进更显著,说明 HLP 优化的 implicit reward 比 explicit tracking 更有效

---

## 5. 与相关工作的对比

### 5.1 vs AMP / CALM / Masked Mimic

- [AMP](https://arxiv.org/abs/2104.02180) (Peng et al. 2021): end-to-end discriminator reward,容易 mode collapse
- [CALM](https://arxiv.org/abs/2305.02165) (Tessler et al. 2023): latent-conditioned discriminator,缓解 mode collapse
- [Masked Mimic](https://arxiv.org/abs2409.11193) (Tessler et al. 2024): transformer-based,unified controller

这些方法都 use implicit reward,但 lose interpretability。AMOR 的 HLP 是 hybrid: low-level 用 explicit reward (interpretable),high-level 用 implicit reward (automated tuning),同时获得 automation + interpretability —— 用户可以 inspect HLP 输出的 weights 来理解 discriminator 在某 state 下偏好什么。

### 5.2 vs DeepMimic / VMP / UniCon

- [DeepMimic](https://arxiv.org/abs/1804.10407) (Peng et al. 2018a): single motion, fixed weights,需要 per-motion tuning
- [UniCon](https://arxiv.org/abs/2011.15119) (Wang et al. 2020): multi-objective 但 fixed weights + cost thresholds,限制了 versatility
- [VMP](https://doi.org/10.1111/cgf.15175) (Serifi et al. 2024): AMOR 的 baseline,用 VAE latent + fixed weights,但需要 sim-to-real 时 retrain

### 5.3 vs MORL 文献

- [Ensemble MORL](https://arxiv.org/abs/1910.10524) (Xu et al. 2020): multiple policies,population-based,scaling 差
- [Preference-conditioned](https://arxiv.org/abs/1909.03837) (Yang et al. 2019): 单 policy + preference input,AMOR 属于这类
- [Alegre et al. 2023](https://arxiv.org/abs/2304.00908): sample-efficient MORL,用 Generalized Policy Improvement
- AMOR 的创新点: 加 context conditioning $\mathbf{c}_t$,让 Pareto front 也是 context-dependent

### 5.4 vs Successor Features (SF)

[Successor Features](https://arxiv.org/abs/1608.04323) (Barreto et al. 2017): 假设 reward 是 linear combination of features,可以 fast adaptation 到新 task。AMOR 的 multi-objective formulation 跟 SF 形式上类似,但 goal 不同 —— SF 追求 transfer,AMOR 追求 Pareto front coverage。[Alegre et al. 2022](https://proceedings.mlr.press/v162/alegre22a.html) 给了 MORL 和 SF 的理论连接。

---

## 6. Intuition Building: 为什么这工作可行?

### 6.1 Weight 作为 Latent Variable

可以把 $\mathbf{w}$ 想成一个 latent variable,描述当前 episode 的 "preference mode"。Policy 学到一个 $\mathbf{w} \to \text{behavior}$ 的 mapping。这个 mapping 之所以 learnable,是因为不同 weight 下的 optimal behavior 之间存在 smooth interpolation —— 都是 motion tracking 的 variant,只是 trade-off 不同。

### 6.2 Convex Pareto Front 的关键性

Paper 假设 continuous robotic control 的 Pareto front 是 convex 的 (footnote 1 引用 [Felten et al. 2023](https://arxiv.org/abs/2308.09968))。这意味着 linear scalarization 可以 recover 所有 Pareto-optimal points。如果 Pareto front 是 non-convex 的,需要更复杂的 scalarization (如 Chebyshev),这也是未来工作方向。

### 6.3 Hierarchical Decomposition 的优势

Low-level policy 处理 motor control (40+ Hz, high-dim action space),high-level policy 处理 weight selection (低频, low-dim output 7 dims)。这种 time-scale separation 让 HLP training 快很多 (paper 中提到这点)。

### 6.4 7 个 Objectives 的设计直觉

为什么选这 7 个? 它们对应 motion tracking 的不同 abstraction level:
- **Joint-level**: upper / lower / feet (3 个,granularity 递增)
- **Task-space**: rigid body poses (end-effector 的 global pose)
- **Global**: root pose + velocities (整个 character 的全局状态)
- **Quality**: smoothness (energy + action smoothness)

这种 decomposition 让 conflict 自然出现 —— 比如 tracking upper body 精确可能需要 lower body 偏离 reference (因为 reference 本身 physically infeasible)。

---

## 7. 局限性 & 未来方向

Paper 自己提到的:
1. MOPPO vs PPO 有 performance gap (因为 problem complexity 增加)
2. 可能需要 large-scale training 才能 close gap
3. Alternative weight sampling strategies 可能更 sample-efficient
4. 未来: robot-in-the-loop online RL ([DayDreamer](https://arxiv.org/abs/2206.14176) 风格) 可以 leverage AMOR 加速

我观察到的潜在问题:
1. **7 个 reward terms 是 hand-crafted 的**,虽然 weights 可调,但 reward function 本身的设计仍然 expert-driven
2. **Dirichlet(1) 采样可能不是最优的**,某些 region 的 Pareto front 可能 under-sampled。可以考虑 [Alegre et al. 2023](https://arxiv.org/abs/2304.00908) 的 prioritized sampling
3. **Linear scalarization 限制**: 只能 cover convex Pareto front。对于 non-convex region,需要 non-linear parameterization
4. **HLP 的 discriminator reward 仍然是 task-specific 的**,要 extend 到不同 task 需要重新 train discriminator
5. **Computational cost**: 5 天 training + 8192 parallel envs,小实验室难以 reproduce
6. **Sim-to-real 只在低 DoF robot 上验证** (20 DoFs),scalability 到 full humanoid robot (e.g., 40+ DoFs) 未验证
7. **Time-varying weights 在 inference 时** (pirouette 例子)是 manually tuned,自动化的 HLP 是否能 capture 这种 phase-dependent 需求还需要更多实验

---

## 8. 跟你 (Karpathy) 的工作可能的相关性

作为 Andrej,你可能会对以下关联感兴趣:

1. **跟 nanoGPT / minGPT 的联系**: AMOR 的 policy network 是 simple MLP (4 layers × 1024 units),不是 transformer。但 [Masked Mimic](https://arxiv.org/abs/2409.11193) 已经探索 transformer-based character control,未来可能 combine AMOR 的 weight-conditioning with transformer backbone。

2. **跟 multi-task learning 的联系**: 这本质上是 multi-task learning,其中 "task" = weight vector。跟 [Hypernetworks](https://arxiv.org/abs/1609.09106) (Ha et al. 2016) 思想类似 —— 用一个 network 生成不同 task-specific behavior。

3. **跟 RLHF 的联系**: HLP 用 discriminator reward 训练,跟 RLHF 中 reward model 训练有相似之处,只是 discriminator 替代了 reward model。

4. **跟 Decision Transformer 的联系**: Context-conditioned policy 跟 Decision Transformer 的 conditioning 类似 (condition on return-to-go),只是 AMOR condition on preference vector。

5. **跟 Robotics 的联系**: 你做过的 [Eureka Labs](https://eureka-labs.ai) 工作,这种 "train once, adapt at inference" 的 philosophy 是 general trend。

---

## Reference Links

1. **AMOR paper (Disney Research)**: https://la.disneyresearch.com/ (具体 paper page 待 SIGGRAPH 2025 发布)
2. **PPO (Schulman et al. 2017)**: https://arxiv.org/abs/1707.06347
3. **DeepMimic (Peng et al. 2018a)**: https://arxiv.org/abs/1804.10407
4. **AMP (Peng et al. 2021)**: https://arxiv.org/abs/2104.02180
5. **CALM (Tessler et al. 2023)**: https://arxiv.org/abs/2305.02165
6. **Masked Mimic (Tessler et al. 2024)**: https://arxiv.org/abs/2409.11193
7. **VMP (Serifi et al. 2024)**: https://doi.org/10.1111/cgf.15175
8. **MORL Survey (Hayes et al. 2022)**: https://arxiv.org/abs/2108.03162
9. **Successor Features (Barreto et al. 2017)**: https://arxiv.org/abs/1608.04323
10. **Isaac Gym**: https://developer.nvidia.com/isaac-gym
11. **On Continuity of Rotation Representations (Zhou et al. 2019)**: https://arxiv.org/abs/1907.07111
12. **f-GAN (Nowozin et al. 2016)**: https://arxiv.org/abs/1606.00709
13. **Which Training Methods for GANs do actually Converge (Mescheder et al. 2018)**: https://arxiv.org/abs/1801.04406
14. **DayDreamer (Wu et al. 2022)**: https://arxiv.org/abs/2206.14176
15. **Bipedal Robotic Character (Grandia et al. 2024)**: https://roboticsproceedings.org/rss20/p063/
16. **CMU Mocap Database**: http://mocap.cs.cmu.edu/
17. **Reallusion**: https://www.reallusion.com/
18. **Alegre et al. 2023 (MORL prioritization)**: https://arxiv.org/abs/2304.00908
19. **Alegre et al. 2022 (MORL-SF connection)**: https://proceedings.mlr.press/v162/alegre22a.html
20. **Ensemble MORL (Xu et al. 2020)**: https://arxiv.org/abs/1910.10524

---

## 总结

AMOR 的核心贡献是把 reward weight tuning 从 training time 解耦到 inference time。技术上是通过:
1. Multi-objective PPO with vector-valued value function
2. Policy conditioned on (state, context, weights)
3. Dirichlet weight sampling 保证 Pareto front 覆盖
4. Hierarchical extension 让 implicit discriminator reward 自动选择 weights,同时保留 explicit reward 的 interpretability

这个工作对 robotics + character animation 都有 practical value —— sim-to-real transfer 从需要多次 retrain 变成 1 天 manual tuning 或自动 HLP adaptation。理论上是 Pareto front learning 的一个具体 instantiation,跟 preference-conditioned RL / successor features / multi-task learning 都有 deep connection。

希望这个深度解析对你的 intuition building 有帮助!如果你想 dive deeper 到某个具体方面 (比如 MOPPO 的 implementation details、HLP training dynamics、或者跟其他 RLHF-style 方法的对比),可以继续聊。
