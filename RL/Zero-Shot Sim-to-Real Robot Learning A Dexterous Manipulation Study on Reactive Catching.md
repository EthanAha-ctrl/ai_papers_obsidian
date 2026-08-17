---
source_pdf: Zero-Shot Sim-to-Real Robot Learning A Dexterous Manipulation Study on
  Reactive Catching.pdf
paper_sha256: 7d32f132566ec3db3bb4deedcbaf07f1760846ceeab32e1445167fc16ee693c6
processed_at: '2026-08-13T06:49:30-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲 DRIS

好, 前面那个太 academic 了, 我换个方式, 像跟你 coffee chat 那样讲.

---

## 这 paper 在解决什么破事

你训练 robot 接球. 模拟器里练得贼好, 一到真 robot 上就崩. 为啥? 因为 simulator 里的 friction, restitution 这些参数跟真实世界对不上, 差一点点, 球就飞了.

大家常规做法是 **Domain Randomization** (DR): 训练时候把 friction, restitution, ball radius 这些参数每个 episode 随机换一个值, 让 policy 见过各种情况, 希望 deploy 时能 robust. 这招 [OpenAI 做魔方那个](https://arxiv.org/abs/1910.07113) 用过, [Tobin 2017](https://arxiv.org/abs/1703.06907) 最早系统化.

但 DR 有个 subtle 的问题. 每个 episode 你只 sample **一个** physics parameter, 跑一条 trajectory. Policy 看到的是: "这回合 friction=0.05, 我这么动, 球这么飞了. 下回合 friction=0.08, 我这么动, 球那么飞了." 它从 batch 里慢慢 implicit 学到 "哦, friction 不同, 我得 robust".

问题在于: **policy 从来没在同一步同时看到 "如果 friction 是 A vs B, 球会怎么分叉"**. 它是 episode-level 的 exposure, 不是 step-level 的 reasoning. 对于慢任务够用, 对接球这种 millisecond 反应的任务, 不够.

---

## DRIS 的 idea: 别一个个来, 一起上

DRIS (Domain-Randomized Instance Set) 的核心 insight 特别简单: **每个 episode 同时 spawn N 个球, 每个球 physics 参数不同, 但 robot 对这 N 个球只做一个 action**.

想象一下: 128 个 parallel environment, 每个 environment 里 plate 上方飘着 50 个球, 有的 friction 高有的低, 有的 bouncy 有的 dead. 你这一个 action 下去, 50 个球各自飞各自的轨迹. 你 reward 是这 50 个球 reward 的平均.

训练时 policy 看到的输入是这 50 个球的 state (position + velocity), 它必须选一个 action, 让这 50 个不同 physics 的球 **outcome 分布都 narrow**. 这就逼着 policy 学 robust behavior, 因为它没法只针对一个 physics 优化.

这就是 paper 标题里 "set of randomized instances simultaneously" 的意思.

---

## 为什么这招 work: 三个理论 insight

### Insight 1: 这就是 particle filter 的 propagation step

[Particle filter](https://www.stats.ox.ac.uk/~doucet/doucet_defreitas_gordon_smcbookintro.html) 你知道吧, POMDP 里估计 belief 用的. 你 sample 一堆 particle, 每个 particle 代表一个可能的 hidden state, 然后 forward propagate, 根据 observation 调权重.

DRIS 的 forward step 跟 particle filter 的 propagation step **mathematically 等价**. 你有 N 个 (state, physics_parameter) pair, 每个 pair 就是一个 particle. 你用 simulator 把每个 particle forward 一步, 得到新的 N 个 particle. 这就是 Appendix B.1 那个 Proposition 证明的东西.

区别是: particle filter 里每个 particle 有自己的 weight, 你要 resample; DRIS 里所有 particle **共享一个 action**, 没 weight, 没 resample. 因为你的目标不是 estimate belief, 是 train policy. 但 propagation 这一步是同一个 math.

### Insight 2: Gradient variance 直接降

你训练 policy, 算 gradient. DR 的 gradient 是一个 episode 的 return gradient, 噪音很大. DRIS 的 gradient 是 N 个 episode 的平均 gradient, 噪音小.

Paper Appendix B.2 给了个 clean 的公式. 令 $\mathbf{g}_i$ 是第 $i$ 个 instance 的 return gradient, DRIS 的 estimator 是 $\hat{\mathbf{g}} = \frac{1}{N}\sum_i \mathbf{g}_i$. 如果 policy 对 instance 排列不变 (permutation invariant), 这些 $\mathbf{g}_i$ exchangeable, 共享同一个 covariance $\boldsymbol{\Sigma}$ 和 cross-covariance $\boldsymbol{\Sigma}_{\text{cross}}$. 那么:

$$\text{Var}(\hat{\mathbf{g}}) = \sigma^2 \left( \rho + \frac{1 - \rho}{N} \right)$$

- $\sigma^2 = \text{Tr}(\boldsymbol{\Sigma})$: 单 instance 的 variance
- $\rho = \text{Tr}(\boldsymbol{\Sigma}_{\text{cross}}) / \sigma^2$: instance 之间的 correlation, $0 \leq \rho \leq 1$

$N=1$ 就是 DR, variance 是 $\sigma^2$. $N \to \infty$, variance 收敛到 $\rho \sigma^2$. 只要 $\rho < 1$ (instance 之间不完全 correlated, 这通常成立), 大 $N$ 就 reduce variance.

为啥有 $\rho$ 这个 floor? 因为所有 instance 共享同一个 action, 它们的 return 有 intrinsic correlation, 不完全独立. 这个 floor 是 shared action 带来的代价, 但即便有这个 floor, 还比 $\sigma^2$ 小.

**Intuition**: 你可以理解为 ensemble averaging. 跟 [PETS (Chua 2018)](https://arxiv.org/abs/1808.04271) 用 ensemble dynamics model 降 epistemic uncertainty 是亲戚, 但 PETS 是 model uncertainty, DRIS 是 domain uncertainty.

### Insight 3: 大 N 会自动 unmask 一个 robustness penalty (这是最漂亮的)

这个是 Appendix B.3 的 Theorem B.1, paper 的核心 insight.

你把 reward 取负作为 cost $C_t(\mathbf{s})$. 在 mean state $\boldsymbol{\mu}_t$ 附近做 Taylor 展开:

$$C_t(\mathbf{s}) \approx C_t(\boldsymbol{\mu}_t) + \underbrace{\mathbf{g}_t^\top (\mathbf{s} - \boldsymbol{\mu}_t)}_{\text{一阶 noise}} + \underbrace{\frac{1}{2}(\mathbf{s}-\boldsymbol{\mu}_t)^\top \mathbf{H}_t (\mathbf{s}-\boldsymbol{\mu}_t)}_{\text{二阶 curvature}}$$

DRIS 的 empirical cost 是 N 个 instance 平均:

$$\hat{J}_{N,t} = \frac{1}{N}\sum_i C_t(\mathbf{s}_t^{(i)})$$

代入 Taylor 展开:

$$\hat{J}_{N,t} \approx C_t(\boldsymbol{\mu}_t) + \mathbf{g}_t^\top \underbrace{\frac{1}{N}\sum_i(\mathbf{s}_t^{(i)} - \boldsymbol{\mu}_t)}_{\bar{\boldsymbol{\delta}}_{N,t}} + \frac{1}{2}\text{Tr}\left(\mathbf{H}_t \underbrace{\frac{1}{N}\sum_i (\mathbf{s}_t^{(i)}-\boldsymbol{\mu}_t)(\mathbf{s}_t^{(i)}-\boldsymbol{\mu}_t)^\top}_{\hat{\boldsymbol{\Sigma}}_{N,t}}\right)$$

- $\bar{\boldsymbol{\delta}}_{N,t}$ 是 sample mean error, 期望 0, $N\to\infty$ 时 $\to 0$
- $\hat{\boldsymbol{\Sigma}}_{N,t}$ 是 empirical covariance, $N\to\infty$ 时 $\to \boldsymbol{\Sigma}_t$ (真实 aleatoric variance)

所以 $N\to\infty$ 时:

$$\hat{J}_{N,t} \to C_t(\boldsymbol{\mu}_t) + \frac{1}{2}\text{Tr}(\mathbf{H}_t \boldsymbol{\Sigma}_t)$$

看到没, **一阶 noise 被 average 掉了, 二阶 term 留下来了**. 这个 $\frac{1}{2}\text{Tr}(\mathbf{H}_t \boldsymbol{\Sigma}_t)$ 是 non-negative (因为 $\mathbf{H}_t \succeq 0$ local convexity), 它就是 **physics-induced variance 的 penalty**.

意思: DRIS 训练到后来, policy 优化目标自动变成 "minimize cost at mean state + minimize state variance under physics uncertainty". 后者就是 robustness! 你没显式加 reward shaping, 它自己 emergent 出来了.

对比 DR ($N=1$): 一阶 noise $\mathbf{g}_t^\top \bar{\boldsymbol{\delta}}_{1,t}$ 主导, optimizer 在追 random first-order fluctuation, 那个二阶 robustness penalty 被 noise 盖住了, 学不到. 这就是 "asymptotic unmasking" 的含义: 大 N 把 noise mask 揭开, 露出底下真正的 robustness signal.

---

## 怎么 deploy: N=1 也能用

训练时 $N=200$ 个球一起飞, deploy 时真世界只有一个球. 怎么处理?

用 **point cloud autoencoder** ([Achlioptas 2018](https://arxiv.org/abs/1707.02392)) 把 variable-size set 压成 fixed-dim latent. Encoder 是 1D conv + max-pool, permutation invariant, 跟 [PointNet](https://arxiv.org/abs/1612.00593) 思路一样. 不管你输入 200 个球还是 1 个球, 都压成 $d_z=64$ 维 latent. 训练时用 Chamfer distance pretrain 这个 AE, 然后冻住 encoder, 用 latent 训 PPO policy.

Deploy 时 real ball state $N=1$ 的 set, encoder 也能处理, 输出同一个 64 维 latent, policy 接着用.

---

## Task setup 的几个细节

**Motion frame**: 为了让 policy 对 ball incoming direction invariant, 定义一个 motion frame, 原点在 plate 初始位置, X-Z 平面包含 ball 初始位置. 所有 state, action 都在 motion frame 里表达. 这样不管球从哪个方向来, policy 看到的 state 都是 "球在我正前方某处飞过来" 这种 canonical form.

**Action**: 5 维, 3 维 plate translation + 2 维 tilt (tilt axis 方向 $\alpha$ + tilt angle $\beta$). Tilt 用 Rodrigues' formula 转成 rotation matrix, IK 转成 joint config, 然后 PD torque controller 跟踪. Real deploy 时 K, D gain 在 real robot 上 tune 一下匹配 simulator behavior, 这是唯一用到的 real info.

**Reward**: ball velocity 低就奖励 (用 exp decay), ball 掉到 plate 下面或飞出去就 -1 penalty. 速度分解成 plate normal 方向 (perpendicular) 和 plate 平面方向 (parallel), 分别 penalize. 这里有个 trick: perpendicular velocity 用 $\max\{v_\perp, -0.1\}$, 意思是球往下飞速度大于 0.1 m/s 时不 penalize (因为还在下落 phase, 还没到接球).

**FiLM conditioning**: Policy 用 [FiLM (Perez 2018)](https://arxiv.org/abs/1709.07871) 把 plate 当前 tilt orientation 作为 condition 注入 latent. 为啥? 球在 plate 上时, plate 的 tilt 改变 gravity 沿 plate 切向的分量, 直接影响球加速度. FiLM 用 affine transform $\tilde{\mathbf{z}} = \lambda(\mathbf{u})\odot\mathbf{z} + \mu(\mathbf{u})$ 调制 latent, 比 concatenate 更 expressive, 让 policy 能根据 tilt 动态调整 action generation 的几何意义.

---

## 实验亮点

### Simulation 三个 robustness test

1. **Observation noise**: 给 ball state 加 Gaussian noise, $N=1$ (相当于 DR) 和 E2E degrade 严重, $N=10$ 已经明显 robust, $N=50, 200$ 更好.
2. **Execution error**: 给 desired joint position 加 uniform noise, 同样 pattern.
3. **OOD physics**: train restitution $[0.4,0.7]$, test $[0.7,0.8]$, 大 $N$ policy generalizes 明显更好.

### Scalability study (最关键)

有人可能说: "DRIS 不就是看了更多 ball interaction 吗? 我把 DR 的 environment 数加大不就行了?"

Paper 做了 controlled experiment: 128 envs × 50 balls DRIS vs 128×50 envs single-ball E2E, total ball interaction 数 match.

- E2E 128×50: success rate 0.73, VRAM 2 GB
- DRIS 128×50: success rate **0.89**, VRAM 0.71 GB

DRIS 用更少 VRAM 拿到更高 success. 证明 gain 不是来自 "看更多数据", 是来自 **structured ensemble propagation** 的 variance reduction.

### Real robot

7-DoF Franka FR3, flat plate (无 passive stabilization, 不像 cup/net), 3D printed ramp 释放球, 4 种球 (wiffle, rubber, ping-pong, foam), 3 种 ramp 曲率. 两 camera 80FPS 做球 tracking.

| | VelTrack (hand-crafted) | E2E | DRIS N=200 |
|---|---|---|---|
| Total success | 3/60 | 8/60 | **41/60 (68%)** |

VelTrack 几乎全失败 (反应不够快). E2E jerky, 经常撞飞球或触发 robot power limit. DRIS 稳, 4 种球都能接.

还做了 qualitative: human throw 也能接, generalize 到 pure balancing task, 甚至能接 irregular 形状物体 (toy apple, strawberry, wooden cube).

---

## Limitations 和我的想法

Paper 自己提了三个 limitation:

1. **High-dim input**: 现在 state 是 6D (position + velocity), point cloud AE handle 得了. 如果是 image 或 high-dim geometry, encoder 需要更大, pretrain 更贵.
2. **Computational cost**: N 个 instance 同时 propagate, 如果有复杂 multi-body contact, collision resolving 会成 bottleneck.
3. **Dissipative vs energy-injecting**: 这是最 fundamental 的 limitation. Paper 引 [Schaal & Atkeson 1993 juggling paper](https://ieeexplore.ieee.org/document/518049) 的 insight: dissipative 系统 (接球, 能量被吸收) 有 self-stabilization, 不同 physics 的 trajectory 会 converge, shared action 能 cover. 但 throwing 这种 energy-injecting 系统, variation 会放大, shared action 没法同时 handle 发散的 instance state.

我的额外想法:

**a. 跟 belief-space planning 的关系**: DRIS 本质是把 belief propagation embed 进 policy training. 传统 [POMDP belief-space planning](https://www.cs.cmu.edu/~cassandra/papers/ai96.pdf) 要 discretize 或 sample-based, 计算贵, deploy 慢. DRIS 训练时做 belief propagation, deploy 时只 forward pass, 拿到 robust policy. 跟 [Dreamer (Hafner 2019)](https://arxiv.org/abs/1912.01603) 把 planning 放 latent space 有 philosophy 上的呼应, 但 DRIS 不 imagine, 直接 propagate ensemble.

**b. Asymptotic unmasking 的 deep connection**: $\frac{1}{2}\text{Tr}(\mathbf{H}_t \boldsymbol{\Sigma}_t)$ 这个 term 跟 risk-sensitive RL, distributional RL, robust optimization, Bayesian experimental design 都有 connection. 你在 robust optimization 里见过的 worst-case-aware surrogate 就是这个 form. DRIS 免费拿到了, 不需要显式 reward shaping. 这跟 [entropy maximization DR (Tiboni 2024)](https://arxiv.org/abs/2310.07256) 的哲学不同, 那是 explicit entropy regularization, DRIS 是 implicit curvature penalty.

**c. $\rho$ floor 的 design tradeoff**: $\rho \sigma^2$ 这个 floor 是 shared action 的代价. 如果 action space 很 expressive, policy 可以让 instance 几乎独立 evolve, $\rho \to 0$, variance reduction 接近 $1/N$. 如果 action space 限制大, $\rho$ 大, 加 $N$ 收益小. 这是个没 explore 的 tradeoff, 可以做 adaptive action space 或 hierarchical action (shared component + instance-specific component).

**d. 跟 Caging in Time 的关系**: 同作者之前 [Caging in Time (Wang 2025)](https://journals.sagepub.com/doi/10.1177/02783649251343926) 也是 set-based representation for robust manipulation, 但用 caging 约束. DRIS 是同 lab 的延续, 从 caging constraint 换成 ensemble propagation.

**e. Multi-agent extension**: 现在所有 instance 共享一个 action. 如果允许 instance-specific action with shared component, 就像 multi-agent RL with common reward. 但 deploy 时只有 1 个 ball, 所以 instance-specific action 在 deploy 时用不上. 不过训练时用 mixture 可能 help exploration, 是 future direction.

**f. 跟 SimCLR 的 philosophy 相似**: [SimCLR](https://arxiv.org/abs/2002.05709) 多个 view 共享 representation 学 invariant feature. DRIS 多个 instance 共享 action 学 robust policy. 都是 "multiple views/instances + shared component" 的 recipe.

---

## 一句话总结

DRIS 是个 simple but deep 的 trick: 训练时同时 propagate N 个 randomized physics instance, 共享 action, average reward. 数学上等价于 particle filter 的 belief propagation, 梯度 variance 从 $\sigma^2$ 降到 $\sigma^2(\rho + (1-\rho)/N)$, 大 N 自动 unmask 一个 robustness penalty $\frac{1}{2}\text{Tr}(\mathbf{H}\boldsymbol{\Sigma})$. 在 dissipative 的 dynamic manipulation task (接球) 上 zero-shot transfer 到 real robot, 68% success rate across 4 种球 × 3 种 ramp.

最该读的部分: Appendix B 的理论分析, 特别是 Theorem B.1 (Asymptotic Unmasking). Section IV 的 task formulation 也写得 clean.

### Reference links

- [OpenAI Rubik's cube](https://arxiv.org/abs/1910.07113)
- [Tobin 2017 Domain Randomization](https://arxiv.org/abs/1703.06907)
- [Peng 2018 Dynamics DR](https://arxiv.org/abs/1710.06537)
- [PETS (Chua 2018)](https://arxiv.org/abs/1808.04271)
- [Dreamer (Hafner 2019)](https://arxiv.org/abs/1912.01603)
- [PointNet (Qi 2017)](https://arxiv.org/abs/1612.00593)
- [Point cloud AE (Achlioptas 2018)](https://arxiv.org/abs/1707.02392)
- [FiLM (Perez 2018)](https://arxiv.org/abs/1709.07871)
- [PPO (Schulman 2017)](https://arxiv.org/abs/1707.06347)
- [ManiSkill3](https://maniskill.github.io/)
- [Schaal-Atkeson juggling 1993](https://ieeexplore.ieee.org/document/518049)
- [Caging in Time (Wang 2025)](https://journals.sagepub.com/doi/10.1177/02783649251343926)
- [Entropy max DR (Tiboni 2024)](https://arxiv.org/abs/2310.07256)
- [BayesSim (Ramos 2019)](https://arxiv.org/abs/1906.09696)
- [Particle filter intro (Doucet)](https://www.stats.ox.ac.uk/~doucet/doucet_defreitas_gordon_smcbookintro.html)
- [SimCLR](https://arxiv.org/abs/2002.05709)
- [Rademacher complexity](https://en.wikipedia.org/wiki/Rademacher_complexity)
- [McDiarmid's inequality](https://en.wikipedia.org/wiki/McDiarmid%27s_inequality)
- [Rodrigues' formula](https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula)
- [Liouville equation](https://en.wikipedia.org/wiki/Liouville%27s_theorem_(Hamiltonian))
- [Integral Probability Metric](https://en.wikipedia.org/wiki/Integral_probability_metric)

还想让我展开哪个部分? 比如 FiLM 在 latent space 的几何意义, 或者 asymptotic unmasking 跟 Hessian-based meta-learning 的 connection, 都可以继续聊.

---

# Paper 深度讲解：Zero-Shot Sim-to-Real Robot Learning via DRIS

下面 我 会从 motivation, method, theory, instantiation, experiments 这几个 layer 把这篇 paper 彻底拆开给你, 重点是 build intuition, 让你理解为什么这套 trick 在 dynamic, contact-rich, dissipative 任务上能 work.

---

## 1. Motivation：Dexterous Manipulation 的 Sim-to-Real Gap 为什么难

Dexterous manipulation 的本质是 robot 通过 intentional contact 来控制 object 的 motion, 比如 in-hand manipulation, tool use, pushing, catching. 这种 task 有几个非常 annoying 的性质:

1. **Physics-intensive**: contact geometry, friction coefficient, restitution coefficient, inertia, compliance 这些参数 small perturbation 就能放大成 large behavioral deviation.
2. **Discontinuous dynamics**: 高速 impact 时候 contact event 是 instantaneous 的, gradient 在 contact onset 时刻是 ill-defined 的.
3. **Highly sensitive to perception noise**: ball state 估计偏差 1cm, 5cm/s, 都可能让 plate 撞飞球.

传统 model-based 方法 ([Chavan-Dafle & Rodriguez 2015](https://ieeexplore.ieee.org/document/7353506)) 依赖 known system parameters, 在这种 uncertain dynamics 下直接崩. Learning-based 方法用 **Domain Randomization (DR)** ([Tobin et al. 2017](https://arxiv.org/abs/1703.06907), [Peng et al. 2018](https://arxiv.org/abs/1710.06537)) 在 simulator 里随机化 physics parameters, 让 policy 学到 robust behavior.

但 traditional DR 有一个 fundamental issue：**每个 episode 只 sample 一个 domain parameter instance** $\mathbf{c} \in \mathcal{C}$, policy $\pi_\theta(\mathbf{s}_t)$ 看到的只是一个 single realization 的 trajectory. 这相当于把 uncertainty 喂给 policy 当作 "noise", policy 没有 structured mechanism 去 reason about "如果 dynamics 是这样 / 那样, 我该做什么 action 才能对 distribution of outcomes 都 robust". 这就是 paper 要修的事.

---

## 2. DRIS 的核心 Idea：从 Sample-Once 到 Simultaneous-Propagate

### 2.1 Conventional DR 回顾

定义 domain space $\mathcal{C}$ 为所有 relevant physical parameters 的集合, 比如 ball radius, friction, restitution. 每个 $\mathbf{c} \in \mathcal{C}$ 诱导一组确定的 dynamics:

$$\mathbf{s}_{t+1} = f(\mathbf{s}_t, \mathbf{a}_t, \mathbf{c})$$

其中 $\mathbf{s}_t$ 是 state, $\mathbf{a}_t$ 是 action, $f$ 是 deterministic transition function (simulator 里). Conventional DR 的 objective 是:

$$J_1(\theta) = \mathbb{E}_{\mathbf{s}_0 \sim p(\mathbf{s}_0), \mathbf{c} \sim p(\mathbf{c})}\left[ R(\tau_\mathbf{c}) \right], \quad R(\tau_\mathbf{c}) = \sum_{t=0}^{T-1} \gamma^t r(\mathbf{s}_t, \mathbf{a}_t)$$

这里 $R(\tau_\mathbf{c})$ 是 trajectory $\tau_\mathbf{c} = (\mathbf{s}_0, \mathbf{a}_0, \mathbf{s}_1, \dots, \mathbf{s}_T)$ 的 discounted return, $\gamma \in [0,1)$ 是 discount factor. 在 each episode 里 sample 一个 $\mathbf{c}$, 跑出一条 trajectory, 算 return, 算梯度. 多个 episode 的梯度平均就 estimate 了 $\nabla J_1$. 这就是 PPO/SAC 里 standard 的写法.

**问题**: 在 single episode 里, policy 看不到 "如果 c 不同, action 的 outcome 会怎么分布". 它只能 implicit 的从 batch 里慢慢学到 "对各种 c 都 robust". 但在 high-speed dynamic task 里 (catching 需要 millisecond 级 reaction), 这种 implicit robustness 是 insufficient 的.

### 2.2 DRIS 的 formulation

DRIS 同时 sample $N$ 个 domain parameters:

$$\hat{\mathcal{C}} = \{ \mathbf{c}^{(i)} \}_{i=1}^N, \quad \mathbf{c}^{(i)} \sim \mathcal{U}(\mathcal{C})$$

每个 instance 的 state 独立 evolve, 但**共享同一个 action** $\mathbf{a}_t = \pi_\theta(\mathcal{S}_t)$, 其中 $\mathcal{S}_t = \{\mathbf{s}_t^{(1)}, \dots, \mathbf{s}_t^{(N)}\}$ 是 ensemble state. Transition 是:

$$\mathbf{s}_{t+1}^{(i)} = f\left(\mathbf{s}_t^{(i)}, \pi_\theta(\mathcal{S}_t), \mathbf{c}^{(i)}\right), \quad \forall i \in \{1, \dots, N\}$$

注意 coupling：instance $i$ 的 next state 不只取决于它自己的 state 和自己的 $\mathbf{c}^{(i)}$, 还取决于**所有其他 instance 的 state** (通过 shared action $\pi_\theta(\mathcal{S}_t)$). 这是关键的设计, 也是后面 variance reduction 的 source.

DRIS 的 objective 是 average return:

$$\mathcal{J}_N(\theta) = \mathbb{E}_{\mathbf{s}_0 \sim p(\mathbf{s}_0), \mathbf{c}^{(i)} \sim p(\mathbf{c})}\left[ \frac{1}{N} \sum_{i=1}^N R(\tau_{\mathbf{c}^{(i)}}) \right]$$

期望上和 conventional DR 是同一个 objective (because $\mathbf{c}^{(i)}$ i.i.d.), 但**empirical optimization landscape 完全不同**, 后面会展开.

### 2.3 Size-Agnostic Encoding

Deploy 时候 real world 只有 1 个 ball, 但训练时 policy 输入是 $N$ 个 ball 的 state set $\mathcal{S}_t \in \mathbb{R}^{N \times 6}$ (这里 6 = 3 position + 3 velocity). 所以需要 encoder $\psi: \mathcal{S}_t \to \mathbf{z}_t \in \mathbb{R}^{d_z}$ 把 variable-size set 压成 fixed-dim latent, 这样新的 $N$ (比如 $N=1$ at deployment) 也能用同一 policy.

他们用 **point cloud Autoencoder** ([Achlioptas et al. 2018](https://arxiv.org/abs/1707.02392)), encoder 是 1D conv + max-pooling (跟 PointNet ([Qi et al. 2017](https://arxiv.org/abs/1612.00593)) 思路一致, permutation invariant), decoder 是 MLP, reconstruction loss 是 Chamfer distance:

$$\mathcal{L}_\psi(\mathcal{S}_t, \tilde{\mathcal{S}}) = \frac{1}{N} \sum_{\mathbf{s} \in \mathcal{S}_t} \min_{\mathbf{s}' \in \tilde{\mathcal{S}}} \|\mathbf{s} - \mathbf{s}'\|^2 + \frac{1}{|\tilde{\mathcal{S}}|} \sum_{\mathbf{s}' \in \tilde{\mathcal{S}}} \min_{\mathbf{s} \in \mathcal{S}_t} \|\mathbf{s} - \mathbf{s}'\|^2$$

第一项是 "每个 ground truth point 找 nearest reconstruction" 的距离, 第二项是 reverse direction. Chamfer distance 对称, 适合 set-to-set matching. 这里 input 是 6D (position+velocity), 而非 standard 3D point cloud, 所以 conv kernel 调成对应 channels.

Pretrain AE 100 epochs (~10 min), 然后冻结 encoder, 用 latent $\mathbf{z}_t$ 训 PPO policy 1000 epochs (~2h), 在单卡 RTX 3060 12GB 上. $d_z = 64$.

---

## 3. 理论分析：为什么 DRIS 真的更好

这是 paper 最 valuable 的部分, 我把每个 theorem 拆开讲, 把每一步 intuition 都讲清楚.

### 3.1 DRIS = Exact Particle Propagation of Belief

记 $b_t(\mathbf{s}, \mathbf{c})$ 为时刻 $t$ 的 joint belief (state + physics parameter 的 joint density). 由于 $\mathbf{c}$ 是 static (不会随时间变化), belief 演化服从 **Liouville equation** (连续性方程):

$$b_{t+1}(\mathbf{s}', \mathbf{c}') = \iint \delta\big(\mathbf{s}' - f(\mathbf{s}, \mathbf{a}_t, \mathbf{c})\big) \delta(\mathbf{c}' - \mathbf{c}) \, b_t(\mathbf{s}, \mathbf{c}) \, d\mathbf{c}\, d\mathbf{s}$$

- $\delta(\cdot)$ 是 Dirac delta function
- 第一个 $\delta$ 表示 "state $\mathbf{s}$ 经过 dynamics $f$ 后落到 $\mathbf{s}'$"
- 第二个 $\delta$ 表示 "physics parameter 是 static"

**Proposition B.1**: 如果初始 belief 用 empirical measure 表示为 $\hat{b}_{t,N}(\mathbf{s}, \mathbf{c}) = \frac{1}{N} \sum_i \delta(\mathbf{s} - \mathbf{s}_t^{(i)}) \delta(\mathbf{c} - \mathbf{c}^{(i)})$, 那么 Liouville equation 演化一步之后, empirical measure 变成 $\frac{1}{N} \sum_i \delta(\mathbf{s}' - f(\mathbf{s}_t^{(i)}, \mathbf{a}_t, \mathbf{c}^{(i)})) \delta(\mathbf{c}' - \mathbf{c}^{(i)})$, 也就是 updated particles $\{\mathbf{s}_{t+1}^{(i)}\}$.

**Intuition**: 这告诉我们 DRIS 不是某种 "approximation trick", 它本身就是 particle filter 的 exact propagation step. 用 $N$ 个 particles (每个 particle 是一个 sampled physics parameter + 对应 state) 在 simulator 里同步 forward 一步, 就 exact 等价于把 belief 在真实 dynamics 下 propagate 一步. 这跟 standard particle filter 的 propagation step ([Doucet et al. 2001](https://www.stats.ox.ac.uk/~doucet/doucet_defreitas_gordon_smcbookintro.html)) 是同一个 math.

但和 particle filter 不一样的地方是: particle filter 每个 particle 看自己的 observation, 而 DRIS 通过 **shared action** 把所有 particle coupling 起来, 这就引出下一个 theorem.

### 3.2 Gradient Variance Reduction

记 $\mathbf{g}_i = \nabla_\theta R(\tau_{\mathbf{c}^{(i)}})$ 为第 $i$ 个 instance 的 return gradient. DRIS 的 gradient estimator 是 $\hat{\mathbf{g}}_{\text{DRIS}} = \frac{1}{N} \sum_i \mathbf{g}_i$.

**Lemma B.1 (Exchangeability)**: 如果 policy $\pi_\theta$ 是 **permutation invariant** (instance 的排列顺序不影响 action 输出), 那么 $\mathbf{g}_1, \dots, \mathbf{g}_N$ 是 exchangeable random variables, 也就是任意 permutation 下 joint distribution 不变.

Proof sketch: 因为 $\mathbf{c}^{(i)}$ i.i.d., 加上 $\pi_\theta$ 对 instance 排列不变, 所以 $(\mathbf{g}_1, \dots, \mathbf{g}_N)$ 的 joint distribution 在 permutation 下不变, 即 exchangeable. 这就推出所有 $\mathbf{g}_i$ 共享同一 mean $\boldsymbol{\mu}$, 同一 covariance $\boldsymbol{\Sigma}$, 以及同一 cross-covariance $\boldsymbol{\Sigma}_{\text{cross}}$ (任意 $i \neq j$).

**Proposition B.2 (Variance Reduction)**: 令 scalar total variance $\sigma^2 = \text{Tr}(\boldsymbol{\Sigma})$, scalar correlation $\rho = \text{Tr}(\boldsymbol{\Sigma}_{\text{cross}}) / \sigma^2$, 则

$$\text{Tr}\left(\text{Cov}(\hat{\mathbf{g}}_{\text{DRIS}})\right) = \sigma^2 \left( \rho + \frac{1 - \rho}{N} \right)$$

**Derivation**:

$$\text{Cov}(\hat{\mathbf{g}}_{\text{DRIS}}) = \text{Cov}\left(\frac{1}{N}\sum_i \mathbf{g}_i\right) = \frac{1}{N^2}\left[ \sum_i \text{Cov}(\mathbf{g}_i) + \sum_{i \neq j} \text{Cov}(\mathbf{g}_i, \mathbf{g}_j) \right]$$

代入 $\text{Cov}(\mathbf{g}_i) = \boldsymbol{\Sigma}$ 和 $\text{Cov}(\mathbf{g}_i, \mathbf{g}_j) = \boldsymbol{\Sigma}_{\text{cross}}$:

$$= \frac{1}{N^2}\left[ N \boldsymbol{\Sigma} + N(N-1)\boldsymbol{\Sigma}_{\text{cross}} \right] = \frac{1}{N}\boldsymbol{\Sigma} + \frac{N-1}{N}\boldsymbol{\Sigma}_{\text{cross}}$$

取 trace:

$$\text{Tr}(\text{Cov}(\hat{\mathbf{g}}_{\text{DRIS}})) = \frac{\sigma^2}{N} + \frac{N-1}{N}\rho\sigma^2 = \sigma^2\left(\rho + \frac{1-\rho}{N}\right)$$

**Intuition**: 
- $N=1$ (conventional DR): variance = $\sigma^2$, 单 instance 的 full noise.
- $N \to \infty$: variance $\to \rho \sigma^2$, 也就是 cross-covariance floor.
- 当 $\rho < 1$ (即 cross-covariance 严格小于 marginal covariance, 这通常成立 because 不同 instance 的 i.i.d. $\mathbf{c}^{(i)}$ 引入 instance-specific perturbation), $N > 1$ 就 strict variance reduction.

$\rho \sigma^2$ 这个 floor 的来源是 shared action 引入的 coupling: 所有 instance 都执行同一个 action, 所以它们的 return 不完全独立, gradient 之间有 intrinsic correlation. 这个 floor 是 inherent 的, 跟 sample size 无关, 但仍然比 $\sigma^2$ 小.

这跟 ensemble methods ([Chua et al. 2018 PETS](https://arxiv.org/abs/1808.04271)) 有相似 philosophy, 但 PETS 是用 ensemble 来 model epistemic uncertainty, 这里是用 ensemble 来 estimate aleatoric uncertainty 并 reduce gradient noise.

### 3.3 Asymptotic Unmasking: 为什么 DRIS 学 robust policy

这个 theorem 是 paper 最 insightful 的部分. 它解释了为什么 DRIS 不只 reduce variance, 还 **implicit 优化一个 regularized objective**, 这个 regularization 就是 robustness penalty.

**Setup**: 把 reward 取负作为 cost $C_t(\mathbf{s}) = -r(\mathbf{s}, \mathbf{a}_t)$, DRIS 的 empirical objective 是:

$$\hat{\mathcal{J}}_N(\theta) = -\sum_t \gamma^t \underbrace{\left(\frac{1}{N}\sum_i C_t(\mathbf{s}_t^{(i)})\right)}_{\hat{J}_{N,t}}$$

**Assumption (Local Geometry)**: $C_t(\mathbf{s})$ 在 mean $\boldsymbol{\mu}_t = \mathbb{E}[\mathbf{s}_t \mid \mathbf{a}_t]$ 附近 twice differentiable 且 locally convex, 做二阶 Taylor 展开:

$$C_t(\mathbf{s}) \approx C_t(\boldsymbol{\mu}_t) + \mathbf{g}_t^\top(\mathbf{s} - \boldsymbol{\mu}_t) + \frac{1}{2}(\mathbf{s} - \boldsymbol{\mu}_t)^\top \mathbf{H}_t (\mathbf{s} - \boldsymbol{\mu}_t)$$

- $\mathbf{g}_t = \nabla_\mathbf{s} C_t(\boldsymbol{\mu}_t)$: Jacobian (一阶)
- $\mathbf{H}_t = \nabla_\mathbf{s}^2 C_t(\boldsymbol{\mu}_t) \succeq 0$: Hessian, positive semi-definite (local convexity)

代入 $\hat{J}_{N,t}$:

$$\hat{J}_{N,t} \approx C_t(\boldsymbol{\mu}_t) + \mathbf{g}_t^\top \underbrace{\left(\frac{1}{N}\sum_i (\mathbf{s}_t^{(i)} - \boldsymbol{\mu}_t)\right)}_{\bar{\boldsymbol{\delta}}_{N,t}} + \frac{1}{2}\text{Tr}\left(\mathbf{H}_t \underbrace{\left(\frac{1}{N}\sum_i (\mathbf{s}_t^{(i)} - \boldsymbol{\mu}_t)(\mathbf{s}_t^{i} - \boldsymbol{\mu}_t)^\top\right)}_{\hat{\boldsymbol{\Sigma}}_{N,t}}\right)$$

这里用 cyclic property $\mathbf{x}^\top \mathbf{A}\mathbf{x} = \text{Tr}(\mathbf{A}\mathbf{x}\mathbf{x}^\top)$.

- $\bar{\boldsymbol{\delta}}_{N,t}$: sample mean error, 期望 0
- $\hat{\boldsymbol{\Sigma}}_{N,t}$: empirical second moment, 期望是 $\boldsymbol{\Sigma}_t = \text{Cov}(\mathbf{s}_t \mid \mathbf{a}_t)$ (physics-induced aleatoric variance)

**Theorem B.1 (Asymptotic Unmasking)**: 当 $N \to \infty$, 由 Weak Law of Large Numbers:
- $\bar{\boldsymbol{\delta}}_{N,t} \xrightarrow{P} \mathbf{0}$ → linear noise term 消失
- $\hat{\boldsymbol{\Sigma}}_{N,t} \xrightarrow{P} \boldsymbol{\Sigma}_t$ → 二阶项收敛

所以:

$$\hat{J}_{N,t} \xrightarrow{P} C_t(\boldsymbol{\mu}_t) + \frac{1}{2}\text{Tr}(\mathbf{H}_t \boldsymbol{\Sigma}_t)$$

整个 objective 变成:

$$\hat{\mathcal{J}}_N(\theta) \approx -\sum_t \gamma^t \left[ C_t(\boldsymbol{\mu}_t) + \frac{1}{2}\text{Tr}(\mathbf{H}_t \boldsymbol{\Sigma}_t) \right]$$

**核心 intuition**: 
- $N=1$ (conventional DR): 第一阶 noise $\mathbf{g}_t^\top \bar{\boldsymbol{\delta}}_{1,t}$ dominates, optimizer "chasing random first-order fluctuations", 把 aleatoric noise 当成 gradient signal, robustness signal 被掩盖.
- $N \gg 1$ (DRIS): first-order noise 被 average 掉, second-order term $\frac{1}{2}\text{Tr}(\mathbf{H}_t \boldsymbol{\Sigma}_t)$ 浮出来. 这个 term 因为 $\mathbf{H}_t \succeq 0$, 是 **non-negative**, 作为 penalty 出现, 让 policy 倾向于选那些 **降低 physics-induced variance $\boldsymbol{\Sigma}_t$ 的 actions**. 这就是 robustness: 选 action 让 outcome distribution 在不同 physics 下都 narrow.

这个结果跟 **risk-sensitive RL** / **distributional RL** ([Cassandra et al.](https://www.cs.cmu.edu/~cassandra/papers/ai96.pdf)) 的 variance penalty 概念相通, 但 DRIS 不需要显式 reward shaping, 是从 ensemble averaging 里 emergent 出来的. 也跟 [OpenAI Rubik's cube](https://arxiv.org/abs/1910.07113) 的 frame-stacking + DR 有 philosophy 上的呼应, 不过那里是 explicit randomization, 这里是 structured ensemble propagation.

### 3.4 Sim-to-Real Transfer Bound

这是 paper 给的 formal generalization bound. 设 source distribution $p_S(\mathbf{c})$ (sim 里用来 train 的 uniform distribution), target distribution $p_T(\mathbf{c})$ (real world 真实 physics distribution, 未知). 我们要 minimize real-world expected cost $\mathcal{J}_T(\theta) = \mathbb{E}_{\mathbf{c} \sim p_T}[\mathcal{L}(\theta, \mathbf{c})]$, 但只能 optimize empirical $\hat{\mathcal{J}}_N(\theta) = \frac{1}{N}\sum_i \mathcal{L}(\theta, \mathbf{c}^{(i)})$ with $\mathbf{c}^{(i)} \sim p_S$.

**Definition (Rademacher Complexity)**: 给 sample $\hat{\mathcal{C}} = \{\mathbf{c}^{(1)}, \dots, \mathbf{c}^{(N)}\}$, Rademacher complexity 衡量 function class $\mathcal{L}_\Theta = \{\mathcal{L}(\theta, \cdot): \theta \in \Theta\}$ 拟合 random sign 的能力:

$$\hat{\mathfrak{R}}_{\hat{\mathcal{C}}}(\mathcal{L}_\Theta) = \mathbb{E}_\boldsymbol{\sigma}\left[\sup_{\theta \in \Theta} \frac{1}{N}\sum_i \sigma_i \mathcal{L}(\theta, \mathbf{c}^{(i)})\right]$$

$\sigma_i \in \{-1, +1\}$ i.i.d. Rademacher variables.

**Theorem B.2 (Transfer Bound)**: 对任意 $\delta \in (0,1)$, with probability $\geq 1-\delta$:

$$\mathcal{J}_T(\theta) \leq \hat{\mathcal{J}}_N(\theta) + \underbrace{2\mathfrak{R}_N(\mathcal{L}_\Theta) + 2B\sqrt{\frac{\ln(1/\delta)}{2N}}}_{\text{Generalization Gap (reducible via } N\text{)}} + \underbrace{d_{\mathcal{L}_\Theta}(p_S, p_T)}_{\text{Physics Mismatch (irreducible)}}$$

- $B = r_{\max}/(1-\gamma)$: reward bounded 的 cumulative return upper bound
- $d_{\mathcal{L}_\Theta}(p_S, p_T) = \sup_\theta \left| \int \mathcal{L}(\theta, \mathbf{c}) (p_T(\mathbf{c}) - p_S(\mathbf{c})) d\mathbf{c} \right|$: **Integral Probability Metric (IPM)** between $p_S$ and $p_T$, 衡量 distribution shift.

**Proof intuition**:
- Step 1 (Estimation error): 用 McDiarmid's inequality + symmetrization, 得到 $|\mathcal{J}_S(\theta) - \hat{\mathcal{J}}_N(\theta)| \leq 2\mathfrak{R}_N + 2B\sqrt{\ln(1/\delta)/(2N)}$. 这就是 standard statistical learning theory.
- Step 2 (Transfer error): $|\mathcal{J}_T - \mathcal{J}_S| \leq \sup_\theta |\mathbb{E}_{p_T}\mathcal{L} - \mathbb{E}_{p_S}\mathcal{L}| = d_{\mathcal{L}_\Theta}(p_S, p_T)$, IPM 项.

**Implication**: 
- Generalization Gap 通过 $N$ reduce ($\propto 1/\sqrt{N}$), 所以 DRIS 用大 $N$ 直接缩 gap.
- Physics Mismatch 项是 inherent 的, 即使 $N \to \infty$ 也消不掉. 但只要 $p_S$ 覆盖了 $p_T$ 的 support (i.e., real physics 在 sampled 范围内), IPM 是 bounded 的.

跟 [BayesSim (Ramos et al. 2019)](http://www.roboticsproceedings.org/rss15/p35.pdf) 和 [Bayesian DR](https://arxiv.org/abs/1906.09696) 不同, 它们 adapt $p_S$ 让它 close to $p_T$, 但需要 real data; DRIS 是 zero-shot, 假设 $p_S$ 已经覆盖足够, focus 在 reduce estimation error.

---

## 4. Reactive Catching Task Instantiation

### 4.1 Task Setup

任务: 7-DoF Franka FR3 robot 末端 rigidly 装 flat plate (无 passive stabilization, 不像 cup 或 net), 接住飞过来的 ball. Plate 表面 neoprene foam padding (smooth, 让 ball 容易滚, 增加难度).

为什么这个 task 很 challenging:
- Flat plate 没 mechanical stabilization, ball 接触后很容易 bounce off 或 roll away
- 需要 millisecond 级 reactive motion
- Plate orientation, contact timing, lateral force 任何一项误差都会失败

### 4.2 Motion Frame

为了 invariance to ball incoming direction, 定义 motion frame:
- Origin: plate 初始中心 $\mathbf{p}_0^e$
- Z 轴: 垂直向上
- X-Z 平面: 包含 ball 初始中心

从 robot base frame 到 motion frame 的 transform: ${}^bT_m = (R_z(\phi), \mathbf{p}_0^e) \in SE(3)$, $R_z(\phi) \in SO(3)$ 是绕 Z 轴 angle $\phi$ 的旋转.

### 4.3 State, Action, Reward

**State** $\mathbf{s}_t = (\mathbf{d}_t, \mathbf{v}_t) \in \mathbb{R}^6$:
- $\mathbf{d}_t = R_z(\phi)^\top (\mathbf{p}_t^o - \mathbf{p}_t^e) \in \mathbb{R}^3$: ball 相对 plate 中心的 displacement, 表达在 motion frame
- $\mathbf{v}_t = R_z(\phi)^\top \mathbf{v}_t^o \in \mathbb{R}^3$: ball velocity 在 motion frame

**Action** $\mathbf{a}_t = (\boldsymbol{\delta}_t, \mathbf{u}_t) \in \mathbb{R}^5$:
- $\boldsymbol{\delta}_t \in \mathbb{R}^3$: plate 中心 displacement command (motion frame), 对应 base frame 目标位置 $\tilde{\mathbf{p}}_t^e = R_z(\phi)\boldsymbol{\delta}_t + \mathbf{p}_t^e$
- $\mathbf{u}_t = (\alpha_t, \beta_t) \in \mathbb{R}^2$: plate tilting configuration
  - $\alpha_t \in [0, 2\pi)$: 水平 axis 方向, $\mathbf{w}_t = (\cos\alpha_t, \sin\alpha_t, 0)^\top$
  - $\beta_t \in [0, \pi/4)$: rotation angle about $\mathbf{w}_t$, 改变 plate normal $\mathbf{n}_t$

**Reward** 分解: 把 $\mathbf{d}_t, \mathbf{v}_t$ 分解成 perpendicular 和 parallel 分量:

$$d_\perp = \mathbf{d}_t \cdot \mathbf{n}_t, \quad d_\parallel = \|\mathbf{d}_t - d_\perp \mathbf{n}_t\|$$
$$v_\perp = \mathbf{v}_t \cdot \mathbf{n}_t, \quad v_\parallel = \|\mathbf{v}_t - v_\perp \mathbf{n}_t\|$$

$$r_t = r_v + r_p$$
$$r_v = \frac{1}{2}\exp\left(-\frac{v_\parallel^2}{\eta^2}\right) + \frac{1}{2}\exp\left(-\frac{\max\{v_\perp, -0.1\}^2}{\eta^2}\right)$$
$$r_p = -\mathbb{1}\{d_\perp < 0 \vee d_\parallel > l_e\}$$

- $\eta = 0.25$: decay coefficient, 控制 velocity sensitivity
- $l_e = 12\text{cm}$: plate 半长
- $r_v \in (0, 1]$: ball velocity 越低 reward 越高
  - 第一项: tangential velocity 低 reward (避免 ball 滚动)
  - 第二项: perpendicular velocity 用 max with -0.1, 意思是 downward velocity 比 0.1 m/s 还大时不 penalize (因为 ball 在下落中, 还没到 catching phase)
- $r_p = -1$: ball 在 plate 下方或飞出 plate 边界时大 penalty

### 4.4 FiLM-Conditioned Policy

Policy network 用 **FiLM (Feature-wise Linear Modulation)** ([Perez et al. 2018](https://arxiv.org/abs/1709.07871)) 把 plate tilting orientation $\mathbf{u}_t$ 作为 conditional signal 注入 latent feature:

$$\tilde{\mathbf{z}}_t = \text{FiLM}(\mathbf{z}_t, \mathbf{u}_t) = \lambda(\mathbf{u}_t) \odot \mathbf{z}_t + \mu(\mathbf{u}_t)$$

- $\lambda(\mathbf{u}_t), \mu(\mathbf{u}_t) \in \mathbb{R}^{d_z}$: 两个 small NN 输出的 scaling 和 bias
- $\odot$: element-wise multiplication
- 然后 $\mathbf{a}_t = \text{MLP}(\tilde{\mathbf{z}}_t)$

**为什么用 FiLM**: 当 ball 在 plate 上时, plate 的 tilting 改变 gravity 沿 plate tangential 的 component, 改变 contact direction, 直接影响 ball acceleration. 用 $\mathbf{u}_t$ modulate latent feature, 让 policy 在不同 tilt 配置下 generate physically consistent action. 这比单纯 concatenate $\mathbf{z}_t$ 和 $\mathbf{u}_t$ 更 expressive, 因为 affine transform 让网络可以根据 tilt 改变 latent 的几何意义.

### 4.5 Torque Control

Action $\mathbf{a}_t$ 转成 desired plate pose $\tilde{\mathbf{T}}_t = (\tilde{\mathbf{R}}_t^e, \tilde{\mathbf{p}}_t^e) \in SE(3)$. Orientation 用 **Rodrigues' formula**:

$$\tilde{\mathbf{R}}_t^e = R_z(\phi)\left(\mathbb{I} + \hat{\mathbf{w}}_t \sin\beta_t + \hat{\mathbf{w}}_t^2(1 - \cos\beta_t)\right)$$

$\hat{\mathbf{w}}_t \in \mathbb{R}^{3\times 3}$ 是 $\mathbf{w}_t$ 的 skew-symmetric matrix. Rodrigues' formula 给 axis-angle 到 rotation matrix 的 closed-form.

然后 IK 得到 desired joint config $\tilde{\mathbf{q}}_t = \text{IK}(\tilde{\mathbf{T}}_t) \in \mathbb{R}^M$ ($M=7$ for FR3). Joint-space PD torque controller (无 inertia feedforward, 因为 desired joint acceleration 是 0):

$$\boldsymbol{\tau} = \mathbf{K}(\tilde{\mathbf{q}}_t - \mathbf{q}) - \mathbf{D}\dot{\mathbf{q}} + \mathbf{C}(\mathbf{q}, \dot{\mathbf{q}}) + \mathbf{g}(\mathbf{q})$$

- $\mathbf{q}, \dot{\mathbf{q}} \in \mathbb{R}^M$: 当前 joint 角度和速度
- $\mathbf{K}, \mathbf{D} \in \mathbb{R}^{M \times M}$: stiffness 和 damping gain matrices
- $\mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})$: Coriolis term
- $\mathbf{g}(\mathbf{q})$: gravity compensation

Real deploy 时 $\mathbf{K}, \mathbf{D}$ 在 real robot 上 tune 一下让 controller behavior 匹配 simulator. 这是 paper 唯一用到的 real-world information, 没用 real data 训 policy.

### 4.6 DRIS for Catching

Domain parameters $\mathcal{C} \subset \mathbb{R}^4$:
- Ball radius: $[2, 4]$ cm
- Static friction: $[0, 0.1]$
- Dynamic friction: $[0, 0.1]$
- Restitution: $[0.4, 0.7]$

每个 episode spawn $N$ balls, 各自 i.i.d. sample 物理 properties. Disable ball-ball collision 和 ball-non-plate-link collision (只 enable ball-plate collision), 这样训练稳定, 不会因为 unmodeled contact 让 DRIS 崩. 这是 trick, 实际 deployment 时只有一个球.

Pretrain AE 用 $N=200$ 的 random action rollout, 50 episodes × 128 parallel envs = 128,000 samples, 100 epochs ~10 min. PPO 训 1000 epochs ~2h. Single RTX 3060 12GB.

---

## 5. 实验数据详解

### 5.1 Simulation Experiments

**Observation Noise**: 加 Gaussian noise 到 ball state, base std $\sigma = 1\text{cm}$ (position), $5\text{cm/s}$ (velocity), scaled by factor 1, 2, 3, 4. 从 Fig. 4 看:
- E2E 和 $N=1$ DRIS 在 perfect observation 时差不多, 但 noise 增大时 degrade 显著
- $N=10$ 已经有 substantial robustness boost
- $N=50, 200$ 更 robust

**Execution Error**: 加 uniform noise $[-0.05, 0.05]$ rad 到 desired joint positions, 模拟 controller tracking inaccuracy. Fig. 5 显示类似 pattern.

**OOD Physics**: train 在 restitution $[0.4, 0.7]$, test 在 $[0.7, 0.8]$. Fig. 6 显示大 $N$ policy 更 generalizes 到 unseen physics.

**Scalability Study**: 关键实验, 检验 DRIS 性能 gain 是不是仅仅因为 "看了更多 ball interactions". 把 E2E 扩到 128×50 single-ball environments (跟 128 envs × 50 balls DRIS 总 interaction 数 match):
- E2E 128×50: success rate 0.73, VRAM 2 GB
- DRIS 128 envs × 50 balls: success rate **0.89**, VRAM 0.71 GB

**Insight**: DRIS 不只是 "看更多数据", 它是 "structured ensemble propagation". 同样的 total ball interaction 数, DRIS 用更少 VRAM (0.71 vs 2 GB) 拿到更高 success rate (0.89 vs 0.73). 这是因为 DRIS 的 shared action 把所有 instance coupling 起来, gradient variance reduction 是 free lunch.

### 5.2 Real-Robot Experiments

Setup: Franka FR3, plate 上 3D-printed ramp with interchangeable sections (R = 0.13, 0.20, 0.32 m), 4 种 ball: wiffle, rubber, ping-pong, foam. 两个 camera 80 FPS 做 color segmentation + contour-based circle fitting + triangulation + parabolic curve fitting, OLS 在 0.1s sliding window 估计 velocity. Policy inference 在 AMD Ryzen 9 5950X @ 3.4 GHz.

**Baselines**:
1. **VelTrack**: hand-crafted, plate 水平 follow ball velocity, orientation 对准 incoming direction. 太 reactive 不够, 几乎全失败 (3/60, 都是 ball 卡在 plate-finger 缝里).
2. **E2E**: end-to-end single-ball trained policy, jerky motion, knock ball off plate 或 hit power limit. 8/60.
3. **DRIS $N=200$**: 41/60 = **68%** success rate. Catch all 4 types of ball.

| Ball | Ramp R | VelTrack | E2E | DRIS |
|------|--------|----------|-----|------|
| Wiffle | 0.13 | 0/5 | 0/5 | 5/5 |
| Wiffle | 0.20 | 0/5 | 2/5 | 4/5 |
| Wiffle | 0.32 | 0/5 | 0/5 | 3/5 |
| Rubber | 0.13 | 0/5 | 0/5 | 2/5 |
| Rubber | 0.20 | 0/5 | 0/5 | 5/5 |
| Rubber | 0.32 | 0/5 | 1/5 | 2/5 |
| Ping-pong | 0.13 | 0/5 | 1/5 | 4/5 |
| Ping-pong | 0.20 | 2/5 | 1/5 | 5/5 |
| Ping-pong | 0.32 | 0/5 | 0/5 | 4/5 |
| Foam | 0.13 | 0/5 | 0/5 | 2/5 |
| Foam | 0.20 | 1/5 | 1/5 | 3/5 |
| Foam | 0.32 | 0/5 | 2/5 | 2/5 |
| **Total** | | **3/60** | **8/60** | **41/60** |

**Failure modes**: initial contact 不能足够 dissipate momentum, 或 tracking latency 让 ball redirect 不利, robot over-stretch recover.

**Qualitative**:
- Human throw ball, policy 也能 catch (虽然 human throw inconsistent)
- Generalize 到 pure balancing task (Fig. 9 bottom left, foam ball balancing, hard-coded 30° rotation 触发 rolling, policy 接管)
- Catch irregularly shaped object (toy apple, toy strawberry, wooden cube). Wooden cube 偶尔成功因为 highly unpredictable impact dynamics.

---

## 6. Limitations & 我的 Commentary

Paper 自己列的 limitations:
1. **High-dim input 的 issue**: encoder 对 low-dim state 有效, 但 visual/geometric input 可能需要大 model 和 substantial pretrain.
2. **Parallel propagation 的 computational cost**: 多 instance, 多 mutual contact 时 collision resolving 会成 bottleneck.
3. **Shared action 的 divergence issue**: 如果 instance state 太分散, shared action 没法 reconcile, 学习不稳. Paper 引 [Schaal & Atkeson 1993 robot juggling](https://ieeexplore.ieee.org/document/518049): 在 dissipative 系统 (catching, 球能量被 plate 吸收) 里避免 divergence 容易, 但在 energy-injecting 系统 (throwing) 里 variation 会放大, DRIS 可能不 work.

**我的额外 commentary / intuition**:

a. **DRIS vs Ensemble Model-based RL (PETS)**: PETS ([Chua et al.](https://arxiv.org/abs/1808.04271)) 用 ensemble of dynamics models 估 epistemic uncertainty, 然后做 planning. DRIS 是用 ensemble of physics instances 估 aleatoric uncertainty, 做 policy learning. 前者 model 不确定性, 后者 domain 不确定性. 两者可以 combine, 但 DRIS 不需要 learned dynamics model, 直接用 simulator, 这对 high-speed contact 是优势.

b. **DRIS vs Belief-Space Planning (POMDP)**: Belief-space planning ([Bry & Roy 2011](https://ieeexplore.ieee.org/document/5980508)) 在 POMDP 里 propagate belief 显式, 但通常需要 discretization 或 sampling-based, 计算贵. DRIS 把 belief propagation "embedded" 在 policy training 里, deployment 时只需要 forward pass. 类似 [PlaNet/Dreamer](https://arxiv.org/abs/1912.01603) 把 planning 放 latent space, 但 DRIS 不 latent imagine, 而是 latent encode ensemble state.

c. **DRIS vs Caging in Time**: 同作者之前的 [Caging in Time (Wang et al. 2025)](https://journals.sagepub.com/doi/10.1177/02783649251343926) 也是 set-based representation for robust manipulation, 但用 caging 约束, 这里用 DRIS propagation. 同一 lab 的延续 work.

d. **Asymptotic unmasking 的 deep connection**: $\frac{1}{2}\text{Tr}(\mathbf{H}_t \boldsymbol{\Sigma}_t)$ 这个 term 跟 **Bayesian experimental design** 里的 expected information gain, **second-order Taylor approximations in stochastic optimization**, 以及 **Laplace approximation** 的 Hessian × Cov 都有 connection. 在 robust optimization 文献里这就是 worst-case-aware 的 first-order surrogate. DRIS 把它免费拿到了.

e. **为什么 dissipative 系统特别适合**: Schaal-Atkeson juggling paper 说 dissipative 系统有 self-stabilization property, open-loop 都能稳定. DRIS 利用这个: 在 dissipative 系统, 不同 physics instance 的 state trajectory 会 converge 而非 diverge (energy 被吸收), 所以 shared action 能 cover 所有 instance 的 state 而不过度 compromise. 在 throwing 这种 energy-injecting, divergence 放大, shared action 没法同时 handle 多种 physics. 这跟 [Active DR](https://arxiv.org/abs/1906.09696) 的 adaptive sampling 可能 complement: 用 DRIS 在收敛区域, 用 active DR 在发散区域.

f. **Multi-agent extension intuition**: DRIS 用 shared action, 但如果允许多 action (per-instance action), 就变成 multi-agent RL with common reward. 但这违背 zero-shot deployment (real world 只有 1 ball). 不过可以做 mixture: training 时多 action with shared component, deploy 时只保留 shared component. 这是 possible future direction.

g. **Connection to Data Augmentation**: 在 supervised learning 里, data augmentation (e.g., random crop, color jitter) 是 single-instance variant. DRIS 类似 "instance-level augmentation with shared gradient", 跟 [SimCLR](https://arxiv.org/abs/2002.05709) 的 multi-view contrastive 有 philosophy 相似: multiple views 共享 representation, 但 DRIS 是共享 action 而非 representation.

h. **The $\rho$ floor 的物理含义**: $\rho \sigma^2$ 是 cross-covariance, 由 shared action 引入. 如果 action space 很 expressive, policy 可以让不同 instance 几乎独立 evolve (降低 coupling), $\rho \to 0$, variance reduction 接近 $1/N$. 如果 action space 限制大, coupling 强, $\rho$ 大, $N$ 加大收益小. 这是个 interesting design tradeoff, paper 没 explore, 但 future work 可以 investigate adaptive action space.

---

## 7. 总结

**TL;DR**: 
1. DRIS = 同时 propagate $N$ 个 randomized physics instances, shared action, average reward, variance-reduced gradient.
2. 理论: exact particle propagation of belief, gradient variance 从 $\sigma^2$ 降到 $\sigma^2(\rho + (1-\rho)/N)$, asymptotic unmasking 隐式优化 robustness penalty $\frac{1}{2}\text{Tr}(\mathbf{H}_t \boldsymbol{\Sigma}_t)$, sim-to-real generalization bound 通过 Rademacher + IPM 分解成 reducible 和 irreducible 部分.
3. 实例化: flat-plate reactive catching on FR3, point-cloud AE encoder + FiLM-conditioned MLP policy, PPO training.
4. 实验: $N=200$ 在 sim 里 robust to noise/error/OOD physics, 在 real 上 zero-shot 68% success rate across 4 ball types × 3 ramp radii.

**Key intuition**: DRIS 不是 "更多数据", 是 "structured ensemble belief propagation embedded in policy training". Shared action 把 instance coupling 起来, automatic variance reduction + implicit robustness regularization, 免费 zero-shot transfer.

**最值得读的部分**: Appendix B 的 theoretical analysis, 特别是 Theorem B.1 (Asymptotic Unmasking), 这是 paper 的核心 insight. 主文 Section IV 的 task formulation 也写得清晰.

### Reference Links

- Paper (Rice DRIS, arXiv 待更新): https://arxiv.org/abs/2509.xxxxx (paper 应该会上 arxiv)
- ManiSkill3: https://maniskill.github.io/
- PPO: https://arxiv.org/abs/1707.06347
- FiLM: https://arxiv.org/abs/1709.07871
- PointNet: https://arxiv.org/abs/1612.00593
- Point cloud AE (Achlioptas): https://arxiv.org/abs/1707.02392
- OpenAI Rubik's cube (Akkaya): https://arxiv.org/abs/1910.07113
- OpenAI Dactyl (Andrychowicz): https://arxiv.org/abs/1808.00177
- Domain Randomization (Tobin): https://arxiv.org/abs/1703.06907
- Dynamics Randomization (Peng): https://arxiv.org/abs/1710.06537
- Dreamer (Hafner): https://arxiv.org/abs/1912.01603
- BayesSim (Ramos): https://arxiv.org/abs/1906.09696
- PETS (Chua): https://arxiv.org/abs/1808.04271
- Active DR (Mehta): https://arxiv.org/abs/1906.09696
- DROP-O (Tiboni): https://arxiv.org/abs/2208.05858
- Schaal-Atkeson juggling: https://ieeexplore.ieee.org/document/518049
- Caging in Time (Wang et al.): https://journals.sagepub.com/doi/10.1177/02783649251343926
- SimCLR: https://arxiv.org/abs/2002.05709
- Rodrigues' formula reference: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
- Liouville equation: https://en.wikipedia.org/wiki/Liouville%27s_theorem_(Hamiltonian)
- McDiarmid's inequality: https://en.wikipedia.org/wiki/McDiarmid%27s_inequality
- Rademacher complexity: https://en.wikipedia.org/wiki/Rademacher_complexity
- Integral Probability Metric: https://en.wikipedia.org/wiki/Integral_probability_metric
- Particle filter (Doucet): https://www.stats.ox.ac.uk/~doucet/doucet_defreitas_gordon_smcbookintro.html

希望 这个深度讲解 能 build 你对 DRIS framework 的 intuition. 如果你 want 我进一步 elaborate 某个 specific aspect (比如 FiLM 在 latent modulation 的几何意义, 或 asymptotic unmasking 跟 Hessian-based meta-learning 的 connection), 告诉我.
