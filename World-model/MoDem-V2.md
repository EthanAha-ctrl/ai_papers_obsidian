---
source_pdf: MoDem-V2.pdf
paper_sha256: 57ade24ddc477b56704b7730aa74b6444cca46f2ecea8cbf8b85a6e413da6a53
processed_at: '2026-08-05T19:54:27-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 MoDem-V2

## 先讲个故事

你想让 robot arm 学一个动作 —— 比如把瓶子从横躺翻到竖直。传统做法是 sim2real: 在 simulation 里狂 train, 然后 transfer 到 real robot。但 contact-rich 任务 sim 很难建准, 物理 engine 对 friction / contact / deformable 的模拟都 garbage in garbage out。

那能不能直接在 real robot 上 train? 可以, 但有两个大坑:

1. **Sample efficiency**: real robot 一小时也就几百次 interaction, RL 要百万步, 不够烧
2. **Safety**: robot 一瞎试就撞坏自己或者 environment

之前有人提出 MoDem (https://arxiv.org/abs/2212.05698), 思路是: 给 10 个 human demo 先 warm start, 然后用 visual world model 做 MBRL, 在 simulation 里 sample-efficient 得一塌糊涂。结果搬到 real robot 上 —— **头两个 episode 直接把 robot 弄 fault 了**。原因很 simple: MoDem 的 planner 一上来就从整个 action space 乱采样, world model 又菜又爱玩, 给出乱七八糟的 value estimate, robot 就按这个乱动, 瞬间 torque 超限。

paper 作者一开始想用最 intuitive 的 fix: 在 reward 里加 torque penalty。结果发现没用, 训练初期 robot 还是先抽风再说。Figure 7 里那行 baseline 显示, 加了 penalty 后 spike 还是在, 只是后面 torque 稍微低一点。在 in-hand reorientation task 上干脆学不出来。

所以问题本质是: **你不能让 robot 在它没见过的区域乱试, 因为它的 world model 在那块完全 hallucinate**。需要 conservative。

---

## MoDem-V2 三个 trick

### Trick 1: Policy Centering — 别从零开始乱采

原版 MoDem 的 MPC (Model Predictive Control) 是这样: 从一个 Gaussian distribution $\mathcal{N}(\mu, \sigma^2)$ 采样一堆 candidate action sequences, 用 world model roll forward, 看哪个 trajectory 的 value 高, 选 top 64 个 elites, 用它们的 weighted average 当 action 执行。初始 $\mu = 0, \sigma = 2$ (action space 已 normalized 到 $[-1, 1]$ 左右)。

问题: $\mathcal{N}(0, 2^2)$ 这个 prior 覆盖了整个 action space, 大部分 sample 出来的 action 都在 BC demo 没见过的区域, world model 对这些 action 的预测是纯 hallucination。

MoDem-V2 改成: 训练初期, 直接从 BC policy $\pi_\theta^{BC}$ 采 actions 当 trajectory seeds:
$$\Gamma := \{\mathbf{a}_0\}^N \sim \pi_\theta^{BC}$$

变量含义:
- $\Gamma$: 一组 candidate trajectories, 每个 trajectory 是一个 action sequence
- $N$: population size = 510 (Table I)
- $\pi_\theta^{BC}$: behavioral cloning 训出来的 policy, 只见过 demo 数据, 输出的 action 都在 demo 附近

直觉: BC policy 已经知道 "大概应该怎么动", 用它的 output 当采样中心, 等于在 action space 里画了个小圈说 "只在这块找"。world model 至少在这块见过 data, 给的 value estimate 可信一些。

### Trick 2: Agency Transfer — 慢慢把控制权交给 planner

光 centering 还不够。如果永远只信 BC, 那 robot 永远不会超越 demo 的水平, 学不到新东西。

MoDem-V2 的方案是搞个超参 $\alpha$, 控制用 BC 还是 MPC:
- 训练开始 $\alpha = 0$, 完全用 BC
- 线性 ramp 到 $\alpha = 1.0$, 完全用 MPC
- schedule 长度: 25k steps (sim incline pushing) 或 100k steps (其他)

Algorithm 1 Line 2: `if rand() > α then` 用 BC mode, 否则用 MPC mode。

直觉: 给 world model 一段时间 "慢慢长大"。前期它菜, 让 BC 老师带; 后期它学好了, BC 老师退场, 让学生自己 plan。类似 DAgger (https://arxiv.org/abs/1011.0686) 的 interactive imitation, 但更 smooth。

这里有个细节: agency transfer 不只是 "用 BC 还是 MPC" 的 hard switch, 而是 probability $\alpha$ 上的 soft switch。每个 timestep 独立 sample, 所以训练中期是 BC 和 MPC 混着用, 相当于一个 stochastic curriculum。

### Trick 3: Actor-Critic Ensembles — 不信单一 value function

原版 MoDem 用单个 critic $Q_\theta$。已知 single critic 容易 overestimation bias (TD3 paper, https://arxiv.org/abs/1806.09414 已经证明了) —— policy 会找到 critic 的高估 region 然后狠狠 exploit。

MoDem-V2 用 5 个 actor-critic pairs。每个 actor $i$ 训练时最大化它对应的 critic $Q^i$, 但 evaluation 时, **不用同一个 actor 的 critic 评估它的 action**, 而是用所有 critic 的 ensemble。

具体公式 (Algorithm 1 Line 14-16):
$$\phi_\Gamma^{1:M} = \phi_\Gamma + \gamma^h Q_\theta^{1:M}(\mathbf{z}_H, \mathbf{a}_H)$$
$$\phi_\Gamma = w_1 \cdot \text{mean}(\phi_\Gamma^{1:M}) + w_2 \cdot \text{std}(\phi_\Gamma^{1:M})$$

变量含义:
- $\phi_\Gamma$: trajectory $\Gamma$ 的 total value estimate (reward sum + terminal value)
- $\gamma = 0.95$: discount factor
- $h = 5$: planning horizon
- $\mathbf{z}_H, \mathbf{a}_H$: horizon 末的 latent state 和 action
- $Q_\theta^{1:M}$: $M=5$ 个 critic, 独立训练
- $w_1 = 1.0, w_2 = -10.0$ (Table I)

注意 $w_2 = -10.0$ 是 **负的**, 且绝对值是 $w_1$ 的 10 倍。意思: 如果 5 个 critic 对某条 trajectory 的 value estimate 差异大 (std 大), 就狠狠扣分。std 大 = epistemic uncertainty 高 = 这块区域 data 少 = critic 们在各自 hallucinate。

这和 PETS (https://arxiv.org/abs/1805.12114) 的 epistemic uncertainty estimation 思想一脉相承, 只是从 probabilistic ensemble NN 换成了 actor-critic ensemble。

---

## 三个 trick 怎么协同

这是 paper 最有意思的部分, 也是 ablation 实验里最 revealing 的。

单独加 Centering: 在 bin picking 上 safety violations 反而 **变多**。为什么? 我的猜测: BC policy 在 bin 边缘这种 OOD 状态下, 输出的 action 本身就 risky (BC 只会 mimic demo, demo 里没见过 bin 边缘的情况)。光 centering 没用, 需要 uncertainty detection 来 filter。

单独加 Ensemble: sample efficiency 最好, 但 safety 比 Schedule 差一点。因为 ensemble 不知道什么时候该 conservative, 它只是给 OOD trajectory 低 value, 但还是会去 explore。

单独加 Schedule (含 Centering): 最 safe, 但 sample efficiency 不是最好。因为前期太保守, 不敢 explore。

**MoDem-V2 = 三个一起**: 拿到 Ensemble 的 sample efficiency + Schedule 的 safety。

这个 synergy 是 non-trivial 的。Figure 5 显示 bin picking 上 Centering 单独反而比 MoDem 原版还差, 但 MoDem-V2 (三个合起来) 显著好。这说明每个 component 都有 failure mode, 组合起来才互相 cover。

---

## 实验数字

### Simulation (Figure 4)

| Task | Method | Sample Efficiency | Safety Violations |
|---|---|---|---|
| Planar Pushing | MoDem | 高 | 中 |
| Planar Pushing | MoDem-V2 | 高 | **极低** |
| Inclined Pushing | MoDem | 高 | 高 |
| Inclined Pushing | MoDem-V2 | 高 | **低** |
| Bin Picking | MoDem | 高 (但 drop) | 高 |
| Bin Picking | MoDem-V2 | 高 | **低** |
| In-Hand Reorient | MoDem | 中 | 极高 |
| In-Hand Reorient | MoDem-V2 | 中 | **低** |

对比 baselines:
- DAPG (有 privileged state + dense reward): 比 MoDem-V2 还慢。Visual MBRL from pixels + sparse reward 居然 beat state-based RL + dense reward, 这挺惊人的
- FERM: 简单 task 行, bin picking 和 in-hand 直接学不出来

### Real World (Figure 6)

- MoDem: 在 real robot 上头两个 episode 就 fault, 根本跑不起来
- MoDem-V2: <2 小时 online interaction 显著超过 BC policy baseline

10 个 demo + 2 小时 online = 学会 4 个 contact-rich manipulation skill, 包括 in-hand reorientation 这种高维 action space (D'Manus 10 DoF hand) 的难任务。

---

## Reward 怎么拿 (real world 巧思)

Real robot 不能像 sim 那样直接读 object pose 给 dense reward。paper 用 vision-based sparse reward, 每个任务的 detection 方式不同:

- **Planar Pushing**: LUV color space color thresholding, 检测 green object 覆盖 red goal area
- **Inclined Pushing**: 同上, gravity reset (block 自己滑下来)
- **Bin Picking**: 拍两张照 subtract (move away 拍一张, open gripper 拍一张, 差分看 object 在不在 grasp)
- **In-Hand Reorientation**: top-down depth camera 看 bottle 是否 upright

reward assignment 是 **backward in time**: episode 成功后, 从最后往前找关键时刻 (比如 gripper 离 table < 10cm 的 timestep), 那个时刻之后所有 timestep 给 +1 reward。这把 episode-level binary success 转成了一段 dense-ish reward signal, 对 TD-learning 友好得多。

这个细节没被 paper 主线 highlight, 但其实挺关键。RL 的 sample efficiency 很大程度取决于 reward shaping, 这个 backward assignment 是个 clever trick。

---

## 为什么这工作 matter

### 1. Visual MBRL 终于在 real world work 了

之前 visual MBRL (Dreamer, TD-MPC, PlaNet 等) 大多在 simulation 证明 sample-efficient。DayDreamer (https://arxiv.org/abs/2206.14176) 在 real robot 上跑过, 但任务简单, 没 demo, 没 contact-rich。MoDem-V2 是第一次 visual MBRL + demo + real world + contact-rich 全集齐。

### 2. Safety 不是靠 constraint, 是靠 conservative exploration

传统 safe RL 走 constrained MDP 路线 (CPO, https://arxiv.org/abs/1705.10528), 需要显式 constraint function。MoDem-V2 的哲学是: **不让 robot 去它 world model 没见过的区域**。这比定义 explicit safety constraint 更 scalable, 因为 real world 的 safety constraint 本身就是 unobservable 的 (thermal, wear, cable fatigue 都没法直接 sense)。

### 3. 和 LLM 的 "don't trust outside training distribution" 同源

这点你应该特别 appreciate。LLM 在 OOD prompt 上 hallucinate, visual world model 在 OOD state-action 上也 hallucinate。MoDem-V2 的三个 trick 本质都是 "检测 OOD 并避开":
- Centering: 把采样中心 anchor 到 known-good region
- Schedule: 在 model immature 时不 trust it
- Ensemble std: 直接 estimate OOD-ness

跟你讲 LLM 时说 "the model knows what it knows, and we should query its confidence" 是一回事。

---

## 我的几个 critique

1. **Environment reset 被低估了**。Paper 依赖 retractor reels / gravity / hand-coded policy 做 reset, 这些其实也是 engineering effort。Autonomous reset 在 manipulation 里还是 open problem。

2. **Generalization 没讨论**。学了 in-hand reorientation 4 小时, 换个瓶子形状能 transfer 吗? Limitations section 自己承认希望 future reuse world model, 但没实验。对比 RT-X (https://robotics-transformer-x.github.io/) 的 large-scale 路线, MoDem-V2 是 small-data 精细化路线, 两者没 merge。

3. **Ensemble std 当 epistemic uncertainty 的 assumption**。5 个 critic 独立训练但共享同样的 replay buffer, 它们的 disagreement 不完全是 epistemic uncertainty, 也有 optimization noise。更 principled 的做法是 Bayesian NN 或 deep ensemble with different data subsets (像 PETS 那样)。但工程上 5 个 critic 已经够用。

4. **为什么 $w_2 = -10$ 而不是 -1 或 -100**。Paper 没 sweep 这个。我猜 -10 是 task-specific tuned, 换 task 可能要重调。这是 MBRL 老问题: hyperparameter sensitivity。

---

## 一些延伸联想

### JEPA 路线赢了吗

MoDem 的 representation learning 用的是 BYOL/JEPA 风格的 stop-gradient + EMA target, 不用 contrastive learning, 不用 reconstruction。Equation 3:
$$\mathcal{L}_{EM} \doteq \| \Delta \mathbf{z}'_t - \text{sg}(h_\phi(\mathbf{s}'_t)) \|_2^2$$

这跟 LeCun 的 I-JEPA (https://arxiv.org/abs/2301.08243) 同源。LeCun 一直说 contrastive learning 是 wrong path, JEPA 才对。MoDem-V2 在 real robot manipulation 上的 success 算是给 JEPA 在 robotics representation learning 上的一个 vote of confidence。

### 和 Diffusion Policy 的对比

Diffusion Policy (https://diffusion-policy.cs.columbia.edu/) 是 Chi et al. 2023 的工作, 用 diffusion model 做 policy, 在 real robot manipulation 上很 work, 但纯 imitation learning, 没 online improvement。MoDem-V2 有 online RL 部分能超越 demo 水平。两者结合: diffusion policy 做 BC prior + MBRL 做 online refinement, 是个自然的 next step。

### 和 Q-Transformer / RT-2 的对比

Google 的 Q-Transformer (https://arxiv.org/abs/2309.10150) 和 RT-2 (https://arxiv.org/abs/2307.15818) 走 large-scale offline Q-learning 或 VLM 路线, 完全 bypass online RL 的 safety 问题, 靠大数据 cover 住。MoDem-V2 是 small-data + online MBRL。两条路:
- Google 路线: 暴力 data, 通用, 但 contact-rich 精细任务难 (RT-2 抓取还行, in-hand reorientation 悬)
- MoDem-V2 路线: 10 demo + 2 小时, sample-efficient, 但每个 task 要重训

我个人觉得 long term 是 merge: 用 large-scale pretrain 给 world model 一个 good initialization, 然后 MoDem-V2 风格的 online refinement 适配具体 task。

### World Model 会成为 robotics 的 GPT 时刻吗

你之前在 podcast 里聊过 "world model is the next big thing"。MoDem-V2 算是个 micro-scale 证明: visual world model + MBRL 在 real robot 上能 work。但要达到 GPT-level 的 generalization, 需要:
- Cross-task / cross-embodiment world model (现在 MoDem-V2 是 per-task 训的)
- Long-horizon planning (现在 $h=5$, 太短)
- Goal-conditioned 而不是 reward-conditioned (现在还是 MDP + reward)

这条 path 还很长, 但 MoDem-V2 算是铺了一块砖。

---

## TL;DR

MoDem-V2 = MoDem + 三个 trick (policy centering + agency transfer + ensemble uncertainty), 让 visual MBRL 第一次在 real robot 上 work, 学会 4 个 contact-rich manipulation task, 每个只要 10 demo + 2 小时 online interaction。核心哲学: **不让 world model 在它没见过的区域做决策**, 这和 LLM 里 "don't trust outside training distribution" 是同一件事。

---

参考资料:
- MoDem-V2 project: https://sites.google.com/view/modemv2W
- MoDem 原论文: https://arxiv.org/abs/2212.05698
- TD-MPC: https://arxiv.org/abs/2203.04955
- I-JEPA: https://arxiv.org/abs/2301.08243
- PETS: https://arxiv.org/abs/1805.12114
- DayDreamer: https://arxiv.org/abs/2206.14176
- SAVED: https://arxiv.org/abs/1910.10551
- DAPG: https://arxiv.org/abs/1709.10087
- TD3: https://arxiv.org/abs/1806.09414
- CPO: https://arxiv.org/abs/1705.10528
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- RT-2: https://arxiv.org/abs/2307.15818
- Q-Transformer: https://arxiv.org/abs/2309.10150
- ROBEL: https://arxiv.org/abs/1909.11664
- Open-X-Embodiment: https://robotics-transformer-x.github.io/

---

# MoDem-V2: Visuo-Motor World Models for Real-World Robot Manipulation - 深度解析

## 1. 背景: 为什么需要 MoDem-V2

这篇 paper 要解决的核心问题是: 如何在 uninstrumented real world 中用 visual MBRL 训练 contact-rich manipulation skills。前作 MoDem (Hansen et al., 2022) 在 simulation 中展示了 sample-efficient 的 visual MBRL 能力, 但是一旦搬到 real robot 上, 立刻暴露出 **aggressive exploration** 导致的 safety violations 问题 —— robot 会瞬间触发 manufacturer-specified torque limits, hardware controller 直接 fault。

关键 insight 是: 仅仅 penalize torque (例如在 reward 里加 L2 penalty on torque) 是无效的 retrospective fix, 它无法 prevent unsafe actions at the onset of exploration。这点在 Figure 2 (right) 和 Figure 7 中被实验证实 —— 加了 torque penalty 后, 训练初期仍然有 huge spike in exerted torque, 而且在 hardest 的 in-hand reorientation task 上完全 stagnates learning。

paper 的论点很直接: 需要 **conservative exploration** + **uncertainty-aware planning**, 而不是事后惩罚。

参考链接:
- MoDem 原论文: https://arxiv.org/abs/2212.05698
- TD-MPC: https://arxiv.org/abs/2203.04955
- Project page: https://sites.google.com/view/modemv2W

---

## 2. MoDem 的基础架构

MoDem 是基于 TD-MPC 的 demonstration-augmented variant。整个 system 学习 5 个组件 (Equation 1):

$$
\begin{aligned}
\text{State embedding} & \quad \mathbf{z} = h_{\theta}(\mathbf{s}) \\
\text{Latent dynamics} & \quad \mathbf{z}' = d_{\theta}(\mathbf{z}, \mathbf{a}) \\
\text{Reward predictor} & \quad \hat{r} = R_{\theta}(\mathbf{z}, \mathbf{a}) \\
\text{Terminal value} & \quad \hat{q} = Q_{\theta}(\mathbf{z}, \mathbf{a}) \\
\text{Policy guide} & \quad \hat{\mathbf{a}} = \pi_{\theta}(\mathbf{z})
\end{aligned}
$$

变量解释:
- $\mathbf{s} = \rho(\mathbf{x}, \mathbf{q})$: state, 是 RGB observation $\mathbf{x}$ (stacked frames) 与 proprioception $\mathbf{q}$ 的组合
- $\mathbf{z} \in \mathbb{R}^{50}$: latent state (latent dim = 50, 见 Table I)
- $\theta$: 所有 network parameters
- $h_{\theta}$: encoder, 把 high-dim pixel + proprio 压成 50-dim latent
- $d_{\theta}$: latent dynamics predictor
- $R_{\theta}, Q_{\theta}$: reward 和 Q 函数 (都 conditioned on latent + action)
- $\pi_{\theta}$: deterministic policy head, 用于 guide sampling-based planner

### Training objective (Equation 2-5, Appendix)

$$
\mathcal{L}(\theta) \doteq \mathbb{E}_{(\mathbf{s}, \mathbf{a}, r, \mathbf{s}')_{0:h} \sim \mathcal{B}} \left[ \sum_{t=0}^{h} \lambda^t (\mathcal{L}_{EM} + \mathcal{L}_{RE} + \mathcal{L}_{TD}) \right]
$$

变量与上下标含义:
- $\mathcal{B}$: replay buffer
- $h$: subtrajectory length (planning horizon, 在 Table I 中 = 5)
- $\lambda \in (0, 1]$: temporal coefficient (=0.5), 给时间上接近的 timestep 更大权重
- $\mathcal{L}_{EM}$: embedding prediction loss
- $\mathcal{L}_{RE}$: reward prediction loss
- $\mathcal{L}_{TD}$: TD-learning loss

**Embedding Prediction (Equation 3, BYOL-style / JEPA-style):**
$$
\mathcal{L}_{EM} \doteq \| \Delta \mathbf{z}'_t - \text{sg}(h_{\phi}(\mathbf{s}'_t)) \|_2^2
$$
- $\Delta \mathbf{z}'_t$: 由 $d_\theta$ 预测出的 next latent
- $\text{sg}(\cdot)$: stop-gradient operator
- $\phi$: exponentially moving average of $\theta$ (类似 BYOL 的 target network)
- 这就是 **JEPA (Joint-Embedding Predictive Architecture)** 思想, 与 LeCun 的 I-JEPA 同源, prevent representation collapse 靠 stop-grad + EMA target

**Reward Prediction:**
$$
\mathcal{L}_{RE} \doteq \| \hat{r}_t - r_t \|_2^2
$$

**TD-learning:**
$$
\mathcal{L}_{TD} \doteq \| \hat{q}_t - q_t \|_2^2, \quad q_t \doteq r_t + Q_\phi(\mathbf{z}'_t, \pi_\theta(\mathbf{z}'_t))
$$

**Policy objective:**
$$
\mathcal{L}_\pi(\theta) \doteq \mathbb{E}_{\mathbf{s}_{0:h} \sim \mathcal{B}} \left[ \sum_{t=0}^{h} \lambda^t Q_\theta(\mathbf{z}_t, \pi_\theta(\mathbf{z}_t)) \right], \quad \mathbf{z}_t = h_\theta(\mathbf{s}_t)
$$
gradient 只流过 $\pi_\theta$ 不流过 $Q_\theta$。

---

## 3. MoDem-V2 的三个核心改进 (Algorithm 1)

Algorithm 1 是 planning procedure, 在 inference 时调用, 是对 TD-MPC 的 MPC sampling 做改造。逐行解析:

### 3.1 Policy Centered Actions (Line 3-4)

原始 MoDem 的 MPC 是从 $\mathcal{N}(\mu, I\sigma^2)$ 中采样 actions, 其中 $\mu, \sigma$ 是初始参数 (Table I 中 $\mu^0=0, \sigma^0=2$)。整个 action space 是被采样的, world model 在远离 BC data 的区域几乎完全是 hallucination。

**MoDem-V2 的修改**: 当 $\text{rand}() > \alpha$ 时 (即 BC mode), 直接从 $\pi_\theta^{BC}$ 采 $N$ 个 actions 作为 trajectory seeds:
$$
\Gamma := \{\mathbf{a}_0\}^N \sim \pi_\theta^{BC}
$$
然后用 $\bar{Q}_\theta^{BC}$ 评估, 选 elites。

**直觉**: BC policy 已经见过 demonstration data, 它的 output distribution 是 well-supported 的。在 training 初期, world model + critic 不可信, 用 BC policy 做 "prior mean" 远比用 $\mathcal{N}(0, 2I)$ 这种无信息 prior 安全。

### 3.2 Agency Transfer (α schedule)

关键超参 $\alpha$ 控制 "用 BC 还是 MPC":
- 训练开始时 $\alpha = 0$ (完全 BC)
- 线性 ramp 到 $\alpha = 1.0$ (完全 MPC) 在固定步数内 (simulation incline pushing 25k steps, 其他 100k steps)

这是一个 **curriculum** 的思想: 在 world model 还没见过多少 data 时, 把 agency 交给 BC; 当 world model 在 BC-anchored 区域 enough coverage 后, 才逐渐把 agency 交给 MPC planner。

**直觉**: 这避免 MoDem 的 failure mode —— 在 episode 开头连续用 undertrained MPC 选 actions, 导致 trajectory drift 到 OOD 区域无法 recover。

### 3.3 Actor-Critic Ensembles + Epistemic Uncertainty (Line 13-16)

这是最 interesting 的部分。原始 MoDem 单个 critic, 容易 overestimation bias (参考 Fujimoto et al., TD3, ICML 2018)。MoDem-V2 用 ensemble size = 5 (Table I) 的 actor-critic pairs。

对每个 candidate trajectory $\Gamma$, 评估 5 个 critic:
$$
\phi_\Gamma^{1:M} = \phi_\Gamma + \gamma^h Q_\theta^{1:M}(\mathbf{z}_H, \mathbf{a}_H)
$$

然后线性组合 mean + std:
$$
\phi_\Gamma = w_1 \cdot \text{mean}(\phi_\Gamma^{1:M}) + w_2 \cdot \text{std}(\phi_\Gamma^{1:M})
$$

其中 $w_1 = 1.0, w_2 = -10.0$ (Table I)。

**关键点**: $w_2$ 是负的! 这是 **epistemic uncertainty penalty** —— 如果 5 个 critic 在某个 trajectory 上 disagree, 就惩罚它。disagreement 意味着这个 trajectory 落在了 OOD 区域, critic 们各自 hallucinate 出不同的值。

**避免 overestimation 的另一个 trick** (Section IV-A 第3点): 每个 actor 训练时最大化它对应的 critic, 但是 evaluation 时, **不用同一个 actor 的 critic 评估它产生的 action**。这避免了 "actor 找到 critic 的 exploit point" 的 overestimation pattern, 类似于 TD3 的 clipped double-Q but extended 到 ensemble。

### 3.4 Trajectory Weighting (Line 17)

最终选 elites 用 temperature-weighted softmax:
$$
\Omega_i = e^{\tau \phi_{\Gamma_i}}, \quad \mu = \frac{\sum_i \Omega_i \Gamma_i}{\sum_i \Omega_i}, \quad \sigma = \sqrt{\frac{\sum_i \Omega_i (\Gamma_i - \mu)^2}{\sum_i \Omega_i}}
$$

- $\tau = 0.5$ (Table I)
- 这是 standard Cross-Entropy Method (CEM) / MPPI 的 refinement step
- Population size = 510, elite fraction = 64 (Table I)

---

## 4. 实验设计

### 4.1 Task Suite (4 tasks, 见 Figure 1)

| Task | Manipulation type | Hardware | Action dim | Difficulty |
|---|---|---|---|---|
| Planar Pushing | Non-prehensile | Franka + Robotiq | 低 | Easy |
| Inclined Pushing | Non-prehensile | Franka + Robotiq | 低 | Medium (gravity reset 增加随机性) |
| Bin Picking | Prehensile | Franka + Robotiq | 中 | Hard (gripper 65% aperture) |
| In-Hand Reorientation | In-hand | Franka + D'Manus (10 DoF) | 高 (>2x) | Hardest |

### 4.2 关键 Hardware 细节

- Franka Panda arm (7 DoF)
- 3× RealSense D435 cameras (left, right, top)
- Robotiq two-fingered gripper (pushing/picking)
- D'Manus hand 10 DoF (in-hand, from ROBEL ecosystem)
- Control rate: 12.5 Hz = 80ms/timestep
- Image: 224×224, frame stack=2

### 4.3 Reward 设计 (real world 用 vision-based sparse reward)

这是这篇 paper 的一个 underrated 贡献 —— 如何在 real world 用 vision 检测 sparse success:

- **Planar Pushing**: LUV color space color thresholding, 检测 green object 是否覆盖 red goal area
- **Inclined Pushing**: 同上
- **Bin Picking**: 拍两次照 (move away → take photo, open gripper → take photo), subtract images, threshold 判断 object 是否被 grasp
- **In-Hand Reorientation**: 用 top-down depth camera 检测 bottle 是否 upright

real-world reward assignment 是 **backward in time**: episode 成功后, 从最后往前找关键时刻, 给关键时刻后的所有 timestep +1 reward。这是 episode-level success → dense reward 的 clever 转换。

### 4.4 仿真实验结果 (Figure 4)

Baselines: DAPG (state + dense reward), FERM, MoDem

关键观察:
1. **Sample efficiency**: MoDem-V2 ≈ MoDem >> DAPG (即使 DAPG 有 privileged state + dense reward)。这是 visual MBRL 强有力的 case study —— 从 pixels + sparse reward 居然比从 state + dense reward 学得快。
2. **Safety**: MoDem-V2 在所有 4 个 task 上 safety violations 远低于 MoDem。MoDem 在 online interaction 开始时 violations 立刻飙升。
3. FERM 在 easy tasks 上 work, 在 hard tasks (bin picking, in-hand) 完全失败。

### 4.5 Ablations (Figure 5)

逐个加 Centering / Schedule / Ensemble 到 MoDem 上:

- **Centering**: 单独使用 reduce violations 但 sample efficiency 一般 (是 Schedule 的 sub-component)
- **Schedule**: 最 safe 但 sample efficiency 不是最好
- **Ensemble**: sample efficiency 最好但 safety 略差
- **MoDem-V2 (三者结合)**: 拿到 Ensemble 的 efficiency + Schedule 的 safety

特别值得注意: 在 Bin Picking 任务上, 单独 Centering 或 Ensemble 反而 safety violations 更多 —— 这是 **non-monotonic** 的, 说明每个 component 单独用都有 side-effect, 只有组合才能 cover 各自的弱点。

### 4.6 Real-World 结果 (Figure 6, 8)

- MoDem 在 real world **完全 infeasible** (在前 2 个 episode 内就 fault)
- MoDem-V2 在 **<2 小时** online interaction data 内显著超过 BC policy 的 success rate
- 所有 4 个 task 都 learn 成功

---

## 5. Hyperparameters 深度解读 (Table I)

挑几个值得讨论的:

| Hyperparameter | Value | 我的解读 |
|---|---|---|
| Discount factor $\gamma$ | 0.95 | 较短 effective horizon (~20 steps), 对 manipulation task 合理 |
| Frame stack | 2 | 提供 velocity info, 比 3-4 stack 保守, 推测是为降 sample complexity |
| Data augmentation | ±10 pixel shifts | translation invariance, 视觉 generalization |
| Seed steps | 5000 (7500 for in-hand) | BC policy 收集的 initial data |
| Demo sampling ratio | 75% → 25% (100K steps) | 类似 DAPG 的 demo annealing |
| Planning horizon $H$ | 5 | short-horizon, 与 $\gamma=0.95$ 协调 |
| $\alpha$ schedule | $0 \to 1$ (25k or 100k) | agency transfer |
| $w_1, w_2$ | $1.0, -10.0$ | uncertainty penalty 比较大, std 权重是 mean 的 10x —— 强烈 prefer low-uncertainty trajectory |
| Initial $\mu^0, \sigma^0$ | $(0, 2)$ | MPC 阶段的 prior (action space 已 normalize) |
| Population size / Elite | 510 / 64 | ~12.5% elite ratio (CEM 标准是 10-25%) |
| Policy fraction | 5% | 510 中 ~25 个 trajectories 用 $\pi_\theta$ seed, 剩下从 $\mathcal{N}(\mu, \sigma^2)$ |
| Latent dim | 50 | 比 TD-MPC 的 50 一致 |
| Ensemble size | 5 | standard |
| $\lambda$ (temporal) | 0.5 | n-step like |
| $c_1, c_2, c_3$ | 0.5, 0.1, 2 | consistency loss ($c_3$) 权重最高, 显示 representation learning 是核心 |
| $\epsilon$ schedule | 0.1 → 0.05 (25k) | exploration noise decay |

---

## 6. 关于 Safety 的定义 (Section III)

这点很重要: paper 显式承认 safety constraints 是 **"diverse, obscure, and unobservable"** —— intrinsic to low-level hardware, 没 sensing 直接 measure。

- **Real world**: 任何需要 human intervention 的 hardware fault
- **Simulation**: torque limit violation (manufacturer spec) 或 contact force > 100 N

这是关键: 即使在 simulation 中, 你也只能 simulate 部分 safety constraints (torque + force), 真实 robot 还有更多 implicit 的 (thermal, joint wear, cable fatigue)。所以 paper 的 approach 是 **不需要显式 model 这些**, 而是用 conservative exploration 间接 avoid。

---

## 7. 联系与延伸 (build intuition)

### 7.1 与 JEPA / Self-Supervised Learning 的关系

MoDem 的 $\mathcal{L}_{EM}$ 与 LeCun 的 I-JEPA / BYOL 高度同源。Stop-gradient + EMA target network + predictive latent 是关键。这暗示 visual MBRL 的 future 可能跟 self-supervised representation learning 深度绑定。

参考: I-JEPA (Assran et al., 2023) https://arxiv.org/abs/2301.08243

### 7.2 与 DayDreamer 的对比

DayDreamer (Wu et al., CoRL 2022) 也是 real-world visual MBRL, 但没用 demonstrations, 任务更简单。MoDem-V2 通过 demo-bootstrapping 解锁了 contact-rich 任务, 这是相对 DayDreamer 的核心增量。

参考: https://arxiv.org/abs/2206.14176

### 7.3 与 SAVED (Thananjeyan et al., 2020) 的对比

SAVED 也用 demos + ensemble uncertainty for safe exploration, 但需要 user-specified safe state function。MoDem-V2 通过 BC-anchored exploration 隐式定义 "safe set", 更 scalable 到 high-dim/部分可观测 setting。

参考: https://arxiv.org/abs/1910.10551

### 7.4 与 RT-1 / RT-2 / RT-X 的关系

Google 的 RT 系列走的是 **large-scale offline + behavior cloning** 路线, 完全 bypass online RL 的 safety 问题。MoDem-V2 走的是 **small-data + online MBRL** 路线。两者是 complementary 的: RT 系列适合 broad coverage, MoDem-V2 适合 contact-rich 精细任务 + 快速 adapt to new task。

参考: RT-1 https://arxiv.org/abs/2212.06817 ; RT-2 https://arxiv.org/abs/2307.15818

### 7.5 Epistemic vs. Aleatoric Uncertainty

MoDem-V2 的 ensemble std 衡量的是 **epistemic uncertainty** (model 不确定, data 不够) 而非 aleatoric (inherent stochasticity)。在 real robot 上, epistemic 是可控的 (多 collect data), aleatoric 不可控。把 exploration budget 集中到 epistemic 高的区域, 是 sample-efficient 的关键。

### 7.6 与 PILCO / PETS 的 lineage

Visual MBRL 的 lineage:
- PILCO (Deisenroch & Rasmussen, 2011): Gaussian Process dynamics, analytic policy gradient
- PETS (Chua et al., NeurIPS 2018): ensemble of probabilistic NN dynamics, sampling-based planner ← **MoDem-V2 的 epistemic uncertainty 思想直接来自这里**
- TD-MPC (Hansen et al., 2022): latent space + JEPA-style representation + MPC
- MoDem (2022): + demos
- MoDem-V2 (this paper): + safety

参考: PETS https://arxiv.org/abs/1805.12114

### 7.7 一个 Critique

paper 没充分讨论的是 **environment reset 的 cost**。Real-world 实验 rely on:
- Retractor reels (planar pushing)
- Gravity (inclined pushing) 
- Hand-coded policy (bin picking, in-hand)

这些 reset 机制其实给 BC policy 提供了 strong prior, 也是 task 能在 2 小时内学成的关键之一。Autonomous reset 在 manipulation 中仍是 open problem。

另一个 limitation: **generalization across objects/goals**。Limitations section 自己也提到希望 future reuse world model。这与 RT-X / Open-X-Embodiment 的大数据路线形成 contrast。

### 7.8 Action Centering 的更深直觉

为什么 action centering 在 ensemble 单独使用时反而更不安全 (Bin Picking 任务)? 我的猜测: Centering 单独使用时, BC policy 可能 overfit 到 demo 的 narrow trajectory, 当 OOD situation 出现 (object 在 bin 边缘) 时, BC 输出的 action 本身就 unsafe。需要 ensemble + schedule 一起, 才能在 BC-anchored exploration 中通过 uncertainty 检测 OOD。

---

## 8. 总结: 这篇 paper 的真正贡献

1. **First visual MBRL with demos trained directly in real world** —— 这个 milestone 本身就有 significance。
2. **三个 simple 但 synergistic 的 components** —— policy centering + agency transfer + actor-critic ensembles。没有一个是 fundamentally new technique, 但组合 + 在 MBRL context 中的应用, 解决了真问题。
3. **Ablation 显示 non-monotonic synergy** —— 单个 component 有时让情况变差 (Bin Picking), 组合后变好, 这在 RL system paper 中少见, 很 honest。
4. **Vision-based sparse reward** —— backward time assignment + image differencing, 实用且 elegant。

如果你 (Karpathy) 关注的视角是 "如何 build agent that learns world model from pixels and acts safely in real world", 这篇 paper 提供了一个 well-engineered reference implementation, 它的核心哲学可以总结成一句: **不要 trust 你的 world model 在它没见过的区域**。这个哲学与你讲 LM 时强调 "don't trust the model outside its training distribution" 完全一致。

进一步阅读建议:
- TD-MPC: https://arxiv.org/abs/2203.04955
- MoDem: https://arxiv.org/abs/2212.05698
- I-JEPA: https://arxiv.org/abs/2301.08243
- PETS: https://arxiv.org/abs/1805.12114
- DayDreamer: https://arxiv.org/abs/2206.14176
- SAVED: https://arxiv.org/abs/1910.10551
- DAPG: https://arxiv.org/abs/1709.10087
- ROBEL: https://arxiv.org/abs/1909.11664
- Open-X-Embodiment: https://robotics-transformer-x.github.io/
