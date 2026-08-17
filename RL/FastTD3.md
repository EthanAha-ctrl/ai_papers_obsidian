---
source_pdf: FastTD3.pdf
paper_sha256: 19a40811f254bdc930d34407ca5151ce5c07bf3f62d253cb098ce4003a7c36fe
processed_at: '2026-08-04T07:46:53-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# FastTD3 人话版

## 一句话讲清楚

这篇 paper 做的事情非常朴素: **拿一个 2018 年的老算法 TD3, 加上 parallel simulation + large batch + distributional critic + 调好的 hyperparameters, 3 小时在单张 A100 上训出能部署到真 humanoid 的 policy**。作者直接声明 "我们不 claim novelty", 这是一篇 engineering report。

---

## 为什么 humanoid RL 一直这么痛苦

先讲背景。humanoid robot 控制的 RL 训练有个老大难问题: **慢**。

HumanoidBench (Sferrazza et al. 2024) 这个 benchmark 出来的时候, state-of-the-art 的算法跑 48 小时, 一堆 task 还 fail。对 practitioner 来说这简直是灾难, 因为 reward design 是迭代的——你训一次, 看结果, 改 reward, 再训一次。48 小时一轮, 一周迭不了几次。

为什么这么慢? 几个原因叠加:
- Humanoid 的 state space 高维 (几十个 joint + velocity + IMU), action space 也高维 (20+ DOF)
- Reward 稀疏, 摔倒就 episode 结束, signal 很难 propagate
- 传统 off-policy RL (SAC, TD3) sample efficient 但 wall-clock 慢, 因为 batch 小、UTD 高、训练不稳
- PPO wall-clock 快 (用 massively parallel sim), 但 sample inefficient, 不能用 demo 初始化, 不好 fine-tune

**痛点总结**: PPO 快但 sample 差, off-policy sample 好但慢。两边都有问题。

参考 HumanoidBench: https://humanoidbench.github.io

---

## TD3 本来是干什么的

要理解 FastTD3, 先得搞懂 TD3 (Fujimoto et al. 2018) 在干嘛。

TD3 是 **actor-critic** 架构, 两个网络:
- **Actor** $\mu_\phi(s)$: 输入 state $s$, 输出一个 deterministic action $a \in \mathbb{R}^{|A|}$
- **Critic** $Q_\theta(s, a)$: 输入 state-action pair, 输出一个标量 Q value

TD3 的三个核心 trick:

**Trick 1: Clipped Double Q-learning (CDQ)**
训两个 critic $Q_{\theta_1}, Q_{\theta_2}$, 算 target 的时候取 min:
$$y = r + \gamma \cdot \min\big(Q_{\bar\theta_1}(s', \tilde a'), Q_{\bar\theta_2}(s', \tilde a')\big)$$
变量含义:
- $r$ = 当前 step 的 reward
- $\gamma$ = discount factor, 通常 0.99
- $\bar\theta_1, \bar\theta_2$ = target network 参数 (Polyak averaged, 慢慢跟着 $\theta_1, \theta_2$ 动)
- $\tilde a'$ = target action, 加了 clipped Gaussian noise (这是 trick 3)

为什么取 min? 因为 Q value 容易 overestimation, 取 min 抑制 overestimation propagation。代价是可能 underestimation, 但 underestimation 不会 self-reinforce, 所以更安全。

**Trick 2: Delayed policy update**
Critic 更新频率高于 actor。TD3 里每 2 次 critic update 才 update 1 次 actor。原因: critic 还没收敛的时候, actor 沿着 noisy Q gradient 走容易崩。

**Trick 3: Target policy smoothing**
算 target action 时加 noise:
$$\tilde a' = \mu_{\bar\phi}(s') + \text{clip}(\epsilon, -c, c), \quad \epsilon \sim \mathcal{N}(0, \tilde\sigma)$$
- $\epsilon$ = Gaussian noise
- $c$ = noise clip 范围 (防止极端值)
- $\tilde\sigma$ = noise std

这相当于在 target 估值时做 local averaging, 让 Q function 更平滑。

**TD3 的 exploration 怎么搞?**
Actor 是 deterministic, exploration 靠在 action 上加 noise:
$$a = \mu_\phi(s) + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
- $\sigma$ = exploration noise scale, 手调

这是 TD3 的 historical 弱点: deterministic policy 探索能力差。

参考 TD3 原文: https://arxiv.org/abs/1802.09477

---

## FastTD3 的核心 insight: Parallel sim 救了 TD3 的 exploration

TD3 一直被诟病 exploration 差, 所以大家更爱用 SAC (stochastic policy, 自带 entropy exploration)。但 FastTD3 发现一个事情:

**当 parallel environment 数量 $N_{\text{env}}$ 极大 (128 ~ 4096) 时, TD3 的 exploration 问题自动消失**。

为什么? 想一下: 4096 个 env 同时跑, 每个 env 不同的初始 state + 不同的 random seed, 即使每个 env 用的 noise scale 一样, **整体 data distribution 极其 diverse**。4096 个 env 同时在 state space 各个角落探索, 相当于免费给 TD3 装了一个 exploration engine。

这跟 PPO 用 parallel sim 的机制本质一样, 但 TD3 的 deterministic policy gradient 有个好处: **gradient variance 低**。PPO 的 stochastic policy gradient $\nabla_\phi \log \pi(a|s) \cdot A$ 方差大, 需要大 batch + importance sampling clip 来稳定。TD3 的 deterministic policy gradient $\nabla_a Q(s,a) \cdot \nabla_\phi \mu_\phi(s)$ 直接是 chain rule, 没有 score function, 方差天然低。

所以组合起来:
- **Parallel sim 提供 exploration (diversity)**
- **Deterministic policy gradient 提供 low-variance exploitation**
- **Large batch 进一步降低 gradient noise**

这三个加起来, 就是 FastTD3 能 work 的根本原因。

参考 PQL (Li et al. 2023b) 最早指出这点: https://arxiv.org/abs/2308.01880

---

## Large batch = 32768 的真正理由

Standard TD3 用 $B=256$。FastTD3 用 $B=32768$, 差 128 倍。这看起来很夸张, 但背后有个简单算术。

考虑 data utilization:
- Buffer size $|\mathcal{D}| = N \cdot N_{\text{env}}$ (FastTD3 的设计, 后面讲)
- 假设 $N = 1000, N_{\text{env}} = 4096$, 那 $|\mathcal{D}| \approx 4 \times 10^6$
- 每 env step 写入 $N_{\text{env}} = 4096$ 个 transitions
- 每次 gradient update 采样 $B$ 个, UTD = $U$, 每 env step 做 $U$ 次更新
- 每 env step 采样总数 = $U \cdot B$

如果 $B = 256, U = 2$:
$$U \cdot B = 512 \ll N_{\text{env}} = 4096$$
这意味着每 step 写入 4096 个 transitions, 只 sample 512 个, **大部分数据在被 replay 之前就被 evict 了**。数据利用率极低。

如果 $B = 32768, U = 2$:
$$U \cdot B = 65536 > N_{\text{env}} = 4096$$
每 step 的 4096 个 transitions 平均会被 sample $\sim 16$ 次。数据被充分用。

**Intuition**: large batch 的目的不是 per-update 更稳, 是 **让 buffer 里的数据不被浪费**。在 massively parallel + 低 UTD 的设定下, small batch 等于把大量 GPU 生成的数据直接扔掉。

Paper Figure 5b 的实验验证这点: $B$ 从 1024 涨到 32768, 性能持续提升; 再涨到 131072 开始饱和。

---

## Distributional critic: 把标量 Q 变成分布

Standard TD3 的 critic 输出标量 $Q(s,a) \in \mathbb{R}$。Distributional critic (Bellemare et al. 2017, C51) 输出一个 **分布**。

具体做法: 把 return 的 support 固定在 $N$ 个 atoms 上:
$$z_i = v_{\min} + i \cdot \Delta z, \quad \Delta z = \frac{v_{\max} - v_{\min}}{N - 1}, \quad i \in \{0, 1, \dots, N-1\}$$

变量:
- $v_{\min}, v_{\max}$ = return 的下界上界, 需要手调
- $N$ = atom 数, 通常 51
- $\Delta z$ = atom 间距
- $z_i$ = 第 $i$ 个 atom 的值

Critic 网络输出 logits, softmax 成概率:
$$p_\theta(z_i | s, a) \geq 0, \quad \sum_{i=0}^{N-1} p_\theta(z_i | s, a) = 1$$

Expected Q value:
$$Q_\theta(s,a) = \sum_{i=0}^{N-1} z_i \cdot p_\theta(z_i | s, a)$$

Target distribution 通过 **project Bellman update** 计算:
$$\hat T z_j = r + \gamma z_j$$
然后 clip 到 $[v_{\min}, v_{\max}]$, 按 $r + \gamma z_j$ 落在哪个相邻 atom 之间线性插值分配 probability mass, 得到 target distribution $\hat p_i$。

Loss 是 cross-entropy:
$$\mathcal{L}(\theta) = -\sum_{i=0}^{N-1} \hat p_i \log p_\theta(z_i | s, a)$$

**为什么这对 humanoid 任务有用?**

Intuition: humanoid 控制里, 同一个 (state, action) 后续可能走稳, 也可能摔倒。return 的分布明显双峰。标量 Q 取期望, 会给一个"中间值", 这个中间值谁都不是, policy gradient 方向模糊。Distributional critic 保留双峰结构, gradient 信号更清晰。

**代价**: $v_{\min}, v_{\max}$ 是新 hyperparameters。Paper 里说不太难调, 但 SimbaV2 (Lee et al. 2025) 用 reward normalization 干掉了这两个参数, FastTD3 未来可以借鉴。

参考 Distributional RL 原文: https://arxiv.org/abs/1707.06887

---

## UTD (Update-to-Data Ratio) 的选择

UTD = 每 env step 做几次 gradient update。

传统 off-policy RL 的趋势是追求高 UTD: BBF (Schwarzer et al. 2023) UTD 到 32, BRO (Nauman et al. 2024b) 更高。理由是 sample efficient。代价是 deadly triad (bootstrapping + function approximation + off-policy) 导致训练不稳, 所以需要 layer norm, residual block, self-imitation 这些拐杖。

FastTD3 反着来: **UTD 极低, 通常 2, 4, 8**。什么拐杖都不用, plain 3-layer MLP 直接 work。

为什么? 重新看 deadly triad:
$$\text{effective off-policyness} \propto U \cdot \text{staleness of samples}$$

低 UTD 意味着每次 gradient update 用的 transitions 都很 fresh (刚进 buffer 不久), function approximation 误差来不及累积, deadly triad 不发作。

**Intuition**: 这跟 supervised learning 里 "small learning rate + many epochs" vs "large batch + few steps" 的争论同构。FastTD3 选了后者, 大 batch + 低 UTD, 每 step 用充分数据, 但不反复挤压同一个 transition。

Paper Figure 6c 实验: UTD 从 1 涨到 16, sample efficiency 一直提升, 但 wall-clock time 线性涨。所以 FastTD3 默认 UTD=2 或 4, 在 sample efficiency 和 wall-clock 之间取 trade-off。

参考 BBF: https://arxiv.org/abs/2305.19427
参考 BRO: https://arxiv.org/abs/2410.07263

---

## Replay buffer 设计: $N \times N_{\text{env}}$

这个细节非常工程, 但影响很大。

传统做法: 固定 buffer size $|\mathcal{D}| = 10^6$。问题:

假设 episode length $T = 1000$:
- $N_{\text{env}} = 1000$: 每 env 能存 $10^3$ 步 = 1 个完整 episode ✓
- $N_{\text{env}} = 2000$: 每 env 只能存 $500$ 步 = 半个 episode ✗

Trajectory 被截断, TD-learning 的 bootstrap consistency 被破坏。

FastTD3 改成:
$$|\mathcal{D}| = N \cdot N_{\text{env}}$$

- $N$ = per-env buffer length, 固定 (比如 1000)
- $N_{\text{env}}$ = parallel env 数, 可调

这样调 $N_{\text{env}}$ 不影响每 env 的 trajectory 完整性。Paper 实验显示 $N$ 越大越好 (代价是 GPU memory), 因为 off-policy 数据 reuse 需要足够大的 buffer。

因为 non-vision domain, 整个 buffer 留 GPU, 避免 CPU↔GPU 传输 overhead。

---

## Architecture: 越简单越好

| 网络 | Layer 1 | Layer 2 | Layer 3 | 参数量 |
|------|---------|---------|---------|--------|
| Critic $Q_{\theta_1}, Q_{\theta_2}$ | 1024 | 512 | 256 | ~0.55M |
| Actor $\mu_\phi$ | 512 | 256 | 128 | ~0.2M |

Paper 试过:
- $D \in \{256, 512, 1024\}$ (第一层 hidden dim): $D=512$ 通常最好, $D=1024$ 略好但慢, $D=256$ 显著差
- Residual path + layer norm (BRO/Simba 风格): **减慢训练且无明显收益**

为什么不需要 layer norm / residual? 因为 low UTD + large batch 让 deadly triad 不发作, 不需要 architectural stabilizer。

**Intuition**: layer norm 这些 trick 是为高 UTD 设计的。高 UTD 把同一个 transition 反复 squeeze, gradient 方向容易 collapse, layer norm 帮你 normalize。FastTD3 低 UTD, 每个 transition 只 squeeze 几次, 没机会 collapse, layer norm 反而增加 overhead。

---

## Exploration noise: $\sigma_{\max} = 0.4$ 很大

PQL 提出 mixed noise: 每个 env 从 $[\sigma_{\min}, \sigma_{\max}]$ 均匀采样 noise scale:
$$\sigma_j \sim \mathcal{U}[\sigma_{\min}, \sigma_{\max}], \quad j \in \{1, \dots, N_{\text{env}}\}$$
$$a_j = \mu_\phi(s_j) + \sigma_j \cdot \epsilon_j, \quad \epsilon_j \sim \mathcal{N}(0, I)$$

FastTD3 实验发现 mixed noise 没显著增益, 但保留因为它代码改动少。关键发现: $\sigma_{\max} = 0.4$ 比 standard TD3 默认 0.2 大很多, 对 humanoid 控制很关键。humanoid action space 高维, 小 noise 探索不到 joint 的 aggressive 配置。

---

## 工程加速: AMP + torch.compile

| 技术 | 加速 |
|------|------|
| AMP (bfloat16 mixed precision) | ~40% |
| `torch.compile` (via LeanRL) | ~35% |
| 两者结合 | ~70% |

为什么选 PyTorch 不选 JAX? 作者明说: simplicity + flexibility。JAX 快但 debug 难, PyTorch 慢但生态好。FastTD3 用 AMP + torch.compile 把 PyTorch 拉到接近 JAX 速度, 保留 PyTorch 易用性。

参考 LeanRL (基于 CleanRL Huang et al. 2022): https://github.com/vwxyzjn/cleanrl

---

## 最有意思的发现: Reward function 是 algorithm-specific

Paper Figure 7 四宫格, 这是整篇 paper 最 underrated 的 part。

实验设置: MuJoCo Playground 上训 G1 humanoid 走路。

| 配置 | 结果 |
|------|------|
| FastTD3 + PPO-tuned reward | G1 abrupt arm movements, undeployable |
| PPO + PPO-tuned reward | Natural walking gait |
| FastTD3 + FastTD3-tuned reward (强 penalty) | Smooth gait |
| PPO + FastTD3-tuned reward | Slow walking, undeployable |

同一个 reward function, PPO 训出来自然步态, FastTD3 训出来甩手臂。改 reward 加强 penalty, FastTD3 步态变好, PPO 反而变差。

**为什么?**

PPO 是 on-policy, gradient 信号来自当前 policy 的 trajectory distribution。PPO 倾向于 find "easy" local optimum: 低 variance, 高 reward, 不 aggressive explore。所以 PPO-tuned reward 会设计得比较 "loose", 允许一些 suboptimal 但 stable 的行为。

TD3 是 off-policy + large replay, aggressive exploit 任何能增加 reward 的小 trick。甩手臂能短暂提高 balance reward, TD3 就学会甩。PPO 因为 on-policy + entropy bonus, 不太会学这种 trick。

要抑制 TD3 的 trick, reward 需要更强 penalty。但强 penalty 让 PPO 训得慢, 因为 PPO 不敢 aggressive explore, 被 penalty 吓住, 走保守步态。

**Implication**: reward design 是 algorithm-specific。episode return 不是 policy usefulness 的好 proxy。这给 iterative inverse RL (Eureka Ma et al. 2023) 提供 motivation: 应该 joint 优化 algorithm + reward。

参考 Eureka: https://arxiv.org/abs/2310.12931

---

## Sim-to-real: Booster T1 部署

Paper 声称这是 **off-policy RL policy 首次部署到 full-size humanoid**。

细节:
- Robot: Booster T1 (full-size humanoid)
- Control: 12-DOF (固定 arms, waist, heading)
- 训练 suite: MuJoCo Playground (Zakka et al. 2025), 作者 fork 出 12-DOF 版本
- Domain randomization + sim-to-real transfer

之前 humanoid sim-to-real (ANYmal Hwangbo et al. 2019, Cassie, Berkeley lower-body humanoids) 多用 PPO。FastTD3 是第一个 off-policy 跑通的。

**为什么 off-policy sim-to-real 之前没人做?**

- PPO 在 parallel sim 里 wall-clock 快, 训练方便
- Off-policy 之前 wall-clock 慢, iteration cycle 长, sim-to-real 调参困难
- Off-policy 训练不稳, sim-to-real 需要多次 iteration, 不稳的算法难迭代

FastTD3 把 off-policy 训练压到 3 小时, iteration cycle 跟 PPO 持平, off-policy 的优势 (sample efficient, 能用 demo 初始化, 能 fine-tune) 终于可用。

参考 MuJoCo Playground: https://playground.mujoco.org
参考 Booster T1: https://www.boosterrobotics.com

---

## FastSAC: 把 recipe 套到 SAC 上

Paper 做了 ablation: 把 FastTD3 recipe (parallel sim + large batch + distributional critic + low UTD) 套到 SAC (Haarnoja et al. 2018) 上, 叫 FastSAC。

结果:
- 比 vanilla SAC 快很多
- 比 FastTD3 仍慢
- 训练不稳

**为什么 SAC 在 humanoid 高维 action space 上不稳?**

SAC 最大化 entropy-augmented objective:
$$J(\phi) = \mathbb{E}\big[\sum_t \gamma^t (r_t + \alpha H(\pi(\cdot|s_t)))\big]$$

- $\alpha$ = temperature parameter, 自动调节
- $H(\pi) = -\int \pi(a|s) \log \pi(a|s) da$ = policy entropy

在高维 action space (humanoid 23-DOF), $H(\pi)$ 估计方差大, $\alpha$ 自动调节 noise 大。TD3 deterministic policy 没这个问题。

**Implication**: SAC 的未来在 architecture 改进 (SimbaV2 的 hyperspherical normalization) 而非 recipe 调整。TD3 的简单结构反而更适合 humanoid。

参考 SAC: https://arxiv.org/abs/1812.05905
参考 SimbaV2: ICML 2025 proceedings

---

## Clipped Double Q-learning 的争议

Nauman et al. 2024a ("Bitter lesson") 报告: 用 **average** 而非 **min** $\min(Q_{\bar\theta_1}, Q_{\bar\theta_2})$ 在配合 layer normalization 时更好。

FastTD3 反过来发现: **没有 layer norm 时, min 仍然更好**。

形式上:
- Min 版 (TD3 原版): $y = r + \gamma \min(Q_{\bar\theta_1}, Q_{\bar\theta_2})$
- Average 版: $y = r + \gamma \cdot \frac{1}{2}(Q_{\bar\theta_1} + Q_{\bar\theta_2})$

Min 的 bias 是 negative (underestimation), average 的 bias 是 positive (overestimation)。Deadly triad 里, overestimation propagation 比 underestimation 更致命, 因为 overestimation 会 self-reinforce (越 overestimate 越往那走), underestimation 不会 self-reinforce。

Layer norm 把 Q value scale 规范化, average 不会爆炸, 所以 average + layer norm work。没有 layer norm, average 容易 overestimate, min 更安全。

FastTD3 没 layer norm, 所以 CDQ min 版仍然重要。Paper 结论: CDQ 是 per-task 需要调的 hyperparameter。

参考 "Bitter lesson": https://arxiv.org/abs/2403.00514

---

## 跟其他 algorithm 的关系

Paper Section 4 明确说: FastTD3 的 insight 都来自 prior work, 没有 novelty。

| Prior work | Contribution | FastTD3 用到 |
|------------|-------------|-------------|
| PQL (Li et al. 2023b) | Parallel sim + large batch + distributional critic for off-policy RL | 全用到, 但简化实现 (无 async) |
| PQN (Gallici et al. 2024) | 同样 observation for discrete control | 概念验证 |
| Raffin (2025) | SAC on massive parallel sim | recipe 通用性验证 |
| Shukla (2025) | Speeding up SAC with parallel sim | recipe 通用性验证 |
| TD3 (Fujimoto et al. 2018) | Base algorithm | 全部 |
| Distributional RL (Bellemare et al. 2017) | Distributional critic | 用到 |
| SimbaV2 (Lee et al. 2025) | Hyperspherical normalization | 没用, 但提到可借鉴去 $v_{\min}, v_{\max}$ |
| BRO (Nauman et al. 2024b) | Residual + layer norm + large UTD | 没用, architectural stabilizer 不需要 |
| BBF (Schwarzer et al. 2023) | High UTD + self-imitation | 没用, 低 UTD 路线 |
| Eureka (Ma et al. 2023) | LLM 生成 reward | 没用, 但 Section 4 提到 future direction |

**FastTD3 的 contribution**: 把这些 insight 打包成 simple PyTorch codebase, 提供详细 ablation, 验证在 humanoid 三个 suite 上的效果, 做到 sim-to-real。

参考 PQL: https://arxiv.org/abs/2308.01880
参考 PQN: https://arxiv.org/abs/2407.04811
参考 Raffin blog: https://araffin.github.io/post/sac-massive-sim/
参考 Shukla blog: https://arthshukla.substack.com/p/speeding-up-sac-with-massively-parallel

---

## 给 Karpathy 的 core intuition

**1. Deterministic policy gradient 在 parallel sim 下找到 natural home**

TD3 一直被嫌弃 exploration 差。Parallel sim 把 exploration 问题外包给环境多样性, TD3 的 deterministic policy gradient 优势 (low variance) 才真正发挥。这暗示: **算法的弱点 + 环境的补偿 = 新的设计空间**。

类似 supervised learning 里, 大 batch training 让 SGD 的 weakness (slow convergence per step) 被 GPU parallelism 补偿, 反而比 sophisticated optimizer 快。

**2. 低 UTD + large batch 是与高 UTD + small batch 正交的设计选择**

之前 off-policy RL 社区追高 UTD (BBF, BRO, Simba), 因为 sample efficient。FastTD3 反向证明: 低 UTD + 大 batch + 充分利用每个 transition, 也能 sample efficient, 同时避免 deadly triad。

这跟 supervised learning 里 "big batch + few epochs" vs "small batch + many epochs" 的争论同构。Big batch + few epochs 在 GPU 上更高效, gradient noise 低, 不需要 sophisticated learning rate schedule。FastTD3 在 RL 里验证同样的 insight。

**3. Reward function 是 algorithm-specific**

这跟 supervised learning 里 loss function 不是 model-agnostic 类似。不同 model 结构 (CNN vs Transformer) 对同一 loss 响应不同。不同 RL algorithm 对同一 reward 响应不同。这给 iterative inverse RL (LLM-as-reward-designer) 提供 motivation: 应该 joint 优化 algorithm + reward。

**4. Engineering > novelty**

Paper 公开声明 "do not aim to claim novelty", 仍然发表了。Community 需要这种 distillation 工作。这跟 Karpathy "software 2.0", "micrograd", "nanoGPT" 风格一致: **把复杂的东西简化到核心, 验证核心 work, 提供易用 codebase**。

---

## 未来方向

Paper Section 4 提到的几个方向:

1. **FastTD3 + Eureka (LLM reward generator)**: FastTD3 短训练 cycle (3 小时) 让 LLM 生成 reward + 训练 + 评估的闭环可以 overnight 跑多轮。这是 humanoid reward design automation 的 next step。

2. **FastTD3 + demo-driven RL (BiGym Chernyadev et al. 2024, RLAS Seo & Abbeel 2024)**: Off-policy 天然支持 demo pretraining, 这是相对 PPO 的优势。

3. **FastTD3 + real-world fine-tuning**: Off-policy sample efficient, 适合 real-world interaction fine-tuning。Paper 声称这是 future direction。

4. **FastTD3 + SimbaV2 architecture**: 把 SimbaV2 的 hyperspherical normalization 加进来, 去掉 $v_{\min}, v_{\max}$, 可能进一步提升。

5. **FastTD3 + MR.Q (Fujimoto et al. 2025)**: MR.Q 是 general-purpose model-free RL, 跟 FastTD3 互补。

6. **FastTD3 + TDMPC2 / TDMPBC**: World model + self-imitation, 可能补 FastTD3 在长 horizon task 上的不足。

参考 BiGym: https://arxiv.org/abs/2406.01500
参考 RLAS: https://arxiv.org/abs/2411.12155
参考 MR.Q: https://arxiv.org/abs/2502.07876 (推测)
参考 TDMPC2: https://arxiv.org/abs/2310.16828

---

## 我的判断

如果让我赌一把, **FastTD3 + Eureka-style LLM reward designer + 12 小时人类睡眠周期** 可以做到 humanoid reward design 的 overnight 自动迭代。

具体流程:
1. 人类睡前给 task description (比如 "让 G1 学会跑步")
2. LLM (GPT-4 / Claude) 生成 candidate reward function
3. FastTD3 训练 3 小时
4. LLM 评估 policy 视频, 改 reward
5. 循环, 人类睡觉 8 小时 = 跑 ~2-3 轮
6. 人类醒来, 拿到 refined policy

这就是 FastTD3 的真正长期价值——把 iteration cycle 从 48 小时压到 3 小时, 让 humanoid RL research 从 "周级迭代" 变成 "夜级迭代"。

这跟 Karpathy 之前讲的 "neural network training as overnight job" 思路完全契合。FastTD3 让 humanoid RL 首次进入这个 regime。

---

## 总结: 一句话 intuition

FastTD3 告诉我们: **off-policy RL 之前慢, 是因为没用对 hardware**。Massively parallel sim 把 exploration 外包给环境, large batch 把 data utilization 最大化, low UTD 把 deadly triad 压住, distributional critic 把 return 多模态性保留。这四个加起来, TD3 这个 2018 年的老算法在 2025 年的 GPU 上, 3 小时训出能部署到真 humanoid 的 policy。

**Engineering + right hardware + old algorithm = SOTA**。这跟 Karpathy 一直强调的 "scale matters" 完全一致。

---

# FastTD3 深度讲解

## 1. Paper 总览与定位

FastTD3 来自 UC Berkeley 的 Pieter Abbeel 组 (Younggyo Seo, Carmelo Sferrazza, Haoran Geng 等)，核心 contribution 极度务实: **把 TD3 (Fujimoto et al. 2018) 配上一组精心调过的 hyperparameters，在 single A100 GPU 上 3 小时内解决 HumanoidBench 任务**，并且在 Booster T1 上做到了 sim-to-real——据作者所知这是 **off-policy RL policy 第一次部署到 full-size humanoid** 的有记录实例。

Paper 的灵魂在于一段话 (Section 4): "**The aim of this work is to distill those insights into a simple algorithm**". 也就是说，它本质上是一篇 distillation + engineering report，把 PQL (Li et al. 2023b)、PQN (Gallici et al. 2024)、Raffin (2025)、Shukla (2025) 的洞察打包成一个简单可用的 PyTorch codebase。

项目主页: https://younggyo.me/fast_td3
arXiv (推测): 在作者主页可找到

---

## 2. 核心配方公式化

可以把 FastTD3 写成一个组合公式:

$$
\text{FastTD3} = \text{TD3} \;+\; \underbrace{\text{ParallelSim}(N_{\text{env}})}_{\text{多样性来源}} \;+\; \underbrace{\text{LargeBatch}(B=32768)}_{\text{稳定 critic}} \;+\; \underbrace{\text{Distributional Critic}(v_{\min}, v_{\max})}_{\text{capture return distribution}} \;+\; \underbrace{\text{UTD}\in\{2,4,8\}}_{\text{低 UTD}} 
$$

这里每个变量含义:
- $N_{\text{env}}$ = parallel environment 数量 (通常 128 ~ 4096)
- $B$ = batch size，每次 gradient update 采样到 replay buffer 的 transitions 数
- $v_{\min}, v_{\max}$ = distributional critic 的 support 下界和上界
- $\text{UTD}$ = update-to-data ratio, 即每 env step 做几次 gradient update

---

## 3. TD3 基础回顾 (因为后面所有修改都建立在这之上)

TD3 的 critic 更新 (clipped double Q-learning):

$$
\mathcal{L}(\theta_1, \theta_2) = \mathbb{E}_{(s,a,r,s')\sim\mathcal{D}}\Big[\big(Q_{\theta_1}(s,a) - y\big)^2 + \big(Q_{\theta_2}(s,a) - y\big)^2\Big]
$$

其中 target:
$$
y = r + \gamma \cdot \min\big(Q_{\bar\theta_1}(s', \tilde a'), Q_{\bar\theta_2}(s', \tilde a')\big)
$$
$$
\tilde a' = \mu_{\bar\phi}(s') + \text{clip}(\epsilon, -c, c), \quad \epsilon \sim \mathcal{N}(0, \tilde\sigma)
$$

变量含义:
- $\theta_1, \theta_2$ = twin Q networks 的参数
- $\bar\theta_1, \bar\theta_2$ = target network 参数, 通过 Polyak averaging $\bar\theta \leftarrow \tau \theta + (1-\tau)\bar\theta$ 更新
- $\mu_\phi$ = deterministic actor
- $\tilde a'$ = target action 加上 clipped Gaussian noise (target policy smoothing)
- $c$ = noise clip 范围
- $\gamma$ = discount factor

Actor 更新 (deterministic policy gradient, Silver et al. 2014):
$$
\nabla_\phi J = \mathbb{E}_{s\sim\mathcal{D}}\Big[\nabla_a Q_{\theta_1}(s,a)\big|_{a=\mu_\phi(s)} \cdot \nabla_\phi \mu_\phi(s)\Big]
$$

Actor delayed update: 每 $d$ 次 critic update 才更新一次 actor (TD3 中 $d=2$).

---

## 4. 把 Distributional RL 加进来

Distributional critic (Bellemare et al. 2017, C51-style) 把 return $Z(s,a)$ 建模成一个分布而非标量。Support 是固定的 $N$ 个 atoms:

$$
z_i = v_{\min} + i \cdot \Delta z, \quad \Delta z = \frac{v_{\max} - v_{\min}}{N - 1}, \quad i \in \{0, 1, \dots, N-1\}
$$

Critic 输出 logits $p_\theta(z_i | s, a)$ via softmax, 满足 $\sum_i p_\theta(z_i|s,a) = 1$. Expected Q value:
$$
Q_\theta(s,a) = \mathbb{E}[Z_\theta(s,a)] = \sum_{i=0}^{N-1} z_i \cdot p_\theta(z_i | s, a)
$$

Target distribution 通过 **projection** 计算 (因为 $\gamma z_j + r$ 不一定落在 support 上):
$$
\hat T z_j = r + \gamma z_j, \quad \text{clip to } [v_{\min}, v_{\max}]
$$
然后 $p_{\bar\theta}(z_j | s', \tilde a')$ 的概率 mass 按 $\hat T z_j$ 落在哪个相邻 atom 之间, 用线性插值分配到 $z_i$ 和 $z_{i+1}$ 上, 得到 target distribution $\hat p_i$.

Loss 是 cross-entropy:
$$
\mathcal{L}(\theta) = -\sum_{i=0}^{N-1} \hat p_i \log p_\theta(z_i | s, a)
$$

**Intuition**: distributional critic 让 network 学到 return 的多模态分布, 这在 humanoid 任务里很关键, 因为同一 (s,a) 后续可能进入稳定的 step, 也可能摔倒, return 的分布明显双峰。标量 Q 取平均会模糊这个信号。

**代价**: $v_{\min}, v_{\max}$ 是新增 hyperparameters. SimbaV2 (Lee et al. 2025) 用 reward normalization 来消除这两个参数, FastTD3 paper 里也提到可以借鉴。

---

## 5. Parallel Environments 为什么对 TD3 特别有效

这是 FastTD3 最有 intuition 的一节. TD3 是 **deterministic policy gradient** method, actor 输出 $\mu_\phi(s) \in \mathbb{R}^{|A|}$ 是 deterministic 的, exploration 完全靠在 action 上加 Gaussian noise:

$$
a = \mu_\phi(s) + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

TD3 的 historical weakness 是 exploration 不足. 但 **当 $N_{\text{env}}$ 很大 (e.g. 4096)** 时, 即便每个 env 用相同的 noise scale, 不同的 random seed + 不同的初始 state 会让 data distribution 极其 diverse. 这相当于 "免费" 给了 TD3 一个 diversity 来源。

**Key insight**: deterministic policy gradient 算法的优势在于 efficient exploitation of value function (因为 gradient 信号比 stochastic policy gradient 方差低); parallel sim 给它补齐 exploration 这一腿。

PQL (Li et al. 2023b) 早就观察到这一点, FastTD3 重新验证并简化了实现。

---

## 6. Large-Batch Training 的真正理由

直觉上, batch $B=32768$ 看起来"浪费", 因为 standard TD3 通常用 $B=256$. 但 FastTD3 的论点是:

**当 $N_{\text{env}}$ 很大 + UTD 很低时**, 每个 env step 产生 $N_{\text{env}}$ 个 transition, replay buffer 增长极快. 如果 $B$ 太小, 大部分 transition 在被 sample 之前就被 evicted, **数据浪费率极高**. 

形式上, 假设 buffer size = $|\mathcal{D}|$, 每 step 写入 $N_{\text{env}}$ 个 transitions, 每次更新采样 $B$ 个, UTD = $U$, 那么:
- 每 env step 采样总数 = $U \cdot B$
- 期望一个 transition 被 sample 到的次数 ≈ $U \cdot B / |\mathcal{D}|$ (在 uniform replay 假设下)

如果 $U \cdot B \ll |\mathcal{D}|$, 大量数据从未被看过. 把 $B$ 推到 32768 让 $U \cdot B$ 与 $|\mathcal{D}}$ 同一量级, **保证 buffer 中大部分数据都被 used**.

这是 FastTD3 选 large batch + 低 UTD 的根本理由 (而 typical off-policy RL 选 small batch + 高 UTD).

---

## 7. UTD 与 deadly triad

经典 RL 理论 (Sutton & Barto 2018): **deadly triad** = bootstrapping + function approximation + off-policy $\Rightarrow$ 训练不稳. 高 UTD 放大这个问题, 所以之前的工作 (D'Oro et al. 2023 BBF, Schwarzer et al. 2023, BRO Nauman et al. 2024b, Simba Lee et al. 2024) 都需要 architectural stabilizers (residual connections, layer norm, self-imitation, model-based augmentation).

FastTD3 的观察: 在 $U \in \{2, 4, 8\}$ 这个区间, 什么都不需要. 它的解读是:

$$
\text{effective off-policyness} \propto U \cdot \text{staleness of samples}
$$

低 UTD 让每个 gradient step 用到的 transitions 都很 "fresh", function approximation 误差不易累积, layer norm / residual block 这些"拐杖"就不必要。这解释了为何 FastTD3 用 plain 3-layer MLP (1024, 512, 256 critic; 512, 256, 128 actor) 就够。

**Intuition 给 Karpathy**: 这跟神经网络训练中 "small learning rate + many steps" vs "big batch + few steps" 的 trade-off 同构. FastTD3 选了后者并验证它在 RL 里也 work.

---

## 8. Clipped Double Q-learning 的争议

Nauman et al. 2024a ("Bitter lesson") 报告: **用 average 而非 min** $\min(Q_{\bar\theta_1}, Q_{\bar\theta_2})$ 在配合 layer normalization 时更好。

FastTD3 反过来发现: **没有 layer norm 时, min 仍然更好**. 这暗示 $\min$ 操作本身是一种 implicit regularizer, 在缺乏 normalization 的情况下抑制 overestimation。所以 paper 结论是 CDQ 仍然是一个 per-task 需要调的 hyperparameter, 不能简单替换为 average。

形式上:
- $\min$ 版 (TD3 原版): $y = r + \gamma \min(Q_{\bar\theta_1}, Q_{\bar\theta_2})$
- $\text{average}$ 版: $y = r + \gamma \cdot \frac{1}{2}(Q_{\bar\theta_1} + Q_{\bar\theta_2})$

Min 的 bias 是 negative (underestimation), average 的 bias 是 positive (overestimation)。在 deadly triad 中, overestimation propagation 比 underestimation 更致命, 因此 min 更安全. Layer norm 的作用是把 Q value 的 scale 规范化, 让 average 不会爆炸, 但代价是引入了训练 instability (这跟 SimbaV2 的 hyperspherical normalization 思路相关)。

---

## 9. Replay Buffer 设计: $N \times \text{num\_envs}$

这个细节容易被忽略, 但极关键. 传统做法是固定 buffer size $|\mathcal{D}| = 10^6$. 但考虑:
- $|\mathcal{D}| = 10^6$, episode length $T = 1000$
- $N_{\text{env}} = 1000$: 每 env 可以存 $10^3$ 步 = 1 个完整 episode
- $N_{\text{env}} = 2000$: 每 env 只能存 $500$ 步 = 半个 episode → **trajectory 被截断**

截断 trajectory 会破坏 TD-learning 中的 bootstrap 一致性. FastTD3 改用:
$$
|\mathcal{D}| = N \cdot N_{\text{env}}, \quad N \text{ 是 per-env buffer length}
$$

这样 $N$ 与 $N_{\text{env}}$ 解耦, 调 $N_{\text{env}}$ 不会影响每个 env 的 trajectory 完整性. 实验显示 $N$ 越大越好 (代价是 GPU memory).

因为 non-vision domain, 整个 buffer 留在 GPU, 避免 CPU↔GPU 传输.

---

## 10. 工程加速

| 技术 | 加速 |
|------|------|
| AMP (bfloat16 mixed precision) | 40% |
| `torch.compile` (via LeanRL) | 35% |
| 两者结合 | ~70% |

PyTorch 而非 JAX 的选择是为了 simplicity/flexibility. 70% 加速让 single A100 跑完 HumanoidBench 任务成为可能.

参考 LeanRL: https://github.com/egoacclean/LeanRL (推测, 基于 Huang et al. 2022 CleanRL 的延伸)

---

## 11. Architecture 表

| 网络 | Layer 1 | Layer 2 | Layer 3 |
|------|---------|---------|---------|
| Critic $Q_{\theta_1}, Q_{\theta_2}$ | 1024 | 512 | 256 |
| Actor $\mu_\phi$ | 512 | 256 | 128 |

实验 (Figure 6a) 试了 $D \in \{256, 512, 1024\}$ (即三层中第一层 hidden dim), 网络参数量约 0.18M / 0.55M / 1.83M. 结果: $D=512$ 通常最好, $D=1024$ 在某些任务略好但训练慢, $D=256$ 显著更差.

Residual path + layer norm (BRO/Simba 风格) **减慢训练且无明显收益** → FastTD3 拒绝。

---

## 12. 探索 noise schedule

PQL 提出的 **mixed noise**: 每个 env 从 $[\sigma_{\min}, \sigma_{\max}]$ 中均匀采样一个 noise scale, 这样不同 env 探索强度不同:
$$
\sigma_j \sim \mathcal{U}[\sigma_{\min}, \sigma_{\max}], \quad j \in \{1, \dots, N_{\text{env}}\}
$$
$$
a_j = \mu_\phi(s_j) + \sigma_j \cdot \epsilon_j, \quad \epsilon_j \sim \mathcal{N}(0, I)
$$

FastTD3 实验发现 mixed noise 没有显著增益, 但仍保留因为代码改动少. 关键发现: **$\sigma_{\max} = 0.4$ 很重要**, 比 standard TD3 默认 0.2 大得多. 这与 humanoid control 中需要大幅 explore 关节配置相关.

---

## 13. 最有意思的发现: 不同算法需要不同 reward function

Figure 7 四宫格是这个 paper 最 underrated 的部分.

| 配置 | 结果 |
|------|------|
| FastTD3 + PPO-tuned reward | G1 abrupt arm movements, undeployable |
| PPO + PPO-tuned reward | Natural walking gait |
| FastTD3 + FastTD3-tuned reward (强 penalty) | Smooth gait |
| PPO + FastTD3-tuned reward | Slow walking, undeployable |

**Intuition**: PPO 因为 on-policy, 倾向于 find "easy" local optimum (低 variance, 高 reward), 所以 reward function 要 "push" 它去 explore more aggressive behaviors. TD3 因为 off-policy + large replay, 倾向于 aggressively exploit 任何能增加 reward 的小 trick, 所以需要更强的 penalty 来抑制这些 trick (比如甩手臂保持平衡).

这暗示了 **reward design 是 algorithm-specific**, episode return 不是 policy usefulness 的好 proxy. 这给 iterative inverse RL (e.g. Eureka Ma et al. 2023, LLM reward generator) 提供了 motivation: 应该 joint 优化 algorithm + reward, 而非独立.

---

## 14. FastSAC 实验 (Figure 8)

把 FastTD3 recipe 套到 SAC 上得到 FastSAC. 结果:
- 比 vanilla SAC 快很多
- 比 FastTD3 仍慢
- 训练不稳

假设原因: SAC 在高维 action space (humanoid 23-DOF) 上最大化 entropy $H(\pi(\cdot|s))$ 很困难. SAC 的 $\alpha$-temperature 自动调节在高维下噪声大. TD3 的 deterministic policy 反而更稳。

SimbaV2 在主实验中比 vanilla SAC 快很多, 暗示 **SAC 的未来在 architecture 改进 (hyperspherical normalization) 而非 recipe 调整**。

---

## 15. Sim-to-real: Booster T1

部署细节:
- Robot: Booster T1 (full-size humanoid)
- Control: 12-DOF (固定 arms, waist, heading; 原 MuJoCo Playground 支持 23-DOF 全关节, 作者 fork 出 12-DOF 版本)
- 训练 suite: MuJoCo Playground (Zakka et al. 2025)
- Domain randomization + sim-to-real: 见项目视频

这是 paper 唯一的 "novelty" 声称: **off-policy RL 首次部署到 full-size humanoid**. 之前的 humanoid sim-to-real (ANYmal Hwangbo et al. 2019, Cassie, Berkeley lower-body humanoids) 多用 PPO。

参考 MuJoCo Playground: https://playground.mujoco.org
Booster T1: https://www.boosterrobotics.com (推测)

---

## 16. 实验数据表

主对比基线:
| Algorithm | 类型 | 速度 | Sample efficiency |
|-----------|------|------|-------------------|
| PPO | on-policy | 快 | 差 |
| SAC | off-policy | 慢 | 中 |
| DreamerV3 | world model | 慢 | 中 |
| TDMPC2 | world model | 中 | 好 |
| SimbaV2 | off-policy + arch | 中 | 好 |
| FastTD3 | off-policy + recipe | 快 | 好 |

HumanoidBench 上, prior SOTA (DreamerV3, TDMPC2, SAC, SimbaV2) 48 小时仍 fail many tasks. FastTD3 < 3 小时解决 multiple tasks.

---

## 17. 给 Karpathy 的核心 intuition

1. **TD3 的 deterministic policy gradient 在大规模 parallel sim 下找到了 natural home**. Deterministic 的 weakness (探索差) 被 parallel envs 的 diversity 弥补, deterministic 的 strength (low-variance gradient) 在 large batch 下被放大。

2. **低 UTD + large batch 是与高 UTD + small batch 正交的设计选择**. 之前 off-policy RL 社区一直追高 UTD (BBF, BRO, Simba), FastTD3 反向证明低 UTD + 把每 step 的数据用充分, 也能达到 sample efficiency, 同时避免 deadly triad. 这跟 supervised learning 中 "big batch + few epochs" vs "small batch + many steps" 的争论同构。

3. **Reward function 不是 algorithm-agnostic**. 这对 future reward design automation (LLM-as-reward-designer) 有直接 implication。

4. **Engineering > novelty**. Paper 公开声明 "do not aim to claim novelty", 仍然发表了, 因为 community 需要这种 distillation 工作. 这跟 Karpathy "software 2.0", "micrograd" 风格一致。

参考链接:
- PQL: https://arxiv.org/abs/2308.01880
- PQN: https://arxiv.org/abs/2407.04811
- TD3 原文: https://arxiv.org/abs/1802.09477
- Distributional RL: https://arxiv.org/abs/1707.06887
- SimbaV2: 见 ICML 2025 proceedings
- HumanoidBench: https://github.com/MarkFzp/HumanoidBench
- IsaacLab: https://github.com/isaac-sim/IsaacLab
- MuJoCo Playground: https://github.com/google-deepmind/mujoco_playground
- Raffin "SAC on massive parallel sim": https://araffin.github.io/post/sac-massive-sim/
- Shukla "Speeding up SAC": https://arthshukla.substack.com/p/speeding-up-sac-with-massively-parallel

---

## 18. 我会推荐的下一步阅读/实验方向

1. **MR.Q (Fujimoto et al. 2025)**: ICLR 2025, general-purpose model-free RL, 跟 FastTD3 互补。
2. **TDMPC2 + TDMPBC (Zhuang et al. 2025)**: self-imitative, 可能补 FastTD3 在长 horizon task 上的不足。
3. **Eureka (Ma et al. 2023)**: LLM 生成 reward, 配合 FastTD3 的短训练 cycle, 是天然的 next step. Paper Section 4 已经 explicitly 提到这点。
4. **Demo-driven RL (BiGym Chernyadev et al. 2024, RLAS Seo & Abbeel 2024)**: FastTD3 作为 off-policy 算法可以无缝接入 demo pretraining, 这是相对 PPO 的另一个优势。

如果让我赌一把, **FastTD3 + Eureka-style LLM reward designer + 12 小时人类睡眠周期** 可以做到 humanoid reward design 的 overnight 自动迭代。这就是这篇 paper 的真正长期价值——它把 iteration cycle 从 48 小时压到 3 小时, 让"睡前训一个, 醒来看结果"成为可能。
