---
source_pdf: FlashSAC- Fast and Stable Off-Policy Reinforcement.pdf
paper_sha256: 583ca4e87a1a909d0adc511daf7d2e1b81f3817a06d66d05500a8843342ea4a3
processed_at: '2026-08-04T08:55:01-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# FlashSAC 用人话讲

## 一句话总结

**让 off-policy RL 在高维机器人任务上又快又稳，关键是用一堆"规规矩矩"的架构技巧把 critic 训住，然后就能放心用大模型少更新，跟 LLM 的 scaling law 一个道理。**

---

## 问题是什么

机器人圈子里训 policy 有两条路:

**第一条路: PPO (on-policy)**
- 每次用当前 policy 跑一堆数据，学完就扔，再跑新的
- 优点: 稳，好调，quadruped 走路、gripper 抓东西这些低维任务上大家都用
- 缺点: 数据用完就丢，太浪费。低维任务 sim 快无所谓，但一到 humanoid (37 维 action)、dexterous hand (20 维)、vision-based，sample 不够覆盖整个空间，PPO 就开始拉胯

**第二条路: SAC / TD3 (off-policy)**
- 把数据存 replay buffer 里反复用
- 优点: 数据效率高，高维任务理论上更合适
- 缺点: 慢 + 不稳。慢是因为要反复 fit critic; 不稳是因为 critic 学 target，target 又依赖 critic 自己 (bootstrapping)，错一步往后全错，高维 + 大网络下尤其严重

所以现实是: 大家明知道 off-policy 数据效率高，但 sim-to-real 还是用 PPO，因为 off-policy 训着训着就崩。

---

## FlashSAC 的核心 idea

作者观察到一个被忽视的事实: **off-policy RL 慢，是因为大家不敢用大网络。不敢用大网络，是因为大网络在 bootstrapping 下会发散。** 一旦把稳定性问题解决，大网络 + 大 batch + 少更新这条 supervised learning 里被验证无数次的路径，在 off-policy RL 里同样 work。

这跟 LLM scaling law 是一个道理: 给定算力，与其用小模型跑 100 次 gradient step，不如用大模型跑 10 次，每次 batch 更大、信息更多。

但前提是: 大模型不能崩。

---

## 怎么让大模型不崩

五个层层叠加的 trick，每个都不新，但组合起来刚好够:

**1. 用 inverted residual block 当 backbone**
就是 Transformer FFN 那种 expand-then-compress 结构 + residual connection。比普通 MLP 稳，gradient 流得好。

**2. Pre-activation Batch Normalization**
BN 放在激活函数前面，防止 dead ReLU、防止 activation 漂移。选 BN 而非 LayerNorm，因为 batch 2048 够大，统计量稳。

**3. Cross-batch value prediction**
BN 有个老问题: 算 $Q(s,a)$ 和 target $Q(s',a')$ 时如果在不同 batch 里 forward，BN 统计不一样，等于在两个坐标系里做减法，error 会爆。解决: 把当前和下一步 transition 拼进同一个 batch forward，共享统计量。这个 trick 来自 CrossQ。

**4. Distributional critic + 自适应 reward scaling**
Q 值不输出一个标量，输出 101 个 atom 上的概率分布。好处是优化 landscape 平滑。但 distributional 有个 support 范围问题 (固定 $[-5,5]$)，return 超出去就漏掉。所以动态 scale reward: 如果 return variance 大就除大一点，让 effective return 始终落在 support 内。

**5. Weight normalization**
每次 gradient step 后把权重投影到单位球面。强制网络只通过方向编码信息，不能靠"权重变大"来放大 Q 值方差。这是关键 —— bootstrapping 下 Q 方差爆炸就是 error 爆炸。

这五个加起来，作者测了 condition number (Hessian 特征值比值)，每加一个就降一截，最终降到 MLP 的十分之一。condition number 低 = 优化好走 = 大模型能训。

---

## 训练配置有多反直觉

跟标准 off-policy 配方比:

| 维度 | 标准 SAC | FlashSAC |
|------|---------|----------|
| 网络大小 | 0.2-0.5M, 2-3 层 | 2.5M, 6 层 |
| Batch size | 256-512 | 2048 |
| Replay buffer | 1M | 10M |
| UTD (更新/数据比) | 1 或更高 | 2/1024 ≈ 0.002 |
| 并行环境 | 几个到几十 | 1024 |

UTD = 2/1024 意味着每收 1024 条新 transition 才更新 2 次。传统 wisdom 觉得这根本学不动，但配合大 batch + 大模型 + 上面那套稳定性 trick，反而收敛更快。

这跟 Kaplan scaling law 直接对应: 固定算力下，模型大、batch 大、step 少，比模型小、step 多更高效。

---

## 探索怎么做

两个小 trick:

**统一 entropy target**: SAC 要设 target entropy，传统是 $-|\mathcal{A}|$ (action 维度的负数)，但 8 维 Franka 和 37 维 humanoid 用同一个公式探索强度差异大。FlashSAC 改成"希望 action std 大概是 0.15" 这种 absolute scale，跨 embodiment 一致。最终全 paper 用同一个值 0.15，不用 per-task 调。

**Noise repetition**: 高维任务需要时间相关的噪声探索 (纯白噪声被惯性系统平均掉了)。pink noise、OU noise 都需要 per-env 维护状态，1024 个并行环境开销大。FlashSAC 的偷懒办法: 采一个噪声向量，重复用 k 步，k 从 Zeta 分布采 (短重复常见，偶尔来个长的)。只需一个 counter + 一个向量，minimal state。效果接近 pink noise，实现简单得多。

---

## 实验结果有多硬

60+ tasks, 10 simulators, 涵盖:
- 低维 state (gripper, quadruped): 跟 PPO 打平 (符合预期，低维 PPO 本来就强)
- 高维 state (dexterous hand, humanoid): 大幅超越 PPO 和 FastTD3
- CPU single env (MuJoCo, DMC, HumanoidBench, MyoSuite): 超越所有 sample-efficient baseline
- Vision (DMControl visual): 超越 DrQ-v2, MR.Q
- **Sim-to-real Unitree G1 humanoid**: 平地 20 分钟训完部署，PPO 要 3 小时; 没见过的 15cm 楼梯 4 小时爬上去，PPO 要 20 小时

最 impress 的是 sim-to-real 那个对比。G1 29 自由度盲走 (没视觉没激光)，4 小时训完能爬 sim 里没见过的楼梯。这把"off-policy 不可靠不适合 sim-to-real"这个 stereotype 直接掀了。

---

## 为什么这套能 work

我的直觉: 之前 off-policy RL 慢不是 off-policy 本身的问题，是 stability workaround 的副作用。大家为了让 critic 不崩，用小网络、慢更新、保守的 EMA target —— 这些都拖慢训练。FlashSAC 把 stability 用架构约束解决 (norm 控制)，就不用在训练 dynamics 上妥协，于是 scaling law 就能浮出来。

换个角度: supervised learning 的 scaling law 能 work，是因为 optimization landscape 良好。off-policy RL 的 landscape 被 bootstrapping 弄得崎岖，所以 scaling law 藏起来了。FlashSAC 的 norm 约束把 landscape 压平，scaling law 重新显现。

---

## 这篇 paper 对领域的意义

如果结果 reproducible， implication 是:

1. **Off-policy RL 不再是 PPO 的备胎**，在高维机器人上反而是首选
2. **Sim-to-real 不再必须是 PPO**，off-policy 在 humanoid 上快近 10 倍
3. **RL 也有 scaling law**，不只是 supervised learning 的专利，只是之前 stability 遮住了
4. **BN 在 off-policy RL 里比 LN 好**，只要处理 cross-batch 问题，大 batch 统计优势巨大

---

## 我会问作者的问题

- UTD 2/1024 还能更低吗？有没有 explicit scaling law 形式 $T = f(N, B, UTD)$ 可以 fit？
- BN + cross-batch 在 RNN 或 transformer policy 上还 work 吗？目前只测 MLP 和简单 CNN
- 大 replay buffer (10M) 在非 stationary 环境 (curriculum, domain randomization 强变化) 还稳吗？
- Weight norm 把信息全压到方向上，会不会限制 network 表达能力？跟 SimbaV2 的 hyperspherical normalization 是同一思路，长期训练会不会饱和？
- Noise repetition 的 Zeta exponent $s=2$ 怎么选的？跟 1/f noise 的 power spectrum 关系？
- Distributional critic 的 support $[-5,5]$ 在长 horizon sparse reward (manipulation 多阶段任务) 够吗？

---

## 一句话直觉

**FlashSAC = SimbaV2 的稳定性 + LLM 的 scaling 配方 + cheap 版 pink noise**。每个组件单独看都是 prior art 的延伸，组合起来解锁了"大模型 + 少更新"这个 off-policy RL 之前进不去的 regime。Engineering heavy，但 insight 是 conceptual: stability 不是 scaling 的对立面，而是 scaling 的前提。

项目主页: https://holiday-robot.github.io/FlashSAC

---

# FlashSAC: Fast and Stable Off-Policy RL 深度解析

## 1. Paper 核心论点与动机

FlashSAC 的核心 thesis 是: **off-policy RL 的两大顽疾——slow training 和 instability——可以通过借鉴 supervised learning 的 scaling laws 同时解决**, 前提是用一组架构层面的 norm constraints 来驯服 bootstrapping 带来的 error accumulation。

这个 motivation 很重要, 因为传统 wisdom 是:
- 想要快 → 用小网络 + 高 UTD (FastTD3 [75] 的路线)
- 想要稳 → 用小网络 + 强约束 (CrossQ [8], Simba [39], XQC [64] 的路线)
- 两条路互斥, 因为大网络在 bootstrapped update 下会发散

FlashSAC 的 key insight 是这俩可以解耦: **scaling laws 的"大模型少更新"在 off-policy RL 里依然成立, 只要你把 critic 的 update dynamics 控制住**。

参考:
- Soft Actor-Critic 原始 paper: https://arxiv.org/abs/1801.01290
- Scaling Laws (Kaplan et al.): https://arxiv.org/abs/2001.08361
- FastTD3: https://arxiv.org/abs/2505.22642
- CrossQ: https://arxiv.org/abs/2310.04235
- Simba: https://arxiv.org/abs/2410.09754
- XQC: https://arxiv.org/abs/2502.08069

---

## 2. SAC Preliminaries 详解

### 2.1 MDP 形式化

MDP $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, r, \gamma)$:
- $\mathcal{S}$: state space (高维时 dimension 可达 376, 如 Humanoid-v4)
- $\mathcal{A}$: continuous action space
- $P(s'|s,a)$: transition dynamics
- $r(s,a)$: reward function
- $\gamma \in [0,1)$: discount factor (IsaacLab 0.99, Playground 0.97)

### 2.2 Policy Loss (Eq. 2)

$$\mathcal{L}_\pi(\theta) = \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi_\theta} \big[ \alpha \log \pi_\theta(a|s) - \min_{i=1,2} Q_{\phi_i}(s,a) \big]$$

变量解释:
- $\theta$: policy network 参数
- $\alpha > 0$: entropy temperature, 自动调
- $\phi_i$: 第 i 个 critic 的参数, $i \in \{1,2\}$ (clipped double Q)
- $\min_{i=1,2}$: 取两个 Q 的最小值, 抑制 optimistic overestimation (来自 TD3 [15] 的 trick)
- $\log \pi_\theta(a|s)$: 在 state $s$ 下取 action $a$ 的 log-probability, 负号是 entropy (鼓励探索)
- $\mathcal{D}$: replay buffer

### 2.3 Target Network EMA (Eq. 3)

$$\bar{\phi}_j \leftarrow \tau \phi_j + (1-\tau) \bar{\phi}_j, \quad j \in \{1,2\}$$

- $\bar{\phi}_j$: target critic 参数 (overbar 表示 target)
- $\phi_j$: online critic 参数
- $\tau \in (0,1)$: target update rate, 这里 $\tau = 0.01$ (相对快)
- 直觉: target 是 online 的低通滤波版本, 减少 Bellman backup 的 moving target 问题

### 2.4 Critic Loss (Eq. 4 + Eq. 5)

$$\mathcal{L}_Q(\phi_i) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \big[ Q_{\phi_i}(s,a) - y \big]^2$$

$$y = r + \gamma \big( \min_{j=1,2} Q_{\bar{\phi}_j}(s', a') - \alpha \log \pi_\theta(a'|s') \big), \quad a' \sim \pi_\theta(\cdot|s')$$

- $y$: TD target (这里加了 entropy term, 即 soft Bellman)
- $\min_{j=1,2}$: 用 target network $\bar{\phi}_j$ (overbar) 而非 online
- $a'$ 是从当前 policy 采样的 next action, 用于估计 $V(s') = \mathbb{E}_{a' \sim \pi}[Q(s',a') - \alpha \log \pi]$
- 关键问题: 当 $Q_{\phi_i}$ 估计不准时, $y$ 也错, 然后这个错被 bootstrapping 不断放大 → deadly triad [87]

参考 deadly triad: https://arxiv.org/abs/1812.02648

---

## 3. FlashSAC 三大支柱深度剖析

### 3.1 Pillar I: Fast Training (§4.1) — Scaling Laws in Off-Policy RL

这一节是最"反直觉"的部分。常规 off-policy RL 的 recipe 是:
- UTD (update-to-data ratio) = 1 或更高 (每条 transition 平均更新 1 次以上)
- 小网络 (0.2-0.5M params)
- batch 256-512

FlashSAC 的配置:
- **1024 parallel environments** (massively parallel simulation, 借鉴 Isaac Gym [51] / Rudin et al. [68])
- **10M replay buffer** (比标准 1M 大 10x)
- **2.5M 参数, 6-layer** actor 和 critic (比标准 0.2-0.5M 大 5-12x)
- **batch size 2048** (近 GPU saturation)
- **UTD = 2/1024** (极低! 即 1024 new transitions 才更新 2 次)

为什么这能 work? 借用 Kaplan et al. [32] 的 scaling law intuition: 在 fixed compute budget 下, **大模型 + 大 batch + 少 step 比小模型 + 小 batch + 多 step 更高效**, 因为每个 gradient step 提供更多信息, 减少了 optimization overhead。

但这里有个 catch: **off-policy RL 的 batch 不是 i.i.d. 的**, replay buffer 里的 data 来自混合 policy (past policies 的 mix), 所以 batch 越大, 里面的 policy distribution 越杂。这正是为什么需要 §4.2 的 stabilization。

Replay buffer size 的 ablation (Figure 8a) 很有意思:
- 0.1M → 1M → 10M: 性能单调提升 (stability 提升)
- 10M → 50M: 性能反而下降! 因为 recent high-quality samples 被稀释了

这是一个 bias-variance trade-off:
- 太小 → catastrophic forgetting of rare states
- 太大 → stale data 占主导, policy mismatch 严重

代码优化细节:
- PyTorch JIT compilation (training + inference)
- Mixed precision [52]: 节省 5-10% wall-clock

参考:
- Isaac Gym: https://arxiv.org/abs/2108.10470
- Rudin et al. "Learning to walk in minutes": https://arxiv.org/abs/2109.11978
- Revisiting experience replay (Fedus et al.): https://arxiv.org/abs/2007.04739

### 3.2 Pillar II: Stable Training (§4.2) — Constrained Critic Update Dynamics

这是 paper 的真正技术核心。bootstrapping + function approximation + off-policy = deadly triad, 而 high-dimensional + large model 让 triad 更致命。FlashSAC 用 5 个机制层层约束:

#### 3.2.1 Inverted Residual Backbone

架构图 (Figure 2):
```
Input → [Inverted Residual Block] × N → RMSNorm → Value Head
```

每个 block 结构 (类比 Transformer FFN block [89]):
1. Linear: $d_{model} \to d_{ff}$ (inverted bottleneck [26], 即 $d_{ff} > d_{model}$, 典型 ratio 4x)
2. Nonlinearity
3. Linear: $d_{ff} \to d_{model}$
4. Residual connection [23]: $x \leftarrow x + \text{block}(x)$

这里 $d_{actor} = 128$, $d_{critic} = 256$, $N = 2$ blocks (actor 和 critic 都是 2 层)。总参数 2.5M 主要是宽度贡献的。

为什么用 inverted residual 而非普通 MLP?
- Residual connection 稳定 gradient flow
- Inverted bottleneck 让信息先 expand 再 compress, 类似 Transformer 的 FFN
- 这是 Simba [39] 系列工作积累的经验

#### 3.2.2 Pre-activation Batch Normalization

关键设计选择: **BN 放在 nonlinearity 之前**, 而非 post-activation。

为什么 pre-activation?
- 避免 dead ReLU [1] (saturation)
- 保留 gradient flow (Dohare et al. [12], Lyle et al. [47])
- 防止 non-stationary replay data 导致的 activation 漂移

为什么 BN 而非 LN [5]?
- BN 用 large batch statistics (2048 samples → 很稳的统计)
- LN 只用 single sample 的统计, 信息量少
- Santurkar et al. [70] 显示 BN smoothens loss landscape, 降低 condition number

但 BN 在 off-policy RL 有个 classic 陷阱: **current state $s$ 和 next state $s'$ 在不同 batch 里 forward, BN 统计不同, 导致 Bellman target 不一致**。

#### 3.2.3 Cross-Batch Value Prediction

这正是 CrossQ [8] 的 trick: **把 $(s, a, r, s')$ 拼成一个 batch 同时 forward**, 让 $Q(s,a)$ 和 $Q(s', a')$ 共享同一组 BN 统计。

具体说, batch 里前一半是 $(s, a)$, 后一半是 $(s', a')$ (shifted by 1), 然后 BN 在整个 batch 上算 mean/var。这样:
- $Q_{\phi_i}(s, a)$ 和 $Q_{\bar{\phi}_j}(s', a')$ 用同一个 normalized feature space
- Bellman target $y$ 和 prediction $Q_{\phi_i}(s,a)$ 数值一致

这是 stability 的关键, 否则 train 和 target 用不同 normalization, 相当于在两个不同坐标系里做差, error 会爆炸。

参考 CrossQ: https://arxiv.org/abs/2310.04235

#### 3.2.4 Distributional Critic with Adaptive Reward Scaling

FlashSAC 用 categorical distributional critic (C51 [7]):
- Q-value 表示为 $n_{atom} = 101$ 个 atoms 上的概率分布
- Support: $[G_{min}, G_{max}] = [-5, 5]$ (均匀分 101 个 bin)
- 训练用 cross-entropy loss 对 projected Bellman target

为什么 distributional?
- Smooth optimization landscape [64]
- 对 noisy target 更 robust
- Categorical 比 quantile (QR-DQN) 实现简单

但有个问题: distributional critic 的 support $[G_{min}, G_{max}]$ 是固定的, 而 return 的 magnitude 因 task 而异。如果 return 超出 support, probability mass 会 pile up 在边界 atoms 上, 信号丢失。

FlashSAC 的解决方案: **adaptive reward scaling (Eq. 6)**:

$$\bar{r}_t = \frac{r_t}{\max\big(\sqrt{\sigma_{t,G}^2 + \epsilon}, \, G_{t,\max} / G_{\max}\big)}$$

变量:
- $r_t$: 原始 reward
- $\bar{r}_t$: scaled reward
- $\sigma_{t,G}^2$: running discounted return variance (维护 EMA)
- $G_{t,\max}$: running max magnitude of return
- $G_{\max}$: critic support 的上界 (这里 5)
- $\epsilon$: small constant for numerical stability
- $\max(\cdot, \cdot)$: 取两者较大值, 既控制 variance 又控制 max outlier

直觉: scale reward 使得 effective return 落在 $[-G_{\max}, G_{\max}]$ 内。用 $\max$ 而非单纯 $\sqrt{\sigma^2}$ 是为了处理 heavy-tailed return (某些 task 偶尔有大 reward spike)。

为什么不 return centering [58] 或 loss scaling [71]?
- Return centering 只调 mean, 不调 variance
- Loss scaling 是 post-hoc 的, 不改变 critic 输出范围
- Direct reward scaling 让 critic 输出始终在 support 内, 没有 mass leakage

参考:
- C51: https://arxiv.org/abs/1707.06887
- Return-based scaling (Schaul et al.): https://arxiv.org/abs/2105.05347
- Reward centering (Naik et al.): https://arxiv.org/abs/2405.09999

#### 3.2.5 Weight Normalization

最后一个 stabilization: 每次梯度 step 后, 把权重 project 到 unit sphere:

$$W \leftarrow \frac{W}{\|W\|_2}$$

每个 normalization 参数 vector $(\gamma, \beta)$ project 到 norm $\sqrt{d}$ ($d$ 是该层宽度, 这样平均 activation 还是 unit scale)。

为什么?
- Uncontrolled weight growth → Q-value variance 爆炸 [48]
- 在 bootstrapping 下, variance 爆炸 = error 爆炸
- Weight norm 强制 network 只通过 direction (而非 scale) 编码信息, 类似 nGPT [46] 的 hyperspherical representation

这是 Lyle et al. [48] "Normalization and effective learning rates" 的应用, 也是 SimbaV2 [40] 的核心思想。Weight norm 单独看 ablation 增益 modest, 但在 sample-limited regime (CPU-based 实验) 下显著提升 robustness。

参考:
- nGPT: https://arxiv.org/abs/2410.01131
- Lyle et al. NeurIPS 2024: https://arxiv.org/abs/2406.12116
- SimbaV2: https://arxiv.org/abs/2502.15280

### 3.3 Pillar III: Exploration (§4.3) — Beyond SAC's Default

#### 3.3.1 Unified Entropy Target (Eq. 7)

SAC 自动调 $\alpha$, 但需要 target entropy $\bar{\mathcal{H}}$ 作为目标。常规做法 (SAC 原始 paper): $\bar{\mathcal{H}} = -|\mathcal{A}|$ (action dim 的负数), 但这对不同 embodiment 不一致 (Humanoid $|\mathcal{A}|=37$ vs Franka $|\mathcal{A}|=8$ 探索强度差异大)。

FlashSAC 的 parameterization:

$$\bar{\mathcal{H}} = \frac{1}{2}|\mathcal{A}| \log(2\pi e \sigma_{tgt}^2)$$

变量:
- $|\mathcal{A}|$: action dimension
- $\sigma_{tgt}$: target action standard deviation (超参, 全 paper 用 $\sigma_{tgt} = 0.15$)
- $2\pi e \sigma_{tgt}^2$: 这是 diagonal Gaussian $\mathcal{N}(0, \sigma_{tgt}^2 I)$ 的 differential entropy per dimension

推导: 对角 Gaussian $\pi(a|s) = \mathcal{N}(\mu, \text{diag}(\sigma^2))$ 的 entropy 是 $\frac{1}{2} \sum_i \log(2\pi e \sigma_i^2) = \frac{1}{2}|\mathcal{A}| \log(2\pi e \sigma^2)$ (假设各维 independent 等方差)。

直觉: **用 absolute action scale ($\sigma_{tgt}$) 而非 relative scale ($-|\mathcal{A}|$) 来定义 exploration 强度**。$\sigma_{tgt} = 0.15$ 意思是希望 policy 的 action 噪声大致在 0.15 的 std (假设 action 归一化到 $[-1, 1]$)。这个值与 action dim 无关, 所以跨 embodiment 一致。

Ablation (Figure 10a): $\sigma_{tgt} \in \{0.05, 0.1, 0.15, 0.2, 0.25\}$ 都能收敛到类似 performance, 0.15-0.20 略优。这表明这个 parameterization 很 robust, 不需要 per-task tuning。

#### 3.3.2 Noise Repetition

Temporally correlated noise 对 sparse-reward / high-dimensional exploration 很重要 (pink noise [13], OU noise [25]), 但这些需要 per-environment state, 在 1024 parallel envs 下 memory/compute overhead 大。

FlashSAC 的 trick: **noise repetition** — 采样一个 $\epsilon \sim \mathcal{N}(0, I)$, 持续 $k$ 步不变, $k$ 从 Zeta distribution 采样:

$$P(k) \propto k^{-s}, \quad s = 2, \quad k_{max} = 16$$

变量:
- $k$: repetition length
- $s$: Zeta distribution 的 exponent (类似 power law)
- $k_{max}$: max repeat limit (clipped)

为什么 Zeta distribution?
- Power law $P(k) \propto k^{-s}$ 是 long-tailed, 短 repeat 频率高, 长 repeat 偶尔出现
- 这模拟了"exploration 的 timescale 多样性" — 有时小幅扰动 ($k=1$), 有时持续探索 ($k=16$)
- 跟 pink noise 的 power spectrum 思想类似, 但实现简单得多 (只需局部 state)

直觉对比:
- White noise: 每步独立 $\epsilon_t$, 高频, 在 inertial system 下被 dynamics 平均掉
- OU noise: 需要 per-env state 维护, memory 1024x
- Pink noise: 需要频域滤波, complex
- Noise repetition: 1 个 counter + 1 个 noise vector per env, minimal state

Ablation (Figure 10b): disable noise repeat → 收敛明显变慢 + asymptotic performance 下降。在高维 control (Humanoid, dexterous hand) 下尤其明显。

参考:
- Pink noise: https://arxiv.org/abs/2306.05826
- Temporally extended ε-greedy (Dabney et al.): https://arxiv.org/abs/2006.01782

---

## 4. Experiments 全面分析

### 4.1 实验矩阵

60+ tasks, 10 simulators, 4 大类:

| 类别 | Simulator | 代表 task | 维度特征 |
|------|-----------|-----------|----------|
| GPU state, low-dim | IsaacLab, ManiSkill, Genesis | Franka manipulation, AnyMal locomotion | $|\mathcal{S}|, |\mathcal{A}| < 50$ |
| GPU state, high-dim | IsaacLab, MuJoCo Playground | Allegro/Shadow Hand, G1/H1/T1 humanoid | $|\mathcal{A}| = 16-37$ |
| CPU state | MuJoCo, DMC, HumanoidBench, MyoSuite | Humanoid-v4 (376-dim obs!), Dog (223-dim) | sample efficiency 关键 |
| Vision | DMControl Visual | Finger, Walker, Cheetah | render 慢, sample budget 1M |
| Sim-to-real | IsaacLab → Unitree G1 | 29-DoF humanoid, rough terrain | 真 hardware deployment |

### 4.2 关键 baseline 对比

- **PPO** [72, 73]: RSL-RL 实现, 代表 sim-to-real SOTA, 训练 200M steps (3x FlashSAC compute)
- **FastTD3** [75]: wall-clock 优化 off-policy, 但用 0.2M 小网络, 限制 asymptotic
- **CrossQ** [8]: BN-based sample efficient
- **SimbaV2** [40]: hyperspherical normalization
- **XQC** [64]: well-conditioned off-policy (ICLR 2026)
- **TD-MPC2** [22]: model-based + planning, vision 强但 per-step 贵
- **MR.Q** [17]: model-free with model-based representation objective
- **DrQ-v2** [92]: vision-based, image augmentation

### 4.3 State-Based Results (Figure 3)

**Low-dim tasks**: FlashSAC ≈ PPO (PPO 在 low-dim 仍强, 因为 sample throughput 极高)。这是 expected — on-policy 在小空间高 throughput 下的优势。

**High-dim tasks**: FlashSAC 显著优于 PPO, 这是 paper 的核心实验 claim。在 dexterous manipulation (Shadow Hand cube reorientation) 和 humanoid locomotion 上, FlashSAC 收敛快 + asymptotic 高。

vs FastTD3: FlashSAC 在 Go2Walk, Franka Pull Cube 等 FastTD3 失败的 task 上稳定收敛。这归因于 FlashSAC 的 stabilization mechanisms — FastTD3 用小网络避开发散, 但牺牲了 capacity。

### 4.4 CPU-Based Results (§5.2)

CPU-based 设置 (1 env, sample efficiency 是 bottleneck) 是 FlashSAC 的"非主场"测试。Hyperparameter 调整:
- batch 2048 → 512
- UTD 2/1024 → 1
- replay buffer 10M → 1M

结果: FlashSAC 仍超越所有 baseline。这表明 stabilization techniques 不依赖 massively parallel simulation, 在 single-env sample-efficient regime 也 work。

### 4.5 Vision-Based Results (§5.3)

8 个 DMControl visual tasks, 关键 modification:
- Lightweight CNN encoder: 3 conv layers + linear bottleneck
- Frame stack 3 (84×84×9), 给 temporal info 不用 RNN
- 3-step returns (better credit assignment in vision)
- Action repeat 2
- UTD 0.5 (vision throughput 低)

vs DrQ-v2: DrQ-v2 sample efficient 但 unstable, Finger Turn Hard 等失败。vs MR.Q: MR.Q 高 asymptotic 但 auxiliary model 增 cost。FlashSAC 用单套 hyperparameter, no auxiliary objective。

### 4.6 Sim-to-Real (§5.4) — 最 impress 的部分

Unitree G1 29-DoF humanoid, blind locomotion (no exteroceptive)。两个 terrain:
- Flat: FlashSAC 20 分钟, PPO 3 小时 (~9x speedup)
- Rough stairs (15cm, unseen during training): FlashSAC 4 小时, PPO 20 小时 (~5x speedup)

关键 sim-to-real 技巧 (与 PPO 公平对比):
1. **Asymmetric actor-critic** [66]: critic 训练时拿 privileged info (height map, contact state, ground-truth velocity), actor 只拿 proprioception → deploy 时 critic 不需要
2. **Context estimator (CENet)** [57]: 用 proprioception history 估计 latent context $z_t$, 用于 implicit system identification
3. **Symmetry augmentation** [53]: 利用 robot bilateral symmetry augment batch
4. **Domain randomization**: friction, mass, motor strength 等
5. **Terrain curriculum** [69]: 10 levels, 50% success 自动升级

Reward 设计 (Table 14) 很有意思:
- FlashSAC 不需要 termination penalty, 只要 alive bonus 1.0
- PPO 需要 -200 termination penalty 才能稳定
- 这反映 FlashSAC 的 off-policy 更稳, 不需要 reward shaping 强行约束 safety

观察空间 (Eq. 8):
$$\mathbf{o}_t = [\omega_t, \mathbf{g}_t, \mathbf{c}_t, q_t, \dot{q}_t, \mathbf{a}_{t-1}]^\top$$

- $\omega_t$: base angular velocity (3D)
- $\mathbf{g}_t$: projected gravity (3D)
- $\mathbf{c}_t$: velocity command (3D)
- $q_t, \dot{q}_t$: joint position / velocity (29D each)
- $\mathbf{a}_{t-1}$: previous action (29D)

参考:
- DreamWaQ: https://arxiv.org/abs/2301.10602
- Asymmetric actor-critic: https://arxiv.org/abs/1710.06542
- RSL-RL: https://arxiv.org/abs/2509.10771

---

## 5. Ablation Studies 深度解读 (§6)

### 5.1 Off-Policy vs On-Policy Data Coverage (§6.1, Figure 7)

Shadow Hand cube reorientation, 1M steps。Off-policy data 从 replay buffer 取, on-policy 是 final policy rollout 1M steps。

2D density plot (object y-position × finger action) 显示:
- Off-policy: **broad coverage**, 覆盖大部分 state-action 空间
- On-policy: **tight cluster** 围绕 final policy 的 mode

这是 paper 的核心 conceptual argument: **high-dim space 下, on-policy 即使 collect 1M steps 也覆盖不全, off-policy 通过 replay buffer 累积 diverse policy 的 data 才能覆盖**。这是为什么 on-policy 在 high-dim 下崩, off-policy (如果稳定) 才能搞定。

### 5.2 Scaling Ablation (§6.2, Figure 8)

5 个 hyperparameter 的 univariate ablation, 在 4 个 IsaacLab tasks 上:

| 维度 | 探索范围 | 趋势 |
|------|----------|------|
| Replay buffer | 0.1M → 50M | 10M 最优, 50M 下降 (stale data) |
| Batch size | 0.5K → 8K | 越大越快 (GPU saturation 上限) |
| Network width | 64 → 1024 | 越大越快 (stability 容许) |
| Network depth | 1 → 4 blocks | 2-3 最优 (过深 diminishing return) |
| UTD | 0.5/1024 → 8/1024 | 越低越快 (compute-efficient) |

这些都是 **wall-clock time** 的 plot, 不是 sample efficiency。在 wall-clock 视角下, "少 update + 大 batch + 大 model" 全面胜出。

这印证了 paper 的 scaling claim: **off-policy RL 也服从 scaling laws, 之前没观察到是因为没解决 stability**。

### 5.3 Architectural Ablation (§6.3, Figure 9)

增量添加组件, 测 parameter/feature/gradient norm + condition number:

MLP → +Residual → +BN → +Post-RMSNorm → +Dist Critic + Reward Scale → +Weight Norm (FlashSAC)

每加一个组件:
- Parameter norm: 保持 bounded (尤其 weight norm 后严格 bounded)
- Feature norm: bounded (RMSNorm 后 bounded per sample)
- Gradient norm: bounded
- **Condition number 单调下降** (loss landscape 越来越 well-conditioned)

Condition number 是 [64] 提出的 metric, 衡量 critic loss Hessian 的 eigenvalue ratio, 越大越难优化。FlashSAC 的 condition number 是 MLP 的 ~1/10, 这是 stability 的直接证据。

### 5.4 Exploration Ablation (§6.4, Figure 10)

- $\sigma_{tgt} \in \{0.05, 0.05, 0.15, 0.2, 0.25\}$: 都能收敛, 0.15-0.20 最佳, 鲁棒
- Noise repeat on/off: on 显著加速收敛

---

## 6. 关键 Insights & Open Questions

### 6.1 为什么 FlashSAC work — 我的 intuition

FlashSAC 的成功可以分解为两层:

**Layer 1 (Optimization)**: 大 batch + 大 model 在 off-policy RL 里其实一直 work, 之前没被观察到是因为 bootstrapping instability 把 signal 埋了。一旦 stability 解决 (通过 norm constraints), scaling laws 浮出水面。

**Layer 2 (Statistical)**: 
- BN 在大 batch 下统计更稳, 这正是 FlashSAC 用 BN 而非 LN 的关键
- Cross-batch prediction 消除 train/target 的 distribution mismatch
- Distributional critic + reward scaling 把数值范围控制住

**Layer 3 (Exploration)**:
- Unified entropy target 让 exploration 强度与 action scale 挂钩, 跨 task 一致
- Noise repetition 是"pink noise 的 cheap approximation", 在 massively parallel setting 下不增 memory

### 6.2 与其他 scaling 方向的关系

**vs Simba [39] / SimbaV2 [40] / XQC [64] / CrossQ [8]**: FlashSAC 把这些工作的 stabilization techniques 整合, 加上 scaling 的视角。可以理解为 "SimbaV2 + scaling + noise repeat"。

**vs Nauman et al. [60, 61] "Bigger, regularized, optimistic"**: 同样主张大 model + regularization, 但 FlashSAC 更强调 UTD 的 role。

**vs Obando-Ceron et al. [62] "Simplicial embeddings"**: 探索不同的 representation space, FlashSAC 用 hyperspherical (weight norm) 是类似 spirit。

### 6.3 Open Questions / 我会问作者的问题

1. **UTD 2/1024 是极限吗?** — 更低 (如 1/2048) 会怎样? 是不是有 scaling law 形式 $UTD \propto 1/\text{batch size}$?
2. **BN 在 RNN / transformer-based policy 上还能 work 吗?** — 当前是 MLP backbone, vision 用 CNN
3. **Distributional critic 的 support $[-5, 5]$ 在更长 horizon task (如 manipulation with sparse reward) 够吗?** — n-step return 会扩大 effective magnitude
4. **Noise repetition 的 Zeta exponent $s=2$ 是怎么选的?** — 跟 1/f noise 的 power spectrum 关系?
5. **Sim-to-real 在更复杂 terrain (gap, slope, soft ground) 上还稳吗?** — paper 只测 stairs
6. **与 AWAC [6], IQL [Ball et al.] 等 offline-to-online 方法的关系?** — FlashSAC 的大 buffer 接近 offline setting
7. **Asymmetric actor-critic 的 privileged info 在 critic 学习中会不会 leak 到 policy?** — 这是 sim-to-real 的经典 concern
8. **Symmetry augmentation 在 non-symmetric task (如 bimanual with different tools) 还 work 吗?**

### 6.4 Failure Modes 我猜的

- **Replay buffer 太大 (50M)**: stale data dilute signal, 收敛慢
- **Network 太深 (4+ blocks)**: gradient 消失? 或 BN 统计不稳?
- **BN 在 RNN 上**: 不适用, 这是为什么 vision 用 frame stack 而非 RNN
- **Reward spike 超过 support**: adaptive scaling 的 $\max$ 项会触发, 但如果 spike 太频繁, scaling 会过 aggressive, 破坏 reward signal
- **Asymmetric critic 的 privileged info 在 inference 不可用**: 如果 actor 学到"作弊"依赖 critic 的 leak, sim-to-real 会失败 — paper 用 context estimator mitigate 这个

### 6.5 对 RL Community 的 implication

FlashSAC 的成功提示:
1. **Off-policy RL 不必慢** — 慢是 stability workaround 的副作用, 解决 stability 就能 scale
2. **Scaling laws 跨越 supervised → RL** — 只要 stability 解决, 同样 laws 适用
3. **Sim-to-real 不必是 PPO** — off-policy + stabilization 在高维 humanoid 上更优
4. **BN > LN in off-policy RL** — 因为 large batch 统计稳, LN 只用 single sample 信息少

### 6.6 与 LLM Scaling 的类比

| LLM | FlashSAC |
|-----|----------|
| Larger model, more data, fewer epoch | Larger model, larger batch, fewer UTD |
| AdamW + gradient clipping | Weight norm + BN |
| Layer norm (post) | RMS norm (post) + BN (pre) |
| Cosine LR schedule | Cosine decay (3e-4 → 1.5e-4) |
| Mixed precision | Mixed precision |
| Scaling law $L \propto N^{-\alpha}$ | Wall-clock $\propto$ ? (paper 没给出 explicit law) |

如果 FlashSAC 真的 obey scaling law, 下一步应该是 fit 一个 explicit law: $T_{\text{converge}} = f(\text{params}, \text{batch}, \text{UTD})$, 像 Chinchilla [Hoffmann et al.] 那样找 optimal allocation。

参考 Chinchilla: https://arxiv.org/abs/2203.15556

---

## 7. Implementation Notes 我注意到的

- **Code 是 PyTorch**, JIT-compiled, mixed precision
- **4096 envs for sim-to-real** (vs 1024 for state-based eval) — 因为 sim-to-real 需要更多 domain randomization samples
- **50Hz policy, 200Hz PD control** — 标准 legged robot 配置
- **Cosine LR decay**: 3e-4 → 1.5e-4 (half)
- **Adam β = (0.9, 0.999)** (default)
- **Target momentum τ = 0.01** (相对快, 配合 low UTD)
- **Critic update delay = 2** (TD3 trick, actor 2 次更新 critic 1 次? 反过来? 需要看代码)

Hyperparameter 总览 (Table 9-11):
- GPU state: batch 2048, UTD 2/1024, buffer 10M, n-step=1
- CPU state: batch 512, UTD 1, buffer 1M, n-step=1
- Vision: batch 256, UTD 0.5, buffer 1M, n-step=3, action repeat 2, frame stack 3

注意 vision 用 3-step return + action repeat 2 = effective 6-step horizon, 这是 sample efficiency 的关键 (TD 错误 backpropagate 6 步)。

---

## 8. 我的 Overall Take

FlashSAC 是一篇"engineering-heavy 但 insight 深"的 paper。它的 contribution 不在单个 trick, 而是**整合 + scaling 视角**:

1. **Conceptual**: 把 supervised learning 的 scaling law thinking 引入 off-policy RL, 证明只要 stability 解决, scaling laws 适用
2. **Engineering**: 集成 5 个 stabilization components, 每个都有 prior work 支撑, 但组合起来解锁大模型 + 少 update 的 regime
3. **Practical**: 60+ tasks, 10 simulators, sim-to-real 验证, 单套 hyperparameter
4. **Limitation**: 主要测 state-based, vision 只在 DMControl, sim-to-real 只在 G1 flat/stairs; complex contact (garment, deformable) 没测

**最强 claim**: off-policy RL 在 high-dim robot control 上不仅可行, 还比 PPO 快近 10x, 这是 paradigm shift — 如果 reproducible, 会改变 robot learning 的 default recipe。

**最弱 link**: stability mechanisms 的组合是"经验性的 stack", 没有理论解释为什么这 5 个组件恰好够。Ablation 是 incremental addition, 没测 component 之间的 interaction。

**未来方向** (paper §7 提到 + 我加的):
- Tactile-based learning (paper 提)
- 复杂 contact (deformable, garment) [91]
- Offline-to-online fine-tuning [6] (paper 提)
- Explicit scaling law fitting (Chinchilla for RL)
- Transformer/Vision-language policy backbone (替换 MLP/CNN)
- Multi-task single buffer (cross-embodiment, 类似 RT-2)
- Asymmetric critic 的 info leak analysis

参考 RT-2: https://arxiv.org/abs/2307.15818
参考 DexMimicGen: https://arxiv.org/abs/2502.07091

---

## 9. Web Reference 汇总

**Core FlashSAC related**:
- Project page: https://holiday-robot.github.io/FlashSAC
- SAC: https://arxiv.org/abs/1801.01290
- TD3: https://arxiv.org/abs/1802.09477
- DDPG: https://arxiv.org/abs/1509.02971

**Stabilization priors**:
- CrossQ (BN in RL): https://arxiv.org/abs/2310.04235
- Simba: https://arxiv.org/abs/2410.09754
- SimbaV2: https://arxiv.org/abs/2502.15280
- XQC: https://arxiv.org/abs/2506.08069
- PLASTIC: https://arxiv.org/abs/2306.13812
- Lyle et al. (plasticity): https://arxiv.org/abs/2306.13812
- Lyle et al. (norm & LR): https://arxiv.org/abs/2406.12116
- Bigger regularized optimistic: https://arxiv.org/abs/2405.16158

**Distributional & scaling**:
- C51: https://arxiv.org/abs/1707.06887
- Reward centering: https://arxiv.org/abs/2405.09999
- Return-based scaling: https://arxiv.org/abs/2105.05347
- nGPT: https://arxiv.org/abs/2410.01131
- Scaling laws (Kaplan): https://arxiv.org/abs/2001.08361
- Chinchilla: https://arxiv.org/abs/2203.15556

**Exploration**:
- Pink noise: https://arxiv.org/abs/2306.05826
- Temporally extended ε-greedy: https://arxiv.org/abs/2006.01782

**Sim-to-real & loco**:
- DreamWaQ: https://arxiv.org/abs/2301.10602
- RMA: https://arxiv.org/abs/2107.04034
- Learning to walk in minutes: https://arxiv.org/abs/2109.11978
- Asymmetric actor-critic: https://arxiv.org/abs/1710.06542
- RSL-RL: https://arxiv.org/abs/2509.10771
- FastTD3: https://arxiv.org/abs/2505.22642
- Anymal parkour: https://www.science.org/doi/10.1126/scirobotics.adi7566

**Simulators**:
- Isaac Gym: https://arxiv.org/abs/2108.10470
- Isaac Lab: https://arxiv.org/abs/2511.04831 (wait, 2025 future date? likely typo in paper)
- ManiSkill3: https://arxiv.org/abs/2410.00425
- MuJoCo Playground: https://arxiv.org/abs/2502.08844
- Genesis: https://github.com/Genesis-Embodied-AI/Genesis
- HumanoidBench: https://arxiv.org/abs/2403.10506
- MyoSuite: https://arxiv.org/abs/2205.13600
- DMC: https://arxiv.org/abs/1801.00690

**Vision RL**:
- DrQ-v2: https://arxiv.org/abs/2107.09645
- MR.Q: https://arxiv.org/abs/2501.16142

**Deadly triad & theory**:
- Deadly triad: https://arxiv.org/abs/1812.02648
- Regulating overfitting in RL: https://arxiv.org/abs/2304.10466

**Other baselines**:
- DreamerV3: https://arxiv.org/abs/2301.04104
- TD-MPC2: https://arxiv.org/abs/2310.16828

**Robot learning broader**:
- OpenAI dexterous hand: https://arxiv.org/abs/1808.00169
- Gemini Robotics: https://arxiv.org/abs/2503.20020
- OpenVLA: https://openvla.github.io/

---

总结一句: FlashSAC 把 off-policy RL 从"slow but data-efficient"的 stereotype 拉到"fast + stable + data-efficient"的新 regime, 关键是 stabilization 解锁 scaling。如果 reproducible, 这会改变 robot learning 的 default algorithm choice, 尤其在 humanoid / dexterous / vision 这类高维任务上。
