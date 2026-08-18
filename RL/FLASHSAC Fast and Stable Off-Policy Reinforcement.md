---
source_pdf: FLASHSAC Fast and Stable Off-Policy Reinforcement.pdf
paper_sha256: e16934541e9ecc677a919f57cfff70b0c74a0d5489d935671a860f7bb383c961
processed_at: '2026-08-18T13:19:38-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# FLASHSAC 用人话讲

## 一句话总结

**Off-policy RL 之所以慢且不稳，本质是因为 critic 在"自我抄笔记"——抄着抄着错越积越多。FLASHSAC 的核心贡献是用一堆 norm 约束让这个"抄笔记"过程稳定下来，从而能用大模型 + 大 batch + 少更新这个 supervised learning 的 scaling law 套路，把 humanoid sim-to-real 从 20 小时压到 4 小时。**

---

## I. 为什么需要这篇 paper：PPO 在 high-dim 上的窘境

### 类比：PPO 是"考完就忘"的学生

PPO 这种 on-policy 方法的工作模式很像一个极端的学生：每次小测考完，试卷直接扔掉，下次考试完全靠最近几天突击。在 quadruped locomotion、Franka gripper 这种 low-dim 任务上这没问题——simulator throughput 极高，"试卷"（数据）便宜得很，扔了再印就行。

但当任务维度上升：
- **Unitree G1 humanoid**：29-DoF action space
- **Shadow Hand dexterous manipulation**：24-DoF hand + 6-DoF object pose
- **Vision-based control**：84×84×9 pixels

state-action space 的 volume 随 dimension 指数膨胀，PPO 每次 rollout 只能在 manifold 上划一道窄窄的轨迹，state-action coverage 严重稀疏。Paper 里 Figure 7 给了个很直观的 2D density plot：off-policy buffer 覆盖一大片区域，on-policy final policy 的 rollout 只集中在一条细线上。

这就是 PPO 在 dexterous manipulation 和 humanoid locomotion 上经常失败的根因——**算法没错，coverage 不够**。

参考 [PPO original paper (Schulman 2017)](https://arxiv.org/abs/1707.06347)、[RSL-RL implementation (Schwarke 2025)](https://arxiv.org/abs/abs/2509.10771)。

---

## II. Off-Policy 的 promise 与 pitfall

### 类比：Off-policy 是"记笔记可复习"，但 critic 在"自我抄笔记"

Off-policy RL（SAC、TD3 这类）用 replay buffer 存历史数据，可以反复学习。直觉上这是解决 coverage 问题的正解。

但 off-policy 有个致命弱点：**critic 的训练 target 依赖 critic 自己的 prediction**。看 Bellman target：

$$y = r + \gamma \min_{j=1,2} Q_{\bar{\phi}_j}(s', a')$$

变量解释：
- $r$ = reward
- $\gamma \in [0,1)$ = discount factor
- $Q_{\bar{\phi}_j}(s', a')$ = target network（slow EMA 副本）预测的 next-state Q-value
- $a' \sim \pi_\theta(\cdot|s')$ = 下一步 action

这个 target 是 critic 用自己的预测值算出来的。如果 critic 在某些 OOD state 上预测偏高，这个误差会通过 bootstrap 递归放大——这就是 [van Hasselt et al. 2018 "deadly triad"](https://arxiv.org/abs/1812.02648) 说的事。

### 既往 workaround 的局限

社区之前分三条路：

1. **提速派**：FastTD3 ([Seo 2025](https://arxiv.org/abs/2505.22642))、FastSAC——靠 parallel sim + 大 buffer 提速，但只能用 0.2M 参数小网络，asymptotic performance 天花板低
2. **稳派**：CrossQ ([Bhatt ICLR 2024](https://arxiv.org/abs/2403.07818))、Simba ([Lee 2024](https://arxiv.org/abs/2410.09754))、XQC ([Palenicek ICLR 2026](https://arxiv.org/abs/2407.04811))——用 norm constraint 稳，但需要多 gradient update 收敛，慢
3. **探索派**：pink noise ([Eberhard ICLR 2023](https://arxiv.org/abs/2207.06236))、OU process——解决 high-dim 探索，但 trainer 慢与不稳没动

FLASHSAC 的贡献：**把三条路统一**，用稳定化让大模型可用，用大模型+大 batch 抵消少更新的损失，用 noise repetition 替代 pink noise。

---

## III. FLASHSAC 的三大设计，用类比讲

### A. Fast Training：把 supervised learning 的 scaling law 搬到 RL

#### 核心反直觉：UTD = 2/1024

UTD (Update-to-Data ratio) = 每采集 1 个 environment step 做几次 gradient update。

- 标准 SAC：UTD = 1
- RedQ：UTD = 20（密集更新）
- **FLASHSAC：UTD = 2/1024 ≈ 0.002**（1024 个 parallel env 共享 2 次更新）

直觉上 UTD 越低 critic 越欠拟合。但 [Kaplan 2020 scaling laws](https://arxiv.org/abs/2001.08361) 与 [Hoffmann 2022 Chinchilla](https://arxiv.org/abs/2206.00685) 告诉我们：固定 FLOP budget 下，大 batch + 大模型 + 少 step 比小 batch + 小模型 + 多 step 更快收敛。LLM 训练都是这个套路。

**为什么 RL 之前没用？** 因为大模型在 bootstrapping 下更不稳。这是 FLASHSAC 要解决的核心矛盾。

#### 具体配置

| 项目 | 标准 SAC | FastTD3 | FLASHSAC |
|---|---|---|---|
| Buffer size | 1M | 1M | **10M** |
| Network params | 0.2M | 0.2M | **2.5M** |
| Batch size | 256 | 1024 | **2048** |
| Layers | 2-3 | 3 | **6** |
| UTD | 1 | 1 | **2/1024** |

10M buffer 的意义：high-dim 任务中 rare but critical 的 state-action pair 在小 buffer 中会被 overwrite（catastrophic forgetting），导致 extrapolation error。这与 [Fedus 2020 "Revisiting Experience Replay"](https://proceedings.mlr.press/v119/fedus20a.html) 的分析一致。

1024 parallel environments：通过 IsaacLab、ManiSkill3、Genesis 这些 GPU-based simulator 实现。

JIT + mixed precision ([Micikevicius 2017](https://arxiv.org/abs/1710.03740))：省 5-10% wall-clock。

---

### B. Stable Training：五重 norm 约束让 bootstrapping 收敛

这是 paper 最核心的技术贡献。直觉是：**bootstrapping 下 error 是否爆炸取决于 critic 是否是 contractive mapping**。Contractive mapping 的 Lipschitz 常数 <1，所以误差会衰减而非放大。Lipschitz 常数与 weight norm、feature norm 直接相关，所以要把它们 bound 住。

下面五项是层层加码的 stabilization，paper Figure 9 的 ablation 显示每加一项 condition number 都单调下降。

#### B.1 Inverted Residual Backbone

灵感：Transformer feedforward block ([Vaswani 2017](https://arxiv.org/abs/1706.03762)) + MobileNet inverted bottleneck ([Howard 2017](https://arxiv.org/abs/1704.04861))。

每个 block 数据流：
1. Input $x \in \mathbb{R}^d$（$d=1024$）
2. Expansion: $x' = W_{\text{up}} x \in \mathbb{R}^{4d}$
3. GELU/SiLU nonlinearity
4. Projection: $y = W_{\text{down}} h \in \mathbb{R}^d$
5. Residual: $x_{\text{out}} = x + y$
6. 6 层堆叠
7. 最后接 RMSNorm（[Zhang & Sennrich 2019](https://arxiv.org/abs/1910.07467)）

RMSNorm 公式：
$$\text{RMSNorm}(x_i) = \frac{x_i}{\sqrt{\frac{1}{d}\sum_j x_j^2 + \epsilon}} \cdot g_i$$

变量：
- $x_i$ = 第 $i$ 维 feature
- $d$ = hidden dim
- $g_i$ = learnable scale
- $\epsilon$ = 防除零

**作用**：bound per-sample feature norm before value head，防止 OOD state 产生 unbounded activation 进入 Bellman target。

#### B.2 Pre-activation Batch Normalization

每层 nonlinearity 之前插 BatchNorm，不是 LayerNorm。

公式：
$$\hat{x}_i = \gamma \cdot \frac{x_i - \hat{\mu}_B}{\sqrt{\hat{\sigma}_B^2 + \epsilon}} + \beta$$

变量：
- $\hat{\mu}_B = \frac{1}{m}\sum_i x_i$ = batch mean
- $\hat{\sigma}_B^2$ = batch variance
- $m$ = batch size（2048）
- $\gamma, \beta$ = learnable scale 与 shift

**为何不用 LayerNorm？** 两个理由：

1. **大 batch 的统计稳定性**：2048 batch 来自 diverse replay data，$\hat{\mu}_B, \hat{\sigma}_B^2$ variance 极低。这呼应 [Santurkar 2018 "How Does BN Help Optimization"](https://arxiv.org/abs/1805.11604)——BN 的真正 benefit 不是减少 internal covariate shift，而是 reparametrize loss landscape 让其 Lipschitz 常数更小、condition number 更低
2. **LayerNorm 在 RL 中丢 cross-sample 信息**，对 diverse replay distribution 不友好

#### B.3 Cross-Batch Value Prediction

这是 BatchNorm 在 RL 中的经典坑。

**问题**：Bellman target $y = r + \gamma Q_{\bar\phi}(s', a')$ 中，$Q_\phi(s, a)$ 与 $Q_{\bar\phi}(s', a')$ 在两次独立 forward pass 中算，各自用不同 batch 统计 $\hat{\mu}_B, \hat{\mu}_{B'}$。同一 state 在不同上下文下被不同 normalize，引入 systematic bias。

**解法**：把 $(s, a, r, s', a')$ 拼成 single batch，让 $Q_\phi(s,a)$ 与 $Q_{\bar\phi}(s',a')$ 共享同一组 BN 统计。

直觉：把 Bellman backup 视作 self-distillation，teacher 与 student 必须在同一个 normalization frame 下比较。这个 trick 来自 [CrossQ (Bhatt ICLR 2024)](https://arxiv.org/abs/2403.07818)。

#### B.4 Distributional Critic + Adaptive Reward Scaling

Q-value 表示为 categorical distribution over $n_{\text{atom}}$ atoms 均匀分布在 $[G_{\min}, G_{\max}]$。来自 [C51 (Bellemare 2017)](https://arxiv.org/abs/1707.06870)。

- 网络输出 softmax 概率 $p_\theta(\cdot|s,a) \in \Delta^{n_{\text{atom}}}$
- 期望 Q：$Q_\theta(s,a) = \sum_i p_i z_i$，其中 $z_i = G_{\min} + \frac{i-1}{n_{\text{atom}}-1}(G_{\max}-G_{\min})$
- 训练用 cross-entropy，比 MSE 平滑

**Adaptive Reward Scaling（Equation 6）**：

$$\bar{r}_t = \frac{r_t}{\max\left(\sqrt{\sigma_{t,G}^2 + \epsilon}, \; G_{t,\max}/G_{\max}\right)}$$

变量：
- $r_t$ = 原始 reward
- $\bar{r}_t$ = normalized reward
- $\sigma_{t,G}^2$ = discounted return variance 的 EMA 估计
- $G_{t,\max}$ = 历史最大 return magnitude 的 EMA
- $G_{\max}$ = categorical critic 固定上界
- $\epsilon$ = 数值稳定常数

**为什么用 max 而不只用 $\sigma$？** Return distribution 偏态严重或有 outlier 时，仅用 std 会被 outlier 主导。max 项确保 normalized return 始终落入 $[-1, 1]$，刚好 fit 进 $[G_{\min}, G_{\max}]$ support。

#### B.5 Weight Normalization（post-step projection）

每个 gradient step 后，weight vector 投影到 unit-norm sphere：

$$W \leftarrow \frac{W}{\|W\|_2}$$

对 BN 参数：$\gamma, \beta \leftarrow \sqrt{d} \cdot \frac{(\gamma, \beta)}{\|(\gamma, \beta)\|_2}$

**为何 $\sqrt{d}$？** $\gamma, \beta$ 是 per-dimension，每维 norm=1 时整个 vector 的 $\ell_2$ norm = $\sqrt{d}$，与 He initialization 一致。

**作用**：强制 network 只通过 weight 方向（angle）编码信息，scale 锁死，阻止 bootstrapping 下 weight norm 无界增长——这是 critic divergence 的常见模式。参考 [Lyle NeurIPS 2024](https://arxiv.org/abs/2410.01755)、[van Laarhoven 2017](https://arxiv.org/abs/1706.05350)。

---

### C. Exploration：Unified Entropy + Noise Repetition

#### C.1 Unified Entropy Target（Equation 7）

标准 SAC 的 target entropy 通常设为 $-\dim(\mathcal{A})$（[Haarnoja 2018](https://arxiv.org/abs/1801.01290)），这是 ad-hoc 经验值，跨 embodiment 不可 scale。

FLASHSAC 用 action std parameterize：

$$\bar{\mathcal{H}} = \frac{1}{2}|\mathcal{A}|\log(2\pi e \sigma_{\text{tgt}}^2)$$

变量：
- $|\mathcal{A}|$ = action dimension
- $\sigma_{\text{tgt}}$ = target action std（FLASHSAC 全实验 = 0.15）
- $2\pi e$ = Gaussian 微分熵常数

**直觉**：固定 action std = 0.15 means "policy 应保持 ~15% max action range 的 stochasticity"，与具体 action dim 解耦，跨 embodiment 不用 re-tune。

#### C.2 Noise Repetition：pink noise 的便宜替代

pink noise ([Eberhard ICLR 2023](https://arxiv.org/abs/2207.06236)) 与 OU process 在 high-dim action space 探索效果好，但需要 per-environment correlated process state，在 1024 parallel envs 下 memory 与 compute overhead 显著。

FLASHSAC 的 Noise Repetition：
1. 每 interval 开始，采 $\epsilon \sim \mathcal{N}(0, I_{|\mathcal{A}|})$
2. 持续 $k$ 步，$k$ 从 Zeta 分布采：$P(k) \propto k^{-s}$
3. $k$ 步后重新采样

**频谱视角直觉**：i.i.d. white noise 是 flat PSD，被 dynamics 频率响应 roll-off 后高频部分被 filter 掉。Pink noise ($1/f$ spectrum) 低频 energy 多，能 drive 慢 dynamics。Noise Repetition 通过 Zeta 分布 heavy-tail 实现 power-law decay 的频谱近似，是 pink noise 的便宜替代。每 env 只需存一个 $\epsilon$ vector + 计数器。来自 [Dabney 2020](https://arxiv.org/abs/2006.01782)。

---

## IV. 实验：60+ tasks × 10 simulators 全景测试

### A. GPU-based 任务的关键结果（Figure 3）

**Setup**：
- 25 个 state-based task，来自 IsaacLab ([Mittal 2025](https://arxiv.org/abs/2511.04831))、ManiSkill3 ([Tao 2024](https://arxiv.org/abs/2410.00425))、Genesis ([2024](https://github.com/Genesis-Embodied-AI/Genesis))、MuJoCo Playground ([Zakka 2025](https://arxiv.org/abs/2502.08844))
- Low-dim: Franka manipulation, AnyMal/Go2 quadruped（15 tasks）
- High-dim: Shadow Hand/Allegro dexterous, G1/H1/T1 humanoid（10 tasks）
- FLASHSAC 训 50M env steps，PPO 训 200M（3× compute）

**结果**：
- Low-dim：FLASHSAC 与 PPO 持平
- **High-dim：FLASHSAC 显著优于 PPO**，wall-clock 快 ~3×，asymptotic return 更高
- FastTD3 在 Go2Walk、Franka Pull Cube 上频繁 diverge，FLASHSAC 全部稳定

### B. CPU-based 样本效率（Figure 4）

40 个单 env 任务，来自 MuJoCo ([Todorov 2012](https://arxiv.org/abs/1907.02057))、DMC ([Tassa 2018](https://arxiv.org/abs/1801.00690))、MyoSuite ([Caggiano 2022](https://arxiv.org/abs/2205.13600))、HumanoidBench ([Sferrazza 2024](https://arxiv.org/abs/2403.10506))

Baselines：PPO、XQC、SimbaV2、TD-MPC2 ([Hansen 2024](https://arxiv.org/abs/2406.16894))、MR.Q ([Fujimoto 2025](https://arxiv.org/abs/2501.16142))

CPU 设定 batch size 降到 512，UTD=1。

**结果**：FLASHSAC 一致匹配或超过所有 sample-efficient baselines。证明 stabilization 机制不依赖 massive parallelism。

### C. Vision-based（Figure 5）

8 个 DMC task。Baselines：DrQ-v2 ([Yarats 2021](https://arxiv.org/abs/2107.09645))、MR.Q。

设定：3-layer CNN encoder + 3-frame stacking (84×84×9) + 3-step return。

**结果**：FLASHSAC 在 DrQ-v2 不稳的 Finger Turn Hard 上稳定收敛，asymptotic 与 MR.Q 持平但 wall-clock 更快。

### D. Sim-to-Real（Figure 1c, 6）

**最 striking 的实验**：Unitree G1 humanoid 29-DoF，blind locomotion（无视觉，仅 proprioception）。

- **Flat ground**：FLASHSAC **20 min** vs PPO **3 hr**（~9× 加速）
- **15cm stair climbing**：FLASHSAC **4 hr** vs PPO **20 hr**（~5× 加速）

共用 sim-to-real pipeline：
- Terrain curriculum（10 levels，stair 0→23cm）+ domain randomization
- Implicit system identification via context estimator（[DreamWaQ, Nahrendra 2023](https://arxiv.org/abs/2301.10602)）
- Asymmetric actor-critic（critic 接收 contact state, height map 等 privileged info，[Pinto 2017](https://arxiv.org/abs/1710.06542)）

**意义**：off-policy RL 长期被认为 sim-to-real unreliable（[Mock 2023](https://books.google.co.kr/books?id=waUG0AEACAAJ)），FLASHSAC 用同一 stabilization 机制让 humanoid 在 4 小时内完成 stair climbing sim-to-real。

---

## V. Analysis：Ablation 与 Intuition

### A. Coverage 对比（Figure 7）

Shadow Hand cube reorientation 任务，训 1M steps 后再 roll out 1M on-policy。

2D density plot 显示：off-policy buffer 覆盖大片区域，on-policy rollout 集中在 final policy 的 narrow manifold。

**Intuition**：high-dim 下要达到相同 coverage，on-policy 需要指数级更多 samples。

### B. Scaling Ablation（Figure 8）

5 个 hyperparameter 单变量 ablation：

| Ablation | 观察 |
|---|---|
| Buffer 1M→10M→50M | 10M 最优；50M 训练慢（recent data 被稀释） |
| Batch 256→2048 | 单调加速 wall-clock |
| Width 256→1024 | 加速收敛 |
| Depth 1→6 block | 加速 |
| UTD 1→2/1024 | 极低 UTD 仍稳定 |

### C. Architectural Ablation（Figure 9）

从 standard MLP 逐项加：
1. Residual blocks
2. Pre-activation BN
3. Post RMSNorm
4. Distributional Critic + Reward Scaling
5. Weight Normalization

每加一项，**parameter norm, feature norm, gradient norm, condition number** 全部单调下降且 bounded。

condition number 监控来自 [XQC (Palenicek ICLR 2026)](https://arxiv.org/abs/2407.04811)：critic loss landscape 的 Hessian condition number，越小说明 optimization landscape 越良态。

### D. Exploration Ablation（Figure 10）

- $\sigma_{\text{tgt}} \in \{0.05, 0.1, 0.15, 0.2, 0.25\}$：performance robust，0.15 最佳
- Noise Repetition 开/关：开启后收敛显著加速

---

## VI. 把 Intuition 建起来：几个 mental model

### 1. Bootstrapping = 自我抄笔记

Critic 训练 target 依赖 critic 自己的 prediction，就像学生抄自己的笔记。若抄写过程中有随机错误，抄 10 遍后错误指数累积。

**Stabilization 的本质**：让"抄写过程"是 contractive mapping——每次抄写误差衰减而非放大。Contractive mapping 的 Lipschitz 常数 <1，关键在限制 weight norm 与 feature norm 上界。这就是为什么五重 norm 约束层层加码。

### 2. Scaling Law 在 RL 中的 transferability

[Kaplan 2020](https://arxiv.org/abs/2001.08361) 在 supervised learning 中证明大模型+大 batch+少 step 最优。RL 之前难以应用是因为 critic 的 self-reference 让大模型更脆弱。

**FLASHSAC 的贡献**：通过显式 norm constraint 让 contractive property 在大模型下仍成立，从而把 scaling law 的 compute efficiency 优势带到 RL。

### 3. Off-Policy Coverage 的几何视角

High-dim state-action space 中，on-policy rollout 是低维 manifold 上的轨迹采样，coverage volume 随 dimensionality 指数衰减。Off-policy buffer 累积多种 policy 轨迹，coverage 是 multi-manifold union。

### 4. Noise Repetition 的频谱视角

i.i.d. white noise 是 flat PSD，被 dynamics filter 掉。Pink noise ($1/f$) 低频 energy 多，能 drive 慢 dynamics。Noise Repetition 用 Zeta 分布 heavy-tail 实现近似 power-law 频谱，是 pink noise 在 parallel sim 中的便宜替代。

### 5. Distributional Critic + Reward Scaling 的耦合必要性

Categorical critic 的 support $[G_{\min}, G_{\max}]$ 固定，return scale 超出则 projection 产生 systematic bias。Reward Scaling 通过动态 normalize reward，保证 projected Bellman target 始终在 support 内。

---

## VII. Limitations 与 Open Questions

1. **Vision-based 仅测 DMC low-dim**：未在 RLBench、ManiSkill vision 任务验证；BatchNorm 在 visual encoder 中的稳定性可能不如 state encoder
2. **Contact-rich deformable tasks 未覆盖**：[DexGarmentLab (Wang 2025)](https://arxiv.org/abs/2505.11032)、[SoftGym (Lin 2021)](https://arxiv.org/abs/2011.07215) 等场景
3. **Sim-to-Real 仅 humanoid locomotion**：dexterous manipulation sim-to-real 仍待验证
4. **Weight Normalization 单独效果 modest**（Figure 9a），但 sample-limited regime 中保留——stabilization 在不同 regime 边际收益不同
5. **未与 offline-to-online RL** ([Ball ICML 2023](https://arxiv.org/abs/2306.09324)) 结合：作者 §VII 提及 support for demonstration mixture 是 future work

---

## VIII. 给 Robotics 社区的启示

**Insight 1**：Off-policy RL 的"慢"源于过拟合于 small model + high UTD 范式。stabilization 解决后，scaling law 允许 RL 像 LLM 一样走 "big model + big batch + low update" 路线，wall-clock efficiency 大幅提升。

**Insight 2**：Critic stability 不是单一 trick 能解决的，而是 weight/feature/gradient norm 的联合 constraint。每一项单独效果有限，但 stacked 起来让 condition number 单调下降，让 bootstrapping 的 contractive property 在大模型下保留。

**Insight 3**：Sim-to-real 瓶颈不在 on-policy vs off-policy 本身，而在 high-dim coverage 与 critic fidelity。Off-policy 用更大 buffer 提供 broader coverage；用 stable large critic 提供 accurate evaluation；两者结合让 humanoid sim-to-real 从 20 hr 压到 4 hr。

对 robotics 社区而言，FLASHSAC 提供了一个跨 embodiment、跨 simulator、跨 sample-efficiency regime 的统一 off-policy baseline。其设计 choices（inverted residual + pre-activation BN + cross-batch + distributional + reward scaling + weight norm + noise repetition）形成可复现 recipe，预计会成为后续 high-dim robot RL 工作的默认比较 baseline。

---

## IX. 参考 Web Links

1. **SAC**: [Haarnoja et al. 2018](https://arxiv.org/abs/1801.01290)
2. **PPO**: [Schulman et al. 2017](https://arxiv.org/abs/1707.06347)
3. **RSL-RL**: [Schwarke et al. 2025](https://arxiv.org/abs/2509.10771)
4. **CrossQ**: [Bhatt et al. ICLR 2024](https://arxiv.org/abs/2403.07818)
5. **Simba**: [Lee et al. 2024](https://arxiv.org/abs/2410.09754)
6. **SimbaV2**: [Lee et al. 2025](https://arxiv.org/abs/2502.15280)
7. **XQC**: [Palenicek et al. ICLR 2026](https://arxiv.org/abs/2407.04811)
8. **FastTD3**: [Seo et al. 2025](https://arxiv.org/abs/2505.22642)
9. **TD-MPC2**: [Hansen et al. 2024](https://arxiv.org/abs/2406.16894)
10. **MR.Q**: [Fujimoto et al. 2025](https://arxiv.org/abs/2501.16142)
11. **DrQ-v2**: [Yarats et al. 2021](https://arxiv.org/abs/2107.09645)
12. **Scaling laws for neural LMs**: [Kaplan et al. 2020](https://arxiv.org/abs/2001.08361)
13. **Chinchilla**: [Hoffmann et al. 2022](https://arxiv.org/abs/2206.00685)
14. **Deadly triad**: [van Hasselt et al. 2018](https://arxiv.org/abs/1812.02648)
15. **C51 distributional RL**: [Bellemare et al. 2017](https://arxiv.org/abs/1707.06870)
16. **RMSNorm**: [Zhang & Sennrich 2019](https://arxiv.org/abs/1910.07467)
17. **Inverted residual/MobileNet**: [Howard 2017](https://arxiv.org/abs/1704.04861)
18. **BatchNorm optimization smoothing**: [Santurkar et al. 2018](https://arxiv.org/abs/1805.11604)
19. **Pink noise exploration**: [Eberhard et al. ICLR 2023](https://arxiv.org/abs/2207.06236)
20. **DreamWaQ**: [Nahrendra et al. 2023](https://arxiv.org/abs/2301.10602)
21. **Asymmetric actor-critic**: [Pinto et al. 2017](https://arxiv.org/abs/1710.06542)
22. **Reward centering**: [Naik et al. 2024](https://arxiv.org/abs/2405.09999)
23. **Return-based scaling**: [Schaul et al. 2021](https://arxiv.org/abs/2105.05347)
24. **Lyle normalization**: [Lyle et al. NeurIPS 2024](https://arxiv.org/abs/2410.01755)
25. **Weight clipping for continual RL**: [Elsayed et al. 2024](https://arxiv.org/abs/2405.16158)
26. **IsaacLab**: [Mittal et al. 2025](https://arxiv.org/abs/2511.04831)
27. **MuJoCo Playground**: [Zakka et al. 2025](https://arxiv.org/abs/2502.08844)
28. **ManiSkill3**: [Tao et al. 2024](https://arxiv.org/abs/2410.00425)
29. **Genesis**: [Genesis Authors 2024](https://github.com/Genesis-Embodied-AI/Genesis)
30. **HumanoidBench**: [Sferrazza et al. 2024](https://arxiv.org/abs/2403.10506)
31. **MyoSuite**: [Caggiano et al. 2022](https://arxiv.org/abs/2205.13600)
32. **DeepMind Control Suite**: [Tassa et al. 2018](https://arxiv.org/abs/1801.00690)
33. **Mixed precision training**: [Micikevicius et al. 2017](https://arxiv.org/abs/1710.03740)
34. **Revisiting experience replay**: [Fedus et al. 2020](https://proceedings.mlr.press/v119/fedus20a.html)
35. **Unitree G1**: https://www.unitree.com/g1/

---

# FLASHSAC: Fast and Stable Off-Policy RL for High-Dimensional Robot Control — 详细技术讲解

## I. 核心动机：为何 Off-Policy RL 在 high-dimensional robotics 中长期被冷落

这篇 paper 的核心 thesis 可以提炼为一句话：**off-policy RL 在 high-dimensional robot control 上的"慢"与"不稳"并非本质缺陷，而是源于 critic 在 bootstrapping 下 error accumulation 与 model capacity 之间的张力未解**。

具体观察到的困境链条如下：

1. **On-policy methods（如 PPO）的 sweet spot 在 shrinking**：当 state-action space 低维（quadruped locomotion, gripper manipulation），simulator throughput 极高，PPO 丢弃数据的代价可以忍受。但当任务维度上升（humanoid 29-DoF、dexterous hand 24-DoF、vision-based 84×84×9），on-policy rollout 的 state-action coverage 急剧稀疏化，policy evaluation 失准。
2. **Off-policy methods 的 promise 与 pitfall**：replay buffer 的 diverse data 理论上能解决 coverage 问题，但 Bellman target $y = r + \gamma Q_{\bar{\phi}}(s', a')$ 中 $Q$ 自我引用，approximation error 与 extrapolation error 通过 bootstrap 累积（参考 [van Hasselt et al., 2018, "deadly triad"](https://arxiv.org/abs/1812.02648)）。
3. **Capacity vs. stability tradeoff**：scaling laws（[Kaplan et al. 2020](https://arxiv.org/abs/2001.08361)）启示 "bigger model + bigger batch + fewer updates" 在 supervised learning 中更快收敛；但 RL 中 bigger critic 在 bootstrapping 下 error amplification 更严重，这是 FastTD3/FastSAC（[Seo et al. 2025](https://arxiv.org/abs/2505.22642)）只能用 ~0.2M 参数的根因。

FLASHSAC 的核心贡献：**通过显式约束 weight/feature/gradient norms，使得 2.5M 参数的大 critic 在 UTD=2/1024 这种极端低更新频率下仍能稳定收敛**，从而把 supervised learning 的 scaling law 移植到 off-policy RL。

---

## II. 方法论拆解：Fast + Stable + Exploration 三层设计

### A. Fast Training：少更新、大模型、大 buffer、大 batch

#### A.1 Update-to-Data Ratio (UTD) 的反直觉选择

FLASHSAC 在 GPU-based simulator 设定下使用 **UTD = 2/1024**，即每采集 1024 条新 transition 只做 2 次 gradient step。这与标准 SAC（UTD=1）、RedQ（UTD=20）、TD3（UTD=1）的设定截然相反。

变量解释：
- **UTD (Update-to-Data ratio)** = gradient steps per environment step，记作 $n_{\text{update}} / n_{\text{env}}$
- 在 FLASHSAC 中 $n_{\text{env}} = 1024$（parallel envs），$n_{\text{update}} = 2$，故 effective UTD = $2/1024 \approx 0.002$

直觉上 UTD 越低 critic 越欠拟合，但作者证明：**当 batch size 与 model capacity 足够大时，单次 SGD step 的 signal-to-noise ratio 大幅提升，少量 step 即可在 GPU 上 saturate compute**。这与 [Hoffmann et al. 2022 Chinchilla](https://arxiv.org/abs/2206.00685) 中 "compute-optimal" 思路同构：在固定 FLOP budget 下，大 batch + 少 step > 小 batch + 多 step。

#### A.2 Replay Buffer 规模

| 配置 | 标准 SAC | FastTD3 | FLASHSAC |
|---|---|---|---|
| Buffer size | 1M | 1M | **10M** |
| Network params | 0.2M-0.5M | 0.2M | **2.5M** |
| Batch size | 256 | 1024 | **2048** |
| Layers | 2-3 | 3 | **6** |
| UTD | 1 | 1 | **2/1024** |

10M buffer 的设计动机：high-dim task 中 rare 但 critical 的 state-action pair 在小 buffer 中被 overwrite（catastrophic forgetting），导致 extrapolation error。这与 [Fedus et al. 2020 "Revisiting Fundamentals of Experience Replay"](https://proceedings.mlr.press/v119/fedus20a.html) 的 analysis 一致。

#### A.3 Massively Parallel Simulation 与 JIT

- **1024 parallel environments** 通过 GPU-based simulators（IsaacLab, ManiSkill3, Genesis, MuJoCo Playground）收集 diverse trajectories。
- **JIT compilation + mixed precision**（[Micikevicius et al. 2017](https://arxiv.org/abs/1710.03740)）：减少 Python overhead 5-10% wall-clock。

---

### B. Stable Training：五重 constraint 抑制 critic error amplification

这是 paper 最核心的技术贡献。下面逐一拆解每个机制的公式与作用。

#### B.1 Inverted Residual Backbone（Figure 2）

架构灵感来自 Transformer feedforward block（[Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)）与 MobileNet inverted bottleneck（[Howard 2017](https://arxiv.org/abs/1704.04861)）。

每个 block 的数据流：
1. Input $x \in \mathbb{R}^d$（$d$ = hidden dim，FLASHSAC 中 = 1024）
2. Inverted bottleneck expansion: $x' = W_{\text{up}} x \in \mathbb{R}^{4d}$（expand 4×）
3. 非线性：$h = \text{GELU/SiLU}(x')$
4. Projection: $y = W_{\text{down}} h \in \mathbb{R}^d$
5. Residual：$x_{\text{out}} = x + y$
6. 6 层堆叠

**为何 inverted residual？** 标准 residual block（如 ResNet）先降维后升维以省 FLOPs；inverted residual 反过来，先升维后降维，目的是在 high-dim 中保留更多 feature expressiveness，同时让 weight norm 更 controllable（因 $W_{\text{down}}$ 起到收缩作用）。

最后一个 block 之后加 **RMSNorm**（[Zhang & Sennrich 2019](https://arxiv.org/abs/1910.07467)）：
$$\text{RMSNorm}(x_i) = \frac{x_i}{\sqrt{\frac{1}{d}\sum_j x_j^2 + \epsilon}} \cdot g_i$$
其中 $g_i$ 是 learnable scale，$\epsilon$ 防除零。

**作用**：bound per-sample feature norm before value head，防止 OOD state 导致 unbounded activation 进入 bootstrapping target。

#### B.2 Pre-activation Batch Normalization

这是与 [CrossQ (Bhatt et al. 2024 ICLR)](https://arxiv.org/abs/2403.07818) 一脉相承的关键设计。在每一层 nonlinearity 之前插入 BatchNorm，而非 LayerNorm。

公式（标准 BatchNorm）：
$$\hat{\mu}_B = \frac{1}{m}\sum_{i=1}^m x_i, \quad \hat{\sigma}_B^2 = \frac{1}{m}\sum_i (x_i - \hat{\mu}_B)^2$$
$$\hat{x}_i = \gamma \cdot \frac{x_i - \hat{\mu}_B}{\sqrt{\hat{\sigma}_B^2 + \epsilon}} + \beta$$

变量：
- $m$ = batch size（FLASHSAC = 2048）
- $\gamma, \beta \in \mathbb{R}^d$ = learnable scale 与 shift
- $\hat{\mu}_B, \hat{\sigma}_B^2$ = batch 统计

**为何选 BatchNorm 而非 LayerNorm？** 作者给出两点理由：
1. **大 batch 的统计稳定性**：2048 batch 来自 diverse replay data，$\hat{\mu}_B, \hat{\sigma}_B^2$ 的 variance 极低，提供更平滑的 loss landscape。这与 [Santurkar et al. 2018](https://arxiv.org/abs/1805.11604) "How Does Batch Normalization Help Optimization" 一致——BatchNorm 的真正 benefit 不是 ICS，而是 reparametrize loss 使其 Lipschitz 常数更小、condition number 更低。
2. **LayerNorm 在 RL 中的局限**：per-sample normalization 会丢失 cross-sample 信息，对 replay 中的 diverse distribution 不友好。

#### B.3 Cross-Batch Value Prediction

这是 BatchNorm 在 RL 中的经典陷阱解决方案（[CrossQ](https://arxiv.org/abs/2403.07818)）。

**问题**：Bellman target $y = r + \gamma Q_{\bar\phi}(s', a')$ 中，$Q_\phi(s, a)$ 与 $Q_{\bar\phi}(s', a')$ 在两次独立 forward pass 中计算，各自有不同 batch 统计 $\hat\mu_B, \hat\mu_{B'}$，导致同一 state 在不同上下文下被不同地 normalize，引入 systematic bias。

**解法**：将 $(s, a, r, s', a')$ 拼成 single batch，让 $Q_\phi(s,a)$ 与 $Q_{\bar\phi}(s',a')$ 共享同一组 BN 统计。

直觉：把 Bellman backup 视作一个 self-distillation 过程，teacher 与 student 必须在同一个 normalization frame 下比较。

#### B.4 Distributional Critic with Adaptive Reward Scaling

Q-value 表示为 categorical distribution over $n_{\text{atom}}$ atoms 均匀分布在 $[G_{\min}, G_{\max}]$（[Bellemare et al. 2017](https://arxiv.org/abs/1707.06870) C51）。

- $n_{\text{atom}}$ = atom 数（典型 51）
- $G_{\min}, G_{\max}$ = return 范围
- 网络输出 softmax 分布 $p_\theta(\cdot | s, a) \in \Delta^{n_{\text{atom}}}$
- 期望 Q 值：$Q_\theta(s,a) = \sum_{i=1}^{n_{\text{atom}}} p_i z_i$，其中 $z_i = G_{\min} + \frac{i-1}{n_{\text{atom}}-1}(G_{\max}-G_{\min})$
- 训练用 cross-entropy loss 而非 MSE，平滑优化 landscape

**Adaptive Reward Scaling 公式**（Equation 6）：

$$\bar{r}_t = \frac{r_t}{\max\left(\sqrt{\sigma_{t,G}^2 + \epsilon}, \; G_{t,\max}/G_{\max}\right)}$$

变量含义：
- $r_t$ = 原始 reward at time $t$
- $\bar{r}_t$ = normalized reward
- $\sigma_{t,G}^2$ = running estimate of discounted return variance，通过 EMA 跟踪
- $G_{t,\max}$ = running estimate of maximum return magnitude（observed so far）
- $G_{\max}$ = categorical critic 的上界 atom 位置（固定）
- $\epsilon$ = numerical stability constant

**为何 max 而非单纯用 $\sigma$？** 当 return distribution 偏态严重或存在极端 outlier 时，仅用 std 会导致 normalized reward 被 outlier 主导。max 项确保 normalized return 始终落入 $[-1, 1]$ 一致 scale，从而 fit 进 $[G_{\min}, G_{\max}]$ 的 categorical support。

#### B.5 Weight Normalization（post-step projection）

每个 gradient step 之后，将 weight vector 投影到 unit-norm sphere（[van Laarhoven 2017](https://arxiv.org/abs/1706.05350); [Lyle et al. 2024 NeurIPS](https://arxiv.org/abs/2410.01755)）：

$$W \leftarrow \frac{W}{\|W\|_2}$$
对 normalization 参数：$\gamma, \beta \leftarrow \sqrt{d} \cdot \frac{(\gamma, \beta)}{\|(\gamma, \beta)\|_2}$

为何 $\sqrt{d}$？因为 $\gamma, \beta$ 是 per-dimension，若每维 norm 为 1，则整个 vector 的 $\ell_2$ norm 为 $\sqrt{d}$，保持与标准初始化（如 He init）一致 scale。

**作用**：强制 network 只通过 weight 的方向（angle）编码信息，scale 被锁死，阻止在 bootstrapping 下 weight norm 无界增长（这是 RL 中 critic divergence 的常见模式，见 [Lyle et al.](https://arxiv.org/abs/2410.01755)）。

---

### C. Exploration：Unified Entropy + Noise Repetition

#### C.1 Unified Entropy Target（Equation 7）

标准 SAC 的 target entropy 通常设为 $-\dim(\mathcal{A})$（如 [Haarnoja 2018](https://arxiv.org/abs/1801.01290)），但这是 ad-hoc 经验值，不能跨 embodiment（quadruped 12-DoF vs. humanoid 29-DoF）scale。

FLASHSAC 用 action std 来 parameterize：

$$\bar{\mathcal{H}} = \frac{1}{2}|\mathcal{A}|\log(2\pi e \sigma_{\text{tgt}}^2)$$

变量：
- $|\mathcal{A}|$ = action dimension
- $\sigma_{\text{tgt}}$ = target action std（FLASHSAC 全实验 = 0.15）
- $\bar{\mathcal{H}}$ = target entropy
- $2\pi e$ = Gaussian differential entropy constant

推导：对角 Gaussian $\pi(a|s) = \mathcal{N}(\mu(s), \text{diag}(\sigma^2))$ 的微分熵为 $\frac{1}{2}|\mathcal{A}|\log(2\pi e \sigma^2)$，将 $\sigma$ 替换为 $\sigma_{\text{tgt}}$ 即得 target。

直觉：固定 action std = 0.15 means "policy 应该保持 ~15% max action 范围的 stochasticity"。这与具体 action dimension 解耦，跨任务 transfer 时无需重新 tune。

#### C.2 Noise Repetition（lightweight pink noise 替代）

Ornstein-Uhlenbeck 与 pink noise（[Eberhard et al. 2023 ICLR](https://arxiv.org/abs/2207.06236)）需要 per-environment correlated process state，在 1024 parallel envs 下 memory 与 compute overhead 显著。

FLASHSAC 的 Noise Repetition：
1. 每个 interval 开始，采样 $\epsilon \sim \mathcal{N}(0, I_{|\mathcal{A}|})$
2. 持续 $k$ 步，$k$ 从 Zeta 分布采样：$P(k) \propto k^{-s}$
3. $k$ 步后重新采样

Zeta 分布（[Dabney et al. 2020](https://arxiv.org/abs/2006.01782)）参数 $s$ 控制 heavy-tail：$s$ 越大，short repeat 越多；$s$ 越小，偶尔长 correlated sequence 越多。

**直觉**：在 high-dim action space 中，i.i.d. Gaussian noise 被 dynamics 平均掉（action 维度高时，单步噪声方差被积分），需要 temporal coherence 让 perturbation 在 trajectory 上累积。Noise Repetition 用极低 memory（每 env 仅一个 $\epsilon$ vector + 计数器）实现近似 pink noise 的 power spectrum。

---

## III. 实验：60+ tasks × 10 simulators 全景测试

### A. 任务分组与 baseline 对比

| Task Category | State dim | Action dim | Simulator | Representative tasks |
|---|---|---|---|---|
| Low-dim state | ~50-100 | 4-12 | IsaacLab, ManiSkill3, Genesis, MuJoCo Playground | Franka manipulation, AnyMal/Go2 walk |
| High-dim state | ~200-500 | 20-30 | IsaacLab, MuJoCo Playground | Shadow Hand cube reorientation, Allegro manipulation, G1/H1/T1 humanoid locomotion |
| CPU single-env | varies | varies | MuJoCo, DMC, MyoSuite, HumanoidBench | 40 tasks total |
| Vision-based | 84×84×9 | low | DMC | Finger Turn Hard, Pendulum, etc. |
| Sim-to-Real | 29-DoF humanoid | 29 | IsaacLab → Unitree G1 real | Flat ground + 15cm stair climbing |

### B. GPU-based 高维任务的关键结果（Figure 3b）

- **PPO 训练 200M env steps** vs. **FLASHSAC 50M env steps**
- FLASHSAC 在 humanoid locomotion 与 dexterous manipulation 上 wall-clock 快 ~3×，asymptotic return 更高
- FastTD3 在 Go2Walk、Franka Pull Cube 上 frequently diverge，FLASHSAC 全部稳定收敛

### C. CPU-based 样本效率（Figure 4）

baseline 包括 [XQC](https://arxiv.org/abs/2407.04811)、[SimbaV2](https://arxiv.org/abs/2502.15280)、[TD-MPC2](https://arxiv.org/abs/2406.16894)、[MR.Q](https://arxiv.org/abs/2501.16142)。

CPU 设定下 batch size 降到 512，UTD = 1（单 env 数据流慢）。

FLASHSAC 在 sample-efficient regime 下匹配或超过 XQC、SimbaV2 等 sample-efficient baselines，证明 stabilization 机制不依赖 massive parallelism。

### D. Vision-based（Figure 5）

baseline：[DrQ-v2](https://arxiv.org/abs/2107.09645)、MR.Q。

视觉设定下：
- 3-layer CNN encoder + linear bottleneck
- 3-frame stacking (84×84×9)
- 3-step return for credit assignment

FLASHSAC 在 Finger Turn Hard 等 DrQ-v2 不稳定任务上稳定收敛，asymptotic 与 MR.Q 持平但 wall-clock 更快（MR.Q 额外 dynamics model 推理开销大）。

### E. Sim-to-Real（Figure 1c, Figure 6）

**Unitree G1 humanoid**，29-DoF，blind locomotion（无视觉，仅 proprioception）。

- **Flat ground**：FLASHSAC 20 min vs. PPO 3 hr（~9× 速度提升）
- **15cm stair climbing**：FLASHSAC 4 hr vs. PPO 20 hr（~5× 速度提升）

Sim-to-real pipeline 共用：
- 地形 curriculum（10 levels，stair 高度 0→23cm）+ domain randomization
- Implicit system identification via context estimator（[DreamWaQ, Nahrendra et al. 2023](https://arxiv.org/abs/2301.10602)）
- Asymmetric actor-critic（critic 接收 privileged info：contact state, height map）[Pinto et al. 2017](https://arxiv.org/abs/1710.06542)

这是 paper 最 striking 的实验：off-policy RL 长期被认为 sim-to-real unreliable（[Mock 2023](https://books.google.co.kr/books?id=waUG0AEACAAJ)），FLASHSAC 用同一 stabilization 机制让 humanoid 在数小时 sim 内完成 stair climbing，跨 sim-to-real gap 仍稳定。

---

## IV. Analysis：Ablation 与 Insight

### A. Off-Policy vs. On-Policy Coverage（Figure 7）

IsaacLab Shadow Hand cube reorientation 任务，训 1M steps 后再 roll out 1M on-policy。

2D density plot 显示：off-policy buffer 覆盖 state-action 大量区域，on-policy 数据严重集中在 final policy 的 narrow manifold。

**Intuition**：高维下，要达到相同 coverage，on-policy 需要指数级更多 samples。这是 PPO 在 dexterous manipulation 上失败的根因，而非 algorithm 不行，而是 coverage insufficient。

### B. Scaling Ablation（Figure 8）

5 个 hyperparameter 单变量 ablation：

| Ablation | 观察 |
|---|---|
| Buffer size 1M→10M→50M | 10M 最优；50M 训练慢（recent data 被稀释）但 asymptotic 略高 |
| Batch size 256→512→1024→2048 | 单调加速 wall-clock，符合 scaling law |
| Network width 256→512→1024 | 加速收敛 |
| Network depth 1→3→6 block | 加速 |
| UTD 1→2/1024 | 极低 UTD 仍稳定，证明 stabilization 机制成功 |

### C. Architectural Ablation（Figure 9）

从 standard MLP 逐项加入：
1. Residual blocks
2. Batch Normalization (pre-activation)
3. Post RMSNorm
4. Distributional Critic + Reward Scaling
5. Weight Normalization

每加一项，**parameter norm, feature norm, gradient norm, condition number** 全部单调下降且 bounded。

condition number 监控来自 [XQC](https://arxiv.org/abs/2407.04811)：critic loss landscape 的 Hessian condition number，越小说明 optimization landscape 越良态，gradient update 越稳定。

### D. Exploration Ablation（Figure 10）

- $\sigma_{\text{tgt}} \in \{0.05, 0.1, 0.15, 0.2, 0.25\}$：performance 对该值 robust，统一 0.15 表现最佳
- Noise Repetition 开/关：开启后收敛显著加速

---

## V. 关键 Takeaway 与 Intuition 构建

### 1. Bootstrapping 下的 Error Amplification 类比传染病学

可以把 critic 视作一个递归预测系统：$Q_{t+1} = f(Q_t)$。若 $f$ 在某些 region 是 expansive mapping（local Lipschitz 常数 >1），error 会指数增长。Stabilization 等于强制 $f$ 是 contractive mapping，关键在限制 weight norm 与 feature norm 上界。

### 2. Scaling Law 在 RL 中的 transferability

[Kaplan 2020](https://arxiv.org/abs/2001.08361) 在 supervised learning 中证明大模型 + 大 batch + 少 step 最优。RL 中难以直接应用的原因是 critic 的 self-reference 让大模型更脆弱。FLASHSAC 的贡献是**通过显式 norm constraint 让 contractive property 在大模型下仍成立**，从而把 scaling law 的 "compute efficiency" 优势带到 RL。

### 3. Off-Policy Coverage 优势的几何视角

在 high-dim state-action space 中，on-policy rollout 是低维 manifold 上的轨迹采样，coverage volume 随 dimensionality 指数衰减。Off-policy buffer 累积多种 policy 的轨迹，coverage 是 multi-manifold union。Figure 7 直观显示这一点。

### 4. Noise Repetition 的频谱视角

i.i.d. white noise 的 power spectral density 是 flat，在 dynamics 频率响应 roll-off 后，高频 noise 被 filter 掉。Pink noise ($1/f$ spectrum) 与 OU process 在低频有更多 energy，能 drive 慢 dynamics。Noise Repetition 通过 Zeta 分布 heavy-tail 实现 power-law decay 的频谱近似，是 pink noise 在 parallel sim 中的 computationally cheap 替代。

### 5. Distributional Critic + Reward Scaling 的耦合必要性

Categorical critic 的 support $[G_{\min}, G_{\max}]$ 固定，若 return scale 超出则 projection 产生系统性 bias。Reward Scaling 通过动态 normalize reward，保证 projected Bellman target 始终在 support 内，否则 distributional critic 会比 pointwise critic 更不稳。

### 6. 与 Concurrent 工作的对比

- **CrossQ ([Bhatt et al. ICLR 2024](https://arxiv.org/abs/2403.07818))**：首提 cross-batch BN + large batch + low UTD，FLASHSAC 在此基础上加 distributional critic、weight normalization、noise repetition，并扩展到 massively parallel sim
- **Simba/SimbaV2 ([Lee et al. 2024/2025](https://arxiv.org/abs/2410.09754))**：inverted residual + hyperspherical normalization，FLASHSAC 沿用此 backbone
- **XQC ([Palenicek et al. ICLR 2026](https://arxiv.org/abs/2407.04811))**：condition number minimization，FLASHSAC ablation 借用其 metric
- **FastTD3/FastSAC ([Seo et al. 2025](https://arxiv.org/abs/2505.22642))**：wall-clock 优化但小网络，FLASHSAC 解决了其 asymptotic limitation

---

## VI. Limitations 与 Open Questions

1. **Vision-based 仅测 DMC low-dim**：未在 RLBench、ManiSkill vision 任务上验证；BatchNorm 在 visual encoder 中的稳定性可能不如 state encoder
2. **Contact-rich deformable tasks 未覆盖**：[DexGarmentLab](https://arxiv.org/abs/2505.11032)、[SoftGym](https://arxiv.org/abs/2011.07215) 等场景
3. **Sim-to-Real 仅 humanoid locomotion**：dexterous manipulation sim-to-real 仍待验证
4. **Weight Normalization 单独效果 modest**（Figure 9a 最后一项），但在 sample-limited CPU regime 中保留——暗示 stabilization 在不同 regime 的边际收益不同
5. **未与 offline-to-online RL**（[Ball et al. ICML 2023](https://arxiv.org/abs/2306.09324)）结合：作者在 §VII 提及 support for demonstration mixture 是 future work

---

## VII. 参考 Web Links

1. **FLASHSAC paper**: 暂未公开 arXiv ID（2025 paper）；作者主页 [Hojoon Lee (Holiday Robotics)](https://hojoon-lee.github.io/)
2. **SAC**: [Haarnoja et al. 2018](https://arxiv.org/abs/1801.01290)
3. **CrossQ**: [Bhatt et al. ICLR 2024](https://arxiv.org/abs/2403.07818)
4. **Simba/SimbaV2**: [Lee et al. 2024](https://arxiv.org/abs/2410.09754), [SimbaV2 2025](https://arxiv.org/abs/2502.15280)
5. **XQC**: [Palenicek et al. ICLR 2026](https://arxiv.org/abs/2407.04811)
6. **FastTD3**: [Seo et al. 2025](https://arxiv.org/abs/2505.22642)
7. **TD-MPC2**: [Hansen et al. 2024](https://arxiv.org/abs/2406.16894)
8. **MR.Q**: [Fujimoto et al. 2025](https://arxiv.org/abs/2501.16142)
9. **DrQ-v2**: [Yarats et al. 2021](https://arxiv.org/abs/2107.09645)
10. **Scaling laws for neural LMs**: [Kaplan et al. 2020](https://arxiv.org/abs/2001.08361)
11. **Chinchilla**: [Hoffmann et al. 2022](https://arxiv.org/abs/2206.00685)
12. **Deadly triad**: [van Hasselt et al. 2018](https://arxiv.org/abs/1812.02648)
13. **C51 distributional RL**: [Bellemare et al. 2017](https://arxiv.org/abs/1707.06870)
14. **RMSNorm**: [Zhang & Sennrich 2019](https://arxiv.org/abs/1910.07467)
15. **Inverted residual/MobileNet**: [Howard 2017](https://arxiv.org/abs/1704.04861)
16. **BatchNorm optimization smoothing**: [Santurkar et al. 2018](https://arxiv.org/abs/1805.11604)
17. **Pink noise exploration**: [Eberhard et al. ICLR 2023](https://arxiv.org/abs/2207.06236) (paper 中标为 [14])
18. **DreamWaQ**: [Nahrendra et al. 2023](https://arxiv.org/abs/2301.10602)
19. **Asymmetric actor-critic**: [Pinto et al. 2017](https://arxiv.org/abs/1710.06542)
20. **Reward centering**: [Naik et al. 2024](https://arxiv.org/abs/2405.09999)
21. **Return-based scaling**: [Schaul et al. 2021](https://arxiv.org/abs/2105.05347)
22. **Lyle normalization**: [Lyle et al. NeurIPS 2024](https://arxiv.org/abs/2410.01755)
23. **Weight clipping for continual RL**: [Elsayed et al. 2024](https://arxiv.org/abs/2405.16158)
24. **IsaacLab**: [Mittal et al. 2025](https://arxiv.org/abs/2511.04831)
25. **MuJoCo Playground**: [Zakka et al. 2025](https://arxiv.org/abs/2502.08844)
26. **ManiSkill3**: [Tao et al. 2024](https://arxiv.org/abs/2410.00425)
27. **Genesis**: [Genesis Authors 2024](https://github.com/Genesis-Embodied-AI/Genesis)
28. **HumanoidBench**: [Sferrazza et al. 2024](https://arxiv.org/abs/2403.10506)
29. **MyoSuite**: [Caggiano et al. 2022](https://arxiv.org/abs/2205.13600)
30. **DeepMind Control Suite**: [Tassa et al. 2018](https://arxiv.org/abs/1801.00690)
31. **Mixed precision training**: [Micikevicius et al. 2017](https://arxiv.org/abs/1710.03740)
32. **Revisiting experience replay**: [Fedus et al. 2020](https://proceedings.mlr.press/v119/fedus20a.html)
33. **PPO**: [Schulman et al. 2017](https://arxiv.org/abs/1707.06347)
34. **RSL-RL**: [Schwarke et al. 2025](https://arxiv.org/abs/2509.10771)
35. **Unitree G1**: https://www.unitree.com/g1/

---

## VIII. 总结：FLASHSAC 给 RL 社区的启示

FLASHSAC 的核心贡献可以压缩为三个 insight：

**Insight 1**：Off-policy RL 的"慢"源于过拟合于 small model + high UTD 范式。一旦 stabilization 解决，scaling law 允许 RL 像 LLM 一样走 "big model + big batch + low update" 路线，wall-clock efficiency 大幅提升。

**Insight 2**：Critic stability 不是单一 trick 能解决的，而是 weight/feature/gradient norm 的联合 constraint。每一项单独效果有限（Figure 9 ablation 证实），但 stacked 起来让 condition number 单调下降，让 bootstrapping 的 contractive property 在大模型下保留。

**Insight 3**：Sim-to-real 的瓶颈不在 on-policy vs. off-policy 本身，而在 high-dim coverage 与 critic fidelity。Off-policy 用更大 buffer 提供 broader coverage；用 stable large critic 提供 accurate evaluation；两者结合让 humanoid locomotion 的 sim-to-real 从 20 hr 压缩到 4 hr。

对 robotics 社区而言，这篇 paper 的 methodological 价值是**提供了一个跨 embodiment、跨 simulator、跨 sample-efficiency regime 的统一 off-policy baseline**，其设计 choices（inverted residual + pre-activation BN + cross-batch + distributional + reward scaling + weight norm + noise repetition）形成可复现的 recipe，预计会成为后续 high-dim robot RL 工作的默认比较 baseline。
