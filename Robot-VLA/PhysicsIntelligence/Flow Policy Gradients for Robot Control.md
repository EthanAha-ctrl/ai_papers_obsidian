---
source_pdf: Flow Policy Gradients for Robot Control.pdf
paper_sha256: 4d139cfb248a7c863431594c40c99326efe03bf12ec3f7399a48d5b28b40cbdf
processed_at: '2026-08-04T09:36:47-07:00'
target_folder: Robot-VLA/PhysicsIntelligence
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

这篇paper的核心其实就一句话：**大家以前训robot control policy都被Gaussian卡死了，想用更牛逼的flow policy但算不出likelihood，这篇paper告诉你怎么绕过likelihood直接算，还能成功sim-to-real跑在真机上。**

我来按人话拆解一下逻辑：

### 1. 以前的痛点：Gaussian 太弱，Flow 算不动

以前RL训机器人，全用Gaussian分布输出action。Gaussian最大的问题是：各个维度是独立的。比如四足机器人走路，左腿和右腿的动作其实是有强相关的（trot步态），Gaussian很难表达这种耦合关系，导致探索效率低，甚至走出很丑的pronk步态（四条腿一起跳）。

那换成能表达复杂多模态分布的flow/diffusion policy行不行？在模仿学习里早就行了（比如π0）。但一旦要用RL的policy gradient去训，就抓瞎了。因为PPO这类算法要求算action的likelihood $\pi_\theta(a_t|o_t)$，而在flow model里算这个likelihood得积分divergence，计算量极大，在RL每步都要更新的高频场景下根本跑不动。

### 2. 前人尝试（FPO）的坑

之前有个叫FPO的方法，说不用算精确likelihood，用CFM loss的差值来**近似**likelihood ratio。公式大概长这样：$\exp(\text{Loss}_{\text{old}} - \text{Loss}_{\text{new}})$。Loss降了，说明policy觉得这个action更可能，隐含likelihood就升了。

听起来很美，但真跑到真机任务上（高维度action空间、真实的力矩限制），FPO极其不稳定，动就collapse。

### 3. 这篇paper的补丁：FPO++

作者加了两个简单但救命的改进：

**改进一：Per-sample ratio（精细化打击）**
FPO原来对一个action采好几个noise sample，先把这些sample的loss差值求平均，再算ratio，最后clip。这会导致：如果平均后超出了trust region，整个action的所有梯度全被clip掉，信息全丢了。
FPO++改成了：每个noise sample独立算ratio，独立clip。这样即便某些sample的ratio爆了，其他sample还能继续提供梯度。这相当于变相增大了batch size，降低了梯度方差。

**改进二：ASPO（不对称信任区）**
作者发现，对于negative advantage（要降低某个action概率）的情况，如果用普通PPO直接clip，梯度很容易导致entropy collapse（策略变成确定性策略，失去探索能力）。
所以他们对negative advantage用了SPO（带二次惩罚的信任区）。如果ratio偏离太远，它不会像PPO那样把梯度直接砍成0，而是给一个拉回来的力。这就保住了entropy，让flow policy能一直保持广泛的探索能力。

### 4. 推理时的作弊小技巧：Zero-sampling

训练时需要从高斯分布采noise来探索，但测试时直接把noise设成 $\epsilon=0$。
直觉上，高斯分布在原点密度最高，你从原点出发走Euler integration，自然会落到learned distribution的mode上。
这个小trick不仅大幅提升了成功率，还让真机部署时的Euler step从50步减到5步都没问题，极大降低了真机延迟。

### 5. 结果有多炸裂？

这是首次有flow policy不用专家蒸馏、只用RL训练就能直接sim-to-real跑到humanoid（Booster T1和Unitree G1）上。
不仅稳，还比传统的Gaussian PPO**sample efficiency更好**。因为flow policy能探索出correlated action，同样的reward，FPO++跑出了漂亮的trot步态，而Gaussian PPO跑成了笨拙的pronk步态。
而且拿去fine-tune用BC预训练的manipulation policy，FPO++也是收敛最快、最稳的。

### 6. 为什么这对未来很重要？

这相当于一个existence proof：**做policy gradient根本不需要精确的likelihood**。
只要你能搞出一个能近似likelihood ratio的surrogate，就能训任意复杂的policy representation。这直接打开了RL的design space，以后energy-based model、VAE甚至是implicit neural representation，只要能写出类似surrogate的，全都能塞进RL里跑了。

### Reference
- Paper: https://hongsukchoi.github.io/fpo-control
- FPO original: https://arxiv.org/abs/2507.21053

---

# Flow Policy Gradients for Robot Control 深度解析

## 核心motivation：为何要绕过likelihood

传统policy gradient方法（PPO等）依赖action的differentiable likelihood $\pi_\theta(a_t|o_t)$。这要求policy输出必须是参数化分布（通常diagonal Gaussian），从而**限制了表达力**。Gaussian的各维度独立采样，无法表达action dimension之间的correlation，这对locomotion这种需要symmetric gait的任务是致命的。

Flow/diffusion policy能表达任意复杂的多模态分布，在imitation learning中已经成功（如π0 [11], Diffusion Policy [10]）。但用RL训练flow policy面临核心障碍：计算flow policy的likelihood $\pi_\theta(a_t|o_t)$ 需要积分divergence of the flow field（参考FFJORD [13]、superposition estimator [12]），在online RL的high-frequency update场景下computationally prohibitive。

FPO [1] 的核心insight：**不需要精确likelihood，只需要likelihood ratio** $\rho_\theta = \pi_\theta(a_t|o_t)/\pi_{\theta_{\mathrm{old}}}(a_t|o_t)$，而这个ratio可以用CFM loss差异来近似。

---

## Background：FPO的数学结构

### Likelihood ratio的CFM surrogate

公式3给出了核心approximation：

$$\hat{\rho}_{\mathrm{FPO}}(\theta) = \exp\left(\hat{\mathcal{L}}_{\mathrm{CFM},\theta_{\mathrm{old}}}(a_t;o_t) - \hat{\mathcal{L}}_{\mathrm{CFM},\theta}(a_t;o_t)\right)$$

**变量解释**：
- $\hat{\rho}_{\mathrm{FPO}}(\theta)$：approximate importance ratio（新策略vs旧策略的action density比）
- $\hat{\mathcal{L}}_{\mathrm{CFM},\theta}(a_t;o_t)$：在参数$\theta$下，给定observation $o_t$，action $a_t$的conditional flow matching loss的Monte Carlo估计
- $\theta_{\mathrm{old}}$：behavior policy参数（rollout时用的）
- $\theta$：current policy参数（being optimized）

**Intuition**：CFM loss衡量policy网络预测velocity field与"true"velocity（指向clean action）的差距。Loss越低 → policy对该action越"确信" → 隐含的likelihood越高。Loss差异的指数就是log-likelihood ratio的近似。

### CFM Loss的具体形式

对每个action $a_t$，采样$N_{\mathrm{mc}}$个$(\tau_i, \epsilon_i)$对：

公式5（linear interpolation schedule）：
$$a_t^{\tau_i} = \tau_i a_t + (1-\tau_i)\epsilon_i$$

- $a_t$：rollout中实际采样的clean action
- $\epsilon_i \sim \mathcal{N}(0,I)$：高斯noise（与action同维度）
- $\tau_i \in [0,1]$：flow step（时间/进度变量）
- $a_t^{\tau_i}$：noised action（在noise和clean之间的线性插值）

公式6（target velocity field）：
$$\partial/\partial\tau_i \, a_t^{\tau_i} = a_t - \epsilon_i$$

这是linear interpolation的解析velocity，恒定指向从$\epsilon_i$到$a_t$。

公式7-8（CFM loss）：
$$\hat{\mathcal{L}}_{\mathrm{CFM},\theta}(a_t;o_t) = \frac{1}{N_{\mathrm{mc}}}\sum_{i=1}^{N_{\mathrm{mc}}}\ell_\theta^{(i,t)}$$

$$\ell_\theta^{(i,t)} = \|\hat{v}_\theta(a_t^{\tau_i}, \tau_i; o_t) - (a_t - \epsilon_i)\|_2^2$$

- $\hat{v}_\theta$：policy网络（参数$\theta$）预测的velocity，输入是noised action $a_t^{\tau_i}$、flow step $\tau_i$、observation $o_t$
- Target是$(a_t - \epsilon_i)$
- 整体是MSE regression

**关键connection**：训练flow policy就是回归velocity field；inference时从$\epsilon \sim \mathcal{N}(0,I)$出发，用Euler integration沿learned velocity field走到action space。这等价于ODE求解，最终的action distribution隐式定义了policy。

### FPO objective

公式4：
$$\max_\theta \mathbb{E}_{\pi_{\theta_{\mathrm{old}}}}\left[\psi_{\mathrm{PPO}}\left(\hat{\rho}_{\mathrm{FPO}}(\theta), \hat{A}_t\right)\right]$$

其中$\psi_{\mathrm{PPO}}$是公式1的clipped objective，$\hat{A}_t$是GAE advantage estimate [17]。

Positive advantage → 降低该action的CFM loss → 增加其隐含likelihood
Negative advantage → 增加CFM loss → 降低其隐含likelihood

---

## FPO++的两个关键改进

### 改进1：Per-sample ratio（公式10）

**原FPO的问题**（公式9）：
$$\hat{\rho}_{\mathrm{FPO}}(\theta) = \exp\left(\frac{1}{N_{\mathrm{mc}}}\sum_{i=1}^{N_{\mathrm{mc}}}(\ell_{\theta_{\mathrm{old}}}^{(i,t)} - \ell_\theta^{(i,t)})\right)$$

这里先对$N_{\mathrm{mc}}$个sample的loss差异取平均，再exp，最后clip。这意味着**对一个action只有单个ratio**，clip要么全clip要么全不clip。

**FPO++的改进**（公式10）：
$$\hat{\rho}_{\mathrm{FPO++}}^{(i)}(\theta) = \exp\left(\ell_{\theta_{\mathrm{old}}}^{(i,t)} - \ell_\theta^{(i,t)}\right)$$

每个$(\tau_i, \epsilon_i)$对独立计算ratio，独立clip，共享同一个advantage $\hat{A}_t$。

**Intuition**：考虑multi-step gradient descent（PPO通常跑多个epoch over同一批data）。某些sample的ratio可能drift出trust region，而其他sample还在范围内。Per-action ratio会把整个action的梯度全部clip掉或全部保留，丢失信息。Per-sample ratio允许**部分sample继续贡献梯度**，等价于增大effective batch size，降低gradient variance（论文Appendix B用cosine similarity metric验证了这一点）。

数学上，on-policy时（第一次update）所有ratio = 1，两种formulation梯度相同。但multi-epoch后per-sample提供finer-grained trust region。

### 改进2：ASPO (Asymmetric Trust Region)

**观察**：FPO训练from scratch时不稳定，尤其是negative advantage的case。

**Asymmetric设计**（公式12）：
$$\psi_{\mathrm{ASPO}}(\rho_\theta, \hat{A}_t) = \begin{cases} \psi_{\mathrm{PPO}}(\rho_\theta, \hat{A}_t), & \hat{A}_t \geq 0 \\ \psi_{\mathrm{SPO}}(\rho_\theta, \hat{A}_t), & \hat{A}_t < 0 \end{cases}$$

**SPO objective**（公式11）：
$$\psi_{\mathrm{SPO}}(\rho_\theta, \hat{A}_t) = \rho_\theta\hat{A}_t - \frac{|\hat{A}_t|}{2\varepsilon^{\mathrm{clip}}}(\rho_\theta - 1)^2$$

**变量解释**：
- 第一项$\rho_\theta\hat{A}_t$：standard policy gradient surrogate
- 第二项$\frac{|\hat{A}_t|}{2\varepsilon^{\mathrm{clip}}}(\rho_\theta - 1)^2$：quadratic penalty，当ratio偏离1时施加恢复力
- $\varepsilon^{\mathrm{clip}}$：clip parameter（与PPO的$\varepsilon^{\mathrm{clip}}$相同）

**对比PPO clip**：PPO在$|\rho - 1| > \varepsilon$时梯度直接为0（no signal），SPO始终提供梯度。

**为何asymmetric**：
- **Positive advantage**（$\hat{A}_t \geq 0$）：gradient pushes to decrease CFM loss → increase likelihood。这是"good action"，PPO clip足够，允许aggressive update。
- **Negative advantage**（$\hat{A}_t < 0$）：gradient pushes to increase CFM loss → decrease likelihood。这里风险高：
  - **(i)** Aggressive likelihood decrease → entropy collapse（policy变deterministic，失去exploration）
  - **(ii)** 从variational bound视角 [1, 33]，CFM loss increase也意味着sampled denoising posterior与learned denoising posterior之间KL divergence增加 → variational gap unstable

SPO的quadratic penalty限制了negative advantage case的aggressive update，**preserve entropy** + **stabilize variational gap**。

**Visualized in Figure 7**：PPO clipping的flow field随训练narrowing（entropy collapse），ASPO保持broad distribution。

### FPO++ Final Objective（公式13）

$$\max_\theta \mathbb{E}_{\pi_{\theta_{\mathrm{old}}}}\left[\sum_{i=1}^{N_{\mathrm{mc}}}\psi_{\mathrm{ASPO}}\left(\hat{\rho}_{\mathrm{FPO++}}^{(i)}(\theta), \hat{A}_t\right)\right]$$

对$N_{\mathrm{mc}}$个per-sample ratios求和（不是平均），每个sample独立用ASPO trust region。

---

## Zero-sampling：Test-time的简单但关键的trick

**Training**：$\epsilon \sim \mathcal{N}(0,I)$ → Euler integration → stochastic policy for exploration
**Test-time**：$\epsilon = \vec{0}$ → deterministic action

**为何有效**：
1. Training时需要stochasticity探索，但inference时random noise引入unnecessary variance
2. Zero-init让flow integration从distribution的"mode"出发（因为prior是zero-mean Gaussian，原点附近密度最高）
3. 与behavior cloning中的观察一致 [35]

**实验证据（Table I, G1 motion tracking）**：

| Sampling method | 5 steps | 50 steps |
|---|---|---|
| Random sampling | 34.7 ± 55.0 | 38.4 ± 26.6 |
| Zero-sampling | **45.1 ± 27.4** | **45.5 ± 23.2** |

注意：5-step zero-sampling甚至比50-step random sampling还好，这对sim-to-real的latency至关重要（real robot onboard compute受限）。

---

## 实验结果与技术细节

### Locomotion benchmarks（IsaacLab）

**Robots**：Go2, Spot（quadrupeds）；H1, G1（humanoids）
**Setup**：4096 parallel envs, 24 env steps between updates, 1500 updates (quadruped) / 2000 (humanoid), 64 Euler steps

**Network**：3-layer MLP, actor 256 hidden units, critic 768

**Key result（Figure 2）**：FPO在所有robot上unstable（local minima, catastrophic failure），即使sweep learning rate $\in \{10^{-5}, 10^{-4}, 3\times10^{-4}\}$, clip $\in \{0.04, 0.05, 0.06\}$, MC samples $\in \{8,16,32\}$。FPO++在所有任务stable。

**Ablation（Figure 5, 6）**：
- Per-sample ratio：consistently improve所有embodiment
- ASPO：consistently improve locomotion，但**degrade manipulation fine-tuning**（Appendix D.5）

### Sim-to-real（首次flow policy sim-to-real）

**Robots**：Booster T1（locomotion）, Unitree G1（motion tracking, 29 DoF, 50Hz control）
**Motion tracking**：6 LAFAN motions（dance, walk, run, fight, jumps），每个~2.5min
**Domain randomization**：friction, mass, external pushes, actuator delays, COM offset

**Deployment**：zero-sampling with 5 flow steps（训练时50 steps），降低latency

**Significance**：首次 (i) flow policy without expert distillation的humanoid sim-to-real，(ii) policy gradient without explicit likelihoods的sim-to-real

### Manipulation fine-tuning

**Tasks**：RoboMimic (Can, Square, Box Cleanup), DexMimicGen (Tray Lift, Threading)
**Base policy**：image-based flow matching, ViT encoder + 3-layer MLP, action chunk horizon 16
**Fine-tuning**：chunk-level ratio（sum CFM losses across chunk timesteps），10 flow steps, 8 MC samples

**Result（Figure 4）**：FPO++ consistently收敛最快，DPPO variants underperform

**DPPO失败原因分析**：
- DPPO将diffusion denoising step视为MDP decision point → inflates credit assignment horizon
- Stochastic noise injection at each diffusion step → 进一步降低rollout success probability → 高gradient variance（Figure A.8）
- DPPO对base policy quality敏感：random sampling success rate低时（如Can的10%），DPPO几乎学不动

### 与Gaussian PPO对比（Figure 8, 9, 10）

**Sample efficiency**：FPO++在大多数environment count下收敛到更高return，variance更低

**Gait emergence（Figure 9）**：
- Gaussian PPO + Spot → "pronk"（symmetric, 所有腿同时离地）
- FPO++ + Spot → "trot"（diagonal legs alternating）

**为何**：Gaussian各维度独立采样，难以探索correlated behaviors。Flow policy可以express coupling。

**Cross-correlation heatmap（Figure 10）**：FPO++训练中emerge出left/right hip negatively correlated（trot gait的特征），Gaussian PPO无法表达。

### Limitations

1. **Wall-clock时间**：FPO++比Gaussian PPO慢~20%（G1 locomotion: 19min vs 23min达到return 25），motion tracking慢~3x
2. **Motion tracking return略低于tuned Gaussian PPO**：作者attribute to缺少entropy regularization和adaptive learning rate（Appendix A.6, D.2尝试加入但仍有gap）
3. **ASPO对fine-tuning有害**：因为fine-tuning已well-initialized，entropy preservation不重要，反而引入noise

---

## 与相关工作的对比

| Method | Likelihood | Architecture | BPTT | Key limitation |
|---|---|---|---|---|
| **FPO++** | Bypassed via CFM surrogate | Any flow model | No | Slower than Gaussian PPO |
| DPPO [8] | From noise | Two-layer MDP | Yes | Long horizon, sensitive to base policy |
| ReinFlow [9] | From learned noise predictor | Specific arch | Partial | Architecture-constrained |
| NCDPO [29] | From initial + sampler noise | Unrolled | Yes | Vanishing/exploding gradients |
| GenPO [30] | Invertible architecture | Normalizing flow-inspired | Yes | Architecture-constrained |
| FQL [64] | None (Q-learning) | One-step flow | No | Offline only |
| Q-score matching [65] | Score linked to Q-grad | Diffusion | No | Offline only |

**FPO++的独特定位**：Online RL + bypass likelihood + architecture-agnostic + sim-to-real validated

---

## 技术实现细节（Appendix C）

### CFM loss clamping（Appendix C.2.3）

Early concern：squaring + exponentiation可能数值不稳定。Solution：
1. Clamp CFM loss before taking differences
2. Clamp the difference before exponentiation

实验验证（Figure A.4）：这比Huber loss或gradient-preserving clamp更简单且有效。

### Motion tracking hyperparameters（Table A.2）

- Flow integration steps: 50（training）→ 5（deployment, zero-sampling）
- Network: (1024, 512, 256) hidden, actor & critic same
- Learning rate: $3\times10^{-4}$
- Clip: 0.01
- GAE $\lambda = 0.95$, $\gamma = 0.99$
- 5 learning epochs, 4 minibatches

### Manipulation fine-tuning hyperparameters

- Actor LR: $1\times10^{-5}$, Critic LR: $1\times10^{-4}$（asymmetric，actor慢更新preserve BC initialization）
- GAE $\lambda = 0.99$
- $\gamma$ varies: 0.99 (Can), 0.995 (Square, Box Cleanup), 0.999 (Tray Lift, Threading) — horizon越长γ越大
- 10 flow steps, 8 MC samples per chunk

---

## Intuition总结

1. **CFM loss = 隐含的negative log-likelihood**：Flow policy训练本质是velocity field regression，但这个loss隐式定义了action distribution的energy landscape。Loss低 → density高。

2. **Per-sample ratio = finer-grained trust region**：Multi-epoch PPO中，不同noise sample的ratio drift不同，per-sample clip保留更多gradient signal，等价于增大batch size。

3. **ASPO asymmetry源于negative advantage的风险**：Decreasing likelihood（negative advantage）比increasing likelihood更危险——前者collapse entropy、destabilize variational gap，后者只是amplify good behavior。

4. **Zero-sampling = mode-seeking**：从prior的mode（原点）出发，flow integration自然走到learned distribution的mode，避免random sampling的variance。

5. **Flow policy的表达力来自coupling**：Gaussian各维度独立，flow的velocity field可以任意couple dimensions，这emerge出correlated gaits（trot vs pronk）。

6. **Sim-to-real成功的key**：Domain randomization + zero-sampling（5 steps够）+ ASPO保持的robustness。

---

## 对未来的implications

1. **Policy gradient不一定需要likelihood**：这open up design space——任何能提供log-likelihood ratio surrogate的训练signal都可以用policy gradient。例如energy-based model, VAE,甚至是implicit neural representation。

2. **Unified algorithm for from-scratch + fine-tuning**：FPO++在两个setting都work，challenge了"diffusion RL需要specialized architecture/MDP formulation"的assumption。

3. **Expressive policy + RL > Gaussian + RL**：当任务需要correlated exploration（locomotion gait, bimanual manipulation）时，flow policy的expressiveness转化为sample efficiency。

4. **Remaining gap**：Motion tracking略低于tuned Gaussian PPO，说明FPO++还缺一些RL engineering tricks（entropy regularization, KL-adaptive LR）。Appendix D.2尝试了non-parametric Kozachenko-Leonenko entropy estimator和KL-adaptive LR via L2 distance on predicted noise，但有gap——这指向flow policy的entropy/KL estimation本身是open problem。

---

## References

- [1] FPO: https://arxiv.org/abs/2507.21053
- [3] PPO: https://arxiv.org/abs/1707.06347
- [8] DPPO: https://arxiv.org/abs/2409.00588
- [9] ReinFlow: https://arxiv.org/abs/2505.22094
- [10] Diffusion Policy: https://arxiv.org/abs/2303.04137
- [11] π0: https://www.physicalintelligence.company/blog/pi0
- [13] FFJORD: https://arxiv.org/abs/1810.01367
- [17] GAE: https://arxiv.org/abs/1506.02438
- [20] Flow Matching: https://arxiv.org/abs/2210.02747
- [32] SPO: https://arxiv.org/abs/2401.16025
- [64] FQL: https://arxiv.org/abs/2502.02538
- [65] Q-score matching: https://arxiv.org/abs/2312.11752
- Project page: https://hongsukchoi.github.io/fpo-control

**一句话总结**：FPO++证明了——通过per-sample ratio clipping和asymmetric trust region，可以稳定地用policy gradient训练flow policy for real robot control，bypassing likelihood computation entirely，同时享受flow representation带来的expressiveness和sample efficiency优势。
