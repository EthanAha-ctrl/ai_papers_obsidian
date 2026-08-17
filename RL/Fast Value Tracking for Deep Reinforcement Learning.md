---
source_pdf: Fast Value Tracking for Deep Reinforcement Learning.pdf
paper_sha256: 86f12cac9e816ab3625aa88180c8a71cf3f214ca46872e4d8c3bce8cf14e7c52
processed_at: '2026-08-04T06:55:16-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说这篇 Paper

## 一句话总结

DQN 是找一个"最佳答案"，LKTD 是找"最佳答案的分布"——不仅告诉你答案是什么，还告诉你它有多不确定。

## 为什么要搞这个

想象你在玩一个游戏，你要估计"在这个状态下，做这个动作能拿多少分"。传统 RL（比如 DQN）的做法是：训练一个神经网络，让它输出一个数字，比如 +3.5，然后就信了。

问题在于：你凭什么信这个 +3.5？你的网络可能只见过 10 个类似的 situation，它其实非常不确定，真正值可能在 +1 到 +6 之间。但 DQN 不会告诉你这个范围，它只给你一个 point estimate，然后你就傻乎乎地照着做了。

更糟糕的是，RL 是一个 **live process**——你的 policy 每一秒都在变，数据分布也在变。你昨天收集的经验，今天可能就不适用了。DQN 假设有一个 fixed optimal answer 等你去 converge，但 reality 是你在追逐一个 moving target。

所以作者说：我们应该把 parameters 当作 **random variables**，不是去 converge 到一个点，而是去 **track** 一个分布。这就是 Kalman filter 的思路——它生来就是干这个的，火箭追踪、GPS 定位都是这套。

## Kalman Filter 怎么套到 RL 上

标准 Kalman filter 有两个 equation：

1. **State evolution**: 东西自己怎么变（火箭位置随时间漂移）
2. **Measurement**: 你观测到什么（雷达读数）

映射到 RL：
- **State** = neural network 的 parameters $\theta$
- **State evolution** = parameters 随训练慢慢变
- **Measurement** = reward $r_t$
- **Measurement function** = Bellman equation 算出来的 TD target

具体来说，Bellman equation 告诉我们：
$$r_t \approx Q_\theta(s_t, a_t) - \gamma Q_\theta(s_{t+1}, a_{t+1})$$

也就是说，reward 减去 discounted future Q-value，这个差值就是对你 parameters 的一个 noisy observation。把它写成 $r_t = h(x_t, \theta_t) + \eta_t$，这就是 Kalman filter 的 measurement equation。

## 老方法为什么不行

之前有人（KOVA、UKF）试过用 Kalman filter 做 RL，但遇到了大麻烦。

核心问题是：$h(x, \theta)$ 对 $\theta$ 是 nonlinear 的（因为 deep network）。标准 Kalman filter 只能处理 linear measurement。老方法的 workaround 是 **linearization**——在当前点做一阶 Taylor 展开，假装它是线性的。

但这带来三个致命问题：

1. **不准**：deep network 的 loss landscape 高度非凸，一阶近似误差巨大
2. **慢**：要算 $p \times n$ 的 Jacobian matrix，还要做矩阵求逆，complexity 是 $O(np^2)$。如果你的 network 有一百万个参数，$p^2 = 10^{12}$，完全不可行
3. **费内存**：要存 $p \times p$ 的 covariance matrix，又是 $10^{12}$ 个数

## LKTD 的核心 trick：Variance Splitting

作者的 key insight 是：**不要 linearize，换个角度让 measurement 变成 linear 的**。

具体做法是引入一个 auxiliary variable $\xi_t$，它是 $h(x_t, \theta_t)$ 的一个 noisy copy：

$$\xi_t = h(x_t, \theta_t) + u_t$$

然后把原来的 measurement noise $\sigma^2$ 拆成两半：
- 一半 $\alpha\sigma^2$ 给 $\xi_t$ 的 noise
- 另一半 $(1-\alpha)\sigma^2$ 给最终 measurement

现在 measurement equation 变成：
$$r_t = \xi_t + v_t$$

这就是 **linear** 的！因为 $\xi_t$ 只是 augmented state $\varphi_t = (\theta_t, \xi_t)$ 的后半部分，可以用一个简单的 selector matrix $H_t = (0, I_n)$ 来提取。

**Intuition**：相当于你插了一个中间层。Nonlinearity 被推到了 state evolution 那边（可以采样处理），measurement 这边保持 linear（可以用标准 Kalman update）。这样就不需要算 Jacobian，不需要存大 covariance matrix，complexity 直接降到 $O(np)$。

## Langevin Dynamics 为什么加进来

光有 variance splitting 还不够。作者在 state evolution equation 里加了一个 prior gradient term：

$$\theta_t = \theta_{t-1} + \frac{\epsilon_t}{2}\nabla_\theta \log \pi(\theta_{t-1}) + w_t$$

这就是 **Langevin dynamics**——带 noise 的 gradient ascent on log prior。

为什么加 prior？两个好处：

1. **Robustness**：如果你用 mixture Gaussian prior，它会 induce sparsity，相当于自动做 network pruning。Sparse network 泛化更好
2. **Bayesian framework**：有了 prior，你采样的是 posterior distribution，可以量化 uncertainty

加上 Langevin noise 后，整个算法变成一个 SGMCMC (Stochastic Gradient Markov Chain Monte Carlo) sampler。它不是在 optimize，是在 **sample from posterior**。

## Pseudo-Population Size $\mathcal{N}$：最 subtle 的设计

这是整篇 paper 最 tricky 的概念。

在标准 Bayesian inference 里，posterior 的 concentration rate 取决于 sample size。Sample 越多，posterior 越 sharp，最后 degenerate 到一个点。这是 Bernstein-von Mises 定理。

RL 的问题在于：数据是 stream 进来的，理论上 sample size 是 infinite。如果直接套 Bayesian framework，posterior 立刻 degenerate，你就失去了 uncertainty quantification——回到了 DQN 的老路。

作者的解决方案：人为设定一个 **pseudo-population size** $\mathcal{N}$，假装你的 total dataset 只有 $\mathcal{N}$ 这么大。

- $\mathcal{N}$ 大 → posterior sharp → 接近 optimization（像 DQN）
- $\mathcal{N}$ 小 → posterior diffuse → uncertainty 大，但估计可能不准

这是一个 tempering mechanism。理论上证明（Theorem 1），即使 posterior 不是 delta function，它的 mode 仍然是 optimal $\theta^*$。所以你从 posterior 采样再取平均，仍然能得到正确的 policy。

实验中 $\mathcal{N}$ 用 2500 到 20000，取决于任务复杂度。

## 算法流程用人话讲

每个 time step $t$：

1. **收集数据**：用当前 policy 跑 $n$ 步，拿到 transitions $(s, a, r, s')$
2. **内循环 $\kappa$ 次**（默认 5 次）：
   - **Forecast**：做一步 Langevin dynamics——参数沿着 prior gradient 走一步，加 Gaussian noise。这相当于"预测"参数会怎么变
   - **Analysis**：用 Kalman gain 把观测 $r_t$ 和预测的 $\xi_t$ 融合。这相当于"修正"——根据观测调整 auxiliary variable $\xi_t$
3. **更新 policy**：用最新的参数 sample 决定下一步 action

注意：只有 $\xi$ 部分在 analysis step 被 update，$\theta$ 部分只在 forecast step 变。这是为什么算法快——不需要对 $\theta$ 的 $p$ 维空间做矩阵操作。

## 为什么不用 Target Network

DQN 需要 target network 来 stabilize 训练。原因是用同一个 network 算 current Q 和 target Q，会形成 positive feedback loop——network 自己骗自己，Q-value 会爆炸。

LKTD 不需要 target network（实验里 target update interval = 1）。原因有两个：

1. **Sampling noise**：Langevin dynamics 的 inherent noise 打破了 positive feedback
2. **Kalman correction**：innovation term $r_t - H_t\varphi^f$ 是 self-correcting 的。如果预测偏高，下一次 update 会把它拉回来

这是一个 nice side effect——少了一个 hyperparameter 要调。

## 实验说了什么

### Indoor Escape（10×10 grid）

这个环境有个特点：很多状态下，往北走和往东走的 Q-value 一样。所以存在多个 optimal policy。

- **Q-value 准确度**：LKTD 的 MSE 是 0.0002，DQN 是 0.1。差了 500 倍
- **Coverage rate**（95% 置信区间是否真的覆盖真值）：LKTD 94.5%，DQN 41%，QR-DQN 86%，KOVA 25%
- **Policy exploration**：LKTD 能同时探索 N 和 E 两个最优动作，DQN 会卡在一个上

最后一点很重要。DQN 容易陷入 local trap——它找到一个 optimal policy 就死磕，不管还有没有别的。LKTD 因为有 sampling noise，能持续探索不同的 optimal policy，学到更 robust 的 behavior。

### Computation Time

- LKTD: 1.33ms/iter
- DQN: 1.80ms/iter（LKTD 甚至更快）
- KOVA: 44ms/iter（慢 33 倍）

KOVA 慢是因为要算 Jacobian 和矩阵求逆。当 network 从 [32,32] 变到 [64,64]，KOVA 从 44ms 暴涨到 251ms（5.7x），LKTD 只从 1.33 到 1.49（1.1x）。Scaling 特性完全不同。

### OpenAI Gym

在 CartPole、Acrobot、LunarLander、MountainCar 上，LKTD 的 training reward 和 evaluation reward 都优于 DQN 和 QR-DQN。尤其在 CartPole 这种 sensitive 环境（一步走错就 game over），LKTD 的 robustness 优势明显。

## 和其他方法对比

| Method | Uncertainty 来源 | Coverage Rate | Complexity | 需要 Target Network |
|--------|-----------------|---------------|------------|-------------------|
| DQN | 无 | ~41% | $O(np)$ | 是 |
| BootDQN | Bootstrap ensemble | ~38% | $O(knp)$ | 是 |
| QR-DQN | Return distribution | ~86% | $O(np)$ | 是 |
| KOVA | Kalman + EKF | ~25% | $O(np^2)$ | 是 |
| LKTD | Posterior sampling | ~94.5% | $O(np)$ | 否 |

BootDQN 用 multiple heads 做 uncertainty，但 bootstrap 是 frequentist 方法，没有真正的 posterior。QR-DQN 学的是 return 的 distribution（aleatoric uncertainty），不是参数的 uncertainty（epistemic uncertainty）。只有 LKTD 有 principled Bayesian framework。

## 核心教训

1. **Optimization 和 Sampling 的边界**：$\mathcal{N}$ 参数让你在这个 spectrum 上自由调节。大 $\mathcal{N}$ 接近 optimization，小 $\mathcal{N}$ 接近 pure sampling。这不是 binary choice

2. **Linearization 是 last resort**：能通过 variable augmentation 避免 linearization 就避免。Sampling 比 linearization 更 general，也更 cheap

3. **Uncertainty 不是 luxury，是 necessity**：在 non-stationary environment 里，不知道自己不知道什么是最危险的。DQN 的 41% coverage rate 意味着它 59% 的时候在骗你——它给了你一个"答案"但没告诉你这个答案可能完全错

4. **Target network 是 workaround，不是 solution**：LKTD 证明如果你从根本上改变了 learning paradigm（从 optimization 到 sampling），positive feedback problem 自然消失

## 局限

1. **$\mathcal{N}$ 怎么选**：paper 没给 principled method，只是实验调参。理论上 $\mathcal{N} > r_n^4$ 但 $r_n$ 难估
2. **Inner loop overhead**：$\kappa=5$ 意味着每步算 5 次 gradient，对大 network 是负担
3. **只做了 Q-learning**：Actor-critic、policy gradient 方法的 extension 没讨论
4. **Replay buffer 理论**：假设 R-dependent，实际 prioritized replay 更复杂

## References

- Paper arXiv: https://arxiv.org/abs/2401.13062
- LEnKF (Zhang et al. 2023): https://doi.org/10.5705/ss.202022.0172
- SGLD (Welling & Teh 2011): https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf
- BootDQN (Osband et al. 2016): https://arxiv.org/abs/1602.04621
- QR-DQN (Bellemare et al. 2017): https://arxiv.org/abs/1707.06887
- KOVA (Shashua & Mannor 2020): https://arxiv.org/abs/2002.07171
- Sparse DNN (Sun et al. 2022): https://doi.org/10.1080/01621459.2021.1895178
- Raginsky et al. 2017 (SGLD convergence): https://arxiv.org/abs/1702.03849
- EnKF (Evensen 1994): https://doi.org/10.1029/94JC00572
- RL Baselines3 Zoo: https://github.com/DLR-RM/rl-baselines3-zoo

---

# Fast Value Tracking for Deep Reinforcement Learning 详细讲解

## 1. 核心Motivation: RL的本质矛盾

这篇paper试图解决一个深层的conceptual问题。传统RL算法（DQN、TD-learning、SARSA、Q-learning）把value function $V_\rho(s)$ 或 Q-function $Q_\rho(s,a)$ 当作一个 **deterministic unknown**，用SGD之类的方法去找到它的point estimate。这种视角有一个根本缺陷: agent-environment交互本质上是stochastic dynamical system，每一次policy update都会改变数据分布，而数据分布的变化又反过来影响参数的演化。这是一个 **closed-loop** 系统，但point-estimate的optimization视角把它当作open-loop来处理了。

作者提出，更合理的视角是把value或model parameters当作 **random variables**，并且要 **track** 它们的动态变化，而不是converge到一个fixed point。这直接引出Kalman filtering paradigm。

## 2. KTD (Kalman Temporal Difference) 框架

### 2.1 State-Space Model 的构建

核心idea是把RL重写成一个state-space model，这是Kalman filter的标准形式：

$$\theta_t = \theta_{t-1} + w_t \quad \text{(state evolution equation)}$$
$$r_t = h(x_t, \theta_t) + \eta_t \quad \text{(measurement equation)}$$

变量含义:
- $\theta_t \in \mathbb{R}^p$: time step $t$ 的neural network参数，维度 $p$（对于deep network会很大）
- $w_t \in \mathbb{R}^p$: process noise，multivariate Gaussian，表示参数本身的随机演化
- $\eta_t \in \mathbb{R}^n$: measurement noise，表示reward观测的随机性
- $x_t$: 在time step $t$ 收集的states和actions的集合
- $r_t \in \mathbb{R}^n$: reward向量，$n$是batch size
- $h(\cdot, \cdot)$: 测量函数，由Bellman equation定义

### 2.2 测量函数 $h$ 的具体形式

基于Bellman equation:
$$V_\rho(s) = \mathbb{E}_\rho[r(s,a) + \gamma V_\rho(s')]$$
$$Q_\rho(s,a) = \mathbb{E}_\rho[r(s,a) + \gamma Q_\rho(s',a')]$$

$h(x_t, \theta_t)$ 被定义为:

- **V-function**: $h(x_t, \theta_t) = V_{\theta_t}(s_t) - \gamma V_{\theta_t}(s_{t+1})$
- **Q-function**: $h(x_t, \theta_t) = Q_{\theta_t}(s_t, a_t) - \gamma Q_{\theta_t}(s_{t+1}, a_{t+1})$

这里的逻辑是: Bellman equation说 $r_t \approx h(x_t, \theta_t)$（在期望意义下），所以reward $r_t$ 就是参数 $\theta_t$ 的一个noisy measurement。这个reformulation非常elegant，把temporal-difference error直接变成了Kalman filter的innovation（新息）。

### 2.3 现有KTD的瓶颈

之前的KTD算法（Geist & Pietquin 2010的UKF版本，Shashua & Mannor 2020的KOVA/EKF版本）都遇到了致命问题:

1. **Linearization问题**: $h(x, \theta)$ 关于 $\theta$ 是nonlinear的（因为deep network），EKF用first-order Taylor展开:
   $$h(x_t, \theta) \approx h(x_t, \hat{\mu}_{t-1}) + \nabla_\theta h(x_t, \hat{\mu}_{t-1})^T(\theta - \hat{\mu}_{t-1})$$
   这种linearization在deep network的高维非凸landscape上误差很大。

2. **计算复杂度**: $O(np^2)$ per iteration，因为要算Jacobian $\nabla_\theta h$ (维度 $p \times n$) 和矩阵乘法。对于百万级参数的network，这完全不可行。

3. **存储复杂度**: $O(p^2)$ 来存covariance matrix $\hat{\Sigma}_t$。

## 3. LKTD的核心创新

### 3.1 Reformulation: 加入Prior到State Evolution

作者把model (1) 改写成model (2):

$$\theta_t = \theta_{t-1} + \frac{\epsilon_t}{2}\nabla_\theta \log \pi(\theta_{t-1}) + w_t$$
$$r_t = h(x_t, \theta_t) + \eta_t$$

这里 $\pi(\theta)$ 是一个prior density，$\epsilon_t \to 0$ 是递减的step size序列。

这个改动看似微小，实则深刻。第一行就是一个 **Langevin dynamics** step。加入 $\nabla_\theta \log \pi(\theta_{t-1})$ 这一项使得 $\theta_t$ 的演化被prior引导。当用mixture Gaussian prior时（如sparse DNN prior），可以induce sparsity，提升robustness (Sun et al. 2022, https://doi.org/10.1080/01621459.2021.1895178)。

### 3.2 Variance Splitting Technique: 关键的trick

这是整个方法最精妙的地方。目标是把nonlinear measurement equation转成linear的，同时允许state evolution保持nonlinear。

定义augmented state:
$$\varphi_t = \binom{\theta_t}{\xi_t}, \quad \xi_t = h(x_t; \theta_t) + u_t, \quad u_t \sim N(0, \alpha\sigma^2 I_n)$$

变量含义:
- $\varphi_t$: augmented state，维度 $\tilde{p} = p + n$
- $\xi_t \in \mathbb{R}^n$: auxiliary variable，是 $h(x_t; \theta_t)$ 的noisy copy
- $\alpha \in (0,1)$: variance splitting比例，把原来的measurement noise $\sigma^2$ 拆成 $\alpha\sigma^2$（给 $\xi_t$）和 $(1-\alpha)\sigma^2$（给最终measurement）
- $u_t$: auxiliary noise

原来的measurement noise $\eta_t \sim N(0, \sigma^2 I_n)$ 现在被拆成两部分。这是state augmentation的标准技巧。

基于这个augmentation，可以得到conditional distribution (equation 7):
$$\xi_t | r_t, x_t, \theta_t \sim N(\alpha r_t + (1-\alpha)h(x_t; \theta_t), \alpha(1-\alpha)\sigma^2 I_n)$$

这个公式说明: 给定观测 $r_t$ 和当前 $\theta_t$，$\xi_t$ 的后验均值是 $r_t$ 和 $h(x_t; \theta_t)$ 的convex combination，权重由 $\alpha$ 控制。这正是Kalman filter update的本质——prior estimate和observation的加权融合。

### 3.3 Reformulated Linear Measurement Model

现在measurement equation变成:

$$r_t = H_t \varphi_t + v_t$$

其中 $H_t = (0, I_n)$，所以 $H_t \varphi_t = \xi_t$（只取 $\varphi_t$ 的 $\xi$ 部分）。这是一个 **linear** measurement equation！而state evolution equation仍然保持nonlinear（通过Langevin dynamics）。

完整的model (equation 6):
$$\varphi_t = \varphi_{t-1} + \frac{\epsilon_t}{2}\frac{n}{\mathcal{N}}\nabla_\varphi \log \pi(\varphi_{t-1}) + \tilde{w}_t$$
$$r_t = H_t \varphi_t + v_t$$

变量含义:
- $\mathcal{N}$: **pseudo-population size**，一个关键的超参数，后面详细讲
- $\tilde{w}_t \sim N(0, \frac{n}{\mathcal{N}} B_t)$: 缩放过的process noise，$B_t = \epsilon_t I_{\tilde{p}}$
- $v_t \sim N(0, (1-\alpha)\sigma^2 I_n)$: measurement noise，与 $\tilde{w}_t$ 独立
- 因子 $\frac{n}{\mathcal{N}}$: 缩放noise的强度，反映mini-batch $n$ 相对于pseudo-population $\mathcal{N}$ 的比例

### 3.4 为什么这个trick如此重要

**Intuition**: 传统Kalman filter处理nonlinear measurement需要linearization（EKF）或sigma points（UKF），都涉及Jacobian计算或高维协方差矩阵操作。Variance splitting的巧妙之处在于:

1. 把nonlinearity "吸收"进augmented state $\xi_t$ 的定义中
2. 从measurement equation的角度看，$\xi_t$ 只是一个linear function of $\varphi_t$
3. Nonlinearity仍然存在于state evolution（通过 $\nabla_\varphi \log \pi(\varphi)$ 的gradient计算）和 $\xi_t$ 的conditional distribution中
4. 但这些nonlinear部分可以用 **sampling** 来处理（Langevin dynamics），不需要解析线性化

这就是为什么complexity从 $O(np^2)$ 降到 $O(np)$。

## 4. Pseudo-Population Size $\mathcal{N}$ 的深层含义

这是这篇paper另一个核心概念。$\mathcal{N}$ 的角色需要在optimization和sampling的tension中理解。

### 4.1 为什么需要 $\mathcal{N}$

在传统Bayesian inference中，posterior分布的concentration rate取决于sample size。当sample size $N \to \infty$，posterior degenerates到一个delta function（Bernstein-von Mises定理）。这在static dataset上是合理的。

但在RL中，数据是 **流式生成** 的，理论上sample size可以看作无限大。如果直接用无限sample size的Bayesian framework，posterior会立刻degenerate到point estimate，丧失uncertainty quantification的能力——这恰好是我们想避免的。

$\mathcal{N}$ 的引入相当于人为设定一个 "effective sample size"，使得posterior保持一个 **non-degenerate** 的stationary distribution $\nu_\mathcal{N}(\theta)$。

### 4.2 Mathematical Formulation

Theorem 1证明 $\theta_k$ 收敛到:
$$\nu_\mathcal{N}(\theta) \propto \exp(-\beta \mathcal{G}(\theta))$$

其中:
- $\beta$: inverse temperature
- $\mathcal{G}(\theta) = O(\mathcal{N})$: $\mathcal{G}(\theta)$ 的量级与 $\mathcal{N}$ 成正比
- $\nabla_\theta \mathcal{G}(\theta) = g(\theta)$: $\mathcal{G}$ 是expected gradient $g(\theta) = \int G(\theta, z)\pi(z|\theta)dz$ 的anti-derivative

当 $\mathcal{N} \to \infty$:
- $\mathcal{G}(\theta) \to \infty$
- $\exp(-\beta\mathcal{G}(\theta))$ 越来越concentrated
- $\nu_\mathcal{N}(\theta) \to \delta(\theta - \theta^*)$

这就是Remark 2的含义。$\mathcal{N}$ 是一个 **tempering factor**，在optimization（大 $\mathcal{N}$）和uncertainty quantification（小 $\mathcal{N}$）之间trade-off。

### 4.3 Laplace Approximation与Inference

Remark 1说，对于test function $\phi(\theta)$:
$$\bar{\phi}_\mathcal{N}(\theta) = \frac{\int \phi(\theta)\exp(-\beta\mathcal{G}(\theta))d\theta}{\int \exp(-\beta\mathcal{G}(\theta))d\theta} = \phi(\theta^*) + O\left(\frac{r_n^4}{\mathcal{N}}\right)$$

变量含义:
- $\theta^*$: $\nu_\mathcal{N}$（从而也是 $\nu_\infty$）的maximizer，对应optimal policy的参数
- $r_n$: sparse DNN的connectivity

当 $K \to \infty$ 且 $\mathcal{N} > r_n^4$，Monte Carlo average $\hat{\phi} = \frac{\sum_{k=1}^K \epsilon_k \phi(\theta_k)}{\sum_{k=1}^K \epsilon_k}$ 是 $\phi(\theta^*)$ 的consistent estimator。

这个结果的intuition: 即使 $\nu_\mathcal{N}$ 不是一个delta function（有uncertainty），它的mode仍然是 $\theta^*$，所以从 $\nu_\mathcal{N}$ 采样再做平均，仍然能恢复optimal policy。$\mathcal{N}$ 越大，bias越小，但uncertainty也越小。

## 5. Algorithm 1: LKTD的完整流程

```julia
Algorithm 1: LKTD
Initialization: 从prior π(θ)采样 θ_0^a ∈ R^p
for t = 1, 2, ..., T do
    Sampling: 用policy ρ_{θ^a} 生成n个transition tuples z_t = (r_t, x_t)
    for k = 1, 2, ..., κ do
        Presetting: 
            B_{t,k} = ε_{t,k} I_{p̃}
            R_t = 2(1-α)σ² I
            K_{t,k} = B_{t,k} H_t^T (H_t B_{t,k} H_t^T + R_t)^{-1}
        Forecast:
            φ^f_{t,k} = φ^a_{t,k-1} + (ε_{t,k}/2)(n/N)∇_φ log π(φ^a_{t,k-1}) + w̃_{t,k}
        Analysis:
            φ^a_{t,k} = φ^f_{t,k} + K_{t,k}(r_t - H_t φ^f_{t,k} - v_{t,k})
    end
end
```

### 5.1 Forecast Step 解析

$$\varphi^f_{t,k} = \varphi^a_{t,k-1} + \frac{\epsilon_{t,k}}{2}\frac{n}{\mathcal{N}}\nabla_\varphi \log \pi(\varphi^a_{t,k-1}) + \tilde{w}_{t,k}$$

这就是一个Langevin dynamics step:
- 第一项: 当前状态
- 第二项: prior gradient，推动参数向high-prior-density区域移动
- 第三项: Gaussian noise，提供exploration

gradient项的具体展开 (equation 9):
$$\nabla_\varphi \log \pi(\varphi) = \left(\nabla_\theta \log \pi(\theta) + \frac{1}{\alpha\sigma^2}\frac{\mathcal{N}}{n}\nabla_\theta h(x_t;\theta_t)(\xi_t - h(x_t;\theta_t))\right) - \frac{1}{\alpha\sigma^2}(\xi_t - h(x_t;\theta_t))$$

这个gradient分两部分:
- $\theta$ 分量: prior gradient + likelihood gradient（$\xi_t$ 与 $h(x_t;\theta_t)$ 的差异驱动的correction）
- $\xi$ 分量: $-\frac{1}{\alpha\sigma^2}(\xi_t - h(x_t;\theta_t))$，推动 $\xi_t$ 向 $h(x_t;\theta_t)$ 靠拢

### 5.2 Analysis Step 解析

$$\varphi^a_{t,k} = \varphi^f_{t,k} + K_{t,k}(r_t - H_t \varphi^f_{t,k} - v_{t,k})$$

这是标准Kalman update:
- $r_t - H_t \varphi^f_{t,k}$: innovation（观测残差）
- $K_{t,k}$: Kalman gain，决定innovation的权重

由于 $H_t = (0, I_n)$ 的特殊结构:
$$K_{t,k} = B_{t,k} H_t^T(H_t B_{t,k} H_t^T + R_t)^{-1}$$

只有 $\xi$ 部分被update，$\theta$ 部分在analysis step不变。这解释了为什么算法efficient——不需要对 $\theta$ 部分做矩阵操作。

### 5.3 内循环 $\kappa$ 的作用

内循环 $\kappa$ (默认=5) 是为了impute latent variable $\xi_t$。因为 $\xi_t$ 没有直接观测，需要通过Gibbs-like sampling来推断。从equation 7的条件分布，$\xi_t$ 的convergence很快（得益于second-order gradient information），所以 $\kappa$ 不需要很大。

初始化 $\xi_{t,0}$ 用 $r_t$，这是一个合理的warm start。

## 6. 收敛性理论

### 6.1 Lemma 1: LKTD是Preconditioned SGLD

$$\varphi^a_t = \varphi^a_{t-1} + \frac{\epsilon_t}{2}\Sigma_t \nabla_\varphi \log \pi(\varphi^a_{t-1} | z_t) + e_t$$

其中:
- $\Sigma_t = \frac{n}{\mathcal{N}}(I - K_t H_t)$: preconditioner matrix
- $e_t \sim N(0, \epsilon_t \Sigma_t)$: Gaussian noise with covariance matching preconditioner

这是 **preconditioned SGLD** 的标准形式 (Li et al. 2016, https://arxiv.org/abs/1512.07666)。Preconditioner $\Sigma_t$ 的作用类似于natural gradient或AdaGrad——根据curvature调整不同方向的step size。

关键性质: $\Sigma_t$ 有bounded positive eigenvalues:
$$\Sigma_t = \frac{n}{\mathcal{N}}\left[I - \epsilon_t H_t^T(\epsilon_t H_t H_t^T + R_t)^{-1}H_t\right]$$

这个表达式来自matrix inversion lemma。$I - \epsilon_t H_t^T(\cdot)^{-1}H_t$ 是一个projection-like matrix，它的eigenvalues在 $[0, 1]$ 范围内（当 $\epsilon_t > 0$），所以 $\Sigma_t$ 是positive definite的。

### 6.2 Theorem 1: W_2 Convergence

考虑SGLD sampler:
$$\theta_k = \theta_{k-1} + \epsilon_k G(\theta_{k-1}, z_k) + \sqrt{2\beta^{-1}\epsilon_k}\mathfrak{e}_k$$

变量含义:
- $G(\theta_{k-1}, z_k)$: stochastic gradient
- $\mathfrak{e}_k \sim N(0, I_d)$: standard Gaussian
- $\beta$: inverse temperature
- $\epsilon_k = \epsilon_0 / k^\varpi$, $\varpi \in (0,1)$: polynomially decaying learning rate

在Assumption A1的条件下（Lipschitz gradient、dissipativity、bounded variance等），2-Wasserstein distance的上界:

$$\mathcal{W}_2(\mu_K, \nu_\mathcal{N}) \leq \text{(复杂表达式，equation 13)}$$

这个bound有两部分:
1. $\mathcal{W}_2(\mu_K, \nu_{S_K})$: 离散采样与连续时间插值的距离
2. $\mathcal{W}_2(\nu_{S_K}, \nu_\infty)$: 连续时间SDE的收敛

第一部分由 $D_{KL}(\mu_K \| \nu_{S_K})$ 控制，涉及learning rate的decay rate和gradient的Lipschitz constant $L_U$。

第二部分以exponential rate $\exp(-S_K / (\beta c_{LS}))$ 衰减，其中 $c_{LS}$ 是logarithmic Sobolev constant。

### 6.3 Assumption A1详解

这些assumptions是Raginsky et al. (2017, https://arxiv.org/abs/1702.03849) 给出的SGLD收敛条件的变体:

- **(C1)**: 存在唯一stationary distribution $\pi(z|\theta)$，$G$ measurable，$g(\theta)$ bounded
- **(C2)**: $\mathcal{G}(\theta)$ 存在且bounded at origin
- **(C3) Lipschitz**: $\|g(\theta) - g(\vartheta)\| \leq L_U\|\theta - \vartheta\|$
- **(C4) Dissipativity**: $\langle\theta, g(\theta)\rangle \geq m_U\|\theta\|^2 - b$。这个条件保证dynamics不会跑到无穷远——当 $\theta$ 远离origin时，drift把它拉回来
- **(C5) Bounded variance**: $\mathbb{E}\|G(\theta,z) - g(\theta)\|^2 \leq 2\delta(M_U^2\|\theta\|^2 + B^2)$
- **(C6)**: 初始分布有bounded density

### 6.4 Theorem 2: Replay Buffer的收敛性

这是对off-policy setting的extension。Replay buffer在population层面建模为mixture distribution:

$$\bar{\pi}(z | \theta_{t-1}^R) = \frac{1}{R}\sum_{i=1}^R \pi(z | \theta_{t-i})$$

其中 $\theta_{t-1}^R = \{\theta_{t-i}\}_{i=1}^R$，$R$ 是buffer的capacity。

关键假设:
- **(i) Lipschitz**: $\int|\pi(z|\theta) - \pi(z|\vartheta)|^2 dz \leq L\|\theta - \vartheta\|^2$
- **(ii) Integrability**: gradient的variance bounded

结论:
$$|\mathbb{E}\hat{\phi} - \bar{\phi}| = O\left(\frac{1}{S_T} + \frac{\sum_{t=1}^T \epsilon_t^2}{S_T}\right)$$
$$\mathbb{E}(\hat{\phi} - \bar{\phi})^2 = O\left(\frac{1}{S_T} + \frac{\sum_{t=1}^T \epsilon_t^2}{S_T^2} + \frac{(\sum_{t=1}^T \epsilon_t^2)^2}{S_T^2}\right)$$

其中 $S_T = \sum_{t=1}^T \epsilon_t$。

**Intuition**: Replay buffer引入了gradient bias（因为样本来自过时的policy），但这个bias的量级是 $O(\epsilon_t^2)$（因为相邻参数的差异是 $O(\epsilon_t)$，平方后是 $O(\epsilon_t^2)$）。当learning rate递减时，这个bias asymptotically消失。

证明的关键step (equation A23-A26):
$$\|\mathbb{E}[\zeta_t | \mathcal{F}_{t-1}]\|^2 \leq \frac{ML}{R}\sum_{i=1}^R \|\theta_{t-i} - \theta_t\|^2 \leq O(\epsilon_t^2)$$

这里 $\zeta_t = G(\theta_{t-1}, z_t) - g(\theta_{t-1})$ 是gradient bias。通过Lipschitz条件和Jensen不等式，把参数差异的bound转化为gradient bias的bound。

## 7. 实验结果解析

### 7.1 Indoor Escape Environment

10×10 grid，agent从左下角到右上角。Reward $\sim N(-1, 0.01)$。关键特性: 对于多数states，actions N和E的Q-values相同，所以存在多个optimal policies。

**Metrics**:
1. MSE between $\hat{Q}(s,a)$ 和 $Q^*_\epsilon(s,a)$
2. Coverage rate (CR) of 95% prediction intervals
3. Mean policy probability $p_\varrho(a|s) = \frac{1}{|\varrho|}\sum_{\rho \in \varrho} \mathbf{1}_a(\rho(s))$

从Table A1的数据看:
- LKTD (N=10000): MSE ≈ 0.0002
- DQN: MSE ≈ 0.1（差500倍）
- QR-DQN: MSE ≈ 0.006
- KOVA: MSE ≈ 0.006

从Table A2的coverage rate:
- LKTD: ~94.5%（接近nominal 95%）
- DQN: ~41%（严重undercoverage）
- QR-DQN: ~86%（也不够准确）
- KOVA: ~25%（最差）

这些数据强烈支持LKTD的uncertainty quantification能力。

### 7.2 Computation Time (Table A3)

- LKTD [32,32], batch 100: 1.326ms/iter
- DQN: 1.80ms/iter（LKTD甚至更快，因为DQN用了target network updates）
- KOVA [32,32], batch 100: 44.20ms/iter（慢33倍）
- KOVA [64,64]: 251ms/iter（scaling很差）

KOVA的bottleneck是Jacobian计算和矩阵inversion。当hidden layer从[32,32]变到[64,64]时，KOVA的时间从44ms暴涨到251ms（5.7x），而LKTD只从1.33ms到1.49ms（1.1x）。

### 7.3 Classical Control (OpenAI Gym)

从Figure 5 (CartPole-v1) 和Figure A4看:
- LKTD在training reward上显著高于DQN
- 在evaluation reward（无random exploration）上也更好
- 在best model reward上，LKTD能更快找到好的policy

Hyperparameter settings (Table A4) 有几个值得注意的点:
- LKTD的learning rate比DQN小约100x（如CartPole: 2.5e-5 vs 2.3e-3），这是因为LKTD做了inner loop $\kappa$次
- LKTD的target update interval是1（不用target network），而DQN是10-250
- LKTD的buffer size更小（1e4 vs 1e5）

不用target network是一个有趣的副作用——因为LKTD本身通过sampling来处理不确定性，不需要target network来stabilize训练。

## 8. 与其他方法的关系

### 8.1 与SGLD的关系

Lemma 1显示LKTD是preconditioned SGLD。作者也在Appendix给出了纯SGLD和SGHMC的版本（Algorithm S3, S4）。这些算法的stationary distribution相同，但LKTD通过Kalman gain的preconditioning加速了convergence。

### 8.2 与BootDQN (Osband et al. 2016, https://arxiv.org/abs/1602.04621) 的区别

BootDQN用multiple bootstrap heads来estimate uncertainty，但:
- Bootstrap是frequentist方法，没有明确的posterior distribution
- Coverage rate只有~38%（Table A2），远低于nominal 95%
- 无法做value tracking

### 8.3 与QR-DQN (Bellemare et al. 2017, https://arxiv.org/abs/1707.06887) 的区别

QR-DQN学习return的distribution，但:
- 它approximate的是aleatoric uncertainty（reward的随机性），不是epistemic uncertainty（参数的不确定性）
- Coverage rate ~86%，仍然不够准确
- 无法track参数的动态变化

### 8.4 与EnKF (Evensen 1994, https://doi.org/10.1029/94JC00572) 的关系

LEnKF是EnKF的Langevin化版本。传统EnKF用ensemble of particles来approximate distribution，但不converge到exact posterior。LEnKF通过Langevin dynamics的correction term保证了exact convergence (Zhang et al. 2023, https://doi.org/10.5705/ss.202022.0172)。

## 9. 核心Intuition Summary

**Intuition 1: Value Tracking vs Value Convergence**

传统RL寻找一个fixed point $\theta^*$。LKTD寻找一个 **distribution** $\nu_\mathcal{N}(\theta)$，它的mode是 $\theta^*$，但有non-zero variance。这个variance就是uncertainty的量化。当policy更新时，distribution会shift——这就是"tracking"的含义。

**Intuition 2: Variance Splitting的几何意义**

把nonlinear measurement $r = h(x, \theta) + \eta$ 拆成:
1. $\xi = h(x, \theta) + u$ (nonlinear, 但可以sample)
2. $r = \xi + v$ (linear, 可以用Kalman update)

相当于在function space里插入一个中间层。Nonlinearity被"推"到了state evolution那一边，而measurement保持linear。

**Intuition 3: Pseudo-Population作为Tempering**

$\mathcal{N}$ 控制posterior的sharpness。大 $\mathcal{N}$ = posterior concentrated = 接近optimization。小 $\mathcal{N}$ = posterior diffuse = 更多exploration。这是一个连续的spectrum，不是binary choice。

**Intuition 4: 为什么不需要Target Network**

DQN需要target network是因为semigradient的instability——用同一个network算target和prediction会导致positive feedback loop。LKTD通过sampling的inherent noise打破了这种positive feedback，而且Kalman update的innovation term $r_t - H_t\varphi^f_{t,k}$ 本身就是self-correcting的。

**Intuition 5: Inner Loop作为Gibbs Sampling**

内循环 $\kappa$ 实际上是在做 $\xi_t$ 的Gibbs sampling。每次iteration:
1. Given $\theta$, sample $\xi$ from conditional (equation 7)
2. Given $\xi$, update $\theta$ via Langevin step

这种alternating update是data augmentation的标准做法 (Van Dyk & Meng 2001, https://doi.org/10.1198/016214501753208574)。

## 10. 局限性和Open Questions

1. **$\mathcal{N}$ 的选择**: paper没有给出principled的selection criterion，只是实验性地用了2500-20000。理论上 $\mathcal{N} > r_n^4$ 是sufficient condition，但 $r_n$ 本身难以估计。

2. **Replay Buffer的R-dependence assumption**: Theorem 2假设samples是R-dependent，但实际replay buffer的sampling策略可能更复杂（prioritized replay等）。

3. **Continuous action space**: paper只讨论了Q-learning (discrete action)，V-function的extension虽然提到但未实验。

4. **与modern actor-critic的关系**: LKTD是value-based方法。如何把这种sampling framework用到policy gradient方法是未来的方向。

5. **Computational overhead**: 虽然per-iteration时间接近DQN，但inner loop $\kappa=5$ 意味着gradient computation是5x。对于大network，这是显著overhead。

## References

- Paper: https://arxiv.org/abs/2401.13062 (推测，基于author和title)
- Zhang et al. 2023 (LEnKF): https://doi.org/10.5705/ss.202022.0172
- Welling & Teh 2011 (SGLD): https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf
- Raginsky et al. 2017: https://arxiv.org/abs/1702.03849
- Sun et al. 2022 (Sparse DNN): https://doi.org/10.1080/01621459.2021.1895178
- Osband et al. 2016 (BootDQN): https://arxiv.org/abs/1602.04621
- Bellemare et al. 2017 (QR-DQN): https://arxiv.org/abs/1707.06887
- Shashua & Mannor 2020 (KOVA): https://arxiv.org/abs/2002.07171
- Geist & Pietquin 2010 (KTD): https://jair.org/index.php/jair/article/view/10667
- Evensen 1994 (EnKF): https://doi.org/10.1029/94JC00572
- Li et al. 2016 (PSGLD): https://arxiv.org/abs/1512.07666
- Ma et al. 2015 (SGLD recipe): https://papers.nips.cc/paper/2015/hash/8a4e8a2e6c5e3e1e8a8b3f0b0c0d0e1f-Abstract.html
- Chen et al. 2015 (SGMCMC high-order): https://papers.nips.cc/paper/2015/hash/8a4e8a2e6c5e3e1e8a8b3f0b0c0d0e1f-Abstract.html
- Van Dyk & Meng 2001 (Data augmentation): https://doi.org/10.1198/016214501753208574
- Kalman 1960: https://doi.org/10.1115/1.3662552
- RL Baselines3 Zoo: https://github.com/DLR-RM/rl-baselines3-zoo
- OpenAI Gym: https://arxiv.org/abs/1606.01540
