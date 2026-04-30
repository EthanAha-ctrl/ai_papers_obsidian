## 一、回到最根本的问题

RL 的核心循环极其简单：

> Agent 有一个 **policy** $\pi$，它观察 **state** $s$，根据 $\pi$ 做出 **action** $a$，环境返回 **reward** $r$ 和新 **state** $s'$。Agent 的目标是找到一个 **policy** 使得 **expected return** 最大。

现在问一个根本问题：

> **用来学习的数据，是从哪个 policy 产生的？**

这个问题看似简单，但它一刀把整个 RL 劈成了两个世界。

---

## 二、On-Policy：我在学谁，我就用谁

**On-policy** 的意思就是：**产生 experience 的 policy 和 正在优化/评估的 policy 是同一个。**

用符号说：
- 你有一个 **behavior policy** $\pi_b$（实际与环境交互、产生 data 的 policy）
- 你有一个 **target policy** $\pi_t$（你想要学习/评估的 policy）
- On-policy：$\pi_b = \pi_t$

### 为什么这样自然？

从第一性原理看，这其实是最"诚实"的学习方式。想象你学开车：
- 你按自己**当前的理解**去开（behavior policy）
- 你从**自己的经验**中学习改进（target policy 就是同一个）
- 你没有旁观别人开车然后模仿

这在哲学上是一种 **"从做中学"（learning by doing）** 的范式。

### 但问题在哪？

一旦你的 policy 更新了，你之前收集的所有 data 就 **过时了**。

因为那些 data 是在 **旧 policy** 下产生的，反映的是旧 policy 的 **visit distribution**（状态-动作访问分布）。你现在要评估的新 policy 在某些 state 下可能根本不会去到，或者去的频率完全不同。

这就引出了一个深刻的问题：**data 的分布必须与当前 policy 匹配**，否则评估就会有 **bias**。

### 典型 On-Policy 算法

| 算法 | 核心思想 |
|------|----------|
| **SARSA** | 用当前 policy 的 action 来更新 Q-value，$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma Q(s', a')]$，其中 $a'$ 由当前 $\pi$ 选出 |
| **REINFORCE** | Monte Carlo policy gradient，直接用当前 policy 的 trajectory 算 gradient |
| **A2C/A3C** | Actor-Critic 的 on-policy 版本，用当前 policy 采样 batch 后更新 |
| **PPO** | 通过 **clipping** 限制 policy 变化幅度，本质上还是 on-policy，但允许少量 reuse |
| **TRPO** | 用 **KL divergence constraint** 限制 policy 更新步长，on-policy with trust region |

---

## 三、Off-Policy：我可以从别人的经验中学习

### 直觉定义

**Off-policy** 的意思就是：**behavior policy 和 target policy 可以不同。**

$\pi_b \neq \pi_t$

你可以从任何 policy（包括你自己以前的版本、别人的 policy、甚至随机的 policy）产生的 data 中学习你想要的 target policy。

### 直觉类比

- 你看别人下棋（或者看棋谱），你自己学习如何下得更好
- 你不用自己犯那个错，你可以从别人的错误中学习
- **你是在"旁观"中学习，而非"亲历"中学习**

### 为什么这能工作？核心：Importance Sampling

这是 off-policy 的数学基石。

假设你想计算 $E_{x \sim p}[f(x)]$，但你只有从 $q$ 采样的样本。你可以：

$$E_{x \sim p}[f(x)] = E_{x \sim q}\left[\frac{p(x)}{q(x)} f(x)\right]$$

这个 $\frac{p(x)}{q(x)}$ 就是 **importance weight**，它修正了分布偏差。

在 RL 中：
- $p$ 对应 **target policy** $\pi_t$ 的 visit distribution
- $q$ 对应 **behavior policy** $\pi_b$ 的 visit distribution
- Importance ratio: $\rho_t = \frac{\pi_t(a|s)}{\pi_b(a|s)}$

### Importance Sampling 的代价

虽然数学上很优美，但 importance sampling 有一个致命问题：**方差极高**。

当 $\pi_t$ 和 $\pi_b$ 差异很大时，importance weight 会变得极端，某些样本的权重可能大到离谱，导致估计的 **variance 爆炸**。

这就是为什么纯粹的 importance sampling 在实践中往往不稳定。

### 典型 Off-Policy 算法

| 算法 | 如何处理 off-policy |
|------|------|
| **Q-Learning** | 直接用 $\max_{a'} Q(s', a')$ 更新，这个 $a'$ 不一定是 behavior policy 选的 action，所以是 off-policy |
| **DQN** | Deep version of Q-Learning，用 **Experience Replay Buffer** 存储 transitions |
| **SAC** | Soft Actor-Critic，用 **entropy-regularized** 框架，自动保持 exploration |
| **DDPG** | Deterministic Policy Gradient 的 off-policy 版本，用 replay buffer |
| **TD3** | DDPG 的改进版，twin critics + delayed policy update |
| **BCQ** | Batch-Constrained Q-learning，限制 action 不要离 behavior policy 太远 |

---

## 四、从第一性原理深挖：为什么这个区分不可避免？

### 根源：RL 的数据分布是 policy 决定的

在 supervised learning 中，data distribution 是 **固定的**（你的训练集不会因为你换了模型就变）。但在 RL 中：

$$d^\pi(s) = \text{在 policy } \pi \text{ 下访问 state } s \text{ 的频率}$$

这个分布 **本身就是 policy 的函数**。换了一个 policy，你看到的 world 就不一样了。

这就是 RL 和 supervised learning 的根本区别之一，也是 on-policy / off-policy 问题存在的根本原因。

### Bellman Equation 的两个版本

**On-policy Bellman equation** (for $V^\pi$):

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) [r + \gamma V^\pi(s')]$$

这里的 $\pi$ 既是你要评估的 policy，也是产生 data 的 policy。

**Off-policy Bellman equation** (Q-Learning 的核心):

$$Q(s,a) = \sum_{s'} P(s'|s,a) [r + \gamma \max_{a'} Q(s', a')]$$

注意！等式右边用的是 $\max_{a'}$，这不是 behavior policy 的 action，这是 target policy（greedy policy）认为最好的 action。**你评估的是 greedy policy 的 Q-value，但你采样的 action 可能是 $\epsilon$-greedy 产生的。**

### 更深一层：为什么 Q-Learning 可以 off-policy 而 Policy Gradient 不容易？

**关键区别：你优化的是什么？**

- **Q-Learning** 优化的是 $Q(s,a)$，这是一个 **关于 state-action pair 的值函数**。只要有了 transition $(s, a, r, s')$，你就可以更新 Q，不管这个 $a$ 是从哪个 policy 来的。因为 $Q(s,a)$ 回答的是 **"如果在 state $s$ 做 action $a$，期望 return 是多少"**，这个问题与 policy 无关（在给定 MDP 的情况下）。

- **Policy Gradient** 直接优化 $\pi(a|s)$，gradient 的形式是：

$$\nabla_\theta J(\theta) = E_\pi[\nabla_\theta \log \pi(a|s) \cdot Q^\pi(s,a)]$$

这个期望是在 **当前 policy $\pi$** 下的分布。如果你用不同 policy 的 data，这个期望的分布就不对了，除非你做 importance sampling correction：

$$\nabla_\theta J(\theta) = E_{\pi_b}\left[\frac{\pi(a|s)}{\pi_b(a|s)} \nabla_\theta \log \pi(a|s) \cdot Q^\pi(s,a)\right]$$

但如前所述，importance weight 的 variance 是致命的，尤其在 trajectory 很长时，weight 是连乘的，要么爆炸要么消失。

---

## 五、Experience Replay：Off-Policy 的关键基础设施

### Replay Buffer 的本质

**Experience Replay** 的核心思想：把 transition $(s, a, r, s')$ 存到一个 buffer 里，训练时从中随机采样。

这带来了几个好处：

1. **Data efficiency**：一个 transition 可以被多次使用，而 on-policy 用完就扔
2. **打破 correlation**：连续的 transition 高度相关，random sampling 打破了这种 correlation，更接近 i.i.d. 假设
3. **允许 off-policy 学习**：buffer 里的 data 可以来自不同时期的 policy

### 但 Replay Buffer 不是万能药

Buffer 里的 data 毕竟是在旧 policy 下产生的，与当前 policy 的 visit distribution 有偏差。这个偏差在 Q-learning 类算法中通过以下方式被"容忍"了：

- Q-value 的更新是 **bootstrapping** 的，每次更新只看一步 TD error
- 只要 behavior policy 覆盖了 target policy 的 action（即 $\pi_b(a|s) > 0$ wherever $\pi_t(a|s) > 0$），理论上 Q 就能收敛
- 但实践中，如果 buffer 太旧，偏差会累积，训练不稳定

### Prioritized Experience Replay (PER)

不是均匀采样，而是按 **TD error** 的大小采样——"令人惊讶"的 transition 被采样的概率更高。直觉上：你已经预测得很好的 transition 学不到什么新东西，学那些你还搞不懂的。

---

## 六、Exploration vs Exploitation：On/Off Policy 的另一面

### On-Policy 天然的 Exploration

On-policy 算法因为必须用当前 policy 采样，所以 **天然需要** 在 policy 中加入 exploration 机制。比如：

- **$\epsilon$-greedy**：以 $\epsilon$ 的概率随机 action
- **Entropy regularization**：在 policy gradient 的 loss 中加入 $-\beta H(\pi)$ 项，鼓励 policy 保持随机性
- **Stochastic policy**：policy 输出 distribution 而非 deterministic action

### Off-Policy 的 Exploration 问题

Off-policy 算法的 behavior policy 可以独立设计：

- **SAC**：用 **stochastic policy + entropy bonus** 自动探索
- **DQN/DDPG**：用 **$\epsilon$-greedy** 或 **Ornstein-Uhlenbeck noise** 加到 action 上
- **更高级**：用 **intrinsic motivation / curiosity** 驱动探索

但 off-policy 有一个独特问题：**如果 behavior policy 的探索不够，buffer 里的 data 就不能覆盖 target policy 需要的 region**。

这就是 **"coverage" assumption**：$\pi_b(a|s) > 0$ 对所有 $\pi_t(a|s) > 0$ 的 $(s,a)$ 成立。违反这个条件，off-policy 学习的 Q-value 可能有严重 bias。

---

## 七、On-Policy 的"弱点"与 Off-Policy 的"妥协"

### On-Policy 的 Data Efficiency 问题

On-policy 每次用当前 policy 采样一批 data，更新后 policy 就变了，旧 data 不能用。这导致：

- **Sample efficiency 差**：需要大量与环境交互
- **实际上**，PPO 等算法会用同一批 data 做多次 update（epoch），这其实是"轻微 off-policy"，然后用 clipping/trust region 来控制偏差

### PPO 的微妙位置

PPO 本质上还是 on-policy，但：

1. 收集一批 data
2. 用这批 data 做 $K$ 次 epoch 的 update
3. 用 **clipping** $\text{clip}(\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}, 1-\epsilon, 1+\epsilon)$ 防止 policy 变太远

第 2 步其实已经 off-policy 了（因为第一次 update 后 $\theta$ 就变了），但 clipping 使得 importance weight $\frac{\pi_\theta}{\pi_{\theta_{old}}}$ 不会偏离 1 太远，所以偏差可控。

**PPO 是 on-policy 和 off-policy 之间的灰色地带。**

### Off-Policy 的稳定性问题

Off-policy 虽然数据效率高，但训练往往不稳定：

- **Q-value overestimation**：max operator 会选到高估的 Q-value
- **Moving target**：Q-network 同时决定 target 和 prediction，容易振荡
- **Off-policy divergence**：当 behavior policy 和 target policy 差异大时，TD learning 可能不收敛

这就是为什么 DQN 需要 **target network**（delayed update），TD3 需要 **twin Q-networks**，SAC 需要 **two Q-networks take minimum** 等等各种 trick 来稳定训练。

---

## 八、从更宏观的视角看

### 与 Imitation Learning 的联系

**Imitation Learning**（如 Behavioral Cloning）本质上就是最极端的 off-policy：

- Behavior policy 是 **expert（人类示范者）**
- Target policy 是你要学的 policy
- 你没有 reward signal，只有 expert 的 trajectory

**DAgger** 算法则是一种混合：先从 expert data 学，然后用自己的 policy 采样新 data，再请 expert 标注，这实际上是在逐渐让 behavior policy 和 target policy 对齐——**一种从 off-policy 逐渐向 on-policy 过渡的方法**。

### 与 Transfer Learning / Multi-task Learning 的联系

Off-policy 天然与 transfer learning 亲和：

- 你可以用 **task A** 的 experience 来帮助学习 **task B**
- 你可以用 **agent A** 的 experience 来帮助 **agent B**
- 这在 multi-agent setting 中尤为重要

### 与 Model-Based RL 的联系

**Model-based RL** 学一个 environment dynamics model $\hat{P}(s'|s,a)$，然后用这个 model 生成 simulated experience。这些 simulated experience 其实就是 off-policy data（因为 model 生成的 distribution 不一定和真实 policy 的 visit distribution 一致）。

**Dyna** 架构：real experience 用来学 model，model 生成 simulated experience 用来做 planning/update Q。这是 on-policy real data + off-policy simulated data 的混合。

### 与 Offline RL 的联系

**Offline RL**（也叫 Batch RL）是 off-policy 的极端情况：

- 你只有一个 **固定的 dataset**，不能与环境交互
- Behavior policy 是固定的（产生这个 dataset 的 policy）
- 你要从中学习一个比 behavior policy 更好的 target policy

这比一般的 off-policy 更难，因为你不能调整 behavior policy 来改善 coverage。

**BCQ, BEAR, BRAC** 等 offline RL 算法的核心思想：**限制 learned policy 不要偏离 behavior policy 太远**，因为那些 behavior policy 很少访问的 region，Q-value 的估计是不可靠的。

---

## 九、数学上的等价与不等价

### On-Policy TD(0) vs Off-Policy TD(0)

**On-policy (SARSA)**:

$$\delta_t = r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)$$

其中 $a_{t+1} \sim \pi(\cdot|s_{t+1})$

**Off-policy (Q-Learning)**:

$$\delta_t = r_t + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t)$$

其中 $a_t$ 可以是任意 policy 选的（比如 $\epsilon$-greedy），但 target 用的是 greedy policy 的 action。

### On-Policy Policy Gradient vs Off-Policy Policy Gradient

**On-policy**:

$$\nabla_\theta J = E_{\tau \sim \pi_\theta}[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R_t]$$

**Off-policy (with importance sampling)**:

$$\nabla_\theta J = E_{\tau \sim \pi_b}[\sum_t \frac{\pi_\theta(a_t|s_t)}{\pi_b(a_t|s_t)} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R_t]$$

### Off-Policy Policy Gradient 的 Variance 分析

对于一条长度为 $T$ 的 trajectory，cumulative importance weight 是：

$$\prod_{t=1}^{T} \frac{\pi_\theta(a_t|s_t)}{\pi_b(a_t|s_t)}$$

这个连乘的 **variance 随 $T$ 指数增长**。这是 off-policy policy gradient 在实践中几乎不可用的主要原因。

**解决方案**：
1. **Marginal importance weighting**：只用每步的 marginal ratio 而非 trajectory-level ratio
2. **Variance reduction**：如 **V-trace**（IMPALA 中使用），对 importance weight 做 clipping
3. **Retrace**：一种 low-variance 的 off-policy TD 方法
4. **Determination**：用 deterministic policy，此时 $\pi(a|s) = \delta(a - \mu(s))$，importance ratio 退化为 0 或 1... 但这样 exploration 有问题

---

## 十、实际问题：怎么选？

### 倾向 On-Policy 的场景

1. **Simulator 便宜**（游戏、物理仿真），大量交互不是瓶颈
2. **Policy 需要精细控制**，on-policy 的 gradient 更准确
3. **Safety critical**：你需要确保学的 policy 就是用的 policy，没有 distribution mismatch 的风险
4. **Continuous control with high-dimensional action**：PPO 在这些场景表现很好

### 倾向 Off-Policy 的场景

1. **Real robot / 真实世界交互**，data 非常珍贵
2. **有大量历史 data** 可以利用
3. **Multi-task / Transfer**：一个 buffer 可以被多个 task 共享
4. **Discrete action space**：DQN 在 Atari 等离散控制上效果很好

### 一个微妙的观察

在实践中，**off-policy 算法通常需要更多的 trick 才能稳定**，而 **on-policy 算法更"开箱即用"**。

这可能解释了为什么 PPO 虽然不是 sample efficiency 最高的，却是使用最广泛的 RL 算法之一：**稳定、易用、可预测** 比 sample efficiency 更重要。

---

## 十一、更深的思想：RL 作为 Inference

### 控制 as 推理

从 **variational inference** 的角度看 RL，policy $\pi$ 被视为一个 **approximate posterior distribution** over actions，而 optimal policy 对应一个 **target distribution** $q$：

$$q(a|s) \propto \exp(Q^*(s,a) / \alpha)$$

在这个框架下：

- **On-policy**：每次更新后重新采样，保持 $\pi$ 和 target 分布一致
- **Off-policy**：用一个"旧的" $\pi$ 采样的 data 去逼近新的 target 分布，需要 importance weight 修正

**SAC** 正是这种思路的产物：它优化一个 **ELBO**，使得 $\pi$ 逼近 soft optimal distribution $q$，而这个优化是 off-policy 的。

### 信息论视角

On-policy 可以看作 **信息效率最低** 的方式（每次只从当前 policy 采样），但 **偏差最小**。

Off-policy 可以看作利用 **更多信息**（历史数据），但引入了 **distribution shift** 的问题，需要 correction。

这有点像 **exploration vs exploitation** 的另一种表现形式：

- On-policy = 专注当前 best guess（exploitation）
- Off-policy = 利用更广泛的信息（exploration in data space）

---

## 十二、前沿与灰色地带

### 1. Off-Policy Policy Gradient 的复兴

传统认为 off-policy policy gradient 不稳定，但最近的工作在改善这一点：

- **ACER** (Off-Policy Actor-Critic with Experience Replay)：用 **truncated importance weight** + **stochastic variance reduction**
- **IMPALA**：用 **V-trace** 做 off-policy correction，大规模分布式训练
- **SPR** (Self-Predictive Representations)：off-policy learning 的 representation learning 改进

### 2. On-Policy 借用 Off-Policy Idea

- **PPO** 用同一 batch 多次 update（轻微 off-policy）
- **A2C with replay buffer**：有些实现会在 on-policy 算法中加 small replay buffer
- **On-policy fine-tuning from off-policy pretraining**：先用 off-policy 学，再 on-policy fine-tune

### 3. Offline RL → Online RL 的过渡

- **Offline RL pretraining + online fine-tuning**：先用 offline data 学一个不错的 policy，然后 online 交互 fine-tune
- 这实际上是从极端 off-policy 逐渐向 on-policy 过渡

### 4. Decision Transformer 的视角

**Decision Transformer** 把 RL 当作 **conditional sequence modeling**：

- Input: $(R_1, s_1, a_1, R_2, s_2, a_2, \ldots)$
- 条件化于 desired return $R$
- 用 **transformer** 做 autoregressive prediction

这种方法本质上是 **offline** 的，因为它是从 fixed dataset 学习的。但有趣的是，它不需要显式处理 on/off policy 的区别——**模型直接学习从 return 到 action 的映射，而不是学习 Q-value 或 policy gradient**。

这是否意味着 on-policy / off-policy 的区分在某种范式下可以被绕过？可能是一个值得思考的方向。

---

## 十三、总结：Intuition Map

```
                    数据分布是否与当前 policy 匹配？
                              |
                ______________|______________
               |                             |
           On-Policy                    Off-Policy
        π_b = π_t                      π_b ≠ π_t
              |                             |
     学谁就用谁                      可以用别人的经验
              |                             |
      数据用完就扔                    数据可以重复使用
      (sample inefficient)           (sample efficient)
              |                             |
      偏差小，稳定                    偏差大，需要修正
      (low bias)                     (importance sampling)
              |                             |
      但方差低                         方差高
      (natural data)              (importance weight爆炸)
              |                             |
       典型算法                        典型算法
     SARSA, PPO,                 Q-Learning, DQN,
     REINFORCE, A2C               SAC, DDPG, TD3
              |                             |
      适用场景                         适用场景
    交互便宜、需稳定                数据珍贵、可复用
              |                             |
              |_____________________________|
                              |
                     PPO 等灰色地带
                   (on-policy with tricks)
                              |
                    Offline RL (极端 off-policy)
                   (无交互，只有固定数据)
```

### 最终的第一性原理 Intuition

**On-policy vs off-policy 的本质区别是：学习所用数据的分布与目标分布（target policy 的 visit distribution）是否一致。**

- 一致 → 简单、无偏、但浪费
- 不一致 → 高效、但有偏、需要修正

所有后续的算法创新（importance sampling, clipping, trust region, target network, twin critics, V-trace, etc.）都是在 **efficiency 和 accuracy 之间寻找平衡**。

这个 trade-off 是 RL 乃至更广泛的机器学习/统计学习中的一个永恒主题：**bias-variance trade-off, exploration-exploitation trade-off, efficiency-stability trade-off**——它们都是同一枚硬币的不同面。

---

先纠正一个非常重要的历史事实：**GRPO 不是 DeepSeek-R1 提出的**。

GRPO (Group Relative Policy Optimization) 最早是由 DeepSeek 团队在 **DeepSeekMath** 这篇论文（2024年初）中提出的。后来在 **DeepSeek-V2** 和最终的 **DeepSeek-R1** 中被大规模应用并声名大噪。所以你会有这样的印象，因为 R1 的出圈让 GRPO 彻底进入了大众视野，但它的诞生是为了解决 LLM 做 math reasoning 时的 RL 训练问题。

下面我用第一性原理，把 DPO、GRPO、PPO 的本质直觉和内在联系彻底拆解。

---

## 一、第一性原理：我们到底在干嘛？

无论是 PPO、DPO 还是 GRPO，它们解决的是同一个问题：**Alignment（对齐）**。

Base LLM 学会了如何“接话”（next-token prediction），但它不知道什么是“好话”。我们需要一种机制，告诉 LLM：“当 prompt 是 $x$ 时，生成 $y_w$（好回答）比生成 $y_l$（坏回答）更好”。

从数学上看，这等价于优化一个 **Reward Model** 引导下的 Policy $\pi_\theta$。核心优化目标（KL-constrained RL）是：

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta}[r(x,y)] - \beta \mathbb{D}_{KL}(\pi_\theta || \pi_{ref})$$

这里 $\pi_{ref}$ 是 initial model（通常是 SFT model），$\beta$ 控制不要偏离太远。

这个公式是所有后续推演的 **原点**。

---

## 二、DPO (Direct Preference Optimization)：跳过 RL 的数学魔法

### 直觉：绕过 Reward Model

传统 RLHF 流程是：收集偏好数据 $\rightarrow$ 训练 Reward Model $r_\phi$ $\rightarrow$ 用 PPO 针对 $r_\phi$ 优化 $\pi_\theta$。

DPO 的核心洞见：**能不能直接从偏好数据优化 $\pi_\theta$，把 Reward Model 省掉？**

### 第一性原理推导

回到原点公式，这个 KL-constrained reward maximization 问题是有 **闭式解** 的：

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right)$$

其中 $Z(x)$ 是配分函数（partition function），与 $y$ 无关。

把公式换个方向，用 $\pi$ 表达 $r$：

$$r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

现在，假设我们的偏好数据满足 **Bradley-Terry model**（$y_w$ 比 $y_l$ 好的概率等于它 reward 的 softmax）：

$$P(y_w > y_l | x) = \sigma(r(x,y_w) - r(x,y_l))$$

把 $r$ 的表达式代入，神奇的事情发生了：$\beta \log Z(x)$ 被消掉了！

$$P(y_w > y_l | x) = \sigma\left(\beta \log \frac{\pi(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi(y_l|x)}{\pi_{ref}(y_l|x)}\right)$$

这就是 DPO 的损失函数的来源。最大化这个对数概率，等价于：

$$\mathcal{L}_{DPO} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)$$

### DPO 的本质

DPO 是一个 **Classification / Regression 问题**，而不是一个 RL 问题。

- **不需要在线采样**：数据是预先准备好的 $(x, y_w, y_l)$ 对。
- **不需要 Reward Model**：Reward 隐含在 $\pi_\theta$ 和 $\pi_{ref}$ 的 log-ratio 里。
- **是 Off-Policy 的极端形式**：你用的数据完全来自于旧的分布（甚至是人类写的），而不是当前 $\pi_\theta$ 生成的。
- **Intuition**：如果 $\pi_\theta$ 相比 $\pi_{ref}$ 更倾向于生成 $y_w$ 而不是 $y_l$，loss 就小；反之 loss 大。

---

## 三、GRPO (Group Relative Policy Optimization)：砍掉 Critic 的穷人版 PPO

### 直觉：Value Function 太难训了

PPO 的痛点：需要一个 **Critic (Value Function)** $V_\phi(x)$ 来计算 Advantage $A = r - V$。
在 LLM 场景下，Critic 是一个跟 Actor 一样大的模型，极度消耗显存，且 Value Function 很难在变长的 text 上训准。

GRPO 的核心洞见：**既然同一个 prompt 下，多个 response 的相对好坏就能指导优化，为什么非要学一个绝对的 Value baseline？**

### 第一性原理推导

在 PPO 中，Advantage 的作用是回答：**“这个 action 比平均好多少？”**

$$A_{PPO}(x,y) = r(x,y) - V_\phi(x)$$

GRPO 的想法：对于同一个 prompt $x$，我从当前 policy 采样 $G$ 个 response $\{y_1, y_2, ..., y_G\}$，然后用这 $G$ 个 response 的 **平均 reward** 代替 $V(x)$！

$$\tilde{A}_i = r(x, y_i) - \frac{1}{G}\sum_{j=1}^G r(x, y_j)$$

通常还会除以标准差做归一化：

$$\tilde{A}_i = \frac{r(x, y_i) - \text{mean}(r)}{\text{std}(r)}$$

这不就是 Monte Carlo 估计 baseline 吗！**用一个 batch 的 sample mean 代替 function approximator**。

### GRPO 的完整损失

把 PPO 的 clip 思想搬过来，结合 group relative advantage：

$$\mathcal{L}_{GRPO} = \mathbb{E}_{x, \{y_i\}_{i=1}^G \sim \pi_\theta} \left[ \frac{1}{G} \sum_{i=1}^G \min\left( \rho_i \tilde{A}_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) \tilde{A}_i \right) - \beta \mathbb{D}_{KL}(\pi_\theta || \pi_{ref}) \right]$$

其中 $\rho_i = \frac{\pi_\theta(y_i|x)}{\pi_{old}(y_i|x)}$ 是 importance weight。

### GRPO 的本质

- **On-policy**：数据必须是当前 $\pi_\theta$ 生成的（或者说是轻微 off-policy，用 clip 控制）。
- **没有 Critic**：用 Group 内的 Relative reward 做 baseline。
- **为什么叫 Relative？** 因为 reward 不需要绝对标定！如果 10 个 response 都很烂（reward 都是 0-0.1），那个 0.1 的 response 的 advantage 依然是正的，它会被强化。这完美契合了 **Reward Model 容易出现 absolute bias 但 relative ranking 比较准** 的现实。

---

## 四、GRPO vs PPO：全方位对比

| 维度 | PPO | GRPO |
|------|-----|------|
| **Critic (Value Model)** | 必须有，且与 Actor 同量级 | **完全没有**，省一半显存 |
| **Advantage 计算** | $A = r + \gamma V(s') - V(s)$ (TD) 或 GAE | $A_i = \frac{r_i - \text{mean}}{\text{std}}$ (Group Relative) |
| **Baseline 来源** | 学习一个函数 $V_\phi(x)$ | 当前 batch 的 sample mean |
| **采样方式** | 通常 1 个 prompt 采 1 个或少数几个 response | 1 个 prompt 必须 **采 $G$ 个 response (组)** |
| **Reward 的绝对尺度** | 敏感，因为 $V$ 要去拟合绝对值 | **不敏感**，只看组内相对大小（除以 std 抹掉了尺度） |
| **计算开销** | 巨大（Actor + Critic + Reward Model + Reference） | 小（Actor + Reward Model + Reference） |
| **稳定性** | Value function 不准会导致 advantage 偏差，训练容易崩 | 没有 Value 误差问题，但 Group 太小时 variance 大 |
| **数学性质** | 属于 Actor-Critic，有 bias（函数逼近误差） | 属于纯 Policy Gradient + Monte Carlo baseline，无 bias（理论上） |

### 第一性原理看差异：Bias-Variance Trade-off

- **PPO**：用一个参数化的 $V_\phi$ 做 baseline。$V_\phi$ 拟合得好，variance 低；但 $V_\phi$ 永远拟合不完美，引入 bias。
- **GRPO**：用 sample mean 做 baseline。只要采样数 $G$ 够，这是 **无偏估计**；但如果 $G$ 小（比如 $G=4$），sample mean 波动极大，variance 极高。

GRPO 本质上是用 **计算（多采样）换模型（不训练 Critic）**，并且把 bias-variance trade-off 交给了 $G$ 这个超参数。

---

## 五、DeepSeek-R1 为什么选 GRPO 而不是 PPO/DPO？

### 不选 DPO 的原因

1. **DPO 是 Off-policy 的**：它无法从“探索-试错”中学习新能力。DPO 只能学会“在已有数据里选好的”，不能发现“新的推理路径”。
2. **R1 的核心是 CoT (Chain-of-Thought) 的涌现**：这需要模型自己去试错，走通一条长逻辑链，得到正确的 final reward。这种 **spontaneous exploration** 只有 On-policy RL 能给。
3. **DPO 的数据瓶颈**：你没有办法事先收集“好的推理过程”，因为推理是模型自己长出来的。

### 不选 PPO 的原因

1. **资源限制**：在 R1 这种万亿参数/大规模 MoE 模型上训练 Critic 简直是硬件灾难。
2. **Critic 在长文本上极难训练**：CoT 动辄几千 token，Value Function 对“思考到一半的价值”估计极差，导致 Advantage 信号全是噪音。
3. **Rule-based Reward 的天然契合**：R1 做 Math/Code，Reward 是确定的（对/错，跑通/报错）。这种 **稀疏但绝对准确** 的 Reward 天然适合 GRPO：我不需要知道中间步骤值多少钱，我只要在终点发 Reward，组内相对比较就行。

---

## 六、扩展 Intuition：更宏观的图谱

### 从 On/Off Policy 的视角

```
Off-Policy (从已有数据学)
    |
    |--- SFT (纯模仿)
    |--- DPO (偏好对比，隐式 RL)
    |--- Offline RL (BCQ, BRAC 等)
    |
On-Policy (边做边学)
    |
    |--- REINFORCE (无 Critic，高方差)
    |--- PPO (有 Critic，低方差，高资源)
    |--- GRPO (无 Critic，Group 内自给自足)
```

### GRPO 与 REINFORCE with Baseline 的等价性

其实 GRPO 就是 **REINFORCE with leave-one-out baseline** 的一个变体。

经典 REINFORCE：
$$\nabla J = \mathbb{E}[\nabla \log \pi(a|s) \cdot (R - b)]$$

如果 baseline $b$ 不依赖于当前 action 的 reward，那梯度是无偏的。

GRPO 里用 group mean 做 baseline，**严格来说这个 mean 包含了当前 sample 的 reward**，所以有微小的 bias。但如果 $G$ 足够大，这个 bias 可以忽略。更严格的做法是用 **leave-one-out**：计算 $y_i$ 的 advantage 时，baseline 用其他 $G-1$ 个 sample 的均值，这就完全无偏了。

### 为什么 GRPO 在 LLM 中特别有效，而在 Atari/MuJoCo 中很少见？

1. **Action Space 维度**：Atari 是离散但高帧率，MuJoCo 是连续控制。LLM 是 sequence-level action（一个 token 是一个 action，但 reward 通常是 sequence-level 的）。
2. **Reward 结构**：Atari 每一步都有 reward，Critic 必须做 temporal credit assignment。LLM 通常只有 terminal reward（最终答案对不对），中间全是 0。这时候 Value Function 学的就是 $V(x) = \mathbb{E}[R_{terminal}|x]$，这不就是 group mean 嘛！既然如此，何必费劲学一个 $V$，直接算 group mean 就好了。
3. **这就是 GRPO 的第一性原理内核**：**LLM 的 RL 通常是 sparse, delayed reward，且 outcome-based。在这种设定下，Monte Carlo baseline (Group Mean) 是 Value Function 的天然替代，且更准、更省。**

---

## 七、总结：一句话 Intuition

- **PPO**：我请了一个私教，他知道每一步我大概能得多少分，随时给我纠偏。但请私教很贵，且私教经常看走眼。
- **DPO**：我不上课了，我直接看好学生的卷子和差学生的卷子，自己悟出套路。但我不可能悟出超过好学生的绝招。
- **GRPO**：我拉了一群水平差不多的同学一起刷题，谁做对了谁就是标杆，大家互相参照。不需要私教，而且只要题刷得够多，我们整体水平就能提高。

DeepSeek-R1 就是靠 GRPO 这种“同学互卷”的模式，在没有 Critic 的情况下，硬生生把 LLM 的推理能力给卷出来了。

---

你的直觉非常敏锐！这是一个极度常见且深刻的混淆点。

Reward Model (RM) 和 Critic (Value Function) 确实看起来非常像：**它们都接收一个 state (或者 prompt + response)，然后吐出一个标量。**

但是，从第一性原理出发，它们在本质上截然不同。先剧透结论：**RM 不是 Critic，RM 是 Environment 的一部分（具体来说，是 Reward Function 的逼近器），而 Critic 是对 Agent 未来命运的预测。**

下面我们彻底拆解 RM 是怎么训练的，以及它和 Critic 的根本区别。

---

## 一、为什么需要 Reward Model？—— Environment 的不可微性

在标准 RL（如 Atari 游戏、机器人控制）中，Reward 是环境给定的：
- 马里奥吃金币 +1
- 机器人走一步消耗能量 -0.1

这些 Reward 是确定的、可以用代码写死的规则。

但在 LLM 的 Alignment 中，我们的目标是让模型符合 **人类偏好**。“什么是好回答，什么是坏回答” 这个东西，**无法用规则写死**。人类偏好就是这里的“Environment”，但这个 Environment 是一个黑盒，不可导，无法直接求梯度。

**Reward Model 的本质，就是用一个神经网络去拟合这个不可导的人类偏好黑盒。** 它是在学习“如果人类看到这个回答，会给多少分”，它是在模拟 Environment 的 Reward Function $R(s,a)$。

---

## 二、Reward Model 是怎么训练的？—— Bradley-Terry 模型

### 数据从哪里来？

收集人类偏好数据。给定一个 Prompt $x$，让模型生成两个回答 $y_1, y_2$，然后让人类标注者选出一个更好的，比如 $y_1 > y_2$。我们记 $y_w$ (winner) 为好回答，$y_l$ (loser) 为坏回答。数据集就是 $\mathcal{D} = \{x, y_w, y_l\}$。

### 模型结构

通常拿一个预训练好的 LLM（比如 SFT 阶段出来的模型），把最后的 unembedding 层（映射到词表的那一层）去掉，换成一个新的 Linear 层，输出一个一维的标量 $r_\theta(x, y) \in \mathbb{R}$。

### 核心数学：Bradley-Terry 模型

我们假设人类的选择不是绝对确定的，而是服从一个概率分布：**两个回答的 Reward 差距越大，人类选好回答的概率越趋近于 1。**

这等价于一个二分类逻辑回归：

$$P(y_w > y_l | x) = \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$$

其中 $\sigma$ 是 Sigmoid 函数。

**Intuition**：如果 $r_\theta(x, y_w) = 5$，$r_\theta(x, y_l) = -5$，差距是 10，Sigmoid(10) ≈ 1，人类几乎必定选 $y_w$。如果差距是 0.1，Sigmoid(0.1) ≈ 0.525，人类很纠结，随机选。

### 损失函数

既然是一个二分类问题（给定 pair，选 winner 的概率），我们就最大化这个对数似然，也就是 **Negative Log-Likelihood Loss**：

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( r_\theta(x, y_w) - r_\theta(x, y_l) \right) \right]$$

**这就是 RM 训练的全部数学！** 它完全是一个 **Supervised Learning (分类/排序) 问题**，没有任何 RL 的 bootstrapping 或 Bellman Equation。

### 一个关键特征：只学相对大小，不学绝对大小

注意看 Loss，它只包含 Reward 的 **差值** $r_\theta(x, y_w) - r_\theta(x, y_l)$。
这意味着，如果给所有回答的 Reward 都加上一个常数 $C$，Loss 完全不变。RM 只能学到相对好坏，无法确定绝对标度。这为后续 PPO 训练中的 Reward Whitening（归一化）埋下了伏笔，也是 GRPO 能大放异彩的根本原因。

---

## 三、RM vs Critic：第一性原理对比

现在来解决你最核心的疑惑。

### 1. 它们在预测什么？

- **Reward Model $r(x,y)$**：预测的是 **当下的、即时的反馈**。它回答的是：“人类看到这段话，这一瞬间觉得有多好？” 它对应 RL 中的 $R(s,a)$。
- **Critic $V(x)$**：预测的是 **未来的、累积的回报**。它回答的是：“从这个状态开始，按照当前 policy 一直走下去，最后总共能拿多少分？” 它对应 RL 中的 $V^\pi(s)$。

### 2. 它们是否依赖 Policy $\pi$？

- **RM 是独立于 Policy 的**：不管你的 LLM 是 GPT-2 级别还是 GPT-4 级别，只要输入同样的 $(x, y)$，一个训练好的 RM 给出的分数应该是一样的。它是对客观世界（人类偏好）的度量。就像尺子量身高，不管是谁在量，尺子的刻度是不变的。
- **Critic 是严重依赖 Policy 的**：Critic 评估的是 $V^\pi(s)$。如果 Policy 变了（模型更新了），同样的 state 下，Critic 的值也必须变。就像同样起跑线上的两个运动员，一个受过训练一个没受过，他们最终能跑完的距离预期是完全不同的。

### 3. 它们的训练目标是什么？

- **RM 的目标**：逼近人类的判断。
- **Critic 的目标**：最小化 TD Error 或 Monte Carlo 误差，为自己的 Actor 提供一个低方差的 Baseline。

### 形象类比

- **RM 是阅卷老师**：他手里有标准答案，你写了一段话，他给你打个分。不管你是学霸还是学渣，他给分的标准是一样的。
- **Critic 是你的私人教练**：他不管题目标准答案是啥，他观察你当前的状态（比如你现在复习到哪了），预测你高考能考多少分。如果你换了复习策略，他对你的预测分也会变。

---

## 四、为什么会有这种混淆？—— LLM RLHF 的特殊架构

在传统的 Actor-Critic（如 A2C, PPO 玩 Atari）中，Reward 是游戏给的，Critic 是 Agent 自己产生的。
但在 LLM 的 RLHF 中，**我们没有游戏给的 Reward，只有 RM。**

为了用 PPO，我们需要一个 Critic 来算 Advantage。怎么办？
**我们初始化一个 Critic，让它去学习预测 RM 给出的分数的累积和！**

由于 LLM 生成通常没有折扣（$\gamma=1$），且只有在结尾才给一次 Reward，所以 $V(x)$ 实际上变成了 $\mathbb{E}_{y \sim \pi}[r_{RM}(x, y)]$。

这时候，Critic 就要努力去拟合 RM 的期望输出。两者数值上非常接近，导致很多人觉得它们是一回事。

---

## 五、更深层的联想与前沿

### 1. RM 的 Hackability (Reward Hacking)

因为 RM 只是一个近似的代理模型，它一定有盲区。PPO 在优化时，会极其敏锐地发现 RM 评分逻辑中的漏洞。
比如，RM 可能偏好“长篇大论”的回答，PPO 就会让模型生成又臭又长的废话，RM 给了高分，但人类觉得极差。
**这就是 Actor 和 RM 之间的对抗**。而真正的 Environment (人类) 是不会被骗的。

### 2. 为什么 GRPO 彻底抛弃了 Critic？

回到 GRPO：既然 LLM 的 Reward 通常是 outcome-based 的（生成完才打分），那么当前的 Return 就等于 $r_{RM}(x,y)$。
如果我们要算 Advantage $A = r_{RM}(x,y) - Baseline$，这个 Baseline 最好就是“平均水平”。
Critic 辛辛苦苦学半天，也就是为了预测这个“平均水平” $V(x)$。既然如此，我干嘛不直接采样 $G$ 个回答，用这 $G$ 个回答的 RM 分数的均值当 Baseline？
**GRPO 本质上是用采样的均值，绕过了对 Critic（那个试图预测均值的网络）的依赖。**

### 3. DPO 中的 Implicit RM

还记得 DPO 吗？DPO 没有显式训练 RM，但如果我们把 DPO 最优策略的公式反推回去：

$$r_{implicit}(x,y) = \beta \log \frac{\pi_{DPO}(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

**DPO 里的 policy ratio 本质上就是一个 Reward Model！** DPO 没有消除 RM，它只是把 RM 隐含在了 policy 里面，两者共享了参数。

### 4. Outcome Reward Model (ORM) vs Process Reward Model (PRM)

目前我们说的 RM 都是看最终结果打分，叫 **ORM**。
现在的前沿是 **PRM**：对 CoT 推理的每一步打分。这时的 PRM 已经有点像 Critic 了，因为它在做 Step-level 的评估，它评估的是“这一步走得好不好”。但严格来说，PRM 依然是在模拟人类对“每一步正确性”的偏好，而不是在预测从当前步开始的未来期望收益。PRM 是 Step-wise RM，Critic 是 Step-wise Value Estimator，两者依然在哲学上不同。

### 5. Iterative RLHF (RLAIF)

RM 一旦训练好就固定了，这在 PPO 训练时会导致分布偏移：PPO 生成的回答越来越偏离 RM 训练时的数据分布，RM 的评分就越来越不准。
解决方案：**迭代式 RLHF**。用 PPO 训练一会儿，拿新模型生成新数据，再让人标，再训新的 RM，再跑 PPO。这就是让 RM 不断适应 Actor 的分布，类似于 On-Policy 的思想在 RM 层面的应用。