---
source_pdf: Reinforcement Learning An Overview.pdf
paper_sha256: 15f481922acde9c4ede73da12783c823bde17ccb5f01af1343166fdb2f56d987
processed_at: '2026-08-11T22:14:14-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# RL 的人话版

好，让我用更直觉的方式讲一遍。把那些公式和术语都还原成"为什么"和"它在解决什么问题"。

## 1. RL 到底在干什么

先说一个类比。Supervised learning 像在教室学做题——题目给好，标准答案给好，你预测，错了扣分。Reinforcement learning 像在野外学骑自行车——没人告诉你每一步该干嘛，你只能 try，摔了知道疼，骑远了知道爽，但你不知道刚才摔倒是因为三秒前那个拐弯动作错了，还是因为五分钟前胎压没打够。

RL 的核心 difficulty 就藏在这个类比里：

**Credit assignment problem**：reward 是延迟的、稀疏的、笼统的。你赢了棋，是哪一步的功劳？你输了游戏，是哪一帧的操作出问题？这个"把最终结果归因到每一步动作"的问题，是 RL 区别于 SL 的本质。

**Exploration vs exploitation**：你已经知道往左走能拿 10 分，但你不知道往右走是 100 分还是 -100 分。你要不要试？试多少次？什么时候停？

**Partial observability**：你看到的不是世界的全貌。在迷宫里你只能看到前面的墙，在 poker 里你不知道对手的牌。你必须基于不完整信息推断真实 state。

**Non-stationarity**（multi-agent）：其他 agent 在学习，你的"环境"在变。你今天的最优 policy 明天就废了。

所有 RL 算法，本质上都是在用不同方式回答这四个问题。

## 2. Bellman equation 为什么是 RL 的一切

先从最根本的 equation 讲起。如果你在 state $s$，做了 action $a$，得到 reward $r$，然后到了 $s'$，那么：

$$Q(s, a) = r + \gamma \max_{a'} Q(s', a')$$

变量说明：$Q(s,a)$ 是"在 state $s$ 做 action $a$ 之后，从今往后能拿到的总 reward"。$\gamma \in [0,1]$ 是 discount factor，意思是"未来的 reward 不如现在的 reward 值钱"。$\max_{a'} Q(s', a')$ 是"到了 $s'$ 之后最优地选择，能拿到多少"。

**人话**：现在的价值 = 立刻的回报 + 未来的最优价值（打个折）。

这个 equation 之所以牛，是因为它把"无限 horizon 的总 reward"这个无法直接算的东西，压缩成"一步 lookahead + 递归"。只要你能解这个 self-consistency equation，你就解了 RL。

**$\gamma$ 的物理直觉**：你可以把 $\gamma$ 看作"每一步有 $1-\gamma$ 概率 game 结束"。如果 $\gamma = 0.99$，agent 预期寿命大约 100 步。如果 $\gamma = 1$，agent 永生，return 可能无穷大，数学上不好处理。所以 $\gamma$ 不只是 "discount rate"，它是 "时间尺度" 的 knob。

## 3. TD vs MC：用猜测更新猜测

现在问题来了：怎么学 $Q(s,a)$？

**Monte Carlo (MC)**：跑完一整局，看总 reward 是多少，把那个值赋给路径上所有 $(s,a)$。
$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \eta [G_t - Q(s_t, a_t)]$$
变量：$G_t = r_t + \gamma r_{t+1} + \dots$ 是从 $t$ 开始的实际 return。$\eta$ 是 learning rate。

问题：必须等 episode 结束才知道 $G_t$，而且 $G_t$ 方差巨大（它是很多 stochastic reward 的和）。

**Temporal Difference (TD)**：不用等结束，只看一步：
$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \eta [r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)]$$

变量：$\delta_t = r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)$ 是 TD error，意思是"实际 reward + 下一 state 的估计 value"和"当前估计"的差。

**人话**：TD 是"用猜测更新猜测"。我猜 $Q(s_t, a_t) = 5$，然后我看到 $r_t = 1$，$Q(s_{t+1}, \cdot) = 6$，那我估计应该是 $1 + 0.9 \times 6 = 6.4$，比我之前的 5 高，所以往上调整一点。

这个"用自己的估计当 target"叫做 **bootstrapping**。它的好处是低 variance（只用一步的 stochasticity），坏处是有 bias（如果 $Q(s_{t+1})$ 估计错了，target 就错了）。TD 和 MC 的 tradeoff 就是 bias-variance tradeoff。

**TD(λ)** 是两者的插值：用 eligibility trace 记录"哪些 state 最近被访问过"，把它们按时间衰减的权重一起更新。$\lambda = 0$ 是纯 TD，$\lambda = 1$ 是纯 MC。

## 4. Q-learning：off-policy 的聪明和危险

Q-learning 的 update：
$$Q(s, a) \leftarrow Q(s, a) + \eta [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$

注意这里用的是 $\max_{a'} Q(s', a')$，不是 $Q(s', a')$（SARSA 是后者）。这意味着 target 是"假设下一步采取最优 action"，而 behavior 可以是任何 policy（通常 ε-greedy）。

**Off-policy 的 power**：你可以用任何数据学最优 Q。用 replay buffer 存旧数据反复用，用人类 demo 数据学，用旧 policy 的 rollout 学——只要 off-policy，数据效率高得多。

**Deadly triad 的危险**：但是！off-policy + bootstrapping + function approximation（neural net）三者同时出现，训练会 diverge。Baird 的 counterexample 是经典——一个 7 state 2 action 的简单 MDP，用 linear Q function + TD(0)，参数直接发散到无穷。

为什么？因为 NN 把不同 state 的 Q 耦合在一起，bootstrap 用自己的估计当 target，off-policy 让 behavior 和 target 分布不一致——三者叠加形成正反馈环。某 state 被高估，相邻 state 跟着被高估，循环放大。

**DQN 的两个 trick** 解决这个问题：
1. **Target network**：用一份 frozen 的 $Q_{\bar w}$ 算 target，过几步再 sync。让 target 慢慢动，避免"自己追自己"。$\bar w$ 通常是 EMA：$\bar w_t = \rho \bar w_{t-1} + (1-\rho) w_t$。
2. **Experience replay**：把 $(s,a,r,s')$ 存进 buffer，minibatch 随机采样。打破时间相关性，提高 data efficiency。

这两个 trick 让 DQN 在 Atari 上第一次达到 human level。但 DQN 还有 **maximization bias**：$\mathbb{E}[\max_a X_a] \geq \max_a \mathbb{E}[X_a]$。如果 Q 有噪声，$\max$ 会 systematic overestimate。**Double DQN** 用两个 Q network，一个选 action 一个 evaluate，解这个问题。

## 5. Policy gradient：直接优化 policy 的"分数函数" trick

如果 action 是连续的（机械臂关节角度），Q-learning 不好用（$\max_a Q(s,a)$ 难解）。Policy gradient 直接参数化 policy $\pi_\theta(a|s)$，优化 expected return。

核心公式：
$$\nabla_\theta J(\theta) = \mathbb{E}_\tau\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]$$

变量：$\nabla_\theta \log \pi_\theta(a_t|s_t)$ 是 "score function"，$G_t$ 是 reward-to-go。

**人话**：这个公式说的是——如果某个 trajectory 的 $G_t$ 大（好结果），就增大产生它的 action 的 log-prob；如果 $G_t$ 小（坏结果），就减小。梯度方向由 score function 决定，大小由 return 决定。

**Log-derivative trick 为什么 work**：$\nabla_\theta \mathbb{E}_{p_\theta}[f] = \mathbb{E}_{p_\theta}[f \nabla_\theta \log p_\theta]$。这个 trick 让你不用对 environment 求导（environment 可能是黑盒的、stochastic 的、不可微的），只对 policy 求导。这是为什么 RL 可以用在 game、robotics、化学合成这种 environment 不可微的场景。

**两个 variance reduction trick**：

1. **Reward-to-go**：用 $G_t = \sum_{l=t}^T \gamma^{l-t} r_l$ 而非 $R(\tau) = \sum_{l=0}^T \gamma^l r_l$。因为 $a_t$ 不能影响 $r_0, \dots, r_{t-1}$（已经发生），那些 reward 是常数，加进来只增加方差不改变 mean。

2. **Baseline**：减去 $b(s_t)$ 不改变梯度期望（因为 $\sum_a \nabla \pi(a|s) = 0$），但减方差。最优 $b(s) = V_\pi(s)$，对应 advantage $A(s,a) = Q(s,a) - V(s)$。Advantage 的直觉："这个 action 比平均好多少"。

## 6. Actor-Critic：方差更小的 policy gradient

REINFORCE 用 MC 估计 $G_t$，方差大。Actor-Critic 用 TD 估计代替：
$$\delta_t = r_t + \gamma V_w(s_{t+1}) - V_w(s_t)$$
$$\theta \leftarrow \theta + \eta \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \delta_t$$

变量：$V_w$ 是 critic 网络，$\delta_t$ 是 TD error，作为 advantage 的估计。Actor 是 policy $\pi_\theta$，Critic 是 value $V_w$。

**人话**：actor 做决定，critic 打分。critic 用 TD 学（它知道当前 state 值多少），actor 用 critic 的分数更新——"critic 觉得这个 action 比预期好，就多做"。

**GAE (Generalized Advantage Estimation)** 是 TD 和 MC 的精细插值：
$$A_t^{\text{GAE}(\lambda)} = \sum_{l=0}^\infty (\gamma\lambda)^l \delta_{t+l}$$

变量：$\lambda = 0$ 是 1-step TD（高 bias 低 variance），$\lambda = 1$ 是 MC（无 bias 高 variance）。中间值 tradeoff。$(\gamma\lambda)^l$ 是衰减——远处的 TD error 影响小。

**直觉**：$\lambda$ 控制"信任 value function 多远"。小 $\lambda$ 早 stop 用 $V$，大 $\lambda$ 一直 rollout 到 episode 结束。

## 7. TRPO 和 PPO：别一步走太远

Policy gradient 是 SGD 风格，一步可能走太远，policy 突然变差。**TRPO** 用 trust region：保证 KL divergence between old and new policy 小于 $\delta$。数学上漂亮但实现复杂（要 conjugate gradient + line search）。

**PPO** 是简化版，用 clip：
$$L^{PPO} = \mathbb{E}[\min(\rho A, \text{clip}(\rho, 1-\epsilon, 1+\epsilon) A)]$$

变量：$\rho = \pi_\theta(a|s) / \pi_{\text{old}}(a|s)$ 是 likelihood ratio，$\epsilon \approx 0.2$。

**人话**：如果 advantage $A > 0$（好 action），让 $\rho$ 增大但不超过 $1+\epsilon$；如果 $A < 0$（坏 action），让 $\rho$ 减小但不低于 $1-\epsilon$。clip 防止单步 update 破坏 policy。

PPO 是个 heuristic 但 work 得特别好，[implementation details blog](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) 列了 30 多个 trick 才能复现得好。最近 [Rudolph et al. 2025](https://arxiv.org/abs/2502.08938) 发现：调好 hyperparameter（尤其 entropy bonus）的 PPO 在 2-player zero-sum imperfect info game 上能 match CFR 和 PSRO——意味着不需要复杂的 CFR，单一 PPO 就能打。

## 8. SAC：带 entropy 的 RL

Soft Actor-Critic 的 objective：
$$J^{SAC}(\theta) = \mathbb{E}[R(s,a) + \alpha \mathbb{H}(\pi(\cdot|s))]$$

变量：$\alpha$ 是 temperature，$\mathbb{H}$ 是 entropy。

**人话**：在最大化 reward 的同时保持 policy 高 entropy。这有两个好处：1) 鼓励 exploration（不会过早 commit 到一个 action）；2) robust（多个好 action 都保留，环境变了还能 adapt）。

SAC 来自 "RL as inference" 视角：引入 optimality variable $\mathcal{O}_t = 1$ 表示"这一步是好的"，$p(\mathcal{O}_t=1|s_t,a_t) \propto \exp(R(s_t,a_t)/\alpha)$。然后做 variational inference，maximize ELBO。这个视角让 RL 和 probabilistic inference 统一——SAC、MPO、AWR 都是 ELBO 的不同解法。

## 9. Model-based RL：先学世界再 act

Model-free（DQN、PPO）sample inefficient——需要几百万帧 Atari 才学好。Model-based 先学 world model $\hat p(s'|s,a)$，然后用 model 在脑中 rollout 生成 imagined data，再 train policy。

**两种用法**：
- **Decision-time planning**：每一步用 model search 最优 action（MCTS、MPC）。慢但准。
- **Background planning**：用 model 生成 imagined rollout，train policy（Dyna、Dreamer）。快但 model 不准时出错。

**MCTS 的 4 步**：Selection（用 UCT 选 promising node）→ Expansion（sample next state）→ Rollout（递归估 value）→ Backup（更新 Q）。

**AlphaZero** 用 NN 替代 rollout：$f_\theta(s) = (v, \pi)$，$\pi$ 作为 prior 引导 search，$v$ 作为 leaf value。Self-play 生成数据，NN 学 MCTS 的 output。

**MuZero** 更进一步：连规则都不知道，自己学一个 latent dynamics model $M_w(z, a) = (z', r)$。在 latent space 做 MCTS。

**Dreamer** 用 RSSM（recurrent state space model）学 stochastic latent state，在 imagination 里用 actor-critic 训 policy。DreamerV3 是第一个无 human demo 在 Minecraft 钻石的方法。

**Objective mismatch** 是 model-based 的核心问题：predict pixel 不等于 predict task-relevant feature。如果图像里背景占 90% 的 MSE，model 可能只学好背景，对小物体位置一塌糊涂。**JEPA** 的解决：不重建 pixel，只在 latent space 预测。$z_t = E(o_t)$，$\hat z_{t+1} = M(z_t, a_t)$，loss 是 $\|z_{t+1} - \hat z_{t+1}\|^2$。

但 JEPA 有 **collapse 问题**：如果 encoder 学 $E(o) = 0$，loss 也是 0 但 representation 无用。**BYOL-style EMA target** 解决：用 EMA encoder $\bar\phi$ 算 target，target 慢慢移动，predictor 追不上 trivial solution。

## 10. Multi-agent RL：博弈论 + 学习

当环境里有其他 agent 在学习，问题变复杂——你的最优 policy 取决于他们，他们的最优 policy 取决于你。

**Nash equilibrium**：没人能单方面改变 policy 改善自己。问题是：1) Prisoner's dilemma 里 NE 不是 Pareto optimal；2) 多重 NE 选哪个？3) 计算 NE 是 PPAD-complete。

**Self-play** 在 symmetric 2p0s perfect info game（chess、Go）收敛到 NE。但 imperfect info 或 general-sum 可能 oscillate。Fix 用 **fictitious play**（对历史平均 best response）或 **population-based training**（多个 policy 互相对抗）。

**CFR (Counterfactual Regret Minimization)** 用于 poker 这种 imperfect info 2p0s game：每个 information set 独立做 no-regret online learning，regret matching 把累积正 regret 按 proportion 转 policy。证明：平均 policy 收敛到 $\epsilon$-NE。DeepStack、Libratus、Pluribus 都基于这个。

## 11. LLM RL：thinking、alignment、agents

**RLHF**：用人类偏好训 reward model $r_\theta(x,y)$（Bradley-Terry: $p(y_w \succ y_l) = \sigma(r(y_w) - r(y_l))$），再用 PPO 优化 policy 最大化 $r$ - KL penalty。

**GRPO**：去掉 critic，用 group statistics 当 baseline。对每个 prompt sample $J$ 个回答，advantage $= (R_j - \mu) / \sigma$。DeepSeek-R1 用这个训出 thinking model。**Dr GRPO** 发现除以 $\sigma$ 导致 difficulty bias，去掉就好。

**DPO**：直接从 preference data 学 policy，不显式训 reward。推导：从 KL-regularized RL 的最优解 $\pi^* \propto \pi_{\text{ref}} \exp(R/\beta)$ 反解 $R$，代入 Bradley-Terry，得到 loss：
$$\mathcal{L} = -\mathbb{E}[\log \sigma(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)})]$$

把 RL 问题变成分类问题，简单稳定。Limitation：只能用 preference data，不能用 verifiable reward。

**Thinking as marginal likelihood**：模型 generate thinking trace $z$ 再给答案 $y$。RL 只监督 $y$ 的 correctness，$z$ 是 latent variable。$\log p(y|x) = \log \sum_z p(y,z|x)$，RL 是 EM 风格优化。

**R1-Zero 的"涌现"**：DeepSeek-R1-Zero 用纯 RL（无 SFT on CoT）训出会 self-reflect 的模型。但 [Liu et al. 2025](https://arxiv.org/abs/2503.20783) 等论证：base model 在预训练时已经见过 CoT pattern，RL 只是放大。如果 base model 没见过 self-reflection，RL 不会凭空造出来。

**Multi-turn RL**：让 LLM 和 environment 多轮交互（tool use、dialog）。比 single-turn 难：context 增长，return 方差大。RAGEN 用 MC return + group normalization，GEM 用 ReBN（return-batch normalization）。

## 12. Offline RL：不能 query 的麻烦

从 fixed dataset 学 policy，不能 query 未知 $(s,a)$ 的 outcome。Q-learning 在 OOD action 上 overestimate。

**CQL (Conservative Q-Learning)**：加 conservative penalty，压低 OOD action 的 Q，抬高 in-distribution action 的 Q：
$$\mathcal{C} = \mathbb{E}_{a \sim \mu}[Q(s,a)] - \mathbb{E}_{a \sim \pi_b}[Q(s,a)]$$

变量：$\mu$ 是学到的 policy，$\pi_b$ 是 behavior policy。Pessimism 让 policy 不敢选没见过的 action。

**Decision Transformer**：sequence modeling 视角，condition on return-to-go (RTG)。Test time 设 RTG 为 high value，autoregressive 生成 action。简单但 stochastic 环境 fail——trajectory 达到 RTG 可能是运气。

## 13. Exploration 的几个方法

**ε-greedy**：以 $1-\epsilon$ 概率 greedy，$\epsilon$ 概率 random。简单但每个 action 至少 $\epsilon/|A|$ 概率，无法 anneal。

**UCB**：$\tilde R(a) = \hat\mu(a) + c/\sqrt{N(a)}$。Optimism in face of uncertainty——选最乐观的 action，自然探索 uncertain 区域。

**Thompson sampling**：从 posterior 采样 model parameter $\tilde\theta \sim p(\theta|h)$，按 $\tilde\theta$ 最优 act。自动 balance explore/exploit——uncertainty 大时 sample 出不同 model 导致不同 action，uncertainty 小时 sample 收敛到 true model 然后 exploit。

**Intrinsic motivation**：reward 稀疏时给自己造 reward。
- **RND**：fixed random target net + learned predictor，prediction error 是 intrinsic reward。
- **ICM**：inverse dynamics 学 controllable feature，forward model 在 feature space 预测。避免 "noisy TV problem"（agent 卡在 stochastic source）。
- **Empowerment**：maximize $I(\text{action}; \text{next state})$，学可控的 skill。

## 14. Successor Representation：model-free 和 model-based 的中间地带

$$M^\pi(s, \tilde s) = \mathbb{E}\left[\sum_t \gamma^t \mathbb{I}(s_t = \tilde s) | s_0 = s\right]$$

变量：$M^\pi(s, \tilde s)$ 是从 $s$ 出发，policy $\pi$ 下访问 $\tilde s$ 的 discounted 次数。

**人话**：SR 记录"从 $s$ 出发，未来会去哪些 state、去多少次"。它有 model-free 的效率（用 TD 学），又有 model-based 的灵活性（换 reward function 时不用重学 SR，只需 $V = M \cdot R$）。

**Successor Features** 把 SR 推广到连续 state：$\psi^\pi(s) = \mathbb{E}[\sum_t \gamma^t \phi(s_t) | s_0 = s]$，reward 假设是 $R(s) = \phi(s)^T w$，那 $V^\pi(s) = \psi^\pi(s)^T w$。换 reward 只需换 $w$。这让 SF 适合 multi-task RL 和 intrinsic reward（reward 老变）。

## 15. 整体直觉：RL 的所有 trick 都在处理四件事

回头看，所有 RL 算法都在处理这四个核心 difficulty：

| Difficulty | 算法 |
|---|---|
| Credit assignment | Bellman、TD、GAE、 eligibility trace |
| Function approximation instability | Target network、EMA、Double Q、LayerNorm |
| Exploration | ε-greedy、UCB、Thompson、intrinsic reward、entropy bonus |
| Sample efficiency | Replay buffer、model-based、world model、offline RL |
| Partial observability | Belief state、RNN policy、recurrent head |
| Multi-agent non-stationarity | Self-play、fictitious play、CFR、PSRO、population |
| Sparse reward | HER、reward shaping、intrinsic motivation、curriculum |
| OOD in offline | CQL、uncertainty penalty、behavior regularization |

**最深的 insight**：RL = sequential Bayesian decision making under uncertainty。从这个视角，SAC、DPO、Dreamer、AlphaZero、Thompson sampling 都是同一框架的不同 instantiation——差别在 latent variable 是什么、用什么 variational family、怎么 amortize inference。RL as inference 是统一的语言。

**为什么 RL 比 SL 难**：SL 的目标固定（predict label），数据分布固定（IID）。RL 的目标是 discovered（reward signal 稀疏延迟），数据分布是 agent 自己造的（non-IID、non-stationary）。每一个 RL trick 都是在处理这两个 fundamental gap 的某个侧面。

**AIXI 是 north star**：理论上，用 Solomonoff induction（最短程序假设）+ expectimax 可以定义"最 intelligent agent"。Intractable 但告诉我们：RL 的本质是"在未知 computable environment 中做 optimal sequential decision"。所有实际算法都是 AIXI 的 tractable approximation——用 NN 替代 Solomonoff，用 MCTS 替代 expectimax，用 Thompson sampling 替代 Bayes average。

更多深入阅读：
- [Sutton & Barto RL book](http://incompleteideas.net/book/RLbook2020.pdf)
- [Levine offline RL tutorial](https://arxiv.org/abs/2005.01643)
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [CleanRL](https://github.com/vwxyzjn/cleanrl)
- [Tinker (LLM RL lib)](https://tinker-docs.thinkingmachines.ai/)
- [GEM multi-turn LLM RL](https://arxiv.org/abs/2510.01051)
- [TÜLU 3 post-training](https://arxiv.org/abs/2411.15124)
- [DreamerV3 Nature paper](https://www.nature.com/articles/s41586-025-08744-2)
- [Rudolph et al. PPO in 2p0s games](https://arxiv.org/abs/2502.08938)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [DPO](https://arxiv.org/abs/2305.18290)
- [GRPO / DeepSeek-Math](https://arxiv.org/abs/2402.03300)

---

# Kevin Murphy《Reinforcement Learning: An Overview》深度讲解

这份 document 由 Kevin Murphy（probabilistic ML 教科书作者，[probml.github.io](https://probml.github.io/)）在 2025 年 12 月发布，是一份覆盖 RL 几乎所有重要分支的综述/教程。它的定位介于教科书与综述之间，假设读者熟悉 ML 与概率论，但想系统地掌握 RL 的现代 landscape。作者本人长期在 Google Research 工作，所以文中引用了大量 DeepMind 的工作，但也涵盖了 Berkeley、Stanford、清华等学术圈的近期结果。

## 1. 全书结构概览

整本书 8 个 chapter，199 页（含 references）：

- **Chapter 1 Introduction**：定义 sequential decision making、canonical models（MDP/POMDP/bandit/belief state MDP）、reward function 问题
- **Chapter 2 Value-based RL**：Q-learning、DQN、deadly triad、Rainbow、BBF
- **Chapter 3 Policy-based RL**：REINFORCE、Actor-Critic、TRPO、PPO、SAC、RL as inference
- **Chapter 4 Model-based RL**：MCTS、MPC、Dyna、Dreamer、JEPA、TD-MPC、Successor representations
- **Chapter 5 Multi-agent RL**：games、Nash equilibrium、CFR、PSRO、self-play
- **Chapter 6 LLMs and RL**：RLHF、GRPO、DPO、thinking models、agents
- **Chapter 7 Other topics**：regret、exploration、distributional RL、intrinsic motivation、HRL、imitation learning、offline RL、AIXI
- **Chapter 8 Acknowledgements**

这份综述的一个突出特点：把 LLM 放在 RL 的语境下统一理解，把 RL as inference、offline RL、world models 等原本独立的话题用概率推断的视角串起来。

---

## 2. Sequential Decision Making 的通用框架

### 2.1 The universal model

Murphy 在 §1.1.3 提出"universal model"，统一了 RL 的所有 variants。关键 insight 是：agent 和 environment 都是 stochastic process，需要区分三件事：

1. **Environment state $w_t$**：世界的真实隐状态，受 agent action $a_t$ 影响，更新规则为 $w_{t+1} = M(w_t, a_t, \epsilon_t^w)$
2. **Observation $o_{t+1} = O(w_{t+1}, \epsilon_{t+1}^o)$**：agent 收到的传感输入，可能 noisy、partial
3. **Agent internal state $z_t$**：agent 维护的 belief，通过 state update 函数 $z_{t+1} = SU(z_t, a_t, o_{t+1})$ 更新

进一步把 $SU$ 拆成 **predict** + **update**：
$$z_{t+1} = U(P(z_t, a_t), E(o_{t+1}))$$
其中 $P$ 是 forward predictor，$E$ 是 observation encoder。这套框架很像 active inference [Friston 2009, https://www.sciencedirect.com/science/article/pii/S1364661309000491] 与 Sutton 的 common model [Sutton 2022, https://arxiv.org/abs/2202.13222]。

这个图（Figure 1.2）的 intuition：**agent 永远在两件事之间纠结**——预测自己的下一个状态，然后基于实际 observation 修正预测。这其实就是 Bayes filter 的 RL 版本。

### 2.2 Maximum expected utility 与 discount factor

目标函数：
$$V_\pi(s_0) = \mathbb{E}_{p(a_0, s_1, \dots, s_T | s_0, \pi)}\left[\sum_{t=0}^T R(s_t, a_t)\right]$$

变量说明：
- $V_\pi(s_0)$：policy $\pi$ 在初始状态 $s_0$ 下的 expected return
- $R(s_t, a_t)$：state-action reward
- expectation 对 trajectory 分布 $p(\tau | s_0, \pi)$ 取

Return 的 discounted 形式：
$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots = r_t + \gamma G_{t+1}$$

**$\gamma$ 的两种解读**（intuition）：第一种是数学上的，保证 infinite horizon 下 return 有界；第二种是概率上的——$\gamma$ 等价于"每一步有 $1-\gamma$ 概率 game 结束"。Murphy 给了一个有用的数：如果每步 0.1 秒，$\gamma = 0.95$ 对应约 2 秒寿命。这个解读把 discount 和"agent 的预期寿命"挂钩，对理解 $\gamma$ 的语义非常重要。

### 2.3 Canonical models 的 hierarchy

| Model | 状态可见性 | Transition 依赖 | 用途 |
|---|---|---|---|
| MDP | 完全可见 $o_t = s_t$ | Markov | 经典 RL |
| POMDP | 部分可见 | Markov 在 latent | 真实世界 |
| Contextual bandit | 状态独立于 action | $p(w_t|w_{t-1}, a_t) = p(w_t)$ | 推荐系统、CTR |
| Belief state MDP | state 是 posterior $b_t = p(w_t|h_t)$ | deterministic Bayes update | Bayesian RL、exploration |
| Goal-conditioned MDP | reward $R(s,a|g) = \mathbb{I}(s=g)$ | Markov + goal conditioning | universal policy |
| Contextual MDP | hidden context $\theta$ 控制 dynamics | $M(\theta)$ | 泛化、procedural generation |

一个值得 highlight 的 insight（§1.2.7）：**bandit、Bayesian optimization、active learning、SGD 都是这个 universal framework 的特例**。例如 SGD 可以看成"在参数空间 $\theta$ 上做 sequential decision，action = 选择 query 点，reward = function value (或 gradient)"。这个视角把优化和决策统一起来，参考 Powell 的书 [Powell 2022, https://www.amazon.com/Reinforcement-Learning-Stochastic-Optimization-Sequential/dp/1119815037]。

---

## 3. Value-based RL：从 tabular 到 DQN 的演化

### 3.1 Bellman equation 的本质

最优 value function 满足：
$$V^*(s) = \max_a R(s, a) + \gamma \mathbb{E}_{p_S(s'|s,a)}[V^*(s')]$$
$$Q^*(s, a) = R(s, a) + \gamma \mathbb{E}_{p_S(s'|s,a)}[\max_{a'} Q^*(s', a')]$$

变量说明：
- $V^*(s)$：最优 state value，从 $s$ 出发的最大 expected return
- $Q^*(s,a)$：最优 state-action value
- $p_S(s'|s,a)$：environment transition kernel

Bellman equation 的 intuition：**最优值 = 立即 reward + 下一步最优值的期望**。这是一个 self-consistency equation，把"无限 horizon"的问题压缩成"一步 lookahead + 递归"。Dynamic programming 的精髓就在这里——我们不需要 enumerate 所有 trajectory，只要在状态空间上做 contraction mapping。

### 3.2 TD learning：从 MC 到 TD(λ)

Monte Carlo update：
$$V(s_t) \leftarrow V(s_t) + \eta[G_t - V(s_t)]$$
需要等到 episode 结束才知道 $G_t$，variance 大。

TD(0) update：
$$V(s_t) \leftarrow V(s_t) + \eta[r_t + \gamma V(s_{t+1}) - V(s_t)]$$
变量：$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是 TD error。

TD 的 intuition（关键）：**TD 用当前估计 $V(s_{t+1})$ 作为 "future reward" 的 proxy，等价于在每一步做一步 rollout + bootstrap**。这是一种 bias-variance tradeoff——TD 有 bias（因为 $V$ 不准确），但 variance 低，且可以在线更新。

TD(λ) 把两者插值，用 eligibility trace：
$$z_t = \gamma\lambda z_{t-1} + \nabla_w V_w(s_t)$$
$$w_{t+1} = w_t + \eta \delta_t z_t$$

变量说明：
- $z_t$：eligibility trace，记录"哪些 state 最近被访问过，权重多大"
- $\lambda$：1 = MC，0 = TD(0)，中间值是插值
- $\gamma\lambda$ 衰减系数

这个 trace 的 intuition：**就像短期记忆的衰减**——最近访问的 state 对当前的 TD error 最"负责"，所以应该多更新它们；远处的 state 应该少更新。

### 3.3 Q-learning 与 deadly triad

Tabular Q-learning：
$$Q(s,a) \leftarrow Q(s,a) + \eta[r + \gamma \max_{a'} Q(s', a') - Q(s,a)]$$

变量：$r$ 是 observed reward，$s'$ 是 next state。注意这里直接用 $\max_{a'} Q(s', a')$，所以是 **off-policy**——目标 policy 是 greedy，behavior policy 可以是 ε-greedy。

**Deadly triad**（§2.5.2.5）是 Sutton 提出的经典论断：以下三者同时出现会导致 divergence：
1. **Function approximation**（neural network）
2. **Bootstrapping**（TD-style update，target 依赖自己的估计）
3. **Off-policy learning**（behavior ≠ target）

Baird's counterexample（Figure 2.6）：一个 7 state 2 action 的简单 MDP，linear value function $V_w(s) = w^T \phi(s)$，TD(0) 会让 $w$ 发散到无穷。

**为什么 deadly triad 危险**？直觉：在 off-policy 下，behavior 产生的 state distribution 和 target policy 想要评估的 state distribution 不同。Bootstrap 用自己的估计当 target，再加上 function approximation 把不同 state 的值耦合在一起，可能形成正反馈环——某 state 的值被 overestimate，相邻 state 也被 overestimate，反复放大。

### 3.4 DQN 与解决方案

DQN [Mnih et al. 2015, https://www.nature.com/articles/nature14236] 用两个 trick 缓解 deadly triad：

**Target network**：
$$y(r, s'; \bar w) = r + \gamma \max_{a'} Q_{\bar w}(s', a')$$
其中 $\bar w$ 是 frozen copy of $w$，每隔几步同步一次（或用 EMA $\bar w_t = \rho \bar w_{t-1} + (1-\rho) w_t$）。Intuition：**让 target 慢慢动，避免"自己追自己"的正反馈**。

**Experience replay**：把 $(s, a, r, s')$ 存进 buffer，随机 minibatch 采样。两个好处：data efficiency + 破坏时间相关性。

**Double DQN** [Hasselt et al. 2016, https://arxiv.org/abs/1509.06461] 解决 **maximization bias**：
$$\mathbb{E}[\max_a X_a] \geq \max_a \mathbb{E}[X_a]$$
所以 $\max$ over noisy Q 会 systematic overestimate。Fix：用两个 Q network $Q_1, Q_2$，一个选 action，一个 evaluate：
$$y = r + \gamma Q_2(s', \arg\max_{a'} Q_1(s', a'))$$

**Rainbow** [Hessel et al. 2018, https://arxiv.org/abs/1710.02298] 整合 6 个 trick：double DQN + prioritized replay + C51 distributional + n-step + dueling + noisy nets。每个 trick 的 marginal gain 在 Figure 2.9 可见。

**BBF** [Schwarzer et al. 2023, https://arxiv.org/abs/2305.19452] 是 2024 年 Atari-100k 的 SOTA，关键 trick：
- 增大网络（modified Impala）
- 提高 update-to-data ratio (UTD)
- 周期性 soft reset（防 elasticity loss）
- n-step returns anneal from 10 to 3
- 自预测 representation loss（self-prediction auxiliary）
- $\gamma$ 从 0.97 退火到 0.997
- weight decay + dueling + distributional

### 3.5 Dueling DQN 的架构

把 $Q$ 分解为 $V + A$：
$$Q(s, a) = V(s) + A(s, a) - \max_{a'} A(s, a')$$
或实际用 mean 而非 max：
$$Q(s, a) = V(s) + A(s, a) - \frac{1}{|A|}\sum_{a'} A(s, a')$$

变量：$V(s)$ state value，$A(s,a)$ advantage。Intuition：**当很多 action 的 Q-value 接近时，学 $V$ 和 $A$ 比直接学 $Q$ 更高效**——$V$ 是共享 baseline，$A$ 只学相对差异。

---

## 4. Policy-based RL：直接优化 policy

### 4.1 Policy gradient theorem 推导

REINFORCE 的核心：用 log-derivative trick
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta}[R(\tau) \nabla_\theta \log p_\theta(\tau)]$$
$$= \mathbb{E}_\tau\left[\sum_{k=1}^T \nabla_\theta \log \pi_\theta(a_k|s_k) \cdot G_k\right]$$

变量：
- $J(\theta) = \mathbb{E}_{p_\theta(\tau)}[R(\tau)]$：expected return
- $G_k = \sum_{l=k}^T \gamma^{l-k} r_l$：reward-to-go from step $k$
- $\nabla_\theta \log \pi_\theta(a_k|s_k)$：score function

**为什么用 reward-to-go $G_k$ 而非 full return $R(\tau)$**？因为 $a_k$ 不能影响 $r_1, \dots, r_{k-1}$（已经发生），所以那些 reward 是 constant，加上它们只会增加 variance 不改变 mean。这是因果性 reduction 的直觉。

**Baseline 减方差**：减去 $b(s)$ 不改变期望（因为 $\sum_a \nabla_\theta \pi_\theta(a|s) = 0$），但能减方差：
$$\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_k \nabla_\theta \log \pi_\theta(a_k|s_k)(G_k - b(s_k))\right]$$
最优 $b(s) = V_\pi(s)$，对应 advantage $A = Q - V$。

### 4.2 Actor-Critic 与 GAE

A2C 用 TD 替代 MC 估计 $G_k$：
$$\delta_t = r_t + \gamma V_w(s_{t+1}) - V_w(s_t)$$
$$\theta \leftarrow \theta + \eta \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \delta_t$$

**GAE** [Schulman et al. 2016, https://arxiv.org/abs/1506.02438] 是 bias-variance 的精细插值：
$$A_t^{(n)} = \sum_{i=0}^{n-1} \gamma^i r_{t+i} + \gamma^n V_w(s_{t+n}) - V_w(s_t)$$
$$A_t^{\text{GAE}(\lambda)} = (1-\lambda)\sum_{n=1}^\infty \lambda^{n-1} A_t^{(n)} = \sum_{l=0}^\infty (\gamma\lambda)^l \delta_{t+l}$$

变量：
- $A_t^{(n)}$：n-step advantage
- $\lambda$：bias-variance tradeoff，1 = MC（高 variance 无 bias），0 = TD（低 variance 有 bias）
- $\delta_{t+l}$：第 $t+l$ 步的 TD error

**GAE 的直觉**：$\lambda$ 决定"信任 value function 多远"——小 $\lambda$ 早 stop 用 $V$，大 $\lambda$ 一直 rollout 到 episode end。$(\gamma\lambda)^l$ 是 decay factor，远处的 TD error 影响小。

### 4.3 TRPO 与 PPO

TRPO [Schulman et al. 2015, https://arxiv.org/abs/1502.05477] 用 trust region：
$$\pi_{k+1} = \arg\max_\pi L(\pi, \pi_k) \text{ s.t. } \mathbb{E}[D_{KL}(\pi_k \| \pi)] \leq \delta$$

其中 surrogate loss：
$$L(\pi, \pi_k) = \mathbb{E}_{p_{\pi_k}^\gamma(s) \pi_k(a|s)}\left[\frac{\pi(a|s)}{\pi_k(a|s)} A^{\pi_k}(s,a)\right]$$

变量：$\frac{\pi(a|s)}{\pi_k(a|s)}$ 是 importance sampling ratio，$\delta$ 是 trust region 半径。

PPO [Schulman et al. 2017, https://arxiv.org/abs/1707.06347] 简化 TRPO，用 clip 替代 KL constraint：
$$L^{PPO}(\theta) = \mathbb{E}\left[\min\left(\rho(\theta) A, \text{clip}(\rho(\theta), 1-\epsilon, 1+\epsilon) A\right)\right]$$
其中 $\rho(\theta) = \pi_\theta(a|s)/\pi_{\text{old}}(a|s)$。

**Clip 的直觉**：如果 advantage $A > 0$（action 好），希望 $\rho$ 增大但不大于 $1+\epsilon$；如果 $A < 0$（action 差），希望 $\rho$ 减小但不小于 $1-\epsilon$。clip 防止单步 update 移动太远，破坏 policy。

PPO 的 implementation details 在 [ICLR blog track 2022](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) 有详细讨论。一个值得注意的事：PPO 在 multi-agent 设置下 [Yu et al. 2022 MAPPO, https://arxiv.org/abs/2103.01955] 和 imperfect information game [Rudolph et al. 2025, https://arxiv.org/abs/2502.08938] 都出奇地有效，前提是 hyperparameter（尤其 entropy bonus $\alpha$）调好。

### 4.4 Deterministic Policy Gradient 与 DDPG/TD3

对 continuous action，policy gradient 的方差大。DPG [Silver et al. 2014, http://proceedings.mlr.press/v32/silver14.pdf] 用 deterministic policy $a = \mu_\theta(s)$：
$$\nabla_\theta J(\mu_\theta) = \mathbb{E}_{\rho_{\mu_\theta}(s)}[\nabla_\theta \mu_\theta(s) \nabla_a Q_{\mu_\theta}(s, a)|_{a=\mu_\theta(s)}]$$

变量：$\nabla_\theta \mu_\theta(s) \in \mathbb{R}^{N_\theta \times N_A}$ 是 Jacobian，$\nabla_a Q$ 是 action-gradient。

**直觉**：DPG 不需要 sample action，直接用 chain rule 通过 $\nabla_a Q$ 反向传播。但 deterministic policy 没 exploration，必须 off-policy（用 stochastic behavior policy + replay buffer）。

DDPG [Lillicrap et al. 2016, https://arxiv.org/abs/1509.02971] = DPG + DQN-style critic + replay。

TD3 [Fujimoto et al. 2018, https://arxiv.org/abs/1802.09477] 加三个 trick：
1. **Target policy smoothing**：在 target action 加 noise $\tilde a = \mu_{\bar\theta}(s') + \text{clip}(\text{noise}, -c, c)$
2. **Clipped double Q**：$y = r + \gamma \min_{i=1,2} Q_{\bar w_i}(s', \tilde a)$，取 min 抑制 overestimation
3. **Delayed policy update**：critic 更新多次后再更新 actor，让 critic 更准确

### 4.5 Soft Actor-Critic（SAC）与 RL as inference

SAC [Haarnoja et al. 2018, https://arxiv.org/abs/1801.01290] 是 maximum entropy RL：
$$J^{\text{SAC}}(\theta) = \mathbb{E}_{p_\pi^\gamma(s)\pi(a|s)}[R(s,a) + \alpha \mathbb{H}(\pi(\cdot|s))]$$

变量：$\alpha$ 是 temperature，$\mathbb{H}$ 是 entropy。**Intuition**：在最大化 reward 的同时保持 policy 高 entropy，鼓励 exploration + 防止 premature convergence。

SAC 的推导来自 "RL as inference" 框架（§3.6）：引入 optimality variable $\mathcal{O}_t = 1$，定义 $p(\mathcal{O}_t=1|s_t,a_t) \propto \exp(\eta^{-1} G(s_t,a_t))$。然后做 variational inference，最大化 ELBO：
$$J(\pi_p, \pi_q) = \sum_t \mathbb{E}_q[\eta^{-1} G(s_t,a_t)] - D_{KL}(\pi_q(\cdot|s_t) \| \pi_p(\cdot|s_t))$$

**两种 solve 方式**：
- **EM control**：E step 求非参 $\pi_q$，M step 投影到参数化 $\pi_p$。MPO [Abdolmaleki et al. 2018, https://openreview.net/pdf?id=S1ANxQW0b] 用这个。
- **KL control**：固定 $\pi_p$ = uniform，只优化 $\pi_q$。SAC 属于这类。

SAC 的 update：
- Critic：$J_Q(w) = \mathbb{E}[\frac{1}{2}(Q_w(s_t, a_t) - y(r, s'))^2]$
- Actor：$J_\pi(\theta) = \mathbb{E}[\alpha \log \pi_\theta(a_t|s_t) - Q_w(s_t, a_t)]$，用 reparameterization $a_t = f_\theta(s_t, \epsilon_t)$ 让梯度可微。

---

## 5. Model-based RL：学习世界

### 5.1 MBRL 的游戏论视角

§4.3.1 把 MBRL 看作 two-player game [Rajeswaran et al. 2020, https://arxiv.org/abs/2005.05952]：
$$\max_\pi J(\pi, \hat M), \quad \min_{\hat M} \ell(\hat M, \mu_{M_{\text{env}}}^\pi)$$

变量：
- $J(\pi, M')$：policy $\pi$ 在模型 $M'$ 中的 value
- $\ell(\hat M, \mu)$：model $\hat M$ 在 state-action 分布 $\mu$ 下的 KL loss
- $\mu_{M_{\text{env}}}^\pi$：真实 environment 下 $\pi$ 诱导的 state-action distribution

**Stackelberg formulation**：
- **Policy as leader (PAL)**：先 fit model（基于 $\pi_k$），再小步更新 $\pi$。Model 是局部的。
- **Model as leader (MAL)**：先用当前 model 学 $\pi$，再 collect data 改进 model。Model 是 global 的。

PAL 收敛性更好（局部 model 在 $\pi$ 附近准确），MAL 泛化更好（model 见过更多 state）。这个视角让 MBRL 算法可以套用 game theory 的 convergence result。

### 5.2 Decision-time planning：MCTS 与 MuZero

MCTS 的核心 4 步：
1. **Selection**：用 UCT $a = \arg\max_a Q(s,a) + c\sqrt{\log N(s)/N(s,a)}$
2. **Expansion**：sample $s' \sim p(s'|s,a)$
3. **Rollout**：递归 evaluate leaf
4. **Backup**：$Q(s,a) \leftarrow Q(s,a) + \frac{1}{N(s,a)}(u - Q(s,a))$

变量：$N(s,a)$ 是 visit count，$c$ 是 exploration bonus，$u$ 是 rollout return。

**AlphaZero** [Silver et al. 2018, https://science.org/doi/10.1126/science.aar6404] 用 neural net $f_\theta(s) = (v_s, \pi_s)$ 替代 rollout。Loss：
$$\mathcal{L}(\theta) = \mathbb{E}[(V^{\text{MCTS}}(s) - V_\theta(s))^2 - \sum_a \pi_s^{\text{MCTS}}(a) \log \pi_\theta(a|s)]$$

变量：$\pi_s^{\text{MCTS}}(a) = [N(s,a)/\sum_b N(s,b)]^{1/\tau}$ 是 MCTS 产生的 visit distribution，作为 policy 的 supervised target。$V^{\text{MCTS}}$ 是 leaf value。

**MuZero** [Schrittwieser et al. 2020, https://www.nature.com/articles/s41586-020-03051-6] 学习 latent dynamics，不需要知道规则：
$$\mathcal{L}(\theta, w, \phi) = \mathbb{E}[(V^{\text{MCTS}}(z) - V_\theta(e_\phi(o)))^2 + \dots + (r - M_w^r(e_\phi(o), a))^2]$$

变量：$z_t = e_\phi(o_t)$ 是 latent embedding，$M_w(z, a) = (z', r)$ 是 latent dynamics。

**EfficientZero** [Ye et al. 2021, https://arxiv.org/abs/2111.00210] 加 self-prediction loss：
$$\mathcal{L}_{EZP}(\phi, \theta; h, a, h') = \|M_\theta(E_\phi(h, a)) - \text{sg}(E_{\bar\phi}(h'))\|_2^2$$

变量：$\text{sg}$ 是 stop-gradient，$\bar\phi$ 是 EMA encoder。这个 loss 防止 representation collapse + 提供 auxiliary supervision。EfficientZero 在 Atari-100k 上把 sample efficiency 大幅提升。

### 5.3 Dreamer 系列：latent imagination

Dreamer [Hafner et al. 2020, https://arxiv.org/abs/1912.01603] 用 RSSM（recurrent state space model）：
- Deterministic state $h_{t+1} = \mathcal{U}(h_t, a_t, z_t)$
- Stochastic prior $\hat z_t \sim P(\hat z_t | h_t)$
- Stochastic posterior $z_t \sim E(z_t | h_t, o_t)$
- Decoder $\hat o_t \sim D(\hat o_t | h_t, \hat z_t)$
- Reward $\hat r_t \sim R(\hat r_t | h_t, \hat z_t)$

Loss：
$$\mathcal{L}^{WM} = \mathbb{E}_q\left[\sum_t \beta_o \mathcal{L}^o(o_t, \hat o_t) + \beta_z \mathcal{L}^z(z_t, \hat z_t)\right]$$

变量：$\mathcal{L}^o$ 是 observation reconstruction loss，$\mathcal{L}^z$ 是 posterior-prior KL。Actor-critic 在 imagination rollout 上训练，用 GAE $G_t^\lambda$。

DreamerV3 [Hafner et al. 2025, https://www.nature.com/articles/s41586-025-08744-2] 是第一个 "无 human demo" 在 Minecraft 钻石的方法。DreamerV4 [Hafner et al. 2025, https://arxiv.org/abs/2509.24527] 用 latent diffusion world model + VPT 数据 offline 预训练 + actor-critic 微调。

### 5.4 JEPA 与 non-generative world models

JEPA [LeCun 2022, https://openreview.net/pdf?id=BZ5a1r-kVsf] 的核心：**不重建 observation，只预测 latent**。

$$z_t = E(o_t), \quad z_{t+1} = E(o_{t+1}), \quad \hat z_{t+1} = M(z_t, a_t; \epsilon_t)$$
$$\text{Loss} = \|z_{t+1} - \hat z_{t+1}\|_2^2$$

**Collapse 问题**：如果 encoder 学到 $E(o) = 0$，loss 也为 0 但 representation 无用。两个解决方法：

1. **EMA target（BYOL-style）** [Grill et al. 2020, https://arxiv.org/abs/2006.07733]：
$$\mathcal{L}_{EZP} = \|M_\theta(E_\phi(h, a)) - \text{sg}(E_{\bar\phi}(h'))\|_2^2$$
$\bar\phi$ 是 EMA，慢慢移动，让 target 稳定。

2. **Information-theoretic regularizer**（VICReg [Bardes et al. 2022, https://arxiv.org/abs/2105.04906]）：variance + invariance + covariance 三项。

**Objective mismatch**（§4.4.2.1）是 MBRL 的核心问题：**predict observation 不等于 predict task-relevant feature**。比如图像背景像素占 90% 的 MSE，agent 可能只学到背景模型而忽略小车。JEPA 通过 latent prediction 绕开这个问题。

### 5.5 TD-MPC2

TD-MPC2 [Hansen et al. 2024, https://arxiv.org/abs/2310.16828] 学 latent dynamics + reward + value + policy prior：
$$\mathcal{L}(\theta) = \mathbb{E}\left[\sum_t \lambda^t (\|z_t' - \text{sg}(E(o_t'))\|_2^2 + \text{CE}(\hat r_t, r_t) + \text{CE}(\hat q_t, q_t))\right]$$

变量：$z_t'$ 是 latent rollout prediction，$\hat q_t$ 是 Q value prediction（用 cross-entropy 在 discretized log-space 上）。Policy 用 SAC loss 在 latent rollout 上训练。Runtime 用 MPPI（CEM 变种）做 planning，policy prior 初始化 candidate action sequences。

---

## 6. Multi-agent RL：博弈论 + 学习

### 6.1 Game 类型与 solution concept

| 概念 | 定义 | 用途 |
|---|---|---|
| Nash equilibrium (NE) | $\forall i, \pi_i': U_i(\pi_i', \pi_{-i}) \leq U_i(\pi)$ | 标准解 |
| $\epsilon$-NE | $\forall i: U_i(\pi_i', \pi_{-i}) - \epsilon \leq U_i(\pi)$ | 近似解 |
| Minimax | $\max_{\pi_i}\min_{\pi_j} U_i$ | 2p0s 零和 |
| Correlated equilibrium (CE) | 允许联合分布 | 比 NE 更宽 |
| Quantal response (QRE) | entropy-regularized NE | 人类行为建模 |
| Stackelberg | leader-follower 顺序博弈 | curriculum design |

**NE 的两个限制**：
1. Prisoner's dilemma：(D,D) 是 NE，但 (C,C) 收益更高——NE 不一定 Pareto optimal
2. 多重 NE：Battle of sexes 有 3 个 NE，选哪个？
3. 计算 NE 是 PPAD-complete 难题

### 6.2 CFR：counterfactual regret minimization

CFR [Zinkevich et al. 2007, https://papers.nips.cc/paper/2007/hash/71c0e1c5b1d3c1d3c5f8c7e2e1f7c7e1-Abstract.html] 用于 imperfect information 2p0s game：
- **Counterfactual value** $q_{\pi,i}^c(h^i, a^i) = \sum_{(\tau, z) \in Z(h^i)} \eta_{-i}^\pi(\tau) \eta^\pi(\tau a^i, z) u_i(z)$
- **Instantaneous regret** $r_i^k(h^i, a^i) = q_{\pi^k, i}^c(h^i, a^i) - v_{\pi^k, i}^c(h^i)$
- **Cumulative regret** $R_i^k(h^i, a^i) = \sum_{j=0}^k r_i^j$
- **Regret matching update** $\pi_i^{k+1}(h^i, a^i) = \frac{R_i^{k,+}(h^i, a^i)}{\sum_a R_i^{k,+}(h^i, a)}$ 或 uniform

变量：$\eta_{-i}^\pi(\tau)$ 是"其他 player + chance"达到 trajectory 的概率，$\eta^\pi(\tau a^i, z)$ 是从 $\tau a^i$ 到 terminal $z$ 的概率。

**CFR 的直觉**：每个 information set 独立做 no-regret online learning，regret matching 把累积正 regret 按 proportion 转 policy。证明：平均 policy 收敛到 $\epsilon$-NE，$\epsilon = O(\sqrt{|A|/t})$。

应用：DeepStack [Moravcik et al. 2017, https://doi.org/10.1126/science.aam6960] (heads-up poker)，Libratus/Pluribus [Brown & Sandholm 2017/2019, https://www.science.org/doi/10.1126/science.aao1731]。

### 6.3 PSRO 与 population-based training

PSRO [Lanctot et al. 2017, https://arxiv.org/abs/1711.00832] 算法：
1. 每代 $k$，每个 agent 有 policy pool $\Pi_i^k$
2. 估计 meta-game reward matrix
3. Solve meta-game（如 Nash）
4. 得到 meta-strategy $\sigma_i^k$
5. Compute best response $\pi_i'$ to $\sigma_{-i}^k$
6. 加入 pool：$\Pi_i^{k+1} = \Pi_i^k \cup \{\pi_i'\}$

**收敛**：如果 oracle 给 exact best response，meta-solver 给 exact NE，$\sigma^k$ 收敛到 underlying game 的 NE。

**AlphaStar** [Vinyals et al. 2019, https://www.nature.com/articles/s41586-019-1724-z] 在 StarCraft II 用 PSRO 变种 + league + self-play 达到 grandmaster。

### 6.4 Self-play 的 trap

Self-play 在 symmetric 2p0s perfect information game（如 chess、Go）收敛到 NE。但 **imperfect information 或 general-sum** 时可能 oscillate 或被 exploit。Wang et al. 2023 [https://arxiv.org/abs/2211.00241] 甚至显示人类可以打败"超人类"Go AI（通过对抗 exploit）。

解决方法：
- **Fictitious play**：对历史平均策略做 best response
- **NFSP** [Heinrich & Silver 2016, https://arxiv.org/abs/1603.01121]：neural 版本
- **Population diversity**：PSRO/AlphaStar league

### 6.5 Magnetic Mirror Descent

MMD [Sokota et al. 2022, https://openreview.net/forum?id=DpE5UYUQzZH] 对 2p0s game 收敛 QRE：
$$\pi_{k+1} = \arg\max_\pi \langle \pi, q_k \rangle - \alpha D_{KL}(\pi, \rho) - \frac{1}{\eta} D_{KL}(\pi, \pi_k)$$

变量：$\rho$ 是 magnet policy（防 oscillation），$\alpha$ 是 entropy-like penalty，$\eta$ 是 stepsize。

闭合解：
$$\pi_{k+1} \propto [\pi_k \rho^{\alpha\eta} e^{\eta q_k}]^{1/(1+\alpha\eta)}$$

实验结论（[Rudolph et al. 2025, https://arxiv.org/abs/2502.08938]）：**调好 hyperparameter 的 PPO 在 2p0s 不完美信息 game 上能 match CFR 和 PSRO**。这个结论很重要——意味着不需要复杂的 CFR，单一 PPO 可以胜任。

---

## 7. LLMs and RL：现代 alignment

### 7.1 RLFT 的问题设置

Single-turn LLM RL 是 contextual bandit：
$$J(\theta) = \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi_\theta(a|s)}[R(s, a)]$$
其中 $s$ 是 prompt，$a$ 是 token sequence。Markovian 化：
$$p(s_t | s_{t-1}, a_t) = \delta(s_t = \text{concat}(s_{t-1}, a_t))$$
reward $R(s_t, a_t) = R(s, a_{1:T})$ if $t = T$ else 0（sparse）。

加 KL regularizer：
$$\hat R_{n,t}^j = R_n^j - \beta D_{KL}(\pi_{\text{old}}(a_{nt}^j | \cdot) \| \pi_{\text{ref}}(a_{nt}^j | \cdot))$$

### 7.2 GRPO：消除 critic

GRPO [Shao et al. 2024, https://arxiv.org/abs/2402.03300] 的核心：**用 group statistics 替代 critic**。对每个 prompt $s_n$，sample $J$ 个回答：
$$\hat A_n^j = \frac{R_n^j - \mu_n}{\sigma_n}, \quad \mu_n = \text{mean}(R_n^j), \sigma_n = \text{std}(R_n^j)$$

Loss：
$$J_{GRPO}(\theta) = \frac{1}{N}\sum_n \frac{1}{J}\sum_j \frac{1}{|a_n^j|}\sum_t \min(\rho_{nt}^j \hat A_{nt}^j, \text{clip}(\rho_{nt}^j, 1-\epsilon, 1+\epsilon)\hat A_{nt}^j)$$

变量：$\rho_{nt}^j = \pi_\theta(a_{nt}^j | \dots) / \pi_{\text{old}}(a_{nt}^j | \dots)$。

**Dr GRPO** [Liu et al. 2025, https://arxiv.org/abs/2503.20783] 发现除以 $\sigma_n$ 导致 difficulty bias——容易/难的 prompt 都被压扁。Fix：去掉 $\sigma_n$：
$$\hat A_{nj}^{\text{Dr GRPO}} = R_{nj} - \mu_n$$

### 7.3 DPO：直接 alignment

DPO [Rafailov et al. 2023, https://arxiv.org/abs/2305.18290] 推导：从 KL-regularized RL 的最优解
$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp(\beta^{-1} R(x,y))$$
反解 reward：
$$R^*(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

代入 Bradley-Terry：
$$p^*(y_w \succ y_l | x) = \sigma(\beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)})$$

Loss：
$$\mathcal{L}(\theta) = -\mathbb{E}[\log \sigma(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)})]$$

变量：$y_w$ 是 preferred answer，$y_l$ 是 rejected answer，$\beta$ 是 KL strength。

**DPO 的直觉**：把 RL 问题转换为分类问题——直接在 preference data 上 maximum likelihood，无需 reward model + RL loop。Limitation：只能用 preference data，不能用 verifiable reward。

### 7.4 Thinking models 与 marginal likelihood

DeepSeek-R1 [DeepSeek 2025, https://arxiv.org/abs/2501.12948] 用 GRPO + 可验证 reward（math、code）训练 thinking model。**Thinking as marginal likelihood maximization**：
$$\log p(y|x) = \log \sum_z p(y, z | x) = \log \int p(y|z, x) p(z|x) dz$$

变量：$z$ 是 latent thinking trace，$y$ 是最终答案。RL 只 optimize 最终 $y$ 的 correctness，不监督 $z$——这是 EM 风格的 latent variable optimization [Hoffman et al. 2023, https://openreview.net/forum?id=7p1tOZ13La]。

**R1-Zero 的 emergent ability** 争议：是否 RL 让模型"涌现"reasoning？[Liu et al. 2025, https://arxiv.org/abs/2503.20783] 等论证：base model 已经在 CoT 数据上预训练过，RL 只是"放大"既有能力。[Gandhi et al. 2025, https://arxiv.org/abs/2503.01307] 进一步：如果 base model 没见过 self-reflection pattern，RL 不会涌现出 reflection。

### 7.5 Multi-turn RL：RAGEN

RAGEN [Wang et al. 2025, https://arxiv.org/abs/2504.20073] 让 LLM 在每轮 generate thinking tokens + action tokens + STOP，environment 返回 observation。Algorithm 是 StarPO（State-Thinking-Action-Reward）：

```
for each step:
    generate N trajectories (parallel)
    compute MC return for each
    advantage = MC - baseline
    update policy
```

**长 horizon 的问题**：
1. Context grows——需要 truncation/summarization
2. REINFORCE estimator variance 大——需要 critic 或 ReBN [Liu et al. 2025, https://arxiv.org/abs/2510.01051]：
$$A_t^n = \frac{G_t^n - \mu}{\sigma}, \quad \mu = \text{mean}(G_{1:T}^{1:N}), \sigma = \text{std}(G_{1:T}^{1:N})$$

### 7.6 Inference-time scaling

Test-time compute 通过 posterior sampling 优化：
$$p(y | x, \mathcal{O}=1) \propto \exp(\beta^{-1} R(x,y)) \pi_{\text{ref}}(y|x)$$

**Best-of-N**：sample N，选 reward 最高。简单但 [Snell et al. 2024, https://arxiv.org/abs/2408.03314] 证明达到 optimal reward-KL tradeoff（在 N 大时）。

**Twisted SMC** [Lawson et al. 2022, https://arxiv.org/abs/2204.05375]：用 twist function $g_t(s_t)$ 近似 future reward，引导粒子向高 reward 路径。等价于在 token 生成时用 value function steer。

### 7.7 RLFT as amortized posterior sampling

KL-regularized RL 等价于 minimize $D_{KL}(\pi_\theta \| \pi^*)$，其中 $\pi^*$ 是 tilted posterior。This view 解释了为什么 fine-tune 后的模型"坍缩"到一种 style——reverse KL 是 mode-seeking。Forward KL $D_{KL}(\pi^* \| \pi_\theta)$ 则 mode-covering，保持多样性 [Korbak et al. 2022, https://arxiv.org/abs/2206.00761]。

---

## 8. 其他重要话题

### 8.1 Regret vs Bayes-optimal

**Bayes-optimal agent**：maximize $\mathbb{E}_{M \sim P_0}[V^\pi(M)]$，需要 prior $P_0$。Posterior uncertainty 自动驱动 exploration。

**Minimax regret agent**：$\min_\pi \max_M \text{Regret}_T(\pi | M)$，无需 prior，对抗鲁棒。

差异：
| | Bayes | Minimax |
|---|---|---|
| 需要 | prior over MDP | 类 $\mathcal{M}$ |
| 探索 | Bayesian optimal | UCB / Thompson |
| 对抗 | 依赖 prior 准确 | 鲁棒 |

### 8.2 Thompson sampling

Bandit case：
$$a_t = \arg\max_a R(s_t, a; \tilde\theta_t), \quad \tilde\theta_t \sim p(\theta | h_t)$$

变量：$\tilde\theta_t$ 是从 posterior 采样的 model parameter sample。**直觉**：sample 一个 model 假装它真实，按它最优 act。多次采样自然 explore 不同 action，posterior 收敛后 exploit。

MDP case：PSRL [Strens 2000, https://www.cs.colorado.edu/~binsted/courses/csci7000/papers/strens2000a.pdf]——每 episode 开始 sample 一个 MDP，solve for $\pi^*$，execute，update posterior。

### 8.3 UCB 与 optimism

Bandit UCB：
$$\tilde R_t(a) = \hat\mu_t(a) + \frac{c}{\sqrt{N_t(a)}}$$

变量：$\hat\mu_t(a)$ 是 empirical mean，$c$ 是 confidence scaling，$N_t(a)$ 是 visit count。**Optimism in face of uncertainty**——选最乐观的 action，自然探索 uncertain 区域。

MDP UCB：UCRL2 [Auer et al. 2008, https://papers.nips.cc/paper/2008/file/e4a6222cdb5b34375400904f03d8e6a5-Paper.pdf]——每 episode 估计 MDP 参数的 confidence set，solve 最优 MDP（在 set 内）。

### 8.4 Distributional RL

预测 return 分布而非 mean。$Z_t^\pi = \sum_{k=0}^{T-t} \gamma^k R(s_{t+k}, a_{t+k})$ 是 random variable。学 $p(Z|s)$。

**C51** [Bellemare et al. 2017, https://arxiv.org/abs/1707.06887]：histogram with 51 bins，cross-entropy loss。

**HL-Gauss** [Farebrother et al. 2024, https://arxiv.org/abs/2403.03950]：把 target $y$ 与 Gaussian 卷积再 discretize。比 MSE、two-hot、C51 都好。Intuition：cross-entropy 比 MSE 更 robust to noise target，HL 比 two-hot 更像 ordinal regression（更 soft target）。

### 8.5 Intrinsic motivation

**Knowledge-based**：奖励预测误差或 information gain
- RND [Burda et al. 2018, https://arxiv.org/abs/1810.12894]：fixed random target net + learned predictor，prediction error 是 intrinsic reward
- ICM [Pathak et al. 2017, https://arxiv.org/abs/1705.05363]：inverse dynamics model 学 controllable feature，forward model 在 feature space 预测——避免 noisy TV 问题

**Competence-based**：自设 goal 的 goal-conditioned RL
- DIAYN [Eysenbach et al. 2019, https://openreview.net/forum?id=SJx63jRqFm]：maximize $I(z; s_T)$，diverse skills
- Go-Explore [Ecoffet et al. 2021, https://www.nature.com/articles/s41586-020-03157-9]：先回到 archive 中"interesting"state，再 explore

**Noisy TV problem**：纯 prediction error 奖励会让 agent 卡在 stochastic source（电视放随机节目）。Fix：用 $D_{KL}(p^* \| q)$ 而非 cross-entropy $H(p^*, q)$——前者在 $q = p^*$ 时为 0 即使 $p^*$ 是 random。

### 8.6 Hierarchical RL

**Options** [Sutton et al. 1999, https://www.sciencedirect.com/science/article/pii/S0004370299000521]：$\omega = (\mathcal{I}_\omega, \pi_\omega, \beta_\omega)$
- $\mathcal{I}_\omega$：initiation set
- $\pi_\omega(a|s)$：intra-option policy
- $\beta_\omega(s) \in [0,1]$：termination probability

Semi-MDP dynamics：
$$T_\gamma(s'|s,\omega) = \sum_{k=1}^\infty \gamma^k \Pr(S_k = s', \beta_\omega(S_k) | S_0 = s, A_{0:k-1} \sim \pi_\omega)$$

**Feudal RL** [Dayan & Hinton 1992]：manager 给 goal $g$，worker 用 $\pi(a|s, g)$ 达到。HIRO [Nachum et al. 2018, https://arxiv.org/abs/1805.08296] 用 hindsight relabeling 处理 nonstationarity——manager 想要 $g_t$，worker 实际达到 $g_t'$，relabel $(s_t, g_t', r, s_{t+c})$。

### 8.7 Offline RL

**核心挑战**：从 fixed dataset 学 policy，不能 query 未知 $(s, a)$ 的 outcome。Q-learning 在 OOD action 上 overestimate。

**CQL** [Kumar et al. 2020, https://arxiv.org/abs/2006.04779]：conservative penalty
$$\mathcal{C}(\mathcal{B}, w) = \mathbb{E}_{s, a \sim \mu}[Q_w(s,a)] - \mathbb{E}_{s, a \sim \pi_b}[Q_w(s,a)]$$

变量：$\mu$ 是学到的 policy，$\pi_b$ 是 behavior policy。**Intuition**：压低 $\mu$ 选的 OOD action 的 Q，抬高 $\pi_b$ 选过的 in-distribution action 的 Q——pessimistic。

**Decision Transformer** [Chen et al. 2021, https://arxiv.org/abs/2106.01345]：sequence modeling 视角，condition on return-to-go (RTG)：
$$\arg\max_\theta \mathbb{E}[\log \pi_\theta(a_t | s_{\leq t}, a_{<t}, \text{RTG}_{\leq t})]$$

Test time 设 RTG 为 high value，autoregressive 生成 action。Limitation：在 stochastic 环境 fail——trajectory 达到 RTG 可能是运气。

### 8.8 Imitation learning 与 divergence minimization

GAIL [Ho & Ermon 2016, https://arxiv.org/abs/1606.03476] 用 GAN-style 训练：
$$\min_\pi \max_w \mathbb{E}_{p_{\pi_{\text{exp}}}^\gamma(s,a)}[T_w(s,a)] - \mathbb{E}_{p_\pi^\gamma(s,a)}[f^*(T_w(s,a))]$$

变量：$T_w$ 是 discriminator，$f^*$ 是 $f$-divergence 的 convex conjugate。统一了 GAIL、AIRL 等 [Ghasemipour et al. 2019, https://arxiv.org/abs/1911.02256]。

### 8.9 AIXI 与 universal AGI

AIXI [Hutter 2005, https://link.springer.com/book/10.1007/3-540-27559-0] 是 RL 的理论极限：
$$a_t = \arg\max_{a_t} \sum_{o_t, r_t} \cdots \max_{a_m} \sum_{o_m, r_m} [r_t + \cdots + r_m] \sum_{p: U(p, a_{1:m}) = (o_1 r_1 \cdots o_m r_m)} 2^{-\ell(p)}$$

变量：$U$ 是 universal Turing machine，$\ell(p)$ 是 program length。Prior $\Pr(p) = 2^{-\ell(p)}$ 是 Solomonoff induction。**Intuition**：在最短程序假设下做 expectimax。Intractable 但给出理论 anchor。

[Arumugam et al. 2024, https://direct.mit.edu/opmi/article-pdf/doi/10.1162/opmi_a_00132/2364075/opmi_a_00132.pdf] 的 Capacity-Limited Bayesian RL 是 computation-bounded 版本——结合 rate-distortion theory 与 Bayesian RL。

---

## 9. 整体直觉总结

读完这份综述，几个 cross-cutting intuition：

1. **Value function 是 compressed policy，policy 是 decompressed value**：Bellman equation 把无限 horizon 压缩成一步 lookahead，Q-learning 与 policy gradient 是同一枚硬币的两面。

2. **Bootstrap 是 RL 的根本张力**：TD 用自己的估计当 target，bias-variance tradeoff + deadly triad 都源自此。Target network、EMA、CQL 的悲观、JEPA 的 EMA target 都是 mitigate 这个张力。

3. **World model 是 amortized planning**：MCTS 是 explicit planning，Dreamer 是 implicit (imagination rollout)。两者本质相同——用 model 在脑中 simulate，再 act。Learned representation 决定 model 质量。

4. **RL as inference 的统一视角**：maxent RL、SAC、MPO、AWR、PPO clip 都可以从 variational inference 视角推。Optimality variable $\mathcal{O}$ 是万能 glue。

5. **Multi-agent 是 single-agent + opponent model**：self-play / fictitious play / CFR 都是不同方式 model "环境"中包含的其他 agent。Population-based 方法 robust 对 cyclical dynamics。

6. **LLM RL 是 contextual bandit 的特例**：single-turn 是 bandit，multi-turn 是真正的 MDP。R1-Zero 的"涌现"争议本质是 base model 的 capability + RL 的 amplification。

7. **Offline RL 的核心是 OOD 保守**：behavior policy 数据覆盖什么，学到的 policy 就只能在那里 reliable。CQL、uncertainty penalty、BC + Q 都是 pessimism 的不同实现。

8. **Exploration 是 epistemic uncertainty 的 exploitation**：Thompson sampling、UCB、intrinsic curiosity 都是从 model uncertainty 获取 information gain 的不同视角。

这份综述的最大价值是**把 RL 的所有子领域用概率推断串起来**——RL = sequential Bayesian decision making under uncertainty。从这个视角，DPO、SAC、Dreamer、AlphaZero、Thompson sampling 都是同一框架的不同 instantiation，差别在 latent variable 是什么、用什么 variational family、怎么 amortize inference。

参考资源：
- Murphy's book page: https://probml.github.io/
- Sutton & Barto: http://incompleteideas.net/book/RLbook2020.pdf
- Levine's offline RL tutorial: https://arxiv.org/abs/2005.01643
- OpenAI Spinning Up: https://spinningup.openai.com/
- Deep RL bootcamp: https://sites.google.com/view/deep-rl-bootcamp/
- Kevin Murphy Probabilistic ML book 2: https://probml.github.io/pml-book/book2.html
- CleanRL: https://github.com/vwxyzjn/cleanrl
- Tinker (LLM RL lib): https://tinker-docs.thinkingmachines.ai/
- GEM (multi-turn LLM RL): https://arxiv.org/abs/2510.01051
- TÜLU 3 post-training: https://arxiv.org/abs/2411.15124
