---
source_pdf: REDUCING BELIEF DEVIATION IN REINFORCEMENT.pdf
paper_sha256: 5aa2d5630e83291ab489ad291bd1fb9cd763d2a3011adcaaaab0018c503cf1e8
processed_at: '2026-08-11T22:00:36-07:00'
target_folder: Agentic-AI
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲讲这篇paper

## 一句话概括

LLM agent在多轮提问推理的时候，经常**越问越糊涂，最后陷入原地打转的死循环**。这篇paper发现：这种死循环不仅浪费token，还会通过RL的credit assignment机制**反过来污染前面那些好的推理步骤**，让模型学到错误信号。T³的办法就是在agent开始打转的那一刻**直接掐断trajectory**，只保留前面有价值的部分来训练。

---

## 问题是什么

想象你玩"20个问题"猜谜游戏——心里想一个东西，别人提问你来猜。

一个聪明的玩家会问"是动物吗？""会飞吗？"这种能大量排除可能性的问题。每问一个，hypothesis space就缩小一次，最后锁定答案。

但LLM agent玩这个游戏的时候经常翻车：
- 问到一半开始重复问已经问过的问题
- 问的问题跟之前的回答完全矛盾
- 在几个可能性之间来回横跳，无法收敛
- 越问越迷茫，最后胡乱猜一个

这就是paper说的**belief deviation**——agent心里那个"嫌疑人名单"已经跟实际情况对不上了，但它自己不知道，还在按错误名单继续问。

---

## 为什么会这样

paper用POMDP的框架解释：agent心里有一个belief $b_t$（对真相的概率分布），理想情况下应该用Bayesian更新来维护它。但LLM实际上是用自己的reasoning来近似这个更新，而这个近似是**有误差的**。

关键insight（Assumption 1）：**误差会随着uncertainty放大**。当agent还很迷茫（hypothesis space大）的时候，它要同时做很多事——记住历史、推理约束、想下一个问题——容易出错。每错一步，belief偏离更多，下一步更迷茫，更容易错。这是一个**正反馈循环**。

打个比方：像人在陌生城市找路，刚开始方向感还行，转错一个弯后开始怀疑自己，越怀疑越容易乱转，最后彻底迷失。

paper证明了：这个偏离过程会**必然**把agent带进一个"belief trap"——一旦进去，无论再问什么问题，belief都不会往真相靠拢了。就像走进了死胡同，往前走后退走都还是死胡同。

---

## 为什么RL救不了

你可能会想：用RL训练一下，让agent学会问好问题不就行了？

paper指出一个微妙的问题：**RL的credit assignment机制会把好的action也带坏**。

具体来说，outcome-based RL只在trajectory最后给一个reward（猜对没猜对）。GAE会把terminal reward"倒推"回前面的每个action，给每个action算一个advantage——"这个action对最终结果有多大贡献"。

现在考虑一个trajectory：
- 前5步：问得很好，hypothesis space快速缩小（informative prefix）
- 第6步：进入belief trap
- 第6-15步：原地打转，越问越糟（uninformative tail）
- 最后：猜错，reward=0

问题出在哪？GAE倒推的时候，tail里那10步"越问越糟"会产生一连串**negative TD-error**，这些negative信号通过discount累加，会**压过**prefix里5步的positive信号。

结果就是：本来应该被鼓励的好问题（prefix里的），它们的advantage被算成了负数。Policy gradient反而会**抑制**这些好问题，鼓励agent避开它们。这就是Theorem 2说的"advantage inversion"。

paper在Appendix C.2做了一个很漂亮的实验验证：看failed rollouts（最终猜错的）前500个token的GAE advantage，发现vanilla方法下early token的advantage明显被压低（negative drift），加了T³后被显著抬起。这直接观测到了理论预测的污染效应。

---

## T³怎么修

想法很直接：**既然tail会污染prefix，那把tail砍掉就行**。

具体做法：在trajectory的每一步检测"是不是已经进trap了"，一旦检测到，立刻truncate，只用前面的部分算gradient。

难点在于：你怎么知道agent进trap了？$b_t$ 在LLM里是隐式的，看不见。

T³的解法是用**observable proxy**——能从外部观测到的"停滞信号"。不同任务用不同的proxy：

**GuessNumbers**：如果agent猜了一个**已经被历史feedback排除的数字**，说明它belief tracking彻底失败了，立刻truncate（$k=1$）。

**SituationPuzzles**：如果judge连续5次回答"unknown"（问题太模糊无法回答），说明agent在问无效问题，truncate。

**CircuitDecoding**：如果连续3次query没有缩小candidate set，truncate。

**PreferenceEstimation**：如果agent的preference estimate连续2步往**远离**真相方向走，truncate。

这些proxy背后的common principle（T³ Condition, Def 2）：**hypothesis space连续k步没有显著收缩**就truncate。本质上是在检测"epistemic stalling"——agent的认知停止进步了。

paper还证明了（Proposition 1）：即使proxy有系统性偏差和噪声，只要设阈值合理，false-truncation概率会以**指数速度**下降。也就是说，这个方法对proxy的质量没那么敏感——不需要完美的detector，差不多就行了。

---

## 实验效果

5个任务，3个RL算法（PPO, GRPO, GSPO），Qwen-2.5-7B-Instruct：

**性能提升**：
- GuessNumbers + GRPO：61.26 → 91.36（**+30**）
- MovieRecommendation + GSPO：14.67 → 55.67（**+41**）
- CircuitDecoding + PPO：61.67 → 77.83（**+16**）
- 大部分指标都有提升，14/18个metrics显著改善

**Token效率**：平均rollout token减少34%。因为trajectory在trap处被切断，不再浪费token在无效循环上。比如PPO on CD达到0.65 reward只用66%的token。

**训练稳定性**：vanilla RL的reward曲线经常部分converge后突然collapse，T³的曲线更平滑、monotonic。

**OOD泛化**：在CD上把candidate pool从10扩大到30，T³依然有+4.2的提升；在PE上把reference movie数量变化，T³在所有设置下都positive。

**架构泛化**：3B模型受益有限，7B/14B受益明显；Qwen比LLaMA受益更多；DeepSeek-R1-distilled的LLaMA受益最大。这说明T³的效果依赖模型的belief tracking基础能力——底子太差truncate也救不了。

---

## 一个有意思的细节

paper在Appendix D.3探索了**不用ground-truth的proxy**——直接看agent自己的preference estimate变化量，如果连续几步变化很小就truncate。结果这个ground-truth-free版本达到了60.33的BinarySim，**比用ground-truth的版本（49.00）还高**。

这说明T³的principle（检测epistemic stalling）比具体的proxy实现更重要——甚至agent自身的output trajectory里就encoded了足够的信息来判断它是否在stall。

---

## 核心intuition总结

1. **LLM agent在multi-turn推理里失败的根源是belief tracking能力不足**，policy本身可能没问题
2. **belief偏差会正反馈放大**，最终必然进入trap（Theorem 1）
3. **trap后的uninformative tail会通过GAE污染prefix的credit assignment**，甚至inverce gradient方向（Theorem 2）——这是RL救不了的根本原因
4. **在trap entry处truncate能精确修复这个bias**（Corollary 1），且对proxy质量robust（Proposition 1）
5. **T³是algorithm-agnostic的meta-wrapper**，drop-in到任何policy optimization方法

**最深的takeaway**：在outcome-based RL for LLM agents中，瓶颈往往不在policy optimization算法，而在agent的belief representation质量。T³是一个practical workaround，但long-term的解法可能是让LLM更好地maintain belief state——无论是通过更大的模型、distillation、还是auxiliary training objective。

这跟model-based RL的困境很像：world model不准，再好的planning算法也白搭。LLM的"内部world model"（即它的reasoning能力）是根本瓶颈。

---

# 这篇paper的核心问题与直觉

让我用最sharp的方式概括这篇paper在做什么：**LLM agent在multi-turn active reasoning时，由于本身的reasoning能力有限，它的internal belief state $b_t$ 会偏离true latent state $s^\star$，且这种偏离具有累积放大效应，最终会进入一个"belief trap"，此后所有action都失去information gain。更严重的是，trap之后那段uninformative tail会通过GAE污染整个trajectory的credit assignment，甚至inverce early informative action的gradient方向。T³的核心修复就是在trajectory进入trap的瞬间truncate，保留informative prefix，让RL optimization专注于真正有信号的部分。**

这个故事的妙处在于：它把一个empirical现象（LLM agent会陷入重复、无效的循环）用POMDP + Bayesian filtering的语言重新formalize，并指出问题的根源不在policy本身而在belief representation。这是一个很经典的"agent能力瓶颈在world model而非decision making"的视角，与model-based RL里"learned world model不准导致planning失败"的思想遥相呼应。

---

# 1. 问题建模：POMDP与belief tracking的张力

## 1.1 Formal setup

active reasoning建模为POMDP $(\mathcal{S}, \mathcal{A}, \mathcal{O}, T, O, R, \gamma)$：
- $\mathcal{S}$: latent state space（agent看不到的真相，比如隐藏的数字、hidden circuit、user preference vector）
- $\mathcal{A}$: action space（提问、猜测、查询）
- $\mathcal{O}$: observation space（环境反馈）
- $O(o|s,a)$: observation model
- $s^\star \in \mathcal{S}$: 真实latent state，**episode内固定**（这是分析用，agent永远看不到）
- $\gamma \in (0,1]$: discount factor

paper构造两个对比reasoner：

**Oracle reasoner**用Bayesian filter:
$$b_{t+1}^\star(s) := B^\star(b_t^\star, a_t, o_t) = \frac{O(o_t|s, a_t) b_t^\star(s)}{p_b(o_t|a_t)}$$

其中 $p_b(o_t|a_t) := \sum_{s' \in \mathcal{S}} O(o_t|s', a_t) b_t^\star(s')$ 是Bayes normalizer（marginal likelihood）。这里 $b_t^\star \in \Delta(\mathcal{S})$ 是belief distribution over latent states。

**LLM agent**用 $B_\theta$:
$$b_{t+1}(s) := B_\theta(b_t, a_t, o_t)$$

$\theta$ 是LLM参数。注意这里 $b_t$ 不是显式存储的probability vector，而是隐式编码在LLM的intermediate reasoning trace和hidden activation里——这是后面T³不能直接用 $b_t$ 的根本原因。

## 1.2 Truth-anchored potential $\Psi(b)$

定义：
$$\Psi(b) := -\log b(s^\star)$$

- Domain: $\Psi(b) \in [0, \infty)$
- $\Psi(b) = 0$ iff $b(s^\star) = 1$（task完成，belief全部集中在true state）
- $\Psi(b)$ 是belief对true state的"surprisal"，衡量不信任程度

这个量借鉴了信息论和Bayesian experimental design里的information gain概念。本质上 $\Psi$ 是cross-entropy loss的一种变体——它只关心true state那一个维度，而不是整个distribution的entropy。

## 1.3 Belief-update discrepancy

$$c_\theta(b_t) := \mathbb{E}_{a_t \sim \pi(\cdot|b_t)} \mathbb{E}_{o_t \sim O(\cdot|s^\star, a_t)} [\Psi(B_\theta(b_t, a_t, o_t)) - \Psi(B^\star(b_t, a_t, o_t))]$$

直觉：给定当前belief $b_t$，如果分别用LLM update rule $B_\theta$ 和Bayesian rule $B^\star$ 更新一步，然后比较新belief对true state的surprisal差距。$c_\theta > 0$ 表示LLM update比Bayes update更"差"——把belief带离true state了。

$c_\theta$ 是paper里最关键的"偏差量"，它衡量LLM reasoning在每一步引入的noise。

---

# 2. Assumption 1: Update-Error Growth

paper的核心假设：存在常数 $m_\theta > 0, c_0 \geq 0, U_0 \geq 0$，使得对所有满足 $\Psi(b) \geq U_0$ 的belief：
$$c_\theta(b) \geq m_\theta \Psi(b) - c_0$$

**变量含义拆解**：
- $m_\theta$: LLM belief update的**error增长率**。$m_\theta$ 大意味着LLM在high-uncertainty区域update得很差。本质上 $m_\theta$ 量化了LLM的"reasoning能力"——弱模型 $m_\theta$ 大
- $c_0$: baseline offset，处理low-$\Psi$ regime（well-behaved region）
- $U_0$: high-uncertainty regime的threshold，只有 $\Psi \geq U_0$ 时这个linear growth才dominates

**直觉**：在low-uncertainty区域（agent已经很确信true state），LLM的update可能没那么差，因为answer空间小、容易保持consistency。但在high-uncertainty区域（agent还很迷茫），LLM要同时做belief tracking、action selection、reasoning，每一步都引入误差，且误差被uncertainty放大——这是positive feedback loop的根源。

这个assumption的物理图像是：**LLM的reasoning error与它当前所处的uncertainty level成正比**，越迷茫越容易犯更多错，越错越迷茫。这跟人类reasoning的"认知过载"现象非常像——当hypothesis space太大时，working memory hold不住，reasoning quality下降。

## 2.1 Empirical verification of Assumption 1 (Appendix C.1)

paper在PE任务上做了实际验证，但需要approximation，因为 $b_t$ 不可观测。Approximation strategy：

**Step 1**: 用 $\hat{\Psi}_t := \|w_t - w^\star\|_2^2$ 作potential proxy。$w_t$ 是agent报告的preference vector estimate，$w^\star$ 是ground-truth preference。这个proxy保留了 $\Psi$ 的核心性质（非负，task complete时为0）。

**Step 2**: 近似Bayesian update $B^\star$ 用linear-Gaussian update：
$$w'_{t+1} := w_t + K_t m_t (o_t - m_t^\top w_t), \quad K_t = \frac{\sigma_0^2}{\sigma_0^2 \|m_t\|_2^2 + \sigma^2}$$

变量含义：
- $m_t \in \mathbb{R}^d$: movie-attribute difference vector（query里两部电影属性差）
- $o_t \in \{-1, +1\}$: binary observation
- $\sigma_0^2, \sigma^2$: prior和observation noise variance，都设为1.0
- $K_t$: Kalman gain

**Step 3**: 构造observable proxy for $c_\theta$:
$$\hat{c}_\theta(b_t) := d(w_{t+1}) - d(w'_{t+1}) = \|w_{t+1} - w^\star\|^2 - \|w'_{t+1} - w^\star\|^2$$

这衡量LLM update和approximate Bayes update后，与ground-truth的距离差。

**Step 4**: 收集150k+ samples from Qwen-2.5-7B/32B rollouts，binning取10th percentile作lower envelope，linear fit得：
- Qwen-2.5-7B: $\hat{c}_\theta = 0.0969 \hat{\Psi} - 3.0478$（即 $\hat{m}_\theta = 0.0969, \hat{c}_0 = 3.05$，$\hat{U}_0 = 10$）
- Qwen-2.5-32B: $\hat{c}_\theta = 0.4655 \hat{\Psi} - 1.5158$（即 $\hat{m}_\theta = 0.4655, \hat{c}_0 = 1.52$，$\hat{U}_0 = 2$）

**值得注意的反直觉点**：32B模型的 $\hat{m}_\theta$ 比7B大5倍。paper里没有详细解释，但我推测几个原因：
1. proxy $\hat{\Psi}$ 本身不是真正的 $\Psi$，是 $\|w_t - w^\star\|^2$ 的proxy
2. 32B可能被放到更难的uncertainty区域采样
3. $\hat{B}$（linear-Gaussian surrogate）对32B的update rule相对优势更大——32B可能用更复杂的reasoning，反而偏离linear-Gaussian更多
4. 真正的Bayesian $B^\star$ 不一定是linear-Gaussian，这个surrogate本身可能对32B不公平

这个empirical verification是paper的弱点之一——它confirm了"线性下界"的存在性，但系数估计并不可靠。

---

# 3. Theorem 1: Belief Trap Region的存在性

## 3.1 Threshold $U$ 的构造

定义technical constant：
$$\bar{B} := 2(-L_\pi \log \eta + 1/\eta)$$

**变量含义**：
- $L_\pi$: policy的Lipschitz常数（Assumption B.3：$TV(\pi(\cdot|b), \pi(\cdot|b')) \leq L_\pi d(b, b')$，$d$ 是 $\ell_1$ distance）
- $\eta$: observation non-degeneracy bound（Assumption B.2：$O(o|s,a) \geq \eta > 0$，意味着no observation能完全排除某个state）

$\bar{B}$ 来自Lemma B.1和B.2的bound：informativeness $\mathcal{T}(b, a)$ 在belief和policy扰动下的敏感度。本质上 $\bar{B}$ 是"perfect oracle的informativeness"在LLM policy偏离最优policy下的上界偏差。

定义BTR threshold：
$$U := \max\{U_0, (\Psi_1^\star + \bar{B} + c_0)/m_\theta\}$$

直觉：threshold由两部分决定——assumption的active region $U_0$，以及oracle的initial surprisal $\Psi_1^\star$ 加上policy-imposed noise $\bar{B}$ 和baseline error $c_0$，除以增长率 $m_\theta$。$m_\theta$ 大时 $U$ 小，trap来得快。

定义trap entry time:
$$t_S := \inf\{t : \Psi_t \geq U\}$$

## 3.2 Theorem 1 statement (Proposition B.1 formal version)

如果 $t_S < \infty$，则对所有 $t \geq t_S$:
$$\mathcal{P}_\theta(b_t) \leq 0 \quad \Leftrightarrow \quad \mathbb{E}[\Psi(b_{t+1})|b_t] \geq \Psi(b_t)$$

其中 $\mathcal{P}_\theta(b) := \Psi(b) - \mathbb{E}_{a \sim \pi(\cdot|b)} \mathbb{E}_{o \sim O(\cdot|s^\star, a)}[\Psi(B_\theta(b, a, o))]$ 是agent的expected one-step progress。

**Mental model**: 一旦 $\Psi_t \geq U$，LLM agent的下一步update期望上**不会减少surprisal**。换言之，agent陷入了一个"absorbing region"，无论做什么动作，belief都不会向true state靠近。这就是Belief Trap Region (BTR)的定义。

## 3.3 Trap entry time的上界 (Proposition B.2)

加强Assumption 1到global（$U_0 = 0$），并假设oracle在trap前 $\Psi_t^\star \geq \mu > 0$ 对所有 $t < t_S$。定义 $\delta := m_\theta \mu - (c_0 + \bar{B}) > 0$，$\Delta_1 := \Psi_1 - \Psi_1^\star$（初始belief偏差）。则：
$$t_S \leq 1 + \left\lceil \log_{1+m_\theta} \frac{m_\theta U + \delta}{m_\theta \Delta_1 + \delta} \right\rceil$$

**关键变量含义**：
- $\Delta_1 = \Psi_1 - \Psi_1^\star$: agent初始belief相对于oracle初始belief的偏差
- $\mu$: oracle在trap前的minimum progress rate
- $\delta = m_\theta \mu - (c_0 + \bar{B})$: 一个"剩余纠正力"——LLM每步引入的error growth $m_\theta \mu$（因为 $\Psi^\star \geq \mu$，agent也至少要推进 $\mu$ 的progress，每步error是 $m_\theta \mu$ 量级）减去baseline修正力 $c_0 + \bar{B}$。$\delta > 0$ 表示error growth dominates correction。

**直觉解读**：
1. $m_\theta$ 大（弱模型）→ log底数 $1+m_\theta$ 大 → trap来得快
2. $\Delta_1$ 大（初始偏离严重）→ ratio小 → trap来得快
3. $\delta$ 大（剩余纠正力强）→ ratio接近1 → trap来得慢

这给出了一个**quantitative prediction**: 弱模型+差初始化=迅速陷入trap。这与实验里3B模型improvement limited的观察一致。

---

# 4. Theorem 2: Credit Assignment的失败

这是paper最核心的insight。考虑outcome-based RL，只有terminal step有非零reward，GAE estimator:
$$\hat{A}_t = \sum_{j=0}^{T-t-1} (\gamma\lambda)^j \delta_{t+j}$$

其中 $\delta_t = r_t + \gamma V_{t+1} - V_t$ 是TD-error，$\lambda \in [0,1]$ 是GAE参数。

## 4.1 关键假设

**Assumption (i)**: Value function calibration: $V_t = g(b_t(s^\star))$，其中 $g$ 可微且 $\inf_x g'(x) \geq \kappa_V > 0$。

含义：value function是true state belief的单调递增函数。$b_t(s^\star)$ 越大（越确信true state），value越高。$\kappa_V$ 是value对true state belief的敏感度下界。

**Assumption (ii)**: Belief drop in BTR: 存在 $\rho_b > 0$ 使得 $\mathbb{E}[b_{k+1}(s^\star) - b_k(s^\star)|\mathcal{F}_k] \leq -\rho_b$ 对所有 $k \geq t_S$。

含义：进入BTR后，true state的belief每步期望下降至少 $\rho_b$。换言之，agent在trap内越走越远离truth。

## 4.2 Theorem 2 statement (Theorem B.1)

对任何 $t < t_S$（即在trap之前的informative action）：
$$\mathbb{E}[\hat{A}_t] \leq \gamma \left(S_{pre}(t) - \kappa_V \rho_b S_{tail}^\ominus(t)\right)$$

其中：
- $S_{pre}(t) = \sum_{j=0}^{t_S - t - 1} (\gamma\lambda)^j$: prefix几何权重（trap前 $t$ 到 $t_S$ 的距离）
- $S_{tail}^\ominus(t) = \sum_{j=t_S - t}^{T - t - 2} (\gamma\lambda)^j$: tail几何权重（trap内 $t_S$ 到 $T-1$ 的距离）

**Advantage inversion的充分条件**：
$$\kappa_V \rho_b > \frac{S_{pre}(t)}{S_{tail}^\ominus(t)}$$

特别地，当 $\gamma\lambda \to 1$（长horizon agentic RL常用设置），这简化为：
$$\kappa_V \rho_b > \frac{\Delta}{L}$$

其中 $\Delta = t_S - t$ 是prefix长度，$L = T - 1 - t_S$ 是tail长度。

## 4.3 这个定理的mental model

考虑一个token位置 $t$ 在trap之前。它的GAE advantage是所有未来TD-errors的加权和：
$$\hat{A}_t = \underbrace{\sum_{j=0}^{t_S-t-1}(\gamma\lambda)^j \delta_{t+j}}_{\text{Pre}(t): \text{trap前的 informative 部分}} + \underbrace{\sum_{j=t_S-t}^{T-t-1}(\gamma\lambda)^j \delta_{t+j}}_{\text{Tail}(t): \text{trap后的 uninformative 部分}}$$

**Pre(t) 部分**: trap前的TD-errors，理论上 $\delta_k = \gamma V_{k+1} - V_k$（无中间reward），$V$ 在增长（agent在进步），所以 $\delta_k$ 期望为正。这部分是真正的"信号"。

**Tail(t) 部分**: trap后belief drop（Assumption ii），$V_{k+1} - V_k = g(b_{k+1}(s^\star)) - g(b_k(s^\star)) \leq -\kappa_V \rho_b$（由 $g' \geq \kappa_V$ 和 belief drop $\leq -\rho_b$）。所以 $\delta_k \leq -\gamma \kappa_V \rho_b$。这部分是"反信号"——negative drift污染advantage。

如果tail足够长，negative drift累积超过prefix的positive contribution，$\hat{A}_t$ 整体变负。**这意味着本应被encouraged的exploratory action反而被penalized**——gradient方向反了。

## 4.4 Corollary 1: Truncation修复bias

定义 $\hat{A}_t^{pre}$ 为在 $t_S$ 处truncate后的advantage estimator（即只保留Pre(t)部分，丢弃Tail(t)）。则：
$$\mathbb{E}[\hat{A}_t^{pre}] \geq \mathbb{E}[\hat{A}_t] + \gamma \kappa_V \rho_b S_{tail}^\ominus(t)$$

含义：truncate掉tail后，advantage estimate的bias被去掉至少 $\gamma \kappa_V \rho_b S_{tail}^\ominus(t)$——恰好是Theorem 2里的negative drift量。**这就是T³的理论基础**。

## 4.5 Empirical verification (Appendix C.2)

paper做了非常clever的empirical test：

1. 固定一个vanilla PPO trained policy
2. 生成两组rollouts：标准方法 vs T³ truncation
3. **只看failed rollouts**（final reward = 0），消除successful outcome的confounding
4. 计算前500 tokens的mean GAE advantage per token position

结果（Fig 8a, 8b）：
- w/o truncation: early-token advantage有clear **negative drift**——confirm了Theorem 2
- w/ T³ truncation: early-token advantage被**显著lift up**——confirm了Corollary 1

进一步的sensitivity tests:
- **Tail length effect** (Fig 8c): max turns从6增加到15（tail变长），early advantage被suppress更严重——定量验证 $\kappa_V \rho_b > \Delta/L$ 的方向
- **Truncation strength** (Fig 8d): 小 $k$（更激进truncate）→ cleaner, less-biased early advantages

这个empirical verification非常compelling——它直接观测到了Theorem 2预测的"negative drift污染early advantage"现象，且显示T³精确修复了它。

---

# 5. T³: From Theory to Practice

## 5.1 实施的两个挑战

理论上的truncation rule是 $t_S := \inf\{t : \Psi_t \geq U\}$，但实际不可观测：

1. **Belief modeling complexity**: $b_t$ 在LLM中是隐式的，编码在reasoning trace和hidden activation里。无法直接recover
2. **Unobservable thresholds**: $U, m_\theta, c_0, \bar{B}$ 都是agent-specific且无法直接测量

## 5.2 T³ Condition (Definition 2)

定义hypothesis space $\mathcal{H}_t$（step $t$ 时仍plausible的latent states）。T³ truncation condition：存在 $\Delta_{min} \geq 0$，使得在窗口 $[t-k, t)$ 内所有steps $\tau$ 满足：
$$d(\mathcal{H}_\tau, \mathcal{H}_{\tau+1}) \leq \Delta_{min}$$

**变量含义**：
- $\mathcal{H}_t$: step $t$ 时的hypothesis space（与interaction history一致的candidates）
- $d(\cdot, \cdot)$: refinement measure，量化hypothesis set在两步间的contraction程度
- $\Delta_{min}$: minimal informative update threshold
- $k$: 窗口大小，要求**sustained stall**而非单步noise

**关键观察**: 如果 $\mathcal{H}_t$ 有限可枚举，且agent belief是uniform over $\mathcal{H}_t$（假设 $s^\star \in \mathcal{H}_t$），则：
$$\Psi(b_t) = -\log b_t(s^\star) = -\log \frac{1}{|\mathcal{H}_t|} = \log|\mathcal{H}_t|$$

这提供了 $\Psi$ 的**exact observable surrogate** in finite enumerable cases。这是T³从理论到实践的关键桥梁——hypothesis space size直接是surprisal的proxy。

## 5.3 Proposition 1: False-truncation probability bound

设true single-step potential progress $g_t := \Psi(b_t) - \Psi(b_{t+1})$，observable refinement signal $d_t := d(\mathcal{H}_t, \mathcal{H}_{t+1})$。

假设：
- (i) BTR外 $g_t \geq \rho > 0$（uniform positive margin）
- (ii) Biased Gaussian noise model: $d_t = g_t + \beta_t + \xi_t$，$|\beta_t| \leq M_d$（systematic bias），$\xi_t \sim \mathcal{N}(0, \sigma^2)$ i.i.d.（stochastic noise）

如果 $\Delta_{min} < \rho - M_d$，则T³ rule在任意k-step non-BTR segment上的false-truncation概率 $\leq \delta \in (0,1)$ 的充分条件：
$$k(\rho - M_d - \Delta_{min})^2 \geq 2\sigma^2 \log(1/\delta)$$

**直觉解读**：
- 减小proxy bias $M_d$（design更好的proxy）：false-truncation率以平方速度下降
- 增大 $k$（要求更长stall）：false-truncation率线性下降
- 减小 $\Delta_{min}$（更敏感的threshold）：false-truncation率以平方速度下降
- 减小noise $\sigma^2$：false-truncation率线性下降

这个bound给出了T³的**statistical robustness guarantee**——即使proxy有systematic bias和noise，T³ rule依然能在non-BTR segment上保持低false-truncation率。

## 5.4 Task-specific instantiations

### GuessNumbers (GN)

- **Task**: 猜a位数字，每位从b个unique symbols中无重复采样。反馈xAyB（x位正确位置，y位正确但错位）
- **Hypothesis space**: $\mathcal{H}_t$ = 所有与 $\{a_{\leq t}, o_{\leq t}\}$ 一致的candidate numbers
- **Refinement metric**: $d(\mathcal{H}_\tau, \mathcal{H}_{\tau+1}) := |\mathcal{H}_\tau| - |\mathcal{H}_{\tau+1}|$（candidate set减少量）
- **Truncation**: $k=1$，如果 $a_t \notin \mathcal{H}_{t-1}$（猜测违反了已积累的逻辑约束，等价于 $d(\mathcal{H}_{t-1}, \mathcal{H}_t) \leq 0$）

这是一个非常elegant的instantiation——直接用set size reduction作refinement metric，且 $k=1$ 因为"猜一个已被排除的数字"是明确的belief tracking failure signal。

### SituationPuzzles (SP)

- **Task**: 解paradoxical puzzle，通过yes/no questions to judge model
- **Hypothesis space**: $\mathcal{H}_t$ = plausible explanations consistent with dialogue history
- **Proxy**: judge回应"unknown" 视为 $d < \Delta_{min}$（uninformative）
- **Truncation**: $k=5$ 连续unknown

SP的 $\mathcal{H}_t$ 不bounded，无法直接用set size，所以用judge feedback作proxy。"unknown"通常意味着question太vague或irrelevant，无法进一步缩小hypothesis space。

### CircuitDecoding (CD)

- **Task**: 从candidate pool中识别hidden boolean circuits
- **Hypothesis space**: $\mathcal{H}_t$ = 与interaction history一致的surviving candidates
- **Refinement metric**: $d := |\mathcal{H}_\tau| - |\mathcal{H}_{\tau+1}|$
- **Truncation**: $k=3$ 连续 $d \leq 0$（candidate set不收缩）

类似GN，但 $k=3$ 因为circuit query的information gain更复杂，单次不收缩可能是noise。

### PreferenceEstimation (PE) / MovieRecommendation (MR)

- **Task**: 推断hidden preference vector $v^\star$，通过pairwise comparison query
- **Hypothesis space**: $\mathcal{H}_t$ = plausible preference vectors consistent with feedback（连续空间，不可枚举）
- **Proxy**: agent报告的estimate $v_t$，用 $\mathrm{Sim}(v_{\tau+1}, v^\star) - \mathrm{Sim}(v_\tau, v^\star)$
- **Truncation**: $k=2$ 连续similarity下降

PE是连续hypothesis space，无法用set size，所以用agent自身的estimate trajectory作proxy。需要access to ground-truth $v^\star$ 在training时（test时不需要，因为只训练用）。

Appendix D.3探索了**不用ground-truth**的alternative proxy：
$$\mathrm{stall}_t = \mathbb{I}\left[\left(\frac{1}{k}\sum_{j=t-k}^{t-1}\|\hat{v}_{j+1} - \hat{v}_j\|_2\right) < \varepsilon\right]$$

即agent belief update的moving average小于threshold就truncate。$\varepsilon$ 从offline rollouts的quantile选。结果显示60% quantile达到60.33 BinarySim，**超过oracle-based T³**的49.00——这暗示general-purpose truncation detector是promising direction。

---

# 6. 实验结果深度解析

## 6.1 Main results (Table 1)

5个任务 × 3个RL算法（PPO, GRPO, GSPO），Qwen-2.5-7B-Instruct：

| Task | Vanilla best | T³ best | Gain |
|------|--------------|---------|------|
| CD (EM) | 79.33 (GRPO) | 81.33 (GRPO+T³) | +2.0 |
| SP (F1-word) | 36.63 (GSPO) | 39.45 (GRPO+T³) | +2.82 |
| GN (EM) | 96.07 (GSPO) | 99.74 (GSPO+T³) | +3.67 |
| PE (BinarySim) | 59.00 (GSPO) | 62.00 (GSPO+T³) | +3.0 |
| MR (EM) | 24.33 (PPO) | 55.67 (GSPO+T³) | +31.34 |

**关键观察**：
- **GN (GRPO)**: 61.26 → 91.36，**+30.1**——最显著的improvement。GRPO在GN上vanilla很差，T³救了它
- **MR (GSPO)**: 14.67 → 55.67，**+41.0**——MR需要OOD generalization，T³显著improve OOD robustness
- **CD (PPO)**: 61.67 → 77.83，**+16.2**——PPO在CD上vanilla unstable，T³ stabilize
- **GN (GSPO)**: 96.07 → 99.74，接近perfect

## 6.2 与frontier reasoning models对比

o3-mini和Gemini-2.5-Pro在finite hypothesis space tasks（GN, CD）上strong，但在unbounded hypothesis space tasks（SP, PE）上lag behind T³-equipped Qwen-7B。这暗示**large-scale RL with outcome reward alone不足以解决unbounded active reasoning**——explicitly addressing credit assignment（如T³）提供complementary benefit。

## 6.3 Training stability (Fig 3)

vanilla RL training dynamics有高方差，reward部分converge后会collapse。T³ training curves更monotonic、stable。这是T³除performance外的dual benefit。

## 6.4 Token efficiency (Fig 4)

虽然早期reward增长稍慢，但average rollout token数显著降低（up to 34% reduction）。例如：
- PPO on CD: 达到0.65 reward，T³只用66.4% tokens
- GSPO on GN: 达到0.96 reward，T³只用76.3% tokens

这个token efficiency improvement对production deployment非常重要——rollout cost是RL训练的主要bottleneck之一。

## 6.5 OOD generalization (Table 2)

**CD (PPO)**: 
- Candidate size 10→30: T³ gain从+18.5到+4.2，但始终positive
- Hidden circuit 2→4: T³ gain从+18.5到+6.6

**PE (PPO)**:
- Reference size 5→30: T³ gain在S=20时最大（+12.7）
- Sampling distribution: max-skewed时T³ gain最大（+10.7）

**Intuition**: 太少references增加ambiguity，太多introduce noise and redundancy，都exacerbate belief-trap dynamics。T³在moderate context最优。

## 6.6 Ablation on truncation conditions (Table 3)

**Window size $k$**:
- SP: $k=5$ 最优（+2.99 F1-word），$k=9$ diminishing（+0.50），$k=3$ 也有效（+2.16）
- CD: $k=3, 4$ 最优（+16.2, +17.6），$k=2$ 也好（+7.50）
- PE: $k=2$ 最优（+7.00），$k=4$ diminish（+2.33），$k=7$ zero gain

**Sim-based proxy (SP)**: 用E5-large-v2 embedding计算query语义相似度，$\alpha=0.9$ 阈值时 +2.98 F1-word，与 $k=5$ 相当。说明T³对proxy formulation robust。

**Random truncation (CD/PE)**:
- CD: $\beta=0.1$ mild improvement（+7.33），$\beta=0.5$ 大幅下降（-48.5）
- PE: $\beta=0.2$ mild improvement（+1.33），$\beta=0.8$ 下降（-3.0）

**Key insight**: 即使random truncation也有mild improvement——这证明BTR问题本身严重，任何形式的tail removal都有帮助。但过激进random truncation会prematurely terminate informative trajectories。

## 6.7 Truncation ratio dynamics (Fig 5)

- **Unbounded space (SP, PE)**: 高且稳定的truncation ratio从早期开始最优。例如SP $\alpha=0.9$ truncation ratio near 1.0最好
- **Finite space (CD)**: 低到中等truncation ratio最优。$k=3,4$ 保持low truncation frequency

**Insight**: unbounded hypothesis space更容易陷入redundancy-induced BTR，需要aggressive truncation；finite space的BTR entry更rare，over-truncation会损失informative prefix。

## 6.8 Architecture impact (Fig 6)

**Size effect** (PE, CD):
- Qwen-2.5-3B: limited improvement
- Qwen-2.5-7B/14B: substantial gains
- 14B tends to benefit more than 7B from T³

**Architecture type** (CD):
- LLaMA-3.1-8B: marginal improvement under T³
- Qwen-2.5-7B: substantial
- DeepSeek-R1-Distill-LLaMA-8B: largest performance gain

**Explanation via $m_\theta$**: 弱模型 $m_\theta$ 大，trap来得快，truncate也救不了——trap后trajectory已经太短，informative prefix不够训练信号。Distillation可能改善belief tracking能力，减小 $m_\theta$，使T³更effective。

---

# 7. 批判性思考与extension

## 7.1 与相关工作的关系

**vs. CURIO (Wan et al. 2025)**: CURIO构造potential function over ideal belief state给intermediate reward，假设latent space有限可枚举。T³不假设enumerable space，且不需要reward shaping，只在trajectory层面truncate。CURIO是"加positive signal"，T³是"去除negative signal"。

**vs. SPA-RL (Wang et al. 2025)**: SPA-RL训练reward model给intermediate rewards，enforce summation constraint。T³不需要额外reward model，zero-overhead。

**vs. Sotopia-RL (Yu et al. 2025)**: 用proprietary LLM做reward labeling。T³完全rule-based或基于agent自身输出，可完全offline。

**vs. Agent-R (Yuan et al. 2025)**: Agent-R用self-reflection训练agent从error中recover。T³是从training data level fix，不是inference time。

**vs. UoT (Hu et al. 2024)**: UoT量化每个question的uncertainty reduction，inference time strategy。T³是training time mechanism，但理论上可以与UoT组合。

## 7.2 Theory vs. Practice的gap

**Gap 1**: Assumption 1的empirical verification (Appendix C.1) 用了proxy $\hat{\Psi} = \|w_t - w^\star\|^2$，但真正的 $\Psi(b) = -\log b(s^\star)$ 是surprisal，不是L2距离。这两个量在belief是Gaussian时相关，但一般不等价。

**Gap 2**: Assumption (ii) of Theorem 2——belief drop in BTR: $\mathbb{E}[b_{k+1}(s^\star) - b_k(s^\star)] \leq -\rho_b$。这个assumption很强——它假设BTR内true state belief单调下降。实际上BTR可能只是stagnant（$\rho_b = 0$），这时Theorem 2的negative drift就不存在。Paper没验证这个assumption的empirical magnitude。

**Gap 3**: T³ condition (Def 2) 用hypothesis space contraction作proxy，但实际instantiation（如SP用judge "unknown"）不一定严格对应 $d(\mathcal{H}_\tau, \mathcal{H}_{\tau+1})$。Proposition 1的robustness guarantee依赖biased Gaussian noise model，这个model本身是stylized assumption。

## 7.3 未充分探索的方向

**General-purpose detector**: Appendix E.1提到hidden state signal做detector的方向——LLM的某些layer的hidden state相似度可以encode reasoning stall。这是一个非常promising的方向，可能lead to architecture-agnostic T³。

**Adaptive threshold**: Appendix D.4的adaptive $\varepsilon$（每6步update）在 $\alpha=0.6$ 时达到60.33 BinarySim，超过oracle-based T³。这暗示adaptive rule有potential，但paper没深入。

**Combination with reward shaping**: T³是trajectory truncation，但理论上可以与intermediate reward shaping组合。例如在trap entry point给一个negative reward signal explicit标记BTR。

**Multi-agent setting**: T³的设计是single agent，但multi-agent辩论、collaborative reasoning中的belief tracking failure可能更复杂。

## 7.4 与model-based RL的深层联系

T³的framework让我联想到model-based RL里的几个概念：

1. **World model inaccuracy**: LLM的 $B_\theta$ 是implicit world model。World model不准导致planning失败——这是T³揭示的failure mode的根源
2. **Imagination-based rollout truncation**: 一些model-based RL工作在world model confidence低时truncate rollout，类似T³
3. **Belief representation in Dreamer系列**: Dreamer用RSSM maintain belief state，explicitly model belief。LLM agent没有这种explicit belief representation，导致T³必须用observable proxy

这个connection暗示未来可能direction：让LLM agent explicit maintain belief state（例如通过auxiliary training objective），可能从根本上减小 $m_\theta$。

---

# 8. 总结：T³的core insight

T³给了一个非常clean的story：

1. **Diagnosis**: LLM agent的belief deviation累积放大（Assumption 1），最终进入BTR（Theorem 1）
2. **Mechanism**: BTR内的uninformative tail通过GAE污染early informative action的credit assignment，甚至inverce gradient方向（Theorem 2）
3. **Fix**: 在BTR entry处truncate trajectory，保留informative prefix（Corollary 1）
4. **Implementation**: 用observable proxy signal（hypothesis space contraction, query redundancy, belief estimate convergence）approximate BTR entry（T³ Condition, Def 2）
5. **Robustness**: 即使proxy有bias和noise，T³ rule也能以exponential rate控制false-truncation（Proposition 1）

T³作为meta-wrapper可以drop-in到PPO/GRPO/GSPO，不需修改algorithm本身，且带来dual benefit: stability + token efficiency + performance。

**最重要的takeaway**: 在outcome-based RL for LLM agents中，**belief tracking quality是核心瓶颈，而非action selection**。Improving belief representation（通过更大模型、distillation、auxiliary training）比改进policy optimization algorithm可能更有效。T³是一个practical workaround，但long-term solution是让LLM agent更好地maintain belief state。

---

# 参考链接

- **Paper PDF** (基于作者和标题推测): https://arxiv.org/abs/2509 (具体arxiv ID需要查证)
- **AR-Bench** (Zhou et al. 2025): https://arxiv.org/abs/2506.08295
- **Multi-Turn Puzzles** (Badola et al. 2025): https://arxiv.org/abs/2508.10142
- **GAE** (Schulman et al. 2015): https://arxiv.org/abs/1506.02438
- **PPO** (Schulman et al. 2017): https://arxiv.org/abs/1707.06347
- **GRPO / DeepSeekMath** (Shao et al. 2024): https://arxiv.org/abs/2402.03300
- **GSPO** (Zheng et al. 2025): https://arxiv.org/abs/2507.18071
- **CURIO** (Wan et al. 2025): https://arxiv.org/abs/2504.03206
- **SPA-RL** (Wang et al. 2025): https://arxiv.org/abs/2505.20732
- **Sotopia-RL** (Yu et al. 2025): https://arxiv.org/abs/2508.03905
- **UoT** (Hu et al. 2024, NeurIPS): https://arxiv.org/abs/2410.20127
- **E5 embeddings** (Wang et al. 2022): https://arxiv.org/abs/2212.03533
- **Verl / Hybridflow** (Sheng et al. 2025): https://arxiv.org/abs/2409.19056
- **Qwen2.5** (Yang et al. 2024): https://arxiv.org/abs/2412.15111
- **POMDP** (Kaelbling et al. 1998): https://www.sciencedirect.com/science/article/pii/S0004370298000231
- **Agent-R** (Yuan et al. 2025): https://arxiv.org/abs/2501.11425
- **DeepSeek-R1** (Guo et al. 2025): https://arxiv.org/abs/2501.12948
- **Cognitive behaviors for reasoning** (Gandhi et al. 2025): https://arxiv.org/abs/2503.01307
- **LLMs get lost in multi-turn** (Laban et al. 2025): https://arxiv.org/abs/2505.06120
