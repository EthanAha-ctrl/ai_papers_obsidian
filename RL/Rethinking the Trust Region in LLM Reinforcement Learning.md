---
source_pdf: Rethinking the Trust Region in LLM Reinforcement Learning.pdf
paper_sha256: 75b09690dcecc6e089eebf6212802a2d14c73cbad2048b7b2444a73708af618f
processed_at: '2026-08-11T23:41:23-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

这篇 paper 的核心发现可以用一句大白话讲清楚：**PPO 用来防止模型学歪的“安全锁”，在 LLM 这种拥有十几万词汇的巨兽面前，锁的机制完全错位了。** DPPO 重新打造了一把符合理论规范的锁。

下面我用人话结合底层技术细节，为你拆解这篇 paper 的精髓。

---

### 1. PPO 的“安全锁”到底出了什么问题？

在 RL 调教 LLM 时，PPO 的核心机制是 ratio clipping。它会计算新 policy 和旧 policy 对同一个 token 输出概率的比值 $r_t$。如果这个比值偏离 1 太多（比如超出了 $[1-\epsilon, 1+\epsilon]$ 的范围），PPO 就会“锁死”这个更新，截断梯度。

这个设计的初衷是维持一个 "Trust Region"（信任区间），防止 policy 一步跨太远导致崩溃。但这个机制在 LLM 的 long-tailed vocabulary（长尾词表）下产生了极其严重的错觉。

**核心错觉：把“比例变化”等同于“概率质量流动”**

我们来看 paper 里举的绝佳例子：
假设模型在某个状态下，有两个 token 可供选择。
- **低频 token**：旧 policy 概率 $\mu = 10^{-4}$，新 policy 概率 $\pi = 10^{-2}$。比例 $r = 100$。PPO 看到比例是 100，远超 $1.2$ 的上限，立刻锁死，停止学习。
- **高频 token**：旧 policy 概率 $\mu = 0.99$，新 policy 概率 $\pi = 0.80$。比例 $r \approx 0.808$。PPO 看到比例在 $[0.8, 1.2]$ 内，觉得“很安全”，放行更新。

直觉上就能感觉到不对劲：低频 token 的概率只增加了 $0.0099$，对整体概率分布几乎没影响；而高频 token 的概率硬生生跌了 $0.19$，这足以让模型在这个状态下的行为发生天翻地覆的变化。PPO 恰好抓错了重点。

**技术深潜：为什么 PPO 会犯这种错？**

从理论上看，PPO 试图约束的真实物理量是 Total Variation (TV) divergence，公式如下：
$$D_{TV}(\mu(\cdot|s_t) \| \pi(\cdot|s_t)) = \frac{1}{2} \mathbb{E}_{y_t \sim \mu}\left[|r_t - 1|\right]$$
变量解析：
- $D_{TV}$: Total Variation divergence，范围 $[0, 1]$，衡量两个概率分布的差异。
- $\mu(\cdot|s_t)$: 旧 policy（behavior policy）在 state $s_t$ 下的 token 概率分布。
- $\pi(\cdot|s_t)$: 新 policy（target policy）在 state $s_t$ 下的 token 概率分布。
- $r_t$: 概率比 $r_t = \frac{\pi(y_t|s_t)}{\mu(y_t|s_t)}$。
- $\mathbb{E}_{y_t \sim \mu}$: 在旧 policy 分布下对 token $y_t$ 求期望。

PPO 的 clipping 实际上是在约束这个期望里的**单样本 Monte Carlo 估计**。它只看到了被采样出来的那一个 token 的 $|r_t - 1|$，却试图以此推断整个分布的 divergence。在传统 RL 中（比如 Atari 只有十几个动作），单样本估计勉强够用；但在 LLM 中，词表高达 15 万，且概率分布极度倾斜，单样本估计的方差大到离谱，导致严重的误判。

---

### 2. DPPO 的解药：直接看真实的“概率质量流动”

DPPO (Divergence Proximal Policy Optimization) 的思路极其直接：既然单样本 ratio 靠不住，那我就直接计算真实的概率分布差异 $D$（可以是 TV divergence 也可以是 KL divergence）来决定要不要锁死更新。

DPPO 的 mask 机制公式如下：
$$M_t^{DPPO} = \begin{cases} 
0, & \text{if } (\hat{A}_t > 0 \text{ and } r_t > 1 \text{ and } D > \delta) \\
 & \text{or } (\hat{A}_t < 0 \text{ and } r_t < 1 \text{ and } D > \delta) \\ 
1, & \text{otherwise} 
\end{cases}$$
变量解析：
- $M_t^{DPPO}$: 二值 mask，$0$ 代表截断此 token 的梯度，$1$ 代表放行。
- $\hat{A}_t$: 估算的 advantage（收益减去基线）。
- $r_t$: 依然是概率比。
- $D$: 整个 token 概率分布的 divergence（如 TV 或 KL）。
- $\delta$: 设定的 divergence 阈值。

**保留 PPO 的不对称性**
注意这个公式里的条件：只有在 $r_t > 1$（推高概率）且 $D > \delta$（分布偏离过大）时才截断。如果 $\hat{A}_t > 0$ 且 $r_t < 1$（ advantage 是正的，但概率比却小于 1，说明模型正在向 trust region 内部收缩），此时即便 $D$ 很大，DPPO 也不会截断。这种不对称性允许模型快速纠错，把偏离的分布拉回正轨。

---

### 3. 计算开销怎么省？神奇的 Binary 近似

直接算 $D_{TV}$ 需要遍历 15 万个 token 的词表，算出 $\frac{1}{2} \sum_{a \in V} |\mu(a|s_t) - \pi(a|s_t)|$，这会导致显存爆炸。DPPO 提出了一个极其廉价且精妙的近似方法：Binary Approximation。

把 15 万分类的分布，压缩成关注当前 token 的二分类问题：当前 sampled token $a_t$ vs 其他所有 token。

Binary TV 的公式：
$$D_{TV}^{Bin}(t) = |\mu(a_t|s_t) - \pi(a_t|s_t)|$$
变量解析：
- $a_t$: 当前实际采样出来的那个 token。
- $\mu(a_t|s_t)$: 旧 policy 对这个 token 的概率。
- $\pi(a_t|s_t)$: 新 policy 对这个 token 的概率。

**为什么这个简单的绝对差是真实 TV 的 principled lower bound？**
原理在于 Triangle Inequality（三角不等式）。
真实的 TV 是所有 token 概率差绝对值之和的一半。如果我们将词表划分为两个集合：$C_1 = \{a_t\}$（当前 token）和 $C_2 = V \setminus \{a_t\}$（其他所有 token）。
那么根据三角不等式：
$$\left| \sum_{a \in C_j} (\mu(a) - \pi(a)) \right| \leq \sum_{a \in C_j} |\mu(a) - \pi(a)|$$
所以，聚合后的分布差异 $D_{TV}^{\mathcal{C}}$ 必然小于等于真实的 $D_{TV}$。这个近似只用了两个标量相减，计算开销几乎为零，但在实验中却达到了和 Top-K 近似几乎一样的效果。

---

### 4. 实验中的三个致命发现

这篇 paper 除了提出算法，还通过严密的消融实验揭示了 LLM RL 训练中的三个工程黑洞：

**发现一：极小 Learning Rate 下 Trust Region 依然必不可少**
实验中将 learning rate 降到极低的 $10^{-6}$，按理说这么小的步伐不可能把模型学崩。如果完全不加 Trust Region（如 PG-IS 算法），训练到后期必然崩溃。原因在于 training-inference mismatch。训练引擎和推理引擎由于底层算子实现不同，存在微小的数值误差。如果没有 Trust Region 兜底，这些微小误差会在长序列生成中复合放大，最终导致训练崩溃。

**发现二：Trust Region 必须锚定在 Rollout 时的 Behavior Policy $\mu_{\theta'}$**
开源界有一种做法叫 Decoupled Objective，为了省事，把 Trust Region 建立在重新计算的 on-policy 分布 $\pi_{\theta'}$ 上。实验证明这会导致极其严重的崩溃。因为采样出的数据是由 $\mu_{\theta'}$ 生成的，只有以 $\mu_{\theta'}$ 为锚点，importance sampling 的无偏性才能成立。偏离这个锚点，梯度估计就带毒了。

**发现三：真正的崩溃元凶是负样本上的暴力惩罚**
Paper 做了一个巧妙的隔离实验。在容易崩溃的 PG-IS 算法上，只加一个极简的 mask：当 advantage 为负，且旧 policy 对该 token 概率很高，但新 policy 猛烈压低它概率（$\mu - \pi > \delta$）时，截断梯度。结果发现仅仅这一个操作就稳住了训练。如果模型本来觉得某个词（比如 "Thus"）概率很高，但在某条错误推理路径中出现了，naive gradient 会拼命压低 "Thus" 的概率，这会破坏模型底层的语言知识，引发连锁崩溃。

---

### 5. 建立 Intuition：电影审查员的类比

把 LLM 想象成一部 15 万帧的极长电影，每一帧是一个 token。

PPO 就像是一个只盯着单帧像素差异的审查员。一帧暗角里的某个像素亮度从 $1$ 变成了 $100$，比例飙升，审查员大惊失色：“变化太剧烈，剪掉！”；而主角的衣服颜色从纯红变成了纯绿，整体色温剧变，审查员看了一眼说：“像素坐标没变多少，比例才 $0.8$，安全，放行。”

DPPO 就像是一个看整体能量流动的审查员。它把电影每一帧的全局色彩能量算一遍。暗角像素那点亮度变化，对全局能量影响微乎其微，放行；主角衣服变色导致全局能量结构剧变，咔嚓剪掉。

这就是为什么 DPPO 能够在保留低频 token 探索能力的同时，卡死高频危险更新的根本原因。

Reference:
- DPPO 原始 Paper: https://github.com/sail-sg/Stable-RL
- PPO 算法原始论文: https://arxiv.org/abs/1707.06347
- LLM Training-inference Mismatch 分析: https://arxiv.org/abs/2510.26788
- GRPO 与 DeepSeekMath: https://arxiv.org/abs/2402.03300

---

# Rethinking the Trust Region in LLM Reinforcement Learning - Deep Dive

## 1. Core Insight: PPO的Ratio Clipping是TV Divergence的Noisy Single-Sample Estimate

这篇paper的核心insight可以从一个看似平凡的观察出发。PPO的clipping机制 $|r_t - 1| \leq \epsilon$ 表面上是一个heuristic，但它实际上隐含地试图约束一个理论量 - TV divergence。关键公式是：

$$D_{TV}(\mu(\cdot|s_t) \| \pi(\cdot|s_t)) = \frac{1}{2} \mathbb{E}_{y_t \sim \mu}\left[|r_t - 1|\right]$$

变量解析：
- $D_{TV}$ 是Total Variation divergence，衡量两个distribution的差异，取值$[0, 1]$
- $\mu(\cdot|s_t)$ 是behavior policy（rollout时用的policy）在state $s_t$ 下的token distribution
- $\pi(\cdot|s_t)$ 是当前正在优化的policy的token distribution
- $r_t = \frac{\pi(y_t|s_t)}{\mu(y_t|s_t)}$ 是probability ratio
- $\mathbb{E}_{y_t \sim \mu}$ 表示在behavior policy下对token $y_t$ 求期望

PPO的clipping condition $|r_t - 1| \leq \epsilon$ 其实就是在约束这个expectation的**single-sample Monte Carlo estimate**。这个观察极其重要，因为它揭示了PPO在LLM regime下失效的root cause。

**为什么classical RL中这个approximation够用？** 因为classical RL的action space有限（比如Atari的离散actions，MuJoCo的连续actions维度也小）。单样本估计虽然noisy，但每个action被sample到的概率不会太极端，ratio的variance是可控的。

**为什么LLM regime下这个approximation崩溃？** LLM的vocabulary动辄15万+ tokens，而且分布是long-tailed。这就导致了paper中那个经典的例子：

```
Low-prob token:  μ(a_low|s) = 10^-4, π(a_low|s) = 10^-2
  → r_low = 100, 远超 [1-ε, 1+ε]，被aggressively clipped
  → 但对TV divergence的贡献只有 |10^-4 - 10^-2|/2 ≈ 0.005，微不足道

High-prob token: μ(a_high|s) = 0.99, π(a_high|s) = 0.80
  → r_high ≈ 0.808，落在 [1-ε, 1+ε] 内，不被clip
  → 但对TV divergence的贡献是 |0.99 - 0.80|/2 = 0.095，是low-prob case的20倍
```

这种structural bias在training-inference mismatch（Yao et al., 2025; Qi et al., 2025b）的背景下被进一步放大。Figure 2清楚地展示：对于low-probability tokens，training engine和inference engine之间的ratio是高度volatile的，但TV divergence却保持stable。这意味着PPO的clipping决策基于一个本来就noisy的signal，在低概率token上做出大量false positive的clip判断。

Reference: 
- PPO原始paper: https://arxiv.org/abs/1707.06347
- TRPO原始paper: https://arxiv.org/abs/1502.05477
- Training-inference mismatch分析: https://arxiv.org/abs/2510.26788

---

## 2. 理论框架：从Classical Trust Region到LLM Regime

### 2.1 Classical Policy Improvement Bound (Theorem 2.1)

Schulman et al. (2015)给出的经典bound：

$$\eta(\pi) - \eta(\mu) \geq \frac{1}{1-\gamma} \mathbb{E}_{s \sim \rho^\mu, a \sim \mu(\cdot|s)}\left[\frac{\pi(a|s)}{\mu(a|s)} A^\mu(s,a)\right] - \frac{2\xi\gamma}{(1-\gamma)^2} D_{TV}^{max}(\mu \| \pi)^2$$

变量解析：
- $\eta(\pi) = \mathbb{E}_{\tau \sim \pi}[\sum_{t=0}^{\infty} \gamma^t r_t]$ 是discounted return
- $\gamma \in [0,1]$ 是discount factor
- $\rho^\mu(s) = (1-\gamma) \sum_{t=0}^{\infty} \gamma^t \Pr(s_t = s | \pi)$ 是normalized discounted state-visitation distribution
- $A^\mu(s,a) = Q^\mu(s,a) - V^\mu(s)$ 是advantage function
- $\xi = \max_{s,a} |A^\mu(s,a)|$ 是最大绝对advantage
- $D_{TV}^{max}(\mu \| \pi) = \max_s D_{TV}(\mu(\cdot|s) \| \pi(\cdot|s))$ 是所有state上的最大TV divergence

这个bound的intuition是：policy improvement由两部分构成 - 一个first-order surrogate term（可以用samples估计）和一个penalty term（衡量policy偏离有多远）。只要保持 $D_{TV}^{max}$ 足够小，surrogate的improvement就能传递到true performance。

### 2.2 LLM Regime的挑战

LLM fine-tuning有两个crucial differences：
1. **Undiscounted ($\gamma = 1$)**：generation是finite-horizon episodic task，没有discount
2. **Sequence-level reward**：reward $R(y)$ 在整个sequence生成后才给出，不是per-token

如果直接用Theorem 2.1，$\frac{1}{1-\gamma}$ 项会diverge to infinity，bound变得ill-defined。这是paper需要重新推导的核心motivation。

### 2.3 Performance Difference Identity for LLMs (Theorem 3.1)

$$\mathcal{I}(\pi) - \mathcal{I}(\mu) = L_\mu'(\pi) - \Delta(\mu, \pi)$$

其中：
$$L_\mu'(\pi) = \mathbb{E}_{y \sim \mu}\left[R(y) \sum_{t=1}^{|y|}\left(\frac{\pi(y_t|s_t)}{\mu(y_t|s_t)} - 1\right)\right]$$

$$\Delta(\mu, \pi) = \mathbb{E}_{y \sim \mu}\left[R(y) \sum_{t=1}^{|y|}\left(\frac{\pi(y_t|s_t)}{\mu(y_t|s_t)} - 1\right)\left(1 - \prod_{j=t+1}^{T} \frac{\pi(y_j|s_j)}{\mu(y_j|s_j)}\right)\right]$$

变量解析：
- $\mathcal{I}(\pi) = \mathbb{E}_{y \sim \pi}[R(y)]$ 是LLM的expected reward
- $L_\mu'(\pi)$ 是surrogate objective，第一-order approximation
- $\Delta(\mu, \pi)$ 是error term，捕获higher-order effects
- $s_t = (x, y_1, ..., y_{t-1})$ 是state，包含prompt和已生成tokens
- $T = |y|$ 是sequence length

**Proof的核心技巧**是telescoping sum identity：
$$\pi(y|x) - \mu(y|x) = \sum_{t=1}^{T} \left(\prod_{k=1}^{t-1} \mu(y_k|s_k)\right) \left(\pi(y_t|s_t) - \mu(y_t|s_t)\right) \left(\prod_{j=t+1}^{T} \pi(y_j|s_j)\right)$$

这个identity把sequence-level probability difference分解成per-token contributions的sum。每一项是"用$\mu$生成前$t-1$步，第$t$步用$\pi$替代$\mu$，后续用$\pi$生成"这样的hybrid trajectory的probability。

**与classical surrogate的对比**（Appendix B.4）：虽然在形式上不同（LLM版本是trajectory-level expectation weighted by $R(y)$，classical版本是state-action pair expectation weighted by $A^\mu(s,a)$），但它们的gradient w.r.t. $\theta$ 是analogous的：

$$\nabla_\theta L_\mu'(\pi_\theta) = \mathbb{E}_{y \sim \mu}\left[\sum_{t=1}^{|y|} \frac{\pi_\theta(y_t|s_t)}{\mu(y_t|s_t)} \nabla_\theta \log \pi_\theta(y_t|s_t) A^\mu(s_t, y_t)\right]$$

其中定义 $A^\mu(s_t, y_t) = R(y) - V(x)$。这验证了LLM formulation是classical trust region的sound adaptation。

### 2.4 Policy Improvement Bound for LLMs (Theorem 3.2)

$$\mathcal{I}(\pi) - \mathcal{I}(\mu) \geq L_\mu'(\pi) - 2\xi T(T-1) \cdot D_{TV}^{max}(\mu \| \pi)^2$$

变量：
- $\xi = \max_y |R(y)|$ 是最大绝对reward（注意是reward不是advantage，因为LLM是sequence-level reward）
- $T$ 是sequence length，扮演类似 $\frac{1}{1-\gamma}$ 的角色
- $D_{TV}^{max}(\mu \| \pi) = \max_{s_t} D_{TV}(\mu(\cdot|s_t) \| \pi(\cdot|s_t))$

**Proof的关键步骤**：
1. 用Lemma B.1 bound sequence-level TV divergence：$D_{TV}(\mu_N(\cdot|s_1) \| \pi_N(\cdot|s_1)) \leq \sum_{t=1}^{N} \mathbb{E}_{s_t \sim \mu}[D_{TV}(\mu(\cdot|s_t) \| \pi(\cdot|s_t))]$
2. 用 $D_{TV} \leq (T-t) D_{TV}^{max}$ bound future trajectory divergence
3. 用 $\mathbb{E}_{y_t \sim \mu}[|r_t - 1|] = 2 D_{TV}(\mu(\cdot|s_t) \| \pi(\cdot|s_t))$ 转换
4. Sum over t得到 $T(T-1)$ 的quadratic dependence

### 2.5 Tighter Linear Bound (Appendix B.3)

Quadratic bound对长sequence太松。利用 $D_{TV} \leq 1$：

$$\mathcal{I}(\pi) - \mathcal{I}(\mu) \geq L_\mu'(\pi) - \min\left(2\xi T(T-1) \cdot D_{TV}^{max2}, 4\xi \cdot \mathbb{E}_{y \sim \mu}\left[\sum_{t=1}^{|y|} D_{TV}(\mu(\cdot|s_t) \| \pi(\cdot|s_t))\right]\right)$$

这个composite bound对小update用quadratic bound（更紧），对大update或长horizon用linear bound（避免爆炸）。这是一个非常实用的engineering insight。

Reference:
- TRPO: https://arxiv.org/abs/1502.05477
- CPO (Constrained Policy Optimization): https://arxiv.org/abs/1705.10528
- Kakade & Langford 2002: https://papers.nips.cc/paper/2001/hash/4e841ccba9398b1bdff9e3f6f0d1cc9c-Abstract.html

---

## 3. DPPO: 用Divergence替代Ratio

### 3.1 算法核心

DPPO的objective：
$$\mathcal{L}_\mu^{DPPO}(\pi) = \mathbb{E}_{y \sim \mu}\left[\sum_{t=1}^{|y|} M_t^{DPPO} \cdot r_t \cdot \hat{A}_t\right]$$

关键创新是mask的设计：
$$M_t^{DPPO} = \begin{cases} 0, & \text{if } (\hat{A}_t > 0 \text{ and } r_t > 1 \text{ and } D > \delta) \\ & \text{or } (\hat{A}_t < 0 \text{ and } r_t < 1 \text{ and } D > \delta) \\ 1, & \text{otherwise} \end{cases}$$

变量：
- $M_t^{DPPO} \in \{0, 1\}$ 是binary mask，决定是否保留第t个token的gradient
- $\hat{A}_t$ 是estimated advantage（通常用GRPO的group-relative estimation）
- $D \equiv D(\mu(\cdot|s_t) \| \pi(\cdot|s_t))$ 是policy distribution之间的divergence（TV或KL）
- $\delta$ 是divergence threshold hyperparameter

### 3.2 Mask设计的三个关键properties

**Property 1: Asymmetric structure保留PPO的优点**
- 当 $\hat{A}_t > 0$ 且 $r_t > 1$：update在增加token概率，但同时push policy远离trust region → 可能block
- 当 $\hat{A}_t > 0$ 且 $r_t < 1$：update在增加token概率，同时push policy **靠近** trust region → 永不block（因为 $r_t$ 趋向1）
- 当 $\hat{A}_t < 0$ 且 $r_t < 1$：update在降低token概率，同时push policy远离trust region → 可能block
- 当 $\hat{A}_t < 0$ 且 $r_t > 1$：update在降低token概率，同时push policy靠近trust region → 永不block

这个asymmetric design非常重要：它允许policy快速recover from bad initializations（比如一个token被over-penalized后，可以快速recover），同时防止runaway updates。

**Property 2: Block决策基于distribution-level divergence**
PPO用single-sample ratio $r_t$ 做决策，DPPO用整个distribution的divergence $D$。这是本质区别 - DPPO问的是"整个policy distribution偏离多远"，PPO问的是"这一个token的ratio偏离多远"。

**Property 3: Trust region anchored to behavior policy $\mu_{\theta'}$**
Paper在Section 5.2强烈建议anchor到rollout时的behavior policy，而不是recomputed on-policy distribution $\pi_{\theta'}$。这看似反直觉（recomputed distribution不是更"准确"吗？），但实验证明anchor到 $\pi_{\theta'}$ 会导致training collapse。原因是trust region的目的是限制 $\pi_\theta$ 相对于data generation distribution的偏离，而data是从 $\mu_{\theta'}$ 生成的。

### 3.3 PPO vs DPPO Mask对比

| 特性 | PPO | DPPO |
|------|-----|------|
| Block signal | Single-sample ratio $r_t$ | Distribution divergence $D$ |
| Block condition | $|r_t - 1| > \epsilon$ | $D > \delta$ 且 update远离trust region |
| Low-prob token | 过度penalize（ratio易爆炸） | 正确评估（divergence贡献小） |
| High-prob token | 不足penalize（ratio变化小） | 正确评估（divergence贡献大） |
| Asymmetry | 通过 $\epsilon_{high} \neq \epsilon_{low}$ 实现 | 通过 $r_t > 1$ / $r_t < 1$ 条件实现 |
| Anchor | 隐式anchor到sampled token | 显式anchor到整个distribution |

---

## 4. Binary和Top-K近似：让Divergence计算可行

直接计算 $D_{TV}(\mu(\cdot|s_t) \| \pi(\cdot|s_t))$ 需要对整个vocabulary（15万+ tokens）求和，memory开销巨大。Paper提出两种approximation，都是true divergence的principled lower bound。

### 4.1 Binary Approximation

把categorical distribution压缩成Bernoulli distribution：sampled token $a_t$ vs 其他所有tokens。

$$D_{TV}^{Bin}(t) = |\mu(a_t|s_t) - \pi(a_t|s_t)|$$

$$D_{KL}^{Bin}(t) = \mu(a_t|s_t) \log \frac{\mu(a_t|s_t)}{\pi(a_t|s_t)} + (1 - \mu(a_t|s_t)) \log \frac{1 - \mu(a_t|s_t)}{1 - \pi(a_t|s_t)}$$

变量：
- $a_t$ 是sampled token
- $\mu(a_t|s_t)$ 和 $\pi(a_t|s_t)$ 是两个policy对该token的概率
- $(1 - \mu(a_t|s_t))$ 和 $(1 - \pi(a_t|s_t))$ 是"其他所有tokens"的aggregated probability

**Intuition**：Binary approximation问的是"sampled token的概率变化了多少"，而不是"整个distribution变化了多少"。这看似coarse，但它正确捕获了sampled token的绝对probability mass shift - 正是PPO ratio clipping失败的核心场景。

**为什么Binary TV是true TV的lower bound**：
对vocabulary做partition $\mathcal{C} = \{C_1, ..., C_m\}$，Binary partition是 $C_1 = \{a_t\}$, $C_2 = \mathcal{A} \setminus \{a_t\}$。由triangle inequality：
$$\left|\sum_{a \in C_j} (\mu(a|s_t) - \pi(a|s_t))\right| \leq \sum_{a \in C_j} |\mu(a|s_t) - \pi(a|s_t)|$$

Sum over partitions得到 $D_{TV}^{\mathcal{C}} \leq D_{TV}$。

### 4.2 Top-K Approximation

显式跟踪最probable的K个tokens + sampled token：

$$\mathcal{A}_t' = \text{TopK}(\mu(\cdot|s_t), K) \cup \{a_t\}$$
$$\mathcal{A}_t'' = \mathcal{A}_t' \cup \{\text{other}\}$$

$$D_{TV}^{TopK}(t) = \frac{1}{2} \sum_{a \in \mathcal{A}_t''} |p_t^\mu(a) - p_t^\pi(a)|$$

$$D_{KL}^{TopK}(t) = \sum_{a \in \mathcal{A}_t''} p_t^\mu(a) \log \frac{p_t^\mu(a)}{p_t^\pi(a)}$$

其中：
- $p_t^\mu(a) = \mu(a|s_t)$ for $a \in \mathcal{A}_t'$
- $p_t^\mu(\text{other}) = 1 - \sum_{a \in \mathcal{A}_t'} \mu(a|s_t)$

**Intuition**：Top-K better captures distribution head的shift，而distribution head通常dominates true divergence value。Appendix C证明Top-K的approximation gap bounded by tail probability mass $\mu(C_{\text{other}}|s_t)$，而tail通常很小。

### 4.3 实验对比：Binary vs Top-K

Figure 10和Figure 16的ablation显示：Binary和Top-K (K=20) 性能几乎相同，都显著优于baseline。这表明简单的Binary approximation已经足够 - 这是一个非常实用的发现，因为Binary的开销几乎为零。

**为什么Binary够用？** 我的intuition是：在LLM RL中，最critical的updates往往集中在少数high-information tokens上（numbers, reasoning words like "Wait", "Thus"）。Binary approximation精确捕获了这些sampled tokens的probability shift，而distribution tail的collective shift虽然理论上是信息，但对training dynamics影响较小。

Reference:
- DPPO GitHub: https://github.com/sail-sg/Stable-RL
- Log-sum inequality proof: standard information theory textbook

---

## 5. 训练稳定性的三大发现

### 5.1 Takeaway 1: Trust Region在Low Learning Rate下仍然必要

实验设置：DeepSeek-R1-Distill-Qwen-1.5B在MATH 1460 problems上fine-tune，learning rate $10^{-6}$，BFloat16 precision（故意暴露numerical instability）。

结果（Figure 3）：
- **PG-IS**（no trust region）：training-inference mismatch持续增长，最终collapse
- **PG-TIS / CISPO**（truncated importance sampling）：同样collapse，甚至更早
- **GRPO with Clip-Higher**：相对稳定但improvement慢
- **MiniRL / MiniRL-TIS**（anchor to recomputed $\pi_{\theta'}$）：collapse
- **DPPO variants**：stable，low mismatch，near-perfect final rewards

**Intuition**：即使learning rate极小（$10^{-6}$），如果没有trust region，training-inference mismatch会compounding accumulate。每一次gradient step都基于一个稍微biased的gradient估计，这个bias在long-horizon generation中会被放大（因为sequence probability是per-token probability的product）。Trust region的作用是确保每一步update不会让 $\pi_\theta$ 偏离 $\mu_{\theta'}$ 太远，从而控制bias accumulation。

### 5.2 Takeaway 2: Trust Region必须Anchor到Behavior Policy $\mu_{\theta'}$

实验（Figure 4）：把稳定的DPPO-KL改成decoupled version（anchor到 $\pi_{\theta'}$），single change导致mismatch增长和performance collapse。

**为什么anchor到 $\pi_{\theta'}$ 有害？** 
- Trust region的目的是限制 $\pi_\theta$ 相对于data generation distribution的偏离
- Data是从 $\mu_{\theta'}$ 生成的，importance sampling weights $\frac{\pi_\theta}{\mu_{\theta'}}$ 是unbiased的
- 如果anchor到 $\pi_{\theta'}$，实际上是在比较 $\pi_\theta$ 和一个recomputed distribution，这个recomputed distribution本身就有numerical error
- 更fundamentally，$\pi_{\theta'}$ 不是data的真正来源，用它做anchor会导致importance sampling的bias correction失效

**Practical benefit**：不需要recompute $\pi_{\theta'}$，节省约25% training cost（Qi et al., 2025b）。

### 5.3 Takeaway 3: 不稳定的主要来源是Negative Samples上的"Bad Updates"

实验方法（Figure 5）：从unstable的PG-IS开始，逐步添加minimal mask来restore stability。发现只需block negative samples上 $\mu_{\theta'}(y_t|s_t) - \pi_\theta(y_t|s_t) \geq \delta$ 的updates（$\delta = 0.5$）就足以stabilize training。

**关键发现**：
- "Bad updates"占比极小（≤0.5%），但是是training collapse的primary culprit
- Bad updates percentage与reward fluctuation强correlation
- Slightly looser mask（$\delta = 0.8$）或anchor to recomputed distribution都fail

**Intuition**：为什么negative samples上的large-divergence updates如此destructive？
当一个token被model认为probable（$\mu_{\theta'}(y_t|s_t)$ 大），但出现在negative-rewarded sample中时，naive policy gradient会push这个token的probability大幅下降。这可能corrupt LLM的internal knowledge - 比如model本来知道"Thus"是reasoning的重要token，但某个bad reasoning sample里有"Thus"，gradient就大幅降低"Thus"的概率。这种corruption会propagate到后续generation，导致更多bad samples，形成positive feedback loop。

### 5.4 TIS (Truncated Importance Sampling)的反效果

Figure 3显示PG-TIS和MiniRL-TIS比对应的non-truncated版本更早collapse。这是一个counterintuitive的发现。

**Hypothesis**：TIS truncate high-ratio samples来reduce variance，但high-ratio samples往往是low-probability tokens（前面分析过ratio易爆炸）。Truncating这些tokens systematically down-weights来自low-probability tokens的gradient signal。而low-probability tokens正是exploration的关键（Wang et al., 2025a发现high-entropy minority tokens驱动RL learning）。所以TIS虽然reduce variance，但introduces harmful bias。

Reference:
- Defeating training-inference mismatch via FP16: https://arxiv.org/abs/2510.26788
- CISPO / MiniMax-M1: https://arxiv.org/abs/2506.13585
- Stabilizing RL with LLMs: https://arxiv.org/abs/2512.01374
- Entropy mechanism in RL for reasoning: https://arxiv.org/abs/2505.22617
- High-entropy minority tokens: https://arxiv.org/abs/2506.01939

---

## 6. 训练效率分析：Low-Probability Tokens的重要性

### 6.1 Relaxing Trust Region for Low-Probability Tokens

实验（Figure 6）：在GRPO baseline上，对 $\mu(y_t|s_t) < \alpha$ 的tokens设 $\epsilon = \infty$（完全放松clipping）。

结果：
- $\alpha = 0$（baseline）：standard GRPO
- $\alpha = 0.1$：显著提升training efficiency
- Clipped tokens主要是low-probability tokens（typically below 0.15）
- Clipped tokens往往有high entropy

**Intuition**：PPO的ratio clipping structural bias against low-probability tokens。这些tokens虽然individual probability小，但collectively是exploration的关键。Relaxing their constraint enables more informative policy updates。

### 6.2 Directional Relaxation Matters

Figure 7的ablation（fixed $\alpha = 0.1$）：
- **Relax-high**（放松 $r_t > 1 + \epsilon_{high}$ 的约束）：maintain high entropy但no significant gain
- **Relax-low**（放松 $r_t < 1 - \epsilon_{low}$ 的约束）：faster initial learning但最终entropy collapse
- **Relax-both**：最优，both efficient and stable

**Intuition**：为什么Relax-low会导致entropy collapse？
当放松对 $r_t < 1$ 的约束，model可以更aggressively降低token概率。这加速initial learning（快速suppress bad tokens），但也会导致某些tokens的概率drop to near-zero，减少exploration，最终entropy collapse。Relax-both平衡了exploration和exploitation。

### 6.3 DPPO的Implicit Relaxation

DPPO通过divergence-based mask naturally实现了Relax-both的效果：
- Low-prob token的ratio大但divergence贡献小 → DPPO不block（相当于Relax-both）
- High-prob token的ratio小但divergence贡献大 → DPPO会block（正确约束）

这是DPPO比PPO更efficient的algorithmic reason。

### 6.4 Clipped Tokens的Characterization (Appendix E)

Paper分析了GRPO训练中最常被clip的tokens，发现它们主要是：
1. **Numerical and mathematical tokens**: "1", "4", "+", "-", "div"
2. **Reasoning and structural words**: "Wait", "Next", "Thus", "Since", "Let", "We", "I"

这些tokens对reasoning task至关重要。PPO系统性interference这些tokens的learning signal，slows learning, stifles exploration。

**Intuition**：reasoning tokens如"Wait"往往initial probability低（model还不确定何时use），但在positive samples中出现会获得high advantage，导致high ratio和clipping。这正是DPPO要解决的failure mode。

Reference:
- DAPO: https://arxiv.org/abs/2503.14476
- Clip-Higher trick (DAPO paper)
- OctoThinker: https://arxiv.org/abs/2506.20512

---

## 7. 大规模实验结果

### 7.1 实验配置

| Config | Base Model | Response Length | Train Batch | LR | Rollout N |
|--------|-----------|-----------------|-------------|-----|-----------|
| MoE Base | Qwen3-30B-A3B-Base | 16384 | 256 | 1e-6 | 16 |
| MoE Base w/ R3 | Qwen3-30B-A3B-Base | 16384 | 256 | 1e-6 | 16 |
| MoE Thinking | Qwen3-30B-A3B | 16384 | 256 | 1e-6 | 16 |
| Dense Base | Qwen3-8B-Base | 8000 | 128 | 1e-6 | 8 |
| MoE Base w/ LoRA | Qwen3-30B-A3B-Base | 8000 | 128 | 1e-5 | 8 |

Dataset: filtered DAPO-Math (~13k samples)
Evaluation: AIME24, AIME25, Avg@32 (temperature=0.7, top-p=0.95)

### 7.2 Main Results (Figures 8-9, 11-15)

**Consistent findings across all 5 configs**：
1. **DPPO optimizes rewards faster** than GRPO-ClipHigher baseline
2. **DPPO achieves better converged performance**
3. **DPPO maintains remarkably stable training** even when baselines collapse
4. **DPPO without R3 outperforms R3-enhanced baselines**（Figure 8）- 这是strong statement
5. **DPPO + R3 yields additional gains**（orthogonal benefits）

**Notable baseline failures**：
- CISPO in MoE Base w/o R3: sudden severe collapse, complete failure
- GRPO-ClipHigher in MoE Thinking: significant training collapse
- CISPO in Dense Base: degenerative trend on AIME25
- GRPO-ClipHigher: consistently excessive entropy in all experiments

### 7.3 Training Dynamics观察

从Figures 11-15可以观察到：
- **Training-inference mismatch** ($|\pi - \mu|$): DPPO保持low且stable，baselines会增长
- **Policy entropy**: DPPO moderate，GRPO-ClipHigher异常high，CISPO fluctuating
- **Response length**: DPPO effectively increases length（除MoE Thinking，因为already long）

**关于GRPO-ClipHigher的excessive entropy**：这是一个puzzling现象。我的intuition是：Clip-Higher trick（增大 $\epsilon_{high}$）允许low-probability tokens的probability大幅增加，这会增加entropy。但同时ratio clipping仍然over-penalize这些tokens，导致不稳定的"push-and-pull" dynamics，manifested as excessive entropy fluctuation。

### 7.4 Generalization (Appendix G.3)

DPPO在以下setting也优于baseline：
1. **Different model family**: OctoThinker-3B-Hybrid-Base on MATH
2. **Abstract reasoning**: Arc1D, Acre tasks
3. **Multi-turn reasoning**: Sudoku-v0-easy

这验证了DPPO的broad applicability。

Reference:
- Qwen3 technical report: https://arxiv.org/abs/2505.09388
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- DeepSeekMath (GRPO): https://arxiv.org/abs/2402.03300
- VeRL framework: https://arxiv.org/abs/2409.19256
- LoRA without regret: https://thinkingmachines.ai/blog/lora/

---

## 8. 更深的Intuition和Open Questions

### 8.1 为什么Single-Sample Ratio Estimate在LLM下特别糟糕？

Classical RL的action space小，每个action被sample到的概率相对均匀，ratio的variance可控。LLM的vocabulary是高度long-tailed的，probability跨越5-6个orders of magnitude。这意味着：

1. **Variance of ratio estimate**：$\text{Var}[r_t] = \text{Var}\left[\frac{\pi(y_t)}{\mu(y_t)}\right]$ 在long-tailed distribution下极大
2. **Bias from clipping**：当ratio distribution heavy-tailed，clipping会systematically bias estimate
3. **Compounding through sequence**：sequence probability是per-token probability的product，error会compound

**Fundamental insight**：PPO的clipping本质上是把一个high-variance的estimator硬truncate到一个固定range，这在classical RL中work（因为estimator本身variance小），但在LLM中fail（因为estimator variance巨大，truncation引入严重bias）。

### 8.2 为什么Binary Approximation够用？

理论上Top-K应该更准确，但实验显示Binary够用。可能的解释：

1. **Sampled token dominates information**：在policy gradient中，只有sampled token的gradient被计算。Binary approximation精确捕获了这个token的probability shift，这正是gradient calculation关心的。
2. **Distribution tail的shift影响小**：虽然tail tokens数量多，但individual probability小，collective对divergence的贡献可能limited。
3. **Trust region的目的是prevent catastrophic shifts**：catastrophic shifts通常发生在high-probability tokens上（massive probability mass movement），Binary能捕获这种shift。

**Open question**：是否存在Binary approximation失败的scenario？比如distribution tail发生coherent shift（很多tail tokens同时增加概率），Binary会miss。这种scenario在实际中是否常见？

### 8.3 与Information Theory的连接

$D_{TV}(\mu \| \pi) \leq \sqrt{\frac{1}{2} D_{KL}(\mu \| \pi)}$ (Pinsker's inequality)

这意味着用KL divergence作为trust region constraint是more conservative的。Paper的实验显示KL和TV variants性能相近，但threshold $\delta$ 不同（KL用0.05，TV用0.15-0.2），这与Pinsker's inequality的order of magnitude一致。

**Deeper connection**：$D_{TV}$ 和 $D_{KL}$ 在information theory中有不同operational meanings：
- $D_{TV}$: hypothesis testing的error probability
- $D_{KL}$: coding的redundancy

在RL context中，$D_{TV}$ 更directly related to policy improvement bound（Theorem 3.2的penalty term是 $D_{TV}^2$），所以理论上TV更principled。但KL更smooth，optimization可能更easy。

### 8.4 与其他Trust Region方法的对比

| Method | Trust Region Signal | Anchor | Asymmetry | Over-constrain Low-prob | Under-constrain High-prob |
|--------|--------------------|---------|-----------:|------------------------|---------------------------|
| PPO | Single-sample ratio | Sampled token | via $\epsilon_{low} \neq \epsilon_{high}$ | Yes | Yes |
| TRPO | KL divergence (full) | Full distribution | No (hard constraint) | No | No |
| Clip-Higher | Single-sample ratio | Sampled token | Larger $\epsilon_{high}$ | Partially mitigated | Still yes |
| CISPO | Truncated ratio | Sampled token | via truncation | Yes (truncation) | Yes |
| MiniRL | Recomputed ratio | Recomputed $\pi_{\theta'}$ | via $\epsilon$ | Yes | Yes + wrong anchor |
| **DPPO** | Distribution divergence | Behavior policy $\mu_{\theta'}$ | via $r_t > 1$ / $r_t < 1$ | No | No |

DPPO是唯一同时解决over-constrain和under-constrain问题的方法，且正确anchor到behavior policy。

### 8.5 与Recent Work的Connection

1. **Wang et al. (2019, 2020)**: 提出adaptive clipping based on KL divergence for classical RL。DPPO是类似idea在LLM regime的realization，但需要Binary approximation使其feasible。
2. **Clip-Higher (Yu et al., 2025)**: 认识到low-probability tokens被over-penalized，但solution是heuristic（增大 $\epsilon_{high}$）。DPPO从root cause解决。
3. **CISPO (Chen et al., 2025)**: 保留所有tokens的gradient，完全ignore trust region。实验显示这在某些setting下不稳定。
4. **TIS (Yao et al., 2025)**: Truncate high-ratio samples reduce variance，但introduce harmful bias against low-probability tokens。
5. **R3 (Ma et al., 2025)**: Engineering solution for MoE training-inference mismatch，与DPPO orthogonal且complementary。

### 8.6 Engineering Considerations

从implementation角度：
1. **Memory overhead**: Binary approximation几乎为零（只需 $\mu(a_t|s_t)$ 和 $\pi(a_t|s_t)$ 两个scalar），Top-K需要存储top-K tokens的log probs
2. **Compute overhead**: Binary的divergence计算是O(1)，Top-K是O(K)
3. **vLLM compatibility**: Top-K受限于vLLM的max 20 candidate tokens限制
4. **Framework integration**: DPPO只改mask logic，与VeRL, OAT等框架兼容

### 8.7 Limitations and Future Directions

Paper没有extensively讨论的points：
1. **Threshold $\delta$ 的sensitivity**: Paper用 $\delta = 0.15$ (TV) 和 $\delta = 0.05$ (KL)，但没有systematic sweep。如何自动选择 $\delta$？
2. **Continuous action space**: Paper focus on discrete token generation，continuous action space（如robotics LLM）如何adapt？
3. **Multi-turn RL**: Appendix G.3的Sudoku实验是preliminary，multi-turn setting的trust region如何定义？
4. **Offline RL**: DPPO假设online data collection，offline setting下 $\mu$ 是fixed dataset distribution，如何adapt？
5. **Theoretical tightness**: Theorem 3.2的bound是否tight？能否用sharper concentration inequalities改进？

### 8.8 Personal Reflections on the Paper's Significance

这篇paper的beauty在于它从一个seemingly technical detail（ratio clipping vs divergence）出发，揭示了PPO在LLM regime下的structural flaw。核心insight可以一句话总结：**PPO用single-sample estimate约束一个expectation，在long-tailed distribution下这个estimate的variance太大，导致systematic bias**。

这种"从heuristic回到first principles"的approach在RL领域有悠久传统：
- TRPO: 从policy improvement theorem推导trust region
- PPO: 简化TRPO为first-order method
- DPPO: 修正PPO在LLM regime的approximation failure

DPPO的practical impact可能很大：
1. 替代GRPO作为default LLM RL algorithm
2. 与R3, LoRA, MoE optimization等技术orthogonal
3. Theoretical framework可以guide future trust region research

**可能的extensions**：
1. **Adaptive $\delta$**: 根据training progress自动调整 $\delta$（早期大，后期小）
2. **Multi-step divergence**: 考虑multi-step trajectory divergence，而非greedy per-step
3. **Learned divergence**: 用一个小network学习divergence estimate
4. **Hierarchical trust region**: 不同层（attention, MLP, embedding）有不同的trust region

### 8.9 与其它领域概念的类比

**Robust statistics**: PPO的ratio clipping类似于用hard threshold做outlier rejection，而DPPO的divergence-based masking类似于用distributional distance做robust estimation。前者对individual sample敏感，后者基于distributional properties更robust。

**Optimization**: PPO的clipping类似于trust region method with approximate constraint（用gradient norm约束），DPPO类似于exact constraint evaluation。前者便宜但inaccurate，后者准确但expensive（DPPO用Binary approximation平衡了两者）。

**Information bottleneck**: DPPO的mask可以看作information bottleneck - 只保留那些"informative且safe"的updates。High-probability tokens的large divergence updates被block（unsafe），low-probability tokens的high-ratio updates被allow（safe but informative）。

### 8.10 实际Deployment建议

基于paper的findings，实际deploy DPPO的建议：

1. **Default config**: DPPO-Binary-TV with $\delta = 0.2$ for most settings
2. **Precision-sensitive setting**: DPPO-Binary-KL with $\delta = 0.05$（KL更sensitive to small changes）
3. **LoRA fine-tuning**: Larger $\delta$（0.15-0.2）因为LoRA updatesmore localized
4. **MoE models**: DPPO without R3 often sufficient，R3 can be added for extra stability
5. **Long sequence generation**: Linear bound (Appendix B.3)更relevant，可能需要smaller $\delta$
6. **Exploration-heavy tasks**: DPPO naturally encourages exploration via low-prob token relaxation

### 8.11 公式汇总表

| Component | Formula | Variables |
|-----------|---------|-----------|
| PPO objective | $L_\mu^{PPO} = \mathbb{E}[\sum_t \min(r_t \hat{A}_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t)]$ | $r_t = \pi/\mu$ |
| DPPO objective | $L_\mu^{DPPO} = \mathbb{E}[\sum_t M_t^{DPPO} r_t \hat{A}_t]$ | $M_t \in \{0,1\}$ |
| TV divergence | $D_{TV} = \frac{1}{2}\sum_a |\mu(a) - \pi(a)|$ | range $[0,1]$ |
| KL divergence | $D_{KL} = \sum_a \mu(a) \log \frac{\mu(a)}{\pi(a)}$ | range $[0, \infty)$ |
| Binary TV | $D_{TV}^{Bin} = |\mu(a_t) - \pi(a_t)|$ | single token |
| Binary KL | $D_{KL}^{Bin} = \mu \log \frac{\mu}{\pi} + (1-\mu)\log\frac{1-\mu}{1-\pi}$ | Bernoulli |
| Top-K TV | $D_{TV}^{TopK} = \frac{1}{2}\sum_{a \in \mathcal{A}''} |p^\mu(a) - p^\pi(a)|$ | reduced vocab |
| LLM improvement bound | $\mathcal{I}(\pi) - \mathcal{I}(\mu) \geq L_\mu'(\pi) - 2\xi T(T-1) D_{TV}^{max2}$ | $\xi = \max |R|$, $T$ = horizon |
| Tighter bound | $\geq L_\mu' - \min(2\xi T(T-1) D_{TV}^{max2}, 4\xi \mathbb{E}[\sum_t D_{TV}])$ | composite |

---

## 9. 总结

这篇paper的核心贡献：

1. **Theoretical**: 推导了LLM-specific的policy improvement bound（finite-horizon, undiscounted），建立了trust region methods在LLM regime的rigorous foundation。

2. **Diagnostic**: 精确识别了PPO在LLM下的structural flaw - ratio clipping是TV divergence的noisy single-sample estimate，在long-tailed vocabulary下导致over-constrain low-prob tokens和under-constrain high-prob tokens。

3. **Algorithmic**: 提出DPPO，用distribution divergence替代single-sample ratio做trust region constraint。Binary approximation使其practically free。

4. **Empirical**: 在5个大规模实验配置中consistently outperform GRPO和CISPO，demonstrating superior stability和efficiency。

5. **Practical guidelines**: 
   - Trust region必须，即使learning rate极小
   - Anchor to behavior policy $\mu_{\theta'}$，不是recomputed $\pi_{\theta'}$
   - Primary instability来自negative samples上的bad updates
   - TIS有害
   - Relaxing low-prob token constraint提升efficiency

对field的影响：DPPO可能成为post-training LLM RL的新default algorithm，它的theoretical framework也可以guide future research on trust region methods in sequence generation settings。

Reference:
- DPPO paper (main): https://github.com/sail-sg/Stable-RL
- Stable-RL code: https://github.com/sail-sg/Stable-RL
- OAT framework: https://github.com/sail-sg/oat
- VeRL: https://github.com/volcengine/verl
- DAPO: https://arxiv.org/abs/2503.14476
- Qwen3: https://arxiv.org/abs/2505.09388
- AIME: https://maa.org/

这篇paper是RL theory与LLM practice的beautiful bridge，值得仔细研读。希望这个deep dive能帮助你build intuition about why trust region matters in LLM RL，以及如何从first principles设计更好的algorithms。
