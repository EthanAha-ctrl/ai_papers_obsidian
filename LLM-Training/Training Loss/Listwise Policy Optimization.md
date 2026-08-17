---
source_pdf: Listwise Policy Optimization.pdf
paper_sha256: a7ab4326e1790a42c6634f477680d45bb4effed2836ef2915d9279a36ad1b946
processed_at: '2026-08-05T15:06:24-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# LPO 人话版

## 一句话总结

**所有 group-based RLVR 方法（GRPO、Dr.GRPO、MaxRL）本质上都在干同一件事：偷偷瞄一眼 K 个 response 的 reward，构造一个 softmax target，然后让 policy 往这个 target 靠。但它们靠的方式很糙——只用 first-order 近似，一旦 off-policy 就开始崩。LPO 说：既然 K 个 response 天然形成一个 finite simplex，为什么不直接做 exact projection？**

---

## 核心比喻

想象你在玩飞镖。靶子上有 K 个区域，每个区域有个分数（reward）。你的策略就是往这 K 个区域上分配"投掷概率"。

**GRPO 的做法**：
1. 看一眼这 K 个区域的分数
2. 心里默默算出一个"理想投掷分布"——高分区域多投，低分区域少投
3. 然后用一只手蒙着眼睛，凭着感觉往那个方向挪

**LPO 的做法**：
1. 看一眼分数
2. **明确算出**理想分布（就是 Gibbs/softmax target）
3. **睁着眼睛**，用尺子量着，精确往那个分布挪

区别在于：GRPO 是"凭感觉挪"（first-order 近似），LPO 是"量着挪"（exact projection）。在 on-policy 点两者重合，一旦 policy 开始 drift，GRPO 的手感就越来越差，LPO 仍然精确。

---

## 为什么 K 个 response 形成 simplex

这是全文最关键的 observation，但 paper 里讲得比较 formal。

LLM 生成一个 response $y_k$，policy 给它一个概率 $\pi_\theta(y_k|x)$。K 个 response 的概率加起来不需要等于 1（因为还有无限多没采样的 response）。但如果定义一个 **relative preference**：

$$P_{\theta,k} = \text{softmax}\left(\log \frac{\pi_\theta(y_k|x)}{\pi_b(y_k|x)}\right)_k$$

这个 $P_\theta$ 就在 simplex 上了——K 个数，非负，和为 1。

**为什么这很重要**：classical RL（robotics）里 action space 是连续的，partition function 不可计算，你永远没法做 exact projection。但 LLM 的 RLVR 里，每个 prompt 采 K 个 response，这 K 个 response 就是一个 finite set，partition function 就是 $\sum_{k=1}^K$，完全可以算。**这是 LLM RLVR 独有的结构红利**。

---

## Advantage normalization 到底在干什么

这是 paper 最有启发性的部分。所有 group-based 方法都在做 advantage normalization：

- GRPO: $A_k = (R_k - \mu_G) / \sigma_G$
- Dr.GRPO: $A_k = R_k - \mu_G$
- MaxRL: $A_k = (R_k - \mu_G) / \mu_G$

表面上这些是"降低方差"的 trick。但 paper 揭示：**它们其实都在构造同一个 softmax target，只是 temperature 不同**。

因为 softmax 是 shift-invariant 的，centering（减 $\mu_G$）在 softmax 里被 cancel 掉了。剩下的 scaling（除以 $\sigma_G$ 或 $\mu_G$）就是 temperature：

$$w^* = \text{softmax}(R / \tau)$$

| 方法 | $\tau$ | target 行为 |
|------|--------|------------|
| Dr.GRPO | 1 | 中等 sharpness |
| GRPO | $\sigma_G$ | 自适应：reward 方差大时 target 平缓，方差小时尖锐 |
| MaxRL | $\mu_G$ | 课程学习：hard prompt（$\mu_G$ 小）target 尖锐，easy prompt 平缓 |

**人话**：所有 advantage normalization scheme 都是在调 target 的"软硬程度"。$\tau$ 小 = target 几乎 one-hot（greedy），$\tau$ 大 = target 接近 uniform（保守）。不同方法只是用了不同的 heuristic 来设这个 $\tau$。

---

## Proposition 1：PG = reverse KL 的一阶近似

这是 paper 的 theoretical core。

**Claim**：在 on-policy 点（$\pi_\theta = \pi_b$），standard policy gradient：

$$g_{\text{PG}} = \frac{1}{K}\sum_k A_k \nabla_\theta \log \pi_\theta(y_k|x)$$

恰好等于 minimize reverse KL divergence $D_{\text{KL}}(P_\theta \| w^*)$ 的 gradient。

**为什么这是真的**：reverse KL 的 gradient coefficient（经过 logit-gap 简化）是：

$$c_k^{\text{rev}} = P_{\theta,k}(d_k - \bar{d})$$

在 on-policy 点，$P_{\theta,k} = 1/K$，$d_k = -A_k$，$\bar{d} = 0$（因为 advantage zero-mean）。所以：

$$c_k^{\text{rev}} = \frac{1}{K}(-A_k) = -\frac{A_k}{K}$$

PG 的 coefficient 是 $+A_k/K$。差一个负号——因为 PG 是 maximize，KL 是 minimize。**完全等价**。

**Off-policy 会怎样**：一旦 $\pi_\theta \neq \pi_b$，approximation error 线性增长：

$$\|g_{\text{PG}} - g_{\text{rev KL}}\| \propto \bar{\delta} \cdot (1 + \|A\|_\infty)$$

其中 $\bar{\delta}$ 是 importance ratio 偏离 1 的程度。

**人话**：PG 方法在 on-policy 点是精确的 reverse KL projection，但一旦 policy 开始 update（mini-batch epochs、clipping 等），就开始偏离。PPO 的 clipping 是一种 ad-hoc 的 band-aid——限制 drift 不让 approximation 崩得太厉害，但本质上还是在用近似。LPO 直接做 exact projection，不需要 clipping，不需要担心 drift。

---

## LPO 的两步 decoupling

LPO 把每次 iteration 拆成两步：

### Step 1：Target 构造

$$w_k^* = \text{softmax}\left(\frac{R_k}{\tau} + s_{t,k}\right)$$

其中 $s_{t,k} = \log(\pi_t(y_k|x) / \pi_b(y_k|x))$ 是 pre-update policy 的 logit。

**人话**：target = "reward 告诉我哪个好" + "当前 policy 觉得哪个可能"。$\tau$ 控制 reward 的权重——$\tau$ 小时 reward 主导（aggressive update），$\tau$ 大时 policy 主导（conservative update）。

在 on-policy setup（$\pi_t = \pi_b$），$s_{t,k} = 0$，target 退化成 $\text{softmax}(R/\tau)$，recover existing methods。

### Step 2：Projection

有了 target $w^*$，把 policy 往 target 上 project。project 的方式用 divergence minimization：

$$\theta' = \arg\min_\theta D(w^* \| P_\theta)$$

paper 实现了两个版本：

- **Forward KL**：$D_{\text{KL}}(w^* \| P_\theta)$
- **Reverse KL**：$D_{\text{KL}}(P_\theta \| w^*)$

---

## Forward KL vs Reverse KL：为什么 Forward 更好

这是实验最重要的发现：**LPO_fwd 在 Pass@k 上系统性碾压 LPO_rev（13/15 scenarios）**。

### Forward KL 的 gradient

$$c_k^{\text{fwd}} = P_{\theta,k} - w_k^*$$

**极其简洁**：current policy 给 response $k$ 的概率减去 target 想要的概率。如果 policy 给多了就往下压，给少了就往上提。

三个结构性质：
1. **Bounded**：$|c_k| \leq 1$，$\sum|c_k| \leq 2$。Gradient 永远不会爆炸。
2. **Zero-sum**：$\sum c_k = 0$。相当于 built-in baseline，自动做 mean-centering，降低 variance。
3. **Self-correcting**：$P_\theta \to w^*$ 时 $c_k \to 0$。Policy 完全 match target 时 gradient 自动消失。

### Reverse KL 的 gradient

$$c_k^{\text{rev}} = P_{\theta,k}(d_k - \bar{d})$$

其中 $d_k$ 是 logit gap。这个形式更复杂，但也是 zero-sum + self-correcting。**但不是 bounded 的**——$P_{\theta,k}$ 可以任意小，$d_k$ 可以任意大，product 没有上界。

### Mode behavior 的根本差异

**Forward KL 是 mass-covering（mode-covering）**：它要求 $P_\theta$ 在 $w^*$ 有 mass 的地方都有 mass。Corollary 2 给出 log-barrier：

$$P_{\theta,k} > \alpha \exp(-D/\alpha - 1)$$

只要 target 给 response $k$ 至少 $\alpha$ 的概率，current policy 在 KL ≤ D 约束下不会让 $P_{\theta,k}$ 跌破这个 lower bound。**Structurally 防止 mode collapse**。

**Reverse KL 是 mode-seeking**：它允许 $P_\theta$ 在 $w^*$ 低概率的地方有 mass，但在 $w^*$ 高概率的地方必须 match。倾向于 concentrate 在一个 mode 上。

### 实验含义

Reasoning task 通常有 multiple valid solution paths（同一道数学题有多种解法）。Forward KL 的 mode-covering 让 policy explore 所有这些 paths，所以 Pass@k（k 个 sample 里至少一个对的概率）高。Reverse KL 倾向于 concentrate 在最 likely 的一个 path 上，Pass@1 可能不差但 Pass@k 退化。

**这也是为什么 standard PG（≈ reverse KL 近似）容易 entropy collapse**——structural 缺陷，不是 tuning 问题。

---

## Pointwise vs Listwise：为什么 normalization 是 essential

paper 做了一个 ablation：把 listwise distribution $P_\theta = \text{softmax}(s_\theta)$ 替换成 pointwise（每个 response 独立，不归一化）：

$$\mathcal{L}_{\text{point}} = -\sum_k w_k^* \log \pi_\theta(y_k|x)$$

这是 classic AWR/MPO 的做法。结果：**pointwise variant 严重退化，比 GRPO 还差**。

**为什么**：pointwise 的 gradient coefficient 是 $c_k^{\text{point}} = -w_k^*$，是 one-sided 的——只把 mass 往高 weight response 上推，没有 counterbalancing 的拉力。它缺乏：
- Zero-sum（$\sum c_k^{\text{point}} = -1$，有 net pull）
- Boundedness（没有 relative scaling）
- Self-correction（$c_k$ 不依赖 $\pi_\theta$，policy match target 后 gradient 不消失）

**人话**：listwise normalization 提供的 shared softmax 是 essential 的 control variate。它让所有 response 通过分母耦合在一起，形成竞争机制——提一个 response 的概率必然降低其他的。这种耦合是 stable optimization 的关键。Pointwise 把每个 response 当独立目标优化，失去了这种竞争结构。

**这也是为什么 advantage normalization 必须是 zero-mean 的**——zero-mean advantage 等价于 listwise normalization 的 zero-sum property。LPO 通过 simplex 几何天然实现这一点。

---

## Monotonic Improvement：为什么 LPO 有保证而 PG 没有

**Theorem 2**：

$$\hat{R}(P_{t+1}) \geq \hat{R}(P_t) + \underbrace{\tau \cdot D_J(w^*, P_t)}_{\text{target gain} \geq 0} - \underbrace{2R_{\max}\epsilon_{\text{proj}}}_{\text{projection error}}$$

其中 $D_J = D_{\text{KL}}(w^*\|P_t) + D_{\text{KL}}(P_t\|w^*)$ 是 Jeffreys divergence（symmetric）。

**人话**：
- Target gain：只要你不是已经在 target 上，往 target 走就一定提升 reward
- Projection error：projection 不完美带来的 loss，bounded by $2R_{\max}\epsilon_{\text{proj}}$
- 只要 projection 足够好（$\epsilon_{\text{proj}}$ 小），reward 单调递增

**PG 方法为什么没这个保证**：PG 是 first-order 近似，off-policy drift 破坏了 trust region。PPO 的 clipping 只是 heuristic 限制 drift，不保证 monotonicity。LPO 在 finite simplex 上做 exact projection，trust region 被精确 maintain。

---

## 实验的 key takeaways

### 1. Paired-temperature comparison

LPO 故意 reuse baseline 的 temperature（GRPO 用 $\sigma_G$，Dr.GRPO 用 1，MaxRL 用 $\mu_G$），只改 projection mechanism。这样 gain 完全归因于 exact projection。

**Result**：Pass@1 LPO_fwd wins 13/15，Pass@k LPO_fwd wins 15/15。

### 2. Training dynamics

- **Entropy**：LPO 保持更高 entropy。LPO_fwd 是 mode-covering，LPO_rev 是 max-entropy objective。两者都 structurally 抵抗 entropy collapse。
- **Gradient norm**：LPO 更低更 stable。Corollary 1 的 boundedness property 直接体现。
- **Response length**：LPO 生成更长 response。More exploration → 更详细 reasoning chain。

### 3. Scalability

Qwen3-14B on Polaris 53k：LPO_fwd 在 70 steps 达到 GRPO 200 steps 的 peak。**3x sample efficiency**。

### 4. Fully on-policy validation

严格 on-policy（1 update per iteration）：LPO_rev 曲线与 GRPO practically 重合。**Empirically 验证 Proposition 1**——on-policy 点 reverse KL = PG。

LPO_fwd 在 on-policy 下仍然有 exploration 优势。

### 5. Generalization

Qwen、DeepSeek、Llama、Mistral 四个 family 上 LPO 一致提升。Model-agnostic。

---

## 对 RLVR 领域的 deep implication

### 1. Advantage normalization 不是 trick，是 target design

所有人在调 GRPO 的 $\sigma_G$ vs Dr.GRPO 的 $\tau=1$ vs MaxRL 的 $\mu_G$，其实都在调 target temperature。这个 unification 让我们 stop 在 advantage formula 上 empirically 试错，start systematically reason about target design。

### 2. Divergence selection 是新的 design axis

Existing PG 方法被锁死在 reverse KL（因为 PG ≈ reverse KL 近似）。LPO 的 decoupled framework unlock 了 divergence choice。Forward KL 的 mode-covering 是 structurally 不同的——它不只是"更好的 reverse KL"，是 fundamentally 不同的几何。

### 3. Simplex geometry 是 LLM RLVR 的独特红利

Classical RL 在 continuous action space 永远没法做 exact projection。LLM 的 K 个 sampled response 形成 finite simplex，partition function 是 finite sum。**这是 LLM RLVR 能跳出 classical RL approximation dilemma 的根本原因**。

### 4. Entropy bonus 是 redundant 的

Appendix C.7 揭示：加 $\gamma H(\pi_\theta)$ 等价于把 $\tau$ 增加到 $\tau + \gamma$。当 $\tau$ 是 explicit hyperparameter 时，entropy bonus 完全 redundant。这解释了为什么 DAPO、Dr.GRPO 去掉 KL penalty 仍然 work。

### 5. Mode collapse 是 reverse KL 的 structural defect

RLVR 的 entropy collapse 不是 tuning 问题，是 reverse KL（≈ PG）的 mode-seeking property 的必然结果。Forward KL 的 log-barrier（Corollary 2）是 structural fix。这给 divergence selection 提供了 principled criterion：要 diversity 用 forward KL，要 single-trajectory performance 用 reverse KL。

---

## 最终直觉

**LPO 的核心 insight**：LLM RLVR 的 group-based sampling 天然创造 finite simplex，使得 exact target-projection 变得 tractable。Existing PG 方法通过 advantage normalization 隐式做这件事但只到 first-order。LPO 通过 explicit divergence minimization：
- Recover exact monotonic improvement
- 提供 bounded/zero-sum/self-correcting gradient
- Unlock divergence selection 作为新 design axis
- Structural 防止 entropy collapse

对 Andrej 来说，这个 work 的 beauty 在于：它把一系列 empirical heuristics 统一到一个 geometric framework，揭示它们都是同一个 Gibbs target family 的不同 temperature instantiations，并指出 divergence selection 是一个 underexplored 但 potentially high-impact 的方向。Forward KL variant 的强势表现 hint：我们可能一直被锁在 reverse KL 的 mode-seeking geometry 里，错过了 mode-covering 的结构性优势。

---

# Listwise Policy Optimization (LPO) 深度解析

Andrej, 这篇paper的核心贡献是为group-based RLVR提供了一个统一的geometric perspective，揭示GRPO/Dr.GRPO/MaxRL等方法的advantage normalization本质上都在response simplex上隐式构造一个reward-weighted softmax target，再用policy gradient做first-order reverse KL projection的approximation。LPO把这件事explicit化，通过exact divergence minimization做projection。下面我详细build up intuition。

---

## 1. 核心几何图景：Response Simplex

### 1.1 为什么需要simplex视角

考虑一个prompt $x$，behavior policy $\pi_b$ 采样K个responses $\{y_1, ..., y_K\}$，每个response有reward $R_k$。在sequence-level上定义**listwise distribution**：

$$P_{\theta,k} = \frac{\exp(s_{\theta,k})}{\sum_{j=1}^K \exp(s_{\theta,j})} = \text{softmax}(s_\theta)_k$$

其中 logit 定义为：

$$s_{\theta,k} = \log \frac{\pi_\theta(y_k|x)}{\pi_b(y_k|x)}$$

**变量解释**：
- $K$：group size，每个prompt采样的response数
- $s_{\theta,k}$：policy $\pi_\theta$ 相对于 behavior policy $\pi_b$ 在response $y_k$ 上的log-ratio，本质是"relative preference logit"
- $P_{\theta,k}$：在K个候选response上的归一化偏好分布

由于 $P_{\theta,k} \geq 0$ 且 $\sum_k P_{\theta,k} = 1$，$P_\theta$ 落在probability simplex $\Delta^{K-1} \subset \mathbb{R}^K$ 上。这就是paper说的"response simplex"——一个有限、closed、tractable的概率空间。

**Key insight**：在on-policy点 $\pi_\theta = \pi_b$，所有 $s_{\theta,k} = 0$，$P_\theta$ 退化为uniform distribution $1/K$。

---

## 2. Proposition 1：PG作为reverse KL的一阶近似

这是全文最关键的theoretical observation。给定zero-mean advantage vector $A$（即 $\sum_k A_k = 0$），定义target：

$$w^* = \text{softmax}(A)$$

**Proposition 1** claim：在on-policy点，标准policy gradient：

$$g_{\text{PG}} = \frac{1}{K}\sum_{k=1}^K A_k \nabla_\theta \log \pi_\theta(y_k|x)$$

恰好等于reverse KL divergence的负梯度：

$$g_{\text{PG}} = -\nabla_\theta D_{\text{KL}}(P_\theta \| w^*)\bigg|_{\pi_\theta = \pi_b}$$

### 2.1 证明intuition（Appendix B.2）

reverse KL的gradient coefficient通过logit-gap简化得到：

$$c_k^{\text{rev}} = P_{\theta,k}(d_k - \bar{d})$$

其中：
- $d_k = s_{\theta,k} - \phi_k$：logit gap，current policy logit与target logit的差
- $\phi_k$：target logit，这里取 $\phi_k = A_k$
- $\bar{d} = \sum_j P_{\theta,j} d_j$：$P_\theta$-weighted mean of logit gap

在on-policy点：$s_{\theta,k} = 0$，所以 $P_{\theta,k} = 1/K$，$d_k = -A_k$。

由zero-mean假设 $\sum_k A_k = 0$：

$$\bar{d} = \frac{1}{K}\sum_k (-A_k) = 0$$

代入：

$$c_k^{\text{rev}}\big|_{\text{on-policy}} = \frac{1}{K}(-A_k - 0) = -\frac{A_k}{K}$$

而PG的coefficient是 $c_k^{\text{PG}} = A_k/K$。两者相差一个负号，因为PG是maximize而KL是minimize。**QED**。

### 2.2 Off-policy approximation error

这个等式只在on-policy点精确成立。一旦policy drift，error scales as：

$$|\Delta_k| \leq \frac{C\bar{\delta}(1 + \|A\|_\infty)}{K}$$

其中 $\bar{\delta} = \max_k |r_k - 1|$，$r_k = \pi_\theta(y_k|x)/\pi_b(y_k|x)$ 是importance ratio。

**Intuition**：$\bar{\delta}$ 是off-policy drift的measure。PG在drift大的时候偏离exact reverse KL projection，这就是为什么PG方法需要clipping（PPO-style）来stabilize——clipping是一种ad-hoc的trust region enforcement，掩盖了底层approximation退化的事实。

### 2.3 现有方法的implicit targets

通过shift-invariance of softmax，advantage $A_k = (R_k - \mu)/\tau$ 的centering $\mu$ 在softmax里被cancel，所以：

$$w^* = \text{softmax}(R/\tau)$$

不同方法的区别仅在于temperature $\tau$：

| Algorithm | Advantage $A_k$ | Implicit target $w^*$ | Temperature $\tau$ |
|-----------|-----------------|----------------------|---------------------|
| Dr.GRPO / RLOO | $R_k - \mu_G$ | $\text{softmax}(R)$ | $\approx 1$ |
| GRPO / DAPO | $(R_k - \mu_G)/\sigma_G$ | $\text{softmax}(R/\sigma_G)$ | $\sigma_G$ |
| MaxRL | $(R_k - \mu_G)/\mu_G$ | $\text{softmax}(R/\mu_G)$ | $\mu_G$ |
| REINFORCE++ w/ Baseline | two-stage norm | $\text{softmax}(R/\sigma_{B'})$ | $\sigma_{B'}$ |

**Intuition**：所有方法都在追求同一个reward-ranked softmax target family，差别仅在sharpness。$\tau$ 小→target尖锐（concentrate on best response）；$\tau$ 大→target平缓（maintain diversity）。

特别地，MaxRL用 $\tau = \mu_G$（success rate）作为implicit curriculum：hard prompts（$\mu_G$小）得到sharp target鼓励exploitation，easy prompts（$\mu_G$大）得到diffuse target保持exploration。

---

## 3. LPO：Explicit Target-Projection Framework

### 3.1 两步decoupled优化

LPO把每次iteration分成两个entangled steps：

$$\underbrace{w^* = \arg\max_{w \in \Delta^{K-1}} \hat{J}(w)}_{\text{(i) Target: aim for what}} \qquad \underbrace{\theta' = \arg\min_\theta D(w^* \| P_\theta)}_{\text{(ii) Projection: how to get there}}$$

**这是关键decoupling**：existing PG方法把"target构造"和"projection方法"耦合在advantage formula里，LPO把它们分离，允许任意选择divergence。

### 3.2 Theorem 1：Listwise Gibbs Target

定义proximal RL objective on simplex：

$$\hat{J}(w) = \sum_{k=1}^K w_k R_k - \tau D_{\text{KL}}(w \| P_t)$$

**变量解释**：
- $w \in \Delta^{K-1}$：待优化的listwise distribution
- $R_k$：response $k$ 的reward
- $\tau > 0$：trust region temperature，控制target sharpness
- $P_t = \text{softmax}(s_t)$：pre-update policy $\pi_t$ 诱导的listwise distribution，作为trust region anchor
- $s_{t,k} = \log(\pi_t(y_k|x)/\pi_b(y_k|x))$

这个objective是classic trust-region RL objective（TRPO/Schulman 2015风格）的listwise版本：maximize expected reward subject to KL constraint around $P_t$。

**Theorem 1** 给出closed-form解：

$$w_k^* = \text{softmax}(\phi)_k, \quad \phi_k = \frac{R_k}{\tau} + s_{t,k}$$

**变量解释**：
- $\phi_k$：target logit，由reward term $R_k/\tau$ 和anchor logit $s_{t,k}$ 相加
- $\tau$：温度，$\tau \to 0$ 时 $w^* \to \arg\max_k R_k$（greedy），$\tau \to \infty$ 时 $w^* \to P_t$（保守）

**极限行为**：
- $\tau \to 0$：target collapse到最大reward的response，aggressive exploitation
- $\tau \to \infty$：target保持不动，no update
- 中间 $\tau$：soft reweighting，trade-off exploration/exploitation

在on-policy setup（$\pi_t = \pi_b$），$s_{t,k} = 0$，$P_t = 1/K$ uniform，于是：

$$w^* = \text{softmax}(R/\tau)$$

**这恰好recover了existing methods的implicit target**！所以LPO的target构造是一个principled generalization，把existing methods的implicit heuristic变成explicit trust-region formulation。

### 3.3 Proof intuition（Appendix B.3）

用Lagrangian：

$$\mathcal{L}(w, \lambda) = \sum_k w_k R_k - \tau \sum_k w_k \log\frac{w_k}{P_{t,k}} + \lambda(1 - \sum_k w_k)$$

Stationary condition $\partial \mathcal{L}/\partial w_k = 0$：

$$R_k - \tau(\log w_k - \log P_{t,k} + 1) - \lambda = 0$$

解出：

$$w_k = P_{t,k} \exp(R_k/\tau) \cdot C$$

其中 $C$ 是normalization constant。这就是**Gibbs distribution**形式：target = anchor × Boltzmann factor of reward。

由于 $\hat{J}(w)$ 在simplex上strictly concave（reward term线性，$D_{\text{KL}}(w \| P_t)$ strictly convex for $P_{t,k} > 0$），maximizer唯一。

### 3.4 Theorem 2：Monotonic Improvement Bound

**Theorem 2**：假设 $|R_k| \leq R_{\max}$，且projection达到 $\text{TV}(P_{t+1}, w^*) \leq \epsilon_{\text{proj}}$，则：

$$\hat{R}(P_{t+1}) \geq \hat{R}(P_t) + \underbrace{\tau[D_{\text{KL}}(w^* \| P_t) + D_{\text{KL}}(P_t \| w^*)]}_{\text{target gain} \geq 0} - \underbrace{2R_{\max}\epsilon_{\text{proj}}}_{\text{projection error}}$$

**变量解释**：
- $\hat{R}(P) = \sum_k P_k R_k$：listwise expected reward
- $\text{TV}(\cdot, \cdot)$：total variation distance
- $\epsilon_{\text{proj}}$：projection error，衡量 $P_{t+1}$ 离target $w^*$ 有多远
- $R_{\max}$：reward上界

**Intuition**：
- Target gain = Jeffreys divergence $D_J(w^*, P_t) = D_{\text{KL}}(w^* \| P_t) + D_{\text{KL}}(P_t \| w^*)$，symmetric，non-negative
- 完美projection（$\epsilon_{\text{proj}} = 0$）时，只要 $P_t \neq w^*$，reward严格单调递增
- Projection error bound通过Pinsker's inequality + Hölder's inequality得到

**为什么这是strong guarantee**：existing PG方法没有这种monotonic improvement guarantee，因为它们是一阶近似，off-policy drift会破坏monotonicity。LPO通过exact projection（在finite simplex上）保留了trust region的monotonic property。

### 3.5 Proposition 2：Full-space convergence

在idealized full policy space（$K \to \infty$），exact proximal update：

$$\pi_{t+1}(y) \propto \pi_t(y) \exp(R(y)/\tau)$$

迭代得到：

$$\pi_t(y) \propto \pi_0(y) \exp(tR(y)/\tau)$$

且 $\mathbb{E}_{\pi_t}[R] \to \max_y R(y)$ as $t \to \infty$。

**Intuition**：这是Boltzmann exploration的classic结果（Ziebart 2010, MaxEnt RL）。LPO在finite response simplex上是这个idealized iteration的tractable approximation——partition function从不可计算的 $\sum_y \pi_0(y)\exp(tR(y)/\tau)$ 变成有限的 $\sum_{k=1}^K P_{t,k}\exp(R_k/\tau)$。

---

## 4. Projection：Forward KL vs Reverse KL

### 4.1 Example 1：Forward KL Projection

Minimize $D_{\text{KL}}(w^* \| P_\theta)$：

$$\nabla_\theta \mathcal{L}_{\text{LPO}_{\text{fwd}}} = \sum_{k=1}^K \underbrace{(P_{\theta,k} - w_k^*)}_{c_k^{\text{fwd}}} \nabla_\theta \log \pi_\theta(y_k|x)$$

**变量解释**：
- $c_k^{\text{fwd}} = P_{\theta,k} - w_k^*$：forward KL gradient coefficient，是current policy listwise probability与target probability的差

**Proof intuition**（Appendix B.1）：

$$D_{\text{KL}}(w^* \| P_\theta) = -\sum_k w_k^* \log P_{\theta,k} - H(w^*)$$

利用log-softmax Jacobian：

$$\nabla_\theta \log P_{\theta,k} = \nabla_\theta s_{\theta,k} - \sum_j P_{\theta,j} \nabla_\theta s_{\theta,j}$$

代入展开：

$$\nabla_\theta D_{\text{KL}}(w^* \| P_\theta) = -\sum_k w_k^* \nabla_\theta s_{\theta,k} + \underbrace{\left(\sum_k w_k^*\right)}_{=1} \sum_j P_{\theta,j} \nabla_\theta s_{\theta,j}$$

reindexing并利用 $\nabla_\theta s_{\theta,k} = \nabla_\theta \log \pi_\theta(y_k|x)$（因为 $\pi_b$ frozen），得到：

$$\nabla_\theta D_{\text{KL}}(w^* \| P_\theta) = \sum_k (P_{\theta,k} - w_k^*) \nabla_\theta \log \pi_\theta(y_k|x)$$

### 4.2 Corollary 1：Forward KL gradient的三个性质

**(a) Bounded**: $|c_k^{\text{fwd}}| \leq 1$

因为 $P_{\theta,k}, w_k^* \in [0,1]$，所以 $c_k^{\text{fwd}} \in [-1, 1]$。

更紧的bound：$\sum_k |c_k^{\text{fwd}}| \leq 2$。

**Proof**：partition positive/negative parts：

$$\sum_{c_k > 0} c_k = -\sum_{c_k < 0} c_k$$

所以 $\sum_k |c_k| = 2\sum_{c_k > 0} c_k \leq 2\sum_{c_k > 0} P_{\theta,k} \leq 2$。

**(b) Zero-sum**: $\sum_k c_k^{\text{fwd}} = 0$

$$\sum_k c_k^{\text{fwd}} = \sum_k P_{\theta,k} - \sum_k w_k^* = 1 - 1 = 0$$

**Intuition**：zero-sum property相当于built-in control variate（Sutton 1988），自动做mean-centering，降低gradient variance。这就是为什么advantage normalization需要zero-mean——LPO的forward KL projection通过simplex结构天然实现。

**(c) Self-correcting**: $c_k^{\text{fwd}} \to 0$ as $P_\theta \to w^*$

当policy完全match target时，gradient自动消失，无需external stopping criterion。

### 4.3 Corollary 2：Mode-Coverage Property

如果 $w_k^* \geq \alpha$ 且 $D_{\text{KL}}(w^* \| P_\theta) \leq D$，则：

$$P_{\theta,k} > \alpha \exp(-D/\alpha - 1)$$

**变量解释**：
- $\alpha$：target distribution下response $k$ 的最小概率
- $D$：KL divergence上界

**Intuition**：forward KL是mode-covering（mass-covering）的。如果target给response $k$ 至少 $\alpha$ 的概率，current policy在KL约束下不会让 $P_{\theta,k}$ 跌破 $\alpha \exp(-D/\alpha - 1)$。这给mode collapse提供了log-barrier。

**Proof sketch**（Appendix B.8）：用Data Processing Inequality构造binary event space（"是response k" vs "不是response k"），binary KL ≤ full KL ≤ $D$。Drop掉non-negative term后得到lower bound。

**Practical implication**：forward KL structurally preserves response diversity，这就是为什么实验里 $\text{LPO}_{\text{fwd}}$ 在Pass@k上比 $\text{LPO}_{\text{rev}}$ 强很多。

### 4.4 Example 2：Reverse KL Projection

Minimize $D_{\text{KL}}(P_\theta \| w^*)$：

$$\nabla_\theta \mathcal{L}_{\text{LPO}_{\text{rev}}} = \sum_{k=1}^K \underbrace{P_{\theta,k}(d_k - \bar{d})}_{c_k^{\text{rev}}} \nabla_\theta \log \pi_\theta(y_k|x)$$

**变量解释**：
- $d_k = s_{\theta,k} - \phi_k$：logit gap，current policy logit与target logit的差
- $\bar{d} = \sum_j P_{\theta,j} d_j$：$P_\theta$-weighted mean of logit gap

**关键性质**：
- Zero-sum：$\sum_k c_k^{\text{rev}} = 0$（同forward KL）
- Self-correcting：$P_\theta \to w^*$ 时 $d_k \to \bar{d}$，$c_k \to 0$
- **Implicit entropy bonus**：reverse KL decompose为 $-H(P_\theta) - \sum_k P_{\theta,k} \phi_k$，所以minimize reverse KL = maximize $H(P_\theta) + \sum_k P_{\theta,k} \phi_k$，这是MaxEnt RL objective（Ziebart 2010, Soft Actor-Critic Haarnoja 2018）
- **Mode-seeking**：reverse KL在 $w^*$ 有多个mode时倾向于concentrate在一个mode上（不像forward KL那样cover所有mode）

**为什么reverse KL在on-policy点recover PG**：由Proposition 1直接得出。reverse KL是LPO里"最接近PG"的variant。

### 4.5 Logit-gap simplification（Appendix B.1）

写出 $P_{\theta,k} = \exp(s_{\theta,k})/Z_s$ 和 $w_k^* = \exp(\phi_k)/Z_\phi$，则：

$$\log\frac{P_{\theta,k}}{w_k^*} = (s_{\theta,k} - \phi_k) - (\log Z_s - \log Z_\phi) = d_k - c_s$$

其中 $c_s = \log Z_s - \log Z_\phi$ 是跨所有k的constant。

reverse KL：

$$D_{\text{KL}}(P_\theta \| w^*) = \sum_k P_{\theta,k}(d_k - c_s) = \bar{d} - c_s$$

代入gradient coefficient：

$$c_k^{\text{rev}} = P_{\theta,k}[(d_k - c_s) - (\bar{d} - c_s)] = P_{\theta,k}(d_k - \bar{d})$$

**Beautiful result**：normalization constant $c_s$ 完美cancel，gradient coefficient只依赖logit gap的centered version。这就是为什么reverse KL在on-policy点严格等于PG——centering mechanism天然实现zero-mean advantage。

---

## 5. Algorithm 1：LPO完整pipeline

```
Input: θ, τ > 0, batch size B, inner epochs E, step size η
1: for each iteration:
2:   π_b ← π_θ (behavior), π_t ← π_θ (anchor)
3:   Sample B prompts
4:   For each x: sample K responses, compute rewards R
5:   Compute target: w*(x) = softmax(R/τ + s_t)  // Eq.8
6:   for e = 1 to E:
7:     Compute coefficients c_k(x)  // Eq.10 (fwd) or Eq.11 (rev)
8:     θ ← θ - η · (1/B) Σ_x Σ_k c_k(x) ∇_θ log π_θ(y_k|x)
9:   end for
10: end for
```

**Computational cost**：与标准group-based PG完全相同。多出来的cost只是计算softmax target（O(K) per prompt）和gradient coefficient（O(K) per prompt），相比rollout和backward pass可忽略。

**Temperature as adaptive baseline**：paper故意不引入新tuning，直接reuse existing methods的normalization statistics：
- GRPO: $\tau = \sigma_G$
- Dr.GRPO: $\tau = 1$
- MaxRL: $\tau = \mu_G$

这样isolates gain来自exact projection而非temperature tuning，做apples-to-apples comparison。

---

## 6. 实验：4个domain × 8个backbone

### 6.1 Setup

| Domain | Task | Model | Train data |
|--------|------|-------|-----------|
| Logic | Countdown | Qwen3-4B-Base | Countdown-34 (2k) |
| Math | MATH | Qwen3-1.7B/8B/14B-Base | MATH 7.5k / Polaris 53k |
| Code | PRIME-Code | Qwen3-1.7B-Base | Eurus-2-RL 25.3k |
| Multi-modal | Geometry3k | Qwen2.5-VL-3B-Instruct | Geometry3k 2.1k |

**Baselines**：GRPO ($\tau=\sigma_G$), Dr.GRPO ($\tau=1$), MaxRL ($\tau=\mu_G$)

**LPO variants**：$\text{LPO}_{\text{fwd}}$, $\text{LPO}_{\text{rev}}$，每个variant配对应baseline的temperature。

**关键设计**：paired-temperature evaluation——同temperature下比较，gain完全归因于projection mechanism。

### 6.2 Main results（Table 3, Qwen3-8B-Base on MATH）

| Method | MATH500 | Olympiad | Minerva | AMC23 | AIME24 | AIME25 | Pass@1 | Pass@k |
|--------|---------|----------|---------|-------|--------|--------|--------|--------|
| Base | 68.0 | 33.7 | 31.7 | 33.3 | 7.9 | 31.8 | 33.3 | 50.0 |
| GRPO | 86.2 | 51.9 | 40.4 | 63.8 | 24.0 | 19.5 | 47.6 | 54.5 |
| →LPO_fwd | **86.4** | **55.8** | **42.3** | **69.1** | **29.3** | 19.1 | **50.3** | **58.3** |
| →LPO_rev | 85.0 | 53.9 | 41.1 | 67.0 | 23.3 | **21.6** | 48.7 | 56.5 |
| Dr.GRPO | 85.8 | 54.7 | 42.2 | 67.7 | 24.9 | 19.3 | 49.1 | 60.4 |
| →LPO_fwd | 87.4 | 51.6 | 42.6 | 70.2 | 26.0 | 17.9 | 49.5 | 59.4 |
| →LPO_rev | 84.6 | 51.0 | 42.0 | 64.9 | 26.0 | 38.4 | 47.7 | 56.9 |
| MaxRL | 86.4 | 53.6 | 42.6 | 66.0 | 23.9 | 18.9 | 48.6 | 58.2 |
| →LPO_fwd | 89.4 | 54.5 | 44.8 | 69.0 | 23.9 | 21.3 | 50.5 | 63.1 |
| →LPO_rev | 87.6 | 55.8 | 45.3 | 70.1 | 22.5 | 22.5 | 50.6 | 60.3 |

**Stats**：
- Pass@1: LPO_fwd wins 13/15, LPO_rev wins 13/15
- Pass@k: LPO_fwd wins 15/15, LPO_rev wins 11/15
- LPO_fwd在Pass@k上系统性优于LPO_rev（13/15），符合mode-coverage理论

### 6.3 Training dynamics（Figure 5）

三个key observations：

**(1) Response entropy**：LPO variants保持更高entropy。LPO_rev是max-entropy objective（reverse KL = $-H - \sum P\phi$），LPO_fwd是mode-covering。两者都structurally抵抗entropy collapse，这是RLVR的common failure mode。

**(2) Gradient norms**：LPO的gradient norm更低更stable。Corollary 1保证 $|c_k| \leq 1$ 且 $\sum|c_k| \leq 2$，给gradient提供intrinsic bound。PG方法的advantage $\|A\|_\infty$ 可以任意大（特别是early training时reward variance大），导致gradient爆炸。

**(3) Response length**：LPO生成更长的response。Longer chain通常correlate with更详细的reasoning（CoT），符合LPO鼓励exploration的特性。

### 6.4 Ablation：Listwise vs Pointwise Projection（Figure 6）

把listwise distribution $P_\theta$ ablate掉，保留target $w^*$，做pointwise projection：

$$\mathcal{L}_{\text{point}} = -\sum_k w_k^* \log \pi_\theta(y_k|x)$$

**Result**：pointwise variant严重退化，比GRPO还差。

**Why**：pointwise projection的gradient coefficient $c_k^{\text{point}} = -w_k^*$，是one-sided的（只push mass toward high-weight responses，没有counterbalancing force）。它缺乏：
- Zero-sum property（$\sum c_k^{\text{point}} = -1$，net pull on parameters）
- Bounded gradients（no relative scaling）
- Self-correcting convergence（$c_k$ constant w.r.t. $\pi_\theta$）

**关键结论**：LPO的gain不只来自target design，来自successfully marrying exact target fitting + listwise structure的variance reduction。Listwise normalization提供的control variate是essential的。

### 6.5 Group size K effect（Figure 7）

在Countdown上测 $K \in \{2, 4, 8, 16, 32\}$：
- LPO在small K时优势更明显——exact projection提升sample efficiency
- LPO_rev在Pass@1上更强
- LPO_fwd在Pass@64上scale exceptionally well，支持mode-coverage理论

**Intuition**：small K时simplex是low-dimensional的，approximation error大，exact projection的gain更显著。Large K时simplex逼近full space，所有方法converge。

### 6.6 Scalability（Figure 8, Appendix E.1）

Qwen3-14B-Base on Polaris 53k：
- LPO_fwd在70 steps达到GRPO 200 steps的peak performance——3x sample efficiency
- LPO_rev在Pass@k上保持robust diversity

### 6.7 Fully on-policy validation（Figure 12, Appendix E.4）

严格on-policy（batch size = mini-batch size = 256，每iteration一次gradient update）：
- LPO_rev曲线与GRPO practically indistinguishable
- 这empirically验证Proposition 1：on-policy点reverse KL = PG

**LPO_fwd在on-policy下仍然有distinct exploration superiority**：early training sample efficiency更高，Pass@k更强。

### 6.8 Generalization across LLM families（Figure 11）

Countdown task上测Qwen, DeepSeek-R1-Distill-Qwen-1.5B, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.1：
- LPO在所有backbone上一致提升
- 不sensitive to specific model architecture
- model-agnostic improvement

---

## 7. 与相关工作的关系

### 7.1 RL as Probabilistic Inference

LPO根植于RL-as-inference传统（Dayan & Hinton 1997, Ziebart 2010, Levine 2018）：
- KL-regularized objective $J(\pi) = \mathbb{E}_\pi[R] - \beta D_{\text{KL}}(\pi \| \pi_{\text{ref}})$
- 最优policy是Gibbs distribution $\pi^*(y) \propto \pi_{\text{ref}}(y)\exp(R(y)/\beta)$
- Trust region structure（TRPO Schulman 2017a）

经典方法如MPO（Abdolmaleki 2018）、AWR（Peng 2019）、V-MPO（Song 2019）、SAC（Haarnoja 2018）、REPS（Peters 2010）都在continuous action space用pointwise projection。**LPO的关键差异**：LLM的sampled responses天然形成finite simplex，partition function是finite sum，可以做exact listwise projection with shared normalization。

### 7.2 DPO和preference optimization

K=2时LPO退化成pairwise objective（Appendix C.5）：

$$\mathcal{L}_{\text{LPO}_{\text{fwd}}} = -\sigma(1/\tau)\log\sigma(s_w - s_l) - \sigma(-1/\tau)\log\sigma(s_l - s_w)$$

对比DPO：

$$\mathcal{L}_{\text{DPO}} = -\log\sigma(\beta(s_w - s_l))$$

**四个差异**：
1. DPO是offline + static dataset；LPO是online RL
2. DPO penalize against static reference $\pi_{\text{ref}}$；LPO用trust region around pre-update $\pi_t$
3. DPO源自Bradley-Terry preference model；LPO源自explicit divergence projection on simplex
4. LPO用soft targets controlled by $\tau$，$\tau \to 0$时recover hard preference

**LiPO（Liu 2025a）** 也是listwise preference optimization，但用Plackett-Luce ranking model从ranked preference data学习。LPO从absolute reward直接构造Gibbs target，无需ranking assumption。

### 7.3 Concurrent works

- **TPO（Kaddour 2026）**：similarly采用cross-entropy on tilted simplex targets，empirically corroborate forward KL efficacy
- **FlowRL（Zhu 2025）**：minimize reverse KL against Gibbs target，但用learned partition function network近似
- **Shu et al. 2026**：reference-sampled Boltzmann projection，target-matched weighted SFT视角

LPO的unique贡献：unifying analytical framework that recovers existing group-based methods as implicit instances，并admits multiple divergences with provable properties。

### 7.4 ListNet（Cao 2007）

Listwise formulation在learning-to-rank有long history。ListNet用Plackett-Luce model优化permutation probability。LPO借用listwise distribution概念但用在RLVR的policy optimization上，从verifiable rewards直接构造target而非从preference data推断。

---

## 8. 为什么Forward KL比Reverse KL好（实验观察）

实验上LPO_fwd在Pass@k上系统性优于LPO_rev（13/15 scenarios）。理论解释：

| Property | Forward KL $D(w^* \| P_\theta)$ | Reverse KL $D(P_\theta \| w^*)$ |
|----------|--------------------------------|--------------------------------|
| Mode behavior | Mass-covering / mean-seeking | Mode-seeking |
| Diversity | Preserves all modes | Concentrates on one mode |
| Gradient coeff | $P_{\theta,k} - w_k^*$ (bounded, zero-sum) | $P_{\theta,k}(d_k - \bar{d})$ (zero-sum) |
| Implicit entropy | No explicit bonus | Yes, $-H(P_\theta)$ term |
| Connection to PG | None directly | On-policy点等于PG |

**Intuition**：reasoning tasks有multiple valid solution paths。Forward KL的mode-covering property确保policy explore所有这些paths，所以Pass@k（k个sample里至少一个对的概率）高。Reverse KL倾向于concentrate在最likely的一个path上，Pass@1可能不差但Pass@k退化。

**RLVR的entropy collapse问题**（Li et al. 2025）：标准PG方法在训练后期policy collapse到少数deterministic trajectories，失去diversity。Forward KL的log-barrier（Corollary 2）structurally防止这种collapse。

---

## 9. Limitations和Future Directions

### 9.1 Step-level extension

Current framework是sequence-level。Multi-turn agentic RL（Search-R1 Jin 2025）或step-level rewards（PRM Lightman 2023）需要：
- 给定shared intermediate state，sample K candidate continuations形成local simplex
- 用value network或rollout估计step-level expected final outcome
- Core LPO machinery carries over

### 9.2 Off-policy replay

LPO理论上可以incorporate off-policy data：
- 记录past $\pi_b$
- 用importance ratio $\pi_t/\pi_b$ 和 $\pi_\theta/\pi_b$ 校正
- Listwise normalization相当于self-normalized importance sampling (SNIS) estimator

**Challenge**：policy evolve后stale checkpoints导致extreme probability ratios，collapse listwise distributions。需要staleness-filtering或trust-region buffer management。

### 9.3 K=1的virtual group

Single-sample pipelines（K=1）没有physical simplex。可以construct virtual group by pairing sampled reward $R$ with batch-level baseline $b$：

$$c = \frac{1}{2} - \sigma\left(\frac{R - b}{\tau}\right)$$

保持bounded $|c| \leq 1/2$，但sacrifice zero-sum property。

### 9.4 Alternative divergences和adaptive scheduling

Decoupled framework允许任意differentiable divergence on $\Delta^{K-1}$。Appendix C.6证明zero-sum property对任意softmax-parameterized objective都成立：

$$\sum_k c_k = \sum_k P_{\theta,k}\frac{\partial \mathcal{L}}{\partial P_{\theta,k}} - \underbrace{\left(\sum_k P_{\theta,k}\right)}_{=1}\left(\sum_j P_{\theta,j}\frac{\partial \mathcal{L}}{\partial P_{\theta,j}}\right) = 0$$

但boundedness和mode behavior依赖具体divergence。可以探索：
- Jensen-Shannon divergence（symmetric, always finite）
- $\alpha$-divergence family（tunable mode-seeking/covering）
- Adaptive scheduling：early training用forward KL鼓励exploration，late training切到reverse KL做exploitation

---

## 10. 个人intuition和open questions

### 10.1 为什么这个framework重要

LPO最大的价值不是empirical gain（虽然consistent），是**conceptual clarity**。它揭示了一个被掩盖的事实：advantage normalization的所有heuristic variations（GRPO的 $\sigma_G$、Dr.GRPO的 $\tau=1$、MaxRL的 $\mu_G$）都是在做同一件事——构造不同sharpness的softmax target。这个unification让我们能systematically design新的RLVR algorithms，而不是在advantage formula上empirically试错。

### 10.2 Off-policy drift的根源

Proposition 1的off-policy error analysis（Eq.28）给出：

$$\|g_{\text{PG}} - g_{\text{rev KL}}\| \leq C'\bar{\delta}(1 + \|A\|_\infty)G_{\max}$$

$\bar{\delta}$ linear in drift，$\|A\|_\infty$ linear in advantage magnitude。这意味着：
- Large advantage（high reward variance group）→ fast approximation degradation
- Multi-step gradient updates（PPO的mini-batch epochs）→ accumulated drift

LPO通过exact projection消除这个error。**这就是为什么LPO在multi-epoch PPO-style training下比PG更stable**——gradient coefficient $c_k$ 始终是exact的，不依赖on-policy assumption。

### 10.3 Simplex geometry的深层意义

Response simplex是LLM RLVR的独特structure。continuous action space的RL（robotics等）必须用function approximation做projection，partition function不可计算。LLM的sampled responses形成finite set，partition function是finite sum $\sum_{k=1}^K$，可以做exact computation。

这解释了为什么LLM RLVR能跳出classical RL的approximation dilemma——finite simplex让exact geometric projection变得tractable。LPO是第一个systematically exploit这个structure的方法。

### 10.4 Connection to natural gradient和mirror descent

Listwise projection本质是mirror descent（Tomar 2020）在simplex上的instantiation。Forward KL对应一个特定的Bregman divergence。可以connection到natural gradient（Amari 1998, Kakade 2001）——KL divergence的Fisher information matrix就是natural metric。

**Speculation**：LPO的forward KL projection可能等价于在listwise distribution space上做natural gradient step toward target。这会给LPO提供information-geometric foundation。

### 10.5 Entropy bonus的redundancy

Appendix C.7揭示：加entropy bonus $\gamma H(\pi_\theta)$ 等价于把target temperature从 $\tau$ 增加到 $\tau + \gamma$：

$$\tilde{w}^* = \text{softmax}(R/(\tau + \gamma))$$

**Implication**：当 $\tau$ 是explicit hyperparameter时，entropy bonus是redundant的。这解释了为什么DAPO、Dr.GRPO等方法去掉KL penalty仍然work——只要 $\tau$ 调好，trust region已经通过target sharpness实现。

### 10.6 Mode collapse的geometric理解

RLVR的entropy collapse本质是reverse KL的mode-seeking property。PG方法（Proposition 1）implicitly做reverse KL projection，所以structurally prone to mode collapse。Forward KL的mode-covering property（Corollary 2）提供log-barrier，structurally防止collapse。

**这给divergence selection提供了principled criterion**：如果想maintain diversity（Pass@k重要），用forward KL；如果想maximize single-trajectory performance（Pass@1重要），用reverse KL。Adaptive scheduling可以两者兼得。

---

## 11. 参考链接

**Core papers**：
- LPO paper本身（这篇）
- GRPO: https://arxiv.org/abs/2402.03300 (DeepSeekMath, Shao et al. 2024)
- PPO: https://arxiv.org/abs/1707.06347 (Schulman et al. 2017b)
- TRPO: https://arxiv.org/abs/1502.05477 (Schulman et al. 2017a)
- DPO: https://arxiv.org/abs/2305.18290 (Rafailov et al. 2024)

**RL as inference**:
- Levine tutorial: https://arxiv.org/abs/1805.00909
- Ziebart PhD thesis (MaxCausalEnt): http://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf
- Soft Actor-Critic: https://arxiv.org/abs/1801.01290 (Haarnoja et al. 2018)
- MPO: https://arxiv.org/abs/1806.06920 (Abdolmaleki et al. 2018)
- AWR: https://arxiv.org/abs/1910.00177 (Peng et al. 2019)
- REPS: Peters et al. 2010 AAAI

**RLVR methods**:
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
- DAPO: https://arxiv.org/abs/2503.14476 (Yu et al. 2025)
- Dr.GRPO: https://arxiv.org/abs/2503.20783 (Liu et al. 2025b)
- REINFORCE++: https://arxiv.org/abs/2501.03262 (Hu 2025)
- PRIME: https://arxiv.org/abs/2502.01456 (Cui et al. 2025)
- MaxRL: https://arxiv.org/abs/2602.02710 (Tajwar et al. 2026)
- GSPO: https://arxiv.org/abs/2507.18071 (Zheng et al. 2025)

**Listwise / Learning to rank**:
- ListNet: Cao et al. 2007 ICML (https://dl.acm.org/doi/10.1145/1273496.1273513)
- Plackett-Luce: Plackett 1975
- LiPO: https://arxiv.org/abs/2402.01878 (Liu et al. 2025a)

**Concurrent works**:
- TPO (Kaddour 2026): https://arxiv.org/abs/2604.06159
- FlowRL: https://arxiv.org/abs/2509.15207 (Zhu et al. 2025)
- Shu et al. 2026: https://arxiv.org/abs/2605.02469

**Frameworks**:
- verl: https://arxiv.org/abs/2409.19256 (Sheng et al. 2024)
- TinyZero: https://github.com/Jiayi-Pan/TinyZero

**Math foundations**:
- Pinsker's inequality: https://en.wikipedia.org/wiki/Pinsker%27s_inequality
- Jeffreys divergence: Jeffreys 1946
- Natural gradient: Amari 1998 Neural Computation
- Mirror descent: Tomar 2020 https://arxiv.org/abs/2005.09814

---

## 12. 总结

LPO的deep insight是：**LLM RLVR的group-based sampling天然创造finite response simplex，使得classical RL中tractable的exact target-projection变得可能**。Existing PG方法通过advantage normalization隐式做这件事，但只到first-order approximation。LPO通过explicit divergence minimization：
1. Recover exact monotonic improvement guarantee（Theorem 2）
2. 提供bounded/zero-sum/self-correcting gradient（Corollary 1）
3. 通过divergence选择unlock新的design axis（forward KL的mode-coverage）
4. Structural防止entropy collapse（Corollary 2的log-barrier）

对 Andrej 来说，这个work的beauty在于它把一系列empirical heuristics（GRPO的 $\sigma_G$ normalization、Dr.GRPO的 $\tau=1$、MaxRL的 $\mu_G$ curriculum）统一到一个geometric framework里，并指出它们都是同一个Gibbs target family的不同temperature instantiations。这种unification让我们能systematically reason about RLVR algorithm design，而非靠empirical trial-and-error。Forward KL variant的强势表现也hint：divergence selection是一个underexplored但potentially high-impact的设计axis，值得更多理论分析。
