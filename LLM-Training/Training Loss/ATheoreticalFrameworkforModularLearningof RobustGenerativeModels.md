---
source_pdf: ATheoreticalFrameworkforModularLearningof RobustGenerativeModels.pdf
paper_sha256: cb9f2a6d74be233a0154dbd774b025bec8aaca8e25ad265e23285dc27c9ce072
processed_at: '2026-07-18T10:01:26-07:00'
target_folder: LLM-Training/Training Loss
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# 这篇 Paper 在干什么——一个 Game-Theoretic 视角下的 Modular LLM 框架

Corinna Cortes / Mehryar Mohri / Yutao Zhong 这篇 paper 想回答一个特别 pragmatic 的问题: 我们能不能不 train 一个 huge monolithic LLM, 而是 train 一堆 small domain experts(math, code, wiki...), 然后用一个 lightweight 的 gate 把它们 glue 起来, 并且这个组合对**任何** test-time data mixture 都 robust——不需要 heuristic 权重调参, 不需要 retrain.

这本质上是 **multiple-source domain adaptation (MSA)** 在 generative modeling 上的延伸, 配合 **Distributionally Robust Optimization (DRO)** 的 minimax formulation. Mohri 这条线已经做了很多年 [^msa], 这篇是把它的 machinery 搬到 KL divergence + global normalization 的 generative setting 上, 几个关键的非 trivial 点我会下面拆开讲.

---

## 1. Setup: 为什么直接拼 expert 不行, 非要引入 gate

我们有 $p$ 个 dataset $D_k$, 每个 $D_k$ 的 empirical distribution 是 $\widehat{\mathsf p}_k$, 已经 pre-train 出 expert $\widehat\pi_k$, 且保证 $\mathsf D_{\mathrm{KL}}(\widehat{\mathsf p}_k \| \widehat\pi_k) \le \epsilon_k$. 这里 $\epsilon_k$ 就是单个 expert 在自己 domain 上的 irreducible error.

Test time 我们不知道 mixture $\lambda \in \Delta([1,p])$ 是什么, target 分布是
$$\widehat{\mathsf p}_\lambda = \sum_{k=1}^p \lambda_k \widehat{\mathsf p}_k$$
其中 $\lambda = (\lambda_1,\dots,\lambda_p)$, $\lambda_k \ge 0$, $\sum_k \lambda_k = 1$, $\Delta([1,p])$ 就是 $p$-simplex.

候选模型是 **gated mixture**:
$$\pi_g(x) = \sum_{k=1}^p g(x,k)\,\widehat\pi_k(x)$$

这里 $g(x,k) \ge 0$, $\sum_k g(x,k)=1$, 即对每个 input $x$, gate 输出一个 expert 上的 categorical distribution. 注意这跟 MoE 的区别: experts 是 frozen 的, gate 只是 reweight, 不动 expert 参数.

**关键约束** $\mathcal G_1$ 这个 space: gate 必须满足 **global normalization**
$$\mathcal G_1 = \{g : Z_g = \sum_{x \in \mathcal X_0} \sum_{k=1}^p g(x,k)\widehat\pi_k(x) = 1\}$$

其中 $\mathcal X_0 = \bigcup_k \mathrm{supp}(\widehat{\mathsf p}_k)$ 是有限 support. 直觉上, 这个 constraint 保证 $\pi_g$ 本身是 valid probability distribution——不是 just per-input 归一, 是整条 distribution 的 mass 加起来等于 1. 这听起来 trivivial, 但当 $g$ 是 input-dependent 时, 每个点的 mixture 重心移动会让全局 mass 偏离 1, 所以必须强制约束. 这条 constraint 是后面所有 algorithmic 麻烦的来源.

---

## 2. Minimax 形式: 对抗 mixture

最终目标:
$$\min_{g\in\mathcal G_1}\;\max_{\lambda\in\Delta([1,p])}\;\mathsf D_{\mathrm{KL}}(\widehat{\mathsf p}_\lambda \,\|\, \pi_g)$$

**直觉**: 一个 adversary $\lambda$ 想找最难拟合的 mixture, gate $g$ 想在这个 worst case 下 still 做到 small divergence. 这相当于 distributionally robust 优化, 参考 [^dro].

paper 指出一个微妙点: **必须用 KL divergence 而不是 cross-entropy**. 在 fixed $\lambda$ 下, minimize CE 等价于 minimize KL (差一个 entropy 常数 $H(\widehat{\mathsf p}_\lambda)$). 但在 robust 设定下, adversary 可以改变 $\lambda$, 从而改变 $H(\widehat{\mathsf p}_\lambda)$, 这个 entropy 项不再 constant, 所以 CE 和 KL 不再等价. 用 KL 才能保证模型真正逼近分布, 而不是只是 cover support.

---

## 3. Linearization Trick: 把 convex-convex 改成 convex-concave

原始 payoff $L(\lambda,g) = \mathsf D_{\mathrm{KL}}(\widehat{\mathsf p}_\lambda \| \pi_g)$ 对 $\lambda$ 是 convex (KL 第一个 argument 上 convex, $\widehat{\mathsf p}_\lambda$ 在 $\lambda$ 上 linear), 对 $g$ 也是 convex. 这就麻烦了——convex-convex 的 saddle point 不存在一般结构, standard no-regret dynamics 不直接适用.

paper 的 trick: 因为 $\max_\lambda L(\lambda, g)$ 在 simplex 上一定取在 vertex (convex function 在 polytope 上 max 在 extreme point), 所以
$$\max_{\lambda\in\Delta} L(\lambda,g) = \max_{k} \mathsf D_{\mathrm{KL}}(\widehat{\mathsf p}_k \| \pi_g)$$

于是引入 **linearized payoff**:
$$\widetilde L(\lambda, g) = \sum_{k=1}^p \lambda_k\, \mathsf D_{\mathrm{KL}}(\widehat{\mathsf p}_k \,\|\, \pi_g)$$

这个 $\widetilde L$ **linear in $\lambda$, convex in $g$**, 是 convex-concave (严格说 linear-concave) game, no-regret dynamics 可以用.

**关键的 identity**:
$$\mathsf D_{\mathrm{KL}}(\widehat{\mathsf p}_\lambda \| \pi_g) = \widetilde L(\lambda,g) - \mathsf D_{\mathrm{JSD}}^\lambda(\widehat{\mathsf p}_1,\dots,\widehat{\mathsf p}_p)$$

其中 **Jensen-Shannon Divergence**
$$\mathsf D_{\mathrm{JSD}}^\lambda(\{\widehat{\mathsf p}_k\}) = \sum_{k=1}^p \lambda_k \mathsf D_{\mathrm{KL}}(\widehat{\mathsf p}_k \| \widehat{\mathsf p}_\lambda)$$

这是 mixture 的 "diversity"——每个 source 离 mixture center 的平均 KL. 它非负, 上界是 Shannon entropy $H(\lambda)=-\sum_k\lambda_k\log\lambda_k$.

这个 identity 是后面所有 magic 的 source: 对 monolithic 模型, JSD 是干扰; 对 modular gate, JSD 是 gain. 见 §4.

---

## 4. Existence: Kakutani Fixed-Point

Theorem 3 证明 robust gate 存在. 用 **Kakutani fixed-point theorem** [^kakutani]: 需要对应(correspondence)是 closed graph + non-empty compact convex value.

定义 best-response correspondence
$$T(\lambda, g) = (\Lambda^*(g), G^*(\lambda))$$
其中
$$\Lambda^*(g) = \arg\max_{\lambda'} \widetilde L(\lambda', g), \quad G^*(\lambda) = \arg\min_{g'} \widetilde L(\lambda, g')$$

- $\widetilde L$ 在 $\lambda$ 上 linear → $\Lambda^*(g)$ 是 simplex 的一个 face, convex ✓
- $\widetilde L$ 在 $g$ 上 convex (KL 第一项 convex combination) → $G^*(\lambda)$ convex ✓
- Berge's Maximum Theorem → closed graph ✓
- domain $\Delta \times \mathcal G_1$ 紧致 (Lemma 1 证明 $\mathcal G_1$ non-empty, compact, convex via Tychonoff theorem)

于是 Kakutani 保证 fixed point $(\lambda^*, g^*)$, 即 saddle point.

**Bound 的推导**(Theorem 3 的核心):
$$\mathsf D_{\mathrm{KL}}(\widehat{\mathsf p}_\lambda \| \pi_{g^*}) \le \log\Big[\sum_{k=1}^p e^{\epsilon_k}\Big] - H_\sigma^{\lambda^*}(K|X) - \mathsf D_{\mathrm{JSD}}^\lambda(\{\widehat{\mathsf p}_k\})$$

三项含义:
- $\log\sum_k e^{\epsilon_k}$ = **Capacity Cost** (LogSumExp). 也就是 static gate 的 fundamental lower bound (Theorem 16, Appendix B 证: disjoint support 下任何 static gate $\mathbf w$ worst-case $\ge \log\sum_k e^{\epsilon_k}$).
- $H_\sigma^{\lambda^*}(K|X) = \sum_k \lambda_k^* \mathbb E_{x\sim\widehat{\mathsf p}_k}\big[-\log\frac{\sigma_k\widehat\pi_k(x)}{\pi_\sigma(x)}\big]$ = **Overlap Gain**. $\sigma_k = e^{\epsilon_k}/\sum_j e^{\epsilon_j}$ 是 softmax over errors, $\pi_\sigma = \sum_k \sigma_k\widehat\pi_k$ 是 "Robust Constant Gate". 当 experts overlap (同样 x 多个 expert 都给高 prob), 这一项大.
- $\mathsf D_{\mathrm{JSD}}^\lambda$ = **Diversity Gain / Separability Gain**.

证明技巧: 用 $\pi_\sigma$ 作为 witness gate (不是 $g^*$ 本身), 因为 $g^*$ 没有显式 closed form (Lemma 15 证明 optimal $g^*$ 是一个 clipped 几何对象, 依赖 pointwise convex hull). Appendix C 讨论 tightness vs interpretability trade-off, 作者承认 bound 不是最紧的, 但能 reveal 结构 phase transition.

### 三个 limiting regime (build intuition)

**Case 1: Disjoint experts (specialization)**. $\mathrm{supp}(\widehat{\mathsf p}_k) \cap \mathrm{supp}(\widehat{\mathsf p}_j) = \emptyset$.
- $H_\sigma^{\lambda^*}(K|X) = 0$ (no overlap)
- $\mathsf D_{\mathrm{JSD}}^\lambda = H(\lambda)$ (diversity 等于 entropy)
- bound = $\log\sum e^{\epsilon_k} - H(\lambda) \approx \epsilon_{\max} + \log p - H(\lambda)$
- 当 $\lambda$ uniform, $H(\lambda) \approx \log p$, capacity cost 完全 cancel. → **最难 mixture (high entropy) 上 modular 反而没有 capacity penalty**.

**Case 2: Identical experts (redundancy)**. 所有 $\widehat\pi_k = \widehat\pi$, $\epsilon_k = \epsilon$.
- $\mathsf D_{\mathrm{JSD}} = 0$, $H(\sigma) = \log p$
- bound = $(\epsilon + \log p) - \log p = \epsilon$
- Modular 退化为单个 expert.

**Case 3: Overlap with heterogeneous errors (ensemble)**. 完全 overlap, 但 $\epsilon_k$ 不同.
- bound = $\log Z - (\log Z - \sum_k \sigma_k \epsilon_k) = \sum_k \sigma_k \epsilon_k$
- 即 **weighted average of errors with softmax weights**. Modular 变成 ensemble.

这三种 case 一起展示了一个 phase diagram: diversity 在 monolithic 是 curse, 在 modular 是 blessing.

---

## 5. The JSD Gap: 模块化为什么打败 monolithic (这是 paper 最漂亮的部分)

**Jensen-Shannon Decomposition Identity** (Theorem 6 proof):
$$\underbrace{\sum_{k=1}^p \lambda_k \mathsf D_{\mathrm{KL}}(\widehat{\mathsf p}_k \| \pi)}_{\text{Average Task Risk}} = \underbrace{\mathsf D_{\mathrm{KL}}(\widehat{\mathsf p}_\lambda \| \pi)}_{\text{Mixture Fit}} + \underbrace{\mathsf D_{\mathrm{JSD}}^\lambda(\widehat{\mathsf p}_1,\dots,\widehat{\mathsf p}_p)}_{\text{Interference}}$$

**对任何模型 $\pi$ 都成立**. 这是个 algebraic identity, 把 average task risk 拆成 mixture fit + JSD. 

含义: **monolithic model 即使在 infinite capacity 下完美 fit mixture** ($\mathsf D_{\mathrm{KL}}(\widehat{\mathsf p}_\lambda\|\pi)\to 0$), **average task risk 也 $\ge \mathsf D_{\mathrm{JSD}}^\lambda$**. JSD 是个 fundamental floor.

Theorem 6 给出 monolithic retrained model $\widehat\pi_\lambda$ 的 lower bound:
$$\mathsf D_{\mathrm{KL}}(\widehat{\mathsf p}_\lambda \| \widehat\pi_\lambda) \ge \sum_k \lambda_k \epsilon_k - \mathsf D_{\mathrm{JSD}}^\lambda$$

而 modular 的 upper bound (Theorem 3):
$$\mathsf D_{\mathrm{KL}}(\widehat{\mathsf p}_\lambda \| \pi_{g^*}) \le \log\sum_k e^{\epsilon_k} - H_\sigma^{\lambda^*}(K|X) - \mathsf D_{\mathrm{JSD}}^\lambda$$

**同一个 $\mathsf D_{\mathrm{JSD}}^\lambda$ 项, monolithic 是 lower bound 里的减号(modular 越大越 hurt), modular 是 upper bound 里的减号(modular 越大越 help)**. 这就是 §4.5.3 说的 "Symmetric Divergence Effect". 

Phase transition 条件: 当 $\max_k \epsilon_k < \mathsf D_{\mathrm{JSD}}^\lambda$ 时, monolithic 必败. 直觉——任务越 diverse 越矛盾, monolithic 越搞不定, modular 越赚.

### Convex Settings 的 Safety (Theorem 7)

担心: 那 experts 兼容时 modular 是不是浪费? Theorem 7 证明如果 $\Pi$ 是 linear model family (e.g. exponential families), 那么:
$$\pi^*(\widehat{\mathsf p}_\lambda) = \sum_{k=1}^p \lambda_k \pi_k$$
即 **retrained optimal 等于 gated mixture of optimal experts**. 用 Csiszár 的 Pythagorean equality [^csiszar] 直接证. 所以 modular 是 "safe prior": convex setting 不亏, non-convex/conflict setting 大赚.

---

## 6. Generalization Bound: 为什么 sample complexity 跟 gate 走, 不跟 expert 走

Theorem 9 是 paper 另一个核心 selling point. 标准 retrain 的 generalization gap $\sim \mathfrak R_m(\Pi)$, 其中 $\Pi$ 是整个 LLM 类——参数量 $10^9$ 到 $10^{11}$, Rademacher complexity 巨大. 

modular 的 gap:
$$\mathbb E_{x\sim\mathsf p_\lambda}[-\log\pi_g(x)] \le \mathbb E_{x\sim\widehat{\mathsf p}_\lambda}[-\log\pi_g(x)] + \sum_k \lambda_k\Big[2\sqrt 2\, C_\Pi e^M \mathfrak R_{m_k}^k(\mathcal G_1) + M\sqrt{\frac{\log(p/\delta)}{2m_k}}\Big]$$

变量解释:
- $M$ = expert log-likelihood bound, $|\log\widehat\pi_k(x)|\le M$ (Assumption 8). LLM softmax 输出, 大 vocab 自然 bounded.
- $\mathfrak R_{m_k}^k(\mathcal G_1)$ = gate class 在 distribution $k$ 上的 Rademacher complexity. 因为 gate 是 lightweight (paper 里 ~290K params), 这一项很小.
- $C_\Pi = \sup_x \|\widehat{\boldsymbol\pi}(x)\|_2$, 其中 $\widehat{\boldsymbol\pi}(x) = (\widehat\pi_1(x),\dots,\widehat\pi_p(x))$ 是 **Expert Coincidence Norm**, 取值 $[0,\sqrt p]$. 是 modular 的 "condition number":
  - disjoint experts (best case): $C_\Pi \approx 1$
  - LLM typical (small token probs): $C_\Pi \ll 1$
  - fully redundant (worst case): $C_\Pi = \sqrt p$

证明用的是 vector-valued Rademacher contraction [^maurer] [^cortes2016]: loss function $\Psi_i(\mathbf u) = -\log(\mathbf u\cdot\widehat{\boldsymbol\pi}(x_i))$ 在 $\mathbf u$ 上 $\ell_2$-Lipschitz, 常数 $\|\nabla\Psi_i\|_2 = \|\widehat{\boldsymbol\pi}(x_i)\|_2 / |\mathbf u\cdot\widehat{\boldsymbol\pi}| \le C_\Pi e^M$.

**直觉**: modular 把 generalization 难度从 expert (billion params) 转嫁到 gate (small MLP) + $C_\Pi$ (a condition number for expert geometry). 当 experts 设计得 orthogonal, $C_\Pi \approx 1$, 整个 gap 几乎只由 gate 决定, cost of robustness 几乎 free.

---

## 7. Optimization: Stochastic Primal-Dual

理论上 Algorithm 1 是 EG (λ-player) + OGD (g-player), 收敛率 $O(\sqrt{\log p / T})$ (Theorem 11). 但 $\prod_{\mathcal G_1}$ 投影到 global normalization constraint 需要 solve 一个 quadratic program over 整个 $\mathcal X_0$, 对 sequence model 不可行.

**Primal-Dual relaxation**: 把 $Z_g = 1$ 用 Lagrange multiplier $\mu$ 软化:
$$\mathcal L(\theta, \lambda, \mu) = \underbrace{\sum_k \lambda_k \mathcal L_{\mathrm{NLL}}(k,\theta)}_{\text{Robust NLL}} + \underbrace{\mu(Z_{g_\theta}-1)}_{\text{Penalty}}$$

三个 player:
- $\lambda$: Exponentiated Gradient (adversary, upweight underperforming expert)
- $\mu$: Dual Ascent (enforce global normalization, $Z_g>1$ 时增 $\mu$ 惩罚)
- $\theta$: AdamW minimize Lagrangian

Theorem 13 证 $O(1/\sqrt T)$ convergence 同时 constraint violation $O(1/\sqrt T)$.

**Practical estimation of $Z_g$** (这是关键工程 trick): 直接 $\sum_{x\in\mathcal X_0}\pi_g(x)$ 不可能. 用 importance sampling, proposal $q(x) = \frac 1 p \sum_k \widehat\pi_k(x)$:
$$\widehat Z = \frac 1{|B|}\sum_{x\in B}\frac{\pi_g(x)}{q(x)}$$

batch $B$ 是从 $\frac 1p \sum_k \widehat{\mathsf p}_k$ 采样, 假设 $\widehat\pi_k \approx \widehat{\mathsf p}_k$, 那 $B$ 也 $\approx q(x)$ 的 sample. 这样零额外 inference cost, 复用 forward pass logits. 加 EMA 平滑 $\overline Z = \alpha\overline Z + (1-\alpha)\widehat Z$.

**Log-space**: $\log\pi_g(x) = \mathrm{LogSumExp}_k(\log g(x,k) + \log\widehat\pi_k(x))$, 防 underflow (长 sequence prob $10^{-100}$).

**Prior knowledge on $\lambda$**: 如果知道 $\lambda\in\Lambda\subset\Delta$, 直接 projection onto $\Lambda$ (KL projection), Theorem 4 + 5 给 quantitative improvement $V_\Delta^* - V_\Lambda^* \le L\cdot d_H(\Lambda,\Delta)$, $d_H$ Hausdorff distance, $L$ Lipschitz constant (explicit form给出).

---

## 8. Sampling & Inference: 非 causal gate 怎么用

这是 paper 的一个 subtle 痛点: optimal $g^*(x,\cdot)$ 是 **non-causal**——它读完整 sequence $x$ 才决定 mixture weights. 所以 standard autoregressive generation 不能直接用.

**Method 1: SIR (Sampling-Importance-Resampling)** (Algorithm 3)
1. Sample $N$ candidates from proposal $q(x) = \frac 1 p \sum_k \widehat\pi_k(x)$: 随机 pick expert, 让它自回归 generate.
2. 计算 importance weight $w(x) = \pi_{g^*}(x)/q(x)$. 这步可以, 因为 $x$ 已经完整.
3. Resample, $P_i \propto w_i$.

Asymptotic exact (support $\mathrm{supp}(\pi_{g^*})\subseteq \mathrm{supp}(q)$ 严格成立). Cost $O(Np)$.

**Method 2: Rejection Sampling** (Algorithm 4)
利用 $\pi_{g^*}(x) = \sum_k g^*(x,k)\widehat\pi_k(x) \le \sum_k \widehat\pi_k(x) = p\cdot q(x)$, envelope 常数 $M=p$. Acceptance $1/p$. $p$ 小时 cheap, $p$ 大时 wasteful.

**Method 3: Monolithic Distillation** (§6.3)
train 一个 causal student $\pi_{\mathrm{causal}}$ mimic $\pi_{g^*}$, standard next-token CE on rejection-sampled data. 但这样 expert 信息丢失, 失去 modularity.

**Method 4: Structural Distillation** (§7, 推荐)
只 distill gate $g^*$ 到一个 lightweight **Causal Router** $\gamma_\phi$:
$$\pi_\gamma(x) = \prod_{t=1}^T \sum_{k=1}^p \gamma_\phi(x_{<t}, k)\,\widehat\pi_k(x_t|x_{<t})$$

**关键区别**: teacher $\pi_{g^*}$ 是 "Mixture of Products" (latent expert sampled once per sequence), student $\pi_\gamma$ 是 "Product of Mixtures" (expert 可以每个 token 切换). 这个 mismatch 在 Theorem 18 证 = 0!

具体: 
- Teacher marginal conditional: $\pi_{g^*}(x_t|h) = \sum_k P_{\pi_{g^*}}(k|h)\widehat\pi_k(x_t|h)$
- Bayes-optimal router: $\gamma_k^*(h) = P_{\pi_{g^*}}(K=k|h)$, 即 posterior of expert given history.
- 设 $\gamma_\phi = \gamma^*$, student 完全 match teacher 的 step-wise conditional.

Theorem 18 (Chain rule decomposition):
$$\mathsf D_{\mathrm{KL}}(\pi_{g^*}\|\pi_\gamma) = \sum_{t=1}^T \mathbb E_{x_{<t}\sim\pi_{g^*}}\big[\mathsf D_{\mathrm{KL}}(\gamma^*(\cdot|x_{<t}) \| \gamma_\phi(\cdot|x_{<t}))\big]$$

没有 irreducible structural error. 这是 modular 相对 monolithic distillation 的最大 advantage: expert 还是 frozen, 升级一个 expert 整个 system 自动升级, 不重训 student.

**Cached-Logit Algorithm 5**: 先用 rejection sampling 生成 $M$ 条 sequence, 每条 cache 所有 expert 的 per-token prob 向量 $\mathbf P_t = [\widehat\pi_1(x_t|x_{<t}),\dots,\widehat\pi_p(x_t|x_{<t})]$. 然后只训 router $\phi$, minimize $-\log(\gamma_\phi(x_{<t})\cdot\mathbf P_t)$. expert inference 是一次性成本, router 训练极快.

---

## 9. 实验: synthetic + real-world

### Synthetic (§8.1)

vocab 100, domain A: $x_{t+1} = (x_t+1)\bmod 100$, domain B: $x_{t+1} = (x_t-1)\bmod 100$. 严格矛盾——对每个 token 梯度直接对冲.

模型: 1 层 transformer, 2 heads, $d_{\mathrm{ff}}=32$, expert $d=8$, gate $d=6$, smaller retrain $d=16$, larger retrain $d=20$. Robust Gate total ≈ smaller retrain, ≈ 2/3 larger retrain.

**结果** (Figure 5):
- Fixed 0.5-mixture retrain 在 high entropy 区域 ($\lambda\in[0.3,0.7]$) 严重下挫
- **Oracle** (cheat, 知道 test $\lambda$ 后重训) 也下挫! **即使 Oracle 用 larger 容量**, 在 $\lambda\approx 0.5$ 处 modular 仍然 beat larger Oracle.
- $\lambda$ 接近 0 或 1, 单一 domain, Oracle 自然反超——这是 Theorem 6 的预言: $\mathsf D_{\mathrm{JSD}}^\lambda \to 0$ 时 modular 优势消失.

Figure 6, 7: 50-50 / 25-75 混合 A 中的 B, 显示 robust gate 仍占优 for 大部分 $\lambda$, 只在 extreme skew 输给 Oracle.

### Algorithm Stability (§8.2)
$\lambda_t \to [0.5, 0.5]$, $\mu_t$ warm-up 后稳定, $Z_g\to 1$. 没 GAN 那种 oscillation, 因为 inner max over $\lambda$ 在 simplex 上是 convex problem.

### Structural Distillation (§8.3, Figure 8)
Causal Router (cyan) 几乎重合 Robust Gate (blue), 显著 beat Larger Fixed (green). 验证 Theorem 18: 结构 mismatch 几乎 0, 只剩 router approximation error.

### Real-world (§8.4, Table 1)

3 个 HF dataset: wikimedia/wikipedia, bigcode/the-stack-smol, fineweb-edu. 3 experts (2-layer, 2-head, $d=256$, ~6.5M each), gate 2-layer 2-head $d=256$, ~290K params. Combined ~20M, 对比 retrain 19.8M (3-layer 4-head $d=184$).

| Model | NLL/token (diff seed) | NLL/token (diff data) |
|---|---|---|
| Wiki Expert | 5.122 ± 0.005 | 5.118 ± 0.011 |
| Code Expert | 4.722 ± 0.045 | 5.267 ± 0.788 |
| FineWeb Expert | 5.623 ± 0.004 | 5.623 ± 0.006 |
| Retrained | 5.133 ± 0.010 | 5.306 ± 0.257 |
| **Gate** | **4.994 ± 0.013** | **5.087 ± 0.141** |

**Gate 在两种 setting 都赢 Retrained**. 注意 Code Expert 在 diff data 上 std 巨大 (0.788), 说明 code domain 高 variance. Gate 把这种 variance 吸收掉.

Table 2: 7 种 test $\lambda$ 组合, Gate 在 7/7 上更 robust, 最大优势在 code-heavy mixture ($1/2, 1/6, 1/3$ 等).

---

## 10. 这篇 paper 在大图里的位置: 联想

### 跟 MoE 的关系 [^moe] [^switch]
传统 MoE 是 end-to-end 联合训 router + experts, 用 auxiliary load-balancing loss 防止 expert collapse. 这篇是 **decoupled**: experts frozen, 只训 gate. 优势: catastrophic forgetting 避免, expert 升级 modular, 不需要重训 whole stack. 劣势: expert 必须事先有, 不能 emergent specialization.

### 跟 Model Soups / Task Arithmetic 关系 [^soups] [^taskarith]
Model Soups 平均 fine-tuned 权重, 是 static + 在 weight space. 这里是 dynamic + 在 output space. 但 paper 没讨论 weight-space merging 的几何, 一个有意思的方向是把 logits-level mixture 和 weight-level merging 的 modularity bound 统一起来.

### 跟 DRO 的关系 [^dro] [^dro2]
经典 DRO 在 group shift 上 robust (Sagawa et al. 2020). 这里是 mixture shift, 对应 source distribution 的任意凸组合. CLIP 类 robust training 也类似 spirit. 这篇的 JSD 分解给了一个新的信息论诊断工具: robustness 的 phase transition 可以从 JSD 直接 read off.

### 跟 DoReMi 关系 [^doremi]
DoReMi 用 adversary 调 data mixture weight 加速预训练, 优化 average perplexity. 这里 adversary 调 mixture 是找 worst case, 不是加速训练, 是 robustness guarantee. 两者其实可以串: 用 least favorable $\lambda^*$ (§4.4) 当 DoReMi 的 weighting, 训练一个 "hardest mixture" 上的 monolithic model. paper 在 §4.4 明确提了这点.

### 跟 CALM / StitchLLM 关系 [^calm] [^stitchllm]
CALM 用 cross-attention 拼 base + augmenting LLM. StitchLLM 在 layer level stitch blocks. 都是 architectural composition 但都没理论保证. 这篇在 output 层做 mixture, 提供了 KL bound 和 generalization bound. 但 architectural composition 可能 expressive 更强 (representation level merging), 这篇 paper 没覆盖.

### 跟 Modular Marketplaces [^marketplace]
Bhawalkar et al. 2025 在 game-theoretic 视角看 module marketplace, 关心 price equilibrium. 这篇给 statistical equilibrium, 即合在一起的 model quality 保证. 二者互补——经济激励 + 统计 robustness 是一个完整 modular ecosystem 的两面.

### 跟 Learning to Defer [^defer] [^defer2]
Cortes/DeSalvo/Mohri 一系列 learning to defer / reject 工作, Mao 最近扩到 multi-class + multi-expert. 跟这篇 conceptual 相似 (router 在 expert 间选), 但 defer 是 hard selection (single expert), 这篇是 soft mixture. defer 关心 cost, 这篇纯统计. 但 cross-fertilization 显然: defer 的 regret bound 思路也许能给 gated mixture 一个新视角.

### 跟 Pythagorean / Information Projection [^csiszar] [^csiszar2]
Theorem 7 用 Csiszár 的 I-projection Pythagorean equality 证 convex family 下 modular = retrain. 这是 information geometry 的经典工具. 一个延伸: 对 non-convex family, mismatch 是否可由 Bregman divergence 量化? 这跟 reinforcement learning 里的 policy mirror descent 也连得上.

### 一个更深的联想: posterior of expert given history
Theorem 17 给出 Bayes-optimal router = $P_{\pi_{g^*}}(K=k|x_{<t})$, 即 expert 的 posterior. 这隐含一个 EM-style interpretation: gate 是 prior over expert, router 是 posterior. 这就回到 latent variable model——mixture of experts 是 sequence-level latent, 每个 sequence 一个 expert. VAE / EM 的工具完全可用, 比如 ELBO decomposition, wake-sleep 训练. 这是一个未开发的 angle.

### 跟 in-context learning 的连接
GPT 类 model 的 in-context learning 本质上是 Bayesian model selection over latent tasks [^icl-bayesian]. 这里 gate 在做 explicit 的 expert selection. 一个 unifying 视角: ICL 是 implicit mixture, modular gate 是 explicit mixture. 后者可解释性更强, robustness 也可证.

### $\mathcal G_1$ 的几何和 Wasserstein gradient flow
$\mathcal G_1$ 是 product of simplices 加一个 affine constraint. Wasserstein gradient flow on this constraint manifold 可能给出更精细的 optimization geometry. Schrödinger bridge / entropy-regularized OT 跟 mixture 也有联系. 也许可以证明 $g^*$ 是某个 OT 问题的解.

### Practical concern: large $p$ 的 reject cost
$p$ 大时 rejection sampling acceptance $1/p$ 太小, SIR 也需要 $N$ candidates. paper 介绍 distillation 解决. 但 distillation 的 cost 是一次性 + 离线. 一个更 elegant 的方向: 用 **particle filtering** 或 **sequential Monte Carlo** along the token axis, 而不是 sequence-level rejection, 可能更 efficient. 这又跟 Theorem 18 的 step-wise posterior formulation特别 match.

### Negative result 值得注意
Theorem 6 的 lower bound $\ge \sum_k\lambda_k\epsilon_k - \mathsf D_{\mathrm{JSD}}^\lambda$ 在 $\mathsf D_{\mathrm{JSD}}^\lambda > \sum_k\lambda_k\epsilon_k$ 时 lower bound 变负, 信息泄露. 实际 lower bound 是 $\max(0, \sum\lambda_k\epsilon_k - \mathsf D_{\mathrm{JSD}}^\lambda)$. paper 没显式 clamp, 但语义上要. 这是一个细节 caveat.

### 一个我不太 buy 的地方
Theorem 3 的 bound 用 $\pi_\sigma$ 当 witness, 作者 Appendix C 承认不是 tightest, 但 claim 是 "interpretable". 我部分 buy: Case 1-3 的 phase diagram 确实 intuition-rich. 但在 intermediate regime (partial overlap + diverse errors), bound 怎么 trade off 没有显式表达. 一个可能改进: 用 witness $g^*$ 自身的 fixed-point characterization 配合 data-dependent 的 $\widehat\pi_k(x)$ 显式 form, 即使复杂, 至少在 special cases (e.g. Gaussians) 可解.

### Comparison with retrained model 的 subtlety
实验比较时 (§8.1) Robust Gate total params ~ smaller retrain (24% 是 gate 可训, 76% frozen expert). 严格说 "fair capacity comparison" 仍 tricky: frozen experts 的 hypothesis space 是 convex hull of experts, 是 retrained model 的 subset. 所以 robust gate 的胜利不是 capacity, 是 **inductive bias**——把 diversity 当 separability 而不是 interference. 这正是 Theorem 6 的核心 message.

### 跟 KL Divergence vs Cross Entropy 的 robust distinction 在 RLHF 里也有
paper §3 remark 指出 robust 设定下 KL ≠ CE. 这个区分在 RLHF 里也出现: DPO 用 $\log \pi(y|x)/\pi_{\mathrm{ref}}(y|x)$, 是 log ratio, 跟 KL 一样 sensitive to entropy shift. 一个 robust-RLHF-mixture 的 formulation 也许借鉴这里.

---

## 11. 给 Andrej 的直觉总结

这篇 paper 把 "modular training of LLM" 这个看似工程的问题 formalize 成 minimax game on KL divergence with global normalization. 几个关键 insight 我 wish 你带走:

1. **JSD 在 modular vs monolithic 上是 mirror image**: monolithic 把它当干扰下界, modular 把它当 separability 上界减项. 这是 Theorem 6 的核心, 也是 paper 最 beautiful 的部分.
2. **Sample complexity 跟 gate 走**: 290K gate params 决定 generalization, 不是 6.5M experts. $C_\Pi$ 是 expert 几何的 condition number, disjoint 时 $\approx 1$, modular 几乎 free lunch.
3. **Global normalization $Z_g=1$ 是必要 evil**: 不加约束 $\pi_g$ 不是 valid distribution, 加了之后 algorithm 需要做 primal-dual. Importance sampling 估计 $\widehat Z$ 用 training batch 是关键工程 trick.
4. **Non-causal gate → causal router 通过 structural distillation, no structural mismatch**: 因为 teacher 的 marginal conditional 就是 mixture with posterior weights, student 直接用 posterior 当 router 即可. 这是 Theorem 18 的妙处.
5. **Least-favorable $\lambda^*$ 是副产品**, 可以拿来当 DoReMi 的 weighting 训 monolithic, 给 "hardest mixture" 上的 baseline.
6. **Modular 是 safe prior**: convex family 下 modular = retrain (Theorem 7), 没损失; non-convex/conflict 下 modular >> retrain. 这是个 "free option".

如果让我推一步, 我会把 latent variable / EM 视角展开: $g^*$ 是 prior, $\gamma^*$ 是 posterior, training loop 看起来像 wake-sleep. 然后 step-wise 的 SMC inference 替代 sequence-level rejection. 这些是这框架没探索的, 但理论上 well-posed.

---

### Web Links for Reference

[^msa]: Mohri, Hoffman, Zhang - Multiple-Source Adaptation Theory and Algorithms: https://www.cs.nyu.edu/~mohri/pub/msa.pdf  
[^kakutani]: Kakutani Fixed-Point Theorem (Wikipedia): https://en.wikipedia.org/wiki/Kakutani_fixed-point_theorem  
[^dro]: Sagawa et al. - Distributionally Robust Neural Networks: https://arxiv.org/abs/1911.08731  
[^dro2]: DRO general overview (Duchi): https://web.stanford.edu/~jduchi/refs/DuchiNamkoogSriperumbudur12.pdf  
[^moe]: Jacobs et al. - Adaptive Mixtures of Local Experts (original MoE): https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf  
[^switch]: Fedus et al. - Switch Transformers: https://arxiv.org/abs/2101.03961  
[^soups]: Wortsman et al. - Model Soups: https://arxiv.org/abs/2203.05482  
[^taskarith]: Ilharco et al. - Task Arithmetic: https://arxiv.org/abs/2212.04089  
[^doremi]: Xie et al. - DoReMi: https://arxiv.org/abs/2305.10429  
[^calm]: Bansal et al. - CALM: https://arxiv.org/abs/2310.13548  
[^stitchllm]: Hu et al. - StitchLLM: https://aclanthology.org/2025.acl-long.1198/  
[^marketplace]: Bhawalkar et al. - Modular Marketplaces: https://arxiv.org/abs/2502.20346  
[^defer]: Cortes, DeSalvo, Mohri - Learning with Rejection: https://link.springer.com/article/10.1007/s10472-024-10018-5  
[^defer2]: Mao, Mohri, Zhong - Multi-Expert Deferral: https://arxiv.org/abs/2408.01168  
[^csiszar]: Csiszár - I-divergence geometry: https://projecteuclid.org/journals/annals-of-probability/annals-of-probability/volume-3/issue-1/I-divergence-geometry-of-probability-distributions-and-minimization/10.1214/aop/1176996764.full  
[^csiszar2]: Csiszár & Matus - Information Projections Revisited: https://ieeexplore.ieee.org/document/1205385  
[^maurer]: Maurer - Vector Contraction Inequality for Rademacher Complexity: https://dl.acm.org/doi/10.1145/2946680  
[^cortes2016]: Cortes, Kuznetsov, Mohri, Yang - Structured Prediction with Factor Graph Complexity: https://papers.nips.cc/paper/2016/hash/7404d7da5d015c9e6f4d35f7a4fb8a93-Abstract.html  
[^icl-bayesian]: Xie et al. / Muller et al. - Bayesian view of ICL: https://arxiv.org/abs/2111.02080  
[^chinchilla]: Hoffmann et al. - Chinchilla (compute-optimal LLM): https://arxiv.org/abs/2203.15556  
[^sir]: Rubin - Sampling Importance Resampling: https://en.wikipedia.org/wiki/Importance_sampling
