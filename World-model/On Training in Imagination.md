---
source_pdf: On Training in Imagination.pdf
paper_sha256: d5090958c236509a195949febba05be65e00413a54c92dffe616f812f1e02d00
processed_at: '2026-08-05T23:42:59-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说这篇 paper

## 一句话版本

Dreamer、MuZero 这帮人在 imagination 里训 policy —— 也就是 dynamics model 和 reward model 学完之后，policy 完全在 model 里 rollout，不再碰 real environment。这套东西 work 得挺好，但**为什么 work、什么时候会崩、预算怎么花**一直没人说清楚。这篇 paper 就是来填这个坑的。

---

## 1. 核心问题：你以为是一个 model，其实是两个

大家平时把 world model 当一个东西看待 —— dynamics 和 reward 共享 encoder、一起训、一起用。但这篇 paper 说：**stop，这俩是两个完全不同的 beast**。

为什么？因为它们犯错的"代价"完全不一样：

- **Reward model 错了**：每一步最多错 $\varepsilon_{\text{rew}}$，trajectory 加起来最多错 $\frac{\varepsilon_{\text{rew}}}{1-\gamma}$。**它不传播**。
- **Dynamics model 错了**：这一步错一点点，下一步因为 policy 还要根据错的 state 选 action，错上加错，**compound**。bound 里的系数是 $\frac{\gamma L_r(1+L_\pi)}{(1-\gamma)(1-\gamma L_f(1+L_\pi))}$，分母那个 $1-\gamma L_f(1+L_\pi)$ 就是 compounding 的元凶 —— 只要它接近 1，一点 dynamics error 就会被放大成灾难。

这就是 Lemma 1（https://arxiv.org/abs/1806.01265 的 Asadi 2018b 只考虑 dynamics，paper 把 reward 也加进来）：

$$|J(\pi, \mathcal{M}) - J(\pi, \hat{\mathcal{M}})| \leq \frac{1}{1-\gamma}\varepsilon_{\text{rew}} + \frac{\gamma L_r(1+L_\pi)}{(1-\gamma)(1-\gamma L_f(1+L_\pi))}\varepsilon_{\text{dyn}}$$

**人话翻译**：return 误差 = reward 误差的简单累加 + dynamics 误差的"滚雪球式"放大。两个误差**separable**，可以独立控制。

---

## 2. 为什么 latent space 有用？因为能降 Lipschitz

JEPA（https://openreview.net/pdf?id=BZ5a1r-kVsf）、VICReg、Dreamer 这些都在 latent space 做预测，但"为什么 latent 比像素好"一直有点 hand-wavy。这篇 paper 给了一个**硬指标**：**latent representation 的目标是降 $L_f, L_r, L_\pi$ 这三个 Lipschitz constant**。

Corollary 1 说：dynamics-error 那一项的系数 $\frac{\gamma L_r(1+L_\pi)}{(1-\gamma)(1-\gamma L_f(1+L_\pi))}$ 对 $L_f, L_r, L_\pi$ 都是单调的。所以**降任何一个都会 tighten bound**。

**人话**：在 pixel space 里，相近的两个 image 在 next-state prediction 上可能差得很远（$L_f$ 巨大）；在好的 latent space 里，相近的两个 latent state 对应的 next latent 也接近（$L_f$ 小）。这就直接让 dynamics error 的 compounding 变慢。

**Caveat**（paper 自己说的）：降 $L_f$ 可能会让 $\varepsilon_{\text{dyn}}$ 升（capacity 限制）。所以**不是无脑小 Lipschitz 就好**，要 net effect 是 tighten 才行。这点挺重要，不能误读。

---

## 3. Temporal straightening：原来这个 loss 也有理论根据

Wang et al. 2026（https://arxiv.org/abs/2603.12231）提了个 temporal straightening loss，让 latent trajectory 尽量"直"——相邻 velocity 向量 $v(z_t) = f(z_t) - z_t$ 和 $v(z_{t+1})$ 尽量平行。当时看像个审美偏好。

Proposition 1 证明：如果 $v$ 是 $\varepsilon$-Lipschitz 且 $\varepsilon < 1$，那么 curvature loss 满足
$$\mathcal{L}_{\text{curv}}(t) \leq \frac{\varepsilon^2}{2(1-\varepsilon)}$$

**人话**：velocity map 越平滑，curvature loss 上界越小。所以 temporal straightening 不是品味问题，它等价于"让 $v$ 的 Lipschitz 小"，进而让 $L_f$ 小，进而 tighten Lemma 1 的 bound。**representation design 的审美被连上了 return error 的硬指标**。

---

## 4. 预算怎么分？看 power-law 谁快

这才是 paper 最 useful 的部分。假设你有钱 $B$，要决定买多少 dynamics transition $(s, a, s')$ 和多少 reward annotation $(s, a, r)$。两边 error 都按 power-law 衰减：

$$\varepsilon_{\text{dyn}}(N_{\text{dyn}}) = A_d N_{\text{dyn}}^{-\alpha}, \qquad \varepsilon_{\text{rew}}(N_{\text{rew}}) = A_r N_{\text{rew}}^{-\beta}$$

Theorem 1 给出最优比例：
$$\frac{N_{\text{dyn}}^*}{N_{\text{rew}}^*} = \frac{\alpha}{\beta} \cdot \frac{\gamma L_r(1+L_\pi)}{1 - \gamma L_f(1+L_\pi)} \cdot \frac{c_{\text{rew}}}{c_{\text{dyn}}} \cdot \frac{\varepsilon_{\text{dyn}}^*}{\varepsilon_{\text{rew}}^*}$$

四个 factor，人话讲：

1. **$\alpha/\beta$**：谁学得快谁就少买。reward 学得快就少买 reward sample。
2. **$\frac{\gamma L_r(1+L_\pi)}{1-\gamma L_f(1+L_\pi)}$**：dynamics error 在 bound 里权重大，就多买 dynamics sample。
3. **$c_{\text{rew}}/c_{\text{dyn}}$**：reward 标注贵（比如 RLHF 要人工）就少买 reward。
4. **$\varepsilon_{\text{dyn}}^*/\varepsilon_{\text{rew}}^*$**：这是 fixed-point，最后谁剩的 error 大谁就该多买。

**实验测出来**：在他们的 synthetic setup 里，$\alpha \approx 0.11, \beta \approx 0.96$。也就是 reward error 每 decade 数据掉 ~10×，dynamics error 每 decade 数据只掉 ~1.26×。**reward 学得快 9 倍**。

直觉解释：reward 是 scalar regression，dynamics 是 $d_s$-维 next-state regression，task 难度差 $d_s$ 倍。所以在中性 cost 假设下，**reward sample 应该少买**，把钱倒给 dynamics。

参考 Chinchilla（https://arxiv.org/abs/2203.15556）的 budget allocation 思路，只不过这里分的是两条 data stream，不是 params vs tokens。

---

## 5. Noisy reward 没事，biased reward 致命

REINFORCE 在 noisy reward 下，Theorem 2 说：

$$\mathbb{E}[\hat{g}] = g_H \quad \text{(unbiased!)}$$
$$\text{Var}[\hat{g}] \leq \text{Var}[\hat{g}]_{\eta \equiv 0} + \frac{\sigma_\eta^2 H W_H^2}{K(1-\gamma)^2}$$

**人话**：
- **Zero-mean noise** 只增加 variance，不偏 gradient。多 rollout 平均能压下去（$1/K$ 衰减）。
- **Bias**（Proposition 2）就不一样了：$\mathbb{E}[\tilde{g}] = g_H + \nabla_\theta B_H(\theta)$，多 rollout 完全消不掉 $\nabla_\theta B_H$ 那一项。

这跟 Gao et al. 2023（https://arxiv.org/abs/2210.10760）的 reward over-optimization 现象对应 —— RLHF 里 proxy reward model 被 optimize 到极致后 true reward 反而下降，这是 systematic bias，不是 noise。

---

## 6. 什么时候买便宜 noisy reward，什么时候买贵 precise reward

Corollary 2 给了个非常 clean 的判据：最小化 $\Phi(c) := c \cdot \sigma_\eta^2(c)$。

意思是：你买 $K = B/c$ 个 rollout，每个 rollout 花 $c$ 元得到 noise variance $\sigma_\eta^2(c)$。Total noise contribution 是 $\frac{\sigma_\eta^2(c)}{K} = \frac{c \sigma_\eta^2(c)}{B} = \frac{\Phi(c)}{B}$。所以 $\Phi(c)$ 越小越好。

三种情况：

**(a) Power-law fidelity** $\sigma_\eta^2(c) = A c^{-p}$：
- $p > 1$：花钱降 noise 比买更多 rollout 值，**买贵的**。
- $p < 1$：买更多 rollout 比降 noise 值，**买便宜的**。
- $p = 1$：临界，怎么都一样。

**(b) Bounded fidelity** $\sigma_\eta^2(c) = \sigma_0^2(1 - c/c_{\max})$：
- $\Phi(c) = \sigma_0^2 c(1-c/c_{\max})$ 是个抛物线。
- **两个极端最优**：要么全买最便宜的（多 rollout），要么买最贵的（noise-free）。中间反而最差。

**(c) Irreducible floor** $\sigma_\eta^2(c) = \sigma_{\text{floor}}^2 + A/c$：
- $\Phi(c) = \sigma_{\text{floor}}^2 c + A$ 单调递增。
- **永远买最便宜的**，因为再花钱也消不掉 floor。

**实操 takeaway**：先估 $\sigma_\eta^2(c)$ 的形状，再决定策略。在 RLHF 场景下（https://arxiv.org/abs/1706.03741），cost 就是 annotator 时间/质量，noise 是 annotator 不一致。这个 framework 可以直接套。

---

## 7. 实验：bound 实际松成什么样

Section 3.1 测 Lemma 1：525 个 config，bound 全部成立（$R \leq 1$），但 median 松散 65 倍，synthetic 上松 286 倍。这是 worst-case bound 的常态，跟 Munos 2003（https://dl.acm.org/doi/10.1145/304281.304286）在 approximate DP 里看到的一致。

Section 4.2 测 Theorem 1：用 global Lipschitz constants 当 multiplier，在 LQG 上 overshoot 真实 ratio **1968 倍**。但如果用 realized sensitivities（在 perturbed model 上 finite-difference 量 value function 敏感度），预测和实测在 factor of 3 内吻合。

**人话**：
- **Proportionality 部分有预测力** —— "ratio 跟 cost、exponent 怎么变" 是对的。
- **Multiplier 是 worst-case over-estimate** —— 全局 Lipschitz 太悲观。要在真实系统上算准需要 realized sensitivities，但那东西要 evaluate $V^\pi$ at perturbed models，scale 上 impractical。
- 所以 Theorem 1 给的是**理论指导方向**，不是即插即用公式。

---

## 8. 我的整体 take

这篇 paper 不 propose 新算法，但把 training in imagination 这个范式的关键 trade-off **拆得明明白白**。对我自己 build intuition 来说，几个抓手：

1. **Dynamics 和 reward 是两个东西**。Dreamer 那种 shared backbone 是工程方便，理论上它们 scaling、cost、error propagation 都不同。设计系统时应该分开想。

2. **Lipschitz 是 representation 的硬指标**。JEPA、spectral norm、temporal straightening 这些"软" motivation 有了"硬"的连接 —— 它们直接 control return bound 的 coefficient。这从"我觉得 latent space 好"变成"我能算给你看 latent 降低了多少 compounding factor"。

3. **Power-law exponents 是 first-class citizen**。要做 MBRL 系统设计，先 fit scaling law，再 allocate budget。这是把 LLM scaling culture 搬到 RL 的思路，Chinchilla for MBRL。

4. **Noise vs bias 是 binary**。Theorem 2 + Proposition 2 合起来说：noise 可以靠 averaging，bias 不行。判据是 $H W_H^2 / (1-\gamma)^2$ 的大小。

5. **Worst-case bound 的局限**。3 orders of magnitude 的 looseness 提醒我们：理论给方向，但不给精确数值。Lemma 1 是 sufficient condition 不是 tight characterization。

6. **没解决的**：stochastic dynamics、$\gamma L_f(1+L_\pi) \geq 1$ 的 regime、$L_f$ vs $\varepsilon_{\text{dyn}}$ 的 trade-off 定量化、power-law exponent 在不同 task 上的可预测性。

**一句话总结**：如果 Dreamer 是"在 imagination 里训 policy"的工程实践，这篇 paper 是它的"理论资产负债表" —— 把成本、风险、回报拆开列清楚，告诉你每个 dial 转一下会发生什么。

---

# On Training in Imagination 深度解读

## 1. Paper 全景：它在回答什么问题

这篇 paper 来自 Weizmann Institute、NYU、Columbia、Yann LeCun (NYU AMI Labs) 的合作，针对 **model-based RL** 里一类特定范式 —— "training in imagination"（Dreamer 3/4、MuZero 这一支）做了一套比较系统的理论分析。它没有 propose 新的算法，只是把这种范式中大家"凭直觉调"的两个东西 —— dynamics model 和 reward model —— 拆开看，回答四个问题：

- **Error attribution**: return gap 里到底 dynamics model error 贡献多少，reward model error 贡献多少？
- **Representation desiderata**: latent representation 该有什么性质才能让想象训练更准确？
- **Budget allocation**: 固定预算下，dynamics samples 和 reward samples 该按什么比例买？
- **Reward fidelity**: REINFORCE 在 noisy reward 下还能用吗？什么时候买"便宜但噪"的 reward label 比买"贵但准"的更划算？

核心 insight 用一句话总结：**dynamics error 和 reward error 在 return bound 里是 separable 的两个 term，而它们之间的权重比 + 不同的 power-law 速度 + 单位 cost，三者组合决定了 optimal sample ratio**。

参考链接：
- Dreamer 3 (Nature 2025): https://www.nature.com/articles/s41586-024-07540-2
- Dreamer 4 (Training agents inside scalable world models): https://arxiv.org/abs/2509.24527
- MuZero (Schrittwieser et al. 2020): https://www.nature.com/articles/s41586-020-03051-4
- Asadi et al. 2018b (Lipschitz continuity in MBRL): https://proceedings.mlr.press/v80/asadi18a.html

---

## 2. 背景设定：MDP 与 Lipschitz 假设

paper 用 deterministic MDP $\mathcal{M} = (\mathcal{S}, \mathcal{A}, f, r, \gamma)$，其中：
- $\mathcal{S} \subseteq \mathbb{R}^{d_s}$：state space
- $\mathcal{A} \subseteq \mathbb{R}^{d_a}$：action space
- $f: \mathcal{S} \times \mathcal{A} \to \mathcal{S}$：deterministic dynamics
- $r: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$：reward
- $\gamma \in [0, 1)$：discount factor

trajectory：$a_t = \pi(s_t),\ s_{t+1} = f(s_t, a_t)$，discounted return $J(\pi, \mathcal{M}) = \sum_{t=0}^\infty \gamma^t r(s_t, a_t)$。

学到的 model 记 $\hat{f}, \hat{r}$，得到 approximate MDP $\hat{\mathcal{M}} = (\mathcal{S}, \mathcal{A}, \hat{f}, \hat{r}, \gamma)$。

两个 worst-case error：
$$\varepsilon_{\text{dyn}} := \sup_{s,a} \|\hat{f}(s,a) - f(s,a)\|, \qquad \varepsilon_{\text{rew}} := \sup_{s,a} |\hat{r}(s,a) - r(s,a)|$$

三个 Lipschitz constants：
- $f$ 是 $L_f$-Lipschitz: $\|f(s,a) - f(s',a')\| \leq L_f(\|s-s'\| + \|a-a'\|)$
- $r$ 是 $L_r$-Lipschitz: $|r(s,a) - r(s',a')| \leq L_r(\|s-s'\| + \|a-a'\|)$
- $\pi$ 是 $L_\pi$-Lipschitz: $\|\pi(s) - \pi(s')\| \leq L_\pi\|s-s'\|$

注意 deterministic 假设不要求环境 memoryless，因为可以把历史 encode 进 state（用 RNN/LSTM），论文 Section 6 明确说了这点，引用了 Rumelhart et al. 1986、Elman 1990、Hochreiter & Schmidhuber 1997、Henaff et al. 2016。

---

## 3. Lemma 1：Simulation Error Decomposition

### 3.1 公式

$$\boxed{\ |J(\pi, \mathcal{M}) - J(\pi, \hat{\mathcal{M}})| \leq \underbrace{\frac{1}{1-\gamma}\varepsilon_{\text{rew}}}_{\text{reward-error term}} + \underbrace{\frac{\gamma L_r(1+L_\pi)}{(1-\gamma)(1-\gamma L_f(1+L_\pi))}\varepsilon_{\text{dyn}}}_{\text{dynamics-error term}}\ }$$

**前提条件**：$\gamma L_f (1+L_\pi) < 1$（contraction condition）。

### 3.2 变量与上标下标语义

- $\gamma$：discount factor，权重几何衰减速率
- $L_f$：dynamics map $f$ 的 Lipschitz constant，刻画 "input 变一点，next state 变多少"
- $L_r$：reward map $r$ 的 Lipschitz constant
- $L_\pi$：policy map $\pi$ 的 Lipschitz constant
- $\varepsilon_{\text{dyn}}$：dynamics model 的 worst-case $\ell_2$ error
- $\varepsilon_{\text{rew}}$：reward model 的 worst-case absolute error

### 3.3 直觉：两项各自的物理意义

**第一项 $\frac{1}{1-\gamma}\varepsilon_{\text{rew}}$**：reward error 不传播。每一步真实 reward $r$ 和预测 reward $\hat{r}$ 之间最多差 $\varepsilon_{\text{rew}}$，整个 trajectory 的几何衰减权重和是 $\frac{1}{1-\gamma}$。即使 dynamics 完全准确，光是 reward 估计错了，最多累加 $\frac{\varepsilon_{\text{rew}}}{1-\gamma}$ 的 return error。

**第二项 dynamics-error term**：dynamics error 会沿 trajectory 累积 + 放大。关键在于 $\frac{1}{1-\gamma L_f(1+L_\pi)}$ 这个 factor。它对应的是 error compounding 的几何级数和。

### 3.4 Proof 思路（Section B.1）

proof 的关键 induction：
- 令 $L_{\text{comp}} := L_f(1+L_\pi)$，这是 "state error 经过 dynamics + policy 反馈到下一 step state error" 的放大系数。
- 真实 trajectory $s_t$ 和 imagination trajectory $\hat{s}_t$（同一初始状态）：
  $$\|s_{t+1} - \hat{s}_{t+1}\| = \|f(s_t, a_t) - \hat{f}(\hat{s}_t, \hat{a}_t)\|$$
  $$\leq \|f(s_t, a_t) - f(\hat{s}_t, \hat{a}_t)\| + \|f(\hat{s}_t, \hat{a}_t) - \hat{f}(\hat{s}_t, \hat{a}_t)\|$$
  $$\leq L_f(\|s_t - \hat{s}_t\| + \|\pi(s_t) - \pi(\hat{s}_t)\|) + \varepsilon_{\text{dyn}}$$
  $$\leq L_f(1 + L_\pi)\|s_t - \hat{s}_t\| + \varepsilon_{\text{dyn}} = L_{\text{comp}}\|s_t - \hat{s}_t\| + \varepsilon_{\text{dyn}}$$
- 归纳得 $\|s_t - \hat{s}_t\| \leq \varepsilon_{\text{dyn}} \sum_{k=0}^{t-1} L_{\text{comp}}^k$。
- reward 误差分成两部分（telescoping trick）：
  $$r(s_t,a_t) - \hat{r}(\hat{s}_t, \hat{a}_t) = \underbrace{[r(s_t,a_t) - r(\hat{s}_t,\hat{a}_t)]}_{\text{dynamics-induced}} + \underbrace{[r(\hat{s}_t,\hat{a}_t) - \hat{r}(\hat{s}_t,\hat{a}_t)]}_{\text{reward-model error}}$$
  第一项用 $L_r(1+L_\pi)\|s_t - \hat{s}_t\| \leq L_r(1+L_\pi)\varepsilon_{\text{dyn}}\sum_k L_{\text{comp}}^k$ 控制，第二项直接用 $\varepsilon_{\text{rew}}$ 控制。
- 求和时把 $\gamma^t$ 和几何级数 $\sum_k L_{\text{comp}}^k$ 交换，得到两个封闭形式的 geometric sum。

### 3.5 与 Asadi et al. 2018b 的差别

Asadi 2018b 假设 access 到 ground-truth reward，所以 bound 里只有 dynamics error，没有 reward error 这一项。Lemma 1 把 reward 也当成 learned，于是在 reward-error term 上 separable 出来。这个 separability 是后续 budget allocation 的关键：**只有当两个 error 的系数独立可调时，"在两者之间分配预算"这个问题才 well-posed**。

参考：Kearns & Singh 2002 simulation lemma 原文 https://link.springer.com/article/10.1023/A:1015398526588

---

## 4. Corollary 1：Lipschitz Constants 控制 Bound 松紧

dynamics-error 的 coefficient $C_{\text{dyn}} := \frac{\gamma L_r(1+L_\pi)}{(1-\gamma)(1-\gamma L_f(1+L_\pi))}$ 对 $L_f, L_r, L_\pi$ 各自**单调非减**。

**直觉**：在固定 $\varepsilon_{\text{dyn}}, \varepsilon_{\text{rew}}$ 下，降低 $L_f, L_r, L_\pi$ 任意一个都会 tighten bound。

实操含义：
- **Latent representation 设计**：用 $z = \phi(s)$ 而不是 raw observation $s$ 作为 model input，可以让 Lipschitz constants 变小。这是因为 latent space 通常比 pixel space "更平滑"。
- **Lipschitz regularization**：spectral normalization (Miyato et al. 2018, https://arxiv.org/abs/1802.05957)、gradient penalty、weight clipping 等技术可以显式控制 Lipschitz constant。
- **JEPA (LeCun 2022)**：joint-embedding predictive architecture 在 latent space 做预测，本质上就是降低 $L_f$，对应 https://openreview.net/pdf?id=BZ5a1r-kVsf。

**重要 caveat**（paper 明确说了）：
1. 降低 $L_f$ 可能让 $\varepsilon_{\text{dyn}}$ 上升（capacity trade-off），只有当 $L_f$ 下降的 gain 超过 $\varepsilon_{\text{dyn}}$ 上升的 loss，bound 才真的 tighten。
2. $\gamma L_f(1+L_\pi) < 1$ 是必要前提，违反时 bound 失效（这个 regime 仍是 open problem）。

---

## 5. Proposition 1：连接 Temporal Straightening

Wang et al. 2026 (https://arxiv.org/abs/2603.12231) 提出一个 temporal straightening loss，鼓励 latent trajectory 接近"直线"。

### 5.1 定义

latent state $z_t = \phi(s_t)$，latent dynamics $z_{t+1} = f(z_t)$（注意：这里的 $f$ 是 latent-level，不是 observation-level，paper 在这里符号重载了）。

**Latent velocity map**: $v(z) := f(z) - z$，这是 latent state 一步的位移。

**Curvature loss**:
$$\mathcal{L}_{\text{curv}}(t) := 1 - \frac{v(z_t)^\top v(z_{t+1})}{\|v(z_t)\|\|v(z_{t+1})\|}$$

这就是 $1 - \cos\theta$，其中 $\theta$ 是相邻两个 velocity 向量之间的夹角。loss 越小说明 velocity 越接近平行 → latent 轨迹越直。

### 5.2 Proposition 1 公式

如果 $v$ 是 $\varepsilon$-Lipschitz 且 $0 < \varepsilon < 1$，那么：
$$\mathcal{L}_{\text{curv}}(t) \leq \frac{\varepsilon^2}{2(1-\varepsilon)}$$

### 5.3 Proof 关键不等式（Section B.2）

令 $a = v(z_t),\ b = v(z_{t+1})$。

1. Lipschitz 给出 $\|b - a\| \leq \varepsilon\|z_{t+1} - z_t\| = \varepsilon\|a\|$。
2. 三角不等式：$\|b\| \geq \|a\| - \|b-a\| \geq (1-\varepsilon)\|a\|$。
3. 关键代数恒等式：
   $$\|b - a\|^2 = (\|a\| - \|b\|)^2 + 2\|a\|\|b\|(1 - C)$$
   其中 $C = \frac{a^\top b}{\|a\|\|b\|}$ 是 cosine similarity。
4. 因为 $(\|a\| - \|b\|)^2 \geq 0$：
   $$1 - C \leq \frac{\|b-a\|^2}{2\|a\|\|b\|} \leq \frac{\varepsilon^2\|a\|^2}{2\|a\|\cdot(1-\varepsilon)\|a\|} = \frac{\varepsilon^2}{2(1-\varepsilon)}$$

### 5.4 直觉

这个 proposition 把两件事联系起来：
- **Temporal straightening 的"why"**：不是审美/几何上的偏好，而是因为 latent velocity map 的 Lipschitz 常数小 → 曲率小 → 长程预测更线性 → return error bound 更紧（通过 Corollary 1 的 $L_f$ 下降）。
- **Lipschitz of $v$ 而非 $f$**：注意这里 Lipschitz 是对 $v = f(z) - z$，不是对 $f$ 本身。这是因为 straightening 关心的是"velocity 的变化率"，而不是"state 的变化率"。

函数 $\frac{\varepsilon^2}{2(1-\varepsilon)}$ 在 $(0,1)$ 上递增，所以进一步降低 $\varepsilon$ 一定 tighten bound。这与 Corollary 1 的方向一致：latent space 设计应该让 velocity map "慢变化"。

---

## 6. Theorem 1：最优样本比例

### 6.1 设定

- 两种样本：dynamics transitions $(s, a, f(s,a))$ 和 reward annotations $(s, a, r(s,a))$。
- 每种样本的单位 cost：$c_{\text{dyn}}, c_{\text{rew}}$。
- 样本数：$N_{\text{dyn}}, N_{\text{rew}}$。
- 预算约束：$c_{\text{dyn}} N_{\text{dyn}} + c_{\text{rew}} N_{\text{rew}} = B$。
- Power-law scaling：
  $$\varepsilon_{\text{dyn}}(N_{\text{dyn}}) = A_d N_{\text{dyn}}^{-\alpha}, \qquad \varepsilon_{\text{rew}}(N_{\text{rew}}) = A_r N_{\text{rew}}^{-\beta}$$
  其中 $\alpha, \beta > 0$ 是 decay exponents，$A_d, A_r$ 是常数。

### 6.2 公式

$$\boxed{\ \frac{N_{\text{dyn}}^*}{N_{\text{rew}}^*} = \frac{\alpha}{\beta} \cdot \frac{\gamma L_r(1+L_\pi)}{1 - \gamma L_f(1+L_\pi)} \cdot \frac{c_{\text{rew}}}{c_{\text{dyn}}} \cdot \frac{\varepsilon_{\text{dyn}}^*}{\varepsilon_{\text{rew}}^*}\ }$$

### 6.3 四个 factor 各自的直觉

**Factor 1: $\alpha / \beta$**：power-law exponent 比。reward 学得越快（$\beta$ 大），相对越不需要那么多 reward 样本，应该多买 dynamics 样本。

**Factor 2: $\frac{\gamma L_r(1+L_\pi)}{1-\gamma L_f(1+L_\pi)}$**：dynamics-error 在 return bound 里的 coefficient（不含 $\frac{1}{1-\gamma}$ 因为分子分母都约掉了，看 Section B.3 proof）。这个 coefficient 越大，dynamics error 对 return 影响越大，应该多买 dynamics 样本来降低 $\varepsilon_{\text{dyn}}$。注意这里 $1-\gamma L_f(1+L_\pi)$ 在分母，contract 越接近 1 这个 factor 越大，符合 error compounding 越严重的直觉。

**Factor 3: $c_{\text{rew}} / c_{\text{dyn}}$**：reward 比 dynamics 越贵，越应该多买 dynamics 样本。这是 straightforward 的"贵的东西少买"。

**Factor 4: $\varepsilon_{\text{dyn}}^* / \varepsilon_{\text{rew}}^*$**：最优 error 比。这是 self-referential 的，因为 $\varepsilon^*$ 自身依赖于 $N^*$。所以这是个 fixed-point equation。

### 6.4 Proof 关键（Section B.3）

用 Lagrangian：
$$\Lambda = C_d N_{\text{dyn}}^{-\alpha} + C_r N_{\text{rew}}^{-\beta} + \lambda(c_{\text{dyn}} N_{\text{dyn}} + c_{\text{rew}} N_{\text{rew}} - B)$$
其中 $C_d, C_r$ 是把 Lemma 1 的 coefficient 和 power-law 的常数 merged 起来：
$$C_d = \frac{\gamma L_r(1+L_\pi) A_d}{(1-\gamma)(1-\gamma L_f(1+L_\pi))}, \quad C_r = \frac{A_r}{1-\gamma}$$

KKT 条件：$\alpha C_d N_{\text{dyn}}^{-\alpha-1} = \lambda c_{\text{dyn}}$，$\beta C_r N_{\text{rew}}^{-\beta-1} = \lambda c_{\text{rew}}$。

两式相除并整理得：
$$\frac{N_{\text{dyn}}}{N_{\text{rew}}} = \frac{\alpha}{\beta}\cdot \frac{c_{\text{rew}}}{c_{\text{dyn}}} \cdot \frac{C_d N_{\text{dyn}}^{-\alpha}}{C_r N_{\text{rew}}^{-\beta}}$$

把 $C_d N_{\text{dyn}}^{*\,-\alpha}$ 和 $C_r N_{\text{rew}}^{*\,-\beta}$ 用 Lemma 1 的 coefficient $\times \varepsilon^*$ 代入，得到 closed form。

凸性保证 KKT 解是唯一 minimizer。

### 6.5 为什么这是 fixed-point

把 (3) 代入 Theorem 1：
$$\frac{N_{\text{dyn}}^*}{N_{\text{rew}}^*} = \text{multiplier} \cdot \frac{A_d (N_{\text{dyn}}^*)^{-\alpha}}{A_r (N_{\text{rew}}^*)^{-\beta}}$$
左边是 $\frac{N_{\text{dyn}}^*}{N_{\text{rew}}^*}$，右边也有 $\frac{(N_{\text{dyn}}^*)^{-\alpha}}{(N_{\text{rew}}^*)^{-\beta}}$。设 $r := N_{\text{dyn}}^*/N_{\text{rew}}^*$：
$$r = \text{multiplier} \cdot \frac{A_d}{A_r} r^{-\alpha} \cdot (N_{\text{rew}}^*)^{\alpha-\beta}$$

这个方程的解需要和预算约束 $c_{\text{dyn}} r N_{\text{rew}}^* + c_{\text{rew}} N_{\text{rew}}^* = B$ 联立。

如果 $\alpha = \beta$，事情简化：$r^{\alpha+1} = \text{multiplier} \cdot \frac{A_d}{A_r} \cdot (N_{\text{rew}}^*)^0$，仍然依赖于 $N_{\text{rew}}^*$，但 ratio 可以独立解出。

### 6.6 规划 horizon 的影响

$\gamma$ 减小 → $\frac{\gamma L_r(1+L_\pi)}{1-\gamma L_f(1+L_\pi)}$ 减小 → dynamics multiplier 减小 → ratio 减小 → 应该多买 reward。

直觉：short horizon 下 dynamics error 没那么多步去 compound，所以不需要那么精确的 dynamics model。这与 control 工程里 "短 horizon MPC 可以容许粗糙 model" 的经验吻合。

参考 Hoffmann et al. 2022 Chinchilla: https://arxiv.org/abs/2203.15556 （类似 budget allocation 思路，但分 model params 和 tokens）；Kaplan et al. 2020: https://arxiv.org/abs/2001.08361

---

## 7. Section 3.1 实验：Lemma 1 Bound 的 Calibration

### 7.1 实验设计

定义 ratio $R := \text{LHS}/\text{RHS}$，其中 LHS 是真实 return gap $|J(\pi, \mathcal{M}) - J(\pi, \hat{\mathcal{M}})|$，RHS 是 Lemma 1 的 bound 右侧。$R \leq 1$ 即 bound 成立，$R$ 越小 bound 越 loose。

两个 benchmark：
- **Synthetic**: globally Lipschitz $f, r$（构造的），$n = 150$ configs。
- **LQG (Linear-Quadratic-Gaussian)**: 二次 reward，只在 bounded domain 内 Lipschitz，$n = 375$ configs。

总共 $n = 525$ configs。

### 7.2 数据

| 指标 | Synthetic | LQG | Pooled |
|---|---|---|---|
| Median $R$ | 0.0035 | 0.034 | 0.015 |
| Max $R$ | 0.9995 | 0.999 | — |
| 松散倍数 (median) | 286× | 29× | 65× |

每个 config 都 $R \leq 1$，bound 总是成立。但 median 上松散约 65 倍，这是 worst-case bound 的典型表现（参考 Munos 2003 https://dl.acm.org/doi/10.1145/304281.304286，Kakade & Langford 2002 https://dl.acm.org/doi/10.1145/780601.780677）。

LQG 上 max $R = 0.999$，离 1 很近，说明在 quadratic value + linear dynamics 这种 "boundary case" 上 bound 几乎 tight。

---

## 8. Section 4.1 实验：Power-law Scaling 验证

### 8.1 实验设计

- Teacher: 两个 2-layer ReLU MLP（dynamics teacher 和 reward teacher），frozen 随机初始化，weights $\sim \mathcal{N}(0,1)$。
- Dimensions: $d_s = 12, d_a = 4, d_h = 64$，episode length $T = 500$。
- Dynamics teacher: $s_{t+1} = \tanh(f_{\text{dyn}}([s_t, a_t]))$，tanh 把 state 限制在 $[-1, 1]^{d_s}$。
- Reward teacher: $r_t = f_{\text{rew}}([s_t, a_t]) / \sqrt{d_h}$，$1/\sqrt{d_h}$ scaling 保证不同宽度下 reward magnitude 可比。
- Student: 同架构 2-layer ReLU MLP，Adam lr=$10^{-3}$，batch 256，200 epochs。
- Anchors $N \in \{2k, 5k, 10k, 20k, 50k, 100k, 200k\}$，每个 anchor 100 seeds。
- Held-out validation: 5000 transitions 独立采样。

### 8.2 拟合结果

$$\varepsilon_{\text{dyn}}(N_{\text{dyn}}) = 0.34 \, N_{\text{dyn}}^{-0.11}, \quad R^2 = 0.954$$
$$\varepsilon_{\text{rew}}(N_{\text{rew}}) = 90.4 \, N_{\text{rew}}^{-0.96}, \quad R^2 = 0.997$$

Bootstrap 95% CI: $\alpha \in [0.09, 0.13]$，$\beta \in [0.93, 0.99]$。

### 8.3 关键观察

$$\frac{\beta}{\alpha} \approx \frac{0.96}{0.11} \approx 9$$

**reward error per decade of data 比 dynamics error 快约 9 倍**。

paper 解释：因为 $\hat{r}$ 预测一个 scalar，$\hat{f}$ 预测一个 $d_s = 12$ 维的 next state。前者是 simpler learning problem，scaling exponent 大。

这个 9× 是 Theorem 1 里 $\alpha/\beta \approx 1/9$ 的 empirical 依据，意味着 (在其他 factor 中性下) reward samples 不需要买太多。

### 8.4 与 LLM scaling laws 的对比

- Kaplan 2020 (https://arxiv.org/abs/2001.08361): LLM loss $\sim N^{-\alpha}$ 典型 $\alpha \approx 0.076$ per token。
- Hoffmann 2022 Chinchilla: $\alpha \approx 0.34$ in compute-optimal regime。
- 这里 dynamics $\alpha \approx 0.11$，reward $\beta \approx 0.96$，差距很大，因为这里是 supervised regression on fixed teacher，不是自回归 generation。

---

## 9. Section 4.2 实验：Theorem 1 预测的 Sample Ratio

### 9.1 实验设计

测试 Equation (4) 的两个 factor 各自：
1. **Proportionality**: $\frac{N_{\text{dyn}}^*}{N_{\text{rew}}^*} \propto \frac{\varepsilon_{\text{dyn}}^*}{\varepsilon_{\text{rew}}^*}$ 是否成立？
2. **Multiplier 松散性**: 用 global Lipschitz constants vs realized sensitivities，差多少？

定义 log-ratio residual: $\ell := \log(N_{\text{dyn}}^*/N_{\text{rew}}^*) - \log(\varepsilon_{\text{dyn}}^*/\varepsilon_{\text{rew}}^*)$。$|\ell| \leq \log 3$ 表示预测和实测 ratio 在 factor of 3 内一致。

### 9.2 Realized Sensitivities

定义：
$$S_f(s,a) := \frac{|V^\pi(s; f + h\Delta f, r) - V^\pi(s; f, r)|}{h\|\Delta f\|}$$
$$S_r(s,a) := \frac{|V^\pi(s; f, r + h\Delta r) - V^\pi(s; f, r)|}{h|\Delta r|}$$

当 $h \to 0$，这逼近 directional derivatives $\partial_f V^\pi$ 和 $\partial_r V^\pi$。这是 value function 对 model perturbation 的实际敏感度，比 global Lipschitz constants 紧得多。

### 9.3 四组配置

| Group | $n$ | Value function | Lipschitz constants |
|---|---|---|---|
| Linear | 30 | linear $V^\pi$ | analytical |
| tanh | 9 | $\tanh$-value | $L_f = L_r = \lambda = 1$ |
| sin | 9 | sin-value | $L_f = L_r = \lambda = 1$ |
| Quadratic sup-norm control | 9 | quadratic-value | 用 sup-norm over-estimate vs realized |

LQG: 30 configs, $\gamma = 0.8$, contraction $\gamma L_f(1+L_\pi) \in [0.224, 0.228]$。

### 9.4 结果

| Group | Median $|\ell|$ | 备注 |
|---|---|---|
| Linear | 0 | 代数 trivially 成立 |
| tanh/sin | 0.054 | 预测与实测在 factor of 1.05 内一致 |
| Sup-norm control | 0.684 | sup-norm overestimate 导致 ~9.25× 松散 |
| LQG (global Lipschitz) | 7.585 | ~1968× 松散 |

Figure 4 显示 LQG 上 30/30 个 config 都 overshoot，one-sided sign test $p = 0.5^{30} \approx 9.31\times 10^{-10}$，几乎不可能 chance。

### 9.5 直觉

- **Theorem 1 的 proportionality 部分是真的有预测能力**：用 realized sensitivities 替代 global Lipschitz，predicted ratio 与 realized ratio 在 factor of 3 内吻合。
- **Global Lipschitz constants 是 worst-case over-estimate**：在 LQG 这种光滑的 LQR 问题上，realized 敏感度比 sup-norm Lipschitz 小 3 orders of magnitude。这与 approximate DP 文献里 simulation lemma 的松散性一致 (Munos 2003)。
- **实操困难**：要算 realized sensitivities 需要evaluate $V^\pi$ at perturbed models，在大 scale 上 impractical。所以 Theorem 1 给的是理论指导（"ratio 怎么随 cost 和 exponent 变"），不是直接计算公式。

---

## 10. Theorem 2：REINFORCE 在 Noisy Rewards 下

### 10.1 设定

- Finite horizon $H$，discount $\gamma \in [0,1)$。
- $J_H(\pi, \mathcal{M}) := \mathbb{E}[\sum_{t=0}^{H-1} \gamma^t r(s_t, a_t)]$。
- $G_t := \sum_{t'=t}^{H-1} \gamma^{t'-t} r(s_{t',a_{t'}})$（从 $t$ 往后的 discounted return）。
- 噪声观测：$\hat{r}_t = r(s_t, a_t) + \eta_t$，其中 $\eta_t$ i.i.d.，$\mathbb{E}[\eta_t] = 0$，$\text{Var}[\eta_t] = \sigma_\eta^2$，且 $\eta_t$ 独立于 history $(s_0, a_0, \ldots, s_t, a_t)$。
- $\hat{G}_t := \sum_{t'=t}^{H-1} \gamma^{t'-t} \hat{r}_{t'}$。
- Single-trajectory REINFORCE: $\hat{g}^{(1)} := \sum_{t=0}^{H-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \hat{G}_t$。
- $K$ trajectory 平均：$\hat{g} := \frac{1}{K}\sum_{k=1}^K \hat{g}^{(k)}$。
- $W_H^2 := \mathbb{E}[\max_{0 \leq t \leq H-1} \|\nabla_\theta \log \pi_\theta(a_t|s_t)\|^2] < \infty$。
- $\text{Var}$ for vector-valued estimators: $\text{Var}[z] := \sum_i \text{Var}[z_i]$。

### 10.2 公式

$$\boxed{\ \mathbb{E}[\hat{g}] = g_H, \qquad \text{Var}[\hat{g}] \leq \text{Var}[\hat{g}]_{\eta \equiv 0} + \frac{\sigma_\eta^2 H W_H^2}{K(1-\gamma)^2}\ }$$

### 10.3 直觉：两个结论

**Unbiased**: zero-mean reward noise 不改变 gradient estimator 的期望。这是因为 $\eta_t$ 的期望是 0，REINFORCE estimator 对 reward 是线性的，所以 noise 在期望里被消掉。

**Variance inflation**: noise 引入额外 variance，按 $1/K$ 衰减（多 rollout 平均能压）。额外 variance 的形式 $\frac{\sigma_\eta^2 H W_H^2}{K(1-\gamma)^2}$ 各项含义：
- $\sigma_\eta^2$：noise variance，越大越糟。
- $H$：horizon 长，更多步累积噪声。
- $W_H^2$：score function $\nabla_\theta \log \pi$ 的最大二阶矩，policy 越敏感 noise 影响越大。
- $1/(1-\gamma)^2$：discount 累积系数的平方，因为 $G_t$ 是 $\sum \gamma^{t'-t} r_{t'}$。
- $1/K$：trajectory 越多 variance 越小。

### 10.4 Proof 关键（Section B.6）

定义 $N_t := \sum_{t' \geq t} \gamma^{t'-t} \eta_{t'}$，则 $\hat{G}_t = G_t + N_t$，于是
$$\hat{g}^{(1)} = g^{(1)} + \delta^{(1)}, \quad \delta^{(1)} := \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) N_t$$

**Unbiased**: 因为 $\eta_{t'}$ 独立于 history 且 $\mathbb{E}[\eta_{t'}] = 0$，
$$\mathbb{E}[N_t | \tau] = \sum_{t' \geq t} \gamma^{t'-t} \mathbb{E}[\eta_{t'} | \tau] = 0$$
所以 $\mathbb{E}[\delta^{(1)} | \tau] = 0$，进而 $\mathbb{E}[\hat{g}^{(1)} | \tau] = g^{(1)}$。

**Variance**: 用 $w_t := \nabla_\theta \log \pi_\theta(a_t|s_t)$，把 $\delta^{(1)}$ 重写：
$$\delta^{(1)} = \sum_t w_t \sum_{t' \geq t} \gamma^{t'-t} \eta_{t'} = \sum_{t'} \eta_{t'} \underbrace{\sum_{t \leq t'} \gamma^{t'-t} w_t}_{=: v_{t'}}$$

由 $\eta_{t'}$ 之间 i.i.d. 且互不相关：
$$\text{Var}[\delta^{(1)} | \tau] = \sigma_\eta^2 \sum_{t'} \|v_{t'}\|^2$$

每项 $\|v_{t'}\| \leq \sum_{t \leq t'} \gamma^{t'-t} \|w_t\| \leq \frac{1}{1-\gamma}\max_t \|w_t\|$，所以
$$\text{Var}[\delta^{(1)} | \tau] \leq \sigma_\eta^2 H \cdot \frac{1}{(1-\gamma)^2} \max_t \|w_t\|^2$$

取期望 + law of total variance 得到 bound。

### 10.5 与现有文献的关系

- **Williams 1992 REINFORCE 原文**: https://link.springer.com/article/10.1007/BF00992696
- **Sutton et al. 1999 policy gradient theorem**: https://papers.nips.cc/paper/1999/hash/464d828b85b0bed98e8020cda2a01adf-Abstract.html
- **Zhang et al. 2021 adversarial corruption** (https://proceedings.mlr.press/v139/zhang21az.html): 处理 adversarial corruption，$\varepsilon$-fraction episodes 任意修改。这个 paper 处理的是 zero-mean i.i.d. noise，更"良性"。
- **Gao et al. 2023 reward over-optimization** (https://arxiv.org/abs/2210.10760): Goodhart-style proxy misspecification，是 systematic bias，paper 在 Proposition 2 里专门讨论。

---

## 11. Corollary 2：最优 Fidelity

### 11.1 设定

- 每个 rollout 的 reward annotation cost: $c > 0$。
- 噪声 variance 是 cost 的函数: $\sigma_\eta^2(c) := \text{Var}[\eta_t | \text{per-rollout cost} = c]$，假设 measurable。
- Budget $B$，可买 $K = B/c$ 个 rollouts。

### 11.2 公式

定义 $\Phi(c) := c \cdot \sigma_\eta^2(c)$。把 $K = B/c$ 代入 Theorem 2 的 variance bound：
$$\frac{\sigma_\eta^2(c) H W_H^2}{K(1-\gamma)^2} = \frac{\Phi(c) H W_H^2}{B(1-\gamma)^2}$$

所以最小化 variance bound 等价于最小化 $\Phi(c)$ over $c > 0$。

### 11.3 三种 Regime 的 $\Phi(c)$ 形状

**(a) Power-law fidelity**: $\sigma_\eta^2(c) = A c^{-p}$
- $\Phi(c) = A c^{1-p}$
- $p > 1$: $\Phi$ 严格递减，应该买最贵的高 fidelity annotation。
- $p < 1$: $\Phi$ 严格递增，应该买最便宜的 annotation，用更多 rollout 压 variance。
- $p = 1$: $\Phi$ 恒等于 $A$，cost 不影响 variance bound。
- **临界点 $p=1$**: 这是 "1 倍 cost 增加 = 1 倍 variance 减小" 的临界。

**(b) Bounded fidelity**: $\sigma_\eta^2(c) = \sigma_0^2(1 - c/c_{\max})$ on $(0, c_{\max}]$
- $\Phi(c) = \sigma_0^2 c(1 - c/c_{\max})$
- 这是开口向下的 parabola，在 $c = c_{\max}/2$ 处取最大。
- $c \to 0$ 和 $c = c_{\max}$ 都让 $\Phi \to 0$，是 minimizer。
- **直觉**: 极端 cost 区间（全买便宜的 + 多 rollout，或者全买贵的 + 完美 annotation）都最优；中间区间反而最差。

**(c) Irreducible noise floor**: $\sigma_\eta^2(c) = \sigma_{\text{floor}}^2 + A/c$
- $\Phi(c) = \sigma_{\text{floor}}^2 c + A$
- 严格递增，最优 $c \to 0$。
- **直觉**: 当钱无法消除 noise floor 时，多花钱只是浪费；不如靠 $1/K$ 把 variance 压下去。

### 11.4 实操含义

这三种 regime 给了 practitioner 一个判断框架：
1. 先估计 $\sigma_\eta^2(c)$ 的形状（即 noise 随 cost 怎么衰减）。
2. 看是哪种 regime。
3. 决定买 high fidelity 还是 buy more rollouts。

特别地，在 RLHF 场景下 (Christiano et al. 2017, https://arxiv.org/abs/1706.03741)，annotation cost 反映人工标注时间或质量，noise 反映 annotator 不一致性，这个 framework 可以指导"是用多个 annotator 平均（降 noise，增 cost）还是用单个 annotator 多标几个 rollout"。

---

## 12. Proposition 2：Biased Rewards

### 12.1 设定

- Reward bias function $b: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$。
- $\tilde{r}(s,a) := r(s,a) + b(s,a)$。
- $\tilde{\mathcal{M}} := (\mathcal{S}, \mathcal{A}, f, \tilde{r}, \gamma)$（dynamics 真实，reward biased）。
- $B_H(\theta) := \mathbb{E}[\sum_{t=0}^{H-1} \gamma^t b(s_t, a_t)]$，是 bias 在 $\pi_\theta$ induced trajectory 下的 discounted 累积。

### 12.2 公式

$$\mathbb{E}[\tilde{g}] = \nabla_\theta J_H(\pi_\theta, \tilde{\mathcal{M}}) = g_H + \nabla_\theta B_H(\theta)$$
$$\mathbb{E}[\|\tilde{g} - g_H\|^2] = \frac{1}{K}\text{Var}[\tilde{g}^{(1)}] + \|\nabla_\theta B_H(\theta)\|^2$$

### 12.3 直觉

- $\nabla_\theta B_H(\theta)$ 是 bias 在 gradient 上的投影，trajectory averaging 消不掉。
- 与 Theorem 2 对比：zero-mean noise 引入 variance 但 unbiased，可以通过 $1/K$ 压；bias 引入 gradient bias，无法通过 averaging 消除。
- 第二个公式是 bias-variance decomposition: $\text{MSE} = \text{Var}/K + \text{Bias}^2$，bias term 独立于 $K$。

### 12.4 Proof 思路

REINFORCE estimator with biased reward $\tilde{r}$ 在 $\tilde{\mathcal{M}}$ 上就是标准 REINFORCE，所以 $\mathbb{E}[\tilde{g}] = \nabla_\theta J_H(\pi_\theta, \tilde{\mathcal{M}})$。

然后 $J_H(\pi, \tilde{\mathcal{M}}) = J_H(\pi, \mathcal{M}) + B_H(\theta)$ 由 $r$ 和 $\tilde{r}$ 的线性关系直接得到。

梯度：$\nabla_\theta J_H(\pi, \tilde{\mathcal{M}}) = \nabla_\theta J_H(\pi, \mathcal{M}) + \nabla_\theta B_H(\theta) = g_H + \nabla_\theta B_H(\theta)$。

MSE 分解用 $\|x+y\|^2 = \|x\|^2 + 2\langle x, y\rangle + \|y\|^2$ 配合 $\mathbb{E}[\tilde{g} - \mathbb{E}[\tilde{g}], \mu_b] = 0$。

### 12.5 与 Goodhart's Law 的联系

Gao et al. 2023 (https://arxiv.org/abs/2210.10760) 在 RLHF 里观察到 proxy reward model 优化到一定程度后，true reward 反而下降。这是 systematic bias 的典型表现，而非 zero-mean noise。Proposition 2 给了这个现象的理论语言：bias 引入 gradient bias，不能通过 averaging 解决，需要重新 align reward model。

---

## 13. 整篇 Paper 的 Conceptual Map

```
Training in Imagination
        │
        ├─ Error Attribution (Lemma 1)
        │       │
        │       ├─ reward error term: (1/(1-γ)) ε_rew
        │       └─ dynamics error term: γ L_r (1+L_π) / [(1-γ)(1-γ L_f(1+L_π))] ε_dyn
        │
        ├─ Representation Desiderata (Corollary 1)
        │       └─ minimize L_f, L_r, L_π → tighter bound
        │           │
        │           └─ temporal straightening (Prop 1)
        │                   └─ Lipschitz of v(z)=f(z)-z controls curvature loss
        │
        ├─ Budget Allocation (Theorem 1)
        │       │
        │       └─ N_dyn*/N_rew* = (α/β) × Lipschitz factor × cost ratio × ε_dyn*/ε_rew*
        │           │
        │           ├─ Power-law scaling (Sec 4.1): β/α ≈ 9, reward 学得快
        │           └─ Empirical validation (Sec 4.2): proportionality holds, multiplier loose
        │
        └─ Reward Fidelity (Theorem 2)
                │
                ├─ Zero-mean noise: unbiased + Var inflation O(1/K)
                │       │
                │       └─ Optimal fidelity (Corollary 2): minimize Φ(c) = c σ²(c)
                │               ├─ power-law: p>1 buy fidelity, p<1 buy rollouts
                │               ├─ bounded: extremes optimal
                │               └─ floor: always buy cheap
                │
                └─ Bias (Proposition 2): gradient bias ∇B_H, can't average out
```

---

## 14. 我自己的 Intuition & Takeaways

**1. Dynamics vs Reward 的非对称性**：在大多数想象训练实现里（Dreamer、MuZero），dynamics 和 reward 用同一个 encoder 共享 backbone，paper 暗示这是个 conceptual mistake —— 它们 scaling exponent 差 9 倍，sample efficiency 完全不同。理论上应该分别采样、分别训练、分别 allocate budget。

**2. Lipschitz constants 作为 representation design target**：JEPA、VICReg、spectral norm 这些技术的 motivation 之前比较"软"（稳定性、可学习性），Corollary 1 + Proposition 1 给了一个"硬"的目标 —— 它们直接 control 一个 return error bound 的 coefficient。这个 connection 让 representation learning 从"美学"变成"优化目标"。

**3. Power-law exponents 是 first-class citizen**：Theorem 1 把 scaling exponents 直接放进 optimal ratio 公式，意味着在做 RL system design 时，应该先做 scaling law experiments (类似 Chinchilla)，fit exponents，再决定 budget allocation。这是把 LLM scaling culture 引入 MBRL 的思路。

**4. Zero-mean noise vs Bias 的二元对立**：Theorem 2 和 Proposition 2 合起来说：noise 可以靠 averaging 解决，bias 不能。这听起来 obvious，但提供了 quantitative form —— 决定 noise 是否"安全"的判据是 $H W_H^2 / (1-\gamma)^2$，即 horizon、score function magnitude、discount 几何累积的组合。

**5. Worst-case bounds 的局限**：Section 4.2 的实验显示 global Lipschitz constants 给的 multiplier 在 LQG 上 overshoots 3 orders of magnitude。这意味着 Lemma 1 在实践中是 sufficient condition，而非 tight characteristic。要 prediction 准确需要 realized sensitivities，但 realized sensitivities 在 scale 上 impractical。这是 worst-case bound 在 high-dim RL 上一贯的尴尬局面。

**6. 没解决的问题**：
- Stochastic dynamics（论文明说留 future work）
- $\gamma L_f(1+L_\pi) \geq 1$ 的 regime（contraction 失败）
- $\varepsilon_{\text{dyn}}$ 和 $L_f$ 之间的 trade-off：降 $L_f$ 通常升 $\varepsilon_{\text{dyn}}$，bound 是 net effect 怎样？
- Power-law exponents $\alpha, \beta$ 自身依赖于 architecture 和 data distribution，怎么在不同 setup 下 predict？

**7. 与 Sutton 1990 Dyna 的传承**：Dyna 在 real environment 和 learned model 之间交替。Training in imagination 是更激进版 —— policy update 完全脱离 real environment。Lemma 1 的 bound 可以看作这种激进的 cost：return error被两个 model 的 quality 决定，要靠 Lemma 1 控制。Sutton 原文：https://www.sciencedirect.com/science/article/pii/B9781558601413500030

**8. Janner et al. 2019 MBPO 的中间道路**：MBPO 用 short imagined rollouts + real environment interaction，规避了 Lemma 1 中 dynamics error compounding 的 $\frac{1}{1-\gamma L_f(1+L_\pi)}$ blow-up（短 rollout 让这个 factor 不会太大）。论文的理论给 MBPO 这种设计提供了 justification —— horizon 短则 dynamics multiplier 小，dynamics 不需要那么精确。MBPO: https://papers.nips.cc/paper/2019/hash/5faf461eff95c7fe5269c0d75c027d6e-Abstract.html

---

## 15. 公式与符号速查表

| 符号 | 含义 |
|---|---|
| $\mathcal{M} = (\mathcal{S}, \mathcal{A}, f, r, \gamma)$ | True MDP |
| $\hat{\mathcal{M}}$ | Approximate MDP with $\hat{f}, \hat{r}$ |
| $J(\pi, \mathcal{M})$ | Discounted return of $\pi$ in $\mathcal{M}$ |
| $\varepsilon_{\text{dyn}}$ | $\sup_{s,a}\|\hat{f}(s,a) - f(s,a)\|$ |
| $\varepsilon_{\text{rew}}$ | $\sup_{s,a}\|\hat{r}(s,a) - r(s,a)\|$ |
| $L_f, L_r, L_\pi$ | Lipschitz constants of $f, r, \pi$ |
| $\gamma$ | Discount factor |
| $L_{\text{comp}} := L_f(1+L_\pi)$ | State error compounding rate per step |
| $N_{\text{dyn}}, N_{\text{rew}}$ | Number of dynamics / reward samples |
| $c_{\text{dyn}}, c_{\text{rew}}$ | Per-sample costs |
| $\alpha, \beta$ | Power-law exponents for $\varepsilon_{\text{dyn}}, \varepsilon_{\text{rew}}$ |
| $A_d, A_r$ | Power-law prefactors |
| $B$ | Total sample budget |
| $H$ | Finite horizon |
| $K$ | Number of rollouts (trajectories) |
| $\sigma_\eta^2$ | Reward noise variance |
| $W_H^2$ | $\mathbb{E}[\max_t\|\nabla_\theta \log \pi_\theta(a_t\|s_t)\|^2]$ |
| $v(z) := f(z) - z$ | Latent velocity map |
| $\mathcal{L}_{\text{curv}}(t)$ | Temporal straightening curvature loss |
| $B_H(\theta)$ | Discounted bias accumulation |
| $\Phi(c) := c\sigma_\eta^2(c)$ | Cost-noise product for fidelity optimization |
| $S_f, S_r$ | Realized value-function sensitivities |

---

## 16. 参考文献链接汇总

**核心背景**：
- Dreamer 3 (Hafner et al. 2025a): https://www.nature.com/articles/s41586-024-07540-2
- Dreamer 4 (Hafner et al. 2025b): https://arxiv.org/abs/2509.24527
- MuZero (Schrittwieser et al. 2020): https://www.nature.com/articles/s41586-020-03051-4
- Dreamer (Hafner et al. 2020): https://openreview.net/forum?id=S1lOTC4tDS
- Dyna (Sutton 1990): https://www.sciencedirect.com/science/article/pii/B9781558601413500030
- MBPO (Janner et al. 2019): https://papers.nips.cc/paper/2019/hash/5faf461eff95c7fe5269c0d75c027d6e-Abstract.html
- World Models (Ha & Schmidhuber 2018): https://arxiv.org/abs/1803.10122
- PILCO (Deisenroth & Rasmussen 2011): http://proceedings.mlr.press/v15/deisenroth11a.html

**Simulation Lemma 谱系**：
- Kearns & Singh 2002: https://link.springer.com/article/10.1023/A:1015398526588
- Asadi et al. 2018b: https://proceedings.mlr.press/v80/asadi18a.html
- Asadi et al. 2018a (Wasserstein=value-aware): https://arxiv.org/abs/1806.01265
- Munos 2003: https://dl.acm.org/doi/10.1145/304281.304286
- Kakade & Langford 2002: https://dl.acm.org/doi/10.1145/780601.780677
- Lobel & Parr 2024: https://openreview.net/forum?id=RcoIAfiM5g
- Farahmand et al. 2017 (value-aware loss): http://proceedings.mlr.press/v54/farahmand17a.html
- Talvitie 2018: http://proceedings.mlr.press/v80/talvitie18a.html

**Representation Learning & Lipschitz**：
- LeCun 2022 JEPA: https://openreview.net/pdf?id=BZ5a1r-kVsf
- Wang et al. 2026 (temporal straightening): https://arxiv.org/abs/2603.12231
- Miyato et al. 2018 (spectral norm): https://arxiv.org/abs/1802.05957
- Gouk et al. 2021 (Lipschitz regularization): https://link.springer.com/article/10.1007/s10994-021-06020-5
- Assran et al. 2023 (JEPA image instantiation): https://openaccess.thecvf.com/content/CVPR2023/papers/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.pdf

**Scaling Laws**：
- Kaplan et al. 2020: https://arxiv.org/abs/2001.08361
- Hoffmann et al. 2022 (Chinchilla): https://arxiv.org/abs/2203.15556
- Hilton et al. 2023 (single-agent RL scaling): https://arxiv.org/abs/2301.13442
- Pearce et al. 2025 (world model pre-training): https://arxiv.org/abs/2505.xxxxx (在 proceedings.mlr.press 上)

**Policy Gradient & Reward Modeling**：
- Williams 1992 REINFORCE: https://link.springer.com/article/10.1007/BF00992696
- Sutton et al. 1999 (policy gradient theorem): https://papers.nips.cc/paper/1999/hash/464d828b85b0bed98e8020cda2a01adf-Abstract.html
- Greensmith et al. 2004 (variance reduction): https://www.jmlr.org/papers/v5/greensmith04a.html
- Zhang et al. 2021 (adversarial corruption): https://proceedings.mlr.press/v139/zhang21az.html
- Cai et al. 2025 (verifiable noisy rewards): https://arxiv.org/abs/2510.00915
- Christiano et al. 2017 (RLHF): https://arxiv.org/abs/1706.03741
- Stiennon et al. 2020 (summarize from human feedback): https://papers.nips.cc/paper/2020/hash/70af9109f6a39c4f8f8c9e5b5d0f1e9-Abstract.html
- Gao et al. 2023 (reward over-optimization): https://arxiv.org/abs/2210.10760

**其他**：
- Bellinger et al. 2020 (active measure RL): https://arxiv.org/abs/2005.12697
- Jin et al. 2020 (reward-free exploration): http://proceedings.mlr.press/v119/jin20k.html
- Hansen et al. 2022 (TDMPC): https://proceedings.mlr.press/v162/hansen22a.html
- Hochreiter & Schmidhuber 1997 (LSTM): https://ieeexplore.ieee.org/document/6795963
- Elman 1990: https://www.sciencedirect.com/science/article/pii/036402139090002E
- Rumelhart et al. 1986 (backprop): https://www.nature.com/articles/323533a0
- Henaff et al. 2016 (RNN long memory): http://proceedings.mlr.press/v48/henaff16.html

---

## 17. Final Thoughts

这篇 paper 在我看来属于"理论清洁工"那一类 —— 不发明新算法，但把一个大家都在用、却没人系统分析过的范式（training in imagination）的关键 trade-off 拆解清楚。它的价值不在 novel algorithm，而在给后续 Dreamer/MuZero 类系统设计提供了一套 principled framework：

- 当你设计 latent space 时，能定量地说"为什么低 Lipschitz 好"，并 connect 到 temporal straightening。
- 当你做 data collection 时，能定量地说"dynamics 和 reward 应该按什么比例采集"，前提是你 fit 了 power-law exponents。
- 当你做 reward annotation 时，能定量地说"什么时候买便宜的 noisy annotation 比买贵的 precise annotation 更划算"。

局限也很清楚：
- Deterministic dynamics 假设避开 stochastic MDP 的复杂性。
- Worst-case bound 在 high-dim 上 loose 3 orders of magnitude，predictive power 受限。
- Power-law 假设的 universality 在不同 task / architecture 上没广泛验证。
- Realized sensitivities 在 scale 上 impractical，所以 Theorem 1 是理论指导而非直接算法。

如果顺着这条线往下做，自然的方向包括：
1. **Stochastic dynamics 推广**：用 Wasserstein distance 替代 sup-norm，借用 Asadi 2018a 的 equivalence 结果。
2. **Lipschitz regularization + capacity trade-off 的定量分析**：降 $L_f$ 升 $\varepsilon_{\text{dyn}}$，net effect 怎么算？
3. **Multi-fidelity reward model**：Corollary 2 的 $\Phi(c)$ 在 RLHF 场景下实测出来是什么形状？annotation 多 annotator 平均、不同 expert level 的 cost-noise curve。
4. **Connection to LLM RLHF**：LLM RLHF 也有 reward model learned from preferences，bias (Proposition 2) 对应 Gao et al. 2023 的 over-optimization，框架可以套用。
5. **Power-law exponents 的 predictability**：不同 task / architecture 上 $\alpha, \beta$ 怎么变？类似 Chinchilla 后续工作找 exponent scaling 规律。
