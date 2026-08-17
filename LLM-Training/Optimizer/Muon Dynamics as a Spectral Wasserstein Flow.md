---
source_pdf: Muon Dynamics as a Spectral Wasserstein Flow.pdf
paper_sha256: f7ea1a068f7f771ab447d631fae3c3d8252a99434150932825fb4433e3500f82
processed_at: '2026-08-05T21:44:42-07:00'
target_folder: LLM-Training/Optimizer
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

下面我用尽量直白的大白话，给你讲讲这篇论文到底在说什么。

### 1. 痛点：参数是矩阵，但梯度下降把它当成了 vector

在深度学习里，每一层的 weight 都是一个矩阵 $X \in \mathbb{R}^{d \times n}$。传统的 gradient descent 更新规则是 $X_{k+1} = X_k - \tau \nabla F(X_k)$。这等价于把矩阵拍平成一个超长的 vector，然后在每个坐标上独立走一小步。

问题在于，矩阵是有内部几何结构的，比如它的 singular values 和 singular vectors 携带了关于特征对齐的重要信息。坐标级别的 Euclidean geometry 把所有方向一视同仁，这没有尊重矩阵本身的形状。

### 2. Muon 怎么做的：SVD 投影

Muon optimizer 的核心动作很简单：拿到梯度矩阵 $G = \nabla F(X)$，做 SVD 分解 $G = U \Sigma V^\top$，扔掉 $\Sigma$，只保留 $U V^\top$，然后乘上总体的 spectral scale $\|G\|_{S_1}$。

直觉上，这意味着只保留梯度的“正交方向”，把所有的 singular values 强行变成 1，再统一缩放。这样在更新参数矩阵时，各个方向都被平等对待，保证了矩阵整体的旋转和形变是协调的。

### 3. Peyré 的视角跳跃：把矩阵列向量看成“粒子”

这篇论文的关键 insight 是把 $X$ 的每一列 $x_i \in \mathbb{R}^d$ 看作一个独立的“粒子”。整个矩阵 $X$ 就变成了 $n$ 个粒子组成的集合。我们可以给这个粒子集合赋予一个 empirical measure（经验测度）：

$$ \mu_X = \frac{1}{n} \sum_{i=1}^n \delta_{x_i} $$

在这个视角下，更新参数矩阵 $X$ 等价于在 measure space 里移动这些粒子。普通 gradient descent 对应于让每个粒子独立地沿着局部梯度走，彼此不交流。Muon 则不同，它强制粒子之间产生协同。

### 4. Optimal Transport 的“搬土”比喻

经典 optimal transport (OT) 里的 Wasserstein distance $W_2$ 可以用“搬土”来理解。你有一堆土（分布 $\mu$），要搬到另一个地方变成分布 $\nu$。你想找一种搬法，使得总的搬运成本（距离平方）最小。

在经典的 $W_2$ 里，成本是这样算的：对于每一对起点 $x$ 和终点 $y$，算一下距离平方 $|y-x|^2$，然后对所有 pair 独立累加。每个粒子在寻找自己的最优路径时，只关心自己的成本，不管其他粒子怎么走。

### 5. Spectral Wasserstein：强制粒子“抱团”的新距离

Peyré 定义了一种新的距离叫 Spectral Wasserstein distance $\mathsf{W}_\gamma$。它改变了成本的评估方式。

它先让所有粒子的 displacement 组装成一个全局的 covariance matrix（协方差矩阵）：

$$ S_\pi = \int (y-x)(y-x)^\top d\pi(x,y) $$

这里 $S_\pi$ 是一个 $d \times d$ 的 PSD 矩阵，描述了所有粒子整体移动方向的分布。然后，它不直接看 scalar cost，而是看这个矩阵的某种 norm $\gamma(S_\pi)$。

这就好比在搬土时，你不单独算每把土的运费，而是先统计所有土的流向，形成一个整体的“流向矩阵”，然后根据这个矩阵的整体特征来收一笔总运费。这样一来，粒子们就不能各自为政了，它们必须集体协商，让整体的“流向矩阵”看起来最省 cost。

### 6. Schatten-$p$ 家族：从 GD 到 Muon 的连续谱

最核心的 $\gamma$ 是 Schatten-$p$ norm $\gamma_p$。它只看矩阵的 eigenvalues $\lambda_1, \dots, \lambda_d$。

$$ \gamma_p(S) = \Big( \sum_{i=1}^d \lambda_i^p \Big)^{1/p} $$

这里 $p$ 是一个旋钮：
*   **$p=1$ (Trace norm)**：所有 eigenvalues 直接加起来 $\sum \lambda_i$。这就是经典 $W_2$ 的行为，每个方向独立贡献成本。对应普通 gradient descent。
*   **$p=\infty$ (Operator norm)**：只看最大的那个 eigenvalue $\lambda_{\max}$。对应 Muon optimizer。
*   **$p \in (1, \infty)$**：介于两者之间，给出一系列 intermediate 算法。

$p$ 越大，就越只关注最强的那个方向，忽略弱方向。Muon（$p=\infty$）相当于说：我只关心整体流向矩阵里最强的那个主轴方向，只要那个方向上的 cost 够小就行。这迫使所有粒子把位移对齐到少数几个主方向上，形成高度协同的集体运动。

### 7. Muon 在这个几何下为什么自然？

论文证明了，在这个 Spectral Wasserstein geometry 下，如果你要找一个 functional $F(\mu)$ 的 steepest descent（最速下降），你算出来的 velocity field 恰好就是 Muon 的更新公式。

具体来说，你要算 force field $g(x)$，然后把它打包成矩阵 $G$，做 SVD。在 $p=\infty$ 的情况下，duality map 会把 $G$ 的所有 singular values 压成 1，只保留 $U V^\top$，再乘上 spectral scale。这就是 Muon。

用 mean-field 的语言说，Muon 不是什么工程上的 hack，而是在 $\mathsf{W}_{\gamma_\infty}$ 这个几何空间里，对 measure $\mu_X$ 做的 genuine gradient flow。

### 8. Intuition：为什么“抱团”有助于训练？

把粒子想象成神经网络的参数。如果每个参数自己走自己的，很容易陷入局部最优，或者产生相互抵消的更新。Muon 通过 $S_\pi$ 强制它们看到彼此的存在。

如果 $g(x_1)$ 往东走，$g(x_2)$ 也往东走，那么在 $S_\pi$ 里“东”这个方向的 eigenvalue 就很大。Muon 会强化这种一致性，让它们更整齐地往东走。如果方向太分散，$S_\pi$ 的 eigenvalues 会很分散，Muon 会对所有方向做归一化，避免某个弱方向被忽略，也避免某个强方向过度支配。

这种 spectral coordination 使得训练时参数矩阵的更新更加 structured，减少了 noise，加快了收敛。

### 9. 这篇论文留下了什么？

这篇论文给了 Muon 一个优雅的几何解释，但也留了很多坑：
*   **Block-wise normalization**：实际 Muon 对每一层 weight 单独做 SVD。Paper 只处理了 single block 的情况，多 block 怎么纳入 mean-field 还不清楚。
*   **Momentum 和 Stochasticity**：实际训练有 momentum (EMA) 和 mini-batch noise。Paper 假设了 continuous-time, deterministic, zero-momentum 极限。
*   **Convergence theory**：在 $p=\infty$ 时，entropy convexity 只能沿某些特定 geodesic 成立，全局收敛性分析比 $W_2$ 难得多。
*   **Spherical reduction**：对于 2-homogeneous models (比如 ReLU)，可以把问题降到 sphere 上，得到一个新的 unbalanced transport geometry，但 static formulation 还没构造出来。

总而言之，Peyré 告诉我们：Muon 不是孤立的工程技巧，它是广义 Wasserstein 几何家族的一员。通过选择不同的 Schatten-$p$ norm，我们可以在“完全局部独立”的普通梯度下降和“完全全局协同”的 Muon 之间自由插值。这为理解神经网络参数更新的几何结构提供了一个很漂亮的框架。

---

# Muon Dynamics as a Spectral Wasserstein Flow 讲解

你好 Karpathy。这篇 paper 是 Gabriel Peyré 在 2026 年 4 月挂出来的, 把 Muon optimizer 放在了 optimal transport (OT) 的几何框架里。下面我尽量讲细, 把变量、几何图景、和实验都讲清楚。

---

## 1. High-Level Intuition: 把 Matrix 参数看成 Particle Cloud

普通 gradient descent 在 $X \in \mathbb{R}^{d \times n}$ 上做 $X_{k+1} = X_k - \tau \nabla F(X_k)$, 但 deep learning 里的参数本身就是 matrix shape 的 (一层 weight, attention 的 head, etc)。Muon 不直接用 Euclidean gradient, 而是先把 gradient $G_k$ 做 SVD $G_k = U_k \Sigma_k V_k^\top$ 再投影到 orthogonal group:

$$
\mathrm{Proj}_{\mathcal{O}}(G_k) = U_k V_k^\top,
$$

然后乘以 spectral scale $\|G_k\|_{S_1} = \mathrm{tr}((G_k^\top G_k)^{1/2})$。连续时间极限:

$$
\dot{X}_t = -\mathrm{tr}\!\big((G_t^\top G_t)^{1/2}\big)\, G_t (G_t^\top G_t)^{\dagger/2}, \qquad G_t = \nabla F(X_t).
$$

Peyré 的核心 insight: 把 $X$ 的每一列 $x_i \in \mathbb{R}^d$ 当作一个 particle, 定义 empirical measure

$$
\mu_X = \frac{1}{n}\sum_{i=1}^n \delta_{x_i}.
$$

那么 Muon update 不是别的, 就是某个新的 Wasserstein-type 距离下的 mean-field gradient flow。这个新距离叫 **Spectral Wasserstein distance** $\mathsf{W}_\gamma$, 由一个 PSD cone 上的 norm $\gamma$ 参数化。

---

## 2. Schatten Family: 一个 $p$-参数连续统

最核心的例子是 Schatten-$p$ norm on PSD matrices:

$$
\gamma_p(S) = \|S\|_{S_p} = \Big(\sum_{i=1}^d \lambda_i(S)^p\Big)^{1/p}, \qquad 1 \le p \le \infty,
$$

其中 $\lambda_1(S) \ge \dots \ge \lambda_d(S) \ge 0$ 是 $S$ 的 eigenvalues (因为 $S \succeq 0$, 所以奇异值 = eigenvalue)。

三个端点:

| $p$ | $\gamma_p$ | 几何意义 | 算法 |
|-----|------------|----------|------|
| $p=1$ | $\mathrm{tr}(S)$ | 经典 $\mathsf{W}_2$ Wasserstein | 普通 gradient descent |
| $p=2$ | $\|S\|_F$ (Frobenius) | 中间 Schatten | intermediate normalizer |
| $p=\infty$ | $\lambda_{\max}(S)$ | Muon / operator-norm 几何 | Muon optimizer |

中间 $p \in (1, \infty)$ 给出 interpolation family。所以整个故事是: **GD 和 Muon 是同一个 Schatten continuum 的两个端点**。

---

## 3. Static Formulation: 把 Cost 从 Scalar 升级到 Matrix-Norm

### 3.1 Definition 2.3 (核心定义)

经典 OT cost 是对 coupling $\pi \in \Pi(\mu,\nu)$ 在每对 $(x,y)$ 上独立累加 scalar cost $|y-x|^2$:

$$
\mathsf{W}_2(\mu,\nu)^2 = \inf_{\pi} \int |y-x|^2 \, d\pi(x,y).
$$

Peyré 把 $|y-x|^2$ 换成 displacement 外积的全局 covariance, 再作用一个 matrix norm:

$$
\boxed{\;\mathsf{W}_\gamma(\mu,\nu)^2 := \inf_{\pi \in \Pi(\mu,\nu)} \gamma\!\left( \int_{\mathbb{R}^d \times \mathbb{R}^d} (y-x)(y-x)^\top \, d\pi(x,y) \right)\;}
$$

记 $S_\pi := \int (y-x)(y-x)^\top d\pi$, 这是 displacement covariance matrix。

**关键区别**: 在 $\mathsf{W}_2$ 里, 每个 pair 独立贡献 $\|y-x\|^2$; 而在 $\mathsf{W}_\gamma$ 里, 所有 pair 的 displacement 被组装成同一个 $d \times d$ PSD matrix $S_\pi$, 然后由 $\gamma$ 全局评估。**Particles 通过共享同一个 $S_\pi$ 强制 collectivize** —— 这就是 Muon 的"全局 spectral coordination" 在数学上的化身。

### 3.2 Monge Restriction 是 Strict 的 (Remark 2.6)

两 Dirac 例子: $\mu = \tfrac12(\delta_{(-1,0)} + \delta_{(1,0)})$, $\nu = \tfrac12(\delta_{(0,-1)} + \delta_{(0,1)})$。
Monge map 必须是 bijection, 两个可能 displacement covariance:

$$
\begin{pmatrix}1&1\\1&1\end{pmatrix} \quad \text{or} \quad \begin{pmatrix}1&-1\\-1&1\end{pmatrix},
$$

$\|S\|_{S_\infty}^2 = 2$, $\|S\|_{S_2}^2 = \sqrt{2} \cdot \sqrt{2} = 2$ (实际是 Frobenius 平方)。但 split coupling 给 $S = I$, $\mathsf{W}_{\gamma_\infty}^2 = 1$, $\mathsf{W}_{\gamma_2}^2 = \sqrt 2$。

**意义**: 经典 Brenier map 的 Monge formulation 在 $\mathsf{W}_\gamma$ 里被 "bypass", 可以混合匹配。Muon 类 update 在 particle 层面不是 deterministic transport map, 而是 genuinely coupling-based。

### 3.3 Max-Min Cost-Robust Formulation (Theorem 2.10)

对任意 convex compact representing set $\mathcal{K}_\gamma$ (满足 $\gamma(S) = \max_{Q\in\mathcal{K}_\gamma} \mathrm{tr}(QS)$),

$$
\mathsf{W}_\gamma(\mu,\nu)^2 = \max_{Q \in \mathcal{K}_\gamma} \inf_{\pi \in \Pi(\mu,\nu)} \int (y-x)^\top Q (y-x) \, d\pi = \max_{Q \in \mathcal{K}_\gamma} \mathsf{W}^Q(\mu,\nu)^2.
$$

变量解读: $Q \in \mathcal{K}_\gamma \subset \mathbb{S}^d$ 是一个 quadratic cost matrix, $\mathsf{W}^Q$ 是以 $(y-x)^\top Q(y-x)$ 为 cost 的 anisotropic Wasserstein。**$\mathsf{W}_\gamma$ 是 cost-robust OT: 对一族 quadratic costs 取 max**。

这与 Paty–Cuturi 的 subspace robust Wasserstein ([Paty & Cuturi 2019, ICML](https://arxiv.org/abs/1905.12143)) 精神一致, 但 Paty–Cuturi 只在 subspaces 之间优化, Peyré 是在任意 PSD norm 的 dual unit ball 上优化 —— 更一般。

对 Schatten-$p$:
- $p=1$: $\mathcal{K}_{\gamma_1} = \{\mathrm{Id}\}$ (退化为单个 $Q$, 经典 $\mathsf{W}_2$)
- $1<p\le\infty$: $\mathcal{K}_{\gamma_p} = \{Q \succeq 0 : \|Q\|_{S_q} \le 1\}$, 其中 $q = p/(p-1)$ 是 dual exponent

monotonicity (Proposition 2.2) 保证 $\mathcal{K}_\gamma$ 可以选进 PSD cone, 这是后面 dynamic formula 的关键。

### 3.4 Gaussian Reduction (Theorem 2.12 + Corollary 2.13)

对 Gaussians $\mu = \mathcal{N}(m_0, \Sigma_0)$, $\nu = \mathcal{N}(m_1, \Sigma_1)$:

$$
\mathsf{W}_\gamma(\mu,\nu)^2 = \inf_{K:\;\begin{pmatrix}\Sigma_0 & K\\K^\top & \Sigma_1\end{pmatrix} \succeq 0} \gamma\!\Big( (m_1-m_0)(m_1-m_0)^\top + \Sigma_0 + \Sigma_1 - K - K^\top \Big).
$$

变量解读: $K$ 是 cross-covariance $\mathbb{E}[(X-m_0)(Y-m_1)^\top]$, block PSD 约束等价于 $K$ 是合法 joint Gaussian coupling 的 second moment。

**Commuting covariances + Schatten-$p$ 闭式** (Corollary 2.13):

$$
\mathsf{B}_{\gamma_p}(\Sigma_0,\Sigma_1)^2 = \Big(\sum_{i=1}^d \big|\sqrt{\lambda_i(\Sigma_0)} - \sqrt{\lambda_i(\Sigma_1)}\big|^{2p}\Big)^{1/p}.
$$

- $p=1$: 经典 [Bures–Wasserstein](https://arxiv.org/abs/1712.01504) formula
- $p=\infty$: $\max_i |\sqrt{\lambda_i(\Sigma_0)} - \sqrt{\lambda_i(\Sigma_1)}|^2$, 只关心最大 eigenvalue 的 mismatch

**Intuition**: Gaussian case 把无穷维 OT 问题坍缩到 $d \times d$ 矩阵优化, 给出一个 covariance metric family。$p=1$ 是 Bures, $p=\infty$ 是 "max-eigenvalue Bures"。

---

## 4. Dynamic Benamou–Brenier Formulation

### 4.1 Definition 3.1

定义 dynamic cost:

$$
\mathsf{W}_\gamma^{\mathrm{BB}}(\mu_0,\mu_1)^2 := \inf_{(\mu_t, v_t)} \int_0^1 \gamma\!\left( \int_{\mathbb{R}^d} v_t(x) v_t(x)^\top d\mu_t(x) \right) dt,
$$

约束 $\partial_t \mu_t + \mathrm{div}(\mu_t v_t) = 0$, $\mu_{t=0} = \mu_0$, $\mu_{t=1} = \mu_1$。

变量: $v_t$ 是 velocity field, $t \in [0,1]$。这里 cost 是 $\gamma$ 作用在 velocity covariance $\int v_t v_t^\top d\mu_t$。

### 4.2 Momentum Reformulation (Proposition 3.2)

令 $m = \mu v$ (momentum), 引入 perspective functional:

$$
\mathcal{A}_\gamma(\mu, m) := \sup_{Q \in \mathcal{K}_\gamma} \int \frac{w(x)^\top Q w(x)}{\rho(x)} d\lambda(x),
$$

其中 $\mu = \rho \lambda$, $m = w\lambda$。这个 formulation 关键好处: 约束 $\partial_t \mu_t + \mathrm{div}(m_t) = 0$ 变 linear, 整个问题 convex。

### 4.3 Static = Dynamic (Theorem 3.3)

在 monotone $\gamma$ 假设下 (所有 Schatten norms 都满足):

$$
\mathsf{W}_\gamma^{\mathrm{BB}}(\mu_0, \mu_1) = \mathsf{W}_\gamma(\mu_0, \mu_1).
$$

**Proof 关键步骤**:
- 上界: 给 static optimal coupling $\pi^*$, 用 displacement interpolation $\mu_t = ((1-t)x + ty)_\# \pi^*$, velocity 沿每条 segment 是 $y-x$, velocity covariance $= S_{\pi^*}$ 与 $t$ 无关, 所以 $\int_0^1 \gamma(S_{\pi^*}) dt = \gamma(S_{\pi^*}) = \mathsf{W}_\gamma^2$。
- 下界: 用 superposition principle 把 dynamic plan 提升到 path measure $\eta$, 对每个 $Q \in \mathcal{K}_\gamma \subset \mathbb{S}_+^d$ (这里 monotonicity 关键: $Q \succeq 0$), 用 Jensen:

$$
\int_0^1 \dot{\gamma}_t^\top Q \dot{\gamma}_t \, dt \ge (\gamma_1 - \gamma_0)^\top Q (\gamma_1 - \gamma_0).
$$

对 path 平均, 再对 $Q$ 取 sup, 得 dynamic cost $\ge$ static cost。

### 4.4 Metric Properties (Corollary 3.4) + Geodesics (Corollary 3.5)

- $\mathsf{W}_\gamma$ 是真 metric, 与 $\mathsf{W}_2$ 拓扑等价 (sandwiched by $c_\gamma \mathsf{W}_2 \le \mathsf{W}_\gamma \le C_\gamma \mathsf{W}_2$)
- Geodesic: $\mu_t = ((1-t)x + ty)_\# \pi^*$ 是 constant-speed geodesic, 跟 $\mathsf{W}_2$ 形式一样但 cost 不同

对 Schatten-$p$:
$$
c_{\gamma_p} = d^{1/p - 1}, \quad C_{\gamma_p} = 1.
$$
具体: $p=2$ 给 $(c, C) = (d^{-1/2}, 1)$, $p=\infty$ 给 $(d^{-1}, 1)$。**对 isotropic Gaussian 对子, lower bound tight** (Remark 2.9)。

---

## 5. Spectral Wasserstein Gradient Flows (Section 4)

### 5.1 Duality Map (Definition 4.1 + Theorem 4.2)

给定 functional $f$ 和它的 Wasserstein gradient $g_\mu(x) = \nabla_x \frac{\delta f}{\delta \mu}(x)$, tangent norm:

$$
\mathcal{N}_\mu(v)^2 := \gamma\!\left( \int v(x) v(x)^\top d\mu(x) \right).
$$

Duality map $J_\mu(g)$ 是 minimizer of:

$$
v \mapsto \int g \cdot v \, d\mu + \tfrac12 \mathcal{N}_\mu(v)^2.
$$

**Theorem 4.2 结构定理**: 令 force covariance

$$
S_\mu(g) := \int g(x) g(x)^\top d\mu(x) \in \mathbb{S}_+^d.
$$

选 active matrix $Q_\mu^* \in \arg\max_{Q \in \mathcal{K}_\gamma} \mathrm{tr}(QS_\mu(g))$。若 $Q_\mu^*$ 可逆, 则

$$
\boxed{\;J_\mu(g)(x) = -(Q_\mu^*)^{-1} g(x)\;}
$$

是合法 selector。

**变量解读**:
- $S_\mu(g) \in \mathbb{R}^{d \times d}$: $d$ 是 particle 维度 (不是 particles 数量), 它是 force field 的二阶矩
- $Q_\mu^*$: force covariance 上 $\gamma$ 的 subgradient, 决定如何 "anisotropically 旋转/scaling" gradient
- $J_\mu(g)(x)$: 每个粒子 $x$ 的 velocity 由 force $g(x)$ 被 $Q_\mu^*$ 的逆作用得到

**Intuition**: 在 $\mathsf{W}_2$ (p=1) 里, $Q^* = \mathrm{Id}$, 所以 $J = -g$ —— 经典 GD。在 Muon (p=∞) 里, $Q^*$ 是 force covariance 的 max-eigenvalue rank-1 投影 (奇异, 需要更细致处理), $J$ 把 force 的所有奇异方向"压平"到单位大小, 再乘 spectral scale。

### 5.2 Finite-Width: Explicit Schatten Selectors (Proposition 4.4)

对 empirical measure, 把 force field stack 成 $G \in \mathbb{R}^{n \times d}$ (rows 是 $g(x_i)$), SVD $G = U\,\mathrm{diag}(\sigma_i)\,W^\top$。

记 $r = 2p$, $q = r/(r-1) = 2p/(2p-1)$ 是 dual exponent:

$$
\boxed{\;\Xi_p(G) = -\|G\|_{S_q}^{2-q} \, U\, \mathrm{diag}(\sigma_i^{q-1})\, W^\top\;}
$$

变量: $\sigma_i$ 是 $G$ 的 singular values, $U$ 和 $W$ 分别是左右 singular vectors, $\|G\|_{S_q}$ 是 Schatten-$q$ norm of $G$ (即 $\ell^q$ norm of singular values)。

三个特殊情形:

| $p$ | $q$ | $\Xi_p(G)$ 公式 | 算法 |
|-----|-----|-----------------|------|
| $1$ | $2$ | $-G$ | 经典 gradient descent |
| $2$ | $4/3$ | $-\|G\|_{S_{4/3}}^{2/3}\, U\,\mathrm{diag}(\sigma_i^{1/3})\,W^\top$ | Frobenius 中间 |
| $\infty$ | $1$ | $-\|G\|_{S_1}\, U W^\top = -\mathrm{tr}((G^\top G)^{1/2})\, G (G^\top G)^{\dagger/2}$ | **Muon** |

**Muon 推导细节**: $G(G^\top G)^{\dagger/2} = U\Sigma W^\top \cdot W\Sigma^\dagger W^\top = U \mathrm{diag}(\mathrm{sgn}\,\sigma_i) W^\top = U W^\top$ (假设 $\sigma_i > 0$), 即把所有奇异方向 normalize 到 unit magnitude。然后乘 $\|G\|_{S_1} = \sum \sigma_i$, 这是 "总 spectral mass"。

**Intuition**: Muon 把 force matrix 的每个奇异方向同等对待 (magnitude 1), 然后乘总 spectral scale。$p=1$ 时所有方向按原 $\sigma_i$ 大小 (各向同性 squared), $p=\infty$ 时所有方向等同, 中间 $p$ 给 partial equalization (按 $\sigma_i^{q-1}$ 幂律加权, 当 $q \to 1$ 时趋均匀)。

### 5.3 Continuous-Time Muon (Corollary 4.5)

若 empirical measure 保持, 则 stacked particle matrix $X_t \in \mathbb{R}^{n \times d}$ 满足:

$$
\dot{X}_t = \Xi_p(\nabla F_n(X_t)).
$$

**这就是 paper 的中心结果**: Muon continuous-time flow 是 Spectral Wasserstein $\mathsf{W}_{\gamma_\infty}$ 的 mean-field gradient flow。Mean-field 视角下, $n$ (particle 数) 可以任意大, 自然被 measure $\mu_X$ 描述。

### 5.4 Gaussian-Preserving Regime (Corollary 4.6)

若 $g_\mu$ 在 Gaussian state 上 affine in $x$, 且 active matrix $Q_\mu^*$ 可逆, 则 Spectral flow 保持 Gaussianity, 退化到 mean/covariance ODE:

$$
\dot{m}_t = a_t + A_t m_t, \qquad \dot{\Sigma}_t = A_t \Sigma_t + \Sigma_t A_t^\top.
$$

适用例子: 相对 entropy $\mathrm{Ent}_\nu$ (Fokker-Planck)、quadratic loss 上训练 linear two-layer net。

---

## 6. Geodesic Convexity (Section 5)

### 6.1 Linear Functionals (Theorem 5.2)

$F_h(\mu) = \int h \, d\mu$:
1. $F_h$ 凸 $\Leftrightarrow$ $h$ 凸
2. $F_h$ $\kappa$-凸 $\Leftrightarrow$ $\nabla^2 h(z) \succeq \kappa Q$ for all $z \in \mathbb{R}^d$, $Q \in \mathcal{K}_\gamma$

对 Schatten family, **条件等价于** $\nabla^2 h(z) \succeq \kappa\,\mathrm{Id}$, 与经典 $\mathsf{W}_2$ 完全一样 (Remark 5.3)。这点其实有点 surprise, 因为虽然 geometry 不同, 但 convexity 阈值一样。

### 6.2 Relative Entropy (Theorems 5.4–5.5)

需要 reference $\nu \propto e^{-V}$。**强结论 (Theorem 5.4)**: 若 $\mathcal{K}_\gamma \subset \mathbb{S}_{++}^d$ (uniformly elliptic) 且 $\nabla^2 V \succeq \kappa Q$ for all $Q$, 则 $\mathrm{Ent}_\nu$ 是 $\kappa$-geodesically convex。

证明技巧: 用 active $Q^* \succ 0$ 把问题变到 $L = (Q^*)^{1/2}$ 变换下的经典 $\mathsf{W}_2$, 套经典 McCann displacement convexity, 然后变回。

**Schatten 家族的 caveat**: $p=1$ 满足 (取 $\mathcal{K}_{\gamma_1} = \{\mathrm{Id}\}$)。但 $p > 1$ 时 $\mathcal{K}_{\gamma_p}$ 含 singular matrices (rank-1 projectors), **强结论不成立**。

**弱结论 (Theorem 5.5)**: 只需 $\mathcal{K}_\gamma \cap \mathbb{S}_{++}^d \ne \emptyset$ (Schatten 都满足), 给出 **at least one geodesic** 上成立 entropy convexity (而非所有 geodesic)。证明用 regularized $\gamma_\varepsilon = (1-\varepsilon)\gamma + \varepsilon \mathrm{tr}(Q_0 \cdot)$ 强 elliptic, 再取 $\varepsilon \to 0$ 极限。

**意义**: Muon 类 ($p=\infty$) 的 entropy 凸性只沿某些特定 geodesic 成立, 比 $\mathsf{W}_2$ 弱 —— 这与 Muon 训练动力学可能更复杂, 收敛性更难分析吻合。

---

## 7. Spherical Reduction for Homogeneous Models (Section 6)

### 7.1 Mean-Field MLP

Predictor $H_\mu(z) = \int \phi(z, x) d\mu(x)$, 风险 $f(\mu) = R(H_\mu)$。Two-layer 标量 output: parameter $x = (u, v) \in \mathbb{R}^d$, feature $\phi(z, x) = u\sigma(v \cdot z)$。Quadratic loss 使 $f$ 成为 pairwise interaction functional, MMD-like。

### 7.2 Positive 2-Homogeneous Reduction (Propositions 6.2–6.4)

若 $\Phi(\lambda x) = \lambda^2 \Phi(x)$ (e.g. ReLU 两层 net 在某些 scaling 下), $x = r\omega$, $\omega \in \mathbb{S}^{d-1}$:

$$
\int \psi \, d\Pi_2(\mu)(\omega) := \int_{\mathbb{R}^d} |x|^2 \psi(x/|x|) d\mu(x).
$$

**Proposition 6.2**: $f(\mu) = \bar f(\Pi_2(\mu))$ —— 只看 weighted spherical projection。

**Proposition 6.3**: Projected PDE 在 sphere 上:

$$
\partial_t \nu_t + \mathrm{div}_{\mathbb{S}^{d-1}}(\nu_t \tau_t) = 2 b_t \nu_t.
$$

变量: $\nu_t = \Pi_2(\mu_t)$ 是 sphere 上的 weighted measure, $\tau_t$ 是 tangential velocity, $b_t$ 是 radial (reaction) coefficient。

**Proposition 6.4**: Ambient Spectral Wasserstein action = spherical unbalanced action:

$$
\mathsf{UW}_\gamma(\nu_0,\nu_1)^2 := \inf_{(\nu_t, b_t, \tau_t)} \int_0^1 \gamma\!\left(\int (b_t\omega + \tau_t)(b_t\omega + \tau_t)^\top d\nu_t\right) dt
$$

约束 $\partial_t \nu_t + \mathrm{div}_{\mathbb{S}^{d-1}}(\nu_t \tau_t) = 2b_t \nu_t$。

**Remark 6.5**: $\gamma = \mathrm{tr}$ 时退化到经典 [Wasserstein–Fisher–Rao](https://arxiv.org/abs/1811.12788) 几何 (Chizat–Bach 等 unbalanced OT on sphere)。**对一般 Schatten-$p$ 给出新的 spherical unbalanced transport geometry** —— 这是 paper 的副产品, 也是 open problem (static formulation 还没构造)。

---

## 8. Numerical Experiments (Section 7)

### 8.1 Static Spectral Couplings

Source: anisotropic Gaussian cloud, Target: far Gaussian mixture, $n = m = 200$, uniform weights $1/n$。

离散问题:

$$
\min_{P \ge 0, \, P\mathbf{1}=a, \, P^\top \mathbf{1}=b} \gamma_p\!\left( \sum_{i,j} P_{ij} (y_j - x_i)(y_j - x_i)^\top \right).
$$

- $p=1$: 退化为经典线性 OT, 等权时是 assignment
- $p=2$: SOCP, CVXPY 解
- $p=\infty$: SDP via maximal eigenvalue, CVXPY 解

Figure 1 (paper 中) 显示: 不同 $p$ 选不同 couplings, $p=\infty$ 倾向于 displacement 在 max-eigenvalue 方向对齐的匹配。

### 8.2 MMD Gradient Flows (Figure 2)

最小化 MMD:
$$
f(\mu) = \mathrm{MMD}(\mu, \nu)^2, \quad k(x,y) = -\|x - y\|_\varepsilon, \quad \varepsilon = 10^{-2}.
$$

$n = m = 200$, $d = 2$, smoother kernel $\|x-y\|_\varepsilon = \sqrt{\|x-y\|_2^2 + \varepsilon^2}$。

Euler discretization:
$$
\mathbf{X}_{k+1} = \mathbf{X}_k + \eta_p \Xi_p(G_k), \quad p \in \{1, 2, \infty\}.
$$

可视化结果 (paper Figure 2 描述):
- **$p=1$ (W_2)**: 粒子 locally 响应 force, 各自直奔最近 target
- **$p=2$ (Frobenius)**: 中等 coordination
- **$p=\infty$ (Muon)**: 最 globally coordinated, 粒子集体沿 dominant spectral direction 运动

直觉: 三个 dynamics 优化同一个 functional, 只是 normalized transport geometry 不同。$p$ 越大, particles 通过共享同一个 covariance matrix 被 "强迫" 越多地 collective 协作。

---

## 9. Connections 和 Open Problems

### 9.1 与 Muon 实践的关系

实际 Muon ([Keller Jordan's blog](https://kellerjordan.github.io/posts/muon/), [Liu et al. 2025](https://arxiv.org/abs/2502.16982)) 还有:
- Per-block normalization (每个 weight block 单独 normalize)
- Momentum (EMA of gradient)
- Stochasticity (mini-batch)

Peyré 把这些 idealize 掉: continuous-time, deterministic, single block。但 mean-field 视角给出 Muon 的几何解释。

### 9.2 与 Cost-Robust OT 的关系

| Work | Cost 形式 | 优化方向 |
|------|---------|----------|
| [Paty–Cuturi 2019](https://arxiv.org/abs/1905.12143) | projected quadratic on subspaces | max over subspaces |
| [Wasserstein Discriminant Analysis](https://arxiv.org/abs/1902.10749) | learned metric | max (supervised) |
| [Ground Metric Learning](https://arxiv.org/abs/1602.00604) | structured ground | min (concave, nonconvex) |
| [Sebbouh et al. 2024](https://arxiv.org/abs/2402.16428) | cost-regularized | min, nonconvex |
| **This paper** | PSD-norm of displacement covariance | **max, convex** |

关键: Peyré 的 max 是 concave (cost 在 OT 是凹的), 所以 max-min 仍 convex, 可分析。这与 Sebbouh 的 min-over-cost (Gromov-Wasserstein 风格) 相反, 后者 inherently nonconvex。

### 9.3 Open Problems (Section 8)

1. 更 sharp 的 optimal coupling characterization, 超出 conditional Brenier
2. 把 [Chizat–Bach 2018](https://arxiv.org/abs/1802.03862) 的 global convergence theory 从 $\mathsf{W}_2$ 扩展到 Schatten-$p$ —— spherical reduction 是关键 tool, 但 $p>1$ 时 entropy 凸性弱
3. Block-wise geometries: 实际 Muon 每 block 单独 normalize, paper 只做 single block mean-field
4. 加 momentum 和 stochasticity 的 metric 解释
5. Spherical $\mathsf{UW}_\gamma$ 的 static Kantorovich formulation

### 9.4 References 我能找到的关键链接

- [Muon optimizer 原博客 (Keller Jordan)](https://kellerjordan.github.io/posts/muon/)
- [Muon is scalable for LLM training (Liu et al. 2025)](https://arxiv.org/abs/2502.16982)
- [Gabriel Peyré homepage](https://www.di.ens.fr/~peyre/)
- [Paty–Cuturi subspace robust Wasserstein (ICML 2019)](https://arxiv.org/abs/1905.12143)
- [Chizat–Bach mean-field OT (NeurIPS 2018)](https://arxiv.org/abs/1802.03862)
- [Bures-Wasserstein distance (Bhatia–Jain–Lim)](https://arxiv.org/abs/1712.01504)
- [Covariance-modulated OT (Burger et al. 2025)](https://arxiv.org/abs/2305.05658)
- [Pethick et al. LMO framework (ICML 2025)](https://arxiv.org/abs/2502.01104)
- [Benamou–Brenier original (2000)](https://link.springer.com/article/10.1007/s002110050020)
- [Cuturi–Avis ground metric learning](https://jmlr.org/papers/v15/cuturi14a.html)
- [Gozlan et al. weak transport](https://arxiv.org/abs1512.06633)
- [Schatten norm on Wikipedia](https://en.wikipedia.org/wiki/Schatten_norm)
- [Code repo (作者提供)](https://github.com/gpeyre/spectral-wasserstein)

---

## 10. Build-Intuition Summary

最后给一个 mental model:

**梯度矩阵 $G$ 的 SVD $U\Sigma W^\top$** 是连接 OT 和 Muon 的桥梁。
- $\Sigma$ = "spectral mass distribution": 哪些方向有 force
- $U$ (left) = particles-side 方向
- $W$ (right) = feature-side 方向

**Schatten-$p$ parameter $p$ 控制方向 equalization 程度**:
- $p=1$: 完全保留 $\Sigma$ (各向异性 squared, 跟原 $G$ 一样)
- $p=\infty$: 完全压平 $\Sigma \to \mathbf{1}$ (所有方向等权)
- 中间 $p$: $\sigma_i^{q-1}$ 加权, $q = 2p/(2p-1)$

**OT 视角**: 每对粒子 displacement 进入共享 covariance $S_\pi$, 再被 $\gamma$ 评估。$\gamma$ "选择" 哪些 displacement 方向 costly: $p=1$ 跟踪所有方向 (trace), $p=\infty$ 只跟踪 top 方向 (operator norm)。Muon 让所有 singular direction 等权 = 等价于在 operator norm 几何下做 steepest descent。

**Mean-field 视角**: 把 $n$ 个 column 想成 $n$ 个 particles, $\mu_X = \frac1n \sum \delta_{x_i}$ 是它们的 empirical measure。Muon 不是用 Euclidean gradient 走 particle, 而是用 $\mathsf{W}_{\gamma_\infty}$ 距离下的 gradient flow —— particles 通过共享 force covariance $S_\mu(g)$ 被强制 collective。

**几何-算法对偶**: 这套理论把 "normalized gradient methods" 和 "generalized OT geometry" 统一了。任何 monotone PSD norm $\gamma$ 同时定义 (a) 一个 OT 距离, (b) 一个 normalized gradient flow。Schatten-$p$ 家族就是这条 continuum 上最自然的离散点。

希望这给你 build intuition。如果还想往下挖, 我觉得最值得追的方向是: (i) $p \in (1, \infty)$ 在实际 LLM 训练中是否比 $p=\infty$ 更稳或更快 (paper 没测, 只有 2D MMD toy); (ii) 多 block 的 mean-field (实际 Muon 每 block 单独 normalize, 单 block mean-field 可能 underestimate coordination); (iii) momentum 的 metric 拓展 (类似 [Wibisono–Wilson–Jordan 2016](https://arxiv.org/abs/1603.04279) 在 $\mathsf{W}_2$ 上做的事, 但在 $\mathsf{W}_\gamma$ 上)。
