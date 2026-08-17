---
source_pdf: Latent Spherical Flow Policy for Reinforcement Learning.pdf
paper_sha256: 21e343d9fda344525415289c186b5c6fb2481439a941552a2a1faf4f25702f7d
processed_at: '2026-08-05T12:18:50-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 LSFLOW

## 这篇 paper 想解决什么问题

想象你在管一个急救调度中心。每分钟要决定：派哪几辆救护车去哪几个地点，每辆车走哪条路，每个病人分配到哪个医院。这些决策有硬约束——车辆数量有限、每辆车有容量、路径必须连通、医院不能超载。可行方案可能有 $2^{100}$ 种，你没法枚举。

传统 RL 在这上面会崩盘。因为 policy 要么直接输出一个 action（连续空间，不知道怎么编码"选3辆车走路线A-B-C"），要么输出一个 softmax over all actions（$2^{100}$ 个 logit，内存爆炸）。Prior work 要么把 Q-network 塞进 MIP solver 每次贪心选 action（SEQUOIA，慢，critic 受限），要么学一个 deterministic structured policy（SRL，没法 explore，容易陷局部最优）。

LSFLOW 的核心 insight：**把 stochasticity 和 feasibility 拆开**。Policy 只负责产生 stochasticity，solver 负责保证 feasibility。

---

## 核心 idea：一个类比

把 LSFLOW 想象成"点菜"：

- **传统 RL**：你直接说"我要第 847 号组合"——但菜单有 $2^{100}$ 项，你说不完
- **LSFLOW**：你说"我现在偏好咸的多一点、辣的少一点、素的优先"——给一个 preference vector $c$，让一个智能服务员（CO solver）根据你的 preference + 餐厅约束（预算、库存、营养均衡）给你端上一个 feasible 组合

你学的是"preference 的分布"，不是"选择的分布"。Preference 是连续的低维 vector，好学；服务员是 deterministic 的优化器，保证给你 feasible 的结果。

**数学上**：
- Policy 输出 cost vector $c \in \mathbb{R}^m$
- Solver 解 $\arg\min_{a \in \mathcal{A}(s)} c^\top a$ 得到 feasible action $a$
- Policy 的 stochasticity 来自 $c$ 的分布，feasibility 来自 solver

---

## 为什么 cost vector 要住在 sphere 上

这里有个特别优雅的小观察（Lemma 3.1）：

Solver 是 $\arg\min_a c^\top a$。如果你把 $c$ 乘以 2，变成 $\arg\min_a 2c^\top a$，最小化的 $a$ 没变——因为正数缩放不改变排序。所以 **solver 只看 $c$ 的方向，不看大小**。

这意味着 $c$ 的 magnitude 是冗余自由度。与其在 $\mathbb{R}^m$ 上学分布（白白浪费容量学无用的 magnitude），不如直接把 $c$ 归一化到 unit sphere $\mathbb{S}^{m-1}$ 上，只学方向。

这个"从 invariance 推 geometry"的思路特别漂亮。Solver 的 scale invariance 自然告诉你 latent space 应该是 sphere，这是个 quotient space 的思路。不这么做（ablation 显示）performance 会掉，因为模型要同时学 magnitude 和 direction，但 magnitude 对最终 action 毫无影响，纯添乱。

---

## 为什么用 flow matching 而不是 diffusion

Flow matching 和 diffusion 都是 generative model，从噪声分布 transport 到 target 分布。区别：

- **Diffusion**：基于 SDE（随机微分方程），加噪声再去噪，实现复杂
- **Flow matching**：基于 ODE（常微分方程），学一个 velocity field 从 $p_0$ 流到 $p_1$，更简单更快

在 sphere 上这个差别更明显。Sphere 是 Riemannian manifold，flow matching 有现成的 Riemannian 版本（Chen & Lipman 2024），geodesic interpolation + projected velocity field 就能搞定。Spherical diffusion 要在 manifold 上处理 Brownian motion、score function 的 SDE，实现起来麻烦得多。

具体怎么在 sphere 上做 flow matching：

1. **Interpolation**：不用直线 $(1-t)c_0 + t c_1$，用大圆弧（SLERP）
$$c_t = \frac{\sin((1-t)\theta)}{\sin\theta} c_0 + \frac{\sin(t\theta)}{\sin\theta} c_1$$
其中 $\theta = \arccos(c_0^\top c_1)$ 是两点间夹角。这保证 $c_t$ 始终在 sphere 上。

2. **Velocity field**：学一个 $v_\theta(c_t, s, t)$，但每个点的 velocity 必须在 tangent space 内（不能有 radial 分量，否则会推离 sphere）。所以投影一下：
$$\frac{dc_t}{dt} = \Pi_{c_t} v_\theta, \quad \Pi_c = I - cc^\top$$

$\Pi_c$ 把任何 vector 投影到 $c$ 处的切空间，扔掉 radial 部分。这就像在地球表面行走——velocity 必须是"沿地面"的，不能"往天上飞"。

---

## Expressivity：这玩意真的能表达任何 stochastic policy 吗？

能。Proposition 3.2 证明：对任意 target 分布 $\mu$ over feasible actions，都存在一个 cost distribution $\pi$ on sphere，使得 pushforward $a^*(s, c) \sim \mu$。

直觉：solver 在 sphere 上切出很多 region（normal fan 结构），每个 region 内部 solver 输出同一个 action。这些 region 覆盖整个 sphere。所以你可以把 sphere surface measure 在每个 region 上 normalize，再用 $\mu$ 做 mixture weight 拼起来，就得到想要的 $\pi$。

这告诉你：**sphere 上的分布是足够 expressive 的**，关键是怎么学。Flow matching 给你 expressive 的工具，剩下的就是 RL 部分。

---

## 计算瓶颈：为什么需要 latent cost space critic

Naive 实现有个大问题。Policy update 时要 evaluate 很多 candidate $c$ 的 value $Q(s, a^*(s, c))$。每次 evaluate 都要 call solver 一次。MIP solver 是 NP-hard，每次几毫秒到几秒，一个 update step 里 sample 几百个 candidate，总计几十分钟。Policy update 慢得没法训练。

LSFLOW 的 fix：**直接在 cost space 学 critic**：
$$\widetilde{Q}_\phi(s, c) := Q(s, a^*(s, c))$$

$\widetilde{Q}_\phi$ 是个普通神经网络，输入 $(s, c)$ 输出 scalar。训练数据来自 environment interaction——每次 env step call 一次 solver 执行 action，记录 reward，用这个 reward 训 critic。之后 policy update 时只 query $\widetilde{Q}_\phi$，不再 call solver。

效果：action-space critic 一步 policy update ~6 分钟，cost-space critic <1 秒。这是 ~360x 加速，method 从"不可行"变成"完全可训练"。

---

## Piecewise-Constant 问题：为什么需要 smoothing

但 latent critic 引出新问题。看 $\widetilde{Q}(s, c) = Q(s, a^*(s, c))$ 作为 $c$ 的函数：

- 在 sphere 上每个 region 内部，$a^*(s, c)$ 是同一个 action，所以 $\widetilde{Q}$ 是常数
- 跨过 region 边界，$a^*$ 跳到另一个 action，$\widetilde{Q}$ 突变到另一个值

这就是 piecewise constant function——阶梯状。问题：

1. **不连续**：Bellman target 在边界附近剧烈跳变，神经网络很难拟合
2. **梯度几乎处处为 0**：region 内部 Q 是常数，gradient 是 0；边界处不可导。网络根本学不动
3. **Bootstrap 不稳定**：target 网络的输出在边界附近 jitter，训练发散

**LSFLOW 的 fix**：用 vMF kernel smoothing。

vMF 是 sphere 上的 Gaussian。在 $\mathbb{R}^d$ 上 Gaussian 是 $\exp(-\|x-\mu\|^2/2\sigma^2)$；在 sphere 上没有"距离平方"的概念，用 cosine similarity 替代：
$$K_\kappa(\tilde{c} | c) \propto \exp(\kappa c^\top \tilde{c})$$

$c^\top \tilde{c} = \cos\theta$ 是两个方向的夹角余弦。$\kappa$ 类似 $1/\sigma^2$：大 $\kappa$ 尖锐，小 $\kappa$ 弥散。

**Smoothed Bellman operator**：
$$(\mathcal{T}_\kappa^\pi Q)(s, c) = \mathbb{E}\left[r(s, a^*(s, \tilde{c})) + \gamma Q(s', \tilde{c}')\right]$$

其中 $\tilde{c} \sim K_\kappa(\cdot | c)$。意思是不再用 $c$ 直接 query，而是从 $c$ 附近的 vMF 分布 sample 一个 $\tilde{c}$ 再 query，相当于对 piecewise constant function 做局部平均。结果是 smooth function。

**理论保证**（Theorem 3.5）：
1. $\gamma$-contraction → 有唯一 fixed point
2. Fixed point 是 $C^\infty$（无限可微）的

证明核心：vMF kernel 是 $C^\infty$ 的，compact manifold 上卷积保留 smoothness，所以每次 Bellman backup 都自动 smooth。

**Ablation 验证**：
- 不 smooth：训练不稳定，性能差
- 中等 $\kappa$：最佳
- 过度 smooth：value landscape 被抹平，引入 bias，性能也掉

经典的 bias-variance tradeoff。$\kappa$ 是 critical hyperparameter。

---

## Weighted Flow Matching：怎么 update policy

有了 critic，policy update 怎么做？LSFLOW 用 weighted flow matching：

1. 从旧 policy $\pi_k$ sample $c_1$
2. 给每个 sample 算一个 importance weight：
$$w(s, c) \propto \exp\left(\frac{1}{\lambda} \widetilde{Q}(s, c)\right)$$
高 value 的 cost direction 权重大
3. 用 weight 加权 flow matching loss，refit 新 policy

**这个 trick 等价于 KL-regularized policy improvement**（Theorem H.1）：
$$\pi_{k+1} = \arg\max_\pi \mathbb{E}_c[Q] - \lambda \mathrm{KL}(\pi \| \pi_k)$$

KL term 像 trust region（PPO/TRPO 的亲戚），防止 update 步子太大崩溃。$\lambda$ 从 2.0 退火到 0.8，前期保守后期激进。

这和 DPO 的 trick 极其相似——DPO 也是把 KL-regularized RL objective 变成 weighted regression，避免显式优化 KL。Ma et al. 2025a 在 diffusion policy 上也用了这个思路。LSFLOW 把它移植到 flow matching 上。

---

## 训练循环（Algorithm 1）

每个 iteration：

**Environment step**：
1. Sample $c \sim \pi_\theta(\cdot | s)$（积分 spherical ODE）
2. vMF 扰动 $\tilde{c} \sim K_\kappa(\cdot | c)$
3. Call solver $a = a^*(s, \tilde{c})$，执行，拿 reward
4. 存 $(s, c, r, s')$ 到 replay buffer

**Critic update**：
1. Sample next $c' \sim \pi_\theta(\cdot | s')$
2. vMF 扰动 $\tilde{c}' \sim K_\kappa(\cdot | c')$，重复 $J$ 次
3. Target $y = r + \gamma \frac{1}{J}\sum_j \widetilde{Q}_{\bar\phi}(s', \tilde{c}'^{(j)})$
4. Minimize $\|\widetilde{Q}_\phi(s, c) - y\|^2$

**Actor update**：
1. Sample $c_1 \sim \pi_\theta(\cdot | s)$
2. Weight $w = \exp(\widetilde{Q}_{\bar\phi}(s, c_1)/\lambda)$
3. Minimize weighted flow matching loss

**Target soft update**：$\bar\phi \leftarrow (1-\eta)\bar\phi + \eta\phi$

注意 env step 只 call 一次 solver，整个 policy/critic update 都不再 call solver。这就是 amortization 的威力。

---

## 实验：真的 work 吗

### Benchmark

四个 task：scheduling, routing, assignment, intervention。都是 combinatorial + sequential。

| Method | Avg Reward | Avg Train Time |
|--------|------------|----------------|
| Random | 10.85 | - |
| Greedy | 14.35 | - |
| DQN-Sampling | 18.15 | 0.52h |
| SEQUOIA | 22.21 | 9.11h |
| SRL | 22.91 | 3.89h |
| **LSFLOW** | **27.62** | **1.29h** |

- 比 SRL 平均高 20.6%
- 比 SRL 快 3x，比 SEQUOIA 快 7x
- 四个 task 全部 best

为什么这么快：SRL 训练时 differentiate through solver，每个 cost vector 要 sample 多个 perturbation 每个 solve 一次；LSFLOW 每个 env step 只 solve 一次，update 时完全不 solve。

为什么这么好：stochastic policy 能 explore multimodal landscape，deterministic SRL 容易陷局部最优。Dynamic Intervention 上提升最大（28.8%），因为 sigmoid response + budget constraint 让 landscape 特别多模态。

### STI Testing 真实应用

Sexually transmitted infection testing on real contact network。四种 disease，budget=5 per round，frontier constraint（只能测已测 node 的邻居或 component root）。

Result：LSFLOW 在低 budget regime 显著优于 SRL。低 budget 时 targeted testing 最有价值，stochastic policy 能更好 explore 不同 component root 的顺序。

---

## 几个值得 internalize 的 takeaway

**1. Invariance 推 geometry**

Solver 的 scale invariance 看起来 trivial，但它告诉你 latent space 应该是 sphere。这种"从对称性推几何"的思路很有 power，类似 gauge theory 里从对称性推相互作用。

**2. Amortization 是核心**

Expensive operation（solver）没法消除，但可以 amortize。通过学一个 critic 把 solver 的输出"缓存"成神经网络查询，policy update 就不再受 solver 拖累。这种思路在 RL with expensive simulator、active learning、BO 里都常见。

**3. Smoothing as regularization**

Piecewise constant 是 combinatorial structure 的本质特征。vMF smoothing 不是 hack，是 principled fix——sphere 上有 closed-form heat kernel，convolution 给 $C^\infty$ 输出，Bellman operator 自动 contract。这种"用 manifold 合适的 kernel 做 smoothing"的思路在 manifold learning、graph neural network 里都有出现。

**4. Weighted regression = KL improvement**

这个 trick 在 generative RL 中反复出现。DPO 用它做 preference learning，diffusion policy 用它做 policy improvement，LSFLOW 把它移植到 flow matching。核心 idea：用 importance reweighting 把 KL-regularized objective 转成 weighted supervised regression，避免显式优化 KL 的高方差。

---

## 这篇 paper 的局限

- 只支持 linear objective solver。很多 CO 问题是 quadratic、semidefinite，需要扩展
- $\kappa$ 是 critical hyperparameter，需要手调。Adaptive schedule 是 open direction
- 实验规模不大（N=40），real-world 大规模 instance 的 solver cost 仍是瓶颈
- 理论只在 fixed policy 下证明 contraction，optimality guarantee、sample complexity 都没分析

---

## 一句话总结

**LSFLOW 把 combinatorial RL 的 feasibility 委托给 solver，把 stochasticity 委托给 spherical flow policy，用 latent critic 摊掉 solver 成本，用 vMF smoothing 处理 solver 引入的不连续性。** 简单、优雅、work。

如果你想进一步 dive in，我建议从三个地方读：
1. Section 3.1 的 solver-induced policy + Lemma 3.1（理解 sphere 的由来）
2. Section 3.3 的 smoothed Bellman operator + Theorem 3.5（理解理论核心）
3. Appendix H 的 weighted flow matching = KL improvement（理解训练 objective 的本质）

这三个 piece 拼起来就是整个 framework 的骨架，其余都是 implementation detail。

---

# LSFLOW: Latent Spherical Flow Policy for Combinatorial RL — 深度解析

## 1. 问题动机：Combinatorial Action Spaces 为什么难

标准 RL 假设 action space 是连续的 $\mathbb{R}^d$ 或一个小规模离散集合。但许多真实决策问题的 action 不是单纯的 scalar，而是 structured decision：

- **Subset selection**：从 N 个候选里选 budget ≤ B 个（$2^N$ 种选择）
- **Routing**：在 graph 上选一条 feasible path
- **Scheduling**：把 workers 分配到 patients 的 time slots
- **Assignment**：capacity-constrained bipartite matching

这些都满足 hard feasibility constraints，feasible set $|\mathcal{A}(s)|$ 随 problem size 指数膨胀，无法 enumerate。Standard RL 的 softmax over discrete actions 直接崩盘。

Prior work 主要两条路：
1. **Solver-based value optimization**（如 SEQUOIA, Xu et al. 2025）：把 learned Q-network embed 进 MIP，每次 greedy action selection 都要 solve 一个 integer program。问题：要求 critic 是 solver-friendly（通常 ReLU MLP），表达能力受限；MIP 规模随网络变大。
2. **Optimization-induced policy**（如 SRL, Hoppe et al. 2025）：把 solver 当作 differentiable layer 嵌入 policy，用 Fenchel-Young loss 训练。问题：得到的是 deterministic policy，exploration 差，在 multimodal landscape 下容易陷局部最优。

LSFLOW 的核心 insight：**把 stochasticity 和 feasibility 解耦** —— 在 continuous latent space 学 expressive stochastic policy，把 feasibility 委托给 CO solver。

参考资料：
- SEQUOIA paper: https://openreview.net/forum?id=DhH3LbA6F6
- SRL paper: https://openreview.net/forum?id=GS9o7u5njS
- Flow Matching on General Geometries (Chen & Lipman): https://openreview.net/forum?id=g7ohDlTITL

---

## 2. 核心架构：Solver-Induced Policy

### 2.1 Two-Stage Construction

给定 state $s$，policy 不直接产生 action，而是：

**Stage 1**：从 latent distribution 采样一个 cost vector $c \in \mathbb{R}^m$：
$$c \sim \pi_\theta(\cdot | s)$$

**Stage 2**：把 $c$ 喂给 CO solver，solve 一个 linear-objective 程序：
$$a^*(s, c) = \arg\min_{a \in \mathcal{A}(s)} c^\top a \quad \text{(Eq. 2)}$$

变量含义：
- $s$：当前 state
- $c \in \mathbb{R}^m$：cost vector，对 action 的每个 binary decision variable 给一个 linear coefficient
- $a \in \{0,1\}^m$：feasible action，编码成 binary decision vector（任何 bounded integer 都能 binary encode，参考 Dantzig 1963）
- $\mathcal{A}(s) \subseteq \{0,1\}^m$：state-dependent feasible set，编码 combinatorial constraints
- $c^\top a = \sum_{i=1}^m c_i a_i$：linear objective

这个 pushforward $\pi_\theta \to a^*$ 的组合 induced 一个 stochastic policy over feasible actions。Stochasticity 完全来自 $c$ 的分布，feasibility 完全来自 solver。

**Intuition**：可以理解为 solver 是一个 "decoder"。Latent space 是 continuous cost direction，solver 是确定性的、保证 feasible 的 decoder。类比 VAE：encoder 学 posterior，decoder 把 latent 映射回 data；这里 latent = cost direction，decoder = LP/MIP solver。

### 2.2 Routing Example（帮助 build intuition）

设想 VRP（vehicle routing）问题：
- $a \in \{0,1\}^m$ 表示选了哪些 edge
- $\mathcal{A}(s)$ 编码 connectivity、flow conservation、budget 约束
- $c_i$ 是 edge $i$ 的 state-dependent cost（拥堵、距离）
- $a^*(s, c)$ 返回 min-cost feasible route

如果 policy 学会了 "在拥堵时把某条 edge 的 cost 拉高"，solver 自然就会避开它。Policy 的表达力转移到 cost 的分布上。

---

## 3. 为什么用 Sphere：Lemma 3.1 的妙用

### 3.1 Positive Scale Invariance

**Lemma 3.1**：对任意 $\alpha > 0$，
$$\arg\min_{a \in \mathcal{A}(s)} (\alpha c)^\top a = \arg\min_{a \in \mathcal{A}(s)} c^\top a$$

证明极简：$(\alpha c)^\top a = \alpha (c^\top a)$，正数乘法保序，minimizer 集合不变。

**重要推论**：solver output 只依赖 $c$ 的 **direction**，不依赖 magnitude。所以 cost vector 的 magnitude 是冗余自由度，可以 fix 到 unit norm：

$$\mathcal{C} = \mathbb{S}^{m-1} := \{c \in \mathbb{R}^m : \|c\|_2 = 1\}$$

### 3.2 为什么 sphere 比 Euclidean 空间好

如果在 $\mathbb{R}^m$ 上学 flow policy，模型需要同时学 magnitude 和 direction。但 magnitude 对最终 action 没有影响，浪费模型容量，且引入额外的 variance。直接在 sphere 上学：
- **Geometry matches the problem**：solver 的 invariance 和 sphere 的 quotient 自然对应
- **Compact latent space**：dimension 一样 ($m-1$)，但去掉了 radial 冗余
- **Flow matching on Riemannian manifold 有现成理论**（Chen & Lipman 2024）

Ablation study（Fig. 4 left）证实：换成 Euclidean flow + Gaussian smoothing，performance 在 Dynamic Scheduling 和 Dynamic Routing 上显著掉。

---

## 4. Spherical Flow Matching：怎么在 Sphere 上学分布

### 4.1 Flow Matching 背景

Flow matching（Lipman et al. 2023; Liu et al. 2023 rectified flow）学一个 time-dependent velocity field $v_\theta(x_t, t)$，从 base distribution $p_0$ transport 到 target $p_1$：

$$\frac{dx_t}{dt} = v_\theta(x_t, t), \quad t \in [0, 1] \quad \text{(Eq. 1)}$$

训练目标：让 $v_\theta$ regress 到 target velocity $u_t$，$u_t = \frac{d}{dt}\phi_t(x_0, x_1)$，其中 $\phi_t$ 是预先选定的 interpolation。最常见 linear interpolation $\phi_t = (1-t)x_0 + t x_1$，$u_t = x_1 - x_0$。

vs. Diffusion (SDE)：flow 是 ODE，没有噪声项，更简单，采样更快，已在 SD3、Flux 等大模型上证明 competitive。

参考资料：
- Flow Matching for Generative Modeling: https://openreview.net/forum?id=PqvMRDCJT9t
- Rectified Flow: https://openreview.net/forum?id=XVjTT1nw5z
- Flow Matching Guide: https://arxiv.org/abs/2412.06264

### 4.2 Spherical 上的 Flow

Sphere 是 Riemannian manifold，需要做两件事：

**1. Geodesic interpolation**（Eq. 9）：
$$c_t = \frac{\sin((1-t)\theta)}{\sin\theta} c_0 + \frac{\sin(t\theta)}{\sin\theta} c_1, \quad \theta = \arccos(c_0^\top c_1)$$

变量：
- $c_0, c_1 \in \mathbb{S}^{m-1}$：base 和 target 样本
- $\theta$：geodesic distance（angle between $c_0$ and $c_1$）
- $t \in [0,1]$：interpolation parameter
- $c_t$：沿大圆（great circle）的插值，保证 $\|c_t\|_2 = 1$

这是 sphere 上的 SLERP (spherical linear interpolation)，源自 quaternion interpolation / Shoemake 1985。

**2. Projected vector field**（Eq. 3）：
$$\frac{dc_t}{dt} = \Pi_{c_t} v_\theta(c_t, s, t), \quad \Pi_c = I - cc^\top$$

变量：
- $v_\theta \in \mathbb{R}^m$：原始 velocity field（可以任意）
- $\Pi_c = I - cc^\top$：orthogonal projection 到 $c$ 处的 tangent space
- $\Pi_{c_t} v_\theta$：projected velocity，保证 $c_t$ 不离开 sphere

**Intuition**：任何 vector field $v$ 都可以分解成 radial component $(c^\top v) c$ 和 tangential component $v - (c^\top v)c = \Pi_c v$。Radial 部分会把 $c$ 推离 sphere，所以扔掉。

ODE 积分时用 Heun (RK2) 30 steps，base distribution $p_0$ 是 uniform on sphere。

**为什么 flow 不 diffusion**（Remark 3.3）：作者说 spherical diffusion 实现复杂，且 spherical flow matching 在 Chen & Lipman 2024 实证上更好。这也合理 —— diffusion 在 manifold 上需要处理 Brownian motion，score function 在 manifold 上的 SDE 推导比较麻烦。

---

## 5. Expressivity：能表达任何 stochastic policy 吗？

**Proposition 3.2**：对任意 target distribution $\mu$ over feasible actions $\mathcal{A}(s)$，存在一个分布 $\pi$ over $\mathbb{S}^{m-1}$ 使得 pushforward $a^*(s, c)$ 的分布恰好是 $\mu$。

证明思路（Appendix F）：
1. 对每个 $a \in \mathcal{A}(s)$，定义 action region $\mathcal{C}_a := \{c \in \mathbb{S}^{m-1} : a^*(s, c) = a\}$
2. 用 tie-breaking 后 $\{\mathcal{C}_a\}$ 是 sphere 的 partition
3. 每个 $\mathcal{C}_a$ 都有正 surface measure（因为每个 feasible action 都是 polytope 的 vertex，存在 separating hyperplane，normal cone 内有 open subset）
4. 把 sphere surface measure 在 $\mathcal{C}_a$ 上 normalize 成 $\pi_a$
5. 混合 $\pi = \sum_a \mu(a) \pi_a$ 即可

**关键 insight**：这个证明揭示了 sphere 上有一个 piecewise constant 的 partition，每个 cell 对应一个 feasible action。这正是 solver 的几何结构 —— 它把 sphere 切成 normal fan regions。这个 partition 是后面 discontinuity 问题的根源。

---

## 6. Latent Cost Space Critic：避免 Solver 反复调用

### 6.1 计算瓶颈分析

Naive 实现：
- Policy update 时 sample 一堆 candidate $c$，对每个 evaluate $Q(s, a^*(s, c))$
- 每个 evaluate 都要 call solver 一次
- Solver 是计算瓶颈（MIP NP-hard）

LSFLOW 的 trick：直接在 latent space 训练 critic：
$$\widetilde{Q}_\phi(s, c) := Q(s, a^*(s, c)) \quad \text{(Eq. 5)}$$

变量：
- $\widetilde{Q}_\phi$：neural network，输入 $(s, c)$，输出 scalar
- $Q$：理论上的 action-value function
- $a^*(s, c)$：solver output

$\widetilde{Q}_\phi$ 用 environment reward 训练。每次 environment step 只 call 一次 solver（执行 action），policy update 时只用 $\widetilde{Q}_\phi(s, c)$，不再 call solver。

Ablation 结果：action-space critic 一步 policy update ~6 分钟，cost-space critic <1 秒。这是 360x 的提升，是方法可行性的关键。

### 6.2 但这引出新问题

$\widetilde{Q}(s, c) = Q(s, a^*(s, c))$ 作为 $c$ 的函数是 **piecewise constant**：在 $\mathcal{C}_a$ 内部是常数 $Q(s, a)$，跨过 boundary 跳到 $Q(s, a')$。

可视化见 Fig. 2a：sphere 被切成 colored regions，每个对应一个 action。

这个 piecewise-constant 函数：
- 不连续 → Bellman target 在 boundary 附近剧烈跳变
- Gradient 无定义（a.e. 为 0，boundary 处无穷）
- 神经网络拟合困难，bootstrap 不稳定

---

## 7. vMF Smoothing：Smoothed Bellman Operator

### 7.1 von Mises–Fisher Kernel

vMF 是 sphere 上的 analog of Gaussian：
$$K_\kappa(\tilde{c} | c) \propto \exp(\kappa c^\top \tilde{c}), \quad c, \tilde{c} \in \mathbb{S}^{m-1} \quad \text{(Eq. 6)}$$

变量：
- $\kappa > 0$：concentration parameter（类比 inverse temperature / 1/方差）
- $c$：center direction
- $\tilde{c}$：perturbed direction
- $c^\top \tilde{c} = \cos\theta$：two directions 的 cosine similarity

行为：
- $\kappa \to 0$：uniform on sphere（最大模糊）
- $\kappa \to \infty$：Dirac at $c$（无模糊）
- 中等 $\kappa$：在 $c$ 附近 cone 内的分布

类比：$\mathbb{R}^d$ 上的 Gaussian $\propto \exp(-\|x-\mu\|^2/(2\sigma^2))$，sphere 上用 cosine distance 替代 Euclidean distance，$\kappa \sim 1/\sigma^2$。

### 7.2 Smoothed Bellman Operator

**定义**（Eq. 7）：
$$(\mathcal{T}_\kappa^\pi Q)(s, c) = \mathbb{E}\left[r(s, a^*(s, \tilde{c})) + \gamma Q(s', \tilde{c}')\right]$$

其中：
- $\tilde{c} \sim K_\kappa(\cdot | c)$：当前 cost 被 perturb 后再送进 solver
- $s' \sim P(\cdot | s, a^*(s, \tilde{c}))$：next state
- $c' \sim \pi(\cdot | s')$：next policy sample
- $\tilde{c}' \sim K_\kappa(\cdot | c')$：next cost 也被 perturb 再 evaluate critic

**两个地方都 smooth**：
1. 当前 step：用 $\tilde{c}$ 而不是 $c$，对多个 nearby directions 取期望
2. Bootstrap：用 $\tilde{c}'$ 而不是 $c'$，next-state critic evaluation 也 smooth

效果：piecewise constant 的 $Q(s, a^*(s, \cdot))$ 被卷积成一个 smooth function $\widetilde{Q}_\kappa(s, c)$（Fig. 2b）。

### 7.3 理论保证（Theorem 3.5）

**三个结论**：
1. **$\gamma$-contraction**：在 $B_\infty$（bounded functions with sup norm）上，$\mathcal{T}_\kappa^\pi$ 是 $\gamma$-contraction → Banach fixed-point theorem → 存在唯一 fixed point $Q_\kappa^\pi$

2. **$C^\infty$ smoothing in $c$**：对任意 $Q \in B_\infty$，$(\mathcal{T}_\kappa^\pi Q)(s, \cdot) \in C^\infty(\mathbb{S}^{m-1})$

3. **Fixed point regularity**：$Q_\kappa^\pi(s, \cdot) \in C^\infty(\mathbb{S}^{m-1})$

**证明关键步骤**（Appendix G）：

Step 1：直接计算 $\|\mathcal{T}_\kappa^\pi Q_1 - \mathcal{T}_\kappa^\pi Q_2\|_\infty \leq \gamma \|Q_1 - Q_2\|_\infty$（reward 项 cancel，剩下 bootstrap 项 bounded by $\gamma$）

Step 2：把 operator 写成积分形式
$$(\mathcal{T}_\kappa^\pi Q)(s, c) = \int_{\mathbb{S}^{m-1}} G_s(\tilde{c}) K_\kappa(\tilde{c} | c) d\sigma(\tilde{c}) \quad \text{(Eq. 11)}$$

$G_s(\tilde{c}) = \mathbb{E}[r + \gamma Q(s', \tilde{c}') | \tilde{c}]$，bounded。

关键：$K_\kappa(\tilde{c} | c) = \exp(\kappa c^\top \tilde{c}) / Z$ 在 $c$ 上是 $C^\infty$，且 $\mathbb{S}^{m-1} \times \mathbb{S}^{m-1}$ compact → 所有导数一致有界 → dominated convergence 允许在积分号下求导 → $C^\infty$。

Step 3：fixed point $Q_\kappa^\pi = \mathcal{T}_\kappa^\pi Q_\kappa^\pi$，由 Step 2 是 $C^\infty$。

**Intuition**：vMF kernel 在 sphere 上是 heat kernel 的类似物，对任意 bounded function 做卷积都得到 smooth function。Bellman operator 内部嵌入了一个 smoothing operator，每次 backup 都自动 smooth，迭代到 fixed point 也保持 smoothness。

### 7.4 Smoothing 的 Bias-Variance Trade-off

Ablation（Fig. 4 right）：
- None（$\kappa \to \infty$，不 smooth）：critic target noisy，不稳定，性能差
- 中等 $\kappa$：最佳
- 过小 $\kappa$（过度 smooth）：value landscape 被抹平，引入 bias，性能下降

实际超参 $\kappa = 28$（benchmark）/ 40-60（STI），$\kappa$ 越大越尖锐。$\kappa$ 是 critical hyperparameter，可能需要 adaptive schedule（作者在 Limitations 中提到）。

---

## 8. Weighted Flow Matching = KL-Regularized Policy Improvement

### 8.1 Weighted Objective

**Importance weight**：
$$w(s, c) \propto \exp\left(\frac{1}{\lambda} Q(s, a^*(s, c))\right)$$

变量：
- $Q$：critic value（实际用 $\widetilde{Q}_\phi$）
- $\lambda > 0$：temperature，控制 update aggressiveness（小 λ 强 reweighting）

**Weighted Flow Matching Loss**（Eq. 4）：
$$\mathcal{L}(\theta) = \mathbb{E}\left[w(s, c_1) \|\Pi_{c_t} v_\theta(c_t, s, t) - u(c_t, s, t)\|_2^2\right]$$

期望 over $s \sim \mathcal{D}$（replay buffer），$c_1 \sim \pi_k(\cdot | s)$（旧 policy），$c_0 \sim p_0$（base），$t \sim U(0,1)$。

### 8.2 等价于 KL-Regularized Update

**Theorem H.1**：设 $w_k \geq 0$，$Z_k(s) = \mathbb{E}_{c \sim \pi_k}[w_k(s,c)] \in (0, \infty)$。定义 reweighted distribution
$$\pi_{k+1}(c | s) := \frac{1}{Z_k(s)} \pi_k(c | s) w_k(s, c) \quad \text{(Eq. 13)}$$

那么：
**(i)** Minimizing Eq. (4) 等价于用 unweighted flow matching 拟合 $\pi_{k+1}$（因为 $Z_k(s)$ 不依赖 $\theta$，乘上去不改变 minimizer）

**(ii)** 当 $w_k \propto \exp(Q/\lambda)$ 时：
$$\pi_{k+1}(c|s) \propto \pi_k(c|s) \exp\left(\frac{1}{\lambda} Q(s, a^*(s,c))\right)$$

恰好是 KL-regularized policy improvement 的 closed-form 解：
$$\pi_{k+1}(\cdot | s) \in \arg\max_\pi \mathbb{E}_{c \sim \pi}[Q(s, a^*(s,c))] - \lambda \mathrm{KL}(\pi \| \pi_k) \quad \text{(Eq. 15)}$$

**Intuition**：这和 DPO（RLHF）的 trick 极其相似 —— 不显式优化 KL-regularized objective，而是用 importance reweighting + supervised regression 实现 implicit optimization。这种 trick 在 diffusion policy（Ma et al. 2025a "Efficient Online RL for Diffusion Policy"）也用过。

参考资料：
- Mirror Descent Policy Optimization (Tomar et al. 2022): https://openreview.net/forum?id=aBO5SvgSt1
- Efficient Online RL for Diffusion Policy: https://openreview.net/forum?id=6Anv3KB9lz
- DPO paper: https://arxiv.org/abs/2305.18290

**KL term 的作用**：trust region，类比 PPO / TRPO，防止 policy update 步子太大崩溃。$\lambda$ schedule 从 2.0 退火到 0.8（benchmark）或 0.5（syphilis），逐渐收紧 trust region 同时 exploit 更精细。

### 8.3 实际 Trick

按 Ma et al. 2025a 的两个 stabilizer：
1. **Multiple candidate selection**：sample 多个 candidate $c$，选 critic value 最高的执行（不是 single sample）
2. **EMA on weights**：用 exponential moving average 平滑 flow-matching weights，减小 variance

---

## 9. Algorithm 1 完整流程

```
输入: λ, κ, γ, J (smoothing size), η (target update rate), solver a*
初始化: replay buffer D, actor v_θ, critic Q̃_φ, target Q̃_φ̄

for k = 1, 2, ...:
    # Environment step
    s ← s'
    c ~ π_θ(·|s)  # 通过 integrate Eq. 3 采样
    c̃ ~ K_κ(·|c)  # vMF 扰动
    a ← a*(s, c̃)  # 调用 solver
    执行 a, 观测 r, s'
    存储 (s, c, r, s') 到 D
    
    # Critic update
    从 D 采样 minibatch
    c' ~ π_θ(·|s')
    {c̃'^(j)}_{j=1}^J ~ K_κ(·|c')
    y_κ = r + γ (1/J) Σ_j Q̃_φ̄(s', c̃'^(j))  # Eq. 8
    min_φ ||Q̃_φ(s, c) - y_κ||_2^2
    
    # Actor update
    从 D 采样 s, c_1 ~ π_θ(·|s)
    w(s, c_1) ← exp((1/λ) Q̃_φ̄(s, c_1))
    min_θ weighted flow matching loss (Eq. 4)
    
    # Target update
    φ̄ ← (1-η) φ̄ + η φ
```

**关键参数**（Table 2）：
- $K=12$ particles（候选数）
- $J=1$ vMF perturbation per candidate
- $\kappa = 28$（benchmark）
- Weight clip $w_{\max} = 4$（防止某些样本主导）
- Q-norm clip 3.0
- 36 flow steps
- Heun integrator, 30 sampling steps

**Critic target**（Eq. 8）只 sample $J=1$ 个 perturbation，是 Monte Carlo estimate。$J=1$ 已经足够（vMF 期望在 $K$ candidates 上也 implicit averaged），更大 $J$ 增加计算。

---

## 10. 实验：Benchmark Suite

### 10.1 四个 Task

1. **Dynamic Scheduling**：N=40 volunteers，J patients，K time slots，budget B=10，horizon H=20。MIP 形式（Eq. 16）：决策变量 $x_{ijk} \in \{0,1\}$ 表示 worker i 在 slot k 服务 patient j。

2. **Dynamic Routing**：London tube network（Fig. 5）。State 是 graph 上的 node，action 是 bounded-length cycle 从 source 出发并返回。决策变量 $f_{j,k,t}$ 表示 timestep t 是否覆盖 edge (j,k)，$a_j$ 是否过 node j（Eq. 17）。Max length T=2B。

3. **Dynamic Assignment**：NP-hard generalized assignment problem。Worker capacity $b_i$，serving patient j cost $c_j$。决策变量 $x_{ij}$（Eq. 18）。

4. **Dynamic Intervention**：public health，sigmoid link 响应（Eq. 20-21），S-shape dose-response（Levy et al. 2006 smoking cessation model）。Budget 约束（Eq. 19）。

### 10.2 Results Table（Table 1）

| Method | Dyn. Sched. | Dyn. Routing | Dyn. Assign. | Dyn. Interv. | Avg | Avg Time (h) |
|--------|---|---|---|---|---|---|
| Random | 10.12±0.84 | 11.11±0.36 | 13.28±0.80 | 8.91±0.08 | 10.85 | - |
| Greedy | 15.12±1.56 | 11.58±0.46 | 20.15±1.00 | 10.55±0.67 | 14.35 | - |
| DQN-Sampling | 16.89±0.95 | 18.29±1.49 | 22.25±2.27 | 15.16±0.25 | 18.15 | 0.52 |
| SEQUOIA | 24.00±1.44 | 20.99±1.92 | 32.99±3.45 | 10.84±0.89 | 22.21 | 9.11 |
| SRL | 24.50±1.09 | 25.32±1.49 | 28.46±0.82 | 13.36±1.93 | 22.91 | 3.89 |
| **LSFLOW** | **28.85±1.48** | **28.51±2.60** | **35.93±2.71** | **17.21±0.39** | **27.62** | **1.29** |

**分析**：
- LSFLOW 在所有 4 个 task 上都最好
- 比 SRL 平均提升 20.6%（27.62 vs 22.91）
- 训练时间 1.29h vs SRL 3.89h（3x 加速）vs SEQUOIA 9.11h（7x 加速）
- Dynamic Intervention 上提升最大：17.21 vs 13.36（28.8%），说明 stochastic policy 在多模态 landscape 下 explore 更好
- Dynamic Assignment SEQUOIA 已经很强（32.99），LSFLOW 仍提升 ~9%（35.93）
- SRL 在 Routing 表现好（25.32）但 Assign 弱（28.46），可能 deterministic policy 在 capacity-constrained assignment 下容易陷入局部最优

**为什么 LSFLOW 快**：
- SEQUOIA 每次 greedy action 都要 solve MIP，MIP 规模随 network 增长
- SRL 训练时对每个 cost vector sample 多个 perturbation，每个都要 solve 一次 CO problem（differentiate through solver）
- LSFLOW 每次 env step 只 solve 一次（执行 action 时），policy update 完全在 latent space

---

## 11. STI Testing 真实应用

### 11.1 任务设定

Sexually transmitted infection testing on real contact networks。ICPSR public dataset（Morris & Rothenberg 2011），四种 disease：Chlamydia, Gonorrhea, HIV, Syphilis。

**MDP 形式化（B-AFEG, Definition J.1）**：
- $G = (V, E)$：sexual contact graph，nodes = individuals
- $Y_v \in \{0,1\}$：true infection status，hidden，从 prior $P$ 采样（Markov wrt $G$）
- State $s_t = X_t \in \{-1, 0, 1\}^V$：每个 node 的 testing status（-1 untested, 0 negative, 1 positive）
- Action $A_t \subseteq \mathcal{F}(X_t)$, $|A_t| \leq B$：选一批 frontier node 测试
- Frontier rule：只能在已测 node 的邻居或 component root 上测（adaptive frontier exploration）
- Reward $R_t = \sum_{v \in A_t} r(Y_v)$，发现 positive 给 reward

### 11.2 Frontier Definition

$$\mathcal{F}(X_t) = \{\rho_j : S_t \cap \mathcal{C}_j = \emptyset\} \cup \{v : X_t(v) = -1 \text{ and } \exists u \in S_t, (u,v) \in E\}$$

- $\rho_j$：component $j$ 的 root（按 marginal P(+)=highest 选）
- $S_t$：已测 nodes
- $\mathcal{C}_j$：第 $j$ 个 connected component

Intuition：每个 component 第一次测必须测 root；之后只能测已测 node 的 untested 邻居。模拟 realistic network-based testing（WHO 2021 推荐）。

### 11.3 Graph Statistics（Table 3）

| Disease | Nodes | Edges | Positive (%) | Components |
|---------|-------|-------|---|---|
| Chlamydia | 100 | 63 | 44.0 | 37 |
| Gonorrhea | 100 | 68 | 9.0 | 32 |
| HIV | 100 | 70 | 27.0 | 37 |
| Syphilis | 101 | 102 | 18.8 | 27 |

### 11.4 实验细节

- Budget $B=5$ per round
- Critic 用 GIN（Graph Isomorphism Network, Xu et al. 2019），3 layers, hidden 128
- 不比较 SEQUOIA（因为 SEQUOIA 限 MLP-based critic，无法处理 graph）
- Hyperparameter 略有调整（Table 4）：syphilis λ end 0.5, steps 30000；$\kappa$ chlamydia/HIV 60，syphilis/gonorrhea 40

### 11.5 结果分析（Fig. 3）

Plot 是 detection efficiency curve（detected positives vs testing fraction）。LSFLOW 在低-中 budget regime 提升最显著，这个 regime operationally 最有价值（资源稀缺时 targeted testing 收益最大）。

- **Chlamydia, Gonorrhea**：LSFLOW 显著优于 SRL
- **HIV**：和 SRL 持平
- **Syphilis**：早期 LSFLOW 更强，晚期 SRL 追平

**为什么 Syphilis 早期 LSFLOW 强**：Syphilis 有 27 components，frontier constraint 很严格（每个 component 必须 first test root）。Stochastic policy 能更好 explore 不同 component 的 root 顺序。后期 testing fraction 大时，几乎所有 component 都已被开过，frontier 限制消失，deterministic SRL 也能找到 optimal。

参考资料：
- GIN paper: https://openreview.net/forum?id=ryGs6iA5Km
- Adaptive Frontier Exploration on Graphs (Choo et al. 2025): https://openreview.net/forum?id=mUJU8LmhZY
- ICPSR HIV data: https://www.icpsr.umich.edu/

---

## 12. Ablation Studies 总结

### 12.1 Spherical vs Euclidean（Fig. 4 left）

去掉 spherical constraint，用 Euclidean flow + Gaussian smoothing。Dynamic Scheduling 和 Dynamic Routing 性能掉。证明 sphere geometry 的必要性 —— 不仅仅是个 trick，而是和 solver 的 invariance 严格对应。

### 12.2 Smoothing Strength（Fig. 4 right）

Sweep $\kappa$：
- None：最差
- 中等：最佳
- 过强：bias 增大，掉

经典 bias-variance trade-off。$\kappa$ 是 critical hyperparameter，可能需要 state-dependent 或 adaptive schedule。

### 12.3 Latent-Space vs Action-Space Critic

Action-space critic 一步 policy update ~6 min，cost-space <1s。证明 latent-space critic 不仅是 nice-to-have，是 method tractable 的必要条件。

---

## 13. Limitations 和 Future Directions

1. **仅限 linear-objective solver**：很多 CO 问题是 quadratic objective、semi-definite program 等。Extension 到 parametric constraints、multi-objective、robust formulations 是 open direction。

2. **Smoothing 引入 bias**：fixed $\kappa$ 是 trade-off。Adaptive / state-dependent $\kappa$ schedule 可能进一步提升。

3. **Solver warm-starting**：每次 call solver 都 cold start。如果用 heuristic 或 learning-based warm start，runtime 可能进一步降低。

4. **高维 sphere 的 vMF sampling**：$\mathbb{S}^{m-1}$ 当 $m$ 大时 vMF sampling 可能 numerically tricky（normalization constant 涉及 modified Bessel function）。

5. **Theoretical analysis 的拓展**：当前只在固定 policy 下证明 contraction。Optimality guarantee、sample complexity、和 solver 性质（PSPICE structure）的关系都未探索。

6. **Connection to imitation learning**：如果有 expert demonstrations，可以用 flow matching 的 conditional variants 学 cost distribution 而不依赖 reward。类似 Diffusion BC + RL 的 hybrid。

7. **Multi-agent extension**：combinatorial action 在 multi-agent setting 下更复杂，每个 agent 的 feasible set 相互依赖。LSFLOW 的 latent space 思路可能可以和信息传递结合。

---

## 14. 与其他工作的关联（My联想）

### 14.1 和 Decision-Focused Learning 的关系

Decision-Focused Learning（Wilder et al. 2019; Elmachtoub & Grigas 2022）也是 predict-then-optimize，但通常 one-shot。LSFLOW 是 sequential decision 的 DFL，每个 timestep 都 predict cost vector 然后 optimize。

参考资料：
- Smart Predict-then-Optimize: https://pubsonline.informs.org/doi/10.1287/mnsc.2021.3982
- Decision-Focused Learning: https://arxiv.org/abs/1902.05924

### 14.2 和 Differentiable Optimization 的关系

Pogancic et al. 2020（black-box combinatorial solver differentiation）、Berthet et al. 2020（perturbed optimizer with implicit differentiation）、Ferber et al. 2020（MIPaaL）都用某种 surrogate gradient 让 solver 可微。LSFLOW 选择 **不** differentiate through solver，而是把 solver 当 black box decoder，在 latent space 训 critic 来 amortize。这是 architectural choice 的不同。

参考资料：
- Diff through black-box combinatorial solvers: https://openreview.net/forum?id=BkevoJSYPB
- Perturbed optimizers: https://arxiv.org/abs/2002.08676

### 14.3 和 DPO / RLHF 的关系

Weighted flow matching ↔ KL-regularized improvement 的对应完全 analogous to DPO 的 derivation：
- DPO：用 closed-form optimal policy under KL constraint，把 RL objective 变成 supervised regression on preference pairs
- LSFLOW：用 closed-form optimal policy under KL constraint，把 policy improvement 变成 weighted regression on samples from old policy

两者都避免了 explicit KL 优化和 importance sampling 的高方差。

### 14.4 和 Mixture of Experts 的隐含联系

Sphere partition $\{\mathcal{C}_a\}$ 把 latent space 切成 cells，每个 cell 对应一个 feasible action。这和 Mixture of Experts 的 routing 极其相似：solver 是一个 hard router（piecewise constant）。vMF smoothing 是 soft routing（每个 expert 加权）。

### 14.5 和 Optimal Transport 的关系

Flow matching 本身是 OT-based generative model。Sphere 上的 flow matching 可以看作 sphere 上 OT map 的 continuous-time relaxation。Solver-induced policy 的 pushforward $\pi \to \mu$ 是一个特殊的 transport plan，由 solver geometry 决定 cell 结构。

### 14.6 和 Convex Geometry / Normal Fan 的联系

Proposition 3.2 的证明用到了 normal fan / normal cone 的概念。$\mathcal{C}_a$ 实际上是 polytope $\mathrm{conv}(\mathcal{A})$ 在 vertex $a$ 处的 normal cone 和 sphere 的 intersection。这是 polyhedral geometry 的标准对象，参考 Ziegler "Lectures on Polytopes"。

参考资料：
- Ziegler, Lectures on Polytopes: https://link.springer.com/book/10.1007/978-1-4613-8431-1

### 14.7 和 Cell-based Value Function 的关系

Cost-space critic 的 piecewise constant 结构让人想起 cell decomposition 在 motion planning 中的应用。每个 cell 对应一个 action，价值函数在 cell 内 constant。Smoothed version 类似 homogenized value function。

### 14.8 和 Reinforcement Learning with Expert Demonstrations

如果存在 expert cost vectors（比如 IL 提供），可以 pretrain flow policy 在 expert cost 上做行为克隆，然后用 LSFLOW 的 RL fine-tune。类似 Diffusion BC + RL pipeline in robotics。

---

## 15. Critical Thoughts / 我自己的看法

### 15.1 Sphere 的选择非常优雅

Lemma 3.1 是个 trivial observation（positive scaling 不改变 argmin），但作者从中推出整个 framework 应该建在 sphere 上。这种 "从 invariance 推 geometry" 的思路在数学物理中很常见（gauge theory），在 ML 里也很有 power。值得 internalize。

### 15.2 Smoothed Bellman Operator 是核心 contribution

Latent cost space critic 的 piecewise-constant 结构是个 fundamental 障碍。如果没有 smoothing，Bellman backup 会在 cell boundary 附近 jitter。vMF smoothing 是个 principled fix：sphere 上有 closed-form kernel，convolution with $C^\infty$ kernel 给 $C^\infty$ output。Theorem 3.5 严谨证明了 contraction + smoothness，给 practical training 提供理论 anchor。

### 15.3 Weighted Flow Matching 的统一视角

把 weighted regression 和 KL-regularized improvement 联系起来，让 LSFLOW 继承了 PPO/TRPO/MDPO 的稳定性分析。这和 Ma et al. 2025a 在 diffusion policy 上做的类似。Flow matching 在这个 sense 下不比 diffusion 弱，反而因为 ODE 形式更简单。

### 15.4 Solver 作为 decoder 的成本

每次 env step 仍然要 call solver（虽然是 amortized）。对 LP 还好，对 NP-hard MIP 在大规模 instance 上仍可能慢。论文实验是 N=40，real-world instance 可能 N=10^4 或更大。Future work 中 solver warm-starting 很关键。

### 15.5 关于 $\kappa$ 的选择

Ablation 显示 $\kappa$ 是 critical。Benchmark $\kappa=28$，STI $\kappa=40-60$。这个值的合理范围和 problem structure 的关系是什么？直觉上，$\kappa$ 应该和 cell 大小相关 —— 如果 sphere 上 cells 很小（feasible actions 多），需要大 $\kappa$ 才能区分。一个粗略 estimate：cells 数量 $|\mathcal{A}| \sim e^{mH}$（$H$ 是 entropy per dim），cell angular size $\sim |\mathcal{A}|^{-1/(m-1)}$，需要 $\kappa \gtrsim 1/\text{cell size}$。

### 15.6 关于 Exploit 的 question

Multiple candidate selection（K=12 particles 选 critic value 最高）和 EMA on weights 是 exploit 技巧。和 KL regularization 的 explore 角色形成 balance。这种 trick 在 diffusion RL 中也常见。

### 15.7 推广到 Continuous Action RL

虽然论文聚焦 combinatorial，但 cost-direction + sphere 的思路对 continuous action RL 也有启发。如果 action 通过某个 deterministic decoder（比如 IK、motion planner）从 latent 生成，类似 trick 可以用。Latent space RL + decoder 的架构在 robotics 中有大量应用场景。

---

## 16. 一句话总结

LSFLOW 把 combinatorial RL 的 feasibility 委托给 CO solver，stochasticity 委托给 spherical flow policy。三个核心 trick 让它 work：
1. **Sphere geometry**（Lemma 3.1 的 invariance）
2. **Latent cost space critic**（amortize solver）
3. **vMF smoothed Bellman operator**（处理 solver-induced discontinuity）

加上 weighted flow matching 和 KL-regularized policy improvement 的等价性，给出一个 tractable、stable、expressive 的 combinatorial RL framework。Empirically 比 SRL 平均提升 20.6%，同时训练快 3x。

参考资料总结：
- 主页 paper (arxiv 应该有): https://arxiv.org/abs/2502.08101 (预估)
- Flow Matching Guide: https://arxiv.org/abs/2412.06264
- Sphere Flow Matching: https://openreview.net/forum?id=g7ohDlTITL
- SEQUOIA: https://openreview.net/forum?id=DhH3LbA6F6
- SRL: https://openreview.net/forum?id=GS9o7u5njS
- MDPO: https://openreview.net/forum?id=aBO5SvgSt1
- GIN: https://openreview.net/forum?id=ryGs6iA5Km
- Diffusion Policy Optimization: https://openreview.net/forum?id=mEpqHvbD2h
- Black-box solver differentiation: https://openreview.net/forum?id=BkevoJSYPB
- AFEG (Choo et al. 2025): https://openreview.net/forum?id=mUJU8LmhZY
- ICPSR HIV dataset: https://www.icpsr.umich.edu/icpsrweb/ICPSR/studies/3311
- WHO HIV guidelines: https://www.who.int/publications/i/item/9789240031593
- Convex Optimization (Boyd): https://web.stanford.edu/~boyd/cvxbook/
- Lectures on Polytopes (Ziegler): https://link.springer.com/book/10.1007/978-1-4613-8431-1
- Diffusion Policy (Chi et al.): https://arxiv.org/abs/2303.04137
- DPO: https://arxiv.org/abs/2305.18290
- PPO: https://arxiv.org/abs/1707.06347
- TRPO: https://arxiv.org/abs/1502.05477
- Flow Q-learning: https://openreview.net/forum?id=KVf2SFL1pi
- Flow Matching Policy Gradients: https://arxiv.org/abs/2507.21053
- ReinFlow: https://openreview.net/forum?id=ACagRwCCqu
- Diffusion actor-critic with entropy regulator: https://openreview.net/forum?id=l0c1j4QvTq
- Q-score matching: https://arxiv.org/abs/2407.01349
- GenCO (Ferber et al. 2024): https://arxiv.org/abs/2410.19754
- MIPaaL (Ferber et al. 2020): https://arxiv.org/abs/1909.00514
- DFL survey: https://arxiv.org/abs/2402.05226
- TorchRL: https://arxiv.org/abs/2306.00577
- Adam: https://arxiv.org/abs/1412.6980
- VAE: https://arxiv.org/abs/1312.6114
- GAN: https://arxiv.org/abs/1406.2661
- DDPM: https://arxiv.org/abs/2006.11239
- Sohl-Dickstein diffusion: https://arxiv.org/abs/1503.03585
- Rectified Flow: https://arxiv.org/abs/2209.03003
- SD3 / Rectified Flow Transformers: https://arxiv.org/abs/2403.03206
- von Mises-Fisher distribution: https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution
- Spherical interpolation (SLERP): https://en.wikipedia.org/wiki/Slerp
- Linear Programming and Extensions (Dantzig): https://press.princeton.edu/books/paperback/9780691059136/linear-programming-and-extensions
- Differentiable Perturbed Optimizers: https://papers.nips.cc/paper/2020/hash/e634d31530d8f5681dabc1f0d1a0c4c9-Abstract.html
- Fenchel-Young losses: https://jmlr.org/papers/v21/19-028.html
- DQN-Sampling (He et al. 2016): https://arxiv.org/abs/1606.03667
- Smoking cessation simulation (Levy et al.): https://pubmed.ncbi.nlm.nih.gov/16501649/
- Bees Algorithm for GAP (Ozbakir et al. 2010): https://www.sciencedirect.com/science/article/pii/S0096300309007482
- Vehicle Routing RL (Delarue et al. 2020): https://arxiv.org/abs/2010.13601
- Cooperative MARL in combinatorial spaces (Tkachuk et al. 2023): https://proceedings.mlr.press/v206/tkachuk23a.html
- Dulac-Arnold large discrete actions: https://arxiv.org/abs/1512.07679
- Flow matching for RL: https://openreview.net/forum?id=KVf2SFL1pi
- Q-weighted variational policy optimization: https://openreview.net/forum?id=UWUUVKtKeu
- GenPO: https://openreview.net/forum?id=BmRNz1TpCc
- Discrete diffusion policies for combinatorial actions (Ma et al. 2025b): https://arxiv.org/abs/2509.22963

---

希望这个深度解析能 build 你的 intuition。最值得 internalize 的几个 takeaway：
1. **Invariance 推 geometry**：从 solver 的 scale invariance 自然推出 sphere geometry
2. **Amortization 思路**：把 expensive operation（solver）amortize 到 critic 学习
3. **Smoothing as regularization**：用 manifold-appropriate kernel（vMF on sphere）smoothing discontinuous operator，得到 well-posed learning problem
4. **Weighted regression = KL improvement**：这个 trick 在 generative RL 中反复出现，是统一 framework

如果你对某个具体细节（比如 spherical flow matching 的 implementation、vMF sampling、MIP formulation 等）想进一步 dive in，告诉我具体哪一块，我可以再展开。
