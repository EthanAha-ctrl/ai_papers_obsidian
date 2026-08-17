---
source_pdf: Reward-Respecting Subtasks.pdf
paper_sha256: d49a3a15ccdbdddb4a4113265c84775d98a2e12536aec0538fc29b992f3d61dc
processed_at: '2026-08-11T23:49:09-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说这篇 paper

## 一句话版本

Sutton 说：**别再搞那些跟任务奖励无关的"技能发现了"**，学出来的技能在规划时根本用不上。让技能一边追逐真实奖励，一边在"某个特征值高、且本身也值钱"的地方停下来——这样学出来的技能，planner 才会真的去用它。

---

## 问题是什么

想象你住在一个城市，要从家到公司。路上有个必经的十字路口（bottleneck）。

以前的方法说："学一个技能：用最短路径走到那个十字路口。"

但最短路径中间穿过一片收费区，每走一步罚 1 块钱。你学出来的技能是"直线冲过去"，但你的最优策略是"绕开收费区"。

结果呢？planner 拿着这个"直线冲"的技能模型去规划，发现它几乎从不是最优选择——**白学了**。更讽刺的是，带上这个没用的技能模型，planner 比只用原始动作还慢，因为每步多算一次没用的 look-ahead。

Sutton 说：这不对。技能发现不应该脱离任务本身。

---

## Sutton 的方案

**技能在跑的过程中，拿的是真实奖励**，不是什么内在奖励、不是 -1 每步、不是 Laplacian eigenvector。

**技能停下来的地方，要有奖励**——但不是随便停，而是停在"某个特征值高 + 这个 state 本身估计价值也高"的地方。

具体来说，你的 value function 是线性的：$v(s) = w_1 x_1 + w_2 x_2 + \ldots + w_d x_d$。假设你关心的是第 $i$ 个特征（比如"是否在十字路口"），你把第 $i$ 维的权重临时换成一个**乐观的高值** $\bar{w}^i$，其他维度不动，算出来的就是 stopping value。

**停下来的额外好处 = $(\bar{w}^i - w_i) \times x_i(s)$**

- 特征 $x_i(s)$ 高 → bonus 大 → 想停
- 特征 $x_i(s)$ 低 → bonus ≈ 0 → 不停，继续拿 reward

就这么简单。技能一边拿真 reward，一边被引导到"特征高且值钱"的地方停。

---

## 为什么这比别的方法好

| 方法 | 技能追逐什么 | 停在哪 | planner 用它 |
|---|---|---|---|
| **Shortest-path** | -1 每步 | bottleneck | 几乎不选，因为穿罚区 |
| **Eigenoptions** | Laplacian 内在奖励 | 任意 | 散得到处都是，信息量低 |
| **Option-critic** | 真 reward | 隐式学 | on-policy 只在走过的 state 好，planner 从随机 state 开始就废 |
| **Reward-respecting** | 真 reward | 特征高 + 值高 | 频繁被 max 选中，planning 快 |

核心 insight：**技能的"有用性"由 planner 会不会选它定义**，不是由它自己能拿多少 reward 定义。

---

## STOMP 是什么

四个步骤的流水线：

1. **Subtask**：定义一个子任务（用 GVF 语言，指定 cumulant 和 stopping value）
2. **Option**：用 actor-critic off-policy 学出这个子任务的最优 policy + stopping rule
3. **Model**：学这个 option 的 transition model 和 reward model
4. **Planning**：拿 model 做近似 value iteration，改进 main task 的 value function

Sutton 说长期来看这四步应该**同时在线跑**，互相 feed back——planning 发现哪个 option 有用，反过来影响下一步该发现什么新 option。

---

## 一个旋钮：bonus weight

$\bar{w}^i$ 控制"多想去碰那个特征"：

- **太大**（比如 100）→ 退化成 shortest-path，只要快到 bottleneck，不管 reward
- **适中**（比如 1）→ 平衡 reward 和 feature attainment，planning 最快
- **太小**（比如 0.1）→ 接近 main task 本身，学不出新东西，planning 后劲不足

论文实验里 $\bar{w}^i = 1$ 效果最好。

---

## 实验结论

**Two-room**：reward-respecting option 让 planning 比 actions-only 快约 3 倍；shortest-path option 反而比 actions-only 还慢。

**Four-room（stochastic）**：四种方法排名——reward-respecting > shortest-path > eigenoptions > option-critic。actions-only 慢起步但最后追上来（可能因为 stochastic 下的 maximization bias）。

---

## 为什么 Option-critic 表现差

Option-critic 看起来也是 reward-respecting（直接优化 main task reward），但它是 **on-policy** 的——只在行为 policy 走过的 state 上学好。Planner 从随机 state 开始 update，碰到很多 option-critic 没见过的 state，那里的 option policy 接近随机，model 也差，planning 被拖累。

**off-policy 学习是 planning 能用的前提**。这是 Sutton 刻意的设计选择。

---

## 对你的直觉

如果你在想 LLM agent / world model 方向：

- **"thought step" 当 option**：定义子任务"思考直到答案的某个 coverage feature 高"，cumulant 用 task reward，stopping bonus 在 coverage 高时给
- **off-policy 是关键**：on-policy 学出来的 skill 在没走过的 state 上是 garbage
- **stopping 是一等公民**：什么时候停 thinking 应该有 explicit bonus signal，不能纯靠 gradient 学
- **expectation model + linear value** 是 planning 的甜区，nonlinear 需要新理论
- **STOMP 是自相似的**：每一层都是 GVF，可以递归 build hierarchy

核心 take-away：**技能的价值不是"它能拿多少 reward"，而是"planner 会不会选它"**。这改变了 option discovery 的目标函数。

---

# Reward-Respecting Subtasks for Model-Based RL — Sutton et al. 深度讲解

你好 Karpathy！这篇 paper 是 Sutton 在 Alberta Plan 路线下的一个核心 piece，本质上是把 "options framework" 重新激活，并给 option discovery 一个新的约束方向：**让 subtask 尊重 original reward，同时通过 stopping bonus 引导 option 在特定 feature 高的 state 停止**。下面我从 intuition、公式变量、STOMP 架构、实验数据、与你熟悉的 system 的联想几个层面铺开。

参考链接：
- paper PDF (arXiv 版本): https://arxiv.org/abs/2302.03018
- Sutton "The Alberta Plan for AI": https://arxiv.org/abs/2208.11173
- Sutton, Precup, Singh 1999 options 原始 paper: https://www.sciencedirect.com/science/article/pii/S0004370299000521
- Machado et al. 2023 successor representation / eigenoptions: https://www.jmlr.org/papers/volume24/21-0399/21-0399.pdf
- Bacon, Harb, Precup 2017 option-critic: https://arxiv.org/abs/1609.05140
- Wan et al. 2019 planning with expectation models: https://arxiv.org/abs/1810.07953
- Modayil, White, Sutton 2014 Horde / GVFs: https://journals.sagepub.com/doi/10.1177/1059712314534789

---

## 1. 这篇 paper 想解决的根本问题

Model-based RL 要 tractable，model 必须 **abstract in state and time**。state abstraction 由 deep learning 解决了，但 **temporal abstraction** 一直没有被大规模用起来，原因之一是 **option space 太大**，而以前的 option discovery 方法（bottleneck shortest-path、eigenoptions、option-critic、auxiliary pixel maximization）定义 subtask 时 **完全不考虑 original reward**，导致学出来的 option 在 planning 时根本不会被 max 选中——planner 顶多会忽略它，最坏还会被 misleading。

Sutton 的核心 move 是：**子任务本身的 cumulant 直接用原始 reward $R_t$**，然后 stopping value 不是 $0$，而是用 "optimistic bonus weight $\bar{w}^i$" 替换 value function 里第 i 个 feature 的权重，得到一个 stopping bonus $(\bar{w}^i - w_i) x_i(s)$。这样 option 一边走一边拿真 reward，到了 feature $x_i$ 高、且 estimated value 也高的 state 才停下来领 bonus。

这背后的 intuition 非常 Sutton：**option discovery 不应该是 unsupervised 的 graph problem，而应该被 "这个 option 将来会被 planner 怎么用" 反向约束**。Planner 在 (18)/(19) 式里取 max，所以 option 要有用，它的 backed-up value $r(s,o) + v(\text{next})$ 就要足够大；reward-respecting 直接对齐这个目标。

---

## 2. STOMP progression：一个统一架构

paper 反复强调 **STOMP** = **Sub**task → **O**ption → **M**odel → **P**lanning，是发展 temporally-abstract cognitive structure 的标准 pipeline：

| Stage | 输入 | 输出 | Sutton 的工具 |
|---|---|---|---|
| Subtask | feature index $i$, bonus weight $\bar{w}^i$ | 一个 GVF task $(c, z, \gamma)$ | GVF framework |
| Option | GVF task | policy $\pi^i$ + stopping $\beta^i$ | actor-critic off-policy |
| Model | option + data | $\hat{r}(\cdot, o), \hat{\mathbf{n}}(\cdot, o)$ | TD(λ) on expectation model |
| Planning | option models | improved $\hat{v}(\cdot, \mathbf{w})$ | approximate value iteration |

paper 里这四步是顺序展示的，但 Sutton 在 conclusion 里明确说 long-term vision 是 **这四步同时在线运行**，互相 feed back（这是他所谓的 "Oak architecture / FCSTOMP"）。这一点和你一直关心的 "system that learns and plans in a single ongoing stream of experience" 完全对齐。

---

## 3. 核心公式逐个拆解

### 3.1 General Value Function (GVF)

$$v_{\pi,\beta}^{c,z}(s) \doteq \mathbb{E}_{\pi,\beta}\left[\sum_{j=1}^{K} \gamma^{j-1} c(S_j) + \gamma^{K-1} z(S_K) \,\middle|\, S_0 = s\right] \tag{2}$$

变量说明：
- $s$：起始 state
- $\pi$：option 的 intra-option policy
- $\beta$：stopping function $\beta: \mathcal{S} \to [0,1]$，每步以 $\beta(S_t)$ 的概率停止
- $c$：cumulant function $c: \mathcal{S} \to \mathbb{R}$，替代 reward，可以是任何要被累加的信号
- $z$：stopping value function $z: \mathcal{S} \to \mathbb{R}$，停止时额外加上的 bonus
- $K$：stopping time，由 $\beta$ 决定的随机停止步数（$K \le L$，$L$ 是 episode 长度）
- $\gamma$：discount factor
- 上标 $c, z$：标明这个 value function 是由哪两个函数定义的（fixed → 定义 subtask）
- 下标 $\pi, \beta$：标明这是哪个 option 的 value（varied → 是 candidate solution）

关键 insight：**superscript 固定 = subtask 定义，subscript 变动 = 解**。整个 option discovery 的不同方法（eigenoptions、shortest-path、reward-respecting、option-critic）只是在这个 GVF 框架下选不同的 $c$ 和 $z$ 而已。

| 方法 | $c(S_t)$ | $z(s)$ |
|---|---|---|
| Main task | $R_t$ | $0$, $\beta=0$ |
| Shortest-path to bottleneck | $-1$ | $0$ at bottleneck, $-\infty$ elsewhere |
| Eigenoptions | $\mathbf{e}_i^\top (x(S_t) - x(S_{t-1}))$ | $0$ |
| **Reward-respecting feature attainment** | **$R_t$** | **$w^\top x(s) - w_i x_i(s) + \bar{w}^i x_i(s)$** |
| Option-critic | $R_t$ | (隐式, 通过 termination gradient) |

这张表是 paper 的灵魂。

### 3.2 线性 value approximation

$$\hat{v}(x(s), w) \doteq w^\top x(s) = \sum_{i=1}^{d} w_i x_i(s) \tag{3}$$

- $x: \mathcal{S} \to \mathbb{R}^d$ 是 feature function（paper 里实验用 one-hot，但 framework 允许任意 feature，包括神经网络 hidden units）
- $w \in \mathbb{R}^d$ 是 weight vector
- $w_i$、$x_i(s)$ 是第 i 维

为什么坚持线性？因为 paper 在 Section 4 末尾说：**线性 value + expectation model 在 planning 时没有额外 approximation error**（Wan et al. 2019 证明）。这是 Sutton 一个非常 deliberate 的选择——他宁可扩展 feature space，也不轻易跳到非线性，因为非线性下 expectation model 不再和 distribution model 等价，planning 会失真。

### 3.3 Reward-respecting subtask 的 stopping value

$$z^i(s) \doteq w^\top x(s) - w_i x_i(s) + \bar{w}^i x_i(s) \tag{4}$$

- $i$：feature 的 index（也是 subtask 的编号）
- $w$：当前 main task 的 weight vector
- $w_i$：第 i 个 feature 当前的 weight（**会随 learning/planning 变化**）
- $\bar{w}^i$：bonus weight，第 i 个 feature 的 **乐观上界估计**（paper 里设为 1）
- $x_i(s)$：state s 的第 i 个 feature 值

直觉：$z^i(s)$ 就是 "把 main-task value function 里第 i 个 feature 的权重临时换成 $\bar{w}^i$" 得到的 value。所以：

- 在 feature $x_i(s)$ 高的 state，bonus $(\bar{w}^i - w_i) x_i(s)$ 大 → option 倾向停在这
- 在 feature $x_i(s)$ 低的 state，bonus 接近 0 → option 不会为了 bonus 而停，会继续走拿 reward

这个 design 的精妙之处在于 **stopping value 和 main-task value 共享同一个 $w$**，区别只在第 i 维。这就避免了 "subtask 完全独立" 导致学出来的 option 和 main task 无关（这是 eigenoptions 的毛病）。同时 bonus weight 是 **optimistic** 的，所以 subtask 总是 "想去看一下" feature i 高的 state 长什么样，这与 "intrinsically motivated exploration" 的 optimistic initialization 思想一致。

### 3.4 General TD error $\delta$

$$\delta(c, z, v, v', \beta) \doteq c + \beta z + \gamma(1-\beta) v' - v \tag{5}$$

- $c$：当前 cumulant
- $z$：当前 state 的 stopping value
- $v$：当前 state 的 estimated subtask value
- $v'$：下一个 state 的 estimated subtask value
- $\beta$：当前 stopping probability

注意这是 **single step** 的 TD error，里面 **$c + \beta z + \gamma(1-\beta) v'$** 是一个混合 target：以 $\beta$ 概率 stop 取 $z$，以 $1-\beta$ 概率 continue 取 $\gamma v'$，加上 cumulant $c$。这统一了 on-policy TD、Q-learning、option-model learning 等所有 update。

### 3.5 UWT (UpdateWeights&Traces)

```
Procedure UWT(w, e, ∇, αδ, ρ, γλ(1-β)):
    e ← ρ(e + ∇)
    w ← w + αδ e
    e ← γλ(1-β) e
```

- $w$：被更新的 weight vector（input/output）
- $e$：eligibility trace（input/output）
- $\nabla$：通常是 $\nabla_w \hat{v}$，linear case 下就是 $x_t$
- $\alpha$：step size
- $\delta$：上面定义的 general TD error
- $\rho$：importance sampling ratio $\pi(A_t|S_t)/\mu(A_t|S_t)$，on-policy 时为 1
- $\lambda$：bootstrapping 参数
- $\beta$：当前 stopping probability（注意 trace decay 用 $\gamma\lambda(1-\beta)$，意思是 stop 时 trace 清零）

Sutton 用 UWT 这个抽象是为了说明：**option learning、policy learning、model learning 在算法层面是同一个 update，只是 target 不同**。这和他 Horde architecture（数千个 GVF 并行学习）的思想一脉相承。

### 3.6 Option 的 stopping function

$$\beta^i(s) = \begin{cases} 1 & \text{if } z^i(s) \geq w^{i\top} x(s) \\ 0 & \text{otherwise} \end{cases} \tag{9}$$

- $z^i(s)$：stopping value
- $w^{i\top} x(s)$：当前 subtask 的 estimated value（注意这是 subtask 自己的 weight $w^i$，不是 main task 的 $w$）

直觉：**当 stopping 比继续更好就 stop**。这是一个 greedy 的 stopping rule，paper 用 deterministic 版本，理论上也可以 soft。这和 option-critic 的 termination gradient 是同一个 idea 的不同实现，但 reward-respecting 多了一个明确的 bonus signal，所以 stopping 会指向 "高 feature 且高 value" 的 state，而不是任意 state。

### 3.7 Option model (expectation model)

Reward part:
$$r(s, o) \doteq \mathbb{E}_{\pi_o, \beta_o}\left[\sum_{t=1}^{K} \gamma^{t-1} R_t \,\middle|\, S_0 = s\right] \tag{12}$$

Transition part (expected discounted next-feature):
$$p(s'|s, o) \doteq \sum_{t=1}^{\infty} \gamma^t \Pr\{K=t, S_t = s' | S_0 = s\} \tag{13}$$

Approximate transition (expectation model):
$$\hat{\mathbf{n}}(x(s), o) \approx \sum_{s'} p(s'|s,o) x(s') = \mathbb{E}_o[\gamma^K x(S_{t+K}) | S_t = s] \tag{15}$$

- $\hat{\mathbf{n}}$：返回一个 **feature space 中的向量**，不是 distribution over states
- $\gamma^K$：discount 到 stopping 时刻
- $x(S_{t+K})$：stop 时的 feature vector

这个 expectation model 的好处是：**planning update 的计算量从 $|\mathcal{S}|$ 降到 $d$**（对 linear case 是 $d^2$），在大 state space 下至关重要。Wan et al. 2019 证明 linear value + expectation model 等价于 distribution model 在 planning 中的效果。

### 3.8 Approximate Value Iteration (AVI) — planning 的核心 update

$$w \gets w + \alpha \left[\max_{o \in \mathcal{O}(x)} \left[\hat{r}(x, o) + \hat{v}(\hat{\mathbf{n}}(x, o), w)\right] - \hat{v}(x, w)\right] \nabla_w \hat{v}(x, w) \tag{19}$$

- $w$：main task 的 value weight（被 planning 改进）
- $\mathcal{O}(x)$：在 state feature $x$ 下可用的 option 集合（包括 primitive actions + learned options）
- $\hat{r}(x, o)$：option $o$ 在 $x$ 下的 expected discounted reward
- $\hat{\mathbf{n}}(x, o)$：option $o$ 从 $x$ 出发 expected discounted next-feature
- $\hat{v}(\hat{\mathbf{n}}(x, o), w) = w^\top \hat{\mathbf{n}}(x, o)$：next state 的 estimated value
- $\nabla_w \hat{v}(x, w) = x$（linear case）

整个 update 是 **gradient descent toward backed-up value**。注意这里 planner 不接触环境，只接触 model。一次 look-ahead = 一次 $\max[\hat{r} + \hat{v}(\hat{\mathbf{n}})]$ 计算，是 Figure 1/5/8 的 x 轴单位。

---

## 4. 为什么 reward-respecting 比 baseline 好？— intuition 拆解

### 4.1 vs Shortest-path option（bottleneck）

在 two-room gridworld 里，shortest-path option 会直接穿过 $-1$ reward 区域到 hallway。这个 option **几乎不会出现在 optimal trajectory 里**（optimal trajectory 是绕开 $-1$ 区的），所以 planner 用它的 model 做更新时，max 几乎不会选它，相当于 dead weight。更糟的是 paper 里 Figure 1 显示 planning with shortest-path option **比 actions-only 还慢**——因为每步多算了一个 look-ahead，但这个 look-ahead 的结果几乎从不会被 max 选中。

Reward-respecting option 因为 cumulant 就是真 reward，会被引导去走 "绕开 $-1$ 区" 的路径，学出来的 option 正好是 optimal policy 的一段，planner max 会频繁选它，所以 update 信息量大。

### 4.2 vs Eigenoptions

Eigenoptions 用 graph Laplacian 的 eigenvector 当 intrinsic reward，**完全无视 task reward**。在 four-room 里 eigenoptions 会让 agent 探索整个空间，但探索到的 state 不一定是 high-value state。planner 用 eigenoption model 时，next-state distribution 散得到处都是，$\hat{v}(\hat{\mathbf{n}})$ 平均化，信息量低。

### 4.3 vs Option-critic

Option-critic 表面上是 reward-respecting 的（它直接优化 main task reward），但 paper 在 Appendix C 里解释了为什么它效果差：
- Option-critic 是 **on-policy** 的，只在行为 policy 访问到的 state 上学好 option
- planner 做 AVI 时是从 state set 里 **随机 sample state** 来 update 的，会碰到很多 option-critic 没见过的 state
- 在这些 state 上 option policy 接近随机，model 也差，planning 被 dirty model 拖累

这一点对你想 build agent 的 intuition 很重要：**off-policy 学习 option 是 planning 用得上的前提**。Sutton 的 STOMP 全栈都是 off-policy 的，这是刻意的。

### 4.4 Bonus weight $\bar{w}^i$ 的影响（Figure 4/5）

| $\bar{w}^h$ | 学到的 option 行为 | planning 速度 |
|---|---|---|
| 0.1 | 接近 main-task optimal policy（绕开 $-1$，直接奔 goal） | 初期快，长期慢（option 不够 "focused"） |
| 1 | 绕开 $-1$ 到 hallway，平衡 | 最快 |
| 10 | 部分走 shortest-path | 中等 |
| 100 | 完全 shortest-path | 慢（和 shortest-path 一样） |

直觉：**bonus weight 是 feature attainment 与 reward 之间的 trade-off 旋钮**。$\bar{w}^i \to \infty$ 时 reward 项被 bonus 完全淹没，subtask 退化成 shortest-path；$\bar{w}^i \to 0$ 时 subtask 退化成 main task（学不出新东西）。Sutton 推荐 $\bar{w}^i$ 取 $w_i$ 历史上的 high quantile，作为 "optimistic but not infinite" 的设置。

---

## 5. 实验数据精读

### 5.1 Two-room gridworld（Figure 1）

- $|\mathcal{S}| = 72$，one-hot feature，$d = 72$
- $\gamma = 0.99$，optimal path 17 步，$v_*(s_0) = 0.99^{17} \approx 0.843$
- 行为 policy：uniform random
- 学习 50,000 步 off-policy 学出 option + model
- Planning: AVI，随机 sample state，$\alpha = 1$
- 结果（到 $\hat{v}(s_0) \ge 0.8$ 所需 look-ahead）：
  - Reward-respecting option：~250 look-aheads
  - Actions only：~700
  - Shortest-path option：~1000（比 actions 还慢）

这个 4x speedup 不大，但 paper 的关键不是绝对数字，而是 **shortest-path 反而拖慢 planning** 这个反直觉现象，说明 option 的 "planning utility" 不是 "shorter is better"。

### 5.2 Four-room stochastic gridworld（Figure 7/8）

- 4 个 hallway states H1-H4，4 个 reward-respecting subtasks
- 动作 stochastic：$2/3$ 按预期方向，$1/9$ 走其他三个方向之一
- 对比 4 种 option discovery：reward-respecting / shortest-path / eigenoptions / option-critic
- 结果（到 $\hat{v}(s_0)$ 接近 $v_*$ 的 look-aheads）：
  1. Reward-respecting 最快
  2. Shortest-path 次之
  3. Eigenoptions 较慢
  4. Option-critic 最慢
  5. Actions-only 一开始慢，最后反超（但可能是 maximization bias）

注意 stochastic 环境下 model learning 永远不会 RMSE = 0（Figure A.1），但 planning 仍然有效——这是 expectation model 的 robustness 体现。

### 5.3 Model learning 曲线（Figure 3 / A.1）

- Transition model RMSE 下降比 reward model 慢
- 但 planning 在 ~20,000 步就达到近 optimal（Figure 3 inset），说明 **不需要完美 model，good enough model 就够 planning 用了**——这和 Dyna 的观察一致

---

## 6. 与你关心方向的联想

### 6.1 与 Dyna / MuZero / Dreamer 的关系

Sutton 1990 年的 Dyna 是 model-based RL 的鼻祖，这篇 paper 是 Dyna 的 **temporally-abstract 升级版**：Dyna 用 primitive action model，这里用 option model。MuZero 学的 latent dynamics model 本质上也是 expectation model（在 latent space 里预测 next latent），planning 用 MCTS；Dreamer 用 latent imagination + actor-critic。STOMP 给的是一个 **更结构化的中间层**：每个 option 是一个 "named skill"，planner 可以直接选 "用哪个 skill" 而不是每步都展开。

如果你把 LLM 的 "chain of thought" 看作一种 temporal abstraction——每个 "thought step" 是一个 option，那么 reward-respecting subtask 给了一个原则：**thought 的边界应该落在 "value 高 + 某个 feature 高" 的中间状态**，而不是固定长度。这和 "let the model decide when to stop thinking" 的 idea 是一致的。

### 6.2 与 Horde / General Value Functions

Sutton 2011 的 Horde 提出 agent 是一堆 GVF 的集合，每个 GVF 是一个 "question about the world"。这篇 paper 把 Horde 的 idea 用在 **option discovery** 上：subtask 本身就是 GVF，option 是 GVF 的解，model 是 GVF 的 transition。整个 agent 是 self-similar 的——每一层都是 GVF。

这对你想 build "system that knows what it knows" 的方向很有启发：GVF 是 first-class object，可以 plan、可以 learn、可以 be discovered，递归下去就是 hierarchical RL 的统一框架。

### 6.3 与 "The Bitter Lesson"

Sutton 2019 的 Bitter Lesson 说：scale + search + model-based 击败 hand-crafted feature。这篇 paper 看似反方向（强调 feature attainment 和人工选 subtask），实际上它在为 "scale" 铺路——STOMP 的 long-term vision 是 **feature construction 也被 planning utility 反向驱动**（FCSTOMP / Oak architecture）。即：feature 不再是 hand-crafted，而是 "什么 feature 对 planning 有用就学什么 feature"。这是一个 search over abstractions，和 AlphaZero 的 policy/value network 一样是 learned component，只是 search space 是 abstraction 而不是 move。

### 6.4 与 continuation learning / continual learning

paper 在 conclusion 提到 "big world hypothesis"（Javed & Sutton in prep）：world 远大于 agent，任何 model 都必然 incomplete。Reward-respecting subtask 的 stopping bonus 本质上是 **"承认不知道远方，但保持乐观去看"** 的 exploration 策略，这在 non-stationary / continual learning 下是必需的。Option 不再是 "solve and forget"，而是 "maintain as the world drifts"。

### 6.5 一个潜在的 hallucination 联想

如果把 reward-respecting subtask 的 stopping value $z^i(s)$ 写成：

$$z^i(s) = \hat{v}(x(s), w) + (\bar{w}^i - w_i) x_i(s)$$

这就是 **"main task value + 一项 intrinsic bonus"**。这个形式和 curiosity-driven exploration (ICM, RND)、count-based exploration、甚至 LLM 的 RLHF reward shaping $r_{\text{total}} = r_{\text{task}} + \beta \cdot r_{\text{intrinsic}}$ 是同构的。差别是 reward-respecting 把 intrinsic bonus 放在 **stopping time** 而不是每步——这避免了 intrinsic reward 把 policy 带偏，是一个更克制的设计。

这个 idea 我觉得可以直接搬到 LLM agent 的 tool-use 上：定义一个 subtask "调用 search tool 直到 retrieval feature 高"，bonus weight 设为 "retrieval 偶尔很有用时的高 value"。这样学出来的 "search option" 不会无脑 search（像 pixel maximization 那样），也不会只在 search 总是好的 state 用，而是 **在 search 偶尔有大用、平时没用** 的 state 才用——这正是 tool-use 的真实分布。

---

## 7. paper 的局限与 open problems

1. **Subtask 选择仍然是 hand-crafted**：feature index $i$ 和 bonus weight $\bar{w}^i$ 是人选的。Sutton 在 future work 里说要让 planning utility 反向驱动 feature 选择（FCSTOMP）。
2. **线性 value approximation**：实验全是 tabular / one-hot。nonlinear neural net 下 expectation model 的 planning correctness 没保证（Wan et al. 2019 的定理不成立）。
3. **Eligibility trace 在实验里 $\lambda = 0$**：framework 支持 $\lambda > 0$ 但没实验。
4. **没有 high-dim perceptual state**：pixel input 没试。
5. **没有证明**：reward-respecting subtask 的 option 在 planning 中一定有用，只有 empirical 证据。
6. **Bonus weight 的 schedule** 没系统研究：什么时候 update $\bar{w}^i$？paper 里是固定 1，但理论上应该是 $w_i$ 的 running max 或 quantile。
7. **Search control**：Appendix C 提到 AVI 从哪些 state sample 是 open problem，这其实是 Dyna 一直没解决的问题。
8. **Option 数量**：实验只 4 个 option，scale 到 1000+ 个 GVF 的 Horde 规模下 planning 的 $\max$ 操作代价会怎样？

---

## 8. 一句话总结

这篇 paper 的贡献是给 option discovery 一个 **"planning-aware" 的约束**：subtask 的 cumulant 用原 reward，stopping value 用 "optimistic bonus weight 替换第 i 维 feature 权重" 的 main-task value。这个约束既让 option discovery space 大幅缩小（部分解决 "option 太多" 问题），又让学出来的 option **自带 "会被 planner max 选中" 的 bias**。整套 STOMP pipeline 用 UWT 这个统一 update 实现完全 off-policy 学习，和 Sutton 长期主张的 "all learning is off-policy, all learning is GVF" 完全对齐。

对你 build agent 的直觉，我觉得最有用的几个 take-away：
- **option 的价值由 "planner 会不会选它" 定义**，不由 "它自己能达到多少 reward" 定义
- **stopping 是 first-class decision**，应该有 explicit bonus signal，不能让 option-critic 那样纯靠 termination gradient 学
- **off-policy 是 planning-friendly 的前提**：on-policy 学出来的 option 在 unvisited state 上是 garbage
- **expectation model + linear value 是 planning 的 sweet spot**，nonlinear 需要 new theory
- **STOMP 是 self-similar pipeline**，每一层都是 GVF，可以递归 build hierarchy

如果你想把这套搬到 LLM agent / world model 方向，最直接的实验是：把 "thought step" 作为 option，定义 subtask "generate thoughts until some feature of the partial answer is high (e.g. answer-coverage feature)，cumulant 用 task reward"。用 reward-respecting stopping bonus 学出来的 "stop thinking" policy 应该比固定 token budget 或 simple heuristic 强。

希望对你 build intuition 有用！如果想深入某一个公式或某一段实验数据再展开讲，尽管问。
