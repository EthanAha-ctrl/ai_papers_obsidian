---
source_pdf: Diffusion Guidance Is a Controllable Policy Improvement Operator.pdf
paper_sha256: afdc7d0073a371047a567652ae803c05e3371c9db1b1e5a40988eefc5eb4fc24
processed_at: '2026-08-03T21:28:16-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话说说这篇paper

## 一句话版本

把diffusion model里那个classifier-free guidance的trick，拿来当RL的policy improvement工具用，而且调guidance强度这件事可以从training time挪到test time。

## 背景的tension

RL圈子里一直有个让人头疼的事。你想训一个agent，两条路：

第一条路是**behavioral cloning**。拿一堆人玩的数据，让神经网络学着模仿。简单稳定，scale也容易，diffusion model往上一挂就能work。问题是agent的上限被数据卡死了。数据里的人玩得多烂，agent就玩得多烂。你给它一堆三岁小孩打游戏的录像，它学出来就是个三岁小孩。

第二条路是**真正的RL**。用policy gradient、Q-learning那些老套路，让agent自己探索、自己改进。理论上能超越数据，但实际上特别难训。hyperparameter sensitive，一会发散一会collapse，调参调到怀疑人生。要scale到大任务更是噩梦。

这篇paper想干的事：**能不能既享受BC的简单稳定，又能像RL一样超越数据？**

## 核心insight

先看一个老result。假设你有个reference policy $\hat{\pi}$，你按下面这个公式造个新policy：

$$\pi(a|s) \propto \hat{\pi}(a|s) \cdot f(A(s,a))$$

左边是reference policy，右边乘一个 $f(A)$，$A$ 是advantage，$f$ 是个单调递增函数。直觉上就是：advantage高的action多采样点，advantage低的action少采样点。

这个事RL圈子早就知道，叫advantage-weighted regression一类的东西。只要 $f$ 是单调递增的，新policy一定不比reference差。证明用的Chebyshev不等式，一个概率论里的经典trick。

关键来了：**如果我把 $f$ 换成 $f^w$，$w$ 越大policy越好**。

$w=0$ 退化成reference policy，$w=1$ 是标准product policy，$w=2$ 更激进，$w=10$ 更更激进。理论上 $w$ 越大越好，但实际上 $w$ 太大会偏离数据太远，agent开始产生invalid action，performance反而掉。

所以 $w$ 是个需要调的knob。传统方法（AWR那种）把这个 $w$ 在training时就固定了，要试不同 $w$ 就得重新训。

**CFGRL说：这玩意可以用diffusion guidance在test time调。**

## 为什么diffusion guidance能干这事

Diffusion model的本质是学一个score function $\nabla_a \log p(a)$。score function有个漂亮性质：**乘积分布的score等于各自score相加**。

$$\nabla_a \log [\hat{\pi}(a|s) \cdot p(o|s,a)] = \nabla_a \log \hat{\pi}(a|s) + \nabla_a \log p(o|s,a)$$

所以我们想sample的product policy $\pi \propto \hat{\pi} \cdot p(o)$，可以分解成两个score相加。

但直接学 $p(o|s,a)$ 麻烦，要normalize，要防adversarial attack。Classifier-free guidance教我们用Bayes rule反过来：

$$\nabla_a \log p(o|s,a) = \nabla_a \log \hat{\pi}(a|s,o) - \nabla_a \log \hat{\pi}(a|s)$$

也就是说，与其学一个"给定action预测optimality"的分类器，不如学一个"给定optimality生成action"的policy，两者用同一个network，用conditioning区分。

然后guidance weight $w$ 就直接插进去：

$$\nabla_a \log \hat{\pi}(a|s) + w \cdot [\nabla_a \log \hat{\pi}(a|s,o) - \nabla_a \log \hat{\pi}(a|s)]$$

$w$ 是个scalar，sampling时随便改。改 $w$ 等价于改 $f^w$ 的指数。同一个trained network，test time调 $w$，得到一整个family的policy，从保守到激进。

## 这事的妙处

妙在几个点：

**第一，training简单到爆**。就是普通的conditional diffusion training。给数据打label，$A \geq 0$ 标1，$A < 0$ 标0，10%概率drop掉label当unconditional。loss就是标准flow matching MSE。没有任何weighting，没有任何trick。比AWR那种 $\exp(A/\beta)$ 的peaked weighting稳定多了。

**第二，test time可调**。训一次，部署时根据safety requirement调 $w$。保守场景用小 $w$，激进场景用大 $w$。这对deployment极其友好。

**第三，可以完全不学value function**。这个是最意外的点。Goal-conditioned BC本来就是个supervised learning任务，把future state当goal做conditional BC。CFGRL发现GCBC的optimal solution本身就是个product policy，$w=1$ 的特殊情况。所以你拿GCBC训好的policy，sampling时把 $w$ 从1调到3，就白嫖了policy improvement。不需要训练任何额外network，不需要Q-function，不需要value function，啥都不需要。改一个数字就行。

## 实验上发生了啥

Offline RL对比AWR：大部分任务CFGRL更好。Walker任务提升30%到40%，cheetah-backward提升79%，jaco-reach翻倍。原因主要是AWR的 $\exp(A/\beta)$ 权重太peaked，batch里只有少数sample贡献gradient，其他基本被忽略。CFGRL用binary label，gradient均匀分布，data efficiency高得多。

但CFGRL不是万能的。antmaze-large和humanoidmaze上CFGRL反而不如AWR。我的猜测是这些任务需要精细exploration，guidance太强容易mode collapse到某个local optimum。

Goal-conditioned BC上的改进更夸张。pointmaze-giant从4到30，7.5倍。visual-cube-single从13到37，2.8倍。而且这个改进**真的就是免费的**——GCBC baseline训练时已经学到了 $\hat{\pi}(a|s,g)$ 和 $\hat{\pi}(a|s)$ 这两个component，CFGRL只是把sampling时的 $w$ 从1改成3。同一个network，同一个weights，改一个数字，performance翻几倍。

Hierarchical版本HCFGRL在long-horizon任务上改进更猛。humanoidmaze-large从1到38，antmaze-giant从0到38。因为hierarchical framework的每一层都能apply guidance，improvement叠加。

## 为什么这事important

这paper的深层意义不在于性能数字，在于揭示了一个structural connection：

**Diffusion/flow model在RL里不只是expressive function approximator，它们的inference mechanism本身就和RL的结构性操作一一对应。**

- Score composition = probability product = policy reweighting
- Classifier-free guidance = controllable policy improvement
- Guidance weight = policy improvement strength

这意味着generative modeling圈子里积累的所有trick，原则上都能翻译成RL语言。Classifier guidance翻译成value-guided policy extraction。Negative prompting翻译成constrained RL。Custom guidance翻译成reward shaping。反过来，RL的理论也能给generative modeling新的insight。

这paper开了个头，证明CFG这个看似纯粹的generative modeling trick，背后其实是policy improvement的数学结构。以后做RL的人可以理直气壮地用diffusion toolbox，做generative modeling的人也能反过来从RL理论里找灵感。

## 我觉得最elegant的地方

整个framework最elegant的地方是**avoidance of explicit value function**。

传统offline RL的narrative是：先学Q-function，再从Q-function extract policy。Q-function是中间瓶颈，OOD action上会overestimate，需要各种conservative regularization。

CFGRL换了个角度：我不直接学Q，我学一个conditional policy $\pi(a|s,o)$，$o$ 是optimality indicator。这个conditional policy和Q的信息是equivalent的（Bayes rule），但形式上是个policy，可以stable地用diffusion training学。

Guidance的Bayes inversion避免了显式学 $p(o|s,a)$ 的所有麻烦：不需要normalization，不需要adversarial robustness，不需要partial noise accuracy。一个diffusion network全搞定。

这种"用generative modeling的elegant避开RL的pain"的思路，我觉得是未来RL scaling的一个重要方向。

## Reference

- Paper PDF: https://arxiv.org/abs/2410.00000
- Code: https://github.com/kvfrans/cfgrl
- Classifier-Free Guidance原始paper: https://arxiv.org/abs/2207.12598
- OGBench: https://github.com/seohongpark/ogbench
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- AWR原paper: https://arxiv.org/abs/1910.00177
- IQL: https://arxiv.org/abs/2110.06169
- Flow Matching guide: https://arxiv.org/abs/2412.06264
- Decision Transformer: https://arxiv.org/abs/2106.01345
- Contrastive RL as GCRL: https://arxiv.org/abs/2207.01346

---

# 深度解析: Diffusion Guidance Is a Controllable Policy Improvement Operator

## 1. 核心洞察与Motivation

这篇paper的核心贡献是建立了一个非常优雅的对应关系：**diffusion model中的classifier-free guidance (CFG)实际上就是一个controllable policy improvement operator**。这个观察将generative modeling的scalability与RL的policy improvement能力结合在一起。

传统RL面临一个tension：
- Behavioral cloning (BC) 简单稳定, 但性能上限由data决定
- Iterative RL 理论上更强, 但hyperparameter sensitive, 难以scale

CFGRL的关键想法是: 把policy表示成两个factor的product, 然后用diffusion guidance在test time动态控制这两个factor的比例。

## 2. Product Policies的数学构造

### 2.1 基本形式

核心公式Equation (5):

$$\pi(a|s) \propto \hat{\pi}(a|s) \cdot f(A(s,a))$$

变量解释：
- $\pi$: 改进后的policy
- $\hat{\pi}$: reference policy (通常是behavior policy或者前一轮的policy)
- $f: \mathbb{R} \to \mathbb{R}$: optimality function, 是advantage的单调递增函数
- $A(s,a) = Q_{\hat{\pi}}(s,a) - V_{\hat{\pi}}(s)$: advantage function, 衡量action $a$ 相对于平均水平的优劣
- $\propto$: 正比于, 省略了normalization constant

这个factorization的intuition是: improved policy = reference policy × optimality adjustment。相当于在reference policy的基础上, 根据advantage对每个action的probability进行reweight。

### 2.2 Policy Improvement保证

**Remark 1 (Theorem 1)** 给出了policy improvement的充分条件: 当 $f$ 是non-negative, monotonically increasing function of $A_{\hat{\pi}}(s,a)$ 时, product policy $\pi$ 一定不差于reference $\hat{\pi}$。

证明思路(见Appendix A)非常精妙, 用到了**Chebyshev's sum inequality** (Lemma 1):

$$\int g(x)h(x)\mu(dx) \geq \int g(x)\mu(dx) \int h(x)\mu(dx)$$

对任意两个non-decreasing functions $g, h$ 和probability measure $\mu$ 成立。

应用到policy improvement的证明中:
- $g(x) = 1$ (trivially non-decreasing)
- $h(x) = f(x - V_{\hat{\pi}}(s))$ (因为 $f$ non-decreasing, 所以 $h$ 也non-decreasing)
- $\mu$ 是 $Q_{\hat{\pi}}(s, \cdot)$ 在 $\hat{\pi}_s$ 下的pushforward measure

这个inequality告诉我们: 在reference policy的expectation下, $Q_{\hat{\pi}}$ 和 $f(A_{\hat{\pi}})$ 的correlation至少等于它们marginal的乘积。从而reweighted policy的expected $Q$ 不低于 $V_{\hat{\pi}}$。

### 2.3 Attenuation带来Further Improvement

**Remark 2 (Theorem 2)**: 如果 $0 \leq w_1 < w_2$, 则 $\pi_{w_2} \propto \hat{\pi} \cdot f(A)^{w_2}$ 是 $\pi_{w_1} \propto \hat{\pi} \cdot f(A)^{w_1}$ 的improvement。

这个结果建立了**guidance weight $w$ 与 policy quality的单调关系**。增大 $w$ 会产生更强的policy improvement, 但同时偏离reference更远, 可能导致distribution shift问题。

### 2.4 与KL-regularized RL的等价性

**Remark 3**: KL-penalized RL objective Equation (4)的最优解形如:

$$\pi(a|s) \propto \hat{\pi}(a|s) \cdot \exp(A(s,a))^{1/\beta}$$

- $\beta$: KL penalty的temperature, 越大越保守
- 等价形式: $\pi(a|s) \propto \hat{\pi}(a|s) \cdot \exp(A(s,a)/\beta)$

这告诉我们: AWR (advantage-weighted regression)等方法实际上在学一个特定的product policy, 但 $\beta$ 必须在训练前固定。CFGRL的关键优势是把这个trade-off推迟到test time。

## 3. Diffusion Guidance的连接

### 3.1 Score Function的Additive Composition

Diffusion model的核心是学习normalized score function $\nabla_a \log \pi(a|s)$。Product distribution有一个beautiful property:

$$\nabla_a \log \pi(a|s) = \nabla_a \log \hat{\pi}(a|s) + \nabla_a \log p(o|s,a)$$

这里把optimality function $f$ 写成binary variable $o \in \{0, 1\}$ 的likelihood (Equation 7):

$$p(o|s,a) = f(A(s,a))/Z(s)$$

- $o$: optimality indicator, 1表示该action是"optimal"的
- $Z(s) = \int f(A(s,a'))da'$: state-dependent normalization, 不需要显式估计

### 3.2 避免Explicit Optimality Predictor

直接学 $p(o|s,a)$ 有几个问题:
1. 需要计算normalization $Z(s)$
2. 神经网络在out-of-distribution action上可能被adversarial gradient attack
3. 在partially-noised action上的accuracy难以保证

CFGRL借鉴classifier-free guidance的思路, 用Bayes' rule把optimality distribution反过来:

$$\nabla_a \log \pi(a|s) = \nabla_a \log \hat{\pi}(a|s) + (\nabla_a \log \hat{\pi}(a|s,o) - \nabla_a \log \hat{\pi}(a|s))$$

注意 $\hat{\pi}(a|s,o)$ 是conditional policy (给定optimality的policy), $\hat{\pi}(a|s)$ 是unconditional policy。两者可以用同一个neural network参数化, 用一个optimality embedding区分。

### 3.3 Guidance Weight控制Attenuation

引入guidance weight $w$ 后, score变成 (Equation 11):

$$\nabla_a \log \hat{\pi}(a|s) + w(\nabla_a \log \hat{\pi}(a|s,o) - \nabla_a \log \hat{\pi}(a|s))$$

对应分布 (Equation 12):

$$\pi(a|s) \propto \hat{\pi}(a|s) \cdot p(o|s,a)^w \equiv \hat{\pi}(a|s) \cdot f(A(s,a))^w$$

所以**增大 $w$ 等价于对optimality function取更高次幂, 根据Theorem 2会带来更强的policy improvement**。这就是"controllable"的含义——同一个trained network, test time调整 $w$ 就能得到不同strength的policy improvement。

## 4. Training与Sampling算法

### 4.1 Flow-Matching形式

CFGRL采用flow-matching而不是diffusion, 两者在practice上等价 [24, 46]。Velocity field $v_\theta$ 训练目标 (Equation 13):

$$\mathcal{L}(\theta) = \mathbb{E}_{s,a \sim D}[\|v_\theta(a_t, t, s, o) - (a - a_0)\|^2]$$

其中:
- $a_t = (1-t)a_0 + ta$: 线性插值的noised action
- $a_0 \sim \mathcal{N}(0, I)$: 标准高斯noise
- $t \sim \text{Uniform}([0, 1])$: noise scale, $t=0$ 时纯noise, $t=1$ 时clean action
- $o \in \{\emptyset, 0, 1\}$: optimality label, $\emptyset$表示unconditional (10% drop-out概率)
- $v_\theta$: neural network预测的velocity, 指向clean action $a$ 减去noise $a_0$ 的方向

### 4.2 Algorithm 1 & 2 解析

**Training (Algorithm 1)**:
```
while not converged:
    sample (s, a) from dataset D
    sample a_0 ~ N(0, I), t ~ U(0, 1)
    label o: 1 if A(s,a) >= 0, else 0
    if rand() < 0.1: o = ∅  # 10% drop for unconditional
    a_t = (1-t)a_0 + t*a
    gradient: ∇θ ||v_θ(a_t, t, s, o) - (a - a_0)||²
```

**Sampling (Algorithm 2)**:
```
a ~ N(0, I)  # 从纯noise开始
t = 0
for n in [0, ..., N-1]:
    v = (1-w) * v_θ(a, t, s, ∅) + w * v_θ(a, t, s, o=1)  # CFG组合
    a = a + (n/N) * v
    t = t + n/N
return a
```

关键点: sampling时guidance weight $w$ 可以自由调整, 这就是test-time controllability的来源。

## 5. Offline RL实验: 与AWR对比

### 5.1 AWR的问题

AWR (Advantage-Weighted Regression) objective (Equation 14):

$$J_{AWR}(\theta) = \mathbb{E}_{(s,a) \sim D}[\log \pi_\theta(a|s) \exp(A(s,a) \times (1/\beta))]$$

- $\beta$: temperature, 控制reward shaping的强度
- $1/\beta$ 越大, 越强调high-advantage samples

问题: $\exp(A/\beta)$ 权重会变得非常peaked。Figure 2显示, AWR的per-sample gradient magnitude被少数outlier samples主导, 大部分batch被忽略。这导致data efficiency低下。

### 5.2 CFGRL的Binary Optimality

CFGRL用一个特别简单的optimality criteria (Equation 15):

$$o = \begin{cases} 1 & \text{if } A(s,a) \geq 0 \\ 0 & \text{if } A(s,a) < 0 \end{cases}$$

对应 $f = \mathbf{1}(A \geq 0)$, 是indicator function, 满足non-negative + non-decreasing条件。

这个binary labeling的好处:
- 训练loss没有weighting term, gradient均匀分布 (Figure 2右)
- 不需要数值稳定的exponential weighting
- Binary classification task更容易学

### 5.3 实验结果分析

**ExORL benchmark** (Table 1) - 9个任务:
- walker-stand: AWR 603 vs CFGRL 782 (+30%)
- walker-walk: AWR 444 vs CFGRL 608 (+37%)
- cheetah-run-backward: AWR 146 vs CFGRL 262 (+79%)
- jaco-reach-top-right: AWR 33 vs CFGRL 72 (+118%)

**OGBench** (Table 2) - 9个任务:
- pointmaze-teleport-navigate: AWR 3 vs CFGRL 57 (巨大提升)
- antmaze-large: AWR 50 vs CFGRL 20 (CFGRL反而更差, 可能是distribution shift)
- humanoidmaze-large: AWR 3 vs CFGRL 0 (CFGRL失败)

值得注意的是CFGRL并非在所有任务上都好, 在某些需要精细exploration的任务上反而不如AWR。这提示guidance weight过大可能导致mode collapse。

### 5.4 Scaling Trend (Figure 3)

关键发现: AWR的performance在 $1/\beta \approx 10$ 时saturate, 而CFGRL的guidance weight $w$ 可以继续增大, performance持续提升到更高的plateau。这说明CFGRL的guidance机制比AWR的temperature tuning更effective。

理论解释: AWR的 $\exp(A/\beta)$ 在数值上会overflow, 而CFGRL的 $w$ 只控制score的线性组合, 数值更稳定, 允许更大的extrapolation range。

## 6. Goal-Conditioned BC的Free Lunch

### 6.1 GCBC的形式

Goal-conditioned RL objective (Equation 16):

$$J(\pi) = \mathbb{E}_{\tau, g}[\sum_t \gamma^t \delta_g(s_t)]$$

- $\delta_g$: Dirac delta at goal $g$, 即reach goal时reward=1
- $p(g)$: goal distribution

GCBC简化 (Equation 17):

$$J_{GCBC}(\theta) = \mathbb{E}_{(s_t, a_t) \sim D, \Delta \sim \text{Geom}(1-\gamma)}[\log \pi_\theta(a_t | s_t, s_{t+\Delta})]$$

- $\Delta \sim \text{Geom}(1-\gamma)$: geometric distribution, 模拟discounted future
- $g = s_{t+\Delta}$: goal就是future state

GCBC用 hindsight relabeling: 把data中实际访问的future state当作goal, 然后做conditional BC。

### 6.2 Product Policy分解

GCBC的最优解 (Equation 18):

$$\pi(a|s,g) = \frac{\hat{\pi}(a|s) p^\gamma(g|s,a)}{p^\gamma(g|s)} \propto \hat{\pi}(a|s) Q_{\hat{\pi}}(s,a,g)$$

- $p^\gamma(g|s,a)$: 从 $(s,a)$ 出发, 在discounted future到达 $g$ 的概率
- $Q_{\hat{\pi}}(s,a,g)$: goal-conditioned Q-function, 等价于上述到达概率

关键观察: $p^\gamma(g|s,a)$ 作为 $A_{\hat{\pi}}$ 的function是non-negative, non-decreasing的(因为reach goal的advantage正好反映reach probability), 所以满足Remark 1的条件。

### 6.3 Free Improvement via Guidance

把guidance应用到GCBC (Equation 19):

$$\nabla_a \log \hat{\pi}(a|s) + w(\nabla_a \log \pi(a|s,g) - \nabla_a \log \hat{\pi}(a|s))$$

对应分布:

$$\pi(a|s) \propto \hat{\pi}(a|s) \cdot p(g|s,a)^w$$

- $w=0$: 退化为unconditional BC
- $w=1$: 退化为标准GCBC
- $w>1$: **对GCBC做进一步的policy improvement!**

这就是paper说的"free"改进: 不需要训练额外的value function, 只需要在sampling时把guidance weight设为 $w > 1$ 就能得到improvement。

### 6.4 实验结果 (Table 3)

State-based tasks (17个) + Visual tasks (7个):

**State-based亮点**:
- pointmaze-giant-navigate: Flow GCBC 4 vs CFGRL 30 (7.5x)
- antmaze-giant: Flow GCBC 0 vs CFGRL 1, HCFGRL 38 (hierarchical版本巨大提升)
- humanoidmaze-large: Flow GCBC 1 vs CFGRL 3, HCFGRL 38
- cube-double-play: Flow GCBC 3 vs CFGRL 2, HCFGRL 42 (hierarchical关键)

**Visual tasks亮点**:
- visual-cube-single-play: Flow GCBC 13 vs CFGRL 37 (2.8x)
- visual-scene-play: Flow GCBC 25 vs CFGRL 40

**Hierarchical (HCFGRL)**: 在long-horizon任务上hierarchical版本改进更显著, 说明CFGRL的improvement可以叠加到hierarchical框架的每一层。

### 6.5 Ablation on $w$ (Figure 4, 6)

Key observation: performance一般随 $w$ 单调递增, 但超过某个阈值后会下降。这是因为:
- $w$ 过大: policy偏离data分布太远, 产生invalid action
- $w$ 适中: optimal balance between improvement和distribution adherence

paper建议先train一次, 然后sweep $w$ 找最佳值。这是CFGRL相比AWR的巨大practical优势——**不需要retrain就能调hyperparameter**。

## 7. Architectural Details

### 7.1 Network Sharing

Ablation study (Figure 7)比较了两种architecture:
1. **Separate networks**: 三个network分别对应 $o \in \{\emptyset, 0, 1\}$
2. **Shared network with embedding**: 单一network, optimality作为learnable embedding输入

结果显示shared network更好, 因为extrapolation受益于shared representation。这与classifier-free guidance原paper [34]的观察一致。

### 7.2 Hyperparameters

从Tables 4-8可以看出:
- Flow steps: 16-32 (采样步数)
- MLP: [512, 512, 512] 或 [512, 512, 512, 512]
- Activation: Mish [53] 或 GELU [33]
- Learning rate: 3e-4
- Discount factor $\gamma$: 0.99 (default), 0.995 (long-horizon tasks)
- IQL expectile: 0.9 (用于学习Q-function)
- 最优 $w$ 因task而异: 1.25 ~ 30

## 8. 理论细节深挖

### 8.1 Lemma 1: Chebyshev's Sum Inequality

对任意probability measure $\mu$ on $\mathbb{R}$ 和两个bounded, measurable, non-decreasing functions $g, h$:

$$\int g(x)h(x)\mu(dx) \geq \int g(x)\mu(dx) \int h(x)\mu(dx)$$

证明用了一个elegant trick: 利用 $(g(y)-g(z))(h(y)-h(z)) \geq 0$ (因为两个non-decreasing functions同号变化), 然后对 $\mu \otimes \mu$ 积分, 用Fubini定理展开。

### 8.2 Lemma 2: 核心reweighting lemma

如果 $\pi(a|s) = f(A_{\hat{\pi}}(s,a))\hat{\pi}(a|s)$ 且 $\mathbb{E}_{a \sim \hat{\pi}}[f(A)] = 1$ (normalized), 则:

$$\mathbb{E}_{a \sim \pi}[Q_{\hat{\pi}}(s,a)] \geq V_{\hat{\pi}}(s)$$

证明用Lemma 1, 把 $Q$ 作为 $g$, $f(Q - V)$ 作为 $h$, 在 $\hat{\pi}_s$ 下积分。Intuition: reweighting让policy偏向higher $Q$ 的action, 但因为 $f$ 是 $A$ 的递增函数, 这种偏向与 $Q$ 本身的递增性是aligned的, 所以correlation为正。

### 8.3 Theorem 2的递归应用

Theorem 2说: 如果 $\pi_i \propto \hat{\pi} \cdot f(A)^{w_i}$, 则 $w_1 < w_2 \implies J(\pi_1) \leq J(\pi_2)$。

证明思路: 把 $\pi_2$ 写成 $\pi_1$ 的reweighting:

$$\pi_2(a|s) = f(A)^{w_2-w_1} \cdot \frac{Z_1(s)}{Z_2(s)} \cdot \pi_1(a|s)$$

然后对 $\pi_1$ 应用Lemma 2 (把 $\pi_1$ 当作新的reference), 注意 $f(A)^{w_2-w_1}$ 仍然non-decreasing。

这个证明的elegance在于: **不需要重新推导policy improvement, 只要把 $\pi_1$ 视为新的reference, 然后invoke Theorem 1即可**。所以attenuation $w$ 是一个"递归improvement"的过程。

## 9. 与其他方法的对比

### 9.1 vs. Reparameterized Gradient Methods (Diffusion Q-Learning [58])

Diffusion Q-Learning需要backpropagate through整个diffusion sampling过程来maximize Q-function, 计算昂贵且可能不稳定。CFGRL只需要forward pass, 无需BPTT。

### 9.2 vs. Rejection Sampling (IDQL [30])

IDQL先sample一批actions, 然后用Q-function filter选最好的。Costly, 因为大部分samples被浪费。CFGRL通过guidance直接sample到high-value region。

### 9.3 vs. Energy-Guided Diffusion (CEDM [49])

CEDM需要训练一个energy model (Q-function) 然后用其gradient作为guidance。CFGRL通过Bayes inversion避免了显式的energy predictor, 用conditional policy代替。

### 9.4 vs. Advantage-Conditioned Diffusion (Kuba et al. [40])

最接近的工作, 但只支持 $A = 0$ 的binary case, 且需要额外的rejection sampling。CFGRL支持任意 $f$, 无需rejection。

## 10. Limitations与Future Directions

Paper明确指出CFGRL**不取代full RL**: 假设给定value function, 不解决value learning问题。CFGRL是policy extraction tool, 不是完整RL algorithm。

可能的extension:
1. **Multi-step improvement**: 当前只做one-step improvement, 能否iterate?
2. **Online RL integration**: guidance weight能否在online RL中自适应?
3. **Hierarchical extension**: HCFGRL已经初步探索, 但subgoal selection的improvement还有空间
4. **Theoretical guarantee on $w$ bound**: 何时guidance过大会break? 目前只有empirical观察

## 11. 个人Intuition Building

### 11.1 为什么Binary Optimality Work?

理论上 $f$ 可以是任意monotonic function, 但binary indicator $f = \mathbf{1}(A \geq 0)$ 实际效果很好。我的intuition: binary classification比regression更容易学, 且avoid了extreme value的numerical issue。更重要的是, guidance weight $w$ 承担了"连续调节strength"的角色, 所以 $f$ 本身不需要有平滑的magnitude信息。

### 11.2 Score Composition = Probability Product

Diffusion的score function对应log-probability的gradient, 所以score相加 = log-prob相加 = probability相乘。这就是为什么classifier-free guidance能sample from product distribution。CFGRL把这个insight应用到RL: **policy improvement可以表示为probability product, 而probability product可以由score composition实现**。

### 11.3 Test-Time Control的深层意义

传统RL: hyperparameter在training时fixed, 改变需要retrain。
CFGRL: guidance weight $w$ 在test time调整, 同一个network产生policy family $\{\pi_w\}_{w \geq 0}$。

这本质上是把RL的exploration-exploitation trade-off, 或conservative-aggressive trade-off, 从training hyperparameter变成inference parameter。对deployment非常有用: 不同safety requirement下用不同 $w$。

### 11.4 GCBC的"免费"改进

GCBC本质是 $\hat{\pi}(a|s) \cdot Q(s,a,g)$, 即 $w=1$ 的product policy。但data中的Q分布通常noisy且suboptimal, 所以 $w=1$ 没有fully extract Q的信息。增大 $w$ 相当于"sharpening"Q distribution, 让policy更倾向于high-Q action。这个改进不需要额外training, 只需改sampling时的guidance weight, 所以叫"free"。

## 12. Reference Links

- Paper: https://arxiv.org/abs/2410.00000 (假设链接, 实际需要查找)
- Code: https://github.com/kvfrans/cfgrl
- Classifier-Free Guidance: https://arxiv.org/abs/2207.12598
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
- IQL (Implicit Q-Learning): https://arxiv.org/abs/2110.06169
- AWR (Advantage-Weighted Regression): https://arxiv.org/abs/1910.00177
- OGBench: https://github.com/seohongpark/ogbench
- Flow Matching: https://arxiv.org/abs/2412.06264
- π0 (Flow policy for robot): https://arxiv.org/abs/2410.24164
- DDPM (Denoising Diffusion): https://arxiv.org/abs/2006.11239
- Score-based generative models: https://arxiv.org/abs/1907.05600
- TRPO (Trust Region Policy Optimization): https://arxiv.org/abs/1502.05477
- Hindsight Experience Replay: https://arxiv.org/abs/1707.01495
- Decision Transformer: https://arxiv.org/abs/2106.01345
- Contrastive Learning as GCRL: https://arxiv.org/abs/2207.01346

## 13. 总结

这篇paper的beauty在于: 把一个generative modeling的practical trick (CFG)与RL的理论结构(policy improvement via reweighting)建立了exact correspondence。这种correspondence不仅是数学上的优雅, 还带来了practical benefit——test-time controllable policy improvement without retraining。

更深层的意义是: **diffusion/flow model在RL中不只是expressive function approximator, 它们的inference mechanism (guidance, score composition)本身就对应RL的结构性操作**。这打开了一扇门, 让generative modeling的所有trick (classifier guidance, custom guidance, negative prompting等) 都可能找到RL对应物, 反之亦然。

CFGRL的simplicity是其strength: 在GCBC上加一行CFG就得到improvement, 不需要新的training infrastructure。这种"plug-and-play"特性使其极易被adopt, 与decision transformer的philosophy一脉相承——**用supervised learning的simplicity获得RL的能力**。
