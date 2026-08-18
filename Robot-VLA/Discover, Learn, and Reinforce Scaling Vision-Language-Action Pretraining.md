---
source_pdf: Discover, Learn, and Reinforce Scaling Vision-Language-Action Pretraining.pdf
paper_sha256: 0bd4e7148dd8bc62985f2f71937d7384cd611ad9436335d1fb1e09ccad2d31e2
processed_at: '2026-08-18T05:54:38-07:00'
target_folder: Robot-VLA
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# 用人话讲 DLR

## 一句话总结

**让 RL 不要只学会一招，而是学会很多招，这样产生的数据才适合拿去 pretrain VLA。**

---

## 问题是什么

现在 VLA（就是能看图、听话、然后输出动作的 robot brain）要变强，靠的是 pretrain。Pretrain 要求数据量大、花样多。

数据从哪来？三条路：

1. **人遥控机器人**——贵，慢，而且人天生懒，倾向于用最省力的方式完成任务，所以哪怕你让 100 个人做同一个任务，他们做出来的轨迹都差不多
2. **用 world model 想象出来**——能生成很多花样，但 world model 做精细操作时会累积误差，越往后越离谱
3. **用 RL 自己探索**——RL 能通过 trial and error 学会完成任务，甚至比人做得更好

听起来 RL 是最好的选择？但有个坑：**标准 RL 会塌陷到单一解法**。

想象一个 drawer 要关上。RL 可能发现 "从左边推" 能成功，reward 最高，于是它就一直从左边推。哪怕 "从右边推" 也能成功，RL 不会去用，因为它已经找到 "最优解" 了。

对于 "完成任务" 这个目标，这没问题。但你的目标是 **拿这些轨迹去 pretrain VLA**，那你需要的不是 "一个最优解"，而是 "很多种不同的成功解法"。单一解法的数据 pretrain 出来的 VLA 泛化能力差，因为它只见过一种做法。

---

## Naive 的尝试：加 diversity reward

最直觉的想法：在 task reward 之外，再加一个 diversity reward，鼓励 policy 产生不同行为。

具体怎么做？引入一个 latent variable $z$（比如 3 个 category），让 policy 变成 $\pi(a|s,z)$——给定不同 $z$ 就产生不同行为。然后加一个 mutual information term $I(Z;S)$，鼓励 "从 state 能猜出是哪个 pattern"。

但作者发现这个 naive 方法有两个致命问题：

### 问题 1：Diversity reward 会惩罚探索

Diversity reward 的核心是一个 discriminator $q_\phi(z|s)$，它看 state 猜 pattern。这个 discriminator 是 on-policy 训练的——也就是说它只认识 policy 已经访问过的 state。

当 policy 尝试探索一个新 state 时（这可能是解决任务必须经过的 state），discriminator 从没见过这个 state，猜不出 pattern，输出接近 prior（均匀分布）。这时候 diversity reward ≈ 0。

而 policy 待在熟悉 state 时，discriminator 很有信心，diversity reward 很高。

所以 policy 收到的信号是："待在熟悉的地方有奖励，去新地方没奖励"。这直接杀死了探索。Task reward 是 sparse 的（只有成功才给），没法及时补偿这个 loss。

### 问题 2：盲目追求 diversity 会偏离任务

Policy 可能学到 "往三个不同的错误方向伸手" 来骗 diversity reward，因为这些行为确实 different，但完全不解决问题。

---

## DLR 的核心 insight

**Diversity 不应该在整个 state space 上追求，而应该只在 "成功轨迹覆盖的 state" 上追求。**

换句话说：先确保任务能完成，在能完成的前提下再追求多样性。

这个 insight 听起来简单，但实现起来有个鸡生蛋问题：你怎么知道哪些 state 是 "成功轨迹覆盖的 state"？你得先能完成任务才知道。但你要能完成任务，又得探索到那些 state……

DLR 的解法是 **三个 stage 分开做**，打破这个循环：

---

## 三个 Stage

### Stage 1: Discover（发现 pattern）

用现成的人类演示数据 $\mathcal{D}_{\text{human}}$。这些数据虽然量少、花样也不算多，但至少都是成功的轨迹。

用一个 VAE 去编码这些人类轨迹的 state，把高维 state 压缩成 latent $z$。VAE 的 reconstruction objective 会强制 latent 捕获轨迹中最重要的特征。训练完之后，encoder $q_\phi(z|s)$ 就能看一个 state，告诉你它属于哪个 pattern。

这一步的本质是 **unsupervised clustering**——发现 "原来人类做这个任务时，其实有几种不同的策略"。

### Stage 2: Learn（学习每个 pattern）

用 Stage 1 的 encoder 给人类数据打 label：每个 state 标上一个 pattern $z$。

然后训一个 conditional policy $\pi(a|s,z)$，用 behavior cloning（BC）去模仿。给定 $z=1$ 就模仿 pattern 1 的轨迹，给定 $z=2$ 就模仿 pattern 2。

这一步的结果：你得到了 K 个 specialized policy，每个负责模仿一种人类策略。它们都待在 "成功 manifold" 附近（因为 BC 隐式地 minimize policy state distribution 和 expert state distribution 之间的 KL divergence）。

### Stage 3: Reinforce（用 RL 精炼每个 pattern）

BC 学到的 policy 不够好——人类数据本身就 suboptimal，BC 只能模仿到人类水平。所以用 PPO 做 online fine-tune，只用 sparse task reward（成功了给 +1）。

关键设计：**这一 stage 完全不用 diversity reward**。因为 Stage 2 已经把每个 pattern 的 policy 放到了 state space 中不同的区域，PPO 只需要在自己那个区域里做 local refinement。

---

## 为什么 PPO 不会把不同 pattern 搞混？

这是论文 theoretical analysis 回答的问题。

核心假设是 **failure moat**：不同 successful pattern 之间被 failure region（任务失败的状态）隔开。想从一个 pattern 跳到另一个，必须经过 failure region，而 failure region 的 reward = 0，不会给 policy 任何正向信号去跳。

再加上 PPO 本身有 KL constraint，限制每步 update 不能太大。所以 policy 没法 "跳过" failure moat，只能在自己那个 pattern 的 region 里 refine。

数学上，作者证明了 cross-pattern gradient 的上界：

$$\|\text{cross-pattern gradient}\| \leq \sqrt{B} \sqrt{\delta_0 + \sqrt{E/2}}$$

- $B$：score function 的 second moment 上界
- $\delta_0$：Stage 2 BC 后的 initial leakage（pattern 之间初始有多混）
- $E$：PPO 的 KL budget（每步允许偏离 initialization 多少）

当 $\delta_0$ 小（BC 学得好，patterns 分得开）且 $E$ 小（PPO 步子小）时，cross-pattern influence 很弱，每个 pattern 自己 refine 自己的。

---

## 实验结果

### 数据多样性

DLR vs 标准 RL，用四个指标衡量轨迹多样性：

| 指标 | 标准 RL | DLR | 倍数 |
|------|---------|-----|------|
| 轨迹间平均距离 | 10.5 | 26.4 | 2.5x |
| 终点方差 | 0.09 | 1.17 | 12.7x |
| 方向方差 | 0.08 | 0.09 | 1.02x |
| 路径长度方差 | 17.2 | 26.5 | 1.5x |

终点方差提升最夸张（12.7x），说明 DLR 的轨迹终点分散得多。方向方差提升最小，因为任务本身的 goal 约束限制了方向不能太散。

### Downstream 性能

在 LIBERO 的四个 downstream suite 上测试（pretrain 在 LIBERO-90，fine-tune 在下游 suite）：

**π0 模型**：
- 标准 RL pretrain：平均 13.45% success rate
- DLR pretrain：平均 16.45% success rate
- 提升 3 个百分点

**OpenVLA 模型**：
- 标准 RL pretrain：平均 9.42% success rate  
- DLR pretrain：平均 16.72% success rate
- 提升 77%

OpenVLA 上提升更大，作者认为是因为 OpenVLA 容量更大，能更好吸收 diverse data 中的信息。

### Data scaling

DLR 有 positive scaling trend：数据越多，downstream 性能越好。标准 RL 没有这个特性，因为增加数据只是重复同一个 pattern，信息量没有增加。

---

## 发现的 Pattern 长什么样

论文给了几个 qualitative 例子，很有意思：

**把书放进 caddy**：
- Pattern 1：顺时针转书对齐
- Pattern 2：逆时针转书对齐

**开炉门**：
- Pattern 1：先用 end-effector 边缘拉，再从中间推
- Pattern 2：一直用 end-effector 拉

**关炉门（有杯子挡路）**：
- Pattern 1：拿起杯子放后面，再关门
- Pattern 2：把杯子往前推让路，再关门

这些都是 **semantic meaningful 的不同策略**，不是 random variation。DLR 发现的是真正不同的 "解题思路"。

---

## 为什么这个工作重要

1. **Conceptual shift**：从 "RL 找最优解" 到 "RL 发现多种好解"。这个思维方式对 data generation 有根本性影响。

2. **Scalable data engine**：如果 RL 能自动生成 diverse data，就不再依赖昂贵的人类遥操作。Data scaling 的瓶颈从 "人力" 变成 "算力"。

3. **Theoretical grounding**：不是 ad hoc 地加 diversity bonus，而是分析了 naive 方法的 failure mode，然后 principled 地 decouple，还有定理保证 pattern preservation。

4. **Practical**：三个 stage 都是成熟技术（VAE, BC, PPO）的组合，容易实现和 deploy。

---

## 还有什么没解决

1. **依赖人类数据**：Stage 1 和 2 还是需要人类演示作为 successful manifold 的 proxy。完全去掉人类数据是下一步。

2. **Pattern 数量是手调的**：实验固定 $|Z|=3$。不同任务可能需要不同数量的 pattern，如何自动确定是 open question。

3. **只在 simulation 验证**：LIBERO 是 simulation benchmark。Real robot 的 reward specification、sim-to-real gap 都没验证。

4. **任务复杂度有限**：tabletop manipulation 相对简单。Locomotion、mobile manipulation 等更复杂任务的效果未知。

---

## 一句话再总结

DLR 告诉我们：**想让 RL 当 data generator，不能让它只找一个解，得让它找多个解。但直接加 diversity reward 会适得其反（惩罚探索），正确做法是先从人类数据里发现有哪些解法（Discover），然后分别模仿（Learn），最后各自用 RL 精炼（Reinforce）。** 这样产生的数据又多又好，pretrain 出来的 VLA 泛化更强，还能 scale。

---

# DLR: Discover, Learn, and Reinforce — 深度技术解析

## 1. 论文背景与核心问题

### 1.1 VLA Pretraining 的数据困境

Vision-Language-Action (VLA) models 遵循 large-scale pretraining → downstream fine-tuning 的 paradigm。这个 paradigm 的 efficacy critically dependent on pretraining data 的 scale 和 diversity。

目前 data 来源主要有三类：

1. **Human teleoperation**: labor-intensive, costly, behavior diversity 有限（human demonstrators natural倾向 few efficient strategies，参考 Zipf's Principle of Least Effort [71]）
2. **World model generation**: 能 generate diverse trajectories，但 current world models struggle with precise, long-horizon robot motions，suffer from accumulated error
3. **RL-generated data**: promising alternative，但 standard RL converges to single solution

### 1.2 Standard RL 的 Mode Collapse 问题

Standard policy-based RL 的 objective 是找 optimal policy：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

这个 objective 的本质是 **optimization for single best solution**。对于 specific task mastery 这是 desirable 的，但对于 generating diverse training data 是 pathological 的。即使 environment 中存在 multiple viable solutions，standard RL 会 collapse 到其中一个。

Reference: 
- DIAYN: https://arxiv.org/abs/1802.06070
- LIBERO benchmark: https://libero-project.github.io/

---

## 2. 核心技术框架：DLR

### 2.1 MDP Formulation

Agent-environment interaction 建模为 Markov Decision Process：

$$\mathcal{M} = (S, \mathcal{A}, \mathcal{P}, \mathcal{R}, \rho_0, \gamma)$$

变量含义：
- $S$: state space（所有可能的状态集合）
- $\mathcal{A}$: action space（所有可能的动作集合）
- $\mathcal{P}(s'|s,a)$: state transition function，给定当前 state $s$ 和 action $a$，转移到 $s'$ 的概率
- $\mathcal{R}(s,a)$: reward function
- $\rho_0(s)$: initial state distribution（初始状态分布）
- $\gamma \in [0,1)$: discount factor（折扣因子，衡量 future reward 的 present value）

**Discounted state visitation distribution**:

$$d^{\pi}(s) = (1-\gamma) \sum_{t=0}^{\infty} \gamma^t \Pr(s_t = s | s_0 \sim \rho_0, \pi)$$

逐项解析：
- $d^{\pi}(s)$: policy $\pi$ 下的 discounted state visitation distribution，描述 policy 在 long run 中访问各个 state 的频率
- $(1-\gamma)$: normalization constant，确保 $\sum_s d^{\pi}(s) = 1$。推导：$\sum_{t=0}^{\infty} \gamma^t = \frac{1}{1-\gamma}$，所以乘以 $(1-\gamma)$ 后归一化
- $\gamma^t$: discount factor 的 $t$ 次幂，给 timestep $t$ 的 state 一个 exponentially decaying weight。越早的 timestep 权重越大（如果 $\gamma < 1$）
- $\Pr(s_t = s | s_0 \sim \rho_0, \pi)$: 从 initial distribution $\rho_0$ 采样 $s_0$，follow policy $\pi$，在 timestep $t$ 处于 state $s$ 的概率

这个 distribution 是理解 DLR 的关键，因为它定义了 policy 在 state space 上的 "footprint"。Standard RL 会把这个 footprint 收缩到一个 narrow region，而 DLR 的目标是 maintain multiple distinct footprints。

### 2.2 Pattern-Conditioned Policy

引入 latent variable $z \in \mathcal{Z}$ 表示不同的 behavioral patterns，得到 pattern-conditioned policy $\pi_\theta(a|s,z)$。

Inference 时从 prior $p(z)$ 采样 $z$（通常是 uniform categorical distribution）。

**Pattern-conditioned objective**:

$$J(\theta) = \mathbb{E}_{z \sim p(z), \tau \sim \pi_\theta(\cdot|z)}[R(\tau)]$$

这里对 $z$ 和 $\tau$ 都取 expectation，意味着我们要 maximize 平均 over patterns 的 expected return。

### 2.3 The Naive Approach and Its Pathology

#### Naive formulation

直接 combine task reward 和 MI-based diversity bonus：

$$\max_\theta \mathbb{E}_{z \sim p(z), \tau \sim \pi_\theta(\cdot|z)}[R(\tau)] + \beta I_\theta(Z; S)$$

变量：
- $\beta \geq 0$: hyperparameter 平衡 task performance 和 diversity
- $I_\theta(Z; S)$: mutual information between latent $z$ 和 state $s$，衡量 "知道 state $s$ 能多大程度推断出 pattern $z$"

#### Variational lower bound

MI 直接计算 intractable，用 variational lower bound：

$$I_\theta(Z; S) \geq \mathbb{E}_{z \sim p(z), s \sim d^{\pi_\theta(\cdot|\cdot,z)}}[\log q_\phi(z|s) - \log p(z)]$$

变量：
- $q_\phi(z|s)$: approximate posterior，参数为 $\phi$，学习从 state 推断 pattern
- $p(z)$: prior over patterns
- $d^{\pi_\theta(\cdot|\cdot,z)}$: 给定 pattern $z$ 的 state visitation distribution

推导逻辑（reverse variational bound）：

$$I(Z; Y) = \mathbb{E}_{p(z,y)}\left[\log \frac{p(z|y)}{p(z)}\right]$$

引入 variational distribution $q_\phi(z|y)$：

$$I(Z; Y) = \mathbb{E}_{p(z,y)}[\log q_\phi(z|y) - \log p(z)] + \mathbb{E}_{p(y)} \text{KL}(p(z|y) \| q_\phi(z|y))$$

由于 KL ≥ 0，得到 lower bound。设 $Y = S$ 即可。

#### Intrinsic diversity reward

这可以 reformulate 为 standard RL objective 加上一个 intrinsic reward：

$$r_{\text{div}}(s, z) = \log q_\phi(z|s) - \log p(z)$$

所以 full objective 变成：

$$\max_\theta \mathbb{E}_{z, \tau, s}[R(\tau) + \beta \cdot (\log q_\phi(z|s) - \log p(z))]$$

### 2.4 The Conflict: Why Naive Combination Fails

这是论文最 insightful 的部分之一。作者识别出两个 fundamental conflicts：

#### Conflict 1: Reward Imbalance Penalizes Exploration

**关键观察**：$q_\phi(z|s)$ 是 on-policy 训练的，所以它 approximates empirical posterior：

$$q_\phi(z|s) \approx d^{\pi}(z|s)$$

于是 intrinsic reward 变成：

$$r_{\text{div}}(s, z) \approx \log d^{\pi}(z|s) - \log p(z)$$

**Case 1: Unseen states**。对于 policy 没访问过的 state $s$，discriminator 没有数据，prediction 会 revert to prior：

$$d^{\pi}(z|s) \approx p(z) \implies r_{\text{div}}(s, z) \approx \log p(z) - \log p(z) = 0$$

**Case 2: Highly discriminable familiar states**。对于频繁访问且高度 indicative of pattern $z'$ 的 state $s'$：

$$d^{\pi}(z'|s') \approx 1 \implies r_{\text{div}}(s', z') \approx \log 1 - \log p(z') = -\log p(z')$$

如果 $p(z) = 1/K$ (uniform over $K$ patterns)，则 maximum reward 是 $\log K$。

**Consequence**: 当 policy 试图从 familiar state $s'$ 探索到 unseen state $s$ 时，intrinsic reward 从 $\beta \log K$ 立即降到约 0，而 task reward $R(\tau)$ 在 short term 内不变（因为 sparse）。对于 sufficiently large $\beta$，$J(\theta)$ 的瞬时变化是 negative 的，所以 gradient-based updates 会 bias against exploratory moves。

**Intuition**: 这就好比一个人在熟悉的地方很有信心（discriminator 能准确判断 pattern），但一旦走到陌生地方就完全迷失（discriminator 输出 prior）。系统会 reward "待在熟悉地方" 的行为，penalize "探索新地方" 的行为，即使新地方可能有 task reward。

#### Conflict 2: Blind Diversity Distracts from Task

Agent 可能学到 distinct 但 useless 的 behaviors（比如 reach 不同的 incorrect target locations）来增加 diversity reward。这是 "diversity for diversity's sake" 的 pathology。

### 2.5 The Decoupled Objective

#### Core Insight

Diversity 不应该 across all trajectories 寻求（这会 penalize exploration），而应该 only among successful trajectories 寻求。这 ensures diversity 是 meaningful variation within successful region，而非 hindering initial discovery。

#### Formal definition

定义 successful trajectory set：

$$\mathcal{T}_{\text{succ}} = \{\tau | \mathbb{I}_{\text{succ}}(\tau) = 1\}$$

定义 successful-state manifold：

$$S^\star = \bigcup_{\tau = (s_0, a_0, \ldots) \in \mathcal{T}_{\text{succ}}} \{s_0, s_1, \ldots\} \subseteq S$$

定义 target distribution over this manifold：

$$d^\star(s) \triangleq p(s | \mathbb{I}_{\text{succ}}(\tau) = 1)$$

#### Ideal objective

$$\max_{\theta, \phi} \mathbb{E}_{z, \tau \sim \pi_\theta(\cdot|z)}[R(\tau)] + \beta I_\theta(Z; S^\star)$$

subject to: $\{s \in S | d^{\pi_\theta}(s) > 0\} \subseteq S^\star$

这个 hard constraint 确保 policy 只生成 successful trajectories。

#### Soft relaxation

$$\max_{\theta, \phi} \underbrace{\mathbb{E}_{z, \tau \sim \pi_\theta(\cdot|z)}[R(\tau)]}_{\text{Reinforce}} + \beta \underbrace{I_\theta(Z; S^\star)}_{\text{Discover}} - \alpha \underbrace{D(d^{\pi}(s) \| d^\star(s))}_{\text{Learn}}$$

变量：
- $\alpha$: soft penalty coefficient
- $D(\cdot \| \cdot)$: divergence measure（如 KL divergence）

三个 terms 分别对应三个 stages：
1. **Reinforce**: task performance
2. **Discover**: pattern diversity on successful manifold
3. **Learn**: stay on successful manifold

### 2.6 Practical Three-Stage Implementation

由于 $S^\star$ 未知，无法直接 optimize。用 human demonstrations $\mathcal{D}_{\text{human}}$ 作为 fixed proxy。

#### Stage 1: Discover

用 VAE-based framework 从 human data 中 discover latent patterns。

**VAE objective**（隐含在论文中）：

$$\max_{\phi, \psi} \mathbb{E}_{s \sim \mathcal{D}_{\text{human}}} [\mathbb{E}_{z \sim q_\phi(z|s)}[\log p_\psi(s|z)] - \text{KL}(q_\phi(z|s) \| p(z))]$$

变量：
- $q_\phi(z|s)$: encoder，从 state 到 latent pattern
- $p_\psi(s|z)$: decoder，从 latent pattern 重建 state
- $p(z)$: prior（通常是 standard normal 或 uniform categorical）

这里 encoder $q_\phi$ 学习将 successful human states 映射到 latent patterns，通过 reconstruction 强制 latent 捕获 trajectory 的 salient features。训练完成后，$q_\phi$ 作为 fixed model 用于后续 stages。

**Intuition**: 这一步相当于在 human data 上做 unsupervised clustering，发现 "原来人类解决这个任务有多种策略"。

#### Stage 2: Learn

用 trained encoder $q_\phi$ 给 human data 打 label：

$$z = \arg\max_{z'} q_\phi(z'|s)$$

得到 labeled dataset $\tilde{\mathcal{D}}_{\text{human}}$，然后 BC：

$$\max_\theta \mathbb{E}_{(s,a,z) \sim \tilde{\mathcal{D}}_{\text{human}}}[\log \pi_\theta(a|s,z)]$$

**Key theoretical result**: BC 隐式地 minimizes KL divergence between policy's state distribution 和 expert's state distribution [18]：

$$\text{BC} \implies \min_\theta D(d^{\pi}(s) \| d_{\mathcal{D}}(s))$$

其中 $d_{\mathcal{D}}(s)$ 是 human data 的 state distribution，作为 $d^\star(s)$ 的 empirical estimate。

**Intuition**: 这一步把 random initial policy 拉到 successful manifold 上，并且根据 pattern 分成 K 个 specialized policies。每个 policy 负责模仿一种 human strategy。

#### Stage 3: Reinforce

PPO fine-tune with sparse reward only：

$$\max_\theta \mathbb{E}_{z \sim p(z), \tau \sim \pi_\theta(\cdot|z)}[R(\tau)]$$

其中 $R(\tau) = \gamma^{T-1} \mathbb{I}_{\text{succ}}(\tau)$。

**Key design choice**: 这一 stage **不用** MI-based intrinsic reward。因为 BC 已经把 policy 拉到 successful manifold 附近，PPO 只需要在这个 manifold 上 refine 每个 pattern 到 optimal。

**Intuition**: Stage 2 提供 good initialization（在 successful manifold 上，且 patterns separated），Stage 3 在这个基础上做 local refinement。由于初始化已经 separated，PPO 的 local updates 不会 collapse patterns。

### 2.7 Theoretical Analysis: Why Diversity is Preserved

#### Setup

Stages 1-2 提供 K patterns $\{z_j\}_{j=1}^K$ 和 successful trajectory partition $\{\mathcal{T}_j^+\}_{j=1}^K$。定义 failure set $\mathcal{T}_0 := \mathcal{T} \setminus \cup_j \mathcal{T}_j^+$，在 $\mathcal{T}_0$ 上 $R(\tau) = 0$。

#### Cross-pattern leakage

$$p_{\text{leak}}^{(j)}(\theta) := \Pr_{\tau \sim p_\theta^j}[\tau \in \cup_{k \neq j} \mathcal{T}_k^+]$$

变量：
- $p_\theta^j(\tau)$: pattern $z_j$ 诱导的 trajectory distribution
- $p_{\text{leak}}^{(j)}(\theta)$: pattern $j$ 的 policy 生成属于其他 pattern 的 successful trajectory 的概率

这个量衡量 "pattern 之间是否会互相干扰"。

#### Four assumptions

(i) **Failure moat**: 任意从 $\mathcal{T}_j^+$ 到 $\mathcal{T}_k^+$ 的 path 都经过 $\mathcal{T}_0$。Intuition: 不同 successful patterns 之间被 failure region 隔开，要从一个 pattern 跳到另一个必须经过 failure。

(ii) **Separated init**: $p_{\text{leak}}^{(j)}(\theta_0) \leq \delta_0$。Intuition: BC 后的初始化已经 well-separated。

(iii) **Proximal PPO target-KL clipping**: $D_{\text{KL}}(p_{\theta_t}^j \| p_{\theta_0}^j) \leq E$ for all $t$。Intuition: PPO 限制每步 update 的大小，防止 policy 偏离 initialization 太远。

(iv) **Regularity**: $\mathbb{E}[\|\nabla_\theta \log p_\theta^j(\tau)\|^2] \leq B$。Intuition: score function 的 second moment 有界。

#### Theorem 1

$$\left\|\mathbb{E}_{\tau \in \cup_{k \neq j} \mathcal{T}_k^+}[\nabla_\theta \log p_{\theta_t}^j(\tau) R(\tau)]\right\| \leq \sqrt{B} \sqrt{\delta_0 + \sqrt{E/2}}$$

#### Proof sketch

**Gradient decomposition**: 对于 pattern $z_j$，policy gradient 可以分解为三部分：

$$\nabla_\theta J_j(\theta_t) = \underbrace{\mathbb{E}_{p_{\theta_t}^j}[\nabla \log p_{\theta_t}^j(\tau) R(\tau) \mathbf{1}\{\tau \in \mathcal{T}_j^+\}]}_{\text{within-pattern}} + \underbrace{\text{cross-pattern}} + \underbrace{\text{failure}}$$

**Failure term = 0**: 因为 $R(\tau) = 0$ on $\mathcal{T}_0$。

**Cross-pattern bound**: 用 Cauchy-Schwarz：

$$\|g_{\text{cross}}^{(j)}(\theta)\| \leq \sqrt{\mathbb{E}[\|\nabla_\theta \log p_\theta^j(\tau)\|^2]} \sqrt{\mathbb{E}[R(\tau)^2 \mathbf{1}\{\tau \in A_j\}]} \leq \sqrt{B} \sqrt{p_{\text{leak}}^{(j)}(\theta)}$$

**Leakage bound**: 用 Pinsker's inequality：

$$p_{\text{leak}}^{(j)}(\theta_t) = p_{\theta_t}^j(A_j) \leq p_{\theta_0}^j(A_j) + \text{TV}(p_{\theta_t}^j, p_{\theta_0}^j) \leq \delta_0 + \sqrt{E/2}$$

合起来得到 Theorem 1。

**Intuition**: 当 Stage 2 学得 well-separated（small $\delta_0$）且 Stage 3 限制 update budget（small $E$）时，cross-pattern influence 很弱，PPO updates 会 converge 到 local optimum within each pattern's region $\mathcal{T}_j^+$。

Reference for PPO: https://arxiv.org/abs/1707.06347
Reference for Pinsker's inequality: https://en.wikipedia.org/wiki/Pinsker%27s_inequality

---

## 3. 实验详解

### 3.1 Setup

- **Pretraining data source**: LIBERO-90 (90 diverse manipulation tasks)
- **Downstream evaluation**: 4 suites
  - LIBERO-Long: 10 long-horizon tasks（task composition）
  - LIBERO-Spatial: 10 tasks with changed spatial relationships
  - LIBERO-Object: 10 tasks with changed objects
  - LIBERO-Goal: 10 tasks with changed goals
- **Baselines**:
  - O2O-RL: offline-to-online RL (BC init + PPO fine-tune)，single-pattern
  - No pretraining: VLM checkpoint 直接 fine-tune
- **VLA models**: $\pi_0$ 和 OpenVLA
- **RL policy**: ResNet18 encoder + MLP head, PPO

### 3.2 Diversity Metrics (Table 1)

| Metric | O2O-RL | DLR | Improvement |
|--------|--------|-----|-------------|
| Mean Pairwise Distance (↑) | 10.487 | 26.405 | 2.5x |
| Endpoint Variance (↑) | 0.092 | 1.170 | 12.7x |
| Direction Variance (↑) | 0.083 | 0.085 | 1.02x |
| Path Length Variance (↑) | 17.193 | 26.546 | 1.54x |

**Metric definitions**:

1. **Mean Pairwise Distance**:

$$\text{Dist}_{\text{mean}} = \frac{1}{N(N-1)/2} \sum_{i<j} \|\tau_i - \tau_j\|_2$$

衡量所有 trajectory pairs 之间的平均距离。$N$ 是 trajectory 数量，$\tau_i$ 是 trajectory 的 feature embedding。

2. **Endpoint Variance**:

$$\text{Var}_{\text{end}} = \text{Var}(\{s_{T_i}^{(i)} | i = 1, \ldots, N\})$$

衡量 final states 的多样性。$s_{T_i}^{(i)}$ 是 trajectory $i$ 的 final state。

3. **Direction Variance**:

$$\text{Var}_{\text{dir}} = \text{Var}\left(\left\{\frac{s_{T_i}^{(i)} - s_0^{(i)}}{\|s_{T_i}^{(i)} - s_0^{(i)}\|_2}\right\}_{i=1}^N\right)$$

衡量 motion direction 的多样性。$s_0^{(i)}$ 和 $s_{T_i}^{(i)}$ 是 trajectory $i$ 的 initial 和 final states。

4. **Path Length Variance**:

$$\text{Var}_{\text{len}} = \text{Var}\left(\left\{\sum_{t=0}^{T_i-1} \|s_{t+1}^{(i)} - s_t^{(i)}\|_2\right\}_{i=1}^N\right)$$

衡量 path lengths 的多样性。

**Intuition**: DLR 在 Mean Pairwise Distance 上提升 2.5x，在 Endpoint Variance 上提升 12.7x，说明 DLR 生成的 trajectories 在 state space 上覆盖更广，endpoints 更分散。Direction Variance 提升较小（1.02x）是因为 task 本身有 goal constraint，direction 不能太分散。

### 3.3 Downstream Performance (Table 2)

| Data Source | Spatial | Object | Goal | Long | Avg. |
|-------------|---------|--------|------|------|------|
| **π0** | | | | | |
| No Pretraining | 0.60 | 0.00 | 0.20 | 0.00 | 0.20 |
| O2O-RL | 4.40 | 15.20 | 19.20 | 15.00 | 13.45 |
| DLR (Ours) | 6.00 | 18.60 | 22.40 | 18.80 | 16.45 |
| **OpenVLA** | | | | | |
| No Pretraining | 1.60 | 2.60 | 17.04 | 2.20 | 5.86 |
| O2O-RL | 11.60 | 3.04 | 19.11 | 3.94 | 9.42 |
| DLR (Ours) | 19.88 | 3.67 | 34.20 | 9.21 | 16.72 |

**Key observations**:

1. DLR consistently outperforms O2O-RL across both VLA architectures
2. OpenVLA 上 DLR 的提升更显著（9.42 → 16.72，提升 77%），特别是在 Spatial (11.60 → 19.88) 和 Goal (19.11 → 34.20) suites
3. No pretraining 的 baseline 表现很差，说明 pretraining 的重要性

### 3.4 Data Scaling (Figure 7)

论文展示了 positive scaling trend：随着 RL-generated data 量增加，downstream performance 提升。这是 single-pattern RL 缺乏的特性。

**Intuition**: Single-pattern RL 的 data 增加只是重复同一个 pattern，information content 没有增加。DLR 的 data 增加会 cover 更多 variations，information content 持续增长。

### 3.5 MI-Shaped Objective Negative Result (Appendix E)

论文还实验了 naive MI-shaped objective (Eq. 7)，发现它 generate visually diverse 但 unsuccessful 的 trajectories。这 empirically 验证了 Section 4.2 的 theoretical analysis。

Reference: 
- π0: https://arxiv.org/abs/2410.24164
- OpenVLA: https://openvla.github.io/

---

## 4. Architecture 与 Pipeline 解析

### 4.1 Data Generation Pipeline (Figure 3)

```
For each task in LIBERO-90:
    1. Train lightweight RL policy using DLR
       - Stage 1: VAE discover patterns from human demos
       - Stage 2: BC train pattern-conditioned policy
       - Stage 3: PPO refine with sparse reward
    2. Collect high-quality trajectories via policy rollouts
    3. Each pattern generates color-coded trajectory subset

Combine data from all 90 tasks → VLA pretraining dataset
```

### 4.2 VLA Pretraining and Fine-tuning

```
Pretraining:
    - SFT on RL-generated data
    - Models: π0, OpenVLA
    - Input: language instruction + visual observation
    - Output: low-level actions
    - Loss: NLL of actions conditioned on instruction + observations

Fine-tuning:
    - On downstream LIBERO suites
    - < 3 epochs
    - Early stopping based on validation success rate
    - 50 rollouts per task for evaluation
```

### 4.3 RL Policy Architecture

```
Input: 
    - Third-person RGB observation
    - Low-dimensional proprioceptive features (end-effector pose, gripper state)

Encoder:
    - ResNet18 backbone → feature vector
    
Concatenation:
    - Visual features + proprioceptive features + latent z embedding (for DLR)

Output:
    - MLP head → continuous actions
```

---

## 5. Qualitative Results 深度解读

### 5.1 Discovered Patterns (Figure 4)

**Task 1: Pick up book and place into caddy**
- Pattern A: Rotate book clockwise to align
- Pattern B: Rotate book counterclockwise to align

**Task 2: Open stove**
- Pattern A: Pull with end-effector's edge, then center push
- Pattern B: Consistently pull with end-effector

**Task 3: Close stove (with cup blocking door)**
- Pattern A: Pick up cup, place backward, then close
- Pattern B: Push cup forward to clear path, then close

**Intuition**: 这些 patterns 都是 semantically meaningful 的 alternative strategies。DLR 不是 generate random variations，而是 discover 真正 different approaches to solve the same task。

### 5.2 Trajectory Visualization (Figure 5)

Task: Close bottom drawer of cabinet

- **O2O-RL**: 所有 trajectories 几乎 identical，converge 到 single dominant strategy
- **DLR**: trajectories 有 higher variance，cover 更宽的 state space area

DTW (Dynamic Time Warping) distance 衡量 trajectory 之间的 similarity，DLR 的 DTW distances 明显更大。

### 5.3 t-SNE Visualization (Figure 6)

- **O2O-RL**: trajectories cluster into few dense clusters → mode collapse
- **DLR**: trajectories spread widely across embedding space in multiple distinct clusters → multi-modal coverage

### 5.4 Distance Matrices (Figure 8)

- **O2O-RL**: uniform, low distances → all trajectories similar
- **DLR**: clear block-diagonal structure → distinct, well-separated behavioral modes

**Intuition**: Block-diagonal structure 说明 DLR 的 trajectories 自然分成几个 groups，每个 group 内部相似，group 之间差异大。这正是 multi-pattern data 的理想结构。

---

## 6. Intuition Building: Why DLR Works

### 6.1 The Core Insight

Standard RL 的 objective 是 **find the best solution**。DLR 的 objective 是 **find multiple good solutions**。这两个 objective 有 fundamentally different implications for data generation。

**Analogy**: 想象你在教一个 robot 抓取物体。Standard RL 会找到 "最有效" 的抓取方式，然后不断重复。DLR 会找到 "几种不同的有效抓取方式"，每种都达到 task success，但 approach 不同。

### 6.2 Why Decoupling is Necessary

MI-based diversity reward 的问题在于它是 **self-referential** 的：discriminator $q_\phi$ 是 on-policy 训练的，所以它只认识 policy 已经访问过的 states。当 policy 探索 new state 时，discriminator 无法判断 pattern，reward drop。这 creates 一个 **anti-exploration bias**。

DLR 的 decoupling 在于：
1. **Discover stage**: 用 fixed human data 训练 discriminator，decouple from policy's exploration
2. **Learn stage**: BC 把 policy 拉到 successful manifold
3. **Reinforce stage**: 只用 task reward，不用 MI reward，避免 anti-exploration bias

### 6.3 Why Theoretical Guarantee Holds

Theorem 1 的核心在于 **failure moat** assumption。不同 successful patterns 之间被 failure region 隔开。要从一个 pattern 跳到另一个，必须经过 failure region，而 failure region 的 $R(\tau) = 0$，不会提供 positive learning signal。

加上 PPO 的 KL constraint 限制 update size，policy 无法 "跳跃" failure moat，所以会 stay 在 initial pattern's region 内 refine。

**Analogy**: 想象 state space 是一个 landscape，successful patterns 是几个 islands，failure region 是 ocean。PPO 是一个不会游泳的探险者，只能在 island 上 local explore，无法跨 ocean 到另一个 island。

### 6.4 Why Data Diversity Translates to Downstream Performance

VLA pretraining 的目标是 expose model to diverse behaviors，让它 acquire broad manipulation capabilities。如果 pretraining data 只有 single pattern，model 只学到一种 approach。当 downstream task 需要 different approach 时，model 无法 adapt。

DLR 的 multi-pattern data 让 model 在 pretraining 时就接触到 multiple approaches，builds a richer prior。当 downstream task 需要 specific approach 时，model 可以 leverage 这个 richer prior。

**Connection to in-context learning**: 多种 patterns 相当于 multiple "modes" in the model's action distribution。Fine-tuning 时，model 可以 select 和 amplify 相关的 mode，suppress 不相关的 mode。这比从 single-mode model 学习 new mode 要容易得多。

### 6.5 The Scaling Story

Figure 7 的 positive scaling trend 是 DLR 的关键 selling point。Single-pattern RL 缺乏 scaling 是因为 data 增加只是重复同一 pattern，marginal information content 递减。DLR 的 data 增加会 cover 更多 variations，marginal information content 保持。

这 positions multi-pattern RL as a **practical, scalable data engine** for embodied foundation models。

---

## 7. 与 Related Work 的对比

### 7.1 vs. DIAYN and Skill Discovery Methods

DIAYN [14] 等 skill discovery methods 纯粹 optimize diversity (MI)，不考虑 task success。这导致 discovered skills 可能 useless for task。

DLR 的关键区别是 **decouple diversity from exploration**，只在 successful trajectories 上 seek diversity。

Reference: https://arxiv.org/abs/1802.06070

### 7.2 vs. Offline-to-Online RL

O2O-RL [41, 42] 是 standard paradigm：BC init + online fine-tune。但它 converge to single solution。

DLR 的区别在于 pattern-conditioned policy 和 three-stage decoupling。

### 7.3 vs. RL for VLA Fine-tuning

之前的工作 [9, 10, 16, 20, 29, 36, 38] 用 RL 直接 fine-tune VLA models。这 costly 且 slow，因为 VLA models 很大。

DLR 的策略是 train small policies 来 generate data，然后用 data pretrain large VLA。这 shifts expensive RL optimization away from large VLA model。

### 7.4 vs. World Model Data Generation

World model approaches [1, 44, 55] 能 generate diverse data，但 suffer from accumulated error 和 struggle with precise manipulation。

DLR 用 real environment rollouts，避免 accumulated error，同时通过 multi-pattern design 保证 diversity。

---

## 8. Limitations 和 Future Directions

### 8.1 Current Limitations

1. **依赖 human demonstrations**: Stage 1 和 Stage 2 都需要 human data 作为 successful manifold 的 proxy。如果 human data 不存在或 quality 差，DLR 的效果会受影响。

2. **Pattern number fixed**: 实验中 $|Z| = 3$ 是预设的。如何 automatically determine optimal pattern number 是 open question。

3. **Simulated environment only**: 实验在 LIBERO simulation 上进行。Real robot 的 reward specification 和 environment interaction 更 challenging。

4. **Limited task complexity**: LIBERO 主要是 tabletop manipulation。更 complex tasks (locomotion, mobile manipulation) 的效果未验证。

### 8.2 Future Directions

1. **Combine with automated task/environment generation**: 论文 conclusion 提到这个方向。如果 task 和 environment 也能 automatically generated，就形成 fully autonomous data pipeline。

2. **Hierarchical patterns**: 目前 patterns 是 flat 的。Hierarchical structure (patterns within patterns) 可能 capture 更 rich behavioral structure。

3. **Cross-task pattern transfer**: 不同 task 的 patterns 可能有 semantic relationship（比如 "approach from left" vs "approach from right"）。Leverage 这些 relationships 可能 improve efficiency。

4. **Real robot deployment**: 把 DLR 应用到 real robot，验证 sim-to-real transfer。

---

## 9. 总结

DLR 的核心 contribution 是一个 **principled three-stage framework** 来 generate diverse, high-quality robotic trajectories for VLA pretraining。

**Key technical insights**:

1. **Decouple diversity from exploration**: MI-based diversity reward 会 penalize exploration，因为 discriminator 无法判断 unseen states 的 pattern。Solution: 只在 successful trajectories 上 seek diversity。

2. **Three-stage approximation**: 因为 successful-state manifold $S^\star$ 未知，用 human data 作为 proxy，分三个 stages 分别 approximates 三个 objective terms。

3. **Theoretical guarantee**: Under failure moat, separated init, KL constraint, regularity assumptions，PPO updates 会 converge to local optimum within each pattern's region，preserving diversity。

**Empirical evidence**:

1. DLR generates 2.5x more diverse trajectories (Mean Pairwise Distance)
2. VLA models pretrained on DLR data outperform O2O-RL by 3% (π0) to 7% (OpenVLA)
3. Positive data scaling trend

**Broader impact**: 这篇 paper positions multi-pattern RL as a practical, scalable data engine for embodied foundation models，potentially shifting from human-centric to algorithmically generated data pipelines。

Reference papers for deep dive:
- DLR paper: https://arxiv.org/abs/2509.19752 (推测，基于 author 的 previous work)
- LIBERO: https://libero-project.github.io/
- π0: https://arxiv.org/abs/2410.24164
- OpenVLA: https://openvla.github.io/
- DIAYN: https://arxiv.org/abs/1802.06070
- PPO: https://arxiv.org/abs/1707.06347
- VAE: https://arxiv.org/abs/1312.6114
- ResNet: https://arxiv.org/abs/1512.03385
- GAIL (BC minimizes KL): https://arxiv.org/abs/1606.03476

---

## 10. 个人思考与 Open Questions

### 10.1 Pattern Number 的选择

论文固定 $|Z| = 3$，但没有讨论如何选择。直觉上，pattern number 应该 depend on task complexity。Simple task 可能只有 2 个 viable patterns，complex task 可能有 10+ 个。

Possible approach: 用 Bayesian non-parametric methods (如 Dirichlet process) 让 pattern number 自动 emerge from data。

### 10.2 Pattern Quality 的评估

论文用 diversity metrics 评估 trajectories，但没有 directly 评估 patterns 的 quality。一个 "good" pattern 应该是：
1. Semantically meaningful（人类能理解）
2. Transferable（能 transfer to downstream tasks）
3. Composable（能 compose with other patterns）

### 10.3 Connection to Curriculum Learning

DLR 的 patterns 有 difficulty gradient（有些 patterns 可能 easier，有些 harder）。这 natural connection to curriculum learning：先用 easier patterns pretrain，再 introduce harder patterns。

### 10.4 Connection to Model-Based RL

DLR 是 model-free 的。如果 combine with model-based RL，可能能 generate even more diverse data，因为 model 可以 imagine trajectories 不需要 real interaction。但 model 的 accumulated error 是 concern。

### 10.5 Connection to Diffusion Policies

Diffusion policies [31, 60] 天然 support multi-modal action distributions。DLR 的 multi-pattern data 可能特别适合 pretrain diffusion-based VLA models，因为 diffusion 能 explicitly represent multiple modes。

Reference: 
- Diffusion policy: https://arxiv.org/abs/2303.04137
- Diffusion for RL: https://arxiv.org/abs/2305.13122

这篇 paper 的 contribution 不仅是 technical framework，更是 conceptual shift：从 "RL finds the best solution" 到 "RL discovers multiple good solutions"。这个 shift 对 embodied AI 的 data scaling 有深远影响。
