---
source_pdf: Reinforcement Learning with Action Chunking.pdf
paper_sha256: 1d2f8cfdbc6c0e9e04bf0fd37de5a381334ba890d94ec6b33228346fe3d240f0
processed_at: '2026-08-11T22:18:47-07:00'
target_folder: RL
model: z-ai/glm-5.2
reasoning_effort: max
followup_prompt: 用人话说说
mineru_required_version: 3.4.4
---

# Q-Chunking 人话版

## 一句话说清楚

**这篇论文发现：在RL里，与其每一步都重新决定动作，不如一次决定未来h步的动作，这样既能让探索更"连贯有章法"，又能让value学习更快还不引入偏差。**

## 这论文要解决什么痛点？

想象你训练一个机械臂去搬运4个方块到指定位置（cube-quadruple任务）。这个任务 horizon 长（1000步），reward sparse（只有方块放对了才有reward）。

**传统offline-to-online RL的死穴**：

1. **从offline data学出来的policy太保守**。IQL、CQL这类方法为了防止offline data外的action over-estimation，会加pessimism。结果online探索时agent像个怂包，不敢离开data覆盖的区域，永远学不会新行为。

2. **1-step TD value传播太慢**。γ=0.99时，effective horizon = 100步。value要从goal反向传到起点，需要100次backup。online data每来一个新transition，value只能往前推1步。long-horizon任务里这是灾难。

3. **探索是random walk**。Gaussian policy每步加随机噪声，机械臂就在原地抖动，根本走不到需要放方块的位置。

## Q-chunking的招数

### 招数一：把action从"1步"扩展到"一个chunk"

原来：policy输出 $a_t$（1个动作）
现在：policy输出 $(a_t, a_{t+1}, a_{t+2}, a_{t+3}, a_{t+4})$（5步的动作序列）

这个chunk用open-loop执行：采一次样，连续5步都按这个chunk走，5步后再重新决策。

**为什么这个对探索好？**

想想人怎么探索陌生环境。你不会每0.1秒重新随机决定走哪边，那样你就在原地打转。你会决定"先往左走10米看看"，这是一个temporally coherent的探索行为。

Offline data里其实有大量这种"连贯行为"——scripted policy、人类遥操作都不是随机抖动，而是一段一段的meaningful motion。Action chunking就是让policy直接从data里学到"什么样的5步序列是合理的"，然后在online探索时复用这种"skill"。

Figure 5特别直观：BFN（1-step baseline）的末端执行器轨迹在抓方块附近有大片dense cluster（原地抖动），而QC的轨迹是流畅的曲线，覆盖更大空间。

### 招数二：Q-function也吃整个chunk

原来：$Q(s_t, a_t)$
现在：$Q(s_t, a_{t:t+h})$ — Q函数评估"在状态$s_t$执行这个5步动作序列好不好"

**这步是关键，解决了n-step return的bias问题**。

先说n-step return为什么有bias。传统n-step return是这样：
$$Q(s_t, a_t) \leftarrow r_t + \gamma r_{t+1} + ... + \gamma^{h-1} r_{t+h-1} + \gamma^h Q(s_{t+h}, a_{t+h})$$

这里 $r_t$ 到 $r_{t+h-1}$ 是从replay buffer取的，由**当时collect data的policy**产生的。但 $a_{t+h}$ 是**现在policy**采样的。两个policy不一样，rewards的分布和Q的分布对不上，有系统性偏差。

Q-chunking的修正很巧妙：
$$Q(s_t, a_{t:t+h}) \leftarrow \sum \gamma^{t'-t} r_{t'} + \gamma^h Q(s_{t+h}, a_{t+h:t+2h})$$

注意这里 $r_{t:t+h}$ 是由 $a_{t:t+h}$ 产生的，而 $a_{t:t+h}$ 就是Q函数的输入！所以reward和Q的input完全对应，没有policy mismatch。

Theorem A.1用几行就证了：因为reward和Q-input是同一组action，n-step return对 $Q^\pi(s_t, a_{t:t+h})$ 是unbiased的。

**结果**：value传播快了h倍（每次backup推h步），还没有bias。这是paper最漂亮的点。

### 招数三：用flow-matching policy + behavior constraint

光有action chunking不够。Figure 2证明：如果用Gaussian policy + naive BC loss，性能反而比不用chunking更差。

为什么？因为offline data是multi-modal的。人类演示里"把方块放左边"和"放右边"是两个完全不同的motion pattern。Gaussian只能model一个mode，强行学就是两边都学不好。

**Solution**：用flow-matching policy（类似diffusion但更简单）建模behavior distribution。Flow-matching学习一个velocity field，把Gaussian noise通过ODE积分成action chunk。这个能capture任意复杂的multi-modal分布。

然后有两种behavior constraint方式：

**QC（implicit KL）**：从flow policy采N个chunk，用Q选最好的那个执行。N越大constraint越松。这个KL上界是 $\log N - (N-1)/N$。

**QC-FQL（explicit Wasserstein-2）**：训练一个noise-conditioned one-step policy $\mu_\psi(s, z)$，loss里加一项让它输出接近flow policy的输出。$\alpha$ 控制constraint强度。

## 为什么这套组合work？

三个机制互相强化：

1. **Chunked exploration**：policy采样的是"5步连贯动作"，不是5个独立随机动作。Flow policy从data学到这些chunk的分布，保证它们都是合理的"skill"。

2. **Unbiased fast backup**：Q-function吃chunk后，TD backup天然是n-step的，传播快，没有n-step return的bias。

3. **Behavior constraint**：防止policy为了maximize Q而跑出offline data的manifold。这个constraint在chunk space比1-step space更有意义，因为data的non-Markovian结构在chunk层面才可见。

## 实验结论

**主表（Table 1）**：

最难的cube-quadruple任务：
- IQL: 0% → 0%（完全失败）
- FQL: 0% → 8%
- FQL-n（n-step return）: 1% → 37%（初期好，但Figure 15显示会collapse）
- **QC: 4% → 74%**
- **QC-FQL: 1% → 77%**

Q-chunking比FQL好9倍。这个gap非常惊人。

**n-step return会collapse**：Figure 15里FQL-n在cube-quadruple初期学到一定success rate，然后崩塌到0。这就是bias累积的后果——Q被over-estimate，policy被带跑偏，正反馈循环崩塌。Q-chunking完全没这个问题。

**chunk size h的影响**（Figure 6 left）：
- h=1：等价FQL，最差
- h=5-10：最好
- h=25：初期快但asymptotic差
- h=50：失败

太小没benefit，太大policy反应迟钝且难学。h=5是性价比最高的选择。

**Gaussian policy不行**（Figure 2）：RLPD-AC（Gaussian + chunking + BC loss）反而比RLPD（不用chunking）还差。必须用expressive policy。

## 我的几点直觉

### 1. Action chunking是RL的正确inductive bias

long-horizon任务的真实结构就是"一段一段的skill"。把方块从A移到B是5步连贯动作，不是5个独立决策。Markovian assumption在最优policy上成立，但在探索阶段，temporally extended行为才有效率。

这跟IL里ACT、Diffusion Policy的发现一致：人类数据天然是chunked的。Q-chunking把这个insight搬到RL，还额外利用了unbiased backup的bonus。

### 2. n-step return的bias被低估了

学术界用n-step return很久了（Rainbow, IMPALA都用了），但offline-to-online setting下这个bias的严重性没被充分讨论。这篇paper的Figure 15是一个警示：online阶段bias会累积放大，导致catastrophic collapse。

Q-chunking的解法很优雅——不是用importance sampling去correct bias（那个variance太大），而是重新定义Q函数的input让bias自然消失。

### 3. Flow policy > Gaussian policy

这个在FQL paper里已经证明了，Q-chunking进一步强化。Gaussian policy是RL的默认选择，但在offline-to-online场景下，behavior distribution的复杂度远超Gaussian能表达的范围。Flow-matching相当于"现代版diffusion"，训练稳定，推理快（10步Euler就够）。

### 4. 与HRL的关系

Q-chunking本质是HRL的special case：low-level skill是open-loop action sequence，固定长度h，no termination function。但这种"简化版HRL"避免了HRL的核心痛点——bi-level optimization的moving target问题。因为这里没有high-level和low-level两个网络互相追逐，只有一个policy在chunk space里学。

这也解释了为什么Q-chunking比SUPE-GT（skill-based HRL方法）效果好：simple is better。

### 5. 与LLM tokenization的类比

这是我自己觉得最有趣的联系：

- 1-step RL ≈ character-level language model（每个决策都是local的）
- Action chunking ≈ token-level LM（一个chunk是一个meaningful unit）

Tokenization让LM训练更稳定、sample-efficient，因为credit assignment自然落在token边界。Action chunking让RL的credit assignment落在chunk边界，TD backup的"step"变成chunk级别的step，传播效率提升h倍。

这个类比还能push：未来能不能learn adaptive chunk boundaries？能不能hierarchical chunking（chunk of chunks）？能不能用VQ-VAE学action的"codebook"？这些都是open direction。

## 代码与复现

- Paper: https://arxiv.org/abs/2502.05258
- Code: https://github.com/ColinQiyangLi/qc
- OGBench: https://arxiv.org/abs/2410.20092
- FQL: https://arxiv.org/abs/2502.02538
- Flow Matching: https://arxiv.org/abs/2209.03003
- ACT (IL里的action chunking): https://arxiv.org/abs/2304.13705

总compute约10000 GPU hours，单run 4-12小时，RTX-A5000即可跑。

---

# Q-Chunking: Action Chunking for Offline-to-Online RL 深度讲解

## 1. 核心问题与 motivation

Offline-to-online RL 的核心难题在于：如何从一个**可能由 sub-optimal、multi-modal、non-Markovian** 行为组成的 offline dataset $\mathcal{D}$ 出发，在 online phase 实现高效 exploration？传统方法（IQL, CQL, AWAC）倾向于过度 pessimistic，导致 online exploration 时 stuck 在 local optima。

Q-chunking 的 insight 极其优雅：**虽然 fully observable MDP 的 optimal policy 是 Markovian 的，但 exploration 问题用 temporally extended、non-Markovian 的 action sequence 处理更高效**。同时，action chunking 提供了一个 "two birds with one stone" 的解决方案——既改善 exploration coherency，又消除 n-step return 的 off-policy bias。

## 2. 核心方法：Q-learning on Temporally Extended Action Space

### 2.1 三种 backup 方式的对比

这是全文最核心的 insight，必须 deep dive：

**Standard 1-step TD**（Eq 5）:
$$Q(s_t, a_t) \gets r_t + \gamma Q(s_{t+1}, a_{t+1})$$

每个 backup step 仅传播 1 步 value，horizon $\tilde{H} = 1/(1-\gamma)$ 越长，propagation 越慢。

**n-step return**（Eq 6）:
$$Q(s_t, a_t) \gets \underbrace{\sum_{t'=t}^{t+h-1} \gamma^{t'-t} r_{t'}}_{\text{biased!}} + \gamma^h Q(s_{t+h}, a_{t+h})$$

变量说明：
- $r_{t'} = r(s_{t'}, a_{t'})$：time $t'$ 的 reward
- $\gamma^{t'-t}$：从 $t$ 到 $t'$ 的 discount factor
- $a_{t+h} \sim \pi_\psi(\cdot | s_{t+h})$：**current policy** 采样的 action

**Bias 来源**：rewards $r_{t:t+h}$ 由 **data collection policy $\pi_\beta$** 产生，但 $Q(s_{t+h}, a_{t+h})$ 的 $a_{t+h}$ 由 **current policy $\pi_\psi$** 产生。当 $\pi_\beta \neq \pi_\psi$ 时，$\sum \gamma^{t'-t} r_{t'}$ 不是 $E_{\pi_\psi}[\sum \gamma^{t'-t} r_{t'}]$ 的 unbiased estimate。

**Q-chunking backup**（Eq 7）:
$$Q(s_t, a_{t:t+h}) \gets \underbrace{\sum_{t'=t}^{t+h-1} \gamma^{t'-t} r_{t'}}_{\text{unbiased!}} + \gamma^h Q(s_{t+h}, a_{t+h:t+2h})$$

变量说明：
- $a_{t:t+h} = (a_t, a_{t+1}, \ldots, a_{t+h-1})$：一个长度为 $h$ 的 action chunk
- $a_{t+h:t+2h} \sim \pi_\psi(\cdot | s_{t+h})$：next chunk，由 current policy 产生
- Q-function 接受**整个 chunk**作为输入

**为什么 unbiased**？因为 $Q(s_t, a_{t:t+h})$ 中的 $a_{t:t+h}$ 与产生 rewards $r_{t:t+h}$ 的 actions 是同一组 actions。Theorem A.1 形式化证明：当 $\hat{V}(s_{t+n})$ 对 $V^\pi(s_{t+n})$ unbiased 时，n-step return 对 $Q^\pi(s_t, a_t, \ldots, a_{t+n-1})$ unbiased。

### 2.2 训练目标

实际的 TD loss（Eq 4）:
$$L(\theta) = \mathbb{E}_{s_t, a_{t:t+h}, s_{t+h} \sim \mathcal{D}} \left[ \left( Q_\theta(s_t, a_{t:t+h}) - \sum_{t'=1}^{h} \gamma^{t'} r_{t+t'} - \gamma^h Q_{\bar{\theta}}(s_{t+h}, a_{t+h:t+2h}) \right)^2 \right]$$

变量说明：
- $\theta$：critic 参数
- $\bar{\theta}$：target network 参数（EMA of $\theta$）
- $a_{t+h:t+2h} \sim \pi_\psi(\cdot | s_{t+h})$：next chunk

## 3. Behavior Constraint：实现 Temporally Coherent Exploration

### 3.1 为什么需要 behavior constraint？

Offline data 通常包含 non-Markovian 结构（scripted policies [56]、human tele-operators [43]、noisy sub-task policies）。Markovian 的 behavior constraint（如 Gaussian BC）无法捕获这种 structure。Figure 2 的实验证明：naïvely 用 Gaussian policy + action chunking + BC loss（QC-RLPD）反而表现更差。

**关键结论**：必须有 expressive policy（flow-matching/diffusion）来建模 multi-modal behavior distribution。

### 3.2 QC: Implicit KL via Best-of-N Sampling

约束目标（Eq 8）:
$$L(\psi) = -\mathbb{E}_{s_t \sim \mathcal{D}, a_{t:t+h} \sim \pi_\psi} [Q_\theta(s_t, a_{t:t+h})], \quad \text{s.t. } D(\pi_\psi \| \pi_\beta) \leq \varepsilon$$

QC 采用 **best-of-N sampling** 实现隐式 KL 约束：

1. 从 flow-matching behavior policy 采样 N 个 chunks：$\{a^1, a^2, \ldots, a^N\} \sim f_\xi(\cdot | s)$
2. 选择 Q-value 最大的：$a^* = \arg\max_{a \in \{a^1, \ldots, a^N\}} Q(s, a)$

KL upper bound（Eq 10, 来自 Hilton 2023 [27]）:
$$D_{KL}(a^* \| f_\xi(\cdot | s)) \leq \log N - \frac{N-1}{N}$$

变量说明：
- $N$：采样数，越大约束越弱
- 当 $N=1$，bound = 0（完全模仿 behavior policy）
- 当 $N \to \infty$，bound $\to \infty$（无约束，纯 greedy）

**Intuition**：best-of-N 是一种 implicit Q-weighted policy improvement，与 AWR、AWAC 的 explicit KL 有联系，但避免了 log-likelihood 计算的困难（flow models 难以计算 log-prob）。

### 3.3 QC-FQL: Explicit Wasserstein-2 Constraint

QC-FQL 基于 FQL [58]，使用 2-Wasserstein distance：
$$W_2(\pi_\psi, f_\xi(\cdot | s)) \leq \varepsilon$$

通过 noise-conditioned policy $\mu_\psi(s, z): \mathcal{S} \times \mathbb{R}^{Ah} \mapsto \mathbb{R}^{Ah}$ 实现，loss（Eq 13）:
$$L(\psi) = \mathbb{E}_{s_t \sim \mathcal{D}, z^0 \sim \mathcal{N}(0, I_{Ah})} \left[ \alpha \| z^1 - \mu_\psi(s_t, z^0) \|_2^2 - Q(s_t, \mu_\psi(s_t, z)) \right]$$

变量说明：
- $z^0 \sim \mathcal{N}(0, I_{Ah})$：input noise（维度 $A \cdot h$，$A$ 是 action dim）
- $z^1$：从 $z^0$ 通过 flow ODE $\mathrm{d}z^u = f_\xi(s_t, z^u, u) \mathrm{d}u$ 积分到 $u=1$ 得到的 target action chunk
- $\alpha$：behavior regularization 系数（Table 4 中各 domain 不同，OGBench 100-300，robomimic 10000）

distillation loss $\|z^1 - \mu_\psi(s_t, z^0)\|_2^2$ 是 $W_2^2$ 的 upper bound（FQL paper 中证明）。

## 4. Flow-Matching Behavior Policy

Flow-matching [41] 学习一个 velocity field $f_\xi(s, z, u): \mathcal{S} \times \mathbb{R}^{Ah} \times [0,1] \mapsto \mathbb{R}^{Ah}$。

训练 loss（Eq 22）:
$$L(\xi, w) = \| f_\xi(s_t, u[a_t, \ldots, a_{t+h-1}] + (1-u)z_t, u) - ([a_t, \ldots, a_{t+h-1}] - z_t) \|_2^2$$

变量说明：
- $u \sim U([0,1])$：flow time
- $z_t \sim \mathcal{N}(0, I_{Ah})$：source noise
- $[a_t, \ldots, a_{t+h-1}]$：target action chunk（来自 data）
- 训练目标是让 velocity field 在 $u \to 1$ 时指向 data，$u \to 0$ 时指向 noise

生成时用 Euler 方法（Algorithm 3）:
```
m^0 ← z_t
for i in {1, ..., T}:
    m^i ← m^{i-1} + f_ξ(s_t, m^{i-1}, (i-1)/T) / T  # 实际上是 m^i = m^{i-1} + f_ξ * (1/T)
return m^T
```

实际实现中 $T=10$ steps，比 diffusion policy 的 100+ steps 快得多。

## 5. 实验深度分析

### 5.1 环境设置

Table 2 的 domain metadata：

| Task | Dataset Size | Episode Length | Action Dim |
|------|-------------|----------------|------------|
| scene-sparse-* | 1M | 750 | 5 |
| puzzle-3x3-sparse-* | 1M | 500 | 5 |
| cube-double-* | 1M | 500 | 5 |
| cube-triple-* | 3M | 1000 | 5 |
| cube-quadruple-100M-* | 100M | 1000 | 5 |
| lift / can / square | 31-80K | 500 | 7 |

**Critical observation**：cube-triple 和 cube-quadruple 是真正的 long-horizon challenge，episode length 1000，需要移动 3-4 个 cube 到 target location。这些任务在 offline 阶段几乎 zero success rate，必须靠 online exploration 解决。

### 5.2 主结果分析（Table 1）

最关键的对比：

**Cube-quadruple（最难）**：
- QC: 4 → 74（offline → online）
- QC-FQL: 1 → 77
- FQL: 0 → 8.3
- FQL-n: 1 → 37
- BFN: 1 → 12
- IQL: 0 → 0
- RLPD: 0 → 20

QC 比 FQL 高 9 倍！这是 action chunking + behavior constraint 的巨大优势。

**n-step return 的 collapse 问题**（Figure 15）：
FQL-n 在 cube-quadruple 上初期有 success，但迅速 collapse 到 0。这正是 n-step return bias 的后果——bias 在 online phase 累积导致 Q-value over-estimation 和 policy collapse。

Q-chunking 完全没有这个问题，因为 backup 是 unbiased 的。

### 5.3 Action Chunk Size $h$ 的影响（Figure 6 left）

在 cube-triple 上的 ablation：
- $h=1$：等价于 FQL，性能最差
- $h=5$：性能提升明显
- $h=10$：性能最佳
- $h=25$：早期快但 asymptotic 差
- $h=50$：完全失败

**Intuition**：
- $h$ 太小：失去 temporal coherence 的好处
- $h$ 太大：policy reactivity 下降（无法响应环境变化），且预测 $h$-length sequence 的 network 难度增大

实际上 $h$ 类似 HRL 中 option 的 duration，存在 exploration vs reactivity trade-off。

### 5.4 Critic Ensemble Size $K$（Figure 6 center）

$K=10$ 比 $K=2$ 对 QC 和 BFN 都有提升。这与 REDQ, TD3 的 ensemble insights 一致：reduce over-estimation bias。在 long-horizon 任务中，over-estimation 问题尤其严重。

### 5.5 Temporal Coherency Analysis（Figure 5）

测量 end-effector 位置的 $L_2$ norm difference：
- QC: 高 temporal coherency（动作连贯）
- BFN: 低 temporal coherency（很多 pause 和 jitter）

BFN 的轨迹在 pickup cube 时有"dense cluster near center"——即 pause。这种 jittery exploration 在 long-horizon 任务中致命，因为无法完成需要 sustained motion 的 sub-task。

## 6. 与相关工作的深度对比

### 6.1 vs. ACT / Diffusion Policy (IL)

ACT [90], Diffusion Policy [11] 在 IL 中使用 action chunking 处理 multi-modal human demonstrations。Q-chunking 的差异：
- IL 中 action chunking 为了 model non-Markovian demonstrations
- RL 中 action chunking 为了 **exploration** 和 **unbiased value backup**
- Q-chunking 还训练 chunked Q-function，IL 中不需要

### 6.2 vs. HRL / Options Framework

Options framework [75] 中 low-level policy 有 initiation set, termination function。Q-chunking 是 special case：
- Initiation set：所有 states（无限制）
- Termination：固定 $h$ 步后强制终止
- Low-level policy：open-loop action sequence

**关键优势**：collapse bi-level optimization 为单层 RL，避免 high-level policy 优化 moving target 的不稳定性。

### 6.3 vs. FQL [58]

FQL = Q-chunking with $h=1$。FQL 用 flow-matching policy + W2 constraint，已是 strong baseline。Q-chunking 的提升完全来自 extended action space。Table 1 显示：
- FQL: 37 → 58 (overall)
- QC-FQL: 38 → 86 (overall)
- 几乎 50% 的提升来自 action chunking

### 6.4 vs. RLPD [7]

RLPD 把 offline data 当作 off-policy data 与 online replay buffer 50/50 混合，从 scratch 训练。RLPD-AC（加 action chunking 但无 behavior constraint）表现差，说明 chunking alone 不够，**必须配合 behavior constraint**。

### 6.5 vs. Seo & Abbeel [65]

他们的 RLAS 也 train critic on action chunks，但用 multi-level factorized critic（coarse-to-fine discretization）。Q-chunking 的优势：
- 无需 factorization 假设
- 用 expressive flow policy 而非 discretized bins
- 连续 action space 上更灵活

### 6.6 vs. Li et al. [38] (TOP-ERL)

TOP-ERL 用 Gaussian policy + motion primitives 在 episodic RL 中实现 chunking。Q-chunking 差异：
- Offline-to-online setting（非 episodic）
- Flow-matching policy（非 Gaussian，Figure 2 证明 Gaussian 失败）
- 直接在 raw action space 预测（非 motion primitive 参数化）

### 6.7 vs. Cal-QL [51]

Cal-QL 调节 offline 阶段的 pessimism。Q-chunking 完全 different mechanism：通过 action chunking 加速 value propagation 和改善 exploration，不直接处理 pessimism。两者可能互补。

## 7. 算法细节（Algorithm 1 & 2）

### QC Algorithm 1 pseudocode 解析

```
Input: Behavior policy f_ξ(a_{t:t+h}|s) and critic Q_θ(s_t, a_{t:t+h})
D ← offline prior data

for every env step t:
    if t mod h ≡ 0:  # 每 h 步才重新决策
        {a_{t:t+h}^i}_{i=1}^N ~ f_ξ(·|s_t)  # 采 N 个 chunks
        a_{t:t+h}^* ← argmax_i Q_θ(s, a_{t:t+h}^i)  # best-of-N
    Act with a_t^*  # 执行 chunk 的第 t mod h 个 action
    Receive s_{t+1}, r_t
    D ← D ∪ {(s_t, a_t^*, s_{t+1}, r_t)}
    Update f_ξ via flow-matching loss
    Update Q_θ via Eq 11
```

**关键 implementation detail**：每 $h$ 步才调用一次 policy（open-loop 执行 chunk），这降低 inference cost $h$ 倍。

### QC-FQL Algorithm 2

```
Input: f_ξ, Q_θ, μ_ψ(s, z)
for every env step t:
    if t mod h ≡ 0:
        z ~ N(0, I_{Ah})
        a_{t:t+h} ← μ_ψ(s_t, z)  # one-step policy
    Act with a_t
    Update f_ξ (flow-matching), μ_ψ (actor loss), Q_θ (TD loss)
```

## 8. 重要的 Implementation Tricks

### 8.1 Cube-quadruple 的 100M dataset 处理

100M transitions 无法 fit CPU memory。Solution：每 1000 gradient steps 加载 1M chunk。Online phase 只用固定 1M chunk + replay buffer（其余 99M 不用）。这解释了为什么 Q-chunking 在 cube-quadruple 上优势巨大——它更 sample-efficient，不需要海量 data。

### 8.2 Hyperparameter Sensitivity

Table 4 的 $\alpha$ 值差异巨大：
- OGBench: 100-300
- Robomimic: 10000

这说明 flow policy 与 behavior distribution 的 scale 强相关。Robomimic 的 human demonstrations 可能 variance 更大，需要更强 regularization。

### 8.3 Network Architecture

- 4 hidden layers, 512 width
- Critic ensemble $K=2$（default），$K=10$ 更好但 expensive
- Action chunk $h=5$（default）
- Flow steps $T=10$（Euler integration）

## 9. 我的 Intuition 总结

### 9.1 为什么 action chunking 在 RL 中有效？

传统 RL 的 1-step action space 中，exploration 是 random walk——long-horizon 任务中几乎不可能到达 goal。Action chunking 把 action space 扩展到 sequence space，相当于在 **skill space** 中 exploration。

Offline data 中的 trajectory segments 提供了 "skill library"。Flow-matching policy 学会了这个 library 的 distribution，best-of-N sampling 在这个 library 中选择 Q-value 最高的 skill。

**类比**：1-step RL 像在 pixel space 生成图像（困难），action chunking 像在 latent space 生成（容易，structured）。

### 9.2 为什么 n-step return collapse 而 Q-chunking 不会？

n-step return 的 bias 是**系统性**的：当 $\pi_\beta$ 比 $\pi_\psi$ 差时，$r_{t:t+h}$ 系统性偏低，Q-value 系统性 under-estimated。当 $\pi_\beta$ 比 $\pi_\psi$ 好时（offline data 包含 expert demos），Q-value 系统性 over-estimated。

这种 bias 在 online phase 累积：policy 根据 over-estimated Q 选择 actions，但实际 reward 更低，Q 被 further over-estimated（positive feedback loop）。最终 policy collapse。

Q-chunking 完全避免这个问题，因为 Q-function 的 input 与 reward-generating actions 完全一致。Theorem A.1 保证了 unbiasedness。

### 9.3 为什么 flow-matching policy 至关重要？

Gaussian policy 是 unimodal 的，无法 capture：
- Multi-modal human demonstrations（lift cube vs place cube 的不同 motion patterns）
- Discrete choices（gripper open vs close）
- Temporal correlations within a chunk

Flow-matching 通过 ODE 在 continuous time 演化 noise 到 action chunk，能 represent 任意 complex distribution。Figure 2 中 QC-RLPD（Gaussian + BC loss）失败证明了这一点。

### 9.4 与 Model-Based RL 的对比

MuZero, Dreamer 等 model-based RL 通过 latent dynamics model 做 planning。Q-chunking 的不同：
- 不学 dynamics model
- Q-function 直接评估 action sequence value
- 不需要 multi-step rollout（inference 更快）

但 model-based 方法可以做更长 horizon 的 planning（hundreds of steps），Q-chunking 受限于 $h$。Hybrid 可能是 future direction。

## 10. Limitations & Future Directions

Paper Section 6 提到：
1. **Fixed chunk size $h$**：需要 task-specific tuning。Adaptive chunk boundary 是 natural next step。
2. **Open-loop execution**：不适合 high-frequency feedback task。
3. **Non-Markovian policy subclass**：action chunking 只是 limited case。

我补充几个方向：
- **Variable-length chunks**：用 termination function学习 adaptive chunk size
- **Closed-loop chunking**：chunk 内允许部分 observation feedback
- **Hierarchical chunking**：multi-level chunking（chunk of chunks）
- **与 world model 结合**：用 latent dynamics model 加速 Q-chunking 训练
- **LLM as high-level policy**：LLM 生成 action chunks 的 sub-goals

## 11. 关键 References

- **Q-chunking paper**: [arXiv:2502.05258](https://arxiv.org/abs/2502.05258) (Li, Zhou, Levine, 2025)
- **Code**: [github.com/ColinQiyangLi/qc](https://github.com/ColinQiyangLi/qc)
- **FQL**: [arXiv:2502.02538](https://arxiv.org/abs/2502.02538) (Park, Li, Levine, 2025)
- **OGBench**: [arXiv:2410.20092](https://arxiv.org/abs/2410.20092) (Park, Frans, Eysenbach, Levine)
- **ACT**: [arXiv:2304.13705](https://arxiv.org/abs/2304.13705) (Zhao, Kumar, Levine, Finn)
- **Diffusion Policy**: [arXiv:2303.04137](https://arxiv.org/abs/2303.04137) (Chi et al.)
- **Flow Matching**: [arXiv:2209.03003](https://arxiv.org/abs/2209.03003) (Liu, Gong, Liu)
- **RLPD**: [arXiv:2302.02948](https://arxiv.org/abs/2302.02948) (Ball, Smith, Kostrikov, Levine)
- **IQL**: [arXiv:2110.06169](https://arxiv.org/abs/2110.06169) (Kostrikov, Nair, Levine)
- **Options Framework**: Sutton, Precup, Singh (AI 1999) - [doi.org/10.1016/S0004-3702(99)00052-2](https://doi.org/10.1016/S0004-3702(99)00052-2)
- **Best-of-N KL bound**: [jacobh.co.uk/bon_kl.pdf](https://www.jacobh.co.uk/bon_kl.pdf) (Hilton 2023)
- **Robomimic**: [arXiv:2108.03298](https://arxiv.org/abs/2108.03298) (Mandlekar et al.)
- **EMaQ**: [arXiv:2102.01257](https://arxiv.org/abs/2102.01257) (Ghasemipour et al.)
- **Seo & Abbeel RLAS**: [arXiv:2411.12155](https://arxiv.org/abs/2411.12155)
- **TOP-ERL**: [openreview](https://openreview.net/forum?id=N4NhVN30ph)
- **Cal-QL**: [arXiv:2403.14534](https://arxiv.org/abs/2403.14534) (Nakamoto et al.)

## 12. 实验复现估算（Table 3 + Appendix D）

Compute: NVIDIA RTX-A5000, 单 run 4-12 hours
- OGBench: 6 domains × 15 methods × 25 tasks × 4 seeds ≈ 9000 GPU hours
- Robomimic: 10 hours × 8 methods × 3 tasks × 5 seeds ≈ 1350 GPU hours
- 总计 ~10,350 GPU hours

Hyperparameter tuning：
- QC: sweep $N \in \{2, 4, 8, 16, 32, 64, 128\}$，在 task2 上选 best
- QC-FQL: $\alpha \in \{\alpha_{default}/3, \alpha_{default}, 3\alpha_{default}\}$
- SUPE-GT: KL coeff $\in \{0.001, 0.003, 0.01, 0.03, 0.1\}$

## 13. 最终 Intuition 总结

Q-chunking 的核心 elegance 在于：**用一个简单的 action sequence representation 同时解决了 RL 的两个核心难题**。

1. **Exploration 问题**：通过 flow-matching policy 学到 skill distribution，best-of-N 在 skill space 中 explore，避免 1-step random walk 的低效。

2. **Value propagation 问题**：Q-function over action chunks 实现了 **unbiased n-step backup**，无需 importance sampling 或其他 correction。这是 paper 的 theoretical contribution。

3. **Distribution shift 问题**：Behavior constraint 在 chunk space 比 1-step space 更有意义，因为 offline data 的 non-Markovian structure 在 chunk level 才能被 capture。

这个工作让我想到 Karpathy 你在 deep learning 中常说的 "inductive bias matters"。Action chunking 不是 free lunch——它假设任务有 temporal structure 可以被 exploit。对于 truly myopic tasks（如 Atari Pong 的瞬间反应），$h=1$ 可能更好。但对于 manipulation, navigation 等 long-horizon tasks，chunking 是正确的 inductive bias。

最后一个直觉：Q-chunking 与 LLM 中的 token-level prediction 有有趣的类比。1-step RL 像 character-level LM（每个 decision 都是 local），action chunking 像 token-level LM（一个 chunk = 一个 meaningful unit）。Token-level LM 训练更稳定、sample-efficient，因为 credit assignment 更自然。Q-chunking 在 RL 中实现了类似的 "temporal tokenization"。
