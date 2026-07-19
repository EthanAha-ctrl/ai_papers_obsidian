---
source_pdf: Affordances Enable Partial World Modeling with LLMs.pdf
paper_sha256: 1f41d68737cfa7a5c858fe642d12853158d5a901a7d8f44f80886ac1d73d63bb
processed_at: '2026-07-18T03:14:36-07:00'
target_folder: World-model
model: z-ai/glm-5.2
reasoning_effort: max
mineru_required_version: 3.4.4
---

# Affordances Enable Partial World Modeling with LLMs - 深度解读

## 1. Paper 的核心 motivation

这篇paper要解决一个非常 fundamental 的问题：**LLM 到底能不能当 world model 用？**

传统MBRL (Model-Based Reinforcement Learning) 的思路是学一个 `P(s'|s,a)` 来approximate真实环境，然后planning。Hao et al. 2023 RAP (https://arxiv.org/abs/2305.14992) 这类工作直接把 LLM 当作 `P_LLM(s'|s,a)` 用，问题是：

- LLM inference cost 极高，MCTS里反复rollout 烧不起
- LLM hallucination严重，rollout error 会 compound
- 机器人 proprioception 和 LLM 文本输入 format mismatch

作者的核心 insight：**don't use LLM as a full world model, use it as a partial world model**。Partial model (Talvitie & Singh 2009, https://papers.nips.cc/paper/2009/hash/a9418f1e6c1b0f5c0f4f5f9b9b6b9b9b9b-Abstract.html) 只对state-action space 的relevant subset做高质量预测。这正好匹配 LLM 的特点—— LLM 知识是 "internet-scale but unevenly distributed"，在 task-relevant subset 上准确度更高。

**Affordances 就是定义这个 "relevant subset" 的钥匙**。Gibson 1977 的 affordance 概念 (https://en.wikipedia.org/wiki/Affordance) 说的是 agent-environment interface 上的 action possibilities。这篇文章 formal 地把 affordance → partial model 这条 chain 写出来了。

---

## 2. 三层 conceptual framework

看 Figure 1，整个 framework 有三个 axes：

- **Agent axis**：robot embodiment，比如 gripper arm 的物理能力
- **Environment axis**：physical workspace，比如 table top 的几何边界
- **Task distribution axis**：`Dist(T)` 的 multi-task setting

Affordances 在 agent-environment 的 boundary 上 emerge，并且 generalize across task distribution——这就是 "distribution-robust" 的来源。

### Intents 的二元分类（这是 paper 的关键 abstraction）

- **Task-agnostic intents `I_agn`**：grounded on agent embodiment。例如 "pick up red block"——只要红色 block 在桌上且上面没东西，gripper 就能抓。这与任务无关，build tower、clean table、sort color 都成立。
- **Task-specific intents `I_task`**：依赖具体任务 context。例如 "stack red on blue"——只在 build-a-stack 任务里有意义，如果换成 put-all-in-box 任务，stacking 反而错误。

这个分类让 affordance 在 multi-task 场景下保持 generalization，是 Khetarpal et al. 2020 (https://arxiv.org/abs/2007.02633) 和 2021 (https://arxiv.org/abs/2110.06011) 单任务版本的 multi-task 推广。

---

## 3. 关键 definitions 详解

### Definition 1: Temporally Extended Intent `I_o`

```
I_o : S → Dist(Γ)
```

- `o ∈ O`：abstract action（比如 pick-up-block，实际是一段 controller / option）
- `S`：state space
- `Γ`：trajectory space（不是单点 state，是整个轨迹分布）
- `P_I(τ|s,o) = I_o(s,τ)`：intent model

为什么是 trajectory distribution 而不是 single outcome？因为同一个 task 可以多种方式完成。例如"抓红色block到 (x,y)"，可以是直接抓起来放，也可以是先推开别的block再抓。intent 描述的是 outcome 的 distribution，而不是 deterministic success state。

### Definition 2: Degree of intent satisfiability ζ

```
d(P_I(τ|s,o), P_truth(τ|s,o)) ≤ ζ, ∀(s,o,τ)
```

- `d`：user-specified probability distribution metric（如 KL divergence、Wasserstein）
- `ζ`：threshold，越小越严格
- `P_truth`：实际 agent-environment 产生的 trajectory distribution

ζ = 0 意味着 intent 完美匹配 reality；ζ = 1 意味着完全 misspecified。

### Definition 3: ζ-affordance set

```
AF_I ⊆ S × O
```

是所有 "intent 能被满足到 ζ 程度" 的 (state, action) pair 集合。

例子：
- Robot hand 在 block 旁边，run "pick-up-block" → block 进入 gripper → intent 满足 → 这是 affordance
- Robot hand 在房间另一边，run "pick-up-block" → 抓空气 → intent 不满足 → 不是 affordance

### Definition 4: Distribution-Robust Agent Affordances（multi-task 扩展的核心）

```
P(d(P_I(τ|s,o,T), P_truth(τ|s,o,T)) ≥ ζ | T ~ Dist(T)) ≤ δ
```

- `T`：从 task distribution `Dist(T)` 采样的任务
- `δ ∈ (0,1)`：distribution-robust 的 failure probability
- `ζ`：intent satisfiability threshold

含义：在 task distribution 上 random sample task T 时，affordance 不满足 intent 的概率被 δ bound 住。

这把 single-task affordance 推到 multi-task：同一个 affordance（比如 pick up unobstructed block）在大多数 task 下都成立 → distribution-robust；只在特定情境下成立（比如把 block 放进另一个 hollow block）→ task-specific。

### Definition 5: Depth-n tasks

`T(n) = {T^i : T^i = ⟨o_{i,0}, ..., o_{i,m}⟩, m ≤ n}`

任务可由 ≤ n 个 primitive sub-task（action `O=o`、outcome `X(s∈g)`、goal `F(s∈g)`）的 sequence 表达。

`X` 是 LTL 的 "Next" 算子，`F` 是 "Eventually" 算子。这是 Linear Temporal Logic (https://en.wikipedia.org/wiki/Linear_temporal_logic) 的 standard formalism。

例子：
- "move red block from x to y" = `⟨O=o, X(s∈g)⟩`——先 take action，再确保 outcome
- "pick up red block eventually" = `⟨F(s∈g)⟩`——只要某时刻达到即可

### Definition 6: (n, ζ, δ)-optimal agent

```
P(R(π_i, T^i) ≥ ζ | T^i ~ T(n)) ≤ δ
```

其中 regret：
```
R(π, T^i) := max_{π'} P_{π'}(τ ⊨ T^i) - P_π(τ ⊨ T^i)
```

- `π_i`：agent 在 task `T^i` 上的 policy
- `R(π, T^i)`：相对于 optimal policy 的 regret（reward 是 task 完成为 +1，否则 0）

含义：agent 在 depth-n task distribution 上有高概率（≥1-δ）满足 ζ-regret bound。这是 "competent planner" 的 formal definition。

---

## 4. Theorem 1 详解 — 这是 paper 最核心的理论

**Statement**: 一个 deterministic (n, ζ, δ)-optimal agent 的 policy π 隐含 encoding 了一个 partial world model `P̂_par(s'|o,s)`，其 worst-case error：

$$
P(|\hat{P}_{par}(s'|o,s) - P(s'|o,s)| \geq \phi + \epsilon) \leq \frac{\rho^{n\epsilon}}{1-\rho}
$$

其中：
- `φ = (1/2)·√((1+ζ) / (n·(1-ζ)))`：characteristic error scale
- `ρ = 2·√(δ·(1-δ))`：exponential decay factor（注意当 δ<1/2 时 ρ<1）
- `ε > 0`：deviation parameter
- `n`：task depth（任务复杂度）
- `ζ`：intent satisfiability
- `δ`：distribution robustness

### Intuition：

1. **如果 agent 能在 depth-n task 上规划得好，那它的 policy 就隐含 encode 了一个 partial world model**。这是 inverse 的问题——从 behavior 反推 model。
2. **Error 的 characteristic scale `φ` 关于 `n` 是 `O(n^{-1/2})`**：任务越复杂（n 越大），从 policy 反推 model 的精度越高。这是 Richens et al. 2025 (https://arxiv.org/abs/2506.01622) "General agents need world models" 这条思路的延续。
3. **`φ` 关于 ζ 是 monotonic increasing 的**：intent mis-specification 越严重（ζ 接近 1），partial model 越不准。当 ζ→1 时 `φ→∞`，bound 没意义。
4. **`ρ^{nε}` 指数衰减**：当 δ<1/2 时 `ρ<1`，所以 deviation 超过 `φ+ε` 的概率关于 `nε` 指数 decay。这意味着**对深度规划能力强的 agent，partial model 的恢复精度是 exponential 保证的**。

### Proof sketch（Appendix B.1）：

构造一个 verification task：让 agent 在 state `s` 下 take action `o_1` n 次，要求 ≤k 次 terminate 在 `s'`。这等价于 binomial experiment，success probability 就是 `p = P(s'|o,s)`。

通过观察 agent 在不同 k 上的 policy choice（选 `o_1` 还是 `o_2`），用 Chebyshev inequality + Chernoff bound 反推出 `p̂`（中位数估计）与 `p` 的偏差 bound。

关键的 random walk 分析：
- `f(k,n) ∈ {0,1}`：agent 对不同 k 返回的 action choice
- 拟合 step function `F(k̂)`
- 误差 `E(k̂, n) = Σ_{k≤k̂} f(k) + Σ_{k>k̂} (1-f(k))`
- 转化为 random walk `S(k) = Σ Y_i`，其中 `Y_i = 2f(k) - 1`，drift `μ = E[Y_i] = 1-2δ`
- Chernoff bound on `P(W_l ≥ 0)` 给出 `ρ^l = (2√(δ(1-δ)))^l`

最后用 `p(1-p) ≤ 1/4` 简化 `√(p(1-p)·(1+ζ)/(n·(1-ζ)))` 得到 `(1/2)·√((1+ζ)/(n·(1-ζ)))`。

---

## 5. Theorem 2 详解 — 搜索效率的保证

**问题**：partial model 可能 miss 一些必要的 intent，导致某些 task 永远解不开。如何在享受 partial model 加速的同时 robust 到 missing intent？

### Corrected Partial Model

定义 `P_cor(I_j | ε_cor)`：

$$
P_{cor}(I_j | \varepsilon_{cor}) = (1-\varepsilon_{cor}) \cdot \frac{1}{k} \cdot [I_j \in \mathcal{I}_{par}] + \varepsilon_{cor} \cdot \frac{1}{n-k} \cdot [I_j \notin \mathcal{I}_{par}]
$$

变量含义：
- `ε_cor ∈ [0,1]`：exploration probability for missing intents
- `k = |I_par|`：partial model 的 intent 数量
- `n = |O|`：全部 abstract action 数量
- `[·]`：indicator function

边界 case：
- `ε_cor = 0`：恢复为纯 partial model
- `ε_cor = (n-k)/n`：恢复为 full model（uniform over all actions）

### Adaptive Corrected Model

由于最优 `ε_cor*` 是 task-dependent 的 unknown，作者构造 adaptive model：在每次 sampling L 长度 sequence 之前，先 uniform 采样 `m ∈ {0, 1, ..., 2L+2}`，设 `ε_cor = m/(2L+2)`。这是经典的 "doubling trick" / "grid search over hyperparameter" 思路。

### Theorem 2 Statement:

$$
N^{ada} \leq e \cdot (2L+3) \cdot \sum_{i \leq |\mathcal{T}|} \frac{1}{\max_{\varepsilon_{cor} \in [0,1]} P_{cor}(\text{success}_i | \varepsilon_{cor})}
$$

变量含义：
- `N^ada`：adaptive model 解所有 task 所需的期望样本数
- `L`：sampled sequence 长度
- `e`：自然常数（≈2.718）
- `2L+3`：grid 的离散点数
- `max_{ε_cor}`：对每个 task 取最优 `ε_cor` 下的 success probability

### Intuition：

**代价只有 `O(L)` 的 factor，但能享受 partial model 潜在的 exponential speedup**。

对比分析：
- Full model：`N_full = n^L · Σ_i 1/Σ P(success|J)` —— branching factor 是 n
- Partial model：`N_par = k^L · ...` —— branching factor 是 k << n，**潜在 (n/k)^L speedup**
- 但 partial model 万一 missing intent，`N_par = ∞`
- Adaptive corrected model：用 `ε_cor > 0` 给 missing intent 一个 small probability，**保证 finite**，且只付出 `O(L)` 的代价

这是经典 multi-armed bandit / online learning 的 "experts" 思路：你不知道哪个 `ε_cor` 好，就 hedge 一点，全部试一遍，最多亏 logarithmic factor。这里亏的是 `O(L)` 而不是 `O(log)`，因为 ε_cor 是 continuous 而 m 是离散 grid。

### Proof 核心 Lemma 1:

对每个 task i，存在 `m ∈ {0,...,2L+2}` 使得对任意 sequence J：

$$
\prod_{I_j \in J} P_{cor}(I_j | \varepsilon_{cor_i^*}) \leq e \cdot \prod_{I_j \in J} P_{cor}(I_j | m/(2L+2))
$$

Proof 分两个 case：
- `ε* ≤ 1/2`：选 `m = ⌈(2L+2)·ε*⌉`，利用 `ln(1+x) ≤ x` 推出 `(L-m_J)·ln((1-ε*)/(1-ε)) + m_J·ln(ε*/ε) ≤ 1`
- `ε* > 1/2`：对称处理

---

## 6. 实验设计详解

### Setup

- Simulator：PyBullet tabletop robotics
- Tasks：3, 5, 7 blocks 的 rearrangement task
- High-level instruction：`move_blocks_close_to_each_other`
- LLM：text-only large model（应该是 Gemini/GPT 类，paper 没明说具体 model name）

### 三种方法对比

1. **Full Model w/ Few-Shot**: `P_LLM(s'|s,o)` 直接用 LLM 4-shot prompt 预测
2. **Partial Model (Ours)**: 先用 LLM 作 affordance model `£_affordances(s_t, m)` 生成 m 个 affordable actions，再用 `P_AFF-LLM(s'|s,o)` 在 affordance 上预测
3. **Partial Model w/ Oracle Affordances**: 用 programmatic ground truth affordances，看 LLM affordance 引入的 bias

### Task-Agnostic Intents 实例（Appendix A.2）

这些是 prompt 里的 task-agnostic intent specifications：

1. "A block is movable only if that color is present in the current state and when nothing is on top of it."
2. "A block placement on (x,y) which has an existing block is valid if it exactly matches the coordinates of the existing block in the current state AND if the block color is present in the current state."
3. "A block placement on (x,y) which has an existing block is valid if it has sufficient overlap with the coordinates of the existing block in the current state AND if the block color is present in the current state."
4. "A block placement on (x,y) is valid if no other block is present in those coordinates."

注意这些都是 **embodiment-grounded** 的：只要 gripper 能 reach，颜色在，没遮挡，就 afford。这些与具体 task 是 build-tower 还是 sort-color 无关。

### MCTS 实现细节

- 4 simulations per planning step
- 每个 simulation expand 一个 unvisited node（random policy over untried actions）
- Random rollout 到 depth 10
- Reward +10 at rewarding states
- True reward model 用在每个 node 上

---

## 7. 实验数据深度解读

### Table 1 (3-blocks & 5-blocks):

| Method | Model | MC-Search Score | Simulations | LLM Calls | Steps to Completion |
|--------|-------|-----------------|-------------|-----------|---------------------|
| Full Model | P_LLM | 0.0 | >4 | 40.0 | >10 |
| Partial Model (Ours) | P_AFF-LLM | 3.75 | 2.75 | 43.25 | 5 |
| Oracle Affordances | P_Oracle-AFF-LLM | 5.0 | 2.5 | 15.75 | 2 |

**Key Observations**:

1. **Full Model 完全失败**：MC-Score 0.0，4 simulations 都没找到 reward state。这验证了 paper 的核心论点—— LLM 作为 full world model 在 planning 上不可用，hallucination 和 branching factor 联手让 search 失效。

2. **Partial Model 成功**：MC-Score 3.75，只需 2.75 simulations 就找到 reward，5 steps 完成。LLM calls 43.25 略高于 full model 的 40.0——因为要 affordance query + world model query 两次 LLM call。

3. **Oracle 的上界**：MC-Score 5.0，2.5 simulations，2 steps。这是 affordance 完美时的 ceiling。差距（5.0 vs 3.75）来自 LLM 生成 affordance 的 noise/bias。

4. **LLM calls 比较**：Partial Model 43.25 vs Oracle 15.75——partial model 多 query 了 ~27 次用于 affordance generation。但仍然比 full model 的失败结果好得多。

### Table 2 (5-blocks & 7-blocks):

5-blocks:
| Method | Avg Leaf-Node Score | Avg Tree Score | Avg Accumulated Reward | Online Policy Eval |
|--------|---------------------|-----------------|------------------------|---------------------|
| Full Model | 0.089 (±0.04) | 0.089 (±0.04) | 0.65 (±1.26) | 0.7895 (±0.25) |
| Partial Model | 0.235 (±0.087) | 0.2314 (±0.086) | 0.9212 (±0.709) | 1.56 (±0.45) |

7-blocks:
| Method | Avg Leaf-Node Score | Avg Tree Score | Avg Accumulated Reward | Online Policy Eval |
|--------|---------------------|-----------------|------------------------|---------------------|
| Full Model | -0.01134 (±0.057) | -0.01134 (±0.057) | -0.1869 (±0.51) | -0.836 (±0.1059) |
| Partial Model | 0.0180 (±0.12) | 0.020 (±0.12) | 0.386 (±0.57) | 0.2141 (±0.617) |

**Key Observations**:

1. **Scaling 行为**：blocks 数从 5 → 7，full model 的 score 从 0.089 掉到 -0.011，online policy eval 从 0.79 掉到 -0.84。Branching factor 爆炸让 full model 完全失效。
2. **Partial model 在 7-blocks 仍然 positive**：虽然 absolute performance 下降（0.235 → 0.018），但仍然 > 0。这验证了 affordance 把 branching factor 从 n^L 压到 k^L 的效果。
3. **Variance 比较**：7-blocks partial model 的 variance ±0.12 比 full model ±0.057 大，说明 LLM affordance 在 harder task 上 less stable。
4. **Distribution-robust 的体现**：3-blocks 的 task-agnostic intents 直接迁移到 5/7-blocks，没有重新设计——这就是 Definition 4 的 distribution-robustness 在实验上的体现。

### Figure 3 解读

这个 plot 用 simulator（perfect world model）+ 程序化 affordance（可控 accuracy）来 isolate affordance accuracy 的影响：

- **Green star**：affordance accuracy 高 → search policy 表现好
- **Red star**：affordance misspecified → catastrophic failure，即使 world model perfect

**Insight**：affordance 出错比 world model 出错更致命。因为 affordance 决定了 search tree 的 branching structure，错 affordance 直接 prune 掉正确路径。

---

## 8. 与相关工作的关系

### Hao et al. 2023 RAP (https://arxiv.org/abs/2305.14992)

RAP 把 LLM 当 world model 在 Blocksworld (PDDL) 和 GSM8k 数学推理上用。差异：
- RAP 是 full model（限制 action 数为固定 sampling），本文是 partial model（affordance-guided action filtering）
- RAP 任务 i.i.d.（GSM8k），本文是 sequential long-horizon tabletop
- RAP 的 action space 限制是 ad-hoc，本文有 formal 的 affordance framework

### Bruce et al. 2024 Genie (https://arxiv.org/abs/2402.15391)

Genie 用 200k hours 视频训练 generative interactive environment。Limitation 是会产生 unrealistic futures（hallucination）。本文方法是 complementary 的——可以给 Genie 加 affordance layer，只在 affordable latent actions 上预测。

### Benechehab et al. 2024 (https://arxiv.org/abs/2410.11711)

LLM 作 transition model，但因为 inference cost，用 state-only dynamics `P(s'|s)` 而非 state-action `P(s'|s,a)`。本文是 online search + planning，他们做 offline RL。本文的 affordance 框架实际能解决他们的 inference cost 问题——只 query LLM 在 affordable (s,a) 上。

### Khetarpal et al. 2020 (https://arxiv.org/abs/2007.02633) & 2021 (https://arxiv.org/abs/2110.06011)

这是本文的 direct theoretical 前作。原本是 single-task，hand-designed intent-completion programs，tabula rasa 学习。本文：
1. 推广到 multi-task（Definition 4）
2. 用 LLM 替代 hand-designed intent programs
3. 提供了 partial model 隐含在 policy 中的 inverse 形式证明（Theorem 1）

### Richens et al. 2025 (https://arxiv.org/abs/2506.01622)

"General agents need world models"——证明 general agent 必然 encode world model。本文 Theorem 1 是这个思路的 partial version + distribution-robust extension。本文的 (n, ζ, δ)-optimality 是 Richens 的 strict regret bound 的 relaxation。

### Hafner et al. 2024 DreamerV3 (https://arxiv.org/abs/2301.04104)

世界模型在 multi-domain 上 scale，但需要从头训练。本文是 "plug in pre-trained LLM as partial model" 的轻量级方案。

### Talvitie & Singh 2009 Partial Models (https://papers.nips.cc/paper/2009/hash/a9418f1e6c1b0f5c0f4f5f9b6b9b9b9b-Abstract.html)

Partial models 的原始论文，只预测部分 state-action space。本文在这个基础上加了 affordance 的 formal 链接和 multi-task distribution-robust 概念。

---

## 9. Intuition building — 为什么这套 framework 是对的

### Intuition 1: LLM 知识是 unevenly distributed 的

LLM 在 internet-scale 数据上训练，对 "block is movable when nothing on top" 这类 common sense 知得很好，但对 "specific (x,y) 坐标精确预测" 知得差。Partial model 让 LLM 只在它擅长的 affordance subset 上工作，避开它的弱点。

### Intuition 2: Branching factor 是 planning 的杀手

考虑 7-blocks，每个 state 下可能有 ~50 个 valid actions，深度 10 的 search tree 有 50^10 ≈ 10^17 nodes。即使每个 LLM call 只要 100ms，10^17 × 0.1s ≈ 3×10^9 年。Partial model 把 valid actions 砍到 5-10 个，5^10 ≈ 10^7，可计算。

### Intuition 3: Affordance 是 hierarchical abstraction

Affordance 本质上是 "domain knowledge as filter"——把 low-level action space 的指数复杂度，通过 high-level intent satisfaction 检查，压缩到 polynomial。这和 hierarchical RL (Feudal Networks, Options framework) 是相通的思路，但本文是 model-based 而非 policy-based。

### Intuition 4: Distribution-robust 是 transfer 的 formal 保证

Definition 4 的 `δ`-robustness 意味着 affordance 在 task distribution 上有 ≥1-δ 的概率成立。这给了 transfer learning 一个 Bayesian interpretation：在 `Dist(T)` 上训练的 affordance belief 可以直接迁移到新 task。

### Intuition 5: Adaptive correction 是 hedge

Theorem 2 的 adaptive model 是 online learning 的 classic hedge——你不知道 partial model 够不够，就给 missing intent 一个 small probability ε。这种思路在 **multi-armed bandit (EXP3)**、**MDP 中的 ε-greedy exploration**、**algorithm portfolio (SATzilla)** 里都有出现。

---

## 10. Critical thoughts 和 extensions

### 限制和未来工作

1. **Task-agnostic intents 仍然是 prompt-engineered**：Appendix A.2 的 4 条规则是 hand-crafted。作者在 Discussion 里提到 program synthesis (Cherif et al. 2025, https://arxiv.org/abs/2504.17282) 是 promising direction——用 LLM 生成 code 而非 natural language 来 specify intents。

2. **Tabletop 太 toy**：3/5/7 blocks 还是 demonstration setting。Real robotics（DeXtNet、ALOHA 类）的 action space 和 perception complexity 高得多。但 paper 自己说是 "empirical illustration" 而非 "evaluation"。

3. **LLM 没指明是哪个**：实验里没说用 Gemini、GPT-4 还是其他。Affordance accuracy 强依赖 LLM 的常识推理能力，这点对 reproducibility 不友好。

4. **Theorem 1 的 δ 假设较强**：Definition 6 要求 `R(π_i, T^i) ≤ ζ` with prob ≥ 1-δ。真实 agent 在 multi-task 上很难 satisfy 这个——尤其当 task distribution 含 outlier task。Adversarial task distribution 会让 bound 失效。

5. **Theorem 2 的 `O(L)` factor**：虽然叫 "small factor"，但当 L（plan horizon）大时，2L+3 也不小。可能可以用 continuous ε_cor with EXP3-style online learning 把 factor 降到 `O(√L)` 或 `O(log L)`。

### 可能的 extensions（hallucination 不限）

1. **Vision-language models as affordance models**：用 Gemini-Pro-Vision / GPT-4V 直接从 image 生成 affordance，bypass 文字 state representation。这能解决 prop/perception mismatch 问题。

2. **Affordance hierarchies**：当前 task-agnostic intents 是 flat list。可以构造 hierarchical intent tree——top level 是 "manipulate object"，sub-level 是 "grasp" / "push" / "lift"，再下面是具体 motor commands。这能进一步压缩 branching factor。

3. **Active affordance learning**：当前 affordance 是 LLM pre-specified。可以让 agent 在线更新 affordance belief——observed (s,o) 不满足 intent 时，Bayesian update `P(affordance | s,o)`。这和 active learning、Bayesian optimization 都相通。

4. **Multi-agent affordances**：当多个 robot 协作时，affordance 变成 `P(intent | s, o, other_agents)`。Joint action space 的 branching factor 是 n^k for k agents，affordance pruning 更关键。

5. **Counterfactual affordances**：用 causal inference 推断 "如果 state 不同，affordance 怎么变"。例如 "如果 block 上没东西，pick 是 affordance；如果上面有别的 block，pick 不是 affordance"。这种 counterfactual reasoning 可能是 LLM 的 sweet spot。

6. **Affordance + Diffusion models**：把 LLM affordance 和 video diffusion world model（Sora、Genie 类）结合。Affordance 提供 high-level constraint，diffusion 提供 low-level dynamics，可能是 end-to-end world model 的下一步。

7. **Affordance discovery via LLM**：让 LLM 不仅 generate affordances for given intents，还能 **propose new intents**。这接近 curiosity-driven learning / empowerment maximization 的思路，但用 LLM 当 proposal engine。

8. **Formal verification of affordances**：把 Definition 4 的 distribution-robustness 用 statistical model checking / PAC-Bayes 推出 finite-sample bounds。当前 Theorem 1 是 asymptotic，工程上需要 finite-sample 版本。

---

## 11. 公式变量汇总速查表

### Theorem 1 变量

| 变量 | 含义 | 取值范围 |
|------|------|----------|
| `s, s'` | 当前 state, 下一 state | ∈ S |
| `o` | abstract action (option) | ∈ O |
| `P̂_par(s'|o,s)` | partial world model 的估计 | [0,1] |
| `P(s'|o,s)` | 真实 transition probability | [0,1] |
| `n` | task depth（任务复杂度） | ∈ ℕ+ |
| `ζ` | intent satisfiability threshold | (0,1) |
| `δ` | distribution-robust failure prob | (0, 1/2) |
| `φ` | characteristic error scale | O(1/√n) |
| `ρ` | exponential decay factor | <1 when δ<1/2 |
| `ε` | deviation parameter | >0 |

### Theorem 2 变量

| 变量 | 含义 | 取值范围 |
|------|------|----------|
| `N^ada` | adaptive model 总采样数 | ≥0 |
| `L` | sampled sequence 长度 | ∈ ℕ+ |
| `k` | partial model intent 数 | ≪ n |
| `n` | 全部 abstract action 数 | ∈ ℕ+ |
| `ε_cor` | correction probability | [0,1] |
| `m` | grid index | {0,...,2L+2} |
| `m_J` | sequence J 中来自 full model 的 intent 数 | ≤ L |
| `e` | Euler's number | ≈2.718 |
| `2L+3` | grid size | O(L) |

---

## 12. 总结

这篇 paper 的 contribution 三层：

1. **Conceptual**：把 affordance 从 single-task 推到 multi-task，区分 task-agnostic 和 task-specific intents。Definition 4 的 distribution-robust 是核心 formal 贡献。

2. **Theoretical**：Theorem 1 证明 competent planner 的 policy 隐含 partial world model，error bound 是 `O(1/√n)` characteristic + exponential tail。Theorem 2 证明 adaptive corrected partial model 享受 exponential speedup 同时 robust to missing intent，代价是 O(L) factor。

3. **Empirical**：Tabletop 3/5/7 blocks 上，partial model 让 LLM 作为 world model 第一次 work，full model 完全失败时 partial model 仍然 positive reward。

整体上，paper 的核心 message 是：**LLM 的知识是 partial but unevenly distributed，与其强行用 full model 不如 embrace partial model + affordance filter**。这个 insight 对 LLM + RL 的未来方向有指导意义——不是让 LLM 替代 learned world model，而是让 LLM 在 learned world model 上做 affordance-guided pruning。

### References

- [Khetarpal et al. 2020 - What can I do here?](https://arxiv.org/abs/2007.02633)
- [Khetarpal et al. 2021 - Temporally Abstract Partial Models](https://arxiv.org/abs/2110.06011)
- [Hao et al. 2023 - RAP: Reasoning with Language Model is Planning](https://arxiv.org/abs/2305.14992)
- [Bruce et al. 2024 - Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391)
- [Benechehab et al. 2024 - Zero-shot MBRL with LLMs](https://arxiv.org/abs/2410.11711)
- [Richens et al. 2025 - General Agents Need World Models](https://arxiv.org/abs/2506.01622)
- [Hafner et al. 2024 - DreamerV3](https://arxiv.org/abs/2301.04104)
- [Talvitie & Singh 2009 - Simple Local Models for Complex Dynamical Systems](https://papers.nips.cc/paper/2009/hash/a9418f1e6c1b0f5c0f4f5f9b6b9b9b9b-Abstract.html)
- [Gibson 1977 - Theory of Affordances](https://en.wikipedia.org/wiki/Affordance)
- [Cherif et al. 2025 - Generative Approach to Affordances for RL](https://arxiv.org/abs/2504.17282)
- [Linear Temporal Logic](https://en.wikipedia.org/wiki/Linear_temporal_logic)
